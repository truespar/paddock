//! Muse Glimmer vision tower (`clip` GGUF, projector `muse-glimmer`) - a
//! 50-layer ViT-G/14 Perception Encoder with 3:1 window attention, then the
//! adapter MLP and the LLM's vision projection.
//!
//! Reference graph: `tools/mtmd/models/muse-glimmer.cpp` + the
//! `PROJECTOR_TYPE_MUSE_GLIMMER` branches of `clip.cpp` (hparams, set_input)
//! and `mtmd-image.cpp` (the grid fit). Dataflow:
//!
//!   pixels (LANCZOS stretch onto an aspect-fitted 28-px grid, ×2-1)
//!   -> 14×14 conv (no bias, as a GEMM over im2row patches)
//!   -> + learned 32×32 pos table, bilinearly resized to the grid
//!   -> 50 × [ LN1 -> q/k/v(+bias) -> 2D NORM rope (x half by width, y half by
//!            height, θ=10000) -> attention: 32×32 WINDOWS except every 4th
//!            layer and the last, which are global -> out(+bias) -> +res
//!            -> LN2 -> up(+bias) -> erf-GELU -> down(+bias) -> +res ]
//!   -> post-LN -> 2×2 pixel shuffle, CHANNEL-outer -> [n/4, 4·1536]
//!   -> mm0 -> erf-GELU -> mm1 -> erf-GELU -> mm2 -> [n/4, 6656]
//!
//! Deltas from every other tower we serve, all of them silent when wrong and
//! none of them stated in the mmproj file:
//!   * rope pairs are NORM (2i, 2i+1), not NEOX - gemma4v's tower is NEOX.
//!   * the FFN is a plain MLP with **exact-erf** GELU, not GEGLU and not the
//!     tanh approximation.
//!   * window attention is BLOCK-DIAGONAL over 32×32 patch tiles, and the
//!     window size comes from the POS TABLE's side length (√1024), not from
//!     any window-size key.
//!   * the 3:1 sparse/global pattern lives in config.json's `layer_types`;
//!     the mmproj carries no trace of it.
//!   * the merge is channel-outer (`out[c*4 + s]`), the transpose of qwen3vl's.
//!
//! f32 activations over f16 weight planes, the class the other towers already
//! run: every GEMM operand is staged to f16 and accumulated in
//! f32, so the tower is resident at the mmproj file's own byte count.

use std::sync::Arc;

use cudarc::driver::CudaSlice;
use paddock_models::gguf::Value;
use paddock_models::mapped::MappedGguf;

use crate::gpu::{GpuError, GpuExecutor, HalfTensor};
use crate::gpu_model::pillow::{Filter, resize_rgb8};

/// Sparse/global period. config.json's `vision_config.layer_types` is
/// `[window, window, window, full] × 12` then `[window, full]`, i.e. every 4th
/// layer is global and so is the last. It is not in the mmproj - llama.cpp
/// hardcodes the same 4 in its `PROJECTOR_TYPE_MUSE_GLIMMER` load branch.
const SPARSE_FACTOR: usize = 4;
/// `vision_config.rope_parameters.rope_theta`. Also absent from the mmproj.
const ROPE_THETA: f32 = 10000.0;
/// `processor_config.image_processor.max_image_tokens`, in MERGED tokens (one
/// per 28×28 source block). llama.cpp's `set_limit_image_tokens(1, 4096)`
/// agrees. The floor really is 1: the grid fit never upscales.
const MAX_TOKENS: usize = 4096;

struct VBlock {
    ln1_w: CudaSlice<f32>,
    ln1_b: CudaSlice<f32>,
    wq: HalfTensor,
    wk: HalfTensor,
    wv: HalfTensor,
    bq: CudaSlice<f32>,
    bk: CudaSlice<f32>,
    bv: CudaSlice<f32>,
    wo: HalfTensor,
    bo: CudaSlice<f32>,
    ln2_w: CudaSlice<f32>,
    ln2_b: CudaSlice<f32>,
    up_w: HalfTensor,
    up_b: CudaSlice<f32>,
    down_w: HalfTensor,
    down_b: CudaSlice<f32>,
}

/// Encoded image: [n_tokens, llm_embd] rows ready for the splice. Same shape as
/// the gemma4v tower's output so `multimodal.rs` does not care which ran.
pub struct VisionOutput {
    pub embd: CudaSlice<f32>,
    pub n_tokens: usize,
}

pub struct VisionModel {
    exec: Arc<GpuExecutor>,
    n_layers: usize,
    embd: usize,
    n_heads: usize,
    head_dim: usize,
    patch: usize,
    merge: usize,
    eps: f32,
    /// learned pos table on host, [n_side², embd] row-major grid
    pos_embd: Vec<f32>,
    /// √(pos table rows) - 32, and also the window side (llama.cpp derives the
    /// window from this very number: `pgrid = sqrt(position_embeddings->ne[1])`)
    n_side: usize,
    image_mean: [f32; 3],
    image_std: [f32; 3],

    conv: HalfTensor, // [3·patch², embd]
    pre_ln_w: CudaSlice<f32>,
    pre_ln_b: CudaSlice<f32>,
    blocks: Vec<VBlock>,
    post_ln_w: CudaSlice<f32>,
    post_ln_b: CudaSlice<f32>,
    mm0: HalfTensor, // [4·embd, 4096]
    mm1: HalfTensor, // [4096, 4096]
    mm2: HalfTensor, // [4096, llm_embd]
}

/// `PADDOCK_MUSE_VIS_DUMP=1` prints a per-stage sum of the tower's activations,
/// to be diffed against `llama-mtmd-debug -p encode`, which prints exactly the
/// same quantity for every node of the reference clip graph. Eight of this
/// tower's constants are silent when wrong, so a stage-resolved oracle is what
/// separates "the rope pairing is wrong" from "the window mask is wrong" -
/// end-to-end tokens only say "something is".
///
/// It SYNCHRONIZES per stage. That is the point, and it is why it is off by
/// default.
fn dump_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_MUSE_VIS_DUMP").is_some())
}

/// Shared with the qwen35 tower: read a GGUF tensor (F32/F16/BF16) as host f32.
fn host_f32(map: &MappedGguf, name: &str) -> Result<(Vec<f32>, Vec<usize>), GpuError> {
    crate::gpu_model::qwen35::vision::host_f32(map, name)
        .map_err(|e| GpuError::Driver(e.to_string()))
}

/// The aspect-fitted patch grid, in MERGED tokens: transformers'
/// `get_aspect_ratio_preserving_size`, which llama.cpp mirrors as
/// `muse_glimmer_grid_size`. Shrinks to the token cap and never upscales; the
/// image is then STRETCHED onto it (no letterbox, unlike qwen).
fn grid_tokens(img_w: usize, img_h: usize, patch_hw: usize, max_tokens: usize) -> (usize, usize) {
    let mut i_nph = img_h as f64 / patch_hw as f64;
    let mut i_npw = img_w as f64 / patch_hw as f64;
    let ratio = if i_nph > 0.0 { i_npw / i_nph } else { 1.0 };
    if i_nph * i_npw > max_tokens as f64 {
        i_nph = (max_tokens as f64 / ratio).sqrt();
        i_npw = i_nph * ratio;
    }
    let hs = [i_nph.floor() as i64, i_nph.ceil() as i64];
    let ws = [i_npw.floor() as i64, i_npw.ceil() as i64];
    let target_ar = img_h as f64 / img_w as f64;
    let (mut best_h, mut best_w, mut best_d) = (-1i64, -1i64, 0.0f64);
    for &nph in &hs {
        for &npw in &ws {
            if nph < 1 || npw < 1 || nph * npw > max_tokens as i64 {
                continue;
            }
            let d = (nph as f64 / npw as f64 - target_ar).abs();
            if best_h < 0 || d < best_d || (d == best_d && nph * npw > best_h * best_w) {
                best_h = nph;
                best_w = npw;
                best_d = d;
            }
        }
    }
    if best_h < 0 {
        // nothing fit under the cap: round and clamp, llama.cpp's own fallback
        best_h = (i_nph.round() as i64).max(1);
        best_w = (i_npw.round() as i64).max(1);
    }
    (best_w as usize, best_h as usize)
}

impl VisionModel {
    pub fn load(exec: Arc<GpuExecutor>, map: &MappedGguf) -> Result<Self, GpuError> {
        let u = |k: &str| -> Result<usize, GpuError> {
            map.gguf()
                .metadata
                .get(k)
                .and_then(Value::as_u64)
                .map(|v| v as usize)
                .ok_or_else(|| GpuError::Driver(format!("mmproj missing {k}")))
        };
        let n_layers = u("clip.vision.block_count")?;
        let embd = u("clip.vision.embedding_length")?;
        let n_heads = u("clip.vision.attention.head_count")?;
        let patch = u("clip.vision.patch_size")?;
        let merge = u("clip.vision.spatial_merge_size").unwrap_or(2);
        let head_dim = embd / n_heads;
        let eps = map
            .gguf()
            .metadata
            .get("clip.vision.attention.layer_norm_epsilon")
            .and_then(Value::as_f32)
            .unwrap_or(1e-5);
        let arr3 = |k: &str| -> [f32; 3] {
            match map.gguf().metadata.get(k) {
                Some(Value::Array(a)) => {
                    let v: Vec<f32> = a.iter().filter_map(Value::as_f32).collect();
                    if v.len() == 3 {
                        [v[0], v[1], v[2]]
                    } else {
                        [0.5, 0.5, 0.5]
                    }
                }
                _ => [0.5, 0.5, 0.5],
            }
        };
        // hd/2 must be even for the NORM pair split to cover the half exactly,
        // and hd/4 is what the rope kernel's thread mapping counts in.
        if !head_dim.is_multiple_of(4) {
            return Err(GpuError::Driver(format!(
                "muse vision head_dim {head_dim} is not a multiple of 4"
            )));
        }

        let e = exec.clone();
        let dt = move |name: &str| -> Result<HalfTensor, GpuError> { e.upload_f16(map, name) };
        let e = exec.clone();
        let vf = move |name: &str| -> Result<CudaSlice<f32>, GpuError> {
            let (host, _) = host_f32(map, name)?;
            e.to_device(&host)
        };

        // The conv weight is stored [kx, ky, c, oc] with kx fastest, which read
        // as rows-per-output-channel is the im2row order `c*patch² + ky*patch +
        // kx` this module builds. So it only needs re-labelling, not a permute.
        let mut conv = dt("v.patch_embd.weight")?;
        let want = patch * patch * 3;
        if conv.bytes() / 2 != want * embd {
            return Err(GpuError::Driver(format!(
                "v.patch_embd.weight is {} elements, expected {}",
                conv.bytes() / 2,
                want * embd
            )));
        }
        conv.dims = vec![want, embd];

        let (pos_embd, pos_dims) = host_f32(map, "v.position_embd.weight")?;
        let n_side = (pos_dims[1] as f64).sqrt() as usize;
        if n_side * n_side != pos_dims[1] {
            return Err(GpuError::Driver(format!(
                "muse pos table is {} rows, not a square grid",
                pos_dims[1]
            )));
        }

        let mut blocks = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let t = |s: &str| format!("v.blk.{i}.{s}");
            blocks.push(VBlock {
                ln1_w: vf(&t("ln1.weight"))?,
                ln1_b: vf(&t("ln1.bias"))?,
                wq: dt(&t("attn_q.weight"))?,
                wk: dt(&t("attn_k.weight"))?,
                wv: dt(&t("attn_v.weight"))?,
                bq: vf(&t("attn_q.bias"))?,
                bk: vf(&t("attn_k.bias"))?,
                bv: vf(&t("attn_v.bias"))?,
                wo: dt(&t("attn_out.weight"))?,
                bo: vf(&t("attn_out.bias"))?,
                ln2_w: vf(&t("ln2.weight"))?,
                ln2_b: vf(&t("ln2.bias"))?,
                up_w: dt(&t("ffn_up.weight"))?,
                up_b: vf(&t("ffn_up.bias"))?,
                down_w: dt(&t("ffn_down.weight"))?,
                down_b: vf(&t("ffn_down.bias"))?,
            });
        }

        let me = Self {
            exec,
            n_layers,
            embd,
            n_heads,
            head_dim,
            patch,
            merge,
            eps,
            pos_embd,
            n_side,
            image_mean: arr3("clip.vision.image_mean"),
            image_std: arr3("clip.vision.image_std"),
            conv,
            pre_ln_w: vf("v.pre_ln.weight")?,
            pre_ln_b: vf("v.pre_ln.bias")?,
            blocks,
            post_ln_w: vf("v.post_ln.weight")?,
            post_ln_b: vf("v.post_ln.bias")?,
            mm0: dt("mm.0.weight")?,
            mm1: dt("mm.1.weight")?,
            mm2: dt("mm.2.weight")?,
        };
        if me.mm0.dims[0] != embd * me.merge * me.merge {
            return Err(GpuError::Driver(format!(
                "adapter takes {} inputs but the shuffle produces {}",
                me.mm0.dims[0],
                embd * me.merge * me.merge
            )));
        }
        tracing::info!(
            weight_mib = me.weight_bytes() / (1 << 20),
            layers = me.n_layers,
            window = me.n_side,
            "muse-glimmer mmproj resident at f16 (f32 accumulate)"
        );
        Ok(me)
    }

    /// Device bytes the f16 weight planes hold - what the estimator prices.
    pub fn weight_bytes(&self) -> usize {
        let blk: usize = self
            .blocks
            .iter()
            .map(|b| {
                b.wq.bytes()
                    + b.wk.bytes()
                    + b.wv.bytes()
                    + b.wo.bytes()
                    + b.up_w.bytes()
                    + b.down_w.bytes()
            })
            .sum();
        blk + self.conv.bytes() + self.mm0.bytes() + self.mm1.bytes() + self.mm2.bytes()
    }

    /// llm-side embedding width (the projector's output).
    pub fn llm_embd(&self) -> usize {
        self.mm2.dims[1]
    }

    /// One vision token per (patch·merge)² source block, capped at the
    /// processor's own `max_image_tokens`. There is no minimum worth stating:
    /// the grid fit shrinks and never grows, so a small image stays small.
    pub fn budget(&self) -> crate::generator::VisionBudget {
        let per = (self.patch * self.merge) as u64;
        crate::generator::VisionBudget {
            max_pixels: MAX_TOKENS as u64 * per * per,
            min_pixels: per * per,
            max_edge: None,
            pixels_per_token: per * per,
            max_tokens: MAX_TOKENS as u32,
            min_tokens: 1,
        }
    }

    /// The patch grid this image will be resized onto (pre-merge).
    fn grid_for(&self, w: usize, h: usize) -> (usize, usize) {
        let (tw, th) = grid_tokens(w, h, self.patch * self.merge, MAX_TOKENS);
        (tw * self.merge, th * self.merge)
    }

    /// Soft tokens a `w`x`h` picture encodes to - what `encode` will report,
    /// from the grid alone.
    pub fn tokens_for(&self, w: usize, h: usize) -> usize {
        let (gw, gh) = self.grid_for(w, h);
        (gw / self.merge) * (gh / self.merge)
    }

    /// Window layout in the permuted token order: the run length of each
    /// 32×32 patch tile, windows walked row-major. A block-diagonal mask over
    /// contiguous runs is exactly a batch of independent attentions, which is
    /// why the mask never materializes.
    fn window_runs(&self, gw: usize, gh: usize) -> Vec<usize> {
        let win = self.n_side;
        let mut runs = Vec::new();
        for wy in 0..gh.div_ceil(win) {
            let rows = win.min(gh - wy * win);
            for wx in 0..gw.div_ceil(win) {
                let cols = win.min(gw - wx * win);
                runs.push(rows * cols);
            }
        }
        runs
    }

    /// Original grid index of each row, in window order (llama.cpp's
    /// `sp_perm`).
    fn window_perm(&self, gw: usize, gh: usize) -> Vec<usize> {
        let win = self.n_side;
        let mut perm = Vec::with_capacity(gw * gh);
        for wy in 0..gh.div_ceil(win) {
            for wx in 0..gw.div_ceil(win) {
                for hh in 0..win {
                    for ww in 0..win {
                        let (gy, gx) = (wy * win + hh, wx * win + ww);
                        if gy < gh && gx < gw {
                            perm.push(gy * gw + gx);
                        }
                    }
                }
            }
        }
        perm
    }

    /// Bilinearly resize the learned pos table to (gh, gw). Half-pixel centres
    /// - `ggml_interpolate`'s BILINEAR semantics, which is what the reference
    ///   graph asks for (`GGML_SCALE_MODE_BILINEAR`, deliberately without the
    ///   antialias flag the default carries).
    fn pos_embd_for(&self, gw: usize, gh: usize) -> Vec<f32> {
        let (e, s) = (self.embd, self.n_side);
        if gw == s && gh == s {
            return self.pos_embd.clone();
        }
        let sample = |gy: usize, gx: usize| -> &[f32] {
            let idx = gy * s + gx;
            &self.pos_embd[idx * e..(idx + 1) * e]
        };
        let mut grid = vec![0f32; gh * gw * e];
        for y in 0..gh {
            let sy = ((y as f32 + 0.5) * s as f32 / gh as f32 - 0.5).clamp(0.0, (s - 1) as f32);
            let y0 = sy.floor() as usize;
            let y1 = (y0 + 1).min(s - 1);
            let fy = sy - y0 as f32;
            for x in 0..gw {
                let sx = ((x as f32 + 0.5) * s as f32 / gw as f32 - 0.5).clamp(0.0, (s - 1) as f32);
                let x0 = sx.floor() as usize;
                let x1 = (x0 + 1).min(s - 1);
                let fx = sx - x0 as f32;
                let (a, b, c, d) = (
                    sample(y0, x0),
                    sample(y0, x1),
                    sample(y1, x0),
                    sample(y1, x1),
                );
                let out = &mut grid[(y * gw + x) * e..(y * gw + x + 1) * e];
                for j in 0..e {
                    let top = a[j] + (b[j] - a[j]) * fx;
                    let bot = c[j] + (d[j] - c[j]) * fx;
                    out[j] = top + (bot - top) * fy;
                }
            }
        }
        grid
    }

    /// LANCZOS stretch onto the aspect-fitted grid, mean/std normalization, and
    /// im2row patches emitted directly in WINDOW order (the reference applies
    /// the permutation as a `get_rows` right after the pos-embd add; both are
    /// row-wise, so folding it into the emit order is the same graph).
    /// Returns (patches [n, 3·patch²], grid_w, grid_h).
    pub fn preprocess_rgb(&self, rgb: &[u8], w: usize, h: usize) -> (Vec<f32>, usize, usize) {
        let (gw, gh) = self.grid_for(w, h);
        let (tw, th) = (gw * self.patch, gh * self.patch);
        let resized = resize_rgb8(rgb, w, h, tw, th, Filter::Lanczos3);
        let mut img = vec![0f32; tw * th * 3];
        for (i, px) in resized.as_chunks::<3>().0.iter().enumerate() {
            for c in 0..3 {
                img[i * 3 + c] = (px[c] as f32 / 255.0 - self.image_mean[c]) / self.image_std[c];
            }
        }
        (self.patch_rows_raw(&img, tw, th), gw, gh)
    }

    /// im2row into WINDOW order from an interleaved-RGB f32 image that is
    /// already at the grid's pixel size and already scaled - no resize, no
    /// normalization. `preprocess_rgb` funnels into this, and so does the stage
    /// oracle, which feeds `llama-mtmd-debug`'s raw 0..1 values straight in
    /// (that tool skips preprocessing too, which is what makes the two
    /// comparable).
    pub fn patch_rows_raw(&self, img: &[f32], w: usize, h: usize) -> Vec<f32> {
        let p = self.patch;
        assert_eq!(img.len(), 3 * w * h);
        assert!(
            w.is_multiple_of(p) && h.is_multiple_of(p),
            "{w}x{h} is not a whole number of {p}px patches"
        );
        let (gw, gh) = (w / p, h / p);
        let pp = p * p;
        let perm = self.window_perm(gw, gh);
        let mut out = vec![0f32; perm.len() * 3 * pp];
        for (row, &orig) in perm.iter().enumerate() {
            let (px, py) = (orig % gw, orig / gw);
            let dst = &mut out[row * 3 * pp..(row + 1) * 3 * pp];
            for c in 0..3 {
                for ky in 0..p {
                    let sy = py * p + ky;
                    for kx in 0..p {
                        let sx = px * p + kx;
                        dst[c * pp + ky * p + kx] = img[(sy * w + sx) * 3 + c];
                    }
                }
            }
        }
        out
    }

    /// Patch edge in pixels, and the post-tower spatial merge factor.
    pub fn patch_size(&self) -> usize {
        self.patch
    }

    pub fn merge_size(&self) -> usize {
        self.merge
    }

    /// Encode window-ordered patches into LLM-space embeddings.
    pub fn encode(&self, patches: &[f32], gw: usize, gh: usize) -> Result<VisionOutput, GpuError> {
        let exec = &self.exec;
        let (e, heads, hd) = (self.embd, self.n_heads, self.head_dim);
        let n = gw * gh;
        let m = self.merge;
        let n_out = (gw / m) * (gh / m);
        assert_eq!(patches.len(), n * 3 * self.patch * self.patch);

        let perm = self.window_perm(gw, gh);
        let runs = self.window_runs(gw, gh);
        // rope positions are 1-INDEXED grid coordinates of each row's ORIGINAL
        // cell (clip.cpp's rpos_w/rpos_h), so they follow the permutation.
        let mut pos_x = vec![0u32; n];
        let mut pos_y = vec![0u32; n];
        let mut inv = vec![0u32; n];
        for (row, &orig) in perm.iter().enumerate() {
            pos_x[row] = (orig % gw) as u32 + 1;
            pos_y[row] = (orig / gw) as u32 + 1;
            inv[orig] = row as u32;
        }
        // pixel-shuffle gather, composed with the inverse permutation so the
        // reference's un-permute is free: `ds_perm` names ORIGINAL cells and our
        // rows sit in window order.
        let mut ds = Vec::with_capacity(n);
        for oy in 0..gh / m {
            for ox in 0..gw / m {
                for ry in 0..m {
                    for rx in 0..m {
                        ds.push(inv[(oy * m + ry) * gw + (ox * m + rx)]);
                    }
                }
            }
        }
        let pe = {
            let grid = self.pos_embd_for(gw, gh);
            let mut v = vec![0f32; n * e];
            for (row, &orig) in perm.iter().enumerate() {
                v[row * e..(row + 1) * e].copy_from_slice(&grid[orig * e..(orig + 1) * e]);
            }
            v
        };

        let d_patches = exec.to_device(patches)?;
        let d_pe = exec.to_device(&pe)?;
        let d_px = exec.to_device_u32(&pos_x)?;
        let d_py = exec.to_device_u32(&pos_y)?;
        let d_ds = exec.to_device_u32(&ds)?;

        let ffn = self.blocks[0].up_w.dims[1];
        let stage = (n * ffn.max(e).max(self.conv.dims[0]))
            .max(n_out * self.mm0.dims[0].max(self.mm0.dims[1]));
        let mut s16 = exec.alloc_f16(stage)?;

        // conv stem (no patch bias in this file) + learned positions
        let mut x = exec.alloc(n * e)?;
        exec.convert_f32_f16(&d_patches, &mut s16, n * self.conv.dims[0])?;
        exec.matvec_batch_f16(&self.conv, &s16, &mut x, n)?;
        exec.add(&mut x, &d_pe, n * e)?;
        self.stage("after_sp_perm", &x, n * e);

        let mut nrm = exec.alloc(n * e)?;
        let mut q = exec.alloc(n * e)?;
        let mut k = exec.alloc(n * e)?;
        let mut v = exec.alloc(n * e)?;
        let mut a = exec.alloc(n * e)?;
        let mut up = exec.alloc(n * ffn)?;
        let scale = 1.0 / (hd as f32).sqrt();
        let theta_scale = ROPE_THETA.powf(-2.0 / (hd / 2) as f32);

        // pre-LN sits outside the loop (build_vit applies it to the stream once
        // before layer 0), so it becomes the residual stream rather than a
        // per-layer temp - normalize into the scratch and swap, since the
        // kernel is not in-place safe.
        exec.layernorm(&x, &self.pre_ln_w, &self.pre_ln_b, &mut nrm, n, e, self.eps)?;
        std::mem::swap(&mut x, &mut nrm);
        self.stage("pre_ln", &x, n * e);

        for (il, blk) in self.blocks.iter().enumerate() {
            exec.layernorm(&x, &blk.ln1_w, &blk.ln1_b, &mut nrm, n, e, self.eps)?;
            exec.convert_f32_f16(&nrm, &mut s16, n * e)?;
            exec.matvec_batch_f16(&blk.wq, &s16, &mut q, n)?;
            exec.matvec_batch_f16(&blk.wk, &s16, &mut k, n)?;
            exec.matvec_batch_f16(&blk.wv, &s16, &mut v, n)?;
            exec.bias_add(&mut q, &blk.bq, n, e)?;
            exec.bias_add(&mut k, &blk.bk, n, e)?;
            exec.bias_add(&mut v, &blk.bv, n, e)?;
            // NORM pairing, width half then height half - see the module note
            exec.rope2d(&mut q, &d_px, &d_py, n, heads, hd, theta_scale, false)?;
            exec.rope2d(&mut k, &d_px, &d_py, n, heads, hd, theta_scale, false)?;

            if Self::is_global(il, self.n_layers) {
                exec.vision_attn_at(&q, &k, &v, &mut a, 0, n, heads, hd, scale)?;
            } else {
                // consecutive windows of equal size share one launch; the
                // ragged right/bottom edges become their own runs
                let mut off = 0usize;
                let mut i = 0usize;
                while i < runs.len() {
                    let len = runs[i];
                    let mut cnt = 1usize;
                    while i + cnt < runs.len() && runs[i + cnt] == len {
                        cnt += 1;
                    }
                    exec.vision_attn_x_at(&q, &k, &v, &mut a, off, len, heads, hd, cnt, scale)?;
                    off += len * cnt;
                    i += cnt;
                }
            }

            exec.convert_f32_f16(&a, &mut s16, n * e)?;
            exec.matvec_batch_f16(&blk.wo, &s16, &mut nrm, n)?;
            exec.bias_add(&mut nrm, &blk.bo, n, e)?;
            exec.add(&mut x, &nrm, n * e)?;

            exec.layernorm(&x, &blk.ln2_w, &blk.ln2_b, &mut nrm, n, e, self.eps)?;
            exec.convert_f32_f16(&nrm, &mut s16, n * e)?;
            exec.matvec_batch_f16(&blk.up_w, &s16, &mut up, n)?;
            exec.bias_add(&mut up, &blk.up_b, n, ffn)?;
            // Exact erf GELU (FFN_GELU_ERF in the reference graph), not the
            // tanh approximation `gelu` - silently different features.
            exec.gelu_erf(&mut up, n * ffn)?;
            exec.convert_f32_f16(&up, &mut s16, n * ffn)?;
            exec.matvec_batch_f16(&blk.down_w, &s16, &mut nrm, n)?;
            exec.bias_add(&mut nrm, &blk.down_b, n, e)?;
            exec.add(&mut x, &nrm, n * e)?;
            if dump_on() {
                self.stage(&format!("layer_out-{il}"), &x, n * e);
            }
        }

        exec.layernorm(
            &x,
            &self.post_ln_w,
            &self.post_ln_b,
            &mut nrm,
            n,
            e,
            self.eps,
        )?;

        // 2×2 pixel shuffle, channel-outer, un-permuting on the way
        let wide = e * m * m;
        let mut merged = exec.alloc(n_out * wide)?;
        exec.pixel_shuffle_rows(&nrm, &d_ds, &mut merged, n_out, m * m, e)?;
        self.stage("encoder_out", &merged, n_out * wide);

        // adapter: fc1 -> erf-GELU -> fc2 -> erf-GELU -> vision_projection
        let mid = self.mm0.dims[1];
        let mut h0 = exec.alloc(n_out * mid)?;
        exec.convert_f32_f16(&merged, &mut s16, n_out * wide)?;
        exec.matvec_batch_f16(&self.mm0, &s16, &mut h0, n_out)?;
        exec.gelu_erf(&mut h0, n_out * mid)?;
        let mut h1 = exec.alloc(n_out * self.mm1.dims[1])?;
        exec.convert_f32_f16(&h0, &mut s16, n_out * mid)?;
        exec.matvec_batch_f16(&self.mm1, &s16, &mut h1, n_out)?;
        exec.gelu_erf(&mut h1, n_out * self.mm1.dims[1])?;
        let out_dim = self.llm_embd();
        let mut out = exec.alloc(n_out * out_dim)?;
        exec.convert_f32_f16(&h1, &mut s16, n_out * self.mm1.dims[1])?;
        exec.matvec_batch_f16(&self.mm2, &s16, &mut out, n_out)?;
        self.stage("projected", &out, n_out * out_dim);

        Ok(VisionOutput {
            embd: out,
            n_tokens: n_out,
        })
    }

    /// See [`dump_on`]. f64 accumulation so the sum of ~6M f32 activations is
    /// comparable against ggml's own (which also accumulates in a wider type);
    /// an f32 running sum would lose the low bits and invent a mismatch.
    fn stage(&self, tag: &str, buf: &CudaSlice<f32>, n: usize) {
        if !dump_on() {
            return;
        }
        match self.exec.to_host_len(buf, n) {
            Ok(h) => {
                let s: f64 = h.iter().map(|&v| v as f64).sum();
                eprintln!("muse-vis {tag:>18}: n={n:<9} sum={s:.4}");
            }
            Err(e) => eprintln!("muse-vis {tag:>18}: readback failed: {e}"),
        }
    }

    /// Every 4th layer and the last one attend globally; the rest see only
    /// their own 32×32 window. Straight off `vision_config.layer_types`, and
    /// identical to the reference graph's
    /// `(il == n_layer - 1) || ((il + 1) % sf == 0)`.
    fn is_global(il: usize, n_layers: usize) -> bool {
        il + 1 == n_layers || (il + 1).is_multiple_of(SPARSE_FACTOR)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The published pattern is `[w, w, w, full] × 12` then `[w, full]` - 50
    /// layers, 13 of them global. Pinning the list rather than the rule so a
    /// re-derivation cannot quietly shift the phase.
    #[test]
    fn global_layers_match_config_layer_types() {
        let globals: Vec<usize> = (0..50).filter(|&i| VisionModel::is_global(i, 50)).collect();
        assert_eq!(
            globals,
            vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 49],
            "3:1 window/full pattern with a global last layer"
        );
    }

    /// The grid fit shrinks to the token cap, keeps the aspect, and never
    /// upscales - the property that separates it from qwen's smart_resize.
    #[test]
    fn grid_fit_caps_and_never_grows() {
        // square, well under the cap: 1024/28 = 36.57 -> 37x37 = 1369 tokens
        assert_eq!(grid_tokens(1024, 1024, 28, 4096), (37, 37));
        // tiny image stays tiny (1 token minimum, no upscale)
        assert_eq!(grid_tokens(20, 20, 28, 4096), (1, 1));
        // way over the cap: shrinks under it while holding the 2:1 aspect
        let (w, h) = grid_tokens(8000, 4000, 28, 4096);
        assert!(w * h <= 4096, "{w}x{h} = {} over the cap", w * h);
        assert!(
            (w as f64 / h as f64 - 2.0).abs() < 0.05,
            "aspect drifted: {w}x{h}"
        );
    }

    /// A grid that fits inside one window is one run - window attention
    /// degenerates to full attention, which is what a 32×32-or-smaller grid
    /// gets in the reference too (its mask is all-zeros).
    #[test]
    fn window_runs_cover_every_row_exactly() {
        for (gw, gh) in [(32usize, 32usize), (74, 74), (128, 8), (33, 65)] {
            let win = 32usize;
            let runs: Vec<usize> = (0..gh.div_ceil(win))
                .flat_map(|wy| {
                    let rows = win.min(gh - wy * win);
                    (0..gw.div_ceil(win)).map(move |wx| rows * win.min(gw - wx * win))
                })
                .collect();
            assert_eq!(
                runs.iter().sum::<usize>(),
                gw * gh,
                "{gw}x{gh} runs lost rows"
            );
        }
    }
}
