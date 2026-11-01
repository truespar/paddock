//! Nemotron 3.5 Super VL's picture path: the processor (one dynamic-resolution
//! tile a picture, torch's float antialiased bicubic, CLIP normalization), the
//! RADIO tower (32 pre-LN ViT blocks, 1280 wide, 16 heads of 80, 3 class
//! tokens + 7 registers, the cropped bilinear position field), and the
//! projector (LayerNorm, the 2 x 2 pixel shuffle, RMSNorm, 5120 -> 20480,
//! relu^2, -> 4096). Out come the language model's rows for the picture's
//! `<image>` placeholders.
//!
//! **The tile** is the processor's `_compute_target_patches`: the picture's
//! patch grid rounded up (Python's `round(x + 0.5)`, half to even), scaled
//! down to the per-picture budget (13312 patches) or up to the floor (1024),
//! each side made even for the shuffle. Tokens = patches / 4.
//!
//! **The resize** is the processor's `interpolate(bicubic, align_corners=False,
//! antialias=True)` on the u8 picture widened to f32 - its float path, NOT
//! PIL's (the processor's authors found PIL's kernel moves the answers). The
//! per-axis plan (taps, weights) is torch's `_compute_indices_min_size_weights_aa`
//! in the reference's own f32 / f64 promotions; width first, then height, a
//! pass skipped when its side does not change (an identity plan here - one
//! tap of weight 1, exact).
//!
//! **The tower** runs Parse's RADIO block kernels unchanged (same family, a
//! different width): LN into f16, the f16 rings, the q/v-bias split (the k
//! bias is dropped - one vector added to every key moves each query's scores
//! by a constant softmax ignores), row-ranged bidirectional attention, the
//! residual + LN seams with LayerScale at its 1.0 init (the checkpoint ships
//! none), fc1 + exact GELU in the landing. The 10 prefix tokens are parked
//! after the patch rows (attention without positions is permutation
//! equivariant) and dropped at the end. No final norm in the tower; the
//! projector's LayerNorm comes first.
//!
//! **The projector** keeps the checkpoint's bf16 planes (the bf16 GEMM,
//! f32 in and out - relu^2 of the 20480-wide hidden would crowd f16).

use std::path::Path;
use std::sync::Arc;

use cudarc::driver::CudaSlice;
use half::f16;
use paddock_models::safetensors::{ShardedSafetensors, StDtype};

use crate::gpu::{GpuExecutor, HalfTensor, QuantTensor, ResampleAxis};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::nemotron_parse::vision::{Norm, Up};
use crate::gpu_model::st_load::{bf16_bytes, f32_tensor};
use paddock_models::ggml_type::GgmlType;
pub use paddock_models::nemotron_vl::SuperVlPictures as VisionConfig;

/// timm's ViT block norm and the projector's LayerNorm (eps 1e-6).
const TOWER_LN_EPS: f32 = 1e-6;
/// The projector's RMSNorm (`mlp1.0`).
const PROJ_RMS_EPS: f32 = 1e-5;

/// Python's `round` (half to even) - the processor's ceil-ish patch count.
fn round_half_even(x: f64) -> i64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 {
        let t = x.trunc() as i64;
        if t % 2 == 0 { t } else { t + x.signum() as i64 }
    } else {
        r as i64
    }
}

/// The processor's `_compute_target_patches`: the tile `(w_patches,
/// h_patches)` for a `w x h` picture with `budget` patches available.
pub fn tile_patches(w: usize, h: usize, cfg: &VisionConfig, budget: usize) -> (usize, usize) {
    let p = cfg.patch as f64;
    let ch = round_half_even(h as f64 / p + 0.5).max(1);
    let cw = round_half_even(w as f64 / p + 0.5).max(1);
    let patches = (ch * cw) as f64;
    let factor = (budget as f64 / patches).sqrt().min(1.0);
    let mut th = (factor * ch as f64).floor() as i64;
    let mut tw = (factor * cw as f64).floor() as i64;
    let min = cfg.min_patches as i64;
    if (budget as i64) > min && th * tw < min && th * tw > 0 {
        let up = (min as f64 / (th * tw) as f64).sqrt();
        th = (up * th as f64).ceil() as i64;
        tw = (up * tw as f64).ceil() as i64;
    }
    let d = cfg.shuffle as i64;
    let budget = budget as i64;
    let rem_h = th % d;
    if rem_h != 0 {
        if (th + d - rem_h) * tw <= budget {
            th += d - rem_h;
        } else {
            th = d.max(th - rem_h);
        }
    }
    let rem_w = tw % d;
    if rem_w != 0 {
        if th * (tw + d - rem_w) <= budget {
            tw += d - rem_w;
        } else {
            tw = d.max(tw - rem_w);
        }
    }
    (tw.max(d) as usize, th.max(d) as usize)
}

/// torch's antialiased bicubic filter (Keys, a = -0.5) in f32.
fn aa_cubic(x: f32) -> f32 {
    let a = -0.5f32;
    let x = x.abs();
    if x < 1.0 {
        ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a
    } else {
        0.0
    }
}

/// One axis of torch's float antialiased bicubic (`in_len` -> `out`): per
/// output its first tap, tap count and `taps` normalized f32 weights. The
/// reference's promotions are kept: `scale` and the weights in f32, the
/// center `scale * (i + 0.5)` and the tap bounds through f64. An unchanged
/// side is the identity (the reference skips that pass).
pub fn aa_plan(in_len: usize, out: usize) -> (Vec<u32>, Vec<u32>, Vec<f32>, usize) {
    if in_len == out {
        return ((0..out as u32).collect(), vec![1; out], vec![1.0; out], 1);
    }
    let scale = in_len as f32 / out as f32;
    let support = if scale >= 1.0 {
        (2.0f64 * scale as f64) as f32
    } else {
        2.0f32
    };
    let taps = (support.ceil() as usize) * 2 + 1;
    let invscale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    let mut xmin = Vec::with_capacity(out);
    let mut xsize = Vec::with_capacity(out);
    let mut w = vec![0f32; out * taps];
    for i in 0..out {
        let center = (scale as f64 * (i as f64 + 0.5)) as f32;
        let lo = (((center - support) as f64) + 0.5) as i64;
        let lo = lo.max(0);
        let hi = ((((center + support) as f64) + 0.5) as i64).min(in_len as i64);
        let n = (hi - lo).clamp(0, taps as i64);
        let row = &mut w[i * taps..(i + 1) * taps];
        let mut total = 0f32;
        for j in 0..n {
            let arg = ((((j + lo) as f32 - center) as f64 + 0.5) * invscale as f64) as f32;
            let v = aa_cubic(arg);
            row[j as usize] = v;
            total += v;
        }
        if total != 0.0 {
            for v in row.iter_mut().take(n as usize) {
                *v /= total;
            }
        }
        xmin.push(lo as u32);
        xsize.push(n as u32);
    }
    (xmin, xsize, w, taps)
}

/// The pixel shuffle's gather (ps_version v2): output row (i, j) of the
/// halved grid takes source rows (2i + dy, 2j + dx), dy then dx - the 2 x 2
/// neighbours concatenated spatial-outer, which a row gather of these
/// indices into 4 consecutive rows lays out exactly.
pub fn shuffle_index(hp: usize, wp: usize) -> Vec<u32> {
    let mut idx = Vec::with_capacity(hp * wp);
    for i in 0..hp / 2 {
        for j in 0..wp / 2 {
            for dy in 0..2 {
                for dx in 0..2 {
                    idx.push(((2 * i + dy) * wp + 2 * j + dx) as u32);
                }
            }
        }
    }
    idx
}

struct Block {
    ln1: Norm,
    wqkv: HalfTensor,
    bq: CudaSlice<f32>,
    bv: CudaSlice<f32>,
    wo: HalfTensor,
    bo: CudaSlice<f32>,
    ln2: Norm,
    fc1: HalfTensor,
    fc1_b: CudaSlice<f32>,
    fc2: HalfTensor,
    fc2_b: CudaSlice<f32>,
}

/// Tower rows a pass is sized for: one picture at the patch budget plus its
/// 10 prefix tokens.
fn ws_rows(cfg: &VisionConfig) -> usize {
    cfg.max_patches + 16
}

struct Workspace {
    rows: usize,
    patches: CudaSlice<f16>,
    x: CudaSlice<f32>,
    n16: CudaSlice<f16>,
    qkv: CudaSlice<f16>,
    q: CudaSlice<f16>,
    k: CudaSlice<f16>,
    v: CudaSlice<f16>,
    a16: CudaSlice<f16>,
    proj: CudaSlice<f16>,
    ff: CudaSlice<f16>,
    e32: CudaSlice<f32>,
    /// the shuffled rows [tokens][4 * dim] and their RMSNorm
    shuf: CudaSlice<f32>,
    shuf_n: CudaSlice<f32>,
    /// the projector's hidden [tokens][20480]
    hid: CudaSlice<f32>,
}

/// A picture's language-model rows: `rows` x hidden f32.
pub struct PictureRows {
    pub rows: CudaSlice<f32>,
    pub n: usize,
}

pub struct SuperVision {
    exec: Arc<GpuExecutor>,
    pub cfg: VisionConfig,
    dim: usize,
    heads: usize,
    hd: usize,
    n_prefix: usize,
    grid_side: usize,
    hidden: usize,
    patch_w: HalfTensor,
    cls: CudaSlice<f32>,
    /// the learned grid `[side * side][dim]`, f32 (bf16 widened)
    pos_grid: CudaSlice<f32>,
    blocks: Vec<Block>,
    ones: CudaSlice<f32>,
    no_rope: CudaSlice<f32>,
    final_ln: Norm,
    rms_w: CudaSlice<f32>,
    fc1: QuantTensor,
    fc2: QuantTensor,
    ws: Option<Workspace>,
    pub weight_bytes: u64,
}

fn tower_tensor_shape(st: &ShardedSafetensors, name: &str) -> Option<Vec<usize>> {
    st.bytes(name).map(|(t, _)| t.shape.clone())
}

impl SuperVision {
    /// The tower + projector from the omni checkpoint. None when it carries
    /// no tower (a text-only export).
    pub fn load(
        exec: Arc<GpuExecutor>,
        st: &ShardedSafetensors,
        dir: &Path,
        hidden: usize,
    ) -> Result<Option<Self>, GpuModelError> {
        let pre = "vision_model.radio_model.model";
        if st
            .bytes(&format!("{pre}.patch_generator.embedder.weight"))
            .is_none()
        {
            return Ok(None);
        }
        if !exec.has_nemotron_vl() {
            return Err(GpuModelError::Unsupported(
                "this kernel pack predates Nemotron 3.5 Super VL's picture lane (slots 904-907) - \
                 rebuild or update the pack"
                    .into(),
            ));
        }
        if !exec.has_dense_pred_h() || !exec.has_f16_gemm_h_gelu() || !exec.has_vision_qkv_h() {
            return Err(GpuModelError::Unsupported(
                "Super VL's tower needs the half tower interface (slots 618-624) - rebuild or \
                 update the pack"
                    .into(),
            ));
        }
        if !exec.has_bf16_dense() {
            return Err(GpuModelError::Unsupported(
                "Super VL's projector needs the bf16 GEMM - rebuild or update the pack".into(),
            ));
        }
        let cfg = VisionConfig::read(dir)
            .map_err(|e| GpuModelError::Unsupported(format!("Super VL pictures: {e}")))?;
        let p = cfg.patch;
        let emb = tower_tensor_shape(st, &format!("{pre}.patch_generator.embedder.weight"))
            .unwrap_or_default();
        let (dim, kin) = match emb.as_slice() {
            [d, k] => (*d, *k),
            _ => {
                return Err(GpuModelError::Unsupported(format!(
                    "Super VL embedder {emb:?}"
                )));
            }
        };
        if kin != 3 * p * p {
            return Err(GpuModelError::Unsupported(format!(
                "Super VL embedder takes {kin}, not 3 x {p} x {p}"
            )));
        }
        let heads = 16usize;
        let hd = dim / heads;
        let cls_shape = tower_tensor_shape(st, &format!("{pre}.patch_generator.cls_token.token"))
            .unwrap_or_default();
        let n_prefix = cls_shape.first().copied().unwrap_or(0);
        let pos_shape =
            tower_tensor_shape(st, &format!("{pre}.patch_generator.pos_embed")).unwrap_or_default();
        let n_pos = pos_shape.get(1).copied().unwrap_or(0);
        let grid_side = (n_pos as f64).sqrt() as usize;
        if grid_side * grid_side != n_pos || pos_shape != [1, n_pos, dim] {
            return Err(GpuModelError::Unsupported(format!(
                "Super VL pos_embed {pos_shape:?} - want a square [1, s*s, {dim}] grid"
            )));
        }
        let mut up = Up {
            exec: &exec,
            bytes: 0,
        };
        let m = |s: &str| format!("{pre}.{s}");
        let patch_w = {
            let w = f32_tensor(st, &m("patch_generator.embedder.weight"), dim * kin)?;
            up.plane(&w, kin, dim, "Super VL patch embedder")?
        };
        let cls = up.dev(&f32_tensor(
            st,
            &m("patch_generator.cls_token.token"),
            n_prefix * dim,
        )?)?;
        let pos_grid = up.dev(&f32_tensor(
            st,
            &m("patch_generator.pos_embed"),
            n_pos * dim,
        )?)?;
        let norm = |up: &mut Up, prefix: &str, n: usize| -> Result<Norm, GpuModelError> {
            Ok(Norm {
                w: up.dev(&f32_tensor(st, &format!("{prefix}.weight"), n)?)?,
                b: up.dev(&f32_tensor(st, &format!("{prefix}.bias"), n)?)?,
            })
        };
        let n_layer = (0..)
            .take_while(|i| st.bytes(&m(&format!("blocks.{i}.norm1.weight"))).is_some())
            .count();
        let ffn = tower_tensor_shape(st, &m("blocks.0.mlp.fc1.bias"))
            .and_then(|s| s.first().copied())
            .unwrap_or(0);
        if n_layer == 0 || ffn == 0 {
            return Err(GpuModelError::MissingMeta("Super VL tower blocks".into()));
        }
        let mut blocks = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let l = |s: &str| m(&format!("blocks.{i}.{s}"));
            let qkv_b = f32_tensor(st, &l("attn.qkv.bias"), 3 * dim)?;
            blocks.push(Block {
                ln1: norm(&mut up, &l("norm1"), dim)?,
                wqkv: up.plane(
                    &f32_tensor(st, &l("attn.qkv.weight"), 3 * dim * dim)?,
                    dim,
                    3 * dim,
                    &l("qkv"),
                )?,
                bq: up.dev(&qkv_b[..dim])?,
                bv: up.dev(&qkv_b[2 * dim..])?,
                wo: up.plane(
                    &f32_tensor(st, &l("attn.proj.weight"), dim * dim)?,
                    dim,
                    dim,
                    &l("proj"),
                )?,
                bo: up.dev(&f32_tensor(st, &l("attn.proj.bias"), dim)?)?,
                ln2: norm(&mut up, &l("norm2"), dim)?,
                fc1: up.plane(
                    &f32_tensor(st, &l("mlp.fc1.weight"), ffn * dim)?,
                    dim,
                    ffn,
                    &l("fc1"),
                )?,
                fc1_b: up.dev(&f32_tensor(st, &l("mlp.fc1.bias"), ffn)?)?,
                fc2: up.plane(
                    &f32_tensor(st, &l("mlp.fc2.weight"), dim * ffn)?,
                    ffn,
                    dim,
                    &l("fc2"),
                )?,
                fc2_b: up.dev(&f32_tensor(st, &l("mlp.fc2.bias"), dim)?)?,
            });
        }
        let final_ln = norm(&mut up, "vision_projector.vision_final_layernorm", dim)?;
        let sh = cfg.shuffle * cfg.shuffle * dim;
        let rms_w = up.dev(&f32_tensor(st, "mlp1.0.weight", sh)?)?;
        let proj_plane = |name: &str, inn: usize| -> Result<(QuantTensor, usize), GpuModelError> {
            let shape = tower_tensor_shape(st, name).unwrap_or_default();
            let [out, k] = shape.as_slice() else {
                return Err(GpuModelError::Unsupported(format!("{name}: {shape:?}")));
            };
            if *k != inn {
                return Err(GpuModelError::Unsupported(format!(
                    "{name}: [{out}, {k}], want input {inn}"
                )));
            }
            let raw = bf16_bytes(st, name, out * k)?;
            Ok((
                QuantTensor {
                    bytes: exec.to_device_u8(raw)?,
                    ty: GgmlType::Bf16,
                    dims: vec![*k, *out],
                },
                *out,
            ))
        };
        let (fc1, hid) = proj_plane("mlp1.1.weight", sh)?;
        let (fc2, out) = proj_plane("mlp1.3.weight", hid)?;
        if out != hidden {
            return Err(GpuModelError::Unsupported(format!(
                "Super VL projector lands {out}, the language model is {hidden} wide"
            )));
        }
        let ones = up.dev(&vec![1.0f32; dim])?;
        let no_rope = up.dev(&[0.0f32])?;
        let weight_bytes = up.bytes + ((sh * hid + hid * hidden) * 2) as u64;
        // summary_idxs is the tower's bookkeeping only: the projector reads
        // the patch features, never the summary tokens
        if let Some((t, _)) = st.bytes("vision_model.summary_idxs")
            && t.dtype != StDtype::I64
        {
            return Err(GpuModelError::Unsupported(
                "Super VL summary_idxs is not i64".into(),
            ));
        }
        tracing::info!(
            blocks = n_layer,
            dim,
            heads,
            hd,
            n_prefix,
            grid_side,
            weights_gib = weight_bytes as f64 / (1u64 << 30) as f64,
            "Super VL picture tower loaded"
        );
        Ok(Some(Self {
            exec,
            cfg,
            dim,
            heads,
            hd,
            n_prefix,
            grid_side,
            hidden,
            patch_w,
            cls,
            pos_grid,
            blocks,
            ones,
            no_rope,
            final_ln,
            rms_w,
            fc1,
            fc2,
            ws: None,
            weight_bytes,
        }))
    }

    /// Allocate the picture workspace now, so the KV plan made after load
    /// sees it as taken rather than the first picture finding it missing.
    pub fn reserve(&mut self) -> Result<(), GpuModelError> {
        self.ensure_ws()
    }

    fn ensure_ws(&mut self) -> Result<(), GpuModelError> {
        if self.ws.is_some() {
            return Ok(());
        }
        let e = &self.exec;
        let rows = ws_rows(&self.cfg);
        let d = self.dim;
        let ffn = self.blocks[0].fc1.dims[1];
        let toks = rows / (self.cfg.shuffle * self.cfg.shuffle);
        let sh = self.cfg.shuffle * self.cfg.shuffle * d;
        let hid = self.fc1.dims[1];
        let p = self.cfg.patch;
        self.ws = Some(Workspace {
            rows,
            patches: e.alloc_f16(rows * 3 * p * p)?,
            x: e.alloc(rows * d)?,
            n16: e.alloc_f16(rows * d)?,
            qkv: e.alloc_f16(rows * 3 * d)?,
            q: e.alloc_f16(rows * d)?,
            k: e.alloc_f16(rows * d)?,
            v: e.alloc_f16(rows * d)?,
            a16: e.alloc_f16(rows * d)?,
            proj: e.alloc_f16(rows * d)?,
            ff: e.alloc_f16(rows * ffn)?,
            e32: e.alloc(rows * d)?,
            shuf: e.alloc(toks * sh)?,
            shuf_n: e.alloc(toks * sh)?,
            hid: e.alloc(toks * hid)?,
        });
        Ok(())
    }

    /// Bytes the picture workspace holds (sized for one picture at the
    /// patch budget).
    pub fn workspace_bytes(&self) -> u64 {
        let rows = ws_rows(&self.cfg);
        let d = self.dim;
        let ffn = self.blocks.first().map_or(0, |b| b.fc1.dims[1]);
        let toks = rows / (self.cfg.shuffle * self.cfg.shuffle);
        let sh = self.cfg.shuffle * self.cfg.shuffle * d;
        let hid = self.fc1.dims[1];
        let p = self.cfg.patch;
        let halves = rows * 3 * p * p + rows * d * 7 + rows * 3 * d + rows * ffn;
        let floats = rows * d * 2 + toks * sh * 2 + toks * hid;
        (halves * 2 + floats * 4) as u64
    }

    /// One picture (`rgb` row-major interleaved, `w x h`) of `n_pictures` in
    /// its request -> its language-model rows.
    pub fn encode(
        &mut self,
        rgb: &[u8],
        w: usize,
        h: usize,
        n_pictures: usize,
    ) -> Result<PictureRows, GpuModelError> {
        if rgb.len() != w * h * 3 || w == 0 || h == 0 {
            return Err(GpuModelError::Unsupported(format!(
                "picture {w} x {h} carries {} bytes",
                rgb.len()
            )));
        }
        self.ensure_ws()?;
        let exec = self.exec.clone();
        let cfg = self.cfg.clone();
        let p = cfg.patch;
        let (wp, hp) = tile_patches(w, h, &cfg, cfg.picture_budget(n_pictures));
        let (tw, th) = (wp * p, hp * p);
        let n_patch = wp * hp;
        let rows = n_patch + self.n_prefix;
        let d = self.dim;
        let ws = self.ws.as_mut().expect("workspace");
        if rows > ws.rows {
            return Err(GpuModelError::BatchTooLarge {
                got: rows,
                max: ws.rows,
            });
        }

        // ---- the processor: resize (width, then height), normalize, patchify
        let axis = |inl: usize, out: usize| -> Result<ResampleAxis, GpuModelError> {
            let (xmin, xsize, wts, taps) = aa_plan(inl, out);
            Ok(ResampleAxis {
                xmin: exec.to_device_u32(&xmin)?,
                xsize: exec.to_device_u32(&xsize)?,
                w: exec.to_device(&wts)?,
                taps,
                out,
            })
        };
        let src = exec.to_device_u8(rgb)?;
        let mut tmp = exec.alloc(h * tw * 3)?;
        exec.nvl_resample(Some(&src), None, &mut tmp, h, w, &axis(w, tw)?, true, 3)?;
        let mut img = exec.alloc(th * tw * 3)?;
        exec.nvl_resample(None, Some(&tmp), &mut img, tw, h, &axis(h, th)?, false, 3)?;
        exec.nvl_patchify_h(&img, &mut ws.patches, th, tw, p, cfg.mean, cfg.std)?;

        // ---- embed: patch GEMM + the cropped field, the prefix after
        ring(&exec, &self.patch_w, &ws.patches, &mut ws.e32, n_patch)?;
        exec.copy_region(&ws.e32, 0, &mut ws.x, 0, n_patch * d)?;
        exec.nvl_pos_add(&mut ws.x, 0, &self.pos_grid, self.grid_side, hp, wp, d)?;
        exec.copy_region(&self.cls, 0, &mut ws.x, n_patch * d, self.n_prefix * d)?;

        // ---- the blocks (Parse's RADIO kernels, one picture a pass)
        let scale = 1.0 / (self.hd as f32).sqrt();
        let first = &self.blocks[0].ln1;
        exec.layernorm_f16(
            &ws.x,
            &first.w,
            &first.b,
            &mut ws.n16,
            rows,
            d,
            TOWER_LN_EPS,
        )?;
        let n_layer = self.blocks.len();
        for li in 0..n_layer {
            let blk = &self.blocks[li];
            exec.matvec_batch_f16_h(&blk.wqkv, &ws.n16, &mut ws.qkv, rows)?;
            exec.dp_qkv_split_rope_h(
                &ws.qkv,
                &blk.bq,
                &blk.bv,
                &self.no_rope,
                &self.no_rope,
                &mut ws.q,
                &mut ws.k,
                &mut ws.v,
                d,
                self.hd,
                rows,
                rows,
                0,
                scale,
            )?;
            exec.vision_attn_h_rows(
                &ws.q,
                &ws.k,
                &ws.v,
                &mut ws.a16,
                0,
                rows,
                self.heads,
                self.hd,
            )?;
            exec.matvec_batch_f16_h(&blk.wo, &ws.a16, &mut ws.proj, rows)?;
            exec.dp_res_ls_ln_h(
                &mut ws.x,
                &ws.proj,
                &blk.bo,
                &self.ones,
                &blk.ln2.w,
                &blk.ln2.b,
                &mut ws.n16,
                rows,
                d,
                TOWER_LN_EPS,
            )?;
            exec.matvec_batch_f16_h_gelu(&blk.fc1, &ws.n16, &mut ws.ff, &blk.fc1_b, rows)?;
            exec.matvec_batch_f16_h(&blk.fc2, &ws.ff, &mut ws.proj, rows)?;
            // the next block's LN1 rides this seam; after the last one the
            // norm it writes is unused (the projector reads the residual)
            let next = if li + 1 < n_layer {
                &self.blocks[li + 1].ln1
            } else {
                &blk.ln2
            };
            exec.dp_res_ls_ln_h(
                &mut ws.x,
                &ws.proj,
                &blk.fc2_b,
                &self.ones,
                &next.w,
                &next.b,
                &mut ws.n16,
                rows,
                d,
                TOWER_LN_EPS,
            )?;
        }

        // ---- the projector over the patch rows (the prefix rows are dropped)
        exec.layernorm(
            &ws.x,
            &self.final_ln.w,
            &self.final_ln.b,
            &mut ws.e32,
            n_patch,
            d,
            TOWER_LN_EPS,
        )?;
        let toks = n_patch / (cfg.shuffle * cfg.shuffle);
        let sh = cfg.shuffle * cfg.shuffle * d;
        let idx = exec.to_device_u32(&shuffle_index(hp, wp))?;
        exec.embed_gather_batch(&ws.e32, &idx, &mut ws.shuf, d, n_patch)?;
        exec.rmsnorm_batch(
            &ws.shuf,
            &self.rms_w,
            &mut ws.shuf_n,
            sh,
            PROJ_RMS_EPS,
            toks,
        )?;
        let hid = self.fc1.dims[1];
        exec.bf16_gemm(&self.fc1, None, &ws.shuf_n, &mut ws.hid, toks)?;
        exec.nvl_relu2(&mut ws.hid, toks * hid)?;
        let mut rows_out = exec.alloc(toks * self.hidden)?;
        exec.bf16_gemm(&self.fc2, None, &ws.hid, &mut rows_out, toks)?;
        Ok(PictureRows {
            rows: rows_out,
            n: toks,
        })
    }
}

impl SuperVision {
    /// The tile a `w x h` picture (one of `n_pictures`) gets: (patches
    /// across, patches down).
    pub fn tile(&self, w: usize, h: usize, n_pictures: usize) -> (usize, usize) {
        tile_patches(w, h, &self.cfg, self.cfg.picture_budget(n_pictures))
    }

    /// The normalized patch rows the last `encode` fed the patch GEMM.
    pub fn last_patches(&self, n_patch: usize) -> Result<Vec<f16>, GpuModelError> {
        let ws = self
            .ws
            .as_ref()
            .ok_or_else(|| GpuModelError::Unsupported("no picture encoded yet".into()))?;
        let k = 3 * self.cfg.patch * self.cfg.patch;
        Ok(self.exec.to_host_f16_len(&ws.patches, n_patch * k)?)
    }
}

/// `y = W x` (f32 landing) on the f16 rings - Parse's tower GEMM.
fn ring(
    exec: &GpuExecutor,
    w: &HalfTensor,
    x16: &CudaSlice<f16>,
    y: &mut CudaSlice<f32>,
    rows: usize,
) -> Result<(), GpuModelError> {
    if exec.has_f16_gemm_ring() {
        exec.f16_gemm_ring_rows(w, x16, 0, y, 0, rows, None)?;
    } else {
        exec.matvec_batch_f16(w, x16, y, rows)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> VisionConfig {
        VisionConfig {
            patch: 16,
            min_patches: 1024,
            max_patches: 13312,
            max_model_len: 16384,
            shuffle: 2,
            mean: [0.0; 3],
            std: [1.0; 3],
            image_id: 18,
            start_id: 19,
            end_id: 20,
        }
    }

    /// The checkpoint's own `_compute_target_patches` on these sizes (run
    /// with the shipped processor): (w, h) -> (w_patches, h_patches).
    #[test]
    fn tiles_match_the_processor() {
        let c = cfg();
        let budget = c.picture_budget(1);
        assert_eq!(budget, 13312);
        for ((w, h), want) in [
            ((640, 480), (40, 30)),
            ((1920, 1080), (120, 68)),
            ((800, 600), (50, 38)),
            ((32, 32), (32, 32)),
            ((4000, 3000), (132, 100)),
            ((1024, 1024), (64, 64)),
            ((1000, 1), (254, 6)),
            ((17, 333), (10, 104)),
            ((2048, 1664), (128, 104)),
            ((512, 512), (32, 32)),
            ((1280, 720), (80, 46)),
            ((48, 48), (32, 32)),
        ] {
            assert_eq!(tile_patches(w, h, &c, budget), want, "{w} x {h}");
        }
    }

    #[test]
    fn resample_plans_are_normalized_and_identity_is_exact() {
        let (xmin, xsize, w, taps) = aa_plan(48, 48);
        assert_eq!(taps, 1);
        assert!(xmin.iter().enumerate().all(|(i, &x)| x as usize == i));
        assert!(xsize.iter().all(|&n| n == 1) && w.iter().all(|&v| v == 1.0));
        for (inl, out) in [(1920, 1280), (480, 512), (17, 160), (4000, 2112)] {
            let (xmin, xsize, w, taps) = aa_plan(inl, out);
            for i in 0..out {
                let n = xsize[i] as usize;
                assert!(n >= 1 && n <= taps && xmin[i] as usize + n <= inl);
                let s: f32 = w[i * taps..i * taps + n].iter().sum();
                assert!((s - 1.0).abs() < 1e-5, "{inl}->{out} row {i} sums {s}");
            }
        }
    }

    #[test]
    fn shuffle_takes_two_by_two_neighbours_row_then_column() {
        // a 4 x 6 grid: output (0, 0) takes source rows 0, 1, 6, 7
        let idx = shuffle_index(4, 6);
        assert_eq!(idx.len(), 24);
        assert_eq!(&idx[..4], &[0, 1, 6, 7]);
        assert_eq!(&idx[4..8], &[2, 3, 8, 9]);
        assert_eq!(&idx[12..16], &[12, 13, 18, 19]);
    }
}
