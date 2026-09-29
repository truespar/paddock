//! PaddleOCR-VL NaViT vision tower + projector. Reference: the
//! checkpoint's `modeling_paddleocr_vl.py` (`PaddleOCRVisionTransformer` +
//! `Projector`); qwen35/vision.rs is the structural template - same SigLIP
//! geometry (1152 embd, 16 heads, head_dim 72), same merged 2×2-block patch
//! order, same vision M-RoPE kernel. Dataflow:
//!
//!   patches (14×14, MERGED 2×2-block order) -> patch GEMM (+bias)
//!   -> +learned pos-embd (27×27 table bilinear-resized to the grid,
//!      half-pixel centers = torch F.interpolate align_corners=False)
//!   -> 27 × [ LN1 -> q/k/v(+bias) -> vision M-RoPE(q,k) -> bidirectional attn
//!            -> out(+bias) -> +res -> LN2 -> up(+bias) -> tanh-GELU -> down(+bias)
//!            -> +res ]
//!   -> post-LN -> projector: pre-norm LN (eps 1e-5!) -> [N/4, 4·embd]
//!   -> mm.1(+bias) -> erf-GELU -> mm.2(+bias) -> [N/4, 1024] embeddings.
//!
//! Two GELUs deliberately: the encoder FFN is `gelu_pytorch_tanh` (`pd_gelu`),
//! the projector is transformers' plain `GELUActivation` = exact erf
//! (`pd_gelu_erf`). Same deepseek_ocr lesson, new tower.
//!
//! The vision M-RoPE layout was verified against the reference before reuse:
//! pairs 0..17 rotate by the patch ROW at exponent p, pairs 18..35 by the
//! COLUMN at p-18, NeoX pair (p, p+36) - exactly `SigLIPRotaryEmbedding(36)`
//! + `rope_emb.repeat(1,2)` + `rotate_half`, and exactly what llama.cpp's
//!   `clip_graph_paddleocr` builds with sections [18,18].
//!
//! f32 activations over f16 weight planes (the mmproj tower class); the
//! oracle gate runs class tolerance against the reference's f32.

use std::sync::Arc;

use cudarc::driver::CudaSlice;
use half::f16;
use paddock_models::gguf::Value;
use paddock_models::mapped::MappedGguf;

use super::preprocess::PixelBudget;
use crate::gpu::{GpuError, GpuExecutor, HalfTensor};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::qwen35::vision::host_f32;

/// The `clip.projector_type` this loader answers to (llama.cpp's converter
/// vocabulary for the family).
pub const PROJECTOR_TYPE: &str = "paddleocr";

/// Loader-shaped error (the family reuses gpt_oss's error enum, which has no
/// dedicated load variant - same convention as the deepseek_ocr loader).
fn load_err(msg: String) -> GpuModelError {
    GpuError::Driver(msg).into()
}

/// The projector's pre-norm LayerNorm eps. HARDCODED 1e-5 in the reference
/// (`Projector.pre_norm = LayerNorm(..., eps=1e-05)`) while the encoder runs
/// 1e-6 - the GGUF has one eps key and it carries the encoder's, so this one
/// is a constant here (llama.cpp hardcodes the same 1e-5 in its graph).
pub const PROJ_EPS: f32 = 1e-5;

struct VBlock {
    ln1_w: CudaSlice<f32>,
    ln1_b: CudaSlice<f32>,
    // q/k/v ship as separate GGUF tensors for this family (no fused plane)
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

/// The encoded image: merged-grid embeddings ready for LLM injection.
pub struct VisionOutput {
    /// [n_tokens, 1024] device-resident image embeddings.
    pub embd: CudaSlice<f32>,
    /// Output grid (post 2×2 merge) - the decoder M-RoPE h/w extents.
    pub nx: usize,
    pub ny: usize,
}

/// Optional stage taps for the oracle gate: filled by
/// [`VisionModel::encode_batch_taps`] when requested, so the gate can bisect
/// a failure to encoder-input vs tower vs projector instead of one opaque
/// end-to-end diff. Serving never asks for them and pays nothing.
#[derive(Default)]
pub struct EncodeTaps {
    /// Encoder input rows [rows, embd] (patch GEMM + pos table), merged order.
    pub embd: Vec<f32>,
    /// Post-post_ln tower output [rows, embd], merged order.
    pub vit: Vec<f32>,
    /// f64 sum of the hidden state after each encoder layer - matches the
    /// oracle manifest's `stage_sums` for layer-level bisecting.
    pub layer_sums: Vec<f64>,
    /// f64 sum after each layer's ATTENTION residual (pre-FFN) - splits a
    /// bad layer into its attention vs FFN half.
    pub attn_sums: Vec<f64>,
    /// Full hidden state after selected layers (0/3/9) - the early-depth
    /// full-tensor anchors the gate diffs element-wise.
    pub layers: std::collections::HashMap<usize, Vec<f32>>,
    /// Layer 0's attention output (post out-proj, PRE-residual) - matches the
    /// oracle's `attn0` hook on `self_attn`, splitting the layer's two halves.
    pub attn0: Vec<f32>,
}

/// Persistent encode workspace. The bring-up path allocated every
/// activation buffer per `encode_batch` call (~12 `alloc_zeros` + pageable
/// `clone_htod` uploads + a full host pos-embd interpolation, per call) - the
/// a c8 trace measured that as ~80 ms of host wall per group against
/// ~5 ms of actual GPU kernels (6% busy). Same lesson as deepseek_ocr's
/// `TowerWs` and transfer.rs's upload-into-slab doctrine, so the tower now
/// carries its slabs for its whole life and grows them on demand: NaViT has
/// no closed geometry set to preallocate against, so the first request at a
/// new high-water geometry pays one alloc burst and steady state pays none.
/// Every consumer writes its full live range before reading (the ops all take
/// explicit `rows`), so slab reuse never observes stale bytes.
struct TowerWs {
    /// f16 staging for every GEMM's activations (one slab, sized by the
    /// largest convert of the pass - the qwen35 rationale, now persistent).
    s16: CudaSlice<f16>,
    x: CudaSlice<f32>,
    n: CudaSlice<f32>,
    q: CudaSlice<f32>,
    k: CudaSlice<f32>,
    v: CudaSlice<f32>,
    a: CudaSlice<f32>,
    up: CudaSlice<f32>,
    m: CudaSlice<f32>,
    out: CudaSlice<f32>,
    patches: CudaSlice<f32>,
    pos: CudaSlice<u32>,
    /// Interpolated + merge-ordered pos-embd per (pw, ph), device-resident -
    /// the host bilinear pass over ph·pw·embd floats ran per call before.
    /// One entry covers one image; group members add it at their row offset.
    pe: std::collections::HashMap<(usize, usize), CudaSlice<f32>>,
    /// Host staging for the im2col + position walk, reused across calls so
    /// the multi-MB `vec![0f32; ..]` zeroing doesn't recur per encode.
    host_patches: Vec<f32>,
    host_pos: Vec<u32>,
}

/// Patch rows one batched tower pass takes before a group stops taking
/// pictures (a picture larger than this runs alone): ~56 KB of workspace per
/// row, so ~0.9 GB - two maximum-size pictures, or three default pages.
pub(crate) const TOWER_PASS_ROWS: usize = 16384;

/// Distinct geometries worth caching pos-embd planes for. Documents pages
/// cluster on a handful of shapes; past this the whole cache resets (simple
/// and honest - no LRU bookkeeping for a table this small).
const PE_CACHE_MAX: usize = 8;

impl TowerWs {
    fn new(exec: &GpuExecutor) -> Result<Self, GpuError> {
        Ok(Self {
            s16: exec.alloc_f16(1)?,
            x: exec.alloc(1)?,
            n: exec.alloc(1)?,
            q: exec.alloc(1)?,
            k: exec.alloc(1)?,
            v: exec.alloc(1)?,
            a: exec.alloc(1)?,
            up: exec.alloc(1)?,
            m: exec.alloc(1)?,
            out: exec.alloc(1)?,
            patches: exec.alloc(1)?,
            pos: exec.alloc_u32(1)?,
            pe: std::collections::HashMap::new(),
            host_patches: Vec::new(),
            host_pos: Vec::new(),
        })
    }

    fn grow_f32(buf: &mut CudaSlice<f32>, exec: &GpuExecutor, need: usize) -> Result<(), GpuError> {
        if buf.len() < need {
            // release the old buffer first: assigning straight over it built
            // the new one while the old was still held (old + new at the peak)
            *buf = exec.alloc(1)?;
            *buf = exec.alloc(need)?;
        }
        Ok(())
    }

    fn grow_f16(buf: &mut CudaSlice<f16>, exec: &GpuExecutor, need: usize) -> Result<(), GpuError> {
        if buf.len() < need {
            // release the old buffer first: assigning straight over it built
            // the new one while the old was still held (old + new at the peak)
            *buf = exec.alloc_f16(1)?;
            *buf = exec.alloc_f16(need)?;
        }
        Ok(())
    }

    fn grow_u32(buf: &mut CudaSlice<u32>, exec: &GpuExecutor, need: usize) -> Result<(), GpuError> {
        if buf.len() < need {
            // release the old buffer first: assigning straight over it built
            // the new one while the old was still held (old + new at the peak)
            *buf = exec.alloc_u32(1)?;
            *buf = exec.alloc_u32(need)?;
        }
        Ok(())
    }
}

pub struct VisionModel {
    exec: Arc<GpuExecutor>,
    ws: TowerWs,
    #[allow(dead_code)] // geometry record; layer count is blocks.len()
    n_layers: usize,
    embd: usize,
    n_heads: usize,
    head_dim: usize,
    patch: usize,
    eps: f32,
    /// Learned 27×27 pos table on host [n_side², embd] row-major - bilinearly
    /// resized + merge-reordered per grid, then uploaded.
    pos_embd: Vec<f32>,
    n_side: usize,
    /// smart_resize area budget, from the GGUF's own
    /// `clip.vision.image_{min,max}_pixels` keys.
    pub budget: PixelBudget,

    patch_w: HalfTensor,
    patch_b: CudaSlice<f32>,
    blocks: Vec<VBlock>,
    post_ln_w: CudaSlice<f32>,
    post_ln_b: CudaSlice<f32>,
    // projector (the reference's mlp_AR)
    proj_norm_w: CudaSlice<f32>,
    proj_norm_b: CudaSlice<f32>,
    mm1: HalfTensor,
    mm1_b: CudaSlice<f32>,
    mm2: HalfTensor,
    mm2_b: CudaSlice<f32>,
}

impl VisionModel {
    pub fn load(exec: Arc<GpuExecutor>, map: &MappedGguf) -> Result<Self, GpuModelError> {
        let meta = |k: &str| map.gguf().metadata.get(k);
        if let Some(t) = meta("clip.projector_type").and_then(Value::as_str)
            && t != PROJECTOR_TYPE
        {
            return Err(load_err(format!(
                "mmproj projector_type is {t:?}, expected {PROJECTOR_TYPE:?}"
            )));
        }
        let u = |k: &str| -> Result<usize, GpuModelError> {
            meta(k)
                .and_then(Value::as_u64)
                .map(|v| v as usize)
                .ok_or_else(|| load_err(format!("mmproj missing {k}")))
        };
        let n_layers = u("clip.vision.block_count")?;
        let embd = u("clip.vision.embedding_length")?;
        let n_heads = u("clip.vision.attention.head_count")?;
        if n_heads == 0 || embd % n_heads != 0 {
            return Err(load_err(format!(
                "vision embd {embd} is not divisible by {n_heads} heads"
            )));
        }
        let eps = meta("clip.vision.attention.layer_norm_epsilon")
            .and_then(Value::as_f32)
            .unwrap_or(1e-6);
        let budget = PixelBudget {
            min_pixels: u("clip.vision.image_min_pixels")
                .unwrap_or(PixelBudget::DEFAULT.min_pixels),
            max_pixels: u("clip.vision.image_max_pixels")
                .unwrap_or(PixelBudget::DEFAULT.max_pixels),
        };

        let e = exec.clone();
        let dt =
            move |name: &str| -> Result<HalfTensor, GpuModelError> { Ok(e.upload_f16(map, name)?) };
        let e = exec.clone();
        let vec1 = move |name: &str| -> Result<CudaSlice<f32>, GpuModelError> {
            let (host, _) = host_f32(map, name)?;
            Ok(e.to_device(&host)?)
        };

        // conv stem [14,14,3,1152] (kx fastest) -> relabel [588, 1152]: read as
        // rows-per-output the layout is already the im2col order this module
        // builds (c·196 + ky·14 + kx) - a relabel, not a permute.
        let mut patch_w = dt("v.patch_embd.weight")?;
        let pd = patch_w.dims.clone();
        if pd.len() != 4 || pd[0] != pd[1] || pd[2] != 3 || pd[3] != embd {
            return Err(load_err(format!(
                "v.patch_embd.weight dims {pd:?} are not [k, k, 3, {embd}]"
            )));
        }
        let patch = pd[0];
        patch_w.dims = vec![patch * patch * 3, embd];

        let (pos_embd, pos_dims) = host_f32(map, "v.position_embd.weight")?;
        if pos_dims.first() != Some(&embd) || pos_dims.len() < 2 {
            return Err(load_err(format!(
                "v.position_embd.weight dims {pos_dims:?} are not [{embd}, positions]"
            )));
        }
        let n_side = (pos_dims[1] as f64).sqrt() as usize;
        if n_side * n_side != pos_dims[1] {
            return Err(load_err(format!(
                "pos-embd grid {} is not square",
                pos_dims[1]
            )));
        }

        let mut blocks = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let t = |s: &str| format!("v.blk.{i}.{s}");
            let square = |w: &HalfTensor, name: &str| -> Result<(), GpuModelError> {
                if w.dims != [embd, embd] {
                    return Err(load_err(format!(
                        "{name} dims {:?} are not [{embd}, {embd}]",
                        w.dims
                    )));
                }
                Ok(())
            };
            let wq = dt(&t("attn_q.weight"))?;
            let wk = dt(&t("attn_k.weight"))?;
            let wv = dt(&t("attn_v.weight"))?;
            let wo = dt(&t("attn_out.weight"))?;
            square(&wq, "attn_q")?;
            square(&wk, "attn_k")?;
            square(&wv, "attn_v")?;
            square(&wo, "attn_out")?;
            blocks.push(VBlock {
                ln1_w: vec1(&t("ln1.weight"))?,
                ln1_b: vec1(&t("ln1.bias"))?,
                wq,
                wk,
                wv,
                bq: vec1(&t("attn_q.bias"))?,
                bk: vec1(&t("attn_k.bias"))?,
                bv: vec1(&t("attn_v.bias"))?,
                wo,
                bo: vec1(&t("attn_out.bias"))?,
                ln2_w: vec1(&t("ln2.weight"))?,
                ln2_b: vec1(&t("ln2.bias"))?,
                up_w: dt(&t("ffn_up.weight"))?,
                up_b: vec1(&t("ffn_up.bias"))?,
                down_w: dt(&t("ffn_down.weight"))?,
                down_b: vec1(&t("ffn_down.bias"))?,
            });
        }

        let mm1 = dt("mm.1.weight")?;
        let mm2 = dt("mm.2.weight")?;
        // the merger eats 4 tower rows per output row; anything else means the
        // file is not this projector
        if mm1.dims != [4 * embd, 4 * embd] || mm2.dims[0] != 4 * embd {
            return Err(load_err(format!(
                "projector dims mm.1 {:?} / mm.2 {:?} are not the 2×2-merge shape",
                mm1.dims, mm2.dims
            )));
        }

        let ws = TowerWs::new(&exec).map_err(|e| load_err(e.to_string()))?;
        let me = Self {
            exec,
            ws,
            n_layers,
            embd,
            n_heads,
            head_dim: embd / n_heads,
            patch,
            eps,
            pos_embd,
            n_side,
            budget,
            patch_w,
            patch_b: vec1("v.patch_embd.bias")?,
            blocks,
            post_ln_w: vec1("v.post_ln.weight")?,
            post_ln_b: vec1("v.post_ln.bias")?,
            proj_norm_w: vec1("mm.input_norm.weight")?,
            proj_norm_b: vec1("mm.input_norm.bias")?,
            mm1,
            mm1_b: vec1("mm.1.bias")?,
            mm2,
            mm2_b: vec1("mm.2.bias")?,
        };
        tracing::info!(
            weight_mib = me.weight_bytes() / (1 << 20),
            layers = me.n_layers,
            budget = ?me.budget,
            "paddleocr-vl mmproj resident at f16 (f32 accumulate)"
        );
        Ok(me)
    }

    /// Device bytes the f16 weight planes hold - the estimator's input.
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
        blk + self.patch_w.bytes() + self.mm1.bytes() + self.mm2.bytes()
    }

    /// Decoder width the projector emits into.
    pub fn llm_embd(&self) -> usize {
        self.mm2.dims[1]
    }

    /// Bilinearly resize the learned pos table to (ph, pw) and reorder into
    /// the merged 2×2-block patch order. Half-pixel centers with edge clamp -
    /// torch `F.interpolate(mode="bilinear", align_corners=False)` semantics,
    /// which is what the reference's `interpolate_pos_encoding` runs.
    fn pos_embd_for(&self, pw: usize, ph: usize) -> Vec<f32> {
        let (e, s) = (self.embd, self.n_side);
        let sample = |gy: usize, gx: usize| -> &[f32] {
            let idx = gy * s + gx;
            &self.pos_embd[idx * e..(idx + 1) * e]
        };
        let mut grid = vec![0f32; ph * pw * e];
        if pw == s && ph == s {
            grid.copy_from_slice(&self.pos_embd);
        } else {
            for y in 0..ph {
                let sy = ((y as f32 + 0.5) * s as f32 / ph as f32 - 0.5).clamp(0.0, (s - 1) as f32);
                let y0 = sy.floor() as usize;
                let y1 = (y0 + 1).min(s - 1);
                let fy = sy - y0 as f32;
                for x in 0..pw {
                    let sx =
                        ((x as f32 + 0.5) * s as f32 / pw as f32 - 0.5).clamp(0.0, (s - 1) as f32);
                    let x0 = sx.floor() as usize;
                    let x1 = (x0 + 1).min(s - 1);
                    let fx = sx - x0 as f32;
                    let (a, b, c, d) = (
                        sample(y0, x0),
                        sample(y0, x1),
                        sample(y1, x0),
                        sample(y1, x1),
                    );
                    let out = &mut grid[(y * pw + x) * e..(y * pw + x + 1) * e];
                    for j in 0..e {
                        let top = a[j] + (b[j] - a[j]) * fx;
                        let bot = c[j] + (d[j] - c[j]) * fx;
                        out[j] = top + (bot - top) * fy;
                    }
                }
            }
        }
        // merged 2×2-block order, matching encode's im2col walk
        let mut out = vec![0f32; ph * pw * e];
        let mut ptr = 0usize;
        for yb in (0..ph).step_by(2) {
            for xb in (0..pw).step_by(2) {
                for dy in 0..2 {
                    for dx in 0..2 {
                        let src = ((yb + dy) * pw + (xb + dx)) * e;
                        out[ptr * e..(ptr + 1) * e].copy_from_slice(&grid[src..src + e]);
                        ptr += 1;
                    }
                }
            }
        }
        out
    }

    /// Size the tower's workspace for a `rows`-patch-row pass. Called at
    /// attach for the widest pass serving can form (`TOWER_PASS_ROWS`, or one
    /// maximum-size picture alone), which puts the workspace in the ledger
    /// before the plan reads its grant; every pass after that fits it.
    ///
    /// It used to grow on demand to the largest group seen - up to eight
    /// same-size pictures, ~3.7 GB at the published maximum - kept that for
    /// the process's life, and held old and new together while growing.
    pub fn size_workspace(&mut self, rows: usize) -> Result<(), GpuError> {
        let exec = Arc::clone(&self.exec);
        let e = self.embd;
        let k2 = self.patch * self.patch;
        let row_f = 3 * k2;
        let ffn = self.blocks[0].up_w.dims[1];
        let stage = (rows * ffn.max(e).max(self.patch_w.dims[0]))
            .max((rows / 4) * self.mm1.dims[0].max(self.mm1.dims[1]));
        let n4 = rows / 4;
        let mid = self.mm1.dims[1];
        let out_dim = self.mm2.dims[1];
        TowerWs::grow_f16(&mut self.ws.s16, &exec, stage)?;
        for (buf, need) in [
            (&mut self.ws.x, rows * e),
            (&mut self.ws.n, rows * e),
            (&mut self.ws.q, rows * e),
            (&mut self.ws.k, rows * e),
            (&mut self.ws.v, rows * e),
            (&mut self.ws.a, rows * e),
            (&mut self.ws.up, rows * ffn),
            (&mut self.ws.m, n4 * mid),
            (&mut self.ws.out, n4 * out_dim),
            (&mut self.ws.patches, rows * row_f),
        ] {
            TowerWs::grow_f32(buf, &exec, need)?;
        }
        TowerWs::grow_u32(&mut self.ws.pos, &exec, 4 * rows)?;
        if self.ws.host_patches.len() < rows * row_f {
            self.ws.host_patches.resize(rows * row_f, 0.0);
        }
        if self.ws.host_pos.len() < 4 * rows {
            self.ws.host_pos.resize(4 * rows, 0);
        }
        Ok(())
    }

    /// Patch rows one picture of `w`x`h` (post-resize) occupies in a pass.
    pub fn patch_rows(&self, w: usize, h: usize) -> usize {
        (w / self.patch) * (h / self.patch)
    }

    /// Encode one normalized planar image ([3][h][w] f32) - see
    /// [`super::preprocess::preprocess_rgb`]. `w`/`h` must be multiples of 28.
    pub fn encode(
        &mut self,
        img: &[f32],
        w: usize,
        h: usize,
    ) -> Result<VisionOutput, GpuModelError> {
        let mut out = self.encode_batch(&[(img, w, h)])?;
        Ok(out.pop().expect("batch of one"))
    }

    /// Encode B same-size images in one tower pass. Attention runs per image
    /// over its own row window, so rows are independent of batch placement -
    /// the qwen35 tower's contract, same reasoning (its module doc has the
    /// full numeric story).
    pub fn encode_batch(
        &mut self,
        imgs: &[(&[f32], usize, usize)],
    ) -> Result<Vec<VisionOutput>, GpuModelError> {
        self.encode_batch_taps(imgs, None)
    }

    /// [`Self::encode_batch`] plus optional stage taps for the oracle gate.
    pub fn encode_batch_taps(
        &mut self,
        imgs: &[(&[f32], usize, usize)],
        mut taps: Option<&mut EncodeTaps>,
    ) -> Result<Vec<VisionOutput>, GpuModelError> {
        let b = imgs.len();
        assert!(b > 0);
        let (_, w, h) = imgs[0];
        let (patch, e) = (self.patch, self.embd);
        assert!(
            w % (patch * 2) == 0 && h % (patch * 2) == 0,
            "image must be 28-aligned"
        );
        for (img, iw, ih) in imgs {
            assert_eq!((*iw, *ih), (w, h), "encode_batch group must share dims");
            assert_eq!(img.len(), 3 * w * h);
        }
        let (pw, ph) = (w / patch, h / patch);
        let n = pw * ph;
        let rows = b * n;
        let exec = Arc::clone(&self.exec);

        // workspace residency for this geometry - sized once at attach for the
        // widest pass (`size_workspace`), so this is a no-op in serving
        self.size_workspace(rows)?;
        let k2 = patch * patch;
        let row_f = 3 * k2;
        let ffn = self.blocks[0].up_w.dims[1];
        let n4 = rows / 4;
        let mid = self.mm1.dims[1];
        let out_dim = self.mm2.dims[1];

        // host im2col in the merged 2×2-block order. Each yb block-row of one
        // image is one contiguous band of 2·pw patch rows, so bands fan out
        // to threads with no overlap while the band-internal walk keeps the
        // serial order exactly - byte-identical output either way.
        let hp = &mut self.ws.host_patches;
        let hpos = &mut self.ws.host_pos;
        let band_elems = 2 * pw * row_f;
        let fill = |gb: usize, dst: &mut [f32]| {
            let (bi, byb) = (gb / (ph / 2), gb % (ph / 2));
            let yb = byb * 2;
            let img = imgs[bi].0;
            let mut p = 0usize;
            for xb in (0..pw).step_by(2) {
                for dy in 0..2 {
                    for dx in 0..2 {
                        let (py, px) = (yb + dy, xb + dx);
                        let d = &mut dst[p * row_f..(p + 1) * row_f];
                        for c in 0..3 {
                            for ky in 0..patch {
                                let src = c * w * h + (py * patch + ky) * w + px * patch;
                                d[c * k2 + ky * patch..c * k2 + ky * patch + patch]
                                    .copy_from_slice(&img[src..src + patch]);
                            }
                        }
                        p += 1;
                    }
                }
            }
        };
        let n_bands = rows / (2 * pw);
        let nt = std::thread::available_parallelism()
            .map_or(1, |v| v.get())
            .min(n_bands);
        if nt > 1 && rows * row_f >= (1 << 20) {
            let mut items: Vec<(usize, &mut [f32])> = hp[..rows * row_f]
                .chunks_mut(band_elems)
                .enumerate()
                .collect();
            let per = items.len().div_ceil(nt);
            std::thread::scope(|s| {
                for group in items.chunks_mut(per) {
                    s.spawn(|| {
                        for (gb, dst) in group.iter_mut() {
                            fill(*gb, dst);
                        }
                    });
                }
            });
        } else {
            for (gb, dst) in hp[..rows * row_f].chunks_mut(band_elems).enumerate() {
                fill(gb, dst);
            }
        }
        // vision rope positions, axis-major [4, rows] = [y, x, y, x] per row
        // (the layout pd_mrope_vision reads)
        for bi in 0..b {
            let base = bi * n;
            let mut ptr = 0usize;
            for yb in (0..ph).step_by(2) {
                for xb in (0..pw).step_by(2) {
                    for dy in 0..2 {
                        for dx in 0..2 {
                            hpos[base + ptr] = (yb + dy) as u32;
                            hpos[rows + base + ptr] = (xb + dx) as u32;
                            hpos[2 * rows + base + ptr] = (yb + dy) as u32;
                            hpos[3 * rows + base + ptr] = (xb + dx) as u32;
                            ptr += 1;
                        }
                    }
                }
            }
        }
        exec.upload_f32(&hp[..rows * row_f], &mut self.ws.patches)?;
        exec.upload_u32(&hpos[..4 * rows], &mut self.ws.pos)?;

        // pos-embd plane for this grid: interpolate once, keep on device.
        // One entry is one image's plane; group members add it at their own
        // row window below (the old path replicated it b× on the host).
        if !self.ws.pe.contains_key(&(pw, ph)) {
            let host = self.pos_embd_for(pw, ph);
            if self.ws.pe.len() >= PE_CACHE_MAX {
                self.ws.pe.clear();
            }
            let dev = exec.to_device(&host)?;
            self.ws.pe.insert((pw, ph), dev);
        }

        let ws = &mut self.ws;
        // patch GEMM + bias + learned positions
        exec.convert_f32_f16(&ws.patches, &mut ws.s16, rows * self.patch_w.dims[0])?;
        exec.matvec_batch_f16(&self.patch_w, &ws.s16, &mut ws.x, rows)?;
        exec.bias_add(&mut ws.x, &self.patch_b, rows, e)?;
        let pe = &ws.pe[&(pw, ph)];
        for bi in 0..b {
            exec.add_at(&mut ws.x, bi * n * e, pe, 0, n * e)?;
        }
        if let Some(t) = taps.as_deref_mut() {
            t.embd = exec.to_host_len(&ws.x, rows * e)?;
        }

        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let theta_scale = 10000f32.powf(-2.0 / (self.head_dim / 2) as f32);

        // The elementwise chain between the GEMMs runs FUSED: LN
        // writes the f16 staging plane directly, the q/k biases ride the rope
        // load, o/down biases ride the residual add, and the FFN's
        // bias+GELU+convert is one pass. Every fusion is bit-identical to the
        // unfused ops (same IEEE order, one final f16 round) - see
        // vision.cuh's fusion header - so the oracle gates hold unchanged.
        for blk in &self.blocks {
            exec.layernorm_f16(
                &ws.x,
                &blk.ln1_w,
                &blk.ln1_b,
                &mut ws.s16,
                rows,
                e,
                self.eps,
            )?;
            exec.matvec_batch_f16(&blk.wq, &ws.s16, &mut ws.q, rows)?;
            exec.matvec_batch_f16(&blk.wk, &ws.s16, &mut ws.k, rows)?;
            exec.matvec_batch_f16(&blk.wv, &ws.s16, &mut ws.v, rows)?;
            exec.bias_add(&mut ws.v, &blk.bv, rows, e)?;
            exec.mrope_vision_bias(
                &mut ws.q,
                &blk.bq,
                &ws.pos,
                rows,
                self.n_heads,
                self.head_dim,
                theta_scale,
            )?;
            exec.mrope_vision_bias(
                &mut ws.k,
                &blk.bk,
                &ws.pos,
                rows,
                self.n_heads,
                self.head_dim,
                theta_scale,
            )?;
            for bi in 0..b {
                exec.vision_attn_at(
                    &ws.q,
                    &ws.k,
                    &ws.v,
                    &mut ws.a,
                    bi * n,
                    n,
                    self.n_heads,
                    self.head_dim,
                    scale,
                )?;
            }
            exec.convert_f32_f16(&ws.a, &mut ws.s16, rows * e)?;
            exec.matvec_batch_f16(&blk.wo, &ws.s16, &mut ws.n, rows)?;
            if let Some(t) = taps.as_deref_mut()
                && t.attn_sums.is_empty()
            {
                // ws.n stays unbiased (the bias rides the residual add
                // below) - reconstruct the biased tap on the host, the
                // same IEEE f32 add the old bias_add kernel did
                let mut host = exec.to_host_len(&ws.n, rows * e)?;
                let bo = exec.to_host_len(&blk.bo, e)?;
                for (i, v) in host.iter_mut().enumerate() {
                    *v += bo[i % e];
                }
                t.attn0 = host;
            }
            exec.add_bias_res(&mut ws.x, &ws.n, &blk.bo, rows, e)?;
            if let Some(t) = taps.as_deref_mut() {
                t.attn_sums.push(
                    exec.to_host_len(&ws.x, rows * e)?
                        .iter()
                        .map(|&v| v as f64)
                        .sum(),
                );
            }

            exec.layernorm_f16(
                &ws.x,
                &blk.ln2_w,
                &blk.ln2_b,
                &mut ws.s16,
                rows,
                e,
                self.eps,
            )?;
            exec.matvec_batch_f16(&blk.up_w, &ws.s16, &mut ws.up, rows)?;
            exec.gelu_bias_f16(&ws.up, &blk.up_b, &mut ws.s16, rows, ffn)?; // gelu_pytorch_tanh
            exec.matvec_batch_f16(&blk.down_w, &ws.s16, &mut ws.n, rows)?;
            exec.add_bias_res(&mut ws.x, &ws.n, &blk.down_b, rows, e)?;
            if let Some(t) = taps.as_deref_mut() {
                let host = exec.to_host_len(&ws.x, rows * e)?;
                let li = t.layer_sums.len();
                t.layer_sums.push(host.iter().map(|&v| v as f64).sum());
                if matches!(li, 0 | 3 | 9) {
                    t.layers.insert(li, host);
                }
            }
        }

        exec.layernorm(
            &ws.x,
            &self.post_ln_w,
            &self.post_ln_b,
            &mut ws.n,
            rows,
            e,
            self.eps,
        )?;
        if let Some(t) = taps {
            t.vit = exec.to_host_len(&ws.n, rows * e)?;
        }

        // projector: per-row pre-norm first (LN commutes with the merge - it
        // normalizes each 1152 sub-vector), then the free [rows/4, 4e] view.
        // Consecutive merged-order rows are the reference's (p1, p2) group.
        exec.layernorm(
            &ws.n,
            &self.proj_norm_w,
            &self.proj_norm_b,
            &mut ws.x,
            rows,
            e,
            PROJ_EPS,
        )?;
        exec.convert_f32_f16(&ws.x, &mut ws.s16, n4 * self.mm1.dims[0])?;
        exec.matvec_batch_f16(&self.mm1, &ws.s16, &mut ws.m, n4)?;
        exec.bias_add(&mut ws.m, &self.mm1_b, n4, mid)?;
        exec.gelu_erf(&mut ws.m, n4 * mid)?; // transformers GELUActivation = erf
        exec.convert_f32_f16(&ws.m, &mut ws.s16, n4 * mid)?;
        exec.matvec_batch_f16(&self.mm2, &ws.s16, &mut ws.out, n4)?;
        exec.bias_add(&mut ws.out, &self.mm2_b, n4, out_dim)?;

        // per-image output planes stay per-encode allocs: they outlive this
        // call (the encode queue holds them until the splice consumes them)
        let per = (n / 4) * out_dim;
        let mut outs = Vec::with_capacity(b);
        for bi in 0..b {
            let mut embd = exec.alloc(per)?;
            exec.copy_region(&ws.out, bi * per, &mut embd, 0, per)?;
            outs.push(VisionOutput {
                embd,
                nx: pw / 2,
                ny: ph / 2,
            });
        }
        Ok(outs)
    }
}
