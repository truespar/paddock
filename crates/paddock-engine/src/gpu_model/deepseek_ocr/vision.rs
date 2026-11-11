//! DeepEncoder - the DeepSeek-OCR family's vision tower, and the reason this
//! family compresses a page so hard. Two pretrained towers run in SERIES rather
//! than in parallel, with a 16x convolutional squeeze between them:
//!
//!   pixels @1024 -> SAM ViT-B (patch 16, 64x64 grid, windowed attention with
//!                  decomposed rel-pos, global only at 4 of 12 blocks)
//!                -> neck (1x1 conv 768->256, LN2d, 3x3 conv, LN2d)
//!                -> net_2 (3x3 s2, 256->512) -> net_3 (3x3 s2, 512->1024)   [16x]
//!                -> CLIP-L consumes that as its patch embeddings (+CLS, +pos)
//!                -> concat(clip[:, 1:], sam_net3_flat) = 2048
//!                -> linear projector -> 1280 = the decoder's width
//!
//! So a 1024px view costs 256 vision tokens, not 4096: SAM sees the page at
//! full patch resolution, and CLIP only ever sees the compressed 16x16 grid.
//!
//! Reference: the checkpoint's own `deepencoder.py` (`ImageEncoderViT`,
//! `VitModel`, `MlpProjector`) and `modeling_unlimitedocr.py`, cross-read
//! against vLLM's `model_executor/models/deepseek_ocr.py`.
//!
//! ## What the mmproj file says that is not true of this tower
//!
//! The GGUF is written by llama.cpp's generic `clip` converter, which has one
//! slot per concept and two towers to describe. Four of its keys are actively
//! wrong here, and every one of them is silent when believed:
//!
//! * `clip.vision.feed_forward_length = 64`. The real CLIP FFN is **4096**
//!   (`vit_model_cfg.ffn_hidden_size`); 64 is not a width this file contains.
//!   We take it from `v.blk.0.ffn_up.weight` instead.
//! * `clip.vision.patch_size = 14` and `clip.vision.image_size = 224`. Those
//!   are CLIP-L/14's nominal geometry, and CLIP never sees pixels here - SAM's
//!   patch 16 at 1024px is what the pipeline runs on. Reading either for token
//!   accounting produces a plausible, wrong number.
//! * `clip.vision.attention.layer_norm_epsilon = 1e-6`. That is **SAM's** eps
//!   (`norm_layer=partial(nn.LayerNorm, eps=1e-6)` in `_build_sam`, and
//!   `LayerNorm2d`'s default). CLIP's is 1e-5 for both its block norms and its
//!   pre-norm. One key, two towers - so [`CLIP_EPS`] is a constant here.
//!
//! And one tensor is dead weight: `v.patch_embd.weight` [14,14,3,1024] is
//! CLIP's own conv stem, which `CLIPVisionEmbeddings.forward` only reaches on
//! its `patch_embeds is None` branch. Every call site in the family passes
//! SAM's net_3 output (`vision_model(img, sam_out)`), so the branch is never
//! taken and the conv is never used. We do not upload it - that is 1.2 MB of
//! VRAM and, more importantly, a graph the reader would otherwise expect.
//!
//! ## Two activations, two epsilons, two attentions
//!
//! Nothing about this tower is uniform across its halves, and the mmproj's
//! single-slot keys cannot say so:
//!
//! | | SAM half | CLIP half |
//! |---|---|---|
//! | FFN activation | exact-erf GELU (`nn.GELU()`) | **quick-GELU** (`x·σ(1.702x)`) |
//! | LayerNorm eps | 1e-6 | 1e-5 |
//! | attention | windowed 14x14 + decomposed rel-pos, global at 4 blocks | plain full attention |
//!
//! `clip.use_gelu = true` is the file's whole vocabulary for that first row.

use std::sync::Arc;

use cudarc::driver::CudaSlice;
use paddock_models::gguf::Value;
use paddock_models::mapped::MappedGguf;

use crate::gpu::{GpuError, GpuExecutor, HalfTensor};

/// The `clip.projector_type` this loader answers to. llama.cpp writes the same
/// string for all three family members even though DeepSeek-OCR-2 swaps CLIP-L
/// for a qwen2-0.5b tower - so it identifies the FAMILY, and the second tower
/// has to be told apart by its tensors (see [`DeepEncoder::load`]).
pub const PROJECTOR_TYPE: &str = "deepseekocr";

/// Side of the whole-page view, in pixels. `config.json`'s
/// `candidate_resolutions: [[1024, 1024]]` and `infer(base_size=1024)`.
pub const BASE_PX: usize = 1024;
/// Side of one crop in gundam mode. `dynamic_preprocess(..., image_size=640)`
/// and `infer(image_size=640)`. Confirmed by parity: only 1024/640
/// reproduces the reference's 907-token prompt for the A4 test page.
pub const TILE_PX: usize = 640;

/// CLIP-L's LayerNorm epsilon - `vit_model_cfg.layernorm_epsilon` and
/// `pre_layernorm_epsilon`, both 1e-5. Not the file's single eps key, which
/// carries SAM's 1e-6. See the module header.
pub const CLIP_EPS: f32 = 1e-5;

/// Channels net_3 emits, and therefore the SAM half of the projector's input.
/// Asserted against the tensor rather than assumed.
const SAM_OUT_CH: usize = 1024;

/// SAM's window side. Read from the file (`clip.vision.window_size`), but the
/// rel-pos tables are what actually pin it: a windowed block's table has
/// `2*window - 1` rows, so a disagreement is caught at load.
fn rel_rows_for(q_side: usize) -> usize {
    2 * q_side - 1
}

/// Geometry of both towers plus the projector, after the file's wrong keys have
/// been replaced by what its tensors actually say.
// No `Eq`: image_mean/image_std/sam_eps are f32.
#[derive(Debug, Clone, PartialEq)]
pub struct VisionHparams {
    // --- SAM ViT-B ---
    pub sam_layers: usize,
    pub sam_embd: usize,
    pub sam_heads: usize,
    pub sam_head_dim: usize,
    pub sam_ff: usize,
    /// Conv-stem patch, 16. From `v.sam.patch_embd.weight`, never from
    /// `clip.vision.patch_size` (which is CLIP-L/14's 14).
    pub sam_patch: usize,
    /// Side of the pos-embed table's grid, 64 - the grid a 1024px view lands
    /// on. Other view sizes resample it.
    pub sam_grid_train: usize,
    /// Windowed-attention side, 14.
    pub window: usize,
    /// Blocks that attend globally, derived from which rel-pos tables are wider
    /// than `2*window - 1` rather than from a hardcoded `[2, 5, 8, 11]`. If a
    /// sibling checkpoint moves them, this follows.
    pub global_blocks: Vec<usize>,

    // --- CLIP-L ---
    pub clip_layers: usize,
    pub clip_embd: usize,
    pub clip_heads: usize,
    pub clip_head_dim: usize,
    /// 4096, from `v.blk.0.ffn_up.weight` - see the header for why not the key.
    pub clip_ff: usize,
    /// Rows in the position table: 257 = 1 CLS + a 16x16 grid. That 16 is
    /// exactly `BASE_PX / sam_patch / 4`, i.e. what net_2 and net_3 leave.
    pub clip_positions: usize,

    // --- projector ---
    /// 2048 = clip_embd + SAM_OUT_CH, the concat the projector eats.
    pub proj_in: usize,
    /// 1280 - the decoder's width.
    pub llm_embd: usize,

    // --- preprocessing ---
    pub min_tiles: usize,
    pub max_tiles: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    /// SAM's LayerNorm eps (1e-6), the one value the file's eps key gets right.
    pub sam_eps: f32,
}

impl VisionHparams {
    /// Token side of one view of `px` pixels: `ceil((px / patch) / 4)`.
    /// Mirrors [`super::tiling::num_queries`], which is the host-side copy that
    /// runs before the tower is loaded.
    pub fn tokens_per_side(&self, px: usize) -> usize {
        px.div_ceil(self.sam_patch).div_ceil(4)
    }

    /// SAM's patch grid side for a view of `px` pixels - 64 at 1024, 40 at 640.
    pub fn sam_grid(&self, px: usize) -> usize {
        px.div_ceil(self.sam_patch)
    }

    /// Is this SAM block global (full attention over the whole grid) rather than
    /// windowed?
    pub fn is_global(&self, block: usize) -> bool {
        self.global_blocks.contains(&block)
    }
}

/// One SAM block. `pre_ln`/`post_ln` are llama.cpp's names for the reference's
/// `norm1`/`norm2` - both are PRE-norms (norm1 before attention, norm2 before
/// the MLP), so "post" here means "second", not "after the residual".
pub(super) struct SamBlock {
    pub(super) pre_ln_w: CudaSlice<f32>,
    pub(super) pre_ln_b: CudaSlice<f32>,
    /// Fused q|k|v, `[embd, 3*embd]` - kept fused: the graph runs one GEMM and
    /// `row_slice`s the landing, exactly the reference's `qkv` Linear.
    pub(super) qkv_w: HalfTensor,
    pub(super) qkv_b: CudaSlice<f32>,
    pub(super) out_w: HalfTensor,
    pub(super) out_b: CudaSlice<f32>,
    /// Decomposed relative-position tables, `[rel_rows, head_dim]` on host.
    ///
    /// They stay on the host because they are RESAMPLED as a function of the
    /// view: `get_rel_pos` interpolates (mode="linear") whenever the table's
    /// row count differs from `2*max(q,k) - 1`. A windowed block never needs it
    /// (q is always 14 -> 27 rows, exact); a global block needs it for every view
    /// that is not 1024px - a 640px crop puts the grid at 40, wanting 79 rows
    /// from a 127-row table. The graph builds and caches per-geometry device
    /// tables; keeping the source here means it can.
    pub(super) rel_pos_h: Vec<f32>,
    pub(super) rel_pos_w: Vec<f32>,
    pub(super) rel_rows: usize,
    pub(super) post_ln_w: CudaSlice<f32>,
    pub(super) post_ln_b: CudaSlice<f32>,
    pub(super) lin1_w: HalfTensor,
    pub(super) lin1_b: CudaSlice<f32>,
    pub(super) lin2_w: HalfTensor,
    pub(super) lin2_b: CudaSlice<f32>,
}

/// One CLIP-L block: pre-norm attention, pre-norm MLP, both residual.
pub(super) struct ClipBlock {
    pub(super) ln1_w: CudaSlice<f32>,
    pub(super) ln1_b: CudaSlice<f32>,
    pub(super) qkv_w: HalfTensor,
    pub(super) qkv_b: CudaSlice<f32>,
    pub(super) out_w: HalfTensor,
    pub(super) out_b: CudaSlice<f32>,
    pub(super) ln2_w: CudaSlice<f32>,
    pub(super) ln2_b: CudaSlice<f32>,
    pub(super) up_w: HalfTensor,
    pub(super) up_b: CudaSlice<f32>,
    pub(super) down_w: HalfTensor,
    pub(super) down_b: CudaSlice<f32>,
}

pub struct DeepEncoder {
    pub(super) exec: Arc<GpuExecutor>,
    pub hp: VisionHparams,

    // --- SAM half ---
    /// `[patch*patch*3, embd]` after re-labelling: the GGUF stores the conv as
    /// [kx, ky, c, oc] with kx fastest, which read as rows-per-output-channel is
    /// already the im2row order `c*patch² + ky*patch + kx`. Same trick as the
    /// muse tower - a relabel, not a permute.
    pub(super) sam_patch_w: HalfTensor,
    pub(super) sam_patch_b: CudaSlice<f32>,
    /// Absolute pos table on host, `[grid*grid, embd]` row-major. Resampled per
    /// view size (bicubic, antialias, align_corners=false - `get_abs_pos_sam`),
    /// so like the rel-pos tables it lives here rather than on the device.
    pub(super) sam_pos: Vec<f32>,
    pub(super) sam_blocks: Vec<SamBlock>,
    /// neck.0: 1x1 conv 768->256, no bias. Relabelled `[768, 256]` - a 1x1 conv
    /// is a matmul over channels.
    pub(super) neck0_w: HalfTensor,
    pub(super) neck1_w: CudaSlice<f32>,
    pub(super) neck1_b: CudaSlice<f32>,
    /// neck.2: 3x3 conv 256->256, pad 1, no bias - PERMUTED at load into the
    /// tap-major GEMM plane `[9*cin, cout]` the im2row gather feeds (row index
    /// `(ky*3 + kx)*cin + c`; the file stores `c*9 + ky*3 + kx` per output).
    /// 2D twin of the whisper loader's conv1d permute.
    pub(super) neck2_w: HalfTensor,
    pub(super) neck3_w: CudaSlice<f32>,
    pub(super) neck3_b: CudaSlice<f32>,
    /// The 16x squeeze: two stride-2 3x3 convs, no bias - tap-major like neck2.
    pub(super) net_2: HalfTensor,
    pub(super) net_3: HalfTensor,

    // --- CLIP half ---
    pub(super) class_embd: CudaSlice<f32>,
    /// `[positions, embd]` on host: resampled (bicubic + antialias, CLS held
    /// out) whenever the grid is not 16x16 - `get_abs_pos`.
    pub(super) clip_pos: Vec<f32>,
    pub(super) pre_ln_w: CudaSlice<f32>,
    pub(super) pre_ln_b: CudaSlice<f32>,
    pub(super) clip_blocks: Vec<ClipBlock>,

    // --- projector and the two learned separators ---
    /// The linear projector `[2048, 1280]`, SPLIT at load into its two input
    /// halves: `fc(concat(clip, sam)) = clip·fc_clip + sam·fc_sam + b`, so the
    /// concat never has to materialize - the two sources stay in their own
    /// buffers and the projector is two GEMMs and an add.
    pub(super) fc_clip: HalfTensor,
    pub(super) fc_sam: HalfTensor,
    pub(super) fc_b: CudaSlice<f32>,
    /// Appended after every ROW of a view's token grid.
    pub(super) image_newline: CudaSlice<f32>,
    /// Appended once per image, at the very end - after the global view, not
    /// between the two views. (Spelling is the checkpoint's.)
    pub(super) view_separator: CudaSlice<f32>,

    /// Persistent activation slabs + geometry caches for the forward graph -
    /// allocated once here so the mid-serve encode path never touches
    /// cudaMalloc (see the `encode` module doc).
    pub(super) ws: super::encode::TowerWs,
}

impl DeepEncoder {
    pub fn load(exec: Arc<GpuExecutor>, map: &MappedGguf) -> Result<Self, GpuError> {
        let meta = |k: &str| map.gguf().metadata.get(k);
        let u = |k: &str| -> Result<usize, GpuError> {
            meta(k)
                .and_then(Value::as_u64)
                .map(|v| v as usize)
                .ok_or_else(|| GpuError::Driver(format!("ocr mmproj missing {k}")))
        };

        if let Some(t) = meta("clip.projector_type").and_then(Value::as_str)
            && t != PROJECTOR_TYPE
        {
            return Err(GpuError::Driver(format!(
                "ocr mmproj projector_type is {t:?}, expected {PROJECTOR_TYPE:?}"
            )));
        }

        let e = exec.clone();
        let dt = move |name: &str| -> Result<HalfTensor, GpuError> { e.upload_f16(map, name) };
        let e = exec.clone();
        let vf = move |name: &str| -> Result<CudaSlice<f32>, GpuError> {
            let (host, _) = host_f32(map, name)?;
            e.to_device(&host)
        };

        // ---- SAM geometry, taken from tensors wherever the keys are unreliable.
        let sam_layers = u("clip.vision.sam.block_count")?;
        let sam_embd = u("clip.vision.sam.embedding_length")?;
        let sam_heads = u("clip.vision.sam.head_count")?;
        if sam_heads == 0 || sam_embd % sam_heads != 0 {
            return Err(GpuError::Driver(format!(
                "sam embd {sam_embd} is not divisible by {sam_heads} heads"
            )));
        }
        let sam_head_dim = sam_embd / sam_heads;
        let window = u("clip.vision.window_size")?;
        if window < 2 {
            return Err(GpuError::Driver(format!(
                "sam window_size {window} is degenerate"
            )));
        }

        let mut sam_patch_w = dt("v.sam.patch_embd.weight")?;
        // [kx, ky, c, oc] - square kernel, 3 channels in.
        let pd = sam_patch_w.dims.clone();
        if pd.len() != 4 || pd[0] != pd[1] || pd[2] != 3 || pd[3] != sam_embd {
            return Err(GpuError::Driver(format!(
                "v.sam.patch_embd.weight dims {pd:?} are not [k, k, 3, {sam_embd}]"
            )));
        }
        let sam_patch = pd[0];
        sam_patch_w.dims = vec![sam_patch * sam_patch * 3, sam_embd];

        // pos table is [embd, grid, grid, 1] in GGUF order (ne[0] fastest).
        let (sam_pos, sam_pos_dims) = host_f32(map, "v.sam.pos_embd.weight")?;
        if sam_pos_dims.first() != Some(&sam_embd) || sam_pos_dims.get(1) != sam_pos_dims.get(2) {
            return Err(GpuError::Driver(format!(
                "v.sam.pos_embd.weight dims {sam_pos_dims:?} are not [{sam_embd}, g, g, 1]"
            )));
        }
        let sam_grid_train = sam_pos_dims[1];
        if sam_grid_train * sam_patch != BASE_PX {
            return Err(GpuError::Driver(format!(
                "sam pos grid {sam_grid_train} x patch {sam_patch} is not the {BASE_PX}px view"
            )));
        }

        let mut sam_blocks = Vec::with_capacity(sam_layers);
        let mut global_blocks = Vec::new();
        let mut sam_ff = 0usize;
        for i in 0..sam_layers {
            let t = |s: &str| format!("v.sam.blk.{i}.{s}");
            let (rel_pos_h, rh_dims) = host_f32(map, &t("attn.pos_h.weight"))?;
            let (rel_pos_w, rw_dims) = host_f32(map, &t("attn.pos_w.weight"))?;
            // GGUF [head_dim, rows] - PyTorch's (rows, head_dim) reversed.
            if rh_dims != rw_dims || rh_dims.first() != Some(&sam_head_dim) {
                return Err(GpuError::Driver(format!(
                    "sam block {i} rel-pos dims {rh_dims:?}/{rw_dims:?} are not [{sam_head_dim}, rows]"
                )));
            }
            let rel_rows = rh_dims[1];
            // The table width is the discriminator: a windowed block's queries
            // are always `window` apart at most, a global block's span the grid.
            if rel_rows != rel_rows_for(window) {
                if rel_rows != rel_rows_for(sam_grid_train) {
                    return Err(GpuError::Driver(format!(
                        "sam block {i} rel-pos has {rel_rows} rows, neither {} (window {window}) \
                         nor {} (grid {sam_grid_train})",
                        rel_rows_for(window),
                        rel_rows_for(sam_grid_train)
                    )));
                }
                global_blocks.push(i);
            }

            let lin1_w = dt(&t("mlp.lin1.weight"))?;
            let ff = lin1_w.dims[1];
            if i == 0 {
                sam_ff = ff;
            } else if ff != sam_ff {
                return Err(GpuError::Driver(format!(
                    "sam block {i} ffn width {ff} != block 0's {sam_ff}"
                )));
            }

            sam_blocks.push(SamBlock {
                pre_ln_w: vf(&t("pre_ln.weight"))?,
                pre_ln_b: vf(&t("pre_ln.bias"))?,
                qkv_w: dt(&t("attn.qkv.weight"))?,
                qkv_b: vf(&t("attn.qkv.bias"))?,
                out_w: dt(&t("attn.out.weight"))?,
                out_b: vf(&t("attn.out.bias"))?,
                rel_pos_h,
                rel_pos_w,
                rel_rows,
                post_ln_w: vf(&t("post_ln.weight"))?,
                post_ln_b: vf(&t("post_ln.bias"))?,
                lin1_w,
                lin1_b: vf(&t("mlp.lin1.bias"))?,
                lin2_w: dt(&t("mlp.lin2.weight"))?,
                lin2_b: vf(&t("mlp.lin2.bias"))?,
            });
        }
        if global_blocks.is_empty() {
            return Err(GpuError::Driver(
                "no sam block attends globally - every rel-pos table is window-sized".into(),
            ));
        }

        // neck.0 is 1x1: [1, 1, in, out] collapses to a plain [in, out] matmul.
        let mut neck0_w = dt("v.sam.neck.0.weight")?;
        let n0 = neck0_w.dims.clone();
        if n0.len() != 4 || n0[0] != 1 || n0[1] != 1 || n0[2] != sam_embd {
            return Err(GpuError::Driver(format!(
                "v.sam.neck.0.weight dims {n0:?} are not [1, 1, {sam_embd}, out]"
            )));
        }
        let neck_ch = n0[3];
        neck0_w.dims = vec![sam_embd, neck_ch];

        // 3x3 convs become GEMMs over gathered 9-tap rows, and the file's
        // layout is already the right one - a relabel, not a permute. The
        // gather is `pixel_shuffle_rows`, which is CHANNEL-outer
        // (`out[c*k + s] = src[idx[s]][c]`), so a gathered row reads
        // `c*9 + tap`; the GGUF stores each output channel's weights as
        // `c*9 + ky*3 + kx` - identical, with taps ordered (ky, kx) row-major.
        //
        // Recorded because it was learned the expensive way: the first cut
        // permuted these planes tap-major (the whisper conv1d precedent, whose
        // gather is tap-outer), and the stage oracle flagged the neck as the
        // first diverging stage - a double permutation, weights one way and
        // data the other.
        let exec3 = exec.clone();
        let conv3x3 = move |name: &str, cin: usize| -> Result<HalfTensor, GpuError> {
            let mut w = exec3.upload_f16(map, name)?;
            if w.dims.len() != 4 || w.dims[0] != 3 || w.dims[1] != 3 || w.dims[2] != cin {
                return Err(GpuError::Driver(format!(
                    "{name} dims {:?} are not [3, 3, {cin}, out]",
                    w.dims
                )));
            }
            w.dims = vec![9 * cin, w.dims[3]];
            Ok(w)
        };

        let neck2_w = conv3x3("v.sam.neck.2.weight", neck_ch)?;
        let net_2 = conv3x3("v.sam.net_2.weight", neck_ch)?;
        let mid_ch = net_2.dims[1];
        let net_3 = conv3x3("v.sam.net_3.weight", mid_ch)?;
        if net_3.dims[1] != SAM_OUT_CH {
            return Err(GpuError::Driver(format!(
                "v.sam.net_3.weight emits {} channels, expected {SAM_OUT_CH}",
                net_3.dims[1]
            )));
        }

        // ---- CLIP-L.
        let clip_layers = u("clip.vision.block_count")?;
        let clip_embd = u("clip.vision.embedding_length")?;
        let clip_heads = u("clip.vision.attention.head_count")?;
        if clip_heads == 0 || clip_embd % clip_heads != 0 {
            return Err(GpuError::Driver(format!(
                "clip embd {clip_embd} is not divisible by {clip_heads} heads"
            )));
        }
        let clip_head_dim = clip_embd / clip_heads;

        let (clip_pos, clip_pos_dims) = host_f32(map, "v.position_embd.weight")?;
        if clip_pos_dims.first() != Some(&clip_embd) || clip_pos_dims.len() < 2 {
            return Err(GpuError::Driver(format!(
                "v.position_embd.weight dims {clip_pos_dims:?} are not [{clip_embd}, positions]"
            )));
        }
        let clip_positions = clip_pos_dims[1];
        // 1 CLS + a square grid, and that grid must be what the 16x squeeze
        // leaves of the base view - otherwise CLIP is being fed the wrong stage.
        let grid = clip_positions - 1;
        let side = (grid as f64).sqrt().round() as usize;
        if side * side != grid || side != sam_grid_train / 4 {
            return Err(GpuError::Driver(format!(
                "clip has {clip_positions} positions (grid {grid}), expected 1 + ({}x{})",
                sam_grid_train / 4,
                sam_grid_train / 4
            )));
        }

        let mut clip_blocks = Vec::with_capacity(clip_layers);
        let mut clip_ff = 0usize;
        for i in 0..clip_layers {
            let t = |s: &str| format!("v.blk.{i}.{s}");
            let up_w = dt(&t("ffn_up.weight"))?;
            let ff = up_w.dims[1];
            if i == 0 {
                clip_ff = ff;
            } else if ff != clip_ff {
                return Err(GpuError::Driver(format!(
                    "clip block {i} ffn width {ff} != block 0's {clip_ff}"
                )));
            }
            clip_blocks.push(ClipBlock {
                ln1_w: vf(&t("ln1.weight"))?,
                ln1_b: vf(&t("ln1.bias"))?,
                qkv_w: dt(&t("attn_qkv.weight"))?,
                qkv_b: vf(&t("attn_qkv.bias"))?,
                out_w: dt(&t("attn_out.weight"))?,
                out_b: vf(&t("attn_out.bias"))?,
                ln2_w: vf(&t("ln2.weight"))?,
                ln2_b: vf(&t("ln2.bias"))?,
                up_w,
                up_b: vf(&t("ffn_up.bias"))?,
                down_w: dt(&t("ffn_down.weight"))?,
                down_b: vf(&t("ffn_down.bias"))?,
            });
        }
        // The file says 64 here. Refuse to agree with it - see the header.
        if let Some(k) = meta("clip.vision.feed_forward_length").and_then(Value::as_u64)
            && k as usize != clip_ff
        {
            tracing::debug!(
                key = k,
                actual = clip_ff,
                "ocr mmproj feed_forward_length disagrees with the weights; using the weights"
            );
        }

        // ---- projector, split into its clip/sam input halves at load:
        // fc(concat(a, b)) = a·W[..1024] + b·W[1024..] + bias, so the concat
        // never materializes. Column split - strided on host, done once.
        let (fc_host, fc_dims) = host_f32(map, "mm.model.fc.weight")?;
        if fc_dims.len() != 2 {
            return Err(GpuError::Driver(format!(
                "mm.model.fc.weight dims {fc_dims:?} are not 2-D"
            )));
        }
        let (proj_in, llm_embd) = (fc_dims[0], fc_dims[1]);
        // The projector eats concat(clip hidden without CLS, sam net_3 flat).
        // If this ever fails the second tower is not CLIP-L - DeepSeek-OCR-2
        // swaps it for qwen2-0.5b and lands at 896 in, not 2048.
        if proj_in != clip_embd + SAM_OUT_CH {
            return Err(GpuError::Driver(format!(
                "projector takes {proj_in} inputs, but clip {clip_embd} + sam {SAM_OUT_CH} \
                 is {}: this mmproj is not the CLIP-L member of the family",
                clip_embd + SAM_OUT_CH
            )));
        }
        if let Some(d) = meta("clip.vision.projection_dim").and_then(Value::as_u64)
            && d as usize != llm_embd
        {
            return Err(GpuError::Driver(format!(
                "projection_dim {d} != the projector's output width {llm_embd}"
            )));
        }
        let (mut clip_half, mut sam_half) = (
            vec![0f32; clip_embd * llm_embd],
            vec![0f32; SAM_OUT_CH * llm_embd],
        );
        for o in 0..llm_embd {
            let row = &fc_host[o * proj_in..][..proj_in];
            clip_half[o * clip_embd..][..clip_embd].copy_from_slice(&row[..clip_embd]);
            sam_half[o * SAM_OUT_CH..][..SAM_OUT_CH].copy_from_slice(&row[clip_embd..]);
        }
        let fc_clip = HalfTensor {
            buf: exec.to_device_f16(&clip_half, "mm.model.fc.weight[clip]")?,
            dims: vec![clip_embd, llm_embd],
        };
        let fc_sam = HalfTensor {
            buf: exec.to_device_f16(&sam_half, "mm.model.fc.weight[sam]")?,
            dims: vec![SAM_OUT_CH, llm_embd],
        };

        let arr3 = |k: &str, dflt: f32| -> [f32; 3] {
            match meta(k) {
                Some(Value::Array(a)) => {
                    let v: Vec<f32> = a.iter().filter_map(Value::as_f32).collect();
                    if v.len() == 3 {
                        [v[0], v[1], v[2]]
                    } else {
                        [dflt; 3]
                    }
                }
                _ => [dflt; 3],
            }
        };

        let hp = VisionHparams {
            sam_layers,
            sam_embd,
            sam_heads,
            sam_head_dim,
            sam_ff,
            sam_patch,
            sam_grid_train,
            window,
            global_blocks,
            clip_layers,
            clip_embd,
            clip_heads,
            clip_head_dim,
            clip_ff,
            clip_positions,
            proj_in,
            llm_embd,
            min_tiles: u("clip.vision.preproc_min_tiles").unwrap_or(2),
            max_tiles: u("clip.vision.preproc_max_tiles").unwrap_or(6),
            image_mean: arr3("clip.vision.image_mean", 0.5),
            image_std: arr3("clip.vision.image_std", 0.5),
            sam_eps: meta("clip.vision.attention.layer_norm_epsilon")
                .and_then(Value::as_f32)
                .unwrap_or(1e-6),
        };

        // Workspace before the struct literal moves the resample sources in:
        // slabs at the max geometry, every canonical cache warmed, so serving
        // never allocates here again.
        let ws = super::encode::TowerWs::new(
            &exec,
            &hp,
            super::encode::TowerDims {
                e: hp.sam_embd,
                sam_ff: hp.sam_ff,
                patch_in: sam_patch_w.dims[0],
                neck_ch: neck0_w.dims[1],
                mid_ch: net_2.dims[1],
                sam_ch: net_3.dims[1],
                ce: hp.clip_embd,
                cff: hp.clip_ff,
                out_dim: hp.llm_embd,
            },
            &sam_pos,
            &clip_pos,
            &sam_blocks,
        )?;

        let me = Self {
            exec,
            hp,
            sam_patch_w,
            sam_patch_b: vf("v.sam.patch_embd.bias")?,
            sam_pos,
            sam_blocks,
            neck0_w,
            neck1_w: vf("v.sam.neck.1.weight")?,
            neck1_b: vf("v.sam.neck.1.bias")?,
            neck2_w,
            neck3_w: vf("v.sam.neck.3.weight")?,
            neck3_b: vf("v.sam.neck.3.bias")?,
            net_2,
            net_3,
            class_embd: vf("v.class_embd")?,
            clip_pos,
            pre_ln_w: vf("v.pre_ln.weight")?,
            pre_ln_b: vf("v.pre_ln.bias")?,
            clip_blocks,
            fc_clip,
            fc_sam,
            fc_b: vf("mm.model.fc.bias")?,
            image_newline: vf("v.image_newline")?,
            view_separator: vf("v.view_seperator")?,
            ws,
        };

        tracing::info!(
            weight_mib = me.weight_bytes() / (1 << 20),
            workspace_mib = me.ws.device_bytes() / (1 << 20),
            sam_layers = me.hp.sam_layers,
            clip_layers = me.hp.clip_layers,
            global_blocks = ?me.hp.global_blocks,
            window = me.hp.window,
            tiles = me.hp.max_tiles,
            "deepencoder mmproj resident at f16 (f32 accumulate)"
        );
        Ok(me)
    }

    /// Device bytes this tower holds - what the fit estimate prices.
    ///
    /// Counts the f32 norm/bias planes as well as the f16 weight planes. They
    /// are only ~1.8 MB against 755 of weights, but they are resident, and a
    /// fit estimate that quietly omits resident bytes is the thing the "no
    /// silent failures" principle is about. The pos/rel-pos tables are the one
    /// deliberate exclusion: they live on the host and only per-geometry
    /// resamples of them reach the device - see [`Self::host_table_bytes`].
    pub fn weight_bytes(&self) -> usize {
        let f32b = |s: &CudaSlice<f32>| s.len() * 4;
        let sam: usize = self
            .sam_blocks
            .iter()
            .map(|b| {
                b.qkv_w.bytes()
                    + b.out_w.bytes()
                    + b.lin1_w.bytes()
                    + b.lin2_w.bytes()
                    + f32b(&b.pre_ln_w)
                    + f32b(&b.pre_ln_b)
                    + f32b(&b.post_ln_w)
                    + f32b(&b.post_ln_b)
                    + f32b(&b.qkv_b)
                    + f32b(&b.out_b)
                    + f32b(&b.lin1_b)
                    + f32b(&b.lin2_b)
            })
            .sum();
        let clip: usize = self
            .clip_blocks
            .iter()
            .map(|b| {
                b.qkv_w.bytes()
                    + b.out_w.bytes()
                    + b.up_w.bytes()
                    + b.down_w.bytes()
                    + f32b(&b.ln1_w)
                    + f32b(&b.ln1_b)
                    + f32b(&b.ln2_w)
                    + f32b(&b.ln2_b)
                    + f32b(&b.qkv_b)
                    + f32b(&b.out_b)
                    + f32b(&b.up_b)
                    + f32b(&b.down_b)
            })
            .sum();
        sam + clip
            + self.sam_patch_w.bytes()
            + f32b(&self.sam_patch_b)
            + self.neck0_w.bytes()
            + f32b(&self.neck1_w)
            + f32b(&self.neck1_b)
            + self.neck2_w.bytes()
            + f32b(&self.neck3_w)
            + f32b(&self.neck3_b)
            + self.net_2.bytes()
            + self.net_3.bytes()
            + f32b(&self.class_embd)
            + f32b(&self.pre_ln_w)
            + f32b(&self.pre_ln_b)
            + self.fc_clip.bytes()
            + self.fc_sam.bytes()
            + f32b(&self.fc_b)
            + f32b(&self.image_newline)
            + f32b(&self.view_separator)
    }

    /// Device bytes the persistent forward workspace pins (on top of
    /// [`Self::weight_bytes`]) - resident for the tower's whole life, so a
    /// fit estimate must price it too.
    pub fn workspace_bytes(&self) -> usize {
        self.ws.device_bytes()
    }

    /// Decoder width the projector emits into.
    pub fn llm_embd(&self) -> usize {
        self.hp.llm_embd
    }

    /// The executor this tower runs on - for tests that need to read an
    /// [`super::encode::EncodedViews`] buffer back.
    pub fn exec_ref(&self) -> &GpuExecutor {
        &self.exec
    }

    /// Run the tower's seams (bias, activation, residual, the q|k|v split) as
    /// their own passes instead of in the GEMM landings - the composed
    /// reference the fused tower must match bit for bit, kept for that gate.
    pub fn set_composed_seams(&mut self, on: bool) {
        self.ws.composed = on;
    }

    /// Host bytes the resample sources hold. Small, but they are the reason
    /// `weight_bytes` and the file size do not match, so they are reportable.
    pub fn host_table_bytes(&self) -> usize {
        let rel: usize = self
            .sam_blocks
            .iter()
            .map(|b| (b.rel_pos_h.len() + b.rel_pos_w.len()) * 4)
            .sum();
        rel + self.sam_pos.len() * 4 + self.clip_pos.len() * 4
    }
}

/// Read a GGUF tensor (F32/F16/BF16) as host f32 - shared with the qwen35 and
/// muse towers.
fn host_f32(map: &MappedGguf, name: &str) -> Result<(Vec<f32>, Vec<usize>), GpuError> {
    crate::gpu_model::qwen35::vision::host_f32(map, name)
        .map_err(|e| GpuError::Driver(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rel-pos row count is what tells a windowed block from a global one,
    /// so pin both numbers this family produces.
    #[test]
    fn rel_pos_rows_discriminate_window_from_global() {
        assert_eq!(rel_rows_for(14), 27, "windowed blocks");
        assert_eq!(rel_rows_for(64), 127, "global blocks at the 1024px view");
        // A 640px crop puts the grid at 40, which wants 79 rows out of the
        // 127-row table - that resample is `get_rel_pos`'s linear branch.
        assert_eq!(rel_rows_for(40), 79);
    }

    fn hp() -> VisionHparams {
        VisionHparams {
            sam_layers: 12,
            sam_embd: 768,
            sam_heads: 12,
            sam_head_dim: 64,
            sam_ff: 3072,
            sam_patch: 16,
            sam_grid_train: 64,
            window: 14,
            global_blocks: vec![2, 5, 8, 11],
            clip_layers: 24,
            clip_embd: 1024,
            clip_heads: 16,
            clip_head_dim: 64,
            clip_ff: 4096,
            clip_positions: 257,
            proj_in: 2048,
            llm_embd: 1280,
            min_tiles: 2,
            max_tiles: 32,
            image_mean: [0.5; 3],
            image_std: [0.5; 3],
            sam_eps: 1e-6,
        }
    }

    /// The 16x squeeze, stated as the two grids it connects: SAM sees 64x64 and
    /// CLIP sees 16x16, which is why a whole page costs 256 tokens and not 4096.
    #[test]
    fn the_squeeze_lands_clip_on_the_position_table() {
        let h = hp();
        assert_eq!(h.sam_grid(BASE_PX), 64);
        assert_eq!(h.tokens_per_side(BASE_PX), 16);
        assert_eq!(h.clip_positions, 1 + h.tokens_per_side(BASE_PX).pow(2));

        // A crop is 640px: SAM 40x40, CLIP 10x10 - neither the pos table nor
        // the global blocks' rel-pos table fits, so both get resampled.
        assert_eq!(h.sam_grid(TILE_PX), 40);
        assert_eq!(h.tokens_per_side(TILE_PX), 10);
        assert_ne!(1 + h.tokens_per_side(TILE_PX).pow(2), h.clip_positions);
    }

    #[test]
    fn global_blocks_are_the_four_the_reference_builds() {
        let h = hp();
        for b in 0..h.sam_layers {
            assert_eq!(
                h.is_global(b),
                matches!(b, 2 | 5 | 8 | 11),
                "block {b} global-ness disagrees with _build_sam's [2, 5, 8, 11]"
            );
        }
    }

    /// The projector's width is the family discriminator: CLIP-L members concat
    /// 1024 + 1024, OCR-2 (qwen2-0.5b tower) lands at 896.
    #[test]
    fn projector_input_is_the_concat_width() {
        let h = hp();
        assert_eq!(h.proj_in, h.clip_embd + SAM_OUT_CH);
        assert_eq!(h.proj_in, 2048);
    }
}
