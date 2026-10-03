//! Clef on CUDA - Cloudflare's decision models (`Cloudflare/clef-flash`):
//! one joint sequence per request through a Qwen3.5 backbone, read by a
//! joint schema head that scores every option of every question in the
//! same pass. Nothing is generated; nothing is cached between requests.
//!
//! Precision: weights stay as stored (BF16, exact). The backbone's
//! projections split the F32 activation two ways in bf16 against them (slot
//! 733, 16 significant bits, two mma a k16); its attention is 3xTF32 on the
//! tensor cores (732, F32 class, K and V read as projected); the residual
//! stream, every norm, the DeltaNet decay and state, every softmax and the
//! whole head (on 722, the F32 class) are F32. Elected by the gate against
//! the reference's F32 evaluation: the vendor's own class (BF16
//! activations) drifted past the vendor's BF16 on one fixture; this holds
//! every fixture ~1000x inside it, zero decision flips. The rope reads a
//! table built once at load (`rope_table`), so no angle loses bits with
//! depth.
//!
//! The tensors are read straight from the safetensors shards through the
//! same transforms llama.cpp's converter applies, so the qwen35 lane's
//! DeltaNet kernels see the data they were qualified on: value heads
//! reordered from grouped to tiled order, `A_log` stored as `-exp(A_log)`,
//! conv1d squeezed, `+1` folded into every RMSNorm but the gated one.
//!
//! Or from a GGUF - ggml-org's conversions of the same release (`gguf.rs`),
//! which hold exactly those planes already. Their Q8_0 projections stay
//! Q8_0 on the device (slot 741: the same two-part split against the exact
//! int8, the file's f16 block scales at the drain), the embeddings Q8_0 rows
//! as stored; the vision tower, which the GGUF does not carry, comes from a
//! companion cut byte for byte from the official checkpoint.

mod forward;
mod gguf;
mod head;
mod load;
mod vision;

pub use head::{
    ClefImage, ClefLogits, ClefQuestion, ClefRequest, MAX_OPTIONS, MAX_QUESTIONS, MAX_REQUESTS,
};

use std::sync::Arc;

use cudarc::driver::CudaSlice;
use paddock_models::clef::ClefConfig;

use crate::gpu::{ClefQ8, GpuExecutor, QuantTensor};

pub use crate::gpu_model::gpt_oss::GpuModelError;

/// A backbone projection's weights as stored: the official checkpoint's
/// BF16 rows (slot 733) or a GGUF's Q8_0, repacked (slot 741).
enum Proj {
    Bf16(QuantTensor),
    Q8(ClefQ8),
}

/// Most rows of one pass (every request of a tick, back to back): the
/// reference's own `max_length`, so one request of any admissible length is
/// one pass.
pub const MAX_ROWS: usize = 16384;

/// One Gated DeltaNet mixer, value heads in tiled order.
struct Gdn {
    /// `[q | k | v]` rows, the v rows tiled
    qkv: Proj,
    /// output gate rows, tiled
    z: Proj,
    /// `[a | b]` rows, `[2 * v_heads][hidden]`, tiled
    ab: Proj,
    /// `[qkv_rows][conv]`, the v channels tiled
    conv: CudaSlice<f32>,
    /// `-exp(A_log)`, tiled
    ssm_a: CudaSlice<f32>,
    dt_bias: CudaSlice<f32>,
    /// the gated RMSNorm's weight (no +1)
    norm: CudaSlice<f32>,
    /// columns tiled
    out: Proj,
}

/// One gated full-attention mixer.
struct Attn {
    /// per head: query then output gate
    q: Proj,
    k: Proj,
    v: Proj,
    q_norm: CudaSlice<f32>,
    k_norm: CudaSlice<f32>,
    o: Proj,
}

enum Mixer {
    Gdn(Gdn),
    Attn(Attn),
}

struct Layer {
    in_norm: CudaSlice<f32>,
    post_norm: CudaSlice<f32>,
    mixer: Mixer,
    /// gate and up rows interleaved (2j gate j, 2j + 1 up j) for the
    /// GEMM's SwiGLU epilogue
    gate_up: Proj,
    down: Proj,
}

/// The resident pass planes, sized once for `max_rows`. With the vision
/// tower loaded three of them double as its planes (it runs before the
/// backbone in a pass, while they are idle): `wide` its residual stream,
/// `dq` its normed rows, `ffn_g` its wide plane; the merged image rows wait
/// in `xn` for the embedding to land in `x`, and `x` stages an image's
/// bytes while the processor resizes it (a decoded image is at most its
/// size: 268 MB at 16384 rows, ~89 MP).
struct Workspace {
    x: CudaSlice<f32>,
    xn: CudaSlice<f32>,
    /// GDN `[q|k|v]` / attention query-and-gate rows
    wide: CudaSlice<f32>,
    dq: CudaSlice<f32>,
    dk: CudaSlice<f32>,
    dv: CudaSlice<f32>,
    z: CudaSlice<f32>,
    ab: CudaSlice<f32>,
    g: CudaSlice<f32>,
    beta: CudaSlice<f32>,
    core: CudaSlice<f32>,
    out: CudaSlice<f32>,
    k: CudaSlice<f32>,
    kn: CudaSlice<f32>,
    v: CudaSlice<f32>,
    /// the MLP's activated rows
    ffn_g: CudaSlice<f32>,
    /// the DeltaNet state of the run being scanned, `[v_heads][128][128]`
    state: CudaSlice<f32>,
    /// chunked-scan scratch for the longest run
    dnc_dw: CudaSlice<f32>,
    dnc_du: CudaSlice<f32>,
    dnc_aqk: CudaSlice<f32>,
    dnc_cg: CudaSlice<f64>,
}

/// One loaded checkpoint.
pub struct GpuClef {
    exec: Arc<GpuExecutor>,
    pub cfg: ClefConfig,
    /// the input embedding, BF16 rows or Q8_0 rows as stored
    embed: QuantTensor,
    /// the output embedding, which the head reads rows of (`lm_head`), BF16
    /// rows or Q8_0 rows as stored
    pub(crate) lm_head: QuantTensor,
    final_norm: CudaSlice<f32>,
    /// the rope's `(cos, sin)` for every (position, pair), `[MAX_ROWS][32]`
    rope: CudaSlice<f32>,
    /// the pairs that read the h / w position axis (interleaved mrope)
    rope_masks: (u32, u32),
    layers: Vec<Layer>,
    /// the vision tower, when the checkpoint ships one and the pack carries
    /// the image lane
    vision: Option<vision::Vision>,
    head: head::HeadW,
    ws: Workspace,
    head_ws: head::HeadWs,
    max_rows: usize,
    weight_bytes: u64,
}

/// One pass: every request's ids back to back, and where each starts.
pub struct ClefPass<'a> {
    pub ids: &'a [u32],
    /// `(first row, rows)` per request, in order, covering `ids`
    pub runs: &'a [(usize, usize)],
    /// every image of the pass, in row order: its first pass row and its
    /// merged token grid (rows, columns); its rows already encoded into the
    /// workspace's `xn` back to back, image order
    pub images: &'a [PassImage],
}

/// One image's place in a pass (see [`ClefPass::images`]).
#[derive(Clone, Copy, Debug)]
pub struct PassImage {
    pub row: usize,
    pub grid: (usize, usize),
}

impl GpuClef {
    pub fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }

    pub fn max_rows(&self) -> usize {
        self.max_rows
    }

    /// Whether requests may carry images (the tower is loaded).
    pub fn has_vision(&self) -> bool {
        self.vision.is_some()
    }

    /// The image processor alone (diagnostics, the gate): `images`' pixel
    /// rows `[patches][1536]` as the tower reads them.
    pub fn image_pixels(&mut self, images: &[&ClefImage]) -> Result<Vec<f32>, GpuModelError> {
        let v = self.vision.as_mut().ok_or_else(|| {
            GpuModelError::Unsupported("Clef: the vision tower is not loaded".into())
        })?;
        let ws = &mut self.ws;
        let (info, _, _) = v.preprocess(
            &self.exec,
            images,
            &mut ws.ffn_g,
            vision::Stage {
                raw: &mut ws.x,
                mid: &mut ws.wide,
                fin: &mut ws.dq,
            },
        )?;
        Ok(self.exec.to_host_len(&ws.ffn_g, info.len() / 4 * 1536)?)
    }

    /// The tower alone (diagnostics, the gate): its last block's rows
    /// `[patches][hidden]` and the merger's `[tokens][out_hidden]`.
    pub fn encode_images(
        &mut self,
        images: &[&ClefImage],
    ) -> Result<(Vec<f32>, Vec<f32>), GpuModelError> {
        let v = self.vision.as_mut().ok_or_else(|| {
            GpuModelError::Unsupported("Clef: the vision tower is not loaded".into())
        })?;
        let ws = &mut self.ws;
        v.encode(
            &self.exec,
            images,
            &mut ws.wide,
            &mut ws.dq,
            &mut ws.ffn_g,
            &mut ws.xn,
            &mut ws.x,
        )?;
        let p: usize = images
            .iter()
            .map(|i| (i.resized.0 / 16) * (i.resized.1 / 16))
            .sum();
        Ok((
            self.exec.to_host_len(&ws.wide, p * v.cfg.hidden)?,
            self.exec.to_host_len(&ws.xn, p / 4 * v.cfg.out_hidden)?,
        ))
    }

    pub fn exec(&self) -> &Arc<GpuExecutor> {
        &self.exec
    }

    /// The resident pass planes, backbone and head.
    pub fn workspace_bytes(&self) -> u64 {
        Workspace::bytes(&self.cfg, self.max_rows, self.vision.is_some())
            + if self.vision.is_some() {
                vision::plan_bytes()
            } else {
                0
            }
            + 4 * head::HeadWs::floats(&self.cfg.head, self.cfg.hidden) as u64
    }
}
