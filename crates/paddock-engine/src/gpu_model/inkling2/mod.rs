//! Thinking Machines' Inkling 2 Mini (`inkling2_loop`) on CUDA. One lane
//! serves both the single stream and the batch: every call is a list of
//! per-slot row runs (a prompt chunk, a decode row each) through the 48
//! layer visits, over paged / ring K / V stores (kv.rs) in the served class
//! - fp8 e4m3 by default, f32 for the reference gates.
//!
//! The model (transformers' `inkling2_loop` and vLLM's `inkling2` are the
//! study references): 24 layers at width 2304, the middle twelve run three
//! times (48 layer visits a token, each with its own K / V and conv states);
//! block attention residuals over the hidden states at layers 0 / 6 / 12 /
//! 18; relative-bias attention with per-token sinks (no RoPE), 20 sliding
//! layers and 4 full; short convolutions on K, V and both residual
//! branches; 128 routed experts, top 8, NVFP4 W4A4 on layers 0-19 and MXFP8
//! on 20-23 (the checkpoint's declared classes); the loop's passes 2 and 3
//! merge messages from the previous pass through gated rank-64 controllers.
//! Everything the checkpoint ships in BF16 is served from its own bytes.

mod draft;
mod forward;
mod generate;
mod kv;
mod load;
mod mm;
mod prefill;
mod prefix;
mod spec;
pub mod vision;

pub use forward::Logits;

use cudarc::driver::CudaSlice;
use paddock_models::inkling2 as ik;

use crate::gpu::{GpuExecutor, Inkling2Act, Inkling2Experts, Inkling2Sorted, QuantTensor};
use crate::gpu_model::gpt_oss::GpuModelError;

/// One of the four loop controllers of a looped layer (q, k, v, mlp).
pub(crate) struct Controller {
    /// `[64][4608]` bf16, the gate's down projection.
    pub down: QuantTensor,
    /// `[2304][64]` bf16 bytes each: forget and message (gain) gates.
    pub forget: CudaSlice<u8>,
    pub gain: CudaSlice<u8>,
}

pub(crate) struct Layer {
    pub full: bool,
    /// q (3072) | relative-state projection (384) | sinks (24) rows over the
    /// query input; the output gate (3072) is its own plane (it lands
    /// contiguous for the elementwise gate).
    pub qrs: QuantTensor,
    pub gate: QuantTensor,
    /// k (1024) [| the dynamic conv's taps (32), full layers] over the key
    /// input; v likewise over the value input.
    pub k: QuantTensor,
    pub v: QuantTensor,
    pub o: QuantTensor,
    /// bf16 bytes: per-(head, dim) query scale [24][128], relative-state
    /// scale [24][16], the relative-bias bank [16][1024].
    pub q_scale: CudaSlice<u8>,
    pub rel_scale: CudaSlice<u8>,
    pub rel_bank: CudaSlice<u8>,
    /// The bank transposed, [1024][16] (the prompt-row attention's).
    pub rel_bank_t: CudaSlice<u8>,
    /// Conv taps, bf16 bytes: K / V [1024][4] (sliding) or [4][1024] by lag
    /// (full, dynamic); the residual branches' [2304][4].
    pub k_conv: CudaSlice<u8>,
    pub v_conv: CudaSlice<u8>,
    pub attn_conv: CudaSlice<u8>,
    pub mlp_conv: CudaSlice<u8>,
    /// RMSNorm weights (f32): sliding layers one input norm, full layers one
    /// each for q / k / v; then the MLP's.
    pub in_norm: Vec<CudaSlice<f32>>,
    pub mlp_norm: CudaSlice<f32>,
    /// Attention-residual queries (f32): the attention input's (none at
    /// layer 0; 3 rows on the full layers) and the MLP input's.
    pub attn_q: Option<CudaSlice<f32>>,
    pub mlp_q: CudaSlice<f32>,
    pub router: QuantTensor,
    pub router_bias: CudaSlice<f32>,
    pub experts: Inkling2Experts,
    /// q, k, v, mlp controllers (looped layers).
    pub ctrl: Option<[Controller; 4]>,
}

/// Per-chunk working planes for up to `rows` tokens.
pub(crate) struct Scratch {
    pub rows: usize,
    /// the gathered embedding rows, before their norm
    pub emb: CudaSlice<f32>,
    /// a non-boundary layer's residual (input + attention branch)
    pub res: CudaSlice<f32>,
    /// the head's input rows and logits, HEAD_ROWS at a time
    pub head_in: CudaSlice<f32>,
    pub head_out: CudaSlice<f32>,
    /// [rows][width] planes addressed by index: hidden states, block states,
    /// the prelude's and the previous pass's - a boundary keeps its buffer.
    pub pool: Vec<CudaSlice<f32>>,
    /// The attention input's up-to-3 rows of queries ([3][rows][width]).
    pub ares: CudaSlice<f32>,
    pub qin: CudaSlice<f32>,
    pub kin: CudaSlice<f32>,
    pub vin: CudaSlice<f32>,
    pub gtmp: CudaSlice<f32>,
    pub qrs: CudaSlice<f32>,
    pub gate: CudaSlice<f32>,
    pub kp: CudaSlice<f32>,
    pub vp: CudaSlice<f32>,
    pub att: CudaSlice<f32>,
    /// A dense plane's input narrowed to bf16 for the prefill pair (widest:
    /// the attention rows into o).
    pub x16: CudaSlice<half::bf16>,
    /// The attention's q / k / v inputs normed straight into bf16 rows.
    pub xin16: [CudaSlice<half::bf16>; 3],
    /// Device picks: the rows' sampling plans and the ids drawn.
    pub pick_par: CudaSlice<u32>,
    pub pick_ids: CudaSlice<u32>,
    pub o: CudaSlice<f32>,
    pub ac: CudaSlice<f32>,
    pub min: CudaSlice<f32>,
    pub mx: CudaSlice<f32>,
    pub y: CudaSlice<f32>,
    pub yc: CudaSlice<f32>,
    /// Loop messages [4][rows][width], the controllers' cat / z.
    pub msg: CudaSlice<f32>,
    pub cat: CudaSlice<f32>,
    pub z: CudaSlice<f32>,
    pub logits: CudaSlice<f32>,
    pub idx: CudaSlice<u32>,
    pub wts: CudaSlice<f32>,
    pub xa: Inkling2Act,
    pub act: CudaSlice<f32>,
    pub aa: Inkling2Act,
    pub part: CudaSlice<f32>,
    pub sorted: Inkling2Sorted,
    pub ids: CudaSlice<u32>,
    pub pos: CudaSlice<u32>,
    /// Each row's slot; the runs' row offsets ([runs + 1]) and slots.
    pub slots: CudaSlice<u32>,
    pub run_off: CudaSlice<u32>,
    pub run_slot: CudaSlice<u32>,
    /// A visit's prepped K / V rows on their way into the stores.
    pub kc: CudaSlice<f32>,
    pub vc: CudaSlice<f32>,
    /// Split attention's partials, SPLIT_ROWS rows at most.
    pub part_o: CudaSlice<f32>,
    pub part_ml: CudaSlice<f32>,
    /// The prompt-row attention's tiles [n][4] (first row, rows, slot,
    /// first position).
    pub tiles: CudaSlice<u32>,
    /// With the draft: each row's features [rows][4][width] (the taps'
    /// visit inputs, the final normed output).
    pub feat: CudaSlice<f32>,
}

impl Scratch {
    /// Planes for up to `rows` tokens a chunk (`draft`: the feature plane).
    pub fn alloc(exec: &GpuExecutor, rows: usize, draft: bool) -> Result<Self, GpuModelError> {
        let w = ik::WIDTH;
        let a = |n: usize| exec.alloc(n).map_err(GpuModelError::from);
        let pairs = rows * ik::TOP_K;
        Ok(Self {
            rows,
            emb: a(rows * w)?,
            res: a(rows * w)?,
            head_in: a(forward::HEAD_ROWS * w)?,
            head_out: a(forward::HEAD_ROWS * ik::VOCAB_REAL)?,
            pool: (0..POOL).map(|_| a(rows * w)).collect::<Result<_, _>>()?,
            ares: a(3 * rows * w)?,
            qin: a(rows * w)?,
            kin: a(rows * w)?,
            vin: a(rows * w)?,
            gtmp: a(rows * w)?,
            qrs: a(rows * QRS)?,
            gate: a(rows * ik::HEADS * ik::HEAD_DIM)?,
            kp: a(rows * (KV + DYN))?,
            vp: a(rows * (KV + DYN))?,
            att: a(rows * ik::HEADS * ik::HEAD_DIM)?,
            x16: exec.stream_alloc_bf16(rows * ik::HEADS * ik::HEAD_DIM)?,
            xin16: [
                exec.stream_alloc_bf16(rows * w)?,
                exec.stream_alloc_bf16(rows * w)?,
                exec.stream_alloc_bf16(rows * w)?,
            ],
            pick_par: exec.alloc_u32(forward::HEAD_ROWS * 4)?,
            pick_ids: exec.alloc_u32(forward::HEAD_ROWS)?,
            o: a(rows * w)?,
            ac: a(rows * w)?,
            min: a(rows * w)?,
            mx: a(rows * w)?,
            y: a(rows * w)?,
            yc: a(rows * w)?,
            msg: a(4 * rows * w)?,
            cat: a(rows * 2 * w)?,
            z: a(rows * ik::GATE_RANK)?,
            logits: a(rows * ik::EXPERTS)?,
            idx: exec.alloc_u32(pairs)?,
            wts: a(pairs)?,
            xa: Inkling2Act::alloc(exec, rows, w)?,
            act: a(pairs * ik::FF)?,
            aa: Inkling2Act::alloc(exec, pairs, ik::FF)?,
            part: a(pairs * w)?,
            sorted: Inkling2Sorted::alloc(exec, rows, ik::TOP_K, ik::EXPERTS)?,
            ids: exec.alloc_u32(rows)?,
            pos: exec.alloc_u32(rows)?,
            slots: exec.alloc_u32(rows)?,
            run_off: exec.alloc_u32(rows + 1)?,
            run_slot: exec.alloc_u32(rows)?,
            kc: a(rows * KV)?,
            vc: a(rows * KV)?,
            part_o: a(SPLIT_ROWS * ik::HEADS * MAX_SPLITS * ik::HEAD_DIM)?,
            part_ml: a(SPLIT_ROWS * ik::HEADS * MAX_SPLITS * 2)?,
            // a tile per TILE_ROWS rows of a run, plus each run's remainder
            tiles: exec.alloc_u32(4 * (rows / TILE_ROWS + rows + 1))?,
            feat: a(if draft { rows * draft::NFEAT * w } else { 1 })?,
        })
    }
}

/// Width of the query input's fused plane: q | relative states | sinks.
pub(crate) const QRS: usize = ik::HEADS * ik::HEAD_DIM + ik::HEADS * ik::D_REL + ik::HEADS;
/// K / V widths and the full layers' dynamic taps (8 heads x 4 lags).
pub(crate) const KV: usize = ik::KV_HEADS * ik::HEAD_DIM;
pub(crate) const DYN: usize = ik::KV_HEADS * ik::SCONV_K;
/// Hidden planes a chunk keeps at once: two hidden, four pass-local block
/// states, the prelude's output, the previous pass's block + output, spares.
const POOL: usize = 12;
/// Expert pairs at or below this many tokens take the decode GEMVs.
const GEMV_ROWS: usize = 4;
/// Calls of up to this many rows may split each row's keys (decode ticks);
/// wider ones fill the die with rows alone.
const SPLIT_ROWS: usize = 64;
const MAX_SPLITS: usize = 16;
/// Rows a prompt-row attention tile takes (slot 993's).
const TILE_ROWS: usize = 32;
/// Rows a speculative verify run takes at most (the draft block; the
/// rollback stash holds as many).
pub(crate) const MAX_VERIFY: usize = 8;

/// A captured decode-tick graph. `CapturedGraph` is not `Send` (CUDA
/// handles are context-bound), but the service drives the model from one
/// engine thread, so the wrapper is sound - as the other families' are.
pub(crate) struct SendGraph(pub(crate) crate::gpu::CapturedGraph);
// SAFETY: the graph never leaves the engine thread that captured it
unsafe impl Send for SendGraph {}

pub struct GpuInkling2 {
    pub(crate) exec: std::sync::Arc<GpuExecutor>,
    pub(crate) embed: QuantTensor,
    pub(crate) embed_norm: CudaSlice<f32>,
    pub(crate) norm: CudaSlice<f32>,
    pub(crate) final_q: CudaSlice<f32>,
    /// [12][4][2304] f32: the loop messages' queries (q, k, v, mlp).
    pub(crate) msg_q: Vec<CudaSlice<f32>>,
    pub(crate) unembed: QuantTensor,
    pub(crate) layers: Vec<Layer>,
    pub(crate) kv: kv::Kv,
    /// The release's DSpark draft, when it ships one and the pack has its
    /// ops.
    pub(crate) draft: Option<draft::Draft>,
    /// The last verify pass's (slot, first position) a request, for
    /// `spec_commit`.
    pub(crate) last_verify: Vec<(usize, usize)>,
    /// Gates only: each draft step's top-1 - top-2 margin a block, when on.
    pub(crate) draft_margins: Option<Vec<Vec<f32>>>,
    /// Prompts advancing through mixed ticks (prefill.rs).
    pub(crate) queue: std::collections::VecDeque<prefill::Pending>,
    /// The picture tower, when the release carries one and pictures are on.
    pub(crate) vision: Option<vision::Tower>,
    /// The prefix cache (prefix.rs) once `enable_batch` seated the slots.
    pub(crate) prefix: Option<prefix::Prefix>,
    /// The pass in flight's picture rows: `(pass row, picture, first
    /// picture row, rows)`, written over the placeholders' embeddings
    /// (mm.rs). Empty outside a picture prompt's pass.
    pub(crate) overlay: Vec<mm::Span>,
    /// Decode-tick and verify-pass graphs by the pass's host-side shape, and the
    /// shapes run once eagerly (a shape is captured the second time).
    pub(crate) graphs: std::collections::HashMap<forward::GraphKey, SendGraph>,
    pub(crate) graph_seen: std::collections::HashSet<forward::GraphKey>,
    pub(crate) scratch: Scratch,
    /// Positions a slot may hold; prompt rows a pass.
    pub(crate) max_ctx: usize,
    pub(crate) chunk: usize,
    /// Device bytes the weights hold, the draft's planes included.
    pub(crate) weights_bytes: u64,
    /// Bring-up trace: with `trace_rows` > 0, a prompt's first chunk keeps
    /// every visit's output for its first `trace_rows` rows (visit order).
    pub trace_rows: usize,
    pub trace: Vec<Vec<f32>>,
    /// With `trace_rows`, visit 0's intermediates by stage name (rows x the
    /// stage's width, our layouts).
    pub stages: Vec<(String, Vec<f32>)>,
}

impl GpuInkling2 {
    /// Tokens slot 0 (the single stream) holds.
    pub fn len(&self) -> usize {
        self.kv.lens[0]
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Forget slot 0's sequence: its blocks go back to the pool and the conv
    /// states restart at zero (K / V rows are position-bounded reads).
    pub fn reset(&mut self) -> Result<(), GpuModelError> {
        self.kv.release(0);
        self.kv.zero_conv(&self.exec, 0)
    }

    /// The K / V row class (0 f16, 1 fp8 e4m3, 2 f32).
    pub fn kv_dtype(&self) -> u32 {
        self.kv.dtype
    }
}
