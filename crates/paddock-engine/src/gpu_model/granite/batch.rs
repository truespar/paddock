//! Granite continuous batching: paged KV, batched decode, coalesced and
//! chunked prefill.
//!
//! This is the SIMPLEST batch lane in the repo, and deliberately so. Granite
//! is dense with full attention on every layer - no SWA rings, no hybrid
//! cache kinds, no MoE expert routing, no per-head gate, no QK-norm, no
//! attention sinks. So there is exactly one cache kind: every one of the
//! 40 (8b) / 64 (30b) layers shares a single budget pool of 16-token blocks
//! addressed through per-slot block tables (the gpt-oss G4a shape). KV
//! capacity follows free VRAM, not `max_ctx × max_batch`.
//!
//! Compute class: batched projections quantize the normed rows once per
//! shared input and dispatch per weight - `mmq_pre_any` for decode rows,
//! `prefill_quant` + `prefill_mm_pre_any` for prefill rows. That is the same
//! activation class qwen35/laguna serve on and llama.cpp's own prefill class.
//! The serial `forward_one` keeps its exact-f32 GEMVs as the parity spine;
//! the serve-level greedy gate against llama.cpp arbitrates this class.
//!
//! The four Granite scalars ride through unchanged, and each one still fails
//! silently if dropped, so they are called out at their sites: embedding
//! ×12.0 after the gather, residual ×0.22 on both residual adds, logits
//! ÷16.0 at the head, and `attention_scale` (1/128) used as the KQ scale.
//! Note the residual pair is deliberately not the fused `add_rmsnorm_batch`
//! the unscaled families use - that kernel folds `x += proj` with no
//! multiplier, so granite takes `scale_add` + `rmsnorm_batch` as two
//! launches. Same arithmetic as the parity-validated serial path; folding
//! the scale into a granite-aware fused kernel is a perf follow-up, not a
//! correctness one.
//!
//! Decode perf: the fixed-r decode tick (embed -> layer walk -> head) is
//! captured into a per-r CUDA graph, one replay instead of the ~500 launches
//! a 40-layer walk costs, with device sampling on top (eligible rows come
//! back as bare ids, no [r, vocab] readback). Capture safety is the standing
//! recipe: scratch is allocated once at enable so addresses never move,
//! every loop bound the kernels read at replay comes from a device buffer
//! (d_pos, the block tables), and all host work - table growth, the d_bt
//! upload, row uploads - happens outside the captured region.
//!
//! Radix prefix caching lives in `prefix.rs`. Deliberately
//! not here yet: the depth-2 decode pipe. Speculation is not planned - granite
//! ships no MTP/nextn tensors and no drafter.

use std::collections::HashMap;

use cudarc::driver::CudaSlice;
use cudarc::driver::sys::CUstreamCaptureMode;

use crate::gpu::{GpuError, KvDtype};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::granite::deepstack::{self, InjectSpan};
use crate::gpu_model::qwen35::prefill_quant_w;
use crate::kv_plan;
use crate::kv_pool::{BlockTable, KvPool};

use super::*;

/// Rows per prefill chunk. GEMMs run whole chunks, so rows-per-pass divide
/// straight into prefill throughput - half the chunk is double the weight
/// traffic for the same prompt. Runtime-settable because the scratch planes
/// scale linearly with it (ffn_gate/up alone are rows × n_ff × 4 B, and
/// granite-30b's n_ff is 32768), so the right value is a VRAM-budget
/// decision a 24 GB card must be able to make differently from a 48 GB one.
/// Read once; the batched scratch is sized from it at enable.
pub(crate) fn pf_rows() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_PF_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n: &usize| (256..=8192).contains(&n) && n % 256 == 0)
            .unwrap_or_else(|| PF_ROWS_ELECT.get().copied().unwrap_or(PF_ROWS_DEFAULT))
    })
}

/// Lane-elected chunk width, registered by `enable_batch_impl` before the
/// first `pf_rows()` read (which caches). The W4A4 lane's prefill GEMM (f4t,
/// the TMA ring) saturates by row count - the GEMM ladder on the granite
/// planes reads 988 / 1072 / 1137 TF at 1152 / 4096 / 8192 rows (gate|up) -
/// so a 1024-row chunk left ~14% of the GEMM's rate on the table and paid its
/// per-chunk fixed cost (attention, glue, admission) 32x per imax wave rather
/// than 4x. Measured 4096 vs 1024 over 3 reps: 30b imax +1.3% and
/// syn_2048x128_c32 +10%, 8b imax +1.0% and 2048x128 +7.5%, each with a TTFT
/// step to pay for it. 8192 buys ~2% more GEMM rate for 2x the scratch and a
/// further TTFT step, so 4096 is the knee. Scratch: every plane scales with it
/// (30b ffn_gu 302 MB -> 1.08 GB). The W4A4 (Nvf4 gate|up + down, f4t
/// present) and the fp8-native (f8row gate/up + down, tw5) lanes elect it;
/// the Q8 rungs keep 1024 (see the PF_ROWS_DEFAULT note below).
static PF_ROWS_ELECT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
pub(crate) const PF_ROWS_W4A4: usize = 4096;

/// 1024. An attempt to raise this to 2048 was REVERTED - the reasoning below
///       is kept because the trap is subtle and re-inviting.
///
/// The bait: `prefill_mm_pre_sk` picks its Q8_0 GEMM rung on
/// `batch > MMQ_HI_MIN_BATCH`, and that constant is 1024. A 1024-row chunk
/// fails a STRICTLY-greater test, so a PURE prefill pass falls through the
/// cp.async `q8_0_gemm_mmq_pipe` and 2-blocks/SM `q8_0_gemm_mmq_hi` rungs
/// onto the synchronous barrier-bound 128x128 `q8_0_gemm_mmq`. A prefill-only
/// profile showed exactly that: 72.5% of prefill GPU time in that one kernel
/// over 18,286 launches. An isolated prefill benchmark then "confirmed" the
/// fix.
///
/// Why that was wrong. The isolated benchmark ran `max_tokens=1`, so it had
/// no decode rows. A real fused mixed tick passes `dec_n + chunk_rows` to the
/// GEMM - 32 decode rows against a 1024-row chunk is 1056, which already
/// clears the gate. Serving was therefore taking the fast rung on mixed ticks
/// all along; the gate only bites pure-prefill and tail chunks. Under real
/// concurrent load 2048 bought throughput on the prefill-heavy shapes at a
/// much larger cost in TTFT - the wrong side of that trade for agentic
/// serving - and opening the rung at 1024 rows LOST on the very shape the
/// isolated number said it would help most.
///
/// LESSON, not just a number: an isolated prefill benchmark cannot rank a
/// prefill change, because it cannot see the decode rows the chunk is
/// competing with. Rank under concurrent load. A real fix here has to reason
/// about the scheduler's row composition, not about a threshold in isolation.
const PF_ROWS_DEFAULT: usize = 1024;

/// VRAM slack the slot-fit math leaves untouched (graph/scratch churn).
const VRAM_HEADROOM: usize = 1 << 30;

/// FlashDecoding split ceiling (partial-scratch sizing).
const MAX_ATTN_SPLITS: usize = 16;
/// Split cap for the vec8 decode walk  - deeper than the fused
/// walk's because its per-(q-head, split) CTAs are barrier-free and cheap.
const MAX_VEC8_SPLITS: usize = 32;

/// Ceiling on a spec-verify round's total row count (sum of every live
/// slot's 1 pending + k drafts). Bounds `head_logits`/`d_spec_pick` sizing -
/// mirrors gpt_oss's own `SPEC_BATCH_MAX_ROWS`.
const SPEC_BATCH_MAX_ROWS: usize = 32;

/// Prefill-mode dispatch cuts for one chunk. `runs` = contiguous same-slot
/// row runs: an attention launch must never mix two slots' query rows (the
/// tile walk assumes one sequence), so a coalesced wave attends per run.
/// Single-prompt chunks carry `runs = [(0, r)]`, which keeps the whole-chunk
/// call byte-for-byte identical to the one-prompt path.
///
/// `dec` = leading rows that are DECODE rows (q_len 1, one per slot) in a
/// fused mixed tick. They form one band - `runs[0] = (0, dec)` - dispatched
/// to the decode-batch kernel. Without the fold the band arrives as `dec`
/// separate one-row runs, each paying its own prefill launch (16-row tiles,
/// 1 row used) at every layer. It is also the industry shape: FlashInfer and
/// vLLM dispatch a unified batch as a decode wrapper over the q_len==1 rows
/// plus a prefill wrapper over the chunk, not one ragged kernel.
struct PfCuts {
    runs: Vec<(usize, usize)>,
    dec: usize,
}

impl PfCuts {
    fn new(runs: Vec<(usize, usize)>) -> Self {
        Self { runs, dec: 0 }
    }

    /// The FUSED mixed-tick flavor: `dec` decode rows at the front as one
    /// band, then the chunk's same-slot runs at offsets >= dec.
    fn fused(dec: usize, chunk_runs: Vec<(usize, usize)>) -> Self {
        let mut c = Self::new(chunk_runs);
        if dec > 0 {
            c.runs.insert(0, (0, dec));
            c.dec = dec;
        }
        c
    }
}

/// True when the pack's batched partial fuses the q-group per KV head for
/// this shape (one K/V stage serves all `group` q-heads). Must mirror the
/// paged launcher's own predicate - split budgeting and the partial-vs-plain
/// dispatch both key off it. granite is 32q/8kv = group 4, so this engages
/// at every batched decode width.
fn attn_gqa_fused(n_heads: usize, n_kv_heads: usize, batch: usize) -> bool {
    let group = n_heads.checked_div(n_kv_heads).unwrap_or(1);
    batch > 1
        && (2..=8).contains(&group)
        && n_kv_heads >= 2
        && n_heads == n_kv_heads * group
        && std::env::var_os("PD_NO_GQA_FUSE").is_none()
}

/// KV splits for the batched decode attention. The unsplit kernel is
/// n_heads×batch blocks, which leaves most of even an 84-SM A6000 idle while
/// every block walks its whole KV run serially. Position-INDEPENDENT so the
/// captured per-r graph can bake it.
/// `vec8`: the register-resident per-(q-head, split) fp8 walk
/// launches n_heads*batch blocks per split - budget the die on Q heads and
/// allow deeper splitting (its CTAs are barrier-free and cheap; the in-kernel
/// 32-token target caps live splits at short ctx, so a big bake costs ~0).
/// Bench (g30_dec_attn_bench.cu): B=1 wants the 32 cap (covers ctx 2048),
/// B=4 the formula's ~9.
fn attn_splits_for(
    n_heads: usize,
    n_kv_heads: usize,
    batch: usize,
    sm_count: usize,
    vec8: bool,
) -> usize {
    if paddock_models::dev_var_os!("PADDOCK_NO_ATTN_SPLIT").is_some() {
        return 1;
    }
    if vec8 {
        let want = (2 * 3 * sm_count).div_ceil(n_heads * batch).max(1);
        return want.min(MAX_VEC8_SPLITS);
    }
    if attn_gqa_fused(n_heads, n_kv_heads, batch) {
        // the fused walk launches n_kv*batch blocks per split, so budget the
        // die on those, not on the q-head count
        let want = (2 * 3 * sm_count).div_ceil(n_kv_heads * batch).max(1);
        return want.min(MAX_ATTN_SPLITS);
    }
    if n_heads * batch >= 2 * 3 * sm_count {
        return 1; // die already saturated (un-fused shapes only)
    }
    if let Some(n) = paddock_models::dev_var!("PADDOCK_ATTN_SPLITS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n >= 1)
    {
        return n.min(MAX_ATTN_SPLITS);
    }
    MAX_ATTN_SPLITS
}

/// PADDOCK_NO_W4A8_PREFILL=1 pins the strided decode ladder at prefill rows -
/// the A/B + bisect escape for the flat-mmq tensor-core class.
fn no_w4a8_prefill() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_W4A8_PREFILL").is_some())
}

/// PADDOCK_NO_W4A8_DECODE=1 pins the r=1 decode arm back on the exact-f32
/// GEMV - the A/B reference for the W4A8 decode election, and the way to
/// reproduce the older numeric class byte for byte.
fn no_w4a8_decode() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_W4A8_DECODE").is_some())
}

/// PADDOCK_NO_FUSE_NORM=1 restores the separate scale_add / rmsnorm_batch /
/// quantize_q8_sums launches - the A/B reference for the residual fusion
/// (slot 482). Outputs are bit-identical either way
/// (gpu_add_rmsnorm_q8), so this switches launch COUNT and nothing else,
/// which is the only thing the fusion claims to change.
fn no_fuse_norm() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_FUSE_NORM").is_some())
}

/// PADDOCK_NO_WMMA_PREFILL=1 pins the scalar tiled prefill attention.
fn no_wmma_prefill() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_WMMA_PREFILL").is_some())
}

/// PADDOCK_NO_GEMV_MULTI=1 restores the split q/k/v and gate/up decode GEMV
/// launches - the A/B reference for the one-launch merge (entry 317).
fn no_gemv_multi() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_GEMV_MULTI").is_some())
}

/// PADDOCK_NO_ROPE_FUSE=1 restores the 4-launch rope(q)/rope(k)/append(k)/
/// append(v) band - the A/B reference for the fused kernel (entry 318).
fn no_rope_fuse() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_ROPE_FUSE").is_some())
}

/// Granite's shape (hd128, G=4) rides the v4 staged-HMMA tile's hd128 arm
/// (same kernel family as qwen35's hd256 G in {4,6,8}, extended
/// to hd128 G in {4,6,9} for granite/laguna) once it grew a raw-e4m3 PIPE
/// arm - before that the elected kv8 class fell to the scalar paged walk.
/// The export falls back to that scalar tile when the v4 arm is killed
/// (PADDOCK_NO_PF_V4), so this gate only decides the ENGINE routing;
/// PADDOCK_NO_NPF8 reverts it (mirrors qwen35's PADDOCK_NO_QPF8).
fn pf_attn_dtype_ok(kv_dtype: KvDtype, n_heads: usize, n_kv_heads: usize) -> bool {
    match kv_dtype {
        KvDtype::Fp16 => true,
        KvDtype::Fp8E4m3 => {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_NPF8").is_none())
                && n_kv_heads > 0
                && n_heads.is_multiple_of(n_kv_heads)
                && matches!(n_heads / n_kv_heads, 4 | 6 | 9)
        }
    }
}

/// PADDOCK_NO_COALESCED_PREFILL=1 prefills a wave one prompt at a time - the
/// provably byte-equal A/B reference for the coalesced pass.
fn no_coalesced_prefill() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_COALESCED_PREFILL").is_some())
}

pub(crate) struct LayerKv {
    pub k: CudaSlice<u8>,
    pub v: CudaSlice<u8>,
}

/// Batched-lane scratch, sized once at enable for `cap`-row passes (decode
/// reuses the same planes at rows = live slots « cap).
pub(crate) struct BatchScratch {
    pub x: CudaSlice<f32>,  // residual [cap, embd]
    pub xn: CudaSlice<f32>, // normed rows [cap, embd]
    /// group-quantized activations - the widest consumer is the FFN's
    /// n_ff-wide down input, so one plane serves every quantize site
    pub xq: CudaSlice<i8>,
    pub xs: CudaSlice<f32>,
    /// per-16 int8 sums (the Q4_K/Q5_K mu term) - only the k-quant arm reads
    /// them, but the plane is unconditional so the 30b's Q4_K_M and the 8b's
    /// Q8_0 take the same code path
    pub ssums: CudaSlice<f32>,
    /// mma_ks K-split partials
    pub part: CudaSlice<f32>,
    /// per-16 e4m3 activation scales for the NVFP4 **W4A4** lane. Separate
    /// from `xs` because that one is f32 (the Q8/k-quant row scales) and
    /// `quantize_nvf4` writes e4m3 bytes. n/16 of the widest quantize site.
    pub xs4: CudaSlice<u8>,
    /// Second nvf4 staging pair: the swiglu+quant epilogue reads
    /// (xq, xs4) as its activation input and writes the down GEMM's pair here.
    pub xq2: CudaSlice<i8>,
    pub xs4_2: CudaSlice<u8>,
    /// flat-mmq activations for the prefill tensor-core GEMMs
    pub yq: CudaSlice<u8>,
    /// per-32-block sums off yq - the W4A8 Q4/Q5 mu term
    pub xsums: CudaSlice<f32>,
    /// Q8_0 mmq stream-k fixup plane (only reachable through a Q8_0 weight)
    pub skfix: CudaSlice<f32>,
    pub q: CudaSlice<f32>,
    pub k: CudaSlice<f32>,
    pub v: CudaSlice<f32>,
    pub attn: CudaSlice<f32>,
    pub proj: CudaSlice<f32>,
    /// no-op sinks [n_heads] - granite ships no attention sinks, so this is
    /// -inf (the softmax-denominator identity), not zero. A zeroed buffer
    /// injects a phantom exp(0-max) term that steals real probability mass;
    /// granite is unusually exposed because its KQ scale is 1/128 rather
    /// than 1/sqrt(128), so scores are ~7x smaller. See `alloc_no_sinks` and
    ///  - this exact bug cost 19 of 24 greedy-parity tokens.
    pub sinks: CudaSlice<f32>,
    pub ffn_gate: CudaSlice<f32>,
    pub ffn_up: CudaSlice<f32>,
    /// [rows, 2*n_ff] landing pad for a merged gate|up GEMM.
    pub ffn_gu: CudaSlice<f32>,
    pub d_toks: CudaSlice<u32>,
    pub d_pos: CudaSlice<u32>,
    pub d_slots: CudaSlice<u32>,
    /// [n_runs + 1] row-offset prefix of the tick's prefill runs for the
    /// batched-runs attention launch (pd_pf_runs_register).
    pub pf_runs: CudaSlice<u32>,
    /// [max(n_slots, SPEC_BATCH_MAX_ROWS), vocab] logits. Decode graphs bake
    /// this address - allocated once, never grown. Sized past n_slots so a
    /// spec-verify round (rows = slots × (1+draft_len), not just slots)
    /// never overflows it.
    pub head_logits: CudaSlice<f32>,
    /// device sampler params [n_slots, 4] (inv_t, u, mode, pad)
    pub d_par: CudaSlice<u32>,
    /// sampled token ids [n_slots]
    pub d_out: CudaSlice<u32>,
    /// mode-5/6 truncation side plane [n_slots, 4] {k, top_p bits, min_p bits,
    /// pad} - granite elects no sampling, so this serves DIALLED requests
    pub d_tpar: CudaSlice<u32>,
    /// decode-pipe sampler-param ring [2, n_slots, 4]
    pub d_pipe_par: CudaSlice<u32>,
    /// pipe ring twin of `d_tpar` ([2, n_slots, 4])
    pub d_pipe_tpar: CudaSlice<u32>,
    /// decode-pipe sampled-id ring [2, n_slots]
    pub d_pipe_out: CudaSlice<u32>,
    /// FlashDecoding partial scratch [n_heads, n_slots, MAX_SPLITS, hd]
    pub attn_o: CudaSlice<f32>,
    /// per-partial (m, l) [n_heads, n_slots, MAX_SPLITS, 2]
    pub attn_ml: CudaSlice<f32>,
}

/// The whole batching state: pool + tables + scratch. One struct so
/// enable/teardown is atomic and the field-borrow splits stay simple.
pub(crate) struct BatchState {
    pub n_slots: usize,
    /// Row capacity of every scratch plane = `pf_rows()` + one row per slot.
    /// A fused mixed tick carries the decode band on TOP of a full chunk, so
    /// the planes hold both - sizing at `pf_rows()` alone would make the band
    /// steal chunk rows, and rows-per-pass divide straight into prefill
    /// throughput.
    pub cap: usize,
    /// logical blocks per slot (max_ctx/16) - the block table's slot stride
    pub bps: usize,
    /// the one budget pool (granite is all-full-attention) + per-slot tables
    pub pool: KvPool,
    pub tables: Vec<BlockTable>,
    pub bt_host: Vec<u32>,
    pub d_bt: CudaSlice<u32>,
    /// per-layer K/V stores, [pool_blocks, 16, kv_dim] × f16
    pub kv: Vec<LayerKv>,
    pub sc: BatchScratch,
    /// Radix prefix cache over `pool` (prefix.rs). Zero-copy: a hit adopts
    /// blocks by refcount. None when PADDOCK_NO_PREFIX_CACHE is set.
    pub prefix: Option<crate::paged_radix::PagedRadix>,
    /// T1 RAM tier over the same pool (kv-offload 1a.3; prefix.rs). None
    /// unless the dev flag arms it AND the pack + host memory cooperate.
    pub tier: Option<super::prefix::GraniteTier>,
    /// device bytes the KV stores hold (accounting)
    pub kv_bytes: u64,
    /// captured decode ticks, keyed by row count r - one replay per step.
    /// Grid-stable for a given r (KV loop bounds come from d_pos / the device
    /// block tables at replay).
    pub graphs: HashMap<usize, super::SendGraph>,
    /// captured decode-pipe SAMPLER chains, keyed by (rows, ring parity,
    /// has-truncation). The top-p sampler is ~11 launch-bound kernels
    /// (`pd_topp_mb_*`) that ran EAGER after the forward graph - ~86 us/token
    /// at c1, and by then the dominant remaining per-token cost (the FFN GEMV
    /// already hides inside `graphs`). Replaying them as one graph kills the
    /// launch train.
    /// Keyed on `ring` because the sampler reads `d_pipe_par[ring]` /
    /// writes `d_pipe_out[ring]` at a baked offset; keyed on `has_trunc`
    /// because trunc rows add the t/p kernels. Fresh `u` still uploads eager
    /// before each replay (the graph reads it from the ring), and the doctrine
    /// says only the DISTRIBUTION is contractual, not the seed->token map - so
    /// a replayed sampler is correct. See `sampler_replay`.
    pub sampler_graphs: HashMap<(usize, usize, bool), super::SendGraph>,
    /// The `pd_topp_mb_*` scratch allocates lazily on the first NON-capturing
    /// call and refuses to allocate mid-capture (it would bake the slow
    /// single-CTA fallback into the graph). So the first sampler tick runs
    /// eager to warm it; `sampler_graphs` capture only starts once this is set.
    pub sampler_warm: bool,
    /// argmax picks for a spec-verify round [SPEC_BATCH_MAX_ROWS]. EAGER, not
    /// graph-captured (see `forward_spec_batch_impl`): a verify round's
    /// same-slot run boundaries are data-dependent (they vary with each
    /// slot's own draft length), so a graph keyed on total row count alone
    /// would replay one round's attention grouping onto another round's rows
    /// - a real correctness bug, not just a missed optimization. Re-derive
    ///   the run boundaries fresh every round instead (`rows_pass_body` already
    ///   does this, cheaply, on the host).
    pub d_spec_pick: CudaSlice<u32>,
}

/// A prompt queued for stall-free chunked prefill.
pub(crate) struct ChunkedPrefill {
    pub slot: usize,
    /// The ROW stream's tokens, one per KV row. On a multimodal prompt the
    /// image rows all carry the `<image>` placeholder id.
    pub tokens: Vec<u32>,
    /// Next row to compute. Starts at the prefix-resume point, not 0, so a
    /// radix hit costs no rows.
    pub cursor: usize,
    /// The RADIX key vector - equal to `tokens` for a text prompt, and with
    /// content-derived values at image rows (prefix.rs). Kept so the insert on
    /// completion keys the same way the match did; a prompt that inserted
    /// under different keys than it matched under would never hit itself.
    pub keys: Vec<u32>,
}

fn drv(e: cudarc::driver::DriverError) -> GpuError {
    crate::gpu::from_driver(e)
}

impl GpuGranite {
    /// Allocate the paged-KV + scratch state for up to `max_batch` slots.
    /// Returns the capacity actually enabled; 1 = stay on the serial loop
    /// (no paged kernels in the pack, or max_batch 1).
    pub(crate) fn enable_batch_impl(&mut self, max_batch: usize) -> Result<usize, GpuModelError> {
        self.pipe_abort();
        // NOTE: max_batch==1 is a real, supported width here, not a synonym
        // for "stay serial" - the batched lane is where prefix caching, the
        // paged KV pool, AND the tuned W4A8 decode GEMV all live (the serial
        // lane's `gemv_any` falls back to the exact-f32 oracle for k-quant
        // weights). A literal `--max-batch 1` used to bail here unconditionally
        // before checking anything else, silently losing all three - see
        // service.rs's routing note.
        if !self.exec.has_paged_kv() {
            return Ok(1);
        }
        // the serial dense KV (n_layer × max_ctx) makes way for the paged
        // stores; the serial lane is dead once the engine goes batched
        self.decode = None;
        self.scratch = None;
        self.batch = None;
        self.exec.trim_mem_pool();

        let hp = &self.hp;
        let (embd, nh, n_kv, hd) = (hp.n_embd, hp.n_heads, hp.n_kv_heads, hp.head_dim);
        let kv_dim = n_kv * hd;
        let kvb = self.kv_dtype.bytes();
        let bps = self.max_ctx.div_ceil(16);
        let n_layer = self.layers.len();

        // One block id addresses every layer (a combined block table - vLLM
        // notes it is free vs per-layer tables and cuts metadata), so a block
        // costs all layers' K+V at once.
        let block_bytes = n_layer * 16 * kv_dim * 2 * kvb;
        let q_dim = nh * hd;
        let wide = hp.n_ff.max(q_dim).max(embd);
        // lane election of the chunk width (see PF_ROWS_ELECT) -- must precede
        // the first pf_rows() read below
        let w4a4_lane = self.exec.has_nvf4_gemm_f4t()
            && !self.layers.is_empty()
            && self.layers.iter().all(|l| {
                matches!(&l.gate_up, Some(GraniteW::Nvf4(_)))
                    && matches!(&l.down, GraniteW::Nvf4(_))
            });
        // the fp8-native lane's wide GEMM (tw5) climbs the same ladder
        // (gate|up 624 -> 661 TF, down 577 -> 665, qkv 520 -> 594,
        // o 516 -> 606 at 1152 -> 4096 rows)
        let f8row_lane = !w4a4_lane
            && !self.layers.is_empty()
            && self.layers.iter().all(|l| {
                (matches!(&l.gate, Some(GraniteW::Fp8 { .. }))
                    || matches!(&l.gate_up, Some(GraniteW::Fp8 { .. })))
                    && matches!(&l.down, GraniteW::Fp8 { .. })
            });
        if w4a4_lane || f8row_lane {
            let _ = PF_ROWS_ELECT.set(PF_ROWS_W4A4);
        }
        let cap = pf_rows() + max_batch;
        // Prefill scratch PROFILED at load (the qwen35 shape, 2026-09-06):
        // allocated here, before the grant is read, so the plan meets its
        // true cost in the ledger. It used to be a hand formula charged as a
        // reserve - one that priced 2 n_ff per row where the planes hold 4
        // (gate, up and the merged gate|up pad) and skipped k/v and the
        // quantize staging - and the 1 GiB slack absorbed the difference.
        // Slot-sized planes take the REQUESTED width: the plan may seat
        // fewer, never more.
        let e = &self.exec;
        let v_scratch0 = cudarc::driver::result::mem_get_info()
            .map(|(f, _)| f)
            .unwrap_or(0);
        let q8 = |w: &GraniteW| matches!(w.quant(), Some(QuantW::Q8(_)));
        let any_q8 = q8(&self.lm_head)
            || self.layers.iter().any(|l| {
                q8(&l.wq)
                    || q8(&l.wk)
                    || q8(&l.wv)
                    || q8(&l.wo)
                    || l.gate.as_ref().is_some_and(&q8)
                    || l.up.as_ref().is_some_and(&q8)
                    || l.gate_up.as_ref().is_some_and(&q8)
                    || q8(&l.down)
            });

        let sc = BatchScratch {
            x: e.alloc(cap * embd)?,
            xn: e.alloc(cap * embd)?,
            xq: e.alloc_i8(cap * wide)?,
            xs: e.alloc(cap * wide / 32)?,
            ssums: e.alloc(cap * wide / 16)?,
            part: e.alloc(8 * 192 * wide)?,
            // one e4m3 scale per 16 elements; xq (i8, cap*wide) already holds
            // the packed nibbles, which need only half that
            xs4: e.alloc_u8(cap * wide / 16 + 64)?,
            xq2: e.alloc_i8(cap * wide / 2)?,
            xs4_2: e.alloc_u8(cap * wide / 16 + 64)?,
            yq: e.alloc_u8(wide.div_ceil(128) * cap.next_multiple_of(128) * 144)?,
            xsums: e.alloc(wide.div_ceil(128) * cap.next_multiple_of(128) * 4)?,
            skfix: e.alloc(if any_q8 { 256 * 128 * 128 + 256 } else { 1 })?,
            q: e.alloc(cap * q_dim)?,
            k: e.alloc(cap * kv_dim)?,
            v: e.alloc(cap * kv_dim)?,
            attn: e.alloc(cap * q_dim)?,
            proj: e.alloc(cap * embd)?,
            sinks: e.alloc_no_sinks(nh)?,
            ffn_gate: e.alloc(cap * hp.n_ff)?,
            ffn_up: e.alloc(cap * hp.n_ff)?,
            ffn_gu: e.alloc(cap * 2 * hp.n_ff)?,
            d_toks: e.alloc_u32(cap)?,
            d_pos: e.alloc_u32(cap)?,
            d_slots: e.alloc_u32(cap)?,
            pf_runs: e.alloc_u32(cap + 2)?,
            head_logits: e.alloc(max_batch.max(SPEC_BATCH_MAX_ROWS) * hp.n_vocab)?,
            d_par: e.alloc_u32(max_batch * 4)?,
            d_out: e.alloc_u32(max_batch)?,
            d_tpar: e.alloc_u32(max_batch.max(SPEC_BATCH_MAX_ROWS) * 4)?,
            d_pipe_par: e.alloc_u32(2 * max_batch * 4)?,
            d_pipe_tpar: e.alloc_u32(2 * max_batch * 4)?,
            d_pipe_out: e.alloc_u32(2 * max_batch)?,
            // sized for the DEEPEST split any election can pick - the vec8
            // walk (32) outsplits the fused walk (16); ~8 MB extra scratch on
            // the granite shape, cheap insurance against a silent overflow
            attn_o: e.alloc(nh * max_batch * MAX_VEC8_SPLITS * hd)?,
            attn_ml: e.alloc(nh * max_batch * MAX_VEC8_SPLITS * 2)?,
        };
        let v_scratch1 = cudarc::driver::result::mem_get_info()
            .map(|(f, _)| f)
            .unwrap_or(0);
        let scratch_bytes = v_scratch0.saturating_sub(v_scratch1);
        tracing::info!(
            "granite VRAM  prefill scratch (cap={cap} rows)     {:>7.2} GB  (in the ledger before the grant)",
            scratch_bytes as f64 / 1e9
        );
        let px_on = !super::prefix::prefix_disabled();
        let retain = if px_on {
            super::prefix::retention_blocks()
        } else {
            0
        };
        // One arbiter sizes the KV store: crate::kv_plan. Granite's
        // own arithmetic was already budget-correct - this is the same solve,
        // moved somewhere a new family cannot forget to do it, and it reports the
        // pool's TOKEN CAPACITY rather than leaving max_ctx to imply it.
        let grant = self
            .exec
            .vram_headroom()
            .ok_or_else(|| GpuError::Driver("no free-VRAM reading".into()))?;
        // What the process holds as the plan is made (weights, the profiled
        // scratch): the audit at the end adds the plan's charges to it.
        let ledger_at_plan = self.exec.process_mem_used().unwrap_or(0);
        let demand = kv_plan::Demand {
            family: "granite",
            max_ctx: self.max_ctx,
            slots: max_batch,
            blocks_per_slot: bps,
            block_bytes: block_bytes as u64,
            // One block id addresses every layer, so no KV is per-slot here.
            per_slot_bytes: 0,
            // The pool is capped at what (slots × max_ctx) can actually address
            // plus explicit radix retention (blocks the tree may hold after their
            // sequence ends). This bit on granite specifically, measured:
            // every one of the 64 layers is full-attention, so one
            // block id costs 4 MiB here versus laguna's ~1 MiB. Carrying over
            // laguna's `(slots + 8) * bps` slack reserved 6144 blocks = 24 GiB at
            // max_batch 4 / max_ctx 8192 when only 2048 blocks = 8 GiB can ever be
            // addressed, and drove the card to 48.0 of 49.1 GB. Granite's
            // retention is explicit and named rather than a slot multiple, for
            // exactly that reason: retention_blocks() defaults to 512 = 2 GiB on
            // granite-30b, 0.5 GiB on the 8b, and 0 when the cache is off.
            retention_blocks: retain,
            // every slot must at least be able to hold a full chunk's worth of
            // prompt, or admission deadlocks on its own first chunk
            floor_blocks_per_slot: pf_rows().div_ceil(16),
            floor_blocks_min: 256,
            reserves: {
                // graphs, the sampler planes, the decode state and allocator
                // slack - the prefill scratch itself precedes the grant now
                let mut r = vec![kv_plan::Reserve::new(
                    "graph/scratch slack",
                    VRAM_HEADROOM as u64,
                )];
                // the tier's device staging extents are VRAM this arbiter
                // must know about (kv-offload 1a.1: staging accounted here)
                if px_on && super::prefix::tier_ram_bytes().is_some() {
                    r.push(kv_plan::Reserve::new(
                        "kv-tier staging",
                        crate::kv_tier::ram_transport::device_staging_bytes(),
                    ));
                }
                r
            },
            ..Default::default()
        };
        // A real Err, not a lying Ok(1): the caller (service.rs) treats Ok(c) as
        // proof self.batch is genuinely populated at capacity c - an Ok(1) here
        // would claim that while leaving self.batch None, and service.rs's
        // single-user-batched-decode branch sets `batched=true` on any Ok(_)
        // unconditionally. self.decode/scratch stay serially self-healing
        // regardless (ensure_decode's lazy rebuild), so the caller's serial
        // fallback on Err is safe.
        let plan = demand
            .plan(grant)
            .map_err(|e| GpuModelError::WontFit(e.message))?;
        plan.report(&demand, grant);
        let pool_blocks = plan.pool_blocks;
        let slots = plan.slots;
        let plan_reserved: u64 = demand.reserves.iter().map(|r| r.bytes).sum::<u64>()
            + plan.pool_bytes
            + plan.slot_bytes;

        let mut kv = Vec::with_capacity(n_layer);
        let mut kv_bytes = 0u64;
        for _ in 0..n_layer {
            let bytes = pool_blocks * 16 * kv_dim * kvb;
            kv_bytes += 2 * bytes as u64;
            kv.push(LayerKv {
                k: e.alloc_u8(bytes)?,
                v: e.alloc_u8(bytes)?,
            });
        }

        // KV tier (kv-offload 1a.3, dev flag): built against the freshly
        // allocated planes, arming the radix's chain keys before any insert
        let mut prefix = px_on.then(crate::paged_radix::PagedRadix::new);
        let tier = match (prefix.as_mut(), super::prefix::tier_ram_bytes()) {
            (Some(radix), Some(ram)) => {
                let t = super::prefix::build_tier(
                    e,
                    &self.hp,
                    self.kv_dtype,
                    &kv,
                    self.max_ctx,
                    ram,
                    self.content_id,
                );
                if let Some(t) = &t {
                    radix.set_tier_root(t.tier_root());
                }
                t
            }
            _ => None,
        };
        self.batch = Some(BatchState {
            n_slots: slots,
            cap,
            bps,
            pool: KvPool::with_blocks(pool_blocks as u32),
            tables: (0..slots).map(|_| BlockTable::new()).collect(),
            bt_host: vec![0u32; slots * bps],
            d_bt: e.alloc_u32(slots * bps)?,
            kv,
            sc,
            prefix,
            tier,
            kv_bytes,
            graphs: HashMap::new(),
            sampler_graphs: HashMap::new(),
            sampler_warm: false,
            d_spec_pick: e.alloc_u32(SPEC_BATCH_MAX_ROWS)?,
        });
        self.last_reused = vec![0; slots];
        self.seal_hist = vec![Vec::new(); slots];
        self.seal_ok = vec![true; slots];
        // Say what was taken AND what was left - this card usually has a
        // desktop session on it, so "how much did paddock just claim" is a
        // number the operator needs without reaching for nvidia-smi.
        tracing::info!(
            "granite batch: {slots} slots, {n_layer}-layer pool {pool_blocks} blocks \
             ({:.2} GiB, {} tokens), {} rows/chunk; left {:.2} GiB of the {:.2} GiB granted",
            (pool_blocks * block_bytes) as f64 / (1u64 << 30) as f64,
            pool_blocks * 16,
            pf_rows(),
            grant.saturating_sub(plan_reserved) as f64 / (1u64 << 30) as f64,
            grant as f64 / (1u64 << 30) as f64,
        );
        // Plan audit (the qwen35 line): the plan's charges on top of what the
        // process held when it was made, against the ledger now. The gap is
        // what `graph/scratch slack` has to cover; keep it visible.
        let expected = ledger_at_plan + plan_reserved;
        let actual = self.exec.process_mem_used().unwrap_or(0);
        tracing::info!(
            expected_gib = expected as f64 / (1u64 << 30) as f64,
            ledger_gib = actual as f64 / (1u64 << 30) as f64,
            unplanned_mib = actual.saturating_sub(expected) as f64 / (1u64 << 20) as f64,
            "granite VRAM plan audit: ledger vs plan after enable_batch"
        );
        Ok(slots)
    }

    /// Back every `(slot, position)` this pass will touch with a physical
    /// pool block, re-uploading the device table once on growth.
    /// PoolExhausted surfaces to the scheduler, which preempts.
    fn ensure_rows(&mut self, slots: &[u32], positions: &[u32]) -> Result<(), GpuModelError> {
        let bs = self.batch.as_mut().expect("batch enabled");
        let mut grew = false;
        for (i, &s) in slots.iter().enumerate() {
            let s = s as usize;
            let before = bs.tables[s].blocks().len();
            loop {
                match bs.tables[s].ensure(positions[i] as usize, &mut bs.pool) {
                    Ok(()) => break,
                    // Dry pool: shed radix retention (LRU leaves) before asking
                    // the scheduler to preempt a live sequence. The cache is
                    // reclaimable capacity, so losing a cached prefix is always
                    // preferable to evicting somebody's in-flight request.
                    // Tier-aware: the shed demotes closing runs first, and a
                    // demote's pin defers the free - drain the transport
                    // briefly before concluding the pool is truly dry.
                    Err(_) => {
                        let shed = match (bs.tier.as_mut(), bs.prefix.as_mut()) {
                            (Some(tier), Some(radix)) => {
                                let want = bs.pool.free_blocks() + 1;
                                let (_e, aux) =
                                    tier.pressure_demote(radix, &mut bs.pool, want, None);
                                for a in aux {
                                    radix.recycle_state(a.state_idx);
                                }
                                let deadline = std::time::Instant::now()
                                    + std::time::Duration::from_millis(50);
                                while bs.pool.free_blocks() < want && tier.stats().2 > 0 {
                                    tier.pump_completions(radix, &mut bs.pool);
                                    if std::time::Instant::now() >= deadline {
                                        break;
                                    }
                                    std::thread::sleep(std::time::Duration::from_micros(200));
                                }
                                bs.pool.free_blocks() >= want
                            }
                            (None, Some(radix)) => radix.evict_lru(&mut bs.pool).is_some(),
                            _ => false,
                        };
                        if !shed {
                            return Err(GpuModelError::PoolExhausted);
                        }
                    }
                }
            }
            let now = bs.tables[s].blocks().len();
            if now > before {
                grew = true;
                let base = s * bs.bps;
                for j in before..now {
                    bs.bt_host[base + j] = bs.tables[s].blocks()[j];
                }
            }
        }
        if grew {
            self.exec
                .stream
                .memcpy_htod(&bs.bt_host, &mut bs.d_bt)
                .map_err(drv)?;
        }
        Ok(())
    }

    /// Free-on-completion: an idle slot's blocks return to the shared pool
    /// immediately instead of being pinned until the slot is next reused.
    /// Reset a slot's seal mirror to the rows already BACKED at admission
    /// (the radix-adopted resume prefix). See `GpuGranite::seal_hist`.
    pub(crate) fn hist_reset(&mut self, slot: usize, backed_keys: &[u32]) {
        if slot < self.seal_hist.len() {
            self.seal_hist[slot].clear();
            self.seal_hist[slot].extend_from_slice(backed_keys);
            self.seal_ok[slot] = true;
        }
    }

    /// Append fed rows to the mirror. `pos` = the first row's KV position
    /// when the caller knows it (positional check: a gap means an uncovered
    /// feed path and POISONS the mirror - never publish a wrong chain);
    /// `None` = sequential source (the decode pipe's drained ids, which
    /// arrive strictly in row order per slot).
    fn hist_append(&mut self, slot: usize, pos: Option<u32>, toks: &[u32]) {
        if slot >= self.seal_hist.len() || !self.seal_ok[slot] {
            return;
        }
        let len = self.seal_hist[slot].len();
        match pos {
            Some(p) if (p as usize) == len => self.seal_hist[slot].extend_from_slice(toks),
            Some(p) if (p as usize) + toks.len() <= len => {} // re-feed (recompute) - known rows
            Some(_) => {
                // gap: rows landed that this mirror never saw (spec rounds
                // until 1b.4). Poison - publication skips this sequence.
                self.seal_ok[slot] = false;
                self.seal_hist[slot].clear();
            }
            None => self.seal_hist[slot].extend_from_slice(toks),
        }
    }

    pub(crate) fn release_inactive_slots_impl(&mut self, occupied: &[bool]) {
        // Vision streams go the same way, and for a bigger reason: a max-grid
        // image is ~130 MB of projected streams held for as long as its
        // registry entry lives. Waiting for the slot's next admission to drop
        // them pins that VRAM across an idle server.
        if !self.media.is_empty() {
            for (s, occ) in occupied.iter().enumerate() {
                if !occ {
                    self.media.clear_slot(s);
                }
            }
        }
        // 1b.1: publish each finished sequence's whole backed chain (prompt
        // + generated tail) before its blocks release - the tail is the next
        // turn's prefix. prefix_insert retains what it publishes; the clear
        // below then drops only the slot's own references.
        let publish: Vec<usize> = {
            let Some(bs) = self.batch.as_ref() else {
                return;
            };
            occupied
                .iter()
                .enumerate()
                .filter(|&(s, occ)| {
                    !occ && s < bs.tables.len()
                        && !bs.tables[s].blocks().is_empty()
                        && s < self.seal_hist.len()
                        && self.seal_ok[s]
                        && self.seal_hist[s].len() >= 16
                })
                .map(|(s, _)| s)
                .collect()
        };
        for s in publish {
            let keys = std::mem::take(&mut self.seal_hist[s]);
            self.prefix_insert(s, &keys);
        }
        let Some(bs) = self.batch.as_mut() else {
            return;
        };
        for (s, occ) in occupied.iter().enumerate() {
            if !occ && s < bs.tables.len() && !bs.tables[s].blocks().is_empty() {
                bs.tables[s].clear(&mut bs.pool);
                if s < self.seal_hist.len() {
                    self.seal_hist[s].clear();
                    self.seal_ok[s] = true;
                }
            }
        }
    }

    /// Free blocks for the admission watermark, INCLUDING what the prefix
    /// cache is holding but could give back. The cache is reclaimable
    /// capacity, not a reservation: counting only `free_blocks()` lets a
    /// retention-heavy workload drive it to ~0 and serialize the whole server
    /// behind slot completions (found on gemma4: c8 TTFT 3.3 s -> 52 s).
    pub(crate) fn pool_free_blocks_impl(&self) -> Option<usize> {
        self.batch
            .as_ref()
            .map(|b| b.pool.free_blocks() + self.prefix_evictable())
    }

    pub(crate) fn kv_mem_bytes_impl(&self) -> Option<u64> {
        self.batch.as_ref().map(|b| b.kv_bytes)
    }

    // ── prefill ─────────────────────────────────────────────────────────────

    /// Prefill a whole prompt into `slot` (chunked at `pf_rows`) and return
    /// the last token's logits. Fresh sequence: the slot's old pool blocks
    /// return first.
    pub(crate) fn forward_prefill_impl(
        &mut self,
        slot: usize,
        tokens: &[u32],
    ) -> Result<Vec<f32>, GpuModelError> {
        self.admit(slot, tokens)?;
        let mut base = self.prefill_resume_rows(slot, tokens, tokens.len())?;
        let mut last_len = 0usize;
        for chunk in tokens[base..].chunks(pf_rows()) {
            let rows: Vec<(u32, u32, u32)> = chunk
                .iter()
                .enumerate()
                .map(|(j, &t)| (slot as u32, (base + j) as u32, t))
                .collect();
            self.rows_pass_body(&rows, 0)?;
            base += chunk.len();
            last_len = chunk.len();
        }
        self.prefix_insert(slot, tokens);
        self.hist_reset(slot, tokens); // whole prompt fed - mirror = all of it
        self.head_row(last_len - 1)
    }

    /// Admission prologue shared by every prefill entry: bounds-check the
    /// prompt, drop the slot's previous sequence, and back the blocks the
    /// whole prompt will need up front (so a mid-prompt chunk can never
    /// discover the pool is dry with rows already written).
    fn admit(&mut self, slot: usize, tokens: &[u32]) -> Result<(), GpuModelError> {
        self.admit_rows(slot, tokens.len())
    }

    /// The row-counted half of admission. The multimodal lane calls this
    /// directly because its ROW count is not its token count: one `<image>`
    /// placeholder expands to the AnyRes row run (144 for a thumbnail, ~2.5k
    /// for a max-grid strip), so there is no token slice to hand `admit`.
    pub(crate) fn admit_rows(&mut self, slot: usize, n_rows: usize) -> Result<(), GpuModelError> {
        let n_slots = self.batch.as_ref().expect("batch enabled").n_slots;
        if slot >= n_slots {
            return Err(GpuModelError::Unsupported(format!(
                "slot {slot} >= enabled {n_slots}"
            )));
        }
        if n_rows == 0 {
            return Err(GpuModelError::Unsupported("empty prompt".into()));
        }
        if n_rows > self.max_ctx {
            return Err(GpuModelError::Unsupported(format!(
                "prompt {n_rows} tokens > max_ctx {}",
                self.max_ctx
            )));
        }
        {
            let bs = self.batch.as_mut().expect("batch enabled");
            bs.tables[slot].clear(&mut bs.pool);
        }
        // A fresh sequence in this slot: whatever pictures the last request put
        // there are gone. Without this the new prompt's rows would still fall
        // inside the old image's position span and get vision features injected
        // into text - fluent, wrong, and completely silent.
        self.media.clear_slot(slot);
        self.ensure_rows(&[slot as u32], &[(n_rows - 1) as u32])
    }

    /// COALESCED multi-prompt prefill: every pending prompt's rows concatenate
    /// into shared `pf_rows` chunks - One weight-amortized pass over the wave
    /// instead of one pass per prompt. A 32×128-token admission burst as 32
    /// sequential passes pays the whole weight stream 32 times.
    ///
    /// Correctness leans on two standing laws: (1) r-invariance - every
    /// prefill row takes the same kernel rungs and its bytes depend only on
    /// its own row content, so sharing a pass with other prompts changes
    /// nothing; (2) run isolation - attention launches never mix two slots'
    /// query rows (PfCuts), and appends land per-row in each slot's own pool
    /// blocks.
    pub(crate) fn forward_prefill_batch_impl(
        &mut self,
        items: &[(usize, Vec<u32>)],
    ) -> Result<Vec<Vec<f32>>, GpuModelError> {
        // single item (or the A/B pin): the serial path, provably byte-equal
        if items.len() == 1 || no_coalesced_prefill() {
            return items
                .iter()
                .map(|(slot, toks)| self.forward_prefill_impl(*slot, toks))
                .collect();
        }
        // Admit, then resume each item off the radix. Items inside one wave
        // cannot reuse each OTHER's prefixes - the insert below needs written
        // blocks - which is the known cost of coalescing, not a bug.
        let mut starts = vec![0usize; items.len()];
        for (it, (slot, tokens)) in items.iter().enumerate() {
            self.admit(*slot, tokens)?;
            starts[it] = self.prefill_resume_rows(*slot, tokens, tokens.len())?;
        }
        // the wave's row stream, items contiguous in order; last_row[it] = the
        // row whose logits item `it` needs. Each item contributes only the
        // rows the cache did not already hold.
        let mut rows: Vec<(u32, u32, u32)> = Vec::new();
        let mut last_row = vec![0usize; items.len()];
        for (it, (slot, toks)) in items.iter().enumerate() {
            for (j, &t) in toks.iter().enumerate().skip(starts[it]) {
                rows.push((*slot as u32, j as u32, t));
            }
            last_row[it] = rows.len() - 1;
        }
        let mut out: Vec<Vec<f32>> = vec![Vec::new(); items.len()];
        let mut base = 0usize;
        for chunk in rows.chunks(pf_rows()) {
            let r = chunk.len();
            // finishers whose last row landed in this chunk read inside the
            // pass - the next chunk's embed overwrites sc.x. Ascending by row
            // because head_row bounces its row through x[0].
            let mut fin: Vec<(usize, usize)> = last_row
                .iter()
                .enumerate()
                .filter(|&(_, &lr)| lr >= base && lr < base + r)
                .map(|(it, &lr)| (lr - base, it))
                .collect();
            fin.sort_unstable();
            self.rows_pass_body(chunk, 0)?;
            for (row, it) in fin {
                out[it] = self.head_row(row)?;
            }
            base += r;
        }
        for (slot, toks) in items {
            self.prefix_insert(*slot, toks);
        }
        Ok(out)
    }

    /// Queue a prompt for STALL-FREE chunked prefill (Sarathi-Serve
    /// Algorithm 3). Does the
    /// whole admission prologue now, so a mixed tick only has to move rows.
    pub(crate) fn prefill_begin_impl(
        &mut self,
        slot: usize,
        tokens: Vec<u32>,
    ) -> Result<(), GpuModelError> {
        // a queued entry for this slot is STALE (the scheduler keeps one chunk
        // per live slot - a duplicate means the old request died and the slot
        // was reused): evict rather than wedge the slot
        self.chunked.retain(|c| c.slot != slot);
        self.admit(slot, &tokens)?;
        // Radix resume after admission: `admit` released this slot's previous
        // blocks and backed the whole prompt, then the resume drops that table
        // again and adopts the cached blocks instead, growing only the tail.
        // A text prompt's keys are its tokens.
        let cursor = self.prefill_resume_rows(slot, &tokens, tokens.len())?;
        self.hist_reset(slot, &tokens[..cursor]);
        self.chunked.push(ChunkedPrefill {
            slot,
            keys: tokens.clone(),
            tokens,
            cursor,
        });
        Ok(())
    }

    /// Shared resume for both lanes: match the radix on `keys`, adopt what it
    /// has, then re-back the tail so every row this prompt will write has a
    /// block. Returns the row to start computing at.
    pub(crate) fn prefill_resume_rows(
        &mut self,
        slot: usize,
        keys: &[u32],
        n_rows: usize,
    ) -> Result<usize, GpuModelError> {
        let start = self.prefix_resume(slot, keys)?;
        if start > 0 {
            self.ensure_rows(&[slot as u32], &[(n_rows - 1) as u32])?;
        }
        Ok(start)
    }

    /// Drop slot's in-flight prefill (client hung up mid-prompt).
    pub(crate) fn prefill_abort_impl(&mut self, slot: usize) -> bool {
        let n = self.chunked.len();
        self.chunked.retain(|c| c.slot != slot);
        self.chunked.len() != n
    }

    /// Pick this tick's chunk rows: FIFO over the queue, up to `budget` rows,
    /// splitting the last prompt if it does not fit. Returns the row stream
    /// and (queue index, rows taken, finishes?) per touched entry.
    fn plan_chunk(&self, budget: usize) -> (Vec<(u32, u32, u32)>, Vec<(usize, usize, bool)>) {
        let mut rows: Vec<(u32, u32, u32)> = Vec::new();
        let mut take: Vec<(usize, usize, bool)> = Vec::new();
        if self.chunked.is_empty() {
            return (rows, take);
        }
        let cap = budget.clamp(1, pf_rows());
        for (qi, c) in self.chunked.iter().enumerate() {
            if rows.len() >= cap {
                break;
            }
            let remaining = c.tokens.len() - c.cursor;
            let n = remaining.min(cap - rows.len()).max(1);
            for j in 0..n {
                let p = c.cursor + j;
                rows.push((c.slot as u32, p as u32, c.tokens[p]));
            }
            take.push((qi, n, n == remaining));
        }
        (rows, take)
    }

    /// Advance cursors and drop finished prompts from the queue.
    fn commit_chunk(
        &mut self,
        take: &[(usize, usize, bool)],
        finished_raw: Vec<(usize, crate::generator::FinishSample)>,
    ) -> Vec<(usize, crate::generator::FinishSample, usize)> {
        for &(qi, n, _) in take {
            self.chunked[qi].cursor += n;
        }
        // seal mirror: these rows' KV just landed (the pass succeeded)
        let fed: Vec<(usize, u32, Vec<u32>)> = take
            .iter()
            .map(|&(qi, n, _)| {
                let c = &self.chunked[qi];
                (
                    c.slot,
                    (c.cursor - n) as u32,
                    c.keys[c.cursor - n..c.cursor].to_vec(),
                )
            })
            .collect();
        for (s, p, ks) in fed {
            self.hist_append(s, Some(p), &ks);
        }
        let mut out = Vec::new();
        for (qi, fs) in finished_raw {
            let slot = self.chunked[qi].slot;
            let toks = std::mem::take(&mut self.chunked[qi].tokens);
            // Publish now that every block is written. A prompt admitted in
            // the same wave as this one cannot have reused it - the insert
            // needs written blocks - which is the known cost of coalescing.
            let keys = std::mem::take(&mut self.chunked[qi].keys);
            self.prefix_insert(slot, &keys);
            out.push((slot, fs, toks.len()));
        }
        self.chunked.retain(|c| !c.tokens.is_empty());
        out
    }

    /// Build the fused tick's row stream: decode rows first (one band), then
    /// as much of the prefill queue as the scratch capacity allows.
    fn fuse_rows(
        &self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
    ) -> (
        Vec<(u32, u32, u32)>,
        usize,
        Vec<(usize, usize)>,
        Vec<(usize, usize, bool)>,
    ) {
        let mut rows: Vec<(u32, u32, u32)> =
            decodes.iter().map(|&(s, t, p)| (s as u32, p, t)).collect();
        let dec_n = rows.len();
        // decode rows and the chunk share one scratch plane, so the tick is
        // bounded by BatchState::cap (= pf_rows + one row per slot), which is
        // exactly why the band never eats chunk rows
        let room = self
            .batch
            .as_ref()
            .expect("batch enabled")
            .cap
            .saturating_sub(dec_n);
        let (chunk_rows, take) = if room == 0 {
            (Vec::new(), Vec::new())
        } else {
            self.plan_chunk(budget.min(room))
        };
        rows.extend_from_slice(&chunk_rows);
        let mut fin: Vec<(usize, usize)> = Vec::new();
        let mut off = dec_n;
        for &(qi, n, done) in &take {
            if done {
                fin.push((off + n - 1, qi));
            }
            off += n;
        }
        (rows, dec_n, fin, take)
    }

    /// One FUSED mixed tick: decode rows and the prefill chunk in a single
    /// weight-amortized pass, with the decode rows DEVICE-SAMPLED. This is
    /// the shape Sarathi-Serve's kernel contract specifies - per-sequence
    /// (q_len, kv_len) in one batch, decodes at q_len 1 and the chunk at
    /// q_len = rows.
    ///
    /// Decode rows give up the CUDA-graph-captured path here (they ride the
    /// prefill class's kernels) and buy back half the tick's weight traffic:
    /// running prefill and decode as two passes streams every weight twice.
    pub(crate) fn forward_mixed_sampled_impl(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
        plans: &[crate::generator::RowSample],
        fin_plans: &[(usize, crate::generator::RowSample)],
    ) -> Result<
        (
            crate::generator::SampledStep,
            Vec<(usize, crate::generator::FinishSample, usize)>,
        ),
        GpuModelError,
    > {
        use crate::generator::{FinishSample, SampledStep};
        // Nothing queued -> a plain decode tick. Fusing here would drag the
        // COMMON tick off its captured graph for no prefill work at all.
        if self.chunked.is_empty() {
            let step = if decodes.is_empty() {
                SampledStep {
                    ids: Vec::new(),
                    host_rows: Vec::new(),
                }
            } else {
                let toks: Vec<u32> = decodes.iter().map(|d| d.1).collect();
                let pos: Vec<u32> = decodes.iter().map(|d| d.2).collect();
                let slots: Vec<u32> = decodes.iter().map(|d| d.0 as u32).collect();
                self.forward_batch_sampled_slots(&toks, &pos, Some(&slots), plans)?
            };
            return Ok((step, Vec::new()));
        }
        let (rows, dec_n, fin, take) = self.fuse_rows(decodes, budget);
        if dec_n > 0 {
            let slots: Vec<u32> = decodes.iter().map(|d| d.0 as u32).collect();
            let pos: Vec<u32> = decodes.iter().map(|d| d.2).collect();
            self.ensure_rows(&slots, &pos)?;
        }
        self.rows_pass_body(&rows, dec_n)?;
        // seal mirror: the fused decode band's KV just landed
        for &(s0, t, p) in decodes.iter().take(dec_n) {
            self.hist_append(s0, Some(p), &[t]);
        }
        // Decode rows first: one bulk norm+head over rows 0..dec_n, then
        // sample on device. It has to precede the finisher heads - head_row
        // bounces its row through x[0] and rewrites head_logits[0..vocab],
        // which is decode row 0's.
        let step = if dec_n > 0 {
            self.head_rows(dec_n)?;
            self.sample_head_rows(dec_n, plans)?
        } else {
            SampledStep {
                ids: Vec::new(),
                host_rows: Vec::new(),
            }
        };
        // Finishers: every finishing prompt used to run its own head pass (an
        // 822 MB lm_head stream, ~540 us) plus its own sampler chain and D2H
        // sync -- serially, ~3.2 ms per finisher on the mixed tick's wall
        // (tick ms ~ 100 + 3.2 x finishers, and a saturated cohort finishes
        // 10-14 prompts per mixed tick). The
        // device-plan finishers are now GATHERED into x[0..n] (rows bounced
        // through proj so no read is clobbered) and take one norm+head at
        // M=n and one device sample, like the decode band. Host-plan
        // finishers (constraints, logprobs, non-device samplers) keep the
        // serial logits path and run first, before the gather overwrites
        // x[0..n]. Per finisher the head math is the decode band's own
        // batched class (the M>1 tier), so the first token is what a decode
        // row would sample. Kill: PADDOCK_NO_FIN_BATCH.
        let fin_batch = paddock_models::dev_var_os!("PADDOCK_NO_FIN_BATCH").is_none();
        let mut finished_raw: Vec<(usize, FinishSample)> = Vec::with_capacity(fin.len());
        let mut dev_fin: Vec<(usize, usize, crate::generator::RowSample)> = Vec::new();
        let mut order: Vec<(usize, Option<FinishSample>)> = Vec::with_capacity(fin.len());
        for &(row, qi) in &fin {
            let slot = self.chunked[qi].slot;
            let plan = fin_plans.iter().find(|(s, _)| *s == slot).map(|(_, p)| *p);
            match plan {
                Some(p @ crate::generator::RowSample::Device(_)) if fin_batch => {
                    dev_fin.push((order.len(), row, p));
                    order.push((qi, None));
                }
                Some(p @ crate::generator::RowSample::Device(_)) => {
                    self.head_row_at(row)?;
                    let s = self.sample_head_rows(1, std::slice::from_ref(&p))?;
                    order.push((qi, Some(FinishSample::Sampled(s.ids[0]))));
                }
                _ => {
                    let fs = FinishSample::Logits(self.head_row(row)?);
                    order.push((qi, Some(fs)));
                }
            }
        }
        if !dev_fin.is_empty() {
            let ids = self.head_sample_rows_gathered(&dev_fin)?;
            for ((oi, _, _), id) in dev_fin.iter().zip(ids) {
                order[*oi].1 = Some(FinishSample::Sampled(id));
            }
        }
        for (qi, fs) in order {
            finished_raw.push((qi, fs.expect("every finisher sampled")));
        }
        let finished = self.commit_chunk(&take, finished_raw);
        Ok((step, finished))
    }

    /// The unsampled mixed tick (full logits readback). Backends keep this so
    /// the scheduler's non-device-sampling path stays available.
    pub(crate) fn forward_mixed_impl(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
    ) -> Result<(Vec<f32>, Vec<(usize, Vec<f32>, usize)>), GpuModelError> {
        if self.chunked.is_empty() {
            if decodes.is_empty() {
                return Ok((Vec::new(), Vec::new()));
            }
            let toks: Vec<u32> = decodes.iter().map(|d| d.1).collect();
            let pos: Vec<u32> = decodes.iter().map(|d| d.2).collect();
            let slots: Vec<u32> = decodes.iter().map(|d| d.0 as u32).collect();
            self.batch_step_slots(&toks, &pos, &slots)?;
            for &(s0, t, p) in decodes {
                self.hist_append(s0, Some(p), &[t]);
            }
            return Ok((self.read_batch_logits(decodes.len())?, Vec::new()));
        }
        let (rows, dec_n, fin, take) = self.fuse_rows(decodes, budget);
        if dec_n > 0 {
            let slots: Vec<u32> = decodes.iter().map(|d| d.0 as u32).collect();
            let pos: Vec<u32> = decodes.iter().map(|d| d.2).collect();
            self.ensure_rows(&slots, &pos)?;
        }
        self.rows_pass_body(&rows, dec_n)?;
        for &(s0, t, p) in decodes.iter().take(dec_n) {
            self.hist_append(s0, Some(p), &[t]);
        }
        let mut dec_logits = Vec::new();
        if dec_n > 0 {
            self.head_rows(dec_n)?;
            dec_logits = self.read_batch_logits(dec_n)?;
        }
        let mut finished_raw = Vec::with_capacity(fin.len());
        for &(row, qi) in &fin {
            finished_raw.push((
                qi,
                crate::generator::FinishSample::Logits(self.head_row(row)?),
            ));
        }
        let finished = self
            .commit_chunk(&take, finished_raw)
            .into_iter()
            .map(|(slot, fs, n)| match fs {
                crate::generator::FinishSample::Logits(l) => (slot, l, n),
                // this path never asks for device sampling
                crate::generator::FinishSample::Sampled(_) => unreachable!("unsampled mixed tick"),
            })
            .collect();
        Ok((dec_logits, finished))
    }

    /// One weight-amortized pass over a ready-made row stream - the shared
    /// body of every prefill lane (whole-prompt, coalesced wave, and the
    /// stall-free mixed tick). `chunk` is (slot, pos, token) with items
    /// contiguous; the leading `dec` rows are a fused tick's decode band.
    ///
    /// Rows may start at any position in their slot - a mid-prompt chunk
    /// resume is the same thing to this pass as a fresh prompt, which is what
    /// makes stall-free batching possible without a second code path.
    pub(super) fn rows_pass_body(
        &mut self,
        chunk: &[(u32, u32, u32)],
        dec: usize,
    ) -> Result<(), GpuModelError> {
        let r = chunk.len();
        let toks: Vec<u32> = chunk.iter().map(|x| x.2).collect();
        let positions: Vec<u32> = chunk.iter().map(|x| x.1).collect();
        let slots_v: Vec<u32> = chunk.iter().map(|x| x.0).collect();
        // contiguous same-slot runs over the PREFILL rows - an attention
        // launch never mixes two slots' query rows. The leading `dec` rows are
        // the decode band and are not run-split: they attend as one launch.
        let mut runs: Vec<(usize, usize)> = Vec::new();
        for (i, x) in chunk.iter().enumerate().skip(dec) {
            match runs.last_mut() {
                Some((off, n)) if chunk[*off].0 == x.0 => *n += 1,
                _ => runs.push((i, 1)),
            }
        }
        // Resolve vision rows from each row's (slot, position) before the walk:
        // a chunk boundary may land inside an image, and only the position tells
        // us how far into its streams to start. Empty on text-only checkpoints.
        let spans = if self.media.is_empty() {
            Vec::new()
        } else {
            self.media.plan(chunk)
        };
        self.upload_rows(&toks, &positions, &slots_v)?;
        self.embed_rows(r, &spans)?;
        self.layer_walk(r, Some(&PfCuts::fused(dec, runs)), &spans)
    }

    // ── model-free speculative decode (prompt-lookup verify) ────────────────

    /// One speculative batched VERIFY round: `reqs[i] = (slot, start_pos,
    /// chunk)` with `chunk[0]` the slot's committed pending token and
    /// `chunk[1..]` the service's n-gram drafts (granite has no MTP head, so
    /// every draft here is model-free prompt-lookup - see `spec.rs`'s
    /// `NgramDraft`). Returns each row's GREEDY argmax pick, flat in request
    /// order, matching the `Generator::forward_spec_batch` contract exactly.
    ///
    /// Reuses `rows_pass_body` UNCHANGED: a verify round is structurally the
    /// same shape as a chunked-prefill pass (multiple rows per slot,
    /// contiguous same-slot runs attending causally to their own prior KV
    /// plus each other) - the only difference is who is asking (the spec
    /// scheduler vs. the prefill queue) and that granite's own greedy pick
    /// replaces the caller's sampling. This is why no new attention/GEMV path
    /// was needed: the causal masking a draft chunk needs is identical to
    /// what a resumed mid-prompt chunk already needs, and that path already
    /// exists and is already correctness-gated (greedy parity).
    ///
    /// Deliberately EAGER (no CUDA-graph capture): see `d_spec_pick`'s doc
    /// comment for why baking a graph here would be a correctness bug, not
    /// just a missed optimization.
    pub(crate) fn forward_spec_batch_impl(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
    ) -> Result<Vec<u32>, GpuModelError> {
        let n_slots = self
            .batch
            .as_ref()
            .ok_or(GpuModelError::BatchDisabled)?
            .n_slots;
        let r: usize = reqs.iter().map(|(_, _, c)| c.len()).sum();
        if r == 0 || r > SPEC_BATCH_MAX_ROWS {
            return Err(GpuModelError::BatchTooLarge {
                got: r,
                max: SPEC_BATCH_MAX_ROWS,
            });
        }
        // Every live slot's n-gram drafter missed this round (freeform prose
        // has few repeats to match) - every chunk is just its pending token,
        // i.e. this round is a plain decode tick wearing the spec API's
        // clothes. Ride the captured-graph decode path instead of paying
        // this fn's eager per-round launch cost for zero draft benefit:
        // that eager cost is real (measured ~2x slower live on non-repeating
        // prompts) and only buys anything back when a draft actually lands.
        if reqs.iter().all(|(_, _, c)| c.len() == 1) {
            let toks: Vec<u32> = reqs.iter().map(|(_, _, c)| c[0]).collect();
            let positions: Vec<u32> = reqs.iter().map(|&(_, p, _)| p as u32).collect();
            let slots: Vec<u32> = reqs.iter().map(|&(s, _, _)| s as u32).collect();
            self.batch_step_slots(&toks, &positions, &slots)?;
            let exec = self.exec.clone();
            let vocab = self.hp.n_vocab;
            let bs = self.batch.as_mut().expect("batch enabled");
            exec.argmax_rows(&bs.sc.head_logits, &mut bs.d_spec_pick, r, vocab)?;
            let view = bs
                .d_spec_pick
                .try_slice(0..r)
                .ok_or_else(|| GpuError::Driver("spec pick slice out of range".into()))?;
            return Ok(exec.stream.clone_dtoh(&view).map_err(drv)?);
        }
        let mut chunk: Vec<(u32, u32, u32)> = Vec::with_capacity(r);
        let mut all_slots: Vec<u32> = Vec::with_capacity(r);
        let mut all_pos: Vec<u32> = Vec::with_capacity(r);
        for &(slot, start_pos, ref toks) in reqs {
            if slot >= n_slots {
                return Err(GpuModelError::BatchTooLarge {
                    got: slot + 1,
                    max: n_slots,
                });
            }
            if toks.is_empty() || start_pos + toks.len() > self.max_ctx {
                return Err(GpuModelError::BatchTooLarge {
                    got: start_pos + toks.len(),
                    max: self.max_ctx,
                });
            }
            for (i, &t) in toks.iter().enumerate() {
                let pos = (start_pos + i) as u32;
                chunk.push((slot as u32, pos, t));
                all_slots.push(slot as u32);
                all_pos.push(pos);
            }
        }
        // Grow the pool for the whole draft span before the walk reads the
        // block table - a slot's draft rows can cross a block boundary that
        // its last COMMITTED token never touched (gpt-oss's P3 lesson: skip
        // this and a post-boundary row aliases block 0, corrupting that
        // slot's first-block KV instead of just mis-predicting).
        self.ensure_rows(&all_slots, &all_pos)?;
        self.rows_pass_body(&chunk, 0)?;
        self.head_rows(r)?;
        let exec = self.exec.clone();
        let vocab = self.hp.n_vocab;
        let bs = self.batch.as_mut().expect("batch enabled");
        exec.argmax_rows(&bs.sc.head_logits, &mut bs.d_spec_pick, r, vocab)?;
        let view = bs
            .d_spec_pick
            .try_slice(0..r)
            .ok_or_else(|| GpuError::Driver("spec pick slice out of range".into()))?;
        Ok(exec.stream.clone_dtoh(&view).map_err(drv)?)
    }

    // ── decode ──────────────────────────────────────────────────────────────

    /// One batched decode step over rows 0..r (row i drives slot i - the
    /// engine's identity contract). Leaves [r, vocab] logits in head_logits.
    pub(crate) fn batch_step(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
    ) -> Result<(), GpuModelError> {
        let ident: Vec<u32> = (0..tokens.len() as u32).collect();
        self.batch_step_slots(tokens, positions, &ident)
    }

    /// `batch_step` with explicit slot ids - the mixed tick's decode half.
    /// The identity mapping only holds when the live set is a dense prefix;
    /// under stall-free batching the decoders are whichever slots finished
    /// prefilling, so rows must be compacted and carry their real slot.
    pub(crate) fn batch_step_slots(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        slots: &[u32],
    ) -> Result<(), GpuModelError> {
        let r = tokens.len();
        assert_eq!(r, positions.len());
        assert_eq!(r, slots.len());
        let n_slots = self.batch.as_ref().expect("batch enabled").n_slots;
        assert!(r <= n_slots, "rows {r} > enabled {n_slots}");
        self.ensure_rows(slots, positions)?;
        self.upload_rows(tokens, positions, slots)?;
        self.step_replay(r)
    }

    /// The pure-device decode tick body - everything the per-r graph captures.
    /// All inputs are device buffers written before replay (d_toks/d_pos/
    /// d_slots + the block tables); all shapes depend only on r and model
    /// constants.
    fn step_body(&mut self, r: usize) -> Result<(), GpuModelError> {
        // No spans, unconditionally: every image row is consumed during
        // prefill, so a decode tick never carries one. That also keeps the
        // captured graph byte-identical to the text-only case - a conditional
        // injection inside the captured body would be a capture-time constant
        // baked into a graph replayed for other requests.
        self.embed_rows(r, &[])?;
        self.layer_walk(r, None, &[])?;
        self.head_rows(r)
    }

    /// Record `body`'s launches into a CUDA graph (recording only - nothing
    /// executes). An alloc during capture is a hard driver error, which is
    /// why every plane is allocated at enable.
    fn capture_body(
        &mut self,
        body: impl FnOnce(&mut Self) -> Result<(), GpuModelError>,
        what: &str,
    ) -> Result<super::SendGraph, GpuModelError> {
        let exec = self.exec.clone();
        exec.stream
            .synchronize()
            .map_err(|e| GpuError::Driver(format!("{what} pre-capture sync: {e}")))?;
        exec.stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|e| GpuError::Driver(format!("{what} begin_capture: {e}")))?;
        let rec = body(self);
        let graph = crate::gpu::end_capture_no_flags(&exec.stream)
            .map_err(|e| GpuError::Driver(format!("{what} end_capture: {e}")));
        rec?; // surface a record failure only after capture is cleanly ended
        let graph =
            graph?.ok_or_else(|| GpuError::Driver(format!("{what} capture produced no graph")))?;
        Ok(super::SendGraph(graph))
    }

    /// Replay the fixed-r decode tick, capturing it first if unseen.
    fn step_replay(&mut self, r: usize) -> Result<(), GpuModelError> {
        if !self
            .batch
            .as_ref()
            .expect("batch enabled")
            .graphs
            .contains_key(&r)
        {
            let g = self.capture_body(|s| s.step_body(r), "decode")?;
            self.batch
                .as_mut()
                .expect("batch enabled")
                .graphs
                .insert(r, g);
        }
        self.batch.as_ref().expect("batch enabled").graphs[&r]
            .0
            .launch()
            .map_err(|e| GpuError::Driver(format!("decode graph launch: {e}")))?;
        Ok(())
    }

    /// The decode-pipe sampler kernels for `r` rows at ring `ring`, reading the
    /// per-row params/trunc planes and writing the sampled ids into the same
    /// ring - the exact calls the eager block in `pipe_launch_tick` made, in
    /// the same order, so the sampled distribution is unchanged. Captured by
    /// `sampler_replay` into a replayable graph.
    fn sampler_body(
        &mut self,
        r: usize,
        ring: usize,
        has_trunc: bool,
    ) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let vocab = self.hp.n_vocab;
        let max_batch = self.batch.as_ref().expect("batch enabled").n_slots;
        let off = ring * max_batch * 4;
        let oring = ring * max_batch;
        let sc = &mut self.batch.as_mut().expect("batch enabled").sc;
        exec.sample_rows_at(
            &sc.head_logits,
            &sc.d_pipe_par,
            off,
            &mut sc.d_pipe_out,
            oring,
            r,
            vocab,
        )?;
        if has_trunc {
            exec.sample_rows_t_at(
                &sc.head_logits,
                &sc.d_pipe_par,
                off,
                &sc.d_pipe_tpar,
                off,
                &mut sc.d_pipe_out,
                oring,
                r,
                vocab,
            )?;
            exec.sample_rows_p_at(
                &sc.head_logits,
                &sc.d_pipe_par,
                off,
                &sc.d_pipe_tpar,
                off,
                &mut sc.d_pipe_out,
                oring,
                r,
                vocab,
            )?;
        }
        Ok(())
    }

    /// Replay the sampler chain from a captured graph instead of the ~11 eager
    /// launches. The first sampler tick per process runs EAGER (warms the
    /// `pd_topp_mb_*` scratch - see `sampler_warm`); after that each distinct
    /// (rows, ring, has_trunc) captures once and replays. The fresh-`u` upload
    /// stays eager in the caller before this runs, so the replay draws with a
    /// new uniform each tick.
    fn sampler_replay(
        &mut self,
        r: usize,
        ring: usize,
        has_trunc: bool,
    ) -> Result<(), GpuModelError> {
        if has_trunc {
            Self::trunc_dev_witness(r);
        }
        let key = (r, ring, has_trunc);
        if !self
            .batch
            .as_ref()
            .expect("batch enabled")
            .sampler_graphs
            .contains_key(&key)
        {
            if !self.batch.as_ref().expect("batch enabled").sampler_warm {
                // Warm the topp_mb scratch eagerly; capture starts next time.
                self.sampler_body(r, ring, has_trunc)?;
                self.batch.as_mut().expect("batch enabled").sampler_warm = true;
                return Ok(());
            }
            let g = self.capture_body(|s| s.sampler_body(r, ring, has_trunc), "sampler")?;
            self.batch
                .as_mut()
                .expect("batch enabled")
                .sampler_graphs
                .insert(key, g);
        }
        self.batch.as_ref().expect("batch enabled").sampler_graphs[&key]
            .0
            .launch()
            .map_err(|e| GpuError::Driver(format!("sampler graph launch: {e}")))?;
        Ok(())
    }

    /// Host->device row streams: tokens, positions, slots.
    fn upload_rows(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        slots: &[u32],
    ) -> Result<(), GpuModelError> {
        let r = tokens.len();
        let bs = self.batch.as_mut().expect("batch enabled");
        let sc = &mut bs.sc;
        let st = &self.exec.stream;
        let mut t = sc
            .d_toks
            .try_slice_mut(0..r)
            .ok_or_else(|| GpuError::Driver("d_toks".into()))?;
        st.memcpy_htod(tokens, &mut t).map_err(drv)?;
        let mut p = sc
            .d_pos
            .try_slice_mut(0..r)
            .ok_or_else(|| GpuError::Driver("d_pos".into()))?;
        st.memcpy_htod(positions, &mut p).map_err(drv)?;
        let mut s = sc
            .d_slots
            .try_slice_mut(0..r)
            .ok_or_else(|| GpuError::Driver("d_slots".into()))?;
        st.memcpy_htod(slots, &mut s).map_err(drv)?;
        Ok(())
    }

    /// Gather the rows' embeddings and apply granite's `embedding_scale`
    /// (12.0) - llama.cpp applies it inside build_inp_embd, and dropping it
    /// produces fluent, wrong text with no error.
    fn embed_rows(&mut self, r: usize, spans: &[InjectSpan]) -> Result<(), GpuModelError> {
        let embd = self.hp.n_embd;
        let scale = self.hp.embedding_scale;
        let bs = self.batch.as_mut().expect("batch enabled");
        let sc = &mut bs.sc;
        match &self.tok_embd {
            TokEmbd::Q8(t) if t.ty == paddock_models::ggml_type::GgmlType::Bf16 => self
                .exec
                .embed_gather_bf16(t, &sc.d_toks, &mut sc.x, embd, r, 1.0)?,
            TokEmbd::Q8(t) => self
                .exec
                .embed_gather_batch_q8(t, &sc.d_toks, &mut sc.x, embd, r)?,
            TokEmbd::Kq(t) => self.exec.kquant_gather(t, &sc.d_toks, &mut sc.x, embd, r)?,
        }
        // Where the media rows enter relative to `embedding_scale` is a
        // per-family fact, and getting it wrong is silent in both directions.
        //
        // granite-VISION: the scale is for token embeddings only, so the
        // DeepStack streams overwrite the already-scaled placeholder.
        // llama.cpp gates the multiply off entirely for an embedding ubatch
        // (`if (f_embedding_scale != 0 && (ubatch.token || n_deepstack == 0))`).
        // Scale them and every image row is 12x too large - fluent, wrong.
        //
        // granite-SPEECH: upstream merges the projector output into
        // `inputs_embeds` and the LLM multiplies AFTERWARDS -
        // `modeling_granite_speech.py`'s `masked_scatter` feeds
        // `modeling_granite.py`'s `inputs_embeds * embedding_multiplier` - so
        // audio rows do take the 12x. Skipping it makes them 12x too small,
        // which is not fluent at all: measured on the battery, every clip came
        // back a degenerate repeat loop ("[1] [2] [3] ...", "＜＜＜...").
        let scale_media = self.audio.is_some();
        if scale_media && !spans.is_empty() {
            deepstack::apply_embed(&self.exec, &mut sc.x, spans, embd)?;
        }
        self.exec.scale(&mut sc.x, scale, r * embd)?;
        if !scale_media && !spans.is_empty() {
            deepstack::apply_embed(&self.exec, &mut sc.x, spans, embd)?;
        }
        Ok(())
    }

    /// The whole-stack walk over r rows. `cuts`: Some = prefill mode (append
    /// the whole chunk, attend per same-slot run); None = decode mode (every
    /// row is one new token of its slot).
    fn layer_walk(
        &mut self,
        r: usize,
        cuts: Option<&PfCuts>,
        spans: &[InjectSpan],
    ) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let hp = &self.hp;
        let (embd, nh, n_kv, hd, n_ff) =
            (hp.n_embd, hp.n_heads, hp.n_kv_heads, hp.head_dim, hp.n_ff);
        let kv_dim = n_kv * hd;
        let q_dim = nh * hd;
        let eps = hp.eps;
        // Granite's own KQ scale - attention.scale (1/128), not 1/sqrt(128).
        let scale = hp.attention_scale;
        let res_s = hp.residual_scale;
        let rope = hp.rope;
        let kv_dtype = self.kv_dtype;
        let bs = self.batch.as_mut().expect("batch enabled");
        let bps = bs.bps;

        // Rung election, keyed on MODE and never on r: a warm-resume tail must
        // reproduce the cold chunk's bytes exactly, so every prefill row -
        // 1024-row chunk or 3-row tail - takes the same rungs. r==1 decode
        // rides the serial lane's exact-f32 GEMVs (the mmq rungs only win once
        // r > 1); decode keeps the strided ladder so the captured graph body
        // stays untouched.
        let pf = cuts.is_some() && !no_w4a8_prefill();
        let r1 = r == 1 && !pf;
        // r=1 k-quant weights take the W4A8 serving GEMV instead of the
        // exact-f32 oracle. `gemv_any`'s Q8 arm was always the tuned kernel,
        // so this gap only ever showed on k-quant files - and it showed big:
        // 93.9% of granite-30b Q4_K_M decode GPU time was pd_kquant_gemv.
        //
        // Gated on the FILE actually holding k-quant weights, not just on the
        // pack carrying the kernel: gemv8_any sends Q8_0 back to the exact
        // GEMV, so on a pure-Q8_0 file (the 8b) the staging quantize would be
        // 4 launches per layer feeding nothing. The 8b's decode graph stays
        // byte-for-byte what it was.
        let kq_file = self.layers.first().is_some_and(|l| {
            l.gate
                .as_ref()
                .is_some_and(|g| matches!(g.quant(), Some(QuantW::Kq(_))))
                || matches!(l.wq.quant(), Some(QuantW::Kq(_)))
        });
        let dw8 = r1 && kq_file && exec.has_kquant_gemv_w4a8() && !no_w4a8_decode();
        // tile-linear f8 checkpoint class: r=1 seats take the lin GEMV
        // (gemv_f8lin, partials plane in hand) instead of the class-less
        // serial gemv - see GraniteW::F8Lin.
        let f8l_file = self
            .layers
            .first()
            .is_some_and(|l| matches!(l.wq, GraniteW::F8Lin { .. }));
        // fold scale_add + rmsnorm_batch + quantize_q8_sums into one
        // launch. Only where the Q8 staging is actually wanted (dw8), because
        // the fusion's whole point is emitting that staging from the values
        // already in registers - without a consumer it is just a norm.
        // Rides dw8's r1, so batch is always 1 here and `embd == r * embd`.
        // Bit-exact against the three (gpu_add_rmsnorm_q8), so there is no
        // byte-invariance seam - but it still gets a kill, because a lane
        // nobody can turn off is a lane nobody can MEASURE, and every other
        // election in this file carries one for exactly that reason.
        let fuse_norm = dw8 && exec.has_add_rmsnorm_q8_xn() && !no_fuse_norm();
        // WMMA (tensor-core) prefill attention: one class for every prefill
        // span at any length - a len-keyed prefill/decode kernel switch would
        // be another byte-invariance seam.
        //
        // 64 as well as 128 because granite-vision-4.1-4b is head_dim 64
        // (40 q / 8 kv × 64) where the text 4.1 sizes are 128. The pack's f16
        // paged prefill instantiates 64/128/256/512, so 64 rides the same
        // tensor-core class; the TILED `attn_prefill_rows_paged` below does
        // not (128/256/512 only) and refuses head_dim 64 outright - see
        // `attend_span`.
        let wmma_pf = cuts.is_some()
            // hd 64 only on f16 KV: the pack's fp8 arms of this entry cover
            // 512 and 256 shapes, so a 64-wide fp8 tile would be refused.
            && (hd == 128 || (hd == 64 && matches!(kv_dtype, KvDtype::Fp16)))
            && pf_attn_dtype_ok(kv_dtype, nh, n_kv)
            && exec.has_attn_prefill_f16_paged()
            && !no_wmma_prefill();
        // Batched-runs prefill attention: the pack carries the v4 kernel's
        // grid.z run table (its prologue in the sm_120-safe index form) and
        // the launcher honors it whenever pd_pf_runs_register is armed;
        // granite used to loop attend_span once per prompt run -- 8 ms of a
        // 1024-row mixed tick burned in 16 us launches, where one varlen
        // launch per layer will do. Arm the tick's
        // prefill runs once (row-offset prefix), attend them in one launch
        // per layer, disarm after the walk. Per-run CTAs compute exactly the
        // per-run launch's work (bit-identical by the kernel's contract).
        // Kill: PADDOCK_NO_PF_RUNS.
        if exec.kernels_pf_runs_available() {
            exec.pf_runs_register(None)?; // never trust a stale table
        }
        let runs_armed: bool = if let Some(c) = cuts {
            let pfr: Vec<(usize, usize)> = c
                .runs
                .iter()
                .copied()
                .filter(|&(off, _)| off >= c.dec)
                .collect();
            let contiguous = pfr.windows(2).all(|w| w[0].0 + w[0].1 == w[1].0);
            if wmma_pf
                && pfr.len() >= 2
                && contiguous
                && exec.kernels_pf_runs_available()
                && paddock_models::dev_var_os!("PADDOCK_NO_PF_RUNS").is_none()
            {
                let mut offs: Vec<u32> = pfr.iter().map(|&(off, _)| off as u32).collect();
                let last = pfr[pfr.len() - 1];
                offs.push((last.0 + last.1) as u32);
                let n = offs.len();
                let mut v = bs
                    .sc
                    .pf_runs
                    .try_slice_mut(0..n)
                    .ok_or_else(|| GpuError::Driver("pf_runs slice".into()))?;
                exec.stream.memcpy_htod(&offs, &mut v).map_err(drv)?;
                let maxn = pfr.iter().map(|&(_, n)| n).max().unwrap_or(0) as u32;
                exec.pf_runs_register(Some((&bs.sc.pf_runs, pfr.len() as u32, maxn)))?;
                true
            } else {
                false
            }
        } else {
            false
        };

        // Per-layer vision taps. Indexing the mapping by layer (rather than
        // walking a list of targets) keeps this a single lookup in the hot loop
        // and is how the file stores it.
        let ds_map: &[i32] = &self.hp.deepstack;

        // nvf4 decode fold-2: on the pure-decode tick the down
        // GEMM leaves RAW split-K partials that the next layer's attn-norm
        // folds (residual + norm + nvf4 quant, one launch), the o-proj fold
        // also quantizes for gate|up, and gate|up's partials go straight
        // through swiglu into the down input -- six launches and two wide f32
        // round trips per layer gone, every kernel bit-identical to the chain
        // it replaces (bench/nv4_fold2_cmp.cu). Kill: PADDOCK_NO_NV4_FOLD2.
        let fold2 = !pf
            && r > 1
            && !fuse_norm
            && super::ops::nvf4_prestaged_ok(&exec, r)
            && exec.has_add_rmsnorm_quant_nvf4_from_parts()
            && exec.has_swiglu_quant_nvf4_from_parts()
            && paddock_models::dev_var_os!("PADDOCK_NO_NV4_FOLD2").is_none();
        // (nz, scale2) of the previous layer's down partials still in sc.part
        let mut pend_down: Option<(u32, f32)> = None;

        for (li, layer) in self.layers.iter().enumerate() {
            let sc = &mut bs.sc;
            let mut attn_staged = false;
            let mut resid_done = false;
            // Before the layer runs - i.e. before attn_norm reads x. Additive
            // into the residual stream, never a scatter-replace.
            if !spans.is_empty()
                && let Some(k) = ds_map.get(li).copied().filter(|&k| k >= 0)
            {
                deepstack::apply_layer(&exec, &mut sc.x, spans, embd, k as usize)?;
            }
            // The entry norm has no residual to fold, so the fusion runs with
            // proj = None (res_scale then unused) - still two launches down to
            // one.
            let f8row_qkv = super::ops::is_f8row(&layer.wq)
                || super::ops::is_f8row(&layer.wk)
                || super::ops::is_f8row(&layer.wv);
            let mut nq_attn = false;
            if let (Some((nz, s2)), Some(_)) = (pend_down.take(), layer.qkv_nv4.as_ref()) {
                // F1: previous down's partials -> residual -> attn-norm -> nvf4
                // pair for the qkv seat (the rmsnorm_batch accumulator family)
                exec.add_rmsnorm_quant_nvf4_from_parts(
                    &mut sc.x,
                    &sc.part,
                    &layer.attn_norm.buf,
                    Some(&mut sc.xn),
                    None,
                    &mut sc.xq,
                    &mut sc.xs4,
                    embd,
                    eps,
                    r,
                    res_s,
                    s2,
                    nz,
                    1,
                )?;
                attn_staged = true;
            } else if fuse_norm {
                exec.add_rmsnorm_q8_xn(
                    &mut sc.x,
                    None,
                    &layer.attn_norm.buf,
                    &mut sc.xn,
                    &mut sc.xq,
                    &mut sc.xs,
                    &mut sc.ssums,
                    embd,
                    r,
                    eps,
                    1.0,
                )?;
            } else {
                // fp8 lane: the norm writes xn AND the e4m3 row
                // pair in one launch (bit-identical to rmsnorm_batch +
                // quantize_e4m3_row); the group quantize below then skips.
                // Kill: PADDOCK_NO_F8R_NQ.
                nq_attn = f8row_qkv
                    && paddock_models::dev_var_os!("PADDOCK_NO_F8R_NQ").is_none()
                    && exec.rmsnorm_quant_e4m3_row(
                        &sc.x,
                        &layer.attn_norm.buf,
                        &mut sc.xn,
                        &mut sc.xq,
                        &mut sc.xs,
                        embd,
                        eps,
                        r,
                    )?;
                if !nq_attn {
                    exec.rmsnorm_batch(&sc.x, &layer.attn_norm.buf, &mut sc.xn, embd, eps, r)?;
                }
            }
            // fused q|k|v NVFP4 plane: one staging, one GEMM over
            // the [q|k|v] plane, every width; the rope site below folds the
            // result (raw K-split partials or the finished plane) and appends.
            // The split-plane arms after this are the GGUF/fp8/bf16 classes.
            let nv4_qkv = match layer.qkv_nv4.as_ref() {
                Some(qkv) => Some(super::ops::nvf4_qkv_into(
                    &exec,
                    qkv,
                    &sc.xn,
                    &mut sc.xq,
                    &mut sc.xs4,
                    &mut sc.part,
                    &mut sc.ffn_gu,
                    r,
                    attn_staged,
                )?),
                None => None,
            };
            // one quantize serves wq/wk/wv (they all read xn - the group
            // dedupe; without it the same bytes quantize three times)
            if nv4_qkv.is_some() {
                // projections done; consumed at the rope site
            } else if dw8 {
                // one staging serves wq/wk/wv - same dedupe the mmq arm below
                // makes, for the same reason. Already emitted when the norm
                // fused, straight out of registers.
                if !fuse_norm {
                    exec.quantize_q8_sums(&sc.xn, &mut sc.xq, &mut sc.xs, &mut sc.ssums, embd)?;
                }
                // q|k|v as one launch - the Q8_0 multi merge's economics on
                // the k-quant family: the split k/v grids are latency-floored
                // at ~6.1 us regardless of bytes (256 blocks on a die seating
                // 2256), and every boundary breaks the PDL cascade. Mixed
                // dtypes are fine (Q4_K_M pairs Q4_K q/k with Q6_K v).
                match super::ops::tri_quant(&layer.wq, &layer.wk, &layer.wv) {
                    Some((QuantW::Kq(wq), QuantW::Kq(wk), QuantW::Kq(wv)))
                        if exec.has_kquant_gemv_w4a8_multi() && !no_gemv_multi() =>
                    {
                        exec.kquant_gemv_w4a8_multi(
                            &mut [(wq, &mut sc.q), (wk, &mut sc.k), (wv, &mut sc.v)],
                            &sc.xq,
                            &sc.xs,
                            &sc.ssums,
                        )?;
                    }
                    _ => {
                        super::ops::gemv8(
                            &exec, &layer.wq, &sc.xn, &mut sc.xq, &mut sc.xs, &sc.ssums, &mut sc.q,
                        )?;
                        super::ops::gemv8(
                            &exec, &layer.wk, &sc.xn, &mut sc.xq, &mut sc.xs, &sc.ssums, &mut sc.k,
                        )?;
                        super::ops::gemv8(
                            &exec, &layer.wv, &sc.xn, &mut sc.xq, &mut sc.xs, &sc.ssums, &mut sc.v,
                        )?;
                    }
                }
            } else if r1 && f8l_file {
                // One e4m3 stage serves wq/wk/wv (group dedup - 4 quantizes
                // per layer instead of 7)
                exec.quantize_e4m3(&sc.xn, &mut sc.xq, &mut sc.xs4, embd)?;
                super::ops::gemv_f8lin(
                    &exec,
                    &layer.wq,
                    &sc.xn,
                    &mut sc.xq,
                    &mut sc.xs4,
                    &mut sc.part,
                    &mut sc.q,
                )?;
                super::ops::gemv_f8lin(
                    &exec,
                    &layer.wk,
                    &sc.xn,
                    &mut sc.xq,
                    &mut sc.xs4,
                    &mut sc.part,
                    &mut sc.k,
                )?;
                super::ops::gemv_f8lin(
                    &exec,
                    &layer.wv,
                    &sc.xn,
                    &mut sc.xq,
                    &mut sc.xs4,
                    &mut sc.part,
                    &mut sc.v,
                )?;
            } else if r1 {
                // Q8_0 q|k|v as one launch: the split 1024-row k/v grids
                // stream at only ~47% of the die's practical read ceiling
                // (launch ramp/drain, not the inner loop - see the pack's
                // multi-kernel note), merged 6144 rows recover ~6 us/layer.
                // Per-row bytes identical to the split launches.
                match super::ops::tri_quant(&layer.wq, &layer.wk, &layer.wv) {
                    Some((QuantW::Q8(wq), QuantW::Q8(wk), QuantW::Q8(wv)))
                        if exec.has_q8_0_gemv_repacked_multi() && !no_gemv_multi() =>
                    {
                        exec.q8_0_gemv_repacked_multi(
                            &mut [(wq, &mut sc.q), (wk, &mut sc.k), (wv, &mut sc.v)],
                            &sc.xn,
                        )?;
                    }
                    _ => {
                        super::ops::gemv_qkv(
                            &exec, &layer.wq, &layer.wk, &layer.wv, &sc.xn, &mut sc.q, &mut sc.k,
                            &mut sc.v,
                        )?;
                    }
                }
            } else if pf {
                // int8/yq staging only for int8-class consumers - the f8
                // arms restage e4m3 in place (same hoist as the decode arms)
                if super::ops::is_quant(&layer.wq)
                    || super::ops::is_quant(&layer.wk)
                    || super::ops::is_quant(&layer.wv)
                {
                    prefill_quant_w(&exec, &mut sc.xq, &mut sc.xs, &mut sc.yq, &sc.xn, embd, r)?;
                } else if super::ops::is_f8w(&layer.wq)
                    || super::ops::is_f8w(&layer.wk)
                    || super::ops::is_f8w(&layer.wv)
                {
                    exec.quantize_e4m3(&sc.xn, &mut sc.xq, &mut sc.xs4, r * embd)?;
                } else if (super::ops::is_f8row(&layer.wq)
                    || super::ops::is_f8row(&layer.wk)
                    || super::ops::is_f8row(&layer.wv))
                    && !nq_attn
                {
                    exec.quantize_e4m3_row(&sc.xn, &mut sc.xq, &mut sc.xs, embd, r)?;
                }
                // pf-side fused qkv (f8row): one GEMM over the q|k|v-concat
                // plane into ffn_gu scratch (consumed by the rope-only twin
                // before the FFN touches it); replaces three GEMMs - wk/wv's
                // 16-tile grids are the wave band's weakest shapes - and the
                // separate rope+append launch at the rope site below.
                let pf_qkv_fused = layer.qkv_f8.is_some()
                    && r >= 2
                    && exec.has_f8row_qkv_rope_from_y()
                    && !no_rope_fuse();
                if pf_qkv_fused {
                    if let (Some(qkv), GraniteW::Fp8 { .. }) = (&layer.qkv_f8, &layer.wq) {
                        let out6 = qkv.scale.len();
                        super::ops::f8row_mm_into(
                            &exec,
                            qkv,
                            embd,
                            out6,
                            &sc.xq,
                            &sc.xs,
                            &mut sc.ffn_gu,
                            r,
                        )?;
                    }
                } else {
                    super::ops::pf_mm(
                        &exec,
                        &layer.wq,
                        &sc.xn,
                        &mut sc.xq,
                        &mut sc.xs,
                        &mut sc.xs4,
                        &sc.yq,
                        &mut sc.xsums,
                        &mut sc.ssums,
                        &mut sc.skfix,
                        &mut sc.part,
                        &mut sc.q,
                        r,
                    )?;
                    super::ops::pf_mm(
                        &exec,
                        &layer.wk,
                        &sc.xn,
                        &mut sc.xq,
                        &mut sc.xs,
                        &mut sc.xs4,
                        &sc.yq,
                        &mut sc.xsums,
                        &mut sc.ssums,
                        &mut sc.skfix,
                        &mut sc.part,
                        &mut sc.k,
                        r,
                    )?;
                    super::ops::pf_mm(
                        &exec,
                        &layer.wv,
                        &sc.xn,
                        &mut sc.xq,
                        &mut sc.xs,
                        &mut sc.xs4,
                        &sc.yq,
                        &mut sc.xsums,
                        &mut sc.ssums,
                        &mut sc.skfix,
                        &mut sc.part,
                        &mut sc.v,
                        r,
                    )?;
                }
            } else {
                // int8 staging only when an int8-class consumer exists - on an
                // all-fp8/bf16 checkpoint this quantize fed nothing (measured
                // c32 fp8-native: 161 dead launches/tick = 0.82ms of ITL); the
                // mm_pre Fp8 arm's own comment prescribes exactly this hoist.
                if super::ops::is_quant(&layer.wq)
                    || super::ops::is_quant(&layer.wk)
                    || super::ops::is_quant(&layer.wv)
                {
                    exec.quantize_q8(&sc.xn, &mut sc.xq, &mut sc.xs, r * embd)?;
                } else if super::ops::is_f8w(&layer.wq)
                    || super::ops::is_f8w(&layer.wk)
                    || super::ops::is_f8w(&layer.wv)
                {
                    exec.quantize_e4m3(&sc.xn, &mut sc.xq, &mut sc.xs4, r * embd)?;
                } else if (super::ops::is_f8row(&layer.wq)
                    || super::ops::is_f8row(&layer.wk)
                    || super::ops::is_f8row(&layer.wv))
                    && !nq_attn
                {
                    exec.quantize_e4m3_row(&sc.xn, &mut sc.xq, &mut sc.xs, embd, r)?;
                }
                // fused wqkv (f8row): the three GEMMs + rope + append run as
                // one mma + one combine at the rope site below - skip the
                // separate projections here. Same staged e4m3-row pair.
                if !(layer.qkv_f8.is_some()
                    && (2..=64).contains(&r)
                    && exec.has_f8row_qkv_rope_norm_paged()
                    && !no_rope_fuse())
                {
                    super::ops::mm_pre(
                        &exec,
                        &layer.wq,
                        &sc.xn,
                        &mut sc.xq,
                        &mut sc.xs,
                        &mut sc.xs4,
                        &mut sc.ssums,
                        &mut sc.part,
                        &mut sc.q,
                        r,
                    )?;
                    super::ops::mm_pre(
                        &exec,
                        &layer.wk,
                        &sc.xn,
                        &mut sc.xq,
                        &mut sc.xs,
                        &mut sc.xs4,
                        &mut sc.ssums,
                        &mut sc.part,
                        &mut sc.k,
                        r,
                    )?;
                    super::ops::mm_pre(
                        &exec,
                        &layer.wv,
                        &sc.xn,
                        &mut sc.xq,
                        &mut sc.xs,
                        &mut sc.xs4,
                        &mut sc.ssums,
                        &mut sc.part,
                        &mut sc.v,
                        r,
                    )?;
                }
            }
            // NORM-convention rope (interleaved pairs) - granite is
            // llama.cpp's LLAMA_ROPE_TYPE_NORM while every other family here
            // is NEOX. Wrong convention = fluent, confidently wrong text that
            // degrades with position. In place, like the serial path: granite
            // has no q/k norm to write a second plane for.
            //
            // The fused arm folds rope(q)+rope(k)+append(k)+append(v) - four
            // 1.3-3.4 us latency-bound launches - into one kernel; roped k
            // goes straight to the pool (nothing reads sc.k after the
            // append). Safe here precisely because granite has no SWA
            // (window 0 on every layer): gemma4's ring-shrink sub-span
            // append contract is what keeps its appends separate.
            let kvs = &mut bs.kv[li];
            let qkv_fused = layer.qkv_f8.is_some()
                && !pf
                && !r1
                && !dw8
                && (2..=64).contains(&r)
                && exec.has_f8row_qkv_rope_norm_paged()
                && !no_rope_fuse();
            let pf_qkv_fused = pf
                && layer.qkv_f8.is_some()
                && r >= 2
                && exec.has_f8row_qkv_rope_from_y()
                && !no_rope_fuse();
            if let Some((nz, pscale, in_part)) = nv4_qkv {
                // NVFP4 fused q|k|v: fold the GEMM's raw K-split slices (or read
                // the finished plane) + NORM-rope + paged append in one launch.
                // Same fixed-order sum and the same post-fold scale2 as the
                // reduce kernel it replaces, so per element this is the split
                // path's arithmetic; what changed is one launch and one y round
                // trip per layer.
                let src: &CudaSlice<f32> = if in_part { &sc.part } else { &sc.ffn_gu };
                exec.qkv_rope_norm_from_parts_paged(
                    src,
                    nz,
                    pscale,
                    &mut sc.q,
                    &mut kvs.k,
                    &mut kvs.v,
                    &sc.d_pos,
                    Some(&sc.d_slots),
                    &bs.d_bt,
                    bps,
                    nh,
                    n_kv,
                    hd,
                    rope,
                    r,
                    kv_dtype,
                )?;
            } else if pf_qkv_fused {
                // the fused plane sits in ffn_gu (written by the pf qkv seat
                // above); combine+NORM-rope+append in one launch
                exec.f8row_qkv_rope_from_y_paged(
                    &sc.ffn_gu,
                    &mut sc.q,
                    &mut kvs.k,
                    &mut kvs.v,
                    &sc.d_pos,
                    Some(&sc.d_slots),
                    &bs.d_bt,
                    bps,
                    nh,
                    n_kv,
                    hd,
                    rope,
                    r,
                    kv_dtype,
                )?;
            } else if qkv_fused {
                // decode-band fused qkv on the GEMM launcher's own decode arm
                // the K-split mma1 streamed the 6144-out concat
                // plane at 23.4 us/layer on the 30b tick; pd_f8row_gemm now
                // elects the producer-warp TMA ring there (~20 us cold). The
                // GEMM writes the fused plane UNSPLIT into ffn_gu (the nz=1
                // partials layout) and the rope-only twin consumes it, so the
                // combine folds away -- the pf seat's exact route, at decode
                // width. Numerics = the unsplit (nz=1) class the K-split path
                // was gated against. Kill: PADDOCK_NO_F8R_QKV_TW.
                // r >= 17 only: at r=8 the 6144-out plane is 96 CTAs on the
                // unsplit ring, too few to fill the die, while the K-split
                // path does; at r=32 the ring wins.
                let qkv_tw = (17..=64).contains(&r)
                    && paddock_models::dev_var_os!("PADDOCK_NO_F8R_QKV_TW").is_none()
                    && exec.has_f8row_qkv_rope_from_y()
                    && match (&layer.qkv_f8, &layer.wq) {
                        (Some(qkv), GraniteW::Fp8 { .. }) => {
                            let out6 = qkv.scale.len();
                            super::ops::f8row_mm_into(
                                &exec,
                                qkv,
                                embd,
                                out6,
                                &sc.xq,
                                &sc.xs,
                                &mut sc.ffn_gu,
                                r,
                            )?;
                            true
                        }
                        _ => false,
                    };
                if qkv_tw {
                    exec.f8row_qkv_rope_from_y_paged(
                        &sc.ffn_gu,
                        &mut sc.q,
                        &mut kvs.k,
                        &mut kvs.v,
                        &sc.d_pos,
                        Some(&sc.d_slots),
                        &bs.d_bt,
                        bps,
                        nh,
                        n_kv,
                        hd,
                        rope,
                        r,
                        kv_dtype,
                    )?;
                } else {
                    // One mma over the q|k|v concat plane + one combine+NORM-rope
                    // +append: replaces three underfilled GEMMs (wq 64 / wk 16 /
                    // wv 16 tiles = 25.2 MB at ~787 GB/s effective) and the rope
                    // launch. Consumes the seat's staged e4m3-row pair; roped q
                    // lands in sc.q for attention, k/v go straight to the pools.
                    exec.f8row_qkv_rope_norm_paged(
                        layer.qkv_f8.as_ref().expect("qkv_fused"),
                        embd,
                        &sc.xq,
                        &sc.xs,
                        &mut sc.part,
                        &mut sc.q,
                        &mut kvs.k,
                        &mut kvs.v,
                        &sc.d_pos,
                        Some(&sc.d_slots),
                        &bs.d_bt,
                        bps,
                        nh,
                        n_kv,
                        hd,
                        rope,
                        r,
                        kv_dtype,
                    )?;
                }
            } else if exec.has_rope_norm_qk_append_paged() && !no_rope_fuse() {
                exec.rope_norm_qk_append_paged(
                    &mut sc.q,
                    &mut sc.k,
                    &sc.v,
                    &mut kvs.k,
                    &mut kvs.v,
                    &sc.d_pos,
                    Some(&sc.d_slots),
                    &bs.d_bt,
                    bps,
                    nh,
                    n_kv,
                    hd,
                    rope,
                    r,
                    kv_dtype,
                )?;
            } else {
                exec.rope_yarn_batch_norm(&mut sc.q, &sc.d_pos, nh, hd, rope, r)?;
                exec.rope_yarn_batch_norm(&mut sc.k, &sc.d_pos, n_kv, hd, rope, r)?;
                // window 0 = full attention on every layer (granite has no SWA)
                exec.kv_append_batch_paged(
                    &sc.k,
                    &mut kvs.k,
                    &sc.d_pos,
                    Some(&sc.d_slots),
                    &bs.d_bt,
                    bps,
                    kv_dim,
                    r,
                    kv_dtype,
                )?;
                exec.kv_append_batch_paged(
                    &sc.v,
                    &mut kvs.v,
                    &sc.d_pos,
                    Some(&sc.d_slots),
                    &bs.d_bt,
                    bps,
                    kv_dim,
                    r,
                    kv_dtype,
                )?;
            }
            match cuts {
                Some(c) if c.runs.len() == 1 && c.dec == 0 => {
                    // single-prompt chunk: the whole-chunk call
                    attend_span(
                        &exec, sc, kvs, &bs.d_bt, bps, nh, n_kv, hd, kv_dim, 0, r, scale, wmma_pf,
                        false, kv_dtype,
                    )?;
                }
                Some(c) => {
                    let mut runs_fired = false;
                    for &(off, len) in &c.runs {
                        // the fused tick's decode band takes the decode kernel
                        let dec_band = off < c.dec;
                        if runs_armed && !dec_band {
                            // every prefill run of the tick in one launch: base
                            // pointers (row 0), the kernel re-aims per run from
                            // the registered prefix table
                            if !runs_fired {
                                attend_span(
                                    &exec, sc, kvs, &bs.d_bt, bps, nh, n_kv, hd, kv_dim, 0, r,
                                    scale, true, false, kv_dtype,
                                )?;
                                runs_fired = true;
                            }
                            continue;
                        }
                        attend_span(
                            &exec,
                            sc,
                            kvs,
                            &bs.d_bt,
                            bps,
                            nh,
                            n_kv,
                            hd,
                            kv_dim,
                            off,
                            len,
                            scale,
                            wmma_pf && !dec_band,
                            dec_band,
                            kv_dtype,
                        )?;
                    }
                }
                None => {
                    if paddock_models::dev_var_os!("PADDOCK_PROBE_SKIP_ATTN").is_some() {
                        // delete-the-work PROBE (timing only, output is garbage): prices the
                        // decode attention band on the tick's critical path. Never default.
                    } else {
                        // FlashDecoding partial+combine when the unsplit grid would
                        // starve the die. Fused GQA shapes (granite is group 4)
                        // take partial+combine even at ns == 1: the plain
                        // per-q-head kernel re-reads each K/V tile group times.
                        // mirror the pack's vec8 gate (fp8 KV, hd128, G4, no
                        // window, AND batch==1) so the split budget matches the
                        // kernel the launcher will actually pick. The `r == 1`
                        // clause is the batch-gate: the pack routes batched
                        // decode (r>1) to the non-vec8 GQA-fused walk, which is far
                        // faster at r=32 / ~1.3k context (vec8's fixed per-layer
                        // floor dominates once rows share the die), so
                        // r>1 must take the fused split budget below, not the
                        // deep vec8 cap. r==1 keeps vec8 (a tie, so c1 is
                        // unchanged). See f32_qkv.cuh's vec8 arm for the evidence.
                        let vec8 = r == 1
                            && kv_dtype == KvDtype::Fp8E4m3
                            && hd == 128
                            && n_kv * 4 == nh
                            && paddock_models::dev_var_os!("PADDOCK_NO_ATTN_VEC8").is_none();
                        let ns = attn_splits_for(nh, n_kv, r, exec.sm_count(), vec8);
                        // v9q hd128/G4 arm (fp8 KV, r>1): the pack's fp8
                        // end-to-end QGMMA decode kernel takes this shape at
                        // n_splits >= 2 (its ns==1 path writes final rows, which the
                        // partial+combine pipeline cannot consume); bench
                        // g30_dec_attn_bench.cu: 2 splits 7.1us at B=8/ctx128 (walk
                        // 17.6), 4 splits 6.9, and >4 only spend CTAs on 1-block
                        // supertiles. The walk's own s_eff already caps live splits
                        // at ctx<=256, so the clamp is a no-op for it. Position-
                        // independent => graph-bakeable, like every other split count.
                        let v9q_shape = r > 1
                            && kv_dtype == KvDtype::Fp8E4m3
                            && hd == 128
                            && n_kv * 4 == nh
                            && exec.has_attn_partial_batch_paged()
                            && paddock_models::dev_var_os!("PADDOCK_NO_V9Q").is_none();
                        // ns1 rung: v9q at one split writes the final
                        // [b][head][hd] rows itself (bench: 1 split == 2 splits at
                        // B=8 and B=32, ctx<=512), so hand it sc.attn as out_o and
                        // launch no combine -- one kernel fewer per layer per tick
                        // on the critical path. Kill: PADDOCK_V9Q_NS1=0 (back to
                        // 2..4 splits + combine).
                        let v9q_ns1 = v9q_shape
                            && paddock_models::dev_var!("PADDOCK_V9Q_NS1")
                                .ok()
                                .map(|v| v != "0")
                                .unwrap_or(true);
                        let ns = if v9q_ns1 {
                            1
                        } else if v9q_shape {
                            ns.clamp(2, 4)
                        } else {
                            ns
                        };
                        let fused1 = ns == 1 && attn_gqa_fused(nh, n_kv, r);
                        if v9q_ns1 {
                            exec.attn_partial_batch_paged(
                                &sc.q,
                                &kvs.k,
                                &kvs.v,
                                &mut sc.attn,
                                &mut sc.attn_ml,
                                &sc.d_pos,
                                Some(&sc.d_slots),
                                &bs.d_bt,
                                bps,
                                nh,
                                n_kv,
                                hd,
                                kv_dim,
                                0,
                                1,
                                r,
                                scale,
                                kv_dtype,
                            )?;
                        } else if (ns > 1 || fused1) && exec.has_attn_partial_batch_paged() {
                            exec.attn_partial_batch_paged(
                                &sc.q,
                                &kvs.k,
                                &kvs.v,
                                &mut sc.attn_o,
                                &mut sc.attn_ml,
                                &sc.d_pos,
                                Some(&sc.d_slots),
                                &bs.d_bt,
                                bps,
                                nh,
                                n_kv,
                                hd,
                                kv_dim,
                                0,
                                ns,
                                r,
                                scale,
                                kv_dtype,
                            )?;
                            exec.attn_combine_batch(
                                &sc.attn_o,
                                &sc.attn_ml,
                                &sc.sinks,
                                &mut sc.attn,
                                nh,
                                hd,
                                ns,
                                r,
                            )?;
                        } else {
                            exec.attn_decode_batch_paged(
                                &sc.q,
                                &kvs.k,
                                &kvs.v,
                                &sc.sinks,
                                &mut sc.attn,
                                &sc.d_pos,
                                Some(&sc.d_slots),
                                &bs.d_bt,
                                bps,
                                nh,
                                n_kv,
                                hd,
                                kv_dim,
                                0,
                                r,
                                scale,
                                kv_dtype,
                            )?;
                        }
                    }
                }
            }
            // NVFP4 o-proj reduce-fold: when o is an nvf4 plane
            // and the ffn-norm consumer is the scaled residual-norm, the o
            // GEMM leaves RAW split-K partials in `part` (no pd_nvf4_sk_reduce
            // launch, no proj round trip) and the norm folds them. Bit-exact vs
            // reduce+scaled_batch (nv4_fold_cmp.cu diffs=0). `Some((nz, scale2))`
            // routes ffn_norm to the from_parts twin. Kill: PADDOCK_NO_NV4_OPROJ_FOLD.
            let mut o_parts: Option<(u32, f32)> = None;
            if dw8 {
                exec.quantize_q8_sums(&sc.attn, &mut sc.xq, &mut sc.xs, &mut sc.ssums, q_dim)?;
                super::ops::gemv8(
                    &exec,
                    &layer.wo,
                    &sc.attn,
                    &mut sc.xq,
                    &mut sc.xs,
                    &sc.ssums,
                    &mut sc.proj,
                )?;
            } else if r1 && f8l_file {
                exec.quantize_e4m3(&sc.attn, &mut sc.xq, &mut sc.xs4, q_dim)?;
                super::ops::gemv_f8lin(
                    &exec,
                    &layer.wo,
                    &sc.attn,
                    &mut sc.xq,
                    &mut sc.xs4,
                    &mut sc.part,
                    &mut sc.proj,
                )?;
            } else if r1 {
                super::ops::gemv(&exec, &layer.wo, &sc.attn, &mut sc.proj)?;
            } else if pf {
                if super::ops::is_quant(&layer.wo) {
                    prefill_quant_w(
                        &exec, &mut sc.xq, &mut sc.xs, &mut sc.yq, &sc.attn, q_dim, r,
                    )?;
                } else if super::ops::is_f8w(&layer.wo) {
                    exec.quantize_e4m3(&sc.attn, &mut sc.xq, &mut sc.xs4, r * q_dim)?;
                } else if super::ops::is_f8row(&layer.wo) {
                    exec.quantize_e4m3_row(&sc.attn, &mut sc.xq, &mut sc.xs, q_dim, r)?;
                }
                super::ops::pf_mm(
                    &exec,
                    &layer.wo,
                    &sc.attn,
                    &mut sc.xq,
                    &mut sc.xs,
                    &mut sc.xs4,
                    &sc.yq,
                    &mut sc.xsums,
                    &mut sc.ssums,
                    &mut sc.skfix,
                    &mut sc.part,
                    &mut sc.proj,
                    r,
                )?;
            } else if let GraniteW::Nvf4(pwo) = &layer.wo
                && !fuse_norm
                && exec.has_add_rmsnorm_scaled_from_parts()
                && paddock_models::dev_var_os!("PADDOCK_NO_NV4_OPROJ_FOLD").is_none()
            {
                // nvf4 o-proj reduce-fold: stage attn once, then the raw-parts
                // GEMM leaves the split-K slices in `part`; the ffn-norm folds
                // them (o_parts). If the shape declines the split, fall back to
                // the reducing GEMM into proj (reusing the staged activation).
                exec.quantize_nvf4(&sc.attn, &mut sc.xq, &mut sc.xs4, r * q_dim)?;
                match exec.nvf4_gemm_f4_raw_parts(pwo, &sc.xq, &sc.xs4, &mut sc.part, r)? {
                    Some(nz) => o_parts = Some((nz, pwo.scale2)),
                    None => exec.nvf4_gemm_f4(
                        pwo,
                        &sc.xq,
                        &sc.xs4,
                        &mut sc.proj,
                        None,
                        r,
                        Some(&mut sc.part),
                    )?,
                }
            } else {
                if super::ops::is_quant(&layer.wo) {
                    exec.quantize_q8(&sc.attn, &mut sc.xq, &mut sc.xs, r * q_dim)?;
                } else if super::ops::is_f8w(&layer.wo) {
                    exec.quantize_e4m3(&sc.attn, &mut sc.xq, &mut sc.xs4, r * q_dim)?;
                } else if super::ops::is_f8row(&layer.wo) {
                    exec.quantize_e4m3_row(&sc.attn, &mut sc.xq, &mut sc.xs, q_dim, r)?;
                }
                super::ops::mm_pre(
                    &exec,
                    &layer.wo,
                    &sc.attn,
                    &mut sc.xq,
                    &mut sc.xs,
                    &mut sc.xs4,
                    &mut sc.ssums,
                    &mut sc.part,
                    &mut sc.proj,
                    r,
                )?;
            }
            // residual_multiplier on the attention branch: x += 0.22·proj.
            // add_rmsnorm_batch and the e4m3 twin both have no multiplier -
            // add_rmsnorm_q8_xn (slot 482) is the one that does, so this is
            // the full triple collapsing to a single launch.
            let f8row_gu = layer.gate.as_ref().is_some_and(super::ops::is_f8row)
                || layer.up.as_ref().is_some_and(super::ops::is_f8row);
            // NVFP4 mixed-tick down-input fusion: swiglu over the
            // merged gate|up plane straight into the down input's nvf4 staging
            // (swiglu_fused_nvf4), skipping the f32 round trip + separate
            // quantize of the WIDEST activation in the tick (n_ff up to 32768).
            // Bit-exact: same silu expression and pd_nvf4_quant8 as the pair.
            // pf (mixed) path only -- decode's elementwise is tiny. Kill:
            // PADDOCK_NO_NV4_FFN_FUSE. (An ffn-norm->nvf4-quant fusion was tried
            // too, but add_rmsnorm_quant_nvf4_batch is a near-tie vs granite's
            // add_rmsnorm_scaled_batch -- not bit-exact -- so it is not shipped.)
            let nv4_swq_fuse = pf
                && matches!(&layer.gate_up, Some(GraniteW::Nvf4(_)))
                && matches!(&layer.down, GraniteW::Nvf4(_))
                && super::ops::nvf4_prestaged_ok(&exec, r)
                && exec.has_swiglu_fused_nvf4()
                && paddock_models::dev_var_os!("PADDOCK_NO_NV4_FFN_FUSE").is_none();
            let mut nq_ffn = false;
            let mut ffn_staged = false;
            if fuse_norm {
                exec.add_rmsnorm_q8_xn(
                    &mut sc.x,
                    Some(&sc.proj),
                    &layer.ffn_norm.buf,
                    &mut sc.xn,
                    &mut sc.xq,
                    &mut sc.xs,
                    &mut sc.ssums,
                    embd,
                    r,
                    eps,
                    res_s,
                )?;
            } else if exec.has_add_rmsnorm_scaled_batch() {
                // The checkpoint classes want a PLAIN f32 xn, so the q8 triple
                // above does not fit them and they used to pay scale_add and
                // rmsnorm_batch as two launches. Slot 493 is that pair with the
                // multiplier folded in - bit-identical, one launch.
                // fp8 lane: the same pair with the gate/up e4m3
                // row stage folded in as well (bit-identical); the group
                // quantize below then skips. Kill: PADDOCK_NO_F8R_NQ.
                nq_ffn = f8row_gu
                    && paddock_models::dev_var_os!("PADDOCK_NO_F8R_NQ").is_none()
                    && exec.add_rmsnorm_scaled_quant_e4m3_row(
                        &mut sc.x,
                        &sc.proj,
                        &layer.ffn_norm.buf,
                        &mut sc.xn,
                        &mut sc.xq,
                        &mut sc.xs,
                        embd,
                        eps,
                        res_s,
                        r,
                    )?;
                if !nq_ffn {
                    if let Some((nz, scale2)) = o_parts {
                        if fold2 && matches!(&layer.gate_up, Some(GraniteW::Nvf4(_))) {
                            // F2: o-proj partials -> residual -> ffn-norm -> nvf4
                            // pair for gate|up (the add_rmsnorm accumulator family)
                            exec.add_rmsnorm_quant_nvf4_from_parts(
                                &mut sc.x,
                                &sc.part,
                                &layer.ffn_norm.buf,
                                Some(&mut sc.xn),
                                None,
                                &mut sc.xq,
                                &mut sc.xs4,
                                embd,
                                eps,
                                r,
                                res_s,
                                scale2,
                                nz,
                                0,
                            )?;
                            ffn_staged = true;
                        } else {
                            exec.add_rmsnorm_scaled_from_parts(
                                &mut sc.x,
                                &sc.part,
                                &layer.ffn_norm.buf,
                                &mut sc.xn,
                                None,
                                embd,
                                eps,
                                r,
                                res_s,
                                scale2,
                                nz,
                            )?;
                        }
                    } else {
                        exec.add_rmsnorm_scaled_batch(
                            &mut sc.x,
                            &sc.proj,
                            &layer.ffn_norm.buf,
                            &mut sc.xn,
                            embd,
                            eps,
                            r,
                            res_s,
                        )?;
                    }
                }
            } else {
                exec.scale_add(&mut sc.x, &sc.proj, res_s, r * embd)?;
                exec.rmsnorm_batch(&sc.x, &layer.ffn_norm.buf, &mut sc.xn, embd, eps, r)?;
            }

            if dw8 {
                // gate/up share xn; down reads the post-SwiGLU plane, so it
                // needs its own staging at n_ff width. xn's staging is already
                // out when the norm fused.
                if !fuse_norm {
                    exec.quantize_q8_sums(&sc.xn, &mut sc.xq, &mut sc.xs, &mut sc.ssums, embd)?;
                }
                // gate|up as one launch (same merge as the qkv arm above -
                // one boundary fewer per layer, one bigger grid); the GLU
                // arm goes further and absorbs the swiglu too (
                // both dots per row over one staged activation, silu(g)*u
                // written directly - bit-exact vs multi+swiglu, gated by
                // kq_w4a8_glu_matches_split). Kill: PADDOCK_NO_KQ_GLU.
                let (sgate, sup) = super::ops::split_ffn(layer)?;
                match super::ops::both_quant(sgate, sup) {
                    Some((QuantW::Kq(wg), QuantW::Kq(wu)))
                        if exec.has_kquant_gemv_w4a8_glu()
                            && !no_gemv_multi()
                            && paddock_models::dev_var_os!("PADDOCK_NO_KQ_GLU").is_none() =>
                    {
                        exec.kquant_gemv_w4a8_glu(
                            wg,
                            wu,
                            &sc.xq,
                            &sc.xs,
                            &sc.ssums,
                            &mut sc.ffn_gate,
                        )?;
                    }
                    Some((QuantW::Kq(wg), QuantW::Kq(wu)))
                        if exec.has_kquant_gemv_w4a8_multi() && !no_gemv_multi() =>
                    {
                        exec.kquant_gemv_w4a8_multi(
                            &mut [(wg, &mut sc.ffn_gate), (wu, &mut sc.ffn_up)],
                            &sc.xq,
                            &sc.xs,
                            &sc.ssums,
                        )?;
                        exec.swiglu(&mut sc.ffn_gate, &sc.ffn_up, n_ff)?;
                    }
                    _ => {
                        super::ops::gemv8(
                            &exec,
                            sgate,
                            &sc.xn,
                            &mut sc.xq,
                            &mut sc.xs,
                            &sc.ssums,
                            &mut sc.ffn_gate,
                        )?;
                        super::ops::gemv8(
                            &exec,
                            sup,
                            &sc.xn,
                            &mut sc.xq,
                            &mut sc.xs,
                            &sc.ssums,
                            &mut sc.ffn_up,
                        )?;
                        exec.swiglu(&mut sc.ffn_gate, &sc.ffn_up, n_ff)?;
                    }
                }
                exec.quantize_q8_sums(&sc.ffn_gate, &mut sc.xq, &mut sc.xs, &mut sc.ssums, n_ff)?;
                super::ops::gemv8(
                    &exec,
                    &layer.down,
                    &sc.ffn_gate,
                    &mut sc.xq,
                    &mut sc.xs,
                    &sc.ssums,
                    &mut sc.proj,
                )?;
            } else if r1 && f8l_file {
                let (sgate, sup) = super::ops::split_ffn(layer)?;
                exec.quantize_e4m3(&sc.xn, &mut sc.xq, &mut sc.xs4, embd)?;
                super::ops::gemv_f8lin(
                    &exec,
                    sgate,
                    &sc.xn,
                    &mut sc.xq,
                    &mut sc.xs4,
                    &mut sc.part,
                    &mut sc.ffn_gate,
                )?;
                super::ops::gemv_f8lin(
                    &exec,
                    sup,
                    &sc.xn,
                    &mut sc.xq,
                    &mut sc.xs4,
                    &mut sc.part,
                    &mut sc.ffn_up,
                )?;
                exec.swiglu(&mut sc.ffn_gate, &sc.ffn_up, n_ff)?;
                exec.quantize_e4m3(&sc.ffn_gate, &mut sc.xq, &mut sc.xs4, n_ff)?;
                super::ops::gemv_f8lin(
                    &exec,
                    &layer.down,
                    &sc.ffn_gate,
                    &mut sc.xq,
                    &mut sc.xs4,
                    &mut sc.part,
                    &mut sc.proj,
                )?;
            } else if r1 {
                // Q8_0 gate|up as one launch (same merge economics as the
                // qkv arm above - measured +2 us/layer on the bench, the
                // 25600-row grid streams at 96% of the practical ceiling vs
                // the split launches' 93%)
                // merged [2*n_ff, embd]: one GEMV high on the out_dim
                // efficiency curve, then the packing epilogue. No `continue`
                // here deliberately -- the shared `down` GEMV and the residual
                // below must stay on one path.
                if let Some(gu) = &layer.gate_up {
                    super::ops::gemv(&exec, gu, &sc.xn, &mut sc.ffn_gu)?;
                    if matches!(gu, GraniteW::Nvf4(p) if p.gu_pairs) {
                        exec.swiglu_fused_il(&sc.ffn_gu, &mut sc.ffn_gate, n_ff, 1)?;
                    } else {
                        exec.swiglu_fused(&sc.ffn_gu, &mut sc.ffn_gate, n_ff, 1)?;
                    }
                } else {
                    let (sgate, sup) = super::ops::split_ffn(layer)?;
                    match super::ops::both_quant(sgate, sup) {
                        Some((QuantW::Q8(wg), QuantW::Q8(wu)))
                            if exec.has_q8_0_gemv_repacked_multi() && !no_gemv_multi() =>
                        {
                            exec.q8_0_gemv_repacked_multi(
                                &mut [(wg, &mut sc.ffn_gate), (wu, &mut sc.ffn_up)],
                                &sc.xn,
                            )?;
                        }
                        _ => {
                            // Not merged, and this has now been measured TWICE.
                            // First leg: the merge lost, blamed on it buying no
                            // occupancy since gate/up already fill the die.
                            // Second leg, after the geometry sweep found the multi
                            // kernel was holding the wrong CTA width above 8192
                            // rows and re-elected it on the combined count: it
                            // lost again by the same margin. So the cost is the
                            // SEGMENTED WALK itself, not the launch geometry -- do
                            // not try this a third time without changing the
                            // kernel. (q|k|v still merges and
                            // still wins: there the segments are 1024-row planes
                            // that cannot fill the die alone.)
                            //
                            // A genuinely CONTIGUOUS merged plane is the remaining
                            // variant and is not the same thing as this segmented
                            // walk -- but qwen35/load.rs already built one (its
                            // `bs_gu_planes` / repack_q8_concat2 rung) for a
                            // very small win, behind PADDOCK_FUSE_GU because it DUPLICATES
                            // gate+up (~12 GB on the 27B) and is gated on 46 GB of
                            // headroom. Its own note says the duplication is not
                            // worth defaulting until the memory-neutral conversion
                            // lands. Same conclusion applies here: do the
                            // memory-neutral plane first -- do not duplicate
                            // 2.36 GB of granite planes for a percent.
                            super::ops::gemv(&exec, sgate, &sc.xn, &mut sc.ffn_gate)?;
                            super::ops::gemv(&exec, sup, &sc.xn, &mut sc.ffn_up)?;
                        }
                    }
                    exec.swiglu(&mut sc.ffn_gate, &sc.ffn_up, n_ff)?;
                }
                super::ops::gemv(&exec, &layer.down, &sc.ffn_gate, &mut sc.proj)?;
            } else if pf {
                let mut pf_swq_done = false;
                let pf_ffn_q8 = layer.gate_up.as_ref().is_some_and(super::ops::is_quant)
                    || layer.gate.as_ref().is_some_and(super::ops::is_quant)
                    || layer.up.as_ref().is_some_and(super::ops::is_quant);
                if pf_ffn_q8 {
                    prefill_quant_w(&exec, &mut sc.xq, &mut sc.xs, &mut sc.yq, &sc.xn, embd, r)?;
                } else if layer.gate.as_ref().is_some_and(super::ops::is_f8w)
                    || layer.up.as_ref().is_some_and(super::ops::is_f8w)
                {
                    exec.quantize_e4m3(&sc.xn, &mut sc.xq, &mut sc.xs4, r * embd)?;
                } else if (layer.gate.as_ref().is_some_and(super::ops::is_f8row)
                    || layer.up.as_ref().is_some_and(super::ops::is_f8row))
                    && !nq_ffn
                {
                    exec.quantize_e4m3_row(&sc.xn, &mut sc.xq, &mut sc.xs, embd, r)?;
                }
                if let (true, Some(gu), GraniteW::Nvf4(pdn)) =
                    (nv4_swq_fuse, &layer.gate_up, &layer.down)
                {
                    // gate|up as normal (quantizes xn -> ffn_gu), then swiglu
                    // straight into the down input's nvf4 staging (xq/xs4), then
                    // down over that pre-staged pair. Removes the widest f32
                    // activation's round trip + its separate quantize.
                    let pairs = matches!(gu, GraniteW::Nvf4(p) if p.gu_pairs);
                    let swq = match gu {
                        GraniteW::Nvf4(pgu)
                            if pgu.gu_pairs && r >= 128 && exec.has_nvf4_gemm_f4t_swq() =>
                        {
                            Some(pgu)
                        }
                        _ => None,
                    };
                    if let Some(pgu) = swq {
                        // the swiglu + nvf4-quant EPILOGUE: quantize xn
                        // once, one GEMM writes the down input's pair -- the f32
                        // [rows, 2ff] landing (302 MB / 1.07 GB per layer on 30b at
                        // 1152 / 4096 rows) never exists
                        exec.quantize_nvf4(&sc.xn, &mut sc.xq, &mut sc.xs4, r * embd)?;
                        exec.nvf4_gemm_f4t_swq(
                            pgu,
                            &sc.xq,
                            &sc.xs4,
                            &mut sc.xq2,
                            &mut sc.xs4_2,
                            r,
                        )?;
                        super::ops::nvf4_mm_prestaged(
                            &exec,
                            pdn,
                            &sc.xq2,
                            &sc.xs4_2,
                            &mut sc.part,
                            &mut sc.proj,
                            r,
                        )?;
                    } else {
                        super::ops::pf_mm(
                            &exec,
                            gu,
                            &sc.xn,
                            &mut sc.xq,
                            &mut sc.xs,
                            &mut sc.xs4,
                            &sc.yq,
                            &mut sc.xsums,
                            &mut sc.ssums,
                            &mut sc.skfix,
                            &mut sc.part,
                            &mut sc.ffn_gu,
                            r,
                        )?;
                        // PROBE (prices the epilogue swiglu-quant): skipping
                        // this launch leaves stale staging for `down` -- garbage output,
                        // valid timing of the wide-row swiglu read + quant.
                        if paddock_models::dev_var_os!("PADDOCK_PROBE_SKIP_SWIGLU_PF").is_none() {
                            if pairs {
                                exec.swiglu_fused_nvf4_il(
                                    &sc.ffn_gu,
                                    &mut sc.xq,
                                    &mut sc.xs4,
                                    n_ff,
                                    r,
                                )?;
                            } else {
                                exec.swiglu_fused_nvf4(
                                    &sc.ffn_gu,
                                    &mut sc.xq,
                                    &mut sc.xs4,
                                    n_ff,
                                    r,
                                )?;
                            }
                        }
                        super::ops::nvf4_mm_prestaged(
                            &exec,
                            pdn,
                            &sc.xq,
                            &sc.xs4,
                            &mut sc.part,
                            &mut sc.proj,
                            r,
                        )?;
                    }
                    pf_swq_done = true; // down already done; the tail below is skipped
                } else if let Some(gu) = &layer.gate_up {
                    super::ops::pf_mm(
                        &exec,
                        gu,
                        &sc.xn,
                        &mut sc.xq,
                        &mut sc.xs,
                        &mut sc.xs4,
                        &sc.yq,
                        &mut sc.xsums,
                        &mut sc.ssums,
                        &mut sc.skfix,
                        &mut sc.part,
                        &mut sc.ffn_gu,
                        r,
                    )?;
                    if matches!(gu, GraniteW::Nvf4(p) if p.gu_pairs) {
                        exec.swiglu_fused_il(&sc.ffn_gu, &mut sc.ffn_gate, n_ff, r)?;
                    } else {
                        exec.swiglu_fused(&sc.ffn_gu, &mut sc.ffn_gate, n_ff, r)?;
                    }
                } else {
                    let (sgate, sup) = super::ops::split_ffn(layer)?;
                    super::ops::pf_mm(
                        &exec,
                        sgate,
                        &sc.xn,
                        &mut sc.xq,
                        &mut sc.xs,
                        &mut sc.xs4,
                        &sc.yq,
                        &mut sc.xsums,
                        &mut sc.ssums,
                        &mut sc.skfix,
                        &mut sc.part,
                        &mut sc.ffn_gate,
                        r,
                    )?;
                    super::ops::pf_mm(
                        &exec,
                        sup,
                        &sc.xn,
                        &mut sc.xq,
                        &mut sc.xs,
                        &mut sc.xs4,
                        &sc.yq,
                        &mut sc.xsums,
                        &mut sc.ssums,
                        &mut sc.skfix,
                        &mut sc.part,
                        &mut sc.ffn_up,
                        r,
                    )?;
                    // prefill swiglu + down staging in one pass: on a burst
                    // tick the unfused chain wrote the activated
                    // 1024 x 32768 f32 plane (134 MB) and re-read it to quantize.
                    // gate stays unmodified; only the staged q feeds the down
                    // GEMM. Kill: PADDOCK_NO_E4Q_SWIGLU_PF.
                    let pf_swq = super::ops::is_f8row(&layer.down)
                        && paddock_models::dev_var_os!("PADDOCK_NO_E4Q_SWIGLU_PF").is_none()
                        && exec.swiglu_quant_e4m3_row(
                            &sc.ffn_gate,
                            &sc.ffn_up,
                            &mut sc.xq,
                            &mut sc.xs,
                            n_ff,
                            r,
                        )?;
                    if !pf_swq {
                        exec.swiglu(&mut sc.ffn_gate, &sc.ffn_up, r * n_ff)?;
                    }
                    if pf_swq {
                        pf_swq_done = true;
                    }
                }
                if !(nv4_swq_fuse && matches!(&layer.down, GraniteW::Nvf4(_))) {
                    if super::ops::is_quant(&layer.down) {
                        prefill_quant_w(
                            &exec,
                            &mut sc.xq,
                            &mut sc.xs,
                            &mut sc.yq,
                            &sc.ffn_gate,
                            n_ff,
                            r,
                        )?;
                    } else if super::ops::is_f8w(&layer.down) {
                        exec.quantize_e4m3(&sc.ffn_gate, &mut sc.xq, &mut sc.xs4, r * n_ff)?;
                    } else if super::ops::is_f8row(&layer.down) && !pf_swq_done {
                        exec.quantize_e4m3_row(&sc.ffn_gate, &mut sc.xq, &mut sc.xs, n_ff, r)?;
                    }
                    super::ops::pf_mm(
                        &exec,
                        &layer.down,
                        &sc.ffn_gate,
                        &mut sc.xq,
                        &mut sc.xs,
                        &mut sc.xs4,
                        &sc.yq,
                        &mut sc.xsums,
                        &mut sc.ssums,
                        &mut sc.skfix,
                        &mut sc.part,
                        &mut sc.proj,
                        r,
                    )?;
                }
            } else {
                let ffn_q8 = layer.gate_up.as_ref().is_some_and(super::ops::is_quant)
                    || layer.gate.as_ref().is_some_and(super::ops::is_quant)
                    || layer.up.as_ref().is_some_and(super::ops::is_quant);
                if ffn_q8 {
                    exec.quantize_q8(&sc.xn, &mut sc.xq, &mut sc.xs, r * embd)?;
                } else if layer.gate.as_ref().is_some_and(super::ops::is_f8w)
                    || layer.up.as_ref().is_some_and(super::ops::is_f8w)
                {
                    exec.quantize_e4m3(&sc.xn, &mut sc.xq, &mut sc.xs4, r * embd)?;
                } else if (layer.gate.as_ref().is_some_and(super::ops::is_f8row)
                    || layer.up.as_ref().is_some_and(super::ops::is_f8row))
                    && !nq_ffn
                {
                    exec.quantize_e4m3_row(&sc.xn, &mut sc.xq, &mut sc.xs, embd, r)?;
                }
                if let (true, Some(GraniteW::Nvf4(pgu)), GraniteW::Nvf4(pdn)) =
                    (fold2, &layer.gate_up, &layer.down)
                {
                    if !ffn_staged {
                        exec.quantize_nvf4(&sc.xn, &mut sc.xq, &mut sc.xs4, r * embd)?;
                    }
                    // F3: gate|up partials -> swiglu -> nvf4 down input, no y round trip
                    match exec.nvf4_gemm_f4_raw_parts(pgu, &sc.xq, &sc.xs4, &mut sc.part, r)? {
                        Some(nz) => {
                            if pgu.gu_pairs {
                                exec.swiglu_quant_nvf4_from_parts_il(
                                    &sc.part,
                                    None,
                                    &mut sc.xq,
                                    &mut sc.xs4,
                                    n_ff,
                                    r,
                                    pgu.scale2,
                                    nz,
                                )?
                            } else {
                                exec.swiglu_quant_nvf4_from_parts(
                                    &sc.part,
                                    None,
                                    &mut sc.xq,
                                    &mut sc.xs4,
                                    n_ff,
                                    r,
                                    pgu.scale2,
                                    nz,
                                )?
                            }
                        }
                        None => {
                            exec.nvf4_gemm_f4(
                                pgu,
                                &sc.xq,
                                &sc.xs4,
                                &mut sc.ffn_gu,
                                None,
                                r,
                                Some(&mut sc.part),
                            )?;
                            if pgu.gu_pairs {
                                exec.swiglu_fused_nvf4_il(
                                    &sc.ffn_gu,
                                    &mut sc.xq,
                                    &mut sc.xs4,
                                    n_ff,
                                    r,
                                )?;
                            } else {
                                exec.swiglu_fused_nvf4(
                                    &sc.ffn_gu,
                                    &mut sc.xq,
                                    &mut sc.xs4,
                                    n_ff,
                                    r,
                                )?;
                            }
                        }
                    }
                    // down: leave raw partials for the next layer's attn-norm fold
                    // (F1) when that layer can take them; the last layer (and any
                    // non-nvf4 successor) reduces + adds as before
                    let defer = self.layers.get(li + 1).is_some_and(|l| l.qkv_nv4.is_some());
                    let mut deferred = false;
                    if defer
                        && let Some(nz) =
                            exec.nvf4_gemm_f4_raw_parts(pdn, &sc.xq, &sc.xs4, &mut sc.part, r)?
                    {
                        pend_down = Some((nz, pdn.scale2));
                        deferred = true;
                    }
                    if !deferred {
                        exec.nvf4_gemm_f4(
                            pdn,
                            &sc.xq,
                            &sc.xs4,
                            &mut sc.proj,
                            None,
                            r,
                            Some(&mut sc.part),
                        )?;
                        exec.scale_add(&mut sc.x, &sc.proj, res_s, r * embd)?;
                    }
                    resid_done = true;
                } else if let Some(gu) = &layer.gate_up {
                    super::ops::mm_pre(
                        &exec,
                        gu,
                        &sc.xn,
                        &mut sc.xq,
                        &mut sc.xs,
                        &mut sc.xs4,
                        &mut sc.ssums,
                        &mut sc.part,
                        &mut sc.ffn_gu,
                        r,
                    )?;
                    if matches!(gu, GraniteW::Nvf4(p) if p.gu_pairs) {
                        exec.swiglu_fused_il(&sc.ffn_gu, &mut sc.ffn_gate, n_ff, r)?;
                    } else {
                        exec.swiglu_fused(&sc.ffn_gu, &mut sc.ffn_gate, n_ff, r)?;
                    }
                } else {
                    let (sgate, sup) = super::ops::split_ffn(layer)?;
                    // two-segment gate|up: one grid over both f8row planes at
                    // decode widths -- 5 GEMM launches per layer becomes 4
                    // without concatenating gate|up into one plane, so it stays
                    // memory-neutral. The pack takes batch <= 16 only (above
                    // that the BN=32 tile loses to the K-split path) and
                    // declines the rest;
                    // then the two single GEMMs run as before.
                    // Kill: PADDOCK_NO_F8R_GEMM2.
                    let fused2 = paddock_models::dev_var_os!("PADDOCK_NO_F8R_GEMM2").is_none()
                        && match (sgate, sup) {
                            (GraniteW::Fp8 { plane: pg, .. }, GraniteW::Fp8 { plane: pu, .. })
                                if pg.scale.len() == pu.scale.len() =>
                            {
                                exec.f8row_gemm2(
                                    pg,
                                    pu,
                                    &sc.xq,
                                    &sc.xs,
                                    &mut sc.ffn_gate,
                                    &mut sc.ffn_up,
                                    embd,
                                    n_ff,
                                    r,
                                )?
                            }
                            _ => false,
                        };
                    if !fused2 {
                        super::ops::mm_pre(
                            &exec,
                            sgate,
                            &sc.xn,
                            &mut sc.xq,
                            &mut sc.xs,
                            &mut sc.xs4,
                            &mut sc.ssums,
                            &mut sc.part,
                            &mut sc.ffn_gate,
                            r,
                        )?;
                        super::ops::mm_pre(
                            &exec,
                            sup,
                            &sc.xn,
                            &mut sc.xq,
                            &mut sc.xs,
                            &mut sc.xs4,
                            &mut sc.ssums,
                            &mut sc.part,
                            &mut sc.ffn_up,
                            r,
                        )?;
                    }
                    exec.swiglu(&mut sc.ffn_gate, &sc.ffn_up, r * n_ff)?;
                }
                if !resid_done {
                    if super::ops::is_quant(&layer.down) {
                        exec.quantize_q8(&sc.ffn_gate, &mut sc.xq, &mut sc.xs, r * n_ff)?;
                    } else if super::ops::is_f8w(&layer.down) {
                        exec.quantize_e4m3(&sc.ffn_gate, &mut sc.xq, &mut sc.xs4, r * n_ff)?;
                    } else if super::ops::is_f8row(&layer.down) {
                        exec.quantize_e4m3_row(&sc.ffn_gate, &mut sc.xq, &mut sc.xs, n_ff, r)?;
                    }
                    super::ops::mm_pre(
                        &exec,
                        &layer.down,
                        &sc.ffn_gate,
                        &mut sc.xq,
                        &mut sc.xs,
                        &mut sc.xs4,
                        &mut sc.ssums,
                        &mut sc.part,
                        &mut sc.proj,
                        r,
                    )?;
                }
            }
            // residual_multiplier again on the FFN branch (the fold2 arm
            // either added it or deferred it into the next layer's attn-norm)
            if !resid_done {
                exec.scale_add(&mut sc.x, &sc.proj, res_s, r * embd)?;
            }
        }
        debug_assert!(
            pend_down.is_none(),
            "nvf4 fold-2: down partials left unfolded after the last layer"
        );
        if runs_armed {
            exec.pf_runs_register(None)?;
        }
        Ok(())
    }

    /// Final norm + lm_head over rows 0..rows, leaving [rows, vocab] in
    /// head_logits with granite's `logit_scale` applied.
    fn head_rows(&mut self, rows: usize) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let hp = &self.hp;
        let (embd, vocab, eps) = (hp.n_embd, hp.n_vocab, hp.eps);
        let inv_logit = 1.0 / hp.logit_scale;
        let bs = self.batch.as_mut().expect("batch enabled");
        let sc = &mut bs.sc;
        exec.rmsnorm_batch(&sc.x, &self.output_norm.buf, &mut sc.xn, embd, eps, rows)?;
        // the vocab head is the single widest decode GEMV (embd x vocab - on
        // the 30b that is 0.35 GB of the 18.4 GB read per token), so it takes
        // the serving class too
        if rows == 1
            && matches!(self.lm_head.quant(), Some(QuantW::Kq(_)))
            && exec.has_kquant_gemv_w4a8()
            && !no_w4a8_decode()
        {
            exec.quantize_q8_sums(&sc.xn, &mut sc.xq, &mut sc.xs, &mut sc.ssums, embd)?;
            super::ops::gemv8(
                &exec,
                &self.lm_head,
                &sc.xn,
                &mut sc.xq,
                &mut sc.xs,
                &sc.ssums,
                &mut sc.head_logits,
            )?;
        } else if rows == 1 {
            super::ops::gemv(&exec, &self.lm_head, &sc.xn, &mut sc.head_logits)?;
        } else {
            if super::ops::is_quant(&self.lm_head) {
                exec.quantize_q8(&sc.xn, &mut sc.xq, &mut sc.xs, rows * embd)?;
            }
            // a vocab-wide out exceeds the mma partial plane, so this lands on
            // the dp4a rungs inside mmq_pre_any
            super::ops::mm_pre(
                &exec,
                &self.lm_head,
                &sc.xn,
                &mut sc.xq,
                &mut sc.xs,
                &mut sc.xs4,
                &mut sc.ssums,
                &mut sc.part,
                &mut sc.head_logits,
                rows,
            )?;
        }
        // logits_scaling: llama.cpp divides by f_logit_scale. This must stay
        // on the device logits rather than folding into sampling - argmax is
        // invariant to it, but logprobs and temperature are not.
        exec.scale(&mut sc.head_logits, inv_logit, rows * vocab)?;
        Ok(())
    }

    /// Stage residual row `row` at row 0 so a single-row head pass reads it.
    /// Bounced through `proj` because src and dst live in the same buffer.
    /// Gather the residual rows of `fins` (order index, row, plan) into
    /// x[0..n] through proj (every source row is read before any x row is
    /// written), then one norm+head at M=n and one device sample. Returns
    /// the sampled ids in `fins` order.
    fn head_sample_rows_gathered(
        &mut self,
        fins: &[(usize, usize, crate::generator::RowSample)],
    ) -> Result<Vec<u32>, GpuModelError> {
        let n_embd = self.hp.n_embd;
        let n = fins.len();
        {
            let exec = self.exec.clone();
            let bs = self.batch.as_mut().expect("batch enabled");
            let sc = &mut bs.sc;
            for (i, &(_, row, _)) in fins.iter().enumerate() {
                let src =
                    sc.x.try_slice(row * n_embd..(row + 1) * n_embd)
                        .ok_or_else(|| GpuError::Driver("x row slice".into()))?;
                let mut dst = sc
                    .proj
                    .try_slice_mut(i * n_embd..(i + 1) * n_embd)
                    .ok_or_else(|| GpuError::Driver("proj row slice".into()))?;
                exec.stream.memcpy_dtod(&src, &mut dst).map_err(drv)?;
            }
            let ps = sc
                .proj
                .try_slice(0..n * n_embd)
                .ok_or_else(|| GpuError::Driver("proj src slice".into()))?;
            let mut xd =
                sc.x.try_slice_mut(0..n * n_embd)
                    .ok_or_else(|| GpuError::Driver("x dst slice".into()))?;
            exec.stream.memcpy_dtod(&ps, &mut xd).map_err(drv)?;
        }
        self.head_rows(n)?;
        let plans: Vec<crate::generator::RowSample> = fins.iter().map(|&(_, _, p)| p).collect();
        let s = self.sample_head_rows(n, &plans)?;
        Ok(s.ids)
    }
    fn head_row_at(&mut self, row: usize) -> Result<(), GpuModelError> {
        let n_embd = self.hp.n_embd;
        if row > 0 {
            let exec = self.exec.clone();
            let bs = self.batch.as_mut().expect("batch enabled");
            let sc = &mut bs.sc;
            let src =
                sc.x.try_slice(row * n_embd..(row + 1) * n_embd)
                    .ok_or_else(|| GpuError::Driver("x row slice".into()))?;
            let mut dst = sc
                .proj
                .try_slice_mut(0..n_embd)
                .ok_or_else(|| GpuError::Driver("proj row slice".into()))?;
            exec.stream.memcpy_dtod(&src, &mut dst).map_err(drv)?;
            let ps = sc
                .proj
                .try_slice(0..n_embd)
                .ok_or_else(|| GpuError::Driver("proj src slice".into()))?;
            let mut xd =
                sc.x.try_slice_mut(0..n_embd)
                    .ok_or_else(|| GpuError::Driver("x dst slice".into()))?;
            exec.stream.memcpy_dtod(&ps, &mut xd).map_err(drv)?;
        }
        self.head_rows(1)
    }

    /// Prefill tail: head over residual row `row`, returning that one vocab
    /// row on the host. Norm+head over the whole tail would waste vocab GEMM
    /// rows, so the single row is staged at row 0 of a fresh pass.
    pub(super) fn head_row(&mut self, row: usize) -> Result<Vec<f32>, GpuModelError> {
        let n_vocab = self.hp.n_vocab;
        self.head_row_at(row)?;
        let bs = self.batch.as_ref().expect("batch enabled");
        let v = bs
            .sc
            .head_logits
            .try_slice(0..n_vocab)
            .ok_or_else(|| GpuError::Driver("head row slice".into()))?;
        Ok(self.exec.stream.clone_dtoh(&v).map_err(drv)?)
    }

    /// Read the [rows, vocab] logits back to the host.
    pub(crate) fn read_batch_logits(&mut self, rows: usize) -> Result<Vec<f32>, GpuModelError> {
        let vocab = self.hp.n_vocab;
        let bs = self.batch.as_ref().expect("batch enabled");
        let v = bs
            .sc
            .head_logits
            .try_slice(0..rows * vocab)
            .ok_or_else(|| GpuError::Driver("batch logits slice".into()))?;
        Ok(self.exec.stream.clone_dtoh(&v).map_err(drv)?)
    }

    // ── device sampling ─────────────────────────────────────────────────────

    /// Pack per-row sampler params (inv_t, u, mode, pad). Host/Hole rows stay
    /// mode 0 = untouched. RsVerify is a spec-only plan; granite has no
    /// drafter, so it can never arrive here.
    /// TruncCat rows pack mode 5 (top_k 1..=64) or mode 6 (k-less)
    /// plus the tpar side plane (Some iff any) - dialled truncation
    /// requests sample fully on device like every other family.
    fn pack_samp_par(plans: &[crate::generator::RowSample]) -> (Vec<u32>, Option<Vec<u32>>) {
        use crate::generator::RowSample;
        use crate::sampler::DevicePlan;
        let mut par = vec![0u32; plans.len() * 4];
        let mut tpar = vec![0u32; plans.len() * 4];
        let mut any_trunc = false;
        for (i, p) in plans.iter().enumerate() {
            match p {
                RowSample::Hole | RowSample::Host => {}
                RowSample::Device(DevicePlan::Greedy) => par[i * 4 + 2] = 1,
                RowSample::Device(DevicePlan::Categorical { inv_t, u }) => {
                    par[i * 4] = inv_t.to_bits();
                    par[i * 4 + 1] = u.to_bits();
                    par[i * 4 + 2] = 2;
                }
                RowSample::Device(DevicePlan::TruncCat {
                    inv_t,
                    u,
                    k,
                    top_p,
                    min_p,
                }) => {
                    par[i * 4] = inv_t.to_bits();
                    par[i * 4 + 1] = u.to_bits();
                    par[i * 4 + 2] = if *k >= 1 && *k <= 64 { 5 } else { 6 };
                    tpar[i * 4] = *k;
                    tpar[i * 4 + 1] = top_p.to_bits();
                    tpar[i * 4 + 2] = min_p.to_bits();
                    any_trunc = true;
                }
                RowSample::Device(DevicePlan::RsVerify { .. })
                | RowSample::Device(DevicePlan::RsTrunc { .. }) => {}
            }
        }
        (par, any_trunc.then_some(tpar))
    }

    /// device-truncation engagement witness (bisect-trap law): once per process.
    fn trunc_dev_witness(rows: usize) {
        static DEV: std::sync::Once = std::sync::Once::new();
        DEV.call_once(|| {
            eprintln!("[trunc-dev6] engaged: r={rows} (granite device truncation sampling)");
        });
    }

    /// TruncCat rows execute fully on device (slots 435+436).
    pub(crate) fn device_trunc_supported(&self) -> bool {
        self.batch.is_some() && self.exec.has_sample_rows_t() && self.exec.has_sample_rows_p()
    }

    pub(crate) fn supports_device_sampling_impl(&self) -> bool {
        self.batch.is_some() && self.exec.has_sample_rows()
    }

    /// Sample head_logits rows 0..r on device with `plans`; only Host-plan
    /// rows pay a vocab-row readback. Assumes the head has already run.
    fn sample_head_rows(
        &mut self,
        r: usize,
        plans: &[crate::generator::RowSample],
    ) -> Result<crate::generator::SampledStep, GpuModelError> {
        use crate::generator::{RowSample, SampledStep};
        assert_eq!(plans.len(), r, "one plan per row");
        let exec = self.exec.clone();
        let vocab = self.hp.n_vocab;
        let (par, tpar) = Self::pack_samp_par(plans);
        {
            let sc = &mut self.batch.as_mut().expect("batch enabled").sc;
            let mut v = sc
                .d_par
                .try_slice_mut(0..r * 4)
                .ok_or_else(|| GpuError::Driver("d_par slice".into()))?;
            exec.stream.memcpy_htod(&par, &mut v).map_err(drv)?;
            if let Some(t) = &tpar {
                let mut v = sc
                    .d_tpar
                    .try_slice_mut(0..r * 4)
                    .ok_or_else(|| GpuError::Driver("d_tpar slice".into()))?;
                exec.stream.memcpy_htod(t, &mut v).map_err(drv)?;
            }
            exec.sample_rows_at(&sc.head_logits, &sc.d_par, 0, &mut sc.d_out, 0, r, vocab)?;
            if tpar.is_some() {
                Self::trunc_dev_witness(r);
                exec.sample_rows_t_at(
                    &sc.head_logits,
                    &sc.d_par,
                    0,
                    &sc.d_tpar,
                    0,
                    &mut sc.d_out,
                    0,
                    r,
                    vocab,
                )?;
                exec.sample_rows_p_at(
                    &sc.head_logits,
                    &sc.d_par,
                    0,
                    &sc.d_tpar,
                    0,
                    &mut sc.d_out,
                    0,
                    r,
                    vocab,
                )?;
            }
        }
        let sc = &self.batch.as_ref().expect("batch enabled").sc;
        let ids_view = sc
            .d_out
            .try_slice(0..r)
            .ok_or_else(|| GpuError::Driver("d_out slice".into()))?;
        let ids = exec.stream.clone_dtoh(&ids_view).map_err(drv)?;
        let mut host_rows = Vec::new();
        for (i, p) in plans.iter().enumerate() {
            if matches!(p, RowSample::Host) {
                let v = sc
                    .head_logits
                    .try_slice(i * vocab..(i + 1) * vocab)
                    .ok_or_else(|| GpuError::Driver("host row slice".into()))?;
                host_rows.push((i, exec.stream.clone_dtoh(&v).map_err(drv)?));
            }
        }
        Ok(SampledStep { ids, host_rows })
    }

    /// Device-sampled decode tick: graph replay + sample_rows, ids come back
    /// as r u32s.
    pub(crate) fn forward_batch_sampled_impl(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        plans: &[crate::generator::RowSample],
    ) -> Result<crate::generator::SampledStep, GpuModelError> {
        self.forward_batch_sampled_slots(tokens, positions, None, plans)
    }

    /// `forward_batch_sampled_impl` with optional explicit slot ids. None
    /// keeps the dense row-i = slot-i mapping.
    pub(crate) fn forward_batch_sampled_slots(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        slots: Option<&[u32]>,
        plans: &[crate::generator::RowSample],
    ) -> Result<crate::generator::SampledStep, GpuModelError> {
        let r = tokens.len();
        let owned: Vec<u32> = (0..r as u32).collect();
        let ident: &[u32] = slots.unwrap_or(&owned);
        assert_eq!(ident.len(), r, "one slot per row");
        self.ensure_rows(ident, positions)?;
        self.upload_rows(tokens, positions, ident)?;
        self.step_replay(r)?;
        let step = self.sample_head_rows(r, plans)?;
        // seal mirror: these decode rows' KV just landed
        for i in 0..r {
            self.hist_append(ident[i] as usize, Some(positions[i]), &tokens[i..i + 1]);
        }
        Ok(step)
    }

    // ── depth-2 decode pipe (the idle-edge door) ─────────────────
    // qwen35's pipe-under-pool is the template (batch.rs there): tick N+1's
    // inputs advance on device from tick N's sampler output, so the host's
    // per-token turnaround (id readback, SSE commit, next submit) overlaps
    // the GPU instead of gapping it. That gap measured ~0.6
    // ms/token on granite-30b at c1 - every steady-state gap terminated at the
    // next replay's first node. Pool growth is handled by the same
    // ensure_rows the classic tick uses, called per tick with the advanced
    // positions before the replay (stream-ordered table upload; sheds radix
    // LRU before reporting PoolExhausted - the scheduler's headroom gate
    // keeps that from firing mid-flight).

    pub(crate) fn supports_decode_pipe_impl(&self) -> bool {
        self.exec.has_sample_rows()
            && self.exec.has_pipe_advance()
            && paddock_models::dev_var_os!("PADDOCK_NO_DECODE_PIPE").is_none()
    }

    fn pipe_launch_tick(
        &mut self,
        plans: &[crate::generator::RowSample],
        advance: bool,
    ) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let (b, tick) = {
            let p = self.pipe.as_ref().expect("pipe active");
            (p.b, p.tick)
        };
        // back every row's THIS-tick write position before anything mutates -
        // a growth error leaves the rings/inputs untouched
        {
            let (pos0, slot_map) = {
                let p = self.pipe.as_ref().expect("pipe active");
                (p.pos0.clone(), p.slots.clone())
            };
            let slots_v: Vec<u32> = (0..b as u32)
                .map(|i| slot_map.as_ref().map_or(i, |s| s[i as usize]))
                .collect();
            let pos_v: Vec<u32> = pos0.iter().map(|&p0| p0 + tick as u32).collect();
            self.ensure_rows(&slots_v, &pos_v)?;
        }
        let ring = (tick % 2) as usize;
        let (par, tpar) = Self::pack_samp_par(plans);
        let max_batch = self.batch.as_ref().expect("batch enabled").n_slots;
        {
            let sc = &mut self.batch.as_mut().expect("batch enabled").sc;
            let off = ring * max_batch * 4;
            let mut v = sc
                .d_pipe_par
                .try_slice_mut(off..off + b * 4)
                .ok_or_else(|| GpuError::Driver("d_pipe_par slice".into()))?;
            exec.stream.memcpy_htod(&par, &mut v).map_err(drv)?;
            if let Some(t) = &tpar {
                let mut v = sc
                    .d_pipe_tpar
                    .try_slice_mut(off..off + b * 4)
                    .ok_or_else(|| GpuError::Driver("d_pipe_tpar slice".into()))?;
                exec.stream.memcpy_htod(t, &mut v).map_err(drv)?;
            }
        }
        if advance {
            // tokens <- previous ring's sampled ids, positions += 1, on device
            let prev = ((tick + 1) % 2) as usize;
            let sc = &mut self.batch.as_mut().expect("batch enabled").sc;
            let (out, tok, pos) = (&sc.d_pipe_out, &mut sc.d_toks, &mut sc.d_pos);
            exec.pipe_advance(out, prev * max_batch, tok, pos, b)?;
        }
        self.step_replay(b)?;
        // The sampler chain (its ~11 launch-bound topp_mb kernels dominate the
        // eager tick) rides its own captured graph now - one replay instead of
        // the launch train. Fresh `u` was uploaded into d_pipe_par[ring] above,
        // so the replay draws anew each tick. trunc rows draw into the
        // same out ring inside `sampler_body`.
        self.sampler_replay(b, ring, tpar.is_some())?;
        let ev = exec.record_event()?;
        self.pipe.as_mut().expect("pipe active").ev[ring] = Some(ev);
        Ok(())
    }

    pub(crate) fn decode_pipe_begin_impl(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        slots: Option<&[u32]>,
        plans: &[crate::generator::RowSample],
    ) -> Result<(), GpuModelError> {
        let b = tokens.len();
        assert_eq!(plans.len(), b, "one plan per row");
        assert_eq!(positions.len(), b, "one position per row");
        if !self.supports_decode_pipe_impl() {
            return Err(GpuModelError::Config("decode pipe unsupported".into()));
        }
        match &self.batch {
            None => return Err(GpuModelError::BatchDisabled),
            Some(bs) if b > bs.n_slots => {
                return Err(GpuModelError::BatchTooLarge {
                    got: b,
                    max: bs.n_slots,
                });
            }
            _ => {}
        }
        if let Some(s) = slots {
            assert_eq!(s.len(), b, "one slot per row");
        }
        assert!(self.pipe.is_none(), "decode pipe already active");
        // tick-0 inputs land in the fixed graph buffers (advance=false leaves
        // them); ensure_rows runs inside pipe_launch_tick with tick=0
        let ident: Vec<u32> = (0..b as u32).collect();
        self.upload_rows(tokens, positions, slots.unwrap_or(&ident))?;
        self.pipe = Some(crate::gpu_model::granite::PipeStateG {
            b,
            tick: 0,
            ev: [None, None],
            pos0: positions.to_vec(),
            slots: slots.map(<[u32]>::to_vec),
        });
        if let Err(e) = self.pipe_launch_tick(plans, false) {
            self.pipe_abort();
            return Err(e);
        }
        // seal mirror: the pipe's tick-0 input rows are now backed
        let feed: Vec<(usize, u32, u32)> = {
            let st = self.pipe.as_ref().expect("just set");
            (0..b)
                .map(|i| {
                    let s0 = st.slots.as_ref().map(|v| v[i] as usize).unwrap_or(i);
                    (s0, tokens[i], positions[i])
                })
                .collect()
        };
        for (s0, t, p) in feed {
            self.hist_append(s0, Some(p), &[t]);
        }
        Ok(())
    }

    /// Enqueue the next tick and return the OLDEST in-flight tick's ids, read
    /// via the copy stream while the new tick executes.
    pub(crate) fn decode_pipe_next_impl(
        &mut self,
        plans: &[crate::generator::RowSample],
    ) -> Result<Vec<u32>, GpuModelError> {
        let exec = self.exec.clone();
        let (b, j) = {
            let p = self
                .pipe
                .as_ref()
                .ok_or_else(|| GpuModelError::Config("decode_pipe_next without begin".into()))?;
            (p.b, p.tick)
        };
        assert_eq!(plans.len(), b, "one plan per row");
        self.pipe.as_mut().expect("live pipe").tick = j + 1;
        if let Err(e) = self.pipe_launch_tick(plans, true) {
            self.pipe_abort();
            return Err(e);
        }
        let ring = (j % 2) as usize;
        let max_batch = self.batch.as_ref().expect("batch enabled").n_slots;
        let r = {
            let sc = &self.batch.as_ref().expect("batch enabled").sc;
            let ev = self.pipe.as_ref().expect("live pipe").ev[ring]
                .as_ref()
                .expect("in-flight event");
            exec.to_host_u32_after(ev, &sc.d_pipe_out, ring * max_batch, b)
        };
        match r {
            Ok(ids) => {
                // seal mirror: these ids' rows were fed by the tick that is
                // already executing (device advance) - backed, in row order
                let feed: Vec<(usize, u32)> = {
                    let st = self.pipe.as_ref().expect("live pipe");
                    ids.iter()
                        .enumerate()
                        .map(|(i, &t)| (st.slots.as_ref().map(|v| v[i] as usize).unwrap_or(i), t))
                        .collect()
                };
                for (s0, t) in feed {
                    self.hist_append(s0, None, &[t]);
                }
                Ok(ids)
            }
            Err(e) => {
                self.pipe_abort();
                Err(e.into())
            }
        }
    }

    /// End the pipe: return the last in-flight tick's ids. The fixed input
    /// buffers are stale after this - every other path re-uploads them.
    pub(crate) fn decode_pipe_drain_impl(&mut self) -> Result<Vec<u32>, GpuModelError> {
        let exec = self.exec.clone();
        let st = self
            .pipe
            .take()
            .ok_or_else(|| GpuModelError::Config("decode_pipe_drain without begin".into()))?;
        let ring = (st.tick % 2) as usize;
        let max_batch = self
            .batch
            .as_ref()
            .ok_or(GpuModelError::BatchDisabled)?
            .n_slots;
        let ev = st.ev[ring].as_ref().expect("in-flight event");
        let sc = &self.batch.as_ref().expect("batch checked above").sc;
        match exec.to_host_u32_after(ev, &sc.d_pipe_out, ring * max_batch, st.b) {
            Ok(ids) => Ok(ids),
            Err(e) => {
                let _ = exec.synchronize(); // state gone - quiesce ring readers
                Err(e.into())
            }
        }
    }

    /// Kill an in-flight pipe (error/reset): quiesce so nothing still reads
    /// the rings, then drop the state.
    pub(crate) fn pipe_abort(&mut self) {
        if self.pipe.take().is_some() {
            let _ = self.exec.synchronize();
        }
    }
}

/// Prefill-mode projection off `prefill_quant`'s planes - a thin alias so
/// every prefill seat provably takes the same rungs (Q8_0 keeps its exact
/// ladder; k-quant rides the W4A8 int8 tensor-core GEMM).
#[allow(clippy::too_many_arguments)]
/// Attend rows [off, off+len) of the current pass. `dec_band` routes a fused
/// tick's q_len==1 rows to the decode kernel; otherwise a span takes the
/// tensor-core prefill kernel when available and the tiled one below it.
fn attend_span(
    exec: &crate::gpu::GpuExecutor,
    sc: &mut BatchScratch,
    kvs: &LayerKv,
    bt: &CudaSlice<u32>,
    bps: usize,
    nh: usize,
    n_kv: usize,
    hd: usize,
    kv_dim: usize,
    off: usize,
    len: usize,
    scale: f32,
    wmma: bool,
    dec_band: bool,
    kv_dtype: KvDtype,
) -> Result<(), GpuModelError> {
    if dec_band {
        // The fused mixed tick's decode band is one run at off == 0 (PfCuts::
        // fused), so the whole-buffer partial+combine pair applies to rows
        // 0..len directly -- the same pipeline the pure-decode tick runs, and
        // the only one that reaches the pack's v9q arm. The rows kernel it
        // replaces ran 78.6 us/layer at c32 (in-graph capture)
        // against the partial+combine pair's 26.8 + 11.5. Kill: PADDOCK_NO_V9Q
        // (falls back to the rows kernel, the pre-rung behavior).
        let band_v9q = off == 0
            && len > 1
            && kv_dtype == KvDtype::Fp8E4m3
            && hd == 128
            && n_kv * 4 == nh
            && exec.has_attn_partial_batch_paged()
            && paddock_models::dev_var_os!("PADDOCK_NO_V9Q").is_none();
        if band_v9q {
            let ns1 = paddock_models::dev_var!("PADDOCK_V9Q_NS1")
                .ok()
                .map(|v| v != "0")
                .unwrap_or(true);
            if ns1 {
                // ns1 rung: one split, the kernel writes the final rows into
                // sc.attn (rows 0..len), no combine
                exec.attn_partial_batch_paged(
                    &sc.q,
                    &kvs.k,
                    &kvs.v,
                    &mut sc.attn,
                    &mut sc.attn_ml,
                    &sc.d_pos,
                    Some(&sc.d_slots),
                    bt,
                    bps,
                    nh,
                    n_kv,
                    hd,
                    kv_dim,
                    0,
                    1,
                    len,
                    scale,
                    kv_dtype,
                )?;
            } else {
                let ns = attn_splits_for(nh, n_kv, len, exec.sm_count(), false).clamp(2, 4);
                exec.attn_partial_batch_paged(
                    &sc.q,
                    &kvs.k,
                    &kvs.v,
                    &mut sc.attn_o,
                    &mut sc.attn_ml,
                    &sc.d_pos,
                    Some(&sc.d_slots),
                    bt,
                    bps,
                    nh,
                    n_kv,
                    hd,
                    kv_dim,
                    0,
                    ns,
                    len,
                    scale,
                    kv_dtype,
                )?;
                exec.attn_combine_batch(
                    &sc.attn_o,
                    &sc.attn_ml,
                    &sc.sinks,
                    &mut sc.attn,
                    nh,
                    hd,
                    ns,
                    len,
                )?;
            }
        } else {
            exec.attn_decode_batch_rows_paged(
                &sc.q,
                &kvs.k,
                &kvs.v,
                &sc.sinks,
                &mut sc.attn,
                &sc.d_pos,
                Some(&sc.d_slots),
                bt,
                bps,
                nh,
                n_kv,
                hd,
                kv_dim,
                0,
                off,
                len,
                scale,
                kv_dtype,
            )?;
        }
    } else if wmma {
        exec.attn_prefill_f16_paged_at(
            &sc.q,
            &kvs.k,
            &kvs.v,
            &sc.sinks,
            &mut sc.attn,
            &sc.d_pos,
            &sc.d_slots,
            off,
            bt,
            bps,
            nh,
            n_kv,
            hd,
            kv_dim,
            0,
            len,
            scale,
            kv_dtype,
        )?;
    } else if len > 24 && exec.has_attn_prefill_paged() && matches!(hd, 128 | 256 | 512) {
        // head_dim gate, not decoration: the pack instantiates this tiled
        // kernel for 128/256/512 only and returns cudaErrorInvalidValue for
        // anything else - which surfaces as a bare "kernel launcher returned
        // CUDA error 1" with nothing naming the head dim. granite-vision's 64
        // therefore falls through to the head_dim-generic decode-class kernel
        // below, which is correct but leaves prefill throughput on the table
        // whenever the WMMA path is off.
        exec.attn_prefill_rows_paged(
            &sc.q,
            &kvs.k,
            &kvs.v,
            &sc.sinks,
            &mut sc.attn,
            &sc.d_pos,
            &sc.d_slots,
            bt,
            bps,
            nh,
            n_kv,
            hd,
            kv_dim,
            0,
            off,
            len,
            scale,
            kv_dtype,
        )?;
    } else {
        exec.attn_decode_batch_rows_paged(
            &sc.q,
            &kvs.k,
            &kvs.v,
            &sc.sinks,
            &mut sc.attn,
            &sc.d_pos,
            Some(&sc.d_slots),
            bt,
            bps,
            nh,
            n_kv,
            hd,
            kv_dim,
            0,
            off,
            len,
            scale,
            kv_dtype,
        )?;
    }
    Ok(())
}
