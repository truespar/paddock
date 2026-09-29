//! gpt-oss on the A6000: the CPU reference graph, on device, op by op.
//!
//! Residency plan: weights stay in their on-disk quantization and dequant on
//! the fly inside fused GEMV kernels - MXFP4 experts (A2) and Q8_0 attention /
//! router / lm_head (A3). Only norms, biases, sinks, and the embedding table
//! live as f32. Keeping the big GEMV weights quantized-resident is both the
//! memory plan (~5 GB less than dequant-to-f32) and the bandwidth plan (decode
//! is memory-bound; fewer weight bytes per token = faster).
//!
//! Perf passes landed: batched flash-style attention in one launch (A1),
//! on-device MoE routing + fused MXFP4 dequant-GEMV (A2), fused Q8_0
//! dequant-GEMV for the dense projections (A3). No host sync inside a token.

use std::sync::Arc;

use cudarc::driver::sys::CUstreamCaptureMode;
use cudarc::driver::{CudaSlice, DevicePtr};
use paddock_kernels::reference::ops::YarnRope;
use paddock_models::gguf::Value;
use paddock_models::mapped::MappedGguf;

use crate::gpu::{
    DeviceTensor, GpuError, GpuExecutor, KvDtype, QuantTensor, RepackedMxfp4, RepackedQ8,
};
use crate::kv_plan;
use crate::kv_pool::{BlockTable, KvPool};
use crate::paged_radix::PagedRadix;
use crate::spec::NgramDraft;

const SWIGLU_ALPHA: f32 = 1.702;
const SWIGLU_LIMIT: f32 = 7.0;

/// Tokens produced per graph-replay burst in the B=1 greedy loop (one sync +
/// readback of `d_g_out` per chunk instead of per token).
const GEN_CHUNK: usize = 64;

/// A captured decode-step graph. The raw handle is only ever used from the
/// engine thread that owns the CUDA context/stream (same contract as qwen35's).
struct SendGraph(crate::gpu::CapturedGraph);
// SAFETY: see above - single-owner-thread usage.
unsafe impl Send for SendGraph {}

/// Prompt tokens processed per prefill pass. Larger = fewer passes (more parallel
/// prefill) but bigger row-scratch; the batched scratch is sized to hold this many
/// rows so a prompt is processed in ceil(P / PREFILL_CHUNK) passes, not P steps.
/// 512 (was 256): llama runs pp512 as one ubatch - chunking at 256 paid every
/// per-pass cost twice AND halved tokens-per-expert, doubling MoE weight reads
/// per token (the launch-shape lesson, again). Scratch cost ~50 MB extra.
// 8192 (was 2048): bigger mixed-tick chunks pack more prompt rows behind one
// MoE weight-read per tick, freeing longer uninterrupted decode-pipe
// stretches. Costs ~2x TTFT under saturation (long mixed ticks delay first
// tokens) + ~1 GB scratch - accepted for max throughput;
// PADDOCK_PREFILL_CHUNK lowers it for TTFT-first setups.
const PREFILL_CHUNK_DEFAULT: usize = 8192;

/// Row-scratch cap (and single-prompt chunk size). Env-overridable, because
/// bigger chunks pack more prompt rows behind one MoE weight-read per mixed
/// tick - concentrating concurrent prefill into a few ~8k-token ticks beats
/// splitting it into many small ones. Costs ~embd*4 B/row extra scratch across
/// a handful of buffers (~1 GB at 8192). Must be >= max_batch; read once at init.
fn prefill_chunk() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_PREFILL_CHUNK")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| (256..=16384).contains(&n))
            .unwrap_or(PREFILL_CHUNK_DEFAULT)
    })
}

/// Batch above which the tensor-core MMA GEMM beats the dp4a MT tile on the
/// dense projections (qwen's measured A6000 crossover: B=32 tie, B=64 MMA
/// 1.8x). Below it the batch fits 1-2 dp4a weight passes and dp4a is
/// bandwidth-optimal on the narrow-N dense grids.
#[allow(dead_code)] // superseded by the env-tuned ks/mmq routing; kept for the measured crossover
const MMA_MIN_BATCH: usize = 32;

/// Largest batch the fused batched-dp4a MoE (mmvq-with-ids shape) serves;
/// above it the sorted mmq tiles win. dp4a re-reads each token's experts
/// (~b x 1.27 GB/step, no cross-token amortization) but launches b x n_active
/// x ff/8 well-filled GEMV blocks; mmq reads touched experts once but its few
/// blocks run latency-bound at tiny batches (5.8 ms/step at B=2 vs dp4a's
/// ~3.6 ms traffic floor). A6000 measured: dp4a wins B=2 (9.00 vs 10.74
/// ms/step) and B=3 (11.99 vs ~13.7); mmq wins from B=4 (13.75 vs 14.82).
/// GB202 measured (sm_120a, where the sorted branch rides the block-scale
/// tensor cores from B=5): dp4a wins through B=8 - 9.05 ms at B=4 (vs
/// sorted-mmq 20.0: its tile count is latency-bound and the tensor-core
/// route hasn't engaged yet), 13.5 at B=8 (vs bs 14.7) - and loses from
/// B=10 (14.45 vs bs 13.62). 2.4x the A6000's bandwidth moves dp4a's
/// traffic ceiling well past the old window. PADDOCK_MOE_DP4A_MAX overrides
/// for measurement.
const MOE_DP4A_MAX_BATCH: usize = 3;

/// The dp4a window cap for this device and model shape. sm_120a widens well
/// past the A6000-measured 3, and the crossover is EXPERT-COUNT-dependent:
/// more experts means tokens share experts later, so dp4a's per-token
/// re-read stays cheaper than the sorted path's mostly-empty 32-row tiles
/// for longer. GB202 measured at serving depth 1100 (post scale-prefetch):
/// 32 experts (20b): dp4a wins <= 7 (13.33 vs 13.63 ms at B=7), sorted
/// from 8 (13.88 vs 14.30). 128 experts (120b): B=12 is the tie (26.08 vs
/// 26.23), sorted clearly ahead by 16 (28.83 vs 32.37) - a window of 7
/// there cost c8 serving 16%.
fn moe_dp4a_max(cc_major: u32, n_experts: usize) -> usize {
    static N: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    let over = *N.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_MOE_DP4A_MAX")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
    });
    over.unwrap_or(if cc_major == 12 {
        // 128e crossover, re-measured after the warp-spec KC=256 bs rewrite:
        // from b=8 up the sorted bs route beats dp4a end to end - the old
        // threshold of 12 predated that rewrite.
        // dp4a still owns <=4 (b1 graph + dc4 shapes).
        if n_experts >= 128 { 4 } else { 7 }
    } else if cc_major == 10 {
        // B200. The generic 3 was tuned on A6000 and costs -94% at
        // b=4 here. The reason is structural, not a tuning drift: the
        // block-scale MoE kernels are NULLed off cc 12 entirely
        // (exports.cuh, mxfp4_moe_gate_up_bs/_down_bs), so on this die the
        // b>max route is sorted-mmq with no tensor-core handoff to grow
        // into - it is latency-bound on tile count and never recovers.
        // Measured 20b/32e, 3 warm reps, one binary, PADDOCK_MOE_DP4A_MAX
        // as the knob: c4 434.8/433.7/436.9 (max=3, mmq) vs
        // 814.4/867.7/845.4 (max=8, dp4a), ITL 8.44 -> 3.95 ms. c1 is
        // untouched (b=1 was already under both) at 334.9/344.2/341.7 vs a
        // recorded 345.41. b>8 is UNMEASURED on this die - max_batch was 8,
        // so the smoke band tops out at c8; mmq may still win above it, and
        // the override is the way to find out.
        8
    } else {
        MOE_DP4A_MAX_BATCH
    })
}

/// Token tile of the sorted MoE GEMM (must match PD_MOE_BM in the pack). The
/// moe_align pass pads each expert's tokens to a multiple of this.
const MOE_TILE_BM: usize = 32;

/// Max rows per speculative verify pass: 1 committed token + up to 15
/// drafts. MoE makes verify intrinsically expensive here - r rows route to
/// ~32(1-(31/32)^4r) distinct experts vs 4 for one row, so verify(8) costs
/// ~2.7 plain steps and verify(16) ~3.3 - the payoff needs LONG accepted
/// runs (verbatim re-emission), which is why the cap is high and the drafter
/// is precision-first.
const SPEC_MAX_ROWS: usize = 16;

/// Max plain graph-loop steps per no-match round (exponential backoff from
/// 2). An r=1 verify pays an upload+readback round-trip per token (~1 ms
/// over the graph loop's amortized readback); bursting keeps no-match
/// workloads at parity with `generate_greedy`, and the backoff keeps the
/// drafter reactive right after a reject (repetitive text usually re-matches
/// within a token or two) while amortizing the readback on genuinely
/// unmatched text.
const SPEC_BURST_MAX: usize = 8;

/// Max TOTAL rows per spec verify pass across all slots (the per-slot
/// serving round: each active slot contributes 1 committed token + its
/// drafts). Bounds d_logits/d_spec_pick sizing and keeps the pass under the
/// pf16-attention dispatch threshold, so a mixed-slot pass can never take
/// the uniform-slot prefill attention path. Qwen's serving-spec rule
/// (B x (K+1) <= 24) fits inside it.
const SPEC_BATCH_MAX_ROWS: usize = 32;
/// Chunked prefills that may ride one mixed tick together. Bounds the
/// attention group count and the finishing-row emit tail (n_emit = decode
/// rows + finishers <= max_batch always: a chunking slot never decodes).
const MAX_CHUNKS: usize = 8;

/// Max aligned blocks for `rows` tokens: each of `n_active` picks per token, grouped
/// by expert, each expert padded up to a full BM block.
fn moe_max_blocks(rows: usize, n_active: usize, n_expert: usize) -> usize {
    (rows * n_active + n_expert * (MOE_TILE_BM - 1)).div_ceil(MOE_TILE_BM)
}

/// Only reuse a cached prefix at least this long - below it the per-layer KV copy
/// isn't worth skipping the (cheap) short prefill.
const MIN_CACHE_PREFIX: usize = 32;

/// P5c: how many SWA-window checkpoints the paged prefix cache keeps. Each is
/// `n_swa * 2 * swa_window * kv_dim * kv_bytes` (~4.5 MB at gpt-oss-120b fp16),
/// so this many distinct reusable prefixes can carry an exact SWA resume. The
/// radix reclaims the LRU checkpoint (keeping its KV page) when they run out.
const SWA_CKPT_SLOTS: usize = 48;

/// FlashDecoding: cap on how many KV-chunks one head's attention splits into
/// (bounds the partial scratch). One block per (head, split) fills the GPU that
/// a single block-per-head launch leaves ~97% idle at batch 1.
const MAX_ATTN_SPLITS: usize = 16;
/// Target KV positions per split; split count is n_pos/this, capped. Set so split
/// only kicks in at genuinely long context: splitting below
/// a few hundred positions costs more (extra combine launch) than the serial loop
/// it shortens, because at short context attention isn't the bottleneck. Above it,
/// the single block-per-head kernel would serialize the whole KV range and split-K
/// keeps attention flat (~5 ms at n_pos 1100 instead of scaling ~8×).
#[allow(dead_code)] // superseded by attn_fill_blocks (SM-scaled); kept for the measured crossover
const ATTN_SPLIT_CHUNK: usize = 256;
/// Batched FlashDecoding: block-count target the split aims to fill - ~3× the
/// device SM count, for occupancy headroom. Was a constant 256 (~3× the
/// A6000's 84 SMs), which silently left most of a 188-SM Blackwell idle; the
/// target now scales with the card. The batched grid is n_heads*batch blocks
/// at n_splits=1; when that underfills, split each sequence's KV range so
/// n_heads*batch*n_splits ≈ this. At high batch the grid is already full ->
/// n_splits=1 (no combine overhead). See [`attn_splits`].
fn attn_fill_blocks(sm_count: usize) -> usize {
    3 * sm_count
}

/// Debug knob to force the plain single-kernel batched attention (no FlashDecoding
/// split), for A/B-ing the split's win at long context. Default: split enabled.
static ATTN_SPLIT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Enable/disable the batched FlashDecoding split globally (default on). Diagnostic.
pub fn set_attn_split(on: bool) {
    ATTN_SPLIT.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Routes prefill-sized MoE passes through the sm_120a block-scale kernels
/// (mxFP4 x FP8, hardware ue8m0 scaling) when the pack + device support them
/// (default on there). fp8-activation numeric class: spec-vs-plain greedy
/// exactness gates pin this off end to end (see the spec parity tests);
/// PADDOCK_NO_MOE_BS pins it off process-wide.
static MOE_BS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Enable/disable the block-scale MoE route globally (default on where
/// supported). Diagnostic/override - the spec exactness gates use this.
pub fn set_moe_bs(on: bool) {
    MOE_BS.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// How many KV-chunks to split each sequence's attention into on the batched path.
/// Returns 1 (the plain single-kernel path) when the grid already fills the GPU;
/// otherwise splits to ~2×[`attn_fill_blocks`] blocks REGARDLESS of context length.
/// The old `n_pos <= ATTN_SPLIT_CHUNK` short-context gate dated from when the step
/// was ~25 ms and attention invisible - at the G0 step (~7 ms) the unsplit B=1
/// kernel (64 blocks × 64 threads, per-position __syncthreads walk) measured
/// 58.7 µs/layer at n_pos ≈ 100 = 19% of the token. Splits are cheap when chunks
/// come up empty (the combine drops -inf partials - parity-tested over-split), and
/// a POSITION-INDEPENDENT count is what lets the captured B=1 decode graph replay
/// correctly and fast at any position (the partial kernel derives its chunk from
/// the device position each replay; only n_splits is baked at record time).
/// True when the pack's batched partial fuses the q-group per KV head for
/// this shape (one K/V stage serves all `group` q-heads). Must mirror the
/// launcher's own predicate (incl. the PD_NO_GQA_FUSE pin) - split
/// budgeting and the partial-vs-plain dispatch both key off it. Fused
/// shapes route through partial+combine even at n_splits == 1: the plain
/// per-q-head kernel re-reads each K/V tile group(=8)x (the 128-key SWA
/// layers spent 106 us/launch there vs ~30 through the fused walk).
fn attn_gqa_fused(n_heads: usize, n_kv_heads: usize, batch: usize) -> bool {
    let group = n_heads.checked_div(n_kv_heads).unwrap_or(1);
    batch > 1
        && (2..=8).contains(&group)
        && n_kv_heads >= 4
        && n_heads == n_kv_heads * group
        && std::env::var_os("PD_NO_GQA_FUSE").is_none()
}

fn attn_splits(
    n_heads: usize,
    n_kv_heads: usize,
    batch: usize,
    _n_pos: usize,
    fill_blocks: usize,
) -> usize {
    // Split budgeting must count KV-HEAD blocks for fused shapes, not
    // q-head blocks. The old q-head base returned n_splits=1 for every
    // B >= 16 and sent serving decode to the plain per-q-head kernel, which
    // re-reads each K/V tile group(=8)x: +10.7 ms/step at B=32 depth 1100 on
    // GB202 (~11% of bandwidth). batch == 1 keeps the q-head base - the B=1
    // decode graph bakes its split count and its economics were tuned
    // separately.
    let fused = attn_gqa_fused(n_heads, n_kv_heads, batch);
    let base = if fused {
        n_kv_heads * batch
    } else {
        n_heads * batch
    };
    if !ATTN_SPLIT.load(std::sync::atomic::Ordering::Relaxed) || base >= fill_blocks {
        return 1;
    }
    if fused {
        // budget: n_heads*batch*splits (the per-q-head partial layout the
        // GQA kernel still writes) must stay <= fill*2*MAX splits of scratch
        // (d_attn_o is sized with the 2x fused headroom); the 4x-fill target
        // keeps the fused walk's serial chains short - the per-block tile
        // chain, not traffic, dominates at decode depths. The depth cap
        // keeps every split >= ~4 tiles (over-fragmenting paid more fixed
        // cost than it hid); position-dependent is fine - the batched path
        // is eager, no baked graph.
        // Each split covers >= 4 tiles (128 keys): finer splits over-fragment
        // the fused serial walk and the extra combine partials cost more than
        // they hide (measured on long-context full-attn). Dropping SWA layers to
        // a 2-tile/64-key floor to chase grid fill was falsified - attention
        // isn't the bottleneck at high concurrency, the tick is MoE
        // weight-BW-bound - so the floor stays 128 keys.
        let tile_keys = 128;
        let depth_cap = _n_pos.div_ceil(tile_keys).max(1);
        ((fill_blocks * 4) / base)
            .min(depth_cap)
            .clamp(1, MAX_ATTN_SPLITS)
    } else {
        (fill_blocks * 2).div_ceil(base).clamp(1, MAX_ATTN_SPLITS)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GpuModelError {
    #[error(transparent)]
    Gpu(#[from] GpuError),
    #[error("missing metadata key: {0}")]
    MissingMeta(String),
    /// The pre-load VRAM admission gate refused (honest will-it-fit: an
    /// oversubscribed card pages to system RAM and freezes the machine).
    #[error("{0}")]
    WontFit(String),
    #[error("batch not enabled - call enable_batch first")]
    BatchDisabled,
    #[error("batch size {got} exceeds max_batch {max}")]
    BatchTooLarge { got: usize, max: usize },
    #[error("prompt length {got} exceeds context window {max}")]
    ContextExceeded { got: usize, max: usize },
    /// The paged KV pool had no free block for a growth this step. The scheduler
    /// catches this (vs. a generic backend error) and PREEMPTS a victim sequence
    /// - freeing its blocks and re-queueing it for recompute - instead of failing
    ///   the whole batch (P5b-3).
    #[error("KV pool exhausted")]
    PoolExhausted,
    /// Startup config the engine can PROVE infeasible at load time (e.g.
    /// --max-ctx × --max-batch beyond the KV budget). The scheduler treats this
    /// as FATAL - no width halving, no serial fallback - so the server refuses
    /// to start with the actionable message instead of silently serving a
    /// different config than the user asked for.
    #[error("{0}")]
    Config(String),
    #[error("{0}")]
    Unsupported(String),
}

/// Batched-decode state: per-slot KV caches + batched scratch, allocated by
/// `enable_batch`. Holds B sequences' worth of everything so `forward_batch`
/// runs one weight-amortized pass over the whole batch.
struct BatchState {
    max_batch: usize,
    /// number of transient rows the scratch buffers hold - max(max_batch,
    /// PREFILL_CHUNK), so a prefill chunk of up to PREFILL_CHUNK tokens fits.
    row_cap: usize,
    k_cache: Vec<CudaSlice<u8>>, // per layer, [max_batch * max_ctx * kv_dim], fp8 E4M3
    v_cache: Vec<CudaSlice<u8>>,
    d_x: CudaSlice<f32>,
    d_xn: CudaSlice<f32>,
    d_q: CudaSlice<f32>,
    d_qkv: CudaSlice<f32>,
    d_kv: CudaSlice<f32>,
    d_attn: CudaSlice<f32>,
    // batched FlashDecoding partials: o [<=fill_blocks*MAX_ATTN_SPLITS, head_dim],
    // ml [.., 2] - split only fires at low batch, so the tuple count stays bounded
    d_attn_o: CudaSlice<f32>,
    d_attn_ml: CudaSlice<f32>,
    d_proj: CudaSlice<f32>,
    d_router: CudaSlice<f32>,
    d_gate_up: CudaSlice<f32>,
    d_logits: CudaSlice<f32>,
    d_topk_idx: CudaSlice<u32>,
    d_topk_w: CudaSlice<f32>,
    // sorted MoE (moe_align) buffers for the int8 mmq / block-scale routes
    // (b above the dp4a window).
    d_sorted_row: CudaSlice<u32>,
    d_sorted_slot: CudaSlice<u32>,
    d_block_expert: CudaSlice<u32>,
    // (the fused down_bs residual fold's last-arrival counter buffer lived
    // here; retired with that arm - see git history if the fold returns)
    // B=1 fast-path activation quant (int8 + per-32-block scales) for the dp4a
    // GEMV/MoE kernels; sized for the largest single-row activation (n_active·ff).
    d_b1_xq: CudaSlice<i8>,
    d_b1_xs: CudaSlice<f32>,
    // batch>1 dense int8 staging: strided int8 + scales for the mma route
    // (batch <= 64), flat mmq layout for batch > 64, and the stream-k fixup
    // scratch (sized by the 256-SM contract in the pack).
    d_p_xq: CudaSlice<i8>,
    d_p_xs: CudaSlice<f32>,
    d_p_yq: CudaSlice<u8>,
    d_p_skfix: CudaSlice<f32>,
    // sorted-MoE mmq staging: the swiglu output (fused_sorted) re-quantized
    // strided for the int8 down GEMM ([max_blocks*BM, ff]).
    d_moe_xq: CudaSlice<i8>,
    d_moe_xs: CudaSlice<f32>,
    // block-scale (sm_120a) MoE staging: ue8m0 activation/swiglu scale bytes
    // (the e4m3 values reuse d_p_xq / d_moe_xq as raw byte planes).
    d_p_xs8: CudaSlice<u8>,
    d_moe_xs8: CudaSlice<u8>,

    // B=1 graph-resident greedy loop: device token + on-device argmax scratch +
    // the captured step graph. The graph captures pointers into this BatchState,
    // so it lives and dies with it (a re-enable_batch drops and recaptures).
    d_g_token: CudaSlice<u32>,
    d_g_mrope: CudaSlice<u32>, // dummy [4]: argmax_advance bumps it; gpt-oss has no mrope
    d_g_out: CudaSlice<u32>,   // [GEN_CHUNK] token ring
    d_g_step: CudaSlice<u32>,
    d_g_pmax: CudaSlice<f32>,
    d_g_pidx: CudaSlice<u32>,
    gen_graph: Option<SendGraph>,
    // Prefill chunk graphs, keyed by chunk size (launch grids bake the row
    // count; positions/slots/tokens come from the fixed d_pf_* buffers on the
    // model). Captures point into this BatchState, so the cache lives and
    // dies with it. Bounded like qwen's (chat traffic -> many tail sizes).
    pf_graphs: std::collections::HashMap<usize, SendGraph>,
    // Batched decode step graphs, keyed by batch size (continuous batching
    // shrinks/grows b as sequences finish, so several coexist). Same fixed
    // d_pf_tok/d_pf_pos inputs and same staleness rules as pf_graphs.
    step_graphs: std::collections::HashMap<usize, SendGraph>,
    // Spec-verify pass graphs, keyed by row count (1 committed + drafts).
    // Same fixed inputs and staleness rules as step_graphs.
    spec_graphs: std::collections::HashMap<usize, SendGraph>,
    // Device argmax of each verify row (only these u32s cross the bus; the
    // r x vocab verify logits stay on device).
    d_spec_pick: CudaSlice<u32>,
    // Fused decode sampling (forward_batch_sampled): per-row packed params
    // {inv_t, u, mode, pad} in and per-row token ids out - only these cross
    // the bus instead of the [b, vocab] logits (25.7 MB/step at B=32).
    d_samp_par: CudaSlice<u32>,
    d_samp_out: CudaSlice<u32>,
    /// mode-5/6 truncation side plane [max_batch, 4] {k, top_p bits, min_p bits,
    /// pad} - gpt-oss elects no sampling, so this serves DIALLED requests
    d_samp_tpar: CudaSlice<u32>,
    /// pipe ring twin ([2, max_batch, 4])
    d_pipe_tpar: CudaSlice<u32>,
    // Pipelined decode (depth-2): double-buffered sampler params/ids so tick
    // N+1's param upload and sampler writes never touch the ring slot tick N
    // still reads (or the host is still reading back). out is alloc-zeroed
    // and mode-0 rows are never written, so hole rows forever feed token 0.
    d_pipe_par: CudaSlice<u32>, // [2 * max_batch * 4]
    d_pipe_out: CudaSlice<u32>, // [2 * max_batch]

    // Paged KV (G3, default on; PADDOCK_NO_PAGED_KV pins dense). When active,
    // k_cache/v_cache are BLOCK
    // POOLS [n_blocks, 16, kv_dim] and every K/V read/write goes through a
    // block table instead of the dense slot*max_ctx stride. Two STATIC tables
    // (precomputed once, graph-safe - captured by pointer, contents never
    // change): full-attn layers use an identity map (same VRAM as dense); SWA
    // layers use a WindowRing map bt[slot*bps+j] = slot*ring + j%ring, so their
    // pool caps at `swa_ring_blocks` (ceil(w/16)+1) blocks/slot instead of the
    // full max_ctx - the ~57× SWA KV saving. bps = max_ctx/16 for both kinds.
    paged: bool,
    blocks_per_slot: usize,
    swa_ring_blocks: usize,
    // [full-attn table, SWA WindowRing] - indexed by `layer.is_swa as usize`.
    // In plain paged mode (G3) d_bt[0] is a STATIC identity map; in POOL mode
    // (G4a, below) it is DYNAMIC - grown per step from `tables` + re-uploaded.
    d_bt: [Option<CudaSlice<u32>>; 2],

    // G4a paged budget POOL - auto-sized by default in enable_batch, or
    // pinned via PADDOCK_KV_POOL_BLOCKS. When Some, the full-attn
    // layers draw their KV from a shared `KvPool` of N 16-token blocks instead
    // of the per-slot `batch × max_ctx` reservation - KV decoupled from
    // batch×ctx, the config-independent memory win PagedAttention buys.
    // Only the full-attn
    // table (`d_bt[0]`) becomes dynamic; the SWA layers keep their static
    // WindowRing (`d_bt[1]`) from G3. `tables[slot]` is the per-slot
    // logical->physical map grown on demand; `block_table_host` mirrors `d_bt[0]`
    // ([max_batch*bps] u32) and is re-uploaded (outside any graph) when a slot
    // grows. None in dense / plain-paged mode.
    pool: Option<KvPool>,
    tables: Vec<BlockTable>,
    block_table_host: Vec<u32>,
    // P5c zero-copy prefix reuse under the pool. `paged_prefix` is a radix over
    // the full-attn pool blocks (shared, refcounted via `pool`); a hit adopts
    // the prefix's blocks with `BlockTable::share_prefix` (no copy). Because the
    // 18 SWA layers live in the per-slot WindowRing (not the pool) and can't be
    // block-shared, each cached node also carries a checkpoint of the trailing
    // `swa_window`-token SWA KV in `d_swa_ckpt` (indexed by the radix state idx),
    // restored into the resuming slot's ring so resume is exact with zero warmup
    // (else a shared-prefix resume would need ~swa_window*n_swa burn-in). None
    // unless the pool is active.
    paged_prefix: Option<PagedRadix>,
    // Device pool of SWA-window-KV checkpoints, `swa_ckpt_bytes` each, indexed by
    // `PagedRadix` state indices. Layout per ckpt: [swa_layer][K,V][window/16
    // blocks][16*kv_dim] raw KV bytes, window blocks in logical (ascending) order.
    d_swa_ckpt: Option<CudaSlice<u8>>,
    swa_ckpt_bytes: usize,
}

/// Panic message when the raw output.weight was dropped by PADDOCK_GPTOSS_SLIM_HEAD
/// yet a large-batch (b>64) lm_head tried to read it. Only reachable if the
/// opt-in flag is combined with batched/spec serving - loud, not silent.
const RAW_HEAD_DROPPED: &str = "raw lm_head dropped by PADDOCK_GPTOSS_SLIM_HEAD (single-user only; unset it for batched/spec >64-row heads)";

struct GpuLayer {
    attn_norm: DeviceTensor,
    // Biases stay resident (F32, tiny). The raw-Q8_0 attention weights that fed
    // the PADDOCK_NO_MMQ / PADDOCK_NO_DP4A_B1 exact-f32 q8_0_gemm lanes were
    // collapsed along with those lanes - the llama.cpp same-weights greedy
    // gates are the correctness bar, and the pinned MoE half of that class had
    // been reading the fused g||u ILV plane as if it were plain.
    bq: DeviceTensor,
    bk: DeviceTensor,
    bv: DeviceTensor,
    bo: DeviceTensor,
    // repacked (aligned data + f16-scale streams) weights for the int8
    // tensor-core batch path (mmq/mma - llama's own prefill numeric class).
    // The sole attention-weight copy since the exact-f32 lane collapse.
    wq_r: RepackedQ8,
    // fused [q|k|v] planes for the batched path: one wide GEMM + one fused
    // rope/append launch instead of three narrow wave-starved GEMMs + four
    // glue launches (the vllm qkv_proj shape; ~150 -> ~60 us/layer of decode
    // dense chain). wq_r/wk_r/wv_r stay for the B=1 graph.
    wqkv_r: RepackedQ8,
    bqkv: CudaSlice<f32>,
    wk_r: RepackedQ8,
    wv_r: RepackedQ8,
    wo_r: RepackedQ8,
    /// per-head sink logits, resident on device for the batched attention kernel
    sinks_dev: CudaSlice<f32>,
    post_norm: DeviceTensor,
    // the router is a tiny F32 weight (2880×32) - no bandwidth to reclaim, so it
    // stays a dense cuBLAS matvec (only the Q8_0 weights go through q8_0_gemv)
    router_w: DeviceTensor,
    router_b: DeviceTensor,
    // expert weights repacked to aligned data + split e8m0 scales (the sole copy -
    // all MoE paths read this layout; the 17-byte upload is freed after repack).
    gate_exps: RepackedMxfp4,
    gate_exps_b: DeviceTensor,
    up_exps: RepackedMxfp4,
    up_exps_b: DeviceTensor,
    down_exps: RepackedMxfp4,
    down_exps_b: DeviceTensor,
    is_swa: bool,
}

pub struct GpuGptOss {
    exec: Arc<GpuExecutor>,
    // geometry
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    embd: usize,
    head_dim: usize,
    n_experts: usize,
    n_active: usize,
    ff_exp: usize,
    swa_window: usize,
    rms_eps: f32,
    pub vocab: usize,
    pub(crate) max_ctx: usize,
    pos: usize,
    /// Pins the spec machinery (prompt ingestion, drafts, verify) to the
    /// int8 MoE classes: spec-vs-plain greedy equality requires one numeric
    /// class end to end, and the block-scale fp8 prefill flips near-ties.
    moe_bs_pin: bool,
    yarn_params: (f32, f32, f32, f32, f32, f32),
    // weights
    // Input embedding table kept RESIDENT as Q8_0 (input row-gather only; the untied
    // output head is a separate weight). We gather+dequant the rows in flight rather
    // than dequantizing the whole table to a 4x-larger f32 plane.
    tok_embd: QuantTensor,
    layers: Vec<GpuLayer>,
    out_norm: DeviceTensor,
    // lm_head is the single largest per-token GEMV (2880×201088); Q8_0-resident
    // Raw output.weight - feeds only the q8_0_gemm large-batch (b>64) lm_head
    // fallback. None when PADDOCK_GPTOSS_SLIM_HEAD is set (opt-in, single-user:
    // the head is always b=1 -> output_r, so raw is unused).
    output: Option<QuantTensor>,
    // repacked copy for the B=1 dp4a fast path's nc GEMV (aligned data + scale
    // streams -> vectorized loads, ~89% DRAM vs the interleaved kernel's ~76%).
    // Dual-copy (+0.6 GB) until the G1 storage migration unifies all consumers.
    output_r: RepackedQ8,
    // Resident weight bytes, sampled at the end of load (weights up, KV not
    // yet - this family's whole KV lives in `batch`, which enable_batch
    // builds later). The weights line of the memory-breakdown API and the
    // number gen-shapes.py publishes as `source = "measured"`.
    weights_bytes: Option<u64>,
    // diagnostic: skip the per-token logits readback (so the CPU can launch the
    // next token's kernels without waiting on the GPU) - measures launch/sync
    // overhead vs GPU-bound before committing to CUDA graphs.
    skip_readback: bool,
    // KV cache element type (default fp16, greedy-exact). fp8 is opt-in and must be
    // set before enable_batch (it sizes the caches). See [`KvDtype`].
    kv_dtype: KvDtype,
    // batched-decode state (None until enable_batch); the single unified forward
    // path - forward_one is just a batch of one through it. The continuous-batching
    // path also lives here.
    batch: Option<BatchState>,
    // Device position for the graph-resident greedy loop. Lives on the model (not
    // BatchState) as an Option so record_gen_step can move it out around the
    // `&mut self` run_layers call (a host-side move - the device address the graph
    // captures is unchanged). Survives enable_batch; the graph itself does not.
    d_g_pos: Option<CudaSlice<u32>>,
    // Prefill chunk inputs (tokens/positions/slots), fixed device addresses the
    // prefill graphs read - same Option take/put trick as d_g_pos. Sized to
    // row_cap on first prefill (re-sized if a bigger enable_batch raises it).
    d_pf_tok: Option<CudaSlice<u32>>,
    d_pf_pos: Option<CudaSlice<u32>>,
    d_pf_slots: Option<CudaSlice<u32>>,
    // Cumulative pages adopted zero-copy from the paged radix prefix cache
    // (test/telemetry hook; the dense RadixKvCache is gone).
    paged_reused_blocks: u64,
    // per-slot tokens served from the prefix cache by the last prefill (usage
    // reporting; taken and zeroed by take_prefill_reused)
    last_reused: Vec<usize>,
    // in-flight CHUNKED prefills (vLLM-class continuous batching): prompts
    // advance a shared budget of rows per forward_mixed tick alongside the
    // live decode rows instead of stalling them. SEVERAL ride together
    // (FIFO-filled, bounded by MAX_CHUNKS) - the one-at-a-time gate made a
    // staggered admission wave serialize into N sequential heavy mixed ticks
    // and split serving into a fast cohort mode / slow staggered mode
    // depending on how arrivals happened to land.
    chunked: Vec<ChunkedPrefill>,
    // in-flight PIPELINED decode (depth-2): tick N+1 is enqueued - fed on
    // device from tick N's sampled ids - before tick N's ids reach the host,
    // so commit/SSE work overlaps the GPU instead of gapping it. The service
    // drains before any other forward call.
    pipe: Option<PipeState>,
}

/// State of the in-flight pipelined decode: `tick` = the last ENQUEUED tick,
/// `ev[tick % 2]` fires when that tick's out-ring slot is readable.
struct PipeState {
    b: usize,
    tick: u64,
    ev: [Option<cudarc::driver::CudaEvent>; 2],
}

/// State of the one in-flight chunked prefill: `tokens[done..]` still needs
/// KV (done starts at the prefix-cache hit length).
struct ChunkedPrefill {
    slot: usize,
    tokens: Vec<u32>,
    done: usize,
}

impl GpuGptOss {
    pub fn load(
        exec: Arc<GpuExecutor>,
        map: &MappedGguf,
        max_ctx: usize,
    ) -> Result<Self, GpuModelError> {
        exec.vram_load_gate(map.total_len(), "gpt-oss")
            .map_err(GpuModelError::WontFit)?;
        let u = |k: &str| {
            map.gguf()
                .arch_field(k)
                .and_then(Value::as_u64)
                .ok_or_else(|| GpuModelError::MissingMeta(k.to_owned()))
        };
        let f = |k: &str| map.gguf().arch_field(k).and_then(Value::as_f32);

        let n_layers = u("block_count")? as usize;
        let n_heads = u("attention.head_count")? as usize;
        let n_kv_heads = u("attention.head_count_kv")? as usize;
        // The g||u interleaved expert layout (see the layer loop) is the
        // on-device layout now. The legacy mmq/grouped/gemm_sorted MoE
        // routes read the old two-plane layout and would serve garbage -
        // reject their pin loudly, and require the bs-capable pack.
        if paddock_models::dev_var_os!("PADDOCK_NO_MOE_BS").is_some() {
            return Err(GpuModelError::Unsupported(
                "PADDOCK_NO_MOE_BS: the legacy MoE routes cannot read the g||u \
                 interleaved expert layout; unset the pin"
                    .into(),
            ));
        }
        if !exec.has_gu_interleave() {
            return Err(GpuModelError::Unsupported(
                "kernel pack lacks mxfp4_gu_interleave (gpt-oss needs the sm120 bs pack)".into(),
            ));
        }
        let embd = u("embedding_length")? as usize;
        let head_dim = u("attention.key_length")
            .map(|v| v as usize)
            .unwrap_or(embd / n_heads);
        let n_experts = u("expert_count")? as usize;
        let n_active = u("expert_used_count")? as usize;
        let ff_exp = u("expert_feed_forward_length")? as usize;
        let swa_window = u("attention.sliding_window")? as usize;
        let rms_eps = f("attention.layer_norm_rms_epsilon").unwrap_or(1e-5);

        let yarn = YarnRope::new(
            head_dim,
            f("rope.freq_base").unwrap_or(150_000.0),
            1.0 / f("rope.scaling.factor").unwrap_or(1.0),
            u("rope.scaling.original_context_length").unwrap_or(4096) as usize,
            1.0,
            1.0,
            32.0,
            1.0,
        );

        // Per-component VRAM ledger (see qwen35::load) - snapshot free VRAM between
        // load phases so the on-GPU footprint is visible, not guessed.
        let vfree = || {
            cudarc::driver::result::mem_get_info()
                .map(|(f, _)| f as u64)
                .unwrap_or(0)
        };
        let gb = |used: u64| used as f64 / 1e9;
        let v_start = vfree();
        // token_embd kept RESIDENT as Q8_0 (input row-gather only); gather+dequant the
        // rows in flight instead of paying 4x VRAM for a dequantized f32 table.
        let tok_embd = exec.upload_raw(map, "token_embd.weight")?;
        let vocab = tok_embd.dims[1];
        let v_embd = vfree();
        tracing::info!(
            "gpt-oss VRAM  input embeddings token_embd (Q8_0)      {:>7.2} GB",
            gb(tok_embd.bytes.len() as u64)
        );

        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let dt = |name: &str| exec.upload(map, &format!("blk.{i}.{name}"));
            let qt = |name: &str| exec.upload_raw(map, &format!("blk.{i}.{name}"));
            let rq = |name: &str| exec.repack_q8(map, &format!("blk.{i}.{name}"));
            // expert weights: upload the 17-byte MXFP4 transiently, repack to the
            // aligned (data, scale) layout every MoE path now reads, then drop the
            // 17-byte copy (the temporaries free at end of scope -> no extra VRAM).
            // g||u INTERLEAVE layout: the bs pair and the dp4a
            // MoE stream one 128 B pair [gate 64 | up 64] per (row, KC=128
            // chunk) - see pd_mxfp4_gu_interleave. gate_exps.data holds the
            // fused plane; up_exps.data is a 16 B DUMMY the kernels never
            // dereference (all bytes come through the gate_data pointer) -
            // kept so wrapper/ABI signatures stay untouched. Scales remain
            // per-plane. Sources drop per layer (transient ~1 GB).
            let gate_src = exec.repack_mxfp4(&qt("ffn_gate_exps.weight")?)?;
            let up_src = exec.repack_mxfp4(&qt("ffn_up_exps.weight")?)?;
            let gu = exec.gu_interleave(&gate_src, &up_src, embd / 32, n_experts * ff_exp)?;
            let RepackedMxfp4 {
                data: _gate_drop,
                scale: gate_scale,
            } = gate_src;
            let RepackedMxfp4 {
                data: _up_drop,
                scale: up_scale,
            } = up_src;
            let gate_exps = RepackedMxfp4 {
                data: gu,
                scale: gate_scale,
            };
            let up_exps = RepackedMxfp4 {
                data: exec.alloc_u8(16)?,
                scale: up_scale,
            };
            let down_exps = exec.repack_mxfp4(&qt("ffn_down_exps.weight")?)?;
            let wq_r = rq("attn_q.weight")?;
            let wk_r = rq("attn_k.weight")?;
            let wv_r = rq("attn_v.weight")?;
            let wqkv_r = exec.concat_q8(&[&wq_r, &wk_r, &wv_r])?;
            let bq = dt("attn_q.bias")?;
            let bk = dt("attn_k.bias")?;
            let bv = dt("attn_v.bias")?;
            let bqkv = exec.concat_f32(&[&bq.buf, &bk.buf, &bv.buf])?;
            layers.push(GpuLayer {
                attn_norm: dt("attn_norm.weight")?,
                bq,
                bk,
                bv,
                bo: dt("attn_output.bias")?,
                wq_r,
                wqkv_r,
                bqkv,
                wk_r,
                wv_r,
                wo_r: rq("attn_output.weight")?,
                // sinks stay resident on device for the batched attention kernel
                sinks_dev: dt("attn_sinks.weight")?.buf,
                post_norm: dt("post_attention_norm.weight")?,
                router_w: dt("ffn_gate_inp.weight")?,
                router_b: dt("ffn_gate_inp.bias")?,
                gate_exps,
                gate_exps_b: dt("ffn_gate_exps.bias")?,
                up_exps,
                up_exps_b: dt("ffn_up_exps.bias")?,
                down_exps,
                down_exps_b: dt("ffn_down_exps.bias")?,
                is_swa: i % 2 == 0,
            });
        }
        let v_back = vfree();
        tracing::info!(
            "gpt-oss VRAM  backbone (attn Q8_0 + MoE mxfp4 experts) {:>7.2} GB",
            gb(v_embd.saturating_sub(v_back))
        );
        // of-which line: the fused qkv concat is a full second residency of
        // q|k|v - the splits stay for the B=1 graph - and it is big enough
        // (~0.53 GiB on 120b) that the VRAM report has to name it
        let qkv_dup: u64 = layers
            .iter()
            .map(|l| (l.wqkv_r.data.len() + l.wqkv_r.scale.len() + l.bqkv.len() * 4) as u64)
            .sum();
        tracing::info!(
            "gpt-oss VRAM    of which fused wqkv|bqkv duplicate planes {:>5.2} GB \
             (splits stay for the b=1 graph)",
            gb(qkv_dup)
        );
        // Output head as locals so its VRAM is measurable. output_r (repacked) is the
        // fast dp4a/mma logits path. The RAW output.weight is a second copy read only
        // by the q8_0_gemm large-batch (b>64) lm_head fallback.
        // Single-user greedy decode emits one row at a time, so its head is
        // always b=1 -> output_r; the raw copy is never touched. PADDOCK_GPTOSS_SLIM_HEAD
        // (default off, opt-in) drops the raw copy to save ~0.75 GB. Do not set it for
        // batched/spec serving that computes >64-row logits in one pass - that path
        // .expect()s the raw copy and panics loudly (a clear error, never silent).
        let out_norm = exec.upload(map, "output_norm.weight")?;
        let slim_head = paddock_models::dev_var_os!("PADDOCK_GPTOSS_SLIM_HEAD").is_some();
        let output = if slim_head {
            None
        } else {
            Some(exec.upload_raw(map, "output.weight")?)
        };
        let output_r = exec.repack_q8(map, "output.weight")?;
        let v_head = vfree();
        tracing::info!(
            "gpt-oss VRAM  output head ({:<18}) {:>7.2} GB",
            if slim_head {
                "repacked only/SLIM"
            } else {
                "raw + repacked"
            },
            gb(v_back.saturating_sub(v_head))
        );
        tracing::info!(
            "gpt-oss VRAM  = model resident total                  {:>7.2} GB",
            gb(v_start.saturating_sub(v_head))
        );
        // The PUBLISHED resident-weight line - the pool's own used counter,
        // not the vfree() bracket the log above prints. That bracket counts
        // the CUDA context, modules and cuBLAS workspaces as weights and does
        // not reproduce between loads; this one is exact and repeatable.
        // Everything below allocates nothing, so weights are complete here.
        let weights_bytes = exec.settled_mem_used();

        Ok(Self {
            // clone the handle for the field; the local `exec` stays live for
            // the remaining alloc calls in this literal (evaluated in order)
            exec: exec.clone(),
            n_layers,
            n_heads,
            n_kv_heads,
            embd,
            head_dim,
            n_experts,
            n_active,
            ff_exp,
            swa_window,
            rms_eps,
            vocab,
            max_ctx,
            pos: 0,
            moe_bs_pin: false,
            yarn_params: yarn.kernel_params(),
            tok_embd,
            out_norm,
            output,
            output_r,
            layers,
            weights_bytes,
            skip_readback: false,
            kv_dtype: KvDtype::Fp16,
            batch: None,
            d_g_pos: None,
            d_pf_tok: None,
            d_pf_pos: None,
            d_pf_slots: None,
            paged_reused_blocks: 0,
            last_reused: Vec::new(),
            chunked: Vec::new(),
            pipe: None,
        })
    }

    pub fn reset(&mut self) {
        self.pipe_abort();
        self.pos = 0;
        // single-stream (forward_one) reuses slot 0 - return its pool blocks so
        // the next sequence regrows from zero (no-op outside pool mode).
        self.pool_clear_slot(0);
    }

    /// Select the KV cache element type (default [`KvDtype::Fp16`], greedy-exact).
    /// [`KvDtype::Fp8E4m3`] is a lossy opt-in throughput/memory mode. Drops any
    /// existing batched state so the caches re-allocate at the new element size on
    /// the next `enable_batch`/`forward_one`; call it before serving.
    pub fn set_kv_dtype(&mut self, dtype: KvDtype) {
        self.pipe_abort();
        self.kv_dtype = dtype;
        self.batch = None;
        self.pos = 0;
    }

    pub fn forward_one(&mut self, token: u32) -> Result<Vec<f32>, GpuModelError> {
        // Single-stream decode is a batch of one through the UNIFIED batched path -
        // the same kernels forward_batch uses (gemm QKV, batched attention with the
        // FlashDecoding split, grouped MoE), not a separate gemv/attn/MoE path. So
        // there is no gemv-vs-gemm divergence to reconcile: b=1 and b=N run identical
        // math. A b=1 batch is lazily allocated the first time this is called.
        if self.batch.is_none() {
            self.enable_batch(1)?;
        }
        let exec = self.exec.clone();
        let (embd, rms_eps, vocab) = (self.embd, self.rms_eps, self.vocab);
        let pos = self.pos;
        assert!(
            pos < self.max_ctx,
            "context overflow is an error (honest-failure rule)"
        );
        // G4a: single-stream decode is slot 0 - grow its pool table to `pos`
        // before the append/read (no-op outside pool mode).
        self.ensure_pool_rows(&[0], &[pos as u32])?;
        let d_pos = exec
            .stream
            .clone_htod(&vec![pos as u32])
            .map_err(|e| GpuError::Driver(e.to_string()))?;
        self.embed_gather(&[token])?;
        self.run_layers(1, &d_pos, None, pos, false, None)?;

        let bs = self.batch.as_mut().ok_or(GpuModelError::BatchDisabled)?;
        exec.rmsnorm_batch(&bs.d_x, &self.out_norm.buf, &mut bs.d_xn, embd, rms_eps, 1)?;
        exec.quantize_q8(&bs.d_xn, &mut bs.d_b1_xq, &mut bs.d_b1_xs, embd)?;
        exec.q8_0_gemv_dp4a_nc(
            &self.output_r,
            &bs.d_b1_xq,
            &bs.d_b1_xs,
            &mut bs.d_logits,
            1,
        )?;

        self.pos += 1;
        if self.skip_readback {
            return Ok(Vec::new());
        }
        let bs = self.batch.as_ref().ok_or(GpuModelError::BatchDisabled)?;
        let logits = exec.to_host(&bs.d_logits)?;
        Ok(logits[..vocab].to_vec())
    }

    /// Diagnostic: is decode launch/sync-bound or GPU-bound? Runs `n` decode
    /// steps synced-per-token (the normal path, GPU waits each token) vs no-sync
    /// (skip readback so the CPU launches ahead, one sync at the end). If no-sync
    /// is much faster, per-launch/sync overhead is exposed -> CUDA graphs will
    /// help; if equal, we're GPU-bound -> graphs would be neutral. Returns
    /// (synced_ms_per_tok, nosync_ms_per_tok).
    pub fn bench_launch_bound(
        &mut self,
        prompt: &[u32],
        n: usize,
    ) -> Result<(f64, f64), GpuModelError> {
        self.reset();
        let mut last = 0u32;
        for &tk in prompt {
            let l = self.forward_one(tk)?;
            last = l
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map_or(0, |(i, _)| i) as u32;
        }
        // synced (normal): to_host each token forces a GPU wait
        for _ in 0..8 {
            self.forward_one(last)?;
        }
        let t0 = std::time::Instant::now();
        for _ in 0..n {
            self.forward_one(last)?;
        }
        let synced = t0.elapsed().as_secs_f64() * 1e3 / n as f64;
        // no-sync: skip readback, one sync at the end via a readback
        self.skip_readback = true;
        for _ in 0..8 {
            self.forward_one(last)?;
        }
        self.exec.synchronize()?; // sync (skip_readback: nothing read back)
        let t1 = std::time::Instant::now();
        for _ in 0..n {
            self.forward_one(last)?;
        }
        self.exec.synchronize()?; // one sync at the end
        let nosync = t1.elapsed().as_secs_f64() * 1e3 / n as f64;
        self.skip_readback = false;
        Ok((synced, nosync))
    }

    /// Measured device bytes this model holds (weights + KV caches) - see
    /// `GpuExecutor::process_mem_used`.
    /// Return the pool's freed memory to the driver (`Generator::release_idle_memory`).
    pub(crate) fn trim_idle(&self) {
        self.exec.trim_mem_pool();
    }

    pub fn device_mem_used(&self) -> Option<u64> {
        self.exec.process_mem_used()
    }

    /// Resident weight bytes measured at load - see the `weights_bytes` field.
    pub fn weights_mem_bytes(&self) -> Option<u64> {
        self.weights_bytes
    }

    /// Batched KV cache bytes (memory-breakdown API) - None until
    /// enable_batch (the serial path's KV lives inside per-layer planes the
    /// breakdown doesn't itemize yet).
    pub fn kv_mem_bytes(&self) -> Option<u64> {
        let b = self.batch.as_ref()?;
        Some(
            b.k_cache
                .iter()
                .chain(&b.v_cache)
                .map(|c| c.len() as u64)
                .sum(),
        )
    }

    /// Allocate the batched-decode state (per-slot KV caches + batched scratch)
    /// for up to `max_batch` concurrent sequences. Call once before forward_batch.
    pub fn enable_batch(&mut self, max_batch: usize) -> Result<(), GpuModelError> {
        self.pipe_abort(); // pipe events/rings point into the old BatchState
        let kv_dim = self.n_kv_heads * self.head_dim;
        let q_dim = self.n_heads * self.head_dim;
        let (embd, ff) = (self.embd, self.ff_exp);
        // KV caches are per-SLOT (max_batch rows); the transient row scratch must
        // also hold a prefill chunk, so it's sized to row_cap = max(max_batch,
        // PREFILL_CHUNK). d_logits stays max_batch (decode fans out to max_batch
        // rows; prefill's final lm_head runs on a single row).
        let row_cap = max_batch.max(prefill_chunk());
        let max_blocks = moe_max_blocks(row_cap, self.n_active, self.n_experts);
        let moe_entries_cap = max_blocks * MOE_TILE_BM;
        // a re-enable drops the KV caches half-done chunked prefills point at
        self.chunked.clear();
        // KV caches are raw bytes: [max_batch, max_ctx, kv_dim] elements × dtype bytes
        let kv_bytes = self.kv_dtype.bytes();
        // Paged KV (G3, default on): K/V become block pools + static
        // block tables. Full-attn layers use an identity map (same VRAM); SWA
        // layers cap at a WindowRing of `swa_ring` blocks - the ~57× SWA saving.
        let bps = self.max_ctx.div_ceil(16); // blocks_per_slot (logical stride)
        // WindowRing size: a chunked-prefill tick appends all of a chunk's
        // positions before any attention read, so the ring must survive one
        // chunk's window-union = row_cap rows + the window (not just the
        // window). (row_cap + window)/16 + 1 blocks, capped at bps (no point
        // exceeding the dense reservation). The SWA KV win therefore scales
        // with how small prefill_chunk is vs max_ctx: at chunk==max_ctx it is
        // bps (identity, no win); at a small chunk the ring is a fraction.
        let swa_ring = if self.swa_window > 0 {
            ((row_cap + self.swa_window).div_ceil(16) + 1).min(bps)
        } else {
            bps
        };
        // G4a budget pool: the full-attn KV as a shared free-list of N blocks.
        // Explicit PADDOCK_KV_POOL_BLOCKS=N wins; else AUTO-SIZE by default
        // (qwen35's scheme, ported here): the P5c zero-copy
        // radix prefix cache only exists in pool mode, so the env-pin-only
        // default served every gpt-oss prompt with zero prefix reuse -
        // cached_tokens was always 0 on the server. Prefix-heavy agentic
        // serving is the tier-1 workload; the pool is capped at the
        // dense-equivalent block count, so KV VRAM matches the old identity
        // mode exactly. The cost: pool mode drops the depth-2 decode pipe
        // (it advances positions on device, but the host must re-upload the
        // grown block table each tick - see supports_decode_pipe). Spec verify
        // grows the pool before its baked table read (P3) and keeps working.
        // PADDOCK_DENSE_KV=1 suppresses the auto-pool (plain-paged identity
        // mode, the old default) for A/B against exactly that trade.
        let paged_capable = self.exec.has_gpt_oss_paged_append()
            && self.exec.has_paged_kv()
            && self.exec.has_attn_prefill_f16_paged()
            && self.exec.has_attn_partial_batch_paged()
            && self.max_ctx.is_multiple_of(16);
        let n_swa = self.layers.iter().filter(|l| l.is_swa).count();
        let n_full = self.n_layers - n_swa;
        // SWA-window checkpoint size (P5c) - hoisted above the pool decision
        // because the auto-sizer must charge the checkpoint pool up front.
        let swa_win_blocks = self.swa_window.div_ceil(16);
        let swa_ckpt_bytes = n_swa * 2 * swa_win_blocks * 16 * kv_dim * kv_bytes;
        let per_block_bytes = 16 * kv_dim * kv_bytes * 2 * n_full;
        let explicit_pool_pin = paddock_models::dev_var!("PADDOCK_KV_POOL_BLOCKS")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .filter(|&n| n > 0);
        // Paged KV is the DEFAULT for gpt-oss (validated P1-P3): the block-table
        // KV + SWA WindowRing is the serving path. PADDOCK_NO_PAGED_KV=1 pins the
        // old dense per-slot cache (escape hatch / A-B). An explicit pool pin
        // still forces the paged lane.
        let want_paged = paddock_models::dev_var_os!("PADDOCK_NO_PAGED_KV").is_none()
            && paddock_models::dev_var_os!("PADDOCK_DENSE_KV").is_none();
        // Honest refusal: silently landing in dense
        // mode when the operator did not opt out used to swallow prefix reuse +
        // the paged serving lanes with no signal. paged_capable false here means
        // the PACK lacks the paged kernels (max_ctx is rounded to the page grid
        // at serve entry).
        if want_paged && !paged_capable {
            return Err(GpuModelError::Config(format!(
                "kernel pack has no paged-KV support for gpt-oss (paged kernels \
                 missing, or max_ctx {} not a multiple of 16) - update the pack; \
                 PADDOCK_NO_PAGED_KV=1 is the explicit dense A/B escape hatch \
                 (no prefix cache)",
                self.max_ctx
            )));
        }
        let paged = (want_paged || explicit_pool_pin.is_some()) && paged_capable;
        // One arbiter sizes the KV store: crate::kv_plan. This family carried
        // qwen35's twin of the defect - an auto-sized pool that read
        // vram_headroom() sitting beside per-slot reservations that did not, with
        // `max_batch <= 1` routing to the second. At max_batch 1 the full-attn
        // layers took `max_batch × bps` blocks (the whole window) with no
        // reference to the budget, and lost prefix reuse on the way past (the
        // half the auto-pool default did not reach).
        self.exec.trim_mem_pool(); // pool-held frees must not read as used
        // Budget-aware headroom, not raw device free: under a configured
        // vram_budget the KV must size inside our slice of the card. A MISSING
        // reading is an error, not permission.
        let grant = self.exec.vram_headroom().ok_or_else(|| {
            GpuModelError::Config(
                "gpt-oss: the driver gave no free-VRAM reading, so the KV cache \
                 cannot be sized inside this server's budget - refusing rather \
                 than allocating blind"
                    .into(),
            )
        })?;
        let rc = row_cap as u64;
        // In dense mode the SWA layers hold a full per-slot context rather than a
        // ring, so the per-slot charge follows the mode we actually landed in.
        let swa_blocks_per_slot = if paged { swa_ring } else { bps };
        let demand = kv_plan::Demand {
            family: "gpt-oss",
            max_ctx: self.max_ctx,
            slots: max_batch,
            blocks_per_slot: bps,
            block_bytes: per_block_bytes as u64,
            // this slot's SWA ring (K+V) and its two block-table rows
            per_slot_bytes: (n_swa * 2 * swa_blocks_per_slot * 16 * kv_dim * kv_bytes) as u64
                + 2 * bps as u64 * 4,
            floor_blocks_per_slot: 128,
            reserves: vec![
                kv_plan::Reserve::new(
                    "SWA-window checkpoints",
                    (SWA_CKPT_SLOTS * swa_ckpt_bytes) as u64,
                ),
                kv_plan::Reserve::new(
                    "logits",
                    (max_batch.max(SPEC_BATCH_MAX_ROWS) * self.vocab) as u64 * 4,
                ),
                // d_x/d_xn/d_proj + d_q/d_qkv/d_attn + d_kv + router + gate_up (f32)
                kv_plan::Reserve::new(
                    "row scratch",
                    rc * (3 * embd + 3 * q_dim + 3 * kv_dim + self.n_experts + self.n_active * ff)
                        as u64
                        * 4,
                ),
                // d_moe_xq (i8) + the /32 scale streams
                kv_plan::Reserve::new(
                    "MoE staging",
                    (moe_entries_cap * ff) as u64 + (moe_entries_cap * ff / 32) as u64 * 5,
                ),
                // prefill/graph-capture scratch + allocator headroom (the qwen35
                // sizer's measured margin on this card class)
                kv_plan::Reserve::new(
                    "graph/prefill scratch",
                    crate::kv_plan::graph_scratch_reserve_bytes(),
                ),
            ],
            // qwen35's Issue-2 guard: a pool that cannot back --max-ctx ×
            // --max-batch refuses loudly rather than under-sizing into unbounded
            // TTFT queues. Dense identity addressing also needs exactly the full
            // ceiling, so oversubscription is only offered on the paged lane.
            when_short: if paged
                && paddock_models::dev_var_os!("PADDOCK_KV_OVERSUBSCRIBE").is_some()
            {
                kv_plan::WhenShort::Narrow
            } else {
                kv_plan::WhenShort::Refuse
            },
            // `reserves` above is hand-enumerated, so cap the damage of an
            // omission at 40% of the grant.
            hedge_fraction: Some(0.4),
            ..Default::default()
        };
        let plan = demand
            .plan(grant)
            .map_err(|e| GpuModelError::Config(e.message))?;
        plan.report(&demand, grant);
        // An explicit pin is a development instrument: it overrides the plan
        // outright, including past the budget, and says so.
        let pool_block_count = match explicit_pool_pin {
            Some(n) => {
                tracing::warn!(
                    pinned = n,
                    planned = plan.pool_blocks,
                    "PADDOCK_KV_POOL_BLOCKS overrides the KV plan - not budget-checked"
                );
                n as usize
            }
            None => plan.pool_blocks,
        };
        // Pool mode is now the whole paged lane. The plain-paged identity map it
        // replaces was only ever reachable when the auto-sizer bailed, and that
        // bail is what was: it kept `max_batch × bps` blocks per
        // full-attn layer off-budget AND served every prompt with zero prefix
        // reuse. Losing the depth-2 decode pipe at B=1 (see supports_decode_pipe)
        // is the cost; prefix-heavy agentic serving is the tier-1 workload.
        let pool_n: Option<u32> = if paged {
            Some(pool_block_count as u32)
        } else {
            None
        };
        let mut k_cache = Vec::with_capacity(self.n_layers);
        let mut v_cache = Vec::with_capacity(self.n_layers);
        for li in 0..self.n_layers {
            // pool = [n_blocks, 16, kv_dim]. Block counts by mode:
            //   dense:                 max_batch * max_ctx (bps*16 == max_ctx)
            //   paged full-attn (G3):  max_batch * bps  (per-slot identity)
            //   paged SWA (G3):        max_batch * swa_ring (per-slot ring)
            //   POOL full-attn (G4a):  pool_n  (shared free-list, not per-slot)
            let elems = if paged {
                let blocks = if self.layers[li].is_swa {
                    max_batch * swa_ring
                } else {
                    pool_block_count // shared pool: N blocks total for the full-attn layer
                };
                blocks * 16 * kv_dim * kv_bytes
            } else {
                // Dense per-slot cache. Not a second sizing philosophy any more:
                // the plan reached the addressable ceiling, so this is
                // max_batch × max_ctx (rounded up to the page grid), and it is
                // refusable like any other plan.
                pool_block_count * 16 * kv_dim * kv_bytes
            };
            k_cache.push(self.exec.alloc_u8(elems)?);
            v_cache.push(self.exec.alloc_u8(elems)?);
        }
        // Two STATIC block tables [max_batch * bps] (u32), precomputed once:
        //   full-attn identity: bt[slot*bps + j] = slot*bps + j
        //   SWA WindowRing:      bt[slot*bps + j] = slot*swa_ring + j%swa_ring
        // Uploaded outside any graph; kernels read by pointer at replay.
        // In POOL mode the full-attn table starts empty (all zeros) and is grown
        // per step from `tables`; in plain-paged mode it is the static identity
        // map. The SWA ring table is static in both paged modes.
        let (d_bt_full, d_bt_swa) = if paged {
            let mut bt_full = vec![0u32; max_batch * bps];
            let mut bt_swa = vec![0u32; max_batch * bps];
            for s in 0..max_batch {
                for j in 0..bps {
                    if pool_n.is_none() {
                        bt_full[s * bps + j] = (s * bps + j) as u32;
                    }
                    bt_swa[s * bps + j] = (s * swa_ring + (j % swa_ring)) as u32;
                }
            }
            (
                Some(self.exec.to_device_u32(&bt_full)?),
                Some(self.exec.to_device_u32(&bt_swa)?),
            )
        } else {
            (None, None)
        };
        // P5c prefix reuse (pool mode only, unless pinned off): a radix over the
        // full-attn pool blocks + a device pool of SWA-window checkpoints. The
        // radix shares `pool` (cached blocks are refcounted); the checkpoints let
        // a shared-prefix resume restore the SWA ring exactly.
        let prefix_on = pool_n.is_some()
            && self.swa_window > 0
            && paddock_models::dev_var_os!("PADDOCK_NO_PREFIX_CACHE").is_none();
        let (paged_prefix_radix, d_swa_ckpt) = if prefix_on {
            let mut pr = PagedRadix::new();
            pr.set_state_capacity(SWA_CKPT_SLOTS as u32);
            let ckpt = self.exec.alloc_u8(SWA_CKPT_SLOTS * swa_ckpt_bytes)?;
            (Some(pr), Some(ckpt))
        } else {
            (None, None)
        };
        self.batch = Some(BatchState {
            max_batch,
            row_cap,
            k_cache,
            v_cache,
            d_x: self.exec.alloc(row_cap * embd)?,
            d_xn: self.exec.alloc(row_cap * embd)?,
            d_q: self.exec.alloc(row_cap * q_dim)?,
            d_qkv: self.exec.alloc(row_cap * (q_dim + 2 * kv_dim))?,
            d_kv: self.exec.alloc(row_cap * kv_dim)?,
            d_attn: self.exec.alloc(row_cap * q_dim)?,
            // n_heads*batch*n_splits ≤ 2*fill_blocks*MAX_ATTN_SPLITS: the
            // q-head split path stays ≤ fill*MAX; the GQA-fused path budgets
            // group*4*fill block-equivalents (attn_splits), hence the 2x
            d_attn_o: self.exec.alloc(
                2 * attn_fill_blocks(self.exec.sm_count()) * MAX_ATTN_SPLITS * self.head_dim,
            )?,
            d_attn_ml: self
                .exec
                .alloc(2 * attn_fill_blocks(self.exec.sm_count()) * MAX_ATTN_SPLITS * 2)?,
            d_proj: self.exec.alloc(row_cap * embd)?,
            d_router: self.exec.alloc(row_cap * self.n_experts)?,
            d_gate_up: self.exec.alloc(row_cap * self.n_active * ff)?,
            // sized for the LARGER of the serving batch and a spec verify
            // pass - G3's verify chunks wrote r x vocab rows here, which
            // overflowed a max_batch=1 allocation silently (pool slack hid
            // it); spec-batch rounds make the r > max_batch case routine
            d_logits: self
                .exec
                .alloc(max_batch.max(SPEC_BATCH_MAX_ROWS) * self.vocab)?,
            d_topk_idx: self.exec.alloc_u32(row_cap * self.n_active)?,
            d_topk_w: self.exec.alloc(row_cap * self.n_active)?,
            // sized for the BM32 sorted layout (each expert padded to a 32-tile)
            d_sorted_row: self.exec.alloc_u32(moe_entries_cap)?,
            d_sorted_slot: self.exec.alloc_u32(moe_entries_cap)?,
            d_block_expert: self.exec.alloc_u32(max_blocks)?,
            d_b1_xq: self
                .exec
                .alloc_i8((self.n_active * ff).max(q_dim).max(embd))?,
            d_b1_xs: self
                .exec
                .alloc((self.n_active * ff).max(q_dim).max(embd) / 32)?,
            d_p_xq: self.exec.alloc_i8(row_cap * q_dim.max(embd))?,
            d_p_xs: self.exec.alloc(row_cap * q_dim.max(embd) / 32)?,
            d_p_yq: self
                .exec
                .alloc_u8(q_dim.max(embd).div_ceil(128) * row_cap.next_multiple_of(128) * 144)?,
            // serves the mma_ks z-split planes (b 9..=64, needs 256 tiles),
            // the legacy stream-k fixup, AND the pipe_sk tail partials
            // (b > 64: tail(<=sm/2) x splits(<=4) tiles - qwen3's sizing)
            d_p_skfix: self
                .exec
                .alloc(((self.exec.sm_count() / 2 + 1) * 4).max(256) * 128 * 128 + 256)?,
            d_moe_xq: self.exec.alloc_i8(moe_entries_cap * ff)?,
            d_moe_xs: self.exec.alloc(moe_entries_cap * ff / 32)?,
            d_p_xs8: self.exec.alloc_u8(row_cap * q_dim.max(embd) / 32)?,
            d_moe_xs8: self.exec.alloc_u8(moe_entries_cap * ff / 32)?,

            d_g_token: self.exec.alloc_u32(1)?,
            d_g_mrope: self.exec.alloc_u32(4)?,
            d_g_out: self.exec.alloc_u32(GEN_CHUNK)?,
            d_g_step: self.exec.alloc_u32(1)?,
            d_g_pmax: self.exec.alloc(512)?,
            d_g_pidx: self.exec.alloc_u32(512)?,
            gen_graph: None,
            pf_graphs: std::collections::HashMap::new(),
            step_graphs: std::collections::HashMap::new(),
            spec_graphs: std::collections::HashMap::new(),
            d_spec_pick: self.exec.alloc_u32(SPEC_BATCH_MAX_ROWS)?,
            d_samp_par: self.exec.alloc_u32(max_batch * 4)?,
            d_samp_out: self.exec.alloc_u32(max_batch)?,
            d_samp_tpar: self.exec.alloc_u32(max_batch * 4)?,
            d_pipe_tpar: self.exec.alloc_u32(2 * max_batch * 4)?,
            d_pipe_par: self.exec.alloc_u32(2 * max_batch * 4)?,
            d_pipe_out: self.exec.alloc_u32(2 * max_batch)?,
            paged,
            blocks_per_slot: bps,
            swa_ring_blocks: swa_ring,
            d_bt: [d_bt_full, d_bt_swa],
            pool: pool_n.map(KvPool::with_blocks),
            tables: if pool_n.is_some() {
                (0..max_batch).map(|_| BlockTable::new()).collect()
            } else {
                Vec::new()
            },
            block_table_host: if pool_n.is_some() {
                vec![0u32; max_batch * bps]
            } else {
                Vec::new()
            },
            paged_prefix: paged_prefix_radix,
            d_swa_ckpt,
            swa_ckpt_bytes,
        });
        if paged {
            match pool_n {
                Some(n) => tracing::info!(
                    "gpt-oss paged KV POOL active: {n_full} full-attn @ shared pool of {n} blocks \
                     ({:.2} GiB/pair), {n_swa} SWA @ WindowRing({}) = {swa_ring} blocks/slot",
                    (n as f64 * 16.0 * kv_dim as f64 * kv_bytes as f64)
                        / (1024.0 * 1024.0 * 1024.0),
                    self.swa_window,
                ),
                None => tracing::info!(
                    "gpt-oss paged KV active: {n_swa} SWA layers @ WindowRing({}) = {swa_ring} \
                     blocks/slot, {n_full} full-attn @ identity {bps} blocks/slot",
                    self.swa_window,
                ),
            }
        }
        self.last_reused = vec![0; max_batch];
        Ok(())
    }

    /// Tokens the last prefill of `slot` served from the prefix cache (taken:
    /// resets to 0). Usage-reporting hook for the engine.
    pub fn take_prefill_reused(&mut self, slot: usize) -> usize {
        self.last_reused.get_mut(slot).map_or(0, std::mem::take)
    }

    /// True when a G4a budget pool is active (the full-attn KV is a shared
    /// free-list). Gates the on-device paths that can't host-grow: the decode
    /// pipe and spec verify advance positions on device, so the host can't
    /// re-upload the block table between their ticks - pool mode forces the
    /// per-tick host-driven decode (`forward_batch_sampled`) instead.
    pub fn pool_active(&self) -> bool {
        self.batch.as_ref().is_some_and(|b| b.pool.is_some())
    }

    /// G4a: grow the full-attn pool tables so every `(slots[i], positions[i])`
    /// pair this pass will touch is backed by a physical block, then re-upload
    /// the changed device table (`d_bt[0]`) once - outside any captured graph,
    /// into the same buffer the graph baked by pointer, so replay reads the
    /// fresh mapping. No-op unless a budget pool is active. `PoolExhausted` on a
    /// dry pool (G4a errors the batch cleanly; preemption is G4b). The SWA ring
    /// (`d_bt[1]`) is static and never grows here.
    fn ensure_pool_rows(&mut self, slots: &[u32], positions: &[u32]) -> Result<(), GpuModelError> {
        debug_assert_eq!(slots.len(), positions.len());
        let exec = self.exec.clone();
        let bs = self.batch.as_mut().ok_or(GpuModelError::BatchDisabled)?;
        if bs.pool.is_none() {
            return Ok(());
        }
        let bps = bs.blocks_per_slot;
        let mut grew = false;
        for (i, &s) in slots.iter().enumerate() {
            let s = s as usize;
            let pos = positions[i] as usize;
            let before = bs.tables[s].blocks().len();
            {
                let pool = bs.pool.as_mut().expect("pool checked above");
                bs.tables[s]
                    .ensure(pos, pool)
                    .map_err(|_| GpuModelError::PoolExhausted)?;
            }
            let now = bs.tables[s].blocks().len();
            if now > before {
                grew = true;
                let base = s * bps;
                for j in before..now {
                    bs.block_table_host[base + j] = bs.tables[s].blocks()[j];
                }
            }
        }
        if grew {
            let dst = bs.d_bt[0].as_mut().expect("pool implies a full-attn table");
            exec.stream
                .memcpy_htod(&bs.block_table_host, dst)
                .map_err(|e| GpuError::Driver(e.to_string()))?;
            if paddock_models::dev_var_os!("PADDOCK_POOL_STATS").is_some() {
                let pool = bs.pool.as_ref().expect("pool checked above");
                tracing::info!(
                    "pool: {} / {} blocks free",
                    pool.free_blocks(),
                    pool.capacity()
                );
            }
        }
        Ok(())
    }

    /// G4a free-on-reuse: return `slot`'s full-attn pool blocks to the free-list
    /// before a fresh prefill regrows it from zero. No-op outside pool mode.
    fn pool_clear_slot(&mut self, slot: usize) {
        let Some(bs) = self.batch.as_mut() else {
            return;
        };
        let Some(pool) = bs.pool.as_mut() else { return };
        bs.tables[slot].clear(pool);
    }

    /// G4a free-on-completion (P5b): return the full-attn KV blocks of every
    /// slot no longer holding a live sequence to the shared pool, so the memory
    /// is available to new admissions the moment a sequence ends. Idempotent;
    /// no-op unless a budget pool is active. A freed slot's stale device table
    /// entries are never read (an inactive slot is regrown at its next prefill).
    pub fn release_inactive_slots(&mut self, occupied: &[bool]) {
        let Some(bs) = self.batch.as_mut() else {
            return;
        };
        let Some(pool) = bs.pool.as_mut() else { return };
        let mut freed = 0usize;
        for (slot, table) in bs.tables.iter_mut().enumerate() {
            let idle = occupied.get(slot).copied() != Some(true);
            if idle && !table.blocks().is_empty() {
                freed += table.blocks().len();
                table.clear(pool);
            }
        }
        if freed > 0 && paddock_models::dev_var_os!("PADDOCK_POOL_STATS").is_some() {
            tracing::info!(
                "pool: freed {freed} blocks on completion  ({}/{} free)",
                pool.free_blocks(),
                pool.capacity()
            );
        }
    }

    /// Test/telemetry hook: number of full-attn pool blocks currently mapped by
    /// `slot`'s block table, or `None` outside pool mode. Used to assert the P3
    /// spec-verify span growth (`ensure_pool_rows`) actually extended the table
    /// to cover the draft positions - a white-box check that sidesteps the
    /// sm120 verify-MoE logit noise.
    pub fn pool_slot_blocks(&self, slot: usize) -> Option<usize> {
        let bs = self.batch.as_ref()?;
        bs.pool.as_ref()?;
        Some(bs.tables.get(slot)?.blocks().len())
    }

    /// Free blocks in the budget pool, or `None` when not in pool mode. Drives
    /// the scheduler's P5b watermark admission (stop admitting when the pool is
    /// nearly full; free-on-completion reopens it).
    pub fn pool_free_blocks(&self) -> Option<usize> {
        let bs = self.batch.as_ref()?;
        let pool = bs.pool.as_ref()?;
        // free + prefix-reclaimable: the radix is a CACHE (slot growth evicts
        // LRU pages on demand), so admission must not count retained pages as
        // spoken for - the raw count watermark-starves retention-heavy
        // workloads - seen live on qwen35, where a salted concurrent run went
        // from a healthy warmup to a ~4x slowdown with multi-minute TTFT tails
        // once every completion had parked its pages in the cache.
        let evictable = bs
            .paged_prefix
            .as_ref()
            .map_or(0, |pr| pr.evictable_blocks(pool));
        Some(pool.free_blocks() + evictable)
    }

    /// Gather `tokens` into the first rows of `d_x` (row i = tok_embd[tokens[i]]):
    /// one id upload + one gather kernel, not a dtod memcpy per token (256 tiny
    /// copies cost ~1.5 ms per prefill chunk).
    fn embed_gather(&mut self, tokens: &[u32]) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let embd = self.embd;
        let bs = self.batch.as_mut().ok_or(GpuModelError::BatchDisabled)?;
        let d_tok = exec
            .stream
            .clone_htod(&tokens.to_vec())
            .map_err(|e| GpuError::Driver(e.to_string()))?;
        exec.embed_gather_batch_q8(&self.tok_embd, &d_tok, &mut bs.d_x, embd, tokens.len())?;
        Ok(())
    }

    /// Run the full transformer stack (all layers) in place on `d_x` for `b` rows
    /// at `d_pos`. `slots` maps rows to KV slots: `None` = row index (decode; one
    /// sequence per row), `Some([S; b])` = prefill (all rows are one sequence,
    /// writing/attending slot S causally via ascending positions). Shared by
    /// forward_batch (decode) and forward_prefill.
    fn run_layers(
        &mut self,
        b: usize,
        d_pos: &CudaSlice<u32>,
        slots: Option<&CudaSlice<u32>>,
        max_pos: usize,
        uniform_slot: bool,
        // host-known row groups of a mixed-slot pass (start, count,
        // single_slot): attention dispatches per group so big SINGLE-SLOT
        // tails ride the f16 WMMA kernel - it reads one slot per launch, so
        // a multi-slot group (the mixed tick's decode rows) must never take
        // it regardless of size (rows 1.. would read slot 0's KV; at C>24
        // that garbage sampled instant stop tokens and killed half the
        // streams). None = the whole pass is one range (decode or
        // uniform-slot prefill).
        groups: Option<&[(usize, usize, bool)]>,
    ) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let (embd, head_dim) = (self.embd, self.head_dim);
        let kv_dim = self.n_kv_heads * head_dim;
        let ff = self.ff_exp;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let (n_heads, n_kv_heads, n_experts, n_active) =
            (self.n_heads, self.n_kv_heads, self.n_experts, self.n_active);
        let (max_ctx, rms_eps, yarn, swa) = (
            self.max_ctx,
            self.rms_eps,
            self.yarn_params,
            self.swa_window,
        );
        let kv_dtype = self.kv_dtype;
        // B=1: llama-mmvq-class dp4a GEMVs (quantized activations) for the
        // dense projections and the fused MoE - the batch-shaped kernels leave
        // most of the GPU idle at one row. q8_1/dp4a numeric class (llama.cpp's
        // own). batch>1: int8 tensor-core dense GEMMs (mma <= 64 rows, mmq
        // stream-k above - llama's own prefill numeric class, qwen P6e reuse).
        // The PADDOCK_NO_MMQ / PADDOCK_NO_DP4A_B1 exact-f32 pins were
        // collapsed: their MoE half fed the fused g||u ILV plane to the
        // plain-layout f32 kernels - garbage plus out-of-bounds reads from the
        // 16-byte up_exps dummy - on every pinned run. The
        // llama.cpp same-weights greedy gates are the correctness bar.
        let b1_fast = b == 1;
        let mm_fast = b > 1;
        // sm_120a block-scale MoE (mxFP4 x FP8 tensor core, hardware ue8m0
        // scaling): capability-gated on the pack entry + CC 12.x, pinned off
        // by PADDOCK_NO_MOE_BS (fp8-activation numeric class). b > 4 keeps
        // the exactness-gated sizes in the int8 classes (the batch-vs-single
        // gate runs B=4; spec verify pins int8 via moe_bs_pin regardless of
        // rows) while serving batches and prefill chunks ride the tensor
        // cores.
        let moe_bs = b > 4
            && !self.moe_bs_pin
            && MOE_BS.load(std::sync::atomic::Ordering::Relaxed)
            && exec
                .kernels()
                .map(|k| k.mxfp4_moe_gate_up_bs.is_some())
                .unwrap_or(false)
            && exec.compute_capability().0 == 12
            && paddock_models::dev_var_os!("PADDOCK_NO_MOE_BS").is_none();
        // real prefill spans on one slot go to the f16 WMMA tiled attention
        // (P6f/P6i class, hd=64 instantiation) - the decode kernel walks keys
        // sequentially per row and dominated the pp512 budget. Multi-slot
        // batched-tail passes keep decode attention (the kernel reads slots[0]).
        //
        // fp8-e4m3 KV used to be excluded here entirely, which silently
        // dropped every gpt-oss prefill to the scalar split/combine tile the
        // instant --kv-cache-dtype fp8_e4m3 was requested (correct output,
        // no tensor cores, gpt_oss.rs never grew the e4m3 arm this kernel's
        // granite/laguna/qwen35 siblings did). Fixed since: the v4
        // staged-HMMA tile (attn/prefill.cuh, `pd_attn_prefill_f16_paged`)
        // grew a hd64/G8 fp8 arm SPECIFICALLY for this shape, PAGED only -
        // the non-paged `pd_attn_prefill_f16` sibling still hard-rejects
        // fp8-e4m3 at the C level, so this stays gated on `paged` too.
        let paged = self.batch.as_ref().is_some_and(|b| b.paged);
        let pf16_attn = mm_fast
            && uniform_slot
            && slots.is_some()
            && b > 24
            && head_dim == 64
            && max_ctx % 64 == 0
            && (matches!(kv_dtype, KvDtype::Fp16)
                || (paged && matches!(kv_dtype, KvDtype::Fp8E4m3) && n_heads == 8 * n_kv_heads));
        let q_dim = n_heads * head_dim;
        let bs = self.batch.as_mut().ok_or(GpuModelError::BatchDisabled)?;
        // cross-layer deferral: layer N's MoE slot-combine folds into layer
        // N+1's norm+quantize (the last layer combines standalone)
        let mut pending_combine = false;
        for li in 0..self.n_layers {
            let layer = &self.layers[li];
            // ---- attention
            let attn_fused_q = mm_fast && b <= 64 && exec.has_rmsnorm_quant_q8();
            // P3: at b > 64 (mixed/prefill ticks) fold the attention-input
            // rmsnorm straight into the MMQ quantize (proj=None, xn=None on the
            // add_rmsnorm_quant_mmq kernel), killing rmsnorm_batch's f32 d_xn
            // round trip - dead the instant the qkv GEMM eats the int8. Opt-in
            // (PADDOCK_ATTN_FUSE_MMQ) while we A/B the mixed-tick thesis.
            let attn_fused_mmq = mm_fast
                && b > 64
                && exec.has_add_rmsnorm_quant_mmq()
                && paddock_models::dev_var_os!("PADDOCK_ATTN_FUSE_MMQ").is_some();
            if !attn_fused_q && !attn_fused_mmq {
                exec.rmsnorm_batch(
                    &bs.d_x,
                    &layer.attn_norm.buf,
                    &mut bs.d_xn,
                    embd,
                    rms_eps,
                    b,
                )?;
            }
            debug_assert!(
                !(pending_combine && !attn_fused_q),
                "deferral gated on the fused arm"
            );
            if b1_fast {
                exec.quantize_q8(&bs.d_xn, &mut bs.d_b1_xq, &mut bs.d_b1_xs, embd)?;
                exec.q8_0_gemv_dp4a_nc_b(
                    &layer.wq_r,
                    Some(&layer.bq.buf),
                    &bs.d_b1_xq,
                    &bs.d_b1_xs,
                    &mut bs.d_q,
                    1,
                )?;
                exec.rope_yarn_batch(&mut bs.d_q, d_pos, n_heads, head_dim, yarn, b)?;
                exec.q8_0_gemv_dp4a_nc_b(
                    &layer.wk_r,
                    Some(&layer.bk.buf),
                    &bs.d_b1_xq,
                    &bs.d_b1_xs,
                    &mut bs.d_kv,
                    1,
                )?;
                exec.rope_yarn_batch(&mut bs.d_kv, d_pos, n_kv_heads, head_dim, yarn, b)?;
                if bs.paged {
                    let bps = bs.blocks_per_slot;
                    exec.kv_append_batch_paged(
                        &bs.d_kv,
                        &mut bs.k_cache[li],
                        d_pos,
                        slots,
                        bs.d_bt[layer.is_swa as usize]
                            .as_ref()
                            .expect("paged block tables built"),
                        bps,
                        kv_dim,
                        b,
                        kv_dtype,
                    )?;
                } else {
                    exec.kv_append_batch(
                        &bs.d_kv,
                        &mut bs.k_cache[li],
                        d_pos,
                        slots,
                        kv_dim,
                        max_ctx,
                        b,
                        kv_dtype,
                    )?;
                }
                exec.q8_0_gemv_dp4a_nc_b(
                    &layer.wv_r,
                    Some(&layer.bv.buf),
                    &bs.d_b1_xq,
                    &bs.d_b1_xs,
                    &mut bs.d_kv,
                    1,
                )?;
                if bs.paged {
                    let bps = bs.blocks_per_slot;
                    exec.kv_append_batch_paged(
                        &bs.d_kv,
                        &mut bs.v_cache[li],
                        d_pos,
                        slots,
                        bs.d_bt[layer.is_swa as usize]
                            .as_ref()
                            .expect("paged block tables built"),
                        bps,
                        kv_dim,
                        b,
                        kv_dtype,
                    )?;
                } else {
                    exec.kv_append_batch(
                        &bs.d_kv,
                        &mut bs.v_cache[li],
                        d_pos,
                        slots,
                        kv_dim,
                        max_ctx,
                        b,
                        kv_dtype,
                    )?;
                }
            } else {
                // fused [q|k|v]: one wide GEMM (quantize the normed rows once
                // - P6j dedup) + one rope/append launch. Values are the same
                // per element as the split path; only the mma_ks z-slicing
                // regroups (same numeric class, token-level gates arbitrate).
                // At b <= 64 the norm+quantize run as one kernel (glue
                // fusion - kills a launch + the f32 round trip per layer).
                if attn_fused_q {
                    if pending_combine {
                        // layer N-1's MoE slot-combine rides this pass (fixed
                        // slot order - bit-identical to the standalone fold)
                        pending_combine = false;
                        exec.moe_combine_rmsnorm_quant_q8_batch(
                            &mut bs.d_x,
                            &bs.d_gate_up,
                            &layer.attn_norm.buf,
                            &mut bs.d_xn,
                            &mut bs.d_p_xq,
                            &mut bs.d_p_xs,
                            embd,
                            n_active,
                            rms_eps,
                            b,
                        )?;
                    } else {
                        exec.rmsnorm_quant_q8_batch(
                            &bs.d_x,
                            &layer.attn_norm.buf,
                            &mut bs.d_xn,
                            &mut bs.d_p_xq,
                            &mut bs.d_p_xs,
                            embd,
                            rms_eps,
                            b,
                        )?;
                    }
                } else if attn_fused_mmq {
                    // rmsnorm + mmq-quant in one pass (no residual add here, no
                    // f32 d_xn write) -> straight into the qkv GEMM's mmq input.
                    exec.add_rmsnorm_quant_mmq(
                        &mut bs.d_x,
                        None,
                        false,
                        &layer.attn_norm.buf,
                        None,
                        &mut bs.d_p_yq,
                        embd,
                        b,
                        rms_eps,
                    )?;
                } else {
                    mm_quant(
                        &exec,
                        &bs.d_xn,
                        &mut bs.d_p_xq,
                        &mut bs.d_p_xs,
                        &mut bs.d_p_yq,
                        embd,
                        b,
                    )?;
                }
                if b <= 64 && exec.has_ks_qkv_rope() {
                    // wqkv all-in-one: GEMM partials + fused combine/rope/
                    // append - no d_qkv round trip, two launches fewer
                    if bs.paged {
                        let bps = bs.blocks_per_slot;
                        exec.q8_0_gemm_mma_ks_qkv_rope_paged(
                            &layer.wqkv_r,
                            &layer.bqkv,
                            &bs.d_p_xq,
                            &bs.d_p_xs,
                            &mut bs.d_p_skfix,
                            &mut bs.d_q,
                            &mut bs.k_cache[li],
                            &mut bs.v_cache[li],
                            d_pos,
                            slots,
                            embd,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            yarn,
                            b,
                            bs.d_bt[layer.is_swa as usize]
                                .as_ref()
                                .expect("paged block tables built"),
                            bps,
                            kv_dtype,
                        )?;
                    } else {
                        exec.q8_0_gemm_mma_ks_qkv_rope(
                            &layer.wqkv_r,
                            &layer.bqkv,
                            &bs.d_p_xq,
                            &bs.d_p_xs,
                            &mut bs.d_p_skfix,
                            &mut bs.d_q,
                            &mut bs.k_cache[li],
                            &mut bs.v_cache[li],
                            d_pos,
                            slots,
                            embd,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            max_ctx,
                            yarn,
                            b,
                            kv_dtype,
                        )?;
                    }
                } else {
                    mm_pre(
                        &exec,
                        &layer.wqkv_r,
                        &layer.bqkv,
                        &bs.d_p_xq,
                        &bs.d_p_xs,
                        &bs.d_p_yq,
                        &mut bs.d_p_skfix,
                        &mut bs.d_qkv,
                        b,
                    )?;
                    if bs.paged {
                        let bps = bs.blocks_per_slot;
                        exec.qkv_rope_append_batch_paged(
                            &bs.d_qkv,
                            &mut bs.d_q,
                            &mut bs.k_cache[li],
                            &mut bs.v_cache[li],
                            d_pos,
                            slots,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            yarn,
                            b,
                            bs.d_bt[layer.is_swa as usize]
                                .as_ref()
                                .expect("paged block tables built"),
                            bps,
                            kv_dtype,
                        )?;
                    } else {
                        exec.qkv_rope_append_batch(
                            &bs.d_qkv,
                            &mut bs.d_q,
                            &mut bs.k_cache[li],
                            &mut bs.v_cache[li],
                            d_pos,
                            slots,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            max_ctx,
                            yarn,
                            b,
                            kv_dtype,
                        )?;
                    }
                }
            }
            let swa_window = if layer.is_swa { swa } else { 0 };
            // FlashDecoding split when the grid underfills (low batch) and context
            // is long; the window caps a SWA layer's effective KV length. n_splits=1
            // collapses to the plain single-kernel path (no combine overhead).
            let eff_n_pos = if swa_window > 0 {
                (max_pos + 1).min(swa_window)
            } else {
                max_pos + 1
            };
            let n_splits = attn_splits(
                n_heads,
                n_kv_heads,
                b,
                eff_n_pos,
                attn_fill_blocks(exec.sm_count()),
            );
            if let (Some(gs), Some(sl)) = (groups, slots) {
                // mixed-slot pass: per-group dispatch so each big SINGLE-SLOT
                // tail rides the f16 WMMA kernel on its slot; small groups
                // (f16 tiles would be mostly dead rows) and multi-slot groups
                // (the WMMA kernel reads one slot per launch) take
                // decode-class.
                for &(start, count, single_slot) in gs {
                    // fp8 arm mirrors `pf16_attn` below: paged
                    // only, because the rows wrappers reuse the same two C
                    // exports - `attn_prefill_f16_rows_paged` lands on
                    // `pd_attn_prefill_f16_paged` (which grew the hd64/G8
                    // e4m3 v4 tile), while the non-paged rows variant lands
                    // on `pd_attn_prefill_f16` (f16-only, hard-rejects fp8).
                    // This groups path is the one live serving actually
                    // takes (mixed ticks); the `pf16_attn` twin below covers
                    // the uniform-slot chunk pass.
                    let f16 = mm_fast
                        && single_slot
                        && count > 24
                        && head_dim == 64
                        && max_ctx % 64 == 0
                        && (matches!(kv_dtype, KvDtype::Fp16)
                            || (bs.paged
                                && matches!(kv_dtype, KvDtype::Fp8E4m3)
                                && n_heads == 8 * n_kv_heads));
                    // the mixed tick's decode group sits at rows 0..count -
                    // exactly the layout the batched FlashDecoding partial +
                    // combine pair operates on, so give those rows the same
                    // GQA-fused split walk the pure-decode tick gets instead
                    // of the unsplit sequential rows-walk
                    if !single_slot && start == 0 && count > 1 {
                        let ns = attn_splits(
                            n_heads,
                            n_kv_heads,
                            count,
                            max_pos + 1,
                            attn_fill_blocks(exec.sm_count()),
                        );
                        if ns > 1 || attn_gqa_fused(n_heads, n_kv_heads, count) {
                            if bs.paged {
                                let bps = bs.blocks_per_slot;
                                exec.attn_partial_batch_paged(
                                    &bs.d_q,
                                    &bs.k_cache[li],
                                    &bs.v_cache[li],
                                    &mut bs.d_attn_o,
                                    &mut bs.d_attn_ml,
                                    d_pos,
                                    Some(sl),
                                    bs.d_bt[layer.is_swa as usize]
                                        .as_ref()
                                        .expect("paged block tables built"),
                                    bps,
                                    n_heads,
                                    n_kv_heads,
                                    head_dim,
                                    kv_dim,
                                    swa_window,
                                    ns,
                                    count,
                                    scale,
                                    kv_dtype,
                                )?;
                            } else {
                                exec.attn_partial_batch(
                                    &bs.d_q,
                                    &bs.k_cache[li],
                                    &bs.v_cache[li],
                                    &mut bs.d_attn_o,
                                    &mut bs.d_attn_ml,
                                    d_pos,
                                    Some(sl),
                                    n_heads,
                                    n_kv_heads,
                                    head_dim,
                                    max_ctx,
                                    kv_dim,
                                    swa_window,
                                    ns,
                                    count,
                                    scale,
                                    kv_dtype,
                                )?;
                            }
                            exec.attn_combine_batch(
                                &bs.d_attn_o,
                                &bs.d_attn_ml,
                                &layer.sinks_dev,
                                &mut bs.d_attn,
                                n_heads,
                                head_dim,
                                ns,
                                count,
                            )?;
                            continue;
                        }
                    }
                    if f16 {
                        if bs.paged {
                            let bps = bs.blocks_per_slot;
                            exec.attn_prefill_f16_rows_paged(
                                &bs.d_q,
                                &bs.k_cache[li],
                                &bs.v_cache[li],
                                &layer.sinks_dev,
                                &mut bs.d_attn,
                                d_pos,
                                None, // causal rows: the bound is the position
                                sl,
                                bs.d_bt[layer.is_swa as usize]
                                    .as_ref()
                                    .expect("paged block tables built"),
                                bps,
                                n_heads,
                                n_kv_heads,
                                head_dim,
                                kv_dim,
                                swa_window,
                                start,
                                count,
                                scale,
                                kv_dtype,
                            )?;
                        } else {
                            exec.attn_prefill_f16_rows(
                                &bs.d_q,
                                &bs.k_cache[li],
                                &bs.v_cache[li],
                                &layer.sinks_dev,
                                &mut bs.d_attn,
                                d_pos,
                                None, // causal rows: the bound is the position
                                sl,
                                n_heads,
                                n_kv_heads,
                                head_dim,
                                max_ctx,
                                kv_dim,
                                swa_window,
                                start,
                                count,
                                scale,
                                kv_dtype,
                            )?;
                        }
                    } else if bs.paged {
                        let bps = bs.blocks_per_slot;
                        exec.attn_decode_batch_rows_paged(
                            &bs.d_q,
                            &bs.k_cache[li],
                            &bs.v_cache[li],
                            &layer.sinks_dev,
                            &mut bs.d_attn,
                            d_pos,
                            Some(sl),
                            bs.d_bt[layer.is_swa as usize]
                                .as_ref()
                                .expect("paged block tables built"),
                            bps,
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            kv_dim,
                            swa_window,
                            start,
                            count,
                            scale,
                            kv_dtype,
                        )?;
                    } else {
                        exec.attn_decode_batch_rows(
                            &bs.d_q,
                            &bs.k_cache[li],
                            &bs.v_cache[li],
                            &layer.sinks_dev,
                            &mut bs.d_attn,
                            d_pos,
                            Some(sl),
                            n_heads,
                            n_kv_heads,
                            head_dim,
                            max_ctx,
                            kv_dim,
                            swa_window,
                            start,
                            count,
                            scale,
                            kv_dtype,
                        )?;
                    }
                }
            } else if pf16_attn {
                let sl = slots.expect("pf16_attn requires slots");
                if bs.paged {
                    let bps = bs.blocks_per_slot;
                    exec.attn_prefill_f16_paged(
                        &bs.d_q,
                        &bs.k_cache[li],
                        &bs.v_cache[li],
                        &layer.sinks_dev,
                        &mut bs.d_attn,
                        d_pos,
                        sl,
                        bs.d_bt[layer.is_swa as usize]
                            .as_ref()
                            .expect("paged block tables built"),
                        bps,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        kv_dim,
                        swa_window,
                        b,
                        scale,
                        kv_dtype,
                    )?;
                } else {
                    exec.attn_prefill_f16(
                        &bs.d_q,
                        &bs.k_cache[li],
                        &bs.v_cache[li],
                        &layer.sinks_dev,
                        &mut bs.d_attn,
                        d_pos,
                        sl,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        max_ctx,
                        kv_dim,
                        swa_window,
                        b,
                        scale,
                        kv_dtype,
                    )?;
                }
            } else if n_splits > 1 || attn_gqa_fused(n_heads, n_kv_heads, b) {
                if bs.paged {
                    let bps = bs.blocks_per_slot;
                    exec.attn_partial_batch_paged(
                        &bs.d_q,
                        &bs.k_cache[li],
                        &bs.v_cache[li],
                        &mut bs.d_attn_o,
                        &mut bs.d_attn_ml,
                        d_pos,
                        slots,
                        bs.d_bt[layer.is_swa as usize]
                            .as_ref()
                            .expect("paged block tables built"),
                        bps,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        kv_dim,
                        swa_window,
                        n_splits,
                        b,
                        scale,
                        kv_dtype,
                    )?;
                } else {
                    exec.attn_partial_batch(
                        &bs.d_q,
                        &bs.k_cache[li],
                        &bs.v_cache[li],
                        &mut bs.d_attn_o,
                        &mut bs.d_attn_ml,
                        d_pos,
                        slots,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        max_ctx,
                        kv_dim,
                        swa_window,
                        n_splits,
                        b,
                        scale,
                        kv_dtype,
                    )?;
                }
                exec.attn_combine_batch(
                    &bs.d_attn_o,
                    &bs.d_attn_ml,
                    &layer.sinks_dev,
                    &mut bs.d_attn,
                    n_heads,
                    head_dim,
                    n_splits,
                    b,
                )?;
            } else if bs.paged {
                let bps = bs.blocks_per_slot;
                exec.attn_decode_batch_paged(
                    &bs.d_q,
                    &bs.k_cache[li],
                    &bs.v_cache[li],
                    &layer.sinks_dev,
                    &mut bs.d_attn,
                    d_pos,
                    slots,
                    bs.d_bt[layer.is_swa as usize]
                        .as_ref()
                        .expect("paged block tables built"),
                    bps,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    kv_dim,
                    swa_window,
                    b,
                    scale,
                    kv_dtype,
                )?;
            } else {
                exec.attn_decode_batch(
                    &bs.d_q,
                    &bs.k_cache[li],
                    &bs.v_cache[li],
                    &layer.sinks_dev,
                    &mut bs.d_attn,
                    d_pos,
                    slots,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    max_ctx,
                    kv_dim,
                    swa_window,
                    b,
                    scale,
                    kv_dtype,
                )?;
            }
            if b1_fast {
                exec.quantize_q8(&bs.d_attn, &mut bs.d_b1_xq, &mut bs.d_b1_xs, q_dim)?;
                exec.q8_0_gemv_dp4a_nc_b(
                    &layer.wo_r,
                    Some(&layer.bo.buf),
                    &bs.d_b1_xq,
                    &bs.d_b1_xs,
                    &mut bs.d_proj,
                    1,
                )?;
            } else {
                mm_quant(
                    &exec,
                    &bs.d_attn,
                    &mut bs.d_p_xq,
                    &mut bs.d_p_xs,
                    &mut bs.d_p_yq,
                    q_dim,
                    b,
                )?;
                mm_pre(
                    &exec,
                    &layer.wo_r,
                    &layer.bo.buf,
                    &bs.d_p_xq,
                    &bs.d_p_xs,
                    &bs.d_p_yq,
                    &mut bs.d_p_skfix,
                    &mut bs.d_proj,
                    b,
                )?;
            }
            // ---- MoE (expert-grouped: each expert row dequanted once, reused
            // across every row that selected it; down atomic-adds into d_x, which
            // holds the post-attention residual). Residual add + post_norm ride
            // one fused launch (bit-identical to the add-then-norm pair).
            let moe_fused_q = moe_bs && exec.has_add_rmsnorm_quant_e4m3();
            if moe_fused_q {
                // post-norm + e4m3 quantize fused: the f32 plane still lands
                // in d_xn (the router reads it); the MoE's activation planes
                // come out of the same pass
                exec.add_rmsnorm_quant_e4m3_batch(
                    &mut bs.d_x,
                    &bs.d_proj,
                    &layer.post_norm.buf,
                    &mut bs.d_xn,
                    &mut bs.d_p_xq,
                    &mut bs.d_p_xs8,
                    embd,
                    rms_eps,
                    b,
                )?;
            } else {
                exec.add_rmsnorm_batch(
                    &mut bs.d_x,
                    &bs.d_proj,
                    &layer.post_norm.buf,
                    &mut bs.d_xn,
                    embd,
                    rms_eps,
                    b,
                )?;
            }
            // router: the in-house tile kernel at every batch. Warm - the
            // serve case, x written by the rmsnorm just above, router weights
            // L2-resident - it beats cuBLAS sgemm at b<=512 and ties from
            // b>=1024, so the old b<=64 cuBLAS split cost latency at small b
            // for nothing.
            exec.matvec_f32_batch(&layer.router_w, &bs.d_xn, &mut bs.d_router, b)?;
            exec.moe_topk_batch(
                &bs.d_router,
                &layer.router_b.buf,
                n_experts,
                n_active,
                &mut bs.d_topk_idx,
                &mut bs.d_topk_w,
                b,
            )?;
            // Route by batch: fast batches above the dp4a window take the
            // SORTED int8 tiled path (moe_align -> contiguous per-expert tokens
            // -> mmq gate_up + down). Even at b=5 it beats grouped: sorted
            // reads each touched expert's weights once, while grouped's per-
            // output-row f32 dots re-read x per (token, expert) pair - 26
            // ms/step at B=2, 101 ms at B=16. M-padding to the 32-row tile is
            // free next to that (weight-bound).
            if mm_fast && b > moe_dp4a_max(exec.compute_capability().0, n_experts) {
                // int8 mmq MoE: quantize the post-norm rows strided, run the
                // sorted gate_up (fp4 experts as int8 tiles), re-quantize the
                // swiglu output, then the int8 down + weighted residual add.
                let max_blocks = moe_max_blocks(b, n_active, n_experts);
                exec.moe_align(
                    &bs.d_topk_idx,
                    &mut bs.d_sorted_row,
                    &mut bs.d_sorted_slot,
                    &mut bs.d_block_expert,
                    b,
                    n_active,
                    n_experts,
                    max_blocks,
                )?;
                if moe_bs {
                    // sm_120a block-scale route: mxFP4 weights straight into
                    // the tensor core, e4m3 activations, hardware ue8m0
                    // scaling - the s8 mmq pair ran at ~1/4 of the tensor
                    // pipe on GB202 (unpack + rescale issue pressure).
                    // NUMERIC CLASS: fp8 activations (PADDOCK_NO_MOE_BS pins
                    // the int8 mmq class).
                    if !moe_fused_q {
                        exec.quantize_e4m3(&bs.d_xn, &mut bs.d_p_xq, &mut bs.d_p_xs8, b * embd)?;
                    }
                    exec.mxfp4_moe_gate_up_bs(
                        &layer.gate_exps,
                        &layer.gate_exps_b.buf,
                        &layer.up_exps,
                        &layer.up_exps_b.buf,
                        &bs.d_sorted_row,
                        &bs.d_block_expert,
                        &bs.d_p_xq,
                        &bs.d_p_xs8,
                        &mut bs.d_moe_xq,
                        &mut bs.d_moe_xs8,
                        embd,
                        ff,
                        max_blocks,
                        b,
                        SWIGLU_ALPHA,
                        SWIGLU_LIMIT,
                        1.0, // gpt-oss SwiGLU: silu(alpha*g)*(clamp(u)+1)
                    )?;
                    exec.mxfp4_moe_down_bs(
                        &layer.down_exps,
                        &layer.down_exps_b.buf,
                        &bs.d_sorted_row,
                        &bs.d_sorted_slot,
                        &bs.d_block_expert,
                        &bs.d_topk_w,
                        &bs.d_moe_xq,
                        &bs.d_moe_xs8,
                        &mut bs.d_gate_up,
                        ff,
                        embd,
                        n_active,
                        max_blocks,
                        b,
                    )?;
                } else {
                    exec.quantize_q8(&bs.d_xn, &mut bs.d_p_xq, &mut bs.d_p_xs, b * embd)?;
                    exec.mxfp4_moe_gate_up_mmq(
                        &layer.gate_exps,
                        &layer.gate_exps_b.buf,
                        &layer.up_exps,
                        &layer.up_exps_b.buf,
                        &bs.d_sorted_row,
                        &bs.d_block_expert,
                        &bs.d_p_xq,
                        &bs.d_p_xs,
                        &mut bs.d_moe_xq,
                        &mut bs.d_moe_xs,
                        embd,
                        ff,
                        max_blocks,
                        SWIGLU_ALPHA,
                        SWIGLU_LIMIT,
                        1.0, // gpt-oss SwiGLU: silu(alpha*g)*(clamp(u)+1)
                    )?;
                    // down emits per-(token, slot) partials (no atomics) into
                    // d_gate_up - idle on this path and exactly [row_cap,
                    // n_active, ff >= embd] - then the fixed-order fold makes
                    // the whole MoE bit-reproducible (an atomic scatter
                    // flipped near-tie greedy tokens at the b9895 gate).
                    exec.mxfp4_moe_down_mmq(
                        &layer.down_exps,
                        &layer.down_exps_b.buf,
                        &bs.d_sorted_row,
                        &bs.d_sorted_slot,
                        &bs.d_block_expert,
                        &bs.d_topk_w,
                        &bs.d_moe_xq,
                        &bs.d_moe_xs,
                        &mut bs.d_gate_up,
                        ff,
                        embd,
                        n_active,
                        max_blocks,
                    )?;
                }
                if attn_fused_q && exec.has_moe_combine_rmsnorm_quant() && li + 1 < self.n_layers {
                    // defer: the fold rides the next layer's norm pass
                    // (d_gate_up stays untouched until then)
                    pending_combine = true;
                } else {
                    exec.moe_slot_combine(&bs.d_gate_up, &mut bs.d_x, embd, n_active, b)?;
                }
            } else if mm_fast {
                // 2..=MOE_DP4A_MAX_BATCH: batched fused dp4a MoE (the llama
                // mmvq-with-ids shape). Each token re-reads its own experts'
                // weights, but the grid is ff/8 x n_active x b well-filled
                // GEMV blocks - the mmq tiles at this size are a handful of
                // blocks whose deep-staged K-walks run latency-bound (5.8
                // ms/step at B=2 vs a ~3.6 ms traffic floor here).
                exec.quantize_q8(&bs.d_xn, &mut bs.d_p_xq, &mut bs.d_p_xs, b * embd)?;
                exec.mxfp4_moe_gate_up_dp4a_b(
                    &layer.gate_exps,
                    &layer.gate_exps_b.buf,
                    &layer.up_exps,
                    &layer.up_exps_b.buf,
                    &bs.d_topk_idx,
                    &bs.d_p_xq,
                    &bs.d_p_xs,
                    &mut bs.d_gate_up,
                    embd,
                    ff,
                    n_active,
                    b,
                    SWIGLU_ALPHA,
                    SWIGLU_LIMIT,
                )?;
                exec.quantize_q8(
                    &bs.d_gate_up,
                    &mut bs.d_moe_xq,
                    &mut bs.d_moe_xs,
                    b * n_active * ff,
                )?;
                exec.mxfp4_moe_down_dp4a_b(
                    &layer.down_exps,
                    &layer.down_exps_b.buf,
                    &bs.d_topk_idx,
                    &bs.d_topk_w,
                    &bs.d_moe_xq,
                    &bs.d_moe_xs,
                    &mut bs.d_x,
                    ff,
                    embd,
                    n_active,
                    b,
                )?;
            } else {
                // fused single-token dp4a MoE: quantize the post-norm activation,
                // gate+up+swiglu in one launch, re-quantize the swiglu output, then
                // down + weighted mix + residual add in one launch.
                exec.quantize_q8(&bs.d_xn, &mut bs.d_b1_xq, &mut bs.d_b1_xs, embd)?;
                exec.mxfp4_moe_gate_up_dp4a(
                    &layer.gate_exps,
                    &layer.gate_exps_b.buf,
                    &layer.up_exps,
                    &layer.up_exps_b.buf,
                    &bs.d_topk_idx,
                    &bs.d_b1_xq,
                    &bs.d_b1_xs,
                    &mut bs.d_gate_up,
                    embd,
                    ff,
                    n_active,
                    SWIGLU_ALPHA,
                    SWIGLU_LIMIT,
                )?;
                exec.quantize_q8(
                    &bs.d_gate_up,
                    &mut bs.d_b1_xq,
                    &mut bs.d_b1_xs,
                    n_active * ff,
                )?;
                exec.mxfp4_moe_down_dp4a(
                    &layer.down_exps,
                    &layer.down_exps_b.buf,
                    &bs.d_topk_idx,
                    &bs.d_topk_w,
                    &bs.d_b1_xq,
                    &bs.d_b1_xs,
                    &mut bs.d_x,
                    ff,
                    embd,
                    n_active,
                )?;
            }
        }
        Ok(())
    }

    /// One batched decode step: `tokens` (one per sequence) at `positions` (each
    /// sequence's own KV position). Returns [B, vocab] logits, row b = sequence
    /// b's next-token logits. Reads each weight once for the whole batch - the
    /// throughput amortization. Uses the float batched kernels.
    pub fn forward_batch(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
    ) -> Result<Vec<f32>, GpuModelError> {
        let exec = self.exec.clone();
        let vocab = self.vocab;
        let b = tokens.len();
        match &self.batch {
            None => return Err(GpuModelError::BatchDisabled),
            Some(bs) if b > bs.max_batch => {
                return Err(GpuModelError::BatchTooLarge {
                    got: b,
                    max: bs.max_batch,
                });
            }
            _ => {}
        }
        self.launch_batch_step(tokens, positions)?;
        let bs = self.batch.as_ref().ok_or(GpuModelError::BatchDisabled)?;
        // read back only the b live rows - the buffer holds max_batch*vocab and
        // copying it whole costs a fixed ~1.4 ms/step at any B (qwen P5c lesson).
        // (The >32 MiB readback cliff was the host allocator, not this copy:
        // see the mallopt note in GpuExecutor::new.)
        let view = bs
            .d_logits
            .try_slice(0..b * vocab)
            .ok_or_else(|| GpuError::Driver("logits slice out of range".into()))?;
        let logits = exec
            .stream
            .clone_dtoh(&view)
            .map_err(|e| GpuError::Driver(e.to_string()))?;
        Ok(logits)
    }

    /// Upload the step inputs and run the batched decode step (graph replay,
    /// or eager when a numerics pin is set), leaving `[b, vocab]` logits in
    /// `d_logits`. Shared by `forward_batch` and `forward_batch_sampled`.
    fn launch_batch_step(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
    ) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let b = tokens.len();
        self.ensure_pf_inputs()?;
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
        // step inputs land in the fixed buffers, outside any graph - only
        // their contents change between replays
        {
            let mut v = self
                .d_pf_tok
                .as_mut()
                .expect("pf inputs ensured")
                .slice_mut(0..b);
            exec.stream.memcpy_htod(tokens, &mut v).map_err(drv)?;
            let mut v = self
                .d_pf_pos
                .as_mut()
                .expect("pf inputs ensured")
                .slice_mut(0..b);
            exec.stream.memcpy_htod(positions, &mut v).map_err(drv)?;
        }
        // G4a: grow the pool for each decode row (the batch step maps row i ->
        // slot i - d_slots is None in record_batch_step's run_layers). Runs
        // before the graph replay, so d_bt[0] holds this step's mapping.
        if self.pool_active() {
            let slots: Vec<u32> = (0..b as u32).collect();
            self.ensure_pool_rows(&slots, positions)?;
        }
        // Diagnostic pins force eager - a cached graph baked the record-time
        // dispatch and would silently ignore the env (P6m lesson).
        let eager = paddock_models::dev_var_os!("PADDOCK_NO_STEP_GRAPH").is_some()
            || paddock_models::dev_var_os!("PADDOCK_STEP_PHASE_TIME").is_some();
        if eager {
            self.record_batch_step(b)?;
        } else {
            if !self
                .batch
                .as_ref()
                .expect("batch enabled")
                .step_graphs
                .contains_key(&b)
            {
                self.capture_batch_step(b)?;
            }
            self.batch.as_ref().expect("batch enabled").step_graphs[&b]
                .0
                .launch()
                .map_err(|e| GpuError::Driver(format!("step graph launch: {e}")))?;
        }
        Ok(())
    }

    /// True when the pack ships the fused row sampler - the engine's
    /// capability probe for `forward_batch_sampled`.
    pub fn supports_device_sampling(&self) -> bool {
        self.exec.has_sample_rows()
    }

    /// `forward_batch` + fused on-device sampling: eligible rows return bare
    /// token ids (b × 4 bytes across the bus) instead of the full `[b, vocab]`
    /// logits readback (25.7 MB/step at B=32); `Host` rows still get their
    /// own logits row copied back individually.
    pub fn forward_batch_sampled(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        plans: &[crate::generator::RowSample],
    ) -> Result<crate::generator::SampledStep, GpuModelError> {
        use crate::generator::{RowSample, SampledStep};
        use crate::sampler::DevicePlan;
        let exec = self.exec.clone();
        let vocab = self.vocab;
        let b = tokens.len();
        assert_eq!(plans.len(), b, "one plan per row");
        match &self.batch {
            None => return Err(GpuModelError::BatchDisabled),
            Some(bs) if b > bs.max_batch => {
                return Err(GpuModelError::BatchTooLarge {
                    got: b,
                    max: bs.max_batch,
                });
            }
            _ => {}
        }
        // pack the per-row sampler params; the kernel skips mode-0 rows
        // (holes AND host rows - the host reads the latter's logits itself)
        let mut par = vec![0u32; b * 4];
        let mut tpar = vec![0u32; b * 4];
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
                // dialled truncation rows sample fully on device
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
                // RS plans are gemma4-only (supports_spec_rs); skip-safe
                RowSample::Device(DevicePlan::RsVerify { .. })
                | RowSample::Device(DevicePlan::RsTrunc { .. }) => {}
            }
        }
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
        {
            let bs = self.batch.as_mut().expect("batch enabled");
            let mut v = bs.d_samp_par.slice_mut(0..b * 4);
            exec.stream.memcpy_htod(&par, &mut v).map_err(drv)?;
            if any_trunc {
                let mut v = bs.d_samp_tpar.slice_mut(0..b * 4);
                exec.stream.memcpy_htod(&tpar, &mut v).map_err(drv)?;
            }
        }
        self.launch_batch_step(tokens, positions)?;
        let bs = self.batch.as_mut().ok_or(GpuModelError::BatchDisabled)?;
        exec.sample_rows(&bs.d_logits, &bs.d_samp_par, &mut bs.d_samp_out, b, vocab)?;
        if any_trunc {
            Self::trunc_dev_witness(b);
            exec.sample_rows_t(
                &bs.d_logits,
                &bs.d_samp_par,
                &bs.d_samp_tpar,
                &mut bs.d_samp_out,
                b,
                vocab,
            )?;
            exec.sample_rows_p(
                &bs.d_logits,
                &bs.d_samp_par,
                &bs.d_samp_tpar,
                &mut bs.d_samp_out,
                b,
                vocab,
            )?;
        }
        let ids_view = bs
            .d_samp_out
            .try_slice(0..b)
            .ok_or_else(|| GpuError::Driver("samp_out slice out of range".into()))?;
        let ids = exec
            .stream
            .clone_dtoh(&ids_view)
            .map_err(|e| GpuError::Driver(e.to_string()))?;
        let mut host_rows = Vec::new();
        for (i, p) in plans.iter().enumerate() {
            if matches!(p, RowSample::Host) {
                let view = bs
                    .d_logits
                    .try_slice(i * vocab..(i + 1) * vocab)
                    .ok_or_else(|| GpuError::Driver("logits row slice out of range".into()))?;
                let row = exec
                    .stream
                    .clone_dtoh(&view)
                    .map_err(|e| GpuError::Driver(e.to_string()))?;
                host_rows.push((i, row));
            }
        }
        Ok(SampledStep { ids, host_rows })
    }

    /// True when the pipelined decode can run: step graphs + device sampling
    /// + the advance kernel, with no diagnostic pin forcing the eager path (a
    ///   pipe would replay a graph that baked the un-pinned dispatch).
    pub fn supports_decode_pipe(&self) -> bool {
        self.exec.has_sample_rows()
            && self.exec.has_pipe_advance()
            && paddock_models::dev_var_os!("PADDOCK_NO_DECODE_PIPE").is_none()
            && paddock_models::dev_var_os!("PADDOCK_NO_STEP_GRAPH").is_none()
            && paddock_models::dev_var_os!("PADDOCK_STEP_PHASE_TIME").is_none()
    }

    /// Pack per-row sampler params exactly like `forward_batch_sampled`
    /// (Host is treated as skip - the pipe never runs with Host rows).
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
                // RS plans are gemma4-only (supports_spec_rs); skip-safe
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
            eprintln!("[trunc-dev6] engaged: r={rows} (gpt-oss device truncation sampling)");
        });
    }

    /// TruncCat rows execute fully on device (slots 435+436).
    pub fn device_trunc_supported(&self) -> bool {
        self.batch.is_some() && self.exec.has_sample_rows_t() && self.exec.has_sample_rows_p()
    }

    /// Begin a pipelined decode: upload the tick-0 inputs and enqueue tick 0
    /// (graph replay + device sampling). No ids come back yet - the first
    /// `decode_pipe_next` call returns them while tick 1 runs.
    pub fn decode_pipe_begin(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        plans: &[crate::generator::RowSample],
    ) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let b = tokens.len();
        assert_eq!(plans.len(), b, "one plan per row");
        if !self.supports_decode_pipe() {
            return Err(GpuModelError::Unsupported("decode pipe".into()));
        }
        match &self.batch {
            None => return Err(GpuModelError::BatchDisabled),
            Some(bs) if b > bs.max_batch => {
                return Err(GpuModelError::BatchTooLarge {
                    got: b,
                    max: bs.max_batch,
                });
            }
            _ => {}
        }
        assert!(self.pipe.is_none(), "decode pipe already active");
        self.ensure_pf_inputs()?;
        if !self
            .batch
            .as_ref()
            .expect("batch enabled")
            .step_graphs
            .contains_key(&b)
        {
            self.capture_batch_step(b)?;
        }
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
        {
            let mut v = self
                .d_pf_tok
                .as_mut()
                .expect("pf inputs ensured")
                .slice_mut(0..b);
            exec.stream.memcpy_htod(tokens, &mut v).map_err(drv)?;
            let mut v = self
                .d_pf_pos
                .as_mut()
                .expect("pf inputs ensured")
                .slice_mut(0..b);
            exec.stream.memcpy_htod(positions, &mut v).map_err(drv)?;
        }
        self.pipe = Some(PipeState {
            b,
            tick: 0,
            ev: [None, None],
        });
        if let Err(e) = self.pipe_launch_tick(plans, false) {
            self.pipe_abort();
            return Err(e);
        }
        Ok(())
    }

    /// Enqueue the next pipelined tick (its tokens/positions advance on
    /// device from the previous tick's sampler output), then return the ids
    /// of the OLDEST in-flight tick - read via the side stream while the new
    /// tick executes. `plans[i]` must be Device or Hole, same rows as begin.
    pub fn decode_pipe_next(
        &mut self,
        plans: &[crate::generator::RowSample],
    ) -> Result<Vec<u32>, GpuModelError> {
        let exec = self.exec.clone();
        let (b, j) = {
            let p = self.pipe.as_ref().ok_or_else(|| {
                GpuModelError::Unsupported("decode_pipe_next without begin".into())
            })?;
            (p.b, p.tick)
        };
        assert_eq!(plans.len(), b, "one plan per row");
        self.pipe.as_mut().expect("pipe checked above").tick = j + 1;
        if let Err(e) = self.pipe_launch_tick(plans, true) {
            self.pipe_abort();
            return Err(e);
        }
        let ring = (j % 2) as usize;
        let r = {
            let bs = self.batch.as_ref().ok_or(GpuModelError::BatchDisabled)?;
            let ev = self.pipe.as_ref().expect("pipe checked above").ev[ring]
                .as_ref()
                .expect("in-flight tick event");
            exec.to_host_u32_after(ev, &bs.d_pipe_out, ring * bs.max_batch, b)
        };
        match r {
            Ok(ids) => Ok(ids),
            Err(e) => {
                self.pipe_abort();
                Err(e.into())
            }
        }
    }

    /// End the pipe: return the last in-flight tick's ids without enqueueing
    /// more work. The fixed input buffers are left stale - every other
    /// forward path re-uploads them before use.
    pub fn decode_pipe_drain(&mut self) -> Result<Vec<u32>, GpuModelError> {
        let exec = self.exec.clone();
        let st = self
            .pipe
            .take()
            .ok_or_else(|| GpuModelError::Unsupported("decode_pipe_drain without begin".into()))?;
        let ring = (st.tick % 2) as usize;
        let ev = st.ev[ring].as_ref().expect("in-flight tick event");
        let bs = self.batch.as_ref().ok_or(GpuModelError::BatchDisabled)?;
        match exec.to_host_u32_after(ev, &bs.d_pipe_out, ring * bs.max_batch, st.b) {
            Ok(ids) => Ok(ids),
            Err(e) => {
                // state is already gone - quiesce so nothing still reads the rings
                let _ = exec.synchronize();
                Err(e.into())
            }
        }
    }

    /// Enqueue pipelined tick `pipe.tick`: params into its par ring slot, the
    /// device-side advance (skipped for tick 0 - its inputs were just
    /// uploaded), the captured step graph, the row sampler into its out ring
    /// slot, and the readability event.
    fn pipe_launch_tick(
        &mut self,
        plans: &[crate::generator::RowSample],
        advance: bool,
    ) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let vocab = self.vocab;
        let (b, tick) = {
            let p = self.pipe.as_ref().expect("pipe active");
            (p.b, p.tick)
        };
        let ring = (tick % 2) as usize;
        let (par, tpar) = Self::pack_samp_par(plans);
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
        let max_batch = self.batch.as_ref().expect("batch enabled").max_batch;
        {
            let bs = self.batch.as_mut().expect("batch enabled");
            let off = ring * max_batch * 4;
            let mut v = bs.d_pipe_par.slice_mut(off..off + b * 4);
            exec.stream.memcpy_htod(&par, &mut v).map_err(drv)?;
            if let Some(t) = &tpar {
                let mut v = bs.d_pipe_tpar.slice_mut(off..off + b * 4);
                exec.stream.memcpy_htod(t, &mut v).map_err(drv)?;
            }
        }
        if advance {
            // previous tick's out ring slot becomes this tick's input tokens
            let prev = ((tick + 1) % 2) as usize;
            let mut d_tok = self.d_pf_tok.take().expect("pf buffers");
            let mut d_pos = self.d_pf_pos.take().expect("pf buffers");
            let r = {
                let bs = self.batch.as_ref().expect("batch enabled");
                exec.pipe_advance(&bs.d_pipe_out, prev * max_batch, &mut d_tok, &mut d_pos, b)
            };
            self.d_pf_tok = Some(d_tok);
            self.d_pf_pos = Some(d_pos);
            r?;
        }
        self.batch.as_ref().expect("batch enabled").step_graphs[&b]
            .0
            .launch()
            .map_err(|e| GpuError::Driver(format!("pipe step graph launch: {e}")))?;
        {
            let bs = self.batch.as_mut().expect("batch enabled");
            // scoped: d_logits immut + d_pipe_out mut are disjoint fields
            let (logits, par_buf, out) = (&bs.d_logits, &bs.d_pipe_par, &mut bs.d_pipe_out);
            exec.sample_rows_at(
                logits,
                par_buf,
                ring * max_batch * 4,
                out,
                ring * max_batch,
                b,
                vocab,
            )?;
            if tpar.is_some() {
                Self::trunc_dev_witness(b);
                let tpar_buf = &bs.d_pipe_tpar;
                exec.sample_rows_t_at(
                    logits,
                    par_buf,
                    ring * max_batch * 4,
                    tpar_buf,
                    ring * max_batch * 4,
                    out,
                    ring * max_batch,
                    b,
                    vocab,
                )?;
                exec.sample_rows_p_at(
                    logits,
                    par_buf,
                    ring * max_batch * 4,
                    tpar_buf,
                    ring * max_batch * 4,
                    out,
                    ring * max_batch,
                    b,
                    vocab,
                )?;
            }
        }
        let ev = exec.record_event()?;
        self.pipe.as_mut().expect("pipe active").ev[ring] = Some(ev);
        Ok(())
    }

    /// Kill an in-flight pipe after an error (or on reset): quiesce the
    /// stream so nothing still reads the pipe buffers, then drop the state.
    fn pipe_abort(&mut self) {
        if self.pipe.take().is_some() {
            let _ = self.exec.synchronize();
        }
    }

    /// Record one batched decode step - embed gather, layers, out-norm,
    /// lm_head; logits left in `d_logits` - onto the stream. Shared by the
    /// eager path and per-B graph capture. Every input reads the fixed
    /// `d_pf_tok`/`d_pf_pos` buffers, nothing allocates or syncs, and no
    /// launch geometry depends on position (the split heuristic is
    /// position-independent), so a capture replays correctly at any position.
    fn record_batch_step(&mut self, b: usize) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let embd = self.embd;
        // move the fixed input buffers out around the `&mut self` run_layers
        // call (host-side move - device addresses unchanged; the d_g_pos trick)
        let d_tok = self.d_pf_tok.take().expect("pf buffers");
        let d_pos = self.d_pf_pos.take().expect("pf buffers");
        let phase_time = paddock_models::dev_var_os!("PADDOCK_STEP_PHASE_TIME").is_some();
        let r = (|| -> Result<(), GpuModelError> {
            {
                let bs = self.batch.as_mut().ok_or(GpuModelError::BatchDisabled)?;
                exec.embed_gather_batch_q8(&self.tok_embd, &d_tok, &mut bs.d_x, embd, b)?;
            }
            let t0 = if phase_time {
                exec.synchronize()?;
                Some(std::time::Instant::now())
            } else {
                None
            };
            // max_pos only feeds the (position-independent) split heuristic
            self.run_layers(b, &d_pos, None, self.max_ctx - 1, false, None)?;
            if let Some(t0) = t0 {
                exec.synchronize()?;
                let t1 = std::time::Instant::now();
                let r = self.record_out_head(b);
                self.exec.synchronize()?;
                tracing::info!(
                    "step b={b}: layers {:.2} ms, head {:.2} ms",
                    (t1 - t0).as_secs_f64() * 1e3,
                    t1.elapsed().as_secs_f64() * 1e3
                );
                return r;
            }
            self.record_out_head(b)
        })();
        self.d_pf_tok = Some(d_tok);
        self.d_pf_pos = Some(d_pos);
        r
    }

    /// Out-norm + lm_head for `b` rows (logits left in `d_logits`) - the
    /// shared tail of the batched decode step and the spec verify pass.
    fn record_out_head(&mut self, b: usize) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let (embd, rms_eps) = (self.embd, self.rms_eps);
        let bs = self.batch.as_mut().ok_or(GpuModelError::BatchDisabled)?;
        let head_fused_q = b > 1 && b <= 64 && exec.has_rmsnorm_quant_q8();
        if head_fused_q {
            exec.rmsnorm_quant_q8_batch(
                &bs.d_x,
                &self.out_norm.buf,
                &mut bs.d_xn,
                &mut bs.d_p_xq,
                &mut bs.d_p_xs,
                embd,
                rms_eps,
                b,
            )?;
        } else {
            exec.rmsnorm_batch(&bs.d_x, &self.out_norm.buf, &mut bs.d_xn, embd, rms_eps, b)?;
        }
        if b == 1 {
            exec.quantize_q8(&bs.d_xn, &mut bs.d_b1_xq, &mut bs.d_b1_xs, embd)?;
            exec.q8_0_gemv_dp4a_nc(
                &self.output_r,
                &bs.d_b1_xq,
                &bs.d_b1_xs,
                &mut bs.d_logits,
                1,
            )?;
        } else if b <= 64 {
            // batched lm_head on the int8 quantized class (same class as
            // the B=1 dp4a path). The exact f32 gemm here was 26 ms/step
            // at B=32 - over half the whole step. Ladder differs from the
            // dense mm_pre one: N=201088 fills the mma grid, so mma takes
            // over as soon as the dp4a MT tile needs a second weight pass
            // (b > 16) - a z-tile re-read of 614 MB loses to one TC pass.
            if !head_fused_q {
                exec.quantize_q8(&bs.d_xn, &mut bs.d_p_xq, &mut bs.d_p_xs, b * embd)?;
            }
            if b <= 4 {
                exec.q8_0_gemv_dp4a_nc(
                    &self.output_r,
                    &bs.d_p_xq,
                    &bs.d_p_xs,
                    &mut bs.d_logits,
                    b,
                )?;
            } else if b <= 32 {
                // ks rungs only (BN16/BN32-ST2, both 2 blocks/SM): the
                // standalone BN64 tile measured 557.6 us = 1.54x floor on
                // this grid (b>16 used to route there).
                // ks BN16 tile: 2-stage cp.async, still 2 blocks/SM at its
                // smem size. Not the standalone BN64-ST2 tile - doubled
                // smem drops that one to 1 block/SM, ~3x slower across a
                // 3142-block vocab grid.
                // nz collapses to 1 at 3142 tiles, so part stays unused.
                // (The pre-pipeline single-pass dp4a MT tile that once won here
                // lost to this ks tile once the cp.async pipeline landed.)
                let (xq, xs, part, logits) =
                    (&bs.d_p_xq, &bs.d_p_xs, &mut bs.d_p_skfix, &mut bs.d_logits);
                exec.q8_0_gemm_mma_ks(&self.output_r, xq, xs, part, logits, b)?;
            } else {
                exec.q8_0_gemm_mma(&self.output_r, &bs.d_p_xq, &bs.d_p_xs, &mut bs.d_logits, b)?;
            }
        } else {
            exec.q8_0_gemm(
                self.output.as_ref().expect(RAW_HEAD_DROPPED),
                None,
                &bs.d_xn,
                &mut bs.d_logits,
                b,
            )?;
        }
        Ok(())
    }

    /// Capture [`Self::record_batch_step`] into a replayable graph, cached per
    /// batch size. Same contract as `capture_prefill_chunk`.
    fn capture_batch_step(&mut self, b: usize) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        exec.stream
            .synchronize()
            .map_err(|e| GpuError::Driver(format!("pre-capture sync: {e}")))?;
        exec.stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|e| GpuError::Driver(format!("begin_capture: {e}")))?;
        let rec = self.record_batch_step(b);
        let graph = crate::gpu::end_capture_no_flags(&exec.stream)
            .map_err(|e| GpuError::Driver(format!("end_capture: {e}")));
        rec?; // surface a record failure only after capture is cleanly ended
        let graph =
            graph?.ok_or_else(|| GpuError::Driver("step capture produced no graph".into()))?;
        let bs = self.batch.as_mut().ok_or(GpuModelError::BatchDisabled)?;
        // continuous batching sweeps b as sequences finish - bound the cache
        if bs.step_graphs.len() >= 32 {
            bs.step_graphs.clear();
        }
        bs.step_graphs.insert(b, SendGraph(graph));
        Ok(())
    }

    /// Record one spec-verify pass: `r` rows (per participating slot: 1
    /// committed token + its drafts) at the positions/slots in the fixed
    /// `d_pf_pos`/`d_pf_slots` buffers, causal within the pass (a slot's rows
    /// ascend in position and each row's KV lands before its attention reads
    /// it - the same mechanics as a prefill chunk; rows of different slots
    /// touch disjoint caches), then out-head + device argmax of every row
    /// into `d_spec_pick`. uniform_slot is false: rows may span slots, and
    /// r <= SPEC_BATCH_MAX_ROWS keeps the pass under the pf16 threshold
    /// anyway. Rollback after a rejected draft is free on a plain
    /// transformer: attention is bounded by each row's own position, so KV
    /// written by rejected rows is never read before a later pass overwrites
    /// those positions.
    fn record_verify_rows(&mut self, r: usize) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let (embd, vocab) = (self.embd, self.vocab);
        let d_tok = self.d_pf_tok.take().expect("pf buffers");
        let d_pos = self.d_pf_pos.take().expect("pf buffers");
        let d_slots = self.d_pf_slots.take().expect("pf buffers");
        let res = (|| -> Result<(), GpuModelError> {
            {
                let bs = self.batch.as_mut().ok_or(GpuModelError::BatchDisabled)?;
                exec.embed_gather_batch_q8(&self.tok_embd, &d_tok, &mut bs.d_x, embd, r)?;
            }
            self.run_layers(r, &d_pos, Some(&d_slots), self.max_ctx - 1, false, None)?;
            self.record_out_head(r)?;
            let bs = self.batch.as_mut().ok_or(GpuModelError::BatchDisabled)?;
            exec.argmax_rows(&bs.d_logits, &mut bs.d_spec_pick, r, vocab)?;
            Ok(())
        })();
        self.d_pf_tok = Some(d_tok);
        self.d_pf_pos = Some(d_pos);
        self.d_pf_slots = Some(d_slots);
        res
    }

    /// Capture [`Self::record_verify_rows`] into a replayable graph, cached
    /// per row count. Same contract as `capture_batch_step`.
    fn capture_verify_rows(&mut self, r: usize) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        exec.stream
            .synchronize()
            .map_err(|e| GpuError::Driver(format!("pre-capture sync: {e}")))?;
        exec.stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|e| GpuError::Driver(format!("begin_capture: {e}")))?;
        let rec = self.record_verify_rows(r);
        let graph = crate::gpu::end_capture_no_flags(&exec.stream)
            .map_err(|e| GpuError::Driver(format!("end_capture: {e}")));
        rec?; // surface a record failure only after capture is cleanly ended
        let graph =
            graph?.ok_or_else(|| GpuError::Driver("verify capture produced no graph".into()))?;
        let bs = self.batch.as_mut().ok_or(GpuModelError::BatchDisabled)?;
        if bs.spec_graphs.len() >= 16 {
            bs.spec_graphs.clear();
        }
        bs.spec_graphs.insert(r, SendGraph(graph));
        Ok(())
    }

    /// One speculative verify pass over RAGGED per-slot chunks - the serving
    /// spec-decode round. `reqs[i] = (slot, start_pos, chunk)` where
    /// `chunk[0]` is that slot's committed pending token and `chunk[1..]`
    /// its drafts; the chunk occupies KV positions `start_pos..` in `slot`.
    /// All chunks run as one multi-slot batched pass (weights read once for
    /// every row of every sequence), and each row's greedy pick is computed
    /// on device - only total-rows u32s cross the bus. Picks return flat, in
    /// request order: caller accepts per slot while `chunk[j+1] ==
    /// picks[base + j]`, emits `picks[base..=base+a]`, and advances that
    /// slot's position by `a + 1`; rejected rows need no rollback (their KV
    /// is overwritten before any row can attend to it). Greedy only - picks
    /// are argmaxes, so sampling callers cannot use this pass.
    pub fn forward_spec_batch(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
    ) -> Result<Vec<u32>, GpuModelError> {
        self.moe_bs_pin = true;
        let r = self.forward_spec_batch_inner(reqs);
        self.moe_bs_pin = false;
        r
    }

    /// Test/telemetry hook: run a spec-verify pass and return the RAW per-row
    /// logits `[r, vocab]` (row-major, request order) instead of the on-device
    /// argmax picks. The verify already fills `d_logits` via `record_out_head`
    /// before `argmax_rows`, so this just reads that buffer back. Used to
    /// validate the pool block-table wiring against the off-pool path at the
    /// LOGIT level - argmax is knife-edge non-reproducible on sm120 multi-row
    /// MoE, but the logit vectors are stable to a small rel_err (P1 method).
    pub fn spec_batch_logits(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
    ) -> Result<Vec<f32>, GpuModelError> {
        self.moe_bs_pin = true;
        let picks = self.forward_spec_batch_inner(reqs);
        self.moe_bs_pin = false;
        let rows = picks?.len();
        let exec = self.exec.clone();
        let vocab = self.vocab;
        let bs = self.batch.as_ref().ok_or(GpuModelError::BatchDisabled)?;
        let view = bs
            .d_logits
            .try_slice(0..rows * vocab)
            .ok_or_else(|| GpuError::Driver("spec logits slice out of range".into()))?;
        let logits = exec
            .stream
            .clone_dtoh(&view)
            .map_err(|e| GpuError::Driver(e.to_string()))?;
        Ok(logits)
    }

    fn forward_spec_batch_inner(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
    ) -> Result<Vec<u32>, GpuModelError> {
        let exec = self.exec.clone();
        let max_batch = match &self.batch {
            None => return Err(GpuModelError::BatchDisabled),
            Some(bs) => bs.max_batch,
        };
        let r: usize = reqs.iter().map(|(_, _, c)| c.len()).sum();
        if r == 0 || r > SPEC_BATCH_MAX_ROWS {
            return Err(GpuModelError::BatchTooLarge {
                got: r,
                max: SPEC_BATCH_MAX_ROWS,
            });
        }
        let (mut tokens, mut positions, mut slots) = (
            Vec::with_capacity(r),
            Vec::with_capacity(r),
            Vec::with_capacity(r),
        );
        for (slot, start_pos, chunk) in reqs {
            if *slot >= max_batch {
                return Err(GpuModelError::BatchTooLarge {
                    got: slot + 1,
                    max: max_batch,
                });
            }
            if chunk.is_empty() || start_pos + chunk.len() > self.max_ctx {
                return Err(GpuModelError::BatchTooLarge {
                    got: start_pos + chunk.len(),
                    max: self.max_ctx,
                });
            }
            for (i, &t) in chunk.iter().enumerate() {
                tokens.push(t);
                positions.push((start_pos + i) as u32);
                slots.push(*slot as u32);
            }
        }
        // P3 (spec under pool): grow the pool for the whole draft span before the
        // verify graph reads the baked block table. Each req's chunk is k+1 ≤ 8
        // rows, so it crosses ≤1 block boundary; ensure_pool_rows uploads d_bt[0]
        // once (outside the graph, into the pointer the graph baked) so the replay
        // sees every draft position's physical block. No-op off-pool. Rejected
        // drafts' blocks stay in the slot table and are reused as pos advances.
        // Without this, a post-boundary draft position has no block-table entry
        // (d_bt[0] is 0 there) and aliases physical block 0 - it writes AND reads
        // that same wrong block, so its own logits barely move (self-consistent)
        // but it silently CLOBBERS the sequence's first-block KV: a real
        // corruption bug, just one whose immediate logit signature is tiny.
        // PADDOCK_NO_SPEC_POOL_GROW disables the growth - a test-only hatch to
        // prove it is load-bearing (block table does not grow across the span).
        if paddock_models::dev_var_os!("PADDOCK_NO_SPEC_POOL_GROW").is_none() {
            self.ensure_pool_rows(&slots, &positions)?;
        }
        self.ensure_pf_inputs()?;
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
        {
            let mut v = self
                .d_pf_tok
                .as_mut()
                .expect("pf inputs ensured")
                .slice_mut(0..r);
            exec.stream.memcpy_htod(&tokens, &mut v).map_err(drv)?;
            let mut v = self
                .d_pf_pos
                .as_mut()
                .expect("pf inputs ensured")
                .slice_mut(0..r);
            exec.stream.memcpy_htod(&positions, &mut v).map_err(drv)?;
            let mut v = self
                .d_pf_slots
                .as_mut()
                .expect("pf inputs ensured")
                .slice_mut(0..r);
            exec.stream.memcpy_htod(&slots, &mut v).map_err(drv)?;
        }
        // Numerics pins force eager (P6m lesson). Graphs key by TOTAL row
        // count - the recorded launches read positions/slots from the fixed
        // device buffers, so a 12-row single-slot pass and a 12-row
        // four-slot pass replay the same graph.
        let eager = paddock_models::dev_var_os!("PADDOCK_NO_SPEC_GRAPH").is_some();
        if eager {
            self.record_verify_rows(r)?;
        } else {
            if !self
                .batch
                .as_ref()
                .expect("batch enabled")
                .spec_graphs
                .contains_key(&r)
            {
                self.capture_verify_rows(r)?;
            }
            self.batch.as_ref().expect("batch enabled").spec_graphs[&r]
                .0
                .launch()
                .map_err(|e| GpuError::Driver(format!("verify graph launch: {e}")))?;
        }
        let bs = self.batch.as_ref().ok_or(GpuModelError::BatchDisabled)?;
        let view = bs
            .d_spec_pick
            .try_slice(0..r)
            .ok_or_else(|| GpuError::Driver("spec pick slice out of range".into()))?;
        let picks = exec
            .stream
            .clone_dtoh(&view)
            .map_err(|e| GpuError::Driver(e.to_string()))?;
        Ok(picks)
    }

    /// One single-sequence verify pass (the B=1 spec loop): `chunk` in slot
    /// 0 at `start_pos`. Thin wrapper over [`Self::forward_spec_batch`].
    fn verify_rows(&mut self, chunk: &[u32], start_pos: usize) -> Result<Vec<u32>, GpuModelError> {
        debug_assert!(!chunk.is_empty() && chunk.len() <= SPEC_MAX_ROWS);
        self.forward_spec_batch(&[(0, start_pos, chunk.to_vec())])
    }

    /// Greedy decode with model-free speculative decoding (prompt-lookup /
    /// n-gram drafting - gpt-oss has no MTP head): draft up to `n_draft`
    /// continuation tokens by matching the current 3-gram (2-gram fallback)
    /// suffix against its most recent prior occurrence in the context, verify
    /// the whole chunk in one batched pass, and emit the accepted run plus
    /// the target's own next token. Every emitted token is the verify pass's
    /// own argmax - the draft only changes how many tokens each weight-read
    /// pass yields. Verify rows ride the batch-class kernels (dense nc/mt,
    /// MoE dp4a-batched/mmq), so streams can differ from `generate_greedy`
    /// at knife-edge tokens - the same numeric-class policy as prefill.
    /// When no draft matches, the round BURSTS plain graph-loop steps
    /// (bit-identical to `generate_greedy`) so unmatched text runs at
    /// parity; the draft length adapts to the observed accepted-run length
    /// because a verify pass costs ~2-3 plain steps here (an MoE quirk: r
    /// rows route to far more experts than one row does, so verify traffic
    /// is a multiple of a step's). Wins need verbatim re-emission - agentic
    /// loops, quoting, repetitive code - not just templated output.
    pub fn generate_greedy_spec(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        n_draft: usize,
    ) -> Result<Vec<u32>, GpuModelError> {
        self.moe_bs_pin = true;
        let r = self.generate_greedy_spec_inner(prompt, max_new, n_draft);
        self.moe_bs_pin = false;
        r
    }

    fn generate_greedy_spec_inner(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        n_draft: usize,
    ) -> Result<Vec<u32>, GpuModelError> {
        assert!(!prompt.is_empty() && max_new > 0);
        assert!(
            (1..SPEC_MAX_ROWS).contains(&n_draft),
            "n_draft+1 must fit SPEC_MAX_ROWS"
        );
        assert!(
            prompt.len() + max_new <= self.max_ctx,
            "context {} + {max_new} exceeds max_ctx {}",
            prompt.len(),
            self.max_ctx
        );
        self.reset();
        if self.batch.is_none() {
            self.enable_batch(1)?;
        }
        let logits = self.forward_prefill(0, prompt)?;
        let argmax_h = |l: &[f32]| -> u32 {
            let mut best = 0usize;
            for (i, v) in l.iter().enumerate() {
                if *v > l[best] {
                    best = i;
                }
            }
            best as u32
        };
        let mut out = Vec::with_capacity(max_new + SPEC_MAX_ROWS);
        out.push(argmax_h(&logits));
        let mut pos = prompt.len();
        let mut dr = NgramDraft::default();
        for &t in prompt.iter().chain(out.iter()) {
            dr.push(t);
        }
        let exec = self.exec.clone();
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
        let graph_ok = paddock_models::dev_var_os!("PADDOCK_NO_B1_GRAPH").is_none();
        let debug = paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some();
        let (mut rounds, mut bursts, mut drafted, mut accepted) = (0usize, 0usize, 0usize, 0usize);
        // Adaptive draft length: verify(r) costs ~2-3 plain steps (r rows
        // route to far more experts than one row), so a draft only pays when
        // the accepted run is long. Grow k on full accepts, shrink it to the
        // observed run length on a reject.
        let mut k_now = n_draft.min(4);
        let mut burst_len = 2usize;
        while out.len() < max_new && pos < self.max_ctx {
            let id_last = *out.last().expect("out seeded with the first token");
            // rows write KV at pos..pos+r-1 -> keep the chunk inside max_ctx
            let k_cap = k_now.min(self.max_ctx - pos - 1);
            let drafts = if k_cap == 0 {
                Vec::new()
            } else {
                dr.draft(k_cap)
            };
            if drafts.is_empty() && graph_ok {
                // No match: burst plain graph steps (bit-identical to
                // generate_greedy's loop) instead of paying an r=1 verify
                // round-trip per token; the drafter keeps ingesting.
                let burst = burst_len.min(self.max_ctx - pos).min(max_new - out.len());
                burst_len = (burst_len * 2).min(SPEC_BURST_MAX);
                if self.d_g_pos.is_none() {
                    self.d_g_pos = Some(exec.alloc_u32(1)?);
                }
                {
                    let bs = self.batch.as_mut().ok_or(GpuModelError::BatchDisabled)?;
                    exec.stream
                        .memcpy_htod(&[id_last], &mut bs.d_g_token)
                        .map_err(drv)?;
                    exec.stream
                        .memcpy_htod(&[0u32], &mut bs.d_g_step)
                        .map_err(drv)?;
                    let d_pos = self.d_g_pos.as_mut().expect("d_g_pos allocated above");
                    exec.stream.memcpy_htod(&[pos as u32], d_pos).map_err(drv)?;
                }
                if self
                    .batch
                    .as_ref()
                    .expect("batch enabled")
                    .gen_graph
                    .is_none()
                {
                    self.capture_gen_graph()?;
                }
                {
                    let g = self
                        .batch
                        .as_ref()
                        .expect("batch enabled")
                        .gen_graph
                        .as_ref()
                        .expect("gen graph captured above");
                    for _ in 0..burst {
                        g.0.launch()
                            .map_err(|x| GpuError::Driver(format!("gen launch: {x}")))?;
                    }
                }
                let ids = exec.to_host_u32(&self.batch.as_ref().expect("batch enabled").d_g_out)?;
                for &t in ids.iter().take(burst) {
                    out.push(t);
                    dr.push(t);
                }
                pos += burst;
                bursts += 1;
                continue;
            }
            let mut chunk = Vec::with_capacity(drafts.len() + 1);
            chunk.push(id_last);
            chunk.extend_from_slice(&drafts);
            let targets = self.verify_rows(&chunk, pos)?;
            // accept drafts while they match the target's own next token
            let mut a = 0usize;
            while a < drafts.len() && drafts[a] == targets[a] {
                a += 1;
            }
            let committed = a + 1; // chunk rows 0..=a become context
            rounds += 1;
            drafted += drafts.len();
            accepted += a;
            if !drafts.is_empty() {
                if a == drafts.len() {
                    k_now = (k_now * 2).min(n_draft);
                } else {
                    k_now = (a + 1).clamp(2, k_now);
                }
                burst_len = 2; // a live pattern: re-check the drafter quickly
            }
            pos += committed;
            for &t in targets.iter().take(committed) {
                out.push(t);
                dr.push(t);
            }
        }
        out.truncate(max_new);
        self.pos = pos;
        if debug {
            tracing::info!(
                "spec k={n_draft}: {} tokens in {rounds} verify rounds + {bursts} bursts, drafted {drafted} accepted {accepted} ({:.0}%)",
                out.len(),
                100.0 * accepted as f64 / drafted.max(1) as f64
            );
        }
        Ok(out)
    }

    /// Prefill a whole prompt into `slot`'s KV cache in one (chunked) pass and
    /// return only the last token's next-token logits. Each chunk of up to
    /// PREFILL_CHUNK tokens runs as a batched pass mapped onto `slot`, attending
    /// causally (ascending positions) to all KV written so far - turning a P-token
    /// prompt from P decode steps into ceil(P / PREFILL_CHUNK) passes. The engine's
    /// TTFT path.
    pub fn forward_prefill(
        &mut self,
        slot: usize,
        tokens: &[u32],
    ) -> Result<Vec<f32>, GpuModelError> {
        let max_batch = match &self.batch {
            None => return Err(GpuModelError::BatchDisabled),
            Some(bs) => bs.max_batch,
        };
        if slot >= max_batch {
            return Err(GpuModelError::BatchTooLarge {
                got: slot + 1,
                max: max_batch,
            });
        }
        if tokens.is_empty() || tokens.len() > self.max_ctx {
            return Err(GpuModelError::BatchTooLarge {
                got: tokens.len(),
                max: self.max_ctx,
            });
        }

        // Prefix reuse: adopt the longest cached prefix zero-copy from the
        // paged radix (share blocks + restore the SWA-window checkpoint), then
        // prefill only the divergent remainder and store this prompt's pages.
        // This serial path previously knew only the dense RadixKvCache, so
        // pool-mode serves got zero reuse here - since the dense-lane nuke
        // it rides the same paged resume as the batch/chunked
        // paths; dense A/B mode prefills cold.
        let l = if self.pool_active() {
            self.paged_prefix_resume(slot, tokens)?
        } else {
            0
        };
        self.last_reused[slot] = l;
        let logits = self.run_prefill(slot, &tokens[l..], l)?;
        if self.pool_active() {
            self.paged_prefix_store(slot, tokens)?;
        }
        Ok(logits)
    }

    /// Cumulative prefix-cache pages reused (test/telemetry hook).
    pub fn prefix_cache_reused_blocks(&self) -> u64 {
        self.paged_reused_blocks
    }

    /// Begin a CHUNKED prefill on `slot` (vLLM-class continuous batching):
    /// match + load the longest cached prefix now; the remainder advances up
    /// to a budget of rows per [`Self::forward_mixed`] tick, riding one pass
    /// with the live decode rows instead of stalling them. One prompt chunks
    /// at a time - the service queues the rest. (Measured motivation: the
    /// blocking batched prefill froze 8 live streams ~3.8 s per 8x4k-token
    /// admission wave.)
    pub fn prefill_begin(&mut self, slot: usize, tokens: Vec<u32>) -> Result<(), GpuModelError> {
        let max_batch = self
            .batch
            .as_ref()
            .map(|b| b.max_batch)
            .ok_or(GpuModelError::BatchDisabled)?;
        if slot >= max_batch {
            return Err(GpuModelError::BatchTooLarge {
                got: slot + 1,
                max: max_batch,
            });
        }
        if tokens.is_empty() || tokens.len() > self.max_ctx {
            return Err(GpuModelError::BatchTooLarge {
                got: tokens.len(),
                max: self.max_ctx,
            });
        }
        if self.chunked.len() >= MAX_CHUNKS {
            return Err(GpuModelError::Unsupported(
                "chunked prefill queue is full".into(),
            ));
        }
        if self.chunked.iter().any(|c| c.slot == slot) {
            return Err(GpuModelError::Unsupported(
                "slot already has a chunked prefill in flight".into(),
            ));
        }
        let l = if self.pool_active() {
            // P5c zero-copy paged reuse: share the full-attn prefix blocks +
            // restore the SWA window, then chunk only from `done = pos`.
            self.paged_prefix_resume(slot, &tokens)?
        } else {
            // dense A/B mode: no prefix cache since the RadixKvCache nuke -
            // prefill cold
            0
        };
        self.last_reused[slot] = l;
        self.chunked.push(ChunkedPrefill {
            slot,
            tokens,
            done: l,
        });
        Ok(())
    }

    /// `Generator::prefill_queue`: `forward_mixed_core` fills its budget FIFO
    /// over `chunked`, from each prompt's `done`.
    pub(crate) fn prefill_queue(&self) -> Vec<(usize, usize, usize)> {
        self.chunked
            .iter()
            .map(|c| (c.slot, c.done, c.tokens.len() - c.done))
            .collect()
    }

    /// `Generator::prefill_tick_cap`: the pass's row cap less its decode rows
    /// (`forward_mixed_core`'s own room).
    pub(crate) fn prefill_tick_cap(&self, decode_rows: usize) -> usize {
        self.batch
            .as_ref()
            .map_or(0, |bs| bs.row_cap.saturating_sub(decode_rows))
    }

    /// One MIXED continuous-batching tick: every live decode row plus up to
    /// `budget` rows of the in-flight chunked prompt run as one
    /// weight-amortized pass. The decode rows form one decode-class attention
    /// group; the chunk rides the f16 WMMA prefill kernel on its slot (both
    /// via run_layers' per-group dispatch). `decodes[i] = (slot, token, pos)`.
    ///
    /// Returns the decode rows' logits (flat [nd, vocab], input order) and,
    /// when the chunk finishes its prompt this tick, `Some((slot, last-token
    /// logits, prompt rows))` - the slot's first sample, ready for the
    /// service's finish-prefill path.
    pub fn forward_mixed(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
    ) -> Result<(Vec<f32>, Vec<(usize, Vec<f32>, usize)>), GpuModelError> {
        let (nd, shares) = self.forward_mixed_core(decodes, budget)?;
        let n_fin = shares.iter().filter(|s| s.2).count();
        let n_emit = nd + n_fin;
        let exec = self.exec.clone();
        let vocab = self.vocab;
        let flat = {
            let bs = self.batch.as_mut().ok_or(GpuModelError::BatchDisabled)?;
            exec.to_host_len(&bs.d_logits, n_emit * vocab)?
        };
        let mut decode_logits = flat;
        // peel the finishing rows off the tail, restoring chunk order
        let mut fin_rows: Vec<Vec<f32>> = Vec::with_capacity(n_fin);
        for i in (0..n_fin).rev() {
            fin_rows.push(decode_logits.split_off((nd + i) * vocab));
        }
        fin_rows.reverse();
        let finished = self.mixed_finish(&shares, fin_rows)?;
        Ok((decode_logits, finished))
    }

    /// `forward_mixed` + fused device sampling for the decode rows - the
    /// mixed-tick twin of [`Self::forward_batch_sampled`]. Under concurrent
    /// load most ticks are mixed (arrivals keep a chunk in flight), and the
    /// unsampled path reads back `[nd, vocab]` f32 logits every tick
    /// (25.7 MB at nd=32) and samples on host - worth a ~7 ms GPU-idle gap
    /// per tick at c32 on a GB202, sitting between the lm-head GEMM and the
    /// next tick's embed gather. Eligible
    /// rows return bare token ids; `Host` rows their own logits row; the
    /// finishing chunk's last row still comes back whole (its slot's first
    /// token samples on host in finish-prefill, once per request).
    pub fn forward_mixed_sampled(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
        plans: &[crate::generator::RowSample],
    ) -> Result<(crate::generator::SampledStep, Vec<(usize, Vec<f32>, usize)>), GpuModelError> {
        use crate::generator::{RowSample, SampledStep};
        use crate::sampler::DevicePlan;
        assert_eq!(plans.len(), decodes.len(), "one plan per decode row");
        let (nd, shares) = self.forward_mixed_core(decodes, budget)?;
        let n_fin = shares.iter().filter(|s| s.2).count();
        let exec = self.exec.clone();
        let vocab = self.vocab;
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
        let (ids, host_rows, fin_rows) = {
            let bs = self.batch.as_mut().ok_or(GpuModelError::BatchDisabled)?;
            // pack per-row sampler params exactly like the dense step; rows
            // are the first nd of d_logits by the core's layout. Upload is
            // stream-ordered ahead of sample_rows, so racing the forward
            // kernels already on the stream is fine.
            let ids = if nd > 0 {
                let mut par = vec![0u32; nd * 4];
                let mut tpar = vec![0u32; nd * 4];
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
                        // RS plans are gemma4-only (supports_spec_rs)
                        RowSample::Device(DevicePlan::RsVerify { .. })
                        | RowSample::Device(DevicePlan::RsTrunc { .. }) => {}
                    }
                }
                let mut v = bs.d_samp_par.slice_mut(0..nd * 4);
                exec.stream.memcpy_htod(&par, &mut v).map_err(drv)?;
                if any_trunc {
                    let mut v = bs.d_samp_tpar.slice_mut(0..nd * 4);
                    exec.stream.memcpy_htod(&tpar, &mut v).map_err(drv)?;
                }
                exec.sample_rows(&bs.d_logits, &bs.d_samp_par, &mut bs.d_samp_out, nd, vocab)?;
                if any_trunc {
                    Self::trunc_dev_witness(nd);
                    exec.sample_rows_t(
                        &bs.d_logits,
                        &bs.d_samp_par,
                        &bs.d_samp_tpar,
                        &mut bs.d_samp_out,
                        nd,
                        vocab,
                    )?;
                    exec.sample_rows_p(
                        &bs.d_logits,
                        &bs.d_samp_par,
                        &bs.d_samp_tpar,
                        &mut bs.d_samp_out,
                        nd,
                        vocab,
                    )?;
                }
                let ids_view = bs
                    .d_samp_out
                    .try_slice(0..nd)
                    .ok_or_else(|| GpuError::Driver("samp_out slice out of range".into()))?;
                exec.stream.clone_dtoh(&ids_view).map_err(drv)?
            } else {
                Vec::new()
            };
            let mut host_rows = Vec::new();
            for (i, p) in plans.iter().enumerate() {
                if matches!(p, RowSample::Host) {
                    let view = bs
                        .d_logits
                        .try_slice(i * vocab..(i + 1) * vocab)
                        .ok_or_else(|| GpuError::Driver("logits row slice out of range".into()))?;
                    host_rows.push((i, exec.stream.clone_dtoh(&view).map_err(drv)?));
                }
            }
            let mut fin_rows: Vec<Vec<f32>> = Vec::with_capacity(n_fin);
            for i in 0..n_fin {
                let view = bs
                    .d_logits
                    .try_slice((nd + i) * vocab..(nd + i + 1) * vocab)
                    .ok_or_else(|| GpuError::Driver("finish row slice out of range".into()))?;
                fin_rows.push(exec.stream.clone_dtoh(&view).map_err(drv)?);
            }
            (ids, host_rows, fin_rows)
        };
        let finished = self.mixed_finish(&shares, fin_rows)?;
        Ok((SampledStep { ids, host_rows }, finished))
    }

    /// Chunk bookkeeping shared by both mixed tails: consume every chunk
    /// that finished this tick (prefix-cache insert + first-sample handoff,
    /// in chunk order - matching `fin_rows`) and advance the rest.
    fn mixed_finish(
        &mut self,
        shares: &[(usize, usize, bool)],
        fin_rows: Vec<Vec<f32>>,
    ) -> Result<Vec<(usize, Vec<f32>, usize)>, GpuModelError> {
        let mut out = Vec::with_capacity(fin_rows.len());
        let mut fin_iter = fin_rows.into_iter();
        let mut remove: Vec<usize> = Vec::new();
        for &(ci, rows, finishing) in shares {
            if finishing {
                let (slot, tokens) = {
                    let ch = &mut self.chunked[ci];
                    (ch.slot, std::mem::take(&mut ch.tokens))
                };
                if self.pool_active() {
                    self.paged_prefix_store(slot, &tokens)?;
                }
                out.push((
                    slot,
                    fin_iter.next().expect("one row per finisher"),
                    tokens.len(),
                ));
                remove.push(ci);
            } else {
                self.chunked[ci].done += rows;
            }
        }
        for &ci in remove.iter().rev() {
            self.chunked.remove(ci);
        }
        Ok(out)
    }

    /// The shared mixed-tick body: one weight-amortized pass over the decode
    /// rows + the next rows of every in-flight chunked prompt (FIFO-filled up
    /// to the row budget), through the lm_head - logits left on device in
    /// `d_logits` (decode rows first, then one finishing row per chunk that
    /// completed, in chunk order). Returns `(nd, shares)` where `shares[i] =
    /// (chunk index, rows taken, finished)`; chunk bookkeeping happens in the
    /// tails after their readbacks via [`Self::mixed_finish`].
    fn forward_mixed_core(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
    ) -> Result<(usize, Vec<(usize, usize, bool)>), GpuModelError> {
        let exec = self.exec.clone();
        let (embd, rms_eps, _vocab) = (self.embd, self.rms_eps, self.vocab);
        let (max_batch, cap) = match &self.batch {
            None => return Err(GpuModelError::BatchDisabled),
            Some(bs) => (bs.max_batch, bs.row_cap),
        };
        let nd = decodes.len();
        for &(k, _, pos) in decodes {
            if k >= max_batch {
                return Err(GpuModelError::BatchTooLarge {
                    got: k + 1,
                    max: max_batch,
                });
            }
            if pos as usize >= self.max_ctx {
                return Err(GpuModelError::BatchTooLarge {
                    got: pos as usize + 1,
                    max: self.max_ctx,
                });
            }
        }
        if self.chunked.is_empty() {
            return Err(GpuModelError::Unsupported(
                "forward_mixed without prefill_begin".into(),
            ));
        }
        // FIFO fill: earlier admissions finish first (first token sooner);
        // a chunk that gets 0 rows this tick just waits for the next one
        let mut room = budget.min(cap.saturating_sub(nd));
        let mut shares: Vec<(usize, usize, bool)> = Vec::new();
        for (ci, ch) in self.chunked.iter().enumerate() {
            if room == 0 {
                break;
            }
            let left = ch.tokens.len() - ch.done;
            let take = left.min(room);
            room -= take;
            shares.push((ci, take, ch.done + take == ch.tokens.len()));
        }
        if shares.iter().all(|s| s.1 == 0) {
            return Err(GpuModelError::BatchTooLarge {
                got: nd + 1,
                max: cap,
            });
        }
        shares.retain(|s| s.1 > 0);
        let c_rows: usize = shares.iter().map(|s| s.1).sum();

        // rows: decode rows first (one decode-class attention group), then
        // each chunk's rows (a WMMA prefill group per chunk slot)
        let b = nd + c_rows;
        let mut toks: Vec<u32> = Vec::with_capacity(b);
        let mut pos: Vec<u32> = Vec::with_capacity(b);
        let mut slt: Vec<u32> = Vec::with_capacity(b);
        for &(k, t, p) in decodes {
            toks.push(t);
            pos.push(p);
            slt.push(k as u32);
        }
        // decode rows span many slots - never WMMA-eligible; each chunk is
        // one slot's tail and may ride it
        let mut groups: Vec<(usize, usize, bool)> = Vec::new();
        if nd > 0 {
            groups.push((0, nd, false));
        }
        let mut start = nd;
        for &(ci, take, _) in &shares {
            let ch = &self.chunked[ci];
            for o in 0..take {
                toks.push(ch.tokens[ch.done + o]);
                pos.push((ch.done + o) as u32);
                slt.push(ch.slot as u32);
            }
            groups.push((start, take, true));
            start += take;
        }

        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
        // G4a: grow the pool for every row (decode + chunk) this mixed tick
        // touches, before the append/read kernels run.
        self.ensure_pool_rows(&slt, &pos)?;
        let d_pos = exec.stream.clone_htod(&pos).map_err(drv)?;
        let d_slots = exec.stream.clone_htod(&slt).map_err(drv)?;
        self.embed_gather(&toks)?;
        let max_pos = pos.iter().copied().max().unwrap_or(0) as usize;
        self.run_layers(b, &d_pos, Some(&d_slots), max_pos, false, Some(&groups))?;

        // lm_head: every decode row, plus each finishing chunk's last row.
        // Decode rows are the first nd rows, so one contiguous region copy
        // stages them; n_emit <= max_batch always (chunking slots never
        // decode), which d_logits is sized for.
        let n_fin = shares.iter().filter(|sh| sh.2).count();
        let n_emit = nd + n_fin;
        let bs = self.batch.as_mut().ok_or(GpuModelError::BatchDisabled)?;
        exec.rmsnorm_batch(&bs.d_x, &self.out_norm.buf, &mut bs.d_xn, embd, rms_eps, b)?;
        if nd > 0 {
            exec.copy_region(&bs.d_xn, 0, &mut bs.d_x, 0, nd * embd)?;
        }
        let mut emit = nd;
        let mut row_base = nd;
        for &(_, take, finishing) in &shares {
            if finishing {
                exec.copy_region(
                    &bs.d_xn,
                    (row_base + take - 1) * embd,
                    &mut bs.d_x,
                    emit * embd,
                    embd,
                )?;
                emit += 1;
            }
            row_base += take;
        }
        if n_emit <= 64 {
            // same int8 TC lm_head class as the batched decode step
            exec.quantize_q8(&bs.d_x, &mut bs.d_p_xq, &mut bs.d_p_xs, n_emit * embd)?;
            exec.q8_0_gemm_mma(
                &self.output_r,
                &bs.d_p_xq,
                &bs.d_p_xs,
                &mut bs.d_logits,
                n_emit,
            )?;
        } else {
            exec.q8_0_gemm(
                self.output.as_ref().expect(RAW_HEAD_DROPPED),
                None,
                &bs.d_x,
                &mut bs.d_logits,
                n_emit,
            )?;
        }
        Ok((nd, shares))
    }

    /// Prefill SEVERAL prompts in one shot: each `(slot, tokens)` reuses its cached
    /// prefix, then all the divergent tails are concatenated into one batched pass
    /// (rows mapped to their slots/positions) - so the model weights are read once
    /// for the whole set instead of once per sequence. Returns each prompt's last-
    /// token logits in input order. Tails that overflow the row cap are processed
    /// standalone; the rest pack into cap-sized batched passes. This is the phase-1
    /// scheduler path when several requests arrive together.
    pub fn forward_prefill_batch(
        &mut self,
        items: &[(usize, Vec<u32>)],
    ) -> Result<Vec<Vec<f32>>, GpuModelError> {
        let (max_batch, cap) = match &self.batch {
            None => return Err(GpuModelError::BatchDisabled),
            Some(bs) => (bs.max_batch, bs.row_cap),
        };
        // match + load each prompt's cached prefix; keep (slot, tail, start_pos)
        let mut prep: Vec<(usize, Vec<u32>, usize)> = Vec::with_capacity(items.len());
        for (slot, tokens) in items {
            if *slot >= max_batch {
                return Err(GpuModelError::BatchTooLarge {
                    got: slot + 1,
                    max: max_batch,
                });
            }
            if tokens.is_empty() || tokens.len() > self.max_ctx {
                return Err(GpuModelError::BatchTooLarge {
                    got: tokens.len(),
                    max: self.max_ctx,
                });
            }
            let l = if self.pool_active() {
                // P5c zero-copy paged reuse (full-attn share + SWA-window restore)
                self.paged_prefix_resume(*slot, tokens)?
            } else {
                // dense A/B mode: no prefix cache
                0
            };
            self.last_reused[*slot] = l;
            prep.push((*slot, tokens[l..].to_vec(), l));
        }

        // pack tails into cap-sized batched passes (long tails go standalone)
        let mut out: Vec<Vec<f32>> = vec![Vec::new(); items.len()];
        let mut group: Vec<usize> = Vec::new();
        let mut rows = 0usize;
        for i in 0..prep.len() {
            let tl = prep[i].1.len();
            if tl > cap {
                if !group.is_empty() {
                    self.prefill_pass(&prep, &group, &mut out)?;
                    group.clear();
                    rows = 0;
                }
                out[i] = self.run_prefill(prep[i].0, &prep[i].1.clone(), prep[i].2)?;
            } else {
                if rows + tl > cap {
                    self.prefill_pass(&prep, &group, &mut out)?;
                    group.clear();
                    rows = 0;
                }
                group.push(i);
                rows += tl;
            }
        }
        if !group.is_empty() {
            self.prefill_pass(&prep, &group, &mut out)?;
        }

        // cache each full prompt now that its slot holds the complete KV
        for (slot, tokens) in items {
            if self.pool_active() {
                self.paged_prefix_store(*slot, tokens)?;
            }
        }
        Ok(out)
    }

    /// One batched-tail pass over `group` (indices into `prep`): concatenate the
    /// tails, run the stack with the per-row slot map, then lm_head each sequence's
    /// last row. Writes each result into `out` at its original index.
    fn prefill_pass(
        &mut self,
        prep: &[(usize, Vec<u32>, usize)],
        group: &[usize],
        out: &mut [Vec<f32>],
    ) -> Result<(), GpuModelError> {
        if group.is_empty() {
            return Ok(());
        }
        let exec = self.exec.clone();
        let (embd, rms_eps, vocab) = (self.embd, self.rms_eps, self.vocab);
        let mut toks: Vec<u32> = Vec::new();
        let mut pos: Vec<u32> = Vec::new();
        let mut slt: Vec<u32> = Vec::new();
        let mut last_rows: Vec<(usize, usize)> = Vec::new(); // (out index, row in batch)
        // per-slot (start, count, single_slot=true): every group here is one
        // prompt's tail on one slot, so each may ride the WMMA kernel
        let mut groups: Vec<(usize, usize, bool)> = Vec::new();
        for &i in group {
            let (slot, tail, start) = &prep[i];
            groups.push((toks.len(), tail.len(), true));
            for (o, &t) in tail.iter().enumerate() {
                toks.push(t);
                pos.push((start + o) as u32);
                slt.push(*slot as u32);
            }
            last_rows.push((i, toks.len() - 1));
        }
        let b = toks.len();
        // G4a: grow the pool for every (slot, pos) in this batched prefill pass.
        self.ensure_pool_rows(&slt, &pos)?;
        let d_pos = exec
            .stream
            .clone_htod(&pos)
            .map_err(|e| GpuError::Driver(e.to_string()))?;
        let d_slots = exec
            .stream
            .clone_htod(&slt)
            .map_err(|e| GpuError::Driver(e.to_string()))?;
        self.embed_gather(&toks)?;
        let max_pos = pos.iter().copied().max().unwrap_or(0) as usize;
        // rows span different slots -> per-group attention dispatch (each big
        // tail rides the f16 WMMA kernel on its own slot)
        self.run_layers(b, &d_pos, Some(&d_slots), max_pos, false, Some(&groups))?;

        let n = group.len();
        let bs = self.batch.as_mut().ok_or(GpuModelError::BatchDisabled)?;
        exec.rmsnorm_batch(&bs.d_x, &self.out_norm.buf, &mut bs.d_xn, embd, rms_eps, b)?;
        // gather each sequence's last hidden row to the front, then one lm_head GEMM
        for (gi, &(_, row)) in last_rows.iter().enumerate() {
            exec.copy_region(&bs.d_xn, row * embd, &mut bs.d_x, gi * embd, embd)?;
        }
        if n <= 64 {
            // same int8 TC lm_head class as the batched decode step
            exec.quantize_q8(&bs.d_x, &mut bs.d_p_xq, &mut bs.d_p_xs, n * embd)?;
            exec.q8_0_gemm_mma(&self.output_r, &bs.d_p_xq, &bs.d_p_xs, &mut bs.d_logits, n)?;
        } else {
            exec.q8_0_gemm(
                self.output.as_ref().expect(RAW_HEAD_DROPPED),
                None,
                &bs.d_x,
                &mut bs.d_logits,
                n,
            )?;
        }
        let logits = exec.to_host(&bs.d_logits)?;
        for (gi, &(oi, _)) in last_rows.iter().enumerate() {
            out[oi] = logits[gi * vocab..(gi + 1) * vocab].to_vec();
        }
        Ok(())
    }

    /// Ensure the fixed-address graph-input buffers (`d_pf_*`) exist at row_cap
    /// size. (Re)allocating moves the device addresses, so every graph cache
    /// that baked them (prefill chunks AND decode steps) is invalidated.
    fn ensure_pf_inputs(&mut self) -> Result<(), GpuModelError> {
        let cap = self
            .batch
            .as_ref()
            .ok_or(GpuModelError::BatchDisabled)?
            .row_cap;
        if self.d_pf_tok.as_ref().is_none_or(|b| b.len() < cap) {
            self.d_pf_tok = Some(self.exec.alloc_u32(cap)?);
            self.d_pf_pos = Some(self.exec.alloc_u32(cap)?);
            self.d_pf_slots = Some(self.exec.alloc_u32(cap)?);
            if let Some(bs) = self.batch.as_mut() {
                bs.pf_graphs.clear();
                bs.step_graphs.clear();
                bs.spec_graphs.clear();
            }
        }
        Ok(())
    }

    /// Prefill `tokens` into `slot` starting at KV position `start_pos` (so the
    /// prompt attends causally to whatever KV already sits at [0, start_pos) - a
    /// loaded cached prefix). Chunked; returns the last token's logits.
    fn run_prefill(
        &mut self,
        slot: usize,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<Vec<f32>, GpuModelError> {
        let exec = self.exec.clone();
        let (embd, rms_eps, vocab) = (self.embd, self.rms_eps, self.vocab);
        let cap = self
            .batch
            .as_ref()
            .ok_or(GpuModelError::BatchDisabled)?
            .row_cap;
        self.ensure_pf_inputs()?;
        // Numerics pins force eager - a cached graph baked the record-time
        // dispatch and would silently ignore the env (P6m lesson).
        let eager = paddock_models::dev_var_os!("PADDOCK_NO_PREFILL_GRAPH").is_some();
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
        // G4a: a fresh prefill (start_pos==0) reuses the slot - return its old
        // pool blocks before regrowing from zero, so a slot recycled without a
        // free-on-completion tick doesn't leak (the scheduler's release runs
        // too, so this is belt-and-suspenders + covers direct forward_prefill).
        if start_pos == 0 {
            self.pool_clear_slot(slot);
        }
        let total = tokens.len();
        let mut out = Vec::new();
        let mut start = 0usize;
        while start < total {
            let cs = (total - start).min(cap);
            let positions: Vec<u32> = (start..start + cs)
                .map(|p| (start_pos + p) as u32)
                .collect();
            let slots_v: Vec<u32> = vec![slot as u32; cs];
            // G4a: grow the pool for this chunk's positions before its append.
            self.ensure_pool_rows(&slots_v, &positions)?;
            // per-chunk inputs land in the fixed buffers, outside any graph -
            // only their contents change between replays
            {
                let mut v = self
                    .d_pf_tok
                    .as_mut()
                    .expect("pf inputs ensured")
                    .slice_mut(0..cs);
                exec.stream
                    .memcpy_htod(&tokens[start..start + cs], &mut v)
                    .map_err(drv)?;
                let mut v = self
                    .d_pf_pos
                    .as_mut()
                    .expect("pf inputs ensured")
                    .slice_mut(0..cs);
                exec.stream.memcpy_htod(&positions, &mut v).map_err(drv)?;
                let mut v = self
                    .d_pf_slots
                    .as_mut()
                    .expect("pf inputs ensured")
                    .slice_mut(0..cs);
                exec.stream.memcpy_htod(&slots_v, &mut v).map_err(drv)?;
            }
            if eager {
                self.record_prefill_chunk(cs)?;
            } else {
                if !self
                    .batch
                    .as_ref()
                    .expect("batch enabled")
                    .pf_graphs
                    .contains_key(&cs)
                {
                    self.capture_prefill_chunk(cs)?;
                }
                self.batch.as_ref().expect("batch enabled").pf_graphs[&cs]
                    .0
                    .launch()
                    .map_err(|e| GpuError::Driver(format!("prefill graph launch: {e}")))?;
            }

            if start + cs >= total {
                // last chunk: norm every row (cheap), then lm_head on the last row
                // only (copy it to the front of d_x scratch first - avoids a full
                // cs-row lm_head GEMM and a cs×vocab readback).
                let bs = self.batch.as_mut().ok_or(GpuModelError::BatchDisabled)?;
                exec.rmsnorm_batch(&bs.d_x, &self.out_norm.buf, &mut bs.d_xn, embd, rms_eps, cs)?;
                exec.copy_slice(&bs.d_xn, (cs - 1) * embd, embd, &mut bs.d_x)?;
                {
                    // single-row lm_head: same dp4a nc class the decode step uses
                    exec.quantize_q8(&bs.d_x, &mut bs.d_b1_xq, &mut bs.d_b1_xs, embd)?;
                    exec.q8_0_gemv_dp4a_nc(
                        &self.output_r,
                        &bs.d_b1_xq,
                        &bs.d_b1_xs,
                        &mut bs.d_logits,
                        1,
                    )?;
                }
                let logits = exec.to_host(&bs.d_logits)?;
                out = logits[..vocab].to_vec();
            }
            start += cs;
        }
        Ok(out)
    }

    /// Record one prefill chunk pass (embed gather -> all layers) onto the
    /// stream - the body shared by the eager path and graph capture. Every
    /// input reads a fixed-address device buffer (`d_pf_*`), so a capture of
    /// these launches replays correctly for any chunk of the same size: the
    /// attention kernels derive their ranges from the device positions, and
    /// the split-count heuristic is position-independent (G0), so no launch
    /// geometry depends on where in the prompt the chunk sits.
    fn record_prefill_chunk(&mut self, cs: usize) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let embd = self.embd;
        // move the fixed input buffers out around the `&mut self` run_layers
        // call (host-side move - device addresses unchanged; the d_g_pos trick)
        let d_tok = self.d_pf_tok.take().expect("pf buffers");
        let d_pos = self.d_pf_pos.take().expect("pf buffers");
        let d_slots = self.d_pf_slots.take().expect("pf buffers");
        let r = (|| -> Result<(), GpuModelError> {
            {
                let bs = self.batch.as_mut().ok_or(GpuModelError::BatchDisabled)?;
                exec.embed_gather_batch_q8(&self.tok_embd, &d_tok, &mut bs.d_x, embd, cs)?;
            }
            // max_pos only feeds the (position-independent) split heuristic
            self.run_layers(cs, &d_pos, Some(&d_slots), self.max_ctx - 1, true, None)
        })();
        self.d_pf_tok = Some(d_tok);
        self.d_pf_pos = Some(d_pos);
        self.d_pf_slots = Some(d_slots);
        r
    }

    /// Capture [`Self::record_prefill_chunk`] into a replayable graph, cached
    /// per chunk size. Same contract as `capture_gen_graph`: the capture only
    /// records; the caller launches.
    fn capture_prefill_chunk(&mut self, cs: usize) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        exec.stream
            .synchronize()
            .map_err(|e| GpuError::Driver(format!("pre-capture sync: {e}")))?;
        exec.stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|e| GpuError::Driver(format!("begin_capture: {e}")))?;
        let rec = self.record_prefill_chunk(cs);
        let graph = crate::gpu::end_capture_no_flags(&exec.stream)
            .map_err(|e| GpuError::Driver(format!("end_capture: {e}")));
        rec?; // surface a record failure only after capture is cleanly ended
        let graph =
            graph?.ok_or_else(|| GpuError::Driver("prefill capture produced no graph".into()))?;
        let bs = self.batch.as_mut().ok_or(GpuModelError::BatchDisabled)?;
        // serving traffic can produce many distinct tail sizes - bound the cache
        if bs.pf_graphs.len() >= 16 {
            bs.pf_graphs.clear();
        }
        bs.pf_graphs.insert(cs, SendGraph(graph));
        Ok(())
    }

    /// P5c: build the batched-copy descriptors linking a slot's SWA ring blocks
    /// for the window `[pos-swa_window, pos)` to checkpoint slot `idx` in
    /// `d_swa_ckpt`. `snapshot` picks the direction (true: ring->ckpt). `pos` must
    /// be a block multiple ≥ swa_window. Ckpt layout: [swa_layer][K,V][win block]
    /// in logical (ascending) order. The SWA ring maps logical block `jb` of slot
    /// `s` to physical `s*swa_ring + jb%swa_ring` (the G3 WindowRing formula).
    fn swa_ckpt_descs(&self, slot: usize, idx: u32, pos: usize, snapshot: bool) -> Vec<u64> {
        let exec = &self.exec;
        let kv_dim = self.n_kv_heads * self.head_dim;
        let kv_bytes = self.kv_dtype.bytes();
        let blk_bytes = (16 * kv_dim * kv_bytes) as u64;
        let bs = self.batch.as_ref().expect("batch enabled");
        let swa_ring = bs.swa_ring_blocks;
        let win_blocks = self.swa_window / 16;
        let first_jb = pos / 16 - win_blocks; // pos is block-aligned, pos>=swa_window
        let (cp, _g) = bs
            .d_swa_ckpt
            .as_ref()
            .expect("paged prefix cache built")
            .device_ptr(&exec.stream);
        let ckpt_base = cp + idx as u64 * bs.swa_ckpt_bytes as u64;
        let mut descs: Vec<u64> = Vec::new();
        let mut sl = 0usize; // sequential SWA-layer index
        for li in 0..self.n_layers {
            if !self.layers[li].is_swa {
                continue;
            }
            for (kv_flag, cache) in [&bs.k_cache[li], &bs.v_cache[li]].into_iter().enumerate() {
                let (base, _g2) = cache.device_ptr(&exec.stream);
                for b in 0..win_blocks {
                    let jb = first_jb + b;
                    let phys = slot * swa_ring + jb % swa_ring;
                    let ring_addr = base + phys as u64 * blk_bytes;
                    let ckpt_addr =
                        ckpt_base + ((sl * 2 + kv_flag) * win_blocks + b) as u64 * blk_bytes;
                    let (src, dst) = if snapshot {
                        (ring_addr, ckpt_addr)
                    } else {
                        (ckpt_addr, ring_addr)
                    };
                    descs.extend([src, dst, blk_bytes]);
                }
            }
            sl += 1;
        }
        descs
    }

    /// P5c: snapshot slot's trailing SWA window KV into checkpoint `idx` (after a
    /// prefill so the ring holds `[pos-swa_window, pos)`), or restore it into the
    /// resuming slot's ring (before the suffix prefill). `pos` block-aligned, ≥ swa_window.
    fn swa_window_copy(
        &mut self,
        slot: usize,
        idx: u32,
        pos: usize,
        snapshot: bool,
    ) -> Result<(), GpuModelError> {
        if self
            .batch
            .as_ref()
            .and_then(|b| b.d_swa_ckpt.as_ref())
            .is_none()
        {
            return Ok(());
        }
        let descs = self.swa_ckpt_descs(slot, idx, pos, snapshot);
        let n = descs.len() / 3;
        let exec = self.exec.clone();
        let d = exec
            .stream
            .clone_htod(&descs)
            .map_err(|e| GpuError::Driver(e.to_string()))?;
        exec.batched_copy(&d, n)?;
        Ok(())
    }

    /// P5c: resume `slot` on the longest cached prefix of `tokens` that carries an
    /// SWA checkpoint. Zero-copy adopts the full-attn prefix blocks
    /// (`share_prefix`) and restores the trailing SWA window into the ring, so the
    /// caller prefills only `tokens[pos..]` (attending the shared prefix exactly).
    /// Returns the resume position `pos` (0 = no reuse). Pool mode only.
    fn paged_prefix_resume(&mut self, slot: usize, tokens: &[u32]) -> Result<usize, GpuModelError> {
        let swa_window = self.swa_window;
        let m = {
            let bs = self.batch.as_mut().ok_or(GpuModelError::BatchDisabled)?;
            match bs.paged_prefix.as_mut() {
                Some(pr) => pr.match_full(tokens),
                None => return Ok(0),
            }
        };
        if paddock_models::dev_var_os!("PADDOCK_POOL_STATS").is_some() {
            tracing::info!(
                "p5c: resume-try slot {slot} tokens {} matched_blocks {} ckpt {:?}",
                tokens.len(),
                m.blocks.len(),
                m.ckpt
            );
        }
        let Some((pos, cidx)) = m.ckpt else {
            return Ok(0);
        };
        // need the full SWA window present in the checkpoint, and a token left to run
        if pos < swa_window || pos < MIN_CACHE_PREFIX || pos >= tokens.len() {
            return Ok(0);
        }
        let nb = pos / 16;
        {
            let bs = self.batch.as_mut().expect("batch enabled");
            let bps = bs.blocks_per_slot;
            let pool = bs.pool.as_mut().expect("prefix requires the pool");
            bs.tables[slot].clear(pool); // release the slot's previous sequence
            bs.tables[slot].share_prefix(&m.blocks[..nb], pool); // adopt, zero copy
            // mirror the shared prefix into the host table so the suffix's first
            // ensure_pool_rows upload publishes it to d_bt[0].
            let base = slot * bps;
            for j in 0..nb {
                bs.block_table_host[base + j] = bs.tables[slot].blocks()[j];
            }
        }
        if paddock_models::dev_var_os!("PADDOCK_P5C_NO_SWA").is_none() {
            self.swa_window_copy(slot, cidx, pos, false)?; // restore SWA ring window
        }
        if paddock_models::dev_var_os!("PADDOCK_POOL_STATS").is_some() {
            tracing::warn!(
                "p5c: slot {slot} resumed at pos {pos} (reused {pos}/{} tokens)",
                tokens.len()
            );
        }
        self.paged_reused_blocks += nb as u64;
        Ok(pos)
    }

    /// P5c: cache `slot`'s completed prompt - insert its full-attn blocks into the
    /// radix (retained in the pool) and snapshot the trailing SWA window at the
    /// last block boundary, so a later shared-prefix request can resume exactly.
    /// Pool mode only; no-op when the paged prefix cache is off.
    fn paged_prefix_store(&mut self, slot: usize, tokens: &[u32]) -> Result<(), GpuModelError> {
        if self
            .batch
            .as_ref()
            .and_then(|b| b.paged_prefix.as_ref())
            .is_none()
        {
            return Ok(());
        }
        let swa_window = self.swa_window;
        let last = (tokens.len() / 16) * 16; // last full block boundary
        // insert the full-attn blocks (retained in the pool) so they can be shared.
        {
            let bs = self.batch.as_mut().expect("batch enabled");
            let blocks = bs.tables[slot].blocks().to_vec();
            let pool = bs.pool.as_mut().expect("prefix requires the pool");
            bs.paged_prefix
                .as_mut()
                .expect("paged prefix checked above")
                .insert(tokens, &blocks, pool);
        }
        // Checkpoint the SWA window at a cadence of `swa_window`-token boundaries
        // (not just the end) so a later request that shares only part of this
        // prompt still finds a checkpoint inside the shared region to resume at.
        // Each checkpoint stores [pos-swa_window, pos) SWA KV - still in the ring
        // right after this prefill (ring holds ≥ row_cap+window positions).
        // The SWA ring only retains the last `swa_ring*16` positions at store
        // time, so a checkpoint whose window falls before that has been evicted
        // and would snapshot stale KV. Only checkpoint boundaries whose full
        // window is still resident (no-op for prompts ≤ ring capacity; for very
        // long prompts it limits reuse to the recent prefix - a documented
        // follow-up would snapshot per-tick as boundaries are crossed).
        let ring_positions = self.batch.as_ref().expect("batch enabled").swa_ring_blocks * 16;
        let min_pos = (tokens.len() + swa_window).saturating_sub(ring_positions);
        let cadence = swa_window.max(16);
        let mut pos = swa_window.max(min_pos.div_ceil(cadence) * cadence); // ≥ window, cadence-aligned
        let mut n_ckpt = 0;
        while pos <= last {
            let idx = {
                let bs = self.batch.as_mut().expect("batch enabled");
                bs.paged_prefix
                    .as_mut()
                    .expect("paged prefix checked above")
                    .attach_state(tokens, pos)
            };
            if let Some(cidx) = idx {
                self.swa_window_copy(slot, cidx, pos, true)?; // snapshot ring -> ckpt
                n_ckpt += 1;
            }
            pos += cadence;
        }
        if paddock_models::dev_var_os!("PADDOCK_POOL_STATS").is_some() {
            tracing::info!(
                "p5c: store slot {slot} tokens {} checkpoints {n_ckpt}",
                tokens.len()
            );
        }
        Ok(())
    }

    /// The device-driven B=1 decode step + on-device argmax epilogue: embed from
    /// `d_g_token`, run the layers with the device position, dp4a lm_head, then
    /// argmax writes the next token into `d_g_token`, appends it to
    /// `d_g_out[d_g_step++]`, and bumps the device position - everything a CUDA
    /// graph can capture (kernel launches only; no host sync or allocation).
    fn record_gen_step(&mut self) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let (embd, rms_eps, vocab) = (self.embd, self.rms_eps, self.vocab);
        {
            let bs = self.batch.as_mut().ok_or(GpuModelError::BatchDisabled)?;
            exec.embed_gather_batch_q8(&self.tok_embd, &bs.d_g_token, &mut bs.d_x, embd, 1)?;
        }
        // Move the device position out around the `&mut self` call (host-side move;
        // the device address the capture records is unchanged). max_pos only picks
        // the attention dispatch at record time - the plain B=1 kernel reads the
        // true position from the device buffer every replay, so any position is
        // correct; capture happens right after the prompt (short), so n_splits=1.
        let d_pos = self
            .d_g_pos
            .take()
            .expect("d_g_pos allocated before capture");
        let r = self.run_layers(1, &d_pos, None, self.pos, false, None);
        self.d_g_pos = Some(d_pos);
        r?;
        let d_pos = self.d_g_pos.as_mut().expect("d_g_pos");
        let bs = self.batch.as_mut().ok_or(GpuModelError::BatchDisabled)?;
        exec.rmsnorm_batch(&bs.d_x, &self.out_norm.buf, &mut bs.d_xn, embd, rms_eps, 1)?;
        exec.quantize_q8(&bs.d_xn, &mut bs.d_b1_xq, &mut bs.d_b1_xs, embd)?;
        exec.q8_0_gemv_dp4a_nc(
            &self.output_r,
            &bs.d_b1_xq,
            &bs.d_b1_xs,
            &mut bs.d_logits,
            1,
        )?;
        exec.argmax_advance(
            &bs.d_logits,
            vocab,
            &mut bs.d_g_pmax,
            &mut bs.d_g_pidx,
            &mut bs.d_g_token,
            d_pos,
            &mut bs.d_g_mrope,
            &mut bs.d_g_out,
            &mut bs.d_g_step,
        )?;
        Ok(())
    }

    /// Capture `record_gen_step` into a replayable graph (qwen35's capture shape:
    /// quiesce -> THREAD_LOCAL capture -> record -> instantiate).
    fn capture_gen_graph(&mut self) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        exec.stream
            .synchronize()
            .map_err(|e| GpuError::Driver(format!("pre-capture sync: {e}")))?;
        exec.stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|e| GpuError::Driver(format!("begin_capture: {e}")))?;
        let rec = self.record_gen_step();
        let graph = crate::gpu::end_capture_no_flags(&exec.stream)
            .map_err(|e| GpuError::Driver(format!("end_capture: {e}")));
        rec?; // surface a record failure only after capture is cleanly ended
        let graph =
            graph?.ok_or_else(|| GpuError::Driver("gen capture produced no graph".into()))?;
        self.batch.as_mut().expect("batch enabled").gen_graph = Some(SendGraph(graph));
        Ok(())
    }

    pub fn generate_greedy(
        &mut self,
        prompt: &[u32],
        max_new: usize,
    ) -> Result<Vec<u32>, GpuModelError> {
        self.reset();
        let mut logits = Vec::new();
        for &t in prompt {
            logits = self.forward_one(t)?;
        }
        let argmax = |l: &[f32]| -> u32 {
            let mut best = 0usize;
            for (i, v) in l.iter().enumerate() {
                if *v > l[best] {
                    best = i;
                }
            }
            best as u32
        };
        let mut out = Vec::with_capacity(max_new);
        let token0 = argmax(&logits);
        out.push(token0);
        if max_new == 1 {
            return Ok(out);
        }

        // Eager loop when graphs are pinned off explicitly - a captured graph
        // bakes the record-time dispatch and would silently ignore the env
        // (P6m lesson).
        let eager = paddock_models::dev_var_os!("PADDOCK_NO_B1_GRAPH").is_some();
        if eager {
            let mut next = token0;
            for _ in 1..max_new {
                let l = self.forward_one(next)?;
                next = argmax(&l);
                out.push(next);
            }
            return Ok(out);
        }

        // Graph-resident generation: seed the device token/position, capture the
        // step once per BatchState, then replay in chunks - one sync + one small
        // readback per GEN_CHUNK tokens instead of a logits round-trip per token.
        let p = self.pos; // = prompt.len()
        assert!(
            p + max_new <= self.max_ctx,
            "context {p} + {max_new} exceeds max_ctx {}",
            self.max_ctx
        );
        let exec = self.exec.clone();
        if self.d_g_pos.is_none() {
            self.d_g_pos = Some(exec.alloc_u32(1)?);
        }
        let drv = |x: cudarc::driver::DriverError| crate::gpu::from_driver(x);
        {
            let bs = self.batch.as_mut().ok_or(GpuModelError::BatchDisabled)?;
            exec.stream
                .memcpy_htod(&[token0], &mut bs.d_g_token)
                .map_err(drv)?;
            let d_pos = self.d_g_pos.as_mut().expect("d_g_pos allocated above");
            exec.stream.memcpy_htod(&[p as u32], d_pos).map_err(drv)?;
        }
        if self
            .batch
            .as_ref()
            .expect("batch enabled")
            .gen_graph
            .is_none()
        {
            self.capture_gen_graph()?;
        }

        let target = max_new - 1; // token0 already emitted
        let mut produced = 0usize;
        while produced < target {
            let k = (target - produced).min(GEN_CHUNK);
            {
                let bs = self.batch.as_mut().expect("batch enabled");
                exec.stream
                    .memcpy_htod(&[0u32], &mut bs.d_g_step)
                    .map_err(drv)?;
                let g = bs.gen_graph.as_ref().expect("gen graph captured above");
                for _ in 0..k {
                    g.0.launch()
                        .map_err(|x| GpuError::Driver(format!("gen launch: {x}")))?;
                }
            }
            let ids = exec.to_host_u32(&self.batch.as_ref().expect("batch enabled").d_g_out)?;
            for &id in ids.iter().take(k) {
                out.push(id);
                produced += 1;
            }
        }
        self.pos = p + produced;
        Ok(out)
    }
}

/// Quantize batch activations once for a run of dense matmuls sharing the same
/// input (wq/wk/wv share the post-norm rows - qwen P6j dedup). Layout picked by
/// the same batch>64 rule the GEMM half uses: >64 -> flat mmq layout, else
/// strided int8 + per-32 scales.
#[allow(clippy::too_many_arguments)]
fn mm_quant(
    exec: &GpuExecutor,
    x: &CudaSlice<f32>,
    xq: &mut CudaSlice<i8>,
    xs: &mut CudaSlice<f32>,
    yq: &mut CudaSlice<u8>,
    in_dim: usize,
    b: usize,
) -> Result<(), GpuModelError> {
    if b > 64 {
        exec.quantize_q8_mmq(x, yq, in_dim, b)?;
    } else {
        exec.quantize_q8(x, xq, xs, b * in_dim)?;
    }
    Ok(())
}

/// GEMM half for activations staged by [`mm_quant`]: the qwen batch ladder on
/// the int8 quantized class, with the row bias folded into each rung's
/// epilogue (bit-exact vs the old trailing `bias_add` launch - the fold adds
/// bias to the completed per-element sum in the same order; the mmq rung
/// splits it store/fixup to keep the order). Killing the 4 bias launches per
/// layer is part of the dense-chain slimming.
/// Ladder: nc GEMV (2..=4, gemv-class weight BW), dp4a MT tile (..=32, batch
/// fits 1-2 weight passes), mma TC tile (..=64), mmq stream-k above. The
/// small-batch rungs matter because the mma grid is N-tiles only - on wk/wv
/// (N=512) that is 8 blocks, which idles the GPU for ~60 us/launch and
/// dominated the serving decode step at low B.
#[allow(clippy::too_many_arguments)]
fn mm_pre(
    exec: &GpuExecutor,
    w: &RepackedQ8,
    bias: &CudaSlice<f32>,
    xq: &CudaSlice<i8>,
    xs: &CudaSlice<f32>,
    yq: &CudaSlice<u8>,
    skfix: &mut CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    b: usize,
) -> Result<(), GpuModelError> {
    if b > 64 {
        // The cp.async pipe kernel now K-pads its final partial 128-chunk
        // (weight tail masked, activation zero-quantized), so gpt-oss's K=2880
        // rides it bit-exactly, and folds bias in its epilogue. Its barrier
        // stall is 0.20 vs stream-k mmq_b's 1.16.
        // Standalone A/B (wqkv 5120x2880 / wo 2880x2880, us):
        //     batch   256    512    1024   2048
        //   mmq_b      39/39  76/52  132/72 257/138
        //   pipe       49/49  57/49  115/62 231/122
        // pipe wins from ~512 up (deep prefetch amortizes; -25% on wqkv@512),
        // but its 1-CTA/SM tiled grid has a ~48us floor that stream-k beats
        // below 512 (wo@256 loses +24%). So gate on b>=512. It cuts prefill
        // TTFT but sits off the throughput-critical path for decode-bound
        // shapes, so aggregate throughput is flat. PADDOCK_NO_MMQ_PIPE
        // disables; PADDOCK_MMQ_TILED pins plain stream-k tiling.
        let pipe = b >= 512
            && exec.has_q8_0_gemm_mmq_pipe()
            && paddock_models::dev_var_os!("PADDOCK_NO_MMQ_PIPE").is_none();
        if pipe {
            exec.q8_0_gemm_mmq_pipe(w, Some(bias), yq, y, b)?; // bias folded in epilogue
        } else {
            let tiled = paddock_models::dev_var_os!("PADDOCK_MMQ_TILED").is_some();
            exec.q8_0_gemm_mmq_b(
                w,
                Some(bias),
                yq,
                if tiled { None } else { Some(skfix) },
                y,
                b,
            )?;
        }
    } else if b > 8 {
        // K-split mma: the plain 64x64-tile grid is out_dim/64 blocks (wk/wv:
        // 8) and idles a 188-SM die. The z-split partial planes + fixed-order
        // combine measured wq 56 -> 39, wk 37 -> 16, wo 56 -> 33 us at B=32
        // (and beat both the plain mma and the 2-pass mt tile at B=64).
        exec.q8_0_gemm_mma_ks_b(w, Some(bias), xq, xs, skfix, y, b)?;
    } else if b > 4 {
        // 5..=8 rode the dp4a MT tile while wk/wv (N=512) starved the mma
        // grid; the fused wqkv (N=5120 -> 80 tiles x z-split) and wo (45
        // tiles) fill the die, and the K-split mma at b<=16 runs the BN16
        // rung (c8 profile: mt was 24.1us avg at 2.6x its read floor).
        exec.q8_0_gemm_mma_ks_b(w, Some(bias), xq, xs, skfix, y, b)?;
    } else {
        exec.q8_0_gemv_dp4a_nc_b(w, Some(bias), xq, xs, y, b)?;
    }
    Ok(())
}
