//!
//! First non-gpt-oss architecture and the generalization proof. Every 4th layer
//! (`full_attention_interval`) is a **gated full-attention** layer with partial
//! M-RoPE (multimodal sectioned rotary); the other three are **Gated DeltaNet**
//! linear-attention layers (recurrent matrix state, no KV cache). Both mixers are
//! gated. All matmul weights are Q8_0 -> reuse the existing fused Q8_0 GEMM;
//! norms / A / dt-bias live as f32.
//!
//! This module is built bottom-up: the loader (weights + geometry, hybrid layer
//! typing) lands first and is validated against the real 9B GGUF; the forward
//! graph and `Generator` impl follow, parity-gated against the pinned b9895
//! llama.cpp binary on the identical Q8_0 GGUF (same-weights greedy match - no
//! bf16/quant noise; see tests/qwen35_vs_llamacpp.rs).
//!
//! Multimodal is in scope: the full-attn M-RoPE already consumes the full 4-axis
//! [t,h,w,e] position layout (see `gpu.rs::mrope`), so the vision tower (P4) slots
//! in without reworking the text path.
//!
//! Split into gemma4-shaped submodules.

use crate::gpu::{
    DeviceTensor, ExpertCache, GpuExecutor, HostMappedKq, KvDtype, QuantTensor, QuantW, RepackedKQ,
    RepackedMxfp4, RepackedQ8,
};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::prefix_cache::BLOCK_TOKENS;
use crate::kv_pool::{BlockTable, KvPool};
use crate::paged_radix::PagedRadix;
use cudarc::driver::CudaSlice;
use std::sync::Arc;

mod batch;
mod dflash;
mod forward;
mod load;
mod multimodal;
mod ops;
mod prefix;
mod spec;
pub mod vision;

pub(crate) use ops::*;

/// Only resume from a checkpoint at least this deep - below it the state
/// restore + KV page copies aren't worth skipping the (cheap) short prefill.
///
/// This is the knob wide-batch bimodality turns out to sit on. With the cache
/// on, a wide cell's median throughput FALLS and its TTFT p50 rises, and a
/// profile of a stalling window shows no GPU idle at all - so a short-prefix
/// resume is not saving work, it is ADDING it. PADDOCK_MIN_CACHE_PREFIX
/// overrides so the threshold can be swept against a real workload instead of
/// guessed.
fn min_cache_prefix() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_MIN_CACHE_PREFIX")
            .ok()
            .and_then(|v| v.parse().ok())
            // 512, not 32. Measured at wide batch with 128-token prompts, so
            // every hit resumed: resumes on runs slower AND bimodal, resumes
            // off via this floor runs faster and tight - while INSERTS still
            // ran in both arms, which is what isolates the resume path as the
            // cost. 512 keeps the cache for prefixes whose prefill is
            // actually worth skipping (~34 ms at this lane's wave rate) and
            // stops it for the short ones where the restore dominates. The
            // pay-off point above 512 is not measured.
            .unwrap_or(512)
    })
}

/// Largest CONFIGURED slot count for which a SHORT-prefix resume is still
/// taken. The resume cost is per-slot and lands in one tick, so a serve with
/// 32 slots pays it 32 times at once; one with eight pays it eight times and
/// it is a clear win there. Gating on the instantaneous live count instead
/// leaks during a cohort ramp and the regression comes back.
///
/// Measured on one binary with 128-token prompts (so every hit resumed),
/// resumes on vs a floor that refused them: resuming wins at c1/c4/c8, ties
/// at c16, and loses badly (plus goes bimodal) at c32. The sign flips between
/// c8 and c16, so the gate sits there.
fn resume_live_max() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_RESUME_LIVE_MAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(12)
    })
}

/// Don't bother snapshotting checkpoints for prompts shorter than this.
const MIN_SNAPSHOT_LEN: usize = 3 * BLOCK_TOKENS;

/// The DeltaNet-state checkpoint boundary for a prompt of `t_len` tokens: its
/// last full page boundary (the multi-turn resume point), or 0 if the prompt is
/// too short to be worth checkpointing. A hybrid model can only resume where
/// state was snapshotted, so this is the single point where the fused prefill
/// and the classic prefill both checkpoint.
fn ckpt_pos(t_len: usize) -> usize {
    if t_len >= MIN_SNAPSHOT_LEN {
        (t_len - 1) / BLOCK_TOKENS * BLOCK_TOKENS
    } else {
        0
    }
}

/// The checkpoint boundaries for a prompt: its last two full page boundaries,
/// ascending ([0, 0] when too short). Two, not one: a re-rendered multi-turn
/// history diverges inside the trailing generation header, and whenever the
/// prompt's final partial page is SHORTER than that header the divergence
/// crosses the last boundary - a checkpoint only there is unreachable for the
/// next turn (~5/16 of prompt lengths; found live as a deterministic 0% reuse
/// under concurrency). The second-to-last boundary stays inside
/// the shared prefix and keeps the next turn resumable; the last keeps the
/// exact-repeat resume maximal. The dense serial path has cut both since P5c -
/// this brings the paged serial + unified planner paths to the same contract.
/// `step` is BLOCK_TOKENS untiered; with the KV tier armed it is the tier's
/// run span (run_blocks x 16), so a demoted boundary's blocks are exactly
/// restorable - runs are the tier's restore granularity. Both boundaries
/// stay `step` apart, preserving the two-boundary divergence contract at the
/// coarser granularity.
fn ckpt_cuts(t_len: usize, step: usize) -> [usize; 2] {
    let step = step.max(BLOCK_TOKENS);
    let b1 = ckpt_pos(t_len) / step * step;
    [b1.saturating_sub(step), b1]
}

// max chunked prefills in flight: crate::service::max_chunks_inflight()
// (one shared value so `prefill_begin` never over-admits)

/// Default chunk-prefill rows advanced per mixed tick, across all in-flight
/// chunks. Bounds how long a mixed tick spends prefilling before the decode
/// rows get their next token - smaller = tighter decode latency, more ticks;
/// larger = fewer, fatter ticks (better prefill amortization, worse decode
/// jitter). The scheduler passes its own outer budget (prefill_tick_rows,
/// 8192); we clamp to this. PADDOCK_CHUNK_TICK_ROWS overrides.
// 8192 (was 2048): with 2048-token prompts a 2048 cap forces exactly one
// prompt per mixed-tick weight pass - the same-volume-over-more-weight-passes
// failure service.rs already fixed on its side (its budget is 8192; this
// backend clamp silently undid it). At 8192 the pass packs ~3 prompts and
// wide-batch output throughput climbs sharply; 4096 is close but behind, so
// the service-matching 8192 is elected. The scratch grows on demand
// (cap = t + headroom), so the only cost is transient VRAM.
const CHUNK_TICK_ROWS_DEFAULT: usize = 8192;
/// Floor of the chunk election: below this the scratch is small change and a
/// refusal names the real term.
pub(super) const PREFILL_CHUNK_ROWS_MIN: usize = 512;

/// One in-flight prompt queued for chunked prefill on `slot`. The unified tick
/// (`forward_unified_sampled`) advances it a budgeted SPAN per tick from `done`
/// (intra-prompt chunking); the legacy `advance_chunks` path prefills it whole.
struct ChunkedPrefill {
    slot: usize,
    tokens: Vec<u32>,
    /// tokens already prefilled into the slot's KV/state (0 = fresh). The next
    /// unified span covers `tokens[done..done+take]` and resumes the slot's
    /// DeltaNet state / conv window in place.
    done: usize,
}

/// The prefill chunk the operator ASKED for. The one served is
/// `GpuQwen35::prefill_chunk_rows`, elected at `enable_batch`: it starts here
/// and halves while the chunk's profiled scratch would not fit beside the KV
/// the configuration needs (never below `PREFILL_CHUNK_ROWS_MIN`).
fn chunk_tick_rows() -> usize {
    static ROWS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ROWS.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_CHUNK_TICK_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n: &usize| (32..=16384).contains(&n))
            .unwrap_or(CHUNK_TICK_ROWS_DEFAULT)
    })
}

/// Prefill rows the UNIFIED fused tick admits per tick. Kept SMALL deliberately:
/// the fused tick's throughput comes from decode staying saturated while a little
/// prefill rides along, so a light prefill share per tick maximizes decode
/// overlap. Qualitatively confirmed saturated CONC=64: small budgets (~32-64) keep
/// decode saturated, while large ones (128+) make prefill-heavy ticks that starve
/// decode and drive timeouts. (Absolute throughput under a client-bound bench
/// here is uninformative - see the client-bound-bench note.) Long prompts still fully
/// prefill: they just chunk over more light ticks (intra-prompt, DeltaNet state +
/// conv window resumed in the slot across ticks). Distinct from `chunk_tick_rows`
/// (the legacy whole-prompt path's much larger budget). Override with
/// PADDOCK_UNIFIED_PREFILL_ROWS.
const UNIFIED_PREFILL_ROWS_DEFAULT: usize = 64;
fn unified_prefill_rows() -> usize {
    static ROWS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ROWS.get_or_init(|| {
        std::env::var("PADDOCK_UNIFIED_PREFILL_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n: &usize| (8..=8192).contains(&n))
            .unwrap_or(UNIFIED_PREFILL_ROWS_DEFAULT)
    })
}

/// Gated DeltaNet (linear-attention) mixer weights - the 3-of-4 layers.
struct DeltaNetWeights {
    /// `attn_qkv` [embd, conv_dim] - in-proj to the pre-conv mixed q,k,v.
    in_qkv: QuantW,
    /// `ssm_conv1d` [conv_dim, k] F32 - depthwise causal conv weight.
    conv_w: DeviceTensor,
    /// `ssm_alpha`/`ssm_beta` [embd, n_v_heads] - data-dependent decay/delta
    /// projections. Some only when the export ships them Q8_0 (the fused
    /// decode kernel + x2 prefill pair are Q8_0-class); non-Q8 exports (UD
    /// k-quant files ship F16 here) load as `ab_f32` alone and every path
    /// takes the ab route. Invariant: alpha_w.is_some() == beta_w.is_some(),
    /// and alpha_w.is_none() implies ab_f32.is_some().
    alpha_w: Option<RepackedQ8>,
    beta_w: Option<RepackedQ8>,
    /// x2-v3 (PADDOCK_AB_F32): alpha||beta dequantized to one f32 plane
    /// [embd, 2*n_v_heads] so big prefill spans run a single 64-col-aligned
    /// tiled f32 GEMM (pd_gemm_f32_nt) instead of the Q8 repacked pair -
    /// same values (exact dequant), tiled accumulation order (PPL-gated).
    /// MANDATORY (all spans, decode included) when alpha_w is None.
    ab_f32: Option<DeviceTensor>,
    /// `ssm_a` [n_v_heads] F32 - per-head decay base, stored as -exp(A_log).
    ssm_a: DeviceTensor,
    /// `ssm_dt.bias` [n_v_heads] F32 - timestep bias.
    dt_bias: DeviceTensor,
    /// `ssm_norm` [state_size] F32 - gated-RMSNorm weight over the head state.
    ssm_norm: DeviceTensor,
    /// `attn_gate` [embd, value_dim] - the z gate (SiLU) for the gated norm.
    gate_w: QuantW,
    /// `ssm_out` [value_dim, embd] - out-proj inner -> hidden.
    out_w: QuantW,
}

/// The `nextn.*` MTP (multi-token-prediction) block - one full gated-attention
/// transformer layer plus the DeepSeek-V3-style draft plumbing (b9895
/// qwen35.cpp::graph_mtp is the reference): draft logits =
/// lm_head(shared_head_norm(attn_layer(eh_proj(enorm(embed(t)) || hnorm(h))))),
/// where `h` is the previous position's post-`output_norm` backbone hidden (or,
/// when chaining drafts, the MTP block's own post-`shared_head_norm` output).
/// The speculative-decoding draft head (P3c).
struct MtpWeights {
    /// `nextn.eh_proj` [2*embd, embd] - projects concat(e_norm, h_norm).
    eh_proj: QuantW,
    /// `nextn.enorm` / `nextn.hnorm` [embd] F32 - RMS norms over the drafted
    /// token's embedding and the incoming hidden state.
    enorm: DeviceTensor,
    hnorm: DeviceTensor,
    /// `nextn.shared_head_norm` [embd] F32 - final norm before the shared lm_head.
    head_norm: DeviceTensor,
    /// The block's own full-attn layer + dense FFN (same shapes as a backbone
    /// full-attn layer; it keeps its own KV cache).
    attn_norm: DeviceTensor,
    post_norm: DeviceTensor,
    attn: FullAttnWeights,
    /// The block's FFN follows the backbone class: dense on the 27B, MoE
    /// (+shexp) on the 35B-A3B.
    ffn: Ffn,
}

/// Gated full-attention mixer weights - every 4th layer.
struct FullAttnWeights {
    /// `attn_q` [embd, 2*q_dim] - query || output-gate, interleaved per head.
    wq: QuantW,
    /// `attn_k` [embd, kv_dim].
    wk: QuantW,
    /// `attn_v` [embd, kv_dim].
    wv: QuantW,
    /// `attn_q_norm` [head_dim] F32 - per-head QK-RMSNorm.
    q_norm: DeviceTensor,
    /// `attn_k_norm` [head_dim] F32.
    k_norm: DeviceTensor,
    /// `attn_output` [q_dim, embd].
    wo: QuantW,
}

/// One transformer block: shared pre-norms + FFN, plus the layer-type-specific
/// mixer.
// one per layer, built once at load and matched on every forward: an indirection
// would cost a hop on the hot path to save nothing that matters
#[allow(clippy::large_enum_variant)]
enum Mixer {
    Linear(DeltaNetWeights),
    Full(FullAttnWeights),
}

/// fp8 (W8A8-e4m3) plane variants of a layer's DENSE projection weights, built
/// at load when `PADDOCK_QWEN35_W8` is set (b1). A lossy throughput class for
/// LARGE-batch (prefill) projection GEMMs: the block-scale fp8 GEMM
/// (`f8_gemm_w8`) runs ~1.6-1.85× the Q8_0 int8-MMA at batch 2048 on SM120
/// (`f8_vs_q8` microbench),
/// at a precision cost validated by perplexity - Not greedy-exact vs Q8_0. The
/// Q8_0 originals stay resident (decode + small-batch prefill keep using them);
/// these planes are consulted only above `w8_min_batch()`. Empty vec = feature
/// off, and the whole path is a no-op. MoE experts are untouched here - that's
/// the separate (b2) grouped-fp8 prototype.
#[derive(Default)]
struct LayerW8 {
    // full-attn projections
    wq: Option<RepackedMxfp4>,
    wk: Option<RepackedMxfp4>,
    wv: Option<RepackedMxfp4>,
    wo: Option<RepackedMxfp4>,
    // deltanet projections
    in_qkv: Option<RepackedMxfp4>,
    gate_w: Option<RepackedMxfp4>,
    out_w: Option<RepackedMxfp4>,
}

/// Batch from which the fp8 W8A8 projection GEMM is chosen over Q8_0 int8-MMA
/// when W8 planes are loaded. 64, re-elected down from 512: the old value
/// came from the pre-kt3 `f8_vs_q8` crossover ("parity <256"); after the
/// kt-route ladder (kt3+ktz+mtail+kt-split) the f8 class wins the sub-512
/// span band outright, on both throughput and TTFT. 256 is measurably worse
/// than 64. Sub-64 rows (tiny chat probes) stay on the exact Q8_0 path.
/// Override with `PADDOCK_QWEN35_W8_MIN`.
///
/// Re-checked on B200 (27B, f8t lanes on, 3 warm reps, one binary): 64 is
/// correct there too - do not re-sweep this on smoke shapes. The 64-vs-256
/// TIE is the informative cell: a 128-token prompt prefills at r=128, which
/// is >=64 and <256, so if the W8 route bound this shape the two would
/// differ. They do not - W8 prefill is not the smoke-shape bottleneck. A
/// floor of 2 collapses, because it forces W8 onto the tiny span/mixed
/// passes. Note generally: the int8 de-rating argument (1148 TOPS vs
/// ~7.5 PFLOPS e4m3) predicts where fp8 wins at LARGE batch and says nothing
/// about lowering thresholds - small-row GEMMs are launch-bound and fp8's
/// extra quantize/scale passes dominate the MMA rate there.
/// Decode-batch floor for the fp8 W8A8 MIXER-PROJECTION lane (the
/// `bs_w8` planes at decode). Hardcoded 8 since the lane landed, which is
/// why lowering `w8_min_batch` (the PREFILL floor) never moved a c1 cell -
/// two different gates, and only this one is on the decode path. Probing
/// it matters because at b=1 the Q8 fallback runs separate gemvs per
/// projection while these planes are already FUSED (wq holds wq|wk|wv,
/// in_qkv holds in_qkv|gate_w), so the f8 lane is both fewer bytes and
/// fewer launches - and the c1 ledger puts 32.6% of the tick in
/// pd_q8_0_gemv_repacked at 315 launches/token.
/// MEASURED AND FALSIFIED at b=1: the lane loses hard on single-stream
/// decode. The planes are fused, but the CONSUMER is `f8d_gemm_mma_ks`,
/// a GEMM built for width; at one row it discards nearly every mma and
/// the fused-launch saving cannot pay for that. So 8 is a correct
/// election, not an untested default, and the c1 door is not this gate:
/// it needs a genuine f8 GEMV over these planes (nemotron's
/// `pd_f8r_gemv_kernel` is the in-house shape - e4m3 bytes + one f32
/// scale per output row, warp-coherent), which is a new kernel plus
/// wiring, not a threshold change.
/// PADDOCK_QWEN35_F8_DEC_MIN overrides (kept as the probe instrument).
fn f8_dec_min() -> usize {
    // ELECTED 1: the DECODE half of the projection REPLACE.
    // Consumers spell `b >= f8_dec_min()`, so 1 leaves the Q8_0 projection
    // planes with no reader. This half was always clean on its own.
    paddock_models::dev_var!("PADDOCK_QWEN35_F8_DEC_MIN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
}

/// PADDOCK_NO_GEMV_MULTI=1 restores the split b=1 decode GEMV launches - the
/// same kill the granite/laguna/asr entry-317 merges use. The merge groups
/// same-input Q8_0 planes (attn q|k|v, DN in_qkv|gate_w and alpha|beta, shexp
/// gate|up) into one launch each; every output byte is identical to the
/// splits (the kernel runs the exact single-plane body per row), so the only
/// thing at stake is launch-boundary economics: qwen35_q8_fuse_bench
/// (DRAM-cold, sm_120a) prices the attn merge at 6.65 us/layer and the DN
/// in-proj merge at 2.34 us/layer against a ~4 us small-gemv latency floor.
fn no_gemv_multi() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_GEMV_MULTI").is_some())
}

/// Decode-batch floor for the fp8 FFN lane (the `bs_f8ffn` planes). Was a
/// bare literal 8 at its two call sites, which is why the b=2..7 band could
/// not be probed at all: distinct from `f8_dec_min` (the MIXER-projection
/// floor) and from `w8_min_batch` (the PREFILL floor) - three gates, and
/// this is the one deciding whether small-batch decode reads e4m3 or the
/// Q8_0 twin. It exists as a knob only so the band can be MEASURED
/// (non-KV-overhead R2.3); the elected value is the default here.
/// `PADDOCK_QWEN35_F8_FFN_MIN` overrides.
fn f8_ffn_min() -> usize {
    // ELECTED 2: the b=2..7 band measures faster on e4m3 than on this gate's
    // old literal 8, and it is not a new numeric class - b>=8 has always
    // served from these same planes. Lowering it is also what lets the Q8_0
    // twins be reclaimed at all.
    paddock_models::dev_var!("PADDOCK_QWEN35_F8_FFN_MIN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2)
}

/// PREFILL row floor for the fp8 FFN arm. Split from `w8_min_batch` (the W8
/// PROJECTION floor): the two used to share one gate, and the recorded sweep
/// that found a floor of 2 "collapses" was measuring the PROJECTIONS being
/// forced onto tiny span/mixed passes -
/// it says nothing about the FFN half, whose planes are the ones the Q8_0
/// reclaim is waiting on. Same conflation the decode side had
/// (`f8_ffn_min` vs `f8_dec_min`). Default tracks w8_min_batch so this
/// split is a no-op until the band is measured;
/// `PADDOCK_QWEN35_F8_FFN_PF_MIN` overrides.
fn f8_ffn_pf_min() -> usize {
    // ELECTED 1: short-prompt TTFT measured neutral-to-better against the
    // inherited 64, and the "lowering to 2 collapses" result that would have
    // deterred this measured the PROJECTIONS, not the FFN half (see the doc
    // comment above).
    //
    // CORRECTED to 0: every consumer spells the gate
    // `r > f8_ffn_pf_min()`, so the elected 1 covered r >= 2 and left r == 1
    // on the Q8_0 arm -- whose planes the reclaim below had already stubbed to
    // 32 bytes. A one-token prompt served '!!!!!!!!' on the default build; the
    // reclaim-off control served correct text. 0 is what "every prefill row
    // reads e4m3" actually spells, and it is the precondition load.rs now
    // demands before it drops a single Q8 plane. Not a new numeric class: the
    // same planes have served r >= 2 since this gate was split.
    paddock_models::dev_var!("PADDOCK_QWEN35_F8_FFN_PF_MIN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

/// Row floor for the f8 lm_head. Was a literal 8 at four sites plus an
/// `n_sh == 1` special case that fell to the Q8_0 head - so b=2..7 (and the
/// single-row sampled pass) were the only readers keeping a 1.35 GB Q8 head
/// resident beside its 1.31 GB e4m3 twin. b=1 in the main decode path has
/// elected f8 since the PPL gate, so lowering this is consistency, not a new
/// class. `PADDOCK_QWEN35_F8_HEAD_MIN` overrides.
fn f8_head_min() -> usize {
    // ELECTED 1 (the head REPLACE lane): every row count reads the e4m3 head,
    // which is the precondition for dropping the Q8_0 twin at load instead of
    // keeping both resident. Not a new numeric class -- b=1 in the main decode
    // path and the sampled pass have elected f8 since the PPL gate, and b>=8
    // for longer than that; this closes the b=2..7 and single-row gaps that
    // were the only readers left. Consumers spell `rows >= f8_head_min()` via
    // `head_f8()`, so 1 covers rows >= 1 and 0 would be indistinguishable.
    paddock_models::dev_var!("PADDOCK_QWEN35_F8_HEAD_MIN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
}

/// The lm_head class election - the one place the boundary is spelled.
///
/// Every head site calls this instead of writing its own row test. That is not
/// tidiness: the Q8 FFN reclaim shipped corruption twice because its coverage
/// argument was the same comparison re-spelled at a dozen call sites, and two
/// of those spellings were bare literals (`r >= 8`) that nobody updated when
/// the floor moved. With one election the head REPLACE lane's precondition can
/// test exactly what the consumers test.
///
/// `None` means this row count has no f8 arm and the caller must fall back -
/// and on a REPLACE build there is nothing to fall back to, which is what
/// `stub_guard` is for.
pub(super) fn head_f8(
    out_f8: Option<&(RepackedMxfp4, usize, usize)>,
    rows: usize,
) -> Option<&(RepackedMxfp4, usize, usize)> {
    out_f8.filter(|_| rows >= f8_head_min())
}

/// The f8 lm_head call itself: quantize the rows to e4m3, then the f8d K-split
/// vocab GEMM. Identical at every site (they differ only in which scratch they
/// own), so it lives here rather than being copied nineteen times.
#[allow(clippy::too_many_arguments)]
pub(super) fn head_f8_gemm(
    exec: &GpuExecutor,
    p: &(RepackedMxfp4, usize, usize),
    x: &CudaSlice<f32>,
    pxq: &mut CudaSlice<i8>,
    exs: &mut CudaSlice<u8>,
    part: &mut CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    rows: usize,
) -> Result<(), GpuModelError> {
    exec.quantize_e4m3(x, pxq, exs, rows * p.1)?;
    exec.f8d_gemm_mma_ks(&p.0, p.1, p.2, pxq, exs, part, y, rows)?;
    Ok(())
}

/// One-row lm_head: f8 when elected, else the Q8 GEMV behind a stub guard.
/// Three sites needed exactly this and each had written the Q8 half only.
#[allow(clippy::too_many_arguments)]
pub(super) fn head_logits_1row(
    exec: &GpuExecutor,
    out_f8: Option<&(RepackedMxfp4, usize, usize)>,
    output: &QuantW,
    x: &CudaSlice<f32>,
    pxq: &mut CudaSlice<i8>,
    exs: &mut CudaSlice<u8>,
    part: &mut CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    site: &str,
) -> Result<(), GpuModelError> {
    if let Some(p) = head_f8(out_f8, 1) {
        head_f8_gemm(exec, p, x, pxq, exs, part, y, 1)
    } else {
        stub_guard(output, site)?;
        Ok(gemv_any(exec, output, x, y)?)
    }
}

/// Refuse loudly if a Q8_0 plane the reclaim stubbed is about to be read.
///
/// The reclaim (`load.rs`) drops dense-FFN planes to 32-byte allocations once
/// it believes every band reads e4m3. That belief is gate arithmetic across a
/// dozen call sites, and it has now been wrong twice: the `r == 1` prefill hole
/// (one-token prompts served `'!!!!!!!!'`) and the whole of `spec.rs`, which
/// the audit never listed at all and which crashed every `--spec auto` request
/// with CUDA_ERROR_ILLEGAL_ADDRESS on the default build.
///
/// A stub whose reader still exists is silent corruption. This turns that into
/// a loud refusal at the exact site that was missed, which is the difference
/// between "a board cell looks fine while output is garbage" and "the request
/// says what is wrong". Call it in every Q8 fallback arm that a reclaimed plane
/// could reach; it is a length compare, so it costs nothing.
///
/// The real fix is structural - one resident plane per tensor, elected at load,
/// so no fallback arm reads a different format at all. See
pub(super) fn stub_guard(w: &QuantW, site: &str) -> Result<(), GpuModelError> {
    if let QuantW::Q8(q) = w
        && q.data.len() == STUB_LEN
        && q.dims.iter().product::<usize>() > STUB_LEN
    {
        return Err(GpuModelError::Unsupported(format!(
            "qwen35: {site} was handed a Q8_0 plane the load-time reclaim \
                 stubbed to 32 bytes -- this site has no e4m3 arm for the \
                 current row count, so it would read freed memory. This is a \
                 COVERAGE BUG in the reclaim's consumer audit, not a config \
                 error. Re-run with PADDOCK_QWEN35_F8_FFN_PF_MIN=2 to keep the \
                 Q8_0 planes resident while it is fixed."
        )));
    }
    Ok(())
}

/// Size of the placeholder a REPLACED Q8_0 plane is left holding. Big enough
/// that nothing dereferences a dangling pointer, small enough to be free, and
/// recognisable - `stub_guard` identifies a replaced plane by this length.
pub(super) const STUB_LEN: usize = 32;

/// The REPLACE half of single-plane residency: drop a Q8_0 source plane at the
/// point its elected replacement is built. Returns the bytes freed (0 if the
/// plane is not Q8_0, or was already replaced).
///
/// **Call this from inside the conversion loop, never from a batch pass at the
/// end of load.** Two reasons, and the second one is the expensive one:
///
/// 1. Coverage cannot drift. The build and the drop become one statement, so
///    there is no second list of "which planes are covered" to fall out of
///    sync with the first. Both corruptions this path has shipped were a batch
///    pass believing a coverage argument the consumers did not honour - the
///    `r == 1` prefill hole (one-token prompts served `'!!!!!!!!'`) and the
///    whole of `spec.rs`, which the audit never listed.
/// 2. The mempool stays compact. `cuMemFreeAsync` returns bytes to the pool,
///    but `cuMemPoolTrimTo` can only hand a BLOCK back to the driver when
///    nothing in it is live. Free every source in one pass at end-of-load and
///    each hole is sandwiched between live replacement planes, so the driver
///    never sees the memory again: measured **5.91 GB retained-not-live against
///    a 0.71 GB no-reclaim control** on qwen3.8-27B, i.e. the reclaim's own
///    footprint was 5.2 GB.
///    Freed in-loop, the next tensor's upload reuses the hole and the pool
///    never grows past steady state - which also drops PEAK load VRAM from
///    ~55 GB to roughly the steady-state figure.
///
/// This is `replace_parameter()`'s shape in vLLM
/// (`model_executor/layers/quantization/utils/layer_utils.py:22`) and the
/// reason that design carries neither the fragmentation nor the coverage
/// bugs.
/// A replaced seat, built without ever allocating the plane it stands in for -
/// the cheapest form of [`replace_q8`], and the one to prefer when the elected
/// replacement sources its bytes from the FILE rather than from the resident
/// Q8_0 plane. Nothing is allocated and nothing is freed, so the mempool never
/// sees a hole at all.
///
/// `dims` are the real tensor dims: every `.dims()` consumer keeps working, and
/// `stub_guard` still recognises the seat and refuses by name.
pub(super) fn stub_plane(exec: &GpuExecutor, dims: Vec<usize>) -> Result<QuantW, GpuModelError> {
    Ok(QuantW::Q8(crate::gpu::RepackedQ8 {
        data: exec.alloc_u8(STUB_LEN)?,
        scale: exec.alloc_u8(STUB_LEN)?,
        dims,
    }))
}

pub(super) fn replace_q8(exec: &GpuExecutor, w: &mut QuantW) -> Result<u64, GpuModelError> {
    let QuantW::Q8(q) = w else { return Ok(0) };
    if q.data.len() == STUB_LEN {
        return Ok(0);
    }
    let freed = (q.data.len() + q.scale.len()) as u64;
    // Build the stub before the assignment drops the source, so the tiny
    // allocation cannot be served out of the hole it is about to make (a
    // 32-byte allocation is enough to pin a whole retained block open).
    let stub = crate::gpu::RepackedQ8 {
        data: exec.alloc_u8(STUB_LEN)?,
        scale: exec.alloc_u8(STUB_LEN)?,
        dims: q.dims.clone(),
    };
    *q = stub;
    Ok(freed)
}

/// PROJECTION floor: rows at/below which the projections read the Q8_0 planes
/// instead of their e4m3 twins. Consumers spell `r > w8_min_batch()`.
///
/// ELECTED 0 (the 7.4 GB projection REPLACE): every band reads
/// e4m3, which is the precondition for dropping the Q8_0 projection planes
/// rather than keeping both resident.
///
/// This floor was 64 for a long time and hid a real defect. Lowering it made
/// unified_launch_core's `lw8` projection arm reachable for the first time
/// (unified caps at unified_prefill_rows() = 64, lw8 was gated above it, so
/// the two never overlapped), and with it the mixer's o16 bf16-epilogue arm.
/// `prefill_add_norm_quant`'s non-mmq route then added a bf16 residual as f32
/// -- an invariant guarded only by a `debug_assert!`, which release builds
/// strip, so it corrupted silently instead of panicking. Fixed in ops.rs;
/// the kernel was never at fault (pd_f8_gemm_lin_kt with o16=1 is clean at
/// batch 1..2048 on every projection shape, max-rel 0.02-0.04 vs q8).
///
/// Do not re-price this rung from a throughput harness alone: the first
/// attempt produced two A/B pairs agreeing to 0.02% - and every one of those
/// legs was serving garbage, because a throughput benchmark never reads the
/// text. Any leg here needs a greedy text control beside it.
fn w8_min_batch() -> usize {
    // ELECTED 0: every band reads e4m3, the precondition for
    // dropping the Q8_0 projection planes. Two defects had to be fixed to
    // get here, both hidden by this floor for as long as it was 64: the o16
    // bf16 residual added as f32 (ops.rs) and mm_prefill_span, the VISION
    // prefill walk, which had no w8 arm at all (prefix.rs).
    paddock_models::dev_var!("PADDOCK_QWEN35_W8_MIN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

/// Rows at/above which Q8-file DeltaNet decay projections take the exact
/// f32-plane path (ab_gate) instead of the Q8 x2 pair. The old 1024
/// crossover was mis-set for the mid-M band: a tick profile at r=512 found
/// the x2 pair costing 9.4 ms/tick right below it, and the re-probe measured
/// ab_gate winning at every tested span, with the default-span cells in
/// noise. Default 128 (the lowest probed row count); `PADDOCK_AB_F32_MIN`
/// overrides.
/// Non-Q8 files ignore this (the ab plane is their only path).
fn ab_f32_min_rows() -> usize {
    paddock_models::dev_var!("PADDOCK_AB_F32_MIN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128)
}

/// Batch (tokens) from which the fp4 W4A8 grouped MoE is chosen over the Q8_0
/// int8-MMA sorted path when fp4 planes are loaded. Like the q8 sorted class it
/// only pays once blocks are well-populated; below this, decode/small batches
/// stay on exact Q8_0. Override with `PADDOCK_QWEN35_MOE_FP4_MIN`.
fn moe_fp4_min_batch() -> usize {
    std::env::var("PADDOCK_QWEN35_MOE_FP4_MIN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256)
}

/// Row cap for one batched prefill pass (Lever 1). Several admitted prompts fuse
/// into one weight-amortized forward up to this many concatenated tokens; the
/// cohort splits into cap-sized passes above it (bounds scratch). The MoE (the
/// dominant weight-bandwidth cost) saturates ~512 tokens, so even 2-4 prompts
/// per pass already amortize the 256-expert weight read - 8192 fuses ~8×1024
/// prompts per read vs the serial default's one read per prompt. Override with
/// `PADDOCK_QWEN35_BATCH_PREFILL_CAP`.
fn batch_prefill_cap() -> usize {
    paddock_models::dev_var!("PADDOCK_QWEN35_BATCH_PREFILL_CAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8192)
}

/// Convert each layer's dense-projection Q8_0 weights to fp8 W8A8 planes (one
/// pass at load). Full-attn layers get wq/wk/wv/wo; DeltaNet layers get
/// in_qkv/gate_w/out_w. The planes roughly double the *dense-projection* weight
/// footprint (a small fraction of an A3B model - the bulk is MoE experts, left
/// on Q8_0).
///
/// The conversion CORE, called per layer from inside the backbone load loop
/// rather than as a pass over finished layers. It reads its Q8_0 inputs through
/// [`ProjSrc`], so the same code serves a resident-plane build and a
/// file-transient one - see [`build_w8_from_file`] and [`build_w8_mixer`].
fn build_w8_from(
    exec: &GpuExecutor,
    li: usize,
    src: ProjSrc<'_>,
    native: &dyn Fn(usize, &str) -> Option<Vec<u8>>,
) -> Result<LayerW8, GpuModelError> {
    // `native` returns bf16 bytes for blk.{i}.{name} from a safetensors
    // checkpoint (BF16 or official-FP8, dequanted upstream) - when present
    // the projection planes skip the Q8 hop like the FFN planes do. Byte
    // lengths are validated against the GGUF dims before use; any miss
    // falls back to the Q8-derived path silently (same plane format).
    let cat_native = |i: usize, names: &[&str], dims: &[(usize, usize)]| -> Option<Vec<u8>> {
        let mut out = Vec::new();
        for (n, &(ind, outd)) in names.iter().zip(dims) {
            let b = native(i, n)?;
            if b.len() != ind * outd * 2 {
                return None;
            }
            out.extend_from_slice(&b);
        }
        Some(out)
    };
    // tile-linear conversion (gemm/f8_lin.cuh): per-CTA contiguous weight
    // streams - the decode GEMM measured at its access-pattern roof on
    // row-major planes (1324 vs the die's 1490 GB/s slab roof). Converted
    // planes carry a 4-byte marker scale; the exec wrappers dispatch on it,
    // so every call site stays layout-blind. Kill: PADDOCK_NO_F8LIN.
    // PROJECTIONS-ONLY layout opt-out. The lin box gives a b=1 GEMV ~128x
    // fewer independent memory streams than row-major (one box = 16.9 KB
    // covering 128 output rows), and a little's-law in-flight model makes
    // stream count the thing that sets achieved
    // bandwidth - measured: lin +12.4% vs the Q8 GEMV on the 40-row-tile out
    // proj, row-major only +5.0%, and staging the box coalesced changed
    // nothing (the swizzle was never the problem).
    //
    // MEASURED - and the first reading of this was wrong twice over, so the
    // corrected version is what stands. The three probed cells did fail, but
    // not by stalling: every request died with CUDA_ERROR_ILLEGAL_ADDRESS
    // (row-major 5/6 boots, lin 6/6 clean including under NO_UNIFIED /
    // NO_CHUNKED_PREFILL / NO_BATCH_PREFILL). The multi-second sample+emit
    // tick-stall first blamed here appears in the PASSING lin leg too - it is
    // c32 chunking warmup, and was never evidence of anything.
    //
    // compute-sanitizer named the fault: pd_q8_0_gemm_mma_kernel reading
    // 186-192 B below its __half scale operand, and only for tile row 1 -
    // the signature of a 32-byte stub (row 0 fits inside it, row 1 lands in
    // the gap, rows 2+ land in the neighbour). That is the Q8_0 FFN reclaim's
    // stub, and turning the reclaim off makes row-major clean 4/4, including
    // the config that was 3/3 fatal with it on. So this switch never priced a
    // layout trade: it perturbed the pool until an EXISTING stub read faulted
    // instead of landing silently in a neighbouring allocation. The defect it
    // exposed was the reclaim's uncovered r == 1 prefill band (fixed - see
    // f8_ffn_pf_min and load.rs's consumer audit).
    //
    // PRICED, and the verdict was not to take it. Once the stub bug was
    // fixed this switch could finally be measured at serving, in four
    // A-B-B-A legs on one binary so drift cannot impersonate the arm.
    // Row-major costs a few percent at width (single-stream is a wash), and
    // it is not a clock artifact: both arms ran the same clocks, and the
    // power cap engaged in both - penalising the arm that won.
    //
    // Row-major is the only layout a b=1 f8 GEMV can read (pd_f8_gemv cannot
    // walk boxes, and the box-walking proto loses on skinny projections:
    // out proj +12.1% at 40 row-tiles). So that width regression is the
    // reclaim's ENTRY PRICE - paid before the b=1 lane even exists. And that
    // lane would not pay it back: f8lin_gemv_bench's row-major arm is only
    // +1.1% qkv / +5.0% out proj against the Q8 GEMV, and only the head wins
    // outright. Buying ~7.4 GB of headroom - which does not even raise
    // token_capacity at the shipped slots x max_ctx - by regressing wide
    // serving is not the trade.
    //
    // So this switch stays a PROBE and the projections keep both residencies.
    // What the numbers establish is the entry price of the row-major route,
    // not that the rung is dead: a kernel reaching the lin width-GEMM's
    // throughput on a ROW-MAJOR plane would erase it, and nobody has tried to
    // write one. (FFN planes keep lin regardless - b=1 FFN depends on the lin
    // GEMV, and that half already paid for itself.)
    let lin_on = f8lin_enabled(exec)
        && paddock_models::dev_var_os!("PADDOCK_QWEN35_PROJ_ROWMAJOR").is_none();
    let lin =
        |w: RepackedMxfp4, in_dim: usize, out_dim: usize| -> Result<RepackedMxfp4, GpuModelError> {
            if lin_on && in_dim.is_multiple_of(128) && out_dim.is_multiple_of(16) {
                Ok(exec.f8w_repack_lin(w, in_dim, out_dim)?)
            } else {
                Ok(w)
            }
        };
    {
        let mut w8 = LayerW8::default();
        match src {
            ProjSrc::Full {
                wq,
                wk,
                wv,
                wo: wo_src,
            } => {
                // FUSED wq|wk|wv f8 plane in the wq slot (out = 2q + 2kv =
                // 14336 - vLLM's exact qkv merge; memory-neutral, wk/wv slots
                // retired). Consumers row-slice at 0 / 2q / 2q+kv; decode
                // runs one GEMM + row_slice x3.
                let nqkv = wq.dims[1] + wk.dims[1] + wv.dims[1];
                let qkv_nat = cat_native(
                    li,
                    &["attn_q.weight", "attn_k.weight", "attn_v.weight"],
                    &[
                        (wq.dims[0], wq.dims[1]),
                        (wk.dims[0], wk.dims[1]),
                        (wv.dims[0], wv.dims[1]),
                    ],
                );
                let qkv = match qkv_nat {
                    Some(b) => exec.bf16_to_f8w(&b)?,
                    None => exec.q8_0_to_f8w_concatn(&[wq, wk, wv])?,
                };
                w8.wq = Some(lin(qkv, wq.dims[0], nqkv)?);
                w8.wk = None;
                w8.wv = None;
                let wo_nat = cat_native(
                    li,
                    &["attn_output.weight"],
                    &[(wo_src.dims[0], wo_src.dims[1])],
                );
                let wo = match wo_nat {
                    Some(b) => exec.bf16_to_f8w(&b)?,
                    None => exec.q8_0_to_f8w(wo_src)?,
                };
                w8.wo = Some(lin(wo, wo_src.dims[0], wo_src.dims[1])?);
            }
            ProjSrc::Linear {
                in_qkv,
                gate_w,
                out_w,
            } => {
                // FUSED in_qkv|gate_w f8 plane in the in_qkv slot (out =
                // conv_dim + value_dim; memory-neutral - replaces the two
                // separate planes). Consumers row-slice: in_qkv at off 0,
                // gate_w at off conv_dim; decode runs it as one 16384-out
                // GEMM + row_slice split (vLLM's exact DN merge).
                let ndn = in_qkv.dims[1] + gate_w.dims[1];
                let dn_nat = cat_native(
                    li,
                    &["attn_qkv.weight", "attn_gate.weight"],
                    &[
                        (in_qkv.dims[0], in_qkv.dims[1]),
                        (gate_w.dims[0], gate_w.dims[1]),
                    ],
                );
                let dn = match dn_nat {
                    Some(b) => exec.bf16_to_f8w(&b)?,
                    None => exec.q8_0_to_f8w_concat2(in_qkv, gate_w)?,
                };
                w8.in_qkv = Some(lin(dn, in_qkv.dims[0], ndn)?);
                w8.gate_w = None;
                let ow_nat = cat_native(li, &["ssm_out.weight"], &[(out_w.dims[0], out_w.dims[1])]);
                let ow = match ow_nat {
                    Some(b) => exec.bf16_to_f8w(&b)?,
                    None => exec.q8_0_to_f8w(out_w)?,
                };
                w8.out_w = Some(lin(ow, out_w.dims[0], out_w.dims[1])?);
            }
        }
        Ok(w8)
    }
}

/// The Q8_0 projection planes a layer's e4m3 twins are converted from - either
/// the layer's own RESIDENT planes, or per-layer transients read straight from
/// the file. One conversion path serves both, so a file-sourced build cannot
/// drift from a resident-sourced one.
enum ProjSrc<'a> {
    Full {
        wq: &'a RepackedQ8,
        wk: &'a RepackedQ8,
        wv: &'a RepackedQ8,
        wo: &'a RepackedQ8,
    },
    Linear {
        in_qkv: &'a RepackedQ8,
        gate_w: &'a RepackedQ8,
        out_w: &'a RepackedQ8,
    },
}

/// Build a layer's e4m3 projection planes from the FILE, so the Q8_0 sources
/// never have to be resident at all - the preferred form, and the one the
/// backbone loop takes on plain-GGUF lanes. Each Q8 transient is dropped as
/// soon as its plane exists, so the staging is one tensor deep.
///
/// Returns None on anything that is not a plain Q8_0 projection set (k-quant
/// seats, missing tensors): the caller then falls back to uploading the
/// resident planes, which is the pre-existing behaviour.
fn build_w8_from_file(
    exec: &GpuExecutor,
    map: &paddock_models::mapped::MappedGguf,
    li: usize,
    is_full: bool,
) -> Option<LayerW8> {
    let q8 = |n: &str| exec.repack_q8(map, &format!("blk.{li}.{n}")).ok();
    // `native` is unreachable here by construction: this path only runs when
    // no safetensors snapshot is attached, so cat_native would return None for
    // every name anyway. Spelled as a closure rather than plumbed through.
    let none = |_: usize, _: &str| None;
    if is_full {
        let (wq, wk, wv, wo) = (
            q8("attn_q.weight")?,
            q8("attn_k.weight")?,
            q8("attn_v.weight")?,
            q8("attn_output.weight")?,
        );
        build_w8_from(
            exec,
            li,
            ProjSrc::Full {
                wq: &wq,
                wk: &wk,
                wv: &wv,
                wo: &wo,
            },
            &none,
        )
        .ok()
    } else {
        let (in_qkv, gate_w, out_w) = (
            q8("attn_qkv.weight")?,
            q8("attn_gate.weight")?,
            q8("ssm_out.weight")?,
        );
        build_w8_from(
            exec,
            li,
            ProjSrc::Linear {
                in_qkv: &in_qkv,
                gate_w: &gate_w,
                out_w: &out_w,
            },
            &none,
        )
        .ok()
    }
}

/// Resident-sourced build + REPLACE, for the lanes the file path cannot serve
/// (an fp8_native/NVFP4 snapshot supplies bf16 bytes through `native`, and the
/// f8t mixer lane still reads the resident planes). The drop is in the same
/// statement as the build, and its conditions are the `is_some()` of the very
/// slots just filled - no separate coverage argument to disagree with.
/// Returns (planes, bytes freed, planes replaced).
fn build_w8_mixer(
    exec: &GpuExecutor,
    mixer: &mut Mixer,
    li: usize,
    replace: bool,
    native: &dyn Fn(usize, &str) -> Option<Vec<u8>>,
) -> Result<(LayerW8, u64, usize), GpuModelError> {
    let w8 = match &*mixer {
        Mixer::Full(w) => build_w8_from(
            exec,
            li,
            ProjSrc::Full {
                wq: w.wq.q8(),
                wk: w.wk.q8(),
                wv: w.wv.q8(),
                wo: w.wo.q8(),
            },
            native,
        )?,
        Mixer::Linear(w) => build_w8_from(
            exec,
            li,
            ProjSrc::Linear {
                in_qkv: w.in_qkv.q8(),
                gate_w: w.gate_w.q8(),
                out_w: w.out_w.q8(),
            },
            native,
        )?,
    };
    let (mut freed, mut replaced) = (0u64, 0usize);
    if replace {
        let mut drop_src = |q: &mut QuantW| -> Result<(), GpuModelError> {
            let n = replace_q8(exec, q)?;
            freed += n;
            replaced += usize::from(n > 0);
            Ok(())
        };
        match &mut *mixer {
            Mixer::Full(w) => {
                // the fused qkv plane covers wq|wk|wv; wo is its own
                if w8.wq.is_some() {
                    drop_src(&mut w.wq)?;
                    drop_src(&mut w.wk)?;
                    drop_src(&mut w.wv)?;
                }
                if w8.wo.is_some() {
                    drop_src(&mut w.wo)?;
                }
            }
            Mixer::Linear(w) => {
                // fused in_qkv|gate_w lives in the in_qkv slot
                if w8.in_qkv.is_some() {
                    drop_src(&mut w.in_qkv)?;
                    drop_src(&mut w.gate_w)?;
                }
                if w8.out_w.is_some() {
                    drop_src(&mut w.out_w)?;
                }
            }
        }
    }
    Ok((w8, freed, replaced))
}

/// The tile-linear f8 layout is on when the pack ships the lane and no kill
/// switch is set. F8_ROWSCALE planes keep their own (per-row) layout and the
/// TMA kill also disables lin (its prefill twin rides the same TMA route).
/// The sm_100 tcgen05 FFN decode lane. It is a labeled precision class
/// (per-ROW e4m3 scales, coarser than the
/// per-32 the default lane serves) exactly as gemma4's twin is, so it gets an
/// explicit flag rather than silently reclassing anyone's serve.
/// has_f8t_gemm() is the arch gate on its own - f8t_gemm and f8_repack_tiles
/// are NULLed off cc 10 in the pack's per-device table.
/// DEFAULT-ON since it passed the B200 PPL gate. `has_f8t_gemm()` is itself
/// the die gate -- the pack NULLs f8t_gemm/f8_repack_tiles off cc 10 -- so
/// this turns the lane on for sm_100 only and changes nothing elsewhere.
///
/// Gate (wikitext-2, 4095 tokens, teacher-forced through the BATCH decode path
/// that actually serves, `PPL_SLOT=1`), full serving stack vs no-flag Q8_0:
///     ppl 5.58237 -> 5.64304   +1.09%
///     cross-lane next-token top-1 agreement   95.1%
///     mean NLL +0.0108 nats, 52.4% of positions worse (a coin flip -- this
///     is perturbation, not systematic degradation)
/// The house precedent is the fp4 MoE lane, PASSED at +3.06% PPL / 93.9%
/// top-1, so this clears
/// it on both axes. Per-rung: f8t FFN half +0.27%, mixer half + the alpha||
/// beta fold +0.69%, f8 lm_head +0.22%; the wmma GEMM is numerically free
/// (fractionally better than tcgen05, i.e. accumulation-order noise).
/// PADDOCK_NO_QWEN_F8T restores the pre-gate lane.
pub(crate) fn f8t_ffn_enabled(exec: &GpuExecutor) -> bool {
    exec.has_f8t_gemm() && paddock_models::dev_var_os!("PADDOCK_NO_QWEN_F8T").is_none()
}

/// The mixer-projection half of the same lane: fused [2q|k|v] + wo on full-attn
/// layers, fused [in_qkv|gate_w] + out_w on DeltaNet layers. Rides the same
/// opt-in flag as the FFN half (one precision class, one decision) with its own
/// kill switch so the increment can be A/B'd against the FFN-only lane.
pub(crate) fn f8t_attn_enabled(exec: &GpuExecutor) -> bool {
    f8t_ffn_enabled(exec) && paddock_models::dev_var_os!("PADDOCK_NO_QWEN_F8T_ATTN").is_none()
}

/// Checkpoint-exact fp8 dense FFN (the f8row class): the three
/// MLP planes of a layer llm-compressor kept at fp8 ("strategy: channel"
/// weights, "strategy: token" activations - the NVFP4 export's fp8 islands,
/// layers 56-63 on Qwen3.8-27B) held as the file's own e4m3 bytes plus one
/// f32 scale per output row. No bf16 hop, no per-32 requantization: this is
/// the checkpoint's declared recipe, and it is the class the rival serves
/// these layers in, and it is the class other engines serve them in too.
/// Every width has an arm in the granite-proven f8row
/// family - `f8r_gemv` at b=1, `f8row_gemm2`/`f8row_gemm` (tw4d/mma64 with
/// K-split) at 2..64, `pd_f8row_gemm`'s wave arms (mcol/tw/tw5) above - so
/// the layer carries no second residency: the lin boxes those layers used
/// to build are simply not built. Motivation: profiling a wide prefill found
/// the lin class's wide arm running these eight layers' GEMMs at ~140 TF/s
/// for 29.6% of GPU time, where tw5 on the same planes measures
/// 634-666 TF/s.
pub(crate) struct F8RowFfn {
    pub gate: crate::gpu::F8RowPlane,
    pub up: crate::gpu::F8RowPlane,
    pub down: crate::gpu::F8RowPlane,
    /// in_dim of gate/up (= out_dim of down)
    pub embd: usize,
    /// out_dim of gate/up (= in_dim of down)
    pub ff: usize,
}

/// The f8row dense-FFN lane's gate: the whole width chain must be in the
/// pack (b=1 GEMV, per-row staging, the width GEMM). Kill (A/B only, labeled
/// class selector): PADDOCK_NO_F8ROW_FFN - the layers fall back to the lin
/// boxes built through the bf16 -> per-32 e8m0 hop.
pub(crate) fn f8row_ffn_enabled(exec: &GpuExecutor) -> bool {
    exec.has_f8row_dense_ffn() && paddock_models::dev_var_os!("PADDOCK_NO_F8ROW_FFN").is_none()
}

/// The NVFP4 layers' WIDE-prefill class, elected by structure:
/// when the pack carries the W4A4 family's prefill arm (`f4t`, rows >= 128)
/// the checkpoint's own W4A4 chain (`nvf4_ffn`) serves every width and the
/// fp8 twin (`nvf4_to_f8w`, ~15 GB on the 27B: 56 layers x 267 MB) is not
/// built. The election that built the twin measured it against the W4A16
/// software-dequant walk, before this arm existed; against f4t the twin loses
/// on wide prefill and ties elsewhere, at equal-or-better PPL - and W4A4 is
/// the class the lane already serves at decode. `PADDOCK_NVF4_F8W=<n>`
/// (n > 0) builds the twin again and elects it above n rows - the labeled
/// A/B; `=0` or unset takes this election.
pub(crate) fn nvf4_wide_w4a4(exec: &GpuExecutor) -> bool {
    let forced_twin = paddock_models::dev_var!("PADDOCK_NVF4_F8W")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .is_some_and(|n| n > 0);
    exec.has_nvf4_gemm_f4()
        && exec.has_nvf4_gemm_f4t()
        && ops::nvf4_w4a4_min_rows() != usize::MAX
        && !forced_twin
}

fn f8lin_enabled(exec: &GpuExecutor) -> bool {
    exec.has_f8_lin()
        && paddock_models::dev_var_os!("PADDOCK_NO_F8LIN").is_none()
        && paddock_models::dev_var_os!("PADDOCK_NO_F8W8_TMA").is_none()
        && paddock_models::dev_var_os!("PADDOCK_F8_ROWSCALE").is_none()
}

/// Batch from which the nvf4 (W4A4) projection GEMM is chosen over Q8_0 int8-MMA
/// when nv4 planes are loaded. The fp4×fp4 MMA's ~2× rate is a large-batch
/// (prefill) win; decode/small prefills keep the exact Q8_0 path. Override with
/// `PADDOCK_QWEN35_PROJ_NV4_MIN`.
fn proj_nv4_min_batch() -> usize {
    paddock_models::dev_var!("PADDOCK_QWEN35_PROJ_NV4_MIN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512)
}

/// Convert each layer's dense-projection Q8_0 weights to nvf4 (W4A4) planes at
/// load. Same footprint as the W8 planes' half (fp4 = ~half the fp8 bytes), a
/// small fraction of the A3B model (bulk is MoE experts, left on Q8_0/fp4-MoE).
fn build_nv4_planes(
    exec: &GpuExecutor,
    layers: &[Qwen35Layer],
) -> Result<Vec<LayerW8>, GpuModelError> {
    let mut out = Vec::with_capacity(layers.len());
    for l in layers {
        let mut p = LayerW8::default();
        match &l.mixer {
            Mixer::Full(w) => {
                p.wq = Some(exec.q8_0_to_nvf4(w.wq.q8())?);
                p.wk = Some(exec.q8_0_to_nvf4(w.wk.q8())?);
                p.wv = Some(exec.q8_0_to_nvf4(w.wv.q8())?);
                p.wo = Some(exec.q8_0_to_nvf4(w.wo.q8())?);
            }
            Mixer::Linear(w) => {
                p.in_qkv = Some(exec.q8_0_to_nvf4(w.in_qkv.q8())?);
                p.gate_w = Some(exec.q8_0_to_nvf4(w.gate_w.q8())?);
                p.out_w = Some(exec.q8_0_to_nvf4(w.out_w.q8())?);
            }
        }
        out.push(p);
    }
    Ok(out)
}

/// Routed-expert MoE FFN weights (qwen3.6-A3B class: 256 tiny experts, top-8,
/// plus a parallel shared expert behind a per-token sigmoid scalar gate -
/// b9951 qwen35moe.cpp::build_layer_ffn is the reference math: softmax over
/// all router logits, top-k, renormalize == softmax over the top-k logits,
/// which is exactly pd_moe_topk_warp's output).
/// Per-tensor routed-expert seat: Q8_0 (default - rides the full
/// sorted/mma/fp4 family) or k-quant-resident (the stage-3 arm: ~0.55x the
/// expert DRAM + VRAM). kq seats ride the token-batched pair at decode and
/// their own sorted mma pair (20_kquant_moe, the ks-ring kernels) past the
/// same pair-count boundary the Q8 seats use - large prefill reads each
/// touched expert's weights once per pass on both seat kinds.
impl GpuQwen35 {
    /// Seat the expert-offload slot cache over every host-mapped MoE layer:
    /// as many experts per layer as `budget` bytes buy (one slot is one
    /// expert's gate+up+down bytes in every such layer), capped by
    /// `[moe_offload] vram_gb`. PADDOCK_MOE_CACHE_SLOTS pins the count
    /// outright (development instrument, not budget-checked). Returns the
    /// slots seated; 0 when nothing is host-mapped or the budget buys too few
    /// to matter, in which case the experts keep serving zero-copy.
    pub fn enable_moe_cache(&mut self, budget: u64) -> Result<usize, GpuModelError> {
        fn host_seats(m: &MoeFfnWeights) -> Option<(&HostMappedKq, &HostMappedKq, &HostMappedKq)> {
            match (&m.gate_exps, &m.up_exps, &m.down_exps) {
                (ExpW::KqHost(g), ExpW::KqHost(u), ExpW::KqHost(d)) => Some((g, u, d)),
                _ => None,
            }
        }
        let mut moe: Vec<&mut MoeFfnWeights> = self
            .layers
            .iter_mut()
            .map(|l| &mut l.ffn)
            .chain(self.mtp.as_mut().map(|m| &mut m.ffn))
            .filter_map(|ffn| match ffn {
                Ffn::Moe(m) if m.cache.is_none() => Some(m),
                _ => None,
            })
            .collect();
        let price: u64 = moe
            .iter()
            .filter_map(|m| host_seats(m))
            .map(|(g, u, d)| ExpertCache::slot_bytes(g, u, d))
            .sum();
        if price == 0 || !self.exec.has_moe_cache() {
            return Ok(0);
        }
        let cfg = crate::gpu::moe_offload();
        let budget = cfg.vram_bytes.map_or(budget, |cap| budget.min(cap));
        let auto = (budget / price) as usize;
        let slots = crate::gpu::moe_cache_slots_pin().unwrap_or(auto);
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        if slots < 8 {
            tracing::warn!(
                slots,
                budget_gib = gib(budget),
                "qwen35 MoE expert offload: no room for a slot cache after the KV plan - \
                 experts serve zero-copy over PCIe (slow); lower max_ctx or max_batch"
            );
            return Ok(0);
        }
        let mut seated = 0;
        for m in moe.iter_mut() {
            if let Some((g, u, d)) = host_seats(m) {
                let n = slots.min(g.dims[2]);
                m.cache = Some(self.exec.new_expert_cache(g, u, d, n, 1024)?);
                seated = n;
            }
        }
        tracing::info!(
            slots = seated,
            cache_gib = gib(seated as u64 * price),
            budget_gib = gib(budget),
            pinned = crate::gpu::moe_cache_slots_pin().is_some(),
            "qwen35 MoE expert offload: VRAM slot cache seated (lower max_ctx / max_batch to grow it)"
        );
        Ok(seated)
    }

    /// True when any layer serves its experts through the offload slot cache.
    pub fn moe_cache_active(&self) -> bool {
        self.layers
            .iter()
            .any(|l| matches!(&l.ffn, Ffn::Moe(m) if m.cache.is_some()))
    }
    /// Release everything a failed `enable_batch` attempt seated. The
    /// width-retry ladder halves the width and calls again; a half-built
    /// attempt holding its BatchState (pool + tier), per-layer MoE caches,
    /// overlap stream, dflash device state and scratch reads as
    /// `0.0 GB free after weights`, and every retry then refuses regardless
    /// of width. Same teardown list as `set_kv_dtype` (the established
    /// "nuke serving state" precedent) plus what only `enable_batch` seats;
    /// the attached DFlash drafter survives - only its device state drops,
    /// `dflash_ensure_state` rebuilds it on the next attempt.
    pub(super) fn release_batch_attempt(&mut self) {
        self.pipe_abort();
        self.batch = None;
        self.spec = None;
        self.spec_batch = None;
        self.scratch = None;
        self.overlap_exec = None;
        if let Some(df) = &mut self.dflash {
            df.state = None;
        }
        for layer in &mut self.layers {
            if let Ffn::Moe(m) = &mut layer.ffn {
                m.cache = None;
            }
        }
        if let Some(mtp) = &mut self.mtp
            && let Ffn::Moe(m) = &mut mtp.ffn
        {
            m.cache = None;
        }
        self.exec.trim_mem_pool();
    }

    /// Cache counters summed over the layers: `(rows resolved, misses)` since
    /// load; `None` without a cache. Syncs the stream - for gates and logs.
    pub fn moe_cache_stats(&self) -> Result<Option<(u64, u64)>, GpuModelError> {
        let mut acc: Option<(u64, u64)> = None;
        for l in &self.layers {
            if let Ffn::Moe(m) = &l.ffn
                && let Some(c) = &m.cache
            {
                let (r, m) = c.stats(&self.exec)?;
                let a = acc.get_or_insert((0, 0));
                a.0 += r;
                a.1 += m;
            }
        }
        Ok(acc)
    }
}

enum ExpW {
    Q8(RepackedQ8),
    Kq(RepackedKQ),
    /// k-quant plane in device-mapped host memory (`[moe_offload]`): the
    /// kernels read it over PCIe on the same addressing as `Kq`. No VRAM.
    KqHost(HostMappedKq),
}

impl ExpW {
    fn q8(&self) -> Option<&RepackedQ8> {
        match self {
            ExpW::Q8(w) => Some(w),
            ExpW::Kq(_) | ExpW::KqHost(_) => None,
        }
    }
    /// The k-quant plane whichever memory it lives in - every launch takes
    /// the same `RepackedKQ` view.
    fn kq(&self) -> Option<&RepackedKQ> {
        match self {
            ExpW::Q8(_) => None,
            ExpW::Kq(w) => Some(w),
            ExpW::KqHost(w) => Some(w),
        }
    }
}

struct MoeFfnWeights {
    /// `ffn_gate_inp` [embd, n_expert] F32 - the router.
    router_w: DeviceTensor,
    /// `ffn_{gate,up}_exps` [embd, moe_ff, n_expert], repacked; row
    /// (e, o) at e*moe_ff + o.
    gate_exps: ExpW,
    up_exps: ExpW,
    /// `ffn_down_exps` [moe_ff, embd, n_expert].
    down_exps: ExpW,
    /// `ffn_gate_inp_shexp` [embd] F32 - the shared expert's sigmoid scalar gate.
    shexp_gate_inp: DeviceTensor,
    /// `ffn_{gate,up,down}_shexp` - a plain dense SwiGLU FFN of width
    /// shexp_ff. Per-tensor seat like the dense FFN: Q8_0 in the UD-Q4_K_XL
    /// exports, k-quant (Q5_K) in the UD-IQ2 ones.
    shexp_gate: QuantW,
    shexp_up: QuantW,
    shexp_down: QuantW,
    /// b2: fp4 (W4A8) plane variants of the routed experts, built at load under
    /// `PADDOCK_QWEN35_MOE_FP4`. fp4 weights are ~half the Q8_0 DRAM - the lever
    /// for the weight-bandwidth-bound MoE. Used only in the sorted prefill path
    /// above the batch threshold; Q8_0 originals stay for decode/small batch.
    /// Lossy (perplexity-gated, not greedy-exact). None = feature off.
    gate_exps_fp4: Option<RepackedMxfp4>,
    up_exps_fp4: Option<RepackedMxfp4>,
    down_exps_fp4: Option<RepackedMxfp4>,
    /// Shared all-zeros bias plane (`n_expert * max(moe_ff, embd)`) for the fp4
    /// mmq kernels, which read a per-(expert,row) bias unconditionally. qwen MoE
    /// has no bias, so every layer points at one buffer via `Arc`.
    moe_zero_bias: Option<Arc<CudaSlice<f32>>>,
    /// `[moe_offload]`: the VRAM slot cache over host-mapped expert planes,
    /// seated by `enable_moe_cache`; the token-batched class serves through
    /// it when the launch's rows fit its slots.
    cache: Option<ExpertCache>,
}

/// Dense SwiGLU or routed-expert MoE - per-layer FFN. MoE expert weights stay
/// Q8_0-only for now (k-quant experts are the stage-3 MoE arm).
// one per layer, built once at load and matched on every forward: an indirection
// would cost a hop on the hot path to save nothing that matters
#[allow(clippy::large_enum_variant)]
enum Ffn {
    Dense {
        gate: QuantW,
        up: QuantW,
        down: QuantW,
    },
    /// Checkpoint-exact NVFP4 dense FFN (the qwen3.8 lane): the
    /// llm-compressor triples uploaded byte-for-byte, served W4A16 through
    /// the nvf4 gemv/gemm family. Replaces the Q8-derived planes for its
    /// layer entirely - none of the Dense aux lanes (gu fusion, f8-ffn,
    /// f8t, W8) build for an Nvf4Dense layer, which is also what makes the
    /// lane fit: ~150 MB/layer of fp4 instead of ~284 MB of Q8_0.
    Nvf4Dense {
        gate: crate::gpu::Nvf4Plane,
        up: crate::gpu::Nvf4Plane,
        down: crate::gpu::Nvf4Plane,
    },
    Moe(MoeFfnWeights),
}

/// MoE geometry (None on dense models).
#[derive(Clone, Copy)]
struct MoeDims {
    n_expert: usize,
    n_active: usize,
    moe_ff: usize,
    shexp_ff: usize,
}

struct Qwen35Layer {
    /// `attn_norm` [embd] F32 - pre-mixer RMSNorm.
    attn_norm: DeviceTensor,
    /// `post_attention_norm` [embd] F32 - pre-FFN RMSNorm.
    post_norm: DeviceTensor,
    ffn: Ffn,
    mixer: Mixer,
}

impl Qwen35Layer {
    fn is_full(&self) -> bool {
        matches!(self.mixer, Mixer::Full(_))
    }
}

/// Qwen3.5 hybrid model, resident on one CUDA device.
pub struct GpuQwen35 {
    exec: Arc<GpuExecutor>,
    /// Second execution lane for pure-decode work (route B): its own
    /// compute+copy streams and cuBLAS handle on the same context
    /// (`fork_stream`). When forked (PADDOCK_OVERLAP, at enable_batch), the
    /// decode graphs capture on this lane's stream - a cudarc graph replays on
    /// the stream that captured it - so classic/pipe decode ticks execute
    /// there and can overlap main-lane prefill spans once the scheduler
    /// interleaves them. None = classic single-lane serve.
    overlap_exec: Option<Arc<GpuExecutor>>,
    /// True while `with_decode_lane` has the decode lane swapped into `exec`
    /// (re-entrancy guard - nested wrapped calls must not swap back).
    lane_swapped: bool,
    /// The one in-flight unified span (overlap scheduler); None otherwise.
    unified_inflight: Option<UnifiedInflight>,

    // ---- geometry ----
    n_layers: usize,
    embd: usize,
    /// full-attention query heads / kv heads / per-head dim.
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    ff: usize,
    /// MoE geometry (None on dense models).
    moe: Option<MoeDims>,
    /// DeltaNet head geometry: state size (head k/v dim), key heads, value heads,
    /// conv kernel. `value_dim == inner_size`.
    state_size: usize,
    n_k_heads: usize,
    n_v_heads: usize,
    conv_k: usize,
    value_dim: usize,
    conv_dim: usize,

    rms_eps: f32,
    /// partial M-RoPE: rotary width and the [t,h,w,e] section pair-counts.
    n_rot: usize,
    sections: [u32; 4],
    /// YaRN kernel params (ext_factor 0 for Qwen3.5 - plain rope), in
    /// `YarnRope::kernel_params` order.
    yarn_params: (f32, f32, f32, f32, f32, f32),

    pub vocab: usize,
    pub(crate) max_ctx: usize,
    /// Token history for the incremental `Generator` seam - recompute-per-step
    /// decode appends here and re-runs `forward_full` (P3 replaces this with the
    /// incremental recurrent-state + KV carry).
    history: Vec<u32>,

    // ---- weights ----
    // Input embedding table kept RESIDENT in its file quant (not dequantized to
    // f32): it's used only for the input row-gather, so we gather+dequant the
    // handful of rows in flight instead of paying 4x VRAM for the whole f32
    // table. The untied `output` projection has its own repacked copy for the
    // logits GEMV/GEMM.
    tok_embd: TokEmbd,
    layers: Vec<Qwen35Layer>,
    /// Per-layer fp8 W8A8 planes for the dense projections (b1). Empty unless
    /// `PADDOCK_QWEN35_W8` was set at load; consulted only above
    /// `w8_min_batch()`. See [`LayerW8`].
    bs_w8: Vec<LayerW8>,
    /// Per-layer nvf4 (W4A4) plane variants of the dense projections, built at
    /// load under `PADDOCK_QWEN35_PROJ_NV4`. Unlike the W8A8 planes (fp8-rate MMA,
    /// ~parity with mmq_pipe - b1) these run the fp4×fp4 block-scale MMA at ~2× the
    /// int8/fp8 rate on SM120, the real lever for the 26%-of-prefill projection
    /// GEMM. Consulted only above `proj_nv4_min_batch()`; Q8_0 originals stay for
    /// decode/small batch. Lossy (fp4 activations - perplexity-gated). See
    bs_nv4: Vec<LayerW8>,
    /// Fused gate|up Q8 planes per layer: one
    /// die-filling ks GEMM for the b>=8 decode band (nz collapses to 1 - no
    /// K-split partials/combine). None = per-tensor path (MoE, non-Q8,
    /// VRAM-tight, or PADDOCK_NO_FUSE_GU).
    bs_gu: Vec<Option<crate::gpu::RepackedQ8>>,
    /// Fused DN in_qkv|gate_w planes per layer (vLLM's 16384-out DN merge):
    /// one 256-tile ks GEMM at decode widths. None = per-tensor (full-attn
    /// layer, non-Q8, VRAM-tight, or PADDOCK_FUSE_DN unset).
    bs_dn: Vec<Option<crate::gpu::RepackedQ8>>,
    /// f8 (e4m3) FFN planes per dense layer - (gate, up, down) converted from
    /// the Q8 originals for the native-fp8 decode lane (PADDOCK_F8_DECODE,
    /// opt-in precision class; ~17 GB dup on the 27B). The dims ride along
    /// since RepackedMxfp4 carries none.
    bs_f8ffn: Vec<Option<[(RepackedMxfp4, usize, usize); 2]>>,
    /// byte-passthrough (bs) FFN decode planes (PADDOCK_FP8_BS): the official
    /// FP8 checkpoint's raw e4m3 bytes as data-only lin boxes + f32 block
    /// scales (marker-8 planes; decode-only - prefill keeps `bs_f8ffn`).
    bs_f8ffn_bs: Vec<Option<[(RepackedMxfp4, usize, usize); 2]>>,
    /// sm_100 tcgen05 FFN decode planes: [gate|up fused, down] as SW128 tile
    /// images (PADDOCK_QWEN_F8T). This is gemma4's v4 tile-image lane ported
    /// verbatim - same source (the plain Q8_0 GGUF via q8_0_to_f8row), same
    /// repacker, same f8t_gemm - and it is the difference between 2.3x and
    /// 4.7x the byte floor on B200: gemma4 decodes roughly twice as fast on
    /// f8t as on the warp-level f8w path, same model, same file, same die.
    /// cc-10 only: f8t_gemm/f8_repack_tiles are NULL off cc 10 in the pack's
    /// per-device table, so has_f8t_gemm() is the whole gate.
    bs_f8t_ffn: Vec<Option<[crate::gpu::F8TilePlane; 2]>>,
    /// Checkpoint-native NVFP4 decode plane for the FUSED gate|up GEMM, one
    /// entry per layer (None off the NVFP4 lane, without CUTLASS, or under a
    /// VRAM-headroom decline). This is the 0.5 B/param twin of
    /// The mixer-projection twin of `bs_f8t_ffn`, one entry per layer, shaped
    /// [fused input projection, output projection] for both layer kinds:
    ///   Full   -> [ wq(=[2q|gate]) | wk | wv  ,  wo    ]
    ///   Linear -> [ in_qkv | gate_w            ,  out_w ]
    /// The fusions are byte-exact tile-stream concats (tile index
    /// (row/128)*nkt + kt is plane-relative), so consumers row-slice with
    /// `f8t_gemm_off` at `out_off / 128` exactly as the warp planes do at a
    /// row offset. Every constituent tensor is 128-dim-checked at build, which
    /// is what makes those offsets tile-aligned by construction.
    bs_f8t_attn: Vec<Option<[crate::gpu::F8TilePlane; 2]>>,
    /// Checkpoint-exact fp8 dense FFN planes (the f8row class), one entry per
    /// layer - Some only for the layers the file keeps at fp8-channel and the
    /// f8row chain serves at every width (see [`F8RowFfn`]). A layer with an
    /// entry here builds no lin/W8 FFN twin: `bs_f8ffn[li]` is None there.
    bs_f8row_ffn: Vec<Option<F8RowFfn>>,
    /// f8 lm_head (PADDOCK_F8_LMHEAD, labeled precision class): the batched
    /// decode logits GEMM on the tile-linear f8 stream (b >= 8 only; b=1 and
    /// small batches keep the exact Q8 ladder).
    out_f8: Option<(RepackedMxfp4, usize, usize)>,
    /// f8t TILE-plane twin of `out_f8` (sm_100): the head on the wmma route.
    out_f8t: Option<(crate::gpu::F8TilePlane, usize, usize)>,
    out_norm: DeviceTensor,
    /// lm_head, quantized-resident (tied to `token_embd` when `output.weight`
    /// absent). UD k-quant exports ship it Q6_K.
    output: QuantW,
    /// True when any weight is k-quant resident (UD/Q4_K-class file). Stage-1
    /// serving keeps such models on the serial spine: decode via the fused
    /// k-quant GEMV, prefill via dequant+f32-GEMM; the batched pipe / spec /
    /// device-sampling paths are Q8_0-class until stage 2 (W4A8 int8-MMA), so
    /// the routing flags below report them unavailable.
    kq_resident: bool,
    /// Largest k-quant LAYER weight in elements - sizes the per-pass dequant
    /// scratch (`d_wdq`). Excludes head/embedding (never dequanted whole).
    kq_max_elems: usize,
    /// nextn/MTP draft block (models exporting `nextn_predict_layers` > 0) - the
    /// speculative-decoding head. Loaded but not yet driven (P3c wiring).
    #[allow(dead_code)]
    mtp: Option<MtpWeights>,
    /// Resident weight bytes, sampled at the end of load (weights up - MTP
    /// included, since spec is on unless PADDOCK_NO_SPEC says otherwise - and
    /// no KV/DeltaNet state yet: those live in `batch`/`decode`). This is the
    /// weights line of the memory-breakdown API and the number gen-shapes.py
    /// publishes as `source = "measured"`. Including MTP is what the estimator
    /// expects: it subtracts `nextn_bytes` again when spec is off.
    weights_bytes: Option<u64>,
    /// Content identity of the loaded weights and tokenizer, captured at
    /// load - the cache namespace's answer to "are these the same bytes?".
    /// Geometry alone stopped being a sufficient key when the tier gained a
    /// store that survives restarts (see `kv_tier::fingerprint`).
    pub(crate) content_id: ([u8; 32], [u8; 32]),
    /// Sideloaded DFlash2 block-diffusion drafter (incoai/z-lab release) -
    /// the second speculative option next to the in-file MTP head. Attached
    /// by the runner when the `mtp` companion's arch says "dflash".
    pub(super) dflash: Option<dflash::DflashDrafter>,
    /// unified tick's row mirrors awaiting the DFlash fuse+append (the walk
    /// holds field borrows; the caller appends once they end)
    pub(super) dflash_pending_append: Option<(Vec<u32>, Vec<u32>, Vec<u32>)>,
    /// Which drafter produced the CURRENT round's chunks (the hybrid: DFlash2
    /// at low live where its block drafts pay handsomely, the MTP chain
    /// above where its near-free async draft wins - the crossover measures
    /// between c4 and c8). Set by the
    /// draft routing pre-tick; the round's eligibility/commit read it.
    pub(super) spec_round_dflash: bool,
    /// Rung G: the service's per-slot RS chain draws for the round about to
    /// be drafted (`spec_rs_stash`); `dflash_draft_launch` consumes them as
    /// the per-block 1/T + seed of the sampled selector walk.
    pub(super) spec_rs_draws: Option<Vec<crate::generator::SpecRsDraw>>,
    /// Rung G: the CURRENT round's drafts came from the SAMPLED selector
    /// walk, so the drafter's `q16` plane is valid and drafted verify rows
    /// may resolve under rejection sampling. False on every other draft
    /// shape - the verify then serves RsTrunc plans under the classic rule
    /// (which is lossless with any draft).
    pub(super) spec_round_rs: bool,

    /// [n_heads] filled with a large-negative sentinel - a no-op attention "sink"
    /// so `attn_decode_batch` (which requires a sinks buffer) does plain softmax.
    /// The sentinel is -inf, the exact identity: see `alloc_no_sinks`.
    sinks: CudaSlice<f32>,
    /// KV cache element type for the full-attn (and MTP) caches. Default
    /// [`KvDtype::Fp16`] (greedy-exact); [`KvDtype::Fp8E4m3`] is the lossy
    /// opt-in throughput/memory mode - see [`Self::set_kv_dtype`].
    kv_dtype: KvDtype,
    /// Lazily-(re)allocated per-pass scratch, sized to the current token count.
    scratch: Option<Scratch>,
    /// Dedicated max_batch-row arena the captured decode graphs run on (pipe-
    /// scratch separation): keeps the graphs independent of the shared prefill
    /// scratch's lifetime and, in the follow-up, lets queued pipe ticks overlap
    /// prefill passes. Swapped into `scratch` around graph capture only.
    decode_arena: Option<Scratch>,
    /// Speculative-decoding scratch (snapshot/rollback + MTP staging), lazy.
    spec: Option<SpecState>,
    /// Continuous-batching per-slot state (enable_batch), lazy.
    batch: Option<BatchState>,
    /// Per-slot (batched) speculative-decoding state (enable_spec_batch), lazy.
    spec_batch: Option<SpecBatchState>,
    /// Open sampled-spec round (forward_spec_verify_mtp -> spec_commit_mtp):
    /// (padded slot-major chunk, pos_before per live slot).
    /// Open sampled-spec round: (padded chunk rows, per-BLOCK pos_before,
    /// block->TRUE-slot map) - the map restores round_slots/bs.d_slots at
    /// commit in case a tick between the split phases re-pointed them.
    spec_pending: Option<(Vec<u32>, Vec<usize>, Vec<u32>)>,
    /// ARMED async draft chain: the draft
    /// graph has LAUNCHED but its ids were not read back - they sit in
    /// `spec_batch.d_draft` ([n_draft, n] i-major) and the verify assembles
    /// its token rows on device from them (pd_spec_toks), so the
    /// chain->verify boundary is a queued stream sequence instead of the
    /// measured ~5.2 ms host stall per round. (chain slot list, k_use).
    /// Drivers PEEK the plane post-verify (the picks readback already
    /// synced) to run accept/commit on real values; the service's
    /// spec_draft_fetch then returns the same values and disarms.
    spec_chain: Option<(Vec<u32>, usize)>,
    /// Scheduler hint: whether prefill should eagerly warm the MTP draft head.
    /// The service turns this off when the live count exceeds the spec live
    /// cap - those slots would pay the warm pass (~1/41 of prefill) yet
    /// never ride a spec round. Default true (bench
    /// paths and single-slot spec don't set hints).
    spec_warm_wanted: bool,
    /// The scheduler judged this round's warmth by the DFlash RING PROBE only
    /// (some ring was warm, so the block drafter owns the round); the MTP
    /// cursors were not gap-synced. Set by `spec_ring_warm`, cleared by any
    /// `spec_ensure_warm` - the chain refuses to draft while it is set.
    spec_ring_probed: bool,
    /// spec live-count cap chosen by width_by_vram when the draft-state
    /// reservation would otherwise eat batch width (None = env default)
    spec_live_vram_cap: Option<usize>,
    /// Prefill rows one tick admits - `chunk_tick_rows()` unless
    /// `enable_batch` had to step it down to fit the profiled scratch.
    prefill_chunk_rows: usize,
    /// Persistent per-layer state for O(1)/token incremental decode (KV caches +
    /// DeltaNet recurrent matrix state + conv windows). None until first `step`.
    decode: Option<DecodeState>,
    /// Vision tower (separate mmproj GGUF), attached after load when serving
    /// takes image input. None = text-only.
    vision: Option<crate::gpu_model::qwen35::vision::VisionModel>,
    /// Per-slot tokens served from the prefix cache by the last prefill
    /// (usage reporting; taken and zeroed by take_prefill_reused).
    last_reused: Vec<usize>,
    /// In-flight chunked prefills (PADDOCK_CHUNKED_PREFILL). Each admission
    /// registers here via `prefill_begin`; `forward_mixed` advances every one
    /// by a budgeted span per tick alongside the live decode rows, so an
    /// admission wave never freezes the streams for its whole prompt.
    chunked: Vec<ChunkedPrefill>,
    /// Vision-tower output cache: a re-sent image (multi-turn vision chat
    /// re-renders the same picture every turn) skips preprocess + tower.
    /// Keyed by the raw request bytes; exact-bytes verified like the radix
    /// cache's tokens, so a hash collision costs a miss, never a wrong reuse.
    image_cache: Vec<ImageCacheEntry>,
    image_cache_clock: u64,
    image_cache_reused: u64,
    /// In-flight pipelined pure-decode (see [`Self::decode_pipe_begin`]). Some
    /// only transiently, between a `decode_pipe_begin` and its matching drain
    /// inside one `run_batched` decode burst; the scheduler always drains before
    /// the outer loop does anything else, so it never survives a tick boundary.
    pipe: Option<PipeStateQ>,
}

/// State of the in-flight pipelined decode: `tick` = the last ENQUEUED tick,
/// `ev[tick % 2]` fires when that tick's out-ring slot is readable. `pos0`/
/// `delta` let each tick rebuild `d_mrope` host-side (positions are
/// deterministic - only the token depends on the device-sampled id), so the
/// hybrid RoPE stays correct without a device->host roundtrip.
struct PipeStateQ {
    b: usize,
    tick: u64,
    ev: [Option<cudarc::driver::CudaEvent>; 2],
    /// tick-0 positions (row-major, len b); tick j uses `pos0[i] + j`.
    pos0: Vec<u32>,
    /// per-row mrope delta snapshot (row-major, len b); constant over the pipe.
    delta: Vec<i64>,
    /// explicit row->slot mapping (len b) for a pipe over an arbitrary slot
    /// set (the overlap scheduler's churn-phase decode set). None = identity
    /// (row i drives slot i - the classic quiet-phase pipe). With Some, the
    /// mapping is written to `d_slots` at begin and identity is RESTORED at
    /// drain/abort - every other decode path assumes identity persists.
    slots: Option<Vec<u32>>,
}

/// An in-flight unified span: everything `unified_finish_core` needs to
/// complete the tick, plus the per-call eager device buffers, which must
/// stay alive until the finish-side drain (freed-buffer reuse hazard - same
/// rule as the single-body version's tail sync). One span in flight at a
/// time; the overlap scheduler pumps decode-pipe ticks between launch and
/// finish.
struct UnifiedInflight {
    /// (chunk index, slot, done, take, finishing, span tokens) - chunk
    /// indices stay valid across the flight window because admission only
    /// APPENDS to `chunked`; removals happen in finish.
    shares: Vec<(usize, usize, usize, usize, bool, Vec<u32>)>,
    b: usize,
    tot: usize,
    nf: usize,
    plans: Vec<crate::generator::RowSample>,
    /// P65: device plans for the finisher rows (rows b..tot in stash order)
    /// - the finish half needs them to head-sample TruncCat finishers
    fin_dev: Vec<crate::sampler::DevicePlan>,
    /// finished prompts, device-planned ids still the Sampled(0) placeholder
    finished: Vec<(usize, crate::generator::FinishSample, usize)>,
    /// fires when the span's GPU work (incl. finisher sampling) completes
    ev: cudarc::driver::CudaEvent,
    // liveness-only holds (dropped by finish after its drain)
    #[allow(dead_code)]
    hold_u32: Vec<CudaSlice<u32>>,
    #[allow(dead_code)]
    hold_f32: Vec<CudaSlice<f32>>,
    #[allow(dead_code)]
    hold_seg: Vec<(CudaSlice<u32>, CudaSlice<u32>)>,
}

/// One cached vision-tower output (device-resident projected embeddings).
struct ImageCacheEntry {
    hash: u64,
    w: usize,
    h: usize,
    rgb: Vec<u8>,
    embd: CudaSlice<f32>,
    nx: usize,
    ny: usize,
    last_used: u64,
}

/// Images held in the vision-tower output cache. Each entry is the projected
/// embedding rows (~15-30 MB device) plus the raw bytes host-side for the
/// exact-match verify. Sized so a multi-image request (e.g. a multi-page
/// document sent as page images) keeps all of its images cached across turns
/// rather than thrashing within a single prefill.
const IMAGE_CACHE_ENTRIES: usize = 16;

/// FNV-1a over the raw image bytes (+ dims folded in by the caller).
fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// The token layout of an interleaved multimodal prompt: the placeholder id
/// stream, the mRoPE grid + attention bound, and where each image's embeddings
/// splice in. Built by [`build_mm_layout`] and consumed identically by both the
/// serial ([`GpuModel::prefill_multimodal`]) and batched-slot
/// ([`GpuModel::forward_prefill_slot_mm`]) prefill paths.
struct MmLayout {
    /// t_len ids: real text tokens, `0` placeholders over every image span.
    ids: Vec<u32>,
    /// axis-major `[4, t_len]` mRoPE positions (t, h, w, extra).
    mrope: Vec<u32>,
    /// per-token causal visibility bound (image rows share their span's last).
    bound: Vec<u32>,
    /// `(seq_offset, n_img)` per image chunk, in order - where to inject the
    /// vision embeddings over the placeholder rows.
    splices: Vec<(usize, usize)>,
    t_len: usize,
    /// the llama-position the first decoded token continues from.
    final_mrope_pos: usize,
}

/// Walk the interleaved chunk list in order and lay out the fused prompt for an
/// arbitrary number of images. A running position cursor advances by 1 per text
/// token and, per image, anchors the whole grid at the cursor (t constant, h/w
/// over the grid) then advances by `max(nx, ny)` - exactly the single-image
/// recipe (`p0` / `after_base`) generalized, so a one-image prompt lays out
/// bit-for-bit as before. `grids` are the `(nx, ny)` merged-grid dims of each
/// encoded image, in the order the `Image` chunks appear. Pure (no device
/// state) so the layout math is unit-testable off-GPU.
fn build_mm_layout(
    chunks: &[crate::service::MmChunk],
    grids: &[(usize, usize)],
) -> Result<MmLayout, GpuModelError> {
    use crate::service::MmChunk;
    let mut ids: Vec<u32> = Vec::new();
    let (mut ts, mut hs, mut ws): (Vec<u32>, Vec<u32>, Vec<u32>) =
        (Vec::new(), Vec::new(), Vec::new());
    let mut bound: Vec<u32> = Vec::new();
    let mut splices: Vec<(usize, usize)> = Vec::new();
    let mut pos: u32 = 0; // mRoPE position cursor
    let mut img_k = 0usize; // index into `grids`
    for c in chunks {
        match c {
            MmChunk::Text(t) => {
                for &tok in t {
                    let seq = ids.len() as u32;
                    ids.push(tok);
                    ts.push(pos);
                    hs.push(pos);
                    ws.push(pos);
                    bound.push(seq);
                    pos += 1;
                }
            }
            MmChunk::Image { .. } => {
                let &(nx, ny) = grids.get(img_k).ok_or_else(|| {
                    GpuModelError::Unsupported("image chunk without an encoded image".into())
                })?;
                img_k += 1;
                let n_img = nx * ny;
                if n_img == 0 {
                    return Err(GpuModelError::Unsupported("empty image grid".into()));
                }
                let base = pos;
                let img_start = ids.len();
                let img_last = (img_start + n_img - 1) as u32;
                splices.push((img_start, n_img));
                for j in 0..n_img {
                    ids.push(0);
                    ts.push(base);
                    hs.push(base + (j / nx) as u32);
                    ws.push(base + (j % nx) as u32);
                    bound.push(img_last);
                }
                pos = base + nx.max(ny) as u32;
            }
            MmChunk::Audio { .. } => {
                return Err(GpuModelError::Unsupported(
                    "qwen35 serves images, not audio - routing bug".into(),
                ));
            }
            MmChunk::OcrCrop(_) => {
                return Err(GpuModelError::Unsupported(
                    "OCR crop directive on qwen35 - routing bug".into(),
                ));
            }
            MmChunk::VisionPixels { .. } => {
                return Err(GpuModelError::Unsupported(
                    "pixel-budget directive on qwen35 - routing bug".into(),
                ));
            }
        }
    }
    if img_k != grids.len() {
        return Err(GpuModelError::Unsupported(format!(
            "{} image chunk(s) but {} encoded image(s)",
            img_k,
            grids.len()
        )));
    }
    if splices.is_empty() {
        return Err(GpuModelError::Unsupported(
            "multimodal prompt has no image".into(),
        ));
    }
    let t_len = ids.len();
    if t_len == 0 {
        return Err(GpuModelError::Unsupported(
            "multimodal prompt is empty".into(),
        ));
    }
    let mut mrope = vec![0u32; 4 * t_len];
    for i in 0..t_len {
        mrope[i] = ts[i];
        mrope[t_len + i] = hs[i];
        mrope[2 * t_len + i] = ws[i];
        mrope[3 * t_len + i] = 0;
    }
    Ok(MmLayout {
        ids,
        mrope,
        bound,
        splices,
        t_len,
        final_mrope_pos: pos as usize,
    })
}

/// The content hash of every image chunk, in the order `build_mm_layout` walks
/// them - so `hashes[k]` pairs with `splices[k]`.
fn mm_image_hashes(chunks: &[crate::service::MmChunk]) -> Vec<u64> {
    chunks
        .iter()
        .filter_map(|c| match c {
            crate::service::MmChunk::Image { rgb, w, h } => {
                Some(multimodal::image_content_hash(rgb, *w, *h))
            }
            _ => None,
        })
        .collect()
}

/// The prefix radix's key vector for a multimodal prompt.
///
/// Not `lay.ids`: every image row there is the same `0` placeholder, so a radix
/// keyed on ids would match one picture's pages against another's and serve the
/// wrong image's KV. Text rows key as themselves; each image row keys off its
/// picture's content hash and its offset within that picture, in a value range
/// provably disjoint from token ids - see
/// [`crate::gpu_model::prefix_cache::image_key_row`].
fn mm_radix_keys(lay: &MmLayout, hashes: &[u64]) -> Vec<u32> {
    debug_assert_eq!(lay.splices.len(), hashes.len(), "one hash per image chunk");
    let mut keys = lay.ids.clone();
    for (k, &(off, n)) in lay.splices.iter().enumerate() {
        for j in 0..n {
            keys[off + j] = crate::gpu_model::prefix_cache::image_key_row(hashes[k], j);
        }
    }
    keys
}

/// Persistent per-layer state for incremental decode. Full-attn layers keep a KV
/// cache; DeltaNet layers keep the [n_v, s, s] recurrent matrix state + the conv
/// window (last k-1 pre-conv inputs). Advanced one token per `step`.
struct DecodeState {
    pos: usize,
    /// The llama-position (mrope scalar for text; all four axes equal) - equals
    /// `pos` for text-only sequences, but after an image chunk it advances by
    /// max(grid_x, grid_y) instead of the image row count, so it diverges.
    mrope_pos: usize,
    kv_k: Vec<Option<CudaSlice<u8>>>,
    kv_v: Vec<Option<CudaSlice<u8>>>,
    recur: Vec<Option<CudaSlice<f32>>>,
    conv_win: Vec<Option<CudaSlice<f32>>>,
    // Device-resident per-token inputs - updated by a tiny htod before each graph
    // replay so the captured graph (fixed addresses) sees the new token/position.
    d_token: CudaSlice<u32>,
    d_pos: CudaSlice<u32>,
    d_slots: CudaSlice<u32>,
    d_mrope: CudaSlice<u32>,
    /// Generated-token ring for the graph-resident loop: the argmax epilogue writes
    /// each new id at `d_step` and bumps it; the host reads a chunk out and resets it.
    d_out: CudaSlice<u32>,
    d_step: CudaSlice<u32>,
    /// Scratch for the parallel argmax pass-1 partials ([ARGMAX_PARTS] max val + idx).
    d_pmax: CudaSlice<f32>,
    d_pidx: CudaSlice<u32>,
    /// MTP draft head's own KV cache (models with a nextn block).
    mtp_kv_k: Option<CudaSlice<u8>>,
    mtp_kv_v: Option<CudaSlice<u8>>,
    /// h_nextn carry: the last committed position's post-out_norm hidden - the h
    /// input paired with the next token (zeroed for position 0, per b9895).
    pending_h: CudaSlice<f32>,
    /// The captured per-token compute (embed..lm_head), replayed once per token by
    /// the host `step` path (prefill / Generator seam). Survives `reset` (the
    /// decode buffers keep their allocations); dropped on scratch reallocation.
    graph: Option<SendGraph>,
    /// Same compute plus the on-device argmax+advance epilogue - the graph-resident
    /// generation loop replays this back-to-back with no per-token host round-trip.
    graph_gen: Option<SendGraph>,
    /// Prefill inputs at fixed addresses so a captured prefill graph stays valid:
    /// positions carry 0..max_ctx once (any prompt length reads a valid prefix,
    /// prefill is always position 0), slots stay zeroed, tokens and the
    /// t_len-strided mrope layout are re-uploaded per call.
    d_pf_tokens: CudaSlice<u32>,
    d_pf_pos: CudaSlice<u32>,
    d_pf_slots: CudaSlice<u32>,
    d_pf_mrope: CudaSlice<u32>,
    /// One captured prefill pass per prompt length (grids bake in t_len) - the
    /// ~700 eager launches of a pass collapse into one submit on replay. P6k
    /// measured the eager launch overhead at ~3.5 ms/pp512, llama replays one
    /// graph. Survives `reset`; cleared on scratch reallocation.
    pf_graphs: std::collections::HashMap<usize, SendGraph>,
}

/// Generated tokens read back per graph-resident chunk. The chunk bounds how far the
/// loop can overrun a stop token (trimmed on the host) and how stale the position
/// bookkeeping gets; 64 keeps the per-token host cost negligible.
const GEN_CHUNK: usize = 64;

/// Continuous-batching state: per-SLOT persistent sequence state (the vLLM
/// axis). Slot i's KV lives at region i of each per-layer cache; DeltaNet
/// recurrent states and conv windows are slot-strided in one buffer so the
/// slot-indexed kernels can scatter/gather. Generator contract: batch row i
/// drives slot i, so the slots buffer is the identity.
struct BatchState {
    max_batch: usize,
    kv_k: Vec<Option<CudaSlice<u8>>>,
    kv_v: Vec<Option<CudaSlice<u8>>>,
    /// per DeltaNet layer: [max_batch, n_v, s, s]
    recur: Vec<Option<CudaSlice<f32>>>,
    /// per DeltaNet layer: [max_batch, k-1, conv_dim]
    conv_win: Vec<Option<CudaSlice<f32>>>,
    d_tokens: CudaSlice<u32>,
    d_pos: CudaSlice<u32>,
    /// identity 0..max_batch (row i = slot i)
    d_slots: CudaSlice<u32>,
    d_mrope: CudaSlice<u32>,
    d_logits: CudaSlice<f32>,
    /// device-sampling scratch: [max_batch, 4] packed per-row sampler params
    /// (inv_t bits, uniform bits, mode, pad) and [max_batch] sampled token ids.
    /// Let `forward_batch_sampled` return bare ids instead of the [B, vocab]
    /// logits readback (25.7 MB/step at B=32).
    d_samp_par: CudaSlice<u32>,
    d_samp_out: CudaSlice<u32>,
    /// P65 host-head sampling: [max_batch, 64, 2] u32 (token id, raw-logit
    /// bits) written by the pd_topk_rows prefilter for TruncCat rows - the
    /// host samples the compact head instead of a [vocab] row readback
    /// (993 KB/row at qwen3.8; the 21.3 ms/round c32 host seam).
    d_samp_head: CudaSlice<u32>,
    /// fin twin (span-side finisher rows sample in d_fin_* buffers)
    d_fin_head: CudaSlice<u32>,
    /// P67 full-device sampling: [max_batch, 4] u32 {k, top_p bits, min_p
    /// bits, pad} side plane for mode-5 rows (pd_sample_rows_t)
    d_samp_tpar: CudaSlice<u32>,
    d_fin_tpar: CudaSlice<u32>,
    /// P67b pipe ring twin: [2, max_batch, 4] - tick N+1's trunc params
    /// never race tick N's in-flight sampler (same double-buffer rule as
    /// d_pipe_par)
    d_pipe_tpar: CudaSlice<u32>,
    /// Pipelined-decode double buffers: 2 rings so tick N+1's sampler-param
    /// upload / sampled-id write never race tick N's still-in-flight readback.
    /// `d_pipe_par` is [2, max_batch, 4] packed params; `d_pipe_out` is
    /// [2, max_batch] sampled ids, read back via the side stream per tick.
    d_pipe_par: CudaSlice<u32>,
    d_pipe_out: CudaSlice<u32>,
    /// Span-side finisher sampling: when a unified tick carries no decode
    /// rows (b == 0 - always true for overlap spans), finishing prompts
    /// stage their last-row logits and sample here instead of in
    /// `d_logits`/`d_samp_*`, whose rows belong to the decode graph that
    /// the overlap scheduler replays concurrently on the decode lane. Same
    /// logits, same sampler - bit-identical ids, disjoint staging.
    d_fin_logits: CudaSlice<f32>,
    d_fin_par: CudaSlice<u32>,
    d_fin_out: CudaSlice<u32>,
    /// int8 MMQ activation staging sized for the batch
    d_xq: CudaSlice<i8>,
    d_xs: CudaSlice<f32>,
    /// K-split mma partial planes (8 x batch x max out_dim) for the B>=KS rung
    d_ks_part: CudaSlice<f32>,
    /// Last-CTA arrival counters for the b=1 lin GEMV's fused K-split combine
    /// (non-KV-overhead R2.2). One region per plane SHAPE - the wrap value is
    /// that shape's elected split, so regions must not be shared. Zeroed once
    /// here; the kernel's atomicInc wraps it back to 0 every launch.
    d_lin_tick: CudaSlice<u32>,
    /// Fused gate|up GEMM landing ([max_batch, 2*ff]) - Some only when fused
    /// planes loaded (the fusion program).
    d_gu_fused: Option<CudaSlice<f32>>,
    /// Fused DN in_qkv|gate_w GEMM landing ([max_batch, conv_dim+value_dim]).
    d_dn_fused: Option<CudaSlice<f32>>,
    /// a zeroed [n_v*s*s] staging block for per-slot state resets on re-prefill
    d_zero_state: CudaSlice<f32>,
    d_zero_win: CudaSlice<f32>,
    /// window-extended pre-conv rows [k-1 + max_ctx, conv_dim] and the conv
    /// output over them - the spec-path pattern generalized to slot prefill,
    /// so a chunk can resume mid-sequence with the slot's window as context
    d_conv_ext: CudaSlice<f32>,
    d_conv_out: CudaSlice<f32>,
    /// two staged DeltaNet checkpoint blobs (per linear layer: [n_v,s,s]
    /// state then [k-1,conv_dim] window), snapshotted at the last two page
    /// boundaries during prefill, inserted into the prefix cache after
    d_ckpt_stage: Vec<CudaSlice<f32>>,
    /// per-slot llama-position offset: mrope = kv_pos + delta. 0 for text
    /// sequences; an image advances mrope by max(grid) instead of its row
    /// count, so multimodal slots carry a (constant) negative delta. Applied
    /// host-side wherever batched mrope is built - decode steps and spec rows.
    mrope_delta: Vec<i64>,
    /// P3 paged KV (default on with a pack that ships the paged kernels;
    /// `PADDOCK_NO_PAGED_KV`/`PADDOCK_DENSE_KV` pin dense).
    /// When active, decode reads/writes the full-attn KV through per-slot block
    /// tables instead of the dense `slot*max_ctx` stride. The pool is the same
    /// `kv_k`/`kv_v` buffers viewed as `[max_batch*bps, 16, kv_dim]`, and the
    /// table is identity-contiguous (`bt[s*bps+j]=s*bps+j`) so the byte layout -
    /// and thus every decoded token - is identical to the dense path. The
    /// non-contiguous, budget-sized pool that actually saves VRAM is P5; this
    /// step proves the block-table plumbing end-to-end, bit-for-bit.
    paged: bool,
    blocks_per_slot: usize,
    d_block_tables: Option<CudaSlice<u32>>,
    /// Drafter-row coverage per state checkpoint: `idx` present => the pages
    /// under that checkpoint carry the drafter's pool-stripe KV rows for
    /// [0..ckpt_pos) (the writer's warm chain covered them with these exact
    /// tokens). Checked at prefix resume: covered => the drafter keeps
    /// drafting at full fidelity on a cross-slot radix hit; absent => the
    /// slot serves dense, the old behavior for an unwarmed span. Idx slots
    /// are recycled by `attach_state`, so every attach must insert OR remove
    /// - a stale bit on a recycled idx would bless garbage drafter rows.
    mtp_cover: std::collections::HashSet<u32>,
    /// DFlash-drafter coverage per state checkpoint, same contract as
    /// `mtp_cover`: `idx` present => the pages under that checkpoint carry
    /// the drafter's feature-KV pool-stripe rows for [0..ckpt_pos) with
    /// exactly the checkpoint's tokens (the writer's ring covered them from
    /// position 0). Checked at prefix resume; every attach must insert OR
    /// remove (recycled idx slots must never bless garbage feature rows).
    dflash_cover: std::collections::HashSet<u32>,
    /// P5 budget pool: a shared free-list of physical blocks + a per-slot block
    /// table that grows from it on demand, so total full-attn KV follows a block
    /// budget (`PADDOCK_KV_POOL_BLOCKS`) rather than `max_batch × max_ctx`. `None`
    /// = identity mode (P4): `d_block_tables` is the fixed identity map and
    /// `kv_k`/`kv_v` are the full dense reservation. `block_table_host` mirrors
    /// `d_block_tables` for cheap whole-table re-upload on growth.
    pool: Option<KvPool>,
    tables: Vec<BlockTable>,
    block_table_host: Vec<u32>,
    /// P5c zero-copy radix prefix cache over the pool: a new sequence sharing a
    /// prompt prefix ADOPTS the cached full-attn KV blocks (refcount++) instead
    /// of recomputing/copying them, and restores the hybrid DeltaNet recurrent
    /// state at the resume boundary from `d_state_pool`. `None` when the pool is
    /// off or `PADDOCK_NO_PREFIX_CACHE` is set.
    paged_prefix: Option<PagedRadix>,
    /// KV tier over the full-attn pool + DeltaNet checkpoint blobs as aux
    /// components (kv-offload 1b.3, the tier-1 family). None unless the
    /// `[kv_offload]` config / dev flag arms it.
    tier: Option<crate::kv_tier::PoolTier<crate::kv_tier::RamTransport>>,
    /// Device pool of DeltaNet state checkpoints (f32), `state_ckpt_f32` per
    /// checkpoint, indexed by `PagedRadix` state indices.
    d_state_pool: Option<CudaSlice<f32>>,
    state_ckpt_f32: usize,
    /// Caller-owned descriptor scratch for the checkpoint snapshot/restore
    /// batched copy. It exists because both used to build their descriptor
    /// list with `clone_htod`, i.e. a fresh device ALLOCATION per slot per
    /// call - 32 of them in one tick when a c32 cohort all resume, and
    /// cudaMalloc is a synchronizing call. That is what made the prefix
    /// cache cost ~1 s tick-stalls and turned the c32 cell bimodal
    /// (2690 good mode vs 2355 with the stalls). Sized for one call's
    /// descriptors (6 u64 per linear layer) and reused forever.
    d_ckpt_desc: Option<CudaSlice<u64>>,
    /// One captured step graph per batch size (grid shapes bake in B): replaying
    /// it collapses the ~500 launches + ~26 quantize/MMQ pairs into one submit -
    /// the low-batch host-loop tax. Invalidated when scratch reallocates.
    graphs: std::collections::HashMap<usize, SendGraph>,
    /// Persistent prefill-pass buffers (chunk-tick graph capture):
    /// the per-pass device inputs/staging whose addresses a captured
    /// `prefill_batch_pass` graph bakes. Taken as owned locals for the pass
    /// (the walk code is unchanged) and put back after; grow-only - any
    /// growth clears `pf_pass_graphs` (its baked addresses died).
    pf_bufs: Option<PfPassBufs>,
    /// Captured admission-pass graphs keyed by `(n_shares, bucket)`: an
    /// eligible pass pads every share to the uniform `bucket` row count and
    /// carries true span geometry in device buffer CONTENTS (VL quads, seg
    /// planes, win quads, gather indices), so one graph serves any slot
    /// assignment and any in-bucket length mix. Invalidated when scratch or
    /// `pf_bufs` reallocate.
    pf_pass_graphs: std::collections::HashMap<(usize, usize), SendGraph>,
}

/// See [`BatchState::pf_bufs`]. One field per former per-pass allocation in
/// `prefill_batch_pass`, destructured into locals of the same names for the
/// duration of the pass.
struct PfPassBufs {
    d_pf_dq: CudaSlice<f32>,
    d_pf_dk: CudaSlice<f32>,
    d_pf_dv: CudaSlice<f32>,
    d_pf_g: CudaSlice<f32>,
    d_pf_beta: CudaSlice<f32>,
    d_pf_dattn: CudaSlice<f32>,
    d_pf_qn: CudaSlice<f32>,
    d_pf_attn: CudaSlice<f32>,
    d_seg_slot: CudaSlice<u32>,
    d_seg_bound: CudaSlice<u32>,
    /// iota 0..take_cap, re-uploaded only on growth
    d_seg_pos: CudaSlice<u32>,
    /// varlen chunked-GDN chunk items + span quads
    d_vl: CudaSlice<u32>,
    /// packed short-span recurrence items (stride 8)
    d_items: CudaSlice<u32>,
    /// conv-window VL store span quads (stride 4)
    d_win: CudaSlice<u32>,
    /// graph-epilogue gather indices: each share's true last row
    d_gidx: CudaSlice<u32>,
    d_tokens: CudaSlice<u32>,
    d_pos: CudaSlice<u32>,
    d_slots: CudaSlice<u32>,
    d_mrope: CudaSlice<u32>,
    take_cap: usize,
    r_cap: usize,
    vl_cap: usize,
    items_cap: usize,
    win_cap: usize,
}

/// Blocks for the parallel argmax pass-1 (each scans vocab/ARGMAX_PARTS logits).
const ARGMAX_PARTS: usize = 512;

/// Max rows per speculative verify chunk (1 committed token + up to SPEC_ROWS-1
/// drafts). Bounds the per-layer recurrent-state snapshot buffers.
const SPEC_ROWS: usize = 12;

/// Rows per MTP warm/catch-up chunk (bounds the concat staging buffers).
const WARM_CHUNK: usize = 64;

/// Batch above which the tensor-core MMA GEMM beats the dp4a MT tile. Below this
/// the batch fits 1-2 dp4a weight passes and dp4a is bandwidth-optimal; above it
/// dp4a's INT pipe saturates and its z-tile weight re-reads dominate. Measured
/// crossover on the A6000 is between 32 and 64 (B=32 tie, B=64 MMA 1.8x).
const MMA_MIN_BATCH: usize = 32;

/// Batch from which the K-split mma (pd_q8_0_gemm_mma_ks) wins every qwen
/// projection shape on the 188-SM GB202 (kbench serving ladder: it took all
/// five shapes from B=24 up, and the narrow out_dims from B=8 - see bmm!).
/// A6000-measured behavior is unknown for qwen shapes; the rung sits behind
/// the same crossover data as the rest of the ladder, re-gate before porting.
const KS_MIN_BATCH: usize = 24;

/// Env-tunable override for `KS_MIN_BATCH` (PADDOCK_KS_MIN): the 24 crossover
/// is a 188-SM GB202 measurement with an explicit re-gate note for the A6000
/// (84 SMs fill at lower batch, so the K-split mma likely wins earlier). Floor
/// 2 -- r=1 stays on the GEMV class.
fn ks_min_batch() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_KS_MIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| n >= 2)
            .unwrap_or(KS_MIN_BATCH)
    })
}

/// k-quant K-split crossover: batches above this take `kquant_gemm_mma_ks`
/// (one weight pass), at or below stay on the dp4a MT tile (one z-pass up to
/// 16 rows, bandwidth-optimal). 16 = where dp4a starts re-reading weights;
/// A6000-measured, retune via PADDOCK_KQ_KS_MIN when the shape set changes.
fn kq_ks_min_batch() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_KQ_KS_MIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| n >= 1)
            .unwrap_or(16)
    })
}

/// The multi-column W4A8 GEMV in SERVING is opt-in (PADDOCK_KQ_NC=1) - Off
/// by default: a serve A/B went 4-for-4 against it despite its micro-bench
/// wins, losing both throughput and TTFT at every width tried. The
/// serve-context loss (graph overlap / prefill-phase engagement at r 2..5)
/// is undiagnosed - re-enable only after a per-shape serve-context win is
/// measured. PADDOCK_NO_KQ_NC=1 still pins it off explicitly (back-compat
/// with existing A/B scripts).
fn kq_nc_off() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var_os!("PADDOCK_NO_KQ_NC").is_some()
            || paddock_models::dev_var_os!("PADDOCK_KQ_NC").is_none()
    })
}

/// ab-plane matvec + delta gate: the FUSED single launch when the pack ships
/// it (bit-identical to the pair - pd_matvec_ab_gate preserves the matvec's
/// per-element summation schedule and applies delta_gate_ab's expressions
/// verbatim; two ~5 us launches per DeltaNet layer per tick collapse to
/// one), the exact pair otherwise. Decode-class batches only - at prefill
/// widths the pair's tiled matvec amortizes better.
/// Exact resident-byte + allocation-count summer for the load-time VRAM
/// audit: every CudaSlice counts its true length, every slice is one
/// cudaMalloc. The audit line compares this against the free-VRAM ledger to
/// expose CUDA heap granularity (small planes reserve page-class chunks).
#[derive(Default, Clone, Copy)]
struct AuSum {
    bytes: u64,
    allocs: u32,
}

impl AuSum {
    fn dt(&mut self, t: &DeviceTensor) {
        self.bytes += (t.buf.len() * 4) as u64;
        self.allocs += 1;
    }
    fn q8(&mut self, w: &RepackedQ8) {
        self.bytes += (w.data.len() + w.scale.len()) as u64;
        self.allocs += 2;
    }
    fn kq(&mut self, w: &crate::gpu::RepackedKQ) {
        self.bytes += (w.data.len() + w.scales.len()) as u64;
        self.allocs += 2;
    }
    fn qw(&mut self, w: &QuantW) {
        match w {
            QuantW::Q8(x) => self.q8(x),
            QuantW::Kq(x) => self.kq(x),
        }
    }
    fn expw(&mut self, w: &ExpW) {
        match w {
            ExpW::Q8(x) => self.q8(x),
            ExpW::Kq(x) => self.kq(x),
            // host-mapped: no VRAM, no device allocation to count
            ExpW::KqHost(_) => {}
        }
    }
    fn fp4(&mut self, w: &RepackedMxfp4) {
        self.bytes += (w.data.len() + w.scale.len()) as u64;
        self.allocs += 2;
    }
    fn attn(&mut self, w: &FullAttnWeights) {
        self.qw(&w.wq);
        self.qw(&w.wk);
        self.qw(&w.wv);
        self.qw(&w.wo);
        self.dt(&w.q_norm);
        self.dt(&w.k_norm);
    }
    fn dn(&mut self, w: &DeltaNetWeights) {
        self.qw(&w.in_qkv);
        self.dt(&w.conv_w);
        if let Some(a) = &w.alpha_w {
            self.q8(a);
        }
        if let Some(b) = &w.beta_w {
            self.q8(b);
        }
        if let Some(t) = &w.ab_f32 {
            self.dt(t);
        }
        self.dt(&w.ssm_a);
        self.dt(&w.dt_bias);
        self.dt(&w.ssm_norm);
        self.qw(&w.gate_w);
        self.qw(&w.out_w);
    }
    fn nvf4(&mut self, w: &crate::gpu::Nvf4Plane) {
        self.bytes += (w.data.len() + w.scale.len()) as u64;
        self.allocs += 2;
    }
    fn ffn(&mut self, f: &Ffn) {
        match f {
            Ffn::Dense { gate, up, down } => {
                self.qw(gate);
                self.qw(up);
                self.qw(down);
            }
            Ffn::Nvf4Dense { gate, up, down } => {
                self.nvf4(gate);
                self.nvf4(up);
                self.nvf4(down);
            }
            Ffn::Moe(m) => {
                self.dt(&m.router_w);
                self.expw(&m.gate_exps);
                self.expw(&m.up_exps);
                self.expw(&m.down_exps);
                self.dt(&m.shexp_gate_inp);
                self.qw(&m.shexp_gate);
                self.qw(&m.shexp_up);
                self.qw(&m.shexp_down);
                if let Some(p) = &m.gate_exps_fp4 {
                    self.fp4(p);
                }
                if let Some(p) = &m.up_exps_fp4 {
                    self.fp4(p);
                }
                if let Some(p) = &m.down_exps_fp4 {
                    self.fp4(p);
                }
                // moe_zero_bias is Arc-shared across layers - charged nowhere
                // here (one buffer, opt-in path only)
            }
        }
    }
}

/// The fused ab matvec+gate is OPT-IN (PADDOCK_AB_GATE=1) - Off by default:
/// the fused path CORRUPTS serving output (9B-Q8 greedy
/// text degrades to drivel; PADDOCK_NO_AB_GATE=1 restored exact text in the
/// bisect). Its bit-identity gate passed on a synthetic plane laid out per
/// the KERNEL's assumption - not against the model's real ab plane and
/// call-site contract (suspects: ab row layout, or a consumer reading the
/// d_ab intermediate the fused path never writes). Do not re-enable without
/// a same-weights greedy-parity gate on a real model.
/// DN v-bf16 route gate (PADDOCK_DNC_BF16OPS): mirrors the chunked-dispatch
/// predicate in `prefill_delta_recurrent` exactly, so a vb16-converted v can
/// only ever be consumed by the chunked pipeline (the sequential kernel
/// reads f32 v). Also requires stage1 mma (the bf16 v read lives there).
fn dn_vb16(exec: &GpuExecutor, r: usize, state_size: usize) -> bool {
    static ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let on = *ENV.get_or_init(|| {
        paddock_models::dev_var_os!("PADDOCK_DNC_BF16OPS").is_some_and(|v| v != "0")
            && paddock_models::dev_var_os!("PADDOCK_NO_CHUNKED_DN").is_none()
            && std::env::var_os("PADDOCK_DNC_S1MMA").is_none_or(|v| v != "0")
    });
    let chunk_min = if exec.sm_count() >= 128 { 384 } else { 128 };
    on && exec.has_dn_vb16() && r >= chunk_min && state_size == 128
}

/// QKC compact-bf16 q/k pair gate: engages the slot-446/447
/// pair (conv emits Hg-compact bf16 q/k, the vl chunked-GDN entry reads
/// them - bit-identical values, 12x fewer q/k bytes). Mirrors the rs-route
/// envs load.rs itself sets, so a kill of any of them silently reverts to
/// the expanded pair. The CALLER must additionally require an all-vl tick:
/// every other consumer of d_dq/d_dk (chunked_at, recurrent v2/_packed)
/// reads f32 expanded.
fn dn_qkc(exec: &GpuExecutor) -> bool {
    static ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let on = *ENV.get_or_init(|| {
        paddock_models::dev_var_os!("PADDOCK_DNC_QKC").is_some_and(|v| v != "0")
            && std::env::var_os("PADDOCK_DNC_S1RS").is_none_or(|v| v != "0")
            && std::env::var_os("PADDOCK_DNC_RS").is_none_or(|v| v != "0")
            && std::env::var_os("PADDOCK_DNC_S1MMA").is_none_or(|v| v != "0")
    });
    on && exec.has_gated_delta_chunked_rs_vl_qkc()
}

fn ab_gate_off() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var_os!("PADDOCK_NO_AB_GATE").is_some()
            || paddock_models::dev_var_os!("PADDOCK_AB_GATE").is_none()
    })
}

#[allow(clippy::too_many_arguments)]
fn ab_gate(
    exec: &GpuExecutor,
    ab: &DeviceTensor,
    xn: &CudaSlice<f32>,
    d_ab: &mut CudaSlice<f32>,
    ssm_a: &CudaSlice<f32>,
    dt_bias: &CudaSlice<f32>,
    g: &mut CudaSlice<f32>,
    beta: &mut CudaSlice<f32>,
    n: usize,
    n_heads: usize,
) -> Result<(), GpuModelError> {
    if n < 16 && !ab_gate_off() && exec.has_matvec_ab_gate() {
        exec.matvec_ab_gate(ab, xn, ssm_a, dt_bias, g, beta, n, n_heads)?;
    } else {
        exec.matvec_f32_batch(ab, xn, d_ab, n)?;
        exec.delta_gate_ab(d_ab, ssm_a, dt_bias, g, beta, n, n_heads)?;
    }
    Ok(())
}

/// Batch from which the high-occupancy `q8_0_gemm_mmq_hi` (2 blocks/SM)
/// replaces the synchronous 128x128 mmq for prefill GEMMs (E1c). Generative /
/// serving prefill is chunked to <=512 tokens, so only the UNCHUNKED encoder
/// (embeddings/rerank, batch = thousands of rows) crosses this - the
/// qwen/gpt-oss prefill path is untouched byte-for-byte. Same Q8_0 numeric class.
const MMQ_HI_MIN_BATCH: usize = 1024;

/// Family-elected pipe/hi gate (see `elect_mmq_hi_min`); 0 = no election.
static MMQ_HI_MIN_ELECTED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// A family whose prefill-chunk regime never reaches the 1024-row default
/// elects its own gate at enable time - the apply_default_stack pattern, a
/// measured regime election rather than a serve-time knob. First election:
/// qwen3_asr  - ASR prompts are ~160-380 rows, so every prefill
/// GEMM sat on the synchronous barrier-bound 128x128 mmq - 26.3% of the
/// GPU at wide batch. Electing 128 on sm_120a lifts request throughput
/// sharply at every width. An explicit PADDOCK_MMQ_HI_MIN still wins -
/// it is the A/B pin, and pins must pin.
pub(crate) fn elect_mmq_hi_min(n: usize) {
    MMQ_HI_MIN_ELECTED.store(n.max(64), std::sync::atomic::Ordering::Relaxed);
}

/// Env-tunable override for `MMQ_HI_MIN_BATCH` (PADDOCK_MMQ_HI_MIN): the 1024
/// gate was set when serving prefill was 512-row chunked; today's unified
/// 2048-row chunks already cross it, but 65..1024-row prompts/chunks still run
/// the synchronous barrier-bound 128x128 mmq. This knob lets small dies A/B
/// the cp.async pipe/hi rungs in that band without a rebuild. Resolution:
/// explicit env pin > family election (`elect_mmq_hi_min`) > the 1024 default.
fn mmq_hi_min_batch() -> usize {
    static ENV: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    let env = *ENV.get_or_init(|| {
        std::env::var("PADDOCK_MMQ_HI_MIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n: &usize| n >= 64)
    });
    if let Some(n) = env {
        return n;
    }
    match MMQ_HI_MIN_ELECTED.load(std::sync::atomic::Ordering::Relaxed) {
        0 => MMQ_HI_MIN_BATCH,
        n => n,
    }
}

/// Speculative-decoding state (lazy - only allocated when spec decode runs).
/// Holds the per-layer rollback buffers for the verify pass plus the MTP head's
/// staging: on partial draft acceptance the recurrent state restores from the
/// per-token snapshot and the conv window re-slices from the extended pre-conv
/// rows; the KV caches need no rollback (stale cells past the accepted position
/// are overwritten before any later query can read them).
struct SpecState {
    /// Per-DeltaNet-layer verify-chunk state snapshots [SPEC_ROWS, n_v, s, s].
    recur_snap: Vec<Option<CudaSlice<f32>>>,
    /// Per-DeltaNet-layer extended pre-conv rows [(k-1) + SPEC_ROWS, conv_dim]:
    /// window prefix + the chunk's mixed rows; window-after-row-r = rows r..r+k-1.
    conv_ext: Vec<Option<CudaSlice<f32>>>,
    /// Verify-chunk outputs: logits for every row + h (post-out_norm) rows.
    d_logits_chunk: CudaSlice<f32>,
    d_h_chunk: CudaSlice<f32>,
    /// MTP staging: concat(e_norm || h_norm) rows, embed/norm scratch, h inputs,
    /// and the head's own h output (the draft-time h chain).
    d_concat: CudaSlice<f32>,
    d_e: CudaSlice<f32>,
    d_en: CudaSlice<f32>,
    d_hn: CudaSlice<f32>,
    d_hin: CudaSlice<f32>,
    d_hout: CudaSlice<f32>,
    d_mtp_tok: CudaSlice<u32>,
    /// int8 MMQ activation staging for the verify chunk (quantize_q8 output).
    d_xq: CudaSlice<i8>,
    d_xs: CudaSlice<f32>,
    /// K-split mma partial planes (8 x rows x projection out_dim) - verify GEMMs
    d_ks_part: CudaSlice<f32>,
}

/// Batched (per-slot) speculative-decoding state: B concurrent sequences each
/// draft K tokens with the MTP head, verified in one backbone pass over
/// B×(K+1) slot-major rows. The backbone KV/state/conv live in `BatchState`
/// (slots 0..batch); this adds the per-slot MTP KV, per-slot pending_h, and
/// the ragged-commit rollback buffers. Chunk row order is slot-major:
/// row b*(K+1)+j = slot b's j-th chunk row - the layout the v2 recurrence's
/// (batch, n_tokens) mode and the snapshot indexing both assume.
struct SpecBatchState {
    /// slots covered by the CURRENT round (serving mutates this per round to
    /// the live count; all state arrays are indexed slot-relative so only row
    /// counts change). `alloc_batch` is the allocation bound.
    batch: usize,
    alloc_batch: usize,
    n_draft: usize,
    /// Depth the MTP draft-chain graph is RECORDED at. Equal to n_draft on a
    /// plain MTP serve; with a DFlash drafter attached, n_draft sizes the
    /// verify rows and d_draft for the BLOCK drafter's k (block-1 = 7) while
    /// the chain records only the MTP election's depth (3/4) - the chain
    /// pays per step, and recording it at the block depth made every MTP
    /// fallback round run 7 sequential drafter forwards to use 3 (measured
    /// flat 10.2ms/round vs ~4ms, the attached-serve residual's main course).
    chain_depth: usize,
    /// per-slot committed position (host mirror of the sequence lengths)
    pos: Vec<usize>,
    /// block->TRUE-slot map of the CURRENT round (block i = row block i of
    /// the round buffers; round_slots[i] = the serving slot it belongs to).
    /// Set by every round driver before staging. The round machinery used to
    /// require reqs[i].slot == i (contiguous-from-0), and any
    /// ragged live set - the common case under churn at width - DECLINED
    /// into the mixed tick's chunk lane, where the verify executed as a
    /// 2-6-row EAGER 65-layer pass on 64x64-tile Q8 kernels (~40 ms/round,
    /// ~95% tile padding: the whole c4-c32 spec loss). The kernels were
    /// always map-driven (staged d_slots_rows / bs.d_slots); this map fixes
    /// the HOST-side block-vs-slot indexing so the graphed round serves
    /// arbitrary slot sets. Identity (0..alloc) at rest, so the contiguous
    /// paths (bench drivers, c1) behave bit-identically.
    round_slots: Vec<u32>,
    /// This round's k1 (max chunk len; <= n_draft+1). The verify/commit
    /// records read it at capture - see the graph key note above.
    round_k1: usize,
    /// [alloc_batch, embd] block-gathered pending_h staging: the draft
    /// graph's step-0 h read is a CONTIGUOUS [0, b*embd) copy baked at
    /// capture, while pending_h is TRUE-slot-strided - mtp_draft_b gathers
    /// round_slots' rows in here (b small D2D copies, eager) before launch.
    d_pending_hb: CudaSlice<f32>,
    /// per-slot MTP warm flag: KV + pending_h cover positions 0..pos[slot].
    /// False after a desync (dense ticks advanced the slot without catchup,
    /// prefix-cache resume) - the slot re-warms at its next full prefill.
    mtp_warm: Vec<bool>,
    /// per-slot token shadow of the MTP KV: mtp_toks[slot][0..pos[slot]] are
    /// exactly the tokens whose rows the MTP KV holds. The MTP row at position
    /// p is a deterministic function of (tokens[0..=p], backbone h[0..p]), so
    /// on a prefix-cache resume a shadow match over the reused span proves the
    /// old rows are the replay we'd otherwise recompute - the cursor can
    /// rewind to the resume point instead of the slot going cold for the whole
    /// session (spec died on every multi-turn/agentic reuse before this).
    mtp_toks: Vec<Vec<u32>>,
    /// MTP block KV, slot-strided [batch, max_ctx, kv_dim] fp16
    mtp_kv_k: CudaSlice<u8>,
    mtp_kv_v: CudaSlice<u8>,
    /// [batch, embd] - previous position's post-out_norm h per slot (zeros at start)
    pending_h: CudaSlice<f32>,
    /// per DeltaNet layer: [batch, K+1, n_v, s, s] v2-kernel snapshots (t-major,
    /// transposed tiles). LEGACY verify path only (PADDOCK_QWEN35_SPEC_SNAPSHOT
    /// or a pack without slot 462/463): the snapshot-free path leaves these
    /// None and stashes the round's split/gate planes instead - the
    /// snapshots were ~87% of the ~1.15 GiB per spec-row draft state and
    /// capped spec width at 14 rows.
    recur_snap: Vec<Option<CudaSlice<f32>>>,
    /// per DeltaNet layer (snapshot-free verify): the round's normalized
    /// k_hat / v planes ([batch, K+1, n_v, s]) and gate planes
    /// ([batch, K+1, n_v]) - exactly the v2 inputs, kept until commit so
    /// gated_delta_commit_walk can recompute the accepted-prefix state from
    /// round-start. ~128x smaller than the snapshots they replace.
    vstash_k: Vec<Option<CudaSlice<f32>>>,
    vstash_v: Vec<Option<CudaSlice<f32>>>,
    vstash_g: Vec<Option<CudaSlice<f32>>>,
    vstash_beta: Vec<Option<CudaSlice<f32>>>,
    /// One slot's recurrent state + conv windows across every DeltaNet layer,
    /// saved/restored around a gap re-warm: the re-warm re-runs the
    /// backbone over tokens the live state already consumed, and the
    /// recurrence is not idempotent. Lazily allocated on first use.
    warm_stash: Option<CudaSlice<f32>>,
    /// per DeltaNet layer: [batch, (k-1) + K+1, conv_dim] extended pre-conv rows
    conv_ext: Vec<Option<CudaSlice<f32>>>,
    /// verify outputs: [batch*(K+1)] rows of logits / h / device argmax picks
    d_logits_chunk: CudaSlice<f32>,
    d_h_chunk: CudaSlice<f32>,
    d_row_tok: CudaSlice<u32>,
    /// draft ring, i-major: d_draft[i*batch + b] = slot b's i-th draft
    d_draft: CudaSlice<u32>,
    d_committed: CudaSlice<u32>,
    // MTP staging, sized R_max = max(batch*(K+1), WARM_CHUNK) rows
    d_concat: CudaSlice<f32>,
    d_e: CudaSlice<f32>,
    d_en: CudaSlice<f32>,
    d_hn: CudaSlice<f32>,
    d_hin: CudaSlice<f32>,
    d_hout: CudaSlice<f32>,
    d_mtp_tok: CudaSlice<u32>,
    /// pd_spec_toks meta staging ([5, alloc_batch] u32) + per-verify-block
    /// real chunk lengths - written by the round drivers when an async
    /// chain is armed; forward_chunk_b assembles verify tokens on device
    /// from d_draft through these (see GpuQwen35::spec_chain).
    d_asm_meta: CudaSlice<u32>,
    chain_lens: Vec<u32>,
    /// the round's own block->slot map for the verify/commit graphs' DeltaNet
    /// state/conv routing. bs.d_slots is staged by decode/mixed passes on the
    /// MAIN lane; when the round overlaps a span (the mixed spec tick), both
    /// staging paths racing one buffer would cross-route state - the round
    /// graphs read this instead, staged in forward_chunk_b.
    d_round_slots: CudaSlice<u32>,
    /// the round's own copy of the paged block tables, same layout as
    /// bs.d_block_tables (None off-pool). ensure_slot_blocks re-uploads the
    /// live table on any slot's growth - under the overlapped mixed tick the
    /// span grows its chunk slot's table nearly every tick, and that upload
    /// (main lane) racing the round's paged reads (decode lane) would be a
    /// cross-stream data race even though the round's rows carry identical
    /// bytes. The round's paged kernels read this mirror instead, staged
    /// from block_table_host by stage_spec_tables at every drafter/round
    /// entry point (uploads from the host truth, so it needs no device
    /// ordering vs the live table).
    d_spec_tables: Option<CudaSlice<u32>>,
    /// fused-GEMM landing for the verify's f8 class mirror (qkv | in_qkv+gate
    /// | gate+up all land here before row_slice/swiglu) - the round's twin of
    /// bs.d_gu_fused/d_dn_fused, which are max_batch rows (too small for the
    /// batch*k1-row verify) and shared with the decode graphs
    d_fused_land: CudaSlice<f32>,
    /// per-row position/slot/mrope staging for draft, verify and catch-up passes
    d_pos_rows: CudaSlice<u32>,
    d_slots_rows: CudaSlice<u32>,
    d_mrope_rows: CudaSlice<u32>,
    /// deepest position across the staged rows (host mirror, kept by
    /// stage_spec_rows) - the verify-attention split election's context
    /// hint: the LAUNCHED split count derives from the round's
    /// deepest row, while the kernel's row-local clamp keeps every block's
    /// effective partition a pure function of its own context
    max_pos_row: u32,
    /// int8 MMQ activation staging for the verify chunk
    d_xq: CudaSlice<i8>,
    d_xs: CudaSlice<f32>,
    /// K-split mma partial planes (8 x min(rows,64) x projection out_dim)
    d_ks_part: CudaSlice<f32>,
    /// device-sampling scratch sized for the full verify chunk (R_max rows, not
    /// max_batch): [R_max,4] packed per-row sampler params + [R_max] sampled ids.
    /// The batch's own d_samp_par/out are only max_batch wide - too small for a
    /// batch*(K+1)-row verify (n*k1 up to 32 at c8), which is why the plans path
    /// owns its own here rather than borrowing BatchState's.
    d_samp_par_chunk: CudaSlice<u32>,
    d_samp_out_chunk: CudaSlice<u32>,
    /// truncation stage (b): [R_max, 4] u32 {k, top_p bits, min_p bits, pad} side
    /// plane - TruncCat verify rows sample as mode 5 on the chunk buffers,
    /// so the device-sampled spec round takes truncation slots. No ring
    /// needed: the plans round reads its picks back before the next round
    /// packs (the one-ahead spec PIPE declines trunc rows - device_plan
    /// returns None for them, ending the segment before entry).
    d_samp_tpar_chunk: CudaSlice<u32>,
    /// captured round graphs (K-step draft loop, verify pass, ragged commit) -
    /// they bake scratch/batch/spec buffer addresses, so any realloc of those
    /// (ensure_scratch, enable_batch, enable_spec_batch) must drop them
    /// Round graphs cached per live COUNT (they bake row counts), exactly the
    /// backbone decode's `BatchState::graphs` pattern. These used to be
    /// single slots dropped on every live change - at c8 the live count
    /// churns {1..8} with every finish/admission, so rounds re-instantiated
    /// all three graphs constantly: the spec round cost ~98 ms against a
    /// 24.5 ms dense tick, the closed-loop controller learned those poisoned
    /// latencies and collapsed K to 1, and spec-at-width lost to dense.
    /// keyed (live, round k1): rounds are RAGGED in k since the hybrid
    /// drafter (DFlash2 drafts 7, the MTP chain 3) - padding every round to
    /// the alloc k1 would double the MTP rounds' verify rows
    graph_draft: std::collections::HashMap<(usize, usize), SendGraph>,
    graph_verify: std::collections::HashMap<(usize, usize), SendGraph>,
    graph_commit: std::collections::HashMap<(usize, usize), SendGraph>,
}

/// Flash-decode KV splits per full-attn head, ceiling for [`attn_splits`]. Fixed
/// (not position-dependent) split counts keep a captured graph's grid constant
/// while each block reads the real position from device.
const MAX_ATTN_SPLITS: usize = 32;

/// FlashDecoding block-count target - ~3x the SM count, same shape gpt_oss uses
/// (the G4 lesson: block targets must scale with the die, not bake in one card).
fn attn_fill_blocks(sm_count: usize) -> usize {
    3 * sm_count
}

/// KV splits for decode-class full attention. The unsplit kernel is n_heads
/// blocks - 16 on qwen - which starves a big die at any context (GB202 kbench:
/// 25-29 GB/s effective unsplit at every ctx; split 2x faster at ctx=128, 8.5x
/// at 2k, 10x at 8k). Small dies (<128 SMs) engage a SMALLER fixed count: the
/// A6000 once measured the split slower at short context, but 16 heads on
/// 84 SMs starves that die too, and with the CURRENT kernels the regression
/// no longer reproduces. Re-measured on an A6000 (9b, 4.7k-token prompt
/// decode): short context is neutral at every count, and long context climbs
/// steeply through splits 1/4/8/16 and plateaus past 16. Small dies take 16
/// (32 is noise-equal, half
/// the combine width). PADDOCK_ATTN_SPLITS=<n> overrides on any die.
///
/// The count is a FIXED constant whenever splitting engages, not scaled by
/// batch or position: the per-slot spec gates compare runs at different row
/// counts (single-slot verify r=K+1 vs batched r=B*(K+1)) and demand
/// token-identical output - a batch-scaled count gives the same row a
/// different combine reduction order on each side and flips marginal tokens
/// (measured: the B=4 spec-batch gate diverged at token 16 with a scaled
/// count), and a position-gated count would let other rows' positions change
/// this row's reduction order in a mixed batch. Fixed count +
/// position-independence also keeps captured graphs replay-safe (the partial
/// kernel reads each row's position from device at replay).
///
/// Collapse to the single-kernel walk only past 2x the fill target, where the
/// unsplit grid saturates the die anyway - this also bounds the partial
/// scratch. Rows crossing that boundary leave the exactness envelope, same
/// spirit as the B*(K+1) spec ceiling. PADDOCK_NO_ATTN_SPLIT=1 pins the old
/// single-kernel path for A/B.
fn attn_splits(n_heads: usize, batch: usize, sm_count: usize) -> usize {
    if paddock_models::dev_var_os!("PADDOCK_NO_ATTN_SPLIT").is_some() {
        return 1;
    }
    if n_heads * batch >= 2 * attn_fill_blocks(sm_count) {
        return 1;
    }
    if let Some(n) = paddock_models::dev_var!("PADDOCK_ATTN_SPLITS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n >= 1)
    {
        return n.min(MAX_ATTN_SPLITS);
    }
    // Big dies took MAX (32) at first; a ladder at a wide serve point with a
    // ~2k prefix and fp8 KV measured 32/16/8/4 as a shallow curve peaking
    // around 8: 32-way splits leave each krs CTA ~64 keys of work and the
    // fixed cost + combine width dominate. 16 banks most of that win while
    // staying a FIXED constant (the law above), and single-stream long-prompt
    // decode measures flat across the whole range - decode attention is a
    // small share there. The measured optimum 8 is PARKED: landing it costs
    // single-stream parallelism at b=1 unless the count tiers by rows, and a
    // rows-tiered count needs the spec-verify paths pinned to one count first
    // (the B=8 K=4 exact gate compares r=40 batched vs r=5 single-slot).
    16
}

/// `CapturedGraph` holds raw CUDA handles and is `!Send`, but the model is driven from
/// a single thread at a time (the engine service owns it on its worker thread), so
/// moving the graph across that one handoff - and never using it concurrently - is
/// sound. This wrapper asserts exactly that.
pub(crate) struct SendGraph(pub(crate) crate::gpu::CapturedGraph);
// SAFETY: the model is never accessed from two threads at once; see above.
unsafe impl Send for SendGraph {}

/// Per-pass device scratch, sized to `cap` tokens. Correctness-first: the whole
/// sequence is (re)processed each forward (fresh DeltaNet state + fresh KV), so
/// there is no cross-call state to carry - the incremental recurrent-state cache
/// is a P3 perf item. All buffers are reused across layers within a pass.
struct Scratch {
    cap: usize,
    // shared residual / norm / mixer-output
    d_x: CudaSlice<f32>,
    d_xn: CudaSlice<f32>,
    d_proj: CudaSlice<f32>,
    // full-attn
    d_qg: CudaSlice<f32>,
    d_q: CudaSlice<f32>,
    d_qn: CudaSlice<f32>,
    d_gate: CudaSlice<f32>,
    d_k: CudaSlice<f32>,
    d_kn: CudaSlice<f32>,
    d_v: CudaSlice<f32>,
    d_attn: CudaSlice<f32>,
    // flash-decode partials: [n_heads, ATTN_SPLITS, head_dim] and [n_heads, ATTN_SPLITS, 2]
    d_attn_o: CudaSlice<f32>,
    d_attn_ml: CudaSlice<f32>,
    // DeltaNet
    d_mixed: CudaSlice<f32>,
    d_conv: CudaSlice<f32>,
    d_dq: CudaSlice<f32>,
    d_dk: CudaSlice<f32>,
    d_dv: CudaSlice<f32>,
    d_a: CudaSlice<f32>,
    d_b: CudaSlice<f32>,
    /// x2-v3 fused decay activations [cap, 2*n_v_heads] (alpha||beta rows)
    d_ab: CudaSlice<f32>,
    d_g: CudaSlice<f32>,
    d_beta: CudaSlice<f32>,
    d_dattn: CudaSlice<f32>,
    d_z: CudaSlice<f32>,
    d_core: CudaSlice<f32>,
    // chunked-scan DeltaNet prefill scratch (P6l), sized per ceil(cap/64)
    // chunks: the stage-1 substituted right-hand sides dw/du [nc, H, 64, D],
    // the pre-folded output coefficients [nc, H, 64, 64], and the f64
    // cumulative log-decay [nc, H, 64]. Sized for the tick's TOTAL varlen
    // chunks (cap/64 + one padding chunk per span, 32 spans max) so the
    // varlen chunked-GDN launch can pack every span's chunks side by side.
    d_dnc_dw: CudaSlice<f32>,
    d_dnc_du: CudaSlice<f32>,
    d_dnc_coef: CudaSlice<f32>,
    d_dnc_cg: CudaSlice<f64>,
    // FFN
    d_ffn_gate: CudaSlice<f32>,
    // MoE scratch (1-element dummies on dense models)
    d_moe_logits: CudaSlice<f32>,
    d_moe_idx: CudaSlice<u32>,
    d_moe_w: CudaSlice<f32>,
    d_moe_fused: CudaSlice<f32>,
    d_moe_fq: CudaSlice<i8>,
    d_moe_fs: CudaSlice<f32>,
    d_moe_xq: CudaSlice<i8>,
    d_moe_xs: CudaSlice<f32>,
    // b2: u8 (ue8m0) scale planes for the fp4 W4A8 block-scale MoE (bs kernels):
    // e4m3 activation scales (d_moe_xs8) and e4m3 fused-output scales (d_moe_fs8).
    // Byte-count twins of d_moe_xs / d_moe_fs. Unused unless fp4 MoE is loaded.
    d_moe_xs8: CudaSlice<u8>,
    d_moe_fs8: CudaSlice<u8>,
    d_zero_bias: CudaSlice<f32>,
    // sorted-MoE staging (moe_align layout)
    d_moe_srow: CudaSlice<u32>,
    d_moe_sslot: CudaSlice<u32>,
    d_moe_bexp: CudaSlice<u32>,
    d_moe_part: CudaSlice<f32>,
    d_ffn_up: CudaSlice<f32>,
    // prefill int8 activation staging for the tensor-core MMA GEMM (reads the
    // weight once as int8, no dequant-to-f16 write). Sized to the widest
    // activation (cap * qw, where qw = max projection input dim).
    /// K-split partial planes for the b=1 f8d PPL-scoring route (8 x out_max)
    d_f8_part: CudaSlice<f32>,
    d_pxq: CudaSlice<i8>,
    /// f8t (tcgen05) activation scratch - 64 rows wide, see the alloc comment:
    /// f8t_gemm's TMA boxes read past `batch`, so this cannot share d_pxq.
    d_f8t_q: CudaSlice<i8>,
    d_f8t_rs: CudaSlice<f32>,
    /// nv4cut activation scratch: e2m1 nibbles and the BLOCKED scale plane
    /// the CUTLASS mainloop reads. Caller-owned and address-stable because
    /// the decode graphs bake addresses (see the batch.rs note) - a lazy
    /// allocation inside the kernel entry cost ~40x of wide-batch throughput
    d_pxs: CudaSlice<f32>,
    // e4m3 per-32-block activation scales (u8, one exponent byte per block) for
    // the fp8 W8A8 projection GEMM (b1). Paired with d_pxq reinterpreted as e4m3
    // bytes; sized like d_pxs (cap * qw / 32). Unused unless W8 planes are loaded.
    d_exs: CudaSlice<u8>,
    // nvf4 per-16 activation scales (u8 e4m3, one byte per 16 elems) for the
    // W4A4 dense-projection GEMM (`mxfp4_gemm_nv4`) - Twice the e4m3 scale count,
    // so sized cap*qw/16. Paired with d_pxq reinterpreted as packed fp4 nibbles.
    // Unused unless nv4 planes are loaded (PADDOCK_QWEN35_PROJ_NV4).
    d_nvs: CudaSlice<u8>,
    // split-K partials for the checkpoint W4A4 GEMM:
    // sk(4) x 128 x embd f32 - the split engages only on tile grids under
    // 64 CTAs, which in this graph is the FFN down plane (out = embd) at
    // decode-band batches (<=128 rows). 1-elem stub off the Nvf4Dense lane.
    d_nv4part: CudaSlice<f32>,
    // flat mmq-layout activation staging for the P6e GEMM: [chunk][col][4 f32
    // scales + 128 int8], columns padded to a multiple of 128.
    d_yq: CudaSlice<u8>,
    // per-32-block activation sums off d_yq ([chunk][col_pad][4] f32) - the
    // W4A8 min-term operand (Q4_K/Q5_K prefill). 1-elem stub on pure-Q8 models.
    d_xsums: CudaSlice<f32>,
    // per-16 sums off the STRIDED int8 staging ([col][in/16] f32) - the dp4a
    // decode/small-prefill ladder's min-term operand. 1-elem stub on pure-Q8.
    d_ssums: CudaSlice<f32>,
    // stream-k fixup scratch for the P6e GEMM (the 256-SM sizing contract)
    d_skfix: CudaSlice<f32>,
    // final
    d_logits: CudaSlice<f32>,
    /// K-split partials for the one-row f8 lm_head on the paths with no
    /// BatchState to borrow (vision prefill, record_prefill/record_step).
    /// f8_gemm_lin wants >= 8 * out_dim * batch f32 and the head is the
    /// widest out_dim, which bs.d_ks_part's 8*64*ks_out_max was never sized
    /// for. ~8 MB, so the REPLACE lane holds on every path.
    d_head_part: CudaSlice<f32>,
    /// h_nextn rows: post-out_norm hidden for every position of the last batched
    /// pass - the MTP draft head's h inputs (b9895: h = result_norm rows).
    d_h: CudaSlice<f32>,
    /// K-quant exact-f32 prefill fallback (PADDOCK_KQ_F32_PREFILL=1, triage
    /// only): transient f32 dequant of one layer weight at a time
    /// (kq_max_elems). 1-elem stub otherwise - W4A8 is the default kq prefill.
    d_wdq: CudaSlice<f32>,
}

/// Input-embedding table, quantized-resident with per-tensor dispatch (rows
/// dequant on the fly in both arms - same values as full dequant).
enum TokEmbd {
    Q8(QuantTensor),
    Kq(RepackedKQ),
}

impl TokEmbd {
    /// Resident device bytes (the VRAM ledger line).
    fn resident_bytes(&self) -> usize {
        match self {
            TokEmbd::Q8(t) => t.bytes.len(),
            TokEmbd::Kq(t) => t.data.len() + t.scales.len(),
        }
    }
    fn label(&self) -> &'static str {
        match self {
            TokEmbd::Q8(_) => "Q8_0",
            TokEmbd::Kq(t) => match t.ty {
                paddock_models::ggml_type::GgmlType::Q4K => "Q4_K",
                paddock_models::ggml_type::GgmlType::Q5K => "Q5_K",
                paddock_models::ggml_type::GgmlType::Q6K => "Q6_K",
                _ => "IQ4_XS",
            },
        }
    }
}

// Embedding row-gather with per-tensor dispatch.
#[cfg(test)]
mod mm_layout_tests {
    //! CPU gate for the multi-image prompt layout. The single-image case is
    //! pinned bit-for-bit against the pre-multi-image formula (kept verbatim as
    //! the oracle), so lifting the one-image cap provably does not perturb an
    //! existing one-image request; multi-image is checked against hand-computed
    //! expectations. No GPU/model needed - [`build_mm_layout`] is pure.
    use super::{MmLayout, build_mm_layout};
    use crate::service::MmChunk;

    fn img() -> MmChunk {
        // build_mm_layout ignores the raw bytes - it reads grid dims from `grids`
        MmChunk::Image {
            rgb: Vec::new(),
            w: 0,
            h: 0,
        }
    }

    /// The exact single-image recipe as it stood before multi-image, the oracle
    /// the generalized walk must reproduce for any one-image prompt.
    fn single_image_oracle(before: &[u32], nx: usize, ny: usize, after: &[u32]) -> MmLayout {
        let n_img = nx * ny;
        let t_len = before.len() + n_img + after.len();
        let mut ids = Vec::with_capacity(t_len);
        ids.extend_from_slice(before);
        ids.extend(std::iter::repeat_n(0u32, n_img));
        ids.extend_from_slice(after);
        let p0 = before.len() as u32;
        let img_last_row = (before.len() + n_img - 1) as u32;
        let after_base = p0 + nx.max(ny) as u32;
        let mut mrope = vec![0u32; 4 * t_len];
        let mut bound = vec![0u32; t_len];
        for i in 0..t_len {
            let (t, h, w) = if i < before.len() {
                (i as u32, i as u32, i as u32)
            } else if i < before.len() + n_img {
                let j = i - before.len();
                (p0, p0 + (j / nx) as u32, p0 + (j % nx) as u32)
            } else {
                let k = (i - before.len() - n_img) as u32;
                (after_base + k, after_base + k, after_base + k)
            };
            mrope[i] = t;
            mrope[t_len + i] = h;
            mrope[2 * t_len + i] = w;
            mrope[3 * t_len + i] = 0;
            bound[i] = if i >= before.len() && i < before.len() + n_img {
                img_last_row
            } else {
                i as u32
            };
        }
        let final_mrope_pos = after_base as usize + after.len();
        MmLayout {
            ids,
            mrope,
            bound,
            splices: vec![(before.len(), n_img)],
            t_len,
            final_mrope_pos,
        }
    }

    #[test]
    fn single_image_matches_legacy_formula() {
        for &(nx, ny) in &[(2usize, 3usize), (3, 2), (4, 4), (1, 5), (5, 1)] {
            for (before, after) in [
                (vec![10u32, 11, 12], vec![20u32, 21]),
                (vec![], vec![9u32]),    // empty before
                (vec![7u32, 8], vec![]), // empty after
            ] {
                let chunks = vec![
                    MmChunk::Text(before.clone()),
                    img(),
                    MmChunk::Text(after.clone()),
                ];
                let got = build_mm_layout(&chunks, &[(nx, ny)]).unwrap();
                let want = single_image_oracle(&before, nx, ny, &after);
                assert_eq!(got.ids, want.ids, "ids nx={nx} ny={ny}");
                assert_eq!(got.mrope, want.mrope, "mrope nx={nx} ny={ny}");
                assert_eq!(got.bound, want.bound, "bound nx={nx} ny={ny}");
                assert_eq!(got.splices, want.splices, "splices nx={nx} ny={ny}");
                assert_eq!(got.t_len, want.t_len, "t_len nx={nx} ny={ny}");
                assert_eq!(
                    got.final_mrope_pos, want.final_mrope_pos,
                    "pos nx={nx} ny={ny}"
                );
            }
        }
    }

    #[test]
    fn two_images_lay_out_in_order() {
        // Text(2) | Image 2x2 | Text(1) | Image 1x3 | Text(2)
        let chunks = vec![
            MmChunk::Text(vec![1, 2]),
            img(),
            MmChunk::Text(vec![3]),
            img(),
            MmChunk::Text(vec![4, 5]),
        ];
        let l = build_mm_layout(&chunks, &[(2, 2), (1, 3)]).unwrap();

        // rows: [1,2, 0,0,0,0, 3, 0,0,0, 4,5]
        assert_eq!(l.t_len, 12);
        assert_eq!(l.ids, vec![1, 2, 0, 0, 0, 0, 3, 0, 0, 0, 4, 5]);
        // image spans at seq offsets 2 (4 rows) and 7 (3 rows)
        assert_eq!(l.splices, vec![(2, 4), (7, 3)]);
        // cursor: text 0,1 -> img A base 2 (+max2) -> text 4 -> img B base 5 (+max3) -> text 8,9
        assert_eq!(l.final_mrope_pos, 10);

        let (t, h, w) = (&l.mrope[0..12], &l.mrope[12..24], &l.mrope[24..36]);
        assert_eq!(t, &[0, 1, 2, 2, 2, 2, 4, 5, 5, 5, 8, 9]);
        // img A 2x2, base 2: h = base + j/nx, w = base + j%nx
        assert_eq!(&h[2..6], &[2, 2, 3, 3]);
        assert_eq!(&w[2..6], &[2, 3, 2, 3]);
        // img B 1x3, base 5: nx=1 -> w const, h = base + j
        assert_eq!(&h[7..10], &[5, 6, 7]);
        assert_eq!(&w[7..10], &[5, 5, 5]);
        // bounds: text = own index; each image span = its last row index
        assert_eq!(l.bound, vec![0, 1, 5, 5, 5, 5, 6, 9, 9, 9, 10, 11]);
    }

    #[test]
    fn image_count_mismatch_errors() {
        let one = vec![MmChunk::Text(vec![1]), img(), MmChunk::Text(vec![2])];
        assert!(build_mm_layout(&one, &[]).is_err()); // chunk but no grid
        assert!(build_mm_layout(&one, &[(2, 2), (2, 2)]).is_err()); // too many grids
        assert!(build_mm_layout(&[MmChunk::Text(vec![1, 2])], &[]).is_err()); // no image
    }
}
