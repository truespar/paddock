//! The mixed tick's row budget, the chunked-prefill bound the scheduler and
//! the chunk-capable backends share, and the first-token decode tick.

/// Prefill rows per MIXED tick (chunked prefill): each tick advances the
/// in-flight admission by this many prompt rows alongside every live decode
/// row. 512 was picked for stream latency (~60 ms mixed ticks), but it costs
/// throughput on MoE models: every mixed tick re-streams the touched experts'
/// weights (~1 GB/layer-pass at 120b) no matter how few prompt rows ride it,
/// so the same prompt volume spread over 8x more ticks is ~8x more
/// expert-weight traffic. vLLM's answer is to concentrate the identical load
/// into ~8k-token prefill ticks, and that is what this default does.
/// Override for A/B via PADDOCK_PREFILL_TICK_ROWS.
// Matches PREFILL_CHUNK_DEFAULT (the row-scratch cap) so the mixed-tick budget
// actually fills the bigger buffers. 8192 (was 2048) packs more prompt rows
// behind one MoE weight-read. Override with PADDOCK_PREFILL_TICK_ROWS; the
// backend clamps to row_cap.
const PREFILL_TICK_ROWS_DEFAULT: usize = 8192;

pub(super) fn prefill_tick_rows() -> usize {
    static ROWS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ROWS.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_PREFILL_TICK_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| (32..=16384).contains(&n))
            .unwrap_or(PREFILL_TICK_ROWS_DEFAULT)
    })
}

/// Bound on prompts advancing through mixed ticks at once - the scheduler
/// stops starting chunks at this count, and both chunk-capable backends cap
/// their queue with the same value (a scheduler bound above the backend's
/// would fail admissions with "queue is full"), which is why this is a
/// constant and not derived from the slot count at one call site. The bound -
/// not the row budget (8192) - set burst prefill wave size: a 32-request
/// short-prompt burst chunked as 5 waves of ~8×156 rows, each paying the
/// ~27 ms fixed pass cost over ~1250 rows.
/// PADDOCK_MAX_CHUNKS overrides.
///
/// The default is 32, raised from 12. At 12 a synchronized 32-request burst
/// - which any fixed-osl, ignore_eos load generator guarantees, since
///   cohorts then finish in lockstep - prefills as 12+12+8 and the last
///   wave's first token waits three passes.
///
/// Safe to raise because the ROW budget stays the real limiter: long prompts
/// self-clamp well below 32 chunks, and servers with fewer slots than the
/// bound are unaffected. The direction also matches this file's own MoE
/// reasoning (concentrating prefill into fewer, fatter passes avoids
/// re-streaming expert weights per pass). Revert per-run with
/// PADDOCK_MAX_CHUNKS=12.
pub(crate) fn max_chunks_inflight() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("PADDOCK_MAX_CHUNKS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| (1..=64).contains(&n))
            .unwrap_or(32)
    })
}

/// Adaptive mixed-tick span (A/B vehicle, PADDOCK_SPAN_ADAPTIVE=1 -
/// falsified, keep off). The decode-stall attribution showed every decode
/// row riding a mixed tick waits the whole span (~250 ms at 2048 rows) for
/// one token, which is where long-context tpot goes. Three span-shrink
/// shapes all lost to the fat span: static r1024, static r512, and this
/// rider-count adaptive (at c32 steady state riders are >20 THROUGHOUT, so
/// adaptive degenerates to static-512; c8 unchanged, the <=8 guard held).
/// The law: the ~27 ms per-pass fixed cost (weight stream ~15 ms + eager
/// launches + glue) must amortize over >=~1500 rows. The route to better
/// tpot is CUTTING the per-pass fixed cost (graph-captured fixed-shape
/// mixed tick), not shrinking spans against it.
pub(super) fn mixed_tick_budget(dec_rows: usize) -> usize {
    static ADAPT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let on = *ADAPT.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_SPAN_ADAPTIVE").is_some());
    if !on || dec_rows <= 8 {
        // Sarathi-Serve Algorithm 3: one token budget per iteration, decode
        // rows counted first and the prefill chunk taking what is left. The
        // decodes are never the thing that gets cut - they cost one row each
        // and always ride - so this is the "stall-free" half of the rule
        // written out honestly rather than letting prefill see a budget that
        // pretends the decode rows are free.
        //
        // Be clear about the magnitude: with tau at 8192 and at most a few
        // dozen decode rows this subtraction is under 1%. It is not a
        // performance change and is not dressed up as one - the reason tau
        // stays large here is measured, not inherited: three span-shrink
        // shapes (static 1024/512 and rider-adaptive) all LOST on this MoE
        // because a pass re-streams every touched expert, so per-pass fixed
        // cost has to amortize over >= ~1500 rows. That is the one place our
        // hardware contradicts the paper, whose chunking-overhead model
        // (extra KV re-reads, unchanged FLOPs) assumes a dense model.
        return prefill_tick_rows().saturating_sub(dec_rows).max(1);
    }
    if dec_rows <= 20 { 1024 } else { 512 }
}

/// A stream that has emitted only its first token decodes once before the
/// next chunk: a reasoning model's first token is usually its hidden opener
/// (`<think>`, a channel marker), so its first VISIBLE token would otherwise
/// wait out a whole chunk tick (~1 s at an 8192-row MoE chunk). One decode
/// step per finished prompt. Kill: PADDOCK_NO_FIRST_TOKEN_TICK.
pub(super) fn first_token_tick() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_FIRST_TOKEN_TICK").is_none())
}
