//! The overlap scheduler's admission knobs and gate diagnostics - what
//! decides whether a mixed tick (decode riders aboard a prefill span) runs
//! on the decode pipe beside the span or fuses into one unified tick.

use super::commit::{commit_device_token, tok_trace};
use super::{EngineError, EngineMetrics, Slot, TokenEvent, admit_mm};
use crate::generator::{Generator, RowSample};
use std::collections::HashSet;

/// Live-REQUEST floor (slots holding a request, chunking or decoding) for
/// the two side-stream decode paths - the OVERLAP block (span-only prefill
/// ticks on the main lane, decoders pumped on the pipe lane) and the
/// classic pure-decode pipe. Below it the decode rows ride the unified
/// prefill pass and ticks run one at a time. 0 = the paths engage at any
/// width (the 188-SM die, where both were measured to win). The qwen35
/// default stack sets 9 on dies under 128 SMs: on GB10 (LPDDR5X, 48 SMs) a
/// decode tick is ~100 ms of weight streaming, so at 8 slots the two lanes
/// just time-slice the same bandwidth - 1024x1024 c8 TTFT p50 2698 -> 2124
/// ms and 128x128 c1 223 -> 136 ms with the paths off, throughput flat or
/// up - while from 16 concurrent up the overlap wins the long cohort
/// (1024x1024 c16 128.9 vs 126 tok/s, TTFT p50 7.3 vs 10 s; c32 205 vs
/// 199-202, 12.4 vs 14-16 s). It is counted in requests, not decoders: the
/// cohort's loss is in its EARLY ticks, so gating on decoders (a floor of
/// 9 or 16 decoders) still lost c16/c32 - the overlap has to own the cohort
/// from its first decoder, which a request count gives. PADDOCK_PIPE_MIN_LIVE
/// overrides.
pub(super) fn pipe_min_live() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("PADDOCK_PIPE_MIN_LIVE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

/// PADDOCK_OVERLAP_OVER_SPEC=1: overlap outranks spec-first on a tick the
/// pipe can take. The spec-in-mixed block gives each rider ONE round per
/// span, so a rider's token gap is the whole span (1.4 s on a 27B at the
/// 4096-row span); the pipe pumps decode ticks for the span's whole life
/// and the slot re-warms its drafter on the next spec tick. Measured on
/// Qwen3.8-27B (12 Claude-Code-shaped sessions, 2 slots): riders got 3-5
/// pipe ticks per span instead of one round, each pipe tick ~250 ms beside
/// the saturating pass, and the fat span stretched. A latency trade; see
/// the A/B in the commit that added it.
pub(super) fn spec_first_after_overlap(spec_mixed_first: bool, overlap_core: bool) -> bool {
    spec_mixed_first
        && !(overlap_core && paddock_models::dev_var_os!("PADDOCK_OVERLAP_OVER_SPEC").is_some())
}

/// Why a mixed tick with riders fused instead of overlapping (debug: one
/// line per such tick), every gate's value and each rider's plannability.
pub(super) fn log_fused_mixed_tick(
    dec: &[(usize, u32, u32)],
    slots: &[Option<Slot>],
    generator: &dyn Generator,
    unified_ok: bool,
    spec_mixed_first: bool,
    live_slots: usize,
    trunc_pipe_ok: bool,
) {
    let rows: Vec<(bool, bool, bool, bool, bool)> = dec
        .iter()
        .map(|&(k, _, _)| {
            let s = slots[k].as_ref().expect("live decode row");
            (
                s.constraint.is_none(),
                s.logprobs.is_none(),
                !s.sampler.in_reasoning(),
                s.sampler.is_device_plannable(),
                s.sampler.is_trunc_plannable(),
            )
        })
        .collect();
    tracing::debug!(
        unified_ok,
        spec_mixed_first,
        supports_overlap = generator.supports_overlap(),
        live_slots,
        pipe_min_live = pipe_min_live(),
        pool_free = ?generator.pool_free_blocks(),
        trunc_pipe_ok,
        ?rows,
        "overlap gate: fused mixed tick (no_constraint, no_logprobs, not_reasoning, plannable, trunc_plannable)"
    );
}

/// The pipe loop's exit and why (debug).
pub(super) fn log_pipe_exit(
    generator: &dyn Generator,
    ovl_ticks: usize,
    low_headroom: bool,
    all_dead: bool,
    near_end: bool,
) {
    tracing::debug!(
        ovl_ticks,
        span_done = generator.unified_span_done(),
        low_headroom,
        all_dead,
        near_end,
        pool_free = ?generator.pool_free_blocks(),
        "overlap: pipe loop exit"
    );
}

/// A/B knob (PADDOCK_PIPE_BREAK_ON_DEATH): the overlap pipe segment ends when
/// any of its rows finishes, instead of carrying the dead row as a hole.
pub(super) fn break_on_death() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_PIPE_BREAK_ON_DEATH").is_some())
}

/// Kill switch (PADDOCK_NO_PIPE_FIRST): launch the prefill span BEFORE the
/// overlap pipe's first decode tick, the order this block shipped with. The
/// span launch is eager - ~1,360 kernels and ~190 copies on the engine
/// thread, ~25 ms at 128 slots on the 27B - and with the span first the decode
/// lane sat idle through all of it: 226 gaps, 5.6 s of a 20 s depth-128
/// window (nsys, B200 recover load). Pipe-first puts tick 0 (one graph
/// replay) on the decode lane before the host starts enqueueing the span.
pub(super) fn pipe_first() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_PIPE_FIRST").is_none())
}

/// Begin the overlap decode pipe over `dec` (slot, pending token, position):
/// draws each row's tick-0 plan and enqueues tick 0 on the decode lane.
/// Returns the plans, or None when the backend declines - the span then runs
/// alone and the riders wait for the next tick.
pub(super) fn pipe_begin(
    generator: &mut dyn Generator,
    slots: &mut [Option<Slot>],
    dec: &[(usize, u32, u32)],
    trunc_pipe_ok: bool,
) -> Option<Vec<RowSample>> {
    let slots_v: Vec<u32> = dec.iter().map(|&(k, _, _)| k as u32).collect();
    let toks: Vec<u32> = dec.iter().map(|&(_, t, _)| t).collect();
    let pos: Vec<u32> = dec.iter().map(|&(_, _, p)| p).collect();
    let plans0: Vec<RowSample> = dec
        .iter()
        .map(|&(k, _, _)| {
            let sl = slots[k].as_mut().expect("live decode row");
            match if trunc_pipe_ok {
                sl.sampler.device_plan_trunc()
            } else {
                sl.sampler.device_plan()
            } {
                Some(p) => RowSample::Device(p),
                None => RowSample::Hole, // plannable-gated by the caller
            }
        })
        .collect();
    match generator.decode_pipe_begin_slots(&slots_v, &toks, &pos, &plans0) {
        Ok(()) => Some(plans0),
        Err(e) => {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!("serve: overlap decode pipe unavailable, span-only ticks: {e}");
            });
            None
        }
    }
}

/// End the overlap pipe: commit the last in-flight tick's ids (rows whose
/// plan was Device). A failed drain fails the riders - their state is gone -
/// never the span's prefills.
pub(super) fn pipe_drain(
    generator: &mut dyn Generator,
    slots: &mut [Option<Slot>],
    dec: &[(usize, u32, u32)],
    prev_plans: &[RowSample],
) {
    match generator.decode_pipe_drain() {
        Ok(ids) => {
            for (i, &(k, _, _)) in dec.iter().enumerate() {
                if matches!(prev_plans[i], RowSample::Device(_)) {
                    tok_trace(k, &slots[k], ids[i], "ovl-drain");
                    commit_device_token(&mut slots[k], ids[i], false);
                }
            }
        }
        Err(e) => {
            tracing::warn!("serve: overlap decode_pipe_drain failed: {e}");
            for &(k, _, _) in dec {
                if let Some(s) = slots[k].take() {
                    let _ = s.events.send(TokenEvent::Error(EngineError::from_gen(&e)));
                }
            }
        }
    }
}

/// One encoder step and the admissions it finished. `overlap`: only the
/// non-blocking half (`encode_step_overlap`) - the pump loop runs it beside
/// the span, in the host time the decode tick it just queued leaves; at the
/// loop top the same work (~20 ms a call on the B200 recover load) idled the
/// drained decode lane.
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_admit(
    generator: &mut dyn Generator,
    slots: &mut [Option<Slot>],
    chunking: &mut HashSet<usize>,
    mm_encoding: &mut HashSet<usize>,
    st_adm: &mut u64,
    metrics: &EngineMetrics,
    overlap: bool,
) -> usize {
    let seg_t = std::time::Instant::now();
    let done = if overlap {
        generator.encode_step_overlap()
    } else {
        generator.encode_step()
    };
    let n = done.len();
    for (k, res) in done {
        mm_encoding.remove(&k);
        admit_mm(generator, slots, chunking, st_adm, k, res, metrics);
    }
    if crate::tickseg::on() {
        crate::tickseg::enc(seg_t.elapsed());
    }
    n
}

/// Pump the decode lane while the span runs on the main lane, then drain.
/// Same protocol as the quiet-phase pipe: plans for tick N+1 are drawn
/// before tick N's ids come back (`prev_plans`), dead rows keep ticking as
/// dummies, and the drain collects the last in-flight tick. With no span in
/// flight (failed, or nothing to launch) span_done is already true, so the
/// loop exits at once and only tick 0 drains. Each tick queued covers the
/// host for one riding encoder step. Returns the encoder results admitted.
#[allow(clippy::too_many_arguments)]
pub(super) fn pump(
    generator: &mut dyn Generator,
    slots: &mut [Option<Slot>],
    dec: &[(usize, u32, u32)],
    mut prev_plans: Vec<RowSample>,
    trunc_pipe_ok: bool,
    max_batch: usize,
    chunking: &mut HashSet<usize>,
    mm_encoding: &mut HashSet<usize>,
    st_adm: &mut u64,
    st_ovl: &mut u64,
    metrics: &EngineMetrics,
) -> usize {
    let mut encoded = 0usize;
    let mut ovl_ticks = 0usize;
    loop {
        // stop pumping when the span completes, the pool runs low, a row
        // needs host sampling, or every row died - the drain collects the
        // in-flight tick
        let all_dead = dec.iter().all(|&(k, _, _)| slots[k].is_none());
        // A/B instrument: PADDOCK_PIPE_BREAK_ON_DEATH ends the segment as
        // soon as any row dies, instead of carrying it as a hole; isolates
        // the dead-row continuation.
        let all_dead =
            all_dead || (break_on_death() && dec.iter().any(|&(k, _, _)| slots[k].is_none()));
        let low_headroom = !generator
            .pool_free_blocks()
            .is_none_or(|f| f > max_batch + super::POOL_WATERMARK_BLOCKS);
        // dead rows tick on too: none may step past its context
        let near_end = dec
            .iter()
            .any(|&(_, _, p)| p as usize + ovl_ticks + 3 >= generator.max_context());
        if generator.unified_span_done() || low_headroom || all_dead || near_end {
            log_pipe_exit(&*generator, ovl_ticks, low_headroom, all_dead, near_end);
            break;
        }
        let mut host_row = false;
        let next_plans: Vec<RowSample> = dec
            .iter()
            .map(|&(k, _, _)| match &mut slots[k] {
                // died mid-pump: dummy row keeps ticking, ids discarded
                // (quiet-pipe trick)
                None => RowSample::Hole,
                Some(sl) if sl.constraint.is_none() && sl.logprobs.is_none() => {
                    match if trunc_pipe_ok {
                        sl.sampler.device_plan_trunc()
                    } else {
                        sl.sampler.device_plan()
                    } {
                        Some(p) => RowSample::Device(p),
                        None => {
                            host_row = true;
                            RowSample::Host
                        }
                    }
                }
                Some(_) => {
                    host_row = true;
                    RowSample::Host
                }
            })
            .collect();
        if host_row {
            break;
        }
        ovl_ticks += 1;
        match generator.decode_pipe_next(&next_plans) {
            Ok(ids) => {
                *st_ovl += 1;
                for (i, &(k, _, _)) in dec.iter().enumerate() {
                    if matches!(prev_plans[i], RowSample::Device(_)) {
                        tok_trace(k, &slots[k], ids[i], "ovl-next");
                        commit_device_token(&mut slots[k], ids[i], false);
                    }
                }
                prev_plans = next_plans;
                // the tick just queued covers the host
                if generator.encoding_pending() {
                    encoded += encode_admit(
                        generator,
                        slots,
                        chunking,
                        mm_encoding,
                        st_adm,
                        metrics,
                        true,
                    );
                }
            }
            Err(e) => {
                // the decode rows' state is compromised; the span (chunking
                // slots) is not - fail only the riders, keep the prefills
                tracing::warn!("serve: overlap decode_pipe_next failed: {e}");
                for &(k, _, _) in dec {
                    if let Some(s) = slots[k].take() {
                        let _ = s.events.send(TokenEvent::Error(EngineError::from_gen(&e)));
                    }
                }
                return encoded;
            }
        }
    }
    pipe_drain(generator, slots, dec, &prev_plans);
    encoded
}
