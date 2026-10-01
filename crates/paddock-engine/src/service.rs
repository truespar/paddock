//! The engine service: a dedicated thread that owns the generator and all its
//! mutable state (the vLLM-V1 pattern). Also mandatory for
//! CUDA - the context is thread-bound, so every device call must happen here.
//! HTTP handlers submit requests and receive token events over channels; they
//! never touch model state.
//!
//! Two loops behind the same channel API: a serial one (`run_request`, one
//! request at a time), and a continuous-batching scheduler (`run_batched`) that
//! drives many sequences through one weight-amortized `forward_batch` step per
//! tick. Which one runs is decided at startup by whether `enable_batch`
//! succeeds - so the serial loop is not a museum piece: a big model on a card
//! whose batched KV won't fit lands there, and everything a client observes
//! (usage included) has to be right on both.

use std::sync::Arc;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::mpsc::{Receiver, TryRecvError};

use tokio::sync::mpsc::UnboundedSender;

use crate::generator::{CanvasStatus, CanvasTickReq, FinishSample, GenError, Generator, RowSample};
use crate::metrics::{EngineMetrics, PHASE_DECODE, PHASE_IDLE, PHASE_PREFILL};
use crate::sampler::{Sampler, SamplingParams, TokenConstraint};
use crate::spec::NgramDraft;
use crate::spec_policy::{RoundTally, SpecController, SpecPolicy, synchronous_draft_budget};

mod timing;

/// Serving spec-decode row budget per round: total rows (1 pending + drafts
/// per slot) must fit the models' verify-pass cap. Default 32 matches
/// gpt-oss's SPEC_BATCH_MAX_ROWS and qwen35's alloc; backends whose verify
/// path holds a whole wide round raise it via PADDOCK_SPEC_MAX_ROWS at load
/// (gemma4+MTP sets 160 = 32 slots × (1+4 drafts) - the wide-batch spec
/// rung). Read once: model load runs before the serve loop starts.
/// Admit full-device truncation rows to the decode pipe and overlap tick.
/// DEFAULT ON. Gotcha: a quiet-pipe in-loop draw must not call
/// device_plan(), or the pipe ends every tick and the whole thing looks
/// like a churn-shaped regression. Kill: PADDOCK_NO_TRUNC_PIPE
/// (PADDOCK_TRUNC_PIPE stays honored as an accepted no-op for scripts).
fn trunc_pipe_env() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_TRUNC_PIPE").is_none())
}

fn serve_spec_max_rows() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("PADDOCK_SPEC_MAX_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(32)
    })
}

/// Per-slot draft-length ceiling (the round's row budget shrinks it further
/// when many slots are active). Default 7 keeps the measured qwen35/gpt-oss
/// behavior; gemma4+MTP raises to 15 at attach (deep chains at low live
/// concurrency).
fn serve_spec_max_k() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("PADDOCK_SPEC_MAX_K")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(7)
    })
}

/// Per-round draft budget. For raised row budgets (>32: the gemma4 wide-spec
/// path) this is TWO-TIER: prefer the half-budget round - 32 rows ride the
/// GEMM class's single weight pass (mma_ks BN32 col-tile; 33..64 rows read
/// weights twice) - and take the full budget only when the half-budget
/// yields zero drafts (wide live counts, where 2x weights for k=1 still
/// pays off). Budget <=32 keeps the classic single-tier formula
/// (qwen35/gpt-oss behavior unchanged).
/// Post-miss draft-depth floor: the classic rule drops k_now to the
/// accepted length after any miss - right for weak drafters, but a
/// ~78%/token MTP drafter spends most rounds re-climbing the ladder.
/// Backends with strong drafters raise it (env; default 1 = classic).
fn serve_spec_k_floor() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("PADDOCK_SPEC_K_MISS_FLOOR")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1)
    })
}

/// draft-length cap for MIXED ticks only (chunk in flight). The wide-spec
/// loss lives entirely in the mixed regime (a per-tick drafter prologue plus
/// extra whole-weight-stream passes) while wide PURE rounds gain - so mixed
/// and pure rounds get independent width policies. Default MAX = one shared
/// policy (status quo).
fn serve_spec_mixed_k_cap() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_SPEC_MIXED_K")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(usize::MAX)
    })
}

/// Mixed rounds book a k-independent prompt-chunk surcharge into the Auto
/// controller's lat cells (lives 17..32 share one bucket), which drags the
/// goodput argmax down and leaves a chunk of the PURE rounds running
/// under-depth. Default: mixed rounds skip observe(); =1 restores the old
/// booking.
fn spec_book_mixed() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_SPEC_BOOK_MIXED")
            .map(|v| v != "0")
            .unwrap_or(false)
    })
}

/// Async spec round kill: PADDOCK_NO_ASYNC_SPEC=1 restores the
/// synchronous drafts-then-verify sequence (the A/B surface for the
/// chain->verify boundary elimination).
fn async_spec_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_ASYNC_SPEC").is_none())
}

/// Narrow-tier 64-row boundary for live 9..=16.
/// PADDOCK_SPEC_NARROW64=0 restores the flat 32-row pin.
fn spec_narrow64_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_SPEC_NARROW64")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

/// Verify-row boundary of the narrow tier's 9..=16 band (the "64" in
/// NARROW64). Drafter-class dependent: the MTP chain measures k=3 best there
/// (128 rows overshoot the band), while the qwen35 block drafter measures
/// k=7 best once the >64-row GEMM wall fell, so its attach sets 128 here
/// (PADDOCK_SPEC_NARROW_ROWS, gemma4-style env election; explicit env wins).
/// Default 64 = every other backend unchanged.
fn spec_narrow_rows() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("PADDOCK_SPEC_NARROW_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| n >= 32)
            .unwrap_or(64)
    })
}

/// Verify-row boundary of the live <= 8 tier. Chain drafters keep the
/// 32-row pin (k3 and k7 measure flat on the MTP chain at c8). A BLOCK
/// drafter's round is one forward over `block` positions per slot, and at
/// the rejection-sampling acceptance the deeper round pays all the way to
/// the block - so the tier is `8 * block` there. PADDOCK_SPEC_LOW_ROWS
/// pins either for the A/B.
fn spec_low_rows(block: Option<usize>) -> usize {
    static V: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    let pin = *V.get_or_init(|| {
        std::env::var("PADDOCK_SPEC_LOW_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| n >= 16)
    });
    pin.unwrap_or_else(|| block.map_or(32, |b| 8 * b.max(4)))
}

fn serve_spec_k_budget(live: usize, block: Option<usize>) -> usize {
    let b = serve_spec_max_rows();
    let deep_live_max: usize = paddock_models::dev_var!("PADDOCK_SPEC_DEEP_LIVE_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0); // deep tier off by default: k15 measured worse than k7
    let k = if b > 32 {
        // low live: take the full budget - deep chains repay the wider round
        // now that the 33..64-row GEMM band is pipelined (BN64-ST2), and the
        // drafter's per-token acceptance compounds (what deep chains buy is
        // depth, not drafter quality)
        if live <= deep_live_max {
            (b / live).saturating_sub(1)
        } else {
            // narrow tier, two-step verify-row boundary: 32 rows (the
            // single-BN32-pass GEMM class) for live <= 8, 64 rows (the
            // BN64-ST2 pipelined band) for 9..=16. The flat 32-row pin put
            // live 11..16 at k=1 - a depth cliff bottoming exactly at 16
            // slots while 17 got (b/live)-1 = 6. k=3 recovers that whole
            // valley; k=7 overshoots the band (128 verify rows), and at
            // c8 k3 and k7 are flat. Kill: PADDOCK_SPEC_NARROW64=0 restores
            // the flat 32-row pin.
            let boundary = if live <= 8 {
                spec_low_rows(block)
            } else if !spec_narrow64_on() {
                32
            } else {
                spec_narrow_rows()
            };
            let narrow = if live <= 16 {
                (boundary / live).saturating_sub(1)
            } else {
                0
            };
            if narrow >= 1 {
                narrow
            } else {
                (b / live).saturating_sub(1)
            }
        }
    } else {
        (b / live).saturating_sub(1)
    };
    k.min(serve_spec_max_k())
}

/// The row budget's depth, as the backend qualifies it.
///
/// Narrowed last by whatever depth the backend has measured its drafter to be
/// worth (`spec_depth_cap`). The row budget is a width argument and knows
/// nothing about what a verify row costs on this board, so a backend with a
/// measured depth curve gets the final say downward - never upward, since the
/// rows still have to fit.
fn backend_spec_k_budget(generator: &dyn Generator, live: usize) -> usize {
    if live == 0 || live > generator.spec_live_cap() || generator.spec_deferred() {
        return 0;
    }
    let qualified = generator
        .spec_fixed_draft_depth()
        .unwrap_or_else(|| serve_spec_k_budget(live, generator.spec_block_width()));
    let qualified = generator
        .spec_depth_cap()
        .map_or(qualified, |cap| qualified.min(cap));
    generator
        .spec_batch_draft_budget(live)
        .map_or(qualified, |cap| cap.min(qualified))
}

fn slot_spec_k(requested: usize, budget: usize, fixed_depth: Option<usize>) -> usize {
    fixed_depth.unwrap_or(requested).min(budget)
}

#[test]
fn fixed_block_depth_bypasses_chain_ramp_but_respects_round_budget() {
    for requested in [0usize, 1, 2, 7, 15, 31] {
        for budget in [0usize, 3, 7, 15, 31] {
            assert_eq!(slot_spec_k(requested, budget, None), requested.min(budget));
            assert_eq!(slot_spec_k(requested, budget, Some(15)), 15.min(budget));
        }
    }
}

/// Per-position draft acceptance, behind `PADDOCK_SPEC_DEBUG`.
///
/// The aggregate "N drafted, M accepted" hides the shape, and the shape is
/// what says which thing is wrong. A head whose weights are off is weak
/// everywhere (0.50/0.42/0.38); a head whose chaining is off is strong at
/// position 0 and collapses after (0.80/0.30/0.10), because each further
/// draft rides state the previous one produced. The published references for
/// this family quote exactly this curve - Mia's 0.80/0.59/0.41 -> 2.80 of 4 -
/// so it is also the only form in which our number can be compared to theirs.
///
/// A round that drafted `d` and committed a run of `a` accepted drafts means:
/// positions 0..a were reached and accepted, position `a` was reached and
/// rejected (when a < d). So position i's rate is
/// `reached_and_accepted[i] / reached[i]`.
fn spec_pos_census(drafted: usize, accepted: usize) {
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    static ON: OnceLock<bool> = OnceLock::new();
    if !*ON.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some()) {
        return;
    }
    const K: usize = 16;
    static REACHED: [AtomicU64; K] = [const { AtomicU64::new(0) }; K];
    static ACCEPT: [AtomicU64; K] = [const { AtomicU64::new(0) }; K];
    static ROUNDS: AtomicU64 = AtomicU64::new(0);
    static TOKENS: AtomicU64 = AtomicU64::new(0);
    static DRAFTED: AtomicU64 = AtomicU64::new(0);
    for i in 0..drafted.min(K) {
        REACHED[i].fetch_add(1, Relaxed);
        if i < accepted {
            ACCEPT[i].fetch_add(1, Relaxed);
        }
    }
    DRAFTED.fetch_add(drafted as u64, Relaxed);
    TOKENS.fetch_add(accepted as u64 + 1, Relaxed);
    let r = ROUNDS.fetch_add(1, Relaxed) + 1;
    if r.is_multiple_of(200) {
        let mut curve = String::new();
        for i in 0..K {
            let n = REACHED[i].load(Relaxed);
            if n == 0 {
                break;
            }
            if i > 0 {
                curve.push('/');
            }
            curve.push_str(&format!("{:.2}", ACCEPT[i].load(Relaxed) as f64 / n as f64));
        }
        let (d, t) = (DRAFTED.load(Relaxed) as f64, TOKENS.load(Relaxed) as f64);
        tracing::info!(
            "[spec-pos] rounds {r} per-position {curve} | overall {:.1}% of drafts, \
             {:.2} tokens a round ({:.2} drafted)",
            100.0 * (t - r as f64) / d,
            t / r as f64,
            d / r as f64,
        );
    }
}

/// The operator's speculation policy, from the runner's `spec` config key
/// (threaded in as PADDOCK_SPEC). Read once - it is a serving-envelope choice,
/// not something to re-evaluate per tick.
///
/// Unset = `Ladder`, i.e. exactly the behavior that predates the controller.
/// The legacy kill switches still win outright: an operator who wrote
/// PADDOCK_NO_SPEC meant it, and having a new key quietly outrank an existing
/// "off" would be the worst kind of surprise.
fn serve_spec_policy() -> SpecPolicy {
    static V: std::sync::OnceLock<SpecPolicy> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        if std::env::var_os("PADDOCK_NO_SPEC").is_some()
            || paddock_models::dev_var_os!("PADDOCK_NO_SERVE_SPEC").is_some()
        {
            return SpecPolicy::Off;
        }
        match std::env::var("PADDOCK_SPEC") {
            Ok(v) => match v.parse::<SpecPolicy>() {
                Ok(p) => p,
                Err(e) => {
                    // A malformed policy must not silently pick a behavior -
                    // say so and keep the status quo (no silent failures).
                    tracing::warn!("{e}; keeping the default ladder");
                    SpecPolicy::Ladder
                }
            },
            Err(_) => SpecPolicy::Ladder,
        }
    })
}

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

fn static_env_u64(name: &'static str, default: u64) -> u64 {
    use std::sync::OnceLock;
    static CACHE: OnceLock<std::sync::Mutex<std::collections::HashMap<&'static str, u64>>> =
        OnceLock::new();
    let m = CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let mut g = m.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    *g.entry(name).or_insert_with(|| {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    })
}

fn service_epoch_ms() -> u64 {
    use std::sync::OnceLock;
    static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
    EPOCH
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_millis() as u64
}

fn prefill_tick_rows() -> usize {
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
fn mixed_tick_budget(dec_rows: usize) -> usize {
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

/// A generation request handed to the engine thread.
pub struct GenRequest {
    /// Prompt token ids (must be non-empty).
    pub prompt: Vec<u32>,
    pub max_tokens: usize,
    pub sampler: SamplingParams,
    /// Generation ends (Stop) when any of these is produced; not emitted.
    pub stop_tokens: Vec<u32>,
    /// Where per-token events are delivered (unbounded: never blocks the engine).
    pub events: UnboundedSender<TokenEvent>,
    /// Interleaved multimodal chunks. When set, `prompt` still carries the TEXT
    /// tokens (history/usage) and the request runs EXCLUSIVELY: the scheduler
    /// drains every batch slot first (the vision prefill resets sequence state)
    /// and admits nothing new until it completes.
    pub mm_chunks: Option<Vec<MmChunk>>,
    /// Output constraint (JSON schema / tool grammar). Built server-side with
    /// tokenizer knowledge; the engine only drives the seam. Constrained
    /// sequences never ride speculative rounds (device argmax can't mask).
    pub constraint: Option<Box<dyn TokenConstraint>>,
    /// Some(k) = attach per-token logprobs (chosen + top-k alternatives, raw
    /// pre-penalty distribution) to every Token event. Such sequences never
    /// ride speculative rounds (spec picks skip host logits entirely).
    pub logprobs: Option<u8>,
    /// Stamped by [`Engine::submit`] - the queue-wait anchor for `RunStats`.
    /// Callers leave it None; a direct `run_request` (warmup) has no queue.
    pub submitted: Option<std::time::Instant>,
    /// A block-diffusion STRUCTURED READ instead of a generation: after the
    /// prompt is prefilled, the seeded canvas is run once and each
    /// position's probability of every label id comes back on the request's
    /// own reply channel; the events channel then carries `Done`. Refused
    /// with an error event on a next-token backend. `max_tokens`, sampler
    /// and stop ids are ignored for a read.
    pub canvas_read: Option<CanvasReadRequest>,
}

/// The structured read's inputs and its reply channel (see
/// `GenRequest::canvas_read`, `Generator::canvas_read`).
pub struct CanvasReadRequest {
    /// the seeded canvas ids (template + noise at the answer slots)
    pub canvas: Vec<u32>,
    /// the label ids whose probabilities are wanted at every position
    pub label_ids: Vec<u32>,
    /// denoising steps before the probabilities are read: 1 is one forward
    /// of the seeded canvas; N runs N - 1 ordinary steps first (the model's
    /// own sampling schedule, accept and re-noise), each followed by the
    /// `pinned` positions going back to their seeded ids, and reads the
    /// N-th forward - so the answer slots condition on each other's
    /// partly settled values while the template holds. Past 1 it needs
    /// `Engine::canvas_read_steps` > 1.
    pub steps: u32,
    /// canvas positions held at their seeded ids between steps (every
    /// position but the answer slots); unused at one step
    pub pinned: Vec<u32>,
    /// where the `[canvas.len()][label_ids.len()]` probabilities, entropies
    /// and argmaxes go
    pub reply: std::sync::mpsc::Sender<crate::generator::CanvasReadOut>,
}

/// Per-token log-probabilities (natural log of the raw softmax).
#[derive(Debug, Clone)]
pub struct TokenLogprobs {
    /// logprob of the emitted token
    pub chosen: f32,
    /// top alternatives, (token id, logprob), probability-descending
    pub top: Vec<(u32, f32)>,
}

/// log_softmax stats of `logits` for the chosen token + top-k alternatives.
/// Runs on the RAW logits, before any penalty/temperature mutation.
fn compute_logprobs(logits: &[f32], chosen: u32, top_n: u8) -> TokenLogprobs {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lse = max + logits.iter().map(|&l| (l - max).exp()).sum::<f32>().ln();
    let mut top: Vec<(u32, f32)> = Vec::new();
    if top_n > 0 {
        let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
        let k = (top_n as usize).min(idx.len());
        idx.select_nth_unstable_by(k - 1, |&a, &b| {
            logits[b as usize].total_cmp(&logits[a as usize])
        });
        idx.truncate(k);
        idx.sort_unstable_by(|&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));
        top = idx
            .into_iter()
            .map(|i| (i, logits[i as usize] - lse))
            .collect();
    }
    TokenLogprobs {
        chosen: logits
            .get(chosen as usize)
            .copied()
            .unwrap_or(f32::NEG_INFINITY)
            - lse,
        top,
    }
}

/// One constrained (or plain) sampling step: stop tokens are legal only when
/// the constraint says the output may end; the accepted token advances it.
/// Err = the constraint has no legal continuation (grammar deadlock).
fn pick_next(
    sampler: &mut Sampler,
    logits: &mut [f32],
    history: &[u32],
    constraint: &mut Option<Box<dyn TokenConstraint>>,
    stop_tokens: &[u32],
) -> Result<u32, String> {
    match constraint {
        None => Ok(sampler.sample(logits, history)),
        Some(c) => {
            let may_stop = c.may_stop();
            let next = sampler
                .sample_constrained(logits, history, &mut |id| {
                    if stop_tokens.contains(&id) {
                        may_stop
                    } else {
                        c.allows(id)
                    }
                })
                .ok_or("constraint deadlock: no legal next token")?;
            if !stop_tokens.contains(&next) {
                c.accept(next);
            }
            Ok(next)
        }
    }
}

/// One piece of a multimodal prompt.
#[derive(Clone, Debug)]
pub enum MmChunk {
    Text(Vec<u32>),
    /// Interleaved 8-bit RGB, row-major, as decoded from the request image.
    Image {
        rgb: Vec<u8>,
        w: usize,
        h: usize,
    },
    /// Mono 16 kHz f32 audio, decoded + resampled by the runner. `mel` is the
    /// clip's log-mel features when the runner already ran the host frontend
    /// (on its own thread pool, off the engine thread - the
    /// per-request mel used to serialize the batched engine's admission
    /// pipeline); None = the engine computes it (serial back-compat and any
    /// caller without the frontend). The tower always runs engine-side.
    Audio {
        samples: Vec<f32>,
        mel: Option<crate::audio::MelFeatures>,
    },
    /// Per-request media-processing directive - the class of knob vLLM
    /// carries as `mm_processor_kwargs` and SGLang as `images_config`.
    /// Contributes no rows itself. Today only the deepseek2-ocr family reads
    /// it (`Base` forces the single-image no-crop layout; multi-image is
    /// always base); every other family errors loudly rather than skipping a
    /// directive it does not understand.
    OcrCrop(OcrCropMode),
    /// Per-request smart-resize pixel budget - vLLM's
    /// `mm_processor_kwargs.{min,max}_pixels`, which the official paddleocr
    /// client sends per BLOCK CLASS (Spotting raises max to 1605632). The
    /// paddleocr family runs its own bit-exact HF-processor resize from the
    /// original pixels, so the caller's budget travels as a directive rather
    /// than a runner-side resample. A missing half keeps the family's own
    /// default (vLLM semantics). Contributes no rows; families that do not
    /// understand it error loudly (the OcrCrop precedent).
    VisionPixels {
        min_pixels: Option<u64>,
        max_pixels: Option<u64>,
    },
}

/// The deepseek2-ocr family's two image-serving geometries - the reference's
/// `crop_mode` flag, SGLang's per-request `images_config.image_mode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OcrCropMode {
    /// One padded 1024² global view plus up to `max_tiles` 640² crops
    /// (single image only - the family default there).
    Gundam,
    /// The no-crop layout: one padded 1024² view per image.
    Base,
}

#[derive(Debug, Clone)]
pub enum TokenEvent {
    /// Prefill finished; `cached` prompt tokens were served from the prefix
    /// cache (usage reporting). Sent exactly once per prefilled request,
    /// before its first Token, on both engine loops - the serial one included.
    /// `cached` is 0 on the lanes that have no cache.
    ///
    /// `rows` is what the prefill actually ran - the slot's KV row count -
    /// which is not the prompt's token count on a multimodal request: one
    /// `<image>` placeholder becomes the picture's whole row run (144 to ~2.5k
    /// on granite-vision's AnyRes, a fixed soft count on gemma4,
    /// grid-dependent on qwen35). Callers report token usage from this, not
    /// from the tokenized length, which under-counts an image request by two
    /// orders of magnitude. OpenAI bills image tokens and llama.cpp reports
    /// them, so a client differencing usage against a text-only request has to
    /// see the same thing here.
    Prefilled {
        cached: u32,
        rows: u32,
    },
    Token {
        id: u32,
        /// present when the request asked for logprobs
        logprobs: Option<TokenLogprobs>,
    },
    Done(FinishReason, RunStats),
    Error(EngineError),
}

/// Engine-measured per-request phase timings + counters, carried
/// on `Done` so the serving layer can put real engine phases into its event
/// records instead of edge approximations. Zero-cost to keep: a handful of
/// `Instant`s and two counters per sequence, read once at completion.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RunStats {
    /// A stop ID was actually sampled and suppressed from the text stream.
    /// Length/context limits, client stop strings and discarded speculative
    /// lookahead do not set this. Consumers add it to emitted-token usage;
    /// a Stop finish reason alone is not evidence of an extra token.
    pub sampled_stop: bool,
    /// submit -> admission (serial: pickup off the queue): scheduler wait.
    pub queued_ms: u32,
    /// admission -> prompt fully prefilled. Chunked prefill spans count whole -
    /// this is the client-experienced prefill wall clock, not GPU-time.
    pub prefill_ms: u32,
    /// prefill done -> finish: the decode-phase wall clock.
    pub decode_ms: u32,
    /// Speculative tokens drafted for this sequence across its rounds...
    pub spec_drafted: u32,
    /// ...and how many of those the verify pass accepted. 0/0 = never rode spec.
    pub spec_accepted: u32,
    /// Page-granular KV footprint at completion (`pos` / block tokens),
    /// counting this sequence's whole context - shared prefix pages included.
    /// 0 on the serial path (no paged pool).
    pub kv_pages: u32,
}

impl RunStats {
    pub fn terminal_tokens(&self) -> usize {
        usize::from(self.sampled_stop)
    }
}

fn finish_sampled_stop(
    events: &UnboundedSender<TokenEvent>,
    metrics: &EngineMetrics,
    stats: RunStats,
) {
    metrics.tokens_generated.fetch_add(1, Relaxed);
    let _ = events.send(TokenEvent::Done(
        FinishReason::Stop,
        RunStats {
            sampled_stop: true,
            ..stats
        },
    ));
}

fn dur_ms(d: std::time::Duration) -> u32 {
    d.as_millis().min(u32::MAX as u128) as u32
}

/// Whose fault an in-flight failure is. The engine decides the class; the HTTP
/// layer picks the status code and per-dialect envelope from it (OpenAI 400 /
/// Anthropic 400 for `InvalidRequest`, 503 / 529 for `Overloaded`, 500 for
/// `Internal`). Keeping the classification here - typed, not sniffed out of
/// message strings - is what makes the mapping reliable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorClass {
    /// The request can never succeed as sent (over-context prompt, empty
    /// prompt, unsupported modality) - the caller must change it.
    InvalidRequest,
    /// Transient capacity exhaustion - the same request can succeed once load
    /// drops; clients should back off and retry.
    Overloaded,
    /// Engine fault. Nothing the caller did; a bug or device failure.
    Internal,
}

/// A classified generation error. `code` is the stable machine-readable
/// identifier (OpenAI's `error.code`, e.g. "context_length_exceeded") when one
/// applies; `message` is the human-readable text. (Distinct from
/// `generator::GenError`, the low-level forward-pass error this usually wraps.)
#[derive(Debug, Clone)]
pub struct EngineError {
    pub class: ErrorClass,
    pub code: Option<&'static str>,
    pub message: String,
    /// `(prompt tokens, context window)` when this is a context overflow. A
    /// surface with its own wording for that renders from these numbers
    /// instead of parsing `message` - Anthropic's "prompt is too long: N tokens
    /// > M maximum" is the text Claude Code keys its compaction on.
    pub overflow: Option<(usize, usize)>,
}

impl EngineError {
    /// The catch-all for an engine fault the caller did not cause (-> 500). It
    /// still recognizes a CUDA out-of-memory as a CAPACITY event, not a bug to
    /// hide behind a 500: an OOM during a large prefill - a many-page PDF or a
    /// high-detail image on a VRAM-constrained or busy device - is something the
    /// caller can act on, so it is reclassified to `Overloaded` (retryable) with
    /// a message naming the levers (fewer/smaller pages, lower image detail).
    ///
    /// The PRIMARY OOM path is now typed, not sniffed: a device OOM is
    /// classified by the driver's numeric result code at `gpu::from_driver`,
    /// carried as `GpuError::OutOfMemory` -> `GenError::OutOfMemory`, and mapped
    /// straight to the capacity error by [`from_gen`] - which every
    /// forward-pass call site now uses instead of `internal`. The `is_cuda_oom`
    /// TEXT match retained here is the FALLBACK for an OOM that still reaches
    /// this funnel untyped - e.g. a cuBLAS allocation whose error we do not yet
    /// classify by code, which surfaces as `GenError::Backend(_)` carrying the
    /// driver's rendered `CUDA_ERROR_OUT_OF_MEMORY`. Belt and suspenders: the
    /// typed variant's own Display also carries that signature, so even a stray
    /// `internal(&gen_oom)` would land right.
    ///
    /// Still REACTIVE, by design for now: the complementary pre-flight step - a
    /// measured per-model vision-prefill VRAM estimate that rejects a request
    /// before allocating - is deferred (part 2) so a coarse estimate
    /// can't false-reject valid requests on a well-provisioned device. Until
    /// that lands with GPU-calibrated constants, this typed capacity error is
    /// the graceful, retryable answer an OOM gets.
    pub fn internal(e: impl std::fmt::Display) -> Self {
        let message = e.to_string();
        if is_cuda_oom(&message) {
            return Self::capacity_oom();
        }
        Self {
            class: ErrorClass::Internal,
            code: None,
            message,
            overflow: None,
        }
    }

    /// Map a forward-pass [`GenError`] to the API error, TYPED. This is the
    /// preferred funnel for a scheduler/driver fault: a `GenError::OutOfMemory`
    /// - classified by the driver's numeric result code back at
    ///   `gpu::from_driver`, not by rendered text - becomes a retryable capacity
    ///   error directly, with no string matching in the path at all. Everything
    ///   else defers to [`internal`], which still keeps the `is_cuda_oom` text
    ///   fallback for an OOM that reached it through an untyped `Backend(_)` (e.g.
    ///   a cuBLAS allocation whose error we do not yet type).
    pub fn from_gen(e: &crate::generator::GenError) -> Self {
        match e {
            crate::generator::GenError::OutOfMemory => Self::capacity_oom(),
            other => Self::internal(other),
        }
    }

    /// The graceful, retryable rejection an out-of-memory forward pass gets:
    /// `Overloaded` (not a 500), naming the levers the caller actually controls.
    /// One constructor so the typed path ([`from_gen`]) and the text fallback
    /// ([`internal`]) return the identical shape.
    fn capacity_oom() -> Self {
        Self {
            class: ErrorClass::Overloaded,
            code: Some("insufficient_memory"),
            message: "not enough GPU memory to process this prompt - most often too many or \
                      too large image/PDF pages for the memory available on the device. Send \
                      fewer or smaller pages (the attachment's `pages` selection, or a lower \
                      image `detail`), or retry once the device is less busy"
                .to_string(),
            overflow: None,
        }
    }

    pub fn invalid(msg: impl Into<String>) -> Self {
        Self {
            class: ErrorClass::InvalidRequest,
            code: None,
            message: msg.into(),
            overflow: None,
        }
    }

    pub fn overloaded(msg: impl Into<String>) -> Self {
        Self {
            class: ErrorClass::Overloaded,
            code: Some("overloaded"),
            message: msg.into(),
            overflow: None,
        }
    }

    /// The clean rejection a prompt over the model's context window gets -
    /// mapped to a 400 `context_length_exceeded` instead of a dead engine.
    pub fn context_overflow(got: usize, max: usize) -> Self {
        Self {
            class: ErrorClass::InvalidRequest,
            code: Some("context_length_exceeded"),
            message: format!(
                "the prompt is {got} tokens but the model's context window is {max}; \
                 reduce the input (or restart with a larger --max-ctx)"
            ),
            overflow: Some((got, max)),
        }
    }

    /// The multimodal flavor: image rows dominate these prompts, and "reduce
    /// the input" would send the user hunting through text - name the levers
    /// that actually move the number (page selection, image detail).
    pub fn context_overflow_images(
        got: usize,
        max: usize,
        image_rows: usize,
        images: usize,
    ) -> Self {
        let s = if images == 1 { "" } else { "s" };
        Self {
            class: ErrorClass::InvalidRequest,
            code: Some("context_length_exceeded"),
            message: format!(
                "the prompt needs {got} tokens ({image_rows} of them from {images} image{s}) but \
                 the model's context window is {max}; send fewer or smaller pages (the \
                 attachment's `pages` selection, or a lower image `detail`), or restart with a \
                 larger --max-ctx"
            ),
            overflow: Some((got, max)),
        }
    }
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

/// FALLBACK OOM detector, kept for errors that reach the funnel untyped. The
/// primary path now classifies by numeric result code at `gpu::from_driver`
/// (see [`EngineError::internal`]); this text match only catches an OOM that
/// still arrives as `GenError::Backend(_)` carrying the driver's rendered
/// `CUDA_ERROR_OUT_OF_MEMORY` - e.g. a cuBLAS allocation whose error we do not
/// yet type by code. The signature is stable across cudarc's `DriverError`
/// Display and our own typed variants' Display alike.
fn is_cuda_oom(msg: &str) -> bool {
    let m = msg.to_ascii_uppercase();
    m.contains("OUT_OF_MEMORY") || m.contains("OUT OF MEMORY")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// hit a stop token
    Stop,
    /// hit max_tokens
    Length,
}

impl FinishReason {
    /// OpenAI `finish_reason` string.
    pub fn as_str(&self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
        }
    }
}

/// Cross-thread shutdown control: a process that
/// dies without freeing its CUDA state leaves the DRIVER to reclaim ~27 GB of
/// dead context asynchronously, and that deferred cleanup measurably stalls
/// every other CUDA process on the card for the next ~1-2 minutes (~850 ms
/// wave stalls in a neighboring server's serving path). The handle side
/// requests a stop; the engine thread acknowledges only after the generator -
/// and with it every device allocation - has been dropped, so the runner can
/// exit knowing the driver has (almost) nothing left to reclaim.
pub struct ShutdownCtl {
    stop: std::sync::atomic::AtomicBool,
    done: std::sync::Mutex<bool>,
    cv: std::sync::Condvar,
}

impl ShutdownCtl {
    fn new() -> Self {
        Self {
            stop: std::sync::atomic::AtomicBool::new(false),
            done: std::sync::Mutex::new(false),
            cv: std::sync::Condvar::new(),
        }
    }

    pub fn request(&self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn stop_requested(&self) -> bool {
        self.stop.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn finish(&self) {
        let mut d = self.done.lock().unwrap_or_else(|e| e.into_inner());
        *d = true;
        self.cv.notify_all();
    }

    fn wait_done(&self, timeout: std::time::Duration) -> bool {
        let d = self.done.lock().unwrap_or_else(|e| e.into_inner());
        let (d, _) = self
            .cv
            .wait_timeout_while(d, timeout, |done| !*done)
            .unwrap_or_else(|e| e.into_inner());
        *d
    }
}

/// Marks the shutdown done even on early return or panic, so a waiter never
/// hangs on a thread that never reached the loop.
struct FinishOnDrop(Arc<ShutdownCtl>);
impl Drop for FinishOnDrop {
    fn drop(&mut self) {
        self.0.finish();
    }
}

/// Handle to the engine thread. Clone-cheap (just a channel sender).
#[derive(Clone)]
pub struct Engine {
    tx: std::sync::mpsc::Sender<GenRequest>,
    shutdown: Arc<ShutdownCtl>,
    /// Live scheduler counters (tok/s, batch, phase, KV) for telemetry.
    metrics: Arc<EngineMetrics>,
    /// How large an image this endpoint's tower can use, sampled once from the
    /// generator at startup. Immutable for the process, so it rides the ready
    /// signal out rather than needing a lock or a round-trip to the engine
    /// thread on every request that carries an image.
    vision_budget: Option<crate::generator::VisionBudget>,
    /// The canvas width of a block-diffusion generator, 0 on next-token
    /// ones - sampled at startup like `vision_budget`, so the serving layer
    /// can offer (or refuse) the structured read without a round trip.
    canvas_width: usize,
    /// The most denoising steps a structured read may take: `READ_MAX_STEPS`
    /// on a block-diffusion backend that pins canvas positions (either
    /// loop), 1 on one that cannot, 0 on a next-token model.
    canvas_read_steps: u32,
    /// True when a structured read (and a canvas generation) may carry
    /// images: a vision companion is attached, and the loop in use prefills
    /// image prompts (the serial loop's whole-prompt pass, or the batched
    /// loop's slot path).
    canvas_images: bool,
}

/// The step cap of a multi-step structured read - the reference example
/// server's clamp. Past a handful of steps the answer slots are generating,
/// not being read.
pub const READ_MAX_STEPS: u32 = 8;

/// What the engine thread reports once the generator is built and warm.
struct Ready {
    vision_budget: Option<crate::generator::VisionBudget>,
    canvas_width: usize,
    canvas_read_steps: u32,
    canvas_images: bool,
}

impl Engine {
    /// Spawn the engine thread. `build` constructs the generator on that thread
    /// (required for CUDA) and may fail; spawn blocks until build finishes and
    /// propagates the error. `max_batch` is the desired continuous-batching
    /// width; if the generator supports batching the thread runs the scheduler,
    /// otherwise it falls back to the serial loop (any `max_batch <= 1` also
    /// forces serial).
    pub fn spawn<F>(max_batch: usize, build: F) -> Result<Self, String>
    where
        F: FnOnce() -> Result<Box<dyn Generator>, String> + Send + 'static,
    {
        Self::spawn_with_metrics(max_batch, Arc::new(EngineMetrics::default()), build)
    }

    /// Residency reloads retain one telemetry identity/counter bank for the
    /// lifetime of the endpoint. Only one engine may own this bank at a time.
    pub fn spawn_with_metrics<F>(
        max_batch: usize,
        metrics: Arc<EngineMetrics>,
        build: F,
    ) -> Result<Self, String>
    where
        F: FnOnce() -> Result<Box<dyn Generator>, String> + Send + 'static,
    {
        let (tx, rx) = std::sync::mpsc::channel::<GenRequest>();
        // Ok carries the generator's vision budget (None = no tower) - the one
        // fact the outside needs from the generator itself, read on the engine
        // thread that owns it.
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<Ready, String>>();

        let thread_metrics = metrics.clone();
        let shutdown = Arc::new(ShutdownCtl::new());
        let thread_shutdown = shutdown.clone();

        std::thread::Builder::new()
            .name("paddock-engine".into())
            .spawn(move || {
                let metrics = thread_metrics;
                let ctl = thread_shutdown;
                let _finish = FinishOnDrop(ctl.clone());
                let mut generator = match build() {
                    Ok(g) => g,
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                // Try to turn on batched decode. `enable_batch` returns the real
                // capacity (1 = unsupported); a failure also means serial.
                // Always attempt it, even at max_batch<=1 (it used to be
                // gated behind `spec_capable() || device_sampling&&decode_pipe`,
                // which silently skipped the call entirely for any family/config
                // that didn't happen to qualify, e.g. a plain granite/laguna
                // server with no drafter attached). The batched loop is where
                // continuous batching, prefix caching, paged KV, AND (for the
                // k-quant families) the tuned decode kernels all live - serial
                // is the fallback for "this family/config genuinely can't",
                // not a default for "nothing special would use it". A family
                // that can't or won't benefit reports that itself: `enable_batch`
                // returns Ok(1) only when self.batch is genuinely built (every
                // family's own impl either builds fully or returns a real Err -
                // granite/laguna's VRAM-insufficient path used to lie with a
                // graceful Ok(1) here).
                // A block-diffusion backend has its OWN batched loop
                // (`run_batched_diffusion`): the next-token scheduler would
                // put a canvas request through decode - wrong silently - so
                // it never elects that one. Its slots and pools come from
                // the same `enable_batch`; a width of 1 (or a pool that does
                // not fit) keeps the serial loop.
                let diffusion = generator.canvas_width() > 0;
                let (cap, batched) = if diffusion {
                    let cap = if max_batch > 1 {
                        match generator.enable_batch(max_batch) {
                            Ok(c) => c,
                            Err(e) => {
                                tracing::warn!(
                                    "paddock: block-diffusion batch of {max_batch} did not fit \
                                     ({e}); serving one request at a time"
                                );
                                1
                            }
                        }
                    } else {
                        1
                    };
                    tracing::info!(
                        "paddock: block-diffusion model - {} (canvas {} positions per block)",
                        if cap > 1 {
                            format!("batched loop, {cap} slots")
                        } else {
                            "serial loop".to_string()
                        },
                        generator.canvas_width()
                    );
                    (cap, cap > 1)
                } else if max_batch > 1 {
                    // Width-by-VRAM backstop: models estimate a fitting width
                    // themselves (qwen35's clamp), but if an allocation still
                    // OOMs, halve and retry instead of cliff-dropping straight
                    // to the serial engine - a narrower batch beats no batch.
                    let mut w = max_batch;
                    let c = loop {
                        match generator.enable_batch(w) {
                            Ok(c) => break c,
                            Err(GenError::Config(msg)) => {
                                // provably infeasible config: fail startup with the
                                // actionable message - halving the width (or serial
                                // fallback) would silently serve a different config
                                // than the user asked for
                                let _ = ready_tx.send(Err(msg));
                                return;
                            }
                            Err(e) => {
                                tracing::warn!("paddock: enable_batch({w}) failed: {e}");
                                if w <= 2 {
                                    break 1;
                                }
                                w /= 2;
                                tracing::info!(
                                    "paddock: retrying continuous batching at width {w} \
                                     (width-by-VRAM backstop)"
                                );
                            }
                        }
                    };
                    if c <= 1 {
                        tracing::warn!(
                            "paddock: continuous batching unavailable at this config - running the \
                             SERIAL engine (concurrent requests won't batch, though long prompts still \
                             bulk-prefill). Lower --max-ctx or PADDOCK_MAX_BATCH so the batched KV fits."
                        );
                    }
                    (c, c > 1)
                } else {
                    // Single-user width (max_batch<=1): try the batched loop
                    // unconditionally rather than only when a drafter or
                    // decode-pipe reason is already known to apply. Two
                    // reasons this can win even with nobody else concurrent:
                    // - spec_capable: the drafter only runs in run_batched
                    //   (qwen35's MTP chain; granite's model-free n-gram
                    //   drafter, same reason).
                    // - decode pipe + device sampling: the serial loop pays a
                    //   per-token host round trip (logits readback + host argmax
                    //   + launch gap) that small-active-set models can't hide -
                    //   on an A6000, gpt-oss ran 5.1 ms/token of kernels inside
                    //   an 11.6 ms/token loop (~6.5 ms host bubble). The
                    //   batched loop's depth-2 pipe + on-device sampling
                    //   erase it.
                    // But even a family with neither reason still wants this
                    // lane if it's k-quant-weighted: the batched lane is the
                    // only place the tuned W4A8 decode GEMV and prefix cache
                    // run (granite/laguna's serial lane falls back to the
                    // exact-f32 oracle GEMV and has no prefix cache at all).
                    match generator.enable_batch(1) {
                        Ok(c) => {
                            if generator.spec_capable() {
                                tracing::info!(
                                    "paddock: single-user batched decode ON (spec/drafter) - \
                                     routing max_batch=1 through the batched loop."
                                );
                            } else if generator.supports_device_sampling()
                                && generator.supports_decode_pipe()
                            {
                                tracing::info!(
                                    "paddock: single-user batched decode ON (decode pipe) - \
                                     routing max_batch=1 through the batched loop."
                                );
                            } else {
                                tracing::info!(
                                    "paddock: single-user batched decode ON - routing max_batch=1 \
                                     through the batched loop (continuous batching/prefix cache/ \
                                     tuned kernels)."
                                );
                            }
                            (c.max(1), true)
                        }
                        Err(GenError::Config(msg)) => {
                            // same fatal contract as the width path above
                            let _ = ready_tx.send(Err(msg));
                            return;
                        }
                        // An OOM here means the batched lane EXISTS and simply did
                        // not fit. Serving would still start without it - and that is
                        // exactly the trap, because what starts is a serve with no
                        // prefix cache, no continuous batching, and the untuned
                        // reference GEMV on k-quant weights. Measured on a
                        // 27B Q4_K at max_ctx 65536: 5.9 tok/s, 57,365 prefill tokens
                        // for a ~20k conversation, one 489-second answer. A start that
                        // silently costs ~10x is worse than a start that does not
                        // happen, so this refuses on the same grounds the width>1
                        // path above does. The levers are CONFIG, not a switch: there
                        // is deliberately no "serve it slowly anyway" flag to hand over.
                        Err(GenError::OutOfMemory) => {
                            let _ = ready_tx.send(Err(
                                "the batched serving lane does not fit on this GPU.\n\n\
                                 It would still start without that lane, but with no prefix cache \
                                 (every turn re-prefills the whole conversation from scratch), no \
                                 continuous batching, and on k-quant weights the untuned reference \
                                 GEMV instead of the tuned W4A8 one. Measured on a 27B Q4_K at \
                                 max_ctx 65536: 5.9 tok/s, and 57,365 prefill tokens for a 20k \
                                 conversation. That is not a serve worth starting quietly.\n\n\
                                 Make the lane fit:\n  \
                                 - lower max_ctx (the lane's prefill scratch and its KV both scale with it)\n  \
                                 - raise vram_budget, or free VRAM on this card\n  \
                                 - lower max_batch if it is above 1"
                                    .to_string(),
                            ));
                            return;
                        }
                        Err(e) => {
                            // Not a cosmetic downgrade, and it used to read like one.
                            // The serial loop has no prefix cache at all, and on
                            // k-quant weights it falls back to the exact-f32 oracle
                            // GEMV, so an agentic serve re-prefills its whole context
                            // on every tool round. Measured on a qwen3.8-27b
                            // UD-Q4_K_XL at max_ctx 65536: 57,365 prefill tokens for a
                            // ~20k conversation, prefix hit rate 0.0, 5.9 tok/s decode,
                            // one 489-second answer - and the only trace was this line.
                            // Say what was lost and what to do about it.
                            tracing::warn!(
                                "paddock: single-user batched decode UNAVAILABLE - serving SERIAL: {e}
  \n                                 lost: the prefix cache (every turn re-prefills the whole conversation), \n                                 continuous batching, and on k-quant weights the tuned W4A8 decode GEMV
  \n                                 usually the batch lane did not FIT: lower max_ctx, or raise vram_budget / \n                                 free VRAM on this card"
                            );
                            (1, false)
                        }
                    }
                };
                // Measured model footprint for telemetry - sampled here, after
                // enable_batch allocated the batch KV/state pools.
                if let Some(b) = generator.device_mem_used() {
                    metrics.model_mem_bytes.store(b, Relaxed);
                }
                if let Some(b) = generator.weights_mem_bytes() {
                    metrics.weights_mem_bytes.store(b, Relaxed);
                }
                if let Some(b) = generator.kv_mem_bytes() {
                    metrics.kv_mem_bytes.store(b, Relaxed);
                }
                // the serving state as planned, before the warm-up runs anything
                // (see `audit_warm_footprint`)
                let planned_pool = generator.device_mem_used();
                // Warm up so the first real request doesn't pay the cold-start
                // cost - CUDA-graph capture (per-token decode graph, short-prefill
                // graph), cuBLAS handle init, and the first large allocations - which
                // is seconds on a big model and shows up as a huge TTFT on request
                // #1 only. One tiny dummy generation on the serial path touches all
                // of it up front. llama.cpp warms up by default (its --no-warmup
                // skips it); PADDOCK_NO_WARMUP skips ours. `_wrx` stays bound so the
                // run completes (unbounded sends just buffer, then drop with it).
                // Only when the SERIAL loop is what serves. A batched serve warms
                // below through the real scheduler, and the serial path is not
                // what it runs (multimodal included, where the family takes
                // images in slots) - but a serial request allocates that lane's
                // own state, which lives on unused: qwen35 keeps a dense
                // max_ctx KV plane per full-attention layer beside the batch
                // pool (+12.6 GB at 160K x 2, +21.8 GB at 256K x 2 on
                // qwen3.8-27B, measured 2026-09-28), outside the KV plan that
                // had already sized the pool against the grant. On a
                // unified-memory box that is the OS's headroom: a DGX Spark
                // serving 256K x 2 sat at ~17 GB free and the kernel's OOM
                // killer took the manager after about a day.
                if paddock_models::dev_var_os!("PADDOCK_NO_WARMUP").is_none() && !batched {
                    let (wtx, _wrx) = tokio::sync::mpsc::unbounded_channel();
                    run_request(
                        generator.as_mut(),
                        GenRequest {
                            prompt: vec![1, 2, 3, 4],
                            max_tokens: 4,
                            sampler: SamplingParams::default(),
                            stop_tokens: Vec::new(),
                            events: wtx,
                            mm_chunks: None,
                            constraint: None,
                            logprobs: None,
                            submitted: None,
                            canvas_read: None,
                        },
                        &metrics,
                    );
                }
                // Batched warm wave. The serial run above touches the serial
                // path only; the first real cohort then paid the batch path's
                // cold start inside its own latency - measured on a 5060 Ti at
                // c8: a 303 ms first prefill span against 155 ms warm, and the
                // first width-8 decode graph captured under the clients. This
                // is what vLLM's and SGLang's startup graph capture buys them:
                // one synthetic cohort of `cap` prompts through the real
                // scheduler (admission, chunked prefill, mixed ticks, the
                // decode graph at every width the cohort passes through on its
                // way out), on a private channel whose sender is dropped so
                // the loop returns the moment the cohort has drained. One
                // prompt is long enough to run the prefill span arms; the rest
                // stay under the checkpoint-snapshot floor so the warm-up
                // leaves at most two entries in the state pool.
                if batched && paddock_models::dev_var_os!("PADDOCK_NO_WARMUP").is_none() {
                    let n = cap.max(1);
                    let vocab_n = generator.vocab().max(128);
                    let max_ctx = generator.max_context().max(8);
                    let (wtx, wrx) = std::sync::mpsc::channel::<GenRequest>();
                    let mut keep_rx = Vec::with_capacity(n);
                    for i in 0..n {
                        let len = if i == 0 { 256 } else { 40 }.min(max_ctx / 2).max(4);
                        let prompt: Vec<u32> = (0..len)
                            .map(|j| ((i * 7919 + j * 104_729 + 17) % (vocab_n - 64) + 32) as u32)
                            .collect();
                        let (etx, erx) = tokio::sync::mpsc::unbounded_channel();
                        keep_rx.push(erx);
                        let _ = wtx.send(GenRequest {
                            prompt,
                            max_tokens: 8,
                            sampler: SamplingParams::default(),
                            stop_tokens: Vec::new(),
                            events: etx,
                            mm_chunks: None,
                            constraint: None,
                            logprobs: None,
                            submitted: None,
                            canvas_read: None,
                        });
                    }
                    drop(wtx);
                    let t0 = std::time::Instant::now();
                    let vocab = generator.vocab();
                    if diffusion {
                        run_batched_diffusion(generator.as_mut(), &wrx, n, &metrics, &ctl);
                    } else {
                        run_batched(generator.as_mut(), &wrx, n, vocab, &metrics, &ctl);
                    }
                    drop(keep_rx);
                    tracing::info!(
                        "engine: batched warm wave done ({n} prompts, {:.0} ms)",
                        t0.elapsed().as_secs_f64() * 1e3
                    );
                }
                // the warm-up allocated what the first requests would have (the
                // drafter's serving state, graph captures): publish the ledger
                // as it now stands
                refresh_model_mem(
                    generator.as_ref(),
                    &metrics,
                    &mut None,
                    MODEL_MEM_REFRESH,
                );
                audit_warm_footprint(generator.as_ref(), planned_pool);
                // Build + enable_batch + warmup are all done - signal ready here, not
                // right after build(), so the server starts listening only once warm.
                // That moves the one-time cold-start cost into load time (a slightly
                // longer "model ready") and makes request #1 fast, like every other.
                let canvas_width = generator.canvas_width();
                let batched_canvas = batched && diffusion;
                let _ = ready_tx.send(Ok(Ready {
                    vision_budget: generator.vision_budget(),
                    canvas_width,
                    canvas_read_steps: if canvas_width == 0 {
                        0
                    } else if generator.canvas_pins() {
                        READ_MAX_STEPS
                    } else {
                        1
                    },
                    canvas_images: canvas_width > 0
                        && if batched_canvas {
                            generator.supports_mm_slots()
                        } else {
                            generator.vision_budget().is_some()
                        },
                }));
                if batched && diffusion {
                    run_batched_diffusion(generator.as_mut(), &rx, cap.max(1), &metrics, &ctl);
                } else if batched {
                    let vocab = generator.vocab();
                    run_batched(generator.as_mut(), &rx, cap.max(1), vocab, &metrics, &ctl);
                } else {
                    let mut mem_seen = None;
                    loop {
                        match rx.recv_timeout(std::time::Duration::from_millis(250)) {
                            Ok(req) => {
                                run_request(generator.as_mut(), req, &metrics);
                                refresh_model_mem(
                                    generator.as_ref(),
                                    &metrics,
                                    &mut mem_seen,
                                    MODEL_MEM_REFRESH,
                                );
                            }
                            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                                if ctl.stop_requested() {
                                    break;
                                }
                            }
                            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                        }
                    }
                }
                // Free every device allocation on this THREAD before the
                // FinishOnDrop guard signals the waiter (drop order: generator
                // first, _finish last).
                drop(generator);
                tracing::info!("engine: generator dropped - device memory freed");
            })
            .map_err(|e| format!("failed to spawn engine thread: {e}"))?;

        match ready_rx.recv() {
            Ok(Ok(ready)) => Ok(Self {
                tx,
                shutdown,
                metrics,
                vision_budget: ready.vision_budget,
                canvas_width: ready.canvas_width,
                canvas_read_steps: ready.canvas_read_steps,
                canvas_images: ready.canvas_images,
            }),
            Ok(Err(e)) => Err(e),
            Err(_) => Err("engine thread died during startup".into()),
        }
    }

    /// Graceful shutdown: ask the engine thread to leave its loop and drop the
    /// generator (freeing all device memory on that thread), then wait up to
    /// `timeout` for the drop to complete. Returns true when the free is
    /// confirmed. Dying without this leaves the driver to reclaim the dead
    /// context asynchronously - which stalls every other CUDA process on the
    /// card for the next ~1-2 minutes (see ShutdownCtl).
    pub fn shutdown(&self, timeout: std::time::Duration) -> bool {
        self.shutdown.request();
        self.shutdown.wait_done(timeout)
    }

    /// The largest image this endpoint's vision tower can use, or None when it
    /// has no tower. The API's `detail` handling and the Studio's per-image
    /// picker both size against this, so there is one number, from the file.
    pub fn vision_budget(&self) -> Option<crate::generator::VisionBudget> {
        self.vision_budget
    }

    /// The canvas width when this engine serves a block-diffusion model
    /// (the structured read's ceiling), 0 on a next-token model.
    pub fn canvas_width(&self) -> usize {
        self.canvas_width
    }

    /// The most denoising steps one structured read may take (see
    /// `READ_MAX_STEPS`); 1 where only single-pass reads are served, 0 on a
    /// next-token model.
    pub fn canvas_read_steps(&self) -> u32 {
        self.canvas_read_steps
    }

    /// True when structured reads may carry images.
    pub fn canvas_images(&self) -> bool {
        self.canvas_images
    }

    pub fn submit(&self, mut req: GenRequest) -> Result<(), String> {
        req.submitted.get_or_insert_with(std::time::Instant::now);
        self.tx
            .send(req)
            .map_err(|_| "engine thread is gone".to_owned())
    }

    /// Live scheduler counters for telemetry (read-only; lock-free).
    pub fn metrics(&self) -> Arc<EngineMetrics> {
        self.metrics.clone()
    }
}

/// Resets the engine phase/slot counters to idle when a serial request exits by
/// any path (done, error, client hang-up).
struct IdleOnDrop<'a>(&'a EngineMetrics);
impl Drop for IdleOnDrop<'_> {
    fn drop(&mut self) {
        self.0.active_slots.store(0, Relaxed);
        self.0.phase.store(PHASE_IDLE, Relaxed);
    }
}

/// Encoder-owned and queued slots have no valid decoder cursor yet.
fn spec_live_slots(
    slots: &[Option<Slot>],
    chunking: &std::collections::HashSet<usize>,
) -> Vec<usize> {
    slots
        .iter()
        .enumerate()
        .filter_map(|(k, s)| {
            s.as_ref()
                .filter(|s| s.prefilled && !chunking.contains(&k))
                .map(|_| k)
        })
        .collect()
}

/// Per-slot spec warmth for one round, in the order the round will see the
/// slots. Two passes when a RING drafter (DFlash) owns the round at this
/// width: the cheap ring probe first - a ring-cold slot under a DFlash round
/// drafts nothing and rides the verify, and the token-replay gap re-warm
/// `spec_ensure_warm` would otherwise pay for it every tick (a serial
/// single-slot backbone re-run ~25 ms, never consumed: DFlash rounds do not
/// advance the MTP cursor - one over-cap admission tick that wipes the rings
/// can otherwise cost a multiple of throughput for the rest of the run).
/// Only when no slot is ring-warm - the ramp, where the ring lane
/// declines and the MTP chain takes the round - does the full ensure_warm
/// pass run, exactly as before.
/// Hand every due reply pin to the backend (`Generator::reply_pin`). Called
/// where nothing has yet fed or committed a slot past the token that opened
/// its call: the top of each scheduler tick (dense sampling feeds that token
/// on the NEXT tick) and just before the sampled round's commit (whose
/// snapshot would otherwise land past it).
fn drain_reply_pins(generator: &mut dyn Generator, slots: &mut [Option<Slot>]) {
    for (k, s) in slots.iter_mut().enumerate() {
        if let Some(s) = s.as_mut()
            && std::mem::take(&mut s.reply_pin_due)
        {
            generator.reply_pin(k);
        }
    }
}

fn spec_warm_vec(generator: &mut dyn Generator, slots: &[Option<Slot>], ks: &[usize]) -> Vec<bool> {
    let ring_owned = generator.spec_ring_owns_round(ks.len());
    let want = |s: &Slot| s.pos.saturating_sub(1);
    let mm_cold = |s: &Slot, g: &dyn Generator| {
        // multimodal slots: image rows advance pos past the token history.
        // KV-space drafters (gemma4's Q-only MTP) draft there natively;
        // token-replay drafters can't sync the gap - report those cold.
        s.pos as usize > s.history.len() && !g.spec_draft_kv_space()
    };
    if ring_owned {
        let probe: Vec<bool> = ks
            .iter()
            .map(|&k| {
                let s = slots[k].as_ref().expect("live");
                if mm_cold(s, generator) {
                    return false;
                }
                generator.spec_ring_warm(k, want(s)).unwrap_or(false)
            })
            .collect();
        if probe.iter().any(|&w| w) {
            // The ring drafter takes this round, and every text slot rides it:
            // the drafter filters warmth per slot itself (a ring-cold slot
            // gets an empty draft list and is verified for one token), and
            // that verify's append is what re-warms it. Reporting the cold
            // ones as cold here would drop them from the round's pendings -
            // a slot outside the round is not stepped that tick, nothing
            // appends its feature, and it stays cold (starved) for as long as
            // the others keep rounding - the old always-true re-warm avoided
            // that. Only multimodal slots the drafter cannot see stay
            // excluded, exactly as before.
            return ks
                .iter()
                .map(|&k| !mm_cold(slots[k].as_ref().expect("live"), generator))
                .collect();
        }
    }
    ks.iter()
        .map(|&k| {
            let s = slots[k].as_ref().expect("live");
            if mm_cold(s, generator) {
                return false;
            }
            let committed = &s.history[..s.history.len().min(s.pos as usize)];
            generator
                .spec_ensure_warm(k, committed, want(s))
                .unwrap_or(false)
        })
        .collect()
}

/// The serial request on a BLOCK-DIFFUSION backend (DiffusionGemma), after
/// `run_request` put the prompt's `rows` in the single-stream slot: the
/// answer is denoised a canvas at a time - up to `canvas_width` positions,
/// narrowed to what `max_tokens` still allows - each block emitted through
/// the same Token events a next-token backend sends, cut at the first stop
/// id, committed (re-run causally) only when the answer goes on.
///
/// What does not apply to a canvas and is refused rather than silently
/// ignored: output constraints (the canvas is not sampled left to right, so
/// a grammar cannot be walked) and per-token logprobs (the read primitive is
/// the structured read, not a token stream). Penalties and top-k/top-p are
/// not part of the model's sampler either; the request's `temperature` maps
/// to deterministic at 0 and to the model's own schedule otherwise - there
/// is no customer knob on the schedule.
fn run_request_diffusion(
    generator: &mut dyn Generator,
    req: GenRequest,
    rows: usize,
    t_admit: std::time::Instant,
    queued_ms: u32,
    metrics: &EngineMetrics,
) {
    if req.constraint.is_some() {
        let _ = req.events.send(TokenEvent::Error(EngineError::invalid(
            "structured outputs are not served on a block-diffusion model: the canvas is not \
             sampled left to right, so a grammar cannot constrain it",
        )));
        return;
    }
    if req.logprobs.is_some() {
        let _ = req.events.send(TokenEvent::Error(EngineError::invalid(
            "per-token logprobs are not served on a block-diffusion model; the structured read \
             (/v1/systemone) is its probability surface",
        )));
        return;
    }
    let width = generator.canvas_width();
    let temperature = if req.sampler.temperature <= 0.0 {
        Some(0.0)
    } else {
        None
    };
    let t_prefilled = std::time::Instant::now();
    let stats = |t_prefilled: std::time::Instant| RunStats {
        sampled_stop: false,
        queued_ms,
        prefill_ms: dur_ms(t_prefilled.saturating_duration_since(t_admit)),
        decode_ms: dur_ms(t_prefilled.elapsed()),
        spec_drafted: 0,
        spec_accepted: 0,
        kv_pages: 0,
    };
    metrics.phase.store(PHASE_DECODE, Relaxed);
    let mut base = rows;
    let mut emitted = 0usize;
    while emitted < req.max_tokens {
        let w = width.min(req.max_tokens - emitted);
        let (block, _steps) = match generator.canvas_block(base, w, temperature, req.sampler.seed) {
            Ok(b) => b,
            Err(e) => {
                let _ = req
                    .events
                    .send(TokenEvent::Error(EngineError::from_gen(&e)));
                return;
            }
        };
        for &id in &block {
            if req.stop_tokens.contains(&id) {
                finish_sampled_stop(&req.events, metrics, stats(t_prefilled));
                return;
            }
            if req
                .events
                .send(TokenEvent::Token { id, logprobs: None })
                .is_err()
            {
                return;
            }
            metrics.tokens_generated.fetch_add(1, Relaxed);
            emitted += 1;
        }
        if emitted >= req.max_tokens {
            break;
        }
        // the answer goes on: the block's K/V become real, the next canvas
        // starts behind it
        if let Err(e) = generator.canvas_commit(base, &block) {
            let _ = req
                .events
                .send(TokenEvent::Error(EngineError::from_gen(&e)));
            return;
        }
        base += block.len();
    }
    let _ = req
        .events
        .send(TokenEvent::Done(FinishReason::Length, stats(t_prefilled)));
}

/// The batched loop for a BLOCK-DIFFUSION backend (DiffusionGemma): up to
/// `max_batch` requests, each in its own batch slot, their canvases denoised
/// together - one `canvas_tick` per round carries every live canvas through
/// one forward (`Generator::canvas_tick`), the way a spec round carries every
/// slot's draft block. Admission is a blocking batched prefill of the new
/// prompts (the classic admission pass); a converged canvas is emitted
/// through the ordinary Token events, cut at the first stop id, committed
/// causally when the answer goes on, and the slot opens its next canvas.
/// Structured reads ride the same tick (one step at temperature 1, no
/// accept) and answer on their own reply channel. Nothing the next-token
/// scheduler has - host sampling, penalties, spec, the decode pipe - applies
/// to a canvas, which is why this is its own loop and not a branch of
/// `run_batched`.
fn run_batched_diffusion(
    generator: &mut dyn Generator,
    rx: &Receiver<GenRequest>,
    max_batch: usize,
    metrics: &Arc<EngineMetrics>,
    ctl: &ShutdownCtl,
) {
    struct DSlot {
        events: UnboundedSender<TokenEvent>,
        stop_tokens: Vec<u32>,
        max_tokens: usize,
        /// Some(0.0) = deterministic; None = the model's schedule
        temperature: Option<f32>,
        seed: u64,
        read: Option<CanvasReadRequest>,
        /// rows committed to the slot's KV (prompt, then each emitted block)
        base: usize,
        emitted: usize,
        /// the open canvas: (handle, width, the seed folded with the block
        /// offset), None between blocks
        canvas: Option<(usize, usize, u64)>,
        steps: u32,
        queued_ms: u32,
        t_admit: std::time::Instant,
        t_prefilled: std::time::Instant,
    }
    impl DSlot {
        fn stats(&self) -> RunStats {
            RunStats {
                sampled_stop: false,
                queued_ms: self.queued_ms,
                prefill_ms: dur_ms(self.t_prefilled.saturating_duration_since(self.t_admit)),
                decode_ms: dur_ms(self.t_prefilled.elapsed()),
                spec_drafted: 0,
                spec_accepted: 0,
                kv_pages: 0,
            }
        }
    }
    let width = generator.canvas_width();
    let max_ctx = generator.max_context();
    let max_steps = generator.canvas_max_steps().max(1);
    let per_tick = generator.canvas_tick_max().max(1);
    let mut slots: Vec<Option<DSlot>> = (0..max_batch).map(|_| None).collect();
    let refuse = |events: &UnboundedSender<TokenEvent>, msg: &str| {
        let _ = events.send(TokenEvent::Error(EngineError::invalid(msg)));
    };
    loop {
        // ── admission: fill the free slots, blocking only when nothing is live
        let mut pending: Vec<(usize, Vec<u32>)> = Vec::new();
        let mut mm_pending: Vec<(usize, Vec<MmChunk>)> = Vec::new();
        while let Some(k) = slots.iter().position(Option::is_none) {
            let idle =
                slots.iter().all(Option::is_none) && pending.is_empty() && mm_pending.is_empty();
            let req = if idle {
                match rx.recv_timeout(std::time::Duration::from_millis(250)) {
                    Ok(r) => r,
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                        if ctl.stop_requested() {
                            return;
                        }
                        continue;
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return,
                }
            } else {
                match rx.try_recv() {
                    Ok(r) => r,
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => break,
                }
            };
            let t_admit = std::time::Instant::now();
            let queued_ms = req
                .submitted
                .map_or(0, |t| dur_ms(t_admit.saturating_duration_since(t)));
            // what a canvas cannot carry, refused by name (the serial loop's
            // rules); a read's canvas is checked like the serial path's. An
            // image prompt prefills through the backend's multimodal slot
            // path (the chat lane's own), which a model loaded without its
            // vision companion does not have.
            if req.mm_chunks.is_some() && !generator.supports_mm_slots() {
                refuse(
                    &req.events,
                    "this block-diffusion model was loaded without its vision companion \
                     (mmproj), so it cannot take images",
                );
                continue;
            }
            if req.constraint.is_some() {
                refuse(
                    &req.events,
                    "structured outputs are not served on a block-diffusion model: the canvas is \
                     not sampled left to right, so a grammar cannot constrain it",
                );
                continue;
            }
            if req.logprobs.is_some() {
                refuse(
                    &req.events,
                    "per-token logprobs are not served on a block-diffusion model; the structured \
                     read (/v1/systemone) is its probability surface",
                );
                continue;
            }
            if req.prompt.is_empty() {
                refuse(&req.events, "empty prompt");
                continue;
            }
            let tail = req
                .canvas_read
                .as_ref()
                .map_or(width.min(req.max_tokens.max(1)), |r| r.canvas.len());
            if let Some(r) = &req.canvas_read
                && (r.canvas.is_empty() || r.canvas.len() > width)
            {
                refuse(
                    &req.events,
                    "the read canvas must hold 1 .. canvas_width positions",
                );
                continue;
            }
            if let Some(r) = &req.canvas_read
                && (r.steps == 0
                    || r.steps > READ_MAX_STEPS
                    || (r.steps > 1 && !generator.canvas_pins())
                    || r.pinned.iter().any(|&p| p as usize >= r.canvas.len()))
            {
                refuse(
                    &req.events,
                    "a read takes 1 denoising step here, or up to 8 on a backend that pins \
                     canvas positions, with every pinned position inside the canvas",
                );
                continue;
            }
            if req.prompt.len() + tail > max_ctx {
                let _ = req
                    .events
                    .send(TokenEvent::Error(EngineError::context_overflow(
                        req.prompt.len() + tail,
                        max_ctx,
                    )));
                continue;
            }
            let temperature = if req.sampler.temperature <= 0.0 {
                Some(0.0)
            } else {
                None
            };
            slots[k] = Some(DSlot {
                events: req.events,
                stop_tokens: req.stop_tokens,
                max_tokens: req.max_tokens,
                temperature,
                seed: req.sampler.seed,
                read: req.canvas_read,
                base: req.prompt.len(),
                emitted: 0,
                canvas: None,
                steps: 0,
                queued_ms,
                t_admit,
                t_prefilled: t_admit,
            });
            match req.mm_chunks {
                Some(chunks) => mm_pending.push((k, chunks)),
                None => pending.push((k, req.prompt)),
            }
        }
        // image prompts: the backend's batched vision encode, then each
        // slot's prefill; the rows it reports (image rows included) are where
        // the slot's canvas starts
        if !mm_pending.is_empty() {
            metrics.phase.store(PHASE_PREFILL, Relaxed);
            for (k, r) in
                generator.forward_prefill_multimodal_batch(std::mem::take(&mut mm_pending))
            {
                let Some(s) = slots[k].as_mut() else {
                    continue;
                };
                let tail = s
                    .read
                    .as_ref()
                    .map_or(width.min(s.max_tokens.max(1)), |r| r.canvas.len());
                match r {
                    Ok((_, rows)) if rows + tail <= max_ctx => {
                        s.base = rows;
                        s.t_prefilled = std::time::Instant::now();
                        let _ = s.events.send(TokenEvent::Prefilled {
                            cached: 0,
                            rows: rows as u32,
                        });
                    }
                    Ok((_, rows)) => {
                        let s = slots[k].take().expect("live");
                        let _ = s
                            .events
                            .send(TokenEvent::Error(EngineError::context_overflow(
                                rows + tail,
                                max_ctx,
                            )));
                    }
                    Err(e) => {
                        let s = slots[k].take().expect("live");
                        let _ = s.events.send(TokenEvent::Error(EngineError::from_gen(&e)));
                    }
                }
            }
        }
        if !pending.is_empty() {
            metrics.phase.store(PHASE_PREFILL, Relaxed);
            match generator.forward_prefill_batch(&pending) {
                Ok(_) => {
                    let now = std::time::Instant::now();
                    for (k, _) in &pending {
                        if let Some(s) = slots[*k].as_mut() {
                            s.t_prefilled = now;
                            let _ = s.events.send(TokenEvent::Prefilled {
                                cached: 0,
                                rows: s.base as u32,
                            });
                        }
                    }
                }
                Err(e) => {
                    for (k, _) in pending {
                        if let Some(s) = slots[k].take() {
                            let _ = s.events.send(TokenEvent::Error(EngineError::from_gen(&e)));
                        }
                    }
                }
            }
        }
        // ── every live slot without a canvas opens its next one
        for slot in slots.iter_mut() {
            let Some(s) = slot.as_mut() else {
                continue;
            };
            if s.canvas.is_some() {
                continue;
            }
            let opened = (|| -> Result<(usize, usize, u64), GenError> {
                if let Some(read) = &s.read {
                    let h = generator.canvas_open(read.canvas.len())?;
                    generator.canvas_set(h, &read.canvas)?;
                    return Ok((h, read.canvas.len(), s.seed));
                }
                let w = width.min(s.max_tokens - s.emitted).max(1);
                let block_offset = (s.base as u32).wrapping_mul(2_654_435_761);
                let h = generator.canvas_open(w)?;
                let ids = generator.canvas_noise(w, s.seed, block_offset);
                generator.canvas_set(h, &ids)?;
                Ok((h, w, s.seed ^ u64::from(block_offset)))
            })();
            match opened {
                Ok(c) => s.canvas = Some(c),
                Err(e) => {
                    let _ = s.events.send(TokenEvent::Error(EngineError::from_gen(&e)));
                    *slot = None;
                }
            }
        }
        // ── the tick: every live canvas, in groups the prefill scratch holds
        let live: Vec<usize> = (0..slots.len())
            .filter(|&k| slots[k].as_ref().is_some_and(|s| s.canvas.is_some()))
            .collect();
        metrics.active_slots.store(live.len() as u32, Relaxed);
        if live.is_empty() {
            metrics.phase.store(PHASE_IDLE, Relaxed);
            continue;
        }
        metrics.phase.store(PHASE_DECODE, Relaxed);
        let mut outcomes: Vec<(usize, CanvasStatus)> = Vec::with_capacity(live.len());
        for group in live.chunks(per_tick) {
            let ticks: Vec<CanvasTickReq> = group
                .iter()
                .map(|&k| {
                    let s = slots[k].as_ref().expect("live");
                    let (handle, _, seed) = s.canvas.expect("live");
                    CanvasTickReq {
                        handle,
                        slot: k,
                        base: s.base,
                        // a read's LAST step reads at temperature 1 and
                        // accepts nothing; its earlier steps (a multi-step
                        // read) denoise like generation, on the model's own
                        // schedule
                        temperature: match &s.read {
                            Some(r) if s.steps + 1 >= r.steps => Some(1.0),
                            Some(_) => None,
                            None => s.temperature,
                        },
                        seed,
                        accept: s.read.as_ref().is_none_or(|r| s.steps + 1 < r.steps),
                    }
                })
                .collect();
            match generator.canvas_tick(&ticks) {
                Ok(st) => outcomes.extend(group.iter().copied().zip(st)),
                Err(e) => {
                    for &k in group {
                        if let Some(s) = slots[k].take() {
                            if let Some((h, _, _)) = s.canvas {
                                generator.canvas_close(h);
                            }
                            let _ = s.events.send(TokenEvent::Error(EngineError::from_gen(&e)));
                        }
                    }
                }
            }
        }
        // ── outcomes: a read answers; a converged (or exhausted) canvas emits
        for (k, status) in outcomes {
            let Some(s) = slots[k].as_mut() else {
                continue;
            };
            let (h, _, _) = s.canvas.expect("ticked");
            s.steps += 1;
            // between the steps of a multi-step read the template goes back
            // on top of what the step accepted and re-noised
            let pin = s.read.as_ref().filter(|r| s.steps < r.steps).map(|r| {
                let ids: Vec<u32> = r.pinned.iter().map(|&p| r.canvas[p as usize]).collect();
                (r.pinned.clone(), ids)
            });
            if let Some((positions, ids)) = pin {
                if let Err(e) = generator.canvas_pin(h, &positions, &ids) {
                    generator.canvas_close(h);
                    let s = slots[k].take().expect("live");
                    let _ = s.events.send(TokenEvent::Error(EngineError::from_gen(&e)));
                }
                continue;
            }
            if let Some(read) = &s.read {
                let res = generator.canvas_result(h, &read.label_ids);
                generator.canvas_close(h);
                let s = slots[k].take().expect("live");
                match res {
                    Ok(out) => {
                        let _ = s.read.as_ref().expect("read").reply.send(out);
                        let _ = s
                            .events
                            .send(TokenEvent::Done(FinishReason::Stop, s.stats()));
                    }
                    Err(e) => {
                        let _ = s.events.send(TokenEvent::Error(EngineError::from_gen(&e)));
                    }
                }
                continue;
            }
            if !status.converged && s.steps < max_steps {
                continue;
            }
            let block = match generator.canvas_result(h, &[]) {
                Ok(out) => out.argmax,
                Err(e) => {
                    generator.canvas_close(h);
                    let s = slots[k].take().expect("live");
                    let _ = s.events.send(TokenEvent::Error(EngineError::from_gen(&e)));
                    continue;
                }
            };
            generator.canvas_close(h);
            s.canvas = None;
            s.steps = 0;
            let mut finished = false;
            for &id in &block {
                if s.stop_tokens.contains(&id) {
                    finish_sampled_stop(&s.events, metrics, s.stats());
                    finished = true;
                    break;
                }
                if s.events
                    .send(TokenEvent::Token { id, logprobs: None })
                    .is_err()
                {
                    finished = true; // the client went away
                    break;
                }
                metrics.tokens_generated.fetch_add(1, Relaxed);
                s.emitted += 1;
                if s.emitted >= s.max_tokens {
                    let _ = s
                        .events
                        .send(TokenEvent::Done(FinishReason::Length, s.stats()));
                    finished = true;
                    break;
                }
            }
            if finished {
                slots[k] = None;
                continue;
            }
            // the answer goes on: the block's K/V become real, the next
            // canvas starts behind it (opened at the top of the next round)
            if let Err(e) = generator.canvas_commit_slot(k, s.base, &block) {
                let s = slots[k].take().expect("live");
                let _ = s.events.send(TokenEvent::Error(EngineError::from_gen(&e)));
                continue;
            }
            s.base += block.len();
        }
        let occupied: Vec<bool> = slots.iter().map(Option::is_some).collect();
        generator.release_inactive_slots(&occupied);
    }
}

/// Re-read the device ledger the runner publishes (`model_mem_bytes`) from the
/// family's live pool counter - a driver attribute read, no sync - at most
/// once per `every`. A reading taken once at setup went stale by everything
/// allocated after it (the warm-up's drafter state, grow-on-demand scratch),
/// and the manager's reconciler measured NVML against that stale figure:
/// 2.6-21.8 GB of "drift" on a healthy runner, warned every two seconds for
/// days.
fn refresh_model_mem(
    generator: &dyn Generator,
    metrics: &EngineMetrics,
    last: &mut Option<std::time::Instant>,
    every: std::time::Duration,
) {
    if last.is_some_and(|t| t.elapsed() < every) {
        return;
    }
    *last = Some(std::time::Instant::now());
    if let Some(b) = generator.device_mem_used() {
        metrics.model_mem_bytes.store(b, Relaxed);
    }
}

/// How often a serving loop refreshes the published device ledger.
const MODEL_MEM_REFRESH: std::time::Duration = std::time::Duration::from_secs(1);

/// Warm-up growth past which the serve is holding state its KV plan never
/// reserved. The warm-up legitimately allocates what the first requests
/// would - a drafter's serving state, verify planes, graph pools: 2.28 GiB on
/// qwen3.8-27B NVFP4 with DFlash2 at 256K x 2 (2026-09-28) - so the line
/// that must be loud is the class above it, the serial lane's 12.6-21.8 GB.
const WARM_GROWTH_WARN: u64 = 4 << 30;

/// Startup's will-it-fit audit, once warm: the plan sized the pools against
/// the grant before anything ran, so what the warm-up adds on top is memory
/// no plan accounted for - said with its size, loudly past
/// `WARM_GROWTH_WARN`. And, where the OS reports it, the system memory left
/// once warm: on a unified-memory box (DGX Spark, Jetson) device memory IS
/// system memory, and the default budget promises the OS a tenth of it
/// (`GpuExecutor`'s 0.9 utilization) - under that, say so before a day of
/// serving finds the OOM killer. The serial warm-up's own dense state was
/// the case this exists for: 12.6-21.8 GB on qwen3.8-27B, invisible to the
/// plan, and the Spark it ran on down to ~17 GB free (2026-09-28).
fn audit_warm_footprint(generator: &dyn Generator, planned: Option<u64>) {
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    if let (Some(before), Some(now)) = (planned, generator.device_mem_used()) {
        let grew = now.saturating_sub(before);
        if grew > WARM_GROWTH_WARN {
            tracing::warn!(
                "warm-up allocated {:.2} GiB beyond the planned serving state ({:.2} -> {:.2} \
                 GiB): memory the KV plan did not reserve, which on a unified-memory box comes \
                 out of the OS's share",
                gib(grew),
                gib(before),
                gib(now)
            );
        } else {
            tracing::info!(
                "warm-up allocated {:.2} GiB beyond the planned serving state",
                gib(grew)
            );
        }
    }
    if let Some((total, avail)) = system_memory() {
        if avail < total / 10 {
            tracing::warn!(
                "system memory once warm: {:.1} of {:.1} GiB free - under the tenth the OS is \
                 budgeted. Lower max_ctx or max_batch before a long serve meets the OOM killer.",
                gib(avail),
                gib(total)
            );
        } else {
            tracing::info!(
                "system memory once warm: {:.1} of {:.1} GiB free",
                gib(avail),
                gib(total)
            );
        }
    }
}

/// (MemTotal, MemAvailable) in bytes, where the OS reports them.
fn system_memory() -> Option<(u64, u64)> {
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        let field = |name: &str| -> Option<u64> {
            let line = text.lines().find(|l| l.starts_with(name))?;
            let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
            Some(kb * 1024)
        };
        Some((field("MemTotal:")?, field("MemAvailable:")?))
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

fn run_request(generator: &mut dyn Generator, req: GenRequest, metrics: &EngineMetrics) {
    generator.reset();
    // Phase anchors: the serial loop picks a request up as soon as the
    // previous one finishes, so queue = submit -> here.
    let t_admit = std::time::Instant::now();
    let queued_ms = req
        .submitted
        .map_or(0, |s| dur_ms(t_admit.saturating_duration_since(s)));

    if req.prompt.is_empty() {
        let _ = req
            .events
            .send(TokenEvent::Error(EngineError::invalid("empty prompt")));
        return;
    }
    let max_ctx = generator.max_context();
    if req.prompt.len() > max_ctx {
        let _ = req
            .events
            .send(TokenEvent::Error(EngineError::context_overflow(
                req.prompt.len(),
                max_ctx,
            )));
        return;
    }
    // Generation cannot run past the window: KV addressing (dense rows and
    // the pooled block table alike) ends exactly at max_ctx, so an unclamped
    // max_tokens let a near-window prompt decode straight off the end and
    // panic the engine thread (index-OOB in the slot's block table). Clamp so the run
    // finishes with an honest Length at the edge; a prompt that fills the
    // window exactly has no room for even one token.
    let room = max_ctx - req.prompt.len();
    if room == 0 {
        let _ = req.events.send(TokenEvent::Error(EngineError::invalid(
            "the prompt fills the model's entire context window; there is no room to generate",
        )));
        return;
    }
    let mut req = req;
    req.max_tokens = req.max_tokens.min(room);

    // telemetry: one active sequence, prefilling now; reset to idle on any exit.
    metrics.active_slots.store(1, Relaxed);
    metrics.phase.store(PHASE_PREFILL, Relaxed);
    let _idle = IdleOnDrop(metrics);

    // prefill: multimodal chunks in one exclusive pass, or feed the prompt
    // token by token; either way keep the last logits and the ROW COUNT the
    // prefill actually ran (they differ: see `rows` below)
    let mut logits = Vec::new();
    let mut rows = req.prompt.len() as u32;
    if let Some(chunks) = &req.mm_chunks {
        match generator.forward_multimodal(chunks) {
            Ok(Some((l, r))) => {
                logits = l;
                rows = r as u32;
            }
            Ok(None) => {
                let _ = req.events.send(TokenEvent::Error(EngineError::invalid(
                    "this model does not support image input",
                )));
                return;
            }
            Err(e) => {
                let _ = req
                    .events
                    .send(TokenEvent::Error(EngineError::from_gen(&e)));
                return;
            }
        }
    } else if paddock_models::dev_var_os!("PADDOCK_SERIAL_TOKEN_PREFILL").is_some() {
        // Legacy token-by-token prefill, kept behind an env pin for the bit-exact
        // A/B gate: one forward pass per prompt token means a long prompt prefills
        // at decode speed (~55 tok/s), so an agentic/tool transcript pays 10s+ of
        // "prefill" per round. The default below bulk-prefills the whole prompt.
        for &t in &req.prompt {
            match generator.forward(t) {
                Ok(l) => logits = l,
                Err(e) => {
                    let _ = req
                        .events
                        .send(TokenEvent::Error(EngineError::from_gen(&e)));
                    return;
                }
            }
        }
    } else {
        // Bulk-prefill the whole prompt in one batched pass (GPU models; the trait
        // default still feeds token-by-token for backends without a fast prefill).
        match generator.forward_prefill_stream(&req.prompt) {
            Ok(l) => logits = l,
            Err(e) => {
                let _ = req
                    .events
                    .send(TokenEvent::Error(EngineError::from_gen(&e)));
                return;
            }
        }
    }

    // Usage reporting, same contract as the batched loop's `finish_prefill`.
    // The serial lane used to stay silent here, which was harmless for text
    // (the caller's tokenized length is the row count) and an order-of-
    // magnitude under-report for an image prompt: a picture's rows never
    // reached the client, so a gemma4 server narrow enough to run serially
    // billed a 1400-row prompt as its 32 text tokens. `cached` is
    // whatever the backend's prefix cache served - 0 on the lanes that have
    // none, non-zero on gemma4's serial mm prefill, which does run through the
    // radix.
    let cached = generator.take_prefill_reused(0) as u32;
    metrics.prefill_tokens_total.fetch_add(rows as u64, Relaxed);
    metrics
        .prefill_tokens_cached
        .fetch_add(cached as u64, Relaxed);
    let _ = req.events.send(TokenEvent::Prefilled { cached, rows });
    // The pre-prefill clamp above used prompt.len(); a multimodal prefill
    // can run more rows than the id stream carries (image rows). Re-clamp
    // against the rows that actually landed in KV - decode past max_ctx
    // indexes off the end of the window (the batched twin corrupted a
    // block-table stripe on exactly this).
    req.max_tokens = req.max_tokens.min(max_ctx.saturating_sub(rows as usize));

    // A structured read: one forward of the seeded canvas after the prompt,
    // the label probabilities on the request's reply channel, then Done.
    // Nothing is generated and nothing is committed.
    if let Some(read) = req.canvas_read.take() {
        if generator.canvas_width() == 0 {
            let _ = req.events.send(TokenEvent::Error(EngineError::invalid(
                "structured reads need a block-diffusion model",
            )));
            return;
        }
        if read.canvas.is_empty() || read.canvas.len() > generator.canvas_width() {
            let _ = req.events.send(TokenEvent::Error(EngineError::invalid(
                "the read canvas must hold 1 .. canvas_width positions",
            )));
            return;
        }
        if read.steps == 0
            || read.steps > READ_MAX_STEPS
            || (read.steps > 1 && !generator.canvas_pins())
            || read.pinned.iter().any(|&p| p as usize >= read.canvas.len())
        {
            let _ = req.events.send(TokenEvent::Error(EngineError::invalid(
                "a read takes 1 denoising step here, or up to 8 on a backend that pins canvas \
                 positions, with every pinned position inside the canvas",
            )));
            return;
        }
        if rows as usize + read.canvas.len() > max_ctx {
            let _ = req
                .events
                .send(TokenEvent::Error(EngineError::context_overflow(
                    rows as usize + read.canvas.len(),
                    max_ctx,
                )));
            return;
        }
        let t_read = std::time::Instant::now();
        match generator.canvas_read_steps(
            rows as usize,
            &read.canvas,
            &read.label_ids,
            read.steps,
            &read.pinned,
            req.sampler.seed,
        ) {
            Ok(out) => {
                let _ = read.reply.send(out);
                let _ = req.events.send(TokenEvent::Done(
                    FinishReason::Stop,
                    RunStats {
                        sampled_stop: false,
                        queued_ms,
                        prefill_ms: dur_ms(t_read.saturating_duration_since(t_admit)),
                        decode_ms: dur_ms(t_read.elapsed()),
                        spec_drafted: 0,
                        spec_accepted: 0,
                        kv_pages: 0,
                    },
                ));
            }
            Err(e) => {
                let _ = req
                    .events
                    .send(TokenEvent::Error(EngineError::from_gen(&e)));
            }
        }
        return;
    }

    // Block diffusion (DiffusionGemma): the prompt is in KV; the answer is
    // written a whole canvas at a time. Its own loop - nothing below (host
    // sampling, penalties, the decode pipe, spec) applies to a canvas.
    if generator.canvas_width() > 0 {
        run_request_diffusion(generator, req, rows as usize, t_admit, queued_ms, metrics);
        return;
    }

    let mut history = req.prompt.clone();
    let mut sampler = Sampler::new(req.sampler);
    let mut constraint = req.constraint;

    let t_prefilled = std::time::Instant::now();
    let stats = |t_prefilled: std::time::Instant| RunStats {
        sampled_stop: false,
        queued_ms,
        prefill_ms: dur_ms(t_prefilled.saturating_duration_since(t_admit)),
        decode_ms: dur_ms(t_prefilled.elapsed()),
        spec_drafted: 0,
        spec_accepted: 0,
        kv_pages: 0, // serial path: no paged pool
    };
    metrics.phase.store(PHASE_DECODE, Relaxed);

    // Serial decode pipe (b): when every draw is device-
    // executable (greedy / temperature-only categorical - no penalties or
    // bias, per is_device_plannable) and nothing needs host logits (no
    // constraint, no logprobs), run depth-2 pipelined graph ticks with
    // on-device sampling: tick N+1 is enqueued before tick N's id lands, so
    // the per-token host round trip (logits readback + host sample + launch
    // gap) leaves the GPU's critical path - the serial twin of the batched
    // loop's decode pipe. Token 0 still comes off the prefill logits on the
    // host, exactly like the loop below. On a backend that reports the
    // capability but can't begin (a family whose pipe needs its batched
    // state, reachable here via the serial VRAM fallback), finish on a plain
    // host loop instead.
    if req.logprobs.is_none()
        && constraint.is_none()
        && sampler.is_device_plannable()
        && generator.supports_decode_pipe()
    {
        let next = match pick_next(
            &mut sampler,
            &mut logits,
            &history,
            &mut constraint,
            &req.stop_tokens,
        ) {
            Ok(t) => t,
            Err(e) => {
                let _ = req
                    .events
                    .send(TokenEvent::Error(EngineError::internal(&e)));
                return;
            }
        };
        if req.stop_tokens.contains(&next) {
            finish_sampled_stop(&req.events, metrics, stats(t_prefilled));
            return;
        }
        history.push(next);
        if req
            .events
            .send(TokenEvent::Token {
                id: next,
                logprobs: None,
            })
            .is_err()
        {
            return;
        }
        metrics.tokens_generated.fetch_add(1, Relaxed);
        // ticks still needed; the pre-clamp (max_tokens <= window room)
        // bounds total enqueued ticks inside the KV window
        let want = req.max_tokens - 1;
        if want == 0 {
            let _ = req
                .events
                .send(TokenEvent::Done(FinishReason::Length, stats(t_prefilled)));
            return;
        }
        // is_device_plannable holds and params are fixed per request, so
        // every per-token plan draw succeeds; each Categorical draw consumes
        // the token's one uniform, keeping the seed stream host-aligned
        let plan0 = crate::generator::RowSample::Device(
            sampler.device_plan().expect("device-plannable sampler"),
        );
        match generator.decode_pipe_begin(&[next], &[rows], &[plan0]) {
            Ok(()) => {
                let mut enqueued: usize = 1; // ticks handed to the backend
                let mut got: usize = 0; // ids landed + emitted
                loop {
                    let ids = if enqueued < want {
                        let p = crate::generator::RowSample::Device(
                            sampler.device_plan().expect("device-plannable sampler"),
                        );
                        let r = generator.decode_pipe_next(&[p]);
                        enqueued += 1;
                        r
                    } else {
                        generator.decode_pipe_drain()
                    };
                    let id = match ids {
                        Ok(v) => v[0],
                        Err(e) => {
                            let _ = req
                                .events
                                .send(TokenEvent::Error(EngineError::from_gen(&e)));
                            return;
                        }
                    };
                    got += 1;
                    if req.stop_tokens.contains(&id) {
                        // one overshoot tick may still be in flight - collect
                        // and discard it (the next request resets all state)
                        if got < enqueued {
                            let _ = generator.decode_pipe_drain();
                        }
                        finish_sampled_stop(&req.events, metrics, stats(t_prefilled));
                        return;
                    }
                    history.push(id);
                    if req
                        .events
                        .send(TokenEvent::Token { id, logprobs: None })
                        .is_err()
                    {
                        if got < enqueued {
                            let _ = generator.decode_pipe_drain();
                        }
                        return;
                    }
                    metrics.tokens_generated.fetch_add(1, Relaxed);
                    if got == want {
                        let _ = req
                            .events
                            .send(TokenEvent::Done(FinishReason::Length, stats(t_prefilled)));
                        return;
                    }
                }
            }
            Err(_) => {
                // host fallback, forward-first since token 0 is already out
                let mut prev = next;
                for _ in 1..req.max_tokens {
                    logits = match generator.forward(prev) {
                        Ok(l) => l,
                        Err(e) => {
                            let _ = req
                                .events
                                .send(TokenEvent::Error(EngineError::from_gen(&e)));
                            return;
                        }
                    };
                    let nxt = match pick_next(
                        &mut sampler,
                        &mut logits,
                        &history,
                        &mut constraint,
                        &req.stop_tokens,
                    ) {
                        Ok(t) => t,
                        Err(e) => {
                            let _ = req
                                .events
                                .send(TokenEvent::Error(EngineError::internal(&e)));
                            return;
                        }
                    };
                    if req.stop_tokens.contains(&nxt) {
                        finish_sampled_stop(&req.events, metrics, stats(t_prefilled));
                        return;
                    }
                    history.push(nxt);
                    if req
                        .events
                        .send(TokenEvent::Token {
                            id: nxt,
                            logprobs: None,
                        })
                        .is_err()
                    {
                        return;
                    }
                    metrics.tokens_generated.fetch_add(1, Relaxed);
                    prev = nxt;
                }
                let _ = req
                    .events
                    .send(TokenEvent::Done(FinishReason::Length, stats(t_prefilled)));
                return;
            }
        }
    }

    for _ in 0..req.max_tokens {
        // raw distribution snapshot before penalties/masking mutate it
        let raw = req.logprobs.map(|_| logits.clone());
        let next = match pick_next(
            &mut sampler,
            &mut logits,
            &history,
            &mut constraint,
            &req.stop_tokens,
        ) {
            Ok(t) => t,
            Err(e) => {
                let _ = req
                    .events
                    .send(TokenEvent::Error(EngineError::internal(&e)));
                return;
            }
        };
        if req.stop_tokens.contains(&next) {
            finish_sampled_stop(&req.events, metrics, stats(t_prefilled));
            return;
        }
        history.push(next);
        let logprobs = req
            .logprobs
            .map(|k| compute_logprobs(raw.as_ref().expect("snapshot"), next, k));
        // a closed receiver (client hung up) means stop early
        if req
            .events
            .send(TokenEvent::Token { id: next, logprobs })
            .is_err()
        {
            return;
        }
        metrics.tokens_generated.fetch_add(1, Relaxed);
        match generator.forward(next) {
            Ok(l) => logits = l,
            Err(e) => {
                let _ = req
                    .events
                    .send(TokenEvent::Error(EngineError::from_gen(&e)));
                return;
            }
        }
    }
    let _ = req
        .events
        .send(TokenEvent::Done(FinishReason::Length, stats(t_prefilled)));
}

/// One in-flight sequence occupying a fixed KV slot (batch row). New requests are
/// bulk-prefilled (whole prompt in one pass) on admission, then decode one token
/// per batched step.
struct Slot {
    /// prompt tokens; consumed (emptied) by the prefill pass, then unused
    prompt: Vec<u32>,
    prefilled: bool,
    /// the token to feed next decode step (valid once prefilled)
    pending: u32,
    /// next KV position for this slot (== tokens already committed to its cache)
    pos: u32,
    /// prompt + generated so far (drives the repetition penalty)
    history: Vec<u32>,
    sampler: Sampler,
    events: UnboundedSender<TokenEvent>,
    stop_tokens: Vec<u32>,
    max_tokens: usize,
    generated: usize,
    /// n-gram drafter for the speculative round (seeded at prefill; fed every
    /// committed token). Idle weight when the batch can't ride spec.
    draft: NgramDraft,
    /// adaptive per-slot draft length (G3 rule: double on a full accept,
    /// shrink to the observed run on a reject)
    k_now: usize,
    /// output constraint; a constrained slot forces the dense decode path
    constraint: Option<Box<dyn TokenConstraint>>,
    /// logprob top-k; such slots force the dense path (spec skips host logits)
    logprobs: Option<u8>,
    /// multimodal chunks (taken by the mm prefill; only present on backends
    /// that support mm slots - others run mm requests exclusively)
    mm: Option<Vec<MmChunk>>,
    /// P5b-3: this slot was PREEMPTED (its KV freed under pool pressure) and is
    /// being recomputed - the prefill re-runs `history[0..pos]` to rebuild the
    /// KV, and its completion must RESUME decode (feed the existing `pending`)
    /// rather than sample a fresh first token.
    recompute: bool,
    /// live counters (telemetry) - bumped once per committed token.
    metrics: Arc<EngineMetrics>,
    /// Phase anchors + spec accounting for the Done-carried RunStats (§8.8).
    /// `submitted` comes from the request; `admitted` = Slot creation;
    /// `prefill_done` is set once by the first finish_prefill (a preemption
    /// recompute re-prefills mid-decode and must not restart the clock).
    submitted: Option<std::time::Instant>,
    admitted: std::time::Instant,
    prefill_done: Option<std::time::Instant>,
    /// When this prompt's (chunked) prefill STARTED - feeds the tier cost
    /// model's measured recompute rate at finish (1a.4 calibration).
    chunk_started: Option<std::time::Instant>,
    spec_drafted: u32,
    spec_accepted: u32,
    /// The constraint was in its free phase after the last committed token
    /// (true without one). `note_constraint` watches it leave: that token
    /// opened the reply's first tool call.
    constraint_free: bool,
    /// The reply's first tool call just started and the backend has not been
    /// told yet - `drain_reply_pins` hands it to `Generator::reply_pin`
    /// before this slot's next forward or commit.
    reply_pin_due: bool,
    /// The pin was taken; once per reply.
    reply_pinned: bool,
}

impl Slot {
    fn new(mut req: GenRequest, metrics: Arc<EngineMetrics>) -> Self {
        let constraint_free = req.constraint.as_ref().is_none_or(|c| c.free_now());
        Slot {
            history: req.prompt.clone(),
            prompt: req.prompt,
            prefilled: false,
            pending: 0,
            pos: 0,
            sampler: Sampler::new(req.sampler),
            events: req.events,
            stop_tokens: req.stop_tokens,
            max_tokens: req.max_tokens,
            generated: 0,
            draft: NgramDraft::default(),
            k_now: 2,
            constraint: req.constraint,
            logprobs: req.logprobs,
            mm: req.mm_chunks.take(),
            recompute: false,
            metrics,
            submitted: req.submitted,
            admitted: std::time::Instant::now(),
            prefill_done: None,
            chunk_started: None,
            spec_drafted: 0,
            spec_accepted: 0,
            constraint_free,
            reply_pin_due: false,
            reply_pinned: false,
        }
    }

    /// Call after the constraint accepted a token: the free phase ending here
    /// means the reply just opened its first tool call (a tool-choice gate
    /// arming, a dispatcher entering a call), and the backend's reply
    /// checkpoint is due to be held there (`Generator::reply_pin`). Reading
    /// the constraint rather than a dialect's call marker keeps it
    /// family-agnostic: the machine is exactly what knows a call began.
    fn note_constraint(&mut self) {
        let Some(c) = self.constraint.as_ref() else {
            return;
        };
        let free = c.free_now();
        if self.constraint_free && !free && !self.reply_pinned {
            self.reply_pinned = true;
            self.reply_pin_due = true;
        }
        self.constraint_free = free;
    }

    /// The §8.8 stats snapshot, taken at Done time.
    fn run_stats(&self) -> RunStats {
        let now = std::time::Instant::now();
        let queued_ms = self
            .submitted
            .map_or(0, |s| dur_ms(self.admitted.saturating_duration_since(s)));
        let (prefill_ms, decode_ms) = match self.prefill_done {
            Some(p) => (
                dur_ms(p.saturating_duration_since(self.admitted)),
                dur_ms(now.saturating_duration_since(p)),
            ),
            None => (0, 0),
        };
        RunStats {
            sampled_stop: false,
            queued_ms,
            prefill_ms,
            decode_ms,
            spec_drafted: self.spec_drafted,
            spec_accepted: self.spec_accepted,
            kv_pages: (self.pos as usize).div_ceil(crate::kv_pool::BLOCK_TOKENS) as u32,
        }
    }

    /// Book one speculative round: `drafted` tokens rode the verify pass for
    /// this slot, `accepted` of them matched. Called by every spec shape.
    ///
    /// The same numbers go two places - this slot's RunStats (what the request
    /// reports back) and `tick`, the whole-tick tally the speculation
    /// controller learns its acceptance rate from. Taking the tally as an
    /// argument is deliberate: it makes a new spec shape that forgets to feed
    /// the controller a compile error instead of a slow throughput drift.
    fn spec_round(&mut self, tick: &mut RoundTally, drafted: usize, accepted: usize) {
        self.spec_drafted += drafted as u32;
        self.spec_accepted += accepted as u32;
        tick.book(drafted, accepted);
        // Here, not at a call site: there are six accept paths (mixed, the
        // batched plans, the single-slot walks) and which one runs depends on
        // the batch shape, so a census hung off one of them silently reports
        // nothing at the width that does not take it. That is exactly what the
        // first cut did - instrumented the MIXED path and printed no curve at
        // all at c1.
        spec_pos_census(drafted, accepted);
    }

    /// Repurpose this slot for recompute after its KV was freed under pool
    /// pressure. The KV to rebuild is `history[0..pos]` (the committed tokens;
    /// `pending == history[pos]` is fed on resume, so it is excluded). Marks the
    /// slot un-prefilled so the admission/prefill phase re-prefills it, and
    /// `recompute` so its completion RESUMES instead of sampling a fresh token.
    /// The number of blocks the recompute will need (for the watermark).
    fn preempt_for_recompute(&mut self) -> usize {
        let keep = self.pos as usize;
        // A never-started waiter (pos == 0, prompt never consumed) has nothing
        // to recompute - rebuilding from its empty history here destroyed the
        // prompt, and the readmit then died on "chunked prompt is 0 tokens"
        // (found live on a gemma4 serve).
        if keep > 0 || self.prompt.is_empty() {
            self.prompt = self.history[..keep].to_vec();
        }
        self.prefilled = false;
        self.recompute = true;
        self.prompt.len().div_ceil(crate::kv_pool::BLOCK_TOKENS)
    }

    /// Record a freshly sampled token: emit it and set it as the next to feed.
    /// Returns false if the sequence should retire (a Done/hang-up already
    /// happened) - the caller then frees the slot.
    fn accept(&mut self, next: u32, logprobs: Option<TokenLogprobs>) -> bool {
        if self.stop_tokens.contains(&next) {
            finish_sampled_stop(&self.events, &self.metrics, self.run_stats());
            return false;
        }
        self.note_constraint();
        self.history.push(next);
        self.generated += 1;
        self.metrics.tokens_generated.fetch_add(1, Relaxed);
        if self
            .events
            .send(TokenEvent::Token { id: next, logprobs })
            .is_err()
        {
            return false; // client hung up
        }
        if self.generated >= self.max_tokens {
            let _ = self
                .events
                .send(TokenEvent::Done(FinishReason::Length, self.run_stats()));
            return false;
        }
        self.pending = next;
        true
    }
}

/// Finish a freshly prefilled slot: report cache reuse, sample + emit the
/// first token, seed the drafter. `rows` = the slot's KV position (the prompt
/// token count for text; text + image rows for multimodal). Frees the slot on
/// error or immediate completion.
/// Longest common prefix of two token streams, in tokens.
fn lcp_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// The in-flight leader (slot, shared tokens) a pending prompt would wait
/// on: the first one it shares at least `floor` tokens with. None when there
/// is no prefix cache (`floor` None) or nothing in flight shares its prefix.
/// Only a leader that is still computing part of the shared span counts: one
/// that itself resumed at or past the shared length took the prefix from the
/// cache, so the prefix is already published and waiting on it would just
/// serialize the followers (measured: one release per ~60 ms tick).
fn shares_inflight(
    inflight: &[(usize, Vec<u32>, usize)],
    floor: Option<usize>,
    k: usize,
    prompt: &[u32],
) -> Option<(usize, usize)> {
    let f = floor?;
    inflight
        .iter()
        .filter(|(j, _, _)| *j != k)
        .map(|(j, q, resumed)| (*j, lcp_len(prompt, q), *resumed))
        .find(|&(_, l, resumed)| l >= f && l > resumed)
        .map(|(j, l, _)| (j, l))
}

/// How much of an in-flight prompt the scheduler keeps for shared-prefix
/// matching. A follower that shares more than this with a leader still
/// adopts everything the leader publishes; the cap only bounds the clone a
/// 100k-token admission would otherwise make of itself.
const PREFIX_LCP_CAP: usize = 8192;

fn trace_us() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_micros())
}

fn finish_prefill(
    generator: &mut dyn Generator,
    slots: &mut [Option<Slot>],
    k: usize,
    mut logits: Vec<f32>,
    rows: u32,
    metrics: &EngineMetrics,
) {
    // A prior tick's error handler (e.g. forward_mixed_sampled's generic Err
    // branch) may have already broadcast-cleared every slot and sent its
    // client an error - that client's request is already resolved, so a
    // completion report that still names this slot is stale, not a bug to
    // crash over. Check before touching the generator: acting on a stale k
    // there could have its own side effects on state the generator no
    // longer associates with an active request.
    if slots[k].is_none() {
        tracing::warn!(
            "serve: finish_prefill for slot {k} but it's already cleared (stale completion after a prior tick error) - dropping"
        );
        return;
    }
    if paddock_models::dev_var_os!("PADDOCK_REQ_TRACE").is_some() {
        tracing::info!(
            "req-trace: prefilled slot {k} rows {rows} at {}",
            trace_us()
        );
    }
    let cached = generator.take_prefill_reused(k) as u32;
    let max_ctx = generator.max_context();
    let slot = slots[k].as_mut().expect("present");
    // Cost-model calibration: the MEASURED recompute rate - rows
    // actually computed over the wall since the chunk started, decode
    // interleaving included (a restore's alternative is a recompute under
    // the current load). No-op on untiered backends.
    if let Some(t0) = slot.chunk_started.take() {
        let computed = rows.saturating_sub(cached);
        if computed > 0 {
            generator.tier_observe_prefill(computed, t0.elapsed().as_micros() as f64);
        }
    }
    slot.prefilled = true;
    slot.pos = rows;
    // Admission clamped max_tokens against prompt.len(), but a multimodal
    // prefill runs more rows than the id stream carries (image rows) - the
    // "open refinement" admit() noted. Re-clamp against the rows that
    // actually landed in KV so decode finishes Length at the window edge
    // instead of growing the block table past its stripe (the paddleocr
    // battery spilled slot block tables into the neighbor's stripe and
    // panicked the flat upload on the last slot).
    slot.max_tokens = slot.max_tokens.min(max_ctx.saturating_sub(rows as usize));
    // first prefill only - a recompute must not restart the phase clock
    slot.prefill_done
        .get_or_insert_with(std::time::Instant::now);
    // Recompute: the slot's KV was just rebuilt from history[0..pos]. Its
    // pending/pos/history/generated are intact - resume decode without sampling a
    // fresh token or re-reporting usage. (The rebuilt KV comes from the prefill
    // numeric class, so a resumed token may differ from the un-preempted run -
    // preemption is deliberately not bit-exact.)
    if std::mem::take(&mut slot.recompute) {
        return;
    }
    // usage reporting: how much of the prompt the prefix cache served
    // (client channel may already be gone). Also feed the running prefix-cache
    // hit-rate counters (non-recompute prefills only - the recompute path
    // returned above, so preemption recovery never skews the reuse metric).
    metrics.prefill_tokens_total.fetch_add(rows as u64, Relaxed);
    metrics
        .prefill_tokens_cached
        .fetch_add(cached as u64, Relaxed);
    let _ = slot.events.send(TokenEvent::Prefilled { cached, rows });
    let raw = slot.logprobs.map(|_| logits.clone());
    let next = match pick_next(
        &mut slot.sampler,
        &mut logits,
        &slot.history,
        &mut slot.constraint,
        &slot.stop_tokens,
    ) {
        Ok(t) => t,
        Err(e) => {
            let _ = slot
                .events
                .send(TokenEvent::Error(EngineError::internal(&e)));
            slots[k] = None;
            return;
        }
    };
    let lp = slot
        .logprobs
        .map(|n| compute_logprobs(raw.as_ref().expect("snapshot"), next, n));
    if !slot.accept(next, lp) {
        slots[k] = None;
    } else {
        // seed the drafter: prompt + first token (history holds exactly
        // that after accept)
        for i in 0..slot.history.len() {
            let t = slot.history[i];
            slot.draft.push(t);
        }
    }
}

/// `finish_prefill` for a finisher the generator device-sampled (fin_plans):
/// same bookkeeping, no logits and no host pick - the plan's peeked uniform
/// is committed here, in `pick_next`'s position (after the recompute check,
/// which never fires for device-planned finishers but stays for safety).
fn finish_prefill_sampled(
    generator: &mut dyn Generator,
    slots: &mut [Option<Slot>],
    k: usize,
    next: u32,
    plan: &crate::sampler::DevicePlan,
    rows: u32,
    metrics: &EngineMetrics,
) {
    // See finish_prefill's matching guard: a stale completion for a slot a
    // prior tick's error handler already cleared+errored is not a bug to
    // crash over.
    if slots[k].is_none() {
        tracing::warn!(
            "serve: finish_prefill_sampled for slot {k} but it's already cleared (stale completion after a prior tick error) - dropping"
        );
        return;
    }
    if paddock_models::dev_var_os!("PADDOCK_REQ_TRACE").is_some() {
        tracing::info!(
            "req-trace: prefilled slot {k} rows {rows} at {}",
            trace_us()
        );
    }
    let cached = generator.take_prefill_reused(k) as u32;
    let max_ctx = generator.max_context();
    let slot = slots[k].as_mut().expect("present");
    // Cost-model calibration: the MEASURED recompute rate - rows
    // actually computed over the wall since the chunk started, decode
    // interleaving included (a restore's alternative is a recompute under
    // the current load). No-op on untiered backends.
    if let Some(t0) = slot.chunk_started.take() {
        let computed = rows.saturating_sub(cached);
        if computed > 0 {
            generator.tier_observe_prefill(computed, t0.elapsed().as_micros() as f64);
        }
    }
    slot.prefilled = true;
    slot.pos = rows;
    // same rows-based re-clamp as finish_prefill (image rows exceed
    // prompt.len(); decode must stop at the window, not the stripe edge)
    slot.max_tokens = slot.max_tokens.min(max_ctx.saturating_sub(rows as usize));
    slot.prefill_done
        .get_or_insert_with(std::time::Instant::now);
    if std::mem::take(&mut slot.recompute) {
        return;
    }
    slot.sampler.commit_device_plan(plan);
    metrics.prefill_tokens_total.fetch_add(rows as u64, Relaxed);
    metrics
        .prefill_tokens_cached
        .fetch_add(cached as u64, Relaxed);
    let _ = slot.events.send(TokenEvent::Prefilled { cached, rows });
    if !slot.accept(next, None) {
        slots[k] = None;
    } else {
        // seed the drafter: prompt + first token, exactly like the host twin
        for i in 0..slot.history.len() {
            let t = slot.history[i];
            slot.draft.push(t);
        }
    }
}

/// Dispatch a finisher by how the backend sampled it: device-
/// sampled entries carry the plan that was PEEKED at launch - committing it
/// here advances the slot's seed stream exactly once, in pick_next's
/// position, matching the host twin's contract.
fn finish_prefill_any(
    generator: &mut dyn Generator,
    slots: &mut [Option<Slot>],
    k: usize,
    fs: crate::generator::FinishSample,
    plan: Option<crate::sampler::DevicePlan>,
    rows: u32,
    metrics: &EngineMetrics,
) {
    match fs {
        crate::generator::FinishSample::Logits(l) => {
            finish_prefill(generator, slots, k, l, rows, metrics)
        }
        crate::generator::FinishSample::Sampled(id) => {
            let plan = plan.expect("sampled finisher carries its peeked plan");
            finish_prefill_sampled(generator, slots, k, id, &plan, rows, metrics)
        }
    }
}

/// G4b mixed-tick preemption: a mixed (chunked-prefill + decode) pass couldn't
/// grow the budget pool. Preempt the NEWEST active DECODE slot - never a
/// chunking slot, whose prefill is making progress - so the retried tick fits
/// with its blocks freed; `chunking` stays intact and the reconcile at the next
/// tick top returns the victim's blocks. No KV half-advanced: a mixed pass hits
/// PoolExhausted inside ensure_pool_rows, before run_layers writes anything. If
/// every active slot is mid-prefill (no decode victim to relieve the pool - a
/// watermark-should-prevent pathological case), fail the stuck chunking slots so
/// the server recovers instead of spinning.
fn preempt_or_fail_mixed(
    slots: &mut [Option<Slot>],
    preempted: &mut Vec<Slot>,
    chunking: &mut std::collections::HashSet<usize>,
    pool_stats: bool,
) {
    let victim = (0..slots.len())
        .rev()
        .find(|&k| slots[k].is_some() && !chunking.contains(&k));
    match victim {
        Some(k) => {
            let need = slots[k].as_mut().expect("present").preempt_for_recompute();
            preempted.push(slots[k].take().expect("present"));
            if pool_stats {
                tracing::warn!(
                    "pool: PREEMPTED slot {k} (mixed) for recompute ({need} blocks), {} queued",
                    preempted.len()
                );
            }
        }
        None => {
            let stuck: Vec<usize> = chunking.iter().copied().collect();
            tracing::warn!(
                "serve: mixed pool-exhausted with no decode victim; failing {} chunking slot(s)",
                stuck.len()
            );
            for k in stuck {
                if let Some(s) = slots[k].take() {
                    let _ = s.events.send(TokenEvent::Error(EngineError::overloaded(
                        "the KV cache pool is exhausted under the current load; retry shortly",
                    )));
                }
                chunking.remove(&k);
            }
        }
    }
}

/// Settle a multimodal admission verdict that is not `Encoding` - the slot is
/// on the chunked queue now, prefilled by the backend's own stepped passes
/// (finished here exactly as a blocking multimodal prefill is), or its request
/// is over.
///
/// A slot that died while the backend was encoding (client hung up, preempted)
/// reports `Queued` with nothing queued behind it; `slots[k]` is already None
/// then, and inserting it into `chunking` would wedge a slot that has no queue
/// entry to finish it. So the insert is conditional on the slot still existing.
fn admit_mm(
    generator: &mut dyn Generator,
    slots: &mut [Option<Slot>],
    chunking: &mut std::collections::HashSet<usize>,
    st_adm: &mut u64,
    k: usize,
    res: crate::generator::MmAdmit,
    metrics: &EngineMetrics,
) {
    use crate::generator::MmAdmit;
    match res {
        MmAdmit::Queued => {
            if slots[k].is_some() {
                chunking.insert(k);
                *st_adm += 1;
            }
        }
        MmAdmit::Prefilled { logits, rows } => {
            // finish_prefill drops a report whose slot was already resolved
            if slots[k].is_some() {
                *st_adm += 1;
            }
            finish_prefill(generator, slots, k, logits, rows as u32, metrics);
        }
        // Reported by the backend only once it has stopped holding the slot,
        // so this branch is unreachable - but silently treating it as queued is
        // exactly the "prefilled from an empty prompt" bug MmAdmit exists to
        // prevent, so it is named rather than absorbed by a catch-all.
        MmAdmit::Encoding => {
            tracing::warn!("serve: backend reported slot {k} as still encoding out of band");
        }
        MmAdmit::Failed(e) => {
            tracing::warn!("serve: mm prefill_begin failed on slot {k}: {e}");
            if let Some(s) = slots[k].take() {
                let _ = s.events.send(TokenEvent::Error(EngineError::from_gen(&e)));
            }
        }
    }
}

/// P5b: charge a pending prompt's blocks against the per-tick admission budget
/// (`ceil(prompt_len / BLOCK_TOKENS)` - the prefill will claim them). No-op when
/// there is no budget pool.
fn debit_pool_budget(budget: &mut Option<usize>, req: &GenRequest) {
    if let Some(b) = budget.as_mut() {
        *b = b.saturating_sub(req.prompt.len().div_ceil(crate::kv_pool::BLOCK_TOKENS));
    }
}

/// Place a request into the lowest free slot. Empty prompts are rejected here
/// (never occupy a slot). Returns false if there was no free slot (caller must
/// not call when full).
fn admit(
    slots: &mut [Option<Slot>],
    req: GenRequest,
    max_ctx: usize,
    metrics: &Arc<EngineMetrics>,
) -> bool {
    if paddock_models::dev_var_os!("PADDOCK_REQ_TRACE").is_some() {
        tracing::info!(
            queued_ms = req.submitted.map(|t| t.elapsed().as_secs_f64() * 1000.),
            "req-trace: admit {} tokens at {}",
            req.prompt.len(),
            trace_us()
        );
    }
    if req.prompt.is_empty() {
        let _ = req
            .events
            .send(TokenEvent::Error(EngineError::invalid("empty prompt")));
        return true; // "handled" - don't hold the slot
    }
    if req.prompt.len() > max_ctx {
        // Reject before prefill so an over-length prompt can't panic the batch.
        let _ = req
            .events
            .send(TokenEvent::Error(EngineError::context_overflow(
                req.prompt.len(),
                max_ctx,
            )));
        return true; // handled - don't hold the slot
    }
    // Same window-edge clamp as the serial path: decode past max_ctx indexes
    // off the end of the slot's KV (paged: block_table_host OOB panic).
    // Length at the edge is the honest finish. Multimodal prompts prefill
    // more rows than prompt.len() (image rows); this clamp can't see them
    // yet, so finish_prefill[_sampled] re-clamps against the rows the
    // prefill actually ran (the paddleocr battery walked a slot's block
    // table into the neighbor's stripe).
    let room = max_ctx - req.prompt.len();
    if room == 0 {
        let _ = req.events.send(TokenEvent::Error(EngineError::invalid(
            "the prompt fills the model's entire context window; there is no room to generate",
        )));
        return true; // handled - don't hold the slot
    }
    let mut req = req;
    req.max_tokens = req.max_tokens.min(room);
    for s in slots.iter_mut() {
        if s.is_none() {
            *s = Some(Slot::new(req, metrics.clone()));
            return true;
        }
    }
    false
}

/// Admission watermark: block headroom the scheduler keeps free when a
/// budget KV pool is active. It stops pulling new requests once the (per-tick,
/// decremented) free-block budget falls to this level, so in-flight sequences
/// retain room to grow a step and a burst can't over-commit the pool in one
/// tick. Generous (512 tokens) to absorb the coarse per-prompt estimate.
const POOL_WATERMARK_BLOCKS: usize = 32;

/// Row cap for OVERLAPPED admissions inside the decode pipe (0 disables): a
/// short text prompt's pure prefill queues behind the in-flight pipe ticks
/// instead of draining them first. Longer prompts keep the drain-first mixed
/// flow - a long decode-row-less prefill would starve decode cadence.
/// Cohort-fuse admission linger (opt-in, PADDOCK_COHORT_FUSE=1). When a whole
/// cohort finishes in one tick (closed-loop benches, synchronized bursts), the
/// replacements all arrive within a few ms - but the try_recv drain takes only
/// what has already landed, and with nothing decoding the scheduler then
/// commits the admitted set to the BLOCKING wave prefill: a request that lands
/// 1 ms after the drain waits out the whole wave (~190 ms at 32x183) before
/// the next drain sees it. Traced live with PADDOCK_REQ_TRACE: the client's
/// sends cluster in 3-5 ms, the drain cuts 29/3 or 31/1 on a knife-edge,
/// and the split RE-SEEDS itself every round after that (uniform lengths keep
/// the sub-cohorts phase-locked) - run-to-run latency bimodality on identical
/// binaries is exactly this race. The linger costs
/// at most the quiet-gap (2 ms) after the last arrival, only fires when a
/// burst is already forming (>=2 admitted this tick) with nothing decoding
/// (a live-decode latecomer takes the cheap chunked join instead), and c1
/// never pays (admitted stays 1), unless a qualified backend explicitly opts
/// an idle multi-slot server into a separate, at-most-2-ms lone grace below.
///
/// DEFAULT on (kill: PADDOCK_NO_COHORT_FUSE). With it off, most synchronized
/// c32 runs land in the split mode; with it on they run clean. c1 latency is
/// unchanged for default backends (the gate never fires at admitted==1), and
/// the fused-round cost is just the 2 ms quiet-gap per burst - a rounding error
/// against a c32 round.
fn cohort_fuse() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_COHORT_FUSE").is_none())
}

fn lone_cohort_grace(
    active: usize,
    admitted: u32,
    text_only: bool,
    hint: std::time::Duration,
) -> std::time::Duration {
    if active == 0 && admitted == 1 && text_only {
        hint.min(std::time::Duration::from_millis(2))
    } else {
        std::time::Duration::ZERO
    }
}

#[cfg(test)]
mod cohort_grace_tests {
    use super::lone_cohort_grace;
    use std::time::Duration;

    #[test]
    fn only_idle_lone_text_can_opt_in_and_never_exceeds_two_ms() {
        let hint = Duration::from_millis(100);
        for active in [0, 1, 4] {
            for admitted in [0, 1, 2, 4] {
                for text in [false, true] {
                    let expected = if active == 0 && admitted == 1 && text {
                        Duration::from_millis(2)
                    } else {
                        Duration::ZERO
                    };
                    assert_eq!(lone_cohort_grace(active, admitted, text, hint), expected);
                }
            }
        }
        assert_eq!(
            lone_cohort_grace(0, 1, true, Duration::ZERO),
            Duration::ZERO
        );
        assert_eq!(
            lone_cohort_grace(0, 1, true, Duration::from_micros(350)),
            Duration::from_micros(350)
        );
    }
}

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
fn pipe_min_live() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_PIPE_MIN_LIVE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

fn overlap_admit_max() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_OVERLAP_ADMIT_MAX")
            .ok()
            .and_then(|v| v.parse().ok())
            // default off (falsified): with the pipe live, an overlapped
            // prefill stalls decode past the 2 queued ticks and churn-shaped
            // load loses badly. The mixed flow keeps decode rows riding the
            // admission pass; arrivals drain the pipe instead.
            .unwrap_or(0)
    })
}

/// Min-free admission gate (PADDOCK_ADM_MIN_FREE=K, opt-in, default 0=off).
/// At high concurrency the steady state staggers completions, so admissions
/// sprinkle: ~every decode round carries a prefill chunk, and each mixed pass
/// re-streams the touched experts' weights no matter how few prompt rows ride
/// it. The observed fast serving mode BUNCHES refills instead.
/// This gate holds admissions until K slots are free
/// (or ADM_MAX_HOLD_MS passes), so a refill lands as one shared wave - the
/// mixed-tick budget (8192 rows, 32 chunks) swallows it in a single pass.
/// Engages only when the server has >= 2K slots; c1..c32 serving is
/// unaffected unless explicitly armed.
fn adm_min_free() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_ADM_MIN_FREE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

/// Predictive admission bunching (PADDOCK_ADM_PREDICT_K, default 0 = off):
/// under ignore_eos the per-slot remaining tokens are exact, so when fewer
/// than K slots are free but at least K will be free within H rounds
/// (PADDOCK_ADM_PREDICT_H), hold the refill and admit the bunch together -
/// one shared prefill wave instead of a sprinkle. The hold self-expires (the
/// counted slots complete within H rounds by construction), bounding the
/// added TTFT at ~H rounds worst-case and ~zero once the cohort phase-locks
/// (the wm3 fast-mode evidence: bunched boots ran -2.5ms/round ITL).
fn adm_predict_k() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_ADM_PREDICT_K")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

fn adm_predict_h() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_ADM_PREDICT_H")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(16)
    })
}

fn adm_max_hold_ms() -> u64 {
    static V: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_ADM_MAX_HOLD_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(25)
    })
}

/// In-leg cohort resync (PADDOCK_ADM_RESYNC=R, default 0 = off): the K=16
/// pipe gate keeps an aligned cohort but cannot FORCE alignment from a
/// staggered start (pipe-gate A/B: 1 lock / 6 legs, the lock born from a
/// full-drain wave). This forces it once per leg: after the cohort first
/// FILLS, hold the pipe poll until R slots are free, admit them as one
/// giant wave (a manufactured drain), then fall back to K maintenance.
/// Re-arms only when the server drains (leg boundary), so steady serving
/// pays the hold exactly once per leg.
fn adm_resync() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_ADM_RESYNC")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

/// The continuous-batching scheduler. New sequences are bulk-prefilled on
/// admission (whole prompt in one pass -> first token), then every tick runs one
/// `forward_batch` decode step over all active sequences (weights read once for
/// the batch). A sequence keeps its KV slot for life so its cache stays
/// contiguous - the batch row equals the KV slot, and idle rows below the
/// high-water mark are fed a harmless dummy token.
fn run_batched(
    generator: &mut dyn Generator,
    rx: &Receiver<GenRequest>,
    max_batch: usize,
    vocab: usize,
    metrics: &Arc<EngineMetrics>,
    ctl: &ShutdownCtl,
) {
    let mut slots: Vec<Option<Slot>> = (0..max_batch).map(|_| None).collect();
    // last refresh of the published device ledger (see `refresh_model_mem`)
    let mut mem_seen: Option<std::time::Instant> = None;
    // Closed-loop speculation control (docs: crate::spec_policy). Under `auto`
    // this replaces the hand-tuned batch->K ladder with a goodput argmax that
    // re-decides every round; under the default `ladder` it hands the legacy
    // rule straight back, so this commit changes nothing until asked.
    let spec_policy = serve_spec_policy();
    let mut spec_ctl = SpecController::new(spec_policy);
    // What the in-flight tick chose, so the next loop-top can attribute its
    // wall time to the right (live, K) cell. Timed loop-top to loop-top: that
    // window includes the host-side accept/SSE/admit work, which is real
    // wall-clock cost between tokens and belongs in a goodput denominator.
    // The bool marks MIXED rounds - their prompt-chunk surcharge is
    // k-independent, so they must not price the per-k comparison (see the
    // observe() call at loop-top).
    let mut tick_open: Option<(std::time::Instant, usize, usize, bool)> = None;
    let mut tick_tally = RoundTally::default();
    if spec_policy != SpecPolicy::Ladder {
        tracing::info!("speculation policy: {spec_policy}");
    }
    // Speculative rounds stay on until the backend says it can't
    // (forward_spec_batch -> Ok(None)) or the env pin turns them off.
    let spec_supported =
        paddock_models::dev_var_os!("PADDOCK_NO_SERVE_SPEC").is_none() && !spec_policy.is_off();
    // Say so when the operator asked for speculation and this serve cannot do
    // it. Until 2026-09-19 that combination was silent: `--spec on --mtp <head>`
    // loaded the drafter, held its VRAM (2.74 GiB on Flash-Next) and then
    // served every token dense, at exactly the no-spec rate. A user reported it
    // as "we cannot get above 25 tok/s" and the only way to tell from outside
    // was to scrape paddock_spec_decode_draft_tokens_total and find a zero.
    // A serving envelope the operator explicitly asked for and did not get is
    // the definition of a silent failure, so it is a WARN on every serve.
    // EXPLICITLY: a `spec` setting, or a drafter file the runner was handed
    // (it exports the default policy by name then, see the runner's spec
    // resolution). The default policy alone is not a request - it used to put
    // this WARN on every start of every model that has nothing to draft with
    // (Flash-Next's NVFP4 export from the Studio, for one), which is noise
    // that teaches operators to ignore the one time it matters.
    let spec_explicit = std::env::var_os("PADDOCK_SPEC").is_some();
    if spec_supported && !generator.spec_capable() && !spec_explicit {
        tracing::info!(
            "speculation is on by default but this serve has no drafter to run it - \
             every token decodes at the no-spec rate"
        );
    }
    if spec_supported && !generator.spec_capable() && spec_explicit {
        tracing::warn!(
            policy = %spec_policy,
            "speculation was requested (spec = {spec_policy}) but this serve CANNOT speculate: \
             no drafter is resident, or the kernel pack lacks an entry the spec path needs. \
             Every token will decode at the no-spec rate. Check that a drafter is attached \
             (--mtp for a sideloaded head, or an in-file MTP block) and that the kernel pack \
             in use ships the sampling entries; `paddock_spec_decode_draft_tokens_total` on \
             /metrics stays 0 while this holds."
        );
    }
    // Watchdog for the other half of the same failure: capable AND enabled,
    // and still drafting nothing. Capability is a load-time answer; drafting is
    // a per-round one, and a drafter that never gets warm (or whose rounds are
    // all declined) reproduces the exact user-visible symptom with every
    // startup check passing. Counted over ticks that actually ran a spec round,
    // and said ONCE - this is a serving-envelope fault, not a per-tick event.
    let mut spec_rounds_seen = 0u64;
    let mut spec_drafted_seen = 0u64;
    let mut spec_silent_warned = false;
    // ~256 decoding rounds is a few seconds of serving at any width, and far
    // past the cooldown a legitimately-declining drafter uses (see the Ok(None)
    // note above), so it cannot fire on a drafter that is merely warming up.
    const SPEC_SILENT_ROUNDS: u64 = 256;
    // Ok(None) is a COOLDOWN, not a permanent disable: model drafters (MTP)
    // legitimately decline single ticks (stale per-slot state after a dense
    // interlude, non-contiguous live set) and become eligible again at the
    // next prefill. A backend with no spec at all just declines once per
    // cooldown window - one wasted probe call per 256 ticks.
    let mut spec_ticks = 0u64;
    let mut spec_retry_at = 0u64;
    // In-leg resync state (see adm_resync): armed at boot / re-armed on
    // near-drain; the hold engages when the cohort first fills and releases
    // (once) when R slots are free - the giant wave the outer refill admits.
    let mut resync_armed = adm_resync() > 0;
    let mut resync_hold = false;
    // Device-sampled spec engages only up to this live count. 8 is the
    // measured boundary: with the rebuilt round (ragged rounds + per-live
    // graphs + async draft chain + qwen35's forward_mixed_spec_plans
    // actually implemented) c4 and c8 both gain, while cap 16 measures
    // worse than dense - the round->span serialization doesn't repay at that
    // width. c16+ keep the fusion until the overlapped round (begin/finish
    // split) or span riders land.
    let dev_spec_live_max: usize = std::env::var("PADDOCK_QWEN35_SPEC_PLANS_LIVE_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8)
        .min(serve_spec_max_rows())
        // never engage spec (warmth checks included) past what the backend's
        // draft-state allocation can actually serve - rounds would decline and
        // ensure_warm would re-warm an unsyncable gap every tick
        .min(generator.spec_live_cap());
    // same backend clamp for the host-2a greedy round (its own gate is the
    // row budget, which doesn't know about a VRAM-degraded draft alloc)
    let spec_live_cap = generator.spec_live_cap();
    let fixed_draft_depth = generator.spec_fixed_draft_depth();
    // Fused device sampling: eligible decode rows come back as bare token ids
    // instead of [B, vocab] logits (25.7 MB/step at B=32) + host sampling.
    // Probed before any per-row uniforms are drawn, so seed streams never pay
    // for a path that won't run. PADDOCK_NO_GPU_SAMPLE pins the host path.
    let samp_supported = generator.supports_device_sampling()
        && paddock_models::dev_var_os!("PADDOCK_NO_GPU_SAMPLE").is_none();
    // Pipelined pure-decode ticks (depth-2): tick N+1 is enqueued on device
    // before tick N's ids reach the host, so commit + SSE work overlaps the
    // GPU instead of gapping it between step-graph replays (at c8 the GPU
    // sat ~7% idle - ~1 ms of host work per ~13 ms tick). The backend
    // probe also fails under the numerics-pin envs; PADDOCK_NO_DECODE_PIPE
    // pins the classic tick-at-a-time path.
    let pipe_supported = samp_supported
        && generator.supports_decode_pipe()
        && paddock_models::dev_var_os!("PADDOCK_NO_DECODE_PIPE").is_none();
    // Device greedy for ngram-only rows: granted per tick, only
    // when the guard would ban nothing at the row's CURRENT history - which
    // is why it must never coexist with the pipe's plan lookahead (tick N+1's
    // plans are drawn before tick N's token lands; the check would go stale).
    let ngram_dev_ok = samp_supported && !pipe_supported && generator.device_greedy_ngram_ok();
    // Chunked prefill (vLLM-class): with live decode streams, admissions
    // advance prefill_tick_rows() per mixed tick instead of freezing every
    // stream for their whole prompts. `chunking` = the slots currently
    // advancing - SEVERAL ride each mixed tick (the old one-at-a-time gate
    // serialized staggered admission waves into N sequential heavy mixed
    // ticks: a fast-cohort / slow-staggered coin flip per run).
    let chunk_ok = generator.supports_chunked_prefill()
        && paddock_models::dev_var_os!("PADDOCK_NO_CHUNKED_PREFILL").is_none();
    // True unified prefill+decode batching: fuse queued prompts into the decode
    // forward (one weight read) instead of the mixed tick's two forwards.
    // DEFAULT-ON for qwen35 via GpuQwen35::apply_default_stack
    // (PADDOCK_UNIFIED=1 + PREFILL_ROWS=2048; kill: PADDOCK_NO_UNIFIED). It
    // only pays with FAT spans: a light per-tick prefill budget regressed
    // hard on every shape until the row budget went up (2048-row spans) and
    // the span attn/DeltaNet/W8/x2 paths plus the spec warm hook were wired.
    let unified_ok = chunk_ok && std::env::var_os("PADDOCK_UNIFIED").is_some();
    // scheduler + backend queue bound on prompts advancing through mixed
    // ticks at once (see max_chunks_inflight)
    let mut chunking: std::collections::HashSet<usize> = std::collections::HashSet::new();
    // time-budgeted mixed ticks at long context, every family (crate::pacing)
    let mut pacers = crate::pacing::Pacers::new();
    // In-flight shared-prefix dedupe - SGLang's in-batch prefix caching in
    // the form a DeltaNet hybrid needs. A cohort that arrives together and
    // shares a prefix (one system prompt, N users; a benchmark's shared
    // prefix) used to prefill it N times: the radix only carries a prefix
    // once its prefill has LANDED, and by then every peer had already been
    // admitted beside it. Measured on a 5060 Ti at c8 x 427 shared tokens:
    // TTFT p50 1184 ms and an inter-token p99 of 88 ms (the early finishers
    // decoded inside the peers' prefill ticks), against 251 ms / 30.5 ms once
    // the prefix was cached. So: the prompts currently chunking, by slot,
    // with a prefix of their tokens; a pending prompt that shares at least
    // `prefix_share_floor` tokens with one of them waits for that leader to
    // land, and the leader is told where each follower diverges so it
    // snapshots resumable state there (`prefill_begin_hinted`).
    let mut inflight_prompts: Vec<(usize, Vec<u32>, usize)> = Vec::new();
    // followers currently held back (trace bookkeeping: log each once)
    let mut prefix_deferred: std::collections::HashSet<usize> = std::collections::HashSet::new();
    // Slots whose images the BACKEND is still encoding under its encoder budget
    // It owns their chunks - the slot's own `mm` is already gone -
    // so a slot in here must be offered to neither the mm wave (it has nothing
    // left to offer) nor the text lane, which would otherwise prefill it from
    // the prompt the wave builder cleared.
    let mut mm_encoding: std::collections::HashSet<usize> = std::collections::HashSet::new();
    // Issue-ahead: finished-prefill work deferred one round - it
    // runs inside the next mixed round's GPU window (between begin and
    // finish) instead of stalling the stream (~1.4ms pure-host pick_next +
    // drafter seeding per finisher, worth 6-8% GPU idle at the mixed
    // boundaries). A deferred slot stays in `chunking` (no readmit)
    // and is not `prefilled` (no decode rows) until its finish runs - one
    // idle round for that slot, bounded and safe.
    let mut mix_deferred: Vec<(
        usize,
        crate::generator::FinishSample,
        usize,
        Option<crate::sampler::DevicePlan>,
    )> = Vec::new();
    // [mix-bnd] timestamp: set when a mixed round's service work completes,
    // read at the next round's draft entry (boundary attribution)
    let mut mix_bnd: Option<std::time::Instant> = None;
    // decode-priority interleave counter (see the mixed-tick gate below)
    let mut dp_count: u32 = 0;
    // Quiet-period pipe economics: a pipe segment's cold begin (~1.4 ms
    // 2000-node graph launch) only amortizes over LONG decode stretches.
    // Admission-heavy regimes (short-prompt churn: an arrival every ~4
    // ticks) measured the pipe neutral-to-negative, steady-width regimes
    // slightly positive: engage only after PADDOCK_PIPE_MIN_QUIET
    // admission-free ticks (default 8).
    let mut ticks_since_admit: u32 = u32::MAX;
    let mut adm_hold_since: Option<std::time::Instant> = None;
    let pipe_min_quiet: u32 = paddock_models::dev_var!("PADDOCK_PIPE_MIN_QUIET")
        .ok()
        .and_then(|v| v.parse().ok())
        // Class-split default. No-spec lanes: 4 - under churn the classic
        // tick's ~1.4 ms host turnaround is fully exposed (1.07 s of
        // inter-tick idle in a 25.7 s window) and the pipe erases it
        // (quiet=1 measured identical to 4; tickstats 112x pipe / 4x
        // classic, drains 1). Spec lanes keep 512 (~15 s): for the spec
        // class, admission-bearing cells measured best on the mixed flow,
        // and that regime has not been re-measured on the current engine -
        // re-check before touching it. Env overrides both.
        //
        // The split keys on the RESOLVED policy, not bare capability:
        // granite/laguna report spec_capable()=true
        // unconditionally (the machinery exists), so a --spec off serve
        // inherited the 512-tick spec-lane quiet and the decode pipe never
        // engaged inside a normal request (~0.6 ms/token of host turnaround
        // fully exposed - the tickstats probe read 323 classic / 0 pipe).
        // "Spec lane" means a lane that will actually RUN spec: capability
        // AND policy.
        .unwrap_or(
            if generator.spec_capable() && spec_policy != SpecPolicy::Off {
                512
            } else {
                4
            },
        );
    // tick-type accounting for the admission-mode investigation
    // (PADDOCK_TICK_STATS=1 prints a summary line every ~5 s)
    let tick_stats = paddock_models::dev_var_os!("PADDOCK_TICK_STATS").is_some();
    // spec rounds and mixed-spec ticks get their own buckets - the
    // wide-spec A/B improved the round economics yet lost end-to-end, so the
    // loss must live in a regime the old 4-bucket table never timed. spec =
    // pure verify rounds (2a/2b-dev, warmth+chain+verify). mspec = mixed
    // ticks that took the spec branch (warmth+chain+pass); its time was
    // previously booked under `mixed` minus the drafter prologue, which was
    // invisible entirely.
    let mut st: [(u64, u64); 6] = [(0, 0); 6]; // [pipe, mixed, classic, prefill, spec, mspec] = (count, ns)
    // tickseg gap anchor: set when a tick's host work ends, read when the
    // next instrumented forward starts - the in-between is scheduler/
    // admission/encode-planning time the GPU spends idle (probe)
    let mut seg_mark: Option<std::time::Instant> = None;
    // Cadence probe: admissions (chunk starts), and per-mspec-tick
    // sums of in-flight chunks and decode rows - separates "more prompt
    // volume" from "same volume spread over more weight-stream passes".
    let mut st_adm = 0u64;
    let mut st_mchk = 0u64;
    let mut st_mdec = 0u64;
    let mut st_drain = 0u64;
    // Adaptive pipe backoff: segments that keep dying young (admission
    // churn - a synchronized c32 load admits every ~4 ticks) pay the ~1.4 ms
    // cold begin without amortizing it, while long-segment regimes win
    // outright. Short segment => exponential extra-quiet, long segment =>
    // decay - the pipe self-selects into the regimes where it pays.
    let mut pipe_backoff: u32 = 0;
    let mut pipe_seg_ticks: u32 = 0;
    // overlapped decode-pipe ticks pumped while a span was in flight (2o)
    let mut st_ovl = 0u64;
    let mut st_last = std::time::Instant::now();
    // Multimodal requests ride ordinary batch slots when the backend supports
    // it. Otherwise the older exclusive path: the request waits here for
    // the active batch to drain and blocks new admissions meanwhile, so it
    // cannot starve behind a stream of text requests.
    let mm_slots = generator.supports_mm_slots();
    let mut mm_pending: Option<GenRequest> = None;
    // true once the idle scheduler has handed its freed memory back; re-armed
    // by any live request (see `Generator::release_idle_memory`)
    let mut idle_released = true;
    // Reused per-tick occupancy mask for the paged free-on-completion hook.
    let mut occupied: Vec<bool> = Vec::with_capacity(max_batch);
    let pool_stats = paddock_models::dev_var_os!("PADDOCK_POOL_STATS").is_some();
    // Sequences PREEMPTED under pool pressure, awaiting recompute. They
    // hold their full state (sampler/history/events) and are re-admitted (with
    // priority over new arrivals) once the pool has room for their recompute.
    let mut preempted: Vec<Slot> = Vec::new();
    // start time of a held-back lone chunk admission (coalescing).
    let mut coalesce_since: Option<std::time::Instant> = None;
    // Warm-server straggler self-report: whole client-visible
    // wave stalls (~450/~850 ms first-prefill-tick walls, host-state
    // dependent) left nothing in the serve log. Any tick that blows past
    // this wall now names itself. 0 disables; loop-top to loop-top, with
    // the blocked idle recv excluded below.
    let stall_warn_ms = static_env_u64("PADDOCK_TICK_STALL_WARN_MS", 750);
    let mut tick_t0 = std::time::Instant::now();
    // Phase marks for the straggler self-report. A stall used to name only its
    // total, which is not actionable: an excursion is a handful of ~1 s ticks
    // (100x a normal tick) and an external profiler cannot attribute it -
    // the release build is stripped and the pdfium crate refuses to relink
    // under debug flags. So the loop reports its own breakdown, which needs
    // no symbols.
    let (mut ph_admit, mut ph_prefill, mut ph_mixed, mut ph_spec, mut ph_decode) = (
        std::time::Duration::ZERO,
        std::time::Duration::ZERO,
        std::time::Duration::ZERO,
        std::time::Duration::ZERO,
        std::time::Duration::ZERO,
    );
    let mut last_stall_warn: Option<std::time::Instant> = None;

    loop {
        // Every reply that opened its first tool call last tick has its
        // checkpoint held before any forward can move the slot past the call.
        drain_reply_pins(generator, &mut slots);
        // Close out the previous tick for the controller before anything else
        // touches the clock. Every decoding tick reports - including the plain
        // dense ones at k=0, which are the baseline the speculative cells are
        // compared against; a controller fed only speculative rounds can never
        // discover that not speculating is faster.
        if let Some((t0, live, k, mixed)) = tick_open.take() {
            // Mixed rounds carry a prompt-chunk surcharge that is
            // k-independent, so booking their wall into lat[bucket][k]
            // poisons the per-k comparison the argmax runs on pure rounds
            // (mixed k=3 books 35-48ms against pure k=3's 17.8ms in the
            // shared 17..32 bucket, and the goodput flip then drops pure
            // rounds to k=2/1). PADDOCK_SPEC_BOOK_MIXED=1 restores the old
            // booking (the A/B surface).
            if !mixed || spec_book_mixed() {
                spec_ctl.observe(live, k, t0.elapsed().as_secs_f64(), tick_tally);
            }
            // The watchdog reads the tick that just closed: k > 0 means this
            // round asked for drafts, so a round that asked and got none is the
            // thing being counted. k == 0 is a plain dense tick and proves
            // nothing about the drafter.
            if spec_supported && k > 0 {
                spec_rounds_seen += 1;
                spec_drafted_seen += tick_tally.drafted as u64;
                if !spec_silent_warned
                    && spec_rounds_seen >= SPEC_SILENT_ROUNDS
                    && spec_drafted_seen == 0
                {
                    spec_silent_warned = true;
                    tracing::warn!(
                        rounds = spec_rounds_seen,
                        "speculation is enabled and the backend reports it is capable, but \
                         {spec_rounds_seen} rounds have drafted ZERO tokens - this serve is \
                         running at the no-spec rate while paying for a resident drafter. \
                         The drafter is never reaching a warm slot (PADDOCK_SPEC_DEBUG=1 \
                         prints the per-round chunk lengths)."
                    );
                }
            }
        }
        tick_tally = RoundTally::default();
        crate::tickseg::maybe_dump();
        if tick_stats && st_last.elapsed().as_secs() >= 5 {
            tracing::info!(
                "tickstats: pipe {}x/{:.2}s mixed {}x/{:.2}s classic {}x/{:.2}s prefill {}x/{:.2}s spec {}x/{:.2}s mspec {}x/{:.2}s drains {}",
                st[0].0,
                st[0].1 as f64 / 1e9,
                st[1].0,
                st[1].1 as f64 / 1e9,
                st[2].0,
                st[2].1 as f64 / 1e9,
                st[3].0,
                st[3].1 as f64 / 1e9,
                st[4].0,
                st[4].1 as f64 / 1e9,
                st[5].0,
                st[5].1 as f64 / 1e9,
                st_drain
            );
            tracing::info!(
                "tickstats2: adm {} mchk {} mdec {} ovl {}",
                st_adm,
                st_mchk,
                st_mdec,
                st_ovl
            );
            st = [(0, 0); 6];
            st_adm = 0;
            st_mchk = 0;
            st_mdec = 0;
            st_drain = 0;
            st_ovl = 0;
            st_last = std::time::Instant::now();
        }
        let active = slots.iter().filter(|s| s.is_some()).count();
        if ctl.stop_requested() {
            tracing::info!("engine: shutdown requested - leaving the scheduler ({active} live)");
            return;
        }
        // Straggler self-report: one full iteration (admission + tick body)
        // took `tick_wall`. active > 0 keeps legit quiet paths (mm exclusive
        // runs, shutdown) silent; 1/s throttle keeps a wedged config readable.
        let tick_wall = tick_t0.elapsed();
        if stall_warn_ms > 0
            && tick_wall.as_millis() as u64 >= stall_warn_ms
            && active > 0
            && last_stall_warn.is_none_or(|t| t.elapsed().as_secs() >= 1)
        {
            last_stall_warn = Some(std::time::Instant::now());
            let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
            let (phase, early_exit_phase) = timing::spans(
                tick_wall,
                [ph_admit, ph_prefill, ph_mixed, ph_spec, ph_decode],
            );
            let phases = format!(
                "admit {:.0} prefill {:.0} mixed {:.0} spec {:.0} decode {:.0} sample+emit {:.0} ms",
                ms(phase[0]),
                ms(phase[1]),
                ms(phase[2]),
                ms(phase[3]),
                ms(phase[4]),
                ms(phase[5]),
            );
            let wall_ms = tick_wall.as_secs_f64() * 1e3;
            // Who waited on this tick: live sequences not mid-way through a
            // chunked prefill or a stepped multimodal one (so decoding, or
            // taking a one-tick prefill) and the preempted ones queued for
            // recompute. With none, every live sequence was spending its own
            // planned prefill - a long prompt's chunk ticks run past the wall
            // by design, one per second for the whole prompt (a 178K prompt on
            // the 27B put ~120 WARNs in a customer log), so those go to debug
            // and the WARN keeps meaning a client was held.
            let waiting =
                active.saturating_sub(chunking.len() + mm_encoding.len()) + preempted.len();
            if waiting > 0 {
                tracing::warn!(
                    ?early_exit_phase,
                    "tick-stall phases: {phases} (unfinished branch includes its emit work)"
                );
                tracing::warn!(
                    "tick-stall: {wall_ms:.0} ms wall (live {active}, chunking {}, preempted {})",
                    chunking.len(),
                    preempted.len()
                );
            } else {
                tracing::debug!(
                    ?early_exit_phase,
                    "tick-stall (prefill only, nobody waiting): {wall_ms:.0} ms wall (live \
                     {active}, chunking {}): {phases}",
                    chunking.len()
                );
            }
        }
        tick_t0 = std::time::Instant::now();
        ph_admit = std::time::Duration::ZERO;
        ph_prefill = std::time::Duration::ZERO;
        ph_mixed = std::time::Duration::ZERO;
        ph_spec = std::time::Duration::ZERO;
        ph_decode = std::time::Duration::ZERO;
        // telemetry (lock-free): batch width, phase, KV occupancy this tick.
        metrics.active_slots.store(active as u32, Relaxed);
        metrics
            .phase
            .store(if active > 0 { PHASE_DECODE } else { PHASE_IDLE }, Relaxed);
        refresh_model_mem(generator, metrics, &mut mem_seen, MODEL_MEM_REFRESH);
        if let Some(free) = generator.pool_free_blocks() {
            let free = free as u32;
            // First idle tick sees free == capacity; latch it as the total.
            if metrics.kv_total.load(Relaxed) < free {
                metrics.kv_total.store(free, Relaxed);
            }
            metrics
                .kv_used
                .store(metrics.kv_total.load(Relaxed).saturating_sub(free), Relaxed);
        }
        // P5b free-on-completion: return finished slots' KV blocks to the paged
        // pool the moment their sequence ends (idempotent; no-op unless a budget
        // pool is active). Without this a completed slot pins its blocks until
        // the slot is next reused by a new admission.
        occupied.clear();
        occupied.extend(slots.iter().map(|s| s.is_some()));
        generator.release_inactive_slots(&occupied);
        // kv-tier maintenance rides the same per-pass slot (no-op untiered)
        generator.tier_pump();
        if let Some(rep) = generator.tier_report() {
            metrics.tier.store(&rep);
        }
        // P5b admission watermark: free blocks the pool has for new prefills this
        // tick (None = no pool -> admit by slots only). Decremented per admit by
        // the prompt's block estimate; the try_recv loop stops once it nears the
        // watermark, so a burst can't over-commit and in-flight sequences keep
        // room to grow. Bit-exact - this only delays admission.
        //
        // The budget must also carry the ALREADY-ADMITTED commitments that have
        // not claimed blocks yet: pool_free_blocks() alone forgets them the
        // tick after admission (chunk prefills claim lazily), so a wide burst
        // re-admitted past pool capacity every tick until the preempt/readmit
        // cycle wedged the server (found live: gemma4 2048x128c32, 31 slots
        // admitted against a 1703-block pool that holds ~16 such prompts).
        // An un-prefilled slot's full need is ceil(history/BT) - history is the
        // whole prompt before decode starts. Chunking slots double-count the
        // blocks their in-flight chunks already claimed (pool free dropped AND
        // still counted here) - conservative, only delays new admissions.
        let committed: usize = slots
            .iter()
            .flatten()
            .filter(|s| !s.prefilled)
            .map(|s| s.history.len().div_ceil(crate::kv_pool::BLOCK_TOKENS))
            .sum();
        let mut admit_budget = generator
            .pool_admission_blocks()
            .map(|f| f.saturating_sub(committed));
        // P5b-3: re-admit preempted (recompute) sequences before new arrivals -
        // they are already-accepted in-flight work. Each needs room for its
        // recompute prefill (history[0..pos]); gate on the same watermark and a
        // free slot, else leave it queued and retry as completions free blocks.
        while let Some(front) = preempted.first() {
            let need = front.prompt.len().div_ceil(crate::kv_pool::BLOCK_TOKENS);
            let fits = admit_budget.is_none_or(|b| need + POOL_WATERMARK_BLOCKS <= b);
            let free = slots.iter().position(Option::is_none);
            match (fits, free) {
                (true, Some(k)) => {
                    if let Some(b) = admit_budget.as_mut() {
                        *b = b.saturating_sub(need);
                    }
                    slots[k] = Some(preempted.remove(0));
                    if pool_stats {
                        tracing::warn!("pool: RESUMED preempted seq into slot {k} ({need} blocks)");
                    }
                }
                _ => break,
            }
        }
        let mut admitted_now: u32 = 0;
        if active == 0
            && let Some(req) = mm_pending.take()
        {
            run_request(generator, req, metrics);
            continue;
        }
        // Nothing in flight: block for the next request (and exit when the last
        // sender is dropped). Otherwise keep the batch moving without blocking.
        // Going idle is also when the pool hands back what the finished
        // requests freed - once, not every 250 ms wake.
        if active > 0 {
            idle_released = false;
        } else if !idle_released {
            generator.release_idle_memory();
            // and the ledger with it: it refreshes on busy passes only, so an
            // idle server would keep publishing its last busy reading
            refresh_model_mem(generator, metrics, &mut None, MODEL_MEM_REFRESH);
            idle_released = true;
        }
        if active == 0 {
            let received = loop {
                match rx.recv_timeout(std::time::Duration::from_millis(250)) {
                    Ok(r) => break Ok(r),
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                        if ctl.stop_requested() {
                            tracing::info!(
                                "engine: shutdown requested - leaving the scheduler (idle)"
                            );
                            return;
                        }
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                        break Err(std::sync::mpsc::RecvError);
                    }
                }
            };
            // blocked-idle waiting for traffic is not tick work
            tick_t0 = std::time::Instant::now();
            match received {
                Ok(req) if req.mm_chunks.is_some() && !mm_slots => {
                    mm_pending = Some(req);
                    continue; // top of loop runs it (active is still 0)
                }
                Ok(req) => {
                    debit_pool_budget(&mut admit_budget, &req);
                    admit(&mut slots, req, generator.max_context(), metrics);
                    admitted_now += 1;
                    ticks_since_admit = 0;
                    // a fresh prefill re-warms the draft head - make spec
                    // eligible immediately (the cooldown comment's "eligible
                    // again at the next prefill", now actually implemented)
                    spec_retry_at = spec_ticks;
                }
                Err(_) => return, // all senders gone
            }
        }
        // Pull in as many waiting requests as there are free slots - unless an
        // exclusive request is waiting for the drain or the pool watermark is hit
        // (remaining requests wait in the channel; completions reopen admission).
        // Min-free admission gate - see adm_min_free(). Open when disarmed,
        // at low batch, when K slots have freed, or after the hold cap.
        let adm_open = {
            // predictive bunching gate (see adm_predict_k) - takes precedence
            // over the plain min-free hold when armed.
            let pk = adm_predict_k();
            if pk > 0 && slots.len() >= 2 * pk {
                let mut free = 0usize;
                let mut soon = 0usize;
                let h = adm_predict_h();
                for s2 in slots.iter() {
                    match s2 {
                        None => free += 1,
                        Some(sl) => {
                            if sl.max_tokens.saturating_sub(sl.generated) <= h {
                                soon += 1;
                            }
                        }
                    }
                }
                let hold = free > 0 && free < pk && free + soon >= pk;
                // throttled diagnostic: what does the gate actually see?
                {
                    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
                    static EVALS: AtomicUsize = AtomicUsize::new(0);
                    static HOLDS: AtomicUsize = AtomicUsize::new(0);
                    if hold {
                        HOLDS.fetch_add(1, Relaxed);
                    }
                    let e = EVALS.fetch_add(1, Relaxed);
                    if e.is_multiple_of(256) {
                        tracing::info!(
                            free,
                            soon,
                            holds = HOLDS.load(Relaxed),
                            evals = e,
                            "adm-predict gate state"
                        );
                    }
                }
                // hold only when a K-bunch is imminent; stragglers admit now
                !hold
            } else {
                let k = adm_min_free();
                if k == 0 || slots.len() < 2 * k {
                    true
                } else {
                    let free = slots.iter().filter(|s| s.is_none()).count();
                    if free == 0 || free >= k {
                        adm_hold_since = None;
                        true
                    } else {
                        let t0 = *adm_hold_since.get_or_insert_with(std::time::Instant::now);
                        if t0.elapsed().as_millis() as u64 >= adm_max_hold_ms() {
                            adm_hold_since = None;
                            true
                        } else {
                            false
                        }
                    }
                }
            }
        };
        while adm_open
            && mm_pending.is_none()
            && slots.iter().any(|s| s.is_none())
            && admit_budget.is_none_or(|b| b > POOL_WATERMARK_BLOCKS)
        {
            match rx.try_recv() {
                Ok(req) if req.mm_chunks.is_some() && !mm_slots => {
                    mm_pending = Some(req);
                    break;
                }
                Ok(req) => {
                    debit_pool_budget(&mut admit_budget, &req);
                    admit(&mut slots, req, generator.max_context(), metrics);
                    admitted_now += 1;
                    {
                        ticks_since_admit = 0;
                    }
                    spec_retry_at = spec_ticks; // eligible again at the next prefill
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break, // let active slots drain
            }
        }

        // Cohort-fuse linger - see cohort_fuse() for the trace evidence. Exit
        // after 2 ms of silence or the 8 ms hard cap; every admit here is one
        // the drain above would have taken next round at +~190 ms TTFT.
        let lone_grace = if active == 0 && admitted_now == 1 && slots.iter().flatten().count() == 1
        {
            lone_cohort_grace(
                active,
                admitted_now,
                slots.iter().flatten().all(|s| s.mm.is_none()) && mm_pending.is_none(),
                slots
                    .iter()
                    .flatten()
                    .next()
                    .map_or(std::time::Duration::ZERO, |s| {
                        generator.idle_admission_grace(&s.prompt)
                    }),
            )
        } else {
            std::time::Duration::ZERO
        };
        if cohort_fuse() && active == 0 && (admitted_now >= 2 || !lone_grace.is_zero()) {
            // caps env-tunable for the c128 cohort-former (defaults = shipped
            // 8ms/2ms): a 128-connection ramp takes ~25ms+ and the 8ms cap
            // splits it into sub-cohorts before any downstream collector runs.
            let hard_ms = static_env_u64("PADDOCK_COHORT_FUSE_HARD_MS", 8);
            let quiet_ms = static_env_u64("PADDOCK_COHORT_FUSE_QUIET_MS", 2);
            // A qualified lone admission shares this window; do not stack a
            // second linger or extend its deadline on repeated arrivals.
            let hard_budget = std::time::Duration::from_millis(hard_ms);
            let quiet_budget = std::time::Duration::from_millis(quiet_ms);
            let (hard_budget, quiet_budget) = if lone_grace.is_zero() {
                (hard_budget, quiet_budget)
            } else {
                (hard_budget.min(lone_grace), quiet_budget.min(lone_grace))
            };
            let hard = std::time::Instant::now() + hard_budget;
            let mut quiet = std::time::Instant::now();
            while std::time::Instant::now() < hard
                && quiet.elapsed() < quiet_budget
                && mm_pending.is_none()
                && slots.iter().any(|s| s.is_none())
                && admit_budget.is_none_or(|b| b > POOL_WATERMARK_BLOCKS)
            {
                match rx.try_recv() {
                    Ok(req) if req.mm_chunks.is_some() && !mm_slots => {
                        mm_pending = Some(req);
                        break;
                    }
                    Ok(req) => {
                        debit_pool_budget(&mut admit_budget, &req);
                        admit(&mut slots, req, generator.max_context(), metrics);
                        ticks_since_admit = 0;
                        spec_retry_at = spec_ticks;
                        quiet = std::time::Instant::now();
                    }
                    Err(TryRecvError::Empty) => {
                        std::thread::sleep(std::time::Duration::from_micros(200));
                    }
                    Err(TryRecvError::Disconnected) => break,
                }
            }
        }
        ph_admit = tick_t0.elapsed();
        // Phase 1 - prefill newly admitted sequences. With live decode streams
        // and a chunk-capable backend, text admissions take CHUNKED prefill:
        // one budgeted chunk per tick rides a mixed pass alongside every
        // decode row, so a 4k-token admission never stalls the streams (the
        // blocking batched pass froze all output ~3.8 s per 8x4k admission
        // wave - llama-server's ubatch scheduler won pf8 on exactly that).
        // Cold bursts (nothing decoding) keep the batched whole-prompt pass:
        // there is nothing to stall, and packing several divergent tails into
        // one pass amortizes the weights best. Multimodal prompts always take
        // the classic path (their image rows don't chunk).
        let _live_decodes = slots
            .iter()
            .any(|s| s.as_ref().is_some_and(|sl| sl.prefilled));
        // Draft-head warm hint: prefills only pay the eager MTP warm pass when
        // the live population could actually ride a spec round (measured c8
        // -2..3% from warming 8 slots the live cap then locked out of spec).
        // Late admissions after the population thins do warm and join spec.
        {
            let live_now = slots.iter().filter(|s| s.is_some()).count();
            // ...and only while speculation is actually the plan for this
            // width. The caps below are structural ("could a round serve this
            // cohort at all"); they cannot see a controller that has measured
            // this bucket and decided not to speculate. At 8 live slots
            // Flash-Next sits inside every cap, picks k=0 - and used to keep
            // paying a head pass per slot per tick for drafts nobody would
            // ask for: 19 ms of a 45 ms tick, which is the whole c8 margin
            // (2026-09-17).
            let plan_spec = spec_supported
                && spec_ctl.will_speculate(live_now, backend_spec_k_budget(generator, live_now));
            generator.spec_warm_hint(plan_spec && live_now <= dev_spec_live_max);
            // Widths above the spec engagement cap take the dense route (the
            // `greedy` gate below tests the same bound), so a drafter fused
            // from every forward would be feeding a ring nothing reads.
            generator.spec_fuse_hint(plan_spec && live_now <= spec_live_cap);
        }
        // Chunked prefill for all text admissions, cold bursts included: the
        // multi-chunk queue packs up to 8 prompts into the same 2048-row
        // ticks (the batched pass's amortization) while each stream starts
        // decoding the moment its prompt finishes. The old cold-burst
        // batched pass froze first tokens behind the whole cohort (1.8 s
        // freezes in the tick stats; TTFT p50 1.9 s cohort vs 0.7 s
        // staggered at c32) and split serving into two behavioral modes.
        if chunk_ok {
            // Admission coalescing (PADDOCK_ADM_COALESCE_MS, default 0
            // = off): under wide spec, per-slot acceptance variance
            // desynchronizes cohort completions, so arrivals spread and
            // chunks stop sharing mixed passes (measured: 3.24 -> 2.34
            // chunks/tick, +17% whole-weight-stream passes for the same
            // admission count). Holding a lone chunk-start briefly - until a
            // second prompt queues or the deadline passes - restores the
            // packing. Only holds when the machine has decode work to run
            // instead (>= 16 live rows); the wide-mode TTFT surplus
            // (~40-60 ms p50 vs capped) is the budget being traded.
            let co_ms: u64 = paddock_models::dev_var!("PADDOCK_ADM_COALESCE_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            let mut co_hold = false;
            // gate on ACTIVE chunks - deferred-done slots park in
            // `chunking` between waves and were holding this shut.
            if co_ms > 0 && chunking.len() <= mix_deferred.len() {
                let pending = slots
                    .iter()
                    .enumerate()
                    .filter(|(k, s)| {
                        !chunking.contains(k)
                            && s.as_ref()
                                .is_some_and(|sl| !sl.prefilled && sl.mm.is_none())
                    })
                    .count();
                let live_n = slots
                    .iter()
                    .filter(|s| s.as_ref().is_some_and(|sl| sl.prefilled))
                    .count();
                // pair-hold (lone pending under a live population) OR
                // burst accumulation: 2..8 pending - cold bursts included -
                // holds while the wave is still landing so the first pass
                // starts packed toward the 8-prompt span budget (ttft_cap:
                // 22 passes for a 32-wave at ~1.5 prompts/pass). A lone cold
                // admission (pending == 1, live < 16 - the c1 shape) never
                // holds.
                // On a wide cohort the fixed 2..8 release splits a
                // 32-session lockstep cohort into ~11-prompt passes - TTFT
                // p50 350 ms, three serial ~106 ms waves, while the whole
                // 4096-row cohort fits one pass (isolated 245 ms; budget
                // 8160 rows). PADDOCK_ADM_HOLD_MAX raises the release point
                // so the hold keeps accumulating toward the full cohort
                // (default 8 = the behavior, unchanged).
                let hold_max = (static_env_u64("PADDOCK_ADM_HOLD_MAX", 8) as usize).max(3);
                let hold_case = (pending == 1 && live_n >= 16) || (2..hold_max).contains(&pending);
                if hold_case {
                    match coalesce_since {
                        None => {
                            coalesce_since = Some(std::time::Instant::now());
                            co_hold = true;
                        }
                        Some(t) if t.elapsed().as_millis() < co_ms as u128 => {
                            co_hold = true;
                        }
                        Some(_) => {}
                    }
                }
            }
            if !co_hold {
                coalesce_since = None;
            }
            // Multimodal admissions, chunked. Taken as a WAVE and
            // ahead of the text loop: the vision encode is the expensive half
            // and batches across requests, so admitting these one at a time
            // would give up what the batched tower pass exists for. Once
            // queued they are ordinary chunked prompts - image rows carry the
            // placeholder id and `rows_pass_body` resolves each row's features
            // from its (slot, position), so a chunk cut inside a picture is
            // just another cut. Before this an AnyRes page (up to ~2.5k rows)
            // took a blocking whole-prompt pass and froze every live stream.
            // One ENCODER BUDGET per tick: the in-flight wave
            // advances a tile group and decode rows run between the groups
            // instead of waiting out a whole picture. Empty result = still
            // going. Deliberately outside the `co_hold` guard below: an already
            // accepted wave must not be held by a coalescing timer that exists
            // to delay new admissions - that would be a stall of exactly the
            // kind this removes, just moved.
            // Encoder-owned slots are not in `chunking` yet. Reclaim a
            // disconnected upload before spending its next encoder budget.
            let cancelled: Vec<_> = mm_encoding
                .iter()
                .copied()
                .filter(|&k| slots[k].as_ref().is_none_or(|s| s.events.is_closed()))
                .collect();
            for k in cancelled {
                if generator.prefill_abort(k) {
                    mm_encoding.remove(&k);
                    slots[k] = None;
                }
            }
            if generator.encoding_pending() {
                metrics.phase.store(PHASE_PREFILL, Relaxed);
                let seg_t = std::time::Instant::now();
                for (k, res) in generator.encode_step() {
                    mm_encoding.remove(&k);
                    admit_mm(
                        generator,
                        &mut slots,
                        &mut chunking,
                        &mut st_adm,
                        k,
                        res,
                        metrics,
                    );
                }
                if crate::tickseg::on() {
                    crate::tickseg::enc(seg_t.elapsed());
                }
            }
            if generator.supports_chunked_multimodal() && !co_hold {
                let room = max_chunks_inflight()
                    .saturating_sub(chunking.len().saturating_sub(mix_deferred.len()));
                let mut wave: Vec<(usize, Vec<MmChunk>)> = Vec::new();
                for (k, s) in slots.iter_mut().enumerate() {
                    if wave.len() >= room {
                        break;
                    }
                    if let Some(slot) = s
                        && !slot.prefilled
                        && !chunking.contains(&k)
                        && slot.mm.is_some()
                    {
                        slot.prompt.clear();
                        wave.push((k, slot.mm.take().expect("is_some")));
                    }
                }
                if !wave.is_empty() {
                    metrics.phase.store(PHASE_PREFILL, Relaxed);
                    let seg_t = std::time::Instant::now();
                    for (k, res) in generator.prefill_begin_multimodal(wave) {
                        if matches!(res, crate::generator::MmAdmit::Encoding) {
                            mm_encoding.insert(k);
                            continue;
                        }
                        admit_mm(
                            generator,
                            &mut slots,
                            &mut chunking,
                            &mut st_adm,
                            k,
                            res,
                            metrics,
                        );
                    }
                    if crate::tickseg::on() {
                        crate::tickseg::adm(seg_t.elapsed());
                    }
                }
            }
            // Deferred slots sit in `chunking` only to block
            // readmission - they are done chunking, so the in-flight bound
            // counts actives only (else a fully-finished wave starves the
            // next wave's admissions for a round).
            // Burst collection window (chunk lane): defer chunk
            // STARTS while admissions are still in motion and nothing is
            // decoding - a split burst pays 2-3 passes and every prompt's
            // first token waits for the last pass anyway.
            let adm_hold = {
                let win_ms = static_env_u64("PADDOCK_ADM_WINDOW_MS", 0);
                if win_ms > 0
                    && chunking.is_empty()
                    && !slots
                        .iter()
                        .any(|s| s.as_ref().map(|x| x.prefilled).unwrap_or(false))
                {
                    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed as Rl};
                    static LAST_N: AtomicUsize = AtomicUsize::new(0);
                    static STABLE_MS: AtomicU64 = AtomicU64::new(0);
                    let n_wait = slots
                        .iter()
                        .filter(|s| s.as_ref().is_some_and(|sl| !sl.prefilled))
                        .count();
                    let now_ms = service_epoch_ms();
                    if LAST_N.swap(n_wait, Rl) != n_wait {
                        STABLE_MS.store(now_ms, Rl);
                    }
                    // > 1, not > 0: a LONE waiting prompt has nothing to be
                    // batched with, so holding it only adds the window to its
                    // TTFT. At a 5 ms window the hold is a clear win on a
                    // wide cohort (and collapses a bimodal latency cell to a
                    // tight one) while costing the solo case a little.
                    // This keeps the burst behaviour and gives the solo case
                    // back: a burst reaches 2 waiting prompts almost at once,
                    // so it still holds until arrivals stabilize.
                    n_wait > 1 && now_ms.saturating_sub(STABLE_MS.load(Rl)) < win_ms
                } else {
                    false
                }
            };
            // leaders that landed or were abandoned no longer hold anyone back
            inflight_prompts.retain(|(k, _, _)| chunking.contains(k));
            prefix_deferred.retain(|k| {
                !chunking.contains(k) && slots[*k].as_ref().is_some_and(|sl| !sl.prefilled)
            });
            let share_floor = generator.prefix_share_floor();
            while !co_hold
                && !adm_hold
                && chunking.len().saturating_sub(mix_deferred.len()) < max_chunks_inflight()
            {
                let next_pending = slots
                    .iter()
                    .enumerate()
                    .find(|(k, s)| {
                        // `mm_encoding` matters here and nowhere else: the wave
                        // builder already took that slot's chunks, so it reads
                        // as an ordinary text prompt - with the prompt cleared.
                        !chunking.contains(k)
                            && !mm_encoding.contains(k)
                            && s.as_ref().is_some_and(|sl| {
                                !sl.prefilled
                                    && sl.mm.is_none()
                                    // Park/wake (KVFlow): a prefix being
                                    // restored from the KV tier parks its
                                    // request - the batch runs other work and
                                    // the per-pass tier_pump's wake re-enters
                                    // it here. First call starts the restore.
                                    && !generator.tier_prefix_loading(*k, &sl.prompt)
                                    // shared-prefix dedupe: wait for the leader
                                    && shares_inflight(&inflight_prompts, share_floor, *k, &sl.prompt)
                                        .is_none()
                            })
                    })
                    .map(|(k, _)| k);
                let Some(k) = next_pending else { break };
                // Checkpoint hints for this leader: where every other pending
                // prompt that will wait on it diverges (page-floored by the
                // backend). Followers then resume exactly there.
                let hints: Vec<usize> = match share_floor {
                    Some(f) => {
                        let lead = &slots[k].as_ref().expect("present").prompt;
                        slots
                            .iter()
                            .enumerate()
                            .filter(|(m, s)| {
                                *m != k
                                    && !chunking.contains(m)
                                    && s.as_ref()
                                        .is_some_and(|sl| !sl.prefilled && sl.mm.is_none())
                            })
                            .filter_map(|(_, s)| {
                                let l = lcp_len(lead, &s.as_ref().expect("present").prompt);
                                (l >= f).then_some(l)
                            })
                            .collect()
                    }
                    None => Vec::new(),
                };
                let prompt = std::mem::take(&mut slots[k].as_mut().expect("present").prompt);
                let p_rows = prompt.len();
                let keep: Vec<u32> = prompt[..p_rows.min(PREFIX_LCP_CAP)].to_vec();
                match generator.prefill_begin_hinted(k, prompt, &hints) {
                    Ok(resumed) => {
                        if let Some(sl) = slots[k].as_mut() {
                            sl.chunk_started = Some(std::time::Instant::now());
                        }
                        if paddock_models::dev_var_os!("PADDOCK_REQ_TRACE").is_some() {
                            tracing::info!(
                                "req-trace: chunk-start slot {k} rows {p_rows} resumed {resumed} hints {hints:?} at {}",
                                trace_us()
                            );
                        }
                        inflight_prompts.push((k, keep, resumed));
                        chunking.insert(k);
                        st_adm += 1;
                    }
                    Err(e) => {
                        tracing::warn!("serve: prefill_begin failed on slot {k}: {e}");
                        if let Some(s) = slots[k].take() {
                            let _ = s.events.send(TokenEvent::Error(EngineError::from_gen(&e)));
                        }
                    }
                }
            }
        }
        if paddock_models::dev_var_os!("PADDOCK_REQ_TRACE").is_some()
            && let Some(f) = generator.prefix_share_floor()
        {
            for (m, s) in slots.iter().enumerate() {
                let Some(sl) = s.as_ref() else { continue };
                if sl.prefilled || chunking.contains(&m) || prefix_deferred.contains(&m) {
                    continue;
                }
                if let Some((j, l)) = shares_inflight(&inflight_prompts, Some(f), m, &sl.prompt) {
                    prefix_deferred.insert(m);
                    tracing::info!(
                        "req-trace: defer slot {m} - shares {l} tokens with in-flight slot {j} at {}",
                        trace_us()
                    );
                }
            }
        }
        // Nothing newly admitted and no active chunks - no next
        // mixed round is coming, so the deferred finish work has no overlap
        // window; settle it now (the common path drains it inside the next
        // round's GPU window instead).
        if !mix_deferred.is_empty() && chunking.len() == mix_deferred.len() {
            for (k, fs, rows, plan) in std::mem::take(&mut mix_deferred) {
                finish_prefill_any(generator, &mut slots, k, fs, plan, rows as u32, metrics);
                chunking.remove(&k);
            }
        }
        // Text prompts on the classic path collect into one batched call: each
        // reuses its cached prefix, then all divergent tails run together so
        // the weights are read once for the whole set. Each yields its first
        // token immediately. While a chunk is in flight (or streams are live
        // on a chunk-capable backend), text admissions defer to chunked
        // prefill instead.
        let defer_text = chunk_ok;
        // Same rule for MEDIA prompts, against the chunked MM lane rather than
        // the text one - the two are separate capabilities and a backend can
        // have one without the other.
        let defer_mm = generator.supports_chunked_multimodal();
        let mut pending: Vec<(usize, Vec<u32>)> = Vec::new();
        let mut pending_mm: Vec<(usize, Vec<MmChunk>)> = Vec::new();
        for (k, s) in slots.iter_mut().enumerate() {
            if let Some(slot) = s
                && !slot.prefilled
            {
                if slot.mm.is_some() && defer_mm {
                    // A chunk-capable backend admits media on the stall-free
                    // lane above; this classic path is a BLOCKING whole-prompt
                    // pass, so taking the slot here reintroduces exactly the
                    // stall the chunked lane exists to remove. It used to
                    // take it anyway whenever the lane above left a slot
                    // behind - which it does routinely at c32, where its
                    // in-flight room runs out - and on the speech lane that
                    // also meant handing an audio prompt to a vision-only
                    // wave, which 500s the request. Deferring
                    // is what text already does one branch down: the slot is
                    // admitted next tick, by the lane that knows its modality.
                } else if let Some(chunks) = slot.mm.take() {
                    slot.prompt.clear();
                    pending_mm.push((k, chunks));
                } else if !defer_text
                    && !chunking.contains(&k)
                    // park/wake, same rule as the chunked lane above
                    && !generator.tier_prefix_loading(k, &slot.prompt)
                {
                    pending.push((k, std::mem::take(&mut slot.prompt)));
                }
            }
        }
        // Burst collection window (PADDOCK_ADM_WINDOW_MS, default
        // off): a cold 32-request burst arrives over ~25ms, and dispatching
        // the first few prompts immediately splits the burst into 2-3
        // passes (traced: embed gathers at t=0/26/245 ms) -
        // every prompt's first token waits for the last pass anyway, so
        // collecting while admissions are still in motion (and nothing is
        // decoding, so nobody stalls) merges the passes.
        if !pending.is_empty() {
            let win_ms: u64 = static_env_u64("PADDOCK_ADM_WINDOW_MS", 0);
            if win_ms > 0
                && !slots
                    .iter()
                    .any(|s| s.as_ref().map(|x| x.prefilled).unwrap_or(false))
            {
                use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed as Rl};
                static LAST_N: AtomicUsize = AtomicUsize::new(0);
                static STABLE_MS: AtomicU64 = AtomicU64::new(0);
                let now_ms = service_epoch_ms();
                if LAST_N.swap(pending.len(), Rl) != pending.len() {
                    STABLE_MS.store(now_ms, Rl);
                }
                if now_ms.saturating_sub(STABLE_MS.load(Rl)) < win_ms {
                    // arrivals still in motion: put the prompts back and
                    // retry next tick
                    for (k, prompt) in pending.drain(..) {
                        if let Some(slot) = slots[k].as_mut() {
                            slot.prompt = prompt;
                        }
                    }
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            }
        }
        if !pending.is_empty() {
            // Report the phase honestly: the classic batched prefill is a
            // blocking whole-prompt pass (the c32 TTFT stall), so it must show
            // as PREFILL - the top-of-loop store latched DECODE/IDLE, which is
            // why prefill time used to masquerade as decode in telemetry.
            metrics.phase.store(PHASE_PREFILL, Relaxed);
            let t0 = std::time::Instant::now();
            match generator.forward_prefill_batch(&pending) {
                Ok(logits_list) => {
                    for ((k, prompt), logits) in pending.into_iter().zip(logits_list) {
                        let rows = prompt.len() as u32;
                        finish_prefill(generator, &mut slots, k, logits, rows, metrics);
                    }
                }
                Err(e) => {
                    for (k, _) in pending {
                        if let Some(s) = slots[k].take() {
                            let _ = s.events.send(TokenEvent::Error(EngineError::from_gen(&e)));
                        }
                    }
                }
            }
            st[3].0 += 1;
            st[3].1 += t0.elapsed().as_nanos() as u64;
        }
        if !pending_mm.is_empty() {
            metrics.phase.store(PHASE_PREFILL, Relaxed);
            // one batched call for the whole mm admission wave: backends with
            // a batched vision encode run one tower pass across every pending
            // request's images (the serial per-request encode was the vi8
            // TTFT staircase); the default impl preserves the serial loop
            for (k, res) in generator.forward_prefill_multimodal_batch(pending_mm) {
                match res {
                    Ok((logits, rows)) => {
                        finish_prefill(generator, &mut slots, k, logits, rows as u32, metrics);
                    }
                    Err(e) => {
                        if let Some(s) = slots[k].take() {
                            let _ = s.events.send(TokenEvent::Error(EngineError::from_gen(&e)));
                        }
                    }
                }
            }
        }

        ph_prefill = tick_t0.elapsed();
        // Phase 2m - MIXED tick while a chunked prefill is in flight: every
        // live decode row plus the next prompt chunk in one weight-amortized
        // pass. Decode rows commit exactly like phase 2/3; when the chunk
        // finishes its prompt, the slot samples its first token and joins
        // the decoders next tick. Spec rounds resume once nothing is
        // chunking (mixed ticks already amortize the pass over the chunk).
        // Decode-priority interleave (PADDOCK_DECODE_TICKS_PER_CHUNK=N,
        // default 0 = off): with live decode streams, run N pure-decode
        // ticks between mixed ticks so admission waves don't pin decode
        // cadence to the ~400 ms mixed-tick wall (decode cadence sags
        // through admissions unless something decodes between chunk steps).
        // TTFT pays ~N*13 ms per chunk. Chunking slots simply wait the
        // interleaved ticks; nothing else changes.
        // A client that hung up mid-prefill used to burn the GPU for the whole
        // (possibly minutes-long) prompt: disconnection was only noticed at
        // the first token SEND, which a prefill never reaches. Sweep the
        // chunking slots each tick and retire the dead ones as soon as the
        // backend can drop them (a fused span in flight defers by one tick).
        if !chunking.is_empty() {
            let dead: Vec<usize> = chunking
                .iter()
                .copied()
                .filter(|&k| slots[k].as_ref().is_some_and(|s| s.events.is_closed()))
                .collect();
            for k in dead {
                if generator.prefill_abort(k) {
                    chunking.remove(&k);
                    slots[k] = None;
                    tracing::info!("serve: client gone mid-prefill - aborted slot {k}");
                }
            }
        }
        let run_mixed = if chunking.is_empty() {
            false
        } else {
            static DPC: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
            let dpc = *DPC.get_or_init(|| {
                paddock_models::dev_var!("PADDOCK_DECODE_TICKS_PER_CHUNK")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0)
            });
            let has_decode = slots
                .iter()
                .enumerate()
                .any(|(k, s)| s.as_ref().is_some_and(|s| s.prefilled) && !chunking.contains(&k));
            if dpc > 0 && has_decode {
                dp_count += 1;
                if dp_count > dpc {
                    dp_count = 0;
                    true
                } else {
                    false
                }
            } else {
                true
            }
        };
        if run_mixed {
            // Report the phase honestly (same rule as the classic batched
            // prefill above): a mixed tick exists because a chunked prefill is
            // in flight, and that prefill dominates its wall time. Without
            // this, a minutes-long 200K prefill reported phase=decode with
            // tok_s=0 and frozen counters - indistinguishable from a wedged
            // engine to any operator or watchdog.
            metrics.phase.store(PHASE_PREFILL, Relaxed);
            let mut dec: Vec<(usize, u32, u32)> = Vec::new();
            for (k, s) in slots.iter().enumerate() {
                if let Some(sl) = s
                    && sl.prefilled
                {
                    dec.push((k, sl.pending, sl.pos));
                }
            }
            // OVERLAPPED span+decode tick (decode lane forked under
            // PADDOCK_OVERLAP): launch the prefill
            // span without decode rows, then pump pipelined decode ticks
            // over the prefilled slots on the decode lane while the span
            // runs on the main lane. Riders get a token every decode tick
            // (~30-60 ms under span contention) instead of one per span
            // (~250 ms at the 2048-row default, which is where the
            // mid-concurrency latency loss came from). Slots are disjoint by
            // construction (chunking slots are never decode rows), and span
            // finishers sample in their own d_fin_* buffers, so the two
            // lanes share no staging. Any row the pipe can't take (host
            // sampling, constraint, logprobs) falls back to the fused tick
            // below, unchanged.
            // Full-device-trunc backends can admit truncation rows to the
            // overlap/pipe paths (mode 5 executes end-to-end on device).
            let trunc_pipe_ok = generator.supports_device_trunc() && trunc_pipe_env();
            // Spec-first arbitration: a narrow-decode mixed tick
            // (dec <= dev_spec_live_max) diverts to the spec-in-mixed block
            // below, whose qwen35 implementation EXISTS
            // (forward_mixed_spec_plans_mtp: the graphed spec round + the
            // prefill span in one tick). An earlier attempt at this lost and
            // was reverted - but qwen35 had no forward_mixed_spec override
            // then, so the diverted ticks hit
            // the trait default's silent decline and ran the PLAIN mixed
            // tick: no drafting AND no fusion. That A/B measured the
            // fallback, not the shape. Conditions mirror the spec-in-mixed
            // gate exactly so a diverted tick actually engages there; cold
            // slots warm via the verify's h re-point in one tick.
            // Kill: PADDOCK_NO_MIXED_SPEC restores fusion-first.
            let spec_mixed_first = samp_supported
                && !generator.spec_deferred()
                && paddock_models::dev_var_os!("PADDOCK_NO_MIXED_SPEC").is_none()
                && generator.spec_live_cap() != usize::MAX // spec actually on
                && !dec.is_empty()
                && dec.len() <= dev_spec_live_max
                // SERVE-width scope: on wider serves the live count dips
                // under the cap during admission waves, and the diverted
                // tick pairs a small round with a FAT span serially - which
                // loses at 16 slots even though the same divert wins at 8.
                // Divert only when the whole serve width fits the round, so wide serves
                // keep the fusion until the overlapped round lands.
                && slots.len() <= dev_spec_live_max
                && dec.iter().all(|&(k, _, _)| {
                    let s = slots[k].as_ref().expect("live decode row");
                    (s.sampler.is_device_plannable()
                        || (generator.supports_device_trunc()
                            && s.sampler.is_trunc_plannable()))
                        && s.constraint.is_none()
                        && s.logprobs.is_none()
                });
            // requests in flight (chunking + decoding) - the pipe_min_live floor
            let live_slots = slots.iter().filter(|s| s.is_some()).count();
            let overlap_ok = unified_ok
                && !spec_mixed_first
                && generator.supports_overlap()
                && !dec.is_empty()
                && live_slots >= pipe_min_live()
                && dec.iter().all(|&(k, _, _)| {
                    let s = slots[k].as_ref().expect("live decode row");
                    s.constraint.is_none()
                        && s.logprobs.is_none()
                        && (s.sampler.is_device_plannable()
                            || (trunc_pipe_ok && s.sampler.is_trunc_plannable()))
                })
                && generator
                    .pool_free_blocks()
                    .is_none_or(|f| f > max_batch + POOL_WATERMARK_BLOCKS);
            if overlap_ok {
                // finisher plans: same peek rules as the fused tick below
                // (finishers sample in the span's own buffers - TruncCat is
                // legal there whenever the backend has the trunc finish,
                // independent of the trunc-pipe gate)
                let fin_trunc_ok =
                    generator.supports_host_head() || generator.supports_device_trunc();
                let fin_plans: Vec<(usize, RowSample)> = chunking
                    .iter()
                    .filter_map(|&k| {
                        let sl = slots[k].as_ref()?;
                        let plan =
                            if !sl.recompute && sl.constraint.is_none() && sl.logprobs.is_none() {
                                match if fin_trunc_ok {
                                    sl.sampler.peek_device_plan_trunc()
                                } else {
                                    sl.sampler.peek_device_plan()
                                } {
                                    Some(p) => RowSample::Device(p),
                                    None => RowSample::Host,
                                }
                            } else {
                                RowSample::Host
                            };
                        Some((k, plan))
                    })
                    .collect();
                let t0 = std::time::Instant::now();
                let launched = match generator
                    .unified_span_launch(mixed_tick_budget(dec.len()), &fin_plans)
                {
                    Ok(l) => l,
                    Err(crate::generator::GenError::PoolExhausted) => {
                        preempt_or_fail_mixed(
                            &mut slots,
                            &mut preempted,
                            &mut chunking,
                            pool_stats,
                        );
                        continue;
                    }
                    Err(e) => {
                        tracing::warn!("serve: unified_span_launch failed: {e}");
                        for slot in slots.iter_mut() {
                            if let Some(s) = slot.take() {
                                let _ = s.events.send(TokenEvent::Error(EngineError::from_gen(&e)));
                            }
                        }
                        chunking.clear();
                        continue;
                    }
                };
                if launched {
                    // pump the decode lane while the span runs on the main
                    // lane. Same protocol as the quiet-phase pipe: plans for
                    // tick N+1 are drawn before tick N's ids come back
                    // (`prev_plans`), dead rows keep ticking as dummies, and
                    // the drain collects the last in-flight tick.
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
                                None => RowSample::Hole, // plannable-gated above
                            }
                        })
                        .collect();
                    let mut pipe_alive =
                        match generator.decode_pipe_begin_slots(&slots_v, &toks, &pos, &plans0) {
                            Ok(()) => true,
                            Err(e) => {
                                static ONCE: std::sync::Once = std::sync::Once::new();
                                ONCE.call_once(|| {
                                tracing::warn!(
                                    "serve: overlap decode pipe unavailable, span-only ticks: {e}"
                                );
                            });
                                false
                            }
                        };
                    if pipe_alive {
                        let mut prev_plans = plans0;
                        loop {
                            // stop pumping when the span completes, the pool
                            // runs low, a row needs host sampling, or every
                            // row died - the drain collects the in-flight tick
                            let all_dead = dec.iter().all(|&(k, _, _)| slots[k].is_none());
                            let low_headroom = !generator
                                .pool_free_blocks()
                                .is_none_or(|f| f > max_batch + POOL_WATERMARK_BLOCKS);
                            if generator.unified_span_done() || low_headroom || all_dead {
                                break;
                            }
                            let mut host_row = false;
                            let next_plans: Vec<RowSample> = dec
                                .iter()
                                .map(|&(k, _, _)| match &mut slots[k] {
                                    // died mid-pump: dummy row keeps ticking,
                                    // ids discarded (quiet-pipe trick)
                                    None => RowSample::Hole,
                                    Some(sl)
                                        if sl.constraint.is_none() && sl.logprobs.is_none() =>
                                    {
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
                            match generator.decode_pipe_next(&next_plans) {
                                Ok(ids) => {
                                    st_ovl += 1;
                                    for (i, &(k, _, _)) in dec.iter().enumerate() {
                                        if matches!(prev_plans[i], RowSample::Device(_)) {
                                            commit_device_token(&mut slots[k], ids[i]);
                                        }
                                    }
                                    prev_plans = next_plans;
                                }
                                Err(e) => {
                                    // the decode rows' state is compromised;
                                    // the span (chunking slots) is not - fail
                                    // only the riders, keep the prefills
                                    tracing::warn!("serve: overlap decode_pipe_next failed: {e}");
                                    for &(k, _, _) in &dec {
                                        if let Some(s) = slots[k].take() {
                                            let _ = s
                                                .events
                                                .send(TokenEvent::Error(EngineError::from_gen(&e)));
                                        }
                                    }
                                    pipe_alive = false;
                                    break;
                                }
                            }
                        }
                        if pipe_alive {
                            match generator.decode_pipe_drain() {
                                Ok(ids) => {
                                    for (i, &(k, _, _)) in dec.iter().enumerate() {
                                        if matches!(prev_plans[i], RowSample::Device(_)) {
                                            commit_device_token(&mut slots[k], ids[i]);
                                        }
                                    }
                                }
                                Err(e) => {
                                    tracing::warn!("serve: overlap decode_pipe_drain failed: {e}");
                                    for &(k, _, _) in &dec {
                                        if let Some(s) = slots[k].take() {
                                            let _ = s
                                                .events
                                                .send(TokenEvent::Error(EngineError::from_gen(&e)));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // complete the span (blocks until the main lane finishes)
                    match generator.unified_span_finish() {
                        Ok(finished) => {
                            for (k, fin, rows) in finished {
                                match fin {
                                    FinishSample::Sampled(id) => {
                                        let plan = fin_plans.iter().find_map(|&(s, p)| match p {
                                            RowSample::Device(dp) if s == k => Some(dp),
                                            _ => None,
                                        });
                                        let plan =
                                            plan.expect("sampled finisher had a device plan");
                                        finish_prefill_sampled(
                                            generator,
                                            &mut slots,
                                            k,
                                            id,
                                            &plan,
                                            rows as u32,
                                            metrics,
                                        );
                                    }
                                    FinishSample::Logits(logits) => {
                                        finish_prefill(
                                            generator,
                                            &mut slots,
                                            k,
                                            logits,
                                            rows as u32,
                                            metrics,
                                        );
                                    }
                                }
                                chunking.remove(&k);
                            }
                        }
                        Err(crate::generator::GenError::PoolExhausted) => {
                            preempt_or_fail_mixed(
                                &mut slots,
                                &mut preempted,
                                &mut chunking,
                                pool_stats,
                            );
                        }
                        Err(e) => {
                            tracing::warn!("serve: unified_span_finish failed: {e}");
                            for slot in slots.iter_mut() {
                                if let Some(s) = slot.take() {
                                    let _ =
                                        s.events.send(TokenEvent::Error(EngineError::from_gen(&e)));
                                }
                            }
                            chunking.clear();
                        }
                    }
                }
                st[1].0 += 1;
                st[1].1 += t0.elapsed().as_nanos() as u64;
                st_mchk += chunking.len() as u64;
                st_mdec += dec.len() as u64;
                continue; // tick done - admissions and the next chunk re-check
            }
            // Device-sampled mixed tick: under load most ticks are mixed
            // (arrivals keep a chunk in flight), so without this every tick
            // paid the [nd, vocab] readback + host sampling that the dense
            // path already killed - measured as ~7 ms of GPU idle per c32
            // tick on a GB202, the big gaps sitting between the lm-head
            // GEMM and the next embed gather. Same plan rules
            // as phase 3; plans index DECODE rows (dec order), not slots.
            // Spec-in-mixed: decode rows ride VERIFY chunks while
            // the prompt chunk streams. The mixed lane previously ran decode
            // UNSPECULATED whenever anything chunked - and under load most
            // ticks are mixed, so a whole c32 sweep could hold as few as
            // NINE verify rounds, all in the drain. Conditions mirror the
            // pure device-spec round
            // (all decode rows device-plannable, drafter warm, model
            // drafts); any decline falls through to the plain mixed tick
            // below. Kill: PADDOCK_NO_MIXED_SPEC.
            let no_mixed_spec = paddock_models::dev_var_os!("PADDOCK_NO_MIXED_SPEC").is_some();
            if samp_supported
                && !no_mixed_spec
                && !generator.spec_deferred()
                && !dec.is_empty()
                && dec.len() <= dev_spec_live_max
                && dec.iter().all(|&(k, _, _)| {
                    let s = slots[k].as_ref().expect("live decode row");
                    // Truncation slots join the mixed-spec round on
                    // full-device-trunc backends (mode-5 verify rows; the
                    // sampled verify + accept-while-match stays exact).
                    // Always-on like the pure spec round - mixed rounds are
                    // synchronous, no pipe begin/drain churn exposure.
                    (s.sampler.is_device_plannable()
                        || (generator.supports_device_trunc() && s.sampler.is_trunc_plannable()))
                        && s.constraint.is_none()
                        && s.logprobs.is_none()
                })
            {
                // PER-SLOT warmth (all-or-nothing never engaged under
                // continuous admissions - every tick had a fresh joiner):
                // warm slots get draft chunks, cold slots ride the same
                // verify front as length-1 chunks, and the verify's h
                // re-point warms them for the next tick. With zero warm
                // slots this degenerates to the plain unified tick plus the
                // h bootstrap - the engagement ramp is one tick per slot.
                let t0m = std::time::Instant::now(); // mspec bucket: warmth + chain + pass
                let t_warm = std::time::Instant::now();
                let dec_ks: Vec<usize> = dec.iter().map(|&(k, _, _)| k).collect();
                let warm: Vec<bool> = spec_warm_vec(generator, &slots, &dec_ks);
                let k_budget = spec_ctl.pick_k(
                    dec.len(),
                    backend_spec_k_budget(generator, dec.len()).min(serve_spec_mixed_k_cap()),
                );
                if paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some() {
                    tracing::info!(
                        "[spec-kb] MIXED dec={} k_budget={k_budget} warm={warm:?}",
                        dec.len()
                    );
                }
                tick_open = Some((std::time::Instant::now(), dec.len(), k_budget, true));
                if paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some() {
                    let wms = t_warm.elapsed().as_secs_f64() * 1e3;
                    if wms > 1.0 {
                        tracing::info!("[mix-warm] {wms:.2}ms ({} rows)", dec.len());
                    }
                    tracing::info!(
                        "[mspec-gate] dec={} warm_n={} k_budget={}",
                        dec.len(),
                        warm.iter().filter(|&&w| w).count(),
                        k_budget
                    );
                }
                // [mix] phase timers: a 128x128 capture shows ~7.3ms of PURE
                // host time per mixed boundary (790ms of 862ms gap total
                // carries no CUDA API at all) - these timers attribute it
                // draft vs prep vs verify-call vs post.
                let t_mx0 = std::time::Instant::now();
                // [mix-bnd]: the whole service-side boundary from
                // the previous mixed round's finish to this round's draft -
                // accept/SSE + loop-top poll/admit + dec/warm build. Even
                // after the drafter fold there is ~6.8ms of GPU idle per
                // boundary; this timer splits ownership.
                if paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some()
                    && let Some(tb) = mix_bnd.take()
                {
                    let ms = tb.elapsed().as_secs_f64() * 1e3;
                    if ms > 2.0 {
                        tracing::info!("[mix-bnd] {ms:.2}ms finish->draft");
                    }
                }
                let pendings: Vec<(usize, u32)> = dec
                    .iter()
                    .zip(&warm)
                    .filter(|&(_, &w)| w)
                    .map(|(&(k, p, _), _)| (k, p))
                    .collect();
                let want_drafts = k_budget > 0 && !pendings.is_empty();
                // Drafter fold: arm the ASYNC chain (the same begin/fetch
                // machinery the pure tick already runs) - no
                // draft readback here; the backend assembles the verify
                // tokens on device, and the 3.2-3.7ms/round draft stall
                // ([mix] timer) becomes queued stream
                // work inside the round's own GPU window. Chunks below get
                // placeholder VALUES; the real drafts arrive via
                // spec_draft_fetch after the round for the accept match.
                // Kill: PADDOCK_NO_ASYNC_SPEC (same switch as the pure tick).
                let async_mk: Option<(usize, Vec<bool>)> = if want_drafts && async_spec_on() {
                    generator
                        .spec_draft_begin(&pendings, k_budget)
                        .unwrap_or(None)
                } else {
                    None
                };
                let model_drafts = if want_drafts && async_mk.is_none() {
                    let draft_k = synchronous_draft_budget(
                        k_budget,
                        generator.spec_block_width(),
                        pendings
                            .iter()
                            .map(|&(k, _)| slots[k].as_ref().expect("warm").k_now),
                    );
                    match generator.spec_draft_batch(&pendings, draft_k) {
                        Ok(d) => d,
                        Err(e) => {
                            if paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some() {
                                tracing::info!("[mixed-spec] draft ERR: {e}");
                            }
                            None
                        }
                    }
                } else {
                    None
                };
                let t_mx_draft = t_mx0.elapsed();
                {
                    let drafts = model_drafts; // Option<Vec<Vec<u32>>> in warm order
                    let mut reqs: Vec<(usize, usize, Vec<u32>)> = Vec::with_capacity(dec.len());
                    let mut wi = 0usize;
                    for (i, &(k, pending, pos)) in dec.iter().enumerate() {
                        let slot = slots[k].as_ref().expect("live decode row");
                        let mut chunk = vec![pending];
                        if warm[i] {
                            if let Some((ke, kept)) = async_mk.as_ref() {
                                // placeholder VALUES (the device assembly
                                // holds the real drafts); LENGTH is the
                                // contract - chain-cold entries stay
                                // length-1 exactly like the sync path
                                if kept.get(wi).copied().unwrap_or(false) {
                                    let cap = slot_spec_k(slot.k_now, k_budget, fixed_draft_depth)
                                        .min(*ke);
                                    chunk.resize(1 + cap, pending);
                                }
                            } else if let Some(d) = drafts.as_ref() {
                                let cap = slot_spec_k(slot.k_now, k_budget, fixed_draft_depth);
                                chunk.extend(d[wi].iter().copied().take(cap));
                            }
                            wi += 1;
                        }
                        reqs.push((k, pos as usize, chunk));
                    }
                    // trunc slots draw TruncCat verify plans (the
                    // gate above admits them only on supports_device_trunc)
                    let spec_trunc_ok = generator.supports_device_trunc();
                    let mut splans: Vec<crate::sampler::DevicePlan> =
                        Vec::with_capacity(reqs.iter().map(|r| r.2.len()).sum());
                    for (ri, &(k, _, _)) in dec.iter().enumerate() {
                        let slot = slots[k].as_mut().expect("live decode row");
                        for _ in 0..reqs[ri].2.len() {
                            splans.push(
                                if spec_trunc_ok {
                                    slot.sampler.device_plan_trunc()
                                } else {
                                    slot.sampler.device_plan()
                                }
                                .unwrap_or(crate::sampler::DevicePlan::Greedy),
                            );
                        }
                    }
                    let chk_now = chunking.len() as u64; // before finished-removal
                    let t_mx_prep = t_mx0.elapsed() - t_mx_draft;
                    let t_mxc = std::time::Instant::now();
                    if t_mx0.elapsed().as_secs_f64() > 3e-3
                        && paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some()
                    {
                        tracing::info!(
                            "[mix] draft {:.2}ms prep {:.2}ms (dec {} chunks {})",
                            t_mx_draft.as_secs_f64() * 1e3,
                            t_mx_prep.as_secs_f64() * 1e3,
                            dec.len(),
                            chk_now
                        );
                    }
                    let _ = t_mxc;
                    // Issue-ahead: enqueue the round, run the
                    // previous round's deferred finish work inside its GPU
                    // window, then block for picks. Errors funnel into the
                    // same arms as the blocking call.
                    // Finisher plans: device-sample the prefill
                    // tail on eligible slots (no constraint/logprobs, seed
                    // plannable) - the backend then skips that finisher's
                    // [1, vocab] logits readback + host pick entirely.
                    // Kill: PADDOCK_NO_FIN_DEV (all-host, yesterday's path).
                    let fin_dev_ok = paddock_models::dev_var_os!("PADDOCK_NO_FIN_DEV").is_none();
                    let fin_trunc_ok =
                        generator.supports_host_head() || generator.supports_device_trunc();
                    let spec_fin_plans: Vec<(usize, RowSample)> = chunking
                        .iter()
                        .filter_map(|&k| {
                            let sl = slots[k].as_ref()?;
                            let plan = if fin_dev_ok
                                && !sl.recompute
                                && sl.constraint.is_none()
                                && sl.logprobs.is_none()
                            {
                                match if fin_trunc_ok {
                                    sl.sampler.peek_device_plan_trunc()
                                } else {
                                    sl.sampler.peek_device_plan()
                                } {
                                    Some(p) => RowSample::Device(p),
                                    None => RowSample::Host,
                                }
                            } else {
                                RowSample::Host
                            };
                            Some((k, plan))
                        })
                        .collect();
                    // the round waits on the span beside it: pace the span
                    let verify_rows: usize = reqs.iter().map(|r| r.2.len()).sum();
                    let (spec_budget, spec_tick, launched) = match pacers.begin(
                        crate::pacing::TickKind::Spec,
                        generator,
                        reqs.len(),
                        verify_rows,
                        prefill_tick_rows(),
                        verify_rows,
                    ) {
                        Err(e) => (prefill_tick_rows(), None, Err(e)),
                        Ok((b, t)) => (
                            b,
                            t,
                            generator.forward_mixed_spec_begin(&reqs, b, &splans, &spec_fin_plans),
                        ),
                    };
                    if let Ok(true) = launched {
                        let t_df = std::time::Instant::now();
                        let ndf = mix_deferred.len();
                        for (k, fs, rows, plan) in mix_deferred.drain(..) {
                            finish_prefill_any(
                                generator,
                                &mut slots,
                                k,
                                fs,
                                plan,
                                rows as u32,
                                metrics,
                            );
                            chunking.remove(&k);
                        }
                        if ndf > 0 && paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some() {
                            let ms = t_df.elapsed().as_secs_f64() * 1e3;
                            if ms > 1.0 {
                                tracing::info!("[mix-defer] {ndf} fins {ms:.2}ms overlapped");
                            }
                        }
                    }
                    let was_launched = matches!(launched, Ok(true));
                    let mixed_res = match launched {
                        Ok(true) => generator.forward_mixed_spec_finish(),
                        Ok(false) => generator.forward_mixed_spec_plans(
                            &reqs,
                            spec_budget,
                            &splans,
                            &spec_fin_plans,
                        ),
                        Err(e) => Err(e),
                    };
                    if mixed_res.is_ok() {
                        pacers.end(&*generator, spec_tick);
                    }
                    match mixed_res {
                        Ok((Some(picks), finished)) => {
                            // async round: fill the placeholder chunks with
                            // the real drafts (fetched post-verify - the
                            // stream is long past the chain; the backend's
                            // wait already did its own fill for the h-map
                            // replay) so the accept match below compares
                            // what verify actually saw. Cold-of-chain slots
                            // keep their pending pads.
                            if async_mk.is_some()
                                && let Ok(Some(dr)) = generator.spec_draft_fetch()
                            {
                                let mut wi2 = 0usize;
                                for (ri, _) in dec.iter().enumerate() {
                                    if warm[ri] {
                                        if let Some(d) = dr.get(wi2) {
                                            let c = &mut reqs[ri].2;
                                            for j in 1..c.len() {
                                                if j - 1 < d.len() {
                                                    c[j] = d[j - 1];
                                                }
                                            }
                                        }
                                        wi2 += 1;
                                    }
                                }
                            }
                            let t_acc = std::time::Instant::now();
                            let mut dead: Vec<usize> = Vec::new();
                            let mut base = 0usize;
                            for (ri, &(k, _, _)) in dec.iter().enumerate() {
                                let chunk = &reqs[ri].2;
                                let slot = slots[k].as_mut().expect("live decode row");
                                let mut a = 0usize;
                                while a + 1 < chunk.len() && chunk[a + 1] == picks[base + a] {
                                    a += 1;
                                }
                                if chunk.len() > 1 {
                                    if a == chunk.len() - 1 {
                                        slot.k_now = (slot.k_now * 2).min(serve_spec_max_k());
                                    } else {
                                        slot.k_now = (a + 1).clamp(
                                            generator
                                                .spec_k_miss_floor()
                                                .unwrap_or_else(serve_spec_k_floor)
                                                .min(slot.k_now),
                                            slot.k_now,
                                        );
                                    }
                                }
                                slot.spec_round(&mut tick_tally, chunk.len() - 1, a);
                                slot.pos += (a + 1) as u32;
                                slot.pending = picks[base + a];
                                let mut slot_dead = false;
                                for &t in &picks[base..=base + a] {
                                    slot.draft.push(t);
                                    if !slot.accept(t, None) {
                                        slot_dead = true;
                                        break;
                                    }
                                }
                                if slot_dead {
                                    dead.push(k);
                                }
                                base += chunk.len();
                            }
                            for k in dead {
                                slots[k] = None;
                            }
                            // Defer the finish work one round -
                            // it runs inside the next round's GPU window.
                            // Only on the issue-ahead path: a backend
                            // without begin() (qwen35) never reaches the
                            // overlapped flush, so it keeps the inline form.
                            let with_plan = |k: usize| {
                                spec_fin_plans.iter().find_map(|&(s, p)| {
                                    (s == k)
                                        .then_some(match p {
                                            RowSample::Device(d) => Some(d),
                                            _ => None,
                                        })
                                        .flatten()
                                })
                            };
                            if was_launched {
                                mix_deferred.extend(
                                    finished
                                        .into_iter()
                                        .map(|(k, fs, r)| (k, fs, r, with_plan(k))),
                                );
                            } else {
                                for (k, fs, rows) in finished {
                                    let plan = with_plan(k);
                                    finish_prefill_any(
                                        generator,
                                        &mut slots,
                                        k,
                                        fs,
                                        plan,
                                        rows as u32,
                                        metrics,
                                    );
                                    chunking.remove(&k);
                                }
                            }
                            if paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some() {
                                let ms = t_acc.elapsed().as_secs_f64() * 1e3;
                                if ms > 1.5 {
                                    tracing::info!(
                                        "[mix-acc] {ms:.2}ms accept+sse ({} rows)",
                                        picks.len()
                                    );
                                }
                            }
                            mix_bnd = Some(std::time::Instant::now());
                            st[5].0 += 1;
                            st[5].1 += t0m.elapsed().as_nanos() as u64;
                            st_mchk += chk_now;
                            st_mdec += dec.len() as u64;
                            continue; // tick done
                        }
                        // decline: chunk untouched - plain mixed tick below.
                        // The drafter prologue was still paid; book it so the
                        // wide-leg diff sees wasted mspec attempts too.
                        Ok((None, _)) => {
                            // clear any armed async plan (a dangling plan
                            // would corrupt the next round's device assembly)
                            if async_mk.is_some() {
                                let _ = generator.spec_draft_fetch();
                            }
                            st[5].1 += t0m.elapsed().as_nanos() as u64;
                        }
                        Err(crate::generator::GenError::PoolExhausted) => {
                            if async_mk.is_some() {
                                let _ = generator.spec_draft_fetch();
                            }
                            // deferred slots look "stuck chunking" to the
                            // preemptor - settle them first
                            for (k, fs, rows, plan) in mix_deferred.drain(..) {
                                finish_prefill_any(
                                    generator,
                                    &mut slots,
                                    k,
                                    fs,
                                    plan,
                                    rows as u32,
                                    metrics,
                                );
                                chunking.remove(&k);
                            }
                            preempt_or_fail_mixed(
                                &mut slots,
                                &mut preempted,
                                &mut chunking,
                                pool_stats,
                            );
                            continue;
                        }
                        Err(e) => {
                            if async_mk.is_some() {
                                let _ = generator.spec_draft_fetch();
                            }
                            // fail-all kills every slot below - deferred
                            // entries would point at dead slots
                            mix_deferred.clear();
                            tracing::warn!(
                                "serve: forward_mixed_spec_plans failed ({} decode rows): {e}",
                                dec.len()
                            );
                            for slot in slots.iter_mut() {
                                if let Some(s2) = slot.take() {
                                    let _ = s2
                                        .events
                                        .send(TokenEvent::Error(EngineError::from_gen(&e)));
                                }
                            }
                            chunking.clear();
                            continue;
                        }
                    }
                }
            }
            if samp_supported {
                // Truncation rows (top-k/top-p) ride the head-sampling
                // finish where the backend implements it - mixed/classic
                // paths only, never the pipes
                let trunc_ok = generator.supports_host_head() || generator.supports_device_trunc();
                let plans: Vec<RowSample> = dec
                    .iter()
                    .map(|&(k, _, _)| {
                        let sl = slots[k].as_mut().expect("live decode row");
                        if sl.constraint.is_none() && sl.logprobs.is_none() {
                            match if trunc_ok {
                                sl.sampler.device_plan_trunc()
                            } else {
                                sl.sampler.device_plan()
                            } {
                                Some(p) => RowSample::Device(p),
                                // greedy rows whose only host-side need is the
                                // no-repeat-ngram guard argmax on device on
                                // ticks where the guard would ban nothing (a
                                // no-op mask leaves raw logits - bit-exact);
                                // ban-live ticks fall back to the exact host
                                // row. Backend-gated via ngram_dev_ok.
                                None if ngram_dev_ok
                                    && sl.sampler.is_greedy_ngram_only()
                                    && !sl.sampler.ngram_would_ban(&sl.history) =>
                                {
                                    RowSample::Device(crate::sampler::DevicePlan::Greedy)
                                }
                                None => RowSample::Host,
                            }
                        } else {
                            RowSample::Host
                        }
                    })
                    .collect();
                // PEEKED plans for chunk-prefilling slots: whichever prompt(s)
                // finish this tick (the generator's budget decides) sample
                // their first token on device - no last-row logits readback,
                // no host softmax (the recorded ~6-7 ms mid-tick stall). The
                // uniform is committed only when `Sampled` confirms the plan
                // ran; recompute slots keep the readback path (their result is
                // discarded, and a device draw would burn a uniform).
                let fin_plans: Vec<(usize, RowSample)> = chunking
                    .iter()
                    .filter_map(|&k| {
                        let sl = slots[k].as_ref()?;
                        let plan =
                            if !sl.recompute && sl.constraint.is_none() && sl.logprobs.is_none() {
                                match if trunc_ok {
                                    sl.sampler.peek_device_plan_trunc()
                                } else {
                                    sl.sampler.peek_device_plan()
                                } {
                                    Some(p) => RowSample::Device(p),
                                    None => RowSample::Host,
                                }
                            } else {
                                RowSample::Host
                            };
                        Some((k, plan))
                    })
                    .collect();
                let t0 = std::time::Instant::now();
                // Per-TICK unified choice on queue depth. The fused tick is
                // what a LIGHT queue wants: at c1 it puts the single prompt
                // on the fast unified prefill route (TTFT 40 vs 432 ms). A
                // DEEP decode batch wants the opposite: the fused pass drags
                // every decode row off its tile-plane route and onto the
                // prefill ladder, so a c32 leg pays ~46.7 ms mixed ticks
                // against ~11.0 ms decode ticks. Process-wide
                // PADDOCK_NO_UNIFIED still forces the split path everywhere.
                // Both conditions matter. dec alone is a TRAP at high
                // concurrency: when a completion wave leaves slots waiting on
                // prefill, dec DIPS -- and that is exactly when a fused fat
                // tick hurts most (gating on dec_max=16 spiralled a c32 leg
                // into uniform 104 ms ticks: slow fused tick -> more
                // completions pile up -> dec stays low -> every tick fused).
                // A deep chunk queue means the tick should split regardless
                // of how few riders remain.
                // (A chunking.len() clause was tried here and MEASURED out:
                // it flaps tick-to-tick at c8 and the resulting mode
                // alternation mid-prompt collapsed c8 into 64 ms ticks. dec
                // is the stable signal.)
                // Threshold needs headroom over the concurrency it protects:
                // at c8, completion/admission overlap makes live flap 8<->9,
                // and with the cutoff at 8 the mode alternated mid-prompt.
                // 12 keeps c8's transients fused and c32 (live ~32) split.
                // The signal must be REGIME, not this tick's dec: dec dips
                // when a completion wave leaves slots waiting on prefill, and
                // gating on it let fat fused ticks in at exactly the wrong
                // moment. Live slot count only moves on admission/completion,
                // so the mode holds steady per load level -- c8 wants fused,
                // c32 wants split.
                let live_now = slots.iter().filter(|s| s.is_some()).count();
                let unified_now = unified_ok
                    && live_now <= {
                        static UDM: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
                        *UDM.get_or_init(|| {
                            paddock_models::dev_var!("PADDOCK_UNIFIED_DEC_MAX")
                                .ok()
                                .and_then(|v| v.parse().ok())
                                .unwrap_or(12)
                        })
                    };
                // tickseg: the sampled mixed pass is the post-devsample hot
                // path - without these timers the probe reads fwd 0x on OCR
                if crate::tickseg::on()
                    && let Some(m) = seg_mark.take()
                {
                    crate::tickseg::gap(m.elapsed());
                }
                let seg_t = std::time::Instant::now();
                let kind = if unified_now {
                    crate::pacing::TickKind::Unified
                } else {
                    crate::pacing::TickKind::Mixed
                };
                let nd = dec.len();
                let mixed_result =
                    match pacers.begin(kind, generator, nd, nd, mixed_tick_budget(nd), nd) {
                        Err(e) => Err(e),
                        Ok((budget, tick)) => {
                            let r = if unified_now {
                                generator.forward_unified_sampled(&dec, budget, &plans, &fin_plans)
                            } else {
                                generator.forward_mixed_sampled(&dec, budget, &plans, &fin_plans)
                            };
                            if r.is_ok() {
                                pacers.end(&*generator, tick);
                            }
                            r
                        }
                    };
                match mixed_result {
                    Ok((step, finished)) => {
                        if crate::tickseg::on() {
                            crate::tickseg::fwd(seg_t.elapsed());
                        }
                        let seg_t = std::time::Instant::now();
                        let mut host_rows = step.host_rows.into_iter();
                        for (i, &(k, _, _)) in dec.iter().enumerate() {
                            match plans[i] {
                                RowSample::Hole => {}
                                RowSample::Device(_) => {
                                    commit_device_token(&mut slots[k], step.ids[i]);
                                }
                                RowSample::Host => {
                                    let (_, mut row) = host_rows.next().expect("host row");
                                    sample_slot_row(&mut slots[k], &mut row);
                                }
                            }
                        }
                        for (k, fin, rows) in finished {
                            match fin {
                                FinishSample::Sampled(id) => {
                                    let plan = fin_plans.iter().find_map(|&(s, p)| match p {
                                        RowSample::Device(dp) if s == k => Some(dp),
                                        _ => None,
                                    });
                                    let plan = plan.expect("sampled finisher had a device plan");
                                    finish_prefill_sampled(
                                        generator,
                                        &mut slots,
                                        k,
                                        id,
                                        &plan,
                                        rows as u32,
                                        metrics,
                                    );
                                }
                                FinishSample::Logits(logits) => {
                                    finish_prefill(
                                        generator,
                                        &mut slots,
                                        k,
                                        logits,
                                        rows as u32,
                                        metrics,
                                    );
                                }
                            }
                            chunking.remove(&k);
                        }
                        if crate::tickseg::on() {
                            crate::tickseg::smp(seg_t.elapsed());
                            seg_mark = Some(std::time::Instant::now());
                        }
                    }
                    Err(crate::generator::GenError::PoolExhausted) => {
                        preempt_or_fail_mixed(
                            &mut slots,
                            &mut preempted,
                            &mut chunking,
                            pool_stats,
                        );
                    }
                    Err(e) => {
                        // same failure semantics as the unsampled mixed pass
                        tracing::warn!(
                            "serve: forward_mixed_sampled failed ({} decode rows): {e}",
                            dec.len()
                        );
                        for slot in slots.iter_mut() {
                            if let Some(s) = slot.take() {
                                let _ = s.events.send(TokenEvent::Error(EngineError::from_gen(&e)));
                            }
                        }
                        // The backend still holds the failed chunked prefills
                        // in its own queue; clearing only our set left every
                        // later prefill_begin on those slots refusing with
                        // "slot already has a chunked prefill in flight" -
                        // one bad prompt poisoned the slot for the server's
                        // lifetime (seen 2026-09-06 on a dense i-quant file
                        // whose >64-row GEMM was refused). Drop them there too.
                        for &k in chunking.iter() {
                            if !generator.prefill_abort(k) {
                                tracing::warn!(
                                    "serve: slot {k}: backend kept its chunked prefill after the failed tick"
                                );
                            }
                        }
                        chunking.clear();
                    }
                }
                st[1].0 += 1;
                st[1].1 += t0.elapsed().as_nanos() as u64;
                if paddock_models::dev_var_os!("PADDOCK_REQ_TRACE").is_some() {
                    tracing::info!(
                        "req-trace: mixed-tick chk={} dec={} ms={:.1} at {}",
                        fin_plans.len(),
                        dec.len(),
                        t0.elapsed().as_secs_f64() * 1e3,
                        trace_us()
                    );
                }
                // rider accounting for plain mixed ticks too (only the mspec
                // branch used to feed these, which left rider counts to be
                // inferred from graph-launch arithmetic)
                st_mchk += chunking.len() as u64;
                st_mdec += dec.len() as u64;
                continue; // tick done - admissions and the next chunk re-check
            }
            if crate::tickseg::on()
                && let Some(m) = seg_mark.take()
            {
                crate::tickseg::gap(m.elapsed());
            }
            let seg_t = std::time::Instant::now();
            let nd = dec.len();
            let mixed_result = match pacers.begin(
                crate::pacing::TickKind::Mixed,
                generator,
                nd,
                nd,
                mixed_tick_budget(nd),
                nd,
            ) {
                Err(e) => Err(e),
                Ok((budget, tick)) => {
                    let r = generator.forward_mixed(&dec, budget);
                    if r.is_ok() {
                        pacers.end(&*generator, tick);
                    }
                    r
                }
            };
            match mixed_result {
                Ok((mut dlogits, finished)) => {
                    if crate::tickseg::on() {
                        crate::tickseg::fwd(seg_t.elapsed());
                    }
                    let seg_t = std::time::Instant::now();
                    // decode rows are compacted (not slot-indexed), so commit
                    // serially - ≤ max_batch-1 rows against a ~60 ms chunk
                    // tick; the scoped-thread fan-out isn't worth the
                    // disjoint-borrow gymnastics here
                    for (i, &(k, _, _)) in dec.iter().enumerate() {
                        let row = &mut dlogits[i * vocab..(i + 1) * vocab];
                        sample_slot_row(&mut slots[k], row);
                    }
                    for (k, logits, rows) in finished {
                        finish_prefill(generator, &mut slots, k, logits, rows as u32, metrics);
                        chunking.remove(&k);
                    }
                    if crate::tickseg::on() {
                        crate::tickseg::smp(seg_t.elapsed());
                        seg_mark = Some(std::time::Instant::now());
                    }
                }
                Err(crate::generator::GenError::PoolExhausted) => {
                    preempt_or_fail_mixed(&mut slots, &mut preempted, &mut chunking, pool_stats);
                }
                Err(e) => {
                    // A mixed-pass failure may have half-advanced KV for the
                    // decode rows - their logits are gone either way, so all
                    // slots must fail. Log it: the events channel is the only
                    // other witness and clients just see empty streams.
                    tracing::warn!(
                        "serve: forward_mixed failed ({} decode rows): {e}",
                        dec.len()
                    );
                    for slot in slots.iter_mut() {
                        if let Some(s) = slot.take() {
                            let _ = s.events.send(TokenEvent::Error(EngineError::from_gen(&e)));
                        }
                    }
                    for &k in chunking.iter() {
                        generator.prefill_abort(k);
                    }
                    chunking.clear();
                }
            }
            // Phase marks for the straggler self-report: this arm returns
            // before the classic marks below are set, so a unified tick used
            // to be booked as "sample+emit" - which is what a 300 ms cold
            // first prefill on a 5060 Ti read as until the req-trace lines
            // named it. The whole tick is mixed work here.
            ph_mixed = tick_t0.elapsed();
            ph_spec = ph_mixed;
            ph_decode = ph_mixed;
            continue; // tick done - admissions and the next chunk re-check
        }

        ph_mixed = tick_t0.elapsed();
        // Phase 2a - speculative round: when every active slot samples pure
        // greedy and the backend verifies drafts, run one ragged multi-slot
        // pass (per slot: pending token + n-gram drafts) instead of the
        // dense one-token step. Each slot commits its accepted run plus the
        // verify pass's own next token - several tokens per weight read.
        // Any sampling slot forces the dense path (it needs full logits).
        spec_ticks += 1;
        if spec_supported && spec_ticks >= spec_retry_at && !generator.spec_deferred() {
            // chunking slots are mid-prefill - Never decodable rows. The
            // original gate made this vacuous (pure-decode only ran with
            // chunking empty); the decode-priority interleave runs this path
            // with chunking slots waiting, so the filter is load-bearing.
            // The encoder has taken an image slot's chunks but has not
            // published it into `chunking`: absence from that set alone is
            // not evidence of a valid decode cursor.
            let live = spec_live_slots(&slots, &chunking);
            // constrained slots force the dense path too: spec picks are
            // device argmaxes, computed where no mask can apply
            let greedy = !live.is_empty()
                && live.len() <= serve_spec_max_rows()
                && live.len() <= spec_live_cap
                && live.iter().all(|&k| {
                    let s = slots[k].as_ref().expect("live");
                    s.sampler.is_pure_greedy()
                        && s.constraint.is_none()
                        && s.logprobs.is_none()
                        // Token-replay drafts cannot bridge image rows.
                        // Native KV-space drafters can, without slicing a
                        // token history at a multimodal KV cursor.
                        && (s.pos as usize <= s.history.len() || generator.spec_draft_kv_space())
                });
            // Is this cohort eligible for the DEVICE-SAMPLED round (phase
            // 2b-dev below)? Hoisted out of its `else if` so the controller can
            // be consulted once per tick, before either branch commits to a
            // round - see the k=0 arm.
            // The device round drafts from a model drafter only (it has no
            // n-gram fallback), so without one every chunk is the pending
            // token alone: a one-row verify - the decode tick's work through
            // the verify path's state save, uncaptured walk and commit. On
            // Flash Next that is 23 tok/s where the captured tick is 25.5.
            let dev_ok = !greedy
                && generator.spec_capable()
                && !live.is_empty()
                && live.len() <= dev_spec_live_max
                && live.len() <= {
                    // gemma4 caps draft engagement (PADDOCK_G4_SPEC_LIVE_MAX,
                    // set at attach on cc10). Beyond it every chunk would be
                    // length-1 - a plain tick that BLOCKS the pipelined phase
                    // 2 below - so fall through instead.
                    static CAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
                    *CAP.get_or_init(|| {
                        paddock_models::dev_var!("PADDOCK_G4_SPEC_LIVE_MAX")
                            .ok()
                            .and_then(|v| v.parse().ok())
                            .unwrap_or(usize::MAX)
                    })
                }
                && live.iter().all(|&k| {
                    let s = slots[k].as_ref().expect("live");
                    (s.sampler.is_device_plannable()
                        || (generator.supports_device_trunc() && s.sampler.is_trunc_plannable()))
                        && s.constraint.is_none()
                        && s.logprobs.is_none()
                });
            // One decision per tick, and only for a cohort that could actually
            // ride a round (an ineligible one must not consume a warm-start
            // anchor it can never measure).
            let k_budget = if greedy || dev_ok {
                spec_ctl.pick_k(live.len(), backend_spec_k_budget(generator, live.len()))
            } else {
                0
            };
            if (greedy || dev_ok) && k_budget == 0 {
                // K=0 means don't speculate, and that has to be the dense tick
                // rather than a round with no drafts. A chunk-per-slot of
                // length 1 still pays the backend's whole round to commit
                // exactly what a decode tick commits - and on this family a
                // round copies the GDN state and the PLE ring, then walks a
                // prefill-shaped wave. Measured at 8 live slots (2026-09-17):
                // those rows=8/committed=8 rounds booked ~44 ms against the
                // dense tick's ~26, so a bucket that converged on k=0 ran the
                // whole serve 40% under its own no-spec throughput, and the
                // k=0 cell every other K is judged against was itself measured
                // on the round instead of the tick it names. Booking the tick
                // here is also what finally makes that cell honest.
                tick_open = Some((std::time::Instant::now(), live.len(), 0, false));
            } else if greedy {
                if paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some() {
                    tracing::info!(
                        "[spec-kb] decode live={} k_budget={k_budget} ladder={} max_rows={}",
                        live.len(),
                        // the row budget's own answer, uncapped - printing the
                        // capped value here would hide the election doing its job
                        serve_spec_k_budget(live.len(), generator.spec_block_width()),
                        serve_spec_max_rows()
                    );
                }
                tick_open = Some((std::time::Instant::now(), live.len(), k_budget, false));
                // MTP catch-up - the mixed tick's per-slot warmth, now on
                // the pure-decode round too. Dense interlude ticks
                // advance a slot without advancing its draft KV; the desync
                // probe in forward_spec_batch then POISONS warmth
                // (mtp_warm=false) and cools down 256 ticks, so after one
                // warm-capped (long) prompt, single-user spec re-engagement
                // was a ~1/256 lottery - in practice every later request ran
                // dense forever (on an A6000 27b that is a permanent ~2x
                // decode loss).
                // ensure_warm extends the draft KV over the gap (bounded by
                // SPEC_WARM_MAX); when every live slot reports warm the probe
                // below is guaranteed position-synced. Cold slots (no MTP,
                // past warm_max) keep the legacy probe + cooldown behavior.
                let t0s = std::time::Instant::now(); // spec bucket: warmth + chain + verify
                let all_warm = spec_warm_vec(generator, &slots, &live).iter().all(|&w| w);
                // model-side drafts (MTP head) when the backend has them -
                // far better acceptance than n-gram on prose; n-gram stays
                // the fallback (and the drafter for declined ticks)
                let pendings: Vec<(usize, u32)> = live
                    .iter()
                    .map(|&k| (k, slots[k].as_ref().expect("live").pending))
                    .collect();
                // `all_warm` is a CONJUNCTION over live slots, so P(all warm)
                // decays with slot count - at 8 slots the model drafter was
                // reached once in 145 ticks, every one of which had a healthy
                // k budget, because a single cold slot sent all eight to the
                // n-gram fallback. A drafter that reports warmth per slot
                // (DFlash: it filters internally and returns an empty list for
                // the cold ones) does not need the conjunction. Token-replay
                // chains still do - a cold slot there desyncs.
                let per_slot_warm = generator.spec_draft_per_slot_warm();
                let model_drafts = if k_budget > 0 && (all_warm || per_slot_warm) {
                    let draft_k = synchronous_draft_budget(
                        k_budget,
                        generator.spec_block_width(),
                        pendings
                            .iter()
                            .map(|&(k, _)| slots[k].as_ref().expect("live").k_now),
                    );
                    match generator.spec_draft_batch(&pendings, draft_k) {
                        Ok(d) => d,
                        Err(e) => {
                            if paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some() {
                                tracing::info!("[spec2a-host] draft ERR: {e}");
                            }
                            None
                        }
                    }
                } else {
                    None
                };
                let mut reqs: Vec<(usize, usize, Vec<u32>)> = Vec::with_capacity(live.len());
                for (i, &k) in live.iter().enumerate() {
                    let slot = slots[k].as_ref().expect("live");
                    let mut chunk = vec![slot.pending];
                    if k_budget > 0 {
                        match model_drafts.as_ref() {
                            Some(d) if !d[i].is_empty() => {
                                let cap = slot_spec_k(slot.k_now, k_budget, fixed_draft_depth);
                                chunk.extend(d[i].iter().copied().take(cap));
                            }
                            // The model drafter declined this slot (cold ring,
                            // ctx-full, past the per-round block cap). Keep its
                            // n-gram drafts rather than sending it dense - that
                            // is what the all-or-nothing gate used to do for
                            // the whole batch.
                            Some(_) => chunk.extend(slot.draft.draft(slot.k_now.min(k_budget))),
                            None => chunk.extend(slot.draft.draft(slot.k_now.min(k_budget))),
                        }
                    }
                    reqs.push((k, slot.pos as usize, chunk));
                }
                // Nothing to verify: no drafter, and the n-gram lookup found
                // no continuation for any slot. The round would be a decode
                // tick done the expensive way; the dense tick below is the
                // same token at its own speed. No cooldown - the next tick's
                // lookup may find one. (With a drafter, all-length-1 rounds
                // still run: the verify re-points cold slots' heads.)
                let nothing_to_verify =
                    !generator.spec_capable() && reqs.iter().all(|r| r.2.len() == 1);
                match if nothing_to_verify {
                    Ok(None)
                } else {
                    generator.forward_spec_batch(&reqs)
                } {
                    Ok(Some(picks)) => {
                        let mut base = 0usize;
                        let mut accs: Vec<usize> = Vec::with_capacity(live.len());
                        for (ri, &k) in live.iter().enumerate() {
                            let chunk = &reqs[ri].2;
                            let mut dead = false;
                            {
                                let slot = slots[k].as_mut().expect("live");
                                let mut a = 0usize;
                                while a + 1 < chunk.len() && chunk[a + 1] == picks[base + a] {
                                    a += 1;
                                }
                                accs.push(a + 1);
                                if chunk.len() > 1 {
                                    if a == chunk.len() - 1 {
                                        slot.k_now = (slot.k_now * 2).min(serve_spec_max_k());
                                    } else {
                                        slot.k_now = (a + 1).clamp(
                                            generator
                                                .spec_k_miss_floor()
                                                .unwrap_or_else(serve_spec_k_floor)
                                                .min(slot.k_now),
                                            slot.k_now,
                                        );
                                    }
                                }
                                slot.spec_round(&mut tick_tally, chunk.len() - 1, a);
                                // rows 0..=a became context; the bonus pick is
                                // the new pending (set by accept)
                                slot.pos += (a + 1) as u32;
                                for &t in &picks[base..=base + a] {
                                    slot.draft.push(t);
                                    if !slot.accept(t, None) {
                                        dead = true;
                                        break;
                                    }
                                }
                            }
                            if dead {
                                slots[k] = None;
                            }
                            base += chunk.len();
                        }
                        if paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some() {
                            let lens: Vec<usize> = reqs.iter().map(|r| r.2.len()).collect();
                            tracing::info!(
                                "[spec2a-host] round n={} chunk_lens={lens:?} accepted={accs:?} round_us={}",
                                reqs.len(),
                                t0s.elapsed().as_micros()
                            );
                            // replay-grade dump: exact per-slot verify inputs
                            // and outputs, so a diverging round can be re-run
                            // in-engine with identical inputs
                            if paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG_IDS").is_some() {
                                let mut base2 = 0usize;
                                for (ri, &(slot, pos, ref chunk)) in reqs.iter().enumerate() {
                                    tracing::info!(
                                        "[spec2a-ids] slot={slot} pos={pos} chunk={chunk:?} picks={:?}",
                                        &picks[base2..base2 + chunk.len()]
                                    );
                                    base2 += chunk.len();
                                    let _ = ri;
                                }
                            }
                        }
                        st[4].0 += 1;
                        st[4].1 += t0s.elapsed().as_nanos() as u64;
                        continue; // round done - next tick
                    }
                    // backend can't spec this tick (no support, or model-draft
                    // state stale) - cool down and re-probe later
                    Ok(None) if !nothing_to_verify => {
                        spec_retry_at = spec_ticks + 256;
                        st[4].1 += t0s.elapsed().as_nanos() as u64;
                    }
                    Ok(None) => {}
                    Err(e) => {
                        for slot in slots.iter_mut() {
                            if let Some(s) = slot.take() {
                                let _ = s.events.send(TokenEvent::Error(EngineError::from_gen(&e)));
                            }
                        }
                        continue;
                    }
                }
            } else if dev_ok {
                // Phase 2b-dev: DEVICE-SAMPLED speculative round - every live
                // slot is greedy or temperature-only, so each verify row is
                // sampled on device with a pre-drawn plan (the dense
                // device-sampling semantics: exact, no logits readback,
                // scales to the full row budget - the c8 path). Backends
                // without it (gpt-oss) decline -> cooldown.
                // Truncation slots join with TruncCat verify
                // plans - sampled verify + accept-while-match is exact
                // rejection sampling for the deterministic MTP drafts
                // (is_spec_safe covers truncation). Not behind the P67b
                // PADDOCK_TRUNC_PIPE gate: spec rounds are synchronous per
                // round, so the pipe begin/drain churn thrash that parked
                // the pipe surfaces cannot occur here (serve gate:
                // 50 [spec-plans] ROUNDs / 0 DECLINEs, coherent).
                //
                // Re-sync the draft head first: mixed dense ticks (which
                // preempt spec while any slot chunks) advance the backbone KV
                // but not the MTP warm, so a cohort's early finishers reach here
                // with the draft head behind their decode cursor. Left alone
                // every round would decline on the pos check and spec would
                // never engage past c1 (which never desyncs - it decodes alone).
                // A no-op when already synced (the steady-state hot path).
                // PER-SLOT warmth: the all-or-nothing gate meant a
                // single cold joiner emptied `pendings`, spec_draft_batch
                // declined, and the Ok(None) arm below tripped a 256-TICK
                // spec blackout - the c32 "nine verify rounds per sweep"
                // pathology. Cold slots now ride the verify as length-1
                // chunks; the verify's h re-point warms them for next tick.
                let t0s = std::time::Instant::now(); // spec bucket: warmth + chain + verify
                let warm: Vec<bool> = spec_warm_vec(generator, &slots, &live);
                // k_budget was decided before the branch (and is > 0 here)
                tick_open = Some((std::time::Instant::now(), live.len(), k_budget, false));
                let any_warm = warm.iter().any(|&w| w);
                let pendings: Vec<(usize, u32)> = if any_warm {
                    live.iter()
                        .zip(&warm)
                        .filter(|&(_, &w)| w)
                        .map(|(&k, _)| (k, slots[k].as_ref().expect("live").pending))
                        .collect()
                } else {
                    Vec::new()
                };
                // Async round: enqueue the chain and launch verify
                // without reading drafts back - the backend assembles the
                // verify tokens on device, and the chain->verify boundary
                // stays a queued stream sequence instead of a host stall.
                // Drafts arrive via spec_draft_fetch after the verify for
                // the accept replay. Kill: PADDOCK_NO_ASYNC_SPEC.
                let async_k: Option<(usize, Vec<bool>)> = if any_warm && async_spec_on() {
                    // PADDOCK_SPEC_RS: hand the backend this
                    // round's chain draws - drafter-softmax inv_t + one
                    // uniform per potential step, from each slot's own seed
                    // stream - so drafts are SAMPLED from q instead of
                    // argmax'd. Greedy slots draw inv_t 0 (argmax rows).
                    if generator.supports_spec_rs() {
                        let draws: Vec<crate::generator::SpecRsDraw> = pendings
                            .iter()
                            .map(|&(k, _)| {
                                let slot = slots[k].as_mut().expect("live");
                                let (inv_t, u) = slot.sampler.rs_chain_draw(k_budget.min(16));
                                crate::generator::SpecRsDraw { slot: k, inv_t, u }
                            })
                            .collect();
                        generator.spec_rs_stash(draws);
                    }
                    generator
                        .spec_draft_begin(&pendings, k_budget)
                        .unwrap_or(None)
                } else {
                    None
                };
                let model_drafts = if async_k.is_some() || !any_warm {
                    None
                } else {
                    let draft_k = synchronous_draft_budget(
                        k_budget,
                        generator.spec_block_width(),
                        pendings
                            .iter()
                            .map(|&(k, _)| slots[k].as_ref().expect("warm").k_now),
                    );
                    match generator.spec_draft_batch(&pendings, draft_k) {
                        Ok(d) => d,
                        Err(e) => {
                            if paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some() {
                                tracing::info!("[spec2a] draft ERR: {e}");
                            }
                            None
                        }
                    }
                };
                {
                    let drafts = model_drafts; // Option<Vec<Vec<u32>>> in warm order
                    let mut reqs: Vec<(usize, usize, Vec<u32>)> = Vec::with_capacity(live.len());
                    let mut wi = 0usize;
                    for (i, &k) in live.iter().enumerate() {
                        let slot = slots[k].as_ref().expect("live");
                        let mut chunk = vec![slot.pending];
                        if warm[i] {
                            if let Some((ke, kept)) = async_k.as_ref() {
                                // placeholder VALUES (device assembly holds
                                // the real drafts); LENGTH is the contract -
                                // chain-cold entries stay length-1 exactly
                                // like the sync path (row counts = graph keys)
                                if kept.get(wi).copied().unwrap_or(false) {
                                    let cap = slot_spec_k(slot.k_now, k_budget, fixed_draft_depth)
                                        .min(*ke);
                                    chunk.resize(1 + cap, slot.pending);
                                }
                            } else if let Some(d) = drafts.as_ref() {
                                let cap = slot_spec_k(slot.k_now, k_budget, fixed_draft_depth);
                                chunk.extend(d[wi].iter().copied().take(cap));
                            }
                            wi += 1;
                        }
                        reqs.push((k, slot.pos as usize, chunk));
                    }
                    // one plan (one uniform) per chunk row from the slot's
                    // own seed stream; rows past the first mismatch discard
                    // theirs (distribution-identical, not seed-replay-equal)
                    let mut plans: Vec<crate::sampler::DevicePlan> =
                        Vec::with_capacity(reqs.iter().map(|r| r.2.len()).sum());
                    // On RS rounds, drafted rows (every row with a
                    // next chunk element) carry the canonical accept/recover
                    // plan (two uniforms); the last row (bonus) and greedy
                    // slots keep the classic plan. Async only - the backend's
                    // chain sampled q exactly when the draws were stashed.
                    let rs_round = async_k.is_some() && generator.supports_spec_rs();
                    // trunc slots draw TruncCat verify plans (never RS
                    // - the RS resolve computes p over the full softmax,
                    // which is the wrong target for a truncated row;
                    // rs_verify_plan already declines them)
                    // Rung G...unless the backend resolves
                    // against the NUCLEUS itself (the DFlash2 lane's
                    // K-candidate sampler): then trunc slots' drafted rows
                    // ride RsTrunc, which carries the slot's k/top_p/min_p.
                    let rs_trunc_ok = rs_round && generator.supports_spec_rs_trunc();
                    let spec_trunc_ok = generator.supports_device_trunc();
                    for (ri, &k) in live.iter().enumerate() {
                        let slot = slots[k].as_mut().expect("live");
                        let clen = reqs[ri].2.len();
                        for j in 0..clen {
                            let base = |s: &mut Slot| {
                                if spec_trunc_ok {
                                    s.sampler.device_plan_trunc()
                                } else {
                                    s.sampler.device_plan()
                                }
                            };
                            let p = if rs_round && j + 1 < clen {
                                slot.sampler
                                    .rs_verify_plan()
                                    .or_else(|| {
                                        if rs_trunc_ok {
                                            slot.sampler.rs_trunc_plan()
                                        } else {
                                            None
                                        }
                                    })
                                    .or_else(|| base(slot))
                            } else {
                                base(slot)
                            };
                            plans.push(p.unwrap_or(crate::sampler::DevicePlan::Greedy));
                        }
                    }
                    // Armed rounds with the device accept
                    // available run STRIP mode - the accept-while-match walk
                    // happens on device, one compact strip comes back, and
                    // the fill + replay below never run.
                    if async_k.is_some() && generator.supports_spec_strip() {
                        match generator.forward_spec_batch_strip(&reqs, &plans) {
                            Ok(Some(strip)) => {
                                let mut dead: Vec<usize> = Vec::new();
                                for (ri, &k) in live.iter().enumerate() {
                                    let sa = &strip[ri];
                                    let chunk_len = reqs[ri].2.len();
                                    let slot = slots[k].as_mut().expect("live");
                                    if chunk_len > 1 {
                                        if sa.accepted == chunk_len {
                                            slot.k_now = (slot.k_now * 2).min(serve_spec_max_k());
                                        } else {
                                            slot.k_now = sa.accepted.clamp(
                                                generator
                                                    .spec_k_miss_floor()
                                                    .unwrap_or_else(serve_spec_k_floor)
                                                    .min(slot.k_now),
                                                slot.k_now,
                                            );
                                        }
                                    }
                                    // sa.accepted counts the committed run (a+1)
                                    slot.spec_round(
                                        &mut tick_tally,
                                        chunk_len - 1,
                                        sa.accepted.saturating_sub(1),
                                    );

                                    slot.pos += sa.accepted as u32;
                                    slot.pending = sa.pending;
                                    let mut slot_dead = false;
                                    for &t in &sa.tokens {
                                        slot.draft.push(t);
                                        if !slot.accept(t, None) {
                                            slot_dead = true;
                                            break;
                                        }
                                    }
                                    if slot_dead {
                                        dead.push(k);
                                    }
                                }
                                let any_dead = !dead.is_empty();
                                for k in dead {
                                    slots[k] = None;
                                }
                                st[4].0 += 1;
                                st[4].1 += t0s.elapsed().as_nanos() as u64;
                                // Steady strip round ->
                                // run the ONE-AHEAD pipeline until an event
                                // (arrival, stop, par unavailable, error).
                                // Round N+1 is enqueued before round N's
                                // strip is read, so the GPU never waits for
                                // the host between rounds. Opt-in:
                                // PADDOCK_SPEC_PIPE=1 (needs strip mode).
                                // entry gates: FULL-DEPTH chunks only (the
                                // pipe pins k for its whole segment, so
                                // entering shallow locks a long stretch at
                                // the entry k while the suspended k_now
                                // adaptation never deepens it) and c32-band
                                // width (narrow rounds just pay strip's
                                // fixed cost).
                                // k_budget >= 2: at k_budget 0/1 the full-depth
                                // check is trivially true (length-1 chunks) and
                                // the pipe pins a NO-SPEC segment - hundreds
                                // of rounds locked at k=1.
                                if !any_dead
                                    && live.len() >= 16
                                    && k_budget >= 2
                                    && reqs[0].2.len() == k_budget + 1
                                    && reqs.iter().all(|q| q.2.len() == reqs[0].2.len())
                                    && generator.spec_pipe_arm()
                                {
                                    let k1 = reqs[0].2.len();
                                    // Book the ENTRY round now and stamp the
                                    // segment start - the pipe loop never
                                    // passes loop-top, so leaving tick_open
                                    // armed books the whole segment's wall as
                                    // one tick at (live, k): the same cell
                                    // poisoning as the mixed-round booking,
                                    // self-reinforcing here because the
                                    // poisoned cell shallows the next entry.
                                    if let Some((t0, ln, kk, _)) = tick_open.take() {
                                        spec_ctl.observe(
                                            ln,
                                            kk,
                                            t0.elapsed().as_secs_f64(),
                                            tick_tally,
                                        );
                                    }
                                    tick_tally = RoundTally::default();
                                    let seg_t0 = std::time::Instant::now();
                                    let mut seg_rounds = 0usize;
                                    let mut read_half = 0usize;
                                    let mut inflight = 0usize;
                                    let mut pipe_admit = None;
                                    let mut alive = true;
                                    loop {
                                        if alive && inflight < 2 {
                                            if pipe_admit.is_none()
                                                && let Ok(req) = rx.try_recv()
                                            {
                                                pipe_admit = Some(req);
                                            }
                                            let par: Option<Vec<u32>> = if pipe_admit.is_some() {
                                                None
                                            } else {
                                                (|| {
                                                    let mut par = vec![0u32; live.len() * k1 * 4];
                                                    for (ri, &k) in live.iter().enumerate() {
                                                        let slot = slots[k].as_mut()?;
                                                        for j in 0..k1 {
                                                            let p = slot.sampler.device_plan()?;
                                                            let i = (ri * k1 + j) * 4;
                                                            match p {
                                                                crate::sampler::DevicePlan::Greedy => par[i + 2] = 1,
                                                                crate::sampler::DevicePlan::Categorical { inv_t, u } => {
                                                                    par[i] = inv_t.to_bits();
                                                                    par[i + 1] = u.to_bits();
                                                                    par[i + 2] = 2;
                                                                }
                                                                // pipe declines RS/TruncCat; device_plan never yields either
                                                                crate::sampler::DevicePlan::RsVerify { .. }
                                                                | crate::sampler::DevicePlan::RsTrunc { .. }
                                                                | crate::sampler::DevicePlan::TruncCat { .. } => par[i + 2] = 1,
                                                            }
                                                        }
                                                    }
                                                    Some(par)
                                                })()
                                            };
                                            match par {
                                                Some(par) => {
                                                    let hs: Vec<u32> =
                                                        live.iter().map(|&k| k as u32).collect();
                                                    let hp: Vec<u32> = live
                                                        .iter()
                                                        .map(|&k| {
                                                            slots[k]
                                                                .as_ref()
                                                                .map(|s| s.pos + 2 * k1 as u32)
                                                                .unwrap_or(0)
                                                        })
                                                        .collect();
                                                    if generator.spec_pipe_ensure(&hs, &hp).is_err()
                                                        || generator.spec_pipe_round(&par).is_err()
                                                    {
                                                        alive = false;
                                                    } else {
                                                        inflight += 1;
                                                    }
                                                }
                                                None => alive = false,
                                            }
                                        }
                                        if inflight == 0 {
                                            break;
                                        }
                                        match generator.spec_pipe_strip(read_half) {
                                            Ok(strip) => {
                                                inflight -= 1;
                                                read_half ^= 1;
                                                for (ri, &k) in live.iter().enumerate() {
                                                    let Some(sa) = strip.get(ri) else {
                                                        continue;
                                                    };
                                                    let Some(slot) = slots[k].as_mut() else {
                                                        alive = false;
                                                        continue;
                                                    };
                                                    // pipe rounds run fixed-depth chunks (k1 rows)
                                                    slot.spec_round(
                                                        &mut tick_tally,
                                                        k1 - 1,
                                                        sa.accepted.saturating_sub(1),
                                                    );
                                                    slot.pos += sa.accepted as u32;
                                                    slot.pending = sa.pending;
                                                    let mut sdead = false;
                                                    for &t in &sa.tokens {
                                                        slot.draft.push(t);
                                                        if !slot.accept(t, None) {
                                                            sdead = true;
                                                            break;
                                                        }
                                                    }
                                                    if sdead {
                                                        slots[k] = None;
                                                        alive = false;
                                                    }
                                                }
                                                st[4].0 += 1;
                                                seg_rounds += 1;
                                            }
                                            Err(e) => {
                                                tracing::warn!(
                                                    "serve: spec pipe strip failed: {e}"
                                                );
                                                alive = false;
                                                inflight = 0;
                                            }
                                        }
                                    }
                                    if let Err(e) = generator.spec_pipe_drain() {
                                        tracing::warn!("serve: spec pipe drain failed: {e}");
                                    }
                                    // Honest segment booking: one observation at
                                    // the segment's per-round average, with the
                                    // aggregate tally (ratio of sums is the
                                    // right alpha update). tick_open is already
                                    // None (taken at entry), so loop-top books
                                    // nothing extra for this iteration.
                                    if seg_rounds > 0 {
                                        spec_ctl.observe(
                                            live.len(),
                                            k1 - 1,
                                            seg_t0.elapsed().as_secs_f64() / seg_rounds as f64,
                                            tick_tally,
                                        );
                                        tick_tally = RoundTally::default();
                                    }
                                    if paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some() {
                                        tracing::info!(
                                            "[g4-pipe] segment done rounds={} exit_admit={}",
                                            st[4].0,
                                            pipe_admit.is_some()
                                        );
                                    }
                                    if let Some(req) = pipe_admit {
                                        debit_pool_budget(&mut admit_budget, &req);
                                        admit(&mut slots, req, generator.max_context(), metrics);
                                        ticks_since_admit = 0;
                                    }
                                }
                                continue; // round done - next tick
                            }
                            Ok(None) => {
                                // round declined mid-arm - clear any armed
                                // plan and cool down like the plans path
                                let _ = generator.spec_draft_fetch();
                                spec_retry_at = spec_ticks + 256;
                                st[4].1 += t0s.elapsed().as_nanos() as u64;
                            }
                            Err(e) => {
                                let _ = generator.spec_draft_fetch();
                                for slot in slots.iter_mut() {
                                    if let Some(s) = slot.take() {
                                        let _ = s
                                            .events
                                            .send(TokenEvent::Error(EngineError::from_gen(&e)));
                                    }
                                }
                                continue;
                            }
                        }
                    } else {
                        match generator.forward_spec_batch_plans(&reqs, &plans) {
                            Ok(Some(picks)) => {
                                // async round: fill the placeholder chunks with
                                // the real drafts (fetched post-verify - the
                                // stream is long past the chain) so the accept
                                // replay below compares what verify actually saw.
                                // Cold-of-chain slots keep their pending pads
                                // (matching the device assembly's pad rule).
                                if async_k.is_some()
                                    && let Ok(Some(dr)) = generator.spec_draft_fetch()
                                {
                                    let mut wi2 = 0usize;
                                    for (ri, _) in live.iter().enumerate() {
                                        if warm[ri] {
                                            if let Some(d) = dr.get(wi2) {
                                                let c = &mut reqs[ri].2;
                                                for j in 1..c.len() {
                                                    if j - 1 < d.len() {
                                                        c[j] = d[j - 1];
                                                    }
                                                }
                                            }
                                            wi2 += 1;
                                        }
                                    }
                                }
                                let mut dead: Vec<usize> = Vec::new();
                                let mut base = 0usize;
                                for (ri, &k) in live.iter().enumerate() {
                                    let chunk = &reqs[ri].2;
                                    let slot = slots[k].as_mut().expect("live");
                                    let mut a = 0usize;
                                    while a + 1 < chunk.len() && chunk[a + 1] == picks[base + a] {
                                        a += 1;
                                    }
                                    if chunk.len() > 1 {
                                        if a == chunk.len() - 1 {
                                            slot.k_now = (slot.k_now * 2).min(serve_spec_max_k());
                                        } else {
                                            slot.k_now = (a + 1).clamp(
                                                generator
                                                    .spec_k_miss_floor()
                                                    .unwrap_or_else(serve_spec_k_floor)
                                                    .min(slot.k_now),
                                                slot.k_now,
                                            );
                                        }
                                    }
                                    slot.spec_round(&mut tick_tally, chunk.len() - 1, a);
                                    slot.pos += (a + 1) as u32;
                                    slot.pending = picks[base + a];
                                    let mut slot_dead = false;
                                    for &t in &picks[base..=base + a] {
                                        slot.draft.push(t);
                                        if !slot.accept(t, None) {
                                            slot_dead = true;
                                            break;
                                        }
                                    }
                                    if slot_dead {
                                        dead.push(k);
                                    }
                                    base += chunk.len();
                                }
                                for k in dead {
                                    slots[k] = None;
                                }
                                st[4].0 += 1;
                                st[4].1 += t0s.elapsed().as_nanos() as u64;
                                continue; // round done - next tick
                            }
                            Ok(None) => {
                                // clear any armed async plan (the drafts are
                                // valid but this round declined - discard)
                                if async_k.is_some() {
                                    let _ = generator.spec_draft_fetch();
                                }
                                spec_retry_at = spec_ticks + 256;
                                st[4].1 += t0s.elapsed().as_nanos() as u64;
                            }
                            Err(e) => {
                                if async_k.is_some() {
                                    let _ = generator.spec_draft_fetch();
                                }
                                for slot in slots.iter_mut() {
                                    if let Some(s) = slot.take() {
                                        let _ = s
                                            .events
                                            .send(TokenEvent::Error(EngineError::from_gen(&e)));
                                    }
                                }
                                continue;
                            }
                        }
                    }
                }
            } else {
                // Phase 2b - SAMPLED speculative round: every live slot's
                // sampling is row-local (temperature/top-k/top-p/min-p; no
                // penalties/bias/constraints/logprobs). With deterministic
                // model drafts, sampling each verify row with the slot's own
                // sampler and accepting while the sample equals the draft is
                // exact rejection sampling - the emitted distribution matches
                // the dense path's (sampler::is_spec_safe). The backend
                // returns raw row logits; commit counts go back after
                // acceptance. RNG draw COUNT differs from the dense path
                // (discarded rows consume uniforms) - distribution-identical,
                // not seed-replay-identical.
                // sampled rounds pay a [rows x vocab] logits readback + host
                // sampling per row - measured NET LOSS at live=8 (pf8 -16%);
                // cap the live count (greedy rounds keep the bigger cap, they
                // read back nothing). Device top-k pre-truncation would lift
                // this - the Phase-B-opt item.
                let sampled_live_max: usize =
                    paddock_models::dev_var!("PADDOCK_SPEC_SAMPLED_LIVE_MAX")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or_else(|| {
                            generator
                                .spec_sampled_live_cap()
                                .map_or(4, |c| c.min(generator.spec_live_cap()))
                        });
                let spec_safe = !live.is_empty()
                    && live.len() <= sampled_live_max.min(serve_spec_max_rows())
                    && live.iter().all(|&k| {
                        let s = slots[k].as_ref().expect("live");
                        s.sampler.is_spec_safe()
                            // Constrained slots join this round in every
                            // phase: the host walk below samples each verify
                            // pick through the machine (exact pick_next
                            // semantics), so free phases speculate mask-free
                            // and active regions (tool-call JSON, forced
                            // structured output) stay grammar-legal by
                            // construction with drafts as accept-while-match
                            // acceleration. No bitmask, no rollback - the
                            // walk is sequential and only committed picks
                            // advance the machine. Device/strip rounds above
                            // keep requiring None: their acceptance resolves
                            // on device where no machine can sit.
                            && s.logprobs.is_none()
                    });
                if spec_safe {
                    let k_budget = backend_spec_k_budget(generator, live.len());
                    let pendings: Vec<(usize, u32)> = live
                        .iter()
                        .map(|&k| (k, slots[k].as_ref().expect("live").pending))
                        .collect();
                    // The warm pass every other round runs first (see the
                    // greedy and device branches): it re-syncs the draft head
                    // over whatever advanced the slot without it - a prefix-
                    // cache resume, a dense tick, a cooldown. Skipped here,
                    // the first desync after a session's first turn left
                    // every later round of this branch - the one constrained
                    // slots (every tool-carrying agent request) speculate
                    // through - drafting from a stale head until the verify
                    // declined into the 256-tick cooldown, over and over.
                    let any_warm =
                        k_budget > 0 && spec_warm_vec(generator, &slots, &live).iter().any(|&w| w);
                    let model_drafts = if any_warm {
                        let draft_k = synchronous_draft_budget(
                            k_budget,
                            generator.spec_block_width(),
                            pendings
                                .iter()
                                .map(|&(k, _)| slots[k].as_ref().expect("live").k_now),
                        );
                        generator
                            .spec_draft_batch(&pendings, draft_k)
                            .unwrap_or(None)
                    } else {
                        None
                    };
                    // sampled rounds need model drafts: n-gram acceptance at
                    // temp>0 is too low to pay for the logits readback
                    if let Some(drafts) = model_drafts {
                        let mut reqs: Vec<(usize, usize, Vec<u32>)> =
                            Vec::with_capacity(live.len());
                        for (i, &k) in live.iter().enumerate() {
                            let slot = slots[k].as_ref().expect("live");
                            let mut chunk = vec![slot.pending];
                            let cap = slot_spec_k(slot.k_now, k_budget, fixed_draft_depth);
                            chunk.extend(drafts[i].iter().copied().take(cap));
                            reqs.push((k, slot.pos as usize, chunk));
                        }
                        match generator.forward_spec_verify(&reqs) {
                            Ok(Some(mut rows)) => {
                                let mut committed: Vec<u32> = Vec::with_capacity(reqs.len());
                                let mut dead: Vec<usize> = Vec::new();
                                let mut base_row = 0usize;
                                for (ri, &k) in live.iter().enumerate() {
                                    let chunk = &reqs[ri].2;
                                    let slot = slots[k].as_mut().expect("live");
                                    // sample rows until the first mismatch -
                                    // that row's sample is the replacement
                                    let mut picks: Vec<u32> = Vec::with_capacity(chunk.len());
                                    let mut deadlock = false;
                                    for j in 0..chunk.len() {
                                        let row = &mut rows
                                            [(base_row + j) * vocab..(base_row + j + 1) * vocab];
                                        // Constrained slots sample each verify
                                        // pick through the machine - literally
                                        // pick_next's semantics at the verify
                                        // position, with the draft row as the
                                        // free acceleration. Free phases cost
                                        // nothing (allows() is all-true); in a
                                        // grammar region the pick is legal by
                                        // construction and drafts that violate
                                        // it simply mismatch and cut the run.
                                        let t = match slot.constraint.as_mut() {
                                            Some(c) => {
                                                let may_stop = c.may_stop();
                                                let stops = &slot.stop_tokens;
                                                match slot.sampler.sample_constrained(
                                                    row,
                                                    &[],
                                                    &mut |id| {
                                                        if stops.contains(&id) {
                                                            may_stop
                                                        } else {
                                                            c.allows(id)
                                                        }
                                                    },
                                                ) {
                                                    Some(t) => t,
                                                    None => {
                                                        // Grammar deadlock -
                                                        // unreachable for live
                                                        // machines. Emit the
                                                        // unconstrained pick
                                                        // without advancing the
                                                        // machine and end the
                                                        // run: the next dense
                                                        // step hits pick_next's
                                                        // own deadlock error
                                                        // and fails the request
                                                        // properly. No novel
                                                        // 0-commit rollback
                                                        // path, no contract
                                                        // bend.
                                                        deadlock = true;
                                                        slot.sampler.sample(row, &[])
                                                    }
                                                }
                                            }
                                            None => slot.sampler.sample(row, &[]),
                                        };
                                        picks.push(t);
                                        if deadlock {
                                            break;
                                        }
                                        if slot.stop_tokens.contains(&t) {
                                            // stop tokens never feed the machine
                                            // (pick_next's contract); the commit
                                            // walk's slot.accept sends Done
                                            break;
                                        }
                                        if let Some(c) = slot.constraint.as_mut() {
                                            c.accept(t);
                                        }
                                        // a call opening mid-round (and maybe
                                        // closing again before the round ends)
                                        // is seen here, per pick
                                        slot.note_constraint();
                                        if j + 1 < chunk.len() && t != chunk[j + 1] {
                                            break;
                                        }
                                    }

                                    let a = picks.len() - 1;
                                    if chunk.len() > 1 {
                                        if a == chunk.len() - 1 {
                                            slot.k_now = (slot.k_now * 2).min(serve_spec_max_k());
                                        } else {
                                            slot.k_now = (a + 1).clamp(
                                                generator
                                                    .spec_k_miss_floor()
                                                    .unwrap_or_else(serve_spec_k_floor)
                                                    .min(slot.k_now),
                                                slot.k_now,
                                            );
                                        }
                                    }
                                    slot.spec_round(&mut tick_tally, chunk.len() - 1, a);
                                    slot.pos += (a + 1) as u32;
                                    slot.pending = picks[a];
                                    let mut slot_dead = false;
                                    for &t in &picks {
                                        slot.draft.push(t);
                                        if !slot.accept(t, None) {
                                            slot_dead = true;
                                            break;
                                        }
                                    }
                                    committed.push((a + 1) as u32);
                                    if slot_dead {
                                        dead.push(k);
                                    }
                                    base_row += chunk.len();
                                }
                                // before the commit snapshots past the call
                                drain_reply_pins(generator, &mut slots);
                                if let Err(e) = generator.spec_commit(&committed) {
                                    for slot in slots.iter_mut() {
                                        if let Some(s) = slot.take() {
                                            let _ = s
                                                .events
                                                .send(TokenEvent::Error(EngineError::from_gen(&e)));
                                        }
                                    }
                                    continue;
                                }
                                for k in dead {
                                    slots[k] = None;
                                }
                                continue; // round done - next tick
                            }
                            Ok(None) => {
                                if paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some() {
                                    tracing::info!(
                                        "[spec-sampled] verify declined - 256-tick cooldown"
                                    );
                                }
                                spec_retry_at = spec_ticks + 256
                            }
                            Err(e) => {
                                for slot in slots.iter_mut() {
                                    if let Some(s) = slot.take() {
                                        let _ = s
                                            .events
                                            .send(TokenEvent::Error(EngineError::from_gen(&e)));
                                    }
                                }
                                continue;
                            }
                        }
                    }
                }
            }
        }

        ph_spec = tick_t0.elapsed();
        // Phase 2 - one decode step over the active (prefilled) slots. Dense batch
        // over [0, high_water): active rows feed their pending token at their
        // position, holes feed (0, 0) and are ignored.
        //
        // "Active" means PREFILLED, not merely occupied, and the difference is
        // not academic: an occupied slot with no KV behind it samples a token
        // out of a hole row and streams it to the client. That used to be
        // unreachable - admission prefilled synchronously, and a mid-chunk slot
        // is only ever reached through the mixed tick above - until the encoder
        // budget gave a slot a legitimate reason to sit occupied and
        // unprefilled for several ticks. It cost exactly one junk token per
        // encode tick, prepended to an otherwise perfect answer.
        let high_water = slots.iter().rposition(|s| s.is_some()).map_or(0, |i| i + 1);
        if high_water == 0 {
            continue;
        }
        let mut tokens = vec![0u32; high_water];
        let mut positions = vec![0u32; high_water];
        for (k, slot) in slots.iter().enumerate().take(high_water) {
            if let Some(s) = slot
                && s.prefilled
            {
                tokens[k] = s.pending;
                positions[k] = s.pos;
            }
        }

        // Device-sampled step: rows whose sampling needs no host logits
        // (greedy / temperature-only - the OpenAI-default hot path) are
        // picked on device and return bare token ids; the rare Host row
        // (penalties, filters, constraint, logprobs) still gets its own
        // logits row back. Kills the [B, vocab] readback AND the host
        // sampling fan-out in one move.
        if samp_supported {
            // Truncation rows head-sample where the backend supports it
            // (the pipe-eligibility test below admits TruncCat only for
            // full-device-trunc backends - P67b)
            let trunc_ok = generator.supports_host_head() || generator.supports_device_trunc();
            let trunc_pipe_ok = generator.supports_device_trunc() && trunc_pipe_env();
            let plans: Vec<RowSample> = slots[..high_water]
                .iter_mut()
                .map(|s| match s {
                    None => RowSample::Hole,
                    // occupied but with no KV yet (mid encoder budget): a hole
                    // for this tick, and drawing no plan keeps its seed stream
                    // untouched - the token it eventually samples must be the
                    // one it would have sampled had the encode been instant
                    Some(sl) if !sl.prefilled => RowSample::Hole,
                    Some(sl) if sl.constraint.is_none() && sl.logprobs.is_none() => {
                        // drawing a Categorical plan consumes this token's
                        // one uniform, exactly like the host draw would
                        match if trunc_ok {
                            sl.sampler.device_plan_trunc()
                        } else {
                            sl.sampler.device_plan()
                        } {
                            Some(p) => RowSample::Device(p),
                            // ngram-only greedy rows: device argmax on ticks
                            // the guard would ban nothing (see the mixed-tick
                            // site) - ngram_dev_ok already excludes the pipe,
                            // whose lookahead this per-tick check can't serve
                            None if ngram_dev_ok
                                && sl.sampler.is_greedy_ngram_only()
                                && !sl.sampler.ngram_would_ban(&sl.history) =>
                            {
                                RowSample::Device(crate::sampler::DevicePlan::Greedy)
                            }
                            None => RowSample::Host,
                        }
                    }
                    Some(_) => RowSample::Host,
                })
                .collect();
            // Pipelined variant: only when every row samples on device (a
            // Host row needs its logits on the tick - no lookahead possible).
            // A begin failure falls through to the classic single-tick path
            // with the same (already-drawn) plans.
            // Dynamic pool-headroom gate (oversubscribed pools):
            // a pipe segment may grow the pool by up to B blocks in one tick
            // (equal-length rows cross block boundaries in sync), so begin -
            // and below, continue - only while the pool holds a worst-case
            // tick of growth plus the admission watermark. Fully-backed pools
            // report enough headroom always; None = no pool = unlimited.
            let pipe_headroom = |g: &mut dyn Generator| {
                g.pool_free_blocks()
                    .is_none_or(|f| f > high_water + POOL_WATERMARK_BLOCKS)
            };
            // requests in flight - the pipe_min_live floor (pure-decode phase:
            // every live slot is a decoder)
            let live_slots = slots.iter().filter(|s| s.is_some()).count();
            let pipe_begun = pipe_supported
                && live_slots >= pipe_min_live()
                && ticks_since_admit >= pipe_min_quiet.saturating_add(pipe_backoff)
                // host-head TruncCat cannot ride the zero-host pipe;
                // full-device (mode 5) TruncCat can
                && plans.iter().all(|p| match p {
                    RowSample::Host => false,
                    RowSample::Device(crate::sampler::DevicePlan::TruncCat { .. }) => {
                        trunc_pipe_ok
                    }
                    _ => true,
                })
                && pipe_headroom(generator)
                && match generator.decode_pipe_begin(&tokens, &positions, &plans) {
                    Ok(()) => true,
                    Err(e) => {
                        // loud once: a silent fallback here costs the whole
                        // overlap win and looks like a kernel regression
                        // from the outside
                        static ONCE: std::sync::Once = std::sync::Once::new();
                        ONCE.call_once(|| {
                            tracing::warn!("serve: decode pipe unavailable, classic ticks: {e}");
                        });
                        false
                    }
                };
            if pipe_begun {
                // `prev_plans` = plans of the OLDEST in-flight tick - the one
                // whose ids the next decode_pipe_next/drain call returns.
                let mut prev_plans = plans;
                loop {
                    // Draw the next tick's plans first (per-slot RNG order is
                    // identical to the classic path - draws don't depend on
                    // the pending token). Any Host-needing row ends the pipe.
                    // the trunc-aware draw here is LOAD-BEARING - with
                    // plain device_plan() a mode-5 row Nones on tick 2, ends
                    // the pipe, and the outer loop re-begins it next
                    // iteration: begin/drain every tick, which is exactly
                    // the churn-shaped syn regression the first P67b smoke
                    // recorded (c32 2321->2039). The overlap twin already
                    // drew trunc-aware; this site was the miss.
                    let mut host_row = false;
                    let next_plans: Vec<RowSample> = slots[..high_water]
                        .iter_mut()
                        .map(|s| match s {
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
                    // Arrivals end the pipe too: the outer loop's admission +
                    // chunked-prefill phases take over next iteration. Only
                    // polled with a free slot, like the outer admission loop.
                    let mut admit_req = None;
                    // Predictive bunching, at the real steady-state admission
                    // site: the pipe breaks per arrival (one mixed pass each -
                    // the sprinkle). With PADDOCK_ADM_PREDICT_K=K>1, poll only
                    // once K slots have freed: the channel holds the bunch and
                    // one pipe-break admits it as a single wave. K=0/1 is
                    // today's behavior exactly.
                    let free_now = slots.iter().filter(|s| s.is_none()).count();
                    // In-leg resync (adm_resync): one-shot escalated hold.
                    let rz = adm_resync().min(slots.len().saturating_sub(4));
                    if rz > 0 {
                        if free_now + 2 >= slots.len() {
                            // true drain (leg boundary): re-arm for the next leg
                            resync_armed = true;
                        } else if resync_armed && free_now == 0 {
                            resync_armed = false;
                            resync_hold = true;
                            if paddock_models::dev_var_os!("PADDOCK_REQ_TRACE").is_some() {
                                tracing::info!(
                                    "req-trace: resync ENGAGED (hold to {rz} free) at {}",
                                    trace_us()
                                );
                            }
                        } else if resync_hold && free_now >= rz {
                            resync_hold = false;
                            if paddock_models::dev_var_os!("PADDOCK_REQ_TRACE").is_some() {
                                tracing::info!(
                                    "req-trace: resync RELEASE (wave {free_now}) at {}",
                                    trace_us()
                                );
                            }
                        }
                    }
                    let poll_min = if resync_hold {
                        rz
                    } else {
                        adm_predict_k().max(1).min(slots.len() / 2)
                    };
                    if !host_row
                        && mm_pending.is_none()
                        && free_now >= poll_min
                        && admit_budget.is_none_or(|b| b > POOL_WATERMARK_BLOCKS)
                        && let Ok(req) = rx.try_recv()
                    {
                        admit_req = Some(req);
                    }
                    // continue-gate: drain before the pool could exhaust under
                    // a worst-case tick of block growth (oversubscribed pools)
                    let low_headroom = !pipe_headroom(generator);
                    if host_row || admit_req.is_some() || low_headroom {
                        if paddock_models::dev_var_os!("PADDOCK_REQ_TRACE").is_some() {
                            tracing::info!(
                                "req-trace: pipe-break host={host_row} arrival={} lowroom={low_headroom} at {}",
                                admit_req.is_some(),
                                trace_us()
                            );
                        }
                        // Overlapped SHORT admission (unlocked by the
                        // decode-graph arena): run the new slot's
                        // whole-prompt pure prefill before draining - on the
                        // one stream its kernels queue behind the ~2 in-flight
                        // pipe ticks, so the GPU stays saturated through the
                        // admission instead of idling for the drain + host
                        // assembly (the churn-load gap). Guards: text
                        // only, prompt <= the row cap (a long decode-row-less
                        // prefill would starve decode cadence - those keep the
                        // mixed flow), and the pass must not realloc scratch
                        // (a realloc would drop graphs with queued replays).
                        if !host_row
                            && admit_req.as_ref().is_some_and(|r| {
                                r.mm_chunks.is_none()
                                    && r.prompt.len() <= overlap_admit_max()
                                    && generator.prefill_scratch_fits(r.prompt.len())
                            })
                        {
                            // WAVE-BATCHED refill (from a width-occupancy
                            // trace): equal-length completions finish in sync,
                            // emptying ~20 slots at once - refilling them one
                            // prefill at a time ran decode at B=7-13 for the
                            // whole refill (mean width 27.2 of 32).
                            // Collect every queued short-text arrival up
                            // to the free slots and prefill them in one
                            // weight-amortized pass; the batch must still fit
                            // the scratch without realloc (queued replays).
                            let mut items: Vec<(usize, Vec<u32>)> = Vec::new();
                            let mut total_rows = 0usize;
                            loop {
                                let Some(req) = admit_req.take() else { break };
                                let free_k = slots.iter().position(|s| s.is_none());
                                debit_pool_budget(&mut admit_budget, &req);
                                admit(&mut slots, req, generator.max_context(), metrics);
                                {
                                    ticks_since_admit = 0;
                                }
                                if let Some(k) = free_k
                                    && slots[k].as_ref().is_some_and(|s| !s.prefilled)
                                {
                                    let prompt = std::mem::take(
                                        &mut slots[k].as_mut().expect("present").prompt,
                                    );
                                    total_rows += prompt.len();
                                    items.push((k, prompt));
                                }
                                // pull the next clustered arrival while slots and
                                // scratch capacity remain (same guards as entry)
                                if slots.iter().any(|s| s.is_none())
                                    && admit_budget.is_none_or(|b| b > POOL_WATERMARK_BLOCKS)
                                    && let Ok(next) = rx.try_recv()
                                {
                                    let ok = next.mm_chunks.is_none()
                                        && next.prompt.len() <= overlap_admit_max()
                                        && generator
                                            .prefill_scratch_fits(total_rows + next.prompt.len());
                                    if ok {
                                        admit_req = Some(next);
                                        continue;
                                    }
                                    // ineligible: hand it to the post-drain path
                                    admit_req = Some(next);
                                }
                                break;
                            }
                            if !items.is_empty() {
                                metrics.phase.store(PHASE_PREFILL, Relaxed);
                                let ks: Vec<usize> = items.iter().map(|(k, _)| *k).collect();
                                let rows: Vec<u32> =
                                    items.iter().map(|(_, p)| p.len() as u32).collect();
                                match generator.forward_prefill_batch(&items) {
                                    Ok(ls) => {
                                        for ((k, logits), r) in ks.iter().zip(ls).zip(rows) {
                                            finish_prefill(
                                                generator, &mut slots, *k, logits, r, metrics,
                                            );
                                        }
                                        spec_retry_at = spec_ticks;
                                    }
                                    Err(e) => {
                                        let ge = EngineError::from_gen(&e);
                                        for k in ks {
                                            if let Some(s) = slots[k].take() {
                                                let _ =
                                                    s.events.send(TokenEvent::Error(ge.clone()));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        st_drain += 1;
                        // backoff update: <12 ticks never amortizes the
                        // ~1.4 ms cold begin; cap keeps the pipe re-triable
                        // (~2 s at c32 cadence) after a churn burst passes
                        if pipe_seg_ticks < 12 {
                            pipe_backoff = (pipe_backoff * 2 + 4).min(192);
                        } else {
                            pipe_backoff /= 2;
                        }
                        pipe_seg_ticks = 0;
                        match generator.decode_pipe_drain() {
                            Ok(ids) => {
                                for (k, plan) in prev_plans.iter().enumerate() {
                                    if matches!(plan, RowSample::Device(_)) {
                                        commit_device_token(&mut slots[k], ids[k]);
                                    }
                                }
                            }
                            Err(e) => {
                                tracing::warn!("serve: decode_pipe_drain failed: {e}");
                                for slot in slots.iter_mut() {
                                    if let Some(s) = slot.take() {
                                        let _ = s
                                            .events
                                            .send(TokenEvent::Error(EngineError::from_gen(&e)));
                                    }
                                }
                            }
                        }
                        if let Some(req) = admit_req {
                            if req.mm_chunks.is_some() && !mm_slots {
                                mm_pending = Some(req);
                            } else {
                                debit_pool_budget(&mut admit_budget, &req);
                                admit(&mut slots, req, generator.max_context(), metrics);
                                {
                                    ticks_since_admit = 0;
                                }
                                spec_retry_at = spec_ticks; // eligible at next prefill
                            }
                        }
                        break;
                    }
                    let t0 = std::time::Instant::now();
                    match generator.decode_pipe_next(&next_plans) {
                        Ok(ids) => {
                            st[0].0 += 1;
                            st[0].1 += t0.elapsed().as_nanos() as u64;
                            ticks_since_admit = ticks_since_admit.saturating_add(1);
                            pipe_seg_ticks = pipe_seg_ticks.saturating_add(1);
                            let mut died = false;
                            for (k, plan) in prev_plans.iter().enumerate() {
                                if matches!(plan, RowSample::Device(_)) {
                                    commit_device_token(&mut slots[k], ids[k]);
                                    if slots[k].is_none() {
                                        died = true;
                                    }
                                }
                            }
                            prev_plans = next_plans;
                            if died && slots[..high_water].iter().any(|s| s.is_some()) {
                                // Keep the pipe HOT past completions: the dead
                                // row keeps ticking as a device-chained dummy -
                                // pipe_launch_tick tops up its blocks each tick
                                // and the None guard discards its ids, so
                                // nothing corrupts. Draining + restarting here
                                // paid a cold ~1.4 ms graph launch per
                                // completion (~8/s in short-prompt churn,
                                // which was the bulk of the measured GPU
                                // idle); batch re-shape now waits for a
                                // natural boundary (arrival / host row).
                                died = false;
                            }
                            if died {
                                // the in-flight tick ran with the dead row(s)
                                // still batched - commit the survivors' ids,
                                // discard the rest (None guard), and let the
                                // outer loop re-shape the batch
                                match generator.decode_pipe_drain() {
                                    Ok(ids) => {
                                        for (k, plan) in prev_plans.iter().enumerate() {
                                            if matches!(plan, RowSample::Device(_)) {
                                                commit_device_token(&mut slots[k], ids[k]);
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        tracing::warn!("serve: decode_pipe_drain failed: {e}");
                                        for slot in slots.iter_mut() {
                                            if let Some(s) = slot.take() {
                                                let _ = s.events.send(TokenEvent::Error(
                                                    EngineError::from_gen(&e),
                                                ));
                                            }
                                        }
                                    }
                                }
                                break;
                            }
                        }
                        Err(e) => {
                            tracing::warn!("serve: decode_pipe_next failed: {e}");
                            for slot in slots.iter_mut() {
                                if let Some(s) = slot.take() {
                                    let _ =
                                        s.events.send(TokenEvent::Error(EngineError::from_gen(&e)));
                                }
                            }
                            break;
                        }
                    }
                }
                continue;
            }
            // tickseg: same wiring as the mixed-sampled pass above - the
            // sampled decode tick is the post-devsample hot path
            if crate::tickseg::on()
                && let Some(m) = seg_mark.take()
            {
                crate::tickseg::gap(m.elapsed());
            }
            let seg_t = std::time::Instant::now();
            let t0 = std::time::Instant::now();
            match generator.forward_batch_sampled(&tokens, &positions, &plans) {
                Ok(step) => {
                    st[2].0 += 1;
                    st[2].1 += t0.elapsed().as_nanos() as u64;
                    if crate::tickseg::on() {
                        crate::tickseg::fwd(seg_t.elapsed());
                    }
                    let seg_t = std::time::Instant::now();
                    ticks_since_admit = ticks_since_admit.saturating_add(1);
                    let mut host_rows = step.host_rows.into_iter();
                    for (k, plan) in plans.iter().enumerate() {
                        match plan {
                            RowSample::Hole => {}
                            RowSample::Device(_) => {
                                commit_device_token(&mut slots[k], step.ids[k]);
                            }
                            RowSample::Host => {
                                let (i, mut row) = host_rows.next().expect("host row");
                                debug_assert_eq!(i, k);
                                sample_slot_row(&mut slots[k], &mut row);
                            }
                        }
                    }
                    if crate::tickseg::on() {
                        crate::tickseg::smp(seg_t.elapsed());
                        seg_mark = Some(std::time::Instant::now());
                    }
                }
                Err(crate::generator::GenError::PoolExhausted) => {
                    // P5b-3 preemption on the SAMPLED decode path (G4b) - gpt-oss
                    // decodes here (device sampling), so this is the exhaustion
                    // point the unsampled handler below never sees. Preempt the
                    // NEWEST active sequence, free its KV for recompute, and let
                    // survivors proceed; the reconcile at the next tick top
                    // returns its blocks and the retried step fits. Preempting the
                    // newest keeps recompute sequences (re-admitted into low slots)
                    // off the preemption target - no livelock. A sampled step that
                    // hit PoolExhausted did so in ensure_pool_rows before any KV
                    // write (growth precedes run_layers), so no row half-advanced.
                    if let Some(k) = slots[..high_water].iter().rposition(|s| s.is_some()) {
                        let need = slots[k].as_mut().expect("present").preempt_for_recompute();
                        preempted.push(slots[k].take().expect("present"));
                        if pool_stats {
                            tracing::warn!(
                                "pool: PREEMPTED slot {k} for recompute ({need} blocks), {} queued",
                                preempted.len()
                            );
                        }
                    }
                }
                Err(e) => {
                    for slot in slots.iter_mut() {
                        if let Some(s) = slot.take() {
                            let _ = s.events.send(TokenEvent::Error(EngineError::from_gen(&e)));
                        }
                    }
                }
            }
            continue;
        }

        if crate::tickseg::on()
            && let Some(m) = seg_mark.take()
        {
            crate::tickseg::gap(m.elapsed());
        }
        let seg_t = std::time::Instant::now();
        let mut logits = match generator.forward_batch(&tokens, &positions) {
            Ok(l) => l,
            Err(crate::generator::GenError::PoolExhausted) => {
                // P5b-3 preemption: the pool couldn't grow every slot this step.
                // Preempt the NEWEST active sequence (highest slot ~ last
                // admitted) - free its KV for recompute and let the survivors
                // proceed. The reconcile at the next tick top returns its blocks;
                // the retried step then fits. Re-admission (above) resumes it.
                // Preempting the newest keeps recompute sequences (which re-admit
                // into low slots) from being re-preempted - avoiding livelock.
                if let Some(k) = slots[..high_water].iter().rposition(|s| s.is_some()) {
                    let need = slots[k].as_mut().expect("present").preempt_for_recompute();
                    preempted.push(slots[k].take().expect("present"));
                    if pool_stats {
                        tracing::warn!(
                            "pool: PREEMPTED slot {k} for recompute ({need} blocks), {} queued",
                            preempted.len()
                        );
                    }
                }
                continue;
            }
            Err(e) => {
                for slot in slots.iter_mut() {
                    if let Some(s) = slot.take() {
                        let _ = s.events.send(TokenEvent::Error(EngineError::from_gen(&e)));
                    }
                }
                continue;
            }
        };

        if crate::tickseg::on() {
            crate::tickseg::fwd(seg_t.elapsed());
        }
        let seg_t = std::time::Instant::now();
        ph_decode = tick_t0.elapsed();
        // Phase 3 - sample every row. Slots are independent, so at wide
        // batches the rows fan out over scoped threads: the tick pays
        // max-of-slots sampling latency instead of sum-of-slots. (Even the
        // rewritten sort-free sampler costs ~100 µs/row at 201k vocab -
        // serial at B=32 that's ~3 ms against a ~15 ms GPU step.)
        let live = slots[..high_water].iter().filter(|s| s.is_some()).count();
        let threads = sample_threads().min(live);
        if threads > 1 && live >= 4 {
            let chunk = high_water.div_ceil(threads);
            std::thread::scope(|sc| {
                for (sl, lg) in slots[..high_water]
                    .chunks_mut(chunk)
                    .zip(logits.chunks_mut(chunk * vocab))
                {
                    sc.spawn(move || {
                        for (s, row) in sl.iter_mut().zip(lg.chunks_mut(vocab)) {
                            sample_slot_row(s, row);
                        }
                    });
                }
            });
        } else {
            for (s, row) in slots[..high_water].iter_mut().zip(logits.chunks_mut(vocab)) {
                sample_slot_row(s, row);
            }
        }
        if crate::tickseg::on() {
            crate::tickseg::smp(seg_t.elapsed());
            seg_mark = Some(std::time::Instant::now());
        }
    }
}

/// Host sampling fan-out width for a decode tick (capped by live slots).
/// Override with PADDOCK_SAMPLE_THREADS; 1 pins the serial loop.
fn sample_threads() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_SAMPLE_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n >= 1)
            .unwrap_or(12)
    })
}

/// Commit a device-sampled token (the phase-3 body for GPU-picked rows: no
/// host logits exist - constraint/logprob slots never take this path).
fn commit_device_token(slot_opt: &mut Option<Slot>, next: u32) {
    let Some(slot) = slot_opt.as_mut() else {
        return;
    };
    // A slot with no KV behind it has no token to commit - whatever came back
    // for its row is a hole. Guarded here as well as at the plan, because the
    // plan is built in several places and this is the one door they all use.
    if !slot.prefilled {
        return;
    }
    slot.pos += 1; // the pending token is now committed at its position
    slot.draft.push(next);
    if !slot.accept(next, None) {
        *slot_opt = None;
    }
}

/// Sample + commit one slot's next token from its decode-tick logits row
/// (the phase-3 body, shared by the serial and scoped-thread paths).
fn sample_slot_row(slot_opt: &mut Option<Slot>, row: &mut [f32]) {
    let Some(slot) = slot_opt.as_mut() else {
        return;
    };
    // Same guard as `commit_device_token`: an unprefilled slot's row is a hole,
    // and sampling it would stream a token drawn from nothing.
    if !slot.prefilled {
        return;
    }
    slot.pos += 1; // the pending token is now committed at its position
    let raw = slot.logprobs.map(|_| row.to_vec());
    let next = match pick_next(
        &mut slot.sampler,
        row,
        &slot.history,
        &mut slot.constraint,
        &slot.stop_tokens,
    ) {
        Ok(t) => t,
        Err(e) => {
            let _ = slot
                .events
                .send(TokenEvent::Error(EngineError::internal(&e)));
            *slot_opt = None;
            return;
        }
    };
    let lp = slot
        .logprobs
        .map(|n| compute_logprobs(raw.as_ref().expect("snapshot"), next, n));
    // keep the drafter's history gapless even on dense ticks (a
    // greedy slot may ride spec rounds later; gapped history only
    // degrades proposals, but why degrade)
    slot.draft.push(next);
    if !slot.accept(next, lp) {
        *slot_opt = None;
    }
}

#[cfg(test)]
mod cache_admission_tests {
    use super::*;

    struct CachedPool {
        waves: Vec<usize>,
        reclaimable: usize,
    }
    impl Generator for CachedPool {
        fn reset(&mut self) {}
        fn vocab(&self) -> usize {
            2
        }
        fn forward(&mut self, _: u32) -> Result<Vec<f32>, GenError> {
            Ok(vec![0., 1.])
        }
        fn pool_free_blocks(&self) -> Option<usize> {
            Some(0)
        }
        fn pool_admission_blocks(&self) -> Option<usize> {
            Some(self.reclaimable)
        }
        fn forward_prefill_batch(
            &mut self,
            items: &[(usize, Vec<u32>)],
        ) -> Result<Vec<Vec<f32>>, GenError> {
            self.waves.push(items.len());
            Ok(items.iter().map(|_| vec![0., 1.]).collect())
        }
    }

    #[test]
    fn cache_only_pages_allow_concurrent_admission_but_active_pages_do_not() {
        for (reclaimable, expected) in [(128, vec![2]), (0, vec![1, 1])] {
            let (send, requests) = std::sync::mpsc::channel();
            let mut receivers = Vec::new();
            for _ in 0..2 {
                let (events, receiver) = tokio::sync::mpsc::unbounded_channel();
                receivers.push(receiver);
                send.send(GenRequest {
                    prompt: vec![0; 16],
                    max_tokens: 1,
                    sampler: SamplingParams::default(),
                    stop_tokens: vec![],
                    events,
                    mm_chunks: None,
                    constraint: None,
                    logprobs: None,
                    submitted: None,
                    canvas_read: None,
                })
                .unwrap();
            }
            drop(send);
            let mut generator = CachedPool {
                waves: Vec::new(),
                reclaimable,
            };
            run_batched(
                &mut generator,
                &requests,
                2,
                2,
                &Arc::new(EngineMetrics::default()),
                &ShutdownCtl::new(),
            );
            assert_eq!(generator.waves, expected);
            assert_eq!(
                generator.pool_free_blocks(),
                Some(0),
                "do not falsify occupancy"
            );
            for mut receiver in receivers {
                let mut completed = false;
                while let Ok(event) = receiver.try_recv() {
                    assert!(!matches!(event, TokenEvent::Error(_)));
                    completed |= matches!(event, TokenEvent::Done(..));
                }
                assert!(completed);
            }
        }
    }
}

#[cfg(test)]
mod diffusion_billing_tests {
    use super::*;

    struct Prefill;
    impl Generator for Prefill {
        fn reset(&mut self) {}
        fn vocab(&self) -> usize {
            2
        }
        fn forward(&mut self, _: u32) -> Result<Vec<f32>, GenError> {
            Ok(vec![0.; 2])
        }
        fn canvas_width(&self) -> usize {
            4
        }
        fn supports_mm_slots(&self) -> bool {
            true
        }
        fn forward_prefill_batch(
            &mut self,
            items: &[(usize, Vec<u32>)],
        ) -> Result<Vec<Vec<f32>>, GenError> {
            Ok(items.iter().map(|_| vec![0.; 2]).collect())
        }
        fn forward_prefill_multimodal(
            &mut self,
            _: usize,
            _: &[MmChunk],
        ) -> Result<(Vec<f32>, usize), GenError> {
            Ok((vec![0.; 2], 283))
        }
    }

    #[test]
    fn diffusion_reports_actual_rows_before_canvas_dispatch() {
        let (send, requests) = std::sync::mpsc::channel();
        let mut receivers = Vec::new();
        for image in [false, true] {
            let (events, receiver) = tokio::sync::mpsc::unbounded_channel();
            receivers.push(receiver);
            send.send(GenRequest {
                prompt: vec![0, 1, 0],
                max_tokens: 1,
                sampler: SamplingParams::default(),
                stop_tokens: vec![],
                events,
                mm_chunks: image.then(|| {
                    vec![MmChunk::Image {
                        rgb: vec![0; 3],
                        w: 1,
                        h: 1,
                    }]
                }),
                constraint: None,
                logprobs: None,
                submitted: None,
                canvas_read: None,
            })
            .unwrap();
        }
        drop(send);
        // The stub deliberately ends at canvas_open. Billing belongs to
        // successful prefill, before either chat or JEV starts its canvas.
        run_batched_diffusion(
            &mut Prefill,
            &requests,
            2,
            &Arc::new(EngineMetrics::default()),
            &ShutdownCtl::new(),
        );
        for (receiver, expected) in receivers.iter_mut().zip([3, 283]) {
            assert!(
                matches!(receiver.try_recv().unwrap(), TokenEvent::Prefilled { cached: 0, rows } if rows == expected)
            );
            assert!(matches!(receiver.try_recv().unwrap(), TokenEvent::Error(_)));
            assert!(receiver.try_recv().is_err());
        }
    }
}

#[cfg(test)]
mod reply_pin_tests {
    use super::*;

    /// Free until token 100 opens a call, constrained until 101 closes it -
    /// the shape of a tool-choice gate / dispatcher.
    struct Calls {
        in_call: bool,
    }

    impl TokenConstraint for Calls {
        fn allows(&self, _id: u32) -> bool {
            true
        }
        fn accept(&mut self, id: u32) {
            match id {
                100 => self.in_call = true,
                101 => self.in_call = false,
                _ => {}
            }
        }
        fn may_stop(&self) -> bool {
            !self.in_call
        }
        fn free_now(&self) -> bool {
            !self.in_call
        }
    }

    struct Pins(Vec<usize>);

    impl Generator for Pins {
        fn reset(&mut self) {}
        fn forward(&mut self, _token: u32) -> Result<Vec<f32>, GenError> {
            Ok(vec![0.0])
        }
        fn vocab(&self) -> usize {
            1
        }
        fn reply_pin(&mut self, slot: usize) {
            self.0.push(slot);
        }
    }

    fn slot(constraint: Option<Box<dyn TokenConstraint>>) -> Slot {
        let (events, rx) = tokio::sync::mpsc::unbounded_channel();
        std::mem::forget(rx); // keep the channel open: accept() sends on it
        Slot::new(
            GenRequest {
                prompt: vec![1],
                max_tokens: 64,
                sampler: SamplingParams::default(),
                stop_tokens: vec![2],
                events,
                mm_chunks: None,
                constraint,
                logprobs: None,
                submitted: None,
                canvas_read: None,
            },
            Arc::new(EngineMetrics::default()),
        )
    }

    /// What pick_next does, then what the tick does with the token.
    fn commit(s: &mut Slot, t: u32) {
        if let Some(c) = s.constraint.as_mut() {
            c.accept(t);
        }
        assert!(s.accept(t, None));
    }

    /// The token that opens the reply's first call makes the pin due; the
    /// drain hands it to the backend once, for that slot; a second call in
    /// the same reply does not move it.
    #[test]
    fn the_first_call_pins_the_reply_once() {
        let mut slots = vec![None, Some(slot(Some(Box::new(Calls { in_call: false }))))];
        let mut g = Pins(Vec::new());
        let s = slots[1].as_mut().unwrap();
        for t in [5, 6, 7] {
            commit(s, t);
        }
        assert!(!s.reply_pin_due, "free-phase tokens pin nothing");
        commit(s, 100);
        assert!(s.reply_pin_due);
        drain_reply_pins(&mut g, &mut slots);
        assert_eq!(g.0, [1]);
        let s = slots[1].as_mut().unwrap();
        for t in [8, 101, 9, 100, 10] {
            commit(s, t);
        }
        drain_reply_pins(&mut g, &mut slots);
        assert_eq!(g.0, [1], "one pin per reply");
    }

    /// A call opened and closed inside one sampled round is still seen: the
    /// round's walk notes the machine after every pick, not once at the end.
    #[test]
    fn a_call_inside_one_round_still_pins() {
        let mut s = slot(Some(Box::new(Calls { in_call: false })));
        for t in [100, 3, 101] {
            s.constraint.as_mut().unwrap().accept(t);
            s.note_constraint();
        }
        assert!(s.reply_pin_due);
    }

    #[test]
    fn an_unconstrained_reply_never_pins() {
        let mut slots = vec![Some(slot(None))];
        for t in [100, 3, 101] {
            commit(slots[0].as_mut().unwrap(), t);
        }
        let mut g = Pins(Vec::new());
        drain_reply_pins(&mut g, &mut slots);
        assert!(g.0.is_empty());
    }
}

#[cfg(test)]
mod usage_tests {
    use super::*;

    #[test]
    fn sampled_stop_is_counted_but_never_emitted_or_added_to_history() {
        for immediate in [false, true] {
            let (events, mut rx) = tokio::sync::mpsc::unbounded_channel();
            let metrics = Arc::new(EngineMetrics::default());
            let mut slot = Slot::new(
                GenRequest {
                    prompt: vec![7, 8],
                    max_tokens: 8,
                    sampler: SamplingParams::default(),
                    stop_tokens: vec![42],
                    events,
                    mm_chunks: None,
                    constraint: None,
                    logprobs: None,
                    submitted: None,
                    canvas_read: None,
                },
                metrics.clone(),
            );
            if !immediate {
                assert!(slot.accept(10, None));
                assert!(matches!(
                    rx.try_recv(),
                    Ok(TokenEvent::Token { id: 10, .. })
                ));
            }
            assert!(!slot.accept(42, None));
            let TokenEvent::Done(FinishReason::Stop, stats) = rx.try_recv().unwrap() else {
                panic!("missing stop")
            };
            assert_eq!(stats.terminal_tokens(), 1);
            assert_eq!(
                metrics.tokens_generated.load(Relaxed),
                if immediate { 1 } else { 2 }
            );
            assert!(!slot.history.contains(&42));
            assert!(rx.try_recv().is_err());
        }
    }

    #[test]
    fn length_finish_does_not_invent_a_terminal_token() {
        let (events, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let metrics = Arc::new(EngineMetrics::default());
        let mut slot = Slot::new(
            GenRequest {
                prompt: vec![7],
                max_tokens: 1,
                sampler: SamplingParams::default(),
                stop_tokens: vec![42],
                events,
                mm_chunks: None,
                constraint: None,
                logprobs: None,
                submitted: None,
                canvas_read: None,
            },
            metrics.clone(),
        );
        assert!(!slot.accept(10, None));
        assert!(matches!(
            rx.try_recv(),
            Ok(TokenEvent::Token { id: 10, .. })
        ));
        let TokenEvent::Done(FinishReason::Length, stats) = rx.try_recv().unwrap() else {
            panic!("missing length finish")
        };
        assert_eq!(stats.terminal_tokens(), 0);
        assert_eq!(metrics.tokens_generated.load(Relaxed), 1);
    }

    struct DeferringSpec {
        deferred: bool,
        cap: Option<usize>,
    }
    impl Generator for DeferringSpec {
        fn reset(&mut self) {}
        fn forward(&mut self, _: u32) -> Result<Vec<f32>, GenError> {
            panic!("policy test performs no inference")
        }
        fn vocab(&self) -> usize {
            2
        }
        fn spec_live_cap(&self) -> usize {
            4
        }
        fn spec_fixed_draft_depth(&self) -> Option<usize> {
            Some(15)
        }
        fn spec_deferred(&self) -> bool {
            self.deferred
        }
        fn spec_batch_draft_budget(&self, live: usize) -> Option<usize> {
            (live == 4).then_some(self.cap).flatten()
        }
    }

    #[test]
    fn backend_spec_budget_defers_and_resumes_without_changing_capacity() {
        let mut g = DeferringSpec {
            deferred: false,
            cap: None,
        };
        assert_eq!(backend_spec_k_budget(&g, 0), 0);
        for live in 1..=4 {
            assert_eq!(backend_spec_k_budget(&g, live), 15);
        }
        assert_eq!(backend_spec_k_budget(&g, 5), 0);
        g.deferred = true;
        for live in 0..=5 {
            assert_eq!(backend_spec_k_budget(&g, live), 0);
        }
        assert_eq!(g.spec_live_cap(), 4);
        g.deferred = false;
        assert_eq!(backend_spec_k_budget(&g, 3), 15);
        g.cap = Some(7);
        assert_eq!(backend_spec_k_budget(&g, 4), 7);
        assert_eq!(backend_spec_k_budget(&g, 3), 15);
        g.cap = Some(31);
        assert_eq!(backend_spec_k_budget(&g, 4), 15);
        g.cap = Some(0);
        assert_eq!(backend_spec_k_budget(&g, 4), 0);
        g.cap = Some(7);
        g.deferred = true;
        assert_eq!(backend_spec_k_budget(&g, 4), 0);
    }

    #[test]
    fn speculation_waits_for_encoder_and_prefill_publication() {
        let make = || {
            let (events, _rx) = tokio::sync::mpsc::unbounded_channel();
            Slot::new(
                GenRequest {
                    prompt: vec![1, 2],
                    max_tokens: 8,
                    sampler: SamplingParams::default(),
                    stop_tokens: Vec::new(),
                    events,
                    mm_chunks: None,
                    constraint: None,
                    logprobs: None,
                    submitted: None,
                    canvas_read: None,
                },
                Arc::new(EngineMetrics::default()),
            )
        };
        let mut slots = vec![Some(make()), Some(make()), None, Some(make())];
        let mut chunking = std::collections::HashSet::new();
        // Encoder has taken mm, but is not yet a chunked prompt. This exact
        // state incorrectly entered target verification at KV position zero.
        assert!(spec_live_slots(&slots, &chunking).is_empty());
        slots[1].as_mut().unwrap().prefilled = true;
        assert_eq!(spec_live_slots(&slots, &chunking), [1]);
        chunking.insert(0);
        assert_eq!(spec_live_slots(&slots, &chunking), [1]);
        // Publish an image prompt: real KV cursor can exceed token history.
        let s = slots[0].as_mut().unwrap();
        s.prefilled = true;
        s.pos = 576;
        assert_eq!(spec_live_slots(&slots, &chunking), [1]);
        chunking.remove(&0);
        assert_eq!(spec_live_slots(&slots, &chunking), [0, 1]);
    }

    /// A generator whose multimodal prefill runs a row count deliberately
    /// unrelated to the prompt's token count - which is what a picture does:
    /// one `<image>` chunk becomes hundreds or thousands of rows.
    struct Stub {
        mm_rows: usize,
    }

    impl Generator for Stub {
        fn reset(&mut self) {}
        fn forward(&mut self, _token: u32) -> Result<Vec<f32>, GenError> {
            Ok(vec![0.0, 1.0])
        }
        fn vocab(&self) -> usize {
            2
        }
        fn forward_prefill_stream(&mut self, _tokens: &[u32]) -> Result<Vec<f32>, GenError> {
            Ok(vec![0.0, 1.0])
        }
        fn forward_multimodal(
            &mut self,
            _chunks: &[MmChunk],
        ) -> Result<Option<(Vec<f32>, usize)>, GenError> {
            Ok(Some((vec![0.0, 1.0], self.mm_rows)))
        }
    }

    /// Run one request through the serial loop and return the `rows` its
    /// `Prefilled` event carried (None = it never sent one).
    fn serial_prefilled_rows(mm_rows: usize, prompt: Vec<u32>, mm: bool) -> Option<u32> {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let metrics = EngineMetrics::default();
        let mut g = Stub { mm_rows };
        run_request(
            &mut g,
            GenRequest {
                prompt,
                max_tokens: 1,
                sampler: SamplingParams::default(),
                stop_tokens: Vec::new(),
                events: tx,
                mm_chunks: mm.then(|| {
                    vec![MmChunk::Image {
                        rgb: Vec::new(),
                        w: 0,
                        h: 0,
                    }]
                }),
                constraint: None,
                logprobs: None,
                submitted: None,
                canvas_read: None,
            },
            &metrics,
        );
        while let Ok(ev) = rx.try_recv() {
            if let TokenEvent::Prefilled { rows, .. } = ev {
                return Some(rows);
            }
        }
        None
    }

    /// A serial-engine image request must bill the rows the engine
    /// prefilled, not the text it was tokenized from. The relationship - not a
    /// magic number - is what this pins: whatever `forward_multimodal` reports
    /// having run is what reaches the client.
    #[test]
    fn a_serial_image_request_bills_the_rows_the_engine_prefilled() {
        // 3 text tokens around a picture that expands to 1471 rows (the
        // qwen35 measurement from the task); the client must see 1471.
        assert_eq!(serial_prefilled_rows(1471, vec![7, 8, 9], true), Some(1471));
        // and it tracks the engine, so a different picture reports differently
        assert_eq!(serial_prefilled_rows(283, vec![7, 8, 9], true), Some(283));
    }

    /// The text lane keeps reporting its token count - the same number the
    /// caller already tokenized, so adding the event changed nothing there.
    #[test]
    fn a_serial_text_request_bills_its_prompt_tokens() {
        assert_eq!(
            serial_prefilled_rows(0, vec![1, 2, 3, 4, 5], false),
            Some(5)
        );
    }
}

#[cfg(test)]
mod serial_pipe_tests {
    use super::*;
    use std::collections::VecDeque;

    /// Pipe-capable stub: scripted tick ids, strict depth accounting. Any
    /// `forward` call while the pipe works is a bug (the branch must never
    /// mix paths), so it panics.
    struct PipeStub {
        script: Vec<u32>,
        enqueued: usize,
        inflight: VecDeque<u32>,
        begin_positions: Vec<u32>,
        next_calls: usize,
        drain_calls: usize,
        fail_begin: bool,
        forward_calls: usize,
    }

    impl PipeStub {
        fn new(script: Vec<u32>, fail_begin: bool) -> Self {
            Self {
                script,
                enqueued: 0,
                inflight: VecDeque::new(),
                begin_positions: Vec::new(),
                next_calls: 0,
                drain_calls: 0,
                fail_begin,
                forward_calls: 0,
            }
        }
    }

    impl Generator for PipeStub {
        fn reset(&mut self) {}
        fn forward(&mut self, token: u32) -> Result<Vec<f32>, GenError> {
            if !self.fail_begin {
                panic!("host forward during an available pipe");
            }
            self.forward_calls += 1;
            // argmax follows the fed token so the fallback loop is observable:
            // token 1 -> [1, 0] -> argmax 0; token 0 -> [0, 1] -> argmax 1
            Ok(if token == 1 {
                vec![1.0, 0.0]
            } else {
                vec![0.0, 1.0]
            })
        }
        fn vocab(&self) -> usize {
            2
        }
        fn forward_prefill_stream(&mut self, _tokens: &[u32]) -> Result<Vec<f32>, GenError> {
            Ok(vec![0.0, 1.0]) // token 0 = argmax = 1
        }
        fn supports_decode_pipe(&self) -> bool {
            true
        }
        fn decode_pipe_begin(
            &mut self,
            tokens: &[u32],
            positions: &[u32],
            plans: &[RowSample],
        ) -> Result<(), GenError> {
            if self.fail_begin {
                return Err(GenError::Backend("no batch state".into()));
            }
            assert_eq!((tokens.len(), positions.len(), plans.len()), (1, 1, 1));
            assert!(matches!(plans[0], RowSample::Device(_)));
            self.begin_positions.push(positions[0]);
            self.inflight.push_back(self.script[self.enqueued]);
            self.enqueued += 1;
            Ok(())
        }
        fn decode_pipe_next(&mut self, plans: &[RowSample]) -> Result<Vec<u32>, GenError> {
            assert_eq!(plans.len(), 1);
            assert!(matches!(plans[0], RowSample::Device(_)));
            self.next_calls += 1;
            self.inflight.push_back(self.script[self.enqueued]);
            self.enqueued += 1;
            Ok(vec![self.inflight.pop_front().expect("in-flight tick")])
        }
        fn decode_pipe_drain(&mut self) -> Result<Vec<u32>, GenError> {
            self.drain_calls += 1;
            assert_eq!(
                self.inflight.len(),
                1,
                "drain expects exactly one in-flight tick"
            );
            Ok(vec![self.inflight.pop_front().expect("in-flight tick")])
        }
    }

    fn run(
        g: &mut PipeStub,
        max_tokens: usize,
        stop: Vec<u32>,
    ) -> (Vec<u32>, Option<FinishReason>) {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let metrics = EngineMetrics::default();
        run_request(
            g,
            GenRequest {
                prompt: vec![7, 8, 9],
                max_tokens,
                sampler: SamplingParams::default(), // greedy => device-plannable
                stop_tokens: stop,
                events: tx,
                mm_chunks: None,
                constraint: None,
                logprobs: None,
                submitted: None,
                canvas_read: None,
            },
            &metrics,
        );
        let mut toks = Vec::new();
        let mut fin = None;
        let mut terminal_tokens = 0;
        while let Ok(ev) = rx.try_recv() {
            match ev {
                TokenEvent::Token { id, .. } => toks.push(id),
                TokenEvent::Done(r, stats) => {
                    fin = Some(r);
                    terminal_tokens = stats.terminal_tokens();
                }
                TokenEvent::Error(e) => panic!("engine error: {e:?}"),
                _ => {}
            }
        }
        assert_eq!(
            metrics.tokens_generated.load(Relaxed) as usize,
            toks.len() + terminal_tokens
        );
        assert_eq!(
            terminal_tokens,
            usize::from(fin == Some(FinishReason::Stop))
        );
        (toks, fin)
    }

    /// Length finish: token 0 off the prefill logits, then exactly `want`
    /// ticks - the last collected by drain, none wasted past it.
    #[test]
    fn pipe_runs_exactly_want_ticks_and_finishes_length() {
        let mut g = PipeStub::new(vec![10, 11, 12, 13, 99], false);
        let (toks, fin) = run(&mut g, 5, vec![]);
        assert_eq!(toks, vec![1, 10, 11, 12, 13]);
        assert_eq!(fin, Some(FinishReason::Length));
        assert_eq!(g.begin_positions, vec![3]); // = prompt rows
        assert_eq!(g.enqueued, 4); // want = max_tokens - 1
        assert_eq!((g.next_calls, g.drain_calls), (3, 1));
        assert!(g.inflight.is_empty(), "nothing left in flight");
    }

    /// A stop id lands one tick late by construction - the already-enqueued
    /// overshoot tick must be drained and discarded, not emitted.
    #[test]
    fn pipe_stop_drains_the_overshoot_tick() {
        let mut g = PipeStub::new(vec![10, 11, 12, 13, 99], false);
        let (toks, fin) = run(&mut g, 5, vec![11]);
        assert_eq!(toks, vec![1, 10]); // 11 is the stop, never emitted
        assert_eq!(fin, Some(FinishReason::Stop));
        assert_eq!(g.enqueued, 3); // begin + 2 nexts when the stop surfaced
        assert_eq!(g.drain_calls, 1); // the discarded overshoot tick
        assert!(g.inflight.is_empty());
    }

    /// A backend that reports the capability but can't begin (serial VRAM
    /// fallback without its batched state) must finish on the host loop.
    #[test]
    fn pipe_begin_failure_falls_back_to_the_host_loop() {
        let mut g = PipeStub::new(vec![], true);
        let (toks, fin) = run(&mut g, 3, vec![]);
        // token 0 = 1 (prefill logits), then forward(1) -> 0, forward(0) -> 1
        assert_eq!(toks, vec![1, 0, 1]);
        assert_eq!(fin, Some(FinishReason::Length));
        assert_eq!(g.forward_calls, 2); // no trailing wasted forward
    }

    #[test]
    fn immediate_and_fallback_stops_are_counted_without_lookahead() {
        let mut g = PipeStub::new(vec![10], false);
        assert_eq!(run(&mut g, 5, vec![1]), (vec![], Some(FinishReason::Stop)));
        assert_eq!(g.enqueued, 0);
        let mut g = PipeStub::new(vec![], true);
        assert_eq!(run(&mut g, 5, vec![0]), (vec![1], Some(FinishReason::Stop)));
        assert_eq!(g.forward_calls, 1);
    }
}

#[cfg(test)]
mod error_class_tests {
    use super::*;

    /// A CUDA out-of-memory - whatever forward pass raised it - must never reach
    /// the client as a raw 500 `api_error`. `internal()` reclassifies it to a
    /// retryable capacity error naming the levers the caller controls, and does
    /// not leak the raw driver text.
    #[test]
    fn cuda_oom_is_reclassified_as_a_graceful_capacity_error() {
        let e = EngineError::internal(
            "generation failed: CUDA driver call failed: \
             DriverError(CUDA_ERROR_OUT_OF_MEMORY, \"out of memory\")",
        );
        assert_eq!(
            e.class,
            ErrorClass::Overloaded,
            "an OOM is capacity, not our 500"
        );
        assert_eq!(e.code, Some("insufficient_memory"));
        assert!(
            e.message.contains("fewer or smaller pages"),
            "names the caller's lever: {}",
            e.message
        );
        assert!(
            !e.message.to_uppercase().contains("CUDA"),
            "raw driver text not leaked: {}",
            e.message
        );
    }

    /// Everything else stays a genuine internal fault (-> 500), verbatim.
    #[test]
    fn a_normal_engine_fault_stays_internal() {
        let e = EngineError::internal("tensor shape mismatch in layer 3");
        assert_eq!(e.class, ErrorClass::Internal);
        assert!(e.code.is_none());
        assert_eq!(e.message, "tensor shape mismatch in layer 3");
    }

    /// The PRIMARY (typed) path: a `GenError::OutOfMemory` - classified by the
    /// driver's numeric result code back at the boundary, not by text - maps to
    /// the same graceful capacity error through `from_gen`, with no string
    /// matching involved.
    #[test]
    fn typed_gen_oom_maps_to_capacity_error() {
        let e = EngineError::from_gen(&crate::generator::GenError::OutOfMemory);
        assert_eq!(e.class, ErrorClass::Overloaded);
        assert_eq!(e.code, Some("insufficient_memory"));
        assert!(
            e.message.contains("fewer or smaller pages"),
            "{}",
            e.message
        );
    }

    /// The FALLBACK path through `from_gen`: an OOM that stayed untyped (a
    /// `Backend(_)` still carrying the driver's rendered signature - e.g. a
    /// cuBLAS alloc) is still caught, via `internal`'s text match.
    #[test]
    fn untyped_backend_oom_still_caught_via_from_gen() {
        let e = EngineError::from_gen(&crate::generator::GenError::Backend(
            "cublas alloc: DriverError(CUDA_ERROR_OUT_OF_MEMORY, \"out of memory\")".into(),
        ));
        assert_eq!(
            e.class,
            ErrorClass::Overloaded,
            "untyped OOM still reclassified"
        );
        assert_eq!(e.code, Some("insufficient_memory"));
    }

    /// A non-OOM backend fault routed through `from_gen` stays a 500.
    #[test]
    fn typed_backend_fault_stays_internal() {
        let e = EngineError::from_gen(&crate::generator::GenError::Backend(
            "illegal address in attention kernel".into(),
        ));
        assert_eq!(e.class, ErrorClass::Internal);
    }

    /// The boundary classifier reads the driver's numeric result code: an
    /// out-of-memory `DriverError` becomes the typed `GpuError::OutOfMemory`,
    /// and any other code keeps its rendered text under `Driver`.
    #[test]
    #[cfg(feature = "cuda")]
    fn from_driver_classifies_oom_by_code() {
        use cudarc::driver::{DriverError, sys::CUresult};
        let oom = crate::gpu::from_driver(DriverError(CUresult::CUDA_ERROR_OUT_OF_MEMORY));
        assert!(
            matches!(oom, crate::gpu::GpuError::OutOfMemory),
            "code 2 -> typed OOM"
        );
        // Rendering any other code goes through cuGetErrorString, which needs
        // the driver library. The OOM arm above is decided by the code alone,
        // so that half holds on a box without a GPU; only this half skips.
        if !crate::cuda::driver_present() {
            eprintln!("SKIP: no CUDA driver here, the Driver(text) arm needs cuGetErrorString");
            return;
        }
        let other = crate::gpu::from_driver(DriverError(CUresult::CUDA_ERROR_INVALID_VALUE));
        assert!(
            matches!(other, crate::gpu::GpuError::Driver(_)),
            "other codes stay Driver(text)"
        );
    }
}
