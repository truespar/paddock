//! Time-budgeted mixed ticks: adaptive prefill chunking at long context.
//!
//! A mixed tick runs prefill rows and the live decode rows ("riders") in one
//! call, and every rider waits the whole tick for its next token. The tick's
//! budget was a ROW count, and at shallow depth rows are what a tick costs.
//! At long context they are not: each prefill row attends over every token
//! before it, so on the full-attention layers a chunk's cost grows with its
//! depth. Measured on the DGX Spark (GB10, qwen3.8-27b Q8_0, 2026-09-23): a
//! prefilled token cost ~0.63 ms plus ~0.0128 ms per 1K tokens of depth, so the
//! same 1024-row chunk that is a ~0.65 s tick at the start of a prompt is a
//! ~3 s tick at 176K - and a decoding session sharing the GPU got one token per
//! three seconds while another session ingested a long tool result.
//!
//! The fix is the one the literature converged on: budget the tick in TIME,
//! not rows, with a cost model that knows depth, and fill it decodes-first.
//! Medha (Agrawal et al., arXiv 2409.17264) packs every decode, then binary-
//! searches the largest prefill chunk its runtime predictor says fits a fixed
//! time target; SGLang's `--enable-dynamic-chunking` solves a fitted latency
//! model for the chunk size each step; the 2026 deadline-aware chunking work
//! refines the same shape. Audited: the Medha and Sarathi-Serve papers.
//!
//! WHERE it lives: in the scheduler, once, for every model family - the way
//! vLLM V1's scheduler owns the per-step token budget and backends only
//! execute it. The row budget the scheduler already hands every backend
//! (`forward_mixed*`, `forward_unified_sampled`, the spec-in-mixed calls) is
//! what carries the decision; a backend contributes only a view of its
//! chunked-prefill queue (`Generator::prefill_queue`: per prompt, its KV depth
//! and rows remaining, in spending order) and its own per-tick row ceiling
//! (`Generator::prefill_tick_cap`). The scheduler replays the backend's FIFO
//! spending over that queue to size the budget ([`TickPacer::plan`]), and
//! learns what the tick really executed from the queue before and after it
//! ([`executed`]) - so no backend reports timings or shapes of its own. The
//! scheduler keeps one pacer per tick KIND (fused unified tick, prefill-then-
//! decode mixed tick, spec-in-mixed tick): their fixed costs differ, and one
//! fit across them would describe none.
//!
//! What this module deliberately does NOT change: the shallow tick. The time
//! target is the fitted cost of a FULL-SIZE tick at depth zero - the tick
//! this engine already runs at the start of every prompt - so a shallow chunk
//! is granted its full rows by construction, and every measured span-size
//! election (the decode-stall attribution of 2026-07-27: shrinking
//! spans at short context LOST, because a pass must amortize its weight
//! stream) stands untouched. What changes is only the deep tick: it is held to
//! the shallow tick's wall time instead of growing with depth. At depth the
//! trade is cheap - attention cost scales with rows and carries no weights, so
//! a smaller chunk costs little per token there.
//!
//! The model is fitted in-run from ticks this process actually timed, never
//! from a cross-shape fit or a per-hardware constant (the trap the attribution
//! doc records: a "~27 ms fixed cost" that was a linear fit across shapes).
//! It is used only once its per-row and depth coefficients are statistically
//! established.
//!
//! Before that, a fresh process has a bootstrap problem the offline-profiled
//! systems do not (Medha profiles with Vidur; SGLang's dynamic chunking
//! profiles at startup): the first long ingest is dozens of ticks of ONE shape
//! (the row cap plus the riders), and with one row count the per-pass and
//! per-row costs are inseparable, so the fit cannot establish itself on
//! exactly the traffic it exists for (measured: a fresh GB10 serve never
//! paced a 150K ingest). So the bootstrap is ACTIVE: while riders wait on a
//! deep tick whose shape's own measured wall has grown past the hysteresis
//! band over that shape's shallowest wall - the band's criterion, read
//! straight off the clock with no model - the tick shrinks, alternating
//! between a half and a quarter of its rows. That is the right move for the
//! riders anyway, and it is the excitation the fit needs: two row counts
//! interleaved at the SAME depths, so the per-row term separates from the
//! depth term in a handful of ticks (a single halved size moves rows and depth
//! together - measured: 243 bootstrap ticks, ~124K tokens, before the fit
//! could establish). The model then takes over. No one waiting, or no measured
//! growth, and the row budget stays exactly as before.
//!
//! The KNEE is the one place a shallow tick is trimmed: while a few streams
//! decode beside an ingest. The span elections behind the full tick were
//! measured for throughput at width, and at one or two streams they leave
//! each one a token round per ~1 s (GB10, 1024 rows; a 188-SM die's 4096-row
//! tick likewise) for the whole ingest. Sarathi-Serve's stall-free batching
//! (OSDI '24) sizes the prefill share that rides beside the decodes as the
//! SMALLEST that still runs efficiently - below that, the per-pass weight
//! stream dominates and the prompt pays for the streams' latency several
//! times over. So with 1..=[`KNEE_STREAMS`] streams waiting, a tick takes the
//! smallest aligned share whose cost per prompt token stays within
//! [`knee::KNEE_COST`] of the full share's - MEASURED by this process on
//! its own ticks ([`knee`]), so the same rule lands where each machine's
//! knee is (GB10: 512 rows, +12% ingest for 1.75x the stream's tokens; a
//! die with a cheap weight stream lands at a fraction of its ~1 s tick).
//! Wider batches keep the elected spans: their throughput elections were
//! measured with every tick shared.

use std::time::{Duration, Instant};

use crate::generator::{GenError, Generator};

mod knee;

/// Rows are granted in multiples of this: the gated-DeltaNet chunk size and a
/// GEMM M-tile multiple (a 257-row chunk costs a 320-row pass).
pub const ROW_ALIGN: usize = 64;

/// The fit's memory is STRATIFIED by tick shape: at most this many recent
/// observations per (row bucket, attention-work bucket) cell. A plain recency
/// window would be the wrong memory here - a long prefill runs dozens of
/// identical 1024-row ticks, which would evict every other row count, and with
/// one row count the per-pass cost and the per-row cost are collinear (the fit
/// cannot tell them apart). Per-cell recency keeps the shapes the process has
/// seen while still tracking a changed regime cell by cell.
const PER_CELL: usize = 8;
/// A tick predicted within this fraction of the shallow tick is left alone.
/// Only once the depth term clearly dominates is a chunk trimmed - and then
/// back to the shallow tick's time, not to the edge of the band. Without it
/// every chunk past the first of any prompt would lose a few rows to a depth
/// term that is noise-level there (and float error trims even depth zero),
/// which is exactly the shallow span shrink the attribution doc measured as
/// a loss. On GB10 with qwen3.8-27b the band leaves prompts up to ~14K tokens
/// untouched - every short-context board cell among them.
const HYSTERESIS: f64 = 0.25;
/// Pacing is a long-context mechanism, and this is where long context starts.
/// Below it attention is a minor share of a row's cost for every model we
/// serve (under ~15 % even for the small ones, inside the hysteresis band),
/// so three things key on it rather than on a fit:
/// - a tick whose first share starts shallower is NEVER trimmed, whatever a
///   model says - the hard form of "shallow ticks are untouched";
/// - the model is trusted only once it has timed ticks this deep - a depth
///   term fitted on short traffic alone is extrapolated noise (measured: a
///   fresh serve on 4K prompts only "established" a model that trimmed a
///   depth-3072 share 1024 -> 640);
/// - the bootstrap acts only on ticks this deep - its growth signal is a
///   wall-clock comparison, and on short traffic one slow tick (a host
///   hiccup) could clear the band.
const PACE_MIN_DEPTH: usize = 8192;
/// Deep ticks the model must have timed before it is trusted.
const MIN_DEEP_OBS: usize = 4;
/// The knee applies while at most this many streams decode beside an ingest:
/// one or two interactive sessions and a small agent fan-out. From here up
/// the board cells' measured span elections (every tick shared, throughput
/// at width) stand.
pub const KNEE_STREAMS: usize = 4;
/// Hard bound on the whole store (oldest dropped first).
const MAX_OBS: usize = 1024;
/// Observations before a fit is attempted.
const MIN_OBS: usize = 12;
/// Both the per-row and the depth coefficient must be at least this many
/// standard errors from zero before the model is trusted to shrink anything:
/// a share's allowance depends on both.
const MIN_T_STAT: f64 = 4.0;

/// Attention work of a prefill share: the KV positions its rows attend over,
/// summed (row i of a share starting at `depth` sees depth + i positions), in
/// millions of row-positions so the fit stays well scaled.
pub fn attn_work(depth: usize, rows: usize) -> f64 {
    rows as f64 * (depth as f64 + rows as f64 / 2.0) / 1e6
}

/// A backend's chunked-prefill queue as `Generator::prefill_queue` reports
/// it: `(slot, depth, remaining)` per prompt, in spending order.
pub type PrefillQueue = [(usize, usize, usize)];

/// What a tick's prefill actually did.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TickShape {
    /// prefill rows executed (riders excluded)
    pub prefill_rows: usize,
    /// their attention work, see [`attn_work`]
    pub attn: f64,
    /// start depth of the deepest share that ran
    pub deepest: usize,
}

/// Read a tick's executed prefill off the queue before and after it: a
/// prompt still queued advanced by its change in depth, one that left the
/// queue finished its remainder. The backend's own decisions - checkpoint
/// cuts, tail absorbs, whole-prompt joins - are all in it, so the fit trains
/// on what ran, not on what was planned.
pub fn executed(before: &PrefillQueue, after: &PrefillQueue) -> TickShape {
    let mut s = TickShape::default();
    for &(slot, depth, remaining) in before {
        let take = match after.iter().find(|&&(k, _, _)| k == slot) {
            Some(&(_, d2, _)) => d2.saturating_sub(depth).min(remaining),
            None => remaining,
        };
        if take > 0 {
            s.prefill_rows += take;
            s.attn += attn_work(depth, take);
            s.deepest = s.deepest.max(depth);
        }
    }
    s
}

/// Fitted wall time of a tick: `c0 + c1 * rows + c2 * attn_work` (ms).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CostModel {
    pub c0: f64,
    pub c1: f64,
    pub c2: f64,
}

impl CostModel {
    pub fn predict_ms(&self, rows: usize, attn: f64) -> f64 {
        self.c0 + self.c1 * rows as f64 + self.c2 * attn
    }

    /// Largest row count a share at `depth` can take within `left_ms` of
    /// marginal time: the positive root of
    /// `c2/2e6 * r^2 + (c1 + c2 * depth / 1e6) * r - left_ms = 0`.
    fn rows_within(&self, depth: usize, left_ms: f64) -> usize {
        if left_ms <= 0.0 {
            return 0;
        }
        let a = self.c2 / 2e6;
        let b = self.c1 + self.c2 * depth as f64 / 1e6;
        let r = if a > 0.0 {
            (-b + (b * b + 4.0 * a * left_ms).sqrt()) / (2.0 * a)
        } else {
            left_ms / b
        };
        if r.is_finite() && r > 0.0 {
            r as usize
        } else {
            0
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Obs {
    rows: f64,
    attn: f64,
    wall_ms: f64,
    /// start of the tick's deepest prefill share
    depth: usize,
    cell: (u32, u32),
    seq: u64,
}

/// The stratification cell of a tick: rows in ROW_ALIGN buckets, attention
/// work on a log2 scale (in quarter-million row-positions), so a sweep
/// through depth lands in a new cell every doubling.
fn cell(rows: usize, attn: f64) -> (u32, u32) {
    let a = (attn * 4.0 + 1.0).log2().max(0.0) as u32;
    ((rows / ROW_ALIGN) as u32, a)
}

/// A paced tick's decision, for the caller to apply and log.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plan {
    /// the prefill row budget to hand the backend
    pub budget: usize,
    /// the tick's time target (ms); None on a bootstrap tick (no model yet)
    pub target_ms: Option<f64>,
    /// the knee capped it (streams waiting), not the depth pacing
    pub knee: bool,
}

/// One tick kind's pacer: records timed ticks, keeps the fit, sizes budgets.
#[derive(Debug, Default)]
pub struct TickPacer {
    obs: Vec<Obs>,
    seq: u64,
    model: Option<CostModel>,
    /// The knee search beside a few waiting streams (see `knee.rs`).
    knee: knee::KneeSearch,
    /// Per row bucket: (shallowest wall seen, latest wall, latest deepest
    /// share start) - the model-free growth signal the bootstrap reads.
    shape_walls: std::collections::HashMap<u32, (f64, f64, usize)>,
}

impl TickPacer {
    pub fn new() -> Self {
        Self::default()
    }

    /// The trusted model, if the fit has established its terms.
    pub fn model(&self) -> Option<CostModel> {
        self.model
    }

    /// A tick of `rows` total rows (riders + prefill) with `attn` prefill
    /// attention work, its deepest share starting at `depth`, took `wall`.
    /// Record only ticks that owned the GPU: a span overlapped with the
    /// decode lane measures contention, not its own cost.
    pub fn record(&mut self, rows: usize, attn: f64, depth: usize, wall: Duration) {
        let wall_ms = wall.as_secs_f64() * 1e3;
        let c = cell(rows, attn);
        let w = self
            .shape_walls
            .entry(c.0)
            .or_insert((wall_ms, wall_ms, depth));
        w.0 = w.0.min(wall_ms);
        w.1 = wall_ms;
        w.2 = depth;
        self.seq += 1;
        self.obs.push(Obs {
            rows: rows as f64,
            attn,
            wall_ms,
            depth,
            cell: c,
            seq: self.seq,
        });
        if self.obs.iter().filter(|o| o.cell == c).count() > PER_CELL {
            let oldest = self
                .obs
                .iter()
                .enumerate()
                .filter(|(_, o)| o.cell == c)
                .min_by_key(|(_, o)| o.seq)
                .map(|(i, _)| i);
            if let Some(i) = oldest {
                self.obs.swap_remove(i);
            }
        }
        if self.obs.len() > MAX_OBS
            && let Some(i) = (0..self.obs.len()).min_by_key(|&i| self.obs[i].seq)
        {
            self.obs.swap_remove(i);
        }
        self.model = fit(&self.obs);
    }

    /// A knee tick (streams waiting, a long head) against the full share
    /// `full` ran `rows` prefill rows in `wall`: the knee search's sample.
    pub fn record_knee(&mut self, full: usize, rows: usize, wall: Duration) {
        self.knee.observe(full, rows, wall.as_secs_f64() * 1e3);
    }

    /// Whether a tick against `full_rows` with this queue is a knee tick.
    pub fn knee_tick(waiting: usize, full_rows: usize, queue: &PrefillQueue) -> bool {
        (1..=KNEE_STREAMS).contains(&waiting)
            && full_rows > ROW_ALIGN
            && queue.first().is_some_and(|&(_, _, rem)| rem >= full_rows)
    }

    /// Size a tick. `waiting`: decode rows that wait on this tick's end (the
    /// caller's question: riders of a fused or prefill-then-decode tick, the
    /// round of a spec-in-mixed tick - but never a route-B overlapped span,
    /// whose riders the decode lane serves meanwhile). `pass_riders`: rows the
    /// call spends on them. `full_rows`: the prefill rows an UNPACED tick
    /// would take (the scheduler's budget under the backend's own ceiling).
    /// `queue`: the backend's queue in spending order.
    ///
    /// The backend spends a budget FIFO and gives a prompt it cannot finish
    /// everything left, so the plan replays exactly that: each prompt gets
    /// what its own depth allows, and the replay stops at the first prompt
    /// that does not finish - a single number then expresses every share.
    /// None keeps the caller's budget unchanged (no one waiting, nothing to
    /// trim, no model and no bootstrap signal).
    pub fn plan(
        &self,
        waiting: usize,
        pass_riders: usize,
        full_rows: usize,
        queue: &PrefillQueue,
    ) -> Option<Plan> {
        if waiting == 0 || full_rows == 0 || queue.is_empty() {
            return None;
        }
        let paced = self.depth_plan(pass_riders, full_rows, queue);
        let knee = if Self::knee_tick(waiting, full_rows, queue) {
            self.knee.size(full_rows).map(|budget| Plan {
                budget,
                target_ms: None,
                knee: true,
            })
        } else {
            None
        };
        match (paced, knee) {
            (Some(p), Some(k)) if k.budget < p.budget => Some(k),
            (p @ Some(_), _) => p,
            (None, k) => k,
        }
    }

    /// The depth pacing proper (see the module docs): None when nothing
    /// would shrink.
    fn depth_plan(
        &self,
        pass_riders: usize,
        full_rows: usize,
        queue: &PrefillQueue,
    ) -> Option<Plan> {
        if let Some(mut tb) = self.budget(pass_riders, full_rows) {
            let mut room = full_rows;
            let mut total = 0;
            for &(_, depth, remaining) in queue {
                if room == 0 {
                    break;
                }
                let allow = tb.rows(depth, room);
                if allow == 0 {
                    break;
                }
                let take = remaining.min(allow);
                tb.charge(depth, take);
                total += take;
                room -= take;
                if take < remaining {
                    break;
                }
            }
            let total = total.max(1);
            return (total < full_rows).then_some(Plan {
                budget: total,
                target_ms: Some(tb.target_ms()),
                knee: false,
            });
        }
        // no model yet: the bootstrap, only for a tick that opens deep
        let &(_, head_depth, _) = queue.first()?;
        if head_depth < PACE_MIN_DEPTH {
            return None;
        }
        self.bootstrap_rows(pass_riders, full_rows)
            .map(|budget| Plan {
                budget,
                target_ms: None,
                knee: false,
            })
    }

    /// Bootstrap row cap for a tick with no trusted model yet, when this tick
    /// shape's latest measured wall has grown past the hysteresis band over
    /// its shallowest: alternately a half and a quarter of `full_rows`
    /// (aligned), so the fit sees two row counts at the same depths. None
    /// otherwise (the caller keeps its row budget).
    pub fn bootstrap_rows(&self, pass_riders: usize, full_rows: usize) -> Option<usize> {
        if self.model.is_some() || full_rows <= ROW_ALIGN {
            return None;
        }
        let &(shallowest, latest, latest_depth) = self
            .shape_walls
            .get(&(((pass_riders + full_rows) / ROW_ALIGN) as u32))?;
        if latest_depth < PACE_MIN_DEPTH || latest < shallowest * (1.0 + HYSTERESIS) {
            return None;
        }
        let div = if self.seq.is_multiple_of(2) { 2 } else { 4 };
        Some((full_rows / div / ROW_ALIGN * ROW_ALIGN).max(ROW_ALIGN))
    }

    /// The time budget for a tick whose call spends `pass_riders` rows on its
    /// riders and whose prefill rows are capped at `full_rows`: the fitted
    /// cost of exactly that tick at depth zero. None without a trusted model.
    pub fn budget(&self, pass_riders: usize, full_rows: usize) -> Option<TickBudget> {
        if full_rows == 0 {
            return None;
        }
        let m = self.model?;
        // the tick this engine already runs at the start of a prompt
        let target = m.predict_ms(pass_riders + full_rows, attn_work(0, full_rows));
        Some(TickBudget {
            model: m,
            target_ms: target,
            fixed_ms: m.c0 + m.c1 * pass_riders as f64,
            spent_ms: 0.0,
            granted_any: false,
        })
    }
}

/// One tick's time budget, spent share by share.
#[derive(Clone, Copy, Debug)]
pub struct TickBudget {
    model: CostModel,
    target_ms: f64,
    /// the pass itself plus the riders' rows
    fixed_ms: f64,
    /// marginal cost of the shares granted so far
    spent_ms: f64,
    granted_any: bool,
}

impl TickBudget {
    pub fn target_ms(&self) -> f64 {
        self.target_ms
    }

    /// Marginal cost of a share of `rows` starting at KV `depth`.
    fn marginal_ms(&self, depth: usize, rows: usize) -> f64 {
        self.model.c1 * rows as f64 + self.model.c2 * attn_work(depth, rows)
    }

    /// Rows a share starting at KV `depth` may take, at most `max_rows`.
    /// `max_rows` untouched while the tick stays inside the hysteresis band
    /// (a shallow share is never trimmed); past it, what fits the target,
    /// aligned down to [`ROW_ALIGN`]. The first share of a tick always gets
    /// at least one aligned block - forward progress beats the target.
    /// 0 = the tick is full.
    pub fn rows(&self, depth: usize, max_rows: usize) -> usize {
        // a tick that opens shallow is never trimmed, whatever the model says
        if !self.granted_any && depth < PACE_MIN_DEPTH {
            return max_rows;
        }
        let full = self.fixed_ms + self.spent_ms + self.marginal_ms(depth, max_rows);
        if full <= self.target_ms * (1.0 + HYSTERESIS) {
            return max_rows;
        }
        let fit = self
            .model
            .rows_within(depth, self.target_ms - self.fixed_ms - self.spent_ms);
        let aligned = fit.min(max_rows) / ROW_ALIGN * ROW_ALIGN;
        if aligned > 0 {
            aligned
        } else if self.granted_any {
            0
        } else {
            ROW_ALIGN.min(max_rows)
        }
    }

    /// Spend the time a share of `rows` at `depth` costs.
    pub fn charge(&mut self, depth: usize, rows: usize) {
        if rows > 0 {
            self.granted_any = true;
            self.spent_ms += self.marginal_ms(depth, rows);
        }
    }
}

/// The mixed-tick shapes the scheduler runs. Each gets its own pacer: a fit
/// across them would describe none (a fused tick's riders share the pass,
/// a split tick's ride a second one, a spec round verifies k rows a rider).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TickKind {
    /// prefill and decode rows in one pass (`forward_unified_sampled`)
    Unified,
    /// `forward_mixed_sampled` / `forward_mixed`: fused or split, per family
    Mixed,
    /// spec-in-mixed: a verify round beside a prefill span
    Spec,
}

impl TickKind {
    fn name(self) -> &'static str {
        match self {
            TickKind::Unified => "unified",
            TickKind::Mixed => "mixed",
            TickKind::Spec => "spec",
        }
    }
}

/// A sized tick in flight: what recording it needs.
pub(crate) struct PacedTick {
    kind: TickKind,
    before: Vec<(usize, usize, usize)>,
    pass_riders: usize,
    /// a knee tick's full share (it samples the knee search)
    knee_full: Option<usize>,
    t0: Instant,
}

/// The scheduler's side: one pacer per tick kind. Every family's mixed tick
/// goes through here - the scheduler hands each backend a row budget, so
/// the budget is where time-pacing lives, and a backend only exposes its
/// queue (see the module docs). Dev switches: PADDOCK_NO_PACE pins the
/// plain row budget, PADDOCK_PACE_LOG logs every paced tick.
pub(crate) struct Pacers {
    unified: TickPacer,
    mixed: TickPacer,
    spec: TickPacer,
    on: bool,
    log: bool,
}

impl Pacers {
    pub(crate) fn new() -> Self {
        Self {
            unified: TickPacer::new(),
            mixed: TickPacer::new(),
            spec: TickPacer::new(),
            on: paddock_models::dev_var_os!("PADDOCK_NO_PACE").is_none(),
            log: paddock_models::dev_var_os!("PADDOCK_PACE_LOG").is_some(),
        }
    }

    fn pacer(&mut self, kind: TickKind) -> &mut TickPacer {
        match kind {
            TickKind::Unified => &mut self.unified,
            TickKind::Mixed => &mut self.mixed,
            TickKind::Spec => &mut self.spec,
        }
    }

    /// Size a mixed tick before its call. `waiting`: decode rows that wait
    /// on the tick's end; `pass_riders`: rows the call spends on them;
    /// `budget`: the scheduler's row budget for the call; `decode_rows`:
    /// what the backend's own ceiling is computed for. Returns the budget to
    /// pass and the tick to hand to [`Self::end`] once the call succeeded.
    /// Err is the backend's `prefill_prepare` error - the tick's own, for
    /// the caller's tick-error arms.
    pub(crate) fn begin(
        &mut self,
        kind: TickKind,
        g: &mut dyn Generator,
        waiting: usize,
        pass_riders: usize,
        budget: usize,
        decode_rows: usize,
    ) -> Result<(usize, Option<PacedTick>), GenError> {
        if !self.on {
            return Ok((budget, None));
        }
        g.prefill_prepare(budget)?;
        let before = g.prefill_queue();
        if before.is_empty() {
            return Ok((budget, None));
        }
        let full = budget.min(g.prefill_tick_cap(decode_rows));
        let knee_full = TickPacer::knee_tick(waiting, full, &before).then_some(full);
        let log = self.log;
        let pacer = self.pacer(kind);
        let sized = match pacer.plan(waiting, pass_riders, full, &before) {
            Some(plan) => {
                if log {
                    let (_, depth, remaining) = before[0];
                    match (plan.target_ms, pacer.model()) {
                        _ if plan.knee => tracing::info!(
                            "pace: {} head depth {depth} (rem {remaining}) rows {full} -> {} \
                             knee (streams {waiting}, riders {pass_riders})",
                            kind.name(),
                            plan.budget,
                        ),
                        (Some(t), Some(m)) => tracing::info!(
                            "pace: {} head depth {depth} (rem {remaining}) rows {full} -> {} \
                             target {t:.0} ms (model {:.1} + {:.3}/row + {:.2}/Mrow-pos, \
                             riders {pass_riders})",
                            kind.name(),
                            plan.budget,
                            m.c0,
                            m.c1,
                            m.c2
                        ),
                        _ => tracing::info!(
                            "pace: {} head depth {depth} (rem {remaining}) rows {full} -> {} \
                             bootstrap (riders {pass_riders})",
                            kind.name(),
                            plan.budget
                        ),
                    }
                }
                plan.budget
            }
            None => budget,
        };
        Ok((
            sized,
            Some(PacedTick {
                kind,
                before,
                pass_riders,
                knee_full,
                t0: Instant::now(),
            }),
        ))
    }

    /// A sized tick's call succeeded: fit it on what it actually executed.
    /// Call right after the backend returns, before the scheduler's own
    /// commit work - the wall is the GPU tick the riders waited on.
    pub(crate) fn end(&mut self, g: &dyn Generator, tick: Option<PacedTick>) {
        let Some(t) = tick else { return };
        let wall = t.t0.elapsed();
        let shape = executed(&t.before, &g.prefill_queue());
        // a call that advanced no prompt (a declined spec round) ran some
        // other tick's shape, not this kind's
        if shape.prefill_rows == 0 {
            return;
        }
        if let Some(full) = t.knee_full {
            self.pacer(t.kind)
                .record_knee(full, shape.prefill_rows, wall);
        }
        self.pacer(t.kind).record(
            t.pass_riders + shape.prefill_rows,
            shape.attn,
            shape.deepest,
            wall,
        );
    }
}

/// Ordinary least squares over the store, trusted only when the terms a
/// share's allowance depends on are established: a sane intercept, and the
/// per-row and depth coefficients each at least MIN_T_STAT standard errors
/// from zero. A process that never ran a deep tick, or only ever one row
/// count, cannot establish them - and then nothing is shrunk.
fn fit(w: &[Obs]) -> Option<CostModel> {
    let n = w.len();
    if n < MIN_OBS || w.iter().filter(|o| o.depth >= PACE_MIN_DEPTH).count() < MIN_DEEP_OBS {
        return None;
    }
    // normal equations X'X c = X'y over x = [1, rows, attn]
    let mut xtx = [[0.0f64; 3]; 3];
    let mut xty = [0.0f64; 3];
    for o in w {
        let x = [1.0, o.rows, o.attn];
        for i in 0..3 {
            for j in 0..3 {
                xtx[i][j] += x[i] * x[j];
            }
            xty[i] += x[i] * o.wall_ms;
        }
    }
    let inv = invert3(xtx)?;
    let c: Vec<f64> = (0..3)
        .map(|i| (0..3).map(|j| inv[i][j] * xty[j]).sum())
        .collect();
    let (c0, c1, c2) = (c[0], c[1], c[2]);
    // a pass has a real floor; a clearly negative intercept means the linear
    // model does not describe these ticks, and under-predicting is the
    // failure that matters (a paced tick would overrun its target)
    if !(c1 > 0.0 && c2 > 0.0 && c0 > -10.0 && c0.is_finite()) {
        return None;
    }
    let ssr: f64 = w
        .iter()
        .map(|o| {
            let e = o.wall_ms - (c0 + c1 * o.rows + c2 * o.attn);
            e * e
        })
        .sum();
    let sigma2 = ssr / (n - 3) as f64;
    // a perfect fit (se 0) is established; otherwise demand the t-stat
    let established = |c: f64, var: f64| {
        let se = (sigma2 * var).max(0.0).sqrt();
        se == 0.0 || c / se >= MIN_T_STAT
    };
    if !(established(c1, inv[1][1]) && established(c2, inv[2][2])) {
        return None;
    }
    Some(CostModel { c0, c1, c2 })
}

/// Inverse of a symmetric 3x3 by cofactors; None when (near-)singular - a
/// window whose shapes never varied cannot separate the terms.
fn invert3(m: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
    let c01 = m[1][2] * m[2][0] - m[1][0] * m[2][2];
    let c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0];
    let det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02;
    let scale = m[0][0].abs() * m[1][1].abs() * m[2][2].abs();
    if !det.is_finite() || det.abs() <= 1e-12 * scale.max(f64::MIN_POSITIVE) {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        [
            c00 * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            c01 * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            c02 * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A batch wider than the knee's: the depth pacing alone.
    const WIDE: usize = KNEE_STREAMS + 1;

    /// Run knee ticks beside one stream (6 verify rows) through `plan` and
    /// `record_knee` under the cost law `truth` until the search settles.
    fn settle_knee(p: &mut TickPacer, truth: CostModel) {
        let q = [(0, 0, 1_000_000)];
        for _ in 0..64 {
            let rows = p.plan(1, 6, 1024, &q).map_or(1024, |plan| plan.budget);
            let wall = truth.predict_ms(6 + rows, attn_work(0, rows));
            p.record_knee(1024, rows, Duration::from_secs_f64(wall / 1e3));
        }
    }

    /// One or two streams beside an ingest: once the search has priced the
    /// shares, the shallow tick takes the knee - on the GB10 law the half,
    /// as measured - and wider batches keep the elected span.
    #[test]
    fn a_few_streams_take_the_measured_knee() {
        let mut p = warmed(GB10, 180_000);
        let q = [(0, 0, 50_000)];
        // a fresh search runs its first knee tick full, to price against
        assert!(p.plan(1, 6, 1024, &q).is_none());
        settle_knee(&mut p, GB10);
        let plan = p.plan(1, 6, 1024, &q).expect("capped");
        assert_eq!((plan.budget, plan.knee), (512, true), "{plan:?}");
        assert!(p.plan(WIDE, 6, 1024, &q).is_none(), "width keeps the span");
        // a remainder the tick would not fill is not a knee tick
        let tail = p.plan(1, 6, 1024, &[(0, 0, 700)]);
        assert!(tail.is_none_or(|t| !t.knee), "{tail:?}");
    }

    /// Deep, the knee and the depth pacing both apply: the tick takes the
    /// smaller - never more than the depth pacing alone grants.
    #[test]
    fn the_knee_only_ever_tightens_the_depth_pacing() {
        let mut p = warmed(GB10, 180_000);
        settle_knee(&mut p, GB10);
        let q = [(0, 176_000, 50_000)];
        let paced = p.plan(WIDE, 6, 1024, &q).expect("paced").budget;
        let both = p.plan(1, 6, 1024, &q).expect("paced").budget;
        assert!(both <= paced && both >= ROW_ALIGN, "{both} vs {paced}");
        let shallow = p.plan(1, 6, 1024, &[(0, 0, 50_000)]).expect("knee");
        assert_eq!(shallow.budget, 512);
    }

    /// The GB10 / qwen3.8-27b shape measured on 2026-09-23: ~0.63 ms per row
    /// plus ~0.0128 ms per row per 1K of depth, on a ~100 ms pass floor.
    const GB10: CostModel = CostModel {
        c0: 100.0,
        c1: 0.63,
        c2: 12.8,
    };

    /// Time one tick of `prefill` rows at `depth` (plus two riders) by the
    /// true model, with a little deterministic jitter so the fit has
    /// residuals.
    fn tick(p: &mut TickPacer, truth: CostModel, prefill: usize, depth: usize, k: &mut u32) {
        let jitter = [1.0, 0.97, 1.03, 0.99, 1.02][(*k % 5) as usize];
        *k += 1;
        let attn = attn_work(depth, prefill);
        let ms = truth.predict_ms(prefill + 2, attn) * jitter;
        p.record(prefill + 2, attn, depth, Duration::from_secs_f64(ms / 1e3));
    }

    /// The ticks a serving process produces: ordinary short traffic of
    /// varied shapes (warmup, short prompts, final partial chunks), then one
    /// long cold prefill marching 1024-row chunks through depth.
    fn warmed(truth: CostModel, deepest: usize) -> TickPacer {
        let mut p = TickPacer::new();
        let mut k = 0u32;
        for &rows in &[128, 258, 398, 518, 698, 898, 1024, 1298, 1598, 2048] {
            for depth in [0, 1500, 3000] {
                tick(&mut p, truth, rows, depth, &mut k);
            }
        }
        let mut depth = 0;
        while depth < deepest {
            tick(&mut p, truth, 1024, depth, &mut k);
            depth += 1024;
        }
        p
    }

    #[test]
    fn the_fit_recovers_the_cost_model() {
        let m = warmed(GB10, 180_000)
            .model()
            .expect("depth term established");
        assert!((m.c2 - GB10.c2).abs() / GB10.c2 < 0.05, "{m:?}");
        assert!((m.c1 - GB10.c1).abs() / GB10.c1 < 0.10, "{m:?}");
    }

    #[test]
    fn nothing_is_shrunk_before_the_terms_are_established() {
        // identical shallow ticks cannot separate anything
        let mut p = TickPacer::new();
        for _ in 0..40 {
            p.record(1026, attn_work(0, 1024), 0, Duration::from_millis(760));
        }
        assert!(p.model().is_none());
        assert!(p.budget(2, 1024).is_none());
        // a long prefill alone - one row count sweeping depth - pins the depth
        // term but not the per-pass/per-row split, so it must not pace either
        let mut only_long = TickPacer::new();
        let mut k = 0;
        let mut depth = 0;
        while depth < 180_000 {
            tick(&mut only_long, GB10, 1024, depth, &mut k);
            depth += 1024;
        }
        assert!(only_long.model().is_none(), "{:?}", only_long.model());
    }

    #[test]
    fn a_span_with_no_rows_riding_is_paced_on_its_own_cost() {
        let p = warmed(GB10, 180_000);
        let m = p.model().expect("model");
        let b = p.budget(0, 1024).expect("model");
        assert!((b.target_ms() - m.predict_ms(1024, attn_work(0, 1024))).abs() < 1e-6);
        assert_eq!(b.rows(0, 1024), 1024);
        assert!(b.rows(176_000, 1024) < 1024);
    }

    #[test]
    fn a_long_prefill_does_not_evict_the_shapes_the_fit_needs() {
        let p = warmed(GB10, 400_000);
        let distinct_rows: std::collections::HashSet<u32> =
            p.obs.iter().map(|o| o.cell.0).collect();
        assert!(distinct_rows.len() >= 8, "{distinct_rows:?}");
        assert!(p.obs.len() <= MAX_OBS);
        assert!(p.model().is_some());
    }

    #[test]
    fn a_shallow_share_keeps_its_full_rows() {
        let p = warmed(GB10, 180_000);
        let b = p.budget(2, 1024).expect("model");
        assert_eq!(b.rows(0, 1024), 1024, "depth 0 is the target itself");
        assert_eq!(b.rows(0, 512), 512, "never grows past the row room");
        // inside the band: a moderately deep chunk keeps its elected span
        assert_eq!(b.rows(8_000, 1024), 1024);
        assert_eq!(b.rows(12_000, 1024), 1024);
    }

    #[test]
    fn a_deep_share_is_held_to_the_shallow_tick_time() {
        let p = warmed(GB10, 180_000);
        let m = p.model().expect("model");
        let mut b = p.budget(2, 1024).expect("model");
        let target = b.target_ms();
        let rows = b.rows(176_000, 1024);
        assert_eq!(rows % ROW_ALIGN, 0);
        assert!((128..=320).contains(&rows), "{rows} rows at 176K");
        let paced = m.predict_ms(2 + rows, attn_work(176_000, rows));
        let unpaced = m.predict_ms(2 + 1024, attn_work(176_000, 1024));
        assert!(paced <= target * 1.01, "{paced} vs target {target}");
        assert!(
            unpaced > 3.0 * target,
            "the stall being fixed: {unpaced} ms"
        );
        // spending it leaves nothing for a second deep share
        b.charge(176_000, rows);
        assert_eq!(b.rows(176_000, 1024), 0);
    }

    #[test]
    fn the_first_share_always_progresses() {
        let p = warmed(GB10, 180_000);
        let b = p.budget(2, 1024).expect("model");
        // deeper than the model can fit one block into: still one block
        assert_eq!(b.rows(50_000_000, 1024), ROW_ALIGN);
        assert_eq!(b.rows(50_000_000, 10), 10, "a tail shorter than a block");
    }

    #[test]
    fn the_depth_trade_is_cheap_per_token() {
        let m = warmed(GB10, 180_000).model().expect("model");
        let b = TickPacer {
            model: Some(m),
            ..Default::default()
        }
        .budget(2, 1024)
        .expect("model");
        let r = b.rows(176_000, 1024);
        let per_tok = |rows: usize| m.predict_ms(2 + rows, attn_work(176_000, rows)) / rows as f64;
        assert!(
            per_tok(r) / per_tok(1024) < 1.25,
            "{} vs {}",
            per_tok(r),
            per_tok(1024)
        );
    }

    #[test]
    fn a_fresh_process_bootstraps_itself_on_a_long_ingest() {
        // one shape only (the cap plus riders) sweeping depth: the fit alone
        // cannot establish, so the bootstrap shrinks the tick once growth is
        // measured - and those ticks are what establish the model
        let mut p = TickPacer::new();
        let mut k = 0u32;
        let mut depth = 0usize;
        let mut boots = 0;
        while depth < 150_000 && p.model().is_none() {
            let q = [(0, depth, 1_000_000)];
            let rows = match p.plan(WIDE, 2, 1024, &q) {
                Some(Plan {
                    budget,
                    target_ms: None,
                    ..
                }) => {
                    boots += 1;
                    budget
                }
                _ => 1024,
            };
            tick(&mut p, GB10, rows, depth, &mut k);
            depth += rows;
        }
        let m = p.model().expect("the bootstrap establishes the model");
        assert!(boots >= 1, "no bootstrap tick ran");
        // interleaved sizes separate the terms fast - not hundreds of ticks
        assert!(boots <= 40, "{boots} bootstrap ticks before establishing");
        assert!(depth < 60_000, "established only at {depth}");
        assert!((m.c2 - GB10.c2).abs() / GB10.c2 < 0.15, "{m:?}");
        // no growth measured yet: never; established: the model rules
        assert!(TickPacer::new().bootstrap_rows(2, 1024).is_none());
        assert!(p.bootstrap_rows(2, 1024).is_none());
    }

    #[test]
    fn short_traffic_alone_never_establishes_a_model() {
        // varied shapes, plenty of samples, a steep fake depth slope - but
        // nothing ever went deep, so the depth term is not trusted
        let steep = CostModel {
            c0: 100.0,
            c1: 0.5,
            c2: 400.0,
        };
        let mut p = TickPacer::new();
        let mut k = 0u32;
        for _ in 0..20 {
            for &rows in &[256, 512, 768, 1024] {
                for depth in [0, 2048, 4096, 6144] {
                    tick(&mut p, steep, rows, depth, &mut k);
                }
            }
        }
        assert!(p.model().is_none(), "{:?}", p.model());
    }

    #[test]
    fn a_tick_that_opens_shallow_is_never_trimmed() {
        // even a model that would trim it: the head share keeps its rows
        let steep = CostModel {
            c0: 100.0,
            c1: 0.5,
            c2: 400.0,
        };
        let b = TickPacer {
            model: Some(steep),
            ..Default::default()
        }
        .budget(1, 1024)
        .expect("model");
        assert_eq!(b.rows(3072, 1024), 1024);
        assert_eq!(b.rows(PACE_MIN_DEPTH - 1, 1024), 1024);
        assert!(b.rows(20_000, 1024) < 1024, "deep shares still pace");
    }

    #[test]
    fn short_context_noise_never_bootstraps() {
        // shallow ticks with a host hiccup far past the band: no depth, so no
        // bootstrap - a shallow tick is never shrunk on a wall-clock blip
        let mut p = TickPacer::new();
        for i in 0..200 {
            let wall = if i % 7 == 3 { 1400 } else { 760 };
            p.record(
                1026,
                attn_work(3000, 1024),
                3000,
                Duration::from_millis(wall),
            );
            assert!(
                p.plan(WIDE, 2, 1024, &[(0, 3000, 50_000)]).is_none(),
                "tick {i}"
            );
        }
    }

    #[test]
    fn the_plan_replays_fifo_spending() {
        let p = warmed(GB10, 180_000);
        // a deep head that cannot finish: it takes the whole (trimmed) budget
        let deep = p
            .plan(WIDE, 2, 1024, &[(1, 176_000, 50_000), (2, 0, 300)])
            .expect("paced");
        assert!(
            deep.budget < 1024 && deep.budget.is_multiple_of(ROW_ALIGN),
            "{deep:?}"
        );
        let alone = p
            .plan(WIDE, 2, 1024, &[(1, 176_000, 50_000)])
            .expect("paced");
        assert_eq!(
            deep.budget, alone.budget,
            "the prompt behind it gets nothing"
        );
        // a shallow head that finishes, then a deep prompt: the head is never
        // trimmed and the deep one gets what is left of the time
        let mixed = p
            .plan(WIDE, 2, 1024, &[(1, 0, 200), (2, 176_000, 50_000)])
            .expect("paced");
        assert!(mixed.budget > 200 && mixed.budget < 1024, "{mixed:?}");
        // nobody waiting, or nothing to trim: the caller's budget stands
        assert!(p.plan(0, 0, 1024, &[(1, 176_000, 50_000)]).is_none());
        assert!(p.plan(WIDE, 2, 1024, &[(1, 0, 50_000)]).is_none());
        assert!(p.plan(WIDE, 2, 1024, &[]).is_none());
    }

    /// A backend reduced to what the scheduler glue sees: a FIFO queue
    /// spent row-exact, like laguna/granite/gpt-oss.
    struct Fifo {
        q: Vec<(usize, usize, usize)>,
        prepared: usize,
    }

    impl Generator for Fifo {
        fn reset(&mut self) {}
        fn forward(&mut self, _token: u32) -> Result<Vec<f32>, GenError> {
            Ok(vec![0.0])
        }
        fn vocab(&self) -> usize {
            1
        }
        fn prefill_queue(&self) -> Vec<(usize, usize, usize)> {
            self.q.clone()
        }
        fn prefill_tick_cap(&self, _decode_rows: usize) -> usize {
            1024
        }
        fn prefill_prepare(&mut self, _budget: usize) -> Result<(), GenError> {
            self.prepared += 1;
            Ok(())
        }
    }

    impl Fifo {
        fn tick(&mut self, budget: usize) {
            let mut room = budget.min(1024);
            for e in self.q.iter_mut() {
                let take = e.2.min(room);
                e.1 += take;
                e.2 -= take;
                room -= take;
            }
            self.q.retain(|e| e.2 > 0);
        }
    }

    fn pacers(mixed: TickPacer, on: bool) -> Pacers {
        Pacers {
            unified: TickPacer::new(),
            mixed,
            spec: TickPacer::new(),
            on,
            log: false,
        }
    }

    #[test]
    fn the_scheduler_paces_any_backend_through_its_queue() {
        let mut p = pacers(warmed(GB10, 180_000), true);
        let seq0 = p.mixed.seq;
        let mut g = Fifo {
            q: vec![(3, 176_000, 50_000)],
            prepared: 0,
        };
        let (budget, tick) = p
            .begin(TickKind::Mixed, &mut g, 2, 2, 8190, 2)
            .expect("prepare");
        assert_eq!(g.prepared, 1, "the queue is resolved before it is priced");
        assert!(
            budget < 1024 && budget.is_multiple_of(ROW_ALIGN),
            "{budget}"
        );
        g.tick(budget);
        p.end(&g, tick);
        assert_eq!(p.mixed.seq, seq0 + 1, "the tick it ran is fitted");
        assert_eq!(p.unified.seq, 0, "on its own kind's pacer only");
        // shallow at width: the scheduler's budget passes through untouched
        let mut shallow = Fifo {
            q: vec![(3, 0, 50_000)],
            prepared: 0,
        };
        let (b, _) = p
            .begin(TickKind::Mixed, &mut shallow, WIDE, WIDE, 8190, WIDE)
            .expect("prepare");
        assert_eq!(b, 8190);
        // ...and beside a couple of streams the knee search starts: a full
        // tick to price against, then the half
        let (b, t) = p
            .begin(TickKind::Mixed, &mut shallow, 2, 2, 8190, 2)
            .expect("prepare");
        assert_eq!(b, 8190, "the first knee tick runs full");
        shallow.tick(b);
        p.end(&shallow, t);
        let (b, _) = p
            .begin(TickKind::Mixed, &mut shallow, 2, 2, 8190, 2)
            .expect("prepare");
        assert_eq!(b, 512, "then probes the half");
        // nothing queued: nothing to size or record
        let mut idle = Fifo {
            q: Vec::new(),
            prepared: 0,
        };
        let (b, t) = p
            .begin(TickKind::Mixed, &mut idle, 2, 2, 8190, 2)
            .expect("prepare");
        assert_eq!(b, 8190);
        assert!(t.is_none());
        // killed: not even the prepare hook runs
        let mut off = pacers(warmed(GB10, 180_000), false);
        let (b, t) = off
            .begin(TickKind::Mixed, &mut g, 2, 2, 8190, 2)
            .expect("prepare");
        assert_eq!((b, t.is_none(), g.prepared), (8190, true, 1));
    }

    #[test]
    fn executed_reads_the_tick_off_the_queue() {
        let before = [(1, 100_000, 5_000), (2, 0, 300), (3, 40, 900)];
        // slot 1 advanced 256, slot 2 finished (left the queue), slot 3 idle
        let after = [(1, 100_256, 4_744), (3, 40, 900)];
        let s = executed(&before, &after);
        assert_eq!(s.prefill_rows, 256 + 300);
        assert!((s.attn - (attn_work(100_000, 256) + attn_work(0, 300))).abs() < 1e-9);
        assert_eq!(s.deepest, 100_000);
        assert_eq!(executed(&before, &before), TickShape::default());
    }
}
