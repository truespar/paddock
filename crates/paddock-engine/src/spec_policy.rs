//! Closed-loop speculation control: pick the draft length K that maximizes
//! **goodput**, re-deciding every round from what the box is actually doing.
//!
//! Why this exists. Speculative decoding is not a feature you turn on - it is a
//! bet that costs a draft pass and a wider verify pass, and pays only when the
//! target accepts. The bet's odds move with both the workload (a drafter that
//! nails boilerplate misses on novel prose) and the load (at c1 the verify pass
//! is free - the weights are streamed anyway; at c32 those extra rows are real
//! compute). So a static "spec on/off", and even a hand-tuned batch->K ladder,
//! is wrong most of the time: it is right for the concurrency it was measured
//! at and wrong on either side.
//!
//! This is a known result. The spec-batching literature concludes you should
//! "disable speculation as effective concurrency rises ... speculation as a
//! low-batch mode, not an always-on feature", and reports vLLM+spec losing to
//! its own non-spec baseline at high concurrency, SGLang+EAGLE likewise.
//!
//! What the field does. vLLM ships a HAND-SPECIFIED ladder
//! (`num_speculative_tokens_per_batch_size = [[1,64,3],[65,128,1],[129,512,0]]`
//! - K=0 meaning "don't speculate here"); its older `disable_by_batch_size`
//!   rotted into dead code. That is a lookup table someone has to re-measure per
//!   model and per box. The state of the art is closed-loop: SmartSpec/TurboSpec
//!   (arXiv 2406.14066) formalize **goodput** - the rate of tokens generated AND
//!   accepted, so rejected drafts count as the waste they are - build a latency
//!   model over (batch x K), and combine it online with the measured acceptance
//!   rate to re-pick K continuously. That removes the tuning step and, more to
//!   the point, "prevent[s] performance regressions under high load or low
//!   acceptance regimes".
//!
//! Our variant. TurboSpec profiles latency offline; we do it online instead -
//! a local product cannot spend minutes profiling at model load, and the table
//! it would build would be wrong the moment the box picks up other work (this
//! is a desktop, not a datacenter node). So we learn `t(live, K)` from the
//! ticks we are already running, and keep it fresh with a small bandit-style
//! probe of the neighbours of the current best (the BanditSpec observation:
//! spec hyperparameters are a bandit, and you can select them online without
//! any offline pass). The cost is a warm-up of a few dozen ticks; the benefit
//! is a table that tracks the ACTUAL box under its ACTUAL load.
//!
//! What that COSTS: measured on an A6000 (qwen3.5-9B Q8_0, in-file MTP, c1),
//! `auto` trails `ladder` on short runs - warm-up is most of a 200-token run,
//! and at 2000 tokens cold it is still paying for exploration - and ties it
//! once warm.
//!
//! So today the closed loop does not beat a ladder that was hand-tuned on that
//! exact GPU and model - it converges to about the same place and charges an
//! exploration toll to get there. That is the honest state of it, and it is
//! why `Ladder` is still the default.
//!
//! The toll is structural, not a bug: every K in 0..=k_cap must be sampled
//! MIN_SAMPLES times before the argmax is trusted, and a short session ends
//! before that amortizes. Two fixes, in order of appeal:
//!   1. Start warm - seed the table from an analytic tick-cost prior, or
//!      persist a converged table per (model, GPU) and reload it at startup.
//!      This is TurboSpec's offline profile arriving through the back door,
//!      and it is the obvious next move.
//!   2. Explore cheaper - bound exploration to K values adjacent to the prior's
//!      guess instead of sweeping the whole range.
//!      Where the controller should win once seeded is the case the ladder cannot
//!      express at all: a workload whose acceptance rate is nothing like the one the
//!      ladder was tuned against (novel prose vs boilerplate), or a machine with
//!      other work on it. That case is not measured yet either - say so until it is.
//!
//! The model. The expected tokens per slot per round at draft length K is
//!     E[tokens](K) = 1 + sum_{i=1..K} S(i)
//! where S(i) = P(the first i drafts are all accepted) - the prefix
//! survival curve, learned per POSITION from every round (a round at depth
//! k observes S(1..k): position i survived iff accepted >= i). K=0 gives
//! exactly 1 (a plain decode tick), so the ladder is compared against not
//! speculating on equal terms; positions deeper than any round has reached
//! extrapolate geometrically from the last observed ratio. Then
//!     goodput(K) = live * E[tokens](K) / t(live, K)
//! and we take the argmax over the K the row budget allows. K*=0 is the "spec
//! off for this tick" answer - it falls out of the same maximization rather
//! than being a separate switch, which is the whole point.
//!
//! Why per POSITION and not Leviathan's geometric (1 - a^(K+1)) / (1 - a)
//! with one learned `a` (the form this controller originally shipped with):
//! a block drafter's survival curve is not geometric - DFlash2 on qwen3.8
//! reads 0.78 / 0.58 / 0.43 / 0.33 / 0.27 / 0.22 / 0.19 per position, a fat
//! tail whose per-draft mean at depth 4 (~0.5) fitted as `a` predicts a +2.6%
//! yield from K=4 to 7 where the curve actually pays +22%. The fitted-`a`
//! controller therefore sat at K~3.8 on a cell that wants 7 - pinning K=7
//! measured clearly faster with the same drafter.

use std::str::FromStr;

/// A synchronous chain only needs the deepest prefix any slot will verify.
/// Drafting the global ceiling (often seven) when all slots consume two
/// wastes target-head reads and GPU steps. Block drafters keep their original
/// geometry: shortening their input changes every position's prediction.
pub(crate) fn synchronous_draft_budget(
    ceiling: usize,
    block: Option<usize>,
    requested: impl IntoIterator<Item = usize>,
) -> usize {
    if block.is_some() {
        ceiling
    } else {
        requested.into_iter().max().unwrap_or(0).min(ceiling)
    }
}

/// What the operator asked for. `Auto` is the interesting one; the rest exist
/// because benchmarking and parity work need to pin the variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpecPolicy {
    /// Never speculate. No drafter is loaded, so the drafter weights and the
    /// wide verify planes cost zero VRAM - the one thing a controller cannot
    /// win back, and the reason this is what the UI's "off" means.
    Off,
    /// The legacy hand-tuned row-budget ladder in `service::serve_spec_k_budget`
    /// - K = (rows / live) - 1 with measured tier boundaries.
    ///
    /// This is the DEFAULT purely so that introducing the controller changes no
    /// existing endpoint's behavior on the commit that adds it. Every serving
    /// flip in this repo is supposed to be measured before it lands (
    /// reverted an unmeasured one that silently corrupted expert repacks), and
    /// a controller that quietly replaced the tuned ladder everywhere would be
    /// exactly that mistake with a nicer story. Measure Ladder vs Auto on the
    /// board, then make Auto the default and delete this variant.
    #[default]
    Ladder,
    /// Closed loop: K re-picked every round to maximize goodput, K=0 included.
    /// This is what "spec on" means - never "always speculate".
    Auto,
    /// Pin K. Bench/A-B/parity only: it defeats the controller by design, so
    /// the serve loop logs it loudly.
    Fixed(usize),
}

impl SpecPolicy {
    pub fn is_off(self) -> bool {
        self == SpecPolicy::Off
    }
    /// Does this policy hand K to the controller? Only `Auto` learns; the
    /// others are open-loop and must not have their latency folded into the
    /// table (a pinned K would teach the controller only about itself).
    pub fn is_closed_loop(self) -> bool {
        self == SpecPolicy::Auto
    }
}

/// Parse the `spec` config key: `off` | `on` | `adaptive` | `<K>`.
///
/// The three user-facing values map onto the policies the way the Studio's
/// labels read: `on` is speculation at the tuned per-batch depth (the ladder),
/// `adaptive` is the closed loop. `auto` and `ladder` are accepted as aliases
/// so older files and A/B scripts keep working.
///
/// A bare integer is a pin, EXCEPT `0` which reads as off - "zero drafts" and
/// "don't speculate" are the same statement, and rejecting it would be pedantry.
impl FromStr for SpecPolicy {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let t = s.trim().to_ascii_lowercase();
        match t.as_str() {
            "off" | "false" | "no" | "none" | "0" => Ok(SpecPolicy::Off),
            "adaptive" | "auto" => Ok(SpecPolicy::Auto),
            "on" | "true" | "yes" | "ladder" | "legacy" => Ok(SpecPolicy::Ladder),
            _ => match t.parse::<usize>() {
                Ok(k) if k <= MAX_K => Ok(SpecPolicy::Fixed(k)),
                Ok(k) => Err(format!(
                    "spec = {k}: draft length above the {MAX_K} ceiling"
                )),
                Err(_) => Err(format!(
                    "spec = {s:?}: expected \"off\", \"on\", \"adaptive\", or a draft length 1..={MAX_K}"
                )),
            },
        }
    }
}

/// Renders back into a value `from_str` accepts - Not a pretty label. The
/// runner hands `to_string()` to the engine through PADDOCK_SPEC, so anything
/// unparseable here degrades to the default policy one process hop after the
/// operator chose it, with nothing to show for it. `display_round_trips_through
/// _the_parser` is the guard; it caught exactly that, on `Fixed`.
impl std::fmt::Display for SpecPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpecPolicy::Off => f.write_str("off"),
            SpecPolicy::Ladder => f.write_str("on"),
            SpecPolicy::Auto => f.write_str("adaptive"),
            SpecPolicy::Fixed(k) => write!(f, "{k}"),
        }
    }
}

/// Draft-length ceiling the tables are sized for. Above this the verify pass
/// stops fitting the row budget on every backend we have.
pub const MAX_K: usize = 16;

/// Live-slot buckets for the latency table. Tick cost is roughly flat inside a
/// bucket and steps between them (weight-stream amortization is the reason:
/// doubling rows in the same GEMM class is nearly free, crossing into another
/// class is not), so a geometric ladder beats per-slot cells that would each
/// see a twentieth of the samples.
const BUCKETS: [usize; 8] = [1, 2, 4, 8, 16, 32, 64, usize::MAX];

fn ctl_debug() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var_os("PADDOCK_SPEC_CTL_DEBUG").is_some())
}

fn bucket_of(live: usize) -> usize {
    BUCKETS
        .iter()
        .position(|&b| live <= b)
        .unwrap_or(BUCKETS.len() - 1)
}

/// EWMA weight for the latency cells. 0.2 tracks a load change in ~15 ticks
/// while still averaging out the odd descheduled tick.
const LAT_W: f64 = 0.2;
/// EWMA weight for acceptance. Slower than latency: acceptance is a property of
/// the model/workload pair and should not swing on one unlucky round.
const ACC_W: f64 = 0.05;
/// Samples before a cell is trusted enough to be chosen on its own merit.
/// Five, not three: the warm start measures the cap first on a fresh serve,
/// and the first rounds of a process are where every one-time stall lands
/// (graph capture, the instantiated graph's upload, lazy scratch
/// allocations, first-touch page faults) - with three samples the k=7
/// anchor's MIN still read 37-40 ms against k=4's 27 ms on c1, so the
/// controller settled on k=4 and left a third of the decode rate on the
/// floor. Two more rounds at the cap are cheap; a mis-seeded cap cell
/// costs the whole serve.
const MIN_SAMPLES: u32 = 5;
/// Observations of draft coverage before its running mean hands over to an
/// EWMA. Coverage is a per-bucket property of the SCHEDULER (how many live
/// slots drafted), not of the drafter, so it keeps its own simple estimator.
const COV_MEAN_N: f64 = 48.0;
/// Shrinkage weight for the per-position survival estimate: how many
/// pseudo-observations of the POOLED curve a position's own evidence has to
/// outweigh before it is trusted on its own (empirical Bayes - see
/// `surv_at`). Position i with no evidence sits exactly on the pooled curve;
/// one with hundreds of samples sits on its own rate.
const SURV_SHRINK: f64 = 24.0;
/// Beta prior on the pooled acceptance rate, centred at 0.7 with the weight
/// of a handful of rounds: high enough to fund exploration on a fresh serve
/// (a pessimistic seed makes K=0 win forever and the controller then only
/// ever observes K=0), light enough to be overwhelmed within a few dozen.
const ALPHA_PRIOR_A: f64 = 7.0;
const ALPHA_PRIOR_B: f64 = 3.0;
/// Per-round decay on the acceptance evidence. Acceptance is not stationary
/// - it swings with content (a code stretch drafts far better than prose),
///   which is the whole premise of confidence-driven drafting - so the counts
///   have to forget.
///
/// 0.99 is an effective window of ~100 rounds - a second or two of serving,
/// so a content shift is believed while it is still happening.
///
/// Chosen against what it replaced rather than from taste: the old
/// per-position EWMA ran at ACC_W = 0.05, a ~20-OBSERVATION window, and a
/// round at depth k fed it k observations - anything much slower here would
/// be a real loss of adaptivity dressed up as stability.
///
/// The window has to be short because the evidence is ASYMMETRIC: a round
/// contributes up to k acceptances but at most one rejection (the geometric
/// MLE's censoring - positions past the first rejection were never tested),
/// so a high estimate takes longer to come down than a low one takes to go
/// up. That is the statistics, not a bug, and the window is what bounds it:
/// at 100 rounds a drafter that stops landing anything is under 0.35 within
/// ~300 rounds and under 0.15 within ~600, while each estimate still rests
/// on several hundred token observations.
const ACC_DECAY: f64 = 0.99;
/// Probe a neighbour of the incumbent every N decisions, so a cell that went
/// stale (or was never visited) gets re-measured instead of being believed
/// forever. Pure exploitation would freeze the first guess in place.
const PROBE_EVERY: u64 = 64;

/// Samples a cold cell keeps on its MIN instead of an EWMA (see `observe`).
const WARM_SAMPLES: u32 = 6;

#[derive(Clone, Copy, Default)]
struct Cell {
    /// Measured wall-clock seconds for one round at this (bucket, K): the
    /// minimum over the first WARM_SAMPLES samples, an EWMA from there.
    t: f64,
    n: u32,
}

impl Cell {
    fn observe(&mut self, t: f64) {
        // Round latency noise is ONE-SIDED: a round can only be slower than
        // its intrinsic cost (graph capture on the first round of a shape,
        // the instantiated graph's first upload on the second, a descheduled
        // host, another slot's prefill chunk landing mid-round) - never
        // faster. So a cold cell keeps its MINIMUM rather than a mean: the
        // two-anchor warm start measures the cap first on a fresh serve, and
        // seeding on the stall-tainted early samples made k=7 read 40.6 ms
        // against k=4's 27.1 at the same c1 width, and the controller then
        // sat at k=4 for the rest of the serve. Once warm the EWMA tracks
        // drift both ways, as before.
        if self.n < WARM_SAMPLES {
            self.t = if self.n == 0 { t } else { self.t.min(t) };
        } else {
            self.t = (1.0 - LAT_W) * self.t + LAT_W * t;
        }
        self.n = self.n.saturating_add(1);
    }
    fn ready(&self) -> bool {
        self.n >= MIN_SAMPLES
    }
}

/// The controller. One per serving loop; cheap to consult (an argmax over at
/// most 17 cells) and updated from the rounds the loop already runs.
pub struct SpecController {
    policy: SpecPolicy,
    /// Per-position evidence, exponentially decayed: `tri[i]` slots that
    /// drafted at least i deep, `hit[i]` those whose first i were all
    /// accepted. So `hit[i]/tri[i]` is the empirical prefix survival S(i).
    ///
    /// These are EVIDENCE, not the estimate - see `surv_at`. Holding them
    /// separately is the whole point: the old design kept
    /// one EWMA per position and used it directly, which made each depth an
    /// independent arm. `tri[i]` only advances when a round drafts at least
    /// i deep, so the moment the argmax settled at k the arm at k+1 stopped
    /// receiving evidence, kept its stale (pessimistic) estimate, and went on
    /// losing - a self-reinforcing incumbent that no exploration rate fixes,
    /// because the model, not the sampling, was wrong.
    tri: [f64; MAX_K + 1],
    hit: [f64; MAX_K + 1],
    /// Pooled acceptance evidence for the geometric backbone: total drafted
    /// tokens accepted, and rounds that were actually stopped by a rejection.
    /// Every round updates these whatever depth it ran at, which is what
    /// makes the estimate for an undrafted depth a real number rather than a
    /// stale one (Leviathan's alpha).
    acc_tokens: f64,
    acc_stops: f64,
    /// `t[bucket][K]`.
    lat: Vec<[Cell; MAX_K + 1]>,
    /// Per-bucket DRAFT COVERAGE: the fraction of live slots that actually
    /// drafted this round, EWMA, seeded 1.0.
    ///
    /// Without this the goodput comparison is dishonest by construction.
    /// `surv[]` is conditioned on a slot HAVING drafted (`tries[i]` counts
    /// slots that drafted at least i deep), so `yield_of(k)` is the expected
    /// tokens for a DRAFTING slot - but the round's wall is paid for every
    /// live slot, drafting or not. Measured on qwen3.8 nvfp4 at c16: the
    /// controller believed k=7 yielded 5.56 tokens/slot at 17.2 ms and so
    /// preferred it 4:1 over k=0. The 17.2 ms was right - the serve loop's
    /// own tick accounting covers 100% of the wall - but the real
    /// per-live-slot yield was 0.81, because only a handful of the 16 slots
    /// were drafting at all (`[verify-fa] ... blocks=6` at live=16).
    /// Speculating cost more than half the decode rate and the controller
    /// could not see it.
    ///
    /// Seeded 1.0 so a fresh serve behaves exactly as before and only ever
    /// corrects DOWNWARD as evidence arrives.
    cov: Vec<(f64, f64)>,
    decisions: u64,
    /// Round-robin cursor for the neighbour probe.
    probe: usize,
}

/// Drafts and accepts booked during one tick, across every slot on it.
///
/// Threaded through `Slot::spec_round` by reference rather than summed after
/// the fact: the serve loop has half a dozen speculative tick shapes, and a new
/// one that forgot to report would silently feed the controller a stale
/// acceptance rate - the kind of bug that shows up as a slow drift in
/// throughput and gets blamed on the box. Making the tally an argument means
/// the compiler asks the question at every site.
///
/// Besides the totals it keeps the per-position prefix census the survival
/// curve learns from: `tries[i]` slots drafted at least `i` deep this tick,
/// `hits[i]` of them had their first `i` drafts accepted.
#[derive(Clone, Copy, Debug)]
pub struct RoundTally {
    pub drafted: usize,
    pub accepted: usize,
    tries: [u32; MAX_K + 1],
    hits: [u32; MAX_K + 1],
    /// Slots whose draft was actually STOPPED by a rejection (`accepted <
    /// drafted`). A slot that had everything accepted is censored, not a
    /// failure: the next position was never tested. Conflating the two is
    /// what turns a good deep drafter into a shallow one.
    stops: u32,
}

impl Default for RoundTally {
    fn default() -> Self {
        Self {
            drafted: 0,
            accepted: 0,
            tries: [0; MAX_K + 1],
            hits: [0; MAX_K + 1],
            stops: 0,
        }
    }
}

impl RoundTally {
    pub fn book(&mut self, drafted: usize, accepted: usize) {
        self.drafted += drafted;
        self.accepted += accepted;
        if accepted < drafted {
            self.stops += 1;
        }
        for i in 1..=drafted.min(MAX_K) {
            self.tries[i] += 1;
            if accepted >= i {
                self.hits[i] += 1;
            }
        }
    }

    /// One slot's round as a tally (tests and single-shape callers).
    pub fn one(drafted: usize, accepted: usize) -> Self {
        let mut t = Self::default();
        t.book(drafted, accepted);
        t
    }
}

impl SpecController {
    pub fn new(policy: SpecPolicy) -> Self {
        Self {
            policy,
            tri: [0.0; MAX_K + 1],
            hit: [0.0; MAX_K + 1],
            acc_tokens: 0.0,
            acc_stops: 0.0,
            lat: vec![[Cell::default(); MAX_K + 1]; BUCKETS.len()],
            cov: vec![(1.0, 0.0); BUCKETS.len()],
            decisions: 0,
            probe: 0,
        }
    }

    pub fn policy(&self) -> SpecPolicy {
        self.policy
    }

    /// Pooled per-token acceptance - Leviathan's alpha, as a Beta posterior
    /// mean over evidence from every round regardless of the depth it ran at.
    ///
    /// A round that drafted `d` and had `a` accepted contributes `a`
    /// successes, plus one failure if it was actually stopped (`a < d`). A
    /// round where everything was accepted is censored, not a failure - the
    /// next position was never tested, and counting it as a rejection is how
    /// a deep drafter gets talked out of drafting deep.
    pub fn alpha(&self) -> f64 {
        let a = ALPHA_PRIOR_A + self.acc_tokens;
        let b = ALPHA_PRIOR_B + self.acc_stops;
        (a / (a + b)).clamp(0.0, 1.0)
    }

    /// Prefix survival S(i) = P(the first i drafts are all accepted).
    ///
    /// Empirical Bayes: the pooled geometric curve `alpha^i` is the prior,
    /// and a position's own evidence pulls it away in proportion to how much
    /// of it there is. A depth nothing has drafted sits exactly on the pooled
    /// curve - a real estimate that moves as the drafter's behaviour moves -
    /// where the old per-position EWMA left it frozen at whatever it last saw.
    /// A depth with real traffic keeps its own measured rate, which is what
    /// captures the decay-with-depth the pooled geometric cannot express
    /// (acceptance falls off with
    /// position, and that fall-off is the whole reason to bound k at all).
    fn surv_at(&self, i: usize) -> f64 {
        if i == 0 {
            return 1.0;
        }
        let prior = self.alpha().powi(i as i32);
        ((self.hit[i] + SURV_SHRINK * prior) / (self.tri[i] + SURV_SHRINK)).clamp(0.0, 1.0)
    }

    /// Expected tokens generated per slot for one round at draft length `k`,
    /// counting the bonus token: 1 + sum_{i<=k} S(i).
    ///
    /// Monotone-clamped: an estimate at depth i can wobble above depth i-1's,
    /// and a prefix probability cannot rise with length.
    fn yield_of(&self, k: usize) -> f64 {
        let mut total = 1.0f64;
        let mut cur = 1.0f64;
        for i in 1..=k.min(MAX_K) {
            cur = cur.min(self.surv_at(i));
            total += cur;
        }
        total
    }

    /// Will a round at this width speculate at all, as far as the table can
    /// tell? A read-only companion to `pick_k` for the scheduler's draft-head
    /// hints: a drafter fed from every forward (qwen4exp stashes each decode
    /// row and drains it through a head pass) otherwise pays that feeding on
    /// ticks no round will ever consume. The width caps alone cannot answer
    /// it - at 8 live slots Flash-Next is inside every cap and still decides
    /// not to speculate, and the feeding was 19 ms of a 45 ms tick.
    ///
    /// Exploration counts as yes: while the bucket's anchors are unmeasured
    /// the controller is about to run rounds, and a head that stopped being
    /// fed cannot serve them.
    pub fn will_speculate(&self, live: usize, k_ladder: usize) -> bool {
        let k_cap = k_ladder.min(MAX_K);
        match self.policy {
            SpecPolicy::Off => false,
            SpecPolicy::Ladder => k_cap > 0,
            SpecPolicy::Fixed(k) => k.min(k_cap) > 0,
            SpecPolicy::Auto => {
                if k_cap == 0 || live == 0 {
                    return false;
                }
                let b = bucket_of(live);
                let row = &self.lat[b];
                let mid = (k_cap / 2).max(1);
                if [0, k_cap, mid].iter().any(|&k| !row[k].ready()) {
                    return true;
                }
                self.argmax(b, k_cap) > 0
            }
        }
    }

    pub fn pick_k(&mut self, live: usize, k_ladder: usize) -> usize {
        let k_cap = k_ladder.min(MAX_K);
        match self.policy {
            SpecPolicy::Off => 0,
            SpecPolicy::Ladder => k_cap,
            SpecPolicy::Fixed(k) => k.min(k_cap),
            SpecPolicy::Auto => self.pick_auto(live, k_cap),
        }
    }

    fn pick_auto(&mut self, live: usize, k_cap: usize) -> usize {
        if k_cap == 0 || live == 0 {
            return 0;
        }
        self.decisions += 1;
        let b = bucket_of(live);
        let row = &self.lat[b];

        // Warm start: a cold bucket used to sweep every K in 0..=k_cap three
        // times before trusting its argmax - 24 rounds of mostly shallow
        // drafting per bucket, which the first run on a fresh serve paid in
        // full (visible as a several-percent gap between the first and
        // second run of the same load). Round latency is close to affine in the
        // verify rows inside a bucket (weights stream once; rows add
        // compute), so a few anchors pin the line and every other K is priced
        // by interpolation until a probe or a natural visit measures it.
        //
        // K=0 is one of those anchors, and it used to be left out with the
        // note that dense ticks measure it for free. They do not: a dense tick
        // is only booked when this controller itself returns 0 (the serve loop
        // opens a tick for the speculative branches only), so on a bucket that
        // never chose 0 the baseline cell stayed n=0 and `argmax` had to
        // extrapolate it off the speculative cells' fit. That fit is the
        // round's own line, fixed cost included - and a family whose round
        // copies recurrent state per round (qwen4exp saves GDN + PLE windows)
        // has a big one. Measured on Flash-Next at 8 live slots, 2026-09-17:
        // the fit put k=0 at ~59 ms against a real dense tick of ~26 ms, so
        // k=2 won the goodput comparison forever and the serve ran 40% under
        // its own no-spec throughput at 79% acceptance. Anchoring the baseline
        // costs one dense tick per bucket and is the cell every other K is
        // judged against.
        let mid = (k_cap / 2).max(1);
        for k in [0, k_cap, mid] {
            if !row[k].ready() {
                return k;
            }
        }

        // dev trace: the bucket's latency row + yields every PROBE_EVERY
        // decisions (PADDOCK_SPEC_CTL_DEBUG=1) - the controller's choices are
        // otherwise invisible in a serve log
        if self.decisions.is_multiple_of(PROBE_EVERY) && ctl_debug() {
            let cells: Vec<String> = (0..=k_cap)
                .map(|k| {
                    let c = self.lat[b][k];
                    format!("k{k}:{:.1}ms/n{}/y{:.2}", c.t * 1e3, c.n, self.yield_of(k))
                })
                .collect();
            eprintln!(
                "[spec-ctl] live={live} bucket={b} best={} cov={:.2} {}",
                self.argmax(b, k_cap),
                self.cov[b].0,
                cells.join(" ")
            );
        }
        // Periodic neighbour probe: the incumbent's ±1 (and K=0, the baseline
        // everything is measured against) get re-sampled so a stale cell cannot
        // hold the decision forever.
        if self.decisions.is_multiple_of(PROBE_EVERY) {
            self.probe = self.probe.wrapping_add(1);
            let best = self.argmax(b, k_cap);
            let cand = match self.probe % 3 {
                0 => 0,
                1 => best.saturating_sub(1),
                _ => (best + 1).min(k_cap),
            };
            return cand;
        }
        self.argmax(b, k_cap)
    }

    /// goodput(K) = live * E[tokens](K) / t(live, K), maximized. `live` cancels
    /// out of the comparison (it is the same for every K this round), so it is
    /// left out of the arithmetic. Unmeasured K ride the affine fit of the
    /// bucket's measured cells (see the warm start in `pick_auto`); a cell
    /// chosen on the fit gets measured by being run, which is the cheapest
    /// exploration there is.
    fn argmax(&self, b: usize, k_cap: usize) -> usize {
        let row = &self.lat[b];
        // least-squares t = a + c*k over the ready cells (>= 2 distinct K)
        let pts: Vec<(f64, f64)> = (0..=k_cap)
            .filter(|&k| row[k].ready() && row[k].t > 0.0)
            .map(|k| (k as f64, row[k].t))
            .collect();
        let fit = if pts.len() >= 2 {
            let n = pts.len() as f64;
            let (sx, sy) = pts
                .iter()
                .fold((0.0, 0.0), |(a, b), &(x, y)| (a + x, b + y));
            let (sxx, sxy) = pts
                .iter()
                .fold((0.0, 0.0), |(a, b), &(x, y)| (a + x * x, b + x * y));
            let den = n * sxx - sx * sx;
            if den.abs() > 1e-12 {
                let c = (n * sxy - sx * sy) / den;
                let a = (sy - c * sx) / n;
                let t_min = pts.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
                Some((a, c, t_min))
            } else {
                None
            }
        } else {
            None
        };
        let mut best_k = 0;
        let mut best_g = f64::NEG_INFINITY;
        for (k, c) in row[..=k_cap].iter().enumerate() {
            let t = if c.ready() && c.t > 0.0 {
                c.t
            } else if let Some((a, slope, t_min)) = fit {
                // never price an unmeasured cell below half the fastest
                // measured one - an optimistic line must not invent a win
                (a + slope * k as f64).max(0.5 * t_min)
            } else {
                continue;
            };
            // Every live slot commits its own target token; only the
            // DRAFTING fraction can commit more. Without the cov term this
            // reads yield_of(k) for slots that never drafted.
            let cov = self.cov[b].0;
            let g = (1.0 + cov * (self.yield_of(k) - 1.0)) / t;
            if g > best_g {
                best_g = g;
                best_k = k;
            }
        }
        best_k
    }

    /// Book one round: the draft length it ran at, how long it took, and how
    /// the drafts fared.
    ///
    /// Plain decode ticks must be reported too, with `k = 0` - that cell is the
    /// baseline every other K is judged against, and a controller that never
    /// measures it can only ever compare speculation against more speculation.
    /// `k` is passed rather than remembered because the serve loop interleaves
    /// tick shapes: a dense tick between a pick and its round would otherwise
    /// land its latency in the speculative cell.
    pub fn observe(&mut self, live: usize, k: usize, secs: f64, tally: RoundTally) {
        if self.policy != SpecPolicy::Auto || secs <= 0.0 || live == 0 {
            return;
        }
        let b = bucket_of(live);
        self.lat[b][k.min(MAX_K)].observe(secs);
        // Draft coverage for this bucket (see `cov`). Only speculative rounds
        // carry the signal - a k=0 tick drafts nothing by definition and would
        // drag the estimate to zero, which is the opposite of what it means.
        if k > 0 {
            let seen = (tally.tries[1] as f64 / live as f64).clamp(0.0, 1.0);
            let (ema, n) = self.cov[b];
            // running mean while thin, EWMA once it has weight
            self.cov[b] = if n < COV_MEAN_N {
                ((ema * n + seen) / (n + 1.0), n + 1.0)
            } else {
                ((1.0 - ACC_W) * ema + ACC_W * seen, n)
            };
        }
        // Acceptance evidence. The POOLED terms take every round whatever
        // depth it ran at - that is what keeps an undrafted depth's estimate
        // live instead of frozen - while the per-position counts record the
        // decay-with-depth that the pooled geometric cannot express.
        self.acc_tokens = self.acc_tokens * ACC_DECAY + tally.accepted as f64;
        self.acc_stops = self.acc_stops * ACC_DECAY + f64::from(tally.stops);
        for i in 1..=MAX_K {
            self.tri[i] *= ACC_DECAY;
            self.hit[i] *= ACC_DECAY;
            self.tri[i] += f64::from(tally.tries[i]);
            self.hit[i] += f64::from(tally.hits[i]);
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
pub(crate) fn spec_pos_census(drafted: usize, accepted: usize) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_parses_the_three_shapes() {
        // the three the Studio writes, spelled as its labels read
        assert_eq!("off".parse::<SpecPolicy>().unwrap(), SpecPolicy::Off);
        assert_eq!("on".parse::<SpecPolicy>().unwrap(), SpecPolicy::Ladder);
        assert_eq!("adaptive".parse::<SpecPolicy>().unwrap(), SpecPolicy::Auto);
        assert_eq!("ADAPTIVE".parse::<SpecPolicy>().unwrap(), SpecPolicy::Auto);
        // aliases: older files and the A/B scripts keep working
        assert_eq!("auto".parse::<SpecPolicy>().unwrap(), SpecPolicy::Auto);
        assert_eq!("ladder".parse::<SpecPolicy>().unwrap(), SpecPolicy::Ladder);
        assert_eq!("4".parse::<SpecPolicy>().unwrap(), SpecPolicy::Fixed(4));
        // 0 drafts and "off" are the same statement
        assert_eq!("0".parse::<SpecPolicy>().unwrap(), SpecPolicy::Off);
        assert!("99".parse::<SpecPolicy>().is_err());
        assert!("sometimes".parse::<SpecPolicy>().is_err());
    }

    /// Display feeds PADDOCK_SPEC, which the engine parses back - so the two
    /// must round-trip or a policy would silently degrade to the default one
    /// process hop after the operator chose it.
    #[test]
    fn display_round_trips_through_the_parser() {
        for p in [
            SpecPolicy::Off,
            SpecPolicy::Ladder,
            SpecPolicy::Auto,
            SpecPolicy::Fixed(5),
        ] {
            assert_eq!(p.to_string().parse::<SpecPolicy>().unwrap(), p, "{p:?}");
        }
    }

    #[test]
    fn off_never_speculates() {
        let mut c = SpecController::new(SpecPolicy::Off);
        assert_eq!(c.pick_k(1, 7), 0);
        assert_eq!(c.pick_k(32, 7), 0);
    }

    #[test]
    fn synchronous_chains_skip_unused_depth_but_blocks_keep_their_geometry() {
        assert_eq!(synchronous_draft_budget(7, None, [2]), 2);
        assert_eq!(synchronous_draft_budget(7, None, [1, 4, 2]), 4);
        assert_eq!(synchronous_draft_budget(3, None, [7, 2]), 3);
        assert_eq!(synchronous_draft_budget(0, None, [7]), 0);
        assert_eq!(synchronous_draft_budget(7, None, []), 0);
        assert_eq!(synchronous_draft_budget(7, Some(8), [2]), 7);
        assert_eq!(synchronous_draft_budget(3, Some(8), [7]), 3);
        assert_eq!(synchronous_draft_budget(0, Some(8), [7]), 0);
    }

    #[test]
    fn ladder_is_the_default_and_reproduces_the_legacy_rule() {
        // an absent `spec` key must serve exactly as it did before the
        // controller existed - the ladder's number, passed straight through
        assert_eq!(SpecPolicy::default(), SpecPolicy::Ladder);
        let mut c = SpecController::new(SpecPolicy::Ladder);
        for (live, ladder) in [(1, 7), (8, 3), (32, 0)] {
            assert_eq!(c.pick_k(live, ladder), ladder);
        }
    }

    /// The invariant that makes this safe to land: whatever the policy, a round
    /// never carries more verify rows than the tuned ladder already allows.
    #[test]
    fn no_policy_can_exceed_the_ladder() {
        for p in [
            SpecPolicy::Off,
            SpecPolicy::Ladder,
            SpecPolicy::Auto,
            SpecPolicy::Fixed(16),
        ] {
            let mut c = SpecController::new(p);
            for live in [1usize, 4, 16, 32] {
                for ladder in [0usize, 1, 3, 7] {
                    // hammer it so Auto passes warm-up and reaches its argmax
                    for _ in 0..200 {
                        let k = c.pick_k(live, ladder);
                        assert!(
                            k <= ladder,
                            "{p:?}: k={k} exceeded ladder={ladder} at live={live}"
                        );
                        c.observe(live, k, 0.01, RoundTally::one(k, k));
                    }
                }
            }
        }
    }

    #[test]
    fn fixed_respects_the_row_budget() {
        let mut c = SpecController::new(SpecPolicy::Fixed(7));
        assert_eq!(c.pick_k(1, 7), 7);
        // the cap is a buffer-capacity invariant, so a pin cannot exceed it
        assert_eq!(c.pick_k(16, 2), 2);
    }

    #[test]
    fn yield_matches_leviathan_before_any_observation() {
        let c = SpecController::new(SpecPolicy::Auto);
        // K=0 is exactly one token: the plain decode baseline
        assert!((c.yield_of(0) - 1.0).abs() < 1e-9);
        // seed a=0.7, K=1 -> 1 + 0.7 = 1.7
        assert!((c.yield_of(1) - 1.7).abs() < 1e-9);
        // seed a=0.7, K=2 -> 1 + 0.7 + 0.49
        assert!((c.yield_of(2) - 2.19).abs() < 1e-9);
    }

    /// The block-drafter shape that the geometric form got wrong: a fat
    /// survival tail must make the deep K pay in the model exactly as it pays
    /// on the box.
    #[test]
    fn yield_follows_a_measured_survival_curve() {
        let mut c = SpecController::new(SpecPolicy::Auto);
        // the rival-logged DFlash2 curve at imax: S(i) for i = 1..7
        let curve = [0.78, 0.58, 0.43, 0.33, 0.27, 0.22, 0.19];
        // 1000 rounds at depth 7 whose accepted counts realize that curve
        let mut t = RoundTally::default();
        let n = 1000usize;
        let mut prev = n;
        let mut accepted_exact = Vec::new();
        for (i, s) in curve.iter().enumerate() {
            let surv = (s * n as f64).round() as usize;
            // (prev - surv) rounds stopped at exactly i accepted
            accepted_exact.push((i, prev - surv));
            prev = surv;
        }
        accepted_exact.push((7, prev));
        for (a, cnt) in accepted_exact {
            for _ in 0..cnt {
                t.book(7, a);
            }
        }
        c.observe(32, 7, 0.08, t);
        // 1 + sum(curve) = 3.80, the rival's "mean acceptance length 3.82"
        assert!((c.yield_of(7) - 3.80).abs() < 0.02, "got {}", c.yield_of(7));
        // and K=4 is worth 1 + 2.12 = 3.12 - a 22% step to K=7, not 2.6%
        assert!((c.yield_of(4) - 3.12).abs() < 0.02, "got {}", c.yield_of(4));
        // unseen depth 8 continues at the last ratio (0.19/0.22)
        assert!(c.yield_of(8) > c.yield_of(7) && c.yield_of(8) < c.yield_of(7) + 0.19);
    }

    /// Drive the controller against a synthetic box where speculation is a
    /// clear win, and check it finds the deep end.
    #[test]
    fn auto_climbs_when_drafts_land() {
        let mut c = SpecController::new(SpecPolicy::Auto);
        for _ in 0..4000 {
            let k = c.pick_k(1, 7);
            // cheap verify rows (the c1 regime: weights stream anyway) and a
            // drafter that hits ~90% of the time
            let secs = 0.010 + 0.0005 * k as f64;
            let accepted = (k as f64 * 0.9).round() as usize;
            c.observe(1, k, secs, RoundTally::one(k, accepted));
        }
        assert!(
            c.pick_k(1, 7) >= 4,
            "should ride deep chains, got {}",
            c.pick_k(1, 7)
        );
    }

    /// The case the hand-tuned ladder gets wrong: same model, heavy load, where
    /// every draft row is real compute and the drafter is mediocre. The
    /// controller has to give up on its own.
    #[test]
    fn auto_backs_off_to_zero_under_load() {
        let mut c = SpecController::new(SpecPolicy::Auto);
        for _ in 0..4000 {
            let k = c.pick_k(32, 7);
            // verify rows are expensive at c32, and drafts mostly miss
            let secs = 0.020 + 0.020 * k as f64;
            let accepted = (k as f64 * 0.25).round() as usize;
            c.observe(32, k, secs, RoundTally::one(k, accepted));
        }
        assert_eq!(
            c.pick_k(32, 7),
            0,
            "speculation loses here; K must fall to 0"
        );
    }

    /// Both regimes at once on one controller: the buckets must not bleed into
    /// each other, or a busy server would poison the idle-path decision.
    #[test]
    fn buckets_hold_independent_verdicts() {
        let mut c = SpecController::new(SpecPolicy::Auto);
        for _ in 0..4000 {
            let k1 = c.pick_k(1, 7);
            let a1 = (k1 as f64 * 0.9).round() as usize;
            c.observe(1, k1, 0.010 + 0.0005 * k1 as f64, RoundTally::one(k1, a1));
            let k32 = c.pick_k(32, 7);
            let a32 = (k32 as f64 * 0.25).round() as usize;
            c.observe(
                32,
                k32,
                0.020 + 0.020 * k32 as f64,
                RoundTally::one(k32, a32),
            );
        }
        assert!(c.pick_k(1, 7) >= 4, "c1 should still speculate");
        assert_eq!(c.pick_k(32, 7), 0, "c32 should still decline");
    }

    /// The regression this rework exists for. Settling on a depth used to
    /// starve the depth above it: `tries[i]` only advances when a round
    /// drafts at least i deep, so the arm at k+1 stopped receiving evidence,
    /// kept whatever stale estimate it had, and went on losing - an incumbent
    /// that reinforces itself. Pooling kills it by construction: every round
    /// updates the shared acceptance rate, so an UNDRAFTED depth is priced by
    /// live evidence, not by a memory.
    #[test]
    fn a_depth_nobody_drafts_still_tracks_the_drafter() {
        let mut c = SpecController::new(SpecPolicy::Auto);
        // a long run of shallow rounds that all accept - nothing ever drafts
        // deeper than 2, so depths 3..7 receive no direct evidence at all
        for _ in 0..500 {
            c.observe(4, 2, 0.012, RoundTally::one(2, 2));
        }
        assert_eq!(c.tri[7], 0.0, "depth 7 must have had no direct evidence");
        let optimistic = c.surv_at(7);
        assert!(
            optimistic > 0.5,
            "a drafter landing everything must not leave depth 7 pessimistic: {optimistic}"
        );

        // now the same drafter goes bad, still only ever drafting 2 deep
        for _ in 0..500 {
            c.observe(4, 2, 0.012, RoundTally::one(2, 0));
        }
        assert_eq!(c.tri[7], 0.0, "still no direct evidence at depth 7");
        let pessimistic = c.surv_at(7);
        assert!(
            pessimistic < optimistic / 2.0,
            "depth 7 must follow the drafter it never drafted at: {optimistic} -> {pessimistic}"
        );
    }

    /// The pooled rate must converge on the drafter's true per-token
    /// acceptance, from rounds of varying depth, with no depth privileged.
    ///
    /// (That the curve equals Leviathan's closed form when nothing has been
    /// observed is already covered by
    /// `yield_matches_leviathan_before_any_observation`. It cannot be checked
    /// against a feed like this one: a deterministic accept-4-reject-5 stream
    /// is not geometric, and the per-position evidence rightly models it more
    /// sharply than any single alpha could - which is the entire reason the
    /// per-position layer exists on top of the pooled one.)
    #[test]
    fn the_pooled_rate_converges_on_the_true_acceptance() {
        // a cheap deterministic LCG: geometric draws at p = 0.8, so the
        // stream really is the model's own assumption
        let mut rng: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((rng >> 33) as f64) / ((1u64 << 31) as f64)
        };
        let mut c = SpecController::new(SpecPolicy::Auto);
        for _ in 0..4000 {
            let d = 5usize;
            let mut a = 0usize;
            while a < d && next() < 0.8 {
                a += 1;
            }
            c.observe(4, d, 0.012, RoundTally::one(d, a));
        }
        let alpha = c.alpha();
        assert!(
            (0.75..0.85).contains(&alpha),
            "pooled rate should find the drafter's 0.8: {alpha}"
        );
    }

    /// A round where every draft was accepted is CENSORED, not a failure -
    /// the next position was never tested. Counting it as a rejection is how
    /// a drafter that is landing everything gets talked out of drafting.
    #[test]
    fn a_fully_accepted_round_is_not_evidence_of_rejection() {
        let mut all_good = SpecController::new(SpecPolicy::Auto);
        let mut sometimes = SpecController::new(SpecPolicy::Auto);
        for _ in 0..300 {
            all_good.observe(4, 4, 0.012, RoundTally::one(4, 4));
            sometimes.observe(4, 4, 0.012, RoundTally::one(4, 3));
        }
        assert!(
            all_good.alpha() > 0.97,
            "never stopped, so nothing observed a rejection: {}",
            all_good.alpha()
        );
        assert!(
            sometimes.alpha() < all_good.alpha(),
            "a stopped round IS evidence: {} vs {}",
            sometimes.alpha(),
            all_good.alpha()
        );
    }

    /// Acceptance is not stationary - it swings with content - so the
    /// evidence has to forget. A drafter that goes bad must be believed
    /// within a bounded number of rounds, not averaged against its history
    /// forever.
    #[test]
    fn the_estimate_forgets_a_drafter_that_changed() {
        let mut c = SpecController::new(SpecPolicy::Auto);
        for _ in 0..3000 {
            c.observe(4, 4, 0.012, RoundTally::one(4, 4));
        }
        assert!(c.alpha() > 0.95);
        // inside the stated ~200-round window it is already most of the way
        for _ in 0..300 {
            c.observe(4, 4, 0.012, RoundTally::one(4, 0));
        }
        let mid = c.alpha();
        assert!(
            mid < 0.35,
            "300 rounds should mostly forget a 3000-round run: {mid}"
        );
        for _ in 0..300 {
            c.observe(4, 4, 0.012, RoundTally::one(4, 0));
        }
        assert!(
            c.alpha() < 0.15,
            "and then settle on the drafter it has: {}",
            c.alpha()
        );
    }

    #[test]
    fn cold_buckets_anchor_at_the_baseline_the_cap_and_its_midpoint() {
        let mut c = SpecController::new(SpecPolicy::Auto);
        let mut seen = [0u32; 8];
        for _ in 0..15 {
            let k = c.pick_k(4, 7);
            seen[k] += 1;
            c.observe(4, k, 0.01 + 0.001 * k as f64, RoundTally::one(k, k));
        }
        // the warm start measures exactly the three anchors, MIN_SAMPLES each,
        // and the baseline is one of them: until k=0 is measured, every
        // goodput comparison runs against an extrapolation of the round's own
        // line (see the note in pick_auto)
        assert_eq!(seen[0], 5, "baseline anchor: {seen:?}");
        assert_eq!(seen[7], 5, "cap anchor: {seen:?}");
        assert_eq!(seen[3], 5, "midpoint anchor: {seen:?}");
        assert_eq!(seen.iter().sum::<u32>(), 15);
        // and from then on the argmax runs on the fitted line - a perfect
        // drafter with near-flat latency rides the cap, no further sweep
        let k = c.pick_k(4, 7);
        assert_eq!(k, 7, "should ride the cap on the fitted line, got {k}");
    }

    /// A round with a big fixed cost has to lose to not speculating, however
    /// well its drafts land. This is the Flash-Next pathology of 2026-09-17:
    /// at 8 live slots its verify round (which copies GDN and PLE state per
    /// round) measured 82/107/131 ms at k=1/2/3 against a 26 ms dense tick,
    /// and 79% of drafts were accepted - yet the serve ran 40% under its own
    /// no-spec throughput, because k=0 had never been measured and the
    /// affine fit through the speculative cells priced it at ~59 ms. With the
    /// baseline anchored, the goodput comparison is against the real thing.
    #[test]
    fn a_round_whose_fixed_cost_dominates_backs_off_to_zero() {
        let mut c = SpecController::new(SpecPolicy::Auto);
        // t(0) = 26 ms; a round costs 59 ms of setup + 23 ms a draft row
        let lat = |k: usize| {
            if k == 0 {
                0.026
            } else {
                0.059 + 0.023 * k as f64
            }
        };
        // drafts land 79% of the time, one accepted per drafted row
        let tally = |k: usize| RoundTally::one(k, (k * 79) / 100);
        for _ in 0..60 {
            let k = c.pick_k(8, 3);
            c.observe(8, k, lat(k), tally(k));
        }
        // goodput at k=0 is 1/26ms = 38/s; the best speculative option is
        // 1+3*0.79 = 3.37 tokens per 128 ms = 26/s. Not speculating wins.
        let k = c.pick_k(8, 3);
        assert_eq!(k, 0, "should stop speculating, got k={k}");
    }
}
