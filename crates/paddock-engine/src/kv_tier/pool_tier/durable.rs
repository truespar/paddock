//! What reaches the disk, and in what order - the disk tier's admission.
//!
//! Every T1 store used to be written through to disk from a queue that
//! dropped its oldest entries when full. Measured on a fleet of agent
//! sessions (Qwen3.8-27B, 16 GB RAM tier): 4% of the bytes written were ever
//! read back, most of them state checkpoints a deeper one had already
//! superseded, and after a restart 5 of 24 sessions resumed - the queue had
//! dropped chain heads, and a probe walks a chain from its head, so one
//! missing run left everything deeper unreachable.
//!
//! The plan here writes per chain instead:
//!
//! - **Targets are tips.** A state blob no deeper blob supersedes - or one
//!   where chains branch (a shared system prompt: every new session resumes
//!   there) - and, for a KV-only family, a run no other T1 run continues.
//! - **Chain order.** A target's runs go root-first, each only behind the
//!   ones before it, then its blob's shards. A link that is neither on disk
//!   nor in T1 makes the target unpersistable as it stands (everything past
//!   a hole is unreachable), so it is skipped, not written.
//! - **Idle first.** A target is written once untouched for [`IDLE`]: a live
//!   session's next turn supersedes its tip, so writing it sooner mostly
//!   writes what the next turn replaces - and the most idle targets are T1's
//!   next victims anyway. A tip superseded mid-plan stops after its runs
//!   (they are the deeper blob's chain prefix) and never ships its state.
//! - **Shutdown** writes every remaining tip, bounded by [`EXIT_BUDGET`].
//!
//! Disk writes are synchronous on the engine thread, so a pass is small
//! ([`PER_PASS`] extents, what the old write-through drained) and runs only
//! while no disk read is queued or in flight (`XferSink::disk_ready`).

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};

use super::*;

/// Untouched this long, a tip is written.
pub(super) const IDLE: Duration = Duration::from_secs(30);
/// Extents one pass writes at most.
const PER_PASS: usize = 4;
/// The shutdown flush's ceiling - a slow disk loses the rest, never a
/// stop the supervisor gives up on.
pub(super) const EXIT_BUDGET: Duration = Duration::from_secs(10);
/// How long the exit copy waits for its stores to land.
const FLUSH_WAIT: Duration = Duration::from_secs(5);
/// Idle marks, and candidate scans that found nothing, at most this often.
const RESCAN: Duration = Duration::from_secs(1);
/// What eviction may write to keep content that exists nowhere else
/// ([`PoolTier::persist_before_evict`]): bursts up to this, paid back at
/// [`EVICT_WRITE_RATE`]. Those writes are synchronous on the engine thread,
/// so bounded - by a rate, not per pump: one eviction freeing a session's
/// worth of T1 can take two tips at once, and a 512 MiB per-pump cap wrote
/// the first and dropped the second (measured: 22K-token Qwen3.8-27B tips,
/// ~480 MB each, ~90 ms to write) - against a recompute of that whole
/// session when it comes back.
pub(super) const EVICT_WRITE_BURST: u64 = 2 << 30;
/// Bytes per second the burst refills at - sustained churn keeps a
/// fraction of the engine thread, never all of it.
const EVICT_WRITE_RATE: f64 = (512u64 << 20) as f64;

/// A T1 run's predecessor on its chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Parent {
    /// The chain's first run.
    Head,
    Key(LogicalKey),
    /// Not known (a run read back from disk - durable by construction).
    Unknown,
}

/// The run before `path[lo..lo + r]`.
pub(super) fn run_parent(path: &[crate::paged_radix::LruPathEntry], lo: usize) -> Parent {
    if lo == 0 {
        Parent::Head
    } else {
        path[lo - 1].tkey.map_or(Parent::Unknown, Parent::Key)
    }
}

#[derive(Debug, Clone, Copy)]
enum Step {
    Run(LogicalKey),
    /// (blob, shard)
    Shard(LogicalKey, LogicalKey),
}

impl Step {
    fn key(&self) -> LogicalKey {
        match *self {
            Step::Run(k) | Step::Shard(_, k) => k,
        }
    }
}

enum Next {
    /// On disk now (true: this pass wrote it).
    Done(bool),
    /// The disk refused for now (budget, reserve, paused, a failed write):
    /// try this step again on a later pass.
    Retry,
    /// The target cannot finish as it stands.
    Abort,
}

/// The run half of RAM eviction's order: what eviction would not lose goes
/// first - runs with a disk copy (they stay restorable, at disk speed) -
/// then runs nothing in T1 continues, least recently used first. A run
/// another T1 run continues waits until its continuation goes, so a chain
/// with no disk copy shrinks from its tail. Plain LRU took a chain's head
/// first (it carries the oldest stamp: chains store root-first) and left
/// everything after it unreachable on every tier.
pub(super) struct RunVictims {
    /// (0 = on disk / 1 = RAM-only leaf, last use, key), first out first
    order: BTreeSet<(u8, u64, LogicalKey)>,
    /// T1 runs continuing each run
    kids: HashMap<LogicalKey, u32>,
    /// RAM-only runs something still continues
    waiting: HashSet<LogicalKey>,
}

#[derive(Default)]
pub(super) struct Durable {
    plan: VecDeque<Step>,
    /// The target the plan writes, and its LRU stamp when planned.
    target: Option<(LogicalKey, u64)>,
    /// (when, the tier's LRU clock then): content stamped at or before a
    /// mark has been untouched since that mark's time.
    marks: VecDeque<(Instant, u64)>,
    /// Targets handled at a stamp - written, or found unpersistable - left
    /// alone until something touches them again.
    settled: HashMap<LogicalKey, u64>,
    /// The last scan that found nothing to write.
    dry_at: Option<Instant>,
    /// Eviction's write debt ([`EVICT_WRITE_BURST`]), as of `debt_at`.
    evict_debt: u64,
    debt_at: Option<Instant>,
}

impl Durable {
    /// Extents queued for disk (the export's "pending durable writes").
    pub(super) fn queued(&self) -> usize {
        self.plan.len()
    }

    /// The newest LRU stamp that is at least [`IDLE`] old, if any yet.
    fn idle_cutoff(&mut self, now: Instant, lru: u64) -> Option<u64> {
        if self
            .marks
            .back()
            .is_none_or(|&(t, _)| now.duration_since(t) >= RESCAN)
        {
            self.marks.push_back((now, lru));
        }
        while self.marks.len() >= 2 && now.duration_since(self.marks[1].0) >= IDLE {
            self.marks.pop_front();
        }
        match self.marks.front() {
            Some(&(t, l)) if now.duration_since(t) >= IDLE => Some(l),
            _ => None,
        }
    }
}

impl<T: XferSink> PoolTier<T> {
    /// One pass of the durability plan. `force` (shutdown): every tip, idle
    /// or not, until done or `deadline`. Returns extents written.
    pub(super) fn persist_pass(&mut self, now: Instant, force: bool, deadline: Instant) -> usize {
        if self.tripped || !self.transport.disk_ready() {
            return 0;
        }
        let lru = self.lru;
        let cutoff = if force {
            Some(u64::MAX)
        } else {
            self.durable.idle_cutoff(now, lru)
        };
        let mut wrote = 0;
        let mut steps = 0;
        while force || steps < PER_PASS {
            if force && Instant::now() >= deadline {
                break;
            }
            if self.durable.plan.is_empty() {
                self.settle_target();
                let Some(cutoff) = cutoff else { break };
                if !force
                    && self
                        .durable
                        .dry_at
                        .is_some_and(|t| now.duration_since(t) < RESCAN)
                {
                    break;
                }
                if !self.plan_next(cutoff) {
                    self.durable.dry_at = Some(now);
                    break;
                }
                self.durable.dry_at = None;
            }
            let Some(step) = self.durable.plan.pop_front() else {
                break;
            };
            steps += 1;
            match self.persist_step(step) {
                Next::Done(w) => wrote += usize::from(w),
                Next::Retry => {
                    self.durable.plan.push_front(step);
                    break;
                }
                Next::Abort => {
                    self.durable.plan.clear();
                    self.settle_target();
                }
            }
            if !self.transport.disk_ready() {
                break;
            }
        }
        wrote
    }

    /// The finished (or abandoned) target is handled at its stamp.
    fn settle_target(&mut self) {
        if let Some((k, stamp)) = self.durable.target.take() {
            self.durable.settled.insert(k, stamp);
        }
    }

    /// A family that resumes only through state blobs (its runs alone
    /// restore nothing).
    pub(super) fn hybrid(&self) -> bool {
        self.blob_geom.is_some() || !self.aux_meta.is_empty()
    }

    /// Write-back at eviction: T1 is about to drop the last copy of
    /// `target` (a state blob nothing supersedes, or a KV-only run with no
    /// disk copy). Write its chain now, root-first by the plan's rules, if
    /// the disk takes writes and the budget holds it - the plan waits for a
    /// tip to idle, and under churn T1 can evict a live session's tip before
    /// it does - the session then recomputes when it comes back. True when
    /// `target` is on disk now.
    ///
    /// A disk read in flight does not stop it (`XferSink::disk_writable`):
    /// the no-duplex rule makes the background plan wait for reads, but
    /// here waiting is not on offer - the write slows one restore, skipping
    /// it loses the session. Live, two tips went that way in one eviction
    /// that a restore's own make-room pass had started.
    pub(super) fn persist_before_evict(&mut self, target: LogicalKey) -> bool {
        if self.tripped || !self.transport.disk_writable() {
            tracing::debug!(
                depth = self.blob_depth_tokens(&target),
                "tier durable: no disk write at eviction - state dropped"
            );
            return false;
        }
        // a hole: nothing past it is reachable, so nothing to save
        let Some(steps) = self.chain_steps(target) else {
            return false;
        };
        let bytes: u64 = steps
            .iter()
            .map(|s| self.catalog.ready_bytes(&s.key(), Tier::Ram).unwrap_or(0))
            .sum();
        let now = Instant::now();
        let d = &mut self.durable;
        if let Some(at) = d.debt_at {
            let paid = now.duration_since(at).as_secs_f64() * EVICT_WRITE_RATE;
            d.evict_debt = d.evict_debt.saturating_sub(paid as u64);
        }
        d.debt_at = Some(now);
        // a clear budget takes any one target, however big
        if d.evict_debt > 0 && d.evict_debt + bytes > EVICT_WRITE_BURST {
            tracing::debug!(
                bytes,
                debt = d.evict_debt,
                "tier durable: eviction writes over budget - state dropped"
            );
            return false;
        }
        d.evict_debt += bytes;
        // one group commit for the chain: it runs on the engine thread, and
        // two flushes per extent made a 400 MB chain a ~1 s stall
        let mut items = Vec::with_capacity(steps.len());
        for step in steps {
            match self.prepare(step) {
                Ok(item) => items.push(item),
                Err(Next::Done(_)) => {}
                Err(_) => return false,
            }
        }
        let landed = self
            .transport
            .persist_batch(&items)
            .iter()
            .all(|p| matches!(p, Persist::Written | Persist::Durable));
        if landed {
            tracing::debug!(
                bytes,
                ms = now.elapsed().as_millis() as u64,
                "KV tier: wrote state to disk before evicting its last copy"
            );
        }
        landed
    }

    /// A blob's boundary in tokens (its bound runs' span; 0 for a run).
    pub(super) fn blob_depth_tokens(&self, b: &LogicalKey) -> usize {
        self.blob_runs
            .get(b)
            .map_or(0, |r| r.len() * self.run_blocks * BLOCK_TOKENS)
    }

    pub(super) fn durable_copy(&self, key: &LogicalKey) -> bool {
        self.transport.t2_entry(&key.0).is_some()
    }

    /// Every shard of blob `b` (of `shards`) has a disk copy.
    pub(super) fn blob_on_disk(&self, b: &LogicalKey, shards: usize) -> bool {
        (0..shards).all(|i| self.durable_copy(&b.child_bytes("aux", &(i as u32).to_le_bytes())))
    }

    /// Whether dropping blob `b` from T1 loses anything worth keeping: not
    /// when a deeper blob supersedes it, nor when the disk holds it whole.
    pub(super) fn blob_loss(&self, b: &LogicalKey) -> bool {
        let shards = self.aux_meta.get(b).map_or(0, |m| m.shards);
        self.blob_worth_keeping(b) && !self.blob_on_disk(b, shards)
    }

    /// RAM eviction's run order for the runs `eligible` admits (see
    /// [`RunVictims`]).
    pub(super) fn run_victims(&self, eligible: impl Fn(&LogicalKey) -> bool) -> RunVictims {
        let mut kids: HashMap<LogicalKey, u32> = HashMap::new();
        for m in self.runs.values() {
            if let Parent::Key(p) = m.parent {
                *kids.entry(p).or_insert(0) += 1;
            }
        }
        let mut v = RunVictims {
            order: BTreeSet::new(),
            kids,
            waiting: HashSet::new(),
        };
        let keys: Vec<LogicalKey> = self.runs.keys().filter(|k| eligible(k)).copied().collect();
        for k in keys {
            self.queue_victim(&mut v, k);
        }
        v
    }

    /// Queue run `k` by what evicting it would lose: nothing (a disk copy),
    /// or - for a run nothing in T1 continues - its content; a run another
    /// T1 run continues waits until that one goes.
    pub(super) fn queue_victim(&self, v: &mut RunVictims, k: LogicalKey) {
        let Some(m) = self.runs.get(&k) else {
            return;
        };
        if self.durable_copy(&k) {
            v.order.insert((0, m.last_used, k));
        } else if v.kids.get(&k).copied().unwrap_or(0) == 0 {
            v.order.insert((1, m.last_used, k));
        } else {
            v.waiting.insert(k);
        }
    }

    /// Evict the next queued run; a parent it was the last continuation of
    /// becomes a candidate. None when the queue is empty.
    pub(super) fn evict_next_victim(&mut self, v: &mut RunVictims) -> Option<LogicalKey> {
        let (class, _, k) = v.order.pop_first()?;
        // a RAM-only leaf of a KV-only family: its chain's last copy (a
        // hybrid's lone runs restore nothing, so they are not worth a write)
        if class == 1 && !self.hybrid() {
            self.persist_before_evict(k);
        }
        let parent = self.runs.get(&k).map(|m| m.parent);
        if self.evict_run_t1(k)
            && let Some(Parent::Key(p)) = parent
            && let Some(n) = v.kids.get_mut(&p)
        {
            *n = n.saturating_sub(1);
            if *n == 0 && v.waiting.remove(&p) {
                self.queue_victim(v, p);
            }
        }
        Some(k)
    }

    /// Plan the most idle target that still has something to write. False
    /// when none does.
    fn plan_next(&mut self, cutoff: u64) -> bool {
        // forget handled targets that left T1
        let (runs, aux) = (&self.runs, &self.aux_meta);
        self.durable
            .settled
            .retain(|k, _| runs.contains_key(k) || aux.contains_key(k));
        loop {
            let Some((target, stamp)) = self.next_target(cutoff) else {
                return false;
            };
            match self.chain_steps(target) {
                Some(steps) if !steps.is_empty() => {
                    tracing::debug!(
                        depth = self.blob_depth_tokens(&target),
                        extents = steps.len(),
                        idle_stamps = self.lru.saturating_sub(stamp),
                        "tier durable: tip planned"
                    );
                    self.durable.plan = steps.into();
                    self.durable.target = Some((target, stamp));
                    return true;
                }
                // all on disk already, or a hole: nothing to do at this stamp
                _ => {
                    self.durable.settled.insert(target, stamp);
                }
            }
        }
    }

    /// The most idle unhandled tip at or under `cutoff`, with its stamp.
    fn next_target(&self, cutoff: u64) -> Option<(LogicalKey, u64)> {
        let unhandled = |k: &LogicalKey, stamp: u64| {
            stamp <= cutoff && self.durable.settled.get(k) != Some(&stamp)
        };
        if self.hybrid() {
            // a hybrid restores only through a blob: blobs are the targets
            return self
                .aux_meta
                .iter()
                .filter(|&(k, m)| unhandled(k, m.last_used) && self.blob_worth_keeping(k))
                .min_by_key(|(_, m)| m.last_used)
                .map(|(k, m)| (*k, m.last_used));
        }
        let continued: HashSet<LogicalKey> = self
            .runs
            .values()
            .filter_map(|m| match m.parent {
                Parent::Key(p) => Some(p),
                _ => None,
            })
            .collect();
        self.runs
            .iter()
            .filter(|&(k, m)| !continued.contains(k) && unhandled(k, m.last_used))
            .min_by_key(|(_, m)| m.last_used)
            .map(|(k, m)| (*k, m.last_used))
    }

    /// A blob worth a disk copy: no deeper blob supersedes it, or chains
    /// branch at it - a shared prompt's state, where every new session on
    /// that prompt resumes (a restart that kept only the session tips would
    /// recompute the shared prompt for each of them) - or it is its chain's
    /// deepest user-turn anchor. The reply path past an anchor does not
    /// survive the next user message (templates drop earlier turns'
    /// reasoning, so the conversation re-renders from the anchor on), nor a
    /// request cut off mid-reply and sent again: that resumes at the anchor
    /// or recomputes. Writing only the deepest blob lost exactly those - a
    /// stop under four live turns kept their reply paths and three of them
    /// recomputed whole after the restart.
    pub(super) fn blob_worth_keeping(&self, b: &LogicalKey) -> bool {
        if self.run_owners.get(b).is_none_or(|&n| n < 2) {
            return true;
        }
        let Some(own) = self.blob_runs.get(b) else {
            return false;
        };
        // a blob's key is its last run's: deeper chains through it carry it
        // at the same index, and their next runs say whether they part there
        if own.last() != Some(b) {
            return false;
        }
        let i = own.len();
        let mut next = HashSet::new();
        let mut deeper_anchor = false;
        for (d, runs) in &self.blob_runs {
            if d != b
                && runs.get(i - 1) == Some(b)
                && let Some(m) = self.aux_meta.get(d)
                && let Some(s) = runs.get(i)
            {
                next.insert(*s);
                if next.len() >= 2 {
                    return true;
                }
                deeper_anchor |= m.anchor;
            }
        }
        self.aux_meta.get(b).is_some_and(|m| m.anchor) && !deeper_anchor
    }

    /// What `target` still needs on disk, root-first; None when a link is
    /// neither on disk nor in T1 (the chain past it is unreachable).
    fn chain_steps(&self, target: LogicalKey) -> Option<Vec<Step>> {
        let in_t1 = |k: &LogicalKey| self.catalog.ready_loc(k, Tier::Ram).is_some();
        let mut steps = Vec::new();
        let hole = |what: &str, at: usize| {
            tracing::debug!(
                depth = self.blob_depth_tokens(&target),
                what,
                at,
                "tier durable: tip unpersistable (a link is on no tier)"
            );
            None
        };
        if let Some(m) = self.aux_meta.get(&target) {
            let Some(runs) = self.blob_runs.get(&target) else {
                return hole("unbound", 0);
            };
            for (i, r) in runs.iter().enumerate() {
                if self.durable_copy(r) {
                    continue;
                }
                if !in_t1(r) {
                    return hole("run", i);
                }
                steps.push(Step::Run(*r));
            }
            for i in 0..m.shards {
                let s = target.child_bytes("aux", &(i as u32).to_le_bytes());
                if self.durable_copy(&s) {
                    continue;
                }
                if !in_t1(&s) {
                    return hole("shard", i);
                }
                steps.push(Step::Shard(target, s));
            }
            return Some(steps);
        }
        // KV-only: walk back from the leaf to the first run already on disk
        let mut k = target;
        for _ in 0..=self.runs.len() {
            if self.durable_copy(&k) {
                break;
            }
            if !in_t1(&k) {
                return None;
            }
            steps.push(Step::Run(k));
            match self.runs.get(&k).map(|m| m.parent) {
                Some(Parent::Key(p)) => k = p,
                Some(Parent::Head) => break,
                _ => return None,
            }
        }
        steps.reverse();
        Some(steps)
    }

    /// What persisting `step` hands the transport: (key, T1 loc, bytes,
    /// checksum), or how the step ends without a write.
    fn prepare(
        &self,
        step: Step,
    ) -> Result<(LogicalKey, Loc, u64, super::super::digest::Checksum), Next> {
        let key = match step {
            Step::Run(k) => k,
            Step::Shard(b, s) => {
                // superseded while its runs went out: those serve the deeper
                // blob's chain, this state no longer serves anything
                if !self.aux_meta.contains_key(&b) || !self.blob_worth_keeping(&b) {
                    return Err(Next::Abort);
                }
                s
            }
        };
        let (Some(loc), Some(bytes), Some(sum)) = (
            self.catalog.ready_loc(&key, Tier::Ram),
            self.catalog.ready_bytes(&key, Tier::Ram),
            self.catalog.ready_checksum(&key, Tier::Ram),
        ) else {
            // left T1 before its turn: fine if a copy is on disk, a hole if not
            return Err(if self.durable_copy(&key) {
                Next::Done(false)
            } else {
                Next::Abort
            });
        };
        Ok((key, loc, bytes, sum))
    }

    fn persist_step(&mut self, step: Step) -> Next {
        let (key, loc, bytes, sum) = match self.prepare(step) {
            Ok(item) => item,
            Err(next) => return next,
        };
        match self.transport.persist(&key, loc, bytes, sum) {
            Persist::Written => Next::Done(true),
            Persist::Durable => Next::Done(false),
            Persist::Declined | Persist::Failed | Persist::NoDisk => Next::Retry,
            Persist::Gone => Next::Abort,
        }
    }

    /// Shutdown, before the device goes: everything the GPU still holds -
    /// every live checkpoint with the runs it resumes over, every cached
    /// chain - copied into T1, then every tip written to disk within
    /// [`EXIT_BUDGET`]. The exit flush alone wrote only what T1 already
    /// held, and a session's newest state usually is not there yet: the
    /// slack mirror ships one checkpoint a pass, so a session whose last
    /// turn ended just before shutdown lost it and recomputed every token
    /// when it came back. The copy is the same gather the mirror runs,
    /// bounded by [`FLUSH_WAIT`] for the stores to land.
    pub fn flush_live(
        &mut self,
        radix: &mut PagedRadix,
        pool: &mut KvPool,
        after: &mut dyn FnMut() -> Option<CudaEvent>,
        state: Option<(u64, u64)>,
    ) {
        let t0 = Instant::now();
        let (runs, blobs) = self.mirror_slack_t1(radix, pool, after(), usize::MAX, state, true);
        while !self.deferred.is_empty() && t0.elapsed() < FLUSH_WAIT {
            self.pump_completions(radix, pool);
            std::thread::sleep(Duration::from_micros(200));
        }
        tracing::info!(
            runs,
            blobs,
            landed_ms = t0.elapsed().as_millis() as u64,
            still_in_flight = self.deferred.len(),
            "KV tier: copied the GPU's live state into RAM for the exit flush"
        );
        self.persist_at_exit();
    }

    /// Shutdown: every tip still RAM-only, within [`EXIT_BUDGET`].
    pub(super) fn persist_at_exit(&mut self) {
        let t0 = Instant::now();
        let n = self.persist_pass(t0, true, t0 + EXIT_BUDGET);
        if n > 0 {
            tracing::info!(
                extents = n,
                ms = t0.elapsed().as_millis() as u64,
                "KV disk cache: wrote the sessions' latest state at shutdown"
            );
        }
    }

    /// Tests: eviction owes `bytes` of writes, as of `ago`.
    #[cfg(test)]
    pub(super) fn owe_eviction_writes(&mut self, bytes: u64, ago: Duration) {
        self.durable.evict_debt = bytes;
        self.durable.debt_at = Some(Instant::now() - ago);
    }

    /// Tests: write everything a forced pass would, now.
    #[cfg(test)]
    pub(super) fn persist_all(&mut self) -> usize {
        let now = Instant::now();
        self.persist_pass(now, true, now + EXIT_BUDGET)
    }
}
