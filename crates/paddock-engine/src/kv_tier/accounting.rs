//! decision accountability: not counters, explanations.
//!
//! The design note is explicit that an occupancy gauge is not observability -
//! what an operator needs is *why the tier did what it did*: how many lookups
//! were eligible, how many could have been served, which arm the cost model
//! chose and whether it was right, how many bytes moved versus how many bytes
//! the request actually needed, and - the one alarm that matters - whether
//! repeated lookups are missing that should have hit.
//!
//! Everything here is single-threaded (the tier lives on the engine thread)
//! and costs an increment per decision. The report is assembled on demand,
//! once per scheduler pass, and copied out through the metrics atomics.

use std::collections::{HashSet, VecDeque};

/// Why a lookup did not turn into a restore. Every miss lands in exactly one
/// bucket - a miss with no reason is a bug in the instrumentation, not an
/// unexplained event, which is the whole point of the enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissReason {
    /// Nothing for this prefix in any tier - the ordinary cold case.
    Cold,
    /// The prefix is known but the resident part stops at or below what the
    /// GPU already holds, so a restore would deliver nothing new.
    NoNewTokens,
    /// The breaker is open: repeated IO or integrity failures took the tier
    /// out of service and serving continues on recompute.
    Tripped,
    /// We held this content and evicted it. This is the alarm: a warm
    /// workload whose repeats keep landing here is being served by a tier
    /// that is too small, or by an eviction policy picking the wrong victim.
    Ghost,
    /// The prefix's KV is held but not the model state a hybrid (DeltaNet,
    /// Mamba, sliding window) needs to resume on it: none was saved at a
    /// boundary the held KV reaches, or it was evicted. Blocks alone are
    /// worthless there, so this is a miss, not a hit.
    NoState,
}

/// How many recently-evicted keys the ghost set remembers. Sized to cover a
/// few agentic sessions' worth of chains without becoming a second cache:
/// 8192 keys is ~256 KB and answers "did we just throw this away?" for the
/// window where the answer is actionable. (The full ghost/TinyLFU sketch that
/// drives ADMISSION is 1b.2; this is the accounting half, and the same signal
/// it will be elected on.)
const GHOST_CAPACITY: usize = 8192;

/// Recently evicted keys, FIFO. Membership turns an ordinary miss into a
/// "should have hit", which is the difference between a tier that is idle
/// because the workload is cold and one that is idle because it is thrashing.
#[derive(Default)]
pub struct GhostSet {
    seen: HashSet<[u8; 32]>,
    order: VecDeque<[u8; 32]>,
}

impl GhostSet {
    pub fn record_eviction(&mut self, key: [u8; 32]) {
        if self.seen.insert(key) {
            self.order.push_back(key);
            if self.order.len() > GHOST_CAPACITY
                && let Some(old) = self.order.pop_front()
            {
                self.seen.remove(&old);
            }
        }
    }

    pub fn contains(&self, key: &[u8; 32]) -> bool {
        self.seen.contains(key)
    }

    pub fn len(&self) -> usize {
        self.seen.len()
    }

    pub fn is_empty(&self) -> bool {
        self.seen.is_empty()
    }
}

/// The decision ledger. Monotonic counters - deltas are the consumer's job,
/// so a restart is visible as a reset rather than hidden by smoothing.
#[derive(Debug, Clone, Copy, Default)]
pub struct TierDecisions {
    /// Lookups the tier was asked about at all.
    pub lookups: u64,
    /// Lookups where a restorable extension existed.
    pub hits: u64,
    pub miss_cold: u64,
    pub miss_no_new_tokens: u64,
    pub miss_tripped: u64,
    /// The alarm: missed on content we evicted (see [`MissReason::Ghost`]).
    pub miss_ghost: u64,
    /// KV held without the state to resume on it (see [`MissReason::NoState`]).
    pub miss_no_state: u64,
    /// Cost-model arms taken on a hit.
    pub elected_restore: u64,
    pub elected_recompute: u64,
    /// Restores that actually started (the request parked) versus those
    /// refused at reservation time - a park refusal means the block pool
    /// could not seat the destination, which is a capacity story, not an
    /// IO one.
    pub parked: u64,
    pub park_refused: u64,
    /// How parked restores ended.
    pub resolved_ok: u64,
    pub abandoned: u64,
    /// Where resolved restores read from.
    pub served_from_ram: u64,
    pub served_from_nvme: u64,
    /// Payload bytes a resolved restore delivered to the GPU.
    pub useful_bytes: u64,
    /// Payload bytes the tier moved for restores, delivered or not -
    /// `moved / useful` is the read amplification an operator can act on
    /// (a ratio above 1 means abandoned restores are burning bandwidth).
    pub moved_bytes: u64,
    /// Bytes written toward T1 by demotes and mirrors - the write side of
    /// the same question.
    pub stored_bytes: u64,
    /// T1 evictions whose durable copy was published as readable on disk
    /// instead of being lost - the difference between a write-through that
    /// pays off within the run and one that only helps after a restart.
    pub promoted_to_disk: u64,
}

impl TierDecisions {
    pub fn record_miss(&mut self, why: MissReason) {
        match why {
            MissReason::Cold => self.miss_cold += 1,
            MissReason::NoNewTokens => self.miss_no_new_tokens += 1,
            MissReason::Tripped => self.miss_tripped += 1,
            MissReason::Ghost => self.miss_ghost += 1,
            MissReason::NoState => self.miss_no_state += 1,
        }
    }

    /// Share of lookups that found something restorable, 0..1. None until
    /// anything was asked - an empty tier reports "no question asked yet",
    /// never a 0% hit rate it did not earn.
    pub fn hit_rate(&self) -> Option<f64> {
        (self.lookups > 0).then(|| self.hits as f64 / self.lookups as f64)
    }

    /// Bytes moved per byte delivered. 1.0 is perfect; higher means restores
    /// are being abandoned after their IO was already paid for.
    pub fn read_amplification(&self) -> Option<f64> {
        (self.useful_bytes > 0).then(|| self.moved_bytes as f64 / self.useful_bytes as f64)
    }

    /// Should an operator be told something is wrong? True when a meaningful
    /// number of lookups missed on content we ourselves threw away - the
    /// alarm condition, stated so that a legitimately cold or one-shot
    /// workload never trips it.
    pub fn ghost_alarm(&self) -> bool {
        self.miss_ghost >= 16 && self.miss_ghost * 4 > self.lookups
    }
}

/// One assembled view of the tier, for `/api/stats` and the Studio panel.
/// Plain numbers: the runner serializes it, the manager passes it through,
/// the panel explains it.
#[derive(Debug, Clone, Copy, Default)]
pub struct TierReport {
    pub decisions: TierDecisions,
    /// Occupancy split, bytes, per tier (reserved / in-flight / ready).
    pub t1_ready_bytes: u64,
    pub t1_in_flight_bytes: u64,
    pub t1_reserved_bytes: u64,
    pub t1_capacity_bytes: u64,
    pub t2_ready_bytes: u64,
    pub t2_capacity_bytes: u64,
    /// Chains resident in T1 right now.
    pub resident_runs: u64,
    /// Demotes in flight and restore tickets open.
    pub in_flight_demotes: u64,
    pub open_tickets: u64,
    /// Durable writes deferred to disk read slack.
    pub pending_durable_writes: u64,
    /// Health.
    pub tripped: bool,
    pub io_failures: u64,
    pub integrity_failures: u64,
    pub evictions: u64,
    pub single_flight_joins: u64,
    pub stale_completions: u64,
    /// Cost model: measured rates in bytes/us and its own honesty score.
    pub rate_ram_bpus: f64,
    pub rate_nvme_bpus: f64,
    pub prediction_error_pct: Option<f64>,
    /// T2 device geometry as MEASURED at open, and today's endurance
    /// spend against the elected budget.
    pub device_read_gbs: f64,
    pub device_write_gbs: f64,
    pub device_unbuffered: bool,
    pub t2_written_day_bytes: u64,
    /// Ghost-set occupancy - context for the alarm.
    pub ghost_keys: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn k(n: u8) -> [u8; 32] {
        [n; 32]
    }

    #[test]
    fn ghost_set_is_bounded_and_fifo() {
        let mut g = GhostSet::default();
        for i in 0..(GHOST_CAPACITY + 100) {
            let mut key = [0u8; 32];
            key[..8].copy_from_slice(&(i as u64).to_le_bytes());
            g.record_eviction(key);
        }
        assert_eq!(g.len(), GHOST_CAPACITY, "ghost set must stay bounded");
        let mut first = [0u8; 32];
        first[..8].copy_from_slice(&0u64.to_le_bytes());
        assert!(!g.contains(&first), "oldest key should have aged out");
        let mut last = [0u8; 32];
        last[..8].copy_from_slice(&((GHOST_CAPACITY + 99) as u64).to_le_bytes());
        assert!(g.contains(&last));
    }

    #[test]
    fn re_evicting_the_same_key_does_not_grow_the_set() {
        let mut g = GhostSet::default();
        for _ in 0..100 {
            g.record_eviction(k(7));
        }
        assert_eq!(g.len(), 1);
    }

    /// A cold workload must never trip the alarm - an occupied tier with no
    /// hits is legitimate for one-shot traffic, and an alarm that cries wolf
    /// there is worse than no alarm.
    #[test]
    fn the_alarm_separates_thrashing_from_cold() {
        let mut d = TierDecisions {
            lookups: 1000,
            ..Default::default()
        };
        for _ in 0..1000 {
            d.record_miss(MissReason::Cold);
        }
        assert!(!d.ghost_alarm(), "cold traffic is not an alarm");
        assert_eq!(d.hit_rate(), Some(0.0));

        let mut t = TierDecisions {
            lookups: 100,
            ..Default::default()
        };
        for _ in 0..40 {
            t.record_miss(MissReason::Ghost);
        }
        assert!(
            t.ghost_alarm(),
            "40% of lookups missing on evicted content is thrashing"
        );

        // and a handful of ghosts in a big healthy sample stays quiet
        let mut q = TierDecisions {
            lookups: 10_000,
            ..Default::default()
        };
        for _ in 0..20 {
            q.record_miss(MissReason::Ghost);
        }
        assert!(!q.ghost_alarm());
    }

    #[test]
    fn amplification_reports_abandoned_io() {
        let mut d = TierDecisions::default();
        assert_eq!(d.read_amplification(), None, "no delivery, no ratio");
        d.useful_bytes = 100;
        d.moved_bytes = 100;
        assert_eq!(d.read_amplification(), Some(1.0));
        d.moved_bytes = 250;
        assert_eq!(d.read_amplification(), Some(2.5));
    }
}
