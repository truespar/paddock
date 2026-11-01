//! Write admission for the disk store: when the volume or the device says
//! no, stop asking every time.
//!
//! Two brakes, both consulted before a write touches the device:
//!
//! - [`VolumeRoom`] keeps the store out of the last free space on its
//!   volume. The quota is a ceiling, not a promise: the volume is shared
//!   with the models, the product database and the OS, and a nearly full
//!   SSD also writes worse (its spare area for garbage collection shrinks
//!   with the free space). The reserve has journald's shape
//!   (`SystemKeepFree`): a tenth of the volume, capped.
//! - [`WriteGate`] is back-pressure from failed writes. A full or read-only
//!   volume fails instantly, so a writer that only measures slow successes
//!   sees a healthy disk and keeps handing it every write - vLLM shipped
//!   exactly that (PR 61780: ENOSPC stores never reached its back-pressure
//!   detector). Here three failures in a row pause writes; one probe write
//!   goes through per back-off interval and the first success resumes, so a
//!   volume somebody cleaned up comes back without a restart.

use std::path::Path;
use std::time::{Duration, Instant};

/// Consecutive failed writes that pause the store (vLLM's number too).
const FAILURES_TO_PAUSE: u32 = 3;
/// First probe interval once paused; each failed probe doubles it.
const PROBE_FIRST: Duration = Duration::from_secs(1);
/// A dead disk gets a probe a minute, not one a second.
const PROBE_MAX: Duration = Duration::from_secs(60);

pub(super) struct WriteGate {
    failures: u32,
    paused: Option<Pause>,
}

struct Pause {
    since: Instant,
    next: Instant,
    every: Duration,
}

impl WriteGate {
    pub(super) fn new() -> Self {
        Self {
            failures: 0,
            paused: None,
        }
    }

    /// May a write touch the device now? Always while healthy; while paused,
    /// one probe per interval (the next probe is booked here, so a probe
    /// whose outcome never reports cannot open the gate for everyone).
    pub(super) fn admit(&mut self, now: Instant) -> bool {
        match &mut self.paused {
            None => true,
            Some(p) if now >= p.next => {
                p.next = now + p.every;
                true
            }
            Some(_) => false,
        }
    }

    /// Paused and not yet probe time - checked before the capacity checks,
    /// so callers do not evict to make room for a write that cannot run.
    /// (Does not book the probe; `admit` does, right before the IO.)
    pub(super) fn blocked(&self, now: Instant) -> bool {
        matches!(&self.paused, Some(p) if now < p.next)
    }

    pub(super) fn succeeded(&mut self, now: Instant) {
        self.failures = 0;
        if let Some(p) = self.paused.take() {
            tracing::info!(
                paused_s = now.duration_since(p.since).as_secs(),
                "KV disk cache: a probe write landed - writes resumed"
            );
        }
    }

    /// Count a failed write. Loud for the first few and for the pause
    /// itself, quiet for failed probes - a full disk must not become a log
    /// flood on top of everything else.
    pub(super) fn failed(&mut self, now: Instant, err: &std::io::Error) {
        self.failures = self.failures.saturating_add(1);
        if let Some(p) = &mut self.paused {
            p.every = (p.every * 2).min(PROBE_MAX);
            p.next = now + p.every;
            tracing::debug!(%err, retry_s = p.every.as_secs(), "KV disk cache: probe write failed");
            return;
        }
        if self.failures < FAILURES_TO_PAUSE {
            tracing::warn!(%err, "KV disk cache: write failed - this entry stays RAM-only");
            return;
        }
        self.paused = Some(Pause {
            since: now,
            next: now + PROBE_FIRST,
            every: PROBE_FIRST,
        });
        tracing::warn!(
            %err,
            failures = self.failures,
            "KV disk cache: writes PAUSED after repeated failures - the RAM tier \
             keeps serving and disk entries stay readable; one probe write per \
             back-off (1 s doubling to 60 s), and the first one that lands resumes"
        );
    }

    pub(super) fn paused(&self) -> bool {
        self.paused.is_some()
    }
}

/// How often free space is re-read. A `statvfs` is microseconds, but the
/// answer only has to track other writers on the volume, not every write of
/// ours - our own growth is counted exactly (see [`VolumeRoom::cap`]).
const SAMPLE_EVERY: Duration = Duration::from_secs(5);
/// After the store deletes segments to give the reserve back, a
/// copy-on-write filesystem reports the freed space late (btrfs commits
/// every 30 s, ZFS every few). A fresh sample inside that window would read
/// the reserve as still short and evict again, and again; the sample taken
/// before the eviction stays authoritative instead - it is expressed against
/// the store's own footprint, which IS current.
const SETTLE: Duration = Duration::from_secs(30);

/// The free space the store leaves on its volume: a tenth of it, at most
/// 32 GiB (journald keeps 15% capped at 4 GiB for its logs; KV writes come
/// in GiB, and models land on the same volume in tens of GiB).
pub(super) fn reserve(total: u64) -> u64 {
    (total / 10).min(32 << 30)
}

/// The most the disk cache writes in a UTC day: a quarter of its volume
/// (0.25 drive-writes per day), at most 1 TiB.
///
/// It used to be a flat 1 TiB/day on every drive. Consumer SSD warranties
/// work out near 0.3 drive-writes per day for TLC and 0.1 for QLC (rated
/// TBW over five years), so on a 1 TB drive the flat budget was about three
/// times a TLC rating. A quarter of a drive a day stays under that, with
/// room for everything else that writes to the volume. QLC is the case this
/// cannot see - nothing statvfs reports says QLC - and spending the whole
/// budget every day would wear one at 2.5x its rating; the budget is a
/// ceiling, not the expected rate. A volume that cannot be measured gets
/// the budget of a 1 TiB one.
pub(super) fn day_budget(volume_total: Option<u64>) -> u64 {
    const CAP: u64 = 1 << 40;
    volume_total.map_or(CAP / 4, |total| (total / 4).min(CAP))
}

/// Reads (available-to-us bytes, volume total) for a path.
pub(super) type SpaceFn = Box<dyn FnMut(&Path) -> Option<(u64, u64)> + Send>;

pub(super) fn volume_space(dir: &Path) -> Option<(u64, u64)> {
    let s = fs4::statvfs(dir).ok()?;
    Some((s.available_space(), s.total_space()))
}

pub(super) struct VolumeRoom {
    probe: SpaceFn,
    sample: Option<Sample>,
    /// When the store last evicted because the reserve (not its quota) was
    /// the binding limit - see [`SETTLE`].
    yielded_at: Option<Instant>,
    /// Writes are being refused for want of space (logged on the edges).
    short: bool,
}

struct Sample {
    at: Instant,
    /// The most the store may hold on disk: its footprint when sampled plus
    /// what the volume could spare beyond the reserve then.
    cap: u64,
}

impl VolumeRoom {
    pub(super) fn new(probe: SpaceFn) -> Self {
        Self {
            probe,
            sample: None,
            yielded_at: None,
            short: false,
        }
    }

    /// The most this store may hold on disk now (`u64::MAX` when the volume
    /// cannot be measured - the quota alone then rules). Between samples the
    /// cap is fixed in terms of footprint, so the store's own writes and
    /// deletes move it toward or away from the limit exactly.
    pub(super) fn cap(&mut self, dir: &Path, now: Instant, footprint: u64) -> u64 {
        let stale = self
            .sample
            .as_ref()
            .is_none_or(|s| now.duration_since(s.at) >= SAMPLE_EVERY);
        let settling = self
            .yielded_at
            .is_some_and(|t| now.duration_since(t) < SETTLE);
        if stale && !settling {
            let cap = (self.probe)(dir).map_or(u64::MAX, |(avail, total)| {
                footprint
                    .saturating_add(avail)
                    .saturating_sub(reserve(total))
            });
            self.sample = Some(Sample { at: now, cap });
        }
        self.sample.as_ref().map_or(u64::MAX, |s| s.cap)
    }

    /// The device just said full (or over quota): whatever the last sample
    /// claimed is stale, settling or not.
    pub(super) fn invalidate(&mut self) {
        self.sample = None;
        self.yielded_at = None;
    }

    pub(super) fn yielded(&mut self, now: Instant) {
        self.yielded_at = Some(now);
    }

    /// Edge-logged state: a refusal for want of space, or a write landing.
    pub(super) fn set_short(&mut self, short: bool, used: u64, cap: u64) {
        if short && !self.short {
            tracing::warn!(
                held_gib = used as f64 / (1u64 << 30) as f64,
                room_gib = cap as f64 / (1u64 << 30) as f64,
                "KV disk cache: its volume is down to the free-space reserve - new \
                 entries stay RAM-only until space frees (the cache never fills the disk)"
            );
        } else if !short && self.short {
            tracing::info!("KV disk cache: volume has room again - disk writes resumed");
        }
        self.short = short;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enospc() -> std::io::Error {
        std::io::Error::new(std::io::ErrorKind::StorageFull, "no space left on device")
    }

    #[test]
    fn three_failures_pause_and_one_probe_per_interval_goes_through() {
        let t0 = Instant::now();
        let mut g = WriteGate::new();
        for _ in 0..2 {
            assert!(g.admit(t0));
            g.failed(t0, &enospc());
        }
        assert!(!g.paused(), "two failures are not a pattern yet");
        assert!(g.admit(t0));
        g.failed(t0, &enospc());
        assert!(g.paused());
        assert!(
            g.blocked(t0) && !g.admit(t0),
            "paused: nothing until the first probe"
        );
        assert!(!g.admit(t0 + Duration::from_millis(999)));
        let p1 = t0 + Duration::from_secs(1);
        assert!(!g.blocked(p1));
        assert!(!g.blocked(p1), "checking does not book the probe");
        assert!(g.admit(p1), "the probe");
        assert!(!g.admit(p1), "one probe per interval, not one per caller");
        g.failed(p1, &enospc());
        assert!(
            !g.admit(p1 + Duration::from_millis(1999)),
            "back-off doubled"
        );
        let p2 = p1 + Duration::from_secs(2);
        assert!(g.admit(p2));
        g.succeeded(p2);
        assert!(!g.paused(), "a landed probe resumes without a restart");
        assert!(g.admit(p2) && g.admit(p2), "healthy: every write admitted");
    }

    #[test]
    fn the_back_off_is_capped() {
        let mut now = Instant::now();
        let mut g = WriteGate::new();
        for _ in 0..3 {
            g.failed(now, &enospc());
        }
        for _ in 0..20 {
            now += Duration::from_secs(3600);
            assert!(g.admit(now));
            g.failed(now, &enospc());
        }
        assert!(g.admit(now + PROBE_MAX), "never longer than a minute apart");
    }

    #[test]
    fn a_success_between_failures_resets_the_count() {
        let t = Instant::now();
        let mut g = WriteGate::new();
        g.failed(t, &enospc());
        g.failed(t, &enospc());
        g.succeeded(t);
        g.failed(t, &enospc());
        g.failed(t, &enospc());
        assert!(!g.paused());
    }

    #[test]
    fn the_day_budget_is_a_quarter_drive_write_capped_at_1_tib() {
        let gib = 1u64 << 30;
        assert_eq!(day_budget(Some(512 * gib)), 128 * gib);
        assert_eq!(day_budget(Some(1 << 40)), 256 * gib);
        assert_eq!(day_budget(Some(2 << 40)), 512 * gib);
        assert_eq!(
            day_budget(Some(8 << 40)),
            1 << 40,
            "big volumes keep the old cap"
        );
        // a 4 TB drive: just under the old flat 1 TiB
        assert_eq!(day_budget(Some(4_096_805_658_624)), 1_024_201_414_656);
        assert_eq!(day_budget(None), 256 * gib, "unmeasured: as a 1 TiB volume");
    }

    #[test]
    fn reserve_is_a_tenth_of_the_volume_capped_at_32_gib() {
        assert_eq!(reserve(100 << 30), 10 << 30);
        assert_eq!(reserve(4 << 40), 32 << 30);
    }

    fn fake(avail: std::sync::Arc<std::sync::atomic::AtomicU64>, total: u64) -> SpaceFn {
        Box::new(move |_| Some((avail.load(std::sync::atomic::Ordering::Relaxed), total)))
    }

    #[test]
    fn the_cap_tracks_our_footprint_between_samples_and_other_writers_at_samples() {
        use std::sync::{Arc, atomic::AtomicU64, atomic::Ordering::Relaxed};
        let gib = 1u64 << 30;
        let avail = Arc::new(AtomicU64::new(50 * gib));
        let mut r = VolumeRoom::new(fake(avail.clone(), 100 * gib));
        let d = Path::new(".");
        let t0 = Instant::now();
        // 20 GiB held, 50 free, 10 reserved: the store may grow to 60
        assert_eq!(r.cap(d, t0, 20 * gib), 60 * gib);
        // someone else writes 30 GiB; inside the sample window we have not
        // seen it yet, at the next sample we have
        avail.store(20 * gib, Relaxed);
        assert_eq!(r.cap(d, t0 + Duration::from_secs(1), 20 * gib), 60 * gib);
        assert_eq!(r.cap(d, t0 + SAMPLE_EVERY, 20 * gib), 30 * gib);
    }

    #[test]
    fn after_a_yield_the_old_sample_holds_until_the_filesystem_settles() {
        use std::sync::{Arc, atomic::AtomicU64};
        let gib = 1u64 << 30;
        // 5 GiB free of 100 (reserve 10), the store holds 40: cap 35
        let avail = Arc::new(AtomicU64::new(5 * gib));
        let mut r = VolumeRoom::new(fake(avail.clone(), 100 * gib));
        let d = Path::new(".");
        let t0 = Instant::now();
        assert_eq!(r.cap(d, t0, 40 * gib), 35 * gib);
        // the store evicts down to 35; a lagging filesystem still reports 5
        // free - a re-sample would now say 30 and evict again
        r.yielded(t0);
        let later = t0 + SAMPLE_EVERY * 2;
        assert_eq!(
            r.cap(d, later, 35 * gib),
            35 * gib,
            "settling: no re-sample"
        );
        assert_eq!(
            r.cap(d, t0 + SETTLE, 35 * gib),
            30 * gib,
            "settled: re-read"
        );
        // and a device ENOSPC overrides the settle window at once
        r.yielded(t0 + SETTLE);
        r.invalidate();
        assert_eq!(r.cap(d, t0 + SETTLE, 30 * gib), 25 * gib);
    }
}
