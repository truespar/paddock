//! The knee search: how small a prefill share can get, beside a few waiting
//! streams, before the ingest pays more than [`KNEE_COST`] per prompt token.
//!
//! Measured, not modeled. The pacer's linear fit is the wrong instrument
//! here: its intercept is a cross-shape artifact (a small share's GEMMs run
//! less efficiently per row, and a waiting stream's spec round is not a
//! constant), and on GB10 it put the per-pass cost at 13 ms when halving
//! the share measured +12% per prompt token - a model-sized knee would have
//! landed on 256 rows, which measured +37%. So the search times the shares
//! themselves: a candidate tick runs between two full ticks, at almost the
//! same depth, and its cost per prompt row over theirs is the candidate's
//! true price - stream rounds, small-M inefficiency and all. Candidates
//! halve from the full share; the smallest one whose median price over
//! [`SAMPLES`] brackets is within the bound is the knee, re-checked every
//! [`REPROBE`] ticks.

use super::ROW_ALIGN;

/// What a waiting stream may cost the ingest: the knee share's cost per
/// prompt token stays within this fraction of the full share's. GB10,
/// qwen3.8-27b beside one stream (16K ingest): 512 rows measured +12% for
/// 1.75x the stream's tokens (2.9 -> 5.0 tok/s, max gap 1.2 -> 0.94 s) and
/// 256 rows +37% - the curve bends between them, and a fifth keeps the
/// accepted side clear of tick noise.
pub const KNEE_COST: f64 = 0.2;
/// Brackets per candidate; the decision takes their median.
const SAMPLES: usize = 3;
/// Knee ticks between re-checks of a settled knee (load and depth drift).
const REPROBE: u64 = 400;
/// A tick "ran" a size when its executed prefill rows are within this
/// fraction of it (a checkpoint cut or a tail shortens a span; those ticks
/// price nothing).
const MATCH: f64 = 0.1;

#[derive(Debug, Default)]
pub(super) struct KneeSearch {
    /// the full share the search prices against (0: not started)
    full: usize,
    /// candidate under test
    cand: usize,
    /// smallest candidate accepted so far (`full` until one passes)
    best: usize,
    /// price samples (candidate per-row cost over its bracket's) for `cand`
    prices: Vec<f64>,
    /// per-row cost of the latest full tick
    prev_full: Option<f64>,
    /// a candidate tick's per-row cost, waiting for the full tick after it
    open: Option<f64>,
    /// the size the next knee tick should take while searching
    next: usize,
    settled: bool,
    /// knee ticks since the knee settled
    since: u64,
}

fn aligned_half(n: usize) -> usize {
    n / 2 / ROW_ALIGN * ROW_ALIGN
}

fn ran(rows: usize, size: usize) -> bool {
    size > 0 && (rows as f64 - size as f64).abs() <= MATCH * size as f64
}

impl KneeSearch {
    fn restart(&mut self, full: usize) {
        *self = Self {
            full,
            cand: aligned_half(full),
            best: full,
            next: full,
            ..Self::default()
        };
        if self.cand < ROW_ALIGN {
            self.settled = true; // nothing smaller to try
        }
    }

    /// The share the next knee tick should take against `full`, when that is
    /// smaller than `full` (a probe or the settled knee). None: run full.
    pub(super) fn size(&self, full: usize) -> Option<usize> {
        if self.full != full {
            return None; // a new full share: the next observation restarts
        }
        let s = if self.settled { self.best } else { self.next };
        (s < full).then_some(s)
    }

    /// The settled knee, if the search found one below the full share.
    #[cfg(test)]
    pub(super) fn knee(&self) -> Option<usize> {
        (self.settled && self.best < self.full).then_some(self.best)
    }

    /// A knee tick against `full` ran `rows` prefill rows in `wall_ms`.
    pub(super) fn observe(&mut self, full: usize, rows: usize, wall_ms: f64) {
        if self.full != full {
            self.restart(full);
        }
        if rows == 0 || !wall_ms.is_finite() || wall_ms <= 0.0 {
            return;
        }
        if self.settled {
            self.since += 1;
            if self.since >= REPROBE {
                self.restart(full);
            }
            return;
        }
        let per_row = wall_ms / rows as f64;
        if ran(rows, full) {
            if let (Some(c), Some(p)) = (self.open.take(), self.prev_full) {
                self.prices.push(c / ((p + per_row) / 2.0));
            }
            self.prev_full = Some(per_row);
            self.next = self.cand;
            if self.prices.len() >= SAMPLES {
                self.decide();
            }
        } else if ran(rows, self.cand) && self.prev_full.is_some() {
            self.open = Some(per_row);
            self.next = self.full;
        }
    }

    fn decide(&mut self) {
        let mut p = std::mem::take(&mut self.prices);
        p.sort_by(f64::total_cmp);
        let median = p[p.len() / 2];
        if median <= 1.0 + KNEE_COST {
            self.best = self.cand;
            self.cand = aligned_half(self.cand);
            self.open = None;
            if self.cand < ROW_ALIGN {
                self.settled = true;
            }
        } else {
            self.settled = true;
        }
        self.next = self.full;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drive a search with a cost law `wall(rows)` until it settles,
    /// returning the knee and the ticks it took.
    fn search(full: usize, wall: impl Fn(usize) -> f64) -> (Option<usize>, usize) {
        let mut k = KneeSearch::default();
        let mut ticks = 0;
        // jitter so the median has something to do
        let jit = [1.0, 1.03, 0.98, 1.01, 0.99];
        while !k.settled && ticks < 200 {
            let rows = k.size(full).unwrap_or(full);
            k.observe(full, rows, wall(rows) * jit[ticks % jit.len()]);
            ticks += 1;
        }
        (k.knee(), ticks)
    }

    /// GB10-shaped: a ~100 ms pass under ~0.63 ms a row - halving costs
    /// ~14% a row, quartering ~42%: the knee is the half.
    #[test]
    fn a_heavy_pass_stops_the_knee_at_the_half() {
        let (knee, ticks) = search(1024, |r| 100.0 + 0.63 * (r + 8) as f64);
        assert_eq!(knee, Some(512));
        assert!(ticks <= 30, "{ticks} ticks");
    }

    /// A cheap pass (fast memory) under a 4096-row share: the knee falls
    /// to a few hundred rows, a fraction of the full tick's wall.
    #[test]
    fn a_cheap_pass_lets_the_knee_fall_far() {
        let wall = |r: usize| 18.0 + 0.24 * (r + 8) as f64;
        let (knee, _) = search(4096, wall);
        let knee = knee.expect("a knee below the full share");
        assert!((256..=1024).contains(&knee), "{knee}");
        assert!(wall(knee) < 0.2 * wall(4096));
    }

    /// No pass cost at all: every halving is free, down to one block.
    #[test]
    fn a_free_pass_halves_to_one_block() {
        let (knee, _) = search(1024, |r| 0.5 * r as f64);
        assert_eq!(knee, Some(ROW_ALIGN));
    }

    /// The pass dominates: even the half costs too much - no knee.
    #[test]
    fn a_pass_too_heavy_keeps_the_full_share() {
        let (knee, ticks) = search(1024, |r| 600.0 + 0.3 * r as f64);
        assert_eq!(knee, None);
        assert!(ticks <= 12, "{ticks} ticks");
    }

    /// Ticks of other sizes (cuts, tails) price nothing, and a changed full
    /// share starts over.
    #[test]
    fn stray_ticks_are_ignored_and_a_new_full_share_restarts() {
        let mut k = KneeSearch::default();
        k.observe(1024, 1024, 750.0);
        k.observe(1024, 300, 400.0); // a tail
        assert_eq!(k.size(1024), Some(512), "still probing the half");
        k.observe(2048, 2048, 1300.0);
        assert_eq!(k.size(2048), Some(1024), "restarted against 2048");
        assert_eq!(k.size(1024), None, "the old share is gone");
    }

    /// A settled knee is re-checked after REPROBE knee ticks.
    #[test]
    fn a_settled_knee_is_rechecked() {
        let wall = |r: usize| 100.0 + 0.63 * (r + 8) as f64;
        let mut k = KneeSearch::default();
        while !k.settled {
            let rows = k.size(1024).unwrap_or(1024);
            k.observe(1024, rows, wall(rows));
        }
        for _ in 0..REPROBE {
            k.observe(1024, 512, wall(512));
        }
        assert!(!k.settled, "probing again");
    }
}
