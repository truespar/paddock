//! Nemotron 3 Diarization's format-independent, bounded streaming contract.
//! This is not the older four-speaker Conformer Sortformer architecture.
use serde::{Deserialize, Serialize};

pub mod checkpoint;
pub mod words;

pub const SAMPLE_RATE: usize = 16_000;
pub const HOP: usize = 160;
pub const MEL: usize = 128;
pub const HIDDEN: usize = 512;
pub const SPEAKERS: usize = 8;
pub const STACK: usize = 8;
pub const CACHE: usize = 264;

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Preset {
    #[default]
    Offline,
    Low,
    VeryLow,
    UltraLow,
}

/// NVIDIA's latency presets, in 80-ms feature-stacked rows. The advertised
/// latency excludes the centered STFT's lookahead and actual execution time.
impl Preset {
    pub fn geometry(self) -> Geometry {
        let (chunk, right, fifo, update) = match self {
            Self::Offline => (340, 40, 40, 300),
            Self::Low => (9, 4, 264, 222),
            Self::VeryLow => (6, 2, 264, 222),
            Self::UltraLow => (3, 1, 264, 222),
        };
        Geometry {
            chunk,
            right,
            fifo,
            update,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    pub chunk: usize,
    pub right: usize,
    pub fifo: usize,
    pub update: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct Segment {
    pub speaker: usize,
    pub start: f64,
    pub end: f64,
}

/// Append-only finalized intervals plus at most eight open intervals. An open
/// interval is a snapshot, not an additional segment: its (speaker, start) key
/// remains stable until it appears in `completed`. No recording history is kept.
pub struct SegmentTracker {
    threshold: f32,
    open: [Option<usize>; SPEAKERS],
    frames: usize,
    finished: bool,
}

#[derive(Debug, Serialize)]
pub struct SegmentUpdate {
    pub completed: Vec<Segment>,
    pub active: Vec<Segment>,
    pub frames: usize,
}

impl SegmentTracker {
    pub fn new(threshold: f32) -> Result<Self, String> {
        if !threshold.is_finite() || !(0.0..1.0).contains(&threshold) {
            return Err("threshold must be in [0,1)".into());
        }
        Ok(Self {
            threshold,
            open: [None; SPEAKERS],
            frames: 0,
            finished: false,
        })
    }

    pub fn push(
        &mut self,
        probs: &[[f32; SPEAKERS]],
        finish: bool,
    ) -> Result<SegmentUpdate, String> {
        if self.finished
            || probs
                .iter()
                .flatten()
                .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        {
            return Err("invalid or finished speaker timeline".into());
        }
        let mut completed = Vec::new();
        for p in probs {
            for (speaker, (&p, start)) in p.iter().zip(&mut self.open).enumerate() {
                match (*start, p > self.threshold) {
                    (None, true) => *start = Some(self.frames),
                    (Some(first), false) => {
                        completed.push(Segment {
                            speaker,
                            start: first as f64 * 0.01,
                            end: self.frames as f64 * 0.01,
                        });
                        *start = None;
                    }
                    _ => {}
                }
            }
            self.frames += 1;
        }
        let mut active = Vec::new();
        for (speaker, start) in self.open.iter_mut().enumerate() {
            if let Some(first) = *start {
                let segment = Segment {
                    speaker,
                    start: first as f64 * 0.01,
                    end: self.frames as f64 * 0.01,
                };
                if finish {
                    completed.push(segment);
                    *start = None;
                } else {
                    active.push(segment);
                }
            }
        }
        completed.sort_by(|a, b| a.start.total_cmp(&b.start).then(a.speaker.cmp(&b.speaker)));
        self.finished = finish;
        Ok(SegmentUpdate {
            completed,
            active,
            frames: self.frames,
        })
    }
}

/// Threshold independently per speaker: argmax would erase overlap. Frame
/// indices are absolute, so network chunk boundaries never change timestamps.
pub fn segments(probs: &[[f32; SPEAKERS]], first_frame: usize, threshold: f32) -> Vec<Segment> {
    let mut out = Vec::new();
    for speaker in 0..SPEAKERS {
        let mut start = None;
        for i in 0..=probs.len() {
            let active = i < probs.len() && probs[i][speaker] > threshold;
            match (start, active) {
                (None, true) => start = Some(i),
                (Some(s), false) => {
                    out.push(Segment {
                        speaker,
                        start: (first_frame + s) as f64 * 0.01,
                        end: (first_frame + i) as f64 * 0.01,
                    });
                    start = None;
                }
                _ => {}
            }
        }
    }
    out.sort_by(|a, b| a.start.total_cmp(&b.start).then(a.speaker.cmp(&b.speaker)));
    out
}

/// Arrival-order cache selection at encoder resolution. None is a learned
/// silence embedding, not a zero vector. Do not reorder the result by score:
/// speaker-major temporal ordering is part of the model's input contract.
pub fn cache_selection(probs: &[[f32; SPEAKERS]]) -> Result<Vec<Option<usize>>, String> {
    if probs.len() <= CACHE
        || probs.len() > 1024
        || probs
            .iter()
            .flatten()
            .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
    {
        return Err("invalid Nemotron speaker-cache probabilities".into());
    }
    let n = probs.len();
    let mut scores = vec![vec![f32::NEG_INFINITY; n + 1]; SPEAKERS];
    for (i, p) in probs.iter().enumerate() {
        let sum: f32 = p.iter().map(|p| (1.0 - p).max(0.25).ln()).sum();
        for s in 0..SPEAKERS {
            if p[s] > 0.5 {
                scores[s][i] =
                    p[s].max(0.25).ln() - (1.0 - p[s]).max(0.25).ln() + sum - 0.5f32.ln();
            }
        }
    }
    for score in &mut scores {
        if score[..n].iter().filter(|&&s| s > 0.0).count() >= 16 {
            for s in &mut score[..n] {
                if *s <= 0.0 {
                    *s = f32::NEG_INFINITY;
                }
            }
        }
        for s in &mut score[CACHE..n] {
            *s += 0.05;
        }
        for (count, scale) in [(24, 2.0f32), (48, 1.0f32)] {
            let mut order: Vec<_> = (0..n).collect();
            select_best(&mut order, count, |i| score[i]);
            for &i in order.iter().take(count) {
                score[i] -= scale * 0.5f32.ln();
            }
        }
        score[n] = f32::INFINITY;
    }
    let mut order: Vec<_> = (0..SPEAKERS * (n + 1)).collect();
    let value = |i: usize| scores[i / (n + 1)][i % (n + 1)];
    select_best(&mut order, CACHE, value);
    order.truncate(CACHE);
    // NeMo replaces disabled indices with a sentinel BEFORE temporal sorting.
    order.sort_by_key(|&i| {
        if value(i) == f32::NEG_INFINITY {
            usize::MAX
        } else {
            i
        }
    });
    Ok(order
        .into_iter()
        .map(|i| (i % (n + 1) < n && value(i) > f32::NEG_INFINITY).then_some(i % (n + 1)))
        .collect())
}

// Only membership of the top set matters to boosting and cache admission.
// Linear-time partitioning avoids fully sorting all 8*(n+1) candidates. The
// index tie-break makes selection deterministic even for silence/overlap ties;
// the caller restores speaker-major temporal order after selecting the set.
fn select_best(order: &mut [usize], count: usize, value: impl Fn(usize) -> f32) {
    if count < order.len() {
        order.select_nth_unstable_by(count, |&a, &b| {
            value(b).total_cmp(&value(a)).then(a.cmp(&b))
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn partial_cache_selection_matches_full_sort_including_ties() {
        for len in [265, 486, 528, 684, 1024, 8200] {
            for seed in 0..8 {
                let values: Vec<_> = (0..len)
                    .map(|i| match (i + seed) % 19 {
                        0 => f32::INFINITY,
                        1..=7 => f32::NEG_INFINITY,
                        8 => -0.0,
                        9 => 0.0,
                        _ => ((i * 103 + seed * 23) % 47) as f32 / 47.,
                    })
                    .collect();
                let mut expected: Vec<_> = (0..len).collect();
                expected.sort_by(|&a, &b| values[b].total_cmp(&values[a]).then(a.cmp(&b)));
                for count in [24, 48, CACHE, len] {
                    let mut actual: Vec<_> = (0..len).collect();
                    select_best(&mut actual, count, |i| values[i]);
                    let mut subset = expected[..count].to_vec();
                    subset.sort_unstable();
                    actual[..count].sort_unstable();
                    assert_eq!(
                        actual[..count],
                        subset,
                        "len={len}, seed={seed}, count={count}"
                    );
                }
            }
        }
    }
    #[test]
    fn incremental_segments_equal_offline_and_do_not_close_at_transport_boundaries() {
        let probs: Vec<_> = (0..103)
            .map(|i| {
                [
                    if !(10..=30).contains(&i) { 0.9 } else { 0. },
                    if i > 5 && i < 80 { 0.8 } else { 0. },
                    0.,
                    0.,
                    0.,
                    0.,
                    0.,
                    0.,
                ]
            })
            .collect();
        for chunk in [1, 7, 31, 103] {
            let mut tracker = SegmentTracker::new(0.5).unwrap();
            let mut completed = Vec::new();
            for p in probs.chunks(chunk) {
                let update = tracker.push(p, false).unwrap();
                assert!(update.active.len() <= 8);
                completed.extend(update.completed);
            }
            let last = tracker.push(&[], true).unwrap();
            assert!(last.active.is_empty());
            completed.extend(last.completed);
            completed.sort_by(|a, b| a.start.total_cmp(&b.start).then(a.speaker.cmp(&b.speaker)));
            assert_eq!(completed, segments(&probs, 0, 0.5));
            assert!(tracker.push(&[], true).is_err());
        }
        let mut tracker = SegmentTracker::new(0.5).unwrap();
        assert!(tracker.push(&[[f32::NAN; 8]], false).is_err());
        assert_eq!(tracker.frames, 0);
        assert!(SegmentTracker::new(f32::NAN).is_err());
    }
    #[test]
    fn overlap_and_absolute_times() {
        let p = [[0.9, 0.8, 0., 0., 0., 0., 0., 0.]; 10];
        let s = segments(&p, 23, 0.5);
        assert_eq!(s.len(), 2);
        assert_eq!((s[0].start, s[0].end), (0.23, 0.33));
        assert_eq!(s[1].speaker, 1);
        assert!(segments(&[[0.5; 8]], 0, 0.5).is_empty());
    }
    #[test]
    fn silence_cache_and_invalid_inputs() {
        assert!(
            cache_selection(&[[0.; 8]; 300])
                .unwrap()
                .iter()
                .all(Option::is_none)
        );
        assert!(cache_selection(&[[f32::NAN; 8]; 300]).is_err());
        assert!(cache_selection(&[[0.; 8]; 10]).is_err());
    }
    #[test]
    fn cache_preserves_all_speakers_in_speaker_temporal_order() {
        let mut p = vec![[0.01; 8]; 512];
        for (i, row) in p.iter_mut().enumerate() {
            row[i / 64] = 0.8 + (i % 64) as f32 * 0.001;
        }
        let picks = cache_selection(&p).unwrap();
        assert_eq!(picks.len(), 264);
        assert_eq!(picks.iter().filter(|p| p.is_none()).count(), 8);
        for s in 0..8 {
            assert!(picks.iter().flatten().filter(|&&i| i / 64 == s).count() >= 24);
        }
    }
    #[test]
    fn presets_are_bounded() {
        for p in [
            Preset::Offline,
            Preset::Low,
            Preset::VeryLow,
            Preset::UltraLow,
        ] {
            let g = p.geometry();
            assert!(CACHE + g.fifo + g.chunk + g.right <= 684);
            assert!(g.chunk > 0 && g.update > 0);
        }
    }
}
