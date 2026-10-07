//! SAM 3's video decisions: Meta's `Sam3VideoBase` update planning, the part
//! of every frame that decides which detections start objects, which tracks
//! a detection matches, which tracks are dropped or suppressed, and what the
//! frame outputs. All of it is bookkeeping over a handful of numbers per
//! object; the masks behind those numbers (IoUs, areas) are the device's
//! (`video_masks.rs`).
//!
//! Meta's settings for the video model (`build_sam3_video_model` with
//! temporal disambiguation, the predictor's default) are fixed here, not
//! knobs: detections over 0.5 (after mask NMS at 0.1 IoU), association at
//! 0.1 IoU, a track unmatched below 0.5, a new object from a detection at
//! 0.7, a 15-frame hot start that removes a new track unmatched on 8 frames or
//! sharing a detection with an older one on 8, occlusion suppression at 0.7
//! IoU, re-conditioning every 16th frame from detections over 0.8 at 0.8 IoU.
//!
//! The order of a frame, as `_det_track_one_frame` runs it:
//! 1. [`Sam3VideoPlanner::plan`]: association, new ids, hot start,
//!    re-conditioning picks, occlusion suppression, the metadata;
//! 2. the caller's device work: the memory update over the (suppressed)
//!    tracks, births, the outputs' masks;
//! 3. [`Sam3VideoPlanner::finish`]: the tracks' scores join the frame's
//!    record.
//!
//! Meta holds each frame's output for the hot-start window and hides, when it
//! finally yields it, every object removed by then; [`Sam3FrameRecord`] is
//! what a frame holds until then and [`Sam3VideoPlanner::hidden`] the
//! removals.
//!
//! Faithful to quirks that change outputs:
//! - with no tracks at all, EVERY detection over 0.5 starts an object (the
//!   0.7 bar only applies once something is tracked);
//! - the per-detection match lists use the IoU after ambiguous matches
//!   (rows / columns with two or more 0.8 matches) are zeroed;
//! - a removed object keeps its frame score: the tracks' sigmoid scores are
//!   written over the -1e4 the removal set, as Meta's update order does;
//! - `obj_id_to_last_occluded` is rebuilt every frame from the tracks of
//!   that frame, so a frame with none forgets it.
//!
//! Keep-alive is counted (Meta does) but suppresses nothing: with
//! `suppress_unmatched_only_within_hotstart` on, that branch never runs.

use std::collections::{BTreeMap, BTreeSet, HashMap};

/// Detections kept (after NMS) over this probability.
pub const SCORE_THRESHOLD_DETECTION: f32 = 0.5;
/// Mask NMS among the detections: a lower-scored mask over this IoU goes.
pub const DET_NMS_THRESH: f32 = 0.1;
const ASSOC_IOU_THRESH: f32 = 0.1;
const TRK_ASSOC_IOU_THRESH: f32 = 0.5;
const NEW_DET_THRESH: f32 = 0.7;
/// Frames an output is held back, and the window a new track can be removed in.
pub const HOTSTART_DELAY: u32 = 15;
const HOTSTART_UNMATCH_THRESH: usize = 8;
const HOTSTART_DUP_THRESH: usize = 8;
const INIT_TRK_KEEP_ALIVE: i32 = 30;
const MAX_TRK_KEEP_ALIVE: i32 = 30;
const MIN_TRK_KEEP_ALIVE: i32 = -1;
const OCCLUSION_IOU_THRESH: f32 = 0.7;
/// The hole and sprinkle fill's area, on the 288^2 masks.
pub const FILL_HOLE_AREA: usize = 16;
const RECONDITION_EVERY: u32 = 16;
const HIGH_CONF_THRESH: f32 = 0.8;
const RECONDITION_IOU_THRESH: f32 = 0.8;
const NEVER_OCCLUDED: i64 = -1;
const ALWAYS_OCCLUDED: i64 = 100_000;
/// A removed object's score.
const REMOVED_SCORE: f32 = -1e4;

/// Meta's `mask_iou` from the device's counts: `inter / max(a + b - inter,
/// 1)`, in f32 as torch forms it (exact integers below 2^24).
pub fn mask_iou(inter: u32, a: u32, b: u32) -> f32 {
    let union = (a as f32 + b as f32) - inter as f32;
    inter as f32 / union.max(1.0)
}

/// One frame's inputs to the planning, every per-track slice in
/// [`Sam3VideoPlanner::tracks`] order.
pub struct Sam3PlanIn<'a> {
    pub frame: u32,
    /// the detections' probabilities (after NMS, over 0.5)
    pub det_scores: &'a [f32],
    /// `[dets][tracks]` mask IoU
    pub det_trk_iou: &'a [f32],
    /// whether each track's propagated mask has any pixel
    pub trk_nonempty: &'a [bool],
    /// `[tracks][tracks]` mask IoU of the propagated masks
    pub trk_trk_iou: &'a [f32],
    /// each track's object-score logit from the propagation
    pub trk_logits: &'a [f32],
}

/// What a frame decided.
#[derive(Clone, Debug, Default)]
pub struct Sam3Plan {
    pub frame: u32,
    /// the tracks this frame propagated (`obj_ids_all_gpu` before it)
    pub tracks: Vec<i64>,
    /// detections that start objects, and the ids they get
    pub new_dets: Vec<usize>,
    pub new_ids: Vec<i64>,
    /// non-empty tracks no detection matched at 0.5
    pub unmatched: Vec<i64>,
    /// tracks whose mask is empty
    pub empty: Vec<i64>,
    /// per detection, the tracks it matched at 0.1 (after the ambiguous
    /// matches are zeroed)
    pub det_matched: Vec<Vec<i64>>,
    /// Meta's `trk_id_to_max_iou_high_conf_det`, in its insertion order
    pub high_conf_det: Vec<(i64, usize)>,
    /// re-conditionings, `(track index, detection)`, in Meta's order
    pub recondition: Vec<(usize, usize)>,
    /// ids removed on this frame
    pub removed: Vec<i64>,
    /// per track: suppressed for occlusion (its mask goes to -10 before the
    /// memory update and the output)
    pub suppress: Vec<bool>,
}

/// One frame's output, held for the hot-start window: the objects in Meta's
/// dict order (the frame's tracks, then its births) with their probability
/// (the score their detection had at birth) and their tracker probability
/// that frame (the output's non-overlap ranks by it).
#[derive(Clone, Debug, Default)]
pub struct Sam3FrameRecord {
    pub frame: u32,
    pub objects: Vec<Sam3RecordObject>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Sam3MaskSource {
    /// track `i` of the frame's propagation
    Track(usize),
    /// detection `d` (a birth)
    Detection(usize),
}

#[derive(Clone, Copy, Debug)]
pub struct Sam3RecordObject {
    pub id: i64,
    pub prob: f32,
    pub tracker_prob: f32,
    pub source: Sam3MaskSource,
}

/// Meta's tracker metadata and its GPU-0 bookkeeping, one video.
#[derive(Default)]
pub struct Sam3VideoPlanner {
    tracks: Vec<i64>,
    max_obj_id: i64,
    score: HashMap<i64, f32>,
    /// frame -> object -> tracker probability (only frames still held)
    tracker_scores: BTreeMap<u32, BTreeMap<i64, f32>>,
    last_occluded: HashMap<i64, i64>,
    first_frame: HashMap<i64, u32>,
    /// in the order objects were first unmatched
    unmatched_frames: Vec<(i64, Vec<u32>)>,
    keep_alive: HashMap<i64, i32>,
    /// (earlier object, later object) -> frames they shared a detection
    overlap: Vec<((i64, i64), Vec<u32>)>,
    removed: BTreeSet<i64>,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl Sam3VideoPlanner {
    pub fn new() -> Self {
        Self {
            max_obj_id: -1,
            ..Self::default()
        }
    }

    /// The tracked objects, in the order the tracker holds them.
    pub fn tracks(&self) -> &[i64] {
        &self.tracks
    }

    /// Every object removed so far (an output yielded now hides these).
    pub fn hidden(&self) -> &BTreeSet<i64> {
        &self.removed
    }

    /// The probability an object was born with (`-1e4` once removed).
    pub fn score(&self, id: i64) -> Option<f32> {
        self.score.get(&id).copied()
    }

    /// The bookkeeping the gate holds to Meta's dumps.
    pub fn max_obj_id(&self) -> i64 {
        self.max_obj_id
    }
    pub fn first_frame(&self, id: i64) -> Option<u32> {
        self.first_frame.get(&id).copied()
    }
    pub fn unmatched_frames(&self, id: i64) -> Option<&[u32]> {
        self.unmatched_frames
            .iter()
            .find(|(i, _)| *i == id)
            .map(|(_, f)| f.as_slice())
    }
    pub fn keep_alive(&self, id: i64) -> Option<i32> {
        self.keep_alive.get(&id).copied()
    }
    pub fn overlap_frames(&self, first: i64, id: i64) -> Option<&[u32]> {
        self.overlap
            .iter()
            .find(|(k, _)| *k == (first, id))
            .map(|(_, f)| f.as_slice())
    }
    pub fn last_occluded(&self, id: i64) -> Option<i64> {
        self.last_occluded.get(&id).copied()
    }

    /// `_associate_det_trk` (its compilable core and the realization).
    fn associate(&self, inp: &Sam3PlanIn<'_>, plan: &mut Sam3Plan) {
        let (nd, nt) = (inp.det_scores.len(), self.tracks.len());
        if nt == 0 {
            // every detection is new, whatever its score
            plan.new_dets = (0..nd).collect();
            return;
        }
        if nd == 0 {
            for (t, &id) in self.tracks.iter().enumerate() {
                if inp.trk_nonempty[t] {
                    plan.unmatched.push(id);
                } else {
                    plan.empty.push(id);
                }
            }
            return;
        }
        let iou = |d: usize, t: usize| inp.det_trk_iou[d * nt + t];
        for (t, &id) in self.tracks.iter().enumerate() {
            let matched = (0..nd).any(|d| iou(d, t) >= TRK_ASSOC_IOU_THRESH);
            if !inp.trk_nonempty[t] {
                plan.empty.push(id);
            } else if !matched {
                plan.unmatched.push(id);
            }
        }
        let is_new: Vec<bool> = (0..nd)
            .map(|d| {
                inp.det_scores[d] >= NEW_DET_THRESH
                    && !(0..nt).any(|t| iou(d, t) >= ASSOC_IOU_THRESH)
            })
            .collect();
        plan.new_dets = (0..nd).filter(|&d| is_new[d]).collect();
        // ambiguous matches out: a track matching two detections at 0.8, or
        // a detection two tracks, has its column / row zeroed
        let many_t: Vec<bool> = (0..nt)
            .map(|t| {
                (0..nd)
                    .filter(|&d| iou(d, t) >= RECONDITION_IOU_THRESH)
                    .count()
                    > 1
            })
            .collect();
        let many_d: Vec<bool> = (0..nd)
            .map(|d| {
                (0..nt)
                    .filter(|&t| iou(d, t) >= RECONDITION_IOU_THRESH)
                    .count()
                    > 1
            })
            .collect();
        let iou2 = |d: usize, t: usize| {
            if many_t[t] || many_d[d] {
                0.0
            } else {
                iou(d, t)
            }
        };
        for (d, (&new, &score)) in is_new.iter().zip(inp.det_scores).enumerate() {
            plan.det_matched.push(
                (0..nt)
                    .filter(|&t| iou2(d, t) >= ASSOC_IOU_THRESH)
                    .map(|t| self.tracks[t])
                    .collect(),
            );
            // torch.argmax: the first maximum
            let (mut best, mut top) = (0usize, iou2(d, 0));
            for t in 1..nt {
                if iou2(d, t) > top {
                    top = iou2(d, t);
                    best = t;
                }
            }
            let high_conf = score >= HIGH_CONF_THRESH && !new;
            if high_conf && top >= RECONDITION_IOU_THRESH {
                let id = self.tracks[best];
                match plan.high_conf_det.iter_mut().find(|(k, _)| *k == id) {
                    Some(e) => e.1 = d,
                    None => plan.high_conf_det.push((id, d)),
                }
            }
        }
    }

    /// `_process_hotstart`, forward tracking.
    fn hotstart(&mut self, frame: u32, plan: &mut Sam3Plan) {
        let diff = frame as i64 - HOTSTART_DELAY as i64;
        for &id in &plan.new_ids {
            self.first_frame.entry(id).or_insert(frame);
            self.keep_alive.insert(id, INIT_TRK_KEEP_ALIVE);
        }
        let matched: BTreeSet<i64> = plan.det_matched.iter().flatten().copied().collect();
        for id in matched {
            let k = self.keep_alive.entry(id).or_insert(0);
            *k = (*k + 1).min(MAX_TRK_KEEP_ALIVE);
        }
        for &id in &plan.unmatched {
            match self.unmatched_frames.iter_mut().find(|(i, _)| *i == id) {
                Some((_, f)) => f.push(frame),
                None => self.unmatched_frames.push((id, vec![frame])),
            }
            let k = self.keep_alive.entry(id).or_insert(0);
            *k = (*k - 1).max(MIN_TRK_KEEP_ALIVE);
        }
        let mut newly: BTreeSet<i64> = BTreeSet::new();
        for (id, frames) in &self.unmatched_frames {
            if self.removed.contains(id) || newly.contains(id) {
                continue;
            }
            let within = self.first_frame.get(id).is_some_and(|&f| f as i64 > diff);
            if frames.len() >= HOTSTART_UNMATCH_THRESH && within {
                newly.insert(*id);
            }
        }
        // a detection matched by two or more tracks: the later ones overlap
        // the one that appeared first (Python's min: the first on a tie)
        for list in &plan.det_matched {
            if list.len() < 2 {
                continue;
            }
            let first_of = |id: &i64| self.first_frame.get(id).copied().unwrap_or(u32::MAX);
            let mut first = list[0];
            for id in &list[1..] {
                if first_of(id) < first_of(&first) {
                    first = *id;
                }
            }
            for &id in list {
                if id != first {
                    match self.overlap.iter_mut().find(|(k, _)| *k == (first, id)) {
                        Some((_, f)) => f.push(frame),
                        None => self.overlap.push(((first, id), vec![frame])),
                    }
                }
            }
        }
        for ((_, id), frames) in &self.overlap {
            if self.removed.contains(id) || newly.contains(id) {
                continue;
            }
            let within = self.first_frame.get(id).is_some_and(|&f| f as i64 > diff);
            if within && frames.len() >= HOTSTART_DUP_THRESH {
                newly.insert(*id);
            }
        }
        self.removed.extend(newly.iter().copied());
        plan.removed = newly.into_iter().collect();
    }

    /// `_suppress_overlapping_based_on_recent_occlusion`.
    fn occlusion(&mut self, inp: &Sam3PlanIn<'_>, plan: &mut Sam3Plan) {
        let nt = self.tracks.len();
        plan.suppress = vec![false; nt];
        if nt == 0 {
            self.last_occluded.clear();
            return;
        }
        let lo: Vec<i64> = self
            .tracks
            .iter()
            .map(|id| match self.last_occluded.get(id) {
                Some(&v) => v,
                None if plan.removed.contains(id) => ALWAYS_OCCLUDED,
                None => NEVER_OCCLUDED,
            })
            .collect();
        if nt > 1 {
            for i in 0..nt {
                for j in i + 1..nt {
                    if inp.trk_trk_iou[i * nt + j] < OCCLUSION_IOU_THRESH {
                        continue;
                    }
                    // the more recently occluded of the two goes, and only if
                    // the other has been occluded at some point
                    if lo[i] > lo[j] && lo[j] > NEVER_OCCLUDED {
                        plan.suppress[i] = true;
                    }
                    if lo[j] > lo[i] && lo[i] > NEVER_OCCLUDED {
                        plan.suppress[j] = true;
                    }
                }
            }
        }
        self.last_occluded = self
            .tracks
            .iter()
            .enumerate()
            .map(|(t, &id)| {
                let occluded = !inp.trk_nonempty[t] || plan.suppress[t];
                (id, if occluded { inp.frame as i64 } else { lo[t] })
            })
            .collect();
    }

    /// Step 1 of a frame (module doc).
    pub fn plan(&mut self, inp: &Sam3PlanIn<'_>) -> Sam3Plan {
        let mut plan = Sam3Plan {
            frame: inp.frame,
            tracks: self.tracks.clone(),
            ..Sam3Plan::default()
        };
        debug_assert_eq!(inp.trk_nonempty.len(), self.tracks.len());
        self.associate(inp, &mut plan);
        plan.new_ids = (0..plan.new_dets.len() as i64)
            .map(|i| self.max_obj_id + 1 + i)
            .collect();
        self.hotstart(inp.frame, &mut plan);

        // re-conditioning from the high-confidence matches, every 16th frame
        if inp.frame.is_multiple_of(RECONDITION_EVERY) {
            for &(id, d) in &plan.high_conf_det {
                if let Some(t) = self.tracks.iter().position(|&x| x == id)
                    && inp.trk_logits[t] > HIGH_CONF_THRESH
                {
                    plan.recondition.push((t, d));
                }
            }
        }
        self.occlusion(inp, &mut plan);

        // the metadata
        let removed: BTreeSet<i64> = plan.removed.iter().copied().collect();
        let mut tracks = self.tracks.clone();
        tracks.extend(plan.new_ids.iter().copied());
        tracks.retain(|id| !removed.contains(id));
        self.tracks = tracks;
        let fs = self.tracker_scores.entry(inp.frame).or_default();
        for (&id, &d) in plan.new_ids.iter().zip(&plan.new_dets) {
            self.score.insert(id, inp.det_scores[d]);
            fs.insert(id, inp.det_scores[d]);
            self.max_obj_id = self.max_obj_id.max(id);
        }
        for &id in &plan.removed {
            self.score.insert(id, REMOVED_SCORE);
            fs.insert(id, REMOVED_SCORE);
            self.last_occluded.remove(&id);
        }
        plan
    }

    /// Step 3 of a frame: the propagated tracks' probabilities join the
    /// frame's scores (over a removal's -1e4, as Meta's order has it), and
    /// the frame's output record is made.
    pub fn finish(&mut self, plan: &Sam3Plan, trk_logits: &[f32]) -> Sam3FrameRecord {
        let fs = self.tracker_scores.entry(plan.frame).or_default();
        for (&id, &l) in plan.tracks.iter().zip(trk_logits) {
            fs.insert(id, sigmoid(l));
        }
        let fs = fs.clone();
        let mut objects = Vec::with_capacity(plan.tracks.len() + plan.new_ids.len());
        let sources = plan
            .tracks
            .iter()
            .enumerate()
            .map(|(t, &id)| (id, Sam3MaskSource::Track(t)))
            .chain(
                plan.new_ids
                    .iter()
                    .zip(&plan.new_dets)
                    .map(|(&id, &d)| (id, Sam3MaskSource::Detection(d))),
            );
        for (id, source) in sources {
            objects.push(Sam3RecordObject {
                id,
                prob: self.score.get(&id).copied().unwrap_or(0.0),
                tracker_prob: fs.get(&id).copied().unwrap_or(0.0),
                source,
            });
        }
        // the hot-start buffer holds at most the window's frames
        let keep_from = plan.frame.saturating_sub(HOTSTART_DELAY + 1);
        self.tracker_scores.retain(|&f, _| f >= keep_from);
        Sam3FrameRecord {
            frame: plan.frame,
            objects,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A frame where every track is a clean match of its own detection.
    fn clean_frame(n: usize, scores: &[f32]) -> (Vec<f32>, Vec<bool>, Vec<f32>) {
        let nd = scores.len();
        let mut iou = vec![0.0; nd * n];
        for i in 0..n.min(nd) {
            iou[i * n + i] = 0.9;
        }
        let mut tt = vec![0.0; n * n];
        for i in 0..n {
            tt[i * n + i] = 1.0;
        }
        (iou, vec![true; n], tt)
    }

    #[test]
    fn first_frame_takes_every_detection() {
        let mut p = Sam3VideoPlanner::new();
        // 0.55 is under the 0.7 birth bar, but nothing is tracked yet
        let scores = [0.9, 0.55];
        let plan = p.plan(&Sam3PlanIn {
            frame: 0,
            det_scores: &scores,
            det_trk_iou: &[],
            trk_nonempty: &[],
            trk_trk_iou: &[],
            trk_logits: &[],
        });
        assert_eq!(plan.new_dets, [0, 1]);
        assert_eq!(plan.new_ids, [0, 1]);
        assert_eq!(p.tracks(), [0, 1]);
        let rec = p.finish(&plan, &[]);
        assert_eq!(rec.objects.len(), 2);
        assert_eq!(rec.objects[1].prob, 0.55);
        assert_eq!(rec.objects[1].tracker_prob, 0.55);
    }

    #[test]
    fn a_new_track_unmatched_through_the_hot_start_is_removed() {
        let mut p = Sam3VideoPlanner::new();
        let plan = p.plan(&Sam3PlanIn {
            frame: 0,
            det_scores: &[0.9],
            det_trk_iou: &[],
            trk_nonempty: &[],
            trk_trk_iou: &[],
            trk_logits: &[],
        });
        p.finish(&plan, &[]);
        // frames 1..8: the track is non-empty and no detection matches
        let mut removed_at = None;
        for f in 1..=8 {
            let plan = p.plan(&Sam3PlanIn {
                frame: f,
                det_scores: &[],
                det_trk_iou: &[],
                trk_nonempty: &[true],
                trk_trk_iou: &[1.0],
                trk_logits: &[5.0],
            });
            assert_eq!(plan.unmatched, [0]);
            if !plan.removed.is_empty() {
                removed_at = Some(f);
            }
            p.finish(&plan, &[5.0]);
        }
        assert_eq!(removed_at, Some(8));
        assert!(p.hidden().contains(&0));
        assert!(p.tracks().is_empty());
        assert_eq!(p.score(0), Some(-1e4));
    }

    #[test]
    fn an_old_track_survives_being_unmatched() {
        let mut p = Sam3VideoPlanner::new();
        let plan = p.plan(&Sam3PlanIn {
            frame: 0,
            det_scores: &[0.9],
            det_trk_iou: &[],
            trk_nonempty: &[],
            trk_trk_iou: &[],
            trk_logits: &[],
        });
        p.finish(&plan, &[]);
        for f in 1..60 {
            // matched for the first 20 frames, unmatched for the 40 after:
            // past the 15-frame window, so never removed
            let matched = f < 20;
            let (iou, ne, tt) = clean_frame(1, if matched { &[0.9] } else { &[] });
            let plan = p.plan(&Sam3PlanIn {
                frame: f,
                det_scores: if matched { &[0.9] } else { &[] },
                det_trk_iou: &iou,
                trk_nonempty: &ne,
                trk_trk_iou: &tt,
                trk_logits: &[5.0],
            });
            assert!(plan.removed.is_empty(), "frame {f}");
            p.finish(&plan, &[5.0]);
        }
        assert_eq!(p.keep_alive(0), Some(-1));
    }

    #[test]
    fn a_later_duplicate_of_a_track_is_removed() {
        let mut p = Sam3VideoPlanner::new();
        let plan = p.plan(&Sam3PlanIn {
            frame: 0,
            det_scores: &[0.9],
            det_trk_iou: &[],
            trk_nonempty: &[],
            trk_trk_iou: &[],
            trk_logits: &[],
        });
        p.finish(&plan, &[]);
        // frame 1: a second detection away from the track is born
        let plan = p.plan(&Sam3PlanIn {
            frame: 1,
            det_scores: &[0.9, 0.9],
            det_trk_iou: &[0.9, 0.0],
            trk_nonempty: &[true],
            trk_trk_iou: &[1.0],
            trk_logits: &[5.0],
        });
        assert_eq!(plan.new_ids, [1]);
        p.finish(&plan, &[5.0]);
        // then one detection matches both tracks (0.5 each): object 1 is
        // the later one; it goes after 8 such frames
        let mut removed_at = None;
        for f in 2..=12 {
            let plan = p.plan(&Sam3PlanIn {
                frame: f,
                det_scores: &[0.9],
                det_trk_iou: &[0.5, 0.5],
                trk_nonempty: &[true, true],
                trk_trk_iou: &[1.0, 0.3, 0.3, 1.0],
                trk_logits: &[5.0, 5.0],
            });
            assert_eq!(plan.det_matched, [vec![0, 1]]);
            if plan.removed == [1] {
                removed_at = Some(f);
            }
            p.finish(&plan, &[5.0, 5.0]);
            if removed_at.is_some() {
                break;
            }
        }
        assert_eq!(removed_at, Some(9));
        assert_eq!(p.overlap_frames(0, 1).map(<[u32]>::len), Some(8));
    }

    #[test]
    fn occlusion_suppresses_the_more_recently_occluded() {
        let mut p = Sam3VideoPlanner::new();
        let plan = p.plan(&Sam3PlanIn {
            frame: 0,
            det_scores: &[0.9, 0.9],
            det_trk_iou: &[],
            trk_nonempty: &[],
            trk_trk_iou: &[],
            trk_logits: &[],
        });
        p.finish(&plan, &[]);
        // frame 1: object 0 is gone (occluded), object 1 is not
        let plan = p.plan(&Sam3PlanIn {
            frame: 1,
            det_scores: &[0.9],
            det_trk_iou: &[0.0, 0.9],
            trk_nonempty: &[false, true],
            trk_trk_iou: &[1.0, 0.0, 0.0, 1.0],
            trk_logits: &[-5.0, 5.0],
        });
        assert_eq!(plan.suppress, [false, false]);
        assert_eq!(p.last_occluded(0), Some(1));
        assert_eq!(p.last_occluded(1), Some(-1));
        p.finish(&plan, &[-5.0, 5.0]);
        // frame 2: both cover the same pixels; object 1 was never occluded,
        // so it cannot suppress object 0 - nothing goes
        let plan = p.plan(&Sam3PlanIn {
            frame: 2,
            det_scores: &[0.9],
            det_trk_iou: &[0.8, 0.8],
            trk_nonempty: &[true, true],
            trk_trk_iou: &[1.0, 0.9, 0.9, 1.0],
            trk_logits: &[5.0, 5.0],
        });
        assert_eq!(plan.suppress, [false, false]);
        p.finish(&plan, &[5.0, 5.0]);
        // frame 3: object 1 is occluded once, object 0 then twice later
        for (f, ne) in [(3, [true, false]), (4, [false, true])] {
            let plan = p.plan(&Sam3PlanIn {
                frame: f,
                det_scores: &[],
                det_trk_iou: &[],
                trk_nonempty: &ne,
                trk_trk_iou: &[1.0, 0.0, 0.0, 1.0],
                trk_logits: &[5.0, 5.0],
            });
            p.finish(&plan, &[5.0, 5.0]);
        }
        // both have been occluded, object 0 most recently: it is suppressed
        let plan = p.plan(&Sam3PlanIn {
            frame: 5,
            det_scores: &[],
            det_trk_iou: &[],
            trk_nonempty: &[true, true],
            trk_trk_iou: &[1.0, 0.9, 0.9, 1.0],
            trk_logits: &[5.0, 5.0],
        });
        assert_eq!(plan.suppress, [true, false]);
        assert_eq!(p.last_occluded(0), Some(5));
        assert_eq!(p.last_occluded(1), Some(3));
    }

    #[test]
    fn mask_iou_is_torch_float_division() {
        assert_eq!(mask_iou(0, 0, 0), 0.0);
        assert_eq!(mask_iou(5, 10, 10), 5.0 / 15.0);
        assert_eq!(mask_iou(3, 3, 3), 1.0);
        // one empty mask: nothing in common, and no division by zero
        assert_eq!(mask_iou(0, 5, 0), 0.0);
    }
}
