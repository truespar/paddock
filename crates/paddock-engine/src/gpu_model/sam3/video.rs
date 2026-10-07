//! SAM 3 on video: Meta's `Sam3VideoInference` frame loop (the predictor's
//! `add_prompt` on the first frame, then `propagate_in_video` forward) on the
//! engine, one frame in at a time.
//!
//! A frame, as `_det_track_one_frame` runs it:
//! 1. detection: the picture detector on the frame (Pillow-resized and
//!    normalized as Meta's frame loader does), every query scored by the
//!    joint score's logit round trip, kept over 0.5, mask NMS at 0.1;
//! 2. propagation: every tracker state's objects through the bank
//!    (`bank.rs`), the memory attention and the heads in tracking mode, the
//!    best of three candidates kept; then the hole and sprinkle fill;
//! 3. the plan (`video_plan.rs`): association, births, the hot start, the
//!    re-conditioning picks, occlusion suppression;
//! 4. re-conditioning: a picked track's mask-as-output pass from its
//!    detection (the frame becomes a conditioning frame, its pointer the
//!    pass's);
//! 5. the memory update: every track's mask to 1152^2, the pixel-wise
//!    non-overlap that suppresses a mask losing most of its area, the memory
//!    encoder;
//! 6. births: the new detections as a new tracker state - each mask through
//!    the video's own size and back (`add_new_mask` and the consolidation),
//!    a mask-as-output pass for the pointer, the memory at 1008^2 binarized;
//!    removals;
//! 7. the output: the frame's record and planes, held for the 15-frame hot
//!    start; a held frame is rendered at the video's size when it leaves.
//!
//! The first frame runs twice, as Meta's predictor does: `add_prompt`'s pass
//! (everything a birth), then propagation's (the births' conditioning frame
//! tracked again, re-conditioned, its memory updated).
//!
//! A session can track several concepts. Meta's predictor takes one text
//! prompt a session; here each concept is Meta's whole pipeline of its own
//! (its detections, planner, tracker states and hold) and only the frame's
//! encoding is shared - the image encoder never reads the prompt, so every
//! concept's result is the one a session of its own would give, at one
//! encode a frame instead of one a concept. An object's id is unique in the
//! session (`local * concepts + concept`, so one concept keeps Meta's ids)
//! and it carries its concept.
//!
//! Precision is each part's own (their gates); the decisions are the
//! planner's on the device's exact counts.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;

use cudarc::driver::CudaSlice;

use super::detector::joint_score;
use super::video_masks::{GpuSam3VideoMasks, LOW_PX, LOW_RES, Sam3VideoObject};
use super::video_plan::{
    DET_NMS_THRESH, HOTSTART_DELAY, SCORE_THRESHOLD_DETECTION, Sam3FrameRecord, Sam3MaskSource,
    Sam3PlanIn, Sam3VideoPlanner,
};
use super::{
    GpuModelError, GpuSam3, GpuSam3BankKv, GpuSam3FrameIn, GpuSam3MemAttn, GpuSam3MemEnc, MEM_DIM,
    PvsFeatures, PvsMask, PvsPrompt, Sam3Bank, Sam3Fail,
};
use crate::gpu::{GpuExecutor, Sam3Resize};

/// The model's input side, Meta's `input_mask_size` and the memory grid.
const SIDE: usize = 1008;
const IN_MASK: usize = 1152;
const IN_PX: usize = IN_MASK * IN_MASK;
const MEM_TOKENS: usize = 72 * 72;
/// Objects one memory-encoder pass takes.
const MEM_CAP: usize = 4;
/// Concepts one session tracks.
pub const MAX_CONCEPTS: usize = 4;
/// `_suppress_shrinked_masks`: a mask keeping under this share of its area
/// after the pixel-wise non-overlap is suppressed for the memory.
const SHRINK_THRESHOLD: f32 = 0.3;
/// The mask-as-output pass's logits for a pixel off / on.
const MASK_OUT_LOGIT: f32 = 10.0;

/// One frame's detections as Meta's video loop keeps them: over 0.5 after
/// mask NMS, in query order. Their 288^2 mask logits are the video parts'
/// detection planes (`[n][288^2]`).
#[derive(Clone, Debug, Default)]
pub struct Sam3VideoDets {
    pub queries: Vec<usize>,
    pub scores: Vec<f32>,
    /// x0, y0, x1, y1 normalized
    pub boxes_xyxy: Vec<[f32; 4]>,
}

/// One frame of output, final (out of the hot-start window).
#[derive(Clone, Debug)]
pub struct Sam3VideoFrame {
    pub frame: u32,
    /// the video's size, the masks' too
    pub width: usize,
    pub height: usize,
    pub objects: Vec<Sam3VideoObject>,
}

/// What [`GpuSam3::video_frame`] returns: the frames whose output is final
/// now, and on request this frame's provisional output (rendered at once,
/// hiding what is removed so far).
#[derive(Clone, Debug, Default)]
pub struct Sam3VideoStep {
    pub frames: Vec<Sam3VideoFrame>,
    pub preview: Option<Sam3VideoFrame>,
}

/// A device plane set grown on demand, keeping what it holds.
struct Planes {
    buf: CudaSlice<f32>,
    plane: usize,
    cap: usize,
}

impl Planes {
    fn new(exec: &GpuExecutor, plane: usize) -> Result<Self, GpuModelError> {
        Ok(Self {
            buf: exec.alloc(plane)?,
            plane,
            cap: 1,
        })
    }
    fn ensure(&mut self, exec: &GpuExecutor, n: usize) -> Result<(), GpuModelError> {
        if n > self.cap {
            let cap = n.next_power_of_two();
            let mut grown = exec.alloc(cap * self.plane)?;
            exec.copy_region(&self.buf, 0, &mut grown, 0, self.cap * self.plane)?;
            self.buf = grown;
            self.cap = cap;
        }
        Ok(())
    }
}

/// The video path's own resident parts (loaded with the model).
pub(super) struct VideoParts {
    pub(super) weight_bytes: u64,
    frames: GpuSam3FrameIn,
    masks: GpuSam3VideoMasks,
    memenc: GpuSam3MemEnc,
    memattn: GpuSam3MemAttn,
    kv: GpuSam3BankKv,
    /// `tracker_model.mask_downsample`, `[16]` + `[1]`
    down_w: CudaSlice<f32>,
    down_b: CudaSlice<f32>,
    /// this frame's candidates before NMS, and its detections after
    cand: Planes,
    det: Planes,
    /// this frame's tracks, 288^2 each
    trk: Planes,
    /// masks at the input-mask size (the memory update; a birth's M)
    big: Planes,
    /// births: their 288^2 consolidated masks and 1008^2 high-res ones
    low: Planes,
    hi: Planes,
    /// one memory-encoder pass's masks, contiguous
    chunk: Planes,
    /// one mask at the video's size
    vres: Planes,
    /// one 288^2 mask prompt
    prompt: CudaSlice<f32>,
    counts: CudaSlice<u32>,
}

impl VideoParts {
    fn load(exec: Arc<GpuExecutor>, dir: &Path) -> Result<Self, GpuModelError> {
        let st = super::checkpoint::open(dir)?;
        if super::checkpoint::is_multiplex(&*st) {
            return Err(GpuModelError::Unsupported(
                "this is SAM 3.1, whose video tracker (Object Multiplex) this build does not run yet - \
                 pictures and clicks are served"
                    .into(),
            ));
        }
        let mut r = super::load::Reader {
            st: &*st,
            exec: &exec,
            bytes: 0,
        };
        let down_w = {
            let v = r.f32s("tracker_model.mask_downsample.weight", &[1, 1, 4, 4])?;
            r.dev(&v)?
        };
        let down_b = r.vec("tracker_model.mask_downsample.bias", 1)?;
        let memenc = GpuSam3MemEnc::load_dir(exec.clone(), dir, MEM_CAP)?;
        let memattn = GpuSam3MemAttn::load_dir(exec.clone(), dir)?;
        let weight_bytes = memenc.weight_bytes() + memattn.weight_bytes() + r.bytes;
        Ok(Self {
            weight_bytes,
            frames: GpuSam3FrameIn::new(exec.clone(), SIDE)?,
            masks: GpuSam3VideoMasks::new(exec.clone())?,
            memenc,
            memattn,
            kv: GpuSam3BankKv::load_dir(exec.clone(), dir)?,
            down_w,
            down_b,
            cand: Planes::new(&exec, LOW_PX)?,
            det: Planes::new(&exec, LOW_PX)?,
            trk: Planes::new(&exec, LOW_PX)?,
            big: Planes::new(&exec, IN_PX)?,
            low: Planes::new(&exec, LOW_PX)?,
            hi: Planes::new(&exec, SIDE * SIDE)?,
            chunk: Planes::new(&exec, IN_PX)?,
            vres: Planes::new(&exec, 1)?,
            prompt: exec.alloc(LOW_PX)?,
            counts: exec.alloc_u32(2)?,
        })
    }

    /// The detection planes after [`GpuSam3::video_detect`], `[n][288^2]`.
    pub(super) fn det_planes(&self) -> &CudaSlice<f32> {
        &self.det.buf
    }
}

/// A tracker state: the objects born on one frame, and their bank.
struct TrackState {
    ids: Vec<i64>,
    bank: Sam3Bank,
    /// the conditioning frame's consolidated masks and logits, for that
    /// frame's second pass (only the first frame has one)
    revisit: Option<(u32, CudaSlice<f32>, Vec<f32>)>,
}

/// A frame held for the hot start: its record and its 288^2 planes in the
/// record's object order.
struct Held {
    record: Sam3FrameRecord,
    planes: CudaSlice<f32>,
}

/// One concept's tracking: Meta's whole video pipeline for its prompt.
struct Track {
    ids: [u32; 32],
    valid: usize,
    planner: Sam3VideoPlanner,
    states: Vec<TrackState>,
    held: VecDeque<Held>,
    /// the last frame's plans (two on the first frame)
    plans: Vec<super::video_plan::Sam3Plan>,
}

/// A frame's context, the same for every concept.
#[derive(Clone, Copy)]
struct Ctx {
    f: u32,
    h: usize,
    w: usize,
    num_frames: Option<u32>,
    preview: bool,
}

/// One video being tracked.
pub struct Sam3VideoSession {
    tracks: Vec<Track>,
    h: usize,
    w: usize,
    num_frames: Option<u32>,
    next: u32,
    preview: bool,
}

impl Sam3VideoSession {
    /// What the last frame decided for `concept` (both passes on the first
    /// frame) - the gate's and a debugger's view.
    pub fn last_plans(&self, concept: usize) -> &[super::video_plan::Sam3Plan] {
        &self.tracks[concept].plans
    }
    /// Frames in so far.
    pub fn frames_in(&self) -> u32 {
        self.next
    }
    /// Concepts tracked.
    pub fn concepts(&self) -> usize {
        self.tracks.len()
    }
    /// Objects tracked now for `concept`, by their local ids.
    pub fn tracks(&self, concept: usize) -> &[i64] {
        self.tracks[concept].planner.tracks()
    }
    fn ctx(&self) -> Ctx {
        Ctx {
            f: self.next,
            h: self.h,
            w: self.w,
            num_frames: self.num_frames,
            preview: self.preview,
        }
    }
}

impl GpuSam3 {
    /// Whether this model can track video (the pack carries slots 784-805
    /// and the click heads).
    pub fn has_video(&self) -> bool {
        self.pvs.is_some() && self.exec.has_sam3_video()
    }

    fn video_parts(&mut self) -> Result<(), GpuModelError> {
        if self.video.is_none() {
            return Err(GpuModelError::Unsupported(self.video_off.clone()));
        }
        Ok(())
    }

    /// Load the video parts next to the picture path, or say why not (the
    /// pack predates them, or they do not fit). Called once, at load.
    pub(super) fn load_video(&mut self) {
        if !self.has_video() {
            self.video_off = "this kernel pack predates SAM 3's video path (slots 784-805) - \
                              rebuild or update the pack"
                .into();
            return;
        }
        match VideoParts::load(self.exec.clone(), &self.dir) {
            Ok(v) => self.video = Some(Box::new(v)),
            Err(e) => {
                tracing::warn!(error = %e, "sam3: video tracking unavailable");
                self.video_off = format!("SAM 3 video tracking is unavailable: {e}");
            }
        }
    }

    /// Whether video sessions are served (the parts loaded).
    pub fn video_ready(&self) -> bool {
        self.video.is_some()
    }

    /// Why video sessions are not served, when they are not.
    pub fn video_unavailable(&self) -> Option<&str> {
        self.video.is_none().then_some(self.video_off.as_str())
    }

    /// Start a video: each concept's tokens (as for a picture), the video's
    /// size (the outputs come back at it), its length when known, and
    /// whether each frame also returns a provisional output.
    pub fn video_start(
        &mut self,
        concepts: &[([u32; 32], usize)],
        (h, w): (usize, usize),
        num_frames: Option<u32>,
        preview: bool,
    ) -> Result<Sam3VideoSession, Sam3Fail> {
        if concepts.is_empty() || concepts.len() > MAX_CONCEPTS {
            return Err(Sam3Fail::Request(format!(
                "{} concepts (a session tracks 1 to {MAX_CONCEPTS})",
                concepts.len()
            )));
        }
        if let Some((_, valid)) = concepts.iter().find(|(ids, v)| *v < 2 || *v > ids.len()) {
            return Err(Sam3Fail::Request(format!(
                "{valid} valid prompt tokens (want SOT and EOT around at least one, of 32)"
            )));
        }
        if h == 0 || w == 0 {
            return Err(Sam3Fail::Request(format!("a {w}x{h} video")));
        }
        self.video_parts()?;
        Ok(Sam3VideoSession {
            tracks: concepts
                .iter()
                .map(|&(ids, valid)| Track {
                    ids,
                    valid,
                    planner: Sam3VideoPlanner::new(),
                    states: Vec::new(),
                    held: VecDeque::new(),
                    plans: Vec::new(),
                })
                .collect(),
            h,
            w,
            num_frames,
            next: 0,
            preview,
        })
    }

    /// Detection on one frame (`rgb` u8 HWC, `h x w`), Meta's video
    /// detector: the frame resized and normalized as its frame loader does,
    /// the picture detector, the joint score's round trip, over 0.5, mask
    /// NMS. The masks land in the video parts' detection planes.
    pub fn video_detect(
        &mut self,
        rgb: &[u8],
        (h, w): (usize, usize),
        ids: &[u32; 32],
        valid: usize,
    ) -> Result<Sam3VideoDets, Sam3Fail> {
        self.video_encode(rgb, (h, w))?;
        self.video_detect_encoded(ids, valid)
    }

    /// The frame into the image encoder (both necks), as Meta's frame loader
    /// leaves it - once a frame, whatever the concepts.
    fn video_encode(&mut self, rgb: &[u8], (h, w): (usize, usize)) -> Result<(), Sam3Fail> {
        self.video_parts()?;
        self.picture_gone();
        let GpuSam3 { vision, video, .. } = self;
        let v = video.as_mut().expect("video parts loaded above");
        v.frames.land(rgb, h, w, vision.input_mut())?;
        vision.set_video_frames(true);
        let enc = vision.encode_staged(1, true);
        vision.set_video_frames(false);
        enc?;
        Ok(())
    }

    /// [`Self::video_detect`] for one concept on the frame the encoder holds.
    fn video_detect_encoded(
        &mut self,
        ids: &[u32; 32],
        valid: usize,
    ) -> Result<Sam3VideoDets, Sam3Fail> {
        let exec = self.exec.clone();
        let GpuSam3 {
            vision,
            text,
            det,
            video,
            last_text,
            ..
        } = self;
        let v = video.as_mut().expect("video parts loaded above");
        if last_text.as_ref() != Some(ids) {
            *last_text = None;
            text.encode(ids, 1)?;
            *last_text = Some(*ids);
        }
        det.encode_prompt(text.features(), 0, valid, vision.det_level(2), &[])?;
        det.fuse(vision.det_level(2))?;
        det.decode()?;
        let d = det.read_detections()?;
        let q: Vec<f32> = d.probs.iter().map(|&p| joint_score(p)).collect();
        let cand: Vec<usize> = (0..q.len())
            .filter(|&i| q[i] > SCORE_THRESHOLD_DETECTION)
            .collect();
        let mut out = Sam3VideoDets::default();
        if cand.is_empty() {
            return Ok(out);
        }
        det.segment(vision.det_level(0), vision.det_level(1))?;
        let nq = det.geom().queries;
        let n = cand.len();
        v.cand.ensure(&exec, n)?;
        for (k, &qi) in cand.iter().enumerate() {
            exec.sam3_mask_pick(
                det.masks_plane(),
                0,
                (LOW_PX, nq, qi),
                &mut v.cand.buf,
                k * LOW_PX,
            )?;
        }
        // Meta's nms_masks: a stable sort by score, then each kept mask
        // drops every later one it overlaps over 0.1
        let ious = v.masks.ious((&v.cand.buf, 0, n), None)?;
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| q[cand[b]].total_cmp(&q[cand[a]]));
        let mut keep = vec![true; n];
        for i in 0..n {
            let a = order[i];
            if !keep[a] {
                continue;
            }
            for &b in &order[i + 1..] {
                if ious.iou[a * n + b] > DET_NMS_THRESH {
                    keep[b] = false;
                }
            }
        }
        let kept: Vec<usize> = (0..n).filter(|&k| keep[k]).collect();
        v.det.ensure(&exec, kept.len())?;
        for (j, &k) in kept.iter().enumerate() {
            exec.copy_region(&v.cand.buf, k * LOW_PX, &mut v.det.buf, j * LOW_PX, LOW_PX)?;
            let qi = cand[k];
            let [cx, cy, bw, bh] = d.boxes[qi];
            out.queries.push(qi);
            out.scores.push(q[qi]);
            out.boxes_xyxy
                .push([cx - 0.5 * bw, cy - 0.5 * bh, cx + 0.5 * bw, cy + 0.5 * bh]);
        }
        Ok(out)
    }

    /// The detection planes of the last [`Self::video_detect`] (the gate's
    /// view), `[n][288^2]` f32.
    pub fn video_det_planes(&self) -> Option<&CudaSlice<f32>> {
        self.video.as_ref().map(|v| v.det_planes())
    }

    /// One frame in, the next of the session's (`rgb` u8 HWC at the
    /// session's size). Returns the frames whose output is now final - Meta
    /// holds each for the 15-frame hot start - and, if the session asked,
    /// this frame's provisional output.
    pub fn video_frame(
        &mut self,
        s: &mut Sam3VideoSession,
        rgb: &[u8],
    ) -> Result<Sam3VideoStep, Sam3Fail> {
        self.video_encode(rgb, (s.h, s.w))?;
        let cx = s.ctx();
        let n = s.tracks.len();
        let mut step = Sam3VideoStep::default();
        let mut preview: Option<Sam3VideoFrame> = None;
        for (k, c) in s.tracks.iter_mut().enumerate() {
            let dets = self.video_detect_encoded(&c.ids.clone(), c.valid)?;
            c.plans.clear();
            if cx.f == 0 {
                // add_prompt's pass: nothing tracked yet, every detection a
                // birth; the first frame's preview is this pass's
                let first = self.video_pass(c, (k, n), cx, &dets, false)?;
                let second = self.video_pass(c, (k, n), cx, &dets, true)?;
                merge(&mut preview, first.or(second));
            } else {
                let p = self.video_pass(c, (k, n), cx, &dets, true)?;
                merge(&mut preview, p);
            }
        }
        step.preview = preview;
        step.frames = self.video_yield(s, false)?;
        s.next += 1;
        Ok(step)
    }

    /// The end of the video: every held frame's output.
    pub fn video_finish(
        &mut self,
        s: &mut Sam3VideoSession,
    ) -> Result<Vec<Sam3VideoFrame>, Sam3Fail> {
        self.video_yield(s, true)
    }

    /// Render concept `k`'s record (its planes in its object order) as a
    /// frame's output, its objects under their session ids.
    #[allow(clippy::too_many_arguments)]
    fn video_render(
        &mut self,
        c: &Track,
        (k, n): (usize, usize),
        cx: Ctx,
        record: &Sam3FrameRecord,
        planes: &CudaSlice<f32>,
        hide_removed: bool,
    ) -> Result<Sam3VideoFrame, Sam3Fail> {
        let v = self.video.as_mut().expect("video parts");
        let objects: Vec<_> = record
            .objects
            .iter()
            .enumerate()
            .map(|(i, o)| (o.id, o.prob, o.tracker_prob, planes, i * LOW_PX))
            .collect();
        let hidden = |id: i64| hide_removed && c.planner.hidden().contains(&id);
        let mut objects = v.masks.outputs(&objects, &hidden, cx.h, cx.w)?;
        for o in &mut objects {
            o.id = o.id * n as i64 + k as i64;
            o.concept = k;
        }
        Ok(Sam3VideoFrame {
            frame: record.frame,
            width: cx.w,
            height: cx.h,
            objects,
        })
    }

    /// The held frames that leave the hot-start window: the oldest once 15
    /// are held, all of them at the end. Every concept holds the same
    /// frames, so they leave together, one output a frame.
    fn video_yield(
        &mut self,
        s: &mut Sam3VideoSession,
        all: bool,
    ) -> Result<Vec<Sam3VideoFrame>, Sam3Fail> {
        let cx = s.ctx();
        let n = s.tracks.len();
        let mut out = Vec::new();
        loop {
            let held = s.tracks[0].held.len();
            if !(held >= HOTSTART_DELAY as usize || (all && held > 0)) {
                break;
            }
            let mut frame: Option<Sam3VideoFrame> = None;
            for k in 0..n {
                let h = s.tracks[k].held.pop_front().expect("held in lockstep");
                let f = self.video_render(&s.tracks[k], (k, n), cx, &h.record, &h.planes, true)?;
                merge(&mut frame, Some(f));
            }
            out.extend(frame);
            if !all {
                break;
            }
        }
        Ok(out)
    }

    /// Meta's mask-as-output pass for detection `d` (`_use_mask_as_output`
    /// through `add_new_mask`): its mask at the input-mask size, binarized,
    /// into `big` plane `slot`; the heads' pointer from its 4x4-downsampled
    /// prompt on the raw tracker feature, or `no_object_pointer` when the
    /// mask is empty. Returns whether it has any pixel; the pointer is the
    /// heads' [`super::GpuSam3Pvs::pointer`] unless it is empty.
    fn mask_as_output(&mut self, d: usize, slot: usize) -> Result<bool, GpuModelError> {
        let exec = self.exec.clone();
        let GpuSam3 {
            vision, pvs, video, ..
        } = self;
        let v = video.as_mut().expect("video parts");
        let pvs = pvs.as_mut().expect("video needs the heads");
        v.big.ensure(&exec, slot + 1)?;
        exec.sam3_resize_f32(
            (&v.det.buf, d * LOW_PX, LOW_PX),
            (&mut v.big.buf, slot * IN_PX, IN_PX),
            1,
            (LOW_RES, LOW_RES),
            (IN_MASK, IN_MASK),
            Sam3Resize::Bilinear,
            Some((0.0, 0.0, 1.0)),
        )?;
        exec.sam3_nonoverlap(
            &mut v.big.buf,
            slot * IN_PX,
            IN_PX,
            1,
            IN_PX,
            Some(&mut v.counts),
            false,
        )?;
        let any = exec.to_host_u32_len(&v.counts, 1)?[0] > 0;
        exec.sam3_mask_down4(
            (&v.big.buf, slot * IN_PX, IN_PX),
            (&mut v.prompt, 0, LOW_PX),
            (&v.down_w, &v.down_b),
            1,
            LOW_RES,
        )?;
        pvs.predict_on(
            PvsFeatures {
                feat: vision.trk_level(2),
                no_memory: false,
                s0: vision.trk_s0(),
                s1: vision.trk_s1(),
                track: true,
            },
            &PvsPrompt {
                points: Vec::new(),
                bbox: None,
                mask: Some(PvsMask::Device(&v.prompt, 0)),
                multimask: false,
            },
        )?;
        Ok(any)
    }

    /// The memories of `n` objects of a state from `planes` (`side^2` each,
    /// `stride` apart from element `off`), written into its bank at `frame`,
    /// a memory-encoder pass at a time.
    #[allow(clippy::too_many_arguments)]
    fn video_memories(
        &mut self,
        st: usize,
        c: &mut Track,
        frame: u32,
        src: Src,
        n: usize,
        side: usize,
        binarize: bool,
        appearing: &[bool],
    ) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let GpuSam3 { vision, video, .. } = self;
        let v = video.as_mut().expect("video parts");
        let px = side * side;
        let mut b0 = 0;
        while b0 < n {
            let k = (n - b0).min(MEM_CAP);
            v.chunk.ensure(&exec, k)?;
            let (buf, off) = match src {
                Src::Big(o) => (&v.big.buf, o),
                Src::Hi(o) => (&v.hi.buf, o),
            };
            exec.copy_region(buf, off + b0 * px, &mut v.chunk.buf, 0, k * px)?;
            v.memenc.encode(
                vision.trk_level(2),
                &v.chunk.buf,
                side,
                binarize,
                &appearing[b0..b0 + k],
            )?;
            for j in 0..k {
                c.states[st].bank.set_memory(
                    frame,
                    b0 + j,
                    v.memenc.memory(),
                    j * MEM_TOKENS * MEM_DIM,
                )?;
            }
            b0 += k;
        }
        Ok(())
    }

    /// One pass of a concept (`concept` = its index, the session's count)
    /// over the session's current frame (module doc, steps 2-7); `stream`
    /// false is the first frame's `add_prompt` pass, whose output is not
    /// part of the stream. Returns the provisional output when the session
    /// asked for it.
    fn video_pass(
        &mut self,
        c: &mut Track,
        concept: (usize, usize),
        cx: Ctx,
        dets: &Sam3VideoDets,
        stream: bool,
    ) -> Result<Option<Sam3VideoFrame>, Sam3Fail> {
        let exec = self.exec.clone();
        let f = cx.f;
        let nd = dets.scores.len();
        let nt = c.planner.tracks().len();

        // ---- 2. propagation ----
        let mut logits = Vec::with_capacity(nt);
        {
            let GpuSam3 {
                vision, pvs, video, ..
            } = self;
            let v = video.as_mut().expect("video parts");
            let pvs = pvs.as_mut().expect("video needs the heads");
            v.trk.ensure(&exec, nt.max(1))?;
            let mut t = 0;
            for st in &mut c.states {
                let rev = st.revisit.take();
                if let Some((rf, planes, lg)) = rev
                    && rf == f
                {
                    // a conditioning frame tracked again: Meta hands back
                    // what the consolidation stored
                    let n = st.ids.len();
                    exec.copy_region(&planes, 0, &mut v.trk.buf, t * LOW_PX, n * LOW_PX)?;
                    logits.extend_from_slice(&lg);
                    t += n;
                    continue;
                }
                let plan = st.bank.plan(f, cx.num_frames)?;
                for b in 0..st.ids.len() {
                    let (nk, nrope) = v.kv.fill(&st.bank, &plan, b)?;
                    v.memattn
                        .run(vision.trk_level(2), v.kv.kin(), v.kv.vmem(), nk, nrope)?;
                    let res = pvs.predict_on(
                        PvsFeatures {
                            feat: v.memattn.output(),
                            no_memory: false,
                            s0: vision.trk_s0(),
                            s1: vision.trk_s1(),
                            track: true,
                        },
                        &PvsPrompt {
                            points: Vec::new(),
                            bbox: None,
                            mask: None,
                            multimask: true,
                        },
                    )?;
                    let k = res.pointer_mask;
                    exec.sam3_mask_pick(
                        pvs.logits(),
                        0,
                        (LOW_PX, 4, k),
                        &mut v.trk.buf,
                        t * LOW_PX,
                    )?;
                    st.bank
                        .track(f, b, pvs.pointer(), res.object_logit, res.iou[k])?;
                    logits.push(res.object_logit);
                    t += 1;
                }
            }
            debug_assert_eq!(t, nt);
            v.masks.clean(&mut v.trk.buf, 0, nt)?;
        }

        // ---- 3. the plan ----
        let plan = {
            let v = self.video.as_mut().expect("video parts");
            let dt = v
                .masks
                .ious((&v.det.buf, 0, nd), Some((&v.trk.buf, 0, nt)))?;
            let tt = v.masks.ious((&v.trk.buf, 0, nt), None)?;
            let nonempty: Vec<bool> = tt.area_a.iter().map(|&a| a > 0).collect();
            c.planner.plan(&Sam3PlanIn {
                frame: f,
                det_scores: &dets.scores,
                det_trk_iou: &dt.iou,
                trk_nonempty: &nonempty,
                trk_trk_iou: &tt.iou,
                trk_logits: &logits,
            })
        };
        // which state and slot each track is
        let mut place = Vec::with_capacity(nt);
        for (si, st) in c.states.iter().enumerate() {
            for b in 0..st.ids.len() {
                place.push((si, b));
            }
        }

        // ---- 4. re-conditioning ----
        for &(t, d) in &plan.recondition {
            let (si, b) = place[t];
            let any = self.mask_as_output(d, 0)?;
            let pvs = self.pvs.as_ref().expect("heads");
            let ptr = if any {
                pvs.pointer()
            } else {
                pvs.no_object_pointer()?
            };
            c.states[si].bank.condition(f, b, ptr)?;
        }

        // ---- 5. occlusion suppression, then the memory update ----
        {
            let v = self.video.as_mut().expect("video parts");
            for (t, &sup) in plan.suppress.iter().enumerate() {
                if sup {
                    v.masks.suppress(&mut v.trk.buf, t * LOW_PX)?;
                }
            }
        }
        if nt > 0 {
            let appearing = {
                let v = self.video.as_mut().expect("video parts");
                v.big.ensure(&exec, nt)?;
                exec.sam3_resize_f32(
                    (&v.trk.buf, 0, LOW_PX),
                    (&mut v.big.buf, 0, IN_PX),
                    nt,
                    (LOW_RES, LOW_RES),
                    (IN_MASK, IN_MASK),
                    Sam3Resize::Bilinear,
                    None,
                )?;
                if v.counts.len() < 2 * nt {
                    v.counts = exec.alloc_u32(2 * nt)?;
                }
                exec.sam3_nonoverlap(
                    &mut v.big.buf,
                    0,
                    IN_PX,
                    nt,
                    IN_PX,
                    Some(&mut v.counts),
                    false,
                )?;
                let c = exec.to_host_u32_len(&v.counts, 2 * nt)?;
                let mut appearing = Vec::with_capacity(nt);
                for t in 0..nt {
                    let (before, after) = (c[2 * t], c[2 * t + 1]);
                    let kept = after as f32 / (before.max(1) as f32) >= SHRINK_THRESHOLD;
                    if !kept {
                        exec.sam3_mask_set(&mut v.big.buf, t * IN_PX, IN_PX, -10.0, true)?;
                    }
                    appearing.push(kept && before > 0);
                }
                appearing
            };
            let mut t0 = 0;
            for si in 0..c.states.len() {
                let n = c.states[si].ids.len();
                self.video_memories(
                    si,
                    c,
                    f,
                    Src::Big(t0 * IN_PX),
                    n,
                    IN_MASK,
                    false,
                    &appearing[t0..t0 + n],
                )?;
                t0 += n;
            }
        }

        // ---- 6. births, then removals ----
        let born = plan.new_dets.len();
        if born > 0 {
            self.video
                .as_mut()
                .expect("video parts")
                .big
                .ensure(&exec, born)?;
            let mut bank = Sam3Bank::new(exec.clone(), born)?;
            let mut any = Vec::with_capacity(born);
            for (k, &d) in plan.new_dets.iter().enumerate() {
                let a = self.mask_as_output(d, k)?;
                let pvs = self.pvs.as_ref().expect("heads");
                bank.condition(
                    f,
                    k,
                    if a {
                        pvs.pointer()
                    } else {
                        pvs.no_object_pointer()?
                    },
                )?;
                any.push(a);
            }
            let revisit = {
                let v = self.video.as_mut().expect("video parts");
                v.low.ensure(&exec, born)?;
                v.hi.ensure(&exec, born)?;
                v.vres.ensure(&exec, cx.h * cx.w)?;
                for k in 0..born {
                    // add_new_mask's video-size mask, then the consolidation
                    // back to 288^2 (both antialiased)
                    exec.sam3_resize_f32(
                        (&v.big.buf, k * IN_PX, IN_PX),
                        (&mut v.vres.buf, 0, cx.h * cx.w),
                        1,
                        (IN_MASK, IN_MASK),
                        (cx.h, cx.w),
                        Sam3Resize::Antialias,
                        Some((0.5, -1024.0, 1024.0)),
                    )?;
                    exec.sam3_resize_f32(
                        (&v.vres.buf, 0, cx.h * cx.w),
                        (&mut v.low.buf, k * LOW_PX, LOW_PX),
                        1,
                        (cx.h, cx.w),
                        (LOW_RES, LOW_RES),
                        Sam3Resize::Antialias,
                        None,
                    )?;
                }
                exec.sam3_resize_f32(
                    (&v.low.buf, 0, LOW_PX),
                    (&mut v.hi.buf, 0, SIDE * SIDE),
                    born,
                    (LOW_RES, LOW_RES),
                    (SIDE, SIDE),
                    Sam3Resize::Bilinear,
                    None,
                )?;
                exec.sam3_nonoverlap(&mut v.hi.buf, 0, SIDE * SIDE, born, SIDE * SIDE, None, true)?;
                let mut keep = exec.alloc(born * LOW_PX)?;
                exec.copy_region(&v.low.buf, 0, &mut keep, 0, born * LOW_PX)?;
                let lg: Vec<f32> = any
                    .iter()
                    .map(|&a| if a { MASK_OUT_LOGIT } else { -MASK_OUT_LOGIT })
                    .collect();
                (f, keep, lg)
            };
            c.states.push(TrackState {
                ids: plan.new_ids.clone(),
                bank,
                revisit: Some(revisit),
            });
            let si = c.states.len() - 1;
            self.video_memories(si, c, f, Src::Hi(0), born, SIDE, true, &any)?;
        }
        for &id in &plan.removed {
            for st in &mut c.states {
                if let Some(b) = st.ids.iter().position(|&x| x == id) {
                    st.bank.remove_object(b)?;
                    st.ids.remove(b);
                }
            }
        }
        c.states.retain(|st| !st.ids.is_empty());

        // ---- 7. the output ----
        let record = c.planner.finish(&plan, &logits);
        c.plans.push(plan.clone());
        let planes = {
            let v = self.video.as_mut().expect("video parts");
            let n = record.objects.len();
            let mut planes = exec.alloc((n * LOW_PX).max(1))?;
            for (i, o) in record.objects.iter().enumerate() {
                match o.source {
                    Sam3MaskSource::Track(t) => {
                        exec.copy_region(&v.trk.buf, t * LOW_PX, &mut planes, i * LOW_PX, LOW_PX)?
                    }
                    Sam3MaskSource::Detection(d) => {
                        exec.copy_region(&v.det.buf, d * LOW_PX, &mut planes, i * LOW_PX, LOW_PX)?
                    }
                }
            }
            // a birth's output is its detection mask after the hole fill
            for (i, o) in record.objects.iter().enumerate() {
                if matches!(o.source, Sam3MaskSource::Detection(_)) {
                    v.masks.clean(&mut planes, i * LOW_PX, 1)?;
                }
            }
            planes
        };
        for st in &mut c.states {
            st.bank.prune(f);
        }
        let preview = if cx.preview {
            // add_prompt's own output hides nothing; a provisional frame
            // hides what is removed so far
            Some(self.video_render(c, concept, cx, &record, &planes, stream)?)
        } else {
            None
        };
        if stream {
            c.held.push_back(Held { record, planes });
        }
        Ok(preview)
    }
}

/// Where a state's memory masks sit.
#[derive(Clone, Copy)]
enum Src {
    Big(usize),
    Hi(usize),
}

/// Fold one concept's output of a frame into the frame's.
fn merge(into: &mut Option<Sam3VideoFrame>, part: Option<Sam3VideoFrame>) {
    match (into.as_mut(), part) {
        (_, None) => {}
        (None, Some(p)) => *into = Some(p),
        (Some(f), Some(p)) => f.objects.extend(p.objects),
    }
}
