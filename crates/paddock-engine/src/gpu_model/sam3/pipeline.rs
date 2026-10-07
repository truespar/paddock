//! One concept-segmentation request, picture to masks - Meta's
//! `Sam3Processor.set_image` + `set_text_prompt` / `add_geometric_prompt`,
//! end to end on the device:
//!
//!   decoded RGB (any size) -> torchvision's antialiased resize to 1008^2
//!   -> image encoder -> prompt (the caller's tokens - the runner tokenizes,
//!   "visual" when only boxes are given, as Meta's processor does)
//!   -> fusion encoder -> decoder + scorer
//!   -> keep every query with sigmoid(logit) * sigmoid(presence) > threshold
//!   -> segmentation head -> each kept mask bilinear back to the picture,
//!   sigmoid > 0.5, COCO RLE.
//!
//! The host only makes the control decision over the 200 readback scores (which
//! queries are kept, in score order) and turns each kept box from normalized
//! cxcywh into the picture's xyxy - the processor's own arithmetic, four floats
//! a box. Every pixel-sized step runs on the device.
//!
//! Both encoders' outputs outlive the request. A picture byte-identical to the
//! last one (same size, same RGB) keeps its encoded planes - Meta's processor
//! splits `set_image` from the prompt calls for exactly this, several prompts
//! on one picture - and a prompt with the same tokens keeps the text tower's
//! features. Everything after them depends on both and always runs. Reuse is
//! exact: the planes are the ones the same inputs produced.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::CudaSlice;

use super::{
    GpuModelError, GpuSam3Detector, GpuSam3Pvs, GpuSam3Text, GpuSam3Vision, PVS_MAX_POINTS,
    PvsMask, PvsPoint, PvsPrompt, Sam3Box,
};
use crate::gpu::GpuExecutor;

/// Run lengths one mask may need (COCO RLE). A real object boundary crosses a
/// column a handful of times; 4M runs is a mask fragmented past any use.
const RLE_CAP: usize = 4 << 20;

/// What the caller asks of one picture.
#[derive(Debug, Clone)]
pub struct Sam3Request {
    /// the prompt's tokens in the text tower's 32-slot layout
    /// (`paddock_tokenizer::sam3`), and how many are valid (SOT..EOT)
    pub ids: [u32; 32],
    pub valid: usize,
    pub boxes: Vec<Sam3Box>,
    /// keep queries scoring above this (Meta's processor default: 0.5)
    pub threshold: f32,
}

/// One click, in the picture's pixels.
#[derive(Debug, Clone, Copy)]
pub struct Sam3Click {
    pub x: f32,
    pub y: f32,
    pub positive: bool,
}

/// What the caller asks of one picture with clicks (interactive single-object
/// segmentation - Meta's SAM3InteractiveImagePredictor).
#[derive(Debug, Clone, Default)]
pub struct Sam3PointRequest {
    pub clicks: Vec<Sam3Click>,
    /// the object's box, x0 y0 x1 y1 in the picture's pixels
    pub bbox: Option<[f32; 4]>,
    /// three candidates or one; default: three for a lone click (the
    /// ambiguous case Meta's notebook asks three of), one otherwise
    pub multimask: Option<bool>,
    /// refine a previous answer on the same picture: its `refine_id` - the
    /// best candidate's low-res logits become the mask prompt
    pub refine: Option<u64>,
}

/// One found instance.
#[derive(Debug, Clone)]
pub struct Sam3Instance {
    /// sigmoid(logit) * sigmoid(presence)
    pub score: f32,
    /// xyxy in the picture's pixels
    pub bbox: [f32; 4],
    /// COCO RLE over the column-major mask: zeros first, alternating runs
    pub rle: Vec<u32>,
    /// mask pixels set
    pub area: u64,
}

#[derive(Debug, Clone, Default)]
pub struct Sam3Timings {
    pub resize_ms: f64,
    pub encode_ms: f64,
    pub prompt_ms: f64,
    pub detect_ms: f64,
    pub masks_ms: f64,
    /// the picture was the last request's: resize and encode were skipped
    pub image_reused: bool,
    /// the prompt's tokens were the last request's: the text tower was skipped
    pub text_reused: bool,
}

#[derive(Debug, Clone)]
pub struct Sam3Output {
    pub width: usize,
    pub height: usize,
    /// sigmoid(presence logit): does the concept appear at all
    pub presence: f32,
    /// kept instances, highest score first
    pub instances: Vec<Sam3Instance>,
    pub tokens: usize,
    pub timings: Sam3Timings,
    /// a click answer's handle: pass it back as `refine` to refine it
    pub refine_id: Option<u64>,
}

/// A request the caller can fix (an over-long prompt, a picture past the
/// endpoint's size, nothing to look for) versus an engine failure.
#[derive(Debug, thiserror::Error)]
pub enum Sam3Fail {
    #[error("{0}")]
    Request(String),
    #[error(transparent)]
    Engine(#[from] GpuModelError),
}

impl From<crate::gpu::GpuError> for Sam3Fail {
    fn from(e: crate::gpu::GpuError) -> Self {
        Self::Engine(e.into())
    }
}

/// SAM 3's concept path, resident: image encoder, text tower, detector and
/// the request-side planes.
pub struct GpuSam3 {
    pub(super) exec: Arc<GpuExecutor>,
    pub(super) vision: GpuSam3Vision,
    pub(super) text: GpuSam3Text,
    pub(super) det: GpuSam3Detector,
    max_pixels: usize,
    src: CudaSlice<u8>,
    mask: CudaSlice<u8>,
    starts: CudaSlice<u32>,
    counts: CudaSlice<u32>,
    nruns: CudaSlice<u32>,
    /// the picture the vision planes hold, if they are whole
    picture: PictureCache,
    /// the tokens the text features hold, if they are whole
    pub(super) last_text: Option<[u32; 32]>,
    /// the click heads (None: the pack predates them - clicks are refused)
    pub(super) pvs: Option<GpuSam3Pvs>,
    /// the last click answer on the current picture: its id and best mask
    last_pvs: Option<(u64, usize)>,
    next_refine: u64,
    /// where the weights came from (the video parts load from it on demand)
    pub(super) dir: PathBuf,
    /// the video path's own parts (None: not served, `video_off` says why)
    pub(super) video: Option<Box<super::video::VideoParts>>,
    pub(super) video_off: String,
}

/// What the vision planes hold: the picture (width, height, RGB) and whether
/// its tracker neck has run as well.
#[derive(Default)]
struct PictureCache {
    image: Option<(usize, usize, Vec<u8>)>,
    tracker: bool,
}

/// Bring the vision planes to `rgb`: reuse them when it is the cached picture
/// (running only the tracker neck when it is asked for and missing),
/// resize + encode otherwise. Returns whether the picture changed.
#[allow(clippy::too_many_arguments)]
fn ensure_picture(
    exec: &GpuExecutor,
    vision: &mut GpuSam3Vision,
    src: &mut CudaSlice<u8>,
    cache: &mut PictureCache,
    rgb: &[u8],
    width: usize,
    height: usize,
    tracker: bool,
    timings: &mut Sam3Timings,
) -> Result<bool, Sam3Fail> {
    let s = vision.config().image_size;
    timings.image_reused = cache
        .image
        .as_ref()
        .is_some_and(|(w, h, b)| *w == width && *h == height && b.as_slice() == rgb);
    if timings.image_reused {
        if tracker && !cache.tracker {
            let t0 = Instant::now();
            vision.tracker_neck(1)?;
            exec.synchronize()?;
            timings.encode_ms = t0.elapsed().as_secs_f64() * 1e3;
            cache.tracker = true;
        }
        return Ok(false);
    }
    // the planes stop being any picture's the moment the upload starts
    let mut keep = cache.image.take().map(|(_, _, b)| b).unwrap_or_default();
    cache.tracker = false;
    let t0 = Instant::now();
    exec.upload_u8(rgb, src)?;
    exec.sam3_resize_aa_u8(src, vision.input_mut(), height, width, s, s, 3)?;
    exec.synchronize()?;
    timings.resize_ms = t0.elapsed().as_secs_f64() * 1e3;

    let t0 = Instant::now();
    vision.encode_staged(1, tracker)?;
    exec.synchronize()?;
    timings.encode_ms = t0.elapsed().as_secs_f64() * 1e3;
    keep.clear();
    keep.extend_from_slice(rgb);
    cache.image = Some((width, height, keep));
    cache.tracker = tracker;
    Ok(true)
}

/// A mask's box from its column-major COCO RLE (zeros first): x0 y0 x1 y1 in
/// pixel edges, so a one-pixel mask is one pixel wide.
fn rle_box(rle: &[u32], h: usize) -> [f32; 4] {
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
    let mut pos = 0usize;
    for (i, &c) in rle.iter().enumerate() {
        let c = c as usize;
        if i % 2 == 1 && c > 0 {
            let (a, b) = (pos, pos + c - 1);
            let (xa, xb) = (a / h, b / h);
            x0 = x0.min(xa);
            x1 = x1.max(xb);
            if xa == xb {
                y0 = y0.min(a % h);
                y1 = y1.max(b % h);
            } else {
                // a run across a column boundary covers the first column's
                // bottom row and the next column's top row
                y0 = 0;
                y1 = h - 1;
            }
        }
        pos += c;
    }
    if x0 == usize::MAX {
        return [0.0; 4];
    }
    [x0 as f32, y0 as f32, (x1 + 1) as f32, (y1 + 1) as f32]
}

impl GpuSam3 {
    /// Load every part from a `facebook/sam3` directory. `max_pixels` bounds a
    /// request picture (it sizes the staging plane and the mask plane).
    pub fn load_dir(
        exec: Arc<GpuExecutor>,
        dir: &Path,
        max_pixels: usize,
        max_boxes: usize,
    ) -> Result<Self, GpuModelError> {
        if !exec.has_sam3_image_io() {
            return Err(GpuModelError::Unsupported(
                "this kernel pack predates SAM 3's picture and mask ops (slots 770-772) - \
                 rebuild or update the pack"
                    .into(),
            ));
        }
        let max_pixels = max_pixels.max(1);
        // the request planes: the picture as decoded, one mask at its size,
        // and the run lists
        let io = (max_pixels * 4 + 2 * RLE_CAP * 4) as u64;
        exec.vram_load_gate(io, "sam3 request planes")
            .map_err(GpuModelError::WontFit)?;
        let vision = GpuSam3Vision::load_dir(exec.clone(), dir, 1)?;
        let text = GpuSam3Text::load_dir(exec.clone(), dir, 1)?;
        let det = GpuSam3Detector::load_dir(exec.clone(), dir, max_boxes)?;
        Ok(Self {
            src: exec.alloc_u8(max_pixels * 3)?,
            mask: exec.alloc_u8(max_pixels)?,
            starts: exec.alloc_u32(RLE_CAP)?,
            counts: exec.alloc_u32(RLE_CAP)?,
            nruns: exec.alloc_u32(1)?,
            picture: PictureCache::default(),
            last_text: None,
            pvs: if exec.has_sam3_tracker() {
                Some(GpuSam3Pvs::load_dir(exec.clone(), dir)?)
            } else {
                tracing::warn!("this kernel pack predates SAM 3's click prompts: concepts only");
                None
            },
            last_pvs: None,
            next_refine: 1,
            dir: dir.to_path_buf(),
            video: None,
            video_off: String::new(),
            exec,
            vision,
            text,
            det,
            max_pixels,
        }
        .with_video())
    }

    fn with_video(mut self) -> Self {
        self.load_video();
        self
    }

    pub fn max_pixels(&self) -> usize {
        self.max_pixels
    }
    pub fn max_boxes(&self) -> usize {
        self.det.geom().max_boxes
    }
    pub fn weight_bytes(&self) -> u64 {
        self.vision.weight_bytes()
            + self.text.weight_bytes()
            + self.det.weight_bytes()
            + self.pvs.as_ref().map_or(0, GpuSam3Pvs::weight_bytes)
            + self.video.as_ref().map_or(0, |v| v.weight_bytes)
    }
    /// Whether click prompts are served (the pack carries their heads).
    pub fn has_clicks(&self) -> bool {
        self.pvs.is_some()
    }
    /// The resized 1008^2 picture the last request fed the encoder - the
    /// gate's view of the resize.
    pub fn read_input(&self) -> Result<Vec<u8>, GpuModelError> {
        let s = self.vision.config().image_size;
        Ok(self.exec.to_host_u8_len(self.vision.input(), s * s * 3)?)
    }

    pub fn workspace_bytes(&self) -> u64 {
        self.vision.workspace_bytes()
            + self.det.workspace_bytes()
            + self.pvs.as_ref().map_or(0, GpuSam3Pvs::workspace_bytes)
    }

    /// The vision planes are about to hold something else (a video frame):
    /// no cached picture, no click answer to refine.
    pub(super) fn picture_gone(&mut self) {
        self.picture = PictureCache::default();
        self.last_pvs = None;
    }

    /// Drop both reuse records: the next request encodes its picture and its
    /// prompt whatever they are (the bench's cold path).
    pub fn forget(&mut self) {
        self.picture = PictureCache::default();
        self.last_text = None;
        self.last_pvs = None;
    }

    /// One picture (`rgb` u8 HWC, `width x height`) and one prompt.
    pub fn segment(
        &mut self,
        rgb: &[u8],
        width: usize,
        height: usize,
        req: &Sam3Request,
    ) -> Result<Sam3Output, Sam3Fail> {
        let px = width * height;
        if px == 0 || rgb.len() != px * 3 {
            return Err(Sam3Fail::Request(format!(
                "{} bytes for a {width}x{height} RGB picture",
                rgb.len()
            )));
        }
        if px > self.max_pixels {
            return Err(Sam3Fail::Request(format!(
                "the picture is {width}x{height} ({px} pixels); this endpoint takes at most {}",
                self.max_pixels
            )));
        }
        if req.valid < 2 || req.valid > req.ids.len() {
            return Err(Sam3Fail::Request(format!(
                "{} valid prompt tokens (want SOT and EOT around at least one, of 32)",
                req.valid
            )));
        }
        if req.boxes.len() > self.max_boxes() {
            return Err(Sam3Fail::Request(format!(
                "{} exemplar boxes; this endpoint takes at most {}",
                req.boxes.len(),
                self.max_boxes()
            )));
        }
        let exec = self.exec.clone();
        let Self {
            vision,
            text: tower,
            det,
            src,
            mask,
            starts,
            counts,
            nruns,
            picture,
            last_text,
            last_pvs,
            ..
        } = self;
        let mut timings = Sam3Timings::default();
        if ensure_picture(
            &exec,
            vision,
            src,
            picture,
            rgb,
            width,
            height,
            false,
            &mut timings,
        )? {
            *last_pvs = None;
        }

        let t0 = Instant::now();
        timings.text_reused = last_text.as_ref() == Some(&req.ids);
        if !timings.text_reused {
            *last_text = None;
            tower.encode(&req.ids, 1)?;
            *last_text = Some(req.ids);
        }
        det.encode_prompt(
            tower.features(),
            0,
            req.valid,
            vision.det_level(2),
            &req.boxes,
        )?;
        exec.synchronize()?;
        timings.prompt_ms = t0.elapsed().as_secs_f64() * 1e3;

        let t0 = Instant::now();
        det.fuse(vision.det_level(2))?;
        det.decode()?;
        let dets = det.read_detections()?;
        // the checkpoint's own scoring (SAM 3.1 folds presence in, joint)
        let scores = det.picture_scores(&dets);
        timings.detect_ms = t0.elapsed().as_secs_f64() * 1e3;

        // the control decision: which queries are kept, best first
        let mut kept: Vec<usize> = (0..scores.len())
            .filter(|&q| scores[q] > req.threshold)
            .collect();
        kept.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));

        let t0 = Instant::now();
        let mut instances = Vec::with_capacity(kept.len());
        if !kept.is_empty() {
            det.segment(vision.det_level(0), vision.det_level(1))?;
            let g = det.geom();
            let side = 4 * g.grid;
            for &q in &kept {
                exec.sam3_mask_up(det.masks_plane(), mask, side, g.queries, q, height, width)?;
                exec.sam3_rle(mask, starts, counts, nruns, px, RLE_CAP)?;
                let n = exec.to_host_u32(nruns)?[0];
                if n == u32::MAX {
                    return Err(Sam3Fail::Request(format!(
                        "instance {q}'s mask needs more than {RLE_CAP} runs"
                    )));
                }
                let rle = exec.to_host_u32_len(counts, n as usize)?;
                let area = rle.iter().skip(1).step_by(2).map(|&c| c as u64).sum();
                // Meta's processor: cxcywh -> xyxy, times [W, H, W, H]
                let [cx, cy, bw, bh] = dets.boxes[q];
                let (wf, hf) = (width as f32, height as f32);
                instances.push(Sam3Instance {
                    score: scores[q],
                    bbox: [
                        (cx - 0.5 * bw) * wf,
                        (cy - 0.5 * bh) * hf,
                        (cx + 0.5 * bw) * wf,
                        (cy + 0.5 * bh) * hf,
                    ],
                    rle,
                    area,
                });
            }
        }
        timings.masks_ms = t0.elapsed().as_secs_f64() * 1e3;
        Ok(Sam3Output {
            width,
            height,
            presence: 1.0 / (1.0 + (-dets.presence_logit).exp()),
            instances,
            tokens: req.valid,
            timings,
            refine_id: None,
        })
    }

    /// One picture and clicks and/or a box: the object under them, as up to
    /// three candidate masks best first, scored by the model's predicted IoU.
    /// `presence` is the object-score head's probability that there is an
    /// object there at all.
    pub fn segment_points(
        &mut self,
        rgb: &[u8],
        width: usize,
        height: usize,
        req: &Sam3PointRequest,
    ) -> Result<Sam3Output, Sam3Fail> {
        let px = width * height;
        if px == 0 || rgb.len() != px * 3 {
            return Err(Sam3Fail::Request(format!(
                "{} bytes for a {width}x{height} RGB picture",
                rgb.len()
            )));
        }
        if px > self.max_pixels {
            return Err(Sam3Fail::Request(format!(
                "the picture is {width}x{height} ({px} pixels); this endpoint takes at most {}",
                self.max_pixels
            )));
        }
        if req.clicks.is_empty() && req.bbox.is_none() {
            return Err(Sam3Fail::Request(
                "a click prompt needs at least one click or an object box".into(),
            ));
        }
        if req.clicks.len() > PVS_MAX_POINTS {
            return Err(Sam3Fail::Request(format!(
                "{} clicks; this endpoint takes at most {PVS_MAX_POINTS}",
                req.clicks.len()
            )));
        }
        if self.pvs.is_none() {
            return Err(Sam3Fail::Request(
                "this endpoint's kernel pack predates click prompts - update it to use them".into(),
            ));
        }
        let exec = self.exec.clone();
        let Self {
            vision,
            src,
            mask,
            starts,
            counts,
            nruns,
            picture,
            pvs,
            last_pvs,
            next_refine,
            ..
        } = self;
        let pvs = pvs.as_mut().expect("checked above");
        let mut timings = Sam3Timings::default();
        if ensure_picture(
            &exec,
            vision,
            src,
            picture,
            rgb,
            width,
            height,
            true,
            &mut timings,
        )? {
            *last_pvs = None;
        }
        let mask_prompt = match req.refine {
            None => None,
            Some(id) => match *last_pvs {
                Some((last, k)) if last == id => Some(PvsMask::Last(k)),
                _ => {
                    return Err(Sam3Fail::Request(format!(
                        "refine {id} is not the last click answer on this picture - send the \
                         clicks without it"
                    )));
                }
            },
        };

        // Meta's transform: pixel / size * 1008, then the encoder's +0.5 and / 1008
        let s = vision.config().image_size as f32;
        let (wf, hf) = (width as f32, height as f32);
        let nx = |x: f32| (x / wf * s + 0.5) / s;
        let ny = |y: f32| (y / hf * s + 0.5) / s;
        let prompt = PvsPrompt {
            points: req
                .clicks
                .iter()
                .map(|c| PvsPoint {
                    x: nx(c.x),
                    y: ny(c.y),
                    positive: c.positive,
                })
                .collect(),
            bbox: req
                .bbox
                .map(|[x0, y0, x1, y1]| [nx(x0), ny(y0), nx(x1), ny(y1)]),
            mask: mask_prompt,
            multimask: req
                .multimask
                .unwrap_or(req.clicks.len() <= 1 && req.bbox.is_none() && req.refine.is_none()),
        };
        let t0 = Instant::now();
        let res = pvs.predict(vision, &prompt)?;
        timings.detect_ms = t0.elapsed().as_secs_f64() * 1e3;
        let id = *next_refine;
        *next_refine += 1;
        *last_pvs = res.candidates.first().map(|&(k, _)| (id, k));

        let t0 = Instant::now();
        let side = 4 * vision.config().grid();
        let mut instances = Vec::with_capacity(res.candidates.len());
        for &(k, score) in &res.candidates {
            exec.sam3_mask_up(pvs.masks(), mask, side, 4, k, height, width)?;
            exec.sam3_rle(mask, starts, counts, nruns, px, RLE_CAP)?;
            let n = exec.to_host_u32(nruns)?[0];
            if n == u32::MAX {
                return Err(Sam3Fail::Request(format!(
                    "the mask needs more than {RLE_CAP} runs"
                )));
            }
            let rle = exec.to_host_u32_len(counts, n as usize)?;
            let area = rle.iter().skip(1).step_by(2).map(|&c| c as u64).sum();
            instances.push(Sam3Instance {
                score,
                bbox: rle_box(&rle, height),
                rle,
                area,
            });
        }
        timings.masks_ms = t0.elapsed().as_secs_f64() * 1e3;
        Ok(Sam3Output {
            width,
            height,
            presence: res.object_score,
            instances,
            tokens: 0,
            timings,
            refine_id: Some(id),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::rle_box;

    #[test]
    fn rle_box_reads_column_major_runs() {
        // h = 4, w = 3: one run in column 1, rows 1..=2
        assert_eq!(rle_box(&[5, 2, 5], 4), [1.0, 1.0, 2.0, 3.0]);
        // a run across the column 0/1 boundary touches the bottom and top rows
        assert_eq!(rle_box(&[3, 2, 7], 4), [0.0, 0.0, 2.0, 4.0]);
        // nothing set
        assert_eq!(rle_box(&[12], 4), [0.0; 4]);
    }
}
