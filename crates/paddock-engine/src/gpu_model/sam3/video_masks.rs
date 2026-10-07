//! SAM 3's video masks on the device, for the decisions and the outputs:
//! the 288^2 low-res masks of a frame's detections and tracks as the IoUs and
//! areas the planner reads (`video_plan.rs`), Meta's hole and sprinkle fill,
//! and the outputs at the video's own size - the threshold, the boxes, one
//! owner a pixel, COCO RLE.
//!
//! Masks are f32 logit planes of 288^2, one object a plane; "the mask" is
//! logit > 0 everywhere in Meta's loop.

use std::sync::Arc;

use cudarc::driver::CudaSlice;

use super::GpuModelError;
use super::video_plan::{FILL_HOLE_AREA, mask_iou};
use crate::gpu::GpuExecutor;

/// The low-res mask side (1008 / 14 * 4) and a plane's values.
pub const LOW_RES: usize = 288;
pub const LOW_PX: usize = LOW_RES * LOW_RES;
const WORDS: usize = LOW_PX.div_ceil(32);
/// Meta's hole fill writes holes at 0.1 and sprinkles at -0.1.
const HOLE_VAL: f32 = 0.1;
const SPRINKLE_VAL: f32 = -0.1;
/// Meta's suppressed-track logit.
pub const NO_OBJ_LOGIT: f32 = -10.0;
/// COCO RLE capacity a mask.
const RLE_CAP: usize = 1 << 20;

/// Two plane sets' IoUs and areas.
pub struct Sam3Ious {
    /// `[a][b]`
    pub iou: Vec<f32>,
    pub area_a: Vec<u32>,
    pub area_b: Vec<u32>,
}

/// One object of a frame's output: its id, probability, box (x, y, w, h
/// normalized by the video's size, Meta's `out_boxes_xywh`) and mask.
#[derive(Clone, Debug)]
pub struct Sam3VideoObject {
    pub id: i64,
    /// the session concept it was found for (0 when there is one)
    pub concept: usize,
    pub prob: f32,
    pub bbox_xywh: [f32; 4],
    /// the same box in pixels, x0 y0 x1 y1 with the extremes inclusive
    pub bbox_px: [u32; 4],
    /// column-major COCO RLE at the video's size, zeros first
    pub rle: Vec<u32>,
    pub area: u64,
}

/// The device side of the video masks: bitplanes, CC scratch, output planes.
pub struct GpuSam3VideoMasks {
    exec: Arc<GpuExecutor>,
    cap: usize,
    bits_a: CudaSlice<u32>,
    bits_b: CudaSlice<u32>,
    area_a: CudaSlice<u32>,
    area_b: CudaSlice<u32>,
    inter: CudaSlice<u32>,
    lab: CudaSlice<u32>,
    larea: CudaSlice<u32>,
    tot: CudaSlice<u32>,
    /// output masks at the video's size, `[out_cap][h * w]` column-major
    out: CudaSlice<u8>,
    out_cap: usize,
    out_px: usize,
    scores: CudaSlice<f32>,
    boxes: CudaSlice<u32>,
    /// one output mask at a time for the RLE (which reads from element 0)
    one: CudaSlice<u8>,
    starts: CudaSlice<u32>,
    counts: CudaSlice<u32>,
    nruns: CudaSlice<u32>,
}

impl GpuSam3VideoMasks {
    pub fn new(exec: Arc<GpuExecutor>) -> Result<Self, GpuModelError> {
        if !exec.has_sam3_video() {
            return Err(GpuModelError::Unsupported(
                "this kernel pack predates SAM 3's video path (slots 792-805) - rebuild or \
                 update the pack"
                    .into(),
            ));
        }
        let cap = 8;
        let u = |n: usize| exec.alloc_u32(n);
        Ok(Self {
            bits_a: u(cap * WORDS)?,
            bits_b: u(cap * WORDS)?,
            area_a: u(cap)?,
            area_b: u(cap)?,
            inter: u(cap * cap)?,
            lab: u(cap * LOW_PX)?,
            larea: u(cap * LOW_PX)?,
            tot: u(cap)?,
            out: exec.alloc_u8(1)?,
            out_cap: 0,
            out_px: 0,
            scores: exec.alloc(cap)?,
            boxes: u(cap * 5)?,
            one: exec.alloc_u8(1)?,
            starts: u(RLE_CAP)?,
            counts: u(RLE_CAP)?,
            nruns: u(1)?,
            cap,
            exec,
        })
    }

    /// Room for `n` planes a call.
    fn ensure(&mut self, n: usize) -> Result<(), GpuModelError> {
        if n <= self.cap {
            return Ok(());
        }
        let cap = n.next_power_of_two();
        let u = |k: usize| self.exec.alloc_u32(k);
        self.bits_a = u(cap * WORDS)?;
        self.bits_b = u(cap * WORDS)?;
        self.area_a = u(cap)?;
        self.area_b = u(cap)?;
        self.inter = u(cap * cap)?;
        self.lab = u(cap * LOW_PX)?;
        self.larea = u(cap * LOW_PX)?;
        self.tot = u(cap)?;
        self.scores = self.exec.alloc(cap)?;
        self.boxes = u(cap * 5)?;
        self.cap = cap;
        Ok(())
    }

    /// The IoUs of plane sets `a` (`n` planes from element `off`, 288^2
    /// apart) and `b` (or `a` with itself), as Meta's `mask_iou` forms
    /// them, and both sets' areas.
    pub fn ious(
        &mut self,
        a: (&CudaSlice<f32>, usize, usize),
        b: Option<(&CudaSlice<f32>, usize, usize)>,
    ) -> Result<Sam3Ious, GpuModelError> {
        let (pa, oa, na) = a;
        let nb = b.map_or(na, |x| x.2);
        self.ensure(na.max(nb))?;
        let exec = self.exec.clone();
        let read = |buf: &CudaSlice<u32>, n: usize| -> Result<Vec<u32>, GpuModelError> {
            Ok(if n == 0 {
                Vec::new()
            } else {
                exec.to_host_u32_len(buf, n)?
            })
        };
        exec.sam3_mask_bits(
            pa,
            oa,
            LOW_PX,
            na,
            LOW_PX,
            &mut self.bits_a,
            &mut self.area_a,
        )?;
        let other = match b {
            Some((pb, ob, nb)) => {
                exec.sam3_mask_bits(
                    pb,
                    ob,
                    LOW_PX,
                    nb,
                    LOW_PX,
                    &mut self.bits_b,
                    &mut self.area_b,
                )?;
                &self.bits_b
            }
            None => &self.bits_a,
        };
        exec.sam3_mask_pairs(&self.bits_a, other, WORDS, (na, nb), &mut self.inter)?;
        let inter = read(&self.inter, na * nb)?;
        let area_a = read(&self.area_a, na)?;
        let area_b = match b {
            Some(_) => read(&self.area_b, nb)?,
            None => area_a.clone(),
        };
        let mut iou = vec![0f32; na * nb];
        for i in 0..na {
            for j in 0..nb {
                iou[i * nb + j] = mask_iou(inter[i * nb + j], area_a[i], area_b[j]);
            }
        }
        Ok(Sam3Ious {
            iou,
            area_a,
            area_b,
        })
    }

    /// Meta's `fill_holes_in_mask_scores` (area 16, holes then sprinkles) on
    /// `n` planes from element `off`.
    pub fn clean(
        &mut self,
        planes: &mut CudaSlice<f32>,
        off: usize,
        n: usize,
    ) -> Result<(), GpuModelError> {
        self.ensure(n)?;
        self.exec.sam3_mask_clean(
            planes,
            off,
            LOW_PX,
            LOW_RES,
            n,
            FILL_HOLE_AREA,
            (HOLE_VAL, SPRINKLE_VAL),
            (&mut self.lab, &mut self.larea, &mut self.tot),
        )?;
        Ok(())
    }

    /// One plane (from element `off`) set to Meta's no-object logit, -10.
    pub fn suppress(
        &mut self,
        planes: &mut CudaSlice<f32>,
        off: usize,
    ) -> Result<(), GpuModelError> {
        self.exec
            .sam3_mask_set(planes, off, LOW_PX, NO_OBJ_LOGIT, false)?;
        Ok(())
    }

    fn ensure_out(&mut self, n: usize, px: usize) -> Result<(), GpuModelError> {
        if n <= self.out_cap && px == self.out_px {
            return Ok(());
        }
        let cap = n.max(self.out_cap).max(1).next_power_of_two();
        self.out = self.exec.alloc_u8(cap * px)?;
        if self.one.len() < px {
            self.one = self.exec.alloc_u8(px)?;
        }
        self.out_cap = cap;
        self.out_px = px;
        Ok(())
    }

    /// Render `planes` (each a 288^2 plane: slice and element offset) at
    /// `h x w`, logit > 0, into output masks `0..n` (column-major).
    pub fn render(
        &mut self,
        planes: &[(&CudaSlice<f32>, usize)],
        h: usize,
        w: usize,
    ) -> Result<(), GpuModelError> {
        self.ensure_out(planes.len(), h * w)?;
        for (i, &(p, off)) in planes.iter().enumerate() {
            self.exec.sam3_mask_up2(
                p,
                off,
                (LOW_RES, 1, 0),
                &mut self.out,
                i * h * w,
                h,
                w,
                true,
            )?;
        }
        Ok(())
    }

    /// Output masks `0..n`'s `{x0, y0, x1, y1, area}`.
    pub fn boxes(&mut self, n: usize, h: usize, w: usize) -> Result<Vec<[u32; 5]>, GpuModelError> {
        if n == 0 {
            return Ok(Vec::new());
        }
        self.ensure(n)?;
        self.exec
            .sam3_mask_boxes(&self.out, n, h, w, &mut self.boxes)?;
        let v = self.exec.to_host_u32_len(&self.boxes, n * 5)?;
        Ok(v.as_chunks::<5>().0.to_vec())
    }

    /// Output mask `i` as host bytes (column-major `[w][h]`), for the gate.
    pub fn read_mask(&self, i: usize) -> Result<Vec<u8>, GpuModelError> {
        let px = self.out_px;
        let all = self.exec.to_host_u8_len(&self.out, (i + 1) * px)?;
        Ok(all[i * px..].to_vec())
    }

    /// Meta's `_postprocess_output` for one frame: the objects (id,
    /// probability, tracker probability, plane) in any order, the ids to
    /// hide, the video's size. Sorted by id, hidden and empty masks dropped,
    /// boxes from the masks as they are, then one owner a pixel by tracker
    /// probability; masks as RLE. The final masks stay in output masks
    /// `0..len` for [`Self::read_mask`].
    pub fn outputs(
        &mut self,
        objects: &[(i64, f32, f32, &CudaSlice<f32>, usize)],
        hidden: &dyn Fn(i64) -> bool,
        h: usize,
        w: usize,
    ) -> Result<Vec<Sam3VideoObject>, GpuModelError> {
        let mut order: Vec<usize> = (0..objects.len())
            .filter(|&i| !hidden(objects[i].0))
            .collect();
        order.sort_by_key(|&i| objects[i].0);
        let planes: Vec<(&CudaSlice<f32>, usize)> = order
            .iter()
            .map(|&i| (objects[i].3, objects[i].4))
            .collect();
        self.render(&planes, h, w)?;
        let boxes = self.boxes(planes.len(), h, w)?;
        let kept: Vec<usize> = (0..order.len()).filter(|&k| boxes[k][4] > 0).collect();
        if kept.len() != order.len() {
            // render the non-empty ones again, contiguous for the owner pass
            let planes: Vec<(&CudaSlice<f32>, usize)> = kept.iter().map(|&k| planes[k]).collect();
            self.render(&planes, h, w)?;
        }
        let n = kept.len();
        let (wf, hf) = (w as f32, h as f32);
        let mut out = Vec::with_capacity(n);
        for &k in &kept {
            let o = &objects[order[k]];
            let b = boxes[k];
            out.push(Sam3VideoObject {
                id: o.0,
                concept: 0,
                prob: o.1,
                // torchvision's xyxy -> xywh: the extremes are inclusive and
                // the width their difference
                bbox_xywh: [
                    b[0] as f32 / wf,
                    b[1] as f32 / hf,
                    (b[2] as f32 - b[0] as f32) / wf,
                    (b[3] as f32 - b[1] as f32) / hf,
                ],
                bbox_px: [b[0], b[1], b[2], b[3]],
                rle: Vec::new(),
                area: 0,
            });
        }
        if n > 1 {
            self.ensure(n)?;
            let tp: Vec<f32> = kept.iter().map(|&k| objects[order[k]].2).collect();
            self.exec.upload_f32(&tp, &mut self.scores)?;
            self.exec
                .sam3_mask_owner(&mut self.out, &self.scores, n, h * w)?;
        }
        let px = h * w;
        for (i, o) in out.iter_mut().enumerate() {
            self.exec
                .copy_region(&self.out, i * px, &mut self.one, 0, px)?;
            self.exec.sam3_rle(
                &self.one,
                &mut self.starts,
                &mut self.counts,
                &mut self.nruns,
                px,
                RLE_CAP,
            )?;
            let nr = self.exec.to_host_u32(&self.nruns)?[0];
            if nr == u32::MAX {
                return Err(GpuModelError::Unsupported(format!(
                    "sam3 video: object {}'s mask needs more than {RLE_CAP} runs",
                    o.id
                )));
            }
            o.rle = self.exec.to_host_u32_len(&self.counts, nr as usize)?;
            o.area = o.rle.iter().skip(1).step_by(2).map(|&c| c as u64).sum();
        }
        Ok(out)
    }
}
