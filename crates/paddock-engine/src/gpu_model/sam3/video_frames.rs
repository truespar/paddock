//! SAM 3's video frames in, the way Meta's JPEG-folder loader makes them:
//! a decoded RGB frame resized to the model's 1008^2 by Pillow's bilinear
//! (torchvision's TF.resize on a PIL image) - two separable passes,
//! horizontal then vertical, 22-bit integer weights, u8 between them - into
//! the image encoder's input plane. The encoder then normalizes it as the
//! loader stores it ([`super::GpuSam3Vision::set_video_frames`]).
//!
//! Decoding the frame is not the model's and stays the caller's: Meta's
//! loader runs libjpeg-turbo through PIL, and the gate hands the engine
//! PIL's decoded frames.

use std::sync::Arc;

use cudarc::driver::CudaSlice;

use super::GpuModelError;
use crate::gpu::{GpuExecutor, pil_ksize};

/// Pillow's taps for one axis.
struct Axis {
    bounds: CudaSlice<i32>,
    kk: CudaSlice<i32>,
}

/// The frame staging and the taps for the frames' size.
pub struct GpuSam3FrameIn {
    exec: Arc<GpuExecutor>,
    side: usize,
    /// (h, w) the taps are for
    geom: Option<(usize, usize)>,
    horiz: Option<Axis>,
    vert: Option<Axis>,
    src: CudaSlice<u8>,
    tmp: CudaSlice<u8>,
}

impl GpuSam3FrameIn {
    pub fn new(exec: Arc<GpuExecutor>, side: usize) -> Result<Self, GpuModelError> {
        Ok(Self {
            src: exec.alloc_u8(1)?,
            tmp: exec.alloc_u8(1)?,
            exec,
            side,
            geom: None,
            horiz: None,
            vert: None,
        })
    }

    fn axis(&self, in_size: usize) -> Result<Option<Axis>, GpuModelError> {
        if in_size == self.side {
            // Pillow skips a pass along an axis that keeps its size
            return Ok(None);
        }
        let k = pil_ksize(in_size, self.side);
        if k > 64 {
            return Err(GpuModelError::Unsupported(format!(
                "sam3 video: a {in_size}-pixel side is over 31x the model's {}",
                self.side
            )));
        }
        let mut bounds = self.exec.alloc_i32(2 * self.side)?;
        let mut kk = self.exec.alloc_i32(self.side * k)?;
        self.exec
            .sam3_pil_coeffs(in_size, self.side, &mut bounds, &mut kk)?;
        Ok(Some(Axis { bounds, kk }))
    }

    /// Resize `rgb` (u8 HWC, `h x w x 3`) into `dst`, the encoder's input
    /// plane (`side x side x 3`).
    pub fn land(
        &mut self,
        rgb: &[u8],
        h: usize,
        w: usize,
        dst: &mut CudaSlice<u8>,
    ) -> Result<(), GpuModelError> {
        let s = self.side;
        if h == 0 || w == 0 || rgb.len() != h * w * 3 {
            return Err(GpuModelError::Unsupported(format!(
                "sam3 video: {} bytes for a {w}x{h} RGB frame",
                rgb.len()
            )));
        }
        if dst.len() < s * s * 3 {
            return Err(GpuModelError::Unsupported(
                "sam3 video: the input plane is under the model's side".into(),
            ));
        }
        if self.geom != Some((h, w)) {
            self.horiz = self.axis(w)?;
            self.vert = self.axis(h)?;
            self.src = self.exec.alloc_u8(h * w * 3)?;
            self.tmp = self.exec.alloc_u8(h * s * 3)?;
            self.geom = Some((h, w));
        }
        let exec = self.exec.clone();
        exec.upload_u8(rgb, &mut self.src)?;
        // horizontal first, the full height of the frame: [h][w] -> [h][s]
        let (mid, mid_w) = match &self.horiz {
            Some(a) => {
                exec.sam3_pil_pass(
                    &self.src,
                    &mut self.tmp,
                    (&a.bounds, &a.kk),
                    h,
                    (w, s),
                    3,
                    0,
                )?;
                (&self.tmp, s)
            }
            None => (&self.src, w),
        };
        match &self.vert {
            Some(a) => {
                exec.sam3_pil_pass(mid, dst, (&a.bounds, &a.kk), mid_w, (h, s), 3, 1)?;
            }
            None => exec.copy_region(mid, 0, dst, 0, s * s * 3)?,
        }
        Ok(())
    }
}
