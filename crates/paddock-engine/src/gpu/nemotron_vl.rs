//! Nemotron 3.5 Super VL picture-lane ops (slots 904-907, the family's glue
//! in `packs/cuda/src/nemotron_vl.cuh`). The graph is
//! `gpu_model::nemotron::vision`.
//!
//! Every buffer is checked against the geometry before a launch; the resize
//! plan (per-output taps and weights) is the caller's to build in range.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::f16;

use super::error::*;
use super::*;

/// One axis of a [`GpuExecutor::nvl_resample`] pass on the device: per
/// output index the first input index, the tap count, and `taps` f32
/// weights (zero past the count).
pub struct ResampleAxis {
    pub xmin: CudaSlice<u32>,
    pub xsize: CudaSlice<u32>,
    pub w: CudaSlice<f32>,
    pub taps: usize,
    pub out: usize,
}

impl GpuExecutor {
    /// Whether the pack carries the picture lane's four kernels.
    pub fn has_nemotron_vl(&self) -> bool {
        let k = &self.kernels;
        k.nvl_resample.is_some()
            && k.nvl_patchify_h.is_some()
            && k.nvl_pos_add.is_some()
            && k.nvl_relu2.is_some()
    }

    /// Slot 904: one separable pass of the float antialiased bicubic.
    /// `horiz`: `src` is `[lines][in][ch]`, `dst` `[lines][out][ch]`; else
    /// `src` `[in][lines][ch]`, `dst` `[out][lines][ch]`. `src_u8` reads the
    /// source as bytes (the decoded picture), else as f32.
    #[allow(clippy::too_many_arguments)]
    pub fn nvl_resample(
        &self,
        src_u8: Option<&CudaSlice<u8>>,
        src_f32: Option<&CudaSlice<f32>>,
        dst: &mut CudaSlice<f32>,
        lines: usize,
        in_len: usize,
        axis: &ResampleAxis,
        horiz: bool,
        ch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .nvl_resample
            .ok_or(GpuError::MissingOp("nvl_resample"))?;
        let need_src = lines * in_len * ch;
        let need_dst = lines * axis.out * ch;
        let src_len = src_u8
            .map(|s| s.len())
            .or(src_f32.map(|s| s.len()))
            .unwrap_or(0);
        if src_len < need_src
            || dst.len() < need_dst
            || axis.xmin.len() < axis.out
            || axis.xsize.len() < axis.out
            || axis.w.len() < axis.out * axis.taps
        {
            return Err(oob("nvl_resample: buffers under the pass geometry"));
        }
        let (dp, _g1) = dst.device_ptr_mut(&self.stream);
        let (xp, _g2) = axis.xmin.device_ptr(&self.stream);
        let (sp, _g3) = axis.xsize.device_ptr(&self.stream);
        let (wp, _g4) = axis.w.device_ptr(&self.stream);
        let (src, is_u8, _g5) = match (src_u8, src_f32) {
            (Some(s), _) => {
                let (p, g) = s.device_ptr(&self.stream);
                (p, 1u32, g)
            }
            (None, Some(s)) => {
                let (p, g) = s.device_ptr(&self.stream);
                (p, 0u32, g)
            }
            (None, None) => return Err(oob("nvl_resample: no source")),
        };
        // SAFETY: ABI contract (slot 904); bounds checked above
        check(unsafe {
            f(
                src as *const _,
                dp as *mut _,
                is_u8,
                lines as u32,
                in_len as u32,
                axis.out as u32,
                u32::from(horiz),
                xp as *const _,
                sp as *const _,
                wp as *const _,
                axis.taps as u32,
                ch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 905: the normalized patch rows of an f32 HWC picture (`h` x `w`
    /// on the 0..255 scale) as f16 `[(h/p)(w/p)][3 p p]`, flattened
    /// (c, ky, kx).
    #[allow(clippy::too_many_arguments)]
    pub fn nvl_patchify_h(
        &self,
        img: &CudaSlice<f32>,
        out: &mut CudaSlice<f16>,
        h: usize,
        w: usize,
        p: usize,
        mean: [f32; 3],
        std: [f32; 3],
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .nvl_patchify_h
            .ok_or(GpuError::MissingOp("nvl_patchify_h"))?;
        if p == 0 || !h.is_multiple_of(p) || !w.is_multiple_of(p) {
            return Err(oob("nvl_patchify_h: picture not a whole number of patches"));
        }
        let rows = (h / p) * (w / p);
        if img.len() < h * w * 3 || out.len() < rows * 3 * p * p {
            return Err(oob("nvl_patchify_h: buffers under the picture"));
        }
        let (ip, _g1) = img.device_ptr(&self.stream);
        let (op, _g2) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 905); bounds checked above
        check(unsafe {
            f(
                ip as *const _,
                op as *mut _,
                h as u32,
                w as u32,
                p as u32,
                mean[0],
                mean[1],
                mean[2],
                std[0],
                std[1],
                std[2],
                self.stream_ptr(),
            )
        })
    }

    /// Slot 906: add RADIO's cropped position field to the `hp x wp` patch
    /// rows starting at row `row0` of `x` (`d` wide): the `g x g` grid
    /// resized to `m x m`, `m` = max(hp, wp).
    #[allow(clippy::too_many_arguments)]
    pub fn nvl_pos_add(
        &self,
        x: &mut CudaSlice<f32>,
        row0: usize,
        grid: &CudaSlice<f32>,
        g: usize,
        hp: usize,
        wp: usize,
        d: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .nvl_pos_add
            .ok_or(GpuError::MissingOp("nvl_pos_add"))?;
        // m may pass g: the grid then stretches rather than shrinks
        let m = hp.max(wp);
        if g == 0 || grid.len() < g * g * d || x.len() < (row0 + hp * wp) * d {
            return Err(oob("nvl_pos_add: buffers under the field"));
        }
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (gp, _g2) = grid.device_ptr(&self.stream);
        let off = (row0 * d * 4) as u64;
        // SAFETY: ABI contract (slot 906); bounds checked above
        check(unsafe {
            f(
                (xp + off) as *mut _,
                gp as *const _,
                g as u32,
                m as u32,
                hp as u32,
                wp as u32,
                d as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 907: `x = relu(x)^2` over the first `n` elements.
    pub fn nvl_relu2(&self, x: &mut CudaSlice<f32>, n: usize) -> Result<(), GpuError> {
        let f = self
            .kernels
            .nvl_relu2
            .ok_or(GpuError::MissingOp("nvl_relu2"))?;
        if x.len() < n {
            return Err(oob("nvl_relu2: buffer under n"));
        }
        let (xp, _g) = x.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 907); bounds checked above
        check(unsafe { f(xp as *mut _, n as u64, self.stream_ptr()) })
    }
}
