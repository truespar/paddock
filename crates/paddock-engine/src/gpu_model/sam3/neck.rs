//! The two FPN necks (Meta's `Sam3DualViTDetNeck`), off the trunk's raster:
//!
//!   x4: convT 2x2/s2 (d -> d/2) -> GELU -> convT (d/2 -> d/4) -> 1x1 -> 3x3
//!   x2: convT (d -> d/2) -> 1x1 -> 3x3
//!   x1: 1x1 -> 3x3
//!
//! all to `fpn_dim` (256) channels, f32 rasters [pics][side^2][256] at 288,
//! 144 and 72. The detector's neck always runs; the tracker's (separate
//! weights, same shape) only when asked, followed by the tracker mask
//! decoder's conv_s0 / conv_s1 over its x4 / x2 levels - Meta applies those at
//! `set_image`, so its golden levels are the post-1x1 planes.
//!
//! Every convT is its GEMM (C_in -> 4 C_out, f32 landing) + the depth-to-space
//! seam that adds the bias (and the GELU) and lands f16 for the next GEMM.
//! Every 1x1 is a GEMM; the 3x3 is the dense lane's im2row + GEMM - the
//! interim the CUDA side documents (implicit-GEMM conv is the target).

use super::{GpuModelError, GpuSam3Vision, Neck};

impl GpuSam3Vision {
    pub(super) fn necks(&mut self, pics: usize, tracker: bool) -> Result<(), GpuModelError> {
        self.neck(pics, false)?;
        if tracker {
            self.necks_tracker_only(pics)?;
        }
        Ok(())
    }

    /// The tracker neck and its conv_s0 / conv_s1, off the trunk raster the
    /// backbone left in the workspace.
    pub(super) fn necks_tracker_only(&mut self, pics: usize) -> Result<(), GpuModelError> {
        self.neck(pics, true)?;
        let exec = self.exec.clone();
        let ws = &mut self.ws;
        let f = self.cfg.fpn_dim;
        for (lv, conv, out) in [
            (0usize, &self.conv_s0, &mut ws.trk_s0),
            (1usize, &self.conv_s1, &mut ws.trk_s1),
        ] {
            let px = pics * ((self.cfg.grid() * 4) >> lv).pow(2);
            exec.convert_f32_f16(&ws.trk[lv], &mut ws.h16b, px * f)?;
            exec.matvec_batch_f16(&conv.w, &ws.h16b, out, px)?;
            exec.bias_add(out, &conv.b, px, conv.w.dims[1])?;
        }
        Ok(())
    }

    fn neck(&mut self, pics: usize, tracker: bool) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let g = self.cfg.grid();
        let f = self.cfg.fpn_dim;
        let neck: &Neck = if tracker {
            &self.trk_neck
        } else {
            &self.det_neck
        };
        let ws = &mut self.ws;
        let levels = if tracker { &mut ws.trk } else { &mut ws.det };

        // ---- x4: two convTs, the first followed by a GELU ----
        let [up0, up1] = &neck.x4_up;
        exec.matvec_batch_f16(&up0.w, &ws.n16, &mut ws.g32, pics * g * g)?;
        exec.sam3_convt2_bias_h(&ws.g32, &up0.b, &mut ws.h16a, pics, g, g, up0.cout, true)?;
        let g2 = 2 * g;
        exec.matvec_batch_f16(&up1.w, &ws.h16a, &mut ws.g32, pics * g2 * g2)?;
        exec.sam3_convt2_bias_h(&ws.g32, &up1.b, &mut ws.h16b, pics, g2, g2, up1.cout, false)?;
        let s4 = 4 * g;
        proj(
            &exec,
            &neck.proj1[0],
            &neck.proj2[0],
            &ws.h16b,
            &mut ws.y32,
            &mut ws.col16,
            &mut levels[0],
            pics,
            s4,
            f,
        )?;

        // ---- x2: one convT ----
        exec.matvec_batch_f16(&neck.x2_up.w, &ws.n16, &mut ws.g32, pics * g * g)?;
        exec.sam3_convt2_bias_h(
            &ws.g32,
            &neck.x2_up.b,
            &mut ws.h16a,
            pics,
            g,
            g,
            neck.x2_up.cout,
            false,
        )?;
        proj(
            &exec,
            &neck.proj1[1],
            &neck.proj2[1],
            &ws.h16a,
            &mut ws.y32,
            &mut ws.col16,
            &mut levels[1],
            pics,
            g2,
            f,
        )?;

        // ---- x1: straight off the trunk ----
        proj(
            &exec,
            &neck.proj1[2],
            &neck.proj2[2],
            &ws.n16,
            &mut ws.y32,
            &mut ws.col16,
            &mut levels[2],
            pics,
            g,
            f,
        )?;
        Ok(())
    }
}

/// A level's tail: 1x1 conv (+ bias) -> 3x3 conv (+ bias), `x16` the f16
/// raster in, `out` the f32 level.
#[allow(clippy::too_many_arguments)]
fn proj(
    exec: &crate::gpu::GpuExecutor,
    c1: &super::Conv,
    c3: &super::Conv,
    x16: &cudarc::driver::CudaSlice<half::f16>,
    y32: &mut cudarc::driver::CudaSlice<f32>,
    col16: &mut cudarc::driver::CudaSlice<half::f16>,
    out: &mut cudarc::driver::CudaSlice<f32>,
    pics: usize,
    side: usize,
    f: usize,
) -> Result<(), GpuModelError> {
    let px = side * side;
    if exec.f16_conv3_elected() {
        // the 1x1 lands f16(acc + b) - what the explicit form's bias pass and
        // f32 im2row round to - in col16's first px * f halves, and the 3x3
        // gathers its taps from there with its bias in the landing. Same bits.
        exec.matvec_batch_f16_h_bias(&c1.w, x16, col16, Some(&c1.b), pics * px)?;
        exec.f16_conv3_gemm(&c3.w, col16, out, Some(&c3.b), pics, side, side, f, px)?;
        return Ok(());
    }
    exec.matvec_batch_f16(&c1.w, x16, y32, pics * px)?;
    exec.bias_add(y32, &c1.b, pics * px, f)?;
    exec.dp_im2row3_f32(y32, col16, pics, side, side, f, px)?;
    exec.matvec_batch_f16(&c3.w, col16, out, pics * px)?;
    exec.bias_add(out, &c3.b, pics * px, f)?;
    Ok(())
}
