//! Inkling 2 Mini's picture tower op wrappers - pack slots 1011-1013: the
//! processor's patches, the embedding (position field + class row +
//! ln_pre) and the q / k / v split with the axial rope. Each works on one
//! picture whose rows start at a row offset of the pass's planes, so a pass
//! can pack several pictures back to back.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::f16;

use super::error::check;
use super::{GpuError, GpuExecutor};

/// Native patch rows a picture of `trows x cols` language-model tokens has.
fn patch_rows(trows: usize, cols: usize) -> usize {
    4 * trows * cols
}

fn under(len: usize, need: usize, what: &str) -> Result<(), GpuError> {
    if len < need {
        return Err(GpuError::Unsupported(format!(
            "{what}: a plane of {len} under the {need} the picture needs"
        )));
    }
    Ok(())
}

impl GpuExecutor {
    /// Whether the pack carries the picture tower's own passes (slots
    /// 1011-1013).
    pub fn has_inkling2_vision(&self) -> bool {
        let k = &self.kernels;
        k.inkling2_vis_patches.is_some()
            && k.inkling2_vis_embed.is_some()
            && k.inkling2_vis_qkv_rope.is_some()
    }

    /// Slot 1011: the `w x h` picture `rgb` (8-bit interleaved RGB) padded to
    /// `trows x cols` 40-pixel cells, normalized, and cut into its native
    /// patches (token-major, `[row][1200]` halves) at row `row0` of `out`.
    #[allow(clippy::too_many_arguments)]
    pub fn inkling2_vis_patches(
        &self,
        rgb: &CudaSlice<u8>,
        w: usize,
        h: usize,
        trows: usize,
        cols: usize,
        out: &mut CudaSlice<f16>,
        row0: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .inkling2_vis_patches
            .ok_or(GpuError::MissingOp("inkling2_vis_patches"))?;
        const F: usize = 1200;
        under(rgb.len(), w * h * 3, "inkling2_vis_patches rgb")?;
        under(
            out.len(),
            (row0 + patch_rows(trows, cols)) * F,
            "inkling2_vis_patches out",
        )?;
        let off = (row0 * F * 2) as u64;
        let (ip, _g1) = rgb.device_ptr(&self.stream);
        let (op, _g2) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 1011); planes checked above
        check(unsafe {
            f(
                ip as *const _,
                w as u32,
                h as u32,
                trows as u32,
                cols as u32,
                (op + off) as *mut _,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 1012, in place on the residual `x` from row `x0`: a picture's
    /// patch rows (holding the patch GEMM's rows) plus the `side x side`
    /// position field `grid` resampled onto its native grid, its class row
    /// (`cls`) after them, all through ln_pre (`lw`, `lb`).
    #[allow(clippy::too_many_arguments)]
    pub fn inkling2_vis_embed(
        &self,
        (x, x0): (&mut CudaSlice<f32>, usize),
        grid: &CudaSlice<f32>,
        cls: &CudaSlice<f32>,
        (lw, lb): (&CudaSlice<f32>, &CudaSlice<f32>),
        trows: usize,
        cols: usize,
        side: usize,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .inkling2_vis_embed
            .ok_or(GpuError::MissingOp("inkling2_vis_embed"))?;
        const D: usize = 1024;
        let n = patch_rows(trows, cols);
        under(x.len(), (x0 + n + 1) * D, "inkling2_vis_embed x")?;
        under(grid.len(), side * side * D, "inkling2_vis_embed grid")?;
        if cls.len() < D || lw.len() < D || lb.len() < D {
            return Err(GpuError::Unsupported(
                "inkling2_vis_embed: class row / ln_pre".into(),
            ));
        }
        let (gp, _g2) = grid.device_ptr(&self.stream);
        let (cp, _g3) = cls.device_ptr(&self.stream);
        let (wp, _g4) = lw.device_ptr(&self.stream);
        let (bp, _g5) = lb.device_ptr(&self.stream);
        let (xp, _g6) = x.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 1012); planes checked above
        check(unsafe {
            f(
                (xp + (x0 * D * 4) as u64) as *mut _,
                gp as *const _,
                cp as *const _,
                wp as *const _,
                bp as *const _,
                trows as u32,
                cols as u32,
                side as u32,
                eps,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 1013: a picture's fused q | k | v landing (`qkv` from row `row0`,
    /// `[row][3072]` halves) into the half q / k / v planes at the same rows,
    /// biases added, q and k on the axial rope, q times `qscale`.
    #[allow(clippy::too_many_arguments)]
    pub fn inkling2_vis_qkv_rope(
        &self,
        qkv: &CudaSlice<f16>,
        row0: usize,
        (bq, bk, bv): (&CudaSlice<f32>, &CudaSlice<f32>, &CudaSlice<f32>),
        (q, k, v): (
            &mut CudaSlice<f16>,
            &mut CudaSlice<f16>,
            &mut CudaSlice<f16>,
        ),
        trows: usize,
        cols: usize,
        hd: usize,
        theta: f32,
        qscale: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .inkling2_vis_qkv_rope
            .ok_or(GpuError::MissingOp("inkling2_vis_qkv_rope"))?;
        const D: usize = 1024;
        let end = row0 + patch_rows(trows, cols) + 1;
        under(qkv.len(), end * 3 * D, "inkling2_vis_qkv_rope qkv")?;
        for (len, what) in [(q.len(), "q"), (k.len(), "k"), (v.len(), "v")] {
            under(len, end * D, &format!("inkling2_vis_qkv_rope {what}"))?;
        }
        if bq.len() < D || bk.len() < D || bv.len() < D {
            return Err(GpuError::Unsupported(
                "inkling2_vis_qkv_rope: biases".into(),
            ));
        }
        let (ip, _g1) = qkv.device_ptr(&self.stream);
        let (bqp, _g2) = bq.device_ptr(&self.stream);
        let (bkp, _g3) = bk.device_ptr(&self.stream);
        let (bvp, _g4) = bv.device_ptr(&self.stream);
        let (qp, _g5) = q.device_ptr_mut(&self.stream);
        let (kp, _g6) = k.device_ptr_mut(&self.stream);
        let (vp, _g7) = v.device_ptr_mut(&self.stream);
        let off = (row0 * D * 2) as u64;
        // SAFETY: ABI contract (slot 1013); planes checked above
        check(unsafe {
            f(
                (ip + 3 * off) as *const _,
                bqp as *const _,
                bkp as *const _,
                bvp as *const _,
                (qp + off) as *mut _,
                (kp + off) as *mut _,
                (vp + off) as *mut _,
                trows as u32,
                cols as u32,
                hd as u32,
                theta,
                qscale,
                self.stream_ptr(),
            )
        })
    }
}
