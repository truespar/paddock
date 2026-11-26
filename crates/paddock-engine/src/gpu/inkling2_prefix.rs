//! Inkling 2 Mini's prefix-cache op wrapper - pack slot 1014: a short conv's
//! carried state at a cut inside a run (a checkpoint landing mid-chunk).

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use super::error::check;
use super::{GpuError, GpuExecutor};

impl GpuExecutor {
    /// Whether the pack carries slot 1014.
    pub fn has_inkling2_conv_capture(&self) -> bool {
        self.kernels.inkling2_conv_capture.is_some()
    }

    /// Slot 1014: for each of `n_items` items `{r0, kk, slot, off}` in
    /// `items`, the carried state of the conv over `x` had its run (from
    /// pass row r0) ended after kk rows, into `out[off + dst ..][3][c]`:
    /// x's rows at the run's kk - 3 .. kk, folded from `k` partials (1 or 8;
    /// 8 reads `[row][8][c]`), or `state`'s old rows (`[slots][3][c]`) where
    /// that is before the run. Launch before the conv rewrites `state`.
    #[allow(clippy::too_many_arguments)]
    pub fn inkling2_conv_capture(
        &self,
        (x, k, xstride): (&CudaSlice<f32>, usize, usize),
        c: usize,
        state: &CudaSlice<f32>,
        (items, n_items): (&CudaSlice<u32>, usize),
        out: &mut CudaSlice<f32>,
        dst: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .inkling2_conv_capture
            .ok_or(GpuError::MissingOp("inkling2_conv_capture"))?;
        if n_items == 0 {
            return Ok(());
        }
        if items.len() < 4 * n_items || dst + 3 * c > out.len() {
            return Err(GpuError::Unsupported(
                "inkling2_conv_capture: items / out".into(),
            ));
        }
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (sp, _g2) = state.device_ptr(&self.stream);
        let (ip, _g3) = items.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 1014); the items' rows and offsets are
        // the caller's pass and checkpoint layout, checked where it builds them
        check(unsafe {
            f(
                xp as *const _,
                k as u32,
                xstride as u32,
                c as u32,
                sp as *const _,
                ip as *const _,
                n_items as u32,
                op as *mut _,
                dst as u32,
                self.stream_ptr(),
            )
        })
    }
}
