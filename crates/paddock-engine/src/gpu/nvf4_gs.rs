//! The static-scale nvfp4 activation quantize (pack slot 971): activations
//! quantized against a checkpoint's per-tensor input scale (ModelOpt NVFP4's
//! `input_scale`), the W4A4 recipe the checkpoint was calibrated for. Pairs
//! with `nvf4_gemm_f4_scaled` (fp4.rs), which carries the scale back.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use super::error::check;
use super::{GpuError, GpuExecutor};

impl GpuExecutor {
    /// Slot 971 is in the pack.
    pub fn has_quantize_nvf4_gs(&self) -> bool {
        self.kernels.quantize_nvf4_gs.is_some()
    }

    /// [`Self::quantize_nvf4`] against a static per-tensor scale `gs`
    /// (ModelOpt's 1 / input_scale) - pair with `nvf4_gemm_f4_scaled` at
    /// `act_scale = 1 / gs`.
    pub fn quantize_nvf4_gs(
        &self,
        x: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        scale: &mut CudaSlice<u8>,
        n: usize,
        gs: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .quantize_nvf4_gs
            .ok_or(GpuError::MissingOp("quantize_nvf4_gs"))?;
        if x.len() < n || q.len() < n / 2 || scale.len() < n / 16 {
            return Err(GpuError::Unsupported(
                "quantize_nvf4_gs: buffers under n".into(),
            ));
        }
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (qp, _g2) = q.device_ptr_mut(&self.stream);
        let (sp, _g3) = scale.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 971); sizes checked above, n % 32 by the kernel
        check(unsafe {
            f(
                xp as *const _,
                qp as *mut _,
                sp as *mut _,
                n as u32,
                gs,
                self.stream_ptr(),
            )
        })
    }
}
