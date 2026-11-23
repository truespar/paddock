//! Ripple-65B op wrappers - pack slots 968-969: the FFN's clamped SwiGLU and
//! the value-residual blend that feeds each layer's v projection. Everything
//! else the family runs is the laguna body's (BF16 / NVFP4 linears, the SWA
//! + NoPE attention, the sandwich norms).

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use super::error::check;
use super::{GpuError, GpuExecutor};

impl GpuExecutor {
    /// True when the pack carries Ripple's two glue kernels; the loader
    /// refuses the checkpoint on a pack without them.
    pub fn has_ripple_ops(&self) -> bool {
        self.kernels.ripple_swiglu_clamp.is_some() && self.kernels.ripple_valres_mix.is_some()
    }

    /// `gate[i] = silu(min(gate[i], limit)) * clamp(up[i], -limit, limit)`
    /// over `n` elements, in place on `gate`.
    pub fn ripple_swiglu_clamp(
        &self,
        gate: &mut CudaSlice<f32>,
        up: &CudaSlice<f32>,
        n: usize,
        limit: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .ripple_swiglu_clamp
            .ok_or(GpuError::MissingOp("ripple_swiglu_clamp"))?;
        if gate.len() < n || up.len() < n {
            return Err(GpuError::Unsupported(
                "ripple_swiglu_clamp: buffers under n".into(),
            ));
        }
        let (gp, _g1) = gate.device_ptr_mut(&self.stream);
        let (up_, _g2) = up.device_ptr(&self.stream);
        // SAFETY: ABI contract; lengths checked above
        check(unsafe {
            f(
                gp as *mut _,
                up_ as *const _,
                n as u32,
                limit,
                self.stream_ptr(),
            )
        })
    }

    /// `out[i] = e[i] + t * (x[i] - e[i])` over `n` elements - the reference's
    /// `lerp(e, x, t)`.
    pub fn ripple_valres_mix(
        &self,
        out: &mut CudaSlice<f32>,
        e: &CudaSlice<f32>,
        x: &CudaSlice<f32>,
        n: usize,
        t: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .ripple_valres_mix
            .ok_or(GpuError::MissingOp("ripple_valres_mix"))?;
        if out.len() < n || e.len() < n || x.len() < n {
            return Err(GpuError::Unsupported(
                "ripple_valres_mix: buffers under n".into(),
            ));
        }
        let (op, _g1) = out.device_ptr_mut(&self.stream);
        let (ep, _g2) = e.device_ptr(&self.stream);
        let (xp, _g3) = x.device_ptr(&self.stream);
        // SAFETY: ABI contract; lengths checked above
        check(unsafe {
            f(
                op as *mut _,
                ep as *const _,
                xp as *const _,
                n as u32,
                t,
                self.stream_ptr(),
            )
        })
    }
}
