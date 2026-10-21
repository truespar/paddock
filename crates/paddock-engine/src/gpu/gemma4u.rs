//! Gemma 4 12B's encoder-free media embedders - kernel side
//! `packs/cuda/src/gemma4u.cuh`, slots 862-864: the norms around the three
//! projections (which are slot 722's F32-class GEMM). Buffers are checked
//! against the geometry before any launch.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use super::error::*;
use super::*;

/// One model patch: 48 x 48 RGB.
pub const G4U_PATCH: usize = 48 * 48 * 3;
/// Raw samples a token (40 ms at 16 kHz).
pub const G4U_FRAME: usize = 640;

impl GpuExecutor {
    /// True when the pack carries the three embedder passes and the GEMM
    /// between them.
    pub fn has_g4u(&self) -> bool {
        let k = &self.kernels;
        k.g4u_patches.is_some()
            && k.g4u_post.is_some()
            && k.g4u_audio.is_some()
            && k.diar_gemm.is_some()
    }

    /// A resized `[h][w][3]` u8 picture's `n` patches (raster order, `gw` a
    /// row) as `[n][6912]` rows through `patch_ln1` (slot 862).
    #[allow(clippy::too_many_arguments)]
    pub fn g4u_patches(
        &self,
        rgb: &CudaSlice<u8>,
        w: usize,
        gw: usize,
        n: usize,
        ln_w: &CudaSlice<f32>,
        ln_b: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .g4u_patches
            .ok_or(GpuError::MissingOp("g4u_patches"))?;
        if n == 0 {
            return Ok(());
        }
        let rows = n.div_ceil(gw.max(1));
        if gw == 0
            || w < gw * 48
            || rgb.len() < rows * 48 * w * 3
            || ln_w.len() < G4U_PATCH
            || ln_b.len() < G4U_PATCH
            || out.len() < n * G4U_PATCH
        {
            return Err(oob("g4u_patches: buffers under the picture geometry"));
        }
        let (rp, _g1) = rgb.device_ptr(&self.stream);
        let (wp, _g2) = ln_w.device_ptr(&self.stream);
        let (bp, _g3) = ln_b.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 862); bounds checked above
        check(unsafe {
            f(
                rp as *const _,
                w as u32,
                gw as u32,
                n as u32,
                wp as *const _,
                bp as *const _,
                op as *mut _,
                eps,
                self.stream_ptr(),
            )
        })
    }

    /// In place over `n` `[dim]` rows of a `gw`-wide patch grid: `patch_ln2`,
    /// `+ pos[0][col] + pos[1][row]` (`pos`: `[2][pos_size][dim]`), `pos_norm`,
    /// the weightless RMS norm (slot 863).
    #[allow(clippy::too_many_arguments)]
    pub fn g4u_post(
        &self,
        h: &mut CudaSlice<f32>,
        ln2: (&CudaSlice<f32>, &CudaSlice<f32>),
        pos: &CudaSlice<f32>,
        pos_size: usize,
        gw: usize,
        ln3: (&CudaSlice<f32>, &CudaSlice<f32>),
        n: usize,
        dim: usize,
        (eps_ln, eps_rms): (f32, f32),
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .g4u_post
            .ok_or(GpuError::MissingOp("g4u_post"))?;
        if n == 0 {
            return Ok(());
        }
        if gw == 0
            || !n.is_multiple_of(gw)
            || gw > pos_size
            || n / gw > pos_size
            || h.len() < n * dim
            || pos.len() < 2 * pos_size * dim
            || [ln2.0, ln2.1, ln3.0, ln3.1].iter().any(|v| v.len() < dim)
        {
            return Err(oob("g4u_post: buffers under the picture geometry"));
        }
        let (hp, _g1) = h.device_ptr_mut(&self.stream);
        let (w2, _g2) = ln2.0.device_ptr(&self.stream);
        let (b2, _g3) = ln2.1.device_ptr(&self.stream);
        let (pp, _g4) = pos.device_ptr(&self.stream);
        let (w3, _g5) = ln3.0.device_ptr(&self.stream);
        let (b3, _g6) = ln3.1.device_ptr(&self.stream);
        // SAFETY: ABI contract (slot 863); bounds checked above
        check(unsafe {
            f(
                hp as *mut _,
                w2 as *const _,
                b2 as *const _,
                pp as *const _,
                pos_size as u32,
                gw as u32,
                w3 as *const _,
                b3 as *const _,
                n as u32,
                dim as u32,
                eps_ln,
                eps_rms,
                self.stream_ptr(),
            )
        })
    }

    /// `n_tok` 640-sample frames of `samples[..n_samples]` (the last one
    /// zero-padded) through the weightless RMS norm into `[n_tok][640]`
    /// (slot 864).
    pub fn g4u_audio(
        &self,
        samples: &CudaSlice<f32>,
        n_samples: usize,
        n_tok: usize,
        out: &mut CudaSlice<f32>,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .g4u_audio
            .ok_or(GpuError::MissingOp("g4u_audio"))?;
        if n_tok == 0 {
            return Ok(());
        }
        if samples.len() < n_samples
            || n_tok * G4U_FRAME < n_samples
            || out.len() < n_tok * G4U_FRAME
            || n_samples > u32::MAX as usize
        {
            return Err(oob("g4u_audio: buffers under the clip geometry"));
        }
        let (sp, _g1) = samples.device_ptr(&self.stream);
        let (op, _g2) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 864); bounds checked above
        check(unsafe {
            f(
                sp as *const _,
                n_samples as u32,
                n_tok as u32,
                op as *mut _,
                eps,
                self.stream_ptr(),
            )
        })
    }
}
