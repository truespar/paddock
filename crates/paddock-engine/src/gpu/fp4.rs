//! NVF4/MXFP4 quantize + GEMM (smooth/rot calibration lanes).

use super::error::*;
use super::*;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

impl GpuExecutor {
    /// Re-quantize a repacked Q8_0 weight to packed-ADJACENT e2m1 planes -
    /// the mxf4 (m16n8k64) A format. Not interchangeable with the GGUF
    /// split-order planes [`q8_0_to_mxfp4`] emits.
    pub fn q8_0_to_fp4p(&self, w: &RepackedQ8) -> Result<RepackedMxfp4, GpuError> {
        let f = self
            .kernels
            .q8_0_to_fp4p
            .ok_or(GpuError::MissingOp("q8_0_to_fp4p"))?;
        let n_blocks = w.data.len() / 32;
        let mut data = self.alloc_u8(n_blocks * 16)?;
        let mut scale = self.alloc_u8(n_blocks)?;
        {
            let (qdp, _g1) = w.data.device_ptr(&self.stream);
            let (qsp, _g2) = w.scale.device_ptr(&self.stream);
            let (dp, _g3) = data.device_ptr_mut(&self.stream);
            let (sp, _g4) = scale.device_ptr_mut(&self.stream);
            // SAFETY: ABI contract; plane sizes derived from the Q8 source
            check(unsafe {
                f(
                    qdp as *const _,
                    qsp as *const _,
                    dp as *mut _,
                    sp as *mut _,
                    n_blocks as u64,
                    self.stream_ptr(),
                )
            })?;
        }
        Ok(RepackedMxfp4 { data, scale })
    }

    /// f32 -> packed-adjacent e2m1 + ue8m0 per-32 (the mxf4 B side; `q` uses
    /// n/2 bytes of the i8 plane).
    pub fn quantize_e2m1(
        &self,
        x: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        scale: &mut CudaSlice<u8>,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .quantize_e2m1
            .ok_or(GpuError::MissingOp("quantize_e2m1"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (qp, _g2) = q.device_ptr_mut(&self.stream);
        let (sp, _g3) = scale.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; n % 32 == 0, buffers sized [n/2]/[n/32]
        check(unsafe {
            f(
                xp as *const _,
                qp as *mut _,
                sp as *mut _,
                n as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Fused SwiGLU + packed e2m1 quantize (the mxf4 ffn_down input).
    pub fn quantize_e2m1_swiglu(
        &self,
        gate: &CudaSlice<f32>,
        up: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        scale: &mut CudaSlice<u8>,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .quantize_e2m1_swiglu
            .ok_or(GpuError::MissingOp("quantize_e2m1_swiglu"))?;
        let (gp, _g1) = gate.device_ptr(&self.stream);
        let (up_p, _g2) = up.device_ptr(&self.stream);
        let (qp, _g3) = q.device_ptr_mut(&self.stream);
        let (sp, _g4) = scale.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; n % 32 == 0
        check(unsafe {
            f(
                gp as *const _,
                up_p as *const _,
                qp as *mut _,
                sp as *mut _,
                n as u32,
                self.stream_ptr(),
            )
        })
    }

    /// sm_120a mxf4 dense GEMM (m16n8k64, fp4 x fp4): the full Blackwell fp4
    /// rate - 2x the mxf8f6f4 issue rate, half the activation bytes. Weights
    /// from [`q8_0_to_fp4p`], activations from [`quantize_e2m1`]. NUMERIC
    /// CLASS: fp4 weights AND activations (lossiest rung, retrieval-quality
    /// gated; [`mxfp4_gemm_bs`] is the e4m3 fallback).
    #[allow(clippy::too_many_arguments)]
    pub fn mxfp4_gemm_f4(
        &self,
        w: &RepackedMxfp4,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<u8>,
        y: &mut CudaSlice<f32>,
        in_dim: usize,
        out_dim: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .mxfp4_gemm_f4
            .ok_or(GpuError::MissingOp("mxfp4_gemm_f4"))?;
        debug_assert_eq!(w.data.len(), out_dim * in_dim / 2);
        let (wdp, _g1) = w.data.device_ptr(&self.stream);
        let (wsp, _g2) = w.scale.device_ptr(&self.stream);
        let (xqp, _g3) = xq.device_ptr(&self.stream);
        let (xsp, _g4) = xs.device_ptr(&self.stream);
        let (yp, _g5) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; in_dim % 32 == 0 checked by the launcher
        check(unsafe {
            f(
                wdp as *const _,
                wsp as *const _,
                xqp as *const _,
                xsp as *const _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Whether the nvf4 (fp4 x fp4, E4M3 scales per 16) dense route is fully
    /// available. Same cc-12 caller obligation as the other bs probes.
    pub fn has_mxfp4_gemm_nv4(&self) -> bool {
        self.kernels.mxfp4_gemm_nv4.is_some()
            && self.kernels.q8_0_to_nvf4.is_some()
            && self.kernels.quantize_nvf4.is_some()
            && self.kernels.quantize_nvf4_swiglu.is_some()
    }

    /// Re-quantize a repacked Q8_0 weight to nvf4 planes (packed-adjacent
    /// e2m1 + E4M3 scales per 16 - the scale plane is numel/16 bytes, twice
    /// the mxfp4 plane). The most precise of the fp4 weight classes.
    pub fn q8_0_to_nvf4(&self, w: &RepackedQ8) -> Result<RepackedMxfp4, GpuError> {
        let f = self
            .kernels
            .q8_0_to_nvf4
            .ok_or(GpuError::MissingOp("q8_0_to_nvf4"))?;
        let n_blocks = w.data.len() / 32;
        let mut data = self.alloc_u8(n_blocks * 16)?;
        let mut scale = self.alloc_u8(n_blocks * 2)?;
        {
            let (qdp, _g1) = w.data.device_ptr(&self.stream);
            let (qsp, _g2) = w.scale.device_ptr(&self.stream);
            let (dp, _g3) = data.device_ptr_mut(&self.stream);
            let (sp, _g4) = scale.device_ptr_mut(&self.stream);
            // SAFETY: ABI contract; plane sizes derived from the Q8 source
            check(unsafe {
                f(
                    qdp as *const _,
                    qsp as *const _,
                    dp as *mut _,
                    sp as *mut _,
                    n_blocks as u64,
                    self.stream_ptr(),
                )
            })?;
        }
        Ok(RepackedMxfp4 { data, scale })
    }

    /// f32 -> nvf4 (packed e2m1, e4m3 per-16; `q` uses n/2 bytes, `scale`
    /// n/16 bytes).
    pub fn quantize_nvf4(
        &self,
        x: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        scale: &mut CudaSlice<u8>,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .quantize_nvf4
            .ok_or(GpuError::MissingOp("quantize_nvf4"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (qp, _g2) = q.device_ptr_mut(&self.stream);
        let (sp, _g3) = scale.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; n % 32 == 0
        check(unsafe {
            f(
                xp as *const _,
                qp as *mut _,
                sp as *mut _,
                n as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Fused SwiGLU + nvf4 quantize (the nvf4 ffn_down input).
    pub fn quantize_nvf4_swiglu(
        &self,
        gate: &CudaSlice<f32>,
        up: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        scale: &mut CudaSlice<u8>,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .quantize_nvf4_swiglu
            .ok_or(GpuError::MissingOp("quantize_nvf4_swiglu"))?;
        let (gp, _g1) = gate.device_ptr(&self.stream);
        let (up_p, _g2) = up.device_ptr(&self.stream);
        let (qp, _g3) = q.device_ptr_mut(&self.stream);
        let (sp, _g4) = scale.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; n % 32 == 0
        check(unsafe {
            f(
                gp as *const _,
                up_p as *const _,
                qp as *mut _,
                sp as *mut _,
                n as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Swiglu over a MERGED `[rows, 2*ff]` gate|up plane straight into the
    /// nvf4 down-input staging (`q`/`scale`), one launch. Bit-identical to
    /// `swiglu_fused` (f32) then `quantize_nvf4` - it just drops the f32
    /// round trip of the widest activation in the tick. `ff % 32 == 0`.
    pub fn swiglu_fused_nvf4(
        &self,
        fused: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        scale: &mut CudaSlice<u8>,
        ff: usize,
        n_rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .swiglu_fused_nvf4
            .ok_or(GpuError::MissingOp("swiglu_fused_nvf4"))?;
        let (fp, _g1) = fused.device_ptr(&self.stream);
        let (qp, _g2) = q.device_ptr_mut(&self.stream);
        let (sp, _g3) = scale.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; ff % 32 == 0
        check(unsafe {
            f(
                fp as *const _,
                qp as *mut _,
                sp as *mut _,
                ff as u32,
                n_rows as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Interleaved-plane twin of [`Self::swiglu_fused_nvf4`] (slot 535).
    pub fn swiglu_fused_nvf4_il(
        &self,
        fused: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        scale: &mut CudaSlice<u8>,
        ff: usize,
        n_rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .swiglu_fused_nvf4_il
            .ok_or(GpuError::MissingOp("swiglu_fused_nvf4_il"))?;
        let (fp, _g1) = fused.device_ptr(&self.stream);
        let (qp, _g2) = q.device_ptr_mut(&self.stream);
        let (sp, _g3) = scale.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; ff % 32 == 0
        check(unsafe {
            f(
                fp as *const _,
                qp as *mut _,
                sp as *mut _,
                ff as u32,
                n_rows as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_swiglu_fused_nvf4_il(&self) -> bool {
        self.kernels.swiglu_fused_nvf4_il.is_some()
    }

    /// f4t with the swiglu + nvf4-quant epilogue (slot 533) over an
    /// INTERLEAVED gate|up plane (`gu_pairs`): stages the down GEMM's
    /// activation pair (`q`, `qs`) directly from the accumulators -- no f32
    /// [batch, 2ff] landing, no separate swiglu/quantize launch. Bit-identical
    /// to f4t + swiglu_fused_nvf4 on the plain plane (bench/nv4_swq_cmp.cu).
    /// Geometry: batch >= 128, in_dim % 256 == 0, out_dim % 256 == 0.
    pub fn nvf4_gemm_f4t_swq(
        &self,
        w: &Nvf4Plane,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<u8>,
        q: &mut CudaSlice<i8>,
        qs: &mut CudaSlice<u8>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .nvf4_gemm_f4t_swq
            .ok_or(GpuError::MissingOp("nvf4_gemm_f4t_swq"))?;
        let ff = w.out_dim / 2;
        if !w.gu_pairs
            || w.layout != Nvf4Layout::Row
            || !w.in_dim.is_multiple_of(256)
            || !w.out_dim.is_multiple_of(256)
            || batch < 128
        {
            return Err(GpuError::Unsupported(format!(
                "nvf4_gemm_f4t_swq: needs an interleaved row-major gate|up plane, in%256, out%256, batch>=128 (got pairs {} in {} out {} batch {batch})",
                w.gu_pairs, w.in_dim, w.out_dim
            )));
        }
        if xq.len() < batch * w.in_dim / 2
            || xs.len() < batch * w.in_dim / 16
            || q.len() < batch * ff / 2
            || qs.len() < batch * ff / 16
        {
            return Err(GpuError::Unsupported(format!(
                "nvf4_gemm_f4t_swq: xq {} / xs {} / q {} / qs {} too small for {batch} x [{}, {}]",
                xq.len(),
                xs.len(),
                q.len(),
                qs.len(),
                w.out_dim,
                w.in_dim
            )));
        }
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (sp, _g2) = w.scale.device_ptr(&self.stream);
        let (xqp, _g3) = xq.device_ptr(&self.stream);
        let (xsp, _g4) = xs.device_ptr(&self.stream);
        let (qp, _g5) = q.device_ptr_mut(&self.stream);
        let (qsp, _g6) = qs.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; geometry validated above
        check(unsafe {
            f(
                dp as *const _,
                sp as *const _,
                xqp as *const _,
                xsp as *const _,
                qp as *mut _,
                qsp as *mut _,
                w.scale2,
                w.in_dim as u32,
                w.out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_nvf4_gemm_f4t_swq(&self) -> bool {
        self.kernels.nvf4_gemm_f4t_swq.is_some()
    }

    /// Whether the merged-swiglu -> nvf4 down-staging kernel is available.
    pub fn has_swiglu_fused_nvf4(&self) -> bool {
        self.kernels.swiglu_fused_nvf4.is_some()
    }

    /// Whether the fused gate+up+swiglu->nvf4 kernel is available.
    pub fn has_mxfp4_gemm_bs_gu(&self) -> bool {
        self.kernels.mxfp4_gemm_bs_gu.is_some()
    }

    /// Fused dense FFN front half: gate+up mxf8f6f4 GEMMs over one e4m3
    /// activation staging + in-register silu(g)*u + nvf4 quantize straight
    /// into the down GEMM's `fq`/`fs` planes. Bit-identical to the unfused
    /// [`mxfp4_gemm_bs`] x2 + [`quantize_nvf4_swiglu`] chain.
    #[allow(clippy::too_many_arguments)]
    pub fn mxfp4_gemm_bs_gu(
        &self,
        gate_w: &RepackedMxfp4,
        up_w: &RepackedMxfp4,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<u8>,
        fq: &mut CudaSlice<i8>,
        fs: &mut CudaSlice<u8>,
        in_dim: usize,
        ff: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .mxfp4_gemm_bs_gu
            .ok_or(GpuError::MissingOp("mxfp4_gemm_bs_gu"))?;
        debug_assert_eq!(gate_w.data.len(), ff * in_dim / 2);
        debug_assert_eq!(up_w.data.len(), ff * in_dim / 2);
        let (gdp, _g1) = gate_w.data.device_ptr(&self.stream);
        let (gsp, _g2) = gate_w.scale.device_ptr(&self.stream);
        let (udp, _g3) = up_w.data.device_ptr(&self.stream);
        let (usp, _g4) = up_w.scale.device_ptr(&self.stream);
        let (xqp, _g5) = xq.device_ptr(&self.stream);
        let (xsp, _g6) = xs.device_ptr(&self.stream);
        let (fqp, _g7) = fq.device_ptr_mut(&self.stream);
        let (fsp, _g8) = fs.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; dims checked by the launcher
        check(unsafe {
            f(
                gdp as *const _,
                gsp as *const _,
                udp as *const _,
                usp as *const _,
                xqp as *const _,
                xsp as *const _,
                fqp as *mut _,
                fsp as *mut _,
                in_dim as u32,
                ff as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Whether the SmoothQuant-folded nvf4 route (per-channel scale
    /// migration) is available on top of the plain nvf4 trio.
    pub fn has_nvf4_smooth(&self) -> bool {
        self.has_mxfp4_gemm_nv4()
            && self.kernels.col_absmax.is_some()
            && self.kernels.q8_0_col_absmax.is_some()
            && self.kernels.quantize_nvf4_smooth.is_some()
            && self.kernels.q8_0_to_nvf4_smooth.is_some()
            && self.kernels.quantize_nvf4_swiglu_smooth.is_some()
    }

    /// Fused SwiGLU + SmoothQuant fold + nvf4 quantize (the smoothed down
    /// site's input).
    #[allow(clippy::too_many_arguments)]
    pub fn quantize_nvf4_swiglu_smooth(
        &self,
        gate: &CudaSlice<f32>,
        up: &CudaSlice<f32>,
        sinv: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        scale: &mut CudaSlice<u8>,
        n: usize,
        in_dim: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .quantize_nvf4_swiglu_smooth
            .ok_or(GpuError::MissingOp("quantize_nvf4_swiglu_smooth"))?;
        let (gp, _g1) = gate.device_ptr(&self.stream);
        let (up_p, _g2) = up.device_ptr(&self.stream);
        let (svp, _g3) = sinv.device_ptr(&self.stream);
        let (qp, _g4) = q.device_ptr_mut(&self.stream);
        let (sp, _g5) = scale.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; n % 32 == 0, in_dim % 8 == 0
        check(unsafe {
            f(
                gp as *const _,
                up_p as *const _,
                svp as *const _,
                qp as *mut _,
                sp as *mut _,
                n as u32,
                in_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Whether the smoothed-F8 class (SmoothQuant-folded mxf8f6f4) is fully
    /// available.
    pub fn has_f8_smooth(&self) -> bool {
        self.kernels.mxfp4_gemm_bs.is_some()
            && self.kernels.quantize_e4m3_smooth.is_some()
            && self.kernels.quantize_e4m3_swiglu_smooth.is_some()
            && self.kernels.q8_0_to_mxfp4_smooth.is_some()
    }

    /// e4m3 activation quantize with the SmoothQuant fold.
    #[allow(clippy::too_many_arguments)]
    pub fn quantize_e4m3_smooth(
        &self,
        x: &CudaSlice<f32>,
        sinv: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        scale: &mut CudaSlice<u8>,
        n: usize,
        in_dim: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .quantize_e4m3_smooth
            .ok_or(GpuError::MissingOp("quantize_e4m3_smooth"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (svp, _g2) = sinv.device_ptr(&self.stream);
        let (qp, _g3) = q.device_ptr_mut(&self.stream);
        let (sp, _g4) = scale.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; n % 32 == 0, in_dim % 4 == 0
        check(unsafe {
            f(
                xp as *const _,
                svp as *const _,
                qp as *mut _,
                sp as *mut _,
                n as u32,
                in_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Fused SwiGLU + SmoothQuant fold + e4m3 quantize.
    #[allow(clippy::too_many_arguments)]
    pub fn quantize_e4m3_swiglu_smooth(
        &self,
        gate: &CudaSlice<f32>,
        up: &CudaSlice<f32>,
        sinv: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        scale: &mut CudaSlice<u8>,
        n: usize,
        in_dim: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .quantize_e4m3_swiglu_smooth
            .ok_or(GpuError::MissingOp("quantize_e4m3_swiglu_smooth"))?;
        let (gp, _g1) = gate.device_ptr(&self.stream);
        let (up_p, _g2) = up.device_ptr(&self.stream);
        let (svp, _g3) = sinv.device_ptr(&self.stream);
        let (qp, _g4) = q.device_ptr_mut(&self.stream);
        let (sp, _g5) = scale.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; n % 32 == 0, in_dim % 4 == 0
        check(unsafe {
            f(
                gp as *const _,
                up_p as *const _,
                svp as *const _,
                qp as *mut _,
                sp as *mut _,
                n as u32,
                in_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Re-quantize a repacked Q8_0 weight to split-order mxfp4 with the
    /// SmoothQuant weight fold.
    pub fn q8_0_to_mxfp4_smooth(
        &self,
        w: &RepackedQ8,
        svec: &CudaSlice<f32>,
    ) -> Result<RepackedMxfp4, GpuError> {
        let f = self
            .kernels
            .q8_0_to_mxfp4_smooth
            .ok_or(GpuError::MissingOp("q8_0_to_mxfp4_smooth"))?;
        let n_blocks = w.data.len() / 32;
        let mut data = self.alloc_u8(n_blocks * 16)?;
        let mut scale = self.alloc_u8(n_blocks)?;
        {
            let (qdp, _g1) = w.data.device_ptr(&self.stream);
            let (qsp, _g2) = w.scale.device_ptr(&self.stream);
            let (svp, _g3) = svec.device_ptr(&self.stream);
            let (dp, _g4) = data.device_ptr_mut(&self.stream);
            let (sp, _g5) = scale.device_ptr_mut(&self.stream);
            // SAFETY: ABI contract; plane sizes derived from the Q8 source
            check(unsafe {
                f(
                    qdp as *const _,
                    qsp as *const _,
                    svp as *const _,
                    dp as *mut _,
                    sp as *mut _,
                    n_blocks as u64,
                    w.dims[0] as u32,
                    self.stream_ptr(),
                )
            })?;
        }
        Ok(RepackedMxfp4 { data, scale })
    }

    /// Device-to-device f32 copy of the first `n` elements.
    pub fn copy_f32(
        &self,
        src: &CudaSlice<f32>,
        dst: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), GpuError> {
        let sv = src
            .try_slice(0..n)
            .ok_or_else(|| oob("copy_f32: src range"))?;
        let mut dv = dst
            .try_slice_mut(0..n)
            .ok_or_else(|| oob("copy_f32: dst range"))?;
        self.stream.memcpy_dtod(&sv, &mut dv).map_err(drv)
    }

    /// Accumulate per-column abs-max of `x` ([rows, n] f32) into `out`
    /// (caller zeroes once) - SmoothQuant activation statistics.
    pub fn col_absmax(
        &self,
        x: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        rows: usize,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .col_absmax
            .ok_or(GpuError::MissingOp("col_absmax"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (op, _g2) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; n % 32 == 0
        check(unsafe {
            f(
                xp as *const _,
                op as *mut _,
                rows as u32,
                n as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Per-column abs-max of a repacked Q8_0 weight into `out` (caller
    /// zeroes once) - the weight half of the SmoothQuant balance.
    pub fn q8_0_col_absmax(
        &self,
        w: &RepackedQ8,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_col_absmax
            .ok_or(GpuError::MissingOp("q8_0_col_absmax"))?;
        let (in_dim, out_dim) = (w.dims[0], w.data.len() / w.dims[0]);
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (sp, _g2) = w.scale.device_ptr(&self.stream);
        let (op, _g3) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; in_dim % 32 == 0
        check(unsafe {
            f(
                dp as *const _,
                sp as *const _,
                op as *mut _,
                in_dim as u32,
                out_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    /// nvf4 activation quantize with the SmoothQuant fold (`sinv` = 1/s per
    /// input channel).
    #[allow(clippy::too_many_arguments)]
    pub fn quantize_nvf4_smooth(
        &self,
        x: &CudaSlice<f32>,
        sinv: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        scale: &mut CudaSlice<u8>,
        n: usize,
        in_dim: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .quantize_nvf4_smooth
            .ok_or(GpuError::MissingOp("quantize_nvf4_smooth"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (svp, _g2) = sinv.device_ptr(&self.stream);
        let (qp, _g3) = q.device_ptr_mut(&self.stream);
        let (sp, _g4) = scale.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; n % 32 == 0, in_dim % 8 == 0
        check(unsafe {
            f(
                xp as *const _,
                svp as *const _,
                qp as *mut _,
                sp as *mut _,
                n as u32,
                in_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Re-quantize a repacked Q8_0 weight to nvf4 with the SmoothQuant fold
    /// (`svec` = s per input channel; pair with `quantize_nvf4_smooth` using
    /// the inverse).
    pub fn q8_0_to_nvf4_smooth(
        &self,
        w: &RepackedQ8,
        svec: &CudaSlice<f32>,
    ) -> Result<RepackedMxfp4, GpuError> {
        let f = self
            .kernels
            .q8_0_to_nvf4_smooth
            .ok_or(GpuError::MissingOp("q8_0_to_nvf4_smooth"))?;
        let n_blocks = w.data.len() / 32;
        let mut data = self.alloc_u8(n_blocks * 16)?;
        let mut scale = self.alloc_u8(n_blocks * 2)?;
        {
            let (qdp, _g1) = w.data.device_ptr(&self.stream);
            let (qsp, _g2) = w.scale.device_ptr(&self.stream);
            let (svp, _g3) = svec.device_ptr(&self.stream);
            let (dp, _g4) = data.device_ptr_mut(&self.stream);
            let (sp, _g5) = scale.device_ptr_mut(&self.stream);
            // SAFETY: ABI contract; plane sizes derived from the Q8 source
            check(unsafe {
                f(
                    qdp as *const _,
                    qsp as *const _,
                    svp as *const _,
                    dp as *mut _,
                    sp as *mut _,
                    n_blocks as u64,
                    w.dims[0] as u32,
                    self.stream_ptr(),
                )
            })?;
        }
        Ok(RepackedMxfp4 { data, scale })
    }

    /// Whether the ROTATED nvf4 route (QuaRot H128 on both sides) is
    /// available on top of the plain nvf4 trio.
    pub fn has_nvf4_rot(&self) -> bool {
        self.has_mxfp4_gemm_nv4()
            && self.kernels.q8_0_to_nvf4_rot.is_some()
            && self.kernels.quantize_nvf4_rot.is_some()
    }

    /// Re-quantize a repacked Q8_0 weight to H128-ROTATED nvf4 planes: pair
    /// only with [`quantize_nvf4_rot`] activations (the rotations cancel in
    /// the GEMM). The QuaRot treatment for outlier-channel inputs.
    pub fn q8_0_to_nvf4_rot(&self, w: &RepackedQ8) -> Result<RepackedMxfp4, GpuError> {
        let f = self
            .kernels
            .q8_0_to_nvf4_rot
            .ok_or(GpuError::MissingOp("q8_0_to_nvf4_rot"))?;
        let n_blocks = w.data.len() / 32;
        let mut data = self.alloc_u8(n_blocks * 16)?;
        let mut scale = self.alloc_u8(n_blocks * 2)?;
        {
            let (qdp, _g1) = w.data.device_ptr(&self.stream);
            let (qsp, _g2) = w.scale.device_ptr(&self.stream);
            let (dp, _g3) = data.device_ptr_mut(&self.stream);
            let (sp, _g4) = scale.device_ptr_mut(&self.stream);
            // SAFETY: ABI contract; in_dim % 128 == 0 checked by the launcher
            check(unsafe {
                f(
                    qdp as *const _,
                    qsp as *const _,
                    dp as *mut _,
                    sp as *mut _,
                    n_blocks as u64,
                    self.stream_ptr(),
                )
            })?;
        }
        Ok(RepackedMxfp4 { data, scale })
    }

    /// f32 -> H128-rotated nvf4 activations (fused QuaRot rotation +
    /// quantize; n % 128 == 0).
    pub fn quantize_nvf4_rot(
        &self,
        x: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        scale: &mut CudaSlice<u8>,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .quantize_nvf4_rot
            .ok_or(GpuError::MissingOp("quantize_nvf4_rot"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (qp, _g2) = q.device_ptr_mut(&self.stream);
        let (sp, _g3) = scale.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; n % 128 == 0 checked by the launcher
        check(unsafe {
            f(
                xp as *const _,
                qp as *mut _,
                sp as *mut _,
                n as u32,
                self.stream_ptr(),
            )
        })
    }

    /// sm_120a nvf4 dense GEMM: mxf4's m16n8k64 issue rate with E4M3-per-16
    /// scaling (the outlier-tolerant fp4 x fp4 class). Weights from
    /// [`q8_0_to_nvf4`], activations from [`quantize_nvf4`].
    #[allow(clippy::too_many_arguments)]
    pub fn mxfp4_gemm_nv4(
        &self,
        w: &RepackedMxfp4,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<u8>,
        y: &mut CudaSlice<f32>,
        in_dim: usize,
        out_dim: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .mxfp4_gemm_nv4
            .ok_or(GpuError::MissingOp("mxfp4_gemm_nv4"))?;
        debug_assert_eq!(w.data.len(), out_dim * in_dim / 2);
        let (wdp, _g1) = w.data.device_ptr(&self.stream);
        let (wsp, _g2) = w.scale.device_ptr(&self.stream);
        let (xqp, _g3) = xq.device_ptr(&self.stream);
        let (xsp, _g4) = xs.device_ptr(&self.stream);
        let (yp, _g5) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; in_dim % 32 == 0 checked by the launcher
        check(unsafe {
            f(
                wdp as *const _,
                wsp as *const _,
                xqp as *const _,
                xsp as *const _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Fused SwiGLU + e4m3/ue8m0 quantize: q = e4m3(silu(gate)*up) - the
    /// dense block-scale ffn_down input, computed in registers (the f32
    /// swiglu product never lands in memory).
    /// Fused GEGLU (gelu_tanh) + e4m3 quantize - the gemma4 twin of
    /// [`quantize_e4m3_swiglu`]; bit-identical to geglu -> quantize_e4m3
    /// (same formula, same scale pick), the f32 activation never lands.
    pub fn quantize_e4m3_geglu(
        &self,
        gate: &CudaSlice<f32>,
        up: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        scale: &mut CudaSlice<u8>,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .quantize_e4m3_geglu
            .ok_or(GpuError::MissingOp("quantize_e4m3_geglu"))?;
        let (gp, _g1) = gate.device_ptr(&self.stream);
        let (up_p, _g2) = up.device_ptr(&self.stream);
        let (qp, _g3) = q.device_ptr_mut(&self.stream);
        let (sp, _g4) = scale.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; n % 32 == 0, buffers sized [n]/[n/32]
        check(unsafe {
            f(
                gp as *const _,
                up_p as *const _,
                qp as *mut _,
                sp as *mut _,
                n as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Whether the pack ships the fused GEGLU e4m3 quantize.
    /// Fused-plane GEGLU quantize: gate|up in one [rows][2*n_ff] buffer
    /// (per-row [gate|up]) -> e4m3 q/scale [rows][n_ff]. Values identical to
    /// quantize_e4m3_geglu on split buffers.
    pub fn quantize_e4m3_glu2(
        &self,
        gu: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        scale: &mut CudaSlice<u8>,
        n_ff: usize,
        rows: usize,
        act: GluAct,
    ) -> Result<(), GpuError> {
        let (f, name) = match act {
            GluAct::Gelu => (self.kernels.quantize_e4m3_geglu2, "quantize_e4m3_geglu2"),
            GluAct::Silu => (self.kernels.quantize_e4m3_swiglu2, "quantize_e4m3_swiglu2"),
        };
        let f = f.ok_or(GpuError::MissingOp(name))?;
        let (gp, _g1) = gu.device_ptr(&self.stream);
        let (qp, _g2) = q.device_ptr_mut(&self.stream);
        let (sp, _g3) = scale.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; gu holds rows*2*n_ff f32, q/scale sized
        // [rows*n_ff]/[rows*n_ff/32]
        check(unsafe {
            f(
                gp as *const _,
                qp as *mut _,
                sp as *mut _,
                n_ff as u32,
                rows as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_quantize_e4m3_glu2i(&self, act: GluAct) -> bool {
        match act {
            GluAct::Gelu => self.kernels.quantize_e4m3_geglu2i.is_some(),
            GluAct::Silu => self.kernels.quantize_e4m3_swiglu2i.is_some(),
        }
    }

    /// Interleaved-plane twin of [`Self::quantize_e4m3_glu2`]: gu came
    /// from a GEMM over a `f8w_repack_lin_gui` plane, so pair p sits at row
    /// offsets (p>>3)*16+(p&7) / +8 inside each [2*n_ff] row. Byte-identical
    /// outputs to glu2 on the unpermuted plane.
    pub fn quantize_e4m3_glu2i(
        &self,
        gu: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        scale: &mut CudaSlice<u8>,
        n_ff: usize,
        rows: usize,
        act: GluAct,
    ) -> Result<(), GpuError> {
        let (f, name) = match act {
            GluAct::Gelu => (self.kernels.quantize_e4m3_geglu2i, "quantize_e4m3_geglu2i"),
            GluAct::Silu => (
                self.kernels.quantize_e4m3_swiglu2i,
                "quantize_e4m3_swiglu2i",
            ),
        };
        let f = f.ok_or(GpuError::MissingOp(name))?;
        let (gp, _g1) = gu.device_ptr(&self.stream);
        let (qp, _g2) = q.device_ptr_mut(&self.stream);
        let (sp, _g3) = scale.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; same sizing as quantize_e4m3_geglu2
        check(unsafe {
            f(
                gp as *const _,
                qp as *mut _,
                sp as *mut _,
                n_ff as u32,
                rows as u32,
                self.stream_ptr(),
            )
        })
    }

    // ---- modelopt NVFP4 checkpoint planes  ----------------------

    /// Whether the checkpoint-NVFP4 consumers (dequant oracle + W4A16 GEMV)
    /// can actually run HERE.
    ///
    /// The symbol test alone is not an answer, and used to be the whole test.
    /// `pd_nvf4_gemv` is exported unconditionally; its real body is compiled
    /// under `PD_BS_OK` = `__CUDA_ARCH__ >= 1200 && __CUDA_ARCH_FEAT_SM120_ALL`,
    /// and the `#else` arm is an empty kernel. So on Ampere/Ada/Hopper - and on
    /// sm_100, which is Blackwell but not the sm_120a feature target - the
    /// launch SUCCEEDS and writes nothing: the output buffer keeps whatever it
    /// held. Not an error, not a fallback, just wrong numbers. Meanwhile
    /// `is_some()` is a BUILD fact ("was compute_120a in the gencode list?"),
    /// true on every GPU our shipped pack serves, so the guard passed on the
    /// A6000 this was traced on, which cannot run NVFP4 at all.
    ///
    /// Ask the device instead. Off sm_120a the caller falls back to the
    /// Q8-derived planes, which is correct - and callers must SAY so, because
    /// a fallback nobody mentions is how you ship a 22 GB download that
    /// changes nothing.
    pub fn has_nvf4_ckpt(&self) -> bool {
        self.kernels.nvf4_dequant.is_some()
            && self.kernels.nvf4_gemv.is_some()
            && self.nvf4_ckpt_arch()
    }

    /// The arch half of [`has_nvf4_ckpt`], separately callable so a caller can
    /// tell "this GPU cannot" from "this pack was not built for it".
    ///
    /// Floor is 8.9, and it must stay equal to the pack's `PD_NV4_OK`
    /// (moe/block_scale_quant.cuh) and to the matching NULL block in
    /// `paddock_pack_kernels_v1` - three gates, one condition.
    ///
    /// It used to read `>= 12` because these kernels were written
    /// beside the sm_120a block-scale family and inherited its gate. They do
    /// not deserve it: the NVFP4 checkpoint consumers decode e2m1 in SIMT and
    /// accumulate on FFMA (the tc tile adds plain bf16 mma, itself 8.0+), and
    /// not one issues a block-scale instruction. On B200 that mistake cost the
    /// entire nemotron NVFP4 lane - the checkpoint loaded, then every request
    /// died on "kernel nvf4_moe_up_relu2 missing from the loaded pack".
    pub fn nvf4_ckpt_arch(&self) -> bool {
        let (major, minor) = self.compute_capability();
        major > 8 || (major == 8 && minor >= 9)
    }

    /// Upload a modelopt NVFP4 triple byte-for-byte (no repacking: the
    /// shipped adjacent-nibble + flat e4m3-per-16 layout is exactly what the
    /// nv4 kernel family consumes). `packed` is [out, in/2], `scales`
    /// [out, in/16]; both validated against the dims here so a geometry
    /// drift fails at load, never as a mis-strided kernel walk.
    pub fn nvf4_upload(
        &self,
        packed: &[u8],
        scales: &[u8],
        scale2: f32,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Nvf4Plane, GpuError> {
        if !in_dim.is_multiple_of(32)
            || packed.len() != out_dim * (in_dim / 2)
            || scales.len() != out_dim * (in_dim / 16)
        {
            return Err(GpuError::Unsupported(format!(
                "nvf4 triple geometry mismatch: packed {} scales {} bytes for [{out_dim}, {in_dim}]",
                packed.len(),
                scales.len()
            )));
        }
        // Stream-ordered upload (clone_htod: alloc + copy on the engine
        // stream). Until 2026-09-08 this was alloc_zeros + a raw cuMemcpyHtoD:
        // the engine stream is non-blocking, so the legacy-stream copy was
        // unordered against the memset still pending on it, and whenever the
        // memset landed second the plane came up partly ZEROED - a different
        // slice per process, fixed for the process's life. Found on GB10 as
        // greedy text that differed across restarts of the NVFP4 lane while
        // three requests inside one process were bit-identical (Spark
        // session 6).
        let data = self.stream.clone_htod(packed).map_err(drv)?;
        let scale = self.stream.clone_htod(scales).map_err(drv)?;
        Ok(Nvf4Plane {
            data,
            scale,
            scale2,
            out_dim,
            in_dim,
            layout: Nvf4Layout::Row,
            gu_pairs: false,
        })
    }

    /// True when the tile-major plane twins (slots 452-454) are all
    /// loadable - the gate for [`Self::nvf4_upload_tiled`]: a tiled plane
    /// must never exist without every consumer class able to read it.
    pub fn has_nvf4_tm(&self) -> bool {
        self.kernels.nvf4_gemv_batch_tm.is_some()
            && self.kernels.nvf4_gemm_mr_tm.is_some()
            && self.kernels.nvf4_gemm_tc_tm.is_some()
    }

    /// True when the fragment-layout plane twins (slots 455-457) are all
    /// loadable - the gate for [`Self::nvf4_upload_frag`], same rule as
    /// [`Self::has_nvf4_tm`].
    pub fn has_nvf4_tf(&self) -> bool {
        self.kernels.nvf4_gemv_batch_tf.is_some()
            && self.kernels.nvf4_gemm_mr_tf.is_some()
            && self.kernels.nvf4_gemm_tc_tf.is_some()
    }

    /// Upload a modelopt NVFP4 triple repacked to the TILE-MAJOR layout the
    /// `_tm` kernel twins read: `[row_tile 128][k_stage 128][row]`, weights
    /// (64 B/row/stage) and e4m3 scale records (8 B/row/stage) each
    /// contiguous per (tile, stage) block, rows padded to 128 and
    /// ZERO-filled (a pad row decodes as 0 x scale-byte-0 = 0.0, and every
    /// consumer guards its stores at `out_dim`). Same bytes per logical
    /// element as [`Self::nvf4_upload`] - the twins are bit-exact per class
    /// - but the tensor-core head's per-stage cp.async becomes one
    ///   sequential 10.25 KB block instead of 128 rows at 1344 B stride
    ///   (probe: 225 -> 205 us b32 at the lm_head shape). Requires
    ///   `in_dim % 128 == 0`; callers gate on [`Self::has_nvf4_tm`].
    pub fn nvf4_upload_tiled(
        &self,
        packed: &[u8],
        scales: &[u8],
        scale2: f32,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Nvf4Plane, GpuError> {
        if !in_dim.is_multiple_of(128)
            || packed.len() != out_dim * (in_dim / 2)
            || scales.len() != out_dim * (in_dim / 16)
        {
            return Err(GpuError::Unsupported(format!(
                "nvf4 tiled triple geometry mismatch: packed {} scales {} bytes for [{out_dim}, {in_dim}]",
                packed.len(),
                scales.len()
            )));
        }
        let nk = in_dim / 128;
        let mt = out_dim.div_ceil(128);
        let mut ptm = vec![0u8; mt * nk * 128 * 64];
        let mut stm = vec![0u8; mt * nk * 128 * 8];
        for r in 0..out_dim {
            let (t, rr) = (r / 128, r % 128);
            for ks in 0..nk {
                let blk = (t * nk + ks) * 128 + rr;
                ptm[blk * 64..blk * 64 + 64]
                    .copy_from_slice(&packed[r * (in_dim / 2) + ks * 64..][..64]);
                stm[blk * 8..blk * 8 + 8]
                    .copy_from_slice(&scales[r * (in_dim / 16) + ks * 8..][..8]);
            }
        }
        // stream-ordered upload - see nvf4_upload's note on the 2026-09-08
        // race (alloc_zeros + a raw cuMemcpyHtoD zeroed planes at random)
        let data = self.stream.clone_htod(&ptm).map_err(drv)?;
        let scale = self.stream.clone_htod(&stm).map_err(drv)?;
        Ok(Nvf4Plane {
            data,
            scale,
            scale2,
            out_dim,
            in_dim,
            layout: Nvf4Layout::Tiled,
            gu_pairs: false,
        })
    }

    /// Upload a modelopt NVFP4 triple in the FRAGMENT layout: the
    /// tile-major blocks of
    /// [`Self::nvf4_upload_tiled`] with each 8 KB (tile, stage) weight block
    /// additionally permuted to `[w:8][k16:8][g:8][u32 t0..t3]`, where u32
    /// (w, k16, g, t) packs the four bytes mma lane (g, t) feeds its
    /// a0..a3 fragment registers: `[row 16w+g byte t, row 16w+g+8 byte t,
    /// row 16w+g byte 4+t, row 16w+g+8 byte 4+t]` of that k16 group. Scale
    /// records keep the tile-major `[row][8B]` order. Same bytes per
    /// logical element - the `_tf` twins are bit-exact per class - but the
    /// tensor-core head's A read becomes one conflict-free LDS.32 per
    /// fragment group (probe: 167.0 -> 159.2 us b32 at the lm_head shape).
    /// Requires `in_dim % 128 == 0`; callers gate on [`Self::has_nvf4_tf`].
    pub fn nvf4_upload_frag(
        &self,
        packed: &[u8],
        scales: &[u8],
        scale2: f32,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Nvf4Plane, GpuError> {
        if !in_dim.is_multiple_of(128)
            || packed.len() != out_dim * (in_dim / 2)
            || scales.len() != out_dim * (in_dim / 16)
        {
            return Err(GpuError::Unsupported(format!(
                "nvf4 frag triple geometry mismatch: packed {} scales {} bytes for [{out_dim}, {in_dim}]",
                packed.len(),
                scales.len()
            )));
        }
        let nk = in_dim / 128;
        let mt = out_dim.div_ceil(128);
        let mut ptm = vec![0u8; mt * nk * 128 * 64];
        let mut stm = vec![0u8; mt * nk * 128 * 8];
        for r in 0..out_dim {
            let (t, rr) = (r / 128, r % 128);
            // fragment coordinates of row rr within its 128-row tile
            let (w, g, hr) = (rr / 16, rr & 7, (rr >> 3) & 1);
            for ks in 0..nk {
                let blk = t * nk + ks;
                let src = &packed[r * (in_dim / 2) + ks * 64..][..64];
                for (b, &byte) in src.iter().enumerate() {
                    let (sk, tt, hb) = (b >> 3, b & 3, (b >> 2) & 1);
                    ptm[blk * 8192 + ((w * 8 + sk) * 8 + g) * 16 + tt * 4 + hb * 2 + hr] = byte;
                }
                let blkr = blk * 128 + rr;
                stm[blkr * 8..blkr * 8 + 8]
                    .copy_from_slice(&scales[r * (in_dim / 16) + ks * 8..][..8]);
            }
        }
        // stream-ordered upload - see nvf4_upload's note on the 2026-09-08
        // race (alloc_zeros + a raw cuMemcpyHtoD zeroed planes at random)
        let data = self.stream.clone_htod(&ptm).map_err(drv)?;
        let scale = self.stream.clone_htod(&stm).map_err(drv)?;
        Ok(Nvf4Plane {
            data,
            scale,
            scale2,
            out_dim,
            in_dim,
            layout: Nvf4Layout::Frag,
            gu_pairs: false,
        })
    }

    /// Tensor-level dequant of a checkpoint plane to f32 - the CUDA side of
    /// the oracle gate (bit-exact vs `paddock_models::modelopt`'s host
    /// reference: (e2m1 * e4m3) * scale2 per element). Debug/oracle only -
    /// serving never materializes a dequantized plane.
    /// NVFP4 plane -> per-ROW e4m3 plane, for the f8t tile-image decode lane.
    ///
    /// Why this exists: the qwen3.8 NVFP4 lane served its FFN through the
    /// W4A16 nvf4 GEMM family, which profiles L1-bound on software dequant -
    /// 0.7 TB/s against a ~7 TB/s roof. On the same checkpoint the Q8-sourced
    /// f8t tile lane decodes ~2.7x faster: the plane class, not the schedule,
    /// is the whole gap.
    ///
    /// This keeps the CHECKPOINT's values (dequant is exact: an e2m1 value
    /// times its e4m3 block scale carries at most 5 mantissa bits) and only
    /// changes the on-device representation to the one the fast lane reads.
    /// The re-encode to per-row e4m3 is a labeled precision step of the same
    /// class the Q8-sourced lane already takes, and strictly more precise than
    /// the rival's W4A4 activations.
    ///
    /// Costs ~4 bytes/param of transient f32 while converting (356 MB at the
    /// widest qwen3.8 plane) and leaves 1 byte/param resident, so it is gated
    /// on VRAM headroom by the caller.
    pub fn nvf4_to_f8row(
        &self,
        w: &crate::gpu::Nvf4Plane,
    ) -> Result<crate::gpu::F8RowPlane, GpuError> {
        let n = w.out_dim * w.in_dim;
        let mut deq: CudaSlice<f32> = self.stream.alloc_zeros(n).map_err(drv)?;
        self.nvf4_dequant_f32(w, &mut deq)?;
        let mut data = self.alloc_u8(n)?;
        let mut scale: CudaSlice<f32> = self.stream.alloc_zeros(w.out_dim).map_err(drv)?;
        // rows are the out dim; each row is one in_dim-long contiguous run,
        // which is exactly the (n_dim, batch) shape the row quantizer takes
        self.quantize_e4m3_row_u8(&deq, &mut data, &mut scale, w.in_dim, w.out_dim)?;
        Ok(crate::gpu::F8RowPlane { data, scale })
    }

    /// Byte-filled device alloc: the DSL gemms' block-scale path treats an
    /// sf byte of 0x00 as UB (tile-wide contamination measured) - activation
    /// sf slabs must hold VALID e4m3 bytes everywhere from the start.
    pub fn alloc_u8_filled(&self, n: usize, byte: u8) -> Result<CudaSlice<u8>, GpuError> {
        self.stream.clone_htod(&vec![byte; n]).map_err(drv)
    }

    /// NVFP4 checkpoint plane -> f8w plane (e4m3 payload + one ue8m0 scale
    /// per 32 values), the class `f8_gemm_w8`/`f8_gemm_w8_o16` read. This is
    /// the PREFILL twin of [`Self::nvf4_to_f8row`]: the tile-image f8t plane
    /// serves the decode band (r <= 64) and this one serves the wide prefill
    /// pass, exactly as the Dense lane pairs its f8t and f8w planes.
    ///
    /// Costs ~4 bytes/param of transient f32 while converting and leaves
    /// ~1.03 bytes/param resident, so the caller gates it on VRAM headroom.
    pub fn nvf4_to_f8w(&self, w: &crate::gpu::Nvf4Plane) -> Result<RepackedMxfp4, GpuError> {
        let n = w.out_dim * w.in_dim;
        if !n.is_multiple_of(32) {
            return Err(GpuError::Unsupported(
                "nvf4_to_f8w needs a 32-aligned plane".into(),
            ));
        }
        let mut deq: CudaSlice<f32> = self.stream.alloc_zeros(n).map_err(drv)?;
        self.nvf4_dequant_f32(w, &mut deq)?;
        let mut data = self.alloc_u8(n)?;
        let mut scale = self.alloc_u8(n / 32)?;
        self.quantize_e4m3_u8(&deq, &mut data, &mut scale, n)?;
        Ok(RepackedMxfp4 { data, scale })
    }

    /// Two NVFP4 planes -> one fused f8w plane concatenated along out rows
    /// ([a-rows | b-rows]) - the gate|up merge the prefill arm row-slices at
    /// offsets 0/ff. Byte-identical to converting the two separately: a
    /// per-32 block never straddles the seam (each row is `in_dim` long and
    /// in_dim % 32 == 0), so this is offset math only.
    pub fn nvf4_to_f8w_concat2(
        &self,
        a: &crate::gpu::Nvf4Plane,
        b: &crate::gpu::Nvf4Plane,
    ) -> Result<RepackedMxfp4, GpuError> {
        if a.in_dim != b.in_dim {
            return Err(GpuError::Unsupported(
                "fused f8w planes need one in_dim".into(),
            ));
        }
        let q = self
            .kernels
            .quantize_e4m3
            .ok_or(GpuError::MissingOp("quantize_e4m3"))?;
        let (na, nb) = (a.out_dim * a.in_dim, b.out_dim * b.in_dim);
        if (na % 32) + (nb % 32) != 0 {
            return Err(GpuError::Unsupported(
                "nvf4_to_f8w needs 32-aligned planes".into(),
            ));
        }
        let mut data = self.alloc_u8(na + nb)?;
        let mut scale = self.alloc_u8((na + nb) / 32)?;
        for (w, off, n) in [(a, 0usize, na), (b, na, nb)] {
            // one plane's f32 dequant at a time (4 B/param transient)
            let mut deq: CudaSlice<f32> = self.stream.alloc_zeros(n).map_err(drv)?;
            self.nvf4_dequant_f32(w, &mut deq)?;
            let (xp, _g1) = deq.device_ptr(&self.stream);
            let (dp, _g2) = data.device_ptr_mut(&self.stream);
            let (sp, _g3) = scale.device_ptr_mut(&self.stream);
            // SAFETY: ABI contract; n % 32 == 0, sub-view bounds checked above
            check(unsafe {
                q(
                    xp as *const _,
                    (dp as usize + off) as *mut _,
                    (sp as usize + off / 32) as *mut _,
                    n as u32,
                    self.stream_ptr(),
                )
            })?;
            self.stream.synchronize().map_err(drv)?;
        }
        Ok(RepackedMxfp4 { data, scale })
    }

    pub fn nvf4_dequant_f32(&self, w: &Nvf4Plane, y: &mut CudaSlice<f32>) -> Result<(), GpuError> {
        let f = self
            .kernels
            .nvf4_dequant
            .ok_or(GpuError::MissingOp("nvf4_dequant"))?;
        if w.layout != Nvf4Layout::Row {
            return Err(GpuError::Unsupported(
                "nvf4_dequant reads the row-major layout; tiled/frag plane refused".into(),
            ));
        }
        if y.len() != w.out_dim * w.in_dim {
            return Err(GpuError::Unsupported(format!(
                "nvf4_dequant out buffer {} != {} x {}",
                y.len(),
                w.out_dim,
                w.in_dim
            )));
        }
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (sp, _g2) = w.scale.device_ptr(&self.stream);
        let (yp, _g3) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; geometry validated above
        check(unsafe {
            f(
                dp as *const _,
                sp as *const _,
                w.scale2,
                yp as *mut _,
                w.in_dim as u32,
                w.out_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    /// W4A16-class GEMV over a checkpoint plane: f32 activations, scale2
    /// folded once after the reduction (exact - it factors out of the dot).
    pub fn nvf4_gemv(
        &self,
        w: &Nvf4Plane,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        bias: Option<&CudaSlice<f32>>,
    ) -> Result<(), GpuError> {
        // a tiled/frag plane rides its batch twin at batch=1 - the batch
        // kernels are bit-exact per row vs this one, so the class holds
        if w.layout != Nvf4Layout::Row {
            return self.nvf4_gemv_batch(w, x, y, bias, 1);
        }
        let f = self
            .kernels
            .nvf4_gemv
            .ok_or(GpuError::MissingOp("nvf4_gemv"))?;
        if x.len() < w.in_dim || y.len() < w.out_dim {
            return Err(GpuError::Unsupported(format!(
                "nvf4_gemv x {} / y {} too small for [{}, {}]",
                x.len(),
                y.len(),
                w.out_dim,
                w.in_dim
            )));
        }
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (sp, _g2) = w.scale.device_ptr(&self.stream);
        let (xp, _g3) = x.device_ptr(&self.stream);
        let (yp, _g4) = y.device_ptr_mut(&self.stream);
        let bp = match bias {
            Some(b) => {
                let (p, _g) = b.device_ptr(&self.stream);
                p as *const core::ffi::c_void
            }
            None => core::ptr::null(),
        };
        // SAFETY: ABI contract; geometry validated above
        check(unsafe {
            f(
                dp as *const _,
                sp as *const _,
                bp,
                xp as *const _,
                yp as *mut _,
                w.scale2,
                w.in_dim as u32,
                w.out_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    /// True when the row-batched GEMV is loadable (the batch lane's lm_head
    /// consumer - `enable_batch` gates on it).
    /// True when the pack carries the merged q|k|v NVFP4 GEMV (slot 492).
    pub fn has_nvf4_gemv_multi(&self) -> bool {
        self.kernels.nvf4_gemv_multi.is_some()
    }

    /// One GEMV launch over up to three checkpoint planes that share `x`.
    ///
    /// The point is OCCUPANCY, not arithmetic: at 8 rows per CTA a plane with
    /// `out_dim` 1024 is 128 CTAs on a 188-SM die, so it pays a full launch's
    /// ramp/drain for a quarter of the bytes. granite's k/v are exactly that
    /// and measured the same ~8.5 us as q/o at 4x the bytes. The Q8
    /// twin already measured this merge on the same die: 1024-row 724 GB/s,
    /// 4096-row 1254, merged 6144-row 1303 -- 26.5 us of separate launches
    /// down to 20.5.
    ///
    /// Bit-exact per row against [`Self::nvf4_gemv`]: same inner loop, same
    /// reduction, same epilogue order. Only the grid changed.
    pub fn nvf4_gemv_multi(
        &self,
        planes: &[(&crate::gpu::Nvf4Plane, &mut CudaSlice<f32>)],
        x: &CudaSlice<f32>,
        in_dim: usize,
    ) -> Result<(), GpuError> {
        #[repr(C)]
        struct Seg {
            data: *const u8,
            scale: *const u8,
            bias: *const f32,
            y: *mut f32,
            scale2: f32,
            out_dim: u32,
        }
        let f = self
            .kernels
            .nvf4_gemv_multi
            .ok_or(GpuError::MissingOp("nvf4_gemv_multi"))?;
        if planes.is_empty() || planes.len() > 3 {
            return Err(GpuError::Unsupported(
                "nvf4_gemv_multi takes 1..=3 planes".into(),
            ));
        }
        // Hold the device-ptr guards for the whole launch - dropping one early
        // would let the pool reuse the allocation under a running kernel.
        let mut guards = Vec::with_capacity(planes.len() * 3);
        let mut segs: Vec<Seg> = Vec::with_capacity(planes.len());
        for (w, _y) in planes {
            if w.layout != Nvf4Layout::Row {
                return Err(GpuError::Unsupported(
                    "nvf4_gemv_multi reads the row-major layout".into(),
                ));
            }
            if w.in_dim != in_dim {
                return Err(GpuError::Unsupported(format!(
                    "nvf4_gemv_multi: plane in_dim {} != shared x {in_dim}",
                    w.in_dim
                )));
            }
            let (dp, g1) = w.data.device_ptr(&self.stream);
            let (sp, g2) = w.scale.device_ptr(&self.stream);
            guards.push(g1);
            guards.push(g2);
            segs.push(Seg {
                data: dp as *const u8,
                scale: sp as *const u8,
                bias: std::ptr::null(),
                y: std::ptr::null_mut(),
                scale2: w.scale2,
                out_dim: w.out_dim as u32,
            });
        }
        // y pointers taken in a second pass: `planes` holds &mut, so the
        // mutable borrows cannot overlap the immutable ones above.
        let mut ys = Vec::with_capacity(planes.len());
        for (i, seg) in segs.iter_mut().enumerate() {
            let (yp, g) = planes[i].1.device_ptr(&self.stream);
            ys.push(g);
            seg.y = yp as *mut f32;
        }
        let (xp, _gx) = x.device_ptr(&self.stream);
        // SAFETY: ABI contract; every plane validated row-major with a
        // matching in_dim, and 1..=3 segments checked above.
        check(unsafe {
            f(
                segs.as_ptr() as *const _,
                xp as *const _,
                in_dim as u32,
                segs.len() as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_nvf4_gemv_batch(&self) -> bool {
        self.kernels.nvf4_gemv_batch.is_some()
    }

    /// True when the tensor-core NVFP4 GEMM (slot 421) is loadable.
    pub fn has_nvf4_gemm_tc(&self) -> bool {
        self.kernels.nvf4_gemm_tc.is_some()
    }

    /// Tensor-core twin of [`Self::nvf4_gemv_batch`]:
    /// exact-dequant bf16 weights on m16n8k16 mma. A numeric CLASS change -
    /// the activation cast f32->bf16 plus mma reassociation - so it is not
    /// bit-comparable to the scalar lane and callers elect it explicitly
    /// (the head sites gate on batch > 8, wide plane, and the
    /// PADDOCK_NVF4_TC kill switch). No fallback here: the caller already
    /// checked `has_nvf4_gemm_tc`.
    pub fn nvf4_gemm_tc(
        &self,
        w: &Nvf4Plane,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        bias: Option<&CudaSlice<f32>>,
        batch: usize,
    ) -> Result<(), GpuError> {
        // tiled/frag planes ride their layout's twin - each bit-exact vs
        // the row-major tc family's class on the same logical plane
        let f = match w.layout {
            Nvf4Layout::Frag => self
                .kernels
                .nvf4_gemm_tc_tf
                .ok_or(GpuError::MissingOp("nvf4_gemm_tc_tf"))?,
            Nvf4Layout::Tiled => self
                .kernels
                .nvf4_gemm_tc_tm
                .ok_or(GpuError::MissingOp("nvf4_gemm_tc_tm"))?,
            Nvf4Layout::Row => self
                .kernels
                .nvf4_gemm_tc
                .ok_or(GpuError::MissingOp("nvf4_gemm_tc"))?,
        };
        if x.len() < batch * w.in_dim || y.len() < batch * w.out_dim {
            return Err(GpuError::Unsupported(format!(
                "nvf4_gemm_tc x {} / y {} too small for {batch} x [{}, {}]",
                x.len(),
                y.len(),
                w.out_dim,
                w.in_dim
            )));
        }
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (sp, _g2) = w.scale.device_ptr(&self.stream);
        let (xp, _g3) = x.device_ptr(&self.stream);
        let (yp, _g4) = y.device_ptr_mut(&self.stream);
        let bp = match bias {
            Some(b) => {
                let (p, _g) = b.device_ptr(&self.stream);
                p as *const core::ffi::c_void
            }
            None => core::ptr::null(),
        };
        // SAFETY: ABI contract; geometry validated above
        check(unsafe {
            f(
                dp as *const _,
                sp as *const _,
                bp,
                xp as *const _,
                yp as *mut _,
                w.scale2,
                w.in_dim as u32,
                w.out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// True when the checkpoint-plane W4A4 GEMM (slot 426) is loadable.
    pub fn has_nvf4_gemm_f4(&self) -> bool {
        self.kernels.nvf4_gemm_f4.is_some()
    }

    /// The W4A4 family's prefill-width arm (the f4t TMA ring, rows >= 128,
    /// in_dim % 256): what a lane asks before electing W4A4 for its wide band.
    pub fn has_nvf4_gemm_f4t(&self) -> bool {
        self.kernels.nvf4_gemm_f4t.is_some()
    }

    /// Checkpoint-plane W4A4 GEMM: the fp4 x fp4 block-scale mma
    /// over an [`Nvf4Plane`], epilogue acc*scale2 (+bias). xq/xs are
    /// nvf4-quantized activations from [`Self::quantize_nvf4`]/
    /// [`Self::quantize_nvf4_swiglu`] - a numeric CLASS change vs the
    /// scalar/tc lanes (lossy e2m1 activations), which is the checkpoint's
    /// own declared recipe (input_activations 4-bit, group 16, e4m3 scales)
    /// and the binding rival's serving class. Callers elect it explicitly
    /// behind the model-level quality gate.
    ///
    /// Routes across the family by probe-elected rules (nv4_ffn_probe at
    /// the qwen3.8 FFN shapes): tile grids under 64 CTAs starve
    /// the machine (down at decode: 40 on 188 SMs) and take the split-K
    /// twin (sk=4, `part` scratch, deterministic two-pass reduce, 3.7-4.9x
    /// there); everything else takes v2's ST=3 ring (async scale planes,
    /// one barrier per K-step, 1.3-1.8x); v1 (slot 426) is the fallback for
    /// old packs and ragged in_dim.
    /// Split-K depth of the nvf4 decode arms (f4cn / f4c / f4s and their raw
    /// twins) -- a function of the plane's out tiles against the die, not a
    /// constant. Cold-L2 sweep (bench/nv4_dec_sk.cu, B=8/32,
    /// granite 8b+30b shapes, weight pool rotated past L2): a grid that fills
    /// the die UNSPLIT (out/128 tiles >= SM count) streams at ~89% of the DRAM
    /// wall and loses to every split -- 8b gate|up 43.1 us sk1 vs 49.7 sk4,
    /// 30b gate|up 110.8 vs 133.8 -- because the split's partials are extra
    /// traffic on a grid that was never starved; the small-out shapes need the
    /// split (down 45% -> 86% of the wall, o 38% -> 50%) and sk4 is their best
    /// depth at every K probed (12800, 32768). The old `const SK: usize = 4`
    /// was that small-shape verdict applied to gate|up on both sizes.
    /// PADDOCK_NV4_SK=<n> pins one depth for every shape (the A/B; 4 = the
    /// shipped constant). Unsplit also means no partials, so the consumer
    /// takes the plain-y path (nvf4_gemm_f4_raw_parts returns None).
    pub fn nv4_decode_sk(&self, out_dim: usize) -> usize {
        static PIN: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
        let pin = *PIN.get_or_init(|| {
            paddock_models::dev_var!("PADDOCK_NV4_SK")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
        });
        if let Some(n) = pin {
            return n.max(1);
        }
        if out_dim.div_ceil(128) >= self.sm_count() {
            1
        } else {
            4
        }
    }

    /// Deep-ring decode arm (bench/nv4_dec_deep{,2}.cu, cold L2).
    /// On a small-out plane (tiles < SM count) the f4cn 2-stage ring waits
    /// for everything before each 256-K chunk and, with ~16 mma per warp per
    /// chunk, runs latency-SERIALIZED; a split only adds partials. The same
    /// tile with 3 chunks in flight, unsplit, is 25-30% faster (qkv 6144x4096:
    /// 17.9 -> 12.3 us). The 128-row layout then caps at ~26 GB/s of cp.async
    /// issue per CTA (one CTA per SM), which the 64-row tile (2 row groups x
    /// 4 col groups per CTA, twice the CTAs) lifts: o 12.3 -> 10.2, 8b down
    /// 26.6 -> 25.2, 30b down 57.7 -> 55.4, qwen3.8 qkv 22.5 -> 16.7 / o 18.4
    /// -> 12.7 / down 42.8 -> 39.8 (rt 64, st 4, sk 1). Every variant is
    /// bit-identical to f4cn at the same split (each output's K sequence is
    /// unchanged). Returns (st, rt) or None. The serve is the arbiter (a
    /// 92 KB 1-CTA/SM ring gained nothing under PDL co-residency on 8b):
    /// PADDOCK_NV4_F4CD_ST=3|4, _RT=64|128, _LONGK=0|1 pin the A/B arms;
    /// kill PADDOCK_NO_NV4_F4CD.
    pub fn nv4_decode_deep(&self, out_dim: usize, in_dim: usize) -> Option<(u32, u32)> {
        static CFG: std::sync::OnceLock<Option<(u32, u32, bool)>> = std::sync::OnceLock::new();
        let cfg = *CFG.get_or_init(|| {
            if paddock_models::dev_var_os!("PADDOCK_NO_NV4_F4CD").is_some() {
                return None;
            }
            let st = paddock_models::dev_var!("PADDOCK_NV4_F4CD_ST")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .filter(|v| *v == 3 || *v == 4)
                .unwrap_or(4);
            let rt = paddock_models::dev_var!("PADDOCK_NV4_F4CD_RT")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .filter(|v| *v == 64 || *v == 128)
                .unwrap_or(64);
            let longk = paddock_models::dev_var!("PADDOCK_NV4_F4CD_LONGK")
                .ok()
                .map(|v| v != "0")
                .unwrap_or(true);
            Some((st, rt, longk))
        });
        let (st, rt, longk) = cfg?;
        if out_dim.div_ceil(128) >= self.sm_count() {
            return None;
        }
        if !longk && in_dim / 256 > 24 {
            return None;
        }
        Some((st, rt))
    }

    pub fn nvf4_gemm_f4(
        &self,
        w: &Nvf4Plane,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<u8>,
        y: &mut CudaSlice<f32>,
        bias: Option<&CudaSlice<f32>>,
        batch: usize,
        mut part: Option<&mut CudaSlice<f32>>,
    ) -> Result<(), GpuError> {
        if w.layout != Nvf4Layout::Row {
            return Err(GpuError::Unsupported(
                "nvf4_gemm_f4 reads the row-major layout; tiled/frag plane refused".into(),
            ));
        }
        if xq.len() < batch * w.in_dim / 2
            || xs.len() < batch * w.in_dim / 16
            || y.len() < batch * w.out_dim
        {
            return Err(GpuError::Unsupported(format!(
                "nvf4_gemm_f4 xq {} / xs {} / y {} too small for {batch} x [{}, {}]",
                xq.len(),
                xs.len(),
                y.len(),
                w.out_dim,
                w.in_dim
            )));
        }
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (sp, _g2) = w.scale.device_ptr(&self.stream);
        let (xqp, _g3) = xq.device_ptr(&self.stream);
        let (xsp, _g4) = xs.device_ptr(&self.stream);
        let (yp, _g5) = y.device_ptr_mut(&self.stream);
        let bp = match bias {
            Some(b) => {
                let (p, _g) = b.device_ptr(&self.stream);
                p as *const core::ffi::c_void
            }
            None => core::ptr::null(),
        };
        let aligned = w.in_dim.is_multiple_of(128);
        let batch_pad = (batch + 127) & !127;
        let ntiles = w.out_dim.div_ceil(128) * (batch_pad >> 7);
        // shape-elected split depth (see nv4_decode_sk); 1 = unsplit
        let sk_e = self.nv4_decode_sk(w.out_dim);
        // Split even the die-filling large-out GEMMs (gate/up) at DECODE batch:
        // nv4c is launch_bounds(256,1) = 1 CTA/SM, so a single CTA walking the
        // full k stalls on memory; K-split shortens each CTA's walk and gives
        // the scheduler more waves to hide latency; gated batch<=32 (only the
        // confirmed-win regime -- adjacent work flags batch<=64 as a broken
        // machine-fill proxy, p64_f16_audio_gemm). Probe (nv4_ffn_probe_gr,
        // granite gate 25600x4096): b32 v3-sk1 65.5us -> v3-sk4 53.4us (-18%);
        // but it INVERTS at b>=128 (sk4 81 vs sk1 68), so gate on batch.
        let decode_split =
            batch <= 32 && paddock_models::dev_var_os!("PADDOCK_NO_NV4_DECODE_SPLIT").is_none();
        // TMA ring: the prefill band's throughput arm. A round of granite
        // garbage output was not this kernel - it is bit-exact vs f4c at
        // every shape (nv4_f4t_1shot.cu) - but the f4t TMA-map cache
        // keyed on the activation POINTER without in_dim, so granite's `down`
        // GEMM reused q's narrow map and read a zero-filled activation. Fixed
        // in nvf4.cuh (key now (ptr, batch, in_dim); weight key adds out_dim).
        // PADDOCK_NVF4_F4T=0 forces the f4c fallback (the safe arm) for A/B.
        static F4T_ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let f4t_on = *F4T_ON.get_or_init(|| {
            paddock_models::dev_var!("PADDOCK_NVF4_F4T")
                .map(|v| v != "0")
                .unwrap_or(true)
        });
        // f4t4: the same ring four 128-K stages deep in the same 72 KB
        // (quant/nvf4.cuh). Elected on dies under 128 SMs, where the 2-stage
        // ring measured ~46% of the fp4 roof (GB10 2026-09-07: three loads
        // in flight instead of one is the whole difference); the 188-SM die
        // keeps f4t until it is measured there. PADDOCK_NVF4_F4T4=1 forces
        // it anywhere, =0 kills it. Bit-exact vs f4t/f4c (same K order).
        static F4T4: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
        let f4t4_pin = *F4T4.get_or_init(|| {
            paddock_models::dev_var!("PADDOCK_NVF4_F4T4")
                .ok()
                .map(|v| v != "0")
        });
        // Shape rule from the bench: f4t4 wins where the epilogue dominates
        // (out_dim > in_dim: gate/up 782 vs 839 us at 1k rows, 1552 vs 1685
        // at 2k) and loses 3-4% on the down plane (out_dim < in_dim), so the
        // default election takes the wide-output planes only.
        let f4t4_on = f4t4_pin.unwrap_or(self.sm_count() < 128 && w.out_dim > w.in_dim);
        if f4t_on
            && f4t4_on
            && w.in_dim.is_multiple_of(256)
            && batch >= 128
            && ntiles >= 64
            && let Some(ft4) = self.kernels.nvf4_gemm_f4t4
        {
            // SAFETY: ABI contract (slot 586 = slot 430's); geometry validated above
            return check(unsafe {
                ft4(
                    dp as *const _,
                    sp as *const _,
                    bp,
                    xqp as *const _,
                    xsp as *const _,
                    yp as *mut _,
                    w.scale2,
                    w.in_dim as u32,
                    w.out_dim as u32,
                    batch as u32,
                    self.stream_ptr(),
                )
            });
        }
        if f4t_on
            && w.in_dim.is_multiple_of(256)
            && batch >= 128
            && ntiles >= 64
            && let Some(ft) = self.kernels.nvf4_gemm_f4t
        {
            // SAFETY: ABI contract; geometry validated above
            return check(unsafe {
                ft(
                    dp as *const _,
                    sp as *const _,
                    bp,
                    xqp as *const _,
                    xsp as *const _,
                    yp as *mut _,
                    w.scale2,
                    w.in_dim as u32,
                    w.out_dim as u32,
                    batch as u32,
                    self.stream_ptr(),
                )
            });
        }
        // Decode narrow-tile arm: the BN=32 / WR=144 twin at
        // launch_bounds(256,2) => 2 CTA/SM. The wide f4c tile (88 KB smem,
        // acc[16][4]) pinned decode at 1 CTA/SM / 16.6% occupancy (DRAM
        // 40%, compute 20% -- occupancy-starved, not bandwidth); f4cn shrinks
        // smem+registers and runs 2 blocks/SM. Bit-exact vs f4c at the same
        // shape (nv4_dec_cmp.cu: diffs=0 at B=8/16/32), ~21-34% faster on every
        // granite decode projection. batch<=32 only (its 32-col tile). sk=4
        // (probe-best); needs `part`. Kill: PADDOCK_NO_NV4_F4CN.
        static F4CN_ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let f4cn_on = *F4CN_ON.get_or_init(|| {
            paddock_models::dev_var!("PADDOCK_NO_NV4_F4CN")
                .map(|v| v == "0")
                .unwrap_or(true)
        });
        if f4cn_on
            && batch <= 32
            && w.in_dim.is_multiple_of(256)
            && let Some((st, rt)) = self.nv4_decode_deep(w.out_dim, w.in_dim)
            && let Some(fcd) = self.kernels.nvf4_gemm_f4cd
        {
            // deep ring, unsplit: y directly, no partials (see nv4_decode_deep)
            // SAFETY: ABI contract; geometry validated above
            return check(unsafe {
                fcd(
                    dp as *const _,
                    sp as *const _,
                    bp,
                    xqp as *const _,
                    xsp as *const _,
                    core::ptr::null_mut(),
                    yp as *mut _,
                    w.scale2,
                    w.in_dim as u32,
                    w.out_dim as u32,
                    batch as u32,
                    1u32,
                    st,
                    rt,
                    self.stream_ptr(),
                )
            });
        }
        if f4cn_on
            && batch <= 32
            && w.in_dim.is_multiple_of(256)
            && let Some(fcn) = self.kernels.nvf4_gemm_f4cn
        {
            // sk_e == 1: the unsplit twin writes y directly, no partials
            let pp: *mut core::ffi::c_void = if sk_e >= 2 {
                match part.as_deref_mut() {
                    Some(p) if p.len() >= sk_e * batch * w.out_dim => {
                        let (pp, _g6) = p.device_ptr_mut(&self.stream);
                        pp as *mut core::ffi::c_void
                    }
                    _ => core::ptr::null_mut(),
                }
            } else {
                core::ptr::null_mut()
            };
            if sk_e < 2 || !pp.is_null() {
                let sk = if pp.is_null() { 1u32 } else { sk_e as u32 };
                // SAFETY: ABI contract; geometry validated above; a null part
                // only ever reaches the launcher with sk == 1 (unsplit path)
                return check(unsafe {
                    fcn(
                        dp as *const _,
                        sp as *const _,
                        bp,
                        xqp as *const _,
                        xsp as *const _,
                        pp,
                        yp as *mut _,
                        w.scale2,
                        w.in_dim as u32,
                        w.out_dim as u32,
                        batch as u32,
                        sk,
                        self.stream_ptr(),
                    )
                });
            }
        }
        // KC=256 arm: fastest cp.async shape, split only on
        // machine-starved grids (nv4_ffn_probe: gate 136 tiles sk1 41.4 us
        // vs sk4 47; down 40 tiles sk4 45 vs sk1 112.9).
        if w.in_dim.is_multiple_of(256)
            && let Some(fc) = self.kernels.nvf4_gemm_f4c
        {
            let want_split = (ntiles < 64 || decode_split) && sk_e >= 2;
            let (sk, pp) = match part {
                Some(p) if want_split && p.len() >= sk_e * batch * w.out_dim => {
                    let (pp, _g6) = p.device_ptr_mut(&self.stream);
                    (sk_e as u32, pp as *mut core::ffi::c_void)
                }
                _ => (1u32, core::ptr::null_mut()),
            };
            // SAFETY: ABI contract; geometry validated above
            return check(unsafe {
                fc(
                    dp as *const _,
                    sp as *const _,
                    bp,
                    xqp as *const _,
                    xsp as *const _,
                    pp,
                    yp as *mut _,
                    w.scale2,
                    w.in_dim as u32,
                    w.out_dim as u32,
                    batch as u32,
                    sk,
                    self.stream_ptr(),
                )
            });
        }
        if aligned
            && (ntiles < 64 || decode_split)
            && sk_e >= 2
            && let (Some(fs), Some(p)) = (self.kernels.nvf4_gemm_f4s, part)
            && p.len() >= sk_e * batch * w.out_dim
            && w.in_dim / 128 >= sk_e
        {
            let (pp, _g6) = p.device_ptr_mut(&self.stream);
            // SAFETY: ABI contract; geometry validated above
            return check(unsafe {
                fs(
                    dp as *const _,
                    sp as *const _,
                    bp,
                    xqp as *const _,
                    xsp as *const _,
                    pp as *mut _,
                    yp as *mut _,
                    w.scale2,
                    w.in_dim as u32,
                    w.out_dim as u32,
                    batch as u32,
                    sk_e as u32,
                    self.stream_ptr(),
                )
            });
        }
        if aligned && let Some(fb) = self.kernels.nvf4_gemm_f4b {
            // SAFETY: ABI contract; geometry validated above
            return check(unsafe {
                fb(
                    dp as *const _,
                    sp as *const _,
                    bp,
                    xqp as *const _,
                    xsp as *const _,
                    yp as *mut _,
                    w.scale2,
                    w.in_dim as u32,
                    w.out_dim as u32,
                    batch as u32,
                    3u32,
                    self.stream_ptr(),
                )
            });
        }
        let f = self
            .kernels
            .nvf4_gemm_f4
            .ok_or(GpuError::MissingOp("nvf4_gemm_f4"))?;
        // SAFETY: ABI contract; geometry validated above
        check(unsafe {
            f(
                dp as *const _,
                sp as *const _,
                bp,
                xqp as *const _,
                xsp as *const _,
                yp as *mut _,
                w.scale2,
                w.in_dim as u32,
                w.out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Split GEMM into RAW K-split partials, no fold (granite NVFP4 fused
    /// qkv). Elects exactly the split the reducing route
    /// ([`Self::nvf4_gemm_f4`]'s f4c arm) would take at this shape -- same
    /// kernel, same sk, same slice order -- and returns that sk so the
    /// consumer folds the planes with the same fixed-order sum and applies
    /// `w.scale2` after it (the reduce kernel's own math). `Ok(None)` means
    /// this shape would not split on the reducing route (wide rows take the
    /// TMA ring, big grids run unsplit): the caller then runs
    /// [`Self::nvf4_gemm_f4`] into `y` and consumes it as one plane.
    /// Bit-identical to reduce-then-consume by construction.
    pub fn nvf4_gemm_f4_raw_parts(
        &self,
        w: &Nvf4Plane,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<u8>,
        part: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<Option<u32>, GpuError> {
        // shape-elected depth; an unsplit shape has no partials to hand out
        let sk_e = self.nv4_decode_sk(w.out_dim);
        if sk_e < 2 {
            return Ok(None);
        }
        let Some(f) = self.kernels.nvf4_gemm_f4c_raw else {
            return Ok(None);
        };
        if w.layout != Nvf4Layout::Row
            || !w.in_dim.is_multiple_of(256)
            || self.kernels.nvf4_gemm_f4c.is_none()
        {
            return Ok(None);
        }
        let batch_pad = (batch + 127) & !127;
        let ntiles = w.out_dim.div_ceil(128) * (batch_pad >> 7);
        // mirror of nvf4_gemm_f4's f4c election: the TMA ring owns batch >= 128
        // on die-filling grids; below that the split is taken on starved
        // grids or at decode batch (the confirmed-win regime)
        if w.in_dim.is_multiple_of(256)
            && batch >= 128
            && ntiles >= 64
            && self.kernels.nvf4_gemm_f4t.is_some()
        {
            return Ok(None);
        }
        let decode_split =
            batch <= 32 && paddock_models::dev_var_os!("PADDOCK_NO_NV4_DECODE_SPLIT").is_none();
        if !(ntiles < 64 || decode_split) || part.len() < sk_e * batch * w.out_dim {
            return Ok(None);
        }
        if xq.len() < batch * w.in_dim / 2 || xs.len() < batch * w.in_dim / 16 {
            return Err(GpuError::Unsupported(format!(
                "nvf4_gemm_f4_raw_parts xq {} / xs {} too small for {batch} x [{}, {}]",
                xq.len(),
                xs.len(),
                w.out_dim,
                w.in_dim
            )));
        }
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (sp, _g2) = w.scale.device_ptr(&self.stream);
        let (xqp, _g3) = xq.device_ptr(&self.stream);
        let (xsp, _g4) = xs.device_ptr(&self.stream);
        let (pp, _g5) = part.device_ptr_mut(&self.stream);
        // Decode narrow-tile arm: at batch<=32 f4cn's 2-CTA/SM
        // tile is ~30% faster than the f4c (1 CTA/SM) raw arm; its RAW twin
        // writes the same [sk][batch][out] partials without the reduce, so the
        // consumer folds them identically. Bit-exact vs f4c_raw (nv4_fold_cmp
        // diffs=0); this speeds the qkv seat too (it already rides raw_parts).
        if batch <= 32
            && let Some((st, rt)) = self.nv4_decode_deep(w.out_dim, w.in_dim)
            && let Some(fcd) = self.kernels.nvf4_gemm_f4cd_raw
        {
            // deep ring, one raw slice (nz = 1): the from_parts consumers fold
            // a single slice x scale2 -- same bytes as f4cn's unsplit y before
            // its epilogue scale
            // SAFETY: ABI contract; geometry validated above
            check(unsafe {
                fcd(
                    dp as *const _,
                    sp as *const _,
                    core::ptr::null(),
                    xqp as *const _,
                    xsp as *const _,
                    pp as *mut _,
                    w.scale2,
                    w.in_dim as u32,
                    w.out_dim as u32,
                    batch as u32,
                    1u32,
                    st,
                    rt,
                    self.stream_ptr(),
                )
            })?;
            return Ok(Some(1u32));
        }
        if batch <= 32
            && let Some(fcn) = self.kernels.nvf4_gemm_f4cn_raw
        {
            // SAFETY: ABI contract; geometry validated above
            check(unsafe {
                fcn(
                    dp as *const _,
                    sp as *const _,
                    core::ptr::null(),
                    xqp as *const _,
                    xsp as *const _,
                    pp as *mut _,
                    w.scale2,
                    w.in_dim as u32,
                    w.out_dim as u32,
                    batch as u32,
                    sk_e as u32,
                    self.stream_ptr(),
                )
            })?;
            return Ok(Some(sk_e as u32));
        }
        // SAFETY: ABI contract; geometry validated above
        check(unsafe {
            f(
                dp as *const _,
                sp as *const _,
                xqp as *const _,
                xsp as *const _,
                pp as *mut _,
                w.in_dim as u32,
                w.out_dim as u32,
                batch as u32,
                sk_e as u32,
                self.stream_ptr(),
            )
        })?;
        Ok(Some(sk_e as u32))
    }

    /// Row-batched twin of [`Self::nvf4_gemv`]: x `[batch, in_dim]`,
    /// y `[batch, out_dim]`. Bit-exact per row vs the 1-row GEMV (the
    /// continuous-batching lm_head consumer).
    pub fn nvf4_gemv_batch(
        &self,
        w: &Nvf4Plane,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        bias: Option<&CudaSlice<f32>>,
        batch: usize,
    ) -> Result<(), GpuError> {
        // Prefer the multi-row twin (slot 419): same per-row math
        // bit-exact, but the weight plane streams ceil(batch/16) times
        // instead of `batch` times - the c32 ledger had the per-row kernel
        // re-reading the 177 MB lm_head 32x per tick (4.2 ms of a 25.6 ms
        // ITL). Old packs fall back to the per-row kernel. Tiled/frag
        // planes take the same election over their layout's twins
        // (bit-exact per class; the upload gates guarantee the slots).
        let f = match w.layout {
            Nvf4Layout::Frag => match (batch >= 2, self.kernels.nvf4_gemm_mr_tf) {
                (true, Some(mr)) => mr,
                _ => self
                    .kernels
                    .nvf4_gemv_batch_tf
                    .ok_or(GpuError::MissingOp("nvf4_gemv_batch_tf"))?,
            },
            Nvf4Layout::Tiled => match (batch >= 2, self.kernels.nvf4_gemm_mr_tm) {
                (true, Some(mr)) => mr,
                _ => self
                    .kernels
                    .nvf4_gemv_batch_tm
                    .ok_or(GpuError::MissingOp("nvf4_gemv_batch_tm"))?,
            },
            Nvf4Layout::Row => match (batch >= 2, self.kernels.nvf4_gemm_mr) {
                (true, Some(mr)) => mr,
                _ => self
                    .kernels
                    .nvf4_gemv_batch
                    .ok_or(GpuError::MissingOp("nvf4_gemv_batch"))?,
            },
        };
        if x.len() < batch * w.in_dim || y.len() < batch * w.out_dim {
            return Err(GpuError::Unsupported(format!(
                "nvf4_gemv_batch x {} / y {} too small for {batch} x [{}, {}]",
                x.len(),
                y.len(),
                w.out_dim,
                w.in_dim
            )));
        }
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (sp, _g2) = w.scale.device_ptr(&self.stream);
        let (xp, _g3) = x.device_ptr(&self.stream);
        let (yp, _g4) = y.device_ptr_mut(&self.stream);
        let bp = match bias {
            Some(b) => {
                let (p, _g) = b.device_ptr(&self.stream);
                p as *const core::ffi::c_void
            }
            None => core::ptr::null(),
        };
        // SAFETY: ABI contract; geometry validated above
        check(unsafe {
            f(
                dp as *const _,
                sp as *const _,
                bp,
                xp as *const _,
                yp as *mut _,
                w.scale2,
                w.in_dim as u32,
                w.out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    // ---- NVFP4 MoE expert consumers  ----------------------------

    pub fn has_nvf4_moe(&self) -> bool {
        self.kernels.nvf4_moe_up_relu2.is_some() && self.kernels.nvf4_moe_down_acc.is_some()
    }

    /// Upload a per-layer MoE expert residency: the caller has concatenated
    /// all experts' packed/scale bytes in expert order (row of expert `e` at
    /// `e * ff + r`); `scale2` is the per-expert f32 array. Byte-exact, no
    /// repacking - same layout contract as [`Self::nvf4_upload`], validated
    /// against the full concatenated geometry.
    pub fn nvf4_moe_upload(
        &self,
        packed: &[u8],
        scales: &[u8],
        scale2: &[f32],
        n_expert: usize,
        ff: usize,
        in_dim: usize,
    ) -> Result<Nvf4MoePlane, GpuError> {
        let rows = n_expert * ff;
        if !in_dim.is_multiple_of(32)
            || scale2.len() != n_expert
            || packed.len() != rows * (in_dim / 2)
            || scales.len() != rows * (in_dim / 16)
        {
            return Err(GpuError::Unsupported(format!(
                "nvf4 moe geometry mismatch: packed {} scales {} scale2 {} for {}x[{ff}, {in_dim}]",
                packed.len(),
                scales.len(),
                scale2.len(),
                n_expert
            )));
        }
        // stream-ordered upload - see nvf4_upload's note on the 2026-09-08
        // race (alloc_zeros + a raw cuMemcpyHtoD zeroed planes at random)
        let data = self.stream.clone_htod(packed).map_err(drv)?;
        let scale = self.stream.clone_htod(scales).map_err(drv)?;
        let scale2 = self.stream.clone_htod(scale2).map_err(drv)?;
        Ok(Nvf4MoePlane {
            data,
            scale,
            row_scale: None,
            scale2,
            n_expert,
            ff,
            in_dim,
            layout: Nvf4MoeLayout::Row,
        })
    }

    /// Upload a per-layer MoE expert residency repacked to the piece-major
    /// 64x64 TILED layout the `_st`/`_stw`/`_mtt` kernel family reads:
    /// data `[e][rt][ks][piece 2][row 64][16 B]`, scales
    /// `[e][rt][ks][row 64][4 B]`. Same bytes per logical element as
    /// [`Self::nvf4_moe_upload`] - both nemotron planes tile exactly
    /// (`ff % 64 == 0 && in_dim % 64 == 0` is the contract, checked here),
    /// so VRAM and DRAM traffic are unchanged; what changes is that every
    /// K-chunk fetch of a 64-row tile is one contiguous span (512 B-class
    /// warp loads - the marlin transaction class; the row-major pair reads
    /// 128 B rows at ~1.4 KB stride). Callers gate on
    /// [`Self::has_nvf4_moe_st`]: a tiled plane must never exist without
    /// every consumer class able to read it (the lm_head has_nvf4_tm law).
    pub fn nvf4_moe_upload_tiled(
        &self,
        packed: &[u8],
        scales: &[u8],
        scale2: &[f32],
        n_expert: usize,
        ff: usize,
        in_dim: usize,
    ) -> Result<Nvf4MoePlane, GpuError> {
        let rows = n_expert * ff;
        if !ff.is_multiple_of(64)
            || !in_dim.is_multiple_of(64)
            || scale2.len() != n_expert
            || packed.len() != rows * (in_dim / 2)
            || scales.len() != rows * (in_dim / 16)
        {
            return Err(GpuError::Unsupported(format!(
                "nvf4 moe tiled geometry mismatch: packed {} scales {} scale2 {} for {}x[{ff}, {in_dim}]",
                packed.len(),
                scales.len(),
                scale2.len(),
                n_expert
            )));
        }
        let nrt = ff / 64;
        let nks = in_dim / 64;
        let mut ptm = vec![0u8; packed.len()];
        let mut stm = vec![0u8; scales.len()];
        for e in 0..n_expert {
            for rt in 0..nrt {
                for r in 0..64usize {
                    let srcr = e * ff + rt * 64 + r;
                    let drow = &packed[srcr * (in_dim / 2)..][..in_dim / 2];
                    let srow = &scales[srcr * (in_dim / 16)..][..in_dim / 16];
                    for ks in 0..nks {
                        let blk = (e * nrt + rt) * nks + ks;
                        // two 16 B pieces per (row, ks), piece-major in the block
                        ptm[blk * 2048 + r * 16..][..16].copy_from_slice(&drow[ks * 32..][..16]);
                        ptm[blk * 2048 + 1024 + r * 16..][..16]
                            .copy_from_slice(&drow[ks * 32 + 16..][..16]);
                        stm[blk * 256 + r * 4..][..4].copy_from_slice(&srow[ks * 4..][..4]);
                    }
                }
            }
        }
        // stream-ordered upload - see nvf4_upload's note on the 2026-09-08
        // race (alloc_zeros + a raw cuMemcpyHtoD zeroed planes at random)
        let data = self.stream.clone_htod(&ptm).map_err(drv)?;
        let scale = self.stream.clone_htod(&stm).map_err(drv)?;
        let scale2 = self.stream.clone_htod(scale2).map_err(drv)?;
        Ok(Nvf4MoePlane {
            data,
            scale,
            row_scale: None,
            scale2,
            n_expert,
            ff,
            in_dim,
            layout: Nvf4MoeLayout::Tiled64,
        })
    }

    /// Loud layout mismatch for a MoE-plane consumer (the wrappers below all
    /// check; a kernel reading the wrong byte order would be silent garbage).
    pub(super) fn moe_layout_ok(
        w: &Nvf4MoePlane,
        want: Nvf4MoeLayout,
        op: &'static str,
    ) -> Result<(), GpuError> {
        if w.layout != want {
            return Err(GpuError::Unsupported(format!(
                "{op}: MoE plane layout {:?} but this kernel class reads {:?}",
                w.layout, want
            )));
        }
        Ok(())
    }

    /// Token-batched expert up GEMV + fused squared-relu: `idx` [batch*k]
    /// expert picks, `x` [batch, in_dim], `y` [batch*k, ff]. The shared
    /// expert rides k=1 with a constant-zero idx over its 1-expert plane.
    #[allow(clippy::too_many_arguments)]
    pub fn nvf4_moe_up_relu2(
        &self,
        w: &Nvf4MoePlane,
        idx: &CudaSlice<u32>,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        k: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .nvf4_moe_up_relu2
            .ok_or(GpuError::MissingOp("nvf4_moe_up_relu2"))?;
        Self::moe_layout_ok(w, Nvf4MoeLayout::Row, "nvf4_moe_up_relu2")?;
        debug_assert!(idx.len() >= batch * k);
        debug_assert!(x.len() >= batch * w.in_dim);
        debug_assert!(y.len() >= batch * k * w.ff);
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (sp, _g2) = w.scale.device_ptr(&self.stream);
        let (s2p, _g3) = w.scale2.device_ptr(&self.stream);
        let (ip, _g4) = idx.device_ptr(&self.stream);
        let (xp, _g5) = x.device_ptr(&self.stream);
        let (yp, _g6) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; sizes checked above
        check(unsafe {
            f(
                dp as *const _,
                sp as *const _,
                s2p as *const _,
                ip as *const _,
                xp as *const _,
                yp as *mut _,
                w.in_dim as u32,
                w.ff as u32,
                k as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Token-batched expert down GEMV + weighted combine (deterministic
    /// ascending-slot order): `xr` [batch*k, ff] the up outputs, `topk_w`
    /// [batch*k], `y` [batch, embd]. `accumulate` adds onto y (the
    /// shared-expert pass); the routed pass runs with `accumulate = false`.
    #[allow(clippy::too_many_arguments)]
    pub fn nvf4_moe_down_acc(
        &self,
        w: &Nvf4MoePlane,
        idx: &CudaSlice<u32>,
        topk_w: &CudaSlice<f32>,
        xr: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        part: Option<&mut CudaSlice<f32>>,
        k: usize,
        batch: usize,
        accumulate: bool,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .nvf4_moe_down_acc
            .ok_or(GpuError::MissingOp("nvf4_moe_down_acc"))?;
        let row_scales: &CudaSlice<u8> = match (&w.layout, &w.row_scale) {
            (Nvf4MoeLayout::Row, _) => &w.scale,
            (Nvf4MoeLayout::CutBlk, Some(rs)) => rs,
            _ => {
                Self::moe_layout_ok(w, Nvf4MoeLayout::Row, "nvf4_moe_down_acc")?;
                unreachable!()
            }
        };
        debug_assert!(idx.len() >= batch * k);
        debug_assert!(topk_w.len() >= batch * k);
        debug_assert!(xr.len() >= batch * k * w.in_dim);
        debug_assert!(y.len() >= batch * w.ff);
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (sp, _g2) = row_scales.device_ptr(&self.stream);
        let (s2p, _g3) = w.scale2.device_ptr(&self.stream);
        let (ip, _g4) = idx.device_ptr(&self.stream);
        let (wp, _g5) = topk_w.device_ptr(&self.stream);
        let (xp, _g6) = xr.device_ptr(&self.stream);
        let (yp, _g7) = y.device_ptr_mut(&self.stream);
        let part_guard = part.map(|m| m.device_ptr_mut(&self.stream));
        let ptp = match &part_guard {
            Some((p, _)) => *p as *mut core::ffi::c_void,
            None => std::ptr::null_mut(),
        };
        // SAFETY: ABI contract; sizes checked above. For a down plane the
        // struct's `ff` is the out dim (embd) and `in_dim` is the expert ff.
        check(unsafe {
            f(
                dp as *const _,
                sp as *const _,
                s2p as *const _,
                ip as *const _,
                wp as *const _,
                xp as *const _,
                yp as *mut _,
                ptp,
                w.in_dim as u32,
                w.ff as u32,
                k as u32,
                batch as u32,
                accumulate as u32,
                self.stream_ptr(),
            )
        })
    }

    // ---- decode multi-task NVFP4 MoE expert GEMVs (decode rung) --

    /// The fused decode MoE chain: the two wave-dense multi-task GEMVs plus
    /// the deterministic slot fold. Absent, decode stays on the 4-launch
    /// GEMV pair above.
    pub fn has_nvf4_moe_mt(&self) -> bool {
        self.kernels.nvf4_moe_up_relu2_mt.is_some()
            && self.kernels.nvf4_moe_down_part.is_some()
            && self.kernels.moe_slot_combine.is_some()
    }

    /// Fused decode expert up + squared-relu: one launch covering all `k`
    /// routed slots of `up` AND the shared expert plane `sh_up`, writing
    /// `act = [k*up.ff | sh_up.ff]` - row-for-row the same values (and
    /// layout) the pair path writes into its two buffers.
    pub fn nvf4_moe_up_relu2_mt(
        &self,
        up: &Nvf4MoePlane,
        sh_up: &Nvf4MoePlane,
        idx: &CudaSlice<u32>,
        x: &CudaSlice<f32>,
        act: &mut CudaSlice<f32>,
        k: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .nvf4_moe_up_relu2_mt
            .ok_or(GpuError::MissingOp("nvf4_moe_up_relu2_mt"))?;
        Self::moe_layout_ok(up, Nvf4MoeLayout::Row, "nvf4_moe_up_relu2_mt")?;
        Self::moe_layout_ok(sh_up, Nvf4MoeLayout::Row, "nvf4_moe_up_relu2_mt")?;
        debug_assert_eq!(up.in_dim, sh_up.in_dim);
        debug_assert!(idx.len() >= k);
        debug_assert!(x.len() >= up.in_dim);
        debug_assert!(act.len() >= k * up.ff + sh_up.ff);
        let (rdp, _g1) = up.data.device_ptr(&self.stream);
        let (rsp, _g2) = up.scale.device_ptr(&self.stream);
        let (rs2, _g3) = up.scale2.device_ptr(&self.stream);
        let (sdp, _g4) = sh_up.data.device_ptr(&self.stream);
        let (ssp, _g5) = sh_up.scale.device_ptr(&self.stream);
        let (ss2, _g6) = sh_up.scale2.device_ptr(&self.stream);
        let (ip, _g7) = idx.device_ptr(&self.stream);
        let (xp, _g8) = x.device_ptr(&self.stream);
        let (ap, _g9) = act.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; sizes checked above
        check(unsafe {
            f(
                rdp as *const _,
                rsp as *const _,
                rs2 as *const _,
                sdp as *const _,
                ssp as *const _,
                ss2 as *const _,
                ip as *const _,
                xp as *const _,
                ap as *mut _,
                up.in_dim as u32,
                up.ff as u32,
                sh_up.ff as u32,
                k as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Fused decode expert down -> pre-weighted per-slot partials at
    /// `part[slot*embd + r]` (shared expert = slot `k`, weight 1.0 like the
    /// pair path's shared pass); fold with [`Self::moe_slot_combine`] at
    /// `np = k + 1`, `rows = 1`. For down planes `ff` is the out dim (embd)
    /// and `in_dim` the expert ff, same convention as the pair.
    #[allow(clippy::too_many_arguments)]
    pub fn nvf4_moe_down_part(
        &self,
        down: &Nvf4MoePlane,
        sh_down: &Nvf4MoePlane,
        idx: &CudaSlice<u32>,
        topk_w: &CudaSlice<f32>,
        act: &CudaSlice<f32>,
        part: &mut CudaSlice<f32>,
        k: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .nvf4_moe_down_part
            .ok_or(GpuError::MissingOp("nvf4_moe_down_part"))?;
        Self::moe_layout_ok(down, Nvf4MoeLayout::Row, "nvf4_moe_down_part")?;
        Self::moe_layout_ok(sh_down, Nvf4MoeLayout::Row, "nvf4_moe_down_part")?;
        debug_assert_eq!(down.ff, sh_down.ff);
        debug_assert!(idx.len() >= k);
        debug_assert!(topk_w.len() >= k);
        debug_assert!(act.len() >= k * down.in_dim + sh_down.in_dim);
        debug_assert!(part.len() >= (k + 1) * down.ff);
        let (rdp, _g1) = down.data.device_ptr(&self.stream);
        let (rsp, _g2) = down.scale.device_ptr(&self.stream);
        let (rs2, _g3) = down.scale2.device_ptr(&self.stream);
        let (sdp, _g4) = sh_down.data.device_ptr(&self.stream);
        let (ssp, _g5) = sh_down.scale.device_ptr(&self.stream);
        let (ss2, _g6) = sh_down.scale2.device_ptr(&self.stream);
        let (ip, _g7) = idx.device_ptr(&self.stream);
        let (wp, _g8) = topk_w.device_ptr(&self.stream);
        let (ap, _g9) = act.device_ptr(&self.stream);
        let (pp, _g10) = part.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; sizes checked above
        check(unsafe {
            f(
                rdp as *const _,
                rsp as *const _,
                rs2 as *const _,
                sdp as *const _,
                ssp as *const _,
                ss2 as *const _,
                ip as *const _,
                wp as *const _,
                ap as *const _,
                pp as *mut _,
                down.in_dim as u32,
                sh_down.in_dim as u32,
                down.ff as u32,
                k as u32,
                self.stream_ptr(),
            )
        })
    }

    // ---- sorted-tile NVFP4 MoE expert GEMMs  -------------

    /// The whole prefill MoE MMA chain: quantize + align + the two sorted
    /// GEMM tiles + the deterministic fold. Absent (non-cc12) the arm falls
    /// back to the GEMV pair above.
    pub fn has_nvf4_moe_bs(&self) -> bool {
        self.kernels.nvf4_moe_up_relu2_bs.is_some()
            && self.kernels.nvf4_moe_down_bs.is_some()
            && self.kernels.quantize_nvf4.is_some()
            && self.kernels.moe_align.is_some()
            && self.kernels.moe_slot_combine.is_some()
    }

    /// Sorted-tile expert up + relu^2 over the moe_align BM=32 layout:
    /// `xq`/`xs` are the token nvf4 activation planes (gathered by
    /// `sorted_row`), `fq`/`fs` the sorted-position nvf4 output planes
    /// ([nb*32, ff/2] + [nb*32, ff/16]) - the down kernel's direct input.
    #[allow(clippy::too_many_arguments)]
    pub fn nvf4_moe_up_relu2_bs(
        &self,
        w: &Nvf4MoePlane,
        sorted_row: &CudaSlice<u32>,
        block_expert: &CudaSlice<u32>,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<u8>,
        fq: &mut CudaSlice<u8>,
        fs: &mut CudaSlice<u8>,
        nb: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .nvf4_moe_up_relu2_bs
            .ok_or(GpuError::MissingOp("nvf4_moe_up_relu2_bs"))?;
        Self::moe_layout_ok(w, Nvf4MoeLayout::Row, "nvf4_moe_up_relu2_bs")?;
        debug_assert!(sorted_row.len() >= nb * 32);
        debug_assert!(block_expert.len() >= nb);
        debug_assert!(fq.len() >= nb * 32 * (w.ff / 2));
        debug_assert!(fs.len() >= nb * 32 * (w.ff / 16));
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (sp, _g2) = w.scale.device_ptr(&self.stream);
        let (s2p, _g3) = w.scale2.device_ptr(&self.stream);
        let (rp, _g4) = sorted_row.device_ptr(&self.stream);
        let (bp, _g5) = block_expert.device_ptr(&self.stream);
        let (xqp, _g6) = xq.device_ptr(&self.stream);
        let (xsp, _g7) = xs.device_ptr(&self.stream);
        let (fqp, _g8) = fq.device_ptr_mut(&self.stream);
        let (fsp, _g9) = fs.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; sizes checked above
        check(unsafe {
            f(
                dp as *const _,
                sp as *const _,
                s2p as *const _,
                rp as *const _,
                bp as *const _,
                xqp as *const _,
                xsp as *const _,
                fqp as *mut _,
                fsp as *mut _,
                w.in_dim as u32,
                w.ff as u32,
                nb as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Sorted-tile expert down -> weighted per-(token, slot) f32 partials at
    /// `part[(tok*np + slt + slot_off) * embd]`; fold with
    /// [`Self::moe_slot_combine`] (fixed slot order). `topk_w` None means
    /// 1.0 (the shared-expert pass); `kw` is topk_w's row stride. For a down
    /// plane `w.ff` is the out dim (embd) and `w.in_dim` the expert ff.
    #[allow(clippy::too_many_arguments)]
    pub fn nvf4_moe_down_bs(
        &self,
        w: &Nvf4MoePlane,
        sorted_row: &CudaSlice<u32>,
        sorted_slot: &CudaSlice<u32>,
        block_expert: &CudaSlice<u32>,
        topk_w: Option<&CudaSlice<f32>>,
        fq: &CudaSlice<u8>,
        fs: &CudaSlice<u8>,
        part: &mut CudaSlice<f32>,
        kw: usize,
        np: usize,
        slot_off: usize,
        nb: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .nvf4_moe_down_bs
            .ok_or(GpuError::MissingOp("nvf4_moe_down_bs"))?;
        Self::moe_layout_ok(w, Nvf4MoeLayout::Row, "nvf4_moe_down_bs")?;
        debug_assert!(sorted_row.len() >= nb * 32);
        debug_assert!(sorted_slot.len() >= nb * 32);
        debug_assert!(block_expert.len() >= nb);
        debug_assert!(fq.len() >= nb * 32 * (w.in_dim / 2));
        debug_assert!(fs.len() >= nb * 32 * (w.in_dim / 16));
        debug_assert!(slot_off < np);
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (sp, _g2) = w.scale.device_ptr(&self.stream);
        let (s2p, _g3) = w.scale2.device_ptr(&self.stream);
        let (rp, _g4) = sorted_row.device_ptr(&self.stream);
        let (slp, _g5) = sorted_slot.device_ptr(&self.stream);
        let (bp, _g6) = block_expert.device_ptr(&self.stream);
        let wp = match topk_w {
            Some(w) => {
                let (p, _g) = w.device_ptr(&self.stream);
                p as *const core::ffi::c_void
            }
            None => core::ptr::null(),
        };
        let (fqp, _g7) = fq.device_ptr(&self.stream);
        let (fsp, _g8) = fs.device_ptr(&self.stream);
        let (pp, _g9) = part.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; sizes checked above
        check(unsafe {
            f(
                dp as *const _,
                sp as *const _,
                s2p as *const _,
                rp as *const _,
                slp as *const _,
                bp as *const _,
                wp,
                fqp as *const _,
                fsp as *const _,
                pp as *mut _,
                w.in_dim as u32,
                w.ff as u32,
                kw as u32,
                np as u32,
                slot_off as u32,
                nb as u32,
                self.stream_ptr(),
            )
        })
    }

    // ---- tiled-layout MoE consumers (slots 472-477) -----

    /// The full tiled-layout consumer set: skinny decode pair (`_st`, BM=8),
    /// wide prefill pair (`_stw`, BM=32), r=1 mt-class twins (`_mtt`), plus
    /// the BM-parameterized align. A tiled plane must never be uploaded
    /// unless this is true - every consumer class has to be able to read it
    /// (the lm_head `has_nvf4_tm` law). cc12-only in the pack.
    pub fn has_nvf4_moe_st(&self) -> bool {
        self.kernels.nvf4_moe_up_relu2_st.is_some()
            && self.kernels.nvf4_moe_down_st.is_some()
            && self.kernels.nvf4_moe_up_relu2_stw.is_some()
            && self.kernels.nvf4_moe_down_stw.is_some()
            && self.kernels.nvf4_moe_up_relu2_mtt.is_some()
            && self.kernels.nvf4_moe_down_part_tt.is_some()
            && self.kernels.moe_align_bm.is_some()
    }

    /// Skinny (BM=8) sorted-tile up over a TILED plane - the decode twin of
    /// [`Self::nvf4_moe_up_relu2_bs`]. Blocks come from `moe_align_bm(bm=8)`;
    /// `sorted_row` is `[nb*8]` and fq/fs rows are `nb*8`. Bit-exact vs the
    /// bs pair on identical routing (same kt/k64 accumulate order); `bm`
    /// selects the wide (32) or skinny (8) instantiation.
    #[allow(clippy::too_many_arguments)]
    pub fn nvf4_moe_up_relu2_st(
        &self,
        w: &Nvf4MoePlane,
        sorted_row: &CudaSlice<u32>,
        block_expert: &CudaSlice<u32>,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<u8>,
        fq: &mut CudaSlice<u8>,
        fs: &mut CudaSlice<u8>,
        nb: usize,
        bm: usize,
    ) -> Result<(), GpuError> {
        let f = match bm {
            8 => self
                .kernels
                .nvf4_moe_up_relu2_st
                .ok_or(GpuError::MissingOp("nvf4_moe_up_relu2_st"))?,
            32 => self
                .kernels
                .nvf4_moe_up_relu2_stw
                .ok_or(GpuError::MissingOp("nvf4_moe_up_relu2_stw"))?,
            _ => {
                return Err(GpuError::Unsupported(format!(
                    "nvf4_moe_up_relu2_st: bm {bm}"
                )));
            }
        };
        Self::moe_layout_ok(w, Nvf4MoeLayout::Tiled64, "nvf4_moe_up_relu2_st")?;
        debug_assert!(sorted_row.len() >= nb * bm);
        debug_assert!(block_expert.len() >= nb);
        debug_assert!(fq.len() >= nb * bm * (w.ff / 2));
        debug_assert!(fs.len() >= nb * bm * (w.ff / 16));
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (sp, _g2) = w.scale.device_ptr(&self.stream);
        let (s2p, _g3) = w.scale2.device_ptr(&self.stream);
        let (rp, _g4) = sorted_row.device_ptr(&self.stream);
        let (bp, _g5) = block_expert.device_ptr(&self.stream);
        let (xqp, _g6) = xq.device_ptr(&self.stream);
        let (xsp, _g7) = xs.device_ptr(&self.stream);
        let (fqp, _g8) = fq.device_ptr_mut(&self.stream);
        let (fsp, _g9) = fs.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; sizes checked above
        check(unsafe {
            f(
                dp as *const _,
                sp as *const _,
                s2p as *const _,
                rp as *const _,
                bp as *const _,
                xqp as *const _,
                xsp as *const _,
                fqp as *mut _,
                fsp as *mut _,
                w.in_dim as u32,
                w.ff as u32,
                nb as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Skinny/wide sorted-tile down over a TILED plane - the twin of
    /// [`Self::nvf4_moe_down_bs`] at the given `bm` block width.
    #[allow(clippy::too_many_arguments)]
    pub fn nvf4_moe_down_st(
        &self,
        w: &Nvf4MoePlane,
        sorted_row: &CudaSlice<u32>,
        sorted_slot: &CudaSlice<u32>,
        block_expert: &CudaSlice<u32>,
        topk_w: Option<&CudaSlice<f32>>,
        fq: &CudaSlice<u8>,
        fs: &CudaSlice<u8>,
        part: &mut CudaSlice<f32>,
        kw: usize,
        np: usize,
        slot_off: usize,
        nb: usize,
        bm: usize,
    ) -> Result<(), GpuError> {
        let f = match bm {
            8 => self
                .kernels
                .nvf4_moe_down_st
                .ok_or(GpuError::MissingOp("nvf4_moe_down_st"))?,
            32 => self
                .kernels
                .nvf4_moe_down_stw
                .ok_or(GpuError::MissingOp("nvf4_moe_down_stw"))?,
            _ => return Err(GpuError::Unsupported(format!("nvf4_moe_down_st: bm {bm}"))),
        };
        Self::moe_layout_ok(w, Nvf4MoeLayout::Tiled64, "nvf4_moe_down_st")?;
        debug_assert!(sorted_row.len() >= nb * bm);
        debug_assert!(sorted_slot.len() >= nb * bm);
        debug_assert!(block_expert.len() >= nb);
        debug_assert!(fq.len() >= nb * bm * (w.in_dim / 2));
        debug_assert!(fs.len() >= nb * bm * (w.in_dim / 16));
        debug_assert!(slot_off < np);
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (sp, _g2) = w.scale.device_ptr(&self.stream);
        let (s2p, _g3) = w.scale2.device_ptr(&self.stream);
        let (rp, _g4) = sorted_row.device_ptr(&self.stream);
        let (slp, _g5) = sorted_slot.device_ptr(&self.stream);
        let (bp, _g6) = block_expert.device_ptr(&self.stream);
        let wp = match topk_w {
            Some(w) => {
                let (p, _g) = w.device_ptr(&self.stream);
                p as *const core::ffi::c_void
            }
            None => core::ptr::null(),
        };
        let (fqp, _g7) = fq.device_ptr(&self.stream);
        let (fsp, _g8) = fs.device_ptr(&self.stream);
        let (pp, _g9) = part.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; sizes checked above
        check(unsafe {
            f(
                dp as *const _,
                sp as *const _,
                s2p as *const _,
                rp as *const _,
                slp as *const _,
                bp as *const _,
                wp,
                fqp as *const _,
                fsp as *const _,
                pp as *mut _,
                w.in_dim as u32,
                w.ff as u32,
                kw as u32,
                np as u32,
                slot_off as u32,
                nb as u32,
                self.stream_ptr(),
            )
        })
    }

    /// r=1 fused decode up over TILED planes - same numeric class and
    /// argument contract as [`Self::nvf4_moe_up_relu2_mt`], regrouped for the
    /// tiled layout (16-row-group CTAs; gates are rel-to-rms + determinism).
    pub fn nvf4_moe_up_relu2_mtt(
        &self,
        up: &Nvf4MoePlane,
        sh_up: &Nvf4MoePlane,
        idx: &CudaSlice<u32>,
        x: &CudaSlice<f32>,
        act: &mut CudaSlice<f32>,
        k: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .nvf4_moe_up_relu2_mtt
            .ok_or(GpuError::MissingOp("nvf4_moe_up_relu2_mtt"))?;
        Self::moe_layout_ok(up, Nvf4MoeLayout::Tiled64, "nvf4_moe_up_relu2_mtt")?;
        Self::moe_layout_ok(sh_up, Nvf4MoeLayout::Tiled64, "nvf4_moe_up_relu2_mtt")?;
        debug_assert_eq!(up.in_dim, sh_up.in_dim);
        debug_assert!(idx.len() >= k);
        debug_assert!(x.len() >= up.in_dim);
        debug_assert!(act.len() >= k * up.ff + sh_up.ff);
        let (rdp, _g1) = up.data.device_ptr(&self.stream);
        let (rsp, _g2) = up.scale.device_ptr(&self.stream);
        let (rs2, _g3) = up.scale2.device_ptr(&self.stream);
        let (sdp, _g4) = sh_up.data.device_ptr(&self.stream);
        let (ssp, _g5) = sh_up.scale.device_ptr(&self.stream);
        let (ss2, _g6) = sh_up.scale2.device_ptr(&self.stream);
        let (ip, _g7) = idx.device_ptr(&self.stream);
        let (xp, _g8) = x.device_ptr(&self.stream);
        let (ap, _g9) = act.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; sizes checked above
        check(unsafe {
            f(
                rdp as *const _,
                rsp as *const _,
                rs2 as *const _,
                sdp as *const _,
                ssp as *const _,
                ss2 as *const _,
                ip as *const _,
                xp as *const _,
                ap as *mut _,
                up.in_dim as u32,
                up.ff as u32,
                sh_up.ff as u32,
                k as u32,
                self.stream_ptr(),
            )
        })
    }

    /// r=1 fused decode down over TILED planes - the
    /// [`Self::nvf4_moe_down_part`] twin, same contract.
    #[allow(clippy::too_many_arguments)]
    pub fn nvf4_moe_down_part_tt(
        &self,
        down: &Nvf4MoePlane,
        sh_down: &Nvf4MoePlane,
        idx: &CudaSlice<u32>,
        topk_w: &CudaSlice<f32>,
        act: &CudaSlice<f32>,
        part: &mut CudaSlice<f32>,
        k: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .nvf4_moe_down_part_tt
            .ok_or(GpuError::MissingOp("nvf4_moe_down_part_tt"))?;
        Self::moe_layout_ok(down, Nvf4MoeLayout::Tiled64, "nvf4_moe_down_part_tt")?;
        Self::moe_layout_ok(sh_down, Nvf4MoeLayout::Tiled64, "nvf4_moe_down_part_tt")?;
        debug_assert_eq!(down.ff, sh_down.ff);
        debug_assert!(idx.len() >= k);
        debug_assert!(topk_w.len() >= k);
        debug_assert!(act.len() >= k * down.in_dim + sh_down.in_dim);
        debug_assert!(part.len() >= (k + 1) * down.ff);
        let (rdp, _g1) = down.data.device_ptr(&self.stream);
        let (rsp, _g2) = down.scale.device_ptr(&self.stream);
        let (rs2, _g3) = down.scale2.device_ptr(&self.stream);
        let (sdp, _g4) = sh_down.data.device_ptr(&self.stream);
        let (ssp, _g5) = sh_down.scale.device_ptr(&self.stream);
        let (ss2, _g6) = sh_down.scale2.device_ptr(&self.stream);
        let (ip, _g7) = idx.device_ptr(&self.stream);
        let (wp, _g8) = topk_w.device_ptr(&self.stream);
        let (ap, _g9) = act.device_ptr(&self.stream);
        let (pp, _g10) = part.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; sizes checked above
        check(unsafe {
            f(
                rdp as *const _,
                rsp as *const _,
                rs2 as *const _,
                sdp as *const _,
                ssp as *const _,
                ss2 as *const _,
                ip as *const _,
                wp as *const _,
                ap as *const _,
                pp as *mut _,
                down.in_dim as u32,
                sh_down.in_dim as u32,
                down.ff as u32,
                k as u32,
                self.stream_ptr(),
            )
        })
    }
}
