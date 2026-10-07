//! BF16 dense weight planes - per-TENSOR quant dispatch.
//!
//! UD quant files are MIXED: muse-glimmer's UD-Q8_K_XL keeps `token_embd`,
//! `output`, `attn_k` and `attn_v` at bf16 while everything else is Q8_0,
//! because the quantizer judged those planes worth the bytes. The project's rule
//! for that is per-tensor dispatch rather than a per-model switch, and the
//! correctness spine is same-weights parity against llama.cpp on the identical
//! file - so down-quantizing a bf16 plane into the Q8_0 lane at load is out on
//! both counts (different weights, no exact greedy target, and a deliberate
//! quality choice silently overridden).
//!
//! A bf16 plane is carried by the plain [`QuantTensor`] the raw uploader
//! already produces (`bytes` + `ty` + `dims`) - no new plane struct, and
//! `dims` keeps the house convention `[in_dim, out_dim]`.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use super::error::check;
use super::{GpuError, GpuExecutor, QuantTensor};
use paddock_models::ggml_type::GgmlType;

impl GpuExecutor {
    /// True when the pack carries the bf16 dense lane. Every consumer checks
    /// this before electing a bf16 plane, so an older pack degrades to a loud
    /// `MissingOp` at load rather than a wrong answer at serve.
    pub fn has_bf16_dense(&self) -> bool {
        self.kernels.bf16_gemv_f32.is_some()
            && self.kernels.bf16_gemm_f32.is_some()
            && self.kernels.bf16_dequant_f32.is_some()
    }

    /// True when the pack can gather rows out of a bf16 embedding table -
    /// loaders elect the bf16-resident table only behind this, so an older
    /// pack falls back to the widened f32 table instead of erroring.
    pub fn has_embed_gather_bf16(&self) -> bool {
        self.kernels.embed_gather_bf16.is_some()
    }

    /// Guard for the plane class itself: a `QuantTensor` reaching these
    /// entry points must actually hold bf16 bytes. The loader decides the
    /// class from the file, so a mismatch here is a routing bug, not user
    /// input - but it would read garbage weights silently, so name it.
    pub(super) fn bf16_plane<'a>(
        &self,
        w: &'a QuantTensor,
        who: &str,
    ) -> Result<&'a QuantTensor, GpuError> {
        if w.ty != GgmlType::Bf16 {
            return Err(GpuError::NoKernel {
                name: format!("{who}: plane is not bf16"),
                ty: w.ty,
            });
        }
        Ok(w)
    }

    /// `y = W x` over a bf16 weight plane - the r==1 twin of
    /// `q8_0_gemv_repacked`, same operand convention (`dims[0]` = in_dim).
    pub fn bf16_gemv(
        &self,
        w: &QuantTensor,
        bias: Option<&CudaSlice<f32>>,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
    ) -> Result<(), GpuError> {
        let w = self.bf16_plane(w, "bf16_gemv")?;
        let f = self
            .kernels
            .bf16_gemv_f32
            .ok_or(GpuError::MissingOp("bf16_gemv_f32"))?;
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        let (wp, _g1) = w.bytes.device_ptr(&self.stream);
        let (xp, _g2) = x.device_ptr(&self.stream);
        let (yp, _g3) = y.device_ptr_mut(&self.stream);
        let (bias_ptr, _gb);
        let bp: *const core::ffi::c_void = match bias {
            Some(b) => {
                (bias_ptr, _gb) = b.device_ptr(&self.stream);
                bias_ptr as *const _
            }
            None => core::ptr::null(),
        };
        // SAFETY: ABI contract; shapes come from the plane's own dims
        check(unsafe {
            f(
                wp as *const _,
                bp,
                xp as *const _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    /// `bf16_gemv` over a row SEGMENT of a fused plane: rows
    /// `[first_row, first_row + out_dim)` of `w` (house dims `[in, out]`).
    /// The nemotron attn twins live as one load-time-concatenated `[q;k;v]`
    /// plane (thin-k/v rung), and the serial decode row still
    /// wants three per-projection GEMVs - this is that read, a plain base
    /// pointer offset (row-major planes make a row range a byte range).
    pub fn bf16_gemv_rows(
        &self,
        w: &QuantTensor,
        first_row: usize,
        out_dim: usize,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
    ) -> Result<(), GpuError> {
        let w = self.bf16_plane(w, "bf16_gemv_rows")?;
        let f = self
            .kernels
            .bf16_gemv_f32
            .ok_or(GpuError::MissingOp("bf16_gemv_f32"))?;
        let in_dim = w.dims[0];
        debug_assert!(first_row + out_dim <= w.dims[1], "segment exceeds plane");
        let (wp, _g1) = w.bytes.device_ptr(&self.stream);
        let (xp, _g2) = x.device_ptr(&self.stream);
        let (yp, _g3) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; the segment is a contiguous byte range of
        // the plane (2 bytes per bf16 element, in_dim elements per row)
        check(unsafe {
            f(
                (wp + (first_row * in_dim * 2) as u64) as *const _,
                core::ptr::null(),
                xp as *const _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    /// `bf16_gemv` with a fused `silu(v * inv)` epilogue over the first
    /// `silu_rows` output rows (slot 520). Bit-identical to the GEMV followed
    /// by a separate scale+silu pass: same dot, same two f32 ops on the same
    /// value. `Ok(false)` when the pack predates the slot.
    pub fn bf16_gemv_silu(
        &self,
        w: &QuantTensor,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        mirror: Option<&mut CudaSlice<half::bf16>>,
        silu_rows: usize,
        inv: f32,
    ) -> Result<bool, GpuError> {
        let Some(f) = self.kernels.bf16_gemv_silu_f32 else {
            return Ok(false);
        };
        let w = self.bf16_plane(w, "bf16_gemv_silu")?;
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        let (wp, _g1) = w.bytes.device_ptr(&self.stream);
        let (xp, _g2) = x.device_ptr(&self.stream);
        let (yp, _g3) = y.device_ptr_mut(&self.stream);
        let m_guard = mirror.map(|m| m.device_ptr_mut(&self.stream));
        let mp = match &m_guard {
            Some((p, _)) => *p as *mut core::ffi::c_void,
            None => std::ptr::null_mut(),
        };
        // SAFETY: ABI contract; shapes come from the plane's own dims
        check(unsafe {
            f(
                wp as *const _,
                core::ptr::null(),
                xp as *const _,
                yp as *mut _,
                mp,
                in_dim as u32,
                out_dim as u32,
                silu_rows as u32,
                inv,
                self.stream_ptr(),
            )
        })?;
        Ok(true)
    }

    /// Multi-row narrow-K GEMV (slot 522) - the batch > 1 arm of
    /// `bf16_gemv_nk`. `x` is `[batch, in_dim]`, `y` is `[batch, out_dim]`.
    /// `Ok(false)` when the pack predates the slot.
    pub fn bf16_gemv_nk_mr(
        &self,
        w: &QuantTensor,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<bool, GpuError> {
        let Some(f) = self.kernels.bf16_gemv_nk_mr_f32 else {
            return Ok(false);
        };
        let w = self.bf16_plane(w, "bf16_gemv_nk_mr")?;
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        let (wp, _g1) = w.bytes.device_ptr(&self.stream);
        let (xp, _g2) = x.device_ptr(&self.stream);
        let (yp, _g3) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; shapes come from the plane's own dims
        check(unsafe {
            f(
                wp as *const _,
                core::ptr::null(),
                xp as *const _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })?;
        Ok(true)
    }

    /// Narrow-K twin of `bf16_gemv` (slot 518): one warp per output row, 8
    /// rows per block. Same contract and same f32 product class; the
    /// reduction is a 32-lane shuffle instead of the 128-thread two-level
    /// tree, so it is a separate slot and is elected by SHAPE at the call
    /// site rather than swapped in under `bf16_gemv`.
    ///
    /// Returns `Ok(false)` when the pack predates the slot, so a caller can
    /// fall back without the plane's shape deciding whether the model runs.
    /// slot 562: the HC `up` plane with `hc_mix` fused into its epilogue -
    /// one warp owns the `hc` gate rows of a hidden index, so the gate plane
    /// never round-trips. `Ok(false)` = not taken (caller runs the pair).
    #[allow(clippy::too_many_arguments)]
    pub fn bf16_gemv_up_hcmix(
        &self,
        w: &QuantTensor,
        x: &CudaSlice<f32>,
        xn: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        out16: Option<&mut CudaSlice<half::bf16>>,
        hidden: usize,
        hc: usize,
    ) -> Result<bool, GpuError> {
        let Some(f) = self.kernels.bf16_gemv_up_hcmix else {
            return Ok(false);
        };
        let w = self.bf16_plane(w, "bf16_gemv_up_hcmix")?;
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        if hc != 4 || out_dim != hc * hidden {
            return Ok(false);
        }
        let (wp, _g1) = w.bytes.device_ptr(&self.stream);
        let (xp, _g2) = x.device_ptr(&self.stream);
        let (np, _g3) = xn.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        let mp = match out16 {
            Some(m) => {
                let (p, _g) = m.device_ptr_mut(&self.stream);
                p as *mut core::ffi::c_void
            }
            None => core::ptr::null_mut(),
        };
        // SAFETY: ABI contract (slot 562); shapes come from the plane's dims
        check(unsafe {
            f(
                wp as *const _,
                xp as *const _,
                np as *const _,
                op as *mut _,
                mp,
                in_dim as u32,
                hidden as u32,
                hc as u32,
                self.stream_ptr(),
            )
        })?;
        Ok(true)
    }

    /// slot 573: pad a bf16 plane's rows with zeros (weight prep).
    pub fn bf16_pad_rows(
        &self,
        src: &CudaSlice<u8>,
        dst: &mut CudaSlice<u8>,
        rows_src: usize,
        rows_dst: usize,
        cols: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .bf16_pad_rows
            .ok_or(GpuError::MissingOp("bf16_pad_rows"))?;
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (dp, _g2) = dst.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 573)
        check(unsafe {
            f(
                sp as *const _,
                dp as *mut _,
                rows_src as u32,
                rows_dst as u32,
                cols as u32,
                self.stream_ptr(),
            )
        })
    }

    /// slot 574: up plane in the gate epilogue's row order, padded.
    #[allow(clippy::too_many_arguments)]
    pub fn bf16_hc_perm_pad(
        &self,
        src: &CudaSlice<u8>,
        dst: &mut CudaSlice<u8>,
        hidden: usize,
        hc: usize,
        lr: usize,
        kpad: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .bf16_hc_perm_pad
            .ok_or(GpuError::MissingOp("bf16_hc_perm_pad"))?;
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (dp, _g2) = dst.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 574)
        check(unsafe {
            f(
                sp as *const _,
                dp as *mut _,
                hidden as u32,
                hc as u32,
                lr as u32,
                kpad as u32,
                self.stream_ptr(),
            )
        })
    }

    /// slot 568: bf16 -> f32 cast.
    pub fn convert_bf16_f32(
        &self,
        src: &CudaSlice<half::bf16>,
        dst: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .convert_bf16_f32
            .ok_or(GpuError::MissingOp("convert_bf16_f32"))?;
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (dp, _g2) = dst.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 568)
        check(unsafe { f(sp as *const _, dp as *mut _, n as u64, self.stream_ptr()) })
    }

    /// slot 569: strided bf16 -> f32, unpadding a padded-N plane.
    pub fn convert_bf16_f32_rows(
        &self,
        src: &CudaSlice<half::bf16>,
        dst: &mut CudaSlice<f32>,
        rows: usize,
        cols: usize,
        src_rs: usize,
        dst_rs: usize,
    ) -> Result<bool, GpuError> {
        let Some(f) = self.kernels.convert_bf16_f32_rows else {
            return Ok(false);
        };
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (dp, _g2) = dst.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 569)
        check(unsafe {
            f(
                sp as *const _,
                dp as *mut _,
                rows as u32,
                cols as u32,
                src_rs as u32,
                dst_rs as u32,
                self.stream_ptr(),
            )
        })?;
        Ok(true)
    }

    /// [`Self::convert_bf16_f32_rows`] starting at a source COLUMN offset -
    /// the second segment of a padded two-segment gemm output.
    #[allow(clippy::too_many_arguments)]
    pub fn convert_bf16_f32_rows_at(
        &self,
        src: &CudaSlice<half::bf16>,
        src_col0: usize,
        dst: &mut CudaSlice<f32>,
        rows: usize,
        cols: usize,
        src_rs: usize,
        dst_rs: usize,
    ) -> Result<bool, GpuError> {
        let Some(f) = self.kernels.convert_bf16_f32_rows else {
            return Ok(false);
        };
        let (sp0, _g1) = src.device_ptr(&self.stream);
        let (dp, _g2) = dst.device_ptr_mut(&self.stream);
        let sp = sp0 + (src_col0 * 2) as u64;
        // SAFETY: ABI contract (slot 569); offset stays inside `src` by
        // construction (src_col0 + cols <= src_rs).
        check(unsafe {
            f(
                sp as *const _,
                dp as *mut _,
                rows as u32,
                cols as u32,
                src_rs as u32,
                dst_rs as u32,
                self.stream_ptr(),
            )
        })?;
        Ok(true)
    }

    /// slot 565: batched block-per-row gemv. One block per output row with
    /// BT accumulators, so the weight row is read once for the whole batch -
    /// the narrow-output planes that the tiled MMA arm starves (the hc down
    /// plane tiles to ELEVEN CTAs at batch 8). `ya`/`yb` are the two output
    /// segments (pass `yb = None` for a single [batch, out_dim] plane).
    #[allow(clippy::too_many_arguments)]
    pub fn bf16_gemv_mrow(
        &self,
        w: &QuantTensor,
        x: &CudaSlice<f32>,
        ya: &mut CudaSlice<f32>,
        yb: Option<&mut CudaSlice<f32>>,
        y16: Option<&mut CudaSlice<half::bf16>>,
        batch: usize,
        split: usize,
        silu_rows: usize,
        inv: f32,
    ) -> Result<bool, GpuError> {
        let Some(f) = self.kernels.bf16_gemv_mrow_f32 else {
            return Ok(false);
        };
        let w = self.bf16_plane(w, "bf16_gemv_mrow")?;
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        if batch == 0 || batch > 8 {
            return Ok(false);
        }
        let (wp, _g1) = w.bytes.device_ptr(&self.stream);
        let (xp, _g2) = x.device_ptr(&self.stream);
        let (ap, _g3) = ya.device_ptr_mut(&self.stream);
        let bp = match yb {
            Some(b) => {
                let (p, _g) = b.device_ptr_mut(&self.stream);
                p as *mut core::ffi::c_void
            }
            None => core::ptr::null_mut(),
        };
        let mp = match y16 {
            Some(m) => {
                let (p, _g) = m.device_ptr_mut(&self.stream);
                p as *mut core::ffi::c_void
            }
            None => core::ptr::null_mut(),
        };
        // SAFETY: ABI contract (slot 565); the pack declines batch > 8 with -1
        let rc = unsafe {
            f(
                wp as *const _,
                core::ptr::null(),
                xp as *const _,
                ap as *mut _,
                mp,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                silu_rows as u32,
                inv,
                bp,
                split as u32,
                self.stream_ptr(),
            )
        };
        if rc == -1 {
            return Ok(false);
        }
        check(rc)?;
        Ok(true)
    }

    /// Block-per-row bf16 gemv for a plane (the silu export with
    /// `silu_rows = 0`, i.e. no epilogue - it carries the tuned block size).
    pub fn bf16_gemv_t(
        &self,
        w: &QuantTensor,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
    ) -> Result<bool, GpuError> {
        let Some(f) = self.kernels.bf16_gemv_silu_f32 else {
            return Ok(false);
        };
        let w = self.bf16_plane(w, "bf16_gemv_t")?;
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        let (wp, _g1) = w.bytes.device_ptr(&self.stream);
        let (xp, _g2) = x.device_ptr(&self.stream);
        let (yp, _g3) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; silu_rows = 0 disables the epilogue
        check(unsafe {
            f(
                wp as *const _,
                core::ptr::null(),
                xp as *const _,
                yp as *mut _,
                core::ptr::null_mut(),
                in_dim as u32,
                out_dim as u32,
                0u32,
                1.0f32,
                self.stream_ptr(),
            )
        })?;
        Ok(true)
    }

    /// CTA-per-row bf16 gemv over a RAW bf16 plane (the model's own bf16
    /// twin buffers, which are byte slices rather than QuantTensors). Uses the
    /// silu export with `silu_rows = 0`, i.e. no epilogue - that entry is the
    /// one carrying the tuned block size.
    #[allow(clippy::too_many_arguments)]
    pub fn bf16_gemv_bytes(
        &self,
        w: &CudaSlice<u8>,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<bool, GpuError> {
        let Some(f) = self.kernels.bf16_gemv_silu_f32 else {
            return Ok(false);
        };
        let (wp, _g1) = w.device_ptr(&self.stream);
        let (xp, _g2) = x.device_ptr(&self.stream);
        let (yp, _g3) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; silu_rows = 0 disables the epilogue entirely
        check(unsafe {
            f(
                wp as *const _,
                core::ptr::null(),
                xp as *const _,
                yp as *mut _,
                core::ptr::null_mut(),
                in_dim as u32,
                out_dim as u32,
                0u32,
                1.0f32,
                self.stream_ptr(),
            )
        })?;
        Ok(true)
    }

    pub fn bf16_gemv_nk(
        &self,
        w: &QuantTensor,
        bias: Option<&CudaSlice<f32>>,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
    ) -> Result<bool, GpuError> {
        let Some(f) = self.kernels.bf16_gemv_nk_f32 else {
            return Ok(false);
        };
        let w = self.bf16_plane(w, "bf16_gemv_nk")?;
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        let (wp, _g1) = w.bytes.device_ptr(&self.stream);
        let (xp, _g2) = x.device_ptr(&self.stream);
        let (yp, _g3) = y.device_ptr_mut(&self.stream);
        let (bias_ptr, _gb);
        let bp: *const core::ffi::c_void = match bias {
            Some(b) => {
                (bias_ptr, _gb) = b.device_ptr(&self.stream);
                bias_ptr as *const _
            }
            None => core::ptr::null(),
        };
        // SAFETY: ABI contract; shapes come from the plane's own dims
        check(unsafe {
            f(
                wp as *const _,
                bp,
                xp as *const _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                self.stream_ptr(),
            )
        })?;
        Ok(true)
    }

    /// `bf16_gemv` writing at an output-row offset inside `y` - the twin of
    /// `q8_0_gemv_repacked_at`, for the fused `[q|k|v]` decode row.
    pub fn bf16_gemv_at(
        &self,
        w: &QuantTensor,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        y_off: usize,
    ) -> Result<(), GpuError> {
        let w = self.bf16_plane(w, "bf16_gemv_at")?;
        let f = self
            .kernels
            .bf16_gemv_f32
            .ok_or(GpuError::MissingOp("bf16_gemv_f32"))?;
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        let (wp, _g1) = w.bytes.device_ptr(&self.stream);
        let (xp, _g2) = x.device_ptr(&self.stream);
        let (yp, _g3) = y.device_ptr_mut(&self.stream);
        // SAFETY: same allocation, offset output rowlet; caller sizes y
        check(unsafe {
            f(
                wp as *const _,
                core::ptr::null(),
                xp as *const _,
                (yp + (y_off * 4) as u64) as *mut _,
                in_dim as u32,
                out_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Batched twin: `x` is `[batch, in_dim]`, `y` is `[batch, out_dim]` -
    /// the same row-major activation layout `q8_0_gemm_repacked` uses.
    pub fn bf16_gemm(
        &self,
        w: &QuantTensor,
        bias: Option<&CudaSlice<f32>>,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let out_dim = w.dims[1];
        self.bf16_gemm_dispatch(w, 0, out_dim, bias, x, y, batch, true)
    }

    /// `bf16_gemm` over a row segment of a fused plane (see
    /// `bf16_gemv_rows`) - the per-projection fallback when the pack lacks
    /// the fused `bf16_qkv_gemm_mma` slot, and the 2..=8 decode band's
    /// path (which stays on the multi-row GEMV class either way).
    pub fn bf16_gemm_rows(
        &self,
        w: &QuantTensor,
        first_row: usize,
        out_dim: usize,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        if batch == 1 {
            return self.bf16_gemv_rows(w, first_row, out_dim, x, y);
        }
        self.bf16_gemm_dispatch(w, first_row, out_dim, None, x, y, batch, true)
    }

    /// `bf16_gemm` with the 2..=8 multi-row GEMV band SKIPPED - the caller
    /// wants the tensor-core tile there.
    ///
    /// Measured on qwen4_exp (same-load width sweep, ms/step, GEMV band ->
    /// tile): c4 16.31 -> 13.77, c8 24.09 -> 16.70 (+44%), c1/c16/c32
    /// unchanged. The band's own kernel streams the plane at DRAM pace with
    /// one launch per plane; the tile splits K and fills the die. Which one
    /// wins is a per-FAMILY question - the multi-row GEMV was measured to own
    /// this band on paddleocr-vl's square planes - so this is an
    /// entry, not a flipped default.
    pub fn bf16_gemm_tile(
        &self,
        w: &QuantTensor,
        bias: Option<&CudaSlice<f32>>,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        out_dim: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        self.bf16_gemm_dispatch(w, 0, out_dim, bias, x, y, batch, false)
    }

    #[allow(clippy::too_many_arguments)]
    fn bf16_gemm_dispatch(
        &self,
        w: &QuantTensor,
        first_row: usize,
        out_dim: usize,
        bias: Option<&CudaSlice<f32>>,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        batch: usize,
        mr_ok: bool,
    ) -> Result<(), GpuError> {
        if batch == 1 {
            debug_assert_eq!(first_row, 0, "gemv segment goes via bf16_gemv_rows");
            return self.bf16_gemv(w, bias, x, y);
        }
        let w = self.bf16_plane(w, "bf16_gemm")?;
        let in_dim = w.dims[0];
        debug_assert!(first_row + out_dim <= w.dims[1], "segment exceeds plane");
        // byte offset of the segment's first row (2 bytes per bf16 element)
        let woff = (first_row * in_dim * 2) as u64;
        // Decode band (2..=8 rows): the 64x64 tile GEMM's grid degenerates to
        // out_dim/64 blocks here - 4 blocks for a 256-wide K/V plane, measured
        // at 794us for a 0.5 MB weight read on a paddleocr c4 leg.
        // The multi-row GEMV twin streams the plane at DRAM pace. Same
        // f32-product class; warp-local summation order (the sanctioned
        // reorder class - serial gemv vs batched gemm already differ the same
        // way and hold greedy parity).
        // NB=8 (batch 5..8) is the mr kernel's issue-bound arm: 16 cvt +
        // 128 FMA per 32 weight bytes per lane - 806 GB/s on the granite
        // lm_head (1028us/tick, the top c8 cost after the wqkv fusion) while
        // the BN=32 tile holds 1476 (bench/bf16_head_mr_bench.cu, cold
        // 4-plane rotation; b2/b4 are tied at the wall). Big outs have
        // die-filling tile grids (out/32 CTAs), so they take the tile; the
        // mr class keeps the small-out planes it was built for (256-wide
        // K/V heads = 4-CTA tile grids).
        //
        // The window is a knob as well. On the qwen4_exp decode planes the
        // tile is ~2x better per row even inside it (c8 4.9 ms/row on this arm
        // against c16's 2.5 ms/row on the tile), so `PADDOCK_BF16_MR=0` hands
        // the whole 2..=8 window to the tile, and `mr_ok` lets a caller that
        // knows its plane decline without the environment.
        static MR_ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let mr_on = *MR_ON
            .get_or_init(|| !matches!(std::env::var("PADDOCK_BF16_MR").ok().as_deref(), Some("0")));
        let mr_band =
            mr_ok && mr_on && (2..=8).contains(&batch) && !(batch >= 5 && out_dim >= 8192);
        if mr_band
            && in_dim % 16 == 0
            && let Some(f) = self.kernels.bf16_gemv_mr_f32
        {
            let (wp, _g1) = w.bytes.device_ptr(&self.stream);
            let (xp, _g2) = x.device_ptr(&self.stream);
            let (yp, _g3) = y.device_ptr_mut(&self.stream);
            let (bias_ptr, _gb);
            let bp: *const core::ffi::c_void = match bias {
                Some(b) => {
                    (bias_ptr, _gb) = b.device_ptr(&self.stream);
                    bias_ptr as *const _
                }
                None => core::ptr::null(),
            };
            // SAFETY: ABI contract; segment offset stays inside the plane
            return check(unsafe {
                f(
                    (wp + woff) as *const _,
                    bp,
                    xp as *const _,
                    yp as *mut _,
                    in_dim as u32,
                    out_dim as u32,
                    batch as u32,
                    self.stream_ptr(),
                )
            });
        }
        // Prefill band (batch > 8): the tensor-core arm. Casts the f32
        // activations to bf16 in its smem stage - the parity reference's own
        // batched-BF16 class (llama.cpp = cublasGemmEx bf16xbf16, f32
        // compute), gated by the parity battery. The f32-FMA tile below stays
        // the fallback for older packs and ragged in_dim.
        if batch >= 2 && in_dim % 16 == 0 {
            // the pack admits 2..8 since the head-band fix; small-out 2..8
            // returned via the mr branch above
            if let Some(f) = self.kernels.bf16_gemm_mma {
                let (wp, _g1) = w.bytes.device_ptr(&self.stream);
                let (xp, _g2) = x.device_ptr(&self.stream);
                let (yp, _g3) = y.device_ptr_mut(&self.stream);
                let (bias_ptr, _gb);
                let bp: *const core::ffi::c_void = match bias {
                    Some(b) => {
                        (bias_ptr, _gb) = b.device_ptr(&self.stream);
                        bias_ptr as *const _
                    }
                    None => core::ptr::null(),
                };
                // SAFETY: ABI contract; segment offset stays inside the plane
                return check(unsafe {
                    f(
                        (wp + woff) as *const _,
                        bp,
                        xp as *const _,
                        yp as *mut _,
                        in_dim as u32,
                        out_dim as u32,
                        batch as u32,
                        self.stream_ptr(),
                    )
                });
            }
        }
        let f = self
            .kernels
            .bf16_gemm_f32
            .ok_or(GpuError::MissingOp("bf16_gemm_f32"))?;
        let (wp, _g1) = w.bytes.device_ptr(&self.stream);
        let (xp, _g2) = x.device_ptr(&self.stream);
        let (yp, _g3) = y.device_ptr_mut(&self.stream);
        let (bias_ptr, _gb);
        let bp: *const core::ffi::c_void = match bias {
            Some(b) => {
                (bias_ptr, _gb) = b.device_ptr(&self.stream);
                bias_ptr as *const _
            }
            None => core::ptr::null(),
        };
        // SAFETY: ABI contract; segment offset stays inside the plane
        check(unsafe {
            f(
                (wp + woff) as *const _,
                bp,
                xp as *const _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// True when the pack carries the fused q|k|v decode-band GEMM (slot
    /// 424) - the caller keeps three per-segment launches otherwise.
    pub fn has_bf16_qkv_gemm(&self) -> bool {
        self.kernels.bf16_qkv_gemm_mma.is_some()
    }

    /// One launch computing q|k|v against the shared `x` over the fused
    /// `[q;k;v]` plane (dims `[in, oq + 2*okv]`), segmented store into the
    /// three y planes. Decode-band only (the launcher declines batch <= 1);
    /// per out-row bit-identical to `bf16_gemm` on the matching segment.
    #[allow(clippy::too_many_arguments)]
    pub fn bf16_qkv_gemm(
        &self,
        w: &QuantTensor,
        x: &CudaSlice<f32>,
        yq: &mut CudaSlice<f32>,
        yk: &mut CudaSlice<f32>,
        yv: &mut CudaSlice<f32>,
        oq: usize,
        okv: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        let w = self.bf16_plane(w, "bf16_qkv_gemm")?;
        let f = self
            .kernels
            .bf16_qkv_gemm_mma
            .ok_or(GpuError::MissingOp("bf16_qkv_gemm_mma"))?;
        let in_dim = w.dims[0];
        debug_assert_eq!(oq + 2 * okv, w.dims[1], "fused plane rows");
        let (wp, _g1) = w.bytes.device_ptr(&self.stream);
        let (xp, _g2) = x.device_ptr(&self.stream);
        let (qp, _g3) = yq.device_ptr_mut(&self.stream);
        let (kp, _g4) = yk.device_ptr_mut(&self.stream);
        let (vp, _g5) = yv.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; y planes are [batch, seg] sized by the caller
        check(unsafe {
            f(
                wp as *const _,
                xp as *const _,
                qp as *mut _,
                kp as *mut _,
                vp as *mut _,
                in_dim as u32,
                oq as u32,
                okv as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// The prefill bf16 pair (slots 756 / 757) is in the pack.
    pub fn has_bf16_gemm_pf(&self) -> bool {
        self.kernels.bf16_gemm_pf.is_some() && self.kernels.bf16_qkv_gemm_pf.is_some()
    }

    /// Whether `rows` of a plane `out_dim` wide take the prefill pair: from
    /// 128 rows, once the 128x128 grid holds a third of a wave - below that
    /// the decode ladder's K-split fills the die better (GB10, 2560 -> 512:
    /// 256 rows 34 vs 54 us on the ladder's side, 512 rows 59 vs 54 on
    /// this one; the wide planes cross near 128 rows). The pair is
    /// bit-identical to the unsplit plain tile, so the election moves
    /// time, and only the K-split regroup class the ladder already has.
    pub fn bf16_pf_elect(&self, out_dim: usize, rows: usize) -> bool {
        rows >= 128
            && 3 * out_dim.div_ceil(128) * rows.div_ceil(128) >= self.sm_count.max(1)
            && self.has_bf16_gemm_pf()
    }

    /// `y = W x (+ bias)` at prefill widths over bf16 rows `x16`
    /// (`convert_f32_bf16` of the f32 rows) - slot 756.
    pub fn bf16_gemm_pf(
        &self,
        w: &QuantTensor,
        bias: Option<&CudaSlice<f32>>,
        x16: &CudaSlice<half::bf16>,
        y: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let w = self.bf16_plane(w, "bf16_gemm_pf")?;
        let f = self
            .kernels
            .bf16_gemm_pf
            .ok_or(GpuError::MissingOp("bf16_gemm_pf"))?;
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        let (wp, _g1) = w.bytes.device_ptr(&self.stream);
        let bp = bias.map(|b| b.device_ptr(&self.stream));
        let (xp, _g3) = x16.device_ptr(&self.stream);
        let (yp, _g4) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 756); x16 holds batch * in_dim, y batch * out_dim
        check(unsafe {
            f(
                wp as *const _,
                bp.as_ref()
                    .map_or(std::ptr::null(), |(p, _)| *p as *const _),
                xp as *const _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// The fused q|k|v decode-band GEMV (slot 773) is in the pack.
    pub fn has_bf16_qkv_gemv_mr(&self) -> bool {
        self.kernels.bf16_qkv_gemv_mr.is_some()
    }

    /// q|k|v off the fused plane in one decode-band multi-row GEMV launch
    /// (slot 773, 2..=8 rows), each row routed to its segment plane - per
    /// row the plain multi-row GEMV's arithmetic. `Ok(false)` when the pack
    /// lacks the slot or the rows are outside the band (nothing launched).
    #[allow(clippy::too_many_arguments)]
    pub fn bf16_qkv_gemv_mr(
        &self,
        w: &QuantTensor,
        x: &CudaSlice<f32>,
        yq: &mut CudaSlice<f32>,
        yk: &mut CudaSlice<f32>,
        yv: &mut CudaSlice<f32>,
        oq: usize,
        okv: usize,
        batch: usize,
    ) -> Result<bool, GpuError> {
        let Some(f) = self.kernels.bf16_qkv_gemv_mr else {
            return Ok(false);
        };
        let w = self.bf16_plane(w, "bf16_qkv_gemv_mr")?;
        debug_assert_eq!(oq + 2 * okv, w.dims[1], "fused plane rows");
        let in_dim = w.dims[0];
        if !(2..=8).contains(&batch) || in_dim % 16 != 0 {
            return Ok(false);
        }
        let (wp, _g1) = w.bytes.device_ptr(&self.stream);
        let (xp, _g2) = x.device_ptr(&self.stream);
        let (qp, _g3) = yq.device_ptr_mut(&self.stream);
        let (kp, _g4) = yk.device_ptr_mut(&self.stream);
        let (vp, _g5) = yv.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 773); y planes are [batch, seg] sized by the caller
        check(unsafe {
            f(
                wp as *const _,
                xp as *const _,
                qp as *mut _,
                kp as *mut _,
                vp as *mut _,
                in_dim as u32,
                oq as u32,
                okv as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })?;
        Ok(true)
    }

    /// The fused q|k|v twin of [`Self::bf16_gemm_pf`] (slot 757).
    #[allow(clippy::too_many_arguments)]
    pub fn bf16_qkv_gemm_pf(
        &self,
        w: &QuantTensor,
        x16: &CudaSlice<half::bf16>,
        yq: &mut CudaSlice<f32>,
        yk: &mut CudaSlice<f32>,
        yv: &mut CudaSlice<f32>,
        oq: usize,
        okv: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        let w = self.bf16_plane(w, "bf16_qkv_gemm_pf")?;
        let f = self
            .kernels
            .bf16_qkv_gemm_pf
            .ok_or(GpuError::MissingOp("bf16_qkv_gemm_pf"))?;
        debug_assert_eq!(oq + 2 * okv, w.dims[1], "fused plane rows");
        let in_dim = w.dims[0];
        let (wp, _g1) = w.bytes.device_ptr(&self.stream);
        let (xp, _g2) = x16.device_ptr(&self.stream);
        let (qp, _g3) = yq.device_ptr_mut(&self.stream);
        let (kp, _g4) = yk.device_ptr_mut(&self.stream);
        let (vp, _g5) = yv.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 757); y planes are [batch, seg] sized by the caller
        check(unsafe {
            f(
                wp as *const _,
                xp as *const _,
                qp as *mut _,
                kp as *mut _,
                vp as *mut _,
                in_dim as u32,
                oq as u32,
                okv as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Two-segment twin of [`Self::bf16_qkv_gemm`] - one launch over a plane
    /// that folds exactly two projections, storing rows `[0, oq)` to `ya` and
    /// `[oq, oq + ob)` to `yb`.
    ///
    /// The kernel is the same one the q|k|v arm uses; only the row-routing
    /// arithmetic differs, and its third segment is simply never reached
    /// because the plane has no row at or beyond `oq + ob`. That is why `yb`
    /// is passed twice: the `yv` pointer is dereferenced only by a row that
    /// cannot exist here, so it needs to be valid, not distinct.
    ///
    /// Why this exists: the hyper-connection down plane folds the low-rank
    /// projection and the block-inject rows into one residency, and at batch 1
    /// the engine already reads both in one launch (the inject comes out in the
    /// low-rank output's tail). Above batch 1 that tail is not contiguous, so
    /// the lane fell back to two `matmul_rows` calls - measured at c32, 2.02 +
    /// 2.00 launches per layer and 0.944 + 0.801 ms/step, i.e. a third of the
    /// whole dense launch count for one projection pair.
    #[allow(clippy::too_many_arguments)]
    pub fn bf16_gemm_2seg(
        &self,
        w: &QuantTensor,
        x: &CudaSlice<f32>,
        ya: &mut CudaSlice<f32>,
        yb: &mut CudaSlice<f32>,
        oq: usize,
        ob: usize,
        batch: usize,
    ) -> Result<bool, GpuError> {
        if batch < 2 {
            return Ok(false);
        }
        let w = self.bf16_plane(w, "bf16_gemm_2seg")?;
        let in_dim = w.dims[0];
        if oq + ob != w.dims[1] || in_dim % 16 != 0 {
            return Ok(false);
        }
        let Some(f) = self.kernels.bf16_seg2_gemm_mma else {
            return Ok(false);
        };
        let (wp, _g1) = w.bytes.device_ptr(&self.stream);
        let (xp, _g2) = x.device_ptr(&self.stream);
        let (ap, _g3) = ya.device_ptr_mut(&self.stream);
        let (bp, _g4) = yb.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 529). The 2-segment launcher computes the
        // fused row count as oq + ob, so it reads exactly the plane it was
        // given. NOTE: the q|k|v launcher cannot be reused here - it computes
        // oq + 2*okv and would read `ob` rows past the end; that was a real
        // gate failure, not a hypothetical.
        let rc = unsafe {
            f(
                wp as *const _,
                xp as *const _,
                ap as *mut _,
                bp as *mut _,
                in_dim as u32,
                oq as u32,
                ob as u32,
                batch as u32,
                self.stream_ptr(),
            )
        };
        if rc == -2 {
            return Ok(false);
        }
        check(rc)?;
        Ok(true)
    }

    /// Build the permuted twin of a hyper-connection up plane (slot 530).
    ///
    /// The fused mix epilogue needs the four `hc` branches of one output
    /// element to land in one thread's registers. At `RG=2` the m16n8k16
    /// fragment gives a thread rows `{r, r+8, r+16, r+24}`, so the permute is
    /// `permuted_row(d, s) = (d/8)*32 + s*8 + (d%8)` - a verified bijection.
    /// Run once at load; the permuted plane is all the mix arm reads.
    pub fn bf16_hcmix_permute(
        &self,
        w: &QuantTensor,
        hidden: usize,
        hc: usize,
    ) -> Result<Option<CudaSlice<u8>>, GpuError> {
        let Some(f) = self.kernels.bf16_hcmix_permute else {
            return Ok(None);
        };
        let p = self.bf16_plane(w, "bf16_hcmix_permute")?;
        let in_dim = p.dims[0];
        if p.dims[1] != hidden * hc || hc != 4 || !hidden.is_multiple_of(8) {
            return Ok(None);
        }
        let mut dst = self.alloc_u8(p.bytes.len())?;
        {
            let (sp, _g1) = p.bytes.device_ptr(&self.stream);
            let (dp, _g2) = dst.device_ptr_mut(&self.stream);
            // SAFETY: ABI contract (slot 530); dst is the same size as src
            check(unsafe {
                f(
                    sp as *const _,
                    dp as *mut _,
                    hidden as u32,
                    hc as u32,
                    in_dim as u32,
                    self.stream_ptr(),
                )
            })?;
        }
        self.synchronize()?;
        Ok(Some(dst))
    }

    /// Up-GEMM with the hyper-connection mix tail fused in (slot 531).
    /// `Ok(false)` means "not taken, do it yourself".
    #[allow(clippy::too_many_arguments)]
    pub fn bf16_hcmix_gemm(
        &self,
        wperm: &CudaSlice<u8>,
        x: &CudaSlice<f32>,
        xn: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        in_dim: usize,
        hidden: usize,
        hc: usize,
        batch: usize,
    ) -> Result<bool, GpuError> {
        let Some(f) = self.kernels.bf16_hcmix_gemm else {
            return Ok(false);
        };
        if batch < 2 || hc != 4 || !in_dim.is_multiple_of(16) || !hidden.is_multiple_of(8) {
            return Ok(false);
        }
        let (wp, _g1) = wperm.device_ptr(&self.stream);
        let (xp, _g2) = x.device_ptr(&self.stream);
        let (np, _g3) = xn.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 531)
        let rc = unsafe {
            f(
                wp as *const _,
                xp as *const _,
                np as *const _,
                op as *mut _,
                in_dim as u32,
                hidden as u32,
                hc as u32,
                batch as u32,
                self.stream_ptr(),
            )
        };
        if rc == -2 {
            return Ok(false);
        }
        check(rc)?;
        Ok(true)
    }

    /// Embedding gather that picks its kernel from the TABLE's own class -
    /// a mixed UD file can hand either a Q8_0 or a bf16 `token_embd`, and the
    /// call sites (decode input, prefill rows, multimodal splice, drafter)
    /// should not each have to remember that.
    pub fn embed_gather_plane(
        &self,
        table: &QuantTensor,
        tokens: &CudaSlice<u32>,
        out: &mut CudaSlice<f32>,
        embd: usize,
        n_tokens: usize,
        scale: f32,
    ) -> Result<(), GpuError> {
        match table.ty {
            GgmlType::Bf16 => self.embed_gather_bf16(table, tokens, out, embd, n_tokens, scale),
            _ => self.embed_gather_q8(table, tokens, out, embd, n_tokens, scale),
        }
    }

    /// bf16 twin of `embed_gather_q8`: device-selected token rows out of a
    /// bf16 embedding table with the fused output scale (graph-capturable).
    pub fn embed_gather_bf16(
        &self,
        table: &QuantTensor,
        tokens: &CudaSlice<u32>,
        out: &mut CudaSlice<f32>,
        embd: usize,
        n_tokens: usize,
        scale: f32,
    ) -> Result<(), GpuError> {
        let table = self.bf16_plane(table, "embed_gather_bf16")?;
        let f = self
            .kernels
            .embed_gather_bf16
            .ok_or(GpuError::MissingOp("embed_gather_bf16"))?;
        let (tp, _g1) = table.bytes.device_ptr(&self.stream);
        let (kp, _g2) = tokens.device_ptr(&self.stream);
        let (op, _g3) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; out is [n_tokens, embd]
        check(unsafe {
            f(
                tp as *const _,
                kp as *const _,
                op as *mut _,
                embd as u32,
                n_tokens as u32,
                scale,
                self.stream_ptr(),
            )
        })
    }
}
