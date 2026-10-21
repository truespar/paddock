//! e4m3 planes: f8row/f8t GEMM, e4m3 quantize + norm fusions.

use super::error::*;
use super::*;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

/// F8CUT (vendored cutlass) is a NOSPEC-lane feature. The intercept
/// REGRESSED the MTP spec verify ticks (gemma c4-spec 849->812, and the m>=32
/// dispatch floor did not recover it - not a batch-threshold effect), so when a
/// drafter attaches (attach_mtp) this flips true and f8t_gemm{,_off} fall back
/// to tc5r. Set once at load before serving threads spawn; process-global
/// because a paddock process serves one model/policy.
static F8CUT_SPEC_OFF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Width floor for the F8CUT intercept in a SPEC process. The regression
/// that motivated it is an ~18-row c4-spec verify; the c32-spec 96-row
/// verify measured cutlass AHEAD (-1.2ms/tick, acceptance-matched). 32
/// splits the regimes: the ~18-row c4-spec verify stays on tc5r while spec
/// decode (32), prefill chunks (65..256) and c32-class verify (~56..296)
/// take the intercept - identical widths to the measured leg, which ran the
/// global m>=16 floor. 0 = no spec floor
/// (nospec lane, or the blanket kill above is in force).
static F8CUT_SPEC_MINB: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Whether the kt lin arm can actually launch here. Set false on the first
/// 801 (its rowwise kernel is compiled for cc 12 only); see f8d_gemm_mma_ks.
static LIN_KT_OK: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

impl GpuExecutor {
    /// Ride tc5r instead of the vendored-cutlass F8CUT intercept, process-wide.
    /// Called from `attach_mtp` - F8CUT wins nospec c32 but loses the spec
    /// verify ticks.
    pub fn set_f8cut_spec_off(&self) {
        F8CUT_SPEC_OFF.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Keep the F8CUT intercept live in a spec process but floor it at
    /// `minb` rows (P44 ship shape; see `attach_mtp`). Same set-once-at-load
    /// contract as `set_f8cut_spec_off`.
    pub fn set_f8cut_spec_minb(&self, minb: usize) {
        F8CUT_SPEC_MINB.store(minb, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether the loaded pack exports the paged KV kernels (decode + append).
    /// Absent on packs before P2 - callers fall back to the dense path.
    pub fn has_paged_kv(&self) -> bool {
        self.kernels.attn_decode_batch_paged.is_some()
            && self.kernels.kv_append_batch_paged.is_some()
    }

    /// Whether the loaded pack exports the multi-slot batched tiled prefill
    /// attention (absent on packs before E1d - callers fall back to decode).
    pub fn has_attn_prefill_batch(&self) -> bool {
        self.kernels.attn_prefill_batch.is_some()
    }

    pub fn has_attn_prefill_batch_f16(&self) -> bool {
        self.kernels.attn_prefill_batch_f16.is_some()
    }

    pub fn has_q8_0_gemm_mmq_pipe64(&self) -> bool {
        self.kernels.q8_0_gemm_mmq_pipe64.is_some()
    }

    pub fn has_q8_0_gemm_mmq_pipe_sk(&self) -> bool {
        self.kernels.q8_0_gemm_mmq_pipe_sk.is_some()
    }

    /// Tail split-K mmq: wave-quantization fix for narrow tiled GEMMs. The
    /// launcher engages only when the tail is worth it (else it runs the
    /// plain pipe kernel). Tail tiles are the mmq CLASS, not bit-identical
    /// (outer f32 sum regroups deterministically).
    pub fn q8_0_gemm_mmq_pipe_sk(
        &self,
        w: &RepackedQ8,
        yq: &CudaSlice<u8>,
        partials: &mut CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_gemm_mmq_pipe_sk
            .ok_or(GpuError::MissingOp("q8_0_gemm_mmq_pipe_sk"))?;
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        if paddock_models::dev_var_os!("PADDOCK_TRACE_SK").is_some() {
            tracing::info!("  [sk] in={in_dim} out={out_dim} batch={batch}");
        }
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (sp, _g2) = w.scale.device_ptr(&self.stream);
        let (yqp, _g3) = yq.device_ptr(&self.stream);
        let (pp, _g4) = partials.device_ptr_mut(&self.stream);
        let (yp, _g5) = y.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                dp as *const _,
                sp as *const _,
                yqp as *const _,
                yp as *mut _,
                pp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.sm_count as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_qkv_norm_rope_append(&self) -> bool {
        self.kernels.qkv_norm_rope_append.is_some()
    }

    /// Concatenate repacked-Q8 weights along the out dimension (fused-QKV
    /// planes: one wide GEMM instead of several narrow wave-starved ones).
    /// All inputs must share dims[0] (the input dim).
    pub fn concat_q8(&self, parts: &[&RepackedQ8]) -> Result<RepackedQ8, GpuError> {
        let in_dim = parts[0].dims[0];
        debug_assert!(parts.iter().all(|p| p.dims[0] == in_dim));
        let out: usize = parts.iter().map(|p| p.dims[1]).sum();
        let mut data = self.alloc_u8(out * in_dim)?;
        let mut scale = self.alloc_u8(out * in_dim / 32 * 2)?;
        let (mut d_off, mut s_off) = (0usize, 0usize);
        for p in parts {
            let dv = data
                .try_slice_mut(d_off..d_off + p.data.len())
                .ok_or_else(|| oob("concat_q8: data range"))?;
            self.stream.memcpy_dtod(&p.data, &mut { dv }).map_err(drv)?;
            let sv = scale
                .try_slice_mut(s_off..s_off + p.scale.len())
                .ok_or_else(|| oob("concat_q8: scale range"))?;
            self.stream
                .memcpy_dtod(&p.scale, &mut { sv })
                .map_err(drv)?;
            d_off += p.data.len();
            s_off += p.scale.len();
        }
        Ok(RepackedQ8 {
            data,
            scale,
            dims: vec![in_dim, out],
        })
    }

    /// Fused-QKV norm/rope/scatter - one launch consuming the combined qkv
    /// GEMM output (see the pack kernel notes). head_dim must be 128.
    #[allow(clippy::too_many_arguments)]
    pub fn qkv_norm_rope_append(
        &self,
        x: &CudaSlice<f32>,
        wq_norm: &CudaSlice<f32>,
        wk_norm: &CudaSlice<f32>,
        qn: &mut CudaSlice<f32>,
        kcache: &mut CudaSlice<u8>,
        vcache: &mut CudaSlice<u8>,
        positions: &CudaSlice<u32>,
        slots: Option<&CudaSlice<u32>>,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_ctx: usize,
        eps: f32,
        rope: (f32, f32, f32, f32, f32, f32),
        batch: usize,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .qkv_norm_rope_append
            .ok_or(GpuError::MissingOp("qkv_norm_rope_append"))?;
        let (theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale) = rope;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (wqp, _g2) = wq_norm.device_ptr(&self.stream);
        let (wkp, _g3) = wk_norm.device_ptr(&self.stream);
        let (qp, _g4) = qn.device_ptr_mut(&self.stream);
        let (kp, _g5) = kcache.device_ptr_mut(&self.stream);
        let (vp, _g6) = vcache.device_ptr_mut(&self.stream);
        let (pp, _g7) = positions.device_ptr(&self.stream);
        let sp = slots.map(|s| s.device_ptr(&self.stream));
        check(unsafe {
            f(
                xp as *const _,
                wqp as *const _,
                wkp as *const _,
                qp as *mut _,
                kp as *mut _,
                vp as *mut _,
                pp as *const _,
                sp.as_ref()
                    .map_or(std::ptr::null(), |(p, _)| *p as *const _),
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                max_ctx as u32,
                eps,
                theta_scale,
                freq_scale,
                corr_low,
                corr_high,
                ext_factor,
                mscale,
                batch as u32,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_fused_qk_norm_rope(&self) -> bool {
        self.kernels.q_norm_rope.is_some() && self.kernels.k_norm_rope_append.is_some()
    }

    pub fn has_f8_gemm_w8(&self) -> bool {
        self.kernels.f8_gemm_w8.is_some()
            && self.kernels.q8_0_to_f8w.is_some()
            && self.kernels.quantize_e4m3.is_some()
    }

    pub fn has_f8row_gemm(&self) -> bool {
        self.kernels.f8row_gemm.is_some()
            && self.kernels.q8_0_to_f8row.is_some()
            && self.kernels.quantize_e4m3_row.is_some()
    }

    /// Q8_0 -> per-ROW e4m3 plane (one f32 power-of-2 scale per output row).
    /// The sm_100 prefill class - scales fold in the GEMM epilogue, never in
    /// the K loop. Lossy (coarser than per-32), quality-gated.
    pub fn q8_0_to_f8row(&self, w: &RepackedQ8) -> Result<F8RowPlane, GpuError> {
        self.q8_0_to_f8row_rows(w, w.dims[1])
    }

    pub fn has_bf16_to_f8row(&self) -> bool {
        self.kernels.bf16_to_f8row.is_some()
    }

    /// bf16 -> the same per-row e4m3 plane, straight from the native weights.
    ///
    /// `q8_0_to_f8row` is the only other producer of this plane and it needs a
    /// Q8 source, so a checkpoint whose lm_head ships bf16 (muse-glimmer) had
    /// no route onto the f8t tile GEMM at all - every head call fell back to a
    /// plain bf16 kernel reading 2x the bytes. This closes that edge without
    /// a Q8 round trip, which would have double-quantized.
    ///
    /// `bytes` is the raw bf16 tensor in row-major [in_dim, out_dim] order,
    /// uploaded here rather than kept resident: the caller is the loader and
    /// the source plane is dropped as soon as the tiles are built.
    pub fn bf16_to_f8row(
        &self,
        bytes: &[u8],
        in_dim: usize,
        out_dim: usize,
    ) -> Result<F8RowPlane, GpuError> {
        let f = self
            .kernels
            .bf16_to_f8row
            .ok_or(GpuError::MissingOp("bf16_to_f8row"))?;
        if bytes.len() != in_dim * out_dim * 2 {
            return Err(GpuError::Driver(format!(
                "bf16_to_f8row: {} bytes for [{in_dim}, {out_dim}] bf16 (want {})",
                bytes.len(),
                in_dim * out_dim * 2
            )));
        }
        let src: CudaSlice<u8> = self.stream.clone_htod(bytes).map_err(drv)?;
        let mut data = self.alloc_u8(in_dim * out_dim)?;
        let mut scale: CudaSlice<f32> = self.stream.alloc_zeros(out_dim).map_err(drv)?;
        {
            let (sp, _g1) = src.device_ptr(&self.stream);
            let (dp, _g2) = data.device_ptr_mut(&self.stream);
            let (rp, _g3) = scale.device_ptr_mut(&self.stream);
            // SAFETY: ABI contract; sizes checked above, one CTA per out row
            check(unsafe {
                f(
                    sp as *const _,
                    dp as *mut _,
                    rp as *mut _,
                    in_dim as u32,
                    out_dim as u32,
                    self.stream_ptr(),
                )
            })?;
        }
        self.synchronize()?; // `src` dies here; the kernel must be done reading it
        Ok(F8RowPlane { data, scale })
    }

    /// `q8_0_to_f8row` with an explicit row count, for planes whose row count
    /// is not `dims[1]`: an expert plane is `[n_embd, ff, n_expert]` and
    /// converts as one flat `n_expert * ff`-row stream (the per-row scale
    /// doesn't care that the rows are grouped by expert).
    pub fn q8_0_to_f8row_rows(
        &self,
        w: &RepackedQ8,
        out_dim: usize,
    ) -> Result<F8RowPlane, GpuError> {
        let f = self
            .kernels
            .q8_0_to_f8row
            .ok_or(GpuError::MissingOp("q8_0_to_f8row"))?;
        let in_dim = w.dims[0];
        debug_assert_eq!(w.data.len(), in_dim * out_dim);
        let mut data = self.alloc_u8(in_dim * out_dim)?;
        let mut scale: CudaSlice<f32> = self.stream.alloc_zeros(out_dim).map_err(drv)?;
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
                    in_dim as u32,
                    out_dim as u32,
                    self.stream_ptr(),
                )
            })?;
        }
        Ok(F8RowPlane { data, scale })
    }

    /// f32 -> per-ROW e4m3 activation quant (f32 scale per token row).
    /// u8-typed twin of `quantize_e4m3_row` for PLANE construction (identical
    /// ABI - the pack writes e4m3 bytes either way; the row scales it emits
    /// are the house pow2 e-pick, which is what makes the pc plane's per-32
    /// strip exactly representable).
    pub fn quantize_e4m3_row_u8(
        &self,
        x: &CudaSlice<f32>,
        q: &mut CudaSlice<u8>,
        rs: &mut CudaSlice<f32>,
        n_dim: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .quantize_e4m3_row
            .ok_or(GpuError::MissingOp("quantize_e4m3_row"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (qp, _g2) = q.device_ptr_mut(&self.stream);
        let (sp, _g3) = rs.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; n_dim % 32 == 0 checked by the launcher
        check(unsafe {
            f(
                xp as *const _,
                qp as *mut _,
                sp as *mut _,
                n_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// u8-typed twin of [`Self::quantize_e4m3`] for PLANE construction
    /// (identical ABI - the pack writes e4m3 bytes either way). The f8w
    /// plane class: e4m3 payload + one ue8m0 scale per 32 values, which is
    /// exactly what `f8_gemm_w8`/`f8_gemm_w8_o16` read.
    pub fn quantize_e4m3_u8(
        &self,
        x: &CudaSlice<f32>,
        q: &mut CudaSlice<u8>,
        scale: &mut CudaSlice<u8>,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .quantize_e4m3
            .ok_or(GpuError::MissingOp("quantize_e4m3"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (qp, _g2) = q.device_ptr_mut(&self.stream);
        let (sp, _g3) = scale.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; n % 32 == 0, buffers sized [n]/[n/32]
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

    pub fn quantize_e4m3_row(
        &self,
        x: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        rs: &mut CudaSlice<f32>,
        n_dim: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .quantize_e4m3_row
            .ok_or(GpuError::MissingOp("quantize_e4m3_row"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (qp, _g2) = q.device_ptr_mut(&self.stream);
        let (sp, _g3) = rs.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; n_dim % 32 == 0 checked by the launcher
        check(unsafe {
            f(
                xp as *const _,
                qp as *mut _,
                sp as *mut _,
                n_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// gate|up as one grid over two f8row planes of the same shape (decode
    /// widths, unsplit). Ok(false) = the pack declined the shape; run the two
    /// single GEMMs instead. Bit-identical to them by construction.
    #[allow(clippy::too_many_arguments)]
    pub fn f8row_gemm2(
        &self,
        w0: &F8RowPlane,
        w1: &F8RowPlane,
        xq: &CudaSlice<i8>,
        xrs: &CudaSlice<f32>,
        y0: &mut CudaSlice<f32>,
        y1: &mut CudaSlice<f32>,
        in_dim: usize,
        out_dim: usize,
        batch: usize,
    ) -> Result<bool, GpuError> {
        let Some(f) = self.kernels.f8row_gemm2 else {
            return Ok(false);
        };
        let (d0, _g1) = w0.data.device_ptr(&self.stream);
        let (s0, _g2) = w0.scale.device_ptr(&self.stream);
        let (d1, _g3) = w1.data.device_ptr(&self.stream);
        let (s1, _g4) = w1.scale.device_ptr(&self.stream);
        let (xp, _g5) = xq.device_ptr(&self.stream);
        let (sp, _g6) = xrs.device_ptr(&self.stream);
        let (o0, _g7) = y0.device_ptr_mut(&self.stream);
        let (o1, _g8) = y1.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; the pack validates the shape and declines with 100
        let rc = unsafe {
            f(
                d0 as *const _,
                s0 as *const _,
                d1 as *const _,
                s1 as *const _,
                xp as *const _,
                sp as *const _,
                o0 as *mut _,
                o1 as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        };
        if rc == 100 {
            return Ok(false);
        }
        check(rc).map(|_| true)
    }

    /// prefill-width swiglu + e4m3-row quant in one pass (gate left as-is; the
    /// down GEMM reads the staged q). Bit-identical to `swiglu` then
    /// `quantize_e4m3_row`. Ok(false) = pack without the entry.
    pub fn swiglu_quant_e4m3_row(
        &self,
        gate: &CudaSlice<f32>,
        up: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        rs: &mut CudaSlice<f32>,
        n_ff: usize,
        batch: usize,
    ) -> Result<bool, GpuError> {
        let Some(f) = self.kernels.swiglu_quant_e4m3_row else {
            return Ok(false);
        };
        let (gp, _g1) = gate.device_ptr(&self.stream);
        let (up_, _g2) = up.device_ptr(&self.stream);
        let (qp, _g3) = q.device_ptr_mut(&self.stream);
        let (sp, _g4) = rs.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; n_ff % 32 == 0 checked by the launcher
        check(unsafe {
            f(
                gp as *const _,
                up_ as *const _,
                qp as *mut _,
                sp as *mut _,
                n_ff as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
        .map(|_| true)
    }

    /// rmsnorm + e4m3-row quant in one launch (decode band): writes xn AND
    /// (q, rs) bit-identical to rmsnorm_batch followed by quantize_e4m3_row.
    /// Ok(false) when the pack lacks it or declines (batch >= 64, odd widths).
    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm_quant_e4m3_row(
        &self,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        xn: &mut CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        rs: &mut CudaSlice<f32>,
        n: usize,
        eps: f32,
        batch: usize,
    ) -> Result<bool, GpuError> {
        let Some(f) = self.kernels.rmsnorm_quant_e4m3_row else {
            return Ok(false);
        };
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let (np, _g3) = xn.device_ptr_mut(&self.stream);
        let (qp, _g4) = q.device_ptr_mut(&self.stream);
        let (sp, _g5) = rs.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; the launcher gates width/batch and returns 100 to decline
        let rc = unsafe {
            f(
                xp as *const _,
                wp as *const _,
                np as *mut _,
                qp as *mut _,
                sp as *mut _,
                n as u32,
                eps,
                batch as u32,
                self.stream_ptr(),
            )
        };
        if rc == 100 {
            return Ok(false);
        }
        check(rc).map(|_| true)
    }

    /// x += pscale*proj, rmsnorm, e4m3-row quant in one launch (the FFN norm
    /// site); same bit-identity contract as `rmsnorm_quant_e4m3_row`.
    #[allow(clippy::too_many_arguments)]
    pub fn add_rmsnorm_scaled_quant_e4m3_row(
        &self,
        x: &mut CudaSlice<f32>,
        proj: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        xn: &mut CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        rs: &mut CudaSlice<f32>,
        n: usize,
        eps: f32,
        pscale: f32,
        batch: usize,
    ) -> Result<bool, GpuError> {
        let Some(f) = self.kernels.add_rmsnorm_scaled_quant_e4m3_row else {
            return Ok(false);
        };
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (pp, _g2) = proj.device_ptr(&self.stream);
        let (wp, _g3) = w.device_ptr(&self.stream);
        let (np, _g4) = xn.device_ptr_mut(&self.stream);
        let (qp, _g5) = q.device_ptr_mut(&self.stream);
        let (sp, _g6) = rs.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; the launcher gates width/batch and returns 100 to decline
        let rc = unsafe {
            f(
                xp as *mut _,
                pp as *const _,
                wp as *const _,
                np as *mut _,
                qp as *mut _,
                sp as *mut _,
                n as u32,
                eps,
                pscale,
                batch as u32,
                self.stream_ptr(),
            )
        };
        if rc == 100 {
            return Ok(false);
        }
        check(rc).map(|_| true)
    }

    /// Fold-free per-row e4m3 GEMM: y = (e4m3 W x e4m3 X) * ws[row] * xs[col]
    /// applied in the epilogue only.
    #[allow(clippy::too_many_arguments)]
    pub fn f8row_gemm(
        &self,
        w: &F8RowPlane,
        xq: &CudaSlice<i8>,
        xrs: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        in_dim: usize,
        out_dim: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        debug_assert_eq!(w.data.len(), out_dim * in_dim);
        // sm_100: the warp-specialized tcgen05 pipeline (slot 870) - the per-tile
        // pd_f8row_gemm_tc5_kt streamed these planes at ~1.2 TB/s in the serve.
        // PADDOCK_NO_F8ROW_TC5X=1 keeps the older arm for A/B.
        static NO_ROW5X: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let no_row5x = *NO_ROW5X
            .get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_F8ROW_TC5X").is_some());
        if let Some(f) = self
            .kernels
            .f8row_gemm_tc5x
            .filter(|_| !no_row5x && in_dim.is_multiple_of(128))
        {
            let (wdp, _g1) = w.data.device_ptr(&self.stream);
            let (wsp, _g2) = w.scale.device_ptr(&self.stream);
            let (xqp, _g3) = xq.device_ptr(&self.stream);
            let (xsp, _g4) = xrs.device_ptr(&self.stream);
            let (yp, _g5) = y.device_ptr_mut(&self.stream);
            // SAFETY: ABI contract (slot 870); the same operands as slot f8row_gemm
            return check(unsafe {
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
            });
        }
        let f = self
            .kernels
            .f8row_gemm
            .ok_or(GpuError::MissingOp("f8row_gemm"))?;
        let (wdp, _g1) = w.data.device_ptr(&self.stream);
        let (wsp, _g2) = w.scale.device_ptr(&self.stream);
        let (xqp, _g3) = xq.device_ptr(&self.stream);
        let (xsp, _g4) = xrs.device_ptr(&self.stream);
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

    pub fn has_f8t_gemm(&self) -> bool {
        self.kernels.f8t_gemm.is_some()
            && self.kernels.f8_repack_tiles.is_some()
            && self.kernels.q8_0_to_f8row.is_some()
            && self.kernels.quantize_e4m3_row.is_some()
    }

    /// f8r plane (bf16-native ingestion) -> f8t tile plane, for the sm_100
    /// tcgen05 lane.
    ///
    /// The two rowwise planes carry the same e4m3 payload under the same
    /// normalization (amax/2^e <= 448, weights pre-divided by 2^e); they differ
    /// only in how the per-output-row scale is ENCODED:
    ///   q8_0_to_f8row -> f32  rscale[row] = 2^e
    ///   bf16_to_f8r   -> e8m0 scale[row]  = e + 127   (multiplier 2^(b-127))
    /// so the transcode is exact, and f8_repack_tiles rewrites only `data` and
    /// carries the scale through untouched. Done once per plane at load, on the
    /// host: out_dim is at most a few 10k scalars, which is not worth a kernel.
    ///
    /// This is what lets the qwen families reach the same tcgen05 GEMM gemma4
    /// already uses on B200-class dies. Measured there, gemma4 - the only
    /// family wired to f8t - was far ahead of qwen35/qwen36 on the warp-level
    /// f8r path, which is what this repack closes.
    pub fn f8r_to_tiles(
        &self,
        w: RepackedMxfp4,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<F8TilePlane, GpuError> {
        debug_assert_eq!(w.data.len(), in_dim * out_dim);
        debug_assert_eq!(w.scale.len(), out_dim);
        let e8m0 = self.stream.clone_dtoh(&w.scale).map_err(drv)?;
        // b is the biased exponent, and an f32 whose value is exactly 2^(b-127)
        // is (b << 23) with a zero mantissa - the same 127 bias - so the
        // transcode is a bit shift, not an exp2 call. b == 0 would mean 2^-127,
        // which the producer never emits (an all-zero row takes the m <= 0
        // branch and stores e = 0 -> b = 127 -> 1.0), and (0 << 23) is +0.0,
        // which is the correct reading of a zero scale anyway.
        let f32s: Vec<f32> = e8m0
            .iter()
            .map(|&b| f32::from_bits((b as u32) << 23))
            .collect();
        let scale: CudaSlice<f32> = self.stream.clone_htod(&f32s).map_err(drv)?;
        self.f8_repack_tiles(
            F8RowPlane {
                data: w.data,
                scale,
            },
            in_dim,
            out_dim,
        )
    }

    /// Bake a rowwise plane's SW128 tile image (consumes the row-major data;
    /// the per-row scales carry over unchanged). Dims must be 128-multiples.
    pub fn f8_repack_tiles(
        &self,
        w: F8RowPlane,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<F8TilePlane, GpuError> {
        let f = self
            .kernels
            .f8_repack_tiles
            .ok_or(GpuError::MissingOp("f8_repack_tiles"))?;
        debug_assert_eq!(w.data.len(), in_dim * out_dim);
        let mut tiles = self.alloc_u8(in_dim * out_dim)?;
        {
            let (sp, _g1) = w.data.device_ptr(&self.stream);
            let (dp, _g2) = tiles.device_ptr_mut(&self.stream);
            // SAFETY: ABI contract; equal-size planes, 128-multiple dims
            check(unsafe {
                f(
                    sp as *const _,
                    dp as *mut _,
                    in_dim as u32,
                    out_dim as u32,
                    self.stream_ptr(),
                )
            })?;
        }
        Ok(F8TilePlane {
            tiles,
            scale: w.scale,
            flat: None,
            flat_minb: 0,
            flat_gui: false,
            scale_il: None,
        })
    }

    //  (closed, finding banked): cuBLASLt could not serve paddock's
    // per-row fp8 on sm_100/CUDA 13 - OUTER_VEC scale mode returns
    // NOT_SUPPORTED; only per-tensor SCALAR runs (the nvjet path). The rowwise
    // library floor is the vendored cutlass (F8CUT) below. The per-tensor
    // cuBLASLt demonstrator that lived here died with the phase-C deletion.

    /// Rowwise tcgen05 decode GEMM over the tile plane (r <= 64 band).
    /// `xq` must be a >= 64-row buffer (64-row TMA boxes read past `batch`;
    /// stale tail rows only feed D columns the epilogue never stores).
    /// `part` is the K-split partial scratch (out_dim * batch * 8 floats max).
    #[allow(clippy::too_many_arguments)]
    /// Row-tile-offset sub-view of a concatenated f8t plane (the qkv-concat
    /// prefill arms): tiles advance row_tile_off * (in/128) * 16KB, scales
    /// row_tile_off * 128 rows. Same launcher (tc5p/tc5q <=64, tc5r >=65).
    #[allow(clippy::too_many_arguments)]
    pub fn f8t_gemm_off(
        &self,
        w: &F8TilePlane,
        row_tile_off: usize,
        xq: &CudaSlice<i8>,
        xrs: &CudaSlice<f32>,
        part: &mut CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        in_dim: usize,
        out_dim: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        if paddock_models::dev_var_os!("PADDOCK_F8CUT_DBG").is_some() && batch >= 1024 {
            eprintln!(
                "[f8cut-dbg] off={} out={} in={} m={} flat={} minb={}",
                row_tile_off,
                out_dim,
                in_dim,
                batch,
                w.flat.is_some(),
                w.flat_minb
            );
        }
        let f = self
            .kernels
            .f8t_gemm
            .ok_or(GpuError::MissingOp("f8t_gemm"))?;
        let nk = in_dim / 128;
        debug_assert!((row_tile_off * 128 + out_dim) * in_dim <= w.tiles.len());
        let (wp, _g1) = w.tiles.device_ptr(&self.stream);
        let (sp, _g2) = w.scale.device_ptr(&self.stream);
        let (xqp, _g3) = xq.device_ptr(&self.stream);
        let (xsp, _g4) = xrs.device_ptr(&self.stream);
        let (pp, _g5) = part.device_ptr_mut(&self.stream);
        let (yp, _g6) = y.device_ptr_mut(&self.stream);
        let wo = wp + (row_tile_off * nk * 16384) as u64;
        let so = sp + (row_tile_off * 128 * 4) as u64;
        // SAFETY: ABI contract; sub-view bounds asserted above
        check(unsafe {
            f(
                wo as *const _,
                so as *const _,
                xqp as *const _,
                xsp as *const _,
                pp as *mut _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
        .inspect_err(|_| {
            tracing::warn!(
                "f8t_gemm_off refused: out_dim={out_dim} in_dim={in_dim} batch={batch} row_tile_off={row_tile_off}"
            )
        })
    }

    /// register (Some) or clear (None) the dim-major twin V pool.
    /// The append kernels and the v9q VD launcher capture the registered
    /// base at launch-enqueue time, so call this per layer before appends.
    pub fn vdim_set(&self, vdim: Option<&CudaSlice<u8>>) -> Result<(), GpuError> {
        let f = self
            .kernels
            .vdim_register
            .ok_or(GpuError::MissingOp("vdim_register"))?;
        let vp = vdim.map(|v| v.device_ptr(&self.stream));
        // SAFETY: ABI contract (slot 375); host-side pointer store only
        check(unsafe {
            f(vp.as_ref()
                .map_or(core::ptr::null_mut(), |(p, _)| *p as *mut _))
        })
    }

    /// transpose freshly appended V rows into the dim-major twin.
    #[allow(clippy::too_many_arguments)]
    pub fn vdim_sync(
        &self,
        pool: &CudaSlice<u8>,
        vdim: &CudaSlice<u8>,
        positions: &CudaSlice<u32>,
        slots: Option<&CudaSlice<u32>>,
        block_tables: Option<(&CudaSlice<u32>, usize)>,
        kv_dim: usize,
        rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .vdim_sync
            .ok_or(GpuError::MissingOp("vdim_sync"))?;
        let (pp, _g1) = pool.device_ptr(&self.stream);
        let (vp, _g2) = vdim.device_ptr(&self.stream);
        let (op, _g3) = positions.device_ptr(&self.stream);
        let sp = slots.map(|s| s.device_ptr(&self.stream));
        let bp = block_tables.map(|(b, _)| b.device_ptr(&self.stream));
        let bps = block_tables.map_or(0, |(_, n)| n) as u32;
        // SAFETY: ABI contract (slot 374)
        check(unsafe {
            f(
                pp as *const _,
                vp as *mut _,
                op as *const _,
                sp.as_ref()
                    .map_or(core::ptr::null(), |(p, _)| *p as *const _),
                bp.as_ref()
                    .map_or(core::ptr::null(), |(p, _)| *p as *const _),
                bps,
                kv_dim as u32,
                rows as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn f8t_gemm(
        &self,
        w: &F8TilePlane,
        xq: &CudaSlice<i8>,
        xrs: &CudaSlice<f32>,
        part: &mut CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        in_dim: usize,
        out_dim: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        debug_assert_eq!(w.tiles.len(), out_dim * in_dim);
        // Whole-plane call == sub-view at row-tile 0; the _off wrapper owns
        // both routes (flat cutlass intercept + classic tc5 launcher).
        self.f8t_gemm_off(w, 0, xq, xrs, part, y, in_dim, out_dim, batch)
    }

    /// No-combine f8t GEMM: leaves the nz partial planes in `part`
    /// and returns nz (1 = y already final). The out-param is written
    /// host-side by the launcher before return - no sync needed.
    #[allow(clippy::too_many_arguments)]
    pub fn f8t_gemm_nc(
        &self,
        w: &F8TilePlane,
        xq: &CudaSlice<i8>,
        xrs: &CudaSlice<f32>,
        part: &mut CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        in_dim: usize,
        out_dim: usize,
        batch: usize,
    ) -> Result<u32, GpuError> {
        let f = self
            .kernels
            .f8t_gemm2
            .ok_or(GpuError::MissingOp("f8t_gemm2"))?;
        debug_assert_eq!(w.tiles.len(), out_dim * in_dim);
        let (wp, _g1) = w.tiles.device_ptr(&self.stream);
        let (sp, _g2) = w.scale.device_ptr(&self.stream);
        let (xqp, _g3) = xq.device_ptr(&self.stream);
        let (xsp, _g4) = xrs.device_ptr(&self.stream);
        let (pp, _g5) = part.device_ptr_mut(&self.stream);
        let (yp, _g6) = y.device_ptr_mut(&self.stream);
        let mut nz_out: u32 = 1;
        // SAFETY: ABI contract; out_nz is a host pointer the launcher fills
        // synchronously
        check(unsafe {
            f(
                wp as *const _,
                sp as *const _,
                xqp as *const _,
                xsp as *const _,
                pp as *mut _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                1u32,
                &mut nz_out as *mut u32,
                self.stream_ptr(),
            )
        })?;
        Ok(nz_out)
    }

    pub fn has_f8t_gemm_nc(&self) -> bool {
        self.kernels.f8t_gemm2.is_some()
            && self.kernels.addnorm_e4m3_nz.is_some()
            && self.kernels.quantize_e4m3_geglu2_nz.is_some()
    }

    /// nz-aware addnorm - `proj` holds the GEMM's nz partial planes.
    #[allow(clippy::too_many_arguments)]
    pub fn addnorm_e4m3_row_nz(
        &self,
        x: &mut CudaSlice<f32>,
        proj: &CudaSlice<f32>,
        postw: &CudaSlice<f32>,
        prew: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        rscale: &mut CudaSlice<f32>,
        n: usize,
        eps: f32,
        s: f32,
        rows: usize,
        nzp: u32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .addnorm_e4m3_nz
            .ok_or(GpuError::MissingOp("addnorm_e4m3_nz"))?;
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (pp, _g2) = proj.device_ptr(&self.stream);
        let (pwp, _g3) = postw.device_ptr(&self.stream);
        let (prp, _g4) = prew.device_ptr(&self.stream);
        let (qp, _g5) = q.device_ptr_mut(&self.stream);
        let (rp, _g6) = rscale.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract
        check(unsafe {
            f(
                xp as *mut _,
                pp as *const _,
                pwp as *const _,
                prp as *const _,
                qp as *mut _,
                rp as *mut _,
                n as u32,
                eps,
                s,
                rows as u32,
                nzp,
                self.stream_ptr(),
            )
        })
    }

    /// nz-aware fused glu2 quant - `gu` holds nz partial planes.
    #[allow(clippy::too_many_arguments)]
    pub fn quantize_e4m3_glu2_row_nz(
        &self,
        gu: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        rscale: &mut CudaSlice<f32>,
        n_ff: usize,
        rows: usize,
        nzp: u32,
        act: GluAct,
    ) -> Result<(), GpuError> {
        let (f, name) = match act {
            GluAct::Gelu => (
                self.kernels.quantize_e4m3_geglu2_nz,
                "quantize_e4m3_geglu2_nz",
            ),
            GluAct::Silu => (
                self.kernels.quantize_e4m3_swiglu2_nz,
                "quantize_e4m3_swiglu2_nz",
            ),
        };
        let f = f.ok_or(GpuError::MissingOp(name))?;
        let (gp, _g1) = gu.device_ptr(&self.stream);
        let (qp, _g2) = q.device_ptr_mut(&self.stream);
        let (rp, _g3) = rscale.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract
        check(unsafe {
            f(
                gp as *const _,
                qp as *mut _,
                rp as *mut _,
                n_ff as u32,
                rows as u32,
                nzp,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_quantize_e4m3_glu2_row(&self, act: GluAct) -> bool {
        match act {
            GluAct::Gelu => self.kernels.quantize_e4m3_geglu2_row.is_some(),
            GluAct::Silu => self.kernels.quantize_e4m3_swiglu2_row.is_some(),
        }
    }

    pub fn quantize_e4m3_glu2_row_b16(
        &self,
        gu: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        rs: &mut CudaSlice<f32>,
        n_ff: usize,
        rows: usize,
        act: GluAct,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .quantize_e4m3_glu2_row_b16
            .ok_or(GpuError::MissingOp("quantize_e4m3_glu2_row_b16"))?;
        let acti = match act {
            GluAct::Gelu => 0u32,
            GluAct::Silu => 1u32,
        };
        let (gp, _g1) = gu.device_ptr(&self.stream);
        let (qp, _g2) = q.device_ptr_mut(&self.stream);
        let (sp, _g3) = rs.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 378); n_ff % 32 checked by the launcher
        check(unsafe {
            f(
                gp as *const _,
                qp as *mut _,
                sp as *mut _,
                n_ff as u32,
                rows as u32,
                acti,
                self.stream_ptr(),
            )
        })
    }

    pub fn quantize_e4m3_glu2_row(
        &self,
        gu: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        rs: &mut CudaSlice<f32>,
        n_ff: usize,
        rows: usize,
        act: GluAct,
    ) -> Result<(), GpuError> {
        let (f, name) = match act {
            GluAct::Gelu => (
                self.kernels.quantize_e4m3_geglu2_row,
                "quantize_e4m3_geglu2_row",
            ),
            GluAct::Silu => (
                self.kernels.quantize_e4m3_swiglu2_row,
                "quantize_e4m3_swiglu2_row",
            ),
        };
        let f = f.ok_or(GpuError::MissingOp(name))?;
        let (gp, _g1) = gu.device_ptr(&self.stream);
        let (qp, _g2) = q.device_ptr_mut(&self.stream);
        let (sp, _g3) = rs.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; n_ff % 32 == 0 checked by the launcher
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

    pub fn has_rmsnorm_e4m3(&self) -> bool {
        self.kernels.rmsnorm_e4m3.is_some()
    }

    /// Fused rmsnorm + per-32 e4m3 quantize over `rows` rows of width `n`.
    pub fn rmsnorm_e4m3(
        &self,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        scale: &mut CudaSlice<u8>,
        n: usize,
        eps: f32,
        rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .rmsnorm_e4m3
            .ok_or(GpuError::MissingOp("rmsnorm_e4m3"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let (qp, _g3) = q.device_ptr_mut(&self.stream);
        let (sp, _g4) = scale.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; n % 32 == 0, buffers sized [rows*n]/[rows*n/32]
        check(unsafe {
            f(
                xp as *const _,
                wp as *const _,
                qp as *mut _,
                sp as *mut _,
                n as u32,
                eps,
                rows as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Fused q-side per-head RMSNorm + YaRN rope - bit-exact with the
    /// rmsnorm_batch -> rope_yarn_batch pair. head_dim must be 128.
    #[allow(clippy::too_many_arguments)]
    pub fn q_norm_rope(
        &self,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        n_heads: usize,
        head_dim: usize,
        eps: f32,
        rope: (f32, f32, f32, f32, f32, f32),
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q_norm_rope
            .ok_or(GpuError::MissingOp("q_norm_rope"))?;
        let (theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale) = rope;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let (op, _g3) = out.device_ptr_mut(&self.stream);
        let (pp, _g4) = positions.device_ptr(&self.stream);
        check(unsafe {
            f(
                xp as *const _,
                wp as *const _,
                op as *mut _,
                pp as *const _,
                n_heads as u32,
                head_dim as u32,
                eps,
                theta_scale,
                freq_scale,
                corr_low,
                corr_high,
                ext_factor,
                mscale,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Fused k-side per-head RMSNorm + YaRN rope + KV-cache scatter -
    /// replaces norm + rope + kv_append_batch. head_dim must be 128.
    #[allow(clippy::too_many_arguments)]
    pub fn k_norm_rope_append(
        &self,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        cache: &mut CudaSlice<u8>,
        positions: &CudaSlice<u32>,
        slots: Option<&CudaSlice<u32>>,
        n_kv_heads: usize,
        head_dim: usize,
        max_ctx: usize,
        eps: f32,
        rope: (f32, f32, f32, f32, f32, f32),
        batch: usize,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .k_norm_rope_append
            .ok_or(GpuError::MissingOp("k_norm_rope_append"))?;
        let (theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale) = rope;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let (cp, _g3) = cache.device_ptr_mut(&self.stream);
        let (pp, _g4) = positions.device_ptr(&self.stream);
        let sp = slots.map(|s| s.device_ptr(&self.stream));
        check(unsafe {
            f(
                xp as *const _,
                wp as *const _,
                cp as *mut _,
                pp as *const _,
                sp.as_ref()
                    .map_or(std::ptr::null(), |(p, _)| *p as *const _),
                n_kv_heads as u32,
                head_dim as u32,
                max_ctx as u32,
                eps,
                theta_scale,
                freq_scale,
                corr_low,
                corr_high,
                ext_factor,
                mscale,
                batch as u32,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Fused batched rmsnorm -> e4m3 quantize: normed f32 never lands.
    /// Bit-identical to rmsnorm_batch + quantize_e4m3 (same width policy).
    pub fn rmsnorm_e4m3_batch(
        &self,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        scale: &mut CudaSlice<u8>,
        n: usize,
        batch: usize,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .rmsnorm_e4m3_batch
            .ok_or(GpuError::MissingOp("rmsnorm_e4m3_batch"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let (qp, _g3) = q.device_ptr_mut(&self.stream);
        let (sp, _g4) = scale.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; n % 32 == 0, q/scale sized [batch*n]/[batch*n/32]
        check(unsafe {
            f(
                xp as *const _,
                wp as *const _,
                qp as *mut _,
                sp as *mut _,
                n as u32,
                batch as u32,
                eps,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_rmsnorm_e4m3_batch(&self) -> bool {
        self.kernels.rmsnorm_e4m3_batch.is_some()
    }

    /// Fused rmsnorm -> ROW-scale e4m3 (the f8t decode band's activation
    /// format). Bit-identical to rmsnorm_batch + quantize_e4m3_row.
    pub fn rmsnorm_e4m3_row(
        &self,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        rs: &mut CudaSlice<f32>,
        n: usize,
        eps: f32,
        rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .rmsnorm_e4m3_row
            .ok_or(GpuError::MissingOp("rmsnorm_e4m3_row"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let (qp, _g3) = q.device_ptr_mut(&self.stream);
        let (sp, _g4) = rs.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; n % 4 == 0 checked by the launcher
        check(unsafe {
            f(
                xp as *const _,
                wp as *const _,
                qp as *mut _,
                sp as *mut _,
                n as u32,
                eps,
                rows as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_rmsnorm_e4m3_row(&self) -> bool {
        self.kernels.rmsnorm_e4m3_row.is_some()
    }

    /// Band-boundary fusion: x = (x + rmsnorm(proj)*post_w)*s, then
    /// e4m3(rmsnorm(x)*pre_w) with a row scale - one kernel for the 3-launch
    /// decode chain. Bit-identical to the chain at the same width.
    #[allow(clippy::too_many_arguments)]
    pub fn addnorm_e4m3_row(
        &self,
        x: &mut CudaSlice<f32>,
        proj: &CudaSlice<f32>,
        post_w: &CudaSlice<f32>,
        pre_w: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        rs: &mut CudaSlice<f32>,
        n: usize,
        eps: f32,
        s: f32,
        rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .addnorm_e4m3_row
            .ok_or(GpuError::MissingOp("addnorm_e4m3_row"))?;
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (pp, _g2) = proj.device_ptr(&self.stream);
        let (pow, _g3) = post_w.device_ptr(&self.stream);
        let (prw, _g4) = pre_w.device_ptr(&self.stream);
        let (qp, _g5) = q.device_ptr_mut(&self.stream);
        let (sp, _g6) = rs.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; n % 4 == 0 checked by the launcher
        check(unsafe {
            f(
                xp as *mut _,
                pp as *const _,
                pow as *const _,
                prw as *const _,
                qp as *mut _,
                sp as *mut _,
                n as u32,
                eps,
                s,
                rows as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_addnorm_e4m3_row(&self) -> bool {
        self.kernels.addnorm_e4m3_row.is_some()
    }

    /// qwen twin: PLAIN residual add + pre-norm + row-e4m3 in one launch.
    /// Bit-identical to `add_rmsnorm_batch` + `quantize_e4m3_row`, so it needs
    /// no precision gate -- it only removes a launch and a f32 round trip.
    #[allow(clippy::too_many_arguments)]
    pub fn add_rmsnorm_e4m3_row(
        &self,
        x: &mut CudaSlice<f32>,
        proj: &CudaSlice<f32>,
        pre_w: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        rs: &mut CudaSlice<f32>,
        n: usize,
        eps: f32,
        rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .add_rmsnorm_e4m3_row
            .ok_or(GpuError::MissingOp("add_rmsnorm_e4m3_row"))?;
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (pp, _g2) = proj.device_ptr(&self.stream);
        let (wp, _g3) = pre_w.device_ptr(&self.stream);
        let (qp, _g4) = q.device_ptr_mut(&self.stream);
        let (sp, _g5) = rs.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; n % 4 == 0 checked by the launcher
        check(unsafe {
            f(
                xp as *mut _,
                pp as *const _,
                wp as *const _,
                qp as *mut _,
                sp as *mut _,
                n as u32,
                eps,
                rows as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_add_rmsnorm_e4m3_row(&self) -> bool {
        self.kernels.add_rmsnorm_e4m3_row.is_some()
    }

    /// Per-32 twin of `addnorm_e4m3_row` (the f8a/f8r wide-decode band):
    /// x = (x + rmsnorm(proj)*post_w)*s, then per-32 e4m3(rmsnorm(x)*pre_w).
    /// Bit-identical to rmsnorm_add_scale -> rmsnorm_e4m3_batch.
    #[allow(clippy::too_many_arguments)]
    pub fn addnorm_e4m3_b32(
        &self,
        x: &mut CudaSlice<f32>,
        proj: &CudaSlice<f32>,
        post_w: &CudaSlice<f32>,
        pre_w: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        scale: &mut CudaSlice<u8>,
        n: usize,
        eps: f32,
        s: f32,
        rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .addnorm_e4m3_b32
            .ok_or(GpuError::MissingOp("addnorm_e4m3_b32"))?;
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (pp, _g2) = proj.device_ptr(&self.stream);
        let (pow, _g3) = post_w.device_ptr(&self.stream);
        let (prw, _g4) = pre_w.device_ptr(&self.stream);
        let (qp, _g5) = q.device_ptr_mut(&self.stream);
        let (sp, _g6) = scale.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; n % 32 == 0, q/scale sized [rows*n]/[rows*n/32]
        check(unsafe {
            f(
                xp as *mut _,
                pp as *const _,
                pow as *const _,
                prw as *const _,
                qp as *mut _,
                sp as *mut _,
                n as u32,
                eps,
                s,
                rows as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_addnorm_e4m3_b32(&self) -> bool {
        self.kernels.addnorm_e4m3_b32.is_some()
    }

    /// Fused FlashDecoding combine + per-ROW e4m3 quant: the wo input never
    /// lands in f32. Bit-identical to attn_combine_batch + quantize_e4m3_row.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_combine_e4m3_row(
        &self,
        in_o: &CudaSlice<f32>,
        in_ml: &CudaSlice<f32>,
        sinks: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        rs: &mut CudaSlice<f32>,
        n_heads: usize,
        head_dim: usize,
        n_splits: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .attn_combine_e4m3_row
            .ok_or(GpuError::MissingOp("attn_combine_e4m3_row"))?;
        let (op, _g1) = in_o.device_ptr(&self.stream);
        let (mp, _g2) = in_ml.device_ptr(&self.stream);
        let (kp, _g3) = sinks.device_ptr(&self.stream);
        let (qp, _g4) = q.device_ptr_mut(&self.stream);
        let (sp, _g5) = rs.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; (n_heads*head_dim) % 4 == 0 by construction
        check(unsafe {
            f(
                op as *const _,
                mp as *const _,
                kp as *const _,
                qp as *mut _,
                sp as *mut _,
                n_heads as u32,
                head_dim as u32,
                n_splits as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_attn_combine_e4m3_row(&self) -> bool {
        self.kernels.attn_combine_e4m3_row.is_some()
    }

    /// Fused prefill QKV epilogue norms + rope (five launches -> one).
    /// Bit-identical to rmsnorm_batch x3 + rope_factors_batch x2.
    #[allow(clippy::too_many_arguments)]
    pub fn qkv_norm_rope_batch(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        qw: &CudaSlice<f32>,
        kw: &CudaSlice<f32>,
        qn: &mut CudaSlice<f32>,
        kn: &mut CudaSlice<f32>,
        vn: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        factors: Option<&CudaSlice<f32>>,
        n_head: usize,
        n_kv: usize,
        head_dim: usize,
        eps: f32,
        params: (f32, f32, f32, f32, f32, f32),
        rows: usize,
        neox: bool,
        vnorm: bool,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .qkv_norm_rope_batch5
            .ok_or(GpuError::MissingOp("qkv_norm_rope_batch5"))?;
        let (theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale) = params;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = k.device_ptr(&self.stream);
        let (vp, _g3) = v.device_ptr(&self.stream);
        let (qwp, _g4) = qw.device_ptr(&self.stream);
        let (kwp, _g5) = kw.device_ptr(&self.stream);
        let (qnp, _g6) = qn.device_ptr_mut(&self.stream);
        let (knp, _g7) = kn.device_ptr_mut(&self.stream);
        let (vnp, _g8) = vn.device_ptr_mut(&self.stream);
        let (pp, _g9) = positions.device_ptr(&self.stream);
        let fac_guard = factors.map(|s| s.device_ptr(&self.stream));
        let fp2 = match &fac_guard {
            Some((p, _)) => *p as *const core::ffi::c_void,
            None => std::ptr::null(),
        };
        // SAFETY: ABI contract; buffers sized [rows*heads*head_dim]
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                qwp as *const _,
                kwp as *const _,
                qnp as *mut _,
                knp as *mut _,
                vnp as *mut _,
                pp as *const _,
                fp2,
                n_head as u32,
                n_kv as u32,
                head_dim as u32,
                eps,
                theta_scale,
                freq_scale,
                corr_low,
                corr_high,
                ext_factor,
                mscale,
                rows as u32,
                0u32,
                0u32,
                neox as u32,
                vnorm as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_qkv_norm_rope_batch(&self) -> bool {
        self.kernels.qkv_norm_rope_batch5.is_some()
    }

    /// i16 twin: `q`/`k`/`v` hold bf16 (the o16 GEMM epilogue's
    /// stream, riding the f32-typed scratch at half occupancy); outputs stay
    /// f32. Same math class as the f32 form on bf16-rounded inputs.
    #[allow(clippy::too_many_arguments)]
    pub fn qkv_norm_rope_batch_i16(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        qw: &CudaSlice<f32>,
        kw: &CudaSlice<f32>,
        qn: &mut CudaSlice<f32>,
        kn: &mut CudaSlice<f32>,
        vn: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        factors: Option<&CudaSlice<f32>>,
        n_head: usize,
        n_kv: usize,
        head_dim: usize,
        eps: f32,
        params: (f32, f32, f32, f32, f32, f32),
        rows: usize,
        neox: bool,
        vnorm: bool,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .qkv_norm_rope_batch5
            .ok_or(GpuError::MissingOp("qkv_norm_rope_batch5"))?;
        let (theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale) = params;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = k.device_ptr(&self.stream);
        let (vp, _g3) = v.device_ptr(&self.stream);
        let (qwp, _g4) = qw.device_ptr(&self.stream);
        let (kwp, _g5) = kw.device_ptr(&self.stream);
        let (qnp, _g6) = qn.device_ptr_mut(&self.stream);
        let (knp, _g7) = kn.device_ptr_mut(&self.stream);
        let (vnp, _g8) = vn.device_ptr_mut(&self.stream);
        let (pp, _g9) = positions.device_ptr(&self.stream);
        let fac_guard = factors.map(|s| s.device_ptr(&self.stream));
        let fp2 = match &fac_guard {
            Some((p, _)) => *p as *const core::ffi::c_void,
            None => std::ptr::null(),
        };
        // SAFETY: ABI contract; q/k/v hold bf16 in the f32-typed scratch
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                qwp as *const _,
                kwp as *const _,
                qnp as *mut _,
                knp as *mut _,
                vnp as *mut _,
                pp as *const _,
                fp2,
                n_head as u32,
                n_kv as u32,
                head_dim as u32,
                eps,
                theta_scale,
                freq_scale,
                corr_low,
                corr_high,
                ext_factor,
                mscale,
                rows as u32,
                1u32,
                0u32,
                neox as u32,
                vnorm as u32,
                self.stream_ptr(),
            )
        })
    }

    /// a16 twin (attention streams): o16=1 writes the f16 q plane
    /// (v3 register form only - one rounding at the store); i16 keeps the
    /// bf16-in meaning. The fused-kv serve path runs this q-only
    /// (n_kv = 0), so kn/vn are never written on the a16 route.
    #[allow(clippy::too_many_arguments)]
    pub fn qkv_norm_rope_batch_a16(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        qw: &CudaSlice<f32>,
        kw: &CudaSlice<f32>,
        qn: &mut CudaSlice<f32>,
        kn: &mut CudaSlice<f32>,
        vn: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        factors: Option<&CudaSlice<f32>>,
        n_head: usize,
        n_kv: usize,
        head_dim: usize,
        eps: f32,
        params: (f32, f32, f32, f32, f32, f32),
        rows: usize,
        i16: bool,
        neox: bool,
        vnorm: bool,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .qkv_norm_rope_batch5
            .ok_or(GpuError::MissingOp("qkv_norm_rope_batch5"))?;
        let (theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale) = params;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = k.device_ptr(&self.stream);
        let (vp, _g3) = v.device_ptr(&self.stream);
        let (qwp, _g4) = qw.device_ptr(&self.stream);
        let (kwp, _g5) = kw.device_ptr(&self.stream);
        let (qnp, _g6) = qn.device_ptr_mut(&self.stream);
        let (knp, _g7) = kn.device_ptr_mut(&self.stream);
        let (vnp, _g8) = vn.device_ptr_mut(&self.stream);
        let (pp, _g9) = positions.device_ptr(&self.stream);
        let fac_guard = factors.map(|s| s.device_ptr(&self.stream));
        let fp2 = match &fac_guard {
            Some((p, _)) => *p as *const core::ffi::c_void,
            None => std::ptr::null(),
        };
        // SAFETY: ABI contract; qn holds f16 in the f32-typed scratch
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                qwp as *const _,
                kwp as *const _,
                qnp as *mut _,
                knp as *mut _,
                vnp as *mut _,
                pp as *const _,
                fp2,
                n_head as u32,
                n_kv as u32,
                head_dim as u32,
                eps,
                theta_scale,
                freq_scale,
                corr_low,
                corr_high,
                ext_factor,
                mscale,
                rows as u32,
                i16 as u32,
                1u32,
                neox as u32,
                vnorm as u32,
                self.stream_ptr(),
            )
        })
    }

    /// f16-in twin of [`Self::quantize_e4m3_row`]: x is an f16
    /// plane held in the f32-typed scratch.
    pub fn quantize_e4m3_row_f16in(
        &self,
        x: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        rs: &mut CudaSlice<f32>,
        n_dim: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .quantize_e4m3_row_i16
            .ok_or(GpuError::MissingOp("quantize_e4m3_row_i16"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (qp, _g2) = q.device_ptr_mut(&self.stream);
        let (rp, _g3) = rs.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; x holds f16 in the f32-typed scratch
        check(unsafe {
            f(
                xp as *const _,
                qp as *mut _,
                rp as *mut _,
                n_dim as u32,
                batch as u32,
                1u32,
                self.stream_ptr(),
            )
        })
    }

    /// o16 twin of `f8_gemm_w8_pc`: the epilogue stores bf16 into
    /// `y` (f32-typed scratch, half occupancy). Same mainloop, only the final
    /// store converts. Ok(false) = route not covered.
    #[allow(clippy::too_many_arguments)]
    pub fn f8_gemm_w8_pc_o16(
        &self,
        w: &RepackedMxfp4,
        row_off: usize,
        xq: &CudaSlice<i8>,
        as_row: &CudaSlice<f32>,
        ws: &CudaSlice<f32>,
        ws_off: usize,
        y: &mut CudaSlice<f32>,
        in_dim: usize,
        out_dim: usize,
        batch: usize,
    ) -> Result<bool, GpuError> {
        let f = if w.scale.len() == 12 {
            self.kernels
                .f8_gemm_w8_pc_r
                .ok_or(GpuError::MissingOp("f8_gemm_w8_pc_r"))?
        } else {
            self.kernels
                .f8_gemm_w8_pc
                .ok_or(GpuError::MissingOp("f8_gemm_w8_pc"))?
        };
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (xqp, _g2) = xq.device_ptr(&self.stream);
        let (asp, _g3) = as_row.device_ptr(&self.stream);
        let (wsp, _g4) = ws.device_ptr(&self.stream);
        let (yp, _g5) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; ws sliced by ws_off covers out_dim rows
        let rc = unsafe {
            f(
                dp as *const _,
                row_off as u32,
                xqp as *const _,
                asp as *const _,
                (wsp + (ws_off * 4) as u64) as *const _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                1u32,
                self.stream_ptr(),
            )
        };
        if rc == -2 {
            return Ok(false);
        }
        check(rc)?;
        Ok(true)
    }

    /// o16 twin of `f8_gemm_w8_pc_qkv`: one launch, bf16 stores
    /// into all three projections. Ok(false) = route not covered.
    #[allow(clippy::too_many_arguments)]
    pub fn f8_gemm_w8_pc_qkv_o16(
        &self,
        w: &RepackedMxfp4,
        xq: &CudaSlice<i8>,
        as_row: &CudaSlice<f32>,
        ws: &CudaSlice<f32>,
        yq: &mut CudaSlice<f32>,
        yk: &mut CudaSlice<f32>,
        yv: &mut CudaSlice<f32>,
        in_dim: usize,
        q_dim: usize,
        kv_dim: usize,
        batch: usize,
    ) -> Result<bool, GpuError> {
        if w.scale.len() != 12 {
            return Ok(false);
        }
        let f = self
            .kernels
            .f8_gemm_w8_pc_qkv_r2
            .ok_or(GpuError::MissingOp("f8_gemm_w8_pc_qkv_r2"))?;
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (xqp, _g2) = xq.device_ptr(&self.stream);
        let (asp, _g3) = as_row.device_ptr(&self.stream);
        let (wsp, _g4) = ws.device_ptr(&self.stream);
        let (yqp, _g5) = yq.device_ptr_mut(&self.stream);
        let (ykp, _g6) = yk.device_ptr_mut(&self.stream);
        let (yvp, _g7) = yv.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; ws covers the full q_dim + 2*kv_dim plane
        let rc = unsafe {
            f(
                dp as *const _,
                xqp as *const _,
                asp as *const _,
                wsp as *const _,
                yqp as *mut _,
                ykp as *mut _,
                yvp as *mut _,
                in_dim as u32,
                q_dim as u32,
                kv_dim as u32,
                batch as u32,
                1u32,
                self.stream_ptr(),
            )
        };
        if rc == -2 {
            return Ok(false);
        }
        check(rc)?;
        Ok(true)
    }

    /// o16 twin of `f8_gemm_w8_pcd`. Ok(false) = not covered.
    #[allow(clippy::too_many_arguments)]
    pub fn f8_gemm_w8_pcd_o16(
        &self,
        w: &RepackedMxfp4,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<u8>,
        ws: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        in_dim: usize,
        out_dim: usize,
        batch: usize,
    ) -> Result<bool, GpuError> {
        let f = if w.scale.len() == 12 {
            self.kernels
                .f8_gemm_w8_pcd_r
                .ok_or(GpuError::MissingOp("f8_gemm_w8_pcd_r"))?
        } else {
            self.kernels
                .f8_gemm_w8_pcd
                .ok_or(GpuError::MissingOp("f8_gemm_w8_pcd"))?
        };
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (xqp, _g2) = xq.device_ptr(&self.stream);
        let (xsp, _g3) = xs.device_ptr(&self.stream);
        let (wsp, _g4) = ws.device_ptr(&self.stream);
        let (yp, _g5) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; ws covers out_dim rows
        let rc = unsafe {
            f(
                dp as *const _,
                0u32,
                xqp as *const _,
                xsp as *const _,
                wsp as *const _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                1u32,
                self.stream_ptr(),
            )
        };
        if rc == -2 {
            return Ok(false);
        }
        check(rc)?;
        Ok(true)
    }

    /// p16 twin of `addnorm_e4m3_row`: `proj` holds bf16.
    #[allow(clippy::too_many_arguments)]
    pub fn addnorm_e4m3_row_p16(
        &self,
        x: &mut CudaSlice<f32>,
        proj: &CudaSlice<f32>,
        post_w: &CudaSlice<f32>,
        pre_w: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        rs: &mut CudaSlice<f32>,
        n: usize,
        eps: f32,
        s: f32,
        rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .addnorm_e4m3_row2
            .ok_or(GpuError::MissingOp("addnorm_e4m3_row2"))?;
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (pp, _g2) = proj.device_ptr(&self.stream);
        let (pow, _g3) = post_w.device_ptr(&self.stream);
        let (prw, _g4) = pre_w.device_ptr(&self.stream);
        let (qp, _g5) = q.device_ptr_mut(&self.stream);
        let (sp, _g6) = rs.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; proj holds bf16 in the f32-typed scratch
        check(unsafe {
            f(
                xp as *mut _,
                pp as *const _,
                pow as *const _,
                prw as *const _,
                qp as *mut _,
                sp as *mut _,
                n as u32,
                eps,
                s,
                rows as u32,
                1u32,
                self.stream_ptr(),
            )
        })
    }

    /// All five chunk-band 16-bit entries present.
    pub fn has_chunk16(&self) -> bool {
        self.kernels.f8_gemm_w8_pc_qkv_r2.is_some()
            // the i16 arms now route through the arch-constant supersets
            // (batch5 / kv_nra_rows3), so the capability must name those -
            // a pack with batch2 but not batch5 would pass this gate and
            // then MissingOp at the first chunk-band layer
            && self.kernels.qkv_norm_rope_batch5.is_some()
            && self.kernels.kv_nra_rows3.is_some()
            && self.kernels.addnorm_e4m3_row2.is_some()
            && self.kernels.rmsnorm_add_scale2.is_some()
    }

    pub fn has_quantize_e4m3_glu2(&self, act: GluAct) -> bool {
        match act {
            GluAct::Gelu => self.kernels.quantize_e4m3_geglu2.is_some(),
            GluAct::Silu => self.kernels.quantize_e4m3_swiglu2.is_some(),
        }
    }

    pub fn has_quantize_e4m3_glu(&self, act: GluAct) -> bool {
        match act {
            GluAct::Gelu => self.kernels.quantize_e4m3_geglu.is_some(),
            GluAct::Silu => self.kernels.quantize_e4m3_swiglu.is_some(),
        }
    }

    /// Split-buffer fused fold + e4m3 quantize, dispatched on the model's
    /// activation (see [`GpuExecutor::glu`] on the two carriers' provenance).
    #[allow(clippy::too_many_arguments)]
    pub fn quantize_e4m3_glu(
        &self,
        gate: &CudaSlice<f32>,
        up: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        scale: &mut CudaSlice<u8>,
        n: usize,
        act: GluAct,
    ) -> Result<(), GpuError> {
        match act {
            GluAct::Gelu => self.quantize_e4m3_geglu(gate, up, q, scale, n),
            GluAct::Silu => self.quantize_e4m3_swiglu(gate, up, q, scale, n),
        }
    }

    pub fn quantize_e4m3_swiglu(
        &self,
        gate: &CudaSlice<f32>,
        up: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        scale: &mut CudaSlice<u8>,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .quantize_e4m3_swiglu
            .ok_or(GpuError::MissingOp("quantize_e4m3_swiglu"))?;
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

    /// Re-quantize a repacked Q8_0 weight to [`RepackedMxfp4`] (e2m1 + ue8m0
    /// per-32, GGUF split nibble order) on device - the block-scale GEMM's
    /// weight format. LOSSY (4-bit); one-time, at model load, sm_120a only.
    pub fn q8_0_to_mxfp4(&self, w: &RepackedQ8) -> Result<RepackedMxfp4, GpuError> {
        let f = self
            .kernels
            .q8_0_to_mxfp4
            .ok_or(GpuError::MissingOp("q8_0_to_mxfp4"))?;
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

    /// Q8_0 -> e4m3 weight planes for the W8A8-FP8 GEMM (full bytes, no
    /// nibble packing; ue8m0/32 scales). Same numeric construction as
    /// quantize_e4m3's 448-bound scale pick.
    /// e4m3 decode-band K-split GEMM: f8w weights x e4m3 activations at
    /// b <= 64 - the fp8 twin of `q8_0_gemm_mma_ks` (1.031 B/param stream,
    /// measured 1.02-1.05x the q8 rung on the 27B decode shapes). PRECISION
    /// CLASS: e4m3 operands - callers route behind an opt-in + gates.
    #[allow(clippy::too_many_arguments)]
    pub fn f8d_gemm_mma_ks(
        &self,
        w: &RepackedMxfp4,
        in_dim: usize,
        out_dim: usize,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<u8>,
        part: &mut CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        if w.scale.len() == 4 || w.scale.len() == 12 {
            // lin/rowwise plane (marker scale): the contiguous-stream decode
            // GEMM - f8_gemm_lin dispatches the rowwise arm itself. It is
            // TMA-mapped at <= 64 rows by construction (one batch tile), so
            // above that the same plane rides its prefill-class kt arm - the
            // route chunked prefill already takes for these planes at chunk
            // widths. dflash rung D: the qwen35 spec verify walk
            // and the block drafter's head call this at batch x k1 rows; the
            // lin refusal above 64 was the wall under every round deeper
            // than k=1 at 32 live (c32 stuck at an acceptance of E~1.9).
            // Rung N: 33..64 rows also ride the kt arm - the 64
            // boundary was the legacy kernel's TMA-map cap, not an election.
            // The pdN-c8 anatomy (M=64, every verify GEMM on the legacy
            // kernel) had the narrow-out planes at ~55-60% of the DRAM roof
            // (80-CTA grids, 0.43 waves, no K-split) where kt+ktz measures
            // ~94% on the same shapes. 32 and below keep the legacy class:
            // c4's verify (M=32) and the winning cells are measured there.
            //
            // ...and the kt arm is not universal. Its rowwise kernel
            // (pd_f8_gemm_lin_kt_r) gates its whole body on `cma == 12`, so on
            // every die that is not sm_120 it answers 801/NotSupported for
            // every call. Routing 33..64 rows here unconditionally meant that
            // on sm_100 the qwen3.8 nvfp4 spec verify tick failed at >= 20 live
            // slots (k1=2 -> 40 rows > 32): the service finished every
            // in-flight sequence, the runner reported finish_reason "stop", and
            // every request came back with one token in it. 16 live
            // (32 rows) sat just under the threshold and looked healthy.
            //
            // Discover the arm at runtime rather than mirroring its cc gate
            // here - a mirrored gate is what drifted in the first place. One
            // refusal disables it process-wide and the legacy arm takes over.
            if batch > 32 && LIN_KT_OK.load(std::sync::atomic::Ordering::Relaxed) {
                match self.f8_gemm_lin_kt(w, 0, xq, xs, y, in_dim, out_dim, batch, false) {
                    Err(GpuError::Launch(801)) => {
                        LIN_KT_OK.store(false, std::sync::atomic::Ordering::Relaxed);
                        tracing::warn!(
                            "f8_gemm_lin: the kt arm is unavailable on this die                              (cc-12 only) - falling back to the legacy lin arm                              for batch > 32"
                        );
                    }
                    other => return other,
                }
            }
            return self.f8_gemm_lin(w, 0, in_dim, out_dim, xq, xs, part, y, batch);
        }
        if w.scale.len() == 8 {
            // bs plane (byte-passthrough): data = boxes ‖ f32 scale tail
            let f = self
                .kernels
                .f8_gemm_lin_bs
                .ok_or(GpuError::MissingOp("f8_gemm_lin_bs"))?;
            let boxes = out_dim.div_ceil(128) * (in_dim / 128) * 16384;
            let (dp, _g1) = w.data.device_ptr(&self.stream);
            let (xqp, _g3) = xq.device_ptr(&self.stream);
            let (xsp, _g4) = xs.device_ptr(&self.stream);
            let (pp, _g5) = part.device_ptr_mut(&self.stream);
            let (yp, _g6) = y.device_ptr_mut(&self.stream);
            let scp = dp + boxes as u64;
            // SAFETY: ABI contract; part >= 8 * out_dim * batch f32
            return check(unsafe {
                f(
                    dp as *const _,
                    scp as *const _,
                    xqp as *const _,
                    xsp as *const _,
                    pp as *mut _,
                    yp as *mut _,
                    in_dim as u32,
                    out_dim as u32,
                    batch as u32,
                    self.stream_ptr(),
                )
            });
        }
        // sm_100: the tcgen05 block-scale arm (slot 869) - the mma.sync K-split
        // GEMM below streamed the decode projections at a fraction of the die's
        // bandwidth (pd_f8_gemm_mma_kernel<64>: 15% of GPU time at depth 64).
        // PADDOCK_NO_F8_TC5X=1 keeps the mma arm for A/B.
        static NO_TC5X: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let no_tc5x =
            *NO_TC5X.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_F8_TC5X").is_some());
        if let Some(f) = self
            .kernels
            .f8_gemm_tc5x
            .filter(|_| !no_tc5x && in_dim.is_multiple_of(128))
        {
            let n = part.len();
            let (dp, _g1) = w.data.device_ptr(&self.stream);
            let (scp, _g2) = w.scale.device_ptr(&self.stream);
            let (xqp, _g3) = xq.device_ptr(&self.stream);
            let (xsp, _g4) = xs.device_ptr(&self.stream);
            let (pp, _g5) = part.device_ptr_mut(&self.stream);
            let (yp, _g6) = y.device_ptr_mut(&self.stream);
            // SAFETY: ABI contract (slot 869); same operands as the mma arm below
            return check(unsafe {
                f(
                    dp as *const _,
                    scp as *const _,
                    xqp as *const _,
                    xsp as *const _,
                    yp as *mut _,
                    in_dim as u32,
                    out_dim as u32,
                    batch as u32,
                    pp as *mut _,
                    n as u64,
                    self.stream_ptr(),
                )
            });
        }
        let f = self
            .kernels
            .f8d_gemm_mma_ks
            .ok_or(GpuError::MissingOp("f8d_gemm_mma_ks"))?;
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (scp, _g2) = w.scale.device_ptr(&self.stream);
        let (xqp, _g3) = xq.device_ptr(&self.stream);
        let (xsp, _g4) = xs.device_ptr(&self.stream);
        let (pp, _g5) = part.device_ptr_mut(&self.stream);
        let (yp, _g6) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; part >= 8 * out_dim * batch f32
        check(unsafe {
            f(
                dp as *const _,
                scp as *const _,
                xqp as *const _,
                xsp as *const _,
                pp as *mut _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// True when the pack ships the e4m3 decode-band GEMM (sm_89+).
    pub fn has_f8d_gemm_mma_ks(&self) -> bool {
        self.kernels.f8d_gemm_mma_ks.is_some()
    }

    /// True when the full tile-linear f8 lane is present (sm_90+ bulk/TMA).
    pub fn has_f8_lin(&self) -> bool {
        self.kernels.f8w_repack_lin.is_some()
            && self.kernels.f8_gemm_lin.is_some()
            && self.kernels.f8_gemm_lin_kt.is_some()
    }

    /// True when the lin lane can serve b=1 from the boxes (slot 481) - the
    /// capability that lets a plane class drop its Q8_0 twin.
    pub fn has_f8lin_gemv(&self) -> bool {
        self.kernels.f8lin_gemv.is_some()
    }

    /// b=1 GEMV over a tile-linear plane (non-KV-overhead R2.2). `part` must
    /// hold `nz * out_dim` floats - pass `y` itself and a null ticket to pin
    /// nz=1. `ticket` is zeroed once at allocation and belongs to one plane
    /// shape (its wrap value is that shape's elected K-split).
    /// Row-window form: serve output rows [`out_off`, `out_off + out_dim`) of
    /// a plane, reading from row-tile `out_off/128` onward and landing at
    /// `y[out_off..]`. This is what lets one fused gate|up plane be driven by
    /// two independent launches - which is not a style choice: the Q8_0 chain
    /// this replaces issues gate and up separately, and the scheduler overlaps
    /// them with each other and with the surrounding norms (measured:
    /// 128 ms of concurrency per window vs 24 ms for a single fused launch,
    /// which cost more than the fused call saved). Concurrent windows of the
    /// same plane must get disjoint `part` and `ticket` regions.
    #[allow(clippy::too_many_arguments)]
    pub fn f8lin_gemv_at(
        &self,
        w: &RepackedMxfp4,
        x: &CudaSlice<f32>,
        part: &mut CudaSlice<f32>,
        part_off: usize,
        y: &mut CudaSlice<f32>,
        out_off: usize,
        ticket: Option<(&mut CudaSlice<u32>, usize)>,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<(), GpuError> {
        debug_assert_eq!(out_off % 128, 0, "row window must start on a box row-tile");
        let f = self
            .kernels
            .f8lin_gemv
            .ok_or(GpuError::MissingOp("f8lin_gemv"))?;
        debug_assert!(w.is_lin(), "f8lin_gemv needs a lin-tile plane");
        let nk = in_dim / 128;
        let box_off = (out_off / 128) * nk * 16896;
        let (wp, _g0) = w.data.device_ptr(&self.stream);
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (pp, _g2) = part.device_ptr_mut(&self.stream);
        let (yp, _g3) = y.device_ptr_mut(&self.stream);
        let tp = match ticket {
            Some((t, off)) => {
                debug_assert!(off + 2 * out_dim.div_ceil(128) <= t.len());
                (t.device_ptr_mut(&self.stream).0 as usize + off * core::mem::size_of::<u32>())
                    as *mut core::ffi::c_void
            }
            None => core::ptr::null_mut(),
        };
        let es = core::mem::size_of::<f32>();
        // SAFETY: ABI contract; windows are bounds-checked by the caller's
        // sizing obligations (documented on the ABI slot).
        check(unsafe {
            f(
                (wp as usize + box_off) as *const _,
                xp as *const _,
                (pp as usize + part_off * es) as *mut _,
                (yp as usize + out_off * es) as *mut _,
                tp,
                in_dim as u32,
                out_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    #[allow(dead_code)]
    pub fn f8lin_gemv(
        &self,
        w: &RepackedMxfp4,
        x: &CudaSlice<f32>,
        part: &mut CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        ticket: Option<(&mut CudaSlice<u32>, usize)>,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .f8lin_gemv
            .ok_or(GpuError::MissingOp("f8lin_gemv"))?;
        debug_assert!(w.is_lin(), "f8lin_gemv needs a lin-tile plane");
        let (wp, _g0) = w.data.device_ptr(&self.stream);
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (pp, _g2) = part.device_ptr_mut(&self.stream);
        let (yp, _g3) = y.device_ptr_mut(&self.stream);
        // the ticket slice is shared across plane shapes by OFFSET: each
        // shape owns [off, off + 2*ceil(out_dim/128)) and nothing else, since
        // the counter's wrap value is that shape's elected K-split
        let tp = match ticket {
            Some((t, off)) => {
                debug_assert!(off + 2 * out_dim.div_ceil(128) <= t.len());
                (t.device_ptr_mut(&self.stream).0 as usize + off * core::mem::size_of::<u32>())
                    as *mut core::ffi::c_void
            }
            None => core::ptr::null_mut(),
        };
        // SAFETY: ABI contract; the plane is lin (marker-checked above) and
        // part/ticket sizing is the caller's documented obligation.
        check(unsafe {
            f(
                wp as *const _,
                xp as *const _,
                pp as *mut _,
                yp as *mut _,
                tp,
                in_dim as u32,
                out_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Repack an f8w plane into the tile-linear box layout (gemm/f8_lin.cuh:
    /// per-CTA contiguous weight streams - the decode GEMM was at its
    /// access-pattern roof on row-major planes). CONSUMES the row-major
    /// plane; the returned plane's `data` holds the boxes (scales embedded,
    /// `scale` is a dummy). Same 1.03125 B/param - VRAM-neutral swap.
    pub fn f8w_repack_lin(
        &self,
        w: RepackedMxfp4,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<RepackedMxfp4, GpuError> {
        let f = self
            .kernels
            .f8w_repack_lin
            .ok_or(GpuError::MissingOp("f8w_repack_lin"))?;
        let nrt = out_dim.div_ceil(128);
        let nk = in_dim / 128;
        // marker first: a 4-byte alloc made after freeing the source plane
        // lands in the just-freed slab and PINS it against pool trimming
        // (~20 MB retained per plane, GBs across a model - measured on the
        // 27B). Allocated up front it packs into the small-size bins.
        let scale: CudaSlice<u8> = self.stream.alloc_zeros(4).map_err(drv)?;
        let mut lin: CudaSlice<u8> = self.stream.alloc_zeros(nrt * nk * 16896).map_err(drv)?;
        {
            let (dp, _g1) = w.data.device_ptr(&self.stream);
            let (scp, _g2) = w.scale.device_ptr(&self.stream);
            let (lp, _g3) = lin.device_ptr_mut(&self.stream);
            check(unsafe {
                f(
                    dp as *const _,
                    scp as *const _,
                    lp as *mut _,
                    in_dim as u32,
                    out_dim as u32,
                    self.stream_ptr(),
                )
            })?;
        }
        // the source plane frees on drop below; fence the repack first
        self.stream.synchronize().map_err(drv)?;
        drop(w);
        // release the freed source slabs to CUDA each plane - the async
        // pool otherwise accumulates partially-pinned slabs across the 192
        // conversions and the KV pool sizes against inflated usage
        self.trim_mem_pool();
        Ok(RepackedMxfp4 { data: lin, scale })
    }

    pub fn has_f8w_repack_lin_gui(&self) -> bool {
        self.kernels.f8w_repack_lin_gui.is_some()
    }

    /// gu-interleave twin of [`Self::f8w_repack_lin`]: the boxed rows are
    /// permuted so gate/up pair p lands at tile rows (p>>3)*16+(p&7) / +8 -
    /// the layout the fused geglu+quant GEMM epilogue (`f8_gemm_lin_gu`)
    /// pairs in-register. Downstream lin GEMMs are layout-blind; geglu
    /// consumers must switch to `quantize_e4m3_geglu2i`.
    pub fn f8w_repack_lin_gui(
        &self,
        w: RepackedMxfp4,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<RepackedMxfp4, GpuError> {
        let f = self
            .kernels
            .f8w_repack_lin_gui
            .ok_or(GpuError::MissingOp("f8w_repack_lin_gui"))?;
        let nrt = out_dim.div_ceil(128);
        let nk = in_dim / 128;
        // marker-first alloc order: see f8w_repack_lin
        let scale: CudaSlice<u8> = self.stream.alloc_zeros(4).map_err(drv)?;
        let mut lin: CudaSlice<u8> = self.stream.alloc_zeros(nrt * nk * 16896).map_err(drv)?;
        {
            let (dp, _g1) = w.data.device_ptr(&self.stream);
            let (scp, _g2) = w.scale.device_ptr(&self.stream);
            let (lp, _g3) = lin.device_ptr_mut(&self.stream);
            check(unsafe {
                f(
                    dp as *const _,
                    scp as *const _,
                    lp as *mut _,
                    in_dim as u32,
                    out_dim as u32,
                    self.stream_ptr(),
                )
            })?;
        }
        self.stream.synchronize().map_err(drv)?;
        drop(w);
        self.trim_mem_pool();
        Ok(RepackedMxfp4 { data: lin, scale })
    }

    pub fn has_f8_gemm_lin_gu(&self, act: GluAct) -> bool {
        match act {
            GluAct::Gelu => self.kernels.f8_gemm_lin_gu.is_some(),
            GluAct::Silu => self.kernels.f8_gemm_lin_gu_silu.is_some(),
        }
    }

    /// Full rowwise (strip-free) lane capability: every consumer of a pc
    /// plane has its rowwise twin, so a strip-free plane can never strand a
    /// route (the plane has no strips to fall back on). The gu twins are
    /// checked for the MODEL's activation - a pack shipping only the GELU
    /// instantiation must not open the rowwise lane to a SwiGLU arch.
    pub fn has_f8_rowvec(&self, act: GluAct) -> bool {
        let (gu_r, gu_pc_r) = match act {
            GluAct::Gelu => (
                self.kernels.f8_gemm_lin_gu_r,
                self.kernels.f8_gemm_lin_gu_pc_r,
            ),
            GluAct::Silu => (
                self.kernels.f8_gemm_lin_gu_r_silu,
                self.kernels.f8_gemm_lin_gu_pc_r_silu,
            ),
        };
        self.kernels.f8w_repack_lin_bs.is_some()
            && self.kernels.f8w_repack_lin_bs_gui.is_some()
            && self.kernels.f8_gemm_lin_r.is_some()
            && self.kernels.f8_gemm_lin_kt_r.is_some()
            && gu_r.is_some()
            && gu_pc_r.is_some()
            && self.kernels.f8_gemm_w8_pc_r.is_some()
            && self.kernels.f8_gemm_w8_pcd_r.is_some()
    }

    /// Build a rowwise (strip-free) pc plane: e4m3 rows -> data-only lin
    /// boxes with the per-row ue8m0 exponent bytes appended at the tail of
    /// `data` (padded to the 128-row box tail); `scale` is the 12-byte
    /// marker the lin wrappers dispatch on. `wse` in SOURCE row order; with
    /// `gui` the boxes AND the tail are gu-interleaved (pair p at tile rows
    /// (p>>3)*16+(p&7) / +8). Tail invariant the wrappers rely on:
    /// data.len() = padded_rows * (in_dim + 1). CONSUMES the raw plane.
    pub fn f8w_build_lin_rw(
        &self,
        raw: CudaSlice<u8>,
        wse: &[u8],
        in_dim: usize,
        out_dim: usize,
        gui: bool,
    ) -> Result<RepackedMxfp4, GpuError> {
        let nrt = out_dim.div_ceil(128);
        let nk = in_dim / 128;
        debug_assert_eq!(wse.len(), out_dim);
        let marker: CudaSlice<u8> = self.stream.alloc_zeros(12).map_err(drv)?;
        let mut data: CudaSlice<u8> = self
            .stream
            .alloc_zeros(nrt * nk * 16384 + nrt * 128)
            .map_err(drv)?;
        {
            let f = if gui {
                self.kernels
                    .f8w_repack_lin_bs_gui
                    .ok_or(GpuError::MissingOp("f8w_repack_lin_bs_gui"))?
            } else {
                self.kernels
                    .f8w_repack_lin_bs
                    .ok_or(GpuError::MissingOp("f8w_repack_lin_bs"))?
            };
            let (dp, _g1) = raw.device_ptr(&self.stream);
            let (op, _g2) = data.device_ptr_mut(&self.stream);
            // SAFETY: ABI contract; data sized above
            check(unsafe {
                f(
                    dp as *const _,
                    op as *mut _,
                    in_dim as u32,
                    out_dim as u32,
                    self.stream_ptr(),
                )
            })?;
        }
        {
            // tail: exponent byte per BOX row (interleaved when gui), pad
            // rows exponent 0 (their data is zero; the epilogues never
            // write them)
            let mut tail_host = vec![0u8; nrt * 128];
            if gui {
                let half = out_dim / 2;
                for (row, t) in tail_host.iter_mut().enumerate().take(out_dim) {
                    let p = (row >> 4) * 8 + (row & 7);
                    let src = if (row & 15) < 8 { p } else { half + p };
                    *t = wse[src];
                }
            } else {
                tail_host[..out_dim].copy_from_slice(wse);
            }
            let mut tail = data.slice_mut(nrt * nk * 16384..);
            self.stream
                .memcpy_htod(&tail_host, &mut tail)
                .map_err(drv)?;
        }
        self.stream.synchronize().map_err(drv)?;
        drop(raw);
        self.trim_mem_pool();
        Ok(RepackedMxfp4 {
            data,
            scale: marker,
        })
    }

    /// Fused gu GEMM + gated-FFN fold + per-32 e4m3 quant over an interleaved
    /// lin plane (`f8w_repack_lin_gui`): writes what quantize_e4m3 would hand
    /// the down GEMM - q `[batch][out_dim/2]` e4m3 bytes, qs the ue8m0
    /// scales - bit-identical to lin_kt -> glu2i at the same `act`. Ok(false)
    /// = the route couldn't engage (no TMA / kt3 off); caller keeps the
    /// 2-launch chain.
    /// q must not alias xq (the kernel reads xq via TMA while storing q).
    #[allow(clippy::too_many_arguments)]
    pub fn f8_gemm_lin_gu(
        &self,
        w: &RepackedMxfp4,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<u8>,
        q: &mut CudaSlice<i8>,
        qs: &mut CudaSlice<u8>,
        in_dim: usize,
        out_dim: usize,
        batch: usize,
        act: GluAct,
    ) -> Result<bool, GpuError> {
        if w.scale.len() == 12 {
            // rowwise gu plane: wse tail already in BOX ROW (interleaved)
            // order (see f8w_build_lin_rw)
            let (f, name) = match act {
                GluAct::Gelu => (self.kernels.f8_gemm_lin_gu_r, "f8_gemm_lin_gu_r"),
                GluAct::Silu => (self.kernels.f8_gemm_lin_gu_r_silu, "f8_gemm_lin_gu_r_silu"),
            };
            let f = f.ok_or(GpuError::MissingOp(name))?;
            let wse_off = (w.data.len() / (in_dim + 1)) * in_dim;
            let (dp, _g1) = w.data.device_ptr(&self.stream);
            let (xqp, _g2) = xq.device_ptr(&self.stream);
            let (xsp, _g3) = xs.device_ptr(&self.stream);
            let (qp, _g4) = q.device_ptr_mut(&self.stream);
            let (qsp, _g5) = qs.device_ptr_mut(&self.stream);
            // SAFETY: ABI contract; q/qs sized [batch*out_dim/2]/[batch*out_dim/64]
            let rc = unsafe {
                f(
                    dp as *const _,
                    (dp as usize + wse_off) as *const _,
                    xqp as *const _,
                    xsp as *const _,
                    qp as *mut _,
                    qsp as *mut _,
                    in_dim as u32,
                    out_dim as u32,
                    batch as u32,
                    self.stream_ptr(),
                )
            };
            if rc == -2 {
                return Ok(false);
            }
            check(rc)?;
            return Ok(true);
        }
        let (f, name) = match act {
            GluAct::Gelu => (self.kernels.f8_gemm_lin_gu, "f8_gemm_lin_gu"),
            GluAct::Silu => (self.kernels.f8_gemm_lin_gu_silu, "f8_gemm_lin_gu_silu"),
        };
        let f = f.ok_or(GpuError::MissingOp(name))?;
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (xqp, _g2) = xq.device_ptr(&self.stream);
        let (xsp, _g3) = xs.device_ptr(&self.stream);
        let (qp, _g4) = q.device_ptr_mut(&self.stream);
        let (qsp, _g5) = qs.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; q/qs sized [batch*out_dim/2]/[batch*out_dim/64]
        let rc = unsafe {
            f(
                dp as *const _,
                xqp as *const _,
                xsp as *const _,
                qp as *mut _,
                qsp as *mut _,
                in_dim as u32,
                out_dim as u32,
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

    /// Per-channel gu GEMM (kt4a scale-free mainloop): `as_row` =
    /// f32 per-token scales from the row quantizer, `ws` = f32 per-channel
    /// scales [out_dim] (gate half at 0, up half at out_dim/2). Serves the
    /// pc plane whose per-row pow2 exponents also fill the per-32 strip, so
    /// every other consumer of the same plane dequantizes identically.
    /// Ok(false) = route not covered (the pack's -2).
    #[allow(clippy::too_many_arguments)]
    pub fn f8_gemm_lin_gu_pc(
        &self,
        w: &RepackedMxfp4,
        xq: &CudaSlice<i8>,
        as_row: &CudaSlice<f32>,
        ws: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        qs: &mut CudaSlice<u8>,
        in_dim: usize,
        out_dim: usize,
        batch: usize,
        act: GluAct,
    ) -> Result<bool, GpuError> {
        // rowwise plane: same signature, box stride handled pack-side
        let (f, name) = match (w.scale.len() == 12, act) {
            (true, GluAct::Gelu) => (self.kernels.f8_gemm_lin_gu_pc_r, "f8_gemm_lin_gu_pc_r"),
            (true, GluAct::Silu) => (
                self.kernels.f8_gemm_lin_gu_pc_r_silu,
                "f8_gemm_lin_gu_pc_r_silu",
            ),
            (false, GluAct::Gelu) => (self.kernels.f8_gemm_lin_gu_pc, "f8_gemm_lin_gu_pc"),
            (false, GluAct::Silu) => (
                self.kernels.f8_gemm_lin_gu_pc_silu,
                "f8_gemm_lin_gu_pc_silu",
            ),
        };
        let f = f.ok_or(GpuError::MissingOp(name))?;
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (xqp, _g2) = xq.device_ptr(&self.stream);
        let (asp, _g3) = as_row.device_ptr(&self.stream);
        let (wsp, _g4) = ws.device_ptr(&self.stream);
        let (qp, _g5) = q.device_ptr_mut(&self.stream);
        let (qsp, _g6) = qs.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; q/qs sized [batch*out_dim/2]/[batch*out_dim/64]
        let rc = unsafe {
            f(
                dp as *const _,
                xqp as *const _,
                asp as *const _,
                wsp as *const _,
                qp as *mut _,
                qsp as *mut _,
                in_dim as u32,
                out_dim as u32,
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

    pub fn has_f8_gemm_lin_gu_pc(&self, act: GluAct) -> bool {
        match act {
            GluAct::Gelu => self.kernels.f8_gemm_lin_gu_pc.is_some(),
            GluAct::Silu => self.kernels.f8_gemm_lin_gu_pc_silu.is_some(),
        }
    }

    /// pc lin GEMM for the qkv/wo classes (kt4 twin). `row_off`
    /// slices a fused lin plane (128-multiple); `ws_off` slices the scale
    /// vector to the same segment. Ok(false) = route not covered.
    #[allow(clippy::too_many_arguments)]
    pub fn f8_gemm_w8_pc(
        &self,
        w: &RepackedMxfp4,
        row_off: usize,
        xq: &CudaSlice<i8>,
        as_row: &CudaSlice<f32>,
        ws: &CudaSlice<f32>,
        ws_off: usize,
        y: &mut CudaSlice<f32>,
        in_dim: usize,
        out_dim: usize,
        batch: usize,
    ) -> Result<bool, GpuError> {
        // rowwise plane: same signature, box stride handled pack-side
        let f = if w.scale.len() == 12 {
            self.kernels
                .f8_gemm_w8_pc_r
                .ok_or(GpuError::MissingOp("f8_gemm_w8_pc_r"))?
        } else {
            self.kernels
                .f8_gemm_w8_pc
                .ok_or(GpuError::MissingOp("f8_gemm_w8_pc"))?
        };
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (xqp, _g2) = xq.device_ptr(&self.stream);
        let (asp, _g3) = as_row.device_ptr(&self.stream);
        let (wsp, _g4) = ws.device_ptr(&self.stream);
        let (yp, _g5) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; ws sliced by ws_off covers out_dim rows
        let rc = unsafe {
            f(
                dp as *const _,
                row_off as u32,
                xqp as *const _,
                asp as *const _,
                (wsp + (ws_off * 4) as u64) as *const _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                0u32,
                self.stream_ptr(),
            )
        };
        if rc == -2 {
            return Ok(false);
        }
        check(rc)?;
        Ok(true)
    }

    pub fn has_f8_gemm_w8_pc(&self) -> bool {
        self.kernels.f8_gemm_w8_pc.is_some()
    }

    /// Fused qkv single-launch on the rowwise pc plane: one grid
    /// over the whole q‖k‖v plane, epilogue scattered to the three dense
    /// per-projection outputs - bit-exact vs the three `f8_gemm_w8_pc`
    /// slices, minus their per-launch ramp/straggler waves at admission-M.
    /// Ok(false) = route not covered (strip plane or pack without the arm) -
    /// caller falls back to the split launches.
    #[allow(clippy::too_many_arguments)]
    pub fn f8_gemm_w8_pc_qkv(
        &self,
        w: &RepackedMxfp4,
        xq: &CudaSlice<i8>,
        as_row: &CudaSlice<f32>,
        ws: &CudaSlice<f32>,
        yq: &mut CudaSlice<f32>,
        yk: &mut CudaSlice<f32>,
        yv: &mut CudaSlice<f32>,
        in_dim: usize,
        q_dim: usize,
        kv_dim: usize,
        batch: usize,
    ) -> Result<bool, GpuError> {
        if w.scale.len() != 12 {
            return Ok(false);
        }
        let f = self
            .kernels
            .f8_gemm_w8_pc_qkv_r
            .ok_or(GpuError::MissingOp("f8_gemm_w8_pc_qkv_r"))?;
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (xqp, _g2) = xq.device_ptr(&self.stream);
        let (asp, _g3) = as_row.device_ptr(&self.stream);
        let (wsp, _g4) = ws.device_ptr(&self.stream);
        let (yqp, _g5) = yq.device_ptr_mut(&self.stream);
        let (ykp, _g6) = yk.device_ptr_mut(&self.stream);
        let (yvp, _g7) = yv.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; ws covers the full q_dim + 2*kv_dim plane
        let rc = unsafe {
            f(
                dp as *const _,
                xqp as *const _,
                asp as *const _,
                wsp as *const _,
                yqp as *mut _,
                ykp as *mut _,
                yvp as *mut _,
                in_dim as u32,
                q_dim as u32,
                kv_dim as u32,
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

    pub fn has_f8_gemm_w8_pc_qkv(&self) -> bool {
        self.kernels.f8_gemm_w8_pc_qkv_r.is_some()
    }

    /// down twin (kt4d): weights per-channel via `ws`, activations
    /// per-32 in-loop via `xs`. Ok(false) = route not covered.
    #[allow(clippy::too_many_arguments)]
    pub fn f8_gemm_w8_pcd(
        &self,
        w: &RepackedMxfp4,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<u8>,
        ws: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        in_dim: usize,
        out_dim: usize,
        batch: usize,
    ) -> Result<bool, GpuError> {
        // rowwise plane: same signature, box stride handled pack-side
        let f = if w.scale.len() == 12 {
            self.kernels
                .f8_gemm_w8_pcd_r
                .ok_or(GpuError::MissingOp("f8_gemm_w8_pcd_r"))?
        } else {
            self.kernels
                .f8_gemm_w8_pcd
                .ok_or(GpuError::MissingOp("f8_gemm_w8_pcd"))?
        };
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (xqp, _g2) = xq.device_ptr(&self.stream);
        let (xsp, _g3) = xs.device_ptr(&self.stream);
        let (wsp, _g4) = ws.device_ptr(&self.stream);
        let (yp, _g5) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; ws covers out_dim rows
        let rc = unsafe {
            f(
                dp as *const _,
                0u32,
                xqp as *const _,
                xsp as *const _,
                wsp as *const _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                0u32,
                self.stream_ptr(),
            )
        };
        if rc == -2 {
            return Ok(false);
        }
        check(rc)?;
        Ok(true)
    }

    pub fn has_f8_gemm_w8_pcd(&self) -> bool {
        self.kernels.f8_gemm_w8_pcd.is_some()
    }

    /// Tile-linear e4m3 decode GEMM - same contract as `f8d_gemm_mma_ks`
    /// but `w.data` holds lin boxes. Call sites swap 1:1 on the lin flag.
    /// `row_off` selects a sub-view of a fused plane (must be a multiple of
    /// 128 - every fused-plane slice boundary is; boxes are row-tile-major
    /// so the sub-view is one pointer offset, same math as the kt twin).
    #[allow(clippy::too_many_arguments)]
    pub fn f8_gemm_lin(
        &self,
        w: &RepackedMxfp4,
        row_off: usize,
        in_dim: usize,
        out_dim: usize,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<u8>,
        part: &mut CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        // Batch cap slicing: the pack guards `batch > 64` with
        // InvalidValue (pd_f8_gemm_lin_go), and at c128 decode the refusal
        // KILLED every rider row (chat closed them as short clean "stop"s;
        // 896/896 requests errored in the c128 trace leg - the same
        // silent-stop shape as the nvfp4 spec-lane one-token incident).
        // Rows are independent in this GEMM: both launch branches below
        // loop >64-row batches in <=64-row slices via pointer offsets -
        // bitwise-identical per row, part scratch reused per slice.
        debug_assert_eq!(row_off % 128, 0, "lin row_off must be box-aligned");
        if w.scale.len() == 12 {
            // rowwise plane: data = 16384B boxes ‖ per-row wse tail
            // (data.len() = padded_rows * (in_dim + 1))
            let f = self
                .kernels
                .f8_gemm_lin_r
                .ok_or(GpuError::MissingOp("f8_gemm_lin_r"))?;
            let padded = w.data.len() / (in_dim + 1);
            let box_off = (row_off / 128) * (in_dim / 128) * 16384;
            let wse_off = padded * in_dim + row_off;
            let (dp, _g1) = w.data.device_ptr(&self.stream);
            let (xqp, _g3) = xq.device_ptr(&self.stream);
            let (xsp, _g4) = xs.device_ptr(&self.stream);
            let (pp, _g5) = part.device_ptr_mut(&self.stream);
            let (yp, _g6) = y.device_ptr_mut(&self.stream);
            // SAFETY: ABI contract; part >= 8 * out_dim * batch f32
            let mut b0 = 0usize;
            while b0 < batch {
                let bn = (batch - b0).min(64);
                check(unsafe {
                    f(
                        (dp as usize + box_off) as *const _,
                        (dp as usize + wse_off) as *const _,
                        (xqp as usize + b0 * in_dim) as *const _,
                        (xsp as usize + b0 * (in_dim / 32)) as *const _,
                        pp as *mut _,
                        (yp as usize + b0 * out_dim * 4) as *mut _,
                        in_dim as u32,
                        out_dim as u32,
                        bn as u32,
                        self.stream_ptr(),
                    )
                })?;
                b0 += bn;
            }
            return Ok(());
        }
        let f = self
            .kernels
            .f8_gemm_lin
            .ok_or(GpuError::MissingOp("f8_gemm_lin"))?;
        let box_off = (row_off / 128) * (in_dim / 128) * 16896;
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (xqp, _g3) = xq.device_ptr(&self.stream);
        let (xsp, _g4) = xs.device_ptr(&self.stream);
        let (pp, _g5) = part.device_ptr_mut(&self.stream);
        let (yp, _g6) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; part >= 8 * out_dim * batch f32
        let mut b0 = 0usize;
        while b0 < batch {
            let bn = (batch - b0).min(64);
            check(unsafe {
                f(
                    (dp as usize + box_off) as *const _,
                    (xqp as usize + b0 * in_dim) as *const _,
                    (xsp as usize + b0 * (in_dim / 32)) as *const _,
                    pp as *mut _,
                    (yp as usize + b0 * out_dim * 4) as *mut _,
                    in_dim as u32,
                    out_dim as u32,
                    bn as u32,
                    self.stream_ptr(),
                )
            })?;
            b0 += bn;
        }
        Ok(())
    }

    /// Block-scale tile-linear decode GEMM (official-FP8 byte passthrough):
    /// data-only 16384 B boxes in `wlin`, one f32 scale per 128×128 block in
    /// `wsc` ([out/128][in/128]). Same contract as [`Self::f8_gemm_lin`].
    #[allow(clippy::too_many_arguments)]
    pub fn f8_gemm_lin_bs(
        &self,
        wlin: &CudaSlice<u8>,
        wsc: &CudaSlice<f32>,
        in_dim: usize,
        out_dim: usize,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<u8>,
        part: &mut CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .f8_gemm_lin_bs
            .ok_or(GpuError::MissingOp("f8_gemm_lin_bs"))?;
        debug_assert!(wsc.len() >= (out_dim / 128) * (in_dim / 128));
        let (dp, _g1) = wlin.device_ptr(&self.stream);
        let (sp, _g2) = wsc.device_ptr(&self.stream);
        let (xqp, _g3) = xq.device_ptr(&self.stream);
        let (xsp, _g4) = xs.device_ptr(&self.stream);
        let (pp, _g5) = part.device_ptr_mut(&self.stream);
        let (yp, _g6) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; part >= 8 * out_dim * batch f32
        check(unsafe {
            f(
                dp as *const _,
                sp as *const _,
                xqp as *const _,
                xsp as *const _,
                pp as *mut _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Build a byte-passthrough (bs) decode plane: raw row-major e4m3 bytes
    /// repacked to data-only lin boxes with the f32 [out/128][in/128] scale
    /// plane appended at the tail of `data`; `scale` is the 8-byte marker
    /// `f8d_gemm_mma_ks` dispatches on. CONSUMES the raw plane.
    pub fn f8w_build_lin_bs(
        &self,
        raw: CudaSlice<u8>,
        scales: &[f32],
        in_dim: usize,
        out_dim: usize,
    ) -> Result<RepackedMxfp4, GpuError> {
        let nrt = out_dim.div_ceil(128);
        let nk = in_dim / 128;
        debug_assert_eq!(scales.len(), nrt * nk);
        let marker: CudaSlice<u8> = self.stream.alloc_zeros(8).map_err(drv)?;
        let mut data: CudaSlice<u8> = self
            .stream
            .alloc_zeros(nrt * nk * 16384 + nrt * nk * 4)
            .map_err(drv)?;
        self.f8w_repack_lin_bs(&raw, &mut data, in_dim, out_dim)?;
        {
            let sb: &[u8] = unsafe {
                std::slice::from_raw_parts(scales.as_ptr() as *const u8, scales.len() * 4)
            };
            let mut tail = data.slice_mut(nrt * nk * 16384..);
            self.stream.memcpy_htod(sb, &mut tail).map_err(drv)?;
        }
        self.stream.synchronize().map_err(drv)?;
        drop(raw);
        self.trim_mem_pool();
        Ok(RepackedMxfp4 {
            data,
            scale: marker,
        })
    }

    /// Load-time byte-passthrough repack: raw row-major e4m3 -> data-only lin
    /// boxes (no per-32 strip; scales live in the separate f32 plane).
    pub fn f8w_repack_lin_bs(
        &self,
        data: &CudaSlice<u8>,
        dst: &mut CudaSlice<u8>,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .f8w_repack_lin_bs
            .ok_or(GpuError::MissingOp("f8w_repack_lin_bs"))?;
        debug_assert!(dst.len() >= out_dim.div_ceil(128) * (in_dim / 128) * 16384);
        let (dp, _g1) = data.device_ptr(&self.stream);
        let (op, _g2) = dst.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract
        check(unsafe {
            f(
                dp as *const _,
                op as *mut _,
                in_dim as u32,
                out_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Tile-linear prefill GEMM (tma_kt twin). `row_off` must be a multiple
    /// of 128 (box rows) - every fused-plane slice boundary is.
    #[allow(clippy::too_many_arguments)]
    pub fn f8_gemm_lin_kt(
        &self,
        w: &RepackedMxfp4,
        row_off: usize,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<u8>,
        y: &mut CudaSlice<f32>,
        in_dim: usize,
        out_dim: usize,
        batch: usize,
        o16: bool,
    ) -> Result<(), GpuError> {
        debug_assert_eq!(row_off % 128, 0, "lin row_off must be box-aligned");
        if w.scale.len() == 12 {
            // rowwise plane (see f8_gemm_lin for the tail invariant)
            let f = self
                .kernels
                .f8_gemm_lin_kt_r
                .ok_or(GpuError::MissingOp("f8_gemm_lin_kt_r"))?;
            let padded = w.data.len() / (in_dim + 1);
            let box_off = (row_off / 128) * (in_dim / 128) * 16384;
            let wse_off = padded * in_dim + row_off;
            let (dp, _g1) = w.data.device_ptr(&self.stream);
            let (xqp, _g3) = xq.device_ptr(&self.stream);
            let (xsp, _g4) = xs.device_ptr(&self.stream);
            let (yp, _g5) = y.device_ptr_mut(&self.stream);
            return check(unsafe {
                f(
                    (dp as usize + box_off) as *const _,
                    (dp as usize + wse_off) as *const _,
                    xqp as *const _,
                    xsp as *const _,
                    yp as *mut _,
                    in_dim as u32,
                    out_dim as u32,
                    batch as u32,
                    o16 as u32,
                    self.stream_ptr(),
                )
            });
        }
        let f = self
            .kernels
            .f8_gemm_lin_kt
            .ok_or(GpuError::MissingOp("f8_gemm_lin_kt"))?;
        let box_off = (row_off / 128) * (in_dim / 128) * 16896;
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (xqp, _g3) = xq.device_ptr(&self.stream);
        let (xsp, _g4) = xs.device_ptr(&self.stream);
        let (yp, _g5) = y.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                (dp as usize + box_off) as *const _,
                xqp as *const _,
                xsp as *const _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                o16 as u32,
                self.stream_ptr(),
            )
        })
    }

    /// q36 DN: one prefill GEMM over a fused two-consumer lin plane
    /// (in_qkv|gate), two-buffer epilogue - rows `[0, out1)` land in `y1`
    /// and `[out1, out1+out2)` in `y2`, each at its own row stride, so both
    /// consumers keep their layouts while the grid pays one wave tail
    /// instead of two (the split gate launch alone is a ~1.0x fractional
    /// wave at every mixed-tick r on a 188-SM die). Bit-exact vs the pair
    /// of `f8_gemm_w8` calls it replaces wherever both take kt3.
    /// `Ok(false)` = route not covered (non-lin plane, no pack entry, or
    /// the pack declined) - the caller keeps its two-launch pair.
    #[allow(clippy::too_many_arguments)]
    pub fn f8_gemm_w8_split(
        &self,
        w: &RepackedMxfp4,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<u8>,
        y1: &mut CudaSlice<f32>,
        y2: &mut CudaSlice<f32>,
        in_dim: usize,
        out1: usize,
        out2: usize,
        batch: usize,
    ) -> Result<bool, GpuError> {
        // strip lin planes only (marker scale 4): the rowwise class would
        // need a _r twin and non-lin planes ride other kernels entirely
        if w.scale.len() != 4 {
            return Ok(false);
        }
        let Some(f) = self.kernels.f8_gemm_lin_kt_split else {
            return Ok(false);
        };
        debug_assert_eq!(
            w.data.len(),
            (out1 + out2).div_ceil(128) * (in_dim / 128) * 16896
        );
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (xqp, _g2) = xq.device_ptr(&self.stream);
        let (xsp, _g3) = xs.device_ptr(&self.stream);
        let (y1p, _g4) = y1.device_ptr_mut(&self.stream);
        let (y2p, _g5) = y2.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; y1/y2 sized [batch*out1]/[batch*out2]
        let rc = unsafe {
            f(
                dp as *const _,
                xqp as *const _,
                xsp as *const _,
                y1p as *mut _,
                y2p as *mut _,
                out1 as u32,
                in_dim as u32,
                (out1 + out2) as u32,
                batch as u32,
                self.stream_ptr(),
            )
        };
        if rc == -2 {
            return Ok(false);
        }
        check(rc)?;
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::info!("DN split GEMM engaged: in={in_dim} out={out1}|{out2} r={batch}");
        });
        Ok(true)
    }

    /// bf16-out f8 prefill GEMM (tma route; probe with `has_f8_o16`).
    #[allow(clippy::too_many_arguments)]
    pub fn f8_gemm_w8_o16(
        &self,
        w: &RepackedMxfp4,
        row_off: usize,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<u8>,
        y: &mut CudaSlice<f32>,
        in_dim: usize,
        out_dim: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        if w.scale.len() == 4 || w.scale.len() == 12 {
            // lin/rowwise plane (marker scale): the tma_kt twin with bf16
            // epilogue - f8_gemm_lin_kt dispatches the rowwise arm itself
            return self.f8_gemm_lin_kt(w, row_off, xq, xs, y, in_dim, out_dim, batch, true);
        }
        let f = self
            .kernels
            .f8_gemm_w8_o16
            .ok_or(GpuError::MissingOp("f8_gemm_w8_o16"))?;
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (scp, _g2) = w.scale.device_ptr(&self.stream);
        let (xqp, _g3) = xq.device_ptr(&self.stream);
        let (xsp, _g4) = xs.device_ptr(&self.stream);
        let (yp, _g5) = y.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                (dp as usize + row_off * in_dim) as *const _,
                (scp as usize + row_off * (in_dim / 32)) as *const _,
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

    /// Fused gate|up-layout bf16 swiglu + e4m3 quant: one [rows][2*ff] bf16
    /// buffer (the single-GEMM prefill FFN epilogue), gate cols [0,ff), up
    /// [ff,2ff). Per-element math identical to `quantize_e4m3_swiglu_b16`.
    pub fn quantize_e4m3_swiglu_b16_gu(
        &self,
        gu_b16: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        scale: &mut CudaSlice<u8>,
        n: usize,
        ff: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .quantize_e4m3_swiglu_b16_gu
            .ok_or(GpuError::MissingOp("quantize_e4m3_swiglu_b16_gu"))?;
        let (gp, _g1) = gu_b16.device_ptr(&self.stream);
        let (qp, _g2) = q.device_ptr_mut(&self.stream);
        let (sp, _g3) = scale.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                gp as *const _,
                qp as *mut _,
                sp as *mut _,
                n as u32,
                ff as u32,
                self.stream_ptr(),
            )
        })
    }

    /// bf16-input swiglu + e4m3 quant (the o16 epilogue's consumer).
    pub fn quantize_e4m3_swiglu_b16(
        &self,
        gate_b16: &CudaSlice<f32>,
        up_b16: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        scale: &mut CudaSlice<u8>,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .quantize_e4m3_swiglu_b16
            .ok_or(GpuError::MissingOp("quantize_e4m3_swiglu_b16"))?;
        let (gp, _g1) = gate_b16.device_ptr(&self.stream);
        let (up_p, _g2) = up_b16.device_ptr(&self.stream);
        let (qp, _g3) = q.device_ptr_mut(&self.stream);
        let (sp, _g4) = scale.device_ptr_mut(&self.stream);
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

    /// True when the o16 epilogue pair is loadable (runtime NotSupported on
    /// non-TMA routes still applies - probe by a live call at plane build).
    pub fn has_f8_o16(&self) -> bool {
        self.kernels.f8_gemm_w8_o16.is_some() && self.kernels.quantize_e4m3_swiglu_b16.is_some()
    }

    pub fn has_swiglu_b16_gu(&self) -> bool {
        self.kernels.quantize_e4m3_swiglu_b16_gu.is_some()
    }

    /// Native-bf16 bytes -> f8w plane (the fp8 ingestion lane): stage the
    /// checkpoint bytes, convert on device - same e4m3+e8m0 class as the
    /// Q8-derived planes but with no Q8 double quantization. `n` = element
    /// count (must be a multiple of 32).
    pub fn bf16_to_f8w(&self, bf16_bytes: &[u8]) -> Result<RepackedMxfp4, GpuError> {
        let f = self
            .kernels
            .bf16_to_f8w
            .ok_or(GpuError::MissingOp("bf16_to_f8w"))?;
        let n = bf16_bytes.len() / 2;
        assert_eq!(n % 32, 0, "bf16 plane must be 32-aligned");
        let n_blocks = n / 32;
        let mut data = self.alloc_u8(n)?;
        let mut scale = self.alloc_u8(n_blocks)?;
        self.with_staged_raw(bf16_bytes, |sp| {
            let (dp, _g1) = data.device_ptr_mut(&self.stream);
            let (scp, _g2) = scale.device_ptr_mut(&self.stream);
            check(unsafe {
                f(
                    sp as *const _,
                    dp as *mut _,
                    scp as *mut _,
                    n_blocks as u64,
                    self.stream_ptr(),
                )
            })
        })?;
        Ok(RepackedMxfp4 { data, scale })
    }

    /// `bf16_to_f8w` over PRE-STAGED device bytes (first `n_bytes` of
    /// `staged`). Lets a bulk-ingestion loop reuse one staging buffer -
    /// per-tensor clone_htod transients leave un-trimmable mempool holes
    /// that shrink the serving pool (churn preemption waves).
    pub fn bf16_to_f8w_dev(
        &self,
        staged: &CudaSlice<u8>,
        n_bytes: usize,
    ) -> Result<RepackedMxfp4, GpuError> {
        let f = self
            .kernels
            .bf16_to_f8w
            .ok_or(GpuError::MissingOp("bf16_to_f8w"))?;
        let n = n_bytes / 2;
        assert_eq!(n % 32, 0, "bf16 plane must be 32-aligned");
        assert!(n_bytes <= staged.len(), "staging buffer too small");
        let n_blocks = n / 32;
        let mut data = self.alloc_u8(n)?;
        let mut scale = self.alloc_u8(n_blocks)?;
        {
            let (sp, _g0) = staged.device_ptr(&self.stream);
            let (dp, _g1) = data.device_ptr_mut(&self.stream);
            let (scp, _g2) = scale.device_ptr_mut(&self.stream);
            check(unsafe {
                f(
                    sp as *const _,
                    dp as *mut _,
                    scp as *mut _,
                    n_blocks as u64,
                    self.stream_ptr(),
                )
            })?;
        }
        Ok(RepackedMxfp4 { data, scale })
    }

    /// bf16 -> f8r plane (per-ROW e8m0 scale; the scale-free stream).
    pub fn bf16_to_f8r(
        &self,
        bf16_bytes: &[u8],
        in_dim: usize,
        out_dim: usize,
    ) -> Result<RepackedMxfp4, GpuError> {
        let f = self
            .kernels
            .bf16_to_f8r
            .ok_or(GpuError::MissingOp("bf16_to_f8r"))?;
        assert_eq!(bf16_bytes.len(), in_dim * out_dim * 2);
        let mut data = self.alloc_u8(in_dim * out_dim)?;
        let mut scale = self.alloc_u8(out_dim)?;
        self.with_staged_raw(bf16_bytes, |sp| {
            let (dp, _g1) = data.device_ptr_mut(&self.stream);
            let (scp, _g2) = scale.device_ptr_mut(&self.stream);
            check(unsafe {
                f(
                    sp as *const _,
                    dp as *mut _,
                    scp as *mut _,
                    in_dim as u32,
                    out_dim as u32,
                    self.stream_ptr(),
                )
            })
        })?;
        Ok(RepackedMxfp4 { data, scale })
    }

    /// Two bf16 planes -> one fused f8r plane (concat along out rows).
    pub fn bf16_to_f8r_concat2(
        &self,
        a: &[u8],
        b: &[u8],
        in_dim: usize,
    ) -> Result<RepackedMxfp4, GpuError> {
        let f = self
            .kernels
            .bf16_to_f8r
            .ok_or(GpuError::MissingOp("bf16_to_f8r"))?;
        let (oa, ob) = (a.len() / 2 / in_dim, b.len() / 2 / in_dim);
        let mut data = self.alloc_u8(in_dim * (oa + ob))?;
        let mut scale = self.alloc_u8(oa + ob)?;
        for (bytes, roff, rows) in [(a, 0usize, oa), (b, oa, ob)] {
            self.with_staged_raw(bytes, |sp| {
                let (dp, _g1) = data.device_ptr_mut(&self.stream);
                let (scp, _g2) = scale.device_ptr_mut(&self.stream);
                check(unsafe {
                    f(
                        sp as *const _,
                        (dp as usize + roff * in_dim) as *mut _,
                        (scp as usize + roff) as *mut _,
                        in_dim as u32,
                        rows as u32,
                        self.stream_ptr(),
                    )
                })
            })?;
        }
        Ok(RepackedMxfp4 { data, scale })
    }

    /// Per-row-scale e4m3 decode GEMM (f8r planes; f8d contract).
    #[allow(clippy::too_many_arguments)]
    pub fn f8r_gemm_mma_ks(
        &self,
        w: &RepackedMxfp4,
        in_dim: usize,
        out_dim: usize,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<u8>,
        part: &mut CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .f8r_gemm_mma_ks
            .ok_or(GpuError::MissingOp("f8r_gemm_mma_ks"))?;
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (scp, _g2) = w.scale.device_ptr(&self.stream);
        let (xqp, _g3) = xq.device_ptr(&self.stream);
        let (xsp, _g4) = xs.device_ptr(&self.stream);
        let (pp, _g5) = part.device_ptr_mut(&self.stream);
        let (yp, _g6) = y.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                dp as *const _,
                scp as *const _,
                xqp as *const _,
                xsp as *const _,
                pp as *mut _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Fused-landing swiglu + e4m3 quant (one kernel; decode step glue).
    pub fn swiglu_fused_e4m3(
        &self,
        fused: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        scale: &mut CudaSlice<u8>,
        ff: usize,
        rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .swiglu_fused_e4m3
            .ok_or(GpuError::MissingOp("swiglu_fused_e4m3"))?;
        let (fp, _g1) = fused.device_ptr(&self.stream);
        let (qp, _g2) = q.device_ptr_mut(&self.stream);
        let (sp, _g3) = scale.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                fp as *const _,
                qp as *mut _,
                sp as *mut _,
                ff as u32,
                rows as u32,
                self.stream_ptr(),
            )
        })
    }

    /// add+rmsnorm writing both xn and e4m3 staging (decode norm+quant fuse).
    #[allow(clippy::too_many_arguments)]
    pub fn add_rmsnorm_e4m3_xn(
        &self,
        x: &mut CudaSlice<f32>,
        proj: Option<&CudaSlice<f32>>,
        w: &CudaSlice<f32>,
        xn: &mut CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        scale: &mut CudaSlice<u8>,
        n: usize,
        batch: usize,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .add_rmsnorm_e4m3_xn
            .ok_or(GpuError::MissingOp("add_rmsnorm_e4m3_xn"))?;
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let pg = proj.map(|p| p.device_ptr(&self.stream));
        let pp = pg
            .as_ref()
            .map_or(std::ptr::null(), |(p, _)| *p as *const core::ffi::c_void);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let (xnp, _g3) = xn.device_ptr_mut(&self.stream);
        let (qp, _g4) = q.device_ptr_mut(&self.stream);
        let (sp, _g5) = scale.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                xp as *mut _,
                pp,
                wp as *const _,
                xnp as *mut _,
                qp as *mut _,
                sp as *mut _,
                n as u32,
                batch as u32,
                eps,
                self.stream_ptr(),
            )
        })
    }

    /// True when the decode norm+quant fuse ships.
    pub fn has_add_rmsnorm_e4m3_xn(&self) -> bool {
        self.kernels.add_rmsnorm_e4m3_xn.is_some()
    }

    /// b16-residual twin: `proj` bytes are bf16 (the o16 prefill chain's
    /// post-norm residual). Same contract otherwise.
    #[allow(clippy::too_many_arguments)]
    pub fn add_rmsnorm_e4m3_xn_b16(
        &self,
        x: &mut CudaSlice<f32>,
        proj: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        xn: &mut CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        scale: &mut CudaSlice<u8>,
        n: usize,
        batch: usize,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .add_rmsnorm_e4m3_xn_b16
            .ok_or(GpuError::MissingOp("add_rmsnorm_e4m3_xn_b16"))?;
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (pp, _g0) = proj.device_ptr(&self.stream);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let (xnp, _g3) = xn.device_ptr_mut(&self.stream);
        let (qp, _g4) = q.device_ptr_mut(&self.stream);
        let (sp, _g5) = scale.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                xp as *mut _,
                pp as *const _,
                wp as *const _,
                xnp as *mut _,
                qp as *mut _,
                sp as *mut _,
                n as u32,
                batch as u32,
                eps,
                self.stream_ptr(),
            )
        })
    }

    /// True when the b16-residual norm+quant fuse ships.
    pub fn has_add_rmsnorm_e4m3_xn_b16(&self) -> bool {
        self.kernels.add_rmsnorm_e4m3_xn_b16.is_some()
    }

    /// Fused gated-rmsnorm + e4m3 quant (DN out_proj prefill glue): q/scale
    /// in quantize_e4m3's block layout - scale math bit-matches
    /// pd_e4m3_quant4. `out` is optional (GDN formulation band):
    /// `None` skips the f32 store entirely for paths whose only consumer is
    /// the fp8 GEMM's q/scale planes; `Some` keeps the fallback-consumer
    /// write.
    #[allow(clippy::too_many_arguments)]
    pub fn gated_rmsnorm_e4m3(
        &self,
        x: &CudaSlice<f32>,
        z: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        out: Option<&mut CudaSlice<f32>>,
        q: &mut CudaSlice<i8>,
        scale: &mut CudaSlice<u8>,
        n_rows: usize,
        d: usize,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .gated_rmsnorm_e4m3
            .ok_or(GpuError::MissingOp("gated_rmsnorm_e4m3"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (zp, _g2) = z.device_ptr(&self.stream);
        let (wp, _g3) = w.device_ptr(&self.stream);
        let mut _g4 = None;
        let op = match out {
            Some(o) => {
                let (p, g) = o.device_ptr_mut(&self.stream);
                _g4 = Some(g);
                p as *mut core::ffi::c_void
            }
            None => core::ptr::null_mut(),
        };
        let (qp, _g5) = q.device_ptr_mut(&self.stream);
        let (sp, _g6) = scale.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                xp as *const _,
                zp as *const _,
                wp as *const _,
                op,
                qp as *mut _,
                sp as *mut _,
                n_rows as u32,
                d as u32,
                eps,
                self.stream_ptr(),
            )
        })
    }

    /// True when the gated-rmsnorm e4m3 fuse ships.
    pub fn has_gated_rmsnorm_e4m3(&self) -> bool {
        self.kernels.gated_rmsnorm_e4m3.is_some()
    }

    /// Row-scale twin for the f8t out_proj arm: gated rmsnorm +
    /// per-ROW e4m3 in one launch. d must be 128, n_heads % 16 == 0.
    /// Bit-identical to `gated_rmsnorm` + `quantize_e4m3_row`.
    #[allow(clippy::too_many_arguments)]
    pub fn gated_rmsnorm_e4m3_row(
        &self,
        x: &CudaSlice<f32>,
        z: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        out: Option<&mut CudaSlice<f32>>,
        q: &mut CudaSlice<i8>,
        rscale: &mut CudaSlice<f32>,
        batch: usize,
        n_heads: usize,
        d: usize,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .gated_rmsnorm_e4m3_row
            .ok_or(GpuError::MissingOp("gated_rmsnorm_e4m3_row"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (zp, _g2) = z.device_ptr(&self.stream);
        let (wp, _g3) = w.device_ptr(&self.stream);
        let mut _g4 = None;
        let op = match out {
            Some(o) => {
                let (p, g) = o.device_ptr_mut(&self.stream);
                _g4 = Some(g);
                p as *mut core::ffi::c_void
            }
            None => core::ptr::null_mut(),
        };
        let (qp, _g5) = q.device_ptr_mut(&self.stream);
        let (sp, _g6) = rscale.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                xp as *const _,
                zp as *const _,
                wp as *const _,
                op,
                qp as *mut _,
                sp as *mut _,
                batch as u32,
                n_heads as u32,
                d as u32,
                eps,
                self.stream_ptr(),
            )
        })
    }

    /// True when the row-scale gated-rmsnorm fuse ships.
    pub fn has_gated_rmsnorm_e4m3_row(&self) -> bool {
        self.kernels.gated_rmsnorm_e4m3_row.is_some()
    }

    /// True when the fused swiglu+e4m3 ships.
    pub fn has_swiglu_fused_e4m3(&self) -> bool {
        self.kernels.swiglu_fused_e4m3.is_some()
    }

    /// True when the f8r pair ships.
    pub fn has_f8r(&self) -> bool {
        self.kernels.bf16_to_f8r.is_some() && self.kernels.f8r_gemm_mma_ks.is_some()
    }

    /// The f8row dense-FFN width chain (qwen35's checkpoint-exact fp8 MLP
    /// lane): per-row e4m3 staging, the b=1 row GEMV, and the width GEMM
    /// whose launcher elects the decode/wave arms. `swiglu_quant_e4m3_row`
    /// is optional (the arm falls back to swiglu + quantize_e4m3_row).
    pub fn has_f8row_dense_ffn(&self) -> bool {
        self.kernels.f8row_gemm.is_some()
            && self.kernels.f8r_gemv.is_some()
            && self.kernels.quantize_e4m3_row.is_some()
            && self.kernels.swiglu.is_some()
    }

    /// Two bf16 planes -> one fused f8w plane (concat along out rows) - the
    /// native-checkpoint twin of `q8_0_to_f8w_concat2`.
    pub fn bf16_to_f8w_concat2(
        &self,
        a_bytes: &[u8],
        b_bytes: &[u8],
    ) -> Result<RepackedMxfp4, GpuError> {
        let f = self
            .kernels
            .bf16_to_f8w
            .ok_or(GpuError::MissingOp("bf16_to_f8w"))?;
        let (na, nb) = (a_bytes.len() / 2, b_bytes.len() / 2);
        assert_eq!((na % 32) + (nb % 32), 0, "bf16 planes must be 32-aligned");
        let mut data = self.alloc_u8(na + nb)?;
        let mut scale = self.alloc_u8((na + nb) / 32)?;
        for (bytes, off) in [(a_bytes, 0usize), (b_bytes, na)] {
            let nblk = bytes.len() / 64;
            self.with_staged_raw(bytes, |sp| {
                let (dp, _g1) = data.device_ptr_mut(&self.stream);
                let (scp, _g2) = scale.device_ptr_mut(&self.stream);
                check(unsafe {
                    f(
                        sp as *const _,
                        (dp as usize + off) as *mut _,
                        (scp as usize + off / 32) as *mut _,
                        nblk as u64,
                        self.stream_ptr(),
                    )
                })
            })?;
        }
        Ok(RepackedMxfp4 { data, scale })
    }

    /// N-way sibling of `q8_0_to_f8w_concat2` (the qkv triple merge).
    /// Q8_0 source(s) -> fused e4m3 -> tile-linear plane, in one call.
    ///
    /// The point is what it does not allocate. Done as separate steps, the
    /// chain allocates an e4m3 plane, allocates the linear output above it,
    /// then drops the e4m3 - and that hole sits under a live plane, so
    /// `cuMemPoolTrimTo` can never hand the block back. Per tensor, every
    /// layer. Here the e4m3 intermediate never escapes, so it lives in one
    /// grow-only scratch reused for every tensor: after the first call this
    /// allocates only the plane it returns, and frees nothing at all.
    ///
    /// `ws` is concatenated along out_dim, so this covers both the single-plane
    /// case and a fused qkv (which otherwise built three separate e4m3 planes
    /// plus a fourth to concat them into - five allocations and four frees per
    /// layer). Falls back to the plain concat when the layout is not taken,
    /// because then the e4m3 plane is the returned plane and cannot be scratch.
    pub fn q8_0_to_f8w_lin(
        &self,
        ws: &[&RepackedQ8],
        in_dim: usize,
        out_dim: usize,
        lin_on: bool,
    ) -> Result<RepackedMxfp4, GpuError> {
        if !lin_on || !in_dim.is_multiple_of(128) || !out_dim.is_multiple_of(16) {
            return self.q8_0_to_f8w_concatn(ws);
        }
        let cvt = self
            .kernels
            .q8_0_to_f8w
            .ok_or(GpuError::MissingOp("q8_0_to_f8w"))?;
        let rp = self
            .kernels
            .f8w_repack_lin
            .ok_or(GpuError::MissingOp("f8w_repack_lin"))?;
        let total: usize = ws
            .iter()
            .map(|w| w.dims.iter().product::<usize>() / 32)
            .sum();
        // grow-only scratch for the intermediate; never returned to a caller,
        // so an oversized buffer is fine (the repack reads in_dim/out_dim)
        let mut slot = self.conv_scratch.lock().unwrap_or_else(|e| e.into_inner());
        if slot
            .as_ref()
            .is_none_or(|(d, s)| d.len() < total * 32 || s.len() < total)
        {
            *slot = None; // drop before realloc so the grow can reuse it
            *slot = Some((self.alloc_u8(total * 32)?, self.alloc_u8(total)?));
        }
        let (data, scale) = slot.as_mut().expect("allocated above");
        let mut off = 0usize;
        for w in ws {
            assert_eq!(w.dims[0], ws[0].dims[0], "fused planes need one in_dim");
            let n = w.dims.iter().product::<usize>() / 32;
            let (qd, _g1) = w.data.device_ptr(&self.stream);
            let (qs, _g2) = w.scale.device_ptr(&self.stream);
            let (fd, _g3) = data.device_ptr_mut(&self.stream);
            let (fs, _g4) = scale.device_ptr_mut(&self.stream);
            // SAFETY: pack ABI v1; the scratch covers `total` blocks and each
            // segment writes its own [off, off+n) window
            check(unsafe {
                cvt(
                    qd as *const _,
                    qs as *const _,
                    (fd as usize + off * 32) as *mut _,
                    (fs as usize + off) as *mut _,
                    n as u64,
                    self.stream_ptr(),
                )
            })?;
            off += n;
        }
        let nrt = out_dim.div_ceil(128);
        let nk = in_dim / 128;
        // marker before the plane, as in f8w_repack_lin: a 4-byte allocation
        // made after a big one lands in its slab and pins it
        let marker: CudaSlice<u8> = self.stream.alloc_zeros(4).map_err(drv)?;
        let mut lin: CudaSlice<u8> = self.stream.alloc_zeros(nrt * nk * 16896).map_err(drv)?;
        {
            let (dp, _g1) = data.device_ptr(&self.stream);
            let (scp, _g2) = scale.device_ptr(&self.stream);
            let (lp, _g3) = lin.device_ptr_mut(&self.stream);
            // SAFETY: pack ABI v1; the scratch holds this tensor's e4m3 image
            check(unsafe {
                rp(
                    dp as *const _,
                    scp as *const _,
                    lp as *mut _,
                    in_dim as u32,
                    out_dim as u32,
                    self.stream_ptr(),
                )
            })?;
        }
        // fence before the scratch is handed to the next tensor
        self.stream.synchronize().map_err(drv)?;
        Ok(RepackedMxfp4 {
            data: lin,
            scale: marker,
        })
    }

    pub fn q8_0_to_f8w_concatn(&self, ws: &[&RepackedQ8]) -> Result<RepackedMxfp4, GpuError> {
        let f = self
            .kernels
            .q8_0_to_f8w
            .ok_or(GpuError::MissingOp("q8_0_to_f8w"))?;
        let total: usize = ws
            .iter()
            .map(|w| w.dims.iter().product::<usize>() / 32)
            .sum();
        let mut data = self.alloc_u8(total * 32)?;
        let mut scale = self.alloc_u8(total)?;
        let mut off = 0usize;
        for w in ws {
            assert_eq!(w.dims[0], ws[0].dims[0], "fused planes need one in_dim");
            let n = w.dims.iter().product::<usize>() / 32;
            let (qd, _g1) = w.data.device_ptr(&self.stream);
            let (qs, _g2) = w.scale.device_ptr(&self.stream);
            let (fd, _g3) = data.device_ptr_mut(&self.stream);
            let (fs, _g4) = scale.device_ptr_mut(&self.stream);
            check(unsafe {
                f(
                    qd as *const _,
                    qs as *const _,
                    (fd as usize + off * 32) as *mut _,
                    (fs as usize + off) as *mut _,
                    n as u64,
                    self.stream_ptr(),
                )
            })?;
            off += n;
        }
        Ok(RepackedMxfp4 { data, scale })
    }

    /// Convert two resident Q8 planes into one fused f8w plane concatenated
    /// along out_dim ([a-rows | b-rows]) - the DN in_qkv|gate_w merge without
    /// re-reading the GGUF (the repack block stream is per-row contiguous).
    pub fn q8_0_to_f8w_concat2(
        &self,
        a: &RepackedQ8,
        b: &RepackedQ8,
    ) -> Result<RepackedMxfp4, GpuError> {
        assert_eq!(a.dims[0], b.dims[0], "fused planes need one in_dim");
        let f = self
            .kernels
            .q8_0_to_f8w
            .ok_or(GpuError::MissingOp("q8_0_to_f8w"))?;
        let na = a.dims.iter().product::<usize>() / 32;
        let nb = b.dims.iter().product::<usize>() / 32;
        let mut data = self.alloc_u8((na + nb) * 32)?;
        let mut scale = self.alloc_u8(na + nb)?;
        for (w, off) in [(a, 0usize), (b, na)] {
            let n = w.dims.iter().product::<usize>() / 32;
            let (qd, _g1) = w.data.device_ptr(&self.stream);
            let (qs, _g2) = w.scale.device_ptr(&self.stream);
            let (fd, _g3) = data.device_ptr_mut(&self.stream);
            let (fs, _g4) = scale.device_ptr_mut(&self.stream);
            check(unsafe {
                f(
                    qd as *const _,
                    qs as *const _,
                    (fd as usize + off * 32) as *mut _,
                    (fs as usize + off) as *mut _,
                    n as u64,
                    self.stream_ptr(),
                )
            })?;
        }
        Ok(RepackedMxfp4 { data, scale })
    }

    pub fn q8_0_to_f8w(&self, w: &RepackedQ8) -> Result<RepackedMxfp4, GpuError> {
        let f = self
            .kernels
            .q8_0_to_f8w
            .ok_or(GpuError::MissingOp("q8_0_to_f8w"))?;
        let n_blocks = w.data.len() / 32;
        let mut data = self.alloc_u8(n_blocks * 32)?;
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
}
