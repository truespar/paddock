//! batched rope/norm/kv-append/qkv fusions + batch tails.

use super::error::*;
use super::*;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::f16;

impl GpuExecutor {
    /// Batched multi-head attention for one decode token (GQA + sinks + sliding
    /// window). `kc`/`vc` are the full per-layer caches; the kernel indexes from
    /// `first_pos` internally. Replaces the per-head gemv/softmax/gemv loop.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode(
        &self,
        q: &CudaSlice<f32>,
        kc: &CudaSlice<f16>,
        vc: &CudaSlice<f16>,
        sinks: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        first_pos: usize,
        n_pos: usize,
        kv_dim: usize,
        scale: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .attn_decode_f32
            .ok_or(GpuError::MissingOp("attn_decode"))?;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = kc.device_ptr(&self.stream);
        let (vp, _g3) = vc.device_ptr(&self.stream);
        let (sp, _g4) = sinks.device_ptr(&self.stream);
        let (op, _g5) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; buffers sized by caller, head_dim is a power of 2
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                sp as *const _,
                op as *mut _,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                first_pos as u32,
                n_pos as u32,
                kv_dim as u32,
                scale,
                self.stream_ptr(),
            )
        })
    }

    /// Batched MoE top-k router over `batch` tokens (bias folded): biased logits
    /// [batch, n_expert] -> `out_idx`/`out_w` [batch, k].
    #[allow(clippy::too_many_arguments)]
    pub fn moe_topk_batch(
        &self,
        logits: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        n_expert: usize,
        k: usize,
        out_idx: &mut CudaSlice<u32>,
        out_w: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .moe_topk_batch
            .ok_or(GpuError::MissingOp("moe_topk_batch"))?;
        let (lp, _g1) = logits.device_ptr(&self.stream);
        let (bp, _g2) = bias.device_ptr(&self.stream);
        let (ip, _g3) = out_idx.device_ptr_mut(&self.stream);
        let (wp, _g4) = out_w.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; logits [batch, n_expert]
        check(unsafe {
            f(
                lp as *const _,
                bp as *const _,
                n_expert as u32,
                k as u32,
                ip as *mut _,
                wp as *mut _,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Laguna sigmoid MoE router: selection = top-k over sigmoid(logits) +
    /// `bias` (the selection-only `exp_probs_b` correction); `out_w` = the
    /// UNBIASED sigmoid scores of the selected experts, sum-normalized,
    /// × `routed_scale` - ready for the down-combine unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_topk_sigmoid_batch(
        &self,
        logits: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        routed_scale: f32,
        n_expert: usize,
        k: usize,
        out_idx: &mut CudaSlice<u32>,
        out_w: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .moe_topk_sigmoid_batch
            .ok_or(GpuError::MissingOp("moe_topk_sigmoid_batch"))?;
        let (lp, _g1) = logits.device_ptr(&self.stream);
        let (bp, _g2) = bias.device_ptr(&self.stream);
        let (ip, _g3) = out_idx.device_ptr_mut(&self.stream);
        let (wp, _g4) = out_w.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; logits [batch, n_expert]
        check(unsafe {
            f(
                lp as *const _,
                bp as *const _,
                routed_scale,
                n_expert as u32,
                k as u32,
                ip as *mut _,
                wp as *mut _,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Kolibri 1 router (slot 748): selection = top-k over the RAW logits +
    /// `bias`; `out_w` = sigmoid of each selected raw logit * `routed_scale`,
    /// never renormalized. Same planes as `moe_topk_sigmoid_batch`.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_topk_logit_sigmoid_batch(
        &self,
        logits: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        routed_scale: f32,
        n_expert: usize,
        k: usize,
        out_idx: &mut CudaSlice<u32>,
        out_w: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .moe_topk_logit_sigmoid_batch
            .ok_or(GpuError::MissingOp("moe_topk_logit_sigmoid_batch"))?;
        let (lp, _g1) = logits.device_ptr(&self.stream);
        let (bp, _g2) = bias.device_ptr(&self.stream);
        let (ip, _g3) = out_idx.device_ptr_mut(&self.stream);
        let (wp, _g4) = out_w.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; logits [batch, n_expert]
        check(unsafe {
            f(
                lp as *const _,
                bp as *const _,
                routed_scale,
                n_expert as u32,
                k as u32,
                ip as *mut _,
                wp as *mut _,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_moe_topk_logit_sigmoid(&self) -> bool {
        self.kernels.moe_topk_logit_sigmoid_batch.is_some()
    }

    /// `qkv_norm_rope_batch` over q/k only, `v` passed through IN PLACE: with
    /// `vnorm` false the kernel stores each V element back exactly as it read
    /// it (same lane, same index), so the V plane doubles as its own output
    /// and no scratch copy is written. NEOX rope; `params` as rope_yarn's.
    #[allow(clippy::too_many_arguments)]
    pub fn qk_norm_rope_batch(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &mut CudaSlice<f32>,
        qw: &CudaSlice<f32>,
        kw: &CudaSlice<f32>,
        qn: &mut CudaSlice<f32>,
        kn: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        n_head: usize,
        n_kv: usize,
        head_dim: usize,
        eps: f32,
        params: (f32, f32, f32, f32, f32, f32),
        rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .qkv_norm_rope_batch5
            .ok_or(GpuError::MissingOp("qkv_norm_rope_batch5"))?;
        let (theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale) = params;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = k.device_ptr(&self.stream);
        let (vp, _g3) = v.device_ptr_mut(&self.stream);
        let (qwp, _g4) = qw.device_ptr(&self.stream);
        let (kwp, _g5) = kw.device_ptr(&self.stream);
        let (qnp, _g6) = qn.device_ptr_mut(&self.stream);
        let (knp, _g7) = kn.device_ptr_mut(&self.stream);
        let (pp, _g8) = positions.device_ptr(&self.stream);
        // SAFETY: ABI contract; v == vn is the documented in-place pass-through
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                qwp as *const _,
                kwp as *const _,
                qnp as *mut _,
                knp as *mut _,
                vp as *mut _,
                pp as *const _,
                std::ptr::null(),
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
                1u32,
                0u32,
                self.stream_ptr(),
            )
        })
    }

    /// `moe_topk_sigmoid_batch` with the shared-expert fold-:
    /// each output row is `k + ns` wide, the trailing `ns` picks are the
    /// shared PSEUDO-expert ids `sh0..sh0+ns` with weight 1.0. One moe_align
    /// over these rows then covers routed + shared in a single bs pair
    /// launch (the separate 1-block shared pass ran at 10-12% of the stream
    /// roof).
    #[allow(clippy::too_many_arguments)]
    pub fn moe_topk_sigmoid_batch_sh(
        &self,
        logits: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        routed_scale: f32,
        n_expert: usize,
        k: usize,
        ns: usize,
        sh0: usize,
        out_idx: &mut CudaSlice<u32>,
        out_w: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .moe_topk_sigmoid_batch_sh
            .ok_or(GpuError::MissingOp("moe_topk_sigmoid_batch_sh"))?;
        let (lp, _g1) = logits.device_ptr(&self.stream);
        let (bp, _g2) = bias.device_ptr(&self.stream);
        let (ip, _g3) = out_idx.device_ptr_mut(&self.stream);
        let (wp, _g4) = out_w.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; logits [batch, n_expert], out rows k+ns wide
        check(unsafe {
            f(
                lp as *const _,
                bp as *const _,
                routed_scale,
                n_expert as u32,
                k as u32,
                ns as u32,
                sh0 as u32,
                ip as *mut _,
                wp as *mut _,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// One-launch f32 batched matvec for tiny GEMMs (the MoE router). See
    /// `KernelTableV1::matvec_f32_batch`; `w` is a [out_dim, in_dim] row-major
    /// DeviceTensor. Sole Class-A GEMM route since the phase-C cuBLAS
    /// deletion (P33: wins or ties the deleted cublasSgemm at every
    /// serving batch, warm).
    pub fn matvec_f32_batch(
        &self,
        w: &DeviceTensor,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        super::basic_ops::gemm_census("A-f32", w.dims[0], w.dims[1], batch);
        self.matvec_f32_raw(&w.buf, w.dims[0], w.dims[1], x, y, batch)
    }

    /// Unconditionally-tiled f32 GEMM over a bare [out, in] weight buffer -
    /// the k-quant interim's compute stage (wide ffn outs make the matvec
    /// tile's per-token weight re-read pathological). Distinct pack entry so
    /// the router (`matvec_f32_batch`) keeps its parity-pinned numerics.
    pub fn gemm_f32(
        &self,
        w: &CudaSlice<f32>,
        in_dim: usize,
        out_dim: usize,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .gemm_f32
            .ok_or(GpuError::MissingOp("gemm_f32"))?;
        let (wp, _g1) = w.device_ptr(&self.stream);
        let (xp, _g2) = x.device_ptr(&self.stream);
        let (yp, _g3) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; caller sizes x [batch, in_dim], y [batch, out_dim]
        check(unsafe {
            f(
                wp as *const _,
                xp as *const _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }
    /// K-split router matvec (slot 486): deterministic ascending-split fold,
    /// caller-owned scratch (>= 8 * batch * out_dim f32; the decode graph
    /// bakes the launch, so no allocation ever happens here). New summation
    /// order vs `matvec_f32_batch` - the token gates arbitrate.
    pub fn matvec_f32_ks(
        &self,
        w: &DeviceTensor,
        x: &CudaSlice<f32>,
        scratch: &mut CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        super::basic_ops::gemm_census("A-f32", w.dims[0], w.dims[1], batch);
        let f = self
            .kernels
            .matvec_f32_ks
            .ok_or(GpuError::MissingOp("matvec_f32_ks"))?;
        let (wp, _g1) = w.buf.device_ptr(&self.stream);
        let (xp, _g2) = x.device_ptr(&self.stream);
        let (sp, _g3) = scratch.device_ptr_mut(&self.stream);
        let (yp, _g4) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; caller sizes x [batch, in], y [batch, out],
        // scratch [8, batch, out]
        check(unsafe {
            f(
                wp as *const _,
                xp as *const _,
                sp as *mut _,
                yp as *mut _,
                w.dims[0] as u32,
                w.dims[1] as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_matvec_f32_ks(&self) -> bool {
        self.kernels.matvec_f32_ks.is_some()
    }

    /// `matvec_f32_batch` over a bare buffer + explicit dims - the k-quant
    /// prefill interim runs it over a transient dequant scratch that has no
    /// DeviceTensor identity (its dims change per weight).
    /// Batch-1 split-K twin of `matvec_f32_raw` (slot 519). `partials` is
    /// `out_dim * split` f32 and `counters` is `out_dim` u32, both caller-owned
    /// and address-stable; the kernel leaves `counters` zeroed so a captured
    /// graph replays cleanly. Deterministic: the winning block combines the
    /// row's partials in index order, so this is bit-stable run to run - it is
    /// not bit-equal to the single-block kernel, which sums the whole K in one
    /// tree, hence a separate slot and a call-site election.
    ///
    /// `Ok(false)` when the pack predates the slot, so the caller falls back.
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_f32_sk(
        &self,
        w: &CudaSlice<f32>,
        in_dim: usize,
        out_dim: usize,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        partials: &mut CudaSlice<f32>,
        counters: &mut CudaSlice<u32>,
        split: u32,
    ) -> Result<bool, GpuError> {
        let Some(f) = self.kernels.matvec_f32_sk else {
            return Ok(false);
        };
        let (wp, _g1) = w.device_ptr(&self.stream);
        let (xp, _g2) = x.device_ptr(&self.stream);
        let (yp, _g3) = y.device_ptr_mut(&self.stream);
        let (pp, _g4) = partials.device_ptr_mut(&self.stream);
        let (cp, _g5) = counters.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; scratch is sized by the caller for out_dim*split
        check(unsafe {
            f(
                wp as *const _,
                xp as *const _,
                yp as *mut _,
                pp as *mut _,
                cp as *mut _,
                in_dim as u32,
                out_dim as u32,
                split,
                self.stream_ptr(),
            )
        })?;
        Ok(true)
    }

    pub fn matvec_f32_raw(
        &self,
        w: &CudaSlice<f32>,
        in_dim: usize,
        out_dim: usize,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .matvec_f32_batch
            .ok_or(GpuError::MissingOp("matvec_f32_batch"))?;
        let (wp, _g1) = w.device_ptr(&self.stream);
        let (xp, _g2) = x.device_ptr(&self.stream);
        let (yp, _g3) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; caller sizes x [batch, in_dim], y [batch, out_dim]
        check(unsafe {
            f(
                wp as *const _,
                xp as *const _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// The pack carries the width-invariant f32 matvec (slot 687).
    pub fn has_matvec_f32_winv(&self) -> bool {
        self.kernels.matvec_f32_winv.is_some()
    }

    /// [`Self::matvec_f32_batch`] with the same K order at every row count -
    /// per row the bits of a 1-row call (slot 687). For decode rows that must
    /// score the same whatever the tick width.
    pub fn matvec_f32_winv(
        &self,
        w: &DeviceTensor,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .matvec_f32_winv
            .ok_or(GpuError::MissingOp("matvec_f32_winv"))?;
        let (wp, _g1) = w.buf.device_ptr(&self.stream);
        let (xp, _g2) = x.device_ptr(&self.stream);
        let (yp, _g3) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; caller sizes x [batch, in_dim], y [batch, out_dim]
        check(unsafe {
            f(
                wp as *const _,
                xp as *const _,
                yp as *mut _,
                w.dims[0] as u32,
                w.dims[1] as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// [`Self::matvec_f32_batch`] over an f16 weight plane and f16 activations,
    /// accumulating in f32 - the tensor-core class. Same layout convention
    /// (dims[0] = in_dim), and the caller naming the precision keeps "which
    /// precision am I running in" readable at the call site instead of hidden
    /// in a tensor field.
    pub fn matvec_batch_f16(
        &self,
        w: &HalfTensor,
        x16: &CudaSlice<f16>,
        y: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        self.gemm_f16_f32(&w.buf, x16, y, w.dims[0], w.dims[1], batch)
    }

    /// Batched fused MoE gate+up+swiglu over `batch` tokens (each with its own
    /// `idx`), into `out` [batch, n_active, ff].
    #[allow(clippy::too_many_arguments)]
    pub fn mxfp4_moe_gate_up_batch(
        &self,
        gate_w: &QuantTensor,
        gate_bias: &CudaSlice<f32>,
        up_w: &QuantTensor,
        up_bias: &CudaSlice<f32>,
        idx: &CudaSlice<u32>,
        x: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        in_dim: usize,
        ff: usize,
        n_active: usize,
        batch: usize,
        alpha: f32,
        limit: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .mxfp4_moe_gate_up_batch
            .ok_or(GpuError::MissingOp("mxfp4_moe_gate_up_batch"))?;
        let (gwp, _g1) = gate_w.bytes.device_ptr(&self.stream);
        let (gbp, _g2) = gate_bias.device_ptr(&self.stream);
        let (uwp, _g3) = up_w.bytes.device_ptr(&self.stream);
        let (ubp, _g4) = up_bias.device_ptr(&self.stream);
        let (ip, _g5) = idx.device_ptr(&self.stream);
        let (xp, _g6) = x.device_ptr(&self.stream);
        let (op, _g7) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; x [batch, in_dim], idx [batch, n_active]
        check(unsafe {
            f(
                gwp as *const _,
                gbp as *const _,
                uwp as *const _,
                ubp as *const _,
                ip as *const _,
                xp as *const _,
                op as *mut _,
                in_dim as u32,
                ff as u32,
                n_active as u32,
                batch as u32,
                alpha,
                limit,
                self.stream_ptr(),
            )
        })
    }

    /// Batched fused MoE down + weighted mix + residual add over `batch` tokens.
    /// `residual` [batch, embd] must be pre-zeroed (or hold the pre-MoE residual).
    #[allow(clippy::too_many_arguments)]
    pub fn mxfp4_moe_down_batch(
        &self,
        down_w: &QuantTensor,
        down_bias: &CudaSlice<f32>,
        idx: &CudaSlice<u32>,
        topk_w: &CudaSlice<f32>,
        fused: &CudaSlice<f32>,
        residual: &mut CudaSlice<f32>,
        ff: usize,
        embd: usize,
        n_active: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .mxfp4_moe_down_batch
            .ok_or(GpuError::MissingOp("mxfp4_moe_down_batch"))?;
        let (dwp, _g1) = down_w.bytes.device_ptr(&self.stream);
        let (dbp, _g2) = down_bias.device_ptr(&self.stream);
        let (ip, _g3) = idx.device_ptr(&self.stream);
        let (wp, _g4) = topk_w.device_ptr(&self.stream);
        let (fp, _g5) = fused.device_ptr(&self.stream);
        let (rp, _g6) = residual.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract
        check(unsafe {
            f(
                dwp as *const _,
                dbp as *const _,
                ip as *const _,
                wp as *const _,
                fp as *const _,
                rp as *mut _,
                ff as u32,
                embd as u32,
                n_active as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// granite's residual fusion: `scale_add` + `rmsnorm_batch` +
    /// `quantize_q8_sums` in one launch (round 3).
    ///
    /// `x` is updated in place with `x += res_scale * proj` (exactly
    /// `scale_add`'s `x += w*y`), `xn` gets the normalized row, and
    /// `q`/`scale`/`sums` get the Q8_0 staging the W4A8 GEMVs eat. `proj`
    /// is None for an entry norm with no residual to fold. `n % 32 == 0`.
    ///
    /// Bit-exact against the three it replaces. The fusion changes the
    /// sumsq's thread width - this is one row-per-CTA norm where
    /// `rmsnorm_batch` runs its own width election - and that is only free
    /// because the double-float accumulator is width-invariant bitwise
    /// Under the previous f64 accumulator this would
    /// have had to inherit the unfused width to stay exact.
    #[allow(clippy::too_many_arguments)]
    pub fn add_rmsnorm_q8_xn(
        &self,
        x: &mut CudaSlice<f32>,
        proj: Option<&CudaSlice<f32>>,
        w: &CudaSlice<f32>,
        xn: &mut CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        scale: &mut CudaSlice<f32>,
        sums: &mut CudaSlice<f32>,
        n: usize,
        batch: usize,
        eps: f32,
        res_scale: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .add_rmsnorm_q8_xn
            .ok_or(GpuError::MissingOp("add_rmsnorm_q8_xn"))?;
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let pg = proj.map(|p| p.device_ptr(&self.stream));
        let pp = pg
            .as_ref()
            .map_or(std::ptr::null(), |(p, _)| *p as *const core::ffi::c_void);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let (xnp, _g3) = xn.device_ptr_mut(&self.stream);
        let (qp, _g4) = q.device_ptr_mut(&self.stream);
        let (sp, _g5) = scale.device_ptr_mut(&self.stream);
        let (mp, _g6) = sums.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract
        check(unsafe {
            f(
                xp as *mut _,
                pp,
                wp as *const _,
                xnp as *mut _,
                qp as *mut _,
                sp as *mut _,
                mp as *mut _,
                n as u32,
                batch as u32,
                eps,
                res_scale,
                self.stream_ptr(),
            )
        })
    }

    /// True when the granite residual fusion ships in this pack.
    pub fn has_add_rmsnorm_q8_xn(&self) -> bool {
        self.kernels.add_rmsnorm_q8_xn.is_some()
    }

    /// Batched rmsnorm over `x` [batch, n] -> `out` [batch, n], shared weight.
    pub fn rmsnorm_batch(
        &self,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n: usize,
        eps: f32,
        batch: usize,
    ) -> Result<(), GpuError> {
        self.rmsnorm_batch_at(x, 0, w, out, n, eps, batch)
    }

    /// `rmsnorm_batch` writing back over its own input.
    ///
    /// The kernel reduces the whole row into shared memory and syncs before
    /// any store, and each thread then stores at exactly the index it read -
    /// so in-place is well-defined, and the only thing stopping the plain call
    /// is that Rust cannot hand out `&x` and `&mut x` at once. Used by the
    /// embedding preamble, which normalizes rows already gathered in place.
    pub fn rmsnorm_batch_inplace(
        &self,
        x: &mut CudaSlice<f32>,
        w: &CudaSlice<f32>,
        n: usize,
        eps: f32,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .rmsnorm_batch
            .ok_or(GpuError::MissingOp("rmsnorm_batch"))?;
        let (wp, _g1) = w.device_ptr(&self.stream);
        let (xp, _g2) = x.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; in==out is safe per the note above
        check(unsafe {
            f(
                xp as *const _,
                wp as *const _,
                xp as *mut _,
                n as u32,
                eps,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// `rmsnorm_batch` reading its input at ELEMENT offset `x_off` - the
    /// merged-projection consumer (e.g. the k rows of a fused [q|k|gate]
    /// GEMV landing). Same kernel, shifted base pointer.
    pub fn rmsnorm_batch_at(
        &self,
        x: &CudaSlice<f32>,
        x_off: usize,
        w: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n: usize,
        eps: f32,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .rmsnorm_batch
            .ok_or(GpuError::MissingOp("rmsnorm_batch"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let (op, _g3) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; x read from element x_off, [batch, n]
        check(unsafe {
            f(
                (xp + (x_off * 4) as u64) as *const _,
                wp as *const _,
                op as *mut _,
                n as u32,
                eps,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Batched YaRN rope in place over `x` [batch, n_heads*head_dim], each
    /// sequence rotated at its own `positions[b]`.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_yarn_batch(
        &self,
        x: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        n_heads: usize,
        head_dim: usize,
        params: (f32, f32, f32, f32, f32, f32),
        batch: usize,
    ) -> Result<(), GpuError> {
        self.rope_yarn_batch_conv(x, positions, n_heads, head_dim, params, batch, true)
    }

    /// NORM-convention rope (llama.cpp ROPE_TYPE_NORM): interleaved `(2k,
    /// 2k+1)` pairs. Granite's convention - everything else we serve is NEOX.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_yarn_batch_norm(
        &self,
        x: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        n_heads: usize,
        head_dim: usize,
        params: (f32, f32, f32, f32, f32, f32),
        batch: usize,
    ) -> Result<(), GpuError> {
        self.rope_yarn_batch_conv(x, positions, n_heads, head_dim, params, batch, false)
    }

    /// Shared body: `neox` picks the pair convention, which is a separate
    /// kernel instantiation in the pack (no runtime branch).
    #[allow(clippy::too_many_arguments)]
    fn rope_yarn_batch_conv(
        &self,
        x: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        n_heads: usize,
        head_dim: usize,
        params: (f32, f32, f32, f32, f32, f32),
        batch: usize,
        neox: bool,
    ) -> Result<(), GpuError> {
        let f = if neox {
            self.kernels
                .rope_yarn_batch
                .ok_or(GpuError::MissingOp("rope_yarn_batch"))?
        } else {
            self.kernels
                .rope_yarn_batch_norm
                .ok_or(GpuError::MissingOp("rope_yarn_batch_norm"))?
        };
        let (theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale) = params;
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (pp, _g2) = positions.device_ptr(&self.stream);
        // SAFETY: ABI contract
        check(unsafe {
            f(
                xp as *mut _,
                pp as *const _,
                n_heads as u32,
                head_dim as u32,
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

    /// Batched KV append: scatter each sequence's kv row into its own cache
    /// `[batch, max_ctx, kv_dim]` at its own `positions[b]`.
    /// `slots` maps each row to its KV slot; `None` means row index (decode).
    /// Prefill passes `Some([S; batch])` so all rows write into slot S.
    #[allow(clippy::too_many_arguments)]
    /// Development probe: what does a real model's K/V look like against e4m3's
    /// range?
    ///
    /// The parity gates proved the e4m3 codec and every kernel that reads an
    /// e4m3 cache are correct - but a faithful round-trip says nothing about
    /// whether it loses anything on real activations. e4m3 tops out at 448 and
    /// its steps are 32 wide in the top binade; a cache full of saturated or
    /// top-binade values would be information no kernel can recover. That is
    /// the one remaining way fp8 KV could produce wrong output with no kernel
    /// at fault, and the only case in which per-tensor/per-head KV scales would
    /// be the right answer rather than treating a symptom.
    ///
    /// So run the model on its normal f16 KV path - which stores the true
    /// values - and measure what e4m3 would have done to them. Arm with
    /// `PADDOCK_KV_MAGNITUDE=1`.
    ///
    /// Reads every appended row back to the host, so it is slow by
    /// construction: a development instrument, never a serving path.
    pub(crate) fn kv_probe(&self, kv: &CudaSlice<f32>, off: usize, n: usize, rows: usize) {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ON.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_KV_MAGNITUDE").is_some()) {
            return;
        }
        // PREFILL-SHAPED APPENDS ONLY. A synchronous device->host copy is
        // illegal inside a CUDA graph capture, and qwen35 captures its decode
        // graphs lazily per batch width - probing a decode-shaped tick killed
        // the process on the first request with nothing in the log. Prefill is
        // never captured, and it is where the bulk of the cache is written
        // anyway, so this costs the measurement nothing.
        if rows < 16 {
            return;
        }
        struct Acc {
            bins: [u64; 26], // binades 2^-12 .. 2^13, clamped at both ends
            total: u64,
            zeros: u64,
            max: f32,
            sat: u64,  // |v| > 448 - would clamp
            top: u64,  // |v| >= 256 - top binade, where the e4m3 step is 32
            tiny: u64, // 0 < |v| < 2^-9 - under min subnormal, flushes to zero
            se: f64,   // sum (v - e4m3(v))^2
            sv: f64,   // sum v^2
            calls: u64,
        }
        static ACC: std::sync::Mutex<Acc> = std::sync::Mutex::new(Acc {
            bins: [0; 26],
            total: 0,
            zeros: 0,
            max: 0.0,
            sat: 0,
            top: 0,
            tiny: 0,
            se: 0.0,
            sv: 0.0,
            calls: 0,
        });

        let Some(view) = kv.try_slice(off..off + n) else {
            return;
        };
        let Ok(host) = self.stream.clone_dtoh(&view) else {
            return;
        };
        let Ok(mut a) = ACC.lock() else { return };
        for &v in &host {
            let x = v.abs();
            a.total += 1;
            if x == 0.0 {
                a.zeros += 1;
                continue;
            }
            if x > a.max {
                a.max = x;
            }
            if x > 448.0 {
                a.sat += 1;
            }
            if x >= 256.0 {
                a.top += 1;
            }
            if x < 0.001_953_125 {
                a.tiny += 1;
            }
            let b = (x.log2().floor() as i32 + 12).clamp(0, 25) as usize;
            a.bins[b] += 1;
            let d = (v - e4m3_roundtrip(v)) as f64;
            a.se += d * d;
            a.sv += (v as f64) * (v as f64);
        }
        a.calls += 1;
        // periodic, because a serve never exits cleanly enough for a Drop
        if a.calls % 64 != 0 {
            return;
        }
        let t = a.total.max(1) as f64;
        let rel = if a.sv > 0.0 {
            (a.se / a.sv).sqrt()
        } else {
            0.0
        };
        let mut hist = String::new();
        for (i, &c) in a.bins.iter().enumerate() {
            if c > 0 {
                hist.push_str(&format!(
                    " 2^{}:{:.2}%",
                    i as i32 - 12,
                    100.0 * c as f64 / t
                ));
            }
        }
        eprintln!(
            "[kv-mag] n={} max={:.4} sat(>448)={} top(>=256)={} tiny(<2^-9)={:.2}%              zero={:.2}% e4m3_rel_rms={:.4}
[kv-mag] binades:{}",
            a.total, a.max, a.sat, a.top,
            100.0 * a.tiny as f64 / t, 100.0 * a.zeros as f64 / t, rel, hist
        );
    }

    pub fn kv_append_batch(
        &self,
        kv: &CudaSlice<f32>,
        cache: &mut CudaSlice<u8>,
        positions: &CudaSlice<u32>,
        slots: Option<&CudaSlice<u32>>,
        kv_dim: usize,
        max_ctx: usize,
        batch: usize,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        self.kv_probe(kv, 0, batch * kv_dim, batch);
        let f = self
            .kernels
            .kv_append_batch
            .ok_or(GpuError::MissingOp("kv_append_batch"))?;
        let (kp, _g1) = kv.device_ptr(&self.stream);
        let (cp, _g2) = cache.device_ptr_mut(&self.stream);
        let (pp, _g3) = positions.device_ptr(&self.stream);
        // hold the guard for the whole call; null when slots is None (identity)
        let slot_guard = slots.map(|s| s.device_ptr(&self.stream));
        let slp = match &slot_guard {
            Some((p, _)) => *p as *const core::ffi::c_void,
            None => std::ptr::null(),
        };
        // SAFETY: ABI contract; cache sized [batch, max_ctx, kv_dim] * dtype bytes
        check(unsafe {
            f(
                kp as *const _,
                cp as *mut _,
                pp as *const _,
                slp,
                kv_dim as u32,
                max_ctx as u32,
                batch as u32,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Fused-QKV consumer (yarn/no-norm family): rope(q)+rope(k)+append(k)+
    /// append(v) in one launch, reading the fused GEMM output. Rope math is
    /// bit-identical to `rope_yarn_batch`.
    #[allow(clippy::too_many_arguments)]
    pub fn qkv_rope_append_batch(
        &self,
        qkv: &CudaSlice<f32>,
        q_out: &mut CudaSlice<f32>,
        k_cache: &mut CudaSlice<u8>,
        v_cache: &mut CudaSlice<u8>,
        positions: &CudaSlice<u32>,
        slots: Option<&CudaSlice<u32>>,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_ctx: usize,
        params: (f32, f32, f32, f32, f32, f32),
        batch: usize,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .qkv_rope_append_batch
            .ok_or(GpuError::MissingOp("qkv_rope_append_batch"))?;
        let (theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale) = params;
        let (qkvp, _g1) = qkv.device_ptr(&self.stream);
        let (qp, _g2) = q_out.device_ptr_mut(&self.stream);
        let (kp, _g3) = k_cache.device_ptr_mut(&self.stream);
        let (vp, _g4) = v_cache.device_ptr_mut(&self.stream);
        let (pp, _g5) = positions.device_ptr(&self.stream);
        let slot_guard = slots.map(|s| s.device_ptr(&self.stream));
        let slp = match &slot_guard {
            Some((p, _)) => *p as *const core::ffi::c_void,
            None => std::ptr::null(),
        };
        // SAFETY: ABI contract; qkv is [batch, qdim + 2*kvdim]
        check(unsafe {
            f(
                qkvp as *const _,
                qp as *mut _,
                kp as *mut _,
                vp as *mut _,
                pp as *const _,
                slp,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                max_ctx as u32,
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

    /// Whether the pack ships the fused qkv rope/append consumer.
    /// wqkv all-in-one: GEMM partials + fused combine/rope/append. Values
    /// identical to `mm_pre(wqkv)` + `qkv_rope_append_batch`.
    #[allow(clippy::too_many_arguments)]
    pub fn q8_0_gemm_mma_ks_qkv_rope(
        &self,
        w: &RepackedQ8,
        bias: &CudaSlice<f32>,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        part: &mut CudaSlice<f32>,
        q_out: &mut CudaSlice<f32>,
        k_cache: &mut CudaSlice<u8>,
        v_cache: &mut CudaSlice<u8>,
        positions: &CudaSlice<u32>,
        slots: Option<&CudaSlice<u32>>,
        in_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_ctx: usize,
        params: (f32, f32, f32, f32, f32, f32),
        batch: usize,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_gemm_mma_ks_qkv_rope
            .ok_or(GpuError::MissingOp("q8_0_gemm_mma_ks_qkv_rope"))?;
        let (theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale) = params;
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (sp, _g2) = w.scale.device_ptr(&self.stream);
        let (bp, _g3) = bias.device_ptr(&self.stream);
        let (xqp, _g4) = xq.device_ptr(&self.stream);
        let (xsp, _g5) = xs.device_ptr(&self.stream);
        let (pp, _g6) = part.device_ptr_mut(&self.stream);
        let (qp, _g7) = q_out.device_ptr_mut(&self.stream);
        let (kp, _g8) = k_cache.device_ptr_mut(&self.stream);
        let (vp, _g9) = v_cache.device_ptr_mut(&self.stream);
        let (posp, _g10) = positions.device_ptr(&self.stream);
        let slp = slots.map(|sl| sl.device_ptr(&self.stream).0).unwrap_or(0);
        check(unsafe {
            f(
                dp as *const _,
                sp as *const _,
                xqp as *const _,
                xsp as *const _,
                bp as *const _,
                pp as *mut _,
                qp as *mut _,
                kp as *mut _,
                vp as *mut _,
                posp as *const _,
                slp as *const _,
                in_dim as u32,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                max_ctx as u32,
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

    pub fn has_ks_qkv_rope(&self) -> bool {
        self.kernels.q8_0_gemm_mma_ks_qkv_rope.is_some()
    }

    /// Whether the pack ships the gpt-oss paged fused-append twins (G1).
    pub fn has_gpt_oss_paged_append(&self) -> bool {
        self.kernels.qkv_rope_append_batch_paged.is_some()
            && self.kernels.q8_0_gemm_mma_ks_qkv_rope_paged.is_some()
    }

    /// Paged twin of `qkv_rope_append_batch`: K/V land in the block pool via
    /// `block_tables` (stride `blocks_per_slot`) instead of the dense
    /// `slot*max_ctx` stride. Bit-exact vs dense under an identity table.
    #[allow(clippy::too_many_arguments)]
    pub fn qkv_rope_append_batch_paged(
        &self,
        qkv: &CudaSlice<f32>,
        q_out: &mut CudaSlice<f32>,
        k_cache: &mut CudaSlice<u8>,
        v_cache: &mut CudaSlice<u8>,
        positions: &CudaSlice<u32>,
        slots: Option<&CudaSlice<u32>>,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        params: (f32, f32, f32, f32, f32, f32),
        batch: usize,
        block_tables: &CudaSlice<u32>,
        blocks_per_slot: usize,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .qkv_rope_append_batch_paged
            .ok_or(GpuError::MissingOp("qkv_rope_append_batch_paged"))?;
        let (theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale) = params;
        let (qkvp, _g1) = qkv.device_ptr(&self.stream);
        let (qp, _g2) = q_out.device_ptr_mut(&self.stream);
        let (kp, _g3) = k_cache.device_ptr_mut(&self.stream);
        let (vp, _g4) = v_cache.device_ptr_mut(&self.stream);
        let (pp, _g5) = positions.device_ptr(&self.stream);
        let slot_guard = slots.map(|s| s.device_ptr(&self.stream));
        let slp = match &slot_guard {
            Some((p, _)) => *p as *const core::ffi::c_void,
            None => std::ptr::null(),
        };
        let (btp, _g6) = block_tables.device_ptr(&self.stream);
        check(unsafe {
            f(
                qkvp as *const _,
                qp as *mut _,
                kp as *mut _,
                vp as *mut _,
                pp as *const _,
                slp,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                theta_scale,
                freq_scale,
                corr_low,
                corr_high,
                ext_factor,
                mscale,
                batch as u32,
                btp as *const _,
                blocks_per_slot as u32,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Paged twin of `q8_0_gemm_mma_ks_qkv_rope`: same GEMM, block-table K/V
    /// append. Bit-exact vs dense under an identity table.
    #[allow(clippy::too_many_arguments)]
    pub fn q8_0_gemm_mma_ks_qkv_rope_paged(
        &self,
        w: &RepackedQ8,
        bias: &CudaSlice<f32>,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        part: &mut CudaSlice<f32>,
        q_out: &mut CudaSlice<f32>,
        k_cache: &mut CudaSlice<u8>,
        v_cache: &mut CudaSlice<u8>,
        positions: &CudaSlice<u32>,
        slots: Option<&CudaSlice<u32>>,
        in_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        params: (f32, f32, f32, f32, f32, f32),
        batch: usize,
        block_tables: &CudaSlice<u32>,
        blocks_per_slot: usize,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_gemm_mma_ks_qkv_rope_paged
            .ok_or(GpuError::MissingOp("q8_0_gemm_mma_ks_qkv_rope_paged"))?;
        let (theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale) = params;
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (sp, _g2) = w.scale.device_ptr(&self.stream);
        let (bp, _g3) = bias.device_ptr(&self.stream);
        let (xqp, _g4) = xq.device_ptr(&self.stream);
        let (xsp, _g5) = xs.device_ptr(&self.stream);
        let (pp, _g6) = part.device_ptr_mut(&self.stream);
        let (qp, _g7) = q_out.device_ptr_mut(&self.stream);
        let (kp, _g8) = k_cache.device_ptr_mut(&self.stream);
        let (vp, _g9) = v_cache.device_ptr_mut(&self.stream);
        let (posp, _g10) = positions.device_ptr(&self.stream);
        let slp = slots.map(|sl| sl.device_ptr(&self.stream).0).unwrap_or(0);
        let (btp, _g11) = block_tables.device_ptr(&self.stream);
        check(unsafe {
            f(
                dp as *const _,
                sp as *const _,
                xqp as *const _,
                xsp as *const _,
                bp as *const _,
                pp as *mut _,
                qp as *mut _,
                kp as *mut _,
                vp as *mut _,
                posp as *const _,
                slp as *const _,
                in_dim as u32,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                theta_scale,
                freq_scale,
                corr_low,
                corr_high,
                ext_factor,
                mscale,
                batch as u32,
                btp as *const _,
                blocks_per_slot as u32,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_moe_combine_rmsnorm_quant(&self) -> bool {
        self.kernels.moe_combine_rmsnorm_quant_q8.is_some()
    }

    pub fn has_gu_interleave(&self) -> bool {
        self.kernels.mxfp4_gu_interleave.is_some()
    }

    /// Fuse two repacked MXFP4 planes (gate, up) into the g||u interleaved
    /// layout the bs / dp4a MoE kernels stream (one 128 B pair per row per
    /// KC=128 chunk). Returns the fused plane; callers drop the sources.
    pub fn gu_interleave(
        &self,
        gate: &RepackedMxfp4,
        up: &RepackedMxfp4,
        n_kb: usize,
        rows: usize,
    ) -> Result<CudaSlice<u8>, GpuError> {
        let f = self
            .kernels
            .mxfp4_gu_interleave
            .ok_or(GpuError::MissingOp("mxfp4_gu_interleave"))?;
        let pitch = n_kb.div_ceil(4) * 128;
        let mut dst = self.alloc_u8(rows * pitch)?;
        {
            let (gp, _g1) = gate.data.device_ptr(&self.stream);
            let (upp, _g2) = up.data.device_ptr(&self.stream);
            let (dp, _g3) = dst.device_ptr_mut(&self.stream);
            check(unsafe {
                f(
                    gp as *const _,
                    upp as *const _,
                    dp as *mut _,
                    n_kb as u32,
                    rows as u64,
                    self.stream_ptr(),
                )
            })?;
        }
        Ok(dst)
    }

    /// Cross-layer fold: MoE slot-combine (fixed slot order) + residual add
    /// + rmsnorm + q8 quantize in one per-token-row pass.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_combine_rmsnorm_quant_q8_batch(
        &self,
        x: &mut CudaSlice<f32>,
        part: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        qs: &mut CudaSlice<f32>,
        n: usize,
        n_active: usize,
        eps: f32,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .moe_combine_rmsnorm_quant_q8
            .ok_or(GpuError::MissingOp("moe_combine_rmsnorm_quant_q8"))?;
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (pp, _g2) = part.device_ptr(&self.stream);
        let (wp, _g3) = w.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        let (qp, _g5) = q.device_ptr_mut(&self.stream);
        let (sp, _g6) = qs.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                xp as *mut _,
                pp as *const _,
                wp as *const _,
                op as *mut _,
                qp as *mut _,
                sp as *mut _,
                n as u32,
                n_active as u32,
                eps,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_qkv_rope_append_batch(&self) -> bool {
        self.kernels.qkv_rope_append_batch.is_some()
    }

    /// Concatenate f32 device buffers (fused-QKV bias planes).
    pub fn concat_f32(&self, parts: &[&CudaSlice<f32>]) -> Result<CudaSlice<f32>, GpuError> {
        let total: usize = parts.iter().map(|p| p.len()).sum();
        let mut out = self.alloc(total)?;
        let mut off = 0usize;
        for p in parts {
            let ov = out
                .try_slice_mut(off..off + p.len())
                .ok_or_else(|| oob("concat_f32: range"))?;
            self.stream.memcpy_dtod(*p, &mut { ov }).map_err(drv)?;
            off += p.len();
        }
        Ok(out)
    }

    /// Fused residual-add + RMSNorm: `x += proj` (written back), `out =
    /// rmsnorm(x, w)` - bit-identical to add-then-norm, one launch.
    /// `add_rmsnorm_batch` with a residual MULTIPLIER: `x += pscale * proj`
    /// (written back), `out = rmsnorm(x, w)`. Bit-identical to `scale_add` +
    /// `rmsnorm_batch` - `pd_scale_add_kernel`'s `x[i] += w * y[i]` contracts
    /// to the same fma the fused kernel issues - and one launch instead of two.
    #[allow(clippy::too_many_arguments)]
    pub fn add_rmsnorm_scaled_batch(
        &self,
        x: &mut CudaSlice<f32>,
        proj: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n: usize,
        eps: f32,
        batch: usize,
        pscale: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .add_rmsnorm_scaled_batch
            .ok_or(GpuError::MissingOp("add_rmsnorm_scaled_batch"))?;
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (pp, _g2) = proj.device_ptr(&self.stream);
        let (wp, _g3) = w.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                xp as *mut _,
                pp as *const _,
                wp as *const _,
                op as *mut _,
                n as u32,
                eps,
                batch as u32,
                self.stream_ptr(),
                pscale,
            )
        })
    }

    /// Whether the pack carries the multiplier-folding fused norm.
    pub fn has_add_rmsnorm_scaled_batch(&self) -> bool {
        self.kernels.add_rmsnorm_scaled_batch.is_some()
    }

    /// `add_rmsnorm_scaled_batch` reading the predecessor nvf4 GEMM's RAW
    /// `nz` split-K partials in `part` (folded with `scale2`) as the residual,
    /// instead of a pre-reduced `proj` -- removes the `pd_nvf4_sk_reduce`
    /// launch + its y round trip. Bit-identical to reduce-then-scaled.
    /// `bias` is `None` for granite's projections.
    #[allow(clippy::too_many_arguments)]
    pub fn add_rmsnorm_scaled_from_parts(
        &self,
        x: &mut CudaSlice<f32>,
        part: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        bias: Option<&CudaSlice<f32>>,
        n: usize,
        eps: f32,
        batch: usize,
        pscale: f32,
        scale2: f32,
        nz: u32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .add_rmsnorm_scaled_from_parts
            .ok_or(GpuError::MissingOp("add_rmsnorm_scaled_from_parts"))?;
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (pp, _g2) = part.device_ptr(&self.stream);
        let (wp, _g3) = w.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        let bias_guard = bias.map(|b| b.device_ptr(&self.stream));
        let bp = match &bias_guard {
            Some((b, _)) => *b as *const f32,
            None => core::ptr::null(),
        };
        check(unsafe {
            f(
                xp as *mut _,
                pp as *const _,
                wp as *const _,
                op as *mut _,
                bp as *const _,
                n as u32,
                eps,
                batch as u32,
                pscale,
                scale2,
                nz,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_add_rmsnorm_scaled_from_parts(&self) -> bool {
        self.kernels.add_rmsnorm_scaled_from_parts.is_some()
    }

    /// nvf4 decode fold-2: fold `nz` raw split-K partials into
    /// the residual (`x = fmaf(pscale, sum*scale2 (+bias), x)`), rmsnorm with
    /// `w`, and write the nvf4 activation pair (`q`, `scale`) for the next
    /// W4A4 GEMM -- one launch for what was reduce + scale_add + rmsnorm +
    /// quantize (acc_sel 1, the rmsnorm_batch accumulator family) or
    /// add_rmsnorm_scaled_from_parts + quantize (acc_sel 0, the add_rmsnorm
    /// family). Bit-identical to those chains by construction
    /// (bench/nv4_fold2_cmp.cu: diffs=0). `out` optionally keeps the f32
    /// normed row. Decode widths only (batch < 64).
    #[allow(clippy::too_many_arguments)]
    pub fn add_rmsnorm_quant_nvf4_from_parts(
        &self,
        x: &mut CudaSlice<f32>,
        part: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        out: Option<&mut CudaSlice<f32>>,
        bias: Option<&CudaSlice<f32>>,
        q: &mut CudaSlice<i8>,
        scale: &mut CudaSlice<u8>,
        n: usize,
        eps: f32,
        batch: usize,
        pscale: f32,
        scale2: f32,
        nz: u32,
        acc_sel: u32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .add_rmsnorm_quant_nvf4_from_parts
            .ok_or(GpuError::MissingOp("add_rmsnorm_quant_nvf4_from_parts"))?;
        if q.len() < batch * n / 2
            || scale.len() < batch * n / 16
            || part.len() < nz as usize * batch * n
        {
            return Err(GpuError::Unsupported(format!(
                "add_rmsnorm_quant_nvf4_from_parts: q {} / scale {} / part {} too small for {batch} x {n} nz {nz}",
                q.len(),
                scale.len(),
                part.len()
            )));
        }
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (pp, _g2) = part.device_ptr(&self.stream);
        let (wp, _g3) = w.device_ptr(&self.stream);
        let out_guard = out.map(|o| o.device_ptr_mut(&self.stream));
        let op = match &out_guard {
            Some((o, _)) => *o as *mut f32,
            None => core::ptr::null_mut(),
        };
        let bias_guard = bias.map(|b| b.device_ptr(&self.stream));
        let bp = match &bias_guard {
            Some((b, _)) => *b as *const f32,
            None => core::ptr::null(),
        };
        let (qp, _g4) = q.device_ptr_mut(&self.stream);
        let (sp, _g5) = scale.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; sizes validated above
        check(unsafe {
            f(
                xp as *mut _,
                pp as *const _,
                wp as *const _,
                op as *mut _,
                bp as *const _,
                qp as *mut _,
                sp as *mut _,
                n as u32,
                eps,
                batch as u32,
                pscale,
                scale2,
                nz,
                acc_sel,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_add_rmsnorm_quant_nvf4_from_parts(&self) -> bool {
        self.kernels.add_rmsnorm_quant_nvf4_from_parts.is_some()
    }

    /// nvf4 decode fold-2: fold the gate|up GEMM's `nz` raw
    /// split-K partials (a [rows, 2*ff] merged plane per slice), swiglu, and
    /// nvf4-quantize the down input in one launch -- what was reduce +
    /// swiglu_fused + quantize, without the f32 y and ffn_gate round trips.
    /// Bit-identical to that chain (bench/nv4_fold2_cmp.cu: diffs=0).
    #[allow(clippy::too_many_arguments)]
    pub fn swiglu_quant_nvf4_from_parts(
        &self,
        part: &CudaSlice<f32>,
        bias: Option<&CudaSlice<f32>>,
        q: &mut CudaSlice<i8>,
        scale: &mut CudaSlice<u8>,
        ff: usize,
        n_rows: usize,
        scale2: f32,
        nz: u32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .swiglu_quant_nvf4_from_parts
            .ok_or(GpuError::MissingOp("swiglu_quant_nvf4_from_parts"))?;
        if q.len() < n_rows * ff / 2
            || scale.len() < n_rows * ff / 16
            || part.len() < nz as usize * n_rows * 2 * ff
        {
            return Err(GpuError::Unsupported(format!(
                "swiglu_quant_nvf4_from_parts: q {} / scale {} / part {} too small for {n_rows} x 2*{ff} nz {nz}",
                q.len(),
                scale.len(),
                part.len()
            )));
        }
        let (pp, _g1) = part.device_ptr(&self.stream);
        let bias_guard = bias.map(|b| b.device_ptr(&self.stream));
        let bp = match &bias_guard {
            Some((b, _)) => *b as *const f32,
            None => core::ptr::null(),
        };
        let (qp, _g2) = q.device_ptr_mut(&self.stream);
        let (sp, _g3) = scale.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; sizes validated above; ff % 32 == 0 checked by the launcher
        check(unsafe {
            f(
                pp as *const _,
                bp as *const _,
                qp as *mut _,
                sp as *mut _,
                ff as u32,
                n_rows as u32,
                scale2,
                nz,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_swiglu_quant_nvf4_from_parts(&self) -> bool {
        self.kernels.swiglu_quant_nvf4_from_parts.is_some()
    }

    /// Slot 536: [`Self::swiglu_quant_nvf4_from_parts`] over INTERLEAVED
    /// partials (`Nvf4Plane::gu_pairs`).
    #[allow(clippy::too_many_arguments)]
    pub fn swiglu_quant_nvf4_from_parts_il(
        &self,
        part: &CudaSlice<f32>,
        bias: Option<&CudaSlice<f32>>,
        q: &mut CudaSlice<i8>,
        scale: &mut CudaSlice<u8>,
        ff: usize,
        n_rows: usize,
        scale2: f32,
        nz: u32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .swiglu_quant_nvf4_from_parts_il
            .ok_or(GpuError::MissingOp("swiglu_quant_nvf4_from_parts_il"))?;
        if q.len() < n_rows * ff / 2
            || scale.len() < n_rows * ff / 16
            || part.len() < nz as usize * n_rows * 2 * ff
        {
            return Err(GpuError::Unsupported(format!(
                "swiglu_quant_nvf4_from_parts_il: q {} / scale {} / part {} too small for {n_rows} x 2*{ff} nz {nz}",
                q.len(),
                scale.len(),
                part.len()
            )));
        }
        let (pp, _g1) = part.device_ptr(&self.stream);
        let bias_guard = bias.map(|b| b.device_ptr(&self.stream));
        let bp = match &bias_guard {
            Some((b, _)) => *b as *const f32,
            None => core::ptr::null(),
        };
        let (qp, _g2) = q.device_ptr_mut(&self.stream);
        let (sp, _g3) = scale.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; sizes validated above
        check(unsafe {
            f(
                pp as *const _,
                bp as *const _,
                qp as *mut _,
                sp as *mut _,
                ff as u32,
                n_rows as u32,
                scale2,
                nz,
                self.stream_ptr(),
            )
        })
    }

    pub fn add_rmsnorm_batch(
        &self,
        x: &mut CudaSlice<f32>,
        proj: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n: usize,
        eps: f32,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .add_rmsnorm_batch
            .ok_or(GpuError::MissingOp("add_rmsnorm_batch"))?;
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (pp, _g2) = proj.device_ptr(&self.stream);
        let (wp, _g3) = w.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                xp as *mut _,
                pp as *const _,
                wp as *const _,
                op as *mut _,
                n as u32,
                eps,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Shared-expert scalar sigmoid gate fold: `dst[b] += sigmoid(x[b].w) * src[b]`.
    pub fn shexp_gate_add(
        &self,
        dst: &mut CudaSlice<f32>,
        src: &CudaSlice<f32>,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        n_out: usize,
        n_in: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .shexp_gate_add
            .ok_or(GpuError::MissingOp("shexp_gate_add"))?;
        let (dp, _g1) = dst.device_ptr_mut(&self.stream);
        let (sp, _g2) = src.device_ptr(&self.stream);
        let (xp, _g3) = x.device_ptr(&self.stream);
        let (wp, _g4) = w.device_ptr(&self.stream);
        check(unsafe {
            f(
                dp as *mut _,
                sp as *const _,
                xp as *const _,
                wp as *const _,
                n_out as u32,
                n_in as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// True when the pack carries the wide-batch Q8_0 dp4a GEMM.
    pub fn has_q8_0_gemm_mt_dp4a_wide(&self) -> bool {
        self.kernels.q8_0_gemm_mt_dp4a_wide.is_some()
    }

    /// Wide-batch (32 rows/weight-pass) dp4a GEMM - the B>=17 serving matmul.
    pub fn q8_0_gemm_mt_dp4a_wide(
        &self,
        w: &RepackedQ8,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_gemm_mt_dp4a_wide
            .ok_or(GpuError::MissingOp("q8_0_gemm_mt_dp4a_wide"))?;
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (scp, _g2) = w.scale.device_ptr(&self.stream);
        let (xqp, _g3) = xq.device_ptr(&self.stream);
        let (xsp, _g4) = xs.device_ptr(&self.stream);
        let (yp, _g5) = y.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                dp as *const _,
                scp as *const _,
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

    /// Batched embedding gather: `out[t] = table[tokens[t]]` (prefill embed).
    pub fn embed_gather_batch(
        &self,
        table: &CudaSlice<f32>,
        tokens: &CudaSlice<u32>,
        out: &mut CudaSlice<f32>,
        embd: usize,
        n_tokens: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .embed_gather_batch
            .ok_or(GpuError::MissingOp("embed_gather_batch"))?;
        let (tp, _g1) = table.device_ptr(&self.stream);
        let (kp, _g2) = tokens.device_ptr(&self.stream);
        let (op, _g3) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                tp as *const _,
                kp as *const _,
                op as *mut _,
                embd as u32,
                n_tokens as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Fused FFN gate+up+SwiGLU over two repacked Q8_0 weights: `out[o] =
    /// silu(gate·x) * (up·x)`, one launch. Replaces two `q8_0_gemv_repacked` + a
    /// `swiglu` and the intermediate buffers. `gate`/`up` share the input width.
    pub fn q8_0_ffn_gate_up_swiglu(
        &self,
        gate: &RepackedQ8,
        up: &RepackedQ8,
        x: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_ffn_gate_up_swiglu
            .ok_or(GpuError::MissingOp("q8_0_ffn_gate_up_swiglu"))?;
        let (in_dim, ff) = (gate.dims[0], gate.dims[1]);
        debug_assert_eq!(gate.dims, up.dims, "gate/up shape mismatch");
        let (gdp, _g1) = gate.data.device_ptr(&self.stream);
        let (gsp, _g2) = gate.scale.device_ptr(&self.stream);
        let (udp, _g3) = up.data.device_ptr(&self.stream);
        let (usp, _g4) = up.scale.device_ptr(&self.stream);
        let (xp, _g5) = x.device_ptr(&self.stream);
        let (op, _g6) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                gdp as *const _,
                gsp as *const _,
                udp as *const _,
                usp as *const _,
                xp as *const _,
                op as *mut _,
                in_dim as u32,
                ff as u32,
                self.stream_ptr(),
            )
        })
    }
}

/// Round an f32 through e4m3 (OCP: bias 7, no Inf, single NaN, max 448, min
/// normal 2^-6, min subnormal 2^-9), saturating on overflow the way
/// `__nv_fp8_e4m3` does. Round-to-nearest-even by binary search over the 127
/// ascending finite magnitudes - the same construction the parity gate uses, so
/// the probe and the gate agree on what "stored as e4m3" means.
fn e4m3_roundtrip(x: f32) -> f32 {
    static MAG: std::sync::OnceLock<[f32; 127]> = std::sync::OnceLock::new();
    let t = MAG.get_or_init(|| {
        let mut t = [0f32; 127];
        for (i, slot) in t.iter_mut().enumerate() {
            let e = ((i >> 3) & 0x0F) as i32;
            let m = (i & 0x07) as f32;
            *slot = if e == 0 {
                m * 2f32.powi(-9)
            } else {
                (1.0 + m / 8.0) * 2f32.powi(e - 7)
            };
        }
        t
    });
    let a = x.abs();
    let hi = t.partition_point(|&v| v < a);
    let q = if hi == 0 {
        0.0
    } else if hi >= t.len() {
        t[t.len() - 1]
    } else {
        let (dl, dh) = (a - t[hi - 1], t[hi] - a);
        if dh < dl || (dh == dl && hi % 2 == 0) {
            t[hi]
        } else {
            t[hi - 1]
        }
    };
    if x.is_sign_negative() { -q } else { q }
}
