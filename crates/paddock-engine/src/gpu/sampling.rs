//! device sampling, pipe advance, conv slot machinery.

use super::error::*;
use super::*;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

impl GpuExecutor {
    /// True when the pack ships the fused row sampler (`sample_rows`).
    pub fn has_sample_rows(&self) -> bool {
        self.kernels.sample_rows.is_some()
    }

    pub fn has_argmax_rows(&self) -> bool {
        self.kernels.argmax_rows.is_some()
    }

    /// True when the pack ships the pipelined-decode advance kernel.
    pub fn has_pipe_advance(&self) -> bool {
        self.kernels.pipe_advance.is_some()
    }

    pub fn has_rmsnorm_quant_q8(&self) -> bool {
        self.kernels.rmsnorm_quant_q8_batch.is_some()
    }

    pub fn has_add_rmsnorm_quant_e4m3(&self) -> bool {
        self.kernels.add_rmsnorm_quant_e4m3_batch.is_some()
    }

    pub fn has_add_rmsnorm_quant_mmq(&self) -> bool {
        self.kernels.add_rmsnorm_quant_mmq.is_some()
    }

    pub fn has_add_rmsnorm_quant_q8(&self) -> bool {
        self.kernels.add_rmsnorm_quant_q8_batch.is_some()
    }

    pub fn has_swiglu_quant_q8(&self) -> bool {
        self.kernels.swiglu_quant_q8.is_some()
    }

    /// Fused rmsnorm + Q8_0 quantize: one pass writes the f32 normed plane
    /// AND its int8/scale planes (kills the standalone quantize launch and
    /// the f32 round trip). Values identical to the two-kernel sequence.
    pub fn rmsnorm_quant_q8_batch(
        &self,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        qs: &mut CudaSlice<f32>,
        n: usize,
        eps: f32,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .rmsnorm_quant_q8_batch
            .ok_or(GpuError::MissingOp("rmsnorm_quant_q8_batch"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let (op, _g3) = out.device_ptr_mut(&self.stream);
        let (qp, _g4) = q.device_ptr_mut(&self.stream);
        let (sp, _g5) = qs.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                xp as *const _,
                wp as *const _,
                op as *mut _,
                qp as *mut _,
                sp as *mut _,
                n as u32,
                eps,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Fused residual-add + rmsnorm + e4m3/ue8m0 quantize (block-scale MoE
    /// input). x += proj is written back; out keeps the f32 plane (router).
    #[allow(clippy::too_many_arguments)]
    pub fn add_rmsnorm_quant_e4m3_batch(
        &self,
        x: &mut CudaSlice<f32>,
        proj: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        s8: &mut CudaSlice<u8>,
        n: usize,
        eps: f32,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .add_rmsnorm_quant_e4m3_batch
            .ok_or(GpuError::MissingOp("add_rmsnorm_quant_e4m3_batch"))?;
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (pp, _g2) = proj.device_ptr(&self.stream);
        let (wp, _g3) = w.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        let (qp, _g5) = q.device_ptr_mut(&self.stream);
        let (sp, _g6) = s8.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                xp as *mut _,
                pp as *const _,
                wp as *const _,
                op as *mut _,
                qp as *mut _,
                sp as *mut _,
                n as u32,
                eps,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Fused residual-add + rmsnorm + nvf4 quantize (glue rung).
    /// Replaces `add` + `rmsnorm_batch` + `quantize_nvf4`, which nemotron
    /// runs once per MoE layer - 23 times per decode tick, three
    /// latency-bound launches each. `out` still carries the f32 normed row
    /// because the router matvec reads it. Bit-exact to that chain.
    #[allow(clippy::too_many_arguments)]
    pub fn add_rmsnorm_quant_nvf4_batch(
        &self,
        x: &mut CudaSlice<f32>,
        proj: Option<&CudaSlice<f32>>,
        w: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        scale: &mut CudaSlice<u8>,
        n: usize,
        eps: f32,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .add_rmsnorm_quant_nvf4_batch
            .ok_or(GpuError::MissingOp("add_rmsnorm_quant_nvf4_batch"))?;
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let proj_guard = proj.map(|p| p.device_ptr(&self.stream));
        let pp = match &proj_guard {
            Some((p, _)) => *p as *const f32,
            None => core::ptr::null(),
        };
        let (wp, _g3) = w.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        let (qp, _g5) = q.device_ptr_mut(&self.stream);
        let (sp, _g6) = scale.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                xp as *mut _,
                pp as *const _,
                wp as *const _,
                op as *mut _,
                qp as *mut _,
                sp as *mut _,
                n as u32,
                eps,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_add_rmsnorm_quant_nvf4(&self) -> bool {
        self.kernels.add_rmsnorm_quant_nvf4_batch.is_some()
    }

    /// Slot 876: `add_rmsnorm_batch` + `quantize_nvf4` in one launch,
    /// bit-identical to the pair (`out` keeps the f32 normed row). `Ok(false)`
    /// when the pack has no slot or declines the width (nothing launched -
    /// run the pair).
    #[allow(clippy::too_many_arguments)]
    pub fn add_rmsnorm_nvf4_batch(
        &self,
        x: &mut CudaSlice<f32>,
        proj: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        scale: &mut CudaSlice<u8>,
        n: usize,
        eps: f32,
        batch: usize,
    ) -> Result<bool, GpuError> {
        let f = self.kernels.add_rmsnorm_nvf4_batch;
        self.add_rmsnorm_q_batch(f, (2, 16), x, proj, w, out, q, scale, n, eps, batch)
    }

    /// Slot 879: [`Self::add_rmsnorm_nvf4_batch`] with the scales in the
    /// tcgen05 GEMM's swizzled atom layout - `scale` holds ceil(batch / 128) *
    /// 128 rows' worth - for [`Self::nvf4_gemm_f4_sw`]. Same decline rule.
    #[allow(clippy::too_many_arguments)]
    pub fn add_rmsnorm_nvf4_batch_sw(
        &self,
        x: &mut CudaSlice<f32>,
        proj: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        scale: &mut CudaSlice<u8>,
        n: usize,
        eps: f32,
        batch: usize,
    ) -> Result<bool, GpuError> {
        if scale.len() < batch.next_multiple_of(128) * n / 16 {
            return Ok(false);
        }
        let f = self.kernels.add_rmsnorm_nvf4_batch_sw;
        self.add_rmsnorm_q_batch(f, (2, 16), x, proj, w, out, q, scale, n, eps, batch)
    }

    /// Slot 877: [`Self::add_rmsnorm_nvf4_batch`] with `quantize_e4m3`'s
    /// per-32 linear e4m3 planes (the W8 GEMMs' input); `sw` (slot 884): the
    /// scales in the tcgen05 GEMM's swizzled atom layout - `scale` holds
    /// ceil(batch / 128) * 128 rows' worth - for `f8d_gemm_t`'s `xs_sw`.
    #[allow(clippy::too_many_arguments)]
    pub fn add_rmsnorm_e4m3_batch(
        &self,
        x: &mut CudaSlice<f32>,
        proj: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        (scale, sw): (&mut CudaSlice<u8>, bool),
        n: usize,
        eps: f32,
        batch: usize,
    ) -> Result<bool, GpuError> {
        if sw && scale.len() < batch.next_multiple_of(128) * n / 32 {
            return Ok(false);
        }
        let f = if sw {
            self.kernels.add_rmsnorm_e4m3_batch_sw
        } else {
            self.kernels.add_rmsnorm_e4m3_batch
        };
        self.add_rmsnorm_q_batch(f, (1, 32), x, proj, w, out, q, scale, n, eps, batch)
    }

    /// Shared body of slots 876 / 877; `(qd, sd)`: elements per q byte / per
    /// scale byte.
    #[allow(clippy::too_many_arguments)]
    fn add_rmsnorm_q_batch(
        &self,
        f: Option<paddock_kernels::abi::AddRmsnormNvf4BatchFn>,
        (qd, sd): (usize, usize),
        x: &mut CudaSlice<f32>,
        proj: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        scale: &mut CudaSlice<u8>,
        n: usize,
        eps: f32,
        batch: usize,
    ) -> Result<bool, GpuError> {
        let Some(f) = f else {
            return Ok(false);
        };
        if q.len() < batch * n / qd || scale.len() < batch * n / sd {
            return Ok(false);
        }
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (pp, _g2) = proj.device_ptr(&self.stream);
        let (wp, _g3) = w.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        let (qp, _g5) = q.device_ptr_mut(&self.stream);
        let (sp, _g6) = scale.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slots 876 / 877); q / scale sized above, x / out as add_rmsnorm_batch
        let rc = unsafe {
            f(
                xp as *mut _,
                pp as *const _,
                wp as *const _,
                op as *mut _,
                qp as *mut _,
                sp as *mut _,
                n as u32,
                eps,
                batch as u32,
                self.stream_ptr(),
            )
        };
        const CUDA_ERROR_NOT_SUPPORTED: i32 = 801;
        if rc == CUDA_ERROR_NOT_SUPPORTED {
            return Ok(false);
        }
        check(rc).map(|()| true)
    }

    /// Fused residual-add + rmsnorm + Q8_0 quantize (the dp4a-class e4m3
    /// sibling): x += proj is written back, out keeps the f32
    /// plane (router / gemv fallbacks), q/qs get the int8 + per-32 scales.
    /// Values identical to add_rmsnorm_batch -> quantize_q8 at the same
    /// block width.
    #[allow(clippy::too_many_arguments)]
    pub fn add_rmsnorm_quant_q8_batch(
        &self,
        x: &mut CudaSlice<f32>,
        proj: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        qs: &mut CudaSlice<f32>,
        n: usize,
        eps: f32,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .add_rmsnorm_quant_q8_batch
            .ok_or(GpuError::MissingOp("add_rmsnorm_quant_q8_batch"))?;
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (pp, _g2) = proj.device_ptr(&self.stream);
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
                eps,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// SwiGLU + Q8_0 quantize in one launch: q/scales are
    /// bit-identical to swiglu -> quantize_q8; the activated plane never
    /// lands (gate stays raw - nothing reads it after the down GEMM).
    pub fn swiglu_quant_q8(
        &self,
        gate: &CudaSlice<f32>,
        up: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        qs: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .swiglu_quant_q8
            .ok_or(GpuError::MissingOp("swiglu_quant_q8"))?;
        let (gp, _g1) = gate.device_ptr(&self.stream);
        let (up_, _g2) = up.device_ptr(&self.stream);
        let (qp, _g3) = q.device_ptr_mut(&self.stream);
        let (sp, _g4) = qs.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                gp as *const _,
                up_ as *const _,
                qp as *mut _,
                sp as *mut _,
                n as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Pipelined-decode tick advance: `tokens[i] = out[i]; positions[i] += 1`
    /// for the first `rows` rows - the previous tick's sampled ids become the
    /// next tick's step-graph inputs without any host round trip. `out_off`
    /// selects the ring slot inside the pipe's out buffer.
    pub fn pipe_advance(
        &self,
        out: &CudaSlice<u32>,
        out_off: usize,
        tokens: &mut CudaSlice<u32>,
        positions: &mut CudaSlice<u32>,
        rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .pipe_advance
            .ok_or(GpuError::MissingOp("pipe_advance"))?;
        let (op, _g1) = out.device_ptr(&self.stream);
        let op = op + (out_off * std::mem::size_of::<u32>()) as u64;
        let (tp, _g2) = tokens.device_ptr_mut(&self.stream);
        let (pp, _g3) = positions.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                op as *const _,
                tp as *mut _,
                pp as *mut _,
                rows as u32,
                self.stream_ptr(),
            )
        })
    }

    /// `sample_rows` with element offsets into ring-buffered `params`/`out`
    /// planes (the pipelined decode double-buffers both so tick N+1's writes
    /// never race tick N's reads). Offsets are in u32 elements.
    pub fn sample_rows_at(
        &self,
        logits: &CudaSlice<f32>,
        params: &CudaSlice<u32>,
        par_off: usize,
        out: &mut CudaSlice<u32>,
        out_off: usize,
        rows: usize,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sample_rows
            .ok_or(GpuError::MissingOp("sample_rows"))?;
        let (lp, _g1) = logits.device_ptr(&self.stream);
        let (pp, _g2) = params.device_ptr(&self.stream);
        let pp = pp + (par_off * std::mem::size_of::<u32>()) as u64;
        let (op, _g3) = out.device_ptr_mut(&self.stream);
        let op = op + (out_off * std::mem::size_of::<u32>()) as u64;
        check(unsafe {
            f(
                lp as *const _,
                pp as *const _,
                op as *mut _,
                rows as u32,
                n as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Fused per-row sampling on decode logits: `params` = rows × 4 u32 words
    /// `{inv_t f32-bits, u f32-bits, mode, pad}`; `out[row]` gets the token id
    /// for device-sampled rows (mode 1 greedy / 2 categorical), skip rows are
    /// left untouched. Replaces the `[rows, vocab]` logits readback.
    pub fn sample_rows(
        &self,
        logits: &CudaSlice<f32>,
        params: &CudaSlice<u32>,
        out: &mut CudaSlice<u32>,
        rows: usize,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sample_rows
            .ok_or(GpuError::MissingOp("sample_rows"))?;
        let (lp, _g1) = logits.device_ptr(&self.stream);
        let (pp, _g2) = params.device_ptr(&self.stream);
        let (op, _g3) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                lp as *const _,
                pp as *const _,
                op as *mut _,
                rows as u32,
                n as u32,
                self.stream_ptr(),
            )
        })
    }

    /// True when the pack ships the P65 device top-K prefilter (slot 434).
    pub fn has_topk_rows(&self) -> bool {
        self.kernels.topk_rows.is_some()
    }

    /// Device top-K prefilter over HOST-HEAD rows (`params` rows carrying
    /// mode 4): writes `out[row*k*2 ..]` = (token id, raw-logit f32 bits)
    /// pairs, K-head in arbitrary order (the host head pipeline re-sorts).
    /// Rows with other modes are untouched. k ≤ 64.
    pub fn topk_rows(
        &self,
        logits: &CudaSlice<f32>,
        params: &CudaSlice<u32>,
        out: &mut CudaSlice<u32>,
        rows: usize,
        n: usize,
        k: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .topk_rows
            .ok_or(GpuError::MissingOp("topk_rows"))?;
        let (lp, _g1) = logits.device_ptr(&self.stream);
        let (pp, _g2) = params.device_ptr(&self.stream);
        let (op, _g3) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                lp as *const _,
                pp as *const _,
                op as *mut _,
                rows as u32,
                n as u32,
                k as u32,
                self.stream_ptr(),
            )
        })
    }

    /// True when the pack ships the P67 full-device truncation sampler
    /// (slot 435).
    pub fn has_sample_rows_t(&self) -> bool {
        self.kernels.sample_rows_t.is_some()
    }

    /// Full-device truncation sampling over mode-5 rows: `trunc` = rows ×
    /// 4 u32 `{k, top_p bits, min_p bits, pad}`; the sampled token lands in
    /// `out[row]` exactly like modes 1/2. Other modes untouched.
    pub fn sample_rows_t(
        &self,
        logits: &CudaSlice<f32>,
        params: &CudaSlice<u32>,
        trunc: &CudaSlice<u32>,
        out: &mut CudaSlice<u32>,
        rows: usize,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sample_rows_t
            .ok_or(GpuError::MissingOp("sample_rows_t"))?;
        let (lp, _g1) = logits.device_ptr(&self.stream);
        let (pp, _g2) = params.device_ptr(&self.stream);
        let (tp, _g3) = trunc.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                lp as *const _,
                pp as *const _,
                tp as *const _,
                op as *mut _,
                rows as u32,
                n as u32,
                self.stream_ptr(),
            )
        })
    }

    /// True when the pack ships the general truncation sampler (slot 436).
    pub fn has_sample_rows_p(&self) -> bool {
        self.kernels.sample_rows_p.is_some()
    }

    /// General truncation sampling over mode-6 rows (no top-k bound:
    /// top-p only / min-p only / both) - same planes as `sample_rows_t`;
    /// the sampled token lands in `out[row]` like modes 1/2/5.
    pub fn sample_rows_p(
        &self,
        logits: &CudaSlice<f32>,
        params: &CudaSlice<u32>,
        trunc: &CudaSlice<u32>,
        out: &mut CudaSlice<u32>,
        rows: usize,
        n: usize,
    ) -> Result<(), GpuError> {
        self.sample_rows_p_at(logits, params, 0, trunc, 0, out, 0, rows, n)
    }

    /// `sample_rows_p` with element offsets into ring-buffered planes
    /// (the decode pipes' double buffering). Offsets in u32 elements.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_rows_p_at(
        &self,
        logits: &CudaSlice<f32>,
        params: &CudaSlice<u32>,
        par_off: usize,
        trunc: &CudaSlice<u32>,
        tpar_off: usize,
        out: &mut CudaSlice<u32>,
        out_off: usize,
        rows: usize,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sample_rows_p
            .ok_or(GpuError::MissingOp("sample_rows_p"))?;
        let (lp, _g1) = logits.device_ptr(&self.stream);
        let (pp, _g2) = params.device_ptr(&self.stream);
        let pp = pp + (par_off * std::mem::size_of::<u32>()) as u64;
        let (tp, _g3) = trunc.device_ptr(&self.stream);
        let tp = tp + (tpar_off * std::mem::size_of::<u32>()) as u64;
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        let op = op + (out_off * std::mem::size_of::<u32>()) as u64;
        check(unsafe {
            f(
                lp as *const _,
                pp as *const _,
                tp as *const _,
                op as *mut _,
                rows as u32,
                n as u32,
                self.stream_ptr(),
            )
        })
    }

    /// `sample_rows_t` with element offsets into ring-buffered params/
    /// trunc-params/out planes (the P67b pipe double-buffers all three).
    #[allow(clippy::too_many_arguments)]
    pub fn sample_rows_t_at(
        &self,
        logits: &CudaSlice<f32>,
        params: &CudaSlice<u32>,
        par_off: usize,
        trunc: &CudaSlice<u32>,
        trunc_off: usize,
        out: &mut CudaSlice<u32>,
        out_off: usize,
        rows: usize,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sample_rows_t
            .ok_or(GpuError::MissingOp("sample_rows_t"))?;
        let (lp, _g1) = logits.device_ptr(&self.stream);
        let (pp, _g2) = params.device_ptr(&self.stream);
        let pp = pp + (par_off * std::mem::size_of::<u32>()) as u64;
        let (tp, _g3) = trunc.device_ptr(&self.stream);
        let tp = tp + (trunc_off * std::mem::size_of::<u32>()) as u64;
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        let op = op + (out_off * std::mem::size_of::<u32>()) as u64;
        check(unsafe {
            f(
                lp as *const _,
                pp as *const _,
                tp as *const _,
                op as *mut _,
                rows as u32,
                n as u32,
                self.stream_ptr(),
            )
        })
    }

    /// True when the pack ships both canonical rejection-sampling kernels
    /// (sampled draft chain + full-q verify resolve).
    pub fn has_spec_rs(&self) -> bool {
        self.kernels.draft_rs.is_some() && self.kernels.spec_rs_resolve.is_some()
    }

    /// RS chain-step draft draw: sample from the drafter softmax at the
    /// per-row temperature, materializing q (fp16 exp mass + exact f32 sum)
    /// into the step-indexed q-store. `invt[row] <= 0` = greedy argmax.
    /// Graph-capturable: the step index is a device counter.
    #[allow(clippy::too_many_arguments)]
    pub fn draft_rs(
        &self,
        logits: &CudaSlice<f32>,
        invt: &CudaSlice<f32>,
        uplane: &CudaSlice<f32>,
        step: &CudaSlice<u32>,
        qstore: &mut CudaSlice<u16>,
        qsum: &mut CudaSlice<f32>,
        tok: &mut CudaSlice<u32>,
        rows: usize,
        n: usize,
        rmax: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .draft_rs
            .ok_or(GpuError::MissingOp("draft_rs"))?;
        let (lp, _g1) = logits.device_ptr(&self.stream);
        let (ip, _g2) = invt.device_ptr(&self.stream);
        let (up, _g3) = uplane.device_ptr(&self.stream);
        let (sp, _g4) = step.device_ptr(&self.stream);
        let (qp, _g5) = qstore.device_ptr_mut(&self.stream);
        let (qs, _g6) = qsum.device_ptr_mut(&self.stream);
        let (tp, _g7) = tok.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                lp as *const _,
                ip as *const _,
                up as *const _,
                sp as *const _,
                qp as *mut _,
                qs as *mut _,
                tp as *mut _,
                rows as u32,
                n as u32,
                rmax as u32,
                self.stream_ptr(),
            )
        })
    }

    /// RS verify resolve: accept-or-recover per drafted verify row against
    /// the tick's softcapped logits; writes the resolved token into the
    /// sampled-ids plane so the accept-while-match walk consumes RS rounds
    /// unchanged. par = 8 u32 words per row (see the ABI doc).
    #[allow(clippy::too_many_arguments)]
    pub fn spec_rs_resolve(
        &self,
        logits: &CudaSlice<f32>,
        drafts: &CudaSlice<u32>,
        qstore: &CudaSlice<u16>,
        qsum: &CudaSlice<f32>,
        par: &CudaSlice<u32>,
        out: &mut CudaSlice<u32>,
        nrs: usize,
        rr: usize,
        n: usize,
        rmax: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .spec_rs_resolve
            .ok_or(GpuError::MissingOp("spec_rs_resolve"))?;
        let (lp, _g1) = logits.device_ptr(&self.stream);
        let (dp, _g2) = drafts.device_ptr(&self.stream);
        let (qp, _g3) = qstore.device_ptr(&self.stream);
        let (qs, _g4) = qsum.device_ptr(&self.stream);
        let (pp, _g5) = par.device_ptr(&self.stream);
        let (op, _g6) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                lp as *const _,
                dp as *const _,
                qp as *const _,
                qs as *const _,
                pp as *const _,
                op as *mut _,
                nrs as u32,
                rr as u32,
                n as u32,
                rmax as u32,
                self.stream_ptr(),
            )
        })
    }

    /// DFlash2 K-candidate rejection-sampling verify resolve (rung G, slot
    /// 471): rows planned mode 7 in `par` (`{inv_t, u1, 7, u2 bits}` + the
    /// mode-5 trunc plane) get the draft at row j+1 (`toks[row+1]`) accepted
    /// with probability min(1, p/q) against the mode-5 nucleus, or the
    /// residual pick; the token lands in `out` like any sampled row. `meta`
    /// is the spec_toks meta plane (block -> chain row at `[n_blocks + i]`),
    /// `drows` the drafter's rows per block, `k` the selector width.
    #[allow(clippy::too_many_arguments)]
    pub fn dflash_rs_resolve(
        &self,
        logits: &CudaSlice<f32>,
        par: &CudaSlice<u32>,
        tpar: &CudaSlice<u32>,
        meta: &CudaSlice<u32>,
        toks: &CudaSlice<u32>,
        cand: &CudaSlice<u32>,
        q16: &CudaSlice<f32>,
        out: &mut CudaSlice<u32>,
        rows: usize,
        n_blocks: usize,
        k1: usize,
        drows: usize,
        k: usize,
        vocab: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .dflash_rs_resolve
            .ok_or(GpuError::MissingOp("dflash_rs_resolve"))?;
        let (lp, _g1) = logits.device_ptr(&self.stream);
        let (pp, _g2) = par.device_ptr(&self.stream);
        let (tp, _g3) = tpar.device_ptr(&self.stream);
        let (mp, _g4) = meta.device_ptr(&self.stream);
        let (kp, _g5) = toks.device_ptr(&self.stream);
        let (cp, _g6) = cand.device_ptr(&self.stream);
        let (qp, _g7) = q16.device_ptr(&self.stream);
        let (op, _g8) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                lp as *const _,
                pp as *const _,
                tp as *const _,
                mp as *const _,
                kp as *const _,
                cp as *const _,
                qp as *const _,
                op as *mut _,
                rows as u32,
                n_blocks as u32,
                k1 as u32,
                drows as u32,
                k as u32,
                vocab as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Batched-spec conv-ext staging: `ext[b]` = window(slots[b]) ++ mixed[b]
    /// (`[batch, km1 + r, conv_dim]`). One launch replaces 2·B copy_regions.
    #[allow(clippy::too_many_arguments)]
    pub fn conv_ext_build_slots(
        &self,
        wins: &CudaSlice<f32>,
        slots: &CudaSlice<u32>,
        mixed: &CudaSlice<f32>,
        ext: &mut CudaSlice<f32>,
        batch: usize,
        km1: usize,
        r: usize,
        conv_dim: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .conv_ext_build_slots
            .ok_or(GpuError::MissingOp("conv_ext_build_slots"))?;
        let (wp, _g1) = wins.device_ptr(&self.stream);
        let (sp, _g2) = slots.device_ptr(&self.stream);
        let (mp, _g3) = mixed.device_ptr(&self.stream);
        let (ep, _g4) = ext.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                wp as *const _,
                sp as *const _,
                mp as *const _,
                ep as *mut _,
                batch as u32,
                km1 as u32,
                r as u32,
                conv_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Causal conv+SiLU over per-slot extended segments, emitting only the r
    /// real rows per slot (`out` [batch, r, conv_dim]). k must equal km1+1.
    #[allow(clippy::too_many_arguments)]
    pub fn conv_chunk_ext(
        &self,
        ext: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        batch: usize,
        km1: usize,
        r: usize,
        conv_dim: usize,
        k: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .conv_chunk_ext
            .ok_or(GpuError::MissingOp("conv_chunk_ext"))?;
        let (ep, _g1) = ext.device_ptr(&self.stream);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let (op, _g3) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                ep as *const _,
                wp as *const _,
                op as *mut _,
                batch as u32,
                km1 as u32,
                r as u32,
                conv_dim as u32,
                k as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Ragged spec commit, state half: roll each short slot (committed[b] < r)
    /// back to snapshot committed[b]-1. Snapshot layout = the v2 kernel's
    /// [batch, r] t-major transposed tiles.
    #[allow(clippy::too_many_arguments)]
    pub fn state_restore_slots(
        &self,
        states: &mut CudaSlice<f32>,
        snap: &CudaSlice<f32>,
        slots: &CudaSlice<u32>,
        committed: &CudaSlice<u32>,
        batch: usize,
        r: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .state_restore_slots
            .ok_or(GpuError::MissingOp("state_restore_slots"))?;
        let (stp, _g1) = states.device_ptr_mut(&self.stream);
        let (snp, _g2) = snap.device_ptr(&self.stream);
        let (slp, _g3) = slots.device_ptr(&self.stream);
        let (cp, _g4) = committed.device_ptr(&self.stream);
        check(unsafe {
            f(
                stp as *mut _,
                snp as *const _,
                slp as *const _,
                cp as *const _,
                batch as u32,
                r as u32,
                n_heads as u32,
                head_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Ragged spec commit, conv half: window(slots[b]) = ext[b] rows
    /// [committed[b], committed[b]+km1).
    #[allow(clippy::too_many_arguments)]
    pub fn conv_commit_slots(
        &self,
        ext: &CudaSlice<f32>,
        wins: &mut CudaSlice<f32>,
        slots: &CudaSlice<u32>,
        committed: &CudaSlice<u32>,
        batch: usize,
        km1: usize,
        r: usize,
        conv_dim: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .conv_commit_slots
            .ok_or(GpuError::MissingOp("conv_commit_slots"))?;
        let (ep, _g1) = ext.device_ptr(&self.stream);
        let (wp, _g2) = wins.device_ptr_mut(&self.stream);
        let (slp, _g3) = slots.device_ptr(&self.stream);
        let (cp, _g4) = committed.device_ptr(&self.stream);
        check(unsafe {
            f(
                ep as *const _,
                wp as *mut _,
                slp as *const _,
                cp as *const _,
                batch as u32,
                km1 as u32,
                r as u32,
                conv_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Advance staged per-row positions on device: pos[0..r] += 1 and
    /// mrope[0..4r] += 1 (graph-capturable draft-step advance).
    pub fn bump_rows_u32(
        &self,
        pos: &mut CudaSlice<u32>,
        mrope: &mut CudaSlice<u32>,
        r: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .bump_rows_u32
            .ok_or(GpuError::MissingOp("bump_rows_u32"))?;
        let (pp, _g1) = pos.device_ptr_mut(&self.stream);
        let (mp, _g2) = mrope.device_ptr_mut(&self.stream);
        check(unsafe { f(pp as *mut _, mp as *mut _, r as u32, self.stream_ptr()) })
    }
}
