//! Mamba-2 (SSD) family ops - nemotron_h_moe's linear-recurrence lane
//! Structural sibling of deltanet.rs; the kernels differ (see
//! packs/cuda/src/mamba/core.cuh for the semantics, cross-checked against
//! the vLLM and llama.cpp references).
//!
//! Offset arguments follow the `f8_gemv_at` idiom: several inputs live as
//! row slices inside the fused in_proj output ([z | x B C | dt] rows), so
//! wrappers take an element offset + row stride rather than forcing copies.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::f16;

use super::error::*;
use super::*;

impl GpuExecutor {
    pub fn has_mamba2(&self) -> bool {
        self.kernels.mamba_conv_step.is_some()
            && self.kernels.mamba2_scan_seq.is_some()
            && self.kernels.mamba_rmsnorm_gated_g.is_some()
            && self.kernels.f8r_gemv.is_some()
    }

    /// True when the batched decode-step pair is loadable (
    /// stage A): the continuous-batching tick's per-slot mamba state advance.
    pub fn has_mamba2_batch(&self) -> bool {
        self.kernels.mamba_conv_step_batch.is_some()
            && self.kernels.mamba2_scan_step_batch.is_some()
    }

    /// Batched single-token conv step over a slot arena of windows
    /// (`win` [n_slots, k-1, conv_dim]): row r of `x` (fused rows of stride
    /// `x_stride`, conv span at `x_off`) advances slot `slots[r]` and writes
    /// `out[r]`. Bit-exact per row vs [`Self::mamba_conv_step`].
    #[allow(clippy::too_many_arguments)]
    pub fn mamba_conv_step_batch(
        &self,
        win: &mut CudaSlice<f32>,
        x: &CudaSlice<f32>,
        x_off: usize,
        x_stride: usize,
        slots: &CudaSlice<u32>,
        w: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        conv_dim: usize,
        k: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .mamba_conv_step_batch
            .ok_or(GpuError::MissingOp("mamba_conv_step_batch"))?;
        debug_assert!(x.len() >= (batch - 1) * x_stride + x_off + conv_dim);
        debug_assert!(slots.len() >= batch);
        debug_assert!(out.len() >= batch * conv_dim);
        let (wp, _g1) = win.device_ptr_mut(&self.stream);
        let (xp, _g2) = x.device_ptr(&self.stream);
        let (sp, _g3) = slots.device_ptr(&self.stream);
        let (cp, _g4) = w.device_ptr(&self.stream);
        let (bp, _g5) = b.device_ptr(&self.stream);
        let (op, _g6) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; geometry checked above.
        check(unsafe {
            f(
                wp as *mut _,
                xp as *const _,
                x_off as u32,
                x_stride as u32,
                sp as *const _,
                cp as *const _,
                bp as *const _,
                op as *mut _,
                conv_dim as u32,
                k as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Batched single-token SSD scan step over a slot arena of states
    /// (`state` [n_slots, n_heads, head_dim, d_state]): row r of `xbc`
    /// (conv output rows [batch, conv_dim]) advances slot `slots[r]`.
    /// `dt_raw`/`dt_off`/`dt_stride` follow [`Self::mamba2_scan_seq`]'s
    /// convention (dt lanes inside fused rows). Bit-exact per row vs the
    /// seq scan at `n_tokens = 1`.
    #[allow(clippy::too_many_arguments)]
    pub fn mamba2_scan_step_batch(
        &self,
        state: &mut CudaSlice<f32>,
        xbc: &CudaSlice<f32>,
        dt: &CudaSlice<f32>,
        dt_off: usize,
        dt_stride: usize,
        slots: &CudaSlice<u32>,
        a: &CudaSlice<f32>,
        d: &CudaSlice<f32>,
        dt_bias: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        batch: usize,
        n_heads: usize,
        head_dim: usize,
        d_state: usize,
        n_groups: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .mamba2_scan_step_batch
            .ok_or(GpuError::MissingOp("mamba2_scan_step_batch"))?;
        debug_assert!(dt.len() >= dt_off + (batch - 1) * dt_stride + n_heads);
        debug_assert!(slots.len() >= batch);
        debug_assert!(y.len() >= batch * n_heads * head_dim);
        let (sp, _g1) = state.device_ptr_mut(&self.stream);
        let (xp, _g2) = xbc.device_ptr(&self.stream);
        let (tp, _g3) = dt.device_ptr(&self.stream);
        let (lp, _g4) = slots.device_ptr(&self.stream);
        let (ap, _g5) = a.device_ptr(&self.stream);
        let (dp, _g6) = d.device_ptr(&self.stream);
        let (bp, _g7) = dt_bias.device_ptr(&self.stream);
        let (yp, _g8) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; geometry validated by the launcher (d_state
        // template pin, group divisibility) and the asserts above.
        check(unsafe {
            f(
                sp as *mut _,
                xp as *const _,
                (tp + (dt_off * 4) as u64) as *const _,
                dt_stride as u32,
                lp as *const _,
                ap as *const _,
                dp as *const _,
                bp as *const _,
                yp as *mut _,
                batch as u32,
                n_heads as u32,
                head_dim as u32,
                d_state as u32,
                n_groups as u32,
                self.stream_ptr(),
            )
        })
    }

    /// The f16 SSM-state class (slot 445): batched decode step over a half-
    /// width state arena. Same geometry contract as `mamba2_scan_step_batch`;
    /// `state` holds `n_slots * n_heads * head_dim * d_state` f16 elements.
    /// State is stored f16 and computed f32 - the same class as vLLM's
    /// `--mamba-ssm-cache-dtype float16`, so electing it is class parity
    /// rather than a lighter class. head_dim is pinned to 64 by the half2
    /// pairing.
    #[allow(clippy::too_many_arguments)]
    pub fn mamba2_scan_step_batch_f16(
        &self,
        state: &mut CudaSlice<f16>,
        xbc: &CudaSlice<f32>,
        dt: &CudaSlice<f32>,
        dt_off: usize,
        dt_stride: usize,
        slots: &CudaSlice<u32>,
        a: &CudaSlice<f32>,
        d: &CudaSlice<f32>,
        dt_bias: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        batch: usize,
        n_heads: usize,
        head_dim: usize,
        d_state: usize,
        n_groups: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .mamba2_scan_step_batch_f16
            .ok_or(GpuError::MissingOp("mamba2_scan_step_batch_f16"))?;
        debug_assert!(dt.len() >= dt_off + (batch - 1) * dt_stride + n_heads);
        debug_assert!(slots.len() >= batch);
        debug_assert!(y.len() >= batch * n_heads * head_dim);
        let (sp, _g1) = state.device_ptr_mut(&self.stream);
        let (xp, _g2) = xbc.device_ptr(&self.stream);
        let (tp, _g3) = dt.device_ptr(&self.stream);
        let (lp, _g4) = slots.device_ptr(&self.stream);
        let (ap, _g5) = a.device_ptr(&self.stream);
        let (dp, _g6) = d.device_ptr(&self.stream);
        let (bp, _g7) = dt_bias.device_ptr(&self.stream);
        let (yp, _g8) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; geometry validated by the launcher.
        check(unsafe {
            f(
                sp as *mut _,
                xp as *const _,
                (tp + (dt_off * 4) as u64) as *const _,
                dt_stride as u32,
                lp as *const _,
                ap as *const _,
                dp as *const _,
                bp as *const _,
                yp as *mut _,
                batch as u32,
                n_heads as u32,
                head_dim as u32,
                d_state as u32,
                n_groups as u32,
                self.stream_ptr(),
            )
        })
    }

    /// The f16 SSM-state class (slot 443): seq walk at an arena offset. The
    /// walk keeps state register-resident for the whole span, so its
    /// per-token arithmetic is bit-identical to the f32 twin - only the
    /// hand-off between launches rounds.
    #[allow(clippy::too_many_arguments)]
    pub fn mamba2_scan_seq_at_f16(
        &self,
        state: &mut CudaSlice<f16>,
        state_off: usize,
        xbc: &CudaSlice<f32>,
        xbc_off: usize,
        dt: &CudaSlice<f32>,
        dt_off: usize,
        dt_stride: usize,
        a: &CudaSlice<f32>,
        d: &CudaSlice<f32>,
        dt_bias: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        y_off: usize,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
        d_state: usize,
        n_groups: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .mamba2_scan_seq_f16
            .ok_or(GpuError::MissingOp("mamba2_scan_seq_f16"))?;
        debug_assert!(state.len() >= state_off + n_heads * head_dim * d_state);
        debug_assert!(dt.len() >= dt_off + (n_tokens - 1) * dt_stride + n_heads);
        debug_assert!(y.len() >= y_off + n_tokens * n_heads * head_dim);
        let (sp, _g1) = state.device_ptr_mut(&self.stream);
        let (xp, _g2) = xbc.device_ptr(&self.stream);
        let (tp, _g3) = dt.device_ptr(&self.stream);
        let (ap, _g4) = a.device_ptr(&self.stream);
        let (dp, _g5) = d.device_ptr(&self.stream);
        let (bp, _g6) = dt_bias.device_ptr(&self.stream);
        let (yp, _g7) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; offsets are element counts, f16 = 2 bytes.
        check(unsafe {
            f(
                (sp + (state_off * 2) as u64) as *mut _,
                (xp + (xbc_off * 4) as u64) as *const _,
                (tp + (dt_off * 4) as u64) as *const _,
                dt_stride as u32,
                ap as *const _,
                dp as *const _,
                bp as *const _,
                (yp + (y_off * 4) as u64) as *mut _,
                n_tokens as u32,
                n_heads as u32,
                head_dim as u32,
                d_state as u32,
                n_groups as u32,
                self.stream_ptr(),
            )
        })
    }

    /// The f16 SSM-state class (slot 444): seq walk + per-row snapshots.
    /// `snap` is f16 too - a partial spec accept rolls back by flat-copying
    /// a snap row over the live state, so the two must share a
    /// representation or the rollback would re-round.
    #[allow(clippy::too_many_arguments)]
    pub fn mamba2_scan_seq_snap_at_f16(
        &self,
        state: &mut CudaSlice<f16>,
        state_off: usize,
        xbc: &CudaSlice<f32>,
        xbc_off: usize,
        dt: &CudaSlice<f32>,
        dt_off: usize,
        dt_stride: usize,
        a: &CudaSlice<f32>,
        d: &CudaSlice<f32>,
        dt_bias: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        y_off: usize,
        snap: &mut CudaSlice<f16>,
        snap_off: usize,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
        d_state: usize,
        n_groups: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .mamba2_scan_seq_snap_f16
            .ok_or(GpuError::MissingOp("mamba2_scan_seq_snap_f16"))?;
        debug_assert!(state.len() >= state_off + n_heads * head_dim * d_state);
        debug_assert!(snap.len() >= snap_off + n_tokens * n_heads * head_dim * d_state);
        let (sp, _g1) = state.device_ptr_mut(&self.stream);
        let (xp, _g2) = xbc.device_ptr(&self.stream);
        let (tp, _g3) = dt.device_ptr(&self.stream);
        let (ap, _g4) = a.device_ptr(&self.stream);
        let (dp, _g5) = d.device_ptr(&self.stream);
        let (bp, _g6) = dt_bias.device_ptr(&self.stream);
        let (yp, _g7) = y.device_ptr_mut(&self.stream);
        let (np, _g8) = snap.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; offsets are element counts, f16 = 2 bytes.
        check(unsafe {
            f(
                (sp + (state_off * 2) as u64) as *mut _,
                (xp + (xbc_off * 4) as u64) as *const _,
                (tp + (dt_off * 4) as u64) as *const _,
                dt_stride as u32,
                ap as *const _,
                dp as *const _,
                bp as *const _,
                (yp + (y_off * 4) as u64) as *mut _,
                (np + (snap_off * 2) as u64) as *mut _,
                n_tokens as u32,
                n_heads as u32,
                head_dim as u32,
                d_state as u32,
                n_groups as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Widen an f16 SSM state region into an f32 checkpoint blob (slot 446).
    /// Exact. `n` is an element count.
    pub fn ssm_state_widen(
        &self,
        src: &CudaSlice<f16>,
        src_off: usize,
        dst: &mut CudaSlice<f32>,
        dst_off: usize,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .ssm_state_widen
            .ok_or(GpuError::MissingOp("ssm_state_widen"))?;
        debug_assert!(src.len() >= src_off + n && dst.len() >= dst_off + n);
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (dp, _g2) = dst.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                (sp + (src_off * 2) as u64) as *const _,
                (dp + (dst_off * 4) as u64) as *mut _,
                n as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Narrow an f32 checkpoint region back to f16 (slot 447). Round-trips
    /// bit-for-bit with `ssm_state_widen` because the blob only ever holds
    /// values that originated as f16.
    pub fn ssm_state_narrow(
        &self,
        src: &CudaSlice<f32>,
        src_off: usize,
        dst: &mut CudaSlice<f16>,
        dst_off: usize,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .ssm_state_narrow
            .ok_or(GpuError::MissingOp("ssm_state_narrow"))?;
        debug_assert!(src.len() >= src_off + n && dst.len() >= dst_off + n);
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (dp, _g2) = dst.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                (sp + (src_off * 4) as u64) as *const _,
                (dp + (dst_off * 2) as u64) as *mut _,
                n as u32,
                self.stream_ptr(),
            )
        })
    }

    /// True when the pack carries the f16 SSM-state class (slots 443-445).
    pub fn has_mamba2_f16_state(&self) -> bool {
        self.kernels.mamba2_scan_seq_f16.is_some()
            && self.kernels.mamba2_scan_seq_snap_f16.is_some()
            && self.kernels.mamba2_scan_step_batch_f16.is_some()
    }

    /// True when the pack carries the snapshot-free verify walk and its
    /// replay (slots 746-747) - the f16 state class's rollback by replay.
    pub fn has_mamba2_rescan(&self) -> bool {
        self.kernels.mamba2_scan_seq_keep_f16.is_some() && self.kernels.mamba2_rescan_f16.is_some()
    }

    /// Slot 746: the verify walk without per-row snapshots. `state` is read
    /// only (it stays the pre-round state); y as
    /// [`Self::mamba2_scan_seq_snap_at_f16`]'s; each row's x | B | raw dt
    /// lands in `keep` at `keep_off` (rows of [`mamba2_keep_row`] f32) for
    /// [`Self::mamba2_rescan_upload`].
    #[allow(clippy::too_many_arguments)]
    pub fn mamba2_scan_seq_keep_at_f16(
        &self,
        state: &CudaSlice<f16>,
        state_off: usize,
        xbc: &CudaSlice<f32>,
        xbc_off: usize,
        dt: &CudaSlice<f32>,
        dt_off: usize,
        dt_stride: usize,
        a: &CudaSlice<f32>,
        d: &CudaSlice<f32>,
        dt_bias: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        y_off: usize,
        keep: &mut CudaSlice<f32>,
        keep_off: usize,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
        d_state: usize,
        n_groups: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .mamba2_scan_seq_keep_f16
            .ok_or(GpuError::MissingOp("mamba2_scan_seq_keep_f16"))?;
        debug_assert!(state.len() >= state_off + n_heads * head_dim * d_state);
        debug_assert!(
            keep.len()
                >= keep_off + n_tokens * mamba2_keep_row(n_heads, head_dim, d_state, n_groups)
        );
        let (sp, _g1) = state.device_ptr(&self.stream);
        let (xp, _g2) = xbc.device_ptr(&self.stream);
        let (tp, _g3) = dt.device_ptr(&self.stream);
        let (ap, _g4) = a.device_ptr(&self.stream);
        let (dp, _g5) = d.device_ptr(&self.stream);
        let (bp, _g6) = dt_bias.device_ptr(&self.stream);
        let (yp, _g7) = y.device_ptr_mut(&self.stream);
        let (kp, _g8) = keep.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; offsets are element counts (f16 2 bytes, f32 4).
        check(unsafe {
            f(
                (sp + (state_off * 2) as u64) as *const _,
                (xp + (xbc_off * 4) as u64) as *const _,
                (tp + (dt_off * 4) as u64) as *const _,
                dt_stride as u32,
                ap as *const _,
                dp as *const _,
                bp as *const _,
                (yp + (y_off * 4) as u64) as *mut _,
                (kp + (keep_off * 4) as u64) as *mut _,
                n_tokens as u32,
                n_heads as u32,
                head_dim as u32,
                d_state as u32,
                n_groups as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 747: upload `descs` (six words a replay: src f16 state, dst f16
    /// state, keep rows, A, dt_bias, rows - device addresses) into `scratch`
    /// and replay them all in one launch. Stream-ordered, so `scratch` may be
    /// reused by the next call.
    pub fn mamba2_rescan_upload(
        &self,
        scratch: &mut CudaSlice<u64>,
        descs: &[u64],
        n_heads: usize,
        head_dim: usize,
        d_state: usize,
        n_groups: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .mamba2_rescan_f16
            .ok_or(GpuError::MissingOp("mamba2_rescan_f16"))?;
        let n = descs.len();
        if n == 0 {
            return Ok(());
        }
        if n > scratch.len() || !n.is_multiple_of(6) {
            return Err(GpuError::Driver(format!(
                "rescan needs {n} descriptor words, scratch holds {}",
                scratch.len()
            )));
        }
        {
            let mut v = scratch.slice_mut(0..n);
            self.stream
                .memcpy_htod(descs, &mut v)
                .map_err(|e| GpuError::Driver(e.to_string()))?;
        }
        let (dp, _g) = scratch.device_ptr(&self.stream);
        // SAFETY: ABI contract; the descriptors' pointers are live by the caller.
        check(unsafe {
            f(
                dp as *const _,
                (n / 6) as u32,
                n_heads as u32,
                head_dim as u32,
                d_state as u32,
                n_groups as u32,
                self.stream_ptr(),
            )
        })
    }

    /// True when the nemotron bulk-prefill kernel set is loadable: the span
    /// conv plus the batched consumers the chunked path rides (W8A8 f8row
    /// GEMM + activation quantizer, the tiled scalar prefill attention -
    /// the hd128 arm - batch embed gather, tiled f32 GEMM, batched router
    /// top-k).
    /// Bulk-prefill kernels every nemotron weight class needs, whatever the
    /// checkpoint. Deliberately does not include the fp8 pair - see
    /// [`Self::has_nemotron_prefill_f8`].
    pub fn has_nemotron_prefill_core(&self) -> bool {
        self.kernels.mamba_conv_seq.is_some()
            && self.kernels.attn_prefill.is_some()
            && self.kernels.embed_gather_batch.is_some()
            && self.kernels.gemm_f32.is_some()
            && self.kernels.moe_topk_sigmoid_batch.is_some()
    }

    /// The NVFP4/fp8 checkpoint's prefill: core plus the W8A8 row GEMM and the
    /// per-token e4m3 activation quantizer that feeds it.
    ///
    /// These two are consumed only by the `LinW::F8` arms of nemotron's mamba
    /// in_proj/out_proj (forward.rs and batch.rs, four sites, all four inside
    /// that match arm). The GGUF lane takes `LinW::Qw` onto the int8 mmq
    /// ladder and never calls either one - which is why the GGUF gate below
    /// does not ask for them.
    pub fn has_nemotron_prefill_f8(&self) -> bool {
        self.has_nemotron_prefill_core()
            && self.kernels.f8row_gemm.is_some()
            && self.kernels.quantize_e4m3_row.is_some()
    }

    /// The GGUF (Q8_0/k-quant) checkpoint's prefill. Core only: the GEMM half
    /// rides `prefill_quant` + `prefill_mm_pre_any`, i.e. the int8 mmq ladder
    /// that granite/laguna/qwen35 already prefill through on every arch we
    /// build for.
    ///
    /// This used to be one bundle that demanded the fp8 pair
    /// from both lanes. `PD_F8W8_OK = __CUDA_ARCH__ >= 890` nulls that family
    /// below Ada, so on sm_86 the bundle went false and took two things with
    /// it: the batch lane (nemotron served every width on the serial loop, and
    /// the drafter with it, since spec only runs in run_batched) AND the serial
    /// lane's own bulk chunked prefill, which then fell back to one full
    /// forward per prompt token. Measured cost on an A6000: 1028 ms to first
    /// token on a 128-token prompt, and roughly half the decode rate at
    /// c1/c4/c8. Nothing about the weights needed fp8 - the gate simply
    /// asked the wrong lane's question.
    pub fn has_nemotron_prefill_gguf(&self) -> bool {
        self.has_nemotron_prefill_core()
    }

    /// Names the absent members, so a refusal can say which kernel is missing
    /// instead of asserting a whole set is. `f8` selects the lane.
    pub fn nemotron_prefill_missing(&self, f8: bool) -> Vec<&'static str> {
        let mut m = Vec::new();
        if self.kernels.mamba_conv_seq.is_none() {
            m.push("mamba_conv_seq");
        }
        if self.kernels.attn_prefill.is_none() {
            m.push("attn_prefill");
        }
        if self.kernels.embed_gather_batch.is_none() {
            m.push("embed_gather_batch");
        }
        if self.kernels.gemm_f32.is_none() {
            m.push("gemm_f32");
        }
        if self.kernels.moe_topk_sigmoid_batch.is_none() {
            m.push("moe_topk_sigmoid_batch");
        }
        if f8 {
            if self.kernels.f8row_gemm.is_none() {
                m.push("f8row_gemm");
            }
            if self.kernels.quantize_e4m3_row.is_none() {
                m.push("quantize_e4m3_row");
            }
        }
        m
    }

    /// Bulk causal conv over `n_tokens` with the same persistent window as
    /// [`Self::mamba_conv_step`] (bit-exact vs stepping serially). Reads the
    /// conv span from fused rows of stride `x_stride` at offset `x_off`;
    /// writes `out` [n_tokens, conv_dim].
    #[allow(clippy::too_many_arguments)]
    pub fn mamba_conv_seq(
        &self,
        win: &mut CudaSlice<f32>,
        xbc: &CudaSlice<f32>,
        x_off: usize,
        x_stride: usize,
        w: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        conv_dim: usize,
        k: usize,
        n_tokens: usize,
    ) -> Result<(), GpuError> {
        self.mamba_conv_seq_at(
            win, 0, xbc, x_off, x_stride, w, b, out, 0, conv_dim, k, n_tokens,
        )
    }

    /// [`Self::mamba_conv_seq`] with element offsets into the window and
    /// output buffers - the batch lane's per-slot form (stage C): the
    /// window lives at `win_off` inside a slot ARENA, and the run's rows sit
    /// at `out_off` inside the shared chunk plane. A run's row base folds
    /// into `x_off` (rows stride `x_stride`), so x needs no extra offset.
    #[allow(clippy::too_many_arguments)]
    pub fn mamba_conv_seq_at(
        &self,
        win: &mut CudaSlice<f32>,
        win_off: usize,
        xbc: &CudaSlice<f32>,
        x_off: usize,
        x_stride: usize,
        w: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        out_off: usize,
        conv_dim: usize,
        k: usize,
        n_tokens: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .mamba_conv_seq
            .ok_or(GpuError::MissingOp("mamba_conv_seq"))?;
        debug_assert!(xbc.len() >= (n_tokens - 1) * x_stride + x_off + conv_dim);
        debug_assert!(win.len() >= win_off + (k - 1) * conv_dim);
        debug_assert!(out.len() >= out_off + n_tokens * conv_dim);
        let (wp, _g1) = win.device_ptr_mut(&self.stream);
        let (xp, _g2) = xbc.device_ptr(&self.stream);
        let (cp, _g3) = w.device_ptr(&self.stream);
        let (bp, _g4) = b.device_ptr(&self.stream);
        let (op, _g5) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; window/x/out sizes checked above.
        check(unsafe {
            f(
                (wp + (win_off * 4) as u64) as *mut _,
                xp as *const _,
                x_off as u32,
                x_stride as u32,
                cp as *const _,
                bp as *const _,
                (op + (out_off * 4) as u64) as *mut _,
                conv_dim as u32,
                k as u32,
                n_tokens as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Single-token causal conv1d + bias + SiLU with a persistent window
    /// (`win` [k-1, conv_dim], oldest-first pre-conv rows - advanced in
    /// place). `x_off` selects the conv span inside a fused row (the
    /// in_proj output's `[z | x B C | dt]` layout puts it at z_len).
    #[allow(clippy::too_many_arguments)]
    pub fn mamba_conv_step(
        &self,
        win: &mut CudaSlice<f32>,
        x_new: &CudaSlice<f32>,
        x_off: usize,
        w: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        conv_dim: usize,
        k: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .mamba_conv_step
            .ok_or(GpuError::MissingOp("mamba_conv_step"))?;
        debug_assert!(x_new.len() >= x_off + conv_dim);
        let (wp, _g1) = win.device_ptr_mut(&self.stream);
        let (xp, _g2) = x_new.device_ptr(&self.stream);
        let (cp, _g3) = w.device_ptr(&self.stream);
        let (bp, _g4) = b.device_ptr(&self.stream);
        let (op, _g5) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; window/x sizes checked above.
        check(unsafe {
            f(
                wp as *mut _,
                (xp + (x_off * 4) as u64) as *const _,
                cp as *const _,
                bp as *const _,
                op as *mut _,
                conv_dim as u32,
                k as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Sequential SSD scan over `n_tokens`: `state` [n_heads, head_dim,
    /// d_state] f32 (read-modify-write), `xbc` the conv OUTPUT rows
    /// [n_tokens, conv_dim] = [x | B | C], `dt` the raw dt lanes riding
    /// inside rows of stride `dt_stride` at element offset `dt_off`
    /// (softplus + bias happen in-kernel), `y` [n_tokens, d_inner] with the
    /// D-skip already applied.
    #[allow(clippy::too_many_arguments)]
    pub fn mamba2_scan_seq(
        &self,
        state: &mut CudaSlice<f32>,
        xbc: &CudaSlice<f32>,
        dt: &CudaSlice<f32>,
        dt_off: usize,
        dt_stride: usize,
        a: &CudaSlice<f32>,
        d: &CudaSlice<f32>,
        dt_bias: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
        d_state: usize,
        n_groups: usize,
    ) -> Result<(), GpuError> {
        self.mamba2_scan_seq_at(
            state, 0, xbc, 0, dt, dt_off, dt_stride, a, d, dt_bias, y, 0, n_tokens, n_heads,
            head_dim, d_state, n_groups,
        )
    }

    /// [`Self::mamba2_scan_seq`] with element offsets into the state, conv
    /// output and y buffers - the batch lane's per-slot form (stage
    /// C): the state lives at `state_off` inside a slot ARENA, and the run's
    /// rows sit at `xbc_off`/`y_off` inside the shared chunk planes. The
    /// run's dt row base folds into `dt_off` (rows stride `dt_stride`).
    #[allow(clippy::too_many_arguments)]
    pub fn mamba2_scan_seq_at(
        &self,
        state: &mut CudaSlice<f32>,
        state_off: usize,
        xbc: &CudaSlice<f32>,
        xbc_off: usize,
        dt: &CudaSlice<f32>,
        dt_off: usize,
        dt_stride: usize,
        a: &CudaSlice<f32>,
        d: &CudaSlice<f32>,
        dt_bias: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        y_off: usize,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
        d_state: usize,
        n_groups: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .mamba2_scan_seq
            .ok_or(GpuError::MissingOp("mamba2_scan_seq"))?;
        debug_assert!(state.len() >= state_off + n_heads * head_dim * d_state);
        debug_assert!(dt.len() >= dt_off + (n_tokens - 1) * dt_stride + n_heads);
        debug_assert!(y.len() >= y_off + n_tokens * n_heads * head_dim);
        let (sp, _g1) = state.device_ptr_mut(&self.stream);
        let (xp, _g2) = xbc.device_ptr(&self.stream);
        let (tp, _g3) = dt.device_ptr(&self.stream);
        let (ap, _g4) = a.device_ptr(&self.stream);
        let (dp, _g5) = d.device_ptr(&self.stream);
        let (bp, _g6) = dt_bias.device_ptr(&self.stream);
        let (yp, _g7) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; geometry validated by the launcher (d_state
        // template pin, group divisibility) and the asserts above.
        check(unsafe {
            f(
                (sp + (state_off * 4) as u64) as *mut _,
                (xp + (xbc_off * 4) as u64) as *const _,
                (tp + (dt_off * 4) as u64) as *const _,
                dt_stride as u32,
                ap as *const _,
                dp as *const _,
                bp as *const _,
                (yp + (y_off * 4) as u64) as *mut _,
                n_tokens as u32,
                n_heads as u32,
                head_dim as u32,
                d_state as u32,
                n_groups as u32,
                self.stream_ptr(),
            )
        })
    }

    /// `mamba2_scan_seq_at` with a per-row state snapshot (the spec verify's
    /// rollback source): `snap[snap_off + t*H*hd*S ..]` holds the state after
    /// row t. The walk itself is bit-identical to the plain scan.
    #[allow(clippy::too_many_arguments)]
    pub fn mamba2_scan_seq_snap_at(
        &self,
        state: &mut CudaSlice<f32>,
        state_off: usize,
        xbc: &CudaSlice<f32>,
        xbc_off: usize,
        dt: &CudaSlice<f32>,
        dt_off: usize,
        dt_stride: usize,
        a: &CudaSlice<f32>,
        d: &CudaSlice<f32>,
        dt_bias: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        y_off: usize,
        snap: &mut CudaSlice<f32>,
        snap_off: usize,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
        d_state: usize,
        n_groups: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .mamba2_scan_seq_snap
            .ok_or(GpuError::MissingOp("mamba2_scan_seq_snap"))?;
        let state_elems = n_heads * head_dim * d_state;
        debug_assert!(state.len() >= state_off + state_elems);
        debug_assert!(snap.len() >= snap_off + n_tokens * state_elems);
        let (sp, _g1) = state.device_ptr_mut(&self.stream);
        let (xp, _g2) = xbc.device_ptr(&self.stream);
        let (tp, _g3) = dt.device_ptr(&self.stream);
        let (ap, _g4) = a.device_ptr(&self.stream);
        let (dp, _g5) = d.device_ptr(&self.stream);
        let (bp, _g6) = dt_bias.device_ptr(&self.stream);
        let (yp, _g7) = y.device_ptr_mut(&self.stream);
        let (np, _g8) = snap.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; same geometry validation as the plain scan.
        check(unsafe {
            f(
                (sp + (state_off * 4) as u64) as *mut _,
                (xp + (xbc_off * 4) as u64) as *const _,
                (tp + (dt_off * 4) as u64) as *const _,
                dt_stride as u32,
                ap as *const _,
                dp as *const _,
                bp as *const _,
                (yp + (y_off * 4) as u64) as *mut _,
                (np + (snap_off * 4) as u64) as *mut _,
                n_tokens as u32,
                n_heads as u32,
                head_dim as u32,
                d_state as u32,
                n_groups as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Both spec-verify kernels present (the nemotron spec core's gate).
    pub fn has_spec_verify_mamba(&self) -> bool {
        self.kernels.mamba2_scan_seq_snap.is_some() && self.kernels.copy_rows_strided.is_some()
    }

    /// Strided-rows copy: `dst[dst_off + r*len ..][..len] = src[src_off +
    /// r*src_stride ..][..len]` (f32 elements) - the verify round's
    /// conv-input snapshots (the xBC span inside the fused in_proj rows).
    #[allow(clippy::too_many_arguments)]
    pub fn copy_rows_strided(
        &self,
        src: &CudaSlice<f32>,
        src_off: usize,
        src_stride: usize,
        dst: &mut CudaSlice<f32>,
        dst_off: usize,
        len: usize,
        rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .copy_rows_strided
            .ok_or(GpuError::MissingOp("copy_rows_strided"))?;
        debug_assert!(src.len() >= src_off + (rows - 1) * src_stride + len);
        debug_assert!(dst.len() >= dst_off + rows * len);
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (dp, _g2) = dst.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; bounds asserted above.
        check(unsafe {
            f(
                sp as *const _,
                src_off as u32,
                src_stride as u32,
                (dp + (dst_off * 4) as u64) as *mut _,
                len as u32,
                rows as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Grouped gated RMSNorm (Mixer2RMSNormGated): gate first in f32
    /// (`x * silu(z)`), variance per group of `d / n_groups` channels,
    /// per-channel `weight` [d]. `z` rides inside rows of stride `z_stride`
    /// at element offset `z_off` (the in_proj output's z span).
    #[allow(clippy::too_many_arguments)]
    pub fn mamba_rmsnorm_gated_g(
        &self,
        x: &CudaSlice<f32>,
        z: &CudaSlice<f32>,
        z_off: usize,
        z_stride: usize,
        weight: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n_tokens: usize,
        d: usize,
        n_groups: usize,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .mamba_rmsnorm_gated_g
            .ok_or(GpuError::MissingOp("mamba_rmsnorm_gated_g"))?;
        debug_assert!(z.len() >= z_off + (n_tokens - 1) * z_stride + d);
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (zp, _g2) = z.device_ptr(&self.stream);
        let (wp, _g3) = weight.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; sizes checked above.
        check(unsafe {
            f(
                xp as *const _,
                (zp + (z_off * 4) as u64) as *const _,
                z_stride as u32,
                wp as *const _,
                op as *mut _,
                n_tokens as u32,
                d as u32,
                n_groups as u32,
                eps,
                self.stream_ptr(),
            )
        })
    }

    /// GEMV over a checkpoint FP8 plane held as an [`F8RowPlane`] whose row
    /// scales are the per-tensor `weight_scale` broadcast (byte-exact
    /// residency - see `fp8_ckpt_to_f8row`). f32 x in, f32 y out.
    pub fn f8r_gemv(
        &self,
        w: &F8RowPlane,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .f8r_gemv
            .ok_or(GpuError::MissingOp("f8r_gemv"))?;
        debug_assert_eq!(w.data.len(), in_dim * out_dim);
        debug_assert_eq!(w.scale.len(), out_dim);
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (sp, _g2) = w.scale.device_ptr(&self.stream);
        let (xp, _g3) = x.device_ptr(&self.stream);
        let (yp, _g4) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; plane geometry checked above.
        check(unsafe {
            f(
                dp as *const _,
                sp as *const _,
                xp as *const _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Upload a modelopt FP8 checkpoint plane (e4m3 bytes [out, in] +
    /// per-tensor f32 `weight_scale`) as an [`F8RowPlane`]: weight bytes
    /// byte-exact, the scalar broadcast into the per-row array. No
    /// requantization anywhere - the consumer multiplies the f32 scale in
    /// the epilogue, so this residency is exactly the checkpoint's numbers.
    pub fn fp8_ckpt_to_f8row(
        &self,
        bytes: &[u8],
        weight_scale: f32,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<F8RowPlane, GpuError> {
        if bytes.len() != in_dim * out_dim {
            return Err(GpuError::Driver(format!(
                "fp8_ckpt_to_f8row: {} bytes for [{out_dim}, {in_dim}] e4m3",
                bytes.len()
            )));
        }
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
        let data: CudaSlice<u8> = self.stream.clone_htod(bytes).map_err(drv)?;
        let scale: CudaSlice<f32> = self
            .stream
            .clone_htod(&vec![weight_scale; out_dim])
            .map_err(drv)?;
        Ok(F8RowPlane { data, scale })
    }

    /// The PER-ROW twin of [`Self::fp8_ckpt_to_f8row`]: a channel-scaled fp8
    /// export ships one scale per output row rather than one per tensor, so
    /// there is nothing to broadcast - the vector goes up as it is.
    ///
    /// This is what IBM's `granite-4.2-*-fp8` carries (`weight` e4m3 [n, k] +
    /// `weight_scale` BF16 [n, 1]), and it is the shape
    /// `modelopt::fp8_channel_view` decodes. Weight bytes are byte-exact and
    /// the scale is applied in the consumer's epilogue, so this residency is
    /// the checkpoint's own numbers - no requantization.
    pub fn fp8_ckpt_to_f8row_rows(
        &self,
        bytes: &[u8],
        row_scales: &[f32],
        in_dim: usize,
        out_dim: usize,
    ) -> Result<F8RowPlane, GpuError> {
        if bytes.len() != in_dim * out_dim {
            return Err(GpuError::Driver(format!(
                "fp8_ckpt_to_f8row_rows: {} bytes for [{out_dim}, {in_dim}] e4m3",
                bytes.len()
            )));
        }
        if row_scales.len() != out_dim {
            return Err(GpuError::Driver(format!(
                "fp8_ckpt_to_f8row_rows: {} scales for {out_dim} output rows",
                row_scales.len()
            )));
        }
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
        let data: CudaSlice<u8> = self.stream.clone_htod(bytes).map_err(drv)?;
        let scale: CudaSlice<f32> = self.stream.clone_htod(row_scales).map_err(drv)?;
        Ok(F8RowPlane { data, scale })
    }
}

/// f32 elements of one kept verify row (slot 746): x (H*hd) | B (G*S) | raw
/// dt (H).
pub fn mamba2_keep_row(n_heads: usize, head_dim: usize, d_state: usize, n_groups: usize) -> usize {
    n_heads * head_dim + n_groups * d_state + n_heads
}
