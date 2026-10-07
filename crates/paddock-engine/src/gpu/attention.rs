//! attention decode/prefill entry points (dense, paged, partial/combine, spec).

use super::error::*;
use super::*;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::f16;

impl GpuExecutor {
    /// The sinks buffer a family passes when it has no attention sinks.
    ///
    /// Every attention entry point below takes `sinks: &CudaSlice<f32>` and folds
    /// it into the softmax denominator as `l += exp(sink - max)` (see the combine
    /// kernel's `l += __expf(sinks[h] - gm)` and the prefill kernels' `m0 =
    /// sinks[h], l0 = 1`). So the buffer is not optional, and its no-op value is
    /// a very negative sentinel, not zero: a zeroed buffer contributes
    /// `exp(0 - max)`, a phantom unit-logit competitor that steals real
    /// probability mass from every actual position. The theft is
    /// ~`exp(-max_score)` of the denominator - invisible at max score 10, ~0.7%
    /// at 5, ~13% at 2 - so it degrades quietly and worst on small KQ scores
    /// (models with a sub-`1/sqrt(head_dim)` scale, short contexts, early
    /// layers).
    ///
    /// Allocate through this instead of `alloc()`, which zeroes. Laguna shipped
    /// with `alloc(n_heads)` and the comment "zeroed by alloc" at every site for
    /// weeks; that is exactly how a wrong-by-default value survives review.
    ///
    /// -1e30 rather than `f32::NEG_INFINITY`, because the two are not
    /// interchangeable at the edges. The sink folds in as
    /// `l = l*exp(m - mnew) + exp(sink - mnew)` with `mnew = max(m, sink)`, and
    /// the kernels init a row's running max to -inf. A row with no live key at
    /// all - a padded row in a ragged tile - therefore reaches the fold with
    /// `m = -inf`; a -inf sink makes `mnew = -inf` and `exp(-inf - -inf)` = NaN,
    /// poisoning that row. -1e30 is finite, so such a row lands on `l = 1,
    /// out = 0`, while live rows still get `exp(-1e30 - m) == 0` - the exact
    /// identity either way. (A/B'd on laguna: -inf and -1e30 produced
    /// byte-identical text on all four temp-0 probes, so the NaN is a latent
    /// hazard here, not an observed one. -1e30 costs nothing to prefer.)
    /// -1e30 is also the kernels' masked-lane sentinel; that collision is
    /// already handled (see prefill.cuh's forced-zero masked weights).
    ///
    /// Magnitude, measured on laguna Q4_K_M vs the same-weights llama.cpp
    /// oracle (which passes no sinks at all, i.e. plain softmax): fixing the
    /// zero moved token-0 logit shape CLOSER to the oracle on 4 of 4 probes
    /// (mean |err| over llama's top-6, anchored on its top-1: 0.174->0.168,
    /// 0.302->0.211, 0.165->0.147, 0.262->0.250 nats). Greedy TEXT agreement
    /// went the other way on 2 of 4 - but the flipped tokens sit on top-1/top-2
    /// gaps of 0.017 and 0.230 nats, inside the ~0.15-0.30 nat engine-vs-engine
    /// noise floor that our int8 activation path carries against ggml's Q8_1.
    /// Exact-greedy agreement at those positions is a coin toss in either
    /// direction; the logit-space number is the one that means something.
    pub fn alloc_no_sinks(&self, n_heads: usize) -> Result<CudaSlice<f32>, GpuError> {
        self.stream
            .clone_htod(&vec![-1e30f32; n_heads])
            .map_err(drv)
    }

    /// Batched decode attention: each of `batch` sequences attends its own KV
    /// cache (contiguous `[batch, max_ctx, kv_dim]`) at its own `positions[b]`.
    /// `swa_window` = 0 for full attention. The per-sequence attention behind
    /// continuous batching.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode_batch(
        &self,
        q: &CudaSlice<f32>,
        kc: &CudaSlice<u8>,
        vc: &CudaSlice<u8>,
        sinks: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        slots: Option<&CudaSlice<u32>>,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_ctx: usize,
        kv_dim: usize,
        swa_window: usize,
        batch: usize,
        scale: f32,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .attn_decode_batch
            .ok_or(GpuError::MissingOp("attn_decode_batch"))?;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = kc.device_ptr(&self.stream);
        let (vp, _g3) = vc.device_ptr(&self.stream);
        let (sp, _g4) = sinks.device_ptr(&self.stream);
        let (op, _g5) = out.device_ptr_mut(&self.stream);
        let (pp, _g6) = positions.device_ptr(&self.stream);
        let slot_guard = slots.map(|s| s.device_ptr(&self.stream));
        let slp = match &slot_guard {
            Some((p, _)) => *p as *const core::ffi::c_void,
            None => std::ptr::null(),
        };
        // SAFETY: ABI contract; per-sequence caches sized [batch, max_ctx, kv_dim]
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                sp as *const _,
                op as *mut _,
                pp as *const _,
                slp,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                max_ctx as u32,
                kv_dim as u32,
                swa_window as u32,
                batch as u32,
                scale,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_attn_decode_batch_ps(&self) -> bool {
        self.kernels.attn_decode_batch_ps.is_some()
    }

    pub fn has_attn_decode_fmha(&self) -> bool {
        self.kernels.attn_decode_fmha.is_some()
    }

    /// `attn_decode_batch` with the PARALLEL-SCORE walk (slot 536).
    ///
    /// Same arguments, same result class, different summation order in the
    /// score: the shipped walk gives the per-key dot product to `n_t` threads
    /// (16 of 256 at head_dim 256) and parks the other seven warps at the next
    /// barrier for its whole 256-step serial loop. ncu on the shipped kernel
    /// at c32: stall_barrier is 41.5% of the 23.0 cycles between issued
    /// instructions, DRAM throughput 1.23%, achieved occupancy 58.9% - the
    /// idle warps bind it, not bandwidth and not occupancy.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode_batch_ps(
        &self,
        q: &CudaSlice<f32>,
        kc: &CudaSlice<u8>,
        vc: &CudaSlice<u8>,
        sinks: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        slots: Option<&CudaSlice<u32>>,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_ctx: usize,
        kv_dim: usize,
        swa_window: usize,
        batch: usize,
        scale: f32,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .attn_decode_batch_ps
            .ok_or(GpuError::MissingOp("attn_decode_batch_ps"))?;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = kc.device_ptr(&self.stream);
        let (vp, _g3) = vc.device_ptr(&self.stream);
        let (sp, _g4) = sinks.device_ptr(&self.stream);
        let (op, _g5) = out.device_ptr_mut(&self.stream);
        let (pp, _g6) = positions.device_ptr(&self.stream);
        let slot_guard = slots.map(|s| s.device_ptr(&self.stream));
        let slp = match &slot_guard {
            Some((p, _)) => *p as *const core::ffi::c_void,
            None => std::ptr::null(),
        };
        // SAFETY: ABI contract; per-sequence caches sized [batch, max_ctx, kv_dim]
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                sp as *const _,
                op as *mut _,
                pp as *const _,
                slp,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                max_ctx as u32,
                kv_dim as u32,
                swa_window as u32,
                batch as u32,
                scale,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Paged KV append: scatter each row's kv into the shared block pool
    /// `[n_blocks, 16, kv_dim]` at `block_tables[slot*blocks_per_slot + pos/16]`,
    /// intra-block row `pos%16`. Paged twin of [`Self::kv_append_batch`].
    #[allow(clippy::too_many_arguments)]
    /// Row-window twin of [`Self::kv_append_batch_paged`] (offset pointers,
    /// the `_rows` pattern): appends rows [row_off, row_off+rows) of the
    /// staged kv/positions/slots buffers. The SWA ring-shrink sub-spans ride
    /// this - append+attend advance one sub-span at a time so the ring only
    /// has to absorb a sub-span, not a whole prefill chunk.
    #[allow(clippy::too_many_arguments)]
    /// SPLIT-KV fmha decode (slot 545): `part` is caller-owned scratch of
    /// batch*n_heads*split*(head_dim+2) f32; own numeric class, env-gated.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode_fmha_sp(
        &self,
        q: &CudaSlice<f32>,
        kc: &CudaSlice<u8>,
        vc: &CudaSlice<u8>,
        sinks: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        part: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        slots: Option<&CudaSlice<u32>>,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_ctx: usize,
        kv_dim: usize,
        swa_window: usize,
        batch: usize,
        split: usize,
        scale: f32,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .attn_decode_fmha_sp
            .ok_or(GpuError::MissingOp("attn_decode_fmha_sp"))?;
        debug_assert!(part.len() >= batch * n_heads * split * (head_dim + 2));
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = kc.device_ptr(&self.stream);
        let (vp, _g3) = vc.device_ptr(&self.stream);
        let (sp, _g4) = sinks.device_ptr(&self.stream);
        let (op, _g5) = out.device_ptr_mut(&self.stream);
        let (ptp, _g6) = part.device_ptr_mut(&self.stream);
        let (pp, _g7) = positions.device_ptr(&self.stream);
        let slot_guard = slots.map(|s| s.device_ptr(&self.stream));
        let slp = match &slot_guard {
            Some((p, _)) => *p as *const core::ffi::c_void,
            None => std::ptr::null(),
        };
        // SAFETY: ABI contract (slot 545); scratch size checked above
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                sp as *const _,
                op as *mut _,
                ptp as *mut _,
                pp as *const _,
                slp,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                max_ctx as u32,
                kv_dim as u32,
                swa_window as u32,
                batch as u32,
                split as u32,
                scale,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_attn_decode_fmha_sp(&self) -> bool {
        self.kernels.attn_decode_fmha_sp.is_some()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode_fmha(
        &self,
        q: &CudaSlice<f32>,
        kc: &CudaSlice<u8>,
        vc: &CudaSlice<u8>,
        sinks: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        slots: Option<&CudaSlice<u32>>,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_ctx: usize,
        kv_dim: usize,
        swa_window: usize,
        batch: usize,
        scale: f32,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .attn_decode_fmha
            .ok_or(GpuError::MissingOp("attn_decode_fmha"))?;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = kc.device_ptr(&self.stream);
        let (vp, _g3) = vc.device_ptr(&self.stream);
        let (sp, _g4) = sinks.device_ptr(&self.stream);
        let (op, _g5) = out.device_ptr_mut(&self.stream);
        let (pp, _g6) = positions.device_ptr(&self.stream);
        let slot_guard = slots.map(|s| s.device_ptr(&self.stream));
        let slp = match &slot_guard {
            Some((p, _)) => *p as *const core::ffi::c_void,
            None => std::ptr::null(),
        };
        // SAFETY: ABI contract; per-sequence caches sized [batch, max_ctx, kv_dim]
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                sp as *const _,
                op as *mut _,
                pp as *const _,
                slp,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                max_ctx as u32,
                kv_dim as u32,
                swa_window as u32,
                batch as u32,
                scale,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Paged KV append: scatter each row's kv into the shared block pool
    /// `[n_blocks, 16, kv_dim]` at `block_tables[slot*blocks_per_slot + pos/16]`,
    /// intra-block row `pos%16`. Paged twin of [`Self::kv_append_batch`].
    #[allow(clippy::too_many_arguments)]
    /// Row-window twin of [`Self::kv_append_batch_paged`] (offset pointers,
    /// the `_rows` pattern): appends rows [row_off, row_off+rows) of the
    /// staged kv/positions/slots buffers. The SWA ring-shrink sub-spans ride
    /// this - append+attend advance one sub-span at a time so the ring only
    /// has to absorb a sub-span, not a whole prefill chunk.
    #[allow(clippy::too_many_arguments)]
    pub fn kv_append_batch_paged_rows(
        &self,
        kv: &CudaSlice<f32>,
        pool: &mut CudaSlice<u8>,
        positions: &CudaSlice<u32>,
        slots: Option<&CudaSlice<u32>>,
        block_tables: &CudaSlice<u32>,
        blocks_per_slot: usize,
        kv_dim: usize,
        row_off: usize,
        rows: usize,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        self.kv_probe(kv, row_off * kv_dim, rows * kv_dim, rows);
        let f = self
            .kernels
            .kv_append_batch_paged
            .ok_or(GpuError::MissingOp("kv_append_batch_paged"))?;
        let kv_bytes = (row_off * kv_dim * 4) as u64;
        let u_bytes = (row_off * 4) as u64;
        let (kp, _g1) = kv.device_ptr(&self.stream);
        let (cp, _g2) = pool.device_ptr_mut(&self.stream);
        let (pp, _g3) = positions.device_ptr(&self.stream);
        let slot_guard = slots.map(|s| s.device_ptr(&self.stream));
        let slp = match &slot_guard {
            Some((p, _)) => (*p + u_bytes) as *const core::ffi::c_void,
            None => std::ptr::null(),
        };
        let (btp, _g4) = block_tables.device_ptr(&self.stream);
        // SAFETY: see kv_append_batch_paged - same allocations, offset rows
        check(unsafe {
            f(
                (kp + kv_bytes) as *const _,
                cp as *mut _,
                (pp + u_bytes) as *const _,
                slp,
                btp as *const _,
                blocks_per_slot as u32,
                kv_dim as u32,
                rows as u32,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Fused K/V norm+rope+append over a row range (kv-epilogue fold): reads
    /// the RAW k/v GEMM planes (`vp` None = V-less layer, v is the weightless
    /// norm of the raw k values), norms + ropes, appends into the paged
    /// caches. Cache bytes bit-identical to qkv_norm_rope_batch (k/v slots)
    /// + kv_append_batch_paged_rows; the kn/vn planes never materialize.
    #[allow(clippy::too_many_arguments)]
    pub fn kv_nra_rows(
        &self,
        kp: &CudaSlice<f32>,
        vp: Option<&CudaSlice<f32>>,
        kw: &CudaSlice<f32>,
        k_pool: &mut CudaSlice<u8>,
        v_pool: &mut CudaSlice<u8>,
        positions: &CudaSlice<u32>,
        slots: Option<&CudaSlice<u32>>,
        factors: Option<&CudaSlice<f32>>,
        block_tables: &CudaSlice<u32>,
        blocks_per_slot: usize,
        n_kv: usize,
        head_dim: usize,
        eps: f32,
        params: (f32, f32, f32, f32, f32, f32),
        row_off: usize,
        rows: usize,
        kv_dtype: KvDtype,
        neox: bool,
        vnorm: bool,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .kv_nra_rows3
            .ok_or(GpuError::MissingOp("kv_nra_rows3"))?;
        let (theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale) = params;
        let kv_dim = n_kv * head_dim;
        let kv_bytes = (row_off * kv_dim * 4) as u64;
        let u_bytes = (row_off * 4) as u64;
        let (kpp, _g1) = kp.device_ptr(&self.stream);
        let vp_guard = vp.map(|s| s.device_ptr(&self.stream));
        // V-less layers alias the raw k plane - the kernel only reads it
        let vpp = match &vp_guard {
            Some((p, _)) => *p,
            None => kpp,
        };
        let (kwp, _g2) = kw.device_ptr(&self.stream);
        let (kcp, _g3) = k_pool.device_ptr_mut(&self.stream);
        let (vcp, _g4) = v_pool.device_ptr_mut(&self.stream);
        let (pp, _g5) = positions.device_ptr(&self.stream);
        let slot_guard = slots.map(|s| s.device_ptr(&self.stream));
        let slp = match &slot_guard {
            Some((p, _)) => (*p + u_bytes) as *const core::ffi::c_void,
            None => std::ptr::null(),
        };
        let fac_guard = factors.map(|s| s.device_ptr(&self.stream));
        let fp = match &fac_guard {
            Some((p, _)) => *p as *const core::ffi::c_void,
            None => std::ptr::null(),
        };
        let (btp, _g6) = block_tables.device_ptr(&self.stream);
        // SAFETY: ABI contract; planes sized [rows*kv_dim], offset rows
        check(unsafe {
            f(
                (kpp + kv_bytes) as *const _,
                (vpp + kv_bytes) as *const _,
                kwp as *const _,
                kcp as *mut _,
                vcp as *mut _,
                (pp + u_bytes) as *const _,
                slp,
                fp,
                btp as *const _,
                blocks_per_slot as u32,
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
                kv_dtype as u32,
                0u32,
                neox as u32,
                vnorm as u32,
                self.stream_ptr(),
            )
        })
    }

    /// i16 twin of `kv_nra_rows`: the raw k/v planes hold bf16
    /// (the o16 GEMM epilogue's stream in the f32-typed scratch), so the
    /// row offset is 2 bytes/element.
    #[allow(clippy::too_many_arguments)]
    pub fn kv_nra_rows_i16(
        &self,
        kp: &CudaSlice<f32>,
        vp: Option<&CudaSlice<f32>>,
        kw: &CudaSlice<f32>,
        k_pool: &mut CudaSlice<u8>,
        v_pool: &mut CudaSlice<u8>,
        positions: &CudaSlice<u32>,
        slots: Option<&CudaSlice<u32>>,
        factors: Option<&CudaSlice<f32>>,
        block_tables: &CudaSlice<u32>,
        blocks_per_slot: usize,
        n_kv: usize,
        head_dim: usize,
        eps: f32,
        params: (f32, f32, f32, f32, f32, f32),
        row_off: usize,
        rows: usize,
        kv_dtype: KvDtype,
        neox: bool,
        vnorm: bool,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .kv_nra_rows3
            .ok_or(GpuError::MissingOp("kv_nra_rows3"))?;
        let (theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale) = params;
        let kv_dim = n_kv * head_dim;
        let kv_bytes = (row_off * kv_dim * 2) as u64;
        let u_bytes = (row_off * 4) as u64;
        let (kpp, _g1) = kp.device_ptr(&self.stream);
        let vp_guard = vp.map(|s| s.device_ptr(&self.stream));
        // V-less layers alias the raw k plane - the kernel only reads it
        let vpp = match &vp_guard {
            Some((p, _)) => *p,
            None => kpp,
        };
        let (kwp, _g2) = kw.device_ptr(&self.stream);
        let (kcp, _g3) = k_pool.device_ptr_mut(&self.stream);
        let (vcp, _g4) = v_pool.device_ptr_mut(&self.stream);
        let (pp, _g5) = positions.device_ptr(&self.stream);
        let slot_guard = slots.map(|s| s.device_ptr(&self.stream));
        let slp = match &slot_guard {
            Some((p, _)) => (*p + u_bytes) as *const core::ffi::c_void,
            None => std::ptr::null(),
        };
        let fac_guard = factors.map(|s| s.device_ptr(&self.stream));
        let fp = match &fac_guard {
            Some((p, _)) => *p as *const core::ffi::c_void,
            None => std::ptr::null(),
        };
        let (btp, _g6) = block_tables.device_ptr(&self.stream);
        // SAFETY: ABI contract; planes hold bf16, offset rows at 2B/elem
        check(unsafe {
            f(
                (kpp + kv_bytes) as *const _,
                (vpp + kv_bytes) as *const _,
                kwp as *const _,
                kcp as *mut _,
                vcp as *mut _,
                (pp + u_bytes) as *const _,
                slp,
                fp,
                btp as *const _,
                blocks_per_slot as u32,
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
                kv_dtype as u32,
                1u32,
                neox as u32,
                vnorm as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_kv_nra_rows(&self) -> bool {
        self.kernels.kv_nra_rows3.is_some()
    }

    pub fn has_dflash_cond_append(&self) -> bool {
        self.kernels.dflash_cond_append.is_some()
    }

    /// DFlash conditioning fold (rung C): per written row
    /// (`rows_w[..nw]`, indices into the staged fk/fv/positions/slots rows),
    /// k-norm + NEOX yarn rope + paged f16 K/V store in one launch per
    /// drafter layer. Pool bytes bit-identical to the
    /// rmsnorm_batch -> rope_yarn_batch -> per-cut kv_append chain it
    /// replaces; `norm_batch` must be the r·n_kv the chain's norm would
    /// have been launched with (it elects the reduction width).
    #[allow(clippy::too_many_arguments)]
    pub fn dflash_cond_append(
        &self,
        fk: &CudaSlice<f32>,
        fv: &CudaSlice<f32>,
        kw: &CudaSlice<f32>,
        pool_k: &mut CudaSlice<u8>,
        pool_v: &mut CudaSlice<u8>,
        rows_w: &CudaSlice<u32>,
        positions: &CudaSlice<u32>,
        slots: &CudaSlice<u32>,
        block_tables: &CudaSlice<u32>,
        blocks_per_slot: usize,
        n_kv: usize,
        head_dim: usize,
        eps: f32,
        params: (f32, f32, f32, f32, f32, f32),
        nw: usize,
        norm_batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .dflash_cond_append
            .ok_or(GpuError::MissingOp("dflash_cond_append"))?;
        let (ts, fs, cl, ch, ef, ms) = params;
        let (kp, _g1) = fk.device_ptr(&self.stream);
        let (vp, _g2) = fv.device_ptr(&self.stream);
        let (wp, _g3) = kw.device_ptr(&self.stream);
        let (pkp, _g4) = pool_k.device_ptr_mut(&self.stream);
        let (pvp, _g5) = pool_v.device_ptr_mut(&self.stream);
        let (rp, _g6) = rows_w.device_ptr(&self.stream);
        let (pp, _g7) = positions.device_ptr(&self.stream);
        let (sp, _g8) = slots.device_ptr(&self.stream);
        let (btp, _g9) = block_tables.device_ptr(&self.stream);
        // SAFETY: all buffers are live device allocations sized by the
        // caller's contract (fk/fv [r, n_kv*head_dim], rows_w[nw] indices
        // < r, pools sized to the block tables' targets)
        check(unsafe {
            f(
                kp as *const _,
                vp as *const _,
                wp as *const _,
                pkp as *mut _,
                pvp as *mut _,
                rp as *const _,
                pp as *const _,
                sp as *const _,
                btp as *const _,
                blocks_per_slot as u32,
                n_kv as u32,
                head_dim as u32,
                eps,
                ts,
                fs,
                cl,
                ch,
                ef,
                ms,
                nw as u32,
                norm_batch as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_lag_qk_nra_rows(&self) -> bool {
        self.kernels.lag_qk_nra_rows.is_some()
    }

    /// Laguna decode-tick epilogue fold: q/k per-head RMS norm + rope (plain
    /// yarn, or sectioned partial mrope when `mpos` is Some) + paged k/v
    /// append in one launch - bit-identical to the six-kernel chain. q/k may
    /// share a fused GEMV plane (q_off/k_off pick the segments); offsets and
    /// strides are in f32 elements.
    #[allow(clippy::too_many_arguments)]
    pub fn lag_qk_nra_rows(
        &self,
        q_src: &CudaSlice<f32>,
        q_off: usize,
        q_stride: usize,
        k_src: &CudaSlice<f32>,
        k_off: usize,
        k_stride: usize,
        v_src: &CudaSlice<f32>,
        v_stride: usize,
        qw: &CudaSlice<f32>,
        kw: &CudaSlice<f32>,
        q_out: &mut CudaSlice<f32>,
        k_pool: &mut CudaSlice<u8>,
        v_pool: &mut CudaSlice<u8>,
        positions: &CudaSlice<u32>,
        slots: Option<&CudaSlice<u32>>,
        mpos: Option<&CudaSlice<u32>>,
        block_tables: &CudaSlice<u32>,
        blocks_per_slot: usize,
        n_head: usize,
        n_kv: usize,
        head_dim: usize,
        n_rot: usize,
        eps: f32,
        params: (f32, f32, f32, f32, f32, f32),
        sections: [u32; 4],
        rows: usize,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .lag_qk_nra_rows
            .ok_or(GpuError::MissingOp("lag_qk_nra_rows"))?;
        let (theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale) = params;
        let (qp, _g1) = q_src.device_ptr(&self.stream);
        let (kp, _g2) = k_src.device_ptr(&self.stream);
        let (vp, _g3) = v_src.device_ptr(&self.stream);
        let (qwp, _g4) = qw.device_ptr(&self.stream);
        let (kwp, _g5) = kw.device_ptr(&self.stream);
        let (qo, _g6) = q_out.device_ptr_mut(&self.stream);
        let (kc, _g7) = k_pool.device_ptr_mut(&self.stream);
        let (vc, _g8) = v_pool.device_ptr_mut(&self.stream);
        let (pp, _g9) = positions.device_ptr(&self.stream);
        let slot_guard = slots.map(|s| s.device_ptr(&self.stream));
        let slp = match &slot_guard {
            Some((p, _)) => *p as *const core::ffi::c_void,
            None => std::ptr::null(),
        };
        let mpos_guard = mpos.map(|m| m.device_ptr(&self.stream));
        let mpp = match &mpos_guard {
            Some((p, _)) => *p as *const core::ffi::c_void,
            None => std::ptr::null(),
        };
        let (btp, _g10) = block_tables.device_ptr(&self.stream);
        // SAFETY: ABI contract; pools sized [n_blocks, 16, kv_dim] * dtype
        check(unsafe {
            f(
                qp as *const _,
                q_off as u32,
                q_stride as u32,
                kp as *const _,
                k_off as u32,
                k_stride as u32,
                vp as *const _,
                v_stride as u32,
                qwp as *const _,
                kwp as *const _,
                qo as *mut _,
                kc as *mut _,
                vc as *mut _,
                pp as *const _,
                slp,
                mpp,
                btp as *const _,
                blocks_per_slot as u32,
                n_head as u32,
                n_kv as u32,
                head_dim as u32,
                n_rot as u32,
                eps,
                theta_scale,
                freq_scale,
                corr_low,
                corr_high,
                ext_factor,
                mscale,
                sections[0],
                sections[1],
                sections[2],
                sections[3],
                rows as u32,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_q36_qkg_nra_rows(&self) -> bool {
        self.kernels.q36_qkg_nra_rows.is_some()
    }

    /// Qwen3.5-family fused-plane prefill consumer: one launch replaces
    /// split_qg + rmsnorm(q) + rmsnorm(k) + mrope(q) + mrope(k) + paged
    /// append(k) + append(v), reading the one-GEMM `[q|gate|k|v]` plane
    /// (q heads `[q(hd)|gate(hd)]` interleaved per head). Bit-identical to
    /// the chain. Offsets/stride in f32 elements; hd 256 / n_rot 64 only.
    #[allow(clippy::too_many_arguments)]
    pub fn q36_qkg_nra_rows(
        &self,
        qkg: &CudaSlice<f32>,
        q_off: usize,
        row_stride: usize,
        k_off: usize,
        v_off: usize,
        qw: &CudaSlice<f32>,
        kw: &CudaSlice<f32>,
        q_out: &mut CudaSlice<f32>,
        gate_out: &mut CudaSlice<f32>,
        k_pool: &mut CudaSlice<u8>,
        v_pool: &mut CudaSlice<u8>,
        positions: &CudaSlice<u32>,
        slots: Option<&CudaSlice<u32>>,
        mpos: &CudaSlice<u32>,
        block_tables: &CudaSlice<u32>,
        blocks_per_slot: usize,
        n_head: usize,
        n_kv: usize,
        head_dim: usize,
        n_rot: usize,
        eps: f32,
        params: (f32, f32, f32, f32, f32, f32),
        sections: [u32; 4],
        rows: usize,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q36_qkg_nra_rows
            .ok_or(GpuError::MissingOp("q36_qkg_nra_rows"))?;
        let (theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale) = params;
        let (qkgp, _g1) = qkg.device_ptr(&self.stream);
        let (qwp, _g2) = qw.device_ptr(&self.stream);
        let (kwp, _g3) = kw.device_ptr(&self.stream);
        let (qo, _g4) = q_out.device_ptr_mut(&self.stream);
        let (go, _g5) = gate_out.device_ptr_mut(&self.stream);
        let (kc, _g6) = k_pool.device_ptr_mut(&self.stream);
        let (vc, _g7) = v_pool.device_ptr_mut(&self.stream);
        let (pp, _g8) = positions.device_ptr(&self.stream);
        let slot_guard = slots.map(|s| s.device_ptr(&self.stream));
        let slp = match &slot_guard {
            Some((p, _)) => *p as *const core::ffi::c_void,
            None => std::ptr::null(),
        };
        let (mpp, _g9) = mpos.device_ptr(&self.stream);
        let (btp, _g10) = block_tables.device_ptr(&self.stream);
        // SAFETY: ABI contract; pools sized [n_blocks, 16, kv_dim] * dtype
        check(unsafe {
            f(
                qkgp as *const _,
                q_off as u32,
                row_stride as u32,
                k_off as u32,
                v_off as u32,
                qwp as *const _,
                kwp as *const _,
                qo as *mut _,
                go as *mut _,
                kc as *mut _,
                vc as *mut _,
                pp as *const _,
                slp,
                mpp as *const _,
                btp as *const _,
                blocks_per_slot as u32,
                n_head as u32,
                n_kv as u32,
                head_dim as u32,
                n_rot as u32,
                eps,
                theta_scale,
                freq_scale,
                corr_low,
                corr_high,
                ext_factor,
                mscale,
                sections[0],
                sections[1],
                sections[2],
                sections[3],
                rows as u32,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn kv_append_batch_paged(
        &self,
        kv: &CudaSlice<f32>,
        pool: &mut CudaSlice<u8>,
        positions: &CudaSlice<u32>,
        slots: Option<&CudaSlice<u32>>,
        block_tables: &CudaSlice<u32>,
        blocks_per_slot: usize,
        kv_dim: usize,
        batch: usize,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        self.kv_probe(kv, 0, batch * kv_dim, batch);
        let f = self
            .kernels
            .kv_append_batch_paged
            .ok_or(GpuError::MissingOp("kv_append_batch_paged"))?;
        let (kp, _g1) = kv.device_ptr(&self.stream);
        let (cp, _g2) = pool.device_ptr_mut(&self.stream);
        let (pp, _g3) = positions.device_ptr(&self.stream);
        let slot_guard = slots.map(|s| s.device_ptr(&self.stream));
        let slp = match &slot_guard {
            Some((p, _)) => *p as *const core::ffi::c_void,
            None => std::ptr::null(),
        };
        let (btp, _g4) = block_tables.device_ptr(&self.stream);
        // SAFETY: ABI contract; pool sized [n_blocks, 16, kv_dim] * dtype bytes
        check(unsafe {
            f(
                kp as *const _,
                cp as *mut _,
                pp as *const _,
                slp,
                btp as *const _,
                blocks_per_slot as u32,
                kv_dim as u32,
                batch as u32,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }

    /// True when the pack carries the fused rope+append (entry 318).
    pub fn has_rope_norm_qk_append_paged(&self) -> bool {
        self.kernels.rope_norm_qk_append_paged.is_some()
    }

    /// Fused NORM-rope(q in place) + NORM-rope(k)->paged append + v paged
    /// append - granite's 4-launch rope/append band as one kernel. `k` is
    /// consumed raw and never written back (its roped values land straight
    /// in the pool; nothing reads the staging plane after the append).
    /// Cache/q bytes bit-identical to the split chain: same per-warp theta
    /// chain, same `pd_kv_store`.
    #[allow(clippy::too_many_arguments)]
    /// granite fused wqkv (f8row class): one mma over the q|k|v-concat plane
    /// into K-split partials in `part`, then combine + NORM-rope + paged
    /// K/V append in one kernel. `part` >= 8 * (nh+2*nkv)*hd * batch f32.
    /// Roped q lands in `q`; k/v go straight to the pools.
    #[allow(clippy::too_many_arguments)]
    pub fn f8row_qkv_rope_norm_paged(
        &self,
        plane: &F8RowPlane,
        in_dim: usize,
        xq: &CudaSlice<i8>,
        xrs: &CudaSlice<f32>,
        part: &mut CudaSlice<f32>,
        q: &mut CudaSlice<f32>,
        pool_k: &mut CudaSlice<u8>,
        pool_v: &mut CudaSlice<u8>,
        positions: &CudaSlice<u32>,
        slots: Option<&CudaSlice<u32>>,
        block_tables: &CudaSlice<u32>,
        blocks_per_slot: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        rope: (f32, f32, f32, f32, f32, f32),
        batch: usize,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .f8row_gemm_mma_qkv_norm_paged
            .ok_or(GpuError::MissingOp("f8row_gemm_mma_qkv_norm_paged"))?;
        let (theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale) = rope;
        let (dp, _g0) = plane.data.device_ptr(&self.stream);
        let (wsp, _g1) = plane.scale.device_ptr(&self.stream);
        let (xqp, _g2) = xq.device_ptr(&self.stream);
        let (xsp, _g3) = xrs.device_ptr(&self.stream);
        let (pp, _g4) = part.device_ptr_mut(&self.stream);
        let (qp, _g5) = q.device_ptr_mut(&self.stream);
        let (pkp, _g6) = pool_k.device_ptr_mut(&self.stream);
        let (pvp, _g7) = pool_v.device_ptr_mut(&self.stream);
        let (posp, _g8) = positions.device_ptr(&self.stream);
        let slot_guard = slots.map(|s| s.device_ptr(&self.stream));
        let slp = match &slot_guard {
            Some((p, _)) => *p as *const core::ffi::c_void,
            None => std::ptr::null(),
        };
        let (btp, _g9) = block_tables.device_ptr(&self.stream);
        // SAFETY: ABI contract; part/pools sized per the field doc
        check(unsafe {
            f(
                dp as *const _,
                wsp as *const _,
                xqp as *const _,
                xsp as *const _,
                pp as *mut _,
                qp as *mut _,
                pkp as *mut _,
                pvp as *mut _,
                posp as *const _,
                slp,
                btp as *const _,
                blocks_per_slot as u32,
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
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Whether the pack ships the granite fused-wqkv f8row entry.
    pub fn has_f8row_qkv_rope_norm_paged(&self) -> bool {
        self.kernels.f8row_gemm_mma_qkv_norm_paged.is_some()
    }

    /// pf-side rope-only twin: combine+NORM-rope+paged-append over an
    /// already-computed fused-qkv plane (`part` in the nz=1 partials layout -
    /// exactly the fused-plane GEMM's y). Batch uncapped.
    #[allow(clippy::too_many_arguments)]
    pub fn f8row_qkv_rope_from_y_paged(
        &self,
        part: &CudaSlice<f32>,
        q: &mut CudaSlice<f32>,
        pool_k: &mut CudaSlice<u8>,
        pool_v: &mut CudaSlice<u8>,
        positions: &CudaSlice<u32>,
        slots: Option<&CudaSlice<u32>>,
        block_tables: &CudaSlice<u32>,
        blocks_per_slot: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        rope: (f32, f32, f32, f32, f32, f32),
        batch: usize,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .f8row_qkv_rope_norm_from_y_paged
            .ok_or(GpuError::MissingOp("f8row_qkv_rope_norm_from_y_paged"))?;
        let (theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale) = rope;
        let (pp, _g0) = part.device_ptr(&self.stream);
        let (qp, _g1) = q.device_ptr_mut(&self.stream);
        let (pkp, _g2) = pool_k.device_ptr_mut(&self.stream);
        let (pvp, _g3) = pool_v.device_ptr_mut(&self.stream);
        let (posp, _g4) = positions.device_ptr(&self.stream);
        let slot_guard = slots.map(|s| s.device_ptr(&self.stream));
        let slp = match &slot_guard {
            Some((p, _)) => *p as *const core::ffi::c_void,
            None => std::ptr::null(),
        };
        let (btp, _g5) = block_tables.device_ptr(&self.stream);
        // SAFETY: ABI contract; part holds [batch, (nh+2*nkv)*hd] f32
        check(unsafe {
            f(
                pp as *const _,
                qp as *mut _,
                pkp as *mut _,
                pvp as *mut _,
                posp as *const _,
                slp,
                btp as *const _,
                blocks_per_slot as u32,
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
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Whether the pack ships the pf-side rope-only twin.
    pub fn has_f8row_qkv_rope_from_y(&self) -> bool {
        self.kernels.f8row_qkv_rope_norm_from_y_paged.is_some()
    }

    /// Fold `nz` raw partial planes of a fused q|k|v GEMM (layout
    /// `[nz][batch][(nh+2*nkv)*hd]` f32, fixed ascending order, then
    /// `part_scale`) and run NORM-rope + paged K/V append in the same
    /// launch. `nz == 1, part_scale == 1.0` is exactly
    /// [`Self::f8row_qkv_rope_from_y_paged`]; the nvf4 route hands it the
    /// split GEMM's raw slices and the plane's `scale2`, which reproduces
    /// pd_nvf4_sk_reduce's arithmetic bit for bit without the reduce launch.
    #[allow(clippy::too_many_arguments)]
    pub fn qkv_rope_norm_from_parts_paged(
        &self,
        part: &CudaSlice<f32>,
        nz: u32,
        part_scale: f32,
        q: &mut CudaSlice<f32>,
        pool_k: &mut CudaSlice<u8>,
        pool_v: &mut CudaSlice<u8>,
        positions: &CudaSlice<u32>,
        slots: Option<&CudaSlice<u32>>,
        block_tables: &CudaSlice<u32>,
        blocks_per_slot: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        rope: (f32, f32, f32, f32, f32, f32),
        batch: usize,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .qkv_rope_norm_from_parts_paged
            .ok_or(GpuError::MissingOp("qkv_rope_norm_from_parts_paged"))?;
        let rowd = (n_heads + 2 * n_kv_heads) * head_dim;
        if part.len() < nz as usize * batch * rowd {
            return Err(GpuError::Unsupported(format!(
                "qkv_rope_norm_from_parts_paged: part {} < {nz} x {batch} x {rowd}",
                part.len()
            )));
        }
        let (theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale) = rope;
        let (pp, _g0) = part.device_ptr(&self.stream);
        let (qp, _g1) = q.device_ptr_mut(&self.stream);
        let (pkp, _g2) = pool_k.device_ptr_mut(&self.stream);
        let (pvp, _g3) = pool_v.device_ptr_mut(&self.stream);
        let (posp, _g4) = positions.device_ptr(&self.stream);
        let slot_guard = slots.map(|s| s.device_ptr(&self.stream));
        let slp = match &slot_guard {
            Some((p, _)) => *p as *const core::ffi::c_void,
            None => std::ptr::null(),
        };
        let (btp, _g5) = block_tables.device_ptr(&self.stream);
        // SAFETY: ABI contract; part holds [nz, batch, rowd] f32 (checked)
        check(unsafe {
            f(
                pp as *const _,
                nz,
                part_scale,
                qp as *mut _,
                pkp as *mut _,
                pvp as *mut _,
                posp as *const _,
                slp,
                btp as *const _,
                blocks_per_slot as u32,
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
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Whether the pack ships the partials-consuming fused-qkv rope kernel.
    pub fn has_qkv_rope_from_parts(&self) -> bool {
        self.kernels.qkv_rope_norm_from_parts_paged.is_some()
    }

    // A kernel launcher: the parameter list is the kernel's.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_norm_qk_append_paged(
        &self,
        q: &mut CudaSlice<f32>,
        k: &mut CudaSlice<f32>,
        v: &CudaSlice<f32>,
        pool_k: &mut CudaSlice<u8>,
        pool_v: &mut CudaSlice<u8>,
        positions: &CudaSlice<u32>,
        slots: Option<&CudaSlice<u32>>,
        block_tables: &CudaSlice<u32>,
        blocks_per_slot: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        rope: (f32, f32, f32, f32, f32, f32),
        batch: usize,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .rope_norm_qk_append_paged
            .ok_or(GpuError::MissingOp("rope_norm_qk_append_paged"))?;
        let (theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale) = rope;
        let (qp, _g1) = q.device_ptr_mut(&self.stream);
        let (kp, _g2) = k.device_ptr_mut(&self.stream);
        let (vp, _g3) = v.device_ptr(&self.stream);
        let (pkp, _g4) = pool_k.device_ptr_mut(&self.stream);
        let (pvp, _g5) = pool_v.device_ptr_mut(&self.stream);
        let (pp, _g6) = positions.device_ptr(&self.stream);
        let slot_guard = slots.map(|s| s.device_ptr(&self.stream));
        let slp = match &slot_guard {
            Some((p, _)) => *p as *const core::ffi::c_void,
            None => std::ptr::null(),
        };
        let (btp, _g7) = block_tables.device_ptr(&self.stream);
        // SAFETY: ABI contract; pools sized [n_blocks, 16, kv_dim] * dtype bytes
        check(unsafe {
            f(
                qp as *mut _,
                kp as *mut _,
                vp as *const _,
                pkp as *mut _,
                pvp as *mut _,
                pp as *const _,
                slp,
                btp as *const _,
                blocks_per_slot as u32,
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
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }

    /// True when the pack carries the ring twin of the fused rope+append
    /// (entry 384).
    pub fn has_rope_qk_append_paged_ring(&self) -> bool {
        self.kernels.rope_qk_append_paged_ring.is_some()
    }

    /// Ring twin of [`Self::rope_norm_qk_append_paged`]: rope
    /// turns by the true position stream (`positions`) while the K/V appends
    /// land at the R-SWA ring's WRITE stream (`wpos`), and `neox` picks the
    /// rope pair layout. Cache/q bytes bit-identical to the four-kernel
    /// chain rope(q) + rope(k) + append(k)@wpos + append(v)@wpos.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_qk_append_paged_ring(
        &self,
        q: &mut CudaSlice<f32>,
        k: &mut CudaSlice<f32>,
        v: &CudaSlice<f32>,
        pool_k: &mut CudaSlice<u8>,
        pool_v: &mut CudaSlice<u8>,
        positions: &CudaSlice<u32>,
        wpos: &CudaSlice<u32>,
        slots: Option<&CudaSlice<u32>>,
        block_tables: &CudaSlice<u32>,
        blocks_per_slot: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        rope: (f32, f32, f32, f32, f32, f32),
        batch: usize,
        neox: bool,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .rope_qk_append_paged_ring
            .ok_or(GpuError::MissingOp("rope_qk_append_paged_ring"))?;
        let (theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale) = rope;
        let (qp, _g1) = q.device_ptr_mut(&self.stream);
        let (kp, _g2) = k.device_ptr_mut(&self.stream);
        let (vp, _g3) = v.device_ptr(&self.stream);
        let (pkp, _g4) = pool_k.device_ptr_mut(&self.stream);
        let (pvp, _g5) = pool_v.device_ptr_mut(&self.stream);
        let (pp, _g6) = positions.device_ptr(&self.stream);
        let (wp, _g7) = wpos.device_ptr(&self.stream);
        let slot_guard = slots.map(|s| s.device_ptr(&self.stream));
        let slp = match &slot_guard {
            Some((p, _)) => *p as *const core::ffi::c_void,
            None => std::ptr::null(),
        };
        let (btp, _g8) = block_tables.device_ptr(&self.stream);
        // SAFETY: ABI contract; pools sized [n_blocks, 16, kv_dim] * dtype bytes
        check(unsafe {
            f(
                qp as *mut _,
                kp as *mut _,
                vp as *const _,
                pkp as *mut _,
                pvp as *mut _,
                pp as *const _,
                wp as *const _,
                slp,
                btp as *const _,
                blocks_per_slot as u32,
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
                neox as u32,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Paged decode attention: reads K/V from the shared block pool via each
    /// slot's block table (`block_tables + slot*blocks_per_slot`). Bit-exact
    /// vs [`Self::attn_decode_batch`] - only the per-token base differs.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode_batch_paged(
        &self,
        q: &CudaSlice<f32>,
        pool_k: &CudaSlice<u8>,
        pool_v: &CudaSlice<u8>,
        sinks: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        slots: Option<&CudaSlice<u32>>,
        block_tables: &CudaSlice<u32>,
        blocks_per_slot: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_dim: usize,
        swa_window: usize,
        batch: usize,
        scale: f32,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .attn_decode_batch_paged
            .ok_or(GpuError::MissingOp("attn_decode_batch_paged"))?;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = pool_k.device_ptr(&self.stream);
        let (vp, _g3) = pool_v.device_ptr(&self.stream);
        let (sp, _g4) = sinks.device_ptr(&self.stream);
        let (op, _g5) = out.device_ptr_mut(&self.stream);
        let (pp, _g6) = positions.device_ptr(&self.stream);
        let slot_guard = slots.map(|s| s.device_ptr(&self.stream));
        let slp = match &slot_guard {
            Some((p, _)) => *p as *const core::ffi::c_void,
            None => std::ptr::null(),
        };
        let (btp, _g7) = block_tables.device_ptr(&self.stream);
        // SAFETY: ABI contract; pools sized [n_blocks, 16, kv_dim] * dtype bytes
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                sp as *const _,
                op as *mut _,
                pp as *const _,
                slp,
                btp as *const _,
                blocks_per_slot as u32,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                kv_dim as u32,
                swa_window as u32,
                batch as u32,
                scale,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }

    /// fused single-pass GQA-16 decode attention (slot 380) - one
    /// launch, FINAL output with the sink folded in-kernel: no partial
    /// planes, no combine. fp8 paged KV, head_dim 128, group 16 only; the
    /// election in gemma4 gates rows >= 24 and the kv_split_band <= 768 so
    /// the smem stage always fits (`pos_max` is the band CEILING, keeping
    /// captured graphs valid across the band).
    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode_fused_gqa16(
        &self,
        q: &CudaSlice<f32>,
        pool_k: &CudaSlice<u8>,
        pool_v: &CudaSlice<u8>,
        sinks: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        slots: Option<&CudaSlice<u32>>,
        block_tables: &CudaSlice<u32>,
        blocks_per_slot: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_dim: usize,
        swa_window: usize,
        batch: usize,
        pos_max: usize,
        scale: f32,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .attn_decode_fused_gqa16
            .ok_or(GpuError::MissingOp("attn_decode_fused_gqa16"))?;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = pool_k.device_ptr(&self.stream);
        let (vp, _g3) = pool_v.device_ptr(&self.stream);
        let (sp, _g4) = sinks.device_ptr(&self.stream);
        let (op, _g5) = out.device_ptr_mut(&self.stream);
        let (pp, _g6) = positions.device_ptr(&self.stream);
        let slot_guard = slots.map(|s| s.device_ptr(&self.stream));
        let slp = match &slot_guard {
            Some((p, _)) => *p as *const core::ffi::c_void,
            None => std::ptr::null(),
        };
        let (btp, _g7) = block_tables.device_ptr(&self.stream);
        // SAFETY: ABI contract; the entry itself re-checks shape (rc -2) and
        // the smem opt-in (rc -3), so a mis-election errors instead of
        // corrupting.
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                sp as *const _,
                op as *mut _,
                pp as *const _,
                slp,
                btp as *const _,
                blocks_per_slot as u32,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                kv_dim as u32,
                swa_window as u32,
                batch as u32,
                pos_max as u32,
                scale,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }

    /// whether the pack carries the fused GQA-16 decode arm (slot 380)
    pub fn has_attn_decode_fused_gqa16(&self) -> bool {
        self.kernels.attn_decode_fused_gqa16.is_some()
    }

    /// slot 458: Q16xKv128 tensor-core decode attention (muse hd128/G16).
    /// FINAL output with the sink folded in - the caller must not combine.
    /// Same params as `attn_decode_fused_gqa16` minus `pos_max`: this arm
    /// chunks the KV walk, so its shared memory is constant in context and it
    /// carries no band ceiling.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode_fmha16(
        &self,
        q: &CudaSlice<f32>,
        pool_k: &CudaSlice<u8>,
        pool_v: &CudaSlice<u8>,
        sinks: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        slots: Option<&CudaSlice<u32>>,
        block_tables: &CudaSlice<u32>,
        blocks_per_slot: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_dim: usize,
        swa_window: usize,
        batch: usize,
        scale: f32,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .attn_decode_fmha16
            .ok_or(GpuError::MissingOp("attn_decode_fmha16"))?;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = pool_k.device_ptr(&self.stream);
        let (vp, _g3) = pool_v.device_ptr(&self.stream);
        let (sp, _g4) = sinks.device_ptr(&self.stream);
        let (op, _g5) = out.device_ptr_mut(&self.stream);
        let (pp, _g6) = positions.device_ptr(&self.stream);
        let slot_guard = slots.map(|s| s.device_ptr(&self.stream));
        let slp = match &slot_guard {
            Some((p, _)) => *p as *const core::ffi::c_void,
            None => std::ptr::null(),
        };
        let (btp, _g7) = block_tables.device_ptr(&self.stream);
        // SAFETY: ABI contract; the entry re-checks shape and cc (rc -2) and
        // the smem opt-in (rc -3), so a mis-election errors instead of
        // corrupting.
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                sp as *const _,
                op as *mut _,
                pp as *const _,
                slp,
                btp as *const _,
                blocks_per_slot as u32,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                kv_dim as u32,
                swa_window as u32,
                batch as u32,
                scale,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }

    /// slot 458 presence.
    pub fn has_attn_decode_fmha16(&self) -> bool {
        self.kernels.attn_decode_fmha16.is_some()
    }

    ///  (slot 431): tcgen05/TMEM decode attention - FINAL
    /// output rows in `out`, no partials/combine. Returns Ok(true) when the
    /// pack accepted and launched (caller skips the combine), Ok(false) when
    /// the shape/arch is not covered (rc -2/-3 - caller keeps the
    /// partial+combine route), Err on a real launch failure.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode_tc5_paged(
        &self,
        q: &CudaSlice<f32>,
        pool_k: &CudaSlice<u8>,
        pool_v: &CudaSlice<u8>,
        sinks: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        slots: Option<&CudaSlice<u32>>,
        block_tables: &CudaSlice<u32>,
        blocks_per_slot: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_dim: usize,
        swa_window: usize,
        batch: usize,
        scale: f32,
        kv_dtype: KvDtype,
    ) -> Result<bool, GpuError> {
        let f = self
            .kernels
            .attn_decode_tc5_paged
            .ok_or(GpuError::MissingOp("attn_decode_tc5_paged"))?;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = pool_k.device_ptr(&self.stream);
        let (vp, _g3) = pool_v.device_ptr(&self.stream);
        let (sp, _g4) = sinks.device_ptr(&self.stream);
        let (op, _g5) = out.device_ptr_mut(&self.stream);
        let (pp, _g6) = positions.device_ptr(&self.stream);
        let slot_guard = slots.map(|s| s.device_ptr(&self.stream));
        let slp = match &slot_guard {
            Some((p, _)) => *p as *const core::ffi::c_void,
            None => std::ptr::null(),
        };
        let (btp, _g7) = block_tables.device_ptr(&self.stream);
        // SAFETY: ABI contract; the entry re-checks shape/arch (rc -2) and
        // the smem opt-in (rc -3), so a mis-election declines instead of
        // corrupting.
        let rc = unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                sp as *const _,
                op as *mut _,
                pp as *const _,
                slp,
                btp as *const _,
                blocks_per_slot as u32,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                kv_dim as u32,
                swa_window as u32,
                batch as u32,
                scale,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        };
        match rc {
            0 => Ok(true),
            -2 | -3 => Ok(false),
            e => {
                check(e)?;
                Ok(false)
            }
        }
    }

    /// whether the pack carries the tcgen05 decode arm (slot 431)
    pub fn has_attn_decode_tc5_paged(&self) -> bool {
        self.kernels.attn_decode_tc5_paged.is_some()
    }

    /// Multi-slot batched TILED (flash) prefill attention - the fast path for
    /// the encoder's many-short-sequences attention. `tile_row0`/`tile_slot`
    /// (len `n_qtiles`) tile each text so a 16-query tile never crosses a slot.
    /// Same numeric class as [`Self::attn_decode_batch`]; head_dim must be 128.
    #[allow(clippy::too_many_arguments)]
    /// whether the pack carries the batched-runs arm (slot 376)
    pub fn kernels_pf_runs_available(&self) -> bool {
        self.kernels.pf_runs_register.is_some()
    }

    /// arm (Some) or disarm (None) the batched-runs pf5 prefill
    /// attention for one coalesced pass. `offs` holds the [n_runs+1] u32
    /// prefix of run row offsets; `max_n` = widest run.
    pub fn pf_runs_register(
        &self,
        armed: Option<(&CudaSlice<u32>, u32, u32)>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .pf_runs_register
            .ok_or(GpuError::MissingOp("pf_runs_register"))?;
        let (ptr, n, maxn) = match armed {
            Some((offs, n, maxn)) => {
                let (p, _g) = offs.device_ptr(&self.stream);
                (p as *const core::ffi::c_void, n, maxn)
            }
            None => (core::ptr::null(), 0u32, 0u32),
        };
        check(unsafe { f(ptr, n, maxn) })
    }

    // A kernel launcher: the parameter list is the kernel's.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_prefill_batch(
        &self,
        q: &CudaSlice<f32>,
        kc: &CudaSlice<u8>,
        vc: &CudaSlice<u8>,
        sinks: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        slots: &CudaSlice<u32>,
        tile_row0: &CudaSlice<u32>,
        tile_slot: &CudaSlice<u32>,
        n_qtiles: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_ctx: usize,
        kv_dim: usize,
        swa_window: usize,
        n_rows: usize,
        scale: f32,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .attn_prefill_batch
            .ok_or(GpuError::MissingOp("attn_prefill_batch"))?;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = kc.device_ptr(&self.stream);
        let (vp, _g3) = vc.device_ptr(&self.stream);
        let (sp, _g4) = sinks.device_ptr(&self.stream);
        let (op, _g5) = out.device_ptr_mut(&self.stream);
        let (pp, _g6) = positions.device_ptr(&self.stream);
        let (slp, _g7) = slots.device_ptr(&self.stream);
        let (trp, _g8) = tile_row0.device_ptr(&self.stream);
        let (tsp, _g9) = tile_slot.device_ptr(&self.stream);
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                sp as *const _,
                op as *mut _,
                pp as *const _,
                slp as *const _,
                trp as *const _,
                tsp as *const _,
                n_qtiles as u32,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                max_ctx as u32,
                kv_dim as u32,
                swa_window as u32,
                n_rows as u32,
                scale,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Tensor-core multi-slot prefill attention - same contract as
    /// [`Self::attn_prefill_batch`] but the tiles stride 32 query rows and the
    /// math runs on WMMA f16 fragments (f32 softmax, f16 O accumulate - a
    /// numeric class change vs the scalar kernel; encoder calibration
    /// re-gates it). Requires head_dim 128, fp16 KV, max_ctx % 64 == 0.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_prefill_batch_f16(
        &self,
        q: &CudaSlice<f32>,
        kc: &CudaSlice<u8>,
        vc: &CudaSlice<u8>,
        sinks: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        slots: &CudaSlice<u32>,
        tile_row0: &CudaSlice<u32>,
        tile_slot: &CudaSlice<u32>,
        n_qtiles: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_ctx: usize,
        kv_dim: usize,
        swa_window: usize,
        n_rows: usize,
        scale: f32,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .attn_prefill_batch_f16
            .ok_or(GpuError::MissingOp("attn_prefill_batch_f16"))?;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = kc.device_ptr(&self.stream);
        let (vp, _g3) = vc.device_ptr(&self.stream);
        let (sp, _g4) = sinks.device_ptr(&self.stream);
        let (op, _g5) = out.device_ptr_mut(&self.stream);
        let (pp, _g6) = positions.device_ptr(&self.stream);
        let (slp, _g7) = slots.device_ptr(&self.stream);
        let (trp, _g8) = tile_row0.device_ptr(&self.stream);
        let (tsp, _g9) = tile_slot.device_ptr(&self.stream);
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                sp as *const _,
                op as *mut _,
                pp as *const _,
                slp as *const _,
                trp as *const _,
                tsp as *const _,
                n_qtiles as u32,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                max_ctx as u32,
                kv_dim as u32,
                swa_window as u32,
                n_rows as u32,
                scale,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Tiled prefill attention - same contract as [`Self::attn_decode_batch`]
    /// but one block per (q-head, 16-query tile) with K/V streamed through
    /// shared. Same numeric class, not bit-identical (per-32-key-tile online
    /// softmax). Requires head_dim == 128 and `slots` with one slot shared by
    /// every row.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_prefill(
        &self,
        q: &CudaSlice<f32>,
        kc: &CudaSlice<u8>,
        vc: &CudaSlice<u8>,
        sinks: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        slots: &CudaSlice<u32>,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_ctx: usize,
        kv_dim: usize,
        swa_window: usize,
        batch: usize,
        scale: f32,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .attn_prefill
            .ok_or(GpuError::MissingOp("attn_prefill"))?;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = kc.device_ptr(&self.stream);
        let (vp, _g3) = vc.device_ptr(&self.stream);
        let (sp, _g4) = sinks.device_ptr(&self.stream);
        let (op, _g5) = out.device_ptr_mut(&self.stream);
        let (pp, _g6) = positions.device_ptr(&self.stream);
        let (slp, _g7) = slots.device_ptr(&self.stream);
        // SAFETY: ABI contract; per-sequence caches sized [batch, max_ctx, kv_dim]
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                sp as *const _,
                op as *mut _,
                pp as *const _,
                slp as *const _,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                max_ctx as u32,
                kv_dim as u32,
                swa_window as u32,
                batch as u32,
                scale,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Whether the pack exports the paged tiled prefill (P4b).
    pub fn has_attn_prefill_paged(&self) -> bool {
        self.kernels.attn_prefill_paged.is_some()
    }

    /// Paged tiled prefill (P4b): [`Self::attn_prefill`] over the block pool.
    /// Single-slot (all rows share `slots[0]`). Bit-exact vs the dense tiled
    /// prefill; gives paged prefill the tiled perf class (vs the P4 decode-class
    /// fallback).
    #[allow(clippy::too_many_arguments)]
    pub fn attn_prefill_paged(
        &self,
        q: &CudaSlice<f32>,
        pool_k: &CudaSlice<u8>,
        pool_v: &CudaSlice<u8>,
        sinks: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        slots: &CudaSlice<u32>,
        block_tables: &CudaSlice<u32>,
        blocks_per_slot: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_dim: usize,
        swa_window: usize,
        batch: usize,
        scale: f32,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .attn_prefill_paged
            .ok_or(GpuError::MissingOp("attn_prefill_paged"))?;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = pool_k.device_ptr(&self.stream);
        let (vp, _g3) = pool_v.device_ptr(&self.stream);
        let (sp, _g4) = sinks.device_ptr(&self.stream);
        let (op, _g5) = out.device_ptr_mut(&self.stream);
        let (pp, _g6) = positions.device_ptr(&self.stream);
        let (slp, _g7) = slots.device_ptr(&self.stream);
        let (btp, _g8) = block_tables.device_ptr(&self.stream);
        // SAFETY: ABI contract; pools sized [n_blocks, 16, kv_dim] * dtype bytes
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                sp as *const _,
                op as *mut _,
                pp as *const _,
                slp as *const _,
                btp as *const _,
                blocks_per_slot as u32,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                kv_dim as u32,
                swa_window as u32,
                batch as u32,
                scale,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Row-window twin of [`Self::attn_prefill_paged`] (same entrypoint,
    /// offset pointers - the `attn_prefill_rows` pattern): rows
    /// [row_off, row_off+rows) of the staged q/positions/slots buffers, for
    /// the per-run dispatch of the coalesced multi-prompt prefill. The
    /// gemma4 global layers (head_dim 512, window 0) ride this over the
    /// budget-pool block table.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_prefill_rows_paged(
        &self,
        q: &CudaSlice<f32>,
        pool_k: &CudaSlice<u8>,
        pool_v: &CudaSlice<u8>,
        sinks: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        slots: &CudaSlice<u32>,
        block_tables: &CudaSlice<u32>,
        blocks_per_slot: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_dim: usize,
        swa_window: usize,
        row_off: usize,
        rows: usize,
        scale: f32,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .attn_prefill_paged
            .ok_or(GpuError::MissingOp("attn_prefill_paged"))?;
        let q_bytes = (row_off * n_heads * head_dim * 4) as u64;
        let u_bytes = (row_off * 4) as u64;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = pool_k.device_ptr(&self.stream);
        let (vp, _g3) = pool_v.device_ptr(&self.stream);
        let (sp, _g4) = sinks.device_ptr(&self.stream);
        let (op, _g5) = out.device_ptr_mut(&self.stream);
        let (pp, _g6) = positions.device_ptr(&self.stream);
        let (slp, _g7) = slots.device_ptr(&self.stream);
        let (btp, _g8) = block_tables.device_ptr(&self.stream);
        // SAFETY: see attn_prefill_rows - same allocations, offset rows
        check(unsafe {
            f(
                (qp + q_bytes) as *const _,
                kp as *const _,
                vp as *const _,
                sp as *const _,
                (op + q_bytes) as *mut _,
                (pp + u_bytes) as *const _,
                (slp + u_bytes) as *const _,
                btp as *const _,
                blocks_per_slot as u32,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                kv_dim as u32,
                swa_window as u32,
                rows as u32,
                scale,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Whether the pack exports the paged f16 WMMA prefill (P4b-2).
    pub fn has_attn_prefill_f16_paged(&self) -> bool {
        self.kernels.attn_prefill_f16_paged.is_some()
    }

    /// [`Self::attn_prefill_f16_paged`] with a q-row offset: reads q/positions/
    /// slots and writes out at row `row_off` of the given buffers - the span
    /// runs in PLACE inside a batched cohort (rows rb..rb+batch), removing the
    /// unified tick's per-span base-0 staging copies (2 × rows×q_dim floats per
    /// span per full-attn layer) and, on the fresh-cohort pass, the per-span
    /// pageable slot upload whose guard drop is a hidden full-stream sync.
    /// Same kernel, same bytes - bit-identical to the staged call.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_prefill_f16_paged_at(
        &self,
        q: &CudaSlice<f32>,
        pool_k: &CudaSlice<u8>,
        pool_v: &CudaSlice<u8>,
        sinks: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        slots: &CudaSlice<u32>,
        row_off: usize,
        block_tables: &CudaSlice<u32>,
        blocks_per_slot: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_dim: usize,
        swa_window: usize,
        batch: usize,
        scale: f32,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .attn_prefill_f16_paged
            .ok_or(GpuError::MissingOp("attn_prefill_f16_paged"))?;
        let q_row = n_heads * head_dim;
        debug_assert!((row_off + batch) * q_row <= q.len());
        debug_assert!((row_off + batch) * q_row <= out.len());
        debug_assert!(row_off + batch <= positions.len());
        debug_assert!(row_off + batch <= slots.len());
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = pool_k.device_ptr(&self.stream);
        let (vp, _g3) = pool_v.device_ptr(&self.stream);
        let (sp, _g4) = sinks.device_ptr(&self.stream);
        let (op, _g5) = out.device_ptr_mut(&self.stream);
        let (pp, _g6) = positions.device_ptr(&self.stream);
        let (slp, _g7) = slots.device_ptr(&self.stream);
        let (btp, _g8) = block_tables.device_ptr(&self.stream);
        let f32sz = std::mem::size_of::<f32>() as u64;
        let u32sz = std::mem::size_of::<u32>() as u64;
        let qp = qp + (row_off * q_row) as u64 * f32sz;
        let op = op + (row_off * q_row) as u64 * f32sz;
        let pp = pp + row_off as u64 * u32sz;
        let slp = slp + row_off as u64 * u32sz;
        // SAFETY: ABI contract; pools sized [n_blocks, 16, kv_dim] * dtype bytes
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                sp as *const _,
                op as *mut _,
                pp as *const _,
                slp as *const _,
                btp as *const _,
                blocks_per_slot as u32,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                kv_dim as u32,
                swa_window as u32,
                batch as u32,
                scale,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Whether the pack exports the pf7 varlen packed prefill (ABI 322).
    pub fn has_attn_prefill_f16_paged_vl(&self) -> bool {
        self.kernels.attn_prefill_f16_paged_vl.is_some()
    }

    /// pf7 varlen packed prefill attention (AF3): one launch per
    /// layer covering every eligible prefill span of the tick. `items` is
    /// stride-4 u32 per 64-head-row tile `(q_row0, span_rows,
    /// tile_flat_row0, slot)` - tiles never cross spans, so each packed CTA
    /// computes bit-identically to the per-span
    /// [`Self::attn_prefill_f16_paged_at`] twin; only the grid packing
    /// changes. fp8 pools at the pf7 shapes only (hd256, G 4/6/8) - callers
    /// pre-check and keep the per-span loop as the fallback.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_prefill_f16_paged_vl(
        &self,
        q: &CudaSlice<f32>,
        pool_k: &CudaSlice<u8>,
        pool_v: &CudaSlice<u8>,
        sinks: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        items: &CudaSlice<u32>,
        n_tiles: usize,
        block_tables: &CudaSlice<u32>,
        blocks_per_slot: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_dim: usize,
        swa_window: usize,
        scale: f32,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .attn_prefill_f16_paged_vl
            .ok_or(GpuError::MissingOp("attn_prefill_f16_paged_vl"))?;
        debug_assert!(n_tiles * 4 <= items.len());
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = pool_k.device_ptr(&self.stream);
        let (vp, _g3) = pool_v.device_ptr(&self.stream);
        let (sp, _g4) = sinks.device_ptr(&self.stream);
        let (op, _g5) = out.device_ptr_mut(&self.stream);
        let (pp, _g6) = positions.device_ptr(&self.stream);
        let (ip, _g7) = items.device_ptr(&self.stream);
        let (btp, _g8) = block_tables.device_ptr(&self.stream);
        // SAFETY: ABI contract; pools sized [n_blocks, 16, kv_dim] * dtype bytes
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                sp as *const _,
                op as *mut _,
                pp as *const _,
                ip as *const _,
                n_tiles as u32,
                btp as *const _,
                blocks_per_slot as u32,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                kv_dim as u32,
                swa_window as u32,
                scale,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Paged f16 WMMA prefill (P4b-2): [`Self::attn_prefill_f16`] over the block
    /// pool (single-slot; head_dim 256/64, fp16 KV). Bit-exact vs the dense f16
    /// prefill; gives paged prefill full perf parity with qwen35's dense default.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_prefill_f16_paged(
        &self,
        q: &CudaSlice<f32>,
        pool_k: &CudaSlice<u8>,
        pool_v: &CudaSlice<u8>,
        sinks: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        slots: &CudaSlice<u32>,
        block_tables: &CudaSlice<u32>,
        blocks_per_slot: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_dim: usize,
        swa_window: usize,
        batch: usize,
        scale: f32,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .attn_prefill_f16_paged
            .ok_or(GpuError::MissingOp("attn_prefill_f16_paged"))?;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = pool_k.device_ptr(&self.stream);
        let (vp, _g3) = pool_v.device_ptr(&self.stream);
        let (sp, _g4) = sinks.device_ptr(&self.stream);
        let (op, _g5) = out.device_ptr_mut(&self.stream);
        let (pp, _g6) = positions.device_ptr(&self.stream);
        let (slp, _g7) = slots.device_ptr(&self.stream);
        let (btp, _g8) = block_tables.device_ptr(&self.stream);
        // SAFETY: ABI contract; pools sized [n_blocks, 16, kv_dim] * dtype bytes
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                sp as *const _,
                op as *mut _,
                pp as *const _,
                slp as *const _,
                btp as *const _,
                blocks_per_slot as u32,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                kv_dim as u32,
                swa_window as u32,
                batch as u32,
                scale,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Whether the pack exports the f16 tensor-core prefill attention (P6i).
    pub fn has_attn_prefill_f16(&self) -> bool {
        self.kernels.attn_prefill_f16.is_some()
    }

    /// Tensor-core (f16 WMMA) prefill attention - same contract as
    /// [`Self::attn_prefill`] but f16 Q/K/V inputs with f16 O accumulation
    /// (llama's own prefill attention class). Requires head_dim == 256,
    /// fp16 KV, and max_ctx % 64 == 0.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_prefill_f16(
        &self,
        q: &CudaSlice<f32>,
        kc: &CudaSlice<u8>,
        vc: &CudaSlice<u8>,
        sinks: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        slots: &CudaSlice<u32>,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_ctx: usize,
        kv_dim: usize,
        swa_window: usize,
        batch: usize,
        scale: f32,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .attn_prefill_f16
            .ok_or(GpuError::MissingOp("attn_prefill_f16"))?;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = kc.device_ptr(&self.stream);
        let (vp, _g3) = vc.device_ptr(&self.stream);
        let (sp, _g4) = sinks.device_ptr(&self.stream);
        let (op, _g5) = out.device_ptr_mut(&self.stream);
        let (pp, _g6) = positions.device_ptr(&self.stream);
        let (slp, _g7) = slots.device_ptr(&self.stream);
        // SAFETY: ABI contract; per-sequence caches sized [batch, max_ctx, kv_dim]
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                sp as *const _,
                op as *mut _,
                pp as *const _,
                slp as *const _,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                max_ctx as u32,
                kv_dim as u32,
                swa_window as u32,
                batch as u32,
                scale,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }

    /// [`Self::attn_decode_batch`] over a CONSECUTIVE ROW SUB-RANGE
    /// [row_off, row_off + rows): the base pointers are offset host-side, so
    /// the kernel sees a smaller pass starting at that row. Lets a mixed-slot
    /// prefill pass dispatch attention per SLOT GROUP (no kernel/ABI change).
    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode_batch_rows(
        &self,
        q: &CudaSlice<f32>,
        kc: &CudaSlice<u8>,
        vc: &CudaSlice<u8>,
        sinks: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        slots: Option<&CudaSlice<u32>>,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_ctx: usize,
        kv_dim: usize,
        swa_window: usize,
        row_off: usize,
        rows: usize,
        scale: f32,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .attn_decode_batch
            .ok_or(GpuError::MissingOp("attn_decode_batch"))?;
        let q_bytes = (row_off * n_heads * head_dim * 4) as u64;
        let u_bytes = (row_off * 4) as u64;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = kc.device_ptr(&self.stream);
        let (vp, _g3) = vc.device_ptr(&self.stream);
        let (sp, _g4) = sinks.device_ptr(&self.stream);
        let (op, _g5) = out.device_ptr_mut(&self.stream);
        let (pp, _g6) = positions.device_ptr(&self.stream);
        let slot_guard = slots.map(|s| s.device_ptr(&self.stream));
        let slp = match &slot_guard {
            Some((p, _)) => (*p + u_bytes) as *const core::ffi::c_void,
            None => std::ptr::null(),
        };
        // SAFETY: ABI contract; caller guarantees row_off + rows <= the pass
        // row count the buffers were filled for
        check(unsafe {
            f(
                (qp + q_bytes) as *const _,
                kp as *const _,
                vp as *const _,
                sp as *const _,
                (op + q_bytes) as *mut _,
                (pp + u_bytes) as *const _,
                slp,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                max_ctx as u32,
                kv_dim as u32,
                swa_window as u32,
                rows as u32,
                scale,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }

    /// [`Self::attn_prefill_f16`] over a CONSECUTIVE ROW SUB-RANGE - see
    /// [`Self::attn_decode_batch_rows`]. The kernel reads its slot from the
    /// first row of the (offset) slots pointer, so every row in the range
    /// must belong to one slot: exactly a prefill pass's per-slot group.
    #[allow(clippy::too_many_arguments)]
    /// [`Self::attn_prefill`] over a row sub-range [row_off, row_off+rows):
    /// pointer-offset dispatch for mixed-slot prefill chunks - the scalar
    /// tiled kernel reads slots[0], so each launch must cover one slot's run.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_prefill_rows(
        &self,
        q: &CudaSlice<f32>,
        kc: &CudaSlice<u8>,
        vc: &CudaSlice<u8>,
        sinks: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        win_pos: Option<&CudaSlice<u32>>,
        slots: &CudaSlice<u32>,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_ctx: usize,
        kv_dim: usize,
        swa_window: usize,
        row_off: usize,
        rows: usize,
        scale: f32,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let q_bytes = (row_off * n_heads * head_dim * 4) as u64;
        let u_bytes = (row_off * 4) as u64;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = kc.device_ptr(&self.stream);
        let (vp, _g3) = vc.device_ptr(&self.stream);
        let (sp, _g4) = sinks.device_ptr(&self.stream);
        let (op, _g5) = out.device_ptr_mut(&self.stream);
        let (pp, _g6) = positions.device_ptr(&self.stream);
        let (slp, _g7) = slots.device_ptr(&self.stream);
        // SAFETY: see attn_decode_batch_rows - same allocation, offset rows
        if let Some(wpos) = win_pos {
            // the window floors from the rows' TRUE positions (slot 651);
            // a pack without it cannot express the span, so say so
            let f = self
                .kernels
                .attn_prefill_wp
                .ok_or(GpuError::MissingOp("attn_prefill_wp"))?;
            let (wpp, _g8) = wpos.device_ptr(&self.stream);
            return check(unsafe {
                f(
                    (qp + q_bytes) as *const _,
                    kp as *const _,
                    vp as *const _,
                    sp as *const _,
                    (op + q_bytes) as *mut _,
                    (pp + u_bytes) as *const _,
                    (wpp + u_bytes) as *const _,
                    (slp + u_bytes) as *const _,
                    n_heads as u32,
                    n_kv_heads as u32,
                    head_dim as u32,
                    max_ctx as u32,
                    kv_dim as u32,
                    swa_window as u32,
                    rows as u32,
                    scale,
                    kv_dtype as u32,
                    self.stream_ptr(),
                )
            });
        }
        let f = self
            .kernels
            .attn_prefill
            .ok_or(GpuError::MissingOp("attn_prefill"))?;
        check(unsafe {
            f(
                (qp + q_bytes) as *const _,
                kp as *const _,
                vp as *const _,
                sp as *const _,
                (op + q_bytes) as *mut _,
                (pp + u_bytes) as *const _,
                (slp + u_bytes) as *const _,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                max_ctx as u32,
                kv_dim as u32,
                swa_window as u32,
                rows as u32,
                scale,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }

    // A kernel launcher: the parameter list is the kernel's.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_prefill_f16_rows(
        &self,
        q: &CudaSlice<f32>,
        kc: &CudaSlice<u8>,
        vc: &CudaSlice<u8>,
        sinks: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        win_pos: Option<&CudaSlice<u32>>,
        slots: &CudaSlice<u32>,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_ctx: usize,
        kv_dim: usize,
        swa_window: usize,
        row_off: usize,
        rows: usize,
        scale: f32,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let q_bytes = (row_off * n_heads * head_dim * 4) as u64;
        let u_bytes = (row_off * 4) as u64;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = kc.device_ptr(&self.stream);
        let (vp, _g3) = vc.device_ptr(&self.stream);
        let (sp, _g4) = sinks.device_ptr(&self.stream);
        let (op, _g5) = out.device_ptr_mut(&self.stream);
        let (pp, _g6) = positions.device_ptr(&self.stream);
        let (slp, _g7) = slots.device_ptr(&self.stream);
        // SAFETY: see attn_decode_batch_rows
        if let Some(wpos) = win_pos {
            // window floors from the true positions (slot 652)
            let f = self
                .kernels
                .attn_prefill_f16_wp
                .ok_or(GpuError::MissingOp("attn_prefill_f16_wp"))?;
            let (wpp, _g8) = wpos.device_ptr(&self.stream);
            return check(unsafe {
                f(
                    (qp + q_bytes) as *const _,
                    kp as *const _,
                    vp as *const _,
                    sp as *const _,
                    (op + q_bytes) as *mut _,
                    (pp + u_bytes) as *const _,
                    (wpp + u_bytes) as *const _,
                    (slp + u_bytes) as *const _,
                    n_heads as u32,
                    n_kv_heads as u32,
                    head_dim as u32,
                    max_ctx as u32,
                    kv_dim as u32,
                    swa_window as u32,
                    rows as u32,
                    scale,
                    kv_dtype as u32,
                    self.stream_ptr(),
                )
            });
        }
        let f = self
            .kernels
            .attn_prefill_f16
            .ok_or(GpuError::MissingOp("attn_prefill_f16"))?;
        check(unsafe {
            f(
                (qp + q_bytes) as *const _,
                kp as *const _,
                vp as *const _,
                sp as *const _,
                (op + q_bytes) as *mut _,
                (pp + u_bytes) as *const _,
                (slp + u_bytes) as *const _,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                max_ctx as u32,
                kv_dim as u32,
                swa_window as u32,
                rows as u32,
                scale,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Paged twin of [`Self::attn_decode_batch_rows`] (gpt-oss G2): the same
    /// row-subrange decode over the block pool. q/out/positions/slots are
    /// offset by `row_off` exactly as the dense rows wrapper; `block_tables` is
    /// the full table (indexed by the row's actual slot value, which the offset
    /// slots pointer provides) so it is not offset. Reuses the bitwise-gated
    /// `attn_decode_batch_paged` kernel - addressing already proven vs dense.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode_batch_rows_paged(
        &self,
        q: &CudaSlice<f32>,
        pool_k: &CudaSlice<u8>,
        pool_v: &CudaSlice<u8>,
        sinks: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        slots: Option<&CudaSlice<u32>>,
        block_tables: &CudaSlice<u32>,
        blocks_per_slot: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_dim: usize,
        swa_window: usize,
        row_off: usize,
        rows: usize,
        scale: f32,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .attn_decode_batch_paged
            .ok_or(GpuError::MissingOp("attn_decode_batch_paged"))?;
        let q_bytes = (row_off * n_heads * head_dim * 4) as u64;
        let u_bytes = (row_off * 4) as u64;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = pool_k.device_ptr(&self.stream);
        let (vp, _g3) = pool_v.device_ptr(&self.stream);
        let (sp, _g4) = sinks.device_ptr(&self.stream);
        let (op, _g5) = out.device_ptr_mut(&self.stream);
        let (pp, _g6) = positions.device_ptr(&self.stream);
        let slot_guard = slots.map(|s| s.device_ptr(&self.stream));
        let slp = match &slot_guard {
            Some((p, _)) => (*p + u_bytes) as *const core::ffi::c_void,
            None => std::ptr::null(),
        };
        let (btp, _g7) = block_tables.device_ptr(&self.stream);
        check(unsafe {
            f(
                (qp + q_bytes) as *const _,
                kp as *const _,
                vp as *const _,
                sp as *const _,
                (op + q_bytes) as *mut _,
                (pp + u_bytes) as *const _,
                slp,
                btp as *const _,
                blocks_per_slot as u32,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                kv_dim as u32,
                swa_window as u32,
                rows as u32,
                scale,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Paged twin of [`Self::attn_prefill_f16_rows`] (gpt-oss G2). Same
    /// row-offset math; block table un-offset (slot-indexed). Reuses the
    /// bitwise-gated `attn_prefill_f16_paged` kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_prefill_f16_rows_paged(
        &self,
        q: &CudaSlice<f32>,
        pool_k: &CudaSlice<u8>,
        pool_v: &CudaSlice<u8>,
        sinks: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        win_pos: Option<&CudaSlice<u32>>,
        slots: &CudaSlice<u32>,
        block_tables: &CudaSlice<u32>,
        blocks_per_slot: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_dim: usize,
        swa_window: usize,
        row_off: usize,
        rows: usize,
        scale: f32,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let q_bytes = (row_off * n_heads * head_dim * 4) as u64;
        let u_bytes = (row_off * 4) as u64;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = pool_k.device_ptr(&self.stream);
        let (vp, _g3) = pool_v.device_ptr(&self.stream);
        let (sp, _g4) = sinks.device_ptr(&self.stream);
        let (op, _g5) = out.device_ptr_mut(&self.stream);
        let (pp, _g6) = positions.device_ptr(&self.stream);
        let (slp, _g7) = slots.device_ptr(&self.stream);
        let (btp, _g8) = block_tables.device_ptr(&self.stream);
        if let Some(wpos) = win_pos {
            // window floors from the true positions (slot 653); offset like
            // positions - the two arrays are row-parallel
            let f = self
                .kernels
                .attn_prefill_f16_paged_wp
                .ok_or(GpuError::MissingOp("attn_prefill_f16_paged_wp"))?;
            let (wpp, _g9) = wpos.device_ptr(&self.stream);
            return check(unsafe {
                f(
                    (qp + q_bytes) as *const _,
                    kp as *const _,
                    vp as *const _,
                    sp as *const _,
                    (op + q_bytes) as *mut _,
                    (pp + u_bytes) as *const _,
                    (wpp + u_bytes) as *const _,
                    (slp + u_bytes) as *const _,
                    btp as *const _,
                    blocks_per_slot as u32,
                    n_heads as u32,
                    n_kv_heads as u32,
                    head_dim as u32,
                    kv_dim as u32,
                    swa_window as u32,
                    rows as u32,
                    scale,
                    kv_dtype as u32,
                    self.stream_ptr(),
                )
            });
        }
        let f = self
            .kernels
            .attn_prefill_f16_paged
            .ok_or(GpuError::MissingOp("attn_prefill_f16_paged"))?;
        check(unsafe {
            f(
                (qp + q_bytes) as *const _,
                kp as *const _,
                vp as *const _,
                sp as *const _,
                (op + q_bytes) as *mut _,
                (pp + u_bytes) as *const _,
                (slp + u_bytes) as *const _,
                btp as *const _,
                blocks_per_slot as u32,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                kv_dim as u32,
                swa_window as u32,
                rows as u32,
                scale,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }

    /// FlashDecoding partial pass: grid (n_heads, n_splits), each block runs
    /// online softmax over its KV slice and writes an unnormalized partial into
    /// `out_o` + (max, sum) into `out_ml`. Pair with `attn_combine`.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_partial(
        &self,
        q: &CudaSlice<f32>,
        kc: &CudaSlice<f16>,
        vc: &CudaSlice<f16>,
        out_o: &mut CudaSlice<f32>,
        out_ml: &mut CudaSlice<f32>,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        first_pos: usize,
        n_pos: usize,
        n_splits: usize,
        kv_dim: usize,
        scale: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .attn_decode_partial
            .ok_or(GpuError::MissingOp("attn_decode_partial"))?;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = kc.device_ptr(&self.stream);
        let (vp, _g3) = vc.device_ptr(&self.stream);
        let (op, _g4) = out_o.device_ptr_mut(&self.stream);
        let (mp, _g5) = out_ml.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; scratch sized by caller, head_dim % 32 == 0
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                op as *mut _,
                mp as *mut _,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                first_pos as u32,
                n_pos as u32,
                n_splits as u32,
                kv_dim as u32,
                scale,
                self.stream_ptr(),
            )
        })
    }

    /// FlashDecoding combine pass: merge the `n_splits` partials per head into
    /// `out`, folding the per-head sink into the denominator.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_combine(
        &self,
        in_o: &CudaSlice<f32>,
        in_ml: &CudaSlice<f32>,
        sinks: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n_heads: usize,
        head_dim: usize,
        n_splits: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .attn_decode_combine
            .ok_or(GpuError::MissingOp("attn_decode_combine"))?;
        let (op_in, _g1) = in_o.device_ptr(&self.stream);
        let (mp, _g2) = in_ml.device_ptr(&self.stream);
        let (sp, _g3) = sinks.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; buffers sized by caller
        check(unsafe {
            f(
                op_in as *const _,
                mp as *const _,
                sp as *const _,
                op as *mut _,
                n_heads as u32,
                head_dim as u32,
                n_splits as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Batched FlashDecoding partial pass: grid (n_heads, batch, n_splits). Each
    /// sequence's KV range is split into `n_splits` chunks; each block writes an
    /// unnormalized partial into `out_o` + (m,l) into `out_ml`. Pair with
    /// `attn_combine_batch`. `kc`/`vc` are the fp16 per-sequence caches.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_partial_batch(
        &self,
        q: &CudaSlice<f32>,
        kc: &CudaSlice<u8>,
        vc: &CudaSlice<u8>,
        out_o: &mut CudaSlice<f32>,
        out_ml: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        slots: Option<&CudaSlice<u32>>,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_ctx: usize,
        kv_dim: usize,
        swa_window: usize,
        n_splits: usize,
        batch: usize,
        scale: f32,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .attn_decode_batch_partial
            .ok_or(GpuError::MissingOp("attn_decode_batch_partial"))?;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = kc.device_ptr(&self.stream);
        let (vp, _g3) = vc.device_ptr(&self.stream);
        let (op, _g4) = out_o.device_ptr_mut(&self.stream);
        let (mp, _g5) = out_ml.device_ptr_mut(&self.stream);
        let (pp, _g6) = positions.device_ptr(&self.stream);
        let slot_guard = slots.map(|s| s.device_ptr(&self.stream));
        let slp = match &slot_guard {
            Some((p, _)) => *p as *const core::ffi::c_void,
            None => std::ptr::null(),
        };
        // SAFETY: ABI contract; partial scratch sized [n_heads*batch*n_splits*head_dim]
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                op as *mut _,
                mp as *mut _,
                pp as *const _,
                slp,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                max_ctx as u32,
                kv_dim as u32,
                swa_window as u32,
                n_splits as u32,
                batch as u32,
                scale,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Whether the pack exports the paged FlashDecoding partial (P3b). Callers on
    /// a ≥128-SM die need it to split the paged decode; absent -> fall back.
    pub fn has_attn_partial_batch_paged(&self) -> bool {
        self.kernels.attn_decode_batch_partial_paged.is_some()
    }

    /// Paged FlashDecoding partial (P3b): the split analog of
    /// [`Self::attn_decode_batch_paged`]. Reads K/V from the block pool via block
    /// tables; pair with the unchanged [`Self::attn_combine_batch`]. Bit-exact vs
    /// [`Self::attn_partial_batch`].
    #[allow(clippy::too_many_arguments)]
    pub fn attn_partial_batch_paged(
        &self,
        q: &CudaSlice<f32>,
        pool_k: &CudaSlice<u8>,
        pool_v: &CudaSlice<u8>,
        out_o: &mut CudaSlice<f32>,
        out_ml: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        slots: Option<&CudaSlice<u32>>,
        block_tables: &CudaSlice<u32>,
        blocks_per_slot: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_dim: usize,
        swa_window: usize,
        n_splits: usize,
        batch: usize,
        scale: f32,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .attn_decode_batch_partial_paged
            .ok_or(GpuError::MissingOp("attn_decode_batch_partial_paged"))?;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = pool_k.device_ptr(&self.stream);
        let (vp, _g3) = pool_v.device_ptr(&self.stream);
        let (op, _g4) = out_o.device_ptr_mut(&self.stream);
        let (mp, _g5) = out_ml.device_ptr_mut(&self.stream);
        let (pp, _g6) = positions.device_ptr(&self.stream);
        let slot_guard = slots.map(|s| s.device_ptr(&self.stream));
        let slp = match &slot_guard {
            Some((p, _)) => *p as *const core::ffi::c_void,
            None => std::ptr::null(),
        };
        let (btp, _g7) = block_tables.device_ptr(&self.stream);
        // SAFETY: ABI contract; pools sized [n_blocks, 16, kv_dim] * dtype bytes
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                op as *mut _,
                mp as *mut _,
                pp as *const _,
                slp,
                btp as *const _,
                blocks_per_slot as u32,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                kv_dim as u32,
                swa_window as u32,
                n_splits as u32,
                batch as u32,
                scale,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Wide-batch spec-verify attention partial: one KV walk per PADDED
    /// slot-major chunk of `k1` consecutive rows (per-row causal/window
    /// masks inside). Same (o, m, l) partial layout as the per-row kernels -
    /// pairs with `attn_combine_batch` unchanged. `has_attn_spec_batch_paged`
    /// gates dispatch (older packs lack the slot).
    #[allow(clippy::too_many_arguments)]
    pub fn attn_spec_batch_paged(
        &self,
        q: &CudaSlice<f32>,
        pool_k: &CudaSlice<u8>,
        pool_v: &CudaSlice<u8>,
        out_o: &mut CudaSlice<f32>,
        out_ml: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        slots: Option<&CudaSlice<u32>>,
        block_tables: &CudaSlice<u32>,
        blocks_per_slot: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_dim: usize,
        swa_window: usize,
        n_splits: usize,
        rows: usize,
        k1: usize,
        scale: f32,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .attn_spec_batch_paged
            .ok_or(GpuError::MissingOp("attn_spec_batch_paged"))?;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = pool_k.device_ptr(&self.stream);
        let (vp, _g3) = pool_v.device_ptr(&self.stream);
        let (op, _g4) = out_o.device_ptr_mut(&self.stream);
        let (mp, _g5) = out_ml.device_ptr_mut(&self.stream);
        let (pp, _g6) = positions.device_ptr(&self.stream);
        let slot_guard = slots.map(|s| s.device_ptr(&self.stream));
        let slp = match &slot_guard {
            Some((p, _)) => *p as *const core::ffi::c_void,
            None => std::ptr::null(),
        };
        let (btp, _g7) = block_tables.device_ptr(&self.stream);
        // SAFETY: ABI contract; pools sized [n_blocks, 16, kv_dim] * dtype bytes
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                op as *mut _,
                mp as *mut _,
                pp as *const _,
                slp,
                btp as *const _,
                blocks_per_slot as u32,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                kv_dim as u32,
                swa_window as u32,
                n_splits as u32,
                rows as u32,
                k1 as u32,
                scale,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_attn_spec_batch_paged(&self) -> bool {
        self.kernels.attn_spec_batch_paged.is_some()
    }

    /// Spec-verify FIN: the FA route at one split with in-kernel finalize -
    /// `out` receives the COMBINED batch-major rows (pf_attn layout),
    /// bit-identical to the partial walk + -inf-sink combine. Returns
    /// Ok(false) when the FA geometry can't engage (the pack's -2) so the
    /// caller keeps the partial+combine chain for that layer.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_spec_batch_fin(
        &self,
        q: &CudaSlice<f32>,
        pool_k: &CudaSlice<u8>,
        pool_v: &CudaSlice<u8>,
        out: &mut CudaSlice<f32>,
        ml_scratch: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        slots: Option<&CudaSlice<u32>>,
        block_tables: &CudaSlice<u32>,
        blocks_per_slot: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_dim: usize,
        swa_window: usize,
        rows: usize,
        k1: usize,
        scale: f32,
        kv_dtype: KvDtype,
    ) -> Result<bool, GpuError> {
        let f = self
            .kernels
            .attn_spec_batch_fin
            .ok_or(GpuError::MissingOp("attn_spec_batch_fin"))?;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = pool_k.device_ptr(&self.stream);
        let (vp, _g3) = pool_v.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        let (mp, _g5) = ml_scratch.device_ptr_mut(&self.stream);
        let (pp, _g6) = positions.device_ptr(&self.stream);
        let slot_guard = slots.map(|s| s.device_ptr(&self.stream));
        let slp = match &slot_guard {
            Some((p, _)) => *p as *const core::ffi::c_void,
            None => std::ptr::null(),
        };
        let (btp, _g7) = block_tables.device_ptr(&self.stream);
        // SAFETY: ABI contract; pools sized [n_blocks, 16, kv_dim] * dtype bytes
        let rc = unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                op as *mut _,
                mp as *mut _,
                pp as *const _,
                slp,
                btp as *const _,
                blocks_per_slot as u32,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                kv_dim as u32,
                swa_window as u32,
                rows as u32,
                k1 as u32,
                scale,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        };
        if rc == -2 {
            return Ok(false);
        }
        check(rc)?;
        Ok(true)
    }

    pub fn has_attn_spec_batch_fin(&self) -> bool {
        self.kernels.attn_spec_batch_fin.is_some()
    }

    /// fin twin with the in-kernel wo-in row quantize (P53, slot 422):
    /// finalized rows land as e4m3 in `out_q` `[rows, n_heads*head_dim]`
    /// with f32 per-row scales in `out_rs` - bit-identical to fin +
    /// `quantize_e4m3_row`. Ok(false) = geometry not covered; run the f32
    /// fin + standalone quantize instead.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_spec_batch_fin_e4(
        &self,
        q: &CudaSlice<f32>,
        pool_k: &CudaSlice<u8>,
        pool_v: &CudaSlice<u8>,
        out_q: &mut CudaSlice<i8>,
        out_rs: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        slots: Option<&CudaSlice<u32>>,
        block_tables: &CudaSlice<u32>,
        blocks_per_slot: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_dim: usize,
        swa_window: usize,
        rows: usize,
        k1: usize,
        scale: f32,
        kv_dtype: KvDtype,
    ) -> Result<bool, GpuError> {
        let f = self
            .kernels
            .attn_spec_batch_fin_e4
            .ok_or(GpuError::MissingOp("attn_spec_batch_fin_e4"))?;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = pool_k.device_ptr(&self.stream);
        let (vp, _g3) = pool_v.device_ptr(&self.stream);
        let (op, _g4) = out_q.device_ptr_mut(&self.stream);
        let (rp, _g5) = out_rs.device_ptr_mut(&self.stream);
        let (pp, _g6) = positions.device_ptr(&self.stream);
        let slot_guard = slots.map(|s| s.device_ptr(&self.stream));
        let slp = match &slot_guard {
            Some((p, _)) => *p as *const core::ffi::c_void,
            None => std::ptr::null(),
        };
        let (btp, _g7) = block_tables.device_ptr(&self.stream);
        // SAFETY: ABI contract; pools sized [n_blocks, 16, kv_dim] * dtype bytes
        let rc = unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                op as *mut _,
                rp as *mut _,
                pp as *const _,
                slp,
                btp as *const _,
                blocks_per_slot as u32,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                kv_dim as u32,
                swa_window as u32,
                rows as u32,
                k1 as u32,
                scale,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        };
        if rc == -2 {
            return Ok(false);
        }
        check(rc)?;
        Ok(true)
    }

    pub fn has_attn_spec_batch_fin_e4(&self) -> bool {
        self.kernels.attn_spec_batch_fin_e4.is_some()
    }

    /// P54 (slot 425): fin twin storing the finalized rows as
    /// e4m3 at STATIC scale 1.0 directly into the wo-in quantized plane
    /// (`out_q` = pf_e4q) - the standalone `quantize_e4m3_row` launch
    /// disappears and the wo GEMM's xrs must be a ONES vector (pf_fae4rs).
    /// `ml` is fin's dead out_ml slot (scratch, untouched under fin).
    /// Ok(false) = same accept envelope as `attn_spec_batch_fin` refusing
    /// - caller keeps the f32 fin + quantize chain.
    // A kernel launcher: the parameter list is the kernel's.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_spec_batch_fin_e4s(
        &self,
        q: &CudaSlice<f32>,
        pool_k: &CudaSlice<u8>,
        pool_v: &CudaSlice<u8>,
        out_q: &mut CudaSlice<i8>,
        ml: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        slots: Option<&CudaSlice<u32>>,
        block_tables: &CudaSlice<u32>,
        blocks_per_slot: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_dim: usize,
        swa_window: usize,
        rows: usize,
        k1: usize,
        scale: f32,
        kv_dtype: KvDtype,
    ) -> Result<bool, GpuError> {
        let f = self
            .kernels
            .attn_spec_batch_fin_e4s
            .ok_or(GpuError::MissingOp("attn_spec_batch_fin_e4s"))?;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = pool_k.device_ptr(&self.stream);
        let (vp, _g3) = pool_v.device_ptr(&self.stream);
        let (op, _g4) = out_q.device_ptr_mut(&self.stream);
        let (mp, _g5) = ml.device_ptr_mut(&self.stream);
        let (pp, _g6) = positions.device_ptr(&self.stream);
        let slot_guard = slots.map(|s| s.device_ptr(&self.stream));
        let slp = match &slot_guard {
            Some((p, _)) => *p as *const core::ffi::c_void,
            None => std::ptr::null(),
        };
        let (btp, _g7) = block_tables.device_ptr(&self.stream);
        // SAFETY: ABI contract (slot 425); pools sized [n_blocks, 16, kv_dim]
        let rc = unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                op as *mut _,
                mp as *mut _,
                pp as *const _,
                slp,
                btp as *const _,
                blocks_per_slot as u32,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                kv_dim as u32,
                swa_window as u32,
                rows as u32,
                k1 as u32,
                scale,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        };
        if rc == -2 {
            return Ok(false);
        }
        check(rc)?;
        Ok(true)
    }

    pub fn has_attn_spec_batch_fin_e4s(&self) -> bool {
        self.kernels.attn_spec_batch_fin_e4s.is_some()
    }

    /// Spec-verify LCO: the krs spec-FA arms with in-kernel
    /// last-CTA-out combine - `out` receives the COMBINED batch-major rows
    /// (pf_attn layout), bit-identical to the partial walk + -inf-sink
    /// combine, and the separate combine launch disappears. `ao`/`aml` stay
    /// partial scratch the merge reads; `tickets` is the per-(kvh, chunk)
    /// arrival counter (wraps in-kernel - zero once at alloc, never again).
    /// Returns Ok(false) when the geometry isn't covered (the pack's -2) so
    /// the caller keeps the partial+combine chain for that layer.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_spec_lco_paged(
        &self,
        q: &CudaSlice<f32>,
        pool_k: &CudaSlice<u8>,
        pool_v: &CudaSlice<u8>,
        ao: &mut CudaSlice<f32>,
        aml: &mut CudaSlice<f32>,
        sinks: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        tickets: &mut CudaSlice<u32>,
        positions: &CudaSlice<u32>,
        slots: Option<&CudaSlice<u32>>,
        block_tables: &CudaSlice<u32>,
        blocks_per_slot: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_dim: usize,
        swa_window: usize,
        n_splits: usize,
        rows: usize,
        k1: usize,
        scale: f32,
        kv_dtype: KvDtype,
    ) -> Result<bool, GpuError> {
        let f = self
            .kernels
            .attn_spec_lco_paged
            .ok_or(GpuError::MissingOp("attn_spec_lco_paged"))?;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = pool_k.device_ptr(&self.stream);
        let (vp, _g3) = pool_v.device_ptr(&self.stream);
        let (aop, _g4) = ao.device_ptr_mut(&self.stream);
        let (mlp, _g5) = aml.device_ptr_mut(&self.stream);
        let (sp, _g6) = sinks.device_ptr(&self.stream);
        let (op, _g7) = out.device_ptr_mut(&self.stream);
        let (tp, _g8) = tickets.device_ptr_mut(&self.stream);
        let (pp, _g9) = positions.device_ptr(&self.stream);
        let slot_guard = slots.map(|s| s.device_ptr(&self.stream));
        let slp = match &slot_guard {
            Some((p, _)) => *p as *const core::ffi::c_void,
            None => std::ptr::null(),
        };
        let (btp, _g10) = block_tables.device_ptr(&self.stream);
        // SAFETY: ABI contract; pools sized [n_blocks, 16, kv_dim] * dtype bytes
        let rc = unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                aop as *mut _,
                mlp as *mut _,
                sp as *const _,
                op as *mut _,
                tp as *mut _,
                pp as *const _,
                slp,
                btp as *const _,
                blocks_per_slot as u32,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                kv_dim as u32,
                swa_window as u32,
                n_splits as u32,
                rows as u32,
                k1 as u32,
                scale,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        };
        if rc == -2 {
            return Ok(false);
        }
        check(rc)?;
        Ok(true)
    }

    pub fn has_attn_spec_lco_paged(&self) -> bool {
        self.kernels.attn_spec_lco_paged.is_some()
    }

    /// Batched FlashDecoding combine: merge the `n_splits` partials per (head,
    /// sequence) into `out` [batch, n_heads, head_dim], folding the per-head sink.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_combine_batch(
        &self,
        in_o: &CudaSlice<f32>,
        in_ml: &CudaSlice<f32>,
        sinks: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n_heads: usize,
        head_dim: usize,
        n_splits: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .attn_decode_batch_combine
            .ok_or(GpuError::MissingOp("attn_decode_batch_combine"))?;
        let (op_in, _g1) = in_o.device_ptr(&self.stream);
        let (mp, _g2) = in_ml.device_ptr(&self.stream);
        let (sp, _g3) = sinks.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; buffers sized by caller
        check(unsafe {
            f(
                op_in as *const _,
                mp as *const _,
                sp as *const _,
                op as *mut _,
                n_heads as u32,
                head_dim as u32,
                n_splits as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// x[..n] += w * y[..n].
    pub fn scale_add(
        &self,
        x: &mut CudaSlice<f32>,
        y: &CudaSlice<f32>,
        w: f32,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .scale_add_f32
            .ok_or(GpuError::MissingOp("scale_add"))?;
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (yp, _g2) = y.device_ptr(&self.stream);
        // SAFETY: ABI contract
        check(unsafe { f(xp as *mut _, yp as *const _, w, n as u32, self.stream_ptr()) })
    }

    /// Attention streams: true when every a16 entry (296..303) is
    /// present - the all-or-nothing capability the forward gate keys on.
    pub fn has_attn16(&self) -> bool {
        self.kernels.qkv_norm_rope_batch5.is_some()
            && self.kernels.attn_prefill_f16_paged2.is_some()
            && self.kernels.attn_spec_batch_paged2.is_some()
            && self.kernels.attn_decode_batch_paged2.is_some()
            && self.kernels.attn_decode_batch_partial_paged2.is_some()
            && self.kernels.attn_decode_batch_combine2.is_some()
            && self.kernels.quantize_e4m3_i16.is_some()
            && self.kernels.quantize_e4m3_row_i16.is_some()
    }

    /// a16 twin of [`Self::attn_prefill_f16_rows_paged`]: q and
    /// out are f16 planes held in the f32-typed scratch (2-byte elements -
    /// the row-offset math halves accordingly).
    #[allow(clippy::too_many_arguments)]
    pub fn attn_prefill_f16_rows_paged_a16(
        &self,
        q: &CudaSlice<f32>,
        pool_k: &CudaSlice<u8>,
        pool_v: &CudaSlice<u8>,
        sinks: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        win_pos: Option<&CudaSlice<u32>>,
        slots: &CudaSlice<u32>,
        block_tables: &CudaSlice<u32>,
        blocks_per_slot: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_dim: usize,
        swa_window: usize,
        row_off: usize,
        rows: usize,
        scale: f32,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let q_bytes = (row_off * n_heads * head_dim * 2) as u64;
        let u_bytes = (row_off * 4) as u64;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = pool_k.device_ptr(&self.stream);
        let (vp, _g3) = pool_v.device_ptr(&self.stream);
        let (sp, _g4) = sinks.device_ptr(&self.stream);
        let (op, _g5) = out.device_ptr_mut(&self.stream);
        let (pp, _g6) = positions.device_ptr(&self.stream);
        let (slp, _g7) = slots.device_ptr(&self.stream);
        let (btp, _g8) = block_tables.device_ptr(&self.stream);
        // SAFETY: ABI contract; q/out hold f16 in the f32-typed scratch
        if let Some(wpos) = win_pos {
            // window floors from the true positions (slot 654)
            let f = self
                .kernels
                .attn_prefill_f16_paged2_wp
                .ok_or(GpuError::MissingOp("attn_prefill_f16_paged2_wp"))?;
            let (wpp, _g9) = wpos.device_ptr(&self.stream);
            return check(unsafe {
                f(
                    (qp + q_bytes) as *const _,
                    kp as *const _,
                    vp as *const _,
                    sp as *const _,
                    (op + q_bytes) as *mut _,
                    (pp + u_bytes) as *const _,
                    (wpp + u_bytes) as *const _,
                    (slp + u_bytes) as *const _,
                    btp as *const _,
                    blocks_per_slot as u32,
                    n_heads as u32,
                    n_kv_heads as u32,
                    head_dim as u32,
                    kv_dim as u32,
                    swa_window as u32,
                    rows as u32,
                    scale,
                    kv_dtype as u32,
                    1u32,
                    self.stream_ptr(),
                )
            });
        }
        let f = self
            .kernels
            .attn_prefill_f16_paged2
            .ok_or(GpuError::MissingOp("attn_prefill_f16_paged2"))?;
        check(unsafe {
            f(
                (qp + q_bytes) as *const _,
                kp as *const _,
                vp as *const _,
                sp as *const _,
                (op + q_bytes) as *mut _,
                (pp + u_bytes) as *const _,
                (slp + u_bytes) as *const _,
                btp as *const _,
                blocks_per_slot as u32,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                kv_dim as u32,
                swa_window as u32,
                rows as u32,
                scale,
                kv_dtype as u32,
                1u32,
                self.stream_ptr(),
            )
        })
    }

    /// a16 twin of [`Self::attn_spec_batch_paged`]: q is an f16
    /// plane; the (o, ml) partials stay f32.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_spec_batch_paged_a16(
        &self,
        q: &CudaSlice<f32>,
        pool_k: &CudaSlice<u8>,
        pool_v: &CudaSlice<u8>,
        out_o: &mut CudaSlice<f32>,
        out_ml: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        slots: Option<&CudaSlice<u32>>,
        block_tables: &CudaSlice<u32>,
        blocks_per_slot: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_dim: usize,
        swa_window: usize,
        n_splits: usize,
        rows: usize,
        k1: usize,
        scale: f32,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .attn_spec_batch_paged2
            .ok_or(GpuError::MissingOp("attn_spec_batch_paged2"))?;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = pool_k.device_ptr(&self.stream);
        let (vp, _g3) = pool_v.device_ptr(&self.stream);
        let (op, _g4) = out_o.device_ptr_mut(&self.stream);
        let (mp, _g5) = out_ml.device_ptr_mut(&self.stream);
        let (pp, _g6) = positions.device_ptr(&self.stream);
        let slot_guard = slots.map(|s| s.device_ptr(&self.stream));
        let slp = match &slot_guard {
            Some((p, _)) => *p as *const core::ffi::c_void,
            None => std::ptr::null(),
        };
        let (btp, _g7) = block_tables.device_ptr(&self.stream);
        // SAFETY: ABI contract; q holds f16 in the f32-typed scratch
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                op as *mut _,
                mp as *mut _,
                pp as *const _,
                slp,
                btp as *const _,
                blocks_per_slot as u32,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                kv_dim as u32,
                swa_window as u32,
                n_splits as u32,
                rows as u32,
                k1 as u32,
                scale,
                kv_dtype as u32,
                1u32,
                self.stream_ptr(),
            )
        })
    }

    /// a16 twin of [`Self::attn_decode_batch_paged`]: q and out
    /// are f16 planes.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode_batch_paged_a16(
        &self,
        q: &CudaSlice<f32>,
        pool_k: &CudaSlice<u8>,
        pool_v: &CudaSlice<u8>,
        sinks: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        slots: Option<&CudaSlice<u32>>,
        block_tables: &CudaSlice<u32>,
        blocks_per_slot: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_dim: usize,
        swa_window: usize,
        batch: usize,
        scale: f32,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .attn_decode_batch_paged2
            .ok_or(GpuError::MissingOp("attn_decode_batch_paged2"))?;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = pool_k.device_ptr(&self.stream);
        let (vp, _g3) = pool_v.device_ptr(&self.stream);
        let (sp, _g4) = sinks.device_ptr(&self.stream);
        let (op, _g5) = out.device_ptr_mut(&self.stream);
        let (pp, _g6) = positions.device_ptr(&self.stream);
        let slot_guard = slots.map(|s| s.device_ptr(&self.stream));
        let slp = match &slot_guard {
            Some((p, _)) => *p as *const core::ffi::c_void,
            None => std::ptr::null(),
        };
        let (btp, _g7) = block_tables.device_ptr(&self.stream);
        // SAFETY: ABI contract; q/out hold f16 in the f32-typed scratch
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                sp as *const _,
                op as *mut _,
                pp as *const _,
                slp,
                btp as *const _,
                blocks_per_slot as u32,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                kv_dim as u32,
                swa_window as u32,
                batch as u32,
                scale,
                kv_dtype as u32,
                1u32,
                self.stream_ptr(),
            )
        })
    }

    /// a16 twin of [`Self::attn_partial_batch_paged`]: q is an
    /// f16 plane; partials stay f32.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_partial_batch_paged_a16(
        &self,
        q: &CudaSlice<f32>,
        pool_k: &CudaSlice<u8>,
        pool_v: &CudaSlice<u8>,
        out_o: &mut CudaSlice<f32>,
        out_ml: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        slots: Option<&CudaSlice<u32>>,
        block_tables: &CudaSlice<u32>,
        blocks_per_slot: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_dim: usize,
        swa_window: usize,
        n_splits: usize,
        batch: usize,
        scale: f32,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .attn_decode_batch_partial_paged2
            .ok_or(GpuError::MissingOp("attn_decode_batch_partial_paged2"))?;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = pool_k.device_ptr(&self.stream);
        let (vp, _g3) = pool_v.device_ptr(&self.stream);
        let (op, _g4) = out_o.device_ptr_mut(&self.stream);
        let (mp, _g5) = out_ml.device_ptr_mut(&self.stream);
        let (pp, _g6) = positions.device_ptr(&self.stream);
        let slot_guard = slots.map(|s| s.device_ptr(&self.stream));
        let slp = match &slot_guard {
            Some((p, _)) => *p as *const core::ffi::c_void,
            None => std::ptr::null(),
        };
        let (btp, _g7) = block_tables.device_ptr(&self.stream);
        // SAFETY: ABI contract; q holds f16 in the f32-typed scratch
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                op as *mut _,
                mp as *mut _,
                pp as *const _,
                slp,
                btp as *const _,
                blocks_per_slot as u32,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                kv_dim as u32,
                swa_window as u32,
                n_splits as u32,
                batch as u32,
                scale,
                kv_dtype as u32,
                1u32,
                self.stream_ptr(),
            )
        })
    }

    /// o16 twin of [`Self::attn_combine_batch`]: the final plane
    /// is f16; the (o, m, l) partials stay f32.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_combine_batch_o16(
        &self,
        in_o: &CudaSlice<f32>,
        in_ml: &CudaSlice<f32>,
        sinks: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n_heads: usize,
        head_dim: usize,
        n_splits: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .attn_decode_batch_combine2
            .ok_or(GpuError::MissingOp("attn_decode_batch_combine2"))?;
        let (op_in, _g1) = in_o.device_ptr(&self.stream);
        let (mp, _g2) = in_ml.device_ptr(&self.stream);
        let (sp, _g3) = sinks.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; out holds f16 in the f32-typed scratch
        check(unsafe {
            f(
                op_in as *const _,
                mp as *const _,
                sp as *const _,
                op as *mut _,
                n_heads as u32,
                head_dim as u32,
                n_splits as u32,
                batch as u32,
                1u32,
                self.stream_ptr(),
            )
        })
    }
}
