//! MXFP4 MoE (grouped/sorted/mmq/dp4a) + interleaved Q8 GEMV primitives.

use super::error::*;
use super::*;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use paddock_models::ggml_type::GgmlType;

impl GpuExecutor {
    /// Build the reverse routing map slot_of[b][e] (255 = not selected) from the
    /// per-token expert ids - the index the grouped MoE kernels iterate over.
    pub fn moe_slot_map(
        &self,
        idx: &CudaSlice<u32>,
        slot_of: &mut CudaSlice<u8>,
        n_active: usize,
        n_expert: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .moe_slot_map
            .ok_or(GpuError::MissingOp("moe_slot_map"))?;
        let (ip, _g1) = idx.device_ptr(&self.stream);
        let (sp, _g2) = slot_of.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; idx [batch, n_active], slot_of [batch, n_expert]
        check(unsafe {
            f(
                ip as *const _,
                sp as *mut _,
                n_active as u32,
                n_expert as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Grouped MoE gate+up+swiglu: each expert row dequanted once, reused across
    /// its tokens (weight-amortized). `out` is [batch, n_active, ff].
    #[allow(clippy::too_many_arguments)]
    pub fn mxfp4_moe_gate_up_grouped(
        &self,
        gate_w: &RepackedMxfp4,
        gate_bias: &CudaSlice<f32>,
        up_w: &RepackedMxfp4,
        up_bias: &CudaSlice<f32>,
        slot_of: &CudaSlice<u8>,
        x: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        in_dim: usize,
        ff: usize,
        n_expert: usize,
        n_active: usize,
        batch: usize,
        alpha: f32,
        limit: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .mxfp4_moe_gate_up_grouped
            .ok_or(GpuError::MissingOp("mxfp4_moe_gate_up_grouped"))?;
        let (gdp, _g1) = gate_w.data.device_ptr(&self.stream);
        let (gsp, _g1s) = gate_w.scale.device_ptr(&self.stream);
        let (gbp, _g2) = gate_bias.device_ptr(&self.stream);
        let (udp, _g3) = up_w.data.device_ptr(&self.stream);
        let (usp, _g3s) = up_w.scale.device_ptr(&self.stream);
        let (ubp, _g4) = up_bias.device_ptr(&self.stream);
        let (sp, _g5) = slot_of.device_ptr(&self.stream);
        let (xp, _g6) = x.device_ptr(&self.stream);
        let (op, _g7) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract
        check(unsafe {
            f(
                gdp as *const _,
                gsp as *const _,
                gbp as *const _,
                udp as *const _,
                usp as *const _,
                ubp as *const _,
                sp as *const _,
                xp as *const _,
                op as *mut _,
                in_dim as u32,
                ff as u32,
                n_expert as u32,
                n_active as u32,
                batch as u32,
                alpha,
                limit,
                self.stream_ptr(),
            )
        })
    }

    /// Tiled grouped MoE gate+up+swiglu (SGEMM shape) - same inputs/output as
    /// `mxfp4_moe_gate_up_grouped`, higher arithmetic intensity.
    #[allow(clippy::too_many_arguments)]
    pub fn mxfp4_moe_gate_up_gemm(
        &self,
        gate_w: &QuantTensor,
        gate_bias: &CudaSlice<f32>,
        up_w: &QuantTensor,
        up_bias: &CudaSlice<f32>,
        slot_of: &CudaSlice<u8>,
        x: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        in_dim: usize,
        ff: usize,
        n_expert: usize,
        n_active: usize,
        batch: usize,
        alpha: f32,
        limit: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .mxfp4_moe_gate_up_gemm
            .ok_or(GpuError::MissingOp("mxfp4_moe_gate_up_gemm"))?;
        let (gwp, _g1) = gate_w.bytes.device_ptr(&self.stream);
        let (gbp, _g2) = gate_bias.device_ptr(&self.stream);
        let (uwp, _g3) = up_w.bytes.device_ptr(&self.stream);
        let (ubp, _g4) = up_bias.device_ptr(&self.stream);
        let (sp, _g5) = slot_of.device_ptr(&self.stream);
        let (xp, _g6) = x.device_ptr(&self.stream);
        let (op, _g7) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract
        check(unsafe {
            f(
                gwp as *const _,
                gbp as *const _,
                uwp as *const _,
                ubp as *const _,
                sp as *const _,
                xp as *const _,
                op as *mut _,
                in_dim as u32,
                ff as u32,
                n_expert as u32,
                n_active as u32,
                batch as u32,
                alpha,
                limit,
                self.stream_ptr(),
            )
        })
    }

    /// moe_align: group token-expert pairs into contiguous BM-padded blocks for
    /// the sorted GEMM. `block_expert` is [max_blocks], the sorted arrays
    /// [max_blocks * BM].
    #[allow(clippy::too_many_arguments)]
    pub fn moe_align(
        &self,
        idx: &CudaSlice<u32>,
        sorted_row: &mut CudaSlice<u32>,
        sorted_slot: &mut CudaSlice<u32>,
        block_expert: &mut CudaSlice<u32>,
        rows: usize,
        n_active: usize,
        n_expert: usize,
        max_blocks: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .moe_align
            .ok_or(GpuError::MissingOp("moe_align"))?;
        let (ip, _g1) = idx.device_ptr(&self.stream);
        let (rp, _g2) = sorted_row.device_ptr_mut(&self.stream);
        let (sp, _g3) = sorted_slot.device_ptr_mut(&self.stream);
        let (bp, _g4) = block_expert.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract
        check(unsafe {
            f(
                ip as *const _,
                rp as *mut _,
                sp as *mut _,
                bp as *mut _,
                rows as u32,
                n_active as u32,
                n_expert as u32,
                max_blocks as u32,
                self.stream_ptr(),
            )
        })
    }

    /// True when the pack carries the block-tile `moe_align` (slot 474).
    pub fn has_moe_align_bm(&self) -> bool {
        self.kernels.moe_align_bm.is_some()
    }

    /// `moe_align` with a caller-chosen power-of-two block tile (bs64 path).
    #[allow(clippy::too_many_arguments)]
    pub fn moe_align_bm(
        &self,
        idx: &CudaSlice<u32>,
        sorted_row: &mut CudaSlice<u32>,
        sorted_slot: &mut CudaSlice<u32>,
        block_expert: &mut CudaSlice<u32>,
        rows: usize,
        n_active: usize,
        n_expert: usize,
        bm: usize,
        max_blocks: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .moe_align_bm
            .ok_or(GpuError::MissingOp("moe_align_bm"))?;
        let (ip, _g1) = idx.device_ptr(&self.stream);
        let (rp, _g2) = sorted_row.device_ptr_mut(&self.stream);
        let (sp, _g3) = sorted_slot.device_ptr_mut(&self.stream);
        let (bp, _g4) = block_expert.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; sorted arrays sized >= max_blocks * bm
        check(unsafe {
            f(
                ip as *const _,
                rp as *mut _,
                sp as *mut _,
                bp as *mut _,
                rows as u32,
                n_active as u32,
                n_expert as u32,
                bm as u32,
                max_blocks as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Sorted tiled MoE gate+up+swiglu -> `fused_sorted` [max_blocks*BM, ff].
    #[allow(clippy::too_many_arguments)]
    pub fn mxfp4_moe_gate_up_gemm_sorted(
        &self,
        gate_w: &RepackedMxfp4,
        gate_bias: &CudaSlice<f32>,
        up_w: &RepackedMxfp4,
        up_bias: &CudaSlice<f32>,
        sorted_row: &CudaSlice<u32>,
        block_expert: &CudaSlice<u32>,
        x: &CudaSlice<f32>,
        fused_sorted: &mut CudaSlice<f32>,
        in_dim: usize,
        ff: usize,
        max_blocks: usize,
        alpha: f32,
        limit: f32,
        use_tc: bool,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .mxfp4_moe_gate_up_gemm_sorted
            .ok_or(GpuError::MissingOp("mxfp4_moe_gate_up_gemm_sorted"))?;
        let (gdp, _g1) = gate_w.data.device_ptr(&self.stream);
        let (gsp, _g1s) = gate_w.scale.device_ptr(&self.stream);
        let (gbp, _g2) = gate_bias.device_ptr(&self.stream);
        let (udp, _g3) = up_w.data.device_ptr(&self.stream);
        let (usp, _g3s) = up_w.scale.device_ptr(&self.stream);
        let (ubp, _g4) = up_bias.device_ptr(&self.stream);
        let (srp, _g5) = sorted_row.device_ptr(&self.stream);
        let (bep, _g6) = block_expert.device_ptr(&self.stream);
        let (xp, _g7) = x.device_ptr(&self.stream);
        let (op, _g8) = fused_sorted.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract
        check(unsafe {
            f(
                gdp as *const _,
                gsp as *const _,
                gbp as *const _,
                udp as *const _,
                usp as *const _,
                ubp as *const _,
                srp as *const _,
                bep as *const _,
                xp as *const _,
                op as *mut _,
                in_dim as u32,
                ff as u32,
                max_blocks as u32,
                alpha,
                limit,
                use_tc as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Sorted tiled MoE down + weighted mix + residual add. `residual` [rows, embd]
    /// must already hold the post-attention hidden state (the mix accumulates on top).
    #[allow(clippy::too_many_arguments)]
    pub fn mxfp4_moe_down_gemm_sorted(
        &self,
        down_w: &RepackedMxfp4,
        down_bias: &CudaSlice<f32>,
        sorted_row: &CudaSlice<u32>,
        sorted_slot: &CudaSlice<u32>,
        block_expert: &CudaSlice<u32>,
        topk_w: &CudaSlice<f32>,
        fused_sorted: &CudaSlice<f32>,
        residual: &mut CudaSlice<f32>,
        ff: usize,
        embd: usize,
        n_active: usize,
        max_blocks: usize,
        use_tc: bool,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .mxfp4_moe_down_gemm_sorted
            .ok_or(GpuError::MissingOp("mxfp4_moe_down_gemm_sorted"))?;
        let (ddp, _g1) = down_w.data.device_ptr(&self.stream);
        let (dsp, _g1s) = down_w.scale.device_ptr(&self.stream);
        let (dbp, _g2) = down_bias.device_ptr(&self.stream);
        let (srp, _g3) = sorted_row.device_ptr(&self.stream);
        let (slp, _g4) = sorted_slot.device_ptr(&self.stream);
        let (bep, _g5) = block_expert.device_ptr(&self.stream);
        let (wp, _g6) = topk_w.device_ptr(&self.stream);
        let (fp, _g7) = fused_sorted.device_ptr(&self.stream);
        let (rp, _g8) = residual.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract
        check(unsafe {
            f(
                ddp as *const _,
                dsp as *const _,
                dbp as *const _,
                srp as *const _,
                slp as *const _,
                bep as *const _,
                wp as *const _,
                fp as *const _,
                rp as *mut _,
                ff as u32,
                embd as u32,
                n_active as u32,
                max_blocks as u32,
                use_tc as u32,
                self.stream_ptr(),
            )
        })
    }

    /// int8 tensor-core sorted MoE gate+up+swiglu (mmq class): activations
    /// pre-quantized in the strided [`Self::quantize_q8`] layout, gathered per
    /// sorted_row; the swiglu output comes back already quantized (`fq`/`fs`,
    /// the down GEMM's direct input - bit-identical to swiglu + quantize run
    /// separately). PAD rows come out as exact zeros. Same numeric class as
    /// [`Self::q8_0_gemm_mmq`], not the f32/f16 `_gemm_sorted` pair.
    #[allow(clippy::too_many_arguments)]
    pub fn mxfp4_moe_gate_up_mmq(
        &self,
        gate_w: &RepackedMxfp4,
        gate_bias: &CudaSlice<f32>,
        up_w: &RepackedMxfp4,
        up_bias: &CudaSlice<f32>,
        sorted_row: &CudaSlice<u32>,
        block_expert: &CudaSlice<u32>,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        fq: &mut CudaSlice<i8>,
        fs: &mut CudaSlice<f32>,
        in_dim: usize,
        ff: usize,
        max_blocks: usize,
        alpha: f32,
        limit: f32,
        // SwiGLU up-term: 1.0 = gpt-oss (u+1); 0.0 = qwen plain silu(g)*u.
        up_add: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .mxfp4_moe_gate_up_mmq
            .ok_or(GpuError::MissingOp("mxfp4_moe_gate_up_mmq"))?;
        let (gdp, _g1) = gate_w.data.device_ptr(&self.stream);
        let (gsp, _g1s) = gate_w.scale.device_ptr(&self.stream);
        let (gbp, _g2) = gate_bias.device_ptr(&self.stream);
        let (udp, _g3) = up_w.data.device_ptr(&self.stream);
        let (usp, _g3s) = up_w.scale.device_ptr(&self.stream);
        let (ubp, _g4) = up_bias.device_ptr(&self.stream);
        let (srp, _g5) = sorted_row.device_ptr(&self.stream);
        let (bep, _g6) = block_expert.device_ptr(&self.stream);
        let (xqp, _g7) = xq.device_ptr(&self.stream);
        let (xsp, _g8) = xs.device_ptr(&self.stream);
        let (fqp, _g9) = fq.device_ptr_mut(&self.stream);
        let (fsp, _g10) = fs.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract
        check(unsafe {
            f(
                gdp as *const _,
                gsp as *const _,
                gbp as *const _,
                udp as *const _,
                usp as *const _,
                ubp as *const _,
                srp as *const _,
                bep as *const _,
                xqp as *const _,
                xsp as *const _,
                fqp as *mut _,
                fsp as *mut _,
                in_dim as u32,
                ff as u32,
                max_blocks as u32,
                alpha,
                limit,
                up_add,
                self.stream_ptr(),
            )
        })
    }

    /// int8 tensor-core sorted MoE down (mmq class): activation rows are the
    /// gate_up-emitted quantized swiglu output (strided, row = blk*32+i).
    /// Writes topk-weighted per-(token, slot) PARTIALS into `part`
    /// ([rows, n_active, embd], no atomics - each cell has one writer); fold
    /// with [`Self::moe_slot_combine`] for the deterministic residual add.
    #[allow(clippy::too_many_arguments)]
    pub fn mxfp4_moe_down_mmq(
        &self,
        down_w: &RepackedMxfp4,
        down_bias: &CudaSlice<f32>,
        sorted_row: &CudaSlice<u32>,
        sorted_slot: &CudaSlice<u32>,
        block_expert: &CudaSlice<u32>,
        topk_w: &CudaSlice<f32>,
        fq: &CudaSlice<i8>,
        fs: &CudaSlice<f32>,
        part: &mut CudaSlice<f32>,
        ff: usize,
        embd: usize,
        n_active: usize,
        max_blocks: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .mxfp4_moe_down_mmq
            .ok_or(GpuError::MissingOp("mxfp4_moe_down_mmq"))?;
        let (ddp, _g1) = down_w.data.device_ptr(&self.stream);
        let (dsp, _g1s) = down_w.scale.device_ptr(&self.stream);
        let (dbp, _g2) = down_bias.device_ptr(&self.stream);
        let (srp, _g3) = sorted_row.device_ptr(&self.stream);
        let (slp, _g4) = sorted_slot.device_ptr(&self.stream);
        let (bep, _g5) = block_expert.device_ptr(&self.stream);
        let (wp, _g6) = topk_w.device_ptr(&self.stream);
        let (fqp, _g7) = fq.device_ptr(&self.stream);
        let (fsp, _g7s) = fs.device_ptr(&self.stream);
        let (rp, _g8) = part.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract
        check(unsafe {
            f(
                ddp as *const _,
                dsp as *const _,
                dbp as *const _,
                srp as *const _,
                slp as *const _,
                bep as *const _,
                wp as *const _,
                fqp as *const _,
                fsp as *const _,
                rp as *mut _,
                ff as u32,
                embd as u32,
                n_active as u32,
                max_blocks as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Fold [`Self::mxfp4_moe_down_mmq`]'s per-(token, slot) partials into
    /// the residual in FIXED slot order - the deterministic tail of the mmq
    /// MoE (no atomic scatter, bit-reproducible run to run).
    /// bf16-partials twin (PADDOCK_MOE_PART_BF16): `part` holds bf16 values
    /// in the same [rows, n_active, embd] index space (half the bytes of the
    /// f32 buffer it aliases); the fold stays f32 in fixed slot order.
    pub fn moe_slot_combine_bf16(
        &self,
        part: &CudaSlice<f32>,
        residual: &mut CudaSlice<f32>,
        embd: usize,
        n_active: usize,
        rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .moe_slot_combine_bf16
            .ok_or(GpuError::MissingOp("moe_slot_combine_bf16"))?;
        let (pp, _g1) = part.device_ptr(&self.stream);
        let (rp, _g2) = residual.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract
        check(unsafe {
            f(
                pp as *const _,
                rp as *mut _,
                embd as u32,
                n_active as u32,
                rows as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_moe_slot_combine_bf16(&self) -> bool {
        self.kernels.moe_slot_combine_bf16.is_some()
    }

    pub fn moe_slot_combine(
        &self,
        part: &CudaSlice<f32>,
        residual: &mut CudaSlice<f32>,
        embd: usize,
        n_active: usize,
        rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .moe_slot_combine
            .ok_or(GpuError::MissingOp("moe_slot_combine"))?;
        let (pp, _g1) = part.device_ptr(&self.stream);
        let (rp, _g2) = residual.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract
        check(unsafe {
            f(
                pp as *const _,
                rp as *mut _,
                embd as u32,
                n_active as u32,
                rows as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Write-out twin of `moe_slot_combine` (slot 485): residual = sum, no
    /// pre-zero needed. Bitwise the memset + combine chain.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_slot_combine_init(
        &self,
        part: &CudaSlice<f32>,
        residual: &mut CudaSlice<f32>,
        embd: usize,
        n_active: usize,
        rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .moe_slot_combine_init
            .ok_or(GpuError::MissingOp("moe_slot_combine_init"))?;
        let (pp, _g1) = part.device_ptr(&self.stream);
        let (rp, _g2) = residual.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract
        check(unsafe {
            f(
                pp as *const _,
                rp as *mut _,
                embd as u32,
                n_active as u32,
                rows as u32,
                self.stream_ptr(),
            )
        })
    }

    /// bm128 -> bm32 pair map (slot 489, prefill dn hybrid).
    pub fn moe_pair_map(
        &self,
        srow32: &CudaSlice<u32>,
        sslot32: &CudaSlice<u32>,
        map: &mut CudaSlice<f32>, // aliased u32 storage (moe_logits, dead here)
        n_active: usize,
        srp32: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .moe_pair_map
            .ok_or(GpuError::MissingOp("moe_pair_map"))?;
        let (rp, _g1) = srow32.device_ptr(&self.stream);
        let (sp, _g2) = sslot32.device_ptr(&self.stream);
        let (mp, _g3) = map.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract
        check(unsafe {
            f(
                rp as *const _,
                sp as *const _,
                mp as *mut _,
                n_active as u32,
                srp32 as u32,
                self.stream_ptr(),
            )
        })
    }

    /// q8 GEGLU remap quantize (slot 490): f8s-gu f32 rows (bm128 order)
    /// into bm32 fq/fs. act: 0 = gelu, 1 = silu.
    #[allow(clippy::too_many_arguments)]
    pub fn quantize_q8_geglu_remap(
        &self,
        gu: &CudaSlice<f32>,
        srow128: &CudaSlice<u32>,
        sslot128: &CudaSlice<u32>,
        map: &CudaSlice<f32>,
        fq: &mut CudaSlice<i8>,
        fs: &mut CudaSlice<f32>,
        n_ff: usize,
        n_active: usize,
        srp128: usize,
        act: u32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .quantize_q8_geglu_remap
            .ok_or(GpuError::MissingOp("quantize_q8_geglu_remap"))?;
        let (gp, _g1) = gu.device_ptr(&self.stream);
        let (rp, _g2) = srow128.device_ptr(&self.stream);
        let (sp, _g3) = sslot128.device_ptr(&self.stream);
        let (mp, _g4) = map.device_ptr(&self.stream);
        let (qp, _g5) = fq.device_ptr_mut(&self.stream);
        let (fp, _g6) = fs.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract
        check(unsafe {
            f(
                gp as *const _,
                rp as *const _,
                sp as *const _,
                mp as *const _,
                qp as *mut _,
                fp as *mut _,
                n_ff as u32,
                n_active as u32,
                srp128 as u32,
                act,
                self.stream_ptr(),
            )
        })
    }

    /// tail+combine fold (slot 491): bitwise the combine_init + moe_tail
    /// chain - the ascending-k part sum happens at the tail's dn reads.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_tail_combine(
        &self,
        x: &mut CudaSlice<f32>,
        proj: &CudaSlice<f32>,
        part: &CudaSlice<f32>,
        pn1: &CudaSlice<f32>,
        pn2: &CudaSlice<f32>,
        postw: &CudaSlice<f32>,
        n: usize,
        n_active: usize,
        eps: f32,
        os: f32,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .moe_tail_combine
            .ok_or(GpuError::MissingOp("moe_tail_combine"))?;
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (pp, _g2) = proj.device_ptr(&self.stream);
        let (pt, _g3) = part.device_ptr(&self.stream);
        let (n1, _g4) = pn1.device_ptr(&self.stream);
        let (n2, _g5) = pn2.device_ptr(&self.stream);
        let (pw, _g6) = postw.device_ptr(&self.stream);
        // SAFETY: ABI contract
        check(unsafe {
            f(
                xp as *mut _,
                pp as *const _,
                pt as *const _,
                n1 as *const _,
                n2 as *const _,
                pw as *const _,
                n as u32,
                n_active as u32,
                eps,
                os,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_moe_tail_combine(&self) -> bool {
        self.kernels.moe_tail_combine.is_some()
    }

    /// P1-1: tail+combine over BF16 partials (f32 sums, same fold order).
    #[allow(clippy::too_many_arguments)]
    pub fn moe_tail_combine_bf16(
        &self,
        x: &mut CudaSlice<f32>,
        proj: &CudaSlice<f32>,
        part: &CudaSlice<f32>,
        pn1: &CudaSlice<f32>,
        pn2: &CudaSlice<f32>,
        postw: &CudaSlice<f32>,
        n: usize,
        n_active: usize,
        eps: f32,
        os: f32,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .moe_tail_combine_bf16
            .ok_or(GpuError::MissingOp("moe_tail_combine_bf16"))?;
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (pp, _g2) = proj.device_ptr(&self.stream);
        let (pt, _g3) = part.device_ptr(&self.stream);
        let (n1, _g4) = pn1.device_ptr(&self.stream);
        let (n2, _g5) = pn2.device_ptr(&self.stream);
        let (pw, _g6) = postw.device_ptr(&self.stream);
        // SAFETY: ABI contract
        check(unsafe {
            f(
                xp as *mut _,
                pp as *const _,
                pt as *const _,
                n1 as *const _,
                n2 as *const _,
                pw as *const _,
                n as u32,
                n_active as u32,
                eps,
                os,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_moe_tail_combine_bf16(&self) -> bool {
        self.kernels.moe_tail_combine_bf16.is_some()
    }

    pub fn has_pf_dn_hybrid(&self) -> bool {
        self.kernels.moe_pair_map.is_some() && self.kernels.quantize_q8_geglu_remap.is_some()
    }

    pub fn has_moe_combine_init(&self) -> bool {
        self.kernels.moe_slot_combine_init.is_some()
    }

    /// One launch for a batch of device-to-device copies: `descs` holds n
    /// consecutive `{src_ptr, dst_ptr, bytes}` u64 triples; bytes % 16 == 0.
    /// The pointers must stay valid until the stream reaches the launch.
    pub fn batched_copy(&self, descs: &CudaSlice<u64>, n: usize) -> Result<(), GpuError> {
        let f = self
            .kernels
            .batched_copy
            .ok_or(GpuError::MissingOp("batched_copy"))?;
        debug_assert!(descs.len() >= n * 3);
        let (dp, _g) = descs.device_ptr(&self.stream);
        // SAFETY: ABI contract; descriptor pointers guaranteed live by caller
        check(unsafe { f(dp as *const _, n as u32, self.stream_ptr()) })
    }

    /// Build a [`RepackedMxfp4`] from an MXFP4 [`QuantTensor`] (17-byte blocks):
    /// splits each block into 16-aligned data + a separate contiguous scale byte,
    /// so the sorted MoE GEMM reads coalesced. One-time, at model load.
    pub fn repack_mxfp4(&self, src: &QuantTensor) -> Result<RepackedMxfp4, GpuError> {
        debug_assert_eq!(src.ty, GgmlType::Mxfp4);
        let f = self
            .kernels
            .mxfp4_repack
            .ok_or(GpuError::MissingOp("mxfp4_repack"))?;
        let n_blocks = src.bytes.len() / 17;
        let mut data = self.alloc_u8(n_blocks * 16)?;
        let mut scale = self.alloc_u8(n_blocks)?;
        {
            let (sp, _g1) = src.bytes.device_ptr(&self.stream);
            let (dp, _g2) = data.device_ptr_mut(&self.stream);
            let (scp, _g3) = scale.device_ptr_mut(&self.stream);
            // SAFETY: ABI contract
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

    /// Grouped MoE down + weighted mix + residual add (atomic). `residual`
    /// [batch, embd] must be pre-zeroed.
    #[allow(clippy::too_many_arguments)]
    pub fn mxfp4_moe_down_grouped(
        &self,
        down_w: &RepackedMxfp4,
        down_bias: &CudaSlice<f32>,
        slot_of: &CudaSlice<u8>,
        topk_w: &CudaSlice<f32>,
        fused: &CudaSlice<f32>,
        residual: &mut CudaSlice<f32>,
        ff: usize,
        embd: usize,
        n_expert: usize,
        n_active: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .mxfp4_moe_down_grouped
            .ok_or(GpuError::MissingOp("mxfp4_moe_down_grouped"))?;
        let (ddp, _g1) = down_w.data.device_ptr(&self.stream);
        let (dsp, _g1s) = down_w.scale.device_ptr(&self.stream);
        let (dbp, _g2) = down_bias.device_ptr(&self.stream);
        let (sp, _g3) = slot_of.device_ptr(&self.stream);
        let (wp, _g4) = topk_w.device_ptr(&self.stream);
        let (fp, _g5) = fused.device_ptr(&self.stream);
        let (rp, _g6) = residual.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; residual pre-zeroed
        check(unsafe {
            f(
                ddp as *const _,
                dsp as *const _,
                dbp as *const _,
                sp as *const _,
                wp as *const _,
                fp as *const _,
                rp as *mut _,
                ff as u32,
                embd as u32,
                n_expert as u32,
                n_active as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Quantize an f32 activation to symmetric int8 + per-32-block scale (Q8_1
    /// style) for the dp4a integer-matmul path. `q` is [n] int8, `scale` [n/32].
    pub fn quantize_q8(
        &self,
        x: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        scale: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .quantize_q8
            .ok_or(GpuError::MissingOp("quantize_q8"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (qp, _g2) = q.device_ptr_mut(&self.stream);
        let (sp, _g3) = scale.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; n % 32 == 0, buffers sized by caller
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

    /// `quantize_q8` with relu(x)^2 folded in front of the per-32 amax
    /// The seam that lets a squared-relu dense FFN - nemotron's
    /// shared expert - run its up plane on the ordinary q8 GEMM ladder: the
    /// GEMM writes raw pre-activation and this applies the activation on the
    /// way into int8. Bit-identical to relu^2-into-f32 then `quantize_q8`.
    pub fn quantize_q8_relu2(
        &self,
        x: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        scale: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .quantize_q8_relu2
            .ok_or(GpuError::MissingOp("quantize_q8_relu2"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (qp, _g2) = q.device_ptr_mut(&self.stream);
        let (sp, _g3) = scale.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; n % 32 == 0, buffers sized by caller
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

    /// True when the relu^2 quantize fusion is present (the dense-ladder
    /// shared-expert route's only new requirement).
    pub fn has_quantize_q8_relu2(&self) -> bool {
        self.kernels.quantize_q8_relu2.is_some()
    }

    /// dp4a MXFP4 GEMV for the expert at `idx[slot]` against a pre-quantized
    /// activation. In-register nibble unpack + integer `__dp4a` - the MoE's
    /// compute lever. `w` is the whole MXFP4 expert tensor.
    #[allow(clippy::too_many_arguments)]
    pub fn mxfp4_gemv_indexed_dp4a(
        &self,
        w: &CudaSlice<u8>,
        bias: Option<&CudaSlice<f32>>,
        idx: &CudaSlice<u32>,
        slot: usize,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .mxfp4_gemv_indexed_dp4a
            .ok_or(GpuError::MissingOp("mxfp4_gemv_indexed_dp4a"))?;
        let (wp, _g1) = w.device_ptr(&self.stream);
        let (ip, _g2) = idx.device_ptr(&self.stream);
        let (xqp, _g3) = xq.device_ptr(&self.stream);
        let (xsp, _g4) = xs.device_ptr(&self.stream);
        let (yp, _g5) = y.device_ptr_mut(&self.stream);
        let (bias_ptr, _gb);
        let bp: *const core::ffi::c_void = match bias {
            Some(b) => {
                (bias_ptr, _gb) = b.device_ptr(&self.stream);
                bias_ptr as *const _
            }
            None => core::ptr::null(),
        };
        // SAFETY: ABI contract; activation pre-quantized, weight block-aligned
        check(unsafe {
            f(
                wp as *const _,
                bp,
                ip as *const _,
                slot as u32,
                xqp as *const _,
                xsp as *const _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    /// dp4a Q8_0 GEMV against a pre-quantized activation (`xq`/`xs`). Integer
    /// dot product - the llama.cpp/mistral.rs method, ~10× cheaper compute than
    /// the f32 dequant GEMV.
    #[allow(clippy::too_many_arguments)]
    pub fn q8_0_gemv_dp4a(
        &self,
        w: &QuantTensor,
        bias: Option<&CudaSlice<f32>>,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_gemv_dp4a
            .ok_or(GpuError::MissingOp("q8_0_gemv_dp4a"))?;
        if w.ty != GgmlType::Q8_0 {
            return Err(GpuError::NoKernel {
                name: "<q8_0_gemv_dp4a weight>".to_owned(),
                ty: w.ty,
            });
        }
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        let (wp, _g1) = w.bytes.device_ptr(&self.stream);
        let (xqp, _g3) = xq.device_ptr(&self.stream);
        let (xsp, _g4) = xs.device_ptr(&self.stream);
        let (yp, _g5) = y.device_ptr_mut(&self.stream);
        let (bias_ptr, _gb);
        let bp: *const core::ffi::c_void = match bias {
            Some(b) => {
                (bias_ptr, _gb) = b.device_ptr(&self.stream);
                bias_ptr as *const _
            }
            None => core::ptr::null(),
        };
        // SAFETY: ABI contract; activation pre-quantized to [in_dim]/[in_dim/32]
        check(unsafe {
            f(
                wp as *const _,
                bp,
                xqp as *const _,
                xsp as *const _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    /// On-device MoE top-k: biased `logits` -> `out_idx` (expert ids) + `out_w`
    /// (softmax-over-selected weights). No host round-trip.
    pub fn moe_topk(
        &self,
        logits: &CudaSlice<f32>,
        n_expert: usize,
        k: usize,
        out_idx: &mut CudaSlice<u32>,
        out_w: &mut CudaSlice<f32>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .moe_topk
            .ok_or(GpuError::MissingOp("moe_topk"))?;
        let (lp, _g1) = logits.device_ptr(&self.stream);
        let (ip, _g2) = out_idx.device_ptr_mut(&self.stream);
        let (wp, _g3) = out_w.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract
        check(unsafe {
            f(
                lp as *const _,
                n_expert as u32,
                k as u32,
                ip as *mut _,
                wp as *mut _,
                self.stream_ptr(),
            )
        })
    }

    /// Fused MXFP4-dequant + GEMV for the expert at `idx[slot]` (device index):
    /// y = bias[e] + W[e]·x. `w` is the whole MXFP4 expert tensor.
    #[allow(clippy::too_many_arguments)]
    pub fn mxfp4_gemv_indexed(
        &self,
        w: &CudaSlice<u8>,
        bias: Option<&CudaSlice<f32>>,
        idx: &CudaSlice<u32>,
        slot: usize,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .mxfp4_gemv_indexed
            .ok_or(GpuError::MissingOp("mxfp4_gemv_indexed"))?;
        let (wp, _g1) = w.device_ptr(&self.stream);
        let (ip, _g2) = idx.device_ptr(&self.stream);
        let (xp, _g3) = x.device_ptr(&self.stream);
        let (yp, _g4) = y.device_ptr_mut(&self.stream);
        // bias is optional - pass a null pointer when absent (kernel checks)
        let (bias_ptr, _gb);
        let bp: *const core::ffi::c_void = match bias {
            Some(b) => {
                (bias_ptr, _gb) = b.device_ptr(&self.stream);
                bias_ptr as *const _
            }
            None => core::ptr::null(),
        };
        // SAFETY: ABI contract; rows block-aligned per caller
        check(unsafe {
            f(
                wp as *const _,
                bp,
                ip as *const _,
                slot as u32,
                xp as *const _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Batched Q8_0 GEMM: `y` [batch, out_dim] = bias + `x` [batch, in_dim] · Wᵀ.
    /// The weight row is dequanted once and reused across the batch - the
    /// weight-read amortization behind concurrent-decode throughput.
    pub fn q8_0_gemm(
        &self,
        w: &QuantTensor,
        bias: Option<&CudaSlice<f32>>,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_gemm
            .ok_or(GpuError::MissingOp("q8_0_gemm"))?;
        if w.ty != GgmlType::Q8_0 {
            return Err(GpuError::NoKernel {
                name: "<q8_0_gemm weight>".to_owned(),
                ty: w.ty,
            });
        }
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        let (wp, _g1) = w.bytes.device_ptr(&self.stream);
        let (xp, _g3) = x.device_ptr(&self.stream);
        let (yp, _g4) = y.device_ptr_mut(&self.stream);
        let (bias_ptr, _gb);
        let bp: *const core::ffi::c_void = match bias {
            Some(b) => {
                (bias_ptr, _gb) = b.device_ptr(&self.stream);
                bias_ptr as *const _
            }
            None => core::ptr::null(),
        };
        // SAFETY: ABI contract; x/y sized [batch, in|out] by caller
        check(unsafe {
            f(
                wp as *const _,
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

    /// Fused Q8_0-dequant + GEMV for a dense weight kept Q8_0-resident:
    /// y = bias + Wᵀ·x. `w` is the Q8_0 weight [in_dim, out_dim] (GGUF layout);
    /// `bias` folds in when present. Reads ~3.8× fewer bytes than dequanting to
    /// f32 and calling cuBLAS.
    #[allow(clippy::too_many_arguments)]
    pub fn q8_0_gemv(
        &self,
        w: &QuantTensor,
        bias: Option<&CudaSlice<f32>>,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_gemv
            .ok_or(GpuError::MissingOp("q8_0_gemv"))?;
        // guard the format: this kernel decodes Q8_0 blocks; handing it any other
        // layout silently produces garbage (the router is F32, not Q8_0)
        if w.ty != GgmlType::Q8_0 {
            return Err(GpuError::NoKernel {
                name: "<q8_0_gemv weight>".to_owned(),
                ty: w.ty,
            });
        }
        let in_dim = w.dims[0];
        let out_dim = w.dims[1];
        let (wp, _g1) = w.bytes.device_ptr(&self.stream);
        let (xp, _g3) = x.device_ptr(&self.stream);
        let (yp, _g4) = y.device_ptr_mut(&self.stream);
        let (bias_ptr, _gb);
        let bp: *const core::ffi::c_void = match bias {
            Some(b) => {
                (bias_ptr, _gb) = b.device_ptr(&self.stream);
                bias_ptr as *const _
            }
            None => core::ptr::null(),
        };
        // SAFETY: ABI contract; rows block-aligned (in_dim % 32 == 0) per caller
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

    /// Fused MoE gate+up+swiglu over the active experts -> `out` [n_active*ff].
    /// `gate`/`up` are the full MXFP4 expert tensors [embd, ff, n_experts]; their
    /// biases are [n_experts, ff]. One launch replaces the per-expert gate/up
    /// GEMV + swiglu.
    #[allow(clippy::too_many_arguments)]
    pub fn mxfp4_moe_gate_up(
        &self,
        gate_w: &RepackedMxfp4,
        gate_bias: &CudaSlice<f32>,
        up_w: &RepackedMxfp4,
        up_bias: &CudaSlice<f32>,
        idx: &CudaSlice<u32>,
        x: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        in_dim: usize,
        ff: usize,
        n_active: usize,
        alpha: f32,
        limit: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .mxfp4_moe_gate_up
            .ok_or(GpuError::MissingOp("mxfp4_moe_gate_up"))?;
        let (gdp, _g1) = gate_w.data.device_ptr(&self.stream);
        let (gsp, _g1s) = gate_w.scale.device_ptr(&self.stream);
        let (gbp, _g2) = gate_bias.device_ptr(&self.stream);
        let (udp, _g3) = up_w.data.device_ptr(&self.stream);
        let (usp, _g3s) = up_w.scale.device_ptr(&self.stream);
        let (ubp, _g4) = up_bias.device_ptr(&self.stream);
        let (ip, _g5) = idx.device_ptr(&self.stream);
        let (xp, _g6) = x.device_ptr(&self.stream);
        let (op, _g7) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; expert tensors block-aligned per caller
        check(unsafe {
            f(
                gdp as *const _,
                gsp as *const _,
                gbp as *const _,
                udp as *const _,
                usp as *const _,
                ubp as *const _,
                ip as *const _,
                xp as *const _,
                op as *mut _,
                in_dim as u32,
                ff as u32,
                n_active as u32,
                alpha,
                limit,
                self.stream_ptr(),
            )
        })
    }

    /// dp4a fused MoE gate+up+swiglu, against a pre-quantized activation.
    #[allow(clippy::too_many_arguments)]
    pub fn mxfp4_moe_gate_up_dp4a(
        &self,
        gate_w: &RepackedMxfp4,
        gate_bias: &CudaSlice<f32>,
        up_w: &RepackedMxfp4,
        up_bias: &CudaSlice<f32>,
        idx: &CudaSlice<u32>,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        in_dim: usize,
        ff: usize,
        n_active: usize,
        alpha: f32,
        limit: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .mxfp4_moe_gate_up_dp4a
            .ok_or(GpuError::MissingOp("mxfp4_moe_gate_up_dp4a"))?;
        let (gdp, _g1) = gate_w.data.device_ptr(&self.stream);
        let (gsp, _g1s) = gate_w.scale.device_ptr(&self.stream);
        let (gbp, _g2) = gate_bias.device_ptr(&self.stream);
        let (udp, _g3) = up_w.data.device_ptr(&self.stream);
        let (usp, _g3s) = up_w.scale.device_ptr(&self.stream);
        let (ubp, _g4) = up_bias.device_ptr(&self.stream);
        let (ip, _g5) = idx.device_ptr(&self.stream);
        let (xqp, _g6) = xq.device_ptr(&self.stream);
        let (xsp, _g7) = xs.device_ptr(&self.stream);
        let (op, _g8) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; activation pre-quantized
        check(unsafe {
            f(
                gdp as *const _,
                gsp as *const _,
                gbp as *const _,
                udp as *const _,
                usp as *const _,
                ubp as *const _,
                ip as *const _,
                xqp as *const _,
                xsp as *const _,
                op as *mut _,
                in_dim as u32,
                ff as u32,
                n_active as u32,
                alpha,
                limit,
                self.stream_ptr(),
            )
        })
    }

    /// dp4a fused MoE down + weighted mix + residual add. `fused_q`/`fused_s` are
    /// the per-slot pre-quantized swiglu outputs.
    #[allow(clippy::too_many_arguments)]
    pub fn mxfp4_moe_down_dp4a(
        &self,
        down_w: &RepackedMxfp4,
        down_bias: &CudaSlice<f32>,
        idx: &CudaSlice<u32>,
        topk_w: &CudaSlice<f32>,
        fused_q: &CudaSlice<i8>,
        fused_s: &CudaSlice<f32>,
        residual: &mut CudaSlice<f32>,
        ff: usize,
        embd: usize,
        n_active: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .mxfp4_moe_down_dp4a
            .ok_or(GpuError::MissingOp("mxfp4_moe_down_dp4a"))?;
        let (ddp, _g1) = down_w.data.device_ptr(&self.stream);
        let (dsp, _g1s) = down_w.scale.device_ptr(&self.stream);
        let (dbp, _g2) = down_bias.device_ptr(&self.stream);
        let (ip, _g3) = idx.device_ptr(&self.stream);
        let (wp, _g4) = topk_w.device_ptr(&self.stream);
        let (fqp, _g5) = fused_q.device_ptr(&self.stream);
        let (fsp, _g6) = fused_s.device_ptr(&self.stream);
        let (rp, _g7) = residual.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract
        check(unsafe {
            f(
                ddp as *const _,
                dsp as *const _,
                dbp as *const _,
                ip as *const _,
                wp as *const _,
                fqp as *const _,
                fsp as *const _,
                rp as *mut _,
                ff as u32,
                embd as u32,
                n_active as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Batched (grid.z = token) fused dp4a MoE gate+up+swiglu - the llama
    /// mmvq-with-ids shape for tiny serving batches. `idx` is [batch,
    /// n_active], `xq`/`xs` [batch, in_dim] strided, `out` [batch, n_active,
    /// ff]. Token 0's math is bit-identical to `mxfp4_moe_gate_up_dp4a`.
    #[allow(clippy::too_many_arguments)]
    pub fn mxfp4_moe_gate_up_dp4a_b(
        &self,
        gate_w: &RepackedMxfp4,
        gate_bias: &CudaSlice<f32>,
        up_w: &RepackedMxfp4,
        up_bias: &CudaSlice<f32>,
        idx: &CudaSlice<u32>,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
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
            .mxfp4_moe_gate_up_dp4a_b
            .ok_or(GpuError::MissingOp("mxfp4_moe_gate_up_dp4a_b"))?;
        let (gdp, _g1) = gate_w.data.device_ptr(&self.stream);
        let (gsp, _g1s) = gate_w.scale.device_ptr(&self.stream);
        let (gbp, _g2) = gate_bias.device_ptr(&self.stream);
        let (udp, _g3) = up_w.data.device_ptr(&self.stream);
        let (usp, _g3s) = up_w.scale.device_ptr(&self.stream);
        let (ubp, _g4) = up_bias.device_ptr(&self.stream);
        let (ip, _g5) = idx.device_ptr(&self.stream);
        let (xqp, _g6) = xq.device_ptr(&self.stream);
        let (xsp, _g7) = xs.device_ptr(&self.stream);
        let (op, _g8) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; activation pre-quantized
        check(unsafe {
            f(
                gdp as *const _,
                gsp as *const _,
                gbp as *const _,
                udp as *const _,
                usp as *const _,
                ubp as *const _,
                ip as *const _,
                xqp as *const _,
                xsp as *const _,
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

    /// Batched fused dp4a MoE down + weighted mix + residual add (companion of
    /// [`Self::mxfp4_moe_gate_up_dp4a_b`]). `residual` is [batch, embd]; one
    /// writer per (token, element) - deterministic.
    #[allow(clippy::too_many_arguments)]
    pub fn mxfp4_moe_down_dp4a_b(
        &self,
        down_w: &RepackedMxfp4,
        down_bias: &CudaSlice<f32>,
        idx: &CudaSlice<u32>,
        topk_w: &CudaSlice<f32>,
        fused_q: &CudaSlice<i8>,
        fused_s: &CudaSlice<f32>,
        residual: &mut CudaSlice<f32>,
        ff: usize,
        embd: usize,
        n_active: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .mxfp4_moe_down_dp4a_b
            .ok_or(GpuError::MissingOp("mxfp4_moe_down_dp4a_b"))?;
        let (ddp, _g1) = down_w.data.device_ptr(&self.stream);
        let (dsp, _g1s) = down_w.scale.device_ptr(&self.stream);
        let (dbp, _g2) = down_bias.device_ptr(&self.stream);
        let (ip, _g3) = idx.device_ptr(&self.stream);
        let (wp, _g4) = topk_w.device_ptr(&self.stream);
        let (fqp, _g5) = fused_q.device_ptr(&self.stream);
        let (fsp, _g6) = fused_s.device_ptr(&self.stream);
        let (rp, _g7) = residual.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract
        check(unsafe {
            f(
                ddp as *const _,
                dsp as *const _,
                dbp as *const _,
                ip as *const _,
                wp as *const _,
                fqp as *const _,
                fsp as *const _,
                rp as *mut _,
                ff as u32,
                embd as u32,
                n_active as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Fused MoE down + weighted expert mix + residual add: `residual` += Σ_slot
    /// w[slot]·(down_e·fused[slot] + bias). One launch replaces the per-expert
    /// down GEMV + scale_add and the final residual add (no scratch, no memset).
    #[allow(clippy::too_many_arguments)]
    pub fn mxfp4_moe_down(
        &self,
        down_w: &RepackedMxfp4,
        down_bias: &CudaSlice<f32>,
        idx: &CudaSlice<u32>,
        topk_w: &CudaSlice<f32>,
        fused: &CudaSlice<f32>,
        residual: &mut CudaSlice<f32>,
        ff: usize,
        embd: usize,
        n_active: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .mxfp4_moe_down
            .ok_or(GpuError::MissingOp("mxfp4_moe_down"))?;
        let (ddp, _g1) = down_w.data.device_ptr(&self.stream);
        let (dsp, _g1s) = down_w.scale.device_ptr(&self.stream);
        let (dbp, _g2) = down_bias.device_ptr(&self.stream);
        let (ip, _g3) = idx.device_ptr(&self.stream);
        let (wp, _g4) = topk_w.device_ptr(&self.stream);
        let (fp, _g5) = fused.device_ptr(&self.stream);
        let (rp, _g6) = residual.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; buffers sized by caller
        check(unsafe {
            f(
                ddp as *const _,
                dsp as *const _,
                dbp as *const _,
                ip as *const _,
                wp as *const _,
                fp as *const _,
                rp as *mut _,
                ff as u32,
                embd as u32,
                n_active as u32,
                self.stream_ptr(),
            )
        })
    }

    /// x[..n] += w[slot] * y[..n], weight from a device buffer.
    pub fn scale_add_dev(
        &self,
        x: &mut CudaSlice<f32>,
        y: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        slot: usize,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .scale_add_dev
            .ok_or(GpuError::MissingOp("scale_add_dev"))?;
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (yp, _g2) = y.device_ptr(&self.stream);
        let (wp, _g3) = w.device_ptr(&self.stream);
        // SAFETY: ABI contract
        check(unsafe {
            f(
                xp as *mut _,
                yp as *const _,
                wp as *const _,
                slot as u32,
                n as u32,
                self.stream_ptr(),
            )
        })
    }
}
