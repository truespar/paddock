//! Q8_0 + f8bs MoE experts (sorted/mma/dec2) + f8w GEMM.

use super::error::*;
use super::*;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

impl GpuExecutor {
    /// sm_120a dense W8A8-FP8 GEMM: e4m3 weights (q8_0_to_f8w planes) x e4m3
    /// activations. Weight-precision class between q8_0 and the fp4 rungs;
    /// retrieval-quality gated like every block-scale class.
    #[allow(clippy::too_many_arguments)]
    /// f8_gemm_w8 over a ROW-OFFSET sub-view of a fused plane (see
    /// f8_gemv_at_off): out-row-major layout, so the sub-plane is pointer
    /// offsets. The offset base stays 16B-aligned (in_dim % 16 == 0).
    #[allow(clippy::too_many_arguments)]
    pub fn f8_gemm_w8_off(
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
        // tile-linear planes: box-offset sub-view on the lin prefill twin
        // (fused-plane slice boundaries are box-aligned by construction)
        if w.scale.len() == 4 || w.scale.len() == 12 {
            return self.f8_gemm_lin_kt(w, row_off, xq, xs, y, in_dim, out_dim, batch, false);
        }
        let f = self
            .kernels
            .f8_gemm_w8
            .ok_or(GpuError::MissingOp("f8_gemm_w8"))?;
        debug_assert!(w.data.len() >= (row_off + out_dim) * in_dim);
        let (wdp, _g1) = w.data.device_ptr(&self.stream);
        let (wsp, _g2) = w.scale.device_ptr(&self.stream);
        let (xqp, _g3) = xq.device_ptr(&self.stream);
        let (xsp, _g4) = xs.device_ptr(&self.stream);
        let (yp, _g5) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; in_dim % 32 == 0 checked by the launcher
        check(unsafe {
            f(
                (wdp + (row_off * in_dim) as u64) as *const _,
                (wsp + (row_off * (in_dim / 32)) as u64) as *const _,
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

    pub fn f8_gemm_w8(
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
        // tile-linear planes carry a 4-byte marker scale (scales live inside
        // the boxes) - route to the lin prefill twin, call sites unchanged
        if w.scale.len() == 4 || w.scale.len() == 12 {
            return self.f8_gemm_lin_kt(w, row_off, xq, xs, y, in_dim, out_dim, batch, false);
        }
        let f = self
            .kernels
            .f8_gemm_w8
            .ok_or(GpuError::MissingOp("f8_gemm_w8"))?;
        debug_assert_eq!(w.data.len(), out_dim * in_dim);
        let (wdp, _g1) = w.data.device_ptr(&self.stream);
        let (wsp, _g2) = w.scale.device_ptr(&self.stream);
        let (xqp, _g3) = xq.device_ptr(&self.stream);
        let (xsp, _g4) = xs.device_ptr(&self.stream);
        let (yp, _g5) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; in_dim % 32 == 0 checked by the launcher
        check(unsafe {
            f(
                (wdp as usize + row_off * in_dim) as *const _,
                (wsp as usize + row_off * (in_dim / 32)) as *const _,
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

    /// f8 mma_ks twin (B <= 64): K-split block-scale MMA GEMM over the f8w
    /// planes for the spec-verify band where the 128-col TMA tile doesn't
    /// amortize. Same e4m3 inputs as [`f8_gemm_w8`] (bit-equal to it per
    /// element at nz==1); `part` is the same stream-k fixup scratch the q8
    /// ks family rides.
    #[allow(clippy::too_many_arguments)]
    pub fn f8_gemm_mma_ks(
        &self,
        w: &RepackedMxfp4,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<u8>,
        part: &mut CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        in_dim: usize,
        out_dim: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        // tile-linear planes ride the lin K-split decode GEMM (same operand
        // classes: per-32 e4m3 activations + the shared part scratch)
        if w.scale.len() == 4 || w.scale.len() == 12 {
            return self.f8_gemm_lin(w, 0, in_dim, out_dim, xq, xs, part, y, batch);
        }
        let f = self
            .kernels
            .f8_gemm_mma_ks
            .ok_or(GpuError::MissingOp("f8_gemm_mma_ks"))?;
        debug_assert_eq!(w.data.len(), out_dim * in_dim);
        let (wdp, _g1) = w.data.device_ptr(&self.stream);
        let (wsp, _g2) = w.scale.device_ptr(&self.stream);
        let (xqp, _g3) = xq.device_ptr(&self.stream);
        let (xsp, _g4) = xs.device_ptr(&self.stream);
        let (pp, _g5) = part.device_ptr_mut(&self.stream);
        let (yp, _g6) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; part >= 8 * out_dim * batch f32
        check(unsafe {
            f(
                wdp as *const _,
                wsp as *const _,
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

    /// Whether the pack ships the f8 mma_ks twin (older packs fall to the
    /// TMA GEMM for the whole >=4 band).
    pub fn has_f8_gemm_mma_ks(&self) -> bool {
        self.kernels.f8_gemm_mma_ks.is_some()
    }

    /// sm_120a dense block-scale GEMM: y[b][o] = W_mxfp4[o] . x_e4m3[b] with
    /// hardware ue8m0 scaling. Activations from [`quantize_e4m3`] (`xq` raw
    /// e4m3 bytes over an i8 plane, `xs` the ue8m0 plane). NUMERIC CLASS:
    /// fp4 weights + fp8 activations - retrieval-quality gated, not
    /// greedy-exact vs the Q8_0 paths.
    #[allow(clippy::too_many_arguments)]
    pub fn mxfp4_gemm_bs(
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
            .mxfp4_gemm_bs
            .ok_or(GpuError::MissingOp("mxfp4_gemm_bs"))?;
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

    /// K-split int8 tensor-core GEMM (B <= 64): fills many-SM dies where the
    /// plain mma grid is a handful of out-tiles. `part` is the stream-k fixup
    /// scratch (always big enough for the dense shapes).
    pub fn q8_0_gemm_mma_ks(
        &self,
        w: &RepackedQ8,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        part: &mut CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_gemm_mma_ks
            .ok_or(GpuError::MissingOp("q8_0_gemm_mma_ks"))?;
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
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

    /// Bias-folding `q8_0_gemm_mma_ks` (bit-exact vs GEMM + `bias_add`; the
    /// bias adds after the completed fixed-order K-split sum).
    pub fn q8_0_gemm_mma_ks_b(
        &self,
        w: &RepackedQ8,
        bias: Option<&CudaSlice<f32>>,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        part: &mut CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_gemm_mma_ks_b
            .ok_or(GpuError::MissingOp("q8_0_gemm_mma_ks_b"))?;
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (scp, _g2) = w.scale.device_ptr(&self.stream);
        let (xqp, _g3) = xq.device_ptr(&self.stream);
        let (xsp, _g4) = xs.device_ptr(&self.stream);
        let (pp, _g5) = part.device_ptr_mut(&self.stream);
        let (yp, _g6) = y.device_ptr_mut(&self.stream);
        let (bias_ptr, _gb);
        let bp: *const core::ffi::c_void = match bias {
            Some(b) => {
                (bias_ptr, _gb) = b.device_ptr(&self.stream);
                bias_ptr as *const _
            }
            None => core::ptr::null(),
        };
        // SAFETY: ABI contract; part >= 8 * out_dim * batch f32
        check(unsafe {
            f(
                dp as *const _,
                scp as *const _,
                xqp as *const _,
                xsp as *const _,
                bp,
                pp as *mut _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Q8_0 routed-expert fused gate+up+SwiGLU (qwen A3B class), token-batched.
    /// `gate`/`up` are repacked 3D expert tensors [in, ff, n_expert]; `idx`
    /// [batch, n_active] picks rows; `out` [batch, n_active, ff].
    #[allow(clippy::too_many_arguments)]
    pub fn q8_0_moe_gate_up(
        &self,
        gate: &RepackedQ8,
        up: &RepackedQ8,
        idx: &CudaSlice<u32>,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n_active: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_moe_gate_up_dp4a
            .ok_or(GpuError::MissingOp("q8_0_moe_gate_up_dp4a"))?;
        let (in_dim, ff) = (gate.dims[0], gate.dims[1]);
        let (gd, _g1) = gate.data.device_ptr(&self.stream);
        let (gs, _g2) = gate.scale.device_ptr(&self.stream);
        let (ud, _g3) = up.data.device_ptr(&self.stream);
        let (us, _g4) = up.scale.device_ptr(&self.stream);
        let (ip, _g5) = idx.device_ptr(&self.stream);
        let (xp, _g6) = xq.device_ptr(&self.stream);
        let (sp, _g7) = xs.device_ptr(&self.stream);
        let (op, _g8) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                gd as *const _,
                gs as *const _,
                ud as *const _,
                us as *const _,
                ip as *const _,
                xp as *const _,
                sp as *const _,
                op as *mut _,
                in_dim as u32,
                ff as u32,
                n_active as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Q8_0 routed-expert down + weighted combine into `out` [batch, embd]
    /// (plain write; caller folds shared expert + residual). `down` is the
    /// repacked [ff, embd, n_expert] expert tensor; `fq`/`fs` the quantized
    /// SwiGLU output [batch, n_active, ff].
    #[allow(clippy::too_many_arguments)]
    pub fn q8_0_moe_down(
        &self,
        down: &RepackedQ8,
        idx: &CudaSlice<u32>,
        topk_w: &CudaSlice<f32>,
        fq: &CudaSlice<i8>,
        fs: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n_active: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_moe_down_dp4a
            .ok_or(GpuError::MissingOp("q8_0_moe_down_dp4a"))?;
        let (ff, embd) = (down.dims[0], down.dims[1]);
        let (dd, _g1) = down.data.device_ptr(&self.stream);
        let (ds, _g2) = down.scale.device_ptr(&self.stream);
        let (ip, _g3) = idx.device_ptr(&self.stream);
        let (wp, _g4) = topk_w.device_ptr(&self.stream);
        let (qp, _g5) = fq.device_ptr(&self.stream);
        let (sp, _g6) = fs.device_ptr(&self.stream);
        let (op, _g7) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                dd as *const _,
                ds as *const _,
                ip as *const _,
                wp as *const _,
                qp as *const _,
                sp as *const _,
                op as *mut _,
                ff as u32,
                embd as u32,
                n_active as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Token-batched Q8_0 single-plane expert up + squared-relu - the
    /// nemotron_h_moe class (no gate matrix; relu(up(x))^2). Same layout and
    /// dp4a class as `q8_0_moe_gate_up` with one weight stream; serves the
    /// shared expert as a 1-expert plane with a zero idx (nvf4 convention).
    #[allow(clippy::too_many_arguments)]
    pub fn q8_0_moe_up_relu2(
        &self,
        up: &RepackedQ8,
        idx: &CudaSlice<u32>,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n_active: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_moe_up_relu2
            .ok_or(GpuError::MissingOp("q8_0_moe_up_relu2"))?;
        let (in_dim, ff) = (up.dims[0], up.dims[1]);
        let (ud, _g1) = up.data.device_ptr(&self.stream);
        let (us, _g2) = up.scale.device_ptr(&self.stream);
        let (ip, _g3) = idx.device_ptr(&self.stream);
        let (xp, _g4) = xq.device_ptr(&self.stream);
        let (sp, _g5) = xs.device_ptr(&self.stream);
        let (op, _g6) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                ud as *const _,
                us as *const _,
                ip as *const _,
                xp as *const _,
                sp as *const _,
                op as *mut _,
                in_dim as u32,
                ff as u32,
                n_active as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Both relu^2 expert kernels present (the nemotron GGUF lane's MoE
    /// requirement; the activation-agnostic down/align/combine glue has its
    /// own gates).
    pub fn has_q8_0_moe_relu2(&self) -> bool {
        self.kernels.q8_0_moe_up_relu2.is_some() && self.kernels.q8_0_moe_up_relu2_sorted.is_some()
    }

    /// GEGLU twin of `q8_0_moe_gate_up` - the gemma4-A4B hybrid FFN's routed
    /// branch (gelu_tanh(gate)*up, pd_geglu constants). Same layout contract.
    #[allow(clippy::too_many_arguments)]
    pub fn q8_0_moe_gate_up_geglu(
        &self,
        gate: &RepackedQ8,
        up: &RepackedQ8,
        idx: &CudaSlice<u32>,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n_active: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_moe_gate_up_geglu
            .ok_or(GpuError::MissingOp("q8_0_moe_gate_up_geglu"))?;
        let (in_dim, ff) = (gate.dims[0], gate.dims[1]);
        let (gd, _g1) = gate.data.device_ptr(&self.stream);
        let (gs, _g2) = gate.scale.device_ptr(&self.stream);
        let (ud, _g3) = up.data.device_ptr(&self.stream);
        let (us, _g4) = up.scale.device_ptr(&self.stream);
        let (ip, _g5) = idx.device_ptr(&self.stream);
        let (xp, _g6) = xq.device_ptr(&self.stream);
        let (sp, _g7) = xs.device_ptr(&self.stream);
        let (op, _g8) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                gd as *const _,
                gs as *const _,
                ud as *const _,
                us as *const _,
                ip as *const _,
                xp as *const _,
                sp as *const _,
                op as *mut _,
                in_dim as u32,
                ff as u32,
                n_active as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Fold per-expert scalars into routed top-k weights: w[i] *= scale[idx[i]]
    /// (gemma4-A4B `ffn_down_exps.scale` before the down combine).
    pub fn moe_scale_w(
        &self,
        w: &mut CudaSlice<f32>,
        idx: &CudaSlice<u32>,
        scale: &CudaSlice<f32>,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .moe_scale_w
            .ok_or(GpuError::MissingOp("moe_scale_w"))?;
        let (wp, _g1) = w.device_ptr_mut(&self.stream);
        let (ip, _g2) = idx.device_ptr(&self.stream);
        let (sp, _g3) = scale.device_ptr(&self.stream);
        check(unsafe {
            f(
                wp as *mut _,
                ip as *const _,
                sp as *const _,
                n as u32,
                self.stream_ptr(),
            )
        })
    }
    /// head+router+topk in one launch (slot 487): bit-identical to the
    /// moe_head -> matvec_f32_batch -> moe_topk_scaled chain (the in-kernel
    /// logit walk reproduces the tile matvec's summation order exactly).
    #[allow(clippy::too_many_arguments)]
    pub fn moe_head_router(
        &self,
        x: &CudaSlice<f32>,
        gamma: &CudaSlice<f32>,
        pre2: &CudaSlice<f32>,
        rw: &crate::gpu::DeviceTensor,
        dscale: &CudaSlice<f32>,
        pn: &mut CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        qs: &mut CudaSlice<f32>,
        idx: &mut CudaSlice<u32>,
        w: &mut CudaSlice<f32>,
        n: usize,
        n_expert: usize,
        k: usize,
        eps: f32,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .moe_head_router
            .ok_or(GpuError::MissingOp("moe_head_router"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (gp, _g2) = gamma.device_ptr(&self.stream);
        let (pp, _g3) = pre2.device_ptr(&self.stream);
        let (wp, _g4) = rw.buf.device_ptr(&self.stream);
        let (dp, _g5) = dscale.device_ptr(&self.stream);
        let (np, _g6) = pn.device_ptr_mut(&self.stream);
        let (qp, _g7) = q.device_ptr_mut(&self.stream);
        let (sp, _g8) = qs.device_ptr_mut(&self.stream);
        let (ip, _g9) = idx.device_ptr_mut(&self.stream);
        let (owp, _g10) = w.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract
        check(unsafe {
            f(
                xp as *const _,
                gp as *const _,
                pp as *const _,
                wp as *const _,
                dp as *const _,
                np as *mut _,
                qp as *mut _,
                sp as *mut _,
                ip as *mut _,
                owp as *mut _,
                n as u32,
                n_expert as u32,
                k as u32,
                eps,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_moe_head_router(&self) -> bool {
        self.kernels.moe_head_router.is_some()
    }

    /// hibatch-lane hb twin of `moe_head_router` (8-token blocks, bf16 smem
    /// rows; precision-class - M1).
    #[allow(clippy::too_many_arguments)]
    pub fn moe_head_router_hb(
        &self,
        x: &CudaSlice<f32>,
        gamma: &CudaSlice<f32>,
        pre2: &CudaSlice<f32>,
        rw: &crate::gpu::DeviceTensor,
        dscale: &CudaSlice<f32>,
        pn: &mut CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        qs: &mut CudaSlice<f32>,
        idx: &mut CudaSlice<u32>,
        w: &mut CudaSlice<f32>,
        n: usize,
        n_expert: usize,
        k: usize,
        eps: f32,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .moe_head_router_hb
            .ok_or(GpuError::MissingOp("moe_head_router_hb"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (gp, _g2) = gamma.device_ptr(&self.stream);
        let (pp, _g3) = pre2.device_ptr(&self.stream);
        let (wp, _g4) = rw.buf.device_ptr(&self.stream);
        let (dp, _g5) = dscale.device_ptr(&self.stream);
        let (np, _g6) = pn.device_ptr_mut(&self.stream);
        let (qp, _g7) = q.device_ptr_mut(&self.stream);
        let (sp, _g8) = qs.device_ptr_mut(&self.stream);
        let (ip, _g9) = idx.device_ptr_mut(&self.stream);
        let (owp, _g10) = w.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract
        check(unsafe {
            f(
                xp as *const _,
                gp as *const _,
                pp as *const _,
                wp as *const _,
                dp as *const _,
                np as *mut _,
                qp as *mut _,
                sp as *mut _,
                ip as *mut _,
                owp as *mut _,
                n as u32,
                n_expert as u32,
                k as u32,
                eps,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_moe_head_router_hb(&self) -> bool {
        self.kernels.moe_head_router_hb.is_some()
    }

    /// P1-2: head twin with per-128 activation-scale quantize (lane).
    #[allow(clippy::too_many_arguments)]
    pub fn moe_head_xg(
        &self,
        x: &CudaSlice<f32>,
        gamma: &CudaSlice<f32>,
        pre2: &CudaSlice<f32>,
        rn: &mut CudaSlice<f32>,
        pn: &mut CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        qs: &mut CudaSlice<f32>,
        n: usize,
        eps: f32,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .moe_head_xg
            .ok_or(GpuError::MissingOp("moe_head_xg"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (gp, _g2) = gamma.device_ptr(&self.stream);
        let (pp, _g3) = pre2.device_ptr(&self.stream);
        let (rp, _g4) = rn.device_ptr_mut(&self.stream);
        let (np, _g5) = pn.device_ptr_mut(&self.stream);
        let (qp, _g6) = q.device_ptr_mut(&self.stream);
        let (sp, _g7) = qs.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract
        check(unsafe {
            f(
                xp as *const _,
                gp as *const _,
                pp as *const _,
                rp as *mut _,
                np as *mut _,
                qp as *mut _,
                sp as *mut _,
                n as u32,
                eps,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_moe_head_xg(&self) -> bool {
        self.kernels.moe_head_xg.is_some()
    }

    /// P1-2: mma2g gate_up (per-128 activation scales, group fold; lane).
    #[allow(clippy::too_many_arguments)]
    pub fn q8_0_moe_gate_up_mma2g_geglu(
        &self,
        gate: &RepackedQ8,
        up: &RepackedQ8,
        sorted_row: &CudaSlice<u32>,
        block_expert: &CudaSlice<u32>,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        fq: &mut CudaSlice<i8>,
        fs: &mut CudaSlice<f32>,
        max_blocks: usize,
        bm: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_moe_gate_up_mma2g_geglu
            .ok_or(GpuError::MissingOp("q8_0_moe_gate_up_mma2g_geglu"))?;
        let (in_dim, _ff) = (gate.dims[0], gate.dims[1]);
        let (gd, _g1) = gate.data.device_ptr(&self.stream);
        let (gs, _g2) = gate.scale.device_ptr(&self.stream);
        let (ud, _g3) = up.data.device_ptr(&self.stream);
        let (us, _g4) = up.scale.device_ptr(&self.stream);
        let (sr, _g5) = sorted_row.device_ptr(&self.stream);
        let (be, _g6) = block_expert.device_ptr(&self.stream);
        let (xqp, _g7) = xq.device_ptr(&self.stream);
        let (xsp, _g8) = xs.device_ptr(&self.stream);
        let (fqp, _g9) = fq.device_ptr_mut(&self.stream);
        let (fsp, _g10) = fs.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract
        check(unsafe {
            f(
                gd as *const _,
                gs as *const _,
                ud as *const _,
                us as *const _,
                sr as *const _,
                be as *const _,
                xqp as *const _,
                xsp as *const _,
                fqp as *mut _,
                fsp as *mut _,
                in_dim as u32,
                gate.dims[1] as u32,
                max_blocks as u32,
                bm as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_q8_moe_mma2g(&self) -> bool {
        self.kernels.q8_0_moe_gate_up_mma2g_geglu.is_some()
    }

    /// P1-1: down twin storing bf16 partials (lane; tail must read bf16).
    #[allow(clippy::too_many_arguments)]
    pub fn q8_0_moe_down_mma2_pbf16(
        &self,
        down: &RepackedQ8,
        sorted_row: &CudaSlice<u32>,
        sorted_slot: &CudaSlice<u32>,
        block_expert: &CudaSlice<u32>,
        topk_w: &CudaSlice<f32>,
        fq: &CudaSlice<i8>,
        fs: &CudaSlice<f32>,
        part: &mut CudaSlice<f32>,
        n_active: usize,
        max_blocks: usize,
        bm: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_moe_down_mma2_pbf16
            .ok_or(GpuError::MissingOp("q8_0_moe_down_mma2_pbf16"))?;
        let (ff, embd) = (down.dims[0], down.dims[1]);
        let (dd, _g1) = down.data.device_ptr(&self.stream);
        let (ds, _g2) = down.scale.device_ptr(&self.stream);
        let (rp, _g3) = sorted_row.device_ptr(&self.stream);
        let (slp, _g4) = sorted_slot.device_ptr(&self.stream);
        let (bp, _g5) = block_expert.device_ptr(&self.stream);
        let (wp, _g6) = topk_w.device_ptr(&self.stream);
        let (qp, _g7) = fq.device_ptr(&self.stream);
        let (sp, _g8) = fs.device_ptr(&self.stream);
        let (op, _g9) = part.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract
        check(unsafe {
            f(
                dd as *const _,
                ds as *const _,
                rp as *const _,
                slp as *const _,
                bp as *const _,
                wp as *const _,
                qp as *const _,
                sp as *const _,
                op as *mut _,
                ff as u32,
                embd as u32,
                n_active as u32,
                max_blocks as u32,
                bm as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_q8_moe_down_pbf16(&self) -> bool {
        self.kernels.q8_0_moe_down_mma2_pbf16.is_some()
    }

    /// P1 dn64: mma2g twin quantizing the GEGLU output per-64 (fs at ff/64).
    #[allow(clippy::too_many_arguments)]
    pub fn q8_0_moe_gate_up_mma2g_y64_geglu(
        &self,
        gate: &RepackedQ8,
        up: &RepackedQ8,
        sorted_row: &CudaSlice<u32>,
        block_expert: &CudaSlice<u32>,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        fq: &mut CudaSlice<i8>,
        fs: &mut CudaSlice<f32>,
        max_blocks: usize,
        bm: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_moe_gate_up_mma2g_y64_geglu
            .ok_or(GpuError::MissingOp("q8_0_moe_gate_up_mma2g_y64_geglu"))?;
        let (in_dim, _ff) = (gate.dims[0], gate.dims[1]);
        let (gd, _g1) = gate.data.device_ptr(&self.stream);
        let (gs, _g2) = gate.scale.device_ptr(&self.stream);
        let (ud, _g3) = up.data.device_ptr(&self.stream);
        let (us, _g4) = up.scale.device_ptr(&self.stream);
        let (sr, _g5) = sorted_row.device_ptr(&self.stream);
        let (be, _g6) = block_expert.device_ptr(&self.stream);
        let (xqp, _g7) = xq.device_ptr(&self.stream);
        let (xsp, _g8) = xs.device_ptr(&self.stream);
        let (fqp, _g9) = fq.device_ptr_mut(&self.stream);
        let (fsp, _g10) = fs.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract
        check(unsafe {
            f(
                gd as *const _,
                gs as *const _,
                ud as *const _,
                us as *const _,
                sr as *const _,
                be as *const _,
                xqp as *const _,
                xsp as *const _,
                fqp as *mut _,
                fsp as *mut _,
                in_dim as u32,
                gate.dims[1] as u32,
                max_blocks as u32,
                bm as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_q8_moe_mma2g_y64(&self) -> bool {
        self.kernels.q8_0_moe_gate_up_mma2g_y64_geglu.is_some()
    }

    /// P1 dn64: down twin consuming per-64 fs (pair-grouped fold); `pbf16`
    /// composes P1-1 (bf16 partials store).
    #[allow(clippy::too_many_arguments)]
    pub fn q8_0_moe_down_mma2_fs64(
        &self,
        down: &RepackedQ8,
        sorted_row: &CudaSlice<u32>,
        sorted_slot: &CudaSlice<u32>,
        block_expert: &CudaSlice<u32>,
        topk_w: &CudaSlice<f32>,
        fq: &CudaSlice<i8>,
        fs: &CudaSlice<f32>,
        part: &mut CudaSlice<f32>,
        n_active: usize,
        max_blocks: usize,
        bm: usize,
        pbf16: bool,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_moe_down_mma2_fs64
            .ok_or(GpuError::MissingOp("q8_0_moe_down_mma2_fs64"))?;
        let (ff, embd) = (down.dims[0], down.dims[1]);
        let (dd, _g1) = down.data.device_ptr(&self.stream);
        let (ds, _g2) = down.scale.device_ptr(&self.stream);
        let (rp, _g3) = sorted_row.device_ptr(&self.stream);
        let (slp, _g4) = sorted_slot.device_ptr(&self.stream);
        let (bp, _g5) = block_expert.device_ptr(&self.stream);
        let (wp, _g6) = topk_w.device_ptr(&self.stream);
        let (qp, _g7) = fq.device_ptr(&self.stream);
        let (sp, _g8) = fs.device_ptr(&self.stream);
        let (op, _g9) = part.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract
        check(unsafe {
            f(
                dd as *const _,
                ds as *const _,
                rp as *const _,
                slp as *const _,
                bp as *const _,
                wp as *const _,
                qp as *const _,
                sp as *const _,
                op as *mut _,
                ff as u32,
                embd as u32,
                n_active as u32,
                max_blocks as u32,
                bm as u32,
                pbf16 as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_q8_moe_down_fs64(&self) -> bool {
        self.kernels.q8_0_moe_down_mma2_fs64.is_some()
    }

    /// B3-1: cooperative router stage (matvec + topk, one kernel).
    #[allow(clippy::too_many_arguments)]
    pub fn moe_router_stage(
        &self,
        rw: &crate::gpu::DeviceTensor,
        x: &CudaSlice<f32>,
        logits: &mut CudaSlice<f32>,
        dscale: &CudaSlice<f32>,
        idx: &mut CudaSlice<u32>,
        w: &mut CudaSlice<f32>,
        in_dim: usize,
        out_dim: usize,
        batch: usize,
        k: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .moe_router_stage
            .ok_or(GpuError::MissingOp("moe_router_stage"))?;
        let (wp, _g1) = rw.buf.device_ptr(&self.stream);
        let (xp, _g2) = x.device_ptr(&self.stream);
        let (lp, _g3) = logits.device_ptr_mut(&self.stream);
        let (dp, _g4) = dscale.device_ptr(&self.stream);
        let (ip, _g5) = idx.device_ptr_mut(&self.stream);
        let (owp, _g6) = w.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract
        check(unsafe {
            f(
                wp as *const _,
                xp as *const _,
                lp as *mut _,
                dp as *const _,
                ip as *mut _,
                owp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                k as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_moe_router_stage(&self) -> bool {
        self.kernels.moe_router_stage.is_some()
    }

    /// Dual-weight MoE head: one sumsq serves the router norm AND the
    /// pre_ffw_norm_2 norm + its q8 quant (3 nodes -> 1).
    #[allow(clippy::too_many_arguments)]
    pub fn moe_head(
        &self,
        x: &CudaSlice<f32>,
        gamma: &CudaSlice<f32>,
        pre2: &CudaSlice<f32>,
        rn: &mut CudaSlice<f32>,
        pn: &mut CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        qs: &mut CudaSlice<f32>,
        n: usize,
        eps: f32,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .moe_head
            .ok_or(GpuError::MissingOp("moe_head"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (gp, _g2) = gamma.device_ptr(&self.stream);
        let (pp, _g3) = pre2.device_ptr(&self.stream);
        let (rp, _g4) = rn.device_ptr_mut(&self.stream);
        let (np, _g5) = pn.device_ptr_mut(&self.stream);
        let (qp, _g6) = q.device_ptr_mut(&self.stream);
        let (sp, _g7) = qs.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                xp as *const _,
                gp as *const _,
                pp as *const _,
                rp as *mut _,
                np as *mut _,
                qp as *mut _,
                sp as *mut _,
                n as u32,
                eps,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// DeepSeek-greedy router epilogue: the same top-k SELECTION as the other
    /// routers, but the weights are the full softmax probabilities -
    /// denominator over all `n_expert` logits, no renormalization among the
    /// selected k (`topk_method=greedy`, `norm_topk_prob=False`, the
    /// DeepSeek-V2/OCR class). The renormalizing kernels pick the same experts
    /// and differ by exactly the top-k's captured probability mass, so
    /// conflating the two is fluent and silently wrong.
    pub fn moe_topk_softmax_all_batch(
        &self,
        logits: &CudaSlice<f32>,
        n_expert: usize,
        k: usize,
        out_idx: &mut CudaSlice<u32>,
        out_w: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .moe_topk_softmax_all
            .ok_or(GpuError::MissingOp("moe_topk_softmax_all"))?;
        let (lp, _g1) = logits.device_ptr(&self.stream);
        let (ip, _g2) = out_idx.device_ptr_mut(&self.stream);
        let (wp, _g3) = out_w.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                lp as *const _,
                n_expert as u32,
                k as u32,
                ip as *mut _,
                wp as *mut _,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// topk + per-expert down-scale fold (bit-identical to the topk +
    /// moe_scale_w pair - same per-element ops in the same order).
    #[allow(clippy::too_many_arguments)]
    pub fn moe_topk_scaled(
        &self,
        logits: &CudaSlice<f32>,
        scale: &CudaSlice<f32>,
        n_expert: usize,
        k: usize,
        out_idx: &mut CudaSlice<u32>,
        out_w: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .moe_topk_scaled
            .ok_or(GpuError::MissingOp("moe_topk_scaled"))?;
        let (lp, _g1) = logits.device_ptr(&self.stream);
        let (sp, _g2) = scale.device_ptr(&self.stream);
        let (ip, _g3) = out_idx.device_ptr_mut(&self.stream);
        let (wp, _g4) = out_w.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                lp as *const _,
                sp as *const _,
                n_expert as u32,
                k as u32,
                ip as *mut _,
                wp as *mut _,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// MoE combine trailer: x = (x + rmsnorm(rmsnorm(proj)*pn1 +
    /// rmsnorm(dn)*pn2) * postw) * os in one launch (4 nodes -> 1).
    #[allow(clippy::too_many_arguments)]
    pub fn moe_tail(
        &self,
        x: &mut CudaSlice<f32>,
        proj: &CudaSlice<f32>,
        dn: &CudaSlice<f32>,
        pn1: &CudaSlice<f32>,
        pn2: &CudaSlice<f32>,
        postw: &CudaSlice<f32>,
        n: usize,
        eps: f32,
        os: f32,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .moe_tail
            .ok_or(GpuError::MissingOp("moe_tail"))?;
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (pp, _g2) = proj.device_ptr(&self.stream);
        let (dp, _g3) = dn.device_ptr(&self.stream);
        let (p1, _g4) = pn1.device_ptr(&self.stream);
        let (p2, _g5) = pn2.device_ptr(&self.stream);
        let (pw, _g6) = postw.device_ptr(&self.stream);
        check(unsafe {
            f(
                xp as *mut _,
                pp as *const _,
                dp as *const _,
                p1 as *const _,
                p2 as *const _,
                pw as *const _,
                n as u32,
                eps,
                os,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// True when the pack carries all three MoE tail fusions.
    pub fn has_moe_fusions(&self) -> bool {
        self.kernels.moe_head.is_some()
            && self.kernels.moe_topk_scaled.is_some()
            && self.kernels.moe_tail.is_some()
    }

    /// Decode-band intensity twins of the token-batched expert pair (4 rows
    /// per block; reorder class - greedy/coherence gates arbitrate).
    #[allow(clippy::too_many_arguments)]
    pub fn q8_0_moe_gu_dec2_geglu(
        &self,
        gate: &RepackedQ8,
        up: &RepackedQ8,
        idx: &CudaSlice<u32>,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n_active: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_moe_gu_dec2_geglu
            .ok_or(GpuError::MissingOp("q8_0_moe_gu_dec2_geglu"))?;
        let (in_dim, ff) = (gate.dims[0], gate.dims[1]);
        let (gd, _g1) = gate.data.device_ptr(&self.stream);
        let (gs, _g2) = gate.scale.device_ptr(&self.stream);
        let (ud, _g3) = up.data.device_ptr(&self.stream);
        let (us, _g4) = up.scale.device_ptr(&self.stream);
        let (ip, _g5) = idx.device_ptr(&self.stream);
        let (xp, _g6) = xq.device_ptr(&self.stream);
        let (sp, _g7) = xs.device_ptr(&self.stream);
        let (op, _g8) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                gd as *const _,
                gs as *const _,
                ud as *const _,
                us as *const _,
                ip as *const _,
                xp as *const _,
                sp as *const _,
                op as *mut _,
                in_dim as u32,
                ff as u32,
                n_active as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Decode-band single-plane relu^2 expert up: the
    /// nemotron_h_moe twin of `q8_0_moe_gu_dec2_geglu`. Warp per output row,
    /// no pad rows, no smem staging - the shape that fits decode widths,
    /// where the sorted BM=32 tile holds one real row in a 32-row block.
    ///
    /// It does not dedup: two rows routed to the same expert stream its plane
    /// twice, so the caller gates on the measured crossover with sorted.
    /// `rows_pb` 0 = the pack's elected rows-per-CTA (the engine always
    /// passes 0; nonzero is the lab sweep's instrument).
    ///
    /// Reorder class vs `q8_0_moe_up_relu2` - parity gates arbitrate.
    #[allow(clippy::too_many_arguments)]
    pub fn q8_0_moe_up_relu2_dec2(
        &self,
        up: &RepackedQ8,
        idx: &CudaSlice<u32>,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n_active: usize,
        batch: usize,
        rows_pb: u32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_moe_up_relu2_dec2
            .ok_or(GpuError::MissingOp("q8_0_moe_up_relu2_dec2"))?;
        let (in_dim, ff) = (up.dims[0], up.dims[1]);
        let (ud, _g1) = up.data.device_ptr(&self.stream);
        let (us, _g2) = up.scale.device_ptr(&self.stream);
        let (ip, _g3) = idx.device_ptr(&self.stream);
        let (xp, _g4) = xq.device_ptr(&self.stream);
        let (sp, _g5) = xs.device_ptr(&self.stream);
        let (op, _g6) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                ud as *const _,
                us as *const _,
                ip as *const _,
                xp as *const _,
                sp as *const _,
                op as *mut _,
                in_dim as u32,
                ff as u32,
                n_active as u32,
                batch as u32,
                rows_pb,
                self.stream_ptr(),
            )
        })
    }

    /// Both halves of the relu^2 decode-band pair are present (the up twin is
    /// a new slot; the down twin is qwen's, unchanged).
    pub fn has_q8_0_moe_relu2_dec2(&self) -> bool {
        self.kernels.q8_0_moe_up_relu2_dec2.is_some() && self.kernels.q8_0_moe_dn_dec2.is_some()
    }

    /// Decode-band down twin (see `q8_0_moe_gu_dec2_geglu`).
    #[allow(clippy::too_many_arguments)]
    pub fn q8_0_moe_dn_dec2(
        &self,
        down: &RepackedQ8,
        idx: &CudaSlice<u32>,
        topk_w: &CudaSlice<f32>,
        fq: &CudaSlice<i8>,
        fs: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n_active: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_moe_dn_dec2
            .ok_or(GpuError::MissingOp("q8_0_moe_dn_dec2"))?;
        let (ff, embd) = (down.dims[0], down.dims[1]);
        let (dd, _g1) = down.data.device_ptr(&self.stream);
        let (ds, _g2) = down.scale.device_ptr(&self.stream);
        let (ip, _g3) = idx.device_ptr(&self.stream);
        let (wp, _g4) = topk_w.device_ptr(&self.stream);
        let (qp, _g5) = fq.device_ptr(&self.stream);
        let (sp, _g6) = fs.device_ptr(&self.stream);
        let (op, _g7) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                dd as *const _,
                ds as *const _,
                ip as *const _,
                wp as *const _,
                qp as *const _,
                sp as *const _,
                op as *mut _,
                ff as u32,
                embd as u32,
                n_active as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// True when the pack carries the dec3 bulk-streamed decode-band trio
    /// (gu/dn/combine + the align that builds their BM=8 CSR). The pack
    /// NULLs the streamed pair per-device below sm_90, so this is an honest
    /// per-device capability.
    pub fn has_moe_dec3(&self) -> bool {
        self.kernels.q8_0_moe_gu_dec3_geglu.is_some()
            && self.kernels.q8_0_moe_dn_dec3.is_some()
            && self.kernels.moe_combine_dec3.is_some()
            && self.kernels.moe_align_bm.is_some()
    }

    /// dec3 gate+up+GEGLU: each touched expert's gate/up rows stream once
    /// through a cp.async.bulk ring and apply to the moe_align BM=8 block's
    /// routed rows. Output layout and per-row math are exactly
    /// `q8_0_moe_gu_dec2_geglu`'s (bitwise).
    #[allow(clippy::too_many_arguments)]
    pub fn q8_0_moe_gu_dec3_geglu(
        &self,
        gate: &RepackedQ8,
        up: &RepackedQ8,
        bexp: &CudaSlice<u32>,
        srow: &CudaSlice<u32>,
        sslot: &CudaSlice<u32>,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n_active: usize,
        max_blocks: usize,
        pairs: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_moe_gu_dec3_geglu
            .ok_or(GpuError::MissingOp("q8_0_moe_gu_dec3_geglu"))?;
        let (in_dim, ff) = (gate.dims[0], gate.dims[1]);
        let (gd, _g1) = gate.data.device_ptr(&self.stream);
        let (gs, _g2) = gate.scale.device_ptr(&self.stream);
        let (ud, _g3) = up.data.device_ptr(&self.stream);
        let (us, _g4) = up.scale.device_ptr(&self.stream);
        let (bp, _g5) = bexp.device_ptr(&self.stream);
        let (rp, _g6) = srow.device_ptr(&self.stream);
        let (lp, _g7) = sslot.device_ptr(&self.stream);
        let (xp, _g8) = xq.device_ptr(&self.stream);
        let (sp, _g9) = xs.device_ptr(&self.stream);
        let (op, _g10) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                gd as *const _,
                gs as *const _,
                ud as *const _,
                us as *const _,
                bp as *const _,
                rp as *const _,
                lp as *const _,
                xp as *const _,
                sp as *const _,
                op as *mut _,
                in_dim as u32,
                ff as u32,
                n_active as u32,
                max_blocks as u32,
                pairs as u32,
                self.stream_ptr(),
            )
        })
    }

    /// dec3 down: streamed like the gate_up half; writes per-(token, slot)
    /// partials (`topk_w * dot`) for `moe_combine_dec3` - the cross-slot sum
    /// is the reorder vs dec2 (per-dot math identical).
    #[allow(clippy::too_many_arguments)]
    pub fn q8_0_moe_dn_dec3(
        &self,
        down: &RepackedQ8,
        bexp: &CudaSlice<u32>,
        srow: &CudaSlice<u32>,
        sslot: &CudaSlice<u32>,
        topk_w: &CudaSlice<f32>,
        fq: &CudaSlice<i8>,
        fs: &CudaSlice<f32>,
        part: &mut CudaSlice<f32>,
        n_active: usize,
        max_blocks: usize,
        pairs: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_moe_dn_dec3
            .ok_or(GpuError::MissingOp("q8_0_moe_dn_dec3"))?;
        let (ff, embd) = (down.dims[0], down.dims[1]);
        let (dd, _g1) = down.data.device_ptr(&self.stream);
        let (ds, _g2) = down.scale.device_ptr(&self.stream);
        let (bp, _g3) = bexp.device_ptr(&self.stream);
        let (rp, _g4) = srow.device_ptr(&self.stream);
        let (lp, _g5) = sslot.device_ptr(&self.stream);
        let (wp, _g6) = topk_w.device_ptr(&self.stream);
        let (qp, _g7) = fq.device_ptr(&self.stream);
        let (sp, _g8) = fs.device_ptr(&self.stream);
        let (pp, _g9) = part.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                dd as *const _,
                ds as *const _,
                bp as *const _,
                rp as *const _,
                lp as *const _,
                wp as *const _,
                qp as *const _,
                sp as *const _,
                pp as *mut _,
                ff as u32,
                embd as u32,
                n_active as u32,
                max_blocks as u32,
                pairs as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Fixed-order combine of the dec3 down partials (dec2's slot-half sum
    /// tree; plain write, so no memset of `out` is needed).
    pub fn moe_combine_dec3(
        &self,
        part: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n: usize,
        n_active: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .moe_combine_dec3
            .ok_or(GpuError::MissingOp("moe_combine_dec3"))?;
        let (pp, _g1) = part.device_ptr(&self.stream);
        let (op, _g2) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                pp as *const _,
                op as *mut _,
                n as u32,
                n_active as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Sorted Q8_0 MoE gate+up+SwiGLU over the moe_align layout (each expert's
    /// weights read once per pass). `fused` is sorted-contiguous
    /// [max_blocks*32, ff], zeros on PAD rows.
    #[allow(clippy::too_many_arguments)]
    pub fn q8_0_moe_gate_up_sorted(
        &self,
        gate: &RepackedQ8,
        up: &RepackedQ8,
        sorted_row: &CudaSlice<u32>,
        block_expert: &CudaSlice<u32>,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        fused: &mut CudaSlice<f32>,
        max_blocks: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_moe_gate_up_sorted
            .ok_or(GpuError::MissingOp("q8_0_moe_gate_up_sorted"))?;
        let (in_dim, ff) = (gate.dims[0], gate.dims[1]);
        let (gd, _g1) = gate.data.device_ptr(&self.stream);
        let (gs, _g2) = gate.scale.device_ptr(&self.stream);
        let (ud, _g3) = up.data.device_ptr(&self.stream);
        let (us, _g4) = up.scale.device_ptr(&self.stream);
        let (rp, _g5) = sorted_row.device_ptr(&self.stream);
        let (bp, _g6) = block_expert.device_ptr(&self.stream);
        let (xp, _g7) = xq.device_ptr(&self.stream);
        let (sp, _g8) = xs.device_ptr(&self.stream);
        let (op, _g9) = fused.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                gd as *const _,
                gs as *const _,
                ud as *const _,
                us as *const _,
                rp as *const _,
                bp as *const _,
                xp as *const _,
                sp as *const _,
                op as *mut _,
                in_dim as u32,
                ff as u32,
                max_blocks as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Sorted single-plane up + squared-relu over the moe_align layout -
    /// the nemotron_h_moe prefill class. K-tail-guarded: in_dim only needs
    /// q8-block (32) alignment, not the 256 the gate_up kernel requires.
    #[allow(clippy::too_many_arguments)]
    pub fn q8_0_moe_up_relu2_sorted(
        &self,
        up: &RepackedQ8,
        sorted_row: &CudaSlice<u32>,
        block_expert: &CudaSlice<u32>,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        fused: &mut CudaSlice<f32>,
        max_blocks: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_moe_up_relu2_sorted
            .ok_or(GpuError::MissingOp("q8_0_moe_up_relu2_sorted"))?;
        let (in_dim, ff) = (up.dims[0], up.dims[1]);
        let (ud, _g1) = up.data.device_ptr(&self.stream);
        let (us, _g2) = up.scale.device_ptr(&self.stream);
        let (rp, _g3) = sorted_row.device_ptr(&self.stream);
        let (bp, _g4) = block_expert.device_ptr(&self.stream);
        let (xp, _g5) = xq.device_ptr(&self.stream);
        let (sp, _g6) = xs.device_ptr(&self.stream);
        let (op, _g7) = fused.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                ud as *const _,
                us as *const _,
                rp as *const _,
                bp as *const _,
                xp as *const _,
                sp as *const _,
                op as *mut _,
                in_dim as u32,
                ff as u32,
                max_blocks as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Sorted Q8_0 MoE down: per-(token, slot) weighted partials from the
    /// sorted-contiguous quantized SwiGLU output; fold with `moe_slot_combine`.
    #[allow(clippy::too_many_arguments)]
    pub fn q8_0_moe_down_sorted(
        &self,
        down: &RepackedQ8,
        sorted_row: &CudaSlice<u32>,
        sorted_slot: &CudaSlice<u32>,
        block_expert: &CudaSlice<u32>,
        topk_w: &CudaSlice<f32>,
        fq: &CudaSlice<i8>,
        fs: &CudaSlice<f32>,
        part: &mut CudaSlice<f32>,
        n_active: usize,
        max_blocks: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_moe_down_sorted
            .ok_or(GpuError::MissingOp("q8_0_moe_down_sorted"))?;
        let (ff, embd) = (down.dims[0], down.dims[1]);
        let (dd, _g1) = down.data.device_ptr(&self.stream);
        let (ds, _g2) = down.scale.device_ptr(&self.stream);
        let (rp, _g3) = sorted_row.device_ptr(&self.stream);
        let (slp, _g4) = sorted_slot.device_ptr(&self.stream);
        let (bp, _g5) = block_expert.device_ptr(&self.stream);
        let (wp, _g6) = topk_w.device_ptr(&self.stream);
        let (qp, _g7) = fq.device_ptr(&self.stream);
        let (sp, _g8) = fs.device_ptr(&self.stream);
        let (op, _g9) = part.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                dd as *const _,
                ds as *const _,
                rp as *const _,
                slp as *const _,
                bp as *const _,
                wp as *const _,
                qp as *const _,
                sp as *const _,
                op as *mut _,
                ff as u32,
                embd as u32,
                n_active as u32,
                max_blocks as u32,
                self.stream_ptr(),
            )
        })
    }

    /// int8-MMA sorted MoE gate+up with fused output quantize (fq/fs direct).
    #[allow(clippy::too_many_arguments)]
    pub fn q8_0_moe_gate_up_mma(
        &self,
        gate: &RepackedQ8,
        up: &RepackedQ8,
        sorted_row: &CudaSlice<u32>,
        block_expert: &CudaSlice<u32>,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        fq: &mut CudaSlice<i8>,
        fs: &mut CudaSlice<f32>,
        max_blocks: usize,
        bm: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_moe_gate_up_mma
            .ok_or(GpuError::MissingOp("q8_0_moe_gate_up_mma"))?;
        let (in_dim, ff) = (gate.dims[0], gate.dims[1]);
        let (gd, _g1) = gate.data.device_ptr(&self.stream);
        let (gs, _g2) = gate.scale.device_ptr(&self.stream);
        let (ud, _g3) = up.data.device_ptr(&self.stream);
        let (us, _g4) = up.scale.device_ptr(&self.stream);
        let (rp, _g5) = sorted_row.device_ptr(&self.stream);
        let (bp, _g6) = block_expert.device_ptr(&self.stream);
        let (xp, _g7) = xq.device_ptr(&self.stream);
        let (sp, _g8) = xs.device_ptr(&self.stream);
        let (qp, _g9) = fq.device_ptr_mut(&self.stream);
        let (fp, _g10) = fs.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                gd as *const _,
                gs as *const _,
                ud as *const _,
                us as *const _,
                rp as *const _,
                bp as *const _,
                xp as *const _,
                sp as *const _,
                qp as *mut _,
                fp as *mut _,
                in_dim as u32,
                ff as u32,
                max_blocks as u32,
                bm as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Single-plane relu(up)^2 twin of `q8_0_moe_gate_up_mma` (slot 743,
    /// nemotron_h_moe): fq/fs straight from the int8 MMA, bitwise vs
    /// `q8_0_moe_up_relu2_sorted` + `quantize_q8`.
    #[allow(clippy::too_many_arguments)]
    pub fn q8_0_moe_up_relu2_mma(
        &self,
        up: &RepackedQ8,
        sorted_row: &CudaSlice<u32>,
        block_expert: &CudaSlice<u32>,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        fq: &mut CudaSlice<i8>,
        fs: &mut CudaSlice<f32>,
        max_blocks: usize,
        bm: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_moe_up_relu2_mma
            .ok_or(GpuError::MissingOp("q8_0_moe_up_relu2_mma"))?;
        let (in_dim, ff) = (up.dims[0], up.dims[1]);
        let (ud, _g1) = up.data.device_ptr(&self.stream);
        let (us, _g2) = up.scale.device_ptr(&self.stream);
        let (rp, _g3) = sorted_row.device_ptr(&self.stream);
        let (bp, _g4) = block_expert.device_ptr(&self.stream);
        let (xp, _g5) = xq.device_ptr(&self.stream);
        let (sp, _g6) = xs.device_ptr(&self.stream);
        let (qp, _g7) = fq.device_ptr_mut(&self.stream);
        let (fp, _g8) = fs.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                ud as *const _,
                us as *const _,
                rp as *const _,
                bp as *const _,
                xp as *const _,
                sp as *const _,
                qp as *mut _,
                fp as *mut _,
                in_dim as u32,
                ff as u32,
                max_blocks as u32,
                bm as u32,
                self.stream_ptr(),
            )
        })
    }

    /// GEGLU twin of `q8_0_moe_gate_up_mma` (gemma4-A4B sorted expert class).
    #[allow(clippy::too_many_arguments)]
    pub fn q8_0_moe_gate_up_mma_geglu(
        &self,
        gate: &RepackedQ8,
        up: &RepackedQ8,
        sorted_row: &CudaSlice<u32>,
        block_expert: &CudaSlice<u32>,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        fq: &mut CudaSlice<i8>,
        fs: &mut CudaSlice<f32>,
        max_blocks: usize,
        bm: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_moe_gate_up_mma_geglu
            .ok_or(GpuError::MissingOp("q8_0_moe_gate_up_mma_geglu"))?;
        let (in_dim, ff) = (gate.dims[0], gate.dims[1]);
        let (gd, _g1) = gate.data.device_ptr(&self.stream);
        let (gs, _g2) = gate.scale.device_ptr(&self.stream);
        let (ud, _g3) = up.data.device_ptr(&self.stream);
        let (us, _g4) = up.scale.device_ptr(&self.stream);
        let (rp, _g5) = sorted_row.device_ptr(&self.stream);
        let (bp, _g6) = block_expert.device_ptr(&self.stream);
        let (xp, _g7) = xq.device_ptr(&self.stream);
        let (sp, _g8) = xs.device_ptr(&self.stream);
        let (qp, _g9) = fq.device_ptr_mut(&self.stream);
        let (fp, _g10) = fs.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                gd as *const _,
                gs as *const _,
                ud as *const _,
                us as *const _,
                rp as *const _,
                bp as *const _,
                xp as *const _,
                sp as *const _,
                qp as *mut _,
                fp as *mut _,
                in_dim as u32,
                ff as u32,
                max_blocks as u32,
                bm as u32,
                self.stream_ptr(),
            )
        })
    }

    /// v2 ring twin of `q8_0_moe_gate_up_mma_geglu` (slot 483,
    /// bitwise on live outputs, bm must be 32.
    #[allow(clippy::too_many_arguments)]
    pub fn q8_0_moe_gate_up_mma2_geglu(
        &self,
        gate: &RepackedQ8,
        up: &RepackedQ8,
        sorted_row: &CudaSlice<u32>,
        block_expert: &CudaSlice<u32>,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        fq: &mut CudaSlice<i8>,
        fs: &mut CudaSlice<f32>,
        max_blocks: usize,
        bm: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_moe_gate_up_mma2_geglu
            .ok_or(GpuError::MissingOp("q8_0_moe_gate_up_mma2_geglu"))?;
        let (in_dim, ff) = (gate.dims[0], gate.dims[1]);
        let (gd, _g1) = gate.data.device_ptr(&self.stream);
        let (gs, _g2) = gate.scale.device_ptr(&self.stream);
        let (ud, _g3) = up.data.device_ptr(&self.stream);
        let (us, _g4) = up.scale.device_ptr(&self.stream);
        let (rp, _g5) = sorted_row.device_ptr(&self.stream);
        let (bp, _g6) = block_expert.device_ptr(&self.stream);
        let (xp, _g7) = xq.device_ptr(&self.stream);
        let (sp, _g8) = xs.device_ptr(&self.stream);
        let (qp, _g9) = fq.device_ptr_mut(&self.stream);
        let (fp, _g10) = fs.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                gd as *const _,
                gs as *const _,
                ud as *const _,
                us as *const _,
                rp as *const _,
                bp as *const _,
                xp as *const _,
                sp as *const _,
                qp as *mut _,
                fp as *mut _,
                in_dim as u32,
                ff as u32,
                max_blocks as u32,
                bm as u32,
                self.stream_ptr(),
            )
        })
    }

    /// v2 ring twin of `q8_0_moe_gate_up_mma_geglu` (slot 483,
    /// bitwise on live outputs, bm must be 32.
    #[allow(clippy::too_many_arguments)]
    pub fn q8_0_moe_gate_up_mma3_geglu(
        &self,
        gate: &RepackedQ8,
        up: &RepackedQ8,
        sorted_row: &CudaSlice<u32>,
        block_expert: &CudaSlice<u32>,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        fq: &mut CudaSlice<i8>,
        fs: &mut CudaSlice<f32>,
        max_blocks: usize,
        bm: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_moe_gate_up_mma3_geglu
            .ok_or(GpuError::MissingOp("q8_0_moe_gate_up_mma3_geglu"))?;
        let (in_dim, ff) = (gate.dims[0], gate.dims[1]);
        let (gd, _g1) = gate.data.device_ptr(&self.stream);
        let (gs, _g2) = gate.scale.device_ptr(&self.stream);
        let (ud, _g3) = up.data.device_ptr(&self.stream);
        let (us, _g4) = up.scale.device_ptr(&self.stream);
        let (rp, _g5) = sorted_row.device_ptr(&self.stream);
        let (bp, _g6) = block_expert.device_ptr(&self.stream);
        let (xp, _g7) = xq.device_ptr(&self.stream);
        let (sp, _g8) = xs.device_ptr(&self.stream);
        let (qp, _g9) = fq.device_ptr_mut(&self.stream);
        let (fp, _g10) = fs.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                gd as *const _,
                gs as *const _,
                ud as *const _,
                us as *const _,
                rp as *const _,
                bp as *const _,
                xp as *const _,
                sp as *const _,
                qp as *mut _,
                fp as *mut _,
                in_dim as u32,
                ff as u32,
                max_blocks as u32,
                bm as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_q8_moe_mma3(&self) -> bool {
        self.kernels.q8_0_moe_gate_up_mma3_geglu.is_some()
    }

    /// Flat-scale (per-output-ROW) e4m3 twin of `q8_0_moe_gate_up_mma_geglu`
    /// Weight scales are loop-invariant here, so the k walk
    /// carries none - that is the whole reason this kernel exists. The
    /// activation side stays per-32 (`quantize_e4m3_b32f`), so `xq`/`xs` have
    /// exactly the layout `quantize_q8` writes, and `fq`/`fs` come out as the
    /// same int8 pair `q8_0_moe_down_mma` already eats.
    #[allow(clippy::too_many_arguments)]
    pub fn f8row_moe_gate_up_mma_geglu(
        &self,
        gate: &F8RowPlane,
        up: &F8RowPlane,
        sorted_row: &CudaSlice<u32>,
        block_expert: &CudaSlice<u32>,
        xq: &CudaSlice<u8>,
        xs: &CudaSlice<f32>,
        fq: &mut CudaSlice<i8>,
        fs: &mut CudaSlice<f32>,
        in_dim: usize,
        ff: usize,
        max_blocks: usize,
        bm: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .f8row_moe_gate_up_mma_geglu
            .ok_or(GpuError::MissingOp("f8row_moe_gate_up_mma_geglu"))?;
        let (gd, _g1) = gate.data.device_ptr(&self.stream);
        let (gs, _g2) = gate.scale.device_ptr(&self.stream);
        let (ud, _g3) = up.data.device_ptr(&self.stream);
        let (us, _g4) = up.scale.device_ptr(&self.stream);
        let (rp, _g5) = sorted_row.device_ptr(&self.stream);
        let (bp, _g6) = block_expert.device_ptr(&self.stream);
        let (xp, _g7) = xq.device_ptr(&self.stream);
        let (sp, _g8) = xs.device_ptr(&self.stream);
        let (qp, _g9) = fq.device_ptr_mut(&self.stream);
        let (fp, _g10) = fs.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                gd as *const _,
                gs as *const _,
                ud as *const _,
                us as *const _,
                rp as *const _,
                bp as *const _,
                xp as *const _,
                sp as *const _,
                qp as *mut _,
                fp as *mut _,
                in_dim as u32,
                ff as u32,
                max_blocks as u32,
                bm as u32,
                self.stream_ptr(),
            )
        })
    }

    /// e4m3-output twin of `f8row_moe_gate_up_mma_geglu`: identical GEMM, the
    /// epilogue hands the flat-scale down half e4m3 per-32 instead of int8
    /// per-32 (same `fs` plane, same buffer sizes).
    #[allow(clippy::too_many_arguments)]
    pub fn f8row_moe_gate_up_mma_geglu_f8(
        &self,
        gate: &F8RowPlane,
        up: &F8RowPlane,
        sorted_row: &CudaSlice<u32>,
        block_expert: &CudaSlice<u32>,
        xq: &CudaSlice<u8>,
        xs: &CudaSlice<f32>,
        fq: &mut CudaSlice<i8>,
        fs: &mut CudaSlice<f32>,
        in_dim: usize,
        ff: usize,
        max_blocks: usize,
        bm: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .f8row_moe_gate_up_mma_geglu_f8
            .ok_or(GpuError::MissingOp("f8row_moe_gate_up_mma_geglu_f8"))?;
        let (gd, _g1) = gate.data.device_ptr(&self.stream);
        let (gs, _g2) = gate.scale.device_ptr(&self.stream);
        let (ud, _g3) = up.data.device_ptr(&self.stream);
        let (us, _g4) = up.scale.device_ptr(&self.stream);
        let (rp, _g5) = sorted_row.device_ptr(&self.stream);
        let (bp, _g6) = block_expert.device_ptr(&self.stream);
        let (xp, _g7) = xq.device_ptr(&self.stream);
        let (sp, _g8) = xs.device_ptr(&self.stream);
        let (qp, _g9) = fq.device_ptr_mut(&self.stream);
        let (fp, _g10) = fs.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                gd as *const _,
                gs as *const _,
                ud as *const _,
                us as *const _,
                rp as *const _,
                bp as *const _,
                xp as *const _,
                sp as *const _,
                qp as *mut _,
                fp as *mut _,
                in_dim as u32,
                ff as u32,
                max_blocks as u32,
                bm as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Flat-scale twin of `q8_0_moe_down_mma`: e4m3 weights with one scale per
    /// output row against the e4m3 per-32 `fq`/`fs` the f8-out gate_up wrote.
    #[allow(clippy::too_many_arguments)]
    pub fn f8row_moe_down_mma(
        &self,
        down: &F8RowPlane,
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
        bm: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .f8row_moe_down_mma
            .ok_or(GpuError::MissingOp("f8row_moe_down_mma"))?;
        let (dd, _g1) = down.data.device_ptr(&self.stream);
        let (dr, _g2) = down.scale.device_ptr(&self.stream);
        let (rp, _g3) = sorted_row.device_ptr(&self.stream);
        let (lp, _g4) = sorted_slot.device_ptr(&self.stream);
        let (bp, _g5) = block_expert.device_ptr(&self.stream);
        let (wp, _g6) = topk_w.device_ptr(&self.stream);
        let (qp, _g7) = fq.device_ptr(&self.stream);
        let (sp, _g8) = fs.device_ptr(&self.stream);
        let (pp, _g9) = part.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                dd as *const _,
                dr as *const _,
                rp as *const _,
                lp as *const _,
                bp as *const _,
                wp as *const _,
                qp as *const _,
                sp as *const _,
                pp as *mut _,
                ff as u32,
                embd as u32,
                n_active as u32,
                max_blocks as u32,
                bm as u32,
                self.stream_ptr(),
            )
        })
    }

    /// f32 -> e4m3 with one f32 scale per 32 elements: the activation half of
    /// the flat-scale expert lane. Same plane shape as `quantize_q8`.
    pub fn quantize_e4m3_b32f(
        &self,
        x: &CudaSlice<f32>,
        q: &mut CudaSlice<u8>,
        scale: &mut CudaSlice<f32>,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .quantize_e4m3_b32f
            .ok_or(GpuError::MissingOp("quantize_e4m3_b32f"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (qp, _g2) = q.device_ptr_mut(&self.stream);
        let (sp, _g3) = scale.device_ptr_mut(&self.stream);
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

    /// True when the flat-scale e4m3 expert lane can serve: the GEMM, its
    /// activation quantizer, and the load-time weight converter.
    pub fn has_f8row_moe(&self) -> bool {
        self.kernels.f8row_moe_gate_up_mma_geglu.is_some()
            && self.kernels.quantize_e4m3_b32f.is_some()
            && self.kernels.q8_0_to_f8row.is_some()
    }

    /// True when the down half of the flat-scale lane is present too, so the
    /// whole expert pair can run e4m3 (gate_up must use the f8-out epilogue).
    pub fn has_f8row_moe_down(&self) -> bool {
        self.has_f8row_moe()
            && self.kernels.f8row_moe_down_mma.is_some()
            && self.kernels.f8row_moe_gate_up_mma_geglu_f8.is_some()
    }

    /// True when the pack carries the v2 ring twins (slots 483/484).
    pub fn has_q8_moe_qmma2(&self) -> bool {
        self.kernels.q8_0_moe_gate_up_mma2_geglu.is_some()
            && self.kernels.q8_0_moe_down_mma2.is_some()
    }

    /// True when the pack carries nemotron's int8-MMA sorted MoE pair: the
    /// relu^2 up twin (slot 743) and the shared mma down.
    pub fn has_q8_moe_relu2_mma(&self) -> bool {
        self.kernels.q8_0_moe_up_relu2_mma.is_some() && self.kernels.q8_0_moe_down_mma.is_some()
    }

    /// True when the pack carries the gemma4 sorted MoE class (geglu mma
    /// gate_up + the shared down/align/combine set).
    pub fn has_q8_moe_geglu_sorted(&self) -> bool {
        self.kernels.q8_0_moe_gate_up_mma_geglu.is_some()
            && self.kernels.q8_0_moe_down_mma.is_some()
            && self.kernels.moe_align.is_some()
            && self.kernels.moe_slot_combine.is_some()
    }

    /// True when the tcgen05 e4m3 grouped-MoE family can SERVE here: table
    /// entries present AND cc-10 (the launchers return NotSupported off it).
    pub fn has_f8bs_moe(&self) -> bool {
        self.compute_capability().0 == 10
            && self.kernels.f8bs_moe_gemm_gu.is_some()
            && self.kernels.f8bs_moe_gemm_dn.is_some()
            && self.kernels.moe_gather_e4m3.is_some()
            && self.kernels.quantize_e4m3_geglu2_pad.is_some()
            && self.kernels.q8_0_to_f8w_pad.is_some()
            && self.kernels.moe_align_bm.is_some()
    }

    /// Q8_0 -> per-32 e4m3 planes with a K-tail pad (zero blocks). `bpr` =
    /// live blocks per row in the source, `bpr_pad` the padded row stride.
    pub fn q8_0_to_f8w_pad(
        &self,
        w: &RepackedQ8,
        bpr: usize,
        bpr_pad: usize,
    ) -> Result<RepackedMxfp4, GpuError> {
        let f = self
            .kernels
            .q8_0_to_f8w_pad
            .ok_or(GpuError::MissingOp("q8_0_to_f8w_pad"))?;
        let rows = w.data.len() / 32 / bpr;
        let mut data = self.alloc_u8(rows * bpr_pad * 32)?;
        let mut scale = self.alloc_u8(rows * bpr_pad)?;
        {
            let (qdp, _g1) = w.data.device_ptr(&self.stream);
            let (qsp, _g2) = w.scale.device_ptr(&self.stream);
            let (dp, _g3) = data.device_ptr_mut(&self.stream);
            let (sp, _g4) = scale.device_ptr_mut(&self.stream);
            check(unsafe {
                f(
                    qdp as *const _,
                    qsp as *const _,
                    dp as *mut _,
                    sp as *mut _,
                    rows as u64,
                    bpr as u32,
                    bpr_pad as u32,
                    self.stream_ptr(),
                )
            })?;
        }
        Ok(RepackedMxfp4 { data, scale })
    }

    /// Sorted gather of e4m3 activations + ue8m0 scales (PAD rows zeroed).
    #[allow(clippy::too_many_arguments)]
    pub fn moe_gather_e4m3(
        &self,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<u8>,
        srow: &CudaSlice<u32>,
        xg: &mut CudaSlice<u8>,
        sg: &mut CudaSlice<u8>,
        in_dim: usize,
        srows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .moe_gather_e4m3
            .ok_or(GpuError::MissingOp("moe_gather_e4m3"))?;
        let (xp, _g1) = xq.device_ptr(&self.stream);
        let (sp, _g2) = xs.device_ptr(&self.stream);
        let (rp, _g3) = srow.device_ptr(&self.stream);
        let (gp, _g4) = xg.device_ptr_mut(&self.stream);
        let (sgp, _g5) = sg.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                xp as *const _,
                sp as *const _,
                rp as *const _,
                gp as *mut _,
                sgp as *mut _,
                in_dim as u32,
                srows as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Fused-plane GEGLU quantize with a padded output row stride (the
    /// caller owns the plane's standing zero K-tail).
    pub fn quantize_e4m3_geglu2_pad(
        &self,
        gu: &CudaSlice<f32>,
        q: &mut CudaSlice<u8>,
        scale: &mut CudaSlice<u8>,
        n_ff: usize,
        n_ff_pad: usize,
        rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .quantize_e4m3_geglu2_pad
            .ok_or(GpuError::MissingOp("quantize_e4m3_geglu2_pad"))?;
        let (gp, _g1) = gu.device_ptr(&self.stream);
        let (qp, _g2) = q.device_ptr_mut(&self.stream);
        let (sp, _g3) = scale.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                gp as *const _,
                qp as *mut _,
                sp as *mut _,
                n_ff as u32,
                n_ff_pad as u32,
                rows as u32,
                self.stream_ptr(),
            )
        })
    }

    /// tcgen05 e4m3 grouped MoE gate_up over fused per-expert planes.
    #[allow(clippy::too_many_arguments)]
    pub fn f8bs_moe_gemm_gu(
        &self,
        w: &RepackedMxfp4,
        xg: &CudaSlice<u8>,
        sg: &CudaSlice<u8>,
        bexp: &CudaSlice<u32>,
        y: &mut CudaSlice<f32>,
        in_dim: usize,
        rows_per_e: usize,
        n_expert: usize,
        srows_pad: usize,
        max_blocks: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .f8bs_moe_gemm_gu
            .ok_or(GpuError::MissingOp("f8bs_moe_gemm_gu"))?;
        let (wd, _g1) = w.data.device_ptr(&self.stream);
        let (ws, _g2) = w.scale.device_ptr(&self.stream);
        let (xp, _g3) = xg.device_ptr(&self.stream);
        let (sp, _g4) = sg.device_ptr(&self.stream);
        let (bp, _g5) = bexp.device_ptr(&self.stream);
        let (yp, _g6) = y.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                wd as *const _,
                ws as *const _,
                xp as *const _,
                sp as *const _,
                bp as *const _,
                yp as *mut _,
                in_dim as u32,
                rows_per_e as u32,
                n_expert as u32,
                srows_pad as u32,
                max_blocks as u32,
                self.stream_ptr(),
            )
        })
    }

    /// tcgen05 e4m3 grouped MoE down (scattered topk_w epilogue).
    #[allow(clippy::too_many_arguments)]
    pub fn f8bs_moe_gemm_dn(
        &self,
        w: &RepackedMxfp4,
        fq8: &CudaSlice<u8>,
        fs8: &CudaSlice<u8>,
        bexp: &CudaSlice<u32>,
        srow: &CudaSlice<u32>,
        sslot: &CudaSlice<u32>,
        topk_w: &CudaSlice<f32>,
        part: &mut CudaSlice<f32>,
        in_dim: usize,
        rows_per_e: usize,
        n_expert: usize,
        srows_pad: usize,
        max_blocks: usize,
        n_active: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .f8bs_moe_gemm_dn
            .ok_or(GpuError::MissingOp("f8bs_moe_gemm_dn"))?;
        let (wd, _g1) = w.data.device_ptr(&self.stream);
        let (ws, _g2) = w.scale.device_ptr(&self.stream);
        let (xp, _g3) = fq8.device_ptr(&self.stream);
        let (sp, _g4) = fs8.device_ptr(&self.stream);
        let (bp, _g5) = bexp.device_ptr(&self.stream);
        let (rp, _g6) = srow.device_ptr(&self.stream);
        let (lp, _g7) = sslot.device_ptr(&self.stream);
        let (tp, _g8) = topk_w.device_ptr(&self.stream);
        let (pp, _g9) = part.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                wd as *const _,
                ws as *const _,
                xp as *const _,
                sp as *const _,
                bp as *const _,
                rp as *const _,
                lp as *const _,
                tp as *const _,
                pp as *mut _,
                in_dim as u32,
                rows_per_e as u32,
                n_expert as u32,
                srows_pad as u32,
                max_blocks as u32,
                n_active as u32,
                self.stream_ptr(),
            )
        })
    }

    /// True when the decode-band f8 shapes can serve (BM=32 gu +
    /// Y-resident dn + PAD-aware geglu; pack NULLs them off cc 10).
    pub fn has_f8d_moe(&self) -> bool {
        self.has_f8bs_moe()
            && self.kernels.f8bs_moe_gemm_gu_d32.is_some()
            && self.kernels.f8bs_moe_gemm_dn_d32.is_some()
            && self.kernels.quantize_e4m3_geglu2_pad_b.is_some()
    }

    /// Decode-band BM=32 grouped tc5 gate_up. Same contract as
    /// `f8bs_moe_gemm_gu` with the align/gather done at bm=32.
    #[allow(clippy::too_many_arguments)]
    pub fn f8bs_moe_gemm_gu_d32(
        &self,
        w: &RepackedMxfp4,
        xg: &CudaSlice<u8>,
        sg: &CudaSlice<u8>,
        bexp: &CudaSlice<u32>,
        y: &mut CudaSlice<f32>,
        in_dim: usize,
        rows_per_e: usize,
        n_expert: usize,
        srows_pad: usize,
        max_blocks: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .f8bs_moe_gemm_gu_d32
            .ok_or(GpuError::MissingOp("f8bs_moe_gemm_gu_d32"))?;
        let (wd, _g1) = w.data.device_ptr(&self.stream);
        let (ws, _g2) = w.scale.device_ptr(&self.stream);
        let (xp, _g3) = xg.device_ptr(&self.stream);
        let (sp, _g4) = sg.device_ptr(&self.stream);
        let (bp, _g5) = bexp.device_ptr(&self.stream);
        let (yp, _g6) = y.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                wd as *const _,
                ws as *const _,
                xp as *const _,
                sp as *const _,
                bp as *const _,
                yp as *mut _,
                in_dim as u32,
                rows_per_e as u32,
                n_expert as u32,
                srows_pad as u32,
                max_blocks as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Decode-band Y-resident BM=32 down: live outputs bitwise the
    /// BM=128 dn. PADDOCK_MOE_F8D_OTL retunes the out-tiles-per-CTA walk.
    #[allow(clippy::too_many_arguments)]
    pub fn f8bs_moe_gemm_dn_d32(
        &self,
        w: &RepackedMxfp4,
        fq8: &CudaSlice<u8>,
        fs8: &CudaSlice<u8>,
        bexp: &CudaSlice<u32>,
        srow: &CudaSlice<u32>,
        sslot: &CudaSlice<u32>,
        topk_w: &CudaSlice<f32>,
        part: &mut CudaSlice<f32>,
        in_dim: usize,
        rows_per_e: usize,
        n_expert: usize,
        srows_pad: usize,
        max_blocks: usize,
        n_active: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .f8bs_moe_gemm_dn_d32
            .ok_or(GpuError::MissingOp("f8bs_moe_gemm_dn_d32"))?;
        let (wd, _g1) = w.data.device_ptr(&self.stream);
        let (ws, _g2) = w.scale.device_ptr(&self.stream);
        let (xp, _g3) = fq8.device_ptr(&self.stream);
        let (sp, _g4) = fs8.device_ptr(&self.stream);
        let (bp, _g5) = bexp.device_ptr(&self.stream);
        let (rp, _g6) = srow.device_ptr(&self.stream);
        let (lp, _g7) = sslot.device_ptr(&self.stream);
        let (tp, _g8) = topk_w.device_ptr(&self.stream);
        let (pp, _g9) = part.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                wd as *const _,
                ws as *const _,
                xp as *const _,
                sp as *const _,
                bp as *const _,
                rp as *const _,
                lp as *const _,
                tp as *const _,
                pp as *mut _,
                in_dim as u32,
                rows_per_e as u32,
                n_expert as u32,
                srows_pad as u32,
                max_blocks as u32,
                n_active as u32,
                self.stream_ptr(),
            )
        })
    }

    /// PAD-block-aware fused GEGLU quantize: rows whose bm-block is
    /// PAD retire after one bexp load.
    #[allow(clippy::too_many_arguments)]
    pub fn quantize_e4m3_geglu2_pad_b(
        &self,
        gu: &CudaSlice<f32>,
        q: &mut CudaSlice<u8>,
        scale: &mut CudaSlice<u8>,
        bexp: &CudaSlice<u32>,
        n_ff: usize,
        n_ff_pad: usize,
        bm: usize,
        rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .quantize_e4m3_geglu2_pad_b
            .ok_or(GpuError::MissingOp("quantize_e4m3_geglu2_pad_b"))?;
        let (gp, _g1) = gu.device_ptr(&self.stream);
        let (qp, _g2) = q.device_ptr_mut(&self.stream);
        let (sp, _g3) = scale.device_ptr_mut(&self.stream);
        let (bp, _g4) = bexp.device_ptr(&self.stream);
        check(unsafe {
            f(
                gp as *const _,
                qp as *mut _,
                sp as *mut _,
                bp as *const _,
                n_ff as u32,
                n_ff_pad as u32,
                bm as u32,
                rows as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Routing diagnostic: accumulate the uniq-experts histogram
    /// for one (tick, layer) into a persistent RAW device buffer (non-pool
    /// cuMemAlloc - see the gemma4 Scratch notes). Launch-only,
    /// so captured decode graphs bake it in. Read-only on `idx`.
    pub fn moe_uniq_hist(
        &self,
        idx: &CudaSlice<u32>,
        pairs: usize,
        n_expert: usize,
        out_dev: u64,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .moe_uniq_hist
            .ok_or(GpuError::MissingOp("moe_uniq_hist"))?;
        let (ip, _g1) = idx.device_ptr(&self.stream);
        check(unsafe {
            f(
                ip as *const _,
                pairs as u32,
                n_expert as u32,
                out_dev as *mut _,
                self.stream_ptr(),
            )
        })
    }

    /// int8-MMA sorted MoE down (deterministic per-(token, slot) partials).
    #[allow(clippy::too_many_arguments)]
    pub fn q8_0_moe_down_mma(
        &self,
        down: &RepackedQ8,
        sorted_row: &CudaSlice<u32>,
        sorted_slot: &CudaSlice<u32>,
        block_expert: &CudaSlice<u32>,
        topk_w: &CudaSlice<f32>,
        fq: &CudaSlice<i8>,
        fs: &CudaSlice<f32>,
        part: &mut CudaSlice<f32>,
        n_active: usize,
        max_blocks: usize,
        bm: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_moe_down_mma
            .ok_or(GpuError::MissingOp("q8_0_moe_down_mma"))?;
        let (ff, embd) = (down.dims[0], down.dims[1]);
        let (dd, _g1) = down.data.device_ptr(&self.stream);
        let (ds, _g2) = down.scale.device_ptr(&self.stream);
        let (rp, _g3) = sorted_row.device_ptr(&self.stream);
        let (slp, _g4) = sorted_slot.device_ptr(&self.stream);
        let (bp, _g5) = block_expert.device_ptr(&self.stream);
        let (wp, _g6) = topk_w.device_ptr(&self.stream);
        let (qp, _g7) = fq.device_ptr(&self.stream);
        let (sp, _g8) = fs.device_ptr(&self.stream);
        let (op, _g9) = part.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                dd as *const _,
                ds as *const _,
                rp as *const _,
                slp as *const _,
                bp as *const _,
                wp as *const _,
                qp as *const _,
                sp as *const _,
                op as *mut _,
                ff as u32,
                embd as u32,
                n_active as u32,
                max_blocks as u32,
                bm as u32,
                self.stream_ptr(),
            )
        })
    }

    /// v2 ring twin of `q8_0_moe_down_mma` (slot 484): bitwise, bm must be
    /// 32, ff % 64 == 0.
    #[allow(clippy::too_many_arguments)]
    pub fn q8_0_moe_down_mma2(
        &self,
        down: &RepackedQ8,
        sorted_row: &CudaSlice<u32>,
        sorted_slot: &CudaSlice<u32>,
        block_expert: &CudaSlice<u32>,
        topk_w: &CudaSlice<f32>,
        fq: &CudaSlice<i8>,
        fs: &CudaSlice<f32>,
        part: &mut CudaSlice<f32>,
        n_active: usize,
        max_blocks: usize,
        bm: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_moe_down_mma2
            .ok_or(GpuError::MissingOp("q8_0_moe_down_mma2"))?;
        let (ff, embd) = (down.dims[0], down.dims[1]);
        let (dd, _g1) = down.data.device_ptr(&self.stream);
        let (ds, _g2) = down.scale.device_ptr(&self.stream);
        let (rp, _g3) = sorted_row.device_ptr(&self.stream);
        let (slp, _g4) = sorted_slot.device_ptr(&self.stream);
        let (bp, _g5) = block_expert.device_ptr(&self.stream);
        let (wp, _g6) = topk_w.device_ptr(&self.stream);
        let (qp, _g7) = fq.device_ptr(&self.stream);
        let (sp, _g8) = fs.device_ptr(&self.stream);
        let (op, _g9) = part.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                dd as *const _,
                ds as *const _,
                rp as *const _,
                slp as *const _,
                bp as *const _,
                wp as *const _,
                qp as *const _,
                sp as *const _,
                op as *mut _,
                ff as u32,
                embd as u32,
                n_active as u32,
                max_blocks as u32,
                bm as u32,
                self.stream_ptr(),
            )
        })
    }

    /// v3t (the wide-batch arm): TMA-staged v2 gate_up, bitwise to
    /// `q8_0_moe_gate_up_mma2_geglu`; sm_90+ packs only. n_expert from
    /// `gate.dims[2]`.
    #[allow(clippy::too_many_arguments)]
    pub fn q8_0_moe_gate_up_mma2t_geglu(
        &self,
        gate: &RepackedQ8,
        up: &RepackedQ8,
        sorted_row: &CudaSlice<u32>,
        block_expert: &CudaSlice<u32>,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        fq: &mut CudaSlice<i8>,
        fs: &mut CudaSlice<f32>,
        max_blocks: usize,
        bm: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_moe_gate_up_mma2t_geglu
            .ok_or(GpuError::MissingOp("q8_0_moe_gate_up_mma2t_geglu"))?;
        let (in_dim, ff, n_e) = (gate.dims[0], gate.dims[1], gate.dims[2]);
        let (gd, _g1) = gate.data.device_ptr(&self.stream);
        let (gs, _g2) = gate.scale.device_ptr(&self.stream);
        let (ud, _g3) = up.data.device_ptr(&self.stream);
        let (us, _g4) = up.scale.device_ptr(&self.stream);
        let (rp, _g5) = sorted_row.device_ptr(&self.stream);
        let (bp, _g6) = block_expert.device_ptr(&self.stream);
        let (xp, _g7) = xq.device_ptr(&self.stream);
        let (sp, _g8) = xs.device_ptr(&self.stream);
        let (qp, _g9) = fq.device_ptr_mut(&self.stream);
        let (fp, _g10) = fs.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                gd as *const _,
                gs as *const _,
                ud as *const _,
                us as *const _,
                rp as *const _,
                bp as *const _,
                xp as *const _,
                sp as *const _,
                qp as *mut _,
                fp as *mut _,
                in_dim as u32,
                ff as u32,
                n_e as u32,
                max_blocks as u32,
                bm as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_q8_moe_qmma2t(&self) -> bool {
        self.kernels.q8_0_moe_gate_up_mma2t_geglu.is_some()
            && self.kernels.q8_0_moe_down_mma2t.is_some()
    }

    /// v3t down twin, bitwise to `q8_0_moe_down_mma2`.
    #[allow(clippy::too_many_arguments)]
    pub fn q8_0_moe_down_mma2t(
        &self,
        down: &RepackedQ8,
        sorted_row: &CudaSlice<u32>,
        sorted_slot: &CudaSlice<u32>,
        block_expert: &CudaSlice<u32>,
        topk_w: &CudaSlice<f32>,
        fq: &CudaSlice<i8>,
        fs: &CudaSlice<f32>,
        part: &mut CudaSlice<f32>,
        n_active: usize,
        max_blocks: usize,
        bm: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_moe_down_mma2t
            .ok_or(GpuError::MissingOp("q8_0_moe_down_mma2t"))?;
        let (ff, embd, n_e) = (down.dims[0], down.dims[1], down.dims[2]);
        let (dd, _g1) = down.data.device_ptr(&self.stream);
        let (ds, _g2) = down.scale.device_ptr(&self.stream);
        let (rp, _g3) = sorted_row.device_ptr(&self.stream);
        let (slp, _g4) = sorted_slot.device_ptr(&self.stream);
        let (bp, _g5) = block_expert.device_ptr(&self.stream);
        let (wp, _g6) = topk_w.device_ptr(&self.stream);
        let (qp, _g7) = fq.device_ptr(&self.stream);
        let (sp, _g8) = fs.device_ptr(&self.stream);
        let (op, _g9) = part.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                dd as *const _,
                ds as *const _,
                rp as *const _,
                slp as *const _,
                bp as *const _,
                wp as *const _,
                qp as *const _,
                sp as *const _,
                op as *mut _,
                ff as u32,
                embd as u32,
                n_e as u32,
                n_active as u32,
                max_blocks as u32,
                bm as u32,
                self.stream_ptr(),
            )
        })
    }

    /// g2 (slot 504): token-major gate_up at bm16, epilogue writes the
    /// standard bm32-row fq/fs via the pair map. Bitwise to v2.
    #[allow(clippy::too_many_arguments)]
    pub fn q8_0_moe_gate_up_g2_geglu(
        &self,
        gate: &RepackedQ8,
        up: &RepackedQ8,
        sorted_row: &CudaSlice<u32>,
        sorted_slot: &CudaSlice<u32>,
        block_expert: &CudaSlice<u32>,
        pmap: &CudaSlice<f32>,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        fq: &mut CudaSlice<i8>,
        fs: &mut CudaSlice<f32>,
        n_active: usize,
        max_blocks: usize,
        bm: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_moe_gate_up_g2_geglu
            .ok_or(GpuError::MissingOp("q8_0_moe_gate_up_g2_geglu"))?;
        let (in_dim, ff, n_e) = (gate.dims[0], gate.dims[1], gate.dims[2]);
        let (gd, _g1) = gate.data.device_ptr(&self.stream);
        let (gs, _g2) = gate.scale.device_ptr(&self.stream);
        let (ud, _g3) = up.data.device_ptr(&self.stream);
        let (us, _g4) = up.scale.device_ptr(&self.stream);
        let (rp, _g5) = sorted_row.device_ptr(&self.stream);
        let (sl, _g6) = sorted_slot.device_ptr(&self.stream);
        let (bp, _g7) = block_expert.device_ptr(&self.stream);
        let (mp, _g8) = pmap.device_ptr(&self.stream);
        let (xp, _g9) = xq.device_ptr(&self.stream);
        let (sp, _g10) = xs.device_ptr(&self.stream);
        let (qp, _g11) = fq.device_ptr_mut(&self.stream);
        let (fp, _g12) = fs.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                gd as *const _,
                gs as *const _,
                ud as *const _,
                us as *const _,
                rp as *const _,
                sl as *const _,
                bp as *const _,
                mp as *const _,
                xp as *const _,
                sp as *const _,
                qp as *mut _,
                fp as *mut _,
                in_dim as u32,
                ff as u32,
                n_e as u32,
                n_active as u32,
                max_blocks as u32,
                bm as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_q8_moe_g2(&self) -> bool {
        self.kernels.q8_0_moe_gate_up_g2_geglu.is_some() && self.kernels.moe_align_dual.is_some()
    }

    /// dual-output align (slot 505): bm32 CSR + bm16 CSR + pair map in one
    /// launch.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_align_dual(
        &self,
        idx: &CudaSlice<u32>,
        sr32: &mut CudaSlice<u32>,
        ss32: &mut CudaSlice<u32>,
        be32: &mut CudaSlice<u32>,
        sr16: &mut CudaSlice<u32>,
        ss16: &mut CudaSlice<u32>,
        be16: &mut CudaSlice<u32>,
        pmap: &mut CudaSlice<f32>,
        rows: usize,
        n_active: usize,
        n_expert: usize,
        mb32: usize,
        mb16: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .moe_align_dual
            .ok_or(GpuError::MissingOp("moe_align_dual"))?;
        let (ip, _g1) = idx.device_ptr(&self.stream);
        let (a1, _g2) = sr32.device_ptr_mut(&self.stream);
        let (a2, _g3) = ss32.device_ptr_mut(&self.stream);
        let (a3, _g4) = be32.device_ptr_mut(&self.stream);
        let (b1, _g5) = sr16.device_ptr_mut(&self.stream);
        let (b2, _g6) = ss16.device_ptr_mut(&self.stream);
        let (b3, _g7) = be16.device_ptr_mut(&self.stream);
        let (mp, _g8) = pmap.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                ip as *const _,
                a1 as *mut _,
                a2 as *mut _,
                a3 as *mut _,
                b1 as *mut _,
                b2 as *mut _,
                b3 as *mut _,
                mp as *mut _,
                rows as u32,
                n_active as u32,
                n_expert as u32,
                mb32 as u32,
                mb16 as u32,
                self.stream_ptr(),
            )
        })
    }
}
