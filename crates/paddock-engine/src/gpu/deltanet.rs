//! DeltaNet recurrent / conv1d / gating family.

use super::error::*;
use super::*;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

impl GpuExecutor {
    /// Qwen3.5 Gated DeltaNet recurrence over `n_tokens` for all `n_heads`.
    /// `q`/`k`/`v` are `[n_tokens, n_heads, head_dim]`, `g`/`beta` are
    /// `[n_tokens, n_heads]`, `state` is `[n_heads, head_dim, head_dim]`
    /// (read-modify-write; pass zeros to start), `out` is `[n_tokens, n_heads,
    /// head_dim]`. L2-norm of q,k and the 1/sqrt(head_dim) q-scale happen inside.
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_recurrent(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        g: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        state: &mut CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .gated_delta_recurrent
            .ok_or(GpuError::MissingOp("gated_delta_recurrent"))?;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = k.device_ptr(&self.stream);
        let (vp, _g3) = v.device_ptr(&self.stream);
        let (gp, _g4) = g.device_ptr(&self.stream);
        let (bp, _g5) = beta.device_ptr(&self.stream);
        let (sp, _g6) = state.device_ptr_mut(&self.stream);
        let (op, _g7) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; buffers are sized per the doc above.
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                gp as *const _,
                bp as *const _,
                sp as *mut _,
                op as *mut _,
                n_tokens as u32,
                n_heads as u32,
                head_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Depthwise causal conv1d (kernel `k`) + SiLU over `x` [n_tokens, conv_dim]
    /// with `w` [conv_dim, k]; writes `out` [n_tokens, conv_dim]. DeltaNet input conv.
    pub fn causal_conv1d_silu(
        &self,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n_tokens: usize,
        conv_dim: usize,
        k: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .causal_conv1d_silu
            .ok_or(GpuError::MissingOp("causal_conv1d_silu"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let (op, _g3) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                xp as *const _,
                wp as *const _,
                op as *mut _,
                n_tokens as u32,
                conv_dim as u32,
                k as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Offset-aware causal conv: exactly `causal_conv1d_silu` but reads the input
    /// span starting at row `in_row_off` of `x` and writes to `out` starting at row
    /// `out_row_off` - both `[*, conv_dim]` row-major. The kernel's `ti >= 0` guard
    /// is relative to the offset base, so rows before `in_row_off` are never read
    /// (they contribute 0), which is identical to a zero-padded left window. That's
    /// why this is bit-identical to the "build [zero-window ++ span] ext, conv,
    /// keep the real rows" dance for a FRESH prompt (window pre-zeroed) - with none
    /// of the base-0 staging copies. Only valid when the slot's persistent conv
    /// window is zero at `in_row_off` (fresh-prompt batched prefill); the caller
    /// commits the trailing `k-1` rows into the window separately.
    pub fn causal_conv1d_silu_at(
        &self,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        in_row_off: usize,
        out_row_off: usize,
        n_tokens: usize,
        conv_dim: usize,
        k: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .causal_conv1d_silu
            .ok_or(GpuError::MissingOp("causal_conv1d_silu"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let (op, _g3) = out.device_ptr_mut(&self.stream);
        let xp = xp + (in_row_off * conv_dim * std::mem::size_of::<f32>()) as u64;
        let op = op + (out_row_off * conv_dim * std::mem::size_of::<f32>()) as u64;
        check(unsafe {
            f(
                xp as *const _,
                wp as *const _,
                op as *mut _,
                n_tokens as u32,
                conv_dim as u32,
                k as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Fused conv1d+SiLU+split+GQA+q/k-norm at a span offset: replaces the
    /// `causal_conv1d_silu_at` + `deltanet_split_gqa_norm` pair for the
    /// fresh-prompt in-place path - bit-exact composition, the d_conv
    /// intermediate never materializes. Same _at semantics: rows before
    /// `in_row_off` contribute zero.
    #[allow(clippy::too_many_arguments)]
    pub fn causal_conv1d_silu_qkv_at(
        &self,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        q: &mut CudaSlice<f32>,
        k: &mut CudaSlice<f32>,
        v: &mut CudaSlice<f32>,
        in_row_off: usize,
        out_row_off: usize,
        n_tokens: usize,
        n_k_heads: usize,
        n_v_heads: usize,
        s: usize,
        conv_k: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .causal_conv1d_silu_qkv
            .ok_or(GpuError::MissingOp("causal_conv1d_silu_qkv"))?;
        let conv_dim = (2 * n_k_heads + n_v_heads) * s;
        let vd = n_v_heads * s;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let (qp, _g3) = q.device_ptr_mut(&self.stream);
        let (kp, _g4) = k.device_ptr_mut(&self.stream);
        let (vp, _g5) = v.device_ptr_mut(&self.stream);
        let xp = xp + (in_row_off * conv_dim * std::mem::size_of::<f32>()) as u64;
        let off = (out_row_off * vd * std::mem::size_of::<f32>()) as u64;
        check(unsafe {
            f(
                xp as *const _,
                wp as *const _,
                (qp + off) as *mut _,
                (kp + off) as *mut _,
                (vp + off) as *mut _,
                n_tokens as u32,
                n_k_heads as u32,
                n_v_heads as u32,
                s as u32,
                conv_k as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Offset-aware `deltanet_split_gqa_norm`: reads conv rows starting at
    /// `in_row_off`, writes q/k/v rows starting at `out_row_off` - pure
    /// pointer offsets on the same kernel (rows are the outer dim).
    #[allow(clippy::too_many_arguments)]
    pub fn deltanet_split_gqa_norm_at(
        &self,
        conv: &CudaSlice<f32>,
        q: &mut CudaSlice<f32>,
        k: &mut CudaSlice<f32>,
        v: &mut CudaSlice<f32>,
        in_row_off: usize,
        out_row_off: usize,
        n_rows: usize,
        n_k_heads: usize,
        n_v_heads: usize,
        s: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .deltanet_split_gqa_norm
            .ok_or(GpuError::MissingOp("deltanet_split_gqa_norm"))?;
        let conv_dim = (2 * n_k_heads + n_v_heads) * s;
        let vd = n_v_heads * s;
        let (cp, _g1) = conv.device_ptr(&self.stream);
        let (qp, _g2) = q.device_ptr_mut(&self.stream);
        let (kp, _g3) = k.device_ptr_mut(&self.stream);
        let (vp, _g4) = v.device_ptr_mut(&self.stream);
        let cp = cp + (in_row_off * conv_dim * std::mem::size_of::<f32>()) as u64;
        let off = (out_row_off * vd * std::mem::size_of::<f32>()) as u64;
        check(unsafe {
            f(
                cp as *const _,
                (qp + off) as *mut _,
                (kp + off) as *mut _,
                (vp + off) as *mut _,
                n_rows as u32,
                n_k_heads as u32,
                n_v_heads as u32,
                s as u32,
                self.stream_ptr(),
            )
        })
    }

    /// v-bf16 twin of `causal_conv1d_silu_qkv_at`: q/k written f32, v
    /// written bf16 into the same buffer capacity (offsets are elem-sized).
    #[allow(clippy::too_many_arguments)]
    pub fn causal_conv1d_silu_qkv_b16_at(
        &self,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        q: &mut CudaSlice<f32>,
        k: &mut CudaSlice<f32>,
        v: &mut CudaSlice<f32>,
        in_row_off: usize,
        out_row_off: usize,
        n_tokens: usize,
        n_k_heads: usize,
        n_v_heads: usize,
        s: usize,
        conv_k: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .causal_conv1d_silu_qkv_b16
            .ok_or(GpuError::MissingOp("causal_conv1d_silu_qkv_b16"))?;
        let conv_dim = (2 * n_k_heads + n_v_heads) * s;
        let vd = n_v_heads * s;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let (qp, _g3) = q.device_ptr_mut(&self.stream);
        let (kp, _g4) = k.device_ptr_mut(&self.stream);
        let (vp, _g5) = v.device_ptr_mut(&self.stream);
        let xp = xp + (in_row_off * conv_dim * 4) as u64;
        let qoff = (out_row_off * vd * 4) as u64;
        let voff = (out_row_off * vd * 2) as u64; // bf16 elems
        check(unsafe {
            f(
                xp as *const _,
                wp as *const _,
                (qp + qoff) as *mut _,
                (kp + qoff) as *mut _,
                (vp + voff) as *mut _,
                n_tokens as u32,
                n_k_heads as u32,
                n_v_heads as u32,
                s as u32,
                conv_k as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_dn_vb16(&self) -> bool {
        self.kernels.causal_conv1d_silu_qkv_b16.is_some()
            && self.kernels.gated_delta_chunked_vb16.is_some()
    }

    pub fn has_conv_silu_qkv(&self) -> bool {
        self.kernels.causal_conv1d_silu_qkv.is_some()
    }

    pub fn has_conv_silu_qkv_vl(&self) -> bool {
        self.kernels.causal_conv1d_silu_qkv_vl.is_some()
    }

    /// P73 VL twin of [`Self::causal_conv1d_silu_qkv_at`]: every fresh span
    /// of a wave pass in one launch; `row0s` holds each row's span start
    /// (per-row u32). Bit-identical to the per-span offset launches.
    #[allow(clippy::too_many_arguments)]
    pub fn causal_conv1d_silu_qkv_vl(
        &self,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        row0s: &CudaSlice<u32>,
        q: &mut CudaSlice<f32>,
        k: &mut CudaSlice<f32>,
        v: &mut CudaSlice<f32>,
        n_rows: usize,
        n_k_heads: usize,
        n_v_heads: usize,
        s: usize,
        conv_k: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .causal_conv1d_silu_qkv_vl
            .ok_or(GpuError::MissingOp("causal_conv1d_silu_qkv_vl"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let (rp, _g3) = row0s.device_ptr(&self.stream);
        let (qp, _g4) = q.device_ptr_mut(&self.stream);
        let (kp, _g5) = k.device_ptr_mut(&self.stream);
        let (vp, _g6) = v.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                xp as *const _,
                wp as *const _,
                rp as *const _,
                qp as *mut _,
                kp as *mut _,
                vp as *mut _,
                n_rows as u32,
                n_k_heads as u32,
                n_v_heads as u32,
                s as u32,
                conv_k as u32,
                self.stream_ptr(),
            )
        })
    }

    /// QKC twin of [`Self::causal_conv1d_silu_qkv_vl`] (slot 446): q/k land
    /// as COMPACT bf16 `[rows, n_k_heads, s]` planes in the f32-sized
    /// buffers (one bf16 round of the same f32 values the consumer used to
    /// round itself - pipeline bit-identical); v stays f32 expanded. Must
    /// be paired with [`Self::gated_delta_chunked_rs_vl_qkc`].
    #[allow(clippy::too_many_arguments)]
    pub fn causal_conv1d_silu_qkv_vl_qkc(
        &self,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        row0s: &CudaSlice<u32>,
        q: &mut CudaSlice<f32>,
        k: &mut CudaSlice<f32>,
        v: &mut CudaSlice<f32>,
        n_rows: usize,
        n_k_heads: usize,
        n_v_heads: usize,
        s: usize,
        conv_k: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .causal_conv1d_silu_qkv_vl_qkc
            .ok_or(GpuError::MissingOp("causal_conv1d_silu_qkv_vl_qkc"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let (rp, _g3) = row0s.device_ptr(&self.stream);
        let (qp, _g4) = q.device_ptr_mut(&self.stream);
        let (kp, _g5) = k.device_ptr_mut(&self.stream);
        let (vp, _g6) = v.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                xp as *const _,
                wp as *const _,
                rp as *const _,
                qp as *mut _,
                kp as *mut _,
                vp as *mut _,
                n_rows as u32,
                n_k_heads as u32,
                n_v_heads as u32,
                s as u32,
                conv_k as u32,
                self.stream_ptr(),
            )
        })
    }

    /// `causal_conv1d_silu_qkv_vl_qkc` on ONE fresh span at a row offset
    /// (the unified tick's per-span conv form, GB10 2026-09-08): the kernel
    /// indexes rows relative to the pointers it gets and reads `row0s[row]`
    /// for the zero-pad edge, so a fresh span is the same launch at offset
    /// pointers with an all-zero row0s plane (>= `n_tokens` entries). q/k
    /// land COMPACT bf16 [rows, HK, s] at `out_row_off` in the compact
    /// planes; v lands expanded f32 as always.
    #[allow(clippy::too_many_arguments)]
    pub fn causal_conv1d_silu_qkv_vl_qkc_at(
        &self,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        row0s_zero: &CudaSlice<u32>,
        q: &mut CudaSlice<f32>,
        k: &mut CudaSlice<f32>,
        v: &mut CudaSlice<f32>,
        in_row_off: usize,
        out_row_off: usize,
        n_tokens: usize,
        n_k_heads: usize,
        n_v_heads: usize,
        s: usize,
        conv_k: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .causal_conv1d_silu_qkv_vl_qkc
            .ok_or(GpuError::MissingOp("causal_conv1d_silu_qkv_vl_qkc"))?;
        if row0s_zero.len() < n_tokens {
            return Err(GpuError::Unsupported(format!(
                "conv qkc at: row0s zero plane holds {} rows, {n_tokens} asked",
                row0s_zero.len()
            )));
        }
        let conv_dim = (2 * n_k_heads + n_v_heads) * s;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let (rp, _g3) = row0s_zero.device_ptr(&self.stream);
        let (qp, _g4) = q.device_ptr_mut(&self.stream);
        let (kp, _g5) = k.device_ptr_mut(&self.stream);
        let (vp, _g6) = v.device_ptr_mut(&self.stream);
        let xp = xp + (in_row_off * conv_dim * std::mem::size_of::<f32>()) as u64;
        // compact bf16 rows: n_k_heads * s * 2 bytes per row
        let qk_off = (out_row_off * n_k_heads * s * 2) as u64;
        let v_off = (out_row_off * n_v_heads * s * std::mem::size_of::<f32>()) as u64;
        check(unsafe {
            f(
                xp as *const _,
                wp as *const _,
                rp as *const _,
                (qp + qk_off) as *mut _,
                (kp + qk_off) as *mut _,
                (vp + v_off) as *mut _,
                n_tokens as u32,
                n_k_heads as u32,
                n_v_heads as u32,
                s as u32,
                conv_k as u32,
                self.stream_ptr(),
            )
        })
    }

    /// DeltaNet gate math: `beta = sigmoid(b)`, `g = ssm_a·softplus(a+dt_bias)`.
    /// `a`/`b` are [n_tokens, n_heads]; `ssm_a`/`dt_bias` are [n_heads].
    #[allow(clippy::too_many_arguments)]
    /// Fused-layout decay gate (x2-v3): `ab` = [n_tokens, 2*n_heads]
    /// (alpha||beta per row) from the one-call f32-plane GEMM.
    pub fn delta_gate_ab(
        &self,
        ab: &CudaSlice<f32>,
        ssm_a: &CudaSlice<f32>,
        dt_bias: &CudaSlice<f32>,
        g: &mut CudaSlice<f32>,
        beta: &mut CudaSlice<f32>,
        n_tokens: usize,
        n_heads: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .delta_gate_ab
            .ok_or(GpuError::MissingOp("delta_gate_ab"))?;
        let (abp, _g1) = ab.device_ptr(&self.stream);
        let (sp, _g2) = ssm_a.device_ptr(&self.stream);
        let (dp, _g3) = dt_bias.device_ptr(&self.stream);
        let (gp, _g4) = g.device_ptr_mut(&self.stream);
        let (betap, _g5) = beta.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                abp as *const _,
                sp as *const _,
                dp as *const _,
                gp as *mut _,
                betap as *mut _,
                n_tokens as u32,
                n_heads as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_delta_gate_ab(&self) -> bool {
        self.kernels.delta_gate_ab.is_some()
    }

    pub fn has_matvec_ab_gate(&self) -> bool {
        self.kernels.matvec_ab_gate.is_some()
    }

    /// Fused ab matvec + delta gate - one launch for the
    /// `matvec_f32_batch(ab)` + `delta_gate_ab` pair, bit-identical outputs
    /// (per-element summation schedule preserved, epilogue verbatim). `ab` is
    /// the [in_dim, 2·n_heads]-dims DeviceTensor plane the pair consumes.
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_ab_gate(
        &self,
        ab: &DeviceTensor,
        x: &CudaSlice<f32>,
        ssm_a: &CudaSlice<f32>,
        dt_bias: &CudaSlice<f32>,
        g: &mut CudaSlice<f32>,
        beta: &mut CudaSlice<f32>,
        n_tokens: usize,
        n_heads: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .matvec_ab_gate
            .ok_or(GpuError::MissingOp("matvec_ab_gate"))?;
        debug_assert_eq!(ab.dims[1], 2 * n_heads);
        let (wp, _g0) = ab.buf.device_ptr(&self.stream);
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (sp, _g2) = ssm_a.device_ptr(&self.stream);
        let (dp, _g3) = dt_bias.device_ptr(&self.stream);
        let (gp, _g4) = g.device_ptr_mut(&self.stream);
        let (betap, _g5) = beta.device_ptr_mut(&self.stream);
        // SAFETY: pack ABI v1 contract; pointers + stream live across the call
        check(unsafe {
            f(
                wp as *const _,
                xp as *const _,
                sp as *const _,
                dp as *const _,
                gp as *mut _,
                betap as *mut _,
                ab.dims[0] as u32,
                n_heads as u32,
                n_tokens as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn delta_gate(
        &self,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        ssm_a: &CudaSlice<f32>,
        dt_bias: &CudaSlice<f32>,
        g: &mut CudaSlice<f32>,
        beta: &mut CudaSlice<f32>,
        n_tokens: usize,
        n_heads: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .delta_gate
            .ok_or(GpuError::MissingOp("delta_gate"))?;
        let (ap, _g1) = a.device_ptr(&self.stream);
        let (bp, _g2) = b.device_ptr(&self.stream);
        let (sp, _g3) = ssm_a.device_ptr(&self.stream);
        let (dp, _g4) = dt_bias.device_ptr(&self.stream);
        let (gp, _g5) = g.device_ptr_mut(&self.stream);
        let (betap, _g6) = beta.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                ap as *const _,
                bp as *const _,
                sp as *const _,
                dp as *const _,
                gp as *mut _,
                betap as *mut _,
                n_tokens as u32,
                n_heads as u32,
                self.stream_ptr(),
            )
        })
    }

    /// DN split + delta gate in one launch: copies (mixed, z) out
    /// of the fused landing exactly as `row_slice4` and computes g/beta from
    /// the `2*n_heads` ab columns at `ab_off` - bit-identical to
    /// `row_slice4` + `delta_gate`, minus one launch and the d_a/d_b copies.
    #[allow(clippy::too_many_arguments)]
    pub fn row_slice2_gate(
        &self,
        src: &CudaSlice<f32>,
        src_stride: usize,
        rows: usize,
        d0: &mut CudaSlice<f32>,
        o0: usize,
        w0: usize,
        d1: &mut CudaSlice<f32>,
        o1: usize,
        w1: usize,
        ab_off: usize,
        n_heads: usize,
        ssm_a: &CudaSlice<f32>,
        dt_bias: &CudaSlice<f32>,
        g: &mut CudaSlice<f32>,
        beta: &mut CudaSlice<f32>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .row_slice2_gate
            .ok_or(GpuError::MissingOp("row_slice2_gate"))?;
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (d0p, _g2) = d0.device_ptr_mut(&self.stream);
        let (d1p, _g3) = d1.device_ptr_mut(&self.stream);
        let (ap, _g4) = ssm_a.device_ptr(&self.stream);
        let (dp, _g5) = dt_bias.device_ptr(&self.stream);
        let (gp, _g6) = g.device_ptr_mut(&self.stream);
        let (betap, _g7) = beta.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                sp as *const _,
                src_stride as u32,
                rows as u32,
                d0p as *mut _,
                o0 as u32,
                w0 as u32,
                d1p as *mut _,
                o1 as u32,
                w1 as u32,
                ab_off as u32,
                n_heads as u32,
                ap as *const _,
                dp as *const _,
                gp as *mut _,
                betap as *mut _,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_row_slice2_gate(&self) -> bool {
        self.kernels.row_slice2_gate.is_some()
    }

    /// Conv-window VL store: each span's last `km1` pre-conv rows of `src`
    /// into its slot's window region of `win`, span `(row0, take, slot, _)`
    /// quads read from device contents. One launch replaces the
    /// per-share `copy_region` pair per Linear layer.
    pub fn conv_win_store_vl(
        &self,
        src: &CudaSlice<f32>,
        spans: &CudaSlice<u32>,
        win: &mut CudaSlice<f32>,
        n_spans: usize,
        km1: usize,
        conv_dim: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .conv_win_store_vl
            .ok_or(GpuError::MissingOp("conv_win_store_vl"))?;
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (ip, _g2) = spans.device_ptr(&self.stream);
        let (wp, _g3) = win.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                sp as *const _,
                ip as *const _,
                wp as *mut _,
                n_spans as u32,
                km1 as u32,
                conv_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_conv_win_store_vl(&self) -> bool {
        self.kernels.conv_win_store_vl.is_some()
    }

    /// Gated RMSNorm over `d` per row: `out = (x·rsqrt(mean(x²)+eps))·weight·silu(z)`.
    /// `x`/`z`/`out` are [n_rows, d]; `weight` is [d].
    #[allow(clippy::too_many_arguments)]
    pub fn gated_rmsnorm(
        &self,
        x: &CudaSlice<f32>,
        z: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n_rows: usize,
        d: usize,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .gated_rmsnorm
            .ok_or(GpuError::MissingOp("gated_rmsnorm"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (zp, _g2) = z.device_ptr(&self.stream);
        let (wp, _g3) = weight.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                xp as *const _,
                zp as *const _,
                wp as *const _,
                op as *mut _,
                n_rows as u32,
                d as u32,
                eps,
                self.stream_ptr(),
            )
        })
    }

    /// Split the DeltaNet conv output [n_tokens, 2·key_dim+value_dim] into q,k
    /// (GQA-repeated to n_v_heads) and v; each output is [n_tokens, n_v_heads, s].
    #[allow(clippy::too_many_arguments)]
    pub fn deltanet_split_gqa(
        &self,
        conv: &CudaSlice<f32>,
        q_out: &mut CudaSlice<f32>,
        k_out: &mut CudaSlice<f32>,
        v_out: &mut CudaSlice<f32>,
        n_tokens: usize,
        n_k_heads: usize,
        n_v_heads: usize,
        s: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .deltanet_split_gqa
            .ok_or(GpuError::MissingOp("deltanet_split_gqa"))?;
        let (cp, _g1) = conv.device_ptr(&self.stream);
        let (qp, _g2) = q_out.device_ptr_mut(&self.stream);
        let (kp, _g3) = k_out.device_ptr_mut(&self.stream);
        let (vp, _g4) = v_out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                cp as *const _,
                qp as *mut _,
                kp as *mut _,
                vp as *mut _,
                n_tokens as u32,
                n_k_heads as u32,
                n_v_heads as u32,
                s as u32,
                self.stream_ptr(),
            )
        })
    }

    /// DeltaNet split+GQA with fused q/k L2-normalization (q pre-scaled by
    /// 1/sqrt(s)) - the v2-recurrence front end. Same layouts as
    /// `deltanet_split_gqa`; `s` must be a multiple of 32, at most 128.
    #[allow(clippy::too_many_arguments)]
    pub fn deltanet_split_gqa_norm(
        &self,
        conv: &CudaSlice<f32>,
        q_out: &mut CudaSlice<f32>,
        k_out: &mut CudaSlice<f32>,
        v_out: &mut CudaSlice<f32>,
        n_tokens: usize,
        n_k_heads: usize,
        n_v_heads: usize,
        s: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .deltanet_split_gqa_norm
            .ok_or(GpuError::MissingOp("deltanet_split_gqa_norm"))?;
        let (cp, _g1) = conv.device_ptr(&self.stream);
        let (qp, _g2) = q_out.device_ptr_mut(&self.stream);
        let (kp, _g3) = k_out.device_ptr_mut(&self.stream);
        let (vp, _g4) = v_out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                cp as *const _,
                qp as *mut _,
                kp as *mut _,
                vp as *mut _,
                n_tokens as u32,
                n_k_heads as u32,
                n_v_heads as u32,
                s as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Gated delta recurrence v2 (warp-per-state-column; llama shape). One call
    /// covers every variant: `slots` None => seq b advances state slot b (from
    /// `state_elem_off`, which must then be used with batch == 1 or slot-0
    /// semantics); `snap` Some => per-token t-major snapshots for speculative
    /// rollback; `n_tokens` > 1 => in-register chunk loop. q/k/v/out are
    /// [batch, n_tokens, n_heads, D] with q,k PRE-normalized
    /// (`deltanet_split_gqa_norm`); g/beta [batch, n_tokens, n_heads]. State and
    /// snapshot [D, D] tiles are TRANSPOSED vs the v1 kernels
    /// (column-contiguous). head_dim must be 128.
    #[allow(clippy::too_many_arguments)]
    /// True when the pack ships the P70 fused decode recurrence (slot 436).
    pub fn has_gated_delta_recurrent_v2f(&self) -> bool {
        self.kernels.gated_delta_recurrent_v2f.is_some()
    }

    /// P70 fused DECODE recurrence: split + qk-L2-norm folded into the v2
    /// body - reads the conv plane directly, byte-identical outputs to
    /// `deltanet_split_gqa_norm` + `gated_delta_recurrent_v2` at
    /// n_tokens = 1 / no snap. `out` is [batch, n_heads, head_dim].
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_recurrent_v2f(
        &self,
        conv: &CudaSlice<f32>,
        g: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        slots: Option<&CudaSlice<u32>>,
        states: &mut CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        batch: usize,
        n_k_heads: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .gated_delta_recurrent_v2f
            .ok_or(GpuError::MissingOp("gated_delta_recurrent_v2f"))?;
        let (cp, _g1) = conv.device_ptr(&self.stream);
        let (gp, _g2) = g.device_ptr(&self.stream);
        let (bp, _g3) = beta.device_ptr(&self.stream);
        let slp = match slots {
            Some(sl) => sl.device_ptr(&self.stream).0,
            None => 0,
        };
        let (sp, _g4) = states.device_ptr_mut(&self.stream);
        let (op, _g5) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                cp as *const _,
                gp as *const _,
                bp as *const _,
                slp as *const _,
                sp as *mut _,
                op as *mut _,
                batch as u32,
                n_k_heads as u32,
                n_heads as u32,
                head_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    /// True when the pack ships all three P71-R2 fused-plane strided ops
    /// (slots 438-440) - the DN split launch can then disappear at decode.
    pub fn has_dn_fused_strided(&self) -> bool {
        self.kernels.conv_step_slots_s.is_some()
            && self.kernels.gated_rmsnorm_s.is_some()
            && self.kernels.gated_delta_recurrent_v2f_g.is_some()
    }

    /// `conv_step_slots` with `x_new` read strided out of the fused plane
    /// (`x_off` element offset, `x_stride` row stride). Bit-identical to
    /// slice-then-conv.
    #[allow(clippy::too_many_arguments)]
    pub fn conv_step_slots_s(
        &self,
        wins: &mut CudaSlice<f32>,
        x_new: &CudaSlice<f32>,
        x_off: usize,
        x_stride: usize,
        w: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        slots: &CudaSlice<u32>,
        batch: usize,
        conv_dim: usize,
        k: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .conv_step_slots_s
            .ok_or(GpuError::MissingOp("conv_step_slots_s"))?;
        debug_assert!(x_off + conv_dim <= x_stride);
        let es = std::mem::size_of::<f32>() as u64;
        let (wp, _g1) = wins.device_ptr_mut(&self.stream);
        let (xp, _g2) = x_new.device_ptr(&self.stream);
        let (cp, _g3) = w.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        let (sp, _g5) = slots.device_ptr(&self.stream);
        check(unsafe {
            f(
                wp as *mut _,
                (xp + x_off as u64 * es) as *const _,
                cp as *const _,
                op as *mut _,
                sp as *const _,
                batch as u32,
                conv_dim as u32,
                k as u32,
                x_stride as u32,
                self.stream_ptr(),
            )
        })
    }

    /// `gated_rmsnorm` with z read strided out of the fused plane: z rows
    /// are (batch, head) pairs; element (r, j) lives at
    /// `(r / rpb) * z_stride + (r % rpb) * d + j` from `z_off`.
    #[allow(clippy::too_many_arguments)]
    pub fn gated_rmsnorm_s(
        &self,
        x: &CudaSlice<f32>,
        z: &CudaSlice<f32>,
        z_off: usize,
        z_stride: usize,
        z_rows_per_b: usize,
        weight: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n_rows: usize,
        d: usize,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .gated_rmsnorm_s
            .ok_or(GpuError::MissingOp("gated_rmsnorm_s"))?;
        let es = std::mem::size_of::<f32>() as u64;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (zp, _g2) = z.device_ptr(&self.stream);
        let (wp, _g3) = weight.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                xp as *const _,
                (zp + z_off as u64 * es) as *const _,
                wp as *const _,
                op as *mut _,
                n_rows as u32,
                d as u32,
                eps,
                z_stride as u32,
                z_rows_per_b as u32,
                self.stream_ptr(),
            )
        })
    }

    /// The v2f recurrence with g/beta computed in-kernel from the fused
    /// plane's alpha/beta columns (`ab_off` element offset within a row of
    /// stride `fused_stride`) - `row_slice2_gate`'s expressions verbatim,
    /// so values are bit-identical while the slice launch disappears.
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_recurrent_v2f_g(
        &self,
        conv: &CudaSlice<f32>,
        fused: &CudaSlice<f32>,
        ab_off: usize,
        fused_stride: usize,
        ssm_a: &CudaSlice<f32>,
        dt_bias: &CudaSlice<f32>,
        slots: Option<&CudaSlice<u32>>,
        states: &mut CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        batch: usize,
        n_k_heads: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .gated_delta_recurrent_v2f_g
            .ok_or(GpuError::MissingOp("gated_delta_recurrent_v2f_g"))?;
        let (cp, _g1) = conv.device_ptr(&self.stream);
        let (fp, _g2) = fused.device_ptr(&self.stream);
        let (ap, _g3) = ssm_a.device_ptr(&self.stream);
        let (dp, _g4) = dt_bias.device_ptr(&self.stream);
        let slp = match slots {
            Some(sl) => sl.device_ptr(&self.stream).0,
            None => 0,
        };
        let (sp, _g5) = states.device_ptr_mut(&self.stream);
        let (op, _g6) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                cp as *const _,
                fp as *const _,
                ab_off as u32,
                fused_stride as u32,
                ap as *const _,
                dp as *const _,
                slp as *const _,
                sp as *mut _,
                op as *mut _,
                batch as u32,
                n_k_heads as u32,
                n_heads as u32,
                head_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    // A kernel launcher: the parameter list is the kernel's.
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_recurrent_v2(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        g: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        slots: Option<&CudaSlice<u32>>,
        states: &mut CudaSlice<f32>,
        state_elem_off: usize,
        snap: Option<&mut CudaSlice<f32>>,
        out: &mut CudaSlice<f32>,
        batch: usize,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .gated_delta_recurrent_v2
            .ok_or(GpuError::MissingOp("gated_delta_recurrent_v2"))?;
        debug_assert!(slots.is_none() || state_elem_off == 0);
        debug_assert!(
            slots.is_some()
                || state_elem_off + batch * n_heads * head_dim * head_dim <= states.len()
        );
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = k.device_ptr(&self.stream);
        let (vp, _g3) = v.device_ptr(&self.stream);
        let (gp, _g4) = g.device_ptr(&self.stream);
        let (bp, _g5) = beta.device_ptr(&self.stream);
        let slp = match slots {
            Some(sl) => sl.device_ptr(&self.stream).0,
            None => 0,
        };
        let (sp, _g6) = states.device_ptr_mut(&self.stream);
        let sp_off = sp + state_elem_off as u64 * Self::dn_state_esz();
        let snp = match snap {
            Some(sn) => sn.device_ptr_mut(&self.stream).0,
            None => 0,
        };
        let (op, _g7) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                gp as *const _,
                bp as *const _,
                slp as *const _,
                sp_off as *mut _,
                snp as *mut _,
                op as *mut _,
                batch as u32,
                n_tokens as u32,
                n_heads as u32,
                head_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Snapshot-free pair present? Both slots ship together (0.20.0), so one
    /// probe gates the whole spec-verify seam.
    pub fn has_gated_delta_commit_walk(&self) -> bool {
        self.kernels.gated_delta_verify_hold.is_some()
            && self.kernels.gated_delta_commit_walk.is_some()
    }

    /// Spec-verify twin of `gated_delta_recurrent_v2`: identical `out[]`
    /// values, no snapshots, no final state writeback - the live state stays
    /// at round-start so `gated_delta_commit_walk` can recompute forward
    /// over just the accepted prefix at commit (dflash).
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_verify_hold(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        g: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        slots: Option<&CudaSlice<u32>>,
        states: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        batch: usize,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .gated_delta_verify_hold
            .ok_or(GpuError::MissingOp("gated_delta_verify_hold"))?;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = k.device_ptr(&self.stream);
        let (vp, _g3) = v.device_ptr(&self.stream);
        let (gp, _g4) = g.device_ptr(&self.stream);
        let (bp, _g5) = beta.device_ptr(&self.stream);
        let slp = match slots {
            Some(sl) => sl.device_ptr(&self.stream).0,
            None => 0,
        };
        let (sp, _g6) = states.device_ptr(&self.stream);
        let (op, _g7) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                gp as *const _,
                bp as *const _,
                slp as *const _,
                sp as *const _,
                op as *mut _,
                batch as u32,
                n_tokens as u32,
                n_heads as u32,
                head_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Commit-time accepted-prefix recompute: re-run the recurrence from the
    /// round-start state over `committed[b]` stashed tokens per row (device-
    /// staged, capture-safe), one state writeback. Bit-exact vs the snapshot
    /// the old `state_restore_slots` path restored. `committed[b] == 0`
    /// leaves the state untouched.
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_commit_walk(
        &self,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        g: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        slots: Option<&CudaSlice<u32>>,
        committed: &CudaSlice<u32>,
        states: &mut CudaSlice<f32>,
        batch: usize,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .gated_delta_commit_walk
            .ok_or(GpuError::MissingOp("gated_delta_commit_walk"))?;
        let (kp, _g1) = k.device_ptr(&self.stream);
        let (vp, _g2) = v.device_ptr(&self.stream);
        let (gp, _g3) = g.device_ptr(&self.stream);
        let (bp, _g4) = beta.device_ptr(&self.stream);
        let slp = match slots {
            Some(sl) => sl.device_ptr(&self.stream).0,
            None => 0,
        };
        let (cp, _g5) = committed.device_ptr(&self.stream);
        let (sp, _g6) = states.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                kp as *const _,
                vp as *const _,
                gp as *const _,
                bp as *const _,
                slp as *const _,
                cp as *const _,
                sp as *mut _,
                batch as u32,
                n_tokens as u32,
                n_heads as u32,
                head_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    /// dflash async round: device-side copy of the block-draft picks from the
    /// draft graph's row-major `d_out` into the MTP chain's i-major `d_draft`
    /// layout (see `KernelTableV1::dflash_chain_picks`).
    pub fn has_dflash_chain_picks(&self) -> bool {
        self.kernels.dflash_chain_picks.is_some()
    }

    pub fn dflash_chain_picks(
        &self,
        out: &CudaSlice<u32>,
        draft: &mut CudaSlice<u32>,
        n: usize,
        rows: usize,
        k_use: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .dflash_chain_picks
            .ok_or(GpuError::MissingOp("dflash_chain_picks"))?;
        let (op, _g1) = out.device_ptr(&self.stream);
        let (dp, _g2) = draft.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                op as *const _,
                dp as *mut _,
                n as u32,
                rows as u32,
                k_use as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Packed multi-span serial recurrence: decode rows (as len-1 items),
    /// independent short span walks, and same-slot fused-ckpt tail CHAINS
    /// (one item per chain - the shares' rows are contiguous) in one launch.
    /// `items` holds u32 descriptors of STRIDE 8 `(row0, len, slot, snapA_t,
    /// snapA_sel, snapB_t, snapB_sel, pad)`; rows address q/k/v/out
    /// ABSOLUTELY (no `_at` pointer offsetting) and each item's state sits
    /// at its slot's stride exactly like the slots-mode call. Internal chain
    /// seams write in-kernel state snapshots into the stage blobs passed as
    /// `snap0`/`snap1` (each `(blob, f32-elem offset)` - the layer's state
    /// region; `sel` picks the blob, `t == 0` means none) - bit-exact
    /// replacements for the between-share `copy_region` staging. Must be
    /// launched after the chunked span calls so chain leaders have advanced
    /// the state. Items must touch DISTINCT slots; no per-token snap array
    /// (spec stays on v2).
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_recurrent_v2_packed(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        g: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        items: &CudaSlice<u32>,
        states: &mut CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        snap0: Option<(&mut CudaSlice<f32>, usize)>,
        snap1: Option<(&mut CudaSlice<f32>, usize)>,
        n_items: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .gated_delta_recurrent_v2_packed
            .ok_or(GpuError::MissingOp("gated_delta_recurrent_v2_packed"))?;
        debug_assert!(items.len() >= 8 * n_items);
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = k.device_ptr(&self.stream);
        let (vp, _g3) = v.device_ptr(&self.stream);
        let (gp, _g4) = g.device_ptr(&self.stream);
        let (bp, _g5) = beta.device_ptr(&self.stream);
        let (ip, _g6) = items.device_ptr(&self.stream);
        let (sp, _g7) = states.device_ptr_mut(&self.stream);
        let (op, _g8) = out.device_ptr_mut(&self.stream);
        // snap offsets are f32-elem positions of the layer's state region in
        // the stage blob (the copy_region convention); the kernel casts to
        // its state dtype at the same byte address.
        let mut snp = [0u64; 2];
        let mut _sng = Vec::new();
        for (i, s) in [snap0, snap1].into_iter().enumerate() {
            if let Some((blob, off)) = s {
                let (p, gg) = blob.device_ptr_mut(&self.stream);
                snp[i] = p + (off * 4) as u64;
                _sng.push(gg);
            }
        }
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                gp as *const _,
                bp as *const _,
                ip as *const _,
                sp as *mut _,
                op as *mut _,
                snp[0] as *mut _,
                snp[1] as *mut _,
                n_items as u32,
                n_heads as u32,
                head_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_dn_recurrent_packed(&self) -> bool {
        self.kernels.gated_delta_recurrent_v2_packed.is_some()
    }

    /// Chunked gated delta rule for prefill spans - the v2 recurrence with only
    /// `ceil(n_tokens/64)` sequential state hops (two launches: chunk-local
    /// solve, then a column-sliced state walk that also assembles the
    /// outputs). Same layouts as `gated_delta_recurrent_v2` with batch fixed
    /// at 1; not bit-identical to v2 (different accumulation structure -
    /// decode/spec must stay on v2). Scratch sizes for `nc =
    /// ceil(n_tokens/64)` chunks: dw/du `nc*n_heads*64*128` f32, aqk
    /// `nc*n_heads*64*64` f32, cg `nc*n_heads*64` f64. head_dim must be 128.
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_chunked(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        g: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        states: &mut CudaSlice<f32>,
        state_elem_off: usize,
        out: &mut CudaSlice<f32>,
        dw: &mut CudaSlice<f32>,
        du: &mut CudaSlice<f32>,
        aqk: &mut CudaSlice<f32>,
        cg: &mut CudaSlice<f64>,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .gated_delta_chunked
            .ok_or(GpuError::MissingOp("gated_delta_chunked"))?;
        let nc = n_tokens.div_ceil(64);
        debug_assert!(state_elem_off + n_heads * head_dim * head_dim <= states.len());
        debug_assert!(dw.len() >= nc * n_heads * 64 * head_dim);
        debug_assert!(du.len() >= nc * n_heads * 64 * head_dim);
        debug_assert!(aqk.len() >= nc * n_heads * 64 * 64);
        debug_assert!(cg.len() >= nc * n_heads * 64);
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = k.device_ptr(&self.stream);
        let (vp, _g3) = v.device_ptr(&self.stream);
        let (gp, _g4) = g.device_ptr(&self.stream);
        let (bp, _g5) = beta.device_ptr(&self.stream);
        let (sp, _g6) = states.device_ptr_mut(&self.stream);
        let sp_off = sp + state_elem_off as u64 * Self::dn_state_esz();
        let (op, _g7) = out.device_ptr_mut(&self.stream);
        let (dwp, _g8) = dw.device_ptr_mut(&self.stream);
        let (dup, _g9) = du.device_ptr_mut(&self.stream);
        let (aqp, _g10) = aqk.device_ptr_mut(&self.stream);
        let (cgp, _g11) = cg.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                gp as *const _,
                bp as *const _,
                sp_off as *mut _,
                op as *mut _,
                dwp as *mut _,
                dup as *mut _,
                aqp as *mut _,
                cgp as *mut _,
                n_tokens as u32,
                n_heads as u32,
                head_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Offset-aware `gated_delta_recurrent_v2` (batch=1, no slots): reads the span
    /// q/k/v/g/beta starting at row `row_off` and writes `out` from the same row -
    /// all packed row-major (`[*, n_heads*head_dim]` for q/k/v/out, `[*, n_heads]`
    /// for g/beta). State is a single slot at `state_elem_off`. The kernel does the
    /// identical arithmetic on the identical bytes, so this is bit-identical to
    /// copying the span to base-0 temps and calling the base-0 wrapper - it just
    /// skips the copies. Used by batched prefill to avoid the per-span copy storm.
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_chunked_vb16(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        g: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        states: &mut CudaSlice<f32>,
        state_elem_off: usize,
        out: &mut CudaSlice<f32>,
        dw: &mut CudaSlice<f32>,
        du: &mut CudaSlice<f32>,
        aqk: &mut CudaSlice<f32>,
        cg: &mut CudaSlice<f64>,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .gated_delta_chunked_vb16
            .ok_or(GpuError::MissingOp("gated_delta_chunked_vb16"))?;
        let nc = n_tokens.div_ceil(64);
        debug_assert!(state_elem_off + n_heads * head_dim * head_dim <= states.len());
        debug_assert!(dw.len() >= nc * n_heads * 64 * head_dim);
        debug_assert!(du.len() >= nc * n_heads * 64 * head_dim);
        debug_assert!(aqk.len() >= nc * n_heads * 64 * 64);
        debug_assert!(cg.len() >= nc * n_heads * 64);
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = k.device_ptr(&self.stream);
        let (vp, _g3) = v.device_ptr(&self.stream);
        let (gp, _g4) = g.device_ptr(&self.stream);
        let (bp, _g5) = beta.device_ptr(&self.stream);
        let (sp, _g6) = states.device_ptr_mut(&self.stream);
        let sp_off = sp + state_elem_off as u64 * Self::dn_state_esz();
        let (op, _g7) = out.device_ptr_mut(&self.stream);
        let (dwp, _g8) = dw.device_ptr_mut(&self.stream);
        let (dup, _g9) = du.device_ptr_mut(&self.stream);
        let (aqp, _g10) = aqk.device_ptr_mut(&self.stream);
        let (cgp, _g11) = cg.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                gp as *const _,
                bp as *const _,
                sp_off as *mut _,
                op as *mut _,
                dwp as *mut _,
                dup as *mut _,
                aqp as *mut _,
                cgp as *mut _,
                n_tokens as u32,
                n_heads as u32,
                head_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Offset-aware `gated_delta_recurrent_v2` (batch=1, no slots): reads the span
    /// q/k/v/g/beta starting at row `row_off` and writes `out` from the same row -
    /// all packed row-major (`[*, n_heads*head_dim]` for q/k/v/out, `[*, n_heads]`
    /// for g/beta). State is a single slot at `state_elem_off`. The kernel does the
    /// identical arithmetic on the identical bytes, so this is bit-identical to
    /// copying the span to base-0 temps and calling the base-0 wrapper - it just
    /// skips the copies. Used by batched prefill to avoid the per-span copy storm.
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_recurrent_v2_at(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        g: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        states: &mut CudaSlice<f32>,
        state_elem_off: usize,
        out: &mut CudaSlice<f32>,
        row_off: usize,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .gated_delta_recurrent_v2
            .ok_or(GpuError::MissingOp("gated_delta_recurrent_v2"))?;
        debug_assert!(state_elem_off + n_heads * head_dim * head_dim <= states.len());
        let hd = n_heads * head_dim; // q/k/v/out row stride
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = k.device_ptr(&self.stream);
        let (vp, _g3) = v.device_ptr(&self.stream);
        let (gp, _g4) = g.device_ptr(&self.stream);
        let (bp, _g5) = beta.device_ptr(&self.stream);
        let (sp, _g6) = states.device_ptr_mut(&self.stream);
        let (op, _g7) = out.device_ptr_mut(&self.stream);
        let fsz = std::mem::size_of::<f32>() as u64;
        let qp = qp + (row_off * hd) as u64 * fsz;
        let kp = kp + (row_off * hd) as u64 * fsz;
        let vp = vp + (row_off * hd) as u64 * fsz;
        let gp = gp + (row_off * n_heads) as u64 * fsz;
        let bp = bp + (row_off * n_heads) as u64 * fsz;
        let op = op + (row_off * hd) as u64 * fsz;
        let sp = sp + state_elem_off as u64 * Self::dn_state_esz();
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                gp as *const _,
                bp as *const _,
                std::ptr::null(),
                sp as *mut _,
                std::ptr::null_mut(),
                op as *mut _,
                1u32,
                n_tokens as u32,
                n_heads as u32,
                head_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Offset-aware `gated_delta_chunked`: reads the span q/k/v/g/beta from row
    /// `row_off`, writes `out` from `row_off`; state a single slot at
    /// `state_elem_off`; scratch base-0 (sized for one span, reused per span).
    /// Same bit-identity argument as `gated_delta_recurrent_v2_at`.
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_chunked_at(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        g: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        states: &mut CudaSlice<f32>,
        state_elem_off: usize,
        out: &mut CudaSlice<f32>,
        row_off: usize,
        dw: &mut CudaSlice<f32>,
        du: &mut CudaSlice<f32>,
        aqk: &mut CudaSlice<f32>,
        cg: &mut CudaSlice<f64>,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .gated_delta_chunked
            .ok_or(GpuError::MissingOp("gated_delta_chunked"))?;
        let nc = n_tokens.div_ceil(64);
        debug_assert!(state_elem_off + n_heads * head_dim * head_dim <= states.len());
        debug_assert!(dw.len() >= nc * n_heads * 64 * head_dim);
        debug_assert!(du.len() >= nc * n_heads * 64 * head_dim);
        debug_assert!(aqk.len() >= nc * n_heads * 64 * 64);
        debug_assert!(cg.len() >= nc * n_heads * 64);
        let hd = n_heads * head_dim;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = k.device_ptr(&self.stream);
        let (vp, _g3) = v.device_ptr(&self.stream);
        let (gp, _g4) = g.device_ptr(&self.stream);
        let (bp, _g5) = beta.device_ptr(&self.stream);
        let (sp, _g6) = states.device_ptr_mut(&self.stream);
        let (op, _g7) = out.device_ptr_mut(&self.stream);
        let (dwp, _g8) = dw.device_ptr_mut(&self.stream);
        let (dup, _g9) = du.device_ptr_mut(&self.stream);
        let (aqp, _g10) = aqk.device_ptr_mut(&self.stream);
        let (cgp, _g11) = cg.device_ptr_mut(&self.stream);
        let fsz = std::mem::size_of::<f32>() as u64;
        let qp = qp + (row_off * hd) as u64 * fsz;
        let kp = kp + (row_off * hd) as u64 * fsz;
        let vp = vp + (row_off * hd) as u64 * fsz;
        let gp = gp + (row_off * n_heads) as u64 * fsz;
        let bp = bp + (row_off * n_heads) as u64 * fsz;
        let op = op + (row_off * hd) as u64 * fsz;
        let sp = sp + state_elem_off as u64 * Self::dn_state_esz();
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                gp as *const _,
                bp as *const _,
                sp as *mut _,
                op as *mut _,
                dwp as *mut _,
                dup as *mut _,
                aqp as *mut _,
                cgp as *mut _,
                n_tokens as u32,
                n_heads as u32,
                head_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Varlen chunked-GDN (ABI 323, GDN formulation band): one
    /// stage1 + register-state-walk launch pair covers every eligible span
    /// of the tick. `items` holds the chunk pairs (stride-2 u32: global
    /// row0, chunk len) followed at u32 offset `span_off` by the span quads
    /// (stride-4 u32: first launch chunk, span rows, state f32 offset, out
    /// row0). Per-span results are bit-identical to the per-span RS
    /// dispatch - only the grid packing changes. The pack mirrors the
    /// RS-route env gates and returns an error when another arm is elected,
    /// so callers must keep the per-span dispatch as the fallback.
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_chunked_rs_vl(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        g: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        states: &mut CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        dw: &mut CudaSlice<f32>,
        du: &mut CudaSlice<f32>,
        aqk: &mut CudaSlice<f32>,
        cg: &mut CudaSlice<f64>,
        items: &CudaSlice<u32>,
        n_chunks: usize,
        span_off: usize,
        n_spans: usize,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .gated_delta_chunked_rs_vl
            .ok_or(GpuError::MissingOp("gated_delta_chunked_rs_vl"))?;
        debug_assert!(dw.len() >= n_chunks * n_heads * 64 * head_dim);
        debug_assert!(du.len() >= n_chunks * n_heads * 64 * head_dim);
        debug_assert!(aqk.len() >= n_chunks * n_heads * 64 * 64);
        debug_assert!(cg.len() >= n_chunks * n_heads * 64);
        debug_assert!(items.len() >= span_off + n_spans * 4);
        debug_assert!(span_off >= n_chunks * 2);
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = k.device_ptr(&self.stream);
        let (vp, _g3) = v.device_ptr(&self.stream);
        let (gp, _g4) = g.device_ptr(&self.stream);
        let (bp, _g5) = beta.device_ptr(&self.stream);
        let (sp, _g6) = states.device_ptr_mut(&self.stream);
        let (op, _g7) = out.device_ptr_mut(&self.stream);
        let (dwp, _g8) = dw.device_ptr_mut(&self.stream);
        let (dup, _g9) = du.device_ptr_mut(&self.stream);
        let (aqp, _g10) = aqk.device_ptr_mut(&self.stream);
        let (cgp, _g11) = cg.device_ptr_mut(&self.stream);
        let (ip, _g12) = items.device_ptr(&self.stream);
        let spanp = ip + (span_off * std::mem::size_of::<u32>()) as u64;
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                gp as *const _,
                bp as *const _,
                sp as *mut _,
                op as *mut _,
                dwp as *mut _,
                dup as *mut _,
                aqp as *mut _,
                cgp as *mut _,
                ip as *const _,
                n_chunks as u32,
                spanp as *const _,
                n_spans as u32,
                n_tokens as u32,
                n_heads as u32,
                head_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_gated_delta_chunked_rs_vl(&self) -> bool {
        self.kernels.gated_delta_chunked_rs_vl.is_some()
    }

    /// QKC twin of [`Self::gated_delta_chunked_rs_vl`] (slot 447): q/k are
    /// the conv qkc twin's COMPACT bf16 planes; extra `n_k_heads`. Fails
    /// loud (NotSupported) if the stage1-rs route is not live - the caller
    /// owns the pairing.
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_chunked_rs_vl_qkc(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        g: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        states: &mut CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        dw: &mut CudaSlice<f32>,
        du: &mut CudaSlice<f32>,
        aqk: &mut CudaSlice<f32>,
        cg: &mut CudaSlice<f64>,
        items: &CudaSlice<u32>,
        n_chunks: usize,
        span_off: usize,
        n_spans: usize,
        n_tokens: usize,
        n_heads: usize,
        n_k_heads: usize,
        head_dim: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .gated_delta_chunked_rs_vl_qkc
            .ok_or(GpuError::MissingOp("gated_delta_chunked_rs_vl_qkc"))?;
        debug_assert!(dw.len() >= n_chunks * n_heads * 64 * head_dim);
        debug_assert!(du.len() >= n_chunks * n_heads * 64 * head_dim);
        debug_assert!(aqk.len() >= n_chunks * n_heads * 64 * 64);
        debug_assert!(cg.len() >= n_chunks * n_heads * 64);
        debug_assert!(items.len() >= span_off + n_spans * 4);
        debug_assert!(span_off >= n_chunks * 2);
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = k.device_ptr(&self.stream);
        let (vp, _g3) = v.device_ptr(&self.stream);
        let (gp, _g4) = g.device_ptr(&self.stream);
        let (bp, _g5) = beta.device_ptr(&self.stream);
        let (sp, _g6) = states.device_ptr_mut(&self.stream);
        let (op, _g7) = out.device_ptr_mut(&self.stream);
        let (dwp, _g8) = dw.device_ptr_mut(&self.stream);
        let (dup, _g9) = du.device_ptr_mut(&self.stream);
        let (aqp, _g10) = aqk.device_ptr_mut(&self.stream);
        let (cgp, _g11) = cg.device_ptr_mut(&self.stream);
        let (ip, _g12) = items.device_ptr(&self.stream);
        let spanp = ip + (span_off * std::mem::size_of::<u32>()) as u64;
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                gp as *const _,
                bp as *const _,
                sp as *mut _,
                op as *mut _,
                dwp as *mut _,
                dup as *mut _,
                aqp as *mut _,
                cgp as *mut _,
                ip as *const _,
                n_chunks as u32,
                spanp as *const _,
                n_spans as u32,
                n_tokens as u32,
                n_heads as u32,
                n_k_heads as u32,
                head_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_gated_delta_chunked_rs_vl_qkc(&self) -> bool {
        self.kernels.gated_delta_chunked_rs_vl_qkc.is_some()
            && self.kernels.causal_conv1d_silu_qkv_vl_qkc.is_some()
    }

    /// Row-wise argmax on device: `out[row]` = index of the max logit in row
    /// (lowest index on ties - matches the host `argmax` scan).
    pub fn argmax_rows(
        &self,
        logits: &CudaSlice<f32>,
        out: &mut CudaSlice<u32>,
        rows: usize,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .argmax_rows
            .ok_or(GpuError::MissingOp("argmax_rows"))?;
        let (lp, _g1) = logits.device_ptr(&self.stream);
        let (op, _g2) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                lp as *const _,
                op as *mut _,
                rows as u32,
                n as u32,
                self.stream_ptr(),
            )
        })
    }

    /// `argmax_rows` plus everything a transcript needs to say how sure the
    /// model was, out of the one log-sum-exp the pick already costs:
    ///
    /// - `alt[row]` - the RUNNER-UP's token id, or `n` when the row had none.
    /// - `stats[row*4 + 0]` = log p(top1)
    /// - `stats[row*4 + 1]` = p(probe); `probe >= n` means "no probe", writes 0
    /// - `stats[row*4 + 2]` = log p(top2), or 0 with no runner-up
    /// - `stats[row*4 + 3]` = Renyi-2 (collision) entropy in nats
    ///
    /// The pick is bit-identical to `argmax_rows` - same tie rule, same walk -
    /// so a lane that turns confidence on cannot get a different transcript
    /// than one that leaves it off.
    pub fn argmax_top2_rows(
        &self,
        logits: &CudaSlice<f32>,
        out: &mut CudaSlice<u32>,
        alt: &mut CudaSlice<u32>,
        stats: &mut CudaSlice<f32>,
        probe: u32,
        rows: usize,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .argmax_top2_rows
            .ok_or(GpuError::MissingOp("argmax_top2_rows"))?;
        let (lp, _g1) = logits.device_ptr(&self.stream);
        let (op, _g2) = out.device_ptr_mut(&self.stream);
        let (ap, _g3) = alt.device_ptr_mut(&self.stream);
        let (sp, _g4) = stats.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                lp as *const _,
                op as *mut _,
                ap as *mut _,
                sp as *mut _,
                probe,
                rows as u32,
                n as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Fused DeltaNet alpha+beta projection + gate: `a=alpha·x`, `b=beta·x`, then
    /// `g[o]=ssm_a[o]*softplus(a+dt_bias[o])`, `beta[o]=sigmoid(b)`. One launch for
    /// the two skinny latency-bound projections + the gate. `alpha`/`beta` share x.
    #[allow(clippy::too_many_arguments)]
    pub fn deltanet_alpha_beta_gate(
        &self,
        alpha: &RepackedQ8,
        beta_w: &RepackedQ8,
        x: &CudaSlice<f32>,
        ssm_a: &CudaSlice<f32>,
        dt_bias: &CudaSlice<f32>,
        g: &mut CudaSlice<f32>,
        beta_out: &mut CudaSlice<f32>,
        n_heads: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .deltanet_alpha_beta_gate
            .ok_or(GpuError::MissingOp("deltanet_alpha_beta_gate"))?;
        let in_dim = alpha.dims[0];
        let (adp, _g1) = alpha.data.device_ptr(&self.stream);
        let (asp, _g2) = alpha.scale.device_ptr(&self.stream);
        let (bdp, _g3) = beta_w.data.device_ptr(&self.stream);
        let (bsp, _g4) = beta_w.scale.device_ptr(&self.stream);
        let (xp, _g5) = x.device_ptr(&self.stream);
        let (sap, _g6) = ssm_a.device_ptr(&self.stream);
        let (dtp, _g7) = dt_bias.device_ptr(&self.stream);
        let (gp, _g8) = g.device_ptr_mut(&self.stream);
        let (bp, _g9) = beta_out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                adp as *const _,
                asp as *const _,
                bdp as *const _,
                bsp as *const _,
                xp as *const _,
                sap as *const _,
                dtp as *const _,
                gp as *mut _,
                bp as *mut _,
                in_dim as u32,
                n_heads as u32,
                self.stream_ptr(),
            )
        })
    }

    /// `gated_delta_recurrent` with per-token state snapshots (`snap` [n_tokens,
    /// n_heads, D, D]) - the speculative-decode verify recurrence; rollback to
    /// position t = one memcpy of snap[t] into `state`.
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_recurrent_snap(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        g: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        state: &mut CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        snap: &mut CudaSlice<f32>,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .gated_delta_recurrent_snap
            .ok_or(GpuError::MissingOp("gated_delta_recurrent_snap"))?;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = k.device_ptr(&self.stream);
        let (vp, _g3) = v.device_ptr(&self.stream);
        let (gp, _g4) = g.device_ptr(&self.stream);
        let (bp, _g5) = beta.device_ptr(&self.stream);
        let (sp, _g6) = state.device_ptr_mut(&self.stream);
        let (op, _g7) = out.device_ptr_mut(&self.stream);
        let (snp, _g8) = snap.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                gp as *const _,
                bp as *const _,
                sp as *mut _,
                op as *mut _,
                snp as *mut _,
                n_tokens as u32,
                n_heads as u32,
                head_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    /// `gated_delta_recurrent` with the state taken at an element offset into a
    /// larger (multi-slot) buffer - the batched-serving prefill writes slot k's
    /// region of the [n_slots, n_heads, D, D] state without a separate allocation.
    #[allow(clippy::too_many_arguments)]
    /// True when the pack carries the batched-runs recurrence (slot 534).
    pub fn has_gated_delta_recurrent_runs(&self) -> bool {
        self.kernels.gated_delta_recurrent_runs.is_some()
    }

    /// `n_runs` INDEPENDENT sequences through the gated-delta recurrence in one
    /// launch (slot 534): run `r` walks rows `run_off[r] .. +run_len[r]` of
    /// q/k/v/g/beta against slot `run_slot[r]`'s state. grid (n_heads, n_runs).
    ///
    /// The single-run entry grids `n_heads` = 48 blocks at 255 registers, so a
    /// serially-prefilled admission wave runs a 32%-occupied 195.9 us launch
    /// per layer per prompt - 7.05 ms of a 33.9 ms 128-token prefill, times 32
    /// prompts. Same arithmetic per run: each block still walks its own
    /// sequence in order.
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_recurrent_runs(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        g: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        state: &mut CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        run_off: &CudaSlice<u32>,
        run_len: &CudaSlice<u32>,
        run_slot: &CudaSlice<u32>,
        n_runs: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .gated_delta_recurrent_runs
            .ok_or(GpuError::MissingOp("gated_delta_recurrent_runs"))?;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = k.device_ptr(&self.stream);
        let (vp, _g3) = v.device_ptr(&self.stream);
        let (gp, _g4) = g.device_ptr(&self.stream);
        let (bp, _g5) = beta.device_ptr(&self.stream);
        let (rop, _g6) = run_off.device_ptr(&self.stream);
        let (rlp, _g7) = run_len.device_ptr(&self.stream);
        let (rsp, _g8) = run_slot.device_ptr(&self.stream);
        let (sp, _g9) = state.device_ptr_mut(&self.stream);
        let (op, _g10) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 534)
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                gp as *const _,
                bp as *const _,
                sp as *mut _,
                op as *mut _,
                rop as *const _,
                rlp as *const _,
                rsp as *const _,
                n_runs as u32,
                n_heads as u32,
                head_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Pre-normed runs walk (slot 541): companion kernel computes every
    /// token's q/k rnorm scalars in parallel with the identical tree, the
    /// walk skips both in-loop reductions. Returns false when the pack lacks
    /// the slot. `rn` must hold n_tokens*n_heads*2 f32, address-stable.
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_recurrent_runs_pn(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        g: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        state: &mut CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        run_off: &CudaSlice<u32>,
        run_len: &CudaSlice<u32>,
        run_slot: &CudaSlice<u32>,
        n_runs: usize,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
        rn: &mut CudaSlice<f32>,
    ) -> Result<bool, GpuError> {
        let Some(f) = self.kernels.gated_delta_recurrent_runs_pn else {
            return Ok(false);
        };
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = k.device_ptr(&self.stream);
        let (vp, _g3) = v.device_ptr(&self.stream);
        let (gp, _g4) = g.device_ptr(&self.stream);
        let (bp, _g5) = beta.device_ptr(&self.stream);
        let (rop, _g6) = run_off.device_ptr(&self.stream);
        let (rlp, _g7) = run_len.device_ptr(&self.stream);
        let (rsp, _g8) = run_slot.device_ptr(&self.stream);
        let (sp, _g9) = state.device_ptr_mut(&self.stream);
        let (op, _g10) = out.device_ptr_mut(&self.stream);
        let (rp, _g11) = rn.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 541)
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                gp as *const _,
                bp as *const _,
                sp as *mut _,
                op as *mut _,
                rop as *const _,
                rlp as *const _,
                rsp as *const _,
                n_runs as u32,
                n_tokens as u32,
                n_heads as u32,
                head_dim as u32,
                rp as *mut _,
                self.stream_ptr(),
            )
        })?;
        Ok(true)
    }

    /// True when the pack carries the pre-normed, P-split single-sequence
    /// walk (slot 596) - the wave-prefill class of `gated_delta_recurrent_at`.
    pub fn has_gated_delta_recurrent_pn(&self) -> bool {
        self.kernels.gated_delta_recurrent_pn.is_some()
    }

    /// [`Self::gated_delta_recurrent_at`] on slot 596: `rn` is caller-owned
    /// scratch of `n_tokens * n_heads * 2` floats (the plane the runs_pn
    /// entry already sizes). Its norms are the tree's bits; the two dots
    /// re-associate across the split lanes.
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_recurrent_pn_at(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        g: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        state: &mut CudaSlice<f32>,
        state_elem_off: usize,
        out: &mut CudaSlice<f32>,
        rn: &mut CudaSlice<f32>,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .gated_delta_recurrent_pn
            .ok_or(GpuError::MissingOp("gated_delta_recurrent_pn"))?;
        debug_assert!(state_elem_off + n_heads * head_dim * head_dim <= state.len());
        debug_assert!(rn.len() >= n_tokens * n_heads * 2);
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = k.device_ptr(&self.stream);
        let (vp, _g3) = v.device_ptr(&self.stream);
        let (gp, _g4) = g.device_ptr(&self.stream);
        let (bp, _g5) = beta.device_ptr(&self.stream);
        let (sp, _g6) = state.device_ptr_mut(&self.stream);
        let (op, _g7) = out.device_ptr_mut(&self.stream);
        let (rp, _g8) = rn.device_ptr_mut(&self.stream);
        let sp_off = sp + state_elem_off as u64 * Self::dn_state_esz();
        // SAFETY: pack ABI v1 contract (slot 596); pointers + stream live across the call
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                gp as *const _,
                bp as *const _,
                sp_off as *mut _,
                op as *mut _,
                n_tokens as u32,
                n_heads as u32,
                head_dim as u32,
                rp as *mut _,
                self.stream_ptr(),
            )
        })
    }

    pub fn gated_delta_recurrent_at(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        g: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        state: &mut CudaSlice<f32>,
        state_elem_off: usize,
        out: &mut CudaSlice<f32>,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .gated_delta_recurrent
            .ok_or(GpuError::MissingOp("gated_delta_recurrent"))?;
        debug_assert!(state_elem_off + n_heads * head_dim * head_dim <= state.len());
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = k.device_ptr(&self.stream);
        let (vp, _g3) = v.device_ptr(&self.stream);
        let (gp, _g4) = g.device_ptr(&self.stream);
        let (bp, _g5) = beta.device_ptr(&self.stream);
        let (sp, _g6) = state.device_ptr_mut(&self.stream);
        let (op, _g7) = out.device_ptr_mut(&self.stream);
        let sp_off = sp + state_elem_off as u64 * Self::dn_state_esz();
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                gp as *const _,
                bp as *const _,
                sp_off as *mut _,
                op as *mut _,
                n_tokens as u32,
                n_heads as u32,
                head_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Slot-indexed gated delta recurrence: B sequences each advance their own
    /// state one token. q/k/v [B, n_heads, D], states [n_slots, n_heads, D, D].
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_recurrent_slots(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        g: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        slots: &CudaSlice<u32>,
        states: &mut CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        batch: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .gated_delta_recurrent_slots
            .ok_or(GpuError::MissingOp("gated_delta_recurrent_slots"))?;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = k.device_ptr(&self.stream);
        let (vp, _g3) = v.device_ptr(&self.stream);
        let (gp, _g4) = g.device_ptr(&self.stream);
        let (bp, _g5) = beta.device_ptr(&self.stream);
        let (slp, _g6) = slots.device_ptr(&self.stream);
        let (sp, _g7) = states.device_ptr_mut(&self.stream);
        let (op, _g8) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                gp as *const _,
                bp as *const _,
                slp as *const _,
                sp as *mut _,
                op as *mut _,
                batch as u32,
                n_heads as u32,
                head_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Slot-indexed single-token conv+silu: B sequences advance their own window.
    /// `wins` [n_slots, k-1, conv_dim]; `x_new`/`out` [B, conv_dim].
    #[allow(clippy::too_many_arguments)]
    /// slot 564: `gated_delta_recurrent_slots` with qwen4_exp's gated norm
    /// fused into the epilogue - `out` receives the NORMED rows, so the
    /// caller skips its own gated-norm launch. `Ok(false)` = declined.
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_recurrent_slots_gn(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        g: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        slots: &CudaSlice<u32>,
        states: &mut CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        gn_z: &CudaSlice<f32>,
        gn_w: &CudaSlice<f32>,
        gn_out16: Option<&mut CudaSlice<half::bf16>>,
        gn_eps: f32,
        batch: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Result<bool, GpuError> {
        let Some(f) = self.kernels.gated_delta_recurrent_slots_gn else {
            return Ok(false);
        };
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = k.device_ptr(&self.stream);
        let (vp, _g3) = v.device_ptr(&self.stream);
        let (gp, _g4) = g.device_ptr(&self.stream);
        let (bp, _g5) = beta.device_ptr(&self.stream);
        let (sp, _g6) = slots.device_ptr(&self.stream);
        let (zp, _g7) = gn_z.device_ptr(&self.stream);
        let (wp, _g8) = gn_w.device_ptr(&self.stream);
        let (stp, _g9) = states.device_ptr_mut(&self.stream);
        let (op, _g10) = out.device_ptr_mut(&self.stream);
        let mp = match gn_out16 {
            Some(m) => {
                let (p, _g) = m.device_ptr_mut(&self.stream);
                p as *mut core::ffi::c_void
            }
            None => core::ptr::null_mut(),
        };
        // SAFETY: ABI contract (slot 564); the pack declines unsupported
        // geometries with -1, carried back as Ok(false).
        let rc = unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                gp as *const _,
                bp as *const _,
                sp as *const _,
                stp as *mut _,
                op as *mut _,
                zp as *const _,
                wp as *const _,
                mp,
                gn_eps,
                batch as u32,
                n_heads as u32,
                head_dim as u32,
                self.stream_ptr(),
            )
        };
        if rc == -1 {
            return Ok(false);
        }
        check(rc)?;
        Ok(true)
    }

    /// slot 563: `conv_step_slots` with the GDN q/k/v split+widen fused into
    /// its epilogue. `Ok(false)` = geometry not supported, run the pair.
    #[allow(clippy::too_many_arguments)]
    pub fn conv_step_slots_split(
        &self,
        wins: &mut CudaSlice<f32>,
        x_new: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        q: &mut CudaSlice<f32>,
        k_out: &mut CudaSlice<f32>,
        v_out: &mut CudaSlice<f32>,
        slots: &CudaSlice<u32>,
        batch: usize,
        conv_dim: usize,
        kconv: usize,
        heads: (usize, usize, usize, usize),
    ) -> Result<bool, GpuError> {
        let Some(f) = self.kernels.conv_step_slots_split else {
            return Ok(false);
        };
        let (hk, hv, kd, vd) = heads;
        let (wp, _g1) = wins.device_ptr_mut(&self.stream);
        let (xp, _g2) = x_new.device_ptr(&self.stream);
        let (cp, _g3) = w.device_ptr(&self.stream);
        let (qp, _g4) = q.device_ptr_mut(&self.stream);
        let (kp, _g5) = k_out.device_ptr_mut(&self.stream);
        let (vp, _g6) = v_out.device_ptr_mut(&self.stream);
        let (sp, _g7) = slots.device_ptr(&self.stream);
        // SAFETY: ABI contract (slot 563); the pack re-checks the geometry and
        // declines with -1, which is what the bool return carries back.
        let rc = unsafe {
            f(
                wp as *mut _,
                xp as *const _,
                cp as *const _,
                qp as *mut _,
                kp as *mut _,
                vp as *mut _,
                sp as *const _,
                batch as u32,
                conv_dim as u32,
                kconv as u32,
                conv_dim as u32,
                hk as u32,
                hv as u32,
                kd as u32,
                vd as u32,
                self.stream_ptr(),
            )
        };
        if rc == -1 {
            return Ok(false);
        }
        check(rc)?;
        Ok(true)
    }

    pub fn conv_step_slots(
        &self,
        wins: &mut CudaSlice<f32>,
        x_new: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        slots: &CudaSlice<u32>,
        batch: usize,
        conv_dim: usize,
        k: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .conv_step_slots
            .ok_or(GpuError::MissingOp("conv_step_slots"))?;
        let (wp, _g1) = wins.device_ptr_mut(&self.stream);
        let (xp, _g2) = x_new.device_ptr(&self.stream);
        let (cwp, _g3) = w.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        let (slp, _g5) = slots.device_ptr(&self.stream);
        check(unsafe {
            f(
                wp as *mut _,
                xp as *const _,
                cwp as *const _,
                op as *mut _,
                slp as *const _,
                batch as u32,
                conv_dim as u32,
                k as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Single-token DeltaNet causal conv+silu with a persistent window (decode).
    /// `win` [(k-1)*conv_dim] is read-modify-write; `x_new` [conv_dim] this token;
    /// `out` [conv_dim] the conv+silu output.
    pub fn conv_step(
        &self,
        win: &mut CudaSlice<f32>,
        x_new: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        conv_dim: usize,
        k: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .conv_step
            .ok_or(GpuError::MissingOp("conv_step"))?;
        let (wp, _g1) = win.device_ptr_mut(&self.stream);
        let (xp, _g2) = x_new.device_ptr(&self.stream);
        let (cwp, _g3) = w.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                wp as *mut _,
                xp as *const _,
                cwp as *const _,
                op as *mut _,
                conv_dim as u32,
                k as u32,
                self.stream_ptr(),
            )
        })
    }
}
