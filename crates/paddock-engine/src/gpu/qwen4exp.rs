//! Qwen3.8-Flash-Next (`qwen4_exp`) op wrappers - pack slots 506-516.
//!
//! The family's genuinely new math: the 4-stream hyper-connection residual
//! (grouped (1+w) norm, low-rank mix, gated combine), the PLE n-gram gate,
//! a causal conv with a DILATION, and the two GDN bits whose existing pack
//! twins are subtly the wrong shape for this checkpoint class - a gated norm
//! that gates with silu instead of sigmoid, and a key-head widening that maps
//! `vh % n_k_heads` (the GGUF lane's load-permuted order) instead of the raw
//! safetensors `repeat_interleave`. Plus the MoE pair this family needs and
//! the pack did not have: a gate+up NVFP4 expert GEMV with a fused swiglu
//! (every other nvf4 expert consumer is nemotron's gate-matrix-free relu2),
//! and the shared expert's per-token SCALAR sigmoid gate.
//!
//! Every entry point is parity-gated against `paddock_kernels::reference::
//! qwen4exp` in `tests/gpu_qwen4exp_ops.rs`.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use super::error::check;
use super::types::{Nvf4MoeLayout, Nvf4MoePlane};
use super::{GpuError, GpuExecutor};

impl GpuExecutor {
    /// True when the pack carries the whole qwen4_exp family. The lane elects
    /// on this once at load, so an older pack is a loud refusal at model-load
    /// rather than a `MissingOp` mid-forward.
    pub fn has_qwen4exp_ops(&self) -> bool {
        self.kernels.q4x_group_norm_1p.is_some()
            && self.kernels.q4x_hc_mix.is_some()
            && self.kernels.q4x_hc_combine.is_some()
            && self.kernels.q4x_scale_silu.is_some()
            && self.kernels.q4x_ple_gate.is_some()
            && self.kernels.q4x_conv_dil.is_some()
            && self.kernels.q4x_conv_dil_step.is_some()
            && self.kernels.q4x_gdn_gated_norm.is_some()
            && self.kernels.q4x_gdn_split_widen.is_some()
            && self.kernels.q4x_add_gated_row.is_some()
            && self.kernels.q4x_moe_gu_swiglu.is_some()
            && self.kernels.q4x_combine_norm.is_some()
    }

    /// `y[r,:] += x[r,:] * sigmoid(s[r])` - the MoE shared expert's per-token
    /// SCALAR gate. Not [`Self::mul_sigmoid`], which gates elementwise against
    /// a full-width plane.
    pub fn q4x_add_gated_row(
        &self,
        y: &mut CudaSlice<f32>,
        x: &CudaSlice<f32>,
        s: &CudaSlice<f32>,
        rows: usize,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q4x_add_gated_row
            .ok_or(GpuError::MissingOp("q4x_add_gated_row"))?;
        let (xp, _g2) = x.device_ptr(&self.stream);
        let (sp, _g3) = s.device_ptr(&self.stream);
        let (yp, _g1) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract
        check(unsafe {
            f(
                yp as *mut _,
                xp as *const _,
                sp as *const _,
                rows as u32,
                n as u32,
                self.stream_ptr(),
            )
        })
    }

    /// [`Self::q4x_hc_combine`] reading the inject vector at an ELEMENT offset
    /// inside `inj`. The launch-folded hyper-connection writes its inject
    /// logits as the tail rows of the low-rank projection's own output, so the
    /// combine reads them in place instead of paying a copy that would give
    /// back the launch the fold just saved. Pure base-pointer arithmetic - the
    /// kernel is slot 508 unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn q4x_hc_combine_at(
        &self,
        h: &mut CudaSlice<f32>,
        block_out: &CudaSlice<f32>,
        inj: &CudaSlice<f32>,
        inj_off: usize,
        rows: usize,
        hc: usize,
        hidden: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q4x_hc_combine
            .ok_or(GpuError::MissingOp("q4x_hc_combine"))?;
        let (bp, _g2) = block_out.device_ptr(&self.stream);
        let (ip, _g3) = inj.device_ptr(&self.stream);
        let (hp, _g1) = h.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; the offset stays inside `inj` by construction
        // (the caller owns the fused layout that put it there)
        check(unsafe {
            f(
                hp as *mut _,
                bp as *const _,
                (ip + (inj_off * 4) as u64) as *const _,
                rows as u32,
                hc as u32,
                hidden as u32,
                self.stream_ptr(),
            )
        })
    }

    /// The hyper-connection combine FUSED with the grouped (1+w) norm that
    /// follows it: `h[s,:] += block_out * 2*sigmoid(inj[s]/hc)`, then the
    /// normalized `(1+w)` image of the UPDATED state into `xn`. One launch and
    /// one pass over the 4-stream state instead of two of each. `inj_off` is
    /// the element offset of the inject vector (non-zero when it rides the
    /// tail of a folded low-rank output); `norm_w` is the FOLLOWING norm's
    /// full-width weight.
    #[allow(clippy::too_many_arguments)]
    pub fn q4x_combine_norm(
        &self,
        h: &mut CudaSlice<f32>,
        block_out: &CudaSlice<f32>,
        inj: &CudaSlice<f32>,
        inj_off: usize,
        norm_w: &CudaSlice<f32>,
        xn: &mut CudaSlice<f32>,
        xn16: Option<&mut CudaSlice<half::bf16>>,
        rows: usize,
        hc: usize,
        hidden: usize,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q4x_combine_norm
            .ok_or(GpuError::MissingOp("q4x_combine_norm"))?;
        let (bp, _g2) = block_out.device_ptr(&self.stream);
        let (ip, _g3) = inj.device_ptr(&self.stream);
        let (wp, _g4) = norm_w.device_ptr(&self.stream);
        let (hp, _g1) = h.device_ptr_mut(&self.stream);
        let (op, _g5) = xn.device_ptr_mut(&self.stream);
        let mp = match xn16 {
            Some(m) => {
                let (p, _g) = m.device_ptr_mut(&self.stream);
                p as *mut core::ffi::c_void
            }
            None => core::ptr::null_mut(),
        };
        // SAFETY: ABI contract; the inject offset is inside `inj` by
        // construction (the caller owns the fused layout that put it there)
        check(unsafe {
            f(
                hp as *mut _,
                bp as *const _,
                (ip + (inj_off * 4) as u64) as *const _,
                wp as *const _,
                op as *mut _,
                rows as u32,
                hc as u32,
                hidden as u32,
                eps,
                mp,
                self.stream_ptr(),
            )
        })
    }

    /// [`Self::q4x_add_gated_row`] reading the per-row gate at an ELEMENT
    /// offset inside `s` - the folded MoE router writes the shared expert's
    /// scalar gate as row `n_expert` of its own logits.
    #[allow(clippy::too_many_arguments)]
    /// One-launch batched router + shared-gate: topk over a [n, ne+1] logits
    /// plane (row stride ne+1), selection bit-identical to the packed form.
    /// Returns false when the pack lacks the strided slots.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_topk_batch_s(
        &self,
        logits: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        n_expert: usize,
        row_stride: usize,
        k: usize,
        idx: &mut CudaSlice<u32>,
        w: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<bool, GpuError> {
        let Some(f) = self.kernels.moe_topk_batch_s else {
            return Ok(false);
        };
        let (lp, _g1) = logits.device_ptr(&self.stream);
        let (bp, _g2) = bias.device_ptr(&self.stream);
        let (ip, _g3) = idx.device_ptr_mut(&self.stream);
        let (wp, _g4) = w.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract
        check(unsafe {
            f(
                lp as *const _,
                bp as *const _,
                n_expert as u32,
                row_stride as u32,
                k as u32,
                ip as *mut _,
                wp as *mut _,
                batch as u32,
                self.stream_ptr(),
            )
        })?;
        Ok(true)
    }

    /// Strided twin of [`Self::q4x_add_gated_row_at`]: gate for row r read at
    /// `s[s_off + r*rs]` - the fused router plane's own layout.
    #[allow(clippy::too_many_arguments)]
    pub fn q4x_add_gated_row_s(
        &self,
        y: &mut CudaSlice<f32>,
        x: &CudaSlice<f32>,
        s: &CudaSlice<f32>,
        s_off: usize,
        rs: usize,
        rows: usize,
        n: usize,
    ) -> Result<bool, GpuError> {
        let Some(f) = self.kernels.q4x_add_gated_row_s else {
            return Ok(false);
        };
        let (xp, _g2) = x.device_ptr(&self.stream);
        let (sp, _g3) = s.device_ptr(&self.stream);
        let (yp, _g1) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; offset inside `s` by construction
        check(unsafe {
            f(
                yp as *mut _,
                xp as *const _,
                (sp + (s_off * 4) as u64) as *const _,
                rs as u32,
                rows as u32,
                n as u32,
                self.stream_ptr(),
            )
        })?;
        Ok(true)
    }

    /// Does this pack carry the low-M cluster arm (slots 543/544) for THIS
    /// die? The pack nulls both on anything but sm_100 (the kernel is
    /// tcgen05/TMEM), so presence is the honest per-device answer and the
    /// load-time warm-up keys off it instead of discovering absence by
    /// launching.
    pub fn has_lowm(&self) -> bool {
        self.kernels.lowm_gemm.is_some() && self.kernels.lowm_warmup.is_some()
    }

    /// slot 543: low-M cluster GEMM over a Dual plane's f16 twin. Returns
    /// false when the pack lacks the slot or the launcher declines.
    pub fn lowm_gemm(
        &self,
        w16: &CudaSlice<half::f16>,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        in_dim: usize,
        out_dim: usize,
        batch: usize,
    ) -> Result<bool, GpuError> {
        let Some(f) = self.kernels.lowm_gemm else {
            return Ok(false);
        };
        let (wp, _g1) = w16.device_ptr(&self.stream);
        let (xp, _g2) = x.device_ptr(&self.stream);
        let (yp, _g3) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 543)
        let rc = unsafe {
            f(
                wp as *const _,
                xp as *const _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        };
        // cudaErrorInvalidValue = 1: launcher declined the shape
        if rc == 1 {
            return Ok(false);
        }
        check(rc)?;
        Ok(true)
    }

    /// slot 544: run the low-M kernel's cluster warmup on a quiet context.
    pub fn lowm_warmup(
        &self,
        w_dummy: &CudaSlice<half::f16>,
        x_dummy: &CudaSlice<f32>,
        y_dummy: &mut CudaSlice<f32>,
    ) -> Result<bool, GpuError> {
        let Some(f) = self.kernels.lowm_warmup else {
            return Ok(false);
        };
        let (wp, _g1) = w_dummy.device_ptr(&self.stream);
        let (xp, _g2) = x_dummy.device_ptr(&self.stream);
        let (yp, _g3) = y_dummy.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 544)
        check(unsafe {
            f(
                wp as *const _,
                xp as *const _,
                yp as *mut _,
                self.stream_ptr(),
            )
        })?;
        Ok(true)
    }

    /// EXPERIMENT (slot 542): cuBLASLt route for a bf16 plane. Returns false
    /// when the pack carries the stub. Never elected outside the
    /// PADDOCK_EXP_CUBLASLT probe legs.
    pub fn exp_lt_gemm(
        &self,
        w: &CudaSlice<u8>,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        in_dim: usize,
        out_dim: usize,
        batch: usize,
    ) -> Result<bool, GpuError> {
        let Some(f) = self.kernels.exp_lt_gemm else {
            return Ok(false);
        };
        let (wp, _g1) = w.device_ptr(&self.stream);
        let (xp, _g2) = x.device_ptr(&self.stream);
        let (yp, _g3) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 542)
        let rc = unsafe {
            f(
                wp as *const _,
                xp as *const _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        };
        // cudaErrorNotSupported = 801: the stub pack (no PD_EXP_LT build)
        if rc == 801 {
            return Ok(false);
        }
        check(rc)?;
        Ok(true)
    }

    pub fn q4x_add_gated_row_at(
        &self,
        y: &mut CudaSlice<f32>,
        x: &CudaSlice<f32>,
        s: &CudaSlice<f32>,
        s_off: usize,
        rows: usize,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q4x_add_gated_row
            .ok_or(GpuError::MissingOp("q4x_add_gated_row"))?;
        let (xp, _g2) = x.device_ptr(&self.stream);
        let (sp, _g3) = s.device_ptr(&self.stream);
        let (yp, _g1) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; offset inside `s` by construction
        check(unsafe {
            f(
                yp as *mut _,
                xp as *const _,
                (sp + (s_off * 4) as u64) as *const _,
                rows as u32,
                n as u32,
                self.stream_ptr(),
            )
        })
    }

    /// `matvec_f32_raw` over a ROW SEGMENT of an f32 plane - the batch > 1 arm
    /// of the folded planes, which read one concatenated residency as two
    /// projections instead of holding two.
    #[allow(clippy::too_many_arguments)]
    pub fn matvec_f32_rows(
        &self,
        w: &CudaSlice<f32>,
        first_row: usize,
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
        // SAFETY: ABI contract; the plane is [out, in] row-major, so a row
        // segment is a contiguous byte range
        check(unsafe {
            f(
                (wp + (first_row * in_dim * 4) as u64) as *const _,
                xp as *const _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn bf16_gemv2_swiglu(
        &self,
        wg: &CudaSlice<u8>,
        wu: &CudaSlice<u8>,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        in_dim: usize,
        out_dim: usize,
        batch: usize,
    ) -> Result<bool, GpuError> {
        let Some(f) = self.kernels.bf16_gemv2_swiglu else {
            return Ok(false);
        };
        let (gp, _g1) = wg.device_ptr(&self.stream);
        let (up, _g2) = wu.device_ptr(&self.stream);
        let (xp, _g3) = x.device_ptr(&self.stream);
        let (yp, _g4) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 546)
        let rc = unsafe {
            f(
                gp as *const _,
                up as *const _,
                xp as *const _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        };
        if rc == 1 {
            return Ok(false);
        }
        check(rc)?;
        Ok(true)
    }

    pub fn q4x_moe_gu_swiglu(
        &self,
        gate: &Nvf4MoePlane,
        up: &Nvf4MoePlane,
        idx: &CudaSlice<u32>,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        k: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q4x_moe_gu_swiglu
            .ok_or(GpuError::MissingOp("q4x_moe_gu_swiglu"))?;
        // dual-resident layouts: a CutBlk plane still serves this Row-reader
        // through its retained row_scale copy (the n=1 election).
        for (w, who) in [(gate, "gate"), (up, "up")] {
            if w.layout != Nvf4MoeLayout::Row
                && !(w.layout == Nvf4MoeLayout::CutBlk && w.row_scale.is_some())
            {
                return Err(GpuError::Unsupported(format!(
                    "q4x_moe_gu_swiglu: {who} plane is {:?}, this kernel reads Row",
                    w.layout
                )));
            }
        }
        if (gate.ff, gate.in_dim) != (up.ff, up.in_dim) {
            return Err(GpuError::Unsupported(
                "q4x_moe_gu_swiglu: gate and up planes disagree on shape".into(),
            ));
        }
        let (gd, _a1) = gate.data.device_ptr(&self.stream);
        let gate_sc = gate.row_scale.as_ref().unwrap_or(&gate.scale);
        let (gs, _a2) = gate_sc.device_ptr(&self.stream);
        let (g2, _a3) = gate.scale2.device_ptr(&self.stream);
        let (ud, _a4) = up.data.device_ptr(&self.stream);
        let up_sc = up.row_scale.as_ref().unwrap_or(&up.scale);
        let (us, _a5) = up_sc.device_ptr(&self.stream);
        let (u2, _a6) = up.scale2.device_ptr(&self.stream);
        let (ip, _a7) = idx.device_ptr(&self.stream);
        let (xp, _a8) = x.device_ptr(&self.stream);
        let (yp, _a9) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; shapes come from the planes themselves
        check(unsafe {
            f(
                gd as *const _,
                gs as *const _,
                g2 as *const _,
                ud as *const _,
                us as *const _,
                u2 as *const _,
                ip as *const _,
                xp as *const _,
                yp as *mut _,
                gate.in_dim as u32,
                gate.ff as u32,
                k as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Grouped Gemma RMSNorm with the (1+w) FMA affine: `rows` rows of
    /// `groups * gd`, each group normalized by its own RMS, `w` spanning the
    /// full row width. `x` and `out` may alias (the kernel reads each element
    /// once before writing it).
    #[allow(clippy::too_many_arguments)]
    pub fn q4x_group_norm_1p(
        &self,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        mirror: Option<&mut CudaSlice<half::bf16>>,
        rows: usize,
        groups: usize,
        gd: usize,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q4x_group_norm_1p
            .ok_or(GpuError::MissingOp("q4x_group_norm_1p"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let (op, _g3) = out.device_ptr_mut(&self.stream);
        let m_guard = mirror.map(|m| m.device_ptr_mut(&self.stream));
        let mp = match &m_guard {
            Some((p, _)) => *p as *mut core::ffi::c_void,
            None => std::ptr::null_mut(),
        };
        // SAFETY: ABI contract; shapes are the caller's, buffers sized by it
        check(unsafe {
            f(
                xp as *const _,
                wp as *const _,
                op as *mut _,
                mp,
                rows as u32,
                groups as u32,
                gd as u32,
                eps,
                self.stream_ptr(),
            )
        })
    }

    /// Hyper-connection mix reduce: `out[r,d] = Σ_s sigmoid(gate[r,s,d]) *
    /// xn[r,s,d] / hc`. `xn`/`gate` are `[rows, hc*hidden]`, `out` is
    /// `[rows, hidden]`.
    pub fn q4x_hc_mix(
        &self,
        xn: &CudaSlice<f32>,
        gate: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        mirror: Option<&mut CudaSlice<half::bf16>>,
        rows: usize,
        hc: usize,
        hidden: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q4x_hc_mix
            .ok_or(GpuError::MissingOp("q4x_hc_mix"))?;
        let (xp, _g1) = xn.device_ptr(&self.stream);
        let (gp, _g2) = gate.device_ptr(&self.stream);
        let (op, _g3) = out.device_ptr_mut(&self.stream);
        let m_guard = mirror.map(|m| m.device_ptr_mut(&self.stream));
        let mp = match &m_guard {
            Some((p, _)) => *p as *mut core::ffi::c_void,
            None => std::ptr::null_mut(),
        };
        // SAFETY: ABI contract
        check(unsafe {
            f(
                xp as *const _,
                gp as *const _,
                op as *mut _,
                mp,
                rows as u32,
                hc as u32,
                hidden as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Hyper-connection combine: `h[r,s,:] += block_out[r,:] *
    /// 2*sigmoid(inj[r,s]/hc)`, in place on the 4-stream state.
    pub fn q4x_hc_combine(
        &self,
        h: &mut CudaSlice<f32>,
        block_out: &CudaSlice<f32>,
        inj: &CudaSlice<f32>,
        rows: usize,
        hc: usize,
        hidden: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q4x_hc_combine
            .ok_or(GpuError::MissingOp("q4x_hc_combine"))?;
        let (bp, _g2) = block_out.device_ptr(&self.stream);
        let (ip, _g3) = inj.device_ptr(&self.stream);
        let (hp, _g1) = h.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract
        check(unsafe {
            f(
                hp as *mut _,
                bp as *const _,
                ip as *const _,
                rows as u32,
                hc as u32,
                hidden as u32,
                self.stream_ptr(),
            )
        })
    }

    /// `m = silu(m * inv)` in place - the low-rank mix's `/hc` then silu.
    pub fn q4x_scale_silu(
        &self,
        m: &mut CudaSlice<f32>,
        n: usize,
        inv: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q4x_scale_silu
            .ok_or(GpuError::MissingOp("q4x_scale_silu"))?;
        let (mp, _g) = m.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract
        check(unsafe { f(mp as *mut _, n as u32, inv, self.stream_ptr()) })
    }

    /// PLE per-stream gate: `gv[r,s,:] = sigmoid(signed_sqrt(K_s·Q_s /
    /// sqrt(hidden))) * value[r,:]`. `kn`/`qn` must already be group-normalized
    /// (`q4x_group_norm_1p` with the key/query norm weights).
    #[allow(clippy::too_many_arguments)]
    /// True when the pack carries the ring-window conv step (slot 533).
    pub fn has_q4x_conv_dil_step_ring(&self) -> bool {
        self.kernels.q4x_conv_dil_step_ring.is_some()
    }

    /// Per-slot dilated conv step over a POSITION-indexed ring window, window
    /// advance fused in (slot 533). `pos[r]` is the position of row r's token;
    /// the pre-conv row for position `q` lives at ring row `q % ((k-1)*dil)`.
    #[allow(clippy::too_many_arguments)]
    pub fn q4x_conv_dil_step_ring(
        &self,
        x: &CudaSlice<f32>,
        win: &mut CudaSlice<f32>,
        w: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        slots: &CudaSlice<u32>,
        pos: &CudaSlice<u32>,
        dim: usize,
        k: usize,
        dil: usize,
        rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q4x_conv_dil_step_ring
            .ok_or(GpuError::MissingOp("q4x_conv_dil_step_ring"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let (sp, _g3) = slots.device_ptr(&self.stream);
        let (pp, _g4) = pos.device_ptr(&self.stream);
        let (np, _g5) = win.device_ptr_mut(&self.stream);
        let (op, _g6) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 533)
        check(unsafe {
            f(
                xp as *const _,
                np as *mut _,
                wp as *const _,
                op as *mut _,
                sp as *const _,
                pp as *const _,
                dim as u32,
                k as u32,
                dil as u32,
                rows as u32,
                self.stream_ptr(),
            )
        })
    }

    /// True when the pack carries the device PLE gather (slot 532).
    pub fn has_q4x_ple_gather(&self) -> bool {
        self.kernels.q4x_ple_gather.is_some()
    }

    /// PLE n-gram row gather off the DEVICE-RESIDENT fp8 table (slot 532).
    ///
    /// `ids` is `[rows, heads]` GLOBAL row ids; `out` is `[rows,
    /// heads*width]` f32, the same layout the host gather produced. The whole
    /// point is that the 51.2 GB table is read on the device: the host twin
    /// is a uniform random read over a 51.2 GB mmap at 160 useful bytes per
    /// 4 KB page, which is what put 0.9-48 s prefill ticks on the ladder.
    #[allow(clippy::too_many_arguments)]
    pub fn q4x_ple_gather(
        &self,
        table: &CudaSlice<u8>,
        ids: &CudaSlice<u32>,
        out: &mut CudaSlice<f32>,
        scale: f32,
        rows: usize,
        heads: usize,
        width: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q4x_ple_gather
            .ok_or(GpuError::MissingOp("q4x_ple_gather"))?;
        let (tp, _g1) = table.device_ptr(&self.stream);
        let (ip, _g2) = ids.device_ptr(&self.stream);
        let (op, _g3) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 532)
        check(unsafe {
            f(
                tp as *const _,
                ip as *const _,
                op as *mut _,
                scale,
                rows as u32,
                heads as u32,
                width as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn q4x_ple_gate(
        &self,
        kn: &CudaSlice<f32>,
        qn: &CudaSlice<f32>,
        value: &CudaSlice<f32>,
        gv: &mut CudaSlice<f32>,
        rows: usize,
        hc: usize,
        hidden: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q4x_ple_gate
            .ok_or(GpuError::MissingOp("q4x_ple_gate"))?;
        let (kp, _g1) = kn.device_ptr(&self.stream);
        let (qp, _g2) = qn.device_ptr(&self.stream);
        let (vp, _g3) = value.device_ptr(&self.stream);
        let (op, _g4) = gv.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract
        check(unsafe {
            f(
                kp as *const _,
                qp as *const _,
                vp as *const _,
                op as *mut _,
                rows as u32,
                hc as u32,
                hidden as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Dilated causal depthwise conv1d + silu over a token sequence, fresh
    /// state. `w` is `[dim, k]`; `src` and `out` must be distinct buffers
    /// (every output row reads earlier input rows).
    #[allow(clippy::too_many_arguments)]
    /// Offset-aware dilated conv (the `causal_conv1d_silu_at` contract): the
    /// span starts at row `row_off` of both `src` and `out`, and the kernel's
    /// own left-pad guard is relative to that base, so rows before it are
    /// never read. That is exactly a fresh sequence's zero left-pad, which is
    /// what makes a batched-runs prefill legal without a runs table here.
    #[allow(clippy::too_many_arguments)]
    pub fn q4x_conv_dil_at(
        &self,
        src: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        row_off: usize,
        n: usize,
        dim: usize,
        k: usize,
        dil: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q4x_conv_dil
            .ok_or(GpuError::MissingOp("q4x_conv_dil"))?;
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let (op, _g3) = out.device_ptr_mut(&self.stream);
        let byte_off = (row_off * dim * std::mem::size_of::<f32>()) as u64;
        // SAFETY: ABI contract
        check(unsafe {
            f(
                (sp + byte_off) as *const _,
                wp as *const _,
                (op + byte_off) as *mut _,
                n as u32,
                dim as u32,
                k as u32,
                dil as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn q4x_conv_dil(
        &self,
        src: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n_tokens: usize,
        dim: usize,
        k: usize,
        dil: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q4x_conv_dil
            .ok_or(GpuError::MissingOp("q4x_conv_dil"))?;
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let (op, _g3) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract
        check(unsafe {
            f(
                sp as *const _,
                wp as *const _,
                op as *mut _,
                n_tokens as u32,
                dim as u32,
                k as u32,
                dil as u32,
                self.stream_ptr(),
            )
        })
    }

    /// One-token twin of [`Self::q4x_conv_dil`] off a carried window.
    /// `win` is `[(k-1)*dil, dim]` OLDEST-FIRST, holding pre-conv rows; the
    /// caller advances it (a device row shift), so this stays graph-safe.
    #[allow(clippy::too_many_arguments)]
    /// Per-slot dilated conv window step (slot 521): `rows` sequences, row `r`
    /// against the window at `slots[r]`. Same expression and order as the
    /// single-row form.
    #[allow(clippy::too_many_arguments)]
    pub fn q4x_conv_dil_step_slots(
        &self,
        x: &CudaSlice<f32>,
        win: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        slots: &CudaSlice<u32>,
        dim: usize,
        k: usize,
        dil: usize,
        rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q4x_conv_dil_step_slots
            .ok_or(GpuError::MissingOp("q4x_conv_dil_step_slots"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (wp, _g2) = win.device_ptr(&self.stream);
        let (cp, _g3) = w.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        let (sp, _g5) = slots.device_ptr(&self.stream);
        // SAFETY: ABI contract; `win` is slots * (k-1)*dil * dim
        check(unsafe {
            f(
                xp as *const _,
                wp as *const _,
                cp as *const _,
                op as *mut _,
                sp as *const _,
                dim as u32,
                k as u32,
                dil as u32,
                rows as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn q4x_conv_dil_step(
        &self,
        x: &CudaSlice<f32>,
        win: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        dim: usize,
        k: usize,
        dil: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q4x_conv_dil_step
            .ok_or(GpuError::MissingOp("q4x_conv_dil_step"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (np, _g2) = win.device_ptr(&self.stream);
        let (wp, _g3) = w.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract
        check(unsafe {
            f(
                xp as *const _,
                np as *const _,
                wp as *const _,
                op as *mut _,
                dim as u32,
                k as u32,
                dil as u32,
                self.stream_ptr(),
            )
        })
    }

    /// GDN output gated norm: per `d`-wide row, `y = w · rms_norm(x) ·
    /// sigmoid(z)` - plain `w`, SIGMOID gate. Not interchangeable with
    /// [`Self::gated_rmsnorm`], which gates with silu.
    #[allow(clippy::too_many_arguments)]
    pub fn q4x_gdn_gated_norm(
        &self,
        x: &CudaSlice<f32>,
        z: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        mirror: Option<&mut CudaSlice<half::bf16>>,
        n_rows: usize,
        d: usize,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q4x_gdn_gated_norm
            .ok_or(GpuError::MissingOp("q4x_gdn_gated_norm"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (zp, _g2) = z.device_ptr(&self.stream);
        let (wp, _g3) = w.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        let m_guard = mirror.map(|mm| mm.device_ptr_mut(&self.stream));
        let mp = match &m_guard {
            Some((p, _)) => *p as *mut core::ffi::c_void,
            None => std::ptr::null_mut(),
        };
        // SAFETY: ABI contract
        check(unsafe {
            f(
                xp as *const _,
                zp as *const _,
                wp as *const _,
                op as *mut _,
                mp,
                n_rows as u32,
                d as u32,
                eps,
                self.stream_ptr(),
            )
        })
    }

    /// Split the GDN conv output `[rows, 2*hk*kd + hv*vd]` into q, k (widened
    /// hk -> hv heads by REPEAT_INTERLEAVE) and v, RAW - `gated_delta_recurrent`
    /// applies the L2 norm and the 1/sqrt(D) scale itself.
    #[allow(clippy::too_many_arguments)]
    pub fn q4x_gdn_split_widen(
        &self,
        conv: &CudaSlice<f32>,
        q: &mut CudaSlice<f32>,
        k: &mut CudaSlice<f32>,
        v: &mut CudaSlice<f32>,
        rows: usize,
        k_heads: usize,
        v_heads: usize,
        k_dim: usize,
        v_dim: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q4x_gdn_split_widen
            .ok_or(GpuError::MissingOp("q4x_gdn_split_widen"))?;
        let (cp, _g1) = conv.device_ptr(&self.stream);
        let (qp, _g2) = q.device_ptr_mut(&self.stream);
        let (kp, _g3) = k.device_ptr_mut(&self.stream);
        let (vp, _g4) = v.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract
        check(unsafe {
            f(
                cp as *const _,
                qp as *mut _,
                kp as *mut _,
                vp as *mut _,
                rows as u32,
                k_heads as u32,
                v_heads as u32,
                k_dim as u32,
                v_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    /// The GGUF lane's twin of `q4x_gdn_split_widen`: tiled value-head order.
    pub fn q4x_gdn_split_widen_tiled(
        &self,
        conv: &CudaSlice<f32>,
        q: &mut CudaSlice<f32>,
        k: &mut CudaSlice<f32>,
        v: &mut CudaSlice<f32>,
        rows: usize,
        k_heads: usize,
        v_heads: usize,
        k_dim: usize,
        v_dim: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q4x_gdn_split_widen_tiled
            .ok_or(GpuError::MissingOp("q4x_gdn_split_widen_tiled"))?;
        let (cp, _g1) = conv.device_ptr(&self.stream);
        let (qp, _g2) = q.device_ptr_mut(&self.stream);
        let (kp, _g3) = k.device_ptr_mut(&self.stream);
        let (vp, _g4) = v.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract
        check(unsafe {
            f(
                cp as *const _,
                qp as *mut _,
                kp as *mut _,
                vp as *mut _,
                rows as u32,
                k_heads as u32,
                v_heads as u32,
                k_dim as u32,
                v_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_q4x_gdn_split_widen_tiled(&self) -> bool {
        self.kernels.q4x_gdn_split_widen_tiled.is_some()
    }
}
