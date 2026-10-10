//! matvec, norms, rope, activations, embed gather, allocs.

use super::error::*;
use super::*;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::f16;
use paddock_models::ggml_type::GgmlType;

/// Shape census for the in-house GEMM helpers (born as, kept
/// after the phase-C cuBLAS deletion - new models' shapes still need
/// enumerating against gauntlet coverage). Behind PADDOCK_LOG_GEMM_SHAPES
/// (off by default), logs each unique (kind, in, out, batch) once. kind:
/// "A-f32" (router matvec class) / "B-gemmEx-f16" (f16xf16->f32, slot 383).
/// Cost when off is one cached OnceLock bool load.
pub(crate) fn gemm_census(kind: &'static str, in_dim: usize, out_dim: usize, batch: usize) {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    static ENABLED: OnceLock<bool> = OnceLock::new();
    if !*ENABLED.get_or_init(|| {
        paddock_models::dev_var_os!("PADDOCK_LOG_GEMM_SHAPES")
            .map(|v| !v.is_empty() && v != std::ffi::OsStr::new("0"))
            .unwrap_or(false)
    }) {
        return;
    }
    static SEEN: OnceLock<Mutex<HashSet<(&'static str, u32, u32, u32)>>> = OnceLock::new();
    let key = (kind, in_dim as u32, out_dim as u32, batch as u32);
    if SEEN
        .get_or_init(|| Mutex::new(HashSet::new()))
        .lock()
        .map(|mut s| s.insert(key))
        .unwrap_or(false)
    {
        eprintln!("[gemm-census] kind={kind} in={in_dim} out={out_dim} batch={batch}");
    }
}

impl GpuExecutor {
    // ---- per-op kernel wrappers (typed, no unsafe at call sites) ----

    pub fn rmsnorm(
        &self,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n: usize,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .rmsnorm_f32
            .ok_or(GpuError::MissingOp("rmsnorm"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let (op, _g3) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; buffers sized by caller
        check(unsafe {
            f(
                xp as *const _,
                wp as *const _,
                op as *mut _,
                n as u32,
                eps,
                self.stream_ptr(),
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rope_yarn(
        &self,
        x: &mut CudaSlice<f32>,
        n_heads: usize,
        head_dim: usize,
        pos: usize,
        params: (f32, f32, f32, f32, f32, f32),
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .rope_yarn_f32
            .ok_or(GpuError::MissingOp("rope_yarn"))?;
        let (theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale) = params;
        let (xp, _g) = x.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract
        check(unsafe {
            f(
                xp as *mut _,
                n_heads as u32,
                head_dim as u32,
                pos as u32,
                theta_scale,
                freq_scale,
                corr_low,
                corr_high,
                ext_factor,
                mscale,
                self.stream_ptr(),
            )
        })
    }

    /// Softmax with sink over the first n elements of `scores`.
    pub fn softmax_sink(
        &self,
        scores: &mut CudaSlice<f32>,
        n: usize,
        sink: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .softmax_sink_f32
            .ok_or(GpuError::MissingOp("softmax_sink"))?;
        let (sp, _g) = scores.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; n <= scores.len() by caller
        check(unsafe { f(sp as *mut _, n as u32, sink, self.stream_ptr()) })
    }

    pub fn swiglu_oai(
        &self,
        gate: &mut CudaSlice<f32>,
        up: &CudaSlice<f32>,
        n: usize,
        alpha: f32,
        limit: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .swiglu_oai_f32
            .ok_or(GpuError::MissingOp("swiglu_oai"))?;
        let (gp, _g1) = gate.device_ptr_mut(&self.stream);
        let (up_p, _g2) = up.device_ptr(&self.stream);
        // SAFETY: ABI contract
        check(unsafe {
            f(
                gp as *mut _,
                up_p as *const _,
                n as u32,
                alpha,
                limit,
                self.stream_ptr(),
            )
        })
    }

    /// x[..n] += y[..n]; y may be any device view (e.g. a bias slice).
    pub fn add<Y: DevicePtr<f32>>(
        &self,
        x: &mut CudaSlice<f32>,
        y: &Y,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .add_inplace_f32
            .ok_or(GpuError::MissingOp("add_inplace"))?;
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (yp, _g2) = y.device_ptr(&self.stream);
        // SAFETY: ABI contract
        check(unsafe { f(xp as *mut _, yp as *const _, n as u32, self.stream_ptr()) })
    }

    /// `x[x_off .. x_off+n] += y[y_off .. y_off+n]` - [`Self::add`] over a
    /// window of both buffers, by offsetting the pointers the way
    /// `vision_attn_at` does. Exists for DeepStack injection: the vision rows
    /// of a prefill chunk are a contiguous span, so one call per image span
    /// per injected layer covers it with no new kernel.
    pub fn add_at(
        &self,
        x: &mut CudaSlice<f32>,
        x_off: usize,
        y: &CudaSlice<f32>,
        y_off: usize,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .add_inplace_f32
            .ok_or(GpuError::MissingOp("add_inplace"))?;
        debug_assert!(
            x_off + n <= x.len() && y_off + n <= y.len(),
            "add_at window out of range"
        );
        let es = std::mem::size_of::<f32>() as u64;
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (yp, _g2) = y.device_ptr(&self.stream);
        // SAFETY: ABI contract; both windows bounds-checked above
        check(unsafe {
            f(
                (xp + x_off as u64 * es) as *mut _,
                (yp + y_off as u64 * es) as *const _,
                n as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Release freed stream-ordered allocations back to the OS. cudarc
    /// allocates through cuMemAllocAsync; dropped buffers return to the
    /// device's CURRENT mempool, and mem_get_info counts pool-held bytes as
    /// used - so load-time staging (repack raw uploads) masquerades as
    /// resident and starves every free-VRAM-based sizer (width, KV pool,
    /// spec cap). Sync first so pending frees land in the pool, then trim.
    /// Best-effort: any failure just leaves the conservative reading.
    pub fn trim_mem_pool(&self) {
        let _ = self.stream.synchronize();
        unsafe {
            if let Ok(dev) = cudarc::driver::result::device::get(self.ctx.ordinal() as i32)
                && let Ok(pool) = cudarc::driver::result::device::get_mem_pool(dev)
            {
                let _ = cudarc::driver::result::mem_pool::trim_to(pool, 0);
            }
        }
    }

    pub fn alloc_u32(&self, n: usize) -> Result<CudaSlice<u32>, GpuError> {
        self.stream.alloc_zeros(n).map_err(drv)
    }

    pub fn alloc_i8(&self, n: usize) -> Result<CudaSlice<i8>, GpuError> {
        self.stream.alloc_zeros(n).map_err(drv)
    }

    pub fn alloc_f64(&self, n: usize) -> Result<CudaSlice<f64>, GpuError> {
        self.stream.alloc_zeros(n).map_err(drv)
    }

    pub fn alloc_u8(&self, n: usize) -> Result<CudaSlice<u8>, GpuError> {
        self.stream.alloc_zeros(n).map_err(drv)
    }

    pub fn alloc_i32(&self, n: usize) -> Result<CudaSlice<i32>, GpuError> {
        self.stream.alloc_zeros(n).map_err(drv)
    }

    pub fn alloc_u16(&self, n: usize) -> Result<CudaSlice<u16>, GpuError> {
        self.stream.alloc_zeros(n).map_err(drv)
    }

    /// Partial sectioned M-RoPE in place over `x` [n_tokens, n_heads*head_dim].
    /// `positions` is [4, n_tokens] axis-major (t,h,w,e); `sections` the per-axis
    /// rotary-pair counts. YaRN params in `reference::ops::YarnRope::kernel_params`
    /// order. Rotates the first `n_rot` dims; the rest pass through.
    #[allow(clippy::too_many_arguments)]
    pub fn mrope(
        &self,
        x: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
        n_rot: usize,
        params: (f32, f32, f32, f32, f32, f32),
        sections: [u32; 4],
    ) -> Result<(), GpuError> {
        let f = self.kernels.mrope.ok_or(GpuError::MissingOp("mrope"))?;
        let (theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale) = params;
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (pp, _g2) = positions.device_ptr(&self.stream);
        check(unsafe {
            f(
                xp as *mut _,
                pp as *const _,
                n_tokens as u32,
                n_heads as u32,
                head_dim as u32,
                n_rot as u32,
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
                self.stream_ptr(),
            )
        })
    }

    /// [`Self::mrope`] with INTERLEAVED sections (HF `mrope_interleaved`,
    /// ggml IMROPE) - rotation pair p reads h when p % 3 == 1, w when
    /// p % 3 == 2, t otherwise, each within its section's reach. What the
    /// qwen3vl text stack applies to image tokens.
    #[allow(clippy::too_many_arguments)]
    pub fn imrope(
        &self,
        x: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
        n_rot: usize,
        params: (f32, f32, f32, f32, f32, f32),
        sections: [u32; 4],
    ) -> Result<(), GpuError> {
        let f = self.kernels.imrope.ok_or(GpuError::MissingOp("imrope"))?;
        let (theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale) = params;
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (pp, _g2) = positions.device_ptr(&self.stream);
        check(unsafe {
            f(
                xp as *mut _,
                pp as *const _,
                n_tokens as u32,
                n_heads as u32,
                head_dim as u32,
                n_rot as u32,
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
                self.stream_ptr(),
            )
        })
    }

    /// Sigmoid output gate in place: `x[i] *= sigmoid(gate[i])`, `n` elements.
    pub fn mul_sigmoid(
        &self,
        x: &mut CudaSlice<f32>,
        gate: &CudaSlice<f32>,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .mul_sigmoid
            .ok_or(GpuError::MissingOp("mul_sigmoid"))?;
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (gp, _g2) = gate.device_ptr(&self.stream);
        check(unsafe { f(xp as *mut _, gp as *const _, n as u32, self.stream_ptr()) })
    }

    /// Laguna per-head softplus output gate in place:
    /// `x[r,h,d] *= softplus(gate[r,h])` (f32, broadcast over head_dim).
    pub fn mul_softplus_head(
        &self,
        x: &mut CudaSlice<f32>,
        gate: &CudaSlice<f32>,
        n_heads: usize,
        head_dim: usize,
        rows: usize,
    ) -> Result<(), GpuError> {
        self.mul_softplus_head_at(x, gate, 0, n_heads, head_dim, rows)
    }

    /// `mul_softplus_head` reading the gate at ELEMENT offset `gate_off` -
    /// the merged-projection consumer (gate rows of a fused [q|k|gate] GEMV
    /// landing). Same kernel, shifted base pointer.
    pub fn mul_softplus_head_at(
        &self,
        x: &mut CudaSlice<f32>,
        gate: &CudaSlice<f32>,
        gate_off: usize,
        n_heads: usize,
        head_dim: usize,
        rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .mul_softplus_head
            .ok_or(GpuError::MissingOp("mul_softplus_head"))?;
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (gp, _g2) = gate.device_ptr(&self.stream);
        // SAFETY: ABI contract; x [rows, n_heads, head_dim], gate read from
        // element gate_off as [rows, n_heads]
        check(unsafe {
            f(
                xp as *mut _,
                (gp + (gate_off * 4) as u64) as *const _,
                n_heads as u32,
                head_dim as u32,
                rows as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Plain SwiGLU in place on `gate`: `gate[i] = silu(gate[i]) * up[i]`, `n`
    /// elements (the standard Llama/Qwen FFN activation).
    pub fn swiglu(
        &self,
        gate: &mut CudaSlice<f32>,
        up: &CudaSlice<f32>,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self.kernels.swiglu.ok_or(GpuError::MissingOp("swiglu"))?;
        let (gp, _g1) = gate.device_ptr_mut(&self.stream);
        let (up_p, _g2) = up.device_ptr(&self.stream);
        check(unsafe { f(gp as *mut _, up_p as *const _, n as u32, self.stream_ptr()) })
    }

    /// slot 570: `swiglu` with a bf16 mirror of the result. Same values as
    /// [`Self::swiglu`]; `Ok(false)` means the slot is absent.
    pub fn swiglu_mir(
        &self,
        gate: &mut CudaSlice<f32>,
        up: &CudaSlice<f32>,
        out16: &mut CudaSlice<half::bf16>,
        n: usize,
    ) -> Result<bool, GpuError> {
        let Some(f) = self.kernels.swiglu_mir else {
            return Ok(false);
        };
        let (up_p, _g2) = up.device_ptr(&self.stream);
        let (mp, _g3) = out16.device_ptr_mut(&self.stream);
        let (gp, _g1) = gate.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 570)
        check(unsafe {
            f(
                gp as *mut _,
                up_p as *const _,
                mp as *mut _,
                n as u32,
                self.stream_ptr(),
            )
        })?;
        Ok(true)
    }

    /// SwiGLU over a fused gate|up GEMM output ([rows, 2*ff] with per-row
    /// [gate|up] halves) into a packed [rows, ff] buffer - the merged
    /// gate_up plane's epilogue. Bit-identical values to `swiglu`.
    pub fn swiglu_fused(
        &self,
        fused: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        ff: usize,
        rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .swiglu_fused
            .ok_or(GpuError::MissingOp("swiglu_fused"))?;
        let (fp, _g1) = fused.device_ptr(&self.stream);
        let (op, _g2) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                fp as *const _,
                op as *mut _,
                ff as u32,
                rows as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 534: [`Self::swiglu_fused`] over an INTERLEAVED [rows, 2ff]
    /// landing (gate at 2j, up at 2j+1 -- `Nvf4Plane::gu_pairs`).
    pub fn has_swiglu_fused_il(&self) -> bool {
        self.kernels.swiglu_fused_il.is_some()
    }

    pub fn swiglu_fused_il(
        &self,
        fused: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        ff: usize,
        rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .swiglu_fused_il
            .ok_or(GpuError::MissingOp("swiglu_fused_il"))?;
        let (fp, _g1) = fused.device_ptr(&self.stream);
        let (op, _g2) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                fp as *const _,
                op as *mut _,
                ff as u32,
                rows as u32,
                self.stream_ptr(),
            )
        })
    }

    /// slot 459: DFlash2's grouped dynamic convolution over `[r, embd]` f32
    /// rows - `out[row][c] = sum_t (base[side][t][c] + delta[row][side][t][g])
    /// * h[row-t][c]`, masked to `row % rows_per_block >= t`, with
    /// `g = c / group_size`.
    ///
    /// `rows_per_block` is the RUNTIME block length (k+1), not the trained
    /// `block_size`: the draft plane packs one block per slot back to back, so
    /// the mask is the only thing keeping a tap from convolving one slot's
    /// leading row against the previous slot's trailing one.
    ///
    /// `out` must be a distinct plane from `h` - the kernel reads row-1 while
    /// writing row and blocks are unordered.
    #[allow(clippy::too_many_arguments)]
    pub fn dflash_conv(
        &self,
        h: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        base: &CudaSlice<f32>,
        delta: &CudaSlice<f32>,
        side: usize,
        embd: usize,
        taps: usize,
        num_groups: usize,
        group_size: usize,
        rows_per_block: usize,
        r: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .dflash_conv
            .ok_or(GpuError::MissingOp("dflash_conv"))?;
        let (hp, _g1) = h.device_ptr(&self.stream);
        let (op, _g2) = out.device_ptr_mut(&self.stream);
        let (bp, _g3) = base.device_ptr(&self.stream);
        let (dp, _g4) = delta.device_ptr(&self.stream);
        // SAFETY: ABI contract; the entry re-checks the channel/group geometry
        // (rc -2) and refuses an aliased out, so a bad shape errors instead of
        // corrupting.
        check(unsafe {
            f(
                hp as *const _,
                op as *mut _,
                bp as *const _,
                dp as *const _,
                side as u32,
                embd as u32,
                taps as u32,
                num_groups as u32,
                group_size as u32,
                rows_per_block as u32,
                r as u32,
                self.stream_ptr(),
            )
        })
    }

    /// slot 459 presence - DFlash2 checkpoints refuse to arm without it.
    pub fn has_dflash_conv(&self) -> bool {
        self.kernels.dflash_conv.is_some()
    }

    /// slot 460: unpack `topk_rows`' interleaved (id, logit-bits) pairs into
    /// the flat id plane `kquant_gather` takes, with each block's anchor token
    /// appended at `r*k + b` so one gather covers candidates AND anchors.
    #[allow(clippy::too_many_arguments)]
    pub fn dflash_cand_ids(
        &self,
        topk: &CudaSlice<u32>,
        toks: &CudaSlice<u32>,
        ids: &mut CudaSlice<u32>,
        k: usize,
        rows_per_block: usize,
        r: usize,
        vocab: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .dflash_cand_ids
            .ok_or(GpuError::MissingOp("dflash_cand_ids"))?;
        let (tp, _g1) = topk.device_ptr(&self.stream);
        let (kp, _g2) = toks.device_ptr(&self.stream);
        let (ip, _g3) = ids.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                tp as *const _,
                kp as *const _,
                ip as *mut _,
                k as u32,
                rows_per_block as u32,
                r as u32,
                vocab as u32,
                self.stream_ptr(),
            )
        })
    }

    /// slot 461: DFlash2's candidate-selector walk - greedy forward pass over
    /// the bilinear edge scores, one CTA per block, writing the chosen token
    /// per row into `out`. Positions `1..rows_per_block` are walked; row 0 is
    /// the committed anchor and its logits are not a draft.
    ///
    /// `scale`/`cap` are the drafter's own logit epilogue and must be passed:
    /// the unary term is added to a bilinear score, so the monotone argument
    /// that lets greedy per-row drafting skip the epilogue does not apply.
    #[allow(clippy::too_many_arguments)]
    pub fn dflash_select(
        &self,
        topk: &CudaSlice<u32>,
        pred: &CudaSlice<f32>,
        succ: &CudaSlice<f32>,
        hs: &CudaSlice<f32>,
        out: &mut CudaSlice<u32>,
        scale: f32,
        cap: f32,
        rank: usize,
        k: usize,
        rows_per_block: usize,
        r: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .dflash_select
            .ok_or(GpuError::MissingOp("dflash_select"))?;
        let (tp, _g1) = topk.device_ptr(&self.stream);
        let (pp, _g2) = pred.device_ptr(&self.stream);
        let (sp, _g3) = succ.device_ptr(&self.stream);
        let (hp, _g4) = hs.device_ptr(&self.stream);
        let (op, _g5) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                tp as *const _,
                pp as *const _,
                sp as *const _,
                hp as *const _,
                op as *mut _,
                scale,
                cap,
                rank as u32,
                k as u32,
                rows_per_block as u32,
                r as u32,
                self.stream_ptr(),
            )
        })
    }

    /// slots 460+461 presence - the DFlash2 selector needs both.
    pub fn has_dflash_select(&self) -> bool {
        self.kernels.dflash_select.is_some() && self.kernels.dflash_cand_ids.is_some()
    }

    /// Sampled twin of [`Self::dflash_select`] (rung G, slot 470): per block
    /// `invt[b]` (1/T, 0 = the greedy walk) + `seeds[b]`; the walk is
    /// Gumbel-max over s/T position by position and `q16[row*k + c]`
    /// receives the row's K-way draft distribution (one-hot on greedy
    /// blocks). Same edge scoring as the greedy kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn dflash_select_rs(
        &self,
        topk: &CudaSlice<u32>,
        pred: &CudaSlice<f32>,
        succ: &CudaSlice<f32>,
        hs: &CudaSlice<f32>,
        invt: &CudaSlice<f32>,
        seeds: &CudaSlice<u32>,
        out: &mut CudaSlice<u32>,
        q16: &mut CudaSlice<f32>,
        scale: f32,
        cap: f32,
        rank: usize,
        k: usize,
        rows_per_block: usize,
        r: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .dflash_select_rs
            .ok_or(GpuError::MissingOp("dflash_select_rs"))?;
        let (tp, _g1) = topk.device_ptr(&self.stream);
        let (pp, _g2) = pred.device_ptr(&self.stream);
        let (sp, _g3) = succ.device_ptr(&self.stream);
        let (hp, _g4) = hs.device_ptr(&self.stream);
        let (ip, _g5) = invt.device_ptr(&self.stream);
        let (sdp, _g6) = seeds.device_ptr(&self.stream);
        let (op, _g7) = out.device_ptr_mut(&self.stream);
        let (qp, _g8) = q16.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                tp as *const _,
                pp as *const _,
                sp as *const _,
                hp as *const _,
                ip as *const _,
                sdp as *const _,
                op as *mut _,
                qp as *mut _,
                scale,
                cap,
                rank as u32,
                k as u32,
                rows_per_block as u32,
                r as u32,
                self.stream_ptr(),
            )
        })
    }

    /// slots 470+471 presence - the DFlash2 rejection-sampling arm.
    pub fn has_dflash_rs(&self) -> bool {
        self.kernels.dflash_select_rs.is_some() && self.kernels.dflash_rs_resolve.is_some()
    }

    /// True when the pack ships the fused-plane SwiGLU epilogue.
    pub fn has_swiglu_fused(&self) -> bool {
        self.kernels.swiglu_fused.is_some()
    }

    /// Packed row-slice from a fused GEMM landing ([rows, src_stride]):
    /// dst[r*width+c] = src[r*src_stride + col_off + c] - the split epilogue
    /// for merged projection planes.
    pub fn row_slice(
        &self,
        src: &CudaSlice<f32>,
        dst: &mut CudaSlice<f32>,
        src_stride: usize,
        col_off: usize,
        width: usize,
        rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .row_slice
            .ok_or(GpuError::MissingOp("row_slice"))?;
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (dp, _g2) = dst.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                sp as *const _,
                dp as *mut _,
                src_stride as u32,
                col_off as u32,
                width as u32,
                rows as u32,
                self.stream_ptr(),
            )
        })
    }

    /// True when the pack ships the fused-landing row slice.
    pub fn has_row_slice(&self) -> bool {
        self.kernels.row_slice.is_some()
    }

    /// Up to 4 slices of the same fused landing in one launch. Bit-identical
    /// to the equivalent `row_slice` calls (same per-element expression); it
    /// only removes launches. The fused planes are split immediately after
    /// every GEMM, and at 240 launches/tick that split was the largest
    /// non-GEMM item in the B200 NO-PDL capture.
    pub fn row_slice4(
        &self,
        src: &CudaSlice<f32>,
        src_stride: usize,
        rows: usize,
        parts: &mut [(&mut CudaSlice<f32>, usize, usize)],
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .row_slice4
            .ok_or(GpuError::MissingOp("row_slice4"))?;
        if parts.is_empty() || parts.len() > 4 {
            return Err(GpuError::Driver("row_slice4: 1..=4 parts".into()));
        }
        let (sp, _g0) = src.device_ptr(&self.stream);
        let mut ptr = [0u64; 4];
        let mut ow = [(0u32, 0u32); 4];
        let mut guards = Vec::with_capacity(4);
        for (i, (d, off, w)) in parts.iter_mut().enumerate() {
            let (dp, g) = d.device_ptr_mut(&self.stream);
            ptr[i] = dp;
            ow[i] = (*off as u32, *w as u32);
            guards.push(g);
        }
        check(unsafe {
            f(
                sp as *const _,
                src_stride as u32,
                rows as u32,
                ptr[0] as *mut _,
                ow[0].0,
                ow[0].1,
                ptr[1] as *mut _,
                ow[1].0,
                ow[1].1,
                ptr[2] as *mut _,
                ow[2].0,
                ow[2].1,
                ptr[3] as *mut _,
                ow[3].0,
                ow[3].1,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_row_slice4(&self) -> bool {
        self.kernels.row_slice4.is_some()
    }

    /// swiglu of a fused gate|up landing straight to per-ROW e4m3, one launch.
    /// Bit-identical to `swiglu_fused` + `quantize_e4m3_row`.
    pub fn swiglu_e4m3_row(
        &self,
        fused: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        rs: &mut CudaSlice<f32>,
        ff: usize,
        rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .swiglu_e4m3_row
            .ok_or(GpuError::MissingOp("swiglu_e4m3_row"))?;
        let (fp, _g1) = fused.device_ptr(&self.stream);
        let (qp, _g2) = q.device_ptr_mut(&self.stream);
        let (sp, _g3) = rs.device_ptr_mut(&self.stream);
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

    pub fn has_swiglu_e4m3_row(&self) -> bool {
        self.kernels.swiglu_e4m3_row.is_some()
    }

    /// x (f32) += y interpreted as bf16 (the o16 down-GEMM's tail residual).
    pub fn add_b16(
        &self,
        x: &mut CudaSlice<f32>,
        y_b16: &CudaSlice<f32>,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .add_inplace_b16
            .ok_or(GpuError::MissingOp("add_inplace_b16"))?;
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (yp, _g2) = y_b16.device_ptr(&self.stream);
        check(unsafe { f(xp as *mut _, yp as *const _, n as u32, self.stream_ptr()) })
    }

    /// True when the bf16 tail-add ships.
    pub fn has_add_b16(&self) -> bool {
        self.kernels.add_inplace_b16.is_some()
    }

    /// GEGLU in place on `gate`: `gate[i] = gelu_tanh(gate[i]) * up[i]`, `n`
    /// elements - the gemma4 FFN activation (ggml-exact GELU constant).
    pub fn geglu(
        &self,
        gate: &mut CudaSlice<f32>,
        up: &CudaSlice<f32>,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self.kernels.geglu.ok_or(GpuError::MissingOp("geglu"))?;
        let (gp, _g1) = gate.device_ptr_mut(&self.stream);
        let (up_p, _g2) = up.device_ptr(&self.stream);
        check(unsafe { f(gp as *mut _, up_p as *const _, n as u32, self.stream_ptr()) })
    }

    /// Gated-FFN fold over SPLIT gate/up buffers, dispatched on the model's
    /// activation. These two carriers predate the `pd_glu_act` template - the
    /// SiLU arm is the pack's own `pd_swiglu`, whose `g / (1 + expf(-g))` is
    /// the same expression the template emits, so the split and concat paths
    /// agree bit-for-bit at either activation.
    pub fn glu(
        &self,
        gate: &mut CudaSlice<f32>,
        up: &CudaSlice<f32>,
        n: usize,
        act: GluAct,
    ) -> Result<(), GpuError> {
        match act {
            GluAct::Gelu => self.geglu(gate, up, n),
            GluAct::Silu => self.swiglu(gate, up, n),
        }
    }

    /// `rope_yarn_batch` with ggml freq_factors: pair k's theta divided by
    /// `factors[k]` (None = plain rope). gemma4 global layers pass rope_freqs
    /// (1e30 entries => frozen pairs = partial rotary).
    ///
    /// `neox` picks the pair layout: true = half-split `(k, k+half)`
    /// (ggml's `ROPE_TYPE_NEOX` - gemma4, qwen35, laguna, gpt-oss), false =
    /// interleaved `(2k, 2k+1)` (`ROPE_TYPE_NORM` - muse-glimmer, granite and
    /// the llama-arch lineage). Same angles either way; only the pairing
    /// differs, and the wrong one degrades quality without ever failing.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_factors_batch(
        &self,
        x: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        factors: Option<&CudaSlice<f32>>,
        n_heads: usize,
        head_dim: usize,
        params: (f32, f32, f32, f32, f32, f32),
        batch: usize,
        neox: bool,
    ) -> Result<(), GpuError> {
        let (f, name) = if neox {
            (self.kernels.rope_factors_batch, "rope_factors_batch")
        } else {
            (
                self.kernels.rope_factors_batch_norm,
                "rope_factors_batch_norm",
            )
        };
        let f = f.ok_or(GpuError::MissingOp(name))?;
        let (theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale) = params;
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (pp, _g2) = positions.device_ptr(&self.stream);
        let fac_guard = factors.map(|s| s.device_ptr(&self.stream));
        let fp = match &fac_guard {
            Some((p, _)) => *p as *const core::ffi::c_void,
            None => std::ptr::null(),
        };
        check(unsafe {
            f(
                xp as *mut _,
                pp as *const _,
                fp,
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

    /// Split the Qwen3.5 full-attn joint QG projection `qg` [n_tokens,
    /// n_heads*2*head_dim] (per head query||gate) into contiguous `q_out` and
    /// `gate_out`, each [n_tokens, n_heads*head_dim].
    pub fn split_qg(
        &self,
        qg: &CudaSlice<f32>,
        q_out: &mut CudaSlice<f32>,
        gate_out: &mut CudaSlice<f32>,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .split_qg
            .ok_or(GpuError::MissingOp("split_qg"))?;
        let (qp, _g1) = qg.device_ptr(&self.stream);
        let (qop, _g2) = q_out.device_ptr_mut(&self.stream);
        let (gop, _g3) = gate_out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                qp as *const _,
                qop as *mut _,
                gop as *mut _,
                n_tokens as u32,
                n_heads as u32,
                head_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Gather one embedding row selected by a device token id into `out` [embd]
    /// (graph-capturable: the token id is read from device, not a host address).
    pub fn embed_gather(
        &self,
        table: &CudaSlice<f32>,
        token: &CudaSlice<u32>,
        out: &mut CudaSlice<f32>,
        embd: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .embed_gather
            .ok_or(GpuError::MissingOp("embed_gather"))?;
        let (tp, _g1) = table.device_ptr(&self.stream);
        let (kp, _g2) = token.device_ptr(&self.stream);
        let (op, _g3) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                tp as *const _,
                kp as *const _,
                op as *mut _,
                embd as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Greedy decode epilogue for a graph-resident loop: argmax(`logits`) -> next
    /// token, then advance all per-token state on-device (write `token`, append to
    /// `out_ids[step]`, bump `step`, set `pos`/`mrope` to pos+1). All-device so the
    /// captured graph replays with no host round-trip. `out_ids` must hold at least
    /// `step`+1 ids across a chunk; the caller zeroes `step` at chunk start.
    #[allow(clippy::too_many_arguments)]
    pub fn argmax_advance(
        &self,
        logits: &CudaSlice<f32>,
        vocab: usize,
        pmax: &mut CudaSlice<f32>,
        pidx: &mut CudaSlice<u32>,
        token: &mut CudaSlice<u32>,
        pos: &mut CudaSlice<u32>,
        mrope: &mut CudaSlice<u32>,
        out_ids: &mut CudaSlice<u32>,
        step: &mut CudaSlice<u32>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .argmax_advance
            .ok_or(GpuError::MissingOp("argmax_advance"))?;
        let n_parts = pmax.len() as u32;
        let (lp, _g1) = logits.device_ptr(&self.stream);
        let (mxp, _g2) = pmax.device_ptr_mut(&self.stream);
        let (ixp, _g3) = pidx.device_ptr_mut(&self.stream);
        let (tp, _g4) = token.device_ptr_mut(&self.stream);
        let (pp, _g5) = pos.device_ptr_mut(&self.stream);
        let (mp, _g6) = mrope.device_ptr_mut(&self.stream);
        let (op, _g7) = out_ids.device_ptr_mut(&self.stream);
        let (sp, _g8) = step.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                lp as *const _,
                vocab as u32,
                mxp as *mut _,
                ixp as *mut _,
                n_parts,
                tp as *mut _,
                pp as *mut _,
                mp as *mut _,
                op as *mut _,
                sp as *mut _,
                self.stream_ptr(),
            )
        })
    }

    /// Row-wise LayerNorm (mean/var + weight + bias) - the ViT norm.
    pub fn layernorm(
        &self,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        rows: usize,
        n: usize,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .layernorm
            .ok_or(GpuError::MissingOp("layernorm"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let (bp, _g3) = b.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                xp as *const _,
                wp as *const _,
                bp as *const _,
                op as *mut _,
                rows as u32,
                n as u32,
                eps,
                self.stream_ptr(),
            )
        })
    }

    /// In-place GELU (tanh approximation - ggml_gelu_f32) over n elements.
    pub fn gelu(&self, x: &mut CudaSlice<f32>, n: usize) -> Result<(), GpuError> {
        let f = self.kernels.gelu.ok_or(GpuError::MissingOp("gelu"))?;
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        check(unsafe { f(xp as *mut _, n as u64, self.stream_ptr()) })
    }

    /// In-place exact-erf GELU - `0.5x(1 + erf(x/√2))`, ggml's `gelu_erf`.
    /// Deliberately not [`Self::gelu`]: granite-vision runs the tanh
    /// approximation in its tower (its GGUF sets `clip.use_gelu`) and this one
    /// in the Q-Former FFN. Swapping them is silent - fluent, wrong features.
    pub fn gelu_erf(&self, x: &mut CudaSlice<f32>, n: usize) -> Result<(), GpuError> {
        let f = self
            .kernels
            .gelu_erf
            .ok_or(GpuError::MissingOp("gelu_erf"))?;
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        check(unsafe { f(xp as *mut _, n as u64, self.stream_ptr()) })
    }

    /// [`Self::layernorm`] writing f16 directly (the GEMM staging dtype) -
    /// kills the LN -> [`Self::convert_f32_f16`] round-trip. Bit-identical to
    /// that chain: f32 memory holds no extra precision, so rounding the
    /// register value equals storing f32 then converting.
    pub fn layernorm_f16(
        &self,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        out: &mut CudaSlice<f16>,
        rows: usize,
        n: usize,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .layernorm_f16
            .ok_or(GpuError::MissingOp("layernorm_f16"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let (bp, _g3) = b.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                xp as *const _,
                wp as *const _,
                bp as *const _,
                op as *mut _,
                rows as u32,
                n as u32,
                eps,
                self.stream_ptr(),
            )
        })
    }

    /// Fused bias + tanh-GELU + f16 store: `out[r][i] = f16(gelu(x[r][i] + bias[i]))`
    /// - replaces [`Self::bias_add`] -> [`Self::gelu`] -> [`Self::convert_f32_f16`]
    ///   on the tower FFN plane. `x` stays unbiased in memory.
    pub fn gelu_bias_f16(
        &self,
        x: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        out: &mut CudaSlice<f16>,
        rows: usize,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .gelu_bias_f16
            .ok_or(GpuError::MissingOp("gelu_bias_f16"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (bp, _g2) = bias.device_ptr(&self.stream);
        let (op, _g3) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                xp as *const _,
                bp as *const _,
                op as *mut _,
                rows as u32,
                n as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Erf twin of [`Self::gelu_bias_f16`] (the projector FFN - see
    /// [`Self::gelu_erf`] on why the two GELUs must never be swapped).
    pub fn gelu_erf_bias_f16(
        &self,
        x: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        out: &mut CudaSlice<f16>,
        rows: usize,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .gelu_erf_bias_f16
            .ok_or(GpuError::MissingOp("gelu_erf_bias_f16"))?;
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (bp, _g2) = bias.device_ptr(&self.stream);
        let (op, _g3) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                xp as *const _,
                bp as *const _,
                op as *mut _,
                rows as u32,
                n as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Fused residual + projection bias: `x[r][i] += src[r][i] + bias[i]` -
    /// replaces [`Self::bias_add`] on `src` followed by [`Self::add`]. Same
    /// association as the unfused chain, so bit-identical; `src` stays
    /// unbiased in memory (taps that need the biased value add it on host).
    pub fn add_bias_res(
        &self,
        x: &mut CudaSlice<f32>,
        src: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        rows: usize,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .add_bias_res
            .ok_or(GpuError::MissingOp("add_bias_res"))?;
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (sp, _g2) = src.device_ptr(&self.stream);
        let (bp, _g3) = bias.device_ptr(&self.stream);
        check(unsafe {
            f(
                xp as *mut _,
                sp as *const _,
                bp as *const _,
                rows as u32,
                n as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Row gather with an averaging fan-in: `out[r] = mean(src[idx[r*k + j]])`.
    /// `k == 1` is a plain `get_rows` (bit-exact - the scale is 1.0); `k == 4`
    /// is a 2×2 average pool driven entirely by the index table.
    pub fn gather_rows_avg(
        &self,
        src: &CudaSlice<f32>,
        idx: &CudaSlice<u32>,
        out: &mut CudaSlice<f32>,
        rows: usize,
        k: usize,
        width: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .gather_rows_avg
            .ok_or(GpuError::MissingOp("gather_rows_avg"))?;
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (ip, _g2) = idx.device_ptr(&self.stream);
        let (op, _g3) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                sp as *const _,
                ip as *const _,
                op as *mut _,
                rows as u32,
                k as u32,
                width as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Broadcast row add `x[r][d] += src[r % src_rows][d]` - a learned table
    /// that repeats across windows (the Q-Former's queries and encoder
    /// positions). `src_rows == 1` is [`Self::bias_add`].
    pub fn add_rows_bcast(
        &self,
        x: &mut CudaSlice<f32>,
        src: &CudaSlice<f32>,
        rows: usize,
        src_rows: usize,
        width: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .add_rows_bcast
            .ok_or(GpuError::MissingOp("add_rows_bcast"))?;
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (sp, _g2) = src.device_ptr(&self.stream);
        check(unsafe {
            f(
                xp as *mut _,
                sp as *const _,
                rows as u32,
                src_rows as u32,
                width as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Broadcast bias add: `x[row][i] += bias[i]`.
    pub fn bias_add(
        &self,
        x: &mut CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        rows: usize,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .bias_add
            .ok_or(GpuError::MissingOp("bias_add"))?;
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (bp, _g2) = bias.device_ptr(&self.stream);
        check(unsafe {
            f(
                xp as *mut _,
                bp as *const _,
                rows as u32,
                n as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Vision M-RoPE (ROPE_TYPE_VISION, indep_sects) over q or k rows.
    pub fn mrope_vision(
        &self,
        x: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
        theta_scale: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .mrope_vision
            .ok_or(GpuError::MissingOp("mrope_vision"))?;
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (pp, _g2) = positions.device_ptr(&self.stream);
        check(unsafe {
            f(
                xp as *mut _,
                pp as *const _,
                n_tokens as u32,
                n_heads as u32,
                head_dim as u32,
                theta_scale,
                self.stream_ptr(),
            )
        })
    }

    /// [`Self::mrope_vision`] with the q/k projection bias folded into the
    /// load: `x = rope(x + bias)` in one pass (the pair walk touches every
    /// head element exactly once). Bias broadcast per head-feature
    /// (n_heads·head_dim). Bit-identical to bias_add -> mrope_vision.
    pub fn mrope_vision_bias(
        &self,
        x: &mut CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
        theta_scale: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .mrope_vision_bias
            .ok_or(GpuError::MissingOp("mrope_vision_bias"))?;
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (bp, _g2) = bias.device_ptr(&self.stream);
        let (pp, _g3) = positions.device_ptr(&self.stream);
        check(unsafe {
            f(
                xp as *mut _,
                bp as *const _,
                pp as *const _,
                n_tokens as u32,
                n_heads as u32,
                head_dim as u32,
                theta_scale,
                self.stream_ptr(),
            )
        })
    }

    /// Whether the pack lands a qwen-family (or PaddleOCR-VL) tower's attention on halves:
    /// [`Self::mrope_vision_qkv_h`] (slot 833) feeding
    /// [`Self::vision_attn_h`] (slot 620). Without both, the tower keeps the
    /// f32 chain, which lands the same bits.
    pub fn has_vision_qkv_h(&self) -> bool {
        self.kernels.mrope_vision_qkv_h.is_some() && self.kernels.vision_attn_h.is_some()
    }

    /// The tower's three f32 projection planes landed as
    /// [`Self::vision_attn_h`]'s halves in one pass:
    /// `q16 = f16(rope(q + bq) * q_mul)`, `k16 = f16(rope(k + bk))`,
    /// `v16 = f16(v + bv)`. With `q_mul = 1/sqrt(head_dim)` the half attention
    /// on these is bit-for-bit `mrope_vision_bias` + `bias_add` +
    /// `vision_attn_at` followed by `convert_f32_f16` - the rotation is the
    /// same expression, and each plane's one round is the one the f32
    /// attention does on its way into the fragments.
    #[allow(clippy::too_many_arguments)]
    pub fn mrope_vision_qkv_h(
        &self,
        (q, k, v): (&CudaSlice<f32>, &CudaSlice<f32>, &CudaSlice<f32>),
        (bq, bk, bv): (&CudaSlice<f32>, &CudaSlice<f32>, &CudaSlice<f32>),
        positions: &CudaSlice<u32>,
        (q16, k16, v16): (
            &mut CudaSlice<f16>,
            &mut CudaSlice<f16>,
            &mut CudaSlice<f16>,
        ),
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
        theta_scale: f32,
        q_mul: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .mrope_vision_qkv_h
            .ok_or(GpuError::MissingOp("mrope_vision_qkv_h"))?;
        let (n, w) = (n_tokens * n_heads * head_dim, n_heads * head_dim);
        if !head_dim.is_multiple_of(2)
            || [q.len(), k.len(), v.len(), q16.len(), k16.len(), v16.len()]
                .iter()
                .any(|&l| l < n)
            || [bq.len(), bk.len(), bv.len()].iter().any(|&l| l < w)
            || positions.len() < 2 * n_tokens
        {
            return Err(oob("mrope_vision_qkv_h: buffers under the tower geometry"));
        }
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = k.device_ptr(&self.stream);
        let (vp, _g3) = v.device_ptr(&self.stream);
        let (bqp, _g4) = bq.device_ptr(&self.stream);
        let (bkp, _g5) = bk.device_ptr(&self.stream);
        let (bvp, _g6) = bv.device_ptr(&self.stream);
        let (pp, _g7) = positions.device_ptr(&self.stream);
        let (q16p, _g8) = q16.device_ptr_mut(&self.stream);
        let (k16p, _g9) = k16.device_ptr_mut(&self.stream);
        let (v16p, _g10) = v16.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 833); bounds checked above
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                bqp as *const _,
                bkp as *const _,
                bvp as *const _,
                pp as *const _,
                q16p as *mut _,
                k16p as *mut _,
                v16p as *mut _,
                n_tokens as u32,
                n_heads as u32,
                head_dim as u32,
                theta_scale,
                q_mul,
                self.stream_ptr(),
            )
        })
    }

    /// Whether the pack folds a tower FFN's seams into its GEMMs: slot 834
    /// (the up landing with bias + GELU) and slot 835 (a projection onto the
    /// residual stream), beside the half attention's slots.
    pub fn has_tower_ffn_fused(&self) -> bool {
        self.has_vision_qkv_h()
            && self.kernels.f16_gemm_h_gelu_tanhf.is_some()
            && self.kernels.f16_gemm_bias_res.is_some()
    }

    /// A tower FFN up landing `y16 = f16(gelu(W x + bias))` in
    /// [`Self::gelu_bias_f16`]'s tanh form, the f32 up plane never landing:
    /// bit-for-bit [`Self::matvec_batch_f16`] + `gelu_bias_f16`. `scratch` (an
    /// f32 `[batch, out]` plane) is where the pack runs those two passes at a
    /// shape its fused landing cannot reproduce. `x16` and `y16` must not
    /// overlap.
    pub fn matvec_batch_f16_gelu_tanh(
        &self,
        w: &HalfTensor,
        x16: &CudaSlice<f16>,
        y16: &mut CudaSlice<f16>,
        bias: &CudaSlice<f32>,
        scratch: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .f16_gemm_h_gelu_tanhf
            .ok_or(GpuError::MissingOp("f16_gemm_h_gelu_tanhf"))?;
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        if w.buf.len() < in_dim * out_dim
            || x16.len() < batch * in_dim
            || y16.len() < batch * out_dim
            || scratch.len() < batch * out_dim
            || bias.len() < out_dim
        {
            return Err(oob(
                "f16_gemm_h_gelu_tanhf: buffers under the GEMM geometry",
            ));
        }
        super::basic_ops::gemm_census("B-gemm-f16-h-gelu-tanhf", in_dim, out_dim, batch);
        let (wp, _g1) = w.buf.device_ptr(&self.stream);
        let (xp, _g2) = x16.device_ptr(&self.stream);
        let (bp, _g3) = bias.device_ptr(&self.stream);
        let (yp, _g4) = y16.device_ptr_mut(&self.stream);
        let (sp, _g5) = scratch.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 834); bounds checked above
        check(unsafe {
            f(
                wp as *const _,
                xp as *const _,
                yp as *mut _,
                bp as *const _,
                sp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// A projection onto an f32 residual stream, `x += W a + bias` in
    /// [`Self::add_bias_res`]'s order, the projection plane never landing:
    /// bit-for-bit [`Self::matvec_batch_f16`] into `scratch` + `add_bias_res`,
    /// which is what the pack runs where its fused landing cannot reproduce it.
    pub fn matvec_batch_f16_bias_res(
        &self,
        w: &HalfTensor,
        a16: &CudaSlice<f16>,
        x: &mut CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        scratch: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .f16_gemm_bias_res
            .ok_or(GpuError::MissingOp("f16_gemm_bias_res"))?;
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        if w.buf.len() < in_dim * out_dim
            || a16.len() < batch * in_dim
            || x.len() < batch * out_dim
            || scratch.len() < batch * out_dim
            || bias.len() < out_dim
        {
            return Err(oob("f16_gemm_bias_res: buffers under the GEMM geometry"));
        }
        super::basic_ops::gemm_census("B-gemm-f16-bias-res", in_dim, out_dim, batch);
        let (wp, _g1) = w.buf.device_ptr(&self.stream);
        let (ap, _g2) = a16.device_ptr(&self.stream);
        let (bp, _g3) = bias.device_ptr(&self.stream);
        let (yp, _g4) = x.device_ptr_mut(&self.stream);
        let (sp, _g5) = scratch.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 835); bounds checked above
        check(unsafe {
            f(
                wp as *const _,
                ap as *const _,
                yp as *mut _,
                bp as *const _,
                sp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Q8_0 embedding gather with fused scale: `out[t] = q8_row(tokens[t])·scale`.
    /// Token ids are device memory - the graph-capturable decode input.
    /// Q8_0 row-gather with unit scale - drop-in for the f32 `embed_gather_batch`
    /// when the embedding table is kept resident as Q8_0 (input lookup only, no
    /// sqrt(n_embd) scaling). Dequantizes only the gathered rows on the fly.
    pub fn embed_gather_batch_q8(
        &self,
        table: &QuantTensor,
        tokens: &CudaSlice<u32>,
        out: &mut CudaSlice<f32>,
        embd: usize,
        n_tokens: usize,
    ) -> Result<(), GpuError> {
        self.embed_gather_q8(table, tokens, out, embd, n_tokens, 1.0)
    }

    pub fn embed_gather_q8(
        &self,
        table: &QuantTensor,
        tokens: &CudaSlice<u32>,
        out: &mut CudaSlice<f32>,
        embd: usize,
        n_tokens: usize,
        scale: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .embed_gather_q8
            .ok_or(GpuError::MissingOp("embed_gather_q8"))?;
        if table.ty != GgmlType::Q8_0 {
            return Err(GpuError::NoKernel {
                name: "<embed_gather_q8>".into(),
                ty: table.ty,
            });
        }
        let (tp, _g1) = table.bytes.device_ptr(&self.stream);
        let (kp, _g2) = tokens.device_ptr(&self.stream);
        let (op, _g3) = out.device_ptr_mut(&self.stream);
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

    /// The pack can gather embedding rows straight out of a REPACKED Q8_0
    /// plane (slot 832) - what lets a tied embedding share the LM head's
    /// plane rather than keep a raw copy of the same matrix.
    pub fn has_embed_gather_q8r(&self) -> bool {
        self.kernels.embed_gather_q8r.is_some()
    }

    /// [`Self::embed_gather_q8`] over a repacked plane: rows of `table`
    /// (`dims[1]` of them, `dims[0] == embd` wide) for each device token id,
    /// times `scale`. Bit-identical to the raw gather on the same matrix.
    pub fn embed_gather_q8r(
        &self,
        table: &RepackedQ8,
        tokens: &CudaSlice<u32>,
        out: &mut CudaSlice<f32>,
        embd: usize,
        n_tokens: usize,
        scale: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .embed_gather_q8r
            .ok_or(GpuError::MissingOp("embed_gather_q8r"))?;
        if table.dims[0] != embd || !embd.is_multiple_of(32) {
            return Err(GpuError::Unsupported(format!(
                "embed_gather_q8r: table row width {} for embd {embd} (needs equal, a multiple of 32)",
                table.dims[0]
            )));
        }
        let (dp, _g1) = table.data.device_ptr(&self.stream);
        let (sp, _g2) = table.scale.device_ptr(&self.stream);
        let (kp, _g3) = tokens.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        // SAFETY: pack ABI v1 contract; pointers + stream live across the call
        check(unsafe {
            f(
                dp as *const _,
                sp as *const _,
                kp as *const _,
                op as *mut _,
                embd as u32,
                n_tokens as u32,
                scale,
                self.stream_ptr(),
            )
        })
    }

    /// `q8_0_gemv_repacked` writing at an element OFFSET into `y` - lets the
    /// separate q/k/v (or gate/up) weights land in one concatenated per-row
    /// buffer without duplicating weight bytes.
    pub fn q8_0_gemv_repacked_at(
        &self,
        w: &RepackedQ8,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        y_off: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_gemv_repacked
            .ok_or(GpuError::MissingOp("q8_0_gemv_repacked"))?;
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (scp, _g2) = w.scale.device_ptr(&self.stream);
        let (xp, _g3) = x.device_ptr(&self.stream);
        let (yp, _g4) = y.device_ptr_mut(&self.stream);
        // SAFETY: same allocation, offset output rowlet; caller sizes y
        check(unsafe {
            f(
                dp as *const _,
                scp as *const _,
                core::ptr::null(),
                xp as *const _,
                (yp + (y_off * 4) as u64) as *mut _,
                in_dim as u32,
                out_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Scalar multiply in place: `x[..n] *= s` (ggml_scale's shape). Granite's
    /// embedding_multiplier and logits_scaling; not the same as `add_scale`,
    /// which scales the SUM `(x + y)·s`, nor `scale_add`, which is `x += w·y`.
    pub fn scale(&self, x: &mut CudaSlice<f32>, s: f32, n: usize) -> Result<(), GpuError> {
        let f = self
            .kernels
            .scale_f32
            .ok_or(GpuError::MissingOp("scale_f32"))?;
        let (xp, _g) = x.device_ptr_mut(&self.stream);
        // SAFETY: pack ABI v1 contract; pointer + stream live across the call
        check(unsafe { f(xp as *mut _, s, n as u32, self.stream_ptr()) })
    }

    /// Fused residual + stream scale: `x = (x + y)·s`.
    pub fn add_scale(
        &self,
        x: &mut CudaSlice<f32>,
        y: &CudaSlice<f32>,
        s: f32,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .add_scale
            .ok_or(GpuError::MissingOp("add_scale"))?;
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (yp, _g2) = y.device_ptr(&self.stream);
        check(unsafe { f(xp as *mut _, yp as *const _, s, n as u32, self.stream_ptr()) })
    }

    /// Gated-FFN fold over concatenated gate|up rows [rows, 2*ff]; gate half
    /// updated in place. `act` is the model's activation (see [`GluAct`]) -
    /// the two arms are the same kernel on the pack's `pd_glu_act` template.
    pub fn glu_pair(
        &self,
        x: &mut CudaSlice<f32>,
        ff: usize,
        rows: usize,
        act: GluAct,
    ) -> Result<(), GpuError> {
        let (f, name) = match act {
            GluAct::Gelu => (self.kernels.geglu_pair, "geglu_pair"),
            GluAct::Silu => (self.kernels.swiglu_pair, "swiglu_pair"),
        };
        let f = f.ok_or(GpuError::MissingOp(name))?;
        let (xp, _g) = x.device_ptr_mut(&self.stream);
        check(unsafe { f(xp as *mut _, ff as u32, rows as u32, self.stream_ptr()) })
    }

    /// Fused post-norm + residual + scale: `x = (x + rmsnorm(proj)·w)·s`.
    #[allow(clippy::too_many_arguments)]
    /// A sandwich block's post-norm fused with the NEXT norm (slot 777):
    /// `x += rmsnorm(proj) * w_post * s`, then `xn = rmsnorm(x) * w_pre` into
    /// the f32 plane and/or its bf16 copy - bit-identical to
    /// [`Self::rmsnorm_add_scale`] + `rmsnorm_batch` (+ `convert_f32_bf16`).
    /// `Ok(false)` (nothing launched) when the pack lacks the slot or the
    /// shape is outside its band (rows < 256, ragged or unaligned planes):
    /// the caller runs the pair.
    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm_add_scale_norm(
        &self,
        x: &mut CudaSlice<f32>,
        proj: &CudaSlice<f32>,
        w_post: &CudaSlice<f32>,
        w_pre: &CudaSlice<f32>,
        xn: Option<&mut CudaSlice<f32>>,
        x16: Option<&mut CudaSlice<half::bf16>>,
        n: usize,
        eps: f32,
        s: f32,
        rows: usize,
    ) -> Result<bool, GpuError> {
        let Some(f) = self.kernels.rmsnorm_add_scale_norm else {
            return Ok(false);
        };
        if rows < 256 || !n.is_multiple_of(4) {
            return Ok(false);
        }
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (pp, _g2) = proj.device_ptr(&self.stream);
        let (wa, _g3) = w_post.device_ptr(&self.stream);
        let (wb, _g4) = w_pre.device_ptr(&self.stream);
        let xn = xn.map(|v| v.device_ptr_mut(&self.stream));
        let x16 = x16.map(|v| v.device_ptr_mut(&self.stream));
        // SAFETY: ABI contract (slot 777); every plane holds rows * n
        let rc = unsafe {
            f(
                xp as *mut _,
                pp as *const _,
                wa as *const _,
                wb as *const _,
                xn.as_ref()
                    .map_or(std::ptr::null_mut(), |(p, _)| *p as *mut _),
                x16.as_ref()
                    .map_or(std::ptr::null_mut(), |(p, _)| *p as *mut _),
                n as u32,
                eps,
                s,
                rows as u32,
                self.stream_ptr(),
            )
        };
        if rc == -2 {
            return Ok(false);
        }
        check(rc)?;
        Ok(true)
    }

    pub fn rmsnorm_add_scale(
        &self,
        x: &mut CudaSlice<f32>,
        proj: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        n: usize,
        eps: f32,
        s: f32,
        rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .rmsnorm_add_scale
            .ok_or(GpuError::MissingOp("rmsnorm_add_scale"))?;
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (pp, _g2) = proj.device_ptr(&self.stream);
        let (wp, _g3) = w.device_ptr(&self.stream);
        check(unsafe {
            f(
                xp as *mut _,
                pp as *const _,
                wp as *const _,
                n as u32,
                eps,
                s,
                rows as u32,
                self.stream_ptr(),
            )
        })
    }

    /// p16 twin of `rmsnorm_add_scale`: `proj` holds bf16 (the
    /// o16 GEMM epilogue's stream riding the f32-typed scratch).
    pub fn rmsnorm_add_scale_p16(
        &self,
        x: &mut CudaSlice<f32>,
        proj: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        n: usize,
        eps: f32,
        s: f32,
        rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .rmsnorm_add_scale2
            .ok_or(GpuError::MissingOp("rmsnorm_add_scale2"))?;
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (pp, _g2) = proj.device_ptr(&self.stream);
        let (wp, _g3) = w.device_ptr(&self.stream);
        // SAFETY: ABI contract; proj holds bf16 in the f32-typed scratch
        check(unsafe {
            f(
                xp as *mut _,
                pp as *const _,
                wp as *const _,
                n as u32,
                eps,
                s,
                rows as u32,
                1u32,
                self.stream_ptr(),
            )
        })
    }
}
