//! Inkling 2 Mini's DSpark draft op wrappers - pack slots 994-999: the tap
//! mix, the draft's q / k norm + RoPE, its block attention, the recurrent
//! head's gate, a draft step's pick and strided row copies.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use super::error::check;
use super::inkling2::span;
use super::{GpuError, GpuExecutor, Inkling2KvLayout};

impl GpuExecutor {
    /// Whether the pack carries the DSpark draft's ops (slots 994-998).
    pub fn has_inkling2_draft(&self) -> bool {
        let k = &self.kernels;
        k.inkling2_mix_norm.is_some()
            && k.inkling2_qk_rope.is_some()
            && k.inkling2_draft_attn.is_some()
            && k.inkling2_markov_gate.is_some()
            && k.inkling2_markov_argmax.is_some()
            && k.inkling2_copy2d.is_some()
            && k.inkling2_conv_stash.is_some()
            && k.inkling2_conv_restore.is_some()
    }

    /// Slot 994: `out[r] = bf16(rmsnorm(sum_t alpha[t] feat[r][t]) *
    /// weight)` - `feat` rows of `taps` planes of `width` at row stride
    /// `fstride`, `alpha` f32 [taps], `weight` bf16 [width].
    #[allow(clippy::too_many_arguments)]
    pub fn inkling2_mix_norm(
        &self,
        (feat, fstride): (&CudaSlice<f32>, usize),
        taps: usize,
        alpha: &CudaSlice<f32>,
        weight: &CudaSlice<u8>,
        out: &mut CudaSlice<f32>,
        rows: usize,
        width: usize,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .inkling2_mix_norm
            .ok_or(GpuError::MissingOp("inkling2_mix_norm"))?;
        span(
            feat.len(),
            0,
            rows,
            fstride,
            taps * width,
            "inkling2_mix_norm feat",
        )?;
        span(out.len(), 0, rows, width, width, "inkling2_mix_norm out")?;
        if alpha.len() < taps || weight.len() < 2 * width {
            return Err(GpuError::Unsupported(
                "inkling2_mix_norm: alpha / weight".into(),
            ));
        }
        let (fp, _g1) = feat.device_ptr(&self.stream);
        let (ap, _g2) = alpha.device_ptr(&self.stream);
        let (wp, _g3) = weight.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 994); planes checked above
        check(unsafe {
            f(
                fp as *const _,
                fstride as u32,
                taps as u32,
                ap as *const _,
                wp as *const _,
                op as *mut _,
                rows as u32,
                width as u32,
                eps,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 995: per (row, 128-wide head) of `x` from column `col0` (row
    /// stride `xs`), in place: with `weight` (bf16 [128]) an RMSNorm x the
    /// weight rounded to bf16; then neox RoPE at `pos[row]` (`inv_freq` f32
    /// [64]), rounded to bf16.
    #[allow(clippy::too_many_arguments)]
    pub fn inkling2_qk_rope(
        &self,
        (x, xs, col0): (&mut CudaSlice<f32>, usize, usize),
        rows: usize,
        heads: usize,
        weight: Option<&CudaSlice<u8>>,
        pos: &CudaSlice<u32>,
        inv_freq: &CudaSlice<f32>,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .inkling2_qk_rope
            .ok_or(GpuError::MissingOp("inkling2_qk_rope"))?;
        span(x.len(), col0, rows, xs, heads * 128, "inkling2_qk_rope x")?;
        if pos.len() < rows || inv_freq.len() < 64 || weight.is_some_and(|w| w.len() < 256) {
            return Err(GpuError::Unsupported(
                "inkling2_qk_rope: pos / freqs / weight".into(),
            ));
        }
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let wg = weight.map(|w| w.device_ptr(&self.stream));
        let wp = wg.as_ref().map_or(0, |g| g.0);
        let (pp, _g2) = pos.device_ptr(&self.stream);
        let (fp, _g3) = inv_freq.device_ptr(&self.stream);
        // SAFETY: ABI contract (slot 995); the plane covers every head of
        // every row (checked above)
        check(unsafe {
            f(
                xp as *mut _,
                xs as u32,
                col0 as u32,
                rows as u32,
                heads as u32,
                wp as *const _,
                pp as *const _,
                fp as *const _,
                eps,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 996: the draft block's attention - `blk` [n_blocks][2] (slot,
    /// first position) for 8 query rows each of `q` (element offset, row
    /// stride), every row seeing its slot's keys up to the block's end (the
    /// last `window + 8` with a window) in a slot-989 store; out rows at
    /// stride `os`. `part` ([n_blocks * 8][heads][splits] x (128 + 2))
    /// when splits > 1.
    #[allow(clippy::too_many_arguments)]
    pub fn inkling2_draft_attn(
        &self,
        q: (&CudaSlice<f32>, usize, usize),
        (blk, n_blocks): (&CudaSlice<u32>, usize),
        (k, v): (&CudaSlice<u8>, &CudaSlice<u8>),
        layout: Inkling2KvLayout<'_>,
        dtype: u32,
        (out, os): (&mut CudaSlice<f32>, usize),
        part: Option<(&mut CudaSlice<f32>, &mut CudaSlice<f32>)>,
        (heads, kv_heads): (usize, usize),
        window: usize,
        splits: usize,
        scale: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .inkling2_draft_attn
            .ok_or(GpuError::MissingOp("inkling2_draft_attn"))?;
        let rows = n_blocks * 8;
        span(
            q.0.len(),
            q.1,
            rows,
            q.2,
            heads * 128,
            "inkling2_draft_attn q",
        )?;
        span(
            out.len(),
            0,
            rows,
            os,
            heads * 128,
            "inkling2_draft_attn out",
        )?;
        if blk.len() < 2 * n_blocks {
            return Err(GpuError::Unsupported("inkling2_draft_attn: blocks".into()));
        }
        if splits > 1
            && part.as_ref().is_none_or(|(o, ml)| {
                o.len() < rows * heads * splits * 128 || ml.len() < rows * heads * splits * 2
            })
        {
            return Err(GpuError::Unsupported(
                "inkling2_draft_attn: split scratch".into(),
            ));
        }
        let (qp, _g1) = q.0.device_ptr(&self.stream);
        let (bp, _g2) = blk.device_ptr(&self.stream);
        let (kp, _g3) = k.device_ptr(&self.stream);
        let (vp, _g4) = v.device_ptr(&self.stream);
        let (op, _g5) = out.device_ptr_mut(&self.stream);
        let (bt, bps, ring) = match layout {
            Inkling2KvLayout::Paged(b, bps) => (Some(b.device_ptr(&self.stream)), bps, 0),
            Inkling2KvLayout::Ring(r) => (None, 0, r),
        };
        let btp = bt.as_ref().map_or(0, |g| g.0);
        let mut part = part;
        let pg = part.as_mut().map(|(o, ml)| {
            (
                o.device_ptr_mut(&self.stream),
                ml.device_ptr_mut(&self.stream),
            )
        });
        let (pop, pmp) = pg.as_ref().map_or((0, 0), |(o, ml)| (o.0, ml.0));
        // the tensor cores (slot 1010) where the pack and the shape allow:
        // the same arguments, NotSupported for another class or head ratio
        if let Some(ft) = self.kernels.inkling2_draft_attn_tc
            && !draft_tc_off()
        {
            // SAFETY: ABI contract (slot 1010); slot 996's, checked above
            let rc = unsafe {
                ft(
                    (qp + (q.1 * 4) as u64) as *const _,
                    q.2 as u32,
                    bp as *const _,
                    n_blocks as u32,
                    kp as *mut _,
                    vp as *mut _,
                    btp as *const _,
                    bps as u32,
                    ring as u32,
                    dtype,
                    op as *mut _,
                    os as u32,
                    pop as *mut _,
                    pmp as *mut _,
                    heads as u32,
                    kv_heads as u32,
                    window as u32,
                    splits as u32,
                    scale,
                    self.stream_ptr(),
                )
            };
            if rc != 801 {
                return check(rc);
            }
        }
        // SAFETY: ABI contract (slot 996); every plane checked above, the
        // stores hold each block's keys (the caller stored them first)
        check(unsafe {
            f(
                (qp + (q.1 * 4) as u64) as *const _,
                q.2 as u32,
                bp as *const _,
                n_blocks as u32,
                kp as *mut _,
                vp as *mut _,
                btp as *const _,
                bps as u32,
                ring as u32,
                dtype,
                op as *mut _,
                os as u32,
                pop as *mut _,
                pmp as *mut _,
                heads as u32,
                kv_heads as u32,
                window as u32,
                splits as u32,
                scale,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 997: the recurrent head's gate - `raw` + `ctx` ([rows][3 rank]
    /// at the given element offsets: the step's input and context terms of
    /// gate | candidate | out), `state` [rows] at row stride `ss` in place,
    /// `latent` [rows][rank].
    #[allow(clippy::too_many_arguments)]
    pub fn inkling2_markov_gate(
        &self,
        raw: &CudaSlice<f32>,
        (ctx, ctx_off): (&CudaSlice<f32>, usize),
        (state, ss): (&mut CudaSlice<f32>, usize),
        latent: &mut CudaSlice<f32>,
        rows: usize,
        rank: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .inkling2_markov_gate
            .ok_or(GpuError::MissingOp("inkling2_markov_gate"))?;
        span(
            raw.len(),
            0,
            rows,
            3 * rank,
            3 * rank,
            "inkling2_markov_gate raw",
        )?;
        span(
            ctx.len(),
            ctx_off,
            rows,
            3 * rank,
            3 * rank,
            "inkling2_markov_gate ctx",
        )?;
        span(state.len(), 0, rows, ss, rank, "inkling2_markov_gate state")?;
        if latent.len() < rows * rank {
            return Err(GpuError::Unsupported("inkling2_markov_gate: latent".into()));
        }
        let (rp, _g1) = raw.device_ptr(&self.stream);
        let (cp, _g2) = ctx.device_ptr(&self.stream);
        let (sp, _g3) = state.device_ptr_mut(&self.stream);
        let (lp, _g4) = latent.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 997); planes checked above
        check(unsafe {
            f(
                rp as *const _,
                (cp + (ctx_off * 4) as u64) as *const _,
                sp as *mut _,
                ss as u32,
                lp as *mut _,
                rows as u32,
                rank as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 998: a draft step's picks into `ids` - per row of `base` (from
    /// an element offset) and `gemm` (row stride `stride`), the first index of the largest
    /// softcapped logit over `vocab` (`fold_bias` bf16 [vocab]); `best` is
    /// [rows] x 8 bytes of scratch.
    #[allow(clippy::too_many_arguments)]
    pub fn inkling2_markov_argmax(
        &self,
        (base, base_off): (&CudaSlice<f32>, usize),
        gemm: &CudaSlice<f32>,
        fold_bias: &CudaSlice<u8>,
        best: &mut CudaSlice<u8>,
        ids: &mut CudaSlice<u32>,
        rows: usize,
        (vocab, stride): (usize, usize),
        cap: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .inkling2_markov_argmax
            .ok_or(GpuError::MissingOp("inkling2_markov_argmax"))?;
        span(
            base.len(),
            base_off,
            rows,
            stride,
            vocab,
            "inkling2_markov_argmax base",
        )?;
        span(
            gemm.len(),
            0,
            rows,
            stride,
            vocab,
            "inkling2_markov_argmax gemm",
        )?;
        if fold_bias.len() < 2 * vocab || best.len() < 8 * rows || ids.len() < rows {
            return Err(GpuError::Unsupported(
                "inkling2_markov_argmax: planes".into(),
            ));
        }
        let (bp, _g1) = base.device_ptr(&self.stream);
        let (gp, _g2) = gemm.device_ptr(&self.stream);
        let (fp, _g3) = fold_bias.device_ptr(&self.stream);
        let (sp, _g4) = best.device_ptr_mut(&self.stream);
        let (ip, _g5) = ids.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 998); planes checked above
        check(unsafe {
            f(
                (bp + (base_off * 4) as u64) as *const _,
                gp as *const _,
                fp as *const _,
                sp as *mut _,
                ip as *mut _,
                rows as u32,
                vocab as u32,
                stride as u32,
                cap,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 999: `rows` rows of `width` f32 from `src` (element offset, row
    /// stride `ss` - 0 broadcasts one row) into `dst` (element offset, row
    /// stride `ds`).
    #[allow(clippy::too_many_arguments)]
    pub fn inkling2_copy2d(
        &self,
        (src, src_off, ss): (&CudaSlice<f32>, usize, usize),
        (dst, dst_off, ds): (&mut CudaSlice<f32>, usize, usize),
        width: usize,
        rows: usize,
    ) -> Result<(), GpuError> {
        span(src.len(), src_off, rows, ss, width, "inkling2_copy2d src")?;
        span(dst.len(), dst_off, rows, ds, width, "inkling2_copy2d dst")?;
        let (dp, _g) = dst.device_ptr_mut(&self.stream);
        self.inkling2_copy2d_raw((src, src_off, ss), (dp, dst_off, ds), width, rows, false)
    }

    /// [`Self::inkling2_copy2d`] into a bf16 plane held as bytes (rounded).
    #[allow(clippy::too_many_arguments)]
    pub fn inkling2_copy2d_bf16(
        &self,
        (src, src_off, ss): (&CudaSlice<f32>, usize, usize),
        (dst, dst_off, ds): (&mut CudaSlice<u8>, usize, usize),
        width: usize,
        rows: usize,
    ) -> Result<(), GpuError> {
        span(
            src.len(),
            src_off,
            rows,
            ss,
            width,
            "inkling2_copy2d_bf16 src",
        )?;
        span(
            dst.len() / 2,
            dst_off,
            rows,
            ds,
            width,
            "inkling2_copy2d_bf16 dst",
        )?;
        let (dp, _g) = dst.device_ptr_mut(&self.stream);
        self.inkling2_copy2d_raw((src, src_off, ss), (dp, dst_off, ds), width, rows, true)
    }

    fn inkling2_copy2d_raw(
        &self,
        (src, src_off, ss): (&CudaSlice<f32>, usize, usize),
        (dp, dst_off, ds): (u64, usize, usize),
        width: usize,
        rows: usize,
        bf16: bool,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .inkling2_copy2d
            .ok_or(GpuError::MissingOp("inkling2_copy2d"))?;
        let (sp, _g) = src.device_ptr(&self.stream);
        // SAFETY: ABI contract (slot 999); the callers checked both spans
        check(unsafe {
            f(
                sp as *const _,
                src_off as u64,
                ss as u32,
                dp as *mut _,
                dst_off as u64,
                ds as u32,
                width as u32,
                rows as u32,
                bf16 as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 1000: before a verify pass's short conv over `x` (element
    /// offset, row stride `xs`, `c` channels), each run's carried rows
    /// (`state` [slots][3][c]) and input rows into `stash` [slots][11][c].
    #[allow(clippy::too_many_arguments)]
    pub fn inkling2_conv_stash(
        &self,
        (x, x_off, xs): (&CudaSlice<f32>, usize, usize),
        c: usize,
        state: &CudaSlice<f32>,
        (run_off, run_slot, n_runs): (&CudaSlice<u32>, &CudaSlice<u32>, usize),
        stash: &mut CudaSlice<f32>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .inkling2_conv_stash
            .ok_or(GpuError::MissingOp("inkling2_conv_stash"))?;
        if run_off.len() < n_runs + 1
            || run_slot.len() < n_runs
            || stash.len() < state.len() / 3 * 11
        {
            return Err(GpuError::Unsupported(
                "inkling2_conv_stash: runs / stash".into(),
            ));
        }
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (sp, _g2) = state.device_ptr(&self.stream);
        let (op, _g3) = run_off.device_ptr(&self.stream);
        let (lp, _g4) = run_slot.device_ptr(&self.stream);
        let (tp, _g5) = stash.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 1000); the runs index the pass's rows of
        // x and the seated slots of state / stash
        check(unsafe {
            f(
                (xp + (x_off * 4) as u64) as *const _,
                xs as u32,
                c as u32,
                sp as *const _,
                op as *const _,
                lp as *const _,
                n_runs as u32,
                tp as *mut _,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 1001: each `(slot, keep)` of `items` ([n][2]) takes the carried
    /// rows after `keep` accepted rows of its stashed run.
    pub fn inkling2_conv_restore(
        &self,
        stash: &CudaSlice<f32>,
        state: &mut CudaSlice<f32>,
        c: usize,
        (items, n): (&CudaSlice<u32>, usize),
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .inkling2_conv_restore
            .ok_or(GpuError::MissingOp("inkling2_conv_restore"))?;
        if items.len() < 2 * n || stash.len() < state.len() / 3 * 11 {
            return Err(GpuError::Unsupported(
                "inkling2_conv_restore: items / stash".into(),
            ));
        }
        let (sp, _g1) = stash.device_ptr(&self.stream);
        let (tp, _g2) = state.device_ptr_mut(&self.stream);
        let (ip, _g3) = items.device_ptr(&self.stream);
        // SAFETY: ABI contract (slot 1001); items name seated slots, keep <=
        // the stashed run's rows
        check(unsafe {
            f(
                sp as *const _,
                tp as *mut _,
                c as u32,
                ip as *const _,
                n as u32,
                self.stream_ptr(),
            )
        })
    }
}

/// PADDOCK_INKLING2_DRAFT_NO_TC: the draft's block attention stays on the
/// CUDA cores (slot 996).
fn draft_tc_off() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_INKLING2_DRAFT_NO_TC").is_some())
}
