//! Inkling 2 Mini op wrappers - pack slots 977-993 and 1002-1009: the block
//! attention residuals, the loop controllers' mix, the attention's head
//! prep, the short convolutions, the softcapped SwiGLU, the experts'
//! activation quantize, decode GEMVs and prompt-width arms, the
//! relative-bias attention, and the one-pass branch tails.
//!
//! Offsets are in elements of the slice's type: the attention's projections
//! land as segments of one fused GEMM output, and these ops read and write
//! those segments in place.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use paddock_models::ggml_type::GgmlType;

use super::error::check;
use super::{GpuError, GpuExecutor, QuantTensor};

/// A layer's routed experts as shipped: gate|up rows interleaved
/// (`[E][2 ff][..]`), down `[E][hidden][..]`. `fmt` 0 = NVFP4 (codes two a
/// byte, e4m3 per 16, a per-expert factor), 1 = MXFP8 (e4m3, ue8m0 per 32).
pub struct Inkling2Experts {
    pub fmt: u32,
    pub w13: CudaSlice<u8>,
    pub s13: CudaSlice<u8>,
    pub f13: Option<CudaSlice<f32>>,
    pub w2: CudaSlice<u8>,
    pub s2: CudaSlice<u8>,
    pub f2: Option<CudaSlice<f32>>,
    pub n_expert: usize,
    pub hidden: usize,
    pub ff: usize,
}

/// A quantized activation plane at the experts' format (`rows` of `k`).
pub struct Inkling2Act {
    pub q: CudaSlice<u8>,
    pub s: CudaSlice<u8>,
    pub outer: CudaSlice<f32>,
}

impl Inkling2Act {
    /// Room for `rows` rows of `k` at either format.
    pub fn alloc(exec: &GpuExecutor, rows: usize, k: usize) -> Result<Self, GpuError> {
        Ok(Self {
            q: exec.alloc_u8(rows * k)?,
            s: exec.alloc_u8(rows * k / 16)?,
            outer: exec.alloc(rows)?,
        })
    }
}

/// `rows` of `width` elements, `stride` apart from `off`, inside a plane of `len`.
pub(super) fn span(
    len: usize,
    off: usize,
    rows: usize,
    stride: usize,
    width: usize,
    what: &str,
) -> Result<(), GpuError> {
    if rows > 0 && off + (rows - 1) * stride + width > len {
        return Err(GpuError::Unsupported(format!(
            "{what}: rows {rows} x stride {stride} (+{width}) at {off} past a plane of {len}"
        )));
    }
    Ok(())
}

impl GpuExecutor {
    /// Slots 977-986 are all in the pack.
    /// Whether the pack has slot 993 (prompt rows on the tensor cores) and
    /// it is not switched off (PADDOCK_INKLING2_NO_TILE).
    pub fn has_inkling2_tile(&self) -> bool {
        static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        self.kernels.inkling2_attn_tile.is_some()
            && !*OFF
                .get_or_init(|| paddock_models::dev_var_os!("PADDOCK_INKLING2_NO_TILE").is_some())
    }

    pub fn has_inkling2(&self) -> bool {
        let k = &self.kernels;
        k.inkling2_attn_res.is_some()
            && k.inkling2_msg_cat.is_some()
            && k.inkling2_msg_mix.is_some()
            && k.inkling2_head_prep.is_some()
            && k.inkling2_sconv.is_some()
            && k.inkling2_swiglu_cap.is_some()
            && k.inkling2_quant.is_some()
            && k.inkling2_moe_gu.is_some()
            && k.inkling2_moe_dn.is_some()
            && k.inkling2_attn.is_some()
            && k.inkling2_moe_gu_grp.is_some()
            && k.inkling2_moe_dn_grp.is_some()
            && k.moe_align.is_some()
            && k.moe_slot_combine.is_some()
            && k.inkling2_kv_store.is_some()
            && k.inkling2_attn_kv.is_some()
    }

    /// Slot 977: `out[q][r] = sum_i softmax_i(<Q_q, n_i> / sqrt(width)) n_i`
    /// over the block `states` and `partial` (at an element offset) at unit
    /// RMS; `query` holds `nq`
    /// rows of `width`, query `q`'s output rows at `out[q * qstride ..]`.
    #[allow(clippy::too_many_arguments)]
    pub fn inkling2_attn_res(
        &self,
        states: &[&CudaSlice<f32>],
        (partial, poff): (&CudaSlice<f32>, usize),
        query: &CudaSlice<f32>,
        nq: usize,
        out: &mut CudaSlice<f32>,
        rows: usize,
        width: usize,
        qstride: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .inkling2_attn_res
            .ok_or(GpuError::MissingOp("inkling2_attn_res"))?;
        if states.len() > 5 || nq == 0 || nq > 4 || query.len() < nq * width {
            return Err(GpuError::Unsupported("inkling2_attn_res: shape".into()));
        }
        for s in states {
            span(s.len(), 0, rows, width, width, "inkling2_attn_res state")?;
        }
        span(
            partial.len(),
            poff,
            rows,
            width,
            width,
            "inkling2_attn_res partial",
        )?;
        span(
            out.len(),
            (nq - 1) * qstride,
            rows,
            width,
            width,
            "inkling2_attn_res out",
        )?;
        let guards: Vec<_> = states.iter().map(|s| s.device_ptr(&self.stream)).collect();
        let ptrs: Vec<*const core::ffi::c_void> =
            guards.iter().map(|(p, _)| *p as *const _).collect();
        let (pp, _g1) = partial.device_ptr(&self.stream);
        let (qp, _g2) = query.device_ptr(&self.stream);
        let (op, _g3) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 977); the host array of state pointers is
        // read by the launcher, every plane checked to cover its rows above
        check(unsafe {
            f(
                ptrs.as_ptr(),
                states.len() as u32,
                (pp + (poff * 4) as u64) as *const _,
                qp as *const _,
                nq as u32,
                op as *mut _,
                rows as u32,
                width as u32,
                qstride as u32,
                inkling2_eps(),
                self.stream_ptr(),
            )
        })
    }

    /// Slot 978: `out[r] = [bf16(rms(h_r)), bf16(rms(m_r))]` (2 * width a
    /// row); h / m at element offsets.
    pub fn inkling2_msg_cat(
        &self,
        (h, hoff): (&CudaSlice<f32>, usize),
        (m, moff): (&CudaSlice<f32>, usize),
        out: &mut CudaSlice<f32>,
        rows: usize,
        width: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .inkling2_msg_cat
            .ok_or(GpuError::MissingOp("inkling2_msg_cat"))?;
        span(h.len(), hoff, rows, width, width, "inkling2_msg_cat h")?;
        span(m.len(), moff, rows, width, width, "inkling2_msg_cat m")?;
        span(
            out.len(),
            0,
            rows,
            2 * width,
            2 * width,
            "inkling2_msg_cat out",
        )?;
        let (hp, _g1) = h.device_ptr(&self.stream);
        let (mp, _g2) = m.device_ptr(&self.stream);
        let (op, _g3) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 978); planes checked above
        check(unsafe {
            f(
                (hp + (hoff * 4) as u64) as *const _,
                (mp + (moff * 4) as u64) as *const _,
                op as *mut _,
                rows as u32,
                width as u32,
                inkling2_eps(),
                self.stream_ptr(),
            )
        })
    }

    /// Slot 1008: slot 978's rows and the controller's down projection in
    /// one pass, z [rows][rank] = `down` (bf16, [rank][2 width]) over
    /// [bf16(h ia) | bf16(m ib)]; h / m at element offsets. Ok(false)
    /// without the slot.
    #[allow(clippy::too_many_arguments)]
    pub fn inkling2_msg_down(
        &self,
        (h, hoff): (&CudaSlice<f32>, usize),
        (m, moff): (&CudaSlice<f32>, usize),
        down: &QuantTensor,
        z: &mut CudaSlice<f32>,
        rows: usize,
        width: usize,
        rank: usize,
    ) -> Result<bool, GpuError> {
        let Some(f) = self.kernels.inkling2_msg_down else {
            return Ok(false);
        };
        span(h.len(), hoff, rows, width, width, "inkling2_msg_down h")?;
        span(m.len(), moff, rows, width, width, "inkling2_msg_down m")?;
        if down.ty != GgmlType::Bf16
            || down.dims.first() != Some(&(2 * width))
            || down.dims.get(1) != Some(&rank)
            || z.len() < rows * rank
        {
            return Err(GpuError::Unsupported(
                "inkling2_msg_down: the down plane / z".into(),
            ));
        }
        let (hp, _g1) = h.device_ptr(&self.stream);
        let (mp, _g2) = m.device_ptr(&self.stream);
        let (dp, _g3) = down.bytes.device_ptr(&self.stream);
        let (zp, _g4) = z.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 1008); planes checked above
        check(unsafe {
            f(
                (hp + (hoff * 4) as u64) as *const _,
                (mp + (moff * 4) as u64) as *const _,
                dp as *const _,
                zp as *mut _,
                rows as u32,
                width as u32,
                rank as u32,
                inkling2_eps(),
                self.stream_ptr(),
            )
        })
        .map(|()| true)
    }

    /// Slot 979: `out = (1 - tanh(F z)) h + tanh(G z) m`, z [rows][rank]
    /// (the down projection's output, taken through bf16), F / G bf16
    /// `[width][rank]` bytes; h / m at element offsets.
    #[allow(clippy::too_many_arguments)]
    pub fn inkling2_msg_mix(
        &self,
        z: &CudaSlice<f32>,
        fw: &CudaSlice<u8>,
        gw: &CudaSlice<u8>,
        (h, hoff): (&CudaSlice<f32>, usize),
        (m, moff): (&CudaSlice<f32>, usize),
        out: &mut CudaSlice<f32>,
        rows: usize,
        width: usize,
        rank: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .inkling2_msg_mix
            .ok_or(GpuError::MissingOp("inkling2_msg_mix"))?;
        if fw.len() < 2 * width * rank || gw.len() < 2 * width * rank {
            return Err(GpuError::Unsupported(
                "inkling2_msg_mix: gate planes".into(),
            ));
        }
        span(z.len(), 0, rows, rank, rank, "inkling2_msg_mix z")?;
        for (s, o, w) in [(h, hoff, "h"), (m, moff, "m")] {
            span(s.len(), o, rows, width, width, w)?;
        }
        span(out.len(), 0, rows, width, width, "inkling2_msg_mix out")?;
        let (zp, _g1) = z.device_ptr(&self.stream);
        let (fp, _g2) = fw.device_ptr(&self.stream);
        let (gp, _g3) = gw.device_ptr(&self.stream);
        let (hp, _g4) = h.device_ptr(&self.stream);
        let (mp, _g5) = m.device_ptr(&self.stream);
        let (op, _g6) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 979); planes checked above
        check(unsafe {
            f(
                zp as *const _,
                fp as *const _,
                gp as *const _,
                (hp + (hoff * 4) as u64) as *const _,
                (mp + (moff * 4) as u64) as *const _,
                op as *mut _,
                rows as u32,
                width as u32,
                rank as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 980: the per-head prep in place over `rows` of `heads x d`
    /// starting at element `off` of `x`, rows `stride` apart: optional bf16
    /// `scale` [heads][d], then an L2 clip to `clip` (0 = none) or unit RMS
    /// (`rms`), then with `pos` the log scale `1 + alpha ln(max((pos + 1) /
    /// floor, 1))`.
    #[allow(clippy::too_many_arguments)]
    pub fn inkling2_head_prep(
        &self,
        x: &mut CudaSlice<f32>,
        off: usize,
        rows: usize,
        heads: usize,
        d: usize,
        stride: usize,
        scale: Option<&CudaSlice<u8>>,
        clip: f32,
        rms: bool,
        pos: Option<&CudaSlice<u32>>,
        log: (f32, f32),
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .inkling2_head_prep
            .ok_or(GpuError::MissingOp("inkling2_head_prep"))?;
        span(x.len(), off, rows, stride, heads * d, "inkling2_head_prep")?;
        if scale.is_some_and(|s| s.len() < 2 * heads * d) || pos.is_some_and(|p| p.len() < rows) {
            return Err(GpuError::Unsupported(
                "inkling2_head_prep: scale / pos".into(),
            ));
        }
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let sg = scale.map(|s| s.device_ptr(&self.stream));
        let pg = pos.map(|p| p.device_ptr(&self.stream));
        let sp = sg.as_ref().map_or(0, |g| g.0);
        let pp = pg.as_ref().map_or(0, |g| g.0);
        // SAFETY: ABI contract (slot 980); the segment checked above
        check(unsafe {
            f(
                (xp + (off * 4) as u64) as *mut _,
                rows as u32,
                heads as u32,
                d as u32,
                stride as u32,
                sp as *const _,
                clip,
                rms as u32,
                inkling2_eps(),
                pp as *const _,
                log.0,
                log.1,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 981: the short causal conv over runs (`run_off` [n_runs + 1]
    /// row offsets, `run_slot` [n_runs] state blocks of 3 x c f32 in
    /// `state`). Static taps: `w` bf16 [c][4], `dyn` None. Dynamic: `w` bf16
    /// [4][c] by lag, `dyn` = (plane, element offset, row stride, heads) of
    /// the per-token head taps. `x` / `out` at element offsets with strides.
    #[allow(clippy::too_many_arguments)]
    pub fn inkling2_sconv(
        &self,
        x: (&CudaSlice<f32>, usize, usize),
        out: (&mut CudaSlice<f32>, usize, usize),
        c: usize,
        w: &CudaSlice<u8>,
        dyn_taps: Option<(&CudaSlice<f32>, usize, usize, usize)>,
        state: &mut CudaSlice<f32>,
        runs: (&CudaSlice<u32>, &CudaSlice<u32>, usize),
        rows: usize,
        max_run: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .inkling2_sconv
            .ok_or(GpuError::MissingOp("inkling2_sconv"))?;
        let (xs, xo, xst) = x;
        let (os, oo, ost) = out;
        span(xs.len(), xo, rows, xst, c, "inkling2_sconv x")?;
        span(os.len(), oo, rows, ost, c, "inkling2_sconv out")?;
        if w.len() < 8 * c {
            return Err(GpuError::Unsupported("inkling2_sconv: taps".into()));
        }
        if let Some((d, doff, dst, heads)) = dyn_taps {
            span(d.len(), doff, rows, dst, 4 * heads, "inkling2_sconv dyn")?;
        }
        let (rof, rsl, n_runs) = runs;
        if rof.len() < n_runs + 1 || rsl.len() < n_runs {
            return Err(GpuError::Unsupported("inkling2_sconv: runs".into()));
        }
        let (xp, _g1) = xs.device_ptr(&self.stream);
        let (op, _g2) = os.device_ptr_mut(&self.stream);
        let (wp, _g3) = w.device_ptr(&self.stream);
        let dg = dyn_taps.map(|(d, ..)| d.device_ptr(&self.stream));
        let (dp, dst, heads) = match (dyn_taps, dg.as_ref()) {
            (Some((_, doff, dst, heads)), Some(g)) => (g.0 + (doff * 4) as u64, dst, heads),
            _ => (0, 0, 0),
        };
        let (sp, _g4) = state.device_ptr_mut(&self.stream);
        let (rp, _g5) = rof.device_ptr(&self.stream);
        let (lp, _g6) = rsl.device_ptr(&self.stream);
        // SAFETY: ABI contract (slot 981); x / out / dyn checked above, the
        // run tables cover n_runs, every run's state block is the caller's
        check(unsafe {
            f(
                (xp + (xo * 4) as u64) as *const _,
                xst as u32,
                (op + (oo * 4) as u64) as *mut _,
                ost as u32,
                c as u32,
                wp as *const _,
                dp as *const _,
                dst as u32,
                heads as u32,
                sp as *mut _,
                rp as *const _,
                lp as *const _,
                n_runs as u32,
                max_run as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 1004 is here.
    pub fn has_inkling2_sconv_sum(&self) -> bool {
        self.kernels.inkling2_sconv_sum.is_some()
    }

    /// Slot 1004: a branch tail in one pass - slot 981's static-tap conv
    /// over `x` = the sum of `k` (1 or 8) consecutive partial rows of
    /// `input` a row (slot_combine's fold), onto `res`: `out = res +
    /// conv(x)` (`res` None: the conv alone). Planes `c` wide, contiguous;
    /// bit-identical to combine -> conv -> copy -> add.
    #[allow(clippy::too_many_arguments)]
    pub fn inkling2_sconv_sum(
        &self,
        (input, k): (&CudaSlice<f32>, usize),
        res: Option<&CudaSlice<f32>>,
        out: &mut CudaSlice<f32>,
        c: usize,
        w: &CudaSlice<u8>,
        state: &mut CudaSlice<f32>,
        runs: (&CudaSlice<u32>, &CudaSlice<u32>, usize),
        rows: usize,
        max_run: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .inkling2_sconv_sum
            .ok_or(GpuError::MissingOp("inkling2_sconv_sum"))?;
        let (rof, rsl, n_runs) = runs;
        if (k != 1 && k != 8)
            || input.len() < rows * k * c
            || out.len() < rows * c
            || res.is_some_and(|r| r.len() < rows * c)
            || w.len() < 8 * c
            || rof.len() < n_runs + 1
            || rsl.len() < n_runs
        {
            return Err(GpuError::Unsupported(format!(
                "inkling2_sconv_sum: {rows} rows x {k} partials of {c} / taps / runs"
            )));
        }
        let (ip, _g1) = input.device_ptr(&self.stream);
        let rg = res.map(|r| r.device_ptr(&self.stream));
        let rp = rg.as_ref().map_or(0, |g| g.0);
        let (op, _g3) = out.device_ptr_mut(&self.stream);
        let (wp, _g4) = w.device_ptr(&self.stream);
        let (sp, _g5) = state.device_ptr_mut(&self.stream);
        let (fp, _g6) = rof.device_ptr(&self.stream);
        let (lp, _g7) = rsl.device_ptr(&self.stream);
        // SAFETY: ABI contract (slot 1004); planes checked above, the run
        // tables cover n_runs, every run's state block is the caller's
        check(unsafe {
            f(
                ip as *const _,
                k as u32,
                rp as *const _,
                op as *mut _,
                c as u32,
                wp as *const _,
                sp as *mut _,
                fp as *const _,
                lp as *const _,
                n_runs as u32,
                max_run as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 982: `out = gc tanh(g / gc) sigmoid(g) uc tanh(u / uc)`.
    pub fn inkling2_swiglu_cap(
        &self,
        g: &CudaSlice<f32>,
        u: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n: usize,
        caps: (f32, f32),
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .inkling2_swiglu_cap
            .ok_or(GpuError::MissingOp("inkling2_swiglu_cap"))?;
        if g.len() < n || u.len() < n || out.len() < n {
            return Err(GpuError::Unsupported("inkling2_swiglu_cap: planes".into()));
        }
        let (gp, _g1) = g.device_ptr(&self.stream);
        let (up, _g2) = u.device_ptr(&self.stream);
        let (op, _g3) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 982); planes cover n
        check(unsafe {
            f(
                gp as *const _,
                up as *const _,
                op as *mut _,
                n as u32,
                caps.0,
                caps.1,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 983: quantize `rows` of `k` (row stride `xs`) at `fmt` into
    /// `act`, optionally times `gamma[row]`, through bf16 first with
    /// `bf16_in`.
    #[allow(clippy::too_many_arguments)]
    pub fn inkling2_quant(
        &self,
        x: &CudaSlice<f32>,
        xs: usize,
        gamma: Option<&CudaSlice<f32>>,
        bf16_in: bool,
        fmt: u32,
        act: &mut Inkling2Act,
        rows: usize,
        k: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .inkling2_quant
            .ok_or(GpuError::MissingOp("inkling2_quant"))?;
        span(x.len(), 0, rows, xs, k, "inkling2_quant x")?;
        let (qb, sb) = if fmt == 0 {
            (k / 2, k / 16)
        } else {
            (k, k / 32)
        };
        if act.q.len() < rows * qb || act.s.len() < rows * sb || act.outer.len() < rows {
            return Err(GpuError::Unsupported(
                "inkling2_quant: activation planes".into(),
            ));
        }
        if gamma.is_some_and(|g| g.len() < rows) {
            return Err(GpuError::Unsupported("inkling2_quant: gamma".into()));
        }
        let (xp, _g1) = x.device_ptr(&self.stream);
        let gg = gamma.map(|g| g.device_ptr(&self.stream));
        let gp = gg.as_ref().map_or(0, |g| g.0);
        let (qp, _g2) = act.q.device_ptr_mut(&self.stream);
        let (sp, _g3) = act.s.device_ptr_mut(&self.stream);
        let (op, _g4) = act.outer.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 983); planes checked above
        check(unsafe {
            f(
                xp as *const _,
                xs as u32,
                gp as *const _,
                bf16_in as u32,
                fmt,
                qp as *mut _,
                sp as *mut _,
                op as *mut _,
                rows as u32,
                k as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 983 over bf16 rows (`bf16_in` 2): `x16` holds `rows` x `k` bf16
    /// values in an f32 plane's memory (slot 1006's act), quantized as they
    /// are.
    pub fn inkling2_quant16(
        &self,
        x16: &CudaSlice<f32>,
        fmt: u32,
        act: &mut Inkling2Act,
        rows: usize,
        k: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .inkling2_quant
            .ok_or(GpuError::MissingOp("inkling2_quant"))?;
        let (qb, sb) = if fmt == 0 {
            (k / 2, k / 16)
        } else {
            (k, k / 32)
        };
        if 2 * x16.len() < rows * k
            || act.q.len() < rows * qb
            || act.s.len() < rows * sb
            || act.outer.len() < rows
        {
            return Err(GpuError::Unsupported("inkling2_quant16: planes".into()));
        }
        let (xp, _g1) = x16.device_ptr(&self.stream);
        let (qp, _g2) = act.q.device_ptr_mut(&self.stream);
        let (sp, _g3) = act.s.device_ptr_mut(&self.stream);
        let (op, _g4) = act.outer.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 983, bf16_in 2 - the caller checked slot
        // 1006, whose packs read it); planes checked above
        check(unsafe {
            f(
                xp as *const _,
                k as u32,
                std::ptr::null(),
                2,
                fmt,
                qp as *mut _,
                sp as *mut _,
                op as *mut _,
                rows as u32,
                k as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 984: gate|up of every (row, slot) pair (`idx` [rows * topk]),
    /// softcapped SwiGLU in the epilogue into `act` [rows * topk][ff].
    #[allow(clippy::too_many_arguments)]
    pub fn inkling2_moe_gu(
        &self,
        e: &Inkling2Experts,
        x: &Inkling2Act,
        idx: &CudaSlice<u32>,
        topk: usize,
        act: &mut CudaSlice<f32>,
        rows: usize,
        caps: (f32, f32),
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .inkling2_moe_gu
            .ok_or(GpuError::MissingOp("inkling2_moe_gu"))?;
        if idx.len() < rows * topk || act.len() < rows * topk * e.ff {
            return Err(GpuError::Unsupported("inkling2_moe_gu: planes".into()));
        }
        let (wp, _g1) = e.w13.device_ptr(&self.stream);
        let (sp, _g2) = e.s13.device_ptr(&self.stream);
        let fg = e.f13.as_ref().map(|s| s.device_ptr(&self.stream));
        let fp = fg.as_ref().map_or(0, |g| g.0);
        let (xq, _g3) = x.q.device_ptr(&self.stream);
        let (xs, _g4) = x.s.device_ptr(&self.stream);
        let (xo, _g5) = x.outer.device_ptr(&self.stream);
        let (ip, _g6) = idx.device_ptr(&self.stream);
        let (ap, _g7) = act.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 984); the expert planes are the layer's
        // own (checked at load), idx / act cover the pairs
        check(unsafe {
            f(
                wp as *const _,
                sp as *const _,
                fp as *const _,
                xq as *const _,
                xs as *const _,
                xo as *const _,
                ip as *const _,
                topk as u32,
                ap as *mut _,
                rows as u32,
                e.hidden as u32,
                e.ff as u32,
                e.fmt,
                caps.0,
                caps.1,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 985: down of every row's topk pairs, summed in slot order into
    /// `out` [rows][hidden]; `a` the pairs' quantized activations.
    pub fn inkling2_moe_dn(
        &self,
        e: &Inkling2Experts,
        a: &Inkling2Act,
        idx: &CudaSlice<u32>,
        topk: usize,
        out: &mut CudaSlice<f32>,
        rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .inkling2_moe_dn
            .ok_or(GpuError::MissingOp("inkling2_moe_dn"))?;
        if idx.len() < rows * topk || out.len() < rows * e.hidden {
            return Err(GpuError::Unsupported("inkling2_moe_dn: planes".into()));
        }
        let (wp, _g1) = e.w2.device_ptr(&self.stream);
        let (sp, _g2) = e.s2.device_ptr(&self.stream);
        let fg = e.f2.as_ref().map(|s| s.device_ptr(&self.stream));
        let fp = fg.as_ref().map_or(0, |g| g.0);
        let (aq, _g3) = a.q.device_ptr(&self.stream);
        let (as_, _g4) = a.s.device_ptr(&self.stream);
        let (ao, _g5) = a.outer.device_ptr(&self.stream);
        let (ip, _g6) = idx.device_ptr(&self.stream);
        let (op, _g7) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 985); as for slot 984
        check(unsafe {
            f(
                wp as *const _,
                sp as *const _,
                fp as *const _,
                aq as *const _,
                as_ as *const _,
                ao as *const _,
                ip as *const _,
                topk as u32,
                op as *mut _,
                rows as u32,
                e.ff as u32,
                e.hidden as u32,
                e.fmt,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 986: the relative-bias attention of `rows` consecutive positions
    /// from `p0` over the contiguous history `k` / `v` ([pos][kv_heads *
    /// 128]); q / rel / sink are (plane, element offset, row stride)
    /// segments, `proj` the layer's bf16 [16][extent] bank; window 0 = full.
    #[allow(clippy::too_many_arguments)]
    pub fn inkling2_attn(
        &self,
        q: (&CudaSlice<f32>, usize, usize),
        rel: (&CudaSlice<f32>, usize, usize),
        sink: (&CudaSlice<f32>, usize, usize),
        (k, v): (&CudaSlice<f32>, &CudaSlice<f32>),
        proj: &CudaSlice<u8>,
        out: &mut CudaSlice<f32>,
        rows: usize,
        p0: usize,
        (heads, kv_heads): (usize, usize),
        (window, extent): (usize, usize),
        scale: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .inkling2_attn
            .ok_or(GpuError::MissingOp("inkling2_attn"))?;
        const HD: usize = 128;
        const DR: usize = 16;
        span(q.0.len(), q.1, rows, q.2, heads * HD, "inkling2_attn q")?;
        span(
            rel.0.len(),
            rel.1,
            rows,
            rel.2,
            heads * DR,
            "inkling2_attn rel",
        )?;
        span(
            sink.0.len(),
            sink.1,
            rows,
            sink.2,
            heads,
            "inkling2_attn sink",
        )?;
        let keys = (p0 + rows) * kv_heads * HD;
        if k.len() < keys
            || v.len() < keys
            || out.len() < rows * heads * HD
            || proj.len() < 2 * DR * extent
        {
            return Err(GpuError::Unsupported(
                "inkling2_attn: history / out / bank".into(),
            ));
        }
        let (qp, _g1) = q.0.device_ptr(&self.stream);
        let (rp, _g2) = rel.0.device_ptr(&self.stream);
        let (sp, _g3) = sink.0.device_ptr(&self.stream);
        let (kp, _g4) = k.device_ptr(&self.stream);
        let (vp, _g5) = v.device_ptr(&self.stream);
        let (pp, _g6) = proj.device_ptr(&self.stream);
        let (op, _g7) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 986); every plane checked above
        check(unsafe {
            f(
                (qp + (q.1 * 4) as u64) as *const _,
                q.2 as u32,
                (rp + (rel.1 * 4) as u64) as *const _,
                rel.2 as u32,
                (sp + (sink.1 * 4) as u64) as *const _,
                sink.2 as u32,
                kp as *const _,
                vp as *const _,
                pp as *const _,
                op as *mut _,
                rows as u32,
                p0 as u32,
                heads as u32,
                kv_heads as u32,
                HD as u32,
                DR as u32,
                window as u32,
                extent as u32,
                scale,
                self.stream_ptr(),
            )
        })
    }
}

/// A routed wave sorted into moe_align's blocks of `bm` pairs (one expert a
/// block): 32 for slots 987-988 / 991-992, 64 for slots 1002-1003.
/// `blocks` is the current wave's block count: moe_align pads only that
/// many, so the arms launch over it, never over the allocation (blocks past
/// it hold an earlier wave's rows).
pub struct Inkling2Sorted {
    pub row: CudaSlice<u32>,
    pub slot: CudaSlice<u32>,
    pub expert: CudaSlice<u32>,
    pub max_blocks: usize,
    pub blocks: usize,
    pub bm: usize,
}

impl Inkling2Sorted {
    /// Room for `rows` tokens of `topk` pairs over `n_expert` experts, at
    /// either block size (32-pair blocks are the most blocks, 64-pair ones
    /// the most padded rows).
    pub fn alloc(
        exec: &GpuExecutor,
        rows: usize,
        topk: usize,
        n_expert: usize,
    ) -> Result<Self, GpuError> {
        let max_blocks = (rows * topk).div_ceil(32) + n_expert;
        let max_rows = ((rows * topk).div_ceil(64) + n_expert) * 64;
        Ok(Self {
            row: exec.alloc_u32(max_rows)?,
            slot: exec.alloc_u32(max_rows)?,
            expert: exec.alloc_u32(max_blocks)?,
            max_blocks,
            blocks: 0,
            bm: 32,
        })
    }
}

impl GpuExecutor {
    /// Sort `rows` tokens' `topk` picks (`idx`) into `s`'s blocks of `bm`
    /// (32 or 64) pairs.
    pub fn inkling2_sort(
        &self,
        idx: &CudaSlice<u32>,
        s: &mut Inkling2Sorted,
        rows: usize,
        topk: usize,
        n_expert: usize,
        bm: usize,
    ) -> Result<(), GpuError> {
        let max_blocks = (rows * topk).div_ceil(bm) + n_expert;
        if (bm != 32 && bm != 64) || max_blocks > s.max_blocks || max_blocks * bm > s.row.len() {
            return Err(GpuError::Unsupported(format!(
                "inkling2_sort: a wave of {rows} x {topk} in {bm}-pair blocks past its room"
            )));
        }
        s.blocks = max_blocks;
        s.bm = bm;
        if bm == 64 {
            return self.moe_align_bm(
                idx,
                &mut s.row,
                &mut s.slot,
                &mut s.expert,
                rows,
                topk,
                n_expert,
                bm,
                max_blocks,
            );
        }
        self.moe_align(
            idx,
            &mut s.row,
            &mut s.slot,
            &mut s.expert,
            rows,
            topk,
            n_expert,
            max_blocks,
        )
    }

    /// Slot 1006 (slot 1002's bf16 act) is here too.
    pub fn has_inkling2_ms16(&self) -> bool {
        self.kernels.inkling2_moe_gu_ms16.is_some() && self.has_inkling2_ms()
    }

    /// Slots 1002-1003 serve this die: the block-scale tensor cores of the
    /// sm_120 family (what the pack's own launchers check) and the slots.
    pub fn has_inkling2_ms(&self) -> bool {
        static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        self.kernels.inkling2_moe_gu_ms.is_some()
            && self.kernels.inkling2_moe_dn_ms.is_some()
            && self.compute_capability().0 == 12
            && !inkling2_tc_off()
            && !*OFF.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_INKLING2_NO_MS").is_some())
    }

    /// Gate|up of a sorted wave into `act` [rows * topk][ff]: on the
    /// block-scale tensor cores (slot 991) where the die has them, else
    /// slot 987.
    pub fn inkling2_moe_gu_grp(
        &self,
        e: &Inkling2Experts,
        x: &Inkling2Act,
        s: &Inkling2Sorted,
        topk: usize,
        act: &mut CudaSlice<f32>,
        caps: (f32, f32),
    ) -> Result<(), GpuError> {
        if s.bm == 64 {
            return self.inkling2_moe_ms(true, e, x, s, topk, act, caps, None);
        }
        if !inkling2_tc_off() && self.inkling2_moe_gu_tc(e, x, s, topk, act, caps)? {
            return Ok(());
        }
        self.inkling2_moe_gu_cc(e, x, s, topk, act, caps)
    }

    /// Slot 987: gate|up of a sorted wave on the CUDA cores (both operands
    /// decoded to f32), every class.
    pub fn inkling2_moe_gu_cc(
        &self,
        e: &Inkling2Experts,
        x: &Inkling2Act,
        s: &Inkling2Sorted,
        topk: usize,
        act: &mut CudaSlice<f32>,
        caps: (f32, f32),
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .inkling2_moe_gu_grp
            .ok_or(GpuError::MissingOp("inkling2_moe_gu_grp"))?;
        let (wp, _g1) = e.w13.device_ptr(&self.stream);
        let (sp, _g2) = e.s13.device_ptr(&self.stream);
        let fg = e.f13.as_ref().map(|s| s.device_ptr(&self.stream));
        let fp = fg.as_ref().map_or(0, |g| g.0);
        let (xq, _g3) = x.q.device_ptr(&self.stream);
        let (xs, _g4) = x.s.device_ptr(&self.stream);
        let (xo, _g5) = x.outer.device_ptr(&self.stream);
        let (rp, _g6) = s.row.device_ptr(&self.stream);
        let (lp, _g7) = s.slot.device_ptr(&self.stream);
        let (bp, _g8) = s.expert.device_ptr(&self.stream);
        let (ap, _g9) = act.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 987); the sorted wave indexes pairs the
        // act plane covers (the caller sized both for the wave)
        check(unsafe {
            f(
                wp as *const _,
                sp as *const _,
                fp as *const _,
                xq as *const _,
                xs as *const _,
                xo as *const _,
                rp as *const _,
                lp as *const _,
                bp as *const _,
                s.blocks as u32,
                topk as u32,
                ap as *mut _,
                e.hidden as u32,
                e.ff as u32,
                e.fmt,
                caps.0,
                caps.1,
                self.stream_ptr(),
            )
        })
    }

    /// Down of a sorted wave into the per-pair partials `part` [rows *
    /// topk][hidden]: slot 992 where the die has it, else slot 988.
    pub fn inkling2_moe_dn_grp(
        &self,
        e: &Inkling2Experts,
        a: &Inkling2Act,
        s: &Inkling2Sorted,
        topk: usize,
        part: &mut CudaSlice<f32>,
    ) -> Result<(), GpuError> {
        if s.bm == 64 {
            return self.inkling2_moe_ms(false, e, a, s, topk, part, (0.0, 0.0), None);
        }
        if !inkling2_tc_off() && self.inkling2_moe_dn_tc(e, a, s, topk, part)? {
            return Ok(());
        }
        self.inkling2_moe_dn_cc(e, a, s, topk, part)
    }

    /// Slot 988: down of a sorted wave on the CUDA cores, every class.
    pub fn inkling2_moe_dn_cc(
        &self,
        e: &Inkling2Experts,
        a: &Inkling2Act,
        s: &Inkling2Sorted,
        topk: usize,
        part: &mut CudaSlice<f32>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .inkling2_moe_dn_grp
            .ok_or(GpuError::MissingOp("inkling2_moe_dn_grp"))?;
        let (wp, _g1) = e.w2.device_ptr(&self.stream);
        let (sp, _g2) = e.s2.device_ptr(&self.stream);
        let fg = e.f2.as_ref().map(|s| s.device_ptr(&self.stream));
        let fp = fg.as_ref().map_or(0, |g| g.0);
        let (aq, _g3) = a.q.device_ptr(&self.stream);
        let (as_, _g4) = a.s.device_ptr(&self.stream);
        let (ao, _g5) = a.outer.device_ptr(&self.stream);
        let (rp, _g6) = s.row.device_ptr(&self.stream);
        let (lp, _g7) = s.slot.device_ptr(&self.stream);
        let (bp, _g8) = s.expert.device_ptr(&self.stream);
        let (pp, _g9) = part.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 988); as slot 987
        check(unsafe {
            f(
                wp as *const _,
                sp as *const _,
                fp as *const _,
                aq as *const _,
                as_ as *const _,
                ao as *const _,
                rp as *const _,
                lp as *const _,
                bp as *const _,
                s.blocks as u32,
                topk as u32,
                pp as *mut _,
                e.ff as u32,
                e.hidden as u32,
                e.fmt,
                self.stream_ptr(),
            )
        })
    }
}

impl GpuExecutor {
    /// Slot 991: [`Self::inkling2_moe_gu_cc`] on the block-scale tensor
    /// cores, either class. Ok(false): not here (no slot, or a die without
    /// block-scale MMA declined it).
    pub fn inkling2_moe_gu_tc(
        &self,
        e: &Inkling2Experts,
        x: &Inkling2Act,
        s: &Inkling2Sorted,
        topk: usize,
        act: &mut CudaSlice<f32>,
        caps: (f32, f32),
    ) -> Result<bool, GpuError> {
        let Some(f) = self.kernels.inkling2_moe_gu_tc else {
            return Ok(false);
        };
        let (wp, _g1) = e.w13.device_ptr(&self.stream);
        let (sp, _g2) = e.s13.device_ptr(&self.stream);
        let fg = e.f13.as_ref().map(|s| s.device_ptr(&self.stream));
        let fp = fg.as_ref().map_or(0, |g| g.0);
        let (xq, _g4) = x.q.device_ptr(&self.stream);
        let (xs, _g5) = x.s.device_ptr(&self.stream);
        let (xo, _g6) = x.outer.device_ptr(&self.stream);
        let (rp, _g7) = s.row.device_ptr(&self.stream);
        let (lp, _g8) = s.slot.device_ptr(&self.stream);
        let (bp, _g9) = s.expert.device_ptr(&self.stream);
        let (ap, _g10) = act.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 991); as slot 987
        let rc = unsafe {
            f(
                wp as *const _,
                sp as *const _,
                fp as *const _,
                xq as *const _,
                xs as *const _,
                xo as *const _,
                rp as *const _,
                lp as *const _,
                bp as *const _,
                s.blocks as u32,
                topk as u32,
                ap as *mut _,
                e.hidden as u32,
                e.ff as u32,
                e.fmt,
                caps.0,
                caps.1,
                self.stream_ptr(),
            )
        };
        if rc == CUDA_ERROR_NOT_SUPPORTED {
            return Ok(false);
        }
        check(rc).map(|()| true)
    }

    /// Slot 992: [`Self::inkling2_moe_dn_cc`] on the block-scale tensor
    /// cores; Ok(false) as slot 991's.
    pub fn inkling2_moe_dn_tc(
        &self,
        e: &Inkling2Experts,
        a: &Inkling2Act,
        s: &Inkling2Sorted,
        topk: usize,
        part: &mut CudaSlice<f32>,
    ) -> Result<bool, GpuError> {
        let Some(f) = self.kernels.inkling2_moe_dn_tc else {
            return Ok(false);
        };
        let (wp, _g1) = e.w2.device_ptr(&self.stream);
        let (sp, _g2) = e.s2.device_ptr(&self.stream);
        let fg = e.f2.as_ref().map(|s| s.device_ptr(&self.stream));
        let fp = fg.as_ref().map_or(0, |g| g.0);
        let (aq, _g4) = a.q.device_ptr(&self.stream);
        let (as_, _g5) = a.s.device_ptr(&self.stream);
        let (ao, _g6) = a.outer.device_ptr(&self.stream);
        let (rp, _g7) = s.row.device_ptr(&self.stream);
        let (lp, _g8) = s.slot.device_ptr(&self.stream);
        let (bp, _g9) = s.expert.device_ptr(&self.stream);
        let (pp, _g10) = part.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 992); as slot 988
        let rc = unsafe {
            f(
                wp as *const _,
                sp as *const _,
                fp as *const _,
                aq as *const _,
                as_ as *const _,
                ao as *const _,
                rp as *const _,
                lp as *const _,
                bp as *const _,
                s.blocks as u32,
                topk as u32,
                pp as *mut _,
                e.ff as u32,
                e.hidden as u32,
                e.fmt,
                self.stream_ptr(),
            )
        };
        if rc == CUDA_ERROR_NOT_SUPPORTED {
            return Ok(false);
        }
        check(rc).map(|()| true)
    }
}

impl GpuExecutor {
    /// Slot 1006: [`Self::inkling2_moe_gu_grp`] over 64-pair blocks with
    /// `act` holding bf16(act x gamma[pair]) rows (gamma the router weights)
    /// in its f32 plane's memory - read it with
    /// [`Self::inkling2_quant16`], which gives slot 983's codes for the f32
    /// act and gamma.
    #[allow(clippy::too_many_arguments)]
    pub fn inkling2_moe_gu_ms16(
        &self,
        e: &Inkling2Experts,
        x: &Inkling2Act,
        s: &Inkling2Sorted,
        topk: usize,
        act: &mut CudaSlice<f32>,
        gamma: &CudaSlice<f32>,
        caps: (f32, f32),
    ) -> Result<(), GpuError> {
        self.inkling2_moe_ms(true, e, x, s, topk, act, caps, Some(gamma))
    }

    /// Slots 1002 (`gu`) / 1003: a 64-pair sorted wave of the NVFP4 class,
    /// gate|up into `out` = act [pairs][ff] or down into `out` = part
    /// [pairs][hidden]; with `gamma`, slot 1006 (gate|up only). Only
    /// [`Self::has_inkling2_ms`] dies sort at 64, so a decline here is an
    /// error, not a fallback (the other arms read 32-pair blocks).
    #[allow(clippy::too_many_arguments)]
    fn inkling2_moe_ms(
        &self,
        gu: bool,
        e: &Inkling2Experts,
        x: &Inkling2Act,
        s: &Inkling2Sorted,
        topk: usize,
        out: &mut CudaSlice<f32>,
        caps: (f32, f32),
        gamma: Option<&CudaSlice<f32>>,
    ) -> Result<(), GpuError> {
        if e.fmt != 0 {
            return Err(GpuError::Unsupported(
                "inkling2: 64-pair blocks are the NVFP4 class's".into(),
            ));
        }
        let (w, sc, fac) = if gu {
            (&e.w13, &e.s13, &e.f13)
        } else {
            (&e.w2, &e.s2, &e.f2)
        };
        let (wp, _g1) = w.device_ptr(&self.stream);
        let (sp, _g2) = sc.device_ptr(&self.stream);
        let fg = fac.as_ref().map(|s| s.device_ptr(&self.stream));
        let fp = fg.as_ref().map_or(0, |g| g.0);
        let (xq, _g4) = x.q.device_ptr(&self.stream);
        let (xs, _g5) = x.s.device_ptr(&self.stream);
        let (xo, _g6) = x.outer.device_ptr(&self.stream);
        let (rp, _g7) = s.row.device_ptr(&self.stream);
        let (lp, _g8) = s.slot.device_ptr(&self.stream);
        let (bp, _g9) = s.expert.device_ptr(&self.stream);
        let (op, _g10) = out.device_ptr_mut(&self.stream);
        let rc = if let Some(g) = gamma {
            let f = self
                .kernels
                .inkling2_moe_gu_ms16
                .ok_or(GpuError::MissingOp("inkling2_moe_gu_ms16"))?;
            let (gp, _g11) = g.device_ptr(&self.stream);
            // SAFETY: ABI contract (slot 1006); as slot 1002, gamma covers the
            // wave's pairs
            unsafe {
                f(
                    wp as *const _,
                    sp as *const _,
                    fp as *const _,
                    xq as *const _,
                    xs as *const _,
                    xo as *const _,
                    rp as *const _,
                    lp as *const _,
                    bp as *const _,
                    s.blocks as u32,
                    topk as u32,
                    op as *mut _,
                    gp as *const _,
                    e.hidden as u32,
                    e.ff as u32,
                    caps.0,
                    caps.1,
                    self.stream_ptr(),
                )
            }
        } else if gu {
            let f = self
                .kernels
                .inkling2_moe_gu_ms
                .ok_or(GpuError::MissingOp("inkling2_moe_gu_ms"))?;
            // SAFETY: ABI contract (slot 1002); as slot 991 over 64-pair blocks
            unsafe {
                f(
                    wp as *const _,
                    sp as *const _,
                    fp as *const _,
                    xq as *const _,
                    xs as *const _,
                    xo as *const _,
                    rp as *const _,
                    lp as *const _,
                    bp as *const _,
                    s.blocks as u32,
                    topk as u32,
                    op as *mut _,
                    e.hidden as u32,
                    e.ff as u32,
                    caps.0,
                    caps.1,
                    self.stream_ptr(),
                )
            }
        } else {
            let f = self
                .kernels
                .inkling2_moe_dn_ms
                .ok_or(GpuError::MissingOp("inkling2_moe_dn_ms"))?;
            // SAFETY: ABI contract (slot 1003); as slot 992 over 64-pair blocks
            unsafe {
                f(
                    wp as *const _,
                    sp as *const _,
                    fp as *const _,
                    xq as *const _,
                    xs as *const _,
                    xo as *const _,
                    rp as *const _,
                    lp as *const _,
                    bp as *const _,
                    s.blocks as u32,
                    topk as u32,
                    op as *mut _,
                    e.ff as u32,
                    e.hidden as u32,
                    self.stream_ptr(),
                )
            }
        };
        check(rc)
    }
}

/// A launcher's "not on this die / not this shape" decline.
const CUDA_ERROR_NOT_SUPPORTED: i32 = 801;

/// PADDOCK_INKLING2_NO_TC: the prompt-width experts stay on the CUDA cores
/// (slots 987-988).
fn inkling2_tc_off() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_INKLING2_NO_TC").is_some())
}

/// Where a visit's K / V rows live (slots 989-990): the full-attention
/// visits page them through the slots' block tables (16-row blocks, shared
/// by every full visit), the sliding visits keep `ring` rows a slot.
#[derive(Clone, Copy)]
pub enum Inkling2KvLayout<'a> {
    Paged(&'a CudaSlice<u32>, usize),
    Ring(usize),
}

/// A K / V row class: 0 f16, 1 fp8 e4m3, 2 f32 (the reference class).
pub const INKLING2_KV_F32: u32 = 2;

impl GpuExecutor {
    /// Slot 989: `rows` prepped rows of `src` (row stride `stride`, element
    /// offset `off`) into a visit's store at (slots[r], pos[r]).
    #[allow(clippy::too_many_arguments)]
    pub fn inkling2_kv_store(
        &self,
        (src, off, stride): (&CudaSlice<f32>, usize, usize),
        (slots, pos): (&CudaSlice<u32>, &CudaSlice<u32>),
        dst: &mut CudaSlice<u8>,
        layout: Inkling2KvLayout<'_>,
        dtype: u32,
        kv_dim: usize,
        rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .inkling2_kv_store
            .ok_or(GpuError::MissingOp("inkling2_kv_store"))?;
        span(
            src.len(),
            off,
            rows,
            stride,
            kv_dim,
            "inkling2_kv_store src",
        )?;
        if slots.len() < rows || pos.len() < rows {
            return Err(GpuError::Unsupported(
                "inkling2_kv_store: row tables".into(),
            ));
        }
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (lp, _g2) = slots.device_ptr(&self.stream);
        let (pp, _g3) = pos.device_ptr(&self.stream);
        let (dp, _g4) = dst.device_ptr_mut(&self.stream);
        let (bt, bps, ring) = match layout {
            Inkling2KvLayout::Paged(b, bps) => (Some(b.device_ptr(&self.stream)), bps, 0),
            Inkling2KvLayout::Ring(r) => (None, 0, r),
        };
        let btp = bt.as_ref().map_or(0, |g| g.0);
        // SAFETY: ABI contract (slot 989); the caller backed every (slot,
        // pos) - the block tables map them, a ring holds them by construction
        check(unsafe {
            f(
                (sp + (off * 4) as u64) as *const _,
                stride as u32,
                lp as *const _,
                pp as *const _,
                dp as *mut _,
                btp as *const _,
                bps as u32,
                ring as u32,
                dtype,
                kv_dim as u32,
                rows as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 990: the relative-bias attention of `rows` rows at (slots[r],
    /// pos[r]) over a visit's K / V stores, keys split `splits` ways
    /// (`part` = the per-split scratch, [rows][heads][splits] x (128 + 2)
    /// f32, needed when splits > 1); q / rel / sink as slot 986's segments.
    #[allow(clippy::too_many_arguments)]
    pub fn inkling2_attn_kv(
        &self,
        q: (&CudaSlice<f32>, usize, usize),
        rel: (&CudaSlice<f32>, usize, usize),
        sink: (&CudaSlice<f32>, usize, usize),
        (k, v): (&CudaSlice<u8>, &CudaSlice<u8>),
        layout: Inkling2KvLayout<'_>,
        dtype: u32,
        (slots, pos): (&CudaSlice<u32>, &CudaSlice<u32>),
        proj: &CudaSlice<u8>,
        out: &mut CudaSlice<f32>,
        part: Option<(&mut CudaSlice<f32>, &mut CudaSlice<f32>)>,
        rows: usize,
        (heads, kv_heads): (usize, usize),
        (window, extent): (usize, usize),
        splits: usize,
        scale: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .inkling2_attn_kv
            .ok_or(GpuError::MissingOp("inkling2_attn_kv"))?;
        const HD: usize = 128;
        const DR: usize = 16;
        span(q.0.len(), q.1, rows, q.2, heads * HD, "inkling2_attn_kv q")?;
        span(
            rel.0.len(),
            rel.1,
            rows,
            rel.2,
            heads * DR,
            "inkling2_attn_kv rel",
        )?;
        span(
            sink.0.len(),
            sink.1,
            rows,
            sink.2,
            heads,
            "inkling2_attn_kv sink",
        )?;
        if out.len() < rows * heads * HD
            || slots.len() < rows
            || pos.len() < rows
            || proj.len() < 2 * DR * extent
        {
            return Err(GpuError::Unsupported(
                "inkling2_attn_kv: out / rows / bank".into(),
            ));
        }
        if splits > 1
            && part.as_ref().is_none_or(|(o, ml)| {
                o.len() < rows * heads * splits * HD || ml.len() < rows * heads * splits * 2
            })
        {
            return Err(GpuError::Unsupported(
                "inkling2_attn_kv: split scratch".into(),
            ));
        }
        let (qp, _g1) = q.0.device_ptr(&self.stream);
        let (rp, _g2) = rel.0.device_ptr(&self.stream);
        let (sp, _g3) = sink.0.device_ptr(&self.stream);
        let (kp, _g4) = k.device_ptr(&self.stream);
        let (vp, _g5) = v.device_ptr(&self.stream);
        let (lp, _g6) = slots.device_ptr(&self.stream);
        let (pp, _g7) = pos.device_ptr(&self.stream);
        let (jp, _g8) = proj.device_ptr(&self.stream);
        let (op, _g9) = out.device_ptr_mut(&self.stream);
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
        // SAFETY: ABI contract (slot 990); every plane checked above, the
        // stores hold every row's keys (the caller stored them first)
        check(unsafe {
            f(
                (qp + (q.1 * 4) as u64) as *const _,
                q.2 as u32,
                (rp + (rel.1 * 4) as u64) as *const _,
                rel.2 as u32,
                (sp + (sink.1 * 4) as u64) as *const _,
                sink.2 as u32,
                kp as *mut _,
                vp as *mut _,
                btp as *const _,
                bps as u32,
                ring as u32,
                dtype,
                lp as *const _,
                pp as *const _,
                jp as *const _,
                op as *mut _,
                pop as *mut _,
                pmp as *mut _,
                rows as u32,
                heads as u32,
                kv_heads as u32,
                window as u32,
                extent as u32,
                splits as u32,
                scale,
                self.stream_ptr(),
            )
        })
    }
}

impl GpuExecutor {
    /// Slot 1009 is here: [`Self::inkling2_attn_tile_split`].
    pub fn has_inkling2_tile_split(&self) -> bool {
        self.kernels.inkling2_attn_tile_split.is_some() && self.has_inkling2_tile()
    }

    /// Slot 1009: [`Self::inkling2_attn_tile`]'s tiles split over keys
    /// (`splits` 2..=64), into slot 990's partials (`part`) and folded with
    /// the sinks over rows [0, rows) - the tiles cover them all, rows <= 64.
    /// Ungated f32 rows into `out`.
    #[allow(clippy::too_many_arguments)]
    pub fn inkling2_attn_tile_split(
        &self,
        q: (&CudaSlice<f32>, usize, usize),
        rel: (&CudaSlice<f32>, usize, usize),
        sink: (&CudaSlice<f32>, usize, usize),
        (k, v): (&CudaSlice<u8>, &CudaSlice<u8>),
        layout: Inkling2KvLayout<'_>,
        dtype: u32,
        (tiles, n_tiles): (&CudaSlice<u32>, usize),
        proj_t: &CudaSlice<u8>,
        out: &mut CudaSlice<f32>,
        (part_o, part_ml): (&mut CudaSlice<f32>, &mut CudaSlice<f32>),
        rows: usize,
        (heads, kv_heads): (usize, usize),
        (window, extent): (usize, usize),
        scale: f32,
        splits: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .inkling2_attn_tile_split
            .ok_or(GpuError::MissingOp("inkling2_attn_tile_split"))?;
        if tiles.len() < n_tiles * 4
            || proj_t.len() < 2 * 16 * extent
            || dtype > 1
            || !(2..=64).contains(&splits)
            || rows > 64
            || out.len() < rows * heads * 128
            || part_o.len() < rows * heads * splits * 128
            || part_ml.len() < rows * heads * splits * 2
        {
            return Err(GpuError::Unsupported(
                "inkling2_attn_tile_split: tiles / bank / class / partials".into(),
            ));
        }
        let (qp, _g1) = q.0.device_ptr(&self.stream);
        let (rp, _g2) = rel.0.device_ptr(&self.stream);
        let (sp, _g3) = sink.0.device_ptr(&self.stream);
        let (kp, _g4) = k.device_ptr(&self.stream);
        let (vp, _g5) = v.device_ptr(&self.stream);
        let (tp, _g6) = tiles.device_ptr(&self.stream);
        let (jp, _g7) = proj_t.device_ptr(&self.stream);
        let (op, _g8) = out.device_ptr_mut(&self.stream);
        let (pp, _g9) = part_o.device_ptr_mut(&self.stream);
        let (mp, _g10) = part_ml.device_ptr_mut(&self.stream);
        let (bt, bps, ring) = match layout {
            Inkling2KvLayout::Paged(b, bps) => (Some(b.device_ptr(&self.stream)), bps, 0),
            Inkling2KvLayout::Ring(r) => (None, 0, r),
        };
        let btp = bt.as_ref().map_or(0, |g| g.0);
        // SAFETY: ABI contract (slot 1009); as slot 993's, the partials sized
        // for rows x heads x splits (checked above)
        check(unsafe {
            f(
                (qp + (q.1 * 4) as u64) as *const _,
                q.2 as u32,
                (rp + (rel.1 * 4) as u64) as *const _,
                rel.2 as u32,
                (sp + (sink.1 * 4) as u64) as *const _,
                sink.2 as u32,
                kp as *mut _,
                vp as *mut _,
                btp as *const _,
                bps as u32,
                ring as u32,
                dtype,
                tp as *const _,
                n_tiles as u32,
                jp as *const _,
                op as *mut _,
                pp as *mut _,
                mp as *mut _,
                rows as u32,
                heads as u32,
                kv_heads as u32,
                window as u32,
                extent as u32,
                scale,
                splits as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 1005 is here: [`Self::inkling2_attn_tile`] takes the gated outputs.
    pub fn has_inkling2_tile_g(&self) -> bool {
        self.kernels.inkling2_attn_tile_g.is_some() && self.has_inkling2_tile()
    }

    /// Slot 993: slot 990's attention for prompt runs on the tensor cores
    /// (f16 / fp8 stores). `tiles` [n_tiles][4]: (first row, rows <= 32,
    /// slot, first position) - rows of one run, consecutive positions;
    /// `proj_t` the bias bank as [extent][16] bf16; the gated outputs are
    /// slot 1005's (the caller checks [`Self::has_inkling2_tile_g`]).
    /// Ok(false) when the pack has no slot 993.
    #[allow(clippy::too_many_arguments)]
    pub fn inkling2_attn_tile(
        &self,
        q: (&CudaSlice<f32>, usize, usize),
        rel: (&CudaSlice<f32>, usize, usize),
        sink: (&CudaSlice<f32>, usize, usize),
        (k, v): (&CudaSlice<u8>, &CudaSlice<u8>),
        layout: Inkling2KvLayout<'_>,
        dtype: u32,
        (tiles, n_tiles): (&CudaSlice<u32>, usize),
        proj_t: &CudaSlice<u8>,
        out: Inkling2AttnOut<'_>,
        (heads, kv_heads): (usize, usize),
        (window, extent): (usize, usize),
        scale: f32,
    ) -> Result<bool, GpuError> {
        let Some(f) = self.kernels.inkling2_attn_tile else {
            return Ok(false);
        };
        if tiles.len() < n_tiles * 4 || proj_t.len() < 2 * 16 * extent || dtype > 1 {
            return Err(GpuError::Unsupported(
                "inkling2_attn_tile: tiles / bank / class".into(),
            ));
        }
        let ((op, _g8), out_rows, gate, out16) = match out {
            Inkling2AttnOut::Plain(o) => {
                let n = o.len() / (heads * 128);
                (o.device_ptr_mut(&self.stream), n, None, false)
            }
            Inkling2AttnOut::Gated(o, g, gs) => {
                let n = o.len() / (heads * 128);
                (o.device_ptr_mut(&self.stream), n, Some((g, gs)), false)
            }
            Inkling2AttnOut::Gated16(o, g, gs) => {
                let n = o.len() / (heads * 128);
                (o.device_ptr_mut(&self.stream), n, Some((g, gs)), true)
            }
        };
        if let Some((g, gs)) = gate
            && (gs < heads * 128 || g.len() < out_rows * gs)
        {
            return Err(GpuError::Unsupported(
                "inkling2_attn_tile: the gate plane".into(),
            ));
        }
        let (qp, _g1) = q.0.device_ptr(&self.stream);
        let (rp, _g2) = rel.0.device_ptr(&self.stream);
        let (sp, _g3) = sink.0.device_ptr(&self.stream);
        let (kp, _g4) = k.device_ptr(&self.stream);
        let (vp, _g5) = v.device_ptr(&self.stream);
        let (tp, _g6) = tiles.device_ptr(&self.stream);
        let (jp, _g7) = proj_t.device_ptr(&self.stream);
        let (bt, bps, ring) = match layout {
            Inkling2KvLayout::Paged(b, bps) => (Some(b.device_ptr(&self.stream)), bps, 0),
            Inkling2KvLayout::Ring(r) => (None, 0, r),
        };
        let btp = bt.as_ref().map_or(0, |g| g.0);
        if let Some((g, gs)) = gate {
            let fg = self
                .kernels
                .inkling2_attn_tile_g
                .ok_or(GpuError::MissingOp("inkling2_attn_tile_g"))?;
            let (gp, _g9) = g.device_ptr(&self.stream);
            // SAFETY: ABI contract (slot 1005); as slot 993's, the gate plane
            // covers the tiles' rows at stride gs (checked above)
            return check(unsafe {
                fg(
                    (qp + (q.1 * 4) as u64) as *const _,
                    q.2 as u32,
                    (rp + (rel.1 * 4) as u64) as *const _,
                    rel.2 as u32,
                    (sp + (sink.1 * 4) as u64) as *const _,
                    sink.2 as u32,
                    kp as *mut _,
                    vp as *mut _,
                    btp as *const _,
                    bps as u32,
                    ring as u32,
                    dtype,
                    tp as *const _,
                    n_tiles as u32,
                    jp as *const _,
                    op as *mut _,
                    heads as u32,
                    kv_heads as u32,
                    window as u32,
                    extent as u32,
                    scale,
                    gp as *const _,
                    gs as u32,
                    out16 as u32,
                    self.stream_ptr(),
                )
            })
            .map(|()| true);
        }
        // SAFETY: ABI contract (slot 993); the caller built `tiles` over rows
        // the q / out planes cover, stored every key they reach first
        check(unsafe {
            f(
                (qp + (q.1 * 4) as u64) as *const _,
                q.2 as u32,
                (rp + (rel.1 * 4) as u64) as *const _,
                rel.2 as u32,
                (sp + (sink.1 * 4) as u64) as *const _,
                sink.2 as u32,
                kp as *mut _,
                vp as *mut _,
                btp as *const _,
                bps as u32,
                ring as u32,
                dtype,
                tp as *const _,
                n_tiles as u32,
                jp as *const _,
                op as *mut _,
                heads as u32,
                kv_heads as u32,
                window as u32,
                extent as u32,
                scale,
                self.stream_ptr(),
            )
        })
        .map(|()| true)
    }
}

/// Where slots 993 / 1005 write the attention rows ([rows][heads x 128]).
pub enum Inkling2AttnOut<'a> {
    /// f32 rows (slot 993).
    Plain(&'a mut CudaSlice<f32>),
    /// f32 rows x sigmoid(gate) (slot 1005): the gate plane, its row stride.
    Gated(&'a mut CudaSlice<f32>, &'a CudaSlice<f32>, usize),
    /// bf16 rows x sigmoid(gate) (slot 1005): convert_f32_bf16 of `Gated`'s.
    Gated16(&'a mut CudaSlice<half::bf16>, &'a CudaSlice<f32>, usize),
}

/// The checkpoint's rms_norm_eps (every unweighted norm of the family).
fn inkling2_eps() -> f32 {
    paddock_models::inkling2::EPS
}
