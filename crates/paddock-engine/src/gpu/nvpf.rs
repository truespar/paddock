//! The W4A16 prefill tile (gemm/nvf4_pf, slots 920-923): NVFP4 row planes x
//! bf16 rows on m16n8k16 at prefill widths, the codes decoded once a tile.
//! Dense planes (attention, dense FFNs, a shared expert) and routed experts
//! over `moe_align_bm(64)`'s blocks. Row-invariant: a token's outputs are the
//! same bits in any chunk shape.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use super::error::check;
use super::{GpuError, GpuExecutor, Nvf4Layout, Nvf4MoeLayout, Nvf4MoePlane, Nvf4Plane};

/// Tokens a routed block holds for slots 922 / 923.
pub const NVPF_MOE_BLOCK: usize = 64;

fn row_plane(w: &Nvf4Plane, op: &'static str) -> Result<(), GpuError> {
    if w.layout != Nvf4Layout::Row || w.gu_pairs || !w.in_dim.is_multiple_of(64) {
        return Err(GpuError::Unsupported(format!(
            "{op}: needs a plain Row plane with in_dim % 64 (got {:?}, in {})",
            w.layout, w.in_dim
        )));
    }
    Ok(())
}

/// The last element a strided row walk touches, +1.
fn span(rows: usize, ld: usize, off: usize, width: usize) -> usize {
    if rows == 0 {
        0
    } else {
        (rows - 1) * ld + off + width
    }
}

/// Slot 934's outputs: q in f16 (rows from the band on), the band's f32 q
/// rows, the normed + roped k, v.
pub struct PfQkvOut<'a> {
    pub q16: &'a mut CudaSlice<half::f16>,
    pub qn: &'a mut CudaSlice<f32>,
    pub kn: &'a mut CudaSlice<f32>,
    pub v: &'a mut CudaSlice<f32>,
}

/// Slot 934's norm + rope: `qk_norm_rope_batch`'s inputs (norm weights,
/// positions, heads, eps, the six rope parameters) plus the decode band's
/// rows and the f16 q scale.
pub struct PfNorm<'a> {
    pub qw: &'a CudaSlice<f32>,
    pub kw: &'a CudaSlice<f32>,
    pub pos: &'a CudaSlice<u32>,
    pub n_head: usize,
    pub n_kv: usize,
    pub dec: usize,
    pub eps: f32,
    pub rope: (f32, f32, f32, f32, f32, f32),
    pub qscale: f32,
}

impl GpuExecutor {
    /// True when all four slots are in the pack.
    pub fn has_nvf4_pf(&self) -> bool {
        self.kernels.nvf4_pf_gemm.is_some()
            && self.kernels.nvf4_pf_gu.is_some()
            && self.kernels.nvf4_pf_moe_gu.is_some()
            && self.kernels.nvf4_pf_moe_dn.is_some()
    }

    /// `y[c * ldy + yoff + r] = (W x_c)_r * scale2` for `ntok` bf16 rows of
    /// `x` (row stride `ldx`, column offset `xoff`), r < `w.out_dim`.
    #[allow(clippy::too_many_arguments)]
    pub fn nvf4_pf_gemm(
        &self,
        w: &Nvf4Plane,
        x: &CudaSlice<half::bf16>,
        ldx: usize,
        xoff: usize,
        y: &mut CudaSlice<f32>,
        ldy: usize,
        yoff: usize,
        ntok: usize,
    ) -> Result<(), GpuError> {
        const OP: &str = "nvf4_pf_gemm";
        let f = self.kernels.nvf4_pf_gemm.ok_or(GpuError::MissingOp(OP))?;
        row_plane(w, OP)?;
        if x.len() < span(ntok, ldx, xoff, w.in_dim) || y.len() < span(ntok, ldy, yoff, w.out_dim) {
            return Err(GpuError::Unsupported(format!(
                "{OP}: x / y too small for {ntok} rows of [{}, {}]",
                w.out_dim, w.in_dim
            )));
        }
        let (dp, _a1) = w.data.device_ptr(&self.stream);
        let (sp, _a2) = w.scale.device_ptr(&self.stream);
        let (xp, _a3) = x.device_ptr(&self.stream);
        let (yp, _a4) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; spans checked above
        check(unsafe {
            f(
                dp as *const _,
                sp as *const _,
                w.scale2,
                xp as *const _,
                ldx as u32,
                xoff as u32,
                yp as *mut _,
                ldy as u32,
                yoff as u32,
                w.in_dim as u32,
                w.out_dim as u32,
                ntok as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Whether the pack carries the q|k|v grid with the norm + rope
    /// epilogues (slot 934).
    pub fn has_nvf4_pf_qkv_nr(&self) -> bool {
        self.kernels.nvf4_pf_qkv_nr.is_some()
    }

    /// The prefill q|k|v planes with the q / k norm + rope in the epilogues
    /// (slot 934): `out.q16` takes q as f16 x `nr.qscale` from row `nr.dec`
    /// on, `out.qn` the f32 rows below it (a decode band's), `out.kn` the
    /// normed + roped k, `out.v` v as projected - each row the bits of
    /// [`Self::nvf4_pf_gemm`] + [`Self::qk_norm_rope_batch_q16`]. `Ok(false)`,
    /// nothing launched, without the slot, off its arm or its shapes.
    pub fn nvf4_pf_qkv_nr(
        &self,
        [wq, wk, wv]: [&Nvf4Plane; 3],
        out: PfQkvOut<'_>,
        nr: &PfNorm<'_>,
        x: &CudaSlice<half::bf16>,
        ntok: usize,
    ) -> Result<bool, GpuError> {
        const OP: &str = "nvf4_pf_qkv_nr";
        let Some(f) = self.kernels.nvf4_pf_qkv_nr else {
            return Ok(false);
        };
        for w in [wq, wk, wv] {
            row_plane(w, OP)?;
        }
        let (k, nh, nkv) = (wq.in_dim, nr.n_head, nr.n_kv);
        let fits = (wk.in_dim, wv.in_dim) == (k, k)
            && (wq.out_dim, wk.out_dim, wv.out_dim) == (nh * 128, nkv * 128, nkv * 128)
            && x.len() >= span(ntok, k, 0, k)
            && out.q16.len() >= ntok * nh * 128
            && out.qn.len() >= nr.dec.min(ntok) * nh * 128
            && out.kn.len() >= ntok * nkv * 128
            && out.v.len() >= ntok * nkv * 128
            && nr.qw.len() >= 128
            && nr.kw.len() >= 128
            && nr.pos.len() >= ntok;
        if !fits {
            return Ok(false);
        }
        let (ts, fs, clo, chi, ext, msc) = nr.rope;
        let rope = [ts, fs, clo, chi, ext, msc];
        let (dq, _a1) = wq.data.device_ptr(&self.stream);
        let (sq, _a2) = wq.scale.device_ptr(&self.stream);
        let (dk, _a3) = wk.data.device_ptr(&self.stream);
        let (sk, _a4) = wk.scale.device_ptr(&self.stream);
        let (dv, _a5) = wv.data.device_ptr(&self.stream);
        let (sv, _a6) = wv.scale.device_ptr(&self.stream);
        let (q16, _a7) = out.q16.device_ptr_mut(&self.stream);
        let (qn, _a8) = out.qn.device_ptr_mut(&self.stream);
        let (kn, _a9) = out.kn.device_ptr_mut(&self.stream);
        let (vp, _a10) = out.v.device_ptr_mut(&self.stream);
        let (qw, _a11) = nr.qw.device_ptr(&self.stream);
        let (kw, _a12) = nr.kw.device_ptr(&self.stream);
        let (pp, _a13) = nr.pos.device_ptr(&self.stream);
        let (xp, _a14) = x.device_ptr(&self.stream);
        // SAFETY: ABI contract (slot 934); spans checked above, `rope` read
        // on the host during the call
        let rc = unsafe {
            f(
                dq as *const _,
                sq as *const _,
                wq.scale2,
                dk as *const _,
                sk as *const _,
                wk.scale2,
                dv as *const _,
                sv as *const _,
                wv.scale2,
                q16 as *mut _,
                qn as *mut _,
                kn as *mut _,
                vp as *mut _,
                nh as u32,
                nkv as u32,
                qw as *const _,
                kw as *const _,
                pp as *const _,
                nr.dec as u32,
                nr.eps,
                rope.as_ptr(),
                nr.qscale,
                xp as *const _,
                k as u32,
                ntok as u32,
                self.stream_ptr(),
            )
        };
        if rc == 801 {
            return Ok(false);
        }
        check(rc)?;
        Ok(true)
    }

    /// Whether the pack carries the routed + shared pair grids (slots 932 /
    /// 933).
    pub fn has_nvf4_pf_moe_sh(&self) -> bool {
        self.kernels.nvf4_pf_moe_gu_sh.is_some() && self.kernels.nvf4_pf_moe_dn_sh.is_some()
    }

    /// [`Self::nvf4_pf_gemm`] over up to three planes sharing `x` and
    /// in_dim in one launch (slot 931; q|k|v), each plane writing its own
    /// rows (`ldy` = its out_dim, no offset) with the per-plane call's bits.
    /// `Ok(false)`, nothing launched, without the slot or off its arm.
    pub fn nvf4_pf_gemm3(
        &self,
        planes: &mut [(&Nvf4Plane, &mut CudaSlice<f32>)],
        x: &CudaSlice<half::bf16>,
        ntok: usize,
    ) -> Result<bool, GpuError> {
        const OP: &str = "nvf4_pf_gemm3";
        let Some(f) = self.kernels.nvf4_pf_gemm3 else {
            return Ok(false);
        };
        let Some(k) = planes.first().map(|(w, _)| w.in_dim) else {
            return Ok(true);
        };
        if planes.len() > 3 || x.len() < span(ntok, k, 0, k) {
            return Ok(false);
        }
        for (w, y) in planes.iter() {
            row_plane(w, OP)?;
            if w.in_dim != k || y.len() < span(ntok, w.out_dim, 0, w.out_dim) {
                return Ok(false);
            }
        }
        let null = std::ptr::null::<core::ffi::c_void>();
        let mut d = [null; 3];
        let mut sc = [null; 3];
        let mut ys = [std::ptr::null_mut::<core::ffi::c_void>(); 3];
        let (mut fs, mut ns) = ([0f32; 3], [0u32; 3]);
        let mut guards = Vec::with_capacity(3 * planes.len());
        for (i, (w, y)) in planes.iter_mut().enumerate() {
            let (dp, g1) = w.data.device_ptr(&self.stream);
            let (sp, g2) = w.scale.device_ptr(&self.stream);
            let (yp, g3) = y.device_ptr_mut(&self.stream);
            (d[i], sc[i], ys[i]) = (dp as *const _, sp as *const _, yp as *mut _);
            (fs[i], ns[i]) = (w.scale2, w.out_dim as u32);
            guards.push((g1, g2, g3));
        }
        let (xp, _gx) = x.device_ptr(&self.stream);
        // SAFETY: ABI contract (slot 931); spans checked above
        let rc = unsafe {
            f(
                d[0],
                sc[0],
                fs[0],
                ys[0],
                ns[0],
                ns[0],
                d[1],
                sc[1],
                fs[1],
                ys[1],
                ns[1],
                ns[1],
                d[2],
                sc[2],
                fs[2],
                ys[2],
                ns[2],
                ns[2],
                xp as *const _,
                k as u32,
                0,
                k as u32,
                ntok as u32,
                self.stream_ptr(),
            )
        };
        drop(guards);
        if rc == 801 {
            return Ok(false);
        }
        check(rc)?;
        Ok(true)
    }

    /// `act[c * lda + aoff + ch] = bf16(silu(gate x_c) * (up x_c))`, ch <
    /// `gate.out_dim` (the factors folded in).
    #[allow(clippy::too_many_arguments)]
    pub fn nvf4_pf_gu(
        &self,
        gate: &Nvf4Plane,
        up: &Nvf4Plane,
        x: &CudaSlice<half::bf16>,
        ldx: usize,
        act: &mut CudaSlice<half::bf16>,
        lda: usize,
        aoff: usize,
        ntok: usize,
    ) -> Result<(), GpuError> {
        const OP: &str = "nvf4_pf_gu";
        let f = self.kernels.nvf4_pf_gu.ok_or(GpuError::MissingOp(OP))?;
        row_plane(gate, OP)?;
        row_plane(up, OP)?;
        let (k, ff) = (gate.in_dim, gate.out_dim);
        if (up.in_dim, up.out_dim) != (k, ff) || !ff.is_multiple_of(64) {
            return Err(GpuError::Unsupported(format!(
                "{OP}: gate / up planes disagree or ff {ff} is not a multiple of 64"
            )));
        }
        if x.len() < span(ntok, ldx, 0, k) || act.len() < span(ntok, lda, aoff, ff) {
            return Err(GpuError::Unsupported(format!(
                "{OP}: x / act too small for {ntok} rows"
            )));
        }
        let (gd, _a1) = gate.data.device_ptr(&self.stream);
        let (gs, _a2) = gate.scale.device_ptr(&self.stream);
        let (ud, _a3) = up.data.device_ptr(&self.stream);
        let (us, _a4) = up.scale.device_ptr(&self.stream);
        let (xp, _a5) = x.device_ptr(&self.stream);
        let (ap, _a6) = act.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; spans checked above
        check(unsafe {
            f(
                gd as *const _,
                gs as *const _,
                gate.scale2,
                ud as *const _,
                us as *const _,
                up.scale2,
                xp as *const _,
                ldx as u32,
                ap as *mut _,
                lda as u32,
                aoff as u32,
                k as u32,
                ff as u32,
                ntok as u32,
                self.stream_ptr(),
            )
        })
    }

    /// The routed gate|up over `moe_align_bm(64)`'s `n_blocks` blocks:
    /// `act[t * aw + slot * ff + ch]` for every sorted (token, slot); `x`
    /// the rows' bf16 activations `[rows, in]`.
    #[allow(clippy::too_many_arguments)]
    pub fn nvf4_pf_moe_gu(
        &self,
        gate: &Nvf4MoePlane,
        up: &Nvf4MoePlane,
        sorted_row: &CudaSlice<u32>,
        sorted_slot: &CudaSlice<u32>,
        block_expert: &CudaSlice<u32>,
        x: &CudaSlice<half::bf16>,
        act: &mut CudaSlice<half::bf16>,
        aw: usize,
        n_blocks: usize,
    ) -> Result<(), GpuError> {
        const OP: &str = "nvf4_pf_moe_gu";
        let f = self.kernels.nvf4_pf_moe_gu.ok_or(GpuError::MissingOp(OP))?;
        Self::moe_layout_ok(gate, Nvf4MoeLayout::Row, OP)?;
        Self::moe_layout_ok(up, Nvf4MoeLayout::Row, OP)?;
        if (up.in_dim, up.ff, up.n_expert) != (gate.in_dim, gate.ff, gate.n_expert)
            || !gate.ff.is_multiple_of(64)
            || !gate.in_dim.is_multiple_of(64)
        {
            return Err(GpuError::Unsupported(format!(
                "{OP}: gate / up planes disagree or are off the 64 grid"
            )));
        }
        if sorted_row.len() < n_blocks * NVPF_MOE_BLOCK
            || sorted_slot.len() < n_blocks * NVPF_MOE_BLOCK
            || block_expert.len() < n_blocks
        {
            return Err(GpuError::Unsupported(format!(
                "{OP}: sorting too small for {n_blocks} blocks"
            )));
        }
        // the kernel indexes x and act by sorted token ids; the caller sized
        // both for the rows it sorted (debug-checked against the planes here)
        debug_assert!(x.len().is_multiple_of(gate.in_dim) && act.len() >= aw);
        let (gd, _a1) = gate.data.device_ptr(&self.stream);
        let (gs, _a2) = gate.scale.device_ptr(&self.stream);
        let (g2, _a3) = gate.scale2.device_ptr(&self.stream);
        let (ud, _a4) = up.data.device_ptr(&self.stream);
        let (us, _a5) = up.scale.device_ptr(&self.stream);
        let (u2, _a6) = up.scale2.device_ptr(&self.stream);
        let (sr, _a7) = sorted_row.device_ptr(&self.stream);
        let (ss, _a8) = sorted_slot.device_ptr(&self.stream);
        let (be, _a9) = block_expert.device_ptr(&self.stream);
        let (xp, _a10) = x.device_ptr(&self.stream);
        let (ap, _a11) = act.device_ptr_mut(&self.stream);
        // slot 926 where the pack has it: the expert count elects the
        // kernel's weight copy width (the bits are the same either way)
        if let Some(fne) = self.kernels.nvf4_pf_moe_gu_ne {
            // SAFETY: ABI contract; shapes checked above
            return check(unsafe {
                fne(
                    gd as *const _,
                    gs as *const _,
                    g2 as *const _,
                    ud as *const _,
                    us as *const _,
                    u2 as *const _,
                    sr as *const _,
                    ss as *const _,
                    be as *const _,
                    xp as *const _,
                    ap as *mut _,
                    gate.in_dim as u32,
                    gate.ff as u32,
                    aw as u32,
                    n_blocks as u32,
                    gate.n_expert as u32,
                    self.stream_ptr(),
                )
            });
        }
        // SAFETY: ABI contract; shapes checked above
        check(unsafe {
            f(
                gd as *const _,
                gs as *const _,
                g2 as *const _,
                ud as *const _,
                us as *const _,
                u2 as *const _,
                sr as *const _,
                ss as *const _,
                be as *const _,
                xp as *const _,
                ap as *mut _,
                gate.in_dim as u32,
                gate.ff as u32,
                aw as u32,
                n_blocks as u32,
                self.stream_ptr(),
            )
        })
    }

    /// [`Self::nvf4_pf_moe_gu`] and [`Self::nvf4_pf_gu`]'s shared gate|up
    /// (`shared` = (gate, up), landing at column `aoff` of the same act rows,
    /// `ntok` rows) in one grid (slot 932), each its own launch's bits.
    /// `Ok(false)`, nothing launched, without the slot or off its arm.
    #[allow(clippy::too_many_arguments)]
    pub fn nvf4_pf_moe_gu_sh(
        &self,
        gate: &Nvf4MoePlane,
        up: &Nvf4MoePlane,
        sorted_row: &CudaSlice<u32>,
        sorted_slot: &CudaSlice<u32>,
        block_expert: &CudaSlice<u32>,
        x: &CudaSlice<half::bf16>,
        act: &mut CudaSlice<half::bf16>,
        aw: usize,
        n_blocks: usize,
        (sg, su): (&Nvf4Plane, &Nvf4Plane),
        aoff: usize,
        ntok: usize,
    ) -> Result<bool, GpuError> {
        const OP: &str = "nvf4_pf_moe_gu_sh";
        let Some(f) = self.kernels.nvf4_pf_moe_gu_sh else {
            return Ok(false);
        };
        Self::moe_layout_ok(gate, Nvf4MoeLayout::Row, OP)?;
        Self::moe_layout_ok(up, Nvf4MoeLayout::Row, OP)?;
        row_plane(sg, OP)?;
        row_plane(su, OP)?;
        let k = gate.in_dim;
        if (up.in_dim, up.ff, up.n_expert) != (k, gate.ff, gate.n_expert)
            || (sg.in_dim, su.in_dim, su.out_dim) != (k, k, sg.out_dim)
            || sorted_row.len() < n_blocks * NVPF_MOE_BLOCK
            || sorted_slot.len() < n_blocks * NVPF_MOE_BLOCK
            || block_expert.len() < n_blocks
            || x.len() < span(ntok, k, 0, k)
            || act.len() < span(ntok, aw, aoff, sg.out_dim)
        {
            return Ok(false);
        }
        let (gd, _a1) = gate.data.device_ptr(&self.stream);
        let (gs, _a2) = gate.scale.device_ptr(&self.stream);
        let (g2, _a3) = gate.scale2.device_ptr(&self.stream);
        let (ud, _a4) = up.data.device_ptr(&self.stream);
        let (us, _a5) = up.scale.device_ptr(&self.stream);
        let (u2, _a6) = up.scale2.device_ptr(&self.stream);
        let (sr, _a7) = sorted_row.device_ptr(&self.stream);
        let (ss, _a8) = sorted_slot.device_ptr(&self.stream);
        let (be, _a9) = block_expert.device_ptr(&self.stream);
        let (xp, _a10) = x.device_ptr(&self.stream);
        let (ap, _a11) = act.device_ptr_mut(&self.stream);
        let (sgd, _a12) = sg.data.device_ptr(&self.stream);
        let (sgs, _a13) = sg.scale.device_ptr(&self.stream);
        let (sud, _a14) = su.data.device_ptr(&self.stream);
        let (sus, _a15) = su.scale.device_ptr(&self.stream);
        // SAFETY: ABI contract (slot 932); shapes checked above
        let rc = unsafe {
            f(
                gd as *const _,
                gs as *const _,
                g2 as *const _,
                ud as *const _,
                us as *const _,
                u2 as *const _,
                sr as *const _,
                ss as *const _,
                be as *const _,
                xp as *const _,
                ap as *mut _,
                k as u32,
                gate.ff as u32,
                aw as u32,
                n_blocks as u32,
                gate.n_expert as u32,
                sgd as *const _,
                sgs as *const _,
                sg.scale2,
                sud as *const _,
                sus as *const _,
                su.scale2,
                sg.out_dim as u32,
                aoff as u32,
                ntok as u32,
                self.stream_ptr(),
            )
        };
        if rc == 801 {
            return Ok(false);
        }
        check(rc)?;
        Ok(true)
    }

    /// [`Self::nvf4_pf_moe_dn`] and the shared down (`sd` over act from
    /// column `aoff`, into `y` [ntok][embd]) in one grid (slot 933), each its
    /// own launch's bits. `Ok(false)`, nothing launched, without the slot or
    /// off its arm.
    #[allow(clippy::too_many_arguments)]
    pub fn nvf4_pf_moe_dn_sh(
        &self,
        down: &Nvf4MoePlane,
        sorted_row: &CudaSlice<u32>,
        sorted_slot: &CudaSlice<u32>,
        block_expert: &CudaSlice<u32>,
        topk_w: &CudaSlice<f32>,
        act: &CudaSlice<half::bf16>,
        aw: usize,
        part: &mut CudaSlice<f32>,
        k: usize,
        n_blocks: usize,
        (sd, y): (&Nvf4Plane, &mut CudaSlice<f32>),
        aoff: usize,
        ntok: usize,
    ) -> Result<bool, GpuError> {
        const OP: &str = "nvf4_pf_moe_dn_sh";
        let Some(f) = self.kernels.nvf4_pf_moe_dn_sh else {
            return Ok(false);
        };
        Self::moe_layout_ok(down, Nvf4MoeLayout::Row, OP)?;
        row_plane(sd, OP)?;
        let (embd, ff) = (down.ff, down.in_dim);
        if sd.out_dim != embd
            || sorted_row.len() < n_blocks * NVPF_MOE_BLOCK
            || sorted_slot.len() < n_blocks * NVPF_MOE_BLOCK
            || block_expert.len() < n_blocks
            || act.len() < span(ntok, aw, aoff, sd.in_dim)
            || y.len() < span(ntok, embd, 0, embd)
        {
            return Ok(false);
        }
        let (dp, _a1) = down.data.device_ptr(&self.stream);
        let (sp, _a2) = down.scale.device_ptr(&self.stream);
        let (s2, _a3) = down.scale2.device_ptr(&self.stream);
        let (sr, _a4) = sorted_row.device_ptr(&self.stream);
        let (ss, _a5) = sorted_slot.device_ptr(&self.stream);
        let (be, _a6) = block_expert.device_ptr(&self.stream);
        let (wp, _a7) = topk_w.device_ptr(&self.stream);
        let (ap, _a8) = act.device_ptr(&self.stream);
        let (pp, _a9) = part.device_ptr_mut(&self.stream);
        let (sdd, _a10) = sd.data.device_ptr(&self.stream);
        let (sds, _a11) = sd.scale.device_ptr(&self.stream);
        let (yp, _a12) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 933); shapes checked above
        let rc = unsafe {
            f(
                dp as *const _,
                sp as *const _,
                s2 as *const _,
                sr as *const _,
                ss as *const _,
                be as *const _,
                wp as *const _,
                ap as *const _,
                aw as u32,
                pp as *mut _,
                ff as u32,
                embd as u32,
                k as u32,
                n_blocks as u32,
                sdd as *const _,
                sds as *const _,
                sd.scale2,
                yp as *mut _,
                sd.in_dim as u32,
                aoff as u32,
                ntok as u32,
                self.stream_ptr(),
            )
        };
        if rc == 801 {
            return Ok(false);
        }
        check(rc)?;
        Ok(true)
    }

    /// The routed down over the same blocks into bf16 partials `[rows][k]
    /// [embd]` held in an f32-typed buffer (`moe_slot_combine_bf16`'s
    /// storage), the top-k weight and expert factor folded in.
    #[allow(clippy::too_many_arguments)]
    pub fn nvf4_pf_moe_dn(
        &self,
        down: &Nvf4MoePlane,
        sorted_row: &CudaSlice<u32>,
        sorted_slot: &CudaSlice<u32>,
        block_expert: &CudaSlice<u32>,
        topk_w: &CudaSlice<f32>,
        act: &CudaSlice<half::bf16>,
        aw: usize,
        part: &mut CudaSlice<f32>,
        k: usize,
        n_blocks: usize,
    ) -> Result<(), GpuError> {
        const OP: &str = "nvf4_pf_moe_dn";
        let f = self.kernels.nvf4_pf_moe_dn.ok_or(GpuError::MissingOp(OP))?;
        Self::moe_layout_ok(down, Nvf4MoeLayout::Row, OP)?;
        let (embd, ff) = (down.ff, down.in_dim);
        if !ff.is_multiple_of(64) || !embd.is_multiple_of(128) || !aw.is_multiple_of(8) {
            return Err(GpuError::Unsupported(format!(
                "{OP}: ff {ff} / embd {embd} / aw {aw} off the tile grid"
            )));
        }
        if sorted_row.len() < n_blocks * NVPF_MOE_BLOCK
            || sorted_slot.len() < n_blocks * NVPF_MOE_BLOCK
            || block_expert.len() < n_blocks
        {
            return Err(GpuError::Unsupported(format!(
                "{OP}: sorting too small for {n_blocks} blocks"
            )));
        }
        debug_assert!((2 * part.len()).is_multiple_of(k * embd) && act.len() >= aw);
        let (dp, _a1) = down.data.device_ptr(&self.stream);
        let (sp, _a2) = down.scale.device_ptr(&self.stream);
        let (s2, _a3) = down.scale2.device_ptr(&self.stream);
        let (sr, _a4) = sorted_row.device_ptr(&self.stream);
        let (ss, _a5) = sorted_slot.device_ptr(&self.stream);
        let (be, _a6) = block_expert.device_ptr(&self.stream);
        let (wp, _a7) = topk_w.device_ptr(&self.stream);
        let (ap, _a8) = act.device_ptr(&self.stream);
        let (pp, _a9) = part.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; shapes checked above
        check(unsafe {
            f(
                dp as *const _,
                sp as *const _,
                s2 as *const _,
                sr as *const _,
                ss as *const _,
                be as *const _,
                wp as *const _,
                ap as *const _,
                aw as u32,
                pp as *mut _,
                ff as u32,
                embd as u32,
                k as u32,
                n_blocks as u32,
                self.stream_ptr(),
            )
        })
    }
}
