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
