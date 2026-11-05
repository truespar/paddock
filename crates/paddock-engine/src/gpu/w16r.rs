//! The W4A16 expert pair over ROW-MAJOR planes (moe/nvf4_w16r, slots
//! 918-919): the checkpoint's 16-bit-activation class on tensor cores for a
//! SwiGLU MoE whose planes stay in the layout the f32 decode GEMVs read, so
//! one residency serves the prefill MMA class and the decode GEMV class.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use super::error::check;
use super::{GpuError, GpuExecutor, Nvf4Layout, Nvf4MoeLayout, Nvf4MoePlane, Nvf4Plane};

/// Tokens a sorted block holds (`moe_align`'s tile, the kernels' PD_W16_BM).
pub const W16R_BLOCK: usize = 32;

fn row_plane(w: &Nvf4Plane, op: &'static str) -> Result<(), GpuError> {
    if w.layout != Nvf4Layout::Row || w.gu_pairs {
        return Err(GpuError::Unsupported(format!(
            "{op}: shared-expert plane is {:?}, this kernel reads plain Row planes",
            w.layout
        )));
    }
    Ok(())
}

impl GpuExecutor {
    /// True when the pack carries both halves of the row-plane W16 pair.
    pub fn has_nvf4_moe_w16r(&self) -> bool {
        self.kernels.nvf4_moe_gu_swiglu_w16r.is_some()
            && self.kernels.nvf4_moe_down_part_w16r.is_some()
    }

    /// Routed + shared gate|up and SwiGLU over `moe_align`'s 32-row sorting
    /// (`n_blocks` blocks of `sorted_row` / `sorted_slot`, `block_expert`):
    /// `act [rows][k * ff_r + ff_s]` bf16 = silu(gate) * up, the expert
    /// factors folded in. `x` = the rows' bf16 activations.
    #[allow(clippy::too_many_arguments)]
    pub fn nvf4_moe_gu_swiglu_w16r(
        &self,
        gate: &Nvf4MoePlane,
        up: &Nvf4MoePlane,
        sh_gate: &Nvf4Plane,
        sh_up: &Nvf4Plane,
        sorted_row: &CudaSlice<u32>,
        sorted_slot: &CudaSlice<u32>,
        block_expert: &CudaSlice<u32>,
        x: &CudaSlice<half::bf16>,
        act: &mut CudaSlice<half::bf16>,
        k: usize,
        n_blocks: usize,
        rows: usize,
    ) -> Result<(), GpuError> {
        const OP: &str = "nvf4_moe_gu_swiglu_w16r";
        let f = self
            .kernels
            .nvf4_moe_gu_swiglu_w16r
            .ok_or(GpuError::MissingOp(OP))?;
        Self::moe_layout_ok(gate, Nvf4MoeLayout::Row, OP)?;
        Self::moe_layout_ok(up, Nvf4MoeLayout::Row, OP)?;
        row_plane(sh_gate, OP)?;
        row_plane(sh_up, OP)?;
        let (in_dim, ff_r, ff_s) = (gate.in_dim, gate.ff, sh_gate.out_dim);
        if (up.in_dim, up.ff, up.n_expert) != (in_dim, ff_r, gate.n_expert)
            || (sh_gate.in_dim, sh_up.in_dim, sh_up.out_dim) != (in_dim, in_dim, ff_s)
        {
            return Err(GpuError::Unsupported(format!(
                "{OP}: gate/up/shared planes disagree on shape"
            )));
        }
        if sorted_row.len() < n_blocks * W16R_BLOCK
            || sorted_slot.len() < n_blocks * W16R_BLOCK
            || block_expert.len() < n_blocks
            || x.len() < rows * in_dim
            || act.len() < rows * (k * ff_r + ff_s)
        {
            return Err(GpuError::Unsupported(format!(
                "{OP}: scratch too small for {rows} rows x {k} picks, {n_blocks} blocks"
            )));
        }
        let (gd, _a1) = gate.data.device_ptr(&self.stream);
        let (gs, _a2) = gate.scale.device_ptr(&self.stream);
        let (g2, _a3) = gate.scale2.device_ptr(&self.stream);
        let (ud, _a4) = up.data.device_ptr(&self.stream);
        let (us, _a5) = up.scale.device_ptr(&self.stream);
        let (u2, _a6) = up.scale2.device_ptr(&self.stream);
        let (sgd, _a7) = sh_gate.data.device_ptr(&self.stream);
        let (sgs, _a8) = sh_gate.scale.device_ptr(&self.stream);
        let (sud, _a9) = sh_up.data.device_ptr(&self.stream);
        let (sus, _a10) = sh_up.scale.device_ptr(&self.stream);
        let (srp, _a11) = sorted_row.device_ptr(&self.stream);
        let (ssl, _a12) = sorted_slot.device_ptr(&self.stream);
        let (bep, _a13) = block_expert.device_ptr(&self.stream);
        let (xp, _a14) = x.device_ptr(&self.stream);
        let (ap, _a15) = act.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; shapes and scratch sizes checked above
        check(unsafe {
            f(
                gd as *const _,
                gs as *const _,
                g2 as *const _,
                ud as *const _,
                us as *const _,
                u2 as *const _,
                sgd as *const _,
                sgs as *const _,
                sud as *const _,
                sus as *const _,
                sh_gate.scale2,
                sh_up.scale2,
                srp as *const _,
                ssl as *const _,
                bep as *const _,
                xp as *const _,
                ap as *mut _,
                in_dim as u32,
                ff_r as u32,
                ff_s as u32,
                k as u32,
                n_blocks as u32,
                rows as u32,
                self.stream_ptr(),
            )
        })
    }

    /// The matching down into `part [rows][k + 1][embd]` f32, each slot's
    /// top-k weight and expert factor folded in (slot k = the shared expert)
    /// - `moe_slot_combine`'s fixed slot order. `topk_w` [rows][k].
    #[allow(clippy::too_many_arguments)]
    pub fn nvf4_moe_down_part_w16r(
        &self,
        down: &Nvf4MoePlane,
        sh_down: &Nvf4Plane,
        sorted_row: &CudaSlice<u32>,
        sorted_slot: &CudaSlice<u32>,
        block_expert: &CudaSlice<u32>,
        topk_w: &CudaSlice<f32>,
        act: &CudaSlice<half::bf16>,
        part: &mut CudaSlice<f32>,
        k: usize,
        n_blocks: usize,
        rows: usize,
    ) -> Result<(), GpuError> {
        const OP: &str = "nvf4_moe_down_part_w16r";
        let f = self
            .kernels
            .nvf4_moe_down_part_w16r
            .ok_or(GpuError::MissingOp(OP))?;
        Self::moe_layout_ok(down, Nvf4MoeLayout::Row, OP)?;
        row_plane(sh_down, OP)?;
        let (embd, ff_r, ff_s) = (down.ff, down.in_dim, sh_down.in_dim);
        if sh_down.out_dim != embd {
            return Err(GpuError::Unsupported(format!(
                "{OP}: shared down lands {} rows, the routed down {embd}",
                sh_down.out_dim
            )));
        }
        if sorted_row.len() < n_blocks * W16R_BLOCK
            || sorted_slot.len() < n_blocks * W16R_BLOCK
            || block_expert.len() < n_blocks
            || topk_w.len() < rows * k
            || act.len() < rows * (k * ff_r + ff_s)
            || part.len() < rows * (k + 1) * embd
        {
            return Err(GpuError::Unsupported(format!(
                "{OP}: scratch too small for {rows} rows x {k} picks, {n_blocks} blocks"
            )));
        }
        let (rd, _a1) = down.data.device_ptr(&self.stream);
        let (rs, _a2) = down.scale.device_ptr(&self.stream);
        let (r2, _a3) = down.scale2.device_ptr(&self.stream);
        let (sd, _a4) = sh_down.data.device_ptr(&self.stream);
        let (ss, _a5) = sh_down.scale.device_ptr(&self.stream);
        let (srp, _a6) = sorted_row.device_ptr(&self.stream);
        let (ssl, _a7) = sorted_slot.device_ptr(&self.stream);
        let (bep, _a8) = block_expert.device_ptr(&self.stream);
        let (wp, _a9) = topk_w.device_ptr(&self.stream);
        let (ap, _a10) = act.device_ptr(&self.stream);
        let (pp, _a11) = part.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; shapes and scratch sizes checked above
        check(unsafe {
            f(
                rd as *const _,
                rs as *const _,
                r2 as *const _,
                sd as *const _,
                ss as *const _,
                sh_down.scale2,
                srp as *const _,
                ssl as *const _,
                bep as *const _,
                wp as *const _,
                ap as *const _,
                pp as *mut _,
                ff_r as u32,
                ff_s as u32,
                embd as u32,
                k as u32,
                n_blocks as u32,
                rows as u32,
                self.stream_ptr(),
            )
        })
    }
}
