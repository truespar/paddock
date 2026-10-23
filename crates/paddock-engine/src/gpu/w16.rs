//! The W16 decode class's ops - every row that stands for a decode step
//! (one-row decode, a decode tick's rows, a spec verify round's) runs one
//! batch-invariant class: the dense projections on bf16 / e4m3-widened-to-f16
//! planes against 16-bit activations (gemm/dense_w16), the checkpoint's
//! W4A16 experts on tensor cores (moe/nvf4_w16) and their routing front (the
//! router, top-k, activations cast and 32-row sorting in one launch).

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use super::error::check;
use super::{DeviceTensor, GpuError, GpuExecutor, Nvf4MoeLayout, Nvf4MoePlane, QuantTensor};

/// Rows past which `dense_w16` reads activations pre-cast to 16 bits (one
/// convert launch) instead of casting f32 in the kernel: an 8-token column's
/// in-kernel cast costs four loads and eight converts per chunk, which the
/// wide calls pay once per column per weight tile (GB10 2026-09-26, 32 rows:
/// 0.59x of the MMA tile cast in-kernel, 1.25x pre-cast).
pub const DENSE_W16_PRECAST_ROWS: usize = 8;

impl GpuExecutor {
    /// True when the pack carries `dense_w16` (slot 682).
    pub fn has_dense_w16(&self) -> bool {
        self.kernels.dense_w16.is_some()
    }

    /// The W16 decode class's dense GEMM over `rows` rows on a bf16 plane's
    /// row segment `[first_row, first_row + out_dim)` (house dims `[in,
    /// out]`, rows contiguous): `y[t][0..out_dim] = W x[t]`, batch-invariant:
    /// a row's bits are the same alone or in a verify round. `x` f32 [rows,
    /// in]; past 8 rows it is cast once into `x16` (bf16, rounds as the
    /// kernel's own cast does) and read 16-bit. `y` row stride `y_stride`.
    #[allow(clippy::too_many_arguments)]
    pub fn dense_w16_bf16(
        &self,
        w: &QuantTensor,
        first_row: usize,
        out_dim: usize,
        x: &CudaSlice<f32>,
        x16: &mut CudaSlice<half::bf16>,
        y: &mut CudaSlice<f32>,
        y_stride: usize,
        rows: usize,
    ) -> Result<(), GpuError> {
        let w = self.bf16_plane(w, "dense_w16_bf16")?;
        let in_dim = w.dims[0];
        debug_assert!(first_row + out_dim <= w.dims[1], "segment exceeds plane");
        let pre = rows > DENSE_W16_PRECAST_ROWS;
        if pre {
            self.convert_f32_bf16(x, x16, rows * in_dim)?;
        }
        let (wp, _g1) = w.bytes.device_ptr(&self.stream);
        let xg = if pre {
            x16.device_ptr(&self.stream).0
        } else {
            x.device_ptr(&self.stream).0
        };
        self.dense_w16_raw(
            wp + (first_row * in_dim * 2) as u64,
            0,
            xg,
            y,
            in_dim,
            out_dim,
            y_stride,
            rows,
            0,
            pre,
        )
    }

    /// [`Self::dense_w16_bf16`] over activations already cast to bf16 (one
    /// convert serving several segments of a fused plane - q, k and v).
    #[allow(clippy::too_many_arguments)]
    pub fn dense_w16_bf16_pre(
        &self,
        w: &QuantTensor,
        first_row: usize,
        out_dim: usize,
        x16: &CudaSlice<half::bf16>,
        y: &mut CudaSlice<f32>,
        y_stride: usize,
        rows: usize,
    ) -> Result<(), GpuError> {
        let w = self.bf16_plane(w, "dense_w16_bf16_pre")?;
        let in_dim = w.dims[0];
        debug_assert!(first_row + out_dim <= w.dims[1], "segment exceeds plane");
        debug_assert!(x16.len() >= rows * in_dim);
        let (wp, _g1) = w.bytes.device_ptr(&self.stream);
        let (xp, _g2) = x16.device_ptr(&self.stream);
        self.dense_w16_raw(
            wp + (first_row * in_dim * 2) as u64,
            0,
            xp,
            y,
            in_dim,
            out_dim,
            y_stride,
            rows,
            0,
            true,
        )
    }

    /// True when the pack carries the segmented `dense_w16` (slot 684).
    pub fn has_dense_w16_seg(&self) -> bool {
        self.kernels.dense_w16_seg.is_some()
    }

    /// [`Self::dense_w16_bf16`] over a whole bf16 plane whose rows split into
    /// three outputs - `[0, n0)` into `y0` `[rows, n0]`, `[n0, n0 + n1)` into
    /// `y1` `[rows, n1]`, the rest into `y2` - in one launch (a fused q|k|v
    /// plane). Each output's bits are those of its own segment's call.
    #[allow(clippy::too_many_arguments)]
    pub fn dense_w16_bf16_seg(
        &self,
        w: &QuantTensor,
        n0: usize,
        n1: usize,
        x: &CudaSlice<f32>,
        x16: &mut CudaSlice<half::bf16>,
        y0: &mut CudaSlice<f32>,
        y1: &mut CudaSlice<f32>,
        y2: &mut CudaSlice<f32>,
        rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .dense_w16_seg
            .ok_or(GpuError::MissingOp("dense_w16_seg"))?;
        let w = self.bf16_plane(w, "dense_w16_bf16_seg")?;
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        let n2 = out_dim - n0 - n1;
        debug_assert!(n0 > 0 && n1 > 0 && n2 > 0, "three segments");
        debug_assert!(y0.len() >= rows * n0 && y1.len() >= rows * n1 && y2.len() >= rows * n2);
        let pre = rows > DENSE_W16_PRECAST_ROWS;
        if pre {
            self.convert_f32_bf16(x, x16, rows * in_dim)?;
        }
        let (wp, _g1) = w.bytes.device_ptr(&self.stream);
        let xg = if pre {
            x16.device_ptr(&self.stream).0
        } else {
            x.device_ptr(&self.stream).0
        };
        let (p0, _g2) = y0.device_ptr_mut(&self.stream);
        let (p1, _g3) = y1.device_ptr_mut(&self.stream);
        let (p2, _g4) = y2.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; the plane holds [out, in] bf16 weights, x
        // [rows, in] (f32, or bf16 when pre-cast), each y [rows, n_seg]
        check(unsafe {
            f(
                wp as *const _,
                core::ptr::null(),
                xg as *const _,
                p0 as *mut _,
                p1 as *mut _,
                p2 as *mut _,
                n0 as u32,
                n1 as u32,
                in_dim as u32,
                out_dim as u32,
                rows as u32,
                0,
                pre as u32,
                self.stream_ptr(),
            )
        })
    }

    /// `dense_w16` on an FP8 plane (e4m3 bytes widened to f16, per-row f32
    /// scale) against f16 activations - see [`Self::dense_w16_bf16`].
    #[allow(clippy::too_many_arguments)]
    pub fn dense_w16_e4m3(
        &self,
        w: &super::F8RowPlane,
        in_dim: usize,
        out_dim: usize,
        x: &CudaSlice<f32>,
        x16: &mut CudaSlice<half::f16>,
        y: &mut CudaSlice<f32>,
        y_stride: usize,
        rows: usize,
    ) -> Result<(), GpuError> {
        debug_assert_eq!(w.data.len(), in_dim * out_dim);
        let pre = rows > DENSE_W16_PRECAST_ROWS;
        if pre {
            self.convert_f32_f16(x, x16, rows * in_dim)?;
        }
        let (wp, _g1) = w.data.device_ptr(&self.stream);
        let (sp, _g2) = w.scale.device_ptr(&self.stream);
        let xg = if pre {
            x16.device_ptr(&self.stream).0
        } else {
            x.device_ptr(&self.stream).0
        };
        self.dense_w16_raw(wp, sp, xg, y, in_dim, out_dim, y_stride, rows, 1, pre)
    }

    /// `dense_w16` on an f16 plane (house dims `[in, out]`, rows contiguous)
    /// against f16 activations already cast (`x16` `[rows][in]`): `y[t][o]`
    /// at row stride `y_stride`, the f16 mma - the same class as
    /// [`Self::matvec_batch_f16`], batch-invariant. Needs a pack whose slot
    /// 682 takes dtype 2.
    pub fn dense_w16_f16(
        &self,
        w: &super::HalfTensor,
        x16: &CudaSlice<half::f16>,
        y: &mut CudaSlice<f32>,
        y_stride: usize,
        rows: usize,
    ) -> Result<(), GpuError> {
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        if w.buf.len() < in_dim * out_dim
            || x16.len() < rows * in_dim
            || y_stride < out_dim
            || (rows > 0 && y.len() < (rows - 1) * y_stride + out_dim)
        {
            return Err(super::error::oob(
                "dense_w16_f16: buffers under the GEMM geometry",
            ));
        }
        let (wp, _g1) = w.buf.device_ptr(&self.stream);
        let (xp, _g2) = x16.device_ptr(&self.stream);
        self.dense_w16_raw(wp, 0, xp, y, in_dim, out_dim, y_stride, rows, 2, true)
    }

    #[allow(clippy::too_many_arguments)]
    fn dense_w16_raw(
        &self,
        w: u64,
        rscale: u64,
        x: u64,
        y: &mut CudaSlice<f32>,
        in_dim: usize,
        out_dim: usize,
        y_stride: usize,
        rows: usize,
        dtype: u32,
        x16: bool,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .dense_w16
            .ok_or(GpuError::MissingOp("dense_w16"))?;
        debug_assert!(y.len() >= (rows - 1) * y_stride + out_dim);
        let (yp, _g) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; the caller's planes hold [out, in] weights and
        // [rows, in] activations, y [rows, y_stride]
        check(unsafe {
            f(
                w as *const _,
                rscale as *const _,
                x as *const _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                rows as u32,
                y_stride as u32,
                dtype,
                x16 as u32,
                self.stream_ptr(),
            )
        })
    }

    /// True when the pack carries the W16 expert pair (680-681) and the slot
    /// combine it folds through.
    pub fn has_nvf4_moe_w16(&self) -> bool {
        self.kernels.nvf4_moe_up_relu2_w16.is_some()
            && self.kernels.nvf4_moe_down_part_w16.is_some()
            && self.kernels.moe_slot_combine.is_some()
    }

    /// The W16 decode class's expert up + relu^2 on TILED planes: the
    /// checkpoint's W4A16 (bf16 activations) on tensor cores, batch-invariant:
    /// a token's activations are the same bits alone or in a verify block.
    /// Routed picks as moe_align's 32-row sorting of k-wide topk rows
    /// (`n_blocks` blocks); the shared expert covers every row. `x` bf16
    /// [rows, in]; `act` bf16 [rows][k * up.ff + sh_up.ff].
    #[allow(clippy::too_many_arguments)]
    pub fn nvf4_moe_up_relu2_w16(
        &self,
        up: &Nvf4MoePlane,
        sh_up: &Nvf4MoePlane,
        sorted_row: &CudaSlice<u32>,
        sorted_slot: &CudaSlice<u32>,
        block_expert: &CudaSlice<u32>,
        x: &CudaSlice<half::bf16>,
        act: &mut CudaSlice<half::bf16>,
        k: usize,
        n_blocks: usize,
        rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .nvf4_moe_up_relu2_w16
            .ok_or(GpuError::MissingOp("nvf4_moe_up_relu2_w16"))?;
        Self::moe_layout_ok(up, Nvf4MoeLayout::Tiled64, "nvf4_moe_up_relu2_w16")?;
        Self::moe_layout_ok(sh_up, Nvf4MoeLayout::Tiled64, "nvf4_moe_up_relu2_w16")?;
        debug_assert_eq!(up.in_dim, sh_up.in_dim);
        debug_assert!(sorted_row.len() >= n_blocks * 32 && sorted_slot.len() >= n_blocks * 32);
        debug_assert!(block_expert.len() >= n_blocks);
        debug_assert!(x.len() >= rows * up.in_dim);
        debug_assert!(act.len() >= rows * (k * up.ff + sh_up.ff));
        let (rdp, _g1) = up.data.device_ptr(&self.stream);
        let (rsp, _g2) = up.scale.device_ptr(&self.stream);
        let (rs2, _g3) = up.scale2.device_ptr(&self.stream);
        let (sdp, _g4) = sh_up.data.device_ptr(&self.stream);
        let (ssp, _g5) = sh_up.scale.device_ptr(&self.stream);
        let (ss2, _g6) = sh_up.scale2.device_ptr(&self.stream);
        let (srp, _g7) = sorted_row.device_ptr(&self.stream);
        let (ssl, _g8) = sorted_slot.device_ptr(&self.stream);
        let (bep, _g9) = block_expert.device_ptr(&self.stream);
        let (xp, _g10) = x.device_ptr(&self.stream);
        let (ap, _g11) = act.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; sizes checked above
        check(unsafe {
            f(
                rdp as *const _,
                rsp as *const _,
                rs2 as *const _,
                sdp as *const _,
                ssp as *const _,
                ss2 as *const _,
                srp as *const _,
                ssl as *const _,
                bep as *const _,
                xp as *const _,
                ap as *mut _,
                up.in_dim as u32,
                up.ff as u32,
                sh_up.ff as u32,
                k as u32,
                n_blocks as u32,
                rows as u32,
                self.stream_ptr(),
            )
        })
    }

    /// The matching W16 down into `part` [rows][k + 1][embd] pre-weighted
    /// partials (slot k = the shared expert) for `moe_slot_combine`. `act`
    /// bf16 from [`Self::nvf4_moe_up_relu2_w16`]; `topk_w` [rows][k].
    #[allow(clippy::too_many_arguments)]
    pub fn nvf4_moe_down_part_w16(
        &self,
        down: &Nvf4MoePlane,
        sh_down: &Nvf4MoePlane,
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
        let f = self
            .kernels
            .nvf4_moe_down_part_w16
            .ok_or(GpuError::MissingOp("nvf4_moe_down_part_w16"))?;
        Self::moe_layout_ok(down, Nvf4MoeLayout::Tiled64, "nvf4_moe_down_part_w16")?;
        Self::moe_layout_ok(sh_down, Nvf4MoeLayout::Tiled64, "nvf4_moe_down_part_w16")?;
        debug_assert_eq!(down.ff, sh_down.ff);
        debug_assert!(sorted_row.len() >= n_blocks * 32 && sorted_slot.len() >= n_blocks * 32);
        debug_assert!(block_expert.len() >= n_blocks);
        debug_assert!(topk_w.len() >= rows * k);
        debug_assert!(act.len() >= rows * (k * down.in_dim + sh_down.in_dim));
        debug_assert!(part.len() >= rows * (k + 1) * down.ff);
        let (rdp, _g1) = down.data.device_ptr(&self.stream);
        let (rsp, _g2) = down.scale.device_ptr(&self.stream);
        let (rs2, _g3) = down.scale2.device_ptr(&self.stream);
        let (sdp, _g4) = sh_down.data.device_ptr(&self.stream);
        let (ssp, _g5) = sh_down.scale.device_ptr(&self.stream);
        let (ss2, _g6) = sh_down.scale2.device_ptr(&self.stream);
        let (srp, _g7) = sorted_row.device_ptr(&self.stream);
        let (ssl, _g8) = sorted_slot.device_ptr(&self.stream);
        let (bep, _g9) = block_expert.device_ptr(&self.stream);
        let (wp, _g10) = topk_w.device_ptr(&self.stream);
        let (ap, _g11) = act.device_ptr(&self.stream);
        let (pp, _g12) = part.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; sizes checked above
        check(unsafe {
            f(
                rdp as *const _,
                rsp as *const _,
                rs2 as *const _,
                sdp as *const _,
                ssp as *const _,
                ss2 as *const _,
                srp as *const _,
                ssl as *const _,
                bep as *const _,
                wp as *const _,
                ap as *const _,
                pp as *mut _,
                down.in_dim as u32,
                sh_down.in_dim as u32,
                down.ff as u32,
                k as u32,
                n_blocks as u32,
                rows as u32,
                self.stream_ptr(),
            )
        })
    }

    /// True when the pack carries the W16 routing front (slot 685).
    pub fn has_moe_route_w16(&self) -> bool {
        self.kernels.moe_route_w16.is_some()
    }

    /// Tickets [`Self::moe_route_w16`] needs for `rows` rows (1 + one per
    /// two-row group), zeroed once at allocation - the kernel leaves them
    /// zero.
    pub fn moe_route_w16_tickets(rows: usize) -> usize {
        1 + rows.div_ceil(2)
    }

    /// The W16 class's routing front in one launch: the router matvec over
    /// `router` (the unfused `matvec_f32_batch`'s bits into `logits`), the
    /// sigmoid top-k (`moe_topk_sigmoid_batch`'s picks into `out_idx` /
    /// `out_w`), the rows' bf16 activations into `x16` (`convert_f32_bf16`'s
    /// bits) and moe_align's 32-row sorting of the picks (`n_blocks` blocks) -
    /// the four launches it replaces, bit for bit.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_route_w16(
        &self,
        router: &DeviceTensor,
        x: &CudaSlice<f32>,
        logits: &mut CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        routed_scale: f32,
        k: usize,
        x16: &mut CudaSlice<half::bf16>,
        out_idx: &mut CudaSlice<u32>,
        out_w: &mut CudaSlice<f32>,
        sorted_row: &mut CudaSlice<u32>,
        sorted_slot: &mut CudaSlice<u32>,
        block_expert: &mut CudaSlice<u32>,
        n_blocks: usize,
        rows: usize,
        tickets: &mut CudaSlice<u32>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .moe_route_w16
            .ok_or(GpuError::MissingOp("moe_route_w16"))?;
        let (in_dim, n_expert) = (router.dims[0], router.dims[1]);
        debug_assert!(x.len() >= rows * in_dim && x16.len() >= rows * in_dim);
        debug_assert!(logits.len() >= rows * n_expert);
        debug_assert!(out_idx.len() >= rows * k && out_w.len() >= rows * k);
        debug_assert!(sorted_row.len() >= n_blocks * 32 && sorted_slot.len() >= n_blocks * 32);
        debug_assert!(block_expert.len() >= n_blocks);
        debug_assert!(tickets.len() >= Self::moe_route_w16_tickets(rows));
        let (wp, _g1) = router.buf.device_ptr(&self.stream);
        let (xp, _g2) = x.device_ptr(&self.stream);
        let (lp, _g3) = logits.device_ptr_mut(&self.stream);
        let (bp, _g4) = bias.device_ptr(&self.stream);
        let (x16p, _g5) = x16.device_ptr_mut(&self.stream);
        let (ip, _g6) = out_idx.device_ptr_mut(&self.stream);
        let (owp, _g7) = out_w.device_ptr_mut(&self.stream);
        let (srp, _g8) = sorted_row.device_ptr_mut(&self.stream);
        let (ssp, _g9) = sorted_slot.device_ptr_mut(&self.stream);
        let (bep, _g10) = block_expert.device_ptr_mut(&self.stream);
        let (tp, _g11) = tickets.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; sizes checked above
        check(unsafe {
            f(
                wp as *const _,
                xp as *const _,
                lp as *mut _,
                bp as *const _,
                routed_scale,
                in_dim as u32,
                n_expert as u32,
                k as u32,
                x16p as *mut _,
                ip as *mut _,
                owp as *mut _,
                srp as *mut _,
                ssp as *mut _,
                bep as *mut _,
                n_blocks as u32,
                rows as u32,
                tp as *mut _,
                self.stream_ptr(),
            )
        })
    }
}
