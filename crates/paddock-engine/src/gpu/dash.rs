//! Unlimited-OCR-2's SR-SWA kernels (attn/dash, slots 935-939): the page-2D
//! rope, the router's block summaries and route / select, the sparse attention
//! over the paged f16 pool, and the page-count guard. The model side
//! (`gpu_model/deepseek_ocr/dash.rs`) owns the per-slot state these read.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use super::error::check;
use super::{GpuError, GpuExecutor};

/// Head dim == router index dim this family serves.
pub const DASH_D: usize = 128;
/// Keys a router block.
pub const DASH_BLOCK: usize = 32;
/// A row's selected-block capacity in the route outputs.
pub const DASH_KCAP: usize = 16;

fn need(ok: bool, what: &str) -> Result<(), GpuError> {
    if ok {
        Ok(())
    } else {
        Err(GpuError::Unsupported(format!("dash: {what}")))
    }
}

impl GpuExecutor {
    /// All five slots are in the pack.
    pub fn has_dash(&self) -> bool {
        self.kernels.dash_rope2d.is_some()
            && self.kernels.dash_summary.is_some()
            && self.kernels.dash_route.is_some()
            && self.kernels.dash_attn_f16.is_some()
            && self.kernels.dash_guard.is_some()
    }

    /// Slot 935: page-2D rope in place on q / k (`[rows][n_head * 128]`) and the
    /// router q / k (row `rstride` apart, router k at `rk_off` floats past q),
    /// the router planes normed first (`rqw` / `rkw` = 1 + w), from `coords`
    /// `[rows][2]` (tok, parity).
    #[allow(clippy::too_many_arguments)]
    pub fn dash_rope2d(
        &self,
        q: &mut CudaSlice<f32>,
        k: &mut CudaSlice<f32>,
        r: &mut CudaSlice<f32>,
        rk_off: usize,
        rstride: usize,
        rqw: &CudaSlice<f32>,
        rkw: &CudaSlice<f32>,
        eps: f32,
        coords: &CudaSlice<u32>,
        inv_tok: &CudaSlice<f32>,
        rows: usize,
        n_head: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .dash_rope2d
            .ok_or(GpuError::MissingOp("dash_rope2d"))?;
        if rows == 0 {
            return Ok(());
        }
        need(
            q.len() >= rows * n_head * DASH_D
                && k.len() >= rows * n_head * DASH_D
                && r.len() >= (rows - 1) * rstride + rk_off + DASH_D,
            "rope2d planes too small",
        )?;
        need(
            coords.len() >= 2 * rows && inv_tok.len() >= DASH_D / 2 - 1,
            "rope2d tables",
        )?;
        let (qp, _g1) = q.device_ptr_mut(&self.stream);
        let (kp, _g2) = k.device_ptr_mut(&self.stream);
        let (rp, _g3) = r.device_ptr_mut(&self.stream);
        let (cp, _g4) = coords.device_ptr(&self.stream);
        let (ip, _g5) = inv_tok.device_ptr(&self.stream);
        let (qw, _g6) = rqw.device_ptr(&self.stream);
        let (kw, _g7) = rkw.device_ptr(&self.stream);
        need(
            rqw.len() >= DASH_D && rkw.len() >= DASH_D,
            "rope2d norm weights",
        )?;
        // SAFETY: ABI contract (slot 935); sizes checked above
        check(unsafe {
            f(
                qp as *mut _,
                kp as *mut _,
                rp as *mut _,
                (rp as usize + rk_off * 4) as *mut _,
                qw as *const _,
                kw as *const _,
                cp as *const _,
                ip as *const _,
                rows as u32,
                n_head as u32,
                rstride as u32,
                eps,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 936: block summaries for `n_items` items (6 u32 each).
    #[allow(clippy::too_many_arguments)]
    pub fn dash_summary(
        &self,
        rk: &CudaSlice<f32>,
        rk_off: usize,
        rstride: usize,
        staging: &mut CudaSlice<f32>,
        kbar: &mut CudaSlice<f32>,
        head_cls: &CudaSlice<f32>,
        items: &CudaSlice<u32>,
        n_items: usize,
        kbar_blocks: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .dash_summary
            .ok_or(GpuError::MissingOp("dash_summary"))?;
        need(
            items.len() >= 6 * n_items && head_cls.len() >= DASH_D,
            "summary inputs",
        )?;
        let (rp, _g1) = rk.device_ptr(&self.stream);
        let (sp, _g2) = staging.device_ptr_mut(&self.stream);
        let (kp, _g3) = kbar.device_ptr_mut(&self.stream);
        let (hp, _g4) = head_cls.device_ptr(&self.stream);
        let (ip, _g5) = items.device_ptr(&self.stream);
        // SAFETY: ABI contract (slot 936); the items were built against these
        // planes' shapes by the caller
        check(unsafe {
            f(
                (rp as usize + rk_off * 4) as *const _,
                rstride as u32,
                sp as *mut _,
                kp as *mut _,
                hp as *const _,
                ip as *const _,
                n_items as u32,
                kbar_blocks as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 937: route / select over `rows` rows (`rinfo` 6 u32 a row).
    #[allow(clippy::too_many_arguments)]
    pub fn dash_route(
        &self,
        rq: &CudaSlice<f32>,
        rstride: usize,
        kbar: &CudaSlice<f32>,
        kbar_blocks: usize,
        rinfo: &CudaSlice<u32>,
        out_blk: &mut CudaSlice<i32>,
        out_bias: &mut CudaSlice<f32>,
        out_n: &mut CudaSlice<u32>,
        rows: usize,
        local_blocks: usize,
        m: usize,
        k: usize,
        score_scale: f32,
        floor: f32,
        inv_sigma: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .dash_route
            .ok_or(GpuError::MissingOp("dash_route"))?;
        need(
            rinfo.len() >= 6 * rows
                && out_blk.len() >= DASH_KCAP * rows
                && out_bias.len() >= DASH_KCAP * rows
                && out_n.len() >= rows,
            "route buffers",
        )?;
        let (qp, _g1) = rq.device_ptr(&self.stream);
        let (kp, _g2) = kbar.device_ptr(&self.stream);
        let (ip, _g3) = rinfo.device_ptr(&self.stream);
        let (bp, _g4) = out_blk.device_ptr_mut(&self.stream);
        let (sp, _g5) = out_bias.device_ptr_mut(&self.stream);
        let (np, _g6) = out_n.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 937); sizes checked above
        check(unsafe {
            f(
                qp as *const _,
                rstride as u32,
                kp as *const _,
                kbar_blocks as u32,
                ip as *const _,
                bp as *mut _,
                sp as *mut _,
                np as *mut _,
                rows as u32,
                local_blocks as u32,
                m as u32,
                k as u32,
                score_scale,
                floor,
                inv_sigma,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 938: the sparse attention (`ainfo` 4 u32 a row) over the f16 pool.
    #[allow(clippy::too_many_arguments)]
    pub fn dash_attn_f16(
        &self,
        q: &CudaSlice<f32>,
        pool_k: &CudaSlice<u8>,
        pool_v: &CudaSlice<u8>,
        out: &mut CudaSlice<f32>,
        block_tables: &CudaSlice<u32>,
        bps: usize,
        ainfo: &CudaSlice<u32>,
        sel_blk: &CudaSlice<i32>,
        sel_bias: &CudaSlice<f32>,
        sel_n: &CudaSlice<u32>,
        rows: usize,
        n_head: usize,
        scale: f32,
        window: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .dash_attn_f16
            .ok_or(GpuError::MissingOp("dash_attn_f16"))?;
        need(
            q.len() >= rows * n_head * DASH_D
                && out.len() >= rows * n_head * DASH_D
                && ainfo.len() >= 4 * rows
                && sel_n.len() >= rows,
            "attn buffers",
        )?;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = pool_k.device_ptr(&self.stream);
        let (vp, _g3) = pool_v.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        let (tp, _g5) = block_tables.device_ptr(&self.stream);
        let (ap, _g6) = ainfo.device_ptr(&self.stream);
        let (bp, _g7) = sel_blk.device_ptr(&self.stream);
        let (sp, _g8) = sel_bias.device_ptr(&self.stream);
        let (np, _g9) = sel_n.device_ptr(&self.stream);
        // SAFETY: ABI contract (slot 938); every pool row the rows can address
        // is mapped by their block tables (the caller's ensure_rows)
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                op as *mut _,
                tp as *const _,
                bps as u32,
                ap as *const _,
                bp as *const _,
                sp as *const _,
                np as *const _,
                rows as u32,
                n_head as u32,
                scale,
                window as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 939: the page-count guard on `[rows][vocab]` logits.
    pub fn dash_guard(
        &self,
        logits: &mut CudaSlice<f32>,
        mode: &CudaSlice<u32>,
        rows: usize,
        vocab: usize,
        eos: u32,
        page: u32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .dash_guard
            .ok_or(GpuError::MissingOp("dash_guard"))?;
        need(
            logits.len() >= rows * vocab && mode.len() >= rows,
            "guard buffers",
        )?;
        let (lp, _g1) = logits.device_ptr_mut(&self.stream);
        let (mp, _g2) = mode.device_ptr(&self.stream);
        // SAFETY: ABI contract (slot 939); sizes checked above
        check(unsafe {
            f(
                lp as *mut _,
                mp as *const _,
                rows as u32,
                vocab as u32,
                eos,
                page,
                self.stream_ptr(),
            )
        })
    }
}
