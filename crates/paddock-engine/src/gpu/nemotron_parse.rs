//! Nemotron 3.5 Parse ops (slots 890-892, `packs/cuda/src/nemotron_parse.cuh`)
//! plus the one row-offset entry the packed tower needs into an existing slot.
//! The graph is `gpu_model::nemotron_parse`.
//!
//! - 890 the rectangular u8 patch stem (the processor's pixel math on device),
//! - 891 the checkpoint's repetition penalty + the greedy pick,
//! - 892 the seen-id bitmaps that penalty reads,
//! - [`GpuExecutor::vision_attn_h_rows`]: slot 620 over one page of a packed
//!   tower pass - pages of different sizes share every row-wise op of a pass
//!   but must attend only within themselves.
//!
//! Every buffer is checked against the geometry before a launch.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::f16;

use super::error::*;
use super::*;

impl GpuExecutor {
    /// Whether the pack carries this family's three kernels.
    pub fn has_nemotron_parse(&self) -> bool {
        let k = &self.kernels;
        k.np_patch_u8.is_some() && k.np_pen_argmax.is_some() && k.np_seen_or.is_some()
    }

    /// Slot 890: one page's RGB8 canvas (`h` x `w`, both multiples of 16,
    /// starting `canvas_off` bytes into `canvas`) to its normalized f16 patch
    /// rows, landing at row `row_off` of `out` (`[rows][768]`).
    #[allow(clippy::too_many_arguments)]
    pub fn np_patch_u8(
        &self,
        canvas: &CudaSlice<u8>,
        canvas_off: usize,
        out: &mut CudaSlice<f16>,
        row_off: usize,
        h: usize,
        w: usize,
        mean: [f32; 3],
        std: [f32; 3],
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .np_patch_u8
            .ok_or(GpuError::MissingOp("np_patch_u8"))?;
        if !h.is_multiple_of(16) || !w.is_multiple_of(16) {
            return Err(oob("np_patch_u8: canvas sides must be multiples of 16"));
        }
        let n = (h / 16) * (w / 16);
        if canvas.len() < canvas_off + 3 * h * w || out.len() < (row_off + n) * 768 {
            return Err(oob("np_patch_u8: buffers under the page geometry"));
        }
        let (cp, _g1) = canvas.device_ptr(&self.stream);
        let (op, _g2) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 890); bounds checked above, offsets in bytes
        check(unsafe {
            f(
                (cp + canvas_off as u64) as *const _,
                (op + (row_off * 768 * 2) as u64) as *mut _,
                h as u32,
                w as u32,
                mean[0],
                mean[1],
                mean[2],
                std[0],
                std[1],
                std[2],
                self.stream_ptr(),
            )
        })
    }

    /// Slot 891: penalize and pick. `logits` `[rows][vocab]`; row `i` reads
    /// slot `row_slot[i]`'s bitmap (`words` u32 a slot) and its own
    /// `n_extra` ids at `extra[i * n_extra..]` (`u32::MAX` = unused).
    #[allow(clippy::too_many_arguments)]
    pub fn np_pen_argmax(
        &self,
        logits: &mut CudaSlice<f32>,
        seen: &CudaSlice<u32>,
        row_slot: &CudaSlice<u32>,
        extra: &CudaSlice<u32>,
        n_extra: usize,
        penalty: f32,
        vocab: usize,
        words: usize,
        rows: usize,
        write_back: bool,
        out_ids: &mut CudaSlice<u32>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .np_pen_argmax
            .ok_or(GpuError::MissingOp("np_pen_argmax"))?;
        if logits.len() < rows * vocab
            || row_slot.len() < rows
            || extra.len() < rows * n_extra
            || out_ids.len() < rows
            || words * 32 < vocab
        {
            return Err(oob("np_pen_argmax: buffers under the row geometry"));
        }
        let (lp, _g1) = logits.device_ptr_mut(&self.stream);
        let (sp, _g2) = seen.device_ptr(&self.stream);
        let (rp, _g3) = row_slot.device_ptr(&self.stream);
        let (ep, _g4) = extra.device_ptr(&self.stream);
        let (op, _g5) = out_ids.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 891); bounds checked above (the bitmap's
        // slot extent is the caller's slot table, sized at enable)
        check(unsafe {
            f(
                lp as *mut _,
                sp as *const _,
                rp as *const _,
                ep as *const _,
                n_extra as u32,
                penalty,
                vocab as u32,
                words as u32,
                rows as u32,
                write_back as u32,
                op as *mut _,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 892: mark `toks[i]` seen in slot `slots[i]`'s bitmap, i < n.
    pub fn np_seen_or(
        &self,
        seen: &mut CudaSlice<u32>,
        slots: &CudaSlice<u32>,
        toks: &CudaSlice<u32>,
        n: usize,
        words: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .np_seen_or
            .ok_or(GpuError::MissingOp("np_seen_or"))?;
        if slots.len() < n || toks.len() < n {
            return Err(oob("np_seen_or: buffers under n"));
        }
        let (bp, _g1) = seen.device_ptr_mut(&self.stream);
        let (sp, _g2) = slots.device_ptr(&self.stream);
        let (tp, _g3) = toks.device_ptr(&self.stream);
        // SAFETY: ABI contract (slot 892); the kernel drops ids past the bitmap
        check(unsafe {
            f(
                bp as *mut _,
                sp as *const _,
                tp as *const _,
                n as u32,
                words as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 620 over rows `[row0, row0 + n)` of packed `[row][head][dim]`
    /// half planes: self-attention of one page of a tower pass, `q` already
    /// carrying 1/sqrt(head_dim) (the split folds it in), `out` landing at
    /// the same rows. Same kernel and bits as [`Self::vision_attn_h`] on a
    /// one-picture group.
    #[allow(clippy::too_many_arguments)]
    pub fn vision_attn_h_rows(
        &self,
        q: &CudaSlice<f16>,
        k: &CudaSlice<f16>,
        v: &CudaSlice<f16>,
        out: &mut CudaSlice<f16>,
        row0: usize,
        n: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .vision_attn_h
            .ok_or(GpuError::MissingOp("vision_attn_h"))?;
        let row = n_heads * head_dim;
        let end = (row0 + n) * row;
        if q.len() < end || k.len() < end || v.len() < end || out.len() < end {
            return Err(oob("vision_attn_h_rows: buffers under the page rows"));
        }
        let off = (row0 * row * 2) as u64;
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = k.device_ptr(&self.stream);
        let (vp, _g3) = v.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 620) on one group whose planes start at
        // row0; bounds checked above
        check(unsafe {
            f(
                (qp + off) as *const _,
                (kp + off) as *const _,
                (vp + off) as *const _,
                (op + off) as *mut _,
                n as u32,
                n as u32,
                n_heads as u32,
                head_dim as u32,
                1,
                self.stream_ptr(),
            )
        })
    }

    /// Whether the pack carries the exact-hd72 tower attention (slot 893).
    pub fn has_np_vit_attn(&self) -> bool {
        self.kernels.np_vit_attn.is_some()
    }

    /// Slot 893 over rows `[row0, row0 + n)` of one page, reading q, k and v
    /// straight off the fused qkv GEMM landing (`[row][3][heads][72]`):
    /// q's bias + `qscale` (log2(e)/sqrt(72)) applied at its load, v's bias
    /// in the epilogue, k as landed. `out` `[row][heads][72]` at the same rows.
    #[allow(clippy::too_many_arguments)]
    pub fn np_vit_attn_qkv(
        &self,
        qkv: &CudaSlice<f16>,
        bq: &CudaSlice<f32>,
        bv: &CudaSlice<f32>,
        out: &mut CudaSlice<f16>,
        row0: usize,
        n: usize,
        n_heads: usize,
        qscale: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .np_vit_attn
            .ok_or(GpuError::MissingOp("np_vit_attn"))?;
        let d = n_heads * 72;
        if qkv.len() < (row0 + n) * 3 * d
            || out.len() < (row0 + n) * d
            || bq.len() < d
            || bv.len() < d
        {
            return Err(oob("np_vit_attn: buffers under the page rows"));
        }
        let (qp, _g1) = qkv.device_ptr(&self.stream);
        let (bqp, _g2) = bq.device_ptr(&self.stream);
        let (bvp, _g3) = bv.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        let base = qp + (row0 * 3 * d * 2) as u64;
        // SAFETY: ABI contract (slot 893) on one page starting at row0;
        // bounds checked above, offsets in bytes
        check(unsafe {
            f(
                base as *const _,
                (base + (d * 2) as u64) as *const _,
                (base + (2 * d * 2) as u64) as *const _,
                (op + (row0 * d * 2) as u64) as *mut _,
                n as u32,
                n_heads as u32,
                (3 * d) as u32,
                bqp as *const _,
                bvp as *const _,
                qscale,
                self.stream_ptr(),
            )
        })
    }

    /// Whether [`Self::np_head8`] and the e4m3 W16 GEMM that reads its plane
    /// are both in the pack.
    pub fn has_np_head8(&self) -> bool {
        self.kernels.quantize_e4m3_row.is_some() && self.has_dense_w16()
    }

    /// An f32 `[rows][k]` table as an e4m3 plane with one power-of-two f32
    /// scale a row (the shared row quantizer), the scales then multiplied by
    /// `rescale` - exact when it is a power of two, so a table stored at a
    /// pow2 multiple of the weights it holds quantizes to the weights' own
    /// bytes.
    pub fn np_head8(
        &self,
        table: &CudaSlice<f32>,
        rows: usize,
        k: usize,
        rescale: f32,
    ) -> Result<F8RowPlane, GpuError> {
        let f = self
            .kernels
            .quantize_e4m3_row
            .ok_or(GpuError::MissingOp("quantize_e4m3_row"))?;
        if table.len() < rows * k || !k.is_multiple_of(32) {
            return Err(oob(
                "np_head8: table under [rows][k] or k not a multiple of 32",
            ));
        }
        let mut data = self.alloc_u8(rows * k)?;
        let mut scale: CudaSlice<f32> = self.stream.alloc_zeros(rows).map_err(from_driver)?;
        {
            let (xp, _g1) = table.device_ptr(&self.stream);
            let (qp, _g2) = data.device_ptr_mut(&self.stream);
            let (sp, _g3) = scale.device_ptr_mut(&self.stream);
            // SAFETY: ABI contract (quantize_e4m3_row); sizes checked above
            check(unsafe {
                f(
                    xp as *const _,
                    qp as *mut _,
                    sp as *mut _,
                    k as u32,
                    rows as u32,
                    self.stream_ptr(),
                )
            })?;
        }
        self.scale(&mut scale, rescale, rows)?;
        Ok(F8RowPlane { data, scale })
    }

    /// Whether the pack carries the always-ring f32 GEMM (slot 894) - and
    /// with it, from the same build, the rows partial at head_dim 64.
    pub fn has_f16_gemm_ring(&self) -> bool {
        self.kernels.f16_gemm_ring.is_some()
    }

    /// Slot 894 over rows `[x_row0, x_row0 + rows)` of `x16` (`[*][in]`)
    /// into rows `[y_row0, ..)` of `y` (`[*][out]` f32): `y = W x`, or with
    /// `bias` `y += W x + bias` (the residual stream). The unsplit rings
    /// only, so a row's bits do not depend on `rows` or its neighbours.
    #[allow(clippy::too_many_arguments)]
    pub fn f16_gemm_ring_rows(
        &self,
        w: &HalfTensor,
        x16: &CudaSlice<f16>,
        x_row0: usize,
        y: &mut CudaSlice<f32>,
        y_row0: usize,
        rows: usize,
        bias: Option<&CudaSlice<f32>>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .f16_gemm_ring
            .ok_or(GpuError::MissingOp("f16_gemm_ring"))?;
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        if rows == 0 {
            return Ok(());
        }
        if w.buf.len() < in_dim * out_dim
            || x16.len() < (x_row0 + rows) * in_dim
            || y.len() < (y_row0 + rows) * out_dim
            || bias.is_some_and(|b| b.len() < out_dim)
        {
            return Err(oob("f16_gemm_ring: buffers under the GEMM geometry"));
        }
        let (wp, _g1) = w.buf.device_ptr(&self.stream);
        let (xp, _g2) = x16.device_ptr(&self.stream);
        let (yp, _g3) = y.device_ptr_mut(&self.stream);
        let bp = bias.map(|b| b.device_ptr(&self.stream));
        // SAFETY: ABI contract (slot 894); bounds checked above, offsets in bytes
        check(unsafe {
            f(
                wp as *const _,
                (xp + (x_row0 * in_dim * 2) as u64) as *const _,
                (yp + (y_row0 * out_dim * 4) as u64) as *mut _,
                bp.as_ref()
                    .map_or(std::ptr::null(), |(p, _)| *p as *const _),
                in_dim as u32,
                out_dim as u32,
                rows as u32,
                bias.is_some() as u32,
                self.stream_ptr(),
            )
        })
    }

    /// [`Self::matvec_batch_f16_h_gelu`] (slot 624) at row offsets: the f16
    /// landing with bias + exact GELU folded in. Its route is ring-only
    /// (no GEMV band, no K-split), so a row's bits are row-count-invariant.
    #[allow(clippy::too_many_arguments)]
    pub fn f16_gemm_h_gelu_rows(
        &self,
        w: &HalfTensor,
        x16: &CudaSlice<f16>,
        x_row0: usize,
        y16: &mut CudaSlice<f16>,
        y_row0: usize,
        bias: &CudaSlice<f32>,
        rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .f16_gemm_h_gelu
            .ok_or(GpuError::MissingOp("f16_gemm_h_gelu"))?;
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        if rows == 0 {
            return Ok(());
        }
        if !in_dim.is_multiple_of(8)
            || w.buf.len() < in_dim * out_dim
            || x16.len() < (x_row0 + rows) * in_dim
            || y16.len() < (y_row0 + rows) * out_dim
            || bias.len() < out_dim
        {
            return Err(oob("f16_gemm_h_gelu_rows: buffers under the GEMM geometry"));
        }
        let (wp, _g1) = w.buf.device_ptr(&self.stream);
        let (xp, _g2) = x16.device_ptr(&self.stream);
        let (bp, _g3) = bias.device_ptr(&self.stream);
        let (yp, _g4) = y16.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 624); bounds checked above, offsets in bytes
        check(unsafe {
            f(
                wp as *const _,
                (xp + (x_row0 * in_dim * 2) as u64) as *const _,
                (yp + (y_row0 * out_dim * 2) as u64) as *mut _,
                bp as *const _,
                in_dim as u32,
                out_dim as u32,
                rows as u32,
                self.stream_ptr(),
            )
        })
    }
}
