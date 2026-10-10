//! PP-DocLayoutV3's glue ops (slots 836-843, `packs/cuda/src/doclayout.cuh`):
//! the SiLU landing, the input rescale, im2row, depthwise conv, the stem pool,
//! channel concat, 2x upsample and adds. Every plane is NHWC halves.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::f16;

use super::error::*;
use super::*;

impl GpuExecutor {
    /// Whether the pack carries PP-DocLayoutV3's glue (slots 836-843).
    pub fn has_doclayout(&self) -> bool {
        let k = &self.kernels;
        k.f16_gemm_h_silu.is_some()
            && k.dl_u8_to_h.is_some()
            && k.dl_im2row_h.is_some()
            && k.dl_dwconv_h.is_some()
            && k.dl_maxpool2_h.is_some()
            && k.dl_concat_h.is_some()
            && k.dl_up2_h.is_some()
            && k.dl_add_h.is_some()
            && k.dl_rowscale_h.is_some()
            && k.dl_gather_rows.is_some()
            && k.dl_mask_ref.is_some()
            && k.dl_msda.is_some()
            && k.dl_ref_step.is_some()
            && k.dl_order_votes.is_some()
    }

    /// `y16 = silu(W x + bias)` at f16 - the landing with the bias (optional)
    /// and SiLU in the epilogue, before the one round.
    pub fn matvec_batch_f16_h_silu(
        &self,
        w: &HalfTensor,
        x16: &CudaSlice<f16>,
        y16: &mut CudaSlice<f16>,
        bias: Option<&CudaSlice<f32>>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .f16_gemm_h_silu
            .ok_or(GpuError::MissingOp("f16_gemm_h_silu"))?;
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        if !in_dim.is_multiple_of(8) {
            return Err(oob("f16_gemm_h_silu: in_dim must be a multiple of 8"));
        }
        if w.buf.len() < in_dim * out_dim
            || x16.len() < batch * in_dim
            || y16.len() < batch * out_dim
            || bias.is_some_and(|b| b.len() < out_dim)
        {
            return Err(oob("f16_gemm_h_silu: buffers under the GEMM geometry"));
        }
        super::basic_ops::gemm_census("B-gemm-f16-h-silu", in_dim, out_dim, batch);
        let (wp, _g1) = w.buf.device_ptr(&self.stream);
        let (xp, _g2) = x16.device_ptr(&self.stream);
        let bp = bias.map(|b| b.device_ptr(&self.stream));
        let (yp, _g4) = y16.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 836); bounds checked above, null bias = none
        check(unsafe {
            f(
                wp as *const _,
                xp as *const _,
                yp as *mut _,
                bp.as_ref().map_or(0, |(p, _)| *p) as *const _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// u8 -> f16 / 255 (`n` values).
    pub fn dl_u8_to_h(
        &self,
        src: &CudaSlice<u8>,
        dst: &mut CudaSlice<f16>,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .dl_u8_to_h
            .ok_or(GpuError::MissingOp("dl_u8_to_h"))?;
        if src.len() < n || dst.len() < n {
            return Err(oob("dl_u8_to_h: buffers under n"));
        }
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (dp, _g2) = dst.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 837); bounds checked above
        check(unsafe { f(sp as *const _, dp as *mut _, n as u32, self.stream_ptr()) })
    }

    /// k x k im2row of an `ih x iw x c` plane into `[oh * ow][kpad]` halves,
    /// tap order (ky, kx, c); `pad` is (top, left).
    #[allow(clippy::too_many_arguments)]
    pub fn dl_im2row_h(
        &self,
        src: &CudaSlice<f16>,
        dst: &mut CudaSlice<f16>,
        (ih, iw, c): (usize, usize, usize),
        (oh, ow): (usize, usize),
        (kh, kw): (usize, usize),
        stride: usize,
        pad: (i32, i32),
        kpad: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .dl_im2row_h
            .ok_or(GpuError::MissingOp("dl_im2row_h"))?;
        if src.len() < ih * iw * c || dst.len() < oh * ow * kpad || kpad < kh * kw * c {
            return Err(oob("dl_im2row_h: buffers under the geometry"));
        }
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (dp, _g2) = dst.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 838); bounds checked above
        check(unsafe {
            f(
                sp as *const _,
                dp as *mut _,
                ih as u32,
                iw as u32,
                c as u32,
                oh as u32,
                ow as u32,
                kh as u32,
                kw as u32,
                stride as u32,
                pad.0,
                pad.1,
                kpad as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Depthwise k x k conv (padding (k - 1) / 2), f32 `[c][k * k]` weights
    /// and bias, act 0 none / 1 ReLU.
    #[allow(clippy::too_many_arguments)]
    pub fn dl_dwconv_h(
        &self,
        src: &CudaSlice<f16>,
        dst: &mut CudaSlice<f16>,
        w: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        (ih, iw, c): (usize, usize, usize),
        (oh, ow): (usize, usize),
        k: usize,
        stride: usize,
        act: u32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .dl_dwconv_h
            .ok_or(GpuError::MissingOp("dl_dwconv_h"))?;
        if src.len() < ih * iw * c || dst.len() < oh * ow * c || w.len() < c * k * k || b.len() < c
        {
            return Err(oob("dl_dwconv_h: buffers under the geometry"));
        }
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (dp, _g2) = dst.device_ptr_mut(&self.stream);
        let (wp, _g3) = w.device_ptr(&self.stream);
        let (bp, _g4) = b.device_ptr(&self.stream);
        // SAFETY: ABI contract (slot 839); bounds checked above
        check(unsafe {
            f(
                sp as *const _,
                dp as *mut _,
                wp as *const _,
                bp as *const _,
                ih as u32,
                iw as u32,
                c as u32,
                oh as u32,
                ow as u32,
                k as u32,
                stride as u32,
                act,
                self.stream_ptr(),
            )
        })
    }

    /// 2 x 2 / stride 1 max pool, one zero row / column past the bottom /
    /// right edge (the output keeps `h x w`).
    pub fn dl_maxpool2_h(
        &self,
        src: &CudaSlice<f16>,
        dst: &mut CudaSlice<f16>,
        (h, w, c): (usize, usize, usize),
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .dl_maxpool2_h
            .ok_or(GpuError::MissingOp("dl_maxpool2_h"))?;
        if src.len() < h * w * c || dst.len() < h * w * c {
            return Err(oob("dl_maxpool2_h: buffers under the geometry"));
        }
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (dp, _g2) = dst.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 840); bounds checked above
        check(unsafe {
            f(
                sp as *const _,
                dp as *mut _,
                h as u32,
                w as u32,
                c as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Copy `[rows][c]` halves into `[rows][ctot]` at channel `off`.
    pub fn dl_concat_h(
        &self,
        src: &CudaSlice<f16>,
        dst: &mut CudaSlice<f16>,
        rows: usize,
        c: usize,
        (ctot, off): (usize, usize),
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .dl_concat_h
            .ok_or(GpuError::MissingOp("dl_concat_h"))?;
        if src.len() < rows * c || dst.len() < rows * ctot || off + c > ctot {
            return Err(oob("dl_concat_h: buffers under the geometry"));
        }
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (dp, _g2) = dst.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 841); bounds checked above
        check(unsafe {
            f(
                sp as *const _,
                dp as *mut _,
                rows as u32,
                c as u32,
                ctot as u32,
                off as u32,
                self.stream_ptr(),
            )
        })
    }

    /// 2x upsample of an `h x w x c` plane - `bilinear` at align_corners
    /// False, else nearest - with `add` (the output's shape) added on.
    pub fn dl_up2_h(
        &self,
        src: &CudaSlice<f16>,
        dst: &mut CudaSlice<f16>,
        add: Option<&CudaSlice<f16>>,
        (h, w, c): (usize, usize, usize),
        bilinear: bool,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .dl_up2_h
            .ok_or(GpuError::MissingOp("dl_up2_h"))?;
        let n = 4 * h * w * c;
        if src.len() < h * w * c || dst.len() < n || add.is_some_and(|a| a.len() < n) {
            return Err(oob("dl_up2_h: buffers under the geometry"));
        }
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (dp, _g2) = dst.device_ptr_mut(&self.stream);
        let ap = add.map(|a| a.device_ptr(&self.stream));
        // SAFETY: ABI contract (slot 842); bounds checked above, null add = none
        check(unsafe {
            f(
                sp as *const _,
                dp as *mut _,
                ap.as_ref().map_or(0, |(p, _)| *p) as *const _,
                h as u32,
                w as u32,
                c as u32,
                u32::from(bilinear),
                self.stream_ptr(),
            )
        })
    }

    /// `dst = act(a + b)` over `n` halves, act 0 none / 1 ReLU / 2 SiLU.
    pub fn dl_add_h(
        &self,
        a: &CudaSlice<f16>,
        b: &CudaSlice<f16>,
        dst: &mut CudaSlice<f16>,
        n: usize,
        act: u32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .dl_add_h
            .ok_or(GpuError::MissingOp("dl_add_h"))?;
        if a.len() < n || b.len() < n || dst.len() < n {
            return Err(oob("dl_add_h: buffers under n"));
        }
        let (ap, _g1) = a.device_ptr(&self.stream);
        let (bp, _g2) = b.device_ptr(&self.stream);
        let (dp, _g3) = dst.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 843); bounds checked above
        check(unsafe {
            f(
                ap as *const _,
                bp as *const _,
                dp as *mut _,
                n as u64,
                act,
                self.stream_ptr(),
            )
        })
    }

    /// `x[r][:] *= scale[r]` over a `[rows][cols]` half plane.
    pub fn dl_rowscale_h(
        &self,
        x: &mut CudaSlice<f16>,
        scale: &CudaSlice<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .dl_rowscale_h
            .ok_or(GpuError::MissingOp("dl_rowscale_h"))?;
        if x.len() < rows * cols || scale.len() < rows {
            return Err(oob("dl_rowscale_h: buffers under the geometry"));
        }
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (sp, _g2) = scale.device_ptr(&self.stream);
        // SAFETY: ABI contract (slot 844); bounds checked above
        check(unsafe {
            f(
                xp as *mut _,
                sp as *const _,
                rows as u32,
                cols as u32,
                self.stream_ptr(),
            )
        })
    }

    /// `dst[i][:] = src[idx[i]][:]` for f32 rows of `cols` values.
    pub fn dl_gather_rows_f32(
        &self,
        src: &CudaSlice<f32>,
        dst: &mut CudaSlice<f32>,
        idx: &CudaSlice<u32>,
        n: usize,
        cols: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .dl_gather_rows
            .ok_or(GpuError::MissingOp("dl_gather_rows"))?;
        if dst.len() < n * cols || idx.len() < n {
            return Err(oob("dl_gather_rows: buffers under the geometry"));
        }
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (dp, _g2) = dst.device_ptr_mut(&self.stream);
        let (ip, _g3) = idx.device_ptr(&self.stream);
        // SAFETY: ABI contract (slot 845); indices are the caller's top-k of
        // `src`'s own rows, bounds checked above for the destination
        check(unsafe {
            f(
                sp as *const _,
                dp as *mut _,
                ip as *const _,
                n as u32,
                cols as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Each query's mask box (logit > 0 over `[q][h * w]`) as the decoder's
    /// initial reference point, f32 `[q][4]`.
    pub fn dl_mask_ref(
        &self,
        logits: &CudaSlice<f32>,
        ref_: &mut CudaSlice<f32>,
        q: usize,
        (h, w): (usize, usize),
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .dl_mask_ref
            .ok_or(GpuError::MissingOp("dl_mask_ref"))?;
        if logits.len() < q * h * w || ref_.len() < q * 4 {
            return Err(oob("dl_mask_ref: buffers under the geometry"));
        }
        let (lp, _g1) = logits.device_ptr(&self.stream);
        let (rp, _g2) = ref_.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 846); bounds checked above
        check(unsafe {
            f(
                lp as *const _,
                rp as *mut _,
                q as u32,
                h as u32,
                w as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Multi-scale deformable attention (8 x 32, 3 levels, 4 points) over
    /// `value [s][256]` halves at the level sizes `hw`, for `q` queries.
    #[allow(clippy::too_many_arguments)]
    pub fn dl_msda(
        &self,
        value: &CudaSlice<f16>,
        off: &CudaSlice<f32>,
        logit: &CudaSlice<f32>,
        ref_: &CudaSlice<f32>,
        out: &mut CudaSlice<f16>,
        q: usize,
        hw: [(usize, usize); 3],
    ) -> Result<(), GpuError> {
        let f = self.kernels.dl_msda.ok_or(GpuError::MissingOp("dl_msda"))?;
        let s: usize = hw.iter().map(|(h, w)| h * w).sum();
        if value.len() < s * 256
            || off.len() < q * 192
            || logit.len() < q * 96
            || ref_.len() < q * 4
            || out.len() < q * 256
        {
            return Err(oob("dl_msda: buffers under the geometry"));
        }
        let (vp, _g1) = value.device_ptr(&self.stream);
        let (op, _g2) = off.device_ptr(&self.stream);
        let (lp, _g3) = logit.device_ptr(&self.stream);
        let (rp, _g4) = ref_.device_ptr(&self.stream);
        let (dp, _g5) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 847); bounds checked above
        check(unsafe {
            f(
                vp as *const _,
                op as *const _,
                lp as *const _,
                rp as *const _,
                dp as *mut _,
                q as u32,
                hw[0].0 as u32,
                hw[0].1 as u32,
                hw[1].0 as u32,
                hw[1].1 as u32,
                hw[2].0 as u32,
                hw[2].1 as u32,
                self.stream_ptr(),
            )
        })
    }

    /// The decoder's reference box step (with `delta`, the refinement) and
    /// the 8-wide f16 copy the query-pos head reads.
    pub fn dl_ref_step(
        &self,
        ref_: &mut CudaSlice<f32>,
        delta: Option<&CudaSlice<f32>>,
        ref16: &mut CudaSlice<f16>,
        q: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .dl_ref_step
            .ok_or(GpuError::MissingOp("dl_ref_step"))?;
        if ref_.len() < q * 4 || ref16.len() < q * 8 || delta.is_some_and(|d| d.len() < q * 4) {
            return Err(oob("dl_ref_step: buffers under the geometry"));
        }
        let (rp, _g1) = ref_.device_ptr_mut(&self.stream);
        let dp = delta.map(|d| d.device_ptr(&self.stream));
        let (hp, _g3) = ref16.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 848); bounds checked above, null delta = none
        check(unsafe {
            f(
                rp as *mut _,
                dp.as_ref().map_or(0, |(p, _)| *p) as *const _,
                hp as *mut _,
                q as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Reading-order votes over the global pointer's `[q][q]` scores.
    pub fn dl_order_votes(
        &self,
        s: &CudaSlice<f32>,
        votes: &mut CudaSlice<f32>,
        q: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .dl_order_votes
            .ok_or(GpuError::MissingOp("dl_order_votes"))?;
        if s.len() < q * q || votes.len() < q {
            return Err(oob("dl_order_votes: buffers under the geometry"));
        }
        let (sp, _g1) = s.device_ptr(&self.stream);
        let (vp, _g2) = votes.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 849); bounds checked above
        check(unsafe { f(sp as *const _, vp as *mut _, q as u32, self.stream_ptr()) })
    }
}
