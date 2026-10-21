//! The W8 GEMM's tcgen05 arm over pre-tiled weight scales (slots 882 / 883) and the
//! swizzled-scale producers feeding it (slots 884-887): the activation scales written in the
//! GEMM's swizzled atom layout, so both operands' scale tiles come by bulk copy with no
//! swizzle launch in between.

use super::error::*;
use super::*;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

const CUDA_ERROR_NOT_SUPPORTED: i32 = 801;

impl GpuExecutor {
    /// Slot 883: a W8 plane's row-major ue8m0 cells (`scale`, [out][in / 32])
    /// re-ordered into the tiles slot 882 bulk-copies - one 512 B canonical
    /// tile per (128-row tile, 128-K slab). None where the slots are absent or
    /// `scale` is not a plain per-32 plane (lin / bs markers).
    pub(crate) fn f8_tc5_tile(
        &self,
        scale: &CudaSlice<u8>,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Option<CudaSlice<u8>>, GpuError> {
        let (Some(f), Some(_)) = (self.kernels.f8_tc5_tile_scales, self.kernels.f8_gemm_tc5x_t)
        else {
            return Ok(None);
        };
        if !in_dim.is_multiple_of(128) || out_dim == 0 || scale.len() != out_dim * in_dim / 32 {
            return Ok(None);
        }
        let mut tiled = self.alloc_u8(out_dim.div_ceil(128) * (in_dim / 128) * 512)?;
        {
            let (sp, _g1) = scale.device_ptr(&self.stream);
            let (tp, _g2) = tiled.device_ptr_mut(&self.stream);
            // SAFETY: ABI contract (slot 883); `scale` holds out*in/32 cells (checked), `tiled`
            // the tile-order bytes allocated above
            check(unsafe {
                f(
                    sp as *const _,
                    tp as *mut _,
                    in_dim as u32,
                    out_dim as u32,
                    self.stream_ptr(),
                )
            })?;
        }
        Ok(Some(tiled))
    }

    /// Slots 884-887 all present: the producers can write the swizzled scales
    /// [`Self::f8d_gemm_t`] reads with `xs_sw`.
    pub(crate) fn has_w8_swz_chain(&self) -> bool {
        let k = &self.kernels;
        k.add_rmsnorm_e4m3_batch_sw.is_some()
            && k.gated_rmsnorm_e4m3_zs_sw.is_some()
            && k.quantize_e4m3_sw.is_some()
            && k.f8_gemm_tc5x_tsw.is_some()
            && k.f8_tc5_tile_scales.is_some()
    }

    /// Slot 882 (887 with `xs_sw`: `xs` already swizzled, ceil(batch / 128) *
    /// 128 rows' worth): [`Self::f8d_gemm_mma_ks`]'s tcgen05 arm over a plane
    /// whose scales [`Self::f8_tc5_tile`] pre-tiled (`tiled`) - both operands'
    /// scale tiles by bulk copy, the same bits. `Ok(false)` where the slot is
    /// absent or declines (batch <= 64) - run the plain arm (never with `xs_sw`).
    #[allow(clippy::too_many_arguments)]
    pub fn f8d_gemm_tc5x_t(
        &self,
        w: &RepackedMxfp4,
        tiled: &CudaSlice<u8>,
        in_dim: usize,
        out_dim: usize,
        xq: &CudaSlice<i8>,
        (xs, xs_sw): (&CudaSlice<u8>, bool),
        part: &mut CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<bool, GpuError> {
        let slot = if xs_sw {
            self.kernels.f8_gemm_tc5x_tsw
        } else {
            self.kernels.f8_gemm_tc5x_t
        };
        let Some(f) = slot else {
            return Ok(false);
        };
        let xs_rows = if xs_sw {
            batch.next_multiple_of(128)
        } else {
            batch
        };
        if batch <= 64
            || tiled.len() < out_dim.div_ceil(128) * (in_dim / 128) * 512
            || xq.len() < batch * in_dim
            || xs.len() < xs_rows * in_dim / 32
            || y.len() < batch * out_dim
        {
            return Ok(false);
        }
        let n = part.len();
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (tp, _g2) = tiled.device_ptr(&self.stream);
        let (xqp, _g3) = xq.device_ptr(&self.stream);
        let (xsp, _g4) = xs.device_ptr(&self.stream);
        let (pp, _g5) = part.device_ptr_mut(&self.stream);
        let (yp, _g6) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slots 882 / 887); operands as slot 869's, `tiled` and the
        // swizzled `xs` sized above
        let rc = unsafe {
            f(
                dp as *const _,
                tp as *const _,
                xqp as *const _,
                xsp as *const _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                pp as *mut _,
                n as u64,
                self.stream_ptr(),
            )
        };
        if rc == CUDA_ERROR_NOT_SUPPORTED {
            return Ok(false);
        }
        check(rc).map(|()| true)
    }

    /// [`Self::f8d_gemm_mma_ks`] taking the plane with its slot-882 tiled
    /// scales where it has them: the bulk-copy arm first, the plain one on a
    /// decline. `xs_sw`: a producer wrote `xs` swizzled ([`w8_swz_on`] held
    /// there) - slot 887 or an error, the plain arm cannot read it. Kill:
    /// PADDOCK_NO_F8_TILED.
    #[allow(clippy::too_many_arguments)]
    pub fn f8d_gemm_t(
        &self,
        (w, tiled): (&RepackedMxfp4, Option<&CudaSlice<u8>>),
        in_dim: usize,
        out_dim: usize,
        xq: &CudaSlice<i8>,
        (xs, xs_sw): (&CudaSlice<u8>, bool),
        part: &mut CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        if let Some(t) = tiled.filter(|_| !f8_tiled_off())
            && self.f8d_gemm_tc5x_t(w, t, in_dim, out_dim, xq, (xs, xs_sw), part, y, batch)?
        {
            return Ok(());
        }
        if xs_sw {
            return Err(GpuError::MissingOp(
                "f8_gemm_tc5x_tsw (swizzled activation scales)",
            ));
        }
        self.f8d_gemm_mma_ks(w, in_dim, out_dim, xq, xs, part, y, batch)
    }

    /// The W8 vocab head: `x` ([batch][in_dim] f32) quantized - the scales
    /// swizzled where [`w8_swz_on`] holds for the head's tiles - then the
    /// logits GEMM over `(w, tiled)` ([`Self::f8d_gemm_t`]).
    #[allow(clippy::too_many_arguments)]
    pub fn f8_head_logits(
        &self,
        (w, tiled): (&RepackedMxfp4, Option<&CudaSlice<u8>>),
        (in_dim, out_dim): (usize, usize),
        x: &CudaSlice<f32>,
        (q, scale): (&mut CudaSlice<i8>, &mut CudaSlice<u8>),
        part: &mut CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let sw = w8_swz_on(self, tiled, batch);
        let sw = self.quantize_e4m3_w8(x, (&mut *q, &mut *scale), (in_dim, batch), sw)?;
        self.f8d_gemm_t((w, tiled), in_dim, out_dim, q, (scale, sw), part, y, batch)
    }

    /// Slot 886: [`Self::quantize_e4m3`] over `rows` rows of `n_row` with the
    /// scales in the swizzled atom layout (`scale` holds ceil(rows / 128) * 128
    /// rows' worth). `Ok(false)`: absent or out of contract, nothing launched.
    pub fn quantize_e4m3_sw(
        &self,
        x: &CudaSlice<f32>,
        q: &mut CudaSlice<i8>,
        scale: &mut CudaSlice<u8>,
        n_row: usize,
        rows: usize,
    ) -> Result<bool, GpuError> {
        let Some(f) = self.kernels.quantize_e4m3_sw else {
            return Ok(false);
        };
        if !n_row.is_multiple_of(128)
            || x.len() < rows * n_row
            || q.len() < rows * n_row
            || scale.len() < rows.next_multiple_of(128) * n_row / 32
        {
            return Ok(false);
        }
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (qp, _g2) = q.device_ptr_mut(&self.stream);
        let (sp, _g3) = scale.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 886); x / q / the swizzled scale plane sized above
        check(unsafe {
            f(
                xp as *const _,
                qp as *mut _,
                sp as *mut _,
                n_row as u32,
                rows as u32,
                self.stream_ptr(),
            )
        })
        .map(|()| true)
    }
}

impl GpuExecutor {
    /// The W8 GEMM's activation quantize over `rows` rows of `n_row`: slot 886
    /// (scales swizzled) where `sw` ([`w8_swz_on`]) and it takes the shape,
    /// else [`Self::quantize_e4m3`]. True = the scales are swizzled.
    pub fn quantize_e4m3_w8(
        &self,
        x: &CudaSlice<f32>,
        (q, scale): (&mut CudaSlice<i8>, &mut CudaSlice<u8>),
        (n_row, rows): (usize, usize),
        sw: bool,
    ) -> Result<bool, GpuError> {
        if sw && self.quantize_e4m3_sw(x, q, scale, n_row, rows)? {
            return Ok(true);
        }
        self.quantize_e4m3(x, q, scale, n_row * rows)
            .map(|()| false)
    }
}

impl GpuExecutor {
    /// Slots 888 and 889 both present.
    pub fn has_prenorm_e4m3_row_xn(&self) -> bool {
        self.kernels.rmsnorm_e4m3_row_xn.is_some() && self.kernels.add_rmsnorm_e4m3_row_xn.is_some()
    }

    /// The qwen35 decode pre-norm into the f8t lane's row-e4m3 input, the f32
    /// normed row landing in `xn` as well (slots 888 / 889): `proj` Some = the
    /// pending residual added first (`add_rmsnorm_e4m3_row`), None = a plain
    /// norm (`rmsnorm_e4m3_row`). The decode step relies on `xn` holding the
    /// layer's pre-norm until its post-norm; without it the f8t lane served
    /// wrong answers on the NVFP4-overlay checkpoint (2026-10-21). `Ok(false)`:
    /// the slot is absent, nothing launched.
    #[allow(clippy::too_many_arguments)]
    pub fn prenorm_e4m3_row_xn(
        &self,
        x: &mut CudaSlice<f32>,
        proj: Option<&CudaSlice<f32>>,
        w: &CudaSlice<f32>,
        xn: &mut CudaSlice<f32>,
        (q, rs): (&mut CudaSlice<i8>, &mut CudaSlice<f32>),
        (n, eps, rows): (usize, f32, usize),
    ) -> Result<bool, GpuError> {
        let (fa, fp) = (
            self.kernels.add_rmsnorm_e4m3_row_xn,
            self.kernels.rmsnorm_e4m3_row_xn,
        );
        if (proj.is_some() && fa.is_none()) || (proj.is_none() && fp.is_none()) {
            return Ok(false);
        }
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let (np, _g3) = xn.device_ptr_mut(&self.stream);
        let (qp, _g4) = q.device_ptr_mut(&self.stream);
        let (sp, _g5) = rs.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slots 888 / 889), the same buffers as the
        // non-xn twins plus `xn` (rows x n f32, as `x`)
        check(unsafe {
            match (proj, fa, fp) {
                (Some(p), Some(f), _) => {
                    let (pp, _g6) = p.device_ptr(&self.stream);
                    f(
                        xp as *mut _,
                        pp as *const _,
                        wp as *const _,
                        np as *mut _,
                        qp as *mut _,
                        sp as *mut _,
                        n as u32,
                        eps,
                        rows as u32,
                        self.stream_ptr(),
                    )
                }
                (None, _, Some(f)) => f(
                    xp as *const _,
                    wp as *const _,
                    np as *mut _,
                    qp as *mut _,
                    sp as *mut _,
                    n as u32,
                    eps,
                    rows as u32,
                    self.stream_ptr(),
                ),
                _ => return Ok(false),
            }
        })
        .map(|()| true)
    }
}

/// PADDOCK_NO_F8_TILED: the W8 GEMMs keep the per-row scale gathers (slot 869).
pub(crate) fn f8_tiled_off() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_F8_TILED").is_some())
}

/// The swizzled-scale chain for a W8 GEMM over `rows` whose plane has tiled
/// scales: 65..=128 rows (one swizzled row group - the decode tick), slots
/// 884-887 present, the tiled arm on. Producer and GEMM must agree, so both
/// ask this. Kill: PADDOCK_NO_W8_SWZ.
pub(crate) fn w8_swz_on(exec: &GpuExecutor, tiled: Option<&CudaSlice<u8>>, rows: usize) -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    (65..=128).contains(&rows)
        && tiled.is_some()
        && !f8_tiled_off()
        && exec.has_w8_swz_chain()
        && !*OFF.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_W8_SWZ").is_some())
}

impl GpuExecutor {
    /// Debug: the classic tc5 route unconditionally (no flat
    /// intercept) - the dump-diff's reference arm.
    #[allow(clippy::too_many_arguments)]
    pub fn f8t_gemm_no_flat(
        &self,
        w: &F8TilePlane,
        xq: &CudaSlice<i8>,
        xrs: &CudaSlice<f32>,
        part: &mut CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        in_dim: usize,
        out_dim: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .f8t_gemm
            .ok_or(GpuError::MissingOp("f8t_gemm"))?;
        let (wp, _g1) = w.tiles.device_ptr(&self.stream);
        let (sp, _g2) = w.scale.device_ptr(&self.stream);
        let (xp, _g3) = xq.device_ptr(&self.stream);
        let (rp, _g4) = xrs.device_ptr(&self.stream);
        let (pp, _g5) = part.device_ptr_mut(&self.stream);
        let (yp, _g6) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (same as f8t_gemm)
        check(unsafe {
            f(
                wp as *const _,
                sp as *const _,
                xp as *const _,
                rp as *const _,
                pp as *mut _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }
}
