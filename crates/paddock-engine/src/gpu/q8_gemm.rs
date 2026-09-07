//! Q8_0 GEMM/GEMV (repacked, dp4a, mma, mmq pipes).

use super::error::*;
use super::*;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::f16;

impl GpuExecutor {
    /// Vectorized fused Q8_0 GEMV over the repacked layout (the decode fast path):
    /// `y[o] = bias[o] + sum_i data[o][i]*scale[o][i/32]*x[i]`, f32 accumulate.
    pub fn q8_0_gemv_repacked(
        &self,
        w: &RepackedQ8,
        bias: Option<&CudaSlice<f32>>,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_gemv_repacked
            .ok_or(GpuError::MissingOp("q8_0_gemv_repacked"))?;
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (scp, _g2) = w.scale.device_ptr(&self.stream);
        let (xp, _g3) = x.device_ptr(&self.stream);
        let (yp, _g4) = y.device_ptr_mut(&self.stream);
        let (bias_ptr, _gb);
        let bp: *const core::ffi::c_void = match bias {
            Some(b) => {
                (bias_ptr, _gb) = b.device_ptr(&self.stream);
                bias_ptr as *const _
            }
            None => core::ptr::null(),
        };
        check(unsafe {
            f(
                dp as *const _,
                scp as *const _,
                bp,
                xp as *const _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    /// True when the pack carries the multi-segment GEMV (entry 317).
    pub fn has_q8_0_gemv_repacked_multi(&self) -> bool {
        self.kernels.q8_0_gemv_repacked_multi.is_some()
    }

    /// Multi-segment sibling of [`Self::q8_0_gemv_repacked`]: up to three
    /// same-`in_dim` planes sharing one activation, one launch (decode QKV
    /// merge, FFN gate|up merge). Each output plane gets exactly the bytes
    /// the split launches produced - the merge only changes launch-boundary
    /// economics (small grids waste ramp/drain; see the pack kernel note).
    pub fn q8_0_gemv_repacked_multi(
        &self,
        segs: &mut [(&RepackedQ8, &mut CudaSlice<f32>)],
        x: &CudaSlice<f32>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_gemv_repacked_multi
            .ok_or(GpuError::MissingOp("q8_0_gemv_repacked_multi"))?;
        let n_segs = segs.len();
        assert!((1..=3).contains(&n_segs), "1..=3 segments");
        let in_dim = segs[0].0.dims[0];
        let mut dp = [std::ptr::null::<core::ffi::c_void>(); 3];
        let mut sp = [std::ptr::null::<core::ffi::c_void>(); 3];
        let mut yp = [std::ptr::null_mut::<core::ffi::c_void>(); 3];
        let mut rows = [0u32; 3];
        let mut guards = Vec::with_capacity(9);
        for (i, (w, y)) in segs.iter_mut().enumerate() {
            assert_eq!(w.dims[0], in_dim, "segments share in_dim");
            let (d, g1) = w.data.device_ptr(&self.stream);
            let (s, g2) = w.scale.device_ptr(&self.stream);
            let (yy, g3) = y.device_ptr_mut(&self.stream);
            dp[i] = d as *const _;
            sp[i] = s as *const _;
            yp[i] = yy as *mut _;
            rows[i] = w.dims[1] as u32;
            guards.push(g1);
            guards.push(g2);
            guards.push(g3);
        }
        let (xp, _gx) = x.device_ptr(&self.stream);
        let null = std::ptr::null::<core::ffi::c_void>();
        // SAFETY: ABI contract; unused trailing segments pass nulls/0
        check(unsafe {
            f(
                dp[0],
                sp[0],
                null,
                yp[0],
                rows[0],
                dp[1],
                sp[1],
                null,
                yp[1],
                rows[1],
                dp[2],
                sp[2],
                null,
                yp[2],
                rows[2],
                xp as *const _,
                in_dim as u32,
                n_segs as u32,
                self.stream_ptr(),
            )
        })
    }

    /// True when the pack carries the multi-segment nc GEMV (entry 320).
    pub fn has_q8_0_gemv_dp4a_nc_multi(&self) -> bool {
        self.kernels.q8_0_gemv_dp4a_nc_multi.is_some()
    }

    /// Multi-segment sibling of [`Self::q8_0_gemv_dp4a_nc`]: up to four
    /// same-`in_dim` Q8_0 planes sharing one staged int8 activation at
    /// `ncols` columns each, one launch (the r=2..4 batched-decode q|k|v|g
    /// merge, shexp gate|up merge). Each plane's `y` gets exactly the bytes
    /// the split nc launches produced - the merge only changes
    /// launch-boundary economics.
    pub fn q8_0_gemv_dp4a_nc_multi(
        &self,
        segs: &mut [(&RepackedQ8, &mut CudaSlice<f32>)],
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        ncols: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_gemv_dp4a_nc_multi
            .ok_or(GpuError::MissingOp("q8_0_gemv_dp4a_nc_multi"))?;
        let n_segs = segs.len();
        assert!((1..=4).contains(&n_segs), "1..=4 segments");
        let in_dim = segs[0].0.dims[0];
        let mut dp = [std::ptr::null::<core::ffi::c_void>(); 4];
        let mut sp = [std::ptr::null::<core::ffi::c_void>(); 4];
        let mut yp = [std::ptr::null_mut::<core::ffi::c_void>(); 4];
        let mut outs = [0u32; 4];
        let mut guards = Vec::with_capacity(12);
        for (i, (w, y)) in segs.iter_mut().enumerate() {
            assert_eq!(w.dims[0], in_dim, "segments share in_dim");
            let (d, g1) = w.data.device_ptr(&self.stream);
            let (s, g2) = w.scale.device_ptr(&self.stream);
            let (yy, g3) = y.device_ptr_mut(&self.stream);
            dp[i] = d as *const _;
            sp[i] = s as *const _;
            yp[i] = yy as *mut _;
            outs[i] = w.dims[1] as u32;
            guards.push(g1);
            guards.push(g2);
            guards.push(g3);
        }
        let (xqp, _gxq) = xq.device_ptr(&self.stream);
        let (xsp, _gxs) = xs.device_ptr(&self.stream);
        let null = std::ptr::null::<core::ffi::c_void>();
        // SAFETY: ABI contract; unused trailing segments pass nulls/0
        check(unsafe {
            f(
                dp[0],
                sp[0],
                null,
                yp[0],
                outs[0],
                dp[1],
                sp[1],
                null,
                yp[1],
                outs[1],
                dp[2],
                sp[2],
                null,
                yp[2],
                outs[2],
                dp[3],
                sp[3],
                null,
                yp[3],
                outs[3],
                xqp as *const _,
                xsp as *const _,
                in_dim as u32,
                n_segs as u32,
                ncols as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Batched GEMM over the repacked Q8_0 layout - the prefill matmul. Weight read
    /// once per 16 batch rows; at batch=1 bit-identical to `q8_0_gemv_repacked`.
    pub fn q8_0_gemm_repacked(
        &self,
        w: &RepackedQ8,
        bias: Option<&CudaSlice<f32>>,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_gemm_repacked
            .ok_or(GpuError::MissingOp("q8_0_gemm_repacked"))?;
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (scp, _g2) = w.scale.device_ptr(&self.stream);
        let (xp, _g3) = x.device_ptr(&self.stream);
        let (yp, _g4) = y.device_ptr_mut(&self.stream);
        let (bias_ptr, _gb);
        let bp: *const core::ffi::c_void = match bias {
            Some(b) => {
                (bias_ptr, _gb) = b.device_ptr(&self.stream);
                bias_ptr as *const _
            }
            None => core::ptr::null(),
        };
        check(unsafe {
            f(
                dp as *const _,
                scp as *const _,
                bp,
                xp as *const _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn has_q8_0_gemm_repacked_x2(&self) -> bool {
        self.kernels.q8_0_gemm_repacked_x2.is_some()
    }

    /// Two-weight fused repacked GEMM (alpha/beta pair): stages the x tile
    /// once, computes both weights' outputs over it - bit-exact per output vs
    /// two `q8_0_gemm_repacked` calls (same thread mapping + reduce tree),
    /// ~13x less activation L2 traffic. Caller guarantees wa/wb share in_dim.
    #[allow(clippy::too_many_arguments)]
    pub fn q8_0_gemm_repacked_x2(
        &self,
        wa: &RepackedQ8,
        wb: &RepackedQ8,
        x: &CudaSlice<f32>,
        ya: &mut CudaSlice<f32>,
        yb: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_gemm_repacked_x2
            .ok_or(GpuError::MissingOp("q8_0_gemm_repacked_x2"))?;
        assert_eq!(wa.dims[0], wb.dims[0], "x2 weights must share in_dim");
        let (in_dim, oda, odb) = (wa.dims[0], wa.dims[1], wb.dims[1]);
        let (dap, _g1) = wa.data.device_ptr(&self.stream);
        let (sap, _g2) = wa.scale.device_ptr(&self.stream);
        let (dbp, _g3) = wb.data.device_ptr(&self.stream);
        let (sbp, _g4) = wb.scale.device_ptr(&self.stream);
        let (xp, _g5) = x.device_ptr(&self.stream);
        let (yap, _g6) = ya.device_ptr_mut(&self.stream);
        let (ybp, _g7) = yb.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                dap as *const _,
                sap as *const _,
                dbp as *const _,
                sbp as *const _,
                xp as *const _,
                yap as *mut _,
                ybp as *mut _,
                in_dim as u32,
                oda as u32,
                odb as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Dequant a repacked Q8_0 weight into a dense f16 buffer (prefill staging).
    pub fn q8_0_repacked_to_f16(
        &self,
        w: &RepackedQ8,
        out: &mut CudaSlice<f16>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_repacked_to_f16
            .ok_or(GpuError::MissingOp("q8_0_repacked_to_f16"))?;
        let n = w.dims[0] * w.dims[1];
        debug_assert!(out.len() >= n, "f16 staging buffer too small");
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (sp, _g2) = w.scale.device_ptr(&self.stream);
        let (op, _g3) = out.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                dp as *const _,
                sp as *const _,
                op as *mut _,
                n as u64,
                self.stream_ptr(),
            )
        })
    }

    /// Tensor-core GEMM for prefill: f16 weight [out_dim, in_dim] (GGUF row-major)
    /// × f16 activations [batch, in_dim] -> f32 `y` [batch, out_dim], f32 compute
    /// (CUBLAS_COMPUTE_32F). Same layout convention as `matvec_batch` (op A = T).
    pub fn gemm_f16_f32(
        &self,
        w16: &CudaSlice<f16>,
        x16: &CudaSlice<f16>,
        y: &mut CudaSlice<f32>,
        in_dim: usize,
        out_dim: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        self.gemm_f16_f32_beta(w16, x16, y, in_dim, out_dim, batch, 0.0)
    }

    /// [`Self::gemm_f16_f32`] with C-accumulate (`y += W^T x` at beta 1) -
    /// the DFlash fusion fc runs as 5 accumulating band GEMMs over the
    /// block-major aux planes instead of staging a per-row concat.
    pub fn gemm_f16_f32_acc(
        &self,
        w16: &CudaSlice<f16>,
        x16: &CudaSlice<f16>,
        y: &mut CudaSlice<f32>,
        in_dim: usize,
        out_dim: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        self.gemm_f16_f32_beta(w16, x16, y, in_dim, out_dim, batch, 1.0)
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm_f16_f32_beta(
        &self,
        w16: &CudaSlice<f16>,
        x16: &CudaSlice<f16>,
        y: &mut CudaSlice<f32>,
        in_dim: usize,
        out_dim: usize,
        batch: usize,
        beta: f32,
    ) -> Result<(), GpuError> {
        super::basic_ops::gemm_census("B-gemmEx-f16", in_dim, out_dim, batch);
        // The in-house tensor-core GEMM (slot 383) is the
        // only lane - a pack without the slot is a load-time misconfiguration,
        // surfaced loudly as MissingOp rather than silently degrading.
        self.f16_gemm(w16, x16, y, in_dim, out_dim, batch, beta)
    }

    /// In-house f16xf16->f32 tensor-core GEMM (slot 383,
    /// PADDOCK_INHOUSE_F16) - the cuBLAS-free twin of `gemm_f16_f32_beta`. Same
    /// semantics: y = beta*y + W^T x, W=[out_dim,in_dim] f16 row-major,
    /// x=[batch,in_dim] f16 row-major, y=[batch,out_dim] f32.
    pub fn f16_gemm(
        &self,
        w16: &CudaSlice<f16>,
        x16: &CudaSlice<f16>,
        y: &mut CudaSlice<f32>,
        in_dim: usize,
        out_dim: usize,
        batch: usize,
        beta: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .f16_gemm
            .ok_or(GpuError::MissingOp("f16_gemm"))?;
        // Route witness - a dev line telling you which arm actually ran.
        // Compiled out with the rest of the development instruments.
        #[cfg(not(feature = "hardened"))]
        {
            use std::sync::Once;
            static LOG: Once = Once::new();
            LOG.call_once(|| eprintln!("[inhouse-f16] pd_f16_gemm active (slot 383)"));
        }
        let (wp, _g1) = w16.device_ptr(&self.stream);
        let (xp, _g2) = x16.device_ptr(&self.stream);
        let (yp, _g3) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 383); buffers sized by the caller.
        check(unsafe {
            f(
                wp as *const _,
                xp as *const _,
                yp as *mut _,
                beta,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// `f16_gemm` over a ROW SEGMENT of the plane: rows `[first_row,
    /// first_row + out_dim)` of a `[*, in_dim]` f16 plane. The kernel takes a
    /// base pointer and a row count, so the segment is a pointer offset - the
    /// same trick `bf16_gemm_rows` uses on the bf16 twin.
    #[allow(clippy::too_many_arguments)]
    pub fn f16_gemm_rows(
        &self,
        w16: &CudaSlice<f16>,
        first_row: usize,
        in_dim: usize,
        out_dim: usize,
        x16: &CudaSlice<f16>,
        y: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .f16_gemm
            .ok_or(GpuError::MissingOp("f16_gemm"))?;
        let (wp, _g1) = w16.device_ptr(&self.stream);
        let (xp, _g2) = x16.device_ptr(&self.stream);
        let (yp, _g3) = y.device_ptr_mut(&self.stream);
        let wp = wp + (first_row * in_dim * std::mem::size_of::<f16>()) as u64;
        // SAFETY: ABI contract (slot 383); the segment stays inside the plane
        check(unsafe {
            f(
                wp as *const _,
                xp as *const _,
                yp as *mut _,
                0.0f32,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Declare whether another kernel may be RESIDENT while the f16 tensor-core
    /// lane runs (slot 535). `false` clamps the tc5g/tc5gp K-split to 1.
    ///
    /// That split is a cross-CTA producer/consumer - slice 0 stores, slices >0
    /// spin on a device flag - and its factor is elected from `2*nsm / U0`,
    /// i.e. on the assumption that the launch owns the machine. A caller with a
    /// forked side stream breaks that assumption and the device hangs at 100%
    /// with no error. Every flag access is guarded by `KS > 1`, so clamping
    /// makes the protocol unreachable rather than merely unlikely.
    ///
    /// Read at DISPATCH time, so a graph captured while it is `false` bakes the
    /// KS=1 election - which is what makes it safe to set once per walk.
    /// No-op on a pack without the slot; the caller checks `has_f16_ksplit_set`
    /// before enabling the lane in a forked walk.
    pub fn f16_ksplit_set(&self, on: bool) -> Result<(), GpuError> {
        let f = self
            .kernels
            .f16_ksplit_set
            .ok_or(GpuError::MissingOp("f16_ksplit_set"))?;
        // SAFETY: ABI contract (slot 535) - a plain atomic store, no buffers
        check(unsafe { f(if on { 1 } else { 0 }) })
    }

    pub fn has_f16_ksplit_set(&self) -> bool {
        self.kernels.f16_ksplit_set.is_some()
    }

    pub fn has_f16_gemm(&self) -> bool {
        self.kernels.f16_gemm.is_some()
    }

    /// Capture-time f16 mmaf election gate (slot 409): `false`
    /// declines the mmaf arm at `pd_f16_gemm` dispatch until restored, so a
    /// graph captured meanwhile bakes the GEMV/tc5g election instead.
    /// Whisper's overlap routing is the only caller. No-op on a pack
    /// without the slot - check `has_f16_mmaf_gate` before relying on it.
    pub fn f16_mmaf_set(&self, on: bool) {
        if let Some(f) = self.kernels.f16_mmaf_set {
            // SAFETY: ABI contract (slot 409); takes no buffers.
            unsafe { f(on as i32) };
        }
    }

    pub fn has_f16_mmaf_gate(&self) -> bool {
        self.kernels.f16_mmaf_set.is_some()
    }

    /// Small-batch (≤12 rows) tiled repacked GEMM - the spec-decode verify matmul:
    /// activations staged in shared across 16 output rows per block, so x traffic
    /// doesn't scale with out_dim (the plain per-row kernel's failure mode).
    pub fn q8_0_gemm_repacked_mt(
        &self,
        w: &RepackedQ8,
        bias: Option<&CudaSlice<f32>>,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_gemm_repacked_mt
            .ok_or(GpuError::MissingOp("q8_0_gemm_repacked_mt"))?;
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (scp, _g2) = w.scale.device_ptr(&self.stream);
        let (xp, _g3) = x.device_ptr(&self.stream);
        let (yp, _g4) = y.device_ptr_mut(&self.stream);
        let (bias_ptr, _gb);
        let bp: *const core::ffi::c_void = match bias {
            Some(b) => {
                (bias_ptr, _gb) = b.device_ptr(&self.stream);
                bias_ptr as *const _
            }
            None => core::ptr::null(),
        };
        check(unsafe {
            f(
                dp as *const _,
                scp as *const _,
                bp,
                xp as *const _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// int8 MMQ small-batch GEMM: pre-quantized activations (`xq`/`xs` from
    /// `quantize_q8`) × repacked Q8_0 weight via dp4a, f32 accumulate. The
    /// spec-verify fast path - weight read once at full bandwidth. Not bit-exact
    /// vs the f32 path (activation quantization, ~4e-3 - llama's own class).
    pub fn q8_0_gemm_mt_dp4a(
        &self,
        w: &RepackedQ8,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_gemm_mt_dp4a
            .ok_or(GpuError::MissingOp("q8_0_gemm_mt_dp4a"))?;
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (scp, _g2) = w.scale.device_ptr(&self.stream);
        let (xqp, _g3) = xq.device_ptr(&self.stream);
        let (xsp, _g4) = xs.device_ptr(&self.stream);
        let (yp, _g5) = y.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                dp as *const _,
                scp as *const _,
                xqp as *const _,
                xsp as *const _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Bias-folding `q8_0_gemm_mt_dp4a` (bit-exact vs GEMM + `bias_add`).
    pub fn q8_0_gemm_mt_dp4a_b(
        &self,
        w: &RepackedQ8,
        bias: Option<&CudaSlice<f32>>,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_gemm_mt_dp4a_b
            .ok_or(GpuError::MissingOp("q8_0_gemm_mt_dp4a_b"))?;
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (scp, _g2) = w.scale.device_ptr(&self.stream);
        let (xqp, _g3) = xq.device_ptr(&self.stream);
        let (xsp, _g4) = xs.device_ptr(&self.stream);
        let (yp, _g5) = y.device_ptr_mut(&self.stream);
        let (bias_ptr, _gb);
        let bp: *const core::ffi::c_void = match bias {
            Some(b) => {
                (bias_ptr, _gb) = b.device_ptr(&self.stream);
                bias_ptr as *const _
            }
            None => core::ptr::null(),
        };
        // SAFETY: ABI contract, same shapes as q8_0_gemm_mt_dp4a
        check(unsafe {
            f(
                dp as *const _,
                scp as *const _,
                xqp as *const _,
                xsp as *const _,
                bp,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// int8 tensor-core Q8_0 GEMM (mma.sync m16n8k32) - same per-block-scale
    /// numeric class as `q8_0_gemm_mt_dp4a`, dot on the s8 tensor cores.
    pub fn q8_0_gemm_mma(
        &self,
        w: &RepackedQ8,
        xq: &CudaSlice<i8>,
        xs: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_gemm_mma
            .ok_or(GpuError::MissingOp("q8_0_gemm_mma"))?;
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (scp, _g2) = w.scale.device_ptr(&self.stream);
        let (xqp, _g3) = xq.device_ptr(&self.stream);
        let (xsp, _g4) = xs.device_ptr(&self.stream);
        let (yp, _g5) = y.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                dp as *const _,
                scp as *const _,
                xqp as *const _,
                xsp as *const _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Activation quantize into the flat mmq layout (`[chunk][col_pad128][4
    /// f32 scales + 128 int8]`). `yq` must hold
    /// `ceil(in_dim/128) * pad128(batch) * 144` bytes.
    pub fn quantize_q8_mmq(
        &self,
        x: &CudaSlice<f32>,
        yq: &mut CudaSlice<u8>,
        in_dim: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        self.quantize_q8_mmq_rows(x, 0, yq, in_dim, batch)
    }

    /// [`Self::quantize_q8_mmq`] from row `x_row0` of `x` - a chunk of a
    /// wider activation quantized into a chunk-sized `yq`.
    pub fn quantize_q8_mmq_rows(
        &self,
        x: &CudaSlice<f32>,
        x_row0: usize,
        yq: &mut CudaSlice<u8>,
        in_dim: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .quantize_q8_mmq
            .ok_or(GpuError::MissingOp("quantize_q8_mmq"))?;
        debug_assert!(x.len() >= (x_row0 + batch) * in_dim);
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (yp, _g2) = yq.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                (xp + (x_row0 * in_dim * 4) as u64) as *const _,
                yp as *mut _,
                in_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Residual-add + rmsnorm + mmq quantize in one pass: `x += proj` (when
    /// `proj` is given; the residual write lands), then the normalized row
    /// is quantized into `yq`; `xn` receives the normalized row when some.
    /// Bit-exact with the separate kernels. n % 4 == 0, n <= 24576.
    #[allow(clippy::too_many_arguments)]
    pub fn add_rmsnorm_quant_mmq(
        &self,
        x: &mut CudaSlice<f32>,
        proj: Option<&CudaSlice<f32>>,
        proj_b16: bool,
        w: &CudaSlice<f32>,
        xn: Option<&mut CudaSlice<f32>>,
        yq: &mut CudaSlice<u8>,
        n: usize,
        batch: usize,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = if proj_b16 {
            self.kernels.add_rmsnorm_quant_mmq_b16
        } else {
            self.kernels.add_rmsnorm_quant_mmq
        }
        .ok_or(GpuError::MissingOp("add_rmsnorm_quant_mmq"))?;
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let proj_guard = proj.map(|p| p.device_ptr(&self.stream));
        let pp = match &proj_guard {
            Some((p, _)) => *p as *const core::ffi::c_void,
            None => std::ptr::null(),
        };
        let (wp, _g2) = w.device_ptr(&self.stream);
        let xn_guard = xn.map(|p| p.device_ptr_mut(&self.stream));
        let xnp = match &xn_guard {
            Some((p, _)) => *p as *mut core::ffi::c_void,
            None => std::ptr::null_mut(),
        };
        let (yp, _g3) = yq.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                xp as *mut _,
                pp,
                wp as *const _,
                xnp,
                yp as *mut _,
                n as u32,
                batch as u32,
                eps,
                self.stream_ptr(),
            )
        })
    }

    /// SwiGLU fused into the mmq quantize: `yq = quantize(silu(gate) * up)`
    /// without materializing the f32 activation. Bit-identical values to
    /// `swiglu` + `quantize_q8_mmq` run separately.
    pub fn quantize_q8_mmq_swiglu(
        &self,
        gate: &CudaSlice<f32>,
        up: &CudaSlice<f32>,
        yq: &mut CudaSlice<u8>,
        in_dim: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .quantize_q8_mmq_swiglu
            .ok_or(GpuError::MissingOp("quantize_q8_mmq_swiglu"))?;
        let (gp, _g1) = gate.device_ptr(&self.stream);
        let (up_, _g2) = up.device_ptr(&self.stream);
        let (yp, _g3) = yq.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                gp as *const _,
                up_ as *const _,
                yp as *mut _,
                in_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// GEGLU-fused mmq quantize (gemma4 FFN-down feed): gate/up read once,
    /// the f32 activation never lands - saves pd_geglu's full round trip
    /// plus the quantize's re-read per prefill chunk.
    pub fn quantize_q8_mmq_geglu(
        &self,
        gate: &CudaSlice<f32>,
        up: &CudaSlice<f32>,
        yq: &mut CudaSlice<u8>,
        in_dim: usize,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .quantize_q8_mmq_geglu
            .ok_or(GpuError::MissingOp("quantize_q8_mmq_geglu"))?;
        let (gp, _g1) = gate.device_ptr(&self.stream);
        let (up_, _g2) = up.device_ptr(&self.stream);
        let (yp, _g3) = yq.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                gp as *const _,
                up_ as *const _,
                yp as *mut _,
                in_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Whether the pack exports the fused mmq quantize for this activation.
    pub fn has_quantize_q8_mmq_glu(&self, act: GluAct) -> bool {
        match act {
            GluAct::Gelu => self.kernels.quantize_q8_mmq_geglu.is_some(),
            GluAct::Silu => self.kernels.quantize_q8_mmq_swiglu.is_some(),
        }
    }

    /// Fused fold + mmq quantize, dispatched on the model's activation.
    pub fn quantize_q8_mmq_glu(
        &self,
        gate: &CudaSlice<f32>,
        up: &CudaSlice<f32>,
        yq: &mut CudaSlice<u8>,
        in_dim: usize,
        batch: usize,
        act: GluAct,
    ) -> Result<(), GpuError> {
        match act {
            GluAct::Gelu => self.quantize_q8_mmq_geglu(gate, up, yq, in_dim, batch),
            GluAct::Silu => self.quantize_q8_mmq_swiglu(gate, up, yq, in_dim, batch),
        }
    }

    /// mmq-class int8 tensor-core GEMM: activations pre-quantized by
    /// [`Self::quantize_q8_mmq`]. Same numeric class as `q8_0_gemm_mma`.
    /// `fixup` is stream-k scratch (>= 256 * 128 * 128 f32); `None` forces
    /// plain tiling (bit-exact with the mma route, but low-tile-count
    /// launches pay the wave-quantization tail).
    pub fn q8_0_gemm_mmq(
        &self,
        w: &RepackedQ8,
        yq: &CudaSlice<u8>,
        fixup: Option<&mut CudaSlice<f32>>,
        y: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_gemm_mmq
            .ok_or(GpuError::MissingOp("q8_0_gemm_mmq"))?;
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (scp, _g2) = w.scale.device_ptr(&self.stream);
        let (yqp, _g3) = yq.device_ptr(&self.stream);
        let fxp = match fixup {
            Some(fx) => {
                let (p, _g) = fx.device_ptr_mut(&self.stream);
                p as *mut core::ffi::c_void
            }
            None => std::ptr::null_mut(),
        };
        let (yp, _g4) = y.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                dp as *const _,
                scp as *const _,
                yqp as *const _,
                fxp,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Bias-folding `q8_0_gemm_mmq` (bit-exact vs GEMM -> fixup -> `bias_add`:
    /// unsplit tiles fold in the store, split tiles in the fixup pass).
    pub fn q8_0_gemm_mmq_b(
        &self,
        w: &RepackedQ8,
        bias: Option<&CudaSlice<f32>>,
        yq: &CudaSlice<u8>,
        fixup: Option<&mut CudaSlice<f32>>,
        y: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_gemm_mmq_b
            .ok_or(GpuError::MissingOp("q8_0_gemm_mmq_b"))?;
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (scp, _g2) = w.scale.device_ptr(&self.stream);
        let (yqp, _g3) = yq.device_ptr(&self.stream);
        let fxp = match fixup {
            Some(fx) => {
                let (p, _g) = fx.device_ptr_mut(&self.stream);
                p as *mut core::ffi::c_void
            }
            None => std::ptr::null_mut(),
        };
        let (yp, _g4) = y.device_ptr_mut(&self.stream);
        let (bias_ptr, _gb);
        let bp: *const core::ffi::c_void = match bias {
            Some(b) => {
                (bias_ptr, _gb) = b.device_ptr(&self.stream);
                bias_ptr as *const _
            }
            None => core::ptr::null(),
        };
        // SAFETY: ABI contract, same shapes as q8_0_gemm_mmq
        check(unsafe {
            f(
                dp as *const _,
                scp as *const _,
                yqp as *const _,
                bp,
                fxp,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Whether the loaded pack exports the high-occupancy `q8_0_gemm_mmq_hi`
    /// kernel (absent on packs built before E1c - callers fall back to `mmq`).
    pub fn has_q8_0_gemm_mmq_hi(&self) -> bool {
        self.kernels.q8_0_gemm_mmq_hi.is_some()
    }

    /// Whether the loaded pack exports the cp.async-pipelined `q8_0_gemm_mmq_pipe`.
    pub fn has_q8_0_gemm_mmq_pipe(&self) -> bool {
        self.kernels.q8_0_gemm_mmq_pipe.is_some()
    }

    /// cp.async-pipelined (2-stage, double-buffered) tiled mmq GEMM for the
    /// very-large-M encoder prefill (the llama mul_mat_q approach). Same
    /// `RepackedQ8` weight + `quantize_q8_mmq` activation layout as
    /// [`q8_0_gemm_mmq`], no stream-k fixup; requires in_dim % 128 == 0.
    pub fn q8_0_gemm_mmq_pipe(
        &self,
        w: &RepackedQ8,
        bias: Option<&CudaSlice<f32>>,
        yq: &CudaSlice<u8>,
        y: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_gemm_mmq_pipe
            .ok_or(GpuError::MissingOp("q8_0_gemm_mmq_pipe"))?;
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (scp, _g2) = w.scale.device_ptr(&self.stream);
        let (yqp, _g3) = yq.device_ptr(&self.stream);
        let (yp, _g4) = y.device_ptr_mut(&self.stream);
        let bp = match bias {
            Some(b) => b.device_ptr(&self.stream).0 as *const core::ffi::c_void,
            None => core::ptr::null(),
        };
        check(unsafe {
            f(
                dp as *const _,
                scp as *const _,
                yqp as *const _,
                bp,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    pub fn q8_0_gemm_mmq_pipe64(
        &self,
        w: &RepackedQ8,
        yq: &CudaSlice<u8>,
        y: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_gemm_mmq_pipe64
            .ok_or(GpuError::MissingOp("q8_0_gemm_mmq_pipe64"))?;
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (scp, _g2) = w.scale.device_ptr(&self.stream);
        let (yqp, _g3) = yq.device_ptr(&self.stream);
        let (yp, _g4) = y.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                dp as *const _,
                scp as *const _,
                yqp as *const _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// High-occupancy (2 blocks/SM) tiled mmq GEMM for the very-large-M encoder
    /// prefill. Same `RepackedQ8` weight + `quantize_q8_mmq` activation layout
    /// as [`q8_0_gemm_mmq`], no stream-k fixup.
    pub fn q8_0_gemm_mmq_hi(
        &self,
        w: &RepackedQ8,
        yq: &CudaSlice<u8>,
        y: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .q8_0_gemm_mmq_hi
            .ok_or(GpuError::MissingOp("q8_0_gemm_mmq_hi"))?;
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        let (dp, _g1) = w.data.device_ptr(&self.stream);
        let (scp, _g2) = w.scale.device_ptr(&self.stream);
        let (yqp, _g3) = yq.device_ptr(&self.stream);
        let (yp, _g4) = y.device_ptr_mut(&self.stream);
        check(unsafe {
            f(
                dp as *const _,
                scp as *const _,
                yqp as *const _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }
}
