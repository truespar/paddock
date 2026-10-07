//! SAM 3's tracker-head ops on one picture (PVS: clicks and a box) - the
//! prompt encoder's point features and mask-prompt downscaling, the mask
//! decoder's upscaling seam, its small MLP heads, the single-mask stability
//! count and Meta's hole fill. Kernel side: `packs/cuda/src/sam3/tracker.cuh`,
//! slots 778-783. The two-way transformer between them is the dense lane's
//! GEMMs, `vision_attn_h` and the detector's `sam3_seam_h`.
//!
//! Buffers are checked against the geometry in Rust before any launch.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::f16;

use super::error::*;
use super::*;

/// A small MLP's three layers as `[out][in]` f32 planes with their biases,
/// several rows' worth end to end when each row has its own (the mask
/// decoder's four hypernetwork MLPs).
pub struct Mlp3<'a> {
    pub w1: &'a CudaSlice<f32>,
    pub b1: &'a CudaSlice<f32>,
    pub w2: &'a CudaSlice<f32>,
    pub b2: &'a CudaSlice<f32>,
    pub w3: &'a CudaSlice<f32>,
    pub b3: &'a CudaSlice<f32>,
    pub inp: usize,
    pub hid: usize,
    pub out: usize,
    /// every row reads its own weights (`rows` sets of each plane)
    pub per_row: bool,
}

impl GpuExecutor {
    /// The tracker heads' own ops (slots 778-783), on top of the detector's
    /// (the seam) and the image I/O (mask upsample + RLE).
    pub fn has_sam3_tracker(&self) -> bool {
        let k = &self.kernels;
        self.has_sam3_detector()
            && self.has_sam3_image_io()
            && k.sam3_point_pe.is_some()
            && k.sam3_mask_down.is_some()
            && k.sam3_up_skip.is_some()
            && k.sam3_mlp3_rows.is_some()
            && k.sam3_mask_stats.is_some()
            && k.sam3_fill_holes.is_some()
    }

    /// Random-Fourier features of `n` points normalized to [0, 1] (`xy`
    /// `[n, 2]`) through the `[2, nf]` matrix `g`, `[sin, cos]` -> `[n, 2 nf]`,
    /// plus each label's embedding (`emb` `[4, 2 nf]`, `nap` `[2 nf]`):
    /// -1 padding (replaced by `nap`), 0..3 added. No labels: the plain pe.
    #[allow(clippy::too_many_arguments)]
    pub fn sam3_point_pe(
        &self,
        xy: &CudaSlice<f32>,
        labels: Option<&CudaSlice<u32>>,
        g: &CudaSlice<f32>,
        emb: &CudaSlice<f32>,
        nap: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        n: usize,
        nf: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sam3_point_pe
            .ok_or(GpuError::MissingOp("sam3_point_pe"))?;
        if xy.len() < 2 * n
            || labels.is_some_and(|l| l.len() < n)
            || g.len() < 2 * nf
            || emb.len() < 4 * 2 * nf
            || nap.len() < 2 * nf
            || out.len() < n * 2 * nf
        {
            return Err(oob("sam3_point_pe: buffers under the point geometry"));
        }
        let (xp, _g1) = xy.device_ptr(&self.stream);
        let lg = labels.map(|l| l.device_ptr(&self.stream));
        let lp = lg.as_ref().map_or(0, |(p, _)| *p);
        let (gp, _g3) = g.device_ptr(&self.stream);
        let (ep, _g4) = emb.device_ptr(&self.stream);
        let (np, _g5) = nap.device_ptr(&self.stream);
        let (op, _g6) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 778); bounds checked above, null labels = plain pe
        check(unsafe {
            f(
                xp as *const _,
                lp as *const _,
                gp as *const _,
                ep as *const _,
                np as *const _,
                op as *mut _,
                n as u32,
                nf as u32,
                self.stream_ptr(),
            )
        })
    }

    /// One mask-prompt downscaling stage: Conv2d k2 s2 (`w` `[cout, cin, 2,
    /// 2]`, `b`) -> LayerNorm2d (`ln`, `eps`) -> GELU, from columns
    /// `in_off..in_off + cin` of an `[h * w, in_stride]` f32 plane (clamped
    /// to +-`clamp` when it is positive) into `[(h/2) * (w/2), cout]`.
    #[allow(clippy::too_many_arguments)]
    pub fn sam3_mask_down(
        &self,
        inp: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        ln: (&CudaSlice<f32>, &CudaSlice<f32>),
        out: Sam3MaskDownOut<'_>,
        h: usize,
        wd: usize,
        cin: usize,
        cout: usize,
        in_stride: usize,
        in_off: usize,
        clamp: f32,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sam3_mask_down
            .ok_or(GpuError::MissingOp("sam3_mask_down"))?;
        let on = (h / 2) * (wd / 2) * cout;
        let out_len = match &out {
            Sam3MaskDownOut::F32(o) => o.len(),
            Sam3MaskDownOut::F16(o) => o.len(),
        };
        if inp.len() < h * wd * in_stride
            || in_off + cin > in_stride
            || w.len() < cout * cin * 4
            || b.len() < cout
            || ln.0.len() < cout
            || ln.1.len() < cout
            || out_len < on
        {
            return Err(oob("sam3_mask_down: buffers under the conv geometry"));
        }
        let (ip, _g1) = inp.device_ptr(&self.stream);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let (bp, _g3) = b.device_ptr(&self.stream);
        let (lwp, _g4) = ln.0.device_ptr(&self.stream);
        let (lbp, _g5) = ln.1.device_ptr(&self.stream);
        let (op, half, _g6) = match out {
            Sam3MaskDownOut::F32(o) => {
                let (p, g) = o.device_ptr_mut(&self.stream);
                (p, 0u32, Sam3Guard::F32(g))
            }
            Sam3MaskDownOut::F16(o) => {
                let (p, g) = o.device_ptr_mut(&self.stream);
                (p, 1u32, Sam3Guard::F16(g))
            }
        };
        // SAFETY: ABI contract (slot 779); bounds checked above
        check(unsafe {
            f(
                ip as *const _,
                wp as *const _,
                bp as *const _,
                lwp as *const _,
                lbp as *const _,
                op as *mut _,
                h as u32,
                wd as u32,
                cin as u32,
                cout as u32,
                in_stride as u32,
                in_off as u32,
                clamp,
                eps,
                half,
                self.stream_ptr(),
            )
        })
    }

    /// The mask decoder's upscaling seam: the convT's GEMM landing `g`
    /// `[h * w, 4 C]` (754's tap-major layout) -> depth-to-space, + `bias` +
    /// `skip` `[2h * 2w, C]`, then LayerNorm2d over C when `ln` is given,
    /// then GELU, into f16 `[2h * 2w, C]`.
    #[allow(clippy::too_many_arguments)]
    pub fn sam3_up_skip(
        &self,
        g: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        skip: &CudaSlice<f32>,
        ln: Option<(&CudaSlice<f32>, &CudaSlice<f32>)>,
        out: &mut CudaSlice<f16>,
        h: usize,
        w: usize,
        c: usize,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sam3_up_skip
            .ok_or(GpuError::MissingOp("sam3_up_skip"))?;
        let po = 4 * h * w;
        if g.len() < h * w * 4 * c
            || bias.len() < c
            || skip.len() < po * c
            || out.len() < po * c
            || ln.is_some_and(|(a, b)| a.len() < c || b.len() < c)
        {
            return Err(oob("sam3_up_skip: buffers under the upscaling geometry"));
        }
        let (gp, _g1) = g.device_ptr(&self.stream);
        let (bp, _g2) = bias.device_ptr(&self.stream);
        let (sp, _g3) = skip.device_ptr(&self.stream);
        let lg = ln.map(|(a, b)| (a.device_ptr(&self.stream), b.device_ptr(&self.stream)));
        let (lwp, lbp) = lg.as_ref().map_or((0, 0), |((a, _), (b, _))| (*a, *b));
        let (op, _g5) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 780); bounds checked above, null ln = no norm
        check(unsafe {
            f(
                gp as *const _,
                bp as *const _,
                sp as *const _,
                lwp as *const _,
                lbp as *const _,
                op as *mut _,
                h as u32,
                w as u32,
                c as u32,
                eps,
                1,
                self.stream_ptr(),
            )
        })
    }

    /// A 3-layer MLP (ReLU between, `sigmoid` at the end when asked) over
    /// `rows` rows of `x` `[rows, inp]` into `out` `[rows, out]` f32, and f16
    /// into `out16` too when given. With `m.per_row` each row reads its own
    /// set of the three planes.
    pub fn sam3_mlp3_rows(
        &self,
        x: &CudaSlice<f32>,
        m: &Mlp3<'_>,
        out: &mut CudaSlice<f32>,
        out16: Option<&mut CudaSlice<f16>>,
        rows: usize,
        sigmoid: bool,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sam3_mlp3_rows
            .ok_or(GpuError::MissingOp("sam3_mlp3_rows"))?;
        let sets = if m.per_row { rows } else { 1 };
        let (ws1, ws2, ws3) = (m.hid * m.inp, m.hid * m.hid, m.out * m.hid);
        if x.len() < rows * m.inp
            || out.len() < rows * m.out
            || out16.as_ref().is_some_and(|o| o.len() < rows * m.out)
            || m.w1.len() < sets * ws1
            || m.w2.len() < sets * ws2
            || m.w3.len() < sets * ws3
            || m.b1.len() < sets * m.hid
            || m.b2.len() < sets * m.hid
            || m.b3.len() < sets * m.out
        {
            return Err(oob("sam3_mlp3_rows: buffers under the MLP geometry"));
        }
        let stride = |s: usize| if m.per_row { s as u32 } else { 0 };
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (w1, _g2) = m.w1.device_ptr(&self.stream);
        let (b1, _g3) = m.b1.device_ptr(&self.stream);
        let (w2, _g4) = m.w2.device_ptr(&self.stream);
        let (b2, _g5) = m.b2.device_ptr(&self.stream);
        let (w3, _g6) = m.w3.device_ptr(&self.stream);
        let (b3, _g7) = m.b3.device_ptr(&self.stream);
        let (op, _g8) = out.device_ptr_mut(&self.stream);
        let hg = out16.map(|o| o.device_ptr_mut(&self.stream));
        let hp = hg.as_ref().map_or(0, |(p, _)| *p);
        // SAFETY: ABI contract (slot 781); bounds checked above, null out16 = none
        check(unsafe {
            f(
                xp as *const _,
                w1 as *const _,
                b1 as *const _,
                w2 as *const _,
                b2 as *const _,
                w3 as *const _,
                b3 as *const _,
                op as *mut _,
                hp as *mut _,
                rows as u32,
                m.inp as u32,
                m.hid as u32,
                m.out as u32,
                stride(ws1),
                stride(ws2),
                stride(ws3),
                stride(m.hid),
                stride(m.out),
                u32::from(sigmoid),
                self.stream_ptr(),
            )
        })
    }

    /// The single-mask stability count over column `k` of a `[px, nm]`
    /// logits plane: `counts[0]` = #(v > delta), `counts[1]` = #(v > -delta).
    pub fn sam3_mask_stats(
        &self,
        m: &CudaSlice<f32>,
        counts: &mut CudaSlice<u32>,
        px: usize,
        nm: usize,
        k: usize,
        delta: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sam3_mask_stats
            .ok_or(GpuError::MissingOp("sam3_mask_stats"))?;
        if m.len() < px * nm || counts.len() < 2 || k >= nm {
            return Err(oob("sam3_mask_stats: buffers under the mask geometry"));
        }
        let (mp, _g1) = m.device_ptr(&self.stream);
        let (cp, _g2) = counts.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 782); bounds checked above
        check(unsafe {
            f(
                mp as *const _,
                cp as *mut _,
                px as u32,
                nm as u32,
                k as u32,
                delta,
                self.stream_ptr(),
            )
        })
    }

    /// Meta's hole fill on column `k` of a `side^2 x nm` logits plane, in
    /// place: 8-connected background components (logit <= 0) of at most
    /// `max_area` pixels become logit 10. `lab` / `area` are u32 scratch of
    /// `side^2`.
    #[allow(clippy::too_many_arguments)]
    pub fn sam3_fill_holes(
        &self,
        m: &mut CudaSlice<f32>,
        lab: &mut CudaSlice<u32>,
        area: &mut CudaSlice<u32>,
        side: usize,
        nm: usize,
        k: usize,
        max_area: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sam3_fill_holes
            .ok_or(GpuError::MissingOp("sam3_fill_holes"))?;
        let px = side * side;
        if m.len() < px * nm || lab.len() < px || area.len() < px || k >= nm {
            return Err(oob("sam3_fill_holes: buffers under the mask geometry"));
        }
        let (mp, _g1) = m.device_ptr_mut(&self.stream);
        let (lp, _g2) = lab.device_ptr_mut(&self.stream);
        let (ap, _g3) = area.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 783); bounds checked above
        check(unsafe {
            f(
                mp as *mut _,
                lp as *mut _,
                ap as *mut _,
                side as u32,
                nm as u32,
                k as u32,
                max_area as u32,
                self.stream_ptr(),
            )
        })
    }
}

/// Where [`GpuExecutor::sam3_mask_down`] lands: f32 for the next stage, f16
/// for the 1x1 GEMM after the last.
pub enum Sam3MaskDownOut<'a> {
    F32(&'a mut CudaSlice<f32>),
    F16(&'a mut CudaSlice<f16>),
}

/// Holds whichever device-pointer guard the output's type gave.
enum Sam3Guard<A, B> {
    F32(A),
    F16(B),
}
