//! Gradium Phonon ops (slots 908-917, `packs/cuda/src/phonon.cuh`). The graph
//! is `gpu_model::phonon`.
//!
//! Every buffer is checked against the geometry before a launch. Positions
//! come from device memory (`pos`), so a captured frame step replays with no
//! host argument; the caller keeps those counters in range.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use super::error::*;
use super::*;

/// What the linear's input passes through before the product (slot 908).
#[derive(Clone, Copy)]
pub enum PhPro<'a> {
    None,
    /// LayerNorm with weight + bias
    Ln(&'a CudaSlice<f32>, &'a CudaSlice<f32>),
    /// LayerNorm with weight + bias, then `(1 + scale) * . + shift` from
    /// `mod` (`[shift; scale]`, each `in` wide)
    LnMod(
        &'a CudaSlice<f32>,
        &'a CudaSlice<f32>,
        &'a CudaSlice<f32>,
        usize,
    ),
    /// LayerNorm without affine, then the same modulation
    PlainMod(&'a CudaSlice<f32>, usize),
}

/// What the linear does with its product (slot 908).
#[derive(Clone, Copy)]
pub enum PhEpi<'a> {
    None,
    /// `res + v`; None = the output itself (an in-place residual add)
    Res(Option<&'a CudaSlice<f32>>),
    Gelu,
    Silu,
    /// `res + gate[o] * v` (gate at an element offset); None = in place
    Gated(Option<&'a CudaSlice<f32>>, &'a CudaSlice<f32>, usize),
}

/// The qkv projection's epilogue (slot 908 mode 5): q rotated into the
/// output (`[rows][512]`), k rotated and v into the caches `[cap][512]` at
/// slot `(*pos + pos0 + r) % cap`.
pub struct PhRope<'a> {
    pub kc: &'a mut CudaSlice<f32>,
    pub vc: &'a mut CudaSlice<f32>,
    pub pos: &'a CudaSlice<u32>,
    pub pos0: usize,
    pub cap: usize,
}

/// The weights' L2 policy on a GEMV (slot 908).
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub enum PhHint {
    #[default]
    Default = 0,
    /// read once a frame and gone: `evict_first`, so it does not displace
    /// what stays
    Stream = 1,
    /// read again next frame and small enough to stay: `evict_last`
    Keep = 2,
}

/// The rest of a GEMV's launch: the rope epilogue, outputs that skip the
/// epilogue (from `esplit` on), the L2 policy.
#[derive(Default)]
pub struct PhExtra<'a> {
    pub rope: Option<PhRope<'a>>,
    pub esplit: Option<usize>,
    pub hint: PhHint,
}

/// A Phonon weight plane, `[out][in]` row-major: the checkpoint's BF16
/// rows as raw little-endian bytes (what the 722 GEMM reads too), or F32.
pub enum PhWeight {
    Bf16(CudaSlice<u8>),
    F32(CudaSlice<f32>),
}

impl PhWeight {
    /// Elements in the plane.
    pub fn len(&self) -> usize {
        match self {
            PhWeight::Bf16(b) => b.len() / 2,
            PhWeight::F32(b) => b.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn bytes(&self) -> u64 {
        match self {
            PhWeight::Bf16(b) => b.len() as u64,
            PhWeight::F32(b) => 4 * b.len() as u64,
        }
    }
}

impl GpuExecutor {
    /// Whether the pack carries the Phonon lane (slots 908-917).
    pub fn has_phonon(&self) -> bool {
        let k = &self.kernels;
        k.ph_linear.is_some()
            && k.ph_rope_kv.is_some()
            && k.ph_attn.is_some()
            && k.ph_layernorm.is_some()
            && k.ph_point.is_some()
            && k.ph_im2col.is_some()
            && k.ph_history.is_some()
            && k.ph_upsample.is_some()
            && k.ph_conv.is_some()
            && k.ph_conv_gemm.is_some()
    }

    /// Slot 908: `y = epi(W pro(x) + b)` over `rows` rows (`x` at element
    /// offset `x_off`, `y` at `y_off`).
    #[allow(clippy::too_many_arguments)]
    pub fn ph_linear<'a>(
        &'a self,
        x: &CudaSlice<f32>,
        x_off: usize,
        pro: PhPro<'a>,
        w: &PhWeight,
        bias: Option<&CudaSlice<f32>>,
        epi: PhEpi<'a>,
        y: &mut CudaSlice<f32>,
        y_off: usize,
        (rows, in_dim, out): (usize, usize, usize),
        eps: f32,
        extra: PhExtra<'a>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .ph_linear
            .ok_or(GpuError::MissingOp("ph_linear"))?;
        let ylen = if extra.rope.is_some() {
            rows * 512
        } else {
            rows * out
        };
        if x.len() < x_off + rows * in_dim
            || !x_off.is_multiple_of(4)
            || y.len() < y_off + ylen
            || w.len() < in_dim * out
            || bias.is_some_and(|b| b.len() < out)
            || extra.rope.as_ref().is_some_and(|r| {
                out != 1536 || r.kc.len() < r.cap * 512 || r.vc.len() < r.cap * 512
            })
        {
            return Err(oob("ph_linear: buffers under the pass"));
        }
        let null = core::ptr::null::<core::ffi::c_void>();
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (yp, _g2) = y.device_ptr_mut(&self.stream);
        let (wp, is_bf16, _g3) = match w {
            PhWeight::Bf16(b) => {
                let (p, g) = b.device_ptr(&self.stream);
                (p, 1u32, g)
            }
            PhWeight::F32(b) => {
                let (p, g) = b.device_ptr(&self.stream);
                (p, 0u32, g)
            }
        };
        let (bp, _g4) = match bias {
            Some(b) => {
                let (p, g) = b.device_ptr(&self.stream);
                (p as *const core::ffi::c_void, Some(g))
            }
            None => (null, None),
        };
        let mut guards = Vec::new();
        let mut ptr = |s: &'a CudaSlice<f32>, off: usize| {
            let (p, g) = s.device_ptr(&self.stream);
            guards.push(g);
            (p + (off * 4) as u64) as *const core::ffi::c_void
        };
        let (lw, lb, md, pro_mode) = match pro {
            PhPro::None => (null, null, null, 0u32),
            PhPro::Ln(w, b) => {
                if w.len() < in_dim || b.len() < in_dim {
                    return Err(oob("ph_linear: norm under the input"));
                }
                (ptr(w, 0), ptr(b, 0), null, 1)
            }
            PhPro::LnMod(w, b, m, off) => {
                if w.len() < in_dim || b.len() < in_dim || m.len() < off + 2 * in_dim {
                    return Err(oob("ph_linear: norm / modulation under the input"));
                }
                (ptr(w, 0), ptr(b, 0), ptr(m, off), 2)
            }
            PhPro::PlainMod(m, off) => {
                if m.len() < off + 2 * in_dim {
                    return Err(oob("ph_linear: modulation under the input"));
                }
                (null, null, ptr(m, off), 3)
            }
        };
        // a residual that IS the output keeps the output's pointer (the
        // kernel reads and writes each element in one thread)
        let yres = (yp + (y_off * 4) as u64) as *const core::ffi::c_void;
        let (rp, gp, mut epi_mode) = match epi {
            PhEpi::None => (null, null, 0u32),
            PhEpi::Res(r) => {
                let p = match r {
                    None => yres,
                    Some(r) => {
                        if r.len() < y_off + rows * out {
                            return Err(oob("ph_linear: residual under the output"));
                        }
                        ptr(r, y_off)
                    }
                };
                (p, null, 1)
            }
            PhEpi::Gelu => (null, null, 2),
            PhEpi::Silu => (null, null, 3),
            PhEpi::Gated(r, g, off) => {
                if g.len() < off + out {
                    return Err(oob("ph_linear: gate under the output"));
                }
                let p = match r {
                    None => yres,
                    Some(r) => {
                        if r.len() < y_off + rows * out {
                            return Err(oob("ph_linear: residual under the output"));
                        }
                        ptr(r, y_off)
                    }
                };
                (p, ptr(g, off), 4)
            }
        };
        let mut rope = extra.rope;
        let (kp, vp, pp, pos0, cap, _gr) = match rope.as_mut() {
            Some(r) => {
                if epi_mode != 0 {
                    return Err(oob("ph_linear: the rope epilogue stands alone"));
                }
                epi_mode = 5;
                let (k, g1) = r.kc.device_ptr_mut(&self.stream);
                let (v, g2) = r.vc.device_ptr_mut(&self.stream);
                let (p, g3) = r.pos.device_ptr(&self.stream);
                (k, v, p, r.pos0, r.cap, Some((g1, g2, g3)))
            }
            None => (0, 0, 0, 0, 0, None),
        };
        let esplit = extra.esplit.map_or(u32::MAX, |e| e as u32);
        // SAFETY: ABI contract (slot 908); every extent checked above
        check(unsafe {
            f(
                (xp + (x_off * 4) as u64) as *const _,
                lw,
                lb,
                md,
                wp as *const _,
                is_bf16,
                bp,
                rp,
                gp,
                (yp + (y_off * 4) as u64) as *mut _,
                rows as u32,
                in_dim as u32,
                out as u32,
                eps,
                pro_mode,
                epi_mode,
                kp as *mut _,
                vp as *mut _,
                pp as *const _,
                pos0 as u32,
                cap as u32,
                esplit,
                extra.hint as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 909: RoPE on the rows' q/k and their k/v into the ring caches.
    #[allow(clippy::too_many_arguments)]
    pub fn ph_rope_kv(
        &self,
        qkv: &CudaSlice<f32>,
        q: &mut CudaSlice<f32>,
        kc: &mut CudaSlice<f32>,
        vc: &mut CudaSlice<f32>,
        pos: &CudaSlice<u32>,
        pos0: usize,
        rows: usize,
        cap: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .ph_rope_kv
            .ok_or(GpuError::MissingOp("ph_rope_kv"))?;
        if qkv.len() < rows * 1536
            || q.len() < rows * 512
            || kc.len() < cap * 512
            || vc.len() < cap * 512
        {
            return Err(oob("ph_rope_kv: buffers under the pass"));
        }
        let (ap, _g1) = qkv.device_ptr(&self.stream);
        let (qp, _g2) = q.device_ptr_mut(&self.stream);
        let (kp, _g3) = kc.device_ptr_mut(&self.stream);
        let (vp, _g4) = vc.device_ptr_mut(&self.stream);
        let (pp, _g5) = pos.device_ptr(&self.stream);
        // SAFETY: ABI contract (slot 909); bounds checked above
        check(unsafe {
            f(
                ap as *const _,
                qp as *mut _,
                kp as *mut _,
                vp as *mut _,
                pp as *const _,
                pos0 as u32,
                rows as u32,
                cap as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 910: the rows' attention over the ring caches into `out`, the
    /// splits merged by each head's last block (`ctr`: one zeroed ticket a
    /// row and head, re-armed by the kernel).
    #[allow(clippy::too_many_arguments)]
    pub fn ph_attention(
        &self,
        q: &CudaSlice<f32>,
        kc: &CudaSlice<f32>,
        vc: &CudaSlice<f32>,
        part: &mut CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        ctr: &mut CudaSlice<u32>,
        pos: &CudaSlice<u32>,
        pos0: usize,
        (rows, cap, window, nsplit): (usize, usize, usize, usize),
    ) -> Result<(), GpuError> {
        let f = self.kernels.ph_attn.ok_or(GpuError::MissingOp("ph_attn"))?;
        if q.len() < rows * 512
            || kc.len() < cap * 512
            || vc.len() < cap * 512
            || part.len() < rows * 8 * nsplit * 66
            || out.len() < rows * 512
            || ctr.len() < rows * 8
        {
            return Err(oob("ph_attention: buffers under the pass"));
        }
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = kc.device_ptr(&self.stream);
        let (vp, _g3) = vc.device_ptr(&self.stream);
        let (pp, _g4) = part.device_ptr_mut(&self.stream);
        let (op, _g5) = out.device_ptr_mut(&self.stream);
        let (cp, _g6) = ctr.device_ptr_mut(&self.stream);
        let (dp, _g7) = pos.device_ptr(&self.stream);
        // SAFETY: ABI contract (slot 910); bounds checked above
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                pp as *mut _,
                op as *mut _,
                cp as *mut _,
                dp as *const _,
                pos0 as u32,
                rows as u32,
                cap as u32,
                window as u32,
                nsplit as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 911: LayerNorm of `rows` rows of `d`.
    #[allow(clippy::too_many_arguments)]
    pub fn ph_layernorm(
        &self,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        rows: usize,
        d: usize,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .ph_layernorm
            .ok_or(GpuError::MissingOp("ph_layernorm"))?;
        if x.len() < rows * d || y.len() < rows * d || w.len() < d || b.len() < d {
            return Err(oob("ph_layernorm: buffers under the pass"));
        }
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let (bp, _g3) = b.device_ptr(&self.stream);
        let (yp, _g4) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 911); bounds checked above
        check(unsafe {
            f(
                xp as *const _,
                wp as *const _,
                bp as *const _,
                yp as *mut _,
                rows as u32,
                d as u32,
                eps,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 912: elementwise ops (see the kernel's modes); `a` None = `y`
    /// itself at `y_off` (in place).
    #[allow(clippy::too_many_arguments)]
    pub fn ph_point(
        &self,
        a: Option<&CudaSlice<f32>>,
        a_off: usize,
        b: Option<&CudaSlice<f32>>,
        y: &mut CudaSlice<f32>,
        y_off: usize,
        n: usize,
        period: usize,
        mode: u32,
        alpha: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .ph_point
            .ok_or(GpuError::MissingOp("ph_point"))?;
        let need_b = match mode {
            3 => 2 * period,
            4 => 0,
            _ => period,
        };
        if a.is_some_and(|a| a.len() < a_off + if mode == 4 { period } else { n })
            || y.len() < y_off + n
            || b.map_or(need_b > 0, |b| b.len() < need_b)
        {
            return Err(oob("ph_point: buffers under the pass"));
        }
        let (yp, _g1) = y.device_ptr_mut(&self.stream);
        let (ap, _ga) = match a {
            None => ((yp + (y_off * 4) as u64) as *const core::ffi::c_void, None),
            Some(a) => {
                let (p, g) = a.device_ptr(&self.stream);
                (
                    (p + (a_off * 4) as u64) as *const core::ffi::c_void,
                    Some(g),
                )
            }
        };
        let (bp, _g2) = match b {
            Some(b) => {
                let (p, g) = b.device_ptr(&self.stream);
                (p as *const core::ffi::c_void, Some(g))
            }
            None => (core::ptr::null(), None),
        };
        // SAFETY: ABI contract (slot 912); bounds checked above (an in-place
        // op reads and writes each element in one thread)
        check(unsafe {
            f(
                ap,
                bp,
                (yp + (y_off * 4) as u64) as *mut _,
                n as u64,
                period as u32,
                mode,
                alpha,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 913: a causal conv's GEMM input `y [rows][k * c]` (from element
    /// `y_off`) over `[hist ; x]` (`hist` the k - 1 earlier raw rows), ELU on
    /// it when `elu`. With k = 1 it is the ELU'd copy of x - how a stacked
    /// `[hist ; x]` plane for slot 917 is built.
    #[allow(clippy::too_many_arguments)]
    pub fn ph_im2col(
        &self,
        x: &CudaSlice<f32>,
        hist: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        y_off: usize,
        rows: usize,
        c: usize,
        k: usize,
        elu: bool,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .ph_im2col
            .ok_or(GpuError::MissingOp("ph_im2col"))?;
        if x.len() < rows * c
            || hist.len() < (k - 1).max(1) * c
            || y.len() < y_off + rows * k * c
            || !y_off.is_multiple_of(4)
        {
            return Err(oob("ph_im2col: buffers under the pass"));
        }
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (hp, _g2) = hist.device_ptr(&self.stream);
        let (yp, _g3) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 913); bounds checked above
        check(unsafe {
            f(
                xp as *const _,
                hp as *const _,
                (yp + (y_off * 4) as u64) as *mut _,
                rows as u32,
                c as u32,
                k as u32,
                u32::from(elu),
                self.stream_ptr(),
            )
        })
    }

    /// Slot 914: `hist` <- the last `keep` raw rows of `[hist ; x]`.
    pub fn ph_history(
        &self,
        x: &CudaSlice<f32>,
        hist: &mut CudaSlice<f32>,
        rows: usize,
        c: usize,
        keep: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .ph_history
            .ok_or(GpuError::MissingOp("ph_history"))?;
        if x.len() < rows * c || hist.len() < keep * c {
            return Err(oob("ph_history: buffers under the pass"));
        }
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (hp, _g2) = hist.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 914); bounds checked above
        check(unsafe {
            f(
                xp as *const _,
                hp as *mut _,
                rows as u32,
                c as u32,
                keep as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 915: Mimi's streamed depthwise x16 upsample (`w` `[512][32]`).
    pub fn ph_upsample(
        &self,
        x: &CudaSlice<f32>,
        hist: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        frames: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .ph_upsample
            .ok_or(GpuError::MissingOp("ph_upsample"))?;
        if x.len() < frames * 512
            || hist.len() < 512
            || w.len() < 512 * 32
            || y.len() < frames * 16 * 512
        {
            return Err(oob("ph_upsample: buffers under the pass"));
        }
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (hp, _g2) = hist.device_ptr(&self.stream);
        let (wp, _g3) = w.device_ptr(&self.stream);
        let (yp, _g4) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 915); bounds checked above
        check(unsafe {
            f(
                xp as *const _,
                hp as *const _,
                wp as *const _,
                yp as *mut _,
                frames as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 916: a narrow causal conv straight from its input rows -
    /// `y [rows][n] (+)= W act([hist ; x]) + b`, `w` the GEMM rows
    /// `[n][k cin]`, `hist` the k - 1 earlier raw rows.
    #[allow(clippy::too_many_arguments)]
    pub fn ph_conv(
        &self,
        x: &CudaSlice<f32>,
        hist: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        (rows, cin, n, k): (usize, usize, usize, usize),
        elu: bool,
        resid: bool,
    ) -> Result<(), GpuError> {
        let f = self.kernels.ph_conv.ok_or(GpuError::MissingOp("ph_conv"))?;
        if x.len() < rows * cin
            || hist.len() < (k - 1).max(1) * cin
            || w.len() < n * k * cin
            || bias.len() < n
            || y.len() < rows * n
        {
            return Err(oob("ph_conv: buffers under the pass"));
        }
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (hp, _g2) = hist.device_ptr(&self.stream);
        let (wp, _g3) = w.device_ptr(&self.stream);
        let (bp, _g4) = bias.device_ptr(&self.stream);
        let (yp, _g5) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 916); bounds checked above
        check(unsafe {
            f(
                xp as *const _,
                hp as *const _,
                wp as *const _,
                bp as *const _,
                yp as *mut _,
                rows as u32,
                cin as u32,
                n as u32,
                k as u32,
                u32::from(elu),
                u32::from(resid),
                self.stream_ptr(),
            )
        })
    }

    /// Slot 917: a causal conv as the 3xTF32 GEMM over its im2col rows in
    /// place - `x` the stacked `[k - 1 history rows ; rows]` plane at pitch
    /// `cin` (ELU already applied where the conv takes it), `w` the GEMM rows
    /// `[n][k cin]`: `y [rows][n] (+)= ... + b`.
    #[allow(clippy::too_many_arguments)]
    pub fn ph_conv_gemm(
        &self,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        (rows, cin, n, k): (usize, usize, usize, usize),
        resid: bool,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .ph_conv_gemm
            .ok_or(GpuError::MissingOp("ph_conv_gemm"))?;
        if x.len() < (rows + k - 1) * cin
            || w.len() < n * k * cin
            || bias.len() < n
            || y.len() < rows * n
        {
            return Err(oob("ph_conv_gemm: buffers under the pass"));
        }
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let (bp, _g3) = bias.device_ptr(&self.stream);
        let (yp, _g4) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 917); bounds checked above
        check(unsafe {
            f(
                xp as *const _,
                wp as *const _,
                bp as *const _,
                yp as *mut _,
                rows as u32,
                cin as u32,
                n as u32,
                k as u32,
                u32::from(resid),
                self.stream_ptr(),
            )
        })
    }
}
