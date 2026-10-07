//! EmbeddingGemma 2's audio-tower ops - kernel side
//! `packs/cuda/src/embedding_gemma2_audio.cuh`, slots 813-819.
//!
//! One clip at a time: every plane is `[rows][channels]` of that clip alone.
//! The GEMMs are the shared f16 tensor-core tile and the f32 one; these are
//! the frontend, the subsampling convs and the seams between the GEMMs.
//! Buffers are checked against the geometry before any launch.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::f16;

use super::error::*;
use super::*;

/// The conformer width every audio row kernel is built for (`PD_EG2A_W`).
pub const EG2A_WIDTH: usize = 1024;
/// Mel bins (`PD_EG2A_MEL`).
pub const EG2A_MEL: usize = 128;
/// Samples a frame advances (`PD_EG2A_HOP`).
pub const EG2A_HOP: usize = 160;

/// The frontend's constant tables on the device: the periodic Hann window
/// (320), the mel filterbank `[128][257]`, each row's nonzero bin span, and
/// the FFT twiddles `e^{-2 pi i k / 512}` for k < 256.
pub struct Eg2aMelTables {
    pub window: CudaSlice<f32>,
    pub fb: CudaSlice<f32>,
    pub spans: CudaSlice<u32>,
    pub twiddle: CudaSlice<f32>,
}

/// The seam's optional steps (slot 815), in the order they run: fold a
/// clipped sublayer output `y` into `x`, the block's closing norm, the next
/// GEMM's clamped f16 input.
pub struct Eg2aSeam<'a> {
    /// The sublayer output, its linear's output clamp, an optional RMS
    /// weight applied to it, and its residual weight.
    pub y: Option<(
        &'a CudaSlice<f32>,
        (f32, f32),
        Option<&'a CudaSlice<f32>>,
        f32,
    )>,
    /// `x = rms(x) * w` after the fold.
    pub out_w: Option<&'a CudaSlice<f32>>,
    /// The next GEMM's input: `(norm, weight, clamp, out)` - with `norm`,
    /// `rms(x) * weight` (None = weightless), else `x` itself.
    pub next: Option<(
        bool,
        Option<&'a CudaSlice<f32>>,
        (f32, f32),
        &'a mut CudaSlice<f16>,
    )>,
}

impl GpuExecutor {
    /// True when the loaded pack carries the whole audio tower (the slots
    /// landed together) and the f32 GEMM its input projection rides.
    pub fn has_eg2_audio(&self) -> bool {
        let k = &self.kernels;
        k.gemm_f32.is_some()
            && k.eg2a_mel.is_some()
            && k.eg2a_sscp.is_some()
            && k.eg2a_rows.is_some()
            && k.eg2a_act.is_some()
            && k.eg2a_attn.is_some()
            && k.eg2a_conv.is_some()
            && k.eg2a_out.is_some()
    }

    /// `frames` log-mel frames `[frames][128]` of the `n` samples in `pcm`.
    pub fn eg2a_mel(
        &self,
        pcm: &CudaSlice<f32>,
        n: usize,
        t: &Eg2aMelTables,
        out: &mut CudaSlice<f32>,
        frames: usize,
        llama_floor: bool,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .eg2a_mel
            .ok_or(GpuError::MissingOp("eg2a_mel"))?;
        if pcm.len() < n
            || out.len() < frames * EG2A_MEL
            || frames * EG2A_HOP > n + EG2A_HOP
            || t.window.len() < 320
            || t.fb.len() < EG2A_MEL * 257
            || t.spans.len() < 2 * EG2A_MEL
            || t.twiddle.len() < 512
        {
            return Err(oob("eg2a_mel: buffers under the frame geometry"));
        }
        let (pp, _g1) = pcm.device_ptr(&self.stream);
        let (wp, _g2) = t.window.device_ptr(&self.stream);
        let (fp, _g3) = t.fb.device_ptr(&self.stream);
        let (sp, _g4) = t.spans.device_ptr(&self.stream);
        let (tp, _g5) = t.twiddle.device_ptr(&self.stream);
        let (op, _g6) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 813); bounds checked above
        check(unsafe {
            f(
                pp as *const _,
                n as u32,
                wp as *const _,
                fp as *const _,
                sp as *const _,
                tp as *const _,
                op as *mut _,
                frames as u32,
                llama_floor as u32,
                self.stream_ptr(),
            )
        })
    }

    /// One subsampling layer: `[t_in][f_in][c_in]` -> `[t_out][f_out][c_out]`
    /// (each side `(n - 1) / 2 + 1`), conv + LayerNorm + ReLU.
    #[allow(clippy::too_many_arguments)]
    pub fn eg2a_sscp(
        &self,
        input: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        nw: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        t_in: usize,
        f_in: usize,
        c_in: usize,
        c_out: usize,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .eg2a_sscp
            .ok_or(GpuError::MissingOp("eg2a_sscp"))?;
        let (t_out, f_out) = ((t_in.max(1) - 1) / 2 + 1, (f_in.max(1) - 1) / 2 + 1);
        if input.len() < t_in * f_in * c_in
            || w.len() < c_out * c_in * 9
            || nw.len() < c_out
            || out.len() < t_out * f_out * c_out
        {
            return Err(oob("eg2a_sscp: buffers under the conv geometry"));
        }
        let (ip, _g1) = input.device_ptr(&self.stream);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let (np, _g3) = nw.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 814); bounds checked above
        check(unsafe {
            f(
                ip as *const _,
                wp as *const _,
                np as *const _,
                op as *mut _,
                t_in as u32,
                f_in as u32,
                c_in as u32,
                c_out as u32,
                eps,
                self.stream_ptr(),
            )
        })
    }

    /// The residual seam over `rows` 1024-wide rows of `x` (see [`Eg2aSeam`]).
    pub fn eg2a_rows(
        &self,
        x: &mut CudaSlice<f32>,
        seam: Eg2aSeam<'_>,
        rows: usize,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .eg2a_rows
            .ok_or(GpuError::MissingOp("eg2a_rows"))?;
        let n = rows * EG2A_WIDTH;
        let short = |b: &CudaSlice<f32>, len: usize| b.len() < len;
        if x.len() < n
            || seam
                .y
                .as_ref()
                .is_some_and(|(y, _, w, _)| short(y, n) || w.is_some_and(|w| short(w, EG2A_WIDTH)))
            || seam.out_w.is_some_and(|w| short(w, EG2A_WIDTH))
            || seam
                .next
                .as_ref()
                .is_some_and(|(_, w, _, o)| w.is_some_and(|w| short(w, EG2A_WIDTH)) || o.len() < n)
        {
            return Err(oob("eg2a_rows: buffers under the row geometry"));
        }
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (y, (y_lo, y_hi), post, y_scale) = match seam.y {
            Some((y, c, w, s)) => (Some(y), c, w, s),
            None => (None, (0.0, 0.0), None, 0.0),
        };
        let yp = y.map(|b| b.device_ptr(&self.stream));
        let pp = post.map(|b| b.device_ptr(&self.stream));
        let op = seam.out_w.map(|b| b.device_ptr(&self.stream));
        let (norm, nw, (n_lo, n_hi), x16) = match seam.next {
            Some((norm, w, c, o)) => (norm, w, c, Some(o)),
            None => (false, None, (0.0, 0.0), None),
        };
        let wp = nw.map(|b| b.device_ptr(&self.stream));
        let hp = x16.map(|b| b.device_ptr_mut(&self.stream));
        let p = |o: &Option<(u64, _)>| o.as_ref().map_or(0, |(p, _)| *p);
        // SAFETY: ABI contract (slot 815); bounds checked above
        check(unsafe {
            f(
                xp as *mut _,
                p(&yp) as *const _,
                y_lo,
                y_hi,
                p(&pp) as *const _,
                y_scale,
                p(&op) as *const _,
                norm as u32,
                p(&wp) as *const _,
                n_lo,
                n_hi,
                hp.as_ref().map_or(0, |(p, _)| *p) as *mut _,
                rows as u32,
                eps,
                self.stream_ptr(),
            )
        })
    }

    /// `out = f16(clamp(act(clamp(y, lo, hi)), lo2, hi2))` over `total`
    /// values, `act` SiLU or identity.
    #[allow(clippy::too_many_arguments)]
    pub fn eg2a_act(
        &self,
        y: &CudaSlice<f32>,
        out: &mut CudaSlice<f16>,
        total: usize,
        clip: (f32, f32),
        silu: bool,
        clip2: (f32, f32),
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .eg2a_act
            .ok_or(GpuError::MissingOp("eg2a_act"))?;
        if y.len() < total || out.len() < total {
            return Err(oob("eg2a_act: buffers under the total"));
        }
        let (yp, _g1) = y.device_ptr(&self.stream);
        let (op, _g2) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 816); bounds checked above
        check(unsafe {
            f(
                yp as *const _,
                op as *mut _,
                total as u64,
                clip.0,
                clip.1,
                silu as u32,
                clip2.0,
                clip2.1,
                self.stream_ptr(),
            )
        })
    }

    /// Chunked local attention over the fused `qkv` landing `[rows][3072]`
    /// into the output projection's f16 input `[rows][1024]`. `lims`: the
    /// q, k, v output clamps and the output linear's input clamp.
    #[allow(clippy::too_many_arguments)]
    pub fn eg2a_attn(
        &self,
        qkv: &CudaSlice<f32>,
        rel: &CudaSlice<f32>,
        pds: &CudaSlice<f32>,
        out: &mut CudaSlice<f16>,
        rows: usize,
        lims: &[f32; 8],
        q_scale: f32,
        k_scale: f32,
        cap: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .eg2a_attn
            .ok_or(GpuError::MissingOp("eg2a_attn"))?;
        if qkv.len() < rows * 3 * EG2A_WIDTH
            || rel.len() < 13 * EG2A_WIDTH
            || pds.len() < 128
            || out.len() < rows * EG2A_WIDTH
        {
            return Err(oob("eg2a_attn: buffers under the row geometry"));
        }
        let (qp, _g1) = qkv.device_ptr(&self.stream);
        let (rp, _g2) = rel.device_ptr(&self.stream);
        let (sp, _g3) = pds.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 817); bounds checked above; `lims` is a
        // host array the launcher reads before returning
        check(unsafe {
            f(
                qp as *const _,
                rp as *const _,
                sp as *const _,
                op as *mut _,
                rows as u32,
                lims.as_ptr(),
                q_scale,
                k_scale,
                cap,
                self.stream_ptr(),
            )
        })
    }

    /// The conv module's middle: GLU of the clamped `g` `[rows][2048]`,
    /// causal depthwise conv (`dw` `[1024][5]`), RMS norm (`nw`), SiLU, into
    /// the next linear's clamped f16 input.
    #[allow(clippy::too_many_arguments)]
    pub fn eg2a_conv(
        &self,
        g: &CudaSlice<f32>,
        dw: &CudaSlice<f32>,
        nw: &CudaSlice<f32>,
        out: &mut CudaSlice<f16>,
        rows: usize,
        g_clip: (f32, f32),
        n_clip: (f32, f32),
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .eg2a_conv
            .ok_or(GpuError::MissingOp("eg2a_conv"))?;
        if g.len() < rows * 2 * EG2A_WIDTH
            || dw.len() < 5 * EG2A_WIDTH
            || nw.len() < EG2A_WIDTH
            || out.len() < rows * EG2A_WIDTH
        {
            return Err(oob("eg2a_conv: buffers under the row geometry"));
        }
        let (gp, _g1) = g.device_ptr(&self.stream);
        let (dp, _g2) = dw.device_ptr(&self.stream);
        let (np, _g3) = nw.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 818); bounds checked above
        check(unsafe {
            f(
                gp as *const _,
                dp as *const _,
                np as *const _,
                op as *mut _,
                rows as u32,
                g_clip.0,
                g_clip.1,
                n_clip.0,
                n_clip.1,
                eps,
                self.stream_ptr(),
            )
        })
    }

    /// `out = f16(rms(y + bias))` over `rows` rows of width `n` (<= 1536).
    pub fn eg2a_out(
        &self,
        y: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        out: &mut CudaSlice<f16>,
        rows: usize,
        n: usize,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .eg2a_out
            .ok_or(GpuError::MissingOp("eg2a_out"))?;
        if n == 0 || n > 1536 || y.len() < rows * n || bias.len() < n || out.len() < rows * n {
            return Err(oob("eg2a_out: buffers under the row geometry"));
        }
        let (yp, _g1) = y.device_ptr(&self.stream);
        let (bp, _g2) = bias.device_ptr(&self.stream);
        let (op, _g3) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 819); bounds checked above
        check(unsafe {
            f(
                yp as *const _,
                bp as *const _,
                op as *mut _,
                rows as u32,
                n as u32,
                eps,
                self.stream_ptr(),
            )
        })
    }
}
