//! Nemotron 3 Diarization ops - the passes the Kumo kernels do not cover.
//! Kernel side: `packs/cuda/src/diarization.cuh`, slots 718-723; the graph
//! that strings them together is `gpu_model::diarization`. The head's small
//! F32 GEMMs are the Kumo one (698).
//!
//! Every plane is a flat f32 buffer, checked against the geometry before any
//! launch.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use super::error::*;
use super::*;

/// What [`GpuExecutor::diar_gemm`] does with `acc + bias`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiarEpi {
    Store = 0,
    Gelu = 1,
    /// accumulate into y - the residual stream
    Resid = 2,
    /// the qkv projection (N 1536): rope on q and k at position row, q into
    /// its own `[rows][512]` plane, k in place, v as computed
    Rope = 3,
    /// a gated MLP's gate and up in one pass over rows interleaved (2j gate
    /// j, 2j + 1 up j): `y[m][j] = silu(gate) * up`, N / 2 wide
    SwiGlu = 4,
    /// the tanh-form GELU (`gelu_pytorch_tanh`)
    GeluTanh = 5,
}

/// What [`GpuExecutor::diar_act`] applies in place.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiarAct {
    Relu = 0,
    Sigmoid = 1,
}

/// How [`GpuExecutor::diar_gemm`] reads a projection's weights.
#[derive(Clone, Copy)]
pub enum DiarWeights<'a> {
    /// BF16 rows `[N][K]`, the raw little-endian bytes
    Bf16(&'a CudaSlice<u8>),
    /// int8 rows `[N][K]` and the F32 block scales `[K/32][N]`
    Q8 {
        q: &'a CudaSlice<u8>,
        scale: &'a CudaSlice<f32>,
    },
}

/// The frontend's constant planes: the 512-tap padded window, the
/// [128][257] filterbank, each band's nonzero span `[first, end)` as u32
/// pairs, and the 256 FFT twiddles as (cos, sin) pairs.
pub struct DiarFrontendPlanes<'a> {
    pub window: &'a CudaSlice<f32>,
    pub fb: &'a CudaSlice<f32>,
    pub spans: &'a CudaSlice<u32>,
    pub twiddle: &'a CudaSlice<f32>,
}

impl GpuExecutor {
    /// True when the pack carries the diarization passes (718-723) and the
    /// Kumo GEMM the head runs on.
    pub fn has_diarization(&self) -> bool {
        let k = &self.kernels;
        k.diar_frontend.is_some()
            && k.diar_norm.is_some()
            && k.diar_conv_rows.is_some()
            && k.diar_act.is_some()
            && k.diar_gemm.is_some()
            && k.diar_attention.is_some()
            && k.kumo_gemm.is_some()
    }

    /// Log-mel frames `start .. start + count` of a PCM window into `out`
    /// `[frames][128]` (slot 718). `pcm` holds samples `offset .. total` of
    /// the recording; frames at or past `count` (up to `frames`) and past the
    /// recording's last full hop are zero.
    #[allow(clippy::too_many_arguments)]
    pub fn diar_frontend(
        &self,
        pcm: &CudaSlice<f32>,
        planes: &DiarFrontendPlanes<'_>,
        out: &mut CudaSlice<f32>,
        (offset, total, start, count): (usize, usize, usize, usize),
        frames: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .diar_frontend
            .ok_or(GpuError::MissingOp("diar_frontend"))?;
        if frames == 0 {
            return Ok(());
        }
        let p = planes;
        if total < offset
            || pcm.len() < total - offset
            || p.window.len() < 512
            || p.fb.len() < 128 * 257
            || p.spans.len() < 256
            || p.twiddle.len() < 512
            || out.len() < frames * 128
            || count > frames
            || total > u32::MAX as usize
        {
            return Err(oob("diar_frontend: buffers under the geometry"));
        }
        let (pp, _g1) = pcm.device_ptr(&self.stream);
        let (wp, _g2) = p.window.device_ptr(&self.stream);
        let (fp, _g3) = p.fb.device_ptr(&self.stream);
        let (sp, _g4) = p.spans.device_ptr(&self.stream);
        let (tp, _g5) = p.twiddle.device_ptr(&self.stream);
        let (op, _g6) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 718); bounds checked above, the
        // kernel's own check covers the offset/start relation
        check(unsafe {
            f(
                pp as *const _,
                wp as *const _,
                fp as *const _,
                sp as *const _,
                tp as *const _,
                op as *mut _,
                offset as u32,
                total as u32,
                start as u32,
                count as u32,
                frames as u32,
                self.stream_ptr(),
            )
        })
    }

    /// LayerNorm (eps 1e-5) of `rows` rows of 512 channels, `x` at an
    /// element offset (slot 719).
    pub fn diar_norm(
        &self,
        x: KumoIn<'_>,
        (w, b): (&CudaSlice<f32>, &CudaSlice<f32>),
        y: &mut CudaSlice<f32>,
        rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .diar_norm
            .ok_or(GpuError::MissingOp("diar_norm"))?;
        if rows == 0 {
            return Ok(());
        }
        if x.0.len() < x.1 + rows * 512 || w.len() < 512 || b.len() < 512 || y.len() < rows * 512 {
            return Err(oob("diar_norm: buffers under the geometry"));
        }
        let (xp, _g1) = x.0.device_ptr(&self.stream);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let (bp, _g3) = b.device_ptr(&self.stream);
        let (yp, _g4) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 719); bounds checked above
        check(unsafe {
            f(
                (xp + 4 * x.1 as u64) as *const _,
                wp as *const _,
                bp as *const _,
                yp as *mut _,
                512,
                rows as u32,
                self.stream_ptr(),
            )
        })
    }

    /// The kernel-3 convolution's rows `[r][576]` from `p` `[r][192]` (slot
    /// 720).
    pub fn diar_conv_rows(
        &self,
        p: &CudaSlice<f32>,
        rows: &mut CudaSlice<f32>,
        r: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .diar_conv_rows
            .ok_or(GpuError::MissingOp("diar_conv_rows"))?;
        if r == 0 {
            return Ok(());
        }
        if p.len() < r * 192 || rows.len() < r * 576 {
            return Err(oob("diar_conv_rows: buffers under the geometry"));
        }
        let (pp, _g1) = p.device_ptr(&self.stream);
        let (rp, _g2) = rows.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 720); bounds checked above
        check(unsafe { f(pp as *const _, rp as *mut _, r as u32, self.stream_ptr()) })
    }

    /// `y = epi(x . W + bias)` over `m` rows on the stored weights (slot
    /// 722). `rope` is `(q plane, (cos, sin) table [m][32] flattened)` and
    /// goes with [`DiarEpi::Rope`] only.
    #[allow(clippy::too_many_arguments)]
    pub fn diar_gemm(
        &self,
        x: KumoIn<'_>,
        w: DiarWeights<'_>,
        bias: Option<&CudaSlice<f32>>,
        y: &mut CudaSlice<f32>,
        (k, n, m): (usize, usize, usize),
        epi: DiarEpi,
        rope: Option<(&mut CudaSlice<f32>, &CudaSlice<f32>)>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .diar_gemm
            .ok_or(GpuError::MissingOp("diar_gemm"))?;
        if m == 0 {
            return Ok(());
        }
        let (wlen, slen) = match w {
            DiarWeights::Bf16(b) => (b.len() >= 2 * n * k, true),
            DiarWeights::Q8 { q, scale } => (q.len() >= n * k, scale.len() >= k / 32 * n),
        };
        let rope_ok = match &rope {
            Some((q, t)) => {
                epi == DiarEpi::Rope && n == 1536 && q.len() >= m * 512 && t.len() >= m * 64
            }
            None => epi != DiarEpi::Rope,
        };
        if k % 32 != 0
            || n % 4 != 0
            || !x.1.is_multiple_of(4)
            || !rope_ok
            || x.0.len() < x.1 + m * k
            || !wlen
            || !slen
            || bias.is_some_and(|b| b.len() < n)
            || (epi == DiarEpi::SwiGlu && n % 2 != 0)
            || y.len() < m * if epi == DiarEpi::SwiGlu { n / 2 } else { n }
        {
            return Err(oob("diar_gemm: buffers under the GEMM geometry"));
        }
        let (xp, _g1) = x.0.device_ptr(&self.stream);
        let (wp, _g2) = match w {
            DiarWeights::Bf16(b) => b.device_ptr(&self.stream),
            DiarWeights::Q8 { q, .. } => q.device_ptr(&self.stream),
        };
        let sp = match w {
            DiarWeights::Q8 { scale, .. } => Some(scale.device_ptr(&self.stream)),
            DiarWeights::Bf16(_) => None,
        };
        let bp = bias.map(|b| b.device_ptr(&self.stream));
        let (yp, _g3) = y.device_ptr_mut(&self.stream);
        let (qp, tp) = match rope {
            Some((q, t)) => (
                Some(q.device_ptr_mut(&self.stream)),
                Some(t.device_ptr(&self.stream)),
            ),
            None => (None, None),
        };
        // SAFETY: ABI contract (slot 722); bounds checked above, null scale
        // only with BF16 weights, null bias = none, the q plane and table
        // only with the rope epilogue
        check(unsafe {
            f(
                (xp + 4 * x.1 as u64) as *const _,
                wp as *const _,
                sp.as_ref().map_or(0, |(p, _)| *p) as *const _,
                bp.as_ref().map_or(0, |(p, _)| *p) as *const _,
                yp as *mut _,
                qp.as_ref().map_or(0, |(p, _)| *p) as *mut _,
                tp.as_ref().map_or(0, |(p, _)| *p) as *const _,
                k as u32,
                n as u32,
                m as u32,
                u32::from(matches!(w, DiarWeights::Q8 { .. })),
                epi as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Attention of every row of a window over its first `klen` rows (slot
    /// 723): 8 heads of 64, q `[rows][512]`, k and v read in place from the
    /// qkv rows `[rows][1536]` (k at +512, v at +1024).
    pub fn diar_attention(
        &self,
        q: &CudaSlice<f32>,
        qkv: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        rows: usize,
        klen: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .diar_attention
            .ok_or(GpuError::MissingOp("diar_attention"))?;
        if rows == 0 {
            return Ok(());
        }
        if klen == 0
            || klen > rows
            || q.len() < rows * 512
            || qkv.len() < rows * 1536
            || out.len() < rows * 512
        {
            return Err(oob("diar_attention: buffers under the geometry"));
        }
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = qkv.device_ptr(&self.stream);
        let (op, _g3) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 723); bounds checked above - the last key
        // read is row klen - 1 of the qkv rows, inside rows * 1536
        check(unsafe {
            f(
                qp as *const _,
                (kp + 4 * 512) as *const _,
                (kp + 4 * 1024) as *const _,
                op as *mut _,
                rows as u32,
                klen as u32,
                24,
                self.stream_ptr(),
            )
        })
    }

    /// ReLU or sigmoid over the first `n` values of `x`, in place (slot 721).
    pub fn diar_act(&self, x: &mut CudaSlice<f32>, n: usize, op: DiarAct) -> Result<(), GpuError> {
        let f = self
            .kernels
            .diar_act
            .ok_or(GpuError::MissingOp("diar_act"))?;
        if n == 0 {
            return Ok(());
        }
        if x.len() < n {
            return Err(oob("diar_act: buffer under the geometry"));
        }
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 721); bounds checked above
        check(unsafe { f(xp as *mut _, n as u64, op as u32, self.stream_ptr()) })
    }
}
