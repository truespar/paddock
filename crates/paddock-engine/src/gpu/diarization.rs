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

    /// Whether the pack carries the BF16-weight GEMM (slot 722) on its own.
    pub fn has_diar_gemm(&self) -> bool {
        self.kernels.diar_gemm.is_some()
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
        shape: (usize, usize, usize),
        epi: DiarEpi,
        rope: Option<(&mut CudaSlice<f32>, &CudaSlice<f32>)>,
    ) -> Result<(), GpuError> {
        self.diar_gemm_parts(x, w, bias, y, shape, epi, rope, 3)
    }

    /// Whether the pack takes the activation's class as an argument (slot
    /// 964): [`Self::diar_gemm_parts`] below 3 parts.
    pub fn has_diar_gemm_parts(&self) -> bool {
        self.kernels.diar_gemm_p.is_some()
    }

    /// [`Self::diar_gemm`] with the activation in `parts` bf16 parts: 3 the
    /// F32 class (slot 722), 2 to 16 significant bits, 1 rounded once to
    /// bf16 - the BF16 class (slot 964).
    #[allow(clippy::too_many_arguments)]
    pub fn diar_gemm_parts(
        &self,
        x: KumoIn<'_>,
        w: DiarWeights<'_>,
        bias: Option<&CudaSlice<f32>>,
        y: &mut CudaSlice<f32>,
        (k, n, m): (usize, usize, usize),
        epi: DiarEpi,
        rope: Option<(&mut CudaSlice<f32>, &CudaSlice<f32>)>,
        parts: u32,
    ) -> Result<(), GpuError> {
        if !(1..=3).contains(&parts) {
            return Err(oob("diar_gemm: parts must be 1, 2 or 3"));
        }
        let f = self
            .kernels
            .diar_gemm
            .ok_or(GpuError::MissingOp("diar_gemm"))?;
        let fp = if parts == 3 {
            None
        } else {
            Some(
                self.kernels
                    .diar_gemm_p
                    .ok_or(GpuError::MissingOp("diar_gemm_p"))?,
            )
        };
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
        let args = (
            (xp + 4 * x.1 as u64) as *const _,
            wp as *const _,
            sp.as_ref().map_or(0, |(p, _)| *p) as *const _,
            bp.as_ref().map_or(0, |(p, _)| *p) as *const _,
            yp as *mut _,
            qp.as_ref().map_or(0, |(p, _)| *p) as *mut _,
            tp.as_ref().map_or(0, |(p, _)| *p) as *const _,
        );
        let wtype = u32::from(matches!(w, DiarWeights::Q8 { .. }));
        // SAFETY: ABI contract (slots 722 / 964); bounds checked above, null
        // scale only with BF16 weights, null bias = none, the q plane and
        // table only with the rope epilogue
        check(unsafe {
            match fp {
                None => f(
                    args.0,
                    args.1,
                    args.2,
                    args.3,
                    args.4,
                    args.5,
                    args.6,
                    k as u32,
                    n as u32,
                    m as u32,
                    wtype,
                    epi as u32,
                    self.stream_ptr(),
                ),
                Some(fp) => fp(
                    args.0,
                    args.1,
                    args.2,
                    args.3,
                    args.4,
                    args.5,
                    args.6,
                    k as u32,
                    n as u32,
                    m as u32,
                    wtype,
                    epi as u32,
                    parts,
                    self.stream_ptr(),
                ),
            }
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

    /// Slot 950: the attention over several windows packed in one pass.
    pub fn has_diar_attention_varlen(&self) -> bool {
        self.kernels.diar_attention_varlen.is_some()
    }

    /// [`Self::diar_attention`] over packed windows (slot 950): `tiles`
    /// (device, `n_tiles` u32 quads {first query row, queries, the window's
    /// first row, its keys}) cut `warps` (2 or 4) x 16 queries high, each
    /// inside one window of the `rows` packed rows.
    #[allow(clippy::too_many_arguments)]
    pub fn diar_attention_varlen(
        &self,
        q: &CudaSlice<f32>,
        qkv: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        rows: usize,
        tiles: &CudaSlice<u32>,
        n_tiles: usize,
        warps: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .diar_attention_varlen
            .ok_or(GpuError::MissingOp("diar_attention_varlen"))?;
        if n_tiles == 0 {
            return Ok(());
        }
        if q.len() < rows * 512
            || qkv.len() < rows * 1536
            || out.len() < rows * 512
            || tiles.len() < 4 * n_tiles
        {
            return Err(oob("diar_attention_varlen: buffers under the geometry"));
        }
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = qkv.device_ptr(&self.stream);
        let (op, _g3) = out.device_ptr_mut(&self.stream);
        let (tp, _g4) = tiles.device_ptr(&self.stream);
        // SAFETY: ABI contract (slot 950); the caller's tiles index rows
        // inside the packed planes checked above
        check(unsafe {
            f(
                qp as *const _,
                (kp + 4 * 512) as *const _,
                (kp + 4 * 1024) as *const _,
                op as *mut _,
                tp as *const _,
                n_tiles as u32,
                warps as u32,
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

/// The attention's keys for [`GpuExecutor::diar_attention_b16`]: one window
/// over its first `klen` keys, or the packed pass's tiles.
pub enum DiarKeys<'a> {
    Window(usize),
    Packed {
        tiles: &'a CudaSlice<u32>,
        n_tiles: usize,
        warps: usize,
    },
}

impl GpuExecutor {
    /// Slots 965-967: the diarization encoder in the BF16 class.
    pub fn has_diar_bf16(&self) -> bool {
        let k = &self.kernels;
        k.diar_b16_gemm.is_some() && k.diar_norm_b16.is_some() && k.diar_attention_b16.is_some()
    }

    /// Slot 965: `y = epi(x16 . W + bias)` over `rows` bf16 rows, W the
    /// stored BF16 rows `[f][k]` (raw bytes). [`DiarEpi::Gelu`] lands bf16 in
    /// `y16`; the others land f32 in `y` (Rope: f 1536, q into `rope.0`).
    #[allow(clippy::too_many_arguments)]
    pub fn diar_b16_gemm(
        &self,
        x16: &CudaSlice<half::bf16>,
        w: &CudaSlice<u8>,
        bias: Option<&CudaSlice<f32>>,
        y: DiarB16Out<'_>,
        (k, f, rows): (usize, usize, usize),
        rope: Option<(&mut CudaSlice<f32>, &CudaSlice<f32>)>,
    ) -> Result<(), GpuError> {
        let func = self
            .kernels
            .diar_b16_gemm
            .ok_or(GpuError::MissingOp("diar_b16_gemm"))?;
        if rows == 0 {
            return Ok(());
        }
        let (epi, ylen) = match &y {
            DiarB16Out::F32(o, e) => (*e as u32, o.len()),
            DiarB16Out::GeluB16(o) => (DiarEpi::Gelu as u32, o.len()),
        };
        let rope_ok = match &rope {
            Some((q, t)) => {
                epi == DiarEpi::Rope as u32
                    && f == 1536
                    && q.len() >= rows * 512
                    && t.len() >= rows * 64
            }
            None => epi != DiarEpi::Rope as u32,
        };
        if !k.is_multiple_of(8)
            || !f.is_multiple_of(64)
            || !(epi == 0 || epi == 1 || epi == 2 || epi == 3)
            || matches!(&y, DiarB16Out::F32(_, e) if *e == DiarEpi::Gelu)
            || !rope_ok
            || x16.len() < rows * k
            || w.len() < 2 * f * k
            || bias.is_some_and(|b| b.len() < f)
            || ylen < rows * f
        {
            return Err(oob("diar_b16_gemm: buffers under the GEMM geometry"));
        }
        let (xp, _g1) = x16.device_ptr(&self.stream);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let bp = bias.map(|b| b.device_ptr(&self.stream));
        let (yp, _g3) = match y {
            DiarB16Out::F32(o, _) => o.device_ptr_mut(&self.stream),
            DiarB16Out::GeluB16(o) => o.device_ptr_mut(&self.stream),
        };
        let (qp, tp) = match rope {
            Some((q, t)) => (
                Some(q.device_ptr_mut(&self.stream)),
                Some(t.device_ptr(&self.stream)),
            ),
            None => (None, None),
        };
        // SAFETY: ABI contract (slot 965); bounds checked above, null bias =
        // none, the q plane and table only with the rope epilogue
        check(unsafe {
            func(
                xp as *const _,
                wp as *const _,
                bp.as_ref().map_or(0, |(p, _)| *p) as *const _,
                yp as *mut _,
                qp.as_ref().map_or(0, |(p, _)| *p) as *mut _,
                tp.as_ref().map_or(0, |(p, _)| *p) as *const _,
                k as u32,
                f as u32,
                rows as u32,
                epi,
                self.stream_ptr(),
            )
        })
    }

    /// Slot 966: slot 719's LayerNorm landed as bf16 `[rows][512]`.
    pub fn diar_norm_b16(
        &self,
        x: KumoIn<'_>,
        (w, b): (&CudaSlice<f32>, &CudaSlice<f32>),
        y16: &mut CudaSlice<half::bf16>,
        rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .diar_norm_b16
            .ok_or(GpuError::MissingOp("diar_norm_b16"))?;
        if rows == 0 {
            return Ok(());
        }
        if x.0.len() < x.1 + rows * 512 || w.len() < 512 || b.len() < 512 || y16.len() < rows * 512
        {
            return Err(oob("diar_norm_b16: buffers under the geometry"));
        }
        let (xp, _g1) = x.0.device_ptr(&self.stream);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let (bp, _g3) = b.device_ptr(&self.stream);
        let (yp, _g4) = y16.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 966); bounds checked above
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

    /// Slot 967: the attention ([`Self::diar_attention`] or the packed
    /// [`Self::diar_attention_varlen`]) landed as bf16 `[rows][512]`.
    pub fn diar_attention_b16(
        &self,
        q: &CudaSlice<f32>,
        qkv: &CudaSlice<f32>,
        out16: &mut CudaSlice<half::bf16>,
        rows: usize,
        keys: DiarKeys<'_>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .diar_attention_b16
            .ok_or(GpuError::MissingOp("diar_attention_b16"))?;
        if rows == 0 {
            return Ok(());
        }
        if q.len() < rows * 512 || qkv.len() < rows * 1536 || out16.len() < rows * 512 {
            return Err(oob("diar_attention_b16: buffers under the geometry"));
        }
        let (klen, tiles, n_tiles, warps) = match keys {
            DiarKeys::Window(klen) => {
                if klen == 0 || klen > rows {
                    return Err(oob("diar_attention_b16: keys past the window"));
                }
                (klen, None, 0, 0)
            }
            DiarKeys::Packed {
                tiles,
                n_tiles,
                warps,
            } => {
                if tiles.len() < 4 * n_tiles {
                    return Err(oob("diar_attention_b16: tiles under the geometry"));
                }
                (0, Some(tiles.device_ptr(&self.stream)), n_tiles, warps)
            }
        };
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = qkv.device_ptr(&self.stream);
        let (op, _g3) = out16.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 967); bounds checked above - the last key
        // read is inside rows * 1536 of the qkv rows
        check(unsafe {
            f(
                qp as *const _,
                (kp + 4 * 512) as *const _,
                (kp + 4 * 1024) as *const _,
                op as *mut _,
                rows as u32,
                klen as u32,
                24,
                tiles.as_ref().map_or(0, |(p, _)| *p) as *const _,
                n_tiles as u32,
                warps as u32,
                self.stream_ptr(),
            )
        })
    }
}

/// Where [`GpuExecutor::diar_b16_gemm`] lands.
pub enum DiarB16Out<'a> {
    /// f32 `[rows][f]`: [`DiarEpi::Store`], [`DiarEpi::Resid`] (accumulate)
    /// or [`DiarEpi::Rope`]
    F32(&'a mut CudaSlice<f32>, DiarEpi),
    /// bf16 `[rows][f]` through the GELU
    GeluB16(&'a mut CudaSlice<half::bf16>),
}
