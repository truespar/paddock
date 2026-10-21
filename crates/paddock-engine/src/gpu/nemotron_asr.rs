//! Nemotron 3.5 ASR ops (slots 853-861, `packs/cuda/src/asr/nemotron_asr.cuh`):
//! the conformer's projections on the stored Q8_0 weights, the causal x8
//! subsampler's convolutions, the relative-position chunked attention, the
//! ring commits, the convolution module's core and the RNN-T greedy walk.
//! The graph that strings them together is `gpu_model::nemotron_asr`; the
//! frontend is the diarization one (718), the norms Clef's (724), the
//! subsampler's pointwise and the prompt projector's F32 products Kumo's
//! (698).
//!
//! Every plane is a flat buffer checked against the geometry before any
//! launch.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use super::error::*;
use super::*;

/// What [`GpuExecutor::nasr_gemm`] does with `acc + bias`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NasrEpi {
    Store = 0,
    /// accumulate into y - the residual stream
    Resid = 2,
    Silu = 6,
    /// `y += 0.5 z` - the macaron feed-forward's half step
    HalfResid = 7,
    /// GLU over rows interleaved (2j the value j, 2j + 1 its gate):
    /// `y[m][j] = a * sigmoid(b)`, N / 2 wide
    Glu = 8,
    Relu = 9,
}

/// [`GpuExecutor::nasr_greedy`]'s decoder state, `PdNasrDecState`: h and c
/// `[2 halves][2 layers][640]`, g `[640]`, then `cur` and `count` (u32).
pub const NASR_STATE_FLOATS: usize = 2 * 2 * 640 * 2 + 640;
pub const NASR_STATE_BYTES: usize = 4 * NASR_STATE_FLOATS + 8;
/// The greedy walk's per-CTA argmax partials: two parities of 64 CTAs x 32
/// rows of floats, then the same of u32.
pub const NASR_PART_BYTES: usize = 2 * 2 * 64 * 32 * 4;
/// Streams one greedy launch walks.
pub const NASR_JOBS: usize = 32;

/// One segment of a pass, `PdNasrSeg` (48 bytes): a stream's rows of every
/// plane, the absolute encoder frame of its first row, its chunk, prompt and
/// the device addresses of its K/V and convolution rings.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct NasrSeg {
    pub p0: i64,
    pub kv: u64,
    pub conv: u64,
    pub row0: u32,
    pub rows: u32,
    pub chunk: u32,
    /// the pass's chunks before this segment
    pub chunk0: u32,
    pub prompt: u32,
    pub pad: u32,
}

/// One stream of a greedy launch, `PdNasrJob` (40 bytes): its decoder state
/// (device address), the absolute frame of its first projected row, its
/// rows, its slice of the token planes, whether to prime the decoder, and
/// (out) how many tokens it emitted.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct NasrJob {
    pub st: u64,
    pub frame0: i64,
    pub row0: u32,
    pub rows: u32,
    pub out0: u32,
    pub cap: u32,
    pub prime: u32,
    pub count: u32,
}

/// The bytes of a descriptor array, as a kernel reads it.
pub fn nasr_bytes<T: Copy>(v: &[T]) -> Vec<u8> {
    let n = std::mem::size_of_val(v);
    // SAFETY: T is a plain repr(C) descriptor (integers only), so its bytes
    // are initialized and any alignment of the copy is fine
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), n) }.to_vec()
}

/// The decoder's weights in slot 860's order: the joint head, two LSTM
/// layers' input and hidden halves, the prediction projection (each Q8_0
/// as stored) and the F16 embedding table.
pub struct NasrDecoderWeights<'a> {
    pub head: (&'a ClefQ8, &'a CudaSlice<f32>),
    pub ih: [(&'a ClefQ8, &'a CudaSlice<f32>); 2],
    pub hh: [(&'a ClefQ8, &'a CudaSlice<f32>); 2],
    pub pred: (&'a ClefQ8, &'a CudaSlice<f32>),
    /// `[13088][640]` f16, the raw little-endian bytes
    pub embed: &'a CudaSlice<u8>,
}

impl GpuExecutor {
    /// True when the pack carries the Nemotron 3.5 ASR passes (853-861) and
    /// the shared ones the lane strings with them.
    pub fn has_nemotron_asr(&self) -> bool {
        let k = &self.kernels;
        k.nasr_gemm.is_some()
            && k.nasr_sub_stem.is_some()
            && k.nasr_sub_dw.is_some()
            && k.nasr_sub_flatten.is_some()
            && k.nasr_attention.is_some()
            && k.nasr_ring_put.is_some()
            && k.nasr_dwconv.is_some()
            && k.nasr_greedy.is_some()
            && k.nasr_prompt_relu.is_some()
            && k.diar_frontend.is_some()
            && k.diar_act.is_some()
            && k.clef_norm.is_some()
            && k.kumo_gemm.is_some()
    }

    /// `y = epi(x . W^T + bias)` over `rows` rows on Q8_0 weights as stored
    /// (slot 853), x read from element `x.1`.
    pub fn nasr_gemm(
        &self,
        x: KumoIn<'_>,
        w: &ClefQ8,
        bias: Option<&CudaSlice<f32>>,
        y: &mut CudaSlice<f32>,
        rows: usize,
        epi: NasrEpi,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .nasr_gemm
            .ok_or(GpuError::MissingOp("nasr_gemm"))?;
        if rows == 0 {
            return Ok(());
        }
        let (k, n) = (w.k, w.n);
        let out = if epi == NasrEpi::Glu { n / 2 } else { n };
        if k % 32 != 0
            || n % 8 != 0
            || !x.1.is_multiple_of(4)
            || w.q.len() < k * n
            || w.scale.len() < k / 32 * n * 2
            || x.0.len() < x.1 + rows * k
            || y.len() < rows * out
            || bias.is_some_and(|b| b.len() < n)
        {
            return Err(oob("nasr_gemm: buffers under the GEMM geometry"));
        }
        let (xp, _g1) = x.0.device_ptr(&self.stream);
        let (wp, _g2) = w.q.device_ptr(&self.stream);
        let (sp, _g3) = w.scale.device_ptr(&self.stream);
        let (yp, _g4) = y.device_ptr_mut(&self.stream);
        let bp = bias.map(|b| b.device_ptr(&self.stream));
        // SAFETY: ABI contract (slot 853); bounds checked above, null bias =
        // none
        check(unsafe {
            f(
                (xp + 4 * x.1 as u64) as *const _,
                wp as *const _,
                sp as *const _,
                bp.as_ref()
                    .map_or(std::ptr::null(), |(p, _)| *p as *const _),
                yp as *mut _,
                k as u32,
                n as u32,
                rows as u32,
                epi as u32,
                self.stream_ptr(),
            )
        })
    }

    /// The subsampler's stem (slot 854): rows `first .. first + count` of
    /// `out` `[t][65][256]` from `mel` `[rows][128]` holding absolute frames
    /// `base ..`, frames at or past `valid` zero.
    #[allow(clippy::too_many_arguments)]
    pub fn nasr_sub_stem(
        &self,
        mel: &CudaSlice<f32>,
        (base, rows, valid): (usize, usize, i64),
        (w, b): (&CudaSlice<f32>, &CudaSlice<f32>),
        out: &mut CudaSlice<f32>,
        first: usize,
        count: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .nasr_sub_stem
            .ok_or(GpuError::MissingOp("nasr_sub_stem"))?;
        if count == 0 {
            return Ok(());
        }
        if mel.len() < rows * 128
            || w.len() < 256 * 9
            || b.len() < 256
            || out.len() < count * 65 * 256
        {
            return Err(oob("nasr_sub_stem: buffers under the geometry"));
        }
        let (mp, _g1) = mel.device_ptr(&self.stream);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let (bp, _g3) = b.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 854); bounds checked above, the kernel
        // reads only plane rows and refuses a plane missing the first tap
        check(unsafe {
            f(
                mp as *const _,
                base as i64,
                rows as u32,
                valid,
                wp as *const _,
                bp as *const _,
                op as *mut _,
                first as i64,
                count as u32,
                self.stream_ptr(),
            )
        })
    }

    /// A separable stage's depthwise half (slot 855): rows `first ..` of
    /// `out` `[t][fin / 2 + 1][256]` from `input` `[rows][fin][256]`
    /// holding absolute rows `base ..`.
    #[allow(clippy::too_many_arguments)]
    pub fn nasr_sub_dw(
        &self,
        input: &CudaSlice<f32>,
        (base, rows, valid): (usize, usize, i64),
        fin: usize,
        (w, b): (&CudaSlice<f32>, &CudaSlice<f32>),
        out: &mut CudaSlice<f32>,
        first: usize,
        count: usize,
        relu_in: bool,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .nasr_sub_dw
            .ok_or(GpuError::MissingOp("nasr_sub_dw"))?;
        if count == 0 {
            return Ok(());
        }
        if fin < 2
            || input.len() < rows * fin * 256
            || w.len() < 256 * 9
            || b.len() < 256
            || out.len() < count * (fin / 2 + 1) * 256
        {
            return Err(oob("nasr_sub_dw: buffers under the geometry"));
        }
        let (ip, _g1) = input.device_ptr(&self.stream);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let (bp, _g3) = b.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 855); bounds checked above
        check(unsafe {
            f(
                ip as *const _,
                base as i64,
                rows as u32,
                valid,
                fin as u32,
                wp as *const _,
                bp as *const _,
                op as *mut _,
                first as i64,
                count as u32,
                u32::from(relu_in),
                self.stream_ptr(),
            )
        })
    }

    /// ReLU and the channel-major flatten `[rows][17][256]` -> `[rows][4352]`
    /// (slot 856), the output at row `out_row`.
    pub fn nasr_sub_flatten(
        &self,
        input: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        out_row: usize,
        rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .nasr_sub_flatten
            .ok_or(GpuError::MissingOp("nasr_sub_flatten"))?;
        if rows == 0 {
            return Ok(());
        }
        if input.len() < rows * 17 * 256 || out.len() < (out_row + rows) * 17 * 256 {
            return Err(oob("nasr_sub_flatten: buffers under the geometry"));
        }
        let (ip, _g1) = input.device_ptr(&self.stream);
        let (op, _g2) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 856); bounds checked above
        check(unsafe {
            f(
                ip as *const _,
                (op + 4 * (out_row * 17 * 256) as u64) as *mut _,
                rows as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Relative-position attention over every segment of a pass (slot
    /// 857): q, k, v the qkv rows `[rows][3072]`, `chunks` the segments'
    /// chunks in all, `rel` the layer's `[83][1024]` relative-key table,
    /// `bu` / `bv` the content and position biases `[8][128]`. The segments'
    /// geometry is the engine's (checked where it builds them).
    #[allow(clippy::too_many_arguments)]
    pub fn nasr_attention(
        &self,
        qkv: &CudaSlice<f32>,
        segs: &CudaSlice<u8>,
        (nseg, chunks, rows): (usize, usize, usize),
        layer: usize,
        rel: &CudaSlice<f32>,
        (bu, bv): (&CudaSlice<f32>, &CudaSlice<f32>),
        out: &mut CudaSlice<f32>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .nasr_attention
            .ok_or(GpuError::MissingOp("nasr_attention"))?;
        if nseg == 0 || chunks == 0 {
            return Ok(());
        }
        if layer >= 24
            || segs.len() < nseg * std::mem::size_of::<NasrSeg>()
            || qkv.len() < rows * 3072
            || rel.len() < 83 * 1024
            || bu.len() < 1024
            || bv.len() < 1024
            || out.len() < rows * 1024
        {
            return Err(oob("nasr_attention: buffers under the geometry"));
        }
        let (qp, _g1) = qkv.device_ptr(&self.stream);
        let (sp, _g2) = segs.device_ptr(&self.stream);
        let (rp, _g4) = rel.device_ptr(&self.stream);
        let (up, _g5) = bu.device_ptr(&self.stream);
        let (bvp, _g6) = bv.device_ptr(&self.stream);
        let (op, _g7) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 857); bounds checked above, the
        // segments' rows inside `rows` and their rings live (the engine's
        // construction)
        check(unsafe {
            f(
                qp as *const _,
                sp as *const _,
                nseg as u32,
                chunks as u32,
                layer as u32,
                rp as *const _,
                up as *const _,
                bvp as *const _,
                op as *mut _,
                (128f32).sqrt().recip(),
                self.stream_ptr(),
            )
        })
    }

    /// Each segment's last rows of `src` (row stride `ld`, from column
    /// `col0`, `count` 1024-wide column blocks) into its layer's rings
    /// `which0 ..` (0 K, 1 V, 2 the convolution's), slot `frame % slots`
    /// (slot 858).
    #[allow(clippy::too_many_arguments)]
    pub fn nasr_ring_put(
        &self,
        src: &CudaSlice<f32>,
        (ld, col0): (usize, usize),
        segs: &CudaSlice<u8>,
        nseg: usize,
        layer: usize,
        (which0, count): (usize, usize),
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .nasr_ring_put
            .ok_or(GpuError::MissingOp("nasr_ring_put"))?;
        if nseg == 0 || count == 0 {
            return Ok(());
        }
        if layer >= 24
            || which0 + count > 3
            || ld < col0 + count * 1024
            || segs.len() < nseg * std::mem::size_of::<NasrSeg>()
        {
            return Err(oob("nasr_ring_put: buffers under the geometry"));
        }
        let (srp, _g1) = src.device_ptr(&self.stream);
        let (sp, _g2) = segs.device_ptr(&self.stream);
        // SAFETY: ABI contract (slot 858); the segments' rows inside `src`
        // and their rings live (the engine's construction)
        check(unsafe {
            f(
                srp as *const _,
                ld as u32,
                col0 as u32,
                sp as *const _,
                nseg as u32,
                layer as u32,
                which0 as u32,
                count as u32,
                self.stream_ptr(),
            )
        })
    }

    /// The convolution module's core over a pass's `rows` rows (slot 859):
    /// the causal kernel-9 depthwise conv of the GLU rows `g` with each
    /// stream's ring of earlier ones, LayerNorm (`nw`, `nb`) and SiLU.
    #[allow(clippy::too_many_arguments)]
    pub fn nasr_dwconv(
        &self,
        g: &CudaSlice<f32>,
        segs: &CudaSlice<u8>,
        (nseg, rows): (usize, usize),
        layer: usize,
        w: &CudaSlice<f32>,
        (nw, nb): (&CudaSlice<f32>, &CudaSlice<f32>),
        out: &mut CudaSlice<f32>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .nasr_dwconv
            .ok_or(GpuError::MissingOp("nasr_dwconv"))?;
        if rows == 0 || nseg == 0 {
            return Ok(());
        }
        if layer >= 24
            || segs.len() < nseg * std::mem::size_of::<NasrSeg>()
            || g.len() < rows * 1024
            || w.len() < 1024 * 9
            || nw.len() < 1024
            || nb.len() < 1024
            || out.len() < rows * 1024
        {
            return Err(oob("nasr_dwconv: buffers under the geometry"));
        }
        let (gp, _g1) = g.device_ptr(&self.stream);
        let (sp, _g2) = segs.device_ptr(&self.stream);
        let (wp, _g3) = w.device_ptr(&self.stream);
        let (np, _g4) = nw.device_ptr(&self.stream);
        let (bp, _g5) = nb.device_ptr(&self.stream);
        let (op, _g6) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 859); bounds checked above
        check(unsafe {
            f(
                gp as *const _,
                sp as *const _,
                nseg as u32,
                rows as u32,
                layer as u32,
                wp as *const _,
                np as *const _,
                bp as *const _,
                op as *mut _,
                self.stream_ptr(),
            )
        })
    }

    /// `y = ReLU(y + table[prompt])` over a pass's `rows` rows of `n`, the
    /// prompt each row's segment's (slot 861).
    pub fn nasr_prompt_relu(
        &self,
        y: &mut CudaSlice<f32>,
        (n, rows): (usize, usize),
        segs: &CudaSlice<u8>,
        nseg: usize,
        table: &CudaSlice<f32>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .nasr_prompt_relu
            .ok_or(GpuError::MissingOp("nasr_prompt_relu"))?;
        if rows == 0 || nseg == 0 {
            return Ok(());
        }
        if y.len() < rows * n
            || segs.len() < nseg * std::mem::size_of::<NasrSeg>()
            || table.len() < 128 * n
        {
            return Err(oob("nasr_prompt_relu: buffers under the geometry"));
        }
        let (yp, _g1) = y.device_ptr_mut(&self.stream);
        let (sp, _g2) = segs.device_ptr(&self.stream);
        let (tp, _g3) = table.device_ptr(&self.stream);
        // SAFETY: ABI contract (slot 861); bounds checked above, prompts
        // below 128 (the engine's construction)
        check(unsafe {
            f(
                yp as *mut _,
                n as u32,
                rows as u32,
                sp as *const _,
                nseg as u32,
                tp as *const _,
                self.stream_ptr(),
            )
        })
    }

    /// The batched RNN-T greedy walk over up to [`NASR_JOBS`] streams (slot
    /// 860, one cooperative grid): `proj` the pass's projected frames
    /// `[rows][640]`, `jobs` their [`NasrJob`]s (each job's `count` written
    /// back), the tokens and their frames into each job's slice.
    #[allow(clippy::too_many_arguments)]
    pub fn nasr_greedy(
        &self,
        w: &NasrDecoderWeights<'_>,
        proj: &CudaSlice<f32>,
        jobs: &mut CudaSlice<u8>,
        count: usize,
        (tokens, frames): (&mut CudaSlice<u32>, &mut CudaSlice<u32>),
        part: &mut CudaSlice<u8>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .nasr_greedy
            .ok_or(GpuError::MissingOp("nasr_greedy"))?;
        if count == 0 {
            return Ok(());
        }
        let q8 = |q: &ClefQ8, rows: usize| {
            q.k == 640 && q.n == rows && q.q.len() >= rows * 640 && q.scale.len() >= 20 * rows * 2
        };
        if count > NASR_JOBS
            || jobs.len() < count * std::mem::size_of::<NasrJob>()
            || part.len() < NASR_PART_BYTES
            || !q8(w.head.0, 13_088)
            || w.head.1.len() < 13_088
            || w.ih
                .iter()
                .chain(&w.hh)
                .any(|(q, b)| !q8(q, 2560) || b.len() < 2560)
            || !q8(w.pred.0, 640)
            || w.pred.1.len() < 640
            || w.embed.len() < 13_088 * 640 * 2
            || proj.is_empty()
            || tokens.is_empty()
            || frames.is_empty()
        {
            return Err(oob("nasr_greedy: buffers under the geometry"));
        }
        let s = &self.stream;
        let (hq, _a1) = w.head.0.q.device_ptr(s);
        let (hs, _a2) = w.head.0.scale.device_ptr(s);
        let (hb, _a3) = w.head.1.device_ptr(s);
        let (i0q, _b1) = w.ih[0].0.q.device_ptr(s);
        let (i0s, _b2) = w.ih[0].0.scale.device_ptr(s);
        let (i0b, _b3) = w.ih[0].1.device_ptr(s);
        let (h0q, _c1) = w.hh[0].0.q.device_ptr(s);
        let (h0s, _c2) = w.hh[0].0.scale.device_ptr(s);
        let (h0b, _c3) = w.hh[0].1.device_ptr(s);
        let (i1q, _d1) = w.ih[1].0.q.device_ptr(s);
        let (i1s, _d2) = w.ih[1].0.scale.device_ptr(s);
        let (i1b, _d3) = w.ih[1].1.device_ptr(s);
        let (h1q, _e1) = w.hh[1].0.q.device_ptr(s);
        let (h1s, _e2) = w.hh[1].0.scale.device_ptr(s);
        let (h1b, _e3) = w.hh[1].1.device_ptr(s);
        let (pq, _f1) = w.pred.0.q.device_ptr(s);
        let (ps, _f2) = w.pred.0.scale.device_ptr(s);
        let (pb, _f3) = w.pred.1.device_ptr(s);
        let (em, _f4) = w.embed.device_ptr(s);
        let ptrs: [*const core::ffi::c_void; 19] = [
            hq, hs, hb, i0q, i0s, i0b, h0q, h0s, h0b, i1q, i1s, i1b, h1q, h1s, h1b, pq, ps, pb, em,
        ]
        .map(|p| p as *const core::ffi::c_void);
        let (jp, _g1) = proj.device_ptr(s);
        let (jbp, _g2) = jobs.device_ptr_mut(s);
        let (tp, _g3) = tokens.device_ptr_mut(s);
        let (fp, _g4) = frames.device_ptr_mut(s);
        let (pp, _g5) = part.device_ptr_mut(s);
        // SAFETY: ABI contract (slot 860); bounds checked above - `ptrs` is a
        // host array the launcher reads before returning, every entry a live
        // device plane of the checked geometry; the jobs' rows, states and
        // token slices are the engine's construction
        check(unsafe {
            f(
                ptrs.as_ptr(),
                jp as *const _,
                jbp as *mut _,
                count as u32,
                tp as *mut _,
                fp as *mut _,
                pp as *mut _,
                self.stream_ptr(),
            )
        })
    }
}
