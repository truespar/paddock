//! Executor ops for the Clef lane: the backbone's attention over one
//! request's rows of a packed pass (slot 732; the pass's other rows are
//! other requests, so every per-row pointer is offset by the run's first
//! row), and the joint schema head's passes (slots 724-731).

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use paddock_models::ggml_type::GgmlType;

use super::error::*;
use super::*;

impl GpuExecutor {
    /// Causal attention over one request's rows (slot 732): query rows
    /// `row_off .. row_off + rows` of `q` (`heads` x 256 wide) over the same
    /// rows of `k` and `v` (`kv_heads` x 256 wide, F32 as projected), the
    /// result into the same rows of `out`. Grouped: q head h reads kv head
    /// h / (heads / kv_heads).
    #[allow(clippy::too_many_arguments)]
    pub fn clef_attn_tc(
        &self,
        q: &CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        row_off: usize,
        rows: usize,
        (heads, kv_heads): (usize, usize),
        scale: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .clef_attn_tc
            .ok_or(GpuError::MissingOp("clef_attn_tc"))?;
        let (qw, kw) = (heads * 256, kv_heads * 256);
        if q.len() < (row_off + rows) * qw
            || out.len() < (row_off + rows) * qw
            || k.len() < (row_off + rows) * kw
            || v.len() < (row_off + rows) * kw
        {
            return Err(oob("clef_attn_tc: buffers under the run"));
        }
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = k.device_ptr(&self.stream);
        let (vp, _g3) = v.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        let (qo, ko) = ((row_off * qw * 4) as u64, (row_off * kw * 4) as u64);
        // SAFETY: ABI contract (slot 732); bounds checked above
        check(unsafe {
            f(
                (qp + qo) as *const _,
                qw as u32,
                (kp + ko) as *const _,
                kw as u32,
                (vp + ko) as *const _,
                kw as u32,
                (op + qo) as *mut _,
                qw as u32,
                rows as u32,
                heads as u32,
                kv_heads as u32,
                scale,
                self.stream_ptr(),
            )
        })
    }
}

impl GpuExecutor {
    /// `y = epi(x . W^T)` over `rows` rows on the stored BF16 weight rows
    /// (`w` dims `[K, N]`), the activation split two ways in bf16 (slot
    /// 733). `epi` as slot 722's; `SwiGlu` writes N / 2 wide.
    pub fn clef_gemm(
        &self,
        x: &CudaSlice<f32>,
        w: &QuantTensor,
        y: &mut CudaSlice<f32>,
        rows: usize,
        epi: DiarEpi,
    ) -> Result<(), GpuError> {
        self.clef_gemm_bias(x, w, None, y, rows, epi)
    }

    /// [`Self::clef_gemm`] with `acc + bias` into the epilogue (the vision
    /// tower's linears).
    pub fn clef_gemm_bias(
        &self,
        x: &CudaSlice<f32>,
        w: &QuantTensor,
        bias: Option<&CudaSlice<f32>>,
        y: &mut CudaSlice<f32>,
        rows: usize,
        epi: DiarEpi,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .clef_gemm
            .ok_or(GpuError::MissingOp("clef_gemm"))?;
        let (k, n) = (w.dims[0], w.dims[1]);
        let out = if epi == DiarEpi::SwiGlu { n / 2 } else { n };
        if rows == 0 {
            return Ok(());
        }
        if w.ty != GgmlType::Bf16
            || epi == DiarEpi::Rope
            || k % 32 != 0
            || n % 4 != 0
            || w.bytes.len() < 2 * k * n
            || x.len() < rows * k
            || y.len() < rows * out
            || bias.is_some_and(|b| b.len() < n)
        {
            return Err(oob("clef_gemm: buffers under the GEMM geometry"));
        }
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (wp, _g2) = w.bytes.device_ptr(&self.stream);
        let (yp, _g3) = y.device_ptr_mut(&self.stream);
        let bp = bias.map(|b| b.device_ptr(&self.stream));
        // SAFETY: ABI contract (slot 733); bounds checked above
        check(unsafe {
            f(
                xp as *const _,
                wp as *const _,
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
}

impl GpuExecutor {
    /// Rotate the first 64 of every 256-wide head of `rows` rows in place
    /// from the `(cos, sin)` table `[max_pos][32]` (slot 734); `pos` is the
    /// per-axis positions `[3][rows]` (t, h, w), `hmask` / `wmask` the pairs
    /// that read the h / w axis.
    #[allow(clippy::too_many_arguments)]
    pub fn clef_rope(
        &self,
        x: &mut CudaSlice<f32>,
        heads: usize,
        rows: usize,
        pos: &CudaSlice<u32>,
        table: &CudaSlice<f32>,
        (hmask, wmask): (u32, u32),
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .clef_rope
            .ok_or(GpuError::MissingOp("clef_rope"))?;
        let max_pos = table.len() / 64;
        if x.len() < rows * heads * 256 || pos.len() < 3 * rows || max_pos == 0 {
            return Err(oob("clef_rope: buffers under the geometry"));
        }
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let (pp, _g2) = pos.device_ptr(&self.stream);
        let (tp, _g3) = table.device_ptr(&self.stream);
        // SAFETY: ABI contract (slot 734); bounds checked above, positions
        // clamped to the table in the kernel
        check(unsafe {
            f(
                xp as *mut _,
                heads as u32,
                rows as u32,
                pp as *const _,
                tp as *const _,
                max_pos as u32,
                hmask,
                wmask,
                self.stream_ptr(),
            )
        })
    }
}

/// A projection's Q8_0 weights as slot 741 reads them: int8 rows `[N][K]`
/// and the file's f16 block scales `[K/32][N]` (their raw bits), repacked
/// from the GGUF's 34-byte blocks at load - the same 8.5 bits a weight.
pub struct ClefQ8 {
    pub q: CudaSlice<u8>,
    pub scale: CudaSlice<u8>,
    pub k: usize,
    pub n: usize,
}

impl ClefQ8 {
    /// Repack `n` Q8_0 rows of `k` (each `k / 32` blocks of an f16 scale and
    /// 32 int8) into slot 741's planes, on the host: `(q, scale)`. Values are
    /// the file's bit for bit.
    pub fn repack(blocks: &[u8], n: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
        let nb = k / 32;
        debug_assert_eq!(blocks.len(), n * nb * 34);
        let (mut q, mut scale) = (vec![0u8; n * k], vec![0u8; nb * n * 2]);
        for (i, blk) in blocks.as_chunks::<34>().0.iter().enumerate() {
            let (row, kb) = (i / nb, i % nb);
            scale[(kb * n + row) * 2..(kb * n + row) * 2 + 2].copy_from_slice(&blk[..2]);
            q[row * k + kb * 32..row * k + kb * 32 + 32].copy_from_slice(&blk[2..]);
        }
        (q, scale)
    }

    pub fn bytes(&self) -> u64 {
        (self.q.len() + self.scale.len()) as u64
    }
}

impl GpuExecutor {
    /// Whether the pack carries the GGUF lane's two passes (slots 741-742).
    pub fn has_clef_gguf(&self) -> bool {
        self.has_clef()
            && self.kernels.clef_gemm_q8.is_some()
            && self.kernels.clef_lex_mean_q8.is_some()
    }

    /// [`Self::clef_gemm_bias`] on Q8_0 weights (slot 741): `y = epi(x .
    /// W^T + bias)`, the activation split two ways in bf16 against the exact
    /// int8, each 32-deep block's sum scaled once.
    pub fn clef_gemm_q8(
        &self,
        x: &CudaSlice<f32>,
        w: &ClefQ8,
        bias: Option<&CudaSlice<f32>>,
        y: &mut CudaSlice<f32>,
        rows: usize,
        epi: DiarEpi,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .clef_gemm_q8
            .ok_or(GpuError::MissingOp("clef_gemm_q8"))?;
        let (k, n) = (w.k, w.n);
        let out = if epi == DiarEpi::SwiGlu { n / 2 } else { n };
        if rows == 0 {
            return Ok(());
        }
        if epi == DiarEpi::Rope
            || k % 32 != 0
            || n % 8 != 0
            || w.q.len() < k * n
            || w.scale.len() < k / 32 * n * 2
            || x.len() < rows * k
            || y.len() < rows * out
            || bias.is_some_and(|b| b.len() < n)
        {
            return Err(oob("clef_gemm_q8: buffers under the GEMM geometry"));
        }
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (wp, _g2) = w.q.device_ptr(&self.stream);
        let (sp, _g3) = w.scale.device_ptr(&self.stream);
        let (yp, _g4) = y.device_ptr_mut(&self.stream);
        let bp = bias.map(|b| b.device_ptr(&self.stream));
        // SAFETY: ABI contract (slot 741); bounds checked above
        check(unsafe {
            f(
                xp as *const _,
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

    /// Means of Q8_0 table rows as stored over each span's token ids (slot
    /// 742) - [`Self::clef_lex_mean`] for the GGUF's output embedding.
    pub fn clef_lex_mean_q8(
        &self,
        table: &QuantTensor,
        ids: &CudaSlice<u32>,
        spans: &CudaSlice<u32>,
        n: usize,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .clef_lex_mean_q8
            .ok_or(GpuError::MissingOp("clef_lex_mean_q8"))?;
        let d = table.dims[0];
        if table.ty != GgmlType::Q8_0
            || !d.is_multiple_of(32)
            || spans.len() < 2 * n
            || out.len() < n * d
        {
            return Err(oob("clef_lex_mean_q8: buffers under the geometry"));
        }
        let (tp, _g1) = table.bytes.device_ptr(&self.stream);
        let (ip, _g2) = ids.device_ptr(&self.stream);
        let (sp, _g3) = spans.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 742); ids are checked against the vocab
        // when the pass is built
        check(unsafe {
            f(
                tp as *const _,
                d as u32,
                ip as *const _,
                sp as *const _,
                n as u32,
                op as *mut _,
                self.stream_ptr(),
            )
        })
    }
}

/// A read-only plane and an element offset into it.
pub type ClefIn<'a> = (&'a CudaSlice<f32>, usize);

impl GpuExecutor {
    /// Whether the pack carries Clef's head (slots 724-731, projecting on
    /// 722) and its backbone attention, GEMM and rope (732-734).
    pub fn has_clef(&self) -> bool {
        let k = &self.kernels;
        k.clef_norm.is_some()
            && k.clef_span_mean.is_some()
            && k.clef_lex_mean.is_some()
            && k.clef_attention.is_some()
            && k.clef_route.is_some()
            && k.clef_gather_add.is_some()
            && k.clef_features.is_some()
            && k.clef_score.is_some()
            && k.diar_gemm.is_some()
            && k.clef_attn_tc.is_some()
            && k.clef_gemm.is_some()
            && k.clef_rope.is_some()
    }

    /// LayerNorm (slot 724) of `rows` rows of `d` channels.
    #[allow(clippy::too_many_arguments)]
    pub fn clef_norm(
        &self,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        d: usize,
        rows: usize,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .clef_norm
            .ok_or(GpuError::MissingOp("clef_norm"))?;
        if x.len() < rows * d || y.len() < rows * d || w.len() < d || b.len() < d {
            return Err(oob("clef_norm: buffers under the geometry"));
        }
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let (bp, _g3) = b.device_ptr(&self.stream);
        let (yp, _g4) = y.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 724); bounds checked above
        check(unsafe {
            f(
                xp as *const _,
                wp as *const _,
                bp as *const _,
                yp as *mut _,
                d as u32,
                rows as u32,
                eps,
                self.stream_ptr(),
            )
        })
    }

    /// Means of rows of `x` (row stride `ld`, `d` wide) over `n` spans
    /// (slot 725); `spans` holds `[start, end)` pairs.
    #[allow(clippy::too_many_arguments)]
    pub fn clef_span_mean(
        &self,
        x: &CudaSlice<f32>,
        ld: usize,
        d: usize,
        spans: &CudaSlice<u32>,
        n: usize,
        out: &mut CudaSlice<f32>,
        rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .clef_span_mean
            .ok_or(GpuError::MissingOp("clef_span_mean"))?;
        if x.len() < rows * ld || spans.len() < 2 * n || out.len() < n * d || ld < d {
            return Err(oob("clef_span_mean: buffers under the geometry"));
        }
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (sp, _g2) = spans.device_ptr(&self.stream);
        let (op, _g3) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 725); spans are built inside `rows`
        check(unsafe {
            f(
                xp as *const _,
                ld as u32,
                d as u32,
                sp as *const _,
                n as u32,
                op as *mut _,
                self.stream_ptr(),
            )
        })
    }

    /// Means of BF16 table rows over each span's token ids (slot 726).
    pub fn clef_lex_mean(
        &self,
        table: &QuantTensor,
        ids: &CudaSlice<u32>,
        spans: &CudaSlice<u32>,
        n: usize,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .clef_lex_mean
            .ok_or(GpuError::MissingOp("clef_lex_mean"))?;
        let d = table.dims[0];
        if table.ty != GgmlType::Bf16 || spans.len() < 2 * n || out.len() < n * d {
            return Err(oob("clef_lex_mean: buffers under the geometry"));
        }
        let (tp, _g1) = table.bytes.device_ptr(&self.stream);
        let (ip, _g2) = ids.device_ptr(&self.stream);
        let (sp, _g3) = spans.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 726); ids are checked against the vocab
        // when the pass is built
        check(unsafe {
            f(
                tp as *const _,
                d as u32,
                ip as *const _,
                sp as *const _,
                n as u32,
                op as *mut _,
                self.stream_ptr(),
            )
        })
    }

    /// Attention, heads of 64, every query over its own key rows (slot 727).
    /// q/k/v are a plane and an element offset; the row strides are the
    /// planes' own.
    #[allow(clippy::too_many_arguments)]
    pub fn clef_attention(
        &self,
        (q, qo, ldq): (&CudaSlice<f32>, usize, usize),
        (k, ko, ldk): (&CudaSlice<f32>, usize, usize),
        (v, vo, ldv): (&CudaSlice<f32>, usize, usize),
        out: &mut CudaSlice<f32>,
        ldo: usize,
        ranges: &CudaSlice<u32>,
        nq: usize,
        heads: usize,
        scale: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .clef_attention
            .ok_or(GpuError::MissingOp("clef_attention"))?;
        if nq == 0 {
            return Ok(());
        }
        if q.len() < qo + (nq - 1) * ldq + heads * 64
            || out.len() < (nq - 1) * ldo + heads * 64
            || ranges.len() < 2 * nq
            || ko >= k.len()
            || vo >= v.len()
        {
            return Err(oob("clef_attention: buffers under the geometry"));
        }
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = k.device_ptr(&self.stream);
        let (vp, _g3) = v.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        let (rp, _g5) = ranges.device_ptr(&self.stream);
        // SAFETY: ABI contract (slot 727); key ranges are built inside the
        // key planes' rows
        check(unsafe {
            f(
                (qp + 4 * qo as u64) as *const _,
                ldq as u32,
                (kp + 4 * ko as u64) as *const _,
                ldk as u32,
                (vp + 4 * vo as u64) as *const _,
                ldv as u32,
                op as *mut _,
                ldo as u32,
                rp as *const _,
                nq as u32,
                heads as u32,
                scale,
                self.stream_ptr(),
            )
        })
    }

    /// Each question's softmax-routed option summary (slot 728).
    #[allow(clippy::too_many_arguments)]
    pub fn clef_route(
        &self,
        opts: &CudaSlice<f32>,
        fields: &CudaSlice<f32>,
        qopts: &CudaSlice<u32>,
        nq: usize,
        d: usize,
        max_opts: usize,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .clef_route
            .ok_or(GpuError::MissingOp("clef_route"))?;
        if fields.len() < nq * d || out.len() < nq * d || qopts.len() < 2 * nq {
            return Err(oob("clef_route: buffers under the geometry"));
        }
        let (op, _g1) = opts.device_ptr(&self.stream);
        let (fp, _g2) = fields.device_ptr(&self.stream);
        let (qp, _g3) = qopts.device_ptr(&self.stream);
        let (yp, _g4) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 728)
        check(unsafe {
            f(
                op as *const _,
                fp as *const _,
                qp as *const _,
                nq as u32,
                d as u32,
                max_opts as u32,
                yp as *mut _,
                self.stream_ptr(),
            )
        })
    }

    /// `y[r] += src[idx[r]]` over `rows` rows `d` wide (slot 729).
    pub fn clef_gather_add(
        &self,
        y: &mut CudaSlice<f32>,
        src: &CudaSlice<f32>,
        idx: &CudaSlice<u32>,
        rows: usize,
        d: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .clef_gather_add
            .ok_or(GpuError::MissingOp("clef_gather_add"))?;
        if y.len() < rows * d || idx.len() < rows {
            return Err(oob("clef_gather_add: buffers under the geometry"));
        }
        let (yp, _g1) = y.device_ptr_mut(&self.stream);
        let (sp, _g2) = src.device_ptr(&self.stream);
        let (ip, _g3) = idx.device_ptr(&self.stream);
        // SAFETY: ABI contract (slot 729); indices are built inside `src`
        check(unsafe {
            f(
                yp as *mut _,
                sp as *const _,
                ip as *const _,
                rows as u32,
                d as u32,
                self.stream_ptr(),
            )
        })
    }

    /// The scorer's feature rows `[f, o, f*o, |f-o|]` (slot 730).
    pub fn clef_features(
        &self,
        fields: &CudaSlice<f32>,
        opts: &CudaSlice<f32>,
        qof: &CudaSlice<u32>,
        nopt: usize,
        d: usize,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .clef_features
            .ok_or(GpuError::MissingOp("clef_features"))?;
        if opts.len() < nopt * d || out.len() < nopt * 4 * d || qof.len() < nopt {
            return Err(oob("clef_features: buffers under the geometry"));
        }
        let (fp, _g1) = fields.device_ptr(&self.stream);
        let (op, _g2) = opts.device_ptr(&self.stream);
        let (qp, _g3) = qof.device_ptr(&self.stream);
        let (yp, _g4) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 730)
        check(unsafe {
            f(
                fp as *const _,
                op as *const _,
                qp as *const _,
                nopt as u32,
                d as u32,
                yp as *mut _,
                self.stream_ptr(),
            )
        })
    }
}

/// The scorer's inputs (slot 731), one plane each.
pub struct ClefScoreIn<'a> {
    pub lex: &'a CudaSlice<f32>,
    pub qvec: &'a CudaSlice<f32>,
    pub glob: &'a CudaSlice<f32>,
    pub qof: &'a CudaSlice<u32>,
    pub rof: &'a CudaSlice<u32>,
    pub fields: &'a CudaSlice<f32>,
    pub opts: &'a CudaSlice<f32>,
    pub hid: &'a CudaSlice<f32>,
    pub w3: &'a CudaSlice<f32>,
    pub b3: f32,
    /// backbone width, head width
    pub dd: usize,
    pub w: usize,
    /// the head's scalars after their clamp/exp and sigmoid
    pub prior_scale: f32,
    pub joint_scale: f32,
    pub gate: f32,
}

impl GpuExecutor {
    /// Every option's logit (slot 731).
    pub fn clef_score(
        &self,
        s: &ClefScoreIn<'_>,
        nopt: usize,
        logits: &mut CudaSlice<f32>,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .clef_score
            .ok_or(GpuError::MissingOp("clef_score"))?;
        if logits.len() < nopt || s.lex.len() < nopt * s.dd || s.hid.len() < nopt * s.w {
            return Err(oob("clef_score: buffers under the geometry"));
        }
        let (lp, _g1) = s.lex.device_ptr(&self.stream);
        let (qp, _g2) = s.qvec.device_ptr(&self.stream);
        let (gp, _g3) = s.glob.device_ptr(&self.stream);
        let (qop, _g4) = s.qof.device_ptr(&self.stream);
        let (rop, _g5) = s.rof.device_ptr(&self.stream);
        let (fp, _g6) = s.fields.device_ptr(&self.stream);
        let (op, _g7) = s.opts.device_ptr(&self.stream);
        let (hp, _g8) = s.hid.device_ptr(&self.stream);
        let (wp, _g9) = s.w3.device_ptr(&self.stream);
        let (yp, _g10) = logits.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 731); the index planes are built inside
        // the question and request planes
        check(unsafe {
            f(
                lp as *const _,
                qp as *const _,
                gp as *const _,
                qop as *const _,
                rop as *const _,
                fp as *const _,
                op as *const _,
                hp as *const _,
                wp as *const _,
                s.b3,
                s.dd as u32,
                s.w as u32,
                s.prior_scale,
                s.joint_scale,
                s.gate,
                nopt as u32,
                yp as *mut _,
                self.stream_ptr(),
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::ClefQ8;

    /// Q8_0 blocks (f16 scale, 32 int8) become int8 rows and the scales
    /// `[K/32][N]`, every byte the file's.
    #[test]
    fn q8_repack_keeps_the_files_bytes() {
        let (n, k) = (3usize, 64usize);
        let mut blocks = Vec::new();
        for row in 0..n {
            for kb in 0..k / 32 {
                let d = half::f16::from_f32(0.5 + (row * 2 + kb) as f32).to_le_bytes();
                blocks.extend_from_slice(&d);
                blocks.extend((0..32).map(|j| (row * 64 + kb * 32 + j) as u8));
            }
        }
        let (q, scale) = ClefQ8::repack(&blocks, n, k);
        assert_eq!(q.len(), n * k);
        assert_eq!(scale.len(), k / 32 * n * 2);
        for row in 0..n {
            for c in 0..k {
                assert_eq!(q[row * k + c], (row * 64 + c) as u8);
            }
            for kb in 0..k / 32 {
                let at = (kb * n + row) * 2;
                let d = half::f16::from_le_bytes([scale[at], scale[at + 1]]).to_f32();
                assert_eq!(d, 0.5 + (row * 2 + kb) as f32);
            }
        }
    }
}
