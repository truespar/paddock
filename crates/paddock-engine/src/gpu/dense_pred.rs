//! Dense-prediction ops - a LayerScale ViT backbone (DINOv3) and the
//! convolutional decoder that turns its patch grid back into a raster. Kernel
//! side: `packs/cuda/src/dense_pred.cuh`, slots 610-617, and the half
//! activation interface of the backbone, slots 618-623 (the f16-landing GEMM
//! in `gemm/f16_dense.cuh`, the f16 attention in `vision.cuh`, three seams).
//!
//! The unit of work is a chip, never a token. Every plane is
//! `[chips][rows][channels]` with the channel innermost (NHWC for the
//! decoder), so a batch of chips is pure row count and the GEMMs are the
//! ordinary row-batched f16 tensor-core ones. What lives here is everything
//! between those GEMMs that no other family needed: a table rope, the
//! LayerScale seam, a real GroupNorm, and the two conv reshapes.
//!
//! Buffers are checked against the geometry in Rust before any launch - a
//! short slice is a logic error here, not something to hand the driver.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::f16;

use super::error::*;
use super::*;

/// Pixels folded per group-norm partial - must match `PD_DP_GN_Q`.
const GN_CHUNK: usize = 64;

impl GpuExecutor {
    /// True when the loaded pack carries the whole dense-prediction lane. The
    /// slots landed together, so one missing means a pack older than the lane.
    pub fn has_dense_pred(&self) -> bool {
        let k = &self.kernels;
        k.dp_u8_patch_rows.is_some()
            && k.dp_qkv_split_rope.is_some()
            && k.dp_res_ls_ln_f16.is_some()
            && k.dp_group_norm_gelu_f16.is_some()
            && k.dp_im2row3_f32.is_some()
            && k.dp_im2row3_f16.is_some()
            && k.dp_convt2_skip.is_some()
            && k.dp_seg_heads.is_some()
    }

    /// f32 scratch elements [`Self::dp_group_norm_gelu_f16`] needs for its
    /// partials at this geometry.
    pub fn dp_group_norm_part_len(chips: usize, pixels: usize, channels: usize) -> usize {
        chips * pixels.div_ceil(GN_CHUNK) * channels
    }

    /// The patch stem: u8 HWC chips of `mean.len()` bands to normalized f16
    /// patch rows `[chips * chip_rows, bands * patch^2]` in the conv stem's
    /// im2row order. Each chip owns `chip_rows` rows; the ones past its patch
    /// grid come back zero, which is where the class and register tokens are
    /// added afterwards.
    #[allow(clippy::too_many_arguments)]
    pub fn dp_u8_patch_rows(
        &self,
        pixels: &CudaSlice<u8>,
        out: &mut CudaSlice<f16>,
        mean: &[f32],
        std: &[f32],
        chips: usize,
        px: usize,
        patch: usize,
        chip_rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .dp_u8_patch_rows
            .ok_or(GpuError::MissingOp("dp_u8_patch_rows"))?;
        let ch = mean.len();
        if ch == 0 || ch > 4 || std.len() != ch {
            return Err(oob("dp_u8_patch_rows: 1..=4 bands, one mean and std each"));
        }
        if pixels.len() < chips * px * px * ch || out.len() < chips * chip_rows * ch * patch * patch
        {
            return Err(oob("dp_u8_patch_rows: buffers under the chip geometry"));
        }
        let at = |v: &[f32], i: usize, d: f32| v.get(i).copied().unwrap_or(d);
        let (sp, _g1) = pixels.device_ptr(&self.stream);
        let (dp, _g2) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; both buffers cover the geometry checked above
        check(unsafe {
            f(
                sp as *const _,
                dp as *mut _,
                at(mean, 0, 0.0),
                at(mean, 1, 0.0),
                at(mean, 2, 0.0),
                at(mean, 3, 0.0),
                at(std, 0, 1.0),
                at(std, 1, 1.0),
                at(std, 2, 1.0),
                at(std, 3, 1.0),
                chips as u32,
                px as u32,
                patch as u32,
                ch as u32,
                chip_rows as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Split the fused q|k|v landing `[rows, 3d]` into the three attention
    /// planes, folding the q and v biases, and rope q and k from a cos/sin
    /// table `[n_rope, hd/2]` on the first `n_rope` rows of every
    /// `chip_rows`-row chip (rotate_half pairs). Rows past `n_rope` - the
    /// class and register tokens - are split and biased, never roped.
    #[allow(clippy::too_many_arguments)]
    pub fn dp_qkv_split_rope(
        &self,
        qkv: &CudaSlice<f32>,
        bq: &CudaSlice<f32>,
        bv: &CudaSlice<f32>,
        cos: &CudaSlice<f32>,
        sin: &CudaSlice<f32>,
        q: &mut CudaSlice<f32>,
        k: &mut CudaSlice<f32>,
        v: &mut CudaSlice<f32>,
        d: usize,
        head_dim: usize,
        rows: usize,
        chip_rows: usize,
        n_rope: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .dp_qkv_split_rope
            .ok_or(GpuError::MissingOp("dp_qkv_split_rope"))?;
        let tab = n_rope * head_dim / 2;
        if qkv.len() < rows * 3 * d
            || q.len() < rows * d
            || k.len() < rows * d
            || v.len() < rows * d
            || bq.len() < d
            || bv.len() < d
            || cos.len() < tab
            || sin.len() < tab
        {
            return Err(oob("dp_qkv_split_rope: buffers under the row geometry"));
        }
        let (sp, _g1) = qkv.device_ptr(&self.stream);
        let (bqp, _g2) = bq.device_ptr(&self.stream);
        let (bvp, _g3) = bv.device_ptr(&self.stream);
        let (cp, _g4) = cos.device_ptr(&self.stream);
        let (snp, _g5) = sin.device_ptr(&self.stream);
        let (qp, _g6) = q.device_ptr_mut(&self.stream);
        let (kp, _g7) = k.device_ptr_mut(&self.stream);
        let (vp, _g8) = v.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; every plane covers rows x its width (above)
        check(unsafe {
            f(
                sp as *const _,
                bqp as *const _,
                bvp as *const _,
                cp as *const _,
                snp as *const _,
                qp as *mut _,
                kp as *mut _,
                vp as *mut _,
                d as u32,
                head_dim as u32,
                rows as u32,
                chip_rows as u32,
                n_rope as u32,
                self.stream_ptr(),
            )
        })
    }

    /// The LayerScale residual seam: `x += ls * (proj + bias)`, then the next
    /// norm out of the updated residual at f16 - one launch for what would be
    /// a bias add, a per-channel multiply, a residual add, a LayerNorm and a
    /// convert.
    #[allow(clippy::too_many_arguments)]
    pub fn dp_res_ls_ln_f16(
        &self,
        x: &mut CudaSlice<f32>,
        proj: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        ls: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        out: &mut CudaSlice<f16>,
        rows: usize,
        n: usize,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .dp_res_ls_ln_f16
            .ok_or(GpuError::MissingOp("dp_res_ls_ln_f16"))?;
        if x.len() < rows * n
            || proj.len() < rows * n
            || out.len() < rows * n
            || bias.len() < n
            || ls.len() < n
            || w.len() < n
            || b.len() < n
        {
            return Err(oob("dp_res_ls_ln_f16: buffers under rows x n"));
        }
        let (pp, _g1) = proj.device_ptr(&self.stream);
        let (bip, _g2) = bias.device_ptr(&self.stream);
        let (lp, _g3) = ls.device_ptr(&self.stream);
        let (wp, _g4) = w.device_ptr(&self.stream);
        let (bp, _g5) = b.device_ptr(&self.stream);
        let (xp, _g6) = x.device_ptr_mut(&self.stream);
        let (op, _g7) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; bounds checked above
        check(unsafe {
            f(
                xp as *mut _,
                pp as *const _,
                bip as *const _,
                lp as *const _,
                wp as *const _,
                bp as *const _,
                op as *mut _,
                rows as u32,
                n as u32,
                eps,
                self.stream_ptr(),
            )
        })
    }

    /// torch `GroupNorm(groups, channels)` over an NHWC plane
    /// `[chips][pixels][channels]`, then exact-erf GELU, written at f16 for
    /// the GEMM that follows. `x` is the producing conv's GEMM landing and
    /// `xb` that conv's bias, added at the load so the plane never pays a
    /// bias pass of its own. Statistics run over every pixel of a chip and
    /// the group's channels, biased variance. `part` and `stat` are scratch:
    /// [`Self::dp_group_norm_part_len`] and `2 * chips * groups` floats.
    #[allow(clippy::too_many_arguments)]
    pub fn dp_group_norm_gelu_f16(
        &self,
        x: &CudaSlice<f32>,
        xb: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        out: &mut CudaSlice<f16>,
        part: &mut CudaSlice<f32>,
        stat: &mut CudaSlice<f32>,
        chips: usize,
        pixels: usize,
        channels: usize,
        groups: usize,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .dp_group_norm_gelu_f16
            .ok_or(GpuError::MissingOp("dp_group_norm_gelu_f16"))?;
        let n = chips * pixels * channels;
        if groups == 0 || !channels.is_multiple_of(groups) {
            return Err(oob(
                "dp_group_norm_gelu_f16: channels not divisible by groups",
            ));
        }
        if x.len() < n
            || out.len() < n
            || xb.len() < channels
            || w.len() < channels
            || b.len() < channels
            || part.len() < Self::dp_group_norm_part_len(chips, pixels, channels)
            || stat.len() < 2 * chips * groups
        {
            return Err(oob(
                "dp_group_norm_gelu_f16: buffers under the plane geometry",
            ));
        }
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (xbp, _g0) = xb.device_ptr(&self.stream);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let (bp, _g3) = b.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        let (pp, _g5) = part.device_ptr_mut(&self.stream);
        let (sp, _g6) = stat.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; plane and both scratch sizes checked above
        check(unsafe {
            f(
                xp as *const _,
                xbp as *const _,
                wp as *const _,
                bp as *const _,
                op as *mut _,
                pp as *mut _,
                sp as *mut _,
                chips as u32,
                pixels as u32,
                channels as u32,
                groups as u32,
                eps,
                self.stream_ptr(),
            )
        })
    }

    /// Whether [`Self::f16_conv3_gemm`] is elected: the pack has slot 775 and
    /// the die is cc 8.x, where its ring is the f16 GEMM's own and the landing
    /// is bit-identical to [`Self::dp_im2row3_f16`] + `matvec_batch_f16`. The
    /// Blackwell f16 entry takes tcgen05 arms the conv entry does not have, so
    /// those dies keep the explicit form until it is measured there.
    pub fn f16_conv3_elected(&self) -> bool {
        self.kernels.f16_conv3_gemm.is_some() && self.compute_capability().0 == 8
    }

    /// A 3x3 / stride 1 / zero-pad 1 convolution as one GEMM with the im2row
    /// gathered in the stage (slot 775): `src` f16 `[chips][src_chip_rows][c]`,
    /// `wt` the im2row weight (`[9 c][out]`, tap-outer), `y` f32
    /// `[chips * h * w][out]`, `bias` added in the landing when given (the
    /// `bias_add` pass, folded). `c` a multiple of 8.
    #[allow(clippy::too_many_arguments)]
    pub fn f16_conv3_gemm(
        &self,
        wt: &HalfTensor,
        src: &CudaSlice<f16>,
        y: &mut CudaSlice<f32>,
        bias: Option<&CudaSlice<f32>>,
        chips: usize,
        h: usize,
        w: usize,
        c: usize,
        src_chip_rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .f16_conv3_gemm
            .ok_or(GpuError::MissingOp("f16_conv3_gemm"))?;
        let out = wt.dims[1];
        if wt.dims[0] != 9 * c
            || src_chip_rows < h * w
            || src.len() < chips * src_chip_rows * c
            || y.len() < chips * h * w * out
            || bias.is_some_and(|b| b.len() < out)
        {
            return Err(oob("f16_conv3_gemm: buffers under the conv geometry"));
        }
        let (wp, _g1) = wt.buf.device_ptr(&self.stream);
        let (sp, _g2) = src.device_ptr(&self.stream);
        let (yp, _g3) = y.device_ptr_mut(&self.stream);
        let bg = bias.map(|b| b.device_ptr(&self.stream));
        let bp = bg.as_ref().map_or(0, |(p, _)| *p);
        // SAFETY: ABI contract (slot 775); bounds checked above, null bias = none
        check(unsafe {
            f(
                wp as *const _,
                sp as *const _,
                yp as *mut _,
                bp as *const _,
                chips as u32,
                h as u32,
                w as u32,
                c as u32,
                out as u32,
                src_chip_rows as u32,
                self.stream_ptr(),
            )
        })
    }

    /// 3x3 / stride 1 / zero-pad 1 im2row off an f32 NHWC plane into the f16
    /// staging `[chips*h*w, 9*c]` its GEMM eats, TAP-outer columns (the weight
    /// is permuted to match at load). `src_chip_rows` is the row stride
    /// between chips in the source, so a `[chips][tokens]` plane whose chips
    /// carry extra rows after their `h*w` grid is read in place.
    #[allow(clippy::too_many_arguments)]
    pub fn dp_im2row3_f32(
        &self,
        src: &CudaSlice<f32>,
        out: &mut CudaSlice<f16>,
        chips: usize,
        h: usize,
        w: usize,
        c: usize,
        src_chip_rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .dp_im2row3_f32
            .ok_or(GpuError::MissingOp("dp_im2row3_f32"))?;
        if src_chip_rows < h * w
            || src.len() < chips * src_chip_rows * c
            || out.len() < chips * h * w * 9 * c
        {
            return Err(oob("dp_im2row3_f32: buffers under the plane geometry"));
        }
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (op, _g2) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; bounds checked above
        check(unsafe {
            f(
                sp as *const _,
                op as *mut _,
                chips as u32,
                h as u32,
                w as u32,
                c as u32,
                src_chip_rows as u32,
                self.stream_ptr(),
            )
        })
    }

    /// [`Self::dp_im2row3_f32`] over an f16 source - the plane a group norm
    /// just wrote - which makes it a pure gather.
    #[allow(clippy::too_many_arguments)]
    pub fn dp_im2row3_f16(
        &self,
        src: &CudaSlice<f16>,
        out: &mut CudaSlice<f16>,
        chips: usize,
        h: usize,
        w: usize,
        c: usize,
        src_chip_rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .dp_im2row3_f16
            .ok_or(GpuError::MissingOp("dp_im2row3_f16"))?;
        if src_chip_rows < h * w
            || src.len() < chips * src_chip_rows * c
            || out.len() < chips * h * w * 9 * c
        {
            return Err(oob("dp_im2row3_f16: buffers under the plane geometry"));
        }
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (op, _g2) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; bounds checked above
        check(unsafe {
            f(
                sp as *const _,
                op as *mut _,
                chips as u32,
                h as u32,
                w as u32,
                c as u32,
                src_chip_rows as u32,
                self.stream_ptr(),
            )
        })
    }

    /// The decoder's stage seam: depth-to-space of a 2x2/stride-2 transposed
    /// conv's GEMM landing `g` `[chips][h*w][4*c]` (tap-major), plus `bias`,
    /// plus the `hs x hs` skip grid sampled bilinearly (align_corners=False) at
    /// the `2h x 2w` output - `out` is `[chips][4*h*w][c]`. `skip` chips are
    /// strided by `skip_chip_rows` rows.
    #[allow(clippy::too_many_arguments)]
    pub fn dp_convt2_skip(
        &self,
        g: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        skip: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        chips: usize,
        h: usize,
        w: usize,
        c: usize,
        hs: usize,
        skip_chip_rows: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .dp_convt2_skip
            .ok_or(GpuError::MissingOp("dp_convt2_skip"))?;
        if skip_chip_rows < hs * hs
            || g.len() < chips * h * w * 4 * c
            || out.len() < chips * 4 * h * w * c
            || skip.len() < chips * skip_chip_rows * c
            || bias.len() < c
        {
            return Err(oob("dp_convt2_skip: buffers under the stage geometry"));
        }
        let (gp, _g1) = g.device_ptr(&self.stream);
        let (bp, _g2) = bias.device_ptr(&self.stream);
        let (sp, _g3) = skip.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; bounds checked above
        check(unsafe {
            f(
                gp as *const _,
                bp as *const _,
                sp as *const _,
                op as *mut _,
                chips as u32,
                h as u32,
                w as u32,
                c as u32,
                hs as u32,
                skip_chip_rows as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Both output heads off their stacked GEMM landing `o`
    /// `[rows][ncls + 1]`: biased argmax to a u8 class raster (lowest index
    /// wins a tie), the last column plus its bias to the regression raster,
    /// and - when asked for - the biased class logits at f16.
    #[allow(clippy::too_many_arguments)]
    pub fn dp_seg_heads(
        &self,
        o: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        cls: &mut CudaSlice<u8>,
        height: &mut CudaSlice<f32>,
        logits: Option<&mut CudaSlice<f16>>,
        rows: usize,
        ncls: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .dp_seg_heads
            .ok_or(GpuError::MissingOp("dp_seg_heads"))?;
        if o.len() < rows * (ncls + 1)
            || bias.len() < ncls + 1
            || cls.len() < rows
            || height.len() < rows
            || logits.as_ref().is_some_and(|l| l.len() < rows * ncls)
        {
            return Err(oob("dp_seg_heads: buffers under rows x classes"));
        }
        let (op, _g1) = o.device_ptr(&self.stream);
        let (bp, _g2) = bias.device_ptr(&self.stream);
        let (cp, _g3) = cls.device_ptr_mut(&self.stream);
        let (hp, _g4) = height.device_ptr_mut(&self.stream);
        let lg = logits.map(|l| l.device_ptr_mut(&self.stream));
        let lp = lg.as_ref().map_or(0, |(p, _)| *p);
        // SAFETY: ABI contract; bounds checked above, NULL logits is the
        // documented "skip" value
        check(unsafe {
            f(
                op as *const _,
                bp as *const _,
                cp as *mut _,
                hp as *mut _,
                lp as *mut _,
                rows as u32,
                ncls as u32,
                self.stream_ptr(),
            )
        })
    }
}

// ---- the half activation interface (slots 618-623) -------------------------
// Between a ViT block's GEMMs everything is a walk over a plane, and every one
// of those walks measured at the DRAM roof - so the planes went to 2 bytes an
// element: the GEMMs land f16, the seams read f16, attention reads and writes
// f16. The residual stream and every accumulate stay f32.
impl GpuExecutor {
    /// True when the pack carries the whole half interface. Landed together.
    pub fn has_dense_pred_h(&self) -> bool {
        let k = &self.kernels;
        k.f16_gemm_h.is_some()
            && k.f16_gemm_h_elected.is_some()
            && k.vision_attn_h.is_some()
            && k.dp_qkv_split_rope_h.is_some()
            && k.dp_res_ls_ln_h.is_some()
            && k.dp_gelu_bias_h.is_some()
    }

    /// Whether the f16-landing GEMM is this device's elected wide-batch route.
    /// False where the f32 entry owns an arm the landing has no twin of
    /// (tcgen05 on cc 10.0): a tower there keeps `matvec_batch_f16` and pays a
    /// convert. Ask once, at load.
    pub fn f16_gemm_h_elected(&self) -> bool {
        match self.kernels.f16_gemm_h_elected {
            // SAFETY: ABI contract; no arguments, reads device attributes only
            Some(f) => unsafe { f() != 0 },
            None => false,
        }
    }

    /// `y16 = W x` with the landing at f16: `w` is `[in, out]` f16, `x16`
    /// `[batch, in]`, `y16` `[batch, out]`. Same ring and accumulation order as
    /// [`Self::matvec_batch_f16`], so this is that result rounded to nearest.
    /// `in` must be a multiple of 8 (the ring stages 16-byte units).
    pub fn matvec_batch_f16_h(
        &self,
        w: &HalfTensor,
        x16: &CudaSlice<f16>,
        y16: &mut CudaSlice<f16>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .f16_gemm_h
            .ok_or(GpuError::MissingOp("f16_gemm_h"))?;
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        if in_dim % 8 != 0 {
            return Err(oob("f16_gemm_h: in_dim must be a multiple of 8"));
        }
        if w.buf.len() < in_dim * out_dim
            || x16.len() < batch * in_dim
            || y16.len() < batch * out_dim
        {
            return Err(oob("f16_gemm_h: buffers under the GEMM geometry"));
        }
        super::basic_ops::gemm_census("B-gemm-f16-h", in_dim, out_dim, batch);
        let (wp, _g1) = w.buf.device_ptr(&self.stream);
        let (xp, _g2) = x16.device_ptr(&self.stream);
        let (yp, _g3) = y16.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 618); bounds checked above
        check(unsafe {
            f(
                wp as *const _,
                xp as *const _,
                yp as *mut _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Whether the pack lands the FFN's bias + GELU inside the GEMM (slot 624).
    pub fn has_f16_gemm_h_gelu(&self) -> bool {
        self.kernels.f16_gemm_h_gelu.is_some()
    }

    /// [`Self::matvec_batch_f16_h`] with `bias` (`[out]` f32) added and the
    /// exact GELU applied in the epilogue, before the one round to f16 -
    /// what [`Self::dp_gelu_bias_h`] did to the landed plane, minus the pass
    /// and minus the intermediate round.
    pub fn matvec_batch_f16_h_gelu(
        &self,
        w: &HalfTensor,
        x16: &CudaSlice<f16>,
        y16: &mut CudaSlice<f16>,
        bias: &CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .f16_gemm_h_gelu
            .ok_or(GpuError::MissingOp("f16_gemm_h_gelu"))?;
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        if in_dim % 8 != 0 {
            return Err(oob("f16_gemm_h_gelu: in_dim must be a multiple of 8"));
        }
        if w.buf.len() < in_dim * out_dim
            || x16.len() < batch * in_dim
            || y16.len() < batch * out_dim
            || bias.len() < out_dim
        {
            return Err(oob("f16_gemm_h_gelu: buffers under the GEMM geometry"));
        }
        super::basic_ops::gemm_census("B-gemm-f16-h-gelu", in_dim, out_dim, batch);
        let (wp, _g1) = w.buf.device_ptr(&self.stream);
        let (xp, _g2) = x16.device_ptr(&self.stream);
        let (bp, _g3) = bias.device_ptr(&self.stream);
        let (yp, _g4) = y16.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 624); bounds checked above
        check(unsafe {
            f(
                wp as *const _,
                xp as *const _,
                yp as *mut _,
                bp as *const _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Bidirectional attention over `n_batch` independent groups, all four
    /// planes f16 `[batch][row][head][dim]`. `q` arrives PRE-SCALED by
    /// `1/sqrt(head_dim)` - [`Self::dp_qkv_split_rope_h`] folds it in before
    /// its round, which is the one round the f32 kernel does on its own q.
    /// Bit-for-bit [`Self::vision_attn_x`]'s mma result rounded to nearest.
    #[allow(clippy::too_many_arguments)]
    pub fn vision_attn_h(
        &self,
        q: &CudaSlice<f16>,
        k: &CudaSlice<f16>,
        v: &CudaSlice<f16>,
        out: &mut CudaSlice<f16>,
        nq: usize,
        nkv: usize,
        n_heads: usize,
        head_dim: usize,
        n_batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .vision_attn_h
            .ok_or(GpuError::MissingOp("vision_attn_h"))?;
        let (qn, kn) = (
            n_batch * nq * n_heads * head_dim,
            n_batch * nkv * n_heads * head_dim,
        );
        if q.len() < qn || out.len() < qn || k.len() < kn || v.len() < kn {
            return Err(oob("vision_attn_h: buffers under the attention geometry"));
        }
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = k.device_ptr(&self.stream);
        let (vp, _g3) = v.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 620); bounds checked above
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                op as *mut _,
                nq as u32,
                nkv as u32,
                n_heads as u32,
                head_dim as u32,
                n_batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// [`Self::dp_qkv_split_rope`] off a half qkv landing into half q/k/v, with
    /// `qscale` (1/sqrt(head_dim)) folded into q before its round.
    #[allow(clippy::too_many_arguments)]
    pub fn dp_qkv_split_rope_h(
        &self,
        qkv: &CudaSlice<f16>,
        bq: &CudaSlice<f32>,
        bv: &CudaSlice<f32>,
        cos: &CudaSlice<f32>,
        sin: &CudaSlice<f32>,
        q: &mut CudaSlice<f16>,
        k: &mut CudaSlice<f16>,
        v: &mut CudaSlice<f16>,
        d: usize,
        head_dim: usize,
        rows: usize,
        chip_rows: usize,
        n_rope: usize,
        qscale: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .dp_qkv_split_rope_h
            .ok_or(GpuError::MissingOp("dp_qkv_split_rope_h"))?;
        let tab = n_rope * head_dim / 2;
        if qkv.len() < rows * 3 * d
            || q.len() < rows * d
            || k.len() < rows * d
            || v.len() < rows * d
            || bq.len() < d
            || bv.len() < d
            || cos.len() < tab
            || sin.len() < tab
        {
            return Err(oob("dp_qkv_split_rope_h: buffers under the row geometry"));
        }
        let (sp, _g1) = qkv.device_ptr(&self.stream);
        let (bqp, _g2) = bq.device_ptr(&self.stream);
        let (bvp, _g3) = bv.device_ptr(&self.stream);
        let (cp, _g4) = cos.device_ptr(&self.stream);
        let (snp, _g5) = sin.device_ptr(&self.stream);
        let (qp, _g6) = q.device_ptr_mut(&self.stream);
        let (kp, _g7) = k.device_ptr_mut(&self.stream);
        let (vp, _g8) = v.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 621); bounds checked above
        check(unsafe {
            f(
                sp as *const _,
                bqp as *const _,
                bvp as *const _,
                cp as *const _,
                snp as *const _,
                qp as *mut _,
                kp as *mut _,
                vp as *mut _,
                d as u32,
                head_dim as u32,
                rows as u32,
                chip_rows as u32,
                n_rope as u32,
                qscale,
                self.stream_ptr(),
            )
        })
    }

    /// [`Self::dp_res_ls_ln_f16`] off a half projection landing.
    #[allow(clippy::too_many_arguments)]
    pub fn dp_res_ls_ln_h(
        &self,
        x: &mut CudaSlice<f32>,
        proj: &CudaSlice<f16>,
        bias: &CudaSlice<f32>,
        ls: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        out: &mut CudaSlice<f16>,
        rows: usize,
        n: usize,
        eps: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .dp_res_ls_ln_h
            .ok_or(GpuError::MissingOp("dp_res_ls_ln_h"))?;
        if x.len() < rows * n
            || proj.len() < rows * n
            || out.len() < rows * n
            || bias.len() < n
            || ls.len() < n
            || w.len() < n
            || b.len() < n
        {
            return Err(oob("dp_res_ls_ln_h: buffers under rows x n"));
        }
        let (pp, _g1) = proj.device_ptr(&self.stream);
        let (bip, _g2) = bias.device_ptr(&self.stream);
        let (lp, _g3) = ls.device_ptr(&self.stream);
        let (wp, _g4) = w.device_ptr(&self.stream);
        let (bp, _g5) = b.device_ptr(&self.stream);
        let (xp, _g6) = x.device_ptr_mut(&self.stream);
        let (op, _g7) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 622); bounds checked above
        check(unsafe {
            f(
                xp as *mut _,
                pp as *const _,
                bip as *const _,
                lp as *const _,
                wp as *const _,
                bp as *const _,
                op as *mut _,
                rows as u32,
                n as u32,
                eps,
                self.stream_ptr(),
            )
        })
    }

    /// `x[r][i] = gelu(x[r][i] + bias[i])` on a half plane, in place - exact
    /// erf GELU computed in f32, one round at the store.
    pub fn dp_gelu_bias_h(
        &self,
        x: &mut CudaSlice<f16>,
        bias: &CudaSlice<f32>,
        rows: usize,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .dp_gelu_bias_h
            .ok_or(GpuError::MissingOp("dp_gelu_bias_h"))?;
        if x.len() < rows * n || bias.len() < n {
            return Err(oob("dp_gelu_bias_h: buffers under rows x n"));
        }
        let (bp, _g1) = bias.device_ptr(&self.stream);
        let (xp, _g2) = x.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 623); bounds checked above
        check(unsafe {
            f(
                xp as *mut _,
                bp as *const _,
                rows as u32,
                n as u32,
                self.stream_ptr(),
            )
        })
    }
}
