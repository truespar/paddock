//! SAM 3's image-encoder ops - the ViT backbone's window-major glue and the
//! FPN necks' transposed-conv seam. Kernel side: `packs/cuda/src/sam3/`,
//! slots 751-754 and 759-768, and the tanh-GELU and bias landings of `gemm/f16_dense.cuh`,
//! slots 755 and 769.
//!
//! Everything else the image encoder runs is the dense-prediction lane's
//! (`dense_pred.rs`: the f16-landing GEMM, `vision_attn_h`, the residual
//! seam, the 3x3 im2row) on the same precision class. What is here is only
//! what SAM 3 does differently: token rows in window-major order (so window
//! attention is plain grouped attention and nothing is ever partitioned), a
//! q|k|v projection with a bias on all three, the tanh GELU Meta's inference
//! fuses into the ViT's fc1, and a convT seam with no skip.
//!
//! Buffers are checked against the geometry in Rust before any launch - a
//! short slice is a logic error here, not something to hand the driver.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use half::f16;

use super::error::*;
use super::*;

impl GpuExecutor {
    /// True when the loaded pack carries SAM 3's image-encoder lane (slots
    /// 751-755, landed together) on top of the dense-prediction one.
    pub fn has_sam3_vision(&self) -> bool {
        let k = &self.kernels;
        k.sam3_patch_rows.is_some()
            && k.sam3_qkv_split_rope_h.is_some()
            && k.sam3_rows_to_raster_h.is_some()
            && k.sam3_convt2_bias_h.is_some()
            && k.f16_gemm_h_gelu_tanh.is_some()
    }

    /// The text tower's one op of its own (slot 759), on top of the image
    /// encoder's lane.
    pub fn has_sam3_text(&self) -> bool {
        self.has_sam3_vision() && self.kernels.sam3_text_attn_h.is_some()
    }

    /// The patch stem: `pics` u8 RGB HWC pictures at `side x side` to the
    /// patch GEMM's f16 rows `[pics * g * g, kp]`, rows window-major (window
    /// `r / win^2`, cell `r % win^2`), columns `c * p * p + ky * p + kx` and
    /// zero past `ch * p * p`.
    #[allow(clippy::too_many_arguments)]
    pub fn sam3_patch_rows(
        &self,
        pixels: &CudaSlice<u8>,
        out: &mut CudaSlice<f16>,
        pics: usize,
        side: usize,
        patch: usize,
        win: usize,
        ch: usize,
        kp: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sam3_patch_rows
            .ok_or(GpuError::MissingOp("sam3_patch_rows"))?;
        if patch == 0 || !side.is_multiple_of(patch) || kp < ch * patch * patch {
            return Err(oob("sam3_patch_rows: geometry"));
        }
        let g = side / patch;
        if win == 0 || !g.is_multiple_of(win) {
            return Err(oob(
                "sam3_patch_rows: the grid is not a whole number of windows",
            ));
        }
        if pixels.len() < pics * side * side * ch || out.len() < pics * g * g * kp {
            return Err(oob("sam3_patch_rows: buffers under the picture geometry"));
        }
        let (pp, _g1) = pixels.device_ptr(&self.stream);
        let (op, _g2) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 751); bounds checked above
        check(unsafe {
            f(
                pp as *const _,
                op as *mut _,
                pics as u32,
                side as u32,
                patch as u32,
                win as u32,
                ch as u32,
                kp as u32,
                self.stream_ptr(),
            )
        })
    }

    /// The fused q|k|v half landing `[rows, 3d]` to the three half planes
    /// [`Self::vision_attn_h`] eats, every bias added and the rope applied to
    /// q and k on every row: row `r` reads row `r % chip_rows` of the
    /// `[chip_rows, head_dim / 2]` cos/sin tables. q is scaled by `qscale`
    /// before its round. `rope: None` is the split without any rotation - the
    /// text tower's, whose positions are learned and added up front.
    #[allow(clippy::too_many_arguments)]
    pub fn sam3_qkv_split_rope_h(
        &self,
        qkv: &CudaSlice<f16>,
        bq: &CudaSlice<f32>,
        bk: &CudaSlice<f32>,
        bv: &CudaSlice<f32>,
        rope: Option<(&CudaSlice<f32>, &CudaSlice<f32>)>,
        q: &mut CudaSlice<f16>,
        k: &mut CudaSlice<f16>,
        v: &mut CudaSlice<f16>,
        d: usize,
        head_dim: usize,
        rows: usize,
        chip_rows: usize,
        qscale: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sam3_qkv_split_rope_h
            .ok_or(GpuError::MissingOp("sam3_qkv_split_rope_h"))?;
        let tab = chip_rows * head_dim / 2;
        if qkv.len() < rows * 3 * d
            || q.len() < rows * d
            || k.len() < rows * d
            || v.len() < rows * d
            || bq.len() < d
            || bk.len() < d
            || bv.len() < d
            || rope.is_some_and(|(c, s)| c.len() < tab || s.len() < tab)
        {
            return Err(oob("sam3_qkv_split_rope_h: buffers under the row geometry"));
        }
        let (sp, _g1) = qkv.device_ptr(&self.stream);
        let (bqp, _g2) = bq.device_ptr(&self.stream);
        let (bkp, _g3) = bk.device_ptr(&self.stream);
        let (bvp, _g4) = bv.device_ptr(&self.stream);
        // null tables are the kernel's "no rope"
        let (cp, _g5, snp, _g6) = match rope {
            Some((c, s)) => {
                let (cp, g5) = c.device_ptr(&self.stream);
                let (snp, g6) = s.device_ptr(&self.stream);
                (cp, Some(g5), snp, Some(g6))
            }
            None => (0, None, 0, None),
        };
        let (qp, _g7) = q.device_ptr_mut(&self.stream);
        let (kp, _g8) = k.device_ptr_mut(&self.stream);
        let (vp, _g9) = v.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 752); bounds checked above
        check(unsafe {
            f(
                sp as *const _,
                bqp as *const _,
                bkp as *const _,
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
                qscale,
                self.stream_ptr(),
            )
        })
    }

    /// The trunk's exit: window-major f32 rows `[pics, g * g, d]` to raster
    /// f16 rows (row `y * g + x`), the necks' input.
    pub fn sam3_rows_to_raster_h(
        &self,
        x: &CudaSlice<f32>,
        out: &mut CudaSlice<f16>,
        pics: usize,
        g: usize,
        win: usize,
        d: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sam3_rows_to_raster_h
            .ok_or(GpuError::MissingOp("sam3_rows_to_raster_h"))?;
        if win == 0 || !g.is_multiple_of(win) {
            return Err(oob(
                "sam3_rows_to_raster_h: the grid is not a whole number of windows",
            ));
        }
        let n = pics * g * g * d;
        if x.len() < n || out.len() < n {
            return Err(oob("sam3_rows_to_raster_h: buffers under pics x g^2 x d"));
        }
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (op, _g2) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 753); bounds checked above
        check(unsafe {
            f(
                xp as *const _,
                op as *mut _,
                pics as u32,
                g as u32,
                win as u32,
                d as u32,
                self.stream_ptr(),
            )
        })
    }

    /// A 2x2 / stride-2 transposed conv's depth-to-space: `g` is its GEMM's
    /// f32 landing `[pics, h * w, 4 * c]` (rows tap-major), `out` the f16
    /// raster `[pics, 2h, 2w, c]` with `bias` added and, when `gelu`, the
    /// exact GELU applied before the round.
    #[allow(clippy::too_many_arguments)]
    pub fn sam3_convt2_bias_h(
        &self,
        g: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        out: &mut CudaSlice<f16>,
        pics: usize,
        h: usize,
        w: usize,
        c: usize,
        gelu: bool,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sam3_convt2_bias_h
            .ok_or(GpuError::MissingOp("sam3_convt2_bias_h"))?;
        let n = pics * 4 * h * w * c;
        if g.len() < n || out.len() < n || bias.len() < c {
            return Err(oob("sam3_convt2_bias_h: buffers under the conv geometry"));
        }
        let (gp, _g1) = g.device_ptr(&self.stream);
        let (bp, _g2) = bias.device_ptr(&self.stream);
        let (op, _g3) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 754); bounds checked above
        check(unsafe {
            f(
                gp as *const _,
                bp as *const _,
                op as *mut _,
                pics as u32,
                h as u32,
                w as u32,
                c as u32,
                u32::from(gelu),
                self.stream_ptr(),
            )
        })
    }

    /// [`Self::matvec_batch_f16_h_gelu`] with the tanh-approximate GELU -
    /// `y16 = f16(gelu_tanh(W x + bias))`, SAM 3's ViT fc1.
    pub fn matvec_batch_f16_h_gelu_tanh(
        &self,
        w: &HalfTensor,
        x16: &CudaSlice<f16>,
        y16: &mut CudaSlice<f16>,
        bias: &CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .f16_gemm_h_gelu_tanh
            .ok_or(GpuError::MissingOp("f16_gemm_h_gelu_tanh"))?;
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        if !in_dim.is_multiple_of(8) {
            return Err(oob("f16_gemm_h_gelu_tanh: in_dim must be a multiple of 8"));
        }
        if w.buf.len() < in_dim * out_dim
            || x16.len() < batch * in_dim
            || y16.len() < batch * out_dim
            || bias.len() < out_dim
        {
            return Err(oob("f16_gemm_h_gelu_tanh: buffers under the GEMM geometry"));
        }
        super::basic_ops::gemm_census("B-gemm-f16-h-gelu-tanh", in_dim, out_dim, batch);
        let (wp, _g1) = w.buf.device_ptr(&self.stream);
        let (xp, _g2) = x16.device_ptr(&self.stream);
        let (bp, _g3) = bias.device_ptr(&self.stream);
        let (yp, _g4) = y16.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 755); bounds checked above
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

    /// Causal self-attention over the text tower's prompts: `q`, `k`, `v`,
    /// `out` f16 `[prompts, t, heads, head_dim]`, q pre-scaled. `t <= 64`,
    /// `head_dim <= 64`.
    #[allow(clippy::too_many_arguments)]
    pub fn sam3_text_attn_h(
        &self,
        q: &CudaSlice<f16>,
        k: &CudaSlice<f16>,
        v: &CudaSlice<f16>,
        out: &mut CudaSlice<f16>,
        prompts: usize,
        t: usize,
        heads: usize,
        head_dim: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sam3_text_attn_h
            .ok_or(GpuError::MissingOp("sam3_text_attn_h"))?;
        if t > 64 || head_dim == 0 || head_dim > 64 {
            return Err(oob(
                "sam3_text_attn_h: at most 64 tokens and a 64-wide head",
            ));
        }
        let n = prompts * t * heads * head_dim;
        if q.len() < n || k.len() < n || v.len() < n || out.len() < n {
            return Err(oob("sam3_text_attn_h: buffers under the prompt geometry"));
        }
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = k.device_ptr(&self.stream);
        let (vp, _g3) = v.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 759); bounds checked above
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                op as *mut _,
                prompts as u32,
                t as u32,
                heads as u32,
                head_dim as u32,
                self.stream_ptr(),
            )
        })
    }

    /// The detector's own ops (slots 760-769), on top of the text tower's.
    pub fn has_sam3_detector(&self) -> bool {
        let k = &self.kernels;
        self.has_sam3_text()
            && k.sam3_seam_h.is_some()
            && k.sam3_box_attn_h.is_some()
            && k.sam3_rpb_tables.is_some()
            && k.sam3_box_sine.is_some()
            && k.sam3_box_refine.is_some()
            && k.sam3_roi_align.is_some()
            && k.sam3_gn_relu_f16.is_some()
            && k.sam3_up2_add_h.is_some()
            && k.sam3_score.is_some()
            && k.f16_gemm_h_bias.is_some()
    }

    /// Every residual seam of the detector, one launch a row:
    /// `x' = x + proj + bias`; with `norm`, pre-norm keeps `x = x'` and takes
    /// `y = LN(x')`, post-norm (`post`) sets `x = y = LN(x')`; with no norm,
    /// `y = x' = x`. Then `out = f16(y)` and `outq = f16(y + pos)`, `pos` a
    /// `[npos, n]` table read at `t = row % period` (rows with `t >= npos` -
    /// the decoder's presence token, its last row - get none). `n <= 1024`.
    #[allow(clippy::too_many_arguments)]
    pub fn sam3_seam_h(
        &self,
        x: &mut CudaSlice<f32>,
        proj: Option<&CudaSlice<f16>>,
        bias: Option<&CudaSlice<f32>>,
        norm: Option<(&CudaSlice<f32>, &CudaSlice<f32>)>,
        pos: Option<(&CudaSlice<f32>, usize, usize)>,
        out: Option<&mut CudaSlice<f16>>,
        outq: Option<&mut CudaSlice<f16>>,
        rows: usize,
        n: usize,
        eps: f32,
        post: bool,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sam3_seam_h
            .ok_or(GpuError::MissingOp("sam3_seam_h"))?;
        let t = rows * n;
        if n > 1024
            || x.len() < t
            || proj.is_some_and(|p| p.len() < t)
            || bias.is_some_and(|b| b.len() < n)
            || norm.is_some_and(|(w, b)| w.len() < n || b.len() < n)
            || pos.is_some_and(|(p, period, npos)| period == 0 || p.len() < npos * n)
            || out.as_ref().is_some_and(|o| o.len() < t)
            || outq.as_ref().is_some_and(|o| o.len() < t)
        {
            return Err(oob("sam3_seam_h: buffers under the row geometry"));
        }
        let flags = u32::from(post) | if norm.is_none() { 2 } else { 0 };
        let (period, npos) = pos.map_or((1, 0), |(_, p, s)| (p, s));
        let (xp, _g1) = x.device_ptr_mut(&self.stream);
        let pr = proj.map(|p| p.device_ptr(&self.stream));
        let br = bias.map(|b| b.device_ptr(&self.stream));
        let wr = norm.map(|(w, _)| w.device_ptr(&self.stream));
        let nbr = norm.map(|(_, b)| b.device_ptr(&self.stream));
        let posr = pos.map(|(p, _, _)| p.device_ptr(&self.stream));
        let or = out.map(|o| o.device_ptr_mut(&self.stream));
        let oqr = outq.map(|o| o.device_ptr_mut(&self.stream));
        let p0 = |r: &Option<(u64, _)>| r.as_ref().map_or(0, |(p, _)| *p);
        // SAFETY: ABI contract (slot 760); bounds checked above, null = absent
        check(unsafe {
            f(
                xp as *mut _,
                p0(&pr) as *const _,
                p0(&br) as *const _,
                p0(&wr) as *const _,
                p0(&nbr) as *const _,
                p0(&posr) as *const _,
                p0(&or) as *mut _,
                p0(&oqr) as *mut _,
                rows as u32,
                n as u32,
                period as u32,
                npos as u32,
                eps,
                flags,
                self.stream_ptr(),
            )
        })
    }

    /// The decoder's image cross-attention with the factorized box bias
    /// `by[q][y][h] + bx[q][x][h]` on its scores (`bx` `[nbias, gw, heads]`,
    /// `by` `[nbias, gh, heads]` f32); rows from `nbias` on get no bias. q
    /// pre-scaled; q / out `[nq, heads, hd]`, k / v `[gh * gw, heads, hd]` f16.
    #[allow(clippy::too_many_arguments)]
    pub fn sam3_box_attn_h(
        &self,
        q: &CudaSlice<f16>,
        k: &CudaSlice<f16>,
        v: &CudaSlice<f16>,
        bx: &CudaSlice<f32>,
        by: &CudaSlice<f32>,
        out: &mut CudaSlice<f16>,
        nq: usize,
        heads: usize,
        hd: usize,
        gh: usize,
        gw: usize,
        nbias: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sam3_box_attn_h
            .ok_or(GpuError::MissingOp("sam3_box_attn_h"))?;
        let (nk, d) = (gh * gw, heads * hd);
        let nb = nbias.min(nq);
        if q.len() < nq * d
            || out.len() < nq * d
            || k.len() < nk * d
            || v.len() < nk * d
            || bx.len() < nb * gw * heads
            || by.len() < nb * gh * heads
        {
            return Err(oob("sam3_box_attn_h: buffers under the attention geometry"));
        }
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = k.device_ptr(&self.stream);
        let (vp, _g3) = v.device_ptr(&self.stream);
        let (bxp, _g4) = bx.device_ptr(&self.stream);
        let (byp, _g5) = by.device_ptr(&self.stream);
        let (op, _g6) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 761); bounds checked above
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                bxp as *const _,
                byp as *const _,
                op as *mut _,
                nq as u32,
                nk as u32,
                heads as u32,
                hd as u32,
                gh as u32,
                gw as u32,
                nbias as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Whether [`Self::sam3_box_attn_mma`] takes this attention: the pack has
    /// slot 774, the die has mma.sync f16, and the shape is the one it is
    /// built for (32-wide heads over a 72-column memory of even height).
    pub fn sam3_box_attn_mma_fits(&self, hd: usize, gh: usize, gw: usize) -> bool {
        self.kernels.sam3_box_attn_mma.is_some()
            && self.compute_capability().0 >= 8
            && hd == 32
            && gw == 72
            && gh > 0
            && gh.is_multiple_of(2)
    }

    /// f32 scratch [`Self::sam3_box_attn_mma`] needs for `nsplit` key splits
    /// (none for one: the kernel stores its rows itself).
    pub fn sam3_box_attn_mma_part_len(nq: usize, heads: usize, hd: usize, nsplit: usize) -> usize {
        if nsplit <= 1 {
            0
        } else {
            nsplit * nq * heads * (hd + 2)
        }
    }

    /// [`Self::sam3_box_attn_h`] on the tensor cores (slot 774), same
    /// arguments plus `part` scratch and the key split count (`1..=gh / 2`
    /// runs of grid-row pairs, folded in order). With `nbias` 0 it is plain
    /// attention over the memory. Shapes outside
    /// [`Self::sam3_box_attn_mma_fits`] are refused.
    #[allow(clippy::too_many_arguments)]
    pub fn sam3_box_attn_mma(
        &self,
        q: &CudaSlice<f16>,
        k: &CudaSlice<f16>,
        v: &CudaSlice<f16>,
        bx: &CudaSlice<f32>,
        by: &CudaSlice<f32>,
        part: &mut CudaSlice<f32>,
        out: &mut CudaSlice<f16>,
        nq: usize,
        heads: usize,
        hd: usize,
        gh: usize,
        gw: usize,
        nbias: usize,
        nsplit: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sam3_box_attn_mma
            .ok_or(GpuError::MissingOp("sam3_box_attn_mma"))?;
        let (nk, d) = (gh * gw, heads * hd);
        let nb = nbias.min(nq);
        if q.len() < nq * d
            || out.len() < nq * d
            || k.len() < nk * d
            || v.len() < nk * d
            || bx.len() < nb * gw * heads
            || by.len() < nb * gh * heads
            || part.len() < Self::sam3_box_attn_mma_part_len(nq, heads, hd, nsplit)
        {
            return Err(oob(
                "sam3_box_attn_mma: buffers under the attention geometry",
            ));
        }
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = k.device_ptr(&self.stream);
        let (vp, _g3) = v.device_ptr(&self.stream);
        let (bxp, _g4) = bx.device_ptr(&self.stream);
        let (byp, _g5) = by.device_ptr(&self.stream);
        let (pp, _g6) = part.device_ptr_mut(&self.stream);
        let (op, _g7) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 774); bounds checked above, the shape by the pack
        check(unsafe {
            f(
                qp as *const _,
                kp as *const _,
                vp as *const _,
                bxp as *const _,
                byp as *const _,
                pp as *mut _,
                op as *mut _,
                nq as u32,
                heads as u32,
                hd as u32,
                gh as u32,
                gw as u32,
                nbias as u32,
                nsplit as u32,
                self.stream_ptr(),
            )
        })
    }

    /// The two box-bias tables for `nq` reference boxes (cxcywh f32): `tx`
    /// `[nq, gw, heads]`, `ty` `[nq, gh, heads]`. Each axis's MLP is
    /// `(w1 [hid, 2], b1 [hid], w2 [heads, hid], b2 [heads])` f32.
    #[allow(clippy::too_many_arguments)]
    pub fn sam3_rpb_tables(
        &self,
        reference: &CudaSlice<f32>,
        mx: (
            &CudaSlice<f32>,
            &CudaSlice<f32>,
            &CudaSlice<f32>,
            &CudaSlice<f32>,
        ),
        my: (
            &CudaSlice<f32>,
            &CudaSlice<f32>,
            &CudaSlice<f32>,
            &CudaSlice<f32>,
        ),
        tx: &mut CudaSlice<f32>,
        ty: &mut CudaSlice<f32>,
        nq: usize,
        gh: usize,
        gw: usize,
        hid: usize,
        heads: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sam3_rpb_tables
            .ok_or(GpuError::MissingOp("sam3_rpb_tables"))?;
        let mlp_ok = |m: &(
            &CudaSlice<f32>,
            &CudaSlice<f32>,
            &CudaSlice<f32>,
            &CudaSlice<f32>,
        )| {
            m.0.len() >= hid * 2
                && m.1.len() >= hid
                && m.2.len() >= heads * hid
                && m.3.len() >= heads
        };
        if heads > 16
            || reference.len() < nq * 4
            || tx.len() < nq * gw * heads
            || ty.len() < nq * gh * heads
            || !mlp_ok(&mx)
            || !mlp_ok(&my)
        {
            return Err(oob("sam3_rpb_tables: buffers under the table geometry"));
        }
        let (rp, _g0) = reference.device_ptr(&self.stream);
        let (a1, _g1) = mx.0.device_ptr(&self.stream);
        let (a2, _g2) = mx.1.device_ptr(&self.stream);
        let (a3, _g3) = mx.2.device_ptr(&self.stream);
        let (a4, _g4) = mx.3.device_ptr(&self.stream);
        let (c1, _g5) = my.0.device_ptr(&self.stream);
        let (c2, _g6) = my.1.device_ptr(&self.stream);
        let (c3, _g7) = my.2.device_ptr(&self.stream);
        let (c4, _g8) = my.3.device_ptr(&self.stream);
        let (txp, _g9) = tx.device_ptr_mut(&self.stream);
        let (typ, _g10) = ty.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 762); bounds checked above
        check(unsafe {
            f(
                rp as *const _,
                a1 as *const _,
                a2 as *const _,
                a3 as *const _,
                a4 as *const _,
                c1 as *const _,
                c2 as *const _,
                c3 as *const _,
                c4 as *const _,
                txp as *mut _,
                typ as *mut _,
                nq as u32,
                gh as u32,
                gw as u32,
                hid as u32,
                heads as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Sine embeddings of `n` cxcywh boxes into f16 rows of `ld`: mode 0 the
    /// decoder's query position `[y | x | w | h]` (4 * npf wide), mode 1 the
    /// geometry encoder's box encoding `[y | x | h | w]` (2 * npf + 2, the last
    /// two raw), the tail of each row zero.
    #[allow(clippy::too_many_arguments)]
    pub fn sam3_box_sine(
        &self,
        boxes: &CudaSlice<f32>,
        out: &mut CudaSlice<f16>,
        n: usize,
        npf: usize,
        mode: u32,
        ld: usize,
        temperature: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sam3_box_sine
            .ok_or(GpuError::MissingOp("sam3_box_sine"))?;
        let width = if mode == 0 { 4 * npf } else { 2 * npf + 2 };
        if mode > 1 || ld < width || boxes.len() < n * 4 || out.len() < n * ld {
            return Err(oob("sam3_box_sine: buffers under the embedding geometry"));
        }
        let (bp, _g1) = boxes.device_ptr(&self.stream);
        let (op, _g2) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 763); bounds checked above
        check(unsafe {
            f(
                bp as *const _,
                op as *mut _,
                n as u32,
                npf as u32,
                mode,
                ld as u32,
                temperature,
                self.stream_ptr(),
            )
        })
    }

    /// Box refinement in place: `ref = sigmoid(delta + bias + inverse_sigmoid(ref))`,
    /// `delta` rows `ld` floats apart (cxcywh in the first 4).
    pub fn sam3_box_refine(
        &self,
        reference: &mut CudaSlice<f32>,
        delta: &CudaSlice<f32>,
        bias: Option<&CudaSlice<f32>>,
        n: usize,
        ld: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sam3_box_refine
            .ok_or(GpuError::MissingOp("sam3_box_refine"))?;
        if ld < 4
            || reference.len() < n * 4
            || delta.len() < n * ld
            || bias.is_some_and(|b| b.len() < 4)
        {
            return Err(oob("sam3_box_refine: buffers under the box geometry"));
        }
        let (rp, _g1) = reference.device_ptr_mut(&self.stream);
        let (dp, _g2) = delta.device_ptr(&self.stream);
        let br = bias.map(|b| b.device_ptr(&self.stream));
        // SAFETY: ABI contract (slot 764); bounds checked above, null bias = none
        check(unsafe {
            f(
                rp as *mut _,
                dp as *const _,
                br.as_ref().map_or(0, |(p, _)| *p) as *const _,
                n as u32,
                ld as u32,
                self.stream_ptr(),
            )
        })
    }

    /// torchvision's `roi_align` (aligned=False, adaptive sampling): an NHWC
    /// f32 raster `[h, w, c]`, `n` xyxy boxes in its pixels, out f16
    /// `[n, c, pooled, pooled]`.
    #[allow(clippy::too_many_arguments)]
    pub fn sam3_roi_align(
        &self,
        feat: &CudaSlice<f32>,
        boxes: &CudaSlice<f32>,
        out: &mut CudaSlice<f16>,
        n: usize,
        h: usize,
        w: usize,
        c: usize,
        pooled: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sam3_roi_align
            .ok_or(GpuError::MissingOp("sam3_roi_align"))?;
        if feat.len() < h * w * c || boxes.len() < n * 4 || out.len() < n * c * pooled * pooled {
            return Err(oob("sam3_roi_align: buffers under the pooling geometry"));
        }
        let (fp, _g1) = feat.device_ptr(&self.stream);
        let (bp, _g2) = boxes.device_ptr(&self.stream);
        let (op, _g3) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 765); bounds checked above
        check(unsafe {
            f(
                fp as *const _,
                bp as *const _,
                op as *mut _,
                n as u32,
                h as u32,
                w as u32,
                c as u32,
                pooled as u32,
                self.stream_ptr(),
            )
        })
    }

    /// [`Self::dp_group_norm_gelu_f16`] with ReLU in place of GELU - the same
    /// arguments, scratch and statistics.
    #[allow(clippy::too_many_arguments)]
    pub fn sam3_gn_relu_f16(
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
            .sam3_gn_relu_f16
            .ok_or(GpuError::MissingOp("sam3_gn_relu_f16"))?;
        let n = chips * pixels * channels;
        if groups == 0
            || !channels.is_multiple_of(groups)
            || x.len() < n
            || out.len() < n
            || xb.len() < channels
            || w.len() < channels
            || b.len() < channels
            || part.len() < Self::dp_group_norm_part_len(chips, pixels, channels)
            || stat.len() < 2 * chips * groups
        {
            return Err(oob("sam3_gn_relu_f16: buffers under the plane geometry"));
        }
        let (xp, _g1) = x.device_ptr(&self.stream);
        let (xbp, _g0) = xb.device_ptr(&self.stream);
        let (wp, _g2) = w.device_ptr(&self.stream);
        let (bp, _g3) = b.device_ptr(&self.stream);
        let (op, _g4) = out.device_ptr_mut(&self.stream);
        let (pp, _g5) = part.device_ptr_mut(&self.stream);
        let (sp, _g6) = stat.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 766); plane and both scratch sizes checked above
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

    /// The pixel decoder seam: `out = f16(skip + nearest_x2(prev))`, `prev` f16
    /// `[pics, h, w, c]`, `skip` f32 `[pics, 2h, 2w, c]`.
    #[allow(clippy::too_many_arguments)]
    pub fn sam3_up2_add_h(
        &self,
        prev: &CudaSlice<f16>,
        skip: &CudaSlice<f32>,
        out: &mut CudaSlice<f16>,
        pics: usize,
        h: usize,
        w: usize,
        c: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sam3_up2_add_h
            .ok_or(GpuError::MissingOp("sam3_up2_add_h"))?;
        let n = pics * 4 * h * w * c;
        if prev.len() < n / 4 || skip.len() < n || out.len() < n {
            return Err(oob("sam3_up2_add_h: buffers under the plane geometry"));
        }
        let (pp, _g1) = prev.device_ptr(&self.stream);
        let (sp, _g2) = skip.device_ptr(&self.stream);
        let (op, _g3) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 767); bounds checked above
        check(unsafe {
            f(
                pp as *const _,
                sp as *const _,
                op as *mut _,
                pics as u32,
                h as u32,
                w as u32,
                c as u32,
                self.stream_ptr(),
            )
        })
    }

    /// The scorer's last step: `logit[q] = clamp(scale * hp[q] . pp, +-clamp)`,
    /// `prob[q] = sigmoid(logit[q]) * sigmoid(presence[0])`.
    #[allow(clippy::too_many_arguments)]
    pub fn sam3_score(
        &self,
        hp: &CudaSlice<f32>,
        pp: &CudaSlice<f32>,
        presence: &CudaSlice<f32>,
        logit: &mut CudaSlice<f32>,
        prob: &mut CudaSlice<f32>,
        nq: usize,
        d: usize,
        scale: f32,
        clamp: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sam3_score
            .ok_or(GpuError::MissingOp("sam3_score"))?;
        if hp.len() < nq * d
            || pp.len() < d
            || presence.is_empty()
            || logit.len() < nq
            || prob.len() < nq
        {
            return Err(oob("sam3_score: buffers under the query geometry"));
        }
        let (hpp, _g1) = hp.device_ptr(&self.stream);
        let (ppp, _g2) = pp.device_ptr(&self.stream);
        let (prp, _g3) = presence.device_ptr(&self.stream);
        let (lp, _g4) = logit.device_ptr_mut(&self.stream);
        let (pbp, _g5) = prob.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 768); bounds checked above
        check(unsafe {
            f(
                hpp as *const _,
                ppp as *const _,
                prp as *const _,
                lp as *mut _,
                pbp as *mut _,
                nq as u32,
                d as u32,
                scale,
                clamp,
                self.stream_ptr(),
            )
        })
    }

    /// [`Self::matvec_batch_f16_h`] with an optional bias added before the
    /// round: `y16 = f16(W x + bias)`.
    /// Whether [`Self::f16_gemm_qkv_rope`] is elected: slot 776 present and a
    /// cc 8.x die, where its ring is the measured blocked one.
    pub fn f16_gemm_qkv_rope_elected(&self) -> bool {
        self.kernels.f16_gemm_qkv_rope.is_some() && self.compute_capability().0 == 8
    }

    /// The q|k|v projection landed as the three half planes the attention
    /// eats (slot 776): `w` the stacked `[in][3 d]` weight, `bias` `[3 d]`,
    /// the rotate-half rope from `rope` (`[chip_rows][hd / 2]` tables, row =
    /// token % chip_rows) on q and k, q scaled by `qscale` after it - the
    /// split kernel's arithmetic on the f32 accumulator, one round. hd 64,
    /// d a multiple of 128.
    #[allow(clippy::too_many_arguments)]
    pub fn f16_gemm_qkv_rope(
        &self,
        w: &HalfTensor,
        x16: &CudaSlice<f16>,
        q: &mut CudaSlice<f16>,
        k: &mut CudaSlice<f16>,
        v: &mut CudaSlice<f16>,
        bias: &CudaSlice<f32>,
        rope: Option<(&CudaSlice<f32>, &CudaSlice<f32>)>,
        d: usize,
        hd: usize,
        batch: usize,
        chip_rows: usize,
        qscale: f32,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .f16_gemm_qkv_rope
            .ok_or(GpuError::MissingOp("f16_gemm_qkv_rope"))?;
        let in_dim = w.dims[0];
        let table = chip_rows * hd / 2;
        if w.dims[1] != 3 * d
            || x16.len() < batch * in_dim
            || q.len() < batch * d
            || k.len() < batch * d
            || v.len() < batch * d
            || bias.len() < 3 * d
            || chip_rows == 0
            || rope.is_some_and(|(c, s)| c.len() < table || s.len() < table)
        {
            return Err(oob("f16_gemm_qkv_rope: buffers under the GEMM geometry"));
        }
        let (wp, _g1) = w.buf.device_ptr(&self.stream);
        let (xp, _g2) = x16.device_ptr(&self.stream);
        let (qp, _g3) = q.device_ptr_mut(&self.stream);
        let (kp, _g4) = k.device_ptr_mut(&self.stream);
        let (vp, _g5) = v.device_ptr_mut(&self.stream);
        let (bp, _g6) = bias.device_ptr(&self.stream);
        let cg = rope.map(|(c, _)| c.device_ptr(&self.stream));
        let sg = rope.map(|(_, s)| s.device_ptr(&self.stream));
        let cp = cg.as_ref().map_or(0, |(p, _)| *p);
        let sp = sg.as_ref().map_or(0, |(p, _)| *p);
        // SAFETY: ABI contract (slot 776); bounds checked above, null tables = no rope
        check(unsafe {
            f(
                wp as *const _,
                xp as *const _,
                qp as *mut _,
                kp as *mut _,
                vp as *mut _,
                bp as *const _,
                cp as *const _,
                sp as *const _,
                in_dim as u32,
                d as u32,
                hd as u32,
                batch as u32,
                chip_rows as u32,
                qscale,
                self.stream_ptr(),
            )
        })
    }

    pub fn matvec_batch_f16_h_bias(
        &self,
        w: &HalfTensor,
        x16: &CudaSlice<f16>,
        y16: &mut CudaSlice<f16>,
        bias: Option<&CudaSlice<f32>>,
        batch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .f16_gemm_h_bias
            .ok_or(GpuError::MissingOp("f16_gemm_h_bias"))?;
        let (in_dim, out_dim) = (w.dims[0], w.dims[1]);
        if !in_dim.is_multiple_of(8) {
            return Err(oob("f16_gemm_h_bias: in_dim must be a multiple of 8"));
        }
        if w.buf.len() < in_dim * out_dim
            || x16.len() < batch * in_dim
            || y16.len() < batch * out_dim
            || bias.is_some_and(|b| b.len() < out_dim)
        {
            return Err(oob("f16_gemm_h_bias: buffers under the GEMM geometry"));
        }
        super::basic_ops::gemm_census("B-gemm-f16-h-bias", in_dim, out_dim, batch);
        let (wp, _g1) = w.buf.device_ptr(&self.stream);
        let (xp, _g2) = x16.device_ptr(&self.stream);
        let br = bias.map(|b| b.device_ptr(&self.stream));
        let (yp, _g4) = y16.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 769); bounds checked above, null bias = none
        check(unsafe {
            f(
                wp as *const _,
                xp as *const _,
                yp as *mut _,
                br.as_ref().map_or(0, |(p, _)| *p) as *const _,
                in_dim as u32,
                out_dim as u32,
                batch as u32,
                self.stream_ptr(),
            )
        })
    }

    /// The picture-in / masks-out ops (slots 770-772).
    pub fn has_sam3_image_io(&self) -> bool {
        let k = &self.kernels;
        k.sam3_resize_aa_u8.is_some() && k.sam3_mask_up.is_some() && k.sam3_rle.is_some()
    }

    /// torchvision's antialiased bilinear resize of a u8 HWC picture
    /// `[h, w, c]` to `[oh, ow, c]`, to the bit (Meta's processor on CUDA).
    #[allow(clippy::too_many_arguments)]
    pub fn sam3_resize_aa_u8(
        &self,
        src: &CudaSlice<u8>,
        dst: &mut CudaSlice<u8>,
        h: usize,
        w: usize,
        oh: usize,
        ow: usize,
        c: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sam3_resize_aa_u8
            .ok_or(GpuError::MissingOp("sam3_resize_aa_u8"))?;
        if src.len() < h * w * c || dst.len() < oh * ow * c {
            return Err(oob("sam3_resize_aa_u8: buffers under the picture geometry"));
        }
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (dp, _g2) = dst.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 770); bounds checked above
        check(unsafe {
            f(
                sp as *const _,
                dp as *mut _,
                h as u32,
                w as u32,
                oh as u32,
                ow as u32,
                c as u32,
                self.stream_ptr(),
            )
        })
    }

    /// Query `q`'s mask logits (`logits` the detector's pixel-major
    /// `[side^2, nq]` landing) bilinear to an `h x w` picture, sigmoid > 0.5,
    /// as u8 0/1 COLUMN-major `[w][h]` into `out`.
    #[allow(clippy::too_many_arguments)]
    pub fn sam3_mask_up(
        &self,
        logits: &CudaSlice<f32>,
        out: &mut CudaSlice<u8>,
        side: usize,
        nq: usize,
        q: usize,
        h: usize,
        w: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sam3_mask_up
            .ok_or(GpuError::MissingOp("sam3_mask_up"))?;
        if q >= nq || logits.len() < side * side * nq || out.len() < h * w {
            return Err(oob("sam3_mask_up: buffers under the mask geometry"));
        }
        let (lp, _g1) = logits.device_ptr(&self.stream);
        let (op, _g2) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 771); bounds checked above
        check(unsafe {
            f(
                lp as *const _,
                op as *mut _,
                side as u32,
                nq as u32,
                q as u32,
                h as u32,
                w as u32,
                self.stream_ptr(),
            )
        })
    }

    /// COCO RLE of a column-major 0/1 mask of `n` pixels into `counts`
    /// (`starts` is scratch of the same capacity); `nruns[0]` gets the count
    /// written, or `u32::MAX` when the mask needs more than `cap`.
    pub fn sam3_rle(
        &self,
        mask: &CudaSlice<u8>,
        starts: &mut CudaSlice<u32>,
        counts: &mut CudaSlice<u32>,
        nruns: &mut CudaSlice<u32>,
        n: usize,
        cap: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .sam3_rle
            .ok_or(GpuError::MissingOp("sam3_rle"))?;
        if mask.len() < n || starts.len() < cap || counts.len() < cap || nruns.is_empty() {
            return Err(oob("sam3_rle: buffers under the run geometry"));
        }
        let (mp, _g1) = mask.device_ptr(&self.stream);
        let (sp, _g2) = starts.device_ptr_mut(&self.stream);
        let (cp, _g3) = counts.device_ptr_mut(&self.stream);
        let (np, _g4) = nruns.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract (slot 772); bounds checked above
        check(unsafe {
            f(
                mp as *const _,
                sp as *mut _,
                cp as *mut _,
                np as *mut _,
                n as u64,
                cap as u32,
                self.stream_ptr(),
            )
        })
    }
}
