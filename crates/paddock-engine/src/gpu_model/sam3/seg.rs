//! The segmentation head (Meta's UniversalSegmentationHead + PixelDecoder).
//!
//!   E' = E + MHA(LN(E), prompt)                     prompt cross-attention, pre-norm
//!   p  = ReLU(GN8(conv3x3(fpn1 + nearest_x2(E'))))   144^2
//!   p  = ReLU(GN8(conv3x3(fpn0 + nearest_x2(p))))    288^2 x 256: the pixel embedding
//!   instance = conv1x1(p);  semantic = conv1x1(p) -> 1
//!   masks[q] = mask_embed(hs_q) . instance           200 x 288^2 logits
//!
//! The 3-stage decoder ships three conv/norm pairs; with the 72x72 level
//! replaced by E' it walks two (144, 288), and `conv_layers.2` / `norms.2` are
//! dead in the checkpoint - not loaded. The mask einsum is one GEMM with the
//! mask embeddings as its weight, landing [pixel][query].

use cudarc::driver::CudaSlice;
use half::f16;

use super::GpuModelError;
use super::detector::{GN_EPS, GpuSam3Detector, LN_EPS};
use crate::gpu::{GpuExecutor, HalfTensor};

/// A pixel-decoder 3x3 over an f16 plane into f32 (its bias rides the group
/// norm that follows): the implicit GEMM where it is elected, else im2row into
/// `col16` and the GEMM - the same bits either way.
fn conv3(
    exec: &GpuExecutor,
    w: &HalfTensor,
    src: &CudaSlice<f16>,
    col16: &mut CudaSlice<f16>,
    out: &mut CudaSlice<f32>,
    side: usize,
    c: usize,
) -> Result<(), GpuModelError> {
    if exec.f16_conv3_elected() {
        exec.f16_conv3_gemm(w, src, out, None, 1, side, side, c, side * side)?;
    } else {
        exec.dp_im2row3_f16(src, col16, 1, side, side, c, side * side)?;
        exec.matvec_batch_f16(w, col16, out, side * side)?;
    }
    Ok(())
}

impl GpuSam3Detector {
    /// Masks and the semantic map for the last [`Self::decode`]. `fpn0` /
    /// `fpn1` are the picture's detector 288^2 / 144^2 levels (f32 rasters).
    pub fn segment(
        &mut self,
        fpn0: &CudaSlice<f32>,
        fpn1: &CudaSlice<f32>,
    ) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let g = self.g;
        let (d, t, hd, heads, nq) = (g.d, g.tokens(), g.hd(), g.heads, g.queries);
        let (s1, s2) = (2 * g.grid, 4 * g.grid);
        let np = self.ws.n_prompt;
        let seg = &self.seg;
        let ws = &mut self.ws;

        // ---- prompt cross-attention on the memory ----
        exec.copy_slice(&ws.enc, 0, t * d, &mut ws.segx)?;
        exec.sam3_seam_h(
            &mut ws.segx,
            None,
            None,
            Some((&seg.ca_norm.w, &seg.ca_norm.b)),
            None,
            Some(&mut ws.seg16),
            None,
            t,
            d,
            LN_EPS,
            false,
        )?;
        exec.matvec_batch_f16_h_bias(&seg.ca.q.w, &ws.seg16, &mut ws.q16, Some(&seg.ca.q.b), t)?;
        exec.matvec_batch_f16_h_bias(
            &seg.ca.k.w,
            &ws.prompt16,
            &mut ws.gk16,
            Some(&seg.ca.k.b),
            np,
        )?;
        exec.matvec_batch_f16_h_bias(
            &seg.ca.v.w,
            &ws.prompt16,
            &mut ws.gv16,
            Some(&seg.ca.v.b),
            np,
        )?;
        exec.vision_attn_h(
            &ws.q16,
            &ws.gk16,
            &ws.gv16,
            &mut ws.att16,
            t,
            np,
            heads,
            hd,
            1,
        )?;
        exec.matvec_batch_f16_h(&seg.ca.o.w, &ws.att16, &mut ws.proj16, t)?;
        exec.sam3_seam_h(
            &mut ws.segx,
            Some(&ws.proj16),
            Some(&seg.ca.o.b),
            None,
            None,
            Some(&mut ws.seg16),
            None,
            t,
            d,
            LN_EPS,
            false,
        )?;

        // ---- pixel decoder: 72 -> 144 -> 288 ----
        exec.sam3_up2_add_h(&ws.seg16, fpn1, &mut ws.up16, 1, g.grid, g.grid, d)?;
        conv3(
            &exec,
            &seg.convs[0].w,
            &ws.up16,
            &mut ws.col16,
            &mut ws.conv32,
            s1,
            d,
        )?;
        exec.sam3_gn_relu_f16(
            &ws.conv32,
            &seg.convs[0].b,
            &seg.norms[0].w,
            &seg.norms[0].b,
            &mut ws.pix16,
            &mut ws.gn_part,
            &mut ws.gn_stat,
            1,
            s1 * s1,
            d,
            8,
            GN_EPS,
        )?;
        // the 288^2 seam's landing is borrowed from the instance plane, which
        // is not written until the decoder is done with it
        exec.sam3_up2_add_h(&ws.pix16, fpn0, &mut ws.inst16, 1, s1, s1, d)?;
        conv3(
            &exec,
            &seg.convs[1].w,
            &ws.inst16,
            &mut ws.col16,
            &mut ws.conv32,
            s2,
            d,
        )?;
        exec.sam3_gn_relu_f16(
            &ws.conv32,
            &seg.convs[1].b,
            &seg.norms[1].w,
            &seg.norms[1].b,
            &mut ws.pix16,
            &mut ws.gn_part,
            &mut ws.gn_stat,
            1,
            s2 * s2,
            d,
            8,
            GN_EPS,
        )?;

        // ---- heads ----
        let px = s2 * s2;
        exec.matvec_batch_f16_h_bias(
            &seg.instance.w,
            &ws.pix16,
            &mut ws.inst16,
            Some(&seg.instance.b),
            px,
        )?;
        exec.matvec_batch_f16(&seg.semantic.w, &ws.pix16, &mut ws.semantic, px)?;
        exec.bias_add(&mut ws.semantic, &seg.semantic.b, px, 1)?;
        let [m0, m1, m2] = &seg.mask_embed;
        exec.matvec_batch_f16_h_relu(&m0.w, &ws.hs16, &mut ws.mlp_a16, Some(&m0.b), nq)?;
        exec.matvec_batch_f16_h_relu(&m1.w, &ws.mlp_a16, &mut ws.mlp_b16, Some(&m1.b), nq)?;
        exec.matvec_batch_f16_h_bias(&m2.w, &ws.mlp_b16, &mut ws.me.buf, Some(&m2.b), nq)?;
        exec.matvec_batch_f16(&ws.me, &ws.inst16, &mut ws.masks, px)?;
        Ok(())
    }
}
