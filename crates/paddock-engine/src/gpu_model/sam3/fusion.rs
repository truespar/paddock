//! The prompt and the fusion encoder.
//!
//! Geometry encoder (Meta's `SequenceGeometryEncoder`): one token per
//! exemplar box, then a learned CLS token - Meta appends the CLS to EVERY
//! prompt, text-only ones included, so even "shoe" carries one geometry token
//! (transformers' port skips it; the reference is Meta's). A box token is
//!   label_embed[label] + direct(cxcywh) + conv7x7(RoIAlign(LN(image), box)) + pos(box)
//! then all tokens go through final_proj -> LN, three pre-norm layers
//! (self-attention without positions, cross-attention to the raw 72x72 level
//! with positions on its keys, ReLU FFN) and an output LN.
//!
//! Fusion encoder (Meta's `TransformerEncoderFusion`): the 5184 image tokens
//! as queries, the prompt as memory, six pre-norm layers - self-attention with
//! positions on q and k, cross-attention to the prompt (not updated), a ReLU
//! FFN, and no final norm. Its output E is the decoder's memory and the
//! segmentation head's input.
//!
//! Every plane is a raster [tokens][256]; positions are the 72x72 sine table.

use cudarc::driver::CudaSlice;

use super::GpuModelError;
use super::detector::{EncLayer, GpuSam3Detector, LN_EPS, memory_attn};

/// A box exemplar: cxcywh normalized to the picture (which Meta resizes to a
/// square without keeping its aspect ratio, so these are also normalized to
/// the model's input), and whether it is a positive example.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sam3Box {
    pub cx: f32,
    pub cy: f32,
    pub w: f32,
    pub h: f32,
    pub positive: bool,
}

impl GpuSam3Detector {
    /// Build the prompt for one picture: the text tower's valid tokens
    /// (`n_valid` rows of `text` starting at element `text_off`), then the
    /// geometry tokens of `boxes` and the CLS. `fpn2` is the picture's
    /// detector 72x72 level, f32 raster [tokens][256].
    pub fn encode_prompt(
        &mut self,
        text: &CudaSlice<f32>,
        text_off: usize,
        n_valid: usize,
        fpn2: &CudaSlice<f32>,
        boxes: &[Sam3Box],
    ) -> Result<(), GpuModelError> {
        let g = self.g;
        let d = g.d;
        if n_valid == 0 || n_valid > g.text_tokens {
            return Err(GpuModelError::Unsupported(format!(
                "sam3: {n_valid} valid text tokens (want 1..={})",
                g.text_tokens
            )));
        }
        if boxes.len() > g.max_boxes {
            return Err(GpuModelError::Unsupported(format!(
                "sam3: {} exemplar boxes; this endpoint takes at most {}",
                boxes.len(),
                g.max_boxes
            )));
        }
        let n_geo = self.geometry(fpn2, boxes)?;
        let exec = self.exec.clone();
        let ws = &mut self.ws;
        exec.copy_region(text, text_off, &mut ws.prompt, 0, n_valid * d)?;
        exec.copy_region(&ws.g32, 0, &mut ws.prompt, n_valid * d, n_geo * d)?;
        ws.n_prompt = n_valid + n_geo;
        exec.convert_f32_f16(&ws.prompt, &mut ws.prompt16, ws.n_prompt * d)?;
        Ok(())
    }

    /// Geometry tokens into `ws.g32` rows `0..boxes.len() + 1`; returns the count.
    fn geometry(
        &mut self,
        fpn2: &CudaSlice<f32>,
        boxes: &[Sam3Box],
    ) -> Result<usize, GpuModelError> {
        let exec = self.exec.clone();
        let g = self.g;
        let (d, t, hd, heads) = (g.d, g.tokens(), g.hd(), g.heads);
        let nb = boxes.len();
        let n = nb + 1;
        let geo = &self.geo;
        let ws = &mut self.ws;

        // the image side of the cross-attention: f16(level) and f16(level + pos)
        exec.copy_slice(fpn2, 0, t * d, &mut ws.x)?;
        exec.sam3_seam_h(
            &mut ws.x,
            None,
            None,
            None,
            Some((&self.pos, t, t)),
            Some(&mut ws.h16),
            Some(&mut ws.hq16),
            t,
            d,
            LN_EPS,
            false,
        )?;

        // ---- the box tokens ----
        if nb > 0 {
            // request prep, the picture's own coordinates: cxcywh as given, and
            // xyxy in 72x72 feature pixels for RoIAlign (Meta's box_cxcywh_to_xyxy
            // then * [W, H, W, H], f32)
            let scale = g.grid as f32;
            let mut cxcywh = Vec::with_capacity(nb * 4);
            let mut xyxy = Vec::with_capacity(nb * 4);
            let mut direct = vec![0f32; nb * 8];
            for (i, b) in boxes.iter().enumerate() {
                cxcywh.extend_from_slice(&[b.cx, b.cy, b.w, b.h]);
                xyxy.extend_from_slice(&[
                    (b.cx - 0.5 * b.w) * scale,
                    (b.cy - 0.5 * b.h) * scale,
                    (b.cx + 0.5 * b.w) * scale,
                    (b.cy + 0.5 * b.h) * scale,
                ]);
                direct[i * 8..i * 8 + 4].copy_from_slice(&[b.cx, b.cy, b.w, b.h]);
            }
            exec.upload_f32(&cxcywh, &mut ws.boxes)?;
            exec.upload_f32(&xyxy, &mut ws.boxes_xyxy)?;
            exec.upload_f32(&direct, &mut ws.g32)?;
            exec.convert_f32_f16(&ws.g32, &mut ws.box_in16, nb * 8)?;
            // label embedding rows start each token, the three projections add on
            for (i, b) in boxes.iter().enumerate() {
                exec.copy_region(
                    &geo.label_embed,
                    usize::from(b.positive) * d,
                    &mut ws.gx,
                    i * d,
                    d,
                )?;
            }
            exec.gemm_f16_f32_acc(&geo.box_direct.w.buf, &ws.box_in16, &mut ws.gx, 8, d, nb)?;
            exec.layernorm(
                fpn2,
                &geo.vision_norm.w,
                &geo.vision_norm.b,
                &mut ws.vnorm,
                t,
                d,
                LN_EPS,
            )?;
            exec.sam3_roi_align(
                &ws.vnorm,
                &ws.boxes_xyxy,
                &mut ws.roi16,
                nb,
                g.grid,
                g.grid,
                d,
                7,
            )?;
            exec.gemm_f16_f32_acc(&geo.box_pool.w.buf, &ws.roi16, &mut ws.gx, d * 49, d, nb)?;
            let kp = geo.box_pos.w.dims[0];
            exec.sam3_box_sine(&ws.boxes, &mut ws.sine16, nb, d / 2, 1, kp, 10000.0)?;
            exec.gemm_f16_f32_acc(&geo.box_pos.w.buf, &ws.sine16, &mut ws.gx, kp, d, nb)?;
            for bias in [&geo.box_direct.b, &geo.box_pool.b, &geo.box_pos.b] {
                exec.bias_add(&mut ws.gx, bias, nb, d)?;
            }
        }
        exec.copy_region(&geo.cls, 0, &mut ws.gx, nb * d, d)?;

        // final_proj -> LN, then layer 0's pre-norm
        exec.convert_f32_f16(&ws.gx, &mut ws.g16, n * d)?;
        exec.matvec_batch_f16(&geo.final_proj.w, &ws.g16, &mut ws.g32, n)?;
        exec.sam3_seam_h(
            &mut ws.g32,
            None,
            Some(&geo.final_proj.b),
            Some((&geo.prompt_norm.w, &geo.prompt_norm.b)),
            None,
            None,
            None,
            n,
            d,
            LN_EPS,
            true,
        )?;
        let first = &geo.layers[0].norm1;
        exec.sam3_seam_h(
            &mut ws.g32,
            None,
            None,
            Some((&first.w, &first.b)),
            None,
            Some(&mut ws.g16),
            None,
            n,
            d,
            LN_EPS,
            false,
        )?;

        let nl = geo.layers.len();
        for (li, l) in geo.layers.iter().enumerate() {
            // self-attention, no positions
            exec.matvec_batch_f16_h_bias(&l.sa.q.w, &ws.g16, &mut ws.gq16, Some(&l.sa.q.b), n)?;
            exec.matvec_batch_f16_h_bias(&l.sa.k.w, &ws.g16, &mut ws.gk16, Some(&l.sa.k.b), n)?;
            exec.matvec_batch_f16_h_bias(&l.sa.v.w, &ws.g16, &mut ws.gv16, Some(&l.sa.v.b), n)?;
            exec.vision_attn_h(
                &ws.gq16,
                &ws.gk16,
                &ws.gv16,
                &mut ws.gatt16,
                n,
                n,
                heads,
                hd,
                1,
            )?;
            exec.matvec_batch_f16_h(&l.sa.o.w, &ws.gatt16, &mut ws.gproj16, n)?;
            exec.sam3_seam_h(
                &mut ws.g32,
                Some(&ws.gproj16),
                Some(&l.sa.o.b),
                Some((&l.norm2.w, &l.norm2.b)),
                None,
                Some(&mut ws.g16),
                None,
                n,
                d,
                LN_EPS,
                false,
            )?;
            // cross-attention to the raw level, positions on the keys
            exec.matvec_batch_f16_h_bias(&l.ca.q.w, &ws.g16, &mut ws.gq16, Some(&l.ca.q.b), n)?;
            exec.matvec_batch_f16_h_bias(&l.ca.k.w, &ws.hq16, &mut ws.gk16, Some(&l.ca.k.b), t)?;
            exec.matvec_batch_f16_h_bias(&l.ca.v.w, &ws.h16, &mut ws.gv16, Some(&l.ca.v.b), t)?;
            memory_attn(
                &exec,
                &g,
                &ws.gq16,
                &ws.gk16,
                &ws.gv16,
                (&ws.rpb_x, &ws.rpb_y),
                &mut ws.box_part,
                &mut ws.gatt16,
                n,
            )?;
            exec.matvec_batch_f16_h(&l.ca.o.w, &ws.gatt16, &mut ws.gproj16, n)?;
            exec.sam3_seam_h(
                &mut ws.g32,
                Some(&ws.gproj16),
                Some(&l.ca.o.b),
                Some((&l.norm3.w, &l.norm3.b)),
                None,
                Some(&mut ws.g16),
                None,
                n,
                d,
                LN_EPS,
                false,
            )?;
            // FFN; the last seam lands the output norm IN the stream (post),
            // which is the geometry token value
            exec.matvec_batch_f16_h_relu(&l.fc1.w, &ws.g16, &mut ws.gff16, Some(&l.fc1.b), n)?;
            exec.matvec_batch_f16_h(&l.fc2.w, &ws.gff16, &mut ws.gproj16, n)?;
            let last = li + 1 == nl;
            let next = if last {
                &geo.out_norm
            } else {
                &geo.layers[li + 1].norm1
            };
            exec.sam3_seam_h(
                &mut ws.g32,
                Some(&ws.gproj16),
                Some(&l.fc2.b),
                Some((&next.w, &next.b)),
                None,
                if last { None } else { Some(&mut ws.g16) },
                None,
                n,
                d,
                LN_EPS,
                last,
            )?;
        }
        Ok(n)
    }

    /// The fusion encoder over one picture's 72x72 detector level `img` (f32
    /// raster [tokens][256]) and the prompt [`Self::encode_prompt`] built.
    /// Leaves E in `ws.enc` (f32) and the decoder's memory planes f16(E) and
    /// f16(E + pos) in `ws.mem16` / `ws.memq16`.
    pub fn fuse(&mut self, img: &CudaSlice<f32>) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let g = self.g;
        let (d, t, hd, heads) = (g.d, g.tokens(), g.hd(), g.heads);
        let np = self.ws.n_prompt;
        if np == 0 {
            return Err(GpuModelError::Unsupported(
                "sam3: fuse before encode_prompt".into(),
            ));
        }
        let pos = &self.pos;
        let layers: &[EncLayer] = &self.fusion;
        let ws = &mut self.ws;
        exec.copy_slice(img, 0, t * d, &mut ws.x)?;
        let first = &layers[0].norm1;
        exec.sam3_seam_h(
            &mut ws.x,
            None,
            None,
            Some((&first.w, &first.b)),
            Some((pos, t, t)),
            Some(&mut ws.h16),
            Some(&mut ws.hq16),
            t,
            d,
            LN_EPS,
            false,
        )?;
        let nl = layers.len();
        for (li, l) in layers.iter().enumerate() {
            // self-attention, positions on q and k
            exec.matvec_batch_f16_h_bias(&l.sa.q.w, &ws.hq16, &mut ws.q16, Some(&l.sa.q.b), t)?;
            exec.matvec_batch_f16_h_bias(&l.sa.k.w, &ws.hq16, &mut ws.k16, Some(&l.sa.k.b), t)?;
            exec.matvec_batch_f16_h_bias(&l.sa.v.w, &ws.h16, &mut ws.v16, Some(&l.sa.v.b), t)?;
            memory_attn(
                &exec,
                &g,
                &ws.q16,
                &ws.k16,
                &ws.v16,
                (&ws.rpb_x, &ws.rpb_y),
                &mut ws.box_part,
                &mut ws.att16,
                t,
            )?;
            exec.matvec_batch_f16_h(&l.sa.o.w, &ws.att16, &mut ws.proj16, t)?;
            exec.sam3_seam_h(
                &mut ws.x,
                Some(&ws.proj16),
                Some(&l.sa.o.b),
                Some((&l.norm2.w, &l.norm2.b)),
                None,
                Some(&mut ws.h16),
                None,
                t,
                d,
                LN_EPS,
                false,
            )?;
            // cross-attention to the prompt, no positions either side
            exec.matvec_batch_f16_h_bias(&l.ca.q.w, &ws.h16, &mut ws.q16, Some(&l.ca.q.b), t)?;
            exec.matvec_batch_f16_h_bias(
                &l.ca.k.w,
                &ws.prompt16,
                &mut ws.gk16,
                Some(&l.ca.k.b),
                np,
            )?;
            exec.matvec_batch_f16_h_bias(
                &l.ca.v.w,
                &ws.prompt16,
                &mut ws.gv16,
                Some(&l.ca.v.b),
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
            exec.matvec_batch_f16_h(&l.ca.o.w, &ws.att16, &mut ws.proj16, t)?;
            exec.sam3_seam_h(
                &mut ws.x,
                Some(&ws.proj16),
                Some(&l.ca.o.b),
                Some((&l.norm3.w, &l.norm3.b)),
                None,
                Some(&mut ws.h16),
                None,
                t,
                d,
                LN_EPS,
                false,
            )?;
            // FFN; after the last layer there is no norm - the stream is E,
            // and the seam lands the decoder's two memory planes
            exec.matvec_batch_f16_h_relu(&l.fc1.w, &ws.h16, &mut ws.ff16, Some(&l.fc1.b), t)?;
            exec.matvec_batch_f16_h(&l.fc2.w, &ws.ff16, &mut ws.proj16, t)?;
            if li + 1 < nl {
                let next = &layers[li + 1].norm1;
                exec.sam3_seam_h(
                    &mut ws.x,
                    Some(&ws.proj16),
                    Some(&l.fc2.b),
                    Some((&next.w, &next.b)),
                    Some((pos, t, t)),
                    Some(&mut ws.h16),
                    Some(&mut ws.hq16),
                    t,
                    d,
                    LN_EPS,
                    false,
                )?;
            } else {
                exec.sam3_seam_h(
                    &mut ws.x,
                    Some(&ws.proj16),
                    Some(&l.fc2.b),
                    None,
                    Some((pos, t, t)),
                    Some(&mut ws.mem16),
                    Some(&mut ws.memq16),
                    t,
                    d,
                    LN_EPS,
                    false,
                )?;
            }
        }
        exec.copy_slice(&ws.x, 0, t * d, &mut ws.enc)?;
        Ok(())
    }

    /// Install an externally computed memory E (the gate feeds Meta's own) and
    /// derive the decoder's planes from it, as [`Self::fuse`] would have.
    pub fn set_memory(&mut self, enc: &CudaSlice<f32>) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let (d, t) = (self.g.d, self.g.tokens());
        let ws = &mut self.ws;
        exec.copy_slice(enc, 0, t * d, &mut ws.enc)?;
        exec.copy_slice(enc, 0, t * d, &mut ws.x)?;
        exec.sam3_seam_h(
            &mut ws.x,
            None,
            None,
            None,
            Some((&self.pos, t, t)),
            Some(&mut ws.mem16),
            Some(&mut ws.memq16),
            t,
            d,
            LN_EPS,
            false,
        )?;
        Ok(())
    }

    /// Install an externally computed prompt (`n` rows f32, valid tokens only)
    /// - the gate's way to run the encoder or decoder on Meta's own prompt.
    pub fn set_prompt(&mut self, prompt: &CudaSlice<f32>, n: usize) -> Result<(), GpuModelError> {
        let d = self.g.d;
        if n == 0 || n > self.g.max_prompt() {
            return Err(GpuModelError::Unsupported(format!(
                "sam3: a {n}-row prompt"
            )));
        }
        let exec = self.exec.clone();
        exec.copy_slice(prompt, 0, n * d, &mut self.ws.prompt)?;
        self.ws.n_prompt = n;
        exec.convert_f32_f16(&self.ws.prompt, &mut self.ws.prompt16, n * d)?;
        Ok(())
    }
}
