//! The DETR decoder, the presence head and the dot-product scorer.
//!
//! 200 learned queries and one presence token (kept as the LAST of the 201
//! rows), six post-norm layers, each:
//!   query positions = MLP(sine(current boxes)); box-bias tables from the boxes
//!   self-attention over all 201 rows (q = k = tgt + pos, v = tgt) -> LN
//!   text cross-attention to the prompt (q = tgt + pos)              -> LN
//!   image cross-attention to E (q = tgt + pos, k = E + pos, v = E,
//!     the box bias on the scores; none for the presence row)        -> LN
//!   ReLU FFN                                                         -> LN
//!   hs = output LN(tgt); boxes = sigmoid(box_head(hs) + inverse_sigmoid(boxes))
//! The presence token never gets a position or a bias (Meta pads both with
//! zeros) and is read once, after the last layer: LN -> MLP -> its logit.
//!
//! Scorer (Meta's DotProductScoring): prompt' = LN(prompt + MLP(prompt)),
//! mean over the valid rows, a projection, then per query
//! clamp(query_proj(hs) . pooled / 16, +-12). The processor's score is
//! sigmoid(that) * sigmoid(presence).
//!
//! Meta clamps the presence logit with a non-in-place `.clamp()` whose result
//! is discarded - a no-op, reproduced as one.

use super::GpuModelError;
use super::detector::{BOX_SPLITS, GpuSam3Detector, LN_EPS};

impl GpuSam3Detector {
    /// Run the decoder and the scorer over the memory [`Self::fuse`] (or
    /// [`Self::set_memory`]) left and the current prompt.
    pub fn decode(&mut self) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let g = self.g;
        let (d, t, hd, heads, nq) = (g.d, g.tokens(), g.hd(), g.heads, g.queries);
        let rows = g.dec_rows();
        let np = self.ws.n_prompt;
        if np == 0 {
            return Err(GpuModelError::Unsupported(
                "sam3: decode before a prompt".into(),
            ));
        }
        let dec = &self.dec;
        let ws = &mut self.ws;

        exec.copy_region(&dec.query_embed, 0, &mut ws.tgt, 0, nq * d)?;
        exec.copy_region(&dec.presence_token, 0, &mut ws.tgt, nq * d, d)?;
        exec.copy_slice(&dec.ref_init, 0, nq * 4, &mut ws.reference)?;

        for l in &dec.layers {
            // ---- this layer's query positions and box-bias tables ----
            exec.sam3_box_sine(&ws.reference, &mut ws.qsine16, nq, d / 2, 0, 2 * d, 10000.0)?;
            let [rh0, rh1] = &dec.ref_head;
            exec.matvec_batch_f16_h_relu(&rh0.w, &ws.qsine16, &mut ws.qhid16, Some(&rh0.b), nq)?;
            exec.matvec_batch_f16(&rh1.w, &ws.qhid16, &mut ws.qpos, nq)?;
            exec.bias_add(&mut ws.qpos, &rh1.b, nq, d)?;
            exec.sam3_rpb_tables(
                &ws.reference,
                (&dec.rpb_x.w1, &dec.rpb_x.b1, &dec.rpb_x.w2, &dec.rpb_x.b2),
                (&dec.rpb_y.w1, &dec.rpb_y.b1, &dec.rpb_y.w2, &dec.rpb_y.b2),
                &mut ws.rpb_x,
                &mut ws.rpb_y,
                nq,
                g.grid,
                g.grid,
                d,
                heads,
            )?;
            let qp = Some((&ws.qpos, rows, nq));
            // the stream's f16 views under this layer's positions
            exec.sam3_seam_h(
                &mut ws.tgt,
                None,
                None,
                None,
                qp,
                Some(&mut ws.t16),
                Some(&mut ws.tq16),
                rows,
                d,
                LN_EPS,
                false,
            )?;

            // ---- self-attention over all rows ----
            exec.matvec_batch_f16_h_bias(&l.sa.q.w, &ws.tq16, &mut ws.dq16, Some(&l.sa.q.b), rows)?;
            exec.matvec_batch_f16_h_bias(&l.sa.k.w, &ws.tq16, &mut ws.dk16, Some(&l.sa.k.b), rows)?;
            exec.matvec_batch_f16_h_bias(&l.sa.v.w, &ws.t16, &mut ws.dv16, Some(&l.sa.v.b), rows)?;
            exec.vision_attn_h(
                &ws.dq16,
                &ws.dk16,
                &ws.dv16,
                &mut ws.datt16,
                rows,
                rows,
                heads,
                hd,
                1,
            )?;
            exec.matvec_batch_f16_h(&l.sa.o.w, &ws.datt16, &mut ws.dproj16, rows)?;
            exec.sam3_seam_h(
                &mut ws.tgt,
                Some(&ws.dproj16),
                Some(&l.sa.o.b),
                Some((&l.sa_norm.w, &l.sa_norm.b)),
                qp,
                Some(&mut ws.t16),
                Some(&mut ws.tq16),
                rows,
                d,
                LN_EPS,
                true,
            )?;

            // ---- text cross-attention ----
            exec.matvec_batch_f16_h_bias(
                &l.ca_text.q.w,
                &ws.tq16,
                &mut ws.dq16,
                Some(&l.ca_text.q.b),
                rows,
            )?;
            exec.matvec_batch_f16_h_bias(
                &l.ca_text.k.w,
                &ws.prompt16,
                &mut ws.gk16,
                Some(&l.ca_text.k.b),
                np,
            )?;
            exec.matvec_batch_f16_h_bias(
                &l.ca_text.v.w,
                &ws.prompt16,
                &mut ws.gv16,
                Some(&l.ca_text.v.b),
                np,
            )?;
            exec.vision_attn_h(
                &ws.dq16,
                &ws.gk16,
                &ws.gv16,
                &mut ws.datt16,
                rows,
                np,
                heads,
                hd,
                1,
            )?;
            exec.matvec_batch_f16_h(&l.ca_text.o.w, &ws.datt16, &mut ws.dproj16, rows)?;
            exec.sam3_seam_h(
                &mut ws.tgt,
                Some(&ws.dproj16),
                Some(&l.ca_text.o.b),
                Some((&l.ca_text_norm.w, &l.ca_text_norm.b)),
                qp,
                Some(&mut ws.t16),
                Some(&mut ws.tq16),
                rows,
                d,
                LN_EPS,
                true,
            )?;

            // ---- image cross-attention with the box bias ----
            exec.matvec_batch_f16_h_bias(
                &l.ca_img.q.w,
                &ws.tq16,
                &mut ws.dq16,
                Some(&l.ca_img.q.b),
                rows,
            )?;
            exec.matvec_batch_f16_h_bias(
                &l.ca_img.k.w,
                &ws.memq16,
                &mut ws.dk16,
                Some(&l.ca_img.k.b),
                t,
            )?;
            exec.matvec_batch_f16_h_bias(
                &l.ca_img.v.w,
                &ws.mem16,
                &mut ws.dv16,
                Some(&l.ca_img.v.b),
                t,
            )?;
            if exec.sam3_box_attn_mma_fits(hd, g.grid, g.grid) {
                exec.sam3_box_attn_mma(
                    &ws.dq16,
                    &ws.dk16,
                    &ws.dv16,
                    &ws.rpb_x,
                    &ws.rpb_y,
                    &mut ws.box_part,
                    &mut ws.datt16,
                    rows,
                    heads,
                    hd,
                    g.grid,
                    g.grid,
                    nq,
                    BOX_SPLITS,
                )?;
            } else {
                exec.sam3_box_attn_h(
                    &ws.dq16,
                    &ws.dk16,
                    &ws.dv16,
                    &ws.rpb_x,
                    &ws.rpb_y,
                    &mut ws.datt16,
                    rows,
                    heads,
                    hd,
                    g.grid,
                    g.grid,
                    nq,
                )?;
            }
            exec.matvec_batch_f16_h(&l.ca_img.o.w, &ws.datt16, &mut ws.dproj16, rows)?;
            exec.sam3_seam_h(
                &mut ws.tgt,
                Some(&ws.dproj16),
                Some(&l.ca_img.o.b),
                Some((&l.ca_img_norm.w, &l.ca_img_norm.b)),
                None,
                Some(&mut ws.t16),
                None,
                rows,
                d,
                LN_EPS,
                true,
            )?;

            // ---- FFN ----
            exec.matvec_batch_f16_h_relu(&l.fc1.w, &ws.t16, &mut ws.dff16, Some(&l.fc1.b), rows)?;
            exec.matvec_batch_f16_h(&l.fc2.w, &ws.dff16, &mut ws.dproj16, rows)?;
            exec.sam3_seam_h(
                &mut ws.tgt,
                Some(&ws.dproj16),
                Some(&l.fc2.b),
                Some((&l.mlp_norm.w, &l.mlp_norm.b)),
                None,
                None,
                None,
                rows,
                d,
                LN_EPS,
                true,
            )?;

            // ---- output norm and box refinement (queries are rows 0..nq) ----
            exec.layernorm(
                &ws.tgt,
                &dec.out_norm.w,
                &dec.out_norm.b,
                &mut ws.hs,
                rows,
                d,
                LN_EPS,
            )?;
            exec.convert_f32_f16(&ws.hs, &mut ws.hs16, rows * d)?;
            let [b0, b1, b2] = &dec.box_head;
            exec.matvec_batch_f16_h_relu(&b0.w, &ws.hs16, &mut ws.mlp_a16, Some(&b0.b), nq)?;
            exec.matvec_batch_f16_h_relu(&b1.w, &ws.mlp_a16, &mut ws.mlp_b16, Some(&b1.b), nq)?;
            exec.matvec_batch_f16(&b2.w, &ws.mlp_b16, &mut ws.box_delta, nq)?;
            exec.sam3_box_refine(&mut ws.reference, &ws.box_delta, Some(&b2.b), nq, 4)?;
        }

        // ---- presence, read once after the last layer ----
        exec.copy_region(&ws.tgt, nq * d, &mut ws.pres32, 0, d)?;
        exec.layernorm(
            &ws.pres32,
            &dec.presence_norm.w,
            &dec.presence_norm.b,
            &mut ws.pooled,
            1,
            d,
            LN_EPS,
        )?;
        exec.convert_f32_f16(&ws.pooled, &mut ws.pres16, d)?;
        let [p0, p1, p2] = &dec.presence_head;
        exec.matvec_batch_f16_h_relu(&p0.w, &ws.pres16, &mut ws.mlp_a16, Some(&p0.b), 1)?;
        exec.matvec_batch_f16_h_relu(&p1.w, &ws.mlp_a16, &mut ws.mlp_b16, Some(&p1.b), 1)?;
        exec.matvec_batch_f16(&p2.w, &ws.mlp_b16, &mut ws.presence, 1)?;
        exec.bias_add(&mut ws.presence, &p2.b, 1, 1)?;

        // ---- scorer ----
        let sc = &self.scorer;
        exec.copy_slice(&ws.prompt, 0, np * d, &mut ws.sp)?;
        exec.matvec_batch_f16_h_relu(
            &sc.mlp1.w,
            &ws.prompt16,
            &mut ws.gff16,
            Some(&sc.mlp1.b),
            np,
        )?;
        exec.matvec_batch_f16_h(&sc.mlp2.w, &ws.gff16, &mut ws.gproj16, np)?;
        exec.sam3_seam_h(
            &mut ws.sp,
            Some(&ws.gproj16),
            Some(&sc.mlp2.b),
            Some((&sc.mlp_norm.w, &sc.mlp_norm.b)),
            None,
            None,
            None,
            np,
            d,
            LN_EPS,
            true,
        )?;
        exec.gather_rows_avg(&ws.sp, &ws.pool_idx, &mut ws.pooled, 1, np, d)?;
        exec.convert_f32_f16(&ws.pooled, &mut ws.pooled16, d)?;
        exec.matvec_batch_f16(&sc.text_proj.w, &ws.pooled16, &mut ws.pp, 1)?;
        exec.bias_add(&mut ws.pp, &sc.text_proj.b, 1, d)?;
        exec.matvec_batch_f16(&sc.query_proj.w, &ws.hs16, &mut ws.hp, nq)?;
        exec.bias_add(&mut ws.hp, &sc.query_proj.b, nq, d)?;
        exec.sam3_score(
            &ws.hp,
            &ws.pp,
            &ws.presence,
            &mut ws.logits,
            &mut ws.probs,
            nq,
            d,
            1.0 / (d as f32).sqrt(),
            12.0,
        )?;
        Ok(())
    }
}
