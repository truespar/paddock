//! One packed pass through the body, enqueued without a host sync; the
//! pooled outputs land in a flip buffer behind a recorded event.
//!
//! A layer is eight launches: the q|k|v GEMM (f16 landing, its input the
//! previous seam's norm), the rope + q scale in place, attention, o landed
//! plain f32, the residual add + the FFN's norm in one pass, the gate|up GEMM
//! with silu(gate) * up landed, down landed plain, the residual add + the next
//! layer's norm.

use super::*;
use crate::gpu::{NE_EMB_PIC, NE_TILE_SHIFT, ne_attn_rows};

impl GpuNemotronEmbed {
    /// Grow the pass planes to hold `rows` packed rows, `seqs` sequences
    /// and `pics` picture rows (rounded up, so a stream of similar passes
    /// does not reallocate each time). Frees are stream-ordered, so dropping
    /// the old planes behind an in-flight pass is safe.
    fn ensure_scratch(
        &mut self,
        rows: usize,
        seqs: usize,
        pics: usize,
    ) -> Result<(), GpuModelError> {
        if self
            .scratch
            .as_ref()
            .is_some_and(|s| s.rows_cap >= rows && s.seq_cap >= seqs && s.pic_cap >= pics)
        {
            return Ok(());
        }
        let (r0, s0, p0) = self
            .scratch
            .as_ref()
            .map_or((0, 0, 0), |s| (s.rows_cap, s.seq_cap, s.pic_cap));
        let budget = self.coalesce_row_budget();
        let rows_cap = rows.max(r0).next_multiple_of(512).min(budget).max(rows);
        let seq_cap = seqs.max(s0).next_multiple_of(16);
        let pic_cap = if pics == 0 {
            p0
        } else {
            pics.max(p0).next_multiple_of(256).min(budget).max(pics)
        };
        let c = &self.cfg;
        let (d, nh, nkv, hd, ffn) = (c.d_model, c.n_heads, c.n_kv_heads, c.head_dim, c.ffn);
        let e = &self.exec;
        self.scratch = None;
        self.scratch = Some(Scratch {
            rows_cap,
            seq_cap,
            pic_cap,
            src: e.alloc_u32(rows_cap)?,
            pos: e.alloc_u32(rows_cap)?,
            cu: e.alloc_u32(seq_cap + 1)?,
            tiles: e.alloc_u32(rows_cap / ne_attn_rows(nh / nkv) + seq_cap)?,
            x: e.alloc(rows_cap * d)?,
            proj: e.alloc(rows_cap * d)?,
            xn16: e.alloc_f16(rows_cap * d)?,
            qkv16: e.alloc_f16(rows_cap * (nh + 2 * nkv) * hd)?,
            attn16: e.alloc_f16(rows_cap * nh * hd)?,
            ff16: e.alloc_f16(rows_cap * ffn)?,
            pic: e.alloc(pic_cap.max(1) * d)?,
            mean: e.alloc(seq_cap * d)?,
        });
        Ok(())
    }

    pub(super) fn submit(
        &mut self,
        seqs: &[Vec<u32>],
        media: &[Vec<crate::service::MmChunk>],
        output: Output,
    ) -> Result<PendingNemotron, GpuModelError> {
        let c = self.cfg.clone();
        let (d, nh, nkv, hd) = (c.d_model, c.n_heads, c.n_kv_heads, c.head_dim);
        let img = c.image_token_id;
        let rows: usize = seqs.iter().map(Vec::len).sum();
        let per = ne_attn_rows(nh / nkv);
        // the body's input rows: a token id, or the next picture row in pass
        // order (the reference's masked_scatter fills [IMG] in order)
        let mut src = Vec::with_capacity(rows);
        let mut pos = Vec::with_capacity(rows);
        let mut cu = Vec::with_capacity(seqs.len() + 1);
        let mut tiles = Vec::new();
        let mut pic_row = 0u32;
        cu.push(0u32);
        for (s, seq) in seqs.iter().enumerate() {
            for &t in seq {
                if t == img {
                    src.push(NE_EMB_PIC | pic_row);
                    pic_row += 1;
                } else {
                    src.push(t);
                }
            }
            pos.extend(0..seq.len() as u32);
            cu.push(src.len() as u32);
            for t in 0..seq.len().div_ceil(per) {
                tiles.push(((s as u32) << NE_TILE_SHIFT) | t as u32);
            }
        }
        let n_pic = pic_row as usize;
        self.ensure_scratch(rows, seqs.len(), n_pic)?;
        let exec = self.exec.clone();
        // the pictures first: their rows land in `pic`, in pass order
        if n_pic > 0 {
            let pics: Vec<Picture<'_>> = media
                .iter()
                .flatten()
                .map(|m| match m {
                    crate::service::MmChunk::Image { rgb, w, h } => {
                        Ok(Picture { rgb, w: *w, h: *h })
                    }
                    _ => Err(GpuModelError::Unsupported(
                        "only pictures are served as media".into(),
                    )),
                })
                .collect::<Result<_, _>>()?;
            let tower = self
                .tower
                .as_mut()
                .ok_or_else(|| GpuModelError::Unsupported("no picture tower loaded".into()))?;
            let sc = self.scratch.as_mut().expect("scratch");
            let got = tower.encode(&exec, &pics, &mut sc.pic, 0)?;
            if got != n_pic {
                return Err(GpuModelError::Unsupported(format!(
                    "the tower made {got} rows for {n_pic} picture tokens"
                )));
            }
        }
        let sc = self.scratch.as_mut().expect("scratch");
        exec.upload_u32(&src, &mut sc.src)?;
        exec.upload_u32(&pos, &mut sc.pos)?;
        exec.upload_u32(&cu, &mut sc.cu)?;
        exec.upload_u32(&tiles, &mut sc.tiles)?;
        exec.ne_embed_rows(
            &sc.src,
            &self.emb,
            c.vocab,
            Some((&sc.pic, n_pic)),
            &mut sc.x,
            rows,
            d,
        )?;
        let stride = (nh + 2 * nkv) * hd;
        let qscale = std::f32::consts::LOG2_E / (hd as f32).sqrt();
        exec.ne_rms_h(&sc.x, &self.layers[0].ln1, &mut sc.xn16, rows, d, c.rms_eps)?;
        for (li, l) in self.layers.iter().enumerate() {
            exec.matvec_batch_f16_h(&l.qkv, &sc.xn16, &mut sc.qkv16, rows)?;
            exec.ne_rope_qk(
                &mut sc.qkv16,
                &sc.pos,
                &self.rope_cos,
                &self.rope_sin,
                self.context,
                rows,
                nh,
                nkv,
                hd,
                stride,
                qscale,
                Some((c.l4_beta, c.l4_orig)),
            )?;
            exec.ne_attn(
                &sc.qkv16,
                &sc.cu,
                &sc.tiles,
                tiles.len(),
                &mut sc.attn16,
                rows,
                nh,
                nkv,
                hd,
            )?;
            exec.f16_gemm_ring_rows(&l.o, &sc.attn16, 0, &mut sc.proj, 0, rows, None)?;
            exec.ne_add_rms_h(
                &mut sc.x,
                &sc.proj,
                Some((&l.ln2, &mut sc.xn16)),
                rows,
                d,
                c.rms_eps,
            )?;
            exec.f16_gemm_h_swiglu(&l.gu, &sc.xn16, &mut sc.ff16, rows)?;
            exec.f16_gemm_ring_rows(&l.down, &sc.ff16, 0, &mut sc.proj, 0, rows, None)?;
            // the residual add, and the next layer's norm where there is one
            let next = self.layers.get(li + 1).map(|n| &n.ln1);
            exec.ne_add_rms_h(
                &mut sc.x,
                &sc.proj,
                next.map(|w| (w, &mut sc.xn16)),
                rows,
                d,
                c.rms_eps,
            )?;
        }
        exec.rmsnorm_batch_inplace(&mut sc.x, &self.norm, d, c.rms_eps, rows)?;

        let n = seqs.len();
        let width = match output {
            Output::Embed { dims } => dims,
            Output::Score => 1,
            Output::Mean => d,
        };
        let flip = self.flip;
        self.flip ^= 1;
        if self.out[flip].as_ref().is_none_or(|b| b.len() < n * width) {
            self.out[flip] = Some(exec.alloc(n.next_multiple_of(16) * width)?);
        }
        let out = self.out[flip].as_mut().expect("output plane");
        match output {
            Output::Embed { dims } => {
                exec.ne_pool(&sc.x, &sc.cu, None, None, out, n, rows, d, dims)?;
            }
            Output::Score => {
                let score = self
                    .score
                    .as_ref()
                    .ok_or_else(|| GpuModelError::Unsupported("no score head".into()))?;
                exec.ne_pool(&sc.x, &sc.cu, Some(score), None, out, n, rows, d, d)?;
            }
            Output::Mean => {
                // the mean lands in the pass plane, then into the output
                let mut unit = exec.alloc(n * d)?;
                exec.ne_pool(
                    &sc.x,
                    &sc.cu,
                    None,
                    Some(&mut sc.mean),
                    &mut unit,
                    n,
                    rows,
                    d,
                    d,
                )?;
                exec.copy_region(&sc.mean, 0, out, 0, n * d)?;
            }
        }
        let ev = exec.record_event()?;
        Ok(PendingNemotron {
            ev,
            flip,
            count: n,
            width,
        })
    }
}
