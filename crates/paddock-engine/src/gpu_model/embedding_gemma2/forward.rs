//! One packed pass through the text backbone, enqueued without a host sync;
//! the pooled vectors land in a flip buffer behind a recorded event.
//!
//! Every GEMM input arrives int8-quantized from the kernel before it (the row
//! norms, the sandwich seam and the GEGLU write the mmq layout directly), so
//! a layer is 13 launches plus the PLE: q|k|v GEMM, head transform,
//! attention, its quantize, o GEMM, sandwich (+ ffn_norm), gate|up GEMM,
//! GEGLU, down GEMM, sandwich (raw x for the PLE gate), inp_gate GEMM, PLE
//! GEGLU, proj GEMM and the sandwich that carries the layer scalar and the
//! next layer's attn_norm.

use crate::gpu::RepackedQ8;

use super::*;

impl GpuEmbeddingGemma2 {
    /// Grow the scratch to hold `rows` packed rows and `seqs` sequences
    /// (rounded up, so a stream of similar passes does not reallocate each
    /// time). Frees are stream-ordered, so dropping the old planes behind an
    /// in-flight pass is safe.
    fn ensure_scratch(&mut self, rows: usize, seqs: usize) -> Result<(), GpuModelError> {
        if self
            .scratch
            .as_ref()
            .is_some_and(|s| s.rows_cap >= rows && s.seq_cap >= seqs)
        {
            return Ok(());
        }
        let (r0, s0) = self
            .scratch
            .as_ref()
            .map_or((0, 0), |s| (s.rows_cap, s.seq_cap));
        let rows_cap = rows
            .max(r0)
            .next_multiple_of(256)
            .min(self.capacity.max(2))
            .max(rows);
        let seq_cap = seqs.max(s0).next_multiple_of(16);
        self.scratch = None;
        self.scratch = Some(Scratch::new(&self.exec, rows_cap, seq_cap)?);
        Ok(())
    }

    pub(super) fn submit(
        &mut self,
        seqs: &[Vec<u32>],
        media: &[Vec<crate::service::MmChunk>],
        dims: usize,
    ) -> Result<PendingEmbedding, GpuModelError> {
        let rows: usize = seqs.iter().map(Vec::len).sum();
        // the PLE tile refuses a single row; a lone one-token pass carries a
        // copy of its row that no sequence owns (attention and pooling read
        // `cu`, so it is never seen)
        let pad = rows.max(2);
        let [r64, r16] = crate::gpu::EG2_ATTN_ROWS;
        let mut ids = Vec::with_capacity(pad);
        let mut pos = Vec::with_capacity(pad);
        let mut cu = Vec::with_capacity(seqs.len() + 1);
        let mut tiles = [Vec::new(), Vec::new()];
        cu.push(0u32);
        for (s, seq) in seqs.iter().enumerate() {
            ids.extend_from_slice(seq);
            pos.extend(0..seq.len() as u32);
            cu.push(ids.len() as u32);
            for (k, per) in [r64, r16].into_iter().enumerate() {
                for t in 0..seq.len().div_ceil(per) {
                    tiles[k].push(((s as u32) << crate::gpu::EG2_TILE_SHIFT) | t as u32);
                }
            }
        }
        while ids.len() < pad {
            ids.push(ids[0]);
            pos.push(0);
        }
        self.ensure_scratch(pad, seqs.len())?;
        let exec = self.exec.clone();
        let eps = self.eps;
        let sc = self.scratch.as_mut().expect("scratch");
        exec.upload_u32(&ids, &mut sc.ids)?;
        exec.upload_u32(&pos, &mut sc.pos)?;
        exec.upload_u32(&cu, &mut sc.cu)?;
        exec.upload_u32(&tiles[0], &mut sc.tiles[0])?;
        exec.upload_u32(&tiles[1], &mut sc.tiles[1])?;
        let gemm = |w: &RepackedQ8, yq: &CudaSlice<u8>, y: &mut CudaSlice<f32>| {
            if pad <= GEMV_ROWS {
                exec.eg2_gemm_rows(w, yq, y, pad)
            } else {
                exec.q8_0_gemm_mmq(w, yq, None, y, pad)
            }
        };

        // x0 = E[ids] * sqrt(512): the residual stream's start and the PLE
        // projection's input for every layer
        let root = (WIDTH as f32).sqrt();
        exec.embed_gather_q8(&self.embd, &sc.ids, &mut sc.x0, WIDTH, pad, root)?;
        // media soft tokens: the tower's rows over the gathered placeholder
        // rows, unscaled, in both the residual stream and the PLE input
        let mut base = 0usize;
        for (i, seq) in seqs.iter().enumerate() {
            let items = media.get(i).map_or(&[][..], Vec::as_slice);
            for (run, item) in super::media::placeholder_runs(seq).iter().zip(items) {
                let (embd, n_tokens) = match item {
                    crate::service::MmChunk::Image { rgb, w, h } => {
                        let o = self
                            .images
                            .as_mut()
                            .ok_or(GpuModelError::MissingMeta("no picture tower".into()))?
                            .encode(rgb, *w, *h)?;
                        (o.embd, o.n_tokens)
                    }
                    crate::service::MmChunk::Audio { samples, .. } => {
                        let o = self
                            .audio
                            .as_ref()
                            .ok_or(GpuModelError::MissingMeta("no audio tower".into()))?
                            .encode(samples)?;
                        (o.embd, o.n_tokens)
                    }
                    _ => {
                        return Err(GpuModelError::MissingMeta(
                            "only pictures and audio are served as media".into(),
                        ));
                    }
                };
                if n_tokens != run.len {
                    return Err(GpuModelError::MissingMeta(format!(
                        "the tower made {n_tokens} rows for a {}-token run",
                        run.len
                    )));
                }
                exec.copy_region(
                    &embd,
                    0,
                    &mut sc.x0,
                    (base + run.start) * WIDTH,
                    run.len * WIDTH,
                )?;
            }
            base += seq.len();
        }
        exec.copy_region(&sc.x0, 0, &mut sc.x, 0, pad * WIDTH)?;
        let upfront = pad <= PLE_UPFRONT_ROWS;
        if upfront {
            // every layer's PLE at once, normalized into layer-major planes
            let all = WIDTH * LAYERS;
            exec.eg2_ple_gemm(&self.ple_proj, 0, all, &sc.x0, &mut sc.ple_raw, pad)?;
            exec.eg2_rms(
                &sc.ple_raw,
                &self.ple_norm,
                Some(&mut sc.ple),
                None,
                pad * LAYERS,
                pad,
                1.0 / root,
                eps,
            )?;
        }
        exec.eg2_rms(
            &sc.x,
            &self.layers[0].attn_norm,
            None,
            Some(&mut sc.yq),
            pad,
            0,
            1.0,
            eps,
        )?;

        for (i, l) in self.layers.iter().enumerate() {
            let hd = l.hd;
            let full = hd == 512;
            // this layer's PLE plane: a [pad][512] slice either way
            let ple_off = if upfront {
                i * pad * WIDTH
            } else {
                exec.eg2_ple_gemm(
                    &self.ple_proj,
                    i * WIDTH,
                    WIDTH,
                    &sc.x0,
                    &mut sc.ple_raw,
                    pad,
                )?;
                exec.eg2_rms(
                    &sc.ple_raw,
                    &self.ple_norm,
                    Some(&mut sc.ple),
                    None,
                    pad,
                    0,
                    1.0 / root,
                    eps,
                )?;
                0
            };

            // attention (yq holds attn_norm(x), quantized)
            gemm(&l.qkv, &sc.yq, &mut sc.qkv)?;
            exec.eg2_heads(
                &sc.qkv,
                &sc.pos,
                &l.q_norm,
                &l.k_norm,
                &mut sc.q16,
                &mut sc.k16,
                &mut sc.v16,
                pad,
                4 * hd + 1024,
                hd,
                self.theta_scale[full as usize],
                eps,
            )?;
            exec.eg2_attn(
                &sc.q16,
                &sc.k16,
                &sc.v16,
                &sc.cu,
                &sc.tiles[full as usize],
                tiles[full as usize].len(),
                &mut sc.attn,
                pad,
                hd,
                if full { 0 } else { self.window },
            )?;
            exec.quantize_q8_mmq(&sc.attn, &mut sc.yq, 4 * hd, pad)?;
            gemm(&l.o, &sc.yq, &mut sc.delta)?;
            exec.eg2_sandwich(
                &mut sc.x,
                &sc.delta,
                &l.post_attn,
                Some((&l.ffn_norm, None)),
                Some(&mut sc.yq),
                pad,
                1.0,
                eps,
            )?;

            // feed-forward: one gate|up GEMM, gelu(gate) * up, down
            gemm(&l.gate_up, &sc.yq, &mut sc.gate_up)?;
            exec.eg2_geglu_q(&sc.gate_up, 0, &sc.gate_up, FF, 2 * FF, &mut sc.yq, FF, pad)?;
            gemm(&l.down, &sc.yq, &mut sc.delta)?;
            exec.eg2_sandwich(
                &mut sc.x,
                &sc.delta,
                &l.post_ffw,
                None,
                Some(&mut sc.yq),
                pad,
                1.0,
                eps,
            )?;

            // the PLE block: gelu(inp_gate . x) * PLE_l -> proj, then the
            // layer scalar on the sum; the next norm rides the same seam
            gemm(&l.inp_gate, &sc.yq, &mut sc.g)?;
            exec.eg2_geglu_q(&sc.g, 0, &sc.ple, ple_off, WIDTH, &mut sc.yq, WIDTH, pad)?;
            gemm(&l.proj, &sc.yq, &mut sc.delta)?;
            let next = self
                .layers
                .get(i + 1)
                .map_or(&self.out_norm, |n| &n.attn_norm);
            exec.eg2_sandwich(
                &mut sc.x,
                &sc.delta,
                &l.post_norm,
                Some((next, None)),
                Some(&mut sc.yq),
                pad,
                l.scale,
                eps,
            )?;
        }

        // output_norm(x) (already quantized) -> 768 per token -> mean pool +
        // L2 over `dims`
        gemm(&self.output, &sc.yq, &mut sc.tok)?;
        let flip = self.flip;
        self.flip ^= 1;
        if self.pool[flip]
            .as_ref()
            .is_none_or(|b| b.len() < seqs.len() * dims)
        {
            self.pool[flip] = Some(exec.alloc(seqs.len().next_multiple_of(16) * DIM)?);
        }
        let out = self.pool[flip].as_mut().expect("pool buffer");
        exec.eg2_pool(&sc.tok, &sc.cu, seqs.len(), dims, out)?;
        let ev = exec.record_event()?;
        Ok(PendingEmbedding {
            ev,
            flip,
            count: seqs.len(),
            dims,
        })
    }
}
