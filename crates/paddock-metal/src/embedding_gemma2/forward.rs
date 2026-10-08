use super::*;
use crate::device::Commands;
#[cfg(test)]
use objc2_metal::MTLBuffer;

impl EmbeddingGemma2 {
    // Follow the unmodified MLX 0.32.3 single-input arithmetic on M5, never
    // its flattened serving-batch shape. Adjacent requests with the same
    // reduction can share a dispatch; their arrival order cannot alter it.
    // Qualification of this contract is separate from the retained original
    // batched-reference test (upstream itself changes across batch sizes).
    pub(super) fn text_linear(
        &self,
        c: &Commands<'_>,
        w: &Weight,
        x: &Buffer,
        y: &Buffer,
        seqs: &[Vec<u32>],
    ) {
        if w.ty != 0x108 {
            self.linear(c, w, x, y, seqs.iter().map(Vec::len).sum());
            return;
        }
        let plan = |len: usize| {
            let vector = len < if w.k <= 2048 && w.n <= 2048 { 33 } else { 13 };
            if vector {
                return (true, 1);
            }
            let mut splits = (512 / (w.n.div_ceil(32) * len.div_ceil(32)))
                .max(1)
                .min(w.k / 64);
            while !w.k.is_multiple_of(splits * 64) {
                splits -= 1;
            }
            (false, splits)
        };
        let mut first = 0;
        for group in seqs.chunk_by(|a, b| plan(a.len()) == plan(b.len())) {
            let (vector, splits) = plan(group[0].len());
            let rows: usize = group.iter().map(Vec::len).sum();
            let wide = rows > 128 && splits == 1;
            c.dispatch_at(
                if vector {
                    "eg2_affine_vector"
                } else if splits > 1 {
                    "eg2_project_a8_partition"
                } else if wide {
                    "eg2_project_a8"
                } else {
                    "eg2_project_a8_narrow"
                },
                &[&w.buffer, x, y],
                &[0, first * w.k * 4, first * w.n * 4],
                &[w.k as u32, w.n as u32, rows as u32, (w.k / splits) as u32],
                if vector {
                    [w.n.div_ceil(16), rows, 1]
                } else if wide {
                    [w.n.div_ceil(64), rows.div_ceil(32), 1]
                } else {
                    [w.n.div_ceil(16), rows.div_ceil(16), 1]
                },
                128,
            );
            first += rows;
        }
    }
    pub(super) fn linear(&self, c: &Commands<'_>, w: &Weight, x: &Buffer, y: &Buffer, rows: usize) {
        if rows <= 128 && matches!(w.ty, 0x108 | 8) {
            c.dispatch(
                if w.ty == 0x108 {
                    "eg2_project_a8_narrow"
                } else {
                    "eg2_project_q8_narrow"
                },
                &[&w.buffer, x, y],
                &[w.k as u32, w.n as u32, rows as u32],
                [w.n.div_ceil(16), rows.div_ceil(16), 1],
                128,
            );
            return;
        }
        let tiled = match w.ty {
            0x100 => Some("eg2_project_a4"),
            0x108 => Some("eg2_project_a8"),
            30 => Some(if self.mlx {
                "eg2_project_bf16"
            } else {
                "eg2_project_gguf_bf16"
            }),
            8 => Some("eg2_project_q8"),
            12 => Some("eg2_project_q4"),
            14 => Some("eg2_project_q6"),
            _ => None,
        };
        if let Some(kernel) = tiled {
            c.dispatch(
                kernel,
                &[&w.buffer, x, y],
                &[w.k as u32, w.n as u32, rows as u32],
                [w.n.div_ceil(64), rows.div_ceil(32), 1],
                128,
            );
            return;
        }
        let kernel = match w.ty {
            0 => "dg_project_f32",
            8 => "dg_project_q8",
            12 => "dg_project_q4",
            14 => "dg_project_q6",
            _ => "dg_project",
        };
        c.dispatch(
            kernel,
            &[&w.buffer, x, y],
            &[w.k as u32, w.n as u32, rows as u32, w.ty, self.mlx as u32],
            [w.n.div_ceil(32), rows.div_ceil(16), 1],
            128,
        );
        self.round(c, y, rows * w.n);
    }
    fn round(&self, c: &Commands<'_>, b: &Buffer, n: usize) {
        if self.mlx {
            c.dispatch(
                "gmlx_round",
                &[b],
                &[n as u32],
                [n.div_ceil(256), 1, 1],
                256,
            );
        }
    }
    pub(super) fn norm(
        &self,
        c: &Commands<'_>,
        x: &Buffer,
        w: &Weight,
        y: &Buffer,
        rows: usize,
        scale: f32,
        residual: Option<(&Buffer, &Weight)>,
    ) {
        let (r, s) = residual.unwrap_or((x, w));
        c.dispatch(
            "eg2_norm",
            &[x, &w.buffer, y, r, &s.buffer],
            &[
                WIDTH as u32,
                w.ty,
                self.mlx as u32,
                scale.to_bits(),
                residual.is_some() as u32,
                s.ty,
            ],
            [rows, 1, 1],
            128,
        );
    }
    pub(super) fn submit(
        &mut self,
        seqs: &[Vec<u32>],
        dimensions: usize,
    ) -> Result<PendingEmbedding> {
        self.submit_with_media(seqs, dimensions, Vec::new())
    }
    pub(super) fn submit_with_media(
        &mut self,
        seqs: &[Vec<u32>],
        dimensions: usize,
        media: Vec<super::media::Injection>,
    ) -> Result<PendingEmbedding> {
        let rows: usize = seqs.iter().map(Vec::len).sum();
        let mut ids = Vec::with_capacity(rows);
        let mut meta = Vec::with_capacity(rows * 2);
        let mut ends = Vec::with_capacity(rows);
        let mut ranges = Vec::new();
        let mut tiles = Vec::new();
        for seq in seqs {
            let first = ids.len();
            ranges.extend_from_slice(&[first as u32, seq.len() as u32]);
            for pos in 0..seq.len() {
                meta.extend_from_slice(&[first as u32, pos as u32]);
                ends.push((seq.len() - 1) as u32);
            }
            for pos in (0..seq.len()).step_by(32) {
                tiles.extend_from_slice(&[(first + pos) as u32, (seq.len() - pos).min(32) as u32]);
            }
            ids.extend_from_slice(seq);
        }
        let upload = |v: &[u32]| -> Result<Buffer> {
            let b = self.device.alloc(std::mem::size_of_val(v))?;
            unsafe {
                b.write_u32(v);
            }
            Ok(b)
        };
        let nt = tiles.len() / 2;
        let inputs = [ids, meta, ends, ranges, tiles]
            .iter()
            .map(|v| upload(v))
            .collect::<Result<Vec<_>>>()?;
        let [ids, meta, ends, ranges, tiles] = inputs.as_slice() else {
            unreachable!()
        };
        let output = self.device.alloc(seqs.len() * dimensions * 4)?;
        #[cfg(test)]
        let trace = std::env::var_os("PADDOCK_EG2_TRACE")
            .map(|_| self.device.alloc((LAYERS + 1) * rows * WIDTH * 4))
            .transpose()?;
        #[cfg(test)]
        if let Some(t) = &trace {
            // Stage captures leave reserved slots unused. Initialize every
            // byte before exporting the diagnostic; this fresh buffer has
            // not been submitted to the GPU yet.
            unsafe { std::ptr::write_bytes(t.raw.contents().as_ptr().cast::<u8>(), 0, t.len()) };
        }
        // Complete every fallible allocation before opening a command encoder.
        // Budget refusal must not abandon a partly encoded request.
        let one = Weight {
            buffer: self.device.upload(&1f32.to_le_bytes())?,
            ty: 0,
            k: 1,
            n: 1,
        };
        let sc = self.workspace(rows)?;
        let c = self.device.begin()?;
        #[cfg(test)]
        let stages = std::env::var_os("PADDOCK_EG2_TRACE_STAGES").is_some();
        #[cfg(test)]
        let stage_layer = std::env::var("PADDOCK_EG2_LAYER")
            .map(|s| s.parse::<usize>().unwrap())
            .unwrap_or(0);
        #[cfg(test)]
        if stages {
            assert!(
                stage_layer < LAYERS && stage_layer % 6 != 5,
                "stage-plane layout requires a local-attention layer"
            );
        }
        #[cfg(test)]
        let capture = |slot: usize, buffer: &Buffer, width: usize| {
            // Slots are WIDTH-wide planes; Q/attention use two and GEGLU
            // four. See --stages in the GPU oracle for the named seams.
            if stages && let Some(t) = &trace {
                c.dispatch(
                    "spec_copy",
                    &[buffer, t],
                    &[0, (slot * rows * WIDTH) as u32, (rows * width) as u32],
                    [(rows * width).div_ceil(256), 1, 1],
                    256,
                );
            }
        };
        c.dispatch(
            "eg2_embed",
            &[&self.embedding.buffer, ids, &sc.x],
            &[
                WIDTH as u32,
                rows as u32,
                self.embedding.ty,
                (WIDTH as f32).sqrt().to_bits(),
                VOCAB as u32,
                self.mlx as u32,
            ],
            [(rows * WIDTH).div_ceil(256), 1, 1],
            256,
        );
        self.round(&c, &sc.x, rows * WIDTH);
        for m in &media {
            c.dispatch(
                "vis_inject",
                &[&m.data, meta, &sc.x],
                &[
                    WIDTH as u32,
                    rows as u32,
                    m.first as u32,
                    m.run.start as u32,
                    m.run.len as u32,
                ],
                [(rows * WIDTH).div_ceil(256), 1, 1],
                256,
            );
        }
        #[cfg(test)]
        if let Some(t) = &trace {
            c.dispatch(
                "spec_copy",
                &[&sc.x, t],
                &[0, 0, (rows * WIDTH) as u32],
                [(rows * WIDTH).div_ceil(256), 1, 1],
                256,
            );
        }
        self.text_linear(&c, &self.ple, &sc.x, &sc.ple, seqs);
        self.norm(
            &c,
            &sc.ple,
            &self.ple_norm,
            &sc.ple,
            rows * LAYERS,
            (WIDTH as f32).sqrt().recip(),
            None,
        );
        for (i, l) in self.layers.iter().enumerate() {
            #[cfg(test)]
            if i == stage_layer {
                capture(0, &sc.x, WIDTH);
            }
            let hd = if i % 6 == 5 { 512 } else { 256 };
            let kh = 512 / hd;
            self.norm(&c, &sc.x, &l.pre, &sc.norm, rows, 1., None);
            #[cfg(test)]
            if i == stage_layer {
                capture(1, &sc.norm, WIDTH);
            }
            for (w, y) in [(&l.q, &sc.q), (&l.k, &sc.k), (&l.v, &sc.v)] {
                self.text_linear(&c, w, &sc.norm, y, seqs);
            }
            #[cfg(test)]
            if i == stage_layer {
                capture(2, &sc.q, 1024);
                capture(4, &sc.k, 512);
                capture(5, &sc.v, 512);
            }
            for (x, y, w, heads, rope) in [
                (
                    &sc.q,
                    &sc.gate,
                    &l.qn,
                    4,
                    if hd == 512 { 1_000_000f32 } else { 10_000f32 },
                ),
                (
                    &sc.k,
                    &sc.norm,
                    &l.kn,
                    kh,
                    if hd == 512 { 1_000_000f32 } else { 10_000f32 },
                ),
                (&sc.v, &sc.delta, &l.kn, kh, 0f32),
            ] {
                c.dispatch(
                    "eg2_heads",
                    &[x, &w.buffer, meta, y],
                    &[
                        heads as u32,
                        hd as u32,
                        w.ty,
                        self.mlx as u32,
                        rope.to_bits(),
                        if self.mlx { 1 } else { 2 },
                    ],
                    [heads, rows, 1],
                    hd / 4,
                );
            }
            if self.mlx {
                let mut start = 0;
                let mut tile_start = 0;
                for seq in seqs {
                    if hd == 256 && seq.len() >= 1024 {
                        c.dispatch_at(
                            "eg2_mlx_attention256",
                            &[&sc.gate, &sc.norm, &sc.delta, meta, ends, &sc.attn, tiles],
                            &[0, 0, 0, 0, 0, 0, tile_start * 8],
                            &[4, 2, 0, 512, 0, 0, 1],
                            [4, seq.len().div_ceil(32), 1],
                            128,
                        );
                        start += seq.len();
                        tile_start += seq.len().div_ceil(32);
                        continue;
                    }
                    for offset in (0..seq.len()).step_by(128) {
                        let count = (seq.len() - offset).min(128);
                        let p = [start as u32, seq.len() as u32, offset as u32, count as u32];
                        c.dispatch(
                            if hd == 512 {
                                "eg2_global_qk"
                            } else {
                                "eg2_local_qk"
                            },
                            &[&sc.gate, &sc.norm, &sc.scores],
                            &p,
                            [seq.len().div_ceil(64), count.div_ceil(32), 4],
                            128,
                        );
                        c.dispatch(
                            if seq.len() > 4096 {
                                "eg2_global_softmax"
                            } else {
                                "eg2_block_softmax"
                            },
                            &[&sc.scores],
                            &p,
                            [count, 4, 1],
                            if seq.len() > 4096 {
                                256
                            } else {
                                seq.len().div_ceil(128) * 32
                            },
                        );
                        c.dispatch(
                            if hd == 512 {
                                "eg2_global_pv"
                            } else {
                                "eg2_local_pv"
                            },
                            &[&sc.scores, &sc.delta, &sc.attn],
                            &p,
                            [hd / 64, count.div_ceil(32), 4],
                            128,
                        );
                    }
                    start += seq.len();
                    tile_start += seq.len().div_ceil(32);
                }
            } else {
                c.dispatch(
                    if hd == 512 {
                        "eg2_half_attention512"
                    } else {
                        "eg2_half_attention256"
                    },
                    &[&sc.gate, &sc.norm, &sc.delta, meta, ends, &sc.attn, tiles],
                    &[
                        4,
                        kh as u32,
                        0,
                        if hd == 512 { 0 } else { 512 },
                        0,
                        0,
                        self.mlx as u32,
                    ],
                    [4, nt, 1],
                    128,
                );
            }
            self.round(&c, &sc.attn, rows * 4 * hd);
            #[cfg(test)]
            if i == stage_layer {
                capture(6, &sc.attn, 1024);
            }
            self.text_linear(&c, &l.o, &sc.attn, &sc.delta, seqs);
            #[cfg(test)]
            if i == stage_layer {
                capture(8, &sc.delta, WIDTH);
            }
            self.norm(&c, &sc.delta, &l.post, &sc.x, rows, 1., Some((&sc.x, &one)));
            #[cfg(test)]
            if i == stage_layer {
                capture(9, &sc.x, WIDTH);
            }
            self.norm(&c, &sc.x, &l.ff_pre, &sc.norm, rows, 1., None);
            #[cfg(test)]
            if i == stage_layer {
                capture(10, &sc.norm, WIDTH);
            }
            self.text_linear(&c, &l.gate, &sc.norm, &sc.gate, seqs);
            self.text_linear(&c, &l.up, &sc.norm, &sc.up, seqs);
            c.dispatch(
                if self.mlx {
                    "gmlx_geglu"
                } else {
                    "gemma_geglu"
                },
                &[&sc.gate, &sc.up],
                &[(rows * FF) as u32],
                [(rows * FF).div_ceil(256), 1, 1],
                256,
            );
            self.text_linear(&c, &l.down, &sc.gate, &sc.delta, seqs);
            #[cfg(test)]
            if i == stage_layer {
                capture(11, &sc.gate, FF);
                capture(15, &sc.delta, WIDTH);
            }
            self.norm(
                &c,
                &sc.delta,
                &l.ff_post,
                &sc.x,
                rows,
                1.,
                Some((&sc.x, &one)),
            );
            self.text_linear(&c, &l.ple_gate, &sc.x, &sc.norm, seqs);
            #[cfg(test)]
            if i == stage_layer {
                capture(16, &sc.x, WIDTH);
                capture(17, &sc.norm, WIDTH);
            }
            c.dispatch(
                "eg2_ple_gate",
                &[&sc.norm, &sc.ple],
                &[rows as u32, i as u32, self.mlx as u32],
                [(rows * WIDTH).div_ceil(256), 1, 1],
                256,
            );
            self.text_linear(&c, &l.ple_out, &sc.norm, &sc.delta, seqs);
            #[cfg(test)]
            if i == stage_layer {
                capture(18, &sc.norm, WIDTH);
                capture(19, &sc.delta, WIDTH);
            }
            self.norm(
                &c,
                &sc.delta,
                &l.ple_norm,
                &sc.x,
                rows,
                1.,
                Some((&sc.x, &l.scalar)),
            );
            #[cfg(test)]
            if let Some(t) = &trace
                && !stages
            {
                c.dispatch(
                    "spec_copy",
                    &[&sc.x, t],
                    &[0, ((i + 1) * rows * WIDTH) as u32, (rows * WIDTH) as u32],
                    [(rows * WIDTH).div_ceil(256), 1, 1],
                    256,
                );
            }
            #[cfg(test)]
            if i == stage_layer {
                capture(20, &sc.x, WIDTH);
            }
        }
        self.norm(&c, &sc.x, &self.norm, &sc.norm, rows, 1., None);
        self.text_linear(&c, &self.output, &sc.norm, &sc.token_out, seqs);
        c.dispatch(
            "eg2_pool",
            &[&sc.token_out, ranges, &output],
            &[dimensions as u32],
            [seqs.len(), 1, 1],
            256,
        );
        #[cfg(test)]
        let completion = c.submit_for_test()?;
        #[cfg(not(test))]
        let completion = c.submit()?;
        let mut inputs = inputs;
        inputs.push(one.buffer);
        inputs.extend(media.into_iter().map(|m| m.data));
        Ok(PendingEmbedding {
            completion,
            identity: self.identity.clone(),
            output,
            count: seqs.len(),
            dimensions,
            _inputs: inputs,
            _scratch: sc,
            #[cfg(test)]
            trace,
        })
    }
}
