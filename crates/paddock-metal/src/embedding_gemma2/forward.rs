use super::*;
use crate::device::Commands;

impl EmbeddingGemma2 {
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
        self.linear(&c, &self.ple, &sc.x, &sc.ple, rows);
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
            let hd = if i % 6 == 5 { 512 } else { 256 };
            let kh = 512 / hd;
            self.norm(&c, &sc.x, &l.pre, &sc.norm, rows, 1., None);
            for (w, y) in [(&l.q, &sc.q), (&l.k, &sc.k), (&l.v, &sc.v)] {
                self.linear(&c, w, &sc.norm, y, rows);
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
                    &[x, &w.buffer, meta, if self.mlx { y } else { x }],
                    &[
                        heads as u32,
                        hd as u32,
                        w.ty,
                        self.mlx as u32,
                        rope.to_bits(),
                        self.mlx as u32,
                    ],
                    [heads, rows, 1],
                    hd / 4,
                );
            }
            if self.mlx && hd == 512 {
                let mut start = 0;
                for seq in seqs {
                    for offset in (0..seq.len()).step_by(128) {
                        let count = (seq.len() - offset).min(128);
                        let p = [start as u32, seq.len() as u32, offset as u32, count as u32];
                        c.dispatch(
                            "eg2_global_qk",
                            &[&sc.gate, &sc.norm, &sc.scores],
                            &p,
                            [seq.len().div_ceil(64), count.div_ceil(32), 4],
                            128,
                        );
                        c.dispatch("eg2_global_softmax", &[&sc.scores], &p, [count, 4, 1], 256);
                        c.dispatch(
                            "eg2_global_pv",
                            &[&sc.scores, &sc.delta, &sc.attn],
                            &p,
                            [8, count.div_ceil(32), 4],
                            128,
                        );
                    }
                    start += seq.len();
                }
            } else {
                c.dispatch(
                    if self.mlx {
                        "eg2_mlx_attention256"
                    } else if hd == 512 {
                        "eg2_attention512"
                    } else {
                        "eg2_attention256"
                    },
                    &[
                        if self.mlx { &sc.gate } else { &sc.q },
                        if self.mlx { &sc.norm } else { &sc.k },
                        if self.mlx { &sc.delta } else { &sc.v },
                        meta,
                        ends,
                        &sc.attn,
                        tiles,
                    ],
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
            self.linear(&c, &l.o, &sc.attn, &sc.delta, rows);
            self.norm(&c, &sc.delta, &l.post, &sc.x, rows, 1., Some((&sc.x, &one)));
            self.norm(&c, &sc.x, &l.ff_pre, &sc.norm, rows, 1., None);
            self.linear(&c, &l.gate, &sc.norm, &sc.gate, rows);
            self.linear(&c, &l.up, &sc.norm, &sc.up, rows);
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
            self.linear(&c, &l.down, &sc.gate, &sc.delta, rows);
            self.norm(
                &c,
                &sc.delta,
                &l.ff_post,
                &sc.x,
                rows,
                1.,
                Some((&sc.x, &one)),
            );
            self.linear(&c, &l.ple_gate, &sc.x, &sc.norm, rows);
            c.dispatch(
                "eg2_ple_gate",
                &[&sc.norm, &sc.ple],
                &[rows as u32, i as u32, self.mlx as u32],
                [(rows * WIDTH).div_ceil(256), 1, 1],
                256,
            );
            self.linear(&c, &l.ple_out, &sc.norm, &sc.delta, rows);
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
            if let Some(t) = &trace {
                c.dispatch(
                    "spec_copy",
                    &[&sc.x, t],
                    &[0, ((i + 1) * rows * WIDTH) as u32, (rows * WIDTH) as u32],
                    [(rows * WIDTH).div_ceil(256), 1, 1],
                    256,
                );
            }
        }
        self.norm(&c, &sc.x, &self.norm, &sc.norm, rows, 1., None);
        self.linear(&c, &self.output, &sc.norm, &sc.token_out, rows);
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
