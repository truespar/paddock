use super::{
    projection::{experts, norm, project},
    *,
};

impl Kolibri {
    pub(super) fn execute(
        &mut self,
        rows: &[(usize, u32, u32)],
        output_rows: &[usize],
    ) -> Result<Vec<f32>> {
        objc2::rc::autoreleasepool(|_| self.execute_inner(rows, output_rows))
    }
    fn execute_inner(
        &mut self,
        rows: &[(usize, u32, u32)],
        output_rows: &[usize],
    ) -> Result<Vec<f32>> {
        let m = rows.len();
        if m == 0
            || m > CHUNK
            || output_rows.len() > self.slots.len()
            || output_rows.iter().any(|&r| r >= m)
        {
            return Err(MetalError::Model(
                "invalid Kolibri execution/output rows".into(),
            ));
        }
        let mut lengths = self
            .slots
            .iter()
            .map(|s| s.history.len())
            .collect::<Vec<_>>();
        for &(slot, token, pos) in rows {
            if slot >= self.slots.len()
                || token as usize >= VOCAB
                || pos as usize >= self.context
                || pos as usize != lengths[slot]
            {
                return Err(MetalError::Model(format!(
                    "invalid Kolibri row slot={slot} token={token} pos={pos}"
                )));
            }
            lengths[slot] += 1;
        }
        for &(slot, _, pos) in rows {
            while self.slots[slot]
                .table
                .ensure(pos as usize, &mut self.pool)
                .is_err()
            {
                if self.radix.evict_lru(&mut self.pool).is_none() {
                    return Err(MetalError::Memory("Kolibri KV pool exhausted".into()));
                }
            }
        }
        let mut pages = vec![0; self.slots.len() * self.page_stride];
        for (i, s) in self.slots.iter().enumerate() {
            pages[i * self.page_stride..i * self.page_stride + s.table.blocks().len()]
                .copy_from_slice(s.table.blocks());
        }
        let mut tiles = Vec::new();
        let mut decodes = Vec::new();
        let mut first = 0;
        while first < m {
            let mut end = first + 1;
            while end < m && rows[end].0 == rows[first].0 {
                end += 1;
            }
            if end - first >= 16 {
                for offset in (first..end).step_by(32) {
                    tiles.extend([offset as u32, (end - offset).min(32) as u32]);
                }
            } else {
                decodes.extend((first..end).map(|i| i as u32));
            }
            first = end;
        }
        let s = &self.scratch;
        // SAFETY: the preceding submission is complete, and all indices and
        // allocation envelopes were validated before writing the row metadata.
        unsafe {
            s.ids
                .write_u32(&rows.iter().map(|r| r.1).collect::<Vec<_>>());
            s.meta.write_u32(
                &rows
                    .iter()
                    .flat_map(|r| [r.0 as u32, r.2])
                    .collect::<Vec<_>>(),
            );
            s.pages.write_u32(&pages);
            s.output_rows
                .write_u32(&output_rows.iter().map(|&r| r as u32).collect::<Vec<_>>());
            s.attention_tiles.write_u32(&tiles);
            s.decode_rows.write_u32(&decodes);
        }
        #[allow(unused_mut)] // mutable only for test-only diagnostic fences
        let mut cmd = self.device.begin()?;
        // Each snapshot fences and starts a fresh submission. Compiled out of
        // normal builds and disabled for every performance measurement.
        macro_rules! snapshot {
            ($cmd:ident, $label:expr, $buffer:expr, $width:expr) => {
                #[cfg(test)]
                {
                    $cmd = super::diagnostic::snapshot(
                        $cmd,
                        &self.device,
                        &$label,
                        $buffer,
                        (m - 1) * $width,
                        $width,
                    )?;
                }
            };
        }
        cmd.dispatch(
            "q4a_gather",
            &[&self.embedding.buffer, &s.ids, &s.x],
            &[WIDTH as u32, VOCAB as u32, m as u32, 8, 64],
            [(m * WIDTH).div_ceil(256), 1, 1],
            256,
        );
        snapshot!(cmd, "embedding", &s.x, WIDTH);
        let residual = |cmd: &Commands<'_>| {
            cmd.dispatch(
                "mlx_residual",
                &[&s.x, &s.normalized],
                &[(m * WIDTH) as u32],
                [(m * WIDTH).div_ceil(256), 1, 1],
                256,
            )
        };
        for (i, l) in self.layers.iter().enumerate() {
            norm(&cmd, &s.x, &l.norm, &s.norm, WIDTH, m);
            snapshot!(cmd, format!("{i:02}-norm"), &s.norm, WIDTH);
            for (w, out) in [(&l.q, &s.q), (&l.k, &s.k), (&l.v, &s.v)] {
                project(&cmd, w, &s.norm, out, m);
            }
            snapshot!(cmd, format!("{i:02}-q-proj"), &s.q, HEADS * HEAD_DIM);
            // RMSNorm rounds before applying the BF16 norm weight. Full
            // layers get the same QK norm, but absolutely NO RoPE.
            for (heads, input, w) in [(HEADS, &s.q, &l.qnorm), (KV_HEADS, &s.k, &l.knorm)] {
                norm(&cmd, input, w, input, HEAD_DIM, m * heads);
                #[cfg(test)]
                if heads == HEADS {
                    snapshot!(cmd, format!("{i:02}-q-norm"), &s.q, HEADS * HEAD_DIM);
                }
                if KolibriConfig::sliding(i) {
                    cmd.dispatch(
                        "kolibri_rope",
                        &[input, &s.meta],
                        &[heads as u32],
                        [heads, m, 1],
                        32,
                    );
                }
            }
            cmd.dispatch(
                "kolibri_store",
                &[&s.k, &s.v, &l.keys, &l.values, &s.meta, &s.pages],
                &[m as u32, self.page_stride as u32],
                [(m * KVWIDTH).div_ceil(256), 1, 1],
                256,
            );
            snapshot!(cmd, format!("{i:02}-q-rope"), &s.q, HEADS * HEAD_DIM);
            let ap = [
                HEADS as u32,
                KV_HEADS as u32,
                self.page_stride as u32,
                if KolibriConfig::sliding(i) {
                    WINDOW as u32
                } else {
                    0
                },
                0,
                SPLITS as u32,
            ];
            if !tiles.is_empty() {
                cmd.dispatch(
                    "kolibri_prefill",
                    &[
                        &s.q,
                        &l.keys,
                        &l.values,
                        &s.meta,
                        &s.pages,
                        &s.attn,
                        &s.attention_tiles,
                    ],
                    &ap,
                    [HEADS, tiles.len() / 2, 1],
                    128,
                );
            }
            if !decodes.is_empty() {
                cmd.dispatch(
                    "kolibri_decode",
                    &[
                        &s.q,
                        &l.keys,
                        &l.values,
                        &s.meta,
                        &s.pages,
                        &s.decode_rows,
                        &s.parts,
                    ],
                    &ap,
                    [KV_HEADS, decodes.len(), SPLITS],
                    128,
                );
                cmd.dispatch(
                    "gemma_merge",
                    &[&s.parts, &s.attn, &s.decode_rows],
                    &[HEADS as u32, SPLITS as u32, HEAD_DIM as u32],
                    [decodes.len() * HEADS, 1, 1],
                    32,
                );
            }
            cmd.dispatch(
                "kolibri_round",
                &[&s.attn],
                &[(m * HEADS * HEAD_DIM) as u32],
                [(m * HEADS * HEAD_DIM).div_ceil(256), 1, 1],
                256,
            );
            snapshot!(cmd, format!("{i:02}-attn"), &s.attn, HEADS * HEAD_DIM);
            project(&cmd, &l.o, &s.attn, &s.delta, m);
            snapshot!(cmd, format!("{i:02}-o-proj"), &s.delta, WIDTH);
            norm(&cmd, &s.delta, &l.post_attn, &s.normalized, WIDTH, m);
            residual(&cmd);
            snapshot!(cmd, format!("{i:02}-attn-residual"), &s.x, WIDTH);
            norm(&cmd, &s.x, &l.pre_ffn, &s.norm, WIDTH, m);
            snapshot!(cmd, format!("{i:02}-ffn-norm"), &s.norm, WIDTH);
            project(&cmd, &l.shared_gate, &s.norm, &s.fg, m);
            project(&cmd, &l.shared_up, &s.norm, &s.fu, m);
            cmd.dispatch(
                "mlx_swiglu",
                &[&s.fg, &s.fu],
                &[(m * FF) as u32],
                [(m * FF).div_ceil(256), 1, 1],
                256,
            );
            project(&cmd, &l.shared_down, &s.fg, &s.delta, m);
            snapshot!(cmd, format!("{i:02}-shared"), &s.delta, WIDTH);
            experts(&cmd, l, s, m);
            snapshot!(cmd, format!("{i:02}-router"), &s.router, EXPERTS);
            snapshot!(cmd, format!("{i:02}-moe"), &s.delta, WIDTH);
            norm(&cmd, &s.delta, &l.post_ffn, &s.normalized, WIDTH, m);
            residual(&cmd);
            snapshot!(cmd, format!("{i:02}-out"), &s.x, WIDTH);
        }
        if !output_rows.is_empty() {
            cmd.dispatch(
                "mlx_rms_selected",
                &[&s.x, &self.output_norm, &s.output_rows, &s.norm],
                &[WIDTH as u32, 0, 1e-6f32.to_bits()],
                [output_rows.len(), 1, 1],
                WIDTH.div_ceil(128) * 32,
            );
            project(&cmd, &self.head, &s.norm, &s.logits, output_rows.len());
        }
        self.last_gpu_seconds = cmd.finish()?;
        for &(slot, token, _) in rows {
            self.slots[slot].history.push(token);
        }
        // SAFETY: command completion precedes the only inference readback.
        Ok(unsafe { s.logits.read_f32(0, output_rows.len() * VOCAB) })
    }
}
