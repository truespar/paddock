use super::*;

#[cfg(test)]
thread_local! {
    pub(super) static CACHE_ONLY_TAIL_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
    pub(super) static PLE_LOOKAHEAD_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    pub(super) static ADAPTIVE_PLE_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}

fn trace_work() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_METAL_WORK_TRACE").is_some())
}

impl FlashNext {
    pub(super) fn execute(
        &mut self,
        rows: &[(usize, u32, u32)],
        outputs: &[usize],
    ) -> Result<Vec<f32>> {
        self.execute_contracts(rows, outputs, None)
    }
    pub(super) fn execute_contracts(
        &mut self,
        rows: &[(usize, u32, u32)],
        outputs: &[usize],
        contracts: Option<&[usize]>,
    ) -> Result<Vec<f32>> {
        objc2::rc::autoreleasepool(|_| self.execute_inner(rows, outputs, contracts))
    }
    fn execute_inner(
        &mut self,
        rows: &[(usize, u32, u32)],
        outputs: &[usize],
        contracts: Option<&[usize]>,
    ) -> Result<Vec<f32>> {
        self.healthy()?;
        let n = rows.len();
        if n == 0
            || n > self.capacity
            || outputs.len() > self.slots.len()
            || outputs.iter().any(|&i| i >= n)
        {
            return Err(MetalError::Model(
                "invalid Flash Next execution/output rows".into(),
            ));
        }
        // Opt-in prefill diagnostics only. Never log token IDs, prompt text,
        // weights or activations; no per-decode logging or extra GPU barrier.
        let trace_started = (n > self.slots.len() && trace_work()).then(std::time::Instant::now);
        let lengths = self.slots.iter().map(|s| s.length).collect::<Vec<_>>();
        let positions = rows.iter().map(|r| (r.0, r.2 as usize)).collect::<Vec<_>>();
        // All three plans are immutable control data, built once per batch,
        // never once per layer. Validate the entire batch before GPU mutation.
        let pp = ple::Plan::new(
            &rows
                .iter()
                .map(|r| (r.0, r.2 as usize, r.1))
                .collect::<Vec<_>>(),
            &lengths,
            self.capacity,
            self.context,
        )?;
        let dp = deltanet::Plan::new(&positions, &lengths, self.capacity, self.context)?;
        let qp = qsa::Plan::new(&positions, &lengths, self.capacity, self.context)?;
        let mut projection_rows = Vec::new();
        if self.is_mlx() {
            for (i, row) in rows.iter().enumerate() {
                if i == 0 || rows[i - 1].0 != row.0 {
                    projection_rows.push((i, 1, 1));
                } else {
                    projection_rows
                        .last_mut()
                        .expect("the first row started a projection span")
                        .1 += 1;
                }
            }
            if contracts.is_some_and(|v| v.len() != projection_rows.len()) {
                return Err(MetalError::Model(
                    "invalid Flash Next logical chunk count".into(),
                ));
            }
            for (i, span) in projection_rows.iter_mut().enumerate() {
                span.2 = contracts.map_or(span.1, |v| v[i]);
                if span.2 < span.1 || span.2 > self.chunk {
                    return Err(MetalError::Model(
                        "invalid Flash Next logical chunk size".into(),
                    ));
                }
            }
        }
        // Fenced compressed-row reads precede GPU mutation. Failed I/O is
        // not a cache miss or a reason to consume stale staging bytes.
        let planned = trace_started.map(|t| t.elapsed());
        #[cfg(test)]
        let ple_started = std::time::Instant::now();
        self.ple_table.stage(&pp, &self.scratch.ple)?;
        let staged = trace_started.map(|t| t.elapsed());
        #[cfg(test)]
        if n > 8 && std::env::var_os("PADDOCK_FLASH_NEXT_PLE_TIMING").is_some() {
            eprintln!(
                "FLASH_PLE_STAGE rows={n} position={} host_ms={:.3}",
                rows[0].2,
                ple_started.elapsed().as_secs_f64() * 1000.0
            );
        }
        let mut next = lengths.clone();
        for r in rows {
            next[r.0] += 1;
        }
        let needed = self
            .slots
            .iter()
            .zip(&next)
            .map(|(s, &len)| {
                len.div_ceil(BLOCK_TOKENS)
                    .saturating_sub(s.table.blocks().len())
            })
            .sum::<usize>();
        if !self.reclaim_prefix_pages(needed) {
            return Err(MetalError::Memory(
                "Flash Next paged KV exhausted before submission".into(),
            ));
        }
        for (slot, &len) in self.slots.iter_mut().zip(&next) {
            if len > 0 {
                slot.table
                    .ensure(len - 1, &mut self.pool)
                    .map_err(|_| MetalError::Memory("Flash Next KV reservation failed".into()))?;
            }
        }
        let mut pages = vec![0; self.pages * self.slots.len()];
        for (i, slot) in self.slots.iter().enumerate() {
            pages[i * self.pages..i * self.pages + slot.table.blocks().len()]
                .copy_from_slice(slot.table.blocks());
        }
        let s = &self.scratch;
        // Previous whole-model work has completed; only bounded host control
        // buffers are written here. Model math and status accumulation stay GPU.
        unsafe {
            s.ids
                .write_u32(&rows.iter().map(|r| r.1).collect::<Vec<_>>());
            s.output_rows
                .write_u32(&outputs.iter().map(|&r| r as u32).collect::<Vec<_>>());
            s.qsa.pages.write_u32(&pages);
            s.bad.write_u32(&[0]);
        }
        s.dn.stage(&dp);
        s.qsa.stage(&qp);
        s.ple.stage(&pp);
        self.poisoned = true;
        let mut cmd = self.device.begin()?;
        if let Some(workspace) = &self.affine_scratch {
            let independent = projection_rows.iter().all(|r| r.2 == 1);
            cmd = cmd
                .with_projection_workspace(workspace)
                .with_projection_rows(&projection_rows)
                .with_independent_rows(independent);
        }
        macro_rules! trace {
            ($layer:expr, $name:expr, $buffer:expr, $count:expr) => {
                #[cfg(test)]
                if std::env::var("PADDOCK_FLASH_NEXT_MLX_TRACE_POSITION")
                    .ok()
                    .and_then(|v| v.parse::<u32>().ok())
                    == Some(rows[0].2)
                {
                    cmd = cmd.trace_row(
                        &format!("rows={n} layer={} stage={}", $layer, $name),
                        $buffer,
                        $count,
                    )?;
                }
            };
        }
        for (slot, (&before, &after)) in lengths.iter().zip(&next).enumerate() {
            if before == 0 && after > 0 {
                s.ple.encode_reset(&cmd, slot);
                for layer in &self.layers {
                    match &layer.mixer {
                        Mixer::Delta(_, cache) => cache.reset(&cmd, slot),
                        Mixer::Qsa(_, cache) => cache.reset(&cmd, slot, self.pages),
                    }
                }
            }
        }
        if self.is_mlx() {
            super::super::affine::gather(&cmd, &self.embedding, &s.ids, &s.x, n);
        } else {
            cmd.dispatch(
                "embed",
                &[&self.embedding.buffer, &s.ids, &s.x],
                &[WIDTH as u32, n as u32, 14, 1f32.to_bits()],
                [(n * WIDTH).div_ceil(256), 1, 1],
                256,
            );
        }
        cmd.dispatch(
            "q4x_hc_init",
            &[&s.x, &s.h],
            &[WIDTH as u32, n as u32],
            [(n * WIDE).div_ceil(256), 1, 1],
            256,
        );
        for (li, layer) in self.layers.iter().enumerate() {
            // GGUF blk.1 is the PLE site. It adds to all four residual streams
            // before the attention HC mixer, not after attention or the FFN.
            if li == 1 {
                self.ple_table
                    .encode(&s.ple, &cmd, &self.ple_weights, &pp, &s.h);
            }
            layer.hc.encode(&cmd, &s.h, &s.hc, n);
            trace!(li, "hc_norm", &s.hc.norm, WIDE);
            trace!(li, "hc_low", &s.hc.low, 320);
            trace!(li, "hc_gate", &s.hc.gate, WIDE);
            trace!(li, "hc_mix", &s.hc.mixed, WIDTH);
            // execute's only consumers are the requested logits and carried
            // caches; no draft head consumes s.h. Do not select/compact rows
            // for mixed calls: their original projection contracts stay intact.
            let cache_only = self.is_mlx() && outputs.is_empty() && li + 1 == self.layers.len();
            #[cfg(test)]
            let cache_only = cache_only && CACHE_ONLY_TAIL_FOR_TEST.with(|v| v.get());
            if cache_only && let Mixer::Qsa(w, cache) = &layer.mixer {
                s.qsa.encode_cache(&cmd, w, cache, &qp, &s.hc.mixed, &s.bad);
                continue;
            }
            match &layer.mixer {
                Mixer::Delta(w, cache) => s.dn.encode(&cmd, w, cache, &dp, &s.hc.mixed, &s.delta),
                Mixer::Qsa(w, cache) => {
                    s.qsa.encode(&cmd, w, cache, &qp, &s.hc.mixed, &s.delta);
                    cmd.dispatch(
                        "q4x_status",
                        &[&s.qsa.counts, &s.bad],
                        &[n as u32, 512, 2],
                        [n.div_ceil(256), 1, 1],
                        256,
                    );
                }
            }
            trace!(li, "mixer", &s.delta, WIDTH);
            let normalized = layer
                .ffn
                .prepare_after_mixer(&cmd, &s.h, &s.delta, &s.hc, n);
            if !normalized {
                layer.hc.combine(&cmd, &s.h, &s.delta, &s.hc, n);
            }
            trace!(li, "attention_out", &s.h, WIDE);
            layer
                .ffn
                .encode_ffn(&cmd, &s.h, &s.hc, &s.moe, n, normalized)?;
            trace!(li, "ffn_out", &s.h, WIDE);
            // Scratch flags are overwritten by the next layer; accumulate
            // them now, rather than inspecting only layer 47 on the host.
            cmd.dispatch(
                "q4x_status",
                &[&s.moe.invalid, &s.bad],
                &[n as u32, 0, 4],
                [n.div_ceil(256), 1, 1],
                256,
            );
            cmd.dispatch(
                "vis_finite",
                &[&s.h, &s.bad],
                &[(n * WIDE) as u32],
                [(n * WIDE).div_ceil(256), 1, 1],
                256,
            );
        }
        if !outputs.is_empty() {
            if self.is_mlx() {
                // Selected rows are independent sequence completions, even
                // when the preceding body was a ragged prefill matrix.
                cmd = cmd.with_independent_rows(true);
            }
            let m = outputs.len();
            cmd.dispatch(
                "q4x_select_rows",
                &[&s.h, &s.output_rows, &s.selected_h],
                &[WIDE as u32, m as u32],
                [(m * WIDE).div_ceil(256), 1, 1],
                256,
            );
            self.output_hc.encode(&cmd, &s.selected_h, &s.hc, m);
            if self.is_mlx() {
                super::super::affine::project(&cmd, &self.head, &s.hc.mixed, &s.logits, m);
            } else {
                crate::projection::project(&cmd, &[(&self.head, &s.logits)], &s.hc.mixed, m);
            }
            cmd.dispatch(
                "vis_finite",
                &[&s.logits, &s.bad],
                &[(m * VOCAB) as u32],
                [(m * VOCAB).div_ceil(256), 1, 1],
                256,
            );
        }
        let submitted = trace_started.map(|t| t.elapsed());
        // Only measured slow storage justifies competing with the GPU for
        // memory bandwidth. The read-ahead does not change admission, token
        // history, arithmetic or the live staging buffer.
        let lookahead =
            matches!(&self.ple_table, ple::Table::Paged(table) if table.should_prefetch());
        #[cfg(test)]
        let lookahead = (lookahead && ADAPTIVE_PLE_FOR_TEST.with(|v| v.get()))
            || PLE_LOOKAHEAD_FOR_TEST.with(|v| v.get());
        if lookahead && let ple::Table::Paged(table) = &mut self.ple_table {
            self.last_gpu_seconds = cmd.finish_with_host_work(|gpu_done| {
                let mut ids = Vec::new();
                for pending in &self.pending {
                    let start = next[pending.slot];
                    if start <= pending.offset || start >= pending.tokens.len() {
                        continue;
                    }
                    let count = (start - pending.offset)
                        .min(pending.tokens.len() - start)
                        .min(super::super::ple_paged::LOOKAHEAD_ROWS - ids.len() / 16);
                    if count == 0 {
                        break;
                    }
                    match super::super::ple_paged::prompt_lookahead(&pending.tokens, start, count) {
                        Ok(rows) => ids.extend(rows),
                        Err(_) => return, // demand staging retains authoritative validation
                    }
                }
                // Speculative I/O failure never fails a completed request or
                // admits partial bytes. A needed row is read/validated again
                // by stage(), which propagates its actual error to the caller.
                let _ = table.prefetch(&ids, gpu_done);
            })?;
        } else {
            self.last_gpu_seconds = cmd.finish()?;
        }
        let finished = trace_started.map(|t| t.elapsed());
        let status = unsafe { s.bad.read_u32(1)[0] };
        if status != 0 {
            return Err(MetalError::Device(format!(
                "Flash Next invalid GPU walk (status {status}); state poisoned, reload required"
            )));
        }
        let logits = unsafe { s.logits.read_f32(0, outputs.len() * VOCAB) };
        for (slot, len) in self.slots.iter_mut().zip(next) {
            slot.length = len;
        }
        self.poisoned = false;
        if let (Some(started), Some(planned), Some(staged), Some(submitted), Some(finished)) =
            (trace_started, planned, staged, submitted, finished)
        {
            tracing::info!(
                rows = n,
                output_rows = outputs.len(),
                min_position = rows.iter().map(|r| r.2).min().unwrap_or(0),
                max_position = rows.iter().map(|r| r.2).max().unwrap_or(0),
                spans = ?projection_rows,
                planning_ms = planned.as_secs_f64() * 1000.,
                ple_ms = (staged - planned).as_secs_f64() * 1000.,
                encode_ms = (submitted - staged).as_secs_f64() * 1000.,
                submit_wait_ms = (finished - submitted).as_secs_f64() * 1000.,
                gpu_ms = self.last_gpu_seconds * 1000.,
                readout_ms = (started.elapsed() - finished).as_secs_f64() * 1000.,
                "Flash Next prefill execution"
            );
        }
        Ok(logits)
    }
}
