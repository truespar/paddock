use super::*;

fn trace_work() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_METAL_WORK_TRACE").is_some())
}

#[derive(Default)]
pub(super) struct AdmissionCost {
    quantum_seconds: Option<f64>,
}

impl AdmissionCost {
    fn observe(&mut self, seconds: f64) {
        if seconds.is_finite() && seconds > 0. {
            self.quantum_seconds = Some(
                self.quantum_seconds
                    .map_or(seconds, |old| old * 0.75 + seconds * 0.25),
            );
        }
    }

    fn grace(&self) -> std::time::Duration {
        // Budget 1/64 of the measured first 32-row wave on collecting
        // an idle burst. Clamp the prior and observations to a hard 0.2-2 ms
        // range; this is a bounded policy, not a promise of a GPU deadline.
        std::time::Duration::from_secs_f64(
            (self.quantum_seconds.unwrap_or(0.064) / 64.).clamp(0.0002, 0.002),
        )
    }
}

#[cfg(test)]
thread_local! {
    pub(super) static SCHEDULE_PROBE: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

// Do useful work immediately, but let an idle wide server revisit admission
// after one efficient MPP tile. Finishing the first short prompt in a long GPU
// pass before its peers arrive creates a decode head start, then recurrent
// mixed-prefill stalls at every turnover. This is a first quantum, not a timer,
// hidden queue or completed-logit holdback. Single-slot and resumed requests
// retain the full grant; once a cohort fills (or makes progress), so does it.
fn admission_cap(cap: usize, decodes: usize, slots: usize, pending: usize, fresh: bool) -> usize {
    if decodes == 0 && slots > 1 && pending > 0 && pending < slots && fresh {
        cap.min(32)
    } else {
        cap
    }
}

fn verify_fits(context: usize, reqs: &[(usize, usize, Vec<u32>)]) -> bool {
    reqs.iter().all(|(_, pos, chunk)| {
        pos.checked_add(chunk.len())
            .is_some_and(|end| end <= context)
    })
}

fn image_rider_cap(cap: usize, decodes: usize, image_pending: bool, fresh_image: bool) -> usize {
    if decodes != 0 && image_pending {
        // Amortize the model walk across a 128-row mixed tile. With setup
        // and merger yields removed, rotating M5 runs measured ~2.20 s image
        // TTFT and ~181 ms worst text gaps. The 96-row grant's kernel remains
        // useful for ragged 65..96-row ends.
        // Encoder work shares a soft 192 ms envelope (see encoder_quantum).
        // This intentionally trades the earlier ~110 ms tail for lower
        // image TTFT; other captures exceeded 210 ms, so this is neither a
        // hard deadline nor a hardware ceiling.
        // Cold image cohorts retain their full 512-row throughput grant.
        // Encoder completion and the first language batch share a tick.
        // Make immediate progress with the small MPP rung, then restore the
        // normal grant; never add another full prefill to the encoder quantum.
        let rung = if fresh_image { 8 } else { 128 };
        cap.min((decodes + 1).next_multiple_of(rung))
    } else {
        cap
    }
}

impl Generator for Qwen35 {
    fn tier_pump(&mut self) {
        self.pump_cold();
    }
    fn tier_prefix_loading(&mut self, slot: usize, tokens: &[u32]) -> bool {
        if let Some(slot) = self.slots.get_mut(slot) {
            slot.cold_consulted = true;
        }
        self.cold_loading(tokens)
    }
    fn tier_stats(&self) -> Option<paddock_engine::kv_tier::TierStats> {
        self.cold_stats()
    }
    fn tier_report(&self) -> Option<paddock_engine::kv_tier::TierReport> {
        self.cold_report()
    }
    fn tier_observe_prefill(&mut self, tokens: u32, wall_us: f64) {
        self.observe_cold_prefill(tokens, wall_us);
    }
    fn reset(&mut self) {
        if let Some(v) = &mut self.spec {
            v.round = None;
        }
        self.pending.clear();
        self.encoding.clear();
        for slot in &mut self.slots {
            slot.table.clear(&mut self.pool);
            slot.history.clear();
            slot.prefill_end = 0;
            slot.reused = 0;
            slot.cold_consulted = false;
            slot.mm = None;
        }
    }
    fn vocab(&self) -> usize {
        self.vocab
    }
    fn vision_budget(&self) -> Option<paddock_engine::generator::VisionBudget> {
        self.vision.as_ref().map(|v| v.budget)
    }
    fn supports_mm_slots(&self) -> bool {
        self.vision.is_some()
    }
    fn supports_chunked_multimodal(&self) -> bool {
        self.vision.is_some()
    }
    fn forward_multimodal(
        &mut self,
        chunks: &[paddock_engine::service::MmChunk],
    ) -> std::result::Result<Option<(Vec<f32>, usize)>, GenError> {
        Ok(Some(self.prefill_images(0, chunks)?))
    }
    fn forward_prefill_multimodal(
        &mut self,
        slot: usize,
        chunks: &[paddock_engine::service::MmChunk],
    ) -> std::result::Result<(Vec<f32>, usize), GenError> {
        Ok(self.prefill_images(slot, chunks)?)
    }
    fn prefill_begin_multimodal(
        &mut self,
        items: Vec<(usize, Vec<paddock_engine::service::MmChunk>)>,
    ) -> Vec<(usize, paddock_engine::generator::MmAdmit)> {
        self.admit_images(items)
    }
    fn encode_step(&mut self) -> Vec<(usize, paddock_engine::generator::MmAdmit)> {
        self.step_images()
    }
    fn encoding_pending(&self) -> bool {
        !self.encoding.is_empty()
    }
    fn spec_capable(&self) -> bool {
        self.mtp.is_some() || self.dflash.is_some()
    }
    fn spec_draft_kv_space(&self) -> bool {
        true
    }
    fn spec_block_width(&self) -> Option<usize> {
        self.dflash.as_ref().map(|_| spec::BLOCK)
    }
    fn spec_k_miss_floor(&self) -> Option<usize> {
        self.dflash.as_ref().map(|_| spec::BLOCK - 1)
    }
    fn spec_batch_draft_budget(&self, live: usize) -> Option<usize> {
        if self.splash && self.dflash.is_some() {
            return Some(if live <= 4 { spec::BLOCK - 1 } else { 0 });
        }
        // The M5 dense-27B packed loader makes three neural proposals useful
        // at c=1; keep that source's cap separate from the copy budget below.
        // Strong copy matches can fill a wider
        // verifier without running the neural drafter; their own measured
        // goodput gates re-election. The neural model keeps eight trained rows.
        // Concurrent MLX verification currently loses useful throughput and
        // raises streaming gaps. Keep ordinary batching there, with the draft
        // conditioning current so the next single-stream round can resume.
        (self.mlx && self.dflash.is_some()).then_some(if live == 1 {
            if self.geometry == Geometry::DENSE_27B {
                if self.device.tensor_accelerated() {
                    lookup::MAX_DRAFT
                } else {
                    2
                }
            } else {
                1
            }
        } else {
            0
        })
    }
    fn spec_live_cap(&self) -> usize {
        self.slots
            .len()
            .min(if self.mlx { 4 } else { CHUNK / spec::BLOCK })
    }
    fn spec_ensure_warm(
        &mut self,
        slot: usize,
        _committed: &[u32],
        want_pos: u32,
    ) -> std::result::Result<bool, GenError> {
        Ok(self.spec_capable()
            && !self.pending.iter().any(|p| p.slot == slot)
            && self
                .slots
                .get(slot)
                .is_some_and(|s| s.history.len() == want_pos as usize + 1))
    }
    fn spec_draft_batch(
        &mut self,
        pendings: &[(usize, u32)],
        k: usize,
    ) -> std::result::Result<Option<Vec<Vec<u32>>>, GenError> {
        if self.dflash.is_some() {
            if self.mlx
                && !self.splash
                && self.geometry == Geometry::DENSE_27B
                && pendings.len() == 1
                && k > 0
            {
                self.require_committed()?;
                let (slot, pending) = pendings[0];
                if slot >= self.slots.len()
                    || pending as usize >= self.vocab
                    || self.slots[slot].history.is_empty()
                    || self.pending.iter().any(|p| p.slot == slot)
                {
                    self.lookup.cancel();
                    return Ok(None);
                }
                let history = &self.slots[slot].history;
                let budget = k.min(self.context.saturating_sub(history.len() + 1));
                let draft = if self.slots[slot].mm.is_none() {
                    self.lookup.propose(history, pending, budget)
                } else {
                    self.lookup.cancel();
                    Vec::new()
                };
                if !draft.is_empty() {
                    self.lookup.begin(true);
                    return Ok(Some(vec![draft]));
                }
                self.lookup.begin(false);
                // Same-checkpoint depth/rollback qualification is M5-only.
                // Older devices retain their existing one-proposal policy.
                let neural_cap = if self.device.tensor_accelerated() {
                    3
                } else {
                    1
                };
                let mut result = self.dflash_draft(pendings, k.min(neural_cap));
                // Its noncausal eight-row training window must not be shrunk;
                // only the proposals handed to target verification are capped.
                if let Ok(Some(drafts)) = &mut result {
                    for draft in drafts {
                        draft.truncate(neural_cap);
                    }
                }
                if !matches!(&result, Ok(Some(d)) if d.iter().any(|r| !r.is_empty())) {
                    self.lookup.cancel();
                }
                return Ok(result?);
            }
            self.lookup.cancel();
            return Ok(self.dflash_draft(pendings, k)?);
        }
        Ok(self.mtp_draft(pendings, k)?)
    }
    fn forward_spec_batch(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
    ) -> std::result::Result<Option<Vec<u32>>, GenError> {
        // The engine may propose n-grams after a context-limited model
        // drafter declines. Leave state untouched and let its dense fallback
        // consume the remaining window, rather than failing a valid request.
        if !self.spec_capable() || !verify_fits(self.context, reqs) {
            return Ok(None);
        }
        if self.mlx && !self.splash && !reqs.is_empty() && reqs.iter().all(|r| r.2.len() == 1) {
            return Ok(Some(self.decode_picks(reqs)?));
        }
        let picks = self.verify_picks(reqs)?;
        let mut base = 0;
        let mut counts = Vec::new();
        for (_, _, chunk) in reqs {
            let n = 1 + chunk[1..]
                .iter()
                .zip(&picks[base..])
                .take_while(|(a, b)| a == b)
                .count();
            counts.push(n as u32);
            base += chunk.len();
        }
        self.commit_verify(&counts)?;
        Ok(Some(picks))
    }
    fn forward_spec_verify(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
    ) -> std::result::Result<Option<Vec<f32>>, GenError> {
        // Exact sample-and-match for deterministic proposals. Full DFlash2
        // stochastic selector q/p + residual sampling is a remaining SOTA
        // performance gap; do not treat draft argmaxes as target samples.
        if !self.spec_capable() || !verify_fits(self.context, reqs) {
            return Ok(None);
        }
        Ok(Some(self.verify(reqs)?))
    }
    fn spec_commit(&mut self, committed: &[u32]) -> std::result::Result<(), GenError> {
        Ok(self.commit_verify(committed)?)
    }
    fn max_context(&self) -> usize {
        self.context
    }
    fn enable_batch(&mut self, max: usize) -> std::result::Result<usize, GenError> {
        Ok(max.min(self.slots.len()))
    }
    fn weights_mem_bytes(&self) -> Option<u64> {
        Some(self.weight_bytes)
    }
    fn kv_mem_bytes(&self) -> Option<u64> {
        Some(self.kv_bytes)
    }
    fn device_mem_used(&self) -> Option<u64> {
        Some(self.device.allocated_bytes())
    }
    fn forward(&mut self, token: u32) -> std::result::Result<Vec<f32>, GenError> {
        Ok(self.execute(&[(0, token, self.slots[0].history.len() as u32)], &[0])?)
    }
    fn forward_prefill_stream(
        &mut self,
        tokens: &[u32],
    ) -> std::result::Result<Vec<f32>, GenError> {
        Ok(self.prefill(0, tokens)?)
    }
    fn forward_prefill(
        &mut self,
        slot: usize,
        tokens: &[u32],
    ) -> std::result::Result<Vec<f32>, GenError> {
        Ok(self.prefill(slot, tokens)?)
    }
    fn forward_batch(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
    ) -> std::result::Result<Vec<f32>, GenError> {
        if tokens.len() != positions.len() {
            return Err(GenError::Backend("batch shape mismatch".into()));
        }
        // The scheduler uses position zero for holes; real decode rows follow
        // a nonempty prefill. Preserve its row indices without writing hole KV.
        let rows: Vec<_> = tokens
            .iter()
            .zip(positions)
            .enumerate()
            .filter(|(_, (_, p))| **p != 0)
            .map(|(i, (&t, &p))| (i, t, p))
            .collect();
        let mut out = vec![0.0; tokens.len() * self.vocab];
        if !rows.is_empty() {
            let logits = self.execute(&rows, &(0..rows.len()).collect::<Vec<_>>())?;
            for (i, row) in rows.iter().enumerate() {
                out[row.0 * self.vocab..(row.0 + 1) * self.vocab]
                    .copy_from_slice(&logits[i * self.vocab..(i + 1) * self.vocab]);
            }
        }
        Ok(out)
    }
    fn take_prefill_reused(&mut self, slot: usize) -> usize {
        std::mem::take(&mut self.slots[slot].reused)
    }
    fn pool_free_blocks(&self) -> Option<usize> {
        Some(self.pool.free_blocks())
    }
    fn release_inactive_slots(&mut self, occupied: &[bool]) {
        for i in 0..self.slots.len() {
            if !occupied.get(i).copied().unwrap_or(false)
                && !self.pending.iter().any(|p| p.slot == i)
            {
                // Exact-boundary checkpoints were captured during prefill.
                self.slots[i].table.clear(&mut self.pool);
                self.slots[i].history.clear();
                self.slots[i].prefill_end = 0;
                self.slots[i].cold_consulted = false;
                self.slots[i].mm = None;
            }
        }
    }
    fn supports_chunked_prefill(&self) -> bool {
        true
    }
    // the scheduler's tick pacer reads the FIFO queue from each offset
    fn prefill_queue(&self) -> Vec<(usize, usize, usize)> {
        self.pending
            .iter()
            .map(|p| (p.slot, p.offset, p.tokens.len() - p.offset))
            .collect()
    }
    // the mixed grant with riders aboard (the 2048-row wave is riderless)
    fn prefill_tick_cap(&self, decode_rows: usize) -> usize {
        crate::schedule::row_cap(decode_rows, CHUNK).saturating_sub(decode_rows)
    }
    fn idle_admission_grace(&self, prompt: &[u32]) -> std::time::Duration {
        if self.stable_affine_contract() && self.slots.len() > 1 && prompt.len() > 32
            // No new hold for resident hits or persistent-cache instances;
            // qualification here covers cold text without tier I/O.
            && self.cold.is_none()
            && !self.cache.iter().any(|c| !c.reserved && !c.history.is_empty()
                && c.images.is_empty() && c.history.len() < prompt.len() && prompt.starts_with(&c.history))
        {
            self.admission_cost.grace()
        } else {
            std::time::Duration::ZERO
        }
    }
    fn prefill_begin(
        &mut self,
        slot: usize,
        tokens: Vec<u32>,
    ) -> std::result::Result<(), GenError> {
        if self.pending.iter().any(|p| p.slot == slot) {
            return Err(GenError::Backend("slot already prefilling".into()));
        }
        let reused = self.prepare(slot, &tokens)?;
        self.pending.push_back(Pending {
            slot,
            work: tokens.len() - reused,
            tokens,
            offset: reused,
        });
        Ok(())
    }
    fn prefill_abort(&mut self, slot: usize) -> bool {
        self.abort_images(slot);
        self.pending.retain(|p| p.slot != slot);
        if let Some(s) = self.slots.get_mut(slot) {
            s.table.clear(&mut self.pool);
            s.history.clear();
            s.prefill_end = 0;
            s.mm = None;
        }
        true
    }
    fn forward_mixed(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
    ) -> std::result::Result<(Vec<f32>, Vec<(usize, Vec<f32>, usize)>), GenError> {
        if decodes.is_empty() && self.cold_image_cohort_encoding() {
            // Finish already-admitted, similar-sized encoder peers before
            // starting their joint backbone batch. No timer, new admission
            // hold, or completed logits withheld; encode_step keeps working.
            return Ok((Vec::new(), Vec::new()));
        }
        let mut rows = decodes.to_vec();
        let mut complete = Vec::new();
        // Keep established streams on the bounded mixed grant. The M5 text
        // FIFO/512-row experiment improved TTFT but caused 0.6-0.9 s stream
        // gaps and changed concurrent generations; it is not a serving default.
        // A single-slot packed server has no admitted decode peer to delay.
        // Its M32 projection tiles reuse weights better in a larger cold wave.
        // Existing multi-user and mixed/image admission bounds stay intact;
        // only elect this workspace with conservative transient headroom.
        let capacity = if self.splash
            && self.slots.len() == 1
            && decodes.is_empty()
            && self.pending.iter().all(|p| self.slots[p.slot].mm.is_none())
            && (self.row_capacity >= 2048
                || self
                    .device
                    .budget_bytes()
                    .saturating_sub(self.device.allocated_bytes())
                    >= 4 << 30)
        {
            2048
        } else {
            CHUNK
        };
        let cap = admission_cap(
            image_rider_cap(
                crate::schedule::row_cap(decodes.len(), capacity),
                decodes.len(),
                self.pending.iter().any(|p| self.slots[p.slot].mm.is_some()),
                self.pending.iter().any(|p| {
                    self.slots[p.slot].mm.is_some() && p.tokens.len() - p.offset == p.work
                }),
            ),
            decodes.len(),
            self.slots.len(),
            self.pending.len(),
            self.pending.iter().all(|p| p.offset == 0),
        );
        // Restored checkpoints often leave only a few template/suffix tokens.
        // Keep their mixed tick inside the tuned single-vector projection rung while
        // another request is decoding. A 9-token tail otherwise chooses the
        // prefill kernel family and stalls an established stream for ~110 ms.
        let short_resume = self.cold.is_some()
            && (1..4).contains(&decodes.len())
            && !self.pending.is_empty()
            && self
                .pending
                .iter()
                .all(|p| p.work <= BLOCK_TOKENS && p.offset >= 3 * BLOCK_TOKENS);
        let cap = if short_resume {
            // GGUF has R1-R4 kernels; native MLX also has a qualified compact
            // five-row kernel. Do not apply the GGUF limit to affine weights.
            cap.min(if self.mlx { 5 } else { 4 })
        } else {
            cap
        };
        #[cfg(test)]
        let cap = if self.stable_affine_contract() && !decodes.is_empty() {
            match SCHEDULE_PROBE.with(|v| v.get()) {
                2 => 64,
                3 => 128,
                4 => 256,
                _ => cap,
            }
        } else {
            cap
        };
        let advances = if short_resume {
            crate::schedule::balanced_grants(
                &self
                    .pending
                    .iter()
                    .map(|p| p.tokens.len() - p.offset)
                    .collect::<Vec<_>>(),
                budget.min(cap.saturating_sub(rows.len())),
            )
        } else {
            crate::schedule::grants(
                &self
                    .pending
                    .iter()
                    .map(|p| (p.tokens.len() - p.offset, p.work))
                    .collect::<Vec<_>>(),
                budget.min(cap.saturating_sub(rows.len())),
                decodes.is_empty(),
            )
        };
        #[cfg(test)]
        let advances = if decodes.is_empty() && SCHEDULE_PROBE.with(|v| v.get()) != 0 {
            let probe = SCHEDULE_PROBE.with(|v| v.get());
            let work: Vec<_> = self
                .pending
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    if probe >= 2 && i >= 2 {
                        0
                    } else {
                        p.tokens.len() - p.offset
                    }
                })
                .collect();
            crate::schedule::tiled_grants(&work, budget.min(cap))
        } else {
            advances
        };
        // Image embeddings obey raster-order causality in this backbone.
        // They use the same bounded hybrid batch as text, sharing every
        // projection with decode riders instead of a second model walk.
        self.reserve_rows(rows.len() + advances.iter().sum::<usize>())?;
        for (pending, &n) in self.pending.iter().zip(&advances) {
            for i in pending.offset..pending.offset + n {
                rows.push((pending.slot, pending.tokens[i], i as u32));
            }
            if n > 0 && pending.offset + n == pending.tokens.len() {
                complete.push((pending.slot, rows.len() - 1, pending.tokens.len()));
            }
        }
        if rows.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let output_rows: Vec<_> = (0..decodes.len())
            .chain(complete.iter().map(|&(_, row, _)| row))
            .collect();
        let started = trace_work().then(std::time::Instant::now);
        let logits = self.execute(&rows, &output_rows)?;
        if self.stable_affine_contract()
            && cap == 32
            && decodes.is_empty()
            && rows.len() == 32
            && output_rows.is_empty()
            && self
                .pending
                .iter()
                .all(|p| p.offset == 0 && self.slots[p.slot].mm.is_none())
        {
            self.admission_cost.observe(self.last_gpu_seconds);
        }
        if let Some(started) = started {
            // Counts/positions only: never log prompt text, tokens or logits.
            // GPU duration is the ordinary completed command's timestamp;
            // no per-dispatch profiling or extra synchronization is enabled.
            tracing::info!(
                decodes = decodes.len(),
                rows = rows.len(),
                heads = output_rows.len(),
                cap,
                gpu_ms = self.last_gpu_seconds * 1000.,
                wall_ms = started.elapsed().as_secs_f64() * 1000.,
                pending = ?self.pending.iter().zip(&advances)
                    .map(|(p, &n)| (p.slot, p.offset, p.tokens.len(), n))
                    .collect::<Vec<_>>(),
                "metal-work"
            );
        }
        for (pending, n) in self.pending.iter_mut().zip(advances) {
            pending.offset += n;
        }
        let done: Vec<_> = complete
            .iter()
            .enumerate()
            .map(|(i, &(slot, _, n))| {
                let row = decodes.len() + i;
                (
                    slot,
                    logits[row * self.vocab..(row + 1) * self.vocab].to_vec(),
                    n,
                )
            })
            .collect();
        self.pending.retain(|p| p.offset < p.tokens.len());
        Ok((logits[..decodes.len() * self.vocab].to_vec(), done))
    }
}

#[cfg(test)]
mod tests {
    use super::admission_cap;

    #[test]
    fn admission_cost_is_bounded_and_ignores_invalid_gpu_samples() {
        let mut cost = super::AdmissionCost::default();
        assert_eq!(cost.grace(), std::time::Duration::from_millis(1));
        cost.observe(0.064);
        for invalid in [0., -1., f64::NAN, f64::INFINITY] {
            cost.observe(invalid);
        }
        assert_eq!(cost.grace(), std::time::Duration::from_millis(1));
        cost.observe(0.128);
        assert_eq!(cost.grace(), std::time::Duration::from_micros(1250));
        cost.observe(1000.);
        assert_eq!(cost.grace(), std::time::Duration::from_millis(2));
        let mut fast = super::AdmissionCost::default();
        fast.observe(0.000001);
        assert_eq!(fast.grace(), std::time::Duration::from_micros(200));
    }

    #[test]
    fn image_riders_keep_bounded_work_without_reducing_cold_or_text_grants() {
        use super::image_rider_cap;
        assert_eq!(image_rider_cap(512, 0, true, true), 512);
        assert_eq!(image_rider_cap(128, 3, false, false), 128);
        assert_eq!(image_rider_cap(128, 3, true, false), 128);
        assert_eq!(image_rider_cap(128, 3, true, true), 8);
        assert_eq!(image_rider_cap(160, 128, true, false), 160);
        assert_eq!(image_rider_cap(17, 3, true, false), 17);
        assert_eq!(image_rider_cap(5, 3, true, true), 5);
        assert_eq!(image_rider_cap(128, 8, true, true), 16);
        assert_eq!(image_rider_cap(128, 95, true, false), 128);
        assert_eq!(image_rider_cap(128, 96, true, false), 128);
        assert_eq!(image_rider_cap(128, 127, true, false), 128);
    }

    #[test]
    fn speculation_declines_only_rounds_past_the_context_edge() {
        use super::verify_fits;
        assert!(verify_fits(4096, &[(0, 4093, vec![1; 3])]));
        assert!(!verify_fits(4096, &[(0, 4093, vec![1; 4])]));
        assert!(!verify_fits(
            4096,
            &[(0, 10, vec![1]), (3, 4095, vec![1; 2])]
        ));
        assert!(!verify_fits(4096, &[(0, usize::MAX, vec![1])]));
    }

    #[test]
    fn idle_admission_quantum_preserves_decode_resume_and_full_cohort_grants() {
        assert_eq!(admission_cap(512, 0, 1, 1, true), 512);
        assert_eq!(admission_cap(512, 0, 4, 0, true), 512);
        for pending in 1..4 {
            assert_eq!(admission_cap(512, 0, 4, pending, true), 32);
            assert_eq!(admission_cap(512, 0, 4, pending, false), 512);
            assert_eq!(admission_cap(128, 1, 4, pending, true), 128);
            assert_eq!(admission_cap(17, 0, 4, pending, true), 17);
        }
        assert_eq!(admission_cap(512, 0, 4, 4, true), 512);
    }
}
