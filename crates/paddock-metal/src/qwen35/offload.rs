//! Exact-boundary Qwen checkpoints: attention pages + every recurrent/conv
//! layer + MTP pending/KV + DFlash conditioning rings. No lossy conversion.
use super::*;
use paddock_engine::kv_tier::{
    cold::{ColdCache, ColdConfig, Key},
    digest::LogicalKey,
    nvme_store::NvmeStore,
};

pub(super) struct Tier {
    // Dropped/joined before any checkpoint destinations can be released.
    scatter: crate::offload::ScatterWorker,
    restoring: Vec<Restoring>,
    cache: ColdCache,
    root: LogicalKey,
    cost: paddock_engine::kv_tier::CostModel,
    measured_prefill: bool,
    decisions: paddock_engine::kv_tier::TierDecisions,
    restores: std::collections::HashMap<Key, (std::time::Instant, f64)>,
}
struct Restoring {
    index: usize,
    table: BlockTable,
    history: Vec<u32>,
    images: Vec<multimodal::ImageKey>,
    key: Key,
    bytes: usize,
    started: std::time::Instant,
    done: std::sync::mpsc::Receiver<()>,
}
impl Tier {
    fn keys(&self, tokens: &[u32]) -> Vec<(Key, usize)> {
        self.image_keys(tokens, &[])
    }
    fn image_keys(&self, tokens: &[u32], images: &[multimodal::ImageKey]) -> Vec<(Key, usize)> {
        let mut key = self.root;
        tokens
            .chunks_exact(BLOCK_TOKENS)
            .enumerate()
            .filter_map(|(i, chunk)| {
                key = key.child(chunk);
                let n = (i + 1) * BLOCK_TOKENS;
                for image in images
                    .iter()
                    .filter(|image| image.starts_in(n - BLOCK_TOKENS, n))
                {
                    key = image.chain(key);
                }
                (n >= 3 * BLOCK_TOKENS && n < tokens.len() && !images.iter().any(|i| i.inside(n)))
                    .then_some((key.0, n))
            })
            .collect()
    }
    #[cfg(test)]
    fn key(&self, tokens: &[u32]) -> Key {
        self.image_key(tokens, &[])
    }
    fn image_key(&self, tokens: &[u32], images: &[multimodal::ImageKey]) -> Key {
        let mut key = self.root;
        for (i, chunk) in tokens.chunks_exact(BLOCK_TOKENS).enumerate() {
            key = key.child(chunk);
            for image in images
                .iter()
                .filter(|image| image.starts_in(i * BLOCK_TOKENS, (i + 1) * BLOCK_TOKENS))
            {
                key = image.chain(key);
            }
        }
        key.0
    }
}
impl Qwen35 {
    /// Called after companions attach and before serving. Paths include every
    /// checkpoint/companion influencing the state; schema also keys exact shapes.
    pub fn enable_kv_offload(
        &mut self,
        config: crate::KvOffloadConfig,
        paths: &[&Path],
    ) -> Result<()> {
        if self.cold.is_some() || self.slots.iter().any(|s| !s.history.is_empty()) {
            return Err(MetalError::Model(
                "enable KV offload once, before inference".into(),
            ));
        }
        let mut layout = format!(
            "qwen-metal-v1:{:?}:mlx={}:ctx={}:mtp={}:dflash={}:eps={}:rope={}:rotary={}:platform={}",
            self.geometry,
            self.mlx,
            self.context,
            self.mtp.is_some(),
            self.dflash.is_some(),
            self.eps.to_bits(),
            self.rope.to_bits(),
            self.rotary,
            self.device.checkpoint_platform()
        );
        if self.splash {
            layout.push_str(":splash-packed-v1:bf16-draft:stable-prefix-attention");
        }
        if self.bonsai.is_some() {
            layout.push_str(":bonsai-ternary-f32-v1");
            if self.stable_bonsai_prefill() {
                // Changed prompt contractions must never reuse older text
                // or image recurrent snapshots, including after restart.
                layout.push_str(":phase-local-m5-prefill-v2");
            }
        }
        if self.ternary.is_some() {
            layout.push_str(":bonsai-ptq1-gguf-f16-v1");
            let add_decode = self.device.tensor_accelerated();
            #[cfg(test)]
            let add_decode = add_decode && !ternary_add_tests::BASELINE_PTQ.with(|v| v.get());
            if add_decode {
                // Changed decode/head contractions influence every later
                // text or image prefix, including after process restart.
                layout.push_str(":ptq1-sign-add-decode-v2");
            }
        }
        #[cfg(test)]
        if (self.bonsai.is_some() || self.ternary.is_some())
            && ternary_add_tests::ADD_PROJECTIONS.with(|v| v.get())
        {
            layout.push_str(":ternary-sign-add-group-scale-v1");
        }
        if self.stable_affine_prefill() {
            // Old admission-sized BF16 partials produce different recurrent
            // states. Never restore them under the stable cold-text contract.
            layout.push_str(":affine-cold-text-prefill512-v1");
            // Decode may feed a later conversation prefix. States produced
            // by batch-dependent softmax reductions are not interchangeable.
            layout.push_str(":sequence-local-decode-attention-v1");
            layout.push_str(":phase-local-affine-f32-v1");
        }
        if self.register_language_attention() {
            // Attention changes every later recurrent snapshot; never restore
            // the old BF16-probability/16-key arithmetic under this graph.
            layout.push_str(":paged-nax-head256-f32-probability-v1");
        }
        crate::offload::require_unchanged(&self.source_versions)?;
        let ns = crate::offload::namespace(paths, layout.as_bytes(), config.scope)?;
        crate::offload::require_unchanged(&self.source_versions)?;
        let max_bytes = self
            .cold_spans(0, &(0..self.page_stride as u32).collect::<Vec<_>>())
            .iter()
            .map(|(_, _, n)| n)
            .sum();
        let minimum = self
            .cold_spans(0, &[0, 1, 2])
            .iter()
            .map(|(_, _, n)| *n as u64)
            .sum::<u64>()
            * 3
            + (128 << 10);
        if config.ram_bytes < minimum {
            return Err(MetalError::Memory(format!(
                "KV offload needs at least {} MiB of unified-RAM transfer budget for this model's complete recurrent checkpoint",
                minimum.div_ceil(1 << 20)
            )));
        }
        let disk = config
            .disk
            .map(|(root, quota)| (NvmeStore::dir_for(&root, &ns), quota));
        let cache = ColdCache::open(
            ColdConfig {
                ram_bytes: config.ram_bytes,
                disk,
            },
            max_bytes,
        )
        .map_err(MetalError::Model)?;
        let mut cost = paddock_engine::kv_tier::CostModel::new();
        cost.seed_nvme(cache.device_read_gbs);
        self.cold = Some(Tier {
            scatter: crate::offload::ScatterWorker::new()?,
            restoring: Vec::new(),
            cache,
            root: ns.root(),
            cost,
            measured_prefill: false,
            decisions: Default::default(),
            restores: Default::default(),
        });
        Ok(())
    }
    fn cold_spans(&self, slot: usize, blocks: &[u32]) -> Vec<crate::offload::Span<'_>> {
        let mut spans = Vec::new();
        let bytes =
            BLOCK_TOKENS * self.geometry.kv_heads * 256 * if self.bonsai.is_some() { 4 } else { 2 };
        for layer in &self.layers {
            if let Mixer::Full(a) = &layer.mixer {
                for buffer in [&a.keys, &a.values] {
                    spans.extend(crate::offload::paged_spans(buffer, blocks, bytes));
                }
            }
        }
        for layer in 0..self.geometry.linear_layers() {
            for (buffer, stride) in [
                (&self.state, self.geometry.state() * 4),
                (&self.conv, self.geometry.conv() * 12),
            ] {
                spans.push((buffer, (layer * self.state_slots + slot) * stride, stride));
            }
        }
        spans.extend(self.mtp_cold_spans(slot, blocks));
        spans.extend(self.dflash_cold_spans(slot));
        spans
    }
    pub(super) fn spill_checkpoint(&mut self, index: usize) {
        let checkpoint = &self.cache[index];
        let Some(tier) = &mut self.cold else {
            return;
        };
        if checkpoint.reserved || checkpoint.history.is_empty() {
            return;
        }
        let key = tier.image_key(&checkpoint.history, &checkpoint.images);
        tier.cache.pump();
        if tier.cache.contains(&key) {
            return;
        }
        let bytes = self
            .cold_spans(self.slots.len() + index, self.cache[index].table.blocks())
            .iter()
            .map(|(_, _, n)| n)
            .sum();
        let Some(reservation) = self
            .cold
            .as_mut()
            .expect("cold tier remains enabled while computing spill spans")
            .cache
            .reserve(bytes)
        else {
            return;
        };
        let spans = self.cold_spans(self.slots.len() + index, self.cache[index].table.blocks());
        match crate::offload::capture(&self.device, &spans, reservation) {
            Ok(payload) => self
                .cold
                .as_mut()
                .expect("capture does not replace the cold tier")
                .cache
                .put(key, payload),
            Err(error) => tracing::warn!(%error, "Metal KV capture failed; prefix will recompute"),
        }
    }
    pub(super) fn cold_loading(&mut self, tokens: &[u32]) -> bool {
        self.cold_loading_images(tokens, &[])
    }
    pub(super) fn cold_loading_images(
        &mut self,
        tokens: &[u32],
        images: &[multimodal::ImageKey],
    ) -> bool {
        if tokens.len() > self.context {
            return false;
        }
        self.pump_cold();
        let Some(tier) = &mut self.cold else {
            return false;
        };
        if tier.restoring.iter().any(|r| {
            r.history.len() < tokens.len()
                && tokens.starts_with(&r.history)
                && multimodal::prefix_images_match(&r.images, images, r.history.len())
        }) {
            return true;
        }
        // Never park for a shorter disk prefix when a longer one is resident.
        let resident = self
            .cache
            .iter()
            .filter(|c| {
                !c.reserved
                    && c.history.len() < tokens.len()
                    && tokens.starts_with(&c.history)
                    && multimodal::prefix_images_match(&c.images, images, c.history.len())
            })
            .map(|c| c.history.len())
            .max()
            .unwrap_or(0);
        tier.cache.pump();
        let hit = tier
            .image_keys(tokens, images)
            .into_iter()
            .rev()
            .find(|(key, n)| *n > resident && tier.cache.contains(key));
        tier.decisions.lookups += 1;
        let Some((key, n)) = hit else {
            tier.decisions.miss_cold += 1;
            return false;
        };
        tier.decisions.hits += 1;
        let Some((bytes, disk)) = tier.cache.hit_size(&key) else {
            return false;
        };
        let transfer = bytes as u64 + if disk { tier.cache.queued_bytes() } else { 0 };
        let election = tier.cost.elect(paddock_engine::kv_tier::HitShape {
            restore_bytes: transfer,
            restore_tokens: (n - resident) as u32,
            // The disk worker serializes durable writes with reads; price their
            // queue at the disk rate, not a CUDA host-transfer bandwidth.
            queued_bytes: 0,
            nvme_bytes: if disk { transfer } else { 0 },
        });
        if !tier.restores.contains_key(&key) && tier.measured_prefill && !election.is_restore() {
            tier.decisions.elected_recompute += 1;
            return false;
        }
        if disk && tier.cache.writes_pending() && !tier.restores.contains_key(&key) {
            // Decode admission never waits behind a durable-write backlog.
            tier.decisions.park_refused += 1;
            return false;
        }
        let (estimate, recompute) = match election {
            paddock_engine::kv_tier::Election::Restore {
                est_us,
                recompute_us,
            } => (est_us, recompute_us),
            paddock_engine::kv_tier::Election::Recompute { est_us, restore_us } => {
                (restore_us, est_us)
            }
        };
        // A fixed 500 ms deadline loses the back half of a four-request cohort
        // even when restoring all four is far cheaper than recomputation.
        let budget_us = (estimate * 1.5 + 50_000.)
            .min(if tier.measured_prefill {
                recompute
            } else {
                2_000_000.
            })
            .clamp(50_000., 2_000_000.);
        let parked = tier
            .cache
            .load_until(key, std::time::Duration::from_secs_f64(budget_us / 1e6));
        if parked && !tier.restores.contains_key(&key) {
            tier.restores
                .insert(key, (std::time::Instant::now(), election.chosen_us()));
            tier.decisions.elected_restore += 1;
            tier.decisions.parked += 1;
        }
        if parked {
            return true;
        }
        let payload = tier.cache.get(&key);
        if let Some(payload) = payload {
            return self.start_cold_scatter(key, &tokens[..n], images, payload);
        }
        false
    }
    fn reserve_cold_table(&mut self, n: usize) -> Option<(usize, BlockTable)> {
        let index = self
            .cache
            .iter()
            .enumerate()
            .filter(|(_, c)| !c.reserved)
            .min_by_key(|(_, c)| (usize::from(!c.history.is_empty()), c.touched))
            .map(|(i, _)| i)?;
        self.spill_checkpoint(index);
        self.cache[index].table.clear(&mut self.pool);
        self.cache[index].history.clear();
        self.cache[index].images.clear();
        while self.pool.free_blocks() < n / BLOCK_TOKENS {
            if !self.evict_checkpoint() {
                return None;
            }
        }
        let mut table = BlockTable::new();
        if table.ensure(n - 1, &mut self.pool).is_err() {
            table.clear(&mut self.pool);
            return None;
        }
        Some((index, table))
    }
    fn start_cold_scatter(
        &mut self,
        key: Key,
        history: &[u32],
        images: &[multimodal::ImageKey],
        payload: std::sync::Arc<paddock_engine::kv_tier::cold::Payload>,
    ) -> bool {
        // One physical staging context, one copy lane. Additional reads may
        // finish in budgeted RAM, but never reserve another unpublishable page
        // table. A cancelled request cannot consume a live slot's growth room.
        let tier = self
            .cold
            .as_ref()
            .expect("scatter starts only after an enabled cold-tier hit");
        if !tier.restoring.is_empty() {
            return true;
        }
        if self.require_committed().is_err() {
            return false;
        }
        let Some((index, mut table)) = self.reserve_cold_table(history.len()) else {
            return false;
        };
        self.cache[index].reserved = true;
        let spans = self.cold_spans(self.slots.len() + index, table.blocks());
        let bytes = payload.bytes().len();
        let result = self
            .cold
            .as_ref()
            .expect("reserving restore pages does not replace the cold tier")
            .scatter
            .submit(&spans, payload);
        match result {
            Ok(done) => {
                self.cold
                    .as_mut()
                    .expect("scatter submission retains the cold tier")
                    .restoring
                    .push(Restoring {
                        index,
                        table,
                        history: history.to_vec(),
                        images: images
                            .iter()
                            .filter(|i| i.end() <= history.len())
                            .cloned()
                            .collect(),
                        key,
                        bytes,
                        started: std::time::Instant::now(),
                        done,
                    });
                true
            }
            Err(error) => {
                self.cache[index].reserved = false;
                table.clear(&mut self.pool);
                tracing::warn!(%error, "Metal KV async scatter refused; prefix will recompute");
                false
            }
        }
    }
    pub(super) fn restore_cold(&mut self, tokens: &[u32]) -> Result<()> {
        let Some(tier) = &mut self.cold else {
            return Ok(());
        };
        tier.cache.pump();
        let hit = tier
            .keys(tokens)
            .into_iter()
            .rev()
            .find_map(|(key, n)| tier.cache.get(&key).map(|p| (key, n, p)));
        let Some((key, n, payload)) = hit else {
            return Ok(());
        };
        let restore_started = std::time::Instant::now();
        if self.cache.iter().any(|c| {
            c.images.is_empty()
                && c.history.len() >= n
                && c.history.len() < tokens.len()
                && tokens.starts_with(&c.history)
        }) {
            return Ok(());
        }
        let Some((index, mut table)) = self.reserve_cold_table(n) else {
            return Ok(());
        };
        let spans = self.cold_spans(self.slots.len() + index, table.blocks());
        if let Err(error) = crate::offload::restore(&spans, payload.bytes()) {
            table.clear(&mut self.pool);
            tracing::warn!(%error, "Metal KV restore rejected; prefix will recompute");
            return Ok(());
        }
        self.clock += 1;
        self.cache[index] = Checkpoint {
            reserved: false,
            table,
            history: tokens[..n].to_vec(),
            touched: self.clock,
            images: Vec::new(),
        };
        let tier = self
            .cold
            .as_mut()
            .expect("restoring a checkpoint does not replace the cold tier");
        let disk = tier.restores.remove(&key);
        let elapsed = disk.map_or_else(
            || restore_started.elapsed().as_secs_f64() * 1e6,
            |(start, _)| start.elapsed().as_secs_f64() * 1e6,
        );
        tier.cost.observe_restore_from(
            payload.bytes().len() as u64,
            elapsed,
            disk.map_or(elapsed, |(_, predicted)| predicted),
            disk.is_some(),
        );
        tier.decisions.resolved_ok += 1;
        tier.decisions.useful_bytes += payload.bytes().len() as u64;
        tier.decisions.moved_bytes += payload.bytes().len() as u64;
        if disk.is_some() {
            tier.decisions.served_from_nvme += 1;
        } else {
            tier.decisions.served_from_ram += 1;
        }
        Ok(())
    }
    pub(super) fn pump_cold(&mut self) {
        let Some(tier) = &mut self.cold else {
            return;
        };
        let mut finished = Vec::new();
        tier.cache.pump();
        let mut i = 0;
        while i < tier.restoring.len() {
            match tier.restoring[i].done.try_recv() {
                Ok(()) => {
                    finished.push((tier.restoring.swap_remove(i), true));
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    finished.push((tier.restoring.swap_remove(i), false));
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    i += 1;
                }
            }
        }
        tier.restores
            .retain(|_, (start, _)| start.elapsed().as_secs() < 3);
        for (mut restore, ok) in finished {
            self.cache[restore.index].reserved = false;
            if !ok {
                restore.table.clear(&mut self.pool);
                tracing::warn!("Metal KV scatter worker failed; unpublished checkpoint discarded");
                continue;
            }
            self.clock += 1;
            self.cache[restore.index] = Checkpoint {
                table: restore.table,
                history: restore.history,
                images: restore.images,
                touched: self.clock,
                ..Default::default()
            };
            let disk = tier.restores.remove(&restore.key);
            let elapsed = disk
                .map_or(restore.started, |(start, _)| start)
                .elapsed()
                .as_secs_f64()
                * 1e6;
            tier.cost.observe_restore_from(
                restore.bytes as u64,
                elapsed,
                disk.map_or(elapsed, |(_, predicted)| predicted),
                disk.is_some(),
            );
            tier.decisions.resolved_ok += 1;
            tier.decisions.useful_bytes += restore.bytes as u64;
            tier.decisions.moved_bytes += restore.bytes as u64;
            if disk.is_some() {
                tier.decisions.served_from_nvme += 1;
            } else {
                tier.decisions.served_from_ram += 1;
            }
        }
    }
    pub(super) fn observe_cold_prefill(&mut self, tokens: u32, us: f64) {
        if let Some(tier) = &mut self.cold {
            if tier.measured_prefill {
                tier.cost.observe_prefill(tokens, us);
            } else if tokens > 0 && us.is_finite() && us > 0.0 {
                tier.cost.seed_prefill(tokens, us);
                tier.measured_prefill = true;
            }
        }
    }
    pub(super) fn cold_report(&self) -> Option<paddock_engine::kv_tier::TierReport> {
        let t = self.cold.as_ref()?;
        let s = t.cache.stats();
        let (ram, nvme) = t.cost.rates_bpus();
        let (ram_capacity, disk_capacity) = t.cache.capacity_bytes();
        Some(paddock_engine::kv_tier::TierReport {
            t1_ready_bytes: s.ready_bytes,
            t1_reserved_bytes: t.cache.allocated_bytes().saturating_sub(s.ready_bytes),
            t1_capacity_bytes: ram_capacity,
            t2_capacity_bytes: disk_capacity,
            t2_ready_bytes: t.cache.disk_bytes(),
            decisions: t.decisions,
            resident_runs: s.resident_runs,
            in_flight_demotes: s.in_flight_demotes,
            open_tickets: s.open_tickets + t.restoring.len() as u64,
            tripped: s.tripped,
            io_failures: s.io_failures,
            integrity_failures: s.integrity_failures,
            evictions: s.evictions,
            rate_ram_bpus: ram,
            rate_nvme_bpus: nvme,
            device_read_gbs: t.cache.device_read_gbs,
            prediction_error_pct: t.cost.prediction_error_pct(),
            t2_written_day_bytes: s.t2_written_day_bytes,
            ..Default::default()
        })
    }
    pub(super) fn cold_stats(&self) -> Option<paddock_engine::kv_tier::TierStats> {
        self.cold.as_ref().map(|t| {
            let mut s = t.cache.stats();
            s.open_tickets += t.restoring.len() as u64;
            s
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires Bonsai PTQ1; changed add/subtract decode must have a different persistent namespace"]
    fn ptq1_add_decode_has_distinct_namespace() {
        struct Reset;
        impl Drop for Reset {
            fn drop(&mut self) {
                ternary_add_tests::BASELINE_PTQ.with(|v| v.set(false));
            }
        }
        let _reset = Reset;
        let path = std::env::var("PADDOCK_METAL_PTQ1_MODEL").unwrap();
        let mut model = Qwen35::load_for_offload(Path::new(&path), 1024, 1, None).unwrap();
        if !model.device.tensor_accelerated() {
            return;
        }
        let mut roots = Vec::new();
        for baseline in [true, false, true, false] {
            ternary_add_tests::BASELINE_PTQ.with(|v| v.set(baseline));
            model
                .enable_kv_offload(
                    crate::KvOffloadConfig {
                        ram_bytes: 1 << 30,
                        disk: None,
                        scope: b"ptq1-add-arithmetic-namespace-test".to_vec(),
                    },
                    &[Path::new(&path)],
                )
                .unwrap();
            roots.push(model.cold.as_ref().unwrap().root);
            drop(model.cold.take());
        }
        assert_ne!(roots[0], roots[1]);
        assert_eq!(roots[0], roots[2]);
        assert_eq!(roots[1], roots[3]);
    }
    #[test]
    #[ignore = "requires Bonsai; changed M5 prompt arithmetic must not share a persistent namespace"]
    fn bonsai_prompt_arithmetic_has_distinct_namespace() {
        struct Reset;
        impl Drop for Reset {
            fn drop(&mut self) {
                bonsai::BASELINE_PREFILL.with(|v| v.set(false));
            }
        }
        let _reset = Reset;
        let path = std::env::var("PADDOCK_METAL_BONSAI_MODEL").unwrap();
        let mut model = Qwen35::load_for_offload(Path::new(&path), 1024, 1, None).unwrap();
        if !model.device.tensor_accelerated() {
            return;
        }
        let mut roots = Vec::new();
        for old in [true, false, true, false] {
            bonsai::BASELINE_PREFILL.with(|v| v.set(old));
            model
                .enable_kv_offload(
                    crate::KvOffloadConfig {
                        ram_bytes: 1 << 30,
                        disk: None,
                        scope: b"bonsai-arithmetic-namespace-test".to_vec(),
                    },
                    &[Path::new(&path)],
                )
                .unwrap();
            roots.push(model.cold.as_ref().unwrap().root);
            drop(model.cold.take());
        }
        assert_ne!(roots[0], roots[1]);
        assert_eq!(roots[0], roots[2]);
        assert_eq!(roots[1], roots[3]);
    }

    #[test]
    #[ignore = "requires 0.8B MLX vision checkpoint; persistent cache namespace guard"]
    fn lighton_attention_arithmetic_has_distinct_namespace() {
        struct Reset;
        impl Drop for Reset {
            fn drop(&mut self) {
                attention::BASELINE_LANGUAGE_NAX_FOR_TEST.with(|v| v.set(false));
            }
        }
        let _reset = Reset;
        let path = std::env::var_os("PADDOCK_METAL_QWEN_MODEL").unwrap();
        let path = Path::new(&path);
        let mut model = Qwen35::load_for_offload(path, 1024, 1, None).unwrap();
        model.attach_vision(path).unwrap();
        if !model.device.tensor_accelerated() {
            return;
        }
        assert_eq!(model.geometry, Geometry::DENSE_08B);
        let mut roots = Vec::new();
        for baseline in [true, false, true, false] {
            attention::BASELINE_LANGUAGE_NAX_FOR_TEST.with(|v| v.set(baseline));
            model
                .enable_kv_offload(
                    crate::KvOffloadConfig {
                        ram_bytes: 1 << 30,
                        disk: None,
                        scope: b"lighton-arithmetic-namespace".to_vec(),
                    },
                    &[path],
                )
                .unwrap();
            roots.push(model.cold.as_ref().unwrap().root);
            drop(model.cold.take());
        }
        assert_ne!(roots[0], roots[1]);
        assert_eq!(roots[0], roots[2]);
        assert_eq!(roots[1], roots[3]);
    }

    #[test]
    #[ignore = "requires PADDOCK_METAL_QWEN_MODEL + PADDOCK_METAL_MMPROJ; image disk restart"]
    fn image_disk_restart_skips_tower_and_preserves_c4_logits() {
        use paddock_engine::service::MmChunk;
        let path = std::env::var("PADDOCK_METAL_QWEN_MODEL").unwrap();
        let mm = std::env::var("PADDOCK_METAL_MMPROJ").unwrap();
        let companion = std::env::var("PADDOCK_METAL_KV_DRAFTER").ok();
        let dir = std::env::temp_dir().join(format!(
            "paddock-image-kv-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&dir).unwrap();
        let cfg = crate::KvOffloadConfig {
            ram_bytes: 2 << 30,
            disk: Some((dir.clone(), 2 << 30)),
            scope: b"image-restart".to_vec(),
        };
        let load = || {
            let mut m = Qwen35::load_for_offload(Path::new(&path), 1024, 4, None).unwrap();
            m.attach_vision(Path::new(&mm)).unwrap();
            let mut paths = vec![Path::new(&path), Path::new(&mm)];
            if let Some(d) = &companion {
                m.attach_dflash(Path::new(d)).unwrap();
                paths.push(Path::new(d));
            }
            m.enable_kv_offload(cfg.clone(), &paths).unwrap();
            m
        };
        let image = |side, color: [u8; 3]| MmChunk::Image {
            rgb: (0..side * side).flat_map(|_| color).collect(),
            w: side,
            h: side,
        };
        let prompt = vec![
            MmChunk::Text(vec![100; 70]),
            image(128, [120, 10, 30]),
            MmChunk::Text(vec![200; 20]),
            image(192, [20, 80, 240]),
            MmChunk::Text(vec![300; 50]),
        ];
        let mut m = load();
        m.prefill_images(0, &prompt).unwrap();
        let images = m.layout(prompt.clone()).unwrap().keys;
        let index = m
            .cache
            .iter()
            .enumerate()
            .filter(|(_, c)| !c.history.is_empty())
            .max_by_key(|(_, c)| c.history.len())
            .unwrap()
            .0;
        let boundary = m.cache[index].history.len();
        assert!(images.iter().all(|i| i.end() <= boundary));
        m.spill_checkpoint(index);
        m.reset();
        let mut references = Vec::new();
        let mut count = 0;
        for s in 0..4 {
            let (logits, n) = m.prefill_images(s, &prompt).unwrap();
            count = n;
            assert_eq!(m.take_prefill_reused(s), boundary);
            assert!(
                m.slots[s].mm.as_ref().unwrap().images.is_empty(),
                "resident hit should skip tower too"
            );
            references.push(logits);
        }
        let mut decodes = Vec::new();
        for step in 0..16 {
            decodes.push(
                m.forward_batch(&[9000 + step; 4], &[count as u32 + step; 4])
                    .unwrap(),
            );
        }
        let companion_round = |m: &mut Qwen35| {
            if !m.spec_capable() {
                return None;
            }
            let pending: Vec<_> = (0..4).map(|s| (s, 9876 + s as u32)).collect();
            let drafts = m.spec_draft_batch(&pending, 3).unwrap().unwrap();
            let requests: Vec<_> = pending
                .iter()
                .zip(&drafts)
                .map(|(&(s, token), draft)| {
                    (
                        s,
                        m.slots[s].history.len(),
                        std::iter::once(token)
                            .chain(draft.iter().copied())
                            .collect::<Vec<_>>(),
                    )
                })
                .collect();
            let logits = m.forward_spec_verify(&requests).unwrap().unwrap();
            m.spec_commit(&[1, 2, 3, 4]).unwrap();
            let rows: Vec<_> = (0..4)
                .map(|s| (s, 7654 + s as u32, m.slots[s].history.len() as u32))
                .collect();
            Some((drafts, logits, m.execute(&rows, &[0, 1, 2, 3]).unwrap()))
        };
        let companion_reference = companion_round(&mut m);
        let start = std::time::Instant::now();
        while m.cold_stats().unwrap().in_flight_demotes != 0 {
            assert!(start.elapsed().as_secs() < 30);
            m.pump_cold();
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        drop(m);
        let mut m = load();
        let start = std::time::Instant::now();
        for (s, reference) in references.iter().enumerate() {
            let (logits, _) = m.prefill_images(s, &prompt).unwrap();
            assert_eq!(
                m.take_prefill_reused(s),
                boundary,
                "disk reuse, not recomputation: {:?}",
                m.cold_report()
            );
            assert_eq!(logits, *reference, "full multimodal suffix logits");
            assert!(m.slots[s].mm.as_ref().unwrap().images.is_empty());
            assert!(
                m.image_cache.is_empty(),
                "no tower invocation after restart"
            );
        }
        for step in 0..16 {
            assert_eq!(
                m.forward_batch(&[9000 + step; 4], &[count as u32 + step; 4])
                    .unwrap(),
                decodes[step as usize],
                "full image decode logits {step}"
            );
        }
        assert!(m.cold_report().unwrap().decisions.served_from_nvme > 0);
        assert_eq!(
            companion_round(&mut m),
            companion_reference,
            "image conditioning, draft/verify and divergent rollback survive restart"
        );
        // Same placeholder token IDs, different pixels: neither resident nor
        // persistent state may alias. Dimensions/position are also committed.
        let mut changed = prompt.clone();
        changed[1] = image(128, [121, 10, 30]);
        let original = m.layout(prompt.clone()).unwrap();
        let different = m.layout(changed).unwrap();
        assert_eq!(original.ids, different.ids);
        let tier = m.cold.as_ref().unwrap();
        assert_ne!(
            tier.image_key(&original.ids[..boundary], &original.keys),
            tier.image_key(&different.ids[..boundary], &different.keys)
        );
        assert!(!multimodal::prefix_images_match(
            &original.keys,
            &different.keys,
            boundary
        ));
        for (_, n) in tier.image_keys(&original.ids, &original.keys) {
            assert!(!original.inside_image(n));
        }
        let text_key = tier.key(&original.ids[..boundary]);
        assert_ne!(
            text_key,
            tier.image_key(&original.ids[..boundary], &original.keys)
        );
        println!(
            "IMAGE_KV_RESTART reused={boundary} c=4 full_logits_steps=16 tower_runs=0 elapsed_ms={:.3}",
            start.elapsed().as_secs_f64() * 1000.
        );
        drop(m);
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    #[ignore = "requires PADDOCK_METAL_QWEN_MODEL; c1/c4 cold/warm engine timing"]
    fn offload_c1_c4_ttft_benchmark() {
        let path = std::env::var("PADDOCK_METAL_QWEN_MODEL").unwrap();
        let dir = std::env::temp_dir().join(format!(
            "paddock-kv-bench-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&dir).unwrap();
        let cfg = crate::KvOffloadConfig {
            ram_bytes: 4 << 30,
            disk: Some((dir.clone(), 4 << 30)),
            scope: b"isolated-timing-test".to_vec(),
        };
        let mut m = Qwen35::load_for_offload(Path::new(&path), 4096, 4, None).unwrap();
        m.enable_kv_offload(cfg, &[Path::new(&path)]).unwrap();
        for concurrency in [1, 4] {
            let prompts: Vec<Vec<u32>> = (0..concurrency)
                .map(|s| {
                    (0..2057)
                        .map(|i| 1000 + (i + s * 3000 + concurrency * 17000) as u32)
                        .collect()
                })
                .collect();
            for mode in ["recompute", "resident", "ssd"] {
                m.reset();
                if mode == "ssd" {
                    for prompt in &prompts {
                        let i = m
                            .cache
                            .iter()
                            .position(|c| c.history == prompt[..2048])
                            .unwrap();
                        m.spill_checkpoint(i);
                    }
                    let start = std::time::Instant::now();
                    while m.cold_stats().unwrap().in_flight_demotes > 0 {
                        assert!(start.elapsed().as_secs() < 30);
                        m.pump_cold();
                        std::thread::sleep(std::time::Duration::from_millis(2));
                    }
                    for prompt in &prompts {
                        let tier = m.cold.as_ref().unwrap();
                        let key = tier.key(&prompt[..2048]);
                        assert!(
                            tier.cache.hit_size(&key).is_some(),
                            "checkpoint was not captured before disk reopen"
                        );
                    }
                    // Reopen the real store, dropping every hot payload. Keep
                    // the loaded graph and measured prefill rate constant.
                    let t = m.cold.as_mut().unwrap();
                    let root = t.root;
                    // The configured namespace directory is discoverable under
                    // this test-owned root only; do not touch any app cache.
                    fn find_store(p: &Path) -> std::path::PathBuf {
                        if p.join("store.meta").is_file() {
                            return p.to_path_buf();
                        }
                        for e in std::fs::read_dir(p).unwrap() {
                            let child = e.unwrap().path();
                            if child.is_dir() {
                                let got = find_store(&child);
                                if !got.as_os_str().is_empty() {
                                    return got;
                                }
                            }
                        }
                        std::path::PathBuf::new()
                    }
                    let store_dir = find_store(&dir);
                    let old = std::mem::replace(
                        &mut t.cache,
                        ColdCache::open(
                            ColdConfig {
                                ram_bytes: 4 << 30,
                                disk: None,
                            },
                            1,
                        )
                        .unwrap(),
                    );
                    drop(old);
                    t.cache = ColdCache::open(
                        ColdConfig {
                            ram_bytes: 4 << 30,
                            disk: Some((store_dir, 4 << 30)),
                        },
                        1 << 30,
                    )
                    .unwrap();
                    assert_eq!(root, t.root);
                    for c in &mut m.cache {
                        c.table.clear(&mut m.pool);
                        c.history.clear();
                        c.images.clear();
                    }
                }
                let start = std::time::Instant::now();
                let mut admitted = vec![false; concurrency];
                let mut finished = vec![false; concurrency];
                let mut ttft = vec![0.; concurrency];
                let mut reused = vec![0; concurrency];
                let mut generated = vec![0usize; concurrency];
                let mut last = vec![0.; concurrency];
                let mut gaps = Vec::new();
                while finished.iter().any(|v| !v) || generated.iter().any(|n| *n < 16) {
                    assert!(start.elapsed().as_secs() < 60);
                    m.tier_pump();
                    for s in 0..concurrency {
                        if !admitted[s] && !m.tier_prefix_loading(s, &prompts[s]) {
                            m.prefill_begin(s, prompts[s].clone()).unwrap();
                            reused[s] = m.take_prefill_reused(s);
                            admitted[s] = true;
                        }
                    }
                    let tick = std::time::Instant::now();
                    let decodes: Vec<_> = (0..concurrency)
                        .filter(|&s| finished[s] && generated[s] < 16)
                        .map(|s| {
                            (
                                s,
                                9000 + generated[s] as u32,
                                m.slots[s].history.len() as u32,
                            )
                        })
                        .collect();
                    let (_, done) = m.forward_mixed(&decodes, 512).unwrap();
                    let now = start.elapsed().as_secs_f64() * 1000.;
                    for &(s, _, _) in &decodes {
                        generated[s] += 1;
                        gaps.push(now - last[s]);
                        last[s] = now;
                    }
                    for (s, _, _) in done {
                        finished[s] = true;
                        ttft[s] = now;
                        last[s] = now;
                    }
                    if mode == "recompute" && m.pending.iter().any(|p| p.offset > 0) {
                        m.tier_observe_prefill(512, tick.elapsed().as_secs_f64() * 1e6);
                    }
                    if !m.pending.is_empty() {
                        continue;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                gaps.sort_by(f64::total_cmp);
                let p99 = gaps[(gaps.len() * 99 / 100).min(gaps.len() - 1)];
                println!(
                    "METAL_KV_BENCH c={concurrency} mode={mode} ttft_ms={ttft:?} reused={reused:?} kv_bytes={} stream_gap_p99_ms={p99:.3}",
                    m.kv_bytes
                );
                if mode == "ssd" {
                    assert!(
                        reused.iter().all(|n| *n == 2048),
                        "SSD hit must not silently recompute: {reused:?}"
                    );
                }
            }
        }
        drop(m);
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    #[ignore = "requires PADDOCK_METAL_QWEN_MODEL; full GGUF/MLX disk restart parity"]
    fn disk_restart_restores_full_qwen_state_and_four_slot_generations() {
        let path = std::env::var("PADDOCK_METAL_QWEN_MODEL").expect("model path");
        let companion = std::env::var("PADDOCK_METAL_KV_DRAFTER").ok();
        let dir = std::env::temp_dir().join(format!(
            "paddock-metal-kv-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&dir).unwrap();
        let cfg = crate::KvOffloadConfig {
            ram_bytes: 1536 << 20,
            disk: Some((dir.clone(), 1 << 30)),
            scope: b"isolated-parity-test".to_vec(),
        };
        let load = || {
            let mut m = Qwen35::load_for_offload(Path::new(&path), 1024, 4, None).unwrap();
            let mut paths = vec![Path::new(&path)];
            if let Some(companion) = &companion {
                let cp = Path::new(companion);
                if companion.contains("dflash") {
                    m.attach_dflash(cp).unwrap();
                } else {
                    m.attach_mtp(cp).unwrap();
                }
                paths.push(cp);
            }
            m.enable_kv_offload(cfg.clone(), &paths).unwrap();
            m
        };
        let mut model = load();
        let p: Vec<u32> = (1000..1537).collect();
        model.prepare(0, &p).unwrap();
        for (start, end) in [(0, 512), (512, 528)] {
            model
                .execute(
                    &(start..end)
                        .map(|i| (0, p[i], i as u32))
                        .collect::<Vec<_>>(),
                    &[],
                )
                .unwrap();
        }
        let index = model
            .cache
            .iter()
            .position(|c| c.history.len() == 528)
            .unwrap();
        let start = std::time::Instant::now();
        model.spill_checkpoint(index);
        let capture_ms = start.elapsed().as_secs_f64() * 1000.;
        let key = model.cold.as_ref().unwrap().key(&p[..528]);
        let checksum = paddock_engine::kv_tier::Checksum::of_payload(
            model
                .cold
                .as_mut()
                .unwrap()
                .cache
                .get(&key)
                .unwrap()
                .bytes(),
        );
        let suffix = |slot| {
            (528..537)
                .map(|i| (slot, p[i], i as u32))
                .collect::<Vec<_>>()
        };
        let mut references = Vec::new();
        for slot in 0..4 {
            assert_eq!(model.prepare(slot, &p).unwrap(), 528);
            references.push(model.execute(&suffix(slot), &[8]).unwrap());
        }
        let mut generations = Vec::new();
        for step in 0..16 {
            let rows = (0..4)
                .map(|slot| (slot, (8000 + step * 4 + slot) as u32, (537 + step) as u32))
                .collect::<Vec<_>>();
            generations.push(model.execute(&rows, &[0, 1, 2, 3]).unwrap());
        }
        let spec_round = |m: &mut Qwen35| {
            if !m.spec_capable() {
                return None;
            }
            let pending: Vec<_> = (0..4).map(|s| (s, 9876 + s as u32)).collect();
            let draft = m.spec_draft_batch(&pending, 3).unwrap().unwrap();
            let reqs: Vec<_> = pending
                .iter()
                .zip(&draft)
                .map(|(&(s, token), d)| {
                    (
                        s,
                        m.slots[s].history.len(),
                        std::iter::once(token)
                            .chain(d.iter().copied())
                            .collect::<Vec<_>>(),
                    )
                })
                .collect();
            let logits = m.forward_spec_verify(&reqs).unwrap().unwrap();
            m.spec_commit(&[1, 2, 3, 4]).unwrap();
            let rows: Vec<_> = (0..4)
                .map(|s| (s, 7654 + s as u32, m.slots[s].history.len() as u32))
                .collect();
            let committed = m.execute(&rows, &[0, 1, 2, 3]).unwrap();
            Some((draft, logits, committed))
        };
        let spec_reference = spec_round(&mut model);
        let start = std::time::Instant::now();
        while model.cold_stats().unwrap().in_flight_demotes != 0 {
            assert!(start.elapsed().as_secs() < 15);
            model.pump_cold();
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert_eq!(model.cold_stats().unwrap().io_failures, 0);
        assert!(model.cold_stats().unwrap().t2_written_day_bytes > 0);
        let compact_bytes = model.kv_bytes;
        drop(model);
        let mut model = load();
        let start = std::time::Instant::now();
        while model.cold_loading(&p) {
            assert!(start.elapsed().as_secs() < 15);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let io_ms = start.elapsed().as_secs_f64() * 1000.;
        let got_checksum = paddock_engine::kv_tier::Checksum::of_payload(
            model
                .cold
                .as_mut()
                .unwrap()
                .cache
                .get(&key)
                .expect("disk result, not recomputation")
                .bytes(),
        );
        assert_eq!(checksum, got_checksum);
        let start = std::time::Instant::now();
        for (slot, reference) in references.iter().enumerate() {
            assert_eq!(
                model.prepare(slot, &p).unwrap(),
                528,
                "disk checkpoint slot {slot}"
            );
            assert_eq!(
                *reference,
                model.execute(&suffix(slot), &[8]).unwrap(),
                "all suffix logits slot {slot}"
            );
        }
        let restore_suffix_ms = start.elapsed().as_secs_f64() * 1000.;
        for (step, generation) in generations.iter().enumerate() {
            let rows = (0..4)
                .map(|slot| (slot, (8000 + step * 4 + slot) as u32, (537 + step) as u32))
                .collect::<Vec<_>>();
            assert_eq!(
                *generation,
                model.execute(&rows, &[0, 1, 2, 3]).unwrap(),
                "all decode logits c=4 step={step}"
            );
        }
        assert_eq!(
            spec_reference,
            spec_round(&mut model),
            "drafts, verifier logits and divergent accept/rollback state after disk restart"
        );
        model.reset();
        for checkpoint in &mut model.cache {
            checkpoint.table.clear(&mut model.pool);
            checkpoint.history.clear();
        }
        let tier = model.cold.as_mut().unwrap();
        tier.cost.seed_prefill(100_000, 1.0);
        tier.measured_prefill = true;
        assert!(!model.tier_prefix_loading(0, &p));
        assert_eq!(
            model.prepare(0, &p).unwrap(),
            0,
            "a recompute election must not turn into a synchronous cold restore during admission"
        );
        // Cancellation can leave copy destinations reserved while a recycled
        // slot reaches a new capture boundary. Planning must skip that optional
        // capture without releasing or overwriting any existing reservation.
        for c in &mut model.cache {
            c.reserved = true;
        }
        let plan = model
            .plan_execution(
                &(0..512).map(|i| (0, p[i], i as u32)).collect::<Vec<_>>(),
                &[],
            )
            .unwrap();
        model.unreserve_plan(&plan);
        assert!(model.cache.iter().all(|c| c.reserved));
        for c in &mut model.cache {
            c.reserved = false;
        }
        println!(
            "METAL_KV_PARITY capture_ms={capture_ms:.3} disk_read_ms={io_ms:.3} restore_four_suffixes_ms={restore_suffix_ms:.3} compact_kv_bytes={compact_bytes}"
        );
        drop(model);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
