//! Complete paged-attention checkpoints shared by non-recurrent families.
//! No live slot is a restore destination: a private table is published to the
//! existing radix only after the copy worker's completion fence.
use crate::{
    device::{Buffer, MetalDevice, MetalError, Result},
    offload,
};
use paddock_engine::{
    kv_pool::{BLOCK_TOKENS, BlockTable, KvPool},
    kv_tier::{
        CostModel, Election, HitShape, TierDecisions, TierReport, TierStats,
        cold::{ColdCache, ColdConfig, Key},
        digest::LogicalKey,
        nvme_store::NvmeStore,
    },
    paged_radix::PagedRadix,
};
use std::{
    collections::HashMap,
    path::Path,
    time::{Duration, Instant},
};

pub(crate) type Plane<'a> = (&'a Buffer, usize);
struct Restore {
    table: BlockTable,
    tokens: Vec<u32>,
    key: Key,
    bytes: usize,
    started: Instant,
    done: std::sync::mpsc::Receiver<()>,
}
pub(crate) struct PagedTier {
    // Join before destinations/Metal graph drop. Each family puts this field first.
    scatter: offload::ScatterWorker,
    restoring: Option<Restore>,
    cache: ColdCache,
    root: LogicalKey,
    cost: CostModel,
    measured: bool,
    decisions: TierDecisions,
    reads: HashMap<Key, (Instant, f64)>,
    context: usize,
}
fn spans<'a>(planes: &[Plane<'a>], blocks: &[u32]) -> Vec<offload::Span<'a>> {
    planes
        .iter()
        .flat_map(|&(b, bytes)| offload::paged_spans(b, blocks, bytes))
        .collect()
}
impl PagedTier {
    pub(crate) fn open(
        device: &MetalDevice,
        config: crate::KvOffloadConfig,
        paths: &[&Path],
        layout: &[u8],
        versions: &[offload::FileVersion],
        planes: &[Plane<'_>],
        context: usize,
    ) -> Result<Self> {
        offload::require_unchanged(versions)?;
        let ns = offload::namespace(paths, layout, config.scope)?;
        offload::require_unchanged(versions)?;
        let page_bytes: usize = planes.iter().map(|p| p.1).sum();
        if config.ram_bytes < (page_bytes * 3 * 3 + (128 << 10)) as u64 {
            return Err(MetalError::Memory(
                "KV offload RAM/transfer budget cannot hold three complete attention pages".into(),
            ));
        }
        let cache = ColdCache::open_calibrated(
            ColdConfig {
                ram_bytes: config.ram_bytes,
                disk: config.disk.map(|(p, q)| (NvmeStore::dir_for(&p, &ns), q)),
            },
            page_bytes * context.div_ceil(BLOCK_TOKENS),
            &device.calibration_device(),
        )
        .map_err(MetalError::Model)?;
        let mut cost = CostModel::new();
        cost.seed_nvme(cache.device_read_gbs);
        let measured = cache.calibration().is_some();
        if let Some(c) = cache.calibration() {
            cost.adopt_calibration(c);
        }
        Ok(Self {
            scatter: offload::ScatterWorker::new()?,
            restoring: None,
            cache,
            root: ns.root(),
            cost,
            measured,
            decisions: Default::default(),
            reads: Default::default(),
            context,
        })
    }
    fn keys(&self, tokens: &[u32]) -> Vec<(Key, usize)> {
        let mut key = self.root;
        tokens
            .chunks_exact(BLOCK_TOKENS)
            .enumerate()
            .filter_map(|(i, t)| {
                key = key.child(t);
                let n = (i + 1) * BLOCK_TOKENS;
                (n >= 3 * BLOCK_TOKENS && n < tokens.len()).then_some((key.0, n))
            })
            .collect()
    }
    pub(crate) fn capture(
        &mut self,
        device: &MetalDevice,
        planes: &[Plane<'_>],
        tokens: &[u32],
        blocks: &[u32],
    ) {
        self.cache.pump();
        let Some((key, n)) = self.keys(tokens).pop() else {
            return;
        };
        if self.cache.contains(&key) || blocks.len() < n / BLOCK_TOKENS {
            return;
        }
        let spans = spans(planes, &blocks[..n / BLOCK_TOKENS]);
        let bytes = spans.iter().map(|s| s.2).sum();
        let Some(reservation) = self.cache.reserve(bytes) else {
            return;
        };
        match offload::capture(device, &spans, reservation) {
            Ok(payload) => self.cache.put(key, payload),
            Err(error) => {
                tracing::warn!(%error, "Metal paged KV capture refused; recompute remains available")
            }
        }
    }
    pub(crate) fn pump(&mut self, pool: &mut KvPool, radix: &mut PagedRadix) {
        self.cache.pump();
        self.reads
            .retain(|_, (t, _)| t.elapsed() < Duration::from_secs(3));
        let Some(restore) = &self.restoring else {
            return;
        };
        let ok = match restore.done.try_recv() {
            Ok(()) => true,
            Err(std::sync::mpsc::TryRecvError::Empty) => return,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => false,
        };
        let mut r = self
            .restoring
            .take()
            .expect("restore was present when its completion was checked");
        if ok {
            radix.insert(&r.tokens, r.table.blocks(), pool);
            let disk = self.reads.remove(&r.key);
            let elapsed = disk.map_or(r.started, |d| d.0).elapsed().as_secs_f64() * 1e6;
            self.cost.observe_restore_from(
                r.bytes as u64,
                elapsed,
                disk.map_or(elapsed, |d| d.1),
                disk.is_some(),
            );
            self.cache.observe_cost(&self.cost);
            self.decisions.resolved_ok += 1;
            self.decisions.useful_bytes += r.bytes as u64;
            self.decisions.moved_bytes += r.bytes as u64;
            if disk.is_some() {
                self.decisions.served_from_nvme += 1;
            } else {
                self.decisions.served_from_ram += 1;
            }
        } else {
            tracing::warn!("Metal paged KV copy failed; unpublished pages discarded");
        }
        r.table.clear(pool);
    }
    pub(crate) fn loading(
        &mut self,
        tokens: &[u32],
        planes: &[Plane<'_>],
        pool: &mut KvPool,
        radix: &mut PagedRadix,
    ) -> bool {
        self.pump(pool, radix);
        if tokens.is_empty() || tokens.len() > self.context {
            return false;
        }
        let resident = radix.match_prefix(tokens).len() * BLOCK_TOKENS;
        if self.restoring.as_ref().is_some_and(|r| {
            r.tokens.len() > resident
                && r.tokens.len() < tokens.len()
                && tokens.starts_with(&r.tokens)
        }) {
            return true;
        }
        self.decisions.lookups += 1;
        let hit = self
            .keys(tokens)
            .into_iter()
            .rev()
            .find(|(k, n)| *n > resident && self.cache.contains(k));
        let Some((key, n)) = hit else {
            self.decisions.miss_cold += 1;
            return false;
        };
        self.decisions.hits += 1;
        let Some((bytes, disk)) = self.cache.hit_size(&key) else {
            return false;
        };
        let transfer = bytes as u64 + if disk { self.cache.queued_bytes() } else { 0 };
        let election = self.cost.elect(HitShape {
            restore_bytes: transfer,
            restore_tokens: (n - resident) as u32,
            queued_bytes: 0,
            nvme_bytes: if disk { transfer } else { 0 },
        });
        if !self.reads.contains_key(&key) && self.measured && !election.is_restore() {
            self.decisions.elected_recompute += 1;
            return false;
        }
        if disk && self.cache.writes_pending() && !self.reads.contains_key(&key) {
            self.decisions.park_refused += 1;
            return false;
        }
        let (estimate, recompute) = match election {
            Election::Restore {
                est_us,
                recompute_us,
            } => (est_us, recompute_us),
            Election::Recompute { est_us, restore_us } => (restore_us, est_us),
        };
        let budget = (estimate * 1.5 + 50_000.)
            .min(if self.measured { recompute } else { 2_000_000. })
            .clamp(50_000., 2_000_000.);
        if self
            .cache
            .load_until(key, Duration::from_secs_f64(budget / 1e6))
        {
            if let std::collections::hash_map::Entry::Vacant(e) = self.reads.entry(key) {
                e.insert((Instant::now(), estimate));
                self.decisions.elected_restore += 1;
                self.decisions.parked += 1;
            }
            return true;
        }
        let Some(payload) = self.cache.get(&key) else {
            return false;
        };
        // c=4 reads can overlap the one physical copy without reserving another
        // context. A miss/recompute is not delayed by this lane.
        if self.restoring.is_some() {
            return true;
        }
        while pool.free_blocks() < n / BLOCK_TOKENS {
            if radix.evict_lru(pool).is_none() {
                return false;
            }
        }
        let mut table = BlockTable::new();
        if table.ensure(n - 1, pool).is_err() {
            table.clear(pool);
            return false;
        }
        let dst = spans(planes, table.blocks());
        let bytes = payload.bytes().len();
        match self.scatter.submit(&dst, payload) {
            Ok(done) => {
                self.restoring = Some(Restore {
                    table,
                    tokens: tokens[..n].to_vec(),
                    key,
                    bytes,
                    started: Instant::now(),
                    done,
                });
                true
            }
            Err(error) => {
                table.clear(pool);
                tracing::warn!(%error, "Metal paged restore refused");
                false
            }
        }
    }
    pub(crate) fn observe(&mut self, n: u32, us: f64) {
        if n > 0 && us.is_finite() && us > 0. {
            self.cost.observe_prefill(n, us);
            self.measured = true;
            self.cache.observe_cost(&self.cost);
        }
    }
    pub(crate) fn stats(&self) -> TierStats {
        let mut s = self.cache.stats();
        s.open_tickets += u64::from(self.restoring.is_some());
        s
    }
    pub(crate) fn report(&self) -> TierReport {
        let s = self.stats();
        let (ram, nvme) = self.cost.rates_bpus();
        let (ram_capacity, disk_capacity) = self.cache.capacity_bytes();
        TierReport {
            decisions: self.decisions,
            t1_ready_bytes: s.ready_bytes,
            t1_reserved_bytes: self.cache.allocated_bytes().saturating_sub(s.ready_bytes),
            t1_capacity_bytes: ram_capacity,
            t2_capacity_bytes: disk_capacity,
            t2_ready_bytes: self.cache.disk_bytes(),
            resident_runs: s.resident_runs,
            open_tickets: s.open_tickets,
            in_flight_demotes: s.in_flight_demotes,
            tripped: s.tripped,
            io_failures: s.io_failures,
            integrity_failures: s.integrity_failures,
            evictions: s.evictions,
            rate_ram_bpus: ram,
            rate_nvme_bpus: nvme,
            device_read_gbs: self.cache.device_read_gbs,
            prediction_error_pct: self.cost.prediction_error_pct(),
            t2_written_day_bytes: s.t2_written_day_bytes,
            t2_day_budget_bytes: s.t2_day_budget_bytes,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_metal::MTLBuffer;
    #[test]
    fn restart_prices_first_hit_before_prefill_and_relearns_from_live_measurement() {
        let dir = std::env::temp_dir().join(format!(
            "paddock-paged-cost-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&dir).unwrap();
        let source = dir.join("identity");
        std::fs::write(&source, b"test-checkpoint").unwrap();
        let device = MetalDevice::new(Some(64 << 20)).unwrap();
        let buffer = device.upload(&vec![29u8; 16 * 256]).unwrap();
        let planes = [(&buffer, 256)];
        let config = crate::KvOffloadConfig {
            ram_bytes: 4 << 20,
            disk: Some((dir.join("disk"), 8 << 20)),
            scope: b"cost-test".to_vec(),
        };
        let versions = offload::versions(&source).unwrap();
        let open = || {
            PagedTier::open(
                &device,
                config.clone(),
                &[&source],
                b"one-plane",
                &versions,
                &planes,
                128,
            )
            .unwrap()
        };
        let mut tier = open();
        let tokens: Vec<u32> = (0..65).collect();
        tier.observe(64, 1.0); // Deliberately cheap recompute: a read should lose.
        tier.capture(&device, &planes, &tokens, &[0, 1, 2, 3]);
        drop(tier); // Joins the worker and preserves measured costs and payload.
        let mut tier = open();
        assert!(
            tier.measured,
            "a device-matched seed can price the first hit"
        );
        assert!(tier.cost.calibration().is_none(), "not measured live yet");
        let mut pool = KvPool::with_blocks(16);
        let mut radix = PagedRadix::new();
        assert!(!tier.loading(&tokens, &planes, &mut pool, &mut radix));
        assert_eq!(tier.decisions.elected_recompute, 1);
        assert!(tier.restoring.is_none());
        assert_eq!(
            tier.cache.stats().open_tickets,
            0,
            "no speculative disk I/O"
        );
        tier.observe(64, 1_000_000.);
        assert_eq!(
            tier.cost.calibration().unwrap().prefill_tpus,
            64. / 1_000_000.,
            "first live sample replaces, not smooths, the seed"
        );
        let start = Instant::now();
        while tier.loading(&tokens, &planes, &mut pool, &mut radix) {
            assert!(start.elapsed() < Duration::from_secs(30));
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(radix.match_prefix(&tokens).len(), 4);
        assert_eq!(tier.decisions.served_from_nvme, 1);
        assert_eq!(tier.stats().integrity_failures, 0);
        drop(tier);
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn paged_restore_owns_pages_until_fenced_even_without_request() {
        let dir = std::env::temp_dir().join(format!(
            "paddock-paged-owner-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&dir).unwrap();
        let source = dir.join("identity");
        std::fs::write(&source, b"test-checkpoint").unwrap();
        let device = MetalDevice::new(Some(64 << 20)).unwrap();
        let a = device.upload(&vec![27u8; 16 * 256]).unwrap();
        let b = device.upload(&vec![42u8; 16 * 256]).unwrap();
        let planes = [(&a, 256), (&b, 256)];
        let mut tier = PagedTier::open(
            &device,
            crate::KvOffloadConfig {
                ram_bytes: 4 << 20,
                disk: None,
                scope: b"test".to_vec(),
            },
            &[&source],
            b"two-plane-layout",
            &offload::versions(&source).unwrap(),
            &planes,
            128,
        )
        .unwrap();
        let prompt: Vec<u32> = (1..66).collect();
        let mut pool = KvPool::with_blocks(16);
        let mut table = BlockTable::new();
        table.ensure(64, &mut pool).unwrap();
        tier.capture(&device, &planes, &prompt, table.blocks());
        table.clear(&mut pool);
        let mut radix = PagedRadix::new();
        assert!(tier.loading(&prompt, &planes, &mut pool, &mut radix));
        assert!(
            radix.match_prefix(&prompt).is_empty(),
            "not published at submit"
        );
        assert_eq!(pool.free_blocks(), 12, "four destination pages retained");
        // No request owns this restore now. Recycled request pressure must
        // never acquire its pages before copy completion/publication.
        let reserved = tier.restoring.as_ref().unwrap().table.blocks().to_vec();
        let mut active = BlockTable::new();
        active.ensure(12 * 16 - 1, &mut pool).unwrap();
        assert!(active.blocks().iter().all(|p| !reserved.contains(p)));
        let start = Instant::now();
        while tier.restoring.is_some() {
            assert!(start.elapsed() < Duration::from_secs(5));
            tier.pump(&mut pool, &mut radix);
            std::thread::yield_now();
        }
        let blocks = radix.match_prefix(&prompt);
        assert_eq!(blocks.len(), 4);
        for &(buffer, value) in &[(&a, 27u8), (&b, 42u8)] {
            for &page in &blocks {
                // SAFETY: acquired worker completion; no GPU writes in flight.
                let got = unsafe {
                    std::slice::from_raw_parts(
                        buffer
                            .raw
                            .contents()
                            .as_ptr()
                            .cast::<u8>()
                            .add(page as usize * 256),
                        256,
                    )
                };
                assert!(got.iter().all(|b| *b == value));
            }
        }
        active.clear(&mut pool);
        while radix.evict_lru(&mut pool).is_some() {}
        assert_eq!(pool.free_blocks(), 16);
        assert_eq!(tier.report().decisions.served_from_ram, 1);
        drop(tier);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
