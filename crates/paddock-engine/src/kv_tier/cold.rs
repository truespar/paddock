//! Bounded, asynchronous immutable checkpoint cache for unified-memory devices.
//! No file I/O, checksum scans, or channel waits occur on the inference thread.
//! RAM includes queued jobs and completed-but-unconsumed reads, not only LRU entries.
//! Device families own the resume schema and publish only after complete validation.
use super::{
    TierStats,
    cost::{Calibration, CostModel},
    nvme_store::{NvmeStore, StoreError},
};
mod calibration;
use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
        mpsc,
    },
    thread,
};

pub type Key = [u8; 32];
pub trait Bytes: AsRef<[u8]> + Send + Sync {}
impl<T: AsRef<[u8]> + Send + Sync> Bytes for T {}

#[derive(Clone, Debug)]
pub struct ColdConfig {
    /// Total cold allocations AND transient transfer copies in unified RAM.
    pub ram_bytes: u64,
    pub disk: Option<(PathBuf, u64)>,
}

/// Reservation outlives every queue/worker/consumer reference to its payload.
pub struct Reservation {
    ledger: Arc<AtomicU64>,
    bytes: u64,
}
impl Drop for Reservation {
    fn drop(&mut self) {
        self.ledger.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}
pub struct Payload {
    data: Box<dyn Bytes>,
    _reservation: Reservation,
}
impl Payload {
    pub fn bytes(&self) -> &[u8] {
        self.data.as_ref().as_ref()
    }
    pub fn new(data: impl Bytes + 'static, reservation: Reservation) -> Self {
        assert!(data.as_ref().len() as u64 <= reservation.bytes / 3);
        Self {
            data: Box::new(data),
            _reservation: reservation,
        }
    }
}
enum Job {
    Store(Key, Arc<Payload>),
    Load(Key, Reservation),
    Calibrate(Calibration),
}
enum Done {
    /// (key, len, landed, failed on the device - capacity refusals are
    /// neither, evicted keys)
    Stored(Key, usize, bool, bool, Vec<Key>),
    Loaded(Key, Option<Arc<Payload>>, bool),
    Calibrated(Calibration, bool),
}
struct Hot {
    payload: Arc<Payload>,
    touched: u64,
}

pub struct ColdCache {
    tx: Option<mpsc::SyncSender<Job>>,
    rx: mpsc::Receiver<Done>,
    worker: Option<thread::JoinHandle<()>>,
    hot: HashMap<Key, Hot>,
    disk: HashMap<Key, usize>,
    writing: HashMap<Key, ()>,
    loading: HashMap<Key, std::time::Instant>,
    ledger: Arc<AtomicU64>,
    budget: u64,
    max_payload: usize,
    clock: u64,
    stats: TierStats,
    pub device_read_gbs: f64,
    disk_quota: u64,
    day: u64,
    calibration: Option<calibration::State>,
}
impl ColdCache {
    /// Startup only. Opening/recovery failure is explicit, never silently disabled.
    pub fn open(config: ColdConfig, max_payload: usize) -> Result<Self, String> {
        Self::open_inner(config, max_payload, None)
    }
    /// Reuse cost measurements only for this device and this locked cache
    /// namespace. The first new live prefill still replaces the restored seed.
    pub fn open_calibrated(
        config: ColdConfig,
        max_payload: usize,
        device: &str,
    ) -> Result<Self, String> {
        Self::open_inner(config, max_payload, Some(device))
    }
    fn open_inner(
        config: ColdConfig,
        max_payload: usize,
        device: Option<&str>,
    ) -> Result<Self, String> {
        if config.ram_bytes == 0 || max_payload == 0 {
            return Err("KV offload requires a nonzero unified-RAM transfer budget".into());
        }
        let (tx, jobs) = mpsc::sync_channel(8);
        let (done, rx) = mpsc::channel();
        let mut disk_index = HashMap::new();
        let mut device_read_gbs = 0.0;
        let mut day_budget = 0;
        let disk_quota = config.disk.as_ref().map_or(0, |(_, q)| *q);
        let mut calibration = None;
        let store = config
            .disk
            .map(|(dir, quota)| {
                let (store, recovery) = NvmeStore::open(&dir, quota).map_err(|e| e.to_string())?;
                calibration = device.map(|device| calibration::State::load(store.dir(), device));
                if let Some(c) = calibration.as_ref().and_then(|state| state.seed) {
                    tracing::info!(
                        prefill_tok_s = (c.prefill_tpus * 1e6).round(),
                        "Metal KV cost model starts from this device's last calibration"
                    );
                }
                tracing::info!(
                    recovered = recovery.recovered_entries,
                    day_budget_gib = store.day_budget() >> 30,
                    "Metal KV cache recovered"
                );
                device_read_gbs = store.device().read_gbs;
                day_budget = store.day_budget();
                for (key, _, _, len, _) in store.live_iter() {
                    if len <= max_payload as u64 {
                        disk_index.insert(*key, len as usize);
                    }
                }
                Ok::<_, String>(store)
            })
            .transpose()?;
        let worker = if let Some(mut store) = store {
            let device = calibration.as_ref().map(|state| state.device.clone());
            Some(thread::Builder::new().name("paddock-kv-ssd".into()).spawn(move || {
                while let Ok(job) = jobs.recv() {
                    let result = match job {
                        Job::Store(key, payload) => {
                            let bytes = payload.bytes();
                            let mut evicted = Vec::new();
                            let mut result = store.store(key, 1, 1, bytes);
                            if matches!(result, Err(StoreError::QuotaExhausted { .. } | StoreError::VolumeFull { .. })) {
                                match store.make_room(bytes.len() as u64) {
                                    Ok(keys) => {
                                        evicted = keys;
                                        result = store.store(key, 1, 1, bytes);
                                    }
                                    Err(error) => tracing::warn!(%error, "Metal KV SSD eviction failed"),
                                }
                            }
                            // the store logs pause/resume and a volume at its
                            // reserve once each; per write this is detail
                            if let Err(error) = &result { tracing::debug!(%error, "Metal KV SSD write declined; RAM cache remains usable"); }
                            let failed = result.as_ref().is_err_and(|e| !e.is_capacity());
                            Done::Stored(key, bytes.len(), result.is_ok(), failed, evicted)
                        }
                        Job::Load(key, reservation) => {
                            match store.read_owned(&key) {
                                Ok((_, bytes)) if bytes.as_ref().len() as u64 <= reservation.bytes / 3 => {
                                    Done::Loaded(key, Some(Arc::new(Payload::new(bytes, reservation))), false)
                                }
                                Err(StoreError::Integrity) => Done::Loaded(key, None, true),
                                _ => Done::Loaded(key, None, false),
                            }
                        }
                        Job::Calibrate(c) => {
                            let saved = device.as_ref().is_some_and(|device| {
                                if let Err(error) = calibration::save(store.dir(), device, c) {
                                    tracing::debug!(%error, "Metal KV cost calibration not saved; cache remains usable");
                                    false
                                } else {
                                    true
                                }
                            });
                            Done::Calibrated(c, saved)
                        }
                    };
                    if done.send(result).is_err() { break; }
                }
            }).map_err(|e| e.to_string())?)
        } else {
            None
        };
        let armed = worker.is_some();
        Ok(Self {
            tx: armed.then_some(tx),
            rx,
            worker,
            hot: HashMap::new(),
            disk: disk_index,
            writing: HashMap::new(),
            loading: HashMap::new(),
            ledger: Arc::new(AtomicU64::new(0)),
            budget: config.ram_bytes,
            max_payload,
            clock: 0,
            stats: TierStats {
                t2_day_budget_bytes: day_budget,
                ..TierStats::default()
            },
            device_read_gbs,
            disk_quota,
            day: utc_day(),
            calibration,
        })
    }
    pub fn pump(&mut self) {
        if self.day != utc_day() {
            self.day = utc_day();
            self.stats.t2_written_day_bytes = 0;
        }
        loop {
            let done = match self.rx.try_recv() {
                Ok(done) => done,
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    if self.tx.take().is_some() {
                        self.stats.tripped = true;
                        self.disk.clear();
                        self.loading.clear();
                        self.writing.clear();
                        tracing::warn!("Metal KV SSD worker stopped; requests will recompute");
                    }
                    break;
                }
            };
            match done {
                Done::Stored(key, len, ok, failed, evicted) => {
                    self.writing.remove(&key);
                    for key in evicted {
                        self.disk.remove(&key);
                        self.stats.evictions += 1;
                    }
                    if ok {
                        self.disk.insert(key, len);
                        self.stats.t2_written_day_bytes += len as u64;
                    } else if failed {
                        self.stats.io_failures += 1;
                    }
                }
                Done::Loaded(key, payload, integrity) => {
                    self.loading.remove(&key);
                    if let Some(payload) = payload {
                        self.insert_hot(key, payload);
                    } else {
                        self.disk.remove(&key);
                        if integrity {
                            self.stats.integrity_failures += 1;
                        } else {
                            self.stats.io_failures += 1;
                        }
                    }
                }
                Done::Calibrated(c, saved) => {
                    if let Some(state) = &mut self.calibration {
                        state.complete(c, saved);
                    }
                }
            }
        }
        self.save_calibration();
    }
    /// Startup-only seed, never fabricated from unmeasured defaults.
    pub fn calibration(&self) -> Option<Calibration> {
        self.calibration.as_ref().and_then(|state| state.seed)
    }
    /// Coalesces to one latest value and at most one queued write. Saturation
    /// retries on a later pump; no file operation or channel wait here.
    pub fn observe_cost(&mut self, cost: &CostModel) {
        if let Some(state) = &mut self.calibration
            && let Some(c) = cost.calibration()
        {
            state.observe(c);
        }
        self.save_calibration();
    }
    fn save_calibration(&mut self) {
        let now = std::time::Instant::now();
        if let (Some(state), Some(tx)) = (&mut self.calibration, &self.tx)
            && let Some(c) = state.next(now)
            && tx.try_send(Job::Calibrate(c)).is_ok()
        {
            state.submitted(now, c);
        }
    }
    fn insert_hot(&mut self, key: Key, payload: Arc<Payload>) {
        self.clock += 1;
        self.hot.insert(
            key,
            Hot {
                payload,
                touched: self.clock,
            },
        );
    }
    /// Conservative 3x charge covers immutable payload + aligned store/read bounce
    /// and the read result. Padding has a separate 128 KiB allowance. In-flight
    /// jobs cannot be evicted from the accounting by dropping an LRU reference.
    pub fn reserve(&mut self, len: usize) -> Option<Reservation> {
        if len > self.max_payload {
            return None;
        }
        let bytes = (len as u64).checked_mul(3)?.checked_add(128 << 10)?;
        loop {
            let used = self.ledger.load(Ordering::Acquire);
            if used.checked_add(bytes)? <= self.budget {
                self.ledger.fetch_add(bytes, Ordering::AcqRel);
                return Some(Reservation {
                    ledger: self.ledger.clone(),
                    bytes,
                });
            }
            let key = self
                .hot
                .iter()
                .filter(|(k, _)| !self.writing.contains_key(*k) && !self.loading.contains_key(*k))
                .min_by_key(|(_, h)| h.touched)
                .map(|(k, _)| *k)?;
            self.hot.remove(&key);
            self.stats.evictions += 1;
        }
    }
    pub fn contains(&self, key: &Key) -> bool {
        self.hot.contains_key(key) || self.disk.contains_key(key) || self.writing.contains_key(key)
    }
    pub fn put(&mut self, key: Key, payload: Payload) {
        if self.contains(&key) {
            return;
        }
        let payload = Arc::new(payload);
        // Same elected daily ceiling as CUDA (a quarter of the volume, at most
        // 1 TiB - `NvmeStore::day_budget`). Stops new durable writes, not
        // inference or RAM caching; this counter resets on startup.
        if let Some(tx) = &self.tx
            && self.stats.t2_written_day_bytes + self.queued_bytes() + payload.bytes().len() as u64
                <= self.stats.t2_day_budget_bytes
        {
            match tx.try_send(Job::Store(key, payload.clone())) {
                Ok(()) => {
                    self.writing.insert(key, ());
                }
                Err(mpsc::TrySendError::Full(_)) => {
                    tracing::debug!("Metal KV durable queue full; retaining RAM checkpoint")
                }
                Err(mpsc::TrySendError::Disconnected(_)) => {
                    self.stats.tripped = true;
                }
            }
        }
        self.insert_hot(key, payload);
    }
    pub fn get(&mut self, key: &Key) -> Option<Arc<Payload>> {
        self.clock += 1;
        let hot = self.hot.get_mut(key)?;
        hot.touched = self.clock;
        Some(hot.payload.clone())
    }
    /// True means parked. Queue saturation/insufficient budget elect recompute;
    /// neither is a reason to block another request or grow a staging queue.
    pub fn load(&mut self, key: Key) -> bool {
        self.load_until(key, std::time::Duration::from_millis(500))
    }
    pub fn load_until(&mut self, key: Key, budget: std::time::Duration) -> bool {
        self.pump();
        if self.hot.contains_key(&key) {
            return false;
        }
        if let Some(deadline) = self.loading.get(&key) {
            // A stalled drive cannot indefinitely hold admission. The immutable
            // late result may populate the cache, never an abandoned slot.
            return std::time::Instant::now() < *deadline;
        }
        let Some(&len) = self.disk.get(&key) else {
            return false;
        };
        let Some(reservation) = self.reserve(len) else {
            return false;
        };
        if self
            .tx
            .as_ref()
            .is_some_and(|tx| tx.try_send(Job::Load(key, reservation)).is_ok())
        {
            self.loading.insert(
                key,
                std::time::Instant::now() + budget.min(std::time::Duration::from_secs(2)),
            );
            true
        } else {
            false
        }
    }
    pub fn stats(&self) -> TierStats {
        TierStats {
            resident_runs: self.hot.len() as u64,
            ready_bytes: self
                .hot
                .values()
                .map(|h| h.payload.bytes().len() as u64)
                .sum(),
            in_flight_demotes: self.writing.len() as u64,
            open_tickets: self.loading.len() as u64,
            ..self.stats
        }
    }
    pub fn allocated_bytes(&self) -> u64 {
        self.ledger.load(Ordering::Acquire)
    }
    pub fn capacity_bytes(&self) -> (u64, u64) {
        (self.budget, self.disk_quota)
    }
    pub fn disk_bytes(&self) -> u64 {
        self.disk.values().map(|n| *n as u64).sum()
    }
    pub fn hit_size(&self, key: &Key) -> Option<(usize, bool)> {
        self.hot
            .get(key)
            .map(|h| (h.payload.bytes().len(), false))
            .or_else(|| self.disk.get(key).map(|&len| (len, true)))
    }
    pub fn queued_bytes(&self) -> u64 {
        self.writing
            .keys()
            .filter_map(|k| self.hot.get(k))
            .map(|h| h.payload.bytes().len() as u64)
            .sum::<u64>()
            + self
                .loading
                .keys()
                .filter_map(|k| self.disk.get(k))
                .map(|n| *n as u64)
                .sum::<u64>()
    }
    pub fn writes_pending(&self) -> bool {
        !self.writing.is_empty()
    }
}
impl Drop for ColdCache {
    fn drop(&mut self) {
        // Shutdown already joins this worker. Flush the last coalesced value
        // without relaxing the write-rate bound during inference.
        if let (Some(state), Some(tx)) = (&self.calibration, &self.tx)
            && let Some(c) = state.shutdown_value()
        {
            let _ = tx.send(Job::Calibrate(c));
        }
        self.tx.take();
        // Only model shutdown waits; normal scheduler methods never join/wait.
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}
fn utc_day() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() / 86400)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn temp() -> PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "paddock-cold-test-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&p).unwrap();
        p
    }
    fn wait(cache: &mut ColdCache) {
        let start = std::time::Instant::now();
        while cache.stats().in_flight_demotes != 0 || cache.stats().open_tickets != 0 {
            assert!(start.elapsed().as_secs() < 10);
            cache.pump();
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }
    #[test]
    fn reservations_survive_eviction_and_all_consumer_references() {
        let mut c = ColdCache::open(
            ColdConfig {
                ram_bytes: 2 << 20,
                disk: None,
            },
            1 << 20,
        )
        .unwrap();
        let r = c.reserve(400_000).unwrap();
        c.put([1; 32], Payload::new(vec![7u8; 400_000], r));
        let pinned = c.get(&[1; 32]).unwrap();
        assert!(c.reserve(400_000).is_none());
        assert!(c.allocated_bytes() > 1_200_000);
        drop(pinned);
        assert_eq!(c.allocated_bytes(), 0);
        assert!(c.reserve(400_000).is_some());
    }
    #[test]
    fn durable_restart_single_flight_and_unix_lock() {
        let dir = temp();
        let cfg = ColdConfig {
            ram_bytes: 8 << 20,
            disk: Some((dir.clone(), 16 << 20)),
        };
        let mut c = ColdCache::open(cfg.clone(), 1 << 20).unwrap();
        assert!(ColdCache::open(cfg.clone(), 1 << 20).is_err());
        let r = c.reserve(100_000).unwrap();
        c.put([2; 32], Payload::new(vec![19u8; 100_000], r));
        wait(&mut c);
        assert_eq!(c.stats().t2_written_day_bytes, 100_000);
        drop(c);
        let mut c = ColdCache::open(cfg, 1 << 20).unwrap();
        assert!(c.contains(&[2; 32]));
        assert!(c.load([2; 32]));
        c.load([2; 32]);
        assert!(c.stats().open_tickets <= 1);
        wait(&mut c);
        assert_eq!(c.get(&[2; 32]).unwrap().bytes(), &[19u8; 100_000]);
        assert!(c.get(&[3; 32]).is_none());
        drop(c);
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn corrupt_payload_is_a_miss_not_partially_ready() {
        use std::io::{Seek, Write};
        let dir = temp();
        let cfg = ColdConfig {
            ram_bytes: 8 << 20,
            disk: Some((dir.clone(), 16 << 20)),
        };
        let mut c = ColdCache::open(cfg.clone(), 1 << 20).unwrap();
        let r = c.reserve(4096).unwrap();
        c.put([4; 32], Payload::new(vec![31u8; 4096], r));
        wait(&mut c);
        drop(c);
        let segment = std::fs::read_dir(dir.join("segments"))
            .unwrap()
            .map(|e| e.unwrap().path())
            .find(|p| p.extension().is_some_and(|e| e == "seg"))
            .unwrap();
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .open(segment)
            .unwrap();
        f.seek(std::io::SeekFrom::Start(0)).unwrap();
        f.write_all(&[99]).unwrap();
        f.sync_all().unwrap();
        drop(f);
        let mut c = ColdCache::open(cfg, 1 << 20).unwrap();
        assert!(c.load([4; 32]));
        wait(&mut c);
        assert!(!c.contains(&[4; 32]));
        assert!(c.get(&[4; 32]).is_none());
        assert_eq!(c.stats().integrity_failures, 1);
        drop(c);
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn sub_segment_disk_quota_evicts_and_keeps_accepting() {
        let dir = temp();
        let (mut s, _) = NvmeStore::open(&dir, 8192).unwrap();
        s.store([1; 32], 1, 1, &[1u8; 8192]).unwrap();
        assert_eq!(s.make_room(4096).unwrap(), vec![[1; 32]]);
        s.store([2; 32], 1, 1, &[2u8; 4096]).unwrap();
        assert_eq!(s.read(&[2; 32]).unwrap().1, &[2u8; 4096]);
        assert!(s.stats().live_bytes <= 8192);
        drop(s);
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn oversized_recovered_records_never_allocate() {
        let dir = temp();
        let (mut s, _) = NvmeStore::open(&dir, 1 << 20).unwrap();
        s.store([9; 32], 1, 1, &[0u8; 4096]).unwrap();
        drop(s);
        let mut c = ColdCache::open(
            ColdConfig {
                ram_bytes: 1 << 20,
                disk: Some((dir.clone(), 1 << 20)),
            },
            1024,
        )
        .unwrap();
        assert!(!c.load([9; 32]));
        assert_eq!(c.allocated_bytes(), 0);
        drop(c);
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn expired_read_does_not_rearm_when_the_same_slot_polls_again() {
        let mut c = ColdCache::open(
            ColdConfig {
                ram_bytes: 1 << 20,
                disk: None,
            },
            1024,
        )
        .unwrap();
        let deadline = std::time::Instant::now() - std::time::Duration::from_secs(1);
        c.loading.insert([6; 32], deadline);
        assert!(!c.load_until([6; 32], std::time::Duration::from_secs(2)));
        assert_eq!(c.loading[&[6; 32]], deadline);
        let r = c.reserve(32).unwrap();
        c.insert_hot([6; 32], Arc::new(Payload::new(vec![16u8; 32], r)));
        assert!(!c.load([6; 32]));
        assert_eq!(c.get(&[6; 32]).unwrap().bytes(), &[16u8; 32]);
    }
}
