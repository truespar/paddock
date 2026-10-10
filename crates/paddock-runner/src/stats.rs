//! Engine self-report sampler - the runner's **inside view** (doc §9 layer 2).
//!
//! Reads the engine's lock-free counters (tok/s, phase, batch, KV pool,
//! allocator-ledger VRAM split) on a dedicated low-cadence thread and publishes
//! snapshots over a watch channel. Metal adds process-local device allocation
//! and completed-command counters on this thread. **No NVML and no CUDA**: device
//! telemetry (temps, power, per-PID memory from outside) belongs to the manager
//! - the inside and outside views must come from different processes or the
//!   reconciliation cross-check is worthless. Reading the atomics never perturbs
//!   the decode loop.

use std::sync::Arc;
use std::sync::atomic::Ordering::Relaxed;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use paddock_engine::metrics::EngineMetrics;
use serde::Serialize;
use tokio::sync::watch;

/// Live engine counters (what the GPU is doing for the current requests).
/// Present only when a generative model is loaded.
#[derive(Debug, Clone, Serialize)]
pub struct EngineSnapshot {
    /// Output tokens/sec, derived from the counter delta between samples.
    pub tok_s: f64,
    /// "idle" | "prefill" | "decode".
    pub phase: &'static str,
    /// Sequences in flight (batch width; 1 on the serial engine).
    pub active_slots: u32,
    /// Paged-KV blocks in use / total (0 when the backend has no pool).
    pub kv_used: u32,
    pub kv_total: u32,
    /// Cumulative output tokens since server start.
    pub tokens_total: u64,
    /// Prefix-cache reuse: cumulative prompt tokens served from cache / total
    /// prompt tokens prefilled (user-facing prefills only). The tier-1 agentic
    /// signal the unique-salt gensweep hides - high on shared-prefix workloads,
    /// ~0 with no reuse.
    pub prefix_hit_rate: f64,
    pub prefill_tokens_total: u64,
    pub prefill_tokens_cached: u64,
    /// Measured VRAM the loaded model holds (weights + KV/state pools), bytes,
    /// from the allocator's own bookkeeping - the self-attribution that stays
    /// meaningful on a bare runner where no NVML view exists.
    pub model_mem: Option<u64>,
    /// Memory breakdown (bytes), where the model family reports it: weight
    /// planes (all serving classes), KV cache, and the derived remainder
    /// (scratch/graphs/other = model_mem - weights - kv). None = family
    /// doesn't itemize that class yet. This is the honest what's-on-the-GPU
    /// split; NVML can't see class boundaries.
    pub weights_mem: Option<u64>,
    pub kv_mem: Option<u64>,
    pub scratch_mem: Option<u64>,
    /// The prefix cache's off-GPU tiers, when the model has them armed -
    /// what the cache decided and why (plan D8). None = no tier.
    pub cache_tier: Option<TierSnapshot>,
}

/// The kv-offload tier's decision record, as the Studio panel reads it.
/// Counters are cumulative since the model started; the panel derives rates
/// and shares, because a restart must be visible as a reset rather than
/// smoothed away.
#[derive(Debug, Clone, Serialize)]
pub struct TierSnapshot {
    /// Lookups asked, and how they resolved.
    pub lookups: u64,
    pub hits: u64,
    pub miss_cold: u64,
    pub miss_no_new_tokens: u64,
    pub miss_tripped: u64,
    /// Missed on content this tier evicted - the capacity alarm's input.
    pub miss_ghost: u64,
    /// Missed on KV held without the state a hybrid model resumes from.
    pub miss_no_state: u64,
    pub elected_restore: u64,
    pub elected_recompute: u64,
    pub parked: u64,
    pub park_refused: u64,
    pub resolved: u64,
    pub abandoned: u64,
    pub served_from_ram: u64,
    pub served_from_nvme: u64,
    /// Evictions that kept their content by handing it to disk.
    pub promoted_to_disk: u64,
    /// Bytes delivered to the GPU, and bytes moved to deliver them.
    pub useful_bytes: u64,
    pub moved_bytes: u64,
    /// Occupancy, bytes.
    pub ram_ready: u64,
    pub ram_in_flight: u64,
    pub ram_reserved: u64,
    pub ram_capacity: u64,
    pub disk_ready: u64,
    pub disk_capacity: u64,
    pub resident_runs: u64,
    pub in_flight_demotes: u64,
    pub open_tickets: u64,
    pub pending_durable_writes: u64,
    /// Health.
    pub tripped: bool,
    pub io_failures: u64,
    pub integrity_failures: u64,
    pub evictions: u64,
    pub single_flight_joins: u64,
    pub stale_completions: u64,
    /// Measured restore rates, bytes/us, per tier.
    pub rate_ram_bpus: f64,
    pub rate_disk_bpus: f64,
    /// Mean absolute error of the cost model's own predictions. None until
    /// a restore has completed - never a confident 0%.
    pub prediction_error_pct: Option<f64>,
    /// What the disk actually measured at open (3.2), not a spec sheet.
    pub disk_read_gbs: f64,
    pub disk_write_gbs: f64,
    pub disk_unbuffered: bool,
    pub disk_written_today: u64,
    pub ghost_keys: u64,
}

impl TierSnapshot {
    fn read(t: &paddock_engine::metrics::TierGauges) -> Option<Self> {
        if t.armed.load(Relaxed) != 1 {
            return None;
        }
        let n = |a: &std::sync::atomic::AtomicU64| a.load(Relaxed);
        let f = |a: &std::sync::atomic::AtomicU64| f64::from_bits(a.load(Relaxed));
        Some(Self {
            lookups: n(&t.lookups),
            hits: n(&t.hits),
            miss_cold: n(&t.miss_cold),
            miss_no_new_tokens: n(&t.miss_no_new_tokens),
            miss_tripped: n(&t.miss_tripped),
            miss_ghost: n(&t.miss_ghost),
            miss_no_state: n(&t.miss_no_state),
            elected_restore: n(&t.elected_restore),
            elected_recompute: n(&t.elected_recompute),
            parked: n(&t.parked),
            park_refused: n(&t.park_refused),
            resolved: n(&t.resolved_ok),
            abandoned: n(&t.abandoned),
            served_from_ram: n(&t.served_from_ram),
            served_from_nvme: n(&t.served_from_nvme),
            promoted_to_disk: n(&t.promoted_to_disk),
            useful_bytes: n(&t.useful_bytes),
            moved_bytes: n(&t.moved_bytes),
            ram_ready: n(&t.t1_ready_bytes),
            ram_in_flight: n(&t.t1_in_flight_bytes),
            ram_reserved: n(&t.t1_reserved_bytes),
            ram_capacity: n(&t.t1_capacity_bytes),
            disk_ready: n(&t.t2_ready_bytes),
            disk_capacity: n(&t.t2_capacity_bytes),
            resident_runs: n(&t.resident_runs),
            in_flight_demotes: n(&t.in_flight_demotes),
            open_tickets: n(&t.open_tickets),
            pending_durable_writes: n(&t.pending_durable_writes),
            tripped: t.tripped.load(Relaxed) == 1,
            io_failures: n(&t.io_failures),
            integrity_failures: n(&t.integrity_failures),
            evictions: n(&t.evictions),
            single_flight_joins: n(&t.single_flight_joins),
            stale_completions: n(&t.stale_completions),
            rate_ram_bpus: f(&t.rate_ram_bpus),
            rate_disk_bpus: f(&t.rate_nvme_bpus),
            prediction_error_pct: (t.has_prediction_error.load(Relaxed) == 1)
                .then(|| f(&t.prediction_error_pct)),
            disk_read_gbs: f(&t.device_read_gbs),
            disk_write_gbs: f(&t.device_write_gbs),
            disk_unbuffered: t.device_unbuffered.load(Relaxed) == 1,
            disk_written_today: n(&t.t2_written_day_bytes),
            ghost_keys: n(&t.ghost_keys),
        })
    }
}

/// One published sample. `engine` is None when no generative model is loaded
/// (encoder-only runners still get timestamps for liveness).
#[derive(Debug, Clone, Serialize)]
pub struct StatsSnapshot {
    /// Unix seconds when sampled.
    pub ts: u64,
    pub pid: u32,
    pub engine: Option<EngineSnapshot>,
    /// Process-local Metal allocations and command timing, including non-LLMs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metal: Option<serde_json::Value>,
}

impl StatsSnapshot {
    fn empty() -> Self {
        Self {
            ts: now_secs(),
            pid: std::process::id(),
            engine: None,
            metal: None,
        }
    }
}

fn phase_str(p: u8) -> &'static str {
    match p {
        1 => "prefill",
        2 => "decode",
        _ => "idle",
    }
}

/// Handle held by `AppState`; hands out the latest snapshot and a stream.
#[derive(Clone)]
pub struct Stats {
    rx: watch::Receiver<Arc<StatsSnapshot>>,
}

impl Stats {
    /// The most recent snapshot (never blocks).
    pub fn latest(&self) -> Arc<StatsSnapshot> {
        self.rx.borrow().clone()
    }

    /// A fresh receiver for a streaming client (WebSocket / the manager's
    /// collector). Each subscriber bumps the sender's receiver count, which
    /// ramps the sample cadence up.
    pub fn subscribe(&self) -> watch::Receiver<Arc<StatsSnapshot>> {
        self.rx.clone()
    }

    /// A disabled handle (tests / no-sampler): a static empty snapshot.
    pub fn disabled() -> Self {
        let (tx, rx) = watch::channel(Arc::new(StatsSnapshot::empty()));
        // Keep the sender alive forever so the receiver stays valid.
        std::mem::forget(tx);
        Stats { rx }
    }
}

/// Spawn the sampler on a dedicated thread and return a handle. `engine` is
/// the generative model's live counters (None when no model is loaded).
pub fn start(engine: Option<Arc<EngineMetrics>>) -> Stats {
    let (tx, rx) = watch::channel(Arc::new(StatsSnapshot::empty()));
    let builder = std::thread::Builder::new().name("engine-stats".to_owned());
    if let Err(e) = builder.spawn(move || run(tx, engine)) {
        tracing::warn!(%e, "could not spawn engine stats thread; self-report disabled");
    }
    Stats { rx }
}

fn metal_snapshot() -> Option<serde_json::Value> {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        paddock_metal::telemetry_snapshot()
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        None
    }
}

fn run(tx: watch::Sender<Arc<StatsSnapshot>>, engine: Option<Arc<EngineMetrics>>) {
    let mut prev_tokens: u64 = 0;
    let mut prev_at = Instant::now();

    loop {
        // Read the lock-free counters and derive tok/s from the token-count
        // delta over the elapsed interval.
        let eng = engine.as_deref().map(|m| {
            let now = Instant::now();
            let dt = now.duration_since(prev_at).as_secs_f64();
            let tokens = m.tokens_generated.load(Relaxed);
            let tok_s = if dt > 0.0 {
                tokens.saturating_sub(prev_tokens) as f64 / dt
            } else {
                0.0
            };
            prev_tokens = tokens;
            prev_at = now;
            let pf_total = m.prefill_tokens_total.load(Relaxed);
            let pf_cached = m.prefill_tokens_cached.load(Relaxed);
            let model_mem = m.model_mem_bytes.load(Relaxed);
            let weights_mem = m.weights_mem_bytes.load(Relaxed);
            let kv_mem = m.kv_mem_bytes.load(Relaxed);
            EngineSnapshot {
                tok_s,
                phase: phase_str(m.phase.load(Relaxed)),
                active_slots: m.active_slots.load(Relaxed),
                kv_used: m.kv_used.load(Relaxed),
                kv_total: m.kv_total.load(Relaxed),
                tokens_total: tokens,
                prefix_hit_rate: if pf_total > 0 {
                    pf_cached as f64 / pf_total as f64
                } else {
                    0.0
                },
                prefill_tokens_total: pf_total,
                prefill_tokens_cached: pf_cached,
                model_mem: (model_mem > 0).then_some(model_mem),
                weights_mem: (weights_mem > 0).then_some(weights_mem),
                kv_mem: (kv_mem > 0).then_some(kv_mem),
                scratch_mem: (model_mem > 0 && weights_mem > 0 && kv_mem > 0)
                    .then(|| model_mem.saturating_sub(weights_mem + kv_mem)),
                cache_tier: TierSnapshot::read(&m.tier),
            }
        });

        let snap = Arc::new(StatsSnapshot {
            ts: now_secs(),
            pid: std::process::id(),
            engine: eng,
            metal: metal_snapshot(),
        });
        // Err only when every receiver has dropped (server shutting down).
        if tx.send(snap).is_err() {
            break;
        }
        // Ramp: fast while a stream client is connected (receiver_count > 1 -
        // the AppState handle always holds one), idle-slow otherwise. Reading
        // atomics is ns either way; this thread never touches CUDA/inference.
        let ms = if tx.receiver_count() > 1 { 400 } else { 2000 };
        std::thread::sleep(Duration::from_millis(ms));
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
