//! Live engine metrics - lock-free counters the scheduler updates and the server
//! reads for telemetry. Pure observability: only `Relaxed` atomic stores on the
//! generation path, so they change no numerics (bit-exact gates unaffected) and
//! never take a lock the decode loop holds.

use std::sync::atomic::{AtomicU8, AtomicU32, AtomicU64};

pub const PHASE_IDLE: u8 = 0;
pub const PHASE_PREFILL: u8 = 1;
pub const PHASE_DECODE: u8 = 2;

/// The KV tier's export: decision accountability, not occupancy.
/// Mirrors `kv_tier::TierReport` field for field so the copy is
/// mechanical; `f64`s ride as bit patterns because an atomic float is not
/// worth a dependency for numbers a human reads once a second.
#[derive(Default)]
pub struct TierGauges {
    /// 1 once a tier has been built for this model.
    pub armed: AtomicU8,
    pub lookups: AtomicU64,
    pub hits: AtomicU64,
    pub miss_cold: AtomicU64,
    pub miss_no_new_tokens: AtomicU64,
    pub miss_tripped: AtomicU64,
    pub miss_ghost: AtomicU64,
    pub miss_no_state: AtomicU64,
    pub elected_restore: AtomicU64,
    pub elected_recompute: AtomicU64,
    pub parked: AtomicU64,
    pub park_refused: AtomicU64,
    pub resolved_ok: AtomicU64,
    pub abandoned: AtomicU64,
    pub served_from_ram: AtomicU64,
    pub served_from_nvme: AtomicU64,
    pub promoted_to_disk: AtomicU64,
    pub useful_bytes: AtomicU64,
    pub moved_bytes: AtomicU64,
    pub t1_ready_bytes: AtomicU64,
    pub t1_in_flight_bytes: AtomicU64,
    pub t1_reserved_bytes: AtomicU64,
    pub t1_capacity_bytes: AtomicU64,
    pub t2_ready_bytes: AtomicU64,
    pub t2_capacity_bytes: AtomicU64,
    pub resident_runs: AtomicU64,
    pub in_flight_demotes: AtomicU64,
    pub open_tickets: AtomicU64,
    pub pending_durable_writes: AtomicU64,
    pub tripped: AtomicU8,
    pub io_failures: AtomicU64,
    pub integrity_failures: AtomicU64,
    pub evictions: AtomicU64,
    pub single_flight_joins: AtomicU64,
    pub stale_completions: AtomicU64,
    /// f64 bit patterns (see the struct note).
    pub rate_ram_bpus: AtomicU64,
    pub rate_nvme_bpus: AtomicU64,
    /// Prediction error, f64 bits; `has_error` is 0 until anything was
    /// observed, so "no data yet" never renders as a confident 0%.
    pub prediction_error_pct: AtomicU64,
    pub has_prediction_error: AtomicU8,
    pub device_read_gbs: AtomicU64,
    pub device_write_gbs: AtomicU64,
    pub device_unbuffered: AtomicU8,
    pub t2_written_day_bytes: AtomicU64,
    pub ghost_keys: AtomicU64,
}

impl TierGauges {
    /// Copy one assembled report in. Called once per scheduler pass from the
    /// engine thread; Relaxed throughout - readers want a recent view, not a
    /// consistent instant, and ordering here would cost the decode loop.
    pub fn store(&self, r: &crate::kv_tier::TierReport) {
        use std::sync::atomic::Ordering::Relaxed;
        let d = &r.decisions;
        self.armed.store(1, Relaxed);
        self.lookups.store(d.lookups, Relaxed);
        self.hits.store(d.hits, Relaxed);
        self.miss_cold.store(d.miss_cold, Relaxed);
        self.miss_no_new_tokens.store(d.miss_no_new_tokens, Relaxed);
        self.miss_tripped.store(d.miss_tripped, Relaxed);
        self.miss_ghost.store(d.miss_ghost, Relaxed);
        self.miss_no_state.store(d.miss_no_state, Relaxed);
        self.elected_restore.store(d.elected_restore, Relaxed);
        self.elected_recompute.store(d.elected_recompute, Relaxed);
        self.parked.store(d.parked, Relaxed);
        self.park_refused.store(d.park_refused, Relaxed);
        self.resolved_ok.store(d.resolved_ok, Relaxed);
        self.abandoned.store(d.abandoned, Relaxed);
        self.served_from_ram.store(d.served_from_ram, Relaxed);
        self.served_from_nvme.store(d.served_from_nvme, Relaxed);
        self.promoted_to_disk.store(d.promoted_to_disk, Relaxed);
        self.useful_bytes.store(d.useful_bytes, Relaxed);
        self.moved_bytes.store(d.moved_bytes, Relaxed);
        self.t1_ready_bytes.store(r.t1_ready_bytes, Relaxed);
        self.t1_in_flight_bytes.store(r.t1_in_flight_bytes, Relaxed);
        self.t1_reserved_bytes.store(r.t1_reserved_bytes, Relaxed);
        self.t1_capacity_bytes.store(r.t1_capacity_bytes, Relaxed);
        self.t2_ready_bytes.store(r.t2_ready_bytes, Relaxed);
        self.t2_capacity_bytes.store(r.t2_capacity_bytes, Relaxed);
        self.resident_runs.store(r.resident_runs, Relaxed);
        self.in_flight_demotes.store(r.in_flight_demotes, Relaxed);
        self.open_tickets.store(r.open_tickets, Relaxed);
        self.pending_durable_writes
            .store(r.pending_durable_writes, Relaxed);
        self.tripped.store(u8::from(r.tripped), Relaxed);
        self.io_failures.store(r.io_failures, Relaxed);
        self.integrity_failures.store(r.integrity_failures, Relaxed);
        self.evictions.store(r.evictions, Relaxed);
        self.single_flight_joins
            .store(r.single_flight_joins, Relaxed);
        self.stale_completions.store(r.stale_completions, Relaxed);
        self.rate_ram_bpus.store(r.rate_ram_bpus.to_bits(), Relaxed);
        self.rate_nvme_bpus
            .store(r.rate_nvme_bpus.to_bits(), Relaxed);
        match r.prediction_error_pct {
            Some(e) => {
                self.prediction_error_pct.store(e.to_bits(), Relaxed);
                self.has_prediction_error.store(1, Relaxed);
            }
            None => self.has_prediction_error.store(0, Relaxed),
        }
        self.device_read_gbs
            .store(r.device_read_gbs.to_bits(), Relaxed);
        self.device_write_gbs
            .store(r.device_write_gbs.to_bits(), Relaxed);
        self.device_unbuffered
            .store(u8::from(r.device_unbuffered), Relaxed);
        self.t2_written_day_bytes
            .store(r.t2_written_day_bytes, Relaxed);
        self.ghost_keys.store(r.ghost_keys, Relaxed);
    }
}

/// Shared between the engine thread (writer) and telemetry (reader) via an
/// `Arc`. Every field is an atomic; readers get an eventually-consistent view.
#[derive(Default)]
pub struct EngineMetrics {
    /// Monotonic count of output tokens committed across all sequences. The
    /// reader derives tok/s from the delta between samples.
    pub tokens_generated: AtomicU64,
    /// Sequences in flight right now (batch width; 1 on the serial engine).
    pub active_slots: AtomicU32,
    /// 0 idle, 1 prefill, 2 decode.
    pub phase: AtomicU8,
    /// Paged-KV blocks in use / total capacity (0 when the backend has no pool).
    pub kv_used: AtomicU32,
    pub kv_total: AtomicU32,
    /// Prefix-cache effectiveness, monotonic over user-facing prefills (excludes
    /// preemption recomputes). `prefill_tokens_cached / prefill_tokens_total` is
    /// the running hit rate; the reader can also diff samples for a windowed
    /// rate. Prompt tokens served from a shared/prior prefix skip recompute, so
    /// this is the direct measure of the tier-1 agentic reuse the gensweep's
    /// unique-salt path hides. Relaxed stores only - no numeric effect.
    pub prefill_tokens_total: AtomicU64,
    pub prefill_tokens_cached: AtomicU64,
    /// KV tier accounting, refreshed once per scheduler
    /// pass while a tier is armed. All zero when untiered - `armed` is what
    /// separates "no tier" from "a tier that has answered nothing yet".
    pub tier: TierGauges,
    /// Measured device bytes the loaded model holds in its mempool (weights,
    /// KV/state pools, scratch), sampled by the engine thread after the batch
    /// pools are allocated, again once warm, and live while serving (at most
    /// once a second). 0 = not sampled (CPU backend, or before the engine
    /// finished setup).
    pub model_mem_bytes: AtomicU64,
    /// Memory breakdown: device bytes held by weights
    /// (all serving classes) and by the KV cache, where the family reports
    /// them. 0 = not reported. scratch/other = model_mem - weights - kv,
    /// derived reader-side.
    pub weights_mem_bytes: AtomicU64,
    pub kv_mem_bytes: AtomicU64,
}
