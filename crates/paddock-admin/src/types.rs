//! Admin wire contract types. The WIRE is what's versioned - these structs are
//! a convenience for in-repo callers; external tooling can speak plain JSON.
//!
//! Compat rules (doc §6): the v1 CORE (identify, health, drain, shutdown)
//! is frozen - fields may be ADDED (serde ignores unknowns on both ends),
//! never renamed, retyped, or removed. Rich surfaces (stats, events) are
//! capability-gated via `Identify::capabilities` and may evolve.

use serde::{Deserialize, Serialize};

/// Bumped only if the core contract itself must change shape - which the
/// design forbids; expect this to stay 1.
pub const WIRE_VERSION: u32 = 1;

/// `GET /v1/identify` - who is on this pipe. The first thing a manager asks;
/// enough to recognize, display, and (via capabilities) know what else works.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Identify {
    pub wire: u32,
    /// Always "runner" today; lets a future artifact share the namespace.
    pub role: String,
    /// The runner artifact's semver.
    pub version: String,
    pub pid: u32,
    /// The inference port this runner serves (also keys the endpoint name).
    pub port: u16,
    /// Served generative model id, if one is loaded.
    pub model: Option<String>,
    /// Served encoder (embeddings/rerank) model id, if one is loaded.
    pub embedder: Option<String>,
    /// Served speech-to-text model id, if one is loaded. A whisper-family
    /// runner has only this - no `model`, no `embedder` - so a manager that
    /// keys on those two reports it as serving nothing and every UI built on
    /// that list loses it. Optional on the wire so an older runner still
    /// identifies.
    #[serde(default)]
    pub asr: Option<String>,
    /// Served forced-alignment model id, if one is loaded. Same story as
    /// `asr`: an aligner-only runner carries only this, so a consumer keyed
    /// on the other three would report it as serving nothing. Optional on the
    /// wire so an older runner still identifies.
    #[serde(default)]
    pub aligner: Option<String>,
    /// Served image-generation model id, if one is loaded. Same story as
    /// `asr` and `aligner`: an image runner carries only this. Optional on
    /// the wire so an older runner still identifies.
    #[serde(default)]
    pub image: Option<String>,
    /// Served decision model id (Laya: typed questions in, calibrated answers
    /// out, `/v1/systemone` and nothing else), if one is loaded. The sixth
    /// serving role, for the reason the others are separate: a reader
    /// refuses chat, so a picker that took it for a chat model would offer a
    /// conversation it cannot have. Optional on the wire for older runners.
    #[serde(default)]
    pub reader: Option<String>,
    /// Served tabular predictor id (Kumo-Tabular: labelled rows in, the
    /// missing column predicted, `/v1/tabular/*` and nothing else), if one
    /// is loaded. The seventh serving role: it takes tables, never a
    /// conversation. Optional on the wire for older runners.
    #[serde(default)]
    pub tabular: Option<String>,
    /// Audio speaker activity, not a chat or transcription model.
    #[serde(default)]
    pub diarization: Option<String>,
    /// Served dense-prediction model id (DINOv3: chips in, class and height
    /// rasters out, `/v1/segmentations` only). Optional on the wire for older
    /// runners - which a manager then reported as serving nothing.
    #[serde(default)]
    pub segmenter: Option<String>,
    /// Served promptable-segmentation model id (SAM 3: a picture and a
    /// concept or exemplar boxes in, instance masks out, `/v1/masks` only).
    /// Its own role for the reason the others are: it takes pictures and
    /// prompts, never a conversation. Optional on the wire for older runners.
    #[serde(default)]
    pub masker: Option<String>,
    /// Unix seconds when the runner started. Reset DETECTION only (the
    /// `process_start_time_seconds` job) - never an identity key: it is
    /// second-resolution, and two generations on one port inside the same
    /// second collided on it. `instance_id` is the key.
    pub started_at_unix: u64,
    /// This GENERATION's identity: a UUID minted once per process start,
    /// held in memory only, dies with the process (`service.instance.id`,
    /// ephemeral by design - a restart is the boundary where counters and
    /// event sequences reset, so the id must change with it). Collision-free
    /// regardless of clock resolution or PID reuse. Empty when talking to a
    /// runner older than this field; consumers synthesize
    /// `legacy-<port>-<started>` then, accepting the old second-resolution
    /// semantics for old binaries.
    #[serde(default)]
    pub instance_id: String,
    /// Rich surfaces this runner supports (e.g. "stats"). The manager degrades
    /// with reasons when one is missing - never blanks.
    pub capabilities: Vec<String>,
    /// Speculation the runner actually wired at load (self-report): the
    /// manager's catalog-side prediction defers to this, falling back for
    /// pre-field runners. Optional on the wire for exactly that skew.
    #[serde(default)]
    pub spec: Option<SpecInfo>,
}

/// The runner's speculation self-report on `identify`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecInfo {
    /// the family drafts from heads in the weights (qwen nextn, nemotron MTP)
    pub heads: bool,
    /// attached companion drafter's file stem ("dflash2-Q4_K_M")
    pub drafter: Option<String>,
    /// the config policy resolved to off (no drafting at all)
    pub off: bool,
}

/// `GET /v1/health`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Health {
    /// "ok" | "draining".
    pub status: String,
    /// Inference requests currently in flight (streaming bodies included).
    pub in_flight: u64,
    pub uptime_s: u64,
}

/// `POST /v1/drain` body (all fields optional).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DrainRequest {
    /// How long to wait for in-flight requests before giving up (the call
    /// returns either way, reporting what happened). Default 30 000.
    pub timeout_ms: Option<u64>,
}

/// Drain outcome/state. Draining is one-way: there is no undrain - a drained
/// runner's next step is exit (shutdown or takeover), per doc §5.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrainState {
    pub draining: bool,
    pub in_flight: u64,
    /// True once in-flight hit zero.
    pub drained: bool,
    /// True if the wait timed out with requests still in flight.
    pub timed_out: bool,
}

/// `POST /v1/shutdown` body.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShutdownRequest {
    /// Drain timeout before the process exits anyway. Default 30 000.
    pub timeout_ms: Option<u64>,
}

/// `POST /v1/shutdown` ack - the process exits shortly after sending this;
/// the manager should then wait on the process handle, not the pipe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShutdownAck {
    /// "draining-then-exit".
    pub status: String,
}

/// `GET /v1/events?since=&max=&wait_ms=` - one page of the runner's event
/// ring (capability "events"). Records stay schemaless here deliberately: the
/// record schema is the RUNNER's (semconv-named, may grow fields freely); the
/// collector stores/forwards them as JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventsPage {
    /// Resume cursor: pass back as `since` for the next page.
    pub next: u64,
    /// Records lost before this page because the reader fell off the ring's
    /// tail ("K events dropped" - never a silent gap).
    pub dropped: u64,
    pub events: Vec<serde_json::Value>,
}

/// One periodic self-snapshot of the runner's counter set (capability
/// "metrics-snapshots"): the metrics-tier analogue of an event
/// record. Taken every minute into a bounded RAM ring; a returning manager
/// replays consecutive pairs to reconstruct a blind window's shape at
/// 1-minute resolution instead of writing one opaque gap. Counters are
/// CUMULATIVE absolutes, exactly like the exposition - a snapshot is "what
/// /metrics would have said at ts_ms", not a delta.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    /// Ring sequence number - the reader's resume cursor.
    pub seq: u64,
    /// Unix millis the snapshot was taken.
    pub ts_ms: u64,
    pub series: Vec<SnapshotSeries>,
    /// Engine-scoped cumulative counters (one model per runner - these have
    /// no per-series dimensions of their own).
    pub spec_drafted: u64,
    pub spec_accepted: u64,
    /// Paged-KV pages in use at snapshot time - a gauge, not a counter.
    pub kv_pages_used: u64,
    /// Server-executed web-search spend, by provider. Defaulted
    /// so a snapshot written by an older runner - or persisted by an older
    /// manager as its attach baseline - still decodes; an absent field reads
    /// as "this generation never searched", which is what it means.
    #[serde(default)]
    pub web: Vec<WebSpendSeries>,
}

/// One provider's cumulative web-search spend inside a snapshot.
///
/// Three counters that are deliberately not one number: `requests` is the
/// only one every provider reports, credits mean nothing outside a given
/// provider's pricing page, and dollars exist only where a provider prices
/// in them. Summing them would invent a currency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSpendSeries {
    pub provider: String,
    pub requests: u64,
    pub credits: u64,
    /// Millionths of a dollar - integer, because this is money and a float
    /// counter accumulating a fraction of a cent per search drifts.
    pub microdollars: u64,
}

/// One (operation, origin, model) series' cumulative counters inside a
/// snapshot. Error-type splits are already merged (requests counts every
/// outcome; the failure classes carry the split), matching what the
/// manager's exposition parser produces from a live scrape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotSeries {
    pub operation: String,
    pub origin: String,
    pub model: String,
    pub requests: u64,
    pub errors_4xx: u64,
    pub errors_5xx: u64,
    pub disconnects: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_tokens: u64,
    pub duration_seconds_sum: f64,
    /// Cumulative histogram counts on the semconv ladder; index 14 is +Inf
    /// (== the observation count). Fixed length deliberately: a ladder change
    /// is a wire change, and a hard decode error beats a misfiled bucket.
    pub e2e: [u64; 15],
    pub ttft: [u64; 15],
}

/// `GET /v1/metrics_snapshots?since=&max=` - one page of the snapshot ring,
/// resumable by sequence exactly like the event ring. No long-poll: the
/// manager pulls this once on attach, never subscribes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotsPage {
    /// Resume cursor: pass back as `since` for the next page.
    pub next: u64,
    /// Snapshots lost before this page (fell off the ring's tail - for a
    /// reader starting at 0 this just counts age-expired snapshots).
    pub dropped: u64,
    pub snapshots: Vec<MetricsSnapshot>,
}

/// Credential-free restart status; absence means an older runner, not applied.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigStatus {
    pub pid: u32,
    pub restart_required: Option<bool>,
    pub changed: Vec<String>,
    pub max_ctx: usize,
    pub max_batch: usize,
    #[serde(default)]
    pub residency_live: Option<bool>,
    /// True only AFTER device resources have been destroyed. Loading,
    /// unloading, failed/unknown disposal, and older runners never imply free.
    #[serde(default)]
    pub residency_unloaded: Option<bool>,
    /// Immutable serving envelope of a residency-managed model (device bytes).
    #[serde(default)]
    pub residency_budget: Option<u64>,
}

impl ConfigStatus {
    pub fn residency_reservation(&self, expected_pid: u32) -> Option<u64> {
        if self.pid != expected_pid || self.residency_live != Some(true) {
            return None;
        }
        if self.residency_released(expected_pid) {
            Some(0)
        } else {
            self.residency_budget
        }
    }
    pub fn residency_released(&self, expected_pid: u32) -> bool {
        self.pid == expected_pid
            && self.residency_live == Some(true)
            && self.residency_unloaded == Some(true)
    }
}

#[cfg(test)]
mod residency_status_tests {
    use super::ConfigStatus;
    #[test]
    fn releasing_a_reservation_needs_new_runner_confirmation_and_matching_pid() {
        let old: ConfigStatus = serde_json::from_str(
            r#"{"pid":42,"restart_required":false,"changed":[],"max_ctx":448,"max_batch":1}"#,
        )
        .unwrap();
        assert!(!old.residency_released(42));
        assert_eq!(old.residency_reservation(42), None);
        let mut status = old;
        status.residency_live = Some(true);
        status.residency_budget = Some(4096);
        assert_eq!(status.residency_reservation(42), Some(4096));
        assert!(!status.residency_released(42));
        status.residency_unloaded = Some(true);
        assert!(status.residency_released(42));
        assert_eq!(status.residency_reservation(42), Some(0));
        assert_eq!(status.residency_reservation(43), None);
        assert!(!status.residency_released(43));
        status.residency_unloaded = Some(false);
        assert!(!status.residency_released(42));
    }
}
