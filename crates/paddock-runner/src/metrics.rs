//! `/metrics` - the runner's Prometheus surface.
//!
//! Why this exists at all: benchmark harnesses scrape `<base>/metrics` at a
//! few hundred ms during a run by default, and without this module the path
//! falls through to the JSON 404 - the scraper sees non-Prometheus content
//! and silently disables server-metrics collection for the whole run. Every
//! other OpenAI-compatible server exposes one, so this is a conformance fix,
//! not an observability nice-to-have.
//!
//! Naming follows the OTel GenAI semantic conventions where one exists (the
//! same rule `events.rs` already follows for records, so one vocabulary
//! covers both sinks): the three server histograms under their standard
//! OTel->Prometheus mapping, on the semconv's own 14-boundary bucket ladder.
//! Engine internals with no semconv concept are `paddock_`-prefixed, the way
//! vLLM prefixes its own. Semconv GenAI server metrics are *Development*
//! stability - accepted; a later change is a rename, not a redesign.
//!
//! Exposition is HAND-ROLLED in both formats because no crate serves both:
//! `prometheus-client` encodes OpenMetrics only, and classic
//! `text/plain; version=0.0.4` is what most scrapers expect.
//! OpenMetrics is negotiated via `Accept` and is the only format that can
//! carry EXEMPLARS - the caller's `traceparent` attached to a histogram
//! bucket, which makes a p99 spike clickable through to the exact slow
//! request at zero cardinality cost. The renderer's authority is external
//! (`promtool check metrics`, a real scraper run, a Prometheus scrape), same
//! posture as every other black-box validator in this repo.
//!
//! Two invariants, both load-bearing:
//! - **Metadata only, never payload.** No label ever carries a session id, a
//!   user id, a request id or a key hash - that is what makes the loopback
//!   scrape safe to leave unauthenticated (§2.1). Enforced by construction
//!   (labels come from closed enums + the served model id) and by test.
//! - **Bounded cardinality.** Label value spaces are finite: operation ×
//!   origin × the runner's own model ids × HTTP status. Nothing
//!   request-unique can become a label.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use paddock_admin::types::{MetricsSnapshot, SnapshotSeries};
use paddock_engine::metrics::EngineMetrics;

/// The GenAI semconv's explicit bucket boundaries for all three server
/// histograms: a power-of-2 ladder from 10 ms. Shared verbatim with the
/// manager's `usage_bucket` rollup columns (`ttft_h0..h13`) so `/metrics`,
/// the Studio chart and the forecaster cannot disagree about shape.
pub const SEMCONV_BOUNDS: [f64; 14] = [
    0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92,
];

/// The boundaries as exposition strings - fixed text, so the `le=` labels can
/// never drift from [`SEMCONV_BOUNDS`] via float formatting.
const BOUND_LABELS: [&str; 14] = [
    "0.01", "0.02", "0.04", "0.08", "0.16", "0.32", "0.64", "1.28", "2.56", "5.12", "10.24",
    "20.48", "40.96", "81.92",
];

/// `gen_ai.operation.name` - a defined value space (semconv's well-known
/// values where they exist, custom-but-bounded ones where the surface has no
/// semconv name yet). A closed enum deliberately: an unbounded label value is
/// unrepresentable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Operation {
    /// /v1/chat/completions, /v1/responses(+/compact), /v1/messages.
    Chat,
    /// /v1/completions.
    TextCompletion,
    /// /v1/embeddings.
    Embeddings,
    /// /v1/rerank.
    Rerank,
    /// /v1/audio/transcriptions.
    Transcription,
    /// /v1/audio/alignments.
    Alignment,
    /// /v1/messages/count_tokens - its own series so a token-count utility
    /// never pollutes the chat latency histograms.
    CountTokens,
    /// /v1/images/generations.
    ImageGeneration,
    /// /v1/images/edits.
    ImageEdit,
}

impl Operation {
    pub fn from_path(path: &str) -> Option<Self> {
        Some(match path {
            "/v1/chat/completions" | "/v1/responses" | "/v1/responses/compact" | "/v1/messages" => {
                Self::Chat
            }
            "/v1/completions" => Self::TextCompletion,
            "/v1/embeddings" => Self::Embeddings,
            "/v1/rerank" => Self::Rerank,
            "/v1/audio/transcriptions" => Self::Transcription,
            "/v1/audio/alignments" => Self::Alignment,
            "/v1/messages/count_tokens" => Self::CountTokens,
            "/v1/images/generations" => Self::ImageGeneration,
            "/v1/images/edits" => Self::ImageEdit,
            _ => return None,
        })
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::TextCompletion => "text_completion",
            Self::Embeddings => "embeddings",
            Self::Rerank => "rerank",
            Self::Transcription => "transcription",
            Self::Alignment => "alignment",
            Self::ImageGeneration => "image_generation",
            Self::ImageEdit => "image_edit",
            Self::CountTokens => "count_tokens",
        }
    }
}

/// Who sent the request  - the label no rival has because no rival
/// schedules its own batch work. Without it the batch forecaster feeds on its
/// own exhaust: batch runs at 03:00, that traffic lands in the history, and
/// next week 03:00 reads as busy. Cheap now, impossible to backfill.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Origin {
    #[default]
    Live,
    Batch,
    Studio,
}

impl Origin {
    /// From the `x-paddock-origin` request header. Anything unrecognized is
    /// `live` - the value space stays closed no matter what a client sends.
    pub fn from_header(v: Option<&str>) -> Self {
        match v {
            Some(s) if s.eq_ignore_ascii_case("batch") => Self::Batch,
            Some(s) if s.eq_ignore_ascii_case("studio") => Self::Studio,
            _ => Self::Live,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Live => "live",
            Self::Batch => "batch",
            Self::Studio => "studio",
        }
    }
}

/// One completed request, as the events middleware saw it. Everything here is
/// already measured for the event record - this sink adds no instrumentation.
#[derive(Debug, Default)]
pub struct Observation<'a> {
    pub path: &'a str,
    pub origin: Origin,
    pub status: u16,
    /// The response body was dropped before it ran to end - the client
    /// vanished mid-stream. Without this bit an abandoned stream records as a
    /// 200 with a short output count.
    pub disconnected: bool,
    /// Served model id (the truthful one). None on refused requests that
    /// never reached a handler - the registry falls back to the id that OWNS
    /// the operation, so label sets stay consistent.
    pub model: Option<&'a str>,
    pub duration: Duration,
    /// Edge time to first response-body byte (streaming: the client-visible
    /// TTFT; non-streaming: equals duration - the client's truth either way).
    pub ttft: Option<Duration>,
    /// Engine-measured decode wall clock - TPOT's numerator.
    pub decode_ms: Option<u64>,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub cached_tokens: Option<u64>,
    pub spec_drafted: Option<u64>,
    pub spec_accepted: Option<u64>,
    /// (trace_id, span_id) from a caller's `traceparent` - becomes an
    /// OpenMetrics exemplar, never a label.
    pub trace: Option<(String, String)>,
}

/// `00-<32 hex trace-id>-<16 hex span-id>-<flags>` -> (trace_id, span_id).
/// Anything malformed is None - exemplars are best-effort decoration.
pub fn parse_traceparent(tp: &str) -> Option<(String, String)> {
    let mut parts = tp.split('-');
    let version = parts.next()?;
    let trace = parts.next()?;
    let span = parts.next()?;
    let _flags = parts.next()?;
    let hex = |s: &str, n: usize| s.len() == n && s.bytes().all(|b| b.is_ascii_hexdigit());
    (version.len() == 2 && hex(trace, 32) && hex(span, 16) && trace.bytes().any(|b| b != b'0'))
        .then(|| (trace.to_ascii_lowercase(), span.to_ascii_lowercase()))
}

// ── internal state ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Exemplar {
    trace_id: String,
    span_id: String,
    value: f64,
    ts_ms: u64,
}

/// Per-bucket counts are stored NON-cumulative (bucket i = observations in
/// (bound[i-1], bound[i]]; slot 14 = +Inf overflow) and cumulated at render -
/// one increment per observation. One exemplar slot per bucket, last trace
/// wins; observations without trace context never evict one.
struct Hist {
    buckets: [u64; 15],
    sum: f64,
    exemplars: [Option<Exemplar>; 15],
}

impl Hist {
    fn new() -> Self {
        Hist {
            buckets: [0; 15],
            sum: 0.0,
            exemplars: std::array::from_fn(|_| None),
        }
    }

    fn observe(&mut self, v: f64, trace: Option<&(String, String)>) {
        // First bound >= v is v's bucket (`le` is inclusive); past the last
        // bound it lands in +Inf.
        let idx = SEMCONV_BOUNDS.partition_point(|b| *b < v);
        self.buckets[idx] += 1;
        self.sum += v;
        if let Some((t, s)) = trace {
            self.exemplars[idx] = Some(Exemplar {
                trace_id: t.clone(),
                span_id: s.clone(),
                value: v,
                ts_ms: now_ms(),
            });
        }
    }
}

/// Request-scoped series identity. `model` is the runner's own model id -
/// a value space of at most three per process (serving/embedder/asr).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct ReqKey {
    op: Operation,
    origin: Origin,
    model: String,
}

/// The duration histogram splits by `error.type` (semconv: conditionally
/// required when the request ended in error): the HTTP status as a string, or
/// "disconnect" for an abandoned 2xx stream. None = success.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct DurKey {
    base: ReqKey,
    error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct FailKey {
    base: ReqKey,
    class: &'static str, // "4xx" | "5xx" | "disconnect"
}

#[derive(Default)]
struct Reg {
    duration: HashMap<DurKey, Hist>,
    ttft: HashMap<ReqKey, Hist>,
    tpot: HashMap<ReqKey, Hist>,
    prompt_tokens: HashMap<ReqKey, u64>,
    cached_tokens: HashMap<ReqKey, u64>,
    generation_tokens: HashMap<ReqKey, u64>,
    failures: HashMap<FailKey, u64>,
    spec_drafted: u64,
    spec_accepted: u64,
}

impl Default for Hist {
    fn default() -> Self {
        Self::new()
    }
}

/// The model ids this runner serves - the fallback labels for requests that
/// were refused before a handler could name the truthful id.
#[derive(Debug, Clone, Default)]
pub struct ModelIds {
    pub serving: Option<String>,
    pub embedder: Option<String>,
    pub asr: Option<String>,
    pub image: Option<String>,
}

#[derive(Clone, Copy)]
pub enum Format {
    /// `text/plain; version=0.0.4` - what most scrapers expect.
    Classic,
    /// `application/openmetrics-text` - Prometheus with exemplar scraping.
    OpenMetrics,
}

impl Format {
    pub fn content_type(&self) -> &'static str {
        match self {
            Format::Classic => "text/plain; version=0.0.4; charset=utf-8",
            Format::OpenMetrics => "application/openmetrics-text; version=1.0.0; charset=utf-8",
        }
    }
}

/// The registry. One short mutex per completed REQUEST (not per token), same
/// cost class as the event ring's push - the §8.7 "observability never blocks
/// serving" invariant holds.
pub struct Metrics {
    enabled: bool,
    start_unix: u64,
    instance_id: String,
    ids: ModelIds,
    /// Engine gauges read live at scrape time (lock-free atomics).
    engine: Option<Arc<EngineMetrics>>,
    reg: Mutex<Reg>,
}

impl Metrics {
    pub fn new(
        instance_id: String,
        ids: ModelIds,
        engine: Option<Arc<EngineMetrics>>,
    ) -> Arc<Self> {
        Arc::new(Metrics {
            enabled: true,
            start_unix: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            instance_id,
            ids,
            engine,
            reg: Mutex::new(Reg::default()),
        })
    }

    /// `--no-metrics`: observations are dropped, `/metrics` 404s, identify
    /// omits the "metrics" capability.
    pub fn disabled() -> Arc<Self> {
        Arc::new(Metrics {
            enabled: false,
            start_unix: 0,
            instance_id: String::new(),
            ids: ModelIds::default(),
            engine: None,
            reg: Mutex::new(Reg::default()),
        })
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    fn fallback_model(&self, op: Operation) -> &str {
        let owner = match op {
            Operation::Embeddings | Operation::Rerank => &self.ids.embedder,
            Operation::Transcription => &self.ids.asr,
            Operation::ImageGeneration | Operation::ImageEdit => &self.ids.image,
            _ => &self.ids.serving,
        };
        owner.as_deref().unwrap_or("")
    }

    /// Fold one completed request in. Unroutable paths (nothing in the
    /// inference set) observe nothing.
    pub fn observe(&self, o: &Observation<'_>) {
        if !self.enabled {
            return;
        }
        let Some(op) = Operation::from_path(o.path) else {
            return;
        };
        let model = o
            .model
            .unwrap_or_else(|| self.fallback_model(op))
            .to_owned();
        let key = ReqKey {
            op,
            origin: o.origin,
            model,
        };

        // error.type for the duration histogram; class for the failure
        // counter. A disconnect on an ERROR response counts as the error -
        // the refusal is what happened; the hang-up is how the client took it.
        let (error, class): (Option<String>, Option<&'static str>) = if o.status >= 500 {
            (Some(o.status.to_string()), Some("5xx"))
        } else if o.status >= 400 {
            (Some(o.status.to_string()), Some("4xx"))
        } else if o.disconnected {
            (Some("disconnect".to_owned()), Some("disconnect"))
        } else {
            (None, None)
        };

        let secs = o.duration.as_secs_f64();
        let Ok(mut reg) = self.reg.lock() else { return };
        reg.duration
            .entry(DurKey {
                base: key.clone(),
                error,
            })
            .or_default()
            .observe(secs, o.trace.as_ref());
        if let Some(class) = class {
            *reg.failures
                .entry(FailKey {
                    base: key.clone(),
                    class,
                })
                .or_insert(0) += 1;
        } else {
            // Success-only latency shape: a 401's ~0 ms TTFT is not a latency.
            if let Some(t) = o.ttft {
                reg.ttft
                    .entry(key.clone())
                    .or_default()
                    .observe(t.as_secs_f64(), o.trace.as_ref());
            }
            if let (Some(d), Some(out)) = (o.decode_ms, o.output_tokens)
                && out > 0
            {
                reg.tpot
                    .entry(key.clone())
                    .or_default()
                    .observe(d as f64 / 1000.0 / out as f64, o.trace.as_ref());
            }
        }
        if let Some(n) = o.input_tokens {
            *reg.prompt_tokens.entry(key.clone()).or_insert(0) += n;
        }
        if let Some(n) = o.cached_tokens {
            *reg.cached_tokens.entry(key.clone()).or_insert(0) += n;
        }
        if let Some(n) = o.output_tokens {
            *reg.generation_tokens.entry(key).or_insert(0) += n;
        }
        reg.spec_drafted += o.spec_drafted.unwrap_or(0);
        reg.spec_accepted += o.spec_accepted.unwrap_or(0);
    }

    // ── exposition ─────────────────────────────────────────────────────────

    /// Render the whole surface. `in_flight` is the drain counter (HTTP
    /// requests currently held), passed in so this module never reaches into
    /// AppState.
    pub fn render(&self, fmt: Format, in_flight: u64) -> String {
        let om = matches!(fmt, Format::OpenMetrics);
        let mut out = String::with_capacity(8 * 1024);

        // Identity first: who this build is, when the process started (the
        // scraper's counter-reset detector), and the OTel resource mapping.
        // The port is deliberately not a label anywhere - in Prometheus the
        // SCRAPER attaches instance identity, and the manager already has the
        // id from identify; target_info publishes it once for everyone else.
        // In OpenMetrics the `info` family is named `target` and its sample
        // carries the `_info` suffix; classic has no info type, so the family
        // is `target_info` and rides as a gauge - the TYPE line must name the
        // sample exactly or promtool rejects the whole page.
        if om {
            out.push_str("# HELP target Target metadata (OTel resource)\n# TYPE target info\n");
        } else {
            out.push_str(
                "# HELP target_info Target metadata (OTel resource)\n# TYPE target_info gauge\n",
            );
        }
        out.push_str(&format!(
            "target_info{{service_name=\"paddock-runner\",service_version=\"{}\",service_instance_id=\"{}\"}} 1\n",
            escape(paddock_admin::version::SEMVER),
            escape(&self.instance_id),
        ));
        meta(
            &mut out,
            "paddock_build_info",
            "gauge",
            "Build identity: semver + the commit that produced these bytes",
            None,
            om,
        );
        out.push_str(&format!(
            "paddock_build_info{{version=\"{}\",commit=\"{}\"}} 1\n",
            escape(paddock_admin::version::SEMVER),
            paddock_admin::version::GIT_SHA.unwrap_or(""),
        ));
        meta(
            &mut out,
            "process_start_time_seconds",
            "gauge",
            "Unix time the runner process started (counter-reset detection)",
            None,
            om,
        );
        out.push_str(&format!("process_start_time_seconds {}\n", self.start_unix));

        // Request-scoped series, under one short lock. Keys are sorted so a
        // scrape is deterministic (and diffs of two scrapes are readable).
        {
            let reg = match self.reg.lock() {
                Ok(r) => r,
                Err(e) => e.into_inner(),
            };
            self.render_histograms(&mut out, &reg, om);
            self.render_counters(&mut out, &reg, om);
        }

        // Engine gauges, read live - the same lock-free atomics /api/stats
        // samples.
        meta(
            &mut out,
            "paddock_num_requests_in_flight",
            "gauge",
            "HTTP inference requests currently held (streaming bodies included)",
            None,
            om,
        );
        out.push_str(&format!("paddock_num_requests_in_flight {in_flight}\n"));
        if let Some(m) = &self.engine {
            use std::sync::atomic::Ordering::Relaxed;
            let (used, total) = (m.kv_used.load(Relaxed), m.kv_total.load(Relaxed));
            meta(
                &mut out,
                "paddock_num_requests_running",
                "gauge",
                "Sequences holding an engine slot right now (batch width)",
                None,
                om,
            );
            out.push_str(&format!(
                "paddock_num_requests_running {}\n",
                m.active_slots.load(Relaxed)
            ));
            meta(
                &mut out,
                "paddock_kv_cache_usage_perc",
                "gauge",
                "Paged-KV pool utilization, 0..1 (0 when no pool)",
                None,
                om,
            );
            let perc = if total > 0 {
                f64::from(used) / f64::from(total)
            } else {
                0.0
            };
            out.push_str(&format!("paddock_kv_cache_usage_perc {perc}\n"));
            // KV tier (kv-offload) D8 export - the DECISION ledger, not just
            // occupancy: why each lookup did or did not become a restore,
            // which arm the cost model took, whether it was right, and how
            // many bytes moved per byte delivered. Omitted entirely when no
            // tier is armed, so an untiered runner's /metrics is unchanged.
            let t = &m.tier;
            if t.armed.load(Relaxed) == 1 {
                let mut g = |name: &str, typ: &str, help: &str, v: String| {
                    meta(&mut out, name, typ, help, None, om);
                    out.push_str(&format!("{name} {v}\n"));
                };
                let n = |a: &std::sync::atomic::AtomicU64| a.load(Relaxed).to_string();
                let f = |a: &std::sync::atomic::AtomicU64| {
                    format!("{:.6}", f64::from_bits(a.load(Relaxed)))
                };
                // -- decisions
                g(
                    "paddock_kv_tier_lookups_total",
                    "counter",
                    "Prefix lookups the tier was asked about",
                    n(&t.lookups),
                );
                g(
                    "paddock_kv_tier_hits_total",
                    "counter",
                    "Lookups with a restorable extension available",
                    n(&t.hits),
                );
                g(
                    "paddock_kv_tier_miss_cold_total",
                    "counter",
                    "Misses on a prefix the tier never held",
                    n(&t.miss_cold),
                );
                g(
                    "paddock_kv_tier_miss_no_new_tokens_total",
                    "counter",
                    "Misses where a restore would deliver nothing beyond the GPU depth",
                    n(&t.miss_no_new_tokens),
                );
                g(
                    "paddock_kv_tier_miss_tripped_total",
                    "counter",
                    "Misses because the circuit breaker is open",
                    n(&t.miss_tripped),
                );
                g(
                    "paddock_kv_tier_miss_ghost_total",
                    "counter",
                    "Misses on content this tier evicted (capacity alarm)",
                    n(&t.miss_ghost),
                );
                g(
                    "paddock_kv_tier_miss_no_state_total",
                    "counter",
                    "Misses on KV held without the model state a hybrid resumes from",
                    n(&t.miss_no_state),
                );
                g(
                    "paddock_kv_tier_elected_restore_total",
                    "counter",
                    "Hits where the cost model chose restore",
                    n(&t.elected_restore),
                );
                g(
                    "paddock_kv_tier_elected_recompute_total",
                    "counter",
                    "Hits where the cost model chose recompute",
                    n(&t.elected_recompute),
                );
                g(
                    "paddock_kv_tier_parked_total",
                    "counter",
                    "Restores started with the request parked (D5)",
                    n(&t.parked),
                );
                g(
                    "paddock_kv_tier_park_refused_total",
                    "counter",
                    "Restores refused at reservation time (block pool could not seat them)",
                    n(&t.park_refused),
                );
                g(
                    "paddock_kv_tier_resolved_total",
                    "counter",
                    "Parked restores that resolved and served their prefix",
                    n(&t.resolved_ok),
                );
                g(
                    "paddock_kv_tier_abandoned_total",
                    "counter",
                    "Parked restores abandoned past their deadline",
                    n(&t.abandoned),
                );
                g(
                    "paddock_kv_tier_served_from_ram_total",
                    "counter",
                    "Resolved restores sourced from the T1 RAM tier",
                    n(&t.served_from_ram),
                );
                g(
                    "paddock_kv_tier_served_from_nvme_total",
                    "counter",
                    "Resolved restores sourced from the T2 disk tier",
                    n(&t.served_from_nvme),
                );
                g(
                    "paddock_kv_tier_promoted_to_disk_total",
                    "counter",
                    "T1 evictions whose durable copy stayed readable on disk",
                    n(&t.promoted_to_disk),
                );
                // -- amplification
                g(
                    "paddock_kv_tier_useful_bytes_total",
                    "counter",
                    "Payload bytes delivered to the GPU by resolved restores",
                    n(&t.useful_bytes),
                );
                g(
                    "paddock_kv_tier_moved_bytes_total",
                    "counter",
                    "Payload bytes moved for restores, delivered or not",
                    n(&t.moved_bytes),
                );
                // -- occupancy
                g(
                    "paddock_kv_tier_ready_bytes",
                    "gauge",
                    "Bytes resident in the T1 RAM tier",
                    n(&t.t1_ready_bytes),
                );
                g(
                    "paddock_kv_tier_in_flight_bytes",
                    "gauge",
                    "T1 bytes in flight",
                    n(&t.t1_in_flight_bytes),
                );
                g(
                    "paddock_kv_tier_reserved_bytes",
                    "gauge",
                    "T1 bytes reserved but not yet stored",
                    n(&t.t1_reserved_bytes),
                );
                g(
                    "paddock_kv_tier_capacity_bytes",
                    "gauge",
                    "T1 RAM tier capacity",
                    n(&t.t1_capacity_bytes),
                );
                g(
                    "paddock_kv_tier_t2_ready_bytes",
                    "gauge",
                    "Bytes readable from the T2 disk tier",
                    n(&t.t2_ready_bytes),
                );
                g(
                    "paddock_kv_tier_t2_capacity_bytes",
                    "gauge",
                    "T2 disk tier quota",
                    n(&t.t2_capacity_bytes),
                );
                g(
                    "paddock_kv_tier_resident_runs",
                    "gauge",
                    "Extents resident in the T1 RAM tier",
                    n(&t.resident_runs),
                );
                g(
                    "paddock_kv_tier_in_flight_demotes",
                    "gauge",
                    "Demote stores in flight",
                    n(&t.in_flight_demotes),
                );
                g(
                    "paddock_kv_tier_open_tickets",
                    "gauge",
                    "Restore tickets open",
                    n(&t.open_tickets),
                );
                g(
                    "paddock_kv_tier_pending_durable_writes",
                    "gauge",
                    "Durable writes deferred to disk read slack",
                    n(&t.pending_durable_writes),
                );
                g(
                    "paddock_kv_tier_ghost_keys",
                    "gauge",
                    "Recently evicted keys remembered for the capacity alarm",
                    n(&t.ghost_keys),
                );
                // -- health
                g(
                    "paddock_kv_tier_tripped",
                    "gauge",
                    "1 when the tier circuit breaker is open",
                    t.tripped.load(Relaxed).to_string(),
                );
                g(
                    "paddock_kv_tier_io_failures_total",
                    "counter",
                    "Transport failures counted against the breaker",
                    n(&t.io_failures),
                );
                g(
                    "paddock_kv_tier_integrity_failures_total",
                    "counter",
                    "Payloads that failed their checksum at read",
                    n(&t.integrity_failures),
                );
                g(
                    "paddock_kv_tier_evictions_total",
                    "counter",
                    "Tier capacity evictions",
                    n(&t.evictions),
                );
                g(
                    "paddock_kv_tier_single_flight_joins_total",
                    "counter",
                    "Lookups that joined an in-flight load instead of starting one",
                    n(&t.single_flight_joins),
                );
                g(
                    "paddock_kv_tier_stale_completions_total",
                    "counter",
                    "Completions that arrived after their op was gone",
                    n(&t.stale_completions),
                );
                // -- the model's own honesty, and the device it measured
                g(
                    "paddock_kv_tier_rate_ram_bytes_per_us",
                    "gauge",
                    "Measured end-to-end T1 restore rate",
                    f(&t.rate_ram_bpus),
                );
                g(
                    "paddock_kv_tier_rate_nvme_bytes_per_us",
                    "gauge",
                    "Measured end-to-end T2 restore rate",
                    f(&t.rate_nvme_bpus),
                );
                if t.has_prediction_error.load(Relaxed) == 1 {
                    g(
                        "paddock_kv_tier_prediction_error_pct",
                        "gauge",
                        "Mean absolute error of restore-time predictions",
                        f(&t.prediction_error_pct),
                    );
                }
                g(
                    "paddock_kv_tier_device_read_gbs",
                    "gauge",
                    "T2 device read bandwidth measured at open",
                    f(&t.device_read_gbs),
                );
                g(
                    "paddock_kv_tier_device_write_gbs",
                    "gauge",
                    "T2 device write bandwidth measured at open",
                    f(&t.device_write_gbs),
                );
                g(
                    "paddock_kv_tier_device_unbuffered",
                    "gauge",
                    "1 when T2 IO bypasses the OS page cache",
                    t.device_unbuffered.load(Relaxed).to_string(),
                );
                g(
                    "paddock_kv_tier_t2_written_day_bytes",
                    "gauge",
                    "T2 payload bytes written this UTC day (endurance budget)",
                    n(&t.t2_written_day_bytes),
                );
            }
            meta(
                &mut out,
                "paddock_kv_pages_free",
                "gauge",
                "Free paged-KV pages (page-granular, because the pool is)",
                None,
                om,
            );
            out.push_str(&format!(
                "paddock_kv_pages_free {}\n",
                total.saturating_sub(used)
            ));
            meta(
                &mut out,
                "paddock_kv_pages_used",
                "gauge",
                "Paged-KV pages in use (the free gauge's complement - exposed \
                 directly so a consumer never reconstructs it from a ratio)",
                None,
                om,
            );
            out.push_str(&format!("paddock_kv_pages_used {used}\n"));
            // Token-granular prefix-reuse counters (vLLM's queries/hits
            // shape): cumulative prompt tokens prefilled vs served from
            // cache. hits/queries over a rate() window is the reuse rate.
            counter_meta(
                &mut out,
                "paddock_prefix_cache_queries",
                "Prompt tokens prefilled (user-facing prefills), cumulative",
                om,
            );
            out.push_str(&format!(
                "paddock_prefix_cache_queries_total {}\n",
                m.prefill_tokens_total.load(Relaxed)
            ));
            counter_meta(
                &mut out,
                "paddock_prefix_cache_hits",
                "Prompt tokens served from the prefix cache, cumulative",
                om,
            );
            out.push_str(&format!(
                "paddock_prefix_cache_hits_total {}\n",
                m.prefill_tokens_cached.load(Relaxed)
            ));
        }

        if om {
            out.push_str("# EOF\n");
        }
        out
    }

    fn render_histograms(&self, out: &mut String, reg: &Reg, om: bool) {
        let sets: [(&str, &str, &HashMap<DurKey, Hist>); 1] = [(
            "gen_ai_server_request_duration_seconds",
            "End-to-end request duration (arrival to response body complete)",
            &reg.duration,
        )];
        for (name, help, map) in sets {
            meta(out, name, "histogram", help, Some("seconds"), om);
            let mut keys: Vec<_> = map.keys().collect();
            keys.sort();
            for k in keys {
                let mut labels = req_labels(&k.base);
                if let Some(e) = &k.error {
                    labels.push_str(&format!(",error_type=\"{}\"", escape(e)));
                }
                write_hist(out, name, &labels, &map[k], om);
            }
        }
        for (name, help, map) in [
            (
                "gen_ai_server_time_to_first_token_seconds",
                "Time to first response-body byte (client-visible TTFT)",
                &reg.ttft,
            ),
            (
                "gen_ai_server_time_per_output_token_seconds",
                "Engine decode wall clock per output token",
                &reg.tpot,
            ),
        ] {
            meta(out, name, "histogram", help, Some("seconds"), om);
            let mut keys: Vec<_> = map.keys().collect();
            keys.sort();
            for k in keys {
                write_hist(out, name, &req_labels(k), &map[k], om);
            }
        }
    }

    fn render_counters(&self, out: &mut String, reg: &Reg, om: bool) {
        for (family, help, map) in [
            (
                "paddock_prompt_tokens",
                "Prompt tokens processed",
                &reg.prompt_tokens,
            ),
            (
                "paddock_prompt_cached_tokens",
                "Prompt tokens served from the prefix cache (usage.cached_tokens)",
                &reg.cached_tokens,
            ),
            (
                "paddock_generation_tokens",
                "Output tokens generated",
                &reg.generation_tokens,
            ),
        ] {
            counter_meta(out, family, help, om);
            let mut keys: Vec<_> = map.keys().collect();
            keys.sort();
            for k in keys {
                out.push_str(&format!("{family}_total{{{}}} {}\n", req_labels(k), map[k]));
            }
        }
        counter_meta(
            out,
            "paddock_request_failure",
            "Requests that did not complete normally, by class (4xx | 5xx | disconnect - \
             a client that vanished mid-stream is not a success)",
            om,
        );
        let mut keys: Vec<_> = reg.failures.keys().collect();
        keys.sort();
        for k in keys {
            out.push_str(&format!(
                "paddock_request_failure_total{{{},class=\"{}\"}} {}\n",
                req_labels(&k.base),
                k.class,
                reg.failures[k]
            ));
        }
        render_web_spend(out, om);
        counter_meta(
            out,
            "paddock_spec_decode_draft_tokens",
            "Speculative tokens drafted",
            om,
        );
        out.push_str(&format!(
            "paddock_spec_decode_draft_tokens_total {}\n",
            reg.spec_drafted
        ));
        counter_meta(
            out,
            "paddock_spec_decode_accepted_tokens",
            "Speculative tokens accepted by verification",
            om,
        );
        out.push_str(&format!(
            "paddock_spec_decode_accepted_tokens_total {}\n",
            reg.spec_accepted
        ));
    }
}

// ── web-search spend  ───────────────────────────────────────────
//
// Server-executed web searches bill the USER'S own provider key, which makes
// them a second spend channel sitting next to tokens - and until this existed
// paddock spent that money without saying so.
//
// These are whole-runner counters rather than per-request ones. A search has
// no model or route of its own, and the question a user actually asks is "what
// did this server spend", not "on which endpoint". Provider is the only label
// worth carrying, since the currencies are not comparable.
//
// The family names here are mirrored by the manager's exposition parser
// (paddock-manager `usage::parse`) and feed the Studio's spend panel - a
// rename on one side alone empties that panel without failing anything.
//
// The three families are kept apart deliberately. Requests is the honest floor:
// it is the only one every provider supports, because Perplexity reports no
// cost at all and Brave prices only through its rate-limit headers. Credits
// mean nothing outside a provider's own pricing page, and are not searches -
// Firecrawl charged 2 for a one-result search with no scraping. Microdollars
// are integer deliberately: a float counter accumulating a fraction of a cent
// per search drifts, and this is money.

/// Per-provider running spend. A short mutex bumped once per completed search
/// - orders of magnitude rarer than a token, so it cannot contend with serving.
fn web_spend() -> &'static Mutex<BTreeMap<&'static str, WebSpend>> {
    static S: OnceLock<Mutex<BTreeMap<&'static str, WebSpend>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(BTreeMap::new()))
}

#[derive(Default, Clone, Copy)]
struct WebSpend {
    requests: u64,
    credits: u64,
    microdollars: u64,
}

/// Record one billed search. Called on every completed web search, whatever
/// the provider reported - a provider that prices nothing still increments
/// `requests`, so "we searched 40 times and cannot tell you what it cost" is
/// visible rather than looking like no activity.
pub fn web_search_billed(
    provider: &crate::websearch::Provider,
    usage: &crate::websearch::SearchUsage,
) {
    let mut m = web_spend().lock().unwrap_or_else(|e| e.into_inner());
    let e = m.entry(provider.as_str()).or_default();
    e.requests += 1;
    if let Some(c) = usage.credits {
        e.credits += c;
    }
    if let Some(d) = usage.dollars {
        // round rather than truncate: at $0.007 a search, truncation would
        // under-report every single one
        e.microdollars += (d * 1_000_000.0).round().max(0.0) as u64;
    }
}

fn render_web_spend(out: &mut String, om: bool) {
    let m = web_spend().lock().unwrap_or_else(|e| e.into_inner());
    counter_meta(
        out,
        "paddock_web_search_requests",
        "Server-executed web searches, by provider (the honest floor - every \
         provider supports this even when it reports no cost)",
        om,
    );
    for (p, s) in m.iter() {
        out.push_str(&format!(
            "paddock_web_search_requests_total{{provider=\"{p}\"}} {}\n",
            s.requests
        ));
    }
    counter_meta(
        out,
        "paddock_web_search_credits",
        "Provider credits spent on web search (Tavily, Firecrawl - not \
         comparable across providers, and not one per search)",
        om,
    );
    for (p, s) in m.iter().filter(|(_, s)| s.credits > 0) {
        out.push_str(&format!(
            "paddock_web_search_credits_total{{provider=\"{p}\"}} {}\n",
            s.credits
        ));
    }
    counter_meta(
        out,
        "paddock_web_search_cost_microdollars",
        "Web-search spend in millionths of a dollar, as the provider priced it \
         (Exa; integer to keep money off floating point)",
        om,
    );
    for (p, s) in m.iter().filter(|(_, s)| s.microdollars > 0) {
        out.push_str(&format!(
            "paddock_web_search_cost_microdollars_total{{provider=\"{p}\"}} {}\n",
            s.microdollars
        ));
    }
}

/// The spend counters in the snapshot ring's wire shape. Providers that have
/// searched are all carried, even at zero cost - a provider that reports no
/// price still moved `requests`, and dropping it would make the recovery path
/// read a blind window as "nobody searched".
fn web_spend_snapshot() -> Vec<paddock_admin::types::WebSpendSeries> {
    let m = web_spend().lock().unwrap_or_else(|e| e.into_inner());
    m.iter()
        .map(|(p, s)| paddock_admin::types::WebSpendSeries {
            provider: (*p).to_owned(),
            requests: s.requests,
            credits: s.credits,
            microdollars: s.microdollars,
        })
        .collect()
}

impl Metrics {
    // ── snapshots  ───────────────────────────────────

    /// The whole counter set as structured absolutes - "what a scrape would
    /// have read right now". Error-type splits merge here exactly the way the
    /// manager's exposition parser merges them (requests count every outcome,
    /// the failure classes keep the split), and histograms come out CUMULATIVE
    /// like the exposition prints them - so a snapshot pair and a scrape pair
    /// produce byte-identical deltas downstream.
    pub fn usage_snapshot(&self) -> MetricsSnapshot {
        // BTreeMap so a snapshot's series order is deterministic, same reason
        // the renderer sorts its keys.
        let mut out: std::collections::BTreeMap<ReqKey, SnapshotSeries> = Default::default();
        let blank = |k: &ReqKey| SnapshotSeries {
            operation: k.op.as_str().to_owned(),
            origin: k.origin.as_str().to_owned(),
            model: k.model.clone(),
            requests: 0,
            errors_4xx: 0,
            errors_5xx: 0,
            disconnects: 0,
            input_tokens: 0,
            output_tokens: 0,
            cached_tokens: 0,
            duration_seconds_sum: 0.0,
            e2e: [0; 15],
            ttft: [0; 15],
        };
        let (spec_drafted, spec_accepted);
        {
            let reg = match self.reg.lock() {
                Ok(r) => r,
                Err(e) => e.into_inner(),
            };
            for (k, h) in &reg.duration {
                let e = out.entry(k.base.clone()).or_insert_with(|| blank(&k.base));
                // Sum of per-split cumulatives == cumulative of the summed
                // series, so `+=` merges the error splits correctly.
                let mut cum = 0u64;
                for i in 0..15 {
                    cum += h.buckets[i];
                    e.e2e[i] += cum;
                }
                e.requests += cum;
                e.duration_seconds_sum += h.sum;
            }
            for (k, h) in &reg.ttft {
                let e = out.entry(k.clone()).or_insert_with(|| blank(k));
                let mut cum = 0u64;
                for i in 0..15 {
                    cum += h.buckets[i];
                    e.ttft[i] += cum;
                }
            }
            for (k, n) in &reg.prompt_tokens {
                out.entry(k.clone())
                    .or_insert_with(|| blank(k))
                    .input_tokens = *n;
            }
            for (k, n) in &reg.cached_tokens {
                out.entry(k.clone())
                    .or_insert_with(|| blank(k))
                    .cached_tokens = *n;
            }
            for (k, n) in &reg.generation_tokens {
                out.entry(k.clone())
                    .or_insert_with(|| blank(k))
                    .output_tokens = *n;
            }
            for (k, n) in &reg.failures {
                let e = out.entry(k.base.clone()).or_insert_with(|| blank(&k.base));
                match k.class {
                    "4xx" => e.errors_4xx = *n,
                    "5xx" => e.errors_5xx = *n,
                    _ => e.disconnects = *n,
                }
            }
            spec_drafted = reg.spec_drafted;
            spec_accepted = reg.spec_accepted;
        }
        let kv_pages_used = self.engine.as_ref().map_or(0, |m| {
            u64::from(m.kv_used.load(std::sync::atomic::Ordering::Relaxed))
        });
        MetricsSnapshot {
            seq: 0, // assigned by the ring's push
            ts_ms: now_ms(),
            series: out.into_values().collect(),
            spec_drafted,
            spec_accepted,
            kv_pages_used,
            web: web_spend_snapshot(),
        }
    }
}

// ── snapshot ring  ───────────────────────────────────

/// One snapshot a minute, 24 hours deep - a few KB per snapshot on a
/// single-model runner, so single-digit MB against the plan's ~7 MB budget.
/// RAM only: the runner stays stateless, and the ring dying with the process
/// is correct - a restart resets the counters the snapshots are absolutes of.
const SNAPSHOT_RING_CAP: usize = 1440;
pub const SNAPSHOT_PERIOD: Duration = Duration::from_secs(60);

struct SnapInner {
    buf: VecDeque<Arc<MetricsSnapshot>>,
    /// Sequence the next snapshot gets; oldest held = next_seq - buf.len().
    next_seq: u64,
}

/// Bounded ring of periodic counter self-snapshots - the metrics-tier
/// analogue of the event ring, with the same resumable-cursor read contract.
/// A returning manager replays consecutive pairs to reconstruct a blind
/// window at 1-minute resolution instead of recording one opaque gap; this
/// is the piece a generic exporter + Prometheus deployment structurally
/// cannot have, because Prometheus does not own its exporter. We do.
pub struct SnapshotRing {
    inner: Mutex<SnapInner>,
}

impl SnapshotRing {
    pub fn new() -> Arc<Self> {
        Arc::new(SnapshotRing {
            inner: Mutex::new(SnapInner {
                buf: VecDeque::with_capacity(64),
                next_seq: 0,
            }),
        })
    }

    pub fn push(&self, mut snap: MetricsSnapshot) {
        let Ok(mut inner) = self.inner.lock() else {
            return;
        };
        snap.seq = inner.next_seq;
        inner.next_seq += 1;
        inner.buf.push_back(Arc::new(snap));
        if inner.buf.len() > SNAPSHOT_RING_CAP {
            inner.buf.pop_front();
        }
    }

    /// Snapshots at sequence ≥ `from`, up to `max`. Returns
    /// (dropped-before-from, next-cursor, snapshots) - same contract as the
    /// event ring's `since`.
    pub fn since(&self, from: u64, max: usize) -> (u64, u64, Vec<Arc<MetricsSnapshot>>) {
        let Ok(inner) = self.inner.lock() else {
            return (0, from, Vec::new());
        };
        let oldest = inner.next_seq - inner.buf.len() as u64;
        let dropped = oldest.saturating_sub(from);
        let start = from.max(oldest);
        let skip = (start - oldest) as usize;
        let out: Vec<Arc<MetricsSnapshot>> =
            inner.buf.iter().skip(skip).take(max).cloned().collect();
        let next = start + out.len() as u64;
        (dropped, next, out)
    }
}

/// Spawn the snapshot task. The first tick fires immediately, so every
/// generation has an anchor snapshot near its start - that is what lets a
/// manager arriving hours late still recover the generation's whole history
/// (delta from the zero baseline through every minute since).
pub fn start_snapshots(metrics: Arc<Metrics>, ring: Arc<SnapshotRing>) {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(SNAPSHOT_PERIOD);
        // After a machine suspend the missed ticks are not owed: nothing ran,
        // the counters are flat, and a burst of identical snapshots would
        // only waste ring depth.
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tick.tick().await;
            ring.push(metrics.usage_snapshot());
        }
    });
}

/// The shared request-scoped label set. `gen_ai_provider_name` is the
/// semconv-required provider attribute, constant for us.
fn req_labels(k: &ReqKey) -> String {
    format!(
        "gen_ai_operation_name=\"{}\",gen_ai_provider_name=\"paddock\",gen_ai_request_model=\"{}\",origin=\"{}\"",
        k.op.as_str(),
        escape(&k.model),
        k.origin.as_str()
    )
}

/// HELP/TYPE (+ UNIT in OpenMetrics) for a family whose sample name equals
/// the family name.
fn meta(out: &mut String, name: &str, typ: &str, help: &str, unit: Option<&str>, om: bool) {
    // Classic exposition has no `info` type; the family rides as a gauge.
    let typ = if !om && typ == "info" { "gauge" } else { typ };
    out.push_str(&format!("# HELP {name} {help}\n# TYPE {name} {typ}\n"));
    if om && let Some(u) = unit {
        out.push_str(&format!("# UNIT {name} {u}\n"));
    }
}

/// Counter metadata differs between the formats: OpenMetrics names the family
/// without `_total` (samples carry the suffix); classic names it with.
fn counter_meta(out: &mut String, family: &str, help: &str, om: bool) {
    if om {
        out.push_str(&format!(
            "# HELP {family} {help}\n# TYPE {family} counter\n"
        ));
    } else {
        out.push_str(&format!(
            "# HELP {family}_total {help}\n# TYPE {family}_total counter\n"
        ));
    }
}

fn write_hist(out: &mut String, name: &str, labels: &str, h: &Hist, om: bool) {
    let mut cum = 0u64;
    for (i, &b) in h.buckets.iter().enumerate() {
        cum += b;
        let le = if i < 14 { BOUND_LABELS[i] } else { "+Inf" };
        out.push_str(&format!("{name}_bucket{{{labels},le=\"{le}\"}} {cum}"));
        // Exemplars exist only in OpenMetrics - a classic parser would choke.
        if om && let Some(e) = &h.exemplars[i] {
            out.push_str(&format!(
                " # {{trace_id=\"{}\",span_id=\"{}\"}} {} {:.3}",
                e.trace_id,
                e.span_id,
                e.value,
                e.ts_ms as f64 / 1000.0
            ));
        }
        out.push('\n');
    }
    out.push_str(&format!("{name}_sum{{{labels}}} {}\n", h.sum));
    out.push_str(&format!("{name}_count{{{labels}}} {cum}\n"));
}

/// Label-value escaping per both specs: backslash, double-quote, newline.
fn escape(s: &str) -> String {
    if !s.contains(['\\', '"', '\n']) {
        return s.to_owned();
    }
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// ── HTTP surface ───────────────────────────────────────────────────────────

/// `GET /metrics` on the inference port.
///
/// Auth posture is FORCED by the benchmark scrapers, which support no bearer
/// token or custom headers: loopback callers are always open
/// (matching vLLM and SGLang), non-loopback callers
/// need the API key by default (Prometheus carries bearer tokens natively),
/// and `metrics_auth` overrides in either direction. What makes the open
/// loopback acceptable is the metadata-only invariant tested below.
pub async fn handle(
    axum::extract::State(state): axum::extract::State<Arc<crate::routes::AppState>>,
    req: axum::extract::Request,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if !state.metrics.enabled() {
        return (
            axum::http::StatusCode::NOT_FOUND,
            axum::Json(paddock_api::ErrorBody::not_found(
                "route /metrics (--no-metrics)",
            )),
        )
            .into_response();
    }
    // the same "is this the box itself" test the API auth makes: behind a
    // declared proxy, or with a forwarding header, loopback is not local
    let local = crate::routes::peer_is_local(&req, state.trusted_proxy);
    let require = state.auth_key.is_some()
        && match state.metrics_auth {
            Some(forced) => forced,
            None => !local,
        };
    if require {
        let ok = req
            .headers()
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.strip_prefix("Bearer "))
            .is_some_and(|k| Some(k) == state.auth_key.as_deref());
        if !ok {
            return (
                axum::http::StatusCode::UNAUTHORIZED,
                axum::Json(paddock_api::ErrorBody::new(
                    "invalid_api_key",
                    "missing or invalid API key (network /metrics scrapes need the key; \
                     set metrics_auth = false to open it)",
                )),
            )
                .into_response();
        }
    }
    render_response(&state, req.headers())
}

/// Shared with the admin surface (where the pipe's OS ACL is the auth).
pub fn render_response(
    state: &crate::routes::AppState,
    headers: &axum::http::HeaderMap,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    let fmt = headers
        .get(axum::http::header::ACCEPT)
        .and_then(|v| v.to_str().ok())
        .filter(|a| a.contains("application/openmetrics-text"))
        .map_or(Format::Classic, |_| Format::OpenMetrics);
    let ct = fmt.content_type();
    let mut body = state.metrics.render(fmt, state.drain.in_flight() as u64);
    if state.metrics.enabled()
        && let Some(residency) = state.asr.as_ref().and_then(|m| m.transcriber.residency())
    {
        let eof = body.ends_with("# EOF\n");
        if eof {
            body.truncate(body.len() - "# EOF\n".len());
        }
        for (name, value) in [
            (
                "paddock_model_resident",
                u64::from(residency.phase == crate::residency::Phase::Loaded),
            ),
            (
                "paddock_model_residency_leases",
                residency.active_leases as u64,
            ),
            (
                "paddock_model_residency_waiters",
                residency.waiting_requests as u64,
            ),
        ] {
            body.push_str(&format!("# TYPE {name} gauge\n{name} {value}\n"));
        }
        for (name, value) in [
            ("paddock_model_loads", residency.loads),
            ("paddock_model_unloads", residency.unloads),
            ("paddock_model_load_failures", residency.load_failures),
        ] {
            let family = if matches!(fmt, Format::Classic) {
                format!("{name}_total")
            } else {
                name.into()
            };
            body.push_str(&format!("# TYPE {family} counter\n{name}_total {value}\n"));
        }
        if eof {
            body.push_str("# EOF\n");
        }
    }
    ([(axum::http::header::CONTENT_TYPE, ct)], body).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs<'a>(path: &'a str, status: u16, ms: u64) -> Observation<'a> {
        Observation {
            path,
            status,
            duration: Duration::from_millis(ms),
            model: Some("m1"),
            ..Default::default()
        }
    }

    #[test]
    fn classic_histogram_is_cumulative_and_consistent() {
        let m = Metrics::new("i1".into(), ModelIds::default(), None);
        // 30ms -> bucket le=0.04; 500ms -> le=0.64; 200s -> +Inf
        for ms in [30, 500, 200_000] {
            m.observe(&obs("/v1/chat/completions", 200, ms));
        }
        let s = m.render(Format::Classic, 0);
        let get = |le: &str| -> u64 {
            s.lines()
                .find(|l| {
                    l.starts_with("gen_ai_server_request_duration_seconds_bucket")
                        && l.contains(&format!("le=\"{le}\""))
                })
                .and_then(|l| l.rsplit(' ').next())
                .and_then(|v| v.parse().ok())
                .unwrap_or_else(|| panic!("no bucket le={le} in:\n{s}"))
        };
        assert_eq!(get("0.02"), 0);
        assert_eq!(get("0.04"), 1);
        assert_eq!(get("0.64"), 2);
        assert_eq!(get("81.92"), 2);
        assert_eq!(get("+Inf"), 3, "+Inf must equal the count");
        assert!(s.contains("gen_ai_server_request_duration_seconds_count{"));
        assert!(s.contains("# TYPE gen_ai_server_request_duration_seconds histogram"));
        // Classic never carries exemplars or the OpenMetrics terminator.
        assert!(!s.contains(" # {"));
        assert!(!s.contains("# EOF"));
        // Identity block.
        assert!(s.contains("process_start_time_seconds "));
        assert!(s.contains("service_instance_id=\"i1\""));
    }

    #[test]
    fn openmetrics_carries_eof_counter_families_and_exemplars() {
        let m = Metrics::new("i2".into(), ModelIds::default(), None);
        let mut o = obs("/v1/chat/completions", 200, 30);
        o.input_tokens = Some(100);
        o.output_tokens = Some(10);
        o.decode_ms = Some(50);
        o.trace = parse_traceparent("00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01");
        m.observe(&o);
        let s = m.render(Format::OpenMetrics, 0);
        assert!(s.ends_with("# EOF\n"));
        // Counter family named without _total, sample with.
        assert!(s.contains("# TYPE paddock_prompt_tokens counter"));
        assert!(s.contains("paddock_prompt_tokens_total{"));
        assert!(!s.contains("# TYPE paddock_prompt_tokens_total"));
        // The 30ms observation's bucket carries the caller's trace.
        let bucket = s
            .lines()
            .find(|l| l.contains("le=\"0.04\"") && l.contains("request_duration"))
            .expect("bucket line");
        assert!(
            bucket.contains("# {trace_id=\"0af7651916cd43dd8448eb211c80319c\""),
            "exemplar missing: {bucket}"
        );
        assert!(s.contains("# UNIT gen_ai_server_request_duration_seconds seconds"));
        // TPOT observed: 50ms / 10 tokens = 5ms -> le="0.01".
        assert!(s.contains("gen_ai_server_time_per_output_token_seconds_bucket"));
    }

    #[test]
    fn errors_and_disconnects_split_series_and_count_failures() {
        let m = Metrics::new("i3".into(), ModelIds::default(), None);
        m.observe(&Observation {
            status: 401,
            ..obs("/v1/chat/completions", 401, 1)
        });
        m.observe(&Observation {
            disconnected: true,
            ..obs("/v1/chat/completions", 200, 900)
        });
        m.observe(&obs("/v1/chat/completions", 200, 900));
        let s = m.render(Format::Classic, 0);
        assert!(s.contains("error_type=\"401\""));
        assert!(s.contains("error_type=\"disconnect\""));
        assert!(s.contains("class=\"4xx\"} 1"));
        assert!(s.contains("class=\"disconnect\"} 1"));
        // The refused request contributes no ttft sample.
        let ttft_count: u64 = s
            .lines()
            .filter(|l| l.starts_with("gen_ai_server_time_to_first_token_seconds_count"))
            .filter_map(|l| l.rsplit(' ').next()?.parse::<u64>().ok())
            .sum();
        assert_eq!(
            ttft_count, 0,
            "no ttft was measured, so none may be observed"
        );
    }

    /// The safety invariant: metadata only, never payload. No label may
    /// carry request identity - that is what makes the open loopback scrape
    /// (which headerless scrapers force) defensible.
    #[test]
    fn no_label_carries_request_identity() {
        let m = Metrics::new("i4".into(), ModelIds::default(), None);
        let mut o = obs("/v1/chat/completions", 200, 30);
        o.trace = parse_traceparent("00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01");
        m.observe(&o);
        let s = m.render(Format::Classic, 1);
        for forbidden in [
            "session_id=",
            "user=",
            "user_id=",
            "api_key",
            "request_id=",
            "traceparent=",
        ] {
            assert!(
                !s.contains(forbidden),
                "label leaks request identity: {forbidden}\n{s}"
            );
        }
    }

    #[test]
    fn unroutable_paths_and_disabled_registry_observe_nothing() {
        let m = Metrics::new("i5".into(), ModelIds::default(), None);
        m.observe(&obs("/healthz", 200, 5));
        m.observe(&obs("/v1/models", 200, 5));
        // Family headers stay (stable families), but no SAMPLE may appear.
        let s = m.render(Format::Classic, 0);
        assert!(!s.contains("gen_ai_server_request_duration_seconds_bucket"));
        assert!(!s.contains("gen_ai_server_request_duration_seconds_count"));
        let d = Metrics::disabled();
        d.observe(&obs("/v1/chat/completions", 200, 5));
        assert!(!d.enabled());
    }

    #[test]
    fn refused_requests_fall_back_to_the_owning_model_id() {
        let ids = ModelIds {
            serving: Some("qwen3.5-9b".into()),
            embedder: None,
            asr: None,
            image: None,
        };
        let m = Metrics::new("i6".into(), ids, None);
        // 401 refused before any handler could stamp the model.
        m.observe(&Observation {
            model: None,
            ..obs("/v1/chat/completions", 401, 1)
        });
        let s = m.render(Format::Classic, 0);
        assert!(s.contains("gen_ai_request_model=\"qwen3.5-9b\""), "{s}");
    }

    #[test]
    fn traceparent_parsing_rejects_malformed_input() {
        assert!(
            parse_traceparent("00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01").is_some()
        );
        assert!(parse_traceparent("junk").is_none());
        assert!(parse_traceparent("00-short-b7ad6b7169203331-01").is_none());
        // all-zero trace id is invalid per W3C
        assert!(
            parse_traceparent("00-00000000000000000000000000000000-b7ad6b7169203331-01").is_none()
        );
    }

    #[test]
    fn label_values_are_escaped() {
        let ids = ModelIds {
            serving: Some("we\"ird\\model".into()),
            embedder: None,
            asr: None,
            image: None,
        };
        let m = Metrics::new("i7".into(), ids, None);
        m.observe(&Observation {
            model: None,
            ..obs("/v1/chat/completions", 200, 10)
        });
        let s = m.render(Format::Classic, 0);
        assert!(
            s.contains("gen_ai_request_model=\"we\\\"ird\\\\model\""),
            "{s}"
        );
    }

    /// The snapshot must say exactly what a scrape would: error splits merged
    /// into one series, histograms cumulative, failure classes kept apart.
    #[test]
    fn usage_snapshot_merges_splits_and_cumulates() {
        let m = Metrics::new("i8".into(), ModelIds::default(), None);
        let mut ok = obs("/v1/chat/completions", 200, 30); // e2e bucket le=0.04 (idx 2)
        ok.input_tokens = Some(100);
        ok.cached_tokens = Some(40);
        ok.output_tokens = Some(10);
        ok.ttft = Some(Duration::from_millis(15)); // ttft bucket le=0.02 (idx 1)
        ok.spec_drafted = Some(8);
        ok.spec_accepted = Some(6);
        m.observe(&ok);
        m.observe(&obs("/v1/chat/completions", 503, 500)); // error split, le=0.64 (idx 6)
        m.observe(&Observation {
            disconnected: true,
            ..obs("/v1/chat/completions", 200, 30)
        });

        let s = m.usage_snapshot();
        assert!(s.ts_ms > 0);
        assert_eq!(s.seq, 0, "seq belongs to the ring");
        assert_eq!(
            (s.spec_drafted, s.spec_accepted, s.kv_pages_used),
            (8, 6, 0)
        );
        assert_eq!(s.series.len(), 1, "splits must merge into one series");
        let sr = &s.series[0];
        assert_eq!(
            (sr.operation.as_str(), sr.origin.as_str(), sr.model.as_str()),
            ("chat", "live", "m1")
        );
        assert_eq!(sr.requests, 3);
        assert_eq!((sr.errors_5xx, sr.disconnects, sr.errors_4xx), (1, 1, 0));
        assert_eq!(
            (sr.input_tokens, sr.cached_tokens, sr.output_tokens),
            (100, 40, 10)
        );
        // Cumulative: two 30ms observations reach idx 2 and carry forward; the
        // 500ms one joins at idx 6; +Inf (idx 14) equals the count.
        assert_eq!(sr.e2e[1], 0);
        assert_eq!(sr.e2e[2], 2);
        assert_eq!(sr.e2e[5], 2);
        assert_eq!(sr.e2e[6], 3);
        assert_eq!(sr.e2e[14], 3);
        // TTFT observed only on the success.
        assert_eq!(sr.ttft[0], 0);
        assert_eq!(sr.ttft[1], 1);
        assert_eq!(sr.ttft[14], 1);
    }

    /// Web spend reaches both sinks, and the three currencies stay apart. The
    /// requests family is the one that must print for a provider that reports
    /// no price at all - otherwise "we searched and cannot tell you what it
    /// cost" would render identically to "nobody searched".
    #[test]
    fn web_search_spend_renders_and_snapshots() {
        use crate::websearch::{Provider, SearchUsage};
        let m = Metrics::new("i10".into(), ModelIds::default(), None);
        web_search_billed(
            &Provider::Exa,
            &SearchUsage {
                dollars: Some(0.007),
                ..Default::default()
            },
        );
        web_search_billed(
            &Provider::Brave,
            &SearchUsage::default(), // priced nothing - still a search
        );

        let s = m.render(Format::Classic, 0);
        assert!(
            s.contains("paddock_web_search_requests_total{provider=\"exa\"} 1"),
            "{s}"
        );
        assert!(
            s.contains("paddock_web_search_requests_total{provider=\"brave\"} 1"),
            "{s}"
        );
        // $0.007 rounds to 7000 µ$ - truncation would have under-reported it
        assert!(
            s.contains("paddock_web_search_cost_microdollars_total{provider=\"exa\"} 7000"),
            "{s}"
        );
        // a provider that priced nothing contributes no money sample at all
        assert!(
            !s.contains("cost_microdollars_total{provider=\"brave\""),
            "{s}"
        );
        assert!(!s.contains("credits_total{provider=\"exa\""), "{s}");

        let snap = m.usage_snapshot();
        let exa = snap
            .web
            .iter()
            .find(|w| w.provider == "exa")
            .expect("exa in snapshot");
        assert_eq!((exa.requests, exa.microdollars, exa.credits), (1, 7000, 0));
        assert!(
            snap.web
                .iter()
                .any(|w| w.provider == "brave" && w.requests == 1)
        );
    }

    #[test]
    fn snapshot_ring_assigns_sequences_and_reports_drops() {
        let m = Metrics::new("i9".into(), ModelIds::default(), None);
        let ring = SnapshotRing::new();
        for _ in 0..(SNAPSHOT_RING_CAP + 5) {
            ring.push(m.usage_snapshot());
        }
        let (dropped, next, page) = ring.since(0, 100);
        assert_eq!(dropped, 5);
        assert_eq!(page.first().unwrap().seq, 5);
        assert_eq!(next, 105);
        let (dropped, next2, page) = ring.since(next, 100_000);
        assert_eq!(dropped, 0);
        assert_eq!(page.len(), SNAPSHOT_RING_CAP - 100);
        assert_eq!(next2, (SNAPSHOT_RING_CAP + 5) as u64);
        let (dropped, _, page) = ring.since(next2, 10);
        assert_eq!((dropped, page.len()), (0, 0));
    }
}
