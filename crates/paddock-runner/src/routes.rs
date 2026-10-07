//! HTTP routes - the runner's network surface: the inference dialects plus a
//! small operational set (/healthz, /api/server, /api/stats). OpenAI-shaped
//! errors everywhere - including 404s, because SDKs parse those too. No Studio,
//! no catalog, no device telemetry: those are the manager's.

use std::sync::Arc;

use axum::Router;
use axum::extract::State;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use paddock_api::{ErrorBody, ModelList, ModelObject};

use crate::serving::{AsrModel, EmbedModel, ServingModel};

/// Live view of the config file's CONTROL-PLANE fields: the MCP tool registry
/// (`mcp_servers`) and the web-search provider. Neither touches VRAM, weights
/// or the engine, so unlike every other config field they re-read when the
/// file's mtime moves: the manager's every-server connectors and a web-search
/// key change apply on the next request, never a model restart
/// (restart-to-apply for a tool flip is not acceptable). A parse
/// error keeps the last good snapshot and logs; engine-binding fields stay
/// restart-only; a runner launched without a config file keeps its startup
/// values.
pub struct LiveConfig {
    path: Option<std::path::PathBuf>,
    cache: std::sync::Mutex<(Option<std::time::SystemTime>, Arc<LiveSnapshot>)>,
}

pub struct LiveSnapshot {
    pub mcp_servers: Vec<serde_json::Value>,
    pub web_search: Option<crate::websearch::SearchConfig>,
}

#[derive(serde::Deserialize)]
struct LiveFields {
    #[serde(default)]
    mcp_servers: Vec<serde_json::Value>,
    #[serde(default)]
    web_search_provider: Option<String>,
    #[serde(default)]
    web_search_api_key: Option<String>,
}

impl LiveConfig {
    pub fn new(path: Option<std::path::PathBuf>, initial: LiveSnapshot) -> Self {
        Self {
            path,
            cache: std::sync::Mutex::new((None, Arc::new(initial))),
        }
    }
    pub fn fixed(initial: LiveSnapshot) -> Self {
        Self::new(None, initial)
    }
    /// The current snapshot - one `stat` when the file hasn't moved.
    pub fn snapshot(&self) -> Arc<LiveSnapshot> {
        let mut c = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(p) = &self.path {
            let mtime = std::fs::metadata(p).and_then(|m| m.modified()).ok();
            if mtime.is_some() && mtime != c.0 {
                match std::fs::read_to_string(p)
                    .map_err(|e| e.to_string())
                    .and_then(|t| toml::from_str::<LiveFields>(&t).map_err(|e| e.to_string()))
                {
                    Ok(f) => {
                        c.1 = Arc::new(LiveSnapshot {
                            mcp_servers: f.mcp_servers,
                            web_search: crate::websearch::SearchConfig::from_fields(
                                f.web_search_provider.as_deref(),
                                f.web_search_api_key.as_deref(),
                            ),
                        });
                    }
                    Err(e) => tracing::warn!(
                        error = %e,
                        "config re-read failed - keeping the last good tool/web-search snapshot"
                    ),
                }
                c.0 = mtime;
            }
        }
        c.1.clone()
    }
}

/// What a request that sends no dials is sampled at.
///
/// Three layers, most specific first: the request field, then an
/// operator pin (`--temp`/`--top-p`/...), then the served checkpoint's own
/// published profile, and only where nobody published anything do the OpenAI
/// wire values stand. That last layer used to be the only layer, so
/// every model was served at temperature 1.0 with no truncation at all - full
/// entropy, which is not what any of these labs measured their model at.
///
/// The elected profile is data, not taste: `paddock_models::sampling` carries
/// the numbers with a citation per row. This struct just decides who wins.
#[derive(Debug, Clone, Copy)]
pub struct SamplingDefaults {
    /// The served checkpoint's published profile, or None when its authors
    /// publish no decoding parameters (gpt-oss, granite) - and None on every
    /// non-generative lane, which never reads this.
    pub elected: Option<paddock_models::sampling::Elected>,
    /// Operator pins. `Some` means a flag/env/config said so and it beats the
    /// election; `None` means nobody did.
    pub temp: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub repeat_penalty: Option<f32>,
    /// Penalty window. Not part of any election - no lab publishes it (it is
    /// a llama.cpp knob), and with `repeat_penalty` at 1.0 and both OpenAI
    /// penalties at 0 it has nothing to do.
    pub repeat_last_n: usize,
    /// None = per-request time-derived seed (OpenAI semantics).
    pub seed: Option<u64>,
}

/// One turn's resolved defaults - what the request's omitted fields become.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Resolved {
    pub temp: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repeat_penalty: f32,
}

impl Default for SamplingDefaults {
    /// No model, no pins: the OpenAI wire. Tests and the non-generative lanes
    /// live here, and a served model replaces it at startup.
    fn default() -> Self {
        Self {
            elected: None,
            temp: None,
            top_k: None,
            top_p: None,
            min_p: None,
            repeat_penalty: None,
            repeat_last_n: 64,
            seed: None,
        }
    }
}

impl SamplingDefaults {
    /// Build from the config's pins and whatever the served architecture
    /// publishes. `arch` is `general.architecture` - the one identity field
    /// every checkpoint fills in honestly (laguna's `general.name` is a bare
    /// commit hash).
    ///
    /// `published` is what the checkpoint's own header said, for the case the
    /// arch string cannot decide (granite 4.1 vs 4.2 - same `arch`, different
    /// published sampling). The table wins wherever it has a row: a model card
    /// can express the thinking/instruct split that a header cannot, so the
    /// file only ever FILLS a hole.
    pub fn for_model(
        cfg: &crate::config::Config,
        arch: Option<&str>,
        published: Option<paddock_models::sampling::Elected>,
    ) -> Self {
        Self {
            elected: arch
                .and_then(paddock_models::sampling::elected)
                .or(published),
            temp: cfg.temp,
            top_k: cfg.top_k,
            top_p: cfg.top_p,
            min_p: cfg.min_p,
            repeat_penalty: cfg.repeat_penalty,
            repeat_last_n: cfg.repeat_last_n,
            seed: cfg.seed,
        }
    }

    /// The defaults for this turn. `thinking` is the effective template
    /// toggle (`Dialect::thinking_open` on the rendered prompt), because the
    /// qwen cards publish a different set for each mode and the rendered
    /// prompt is the only place that knows which one the turn is in.
    pub fn resolve(&self, thinking: bool) -> Resolved {
        let base = self
            .elected
            .map_or(paddock_models::sampling::WIRE_DEFAULTS, |e| {
                e.knobs(thinking)
            });
        Resolved {
            temp: self.temp.unwrap_or(base.temperature),
            top_k: self.top_k.unwrap_or(base.top_k),
            top_p: self.top_p.unwrap_or(base.top_p),
            min_p: self.min_p.unwrap_or(base.min_p),
            // no election publishes a repetition penalty, so this one is
            // pin-or-off rather than pin-or-elected
            repeat_penalty: self.repeat_penalty.unwrap_or(1.0),
        }
    }

    /// Where the resolved numbers came from, in a phrase short enough for a
    /// popover. A user must be able to see why their model samples the way it
    /// does without reading our source.
    pub fn provenance(&self) -> &'static str {
        match (self.elected.is_some(), self.any_pin()) {
            (true, false) => "this model's published defaults",
            (true, true) => "this model's published defaults, with server settings on top",
            (false, false) => "the OpenAI API defaults - this model publishes none of its own",
            (false, true) => "server settings - this model publishes no defaults of its own",
        }
    }

    /// The citation behind `provenance`, when there is one: which artifact of
    /// the model's the numbers were read out of. Long by design - it is for
    /// the startup log and anyone auditing the API, not for a menu.
    pub fn provenance_detail(&self) -> Option<&'static str> {
        self.elected.map(|e| e.source)
    }

    fn any_pin(&self) -> bool {
        self.temp.is_some()
            || self.top_k.is_some()
            || self.top_p.is_some()
            || self.min_p.is_some()
            || self.repeat_penalty.is_some()
    }

    /// Resolve a request's omitted seed: server default, else time-derived.
    pub fn seed_or_now(&self, req_seed: Option<u64>) -> u64 {
        req_seed.or(self.seed).unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0)
        })
    }
}

/// Shared server state. Grows scheduler/router handles as phases land.
pub struct AppState {
    /// Required Bearer key for /v1 + /api, or None for no auth (loopback).
    /// The runner holds exactly one key, from its config - the manager issues
    /// it at spawn (doc §5.1); there is no key store in the data plane.
    pub auth_key: Option<String>,
    /// The operator declared a reverse proxy in front of this runner
    /// (`trusted_proxy`). Then every caller arrives from 127.0.0.1 and the
    /// loopback exemption in `auth_mw` would be no exemption at all, so it is
    /// off.
    pub trusted_proxy: bool,
    /// The config's speculation policy resolved to off - part of the admin
    /// identify self-report (`SpecInfo.off`).
    pub spec_off: bool,
    /// This GENERATION's identity (`service.instance.id`): a UUID
    /// minted once per process start, in memory only, dies with the process.
    /// Ephemeral by design - a restart is the boundary where counters and
    /// event sequences reset, so the id must change with it. `started_at_unix`
    /// stays for reset DETECTION only; keying on it collided at second
    /// resolution.
    pub instance_id: String,
    /// The /metrics registry  - second sink beside the event ring,
    /// independently gated (`--no-metrics` vs `--no-events`).
    pub metrics: Arc<crate::metrics::Metrics>,
    /// 1-minute counter self-snapshots  - what lets a
    /// returning manager reconstruct a blind window's shape instead of
    /// recording one opaque gap. Rides the metrics gate: with `--no-metrics`
    /// the task never spawns and the ring stays empty.
    pub snapshots: Arc<crate::metrics::SnapshotRing>,
    /// /metrics auth override: None = the key is required for
    /// non-loopback callers only (when one is configured); Some(true)/(false)
    /// force it for everyone / no one. The default exists because the common
    /// benchmark scrapers cannot send any auth header.
    pub metrics_auth: Option<bool>,
    /// The loaded generative model, if one was configured at startup.
    pub serving: Option<ServingModel>,
    /// Server-side sampling defaults (request fields win; see the struct doc).
    pub sampling: SamplingDefaults,
    /// The loaded encoder model (embeddings/rerank), if one was configured.
    pub embedder: Option<EmbedModel>,
    /// The loaded speech-to-text model (whisper family), if one was
    /// configured - serves `/v1/audio/transcriptions` only.
    pub asr: Option<AsrModel>,
    /// The loaded forced-alignment model (Qwen3-ForcedAligner), if one was
    /// configured - serves `/v1/audio/alignments` only.
    pub aligner: Option<crate::serving::AlignModel>,
    /// The loaded dense-prediction model (tic-forestry: DINOv3 + decoder), if
    /// one was configured - serves `/v1/segmentations` only.
    pub segmenter: Option<crate::serving::SegmentModel>,
    /// Promptable segmentation (SAM 3): `/v1/masks`.
    pub masker: Option<crate::serving::MaskModel>,
    /// The loaded decision model (a Laya bundle), if one was configured -
    /// serves `POST /v1/systemone` only.
    pub laya: Option<crate::systemone::laya::LayaModel>,
    /// The loaded decision model (a Clef checkpoint), if one was configured -
    /// serves `POST /v1/systemone` only.
    pub clef: Option<crate::systemone::clef::ClefModel>,
    pub tabular: Option<crate::tabular::TabularModel>,
    pub diarization: Option<crate::diarizations::DiarizationModel>,
    /// The loaded image-generation model (Qwen-Image), if one was configured
    /// - serves `/v1/images/*` only.
    pub image: Option<crate::serving::ImageModel>,
    /// Served context window (`--max-ctx`) - the hard ceiling for a reply's
    /// tokens; clients can read it from /api/server.
    pub max_ctx: usize,
    /// `--vad-gate`: skip transcription windows the VAD finds no speech in
    /// Whisper lane only - the generative families take the whole
    /// clip as one prompt, so there is no window to skip.
    pub vad_gate: bool,
    /// Continuous-batching width (`--max-batch`). Half of the serving envelope
    /// the will-it-fit estimate is priced against - KV cost is ctx × batch, so
    /// showing a VRAM figure without both is meaningless.
    pub max_batch: usize,
    /// Output cap for a request that sends none: the config's `max_tokens`
    /// (`--max-output-tokens`), else the context window. The window is what
    /// an omitted cap means everywhere else (OpenAI, vLLM, llama.cpp: the
    /// model stops or the window ends), and the engine clamps every run to
    /// the room its prompt leaves anyway. It was a flat 1024 until
    /// 2026-09-22: a reasoning model asked for "about 200 words" spent all of
    /// it counting words in its thinking and the reply came back with no
    /// content at all, a `length` finish, and a client showing the counting
    /// as the answer.
    pub default_max_output_tokens: usize,
    /// Hard ceiling that clamps a request's `max_output_tokens` (`None` = no
    /// clamp). Set on an exposed instance so a request can't demand a huge,
    /// costly generation regardless of what it asks for.
    pub max_output_ceiling: Option<usize>,
    /// This endpoint's SERVER TOOLS and web-search provider - a live view of
    /// the config file (see LiveConfig): control-plane state with no engine
    /// coupling, so a tool-list or search-key change must never cost a model
    /// restart. MCP servers here are callable by bare `server_label` (OpenAI)
    /// or name-only entry (Anthropic); callers' own inline servers always
    /// work regardless.
    pub live: LiveConfig,
    /// Per-client abuse limiter (in-memory). A no-op unless limits are set.
    pub rate_limiter: Arc<crate::ratelimit::RateLimiter>,
    /// MCP connection pool for the per-request `mcp_servers` connector (API
    /// conformance - lazy, shared across requests; see paddock-mcp). Stored
    /// MCP servers are a manager feature; the runner only ever connects to
    /// servers named inline in a request.
    pub mcp: std::sync::Arc<paddock_mcp::McpManager>,
    /// Pending human-in-the-loop MCP tool approvals (streaming agent loop ↔
    /// `/api/mcp-approvals/{id}`).
    pub approvals: std::sync::Arc<crate::responses::ApprovalGate>,
    /// Responses paused on MCP approval, resumed via `previous_response_id` +
    /// `mcp_approval_response` (the OpenAI spec approval flow).
    pub approval_store: std::sync::Arc<crate::responses::ApprovalStore>,
    /// Engine self-report sampler (inside view; no NVML) - read-only handle.
    pub stats: crate::stats::Stats,
    /// PDF rasterization caps (pages per file, target resolution). PDF content
    /// parts expand to page images through this; pdfium itself is linked in, so
    /// rendering is always available.
    pub pdf: crate::pdf::PdfConfig,
    /// Drain state (doc §5): set by the admin surface, read by the drain
    /// middleware and /healthz.
    pub drain: Arc<crate::drain::DrainCtl>,
    /// Per-request event ring (doc §8.1) - filled by events_mw + handlers,
    /// read by the admin surface's /v1/events subscription.
    pub events: Arc<crate::events::EventRing>,
    /// Session headers captured into event records (lowercase; configurable -
    /// default X-Session-ID + X-Litellm-Session-Id, the llama-swap set).
    pub session_headers: Vec<String>,
    /// Request filters (doc §13): aliases, param variants, strip/force
    /// enforcement. Always present; `Default` = everything off.
    pub filters: Arc<crate::filters::Filters>,
    /// Explicit admission cap (doc §13, llama-swap's `concurrencyLimit`):
    /// max in-flight inference requests before refusal - queue depth, distinct
    /// from `max_batch`'s compute width. `None` = uncapped (the default).
    pub concurrency_limit: Option<usize>,
    /// Forensic preprocessing runtime ([forensics] gate). `None` when disabled;
    /// when present, image attachments are run through paddock-forensics on their
    /// original bytes and the findings are injected into the model's context.
    pub forensics: Option<std::sync::Arc<crate::forensics::ForensicRuntime>>,
}

/// What bounds a request's generation: the server default for a request that
/// names no cap, and the deployment ceiling over both. One value handed to
/// every prepare() so the routes cannot drift apart on either half.
#[derive(Clone, Copy, Debug)]
pub struct OutputCaps {
    pub default: usize,
    pub ceiling: Option<usize>,
}

impl OutputCaps {
    /// The cap a request generates under: what it asked for, else the
    /// default, and never above the ceiling.
    pub fn resolve(self, asked: Option<usize>) -> usize {
        let n = asked.unwrap_or(self.default);
        self.ceiling.map_or(n, |c| n.min(c))
    }
}

impl AppState {
    pub fn residency(&self) -> Option<crate::residency::Snapshot> {
        self.asr
            .as_ref()
            .and_then(|m| m.transcriber.residency())
            .or_else(|| self.serving.as_ref().and_then(|m| m.engine.residency()))
    }

    pub fn output_caps(&self) -> OutputCaps {
        OutputCaps {
            default: self.default_max_output_tokens,
            ceiling: self.max_output_ceiling,
        }
    }

    /// Minimal state for integration tests: a served model (or none) and inert
    /// defaults for everything else. Tests must build state through this - a
    /// struct literal in a test rots every time `AppState` grows a field
    /// (that's how completions_http/sdk_conformance broke once).
    #[doc(hidden)]
    pub fn for_tests(serving: Option<ServingModel>) -> Self {
        AppState {
            auth_key: None,
            trusted_proxy: false,
            spec_off: false,
            instance_id: "test-instance".to_owned(),
            metrics: crate::metrics::Metrics::new(
                "test-instance".to_owned(),
                crate::metrics::ModelIds::default(),
                None,
            ),
            snapshots: crate::metrics::SnapshotRing::new(),
            metrics_auth: None,
            serving,
            sampling: SamplingDefaults::default(),
            embedder: None,
            asr: None,
            aligner: None,
            segmenter: None,
            masker: None,
            laya: None,
            clef: None,
            tabular: None,
            diarization: None,
            image: None,
            max_ctx: 8192,
            vad_gate: false,
            max_batch: 8,
            // the window, like a real state's (see the field)
            default_max_output_tokens: 8192,
            max_output_ceiling: None,
            rate_limiter: Arc::new(crate::ratelimit::RateLimiter::new(
                crate::ratelimit::Limits {
                    per_minute: None,
                    per_day: None,
                },
                false,
            )),
            mcp: Arc::new(paddock_mcp::McpManager::new()),
            approvals: Arc::new(crate::responses::ApprovalGate::default()),
            approval_store: Arc::new(crate::responses::ApprovalStore::default()),
            stats: crate::stats::Stats::disabled(),
            pdf: crate::pdf::PdfConfig {
                max_pages: 8,
                long_edge: 1568,
            },
            drain: Arc::new(crate::drain::DrainCtl::default()),
            events: crate::events::EventRing::new(),
            session_headers: crate::events::default_session_headers(),
            filters: Arc::new(crate::filters::Filters::default()),
            concurrency_limit: None,
            forensics: None,
            live: LiveConfig::fixed(LiveSnapshot {
                mcp_servers: Vec::new(),
                web_search: None,
            }),
        }
    }

    /// Same, for an ASR-only runner. A whisper checkpoint has no chat surface
    /// at all, so `serving` stays None and `/v1/audio/transcriptions` is the
    /// only thing that answers - which is exactly the shape the audio half of
    /// the spec gate needs (sdk_conformance.rs::spec_gate_whisper).
    #[doc(hidden)]
    pub fn for_tests_asr(asr: AsrModel) -> Self {
        AppState {
            asr: Some(asr),
            ..Self::for_tests(None)
        }
    }

    /// Same, for an image-generation runner: `/v1/images/*` and nothing else.
    #[doc(hidden)]
    pub fn for_tests_image(image: crate::serving::ImageModel) -> Self {
        AppState {
            image: Some(image),
            ..Self::for_tests(None)
        }
    }
}

/// Largest request body the runner will buffer. See the layer below for why
/// this is far above axum's 2 MB default and why it is still bounded.
const MAX_BODY: usize = 192 * 1024 * 1024;

pub fn router(state: Arc<AppState>) -> Router {
    // The inference API + operational endpoints. Every unmatched path gets the
    // OpenAI-shaped JSON 404, which SDKs parse - the runner has no SPA shell.
    Router::new()
        .route("/healthz", get(healthz))
        // The runner's own OpenAPI description - outside /v1 + /api on
        // purpose, like /healthz: a discovery document carries no secrets and
        // must be readable before a client has a key. NOTE: every route added
        // to this router must also be added to ../openapi.json - the
        // spec-drift test below fails when the spec names a dead path, and
        // this comment is the reminder for the direction a test cannot see.
        .route("/openapi.json", get(openapi_spec))
        // Prometheus exposition - benchmark harnesses scrape this on every
        // run by default; before the route existed it fell through to
        // the JSON 404 and the scraper silently disabled server-metrics for
        // the run. Auth posture lives in the handler (loopback open, network
        // keyed), not in auth_mw - /metrics is outside /v1 + /api deliberately.
        .route("/metrics", get(crate::metrics::handle))
        .route("/v1/models", get(list_models))
        .route("/api/server", get(server_info))
        .route("/api/extract", post(extract_preview))
        .route("/api/metadata", post(file_metadata))
        .route("/api/stats", get(stats_info))
        .route("/api/stats/stream", get(stats_stream))
        .route("/api/mcp-approvals/{id}", post(mcp_approve))
        .route("/v1/realtime", get(crate::realtime::handle))
        .route("/v1/completions", post(crate::completions::handle))
        .route("/v1/chat/completions", post(crate::chat::handle))
        .route("/v1/responses", post(crate::responses::handle))
        .route(
            "/v1/responses/compact",
            post(crate::responses::handle_compact),
        )
        .route("/v1/messages", post(crate::messages::handle))
        .route(
            "/v1/messages/count_tokens",
            post(crate::messages::count_tokens),
        )
        .route("/v1/embeddings", post(crate::embeddings::embeddings))
        .route("/v1/rerank", post(crate::embeddings::rerank))
        .route(
            "/v1/audio/transcriptions",
            post(crate::transcriptions::handle),
        )
        .route("/v1/audio/alignments", post(crate::alignments::handle))
        .route(
            "/v1/audio/diarizations/stream",
            get(crate::diarization_live::route),
        )
        .route(
            "/v1/audio/diarizations",
            post(crate::diarizations::handle)
                .layer(axum::extract::DefaultBodyLimit::max(25 * 1024 * 1024)),
        )
        .route("/v1/segmentations", post(crate::segmentations::handle))
        .route("/v1/masks", post(crate::masks::handle))
        .route("/v1/masks/sessions", post(crate::mask_sessions::create))
        .route(
            "/v1/masks/sessions/{id}/frames",
            post(crate::mask_sessions::frame),
        )
        .route(
            "/v1/masks/sessions/{id}/finish",
            post(crate::mask_sessions::finish),
        )
        .route(
            "/v1/masks/sessions/{id}",
            axum::routing::delete(crate::mask_sessions::delete),
        )
        .route("/v1/images/generations", post(crate::images::generations))
        .route("/v1/images/edits", post(crate::images::edits))
        .route("/v1/systemone", post(crate::systemone::handle))
        .route(
            "/v1/tabular/predictions",
            post(crate::tabular::handle)
                .layer(axum::extract::DefaultBodyLimit::max(8 * 1024 * 1024)),
        )
        .route(
            "/v1/tabular/contexts",
            post(crate::tabular::fit).layer(axum::extract::DefaultBodyLimit::max(8 * 1024 * 1024)),
        )
        .route(
            "/v1/tabular/contexts/{context_id}",
            axum::routing::delete(crate::tabular::release),
        )
        .fallback(not_found)
        // Axum's default body limit is 2 MB, which is below what a single
        // legitimate request carries here: a data-URI image inflates 4/3 in
        // base64, so an ordinary 3 MB screenshot was refused with "Failed to
        // buffer the request body" before any handler saw it - `detail: high`
        // was a knob the transport could not deliver. The manager already hit
        // this and raised its own /api limit to the 100 MB attachment cap; the
        // runner never did.
        //
        // MAX_BODY covers the client's own ceiling with the base64 expansion
        // applied (100 MB file -> ~133 MB on the wire) plus room for several
        // images in one turn. It is a LIMIT, not an allocation: only a request
        // that really is that big ever buffers that much. Kept explicit rather
        // than removed, because the API is reachable over the network and an
        // unbounded body is a memory-exhaustion surface.
        .layer(axum::extract::DefaultBodyLimit::max(MAX_BODY))
        // Request filters (doc §13) - innermost, so refused requests (drain/
        // auth/ratelimit) never buffer a body, while the unknown-variant 404
        // still gets its event record. No-op unless filters are configured.
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            crate::filters::filters_mw,
        ))
        // Per-client rate limiting on the generation endpoints (no-op unless
        // limits are configured).
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            ratelimit_mw,
        ))
        // Bearer auth over /v1 + /api (no-op when auth_key is None).
        .layer(axum::middleware::from_fn_with_state(state.clone(), auth_mw))
        // Drain gate + in-flight counting for inference requests (outside auth:
        // a draining runner 503s before key checks - see drain.rs).
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            crate::drain::drain_mw,
        ))
        // Event records (doc §8.1) - outside drain/auth so refused requests
        // (503/401/429) are recorded too; the record pushes when the response
        // BODY completes, so streaming durations are true.
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            crate::events::events_mw,
        ))
        // Stamp `charset=utf-8` on JSON responses (outermost, so it also covers
        // auth-rejection + 404 bodies). Non-JSON (SSE, wasm) is untouched.
        .layer(axum::middleware::map_response(json_utf8_charset))
        .with_state(state)
}

/// Add `charset=utf-8` to bare `application/json` responses. JSON is UTF-8 by
/// spec and browsers assume it, but lenient clients - notably PowerShell's
/// Invoke-RestMethod - default to Latin-1 without the charset and mangle
/// non-ASCII text. Only the exact `application/json` type is rewritten, so SSE
/// (`text/event-stream`) and already-charset'd responses are left alone.
async fn json_utf8_charset(mut res: Response) -> Response {
    use axum::http::header::CONTENT_TYPE;
    if res
        .headers()
        .get(CONTENT_TYPE)
        .is_some_and(|v| v.as_bytes() == b"application/json")
    {
        res.headers_mut().insert(
            CONTENT_TYPE,
            axum::http::HeaderValue::from_static("application/json; charset=utf-8"),
        );
    }
    res
}

/// Liveness + readiness in one: 200 "ok" while serving, 503 "draining" during
/// drain (the manager's health-gate and any LB-style check read this).
async fn healthz(State(state): State<Arc<AppState>>) -> Response {
    if state.drain.is_draining() {
        (StatusCode::SERVICE_UNAVAILABLE, "draining").into_response()
    } else {
        "ok".into_response()
    }
}

/// "What the model reads" - extraction preview for one attachment. Runs the
/// same lanes chat expansion runs (sift PDF text, scriptor docx, calamine
/// sheets, text-native decode, photo metadata) and returns the injection
/// text verbatim, so a UI can show the user exactly what a prompt carries -
/// metadata block included (`file_metadata: "off"` drops it, same contract
/// as the inference surfaces). Lane refusals (encrypted PDF, binary bytes)
/// come back as the same honest 400s a send would produce.
async fn extract_preview(
    State(state): State<Arc<AppState>>,
    Json(req): Json<serde_json::Value>,
) -> Response {
    let Some(data) = req.get("data").and_then(serde_json::Value::as_str) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorBody::new(
                "invalid_request_error",
                "missing `data` (data URI or base64)",
            )),
        )
            .into_response();
    };
    let with_meta = match crate::chat::file_metadata_on(
        req.get("file_metadata").and_then(serde_json::Value::as_str),
    ) {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorBody::new("invalid_request_error", &e)),
            )
                .into_response();
        }
    };
    let bytes = match crate::pdf::decode_pdf_payload(data) {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorBody::new("invalid_request_error", &e)),
            )
                .into_response();
        }
    };
    let filename = req
        .get("filename")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    let mime = crate::doc::data_uri_mime(data).map(str::to_owned);
    let max_ctx = state.max_ctx;
    // whether SENDING would take the render route here - the panel must
    // describe what this server actually does, not what the text lane thinks
    let can_render = state.serving.as_ref().is_some_and(|m| m.supports_vision)
        && crate::pdf::available(&state.pdf);
    let joined = tokio::task::spawn_blocking(move || {
        crate::doc::extract_preview(
            &bytes,
            filename.as_deref(),
            mime.as_deref(),
            with_meta,
            max_ctx,
            can_render,
        )
    })
    .await;
    match joined {
        Ok(Ok(p)) => Json(serde_json::json!({
            "text": p.text,
            "kind": p.kind,
            "pages": p.pages,
            // What this file adds to the SYSTEM turn. Absent for almost every
            // file; a panel called "what the model reads" that omitted it was
            // showing half the answer.
            "system": p.system,
        }))
        .into_response(),
        Ok(Err(e)) => (
            StatusCode::BAD_REQUEST,
            Json(ErrorBody::new("invalid_request_error", &e)),
        )
            .into_response(),
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorBody::new("internal_error", "extraction task panicked")),
        )
            .into_response(),
    }
}

/// "What is in this file" - every embedded metadata field, grouped
/// (EXIF/XMP/IPTC/ICC/PDF/QuickTime via sift, Office core/app/custom
/// properties via scriptor).
///
/// Distinct from `/api/extract`, which answers "what does the MODEL read" and
/// is per-server by nature (`max_ctx`, vision + pdfium, the `file_metadata`
/// toggle). This one depends on nothing but the bytes, which is why the
/// manager can answer the same question off a stored attachment with no model
/// loaded. Both call the same crate, so the two answers cannot
/// drift.
///
/// It lives on the runner because the runner is the API product: a client
/// pointing at a runner with no manager in the picture gets it too.
async fn file_metadata(Json(req): Json<serde_json::Value>) -> Response {
    let Some(data) = req.get("data").and_then(serde_json::Value::as_str) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorBody::new(
                "invalid_request_error",
                "missing `data` (data URI or base64)",
            )),
        )
            .into_response();
    };
    // Same decoder every attachment lane uses - data URI or bare base64, with
    // the same size ceiling.
    let bytes = match crate::pdf::decode_pdf_payload(data) {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorBody::new("invalid_request_error", &e)),
            )
                .into_response();
        }
    };
    // Parsing a document blocks, and hostile bytes are the whole input here:
    // its own task so a parser panic is a 500, not a dead runner.
    match tokio::task::spawn_blocking(move || paddock_filemeta::read(&bytes)).await {
        Ok(meta) => Json(meta).into_response(),
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorBody::new("internal_error", "metadata task panicked")),
        )
            .into_response(),
    }
}

/// Runtime facts a client (or the manager) needs: the served model, context
/// window, output caps, reasoning style, PDF capability. Set at launch, so a
/// caller can see which ranges are unreachable without a relaunch.
async fn server_info(State(state): State<Arc<AppState>>) -> Response {
    // Reasoning control the loaded model supports, so a UI can draw the right
    // thing. `reasoning` is the control SHAPE and keeps its three old values
    // ("effort" | "toggle" | "none"); the three fields beside it are what a
    // client needs to draw the shape honestly, and they are measured from the
    // served template rather than assumed from the family:
    //
    //   reasoning_levels  the rungs, in the model's own spelling - Qwen3.8
    //                     says xhigh, not high, and a picker that says "high"
    //                     is naming a level the model does not have
    //   reasoning_default which rung an unset request gets (its published
    //                     default), so a picker can open on the truth
    //   reasoning_off     whether reasoning can be turned off at all. gpt-oss
    //                     and muse render their preamble unconditionally, so
    //                     an Off item there would be a lie; Qwen3.8 has both a
    //                     ladder and an off position, which is the case the
    //                     old two-styles-only surface could not express.
    //   reasoning_preserve
    //                     whether the caller may decide if a prior turn's
    //                     thinking stays in the prompt (`preserve_thinking`).
    //                     Only the qwen3.6/3.8 templates grade it, so this is
    //                     what stops a client drawing that switch for a model
    //                     that would ignore it.
    let caps = state.serving.as_ref().map(|s| &s.reasoning);
    let reasoning = caps.map(|c| c.style());
    let dflt = state.sampling.resolve(true);
    // built outside the big literal below: one more nested object in it and
    // `json!` runs out of macro recursion
    let image_generation = state.image.as_ref().map(|m| {
        serde_json::json!({
            "size_multiple": m.service.info().size_multiple,
            "max_side": crate::images::MAX_SIDE,
            "default_size": format!("{0}x{0}", crate::images::DEFAULT_SIDE),
            "default_steps": crate::images::DEFAULT_STEPS,
            "max_steps": crate::images::MAX_STEPS,
            "max_n": crate::images::MAX_N,
            "edit": m.service.info().edit && m.image_pad_id.is_some(),
            "max_references": crate::images::MAX_REFERENCES,
            "stream": true,
            "max_partial_images": crate::images::MAX_PARTIAL_IMAGES,
            "output_formats": ["png", "webp", "jpeg"],
            "weights_bytes": m.service.info().weights_bytes,
        })
    });
    Json(serde_json::json!({
        "role": "runner",
        "version": env!("CARGO_PKG_VERSION"),
        "pid": std::process::id(),
        "max_ctx": state.max_ctx,
        "max_batch": state.max_batch,
        "default_max_output_tokens": state.default_max_output_tokens,
        "model": state.serving.as_ref().map(|s| s.id.clone()),
        "embedder": state.embedder.as_ref().map(|e| e.id.clone()),
        // Functional truth, same rule as /v1/models capabilities: rerank works
        // iff the vocab carries the yes/no relevance tokens. The Studio's
        // Embeddings page picks its MODE from this - without it every encoder
        // read as an embedder, and a reranker landed on the wrong page.
        "reranker": state.embedder.as_ref().map(|e| e.yes_id.is_some() && e.no_id.is_some()),
        // what an /v1/embeddings item may carry beside text - EmbeddingGemma
        // 2's towers, as built (the Embeddings page offers pictures / clips
        // only when the endpoint takes them)
        "embedder_inputs": state.embedder.as_ref().map(|e| {
            let mut kinds = vec!["text"];
            if e.image_budget.is_some() {
                kinds.push("image");
            }
            if e.audio {
                kinds.push("audio");
            }
            kinds
        }),
        "reasoning": reasoning,
        "reasoning_levels": caps.map(|c| c.levels.clone()),
        "reasoning_default": caps.and_then(|c| c.default_level.clone()),
        "reasoning_off": caps.map(|c| c.off),
        "reasoning_preserve": caps.map(|c| c.preserve),
        // Whether a thinking budget (reasoning.max_tokens /
        // thinking.budget_tokens) can be ENFORCED on this lane - same
        // dialect-shaped truth as /v1/models' capabilities.thinking_budget.
        // What stops a client drawing a budget control for gpt-oss/muse,
        // whose channel-structured reasoning refuses the knob.
        "thinking_budget": state.serving.as_ref().map(|s| {
            s.reasoning.reasons()
                && match s.dialect {
                    crate::parsers::Dialect::QwenXml | crate::parsers::Dialect::Laguna =>
                        s.tokenizer.token_to_id("</think>").is_some(),
                    crate::parsers::Dialect::GemmaChannel =>
                        s.tokenizer.token_to_id(crate::parsers::G_CLOSE).is_some(),
                    _ => false,
                }
        }),
        // The sampling a request gets when it sends no dials - surfaced so the
        // Studio's sampler popover can say what "default" actually is instead
        // of a blind word (a request field always wins). `source` names where
        // the numbers came: they are usually the model's
        // own published profile, not a house value, and a user comparing two
        // models should be able to see that without asking us.
        //
        // The model's DEFAULT mode is reported. A family that publishes a
        // second set for thinking-off (qwen3.5/3.6) samples that turn
        // differently, which the popover cannot say in one row; `source`
        // naming the card is the honest pointer.
        // `as_written` on every float: these are f32 and serde would widen them
        // to f64, publishing 0.949999988079071 for a published 0.95 - which the
        // composer's sampling popover then showed as the model's "default".
        "sampling": {
            "temperature": paddock_models::sampling::as_written(dflt.temp),
            "top_p": paddock_models::sampling::as_written(dflt.top_p),
            "top_k": dflt.top_k,
            "min_p": paddock_models::sampling::as_written(dflt.min_p),
            "repeat_penalty": paddock_models::sampling::as_written(dflt.repeat_penalty),
            "source": state.sampling.provenance(),
            "source_detail": state.sampling.provenance_detail(),
        },
        // Image input capability (mmproj loaded) - surfaced here as well as on
        // /v1/models so one relayed call gives the Studio the full model card.
        "vision": state.serving.as_ref().map(|s| s.supports_vision),
        // Document-parser-only model (deepseek2-ocr, paddleocr): text-only
        // chat 400s here, so the Studio's composer requires a document before
        // it sends. Same one-call model-card rule as `vision`.
        "document_parser": state.serving.as_ref().map(|s| s.document_parser),
        // Audio input capability (ASR mmproj loaded): POST
        // /v1/audio/transcriptions works iff this is true. Same one-call
        // model-card rule as `vision` - a client that has to probe the
        // endpoint with a request to learn it exists learns it by failing.
        // Either source serves it: a generative ASR model with its audio
        // mmproj, or a dedicated whisper-family model with no generative
        // model loaded at all - so this is a plain bool, not the serving
        // model's own flag.
        "audio": state.serving.as_ref().is_some_and(|s| s.supports_audio) || state.asr.is_some(),
        "realtime_transcription": crate::realtime::capabilities(
            state.asr.is_some() || state.serving.as_ref().is_some_and(|s| s.supports_audio),
            state.asr.is_some(),
            state.serving.as_ref().filter(|s| s.supports_audio).and_then(|s| s.audio_frontend.max_clip_s(state.max_ctx.saturating_sub(512))),
        ),
        "asr": state.asr.as_ref().map(|a| a.id.clone()),
        // Forced-alignment model id: POST /v1/audio/alignments works iff this
        // is set. Same one-call model-card rule as `asr` - an aligner-only
        // runner has no `model`, no `embedder`, no `asr`, and without its own
        // key it reads as serving nothing (the whisper lanes taught this).
        "aligner": state.aligner.as_ref().map(|a| a.id.clone()),
        // The longest clip one alignment call can address (the head's time-bin
        // budget) - surfaced so a client can split-or-skip before sending
        // bytes instead of learning the cap from a 400.
        "alignment_max_clip_s": state.aligner.as_ref().map(|a| a.max_clip_s),
        // Dense-prediction model id: POST /v1/segmentations works iff this is
        // set. Same one-call model-card rule - a segmentation-only runner has
        // none of the keys above. `segmentation` is everything a client needs
        // to build a valid request without probing: the chip it must send and
        // the rasters it will get back.
        "segmenter": state.segmenter.as_ref().map(|s| s.id.clone()),
        "segmentation": state.segmenter.as_ref().map(|s| {
            let i = s.segmenter.info();
            let c = &i.config;
            serde_json::json!({
                "input": {
                    "size": c.image_size,
                    "bands": c.band_names,
                    "dtype": "uint8",
                    "pixel_size_m": c.pixel_size_m,
                    "chip_bytes": c.image_size * c.image_size * c.channels,
                },
                "output": {
                    "size": c.out_size,
                    "pixel_size_m": c.out_pixel_size_m,
                    "class_names": c.class_names,
                    "regression": c.regression_name,
                },
                "epsg": c.epsg,
                "chips_per_pass": i.max_batch,
            })
        }),
        // Promptable segmentation (SAM 3): POST /v1/masks works iff this is
        // set, and `masks` is what a valid request looks like - the picture
        // ceiling, the exemplar-box cap, the prompt-token cap (Meta's tower
        // reads 32 with its start and end markers), whether click prompts are
        // served and how many clicks one takes, and the mask encoding.
        "masker": state.masker.as_ref().map(|m| m.id.clone()),
        "masks": state.masker.as_ref().map(|m| {
            let i = m.masker.info();
            serde_json::json!({
                "max_pixels": i.max_pixels,
                "max_boxes": i.max_boxes,
                "max_prompt_tokens": paddock_tokenizer::sam3::SAM3_CONTEXT - 2,
                "clicks": i.clicks,
                "max_points": if i.clicks {
                    paddock_engine::gpu_model::sam3::PVS_MAX_POINTS
                } else {
                    0
                },
                "mask_format": "coco_rle",
                // video sessions (/v1/masks/sessions): served or why not,
                // how many frames an output is held, the live-session cap
                "video": i.video_off.is_none(),
                "video_unavailable": i.video_off,
                "hold_frames": paddock_engine::gpu_model::sam3::HOTSTART_DELAY,
                "max_sessions": paddock_engine::masks::MAX_SESSIONS,
                "max_concepts": paddock_engine::gpu_model::sam3::MAX_CONCEPTS,
            })
        }),
        // Image generation: the served model and what a valid request looks
        // like - the size grid, the ceiling, the step default and the
        // encodings - so a client builds one without probing.
        "image_model": state.image.as_ref().map(|m| m.id.clone()),
        "image_generation": image_generation,
        // Structured decisions (POST /v1/systemone) serve iff the chat model
        // generates by block diffusion: the canvas width bounds how many
        // `id: label` lines one call can read, and the caps say so up front.
        "structured_read": state.serving.as_ref().filter(|s| s.engine.canvas_width() > 0).map(|s| {
            serde_json::json!({
                "canvas_width": s.engine.canvas_width(),
                "max_questions": crate::systemone::MAX_QUESTIONS,
                "max_samples": crate::systemone::MAX_SAMPLES,
                "max_steps": s.engine.canvas_read_steps(),
                "images": s.supports_vision && s.engine.canvas_images(),
                "conditional": true,
                "think": true,
                "types": ["noul", "choice", "score"],
            })
        // a decision model reads and does nothing else: the same key, its
        // own values (one read, one step, text only, no thought)
        }).or_else(|| state.laya.as_ref().map(|l| l.caps()))
          .or_else(|| state.clef.as_ref().map(|c| c.caps())),
        // Decision model id (Laya): POST /v1/systemone works iff this is set,
        // and it is the ONLY thing such a runner serves - the sixth role.
        "reader": state.laya.as_ref().map(|l| l.id.clone())
            .or_else(|| state.clef.as_ref().map(|c| c.id.clone())),
        "tabular": state.tabular.as_ref().map(|m| serde_json::json!({"model":m.id,"capabilities":m.capabilities()})),
        "diarization": state.diarization.as_ref().map(|m|serde_json::json!({"model":m.id,"capabilities":m.capabilities()})),
        // The longest clip TRANSCRIPTION can take, same reason and same shape
        // as the alignment cap above. Null means no ceiling worth
        // publishing: whisper windows a clip into 30 s pieces, so length costs
        // time and nothing else. A GENERATIVE ASR lane spends the whole clip
        // as prompt rows, so its ceiling is the context window - and it is
        // lower than people guess (Qwen3-ASR bills 13 rows a second: ~42 min
        // at 32k, ~10 at 8k). Learning that from a refusal after uploading and
        // decoding an hour of audio is exactly the shape this avoids.
        "transcription_max_clip_s": state
            .asr
            .is_none()
            .then(|| {
                state.serving.as_ref().filter(|s| s.supports_audio).and_then(|s| {
                    // leave the generation budget the handler itself reserves
                    s.audio_frontend.max_clip_s(state.max_ctx.saturating_sub(512))
                })
            })
            .flatten(),
        // Which `timestamp_granularities[]` this runner can answer - the same
        // computed truth /v1/models publishes, repeated here so the one relayed
        // model-card call tells a UI whether to offer times at all. Whisper
        // emits them as vocabulary tokens; the generative ASR families have no
        // timestamp vocabulary, and an empty list is how they say so instead of
        // a UI discovering it from a 400 (or, worse, offering a control that
        // returns an empty `segments`).
        // Whisper answers both; granite-speech-plus answers `word` only - it
        // writes word end times into its transcript when instructed to, but
        // that mode emits no punctuation, so there are no sentence boundaries
        // to cut segments on. Whisper first, because the handler picks the
        // whisper lane when a runner carries both.
        "timestamp_granularities": if let Some(asr) = &state.asr {
            asr.timestamp_granularities()
        } else if state.serving.as_ref().is_some_and(|s| s.audio_word_times) {
            serde_json::json!(["word"])
        } else {
            serde_json::json!([])
        },
        // What this runner can do about LANGUAGE: whether the lane
        // can name the spoken language at all, whether it returns a posterior
        // with it, whether `languages` biases that detection, and the exact
        // codes the loaded checkpoint declares. The last one retires a
        // hard-coded 99-language list in the Studio - a client should offer
        // the languages this model actually has, not the ones whisper-large
        // shipped with.
        "language_detection": crate::language::caps_json(
            state.asr.as_ref(),
            state.serving.as_ref(),
        ),
        // Which `include` values /v1/audio/transcriptions honours. `logprobs`
        // asks for per-word confidence, and every lane that transcribes can
        // answer it - whisper hangs the words on its segments, the generative
        // families answer top-level because they have no segments to hang
        // anything on. Advertised separately from
        // `timestamp_granularities` precisely because the two stopped being
        // the same question: "how sure were you" needs no clock.
        "include": if state.asr.is_some()
            || state.serving.as_ref().is_some_and(|s| s.supports_audio)
        {
            serde_json::json!(["logprobs"])
        } else {
            serde_json::json!([])
        },
        // What an image actually costs here, so a client can price one before
        // sending it and a picker can show real numbers instead of a guess.
        // Same reasoning as the PDF page cap above: a cap the caller cannot see
        // is a cap the caller finds out about by being surprised.
        "vision_budget": state.serving.as_ref().and_then(vision_budget_json),
        // Task tags the loaded model's template expands (granite-vision).
        // Advertised because they are unreachable otherwise: the only way in is
        // to type the exact literal, and nothing tells you the literals exist.
        "task_tags": state.serving.as_ref().map(|s| s.task_tags.clone()).unwrap_or_default(),
        // The deepseek2-ocr request surface: the mode vocabulary,
        // crop classes and the grounded-region extension, so a client can
        // offer the reading modes as controls without knowing the family.
        // Same unreachable-otherwise reasoning as task_tags - the modes are
        // this model's real interface, and nothing else lists them. null on
        // every other model.
        "ocr": state.serving.as_ref().and_then(|s| {
            if s.ocr {
                Some(crate::deepseek_ocr::caps_json())
            } else if s.paddleocr {
                Some(crate::paddle_ocr::caps_json())
            } else {
                None
            }
        }),
        // PDF attachments are always accepted: sift text extraction is
        // compiled in, so any model reads a PDF's text layer. `raster` says
        // whether a vision model additionally gets page images (pdfium
        // present); `max_pages` is the raster cap, so a client can warn
        // "N of M pages - first {max} sent".
        "pdf": {
            "enabled": true,
            "raster": crate::pdf::available(&state.pdf),
            "max_pages": state.pdf.max_pages,
        },
        // Forensics ([forensics] gate). null when the endpoint has it
        // disabled; an object when configured, carrying the always-on scope
        // (`auto`), whether the on-demand tool is exposed (`tool`), and whether
        // this model can actually use the findings (`vision`) - forensics is
        // VLM-coupled, so the Studio only offers the control on a vision model.
        // Same one-call model-card rule as `vision`/`web_search`.
        "forensics": state.forensics.as_ref().map(|f| serde_json::json!({
            "auto": f.auto_word(),
            "tool": f.tool,
            "vision": state.serving.as_ref().is_some_and(|s| s.supports_vision),
        })),
        "web_search": state.live.snapshot().web_search.is_some(),
        // The builtin clock tool - present wherever chat serves (it needs no
        // provider), so a client gates its declaration on this, which also
        // reads false on older runner builds that would 400 the tool type.
        "current_time": state.serving.is_some(),
        // the endpoint's server-tool MCP labels - clients (the Studio) build
        // per-model tool declarations from this, so a compare lane only ever
        // asks a server for tools it actually has
        "mcp_servers": state
            .live
            .snapshot()
            .mcp_servers
            .iter()
            .filter_map(|e| e.get("server_label").and_then(serde_json::Value::as_str).map(str::to_owned))
            .collect::<Vec<_>>(),
        // Filter surface (doc §13) so the Studio's model card shows the whole
        // selectable id set and the admission posture without a second call.
        "aliases": state.filters.aliases,
        "variants": state.filters.variants.iter().map(|v| v.name.clone()).collect::<Vec<_>>(),
        "concurrency_limit": state.concurrency_limit,
        "residency": state.residency(),
    }))
    .into_response()
}

/// The served tower's image budget as JSON, or None on a model without one.
///
/// Deliberately reports the LEVELS as well as the raw numbers: a client should
/// be able to render "high - 3.2k tokens" without reimplementing our detail
/// policy, and if it does reimplement it, `auto_max_tokens` is the one number
/// that keeps the two in step.
pub(crate) fn vision_budget_json(s: &crate::serving::ServingModel) -> Option<serde_json::Value> {
    if !s.supports_vision {
        return None;
    }
    let b = s.engine.vision_budget()?;
    Some(serde_json::json!({
        "max_pixels": b.max_pixels,
        "min_pixels": b.min_pixels,
        "max_edge": b.max_edge,
        // tokens ~= ceil(w * h / pixels_per_token), clamped to [min, max]
        "pixels_per_token": b.pixels_per_token,
        "max_tokens": b.max_tokens,
        "min_tokens": b.min_tokens,
        // what `detail` resolves to on this model - `auto` is the default and
        // is a token cap, not a pixel one, so it means the same thing across
        // families whose pixels-per-token differ by 5x
        "auto_max_tokens": paddock_engine::generator::AUTO_MAX_TOKENS.min(b.max_tokens),
        "detail_levels": ["auto", "low", "high"],
    }))
}

/// A path as the host OS spells it. Windows accepts `/` and `PathBuf::display`
/// echoes back whatever the config wrote, so a `paddock.toml` with forward
/// slashes surfaced `D:/models` in the UI - technically valid, visibly
/// foreign. Anything shown to a user, or copied to paste into a shell, should
/// look native.
pub fn native_path(p: &std::path::Path) -> String {
    let s = p.display().to_string();
    if cfg!(windows) {
        s.replace('/', "\\")
    } else {
        s
    }
}

/// The runner's self-report: engine counters + allocator-ledger VRAM (the
/// inside view of doc §9). Cheap: returns the last background sample, never
/// blocks on the engine and never touches inference.
async fn stats_info(State(state): State<Arc<AppState>>) -> Response {
    Json(&*state.stats.latest()).into_response()
}

/// Live engine stats over WebSocket: pushes the current snapshot on connect,
/// then every new sample. No polling. Subscribing ramps the sampler's cadence.
async fn stats_stream(State(state): State<Arc<AppState>>, ws: WebSocketUpgrade) -> Response {
    let rx = state.stats.subscribe();
    ws.on_upgrade(move |socket| stats_ws(socket, rx))
}

async fn stats_ws(
    mut socket: WebSocket,
    mut rx: tokio::sync::watch::Receiver<std::sync::Arc<crate::stats::StatsSnapshot>>,
) {
    loop {
        // Serialize the latest snapshot; scope the borrow so it never crosses an
        // await point (watch Ref must not be held across .await).
        let payload = {
            let snap = rx.borrow_and_update().clone();
            serde_json::to_string(&*snap).unwrap_or_else(|_| "{}".to_owned())
        };
        if socket.send(Message::Text(payload.into())).await.is_err() {
            return; // client gone
        }
        // Wait for the next sample, but also notice the client hanging up.
        tokio::select! {
            changed = rx.changed() => {
                if changed.is_err() {
                    return; // sampler stopped (shutdown)
                }
            }
            msg = socket.recv() => {
                match msg {
                    None | Some(Ok(Message::Close(_))) | Some(Err(_)) => return,
                    _ => {}
                }
            }
        }
    }
}

/// Resolve a pending human-in-the-loop MCP approval. The streaming agent loop
/// is parked on it; body `{ "approve": bool }`. Lives on the runner because the
/// parked loop is in-process state - the manager's Studio calls this as an
/// ordinary API client. 404 when the id is unknown (already decided, timed
/// out, or the stream ended).
async fn mcp_approve(
    State(s): State<Arc<AppState>>,
    axum::extract::Path(id): axum::extract::Path<String>,
    Json(body): Json<serde_json::Value>,
) -> Response {
    let approve = body
        .get("approve")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    if s.approvals.resolve(&id, approve) {
        Json(serde_json::json!({ "ok": true, "approved": approve })).into_response()
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorBody::new(
                "not_found",
                "no pending approval with that id",
            )),
        )
            .into_response()
    }
}

/// The runner serves exactly one model (plus an encoder companion where the
/// pipeline has one) - /v1/models lists what this endpoint serves, not a
/// directory scan. Clients select a model by endpoint (doc §5); the catalog
/// of installed files is the manager's to list.
///
/// Entries carry the capability-metadata convention llama-swap seeded (doc
/// §13): architecture/capabilities/supported_parameters/context_length -
/// COMPUTED from what is actually loaded, never hand-declared, so the listing
/// can't lie. Declared aliases ride the served record; parameter variants
/// (doc §13, `<model>:high`) are listed as selectable entries of their own.
async fn list_models(State(state): State<Arc<AppState>>) -> Response {
    let mut data: Vec<ModelObject> = Vec::new();
    if let Some(s) = &state.serving {
        let mut input_modalities = vec!["text"];
        if s.supports_vision {
            input_modalities.push("image");
        }
        if s.supports_audio {
            input_modalities.push("audio");
        }
        let architecture = serde_json::json!({
            "input_modalities": input_modalities,
            "output_modalities": ["text"],
            "modality": format!("{}->text", input_modalities.join("+")),
        });
        // Tool calling is a chat-dialect feature, not a transcript feature;
        // vision/transcription only
        // when the matching mmproj is actually attached. `transcription` is the
        // functional truth for POST /v1/audio/transcriptions - same computed-
        // never-declared rule as everything else in this listing.
        let mut capabilities = serde_json::json!({
            "function_calling": s.dialect.supports_tools(),
            "vision": s.supports_vision,
            // A document parser reads pages and nothing else: text-only chat
            // is refused with a 400 (its decoder free-runs noise on a bare
            // prompt), so a client should gate its composer on this.
            "document_parser": s.document_parser,
            "transcription": s.supports_audio,
            "realtime_transcription": crate::realtime::capabilities(s.supports_audio, false, s.audio_frontend.max_clip_s(state.max_ctx.saturating_sub(512))),
            // Usually empty even when transcription works: Qwen3-ASR and the
            // base granite-speech emit a bare transcript with no timestamp
            // vocabulary at all, so there is no granularity they can answer,
            // and saying so beats a caller finding out from a 400.
            // granite-speech-plus is the exception - asking it to time words
            // makes it write the times into its transcript  - but
            // `segment` stays out of reach even there: that mode emits no
            // punctuation, so there are no sentences to cut cues on.
            "timestamp_granularities": if s.audio_word_times { vec!["word"] } else { vec![] },
        });
        // Only on a lane that transcribes - `caps_json` returns None
        // otherwise, and a null capability is noise on a chat model.
        if let Some(l) = crate::language::caps_json(None, Some(s)) {
            capabilities["language_detection"] = l;
        }
        // The request fields this model actually honors - `reasoning_effort`
        // is listed iff its own template implements a reasoning control at
        // all: a graded ladder (gpt-oss, muse-glimmer, Qwen3.8), an on/off
        // switch (Qwen3.5/3.6, gemma4, laguna) or both. Everything else
        // rejects it honestly rather than accepting it as a no-op.
        let mut params: Vec<String> = [
            "chat_template_kwargs",
            "frequency_penalty",
            "ignore_eos",
            "logit_bias",
            "logprobs",
            "max_tokens",
            "min_p",
            "n",
            "parallel_tool_calls",
            "presence_penalty",
            "repeat_last_n",
            "repeat_penalty",
            "response_format",
            "seed",
            "stop",
            "stream",
            "stream_options",
            "temperature",
            "tool_choice",
            "tools",
            "top_k",
            "top_p",
        ]
        .map(str::to_owned)
        .into();
        if !s.dialect.supports_tools() {
            params
                .retain(|p| !matches!(p.as_str(), "tools" | "tool_choice" | "parallel_tool_calls"));
        }
        if s.reasoning.reasons() {
            params.push("reasoning_effort".to_owned());
            params.sort();
        }
        // The rungs belong in the listing too: `supported_parameters` says the
        // field is accepted, and this says what to put in it. Absent on a model
        // with only a switch - there is no vocabulary to publish there.
        if !s.reasoning.levels.is_empty() {
            capabilities["reasoning_effort"] = serde_json::json!({
                "levels": s.reasoning.levels,
                "default": s.reasoning.default_level,
                "off": s.reasoning.off,
            });
        }
        // Whether a thinking budget (`reasoning.max_tokens` here,
        // `thinking.budget_tokens` on /v1/messages) can be ENFORCED on this
        // lane: the model reasons AND its dialect has a single-token
        // think-close the runner can force (see `chat::think_budget`).
        // gpt-oss and muse reason in channel structures and refuse the knob
        // honestly, so a client should gate its budget control on this.
        capabilities["thinking_budget"] = serde_json::json!(
            s.reasoning.reasons()
                && match s.dialect {
                    crate::parsers::Dialect::QwenXml | crate::parsers::Dialect::Laguna =>
                        s.tokenizer.token_to_id("</think>").is_some(),
                    crate::parsers::Dialect::GemmaChannel =>
                        s.tokenizer.token_to_id(crate::parsers::G_CLOSE).is_some(),
                    _ => false,
                }
        );
        if s.ocr || s.paddleocr {
            // the document-parser request object - named here, and its
            // vocabulary spelled out in `capabilities` (the parameter name
            // alone leaves the modes discoverable by 400)
            params.push("ocr".to_owned());
            params.sort();
            capabilities["ocr"] = if s.ocr {
                crate::deepseek_ocr::caps_json()
            } else {
                crate::paddle_ocr::caps_json()
            };
        }
        data.push(
            ModelObject::new(s.id.clone(), 0, "paddock")
                .with_vision(s.supports_vision)
                .with_vision_budget(vision_budget_json(s))
                .with_task_tags(&s.task_tags)
                .with_listing_meta(architecture, capabilities, params, state.max_ctx)
                .with_aliases(state.filters.aliases.clone()),
        );
        for v in &state.filters.variants {
            data.push(
                ModelObject::new(format!("{}:{}", s.id, v.name), 0, "paddock")
                    .as_variant(s.id.clone(), serde_json::Value::Object(v.params.clone())),
            );
        }
    }
    if let Some(e) = &state.embedder {
        let capabilities = serde_json::json!({
            "embeddings": true,
            // functional truth: rerank works iff the vocab carries the yes/no
            // relevance tokens, not what the file names itself
            "reranker": e.yes_id.is_some() && e.no_id.is_some(),
        });
        let mut inputs = vec!["text"];
        if e.image_budget.is_some() {
            inputs.push("image");
        }
        if e.audio {
            inputs.push("audio");
        }
        let architecture = serde_json::json!({
            "input_modalities": inputs,
            "output_modalities": [],
            "modality": format!("{}->embedding", inputs.join("+")),
        });
        data.push(
            ModelObject::new(e.id.clone(), 0, "paddock").with_listing_meta(
                architecture,
                capabilities,
                if e.embedding_gemma2 {
                    vec![
                        "dimensions".to_owned(),
                        "encoding_format".to_owned(),
                        "task".to_owned(),
                    ]
                } else {
                    vec!["encoding_format".to_owned()]
                },
                e.max_input_tokens,
            ),
        );
    }
    if let Some(a) = &state.asr {
        // speech-to-text only: no chat, no completions, no embeddings - say
        // exactly that rather than let a client infer a generative surface
        data.push(
            ModelObject::new(a.id.clone(), 0, "paddock").with_listing_meta(
                serde_json::json!({
                    "input_modalities": ["audio"],
                    "output_modalities": ["text"],
                    "modality": "audio->text",
                }),
                serde_json::json!({
                    "transcription": true,
                    "realtime_transcription": crate::realtime::capabilities(true, true, None),
                    // Which `timestamp_granularities[]` this model can actually
                    // answer, so a UI knows what to offer instead of discovering
                    // it from a 400. Two mechanisms, both real on this family:
                    // `segment` from whisper's own timestamp VOCABULARY,
                    // `word` from cross-attention DTW over a second,
                    // teacher-forced pass. A model with neither says
                    // so with an empty list, never with a silently-empty array.
                    "timestamp_granularities": a.timestamp_granularities(),
                    // The trained language-detection pass, its posterior, and the
                    // exact language set this checkpoint declares  -
                    // read out of the file's own map, so a converted checkpoint
                    // with a shorter set says so instead of being offered 99.
                    "language_detection": crate::language::caps_json(Some(a), None),
                }),
                vec![
                    "language".to_owned(),
                    // OpenAI's plural: the caller's candidate languages, applied
                    // here as a soft prior over detection (biasing, never
                    // filtering). Only on this lane - the generative ASR families
                    // have no detection to bias and refuse it by name.
                    "languages".to_owned(),
                    "response_format".to_owned(),
                    "timestamp_granularities".to_owned(),
                ],
                a.max_tokens,
            ),
        );
    }
    if let Some(al) = &state.aligner {
        // forced alignment only: audio + an existing transcript in, word
        // times out - /v1/audio/alignments and nothing else
        data.push(
            ModelObject::new(al.id.clone(), 0, "paddock").with_listing_meta(
                serde_json::json!({
                    "input_modalities": ["audio", "text"],
                    "output_modalities": ["text"],
                    "modality": "audio+text->timestamps",
                }),
                serde_json::json!({
                    "alignment": true,
                    // Effective backend limit, which may be below the bin budget.
                    "alignment_max_clip_s": al.max_clip_s,
                }),
                vec!["language".to_owned(), "text".to_owned()],
                al.max_ctx,
            ),
        );
    }
    if let Some(sg) = &state.segmenter {
        // dense prediction only: image chips in, rasters out -
        // /v1/segmentations and nothing else. No context window: the listing
        // slot carries the chip side instead of inventing a token count.
        let c = &sg.segmenter.info().config;
        data.push(
            ModelObject::new(sg.id.clone(), 0, "paddock").with_listing_meta(
                serde_json::json!({
                    "input_modalities": ["image"],
                    "output_modalities": ["raster"],
                    "modality": "image->raster",
                }),
                serde_json::json!({
                    "segmentation": true,
                    "segmentation_classes": c.class_names,
                    "segmentation_regression": c.regression_name,
                }),
                vec!["response_format".to_owned(), "logits".to_owned()],
                0,
            ),
        );
    }
    if let Some(mk) = &state.masker {
        // promptable segmentation only: a picture and a prompt in, every
        // instance's mask out (or one clicked object's) - /v1/masks and
        // nothing else
        let mut params: Vec<String> = ["text", "boxes", "threshold", "max_instances"]
            .map(str::to_owned)
            .into();
        if mk.masker.info().clicks {
            params.extend(["points", "object_box", "multimask", "refine"].map(str::to_owned));
        }
        data.push(
            ModelObject::new(mk.id.clone(), 0, "paddock").with_listing_meta(
                serde_json::json!({
                    "input_modalities": ["image", "text"],
                    "output_modalities": ["mask"],
                    "modality": "image+text->mask",
                }),
                serde_json::json!({
                    "masks": true,
                    "mask_sessions": mk.masker.info().video_off.is_none(),
                }),
                params,
                0,
            ),
        );
    }
    if let Some(cm) = &state.clef {
        // a decision model only: one joint sequence per request, every
        // question answered in the same pass - /v1/systemone and nothing else
        data.push(
            ModelObject::new(cm.id.clone(), 0, "paddock").with_listing_meta(
                serde_json::json!({
                    "input_modalities": ["text"],
                    "output_modalities": ["decision"],
                    "modality": "text->decision",
                }),
                serde_json::json!({"structured_read": true}),
                vec![
                    "model".to_owned(),
                    "ask".to_owned(),
                    "max_state_tokens".to_owned(),
                ],
                cm.max_len(),
            ),
        );
    }
    if let Some(lm) = &state.laya {
        // a decision model only: a state and typed questions in, calibrated
        // answers out - /v1/systemone and nothing else. Its context is the
        // longest sequence a checkpoint reads (the question + options + one
        // window of the state), not a chat window.
        let max_len = lm
            .decider
            .info()
            .checkpoints
            .iter()
            .map(|(_, c)| c.max_len)
            .max()
            .unwrap_or(0);
        data.push(
            ModelObject::new(lm.id.clone(), 0, "paddock").with_listing_meta(
                serde_json::json!({
                    "input_modalities": ["text"],
                    "output_modalities": ["decision"],
                    "modality": "text->decision",
                }),
                serde_json::json!({
                    "structured_read": true,
                    "checkpoints": lm.checkpoints().iter().map(|c| c.name()).collect::<Vec<_>>(),
                }),
                vec!["model".to_owned(), "lang".to_owned(), "ask".to_owned()],
                max_len,
            ),
        );
    }
    if let Some(m) = &state.diarization {
        data.push(ModelObject::new(m.id.clone(),0,"paddock").with_listing_meta(
            serde_json::json!({"input_modalities":["audio"],"output_modalities":["speaker_segments"],"modality":"audio->speaker_segments"}),
            m.capabilities(),vec!["file".into(),"model".into(),"preset".into(),"threshold".into()],0));
    }
    if let Some(m) = &state.tabular {
        data.push(ModelObject::new(m.id.clone(),0,"paddock").with_listing_meta(
            serde_json::json!({"input_modalities":["table"],"output_modalities":["prediction"],"modality":"table->prediction"}),
            m.capabilities(),vec!["context".into(),"targets".into(),"query".into(),"categorical".into(),"preprocessing".into()],0));
    }
    if let Some(im) = &state.image {
        // image generation only: text in, an image out - /v1/images/* and
        // nothing else. No context window; the listing says so with a 0.
        data.push(
            ModelObject::new(im.id.clone(), 0, "paddock").with_listing_meta(
                serde_json::json!({
                    "input_modalities": ["text"],
                    "output_modalities": ["image"],
                    "modality": "text->image",
                }),
                serde_json::json!({
                    "image_generation": true,
                    "image_edit": false,
                    "stream": true,
                    "size_multiple": im.service.info().size_multiple,
                    "max_side": crate::images::MAX_SIDE,
                    "default_steps": crate::images::DEFAULT_STEPS,
                    "output_formats": ["png", "webp", "jpeg"],
                }),
                crate::images::SUPPORTED_PARAMETERS
                    .iter()
                    .map(|s| (*s).to_owned())
                    .collect(),
                0,
            ),
        );
    }
    Json(ModelList::new(data)).into_response()
}

/// The embedded OpenAPI 3.1 description, with the crate version stamped in.
/// Hand-curated (crates/paddock-runner/openapi.json) rather than generated:
/// the OpenAI/Anthropic-compatible operations are specified by their
/// references and only carry deviation notes here, so derive-macro generation
/// would re-specify what we deliberately do not own. The spec-drift test
/// keeps the curated file honest against the router.
async fn openapi_spec() -> Response {
    static SPEC: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let body = SPEC.get_or_init(|| {
        include_str!("../openapi.json").replace("__VERSION__", env!("CARGO_PKG_VERSION"))
    });
    (
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        body.clone(),
    )
        .into_response()
}

pub(crate) async fn not_found(uri: axum::http::Uri) -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(ErrorBody::not_found(format!("route {uri}"))),
    )
        .into_response()
}

/// The expensive (generation) endpoints the rate limiter guards. Reads
/// (`/v1/models`, `/api/*`) and the WS are cheap and left unthrottled here.
/// (Also the set the request filters transform - filters.rs.)
pub(crate) fn is_generation_path(path: &str) -> bool {
    matches!(
        path,
        "/v1/completions"
            | "/v1/chat/completions"
            | "/v1/responses"
            | "/v1/responses/compact"
            | "/v1/messages"
            | "/v1/images/generations"
            | "/v1/images/edits"
            | "/v1/systemone"
            | "/v1/tabular/predictions"
            | "/v1/audio/diarizations"
            | "/v1/audio/diarizations/stream"
            | "/v1/tabular/contexts"
    )
}

/// Per-client rate limiting for the generation endpoints. A no-op unless limits
/// are configured. On refusal returns an OpenAI-shaped 429 + `Retry-After`.
async fn ratelimit_mw(
    State(state): State<Arc<AppState>>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    if state.rate_limiter.is_enabled() && is_generation_path(req.uri().path()) {
        let peer = req
            .extensions()
            .get::<axum::extract::ConnectInfo<std::net::SocketAddr>>()
            .map(|ci| ci.0);
        if let Some(ip) = state.rate_limiter.client_key(req.headers(), peer)
            && let Err(reason) = state.rate_limiter.check(ip, std::time::Instant::now())
        {
            let (retry, msg) = match reason {
                crate::ratelimit::Reject::PerMinute => (
                    "60",
                    "Rate limit exceeded. Please wait a minute and try again.",
                ),
                crate::ratelimit::Reject::PerDay => {
                    ("3600", "Daily demo limit reached. Please try again later.")
                }
            };
            let mut res = (
                StatusCode::TOO_MANY_REQUESTS,
                Json(ErrorBody::new("rate_limit_exceeded", msg)),
            )
                .into_response();
            res.headers_mut().insert(
                axum::http::header::RETRY_AFTER,
                axum::http::HeaderValue::from_static(retry),
            );
            return res;
        }
    }
    next.run(req).await
}

/// Is this request from the box itself - a loopback peer, with nothing saying
/// a proxy stood in between? Two things say so: `trusted_proxy` (the operator
/// declared one, so every caller is 127.0.0.1 and the peer address means
/// nothing), and a forwarding header on the request itself (`X-Forwarded-For`,
/// `X-Real-IP`, `Forwarded` - what caddy and traefik add by default, nginx
/// only when told to). Honouring a forged header is safe in this direction:
/// it can only cost a caller its exemption, never grant one. A missing
/// ConnectInfo reads as remote - when in doubt, require the key.
pub(crate) fn peer_is_local(req: &axum::extract::Request, trusted_proxy: bool) -> bool {
    if trusted_proxy {
        return false;
    }
    let loopback = req
        .extensions()
        .get::<axum::extract::ConnectInfo<std::net::SocketAddr>>()
        .is_some_and(|ci| ci.0.ip().is_loopback());
    loopback
        && !["x-forwarded-for", "x-real-ip", "forwarded"]
            .iter()
            .any(|h| req.headers().contains_key(*h))
}

/// Bearer auth for `/v1` + `/api` - but only for NETWORK callers. Loopback
/// peers are exempt (`peer_is_local`): the runner binds all interfaces by
/// default (a serving product that hides on localhost fights its own tier-1
/// workload), and the key is what stands between the LAN and the model. This
/// split keeps local tools keyless while every non-loopback caller must
/// present the configured key - the runner has no key store, so there is
/// exactly one (doc §5.1: the manager issues it at spawn and sends it on its
/// own relay calls; inference keys never grant admin ops, which live on the
/// local admin surface, not TCP). Behind a reverse proxy on the same host
/// every caller is loopback, which is why the exemption switches off the
/// moment anything says a proxy is there.
async fn auth_mw(
    State(state): State<Arc<AppState>>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    if let Some(required) = &state.auth_key {
        let local = peer_is_local(&req, state.trusted_proxy);
        let path = req.uri().path();
        if !local && (path.starts_with("/v1/") || path.starts_with("/api/")) {
            let ok = req
                .headers()
                .get(axum::http::header::AUTHORIZATION)
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.strip_prefix("Bearer "))
                .is_some_and(|k| k == required);
            if !ok {
                return (
                    StatusCode::UNAUTHORIZED,
                    Json(ErrorBody::new(
                        "invalid_api_key",
                        "missing or invalid API key",
                    )),
                )
                    .into_response();
            }
        }
    }
    next.run(req).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    async fn body_json(res: Response) -> serde_json::Value {
        let bytes = res.into_body().collect().await.expect("body").to_bytes();
        serde_json::from_slice(&bytes).expect("json body")
    }

    fn test_router() -> Router {
        router(Arc::new(AppState::for_tests(None)))
    }

    #[test]
    fn an_omitted_output_cap_is_the_window_and_the_ceiling_bounds_both() {
        // the request said nothing: it runs to the window (the engine clamps
        // to the room its prompt leaves), never to a small fixed number
        let caps = OutputCaps {
            default: 65_536,
            ceiling: None,
        };
        assert_eq!(caps.resolve(None), 65_536);
        assert_eq!(caps.resolve(Some(300)), 300);
        // a configured default is what an omitted cap gets instead
        let caps = OutputCaps {
            default: 2_048,
            ceiling: None,
        };
        assert_eq!(caps.resolve(None), 2_048);
        // the deployment ceiling clamps the default and the ask alike
        let caps = OutputCaps {
            default: 65_536,
            ceiling: Some(4_096),
        };
        assert_eq!(caps.resolve(None), 4_096);
        assert_eq!(caps.resolve(Some(100_000)), 4_096);
        assert_eq!(caps.resolve(Some(16)), 16);
        // the Responses wire leaves the field absent, not defaulted, so the
        // server decides (it used to deserialize straight to 1024)
        let req: paddock_api::responses::ResponsesRequest =
            serde_json::from_value(serde_json::json!({"model": "m", "input": "hi"})).unwrap();
        assert_eq!(req.max_output_tokens, None);
    }

    #[test]
    fn an_unpinned_server_samples_the_model_at_its_published_numbers() {
        // the whole point: nobody configured anything, so the
        // checkpoint's own card decides - not temp 1.0 with no truncation
        let sd =
            SamplingDefaults::for_model(&crate::config::Config::default(), Some("qwen35"), None);
        let d = sd.resolve(true);
        assert_eq!(d.temp, 1.0);
        assert_eq!(d.top_k, 20);
        assert_eq!(d.top_p, 0.95);
        assert_eq!(d.min_p, 0.0);
        // no election publishes a repetition penalty, so it stays off
        assert_eq!(d.repeat_penalty, 1.0);
        assert!(sd.provenance().contains("published defaults"));
        assert!(
            sd.provenance_detail().is_some_and(|c| c.contains("Qwen")),
            "the citation must name the artifact these numbers came from"
        );
    }

    #[test]
    fn the_thinking_toggle_picks_the_row_the_card_published_for_it() {
        let sd =
            SamplingDefaults::for_model(&crate::config::Config::default(), Some("qwen35"), None);
        assert_eq!(sd.resolve(false).temp, 0.7);
        assert_eq!(sd.resolve(false).top_p, 0.8);
        // a family that published one set is unmoved by the toggle
        let g =
            SamplingDefaults::for_model(&crate::config::Config::default(), Some("gemma4"), None);
        assert_eq!(g.resolve(true), g.resolve(false));
        assert_eq!(g.resolve(true).top_k, 64);
    }

    #[test]
    fn an_operator_pin_beats_the_election_knob_by_knob() {
        // --top-k 0 means "I want no top-k on this deployment", and it must
        // survive a model whose card asks for 20 - while the knobs the
        // operator did not pin stay elected
        let mut cfg = crate::config::Config {
            top_k: Some(0),
            ..Default::default()
        };
        let sd = SamplingDefaults::for_model(&cfg, Some("qwen35"), None);
        assert_eq!(sd.resolve(true).top_k, 0);
        assert_eq!(
            sd.resolve(true).top_p,
            0.95,
            "an unpinned knob stays elected"
        );
        assert!(sd.provenance().contains("server settings"));

        cfg.temp = Some(0.0);
        let greedy = SamplingDefaults::for_model(&cfg, Some("qwen35"), None);
        assert_eq!(
            greedy.resolve(true).temp,
            0.0,
            "--temp 0 must still mean greedy"
        );
    }

    #[test]
    fn a_checkpoint_that_publishes_nothing_keeps_the_openai_wire() {
        for arch in ["gpt-oss", "granite"] {
            let sd =
                SamplingDefaults::for_model(&crate::config::Config::default(), Some(arch), None);
            let d = sd.resolve(true);
            assert_eq!(
                (d.temp, d.top_k, d.top_p, d.min_p),
                (1.0, 0, 1.0, 0.0),
                "{arch}"
            );
            // and it SAYS so rather than passing the wire values off as the
            // model's own
            assert!(
                sd.provenance().contains("publishes none of its own"),
                "{arch}"
            );
            assert!(sd.provenance_detail().is_none(), "{arch}");
        }
        // and a server with no generative model at all is the same
        let none = SamplingDefaults::for_model(&crate::config::Config::default(), None, None);
        assert_eq!(none.resolve(true).top_k, 0);
    }

    #[tokio::test]
    async fn server_info_says_where_its_sampling_came_from() {
        // "no silent failures" applied to a number the user never typed: the
        // Studio's popover has to be able to name the source
        let res = test_router()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/api/server")
                    .body(axum::body::Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        let v = body_json(res).await;
        assert_eq!(v["sampling"]["temperature"], 1.0);
        assert!(
            v["sampling"]["source"]
                .as_str()
                .is_some_and(|s| !s.is_empty()),
            "sampling must name its provenance, got {:?}",
            v["sampling"]
        );
    }

    #[tokio::test]
    async fn v1_models_lists_only_the_served_model() {
        // No model loaded -> an empty OpenAI-shaped list, never a dir scan.
        let res = test_router()
            .oneshot(
                axum::http::Request::get("/v1/models")
                    .body(axum::body::Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(res.status(), StatusCode::OK);
        let json = body_json(res).await;
        assert_eq!(json["object"], "list");
        assert_eq!(json["data"], serde_json::json!([]));
    }

    /// The scraper conformance gate at the router level: a plain
    /// GET /metrics - no Accept header, no auth, exactly what a benchmark
    /// scraper sends - must answer classic Prometheus exposition, not the
    /// JSON 404 it used to fall through to.
    #[tokio::test]
    async fn metrics_answers_classic_prometheus_at_vanilla_defaults() {
        let res = test_router()
            .oneshot(
                axum::http::Request::get("/metrics")
                    .body(axum::body::Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(res.status(), StatusCode::OK);
        let ct = res
            .headers()
            .get(axum::http::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_owned();
        assert!(
            ct.starts_with("text/plain; version=0.0.4"),
            "content-type: {ct}"
        );
        let body = res.into_body().collect().await.expect("body").to_bytes();
        let text = std::str::from_utf8(&body).expect("utf8");
        assert!(text.contains("process_start_time_seconds "), "{text}");
        assert!(
            !text.contains("# EOF"),
            "classic format must not carry the OpenMetrics EOF"
        );

        // ...and Prometheus asking for OpenMetrics gets it, exemplar-capable.
        let res = test_router()
            .oneshot(
                axum::http::Request::get("/metrics")
                    .header("accept", "application/openmetrics-text; version=1.0.0")
                    .body(axum::body::Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        let ct = res
            .headers()
            .get(axum::http::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_owned();
        assert!(
            ct.starts_with("application/openmetrics-text"),
            "content-type: {ct}"
        );
        let body = res.into_body().collect().await.expect("body").to_bytes();
        assert!(
            std::str::from_utf8(&body)
                .expect("utf8")
                .ends_with("# EOF\n")
        );
    }

    /// §2.1 auth posture: with a key configured and no ConnectInfo (reads as
    /// not-loopback - when in doubt, require the key), the scrape 401s bare
    /// and opens with the key; `metrics_auth = false` force-opens it.
    #[tokio::test]
    async fn metrics_auth_follows_the_forced_posture() {
        let keyed = || {
            let mut s = AppState::for_tests(None);
            s.auth_key = Some("pk-test".into());
            s
        };
        let bare = router(Arc::new(keyed()))
            .oneshot(
                axum::http::Request::get("/metrics")
                    .body(axum::body::Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(bare.status(), StatusCode::UNAUTHORIZED);
        let with_key = router(Arc::new(keyed()))
            .oneshot(
                axum::http::Request::get("/metrics")
                    .header("authorization", "Bearer pk-test")
                    .body(axum::body::Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(with_key.status(), StatusCode::OK);
        let mut open = keyed();
        open.metrics_auth = Some(false);
        let forced_open = router(Arc::new(open))
            .oneshot(
                axum::http::Request::get("/metrics")
                    .body(axum::body::Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(forced_open.status(), StatusCode::OK);
    }

    /// The loopback exemption is for the box's own callers, and only them: a
    /// forwarding header or a declared proxy means the peer address is the
    /// proxy's, and the key is required again.
    #[tokio::test]
    async fn loopback_is_exempt_only_when_nothing_says_proxy() {
        use axum::extract::ConnectInfo;
        let local: std::net::SocketAddr = "127.0.0.1:4242".parse().expect("addr");
        let keyed = |trusted_proxy: bool| {
            let mut s = AppState::for_tests(None);
            s.auth_key = Some("pk-test".into());
            s.trusted_proxy = trusted_proxy;
            Arc::new(s)
        };
        let get = |headers: &[(&str, &str)]| {
            let mut req = axum::http::Request::get("/v1/models");
            for (k, v) in headers {
                req = req.header(*k, *v);
            }
            let mut req = req.body(axum::body::Body::empty()).expect("request");
            req.extensions_mut().insert(ConnectInfo(local));
            req
        };
        // the box's own tools: no key needed
        let plain = router(keyed(false))
            .oneshot(get(&[]))
            .await
            .expect("response");
        assert_eq!(plain.status(), StatusCode::OK);
        // the same peer, but a proxy left its mark: key required
        let forwarded = router(keyed(false))
            .oneshot(get(&[("x-forwarded-for", "203.0.113.7")]))
            .await
            .expect("response");
        assert_eq!(forwarded.status(), StatusCode::UNAUTHORIZED);
        // the operator declared a proxy: loopback means nothing, key required
        let declared = router(keyed(true))
            .oneshot(get(&[]))
            .await
            .expect("response");
        assert_eq!(declared.status(), StatusCode::UNAUTHORIZED);
        // and the key still opens it
        let with_key = router(keyed(true))
            .oneshot(get(&[("authorization", "Bearer pk-test")]))
            .await
            .expect("response");
        assert_eq!(with_key.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn unknown_routes_get_openai_shaped_404() {
        let res = test_router()
            .oneshot(
                axum::http::Request::get("/v1/definitely-not-a-route")
                    .body(axum::body::Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(res.status(), StatusCode::NOT_FOUND);
        let json = body_json(res).await;
        assert_eq!(json["error"]["type"], "not_found_error");
    }

    #[tokio::test]
    async fn draining_refuses_inference_and_fails_healthz() {
        let state = Arc::new(AppState::for_tests(None));
        state.drain.begin();
        let app = router(state);

        // New inference work: 503 + Retry-After (retry-capable SDKs recover).
        let res = app
            .clone()
            .oneshot(
                axum::http::Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from("{}"))
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            res.headers()
                .get(axum::http::header::RETRY_AFTER)
                .and_then(|v| v.to_str().ok()),
            Some("15"),
        );

        // Readiness fails during drain (the manager's health-gate reads this).
        let res = app
            .oneshot(
                axum::http::Request::get("/healthz")
                    .body(axum::body::Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn admission_cap_refuses_in_each_dialects_shape() {
        // cap 0: the first request already pushes in_flight past the cap, so
        // every generation request is refused - the degenerate case that
        // exercises the refusal path without needing a served model.
        let mut state = AppState::for_tests(None);
        state.concurrency_limit = Some(0);
        let app = router(Arc::new(state));

        // OpenAI side: 503 + overloaded_error + Retry-After.
        let res = app
            .clone()
            .oneshot(
                axum::http::Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from("{}"))
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(res.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            res.headers()
                .get(axum::http::header::RETRY_AFTER)
                .and_then(|v| v.to_str().ok()),
            Some("1"),
        );
        let json = body_json(res).await;
        assert_eq!(json["error"]["type"], "overloaded_error");

        // Anthropic side: 529 + their error envelope.
        let res = app
            .clone()
            .oneshot(
                axum::http::Request::post("/v1/messages")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from("{}"))
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(res.status().as_u16(), 529);
        let json = body_json(res).await;
        assert_eq!(json["type"], "error");
        assert_eq!(json["error"]["type"], "overloaded_error");

        // count_tokens is never capped (pure tokenize - token budgeting stays
        // alive on a saturated endpoint); with no model it 503s from the
        // handler, not 529 from the cap.
        let res = app
            .oneshot(
                axum::http::Request::post("/v1/messages/count_tokens")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(r#"{"model":"m","messages":[]}"#))
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_ne!(
            res.status().as_u16(),
            529,
            "count_tokens must not be admission-capped"
        );
    }

    #[tokio::test]
    async fn server_info_reports_runner_role() {
        let res = test_router()
            .oneshot(
                axum::http::Request::get("/api/server")
                    .body(axum::body::Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(res.status(), StatusCode::OK);
        let json = body_json(res).await;
        assert_eq!(json["role"], "runner");
        assert!(json["pid"].is_number());
    }

    #[tokio::test]
    async fn openapi_spec_serves_and_parses() {
        let res = test_router()
            .oneshot(
                axum::http::Request::get("/openapi.json")
                    .body(axum::body::Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(res.status(), StatusCode::OK);
        let spec = body_json(res).await;
        assert_eq!(spec["openapi"], "3.1.0");
        // the version placeholder must have been stamped, not served raw
        assert_eq!(spec["info"]["version"], env!("CARGO_PKG_VERSION"));
    }

    /// `/api/metadata` is the runner's own answer to "what is in this file",
    /// and it depends on nothing but the bytes - no model, no context window,
    /// no pdfium. The manager serves the same crate off a stored attachment
    /// this route is why an API client that never runs a manager
    /// still gets it.
    #[tokio::test]
    async fn file_metadata_reads_a_photo_from_bytes_alone() {
        use base64::Engine as _;
        // Minimal JPEG with one EXIF tag (Make = TestCam) - same fixture the
        // doc.rs photo tests use.
        let tiff: Vec<u8> = [
            &[0x49, 0x49, 0x2A, 0x00, 0x08, 0x00, 0x00, 0x00][..],
            &[0x01, 0x00][..],
            &[
                0x0F, 0x01, 0x02, 0x00, 0x08, 0x00, 0x00, 0x00, 0x1A, 0x00, 0x00, 0x00,
            ][..],
            &[0x00, 0x00, 0x00, 0x00][..],
            b"TestCam\0",
        ]
        .concat();
        let mut jpeg = vec![0xFF, 0xD8, 0xFF, 0xE1];
        jpeg.extend_from_slice(&((2 + 6 + tiff.len()) as u16).to_be_bytes());
        jpeg.extend_from_slice(b"Exif\0\0");
        jpeg.extend_from_slice(&tiff);
        jpeg.extend_from_slice(&[0xFF, 0xD9]);
        let data = format!(
            "data:image/jpeg;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(&jpeg)
        );

        let res = test_router()
            .oneshot(
                axum::http::Request::post("/api/metadata")
                    .header("content-type", "application/json")
                    .body(axum::body::Body::from(
                        serde_json::json!({ "data": data }).to_string(),
                    ))
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(res.status(), StatusCode::OK);
        let json = body_json(res).await;
        assert_eq!(json["format"], "JPEG");
        assert_eq!(json["reader"], "sift");
        assert_eq!(json["groups"][0]["name"], "EXIF");
        let make = json["groups"][0]["tags"]
            .as_array()
            .expect("tags")
            .iter()
            .find(|t| t["name"] == "Make")
            .map(|t| t["value"].clone());
        assert_eq!(make, Some(serde_json::json!("TestCam")), "got {json}");
    }

    /// The spec-drift guard: every path the OpenAPI document claims must be a
    /// route the router actually serves. An OPTIONS probe distinguishes the
    /// two without needing valid bodies - axum answers 405 (method not
    /// allowed) on a registered path and falls through to the JSON 404 on an
    /// unregistered one. This catches the spec lying about a route; the
    /// reverse direction (a route added without a spec entry) is a comment on
    /// the router, because a test cannot enumerate axum's routes.
    #[tokio::test]
    async fn openapi_spec_names_only_live_routes() {
        let spec: serde_json::Value =
            serde_json::from_str(include_str!("../openapi.json")).expect("spec parses");
        let paths = spec["paths"].as_object().expect("spec has paths");
        assert!(!paths.is_empty());
        for (path, _) in paths {
            // path templates hold exactly the params the router declares;
            // substitute a literal so the probe hits the real route
            let concrete = path.replace("{id}", "probe-id");
            let res = test_router()
                .oneshot(
                    axum::http::Request::builder()
                        .method("OPTIONS")
                        .uri(&concrete)
                        .body(axum::body::Body::empty())
                        .expect("request"),
                )
                .await
                .expect("response");
            assert_ne!(
                res.status(),
                StatusCode::NOT_FOUND,
                "openapi.json documents {path} but the router does not serve it"
            );
        }
    }
}
