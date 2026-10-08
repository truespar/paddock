//! Manager HTTP routes: the Studio SPA + its store surface (conversations,
//! prompts, settings, keys, MCP), the model catalog/pull/estimate endpoints,
//! and device telemetry. No inference routes - /v1/* lives on runners, and the
//! manager never proxies a byte of it (doc §1).

use std::sync::Arc;

use axum::Router;
use axum::extract::State;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use paddock_api::ErrorBody;

/// Shared manager state.
pub struct AppState {
    /// Embedded SQLite store: conversations, prompts, settings, API keys, MCP
    /// servers, collected activity - The one database on the box (runners are
    /// stateless). Arc because the collector task writes it concurrently.
    pub db: std::sync::Arc<crate::store::Store>,
    /// Required Bearer key for /api, or None for no auth (loopback).
    pub auth_key: Option<String>,
    /// A reverse proxy is declared in front (`trusted_proxy`): loopback peers
    /// are then not the box's own callers, and the key applies to them too.
    pub trusted_proxy: bool,
    /// Host headers the MCP endpoints accept (rmcp's DNS-rebinding guard).
    /// Loopback-only on a loopback bind; the host's real names on a network
    /// bind. Never empty - rmcp reads an empty list as "allow any host".
    pub mcp_allowed_hosts: Vec<String>,
    /// Graph bridge: conversation -> the Studio tab's live graph session, so
    /// the graph_query MCP tool can execute where the engine actually is
    /// (D4).
    pub graphs: std::sync::Arc<crate::graph::Bridge>,
    /// NVML device telemetry handle (sampler on its own thread) - read-only.
    pub gpu: crate::telemetry::Telemetry,
    /// Whether this computer can serve models at all, and what to say if not
    /// Probed once at startup: hardware does not change under a
    /// running process.
    pub readiness: std::sync::Arc<crate::readiness::Readiness>,
    /// Model registry: browse + pull models from this build's compiled-in
    /// manifest (`models.toml`) - the origin (Cloudflare R2) is a dumb file host.
    pub registry: std::sync::Arc<crate::registry::Registry>,
    /// Memoized GGUF header probes behind `/api/models/estimate`, so moving the
    /// context slider doesn't re-read every installed model's header.
    pub probes: crate::estimate::ProbeCache,
    /// Default serving envelope for estimator math (spawn defaults later).
    pub max_ctx: usize,
    pub max_batch: usize,
    /// Runner supervision: spawn/stop/takeover/reconcile (doc §3).
    pub supervisor: std::sync::Arc<crate::supervisor::Supervisor>,
    /// Desired-state election set (managed.toml, §11.2). None in tests.
    pub elections: Option<std::sync::Arc<crate::elections::Elections>>,
    /// The §9 VRAM reconciliation gauge (inside vs outside view, drift +
    /// anomaly flag), published by the reconciler task; None until sampled.
    pub recon: tokio::sync::watch::Receiver<Arc<Option<crate::telemetry::Reconciliation>>>,
    /// Server-push fan-out to open Studio tabs (`/api/events`, SSE): fleet
    /// and update state on CHANGE, replacing the per-tab poll loops.
    pub push: crate::push::Hub,
    /// Held from admission through launch/restart, closing the reservation
    /// gap before a config/runner appears in fleet accounting. Read paths and
    /// inference never take this lock.
    pub admission: tokio::sync::Mutex<()>,
    /// Last "is there a newer paddock" answer, so a UI that polls does not turn
    /// into an outbound request per poll.
    pub updates: crate::updates::Cache,
    /// The update download, while it runs. One per process; a second request
    /// joins it rather than racing onto the same file.
    pub update_dl: std::sync::Mutex<Option<Arc<crate::updates::Download>>>,
    /// This box's TLS facts, for the trust page and the root download.
    /// None when the identity could not be established and the manager
    /// fell back to cleartext - the UI says so rather than offering a
    /// certificate that does not exist.
    pub tls: Option<Arc<TlsFacts>>,
}

/// What the Studio needs to explain https to a person: the root to install,
/// its fingerprint to check it against, and the addresses it is good for.
pub struct TlsFacts {
    pub root_pem: String,
    pub fingerprint: String,
    pub names: Vec<String>,
}

impl AppState {
    /// Minimal state for tests (in-memory db, telemetry disabled).
    #[doc(hidden)]
    pub fn for_tests() -> Self {
        AppState {
            // tests bind loopback, so the real bind-derived list would only be
            // noise here
            mcp_allowed_hosts: vec!["localhost".into(), "127.0.0.1".into(), "::1".into()],
            db: Arc::new(
                crate::store::Store::open(&std::path::PathBuf::from(":memory:"))
                    .expect("mem test db"),
            ),
            auth_key: None,
            trusted_proxy: false,
            push: crate::push::Hub::new(),
            graphs: Arc::new(crate::graph::Bridge::new()),
            gpu: crate::telemetry::Telemetry::disabled(),
            // The real probe: it never fails and never blocks, and a test box
            // that genuinely has no card should see exactly what a user's
            // would.
            readiness: Arc::new(crate::readiness::probe()),
            registry: Arc::new(crate::registry::Registry::new(std::env::temp_dir())),
            probes: crate::estimate::ProbeCache::default(),
            max_ctx: 4096,
            max_batch: 32,
            supervisor: {
                let registry = Arc::new(crate::registry::Registry::new(std::env::temp_dir()));
                Arc::new(crate::supervisor::Supervisor::new(
                    crate::supervisor::SpawnDefaults {
                        runner_bin: None,
                        device: "cuda".into(),
                        kernel_pack: None,
                        models_dirs: vec![],
                        logs_dir: std::env::temp_dir(),
                        runners_dir: std::env::temp_dir().join("paddock-test-runners"),
                        work_dir: std::env::temp_dir(),
                        base_port: 41540,
                        health_timeout: std::time::Duration::from_secs(5),
                    },
                    registry,
                    None, // unit tests never persist elections
                    None, // no NVML in tests
                ))
            },
            elections: None,
            recon: {
                let (tx, rx) = tokio::sync::watch::channel(Arc::new(None));
                std::mem::forget(tx); // keep the receiver valid forever
                rx
            },
            admission: tokio::sync::Mutex::new(()),
            updates: crate::updates::Cache::default(),
            update_dl: std::sync::Mutex::new(None),
            tls: None,
        }
    }
}

/// Largest request body the runner relays will buffer - mirrors the runner's
/// own MAX_BODY (192 MB: a 100 MB attachment base64-inflates to ~134 MB plus
/// JSON overhead). Bounded, because the manager port can be network-exposed.
const RELAY_MAX_BODY: usize = 192 * 1024 * 1024;

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/auth/login", post(auth_login))
        .route("/auth/logout", post(auth_logout))
        // The box's root certificate, and the facts a person needs to decide
        // whether to install it. Outside /api deliberately, so it is
        // reachable before any key gate and - with the listener's redirect
        // carve-out - over plain http too: a client that does not yet trust us
        // cannot be asked to trust us in order to fetch the thing that would
        // make it trust us.
        // Both outside /api, so the trust page renders before the key gate
        // does. Neither is a secret: the fingerprint and the names are printed
        // in the certificate that every connecting client already receives.
        .route(paddock_tls::serve::ROOT_PATH, get(tls_root))
        .route("/tls/info", get(tls_info))
        .route("/api/server", get(server_info))
        .route("/api/gpu", get(gpu_info))
        .route("/api/cache", get(cache_info))
        .route("/api/gpu/stream", get(gpu_stream))
        .route("/api/readiness", get(readiness))
        .route("/api/gpus", get(gpu_sheet))
        .route("/api/events", get(crate::push::events_stream))
        .route("/api/updates", get(updates_status))
        .route("/api/updates/download", post(update_download_start))
        .route("/api/updates/download/cancel", post(update_download_cancel))
        // Feedback goes out to the truespar API; the context endpoint is the
        // preview the dialog shows, and returns the very struct the POST
        // attaches (crate::feedback).
        .route("/api/feedback", post(crate::feedback::submit))
        .route("/api/feedback/context", get(crate::feedback::preview))
        // Forensics persistence + read surface. The runner emits the report as
        // a /v1/responses output item (standalone); the manager persists it
        // here and serves it back to the Studio. Fully decoupled - the runner
        // never calls these.
        .route("/api/forensics", post(crate::forensics::persist))
        .route("/api/forensics/{id}", get(crate::forensics::get_report))
        .route(
            "/api/conversations/{conversation_id}/forensics",
            get(crate::forensics::list_for_conversation),
        )
        .route(
            "/api/attachments/{attachment_id}/forensics",
            get(crate::forensics::latest_for_attachment),
        )
        // /api/setup/cuda lived here, for the one-time fetch of
        // NVIDIA's maths libraries. Paddock ships and fetches none:
        // the exes import no NVIDIA DLL, the kernel pack imports
        // KERNEL32 only, and the binaries do not contain the string
        // "cublas64"/"cudart64" at all, so they could not ask for one. The only
        // NVIDIA binary the engine loads is the display driver's nvcuda.
        .route("/api/models/catalog", get(models_catalog))
        .route("/api/models/estimate", get(crate::estimate::handle))
        .route("/api/models/pull", post(models_pull))
        .route("/api/models/pull/{job}", get(models_pull_status))
        .route("/api/models/pull/{job}/cancel", post(models_pull_cancel))
        .route("/api/models/pull/{job}/resume", post(models_pull_resume))
        .route("/api/models/pulls", get(models_pulls))
        .route("/api/models/pulls/events", get(models_pulls_events))
        .route("/api/runners", get(runners_list).post(runners_spawn))
        .route("/api/servers", get(servers_list))
        .route("/api/servers/{port}", axum::routing::delete(servers_remove))
        .route("/api/servers/files", get(servers_files))
        .route("/api/servers/preview", post(servers_preview))
        .route("/api/servers/project", post(servers_project))
        .route(
            "/api/servers/{port}/file",
            get(servers_file_get).put(servers_file_put),
        )
        .route("/api/servers/{port}/start", post(servers_start))
        .route("/api/elections", get(elections_list))
        .route("/api/activity", get(activity_list).delete(activity_purge))
        .route("/api/usage/history", get(usage_history))
        // Studio-chat seam (doc §10): the browser talks only to the manager;
        // the manager originates the runner call as an ordinary API client
        // (runner keys stay server-side, no CORS to runner ports). This is
        // the Studio's private path - external clients still hit runners
        // directly; nothing here makes the manager an inference proxy for
        // the outside world (it lives under /api behind manager auth, and
        // /v1/* deliberately does not exist on this port).
        .route("/api/runners/{port}/v1/responses", post(relay_responses))
        // the Studio playground: embeddings + rerank ride the same relay as
        // chat - the manager is the CLIENT (it holds the runner key)
        .route("/api/runners/{port}/v1/embeddings", post(relay_embeddings))
        .route("/api/runners/{port}/v1/rerank", post(relay_rerank))
        .route("/api/runners/{port}/v1/systemone", post(relay_systemone))
        // the Masks page (SAM 3): a data-URL picture and a prompt in, COCO
        // RLE masks out - JSON both ways, under the relay's own body cap
        .route("/api/runners/{port}/v1/masks", post(relay_masks))
        // its camera: a video session, its frames, its end (a DELETE drops it)
        .route(
            "/api/runners/{port}/v1/masks/sessions",
            post(relay_mask_session),
        )
        .route(
            "/api/runners/{port}/v1/masks/sessions/{id}/frames",
            post(relay_mask_frame),
        )
        .route(
            "/api/runners/{port}/v1/masks/sessions/{id}/finish",
            post(relay_mask_finish),
        )
        .route(
            "/api/runners/{port}/v1/masks/sessions/{id}",
            axum::routing::delete(relay_mask_drop),
        )
        // Tables (Kumo-Tabular): a one-shot or fitted-context prediction, a
        // context fitted for reuse, and its release - JSON both ways; the
        // release is a DELETE, so this relay carries the method.
        .route(
            "/api/runners/{port}/v1/tabular/predictions",
            post(relay_tabular_predictions),
        )
        .route(
            "/api/runners/{port}/v1/tabular/contexts",
            post(relay_tabular_contexts),
        )
        .route(
            "/api/runners/{port}/v1/tabular/contexts/{id}",
            axum::routing::delete(relay_tabular_release),
        )
        // Transcribe: an audio FILE goes up, so this relay carries the
        // caller's multipart content-type (boundary and all) rather than the
        // JSON every other relay assumes.
        .route(
            "/api/runners/{port}/v1/audio/transcriptions",
            post(relay_transcriptions),
        )
        .route(
            "/api/runners/{port}/v1/audio/diarizations",
            post(relay_diarizations).layer(axum::extract::DefaultBodyLimit::max(25 * 1024 * 1024)),
        )
        .route(
            "/api/runners/{port}/v1/audio/diarizations/stream",
            get(relay_diarization_live),
        )
        // Forced alignment: same multipart story as transcribe -
        // the Studio's enrichment pass sends the clip + transcript here after
        // a lane without word times settles.
        .route(
            "/api/runners/{port}/v1/audio/alignments",
            post(relay_alignments),
        )
        // Images: generations is JSON like chat; edits carries image FILES
        // (multipart), so it forwards the content-type verbatim the way
        // transcriptions does.
        .route(
            "/api/runners/{port}/v1/images/generations",
            post(relay_image_generations),
        )
        .route(
            "/api/runners/{port}/v1/images/edits",
            post(relay_image_edits),
        )
        // Composer attachment costing: the Studio asks the runner's own
        // count_tokens what a staged file will really cost (real extraction,
        // real tokenizer) - never a client-side bytes/4 guess
        .route(
            "/api/runners/{port}/v1/messages/count_tokens",
            post(relay_count_tokens),
        )
        // "What the model reads": the runner's extraction preview for one
        // attachment (injection text incl. the metadata block, verbatim)
        .route("/api/runners/{port}/extract", post(relay_extract))
        .route("/api/runners/{port}/server", get(relay_server))
        // The runner's OpenAPI document, for the Studio's API reference page
        // It lives at the runner ROOT - outside /v1 and /api,
        // open like /healthz - so no other relay line covers it.
        .route("/api/runners/{port}/openapi.json", get(relay_openapi))
        .route("/api/runners/{port}/v1/realtime", get(relay_realtime))
        .route(
            "/api/runners/{port}/mcp-approvals/{id}",
            post(relay_approval),
        )
        .route("/api/runners/{port}", axum::routing::delete(runners_stop))
        .route("/api/runners/{port}/switch", post(runners_switch))
        .route("/api/runners/{port}/pin", post(runners_pin))
        .route("/api/runners/{port}/persist", post(runners_persist))
        .route("/api/runners/{port}/logs", get(runners_logs))
        // Cloud models (BYO-key external providers, doc §1): endpoint CRUD,
        // the provider's model list for the enable-picker, and the chat seam
        // that translates provider dialects into the one Responses stream the
        // Studio speaks. Keys stay in the DB - list() returns hasKey only.
        .route(
            "/api/connectors",
            get(crate::connectors::list).post(crate::connectors::create),
        )
        .route(
            "/api/connectors/{id}",
            axum::routing::put(crate::connectors::update).delete(crate::connectors::remove),
        )
        .route("/api/connectors/check", post(crate::connectors::check))
        .route("/api/mcp/tools", post(crate::connectors::tools))
        .route(
            "/api/cloud/mcp-approvals/{id}",
            post(crate::cloud::approval),
        )
        .route("/api/connectors/{id}/scope", post(crate::connectors::scope))
        .route(
            "/api/connectors/{id}/oauth/start",
            post(crate::oauth::start),
        )
        .route(
            "/api/connectors/{id}/oauth/disconnect",
            post(crate::oauth::disconnect),
        )
        .route(
            "/api/connectors/oauth/callback",
            get(crate::oauth::callback),
        )
        .route(
            "/api/cloud",
            get(crate::cloud::list).post(crate::cloud::create),
        )
        .route("/api/cloud/usage", get(crate::cloud::usage))
        // the key-less OpenRouter catalog (public endpoint) - the Cloud
        // page's browse-first landing; static segment, so it can't collide
        // with the {id} routes below
        .route("/api/cloud/browse", get(crate::cloud::browse))
        .route(
            "/api/cloud/browse/endpoints",
            get(crate::cloud::browse_endpoints),
        )
        .route(
            "/api/cloud/{id}",
            axum::routing::patch(crate::cloud::update).delete(crate::cloud::delete),
        )
        .route("/api/cloud/{id}/models", get(crate::cloud::models))
        .route("/api/cloud/{id}/check", post(crate::cloud::check))
        .route(
            "/api/cloud/{id}/v1/responses",
            post(crate::cloud::responses),
        )
        .route(
            "/api/cloud/{id}/v1/audio/transcriptions",
            post(crate::cloud::transcriptions),
        )
        // Log streaming (§11.3): manager | runner | merged, history-first
        // with opt-out, optional follow. The per-runner tail above stays for
        // the Studio's "last log lines" card.
        .route("/api/logs", get(crate::logs::handle))
        // Artifacts: the Studio's REST view plus the sandboxed
        // frame shell. The MCP endpoint itself is mounted below.
        .merge(crate::artifacts::routes())
        // The manager's own first-party MCP server. Under /api so the bearer
        // auth middleware covers it; the Studio hands its URL to the model as
        // an inline `mcp` tool, so a runner reaches it with the MCP client it
        // already has and needs no code change at all.
        .route_service(
            "/api/mcp/artifacts",
            crate::artifacts::mcp_service(state.db.clone(), state.mcp_allowed_hosts.clone()),
        )
        // Its graph twin: graph_query, bridged to the Studio tab's WASM engine
        // over /api/graph/bridge (crate::graph module doc).
        .merge(crate::graph::routes())
        .route_service(
            "/api/mcp/graph",
            crate::graph::mcp_service(state.graphs.clone(), state.mcp_allowed_hosts.clone()),
        )
        // Studio store surface (conversations, prompts, settings, keys, MCP) +
        // the embedded SPA; API misses still return the OpenAI-shaped JSON 404.
        .merge(crate::api::routes())
        .fallback(crate::static_assets::serve)
        // The runner relays carry whole attachments base64-inlined in chat
        // bodies; axum's 2 MB default cut the connection mid-upload for any
        // image/file over it, which the browser reports as "server
        // unreachable" (a >2 MB TIFF found it). Match the
        // RUNNER's own cap so the relay is never the narrower pipe. The
        // store surface keeps its tighter 100 MB attachment limit (its own
        // inner layer wins there).
        .layer(axum::extract::DefaultBodyLimit::max(RELAY_MAX_BODY))
        // Bearer auth over /api (no-op when auth_key is None).
        .layer(axum::middleware::from_fn_with_state(state.clone(), auth_mw))
        // Stamp `charset=utf-8` on JSON responses (outermost, so it also covers
        // auth-rejection + 404 bodies). Non-JSON (SSE, the SPA) is untouched.
        .layer(axum::middleware::map_response(json_utf8_charset))
        .with_state(state)
}

/// Add `charset=utf-8` to bare `application/json` responses (see the runner's
/// twin - lenient clients like PowerShell mangle non-ASCII without it).
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

async fn healthz() -> &'static str {
    "ok"
}

/// The box's root certificate, as a file a browser or an OS will offer to
/// install.
///
/// `application/x-x509-ca-cert` is what makes the download offer to install
/// rather than open in a text view. The `.crt` filename matters on Windows,
/// where the shell decides what a double-click does from the extension alone.
async fn tls_root(State(state): State<Arc<AppState>>) -> Response {
    let Some(tls) = state.tls.as_ref() else {
        return (
            StatusCode::NOT_FOUND,
            Json(ErrorBody::new(
                "no_tls",
                "this paddock is serving without https, so it has no certificate to hand out",
            )),
        )
            .into_response();
    };
    (
        StatusCode::OK,
        [
            (
                axum::http::header::CONTENT_TYPE,
                "application/x-x509-ca-cert",
            ),
            (
                axum::http::header::CONTENT_DISPOSITION,
                "attachment; filename=\"paddock-root.crt\"",
            ),
        ],
        tls.root_pem.clone(),
    )
        .into_response()
}

/// What the trust page renders: whether https is on, which names the local
/// certificate covers, and the fingerprint to check the root against.
async fn tls_info(State(state): State<Arc<AppState>>) -> Response {
    match state.tls.as_ref() {
        Some(tls) => Json(serde_json::json!({
            "enabled": true,
            "fingerprint": tls.fingerprint,
            "names": tls.names,
            "root_url": paddock_tls::serve::ROOT_PATH,
        }))
        .into_response(),
        None => Json(serde_json::json!({ "enabled": false })).into_response(),
    }
}

/// Manager identity + the box-level facts the Studio needs at boot.
async fn server_info(State(state): State<Arc<AppState>>) -> Response {
    Json(serde_json::json!({
        "role": "manager",
        // `version` stays the BARE SemVer forever - the update check and the
        // runner comparison parse it, and folding the commit in here would
        // make every one of them think a dev build is a different product.
        // The stamp rides alongside for display and bug reports.
        "version": paddock_admin::version::SEMVER,
        "build": paddock_admin::version::LONG,
        "commit": paddock_admin::version::GIT_SHA,
        "pid": std::process::id(),
        "max_ctx": state.max_ctx,
        "max_batch": state.max_batch,
        // Model registry: the on-disk models folder the manager pulls into +
        // free/total space on its volume, so the Studio can show the location
        // and warn before a download that wouldn't fit.
        // Progressive tool disclosure: the runner spends this tool budget one
        // SERVER at a time, smallest first, and hides whichever servers do not
        // fit behind mcp_search_tools/mcp_call_tool - so a small server keeps
        // its real schemas next to a big one. Whichever of the three notes
        // applies is injected into the system prompt. Published so the
        // Studio's panel can show the same text instead of keeping its own
        // copy - a duplicated sentence is a drift waiting to happen.
        // (The runner also weighs the schemas against the model's context, a
        // third of it at most, so a small context can hide more than the
        // count alone would.)
        "tool_search": {
            "threshold": paddock_mcp::tool_search::SEARCH_DISCLOSURE_THRESHOLD,
            "hidden": paddock_mcp::tool_search::SEARCH_MODE_INSTRUCTIONS,
            "available": paddock_mcp::tool_search::SEARCH_AVAILABLE_INSTRUCTIONS,
            "partial": paddock_mcp::tool_search::SEARCH_PARTIAL_TEMPLATE,
        },
        "registry": {
            "enabled": state.registry.enabled(),
            "models_dir": native_path(state.registry.models_dir()),
            "disk_free": crate::registry::disk_free(state.registry.models_dir()),
            "disk_total": crate::registry::disk_total(state.registry.models_dir()),
        },
    }))
    .into_response()
}

/// A path as the host OS spells it (see the runner's twin for the rationale).
pub fn native_path(p: &std::path::Path) -> String {
    let s = p.display().to_string();
    if cfg!(windows) {
        s.replace('/', "\\")
    } else {
        s
    }
}

/// Registry catalog - the models this build ships in its embedded manifest,
/// each annotated with `installed` + `total_size`. Never fails: it's compiled in.
async fn models_catalog(State(state): State<Arc<AppState>>) -> Response {
    Json(state.registry.catalog_annotated()).into_response()
}

/// Start pulling a model (`{ "id": "<model-id>" }`) -> `{ "job": "<id>" }`. The
/// download runs in the background; poll `/api/models/pull/{job}` or stream
/// `/api/models/pulls/events`.
///
/// Optional `then_start` (a SpawnSpec) + `then_action` ("spawn" | "switch"):
/// the MANAGER holds the plan and starts the endpoint with exactly those
/// settings the moment the bytes land - the browser that queued it can close.
/// This keeps the no-silent-download contract: the user explicitly clicked a
/// button that named the bytes; /api/runners itself still never downloads.
async fn models_pull(
    State(state): State<Arc<AppState>>,
    Json(body): Json<serde_json::Value>,
) -> Response {
    let Some(id) = body.get("id").and_then(|v| v.as_str()) else {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": {"type": "invalid_request_error", "message": "missing model id"}})),
        )
            .into_response();
    };
    // Optional piece selection (schema 3): {"artifacts": ["q4","vision"]}.
    // Absent = the default bundle (default weights + default companions).
    let artifacts: Option<Vec<String>> = body.get("artifacts").and_then(|v| {
        v.as_array().map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(str::to_owned))
                .collect::<Vec<_>>()
        })
    });
    // Validate the queued start now - a spec that can't parse must fail this
    // request, not surface as a mystery after a 20-GB download.
    let follow = match body.get("then_start") {
        Some(spec_v) => {
            let spec: Result<crate::supervisor::SpawnSpec, _> =
                serde_json::from_value(spec_v.clone());
            let action = body
                .get("then_action")
                .and_then(|v| v.as_str())
                .unwrap_or("spawn");
            match (spec, action) {
                (Err(e), _) => {
                    return relay_err(StatusCode::BAD_REQUEST, format!("bad then_start spec: {e}"));
                }
                (Ok(s), "switch") if s.port.is_none() => {
                    return relay_err(
                        StatusCode::BAD_REQUEST,
                        "then_action switch needs a port in the spec".into(),
                    );
                }
                (Ok(_), a) if a != "spawn" && a != "switch" => {
                    return relay_err(
                        StatusCode::BAD_REQUEST,
                        format!("then_action must be spawn or switch, not {a:?}"),
                    );
                }
                _ => Some(serde_json::json!({ "action": action, "spec": spec_v })),
            }
        }
        None => None,
    };
    match state.registry.start_pull(id, artifacts.as_deref()) {
        Ok(job_id) => {
            if let (Some(f), Some(job)) = (follow, state.registry.job(&job_id)) {
                *job.follow
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(f);
                *job.follow_state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) =
                    Some(serde_json::json!({ "state": "queued" }));
                arm_follow(state.clone(), job_id.clone());
            }
            Json(serde_json::json!({ "job": job_id })).into_response()
        }
        Err(e) => registry_err(e),
    }
}

/// The manager-owned tail of "Download & start": watch the pull job and, when
/// the bytes land, drive the queued spawn/switch with the retained spec. Runs
/// server-side so the start happens even if the browser that queued it is
/// gone; progress rides the job snapshot (`start.state`).
fn arm_follow(state: Arc<AppState>, job_id: String) {
    tokio::spawn(async move {
        use crate::registry::PullStatus;
        loop {
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            let Some(job) = state.registry.job(&job_id) else {
                return;
            };
            let status = job
                .status
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone();
            match status {
                PullStatus::Running => continue,
                PullStatus::Done => {
                    let Some(f) = job
                        .follow
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .clone()
                    else {
                        return;
                    };
                    *job.follow_state
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner) =
                        Some(serde_json::json!({ "state": "starting" }));
                    let action = f
                        .get("action")
                        .and_then(|v| v.as_str())
                        .unwrap_or("spawn")
                        .to_owned();
                    let mut spec: crate::supervisor::SpawnSpec =
                        match serde_json::from_value(f.get("spec").cloned().unwrap_or_default()) {
                            Ok(s) => s,
                            Err(e) => {
                                *job.follow_state
                                    .lock()
                                    .unwrap_or_else(std::sync::PoisonError::into_inner) =
                                    Some(serde_json::json!({
                                        "state": "error", "message": format!("bad spec: {e}"),
                                    }));
                                return;
                            }
                        };
                    tracing::info!(model = %spec.model, port = ?spec.port, %action, "download finished - running the queued start");
                    // same admission as a live click - a queued start must
                    // not freeze the box either (and the freshly landed file
                    // is probeable now, so it gets a real budget grant). A
                    // caller-approved eviction plan queued with the download
                    // executes here the same as it would on a live click.
                    let freeing = (action == "switch").then(|| spec.port.unwrap_or_default());
                    let evict: Vec<u16> = spec
                        .evict
                        .iter()
                        .copied()
                        .filter(|p| Some(*p) != freeing)
                        .collect();
                    if !spec.on_demand()
                        && let Err(e) = perform_evictions(&state, &evict).await
                    {
                        *job.follow_state
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner) =
                            Some(serde_json::json!({ "state": "error", "message": e }));
                        return;
                    }
                    let fit_ctx = pin_envelope(&mut spec, &state.registry);
                    let _admission = state.admission.lock().await;
                    let req = AdmitReq {
                        fit_ctx,
                        ..AdmitReq::for_spec(&spec, freeing)
                    };
                    match vram_admission(&state, req).await {
                        Err(refusal) => {
                            *job.follow_state
                                .lock()
                                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(
                                serde_json::json!({ "state": "error", "message": refusal.message }),
                            );
                            return;
                        }
                        Ok(admitted) => admitted.apply(&mut spec),
                    }
                    let outcome = if action == "switch" {
                        // honor the edit page's optimistic-concurrency token:
                        // a config file hand-edited DURING the download is
                        // refused, never clobbered
                        let expect = f
                            .get("spec")
                            .and_then(|s| s.get("expect_config_hash"))
                            .and_then(|v| v.as_str())
                            .map(str::to_owned);
                        let port = spec.port.unwrap_or_default();
                        state.supervisor.switch(port, spec, 30_000, expect).await
                    } else {
                        state
                            .supervisor
                            .spawn(spec)
                            .await
                            .map_err(|e| e.to_string())
                    };
                    *job.follow_state
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(match outcome {
                        Ok(v) => serde_json::json!({ "state": "ok", "port": v.port }),
                        Err(e) => serde_json::json!({ "state": "error", "message": e }),
                    });
                    return;
                }
                // cancelled or failed: the plan stays queued on the job (a
                // resume re-arms it); no start happens from this watcher
                PullStatus::Cancelled | PullStatus::Error { .. } => return,
            }
        }
    });
}

/// Poll a pull job: `{ id, model, display, downloaded, total, status, start? }`.
async fn models_pull_status(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(job): axum::extract::Path<String>,
) -> Response {
    match state.registry.job(&job) {
        Some(j) => Json(j.snapshot()).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": {"type": "not_found_error", "message": "unknown pull job"}})),
        )
            .into_response(),
    }
}

/// Ask a running download to stop. Partial bytes + segment sidecars stay on
/// disk, so Resume continues where this left off.
async fn models_pull_cancel(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(job): axum::extract::Path<String>,
) -> Response {
    if state.registry.cancel_pull(&job) {
        Json(serde_json::json!({ "ok": true })).into_response()
    } else {
        (
            StatusCode::CONFLICT,
            Json(serde_json::json!({"error": {"type": "invalid_request_error", "message": "job is not running (unknown, finished, or already stopped)"}})),
        )
            .into_response()
    }
}

/// Resume a cancelled/failed download: a fresh job over the same selection
/// (complete files skip, partial files continue). A queued start carries over
/// and is re-armed.
async fn models_pull_resume(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(job): axum::extract::Path<String>,
) -> Response {
    match state.registry.resume_pull(&job) {
        Ok(new_id) => {
            let follows = state.registry.job(&new_id).is_some_and(|j| {
                j.follow
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .is_some()
            });
            if follows {
                if let Some(j) = state.registry.job(&new_id) {
                    *j.follow_state
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner) =
                        Some(serde_json::json!({ "state": "queued" }));
                }
                arm_follow(state.clone(), new_id.clone());
            }
            Json(serde_json::json!({ "job": new_id })).into_response()
        }
        Err(e) => registry_err(e),
    }
}

/// Every pull job since boot, oldest first - what the Studio's download
/// indicator restores from on load.
async fn models_pulls(State(state): State<Arc<AppState>>) -> Response {
    let snaps: Vec<serde_json::Value> =
        state.registry.jobs().iter().map(|j| j.snapshot()).collect();
    Json(snaps).into_response()
}

/// Live download progress as SSE: the full jobs array every ~600 ms (first
/// event immediately). The Studio's header indicator listens here while
/// anything is running.
async fn models_pulls_events(State(state): State<Arc<AppState>>) -> Response {
    use axum::response::sse::{Event, KeepAlive, Sse};
    let stream = futures::stream::unfold((state, true), |(state, first)| async move {
        if !first {
            tokio::time::sleep(std::time::Duration::from_millis(600)).await;
        }
        let snaps: Vec<serde_json::Value> =
            state.registry.jobs().iter().map(|j| j.snapshot()).collect();
        let ev = Event::default().data(serde_json::Value::Array(snaps).to_string());
        Some((Ok::<_, std::convert::Infallible>(ev), (state, false)))
    });
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

fn registry_err(e: crate::registry::DlError) -> Response {
    use crate::registry::DlError;
    let (code, kind) = match &e {
        DlError::Disk { .. } => (StatusCode::INSUFFICIENT_STORAGE, "insufficient_storage"),
        // the manifest lists it but the origin no longer has it (force-deleted etc.)
        DlError::NotFound { .. } => (StatusCode::NOT_FOUND, "model_unavailable"),
        _ => (StatusCode::BAD_GATEWAY, "registry_error"),
    };
    (
        code,
        Json(serde_json::json!({"error": {"type": kind, "message": e.to_string()}})),
    )
        .into_response()
}

// ── VRAM admission: an oversubscribed card must be IMPOSSIBLE ───────────────
// Windows WDDM lets cudaMalloc past dedicated VRAM by paging into system RAM,
// which froze the whole box when gemma (39.6 GiB planned) loaded beside qwen
// (23.6 GiB resident) on a 48 GB card. Every spawn path
// prices itself here before launching: exact weights bytes + a conservative
// floor for state/KV/scratch, against total minus the fleet's own ledgers.
// Refusal is honest and names who holds what. The engine carries its own
// last-line gate too; this one exists so the user hears "no" in the UI
// instead of feeling it as a frozen desktop.

/// state/KV/scratch floor beyond weights - deliberately conservative (real
/// serving needs more; the engine's gate + pool sizing handle the rest).
const ADMIT_FLOOR: u64 = 2 << 30;
/// desktop/compositor + driver slack that is never ours to take.
const ADMIT_MARGIN: u64 = 1 << 30;

/// One endpoint whose stop would give the refused start its bytes back.
#[derive(serde::Serialize)]
struct EvictCandidate {
    port: u16,
    /// the human name every UI surface shows ("Qwen 3.5 9B")
    display: Option<String>,
    /// bytes admission counts back the moment it stops - its budget grant or
    /// live ledger, whichever admission was charging
    frees: u64,
    /// bytes of weights a later restart would reload (the llama-swap
    /// eviction-cost lesson: a small fast-loading model yields first)
    restore_cost: Option<u64>,
}

/// The actionable half of a refusal (layer 2 of the vram-budget plan): which
/// stops would make the start fit. The UI turns this into "Stop X and
/// start?"; the caller's explicit yes comes back as the `evict` list.
#[derive(serde::Serialize)]
struct EvictionOffer {
    /// bytes the refused start needs unclaimed
    need: u64,
    /// bytes unclaimed right now
    residual: u64,
    /// unpinned running endpoints on this device, cheapest-to-restore first
    candidates: Vec<EvictCandidate>,
    /// minimal prefix of `candidates` whose stop makes it fit; empty = even
    /// stopping every unpinned model would not (pick a smaller quant)
    plan: Vec<u16>,
}

/// A refusal with its offer. `message` alone still tells the whole story in
/// text (logs, CLI, the download chip); `eviction` is the structured form
/// the Studio's confirm dialog is built from.
struct AdmissionRefusal {
    message: String,
    eviction: Option<EvictionOffer>,
}

/// Build a refusal's offer: candidates cheapest-to-restore first (the
/// llama-swap `evict_costs` ordering - a small fast-reloading model yields
/// before a 30 GB one), plus the minimal greedy prefix whose stop covers
/// `need`. An empty plan = even stopping everything unpinned wouldn't fit.
fn eviction_offer(mut candidates: Vec<EvictCandidate>, residual: u64, need: u64) -> EvictionOffer {
    candidates.sort_by_key(|c| c.restore_cost.unwrap_or(c.frees));
    let mut plan = Vec::new();
    let mut freed = 0u64;
    for c in &candidates {
        if residual + freed >= need {
            break;
        }
        plan.push(c.port);
        freed += c.frees;
    }
    if residual + freed < need {
        plan.clear();
    }
    EvictionOffer {
        need,
        residual,
        candidates,
        plan,
    }
}

/// The 507 the handlers answer with on refusal.
fn admission_refused(r: AdmissionRefusal) -> Response {
    let mut err = serde_json::json!({"type": "insufficient_vram", "message": r.message});
    if let Some(ev) = r.eviction {
        err["eviction"] = serde_json::to_value(ev).unwrap_or(serde_json::Value::Null);
    }
    (
        StatusCode::INSUFFICIENT_STORAGE,
        Json(serde_json::json!({ "error": err })),
    )
        .into_response()
}

/// Execute a caller-approved eviction plan: drain-stop each named endpoint.
/// All-or-nothing validation first - the caller said yes to a SPECIFIC plan,
/// so a pinned or already-gone port refuses the whole action rather than
/// stopping a subset the user never confirmed. Stops run outside the
/// admission gate (drains take seconds); the admission that follows re-reads
/// the world, so a raced grant just yields a fresh refusal + fresh offer.
async fn perform_evictions(state: &Arc<AppState>, ports: &[u16]) -> Result<(), String> {
    if ports.is_empty() {
        return Ok(());
    }
    let views = state.supervisor.list().await;
    for &p in ports {
        let Some(v) = views.iter().find(|v| v.port == p) else {
            return Err(format!(
                "evict: port {p} is not running (already stopped?) - retry the start"
            ));
        };
        if v.pinned {
            let name = v
                .display
                .clone()
                .or_else(|| v.model.clone())
                .unwrap_or_else(|| format!("port {p}"));
            return Err(format!(
                "evict: {name} (port {p}) is pinned and never yields - unpin it first, or stop something else"
            ));
        }
    }
    for &p in ports {
        state
            .supervisor
            .stop(p, 30_000)
            .await
            .map_err(|e| format!("evict: stopping port {p} failed: {e}"))?;
        // The collector observing this death cannot tell a clean stop from a
        // crash - the route can, so it stamps the lifecycle band.
        let _ = state.db.stamp_end_cause(p, "stopped");
        tracing::info!(port = p, "evicted (drain-stopped) to make room for a start");
    }
    Ok(())
}

/// What admission needs to know about the start being priced.
struct AdmitReq<'a> {
    on_demand: bool,
    model: &'a str,
    artifact: Option<&'a str>,
    gpu_pin: Option<&'a str>,
    /// Same-port takeover: the incumbent drains first, so its grant/ledger
    /// counts as available.
    freeing_port: Option<u16>,
    /// The endpoint's own configured `vram_budget` (bytes) - a verbatim start
    /// of an existing file, or an operator's explicit grant: admit exactly
    /// this, never compute a new one.
    fixed_need: Option<u64>,
    /// The grant envelope (spec fields; the runner's own defaults otherwise).
    max_batch: Option<usize>,
    max_ctx: Option<usize>,
    /// The KV class the endpoint asks for: `Some(true)` fp8, `Some(false)`
    /// f16, `None` = auto/absent, which the runner serves at the family's
    /// own default - priced from the catalog row's `kv_default` (KV8 for
    /// every generative family but three, since 2026-10-04). Pricing auto at
    /// f16 charged KV8 endpoints twice the pool they draw.
    fp8_kv: Option<bool>,
    /// Endpoint arms the prefix-cache tier: device staging goes resident for
    /// as long as it is armed. Admission has to price it for the same reason
    /// it prices speculation - a start admitted on arithmetic the runner does
    /// not use is a start that then cannot seat what it was promised.
    offload_ram_bytes: Option<u64>,
    /// Endpoint speculates: a drafter goes resident and the verify plane
    /// widens. Admission has to price it, or it admits a spawn on arithmetic
    /// the runner will then exceed - and the whole point of this guard is that
    /// a start either fits or is refused, never OOMs.
    spec: bool,
    /// Endpoint serves vision/audio: the mmproj tower goes resident. Same
    /// reason as `spec` - the supervisor drops the mmproj when the spec says
    /// `vision: false`, so charging it unconditionally refused starts that
    /// would have fit.
    vision: bool,
    /// A split-tower model's audio tower (`spec.audio`; absent = the
    /// catalog default) - priced by the same rule the supervisor serves by.
    audio: Option<bool>,
    /// `max_ctx` is the catalog's TARGET, not a choice anyone made (the start
    /// left it unset and `pin_envelope` filled it): shrink it onto what the
    /// card backs instead of pricing it verbatim. Never set for an explicit
    /// window - a typed number is the operator's, and is priced as asked.
    fit_ctx: bool,
}

/// What admission hands back: the budget grant (MiB), and for a start whose
/// window was the catalog target, the window it fitted that target down to.
#[derive(Debug, Default)]
struct Admitted {
    grant: Option<u64>,
    max_ctx: Option<usize>,
}

impl Admitted {
    /// Fill what the spec left open: an explicit budget was admitted verbatim
    /// and keeps its value; a fitted window replaces the target it came from.
    fn apply(self, spec: &mut crate::supervisor::SpawnSpec) {
        if spec.vram_budget.is_none() {
            spec.automatic_budget = self.grant;
        }
        spec.vram_budget = spec.vram_budget.or(self.grant);
        if let Some(ctx) = self.max_ctx {
            spec.max_ctx = Some(ctx);
        }
    }
}

/// The Studio's context ladder (`CTX_STEPS` in studio/src/lib/ctx-ladder.ts).
/// A defaulted window lands on one of these rungs, so a CLI/API start and the
/// form pick the same number for the same card.
const CTX_STEPS: [u64; 7] = [4096, 8192, 16384, 32768, 65536, 131072, 262144];

/// Fit a DEFAULTED window to the card (decided 2026-10-04: "fit-aware long
/// ctx" - the catalog names a target, e.g. the full 262K at 4 slots, and the
/// card decides how much of it is real). The target when it fits at this
/// concurrency, else the largest ladder rung under it that does, else the
/// exact page-aligned window the smallest rung's pool backs - the form's
/// fallback too. `price(ctx)` is the caller's own estimate at that window, so
/// the pick and the grant come out of one calculation (and the dense spec
/// plane, which scales with the window, is re-priced at every step). When
/// nothing fits the target's price comes back unchanged for the refusal.
fn fit_window(
    target: u64,
    price: impl Fn(u64) -> paddock_estimator::Estimate,
) -> (u64, paddock_estimator::Estimate) {
    let fits = |e: &paddock_estimator::Estimate, ctx: u64| {
        !matches!(e.fit, paddock_estimator::Fit::DoesNotFit { .. }) && e.max_ctx >= ctx
    };
    let at_target = price(target);
    if fits(&at_target, target) {
        return (target, at_target);
    }
    let mut smallest = None;
    for &rung in CTX_STEPS.iter().rev().filter(|&&r| r < target) {
        let e = price(rung);
        if fits(&e, rung) {
            return (rung, e);
        }
        smallest = Some(e);
    }
    if let Some(e) = smallest
        .filter(|e| !matches!(e.fit, paddock_estimator::Fit::DoesNotFit { .. }) && e.max_ctx >= 16)
    {
        let ctx = e.max_ctx;
        let e = price(ctx);
        if fits(&e, ctx) {
            return (ctx, e);
        }
    }
    (target, at_target)
}

/// An endpoint's `kv_cache_dtype` as admission reads it: `Some(true)` fp8,
/// `Some(false)` an explicit exact class, `None` auto/absent (the family
/// default - see `AdmitReq::fp8_kv`).
fn kv_ask(v: Option<&str>) -> Option<bool> {
    match v {
        Some("fp8_e4m3" | "fp8") => Some(true),
        Some("f16" | "bf16" | "f32") => Some(false),
        _ => None,
    }
}

/// The runner's own config defaults (`paddock-runner/src/config.rs`). They
/// live here as named constants because admission PRICES against them and the
/// config file is then written with them - two uses that must agree, and used
/// to agree only by both spelling `4096` and `32` inline.
const RUNNER_DEFAULT_MAX_CTX: usize = 4096;
const RUNNER_DEFAULT_MAX_BATCH: usize = 32;

/// Write the envelope the budget was priced at into the spec, so the config
/// file carries both numbers.
///
/// A `vram_budget` means nothing on its own: it is the answer to "how much does
/// this model need at this max_ctx x max_batch, with this KV width and this
/// spec policy". The file used to record only the answer, leaving the question
/// to defaults in another crate - so the budget and the envelope it was
/// computed for were coupled at a distance, across a process boundary, with
/// only one of the two written down. That is also why a hand-edit of max_ctx
/// silently invalidates the budget with nothing to notice it.
///
/// Only fills what is absent: an explicit choice (CLI flag, Studio form, or a
/// hand-edited file on a verbatim start) always wins. True when the window
/// was filled here - the catalog's target, which admission then fits to the
/// card (`AdmitReq::fit_ctx`).
fn pin_envelope(
    spec: &mut crate::supervisor::SpawnSpec,
    registry: &crate::registry::Registry,
) -> bool {
    // An image lane has no envelope to pin: nothing is priced per token or
    // per slot, and the runner reads neither key for it.
    if crate::estimate::is_image_lane(registry, &spec.model) {
        return false;
    }
    let (ctx, batch) = registry.default_envelope(&spec.model, spec.artifact.as_deref());
    let defaulted = spec.max_ctx.is_none();
    spec.max_ctx.get_or_insert(ctx);
    spec.max_batch.get_or_insert(batch);
    // `spec` is deliberately not pinned here. The Studio form writes an
    // explicit value for every model that can speculate (its capability
    // gate hides the control - and omits the key - for the rest), so a
    // blind "on" pin only ever manufactured the key on non-speculative
    // models, where it is a claim the engine cannot honor: the spec
    // pre-flight then refused the start (granite-4.1 showed this) and the
    // legacy heal had to warn it away. Absent is the honest spelling for
    // "the engine decides per capability".
    defaulted
}

/// Is this endpoint's `spec` key anything but off? Matches the runner's
/// parser (crate spec_policy) on the values that mean "do not speculate";
/// absent means the engine's default, which speculates where it can.
fn spec_wanted(v: Option<&str>) -> bool {
    !matches!(
        v.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("off" | "false" | "no" | "none" | "0")
    )
}

impl<'a> AdmitReq<'a> {
    fn for_spec(spec: &'a crate::supervisor::SpawnSpec, freeing_port: Option<u16>) -> Self {
        Self {
            on_demand: spec.on_demand(),
            model: &spec.model,
            artifact: spec.artifact.as_deref(),
            gpu_pin: spec.gpu.as_deref(),
            freeing_port,
            fixed_need: spec.vram_budget.map(|mib| mib << 20),
            max_batch: spec.max_batch,
            max_ctx: spec.max_ctx,
            fp8_kv: kv_ask(spec.kv_cache_dtype.as_deref()),
            offload_ram_bytes: spec
                .kv_offload
                .as_ref()
                .filter(|k| k.enabled && k.ram_gb > 0.0)
                .map(|k| (k.ram_gb * (1u64 << 30) as f64) as u64),
            spec: spec_wanted(spec.spec_policy.as_deref()),
            // absent = on, matching `supervisor.rs`, which drops the mmproj
            // only on an explicit `Some(false)`
            vision: spec.vision != Some(false),
            audio: spec.audio,
            fit_ctx: false,
        }
    }
}

/// Refuse a start that cannot fit beside the fleet, and compute the
/// `vram_budget` grant (MiB) the endpoint's config file gets. The arithmetic
/// is budget-first: committed = Σ over
/// running AND mid-spawn endpoints of max(configured budget, live ledger) -
/// deterministic where budgets exist, honest where only the ledger does.
/// Ok(None) = admitted without a grant (bypass, no NVML view, unresolvable
/// model, unprobeable file, or a caller-fixed budget): the runner then sizes
/// free-at-load and its ledger carries the accounting, as before budgets.
/// Err = the honest refusal, carrying the eviction offer when stopping
/// something would help.
async fn vram_admission(
    state: &Arc<AppState>,
    req: AdmitReq<'_>,
) -> Result<Admitted, AdmissionRefusal> {
    // This starts only an API listener. The runner admits the actual serving
    // envelope against fresh device/host availability on EACH cold load.
    if req.on_demand {
        return Ok(Admitted::default());
    }
    // loud escape hatch for benches/experts - default is the hard no
    if std::env::var_os("PADDOCK_ALLOW_VRAM_OVERCOMMIT").is_some() {
        tracing::warn!(model = %req.model, "VRAM admission BYPASSED (PADDOCK_ALLOW_VRAM_OVERCOMMIT)");
        return Ok(Admitted::default());
    }
    // one admission at a time: two concurrent starts must not both price
    // themselves against the same residual
    // Caller holds admission until its launch finishes (including failures).
    if state.readiness.backend == "metal" {
        return metal_admission(state, req).await;
    }
    let snap = state.gpu.latest();
    if !snap.available || snap.gpus.is_empty() {
        return Ok(Admitted::default()); // no NVML view - nothing honest to refuse on
    }
    // resolve a pin (UUID or ordinal) to an NVML index
    let resolve_pin = |p: &str| {
        snap.gpus
            .iter()
            .find(|g| g.uuid.as_deref() == Some(p))
            .map(|g| g.index)
            .or_else(|| p.parse::<u32>().ok())
    };
    let sel = req.gpu_pin.and_then(resolve_pin).unwrap_or(0);
    let Some(total) = snap
        .gpus
        .iter()
        .find(|g| g.index == sel)
        .and_then(|g| g.mem_total)
    else {
        return Ok(Admitted::default());
    };
    let single_gpu = snap.gpus.len() <= 1;

    // Committed VRAM on this device: running endpoints (supervisor table ∪
    // admin enumeration, joined with the §9 reconciler ledgers) plus ports
    // whose spawn is still in flight - a record lands only after the health
    // gate, so a 30B loading for a minute is otherwise invisible and a
    // second start during the load would re-sell its bytes.
    let views = state.supervisor.list().await;
    let recon = state.recon.borrow().clone();
    let recon: Option<&crate::telemetry::Reconciliation> = recon.as_ref().as_ref();
    let mut committed: u64 = 0;
    let mut holders: Vec<(u16, u64)> = Vec::new();
    let mut cands: Vec<EvictCandidate> = Vec::new();
    for v in &views {
        if Some(v.port) == req.freeing_port {
            continue; // a takeover drains this one first - its VRAM comes back
        }
        let rv = recon.and_then(|r| r.runners.iter().find(|r| r.port == v.port));
        // device attribution: NVML's per-process view when it has one, else
        // the endpoint's own GPU pin; a single-GPU box counts the
        // WDDM-unattributable too
        let on_device = match rv.and_then(|rv| rv.gpu) {
            Some(g) => g == sel,
            None => {
                single_gpu
                    || v.config
                        .as_ref()
                        .and_then(|c| c.gpu.as_deref())
                        .and_then(resolve_pin)
                        == Some(sel)
            }
        };
        if !on_device {
            continue;
        }
        let dynamic = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            paddock_admin::client::AdminClient::new(v.port).config_status(),
        )
        .await
        .ok()
        .and_then(Result::ok)
        .and_then(|s| s.residency_reservation(v.pid));
        if dynamic == Some(0) {
            continue;
        }
        let budget = dynamic.unwrap_or_else(|| {
            state
                .supervisor
                .config_vram_budget(v.port)
                .map(|mib| mib << 20)
                .unwrap_or(0)
        });
        let ledger = rv.and_then(|rv| rv.self_mem).unwrap_or(0);
        let held = budget.max(ledger);
        committed += held;
        if held > 0 {
            holders.push((v.port, held));
            // stoppable = a refusal's eviction candidate (§10.1: pinned never
            // yields; the offer only ever names what a stop actually frees)
            if !v.pinned {
                cands.push(EvictCandidate {
                    port: v.port,
                    display: v.display.clone().or_else(|| v.model.clone()),
                    frees: held,
                    restore_cost: v
                        .model
                        .as_deref()
                        .and_then(|m| crate::estimate::resolve_weight_bytes_for(state, m, None)),
                });
            }
        }
    }
    for port in state.supervisor.spawning_ports() {
        if Some(port) == req.freeing_port || views.iter().any(|v| v.port == port) {
            continue; // already counted above
        }
        // mid-spawn: on a multi-GPU box trust the file's own pin; the config
        // was written before launch, so the budget (when granted) is there
        if !single_gpu {
            let pinned_here = state
                .supervisor
                .spec_from_config_file(&state.supervisor.server_config_path(port))
                .ok()
                .and_then(|s| s.gpu)
                .as_deref()
                .and_then(resolve_pin)
                == Some(sel);
            if !pinned_here {
                continue;
            }
        }
        let held = state
            .supervisor
            .config_vram_budget(port)
            .map(|mib| mib << 20)
            .unwrap_or(0);
        committed += held;
        if held > 0 {
            holders.push((port, held));
        }
    }
    let residual = total.saturating_sub(committed).saturating_sub(ADMIT_MARGIN);

    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    let who: Vec<String> = holders
        .iter()
        .map(|(p, held)| {
            let name = views
                .iter()
                .find(|v| v.port == *p)
                .and_then(|v| v.display.clone().or_else(|| v.model.clone()))
                .unwrap_or_else(|| format!("port {p}"));
            format!("{name} (port {p}) holds {:.1} GiB", gib(*held))
        })
        .collect();
    let others = if who.is_empty() {
        "other programs hold the rest".to_string()
    } else {
        who.join(", ")
    };

    // A verbatim start brings its own budget: that number is the ask, and it
    // has to clear two bars, not one.
    //
    // Bar 1, here: does the number fit beside the fleet.
    if let Some(need) = req.fixed_need
        && need > residual
    {
        tracing::warn!(model = %req.model, need, residual, total, "VRAM admission refused a start (configured budget over residual)");
        return Err(AdmissionRefusal {
            message: format!(
                "won't fit: this endpoint's configured vram_budget is {:.1} GiB but only {:.1} GiB of the card's {:.1} GiB is unclaimed - {}. Stop a model first, or lower vram_budget in the config file.",
                gib(need),
                gib(residual),
                gib(total),
                others,
            ),
            eviction: Some(eviction_offer(cands, residual, need)),
        });
    }
    // Bar 2 is the pricing block below, entered with the BUDGET standing in
    // for free VRAM. It used to `return Ok(None)` right here, which asked
    // only "does this number fit on the card" and never "can this number
    // serve this model at this envelope" - so a config could be admitted
    // and then refused by the runner's own planner a minute into the load,
    // with a wall of arithmetic instead of a manager-side explanation.
    // Measured: a 47.0 GiB budget on a 48 GiB card admitted a 27B at
    // 4096x32 that the runner then refused (needs 8.00 GiB of KV, 3.70
    // fits). The budget is the endpoint's world, so pricing against it
    // rather than against `residual` is the same question the runner asks.

    // Price the model. Unresolvable = nothing honest to refuse on (the
    // engine's own load gate still guards the actual load).
    let Some((path, weights, kind, workspace)) =
        crate::estimate::resolve_weights_for(state, req.model, req.artifact)
    else {
        return Ok(Admitted::default());
    };

    // The grant, estimator-priced when the file is on disk to probe. The
    // envelope is what this endpoint will actually serve (its own max_ctx /
    // max_batch, runner defaults when unset): budgets pay for the config
    // asked for, not the model's theoretical maximum - that's what leaves
    // the remainder of the card startable for the next model.
    if let Some(report) = state.probes.get(&path) {
        use paddock_estimator::{Device, Envelope, Fit, KvDtype, ModelShape, estimate};
        let mut shape = ModelShape::from_report(&report, weights, kind);
        // Match the resolved file, not an implicit default quant. The same
        // backend layout contract prices both the catalog and actual admission.
        let runtime = state
            .registry
            .catalog()
            .models
            .iter()
            .flat_map(|m| m.weights())
            .find(|a| {
                a.files
                    .first()
                    .is_some_and(|f| state.registry.models_dir().join(&f.dest) == path)
            })
            .map(|a| &a.runtime);
        if let Some(memory) = runtime.and_then(|r| r.memory.as_ref()) {
            memory.apply(&mut shape);
        }
        // Declared serving scratch (MoE staging) is pinned at load, so the
        // grant has to cover it just like the tower below.
        shape.workspace_bytes = workspace;
        let ctx_cap = req.max_ctx.unwrap_or(RUNNER_DEFAULT_MAX_CTX) as u64;
        shape.max_ctx = if shape.max_ctx == 0 {
            ctx_cap
        } else {
            shape.max_ctx.min(ctx_cap)
        };
        // A vision endpoint holds its mmproj resident from load, so the budget
        // has to cover it or the grant is short by the tower's whole file size
        // on every vision start. Only when vision is actually on: the
        // supervisor drops the mmproj on `spec.vision == Some(false)`, and
        // charging it anyway refused starts that would have fit.
        //
        // An image lane's text encoder and VAE ride the same term with no
        // switch at all - resident from load, for the endpoint's life - and
        // are priced for the DiT actually resolved, since the compact DiT
        // pairs with the compact encoder. The first image start was granted
        // 15.6 GB against 16.85 GB of weights alone because this was missing.
        let row = state
            .registry
            .catalog()
            .models
            .iter()
            .find(|m| m.id == req.model);
        let lane = row.map_or(0, |m| {
            let weights_artifact = m.weights().find(|a| {
                a.files
                    .first()
                    .is_some_and(|f| state.registry.models_dir().join(&f.dest) == path)
            });
            crate::estimate::lane_companion_bytes_for(m, &state.registry, weights_artifact)
        });
        shape.tower_bytes = lane
            + row.map_or(0, |m| {
                crate::estimate::towers_bytes_for(m, &state.registry, None, req.vision, req.audio)
            });
        // A speculating endpoint holds its drafter resident for its whole life,
        // so it comes out of the same budget as the weights. In-file MTP adds
        // no drafter bytes (already inside `weights`) but still widens the
        // verify plane, which SpecCost's default carries.
        let drafter_bytes = state
            .registry
            .catalog()
            .models
            .iter()
            .find(|m| m.id == req.model)
            .and_then(|m| {
                m.artifacts
                    .iter()
                    .find(|a| a.kind == crate::registry::ArtifactKind::Drafter)
            })
            .map_or(0, |a| a.total_size());
        let mut env = Envelope {
            concurrency: req.max_batch.unwrap_or(RUNNER_DEFAULT_MAX_BATCH).max(1) as u64,
            // The width the RUNNER will serve. A card with no FP8 tensor
            // cores has its fp8 request downgraded to f16 at load
            // (serving.rs::apply_kv_dtype), which DOUBLES the KV pool - so
            // admitting on the fp8 rate grants a budget the runner then
            // exceeds, which is exactly what this guard exists to prevent.
            kv_dtype: if req.fp8_kv.unwrap_or_else(|| {
                row.and_then(|m| m.kv_default.as_deref())
                    .is_some_and(|k| matches!(k, "fp8_e4m3" | "fp8"))
            }) && state
                .readiness
                .cc
                .is_none_or(|cc| paddock_models::gpu_support::fp8_kv((cc[0], cc[1])))
            {
                KvDtype::Fp8E4m3
            } else {
                KvDtype::F16
            },
            spec: req.spec.then(|| paddock_estimator::SpecCost {
                drafter_bytes,
                ..Default::default()
            }),
            offload: req
                .offload_ram_bytes
                .map(paddock_estimator::OffloadCost::armed),
        };
        if let Some(runtime) = runtime {
            env.kv_dtype = runtime.estimate_kv_dtype(env.kv_dtype);
        }
        // A fixed budget is the endpoint's whole world - price inside it, not
        // inside the card's residual (see bar 2 above).
        let ceiling = req.fixed_need.unwrap_or(residual);
        let device = Device {
            free_bytes: ceiling,
            total_bytes: total,
        };
        // A defaulted window is the catalog's target: fit it to this card at
        // this concurrency before pricing it (fit_window).
        let mut fitted = None;
        let est = if req.fit_ctx && shape.kind == paddock_estimator::ModelKind::Generative {
            let (ctx, est) = fit_window(shape.max_ctx, |ctx| {
                let mut at = shape.clone();
                at.max_ctx = ctx;
                estimate(&at, &env, &device)
            });
            if req.max_ctx != Some(ctx as usize) {
                tracing::info!(model = %req.model, target = ?req.max_ctx, fitted = ctx, "default window fitted to the card");
                fitted = Some(ctx as usize);
            }
            shape.max_ctx = ctx;
            est
        } else {
            estimate(&shape, &env, &device)
        };
        if let Fit::DoesNotFit { short_by_bytes } = est.fit {
            if let Some(budget) = req.fixed_need {
                tracing::warn!(model = %req.model, resident = est.resident, budget, "VRAM admission refused a start (configured budget too small for the envelope)");
                return Err(AdmissionRefusal {
                    message: format!(
                        "won't start: this endpoint's configured vram_budget is {:.1} GiB, but serving {} at max_ctx {} x max_batch {} needs at least {:.1} GiB (short {:.1} GiB). Raise vram_budget, or lower max_ctx / max_batch in the config file.",
                        gib(budget),
                        req.model,
                        req.max_ctx.unwrap_or(RUNNER_DEFAULT_MAX_CTX),
                        req.max_batch.unwrap_or(RUNNER_DEFAULT_MAX_BATCH),
                        gib(est.resident),
                        gib(short_by_bytes),
                    ),
                    // Nothing to evict: the ceiling is this file's own number,
                    // not the fleet's claim on the card.
                    eviction: None,
                });
            }
            tracing::warn!(model = %req.model, resident = est.resident, residual, total, "VRAM admission refused a spawn");
            return Err(AdmissionRefusal {
                message: format!(
                    "won't fit: this model needs at least {:.1} GiB resident but only {:.1} GiB of the card's {:.1} GiB is unclaimed (short {:.1} GiB) - {}. Stop a model first, or pick a smaller quant. (Loading anyway would page VRAM into system RAM and freeze the machine.)",
                    gib(est.resident),
                    gib(residual),
                    gib(total),
                    gib(short_by_bytes),
                    others,
                ),
                eviction: Some(eviction_offer(cands, residual, est.resident)),
            });
        }
        // resident must fit; the KV pool takes what this envelope can use of
        // the rest - grant exactly that, never the whole residual. Floor at
        // weights + 1.5 GiB: the engine's own load gate wants weights + 1 GiB
        // inside the budget, and an ENCODER's resident (weights + context, no
        // decode pools) sits under that while its transient activations still
        // need real headroom.
        // A verbatim start priced clean: keep the file's own number, never
        // rewrite it from under the operator.
        if req.fixed_need.is_some() {
            return Ok(Admitted {
                grant: None,
                max_ctx: fitted,
            });
        }
        let grant_mib = (est.resident + est.kv_pool)
            .max(weights + (3 << 29))
            .min(residual)
            .div_ceil(1 << 20);
        tracing::info!(
            model = %req.model,
            grant_mib,
            resident = est.resident,
            kv_pool = est.kv_pool,
            residual,
            max_ctx = shape.max_ctx,
            "VRAM admission granted a budget"
        );
        return Ok(Admitted {
            grant: Some(grant_mib),
            max_ctx: fitted,
        });
    }

    // No probeable file yet (pre-download spawn, foreign path): the plain
    // arithmetic - exact weights + the vision tower's file size + a
    // conservative floor - admits or refuses, and no budget is written (the
    // runner sizes free-at-load; committed accounting falls back to its live
    // ledger). The companion bytes - the mmproj, an image lane's text
    // encoder and VAE - are known from the catalog even here: the file does
    // not have to exist for its size to be a fact.
    let mmproj = state
        .registry
        .catalog()
        .models
        .iter()
        .find(|m| m.id == req.model)
        .map_or(0, |m| {
            // the floor charges the picture tower whatever the switch (as
            // it always did); a split model's audio tower by its own
            crate::estimate::towers_bytes_for(m, &state.registry, None, true, req.audio)
                + crate::estimate::lane_companion_bytes_for(m, &state.registry, None)
        });
    // Unprobeable + a configured budget: bar 1 already cleared it against the
    // fleet, and there is no shape to price bar 2 with. The engine's own load
    // gate is the remaining guard, as it was before budgets.
    if req.fixed_need.is_some() {
        return Ok(Admitted::default());
    }
    // the declared serving workspace is pinned at load like the companions
    let need = weights + mmproj + workspace + ADMIT_FLOOR;
    if need > residual {
        tracing::warn!(model = %req.model, need, residual, total, "VRAM admission refused a spawn");
        return Err(AdmissionRefusal {
            message: format!(
                "won't fit: this model needs at least {:.1} GiB but only {:.1} GiB of the card's {:.1} GiB is unclaimed - {}. Stop a model first, or pick a smaller quant. (Loading anyway would page VRAM into system RAM and freeze the machine.)",
                gib(need),
                gib(residual),
                gib(total),
                others,
            ),
            eviction: Some(eviction_offer(cands, residual, need)),
        });
    }
    Ok(Admitted::default())
}

fn pin_budget_text(text: &str, grant: u64) -> String {
    crate::automatic_budget::pin(text, grant)
}

async fn metal_admission(
    state: &Arc<AppState>,
    req: AdmitReq<'_>,
) -> Result<Admitted, AdmissionRefusal> {
    use paddock_estimator::{Device, Envelope, KvDtype};
    let refuse = |message| AdmissionRefusal {
        message,
        eviction: None,
    };
    let (snapshot, free) = crate::metal_memory::available(state, req.freeing_port)
        .await
        .ok_or_else(|| {
            refuse("Cannot read Apple unified-memory availability; no model was started.".into())
        })?;
    let host = req.offload_ram_bytes.unwrap_or(0);
    let free = free.saturating_sub(host);
    if req.fixed_need.is_some_and(|need| need > free) {
        return Err(refuse(format!(
            "The configured memory budget exceeds the {:.1} GiB available after other models and RAM caches. Lower the budget or stop another instance.",
            free as f64 / (1u64 << 30) as f64
        )));
    }
    let identified = state
        .registry
        .identify_weights(std::path::Path::new(req.model));
    let model = state.registry.catalog_of(req.model).or_else(|| {
        identified
            .as_ref()
            .and_then(|(m, _)| state.registry.catalog_of(m))
    });
    let artifact = model.and_then(|m| {
        req.artifact
            .or_else(|| identified.as_ref().map(|(_, a)| a.as_str()))
            .and_then(|a| m.artifact(a))
            .or_else(|| m.default_weights_for_backend("metal", None))
    });
    let (mut shape, env) = if let (Some(model), Some(artifact)) = (model, artifact) {
        let shape = crate::estimate::artifact_shape(state, model, artifact, req.vision, req.audio);
        (
            shape,
            Envelope {
                concurrency: req.max_batch.unwrap_or(1).max(1) as u64,
                kv_dtype: artifact.runtime.estimate_kv_dtype(KvDtype::F16),
                spec: crate::estimate::artifact_spec(model, artifact, req.spec),
                offload: req
                    .offload_ram_bytes
                    .map(paddock_estimator::OffloadCost::armed),
            },
        )
    } else {
        (
            None,
            Envelope {
                concurrency: req.max_batch.unwrap_or(1).max(1) as u64,
                kv_dtype: KvDtype::F16,
                spec: None,
                offload: None,
            },
        )
    };
    let ceiling = req.fixed_need.unwrap_or(free);
    let mut fitted = None;
    let required = if let Some(shape) = &mut shape {
        shape.max_ctx = shape.max_ctx.min(req.max_ctx.unwrap_or(32768) as u64);
        let device = Device {
            free_bytes: ceiling,
            total_bytes: snapshot.limit,
        };
        // The cache geometry follows the window, so every try re-derives it.
        let at = |ctx: u64| {
            let mut s = shape.clone();
            s.max_ctx = ctx;
            if let Some(model) = model {
                crate::estimate::metal_cache_shape(&mut s, model, &env);
            }
            s
        };
        // Same fit as CUDA admission: a defaulted window is a target.
        let ctx = if req.fit_ctx && shape.kind == paddock_estimator::ModelKind::Generative {
            let ctx = fit_window(shape.max_ctx, |ctx| {
                crate::estimate::backend_estimate("metal", &at(ctx), &env, &device)
            })
            .0;
            if req.max_ctx != Some(ctx as usize) {
                fitted = Some(ctx as usize);
            }
            ctx
        } else {
            shape.max_ctx
        };
        let priced = at(ctx);
        *shape = priced;
        let estimate = crate::estimate::backend_estimate("metal", shape, &env, &device);
        if matches!(estimate.fit, paddock_estimator::Fit::DoesNotFit { .. })
            || (shape.kind == paddock_estimator::ModelKind::Generative
                && estimate.max_ctx < shape.max_ctx)
        {
            return Err(refuse(format!(
                "This model does not fit the available unified-memory budget at {} tokens and {} concurrent requests. Lower context/concurrency or stop another instance.",
                shape.max_ctx, env.concurrency
            )));
        }
        (estimate.resident + estimate.kv_pool).max(shape.weight_bytes + (1 << 30))
    } else {
        // Unknown/imported layouts get a real hard ceiling, never an unbounded
        // load. The runner validates the layout inside that reserved budget.
        let weights = artifact
            .map(|a| a.total_size())
            .or_else(|| crate::estimate::resolve_weight_bytes_for(state, req.model, req.artifact))
            .ok_or_else(|| {
                refuse(
                    "Cannot determine checkpoint weight size. Select an installed checkpoint path or a catalog export."
                        .into(),
                )
            })?;
        if weights.saturating_add(3 << 30) > ceiling {
            return Err(refuse(
                "The checkpoint weights and runtime allowance exceed available unified memory."
                    .into(),
            ));
        }
        ceiling
    };
    if required > ceiling || ceiling < 1 << 20 {
        return Err(refuse("Insufficient unified memory for this model.".into()));
    }
    Ok(Admitted {
        grant: req
            .fixed_need
            .is_none()
            .then_some(required.div_ceil(1 << 20).min(free >> 20)),
        max_ctx: fitted,
    })
}

// ── runner supervision (doc §3, §5) ─────────────────────────────────────────

/// The reconciled runner table: own + adopted, live-queried, each row joined
/// with its VRAM attribution (NVML outside view + ledger inside view + drift)
/// from the reconciler's latest sample.
async fn runners_list(State(state): State<Arc<AppState>>) -> Response {
    // one builder with the SSE push (crate::push::fleet_rows), so the pushed
    // state can never drift from the polled state
    Json(crate::push::fleet_rows(&state).await).into_response()
}

/// The desired-state election set (managed.toml, §11.2): what respawns on
/// boot. Small since the config-file split: each row points at the endpoint's
/// own servers/<port>.toml - The configuration.
async fn elections_list(State(state): State<Arc<AppState>>) -> Response {
    let Some(el) = &state.elections else {
        return Json(serde_json::json!({ "path": null, "elections": [] })).into_response();
    };
    let rows: Vec<serde_json::Value> = el
        .list()
        .into_iter()
        .map(|e| {
            serde_json::json!({
                "model": e.model,
                "artifact": e.artifact,
                "port": e.port,
                "config": e.config.display().to_string(),
                "runner_version": e.runner_version,
                "pinned": e.pinned,
            })
        })
        .collect();
    Json(serde_json::json!({ "path": el.path().display().to_string(), "elections": rows }))
        .into_response()
}

/// Studio chat: forward the Responses request to the runner and stream its
/// answer back (SSE or JSON, verbatim). The manager is the CLIENT here - it
/// adds the runner key it issued at spawn; bytes stream through untouched.
/// Deliberately no body rewriting: tools/web search are the RUNNER's own
/// per-model config (its config file), and the Studio declares only what an
/// endpoint advertises on /api/server.
async fn relay_responses(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(port): axum::extract::Path<u16>,
    body: axum::body::Bytes,
) -> Response {
    studio_responses(state, port, body).await
}

/// Shared authenticated client for the browser relay and embedded native Studio.
/// The native adapter validates endpoint identity before calling; neither host
/// receives the runner key. No additional management listener is required.
pub async fn studio_responses(
    state: Arc<AppState>,
    port: u16,
    body: axum::body::Bytes,
) -> Response {
    relay_v1(state, port, "v1/responses", body).await
}

/// Playground relays - same contract as chat: verbatim bytes, runner key
/// added, streaming passthrough (embeddings/rerank answer in one JSON body,
/// but nothing here needs to know that).
async fn relay_embeddings(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(port): axum::extract::Path<u16>,
    body: axum::body::Bytes,
) -> Response {
    relay_v1(state, port, "v1/embeddings", body).await
}

async fn relay_rerank(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(port): axum::extract::Path<u16>,
    body: axum::body::Bytes,
) -> Response {
    relay_v1(state, port, "v1/rerank", body).await
}

/// Structured reads (the Studio's Reads page): one JSON body in, one JSON
/// answer out, on a block-diffusion runner - the same relay shape as rerank.
async fn relay_systemone(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(port): axum::extract::Path<u16>,
    body: axum::body::Bytes,
) -> Response {
    relay_v1(state, port, "v1/systemone", body).await
}

async fn relay_masks(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(port): axum::extract::Path<u16>,
    body: axum::body::Bytes,
) -> Response {
    relay_v1(state, port, "v1/masks", body).await
}

async fn relay_mask_session(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(port): axum::extract::Path<u16>,
    body: axum::body::Bytes,
) -> Response {
    relay_v1(state, port, "v1/masks/sessions", body).await
}

/// A video session id is the runner's number; anything else never reaches
/// the runner's path (the refusal, when it is not one).
fn bad_mask_session_id(id: &str) -> Option<Response> {
    (id.is_empty() || id.len() > 20 || !id.bytes().all(|b| b.is_ascii_digit()))
        .then(|| relay_err(StatusCode::BAD_REQUEST, "not a video session id".into()))
}

async fn relay_mask_frame(
    State(state): State<Arc<AppState>>,
    axum::extract::Path((port, id)): axum::extract::Path<(u16, String)>,
    body: axum::body::Bytes,
) -> Response {
    if let Some(r) = bad_mask_session_id(&id) {
        return r;
    }
    relay_v1(state, port, &format!("v1/masks/sessions/{id}/frames"), body).await
}

async fn relay_mask_finish(
    State(state): State<Arc<AppState>>,
    axum::extract::Path((port, id)): axum::extract::Path<(u16, String)>,
) -> Response {
    if let Some(r) = bad_mask_session_id(&id) {
        return r;
    }
    relay_v1(
        state,
        port,
        &format!("v1/masks/sessions/{id}/finish"),
        axum::body::Bytes::new(),
    )
    .await
}

async fn relay_mask_drop(
    State(state): State<Arc<AppState>>,
    axum::extract::Path((port, id)): axum::extract::Path<(u16, String)>,
) -> Response {
    if let Some(r) = bad_mask_session_id(&id) {
        return r;
    }
    relay_method(
        state,
        port,
        reqwest::Method::DELETE,
        &format!("v1/masks/sessions/{id}"),
        "application/json",
        axum::body::Bytes::new(),
    )
    .await
}

async fn relay_tabular_predictions(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(port): axum::extract::Path<u16>,
    body: axum::body::Bytes,
) -> Response {
    relay_v1(state, port, "v1/tabular/predictions", body).await
}

async fn relay_tabular_contexts(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(port): axum::extract::Path<u16>,
    body: axum::body::Bytes,
) -> Response {
    relay_v1(state, port, "v1/tabular/contexts", body).await
}

/// Release a fitted context. The id goes into the runner's URL, so only the
/// characters a runner mints (`ctx_` + hex) pass - never a path segment of
/// the caller's choosing.
async fn relay_tabular_release(
    State(state): State<Arc<AppState>>,
    axum::extract::Path((port, id)): axum::extract::Path<(u16, String)>,
) -> Response {
    if id.is_empty()
        || id.len() > 64
        || !id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
    {
        return relay_err(StatusCode::BAD_REQUEST, "not a tabular context id".into());
    }
    relay_method(
        state,
        port,
        reqwest::Method::DELETE,
        &format!("v1/tabular/contexts/{id}"),
        "application/json",
        axum::body::Bytes::new(),
    )
    .await
}

async fn relay_count_tokens(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(port): axum::extract::Path<u16>,
    body: axum::body::Bytes,
) -> Response {
    relay_v1(state, port, "v1/messages/count_tokens", body).await
}

async fn relay_extract(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(port): axum::extract::Path<u16>,
    body: axum::body::Bytes,
) -> Response {
    relay_v1(state, port, "api/extract", body).await
}

/// Transcriptions are the one Studio relay that is not JSON: the endpoint
/// takes multipart/form-data, and a multipart body without its own
/// `boundary=` parameter is unparseable. So this one forwards the caller's
/// content-type verbatim instead of stamping application/json over it.
async fn relay_transcriptions(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(port): axum::extract::Path<u16>,
    headers: axum::http::HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    let ct = headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/octet-stream")
        .to_owned();
    relay_raw(state, port, "v1/audio/transcriptions", &ct, body).await
}

/// Same multipart-verbatim rule as transcriptions - the alignment endpoint
/// carries an audio file too.
async fn relay_alignments(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(port): axum::extract::Path<u16>,
    headers: axum::http::HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    let ct = headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/octet-stream")
        .to_owned();
    relay_raw(state, port, "v1/audio/alignments", &ct, body).await
}

async fn relay_diarizations(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(port): axum::extract::Path<u16>,
    headers: axum::http::HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    let ct = headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/octet-stream");
    relay_raw(state, port, "v1/audio/diarizations", ct, body).await
}

async fn relay_diarization_live(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(port): axum::extract::Path<u16>,
    axum::extract::RawQuery(q): axum::extract::RawQuery,
    ws: WebSocketUpgrade,
) -> Response {
    let key = state.supervisor.runner_key(port).await;
    let query = q
        .filter(|s| !s.is_empty())
        .map(|s| format!("?{s}"))
        .unwrap_or_default();
    let url = format!("ws://127.0.0.1:{port}/v1/audio/diarizations/stream{query}");
    ws.max_message_size(32000)
        .max_frame_size(32000)
        .write_buffer_size(4096)
        .max_write_buffer_size(2 * 1024 * 1024)
        .on_upgrade(move |socket| async move {
            let _ = tokio::time::timeout(
                std::time::Duration::from_secs(900),
                realtime_ws(socket, url, key),
            )
            .await;
        })
}

/// The Studio's Images page: JSON in, base64 images out, same verbatim
/// contract as chat.
async fn relay_image_generations(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(port): axum::extract::Path<u16>,
    body: axum::body::Bytes,
) -> Response {
    relay_v1(state, port, "v1/images/generations", body).await
}

/// Edits carry the source image(s) as multipart parts - the transcriptions
/// rule: the boundary lives in the caller's content-type, so it is forwarded
/// as sent.
async fn relay_image_edits(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(port): axum::extract::Path<u16>,
    headers: axum::http::HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    let ct = headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/octet-stream")
        .to_owned();
    relay_raw(state, port, "v1/images/edits", &ct, body).await
}

async fn relay_v1(
    state: Arc<AppState>,
    port: u16,
    path: &str,
    body: axum::body::Bytes,
) -> Response {
    relay_raw(state, port, path, "application/json", body).await
}

async fn relay_raw(
    state: Arc<AppState>,
    port: u16,
    path: &str,
    content_type: &str,
    body: axum::body::Bytes,
) -> Response {
    relay_method(state, port, reqwest::Method::POST, path, content_type, body).await
}

async fn relay_method(
    state: Arc<AppState>,
    port: u16,
    method: reqwest::Method,
    path: &str,
    content_type: &str,
    body: axum::body::Bytes,
) -> Response {
    let key = state.supervisor.runner_key(port).await;
    // No total timeout: a long generation streams for minutes. Connect
    // timeout keeps a dead runner from hanging the Studio.
    let client = match reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(5))
        .build()
    {
        Ok(c) => c,
        Err(e) => return relay_err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    let mut req = client
        .request(method, format!("http://127.0.0.1:{port}/{path}"))
        .header(axum::http::header::CONTENT_TYPE, content_type)
        .body(body);
    if let Some(k) = key {
        req = req.bearer_auth(k);
    }
    let res = match req.send().await {
        Ok(r) => r,
        Err(e) => {
            return relay_err(
                StatusCode::BAD_GATEWAY,
                format!("runner on port {port} is not answering: {e}"),
            );
        }
    };
    let status = StatusCode::from_u16(res.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let content_type = res
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_owned();
    let mut out = Response::new(axum::body::Body::from_stream(res.bytes_stream()));
    *out.status_mut() = status;
    if let Ok(v) = axum::http::HeaderValue::from_str(&content_type) {
        out.headers_mut()
            .insert(axum::http::header::CONTENT_TYPE, v);
    }
    out
}

/// The selected runner's /api/server (model facts: max_ctx, reasoning style,
/// pdf, vision) fetched by the manager as a client, for the Studio.
async fn relay_server(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(port): axum::extract::Path<u16>,
) -> Response {
    let key = state.supervisor.runner_key(port).await;
    let client = reqwest::Client::new();
    let mut req = client
        .get(format!("http://127.0.0.1:{port}/api/server"))
        .timeout(std::time::Duration::from_secs(5));
    if let Some(k) = key {
        req = req.bearer_auth(k);
    }
    match req.send().await {
        Ok(r) => {
            let status =
                StatusCode::from_u16(r.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
            match r.json::<serde_json::Value>().await {
                Ok(v) => (status, Json(v)).into_response(),
                Err(e) => relay_err(StatusCode::BAD_GATEWAY, e.to_string()),
            }
        }
        Err(e) => relay_err(
            StatusCode::BAD_GATEWAY,
            format!("runner on port {port} is not answering: {e}"),
        ),
    }
}

/// The runner's own /openapi.json, fetched by the manager as a client. The
/// document is open on the runner (like /healthz), but the key rides along
/// anyway so this stays a verbatim copy of relay_server's client stance.
async fn relay_openapi(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(port): axum::extract::Path<u16>,
) -> Response {
    let key = state.supervisor.runner_key(port).await;
    let client = reqwest::Client::new();
    let mut req = client
        .get(format!("http://127.0.0.1:{port}/openapi.json"))
        .timeout(std::time::Duration::from_secs(5));
    if let Some(k) = key {
        req = req.bearer_auth(k);
    }
    match req.send().await {
        Ok(r) => {
            let status =
                StatusCode::from_u16(r.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
            match r.json::<serde_json::Value>().await {
                Ok(v) => (status, Json(v)).into_response(),
                Err(e) => relay_err(StatusCode::BAD_GATEWAY, e.to_string()),
            }
        }
        Err(e) => relay_err(
            StatusCode::BAD_GATEWAY,
            format!("runner on port {port} is not answering: {e}"),
        ),
    }
}

fn relay_err(status: StatusCode, msg: String) -> Response {
    (
        status,
        Json(serde_json::json!({"error": {"type": "relay_error", "message": msg}})),
    )
        .into_response()
}

/// Resolve a human-in-the-loop MCP approval on the runner holding the parked
/// agent loop (in-process state - only that runner can answer it).
async fn relay_approval(
    State(state): State<Arc<AppState>>,
    axum::extract::Path((port, id)): axum::extract::Path<(u16, String)>,
    body: axum::body::Bytes,
) -> Response {
    let key = state.supervisor.runner_key(port).await;
    let client = reqwest::Client::new();
    let mut req = client
        .post(format!("http://127.0.0.1:{port}/api/mcp-approvals/{id}"))
        .header(axum::http::header::CONTENT_TYPE, "application/json")
        .timeout(std::time::Duration::from_secs(10))
        .body(body);
    if let Some(k) = key {
        req = req.bearer_auth(k);
    }
    match req.send().await {
        Ok(r) => {
            let status =
                StatusCode::from_u16(r.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
            match r.json::<serde_json::Value>().await {
                Ok(v) => (status, Json(v)).into_response(),
                Err(e) => relay_err(StatusCode::BAD_GATEWAY, e.to_string()),
            }
        }
        Err(e) => relay_err(
            StatusCode::BAD_GATEWAY,
            format!("runner on port {port} is not answering: {e}"),
        ),
    }
}

#[derive(serde::Deserialize)]
struct ActivityQuery {
    limit: Option<usize>,
    /// Pagination: records strictly older than this unix-millis timestamp.
    before: Option<i64>,
    port: Option<u16>,
    model: Option<String>,
    session: Option<String>,
}

/// Collected request records (doc §8.1), newest first. Filters are optional;
/// rows are the runner's full semconv-named records plus the collector's port.
async fn activity_list(
    State(state): State<Arc<AppState>>,
    axum::extract::Query(q): axum::extract::Query<ActivityQuery>,
) -> Response {
    let limit = q.limit.unwrap_or(100).min(1000);
    match state.db.list_activity(
        limit,
        q.before,
        q.port,
        q.model.as_deref(),
        q.session.as_deref(),
    ) {
        Ok(rows) => Json(serde_json::json!({ "events": rows })).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(
                serde_json::json!({"error": {"type": "internal_error", "message": e.to_string()}}),
            ),
        )
            .into_response(),
    }
}

#[derive(serde::Deserialize)]
struct UsageHistoryQuery {
    /// Window in unix millis; defaults: to = now, from = to - 24 h.
    from: Option<i64>,
    to: Option<i64>,
    port: Option<u16>,
    /// Poll optimization: read buckets only from here (the client caches the
    /// closed ones - they never change again). Gaps and generations always
    /// come for the whole window: a gap covering yesterday can be *inserted*
    /// today when the manager reattaches, so they are never cacheable.
    buckets_from: Option<i64>,
}

const DAY_MS: i64 = 86_400_000;

/// The usage timeline: SQL over `usage_bucket` only -
/// the grain is picked from the span server-side so a year view never ships
/// 5-minute rows to the browser to discard 99% of them.
async fn usage_history(
    State(state): State<Arc<AppState>>,
    axum::extract::Query(q): axum::extract::Query<UsageHistoryQuery>,
) -> Response {
    let now = chrono::Utc::now().timestamp_millis();
    let to = q.to.unwrap_or(now);
    let from = q.from.unwrap_or(to - DAY_MS);
    if from >= to {
        return relay_err(StatusCode::BAD_REQUEST, "from must be before to".into());
    }
    // Grain from span: 5-minute rows only for short windows (they are also the
    // grain with finite retention), hourly beyond, and hourly REGROUPED into
    // 6 h / daily slots for quarter/year spans - ≤ ~1.1k slots whatever the ask.
    let span = to - from;
    let (grain, group_ms) = match span {
        s if s <= 3 * DAY_MS => ("5m", 300_000),
        s if s <= 45 * DAY_MS => ("1h", 3_600_000),
        s if s <= 200 * DAY_MS => ("1h", 6 * 3_600_000),
        _ => ("1h", DAY_MS),
    };
    // Everything before this boundary can never change again: folds land in
    // the bucket of their scrape time, so once the wall clock (plus a slack
    // for an in-flight scrape) passes a boundary, earlier rows are immutable.
    let closed_through_ms = (now - 30_000).div_euclid(group_ms) * group_ms;
    let buckets_from = q.buckets_from.unwrap_or(from).max(from);
    let port = q.port.map_or(-1, i64::from);
    let result = state
        .db
        .usage_history(grain, group_ms, buckets_from, to, port)
        .and_then(|buckets| {
            let gaps = state.db.usage_gaps_in(from, to, port)?;
            let generations = state.db.usage_generations_in(from, to, port)?;
            let extent = state.db.usage_extent()?;
            // Same window, same grain, one fetch - the spend panel pans and
            // zooms with the rest of the board instead of drifting off it.
            let web = state
                .db
                .web_history(grain, group_ms, buckets_from, to, port)?;
            Ok((buckets, gaps, generations, extent, web))
        });
    match result {
        Ok((buckets, gaps, generations, extent, web)) => Json(serde_json::json!({
            "grain_ms": group_ms,
            "closed_through_ms": closed_through_ms,
            "now_ms": now,
            // the all-history left edge: the pan/zoom axis spans
            // from here to now, whatever window this response covers
            "extent_from_ms": extent,
            "buckets": buckets,
            "gaps": gaps,
            "generations": generations,
            "web": web,
        }))
        .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(
                serde_json::json!({"error": {"type": "internal_error", "message": e.to_string()}}),
            ),
        )
            .into_response(),
    }
}

/// Explicit user purge - drop all collected activity now (the §8.1 retention
/// contract's manual end; the collector keeps collecting unless disabled).
async fn activity_purge(State(state): State<Arc<AppState>>) -> Response {
    match state.db.clear_activity() {
        Ok(n) => Json(serde_json::json!({ "purged": n })).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(
                serde_json::json!({"error": {"type": "internal_error", "message": e.to_string()}}),
            ),
        )
            .into_response(),
    }
}

/// The reverse of `servers_preview`: config TEXT in, the settings the Simple tab
/// binds out - model identity included.
///
/// This endpoint exists to DELETE a rule from the browser, not to add a feature.
/// The Simple tab edits an unsaved buffer, so it cannot read the saved file, and
/// the answer used to be re-deriving the model identity client-side from the
/// weights filename. That produced two independent disagreements in one day:
/// `/api/servers` against `heal_spec_identity`, then the browser
/// against `identify_weights` - the second one showing "Qwen 3.8 27B" for the
/// forced aligner, because a browser copy of the rule matched every artifact
/// kind where the real rule matches weights only. Both were fixed by making the
/// copy agree; neither fix stopped the next copy from drifting.
///
/// So there is no copy now. `Supervisor::project_config_text` is the one
/// implementation, the same one `spec_from_config_file` uses for a start.
///
/// Body: `{ "toml": "<the buffer>" }`. A buffer that does not parse is a 400
/// carrying the parser's own words - the Simple tab shows that instead of
/// silently rendering an empty form, which is how the BOM bug hid.
async fn servers_project(
    State(state): State<Arc<AppState>>,
    Json(body): Json<ProjectReq>,
) -> Response {
    match state.supervisor.project_config_text(&body.toml) {
        Ok(p) => Json(p).into_response(),
        Err(msg) => relay_err(StatusCode::BAD_REQUEST, msg),
    }
}

#[derive(serde::Deserialize)]
struct ProjectReq {
    toml: String,
}

/// The config file a spec would produce - the Start/Edit page's live preview.
/// Rendered by the same serializer Save uses, so it is byte-identical to what
/// lands in servers/<port>.toml; never a second implementation to drift.
///
/// Body = `SpawnSpec` plus two optional fields:
///
/// - `merge_with` - the file text as it stands. The answer then lays the render
///   over it: owned keys from the render, everything else preserved, comments and
///   layout included. This is what makes the edit page's three tabs one document:
///   the Simple tab has no TOML of its own, so it asks for its own settings as
///   text before handing over to the Advanced/file tabs.
/// - `for_edit` - this port already exists, so inherit the identity the editor
///   never shows (API key, GPU pin, fp8 planes, runner-version pin) exactly as a
///   takeover would. Without it the preview would omit the API key, and saving
///   that text verbatim would revoke every client's credential.
async fn servers_preview(
    State(state): State<Arc<AppState>>,
    Json(mut body): Json<serde_json::Value>,
) -> Response {
    let obj = body.as_object_mut();
    let merge_with = obj
        .as_ref()
        .and_then(|o| o.get("merge_with"))
        .and_then(|v| v.as_str().map(str::to_owned));
    let for_edit = obj
        .as_ref()
        .and_then(|o| o.get("for_edit"))
        .and_then(serde_json::Value::as_bool)
        == Some(true);
    // Connector membership the caller has STAGED but not yet committed through
    // the scope API. Materialized here by `entry_from_row` - the same function
    // the scope API uses - so a preview shows the entries a save would write,
    // including the OAuth bearer, rather than the Studio guessing at the shape.
    let staged: Option<Vec<String>> = obj
        .as_ref()
        .and_then(|o| o.get("connectors"))
        .and_then(serde_json::Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect()
        });
    if let Some(o) = obj {
        o.remove("merge_with");
        o.remove("for_edit");
        o.remove("connectors");
    }
    let mut spec: crate::supervisor::SpawnSpec = match serde_json::from_value(body) {
        Ok(s) => s,
        Err(e) => return relay_err(StatusCode::BAD_REQUEST, format!("bad spec: {e}")),
    };
    if let Some(ids) = staged {
        // library-owned entries are replaced wholesale; hand-written MCP rows
        // (no connector_id) are the caller's and stay exactly as sent
        spec.mcp_servers.retain(|e| e.get("connector_id").is_none());
        for id in ids {
            if let Ok(Some(row)) = state.db.get_connector(&id) {
                spec.mcp_servers
                    .push(crate::connectors::entry_from_row(&id, &row));
            }
        }
    }
    let rendered = if for_edit {
        let Some(port) = spec.port else {
            return relay_err(StatusCode::BAD_REQUEST, "for_edit needs a port".to_string());
        };
        state.supervisor.render_spec_config(port, spec).await
    } else {
        state
            .supervisor
            .preview_config(spec)
            .await
            .map_err(|e| e.to_string())
    };
    let toml = match rendered {
        Ok(t) => t,
        Err(e) => return relay_err(StatusCode::BAD_REQUEST, e),
    };
    let toml = match merge_with {
        Some(cur) => match crate::supervisor::merge_owned_keys(&cur, &toml) {
            Ok(m) => m,
            Err(e) => return relay_err(StatusCode::BAD_REQUEST, e),
        },
        None => toml,
    };
    if let Err(error) = state.supervisor.validate_backend_config(&toml) {
        return relay_err(StatusCode::BAD_REQUEST, error);
    }
    Json(serde_json::json!({ "toml": toml })).into_response()
}

/// Spawn a runner. Body = `SpawnSpec` (model id/name/path + options). Blocks
/// through pull + load + health-gate - the caller sees either a healthy
/// endpoint or the actual failure with the runner's log tail.
async fn runners_spawn(
    State(state): State<Arc<AppState>>,
    Json(mut spec): Json<crate::supervisor::SpawnSpec>,
) -> Response {
    let _admission = state.admission.lock().await;
    // caller-approved evictions first (the confirmed 507 offer)
    if !spec.on_demand()
        && let Err(e) = perform_evictions(&state, &spec.evict).await
    {
        return relay_err(StatusCode::CONFLICT, e);
    }
    let fit_ctx = pin_envelope(&mut spec, &state.registry);
    let req = AdmitReq {
        fit_ctx,
        ..AdmitReq::for_spec(&spec, None)
    };
    match vram_admission(&state, req).await {
        Err(msg) => return admission_refused(msg),
        // the grant becomes the endpoint's vram_budget (config-file field);
        // an explicit caller value was already admitted verbatim. A fitted
        // default window replaces the target it came from.
        Ok(admitted) => admitted.apply(&mut spec),
    }
    // every-server connectors join a new endpoint's config at birth (existing
    // configs were rewritten when the checkbox flipped) - a chat feature, so
    // not on an image lane, which serves no tool-calling surface to wire
    // them into
    if !crate::estimate::is_image_lane(&state.registry, &spec.model) {
        spec.mcp_servers.extend(crate::connectors::system_entries(
            &state.db,
            &spec.mcp_servers,
        ));
    }
    // A user asked for this start - note it for the lifecycle band the
    // collector opens (auto-allocated ports miss the note; the band then
    // honestly reads "cause unobserved" rather than guessing).
    if let Some(p) = spec.port {
        let _ = state.db.note_start_cause(p, "manual");
    }
    match state.supervisor.spawn(spec).await {
        Ok(view) => (StatusCode::CREATED, Json(view)).into_response(),
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({"error": {"type": "spawn_failed", "reason": e.reason(), "message": e.to_string()}})),
        )
            .into_response(),
    }
}

#[derive(serde::Deserialize)]
struct PinBody {
    pinned: bool,
}

/// §10.1 pin toggle: a pinned runner is never auto-stopped to make room and
/// its VRAM leaves the estimator's reclaimable figure. Persists through the
/// election (when one exists) so it survives a manager restart.
async fn runners_pin(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(port): axum::extract::Path<u16>,
    Json(body): Json<PinBody>,
) -> Response {
    match state.supervisor.set_pinned(port, body.pinned).await {
        Ok(()) => Json(serde_json::json!({ "port": port, "pinned": body.pinned })).into_response(),
        Err(e) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": {"type": "not_found_error", "message": e}})),
        )
            .into_response(),
    }
}

#[derive(serde::Deserialize)]
struct PersistBody {
    persist: bool,
}

/// Start-on-boot toggle: records/removes the managed.toml election for a
/// RUNNING runner, from the spawn spec the supervisor retained. Adopted
/// runners get the honest refusal (their config is not ours to guess).
async fn runners_persist(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(port): axum::extract::Path<u16>,
    Json(body): Json<PersistBody>,
) -> Response {
    match state.supervisor.set_persist(port, body.persist).await {
        Ok(()) => {
            Json(serde_json::json!({ "port": port, "persist": body.persist })).into_response()
        }
        Err(e) => (
            StatusCode::CONFLICT,
            Json(serde_json::json!({"error": {"type": "invalid_request_error", "message": e}})),
        )
            .into_response(),
    }
}

#[derive(serde::Deserialize)]
struct StopQuery {
    /// Drain timeout before escalation; default 30s.
    timeout_ms: Option<u64>,
}

/// Stop the runner on a port: in-band drain+shutdown, wait for process exit.
async fn runners_stop(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(port): axum::extract::Path<u16>,
    axum::extract::Query(q): axum::extract::Query<StopQuery>,
) -> Response {
    match state
        .supervisor
        .stop(port, q.timeout_ms.unwrap_or(30_000))
        .await
    {
        Ok(outcome) => {
            // The collector's observational close cannot tell a clean stop
            // from a crash; this route can.
            let _ = state.db.stamp_end_cause(port, "stopped");
            Json(serde_json::json!({ "port": port, "outcome": outcome })).into_response()
        }
        Err(e) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": {"type": "not_found_error", "message": e}})),
        )
            .into_response(),
    }
}

/// Same-port takeover (§5): drain + exit the incumbent, spawn the new model
/// on the same port. Clients keep their base_url; outage = model-load time.
/// `expect_config_hash` (from GET /api/servers/{port}/file) makes the edit
/// optimistic-concurrent: a file that moved since the edit opened refuses.
async fn runners_switch(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(port): axum::extract::Path<u16>,
    Json(mut body): Json<serde_json::Value>,
) -> Response {
    let expect_hash = body
        .as_object_mut()
        .and_then(|o| o.remove("expect_config_hash"))
        .and_then(|v| v.as_str().map(str::to_owned));
    // `apply` splits saving from applying. Absent = "restart",
    // the behaviour this route has always had. "defer" writes the file and
    // leaves the process alone - the edit page's "Save without restarting",
    // and the only way a STOPPED endpoint can be edited without being started
    // by the edit.
    let defer = body
        .as_object_mut()
        .and_then(|o| o.remove("apply"))
        .and_then(|v| v.as_str().map(str::to_owned))
        .as_deref()
        == Some("defer");
    let mut spec: crate::supervisor::SpawnSpec = match serde_json::from_value(body) {
        Ok(s) => s,
        Err(e) => return relay_err(StatusCode::BAD_REQUEST, format!("bad spec: {e}")),
    };
    if defer {
        let _admission = state.admission.lock().await;
        return match state
            .supervisor
            .write_spec_config(port, spec, expect_hash)
            .await
        {
            Ok(path) => Json(serde_json::json!({
                "port": port,
                "applied": "saved",
                "config": path.display().to_string(),
            }))
            .into_response(),
            Err(e) => (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({"error": {"type": "config_write_failed", "message": e}})),
            )
                .into_response(),
        };
    }
    // caller-approved evictions first (never the takeover's own incumbent -
    // that one drains as part of the switch itself)
    let evict: Vec<u16> = spec.evict.iter().copied().filter(|p| *p != port).collect();
    if !spec.on_demand()
        && let Err(e) = perform_evictions(&state, &evict).await
    {
        return relay_err(StatusCode::CONFLICT, e);
    }
    // a takeover frees its own incumbent - that VRAM counts as available;
    // the edit gets a FRESH grant (its envelope may have changed)
    let fit_ctx = pin_envelope(&mut spec, &state.registry);
    let _admission = state.admission.lock().await;
    let req = AdmitReq {
        fit_ctx,
        ..AdmitReq::for_spec(&spec, Some(port))
    };
    match vram_admission(&state, req).await {
        Err(msg) => return admission_refused(msg),
        Ok(admitted) => admitted.apply(&mut spec),
    }
    let _ = state.db.note_start_cause(port, "manual");
    match state
        .supervisor
        .switch(port, spec, 30_000, expect_hash)
        .await
    {
        Ok(view) => Json(view).into_response(),
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({"error": {"type": "switch_failed", "message": e}})),
        )
            .into_response(),
    }
}

/// The endpoint's config FILE, verbatim, plus its SHA-256 - what the edit
/// page loads (the Advanced editor's content AND the concurrency token for
/// both save paths).
async fn servers_file_get(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(port): axum::extract::Path<u16>,
) -> Response {
    match state.supervisor.read_config_file(port) {
        Ok((content, hash)) => Json(serde_json::json!({
            "runtime_state": tokio::time::timeout(std::time::Duration::from_secs(2),
                paddock_admin::client::AdminClient::new(port).config_status()).await.ok().and_then(Result::ok),
            "path": state.supervisor.server_config_path(port).display().to_string(),
            "content": content,
            "hash": hash,
            // The same file PARSED, for a client that wants to change one
            // field and resend the rest rather than re-implement the TOML
            // reading. `switch` reads an absent owned key as "cleared", so a
            // caller that cannot fill in the fields it does not care about has
            // no way to leave them alone. The Studio ignores this key; the CLI
            // is built on it. Absent when the file does not parse - the
            // `content` above is still returned, which is what the raw editor
            // needs in order to FIX an unparseable file.
            "spec": state
                .supervisor
                .spec_from_config_text(&content)
                .ok()
                .and_then(|s| serde_json::to_value(s).ok()),
        }))
        .into_response(),
        Err(e) => relay_err(StatusCode::NOT_FOUND, e),
    }
}

#[derive(serde::Deserialize)]
struct ConfigFilePut {
    content: String,
    /// The hash the editor loaded; a mismatch on disk refuses the write.
    expect_hash: Option<String>,
    /// "defer" = write the file and leave the process alone. Absent = apply,
    /// which is what this route has always done.
    #[serde(default)]
    apply: Option<String>,
    /// Native confirmation binds to the process that was reviewed, not just
    /// the port. Internal-only: web callers retain their existing semantics.
    #[serde(skip)]
    expected_pid: Option<u32>,
}

/// Native callers supply a Rust-prepared, reviewed patch. Reuse the same
/// admission and apply semantics without serializing credentials through an
/// internal HTTP request (or exposing a raw-file API to Swift).
pub(crate) async fn save_endpoint_file(
    state: Arc<AppState>,
    port: u16,
    content: String,
    expected: String,
    deferred: bool,
    expected_pid: Option<u32>,
) -> Response {
    servers_file_put(
        State(state),
        axum::extract::Path(port),
        Json(ConfigFilePut {
            content,
            expect_hash: Some(expected),
            apply: deferred.then(|| "defer".into()),
            expected_pid,
        }),
    )
    .await
}

/// The Advanced editor's Save: write the file VERBATIM (hash-guarded) and
/// restart the endpoint from it. Never the spec renderer - every knob the
/// runner's config surface has is honored, known to the manager or not.
async fn servers_file_put(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(port): axum::extract::Path<u16>,
    Json(mut body): Json<ConfigFilePut>,
) -> Response {
    // "Save without restarting": the file lands, nothing starts or stops, and
    // no admission runs - saving a configuration is not loading it.
    if body.apply.as_deref() == Some("defer") {
        let _admission = state.admission.lock().await;
        return match state.supervisor.write_config_file_deferred(
            port,
            &body.content,
            body.expect_hash.as_deref(),
        ) {
            Ok(()) => Json(serde_json::json!({"port": port, "applied": "saved"})).into_response(),
            Err(e) => (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({"error": {"type": "config_write_failed", "message": e}})),
            )
                .into_response(),
        };
    }
    // the edited file restarts this port - price it (freeing the incumbent)
    // before writing anything. The file's own vram_budget, when present, is
    // the ask. Metal fills an absent ceiling with its automatic reservation;
    // every other key and comment is preserved.
    let _admission = state.admission.lock().await;
    if let Err(error) = state.supervisor.validate_backend_config(&body.content) {
        return relay_err(StatusCode::BAD_REQUEST, error);
    }
    if !crate::native_endpoints::on_demand_config(&body.content)
        && !state.supervisor.can_apply_live(port, &body.content).await
        && let Ok(doc) = toml::from_str::<toml::Value>(&body.content)
        && let Some(model) = doc.get("model").and_then(toml::Value::as_str)
    {
        let get_int = |k: &str| doc.get(k).and_then(toml::Value::as_integer);
        let media = match state.supervisor.project_config_text(&body.content) {
            Ok(media) => media,
            Err(error) => return relay_err(StatusCode::BAD_REQUEST, error),
        };
        let req = AdmitReq {
            on_demand: false,
            model,
            artifact: None,
            gpu_pin: doc.get("gpu").and_then(toml::Value::as_str),
            freeing_port: Some(port),
            fixed_need: media.vram_budget.map(|n| n << 20),
            max_batch: get_int("max_batch").map(|n| n as usize),
            max_ctx: get_int("max_ctx").map(|n| n as usize),
            fp8_kv: kv_ask(doc.get("kv_cache_dtype").and_then(toml::Value::as_str)),
            // same verbatim rule as `mmproj` below: the FILE's own block is
            // the answer on this path, not anything the manager remembers
            offload_ram_bytes: doc.get("kv_offload").and_then(|k| {
                let on = k
                    .get("enabled")
                    .and_then(toml::Value::as_bool)
                    .unwrap_or(false);
                let gb = k
                    .get("ram_gb")
                    .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|n| n as f64)))
                    .unwrap_or(0.0);
                (on && gb > 0.0).then_some((gb * (1u64 << 30) as f64) as u64)
            }),
            spec: spec_wanted(doc.get("spec").and_then(toml::Value::as_str)),
            // Same file/registry projection as native controls: optional
            // bundled towers have booleans but no mmproj paths to count.
            vision: media.vision,
            audio: Some(media.audio),
            // a saved file's window is the file's - priced as written
            fit_ctx: false,
        };
        match vram_admission(&state, req).await {
            Err(msg) => return admission_refused(msg),
            Ok(Admitted {
                grant: Some(grant), ..
            }) if state.readiness.backend == "metal" => {
                body.content = pin_budget_text(&body.content, grant);
            }
            _ => {}
        }
    }
    match state
        .supervisor
        .write_config_file(
            port,
            &body.content,
            body.expect_hash.as_deref(),
            30_000,
            body.expected_pid,
        )
        .await
    {
        // `applied` tells the Studio which way the save landed: "live" =
        // control-plane-only change, already serving; "restart" = takeover ran
        Ok((view, live)) => {
            let mut v = serde_json::to_value(&view).unwrap_or_default();
            if let Some(o) = v.as_object_mut() {
                o.insert(
                    "applied".into(),
                    serde_json::json!(if live { "live" } else { "restart" }),
                );
            }
            Json(v).into_response()
        }
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            Json(serde_json::json!({"error": {"type": "config_write_failed", "message": e}})),
        )
            .into_response(),
    }
}

/// Candidate files for the Advanced editor's path pickers: every GGUF under
/// the model dirs, split by role (weights / mmproj vision tower / mtp
/// drafter), plus safetensors snapshot DIRS for fp8_native. Suggestions
/// only - a hand-typed path anywhere on disk stays first-class.
async fn servers_files(State(state): State<Arc<AppState>>) -> Response {
    let mut gguf: Vec<String> = Vec::new();
    let mut mmproj: Vec<String> = Vec::new();
    let mut mtp: Vec<String> = Vec::new();
    let mut fp8_dirs: Vec<String> = Vec::new();
    // model dirs are shallow by convention (<dir>/<model-folder>/<files>);
    // walk two levels so both layouts land without a walkdir dependency
    fn scan(
        dir: &std::path::Path,
        depth: u8,
        gguf: &mut Vec<String>,
        mmproj: &mut Vec<String>,
        mtp: &mut Vec<String>,
        fp8_dirs: &mut Vec<String>,
    ) {
        let Ok(rd) = std::fs::read_dir(dir) else {
            return;
        };
        let mut has_safetensors = false;
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                if depth > 0 {
                    scan(&p, depth - 1, gguf, mmproj, mtp, fp8_dirs);
                }
                continue;
            }
            let name = e.file_name().to_string_lossy().to_lowercase();
            if name.ends_with(".gguf") {
                let out = if name.starts_with("mmproj") {
                    &mut *mmproj
                } else if name.starts_with("mtp") {
                    &mut *mtp
                } else {
                    &mut *gguf
                };
                out.push(p.display().to_string());
            } else if name.ends_with(".safetensors") {
                has_safetensors = true;
            }
        }
        if has_safetensors {
            fp8_dirs.push(dir.display().to_string());
        }
    }
    for d in state.supervisor.models_dirs() {
        scan(d, 2, &mut gguf, &mut mmproj, &mut mtp, &mut fp8_dirs);
    }
    for v in [&mut gguf, &mut mmproj, &mut mtp, &mut fp8_dirs] {
        v.sort();
    }
    // paths the manager itself is configured with - the known-good candidates
    // for the corresponding fields
    let model_dirs: Vec<String> = state
        .supervisor
        .models_dirs()
        .iter()
        .map(|d| d.display().to_string())
        .collect();
    let kernel_packs: Vec<String> = state
        .supervisor
        .kernel_pack()
        .map(|p| vec![p.display().to_string()])
        .unwrap_or_default();
    // No pdfium entry any more: it is linked into the runner, so there is no
    // path to suggest and nothing a user could point at.
    Json(serde_json::json!({
        "gguf": gguf,
        "mmproj": mmproj,
        "mtp": mtp,
        "fp8_dirs": fp8_dirs,
        "model_dirs": model_dirs,
        "kernel_packs": kernel_packs,
    }))
    .into_response()
}

/// Every configured endpoint on disk (servers/*.toml) - the filesystem is
/// the enumeration; a stopped endpoint is still configured. `paddock start
/// <model>` resolves names against this, and the Studio's fleet page shows
/// the not-running ones as stopped rows (display/vendor resolved here so the
/// row reads "Qwen 3.5 9B", not a weights path).
async fn servers_list(State(state): State<Arc<AppState>>) -> Response {
    let rows: Vec<serde_json::Value> = state
        .supervisor
        .configured()
        .await
        .into_iter()
        .map(|c| {
            let entry = c
                .model
                .as_deref()
                .and_then(|m| state.registry.catalog_of(m));
            // `model` is the endpoint's IDENTITY (a catalog id) and `weights`
            // the file it loads - both read off the config file by
            // `spec_from_config_file`, which is also what the start path reads.
            // They used to be derived separately here, and the page showed the
            // consequence: the row said "KB-Whisper Large" while the selector
            // beside it sat on "select" .
            serde_json::json!({
                "port": c.port,
                "model": c.model,
                "artifact": c.artifact,
                "weights": c.weights,
                "running": c.running,
                "display": entry.map(|m| m.display.clone()),
                "vendor": entry.and_then(|m| m.vendor.clone()),
                // What starting this would GET you. A running endpoint
                // advertises its own capability; a stopped one cannot be
                // asked, so the catalog answers instead - that is what lets
                // the composer's mic offer "start this whisper" rather than
                // sending you to pick a model you already configured
                // Absent for anything the catalog does not know.
                "capability": entry.map(|m| m.capability.clone()),
                // same stopped-row logic as capability: what starting this
                // would wire, from config + catalog
                "spec": c.spec_desc,
            })
        })
        .collect();
    Json(rows).into_response()
}

/// Remove a STOPPED endpoint's configuration (its servers/<port>.toml + any
/// election). A serving port refuses - stop first.
async fn servers_remove(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(port): axum::extract::Path<u16>,
) -> Response {
    match state.supervisor.remove_config(port).await {
        Ok(()) => Json(serde_json::json!({ "port": port, "removed": true })).into_response(),
        Err(e) => (
            StatusCode::CONFLICT,
            Json(serde_json::json!({"error": {"type": "invalid_request_error", "message": e}})),
        )
            .into_response(),
    }
}

#[derive(serde::Deserialize, Default)]
struct StartBody {
    /// Caller-approved eviction plan (the 507 offer's `plan`, confirmed):
    /// drain-stop these unpinned endpoints before pricing the start.
    #[serde(default)]
    evict: Vec<u16>,
}

/// Start an already-configured endpoint from its file, verbatim (`paddock
/// start <port>`; a stopped endpoint's file outlives its election).
pub(crate) async fn resume_elected(state: Arc<AppState>, port: u16) -> Response {
    start_saved(state, port, vec![], "boot-election").await
}

async fn servers_start(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(port): axum::extract::Path<u16>,
    body: Option<Json<StartBody>>,
) -> Response {
    let evict: Vec<u16> = body
        .map(|Json(b)| b.evict)
        .unwrap_or_default()
        .into_iter()
        .filter(|p| *p != port)
        .collect();
    start_saved(state, port, evict, "manual").await
}

async fn start_saved(state: Arc<AppState>, port: u16, evict: Vec<u16>, cause: &str) -> Response {
    let _admission = state.admission.lock().await;
    if state.supervisor.is_serving(port).await {
        return relay_err(
            StatusCode::CONFLICT,
            "This instance is already running.".into(),
        );
    }
    let on_demand = state
        .supervisor
        .read_config_file(port)
        .ok()
        .is_some_and(|(raw, _)| crate::native_endpoints::on_demand_config(&raw));
    if !on_demand && let Err(e) = perform_evictions(&state, &evict).await {
        return relay_err(StatusCode::CONFLICT, e);
    }
    // price the file before the verbatim start: its own vram_budget is the
    // ask when present; otherwise its envelope prices a plain check. The
    // Metal fills an absent ceiling before launching the saved configuration.
    if !on_demand
        && let Ok(spec) = state
            .supervisor
            .spec_from_config_file(&state.supervisor.server_config_path(port))
        && !spec.model.is_empty()
    {
        match vram_admission(&state, AdmitReq::for_spec(&spec, None)).await {
            Err(msg) => return admission_refused(msg),
            Ok(Admitted {
                grant: Some(grant), ..
            }) if state.readiness.backend == "metal" => {
                let Ok((text, hash)) = state.supervisor.read_config_file(port) else {
                    return relay_err(
                        StatusCode::BAD_REQUEST,
                        "Cannot read the saved configuration".into(),
                    );
                };
                if let Err(error) = state.supervisor.write_config_file_deferred(
                    port,
                    &pin_budget_text(&text, grant),
                    Some(&hash),
                ) {
                    return relay_err(StatusCode::BAD_REQUEST, error);
                }
            }
            _ => {}
        }
    }
    let _ = state.db.note_start_cause(port, cause);
    match state.supervisor.start_config(port).await {
        Ok(view) => Json(view).into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": {"type": "spawn_failed", "reason": e.reason(), "message": e.to_string()}})),
        ).into_response(),
    }
}

#[derive(serde::Deserialize)]
struct LogsQuery {
    tail: Option<usize>,
}

/// Tail of a manager-spawned runner's log file (buffered-history-first is the
/// stream shape; this is the simple snapshot form).
async fn runners_logs(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(port): axum::extract::Path<u16>,
    axum::extract::Query(q): axum::extract::Query<LogsQuery>,
) -> Response {
    let path = state.supervisor.log_path(port);
    match std::fs::read_to_string(&path) {
        Ok(s) => {
            let lines: Vec<&str> = s.lines().collect();
            let start = lines.len().saturating_sub(q.tail.unwrap_or(200));
            lines[start..].join("\n").into_response()
        }
        Err(_) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": {"type": "not_found_error", "message": format!("no log for port {port} ({})", path.display())}})),
        )
            .into_response(),
    }
}

/// Can this computer run models on its own, and what does the user do about
/// it. Probed once at startup - hardware does not change under a
/// running process, and a per-request NVML round trip to learn a constant is
/// a cost with no answer attached.
async fn readiness(State(state): State<Arc<AppState>>) -> Response {
    // The boot verdict is the current one: hardware does not change under a
    // running manager, and the one thing that used to (the CUDA fetch flipping
    // NeedsSetup to Ready) no longer exists.
    Json((*state.readiness).clone()).into_response()
}

/// Every graphics card this build knows about, one row each, so a person can
/// look up their own rather than work out which generation it belongs to.
/// Static for the life of the process - it is a fact about the build.
async fn gpu_sheet(State(state): State<Arc<AppState>>) -> Response {
    Json(serde_json::json!({
        "cards": crate::readiness::card_sheet(),
        // What this machine has, so its row can be marked "yours" instead of
        // making somebody match a name by eye.
        "yours": state.readiness.card,
    }))
    .into_response()
}

/// How the one-time CUDA fetch is going.
///
/// Answers even when nothing is running, because "already done" and "this
/// machine would never need them" are both real answers a UI has to render, and
/// neither of them is a job.
/// GET /api/updates - is there a newer paddock.
///
/// Cached for an hour (`updates::CHECK_INTERVAL`), so the Studio can poll this
/// as freely as it polls anything else without turning every render into an
/// outbound request. Never fails: "we could not reach the release server" comes
/// back as the `unknown` state, because a Manager that breaks when a laptop is
/// on a train is worse than one that admits it does not know.
async fn updates_status(State(state): State<Arc<AppState>>) -> Response {
    // one builder with the SSE push, same drift rule as runners_list
    Json(crate::push::update_info(&state).await).into_response()
}

/// POST /api/updates/download - fetch the newest package and verify it.
///
/// Downloads, hash-checks, and leaves the zip in `<data>/updates/`. It does not
/// install: on Windows a running exe cannot be replaced, and per the
/// "the maintainer starts services" rule the manager must never restart itself. Applying
/// is a deliberate, separate act.
///
/// Idempotent - a run already in flight is returned as-is rather than started
/// twice onto the same file.
async fn update_download_start(State(state): State<Arc<AppState>>) -> Response {
    if let Some(existing) = state.update_dl.lock().expect("update dl mutex").clone() {
        let phase = *existing.phase.lock().expect("dl phase");
        if matches!(phase, crate::updates::Phase::Running) {
            return Json(existing.status()).into_response();
        }
    }

    // Ask fresh rather than trusting the cache: the user clicked, and an hour-old
    // answer could name a version that has since been unpublished.
    let latest = match crate::updates::latest(crate::updates::http()).await {
        Ok(Some(l)) if l.download_available => l,
        Ok(Some(_)) => {
            return (
                StatusCode::CONFLICT,
                Json(serde_json::json!({"error": "that release has no downloadable package for this platform"})),
            )
                .into_response();
        }
        // A 404 here means nothing is published yet - a legitimate answer, not
        // a fault, so it is 409 (nothing to do) rather than 502 (we broke).
        Ok(None) => {
            return (
                StatusCode::CONFLICT,
                Json(serde_json::json!({"error": "no release has been published for this platform yet"})),
            )
                .into_response();
        }
        Err(why) => {
            return (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({"error": format!("could not reach the release server: {why}")})),
            )
                .into_response();
        }
    };

    let dl = Arc::new(crate::updates::Download {
        version: latest.version.clone(),
        phase: std::sync::Mutex::new(crate::updates::Phase::Idle),
        received: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        total: latest.file_size.unwrap_or(0).max(0) as u64,
        path: std::sync::Mutex::new(None),
        error: std::sync::Mutex::new(None),
        cancel: Arc::new(std::sync::atomic::AtomicBool::new(false)),
    });
    *state.update_dl.lock().expect("update dl mutex") = Some(dl.clone());

    let spawned = dl.clone();
    tokio::spawn(async move { crate::updates::download(&latest, spawned).await });
    Json(dl.status()).into_response()
}

/// POST /api/updates/download/cancel - stop it and remove the partial file.
async fn update_download_cancel(State(state): State<Arc<AppState>>) -> Response {
    if let Some(dl) = state.update_dl.lock().expect("update dl mutex").clone() {
        dl.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
        return Json(dl.status()).into_response();
    }
    Json(serde_json::json!({"phase": "idle"})).into_response()
}

/// Latest device telemetry snapshot (all GPUs). `{ available: false }` when
/// NVML isn't present. Cheap: returns the last background sample, never blocks
/// on the driver.
/// The prefix cache's off-GPU tiers, per running model (plan D8).
///
/// A live fan-out, not a stored series: the panel asks what the cache is
/// deciding right now, and the manager is an API client of each runner here,
/// never a proxy - it reads `/v1/stats` and republishes the section, exactly
/// as the reconciler does for memory. Runners without a tier armed are
/// omitted rather than reported as empty, so "no cache tier here"
/// reads as an absence instead of a row of zeros.
async fn cache_info(State(state): State<Arc<AppState>>) -> Response {
    use futures::stream::{FuturesUnordered, StreamExt};
    let views = state.supervisor.list().await;
    let mut jobs: FuturesUnordered<_> = views
        .into_iter()
        .filter(|v| v.status != "unreachable")
        .map(|v| async move {
            let stats = tokio::time::timeout(
                std::time::Duration::from_secs(2),
                paddock_admin::client::AdminClient::new(v.port).stats(),
            )
            .await
            .ok()
            .and_then(Result::ok)?;
            let tier = stats
                .get("engine")?
                .get("cache_tier")
                .filter(|t| !t.is_null())?
                .clone();
            Some(serde_json::json!({
                "port": v.port,
                "model": v.model,
                "tier": tier,
            }))
        })
        .collect();
    let mut servers = Vec::new();
    while let Some(row) = jobs.next().await {
        if let Some(r) = row {
            servers.push(r);
        }
    }
    servers.sort_by_key(|r| {
        r.get("port")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
    });
    Json(serde_json::json!({ "servers": servers })).into_response()
}

async fn gpu_info(State(state): State<Arc<AppState>>) -> Response {
    Json(gpu_payload(&state.gpu.latest(), &state.recon)).into_response()
}

/// Device snapshot + the §9 reconciliation section in one payload. Every fact
/// keeps its authoritative producer: `gpus` is NVML or Metal capacity, `reconciliation`
/// joins it with runner self-reports; nothing is re-labeled.
fn gpu_payload(
    snap: &crate::telemetry::GpuSnapshot,
    recon: &tokio::sync::watch::Receiver<Arc<Option<crate::telemetry::Reconciliation>>>,
) -> serde_json::Value {
    let mut v = serde_json::to_value(snap).unwrap_or_else(|_| serde_json::json!({}));
    let r = recon.borrow().clone();
    v["reconciliation"] = serde_json::to_value(&*r).unwrap_or(serde_json::Value::Null);
    v
}

/// Relay the Studio's realtime transcription socket to a runner's
/// `/v1/realtime`.
///
/// The same shape as every other `/api/runners/{port}/...` route and for the
/// same reason: the browser never opens a runner port itself, so runner API
/// keys stay server-side and there is no cross-origin question. This is the
/// manager acting as its own API client on behalf of its UI - which is what
/// every relay here is - not a general proxy for third-party callers.
///
/// Frames pass through UNREAD. This relay has no opinion about the realtime
/// protocol, so a new client or server event needs no change here.
async fn relay_realtime(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(port): axum::extract::Path<u16>,
    axum::extract::RawQuery(q): axum::extract::RawQuery,
    ws: WebSocketUpgrade,
) -> Response {
    let key = state.supervisor.runner_key(port).await;
    let query = q
        .filter(|s| !s.is_empty())
        .map(|s| format!("?{s}"))
        .unwrap_or_default();
    let url = format!("ws://127.0.0.1:{port}/v1/realtime{query}");
    ws.on_upgrade(move |socket| realtime_ws(socket, url, key))
}

async fn realtime_ws(mut client: WebSocket, url: String, key: Option<String>) {
    use futures::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::Message as Up;

    let mut req =
        match tokio_tungstenite::tungstenite::client::IntoClientRequest::into_client_request(url) {
            Ok(r) => r,
            Err(e) => {
                let _ = client
                    .send(Message::Text(relay_ws_err(&e.to_string()).into()))
                    .await;
                return;
            }
        };
    if let Some(k) = key
        && let Ok(v) = format!("Bearer {k}").parse()
    {
        req.headers_mut()
            .insert(axum::http::header::AUTHORIZATION, v);
    }
    let upstream = match tokio_tungstenite::connect_async(req).await {
        Ok((s, _)) => s,
        Err(e) => {
            // The socket is already open, so a failure to reach the runner can
            // only be reported as an event - and it says which runner, because
            // "connection closed" on its own is the least useful thing a live
            // microphone can tell someone.
            let msg = format!("the runner is not answering on its realtime endpoint: {e}");
            let _ = client.send(Message::Text(relay_ws_err(&msg).into())).await;
            return;
        }
    };
    let (mut up_tx, mut up_rx) = upstream.split();
    loop {
        tokio::select! {
            from_client = client.recv() => match from_client {
                Some(Ok(Message::Text(t))) => {
                    if up_tx.send(Up::Text(t.as_str().into())).await.is_err() {
                        return;
                    }
                }
                Some(Ok(Message::Binary(b))) => {
                    if up_tx.send(Up::Binary(b.to_vec().into())).await.is_err() {
                        return;
                    }
                }
                Some(Ok(Message::Ping(_) | Message::Pong(_))) => {}
                // a closed browser tab must not leave the runner holding a
                // session open forever
                None | Some(Ok(Message::Close(_))) | Some(Err(_)) => {
                    let _ = up_tx.send(Up::Close(None)).await;
                    return;
                }
            },
            from_runner = up_rx.next() => match from_runner {
                Some(Ok(Up::Text(t))) => {
                    if client.send(Message::Text(t.as_str().into())).await.is_err() {
                        return;
                    }
                }
                Some(Ok(Up::Binary(b))) => {
                    if client.send(Message::Binary(b.to_vec().into())).await.is_err() {
                        return;
                    }
                }
                Some(Ok(_)) => {}
                None | Some(Err(_)) => return,
            },
        }
    }
}

fn relay_ws_err(message: &str) -> String {
    serde_json::json!({
        "type": "error",
        "error": {"type": "connection_error", "message": message},
    })
    .to_string()
}

/// Live device telemetry over WebSocket: pushes the current snapshot on
/// connect, then every new sample. Subscribing ramps the sampler's cadence.
async fn gpu_stream(State(state): State<Arc<AppState>>, ws: WebSocketUpgrade) -> Response {
    let rx = state.gpu.subscribe();
    let recon = state.recon.clone();
    ws.on_upgrade(move |socket| gpu_ws(socket, rx, recon))
}

async fn gpu_ws(
    mut socket: WebSocket,
    mut rx: tokio::sync::watch::Receiver<std::sync::Arc<crate::telemetry::GpuSnapshot>>,
    recon: tokio::sync::watch::Receiver<Arc<Option<crate::telemetry::Reconciliation>>>,
) {
    loop {
        // Serialize the latest snapshot; scope the borrow so it never crosses an
        // await point (watch Ref must not be held across .await).
        let payload = {
            let snap = rx.borrow_and_update().clone();
            gpu_payload(&snap, &recon).to_string()
        };
        if socket.send(Message::Text(payload.into())).await.is_err() {
            return; // client gone
        }
        tokio::select! {
            changed = rx.changed() => {
                if changed.is_err() {
                    return; // sampler stopped (shutdown)
                }
            }
            msg = socket.recv() => {
                match msg {
                    Some(Ok(Message::Close(_))) => {
                        // Flush tungstenite's queued close reply before dropping
                        // the transport; normal panel closes aren't WS errors.
                        use futures::SinkExt;
                        let _ = socket.flush().await;
                        return;
                    }
                    None | Some(Err(_)) => return,
                    _ => {}
                }
            }
        }
    }
}

pub(crate) async fn not_found(uri: axum::http::Uri) -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(ErrorBody::not_found(format!("route {uri}"))),
    )
        .into_response()
}

/// Is this request from the box itself? The runner's test, kept in step with
/// `paddock_runner::routes::peer_is_local`: a loopback peer, no declared proxy
/// (`trusted_proxy`), and no forwarding header on the request - behind a
/// proxy on the same host every caller is 127.0.0.1, and a forged header can
/// only cost a caller its exemption, never grant one.
fn peer_is_local(req: &axum::extract::Request, trusted_proxy: bool) -> bool {
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

/// Bearer auth for `/api`. A no-op when no key is configured (loopback
/// default); otherwise the header key must equal the configured key or match a
/// stored (db-backed) API key. Non-API paths (the SPA) always pass so the UI
/// can load. Loopback PEERS are exempt (`peer_is_local`) - the runner's
/// policy, and what makes a network bind usable at all: without it, --host
/// 0.0.0.0 locked the box's own Studio out of every /api route and every page
/// rendered empty. The exemption ends where a reverse proxy begins.
async fn auth_mw(
    State(state): State<Arc<AppState>>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    if let Some(required) = &state.auth_key {
        let local = peer_is_local(&req, state.trusted_proxy);
        let path = req.uri().path();
        if !local && path.starts_with("/api/") {
            let verify = |k: &str| k == required || state.db.verify_api_key(k).unwrap_or(false);
            let bearer_ok = req
                .headers()
                .get(axum::http::header::AUTHORIZATION)
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.strip_prefix("Bearer "))
                .is_some_and(verify);
            // The Studio's key-gate session: a browser cannot attach Bearer
            // headers to every fetch/SSE the SPA makes, so /auth/login turns
            // the key into an HttpOnly cookie and this accepts it.
            let cookie_ok = req
                .headers()
                .get(axum::http::header::COOKIE)
                .and_then(|v| v.to_str().ok())
                .and_then(|c| {
                    c.split(';')
                        .map(str::trim)
                        .find_map(|p| p.strip_prefix("paddock_key="))
                })
                .is_some_and(verify);
            if !bearer_ok && !cookie_ok {
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

#[derive(serde::Deserialize)]
struct LoginBody {
    key: String,
}

/// `POST /auth/login` - the Studio's key gate for network browsers. Lives
/// outside `/api` so it is reachable before auth. A valid key comes back as
/// an HttpOnly cookie (30 days), which `auth_mw` accepts alongside the
/// Bearer header - the key never sits in script-readable storage.
async fn auth_login(State(state): State<Arc<AppState>>, Json(b): Json<LoginBody>) -> Response {
    let ok = state.auth_key.as_deref().is_some_and(|k| b.key == k)
        || state.db.verify_api_key(&b.key).unwrap_or(false);
    if !ok {
        return (
            StatusCode::UNAUTHORIZED,
            Json(ErrorBody::new(
                "invalid_api_key",
                "that key does not open this paddock",
            )),
        )
            .into_response();
    }
    // No `Secure` flag, still deliberately - for a different reason.
    // Network browsers now reach this over https, so `Secure` would
    // normally be right. But the identity can fail to establish (an unwritable
    // data root), and the fallback is cleartext: a `Secure` cookie would then
    // never be sent, so the key gate would accept a correct key and refuse
    // every request after it. Loopback callers are exempt from auth entirely
    // and never need this cookie, so the flag would buy nothing it does not
    // also risk.
    let cookie = format!(
        "paddock_key={}; HttpOnly; SameSite=Lax; Path=/; Max-Age=2592000",
        b.key
    );
    set_cookie_response(&cookie)
}

/// `POST /auth/logout` - drops the session cookie.
async fn auth_logout() -> Response {
    set_cookie_response("paddock_key=; HttpOnly; SameSite=Lax; Path=/; Max-Age=0")
}

fn set_cookie_response(cookie: &str) -> Response {
    let mut res = StatusCode::NO_CONTENT.into_response();
    if let Ok(v) = axum::http::HeaderValue::from_str(cookie) {
        res.headers_mut().insert(axum::http::header::SET_COOKIE, v);
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    #[test]
    fn automatic_budget_preserves_configuration_and_explicit_ceiling() {
        let original = "# operator settings\nmodel = 'local'\napi_key = 'fixture-only'\n[custom]\nkeep = true\n";
        let pinned = pin_budget_text(original, 12345);
        let before: toml::Value = toml::from_str(original).unwrap();
        let mut after: toml::Value = toml::from_str(&pinned).unwrap();
        assert_eq!(
            after
                .as_table_mut()
                .unwrap()
                .remove("vram_budget")
                .unwrap()
                .as_integer(),
            Some(12345)
        );
        assert_eq!(before, after);
        assert!(pinned.starts_with("# operator settings"));
        assert!(crate::automatic_budget::is_automatic(&pinned));
        let explicit = "vram_budget = 12345\n";
        assert_eq!(pin_budget_text(explicit, 99999), explicit);
    }

    /// The fit-aware default (decided 2026-10-04): a defaulted window is the
    /// catalog target when the card backs it at the default slot count, the
    /// largest Studio rung under it when not, and never more than the card.
    /// Priced by the real estimator on qwen3.8's published Q8_0 shape.
    #[test]
    fn a_default_window_fits_onto_the_studio_ladder() {
        use paddock_estimator::{Device, Envelope, KvDtype, estimate};
        let registry = crate::registry::Registry::new(std::env::temp_dir());
        let a = registry
            .catalog_of("qwen3.8-27b")
            .and_then(|m| m.artifact("q8"))
            .unwrap();
        let shape = a.shape.clone().unwrap().into_model_shape(0, 0);
        let env = Envelope {
            concurrency: 4,
            kv_dtype: KvDtype::Fp8E4m3,
            spec: None,
            offload: None,
        };
        let fit = |gib: u64| {
            let dev = Device {
                free_bytes: gib << 30,
                total_bytes: gib << 30,
            };
            fit_window(262144, |ctx| {
                let mut s = shape.clone();
                s.max_ctx = ctx;
                estimate(&s, &env, &dev)
            })
        };
        // a Spark-class budget backs the whole window at 4 slots
        assert_eq!(fit(100).0, 262144);
        // a 48 GB card does not; it lands on a rung, and that rung fits
        let (ctx, est) = fit(44);
        assert!(CTX_STEPS.contains(&ctx) && ctx < 262144, "{ctx}");
        assert!(est.max_ctx >= ctx, "{} < {ctx}", est.max_ctx);
        // more card never buys less window
        assert!(fit(36).0 <= ctx && ctx <= fit(100).0);
    }

    /// An image lane gets no envelope pinned: the first image endpoint's
    /// file carried `max_ctx = 4096` / `max_batch = 32` for a runner that
    /// reads neither, and the Studio then drew a context meter from them.
    #[test]
    fn an_image_lane_gets_no_token_envelope_pinned() {
        let registry = crate::registry::Registry::new(std::env::temp_dir());
        let mut lanes = 0;
        for model in &registry.catalog().models {
            let mut spec = crate::supervisor::SpawnSpec {
                model: model.id.clone(),
                ..Default::default()
            };
            pin_envelope(&mut spec, &registry);
            let image = model.capability.iter().any(|c| c == "image-generation");
            assert_eq!(
                crate::estimate::is_image_lane(&registry, &model.id),
                image,
                "{}",
                model.id
            );
            if image {
                lanes += 1;
                assert!(
                    spec.max_ctx.is_none() && spec.max_batch.is_none(),
                    "{}: an image lane carries no max_ctx / max_batch",
                    model.id
                );
            } else {
                assert!(
                    spec.max_ctx.is_some() && spec.max_batch.is_some(),
                    "{}: every other model is pinned to its envelope",
                    model.id
                );
            }
        }
        assert!(lanes > 0, "the catalog carries an image lane to check");
    }

    #[test]
    fn default_start_envelope_matches_each_metal_artifact_before_admission() {
        let registry = crate::registry::Registry::new(std::env::temp_dir()).with_backend("metal");
        for model in &registry.catalog().models {
            for artifact in model
                .weights()
                .filter(|a| a.runtime.supports_backend("metal"))
            {
                let mut spec = crate::supervisor::SpawnSpec {
                    model: model.id.clone(),
                    artifact: Some(artifact.id.clone()),
                    ..Default::default()
                };
                pin_envelope(&mut spec, &registry);
                if let Some(memory) = &artifact.runtime.memory {
                    assert!(
                        spec.max_ctx.unwrap() as u64 <= memory.max_ctx,
                        "{} {} context",
                        model.id,
                        artifact.id
                    );
                    assert!(
                        spec.max_batch.unwrap() as u64 <= memory.max_batch,
                        "{} {} batch",
                        model.id,
                        artifact.id
                    );
                }
                let priced = AdmitReq::for_spec(&spec, None);
                assert_eq!(priced.max_ctx, spec.max_ctx);
                assert_eq!(priced.max_batch, spec.max_batch);
            }
        }
        for model in [
            "kb-whisper-large",
            "nb-whisper-large",
            "roest-v3-whisper-1.5b",
        ] {
            let mut spec = crate::supervisor::SpawnSpec {
                model: model.into(),
                ..Default::default()
            };
            pin_envelope(&mut spec, &registry);
            assert_eq!((spec.max_ctx, spec.max_batch), (Some(448), Some(16)));
            // Never silently shrink a deliberate (even invalid) user choice.
            spec.max_ctx = Some(4096);
            spec.max_batch = Some(32);
            pin_envelope(&mut spec, &registry);
            assert_eq!((spec.max_ctx, spec.max_batch), (Some(4096), Some(32)));
        }
    }

    async fn body_json(res: Response) -> serde_json::Value {
        let bytes = res.into_body().collect().await.expect("body").to_bytes();
        serde_json::from_slice(&bytes).expect("json body")
    }

    /// The loopback exemption is for the box's own browser and tools, and only
    /// them: a forwarding header or a declared proxy means the peer address is
    /// the proxy's, and the key is required again.
    #[tokio::test]
    async fn loopback_is_exempt_only_when_nothing_says_proxy() {
        use axum::extract::ConnectInfo;
        let local: std::net::SocketAddr = "127.0.0.1:4242".parse().expect("addr");
        let keyed = |trusted_proxy: bool| {
            let mut s = AppState::for_tests();
            s.auth_key = Some("pk-test".into());
            s.trusted_proxy = trusted_proxy;
            Arc::new(s)
        };
        // a real route, so the answer tells auth apart from a 404
        let get = |headers: &[(&str, &str)]| {
            let mut req = axum::http::Request::get("/api/definitely-not-a-route");
            for (k, v) in headers {
                req = req.header(*k, *v);
            }
            let mut req = req.body(axum::body::Body::empty()).expect("request");
            req.extensions_mut().insert(ConnectInfo(local));
            req
        };
        // the box's own browser: through to the router (which 404s this path)
        let plain = router(keyed(false))
            .oneshot(get(&[]))
            .await
            .expect("response");
        assert_eq!(plain.status(), StatusCode::NOT_FOUND);
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
        assert_eq!(with_key.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn api_misses_get_openai_shaped_404() {
        let res = router(Arc::new(AppState::for_tests()))
            .oneshot(
                axum::http::Request::get("/api/definitely-not-a-route")
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
    async fn server_info_reports_manager_role() {
        let res = router(Arc::new(AppState::for_tests()))
            .oneshot(
                axum::http::Request::get("/api/server")
                    .body(axum::body::Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(res.status(), StatusCode::OK);
        let json = body_json(res).await;
        assert_eq!(json["role"], "manager");
    }
}
