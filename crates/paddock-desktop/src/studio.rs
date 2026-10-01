//! App-lifetime, loopback-only API relay. Native callers need no HTML assets
//! or browser. Viewer resources can be mounted separately when needed; the
//! existing workspace still mounts them during the native runtime migration.
use axum::{
    body::{Body, Bytes},
    extract::{Request, State},
    http::{HeaderValue, Method, StatusCode, header},
    middleware::Next,
    response::{IntoResponse, Response},
};
use std::{
    path::PathBuf,
    sync::{Arc, RwLock},
};
use tokio::io::AsyncReadExt;

const MAX_BODY: usize = 192 * 1024 * 1024;
pub const COOKIE: &str = "paddock_desktop_session";

pub struct Host {
    pub origin: String,
    pub session: String,
    task: tokio::task::JoinHandle<()>,
    assets: Arc<RwLock<Option<PathBuf>>>,
}
impl Drop for Host {
    fn drop(&mut self) {
        self.task.abort();
    }
}
#[derive(Clone)]
struct Policy {
    authority: String,
    origin: String,
    session: String,
    mcp_token: String,
    assets: Arc<RwLock<Option<PathBuf>>>,
}

impl Host {
    pub async fn start(
        state: Arc<paddock_manager::routes::AppState>,
        assets: Option<PathBuf>,
    ) -> Result<Self, String> {
        let assets = Arc::new(RwLock::new(assets.map(validate_assets).transpose()?));
        let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
            .await
            .map_err(|e| e.to_string())?;
        let authority = listener
            .local_addr()
            .map_err(|e| e.to_string())?
            .to_string();
        let origin = format!("http://{authority}");
        let session = format!(
            "{}{}",
            uuid::Uuid::new_v4().simple(),
            uuid::Uuid::new_v4().simple()
        );
        let policy = Policy {
            authority,
            origin: origin.clone(),
            session: session.clone(),
            assets: assets.clone(),
            mcp_token: format!(
                "{}{}",
                uuid::Uuid::new_v4().simple(),
                uuid::Uuid::new_v4().simple()
            ),
        };
        let app = paddock_manager::routes::router(state.clone())
            .layer(axum::middleware::from_fn_with_state(
                Arc::new(policy),
                guard,
            ))
            .layer(axum::Extension(state));
        let task = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        Ok(Self {
            origin,
            session,
            task,
            assets,
        })
    }
    pub fn mount_assets(&self, assets: PathBuf) -> Result<(), String> {
        let root = validate_assets(assets)?;
        let mut mounted = self
            .assets
            .write()
            .map_err(|_| "Viewer assets unavailable")?;
        // An already-running resource origin must never change underneath a
        // request. A different bundle requires a fresh app/core lifetime.
        if mounted.as_ref().is_some_and(|old| *old != root) {
            return Err("Viewer assets are already mounted from another bundle".into());
        }
        *mounted = Some(root);
        Ok(())
    }
    pub fn descriptor(&self) -> serde_json::Value {
        serde_json::json!({"origin": self.origin, "cookieName": COOKIE, "session": self.session})
    }
}

fn validate_assets(assets: PathBuf) -> Result<PathBuf, String> {
    let root = assets
        .canonicalize()
        .map_err(|_| "The bundled viewer assets are missing")?;
    if !root.join("index.html").is_file() {
        return Err("The bundled viewer assets are missing".into());
    }
    Ok(root)
}

fn value<'a>(request: &'a Request, name: &str) -> Option<&'a str> {
    request.headers().get(name)?.to_str().ok()
}

fn authorized(policy: &Policy, req: &Request) -> bool {
    if value(req, "host") != Some(policy.authority.as_str()) {
        return false;
    }
    if value(req, "origin").is_some_and(|o| o != policy.origin) {
        return false;
    }
    if value(req, "sec-fetch-site").is_some_and(|v| !matches!(v, "same-origin" | "none")) {
        return false;
    }
    if matches!(req.uri().path(), "/api/mcp/artifacts" | "/api/mcp/graph") {
        // Runner callbacks get only this capability, injected in Rust. It does
        // not unlock conversations, files, cloud settings or model lifecycle.
        return value(req, "authorization").and_then(|v| v.strip_prefix("Bearer "))
            == Some(policy.mcp_token.as_str());
    }
    value(req, "cookie").is_some_and(|cookies| {
        cookies.split(';').any(|cookie| {
            cookie.trim().strip_prefix(&format!("{COOKIE}=")) == Some(policy.session.as_str())
        })
    })
}

fn api_allowed(method: &Method, path: &str) -> bool {
    let parts: Vec<_> = path.trim_matches('/').split('/').collect();
    match parts.as_slice() {
        [
            "api",
            "server" | "readiness" | "runners" | "servers" | "gpu" | "cache" | "events"
            | "elections",
        ] => *method == Method::GET,
        ["api", "gpu", "stream"] => *method == Method::GET,
        ["api", "models", "catalog" | "estimate" | "pulls"] => *method == Method::GET,
        ["api", "usage", "history"] => *method == Method::GET,
        ["api", "reads"] => matches!(*method, Method::GET | Method::POST),
        ["api", "read-history"] => *method == Method::GET,
        ["api", "table-history"] => *method == Method::GET,
        ["api", "table-history", _] => {
            matches!(*method, Method::GET | Method::PUT | Method::DELETE)
        }
        ["api", "read-history", _] => matches!(*method, Method::GET | Method::PUT | Method::DELETE),
        ["api", "read-runs", _] => matches!(*method, Method::GET | Method::POST | Method::DELETE),
        ["api", "reads", _] => matches!(*method, Method::GET | Method::PUT | Method::DELETE),
        ["api", "runners", port, "v1", "systemone"] if port.parse::<u16>().is_ok() => {
            *method == Method::POST
        }
        [
            "api",
            "runners",
            port,
            "v1",
            "tabular",
            "predictions" | "contexts",
        ] if port.parse::<u16>().is_ok_and(|p| p != 0) => *method == Method::POST,
        ["api", "runners", port, "v1", "tabular", "contexts", id]
            if port.parse::<u16>().is_ok_and(|p| p != 0)
                && !id.is_empty()
                && id.len() <= 64
                && id
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-') =>
        {
            *method == Method::DELETE
        }
        ["api", "mcp", "artifacts" | "graph" | "tools"] => true,
        ["api", "connectors"] => *method == Method::GET,
        ["api", "cloud"] | ["api", "cloud", _, "models"] => *method == Method::GET,
        ["api", "cloud", "mcp-approvals", _] => *method == Method::POST,
        ["api", "cloud", _, "v1", "responses"]
        | ["api", "cloud", _, "v1", "audio", "transcriptions"] => *method == Method::POST,
        ["api", "runners", port, tail @ ..] if port.parse::<u16>().is_ok() => matches!(
            tail,
            ["server"]
                | ["v1", "responses"]
                | ["v1", "embeddings"]
                | ["v1", "rerank"]
                | ["v1", "realtime"]
                | ["v1", "audio", "transcriptions" | "alignments"]
                | ["v1", "images", "generations" | "edits"]
                | ["v1", "messages", "count_tokens"]
                | ["extract"]
                | ["mcp-approvals", _]
        ),
        [
            "api",
            "conversations" | "prompts" | "settings" | "attachments" | "artifacts" | "graph"
            | "forensics",
            ..,
        ] => true,
        _ => false,
    }
}

fn preview_request_allowed(policy: &Policy, req: &Request) -> bool {
    matches!(*req.method(), Method::GET | Method::HEAD)
        && value(req, "host") == Some(policy.authority.as_str())
        && value(req, "origin").is_none_or(|o| o == policy.origin)
        && value(req, "sec-fetch-site").is_none_or(|v| matches!(v, "same-origin" | "none"))
        && match req.uri().path() {
            "/native-artifact-host" => req.uri().query().is_none(),
            "/artifact-frame" => matches!(req.uri().query(), None | Some("img=1")),
            _ => false,
        }
}

fn preview_host(head: bool) -> Response {
    (
        [
            (header::CONTENT_TYPE, "text/html; charset=utf-8"),
            (header::CACHE_CONTROL, "no-store"),
            (header::REFERRER_POLICY, "no-referrer"),
            (header::X_CONTENT_TYPE_OPTIONS, "nosniff"),
            (header::CONTENT_SECURITY_POLICY, "default-src 'none'; script-src 'none'; style-src 'unsafe-inline'; frame-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"),
            (header::HeaderName::from_static("permissions-policy"), "camera=(), microphone=(), geolocation=(), display-capture=(), clipboard-read=(), clipboard-write=(), payment=(), usb=()"),
        ],
        if head { "" } else { include_str!("native-artifact-host.html") },
    ).into_response()
}

async fn guard(
    State(policy): State<Arc<Policy>>,
    axum::Extension(core): axum::Extension<Arc<paddock_manager::routes::AppState>>,
    mut req: Request,
    next: Next,
) -> Response {
    // Credential-free, content-free shells only. Generated content is delivered
    // by the native host into the opaque-origin iframe, never through this API.
    // The dedicated WKWebView has no app cookie or privileged script bridge.
    if matches!(
        req.uri().path(),
        "/native-artifact-host" | "/artifact-frame"
    ) {
        if !preview_request_allowed(&policy, &req) {
            return StatusCode::FORBIDDEN.into_response();
        }
        if req.uri().path() == "/native-artifact-host" {
            return preview_host(req.method() == Method::HEAD);
        }
        return next.run(req).await; // The exact web Studio frame and header CSP.
    }
    // The external browser has no app cookie. This one callback authenticates
    // using the existing single-use OAuth state + PKCE flow; nothing else is
    // reachable without the Studio session. Strict CSP covers provider errors.
    if req.method() == Method::GET
        && req.uri().path() == "/api/connectors/oauth/callback"
        && value(&req, "host") == Some(policy.authority.as_str())
    {
        let mut response = next.run(req).await;
        response.headers_mut().insert(header::CONTENT_SECURITY_POLICY, HeaderValue::from_static("default-src 'none'; style-src 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"));
        response
            .headers_mut()
            .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
        return response;
    }
    if !authorized(&policy, &req) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let path = req.uri().path().to_string();
    if !path.starts_with("/api/") {
        return if matches!(*req.method(), Method::GET | Method::HEAD) {
            asset(&policy, &path, req.method() == Method::HEAD).await
        } else {
            StatusCode::METHOD_NOT_ALLOWED.into_response()
        };
    }
    if !api_allowed(req.method(), &path) {
        return (StatusCode::FORBIDDEN, axum::Json(serde_json::json!({"error":{"message":"This operation is not available in this app session."}}))).into_response();
    }
    if path == "/api/events" {
        return events(core).await;
    }
    let mut private_tokens = vec![policy.mcp_token.as_bytes().to_vec()];
    // Built-in tool callbacks must authenticate without giving JavaScript a
    // bearer token. Only exact same-host first-party URLs receive this token;
    // user-configured remote connectors are never given it.
    if req.method() == Method::POST && path.ends_with("/v1/responses") {
        let (mut parts, body) = req.into_parts();
        let bytes = match axum::body::to_bytes(body, MAX_BODY).await {
            Ok(bytes) => bytes,
            Err(_) => return StatusCode::PAYLOAD_TOO_LARGE.into_response(),
        };
        let body = match inject_tools(&bytes, &policy) {
            Ok(body) => body,
            Err(_) => return StatusCode::BAD_REQUEST.into_response(),
        };
        let (body, connector_tokens) = match inject_connectors(&body, &core).await {
            Ok(result) => result,
            Err(()) => return StatusCode::BAD_GATEWAY.into_response(),
        };
        private_tokens.extend(connector_tokens);
        parts.headers.remove(header::CONTENT_LENGTH);
        req = Request::from_parts(parts, Body::from(body));
    }
    let mut response = next.run(req).await;
    if path == "/api/connectors" && response.status().is_success() {
        let (mut parts, body) = response.into_parts();
        let bytes = match axum::body::to_bytes(body, 8 * 1024 * 1024).await {
            Ok(bytes) => bytes,
            Err(_) => return StatusCode::BAD_GATEWAY.into_response(),
        };
        let mut data: serde_json::Value = match serde_json::from_slice(&bytes) {
            Ok(data) => data,
            Err(_) => return StatusCode::BAD_GATEWAY.into_response(),
        };
        if let Some(rows) = data.as_array_mut() {
            for row in rows {
                row["headers"] = serde_json::json!({});
            }
        }
        parts.headers.remove(header::CONTENT_LENGTH);
        response = Response::from_parts(parts, Body::from(data.to_string()));
    }
    // The browser fleet uses the saved inventory, but must never see raw TOML
    // or runner/search keys (the ordinary web manager has a broader role).
    if matches!(path.as_str(), "/api/servers" | "/api/runners") && response.status().is_success() {
        let (parts, body) = response.into_parts();
        let bytes = match axum::body::to_bytes(body, 8 * 1024 * 1024).await {
            Ok(bytes) => bytes,
            Err(_) => return StatusCode::BAD_GATEWAY.into_response(),
        };
        let mut data: serde_json::Value = match serde_json::from_slice(&bytes) {
            Ok(data) => data,
            Err(_) => return StatusCode::BAD_GATEWAY.into_response(),
        };
        if super::project_for_ui(
            if path == "/api/servers" {
                "servers"
            } else {
                "runners"
            },
            &mut data,
        )
        .is_err()
        {
            return StatusCode::BAD_GATEWAY.into_response();
        }
        response = Response::from_parts(parts, Body::from(data.to_string()));
        response.headers_mut().remove(header::CONTENT_LENGTH);
    }
    if path.ends_with("/v1/responses") {
        // Terminal metadata may echo input tool descriptions. Strip the
        // callback capability across chunk boundaries without delaying normal
        // tokens: retain only a suffix matching the beginning of the secret.
        use futures_util::StreamExt;
        let (mut parts, body) = response.into_parts();
        parts.headers.remove(header::CONTENT_LENGTH);
        let tokens = private_tokens;
        let stream = futures_util::stream::try_unfold(
            (body.into_data_stream(), Vec::new(), tokens),
            |(mut stream, mut pending, tokens)| async move {
                loop {
                    match stream.next().await {
                        Some(Ok(bytes)) => {
                            pending.extend_from_slice(&bytes);
                            let output = redact_tokens(&mut pending, &tokens, false);
                            if !output.is_empty() {
                                return Ok::<_, axum::Error>(Some((
                                    Bytes::from(output),
                                    (stream, pending, tokens),
                                )));
                            }
                        }
                        Some(Err(error)) => return Err(error),
                        None => {
                            let output = redact_tokens(&mut pending, &tokens, true);
                            return Ok((!output.is_empty())
                                .then_some((Bytes::from(output), (stream, pending, tokens))));
                        }
                    }
                }
            },
        );
        response = Response::from_parts(parts, Body::from_stream(stream));
    }
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    response.headers_mut().insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    response
}

#[cfg(test)]
fn redact(pending: &mut Vec<u8>, token: &[u8], final_chunk: bool) -> Vec<u8> {
    redact_tokens(pending, &[token.to_vec()], final_chunk)
}

fn redact_tokens(pending: &mut Vec<u8>, tokens: &[Vec<u8>], final_chunk: bool) -> Vec<u8> {
    let mut output = Vec::new();
    let mut cursor = 0;
    while cursor < pending.len() {
        let remaining = &pending[cursor..];
        if let Some(token) = tokens
            .iter()
            .find(|token| !token.is_empty() && remaining.starts_with(token))
        {
            output.extend_from_slice(b"[private]");
            cursor += token.len();
        } else if !final_chunk && tokens.iter().any(|token| token.starts_with(remaining)) {
            break;
        } else {
            output.push(pending[cursor]);
            cursor += 1;
        }
    }
    pending.drain(..cursor);
    output
}

/// The shared web orchestrator names selected connectors, but their headers
/// stay behind this host. Resolve only an exact stored label + URL pair; a
/// model-produced URL can never trick us into forwarding another connector's
/// credential. OAuth refresh uses the existing manager implementation.
async fn inject_connectors(
    bytes: &[u8],
    core: &Arc<paddock_manager::routes::AppState>,
) -> Result<(Vec<u8>, Vec<Vec<u8>>), ()> {
    let mut data: serde_json::Value = serde_json::from_slice(bytes).map_err(|_| ())?;
    let mut secrets = Vec::new();
    if let Some(tools) = data.get_mut("tools").and_then(|v| v.as_array_mut()) {
        let rows = core.db.list_connectors().map_err(|_| ())?;
        for tool in tools {
            if tool["type"] != "mcp" {
                continue;
            }
            let Some(row) = rows.iter().find(|row| {
                row["label"] == tool["server_label"] && row["url"] == tool["server_url"]
            }) else {
                continue;
            };
            let row = paddock_manager::oauth::ensure_fresh(core, row.clone()).await;
            let mut headers = row["headers"].as_object().cloned().unwrap_or_default();
            if let Some(token) = row["oauth"]["access_token"].as_str() {
                headers
                    .entry("Authorization")
                    .or_insert_with(|| serde_json::json!(format!("Bearer {token}")));
            }
            for value in headers.values().filter_map(|v| v.as_str()) {
                if !value.is_empty() {
                    secrets.push(value.as_bytes().to_vec());
                }
                if let Some(token) = value.strip_prefix("Bearer ")
                    && !token.is_empty()
                {
                    secrets.push(token.as_bytes().to_vec());
                }
            }
            tool["headers"] = serde_json::Value::Object(headers);
        }
    }
    Ok((serde_json::to_vec(&data).map_err(|_| ())?, secrets))
}

async fn events(core: Arc<paddock_manager::routes::AppState>) -> Response {
    use axum::response::sse::{Event, KeepAlive, Sse};
    let rx = core.push.subscribe();
    let initial = serde_json::Value::Array(paddock_manager::push::fleet_rows(&core).await);
    let stream =
        futures_util::stream::unfold((rx, Some(initial)), |(mut rx, mut initial)| async move {
            loop {
                let mut data = match initial.take() {
                    Some(data) => data,
                    None => match rx.recv().await {
                        Ok(event) if event.kind == "fleet" => (*event.data).clone(),
                        Ok(_) | Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                            continue;
                        }
                        Err(_) => return None,
                    },
                };
                if super::project_for_ui("runners", &mut data).is_err() {
                    continue;
                }
                let event = Event::default().event("fleet").data(data.to_string());
                return Some((Ok::<_, std::convert::Infallible>(event), (rx, initial)));
            }
        });
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

fn inject_tools(bytes: &[u8], policy: &Policy) -> Result<Vec<u8>, serde_json::Error> {
    let mut data: serde_json::Value = serde_json::from_slice(bytes)?;
    if let Some(tools) = data.get_mut("tools").and_then(|v| v.as_array_mut()) {
        for tool in tools {
            if tool["type"] != "mcp" {
                continue;
            }
            if ["artifacts", "graph"]
                .iter()
                .any(|name| tool["server_url"] == format!("{}/api/mcp/{name}", policy.origin))
            {
                if !tool["headers"].is_object() {
                    tool["headers"] = serde_json::json!({});
                }
                tool["headers"]["Authorization"] =
                    serde_json::json!(format!("Bearer {}", policy.mcp_token));
            }
        }
    }
    serde_json::to_vec(&data)
}

async fn asset(policy: &Policy, path: &str, head: bool) -> Response {
    let root = policy.assets.read().ok().and_then(|assets| assets.clone());
    let Some(root) = root else {
        return StatusCode::NOT_FOUND.into_response();
    };
    // No user-chosen filesystem path, symlink escape, directory listing, or SPA
    // fallback for a missing JS/WASM file. Everything comes from this bundle.
    if path.contains('%') || path.contains('\\') || path.split('/').any(|p| p == ".." || p == ".") {
        return StatusCode::NOT_FOUND.into_response();
    }
    let name = if path == "/" || path == "/studio" || path.starts_with("/studio/") {
        "index.html"
    } else {
        path.trim_start_matches('/')
    };
    let file = match root.join(name).canonicalize() {
        Ok(file) if file.starts_with(&root) && file.is_file() => file,
        _ => return StatusCode::NOT_FOUND.into_response(),
    };
    let mime = match file.extension().and_then(|s| s.to_str()).unwrap_or("") {
        "html" => "text/html; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "json" => "application/json",
        "wasm" => "application/wasm",
        "woff2" => "font/woff2",
        "woff" => "font/woff",
        "ttf" => "font/ttf",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "ico" => "image/x-icon",
        _ => "application/octet-stream",
    };
    let file = match tokio::fs::File::open(file).await {
        Ok(f) => f,
        Err(_) => return StatusCode::NOT_FOUND.into_response(),
    };
    let body = if head {
        Body::empty()
    } else {
        Body::from_stream(futures_util::stream::try_unfold(
            file,
            |mut file| async move {
                let mut buffer = vec![0; 64 * 1024];
                let n = file.read(&mut buffer).await?;
                buffer.truncate(n);
                Ok::<_, std::io::Error>((n != 0).then_some((Bytes::from(buffer), file)))
            },
        ))
    };
    let mut response = body.into_response();
    let headers = response.headers_mut();
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static(mime));
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    headers.insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    headers.insert(
        header::REFERRER_POLICY,
        HeaderValue::from_static("no-referrer"),
    );
    headers.insert(header::CONTENT_SECURITY_POLICY, format!(
        "default-src 'none'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self' 'unsafe-inline'; font-src 'self' data:; img-src 'self' data: blob:; media-src 'self' blob: data:; connect-src 'self' ws://{}; worker-src 'self' blob:; frame-src 'self' blob:; frame-ancestors 'none'; object-src 'none'; base-uri 'none'; form-action 'none'", policy.authority).parse().expect("loopback address produces a valid CSP header"));
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use tower::ServiceExt;

    #[test]
    fn native_reads_allow_only_the_reading_contract() {
        for (method, path) in [
            (Method::GET, "/api/reads"),
            (Method::POST, "/api/reads"),
            (Method::GET, "/api/read-history"),
            (Method::GET, "/api/read-history/a"),
            (Method::PUT, "/api/read-history/a"),
            (Method::DELETE, "/api/read-history/a"),
            (Method::GET, "/api/reads/set-1"),
            (Method::PUT, "/api/reads/set-1"),
            (Method::DELETE, "/api/reads/set-1"),
            (Method::POST, "/api/runners/12587/v1/systemone"),
            (Method::POST, "/api/runners/12587/v1/tabular/predictions"),
            (Method::POST, "/api/runners/12587/v1/tabular/contexts"),
            (
                Method::DELETE,
                "/api/runners/12587/v1/tabular/contexts/ctx_1",
            ),
        ] {
            assert!(api_allowed(&method, path));
            assert!(!authorized(
                &policy(),
                &Request::builder()
                    .method(method.clone())
                    .uri(path)
                    .header("host", "127.0.0.1:1234")
                    .body(Body::empty())
                    .unwrap()
            ));
            assert!(authorized(
                &policy(),
                &request(path).method(method).body(Body::empty()).unwrap()
            ));
        }
        for (method, path) in [
            (Method::DELETE, "/api/reads"),
            (Method::POST, "/api/read-history"),
            (Method::DELETE, "/api/read-history"),
            (Method::GET, "/api/read-history/a/secret"),
            (Method::POST, "/api/reads/set-1"),
            (Method::GET, "/api/reads/set-1/secret"),
            (Method::GET, "/api/runners/12587/v1/systemone"),
            (Method::POST, "/api/runners/invalid/v1/systemone"),
            (Method::POST, "/api/runners/12587/stop"),
            (Method::GET, "/api/runners/12587/v1/tabular/predictions"),
            (Method::POST, "/api/runners/0/v1/tabular/predictions"),
            (Method::POST, "/api/runners/12587/v1/tabular/contexts/ctx-1"),
            (
                Method::DELETE,
                "/api/runners/12587/v1/tabular/contexts/../secret",
            ),
        ] {
            assert!(!api_allowed(&method, path));
        }
    }

    #[tokio::test]
    async fn sqlite_preferences_import_and_patch_cross_the_native_guard() {
        let state = Arc::new(paddock_manager::routes::AppState::for_tests());
        let app = paddock_manager::routes::router(state.clone())
            .layer(axum::middleware::from_fn_with_state(
                Arc::new(policy()),
                guard,
            ))
            .layer(axum::Extension(state));
        for (path, body, status) in [
            (
                "/api/settings/import",
                r#"{"studio.pk_theme":"light"}"#,
                StatusCode::OK,
            ),
            (
                "/api/settings",
                r#"{"studio.pk_theme":"dark"}"#,
                StatusCode::OK,
            ),
            (
                "/api/settings/import",
                r#"{"studio.pk_theme":"light","studio.pk_sidebar_width":"0"}"#,
                StatusCode::OK,
            ),
            (
                "/api/settings/import",
                r#"{"credentials":"must not import"}"#,
                StatusCode::BAD_REQUEST,
            ),
        ] {
            let response = app
                .clone()
                .oneshot(
                    request(path)
                        .method(Method::PUT)
                        .header(header::CONTENT_TYPE, "application/json")
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), status);
        }
        let response = app
            .oneshot(request("/api/settings").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let body = axum::body::to_bytes(response.into_body(), 8192)
            .await
            .unwrap();
        let settings: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(settings["studio.pk_theme"], "dark");
        assert_eq!(settings["studio.pk_sidebar_width"], "0");
        assert!(settings.get("credentials").is_none());
    }

    #[tokio::test]
    async fn table_history_uses_sqlite_revisions_through_the_native_guard() {
        let state = Arc::new(paddock_manager::routes::AppState::for_tests());
        let app = paddock_manager::routes::router(state.clone())
            .layer(axum::middleware::from_fn_with_state(
                Arc::new(policy()),
                guard,
            ))
            .layer(axum::Extension(state));
        let call = |method: Method, path: &str, body: String| {
            app.clone().oneshot(
                request(path)
                    .method(method)
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
        };
        let key = "a47678d9564a8102e8467fd45a06b9e753a5450ed76f3af163a15eea89c9f3c0";
        let mut doc = serde_json::json!({"version":1,"id":"shared","title":"Table","model":"kumo","createdAt":1,"updatedAt":2,
            "draft":{"dataset":key,"fileName":"Example.csv","model":"kumo","port":11544,"estimators":8,"seed":0,"spec":null},
            "datasets":{key:"size,city,label\n1,a,yes\n2,b,no\n3,c,\n"},"runs":[]});
        assert_eq!(
            call(Method::PUT, "/api/table-history/shared", doc.to_string())
                .await
                .unwrap()
                .status(),
            StatusCode::BAD_REQUEST
        );
        let reply = call(
            Method::PUT,
            "/api/table-history/shared?revision=",
            doc.to_string(),
        )
        .await
        .unwrap();
        assert_eq!(reply.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(reply.into_body(), 8192).await.unwrap();
        let row: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let path = format!(
            "/api/table-history/shared?revision={}",
            row["revision"].as_str().unwrap()
        );
        doc["title"] = serde_json::json!("Native edit");
        assert_eq!(
            call(Method::PUT, &path, doc.to_string())
                .await
                .unwrap()
                .status(),
            StatusCode::OK
        );
        assert_eq!(
            call(Method::PUT, &path, doc.to_string())
                .await
                .unwrap()
                .status(),
            StatusCode::OK
        );
        assert_eq!(
            call(Method::DELETE, &path, String::new())
                .await
                .unwrap()
                .status(),
            StatusCode::CONFLICT
        );
        let reply = call(Method::GET, "/api/table-history/shared", String::new())
            .await
            .unwrap();
        let bytes = axum::body::to_bytes(reply.into_body(), 8192).await.unwrap();
        let snapshot: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(snapshot["doc"], doc.to_string());
        assert_eq!(
            call(Method::POST, "/api/table-history/shared", String::new())
                .await
                .unwrap()
                .status(),
            StatusCode::FORBIDDEN
        );
    }

    #[tokio::test]
    async fn web_and_native_reads_share_one_sqlite_document_and_conflict_contract() {
        let state = Arc::new(paddock_manager::routes::AppState::for_tests());
        let app = paddock_manager::routes::router(state.clone())
            .layer(axum::middleware::from_fn_with_state(
                Arc::new(policy()),
                guard,
            ))
            .layer(axum::Extension(state));
        let call = |method: Method, path: &str, body: String| {
            app.clone().oneshot(
                request(path)
                    .method(method)
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
        };
        let web = r#"{"id":"shared","title":"Web read","createdAt":1,"updatedAt":1,"runs":[{"state":"Full input","questions":{"z":{},"a":{}}}]}"#;
        let reply = call(
            Method::PUT,
            "/api/read-history/shared?revision=",
            web.into(),
        )
        .await
        .unwrap();
        let status = reply.status();
        let body = axum::body::to_bytes(reply.into_body(), 1024 * 1024)
            .await
            .unwrap();
        assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
        let reply = call(
            Method::GET,
            "/api/read-history/shared?envelope=true",
            String::new(),
        )
        .await
        .unwrap();
        let bytes = axum::body::to_bytes(reply.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let snapshot: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(snapshot["doc"], web);
        let revision = snapshot["revision"].as_str().unwrap();
        let native = web.replace("Web read", "Native read");
        let envelope = serde_json::json!({"doc":native}).to_string();
        let path = format!("/api/read-history/shared?envelope=true&revision={revision}");
        let reply = call(Method::PUT, &path, envelope.clone()).await.unwrap();
        assert_eq!(reply.status(), StatusCode::OK);
        let reply = call(Method::PUT, &path, envelope).await.unwrap();
        assert_eq!(reply.status(), StatusCode::OK, "a lost reply is retryable");
        let reply = call(Method::GET, "/api/read-history/shared", String::new())
            .await
            .unwrap();
        let bytes = axum::body::to_bytes(reply.into_body(), 1024 * 1024)
            .await
            .unwrap();
        assert_eq!(std::str::from_utf8(&bytes).unwrap(), native);
        let stale = format!("/api/read-history/shared?revision={revision}");
        let reply = call(Method::PUT, &stale, web.into()).await.unwrap();
        assert_eq!(reply.status(), StatusCode::CONFLICT);
        let reply = call(Method::DELETE, &stale, String::new()).await.unwrap();
        assert_eq!(reply.status(), StatusCode::CONFLICT);
    }

    #[tokio::test]
    async fn reads_crud_and_inference_pass_through_the_real_native_guard() {
        let state = Arc::new(paddock_manager::routes::AppState::for_tests());
        let app = paddock_manager::routes::router(state.clone())
            .layer(axum::middleware::from_fn_with_state(
                Arc::new(policy()),
                guard,
            ))
            .layer(axum::Extension(state));
        let call = |method: Method, path: &str, body: serde_json::Value| {
            app.clone().oneshot(
                request(path)
                    .method(method)
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
        };
        let response = call(Method::GET, "/api/reads", serde_json::Value::Null)
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let doc = serde_json::json!({"id":"reading","name":"Check","body":"{}","revision":""});
        let response = call(Method::POST, "/api/reads", doc.clone()).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), 8192)
            .await
            .unwrap();
        let saved: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let revision = saved["set"]["revision"].as_str().unwrap();
        let response = call(Method::GET, "/api/reads/reading", serde_json::Value::Null)
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let response = call(Method::PUT, "/api/reads/reading", doc).await.unwrap();
        assert_eq!(response.status(), StatusCode::CONFLICT);
        let response = call(
            Method::DELETE,
            &format!("/api/reads/reading?revision={revision}"),
            serde_json::Value::Null,
        )
        .await
        .unwrap();
        assert_eq!(response.status(), StatusCode::NO_CONTENT);
        // An isolated loopback runner proves inference crosses the same guard,
        // not just a policy predicate. Never contact a user's model/port.
        let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))
            .await
            .unwrap();
        let port = listener.local_addr().unwrap().port();
        let runner = axum::Router::new().route("/v1/systemone", axum::routing::post(
            |axum::Json(body): axum::Json<serde_json::Value>| async move {
                assert_eq!(body["model"], "reading-model");
                assert_eq!(body["state"], "The sky is blue.");
                axum::Json(serde_json::json!({"model":"reading-model","answers":{"q1":{"type":"noul","noul":0.9}}}))
            }
        ));
        let task = tokio::spawn(async move { axum::serve(listener, runner).await.unwrap() });
        let response = call(
            Method::POST,
            &format!("/api/runners/{port}/v1/systemone"),
            serde_json::json!({"model":"reading-model","state":"The sky is blue.","questions":{"q1":{"type":"noul","instructions":"Is the sky blue?"}}}),
        )
        .await
        .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), 8192)
            .await
            .unwrap();
        let result: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(result["answers"]["q1"]["noul"], 0.9);
        task.abort();
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/reads")
                    .header("host", "127.0.0.1:1234")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[test]
    fn native_image_relay_keeps_the_session_boundary() {
        for suffix in ["generations", "edits"] {
            let path = format!("/api/runners/12587/v1/images/{suffix}");
            assert!(api_allowed(&Method::POST, &path));
            assert!(!authorized(
                &policy(),
                &Request::builder()
                    .uri(&path)
                    .header("host", "127.0.0.1:1234")
                    .body(Body::empty())
                    .unwrap()
            ));
            assert!(authorized(
                &policy(),
                &request(&path).method("POST").body(Body::empty()).unwrap()
            ));
        }
        assert!(!api_allowed(
            &Method::POST,
            "/api/runners/12587/v1/images/delete"
        ));
    }
    fn policy() -> Policy {
        Policy {
            authority: "127.0.0.1:1234".into(),
            origin: "http://127.0.0.1:1234".into(),
            session: "session-secret".into(),
            mcp_token: "mcp-secret".into(),
            assets: Arc::new(RwLock::new(None)),
        }
    }
    fn request(path: &str) -> axum::http::request::Builder {
        Request::builder()
            .uri(path)
            .header("host", "127.0.0.1:1234")
            .header("cookie", format!("{COOKIE}=session-secret"))
    }

    #[tokio::test]
    async fn native_api_host_has_no_web_assets_until_explicitly_mounted() {
        let policy = policy();
        for path in [
            "/",
            "/studio",
            "/studio/chat/new",
            "/index.html",
            "/assets/app.js",
        ] {
            assert_eq!(
                asset(&policy, path, false).await.status(),
                StatusCode::NOT_FOUND
            );
        }
        assert!(validate_assets(PathBuf::from("/not-a-paddock-viewer-bundle")).is_err());
    }
    #[test]
    fn artifact_shell_is_public_but_has_no_data_or_api_authority() {
        let p = policy();
        for path in [
            "/native-artifact-host",
            "/artifact-frame",
            "/artifact-frame?img=1",
        ] {
            let r = Request::builder()
                .uri(path)
                .header("host", &p.authority)
                .body(Body::empty())
                .unwrap();
            assert!(preview_request_allowed(&p, &r));
            assert!(!authorized(&p, &r));
        }
        for path in [
            "/api/conversations",
            "/api/cloud",
            "/api/mcp/artifacts",
            "/studio",
            "/native-artifact-host?html=secret",
            "/artifact-frame?img=1&token=secret",
        ] {
            let r = Request::builder()
                .uri(path)
                .header("host", &p.authority)
                .body(Body::empty())
                .unwrap();
            assert!(!preview_request_allowed(&p, &r));
            assert!(!authorized(&p, &r));
        }
        for (key, value) in [
            ("host", "evil.test:1234"),
            ("origin", "null"),
            ("origin", "https://evil.test"),
            ("sec-fetch-site", "cross-site"),
        ] {
            let mut r = Request::builder()
                .uri("/artifact-frame")
                .header("host", &p.authority)
                .body(Body::empty())
                .unwrap();
            r.headers_mut().insert(
                header::HeaderName::from_bytes(key.as_bytes()).unwrap(),
                value.parse().unwrap(),
            );
            assert!(!preview_request_allowed(&p, &r));
        }
        let r = Request::builder()
            .method("POST")
            .uri("/artifact-frame")
            .header("host", &p.authority)
            .body(Body::empty())
            .unwrap();
        assert!(!preview_request_allowed(&p, &r));
    }
    #[tokio::test]
    async fn artifact_host_contains_no_content_or_privileged_script() {
        let response = preview_host(false);
        assert_eq!(response.headers()[header::CACHE_CONTROL], "no-store");
        let csp = response.headers()[header::CONTENT_SECURITY_POLICY]
            .to_str()
            .unwrap();
        assert!(csp.contains("script-src 'none'") && csp.contains("frame-ancestors 'none'"));
        assert!(
            response.headers()["permissions-policy"]
                .to_str()
                .unwrap()
                .contains("microphone=()")
        );
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let html = std::str::from_utf8(&body).unwrap();
        assert!(!html.contains("<script") && !html.contains("session") && !html.contains("/api/"));
        assert!(
            axum::body::to_bytes(preview_host(true).into_body(), 4096)
                .await
                .unwrap()
                .is_empty()
        );
    }
    #[test]
    fn session_is_required_even_on_loopback_and_rebinding_is_rejected() {
        let p = policy();
        assert!(authorized(
            &p,
            &request("/api/server").body(Body::empty()).unwrap()
        ));
        for (key, bad) in [
            ("origin", "http://127.0.0.1:9999"),
            ("origin", "null"),
            ("sec-fetch-site", "cross-site"),
            ("host", "evil.test:1234"),
        ] {
            let mut r = request("/api/server").body(Body::empty()).unwrap();
            r.headers_mut().insert(
                axum::http::HeaderName::from_bytes(key.as_bytes()).unwrap(),
                bad.parse().unwrap(),
            );
            assert!(!authorized(&p, &r));
        }
        let mut r = request("/api/server").body(Body::empty()).unwrap();
        r.headers_mut().remove("cookie");
        assert!(!authorized(&p, &r));
    }
    #[test]
    fn native_management_secrets_are_not_routes() {
        for path in [
            "/api/servers/1234/file",
            "/api/keys",
            "/api/export",
            "/api/updates/download",
            "/api/servers/files",
        ] {
            assert!(!api_allowed(&Method::GET, path));
        }
        assert!(!api_allowed(&Method::POST, "/api/runners"));
        assert!(api_allowed(&Method::POST, "/api/runners/1234/v1/responses"));
        assert!(api_allowed(&Method::PUT, "/api/conversations/test"));
        assert!(api_allowed(
            &Method::POST,
            "/api/cloud/mcp-approvals/fixture"
        ));
        assert!(!api_allowed(
            &Method::GET,
            "/api/cloud/mcp-approvals/fixture"
        ));
        for path in [
            "/api/connectors",
            "/api/connectors/check",
            "/api/connectors/fixture/oauth/start",
            "/api/connectors/fixture/scope",
        ] {
            assert!(!api_allowed(&Method::POST, path));
        }
    }
    #[test]
    fn mcp_capability_is_scoped_and_never_injected_into_remote_tools() {
        let p = policy();
        let r = request("/api/mcp/artifacts").body(Body::empty()).unwrap();
        assert!(!authorized(&p, &r));
        let r = Request::builder()
            .uri("/api/mcp/graph")
            .header("host", &p.authority)
            .header("authorization", "Bearer mcp-secret")
            .body(Body::empty())
            .unwrap();
        assert!(authorized(&p, &r));
        let input = serde_json::json!({"tools":[{"type":"mcp","server_url":"http://127.0.0.1:1234/api/mcp/artifacts"},{"type":"mcp","server_url":"https://evil.test/api/mcp/artifacts"}]});
        let out: serde_json::Value =
            serde_json::from_slice(&inject_tools(input.to_string().as_bytes(), &p).unwrap())
                .unwrap();
        assert_eq!(
            out["tools"][0]["headers"]["Authorization"],
            "Bearer mcp-secret"
        );
        assert!(out["tools"][1]["headers"].is_null());
    }
    #[test]
    fn echoed_private_capabilities_are_removed_at_every_chunk_boundary() {
        let input = b"data: {\"headers\":\"mcp-secret\"}\n\ndata: text\n\n";
        for split in 0..=input.len() {
            let mut pending = input[..split].to_vec();
            let mut out = redact(&mut pending, b"mcp-secret", false);
            pending.extend_from_slice(&input[split..]);
            out.extend(redact(&mut pending, b"mcp-secret", true));
            assert_eq!(
                String::from_utf8(out).unwrap(),
                "data: {\"headers\":\"[private]\"}\n\ndata: text\n\n"
            );
        }
        let mut plain = b"data: hello\n\n".to_vec();
        assert_eq!(redact(&mut plain, b"mcp-secret", false), b"data: hello\n\n");
        assert!(plain.is_empty());
    }
}
