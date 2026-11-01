//! The manager's key gate: which caller may reach `/api` and `/v1`, and the
//! Studio's login cookie. Split out of routes.rs; the router mounts these.
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use paddock_api::ErrorBody;

use super::AppState;
use crate::store::KeyScope;

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

/// Bearer auth for `/api` and `/v1`. A no-op when no key is configured
/// (loopback default); otherwise the header key must equal the configured key
/// or match a stored (db-backed) API key, and `/api` - the control plane:
/// spawning models, writing their configs, minting keys - also needs that key
/// to be admin-scoped (`KeyScope`). The configured key is always admin.
/// Non-API paths (the SPA) always pass so the UI can load. Loopback PEERS are exempt (`peer_is_local`) - the runner's
/// policy, and what makes a network bind usable at all: without it, --host
/// 0.0.0.0 locked the box's own Studio out of every /api route and every page
/// rendered empty. The exemption ends where a reverse proxy begins.
pub(super) async fn auth_mw(
    State(state): State<Arc<AppState>>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    if let Some(required) = &state.auth_key {
        let local = peer_is_local(&req, state.trusted_proxy);
        let path = req.uri().path();
        if !local && (path.starts_with("/api/") || path.starts_with("/v1/")) {
            let scope_of = |k: &str| key_scope(&state, required, k);
            // Bearer for OpenAI-style clients; `x-api-key` is how the
            // Anthropic SDKs send theirs (the batch surface speaks both)
            let bearer = req
                .headers()
                .get(axum::http::header::AUTHORIZATION)
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.strip_prefix("Bearer "))
                .or_else(|| req.headers().get("x-api-key").and_then(|v| v.to_str().ok()))
                .and_then(scope_of);
            // The Studio's key-gate session: a browser cannot attach Bearer
            // headers to every fetch/SSE the SPA makes, so /auth/login turns
            // the key into an HttpOnly cookie and this accepts it.
            let cookie = req
                .headers()
                .get(axum::http::header::COOKIE)
                .and_then(|v| v.to_str().ok())
                .and_then(|c| {
                    c.split(';')
                        .map(str::trim)
                        .find_map(|p| p.strip_prefix("paddock_key="))
                })
                .and_then(scope_of);
            let scope = match (bearer, cookie) {
                (Some(KeyScope::Admin), _) | (_, Some(KeyScope::Admin)) => KeyScope::Admin,
                (Some(s), _) | (_, Some(s)) => s,
                (None, None) => {
                    return (
                        StatusCode::UNAUTHORIZED,
                        Json(ErrorBody::new(
                            "invalid_api_key",
                            "missing or invalid API key",
                        )),
                    )
                        .into_response();
                }
            };
            if path.starts_with("/api/") && scope != KeyScope::Admin {
                return (
                    StatusCode::FORBIDDEN,
                    Json(ErrorBody::new(
                        "permission_error",
                        "this key is limited to the inference API (/v1); \
                         the Studio and /api need an admin key",
                    )),
                )
                    .into_response();
            }
        }
    }
    next.run(req).await
}

/// What `key` opens: the configured key is admin by definition; anything
/// else is whatever the store says it was minted as.
fn key_scope(state: &AppState, required: &str, key: &str) -> Option<KeyScope> {
    if key == required {
        return Some(KeyScope::Admin);
    }
    state.db.verify_api_key(key).ok().flatten()
}

#[derive(serde::Deserialize)]
pub(super) struct LoginBody {
    key: String,
}

/// `POST /auth/login` - the Studio's key gate for network browsers. Lives
/// outside `/api` so it is reachable before auth. A valid key comes back as
/// an HttpOnly cookie (30 days), which `auth_mw` accepts alongside the
/// Bearer header - the key never sits in script-readable storage.
pub(super) async fn auth_login(
    State(state): State<Arc<AppState>>,
    Json(b): Json<LoginBody>,
) -> Response {
    // The Studio is the control plane, so only an admin key opens it: an
    // inference key would log in to a Studio where every page is refused.
    let scope = match state.auth_key.as_deref() {
        Some(k) => key_scope(&state, k, &b.key),
        None => state.db.verify_api_key(&b.key).ok().flatten(),
    };
    match scope {
        Some(KeyScope::Admin) => {}
        Some(KeyScope::Inference) => {
            return (
                StatusCode::FORBIDDEN,
                Json(ErrorBody::new(
                    "permission_error",
                    "that key is for the inference API; the Studio needs an admin key",
                )),
            )
                .into_response();
        }
        None => {
            return (
                StatusCode::UNAUTHORIZED,
                Json(ErrorBody::new(
                    "invalid_api_key",
                    "that key does not open this paddock",
                )),
            )
                .into_response();
        }
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
pub(super) async fn auth_logout() -> Response {
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
    use super::super::{AppState, router};
    use super::*;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

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

    /// An inference key opens the manager's /v1 surface and nothing under
    /// /api; an admin key and the configured key open both; the Studio's
    /// login takes admin keys only.
    #[tokio::test]
    async fn inference_keys_stay_out_of_the_control_plane() {
        let mut state = AppState::for_tests();
        state.auth_key = Some("pk-configured".into());
        let (_, serving) = state
            .db
            .create_api_key("serving", KeyScope::Inference)
            .expect("mint");
        let (_, ops) = state
            .db
            .create_api_key("ops", KeyScope::Admin)
            .expect("mint");
        let app = router(Arc::new(state));
        let send = |method: &str, path: &str, key: &str, body: &str| {
            axum::http::Request::builder()
                .method(method)
                .uri(path)
                .header("authorization", format!("Bearer {key}"))
                .header("content-type", "application/json")
                .body(axum::body::Body::from(body.to_owned()))
                .expect("request")
        };
        let status = |req: axum::http::Request<axum::body::Body>| {
            let app = app.clone();
            async move { app.oneshot(req).await.expect("response").status() }
        };
        // the serving key: batches yes, control plane no
        assert_eq!(
            status(send("GET", "/v1/batches", &serving, "")).await,
            StatusCode::OK
        );
        for (method, path) in [
            ("GET", "/api/keys"),
            ("POST", "/api/keys"),
            ("POST", "/api/runners"),
            ("GET", "/api/servers/12481/file"),
            ("PUT", "/api/servers/12481/file"),
            ("GET", "/api/settings"),
        ] {
            let res = app
                .clone()
                .oneshot(send(method, path, &serving, "{}"))
                .await
                .expect("response");
            assert_eq!(res.status(), StatusCode::FORBIDDEN, "{method} {path}");
            assert_eq!(body_json(res).await["error"]["type"], "permission_error");
        }
        // the cookie route is held to the same line
        let cookie = axum::http::Request::get("/api/keys")
            .header("cookie", format!("paddock_key={serving}"))
            .body(axum::body::Body::empty())
            .expect("request");
        assert_eq!(status(cookie).await, StatusCode::FORBIDDEN);
        // admin and configured keys open the control plane
        for key in [ops.as_str(), "pk-configured"] {
            assert_eq!(
                status(send("GET", "/api/keys", key, "")).await,
                StatusCode::OK
            );
            assert_eq!(
                status(send("GET", "/v1/batches", key, "")).await,
                StatusCode::OK
            );
        }
        // a minted key is inference unless admin is asked for by name
        let res = app
            .clone()
            .oneshot(send("POST", "/api/keys", &ops, r#"{"name":"x"}"#))
            .await
            .expect("response");
        assert_eq!(res.status(), StatusCode::CREATED);
        assert_eq!(body_json(res).await["scope"], "inference");
        let bad = send("POST", "/api/keys", &ops, r#"{"scope":"root"}"#);
        assert_eq!(status(bad).await, StatusCode::BAD_REQUEST);
        // the Studio's login: admin keys only
        let login = |key: &str| {
            axum::http::Request::post("/auth/login")
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({ "key": key }).to_string(),
                ))
                .expect("request")
        };
        assert_eq!(status(login(&serving)).await, StatusCode::FORBIDDEN);
        assert_eq!(status(login(&ops)).await, StatusCode::NO_CONTENT);
        assert_eq!(status(login("pk-nope")).await, StatusCode::UNAUTHORIZED);
    }
}
