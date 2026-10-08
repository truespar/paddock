//! `/api/updates` answers, and answers honestly when the release server is not
//! there.
//!
//! The offline case is the one that matters. A Manager whose UI breaks because
//! a laptop is on a train is worse than one that says it does not know, so the
//! endpoint must be 200-with-`unknown` rather than a 5xx - and that is easy to
//! get wrong by letting a `?` escape.
// Test code: a failed assumption stops the test where it happened.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use paddock_manager::routes::{AppState, router};
use tower::ServiceExt;

/// Both tests point `PADDOCK_API_BASE` at a dead port and unset it when done,
/// and the harness runs them on parallel threads: whichever finished first
/// unset the variable under the other, which then asked the REAL release
/// server and read `current`. One holds the variable at a time.
static API_BASE: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

async fn get(app: axum::Router, uri: &str) -> (StatusCode, serde_json::Value) {
    let resp = app
        .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
        .await
        .expect("router responds");
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .expect("body");
    let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, json)
}

/// Point at a host that cannot answer and confirm the endpoint still 200s with
/// an honest `unknown`, rather than propagating a transport error to the UI.
#[tokio::test]
async fn an_unreachable_release_server_is_reported_not_thrown() {
    let _env = API_BASE.lock().await;
    // 127.0.0.1:1 - reserved, nothing listens, connection refused immediately.
    // Better than a bogus hostname: no DNS wait, so the test stays fast.
    unsafe { std::env::set_var("PADDOCK_API_BASE", "http://127.0.0.1:1") };

    let app = router(Arc::new(AppState::for_tests()));
    let (status, body) = get(app, "/api/updates").await;

    assert_eq!(
        status,
        StatusCode::OK,
        "an offline check is information, not a failure"
    );
    assert_eq!(body["state"], "unknown", "got {body}");
    assert!(
        body["current"].is_string(),
        "must still say what WE are running"
    );
    assert!(
        body["why"].is_string(),
        "the reason belongs in the payload for the log"
    );
    // The download slot rides along so the UI needs one poll, not two.
    assert!(
        body.get("download").is_some(),
        "download key must always be present"
    );
    assert!(body["download"].is_null(), "nothing downloading yet");

    unsafe { std::env::remove_var("PADDOCK_API_BASE") };
}

/// Starting a download with no reachable server must refuse cleanly rather than
/// leaving a half-built job in state that the UI then polls forever.
#[tokio::test]
async fn a_download_with_no_reachable_server_refuses_and_starts_nothing() {
    let _env = API_BASE.lock().await;
    unsafe { std::env::set_var("PADDOCK_API_BASE", "http://127.0.0.1:1") };

    let state = Arc::new(AppState::for_tests());
    let app = router(state.clone());
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/updates/download")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .expect("router responds");

    assert_eq!(
        resp.status(),
        StatusCode::BAD_GATEWAY,
        "unreachable is a gateway problem"
    );
    assert!(
        state.update_dl.lock().unwrap().is_none(),
        "a refused start must leave NO job behind - a stuck 'running' the UI polls \
         forever is worse than a clear failure"
    );

    unsafe { std::env::remove_var("PADDOCK_API_BASE") };
}
