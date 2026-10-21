//! The Anthropic dialect: `/v1/messages/batches`, shaped as the official SDK
//! reads it. Requests arrive inline. The body is streamed to disk first and
//! then converted to batch lines one request at a time, so a create carrying
//! ten thousand base64 pictures never sits in memory whole.

use std::sync::Arc;

use axum::Router;
use axum::extract::{DefaultBodyLimit, Path, Query, Request, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use serde::Deserialize;
use serde_json::{Value, json};

use super::{Batches, files, idem_key, new_id, now_ms, parse_window};
use crate::routes::AppState;
use crate::store::BatchRow;

pub(crate) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route(
            "/v1/messages/batches",
            post(create).layer(DefaultBodyLimit::disable()).get(list),
        )
        .route("/v1/messages/batches/{id}", get(retrieve).delete(delete))
        .route("/v1/messages/batches/{id}/cancel", post(cancel))
        .route("/v1/messages/batches/{id}/results", get(results))
}

/// An Anthropic-shaped error.
fn error(status: StatusCode, kind: &str, message: impl Into<String>) -> Response {
    (
        status,
        Json(json!({"type": "error", "error": {"type": kind, "message": message.into()}})),
    )
        .into_response()
}

fn bad(message: impl Into<String>) -> Response {
    error(StatusCode::BAD_REQUEST, "invalid_request_error", message)
}

fn not_found() -> Response {
    error(
        StatusCode::NOT_FOUND,
        "not_found_error",
        "message batch not found",
    )
}

fn internal(e: impl std::fmt::Display) -> Response {
    error(
        StatusCode::INTERNAL_SERVER_ERROR,
        "api_error",
        e.to_string(),
    )
}

fn rfc3339(ms: i64) -> String {
    chrono::DateTime::from_timestamp_millis(ms)
        .unwrap_or_default()
        .to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

/// The scheme and authority this request reached us on, for `results_url`.
fn base_url(headers: &HeaderMap, secure: bool) -> String {
    let host = headers
        .get(axum::http::header::HOST)
        .and_then(|v| v.to_str().ok())
        .filter(|h| !h.is_empty())
        .unwrap_or("localhost");
    format!("{}://{host}", if secure { "https" } else { "http" })
}

async fn create(State(st): State<Arc<AppState>>, req: Request) -> Response {
    let b = st.batches.clone();
    let headers = req.headers().clone();
    let secure = req
        .extensions()
        .get::<paddock_tls::serve::Secure>()
        .is_some_and(|s| s.0);
    // a retry of a create that already landed: answer with the first one,
    // before streaming the body a second time
    let key = idem_key(&headers);
    if let Some(k) = &key {
        match b.db.batch_by_idem(k) {
            Ok(Some(r)) if r.dialect == "anthropic" => {
                return Json(batch_object(&b, &r, &base_url(&headers, secure))).into_response();
            }
            Ok(Some(_)) => return bad("Idempotency-Key: already used for an OpenAI batch"),
            Ok(None) => {}
            Err(e) => return internal(e),
        }
    }
    if let Some(len) = headers
        .get(axum::http::header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        // the body lands twice: once as received, once as lines
        && let Err(e) = files::headroom(&b.root, len.saturating_mul(2))
    {
        return error(StatusCode::INSUFFICIENT_STORAGE, "api_error", e);
    }
    let id = new_id("msgbatch_");
    let dir = b.job_dir(&id);
    let raw = dir.join("create.json");
    let stream = req.into_body().into_data_stream();
    match files::stream_to_file(stream, &raw).await {
        Ok(_) => {}
        Err(files::UploadError::Disk(e)) => {
            let _ = tokio::fs::remove_dir_all(&dir).await;
            return error(StatusCode::INSUFFICIENT_STORAGE, "api_error", e);
        }
        Err(e) => {
            let _ = tokio::fs::remove_dir_all(&dir).await;
            return bad(e.to_string());
        }
    }
    let lines = dir.join("in.jsonl");
    let (src, dst) = (raw.clone(), lines.clone());
    let converted = tokio::task::spawn_blocking(move || {
        let r = std::fs::File::open(&src).map_err(|e| e.to_string())?;
        let w = std::fs::File::create(&dst).map_err(|e| e.to_string())?;
        let c = super::index::convert_anthropic(r, &w)?;
        w.sync_all().map_err(|e| e.to_string())?;
        Ok::<_, String>(c)
    })
    .await;
    let _ = tokio::fs::remove_file(&raw).await;
    let c = match converted {
        Ok(Ok(c)) => c,
        Ok(Err(e)) => {
            let _ = tokio::fs::remove_dir_all(&dir).await;
            return bad(e);
        }
        Err(e) => {
            let _ = tokio::fs::remove_dir_all(&dir).await;
            return internal(e);
        }
    };
    let fail = |r: Response| async {
        let _ = tokio::fs::remove_dir_all(&dir).await;
        r
    };
    let opts = match super::openai::options(Some(&c.paddock)) {
        Ok(o) => o,
        Err(e) => return fail(bad(e)).await,
    };
    let window = opts["completion_window"]
        .as_str()
        .unwrap_or("24h")
        .to_owned();
    let Some(window_ms) = parse_window(&window) else {
        return fail(bad(format!(
            "paddock.completion_window: {window:?} - expected hours or days, \"24h\" up to \"30d\""
        )))
        .await;
    };
    if let Some(e) = super::openai::unserved(&b, &c.model, "/v1/messages").await {
        return fail(bad(e)).await;
    }
    let now = now_ms();
    let row = BatchRow {
        id: id.clone(),
        dialect: "anthropic".into(),
        endpoint: "/v1/messages".into(),
        model: c.model,
        completion_window: window,
        // indexed by the run before it starts; reads as in_progress here
        status: "validating".into(),
        created_ms: now,
        expires_ms: now + window_ms,
        metadata: Value::Null,
        errors: Value::Null,
        options: opts,
        total: c.total,
        idem_key: key.clone(),
        ..Default::default()
    };
    if let Err(e) = b.db.batch_insert(&row) {
        if crate::store::is_conflict(&e)
            && let Some(k) = &key
            && let Ok(Some(r)) = b.db.batch_by_idem(k)
        {
            return fail(Json(batch_object(&b, &r, &base_url(&headers, secure))).into_response())
                .await;
        }
        return fail(internal(e)).await;
    }
    b.wake.notify_one();
    tracing::info!(batch = %id, model = %row.model, requests = c.total, "batch: created");
    Json(batch_object(&b, &row, &base_url(&headers, secure))).into_response()
}

#[derive(Deserialize)]
struct ListQuery {
    limit: Option<usize>,
    before_id: Option<String>,
    after_id: Option<String>,
}

async fn list(
    State(st): State<Arc<AppState>>,
    secure: Option<axum::Extension<paddock_tls::serve::Secure>>,
    headers: HeaderMap,
    Query(q): Query<ListQuery>,
) -> Response {
    let b = &st.batches;
    let base = base_url(&headers, secure.is_some_and(|s| s.0.0));
    let limit = q.limit.unwrap_or(20).clamp(1, 1000);
    match b.db.batch_list(
        "anthropic",
        q.after_id.as_deref(),
        q.before_id.as_deref(),
        limit + 1,
    ) {
        Ok(mut rows) => {
            let has_more = rows.len() > limit;
            if q.before_id.is_some() && has_more {
                rows.remove(0); // nearest-newer page: drop the far end
            } else {
                rows.truncate(limit);
            }
            let data: Vec<Value> = rows.iter().map(|r| batch_object(b, r, &base)).collect();
            Json(json!({
                "first_id": data.first().map(|v| v["id"].clone()),
                "last_id": data.last().map(|v| v["id"].clone()),
                "has_more": has_more,
                "data": data,
            }))
            .into_response()
        }
        Err(e) => internal(e),
    }
}

fn anthropic_row(b: &Batches, id: &str) -> Result<BatchRow, Box<Response>> {
    match b.db.batch_get(id) {
        Ok(Some(r)) if r.dialect == "anthropic" => Ok(r),
        Ok(_) => Err(Box::new(not_found())),
        Err(e) => Err(Box::new(internal(e))),
    }
}

async fn retrieve(
    State(st): State<Arc<AppState>>,
    secure: Option<axum::Extension<paddock_tls::serve::Secure>>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    let base = base_url(&headers, secure.is_some_and(|s| s.0.0));
    match anthropic_row(&st.batches, &id) {
        Ok(r) => Json(batch_object(&st.batches, &r, &base)).into_response(),
        Err(e) => *e,
    }
}

async fn cancel(
    State(st): State<Arc<AppState>>,
    secure: Option<axum::Extension<paddock_tls::serve::Secure>>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    let b = &st.batches;
    let base = base_url(&headers, secure.is_some_and(|s| s.0.0));
    let row = match anthropic_row(b, &id) {
        Ok(r) => r,
        Err(e) => return *e,
    };
    // cancelling an ended batch is a no-op that returns it, as the API does
    if !row.is_terminal()
        && row.status != "cancelling"
        && let Err(e) = b.db.batch_transition(&id, &row.status, "cancelling")
    {
        return internal(e);
    }
    b.wake.notify_one();
    match anthropic_row(b, &id) {
        Ok(r) => Json(batch_object(b, &r, &base)).into_response(),
        Err(e) => *e,
    }
}

async fn delete(State(st): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    let b = &st.batches;
    let row = match anthropic_row(b, &id) {
        Ok(r) => r,
        Err(e) => return *e,
    };
    if !row.is_terminal() {
        return bad(format!(
            "message batch {id} is still processing - cancel it, and delete it once it has ended"
        ));
    }
    if let Err(e) = b.db.batch_delete(&id) {
        return internal(e);
    }
    let _ = tokio::fs::remove_dir_all(b.job_dir(&id)).await;
    Json(json!({"id": id, "type": "message_batch_deleted"})).into_response()
}

async fn results(State(st): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    let b = &st.batches;
    let row = match anthropic_row(b, &id) {
        Ok(r) => r,
        Err(e) => return *e,
    };
    if !row.is_terminal() {
        return bad(format!(
            "message batch {id} has not ended yet; results are available once processing_status \
             is \"ended\""
        ));
    }
    match files::body_of(&b.job_dir(&id).join("out.jsonl"), row.out_len as u64).await {
        Ok(body) => (
            [(axum::http::header::CONTENT_TYPE, "application/x-jsonl")],
            body,
        )
            .into_response(),
        Err(e) => internal(e),
    }
}

/// The MessageBatch object, plus the `paddock` progress block.
fn batch_object(b: &Batches, r: &BatchRow, base: &str) -> Value {
    let ended = r.is_terminal();
    let processing_status = if ended {
        "ended"
    } else if r.status == "cancelling" {
        "canceling"
    } else {
        "in_progress"
    };
    let settled = r.succeeded + r.errored + r.canceled + r.expired;
    let ended_at = [r.completed_ms, r.cancelled_ms, r.expired_ms, r.failed_ms]
        .into_iter()
        .flatten()
        .next();
    json!({
        "id": r.id,
        "type": "message_batch",
        "processing_status": processing_status,
        "request_counts": {
            "processing": if ended { 0 } else { (r.total - settled).max(0) },
            "succeeded": r.succeeded,
            "errored": r.errored,
            "canceled": r.canceled,
            "expired": r.expired,
        },
        "ended_at": ended_at.map(rfc3339),
        "created_at": rfc3339(r.created_ms),
        "expires_at": rfc3339(r.expires_ms),
        "archived_at": null,
        "cancel_initiated_at": r.cancelling_ms.map(rfc3339),
        "results_url": ended.then(|| format!("{base}/v1/messages/batches/{}/results", r.id)),
        "paddock": super::openai::paddock_block(b, r),
    })
}
