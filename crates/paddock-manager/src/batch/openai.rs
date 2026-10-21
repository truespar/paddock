//! The OpenAI dialect: `/v1/files` and `/v1/batches`, shaped as the official
//! SDK reads them. Wire shapes follow the API reference; the one addition is a
//! `paddock` object - on create, the extension options; on every batch, the
//! run's live progress.

use std::sync::Arc;

use axum::Router;
use axum::extract::{DefaultBodyLimit, Multipart, Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use serde::Deserialize;
use serde_json::{Value, json};

use super::{Batches, ENDPOINTS, files, idem_key, new_id, now_ms, parse_window};
use crate::routes::AppState;
use crate::store::{BatchFileRow, BatchRow};

pub(crate) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        // an input of base64 pictures is tens of GB: no body cap, the disk
        // headroom check is the limit
        .route(
            "/v1/files",
            post(upload)
                .layer(DefaultBodyLimit::disable())
                .get(list_files),
        )
        .route("/v1/files/{id}", get(get_file).delete(delete_file))
        .route("/v1/files/{id}/content", get(file_content))
        .route("/v1/batches", post(create).get(list))
        .route("/v1/batches/{id}", get(retrieve))
        .route("/v1/batches/{id}/cancel", post(cancel))
        .merge(super::uploads::routes())
}

/// An OpenAI-shaped error.
pub(crate) fn error(status: StatusCode, kind: &str, message: impl Into<String>) -> Response {
    (status, Json(paddock_api::ErrorBody::new(kind, message))).into_response()
}

fn bad(message: impl Into<String>) -> Response {
    error(StatusCode::BAD_REQUEST, "invalid_request_error", message)
}

fn not_found(what: &str) -> Response {
    error(
        StatusCode::NOT_FOUND,
        "invalid_request_error",
        format!("No such {what}"),
    )
}

fn internal(e: impl std::fmt::Display) -> Response {
    error(
        StatusCode::INTERNAL_SERVER_ERROR,
        "server_error",
        e.to_string(),
    )
}

fn secs(ms: i64) -> i64 {
    ms / 1000
}

pub(crate) fn file_object(f: &BatchFileRow) -> Value {
    json!({
        "id": f.id,
        "object": "file",
        "bytes": f.bytes,
        "created_at": secs(f.created_ms),
        "filename": f.filename,
        "purpose": f.purpose,
        "status": "processed",
        "expires_at": null,
        "status_details": null,
    })
}

/// `POST /v1/files` - multipart `purpose` + `file`, the file streamed to disk.
async fn upload(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    mut mp: Multipart,
) -> Response {
    let b = &st.batches;
    let key = idem_key(&headers);
    if let Some(k) = &key {
        match b.db.batch_file_by_idem(k) {
            Ok(Some(f)) => return Json(file_object(&f)).into_response(),
            Ok(None) => {}
            Err(e) => return internal(e),
        }
    }
    let dir = b.files_dir();
    if let Some(len) = headers
        .get(axum::http::header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        && let Err(e) = files::headroom(&dir, len)
    {
        return error(StatusCode::INSUFFICIENT_STORAGE, "insufficient_storage", e);
    }
    let id = new_id("file-");
    let dest = dir.join(&id);
    let mut purpose: Option<String> = None;
    let mut written: Option<(u64, String)> = None;
    loop {
        let field = match mp.next_field().await {
            Ok(Some(f)) => f,
            Ok(None) => break,
            Err(e) => {
                let _ = tokio::fs::remove_file(&dest).await;
                return bad(format!("invalid multipart body: {e}"));
            }
        };
        match field.name() {
            Some("purpose") => match field.text().await {
                Ok(t) => purpose = Some(t),
                Err(e) => return bad(format!("purpose: {e}")),
            },
            Some("file") if written.is_none() => {
                let name = field.file_name().unwrap_or("upload.jsonl").to_owned();
                match files::stream_to_file(Box::pin(field), &dest).await {
                    Ok(n) => written = Some((n, name)),
                    Err(files::UploadError::Disk(e)) => {
                        return error(StatusCode::INSUFFICIENT_STORAGE, "insufficient_storage", e);
                    }
                    Err(e) => return bad(e.to_string()),
                }
            }
            _ => {}
        }
    }
    let Some((bytes, filename)) = written else {
        return bad("file: a file part is required");
    };
    if purpose.as_deref() != Some("batch") {
        let _ = tokio::fs::remove_file(&dest).await;
        return bad(format!(
            "purpose: {:?} is not supported here - this server stores files for batches \
             (purpose \"batch\")",
            purpose.unwrap_or_default()
        ));
    }
    let row = BatchFileRow {
        id: id.clone(),
        purpose: "batch".into(),
        filename,
        bytes: bytes as i64,
        created_ms: now_ms(),
        path: format!("files/{id}"),
        batch_id: None,
    };
    if let Err(e) = b.db.batch_file_insert(&row) {
        let _ = tokio::fs::remove_file(&dest).await;
        return internal(e);
    }
    if let Some(k) = &key {
        match b.db.batch_file_set_idem(&id, k) {
            Ok(true) => {}
            // a concurrent twin stored it first: keep that one
            Ok(false) => {
                let _ = b.db.batch_file_delete(&id);
                let _ = tokio::fs::remove_file(&dest).await;
                return match b.db.batch_file_by_idem(k) {
                    Ok(Some(f)) => Json(file_object(&f)).into_response(),
                    Ok(None) => internal("idempotent upload vanished"),
                    Err(e) => internal(e),
                };
            }
            Err(e) => return internal(e),
        }
    }
    tracing::info!(file = %id, bytes, "batch: input file stored");
    Json(file_object(&row)).into_response()
}

#[derive(Deserialize)]
struct FileListQuery {
    purpose: Option<String>,
    after: Option<String>,
    limit: Option<usize>,
    order: Option<String>,
}

async fn list_files(State(st): State<Arc<AppState>>, Query(q): Query<FileListQuery>) -> Response {
    let limit = q.limit.unwrap_or(10_000).clamp(1, 10_000);
    let asc = q.order.as_deref() == Some("asc");
    match st
        .batches
        .db
        .batch_files_list(q.purpose.as_deref(), q.after.as_deref(), limit + 1, asc)
    {
        Ok(mut rows) => {
            let has_more = rows.len() > limit;
            rows.truncate(limit);
            Json(page(rows.iter().map(file_object).collect(), has_more)).into_response()
        }
        Err(e) => internal(e),
    }
}

/// The list envelope both list endpoints share.
fn page(data: Vec<Value>, has_more: bool) -> Value {
    json!({
        "object": "list",
        "first_id": data.first().map(|v| v["id"].clone()),
        "last_id": data.last().map(|v| v["id"].clone()),
        "has_more": has_more,
        "data": data,
    })
}

async fn get_file(State(st): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    match st.batches.db.batch_file_get(&id) {
        Ok(Some(f)) => Json(file_object(&f)).into_response(),
        Ok(None) => not_found("file"),
        Err(e) => internal(e),
    }
}

async fn delete_file(State(st): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    let b = &st.batches;
    let f = match b.db.batch_file_get(&id) {
        Ok(Some(f)) => f,
        Ok(None) => return not_found("file"),
        Err(e) => return internal(e),
    };
    match b.db.batch_file_in_use(&id) {
        Ok(true) => {
            return error(
                StatusCode::CONFLICT,
                "invalid_request_error",
                "this file is the input of a batch that is still running; cancel the batch first",
            );
        }
        Ok(false) => {}
        Err(e) => return internal(e),
    }
    if let Err(e) = b.db.batch_file_delete(&id) {
        return internal(e);
    }
    let _ = tokio::fs::remove_file(b.root.join(&f.path)).await;
    Json(json!({"id": id, "object": "file", "deleted": true})).into_response()
}

async fn file_content(State(st): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    let b = &st.batches;
    let f = match b.db.batch_file_get(&id) {
        Ok(Some(f)) => f,
        Ok(None) => return not_found("file"),
        Err(e) => return internal(e),
    };
    // a result file is served at its registered length: the bytes past it
    // (none, once finalized) are not part of the file
    match files::body_of(&b.root.join(&f.path), f.bytes as u64).await {
        Ok(body) => (
            [(axum::http::header::CONTENT_TYPE, "application/octet-stream")],
            body,
        )
            .into_response(),
        Err(e) => internal(e),
    }
}

#[derive(Deserialize)]
struct CreateBody {
    input_file_id: String,
    endpoint: String,
    completion_window: String,
    #[serde(default)]
    metadata: Option<Value>,
    #[serde(default)]
    paddock: Option<Value>,
}

/// The `paddock` extension options, checked. Shared with the Anthropic dialect.
pub(crate) fn options(ext: Option<&Value>) -> Result<Value, String> {
    let mut out = serde_json::Map::new();
    let Some(ext) = ext.filter(|v| !v.is_null()) else {
        return Ok(Value::Object(out));
    };
    let o = ext.as_object().ok_or("paddock: must be an object")?;
    for (k, v) in o {
        match k.as_str() {
            "webhook_url" => {
                let url = v.as_str().ok_or("paddock.webhook_url: must be a string")?;
                let parsed =
                    reqwest::Url::parse(url).map_err(|e| format!("paddock.webhook_url: {e}"))?;
                if !matches!(parsed.scheme(), "http" | "https") {
                    return Err("paddock.webhook_url: must be http or https".into());
                }
                out.insert(k.clone(), v.clone());
            }
            "yield_to_live" => {
                v.as_bool()
                    .ok_or("paddock.yield_to_live: must be a boolean")?;
                out.insert(k.clone(), v.clone());
            }
            "concurrency" => {
                v.as_u64()
                    .filter(|c| (1..=1024).contains(c))
                    .ok_or("paddock.concurrency: must be an integer from 1 to 1024")?;
                out.insert(k.clone(), v.clone());
            }
            "completion_window" => {
                out.insert(k.clone(), v.clone());
            }
            other => return Err(format!("paddock.{other}: unknown option")),
        }
    }
    Ok(Value::Object(out))
}

/// OpenAI's metadata rules: up to 16 string pairs, keys to 64 chars, values to
/// 512.
fn check_metadata(m: &Value) -> Result<(), String> {
    if m.is_null() {
        return Ok(());
    }
    let o = m.as_object().ok_or("metadata: must be an object")?;
    if o.len() > 16 {
        return Err("metadata: at most 16 keys".into());
    }
    for (k, v) in o {
        let ok = k.chars().count() <= 64 && v.as_str().is_some_and(|s| s.chars().count() <= 512);
        if !ok {
            return Err(format!(
                "metadata.{k}: keys are up to 64 characters, values strings up to 512"
            ));
        }
    }
    Ok(())
}

/// Refuse a model nothing serves, naming what is served.
pub(crate) async fn unserved(b: &Batches, model: &str, endpoint: &str) -> Option<String> {
    if b.fleet.resolve(model, endpoint).await.is_some() {
        return None;
    }
    let served = b.fleet.served(endpoint).await;
    let listing = if served.is_empty() {
        "nothing is serving right now".to_string()
    } else {
        format!("serving: {}", served.join(", "))
    };
    Some(format!(
        "no running service serves model {model:?} ({listing}). A batch runs on a model that is \
         already being served - start it first"
    ))
}

async fn create(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    let b = &st.batches;
    let key = idem_key(&headers);
    if let Some(k) = &key {
        match b.db.batch_by_idem(k) {
            Ok(Some(r)) if r.dialect == "openai" => {
                return Json(batch_object(b, &r)).into_response();
            }
            Ok(Some(_)) => return bad("Idempotency-Key: already used for a message batch"),
            Ok(None) => {}
            Err(e) => return internal(e),
        }
    }
    let req: CreateBody = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return bad(format!("invalid request body: {e}")),
    };
    if !ENDPOINTS.contains(&req.endpoint.as_str()) || req.endpoint == "/v1/messages" {
        return bad(format!(
            "endpoint: {:?} is not a batch endpoint; use /v1/chat/completions, /v1/responses, \
             /v1/completions or /v1/embeddings",
            req.endpoint
        ));
    }
    let Some(window) = parse_window(&req.completion_window) else {
        return bad(format!(
            "completion_window: {:?} - expected hours or days, \"24h\" up to \"30d\"",
            req.completion_window
        ));
    };
    let metadata = req.metadata.unwrap_or(Value::Null);
    if let Err(e) = check_metadata(&metadata) {
        return bad(e);
    }
    let opts = match options(req.paddock.as_ref()) {
        Ok(o) => o,
        Err(e) => return bad(e),
    };
    let file = match b.db.batch_file_get(&req.input_file_id) {
        Ok(Some(f)) if f.purpose == "batch" => f,
        Ok(Some(_)) => {
            return bad("input_file_id: the file was not uploaded with purpose \"batch\"");
        }
        Ok(None) => return not_found("file"),
        Err(e) => return internal(e),
    };
    let path = b.root.join(&file.path);
    let model = match tokio::task::spawn_blocking(move || super::index::first_model(&path)).await {
        Ok(Ok(Some(m))) => m,
        Ok(Ok(None)) => {
            return bad("input_file_id: the file's first request names no model in its body");
        }
        Ok(Err(e)) => return internal(e),
        Err(e) => return internal(e),
    };
    if let Some(e) = unserved(b, &model, &req.endpoint).await {
        return bad(e);
    }
    let now = now_ms();
    let row = BatchRow {
        id: new_id("batch_"),
        dialect: "openai".into(),
        endpoint: req.endpoint,
        model,
        input_file_id: file.id,
        completion_window: req.completion_window,
        status: "validating".into(),
        created_ms: now,
        expires_ms: now + window,
        metadata,
        errors: Value::Null,
        options: opts,
        idem_key: key.clone(),
        ..Default::default()
    };
    if let Err(e) = b.db.batch_insert(&row) {
        if crate::store::is_conflict(&e)
            && let Some(k) = &key
            && let Ok(Some(r)) = b.db.batch_by_idem(k)
        {
            return Json(batch_object(b, &r)).into_response();
        }
        return internal(e);
    }
    b.wake.notify_one();
    tracing::info!(batch = %row.id, model = %row.model, endpoint = %row.endpoint, "batch: created");
    Json(batch_object(b, &row)).into_response()
}

#[derive(Deserialize)]
struct ListQuery {
    after: Option<String>,
    limit: Option<usize>,
}

async fn list(State(st): State<Arc<AppState>>, Query(q): Query<ListQuery>) -> Response {
    let b = &st.batches;
    let limit = q.limit.unwrap_or(20).clamp(1, 100);
    match b
        .db
        .batch_list("openai", q.after.as_deref(), None, limit + 1)
    {
        Ok(mut rows) => {
            let has_more = rows.len() > limit;
            rows.truncate(limit);
            Json(page(
                rows.iter().map(|r| batch_object(b, r)).collect(),
                has_more,
            ))
            .into_response()
        }
        Err(e) => internal(e),
    }
}

fn openai_row(b: &Batches, id: &str) -> Result<BatchRow, Box<Response>> {
    match b.db.batch_get(id) {
        Ok(Some(r)) if r.dialect == "openai" => Ok(r),
        Ok(_) => Err(Box::new(not_found("batch"))),
        Err(e) => Err(Box::new(internal(e))),
    }
}

async fn retrieve(State(st): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    match openai_row(&st.batches, &id) {
        Ok(r) => Json(batch_object(&st.batches, &r)).into_response(),
        Err(e) => *e,
    }
}

async fn cancel(State(st): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    let b = &st.batches;
    let row = match openai_row(b, &id) {
        Ok(r) => r,
        Err(e) => return *e,
    };
    if row.is_terminal() {
        return error(
            StatusCode::CONFLICT,
            "invalid_request_error",
            format!("Cannot cancel a batch with status {}", row.status),
        );
    }
    if row.status != "cancelling"
        && let Err(e) = b.db.batch_transition(&id, &row.status, "cancelling")
    {
        return internal(e);
    }
    b.wake.notify_one();
    match openai_row(b, &id) {
        Ok(r) => Json(batch_object(b, &r)).into_response(),
        Err(e) => *e,
    }
}

/// The Batch object, plus the `paddock` progress block.
pub(crate) fn batch_object(b: &Batches, r: &BatchRow) -> Value {
    let ts = |v: Option<i64>| v.map(secs);
    let failed = r.errored + r.canceled + r.expired;
    json!({
        "id": r.id,
        "object": "batch",
        "endpoint": r.endpoint,
        "model": r.model,
        "errors": r.errors,
        "input_file_id": r.input_file_id,
        "completion_window": r.completion_window,
        "status": r.status,
        "output_file_id": r.output_file_id,
        "error_file_id": r.error_file_id,
        "created_at": secs(r.created_ms),
        "in_progress_at": ts(r.in_progress_ms),
        "expires_at": secs(r.expires_ms),
        "finalizing_at": ts(r.finalizing_ms),
        "completed_at": ts(r.completed_ms),
        "failed_at": ts(r.failed_ms),
        "expired_at": ts(r.expired_ms),
        "cancelling_at": ts(r.cancelling_ms),
        "cancelled_at": ts(r.cancelled_ms),
        "request_counts": {"total": r.total, "completed": r.succeeded, "failed": failed},
        "usage": {
            "input_tokens": r.input_tokens,
            "input_tokens_details": {"cached_tokens": r.cached_tokens},
            "output_tokens": r.output_tokens,
            "output_tokens_details": {"reasoning_tokens": r.reasoning_tokens},
            "total_tokens": r.input_tokens + r.output_tokens,
        },
        "metadata": r.metadata,
        "paddock": paddock_block(b, r),
    })
}

/// What the run is doing, plus the counts the vendor objects fold together.
pub(crate) fn paddock_block(b: &Batches, r: &BatchRow) -> Value {
    let mut v = match b.progress_of(&r.id) {
        Some(p) if !r.is_terminal() => serde_json::to_value(p).unwrap_or_else(|_| json!({})),
        _ => json!({"state": if r.is_terminal() { "done" } else { "starting" }}),
    };
    v["succeeded"] = json!(r.succeeded);
    v["errored"] = json!(r.errored);
    v["canceled"] = json!(r.canceled);
    v["expired"] = json!(r.expired);
    if !r.options.as_object().is_none_or(|o| o.is_empty()) {
        v["options"] = r.options.clone();
    }
    v
}
