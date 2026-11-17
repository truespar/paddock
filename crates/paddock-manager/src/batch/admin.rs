//! The Studio's view of every batch on the box - `/api/batches`, behind the
//! admin scope like the rest of `/api`.
//!
//! The vendor routes answer their own dialect and their own caller; the
//! monitor wants all of it in one list: OpenAI and Anthropic batches together,
//! the live progress block, results paged out of the result files while the
//! run is still writing them, and cancel / delete / download. Nothing here
//! changes how a batch runs - it reads the same store rows and the same
//! committed result bytes the vendor routes serve.
//!
//! Results come out of the files, not the store: a finished batch's item rows
//! are purged (a million-request batch would otherwise keep a million rows),
//! so the result lines ARE the record. Paging is by byte offset into the file,
//! up to the committed length - a line past it may be torn, and is never shown.

use std::path::PathBuf;
use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{Value, json};

use super::{Batches, files};
use crate::routes::AppState;
use crate::store::BatchRow;

pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/api/batches", get(list))
        .route("/api/batches/{id}", get(retrieve).delete(delete))
        .route("/api/batches/{id}/cancel", post(cancel))
        .route("/api/batches/{id}/results", get(results))
        .route("/api/batches/{id}/results/line", get(line))
        .route("/api/batches/{id}/files/{which}", get(download))
}

fn error(status: StatusCode, message: impl Into<String>) -> Response {
    let kind = if status.is_server_error() {
        "server_error"
    } else {
        "invalid_request_error"
    };
    (status, Json(paddock_api::ErrorBody::new(kind, message))).into_response()
}

fn internal(e: impl std::fmt::Display) -> Response {
    error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
}

fn row_of(b: &Batches, id: &str) -> Result<BatchRow, Box<Response>> {
    match b.db.batch_get(id) {
        Ok(Some(r)) => Ok(r),
        Ok(None) => Err(Box::new(error(StatusCode::NOT_FOUND, "No such batch"))),
        Err(e) => Err(Box::new(internal(e))),
    }
}

/// One batch for the monitor: both dialects in one shape, millisecond times,
/// the counts unfolded, and the live block while it runs.
pub(crate) fn studio_object(b: &Batches, r: &BatchRow) -> Value {
    let settled = r.succeeded + r.errored + r.canceled + r.expired;
    let ended_ms = [r.completed_ms, r.failed_ms, r.expired_ms, r.cancelled_ms]
        .into_iter()
        .flatten()
        .next();
    let progress = if r.is_terminal() {
        Value::Null
    } else {
        match b.progress_of(&r.id) {
            Some(p) => serde_json::to_value(p).unwrap_or(Value::Null),
            None => json!({"state": "starting"}),
        }
    };
    let input_bytes = b
        .input_path(r)
        .and_then(|p| std::fs::metadata(p).ok())
        .map(|m| m.len());
    json!({
        "id": r.id,
        "dialect": r.dialect,
        "endpoint": r.endpoint,
        "model": r.model,
        "status": r.status,
        "created_ms": r.created_ms,
        "in_progress_ms": r.in_progress_ms,
        "ended_ms": ended_ms,
        "expires_ms": r.expires_ms,
        "completion_window": r.completion_window,
        "counts": {
            "total": r.total,
            "succeeded": r.succeeded,
            "errored": r.errored,
            "canceled": r.canceled,
            "expired": r.expired,
            "pending": (r.total - settled).max(0),
        },
        "usage": {
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "cached_tokens": r.cached_tokens,
            "reasoning_tokens": r.reasoning_tokens,
        },
        "progress": progress,
        "options": r.options,
        "metadata": r.metadata,
        "errors": r.errors,
        "files": {
            "input": input_bytes,
            "output": r.out_len,
            // Anthropic results are one file; OpenAI keeps failures apart
            "errors": (r.dialect == "openai").then_some(r.err_len),
        },
    })
}

#[derive(Deserialize)]
struct ListQuery {
    /// `active` | `ended` | anything else = every batch
    state: Option<String>,
    after: Option<String>,
    limit: Option<usize>,
}

/// `GET /api/batches` - newest first, both dialects.
async fn list(State(st): State<Arc<AppState>>, Query(q): Query<ListQuery>) -> Response {
    let b = &st.batches;
    let limit = q.limit.unwrap_or(50).clamp(1, 200);
    let active = match q.state.as_deref() {
        Some("active") => Some(true),
        Some("ended") => Some(false),
        _ => None,
    };
    match b.db.batch_list_any(q.after.as_deref(), active, limit + 1) {
        Ok(mut rows) => {
            let has_more = rows.len() > limit;
            rows.truncate(limit);
            Json(json!({
                "data": rows.iter().map(|r| studio_object(b, r)).collect::<Vec<_>>(),
                "has_more": has_more,
            }))
            .into_response()
        }
        Err(e) => internal(e),
    }
}

async fn retrieve(State(st): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    match row_of(&st.batches, &id) {
        Ok(r) => Json(studio_object(&st.batches, &r)).into_response(),
        Err(e) => *e,
    }
}

/// `POST /api/batches/{id}/cancel` - either dialect; the run settles what is
/// in flight and marks the rest cancelled, as the vendor cancels do.
async fn cancel(State(st): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    let b = &st.batches;
    let row = match row_of(b, &id) {
        Ok(r) => r,
        Err(e) => return *e,
    };
    if row.is_terminal() {
        return error(
            StatusCode::CONFLICT,
            format!("Cannot cancel a batch with status {}", row.status),
        );
    }
    if row.status != "cancelling"
        && let Err(e) = b.db.batch_transition(&id, &row.status, "cancelling")
    {
        return internal(e);
    }
    b.wake.notify_one();
    match row_of(b, &id) {
        Ok(r) => Json(studio_object(b, &r)).into_response(),
        Err(e) => *e,
    }
}

/// `DELETE /api/batches/{id}` - an ended batch's record and the result files
/// it wrote. An OpenAI input file is the caller's own upload and stays (its
/// `/v1/files` id still works); an Anthropic batch's converted input lives in
/// the batch's directory and goes with it.
async fn delete(State(st): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    let b = &st.batches;
    let row = match row_of(b, &id) {
        Ok(r) => r,
        Err(e) => return *e,
    };
    if !row.is_terminal() {
        return error(
            StatusCode::CONFLICT,
            "the batch is still running - cancel it, and delete it once it has ended",
        );
    }
    if let Err(e) = b.db.batch_delete(&id) {
        return internal(e);
    }
    let _ = tokio::fs::remove_dir_all(b.job_dir(&id)).await;
    Json(json!({"id": id, "deleted": true})).into_response()
}

/// Which results a page reads: the successes or the failures.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Ok,
    Failed,
}

impl Kind {
    fn parse(s: Option<&str>) -> Result<Kind, Box<Response>> {
        match s.unwrap_or("ok") {
            "ok" => Ok(Kind::Ok),
            "failed" => Ok(Kind::Failed),
            other => Err(Box::new(error(
                StatusCode::BAD_REQUEST,
                format!("kind must be ok or failed, not {other:?}"),
            ))),
        }
    }
}

/// The file a kind lives in and how much of it is committed. OpenAI writes
/// failures to their own file; Anthropic writes every result to one, so both
/// kinds read it and the line filter does the split.
fn source(b: &Batches, r: &BatchRow, kind: Kind) -> (PathBuf, u64) {
    let dir = b.job_dir(&r.id);
    if r.dialect == "openai" && kind == Kind::Failed {
        (dir.join("err.jsonl"), r.err_len.max(0) as u64)
    } else {
        (dir.join("out.jsonl"), r.out_len.max(0) as u64)
    }
}

/// A line that belongs to `kind`: an Anthropic line by its result type; in
/// the OpenAI files the file already decided.
fn wanted(r: &BatchRow, kind: Kind, v: &Value) -> bool {
    if r.dialect != "anthropic" {
        return true;
    }
    let ok = v["result"]["type"] == "succeeded";
    ok == (kind == Kind::Ok)
}

/// Bytes a page reads at most before it stops and hands back its cursor -
/// keeps a request bounded whatever the line sizes (a filter that matches
/// nothing would otherwise read a whole multi-gigabyte file in one call).
const PAGE_BYTES: usize = 4 << 20;
/// The longest single result line a page will return (an embeddings batch
/// writes big lines; a line past this is offered as a download instead).
const MAX_LINE: usize = 64 << 20;

#[derive(Deserialize)]
struct ResultsQuery {
    kind: Option<String>,
    cursor: Option<u64>,
    limit: Option<usize>,
}

/// `GET /api/batches/{id}/results?kind=ok|failed&cursor=&limit=` - result
/// summaries in file order. `next_cursor` is where the next page starts;
/// `end` says the committed bytes ran out (a running batch adds more later).
async fn results(
    State(st): State<Arc<AppState>>,
    Path(id): Path<String>,
    Query(q): Query<ResultsQuery>,
) -> Response {
    let b = &st.batches;
    let row = match row_of(b, &id) {
        Ok(r) => r,
        Err(e) => return *e,
    };
    let kind = match Kind::parse(q.kind.as_deref()) {
        Ok(k) => k,
        Err(e) => return *e,
    };
    let limit = q.limit.unwrap_or(50).clamp(1, 500);
    let (path, committed) = source(b, &row, kind);
    let start = q.cursor.unwrap_or(0).min(committed);
    let page = tokio::task::spawn_blocking(move || read_page(&path, committed, start, limit));
    let (lines, next) = match page.await {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => return internal(e),
        Err(e) => return internal(e),
    };
    let items: Vec<Value> = lines
        .iter()
        .filter(|(_, v)| wanted(&row, kind, v))
        .map(|(off, v)| summary(&row, *off, v))
        .collect();
    Json(json!({
        "items": items,
        "next_cursor": next,
        "end": next >= committed,
    }))
    .into_response()
}

/// Up to `limit` whole lines from `start`, within `committed`, reading at
/// most [`PAGE_BYTES`] past the first line: (line offset, parsed line) pairs
/// and the offset after the last one consumed.
fn read_page(
    path: &std::path::Path,
    committed: u64,
    start: u64,
    limit: usize,
) -> std::io::Result<(Vec<(u64, Value)>, u64)> {
    let mut out = Vec::new();
    let mut pos = start;
    let mut read = 0usize;
    while pos < committed && out.len() < limit && read < PAGE_BYTES {
        let (line, next) = read_line(path, committed, pos)?;
        read += (next - pos) as usize;
        // a line that is not JSON is skipped, never fatal: the page still
        // shows everything around it
        if let Ok(v) = serde_json::from_slice::<Value>(&line) {
            out.push((pos, v));
        }
        pos = next;
    }
    Ok((out, pos))
}

/// The whole line starting at `off` (newline excluded) and the offset after
/// it, growing the read until the newline shows or the committed bytes end.
fn read_line(path: &std::path::Path, committed: u64, off: u64) -> std::io::Result<(Vec<u8>, u64)> {
    let mut want = 64usize << 10;
    loop {
        let len = want.min((committed - off) as usize);
        let buf = files::read_at(path, off, len)?;
        if let Some(i) = buf.iter().position(|&c| c == b'\n') {
            let mut line = buf;
            line.truncate(i);
            return Ok((line, off + i as u64 + 1));
        }
        if off + len as u64 >= committed {
            // the committed bytes end on a newline (group commit), so this is
            // the file's last line without one - take it whole
            return Ok((buf, committed));
        }
        if want >= MAX_LINE {
            return Err(std::io::Error::other(format!(
                "a result line at byte {off} is longer than {} MiB - download the file",
                MAX_LINE >> 20
            )));
        }
        want = (want * 4).min(MAX_LINE);
    }
}

/// Text a preview shows at most.
const PREVIEW: usize = 600;

/// One result line as the monitor's row: who, whether it worked, what came
/// back (the first characters of the answer, or the error), and its tokens.
fn summary(r: &BatchRow, off: u64, v: &Value) -> Value {
    let custom_id = v["custom_id"].as_str().unwrap_or("");
    let (ok, status, body, err) = if r.dialect == "anthropic" {
        let res = &v["result"];
        let t = res["type"].as_str().unwrap_or("errored");
        let err = match t {
            "succeeded" => None,
            "errored" => Some(
                res["error"]["error"]["message"]
                    .as_str()
                    .unwrap_or("errored")
                    .to_owned(),
            ),
            other => Some(other.to_owned()),
        };
        (t == "succeeded", json!(t), &res["message"], err)
    } else {
        let resp = &v["response"];
        let code = resp["status_code"].as_u64();
        let ok = code.is_some_and(|c| (200..300).contains(&c));
        let err = if ok {
            None
        } else {
            Some(
                resp["body"]["error"]["message"]
                    .as_str()
                    .or_else(|| v["error"]["message"].as_str())
                    .or_else(|| v["error"]["code"].as_str())
                    .unwrap_or("failed")
                    .to_owned(),
            )
        };
        (ok, json!(code), &resp["body"], err)
    };
    let u = &body["usage"];
    let n = |v: &Value| v.as_i64().unwrap_or(0);
    json!({
        "offset": off,
        "custom_id": custom_id,
        "ok": ok,
        "status": status,
        "preview": if ok { preview(body) } else { Value::Null },
        "error": err,
        "input_tokens": n(&u["prompt_tokens"]) + n(&u["input_tokens"]),
        "output_tokens": n(&u["completion_tokens"]) + n(&u["output_tokens"]),
    })
}

/// The answer's opening text, whatever endpoint the batch called: chat
/// completions, completions, responses, messages, or an embedding's shape.
fn preview(body: &Value) -> Value {
    let mut text = String::new();
    let mut push = |s: &str| {
        if !text.is_empty() && !s.is_empty() {
            text.push('\n');
        }
        text.push_str(s);
    };
    if let Some(choice) = body["choices"].get(0) {
        let msg = &choice["message"];
        match &msg["content"] {
            Value::String(s) => push(s),
            Value::Array(parts) => parts
                .iter()
                .filter_map(|p| p["text"].as_str())
                .for_each(&mut push),
            _ => {}
        }
        if let Some(calls) = msg["tool_calls"].as_array() {
            for c in calls {
                let f = &c["function"];
                push(&format!(
                    "{}({})",
                    f["name"].as_str().unwrap_or("tool"),
                    f["arguments"].as_str().unwrap_or("")
                ));
            }
        }
        if let Some(s) = choice["text"].as_str() {
            push(s);
        }
    } else if let Some(s) = body["output_text"].as_str() {
        push(s);
    } else if let Some(items) = body["output"].as_array() {
        for item in items {
            if let Some(parts) = item["content"].as_array() {
                parts
                    .iter()
                    .filter(|p| p["type"] == "output_text")
                    .filter_map(|p| p["text"].as_str())
                    .for_each(&mut push);
            }
        }
    } else if let Some(blocks) = body["content"].as_array() {
        blocks
            .iter()
            .filter_map(|b| b["text"].as_str())
            .for_each(&mut push);
    } else if let Some(data) = body["data"].as_array() {
        let dims = data
            .first()
            .and_then(|d| d["embedding"].as_array())
            .map_or(0, Vec::len);
        push(&format!("{} embedding(s), {dims} dimensions", data.len()));
    }
    if text.chars().count() > PREVIEW {
        text = text.chars().take(PREVIEW).collect::<String>() + "...";
    }
    json!(text)
}

#[derive(Deserialize)]
struct LineQuery {
    kind: Option<String>,
    offset: u64,
}

/// `GET /api/batches/{id}/results/line?kind=&offset=` - one result line in
/// full, for the row a person opened. The offset must be one a page returned.
async fn line(
    State(st): State<Arc<AppState>>,
    Path(id): Path<String>,
    Query(q): Query<LineQuery>,
) -> Response {
    let b = &st.batches;
    let row = match row_of(b, &id) {
        Ok(r) => r,
        Err(e) => return *e,
    };
    let kind = match Kind::parse(q.kind.as_deref()) {
        Ok(k) => k,
        Err(e) => return *e,
    };
    let (path, committed) = source(b, &row, kind);
    if q.offset >= committed {
        return error(
            StatusCode::BAD_REQUEST,
            "offset is past the committed results",
        );
    }
    let off = q.offset;
    let got = tokio::task::spawn_blocking(move || read_line(&path, committed, off)).await;
    match got {
        Ok(Ok((bytes, _))) => match serde_json::from_slice::<Value>(&bytes) {
            Ok(v) => Json(v).into_response(),
            Err(_) => error(
                StatusCode::BAD_REQUEST,
                "offset is not the start of a result line",
            ),
        },
        Ok(Err(e)) => internal(e),
        Err(e) => internal(e),
    }
}

/// `GET /api/batches/{id}/files/{input|output|errors}` - the JSONL to keep.
/// Result files stream up to their committed length, so a running batch's
/// download is everything settled so far and never a torn line.
async fn download(
    State(st): State<Arc<AppState>>,
    Path((id, which)): Path<(String, String)>,
) -> Response {
    let b = &st.batches;
    let row = match row_of(b, &id) {
        Ok(r) => r,
        Err(e) => return *e,
    };
    let (path, len) = match which.as_str() {
        "input" => match b.input_path(&row) {
            Some(p) => match tokio::fs::metadata(&p).await {
                Ok(m) => (p, m.len()),
                Err(_) => return error(StatusCode::NOT_FOUND, "the input file is gone"),
            },
            None => return error(StatusCode::NOT_FOUND, "the input file is gone"),
        },
        "output" => source(b, &row, Kind::Ok),
        "errors" if row.dialect == "openai" => source(b, &row, Kind::Failed),
        _ => return error(StatusCode::NOT_FOUND, "files are input, output or errors"),
    };
    match files::body_of(&path, len).await {
        Ok(body) => (
            [
                (header::CONTENT_TYPE, "application/x-jsonl".to_owned()),
                (
                    header::CONTENT_DISPOSITION,
                    format!("attachment; filename=\"{id}_{which}.jsonl\""),
                ),
            ],
            body,
        )
            .into_response(),
        // nothing written yet: an empty file, not an error
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => (
            [(header::CONTENT_TYPE, "application/x-jsonl")],
            String::new(),
        )
            .into_response(),
        Err(e) => internal(e),
    }
}
