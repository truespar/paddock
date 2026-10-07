//! `/api/*` - the Studio's own surface over the SQLite store: conversations,
//! prompts, settings, and API keys. (Inference stays on `/v1/*`.) Errors are
//! OpenAI-shaped so one client error handler covers everything.

use std::sync::Arc;

use axum::Router;
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{delete, get, put};
use paddock_api::ErrorBody;
use serde_json::{Value, json};

use crate::routes::AppState;

#[path = "api_mask_history.rs"]
mod mask_history;
#[path = "api_table_history.rs"]
mod table_history;

fn err500(e: impl std::fmt::Display) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorBody::new("internal_error", e.to_string())),
    )
        .into_response()
}

fn errx(status: StatusCode, kind: &str, msg: impl std::fmt::Display) -> Response {
    (status, Json(ErrorBody::new(kind, msg.to_string()))).into_response()
}

pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .merge(table_history::routes())
        .merge(mask_history::routes())
        .route("/api/conversations", get(list_conversations))
        .route(
            "/api/conversations/{id}",
            get(get_conversation)
                .put(put_conversation)
                .delete(delete_conversation),
        )
        .route("/api/prompts", get(list_prompts).post(create_prompt))
        .route(
            "/api/prompts/{id}",
            get(get_prompt).put(update_prompt).delete(delete_prompt),
        )
        // the Reads page's saved question sets - the prompt library's shape
        .route("/api/reads", get(list_read_sets).post(create_read_set))
        .route(
            "/api/read-runs/{scope}",
            get(list_read_runs)
                .post(save_read_run)
                .delete(clear_read_runs),
        )
        .route(
            "/api/reads/{id}",
            get(get_read_set)
                .put(update_read_set)
                .delete(delete_read_set),
        )
        // the Reads page's earlier reads - the conversation list's shape
        .route("/api/read-history", get(list_read_history))
        .route(
            "/api/read-history/{id}",
            get(get_read).put(put_read).delete(delete_read),
        )
        .route("/api/settings", get(get_settings).put(put_settings))
        .route("/api/settings/import", put(import_settings))
        // the Hugging Face token gated downloads carry (registry/gated.rs) -
        // its own surface: written here, never read back out
        .route(
            "/api/huggingface/token",
            get(get_hf_token).put(put_hf_token).delete(delete_hf_token),
        )
        .route("/api/export", get(export_db))
        .route(
            "/api/attachments/{id}",
            axum::routing::put(put_attachment).get(get_attachment),
        )
        .route(
            "/api/attachments/{id}/metadata",
            get(attachment_metadata).post(store_attachment_metadata),
        )
        .route("/api/attachments/{id}/rendition", get(attachment_rendition))
        .route("/api/keys", get(list_keys).post(create_key))
        .route("/api/keys/{id}", delete(revoke_key))
        // No /api/mcp or /api/search here: web search and MCP
        // servers are MODEL configuration - they live in each endpoint's
        // servers/<port>.toml and travel as launch config. The manager
        // stores none of it. (MCP tool APPROVALS remain a runner endpoint -
        // the parked agent loop is in-process state there; the Studio
        // resolves them as an API client via the runner relay.)
        // Attachment view-copies (and image-bearing conversation docs) exceed the
        // 2 MB default body limit - raise it to the attachment cap for /api.
        .layer(axum::extract::DefaultBodyLimit::max(MAX_ATTACHMENT))
}

// ── conversations ────────────────────────────────────────────────────────────

async fn list_conversations(State(s): State<Arc<AppState>>) -> Response {
    match s.db.list_conversations() {
        Ok(v) => Json(v).into_response(),
        Err(e) => err500(e),
    }
}

async fn get_conversation(State(s): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    match s.db.get_conversation(&id) {
        Ok(Some(v)) => Json(v).into_response(),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(ErrorBody::not_found(format!("conversation {id}"))),
        )
            .into_response(),
        Err(e) => err500(e),
    }
}

async fn put_conversation(
    State(s): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(mut doc): Json<Value>,
) -> Response {
    doc["id"] = Value::String(id); // the path is authoritative
    match s.db.put_conversation(&doc) {
        Ok(()) => Json(json!({ "ok": true })).into_response(),
        Err(e) => err500(e),
    }
}

async fn delete_conversation(State(s): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    match s.db.delete_conversation(&id) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => err500(e),
    }
}

// ── prompts ──────────────────────────────────────────────────────────────────

async fn list_prompts(State(s): State<Arc<AppState>>) -> Response {
    match s.db.list_prompts() {
        Ok(v) => Json(v).into_response(),
        Err(e) => err500(e),
    }
}

async fn get_prompt(State(s): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    match s.db.list_prompts() {
        Ok(v) => match v
            .into_iter()
            .find(|p| p.get("id").and_then(Value::as_str) == Some(id.as_str()))
        {
            Some(p) => Json(p).into_response(),
            None => (
                StatusCode::NOT_FOUND,
                Json(ErrorBody::not_found(format!("prompt {id}"))),
            )
                .into_response(),
        },
        Err(e) => err500(e),
    }
}

async fn create_prompt(State(s): State<Arc<AppState>>, Json(doc): Json<Value>) -> Response {
    match s.db.put_prompt(&doc) {
        Ok(prompt) => Json(json!({ "ok": true, "prompt": prompt })).into_response(),
        Err(e) => prompt_error(e),
    }
}

async fn update_prompt(
    State(s): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(mut doc): Json<Value>,
) -> Response {
    doc["id"] = Value::String(id);
    match s.db.put_prompt(&doc) {
        Ok(prompt) => Json(json!({ "ok": true, "prompt": prompt })).into_response(),
        Err(e) => prompt_error(e),
    }
}

#[derive(serde::Deserialize)]
struct PromptRevision {
    revision: Option<String>,
}

fn prompt_error(e: crate::store::StoreError) -> Response {
    match e {
        crate::store::StoreError::Conflict(m) => errx(StatusCode::CONFLICT, "conflict", m),
        crate::store::StoreError::Bad(m) => errx(StatusCode::BAD_REQUEST, "invalid_request", m),
        e => err500(e),
    }
}

async fn delete_prompt(
    State(s): State<Arc<AppState>>,
    Path(id): Path<String>,
    Query(q): Query<PromptRevision>,
) -> Response {
    match s.db.delete_prompt_checked(&id, q.revision.as_deref()) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => prompt_error(e),
    }
}

// ── read sets ────────────────────────────────────────────────────────────────

async fn list_read_runs(State(s): State<Arc<AppState>>, Path(scope): Path<String>) -> Response {
    match s.db.read_runs(&scope) {
        Ok(v) => Json(v).into_response(),
        Err(e) => prompt_error(e),
    }
}
async fn save_read_run(
    State(s): State<Arc<AppState>>,
    Path(scope): Path<String>,
    Json(doc): Json<Value>,
) -> Response {
    match s.db.save_read_run(&scope, &doc) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => prompt_error(e),
    }
}
async fn clear_read_runs(State(s): State<Arc<AppState>>, Path(scope): Path<String>) -> Response {
    match s.db.clear_read_runs(&scope) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => prompt_error(e),
    }
}

async fn list_read_sets(State(s): State<Arc<AppState>>) -> Response {
    match s.db.list_read_sets() {
        Ok(v) => Json(v).into_response(),
        Err(e) => err500(e),
    }
}

async fn get_read_set(State(s): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    match s.db.list_read_sets() {
        Ok(v) => match v
            .into_iter()
            .find(|p| p.get("id").and_then(Value::as_str) == Some(id.as_str()))
        {
            Some(p) => Json(p).into_response(),
            None => (
                StatusCode::NOT_FOUND,
                Json(ErrorBody::not_found(format!("read set {id}"))),
            )
                .into_response(),
        },
        Err(e) => err500(e),
    }
}

async fn create_read_set(State(s): State<Arc<AppState>>, Json(doc): Json<Value>) -> Response {
    match s.db.put_read_set(&doc) {
        Ok(set) => Json(json!({ "ok": true, "set": set })).into_response(),
        Err(e) => prompt_error(e),
    }
}

async fn update_read_set(
    State(s): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(mut doc): Json<Value>,
) -> Response {
    doc["id"] = Value::String(id);
    match s.db.put_read_set(&doc) {
        Ok(set) => Json(json!({ "ok": true, "set": set })).into_response(),
        Err(e) => prompt_error(e),
    }
}

async fn delete_read_set(
    State(s): State<Arc<AppState>>,
    Path(id): Path<String>,
    Query(q): Query<PromptRevision>,
) -> Response {
    match s.db.delete_read_set_checked(&id, q.revision.as_deref()) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => prompt_error(e),
    }
}

async fn list_read_history(State(s): State<Arc<AppState>>) -> Response {
    match s.db.list_read_history() {
        Ok(v) => Json(v).into_response(),
        Err(e) => err500(e),
    }
}

#[derive(serde::Deserialize, Default)]
struct ReadHistoryQuery {
    #[serde(default)]
    envelope: bool,
    revision: Option<String>,
}

async fn get_read(
    State(s): State<Arc<AppState>>,
    Path(id): Path<String>,
    Query(q): Query<ReadHistoryQuery>,
) -> Response {
    if q.envelope {
        return match s.db.read_snapshot(&id) {
            Ok(Some(doc)) => Json(doc).into_response(),
            Ok(None) => (
                StatusCode::NOT_FOUND,
                Json(ErrorBody::not_found(format!("read {id}"))),
            )
                .into_response(),
            Err(e) => err500(e),
        };
    }
    match s.db.get_read(&id) {
        // the saved text as it was sent - see store/read_history.rs on why a
        // read never passes through Value on the way out
        Ok(Some(text)) => (
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            text,
        )
            .into_response(),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(ErrorBody::not_found(format!("read {id}"))),
        )
            .into_response(),
        Err(e) => err500(e),
    }
}

async fn put_read(
    State(s): State<Arc<AppState>>,
    Path(id): Path<String>,
    Query(q): Query<ReadHistoryQuery>,
    text: String,
) -> Response {
    let text = if q.envelope {
        match serde_json::from_str::<Value>(&text)
            .ok()
            .and_then(|v| v["doc"].as_str().map(str::to_owned))
        {
            Some(text) => text,
            None => {
                return prompt_error(crate::store::StoreError::Bad(
                    "Expected a read document".into(),
                ));
            }
        }
    } else {
        text
    };
    // kept as the text it arrived as, so the questions keep their order
    match s.db.put_read_checked(&id, &text, q.revision.as_deref()) {
        Ok(read) => Json(json!({ "ok": true, "read": read })).into_response(),
        Err(e) => prompt_error(e),
    }
}

async fn delete_read(
    State(s): State<Arc<AppState>>,
    Path(id): Path<String>,
    Query(q): Query<ReadHistoryQuery>,
) -> Response {
    match s.db.delete_read_checked(&id, q.revision.as_deref()) {
        Ok(_) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => prompt_error(e),
    }
}

// ── settings ─────────────────────────────────────────────────────────────────

async fn get_settings(State(s): State<Arc<AppState>>) -> Response {
    match s.db.all_settings() {
        Ok(v) => Json(v).into_response(),
        Err(e) => err500(e),
    }
}

/// Merge the posted object into settings (each key upserts).
async fn put_settings(State(s): State<Arc<AppState>>, Json(obj): Json<Value>) -> Response {
    if let Err(e) = s.db.patch_settings(&obj, false) {
        return prompt_error(e);
    }
    match s.db.all_settings() {
        Ok(v) => Json(v).into_response(),
        Err(e) => err500(e),
    }
}

async fn import_settings(State(s): State<Arc<AppState>>, Json(obj): Json<Value>) -> Response {
    if let Err(e) = s.db.patch_settings(&obj, true) {
        return prompt_error(e);
    }
    get_settings(State(s)).await
}

// ── the Hugging Face token ───────────────────────────────────────────────────

/// What the Settings card shows: whether a token is in reach and where it
/// came from. The token itself never leaves the manager.
fn hf_token_status(s: &AppState) -> Value {
    match crate::registry::huggingface_token(Some(&s.db)) {
        Some((_, source)) => json!({ "configured": true, "source": source }),
        None => json!({ "configured": false, "source": null }),
    }
}

async fn get_hf_token(State(s): State<Arc<AppState>>) -> Response {
    Json(hf_token_status(&s)).into_response()
}

/// Ask Hugging Face who a token belongs to. `Ok(None)` = it refused the
/// token; `Err` = no answer (offline), which does not block saving.
async fn hf_whoami(token: &str) -> Result<Option<String>, String> {
    let resp = reqwest::Client::new()
        .get("https://huggingface.co/api/whoami-v2")
        .bearer_auth(token)
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
        .map_err(|e| e.without_url().to_string())?;
    if matches!(resp.status().as_u16(), 401 | 403) {
        return Ok(None);
    }
    if !resp.status().is_success() {
        return Err(format!("status {}", resp.status().as_u16()));
    }
    let body: Value = resp.json().await.map_err(|e| e.without_url().to_string())?;
    Ok(Some(
        body.get("name")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned(),
    ))
}

/// Save a token. Hugging Face is asked who it belongs to first: a token it
/// refuses is not saved; no answer at all (offline) saves it unverified.
async fn put_hf_token(State(s): State<Arc<AppState>>, Json(body): Json<Value>) -> Response {
    let token = body
        .get("token")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or("");
    if !crate::registry::valid_token(token) {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorBody::new(
                "invalid_request_error",
                "that is not a Hugging Face token - copy one from huggingface.co/settings/tokens",
            )),
        )
            .into_response();
    }
    let user = match hf_whoami(token).await {
        Ok(Some(user)) => Some(user),
        Ok(None) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorBody::new(
                    "invalid_request_error",
                    "Hugging Face does not accept this token - create a new one at \
                     huggingface.co/settings/tokens",
                )),
            )
                .into_response();
        }
        Err(e) => {
            tracing::warn!(error = %e, "could not reach Hugging Face to check the token; saving it unverified");
            None
        }
    };
    if let Err(e) = s.db.set_huggingface_token(Some(token)) {
        return err500(e);
    }
    let mut out = hf_token_status(&s);
    out["user"] = user.map_or(Value::Null, Value::String);
    Json(out).into_response()
}

async fn delete_hf_token(State(s): State<Arc<AppState>>) -> Response {
    if let Err(e) = s.db.set_huggingface_token(None) {
        return err500(e);
    }
    Json(hf_token_status(&s)).into_response()
}

// NOTE: there is deliberately no web-search or MCP section here.
// Both are MODEL configuration - per-endpoint, in servers/<port>.toml, edited
// on the Start/Edit page and served by the runner with zero manager linkage.

// ── api keys ─────────────────────────────────────────────────────────────────

async fn list_keys(State(s): State<Arc<AppState>>) -> Response {
    match s.db.list_api_keys() {
        Ok(v) => Json(v).into_response(),
        Err(e) => err500(e),
    }
}

/// Create a key; the plaintext is returned once in `key` and never stored.
async fn create_key(State(s): State<Arc<AppState>>, Json(body): Json<Value>) -> Response {
    let name = body.get("name").and_then(Value::as_str).unwrap_or("key");
    match s.db.create_api_key(name) {
        Ok((mut record, key)) => {
            record["key"] = Value::String(key);
            (StatusCode::CREATED, Json(record)).into_response()
        }
        Err(e) => err500(e),
    }
}

async fn revoke_key(State(s): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    match s.db.revoke_api_key(&id) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => err500(e),
    }
}

/// Query metadata for an attachment upload (bytes ride in the body).
#[derive(serde::Deserialize)]
struct AttachMeta {
    name: Option<String>,
    w: Option<i64>,
    h: Option<i64>,
    conv: Option<String>,
}

/// Max attachment upload - mirrors the client's MAX_FILE_MB (100 MB).
const MAX_ATTACHMENT: usize = 100 * 1024 * 1024;

/// `PUT /api/attachments/{id}` - store an attachment's ("view") bytes. The id is
/// client-supplied (a uuid) so the message can reference it before the save
/// round-trip; the mime rides in Content-Type, other metadata in the query.
async fn put_attachment(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Query(q): Query<AttachMeta>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    if body.len() > MAX_ATTACHMENT {
        return (StatusCode::PAYLOAD_TOO_LARGE, "attachment too large").into_response();
    }
    let mime = headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/octet-stream")
        .to_owned();
    match state.db.put_attachment(
        &id,
        q.conv.as_deref(),
        &mime,
        q.name.as_deref().unwrap_or(""),
        q.w,
        q.h,
        &body,
    ) {
        Ok(()) => Json(serde_json::json!({ "id": id })).into_response(),
        Err(e) => {
            tracing::error!(%e, "attachment store failed");
            (StatusCode::INTERNAL_SERVER_ERROR, "store failed").into_response()
        }
    }
}

/// `GET /api/attachments/{id}` - stream the stored bytes (for the modal viewer
/// + thumbnails), typed by the stored mime and cached hard (bytes never change).
async fn get_attachment(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    match state.db.get_attachment(&id) {
        Ok(Some((mime, bytes))) => axum::response::Response::builder()
            .header(axum::http::header::CONTENT_TYPE, mime)
            .header(
                axum::http::header::CACHE_CONTROL,
                "private, max-age=31536000, immutable",
            )
            .body(axum::body::Body::from(bytes))
            .expect("valid response"),
        Ok(None) => (StatusCode::NOT_FOUND, "not found").into_response(),
        Err(e) => {
            tracing::error!(%e, "attachment fetch failed");
            (StatusCode::INTERNAL_SERVER_ERROR, "fetch failed").into_response()
        }
    }
}

/// The metadata answer: what the file is, plus what it says about itself.
/// The stored identity rides along so one call answers the whole question.
#[derive(serde::Serialize)]
struct AttachmentMetadata {
    id: String,
    /// As uploaded. Empty for a microphone recording, which has no file name.
    name: String,
    /// The browser's guess from the extension - `format` inside the flattened
    /// metadata is what the BYTES say, and the two disagreeing is itself
    /// worth seeing.
    mime: String,
    size: usize,
    #[serde(flatten)]
    meta: paddock_filemeta::FileMetadata,
}

/// `GET /api/attachments/{id}/metadata` - everything the stored file says
/// about itself: EXIF/XMP/IPTC/ICC/GPS for photos, the Info dict for PDFs,
/// core/app/custom properties for Office packages.
///
/// The manager answers this itself rather than forwarding to a
/// runner's `/api/extract`. A photo's capture time has nothing to do with
/// which model is loaded, and gating it on a live GPU server meant it was
/// unavailable with nothing running, unavailable on a cloud-model chat, and
/// re-uploaded whole over HTTP on every open.
///
/// Nothing is CACHED and nothing is stored. Measured before deciding: tags
/// off a 22.7 MB PDF take ~21 ms including a ~31 ms process-spawn baseline -
/// the parse is below the noise floor of the process that runs it. A cache
/// would buy nothing and would leave users looking at stale metadata after a
/// future extractor fix. If a pathological file ever proves slow, cache it
/// then, keyed on content hash plus an extractor version.
async fn attachment_metadata(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Response {
    // Identity + byte size + the stored metadata JSON, without materializing the
    // blob. The runner ships full metadata during a chat turn and it is written
    // through (POST below); this serves that cache so the common path never
    // re-parses - the metadata twin of the forensics store.
    let (mime, name, size, stored) = match state.db.attachment_lite(&id) {
        Ok(Some(v)) => v,
        Ok(None) => {
            return (
                StatusCode::NOT_FOUND,
                Json(ErrorBody::not_found(format!("attachment {id}"))),
            )
                .into_response();
        }
        Err(e) => return err500(e),
    };
    // Cache hit: return the stored file-metadata JSON with the identity merged
    // in (the AttachmentMetadata shape flattens `meta`, so this reproduces it).
    if let Some(meta_json) = stored
        && let Ok(Value::Object(mut m)) = serde_json::from_str::<Value>(&meta_json)
    {
        m.insert("id".into(), json!(id));
        m.insert("name".into(), json!(name));
        m.insert("mime".into(), json!(mime));
        m.insert("size".into(), json!(size));
        return Json(Value::Object(m)).into_response();
    }
    // Unparseable stored blob (shouldn't happen): fall through to re-read.
    // Miss (never chatted, or a pre-existing attachment): read from the bytes,
    // return, and backfill the cache so nothing runner-independent re-parses.
    let bytes = match state.db.get_attachment(&id) {
        Ok(Some((_, b))) => b,
        Ok(None) => {
            return (
                StatusCode::NOT_FOUND,
                Json(ErrorBody::not_found(format!("attachment {id}"))),
            )
                .into_response();
        }
        Err(e) => return err500(e),
    };
    // Parsing a document is blocking work: off the executor, and its own task
    // so a parser panic on hostile bytes lands as a 500 instead of taking the
    // manager down.
    let meta = match tokio::task::spawn_blocking(move || paddock_filemeta::read(&bytes)).await {
        Ok(m) => m,
        Err(e) => {
            tracing::error!(%e, %id, "file metadata read panicked");
            return errx(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                "could not read this file's metadata",
            );
        }
    };
    if let Ok(js) = serde_json::to_string(&meta) {
        let _ = state.db.set_attachment_metadata(&id, &js);
    }
    Json(AttachmentMetadata {
        id,
        name,
        mime,
        size: size as usize,
        meta,
    })
    .into_response()
}

/// `POST /api/attachments/{id}/metadata` - write through the full file metadata
/// the runner shipped in a chat turn. Body is the runner item's `meta` object
/// (or `{ "meta": {...} }`). The bytes are immutable, so this is idempotent.
async fn store_attachment_metadata(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    let meta = body.get("meta").cloned().unwrap_or(body);
    if !meta.is_object() {
        return errx(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "metadata must be an object",
        );
    }
    match state.db.set_attachment_metadata(&id, &meta.to_string()) {
        Ok(()) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => err500(e),
    }
}

#[derive(serde::Deserialize)]
struct RenditionQuery {
    /// Longest edge, in pixels.
    max: Option<u32>,
}

/// Default longest edge: big enough for the document side panel on a normal
/// display, small enough that the JPEG is a fraction of the original.
const RENDITION_DEFAULT: u32 = 1600;
/// Past this the rendition stops being a viewing copy and becomes a second
/// copy of the photo. Anyone who wants the real thing has `GET /api/
/// attachments/{id}`, which serves the untouched original.
const RENDITION_MAX: u32 = 4096;

/// `GET /api/attachments/{id}/rendition?max=N` - a viewable JPEG of a photo
/// this browser cannot decode itself.
///
/// HEIC is why this exists. It is HEVC, which sits in a patent pool, so no
/// browser but Safari will touch it - while an iPhone writes it by default.
/// AVIF is served here too even though every current browser decodes it: it
/// costs nothing, and it covers the older ones.
///
/// What this is not: a modification. The stored bytes are never touched, and
/// `/metadata` keeps reading the original - a re-encode is a viewing copy and
/// never the record. That separation matters: deriving the stored copy from a
/// canvas silently throws away every photo tag the user had.
///
/// No EXIF ROTATION here, deliberately. paddock-heif applies the container's
/// own irot/imir transforms, so the pixels arrive upright; applying an EXIF
/// Orientation tag on top of that would rotate a portrait photo twice.
/// Different from the JPEG path, and worth stating rather than discovering.
async fn attachment_rendition(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Query(q): Query<RenditionQuery>,
) -> Response {
    let bytes = match state.db.get_attachment(&id) {
        Ok(Some((_mime, b))) => b,
        Ok(None) => {
            return errx(
                StatusCode::NOT_FOUND,
                "not_found",
                format!("attachment {id}"),
            );
        }
        Err(e) => return err500(e),
    };

    // Sniff before deciding anything: the stored mime came from the browser,
    // and a browser leaves `File.type` empty for .HEIC often enough that it is
    // exactly the field we cannot trust here.
    let Some(codec) = paddock_heif::sniff(&bytes) else {
        return errx(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "invalid_request_error",
            "this attachment is not a HEIC or AVIF photo - fetch it directly instead",
        );
    };
    let max = q.max.unwrap_or(RENDITION_DEFAULT).clamp(16, RENDITION_MAX);

    // Decode + resize + re-encode is CPU work measured in tens to hundreds of
    // milliseconds on a full-size photo, and a panic on hostile bytes must land
    // as a 500 rather than take the manager down with it.
    let out = tokio::task::spawn_blocking(move || render_heif(&bytes, max)).await;
    match out {
        Ok(Ok(jpeg)) => axum::response::Response::builder()
            .header(axum::http::header::CONTENT_TYPE, "image/jpeg")
            // The stored bytes never change and `max` is in the URL, so the
            // response is fully determined by the request.
            .header(
                axum::http::header::CACHE_CONTROL,
                "private, max-age=31536000, immutable",
            )
            .body(axum::body::Body::from(jpeg))
            .expect("valid response"),
        // HEIC, permanently - not a missing install. HEVC has no decoder that
        // can be embedded in a closed binary, so the message
        // must not imply there is something to add. 501 is still the right
        // code: the server does not implement this, and no request will change
        // that.
        Ok(Err(paddock_heif::Error::NoDecoder { codec })) => errx(
            StatusCode::NOT_IMPLEMENTED,
            "not_implemented",
            format!(
                "{} photos use HEVC, which Paddock cannot decode. The file is stored \
                 intact and its details are readable - only the preview is unavailable.",
                codec.label()
            ),
        ),
        Ok(Err(e)) => errx(
            StatusCode::UNPROCESSABLE_ENTITY,
            "invalid_request_error",
            format!("could not read this {} photo: {e}", codec.label()),
        ),
        Err(e) => {
            tracing::error!(%e, %id, "rendition panicked");
            errx(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                "could not render this photo",
            )
        }
    }
}

/// Decode, fit inside `max` on the longest edge, encode JPEG. Blocking.
fn render_heif(bytes: &[u8], max: u32) -> Result<Vec<u8>, paddock_heif::Error> {
    let r = paddock_heif::decode(bytes)?;
    let img = image::RgbImage::from_raw(r.width, r.height, r.rgb)
        .ok_or_else(|| paddock_heif::Error::Decode("decoded plane is the wrong size".into()))?;

    // Only ever down. A 400px photo asked for at max=1600 stays 400px rather
    // than becoming a blurry 1600, which is what a naive `resize` would do.
    let long = img.width().max(img.height());
    let img = if long > max {
        let scale = f64::from(max) / f64::from(long);
        let w = ((f64::from(img.width()) * scale).round() as u32).max(1);
        let h = ((f64::from(img.height()) * scale).round() as u32).max(1);
        // Lanczos3 because this is a photograph being shrunk a long way, where
        // the cheaper filters alias visibly on fine detail. It is the expensive
        // one, and it is paid once per (attachment, max).
        image::imageops::resize(&img, w, h, image::imageops::FilterType::Lanczos3)
    } else {
        img
    };

    let mut jpeg = Vec::new();
    // 85 is the usual "no visible loss at normal viewing" point for photos, and
    // roughly a tenth the bytes of the decoded plane.
    image::codecs::jpeg::JpegEncoder::new_with_quality(std::io::Cursor::new(&mut jpeg), 85)
        .encode(
            img.as_raw(),
            img.width(),
            img.height(),
            image::ExtendedColorType::Rgb8,
        )
        .map_err(|e| paddock_heif::Error::Decode(format!("could not encode a JPEG: {e}")))?;
    Ok(jpeg)
}

/// `GET /api/export` - a SANITIZED SQLite copy of the store for download. We
/// snapshot the DB (`VACUUM INTO`), then strip every credential: the
/// `api_keys` rows, plus the legacy model-config leftovers (mcp_servers
/// table, settings.web_search) that Store::open already purges - kept here
/// too as belt-and-braces for a snapshot of any DB. The result is safe to
/// share - conversations, prompts, settings and run-metrics survive; secrets
/// do not.
async fn export_db(State(state): State<Arc<AppState>>) -> Response {
    let tmp = match state.db.snapshot_to_temp() {
        Ok(p) => p,
        Err(e) => {
            tracing::error!(%e, "db export snapshot failed");
            return (StatusCode::INTERNAL_SERVER_ERROR, "export failed").into_response();
        }
    };
    let result = sanitize_export(&tmp);
    let _ = std::fs::remove_file(&tmp);
    match result {
        Ok(bytes) => axum::response::Response::builder()
            .header(axum::http::header::CONTENT_TYPE, "application/vnd.sqlite3")
            .header(
                axum::http::header::CONTENT_DISPOSITION,
                "attachment; filename=\"paddock-export.db\"",
            )
            .body(axum::body::Body::from(bytes))
            .expect("valid response"),
        Err(e) => {
            tracing::error!(%e, "db export sanitize failed");
            (StatusCode::INTERNAL_SERVER_ERROR, "export failed").into_response()
        }
    }
}

/// Strip credentials from a snapshot copy in place, then read its bytes.
fn sanitize_export(path: &std::path::Path) -> Result<Vec<u8>, String> {
    sanitize_export_file(path)?;
    std::fs::read(path).map_err(|e| e.to_string())
}

pub(crate) fn sanitize_export_file(path: &std::path::Path) -> Result<(), String> {
    let conn = rusqlite::Connection::open(path).map_err(|e| e.to_string())?;
    // API keys are hashes, but drop them regardless - nothing to share here.
    conn.execute("DELETE FROM api_keys", [])
        .map_err(|e| e.to_string())?;
    // Cloud-endpoint keys are the user's PROVIDER credentials (plaintext by
    // necessity - they go on the wire to the provider). The endpoints
    // themselves (name/URL/model picks) are shareable config; the keys are
    // not. If EXISTS via sqlite_master guard is unnecessary: the snapshot
    // always carries the current schema.
    conn.execute("UPDATE cloud_endpoints SET api_key = ''", [])
        .map_err(|e| e.to_string())?;
    // Both web plaintext and native Keychain references must stay local.
    conn.execute("UPDATE connectors SET headers = '{}', oauth = ''", [])
        .map_err(|e| e.to_string())?;
    // Legacy rows from before the config-file split: MCP servers
    // and the web-search key no longer live in this DB - drop the leftovers
    // from older installs so an export can't leak what current code never
    // writes.
    conn.execute("DROP TABLE IF EXISTS mcp_servers", [])
        .map_err(|e| e.to_string())?;
    conn.execute("DELETE FROM settings WHERE key = 'web_search'", [])
        .map_err(|e| e.to_string())?;
    // saved credentials (the Hugging Face token) - never shared
    conn.execute(
        "DELETE FROM settings WHERE key LIKE ?1",
        rusqlite::params![format!("{}%", crate::store::SECRET_PREFIX)],
    )
    .map_err(|e| e.to_string())?;
    // Mandatory: a failed compaction must not export deleted credential bytes.
    conn.execute_batch("VACUUM").map_err(|e| e.to_string())?;
    Ok(())
}
