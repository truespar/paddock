//! OpenAI's Uploads API, the resumable way in for a big batch input. Plain
//! `/v1/files` is one request, so a 60 GB upload that drops at 59 GB starts
//! over. Here the client creates an Upload, sends it in parts (64 MB each from
//! the SDK's own `upload_file_chunked`), and completes it with the part order.
//! A part that fails is sent again on its own.
//!
//! Completion is resumable too. Parts are appended to one assembly in the
//! given order, and each append is synced before the store records it; only
//! then is that part's file deleted. A completion cut short (a crash, a client
//! that gave up waiting) continues from the last recorded part when it is
//! called again. The disk holds the file plus one part at the peak, never the
//! file twice.
//!
//! Deviations: a pending upload lives 7 days, not an hour, because a large one
//! over a slow link takes longer than that. Parts may be up to 1 GB, and the
//! purpose is `batch` only, as on `/v1/files`.

use std::collections::HashSet;
use std::io::Read;
use std::sync::{Arc, Mutex};

use axum::Router;
use axum::extract::{DefaultBodyLimit, Multipart, Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use axum::routing::post;
use md5::Digest;
use serde::Deserialize;
use serde_json::{Value, json};

use super::openai::{error, file_object};
use super::{Batches, files, new_id, now_ms};
use crate::routes::AppState;
use crate::store::{BatchFileRow, UploadRow};

/// How long a pending upload waits for its parts and completion.
const PENDING_MS: i64 = 7 * 86_400_000;

/// Largest single part.
const MAX_PART: u64 = 1 << 30;

pub(crate) fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/v1/uploads", post(create))
        .route(
            "/v1/uploads/{id}/parts",
            post(add_part).layer(DefaultBodyLimit::max(MAX_PART as usize + (1 << 20))),
        )
        .route("/v1/uploads/{id}/complete", post(complete))
        .route("/v1/uploads/{id}/cancel", post(cancel))
}

fn bad(m: impl Into<String>) -> Response {
    error(StatusCode::BAD_REQUEST, "invalid_request_error", m)
}

fn internal(e: impl std::fmt::Display) -> Response {
    error(
        StatusCode::INTERNAL_SERVER_ERROR,
        "server_error",
        e.to_string(),
    )
}

fn not_found() -> Response {
    error(
        StatusCode::NOT_FOUND,
        "invalid_request_error",
        "No such upload",
    )
}

fn dir(b: &Batches, id: &str) -> std::path::PathBuf {
    b.root.join("uploads").join(id)
}

/// The file an upload becomes: its id derives from the upload's, so a
/// completion that is redone lands on the same file.
fn file_id_of(upload: &str) -> String {
    format!("file-{}u", upload.rsplit('_').next().unwrap_or(upload))
}

fn upload_object(b: &Batches, u: &UploadRow) -> Value {
    let file = u
        .file_id
        .as_deref()
        .and_then(|f| b.db.batch_file_get(f).ok().flatten())
        .map(|f| file_object(&f));
    json!({
        "id": u.id,
        "object": "upload",
        "bytes": u.bytes,
        "created_at": u.created_ms / 1000,
        "filename": u.filename,
        "purpose": u.purpose,
        "status": u.status,
        "expires_at": u.expires_ms / 1000,
        "file": file,
    })
}

#[derive(Deserialize)]
struct CreateBody {
    bytes: i64,
    filename: String,
    mime_type: String,
    purpose: String,
}

async fn create(State(st): State<Arc<AppState>>, Json(req): Json<CreateBody>) -> Response {
    let b = &st.batches;
    if req.purpose != "batch" {
        return bad(format!(
            "purpose: {:?} is not supported here - uploads are for batch inputs (\"batch\")",
            req.purpose
        ));
    }
    if req.bytes <= 0 {
        return bad("bytes: must be positive");
    }
    if let Err(e) = files::headroom(&b.root, req.bytes as u64) {
        return error(StatusCode::INSUFFICIENT_STORAGE, "insufficient_storage", e);
    }
    let now = now_ms();
    let u = UploadRow {
        id: new_id("upload_"),
        filename: req.filename,
        purpose: req.purpose,
        mime_type: req.mime_type,
        bytes: req.bytes,
        created_ms: now,
        expires_ms: now + PENDING_MS,
        status: "pending".into(),
        part_order: None,
        assembled: 0,
        assembled_bytes: 0,
        file_id: None,
    };
    if let Err(e) = b.db.upload_insert(&u) {
        return internal(e);
    }
    Json(upload_object(b, &u)).into_response()
}

fn pending(b: &Batches, id: &str) -> Result<UploadRow, Box<Response>> {
    match b.db.upload_get(id) {
        Ok(Some(u)) if u.status == "pending" => Ok(u),
        Ok(Some(u)) => Err(Box::new(bad(format!(
            "upload {id} is {}, not pending",
            u.status
        )))),
        Ok(None) => Err(Box::new(not_found())),
        Err(e) => Err(Box::new(internal(e))),
    }
}

async fn add_part(
    State(st): State<Arc<AppState>>,
    Path(id): Path<String>,
    mut mp: Multipart,
) -> Response {
    let b = &st.batches;
    if let Err(r) = pending(b, &id) {
        return *r;
    }
    let part = new_id("part_");
    let dest = dir(b, &id).join(&part);
    loop {
        match mp.next_field().await {
            Ok(Some(f)) if f.name() == Some("data") => {
                let n = match files::stream_to_file(Box::pin(f), &dest).await {
                    Ok(n) => n,
                    Err(files::UploadError::Disk(e)) => {
                        return error(StatusCode::INSUFFICIENT_STORAGE, "insufficient_storage", e);
                    }
                    Err(e) => return bad(e.to_string()),
                };
                if let Err(e) = b.db.upload_part_insert(&id, &part, n as i64) {
                    let _ = tokio::fs::remove_file(&dest).await;
                    return internal(e);
                }
                return Json(json!({
                    "id": part,
                    "object": "upload.part",
                    "created_at": now_ms() / 1000,
                    "upload_id": id,
                }))
                .into_response();
            }
            Ok(Some(_)) => {}
            Ok(None) => return bad("data: a part needs a data field"),
            Err(e) => return bad(format!("invalid multipart body: {e}")),
        }
    }
}

#[derive(Deserialize)]
struct CompleteBody {
    part_ids: Vec<String>,
    #[serde(default)]
    md5: Option<String>,
}

/// Uploads being assembled right now: a second completion of the same one
/// waits its turn rather than appending alongside the first.
static ASSEMBLING: Mutex<Option<HashSet<String>>> = Mutex::new(None);

struct Assembling(String);

impl Assembling {
    fn take(id: &str) -> Option<Self> {
        let mut g = ASSEMBLING
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        g.get_or_insert_with(HashSet::new)
            .insert(id.to_owned())
            .then(|| Self(id.to_owned()))
    }
}

impl Drop for Assembling {
    fn drop(&mut self) {
        if let Some(s) = ASSEMBLING
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_mut()
        {
            s.remove(&self.0);
        }
    }
}

async fn complete(
    State(st): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(req): Json<CompleteBody>,
) -> Response {
    let b = st.batches.clone();
    // already done: the same answer again (a client that timed out waiting)
    if let Ok(Some(u)) = b.db.upload_get(&id)
        && u.status == "completed"
    {
        return Json(upload_object(&b, &u)).into_response();
    }
    let Some(_guard) = Assembling::take(&id) else {
        return error(
            StatusCode::CONFLICT,
            "invalid_request_error",
            "this upload is being completed by another request; retry shortly",
        );
    };
    // read under the guard: a completion that ran just before this one moved
    // the assembled count
    let u = match pending(&b, &id) {
        Ok(u) => u,
        Err(r) => return *r,
    };
    let parts = match b.db.upload_parts(&id) {
        Ok(p) => p.into_iter().collect::<std::collections::HashMap<_, _>>(),
        Err(e) => return internal(e),
    };
    // part rows stay until the upload completes, so a resumed completion
    // still finds every id (only the appended parts' files are gone)
    let mut seen = HashSet::new();
    let mut total = 0i64;
    for p in &req.part_ids {
        let Some(n) = parts.get(p) else {
            return bad(format!("part_ids: {p} is not a part of upload {id}"));
        };
        if !seen.insert(p) {
            return bad(format!("part_ids: {p} is listed twice"));
        }
        total += n;
    }
    if total != u.bytes {
        return bad(format!(
            "part_ids: the parts hold {total} bytes, the upload was created for {}",
            u.bytes
        ));
    }
    let order = serde_json::to_string(&req.part_ids).unwrap_or_default();
    match b.db.upload_set_order(&id, &order) {
        Ok(true) => {}
        Ok(false) => {
            return bad("part_ids: a completion already started with a different order");
        }
        Err(e) => return internal(e),
    }
    let fid = file_id_of(&id);
    let db = b.db.clone();
    let (root, up_dir) = (b.root.clone(), dir(&b, &id));
    let (order, md5) = (req.part_ids.clone(), req.md5.clone());
    let assembled = tokio::task::spawn_blocking(move || {
        assemble(&db, &root, &up_dir, &u, &order, md5.as_deref(), &fid)
    })
    .await;
    match assembled {
        Ok(Ok(())) => {}
        Ok(Err(e)) => return bad(e),
        Err(e) => return internal(e),
    }
    let _ = tokio::fs::remove_dir_all(dir(&b, &id)).await;
    match b.db.upload_get(&id) {
        Ok(Some(u)) => Json(upload_object(&b, &u)).into_response(),
        Ok(None) => not_found(),
        Err(e) => internal(e),
    }
}

/// Append the parts in order, from wherever a previous completion stopped,
/// then publish the file. Blocking: run on the blocking pool.
fn assemble(
    db: &crate::store::Store,
    root: &std::path::Path,
    up_dir: &std::path::Path,
    u: &UploadRow,
    order: &[String],
    md5: Option<&str>,
    fid: &str,
) -> Result<(), String> {
    let io = |e: std::io::Error| e.to_string();
    let target = root.join("files").join(fid);
    let assembly = up_dir.join("assembly");
    // published already, the store not told yet (a crash in between)
    if !target.exists() {
        let mut out = files::ResultFile::open(&assembly, u.assembled_bytes as u64).map_err(io)?;
        let mut buf = vec![0u8; 1 << 20];
        for (k, part) in order.iter().enumerate().skip(u.assembled as usize) {
            let src = up_dir.join(part);
            let mut f = std::fs::File::open(&src).map_err(|e| format!("part {part}: {e}"))?;
            loop {
                let n = f.read(&mut buf).map_err(io)?;
                if n == 0 {
                    break;
                }
                out.append(&buf[..n]).map_err(io)?;
            }
            out.sync().map_err(io)?;
            db.upload_assembled(&u.id, k as i64 + 1, out.len as i64)
                .map_err(|e| e.to_string())?;
            let _ = std::fs::remove_file(&src);
        }
        drop(out);
        if let Some(want) = md5 {
            let mut h = md5::Md5::new();
            let mut f = std::fs::File::open(&assembly).map_err(io)?;
            let mut buf = vec![0u8; 1 << 20];
            loop {
                let n = f.read(&mut buf).map_err(io)?;
                if n == 0 {
                    break;
                }
                h.update(&buf[..n]);
            }
            let got: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
            if !got.eq_ignore_ascii_case(want.trim()) {
                // the bytes are not the ones the client meant: nothing to resume
                let _ = db.upload_close(&u.id, "cancelled");
                let _ = std::fs::remove_dir_all(up_dir);
                return Err(format!(
                    "md5: the uploaded bytes hash to {got}, not {want}; the upload is cancelled"
                ));
            }
        }
        std::fs::create_dir_all(root.join("files")).map_err(io)?;
        std::fs::rename(&assembly, &target).map_err(io)?;
        if let Ok(d) = std::fs::File::open(root.join("files")) {
            let _ = d.sync_all();
        }
    }
    let file = BatchFileRow {
        id: fid.to_owned(),
        purpose: "batch".into(),
        filename: u.filename.clone(),
        bytes: u.bytes,
        created_ms: now_ms(),
        path: format!("files/{fid}"),
        batch_id: None,
    };
    db.upload_finish(&u.id, &file).map_err(|e| e.to_string())
}

async fn cancel(State(st): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    let b = &st.batches;
    let u = match pending(b, &id) {
        Ok(u) => u,
        Err(r) => return *r,
    };
    if Assembling::take(&id).is_none() {
        return error(
            StatusCode::CONFLICT,
            "invalid_request_error",
            "this upload is being completed; it cannot be cancelled now",
        );
    }
    if let Err(e) = b.db.upload_close(&u.id, "cancelled") {
        return internal(e);
    }
    let _ = tokio::fs::remove_dir_all(dir(b, &id)).await;
    match b.db.upload_get(&id) {
        Ok(Some(u)) => Json(upload_object(b, &u)).into_response(),
        _ => not_found(),
    }
}

/// Expire pending uploads past their deadline (the courier calls this).
pub(crate) fn sweep(b: &Batches) {
    for id in b.db.uploads_expired(now_ms()).unwrap_or_default() {
        if Assembling::take(&id).is_some() && b.db.upload_close(&id, "expired").unwrap_or(false) {
            let _ = std::fs::remove_dir_all(dir(b, &id));
            tracing::info!(upload = %id, "batch: pending upload expired");
        }
    }
}
