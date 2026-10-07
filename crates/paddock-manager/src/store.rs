//! Embedded SQLite store - the single source of truth for the Studio.
//!
//! Conversations, prompts, presets, settings, and API keys live in one file at
//! `~/paddock/paddock.db` (bundled SQLite, so the binary stays self-contained).
//! Conversations are stored as a JSON document plus a few indexed metadata
//! columns for listing/sorting; attachment bytes live on disk (later).

use std::path::PathBuf;
use std::sync::Mutex;

use rusqlite::{Connection, OptionalExtension, params};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use uuid::Uuid;

mod connections;
mod downloads;
mod integrations;
mod mask_history;
mod model_profiles;
mod native_benchmarks;
mod prompts;
mod read_history;
mod read_history_migration;
mod read_runs;
mod reads;
mod settings;
pub(crate) use settings::SECRET_PREFIX;
mod table_history;

pub struct Store {
    conn: Mutex<Connection>,
    cloud_credentials: crate::credentials::Session,
    connector_credentials: crate::credentials::Session,
    connector_credential_updates: std::sync::Mutex<()>,
}

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("database error: {0}")]
    Db(#[from] rusqlite::Error),
    #[error("invalid data: {0}")]
    Bad(String),
    #[error("{0}")]
    Conflict(String),
}

/// Default DB path: `<data root>/paddock.db`, alongside the models dir
/// (three-mode resolution).
pub fn default_db_path() -> PathBuf {
    paddock_admin::data_root().join("paddock.db")
}

fn now_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

/// A forensic analysis ready to persist: the per-attachment header plus its
/// findings. Deliberately a plain owned type (not `paddock_forensics::Report`)
/// so the manager's DB layer carries no dependency on the forensics engine - the
/// runner computes findings and hands the manager these strings/scalars.
#[derive(Debug, Default, Clone)]
pub struct NewForensicReport {
    /// The attachment this analyzed, when it came from a stored attachment.
    pub attachment_id: Option<String>,
    pub conversation_id: Option<String>,
    /// SHA-256 of the analyzed bytes (hex) - dedup + cache key across uploads.
    pub sha256: String,
    /// `"image"` or `"pdf"`.
    pub kind: String,
    pub mime: String,
    pub name: String,
    pub width: Option<i64>,
    pub height: Option<i64>,
    /// Content classification: `"photo"|"document"|"mixed"|"unknown"`.
    pub content_type: String,
    /// Decoded raster format (e.g. `"jpeg"`), or `""` for PDFs.
    pub format: String,
    /// Aggregate risk in 0.0..=1.0 (risk scorer).
    pub risk_score: f64,
    /// Short human verdict (`risk::Verdict` rendered).
    pub verdict: String,
    // ── the risk layer (paddock_forensics::RiskReport), captured in full so the
    //    durable record loses nothing the scorer produced ──────────────────────
    /// Leveled verdict severity - `info|low|medium|high|critical`. Distinct from
    /// `max_severity`: that is the loudest RAW finding, this is what the scorer
    /// concluded after corroboration + diminishing-returns weighting.
    pub risk_level: String,
    /// Independent analyzer families that agreed on something material.
    pub corroborating_stages: i64,
    /// Plain-language explanation (`ForensicExplanation`): the four narrative
    /// slots as scalars; the per-category breakdown rides `explanation_categories`.
    pub explanation_summary: String,
    pub explanation_visual_review: Option<String>,
    pub explanation_cross_corroboration: Option<String>,
    pub explanation_anti_forensics: Option<String>,
    /// Whether the GPU path served the analysis.
    pub gpu: bool,
    pub elapsed_ms: i64,
    /// Deduped, collapsed headline findings (`RiskReport::key_findings`).
    pub key_findings: Vec<NewForensicKeyFinding>,
    /// Per-category explanation rows (`ForensicExplanation::categories`).
    pub explanation_categories: Vec<NewForensicExplanationCategory>,
    /// The raw per-analyzer findings.
    pub findings: Vec<NewForensicFinding>,
}

/// One finding to persist. Severity is the lowercase string form of
/// `paddock_forensics::Severity`; region is the finding's `Region` as JSON (or
/// `""` when it has none).
#[derive(Debug, Default, Clone)]
pub struct NewForensicFinding {
    pub analyzer: String,
    pub code: String,
    pub severity: String,
    pub confidence: f64,
    pub description: String,
    pub region: String,
}

/// One collapsed headline finding (`paddock_forensics::risk::KeyFinding`) -
/// several raw findings of the same category merged into one line. `sources` is
/// the JSON array of contributing analyzers; `region` the JSON of the most
/// specific contributing region (or `""`); `count` how many raw findings folded in.
#[derive(Debug, Default, Clone)]
pub struct NewForensicKeyFinding {
    pub title: String,
    pub description: String,
    pub severity: String,
    pub confidence: f64,
    pub sources: Vec<String>,
    pub region: String,
    pub count: i64,
}

/// One explanation category (`paddock_forensics::risk::ExplanationCategory`) -
/// a family of findings with a shared plain-language write-up. `finding_codes`
/// is the JSON array of the codes that fed it.
#[derive(Debug, Default, Clone)]
pub struct NewForensicExplanationCategory {
    pub name: String,
    pub finding_count: i64,
    pub max_severity: String,
    pub explanation: String,
    pub finding_codes: Vec<String>,
}

/// Order forensic severities so the report can pick the highest present.
fn severity_rank(sev: &str) -> u8 {
    match sev {
        "critical" => 4,
        "high" => 3,
        "medium" => 2,
        "low" => 1,
        _ => 0, // "info" and anything unrecognized
    }
}

/// Map a `forensic_reports` row (24 columns, in the SELECT order used by every
/// query above) to JSON. The risk-layer child collections (`key_findings`,
/// `explanation.categories`) and the raw `findings` are attached by the caller;
/// the four explanation narrative slots and the scalar risk fields are here so a
/// summary listing carries the verdict without a second query.
fn forensic_report_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<Value> {
    Ok(json!({
        "id": r.get::<_, String>(0)?,
        "attachment_id": r.get::<_, Option<String>>(1)?,
        "conversation_id": r.get::<_, Option<String>>(2)?,
        "sha256": r.get::<_, String>(3)?,
        "kind": r.get::<_, String>(4)?,
        "mime": r.get::<_, String>(5)?,
        "name": r.get::<_, String>(6)?,
        "width": r.get::<_, Option<i64>>(7)?,
        "height": r.get::<_, Option<i64>>(8)?,
        "content_type": r.get::<_, String>(9)?,
        "format": r.get::<_, String>(10)?,
        "finding_count": r.get::<_, i64>(11)?,
        "max_severity": r.get::<_, String>(12)?,
        "risk_score": r.get::<_, f64>(13)?,
        "verdict": r.get::<_, String>(14)?,
        "gpu": r.get::<_, i64>(15)? != 0,
        "elapsed_ms": r.get::<_, i64>(16)?,
        "created_at": r.get::<_, i64>(17)?,
        "risk_level": r.get::<_, String>(18)?,
        "corroborating_stages": r.get::<_, i64>(19)?,
        "explanation": {
            "summary": r.get::<_, String>(20)?,
            "visual_review": r.get::<_, Option<String>>(21)?,
            "cross_corroboration": r.get::<_, Option<String>>(22)?,
            "anti_forensics_warning": r.get::<_, Option<String>>(23)?,
        },
    }))
}

/// Map a `forensic_key_findings` row (9 columns) to JSON. `sources` and `region`
/// are stored as JSON strings, so re-parse them back into structure.
fn forensic_key_finding_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<Value> {
    let sources: Value =
        serde_json::from_str(&r.get::<_, String>(4)?).unwrap_or_else(|_| Value::Array(vec![]));
    let region_raw = r.get::<_, String>(5)?;
    let region = if region_raw.is_empty() {
        Value::Null
    } else {
        serde_json::from_str(&region_raw).unwrap_or(Value::Null)
    };
    Ok(json!({
        "title": r.get::<_, String>(0)?,
        "description": r.get::<_, String>(1)?,
        "severity": r.get::<_, String>(2)?,
        "confidence": r.get::<_, f64>(3)?,
        "sources": sources,
        "region": region,
        "count": r.get::<_, i64>(6)?,
        "seq": r.get::<_, i64>(7)?,
    }))
}

/// Map a `forensic_explanation_categories` row (6 columns) to JSON.
/// `finding_codes` is a JSON array string.
fn forensic_explanation_category_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<Value> {
    let codes: Value =
        serde_json::from_str(&r.get::<_, String>(4)?).unwrap_or_else(|_| Value::Array(vec![]));
    Ok(json!({
        "name": r.get::<_, String>(0)?,
        "finding_count": r.get::<_, i64>(1)?,
        "max_severity": r.get::<_, String>(2)?,
        "explanation": r.get::<_, String>(3)?,
        "finding_codes": codes,
        "seq": r.get::<_, i64>(5)?,
    }))
}

/// Map a `forensic_findings` row (7 columns) to JSON. `region` is stored as a
/// JSON string, so re-parse it back into structure when it is non-empty.
fn forensic_finding_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<Value> {
    let region_raw = r.get::<_, String>(5)?;
    let region = if region_raw.is_empty() {
        Value::Null
    } else {
        serde_json::from_str(&region_raw).unwrap_or(Value::Null)
    };
    Ok(json!({
        "analyzer": r.get::<_, String>(0)?,
        "code": r.get::<_, String>(1)?,
        "severity": r.get::<_, String>(2)?,
        "confidence": r.get::<_, f64>(3)?,
        "description": r.get::<_, String>(4)?,
        "region": region,
        "seq": r.get::<_, i64>(6)?,
    }))
}

/// What a conversation turned out to be, from its own turns: `"document"`,
/// `"image"`, `"transcription"`, or `"chat"`.
///
/// Decided here, on the way in, because this is the only place that has both
/// the messages and the row. The Studio's list is summaries - every unopened
/// conversation arrives with an empty message array - so a client-side answer
/// was structurally unable to see the evidence and called everything a chat
/// until you clicked it.
///
/// EVIDENCE, never the model's capabilities. A model does not decide what you
/// did with it: granite-vision is catalogued `documents` and chats perfectly
/// well (13 such conversations on the maintainers' box, not one with a document run),
/// while a plain chat model handed a PDF produces a real document run. Only
/// the turns know.
///
/// Document beats transcription when somehow both are present: a document run
/// is the more specific claim, and an audio part in the same thread is an
/// attachment rather than the point of it.
pub fn conversation_kind(doc: &Value) -> &'static str {
    let msgs = doc
        .get("messages")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let parts = |m: &Value| -> Vec<Value> {
        m.get("content")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
    };
    // A per-page document run (docRun) or a single OCR pass (ocr) - the two
    // shapes the document lane writes onto its assistant turn.
    let documented = msgs.iter().any(|m| {
        m.get("role").and_then(Value::as_str) == Some("assistant")
            && (m.get("docRun").is_some_and(|v| !v.is_null())
                || m.get("ocr").is_some_and(|v| !v.is_null()))
    });
    if documented {
        return "document";
    }
    // A picture the MODEL made: an image part on an ASSISTANT turn. A person's
    // attached photo is a user part and stays a chat about it.
    let drew = msgs.iter().any(|m| {
        m.get("role").and_then(Value::as_str) == Some("assistant")
            && parts(m)
                .iter()
                .any(|p| p.get("type").and_then(Value::as_str) == Some("image"))
    });
    if drew {
        return "image";
    }
    let heard = msgs.iter().any(|m| {
        parts(m)
            .iter()
            .any(|p| p.get("type").and_then(Value::as_str) == Some("audio"))
    });
    if heard { "transcription" } else { "chat" }
}

impl Store {
    /// A CONSISTENT copy of the DB written to a fresh temp file (`VACUUM INTO`,
    /// so no WAL/locking surprises). The caller owns the file - sanitize it,
    /// read it, then delete it. Used by the sanitized export endpoint.
    pub fn snapshot_to_temp(&self) -> Result<PathBuf, StoreError> {
        // The unsanitized snapshot contains credentials. Random exclusive
        // creation and mode 0600 apply BEFORE SQLite starts writing.
        let tmp = tempfile::Builder::new()
            .prefix("paddock-export-")
            .suffix(".db")
            .tempfile()
            .map_err(|e| StoreError::Bad(e.to_string()))?
            .into_temp_path();
        let conn = self.lock();
        conn.execute("VACUUM INTO ?1", params![tmp.to_string_lossy()])?;
        tmp.keep().map_err(|e| StoreError::Bad(e.to_string()))
    }
}

fn hash_key(key: &str) -> String {
    let mut h = Sha256::new();
    h.update(key.as_bytes());
    crate::registry::hex(&h.finalize())
}

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS conversations (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    model      TEXT NOT NULL DEFAULT '',
    pinned     INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    -- 'chat' | 'transcription' | 'document': what this conversation turned out
    -- to BE, decided from its own turns on the way in. A metadata column for
    -- the same reason title and model are: the list must answer without
    -- reading `doc`. The Studio used to work it out client-side and could
    -- not - the list ships summaries, so every unopened row had an empty
    -- messages array and read as a plain chat until you clicked it.
    kind       TEXT NOT NULL DEFAULT '',
    doc        TEXT NOT NULL
);
-- The list index is NOT here: it covers `kind`, which arrives by ALTER in
-- `open` (see `conversations_list` there).

CREATE TABLE IF NOT EXISTS prompts (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    body       TEXT NOT NULL,
    variables  TEXT NOT NULL DEFAULT '[]',
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS presets (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    params     TEXT NOT NULL,
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS model_profiles (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    model TEXT NOT NULL,
    artifact TEXT NOT NULL,
    settings TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    UNIQUE(model, artifact, name)
);

CREATE TABLE IF NOT EXISTS native_benchmarks (
    id TEXT PRIMARY KEY,
    created_at INTEGER NOT NULL,
    report TEXT NOT NULL
);

-- The Reads page's saved question sets (store/reads.rs): `body` is the JSON
-- text of { questions, samples } in the /v1/systemone shape. User content,
-- like prompts - never model configuration.
CREATE TABLE IF NOT EXISTS read_sets (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    body       TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

-- the Reads page's earlier reads (store/read_history.rs): a text, its
-- questions and every run, like a conversation - title/model/runs are columns
-- so the side panel lists without reading `doc`
CREATE TABLE IF NOT EXISTS read_history (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    model      TEXT NOT NULL DEFAULT '',
    runs       INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    doc        TEXT NOT NULL
);

-- the native app's bounded result history (store/read_runs.rs): the last
-- results per question set, a short excerpt and never the document
CREATE TABLE IF NOT EXISTS read_runs (
    scope TEXT NOT NULL,
    id TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    body TEXT NOT NULL,
    PRIMARY KEY(scope,id)
);
CREATE INDEX IF NOT EXISTS read_runs_recent ON read_runs(created_at DESC);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS api_keys (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    key_hash     TEXT NOT NULL,
    prefix       TEXT NOT NULL,
    created_at   INTEGER NOT NULL,
    last_used_at INTEGER,
    revoked      INTEGER NOT NULL DEFAULT 0
);

-- BYO-key cloud endpoints (manager-runner doc: the manager's external-provider
-- client role). An endpoint is a provider base URL + the user's own API key +
-- the models they enabled from it; those models join the Studio's pickers as
-- Cloud models for chat/compare. The key never leaves this machine except
-- toward the provider itself: the list API reports only has_key, and exports
-- blank the column. `kind` picks the wire dialect (openai | openai-compat |
-- anthropic); `models` is a JSON array of {id, display?, ctx?, vision?}.
CREATE TABLE IF NOT EXISTS cloud_endpoints (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    kind       TEXT NOT NULL,
    base_url   TEXT NOT NULL,
    api_key    TEXT NOT NULL DEFAULT '',
    models     TEXT NOT NULL DEFAULT '[]',
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS attachments (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT,
    mime            TEXT NOT NULL,
    name            TEXT NOT NULL DEFAULT '',
    width           INTEGER,
    height          INTEGER,
    bytes           BLOB NOT NULL,
    created_at      INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS attachments_conv ON attachments(conversation_id);

-- The Studio's personal MCP connector library: hosted MCP
-- servers the user tries per chat. Distinct from a served endpoint's own
-- mcp_servers (those are endpoint contract and live in servers/<port>.toml) -
-- a connector rides per REQUEST as the OpenAI inline `mcp` tool
-- (server_url + headers), so nothing here configures a runner and the
-- no-model-config-in-this-DB rule is not breached. `headers` is a
-- JSON object and may hold a bearer secret; the Studio must read it back to
-- build requests, so unlike cloud keys it IS returned - a deliberate
-- loopback tradeoff (the hardening path is runner-side label resolution).
-- `registry_key` is the stable truespar-registry key when picked from there
-- ('' = hand-entered).
-- system = available-on-every-server: the manager MATERIALIZES the connector
-- into every servers/<port>.toml (marked with connector_id so unchecking or
-- deleting strips exactly its entries and never a hand-added tool). The
-- library row is the intent; the TOMLs stay the endpoint truth.
CREATE TABLE IF NOT EXISTS connectors (
    id           TEXT PRIMARY KEY,
    label        TEXT NOT NULL,
    url          TEXT NOT NULL,
    headers      TEXT NOT NULL DEFAULT '{}',
    registry_key TEXT NOT NULL DEFAULT '',
    system       INTEGER NOT NULL DEFAULT 0,
    ports        TEXT NOT NULL DEFAULT '[]',
    oauth        TEXT NOT NULL DEFAULT '',
    created_at   INTEGER NOT NULL
);

-- (instance_id, seq) identifies a record across runner restarts: instance_id
-- is the runner's per-process-start UUID from identify, so a
-- restarted runner - whose sequences reset to 0 - can never collide with its
-- predecessor, however fast the respawn. The old key (port, runner_started_at)
-- was second-resolution and silently dropped a same-second successor's rows.
-- port/runner_started_at stay as plain columns for filtering and display.
CREATE TABLE IF NOT EXISTS activity (
    instance_id       TEXT NOT NULL,
    seq               INTEGER NOT NULL,
    port              INTEGER NOT NULL,
    runner_started_at INTEGER NOT NULL,
    ts_ms             INTEGER NOT NULL,
    request_id        TEXT NOT NULL DEFAULT '',
    endpoint          TEXT NOT NULL DEFAULT '',
    status            INTEGER NOT NULL DEFAULT 0,
    model             TEXT,
    session_id        TEXT,
    -- paddock_admin::codec blob: CBOR in a dictionary-compressed
    -- zstd frame behind a version byte - ~5x under the JSON text it replaced,
    -- which at the hammered rate is 218 GB/30 d of key names.
    -- Rows written before the codec are plain JSON TEXT; decode_record
    -- sniffs the first byte and reads both.
    record            BLOB NOT NULL,
    PRIMARY KEY (instance_id, seq)
);
CREATE INDEX IF NOT EXISTS activity_ts ON activity(ts_ms DESC);

CREATE TABLE IF NOT EXISTS cloud_usage (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ms            INTEGER NOT NULL,
    endpoint         TEXT NOT NULL,
    model            TEXT NOT NULL,
    input_tokens     INTEGER NOT NULL DEFAULT 0,
    output_tokens    INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
    cost             REAL,
    -- Seconds of AUDIO, for a transcription. Its own column
    -- because it is a different unit, not a different number: a whisper-class
    -- model bills per second and reports no tokens at all, so a row reading
    -- zero tokens and four tenths of a cent would be true and useless.
    -- NULL on every text request.
    audio_seconds    REAL
);
CREATE INDEX IF NOT EXISTS cloud_usage_ts ON cloud_usage(ts_ms DESC);

-- Model-authored artifacts: substantial standalone content the
-- user iterates on across turns - a chart, a page, a document. The
-- conversation carries the OPERATIONS (an artifact_update call and its
-- one-line result); the BODY lives here. That is the whole point: a
-- 600-line canvas edited four times costs four short tool results in
-- context instead of four full copies, and artifact_read pulls a body back
-- only when the model actually needs to see it again.
CREATE TABLE IF NOT EXISTS artifacts (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    kind            TEXT NOT NULL,
    title           TEXT NOT NULL DEFAULT '',
    language        TEXT NOT NULL DEFAULT '',
    -- Who wrote it. A compare turn runs several models against ONE
    -- conversation, so without this their artifacts land in a single flat
    -- list with no way to tell them apart - let alone show them side by side,
    -- which is the whole point of comparing on a wide screen.
    model           TEXT NOT NULL DEFAULT '',
    created_at      INTEGER NOT NULL,
    updated_at      INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS artifacts_conv ON artifacts(conversation_id, updated_at DESC);

-- Every edit appends a row; nothing is ever mutated in place, so the version
-- selector and a diff between any two versions come free. `op` records how
-- the version was produced (create / update / rewrite) so the panel can say
-- what happened without re-deriving it from the content.
CREATE TABLE IF NOT EXISTS artifact_versions (
    artifact_id TEXT NOT NULL,
    seq         INTEGER NOT NULL,
    op          TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  INTEGER NOT NULL,
    PRIMARY KEY (artifact_id, seq)
);

-- ── usage tier: the manager's scrape of runner /metrics ──

-- Interned series dimensions: ENDPOINT-level only, deliberately no
-- generation - a generation in this key would mint a fresh series set on
-- every restart. `operation = 'engine'` is the one pseudo-value: the home of
-- engine-scoped numbers (spec decode, KV high-water) that have no operation
-- or origin of their own; its request/token columns stay zero.
CREATE TABLE IF NOT EXISTS usage_series (
    id        INTEGER PRIMARY KEY,
    port      INTEGER NOT NULL,
    model     TEXT NOT NULL,
    operation TEXT NOT NULL,
    origin    TEXT NOT NULL,
    UNIQUE(port, model, operation, origin)
);

-- The rate/shape tier: sparse (rows exist only where traffic did), fed by
-- scrape deltas, holes where nobody was scraping (usage_gap says so).
-- ttft_h*/e2e_h* are NON-cumulative per-bucket increments on the 14-step
-- semconv ladder; the +Inf overflow is derivable (requests - Σ e2e_h*, and
-- successes - Σ ttft_h* for TTFT, which observes only successes). busy_ms
-- stays 0 until the event tier enriches it - scrape deltas cannot spread a
-- request across the buckets it spanned.
CREATE TABLE IF NOT EXISTS usage_bucket (
    series_id       INTEGER NOT NULL,
    grain           TEXT NOT NULL,
    bucket_start_ms INTEGER NOT NULL,
    requests        INTEGER NOT NULL DEFAULT 0,
    disconnects     INTEGER NOT NULL DEFAULT 0,
    errors_4xx      INTEGER NOT NULL DEFAULT 0,
    errors_5xx      INTEGER NOT NULL DEFAULT 0,
    input_tokens    INTEGER NOT NULL DEFAULT 0,
    output_tokens   INTEGER NOT NULL DEFAULT 0,
    cached_tokens   INTEGER NOT NULL DEFAULT 0,
    duration_ms_sum INTEGER NOT NULL DEFAULT 0,
    busy_ms         INTEGER NOT NULL DEFAULT 0,
    ttft_h0 INTEGER NOT NULL DEFAULT 0, ttft_h1 INTEGER NOT NULL DEFAULT 0,
    ttft_h2 INTEGER NOT NULL DEFAULT 0, ttft_h3 INTEGER NOT NULL DEFAULT 0,
    ttft_h4 INTEGER NOT NULL DEFAULT 0, ttft_h5 INTEGER NOT NULL DEFAULT 0,
    ttft_h6 INTEGER NOT NULL DEFAULT 0, ttft_h7 INTEGER NOT NULL DEFAULT 0,
    ttft_h8 INTEGER NOT NULL DEFAULT 0, ttft_h9 INTEGER NOT NULL DEFAULT 0,
    ttft_h10 INTEGER NOT NULL DEFAULT 0, ttft_h11 INTEGER NOT NULL DEFAULT 0,
    ttft_h12 INTEGER NOT NULL DEFAULT 0, ttft_h13 INTEGER NOT NULL DEFAULT 0,
    e2e_h0 INTEGER NOT NULL DEFAULT 0, e2e_h1 INTEGER NOT NULL DEFAULT 0,
    e2e_h2 INTEGER NOT NULL DEFAULT 0, e2e_h3 INTEGER NOT NULL DEFAULT 0,
    e2e_h4 INTEGER NOT NULL DEFAULT 0, e2e_h5 INTEGER NOT NULL DEFAULT 0,
    e2e_h6 INTEGER NOT NULL DEFAULT 0, e2e_h7 INTEGER NOT NULL DEFAULT 0,
    e2e_h8 INTEGER NOT NULL DEFAULT 0, e2e_h9 INTEGER NOT NULL DEFAULT 0,
    e2e_h10 INTEGER NOT NULL DEFAULT 0, e2e_h11 INTEGER NOT NULL DEFAULT 0,
    e2e_h12 INTEGER NOT NULL DEFAULT 0, e2e_h13 INTEGER NOT NULL DEFAULT 0,
    spec_drafted  INTEGER NOT NULL DEFAULT 0,
    spec_accepted INTEGER NOT NULL DEFAULT 0,
    kv_pages_max  INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (series_id, grain, bucket_start_ms)
);
CREATE INDEX IF NOT EXISTS usage_bucket_window ON usage_bucket(grain, bucket_start_ms);

-- The totals tier: last-observed CUMULATIVE counter values per series per
-- GENERATION (instance_id - a per-port total would read a
-- successor's fresh counters as a negative delta). A high-water mark
-- overwritten on every scrape, never a time series; lifetime numbers stay
-- exact across any collector blind spell.
CREATE TABLE IF NOT EXISTS usage_total (
    series_id      INTEGER NOT NULL,
    instance_id    TEXT NOT NULL,
    started_ms     INTEGER NOT NULL,
    requests       INTEGER NOT NULL DEFAULT 0,
    input_tokens   INTEGER NOT NULL DEFAULT 0,
    output_tokens  INTEGER NOT NULL DEFAULT 0,
    cached_tokens  INTEGER NOT NULL DEFAULT 0,
    spec_drafted   INTEGER NOT NULL DEFAULT 0,
    spec_accepted  INTEGER NOT NULL DEFAULT 0,
    last_scrape_ms INTEGER NOT NULL,
    PRIMARY KEY (series_id, instance_id)
);
CREATE INDEX IF NOT EXISTS usage_total_instance ON usage_total(instance_id);

-- The collector's FULL last-observed counter state per generation:
-- every column, serialized in the same wire shape the runner's
-- snapshot ring speaks. This is the attach baseline snapshot recovery folds
-- from - usage_total's four columns cannot rebuild error counts or latency
-- histograms for the head interval of a blind window; this can. Overwritten
-- atomically WITH the folds it follows, so it is also the exact crash-resume
-- cursor. One small row per live generation, never a time series.
CREATE TABLE IF NOT EXISTS usage_state (
    instance_id TEXT PRIMARY KEY,
    ts_ms       INTEGER NOT NULL,
    state       TEXT NOT NULL
);

-- Lifecycle bands: distinguishes idle from not-running, and a crash band
-- beside a traffic hole answers the question the hole raises. Causes may be
-- NULL: the collector writes what it OBSERVES; the manager's own routes
-- upgrade what they know (a clean stop, a manual start).
CREATE TABLE IF NOT EXISTS service_generation (
    instance_id    TEXT PRIMARY KEY,
    port           INTEGER NOT NULL,
    pid            INTEGER NOT NULL,
    runner_version TEXT NOT NULL DEFAULT '',
    model          TEXT,
    embedder       TEXT,
    asr            TEXT,
    -- Fourth serving role, same story as asr: an aligner-only runner carries
    -- ONLY this, so a band keyed on the other three names nothing and the
    -- usage chart draws it as a dash.
    aligner        TEXT,
    started_ms     INTEGER NOT NULL,
    ended_ms       INTEGER,
    start_cause    TEXT,
    end_cause      TEXT,
    -- When the MACHINE booted, as this band saw it. A band whose boot differs
    -- from the current one cannot still be running whatever else we failed to
    -- observe: no process survives a reboot. That is the one end a sweep can
    -- state as fact rather than as an absence of sightings.
    boot_ms        INTEGER
);
CREATE INDEX IF NOT EXISTS service_generation_port ON service_generation(port, started_ms);

-- Holes, so a chart never quietly under-reports. Metrics-tier holes carry
-- the EXACT lost totals (a counter remembers); event-tier holes carry the
-- seq range instead and NULL totals.
CREATE TABLE IF NOT EXISTS usage_gap (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    port              INTEGER NOT NULL,
    instance_id       TEXT NOT NULL,
    from_ts_ms        INTEGER NOT NULL,
    to_ts_ms          INTEGER NOT NULL,
    noticed_ms        INTEGER NOT NULL,
    cause             TEXT NOT NULL,
    from_seq          INTEGER,
    to_seq            INTEGER,
    lost_requests     INTEGER,
    lost_input_tokens INTEGER,
    lost_output_tokens INTEGER
);
CREATE INDEX IF NOT EXISTS usage_gap_window ON usage_gap(from_ts_ms);

-- Web-search spend, the second thing a box spends besides GPU
-- time - and the only one that leaves the machine and bills a card. Its own
-- tier because its one dimension is the PROVIDER: pushing it through
-- usage_series would have written 'exa' into a column named `model`.
--
-- The three counters stay apart on purpose. requests is the honest floor
-- (every provider reports it, even the two that price nothing); credits are
-- meaningless outside one provider's pricing page and are NOT one per search
-- (a Firecrawl search that scraped three pages cost 38); microdollars are
-- integer so money never rides on a float. Nothing here may be summed across
-- providers without naming which currency.
CREATE TABLE IF NOT EXISTS web_search_bucket (
    port            INTEGER NOT NULL,
    provider        TEXT NOT NULL,
    grain           TEXT NOT NULL,
    bucket_start_ms INTEGER NOT NULL,
    requests        INTEGER NOT NULL DEFAULT 0,
    credits         INTEGER NOT NULL DEFAULT 0,
    microdollars    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (port, provider, grain, bucket_start_ms)
);
CREATE INDEX IF NOT EXISTS web_search_bucket_window
    ON web_search_bucket(grain, bucket_start_ms);

-- Last-observed CUMULATIVE spend per generation per provider - the same
-- high-water shape as usage_total, and for the same reason: the bucket tier
-- has holes wherever nobody was scraping, but a counter remembers. This is
-- what keeps the lifetime answer (what has this key spent?) exact across any
-- manager outage.
CREATE TABLE IF NOT EXISTS web_search_total (
    instance_id    TEXT NOT NULL,
    port           INTEGER NOT NULL,
    provider       TEXT NOT NULL,
    started_ms     INTEGER NOT NULL,
    requests       INTEGER NOT NULL DEFAULT 0,
    credits        INTEGER NOT NULL DEFAULT 0,
    microdollars   INTEGER NOT NULL DEFAULT 0,
    last_scrape_ms INTEGER NOT NULL,
    PRIMARY KEY (instance_id, provider)
);

-- Start-cause handoff: a route that spawns a runner cannot know the new
-- generation's instance_id yet, so it notes the cause by port and the
-- collector consumes it when the generation first appears.
CREATE TABLE IF NOT EXISTS usage_pending_cause (
    port     INTEGER PRIMARY KEY,
    cause    TEXT NOT NULL,
    noted_ms INTEGER NOT NULL
);

-- Forensic analysis (paddock-forensics): one report per analyzed attachment
-- plus its per-analyzer findings, mirroring the `attachments` shape. A report
-- is the durable record of an image/PDF forensic run - every field the crate
-- produces is captured here so nothing is lost between analysis and the Studio.
-- `attachment_id`/`conversation_id` link it to the chat that triggered it (both
-- nullable for ad-hoc/API analysis). `sha256` is the hash of the ORIGINAL
-- analyzed bytes, so the same file re-appearing is recognizable.
CREATE TABLE IF NOT EXISTS forensic_reports (
    id              TEXT PRIMARY KEY,
    attachment_id   TEXT,
    conversation_id TEXT,
    sha256          TEXT NOT NULL,
    kind            TEXT NOT NULL,                     -- 'image' | 'pdf'
    mime            TEXT NOT NULL DEFAULT '',
    name            TEXT NOT NULL DEFAULT '',
    width           INTEGER,                           -- NULL for PDFs
    height          INTEGER,
    content_type    TEXT NOT NULL DEFAULT 'unknown',   -- photo|document|mixed|unknown
    format          TEXT NOT NULL DEFAULT '',          -- jpeg|png|tiff|... or ''
    finding_count   INTEGER NOT NULL DEFAULT 0,
    max_severity    TEXT NOT NULL DEFAULT 'info',      -- highest across findings
    risk_score      REAL NOT NULL DEFAULT 0,           -- verdict 0..1 (risk scorer)
    verdict         TEXT NOT NULL DEFAULT '',          -- one-line summary
    gpu             INTEGER NOT NULL DEFAULT 0,        -- 1 if the GPU path ran
    elapsed_ms      INTEGER NOT NULL DEFAULT 0,
    created_at      INTEGER NOT NULL,
    -- risk layer (RiskReport). Appended after created_at so the positional
    -- column order matches what ALTER TABLE ADD COLUMN gives an older DB.
    risk_level                      TEXT    NOT NULL DEFAULT 'info', -- leveled scorer verdict
    corroborating_stages            INTEGER NOT NULL DEFAULT 0,      -- independent families agreeing
    explanation_summary             TEXT    NOT NULL DEFAULT '',
    explanation_visual_review       TEXT,                            -- nullable narrative slots
    explanation_cross_corroboration TEXT,
    explanation_anti_forensics      TEXT
);
CREATE INDEX IF NOT EXISTS forensic_reports_attach ON forensic_reports(attachment_id);
CREATE INDEX IF NOT EXISTS forensic_reports_conv   ON forensic_reports(conversation_id);
CREATE INDEX IF NOT EXISTS forensic_reports_sha    ON forensic_reports(sha256);

-- Collapsed headline findings (RiskReport::key_findings): several raw findings
-- of one category merged into a single line the Studio/model leads with.
CREATE TABLE IF NOT EXISTS forensic_key_findings (
    id          TEXT PRIMARY KEY,
    report_id   TEXT NOT NULL,
    title       TEXT NOT NULL,
    description TEXT NOT NULL,
    severity    TEXT NOT NULL,
    confidence  REAL NOT NULL,
    sources     TEXT NOT NULL DEFAULT '[]',            -- JSON array of analyzer names
    region      TEXT NOT NULL DEFAULT '',              -- JSON (most specific region) or ''
    raw_count   INTEGER NOT NULL DEFAULT 0,            -- raw findings collapsed in
    seq         INTEGER NOT NULL DEFAULT 0,
    created_at  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS forensic_key_findings_report ON forensic_key_findings(report_id);

-- Per-category explanation rows (ForensicExplanation::categories): a family of
-- findings with a shared plain-language write-up.
CREATE TABLE IF NOT EXISTS forensic_explanation_categories (
    id            TEXT PRIMARY KEY,
    report_id     TEXT NOT NULL,
    name          TEXT NOT NULL,
    finding_count INTEGER NOT NULL DEFAULT 0,
    max_severity  TEXT NOT NULL DEFAULT 'info',
    explanation   TEXT NOT NULL DEFAULT '',
    finding_codes TEXT NOT NULL DEFAULT '[]',          -- JSON array of finding codes
    seq           INTEGER NOT NULL DEFAULT 0,
    created_at    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS forensic_explanation_categories_report
    ON forensic_explanation_categories(report_id);

-- One row per finding, every field of paddock_forensics::Finding preserved
-- (analyzer, code, severity, confidence, description, spatial region as JSON),
-- plus `seq` to keep the report's ordering (strongest-first).
CREATE TABLE IF NOT EXISTS forensic_findings (
    id          TEXT PRIMARY KEY,
    report_id   TEXT NOT NULL,
    analyzer    TEXT NOT NULL,                         -- ela | noise | pdf_overlay | ...
    code        TEXT NOT NULL,                         -- ela_block_outliers | ...
    severity    TEXT NOT NULL,                         -- info|low|medium|high|critical
    confidence  REAL NOT NULL,                         -- 0..1
    description TEXT NOT NULL,
    region      TEXT NOT NULL DEFAULT '',              -- JSON (bounding_box/points/mask) or ''
    seq         INTEGER NOT NULL DEFAULT 0,
    created_at  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS forensic_findings_report   ON forensic_findings(report_id);
CREATE INDEX IF NOT EXISTS forensic_findings_analyzer ON forensic_findings(analyzer);
CREATE INDEX IF NOT EXISTS forensic_findings_severity ON forensic_findings(severity);
";

impl Store {
    pub fn open(path: &PathBuf) -> Result<Self, StoreError> {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let conn = Connection::open(path)?;
        // Concurrent opens (a second process, or parallel tests on one file)
        // otherwise fail instantly with SQLITE_BUSY during the WAL flip / DDL.
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        conn.execute_batch(SCHEMA)?;
        downloads::migrate(&conn)?;
        connections::migrate(&conn)?;
        table_history::migrate(&conn)?;
        mask_history::migrate(&conn)?;
        // Hygiene migration: MODEL configuration must never live
        // in this database - it is each endpoint's servers/<port>.toml,
        // entirely. Older schemas kept an mcp_servers table and a
        // settings.web_search key here; drop the leftovers at every open so
        // nothing (code or curious operator) can read stale model config out
        // of the manager's DB.
        conn.execute("DROP TABLE IF EXISTS mcp_servers", [])?;
        conn.execute("DELETE FROM settings WHERE key = 'web_search'", [])?;
        // connectors.system arrived after the table shipped -
        // ALTER is a no-op error on DBs that already have it
        let _ = conn.execute(
            "ALTER TABLE connectors ADD COLUMN system INTEGER NOT NULL DEFAULT 0",
            [],
        );
        // artifacts.model arrived after the table shipped - ALTER
        // is a harmless error on a DB that already has it.
        let _ = conn.execute(
            "ALTER TABLE artifacts ADD COLUMN model TEXT NOT NULL DEFAULT ''",
            [],
        );
        let _ = conn.execute(
            "ALTER TABLE connectors ADD COLUMN ports TEXT NOT NULL DEFAULT '[]'",
            [],
        );
        let _ = conn.execute(
            "ALTER TABLE connectors ADD COLUMN oauth TEXT NOT NULL DEFAULT ''",
            [],
        );
        integrations::migrate(&conn)?;
        // cloud_usage.audio_seconds arrived with cloud transcription -
        // same harmless-error ALTER as the ones above.
        let _ = conn.execute("ALTER TABLE cloud_usage ADD COLUMN audio_seconds REAL", []);
        // forensic_reports risk-layer columns arrived after the table shipped
        // (RiskReport capture) - appended so an older DB's positional column
        // order matches a fresh CREATE. Harmless error where already present.
        for col in [
            "risk_level TEXT NOT NULL DEFAULT 'info'",
            "corroborating_stages INTEGER NOT NULL DEFAULT 0",
            "explanation_summary TEXT NOT NULL DEFAULT ''",
            "explanation_visual_review TEXT",
            "explanation_cross_corroboration TEXT",
            "explanation_anti_forensics TEXT",
        ] {
            let _ = conn.execute(
                &format!("ALTER TABLE forensic_reports ADD COLUMN {col}"),
                [],
            );
        }
        // attachments.metadata: the full paddock_filemeta view (JSON), shipped by
        // the runner during a chat turn and written through (task: filemeta
        // unification). NULL until analyzed; the metadata route re-reads + backfills.
        let _ = conn.execute("ALTER TABLE attachments ADD COLUMN metadata TEXT", []);
        // service_generation.aligner arrived with the forced-alignment lane
        // - same harmless-error ALTER. Bands already
        // written for an aligner runner stay NULL: the column cannot recover a
        // name nothing ever recorded, and inventing one would be worse than
        // the dash it replaces.
        let _ = conn.execute("ALTER TABLE service_generation ADD COLUMN aligner TEXT", []);
        // service_generation.boot_ms: the machine boot this band belongs to.
        // Older rows stay NULL and the sweep treats that as "boot unknown",
        // which is why it still needs the no-listener evidence below.
        let _ = conn.execute(
            "ALTER TABLE service_generation ADD COLUMN boot_ms INTEGER",
            [],
        );
        // conversations.kind arrived later - same harmless-error ALTER,
        // then a ONE-TIME backfill: every existing row is '' and would show
        // the wrong icon forever otherwise. `conversation_kind` never returns
        // '', so a row is re-read exactly once no matter how often we open.
        let fresh_kind = conn
            .execute(
                "ALTER TABLE conversations ADD COLUMN kind TEXT NOT NULL DEFAULT ''",
                [],
            )
            .is_ok();
        if fresh_kind {
            let rows: Vec<(String, String)> = {
                let mut stmt = conn.prepare("SELECT id, doc FROM conversations WHERE kind = ''")?;
                let it = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?;
                it.collect::<Result<Vec<_>, _>>()?
            };
            for (id, doc) in rows {
                // An unreadable doc is not worth failing an open over; it just
                // keeps the default and reads as a chat.
                let Ok(v) = serde_json::from_str::<Value>(&doc) else {
                    continue;
                };
                let _ = conn.execute(
                    "UPDATE conversations SET kind = ?2 WHERE id = ?1",
                    params![id, conversation_kind(&v)],
                );
            }
        }
        // The conversation list is a COVERING index, and that is the whole
        // point: every column `list_conversations` selects lives in the index,
        // so the query never touches the table. It has to be here rather than
        // in SCHEMA because `kind` is one of those columns and it arrives by
        // the ALTER just above.
        //
        // Why it matters (measured 2026-09-09, staged install on a spinning
        // RAID array, 324 conversations in a 234 MB file): ordering alone -
        // the old `conversations_updated` - still fetched each row FROM THE
        // TABLE, and rows are far apart because each one carries its whole
        // conversation as a ~240 KB `doc` (78 MB across the table). That is
        // 324 scattered reads at ~92 ms of seek each, and the first load of
        // the Studio after a boot spent 38.7 s inside this one query while
        // every other endpoint answered in ~200 ms. A covering scan reads a
        // few contiguous index pages instead. Warm it never showed, which is
        // why it survived this long: only a cold cache and a slow disk bill
        // for row fetches.
        let _ = conn.execute(
            "CREATE INDEX IF NOT EXISTS conversations_list
             ON conversations(updated_at DESC, id, title, model, pinned, created_at, kind)",
            [],
        );
        // Redundant once the above exists: same leading column, so it can
        // serve every ordering the old one did, and a second index is only
        // more write cost per conversation.
        let _ = conn.execute("DROP INDEX IF EXISTS conversations_updated", []);
        // activity rekeyed on instance_id. A PK
        // change is a table REBUILD in SQLite, not an ALTER: detect the old
        // shape by the missing column and copy rows across, stamping each old
        // generation with the synthesized id the collector also derives for
        // earlier runners - so live cursors keep lining up mid-upgrade.
        let old_shape: i64 = conn.query_row(
            "SELECT COUNT(*) FROM pragma_table_info('activity') WHERE name = 'instance_id'",
            [],
            |r| r.get(0),
        )?;
        if old_shape == 0 {
            conn.execute_batch(
                "BEGIN;
                 ALTER TABLE activity RENAME TO activity_legacy;
                 CREATE TABLE activity (
                     instance_id       TEXT NOT NULL,
                     seq               INTEGER NOT NULL,
                     port              INTEGER NOT NULL,
                     runner_started_at INTEGER NOT NULL,
                     ts_ms             INTEGER NOT NULL,
                     request_id        TEXT NOT NULL DEFAULT '',
                     endpoint          TEXT NOT NULL DEFAULT '',
                     status            INTEGER NOT NULL DEFAULT 0,
                     model             TEXT,
                     session_id        TEXT,
                     record            BLOB NOT NULL,
                     PRIMARY KEY (instance_id, seq)
                 );
                 INSERT OR IGNORE INTO activity
                     (instance_id, seq, port, runner_started_at, ts_ms, request_id,
                      endpoint, status, model, session_id, record)
                   SELECT 'legacy-' || port || '-' || runner_started_at, seq, port,
                          runner_started_at, ts_ms, request_id, endpoint, status,
                          model, session_id, record
                   FROM activity_legacy;
                 DROP TABLE activity_legacy;
                 CREATE INDEX IF NOT EXISTS activity_ts ON activity(ts_ms DESC);
                 COMMIT;",
            )?;
        }
        Ok(Store {
            conn: Mutex::new(conn),
            cloud_credentials: Default::default(),
            connector_credentials: Default::default(),
            connector_credential_updates: Default::default(),
        })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.conn.lock().unwrap_or_else(|e| e.into_inner())
    }

    // ── conversations ───────────────────────────────────────────────────────

    /// Summaries only (id/title/model/kind/pinned/timestamps), newest first.
    ///
    /// `kind` rides along because the Studio cannot work it out from what this
    /// returns - see [`conversation_kind`].
    pub fn list_conversations(&self) -> Result<Vec<Value>, StoreError> {
        let conn = self.lock();
        let mut stmt = conn.prepare(
            "SELECT id, title, model, pinned, updated_at, created_at, kind
             FROM conversations ORDER BY updated_at DESC",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(json!({
                "id": r.get::<_, String>(0)?,
                "title": r.get::<_, String>(1)?,
                "model": r.get::<_, String>(2)?,
                "pinned": r.get::<_, i64>(3)? != 0,
                "updatedAt": r.get::<_, i64>(4)?,
                "createdAt": r.get::<_, i64>(5)?,
                "kind": r.get::<_, String>(6)?,
            }))
        })?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    pub fn get_conversation(&self, id: &str) -> Result<Option<Value>, StoreError> {
        let conn = self.lock();
        let doc: Option<String> = conn
            .query_row(
                "SELECT doc FROM conversations WHERE id = ?1",
                params![id],
                |r| r.get(0),
            )
            .optional()?;
        match doc {
            Some(s) => Ok(Some(
                serde_json::from_str(&s).map_err(|e| StoreError::Bad(e.to_string()))?,
            )),
            None => Ok(None),
        }
    }

    pub fn put_conversation(&self, doc: &Value) -> Result<(), StoreError> {
        let id = doc
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| StoreError::Bad("missing id".into()))?;
        let title = doc
            .get("title")
            .and_then(Value::as_str)
            .unwrap_or("Untitled");
        let model = doc.get("model").and_then(Value::as_str).unwrap_or("");
        let pinned = i64::from(doc.get("pinned").and_then(Value::as_bool).unwrap_or(false));
        let created = doc
            .get("createdAt")
            .and_then(Value::as_i64)
            .unwrap_or_else(now_ms);
        let updated = doc
            .get("updatedAt")
            .and_then(Value::as_i64)
            .unwrap_or(created);
        let kind = conversation_kind(doc);
        let body = serde_json::to_string(doc).map_err(|e| StoreError::Bad(e.to_string()))?;
        let conn = self.lock();
        conn.execute(
            "INSERT INTO conversations (id, title, model, pinned, created_at, updated_at, kind, doc)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8)
             ON CONFLICT(id) DO UPDATE SET
               title=?2, model=?3, pinned=?4, updated_at=?6, kind=?7, doc=?8",
            params![id, title, model, pinned, created, updated, kind, body],
        )?;
        Ok(())
    }

    pub fn delete_conversation(&self, id: &str) -> Result<(), StoreError> {
        let conn = self.lock();
        conn.execute("DELETE FROM conversations WHERE id = ?1", params![id])?;
        // Cascade: drop the conversation's attachments so blobs don't orphan.
        conn.execute(
            "DELETE FROM attachments WHERE conversation_id = ?1",
            params![id],
        )?;
        // Same for artifacts - versions first, since they key off the parent.
        conn.execute(
            "DELETE FROM artifact_versions WHERE artifact_id IN
               (SELECT id FROM artifacts WHERE conversation_id = ?1)",
            params![id],
        )?;
        conn.execute(
            "DELETE FROM artifacts WHERE conversation_id = ?1",
            params![id],
        )?;
        Ok(())
    }

    // ── attachments (image/doc bytes, kept out of the conversation doc) ────────

    /// Store an attachment's original ("view") bytes; returns nothing (the id is
    /// caller-supplied so the client can reference it before the save round-trip).
    pub fn put_attachment(
        &self,
        id: &str,
        conversation_id: Option<&str>,
        mime: &str,
        name: &str,
        width: Option<i64>,
        height: Option<i64>,
        bytes: &[u8],
    ) -> Result<(), StoreError> {
        self.lock().execute(
            "INSERT INTO attachments (id, conversation_id, mime, name, width, height, bytes, created_at)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8)
             ON CONFLICT(id) DO UPDATE SET
               conversation_id=?2, mime=?3, name=?4, width=?5, height=?6, bytes=?7",
            params![id, conversation_id, mime, name, width, height, bytes, now_ms()],
        )?;
        Ok(())
    }

    /// Fetch an attachment's bytes with its stored identity - mime AND file
    /// name. The metadata route reports both back, so a client
    /// that asks for a file's metadata does not need a second call to learn
    /// which file it asked about.
    pub fn get_attachment_named(
        &self,
        id: &str,
    ) -> Result<Option<(String, String, Vec<u8>)>, StoreError> {
        let conn = self.lock();
        let row = conn
            .query_row(
                "SELECT mime, name, bytes FROM attachments WHERE id = ?1",
                params![id],
                |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, Vec<u8>>(2)?,
                    ))
                },
            )
            .optional()?;
        Ok(row)
    }

    /// Fetch an attachment's bytes + mime for streaming.
    pub fn get_attachment(&self, id: &str) -> Result<Option<(String, Vec<u8>)>, StoreError> {
        let conn = self.lock();
        let row = conn
            .query_row(
                "SELECT mime, bytes FROM attachments WHERE id = ?1",
                params![id],
                |r| Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?)),
            )
            .optional()?;
        Ok(row)
    }

    /// An attachment's identity + byte size + stored metadata JSON, without
    /// materializing the blob (`length(bytes)` reads the record header, not the
    /// data). The metadata route serves from `metadata` when present so it never
    /// re-parses - the write-through cache from the runner-shipped file metadata.
    pub fn attachment_lite(
        &self,
        id: &str,
    ) -> Result<Option<(String, String, i64, Option<String>)>, StoreError> {
        let conn = self.lock();
        let row = conn
            .query_row(
                "SELECT mime, name, length(bytes), metadata FROM attachments WHERE id = ?1",
                params![id],
                |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, i64>(2)?,
                        r.get::<_, Option<String>>(3)?,
                    ))
                },
            )
            .optional()?;
        Ok(row)
    }

    /// Store an attachment's full file-metadata JSON (paddock_filemeta view). A
    /// no-op if the attachment does not exist. Idempotent - overwrites, since the
    /// bytes are immutable so the metadata is stable.
    pub fn set_attachment_metadata(&self, id: &str, json: &str) -> Result<(), StoreError> {
        self.lock().execute(
            "UPDATE attachments SET metadata = ?2 WHERE id = ?1",
            params![id, json],
        )?;
        Ok(())
    }

    // ── forensic reports (image/PDF signal extraction; paddock-forensics) ────

    /// Persist one forensic analysis (the report row + all its findings) in a
    /// single transaction; returns the generated report id. Aggregate columns
    /// (`finding_count`, `max_severity`) are derived here from the findings so a
    /// caller cannot forget to keep them in sync with the child rows.
    ///
    /// The id is caller-visible (it feeds the Studio's forensics panel and the
    /// `analyze_document_forensics` tool result), so it is short and prefixed
    /// rather than a bare UUID, matching `artifacts`.
    pub fn save_forensic_report(&self, rep: &NewForensicReport) -> Result<String, StoreError> {
        let id = format!(
            "fr_{}",
            Uuid::new_v4().simple().to_string()[..12].to_owned()
        );
        let now = now_ms();
        let finding_count = rep.findings.len() as i64;
        // Highest severity present drives the report-level badge; "info" when
        // the analysis ran clean (no findings).
        let max_severity = rep
            .findings
            .iter()
            .max_by_key(|f| severity_rank(&f.severity))
            .map(|f| f.severity.clone())
            .unwrap_or_else(|| "info".to_string());
        // The scorer always names a level; normalize a caller that left it
        // blank (e.g. a report built before the risk pass) to the clean word so
        // the column is never an empty severity.
        let risk_level = if rep.risk_level.is_empty() {
            "info"
        } else {
            rep.risk_level.as_str()
        };

        let mut conn = self.lock();
        let tx = conn.transaction()?;
        tx.execute(
            "INSERT INTO forensic_reports
               (id, attachment_id, conversation_id, sha256, kind, mime, name,
                width, height, content_type, format, finding_count, max_severity,
                risk_score, verdict, gpu, elapsed_ms, created_at,
                risk_level, corroborating_stages, explanation_summary,
                explanation_visual_review, explanation_cross_corroboration,
                explanation_anti_forensics)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,
                     ?19,?20,?21,?22,?23,?24)",
            params![
                id,
                rep.attachment_id,
                rep.conversation_id,
                rep.sha256,
                rep.kind,
                rep.mime,
                rep.name,
                rep.width,
                rep.height,
                rep.content_type,
                rep.format,
                finding_count,
                max_severity,
                rep.risk_score,
                rep.verdict,
                rep.gpu as i64,
                rep.elapsed_ms,
                now,
                risk_level,
                rep.corroborating_stages,
                rep.explanation_summary,
                rep.explanation_visual_review,
                rep.explanation_cross_corroboration,
                rep.explanation_anti_forensics,
            ],
        )?;
        for (seq, f) in rep.findings.iter().enumerate() {
            let fid = format!(
                "ff_{}",
                Uuid::new_v4().simple().to_string()[..12].to_owned()
            );
            tx.execute(
                "INSERT INTO forensic_findings
                   (id, report_id, analyzer, code, severity, confidence,
                    description, region, seq, created_at)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",
                params![
                    fid,
                    id,
                    f.analyzer,
                    f.code,
                    f.severity,
                    f.confidence,
                    f.description,
                    f.region,
                    seq as i64,
                    now,
                ],
            )?;
        }
        for (seq, k) in rep.key_findings.iter().enumerate() {
            let kid = format!(
                "fk_{}",
                Uuid::new_v4().simple().to_string()[..12].to_owned()
            );
            let sources = serde_json::to_string(&k.sources).unwrap_or_else(|_| "[]".into());
            tx.execute(
                "INSERT INTO forensic_key_findings
                   (id, report_id, title, description, severity, confidence,
                    sources, region, raw_count, seq, created_at)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)",
                params![
                    kid,
                    id,
                    k.title,
                    k.description,
                    k.severity,
                    k.confidence,
                    sources,
                    k.region,
                    k.count,
                    seq as i64,
                    now,
                ],
            )?;
        }
        for (seq, c) in rep.explanation_categories.iter().enumerate() {
            let cid = format!(
                "fx_{}",
                Uuid::new_v4().simple().to_string()[..12].to_owned()
            );
            let codes = serde_json::to_string(&c.finding_codes).unwrap_or_else(|_| "[]".into());
            tx.execute(
                "INSERT INTO forensic_explanation_categories
                   (id, report_id, name, finding_count, max_severity,
                    explanation, finding_codes, seq, created_at)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9)",
                params![
                    cid,
                    id,
                    c.name,
                    c.finding_count,
                    c.max_severity,
                    c.explanation,
                    codes,
                    seq as i64,
                    now,
                ],
            )?;
        }
        tx.commit()?;
        Ok(id)
    }

    /// A full report (row columns + its findings, seq-ordered) as JSON, or
    /// `None` if no such report.
    pub fn get_forensic_report(&self, id: &str) -> Result<Option<Value>, StoreError> {
        let conn = self.lock();
        let report = conn
            .query_row(
                "SELECT id, attachment_id, conversation_id, sha256, kind, mime, name,
                        width, height, content_type, format, finding_count, max_severity,
                        risk_score, verdict, gpu, elapsed_ms, created_at,
                        risk_level, corroborating_stages, explanation_summary,
                        explanation_visual_review, explanation_cross_corroboration,
                        explanation_anti_forensics
                 FROM forensic_reports WHERE id = ?1",
                params![id],
                forensic_report_row,
            )
            .optional()?;
        let Some(mut report) = report else {
            return Ok(None);
        };
        let mut stmt = conn.prepare(
            "SELECT analyzer, code, severity, confidence, description, region, seq
             FROM forensic_findings WHERE report_id = ?1 ORDER BY seq ASC",
        )?;
        let findings: Vec<Value> = stmt
            .query_map(params![id], forensic_finding_row)?
            .collect::<Result<Vec<_>, _>>()?;
        report["findings"] = Value::Array(findings);

        // Collapsed headline findings (seq-ordered, strongest-first).
        let mut kstmt = conn.prepare(
            "SELECT title, description, severity, confidence, sources, region,
                    raw_count, seq
             FROM forensic_key_findings WHERE report_id = ?1 ORDER BY seq ASC",
        )?;
        let key_findings: Vec<Value> = kstmt
            .query_map(params![id], forensic_key_finding_row)?
            .collect::<Result<Vec<_>, _>>()?;
        report["key_findings"] = Value::Array(key_findings);

        // Per-category explanation breakdown, nested under the narrative slots
        // the report row already carries.
        let mut cstmt = conn.prepare(
            "SELECT name, finding_count, max_severity, explanation, finding_codes, seq
             FROM forensic_explanation_categories WHERE report_id = ?1 ORDER BY seq ASC",
        )?;
        let categories: Vec<Value> = cstmt
            .query_map(params![id], forensic_explanation_category_row)?
            .collect::<Result<Vec<_>, _>>()?;
        report["explanation"]["categories"] = Value::Array(categories);
        Ok(Some(report))
    }

    /// Report summaries (no findings) for one conversation, newest first - the
    /// Studio's per-conversation forensics list.
    pub fn list_forensic_reports(&self, conversation_id: &str) -> Result<Vec<Value>, StoreError> {
        let conn = self.lock();
        let mut stmt = conn.prepare(
            "SELECT id, attachment_id, conversation_id, sha256, kind, mime, name,
                    width, height, content_type, format, finding_count, max_severity,
                    risk_score, verdict, gpu, elapsed_ms, created_at,
                    risk_level, corroborating_stages, explanation_summary,
                    explanation_visual_review, explanation_cross_corroboration,
                    explanation_anti_forensics
             FROM forensic_reports WHERE conversation_id = ?1 ORDER BY created_at DESC",
        )?;
        let rows: Vec<Value> = stmt
            .query_map(params![conversation_id], forensic_report_row)?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// The most recent report for a given attachment id (a re-analysis writes a
    /// new row rather than overwriting, so history is preserved). `None` if the
    /// attachment was never analyzed.
    pub fn latest_forensic_report_for_attachment(
        &self,
        attachment_id: &str,
    ) -> Result<Option<Value>, StoreError> {
        let conn = self.lock();
        let id = conn
            .query_row(
                "SELECT id FROM forensic_reports WHERE attachment_id = ?1
                 ORDER BY created_at DESC LIMIT 1",
                params![attachment_id],
                |r| r.get::<_, String>(0),
            )
            .optional()?;
        drop(conn);
        match id {
            Some(id) => self.get_forensic_report(&id),
            None => Ok(None),
        }
    }

    // ── artifacts (model-authored content, versioned) ─────────────

    /// Create an artifact at version 1 and return its id. The id is short and
    /// prefixed rather than a bare UUID because the model echoes it back in
    /// every later call - it lands in the transcript, so it may as well be
    /// readable.
    pub fn create_artifact(
        &self,
        conversation_id: &str,
        kind: &str,
        title: &str,
        language: &str,
        model: &str,
        content: &str,
    ) -> Result<String, StoreError> {
        let id = format!(
            "art_{}",
            Uuid::new_v4().simple().to_string()[..12].to_owned()
        );
        let now = now_ms();
        let conn = self.lock();
        conn.execute(
            "INSERT INTO artifacts (id, conversation_id, kind, title, language, model, created_at, updated_at)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?7)",
            params![id, conversation_id, kind, title, language, model, now],
        )?;
        conn.execute(
            "INSERT INTO artifact_versions (artifact_id, seq, op, content, created_at)
             VALUES (?1, 1, 'create', ?2, ?3)",
            params![id, content, now],
        )?;
        Ok(id)
    }

    /// The artifact's kind + the content at `seq` (latest when None).
    pub fn artifact_content(
        &self,
        id: &str,
        seq: Option<i64>,
    ) -> Result<Option<(String, String, i64)>, StoreError> {
        let conn = self.lock();
        let sql = match seq {
            Some(_) => {
                "SELECT a.kind, v.content, v.seq FROM artifacts a
                 JOIN artifact_versions v ON v.artifact_id = a.id
                 WHERE a.id = ?1 AND v.seq = ?2"
            }
            // ?2 is bound either way so the two arms share one query_row call;
            // the latest arm ignores it.
            None => {
                "SELECT a.kind, v.content, v.seq FROM artifacts a
                 JOIN artifact_versions v ON v.artifact_id = a.id
                 WHERE a.id = ?1 AND ?2 IS NULL ORDER BY v.seq DESC LIMIT 1"
            }
        };
        let row = conn
            .query_row(sql, params![id, seq], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, i64>(2)?,
                ))
            })
            .optional()?;
        Ok(row)
    }

    /// Append a new version and return its seq. Content identical to the
    /// current version is not appended (the seq comes back unchanged) - a
    /// rewrite that changes nothing should not manufacture a version the user
    /// then has to scroll past.
    pub fn append_artifact_version(
        &self,
        id: &str,
        op: &str,
        content: &str,
    ) -> Result<i64, StoreError> {
        self.append_artifact_version_if_current(id, op, content, None)
    }

    /// Check the editor's revision and append under one database transaction.
    /// A model update between the editor's GET and PUT must not be overwritten.
    pub fn append_artifact_version_if_current(
        &self,
        id: &str,
        op: &str,
        content: &str,
        expected: Option<i64>,
    ) -> Result<i64, StoreError> {
        let now = now_ms();
        let mut guard = self.lock();
        let conn = guard.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let cur: Option<(i64, String)> = conn
            .query_row(
                "SELECT seq, content FROM artifact_versions
                 WHERE artifact_id = ?1 ORDER BY seq DESC LIMIT 1",
                params![id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        let Some((seq, prev)) = cur else {
            return Err(StoreError::Bad(format!("no such artifact: {id}")));
        };
        if expected.is_some_and(|expected| expected != seq) {
            return Err(StoreError::Conflict("The artifact changed while you edited it. Your draft is retained; review the latest version before saving.".into()));
        }
        if prev == content {
            return Ok(seq);
        }
        let next = seq + 1;
        conn.execute(
            "INSERT INTO artifact_versions (artifact_id, seq, op, content, created_at)
             VALUES (?1,?2,?3,?4,?5)",
            params![id, next, op, content, now],
        )?;
        conn.execute(
            "UPDATE artifacts SET updated_at = ?2 WHERE id = ?1",
            params![id, now],
        )?;
        conn.commit()?;
        Ok(next)
    }

    /// Metadata for every artifact in a conversation, newest first. Bodies are
    /// deliberately absent - the panel fetches one version at a time.
    pub fn list_artifacts(&self, conversation_id: &str) -> Result<Vec<Value>, StoreError> {
        let conn = self.lock();
        let mut stmt = conn.prepare(
            "SELECT a.id, a.kind, a.title, a.language, a.created_at, a.updated_at,
                    (SELECT MAX(seq) FROM artifact_versions v WHERE v.artifact_id = a.id),
                    a.model
             FROM artifacts a WHERE a.conversation_id = ?1 ORDER BY a.updated_at DESC",
        )?;
        let rows = stmt.query_map(params![conversation_id], |r| {
            Ok(json!({
                "id": r.get::<_, String>(0)?,
                "kind": r.get::<_, String>(1)?,
                "title": r.get::<_, String>(2)?,
                "language": r.get::<_, String>(3)?,
                "createdAt": r.get::<_, i64>(4)?,
                "updatedAt": r.get::<_, i64>(5)?,
                "versions": r.get::<_, Option<i64>>(6)?.unwrap_or(0),
                "model": r.get::<_, String>(7)?,
            }))
        })?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(StoreError::from)
    }

    /// Every version of one artifact (seq + op + timestamp), oldest first.
    pub fn artifact_versions(&self, id: &str) -> Result<Vec<Value>, StoreError> {
        let conn = self.lock();
        let mut stmt = conn.prepare(
            "SELECT seq, op, created_at, LENGTH(content) FROM artifact_versions
             WHERE artifact_id = ?1 ORDER BY seq",
        )?;
        let rows = stmt.query_map(params![id], |r| {
            Ok(json!({
                "seq": r.get::<_, i64>(0)?,
                "op": r.get::<_, String>(1)?,
                "createdAt": r.get::<_, i64>(2)?,
                "bytes": r.get::<_, i64>(3)?,
            }))
        })?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(StoreError::from)
    }

    /// Which conversation an artifact belongs to - the authorization check for
    /// every artifact route (a panel may only read its own chat's artifacts).
    pub fn artifact_conversation(&self, id: &str) -> Result<Option<String>, StoreError> {
        let conn = self.lock();
        let row = conn
            .query_row(
                "SELECT conversation_id FROM artifacts WHERE id = ?1",
                params![id],
                |r| r.get::<_, String>(0),
            )
            .optional()?;
        Ok(row)
    }

    // ── api keys ─────────────────────────────────────────────────────────────

    /// Create a named key. Returns `(public_record, plaintext_key)` - the
    /// plaintext is shown once and never stored (only its hash).
    pub fn create_api_key(&self, name: &str) -> Result<(Value, String), StoreError> {
        let id = Uuid::new_v4().to_string();
        let key = format!("pk-{}", Uuid::new_v4().simple());
        let prefix = key.chars().take(9).collect::<String>();
        let created = now_ms();
        self.lock().execute(
            "INSERT INTO api_keys (id, name, key_hash, prefix, created_at) VALUES (?1,?2,?3,?4,?5)",
            params![id, name, hash_key(&key), prefix, created],
        )?;
        let record = json!({ "id": id, "name": name, "prefix": prefix, "createdAt": created, "revoked": false });
        Ok((record, key))
    }

    pub fn list_api_keys(&self) -> Result<Vec<Value>, StoreError> {
        let conn = self.lock();
        let mut stmt = conn.prepare(
            "SELECT id, name, prefix, created_at, last_used_at, revoked
             FROM api_keys ORDER BY created_at DESC",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(json!({
                "id": r.get::<_, String>(0)?,
                "name": r.get::<_, String>(1)?,
                "prefix": r.get::<_, String>(2)?,
                "createdAt": r.get::<_, i64>(3)?,
                "lastUsedAt": r.get::<_, Option<i64>>(4)?,
                "revoked": r.get::<_, i64>(5)? != 0,
            }))
        })?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    pub fn revoke_api_key(&self, id: &str) -> Result<(), StoreError> {
        self.lock()
            .execute("UPDATE api_keys SET revoked = 1 WHERE id = ?1", params![id])?;
        Ok(())
    }

    /// True if the key matches a non-revoked stored key; also stamps last_used.
    pub fn verify_api_key(&self, key: &str) -> Result<bool, StoreError> {
        let hash = hash_key(key);
        let conn = self.lock();
        let id: Option<String> = conn
            .query_row(
                "SELECT id FROM api_keys WHERE key_hash = ?1 AND revoked = 0",
                params![hash],
                |r| r.get(0),
            )
            .optional()?;
        if let Some(id) = id {
            conn.execute(
                "UPDATE api_keys SET last_used_at = ?2 WHERE id = ?1",
                params![id, now_ms()],
            )?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Number of non-revoked keys (used by the auth policy).
    pub fn active_key_count(&self) -> Result<i64, StoreError> {
        Ok(self
            .lock()
            .query_row("SELECT COUNT(*) FROM api_keys WHERE revoked = 0", [], |r| {
                r.get(0)
            })?)
    }

    // NOTE: there are deliberately no MCP-server methods here.
    // MCP servers and the web-search key are MODEL configuration - they live
    // in each endpoint's servers/<port>.toml, never in this database.

    // ── cloud endpoints (BYO-key external providers) ─────────────────────────
    //
    // Unlike MCP/web-search these are not model config for a runner - no
    // runner ever sees them. They configure the MANAGER's own client role
    // (doc §1: "BYO-key external providers"), so this database is exactly
    // where they belong.

    /// Public row shape: everything except the key itself (`hasKey` only).
    fn cloud_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<Value> {
        let models: String = r.get(4)?;
        Ok(json!({
            "id": r.get::<_, String>(0)?,
            "name": r.get::<_, String>(1)?,
            "kind": r.get::<_, String>(2)?,
            "baseUrl": r.get::<_, String>(3)?,
            "models": serde_json::from_str::<Value>(&models).unwrap_or_else(|_| json!([])),
            "hasKey": !r.get::<_, String>(5)?.is_empty(),
            "createdAt": r.get::<_, i64>(6)?,
            "revision": r.get::<_, i64>(7)?,
            "allowUnauthenticated": r.get::<_, bool>(8)?,
            "credentialStorage": if r.get::<_, String>(5)?.starts_with("paddock-keychain:v1:") { "keychain" } else { "database" },
        }))
    }

    pub fn list_cloud_endpoints(&self) -> Result<Vec<Value>, StoreError> {
        let conn = self.lock();
        let mut stmt = conn.prepare(
            "SELECT id, name, kind, base_url, models, api_key, created_at, revision, allow_unauthenticated
             FROM cloud_endpoints ORDER BY created_at ASC",
        )?;
        let rows = stmt.query_map([], |r| {
            let mut row = Self::cloud_row(r)?;
            row["credentialReady"] =
                json!(self.cloud_credentials.available(&r.get::<_, String>(5)?));
            Ok(row)
        })?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    /// Create from a client doc; returns the public row. The key is stored
    /// verbatim (it has to be sent to the provider), never returned.
    pub fn create_cloud_endpoint(&self, doc: &Value) -> Result<Value, StoreError> {
        let id = Uuid::new_v4().to_string();
        let name = doc
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_owned();
        let kind = doc
            .get("kind")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();
        let base = doc
            .get("baseUrl")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .trim_end_matches('/')
            .to_owned();
        if name.is_empty() || base.is_empty() {
            return Err(StoreError::Bad("name and baseUrl are required".into()));
        }
        if !matches!(kind.as_str(), "openai" | "openai-compat" | "anthropic") {
            return Err(StoreError::Bad(format!("unknown kind \"{kind}\"")));
        }
        let key = doc
            .get("apiKey")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_owned();
        let models = doc.get("models").cloned().unwrap_or_else(|| json!([]));
        let models_s = serde_json::to_string(&models).unwrap_or_else(|_| "[]".into());
        let created = now_ms();
        self.lock().execute(
            "INSERT INTO cloud_endpoints (id, name, kind, base_url, api_key, models, created_at)
             VALUES (?1,?2,?3,?4,?5,?6,?7)",
            params![id, name, kind, base, key, models_s, created],
        )?;
        Ok(json!({
            "id": id, "name": name, "kind": kind, "baseUrl": base,
            "models": models, "hasKey": !key.is_empty(), "createdAt": created,
        }))
    }

    /// Patch name/baseUrl/models; `apiKey` only when present AND non-empty
    /// (the edit form round-trips without the key, so an untouched field must
    /// not blank the stored one).
    pub fn update_cloud_endpoint(&self, id: &str, doc: &Value) -> Result<(), StoreError> {
        // One statement is atomic even for the web form's partial patch. Native
        // reviews also observe this revision, including key-only edits.
        let name = doc.get("name").and_then(Value::as_str).map(str::trim);
        let base = doc
            .get("baseUrl")
            .and_then(Value::as_str)
            .map(|s| s.trim().trim_end_matches('/'));
        let models = doc
            .get("models")
            .map(serde_json::to_string)
            .transpose()
            .map_err(|e| StoreError::Bad(e.to_string()))?;
        let key = doc
            .get("apiKey")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty());
        let mut conn = self.lock();
        let tx = conn.transaction()?;
        let (old, old_base): (String, String) = tx.query_row(
            "SELECT api_key, base_url FROM cloud_endpoints WHERE id=?1",
            [id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?;
        if base.is_some_and(|base| base != old_base) && key.is_none() && !old.is_empty() {
            return Err(StoreError::Bad(
                "Enter the key again when changing a connection's destination.".into(),
            ));
        }
        // A later edit in web Studio must not downgrade a native Keychain
        // credential to plaintext SQLite storage.
        let protected = key
            .filter(|_| old.starts_with("paddock-keychain:v1:"))
            .map(crate::credentials::protect)
            .transpose()
            .map_err(StoreError::Bad)?;
        let result = (|| -> Result<(), StoreError> {
            tx.execute(
            "UPDATE cloud_endpoints SET name=coalesce(?2,name),base_url=coalesce(?3,base_url),models=coalesce(?4,models),api_key=coalesce(?5,api_key),revision=revision+1 WHERE id=?1",
            params![id,name,base,models,protected.as_deref().or(key)])?;
            tx.commit()?;
            Ok(())
        })();
        drop(conn);
        if let Err(error) = result {
            if let Some(protected) = protected {
                crate::credentials::retire(&protected);
            }
            return Err(error);
        }
        if key.is_some() {
            if let (Some(protected), Some(key)) = (protected.as_deref(), key) {
                self.cloud_credentials.remember(protected, key);
            }
            self.cloud_credentials.forget(&old);
            crate::credentials::retire(&old);
        }
        Ok(())
    }

    pub fn delete_cloud_endpoint(&self, id: &str) -> Result<(), StoreError> {
        let mut conn = self.lock();
        let tx = conn.transaction()?;
        let old: Option<String> = tx
            .query_row(
                "SELECT api_key FROM cloud_endpoints WHERE id=?1",
                [id],
                |r| r.get(0),
            )
            .optional()?;
        tx.execute("DELETE FROM cloud_endpoints WHERE id=?1", [id])?;
        tx.commit()?;
        drop(conn);
        if let Some(old) = old {
            self.cloud_credentials.forget(&old);
            crate::credentials::retire(&old);
        }
        Ok(())
    }

    // ── connectors (the Studio's personal MCP library) ──────────────────────

    fn connector_row(row: &rusqlite::Row) -> rusqlite::Result<Value> {
        let headers: String = row.get(3)?;
        let ports: String = row.get(6)?;
        let oauth: String = row.get(7)?;
        Ok(json!({
            "id": row.get::<_, String>(0)?,
            "label": row.get::<_, String>(1)?,
            "url": row.get::<_, String>(2)?,
            "headers": headers,
            "registryKey": row.get::<_, String>(4)?,
            "system": row.get::<_, i64>(5)? != 0,
            "ports": serde_json::from_str::<Value>(&ports).unwrap_or_else(|_| json!([])),
            // INTERNAL: tokens + endpoints; the API layer strips this and
            // exposes only `connected` (+ the bearer merged into headers)
            "oauth": oauth,
            "createdAt": row.get::<_, i64>(8)?,
            "revision": row.get::<_, i64>(9)?,
        }))
    }

    pub fn list_connectors(&self) -> Result<Vec<Value>, StoreError> {
        let conn = self.lock();
        let mut stmt = conn.prepare(
            "SELECT id, label, url, headers, registry_key, system, ports, oauth, created_at, revision
             FROM connectors ORDER BY created_at ASC",
        )?;
        let rows = stmt.query_map([], Self::connector_row)?;
        let rows = rows.collect::<Result<Vec<_>, _>>()?;
        drop(stmt);
        drop(conn);
        rows.into_iter()
            .map(|row| self.resolve_connector(row))
            .collect()
    }

    pub fn get_connector(&self, id: &str) -> Result<Option<Value>, StoreError> {
        let conn = self.lock();
        let mut stmt = conn.prepare(
            "SELECT id, label, url, headers, registry_key, system, ports, oauth, created_at, revision
             FROM connectors WHERE id = ?1",
        )?;
        let mut rows = stmt.query_map(params![id], Self::connector_row)?;
        let row = rows.next().transpose()?;
        drop(rows);
        drop(stmt);
        drop(conn);
        row.map(|row| self.resolve_connector(row)).transpose()
    }

    /// Store the OAuth state blob ('' clears - disconnect).
    pub fn set_connector_oauth(&self, id: &str, oauth_json: &str) -> Result<(), StoreError> {
        let revision: i64 =
            self.lock()
                .query_row("SELECT revision FROM connectors WHERE id=?1", [id], |r| {
                    r.get(0)
                })?;
        self.set_connector_oauth_checked(id, revision as u64, oauth_json)
    }

    /// The scope: `all` = every model incl. future ones (spawn inherits);
    /// `ports` = exactly these endpoints. Both empty = per-chat only.
    pub fn set_connector_scope(
        &self,
        id: &str,
        all: bool,
        ports: &[u16],
    ) -> Result<(), StoreError> {
        let ports_s = serde_json::to_string(ports).unwrap_or_else(|_| "[]".into());
        let n = self.lock().execute(
            "UPDATE connectors SET system = ?2, ports = ?3, revision = revision + 1 WHERE id = ?1",
            params![id, all as i64, ports_s],
        )?;
        if n == 0 {
            return Err(StoreError::Bad(format!("no connector with id {id}")));
        }
        Ok(())
    }

    /// The label doubles as the wire `server_label` (it names the tool bundle
    /// to the model), so it is validated to the conservative slug set and must
    /// be unique - two selected connectors with one label would collide in a
    /// single request's tool list.
    fn connector_fields(doc: &Value) -> Result<(String, String, String), StoreError> {
        let label = doc
            .get("label")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_owned();
        if label.is_empty()
            || label.len() > 64
            || !label
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            return Err(StoreError::Bad(
                "label must be 1-64 characters of letters, digits, - or _".into(),
            ));
        }
        let url = doc
            .get("url")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_owned();
        if !(url.starts_with("http://") || url.starts_with("https://")) {
            return Err(StoreError::Bad(
                "url must start with http:// or https://".into(),
            ));
        }
        let headers = doc.get("headers").cloned().unwrap_or_else(|| json!({}));
        if !headers.is_object()
            || headers
                .as_object()
                .is_some_and(|o| o.values().any(|v| !v.is_string()))
        {
            return Err(StoreError::Bad(
                "headers must be an object of string values".into(),
            ));
        }
        let headers_s = serde_json::to_string(&headers).unwrap_or_else(|_| "{}".into());
        Ok((label, url, headers_s))
    }

    pub fn create_connector(&self, doc: &Value) -> Result<Value, StoreError> {
        let (label, url, headers_s) = Self::connector_fields(doc)?;
        let registry_key = doc
            .get("registryKey")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_owned();
        let conn = self.lock();
        let dup: i64 = conn.query_row(
            "SELECT COUNT(*) FROM connectors WHERE label = ?1",
            params![label],
            |r| r.get(0),
        )?;
        if dup > 0 {
            return Err(StoreError::Bad(format!(
                "a connector labeled \"{label}\" already exists"
            )));
        }
        let id = Uuid::new_v4().to_string();
        let created = now_ms();
        conn.execute(
            "INSERT INTO connectors (id, label, url, headers, registry_key, created_at)
             VALUES (?1,?2,?3,?4,?5,?6)",
            params![id, label, url, headers_s, registry_key, created],
        )?;
        Ok(json!({
            "id": id, "label": label, "url": url,
            "headers": serde_json::from_str::<Value>(&headers_s).unwrap_or_else(|_| json!({})),
            "registryKey": registry_key, "createdAt": created,
        }))
    }

    /// Full-row update (the edit form always round-trips every field - headers
    /// included, since the Studio reads them back anyway).
    pub fn update_connector(&self, id: &str, doc: &Value) -> Result<(), StoreError> {
        if self.connector_uses_keychain(id) {
            let revision: i64 = self.lock().query_row(
                "SELECT revision FROM connectors WHERE id=?1",
                [id],
                |r| r.get(0),
            )?;
            let mut native = doc.clone();
            native["id"] = json!(id);
            if native.get("revision").is_none() {
                native["revision"] = json!(revision);
            }
            return self.save_native_connector(&native).map(|_| ());
        }
        let (label, url, headers_s) = Self::connector_fields(doc)?;
        let conn = self.lock();
        let dup: i64 = conn.query_row(
            "SELECT COUNT(*) FROM connectors WHERE label = ?1 AND id != ?2",
            params![label, id],
            |r| r.get(0),
        )?;
        if dup > 0 {
            return Err(StoreError::Bad(format!(
                "a connector labeled \"{label}\" already exists"
            )));
        }
        let n = conn.execute(
            "UPDATE connectors SET label = ?2, oauth = CASE WHEN url != ?3 THEN '' ELSE oauth END,
             url = ?3, headers = ?4, revision = revision + 1 WHERE id = ?1
             AND (?5 IS NULL OR revision = ?5)",
            params![
                id,
                label,
                url,
                headers_s,
                doc.get("revision").and_then(Value::as_i64)
            ],
        )?;
        if n == 0 {
            return Err(StoreError::Bad(format!("no connector with id {id}")));
        }
        Ok(())
    }

    pub fn delete_connector(&self, id: &str) -> Result<(), StoreError> {
        let _publication = self
            .connector_credential_updates
            .lock()
            .expect("connector publication");
        let mut conn = self.lock();
        let tx = conn.transaction()?;
        let stored: Option<(String, String)> = tx
            .query_row(
                "SELECT headers,oauth FROM connectors WHERE id=?1",
                [id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        tx.execute("DELETE FROM connectors WHERE id = ?1", params![id])?;
        tx.commit()?;
        drop(conn);
        if let Some((headers, oauth)) = stored {
            self.connector_credentials.forget(&headers);
            self.connector_credentials.forget(&oauth);
            crate::credentials::retire(&headers);
            crate::credentials::retire(&oauth);
        }
        Ok(())
    }

    // ── cloud usage ledger (per-request tokens + provider-reported cost) ────

    /// One completed cloud request. `cost` is the provider's own number
    /// (OpenRouter reports it per response); NULL when the provider doesn't.
    pub fn insert_cloud_usage(
        &self,
        endpoint: &str,
        model: &str,
        input_tokens: u64,
        output_tokens: u64,
        reasoning_tokens: u64,
        cost: Option<f64>,
    ) -> Result<(), StoreError> {
        self.insert_cloud_usage_row(
            endpoint,
            model,
            input_tokens,
            output_tokens,
            reasoning_tokens,
            cost,
            None,
        )
    }

    /// One completed cloud request, with the seconds of AUDIO it consumed
    /// where that is what the provider billed.
    ///
    /// A whisper-class transcription reports no tokens at all - it charges per
    /// audio second - so a row of "0 tokens, $0.004" would be true and say
    /// nothing about what was bought. The newer speech models do bill by token
    /// and fill both. Either way the seconds ride in their own column, because
    /// they are a different unit and not a different number.
    pub fn insert_cloud_usage_row(
        &self,
        endpoint: &str,
        model: &str,
        input_tokens: u64,
        output_tokens: u64,
        reasoning_tokens: u64,
        cost: Option<f64>,
        audio_seconds: Option<f64>,
    ) -> Result<(), StoreError> {
        self.lock().execute(
            "INSERT INTO cloud_usage (ts_ms, endpoint, model, input_tokens, output_tokens, \
             reasoning_tokens, cost, audio_seconds) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                now_ms(),
                endpoint,
                model,
                input_tokens as i64,
                output_tokens as i64,
                reasoning_tokens as i64,
                cost,
                audio_seconds
            ],
        )?;
        Ok(())
    }

    /// Per-endpoint totals - all time and the trailing 24h - for the usage
    /// endpoint. Costs sum only over rows that have one (SUM skips NULLs), so
    /// a mixed OpenRouter/Anthropic history never reads as "free".
    ///
    /// `audioSeconds` sums the same way and is NULL where no transcription ran
    /// - the point of the column is that a whisper-class request buys seconds
    ///   and not tokens, so an endpoint that has only ever chatted must not
    ///   report "0 seconds of audio" as though that were a measurement.
    pub fn cloud_usage_summary(&self) -> Result<Vec<Value>, StoreError> {
        let conn = self.lock();
        let day_ago = now_ms() - 24 * 3600 * 1000;
        let mut stmt = conn.prepare(
            "SELECT endpoint, COUNT(*), SUM(input_tokens), SUM(output_tokens), \
             SUM(reasoning_tokens), SUM(cost), SUM(audio_seconds), \
             SUM(CASE WHEN ts_ms >= ?1 THEN 1 ELSE 0 END), \
             SUM(CASE WHEN ts_ms >= ?1 THEN input_tokens ELSE 0 END), \
             SUM(CASE WHEN ts_ms >= ?1 THEN output_tokens ELSE 0 END), \
             SUM(CASE WHEN ts_ms >= ?1 THEN cost ELSE NULL END), \
             SUM(CASE WHEN ts_ms >= ?1 THEN audio_seconds ELSE NULL END) \
             FROM cloud_usage GROUP BY endpoint ORDER BY endpoint",
        )?;
        let rows = stmt
            .query_map(params![day_ago], |r| {
                Ok(serde_json::json!({
                    "endpoint": r.get::<_, String>(0)?,
                    "requests": r.get::<_, i64>(1)?,
                    "inputTokens": r.get::<_, i64>(2)?,
                    "outputTokens": r.get::<_, i64>(3)?,
                    "reasoningTokens": r.get::<_, i64>(4)?,
                    "cost": r.get::<_, Option<f64>>(5)?,
                    "audioSeconds": r.get::<_, Option<f64>>(6)?,
                    "today": {
                        "requests": r.get::<_, i64>(7)?,
                        "inputTokens": r.get::<_, i64>(8)?,
                        "outputTokens": r.get::<_, i64>(9)?,
                        "cost": r.get::<_, Option<f64>>(10)?,
                        "audioSeconds": r.get::<_, Option<f64>>(11)?,
                    },
                }))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// The relay's view: kind, base URL and the key itself. Internal only -
    /// no route returns this shape.
    /// The stored pick's published output ceiling (`maxOut`), when this
    /// endpoint's models JSON carries one for the model id. The relay clamps
    /// `max_output_tokens` with it - the client plans the same clamp, but a
    /// pick stored before the ceiling was recorded plans from the context
    /// window alone and overshoots (observed: 128411 vs claude-sonnet-5's
    /// 128000, the whole send dead on the provider's 400).
    pub fn cloud_endpoint_out_cap(&self, id: &str, model: &str) -> Option<u64> {
        let conn = self.lock();
        let models: String = conn
            .query_row(
                "SELECT models FROM cloud_endpoints WHERE id = ?1",
                params![id],
                |r| r.get(0),
            )
            .ok()?;
        let doc: Value = serde_json::from_str(&models).ok()?;
        doc.as_array()?
            .iter()
            .find(|m| m.get("id").and_then(Value::as_str) == Some(model))?
            .get("maxOut")
            .and_then(Value::as_u64)
    }

    pub fn cloud_endpoint_secret(
        &self,
        id: &str,
    ) -> Result<Option<(String, String, String)>, StoreError> {
        let conn = self.lock();
        let row = conn
            .query_row(
                "SELECT kind, base_url, api_key FROM cloud_endpoints WHERE id = ?1",
                params![id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .optional()?;
        drop(conn);
        row.map(|(kind, base, key)| {
            self.cloud_credentials
                .resolve(key)
                .map(|key| (kind, base, key))
                .map_err(StoreError::Bad)
        })
        .transpose()
    }

    // ── activity (collected runner event records, doc §8.1) ──────────────────

    /// The collector's resume cursor for one runner GENERATION (its
    /// per-process-start instance id): the sequence to ask for
    /// next (max stored + 1, or 0 when nothing was collected yet).
    pub fn activity_cursor(&self, instance_id: &str) -> Result<u64, StoreError> {
        let conn = self.lock();
        let max: Option<i64> = conn.query_row(
            "SELECT MAX(seq) FROM activity WHERE instance_id = ?1",
            params![instance_id],
            |r| r.get(0),
        )?;
        Ok(max.map_or(0, |m| m as u64 + 1))
    }

    /// Batch-insert one collected page (single transaction, WAL - the §8.7
    /// "thousands/s capacity against tens needed" path). INSERT OR IGNORE on
    /// the (instance_id, seq) key keeps re-collection idempotent - and since
    /// the id is a per-generation UUID, a same-second respawn can no longer
    /// alias its predecessor's key space and lose rows.
    pub fn insert_activity(
        &self,
        instance_id: &str,
        port: u16,
        started_at: u64,
        events: &[Value],
    ) -> Result<usize, StoreError> {
        let mut conn = self.lock();
        let tx = conn.transaction()?;
        let mut n = 0usize;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT OR IGNORE INTO activity
                 (instance_id, seq, port, runner_started_at, ts_ms, request_id, endpoint, status, model, session_id, record)
                 VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)",
            )?;
            for ev in events {
                let seq = ev.get("seq").and_then(Value::as_u64).unwrap_or(0) as i64;
                let ts = ev.get("ts_ms").and_then(Value::as_u64).unwrap_or(0) as i64;
                let rid = ev.get("request_id").and_then(Value::as_str).unwrap_or("");
                let endpoint = ev.get("endpoint").and_then(Value::as_str).unwrap_or("");
                let status = ev.get("status").and_then(Value::as_u64).unwrap_or(0) as i64;
                let model = ev.get("gen_ai.request.model").and_then(Value::as_str);
                let session = ev.get("session_id").and_then(Value::as_str);
                let record = paddock_admin::codec::encode_record(ev)
                    .map_err(|e| StoreError::Bad(e.to_string()))?;
                n += stmt.execute(params![
                    instance_id,
                    seq,
                    port,
                    started_at as i64,
                    ts,
                    rid,
                    endpoint,
                    status,
                    model,
                    session,
                    record
                ])?;
            }
        }
        tx.commit()?;
        Ok(n)
    }

    /// Recent activity, newest first. Optional filters; each row is the full
    /// stored record plus the collector's `port`.
    pub fn list_activity(
        &self,
        limit: usize,
        before_ts_ms: Option<i64>,
        port: Option<u16>,
        model: Option<&str>,
        session_id: Option<&str>,
    ) -> Result<Vec<Value>, StoreError> {
        let conn = self.lock();
        // Static query with COALESCE-style optional filters (NULL = no filter):
        // simpler and plan-cacheable vs. building SQL strings.
        let mut stmt = conn.prepare_cached(
            "SELECT port, record FROM activity
             WHERE (?2 IS NULL OR ts_ms < ?2)
               AND (?3 IS NULL OR port = ?3)
               AND (?4 IS NULL OR model = ?4)
               AND (?5 IS NULL OR session_id = ?5)
             ORDER BY ts_ms DESC, seq DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map(
            params![limit as i64, before_ts_ms, port, model, session_id],
            |r| {
                // Codec blobs read as Blob; earlier rows are JSON TEXT.
                // Either way hand the raw bytes to decode_record's sniff.
                let record = match r.get_ref(1)? {
                    rusqlite::types::ValueRef::Blob(b) => b.to_vec(),
                    rusqlite::types::ValueRef::Text(t) => t.to_vec(),
                    other => {
                        return Err(rusqlite::Error::InvalidColumnType(
                            1,
                            "record".into(),
                            other.data_type(),
                        ));
                    }
                };
                Ok((r.get::<_, i64>(0)?, record))
            },
        )?;
        let mut out = Vec::new();
        for row in rows {
            let (port, record) = row?;
            let mut v = paddock_admin::codec::decode_record(&record)
                .map_err(|e| StoreError::Bad(e.to_string()))?;
            v["port"] = json!(port);
            out.push(v);
        }
        Ok(out)
    }

    /// Retention: drop records older than the cutoff. Returns rows deleted.
    pub fn purge_activity_before(&self, cutoff_ts_ms: i64) -> Result<usize, StoreError> {
        Ok(self.lock().execute(
            "DELETE FROM activity WHERE ts_ms < ?1",
            params![cutoff_ts_ms],
        )?)
    }

    /// Explicit user purge: everything, now.
    pub fn clear_activity(&self) -> Result<usize, StoreError> {
        Ok(self.lock().execute("DELETE FROM activity", [])?)
    }

    // ── usage tier (scraped metrics rollups,  /) ────────────

    /// Intern one (port, model, operation, origin) into `usage_series`.
    /// Callers cache the result per scrape task; this only hits SQLite on the
    /// first sighting of a dimension combination.
    pub fn intern_usage_series(
        &self,
        port: u16,
        model: &str,
        operation: &str,
        origin: &str,
    ) -> Result<i64, StoreError> {
        let conn = self.lock();
        conn.execute(
            "INSERT OR IGNORE INTO usage_series (port, model, operation, origin)
             VALUES (?1, ?2, ?3, ?4)",
            params![port, model, operation, origin],
        )?;
        Ok(conn.query_row(
            "SELECT id FROM usage_series
             WHERE port = ?1 AND model = ?2 AND operation = ?3 AND origin = ?4",
            params![port, model, operation, origin],
            |r| r.get(0),
        )?)
    }

    /// Fold one scrape delta into the open 5-minute AND hourly rows (both
    /// grains written on ingest - simpler and idempotent-per-delta versus a
    /// later rollup job). Additive columns accumulate; `kv_pages_max` is a
    /// gauge high-water and MAX-merges.
    pub fn fold_usage_bucket(
        &self,
        series_id: i64,
        ts_ms: i64,
        d: &crate::usage::BucketDelta,
    ) -> Result<(), StoreError> {
        let mut conn = self.lock();
        let tx = conn.transaction()?;
        fold_bucket_in(&tx, series_id, ts_ms, d)?;
        tx.commit()?;
        Ok(())
    }

    /// One atomic usage step: a scrape interval's (or
    /// recovered interval's) bucket folds, the totals rows they advance to,
    /// AND the full-state attach baseline, committed together. The state is
    /// the collector's resume cursor - one transaction means a crash at any
    /// point re-runs from the last committed interval instead of
    /// double-counting or losing anything.
    pub fn fold_usage_step(
        &self,
        instance_id: &str,
        started_ms: i64,
        ts_ms: i64,
        items: &[crate::usage::UsageFoldItem],
        web: &[crate::usage::WebFoldItem],
        port: u16,
        state_json: &str,
    ) -> Result<(), StoreError> {
        let mut conn = self.lock();
        let tx = conn.transaction()?;
        for w in web {
            // Same transaction as the state write, for the same reason the
            // series folds are: the persisted state is the resume cursor, so
            // a crash between a spend fold and the state advance would charge
            // the same searches twice on the next attach.
            if let Some(d) = &w.delta {
                fold_web_bucket_in(&tx, port, &w.provider, ts_ms, d)?;
            }
            upsert_web_total_in(
                &tx,
                instance_id,
                port,
                &w.provider,
                started_ms,
                &w.absolute,
                ts_ms,
            )?;
        }
        for it in items {
            if let Some(d) = &it.delta {
                fold_bucket_in(&tx, it.series_id, ts_ms, d)?;
            }
            upsert_total_in(
                &tx,
                it.series_id,
                instance_id,
                started_ms,
                it.requests,
                it.input_tokens,
                it.output_tokens,
                it.cached_tokens,
                it.spec_drafted,
                it.spec_accepted,
                ts_ms,
            )?;
        }
        tx.execute(
            "INSERT INTO usage_state (instance_id, ts_ms, state) VALUES (?1, ?2, ?3)
             ON CONFLICT(instance_id) DO UPDATE SET
                 ts_ms = excluded.ts_ms, state = excluded.state",
            params![instance_id, ts_ms, state_json],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// The collector's persisted full counter state for one generation, with
    /// the time it was observed - the attach baseline. None on a database
    /// from before the state tier existed (the totals then serve, at their
    /// four-column fidelity).
    pub fn usage_state_of(&self, instance_id: &str) -> Result<Option<(i64, String)>, StoreError> {
        let conn = self.lock();
        let row = conn
            .query_row(
                "SELECT ts_ms, state FROM usage_state WHERE instance_id = ?1",
                params![instance_id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        Ok(row)
    }

    /// Overwrite one series+generation's last-observed cumulative counters.
    /// A high-water mark, not a time series (keyed on the
    /// generation so a successor's fresh counters can never read as a
    /// negative delta against a per-port total).
    #[allow(clippy::too_many_arguments)]
    pub fn upsert_usage_total(
        &self,
        series_id: i64,
        instance_id: &str,
        started_ms: i64,
        requests: u64,
        input_tokens: u64,
        output_tokens: u64,
        cached_tokens: u64,
        spec_drafted: u64,
        spec_accepted: u64,
        last_scrape_ms: i64,
    ) -> Result<(), StoreError> {
        upsert_total_in(
            &self.lock(),
            series_id,
            instance_id,
            started_ms,
            requests,
            input_tokens,
            output_tokens,
            cached_tokens,
            spec_drafted,
            spec_accepted,
            last_scrape_ms,
        )?;
        Ok(())
    }

    /// Bump `last_scrape_ms` on every row of a generation - an idle scrape
    /// still proves the manager was watching, which is what keeps a later
    /// gap window honest.
    pub fn touch_usage_totals(
        &self,
        instance_id: &str,
        last_scrape_ms: i64,
    ) -> Result<(), StoreError> {
        self.lock().execute(
            "UPDATE usage_total SET last_scrape_ms = ?2 WHERE instance_id = ?1",
            params![instance_id, last_scrape_ms],
        )?;
        Ok(())
    }

    /// The stored totals for one generation, dimensions included - the
    /// collector's attach baseline (what was already accounted before a
    /// blind window).
    pub fn usage_totals_for_instance(
        &self,
        instance_id: &str,
    ) -> Result<Vec<crate::usage::TotalRow>, StoreError> {
        let conn = self.lock();
        let mut stmt = conn.prepare_cached(
            "SELECT t.series_id, s.operation, s.origin, s.model,
                    t.requests, t.input_tokens, t.output_tokens, t.cached_tokens,
                    t.spec_drafted, t.spec_accepted, t.last_scrape_ms
             FROM usage_total t JOIN usage_series s ON s.id = t.series_id
             WHERE t.instance_id = ?1",
        )?;
        let rows = stmt.query_map(params![instance_id], |r| {
            Ok(crate::usage::TotalRow {
                series_id: r.get(0)?,
                key: crate::usage::SeriesKey {
                    operation: r.get(1)?,
                    origin: r.get(2)?,
                    model: r.get(3)?,
                },
                requests: r.get::<_, i64>(4)? as u64,
                input_tokens: r.get::<_, i64>(5)? as u64,
                output_tokens: r.get::<_, i64>(6)? as u64,
                cached_tokens: r.get::<_, i64>(7)? as u64,
                spec_drafted: r.get::<_, i64>(8)? as u64,
                spec_accepted: r.get::<_, i64>(9)? as u64,
                last_scrape_ms: r.get(10)?,
            })
        })?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    /// When this generation was last scraped, if ever.
    pub fn last_scrape_of(&self, instance_id: &str) -> Result<Option<i64>, StoreError> {
        let conn = self.lock();
        let v: Option<i64> = conn.query_row(
            "SELECT MAX(last_scrape_ms) FROM usage_total WHERE instance_id = ?1",
            params![instance_id],
            |r| r.get(0),
        )?;
        Ok(v)
    }

    /// Open a lifecycle band for a newly-seen generation. Idempotent: a
    /// re-attach after a manager restart is a no-op. Returns true when the
    /// row is new, in which case a fresh pending start cause for the port
    /// (noted by the route that spawned it) is consumed into the band.
    #[allow(clippy::too_many_arguments)]
    pub fn open_generation(
        &self,
        instance_id: &str,
        port: u16,
        pid: u32,
        runner_version: &str,
        model: Option<&str>,
        embedder: Option<&str>,
        asr: Option<&str>,
        aligner: Option<&str>,
        started_ms: i64,
        boot_ms: Option<i64>,
    ) -> Result<bool, StoreError> {
        let conn = self.lock();
        let n = conn.execute(
            "INSERT OR IGNORE INTO service_generation
                 (instance_id, port, pid, runner_version, model, embedder, asr, aligner,
                  started_ms, boot_ms)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",
            params![
                instance_id,
                port,
                pid,
                runner_version,
                model,
                embedder,
                asr,
                aligner,
                started_ms,
                boot_ms
            ],
        )?;
        if n > 0 {
            let pending: Option<(String, i64)> = conn
                .query_row(
                    "SELECT cause, noted_ms FROM usage_pending_cause WHERE port = ?1",
                    params![port],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .optional()?;
            if let Some((cause, noted)) = pending {
                conn.execute(
                    "DELETE FROM usage_pending_cause WHERE port = ?1",
                    params![port],
                )?;
                // A stale note (spawn that never came up) must not label a
                // much later unrelated start.
                if now_ms() - noted < 600_000 {
                    conn.execute(
                        "UPDATE service_generation SET start_cause = ?2 WHERE instance_id = ?1",
                        params![instance_id, cause],
                    )?;
                }
            }
        }
        Ok(n > 0)
    }

    /// Close a band, filling only what is not already known - a route-stamped
    /// end cause ('stopped') survives the collector's later observational
    /// close ('takeover' / 'unknown').
    pub fn close_generation(
        &self,
        instance_id: &str,
        ended_ms: i64,
        cause: &str,
    ) -> Result<(), StoreError> {
        self.lock().execute(
            "UPDATE service_generation
             SET ended_ms = COALESCE(ended_ms, ?2), end_cause = COALESCE(end_cause, ?3)
             WHERE instance_id = ?1",
            params![instance_id, ended_ms, cause],
        )?;
        Ok(())
    }

    /// A route that knows why the newest generation on a port is ending
    /// (a clean stop) stamps it - overwriting an observational guess.
    pub fn stamp_end_cause(&self, port: u16, cause: &str) -> Result<(), StoreError> {
        self.lock().execute(
            "UPDATE service_generation SET end_cause = ?2
             WHERE instance_id = (SELECT instance_id FROM service_generation
                                  WHERE port = ?1 ORDER BY started_ms DESC, rowid DESC LIMIT 1)",
            params![port, cause],
        )?;
        Ok(())
    }

    /// The newest still-open band on a port other than `excluding` - a
    /// predecessor whose death the manager never observed (it would have been
    /// closed otherwise).
    pub fn open_predecessor_on_port(
        &self,
        port: u16,
        excluding_instance: &str,
    ) -> Result<Option<String>, StoreError> {
        let conn = self.lock();
        let row: Option<String> = conn
            .query_row(
                "SELECT instance_id FROM service_generation
                 WHERE port = ?1 AND ended_ms IS NULL AND instance_id != ?2
                 ORDER BY started_ms DESC LIMIT 1",
                params![port, excluding_instance],
                |r| r.get(0),
            )
            .optional()?;
        Ok(row)
    }

    /// Every open band, oldest first: (instance_id, port, started_ms, boot_ms).
    /// The sweep's input; `close_orphan_generations` decides which of them are
    /// corpses.
    pub fn open_generations(&self) -> Result<Vec<(String, u16, i64, Option<i64>)>, StoreError> {
        let conn = self.lock();
        let mut stmt = conn.prepare_cached(
            "SELECT instance_id, port, started_ms, boot_ms FROM service_generation
             WHERE ended_ms IS NULL ORDER BY started_ms",
        )?;
        let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    /// Close one band that nothing is running any more, ending it where the
    /// evidence ends: its last scrape, else its start. Never `now` - the time
    /// between the last heartbeat and noticing is exactly what we do not know,
    /// and charging it as uptime is the lie this sweep exists to stop. The
    /// caller journals that stretch as a gap.
    ///
    /// Returns the end it chose, so the caller can record the hole after it.
    pub fn close_orphan_generation(
        &self,
        instance_id: &str,
        started_ms: i64,
        cause: &str,
    ) -> Result<i64, StoreError> {
        let ended = self.last_scrape_of(instance_id)?.unwrap_or(started_ms);
        // A scrape can land a hair before the band's own start on a clock
        // edge; an end before its start would draw backwards.
        let ended = ended.max(started_ms);
        self.close_generation(instance_id, ended, cause)?;
        Ok(ended)
    }

    /// Note why the next generation on a port will exist ('manual',
    /// 'boot-election', 'batch-restore') - consumed by `open_generation`.
    pub fn note_start_cause(&self, port: u16, cause: &str) -> Result<(), StoreError> {
        self.lock().execute(
            "INSERT INTO usage_pending_cause (port, cause, noted_ms) VALUES (?1, ?2, ?3)
             ON CONFLICT(port) DO UPDATE SET cause = excluded.cause, noted_ms = excluded.noted_ms",
            params![port, cause, now_ms()],
        )?;
        Ok(())
    }

    /// Record a hole. Metrics-tier causes carry exact lost totals; event-tier
    /// causes carry the seq range and NULL totals (a bounded ring genuinely
    /// forgets).
    #[allow(clippy::too_many_arguments)]
    pub fn insert_usage_gap(
        &self,
        port: u16,
        instance_id: &str,
        from_ts_ms: i64,
        to_ts_ms: i64,
        cause: &str,
        seq_range: Option<(i64, i64)>,
        lost: Option<(i64, i64, i64)>,
    ) -> Result<(), StoreError> {
        self.lock().execute(
            "INSERT INTO usage_gap (port, instance_id, from_ts_ms, to_ts_ms, noticed_ms, cause,
                 from_seq, to_seq, lost_requests, lost_input_tokens, lost_output_tokens)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)",
            params![
                port,
                instance_id,
                from_ts_ms,
                to_ts_ms,
                now_ms(),
                cause,
                seq_range.map(|s| s.0),
                seq_range.map(|s| s.1),
                lost.map(|l| l.0),
                lost.map(|l| l.1),
                lost.map(|l| l.2),
            ],
        )?;
        Ok(())
    }

    /// Retention for the 5-minute grain (kept ~90 days; hourly is forever).
    pub fn purge_usage_5m_before(&self, cutoff_ts_ms: i64) -> Result<usize, StoreError> {
        Ok(self.lock().execute(
            "DELETE FROM usage_bucket WHERE grain = '5m' AND bucket_start_ms < ?1",
            params![cutoff_ts_ms],
        )?)
    }

    // ── usage read path: the timeline API's queries ──────────────

    /// Timeline slots in `[from, to)`, aggregated per (port, slot). `group_ms`
    /// regroups stored buckets server-side (a year view reads hourly rows into
    /// daily slots instead of shipping 8.7k rows to the browser to discard) -
    /// pass the grain's own width for no regrouping. `port < 0` means all
    /// ports. Range scan on the (grain, bucket_start_ms) index.
    pub fn usage_history(
        &self,
        grain: &str,
        group_ms: i64,
        from_ms: i64,
        to_ms: i64,
        port: i64,
    ) -> Result<Vec<crate::usage::UsageSlot>, StoreError> {
        // Static shape, built once: the 28 histogram-step sums would be
        // unreadable inline and the ladder width is a compile-time fact.
        static HISTORY_SQL: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            let e2e: String = (0..14).map(|i| format!(", SUM(b.e2e_h{i})")).collect();
            let ttft: String = (0..14).map(|i| format!(", SUM(b.ttft_h{i})")).collect();
            format!(
                "SELECT s.port, (b.bucket_start_ms / ?2) * ?2 AS t,
                        SUM(b.requests), SUM(b.errors_4xx), SUM(b.errors_5xx), SUM(b.disconnects),
                        SUM(b.input_tokens), SUM(b.output_tokens), SUM(b.cached_tokens),
                        SUM(b.duration_ms_sum), SUM(b.spec_drafted), SUM(b.spec_accepted),
                        MAX(b.kv_pages_max){e2e}{ttft}
                 FROM usage_bucket b JOIN usage_series s ON s.id = b.series_id
                 WHERE b.grain = ?1 AND b.bucket_start_ms >= ?3 AND b.bucket_start_ms < ?4
                   AND (?5 < 0 OR s.port = ?5)
                 GROUP BY s.port, t
                 ORDER BY t, s.port"
            )
        });
        let conn = self.lock();
        let mut stmt = conn.prepare_cached(&HISTORY_SQL)?;
        let rows = stmt.query_map(params![grain, group_ms, from_ms, to_ms, port], |r| {
            let mut e2e_h = [0i64; 14];
            let mut ttft_h = [0i64; 14];
            for i in 0..14 {
                e2e_h[i] = r.get(13 + i)?;
                ttft_h[i] = r.get(27 + i)?;
            }
            Ok(crate::usage::UsageSlot {
                port: r.get(0)?,
                t: r.get(1)?,
                requests: r.get(2)?,
                errors_4xx: r.get(3)?,
                errors_5xx: r.get(4)?,
                disconnects: r.get(5)?,
                input_tokens: r.get(6)?,
                output_tokens: r.get(7)?,
                cached_tokens: r.get(8)?,
                duration_ms_sum: r.get(9)?,
                spec_drafted: r.get(10)?,
                spec_accepted: r.get(11)?,
                kv_pages_max: r.get(12)?,
                e2e_h,
                ttft_h,
            })
        })?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    /// Web-search spend slots in `[from, to)`, on the same grain/regrouping
    /// contract as [`Store::usage_history`]. Grouped by PROVIDER as well as
    /// port: three providers charging in three currencies is the whole point,
    /// and a summed row could only be reported in a currency nobody uses.
    pub fn web_history(
        &self,
        grain: &str,
        group_ms: i64,
        from_ms: i64,
        to_ms: i64,
        port: i64,
    ) -> Result<Vec<crate::usage::WebSlot>, StoreError> {
        let conn = self.lock();
        let mut stmt = conn.prepare_cached(
            "SELECT port, provider, (bucket_start_ms / ?2) * ?2 AS t,
                    SUM(requests), SUM(credits), SUM(microdollars)
             FROM web_search_bucket
             WHERE grain = ?1 AND bucket_start_ms >= ?3 AND bucket_start_ms < ?4
               AND (?5 < 0 OR port = ?5)
             GROUP BY port, provider, t
             ORDER BY t, port, provider",
        )?;
        let rows = stmt.query_map(params![grain, group_ms, from_ms, to_ms, port], |r| {
            Ok(crate::usage::WebSlot {
                port: r.get(0)?,
                provider: r.get(1)?,
                t: r.get(2)?,
                requests: r.get(3)?,
                credits: r.get(4)?,
                microdollars: r.get(5)?,
            })
        })?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    /// The first instant there is any usage record for - the left edge of
    /// the all-history axis. Buckets and lifecycle bands can each
    /// predate the other (a generation opens before its first scrape), so the
    /// extent is the min of both.
    pub fn usage_extent(&self) -> Result<Option<i64>, StoreError> {
        let conn = self.lock();
        let b: Option<i64> =
            conn.query_row("SELECT MIN(bucket_start_ms) FROM usage_bucket", [], |r| {
                r.get(0)
            })?;
        let g: Option<i64> =
            conn.query_row("SELECT MIN(started_ms) FROM service_generation", [], |r| {
                r.get(0)
            })?;
        Ok(match (b, g) {
            (Some(a), Some(c)) => Some(a.min(c)),
            (x, None) => x,
            (None, y) => y,
        })
    }

    /// Gap rows overlapping `[from, to)`. Read for the whole window on every
    /// poll, never incrementally: a gap covering yesterday is inserted the
    /// moment the manager comes back TODAY (attach), so "closed" time can
    /// still grow new gap rows - only buckets are immutable once closed.
    pub fn usage_gaps_in(
        &self,
        from_ms: i64,
        to_ms: i64,
        port: i64,
    ) -> Result<Vec<crate::usage::UsageGapRow>, StoreError> {
        let conn = self.lock();
        let mut stmt = conn.prepare_cached(
            "SELECT id, port, from_ts_ms, to_ts_ms, cause,
                    lost_requests, lost_input_tokens, lost_output_tokens, from_seq, to_seq
             FROM usage_gap
             WHERE from_ts_ms < ?2 AND to_ts_ms > ?1 AND (?3 < 0 OR port = ?3)
             ORDER BY from_ts_ms",
        )?;
        let rows = stmt.query_map(params![from_ms, to_ms, port], |r| {
            Ok(crate::usage::UsageGapRow {
                id: r.get(0)?,
                port: r.get(1)?,
                from_ts_ms: r.get(2)?,
                to_ts_ms: r.get(3)?,
                cause: r.get(4)?,
                lost_requests: r.get(5)?,
                lost_input_tokens: r.get(6)?,
                lost_output_tokens: r.get(7)?,
                from_seq: r.get(8)?,
                to_seq: r.get(9)?,
            })
        })?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    /// Lifecycle bands overlapping `[from, to)` (open bands overlap any
    /// window reaching now).
    pub fn usage_generations_in(
        &self,
        from_ms: i64,
        to_ms: i64,
        port: i64,
    ) -> Result<Vec<crate::usage::GenerationRow>, StoreError> {
        let conn = self.lock();
        let mut stmt = conn.prepare_cached(
            // last_seen_ms is the band's own heartbeat: the newest scrape the
            // collector folded for it. An open band with a stale one is a
            // runner nobody has heard from, which the page says out loud
            // instead of drawing it as healthy.
            "SELECT g.instance_id, g.port, g.runner_version, g.model, g.embedder, g.asr,
                    g.aligner, g.started_ms, g.ended_ms, g.start_cause, g.end_cause,
                    (SELECT MAX(u.last_scrape_ms) FROM usage_total u
                      WHERE u.instance_id = g.instance_id)
             FROM service_generation g
             WHERE g.started_ms < ?2 AND (g.ended_ms IS NULL OR g.ended_ms > ?1)
               AND (?3 < 0 OR g.port = ?3)
             ORDER BY g.port, g.started_ms",
        )?;
        let rows = stmt.query_map(params![from_ms, to_ms, port], |r| {
            Ok(crate::usage::GenerationRow {
                instance_id: r.get(0)?,
                port: r.get(1)?,
                runner_version: r.get(2)?,
                model: r.get(3)?,
                embedder: r.get(4)?,
                asr: r.get(5)?,
                aligner: r.get(6)?,
                started_ms: r.get(7)?,
                ended_ms: r.get(8)?,
                start_cause: r.get(9)?,
                end_cause: r.get(10)?,
                last_seen_ms: r.get(11)?,
            })
        })?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }
}

/// The bucket fold, on whatever connection the caller holds - a bare one
/// (single fold) or a recovery step's transaction. `Transaction` derefs to
/// `Connection`, so both call this.
fn fold_bucket_in(
    conn: &rusqlite::Connection,
    series_id: i64,
    ts_ms: i64,
    d: &crate::usage::BucketDelta,
) -> Result<(), rusqlite::Error> {
    use std::sync::LazyLock;
    static SQL: LazyLock<String> = LazyLock::new(|| {
        let cols = crate::usage::BucketDelta::columns();
        let updates: Vec<String> = cols
            .iter()
            .map(|c| format!("{c} = {c} + excluded.{c}"))
            .collect();
        format!(
            "INSERT INTO usage_bucket (series_id, grain, bucket_start_ms, {}, kv_pages_max)
             VALUES (?, ?, ?, {}, ?)
             ON CONFLICT(series_id, grain, bucket_start_ms) DO UPDATE SET
             {}, kv_pages_max = MAX(kv_pages_max, excluded.kv_pages_max)",
            cols.join(", "),
            vec!["?"; cols.len()].join(", "),
            updates.join(", "),
        )
    });
    let mut stmt = conn.prepare_cached(&SQL)?;
    for (grain, width) in [("5m", 300_000i64), ("1h", 3_600_000i64)] {
        let mut vals: Vec<rusqlite::types::Value> = vec![
            series_id.into(),
            grain.to_owned().into(),
            (ts_ms - ts_ms.rem_euclid(width)).into(),
        ];
        vals.extend(d.add_values().into_iter().map(rusqlite::types::Value::from));
        vals.push(d.kv_pages_max.into());
        stmt.execute(rusqlite::params_from_iter(vals))?;
    }
    Ok(())
}

/// One provider's spend delta into both grains, connection-agnostic for the
/// same reason as [`fold_bucket_in`].
fn fold_web_bucket_in(
    conn: &rusqlite::Connection,
    port: u16,
    provider: &str,
    ts_ms: i64,
    d: &crate::usage::WebSpend,
) -> Result<(), rusqlite::Error> {
    let mut stmt = conn.prepare_cached(
        "INSERT INTO web_search_bucket
             (port, provider, grain, bucket_start_ms, requests, credits, microdollars)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
         ON CONFLICT(port, provider, grain, bucket_start_ms) DO UPDATE SET
             requests = requests + excluded.requests,
             credits = credits + excluded.credits,
             microdollars = microdollars + excluded.microdollars",
    )?;
    for (grain, width) in [("5m", 300_000i64), ("1h", 3_600_000i64)] {
        stmt.execute(params![
            port,
            provider,
            grain,
            ts_ms - ts_ms.rem_euclid(width),
            d.requests as i64,
            d.credits as i64,
            d.microdollars as i64,
        ])?;
    }
    Ok(())
}

/// One provider's cumulative spend for a generation - a high-water overwrite,
/// never a time series, exactly like [`upsert_total_in`].
fn upsert_web_total_in(
    conn: &rusqlite::Connection,
    instance_id: &str,
    port: u16,
    provider: &str,
    started_ms: i64,
    abs: &crate::usage::WebSpend,
    last_scrape_ms: i64,
) -> Result<(), rusqlite::Error> {
    conn.execute(
        "INSERT INTO web_search_total (instance_id, port, provider, started_ms,
             requests, credits, microdollars, last_scrape_ms)
         VALUES (?1,?2,?3,?4,?5,?6,?7,?8)
         ON CONFLICT(instance_id, provider) DO UPDATE SET
             requests = excluded.requests,
             credits = excluded.credits,
             microdollars = excluded.microdollars,
             last_scrape_ms = excluded.last_scrape_ms",
        params![
            instance_id,
            port,
            provider,
            started_ms,
            abs.requests as i64,
            abs.credits as i64,
            abs.microdollars as i64,
            last_scrape_ms,
        ],
    )?;
    Ok(())
}

/// The totals upsert, connection-agnostic for the same reason as
/// [`fold_bucket_in`].
#[allow(clippy::too_many_arguments)]
fn upsert_total_in(
    conn: &rusqlite::Connection,
    series_id: i64,
    instance_id: &str,
    started_ms: i64,
    requests: u64,
    input_tokens: u64,
    output_tokens: u64,
    cached_tokens: u64,
    spec_drafted: u64,
    spec_accepted: u64,
    last_scrape_ms: i64,
) -> Result<(), rusqlite::Error> {
    conn.execute(
        "INSERT INTO usage_total (series_id, instance_id, started_ms, requests,
             input_tokens, output_tokens, cached_tokens, spec_drafted, spec_accepted,
             last_scrape_ms)
         VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)
         ON CONFLICT(series_id, instance_id) DO UPDATE SET
             requests = excluded.requests,
             input_tokens = excluded.input_tokens,
             output_tokens = excluded.output_tokens,
             cached_tokens = excluded.cached_tokens,
             spec_drafted = excluded.spec_drafted,
             spec_accepted = excluded.spec_accepted,
             last_scrape_ms = excluded.last_scrape_ms",
        params![
            series_id,
            instance_id,
            started_ms,
            requests as i64,
            input_tokens as i64,
            output_tokens as i64,
            cached_tokens as i64,
            spec_drafted as i64,
            spec_accepted as i64,
            last_scrape_ms
        ],
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mem_store() -> Store {
        Store::open(&PathBuf::from(":memory:")).expect("mem db")
    }

    #[test]
    fn settings_import_preserves_existing_keys_and_tombstones() {
        let s = mem_store();
        s.patch_settings(
            &json!({"studio.pk_theme": "dark", "studio.pk_model": null}),
            false,
        )
        .unwrap();
        s.patch_settings(&json!({"studio.pk_theme": "light", "studio.pk_model": "old", "studio.pk_sidebar_width": "0"}), true).unwrap();
        let saved = s.all_settings().unwrap();
        assert_eq!(saved["studio.pk_theme"], "dark");
        assert!(saved["studio.pk_model"].is_null());
        assert_eq!(saved["studio.pk_sidebar_width"], "0");
    }

    #[test]
    fn settings_patch_validation_prevents_partial_writes() {
        let s = mem_store();
        assert!(
            s.patch_settings(
                &json!({"studio.pk_theme": "light", "not-a-preference": "x"}),
                true
            )
            .is_err()
        );
        assert_eq!(s.all_settings().unwrap(), json!({}));
        assert!(s.patch_settings(&json!(["bad"]), false).is_err());
        assert!(
            s.patch_settings(&json!({"x": "x".repeat(1024 * 1024)}), false)
                .is_err()
        );
    }

    #[test]
    fn settings_sql_failure_rolls_back_the_entire_patch() {
        let s = mem_store();
        s.lock()
            .execute_batch(
                "CREATE TRIGGER fail_setting BEFORE INSERT ON settings
            WHEN NEW.key = 'studio.z' BEGIN SELECT RAISE(ABORT, 'test disk error'); END;",
            )
            .unwrap();
        assert!(
            s.patch_settings(&json!({"studio.a": "first", "studio.z": "last"}), false)
                .is_err()
        );
        assert_eq!(s.all_settings().unwrap(), json!({}));
    }

    fn conv(id: &str, messages: Value) -> Value {
        json!({"id": id, "title": "t", "model": "m", "createdAt": 1, "updatedAt": 1,
               "messages": messages})
    }

    /// The list query must never touch the conversations TABLE: every column
    /// it selects is in `conversations_list`, and the row it would otherwise
    /// fetch carries the whole conversation document with it. On a slow disk
    /// with a cold cache that difference was 38.7 s against ~0. A column added
    /// to the SELECT without being added to the index brings the row fetches
    /// back, silently and only on someone else's hardware - so assert the
    /// plan, not the timing.
    #[test]
    fn the_conversation_list_is_answered_by_a_covering_index() {
        let s = mem_store();
        let conn = s.lock();
        let plan: String = conn
            .query_row(
                "EXPLAIN QUERY PLAN
                 SELECT id, title, model, pinned, updated_at, created_at, kind
                 FROM conversations ORDER BY updated_at DESC",
                [],
                |r| r.get(3),
            )
            .expect("plan");
        assert!(
            plan.contains("COVERING INDEX conversations_list"),
            "the list query fell back to row fetches: {plan}"
        );
        // and the index it replaced is gone, not carried along as write cost
        let leftover: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master
                 WHERE type = 'index' AND name = 'conversations_updated'",
                [],
                |r| r.get(0),
            )
            .expect("count");
        assert_eq!(leftover, 0, "the superseded index is still there");
    }

    /// The whole point: the LIST answers the kind, so an unopened row does not
    /// have to be opened to be drawn correctly.
    #[test]
    fn the_summary_carries_the_kind_without_reading_the_document() {
        let s = mem_store();
        s.put_conversation(&conv(
            "doc",
            json!([{"role": "user", "content": [{"type": "file", "name": "a.pdf"}]},
                   {"role": "assistant", "content": [], "docRun": {"pages": []}}]),
        ))
        .expect("put doc");
        s.put_conversation(&conv(
            "heard",
            json!([{"role": "user", "content": [{"type": "audio", "name": "a.wav"}]}]),
        ))
        .expect("put audio");
        s.put_conversation(&conv(
            "plain",
            json!([{"role": "user", "content": [{"type": "text"}]}]),
        ))
        .expect("put chat");
        // a picture the model MADE is an image conversation; a photo the
        // person attached is a chat about it
        s.put_conversation(&conv(
            "drew",
            json!([{"role": "user", "content": [{"type": "text", "text": "a cat"}]},
                   {"role": "assistant", "content": [{"type": "image", "name": "cat.png"}]}]),
        ))
        .expect("put image");
        s.put_conversation(&conv(
            "shown",
            json!([{"role": "user", "content": [{"type": "image", "name": "me.jpg"}, {"type": "text"}]},
                   {"role": "assistant", "content": [{"type": "text", "text": "a person"}]}]),
        ))
        .expect("put vision chat");

        let kinds: std::collections::HashMap<String, String> = s
            .list_conversations()
            .expect("list")
            .into_iter()
            .map(|v| {
                (
                    v["id"].as_str().unwrap_or_default().to_owned(),
                    v["kind"].as_str().unwrap_or_default().to_owned(),
                )
            })
            .collect();
        assert_eq!(kinds["doc"], "document");
        assert_eq!(kinds["heard"], "transcription");
        assert_eq!(kinds["plain"], "chat");
        assert_eq!(kinds["drew"], "image");
        assert_eq!(kinds["shown"], "chat");
    }

    /// A single OCR pass is the other shape the document lane writes.
    #[test]
    fn an_ocr_turn_counts_as_a_document_run() {
        let d = conv(
            "x",
            json!([{"role": "assistant", "content": [], "ocr": {"blocks": []}}]),
        );
        assert_eq!(conversation_kind(&d), "document");
    }

    /// The kind is EVIDENCE, so it has to change when the evidence does - a
    /// conversation is a chat right up until its first document run lands.
    #[test]
    fn a_chat_becomes_a_document_when_the_run_arrives() {
        let s = mem_store();
        s.put_conversation(&conv("c", json!([{"role": "user", "content": []}])))
            .expect("put");
        s.put_conversation(&conv(
            "c",
            json!([{"role": "user", "content": []},
                   {"role": "assistant", "content": [], "docRun": {"pages": []}}]),
        ))
        .expect("update");
        let row = s.list_conversations().expect("list");
        assert_eq!(
            row[0]["kind"], "document",
            "the update must restamp the kind"
        );
    }

    /// Malformed or absent messages must not panic or invent a kind.
    #[test]
    fn a_document_with_nothing_to_read_is_a_chat() {
        assert_eq!(conversation_kind(&json!({})), "chat");
        assert_eq!(
            conversation_kind(&json!({"messages": "not an array"})),
            "chat"
        );
        assert_eq!(
            conversation_kind(&json!({"messages": [{"role": "user"}]})),
            "chat"
        );
    }

    fn record(seq: u64, ts: u64, model: &str) -> Value {
        json!({
            "seq": seq,
            "ts_ms": ts,
            "endpoint": "/v1/chat/completions",
            "status": 200,
            "request_id": format!("req_{seq}"),
            "session_id": "s1",
            "gen_ai.request.model": model,
            "gen_ai.usage.input_tokens": 10,
        })
    }

    #[test]
    fn open_purges_legacy_model_config() {
        // an older install's DB: the pre-split mcp_servers table + the
        // web_search settings key - model config that must not survive open()
        let dir = std::env::temp_dir().join(format!("paddock-store-mig-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("paddock.db");
        {
            std::fs::create_dir_all(&dir).unwrap();
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE mcp_servers (id TEXT PRIMARY KEY, doc TEXT);
                 INSERT INTO mcp_servers VALUES ('x', '{\"server_url\":\"http://old\"}');
                 CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                 INSERT INTO settings VALUES ('web_search', '{\"key\":\"secret\"}');
                 INSERT INTO settings VALUES ('theme', '\"dark\"');",
            )
            .unwrap();
        }
        let s = Store::open(&path).unwrap();
        let conn = s.lock();
        let mcp: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='mcp_servers'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(mcp, 0, "legacy mcp_servers table dropped");
        let ws: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM settings WHERE key='web_search'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(ws, 0, "legacy web_search setting purged");
        let kept: i64 = conn
            .query_row("SELECT COUNT(*) FROM settings WHERE key='theme'", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(kept, 1, "studio settings survive");
        drop(conn);
        drop(s);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cloud_endpoint_roundtrip_hides_the_key() {
        let s = mem_store();
        let row = s
            .create_cloud_endpoint(&json!({
                "name": "OpenRouter", "kind": "openai-compat",
                "baseUrl": "https://openrouter.ai/api/v1/",
                "apiKey": "sk-or-secret",
                "models": [{"id": "meta/llama-3:free"}],
            }))
            .unwrap();
        let id = row["id"].as_str().unwrap().to_owned();
        // trailing slash normalized away; the key never appears in a row
        assert_eq!(row["baseUrl"], "https://openrouter.ai/api/v1");
        assert_eq!(row["hasKey"], true);
        assert!(row.get("apiKey").is_none());
        let listed = s.list_cloud_endpoints().unwrap();
        assert_eq!(listed.len(), 1);
        assert!(listed[0].get("apiKey").is_none());
        assert_eq!(listed[0]["models"][0]["id"], "meta/llama-3:free");

        // patch without apiKey keeps the stored secret; with one, replaces it
        s.update_cloud_endpoint(&id, &json!({"name": "OR", "models": []}))
            .unwrap();
        let (kind, base, key) = s.cloud_endpoint_secret(&id).unwrap().unwrap();
        assert_eq!(
            (kind.as_str(), base.as_str(), key.as_str()),
            (
                "openai-compat",
                "https://openrouter.ai/api/v1",
                "sk-or-secret"
            )
        );
        s.update_cloud_endpoint(&id, &json!({"apiKey": "sk-new"}))
            .unwrap();
        assert_eq!(s.cloud_endpoint_secret(&id).unwrap().unwrap().2, "sk-new");

        // bad kind / missing name are refused
        assert!(
            s.create_cloud_endpoint(&json!({"name": "x", "kind": "grpc", "baseUrl": "http://h"}))
                .is_err()
        );
        assert!(
            s.create_cloud_endpoint(&json!({"name": "", "kind": "openai", "baseUrl": "http://h"}))
                .is_err()
        );

        s.delete_cloud_endpoint(&id).unwrap();
        assert!(s.list_cloud_endpoints().unwrap().is_empty());
        assert!(s.cloud_endpoint_secret(&id).unwrap().is_none());
    }

    #[test]
    fn cloud_usage_ledger_sums_and_keeps_costless_rows_honest() {
        let s = mem_store();
        assert!(s.cloud_usage_summary().unwrap().is_empty());
        // OpenRouter reports cost; Anthropic doesn't (NULL, never 0)
        s.insert_cloud_usage(
            "or1",
            "anthropic/claude-sonnet-5",
            62842,
            1632,
            0,
            Some(0.142),
        )
        .unwrap();
        s.insert_cloud_usage(
            "or1",
            "anthropic/claude-sonnet-5",
            88,
            1055,
            0,
            Some(0.0107),
        )
        .unwrap();
        s.insert_cloud_usage("anth1", "claude-sonnet-5", 500, 200, 0, None)
            .unwrap();
        let rows = s.cloud_usage_summary().unwrap();
        assert_eq!(rows.len(), 2);
        // An endpoint that has only ever chatted reports no audio, not zero
        // audio - the column exists to say what was bought, and "0 seconds" on
        // a text endpoint would be a measurement nobody took.
        assert!(rows.iter().all(|r| r["audioSeconds"].is_null()));
        let anth = rows.iter().find(|r| r["endpoint"] == "anth1").unwrap();
        assert_eq!(anth["requests"], 1);
        assert_eq!(anth["inputTokens"], 500);
        assert!(anth["cost"].is_null(), "no provider cost never reads as $0");
        let or = rows.iter().find(|r| r["endpoint"] == "or1").unwrap();
        assert_eq!(or["requests"], 2);
        assert_eq!(or["inputTokens"], 62930);
        assert_eq!(or["outputTokens"], 2687);
        let c = or["cost"].as_f64().unwrap();
        assert!((c - 0.1527).abs() < 1e-9);
        // rows just written are inside the 24h window
        assert_eq!(or["today"]["requests"], 2);
    }

    /// A transcription books itself: the relay writes the row
    /// directly, because it answers with one JSON body and never passes the
    /// SSE tap that books every chat request.
    #[test]
    fn a_transcription_books_seconds_of_audio_beside_the_token_rows() {
        let s = mem_store();
        // whisper-class: bills per audio SECOND and reports no tokens at all
        s.insert_cloud_usage_row(
            "or1",
            "openai/whisper-large-v3",
            0,
            0,
            0,
            Some(0.0138),
            Some(9.2),
        )
        .unwrap();
        // a newer speech model bills by token AND times the clip
        s.insert_cloud_usage_row(
            "or1",
            "openai/gpt-4o-transcribe",
            83,
            30,
            0,
            Some(0.0005),
            Some(9.2),
        )
        .unwrap();
        // and an ordinary chat turn on the same endpoint has no audio at all
        s.insert_cloud_usage("or1", "z-ai/glm-5", 900, 120, 0, Some(0.002))
            .unwrap();

        let or = &s.cloud_usage_summary().unwrap()[0];
        assert_eq!(
            or["requests"], 3,
            "a transcription is a request like any other"
        );
        assert_eq!(
            or["inputTokens"], 983,
            "the token-less row contributes nothing, not a gap"
        );
        let secs = or["audioSeconds"].as_f64().unwrap();
        assert!(
            (secs - 18.4).abs() < 1e-9,
            "audio sums across both speech rows: {secs}"
        );
        let cost = or["cost"].as_f64().unwrap();
        assert!(
            (cost - 0.0163).abs() < 1e-9,
            "...and the money is the whole endpoint's: {cost}"
        );
        assert_eq!(or["today"]["audioSeconds"].as_f64().unwrap(), 18.4);
    }

    #[test]
    fn activity_roundtrip_cursor_and_idempotent_insert() {
        let s = mem_store();
        assert_eq!(s.activity_cursor("gen-a").unwrap(), 0);
        let evs = vec![record(0, 1000, "m1"), record(1, 2000, "m1")];
        assert_eq!(s.insert_activity("gen-a", 11540, 111, &evs).unwrap(), 2);
        // Re-collection of the same page is a no-op (INSERT OR IGNORE).
        assert_eq!(s.insert_activity("gen-a", 11540, 111, &evs).unwrap(), 0);
        // Cursor resumes past what's stored.
        assert_eq!(s.activity_cursor("gen-a").unwrap(), 2);
        // A restarted runner instance (new id) has its own cursor.
        assert_eq!(s.activity_cursor("gen-b").unwrap(), 0);

        // Newest first; the stored record comes back whole, plus the port.
        let rows = s.list_activity(10, None, None, None, None).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["seq"], 1);
        assert_eq!(rows[0]["port"], 11540);
        assert_eq!(rows[0]["gen_ai.request.model"], "m1");
        // Filters: model + before-timestamp pagination.
        let rows = s
            .list_activity(10, Some(2000), None, Some("m1"), None)
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["seq"], 0);
        let rows = s.list_activity(10, None, None, Some("nope"), None).unwrap();
        assert!(rows.is_empty());

        // Retention purge drops old rows; explicit purge drops the rest.
        assert_eq!(s.purge_activity_before(1500).unwrap(), 1);
        assert_eq!(s.clear_activity().unwrap(), 1);
    }

    /// What an upgraded database actually holds: codec blobs and
    /// pre-codec JSON TEXT rows in the same table. Both must list, and the
    /// new rows must really be stored as the compressed blob, not text.
    #[test]
    fn codec_and_legacy_rows_list_together() {
        let s = mem_store();
        assert_eq!(
            s.insert_activity("gen-new", 11540, 111, &[record(0, 2000, "mNew")])
                .unwrap(),
            1
        );
        {
            let conn = s.lock();
            conn.execute(
                "INSERT INTO activity (instance_id, seq, port, runner_started_at, ts_ms, record)
                 VALUES ('gen-old', 0, 11540, 100, 1000, '{\"seq\":0,\"gen_ai.request.model\":\"mOld\"}')",
                [],
            )
            .unwrap();
            let (kind, len): (String, i64) = conn
                .query_row(
                    "SELECT typeof(record), length(record) FROM activity WHERE instance_id = 'gen-new'",
                    [],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .unwrap();
            assert_eq!(kind, "blob", "new rows must land as codec blobs");
            let json_len = serde_json::to_string(&record(0, 2000, "mNew"))
                .unwrap()
                .len() as i64;
            assert!(
                len < json_len,
                "blob ({len} B) must undercut the JSON it replaced ({json_len} B)"
            );
        }
        let rows = s.list_activity(10, None, None, None, None).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["gen_ai.request.model"], "mNew", "codec row decodes");
        assert_eq!(
            rows[1]["gen_ai.request.model"], "mOld",
            "legacy TEXT row still reads"
        );
    }

    /// The regression: two generations on one port starting inside the
    /// same SECOND (a load-fail respawn, a fast takeover). Under the old
    /// (port, started_at, seq) key they shared a namespace and the second
    /// generation's rows were silently dropped; per-generation instance ids
    /// make the collision unrepresentable.
    #[test]
    fn same_second_respawn_keeps_both_generations() {
        let s = mem_store();
        // identical port AND started_at; both generations begin at seq 0
        assert_eq!(
            s.insert_activity("gen-1", 11540, 111, &[record(0, 1000, "mA")])
                .unwrap(),
            1
        );
        assert_eq!(
            s.insert_activity("gen-2", 11540, 111, &[record(0, 1001, "mB")])
                .unwrap(),
            1
        );
        let rows = s.list_activity(10, None, None, None, None).unwrap();
        assert_eq!(
            rows.len(),
            2,
            "the second generation's records must survive"
        );
        // and each generation resumes from its own cursor
        assert_eq!(s.activity_cursor("gen-1").unwrap(), 1);
        assert_eq!(s.activity_cursor("gen-2").unwrap(), 1);
    }

    /// Opening an earlier database rebuilds `activity` onto the new key,
    /// stamping old rows with the synthesized legacy id - the same one the
    /// collector derives for a runner that predates instance ids, so a live
    /// cursor keeps lining up across the upgrade.
    #[test]
    fn open_migrates_old_activity_to_instance_ids() {
        let dir = std::env::temp_dir().join(format!("paddock-store-207-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("paddock.db");
        {
            std::fs::create_dir_all(&dir).unwrap();
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE activity (
                     port              INTEGER NOT NULL,
                     runner_started_at INTEGER NOT NULL,
                     seq               INTEGER NOT NULL,
                     ts_ms             INTEGER NOT NULL,
                     request_id        TEXT NOT NULL DEFAULT '',
                     endpoint          TEXT NOT NULL DEFAULT '',
                     status            INTEGER NOT NULL DEFAULT 0,
                     model             TEXT,
                     session_id        TEXT,
                     record            TEXT NOT NULL,
                     PRIMARY KEY (port, runner_started_at, seq)
                 );
                 CREATE INDEX activity_ts ON activity(ts_ms DESC);
                 INSERT INTO activity VALUES
                     (11540, 111, 0, 1000, 'r0', '/v1/chat/completions', 200, 'm1', NULL, '{\"seq\":0}'),
                     (11540, 111, 1, 2000, 'r1', '/v1/chat/completions', 200, 'm1', NULL, '{\"seq\":1}'),
                     (11550, 222, 0, 3000, 'r2', '/v1/embeddings',       200, 'e1', NULL, '{\"seq\":0}');",
            )
            .unwrap();
        }
        let s = Store::open(&path).unwrap();
        // Every row survived, keyed by the synthesized per-generation id.
        assert_eq!(s.activity_cursor("legacy-11540-111").unwrap(), 2);
        assert_eq!(s.activity_cursor("legacy-11550-222").unwrap(), 1);
        let rows = s.list_activity(10, None, Some(11540), None, None).unwrap();
        assert_eq!(rows.len(), 2, "migrated rows still list by port");
        // A second open is a no-op (the shape check, not a re-run).
        drop(s);
        let s = Store::open(&path).unwrap();
        assert_eq!(s.activity_cursor("legacy-11540-111").unwrap(), 2);
        drop(s);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── usage tier  ──────────────────────────────────────────────

    #[test]
    fn usage_series_interning_is_stable() {
        let s = mem_store();
        let a = s.intern_usage_series(11540, "m1", "chat", "live").unwrap();
        let b = s.intern_usage_series(11540, "m1", "chat", "live").unwrap();
        assert_eq!(a, b, "same dimensions, same id");
        let c = s
            .intern_usage_series(11540, "m1", "chat", "studio")
            .unwrap();
        assert_ne!(a, c, "any dimension change is a new series");
    }

    #[test]
    fn fold_accumulates_into_both_grains_and_maxes_kv() {
        let s = mem_store();
        let id = s.intern_usage_series(11540, "m1", "chat", "live").unwrap();
        let mut d = crate::usage::BucketDelta {
            requests: 3,
            input_tokens: 100,
            duration_ms_sum: 450,
            kv_pages_max: 10,
            ..Default::default()
        };
        d.e2e[1] = 3;
        d.ttft[0] = 3;
        // Two folds inside one 5-minute bucket (ts 100_000 and 200_000).
        s.fold_usage_bucket(id, 100_000, &d).unwrap();
        d.kv_pages_max = 7; // gauge went down - the row keeps the high-water
        s.fold_usage_bucket(id, 200_000, &d).unwrap();

        let conn = s.lock();
        let (req, inp, dur, e2e1, ttft0, kv): (i64, i64, i64, i64, i64, i64) = conn
            .query_row(
                "SELECT requests, input_tokens, duration_ms_sum, e2e_h1, ttft_h0, kv_pages_max
                 FROM usage_bucket WHERE series_id = ?1 AND grain = '5m' AND bucket_start_ms = 0",
                params![id],
                |r| {
                    Ok((
                        r.get(0)?,
                        r.get(1)?,
                        r.get(2)?,
                        r.get(3)?,
                        r.get(4)?,
                        r.get(5)?,
                    ))
                },
            )
            .unwrap();
        assert_eq!(
            (req, inp, dur),
            (6, 200, 900),
            "additive columns accumulate"
        );
        assert_eq!((e2e1, ttft0), (6, 6), "histogram cells accumulate");
        assert_eq!(kv, 10, "kv high-water MAX-merges, never regresses");
        // The hourly grain got the same folds in its own row.
        let hourly_req: i64 = conn
            .query_row(
                "SELECT requests FROM usage_bucket
                 WHERE series_id = ?1 AND grain = '1h' AND bucket_start_ms = 0",
                params![id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(hourly_req, 6);
    }

    #[test]
    fn totals_overwrite_and_read_back_with_dimensions() {
        let s = mem_store();
        let id = s.intern_usage_series(11540, "m1", "chat", "live").unwrap();
        s.upsert_usage_total(id, "gen-a", 111_000, 10, 1000, 200, 50, 0, 0, 5_000)
            .unwrap();
        s.upsert_usage_total(id, "gen-a", 111_000, 25, 2500, 700, 90, 0, 0, 20_000)
            .unwrap();
        let rows = s.usage_totals_for_instance("gen-a").unwrap();
        assert_eq!(rows.len(), 1, "a high-water mark, not a time series");
        assert_eq!(rows[0].requests, 25);
        assert_eq!(rows[0].key.operation, "chat");
        assert_eq!(rows[0].last_scrape_ms, 20_000);
        // touch bumps the watch edge without touching counters
        s.touch_usage_totals("gen-a", 35_000).unwrap();
        assert_eq!(s.last_scrape_of("gen-a").unwrap(), Some(35_000));
        assert_eq!(
            s.usage_totals_for_instance("gen-a").unwrap()[0].requests,
            25
        );
        // a successor generation starts its own rows
        assert!(s.usage_totals_for_instance("gen-b").unwrap().is_empty());
    }

    #[test]
    fn generation_bands_open_close_and_keep_route_stamps() {
        let s = mem_store();
        assert!(
            s.open_generation(
                "gen-a",
                11540,
                42,
                "0.1.0",
                Some("m1"),
                None,
                None,
                None,
                1000,
                None,
            )
            .unwrap()
        );
        // re-attach after a manager restart: same row, not a new band
        assert!(
            !s.open_generation(
                "gen-a",
                11540,
                42,
                "0.1.0",
                Some("m1"),
                None,
                None,
                None,
                1000,
                None,
            )
            .unwrap()
        );

        // the stop route knows this was clean; the collector's later
        // observational close must not overwrite that knowledge
        s.stamp_end_cause(11540, "stopped").unwrap();
        s.close_generation("gen-a", 9000, "unknown").unwrap();
        let conn = s.lock();
        let (ended, cause): (i64, String) = conn
            .query_row(
                "SELECT ended_ms, end_cause FROM service_generation WHERE instance_id = 'gen-a'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!((ended, cause.as_str()), (9000, "stopped"));
        drop(conn);

        // an open predecessor is findable from a successor's attach...
        assert!(
            s.open_generation(
                "gen-b",
                11540,
                43,
                "0.1.0",
                Some("m2"),
                None,
                None,
                None,
                10_000,
                None,
            )
            .unwrap()
        );
        assert_eq!(
            s.open_predecessor_on_port(11540, "gen-b").unwrap(),
            None,
            "gen-a is closed"
        );
        assert!(
            s.open_generation(
                "gen-c",
                11540,
                44,
                "0.1.0",
                Some("m2"),
                None,
                None,
                None,
                20_000,
                None,
            )
            .unwrap()
        );
        assert_eq!(
            s.open_predecessor_on_port(11540, "gen-c")
                .unwrap()
                .as_deref(),
            Some("gen-b"),
            "...and it is the newest open band that is not the successor itself"
        );
    }

    /// a band has four serving roles to carry, and every one of them
    /// has to survive the round trip. An aligner-only runner holds nothing but
    /// `aligner`, so a table (or a SELECT) that stops at three renders it as a
    /// dash with no vendor mark on the usage chart - which is exactly what the
    /// Qwen3-ForcedAligner lane looked like until the column existed.
    #[test]
    fn a_band_names_its_model_in_every_serving_role() {
        let s = mem_store();
        let roles: [(&str, Option<&str>, Option<&str>, Option<&str>, Option<&str>); 4] = [
            ("gen-chat", Some("qwen3.6-27b"), None, None, None),
            ("gen-embed", None, Some("embeddinggemma-300m"), None, None),
            ("gen-asr", None, None, Some("kb-whisper-large"), None),
            (
                "gen-align",
                None,
                None,
                None,
                Some("Qwen3-ForcedAligner-0.6B"),
            ),
        ];
        for (i, (id, model, embedder, asr, aligner)) in roles.iter().enumerate() {
            let port = 11540 + i as u16;
            assert!(
                s.open_generation(
                    id, port, 1, "0.1.0", *model, *embedder, *asr, *aligner, 1000, None
                )
                .unwrap()
            );
        }
        let rows = s.usage_generations_in(0, 2000, -1).unwrap();
        assert_eq!(rows.len(), 4);
        let named: Vec<&str> = rows
            .iter()
            .map(|r| {
                r.model
                    .as_deref()
                    .or(r.embedder.as_deref())
                    .or(r.asr.as_deref())
                    .or(r.aligner.as_deref())
                    .expect("every band names the one model it served")
            })
            .collect();
        assert_eq!(
            named,
            [
                "qwen3.6-27b",
                "embeddinggemma-300m",
                "kb-whisper-large",
                "Qwen3-ForcedAligner-0.6B",
            ]
        );
    }

    /// The sweep's arithmetic, which is the whole reason it exists: a band
    /// nobody is running ends where its evidence ended, not when we noticed.
    /// Closing at `now` would charge a month of downtime as uptime - the shape
    /// of the bug this replaced, where two bands read "running" since August.
    #[test]
    fn an_orphaned_band_ends_at_its_last_heartbeat_not_at_the_sweep() {
        let s = mem_store();
        s.open_generation(
            "gen-a",
            11540,
            1,
            "0.1.0",
            Some("m"),
            None,
            None,
            None,
            1000,
            Some(500),
        )
        .unwrap();
        s.lock()
            .execute(
                "INSERT INTO usage_total (series_id, instance_id, started_ms, last_scrape_ms)
                 VALUES (1, 'gen-a', 1000, 5000)",
                [],
            )
            .unwrap();
        let ended = s
            .close_orphan_generation("gen-a", 1000, "unobserved")
            .unwrap();
        assert_eq!(ended, 5000, "the last scrape is the last thing we know");
        let (e, cause): (i64, String) = s
            .lock()
            .query_row(
                "SELECT ended_ms, end_cause FROM service_generation WHERE instance_id = 'gen-a'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!((e, cause.as_str()), (5000, "unobserved"));
    }

    /// A band that never reported has no heartbeat to end at, so its start is
    /// the only honest end - a zero-length band, not one running forever.
    #[test]
    fn a_band_that_never_reported_ends_at_its_start() {
        let s = mem_store();
        s.open_generation(
            "gen-a", 11540, 1, "0.1.0", None, None, None, None, 7000, None,
        )
        .unwrap();
        assert_eq!(
            s.close_orphan_generation("gen-a", 7000, "machine-restarted")
                .unwrap(),
            7000
        );
        assert!(
            s.open_generations().unwrap().is_empty(),
            "a swept band is not open any more"
        );
    }

    /// `open_generations` is the sweep's input: open bands only, with the boot
    /// they belong to, so the caller can tell a reboot from a blind window.
    #[test]
    fn open_generations_lists_only_unfinished_bands_with_their_boot() {
        let s = mem_store();
        s.open_generation(
            "gen-a",
            11540,
            1,
            "0.1.0",
            None,
            None,
            None,
            None,
            1000,
            Some(900),
        )
        .unwrap();
        s.open_generation(
            "gen-b", 11541, 2, "0.1.0", None, None, None, None, 2000, None,
        )
        .unwrap();
        s.close_generation("gen-a", 3000, "stopped").unwrap();
        let open = s.open_generations().unwrap();
        assert_eq!(open.len(), 1);
        assert_eq!(open[0].0, "gen-b");
        assert_eq!(open[0].1, 11541);
        assert_eq!(open[0].3, None, "an older row carries no boot reading");
    }

    #[test]
    fn a_noted_start_cause_labels_only_the_next_new_band() {
        let s = mem_store();
        s.note_start_cause(11540, "manual").unwrap();
        assert!(
            s.open_generation(
                "gen-a", 11540, 1, "0.1.0", None, None, None, None, 1000, None
            )
            .unwrap()
        );
        let cause: Option<String> = s
            .lock()
            .query_row(
                "SELECT start_cause FROM service_generation WHERE instance_id = 'gen-a'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(cause.as_deref(), Some("manual"));
        // consumed: the next band on the port carries no stale label
        assert!(
            s.open_generation(
                "gen-b", 11540, 2, "0.1.0", None, None, None, None, 2000, None
            )
            .unwrap()
        );
        let cause: Option<String> = s
            .lock()
            .query_row(
                "SELECT start_cause FROM service_generation WHERE instance_id = 'gen-b'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(cause, None, "a note is consumed by exactly one band");
    }

    #[test]
    fn gap_rows_carry_totals_or_seq_ranges_and_5m_purge_spares_hourly() {
        let s = mem_store();
        s.insert_usage_gap(
            11540,
            "gen-a",
            1000,
            9000,
            "manager-down",
            None,
            Some((28, 1000, 2000)),
        )
        .unwrap();
        s.insert_usage_gap(
            11540,
            "gen-a",
            500,
            500,
            "ring-overrun",
            Some((10, 25)),
            None,
        )
        .unwrap();
        let conn = s.lock();
        let (cause, lost_req, from_seq): (String, Option<i64>, Option<i64>) = conn
            .query_row(
                "SELECT cause, lost_requests, from_seq FROM usage_gap WHERE cause = 'manager-down'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .unwrap();
        assert_eq!(
            (cause.as_str(), lost_req, from_seq),
            ("manager-down", Some(28), None)
        );
        let (lost, seqs): (Option<i64>, Option<i64>) = conn
            .query_row(
                "SELECT lost_requests, to_seq FROM usage_gap WHERE cause = 'ring-overrun'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(
            (lost, seqs),
            (None, Some(25)),
            "event-tier holes know seqs, never totals"
        );
        drop(conn);

        let id = s.intern_usage_series(11540, "m1", "chat", "live").unwrap();
        let d = crate::usage::BucketDelta {
            requests: 1,
            ..Default::default()
        };
        s.fold_usage_bucket(id, 1_000, &d).unwrap();
        assert_eq!(
            s.purge_usage_5m_before(10_000_000).unwrap(),
            1,
            "only the 5m row goes"
        );
        let hourly: i64 = s
            .lock()
            .query_row(
                "SELECT COUNT(*) FROM usage_bucket WHERE grain = '1h'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(hourly, 1, "the hourly grain is kept indefinitely");
    }

    #[test]
    fn usage_history_groups_and_filters() {
        const H: i64 = 3_600_000;
        let s = mem_store();
        let a = s
            .intern_usage_series(11540, "m1", "chat", "studio")
            .unwrap();
        let b = s.intern_usage_series(11541, "m2", "chat", "live").unwrap();
        let mut d = crate::usage::BucketDelta {
            requests: 2,
            input_tokens: 100,
            cached_tokens: 40,
            output_tokens: 10,
            ..Default::default()
        };
        s.fold_usage_bucket(a, H, &d).unwrap();
        d.requests = 5;
        s.fold_usage_bucket(a, 2 * H, &d).unwrap();
        d.requests = 1;
        s.fold_usage_bucket(b, H, &d).unwrap();

        // hourly, no regroup: two slots for port a, one for b, ordered by t
        let rows = s.usage_history("1h", H, 0, i64::MAX, -1).unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!((rows[0].t, rows[0].port, rows[0].requests), (H, 11540, 2));
        assert_eq!((rows[1].t, rows[1].port, rows[1].requests), (H, 11541, 1));

        // regrouped to 6h: port a's two hours collapse into one summed slot
        let rows = s.usage_history("1h", 6 * H, 0, i64::MAX, -1).unwrap();
        assert_eq!(rows.len(), 2);
        let pa = rows.iter().find(|r| r.port == 11540).unwrap();
        assert_eq!(
            (pa.t, pa.requests, pa.input_tokens, pa.cached_tokens),
            (0, 7, 200, 80)
        );

        // window bounds respect from/to; port filter narrows
        assert_eq!(
            s.usage_history("1h", H, 2 * H, i64::MAX, -1).unwrap().len(),
            1
        );
        let rows = s.usage_history("1h", H, 0, i64::MAX, 11541).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].port, 11541);
    }

    /// one usage step commits its bucket folds, the totals, AND
    /// the full-state baseline together - the state is the crash-resume
    /// cursor, so any of the three moving separately would double-count or
    /// lose a replayed interval.
    #[test]
    fn usage_step_folds_totals_and_state_advance_together() {
        const H: i64 = 3_600_000;
        let s = mem_store();
        let id = s.intern_usage_series(11540, "m1", "chat", "live").unwrap();
        let items = vec![
            crate::usage::UsageFoldItem {
                series_id: id,
                delta: Some(crate::usage::BucketDelta {
                    requests: 3,
                    input_tokens: 120,
                    ..Default::default()
                }),
                requests: 10,
                input_tokens: 520,
                output_tokens: 55,
                cached_tokens: 40,
                spec_drafted: 0,
                spec_accepted: 0,
            },
            // totals-only advance: nothing moved, the sparse tier writes no row
            crate::usage::UsageFoldItem {
                series_id: s
                    .intern_usage_series(11540, "m1", "embeddings", "live")
                    .unwrap(),
                delta: None,
                requests: 7,
                input_tokens: 70,
                output_tokens: 0,
                cached_tokens: 0,
                spec_drafted: 0,
                spec_accepted: 0,
            },
        ];
        // Web spend rides the same step: one provider whose delta folds, one
        // that only advances its totals (the blind-window shape).
        let web = vec![
            crate::usage::WebFoldItem {
                provider: "exa".into(),
                delta: Some(crate::usage::WebSpend {
                    requests: 2,
                    credits: 0,
                    microdollars: 14_000,
                }),
                absolute: crate::usage::WebSpend {
                    requests: 12,
                    credits: 0,
                    microdollars: 84_000,
                },
            },
            crate::usage::WebFoldItem {
                provider: "brave".into(),
                delta: None,
                absolute: crate::usage::WebSpend {
                    requests: 5,
                    credits: 0,
                    microdollars: 0,
                },
            },
        ];
        s.fold_usage_step("inst-a", 1_000, H, &items, &web, 11540, "{\"seq\":0}")
            .unwrap();

        let rows = s.usage_history("1h", H, 0, i64::MAX, -1).unwrap();
        assert_eq!(rows.len(), 1, "the delta-less series must fold no bucket");
        assert_eq!(
            (rows[0].t, rows[0].requests, rows[0].input_tokens),
            (H, 3, 120)
        );

        let totals = s.usage_totals_for_instance("inst-a").unwrap();
        assert_eq!(totals.len(), 2, "both series' totals advance");
        let chat = totals.iter().find(|t| t.key.operation == "chat").unwrap();
        assert_eq!((chat.requests, chat.input_tokens), (10, 520));
        assert_eq!(chat.last_scrape_ms, H);

        let (ts, state) = s
            .usage_state_of("inst-a")
            .unwrap()
            .expect("state persisted");
        assert_eq!(
            (ts, state.as_str()),
            (H, "{\"seq\":0}"),
            "the state IS the resume cursor"
        );
        assert!(s.usage_state_of("inst-b").unwrap().is_none());

        // Spend: only the provider that moved gets a bucket row; both keep
        // exact lifetime totals, which is what survives a blind window.
        let web = s.web_history("1h", H, 0, i64::MAX, -1).unwrap();
        assert_eq!(web.len(), 1, "the delta-less provider must fold no bucket");
        assert_eq!(web[0].provider, "exa");
        assert_eq!(
            (web[0].t, web[0].requests, web[0].microdollars),
            (H, 2, 14_000)
        );
        assert_eq!(web[0].port, 11540);

        let conn = s.lock();
        let mut q = conn
            .prepare(
                "SELECT provider, requests, microdollars FROM web_search_total
                      WHERE instance_id = 'inst-a' ORDER BY provider",
            )
            .unwrap();
        let totals: Vec<(String, i64, i64)> = q
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
            .unwrap()
            .map(Result::unwrap)
            .collect();
        assert_eq!(
            totals,
            vec![("brave".into(), 5, 0), ("exa".into(), 12, 84_000)],
            "every provider's cumulative spend advances, folded or not"
        );
    }

    /// The whole seam on real exposition text: what the runner PRINTS, through
    /// the parser and the delta rule, into both tiers, and back out of the
    /// read query. The unit tests each cover one link; this is the one that
    /// fails if a family is ever renamed on one side only.
    #[test]
    fn web_spend_survives_the_scrape_to_chart_pipeline() {
        const H: i64 = 3_600_000;
        let s = mem_store();
        // Verbatim shape of paddock-runner's render_web_spend output.
        let scrape = |exa_req: u64, exa_micro: u64, brave: u64| {
            format!(
                "paddock_web_search_requests_total{{provider=\"exa\"}} {exa_req}\n\
                 paddock_web_search_cost_microdollars_total{{provider=\"exa\"}} {exa_micro}\n\
                 paddock_web_search_requests_total{{provider=\"brave\"}} {brave}\n"
            )
        };
        let fold = |prev: &crate::usage::Snapshot, text: &str, ts: i64| {
            let cur = crate::usage::parse(text);
            let moved: std::collections::HashMap<_, _> =
                crate::usage::web_deltas(prev, &cur).into_iter().collect();
            let items: Vec<_> = cur
                .web
                .iter()
                .map(|(p, abs)| crate::usage::WebFoldItem {
                    provider: p.clone(),
                    delta: moved.get(p).copied(),
                    absolute: *abs,
                })
                .collect();
            s.fold_usage_step("inst-a", 0, ts, &[], &items, 11540, "{}")
                .unwrap();
            cur
        };

        // First scrape of a fresh generation: everything deltas from zero.
        let a = fold(
            &crate::usage::Snapshot::default(),
            &scrape(2, 14_000, 1),
            H + 60_000,
        );
        // Second: only the movement folds.
        fold(&a, &scrape(5, 35_000, 1), H + 360_000);

        let rows = s.web_history("5m", 300_000, 0, i64::MAX, -1).unwrap();
        let exa: Vec<_> = rows.iter().filter(|r| r.provider == "exa").collect();
        assert_eq!(exa.len(), 2);
        assert_eq!((exa[0].requests, exa[0].microdollars), (2, 14_000));
        assert_eq!((exa[1].requests, exa[1].microdollars), (3, 21_000));
        // Brave was flat across the second scrape, so it folds one row only -
        // and never a money row, because Brave quotes no price.
        let brave: Vec<_> = rows.iter().filter(|r| r.provider == "brave").collect();
        assert_eq!(brave.len(), 1, "a flat provider writes no second bucket");
        assert_eq!((brave[0].requests, brave[0].microdollars), (1, 0));

        // The hourly rollup is the window total, and it matches the counter.
        let hourly = s.web_history("1h", H, 0, i64::MAX, -1).unwrap();
        let exa_h = hourly.iter().find(|r| r.provider == "exa").unwrap();
        assert_eq!((exa_h.requests, exa_h.microdollars), (5, 35_000));
    }

    /// A second scrape ADDS into the same bucket and OVERWRITES the total -
    /// the two tiers have opposite merge rules and getting them the wrong way
    /// round would either double-count lifetime spend or flatten the timeline.
    #[test]
    fn web_spend_buckets_accumulate_while_totals_overwrite() {
        const H: i64 = 3_600_000;
        let s = mem_store();
        let step = |ts: i64, delta: u64, abs: u64| {
            let web = vec![crate::usage::WebFoldItem {
                provider: "tavily".into(),
                delta: Some(crate::usage::WebSpend {
                    requests: delta,
                    credits: delta,
                    microdollars: 0,
                }),
                absolute: crate::usage::WebSpend {
                    requests: abs,
                    credits: abs,
                    microdollars: 0,
                },
            }];
            s.fold_usage_step("inst-a", 0, ts, &[], &web, 11540, "{}")
                .unwrap();
        };
        step(H + 60_000, 3, 3);
        step(H + 360_000, 4, 7); // a later 5-minute bucket, the same hour

        // Same hour, two 5-minute buckets: the hourly row sums both scrapes.
        let hourly = s.web_history("1h", H, 0, i64::MAX, -1).unwrap();
        assert_eq!(hourly.len(), 1);
        assert_eq!((hourly[0].requests, hourly[0].credits), (7, 7));
        let fine = s.web_history("5m", 300_000, 0, i64::MAX, -1).unwrap();
        assert_eq!(fine.len(), 2, "distinct 5-minute buckets: {fine:?}");

        let conn = s.lock();
        let total: i64 = conn
            .query_row(
                "SELECT requests FROM web_search_total WHERE instance_id = 'inst-a'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            total, 7,
            "the total is the last absolute, never a sum of deltas"
        );
    }

    #[test]
    fn usage_read_window_overlap_for_gaps_and_generations() {
        let s = mem_store();
        s.insert_usage_gap(
            11540,
            "gen-a",
            1000,
            5000,
            "manager-down",
            None,
            Some((3, 10, 20)),
        )
        .unwrap();
        s.insert_usage_gap(
            11541,
            "gen-b",
            9000,
            12000,
            "ring-overrun",
            Some((1, 5)),
            None,
        )
        .unwrap();
        let g = s.usage_gaps_in(4000, 8000, -1).unwrap();
        assert_eq!(
            g.len(),
            1,
            "only the overlapping gap is in a [4000,8000) window"
        );
        assert_eq!(g[0].cause, "manager-down");
        assert_eq!(g[0].lost_requests, Some(3));
        assert_eq!(s.usage_gaps_in(0, 20_000, -1).unwrap().len(), 2);
        assert_eq!(s.usage_gaps_in(0, 20_000, 11540).unwrap().len(), 1);

        s.open_generation(
            "gen-a",
            11540,
            1,
            "0.1.0",
            Some("m"),
            None,
            None,
            None,
            1000,
            None,
        )
        .unwrap();
        s.close_generation("gen-a", 2000, "stopped").unwrap();
        s.open_generation(
            "gen-b",
            11540,
            2,
            "0.1.0",
            Some("m"),
            None,
            None,
            None,
            3000,
            None,
        )
        .unwrap();
        let bands = s.usage_generations_in(10_000, 20_000, -1).unwrap();
        assert_eq!(
            bands.len(),
            1,
            "an OPEN band overlaps any window reaching now"
        );
        assert_eq!(bands[0].instance_id, "gen-b");
        assert_eq!(bands[0].end_cause, None);
        let bands = s.usage_generations_in(0, 2_500, 11540).unwrap();
        assert_eq!(
            bands.len(),
            1,
            "gen-b starts at 3000, after this window ends"
        );
        assert_eq!(bands[0].instance_id, "gen-a");
        assert_eq!(bands[0].end_cause.as_deref(), Some("stopped"));
        let bands = s.usage_generations_in(0, 5_000, 11540).unwrap();
        assert_eq!(
            bands.len(),
            2,
            "a window past gen-b's start sees both bands"
        );
    }

    // ── forensic reports (paddock-forensics persistence) ─────────────────────

    fn finding(analyzer: &str, code: &str, sev: &str, region: &str) -> NewForensicFinding {
        NewForensicFinding {
            analyzer: analyzer.into(),
            code: code.into(),
            severity: sev.into(),
            confidence: 0.8,
            description: format!("{analyzer}: {code}"),
            region: region.into(),
        }
    }

    #[test]
    fn forensic_report_round_trips_with_derived_aggregates() {
        let s = mem_store();
        let rep = NewForensicReport {
            attachment_id: Some("att-1".into()),
            conversation_id: Some("conv-1".into()),
            sha256: "deadbeef".into(),
            kind: "image".into(),
            mime: "image/jpeg".into(),
            name: "photo.jpg".into(),
            width: Some(640),
            height: Some(480),
            content_type: "photo".into(),
            format: "jpeg".into(),
            risk_score: 0.72,
            verdict: "Likely manipulated".into(),
            risk_level: "high".into(),
            corroborating_stages: 2,
            explanation_summary: "Multiple independent signals of tampering.".into(),
            explanation_visual_review: Some("Check the sky region.".into()),
            explanation_cross_corroboration: Some("Noise and ELA agree.".into()),
            explanation_anti_forensics: None,
            key_findings: vec![NewForensicKeyFinding {
                title: "Local noise inconsistency".into(),
                description: "A region's noise floor differs from the frame.".into(),
                severity: "high".into(),
                confidence: 0.81,
                sources: vec!["noise".into(), "ela".into()],
                region: r#"{"type":"bounding_box","x":5,"y":6,"width":7,"height":8}"#.into(),
                count: 3,
            }],
            explanation_categories: vec![NewForensicExplanationCategory {
                name: "Sensor noise".into(),
                finding_count: 2,
                max_severity: "high".into(),
                explanation: "Noise statistics vary across the frame.".into(),
                finding_codes: vec!["noise_inconsistency".into()],
            }],
            gpu: true,
            elapsed_ms: 42,
            findings: vec![
                finding("ela", "ela_block_outliers", "low", ""),
                finding(
                    "noise",
                    "noise_inconsistency",
                    "high",
                    r#"{"type":"bounding_box","x":1,"y":2,"width":3,"height":4}"#,
                ),
            ],
        };
        let id = s.save_forensic_report(&rep).unwrap();

        let got = s.get_forensic_report(&id).unwrap().expect("report exists");
        assert_eq!(got["kind"], "image");
        assert_eq!(got["gpu"], true, "gpu bool round-trips");
        assert_eq!(got["finding_count"], 2, "count derived from findings");
        assert_eq!(got["max_severity"], "high", "highest severity present wins");
        // Risk layer round-trips in full.
        assert_eq!(got["risk_score"], 0.72);
        assert_eq!(
            got["risk_level"], "high",
            "leveled verdict, distinct from max_severity"
        );
        assert_eq!(got["corroborating_stages"], 2);
        assert_eq!(
            got["explanation"]["summary"],
            "Multiple independent signals of tampering."
        );
        assert_eq!(got["explanation"]["visual_review"], "Check the sky region.");
        assert_eq!(
            got["explanation"]["anti_forensics_warning"],
            Value::Null,
            "None -> null"
        );
        let kf = got["key_findings"].as_array().unwrap();
        assert_eq!(kf.len(), 1);
        assert_eq!(kf[0]["title"], "Local noise inconsistency");
        assert_eq!(kf[0]["count"], 3, "collapsed raw-finding count");
        assert_eq!(
            kf[0]["sources"],
            json!(["noise", "ela"]),
            "sources JSON re-parses"
        );
        assert_eq!(
            kf[0]["region"]["type"], "bounding_box",
            "key-finding region re-parses"
        );
        let cats = got["explanation"]["categories"].as_array().unwrap();
        assert_eq!(cats.len(), 1);
        assert_eq!(cats[0]["name"], "Sensor noise");
        assert_eq!(cats[0]["finding_codes"], json!(["noise_inconsistency"]));
        let findings = got["findings"].as_array().unwrap();
        assert_eq!(findings.len(), 2);
        assert_eq!(findings[0]["seq"], 0, "findings keep insertion order");
        assert_eq!(findings[0]["region"], Value::Null, "empty region -> null");
        assert_eq!(
            findings[1]["region"]["type"], "bounding_box",
            "region JSON re-parses back to structure"
        );

        // Listing by conversation returns the summary (header + scalar risk,
        // no child collections).
        let list = s.list_forensic_reports("conv-1").unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0]["id"], id);
        assert_eq!(list[0]["risk_level"], "high", "summary carries the verdict");
        assert_eq!(
            list[0]["explanation"]["summary"], "Multiple independent signals of tampering.",
            "summary carries the narrative"
        );
        assert!(list[0].get("findings").is_none(), "list is header-only");
        assert!(
            list[0].get("key_findings").is_none(),
            "list omits child collections"
        );

        // Latest-for-attachment resolves back to the full report.
        let latest = s
            .latest_forensic_report_for_attachment("att-1")
            .unwrap()
            .expect("attachment has a report");
        assert_eq!(latest["id"], id);
        assert!(
            s.latest_forensic_report_for_attachment("nope")
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn forensic_report_with_no_findings_is_info() {
        let s = mem_store();
        let rep = NewForensicReport {
            sha256: "00".into(),
            kind: "pdf".into(),
            mime: "application/pdf".into(),
            name: "clean.pdf".into(),
            content_type: "document".into(),
            elapsed_ms: 5,
            ..Default::default()
        };
        let id = s.save_forensic_report(&rep).unwrap();
        let got = s.get_forensic_report(&id).unwrap().unwrap();
        assert_eq!(got["finding_count"], 0);
        assert_eq!(got["max_severity"], "info", "a clean report reads as info");
        assert_eq!(got["risk_level"], "info", "a clean report levels as info");
        assert_eq!(got["findings"].as_array().unwrap().len(), 0);
        assert_eq!(got["key_findings"].as_array().unwrap().len(), 0);
        assert_eq!(
            got["explanation"]["categories"].as_array().unwrap().len(),
            0
        );
    }

    #[test]
    fn attachment_metadata_writes_through_and_reads_lite() {
        let s = mem_store();
        s.put_attachment(
            "att-m",
            Some("conv-m"),
            "image/jpeg",
            "p.jpg",
            Some(4),
            Some(4),
            b"twelve-bytes!",
        )
        .unwrap();

        // Before any write-through: identity + size resolve, metadata is None.
        let (mime, name, size, meta) = s.attachment_lite("att-m").unwrap().expect("attachment");
        assert_eq!(mime, "image/jpeg");
        assert_eq!(name, "p.jpg");
        assert_eq!(size, 13, "length(bytes) without materializing the blob");
        assert!(meta.is_none(), "no metadata until the runner ships it");

        // Write through the runner-shipped metadata; it reads back verbatim.
        s.set_attachment_metadata("att-m", r#"{"format":"jpeg","groups":[]}"#)
            .unwrap();
        let (_, _, _, meta) = s.attachment_lite("att-m").unwrap().unwrap();
        assert_eq!(meta.as_deref(), Some(r#"{"format":"jpeg","groups":[]}"#));

        // Unknown attachment -> None, not an error.
        assert!(s.attachment_lite("nope").unwrap().is_none());
    }
}
