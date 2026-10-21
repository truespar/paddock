//! Batch inference: the vendors' batch APIs, run by the manager against the
//! services already serving.
//!
//! Store-and-forward, never a proxy. The caller's connection ends when the
//! batch is accepted; the manager then originates the runner calls itself, as
//! an API client, the way the Studio relay and compare already do. Each call
//! carries `x-paddock-origin: batch`, so the usage tier and the forecaster can
//! tell batch traffic from live traffic.
//!
//! Two dialects over one job store:
//! - OpenAI: `/v1/files` (a JSONL of `{custom_id, method, url, body}` lines),
//!   then `/v1/batches`, then poll, then the output file.
//! - Anthropic: `/v1/messages/batches`, with inline requests and JSONL
//!   results.
//!
//! The run is built to last for days and to survive anything a box can do
//! during them:
//! - The input is streamed to disk and read back one item at a time.
//! - Results are appended and group-committed (`crate::store::batches` holds
//!   the exactly-once invariant).
//! - A batch is bound to a model, not a port, so it waits out a service that
//!   is down or restarting.
//! - A request that keeps killing its runner is isolated and failed on its
//!   own, instead of crash-looping the run.
//!
//! Module map:
//! - `files`: the bytes - uploads, positional reads, result files.
//! - `index`: line validation and the Anthropic-to-lines conversion.
//! - `dispatch`: the run itself.
//! - `openai`, `anthropic`: the two HTTP dialects.
//! - `uploads`: OpenAI's resumable Uploads API, the way in for big inputs.
//! - `webhook`: completion delivery.

pub(crate) mod anthropic;
pub(crate) mod dispatch;
pub(crate) mod files;
pub(crate) mod index;
pub(crate) mod openai;
#[cfg(test)]
mod tests;
pub(crate) mod uploads;
pub(crate) mod webhook;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::Router;
use futures::future::BoxFuture;
use serde::Serialize;

use crate::routes::AppState;
use crate::store::Store;

/// Largest request count one batch may carry. Ten times Anthropic's cap
/// (twenty times OpenAI's): a local box has no fleet to share, and a photo
/// archive is easily six figures. What bounds it is validation's custom_id set
/// (~50 MB at the cap) and the store, which holds one row per request only
/// while the batch is open.
pub const MAX_REQUESTS: usize = 1_000_000;

/// Largest single request line - the runner's own body cap, so a line the
/// index accepts is never one the runner would refuse for size.
pub const MAX_LINE: usize = 192 * 1024 * 1024;

/// The endpoints a batch line may target.
pub const ENDPOINTS: [&str; 5] = [
    "/v1/chat/completions",
    "/v1/responses",
    "/v1/completions",
    "/v1/embeddings",
    "/v1/messages",
];

/// Knobs of the run. Production uses `Default`; tests shrink the clocks.
#[derive(Debug, Clone)]
pub struct Tuning {
    /// One request's ceiling. Generous on purpose: a long answer about a
    /// picture on a busy 27B takes minutes, and a request that is merely slow
    /// must not be retried into a second copy of itself.
    pub request_timeout: Duration,
    /// Group commit: sync and record results at least this often...
    pub commit_every: Duration,
    /// ...or once this many have accumulated.
    pub commit_items: usize,
    /// How long a service must be free of live traffic before batch work
    /// resumes on it.
    pub yield_dwell: Duration,
    /// How often the run re-reads service health.
    pub tick: Duration,
    /// Attempts at an item that keeps failing on the server side.
    pub max_attempts: i64,
    /// First retry delay; doubles per attempt.
    pub retry_base: Duration,
    /// Crash strikes after which an item runs alone; one more alone fails it.
    pub isolate_after: i64,
    /// How soon a batch whose service went away looks for it again.
    pub rebind: Duration,
    /// How long a run paused by a failed result write waits before checking
    /// the disk again.
    pub disk_retry: Duration,
    /// Timeouts in a row, with nothing answered between them, that mark a
    /// service as wedged rather than a request as slow.
    pub wedge_after: u32,
    /// How long a wedged service is left alone before one request probes it.
    pub wedge_pause: Duration,
}

impl Default for Tuning {
    fn default() -> Self {
        Self {
            request_timeout: Duration::from_secs(30 * 60),
            commit_every: Duration::from_millis(250),
            // a ceiling, not a cadence: at a picture a second the 250 ms timer
            // commits long before this; at thousands of answers a second it
            // keeps commits (each an fsync under the store lock the Studio
            // also takes) to a handful a second instead of a hundred
            commit_items: 1024,
            yield_dwell: Duration::from_secs(30),
            tick: Duration::from_secs(1),
            max_attempts: 3,
            retry_base: Duration::from_secs(2),
            isolate_after: 2,
            rebind: Duration::from_secs(5),
            disk_retry: Duration::from_secs(30),
            wedge_after: 3,
            wedge_pause: Duration::from_secs(300),
        }
    }
}

/// A running service a batch can be sent to.
#[derive(Debug, Clone, PartialEq)]
pub struct Service {
    pub port: u16,
    pub pid: u32,
    /// `http://127.0.0.1:<port>` in production; a test server otherwise.
    pub base: String,
    pub key: Option<String>,
    /// The runner's continuous-batching width.
    pub max_batch: usize,
}

/// One health sample of a service.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Health {
    pub in_flight: u64,
    pub draining: bool,
    pub pid: u32,
    /// The served model still is the one asked about.
    pub serves: bool,
}

/// Where the run finds services. Production reads the runners' admin sockets;
/// tests stand up a fake runner.
pub trait Fleet: Send + Sync + 'static {
    /// The service serving `model` for `endpoint` right now, if any.
    fn resolve<'a>(&'a self, model: &'a str, endpoint: &'a str) -> BoxFuture<'a, Option<Service>>;
    /// Every model id served for `endpoint` (for the refusal that names them).
    fn served<'a>(&'a self, endpoint: &'a str) -> BoxFuture<'a, Vec<String>>;
    /// A health sample of the service on `port`, as seen for `model`.
    fn health<'a>(&'a self, port: u16, model: &'a str) -> BoxFuture<'a, Option<Health>>;
}

/// What the run is doing for one batch, as the `paddock` block of a batch
/// object reports it.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Progress {
    /// validating | running | queued | waiting_for_service | yielding |
    /// backing_off | isolating | cancelling | finalizing
    pub state: &'static str,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub port: Option<u16>,
    pub in_flight: usize,
    /// The service's dispatch width right now.
    pub concurrency: usize,
    /// Requests settled per minute over the recent window.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rate_per_minute: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eta_seconds: Option<u64>,
    /// Requests sent again after a server-side failure or a lost runner.
    pub retries: u64,
}

/// The batch runtime, shared by the HTTP dialects and the run.
pub struct Batches {
    pub(crate) root: PathBuf,
    pub(crate) db: Arc<Store>,
    pub(crate) fleet: Arc<dyn Fleet>,
    pub(crate) tuning: Tuning,
    /// Wakes the run: a batch was created or cancelled.
    pub(crate) wake: tokio::sync::Notify,
    pub(crate) progress: Mutex<HashMap<String, Progress>>,
}

impl Batches {
    pub fn new(root: PathBuf, db: Arc<Store>, fleet: Arc<dyn Fleet>, tuning: Tuning) -> Self {
        Self {
            root,
            db,
            fleet,
            tuning,
            wake: tokio::sync::Notify::new(),
            progress: Mutex::new(HashMap::new()),
        }
    }

    /// The production runtime under `<data>/batches`.
    pub fn for_supervisor(
        data: &Path,
        db: Arc<Store>,
        sup: Arc<crate::supervisor::Supervisor>,
    ) -> Self {
        Self::new(
            data.join("batches"),
            db,
            Arc::new(SupervisorFleet::new(sup)),
            Tuning::default(),
        )
    }

    /// Start the run and the webhook courier. Called once per process.
    pub fn start(self: &Arc<Self>) {
        tokio::spawn(dispatch::run(self.clone()));
        tokio::spawn(webhook::run(self.clone()));
    }

    pub(crate) fn progress_of(&self, id: &str) -> Option<Progress> {
        self.progress
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(id)
            .cloned()
    }

    pub(crate) fn files_dir(&self) -> PathBuf {
        self.root.join("files")
    }

    pub(crate) fn job_dir(&self, batch_id: &str) -> PathBuf {
        self.root.join("jobs").join(batch_id)
    }

    /// Where a batch's input lines are: the uploaded file (OpenAI) or the
    /// converted inline requests (Anthropic).
    pub(crate) fn input_path(&self, row: &crate::store::BatchRow) -> Option<PathBuf> {
        if row.dialect == "anthropic" {
            return Some(self.job_dir(&row.id).join("in.jsonl"));
        }
        let f = self.db.batch_file_get(&row.input_file_id).ok()??;
        Some(self.root.join(f.path))
    }
}

/// The HTTP surface, both dialects. Mounted at the manager's root so vendor
/// SDKs work with nothing but a base URL.
pub fn routes() -> Router<Arc<AppState>> {
    openai::routes().merge(anthropic::routes())
}

/// A vendor-style id: the prefix plus 32 hex digits.
pub(crate) fn new_id(prefix: &str) -> String {
    format!("{prefix}{}", uuid::Uuid::new_v4().simple())
}

/// The request's `Idempotency-Key`, if it sent a usable one. A create or upload
/// retried under the same key returns the first result instead of doing the
/// work again - the case is a reply lost to a dropped connection, which an SDK
/// then retries on its own.
pub(crate) fn idem_key(h: &axum::http::HeaderMap) -> Option<String> {
    h.get("idempotency-key")
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|k| !k.is_empty() && k.len() <= 255)
        .map(str::to_owned)
}

pub(crate) fn now_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

/// `"24h"`, `"7d"` -> milliseconds; 1 h to 30 d.
pub(crate) fn parse_window(w: &str) -> Option<i64> {
    let (n, unit) = w.split_at(w.len().checked_sub(1)?);
    let n: i64 = n.parse().ok()?;
    let ms = match unit {
        "h" => n.checked_mul(3_600_000)?,
        "d" => n.checked_mul(86_400_000)?,
        _ => return None,
    };
    (3_600_000..=30 * 86_400_000).contains(&ms).then_some(ms)
}

/// The served-id comparison: exact, else ASCII case-insensitive (ids are
/// lower-case by convention but a client may echo a catalog name).
pub(crate) fn same_model(a: &str, b: &str) -> bool {
    a == b || a.eq_ignore_ascii_case(b)
}

/// The production fleet: the runners' admin sockets, cached briefly so a run
/// on ten batches does not identify every port ten times a second.
pub struct SupervisorFleet {
    sup: Arc<crate::supervisor::Supervisor>,
    http: reqwest::Client,
    /// pid -> max_batch, read once per runner process.
    widths: Mutex<HashMap<u32, usize>>,
    /// The last port sweep and when it was taken.
    sweep: tokio::sync::Mutex<(Option<Instant>, Vec<(u16, paddock_admin::types::Identify)>)>,
}

impl SupervisorFleet {
    pub fn new(sup: Arc<crate::supervisor::Supervisor>) -> Self {
        Self {
            sup,
            http: reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(3))
                .timeout(Duration::from_secs(5))
                .build()
                .unwrap_or_default(),
            widths: Mutex::new(HashMap::new()),
            sweep: tokio::sync::Mutex::new((None, Vec::new())),
        }
    }

    /// Every answering runner and what it serves, at most 3 s old.
    async fn identities(&self) -> Vec<(u16, paddock_admin::types::Identify)> {
        let mut g = self.sweep.lock().await;
        if g.0.is_some_and(|t| t.elapsed() < Duration::from_secs(3)) {
            return g.1.clone();
        }
        let mut ports = paddock_admin::enumerate();
        for (p, _, _) in self.sup.fleet_meta().await {
            if !ports.contains(&p) {
                ports.push(p);
            }
        }
        ports.sort_unstable();
        let mut out = Vec::new();
        for port in ports {
            let id = tokio::time::timeout(
                Duration::from_secs(2),
                paddock_admin::client::AdminClient::new(port).identify(),
            )
            .await;
            if let Ok(Ok(id)) = id {
                out.push((port, id));
            }
        }
        *g = (Some(Instant::now()), out.clone());
        out
    }

    /// The runner's `max_batch`, from its own `/api/server`.
    async fn width(&self, port: u16, pid: u32, key: Option<&str>) -> usize {
        if let Some(w) = self
            .widths
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&pid)
        {
            return *w;
        }
        let mut req = self.http.get(format!("http://127.0.0.1:{port}/api/server"));
        if let Some(k) = key {
            req = req.bearer_auth(k);
        }
        let w = match req.send().await {
            Ok(r) => r
                .json::<serde_json::Value>()
                .await
                .ok()
                .and_then(|v| v["max_batch"].as_u64())
                .map(|w| w.clamp(1, 1024) as usize),
            Err(_) => None,
        };
        match w {
            Some(w) => {
                self.widths
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .insert(pid, w);
                w
            }
            // unknown is not cached: next resolve asks again
            None => 4,
        }
    }
}

/// The id a runner serves for `endpoint`'s role.
fn role_id<'a>(id: &'a paddock_admin::types::Identify, endpoint: &str) -> Option<&'a str> {
    if endpoint == "/v1/embeddings" {
        id.embedder.as_deref()
    } else {
        id.model.as_deref()
    }
}

impl Fleet for SupervisorFleet {
    fn resolve<'a>(&'a self, model: &'a str, endpoint: &'a str) -> BoxFuture<'a, Option<Service>> {
        Box::pin(async move {
            let ids = self.identities().await;
            let (port, id) = ids
                .into_iter()
                .find(|(_, id)| role_id(id, endpoint).is_some_and(|m| same_model(m, model)))?;
            let key = self.sup.runner_key(port).await;
            let max_batch = self.width(port, id.pid, key.as_deref()).await;
            Some(Service {
                port,
                pid: id.pid,
                base: format!("http://127.0.0.1:{port}"),
                key,
                max_batch,
            })
        })
    }

    fn served<'a>(&'a self, endpoint: &'a str) -> BoxFuture<'a, Vec<String>> {
        Box::pin(async move {
            self.identities()
                .await
                .iter()
                .filter_map(|(_, id)| role_id(id, endpoint).map(str::to_owned))
                .collect()
        })
    }

    fn health<'a>(&'a self, port: u16, model: &'a str) -> BoxFuture<'a, Option<Health>> {
        Box::pin(async move {
            let c = paddock_admin::client::AdminClient::new(port);
            let id = tokio::time::timeout(Duration::from_secs(2), c.identify())
                .await
                .ok()?
                .ok()?;
            let h = tokio::time::timeout(Duration::from_secs(2), c.health())
                .await
                .ok()?
                .ok()?;
            Some(Health {
                in_flight: h.in_flight,
                draining: h.status == "draining",
                pid: id.pid,
                // either role: the batch's endpoint picked it at bind time
                serves: [id.model.as_deref(), id.embedder.as_deref()]
                    .into_iter()
                    .flatten()
                    .any(|m| same_model(m, model)),
            })
        })
    }
}

/// A fleet with nothing in it (tests that never dispatch).
pub struct NoFleet;

impl Fleet for NoFleet {
    fn resolve<'a>(&'a self, _: &'a str, _: &'a str) -> BoxFuture<'a, Option<Service>> {
        Box::pin(async { None })
    }
    fn served<'a>(&'a self, _: &'a str) -> BoxFuture<'a, Vec<String>> {
        Box::pin(async { Vec::new() })
    }
    fn health<'a>(&'a self, _: u16, _: &'a str) -> BoxFuture<'a, Option<Health>> {
        Box::pin(async { None })
    }
}
