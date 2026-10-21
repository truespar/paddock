//! The run: one task that keeps every serving port as full as its batches can
//! make it, for as long as they take.
//!
//! Width. A runner batches up to `max_batch` sequences, so the run keeps that
//! many requests in flight plus a quarter more (at least one) waiting in the
//! runner's own queue. A slot that frees is then refilled by the runner at
//! once, not after the manager has seen the answer, read the next item off disk
//! and sent it. Backpressure (429/503/529, which is drain or the admission cap)
//! halves the width and honors Retry-After, and each success grows it back by
//! one.
//!
//! Binding. A batch names a model, not a port. Each tick re-reads health on
//! every port in use. A port that stops answering, restarts as a different
//! process or switches models drops its batches back to "waiting for
//! service", and they rebind wherever that model is served next. Nothing is
//! ever sent to a model the batch did not name.
//!
//! Live traffic. Strict time separation, per the plan, because the engine has
//! no priority tier. The runner's in-flight count, minus ours, is somebody
//! else's request. Once seen, new batch dispatch stops until the service has
//! been quiet for the dwell. Batch requests already sent run to completion.
//!
//! Failures, by whose fault they are:
//! - The item's (a 4xx): recorded verbatim, never retried.
//! - The service's (a 5xx, a timeout): retried with backoff, then recorded.
//! - Nobody's (connection refused, backpressure): retried without counting.
//! - A runner that died with items in flight: each of those items gets a
//!   crash strike. Struck twice, an item runs alone. If it kills the runner
//!   alone, it is recorded as the cause and the run moves on. One poisoned
//!   picture cannot crash-loop ten thousand.
//!
//! Durability is the group commit described in `crate::store::batches`:
//! results are appended, synced, then recorded, at most every
//! `commit_every` or `commit_items`.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use tokio::task::JoinSet;

use super::files::ResultFile;
use super::{Batches, Progress, Service, new_id, now_ms};
use crate::store::{
    BatchCommit, BatchItemRow, BatchItemUpdate, BatchRow, ITEM_CANCELED, ITEM_ERRORED,
    ITEM_EXPIRED, ITEM_PENDING, ITEM_SUCCEEDED,
};

/// Pending items read from the store per query.
const PREFETCH: usize = 1024;
/// Recent settles kept for the rate and ETA.
const RATE_WINDOW: usize = 512;

#[derive(Debug, Clone)]
struct Item {
    idx: i64,
    custom_id: String,
    off: i64,
    len: i64,
    attempts: i64,
    strikes: i64,
}

impl From<BatchItemRow> for Item {
    fn from(r: BatchItemRow) -> Self {
        Self {
            idx: r.idx,
            custom_id: r.custom_id,
            off: r.off,
            len: r.len,
            attempts: r.attempts,
            strikes: r.strikes,
        }
    }
}

#[derive(Debug)]
enum Outcome {
    Http {
        status: u16,
        request_id: Option<String>,
        body: Vec<u8>,
        retry_after: Option<Duration>,
    },
    /// Nothing reached the runner.
    Refused(String),
    /// The connection died with the request on it.
    Broken(String),
    Timeout,
    /// The item's bytes could not be read from the batch input.
    Input(String),
}

struct Sent {
    batch: String,
    item: Item,
    port: u16,
    /// The runner process it went to: a port that has since restarted is a
    /// new `Port`, and this one's answer must not settle that one's count.
    pid: u32,
    alone: bool,
    abort: tokio::task::AbortHandle,
}

/// How a run is being wound down.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Stop {
    Cancel,
    Expire,
}

struct Run {
    row: BatchRow,
    input: PathBuf,
    /// OpenAI's output file, or Anthropic's single results file.
    out: ResultFile,
    /// OpenAI's error file.
    err: Option<ResultFile>,
    queue: VecDeque<Item>,
    retry: Vec<(Instant, Item)>,
    isolate: VecDeque<Item>,
    /// Highest idx read from the store; None before the first read.
    cursor: i64,
    exhausted: bool,
    in_flight: usize,
    commit: BatchCommit,
    commit_since: Option<Instant>,
    /// Settled in memory but not yet committed (the ETA counts them).
    unflushed: i64,
    port: Option<u16>,
    resolve_at: Instant,
    /// paddock.concurrency, if the batch set one.
    cap: usize,
    yield_to_live: bool,
    stop: Option<Stop>,
    aborted: bool,
    state: &'static str,
    reason: String,
    settled: VecDeque<Instant>,
    retries: u64,
    /// A result write failed; dispatch pauses until the disk recovers. Sending
    /// on regardless would re-run requests whose answers cannot be kept, so the
    /// space is re-checked on a timer (`disk_until`) and the run resumes on its
    /// own once a write can succeed again.
    disk: Option<String>,
    disk_until: Option<Instant>,
}

struct Port {
    svc: Service,
    model: String,
    target: usize,
    width: usize,
    in_flight: usize,
    /// Our requests the runner may be holding right now: raised just before a
    /// send, dropped the moment the answer is read, inside the request task.
    /// `in_flight` lags it by however long the loop takes to see a finished
    /// task, and a lagging count reads the runner's own finished work as
    /// somebody else's absence - the live-traffic estimate needs this one.
    on_wire: Arc<AtomicUsize>,
    backoff_until: Option<Instant>,
    yield_until: Option<Instant>,
    /// An isolated item is in flight: nothing else goes out.
    alone: bool,
    /// A connection broke: hold until health answers again.
    hold: bool,
    /// Timeouts in a row with no answer between them.
    timeouts: u32,
    /// The service is answering health checks but not requests: send nothing
    /// until then, and one request at a time after.
    wedged_until: Option<Instant>,
    health_at: Instant,
    down: bool,
}

impl Port {
    fn new(svc: Service, model: &str) -> Self {
        let target = svc.max_batch + (svc.max_batch / 4).max(1);
        Self {
            svc,
            model: model.to_owned(),
            target,
            width: target,
            in_flight: 0,
            on_wire: Arc::new(AtomicUsize::new(0)),
            backoff_until: None,
            yield_until: None,
            alone: false,
            hold: false,
            timeouts: 0,
            wedged_until: None,
            health_at: Instant::now(),
            down: false,
        }
    }

    fn yielding(&self, now: Instant) -> bool {
        self.yield_until.is_some_and(|t| now < t)
    }

    fn backing_off(&self, now: Instant) -> bool {
        self.backoff_until.is_some_and(|t| now < t)
    }

    fn wedged(&self, now: Instant) -> bool {
        self.wedged_until.is_some_and(|t| now < t)
    }
}

struct Dispatcher {
    b: Arc<Batches>,
    http: reqwest::Client,
    runs: HashMap<String, Run>,
    /// Service order: oldest batch first.
    order: Vec<String>,
    ports: HashMap<u16, Port>,
    tasks: JoinSet<Outcome>,
    sent: HashMap<tokio::task::Id, Sent>,
    validating: HashSet<String>,
    validations: JoinSet<(String, Result<super::index::Outcome, String>)>,
    reload_at: Instant,
}

pub(crate) async fn run(b: Arc<Batches>) {
    let http = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .pool_idle_timeout(Duration::from_secs(30))
        .build()
        .unwrap_or_default();
    // finish purges a crash interrupted (see store::batch_items_purge)
    {
        let db = b.db.clone();
        tokio::task::spawn_blocking(move || {
            for id in db.batch_items_leftover().unwrap_or_default() {
                let _ = db.batch_items_purge(&id);
            }
        });
    }
    let mut d = Dispatcher {
        b,
        http,
        runs: HashMap::new(),
        order: Vec::new(),
        ports: HashMap::new(),
        tasks: JoinSet::new(),
        sent: HashMap::new(),
        validating: HashSet::new(),
        validations: JoinSet::new(),
        reload_at: Instant::now(),
    };
    loop {
        let now = Instant::now();
        if now >= d.reload_at {
            d.reload();
            d.reload_at = now + d.b.tuning.tick;
        }
        d.refresh().await;
        d.fill();
        d.flush(false);
        d.wind_down();
        d.publish();
        let wait = d.next_wait();
        tokio::select! {
            Some(r) = d.tasks.join_next_with_id(), if !d.tasks.is_empty() => {
                d.on_done(r);
                // everything else that finished meanwhile, before the next
                // refill: one loop pass per answer capped a fast runner at a
                // fraction of its width
                while let Some(r) = d.tasks.try_join_next_with_id() {
                    d.on_done(r);
                }
            }
            Some(v) = d.validations.join_next(), if !d.validations.is_empty() => d.on_validated(v),
            _ = d.b.wake.notified() => d.reload_at = Instant::now(),
            _ = tokio::time::sleep(wait) => {}
        }
    }
}

impl Dispatcher {
    /// Pick up new batches, cancellations and deletions from the store.
    fn reload(&mut self) {
        let rows = match self.b.db.batch_open() {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(%e, "batch: reading open batches failed");
                return;
            }
        };
        let mut live: HashSet<String> = HashSet::new();
        for row in rows {
            if row.is_terminal() {
                continue; // webhook-only rows are the courier's
            }
            live.insert(row.id.clone());
            if self.validating.contains(&row.id) {
                continue; // its index is still being written
            }
            if row.status == "validating" {
                if !self.validating.contains(&row.id) {
                    self.start_validation(row);
                }
                continue;
            }
            match self.runs.get_mut(&row.id) {
                Some(run) => {
                    if row.status == "cancelling" && run.stop.is_none() {
                        run.stop = Some(Stop::Cancel);
                    }
                    run.row.status = row.status;
                    run.row.expires_ms = row.expires_ms;
                }
                None => self.open_run(row),
            }
        }
        // deleted out from under us (or failed elsewhere): stop sending
        let gone: Vec<String> = self
            .runs
            .keys()
            .filter(|id| !live.contains(*id))
            .cloned()
            .collect();
        for id in gone {
            self.abort_run(&id);
            self.runs.remove(&id);
            self.forget_progress(&id);
        }
        let mut order: Vec<(i64, String)> = self
            .runs
            .values()
            .map(|r| (r.row.created_ms, r.row.id.clone()))
            .collect();
        order.sort();
        self.order = order.into_iter().map(|(_, id)| id).collect();
    }

    fn open_run(&mut self, row: BatchRow) {
        let id = row.id.clone();
        let dir = self.b.job_dir(&id);
        let opened = (|| -> Result<Run, String> {
            let input = self
                .b
                .input_path(&row)
                .ok_or_else(|| "the batch's input file is gone".to_string())?;
            let out = ResultFile::open(&dir.join("out.jsonl"), row.out_len as u64)
                .map_err(|e| e.to_string())?;
            let err = if row.dialect == "openai" {
                Some(
                    ResultFile::open(&dir.join("err.jsonl"), row.err_len as u64)
                        .map_err(|e| e.to_string())?,
                )
            } else {
                None
            };
            let opts = &row.options;
            Ok(Run {
                input,
                out,
                err,
                queue: VecDeque::new(),
                retry: Vec::new(),
                isolate: VecDeque::new(),
                cursor: -1,
                exhausted: false,
                in_flight: 0,
                commit: BatchCommit::default(),
                commit_since: None,
                unflushed: 0,
                port: None,
                resolve_at: Instant::now(),
                cap: opts["concurrency"]
                    .as_u64()
                    .filter(|&c| c > 0)
                    .map_or(usize::MAX, |c| c as usize),
                yield_to_live: opts["yield_to_live"].as_bool().unwrap_or(true),
                stop: (row.status == "cancelling").then_some(Stop::Cancel),
                aborted: false,
                state: "running",
                reason: String::new(),
                settled: VecDeque::new(),
                retries: 0,
                disk: None,
                disk_until: None,
                row,
            })
        })();
        match opened {
            Ok(run) => {
                self.runs.insert(id, run);
            }
            Err(e) => {
                tracing::error!(batch = %id, %e, "batch: cannot resume");
                let errors = json!({"object": "list", "data": [{
                    "code": "resume_failed", "message": format!("the batch could not be resumed: {e}"),
                    "param": null, "line": null,
                }]});
                let _ = self.b.db.batch_set_failed(&id, &errors);
            }
        }
    }

    fn start_validation(&mut self, row: BatchRow) {
        let Some(input) = self.b.input_path(&row) else {
            let errors = json!({"object": "list", "data": [{
                "code": "missing_file", "message": "the input file no longer exists",
                "param": "input_file_id", "line": null,
            }]});
            let _ = self.b.db.batch_set_failed(&row.id, &errors);
            return;
        };
        self.validating.insert(row.id.clone());
        let db = self.b.db.clone();
        self.validations.spawn_blocking(move || {
            // a validation interrupted by a crash starts over clean
            let res = db
                .batch_items_purge(&row.id)
                .map_err(|e| e.to_string())
                .and_then(|_| {
                    super::index::index_jsonl(&input, &row.endpoint, &row.model, |chunk| {
                        db.batch_items_insert(&row.id, chunk)
                            .map_err(|e| e.to_string())
                    })
                });
            (row.id, res)
        });
    }

    fn on_validated(
        &mut self,
        v: Result<(String, Result<super::index::Outcome, String>), tokio::task::JoinError>,
    ) {
        let Ok((id, res)) = v else { return };
        self.validating.remove(&id);
        match res {
            Ok(o) if o.errors.is_empty() => {
                let _ = self.b.db.batch_set_total(&id, o.total);
                // a cancel that landed during validation keeps its status
                let _ = self.b.db.batch_transition(&id, "validating", "in_progress");
                tracing::info!(batch = %id, requests = o.total, "batch: validated, running");
            }
            Ok(o) => {
                self.purge_later(&id);
                let errors = json!({"object": "list", "data": o.errors});
                let _ = self.b.db.batch_set_failed(&id, &errors);
                tracing::info!(batch = %id, "batch: input failed validation");
            }
            Err(e) => {
                self.purge_later(&id);
                let errors = json!({"object": "list", "data": [{
                    "code": "validation_failed", "message": e, "param": null, "line": null,
                }]});
                let _ = self.b.db.batch_set_failed(&id, &errors);
            }
        }
        self.reload_at = Instant::now();
    }

    /// Bind batches to services and sample every port in use.
    async fn refresh(&mut self) {
        let now = Instant::now();
        // bind the unbound
        let ids: Vec<String> = self.order.clone();
        for id in &ids {
            let Some(run) = self.runs.get(id) else {
                continue;
            };
            if run.port.is_some() || run.stop.is_some() || now < run.resolve_at {
                continue;
            }
            let (model, endpoint) = (run.row.model.clone(), run.row.endpoint.clone());
            let svc = self.b.fleet.resolve(&model, &endpoint).await;
            let Some(run) = self.runs.get_mut(id) else {
                continue;
            };
            match svc {
                Some(svc) => {
                    let port = svc.port;
                    match self.ports.get_mut(&port) {
                        Some(p) if p.svc.pid == svc.pid => p.down = false,
                        _ => {
                            self.ports.insert(port, Port::new(svc, &model));
                        }
                    }
                    run.port = Some(port);
                }
                None => {
                    run.state = "waiting_for_service";
                    run.reason = format!(
                        "no running service serves {model} - the batch resumes when one does"
                    );
                    run.resolve_at = now + self.b.tuning.rebind;
                }
            }
        }
        // sample the ports in use
        let ports: Vec<u16> = self.ports.keys().copied().collect();
        for port in ports {
            let p = &self.ports[&port];
            if now < p.health_at {
                continue;
            }
            let model = p.model.clone();
            let h = self.b.fleet.health(port, &model).await;
            let dwell = self.b.tuning.yield_dwell;
            let tick = self.b.tuning.tick;
            let Some(p) = self.ports.get_mut(&port) else {
                continue;
            };
            p.health_at = now + tick;
            let lost = match h {
                None => true,
                Some(h) => h.pid != p.svc.pid || !h.serves,
            };
            if lost {
                // gone, restarted or switched: everything bound here rebinds
                p.down = true;
                p.hold = false;
                for run in self.runs.values_mut() {
                    if run.port == Some(port) {
                        run.port = None;
                        run.resolve_at = now + self.b.tuning.rebind / 2;
                    }
                }
                if p.in_flight == 0 {
                    self.ports.remove(&port);
                }
                continue;
            }
            let h = h.expect("checked above");
            p.hold = false;
            if h.draining {
                p.backoff_until = Some(now + Duration::from_secs(5));
            }
            // One sample is enough: `on_wire` counts a request from before it
            // is sent until after its answer is read, so the runner can only
            // ever show FEWER of ours than that - any excess is somebody
            // else's. (Not quite always: a request we aborted runs on until the
            // runner notices, which costs one needless dwell, rarely.)
            let live = h
                .in_flight
                .saturating_sub(p.on_wire.load(Ordering::SeqCst) as u64);
            if live > 0 {
                p.yield_until = Some(now + dwell);
            }
        }
        // ports nobody uses any more
        let used: HashSet<u16> = self.runs.values().filter_map(|r| r.port).collect();
        self.ports
            .retain(|port, p| used.contains(port) || p.in_flight > 0);
    }

    /// Send what the ports have room for, oldest batch first.
    fn fill(&mut self) {
        let now = Instant::now();
        // a port with an item awaiting isolation takes nothing else new
        let mut isolating: HashSet<u16> = HashSet::new();
        for id in &self.order {
            if let Some(r) = self.runs.get(id)
                && let Some(p) = r.port
                && !r.isolate.is_empty()
                && r.stop.is_none()
            {
                isolating.insert(p);
            }
        }
        let order = self.order.clone();
        let mut ahead: HashMap<u16, String> = HashMap::new();
        for id in order {
            let Some(run) = self.runs.get_mut(&id) else {
                continue;
            };
            if run.row.status != "in_progress" || run.stop.is_some() {
                continue;
            }
            let Some(port_no) = run.port else { continue };
            let Some(port) = self.ports.get_mut(&port_no) else {
                run.port = None;
                continue;
            };
            if let Some(e) = &run.disk {
                run.state = "paused";
                run.reason = format!(
                    "cannot write results ({e}) - free some disk space; the batch resumes on its own"
                );
                let until = *run.disk_until.get_or_insert(now + self.b.tuning.disk_retry);
                if now < until {
                    continue;
                }
                if super::files::headroom(&self.b.job_dir(&id), 64 << 20).is_err() {
                    run.disk_until = Some(now + self.b.tuning.disk_retry);
                    continue;
                }
                // try again; a write that fails again pauses again
                run.disk = None;
                run.disk_until = None;
            }
            if port.down || port.hold {
                run.state = "waiting_for_service";
                run.reason = format!("waiting for the service on port {port_no} to answer");
                continue;
            }
            if port.wedged(now) {
                run.state = "waiting_for_service";
                run.reason = format!(
                    "the service on port {port_no} stopped answering ({} requests in a row timed \
                     out); one request will probe it again shortly",
                    port.timeouts
                );
                continue;
            }
            if port.backing_off(now) {
                run.state = "backing_off";
                run.reason = format!("the service on port {port_no} asked for a pause");
                continue;
            }
            if port.yielding(now) && run.yield_to_live {
                run.state = "yielding";
                run.reason = format!(
                    "live traffic on port {port_no}: batch work resumes after {} s without it",
                    self.b.tuning.yield_dwell.as_secs()
                );
                continue;
            }
            if port.alone {
                run.state = "isolating";
                run.reason = "one request that stopped the server is running alone".into();
                continue;
            }
            if isolating.contains(&port_no) {
                if let Some(item) = run.isolate.front().cloned() {
                    if port.in_flight == 0 {
                        run.isolate.pop_front();
                        port.alone = true;
                        run.state = "isolating";
                        run.reason = format!(
                            "request {} stopped the server {} times; running it alone",
                            item.custom_id, item.strikes
                        );
                        spawn(
                            &mut self.tasks,
                            &mut self.sent,
                            &self.http,
                            &self.b.tuning,
                            run,
                            port,
                            item,
                            true,
                        );
                    }
                } else {
                    run.state = "isolating";
                    run.reason = "waiting for an isolated request to run alone".into();
                }
                continue;
            }
            let mut sent_any = false;
            while port.in_flight < port.width && run.in_flight < run.cap {
                let Some(item) = next_item(&self.b, run, now) else {
                    break;
                };
                spawn(
                    &mut self.tasks,
                    &mut self.sent,
                    &self.http,
                    &self.b.tuning,
                    run,
                    port,
                    item,
                    false,
                );
                sent_any = true;
            }
            if sent_any || run.in_flight > 0 {
                run.state = "running";
                run.reason.clear();
            } else if !run.retry.is_empty() && run.queue.is_empty() {
                run.state = "backing_off";
                run.reason = format!(
                    "{} requests are waiting to be retried after server errors",
                    run.retry.len()
                );
            } else if port.in_flight >= port.width {
                run.state = "queued";
                run.reason = match ahead.get(&port_no) {
                    Some(a) => format!("waiting behind batch {a} on port {port_no}"),
                    None => format!("port {port_no} is at its width"),
                };
            }
            ahead.entry(port_no).or_insert(id);
        }
    }

    fn on_done(&mut self, r: Result<(tokio::task::Id, Outcome), tokio::task::JoinError>) {
        let (task, outcome) = match r {
            Ok((id, o)) => (id, Some(o)),
            Err(e) => (e.id(), None),
        };
        let Some(sent) = self.sent.remove(&task) else {
            return;
        };
        if let Some(p) = self.ports.get_mut(&sent.port)
            && p.svc.pid == sent.pid
        {
            p.in_flight = p.in_flight.saturating_sub(1);
            if sent.alone {
                p.alone = false;
            }
        }
        let tuning = self.b.tuning.clone();
        let Some(run) = self.runs.get_mut(&sent.batch) else {
            return;
        };
        run.in_flight = run.in_flight.saturating_sub(1);
        // aborted (cancel, expiry, deletion): the item simply stays pending
        let Some(outcome) = outcome else { return };
        let mut item = sent.item;
        let now = Instant::now();
        match outcome {
            Outcome::Http {
                status,
                request_id,
                body,
                ..
            } if (200..300).contains(&status) => {
                if let Some(p) = self.ports.get_mut(&sent.port) {
                    p.width = (p.width + 1).min(p.target);
                    p.timeouts = 0;
                }
                settle_success(run, &item, status, request_id, &body);
            }
            Outcome::Http {
                status: 429 | 503 | 529,
                retry_after,
                ..
            } => {
                run.queue.push_front(item);
                if let Some(p) = self.ports.get_mut(&sent.port) {
                    p.width = (p.width / 2).max(1);
                    let wait = retry_after
                        .unwrap_or(Duration::from_secs(2))
                        .min(Duration::from_secs(60));
                    p.backoff_until = Some(now + wait);
                }
            }
            Outcome::Http {
                status: 401 | 403, ..
            } => {
                // the runner turned the manager's key away: a respawn with a
                // new key - rebind, which re-reads it
                run.queue.push_front(item);
                run.port = None;
                run.resolve_at = now + tuning.rebind;
                if let Some(p) = self.ports.get_mut(&sent.port) {
                    p.down = true;
                }
            }
            Outcome::Http {
                status,
                request_id,
                body,
                ..
            } if status == 408 || status >= 500 => {
                item.attempts += 1;
                if item.attempts >= tuning.max_attempts {
                    settle_error(
                        run,
                        &item,
                        Some((status, request_id, &body)),
                        "server_error",
                        &format!(
                            "the server failed this request {} times (HTTP {status})",
                            item.attempts
                        ),
                    );
                } else {
                    retry_later(run, item, &tuning, now);
                }
            }
            Outcome::Http {
                status,
                request_id,
                body,
                ..
            } => {
                settle_error(
                    run,
                    &item,
                    Some((status, request_id, &body)),
                    "invalid_request",
                    "",
                );
            }
            Outcome::Refused(e) => {
                tracing::debug!(batch = %sent.batch, %e, "batch: service refused the connection");
                run.queue.push_front(item);
                run.port = None;
                run.resolve_at = now + tuning.rebind / 2;
                if let Some(p) = self.ports.get_mut(&sent.port) {
                    p.down = true;
                }
            }
            Outcome::Broken(e) => {
                item.strikes += 1;
                tracing::warn!(batch = %sent.batch, custom_id = %item.custom_id,
                    strikes = item.strikes, %e, "batch: the connection broke mid-request");
                if let Some(p) = self.ports.get_mut(&sent.port) {
                    p.hold = true;
                    p.health_at = now;
                }
                record_attempt(run, &item);
                if sent.alone && item.strikes > tuning.isolate_after {
                    settle_error(
                        run,
                        &item,
                        None,
                        "server_error",
                        &format!(
                            "the server stopped while processing this request ({} times, the \
                             last with it running alone)",
                            item.strikes
                        ),
                    );
                } else if item.strikes >= tuning.isolate_after {
                    run.isolate.push_back(item);
                } else {
                    run.retries += 1;
                    run.queue.push_front(item);
                }
            }
            Outcome::Timeout => {
                // Several in a row with nothing answered between them is the
                // service, not the request: a runner that still answers health
                // but no longer finishes work (a wedged GPU, a stuck kernel).
                // Then the port pauses and probes with one request, and these
                // requests take a strike - the crash rule - instead of using up
                // their attempts on a fault that is not theirs.
                let systemic = match self.ports.get_mut(&sent.port) {
                    Some(p) => {
                        p.timeouts += 1;
                        let was = p.wedged(now);
                        if !was && p.timeouts >= tuning.wedge_after {
                            p.wedged_until = Some(now + tuning.wedge_pause);
                            p.width = 1;
                            tracing::warn!(
                                port = sent.port,
                                timeouts = p.timeouts,
                                "batch: the service stopped answering; pausing it"
                            );
                        }
                        was || p.timeouts >= tuning.wedge_after
                    }
                    None => false,
                };
                if systemic {
                    item.strikes += 1;
                    record_attempt(run, &item);
                    if sent.alone && item.strikes > tuning.isolate_after {
                        settle_error(
                            run,
                            &item,
                            None,
                            "timeout",
                            &format!(
                                "the server stopped answering while processing this request \
                                 ({} times, the last with it running alone)",
                                item.strikes
                            ),
                        );
                    } else if item.strikes >= tuning.isolate_after {
                        run.isolate.push_back(item);
                    } else {
                        run.retries += 1;
                        run.queue.push_front(item);
                    }
                    return;
                }
                item.attempts += 1;
                if item.attempts >= tuning.max_attempts {
                    settle_error(
                        run,
                        &item,
                        None,
                        "timeout",
                        &format!(
                            "no answer within {} s, {} times",
                            tuning.request_timeout.as_secs(),
                            item.attempts
                        ),
                    );
                } else {
                    retry_later(run, item, &tuning, now);
                }
            }
            Outcome::Input(e) => {
                settle_error(
                    run,
                    &item,
                    None,
                    "input_unreadable",
                    &format!("could not read this request from the batch input: {e}"),
                );
            }
        }
    }

    /// Group-commit every run that is due (or all, with `force`).
    fn flush(&mut self, force: bool) {
        let every = self.b.tuning.commit_every;
        let items = self.b.tuning.commit_items;
        for run in self.runs.values_mut() {
            flush_run(&self.b, run, force, every, items);
        }
    }

    /// Cancellation, expiry and completion.
    fn wind_down(&mut self) {
        let now_wall = now_ms();
        let ids: Vec<String> = self.order.clone();
        for id in ids {
            let Some(run) = self.runs.get_mut(&id) else {
                continue;
            };
            if run.stop.is_none()
                && run.row.status == "in_progress"
                && now_wall >= run.row.expires_ms
            {
                run.stop = Some(Stop::Expire);
                tracing::info!(batch = %id, "batch: completion window reached");
            }
            if run.stop.is_some() && !run.aborted {
                run.aborted = true;
                run.state = "cancelling";
                self.abort_run(&id);
            }
            let Some(run) = self.runs.get_mut(&id) else {
                continue;
            };
            if run.in_flight > 0 {
                continue;
            }
            let done = match run.stop {
                Some(_) => true,
                None => {
                    run.row.status == "finalizing"
                        || (run.exhausted
                            && run.queue.is_empty()
                            && run.retry.is_empty()
                            && run.isolate.is_empty())
                }
            };
            if !done {
                continue;
            }
            let every = self.b.tuning.commit_every;
            let items = self.b.tuning.commit_items;
            flush_run(&self.b, run, true, every, items);
            if !run.commit.items.is_empty() {
                continue; // the commit failed; try again next pass
            }
            let status = match run.stop {
                Some(Stop::Cancel) => "cancelled",
                Some(Stop::Expire) => "expired",
                None => "completed",
            };
            if let Some(stop) = run.stop
                && let Err(e) = close_remaining(&self.b, run, stop)
            {
                tracing::warn!(batch = %id, %e, "batch: closing unfinished requests failed");
                continue;
            }
            match finalize(&self.b, run, status) {
                Ok(()) => {
                    tracing::info!(batch = %id, status, succeeded = run.row.succeeded,
                        errored = run.row.errored, "batch: finished");
                    // the item rows did their job; the files hold the results
                    self.purge_later(&id);
                    self.runs.remove(&id);
                    self.forget_progress(&id);
                }
                Err(e) => tracing::warn!(batch = %id, %e, "batch: finalizing failed"),
            }
        }
        self.order.retain(|id| self.runs.contains_key(id));
    }

    /// Delete a batch's item rows off the loop (a million of them take a
    /// while, and the run has other batches to feed meanwhile).
    fn purge_later(&self, id: &str) {
        let db = self.b.db.clone();
        let id = id.to_owned();
        tokio::task::spawn_blocking(move || db.batch_items_purge(&id));
    }

    fn abort_run(&self, id: &str) {
        for s in self.sent.values() {
            if s.batch == id {
                s.abort.abort();
            }
        }
    }

    fn forget_progress(&self, id: &str) {
        self.b
            .progress
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(id);
    }

    /// Publish each run's progress for the batch objects' `paddock` block.
    fn publish(&self) {
        let now = Instant::now();
        let mut map = self
            .b
            .progress
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for id in &self.validating {
            map.insert(
                id.clone(),
                Progress {
                    state: "validating",
                    ..Default::default()
                },
            );
        }
        for (id, run) in &self.runs {
            let port = run.port.and_then(|p| self.ports.get(&p));
            let rate = rate_per_minute(&run.settled, now);
            let r = &run.row;
            let left =
                (r.total - r.succeeded - r.errored - r.canceled - r.expired - run.unflushed).max(0);
            map.insert(
                id.clone(),
                Progress {
                    state: if run.stop.is_some() {
                        "cancelling"
                    } else {
                        run.state
                    },
                    reason: run.reason.clone(),
                    port: run.port,
                    in_flight: run.in_flight,
                    concurrency: port.map_or(0, |p| p.width),
                    rate_per_minute: rate.map(|r| (r * 10.0).round() / 10.0),
                    eta_seconds: rate
                        .filter(|r| *r > 0.0)
                        .map(|r| (left as f64 / r * 60.0).round() as u64),
                    retries: run.retries,
                },
            );
        }
    }

    /// Sleep until the earliest thing that needs doing.
    fn next_wait(&self) -> Duration {
        let now = Instant::now();
        let mut wait = self.b.tuning.tick;
        for run in self.runs.values() {
            if let Some(t) = run.commit_since {
                wait = wait.min((t + self.b.tuning.commit_every).saturating_duration_since(now));
            }
            for (t, _) in &run.retry {
                wait = wait.min(t.saturating_duration_since(now));
            }
        }
        wait.max(Duration::from_millis(5))
    }
}

/// The next item a run should send: a retry whose time has come, a requeued
/// one, or the next pending one from the store.
fn next_item(b: &Batches, run: &mut Run, now: Instant) -> Option<Item> {
    if let Some(i) = run.retry.iter().position(|(t, _)| *t <= now) {
        return Some(run.retry.swap_remove(i).1);
    }
    if run.queue.is_empty() && !run.exhausted {
        match b.db.batch_items_pending(&run.row.id, run.cursor, PREFETCH) {
            Ok(rows) => {
                if rows.len() < PREFETCH {
                    run.exhausted = true;
                }
                if let Some(last) = rows.last() {
                    run.cursor = last.idx;
                }
                run.queue.extend(rows.into_iter().map(Item::from));
            }
            Err(e) => {
                tracing::warn!(batch = %run.row.id, %e, "batch: reading pending items failed")
            }
        }
    }
    run.queue.pop_front()
}

#[allow(clippy::too_many_arguments)]
fn spawn(
    tasks: &mut JoinSet<Outcome>,
    sent: &mut HashMap<tokio::task::Id, Sent>,
    http: &reqwest::Client,
    tuning: &super::Tuning,
    run: &mut Run,
    port: &mut Port,
    item: Item,
    alone: bool,
) {
    port.in_flight += 1;
    run.in_flight += 1;
    let fut = request(
        http.clone(),
        port.on_wire.clone(),
        port.svc.clone(),
        run.row.endpoint.clone(),
        run.input.clone(),
        item.off as u64,
        item.len as usize,
        tuning.request_timeout,
    );
    let abort = tasks.spawn(fut);
    sent.insert(
        abort.id(),
        Sent {
            batch: run.row.id.clone(),
            item,
            port: port.svc.port,
            pid: port.svc.pid,
            alone,
            abort,
        },
    );
}

/// One request, start to finish. Reads the item's body off disk and sends it
/// byte for byte.
async fn request(
    http: reqwest::Client,
    on_wire: Arc<AtomicUsize>,
    svc: Service,
    endpoint: String,
    input: PathBuf,
    off: u64,
    len: usize,
    timeout: Duration,
) -> Outcome {
    let body =
        match tokio::task::spawn_blocking(move || super::files::read_at(&input, off, len)).await {
            Ok(Ok(b)) => b,
            Ok(Err(e)) => return Outcome::Input(e.to_string()),
            Err(e) => return Outcome::Input(e.to_string()),
        };
    let mut req = http
        .post(format!("{}{endpoint}", svc.base))
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .header("x-paddock-origin", "batch")
        .timeout(timeout)
        .body(body);
    if endpoint == "/v1/messages" {
        req = req.header("anthropic-version", "2023-06-01");
    }
    if let Some(k) = &svc.key {
        req = req.bearer_auth(k);
    }
    let _wire = OnWire::new(on_wire);
    let res = match req.send().await {
        Ok(r) => r,
        Err(e) if e.is_timeout() => return Outcome::Timeout,
        Err(e) if e.is_connect() => return Outcome::Refused(e.to_string()),
        Err(e) => return Outcome::Broken(e.to_string()),
    };
    let status = res.status().as_u16();
    let header = |n: &str| {
        res.headers()
            .get(n)
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned)
    };
    let request_id = header("x-request-id");
    let retry_after = header("retry-after")
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map(Duration::from_secs);
    match res.bytes().await {
        Ok(b) => Outcome::Http {
            status,
            request_id,
            body: b.to_vec(),
            retry_after,
        },
        Err(e) if e.is_timeout() => Outcome::Timeout,
        Err(e) => Outcome::Broken(e.to_string()),
    }
}

/// Holds one count in a port's `on_wire` for the life of a request (aborts
/// included: the guard drops with the task).
struct OnWire(Arc<AtomicUsize>);

impl OnWire {
    fn new(c: Arc<AtomicUsize>) -> Self {
        c.fetch_add(1, Ordering::SeqCst);
        Self(c)
    }
}

impl Drop for OnWire {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

fn retry_later(run: &mut Run, item: Item, tuning: &super::Tuning, now: Instant) {
    run.retries += 1;
    record_attempt(run, &item);
    let doubling = 2u32.saturating_pow((item.attempts - 1).clamp(0, 16) as u32);
    let backoff = tuning
        .retry_base
        .checked_mul(doubling)
        .unwrap_or(Duration::MAX)
        .min(Duration::from_secs(300));
    run.retry.push((now + backoff, item));
}

/// Persist an item's attempt and strike counts (with the next group commit),
/// so a restart does not hand a poisoned request a fresh set of lives.
fn record_attempt(run: &mut Run, item: &Item) {
    push_update(run, item, ITEM_PENDING);
}

fn push_update(run: &mut Run, item: &Item, state: i64) {
    run.commit.items.push(BatchItemUpdate {
        idx: item.idx,
        state,
        attempts: item.attempts,
        strikes: item.strikes,
    });
    run.commit_since.get_or_insert_with(Instant::now);
}

/// Append a line to a result file; a failed write pauses the run.
fn append(run: &mut Run, to_err: bool, line: &Value) -> bool {
    let mut bytes = match serde_json::to_vec(line) {
        Ok(b) => b,
        Err(e) => {
            run.disk = Some(e.to_string());
            return false;
        }
    };
    bytes.push(b'\n');
    let f = match (to_err, run.err.as_mut()) {
        (true, Some(f)) => f,
        _ => &mut run.out,
    };
    match f.append(&bytes) {
        Ok(()) => true,
        Err(e) => {
            run.disk = Some(e.to_string());
            false
        }
    }
}

/// A request that came back 2xx.
fn settle_success(
    run: &mut Run,
    item: &Item,
    status: u16,
    request_id: Option<String>,
    body: &[u8],
) {
    let Ok(v) = serde_json::from_slice::<Value>(body) else {
        settle_error(
            run,
            item,
            None,
            "server_error",
            "the server answered with a body that is not JSON",
        );
        return;
    };
    let line = if run.row.dialect == "anthropic" {
        json!({"custom_id": item.custom_id, "result": {"type": "succeeded", "message": v}})
    } else {
        json!({
            "id": new_id("batch_req_"),
            "custom_id": item.custom_id,
            "response": {"status_code": status, "request_id": request_id, "body": v},
            "error": null,
        })
    };
    if !append(run, false, &line) {
        run.queue.push_front(item.clone());
        return;
    }
    let u = &v["usage"];
    let n = |v: &Value| v.as_i64().unwrap_or(0);
    let input = n(&u["prompt_tokens"]) + n(&u["input_tokens"]);
    run.commit.input_tokens += input;
    run.commit.output_tokens += n(&u["completion_tokens"]) + n(&u["output_tokens"]);
    run.commit.cached_tokens += n(&u["prompt_tokens_details"]["cached_tokens"])
        + n(&u["input_tokens_details"]["cached_tokens"])
        + n(&u["cache_read_input_tokens"]);
    run.commit.reasoning_tokens += n(&u["completion_tokens_details"]["reasoning_tokens"])
        + n(&u["output_tokens_details"]["reasoning_tokens"]);
    run.commit.succeeded += 1;
    push_update(run, item, ITEM_SUCCEEDED);
    settled(run);
}

/// A request that is done failing. `http` is the runner's answer when there
/// was one, recorded verbatim; otherwise `code` / `message` describe it.
fn settle_error(
    run: &mut Run,
    item: &Item,
    http: Option<(u16, Option<String>, &[u8])>,
    code: &str,
    message: &str,
) {
    let status = http.as_ref().map(|h| h.0);
    let body = http.as_ref().map(|(_, _, b)| {
        serde_json::from_slice::<Value>(b)
            .unwrap_or_else(|_| json!({"error": {"message": String::from_utf8_lossy(b)}}))
    });
    let line = if run.row.dialect == "anthropic" {
        let error = match &body {
            Some(v) if v["type"] == "error" => v.clone(),
            Some(v) => json!({"type": "error", "error": {
                "type": if status.is_some_and(|s| s < 500) { "invalid_request_error" } else { "api_error" },
                "message": v["error"]["message"].as_str().unwrap_or(message),
            }}),
            None => json!({"type": "error", "error": {"type": "api_error", "message": message}}),
        };
        json!({"custom_id": item.custom_id, "result": {"type": "errored", "error": error}})
    } else {
        match http {
            Some((status, rid, _)) => json!({
                "id": new_id("batch_req_"),
                "custom_id": item.custom_id,
                "response": {"status_code": status, "request_id": rid, "body": body},
                "error": null,
            }),
            None => json!({
                "id": new_id("batch_req_"),
                "custom_id": item.custom_id,
                "response": null,
                "error": {"code": code, "message": message},
            }),
        }
    };
    if !append(run, true, &line) {
        run.queue.push_front(item.clone());
        return;
    }
    run.commit.errored += 1;
    push_update(run, item, ITEM_ERRORED);
    settled(run);
}

fn settled(run: &mut Run) {
    run.unflushed += 1;
    run.settled.push_back(Instant::now());
    while run.settled.len() > RATE_WINDOW {
        run.settled.pop_front();
    }
}

/// Requests per minute over the settles of the last five minutes (at least 30 s
/// of them, so the first two answers do not promise a rate). Time-bounded on
/// purpose: a pause - a yield, a runner restart - drops out of the ETA within
/// minutes instead of dragging it for the length of a fixed count window.
pub(super) fn rate_per_minute(settled: &VecDeque<Instant>, now: Instant) -> Option<f64> {
    let horizon = Duration::from_secs(300);
    let recent = settled
        .iter()
        .filter(|t| now.saturating_duration_since(**t) <= horizon);
    let first = *recent.clone().next()?;
    let n = recent.count();
    let span = now.saturating_duration_since(first).as_secs_f64();
    if span < 30.0 || n < 2 {
        return None;
    }
    Some(n as f64 / span * 60.0)
}

fn flush_run(b: &Batches, run: &mut Run, force: bool, every: Duration, items: usize) {
    if run.commit.items.is_empty() {
        return;
    }
    let due = force
        || run.commit.items.len() >= items
        || run.commit_since.is_some_and(|t| t.elapsed() >= every);
    if !due {
        return;
    }
    // bytes durable first, then the rows that point at them
    let synced = run
        .out
        .sync()
        .and_then(|_| run.err.as_mut().map_or(Ok(()), |f| f.sync()));
    if let Err(e) = synced {
        run.disk = Some(e.to_string());
        return;
    }
    run.commit.out_len = run.out.len as i64;
    run.commit.err_len = run.err.as_ref().map_or(0, |f| f.len as i64);
    match b.db.batch_commit(&run.row.id, &run.commit) {
        Ok(()) => {
            let c = std::mem::take(&mut run.commit);
            let r = &mut run.row;
            r.out_len = c.out_len;
            r.err_len = c.err_len;
            r.succeeded += c.succeeded;
            r.errored += c.errored;
            r.canceled += c.canceled;
            r.expired += c.expired;
            r.input_tokens += c.input_tokens;
            r.output_tokens += c.output_tokens;
            r.cached_tokens += c.cached_tokens;
            r.reasoning_tokens += c.reasoning_tokens;
            run.commit_since = None;
            run.unflushed = 0;
        }
        Err(e) => tracing::warn!(batch = %run.row.id, %e, "batch: commit failed; retrying"),
    }
}

/// Record every still-pending item as cancelled or expired, in the dialect's
/// own words, through the same append-then-commit path as a result.
fn close_remaining(b: &Batches, run: &mut Run, stop: Stop) -> Result<(), String> {
    let (state, code, message, kind) = match stop {
        Stop::Cancel => (
            ITEM_CANCELED,
            "batch_cancelled",
            "This request was not run: the batch was cancelled.",
            "canceled",
        ),
        Stop::Expire => (
            ITEM_EXPIRED,
            "batch_expired",
            "This request could not be executed before the completion window expired.",
            "expired",
        ),
    };
    // a cursor, not "the first pending" again: the rows closed so far stay in
    // the pending index until their chunk commits
    let mut after = -1;
    loop {
        let rows =
            b.db.batch_items_pending(&run.row.id, after, 2_000)
                .map_err(|e| e.to_string())?;
        let Some(last) = rows.last() else {
            return Ok(());
        };
        after = last.idx;
        for r in rows {
            let item = Item::from(r);
            let line = if run.row.dialect == "anthropic" {
                json!({"custom_id": item.custom_id, "result": {"type": kind}})
            } else {
                json!({
                    "id": new_id("batch_req_"),
                    "custom_id": item.custom_id,
                    "response": null,
                    "error": {"code": code, "message": message},
                })
            };
            if !append(run, true, &line) {
                return Err(run.disk.clone().unwrap_or_default());
            }
            match stop {
                Stop::Cancel => run.commit.canceled += 1,
                Stop::Expire => run.commit.expired += 1,
            }
            push_update(run, &item, state);
        }
        flush_run(b, run, true, Duration::ZERO, 0);
        if !run.commit.items.is_empty() {
            return Err("commit failed".into());
        }
    }
}

/// Register the result files and close the batch. Idempotent: the file ids
/// derive from the batch id, so a finalize interrupted by a crash and redone
/// lands on the same rows.
fn finalize(b: &Batches, run: &mut Run, status: &str) -> Result<(), String> {
    if status == "completed" {
        let _ = b.db.batch_set_status(&run.row.id, "finalizing");
    }
    let id = &run.row.id;
    let hex = id.rsplit('_').next().unwrap_or(id);
    let rel = |name: &str| format!("jobs/{id}/{name}");
    let now = now_ms();
    let mut files = Vec::new();
    let mut out_id = None;
    let mut err_id = None;
    if run.row.dialect == "openai" {
        if run.out.len > 0 {
            let fid = format!("file-{hex}o");
            files.push(crate::store::BatchFileRow {
                id: fid.clone(),
                purpose: "batch_output".into(),
                filename: format!("{id}_output.jsonl"),
                bytes: run.out.len as i64,
                created_ms: now,
                path: rel("out.jsonl"),
                batch_id: Some(id.clone()),
            });
            out_id = Some(fid);
        }
        if let Some(e) = run.err.as_ref().filter(|f| f.len > 0) {
            let fid = format!("file-{hex}e");
            files.push(crate::store::BatchFileRow {
                id: fid.clone(),
                purpose: "batch_output".into(),
                filename: format!("{id}_error.jsonl"),
                bytes: e.len as i64,
                created_ms: now,
                path: rel("err.jsonl"),
                batch_id: Some(id.clone()),
            });
            err_id = Some(fid);
        }
    }
    let webhook = run.row.options["webhook_url"].is_string();
    b.db.batch_finish(
        id,
        status,
        &files,
        out_id.as_deref(),
        err_id.as_deref(),
        webhook,
    )
    .map_err(|e| e.to_string())
}
