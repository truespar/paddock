//! The batch surface against a fake runner: both dialects end to end, and the
//! failure modes a days-long run meets - server errors, a request that kills
//! its connection, a service that goes away, live traffic, a manager crash
//! with torn result bytes on disk, a cancel mid-run.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::time::Duration;

use axum::Json;
use axum::body::Body;
use axum::extract::State;
use axum::http::{Request, StatusCode};
use axum::response::{IntoResponse, Response};
use futures::future::BoxFuture;
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tower::ServiceExt;

use super::*;
use crate::store::BatchRow;

const MODEL: &str = "test-model";

#[derive(Default)]
struct Fake {
    in_flight: AtomicU64,
    peak: AtomicU64,
    /// Live traffic the health sample should report on top of ours.
    live: AtomicU64,
    delay_ms: AtomicU64,
    /// Every request hangs while this is set: a runner that still answers its
    /// health check but has stopped finishing work.
    wedged: AtomicBool,
    calls: std::sync::Mutex<HashMap<String, u32>>,
    hooks: std::sync::Mutex<Vec<Value>>,
}

struct InFlight<'a>(&'a Fake);

impl Drop for InFlight<'_> {
    fn drop(&mut self) {
        self.0.in_flight.fetch_sub(1, Ordering::SeqCst);
    }
}

/// What the fake does with a request, by its prompt text.
async fn answer(f: &Fake, content: &str) -> Result<String, Box<Response>> {
    let _g = {
        let n = f.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        f.peak.fetch_max(n, Ordering::SeqCst);
        InFlight(f)
    };
    let calls = {
        let mut c = f.calls.lock().unwrap();
        let n = c.entry(content.to_owned()).or_default();
        *n += 1;
        *n
    };
    while f.wedged.load(Ordering::SeqCst) {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    if content == "hang" {
        tokio::time::sleep(Duration::from_secs(30)).await;
    }
    let delay = f.delay_ms.load(Ordering::SeqCst);
    if delay > 0 {
        tokio::time::sleep(Duration::from_millis(delay)).await;
    }
    match content {
        "bad" => Err(Box::new((
            StatusCode::BAD_REQUEST,
            Json(json!({"error": {"message": "the picture cannot be read", "type": "invalid_request_error"}})),
        )
            .into_response())),
        "flaky" if calls <= 2 => Err(Box::new((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": {"message": "transient", "type": "server_error"}})),
        )
            .into_response())),
        // the runner dies with the request on it: the connection drops
        // without an answer
        "crash" => panic!("simulated runner crash"),
        _ => Ok(format!("echo:{content}")),
    }
}

async fn fake_chat(State(f): State<Arc<Fake>>, Json(body): Json<Value>) -> Response {
    let content = body["messages"][0]["content"]
        .as_str()
        .unwrap_or("")
        .to_owned();
    match answer(&f, &content).await {
        Ok(text) => Json(json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": body["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }))
        .into_response(),
        Err(r) => *r,
    }
}

async fn fake_messages(State(f): State<Arc<Fake>>, Json(body): Json<Value>) -> Response {
    let content = body["messages"][0]["content"]
        .as_str()
        .unwrap_or("")
        .to_owned();
    match answer(&f, &content).await {
        Ok(text) => Json(json!({
            "id": "msg_1", "type": "message", "role": "assistant", "model": body["model"],
            "content": [{"type": "text", "text": text}], "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }))
        .into_response(),
        Err(r) => *r,
    }
}

async fn fake_hook(State(f): State<Arc<Fake>>, Json(body): Json<Value>) -> StatusCode {
    f.hooks.lock().unwrap().push(body);
    StatusCode::OK
}

async fn start_fake() -> (u16, Arc<Fake>) {
    let fake = Arc::new(Fake::default());
    let app = axum::Router::new()
        .route("/v1/chat/completions", axum::routing::post(fake_chat))
        .route("/v1/messages", axum::routing::post(fake_messages))
        .route("/hook", axum::routing::post(fake_hook))
        .with_state(fake.clone());
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = l.local_addr().unwrap().port();
    tokio::spawn(async move { axum::serve(l, app).await.unwrap() });
    (port, fake)
}

struct TestFleet {
    port: u16,
    pid: AtomicU32,
    up: AtomicBool,
    max_batch: usize,
    fake: Arc<Fake>,
}

impl Fleet for TestFleet {
    fn resolve<'a>(&'a self, model: &'a str, _: &'a str) -> BoxFuture<'a, Option<Service>> {
        Box::pin(async move {
            (self.up.load(Ordering::SeqCst) && model == MODEL).then(|| Service {
                port: self.port,
                pid: self.pid.load(Ordering::SeqCst),
                base: format!("http://127.0.0.1:{}", self.port),
                key: None,
                max_batch: self.max_batch,
            })
        })
    }
    fn served<'a>(&'a self, _: &'a str) -> BoxFuture<'a, Vec<String>> {
        Box::pin(async { vec![MODEL.to_owned()] })
    }
    fn health<'a>(&'a self, _: u16, _: &'a str) -> BoxFuture<'a, Option<Health>> {
        Box::pin(async move {
            self.up.load(Ordering::SeqCst).then(|| Health {
                in_flight: self.fake.in_flight.load(Ordering::SeqCst)
                    + self.fake.live.load(Ordering::SeqCst),
                draining: false,
                pid: self.pid.load(Ordering::SeqCst),
                serves: true,
            })
        })
    }
}

fn fast() -> Tuning {
    Tuning {
        request_timeout: Duration::from_secs(10),
        commit_every: Duration::from_millis(20),
        commit_items: 8,
        yield_dwell: Duration::from_millis(400),
        tick: Duration::from_millis(40),
        max_attempts: 3,
        retry_base: Duration::from_millis(20),
        isolate_after: 2,
        rebind: Duration::from_millis(60),
        disk_retry: Duration::from_millis(100),
        wedge_after: 3,
        wedge_pause: Duration::from_millis(300),
    }
}

struct Rig {
    app: axum::Router,
    b: Arc<Batches>,
    fleet: Arc<TestFleet>,
    fake: Arc<Fake>,
    root: std::path::PathBuf,
    db: Arc<crate::store::Store>,
}

async fn rig(max_batch: usize) -> Rig {
    rig_tuned(max_batch, fast()).await
}

async fn rig_tuned(max_batch: usize, tuning: Tuning) -> Rig {
    let (port, fake) = start_fake().await;
    let fleet = Arc::new(TestFleet {
        port,
        pid: AtomicU32::new(4242),
        up: AtomicBool::new(true),
        max_batch,
        fake: fake.clone(),
    });
    let root = std::env::temp_dir().join(format!("paddock-batch-test-{}", uuid::Uuid::new_v4()));
    let mut state = crate::routes::AppState::for_tests();
    let db = state.db.clone();
    let b = Arc::new(Batches::new(
        root.clone(),
        db.clone(),
        fleet.clone(),
        tuning,
    ));
    state.batches = b.clone();
    let app = crate::routes::router(Arc::new(state));
    Rig {
        app,
        b,
        fleet,
        fake,
        root,
        db,
    }
}

async fn call(app: &axum::Router, req: Request<Body>) -> (StatusCode, Value) {
    let res = app.clone().oneshot(req).await.unwrap();
    let status = res.status();
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let v = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()));
    (status, v)
}

async fn raw(app: &axum::Router, req: Request<Body>) -> (StatusCode, Vec<u8>) {
    let res = app.clone().oneshot(req).await.unwrap();
    let status = res.status();
    (
        status,
        res.into_body().collect().await.unwrap().to_bytes().to_vec(),
    )
}

fn post_json(uri: &str, v: Value) -> Request<Body> {
    Request::post(uri)
        .header("content-type", "application/json")
        .body(Body::from(v.to_string()))
        .unwrap()
}

fn get(uri: &str) -> Request<Body> {
    Request::get(uri).body(Body::empty()).unwrap()
}

fn chat_line(cid: &str, content: &str) -> String {
    json!({"custom_id": cid, "method": "POST", "url": "/v1/chat/completions",
           "body": {"model": MODEL, "messages": [{"role": "user", "content": content}]}})
    .to_string()
}

async fn upload(app: &axum::Router, jsonl: &str) -> String {
    let boundary = "XpaddockX";
    let body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n\
         --{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"in.jsonl\"\r\n\
         Content-Type: application/jsonl\r\n\r\n{jsonl}\r\n--{boundary}--\r\n"
    );
    let req = Request::post("/v1/files")
        .header(
            "content-type",
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(Body::from(body))
        .unwrap();
    let (s, v) = call(app, req).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["object"], "file");
    v["id"].as_str().unwrap().to_owned()
}

async fn create_openai(app: &axum::Router, file: &str, extra: Value) -> String {
    let mut body = json!({"input_file_id": file, "endpoint": "/v1/chat/completions",
                          "completion_window": "24h"});
    if let Some(o) = extra.as_object() {
        for (k, v) in o {
            body[k] = v.clone();
        }
    }
    let (s, v) = call(app, post_json("/v1/batches", body)).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    v["id"].as_str().unwrap().to_owned()
}

async fn wait_for(
    db: &crate::store::Store,
    id: &str,
    pred: impl Fn(&BatchRow) -> bool,
) -> BatchRow {
    for _ in 0..600 {
        if let Some(r) = db.batch_get(id).unwrap()
            && pred(&r)
        {
            return r;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!(
        "batch {id} never got there: {:?}",
        db.batch_get(id).unwrap()
    );
}

async fn wait_terminal(db: &crate::store::Store, id: &str) -> BatchRow {
    wait_for(db, id, BatchRow::is_terminal).await
}

/// Poll a batch until its `paddock.state` reads `state`.
async fn wait_state(app: &axum::Router, id: &str, state: &str) -> Value {
    for _ in 0..200 {
        let (_, v) = call(app, get(&format!("/v1/batches/{id}"))).await;
        if v["paddock"]["state"] == state {
            return v;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let (_, v) = call(app, get(&format!("/v1/batches/{id}"))).await;
    panic!("batch {id} never reached {state}: {v}");
}

fn lines(bytes: &[u8]) -> Vec<Value> {
    bytes
        .split(|b| *b == b'\n')
        .filter(|l| !l.is_empty())
        .map(|l| serde_json::from_slice(l).expect("every result line is JSON"))
        .collect()
}

async fn file_lines(app: &axum::Router, id: Option<&str>) -> Vec<Value> {
    let Some(id) = id else { return Vec::new() };
    let (s, bytes) = raw(app, get(&format!("/v1/files/{id}/content"))).await;
    assert_eq!(s, StatusCode::OK);
    lines(&bytes)
}

/// Every input custom_id appears exactly once across the result files.
fn exactly_once(results: &[Value], expect: &HashSet<String>) {
    let mut seen = HashSet::new();
    for l in results {
        let cid = l["custom_id"].as_str().unwrap().to_owned();
        assert!(seen.insert(cid.clone()), "custom_id {cid} answered twice");
    }
    assert_eq!(&seen, expect);
}

#[test]
fn index_records_body_spans_and_refuses_bad_lines() {
    let dir = std::env::temp_dir().join(format!("paddock-index-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let good = dir.join("good.jsonl");
    let text = format!(
        "{}\r\n\n{}\n{}",
        chat_line("a", "one"),
        chat_line("b", "two \"quoted\""),
        chat_line("c", "three")
    );
    std::fs::write(&good, &text).unwrap();
    let mut rows = Vec::new();
    let o = index::index_jsonl(&good, "/v1/chat/completions", MODEL, |c| {
        rows.extend_from_slice(c);
        Ok(())
    })
    .unwrap();
    assert!(o.errors.is_empty(), "{:?}", o.errors);
    assert_eq!(o.total, 3);
    for (i, (_, cid, off, len)) in rows.iter().enumerate() {
        let body = files::read_at(&good, *off as u64, *len as usize).unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["model"], MODEL, "row {i} ({cid}) points at its body");
    }
    assert_eq!(rows[1].1, "b");

    let bad = dir.join("bad.jsonl");
    let lines = [
        chat_line("a", "x"),
        chat_line("a", "dup"),
        "{not json".into(),
        json!({"custom_id": "w", "method": "POST", "url": "/v1/embeddings",
               "body": {"model": MODEL}})
        .to_string(),
        json!({"custom_id": "m", "method": "POST", "url": "/v1/chat/completions",
               "body": {"model": "other"}})
        .to_string(),
        json!({"custom_id": "s", "method": "POST", "url": "/v1/chat/completions",
               "body": {"model": MODEL, "stream": true}})
        .to_string(),
    ];
    std::fs::write(&bad, lines.join("\n")).unwrap();
    let o = index::index_jsonl(&bad, "/v1/chat/completions", MODEL, |_| Ok(())).unwrap();
    let codes: Vec<&str> = o
        .errors
        .iter()
        .map(|e| e["code"].as_str().unwrap())
        .collect();
    assert_eq!(
        codes,
        [
            "duplicate_custom_id",
            "invalid_json_line",
            "mismatched_endpoint",
            "mismatched_model",
            "stream_not_supported"
        ]
    );
    assert_eq!(o.errors[0]["line"], 2);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn anthropic_requests_convert_to_compact_lines() {
    let body = json!({"requests": [
        {"custom_id": "r1", "params": {"model": MODEL, "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}]}},
        {"custom_id": "r2", "params": {"model": MODEL, "max_tokens": 8,
            "messages": [{"role": "user", "content": "there"}]}},
    ], "paddock": {"concurrency": 2}});
    // pretty-printed on purpose: the framing must survive it
    let src = serde_json::to_string_pretty(&body).unwrap();
    let mut out = Vec::new();
    let c = index::convert_anthropic(src.as_bytes(), &mut out).unwrap();
    assert_eq!((c.total, c.model.as_str()), (2, MODEL));
    assert_eq!(c.paddock["concurrency"], 2);
    let ls = lines(&out);
    assert_eq!(ls.len(), 2);
    assert_eq!(ls[1]["custom_id"], "r2");
    assert_eq!(ls[1]["url"], "/v1/messages");
    assert_eq!(ls[1]["body"]["messages"][0]["content"], "there");

    let dup = json!({"requests": [
        {"custom_id": "r1", "params": {"model": MODEL}},
        {"custom_id": "r1", "params": {"model": MODEL}},
    ]});
    let e = index::convert_anthropic(dup.to_string().as_bytes(), Vec::new())
        .err()
        .unwrap();
    assert!(e.contains("used more than once"), "{e}");
    let bad_id = json!({"requests": [{"custom_id": "no spaces", "params": {"model": MODEL}}]});
    assert!(index::convert_anthropic(bad_id.to_string().as_bytes(), Vec::new()).is_err());
    let mixed = json!({"requests": [
        {"custom_id": "a", "params": {"model": MODEL}},
        {"custom_id": "b", "params": {"model": "other"}},
    ]});
    let e = index::convert_anthropic(mixed.to_string().as_bytes(), Vec::new())
        .err()
        .unwrap();
    assert!(e.contains("one batch runs on one model"), "{e}");
}

#[test]
fn result_file_reopens_at_its_committed_length() {
    let dir = std::env::temp_dir().join(format!("paddock-rf-{}", uuid::Uuid::new_v4()));
    let p = dir.join("out.jsonl");
    let mut f = files::ResultFile::open(&p, 0).unwrap();
    f.append(b"{\"a\":1}\n").unwrap();
    f.sync().unwrap();
    let committed = f.len;
    f.append(b"{\"torn\":").unwrap(); // never committed
    drop(f);
    let mut f = files::ResultFile::open(&p, committed).unwrap();
    assert_eq!(std::fs::read(&p).unwrap(), b"{\"a\":1}\n");
    f.append(b"{\"b\":2}\n").unwrap();
    f.sync().unwrap();
    assert_eq!(std::fs::read(&p).unwrap(), b"{\"a\":1}\n{\"b\":2}\n");
    // a committed length the file cannot back is refused, never papered over
    assert!(files::ResultFile::open(&p, 999).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_rate_forgets_a_pause_within_minutes() {
    use std::collections::VecDeque;
    use std::time::Instant;
    let now = Instant::now() + Duration::from_secs(3600);
    let at = |s: u64| now - Duration::from_secs(s);
    // 120 settles in the last two minutes, after a run that stalled 20 min ago
    let mut settled: VecDeque<Instant> = (0..50).map(|i| at(1500 - i)).collect();
    settled.extend((0..120).map(|i| at(120 - i)));
    let r = dispatch::rate_per_minute(&settled, now).unwrap();
    assert!(
        (r - 60.0).abs() < 1.0,
        "{r}: only the last five minutes count"
    );
    // too little history promises nothing
    let early: VecDeque<Instant> = [at(10), at(5)].into_iter().collect();
    assert_eq!(dispatch::rate_per_minute(&early, now), None);
}

#[test]
fn completion_windows() {
    assert_eq!(parse_window("24h"), Some(86_400_000));
    assert_eq!(parse_window("7d"), Some(7 * 86_400_000));
    assert_eq!(parse_window("30d"), Some(30 * 86_400_000));
    for bad in ["", "0h", "31d", "24", "1w", "-1h", "h"] {
        assert_eq!(parse_window(bad), None, "{bad}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn openai_batch_runs_to_completion_with_every_failure_class() {
    let r = rig(4).await;
    tokio::spawn(dispatch::run(r.b.clone()));
    let mut jsonl = Vec::new();
    let mut expect = HashSet::new();
    for i in 0..60 {
        let content = match i {
            7 => "bad".to_string(),
            13 => "flaky".to_string(),
            _ => format!("q{i}"),
        };
        jsonl.push(chat_line(&format!("req-{i}"), &content));
        expect.insert(format!("req-{i}"));
    }
    let file = upload(&r.app, &jsonl.join("\n")).await;
    let id = create_openai(&r.app, &file, json!({"metadata": {"job": "t1"}})).await;
    let row = wait_terminal(&r.db, &id).await;
    assert_eq!(row.status, "completed");
    assert_eq!((row.total, row.succeeded, row.errored), (60, 59, 1));
    assert_eq!(row.input_tokens, 590);

    let (_, v) = call(&r.app, get(&format!("/v1/batches/{id}"))).await;
    assert_eq!(v["status"], "completed");
    assert_eq!(
        v["request_counts"],
        json!({"total": 60, "completed": 59, "failed": 1})
    );
    assert_eq!(v["metadata"]["job"], "t1");
    assert_eq!(v["usage"]["output_tokens"], 295);
    let out = file_lines(&r.app, v["output_file_id"].as_str()).await;
    let err = file_lines(&r.app, v["error_file_id"].as_str()).await;
    assert_eq!((out.len(), err.len()), (59, 1));
    for l in &out {
        let cid = l["custom_id"].as_str().unwrap();
        let n: usize = cid.trim_start_matches("req-").parse().unwrap();
        let want = if n == 13 {
            "echo:flaky".to_string()
        } else {
            format!("echo:q{n}")
        };
        assert_eq!(l["response"]["status_code"], 200);
        assert_eq!(
            l["response"]["body"]["choices"][0]["message"]["content"], want,
            "{cid} carries its own answer"
        );
        assert!(l["id"].as_str().unwrap().starts_with("batch_req_"));
    }
    assert_eq!(err[0]["custom_id"], "req-7");
    assert_eq!(err[0]["response"]["status_code"], 400);
    assert_eq!(
        err[0]["response"]["body"]["error"]["message"],
        "the picture cannot be read"
    );
    let mut all = out.clone();
    all.extend(err);
    exactly_once(&all, &expect);
    // the flaky one took three tries, the bad one exactly one
    let calls = r.fake.calls.lock().unwrap().clone();
    assert_eq!(calls["flaky"], 3);
    assert_eq!(calls["bad"], 1);
    // width: max_batch 4 plus one waiting in the runner's queue
    assert!(r.fake.peak.load(Ordering::SeqCst) <= 5);
    let (_, l) = call(&r.app, get("/v1/batches?limit=10")).await;
    assert_eq!(l["data"][0]["id"], id.as_str());
    let _ = std::fs::remove_dir_all(&r.root);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_request_that_kills_its_runner_is_isolated_and_failed_alone() {
    let r = rig(3).await;
    tokio::spawn(dispatch::run(r.b.clone()));
    let mut jsonl = Vec::new();
    let mut expect = HashSet::new();
    for i in 0..20 {
        let content = if i == 5 {
            "crash".to_string()
        } else {
            format!("q{i}")
        };
        jsonl.push(chat_line(&format!("c{i}"), &content));
        expect.insert(format!("c{i}"));
    }
    let file = upload(&r.app, &jsonl.join("\n")).await;
    let id = create_openai(&r.app, &file, json!({})).await;
    let row = wait_terminal(&r.db, &id).await;
    assert_eq!(
        (row.status.as_str(), row.succeeded, row.errored),
        ("completed", 19, 1)
    );
    let (_, v) = call(&r.app, get(&format!("/v1/batches/{id}"))).await;
    let err = file_lines(&r.app, v["error_file_id"].as_str()).await;
    assert_eq!(err[0]["custom_id"], "c5");
    assert!(err[0]["response"].is_null());
    assert!(
        err[0]["error"]["message"]
            .as_str()
            .unwrap()
            .contains("running alone"),
        "{}",
        err[0]
    );
    // two strikes in company, then once alone - never a crash loop
    assert_eq!(r.fake.calls.lock().unwrap()["crash"], 3);
    let mut all = file_lines(&r.app, v["output_file_id"].as_str()).await;
    all.extend(err);
    exactly_once(&all, &expect);
    let _ = std::fs::remove_dir_all(&r.root);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_manager_crash_resumes_exactly_once_over_torn_bytes() {
    let r = rig(2).await;
    r.fake.delay_ms.store(30, Ordering::SeqCst);
    let first = tokio::spawn(dispatch::run(r.b.clone()));
    let mut jsonl = Vec::new();
    let mut expect = HashSet::new();
    for i in 0..40 {
        jsonl.push(chat_line(&format!("k{i}"), &format!("q{i}")));
        expect.insert(format!("k{i}"));
    }
    let file = upload(&r.app, &jsonl.join("\n")).await;
    let id = create_openai(&r.app, &file, json!({})).await;
    wait_for(&r.db, &id, |r| r.succeeded >= 10).await;
    // the manager dies mid-run...
    first.abort();
    let _ = first.await;
    // ...leaving a torn line past the committed length
    let out = r.root.join("jobs").join(&id).join("out.jsonl");
    {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new().append(true).open(&out).unwrap();
        f.write_all(b"{\"id\":\"torn").unwrap();
    }
    // a fresh process on the same store and directory
    let b2 = Arc::new(Batches::new(
        r.root.clone(),
        r.db.clone(),
        r.fleet.clone(),
        fast(),
    ));
    tokio::spawn(dispatch::run(b2));
    let row = wait_terminal(&r.db, &id).await;
    assert_eq!((row.status.as_str(), row.succeeded), ("completed", 40));
    let (_, v) = call(&r.app, get(&format!("/v1/batches/{id}"))).await;
    let all = file_lines(&r.app, v["output_file_id"].as_str()).await;
    exactly_once(&all, &expect);
    let _ = std::fs::remove_dir_all(&r.root);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancel_keeps_finished_results_and_accounts_for_the_rest() {
    let r = rig(2).await;
    r.fake.delay_ms.store(60, Ordering::SeqCst);
    tokio::spawn(dispatch::run(r.b.clone()));
    let mut jsonl = Vec::new();
    let mut expect = HashSet::new();
    for i in 0..40 {
        jsonl.push(chat_line(&format!("x{i}"), &format!("q{i}")));
        expect.insert(format!("x{i}"));
    }
    let file = upload(&r.app, &jsonl.join("\n")).await;
    let id = create_openai(&r.app, &file, json!({})).await;
    wait_for(&r.db, &id, |r| r.succeeded >= 4).await;
    let (s, v) = call(
        &r.app,
        post_json(&format!("/v1/batches/{id}/cancel"), json!({})),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(v["status"], "cancelling");
    let row = wait_terminal(&r.db, &id).await;
    assert_eq!(row.status, "cancelled");
    assert!(row.canceled > 0 && row.succeeded + row.canceled + row.errored == 40);
    let (_, v) = call(&r.app, get(&format!("/v1/batches/{id}"))).await;
    let mut all = file_lines(&r.app, v["output_file_id"].as_str()).await;
    let err = file_lines(&r.app, v["error_file_id"].as_str()).await;
    assert!(err.iter().all(|l| l["error"]["code"] == "batch_cancelled"));
    all.extend(err);
    exactly_once(&all, &expect);
    // cancelling a finished batch is refused
    let (s, _) = call(
        &r.app,
        post_json(&format!("/v1/batches/{id}/cancel"), json!({})),
    )
    .await;
    assert_eq!(s, StatusCode::CONFLICT);
    let _ = std::fs::remove_dir_all(&r.root);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_batch_waits_out_its_service_and_yields_to_live_traffic() {
    let r = rig(2).await;
    r.fake.delay_ms.store(20, Ordering::SeqCst);
    tokio::spawn(dispatch::run(r.b.clone()));
    let jsonl: Vec<String> = (0..30)
        .map(|i| chat_line(&format!("w{i}"), &format!("q{i}")))
        .collect();
    let file = upload(&r.app, &jsonl.join("\n")).await;
    let id = create_openai(&r.app, &file, json!({})).await;
    wait_for(&r.db, &id, |r| r.succeeded >= 3).await;
    // the service goes away: the batch waits rather than failing
    r.fleet.up.store(false, Ordering::SeqCst);
    let v = wait_state(&r.app, &id, "waiting_for_service").await;
    assert_eq!(v["status"], "in_progress");
    // it comes back as a new process, with somebody else's request on it
    r.fake.live.store(1, Ordering::SeqCst);
    r.fleet.pid.store(4343, Ordering::SeqCst);
    r.fleet.up.store(true, Ordering::SeqCst);
    wait_state(&r.app, &id, "yielding").await;
    // what was already sent finishes; nothing new goes out while it lasts
    tokio::time::sleep(Duration::from_millis(150)).await;
    let s1 = r.db.batch_get(&id).unwrap().unwrap().succeeded;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let s2 = r.db.batch_get(&id).unwrap().unwrap().succeeded;
    assert_eq!(s1, s2, "no batch work while live traffic runs");
    assert!(s2 < 30);
    // live traffic ends; after the dwell the batch finishes
    r.fake.live.store(0, Ordering::SeqCst);
    let row = wait_terminal(&r.db, &id).await;
    assert_eq!((row.status.as_str(), row.succeeded), ("completed", 30));
    let _ = std::fs::remove_dir_all(&r.root);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn anthropic_batch_and_webhook() {
    let r = rig(4).await;
    tokio::spawn(dispatch::run(r.b.clone()));
    tokio::spawn(webhook::run(r.b.clone()));
    let mut requests = Vec::new();
    let mut expect = HashSet::new();
    for i in 0..25 {
        let content = if i == 3 {
            "bad".to_string()
        } else {
            format!("m{i}")
        };
        requests.push(json!({"custom_id": format!("a-{i}"), "params": {
            "model": MODEL, "max_tokens": 16,
            "messages": [{"role": "user", "content": content}]}}));
        expect.insert(format!("a-{i}"));
    }
    let hook = format!("http://127.0.0.1:{}/hook", r.fleet.port);
    let (s, v) = call(
        &r.app,
        post_json(
            "/v1/messages/batches",
            json!({"requests": requests, "paddock": {"webhook_url": hook}}),
        ),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["type"], "message_batch");
    assert_eq!(v["processing_status"], "in_progress");
    assert_eq!(v["request_counts"]["processing"], 25);
    assert!(v["results_url"].is_null());
    let id = v["id"].as_str().unwrap().to_owned();
    assert!(id.starts_with("msgbatch_"));
    wait_terminal(&r.db, &id).await;
    let (_, v) = call(&r.app, get(&format!("/v1/messages/batches/{id}"))).await;
    assert_eq!(v["processing_status"], "ended");
    assert_eq!(
        v["request_counts"],
        json!({"processing": 0, "succeeded": 24, "errored": 1, "canceled": 0, "expired": 0})
    );
    assert!(
        v["results_url"]
            .as_str()
            .unwrap()
            .ends_with(&format!("/v1/messages/batches/{id}/results"))
    );
    let (s, bytes) = raw(&r.app, get(&format!("/v1/messages/batches/{id}/results"))).await;
    assert_eq!(s, StatusCode::OK);
    let res = lines(&bytes);
    exactly_once(&res, &expect);
    for l in &res {
        if l["custom_id"] == "a-3" {
            assert_eq!(l["result"]["type"], "errored");
            assert_eq!(l["result"]["error"]["type"], "error");
            assert_eq!(
                l["result"]["error"]["error"]["type"],
                "invalid_request_error"
            );
        } else {
            assert_eq!(l["result"]["type"], "succeeded");
            assert_eq!(l["result"]["message"]["type"], "message");
        }
    }
    // the webhook arrives once, in the OpenAI event shape
    for _ in 0..100 {
        if !r.fake.hooks.lock().unwrap().is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    tokio::time::sleep(Duration::from_millis(500)).await;
    let hooks = r.fake.hooks.lock().unwrap().clone();
    assert_eq!(hooks.len(), 1, "{hooks:?}");
    assert_eq!(hooks[0]["type"], "batch.completed");
    assert_eq!(hooks[0]["data"]["id"], id.as_str());
    // ended batches can be deleted; running ones cannot
    let del = Request::delete(format!("/v1/messages/batches/{id}"))
        .body(Body::empty())
        .unwrap();
    let (s, v) = call(&r.app, del).await;
    assert_eq!(
        (s, v["type"].as_str()),
        (StatusCode::OK, Some("message_batch_deleted"))
    );
    let (s, _) = call(&r.app, get(&format!("/v1/messages/batches/{id}"))).await;
    assert_eq!(s, StatusCode::NOT_FOUND);
    let _ = std::fs::remove_dir_all(&r.root);
}

#[tokio::test]
async fn create_refuses_what_cannot_run() {
    let r = rig(2).await;
    // a model nothing serves
    let line = json!({"custom_id": "a", "method": "POST", "url": "/v1/chat/completions",
                      "body": {"model": "nobody-serves-this"}})
    .to_string();
    let file = upload(&r.app, &line).await;
    let (s, v) = call(
        &r.app,
        post_json(
            "/v1/batches",
            json!({"input_file_id": file, "endpoint": "/v1/chat/completions",
                   "completion_window": "24h"}),
        ),
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    let msg = v["error"]["message"].as_str().unwrap();
    assert!(
        msg.contains("nobody-serves-this") && msg.contains(MODEL),
        "{msg}"
    );
    // a window outside the accepted range, an unknown endpoint, a bad option
    let ok_file = upload(&r.app, &chat_line("a", "x")).await;
    for (body, needle) in [
        (json!({"completion_window": "90d"}), "completion_window"),
        (json!({"endpoint": "/v1/images/generations"}), "endpoint"),
        (
            json!({"paddock": {"concurrency": 0}}),
            "paddock.concurrency",
        ),
        (json!({"paddock": {"turbo": true}}), "paddock.turbo"),
    ] {
        let mut b = json!({"input_file_id": ok_file, "endpoint": "/v1/chat/completions",
                           "completion_window": "24h"});
        for (k, v) in body.as_object().unwrap() {
            b[k] = v.clone();
        }
        let (s, v) = call(&r.app, post_json("/v1/batches", b)).await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{needle}");
        assert!(
            v["error"]["message"].as_str().unwrap().contains(needle),
            "{v}"
        );
    }
    // the input of a running batch cannot be deleted from under it
    let id = create_openai(&r.app, &ok_file, json!({})).await;
    let del = Request::delete(format!("/v1/files/{ok_file}"))
        .body(Body::empty())
        .unwrap();
    let (s, _) = call(&r.app, del).await;
    assert_eq!(s, StatusCode::CONFLICT, "batch {id} still reads it");
    let _ = std::fs::remove_dir_all(&r.root);
}

#[tokio::test]
async fn validation_failure_fails_the_batch_with_line_errors() {
    let r = rig(2).await;
    tokio::spawn(dispatch::run(r.b.clone()));
    let text = [chat_line("a", "x"), chat_line("a", "y")].join("\n");
    let file = upload(&r.app, &text).await;
    let id = create_openai(&r.app, &file, json!({})).await;
    let row = wait_terminal(&r.db, &id).await;
    assert_eq!(row.status, "failed");
    let (_, v) = call(&r.app, get(&format!("/v1/batches/{id}"))).await;
    assert_eq!(v["errors"]["data"][0]["code"], "duplicate_custom_id");
    assert_eq!(v["errors"]["data"][0]["line"], 2);
    assert!(r.fake.calls.lock().unwrap().is_empty(), "nothing ran");
    let _ = std::fs::remove_dir_all(&r.root);
}

#[tokio::test]
async fn the_batch_surface_takes_either_key_header() {
    let mut state = crate::routes::AppState::for_tests();
    state.auth_key = Some("sekrit".into());
    let app = crate::routes::router(Arc::new(state));
    let (s, _) = call(&app, get("/v1/batches")).await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);
    let req = Request::get("/v1/batches")
        .header("authorization", "Bearer sekrit")
        .body(Body::empty())
        .unwrap();
    assert_eq!(call(&app, req).await.0, StatusCode::OK);
    let req = Request::get("/v1/messages/batches")
        .header("x-api-key", "sekrit")
        .body(Body::empty())
        .unwrap();
    assert_eq!(call(&app, req).await.0, StatusCode::OK);
}

/// The store at scale: one batch of `SCALE_N` (default 200 000) requests
/// against a runner that answers at once, with production commit settings,
/// while another thread keeps reading the store the way the Studio does.
/// Opt-in (`cargo test -p paddock-manager --lib scale -- --ignored
/// --nocapture`); it prints what it measured.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore]
async fn scale_one_batch_of_many_requests() {
    use std::time::Instant;
    let n: usize = std::env::var("SCALE_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200_000);
    let (port, fake) = start_fake().await;
    let fleet = Arc::new(TestFleet {
        port,
        pid: AtomicU32::new(1),
        up: AtomicBool::new(true),
        max_batch: 64,
        fake: fake.clone(),
    });
    let dir = std::env::temp_dir().join(format!("paddock-batch-scale-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let db_path = dir.join("paddock.db");
    let db = Arc::new(crate::store::Store::open(&db_path).unwrap());
    let mut state = crate::routes::AppState::for_tests();
    state.db = db.clone();
    let tuning = Tuning {
        tick: Duration::from_millis(200),
        ..Tuning::default()
    };
    let b = Arc::new(Batches::new(dir.join("batches"), db.clone(), fleet, tuning));
    state.batches = b.clone();
    let app = crate::routes::router(Arc::new(state));
    tokio::spawn(dispatch::run(b.clone()));

    // the same read on an idle store, for the baseline
    let idle = {
        let mut v: Vec<Duration> = (0..400)
            .map(|_| {
                let t = Instant::now();
                let _ = db.list_conversations();
                std::thread::sleep(Duration::from_millis(1));
                t.elapsed().saturating_sub(Duration::from_millis(1))
            })
            .collect();
        v.sort();
        (v[200], v[396], v[399])
    };
    // the Studio's side of the lock: a read every 5 ms, timed
    let stop = Arc::new(AtomicBool::new(false));
    let origin = Instant::now();
    let reader = {
        let (db, stop) = (db.clone(), stop.clone());
        std::thread::spawn(move || {
            let mut lat = Vec::new();
            let mut worst: Vec<(Duration, Duration)> = Vec::new();
            while !stop.load(Ordering::SeqCst) {
                let t = Instant::now();
                let _ = db.list_conversations();
                let d = t.elapsed();
                lat.push(d);
                worst.push((d, t - origin));
                std::thread::sleep(Duration::from_millis(5));
            }
            lat.sort();
            worst.sort_by_key(|w| std::cmp::Reverse(w.0));
            worst.truncate(3);
            (lat, worst)
        })
    };
    let mut jsonl = String::with_capacity(n * 140);
    for i in 0..n {
        jsonl.push_str(&chat_line(&format!("s{i}"), &format!("q{i}")));
        jsonl.push('\n');
    }
    let t0 = Instant::now();
    let file = upload(&app, &jsonl).await;
    let t_up = t0.elapsed();
    let id = create_openai(&app, &file, json!({})).await;
    let r = wait_for(&db, &id, |r| r.status != "validating").await;
    let t_valid = t0.elapsed() - t_up;
    let db_size = |p: &std::path::Path| {
        ["", "-wal"]
            .iter()
            .map(|s| {
                std::fs::metadata(format!("{}{s}", p.display()))
                    .map(|m| m.len())
                    .unwrap_or(0)
            })
            .sum::<u64>()
    };
    let size_indexed = db_size(&db_path);
    assert_eq!(r.total, n as i64);
    let t_run0 = Instant::now();
    let mut row = r;
    // SCALE_CANCEL=1: cancel a tenth of the way in, timing the close of the rest
    if std::env::var("SCALE_CANCEL").is_ok() {
        wait_for(&db, &id, |r| r.succeeded >= n as i64 / 10).await;
        let t = Instant::now();
        let (s, _) = call(
            &app,
            post_json(&format!("/v1/batches/{id}/cancel"), json!({})),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        let r = wait_terminal(&db, &id).await;
        println!(
            "SCALE cancel at {}: {} closed as cancelled in {:.1}s",
            r.succeeded,
            r.canceled,
            t.elapsed().as_secs_f64()
        );
        assert_eq!(r.succeeded + r.errored + r.canceled, n as i64);
        row = r;
    }
    while !row.is_terminal() {
        tokio::time::sleep(Duration::from_millis(200)).await;
        row = db.batch_get(&id).unwrap().unwrap();
    }
    let t_run = t_run0.elapsed();
    stop.store(true, Ordering::SeqCst);
    let (lat, worst) = reader.join().unwrap();
    println!(
        "SCALE phases: uploaded {:?}, indexed {:?}, done {:?}; worst reads (latency, at) {worst:?}",
        t_up,
        t_up + t_valid,
        t_up + t_valid + t_run
    );
    let pct = |p: f64| lat[((lat.len() as f64 * p) as usize).min(lat.len() - 1)];
    // what is left in the store once the batch is done
    tokio::time::sleep(Duration::from_millis(1500)).await;
    let size_done = db_size(&db_path);
    println!(
        "SCALE idle read p50/p99/max {:?}/{:?}/{:?}\nSCALE n={n}: upload {:.1}s, validate+index {:.1}s ({:.0} lines/s), run {:.1}s ({:.0} req/s), \
         db {} MB indexed, {} MB after; studio read p50 {:?} p99 {:?} max {:?} over {} reads; peak in flight {}",
        idle.0,
        idle.1,
        idle.2,
        t_up.as_secs_f64(),
        t_valid.as_secs_f64(),
        n as f64 / t_valid.as_secs_f64(),
        t_run.as_secs_f64(),
        n as f64 / t_run.as_secs_f64(),
        size_indexed >> 20,
        size_done >> 20,
        pct(0.5),
        pct(0.99),
        lat.last().unwrap(),
        lat.len(),
        fake.peak.load(Ordering::SeqCst),
    );
    if std::env::var("SCALE_CANCEL").is_err() {
        assert_eq!(
            (row.status.as_str(), row.succeeded),
            ("completed", n as i64)
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

fn keyed(mut req: Request<Body>, key: &str) -> Request<Body> {
    req.headers_mut()
        .insert("idempotency-key", key.parse().unwrap());
    req
}

#[tokio::test]
async fn a_retried_create_or_upload_returns_the_first() {
    let r = rig(2).await;
    let line = chat_line("a", "x");
    let boundary = "XpaddockX";
    let up = || {
        let body = format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n\
             --{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"in.jsonl\"\r\n\r\n\
             {line}\r\n--{boundary}--\r\n"
        );
        keyed(
            Request::post("/v1/files")
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body))
                .unwrap(),
            "up-1",
        )
    };
    let (_, f1) = call(&r.app, up()).await;
    let (_, f2) = call(&r.app, up()).await;
    assert_eq!(f1["id"], f2["id"], "the retried upload is the same file");
    let (_, list) = call(&r.app, get("/v1/files")).await;
    assert_eq!(list["data"].as_array().unwrap().len(), 1);
    let create = || {
        keyed(
            post_json(
                "/v1/batches",
                json!({"input_file_id": f1["id"], "endpoint": "/v1/chat/completions",
                       "completion_window": "24h"}),
            ),
            "create-1",
        )
    };
    let (_, b1) = call(&r.app, create()).await;
    let (_, b2) = call(&r.app, create()).await;
    assert_eq!(b1["id"], b2["id"], "the retried create is the same batch");
    let (_, list) = call(&r.app, get("/v1/batches")).await;
    assert_eq!(list["data"].as_array().unwrap().len(), 1);
    // Anthropic: the replay is answered before its body is read again
    let msg = || {
        keyed(
            post_json(
                "/v1/messages/batches",
                json!({"requests": [{"custom_id": "a", "params": {"model": MODEL,
                    "messages": [{"role": "user", "content": "x"}]}}]}),
            ),
            "msg-1",
        )
    };
    let (_, m1) = call(&r.app, msg()).await;
    let (_, m2) = call(&r.app, msg()).await;
    assert_eq!(m1["id"], m2["id"]);
    // a key belongs to one dialect
    let (s, _) = call(
        &r.app,
        keyed(
            post_json(
                "/v1/batches",
                json!({"input_file_id": f1["id"], "endpoint": "/v1/chat/completions",
                       "completion_window": "24h"}),
            ),
            "msg-1",
        ),
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    let _ = std::fs::remove_dir_all(&r.root);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn expiry_keeps_what_finished_and_names_the_rest() {
    let r = rig(2).await;
    r.fake.delay_ms.store(40, Ordering::SeqCst);
    tokio::spawn(dispatch::run(r.b.clone()));
    let jsonl: Vec<String> = (0..40)
        .map(|i| chat_line(&format!("e{i}"), &format!("q{i}")))
        .collect();
    let expect: HashSet<String> = (0..40).map(|i| format!("e{i}")).collect();
    let file = upload(&r.app, &jsonl.join("\n")).await;
    let id = create_openai(&r.app, &file, json!({})).await;
    wait_for(&r.db, &id, |r| r.succeeded >= 5).await;
    r.db.batch_set_expires(&id, 0);
    let row = wait_terminal(&r.db, &id).await;
    assert_eq!(row.status, "expired");
    assert!(
        row.expired > 0 && row.succeeded + row.expired == 40,
        "{row:?}"
    );
    let (_, v) = call(&r.app, get(&format!("/v1/batches/{id}"))).await;
    assert!(v["expired_at"].is_i64());
    let mut all = file_lines(&r.app, v["output_file_id"].as_str()).await;
    let err = file_lines(&r.app, v["error_file_id"].as_str()).await;
    assert!(err.iter().all(|l| l["error"]["code"] == "batch_expired"));
    all.extend(err);
    exactly_once(&all, &expect);
    // the finished batch's item rows are gone; its results are in the files
    for _ in 0..100 {
        if r.db.batch_items_count(&id) == 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(r.db.batch_items_count(&id), 0);
    let _ = std::fs::remove_dir_all(&r.root);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn one_request_that_never_answers_fails_after_its_attempts() {
    let tuning = Tuning {
        request_timeout: Duration::from_millis(300),
        ..fast()
    };
    let r = rig_tuned(3, tuning).await;
    r.fake.delay_ms.store(20, Ordering::SeqCst);
    tokio::spawn(dispatch::run(r.b.clone()));
    let jsonl: Vec<String> = (0..30)
        .map(|i| chat_line(&format!("h{i}"), if i == 4 { "hang" } else { "ok" }))
        .collect();
    let file = upload(&r.app, &jsonl.join("\n")).await;
    let id = create_openai(&r.app, &file, json!({})).await;
    let row = wait_terminal(&r.db, &id).await;
    assert_eq!(
        (row.status.as_str(), row.succeeded, row.errored),
        ("completed", 29, 1)
    );
    let (_, v) = call(&r.app, get(&format!("/v1/batches/{id}"))).await;
    let err = file_lines(&r.app, v["error_file_id"].as_str()).await;
    assert_eq!(err[0]["custom_id"], "h4");
    assert_eq!(err[0]["error"]["code"], "timeout");
    // Three attempts while the others answered between its timeouts (a slow
    // request, not a wedged service). Once it is the last one running, more
    // timeouts in a row read as the service stalling, so it takes strikes
    // and fails alone: a bounded handful of tries either way, never a loop.
    let calls = r.fake.calls.lock().unwrap()["hang"];
    assert!((3..=6).contains(&calls), "{calls} tries");
    let _ = std::fs::remove_dir_all(&r.root);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_wedged_service_pauses_the_batch_without_failing_it() {
    let tuning = Tuning {
        request_timeout: Duration::from_millis(250),
        ..fast()
    };
    let r = rig_tuned(2, tuning).await;
    r.fake.delay_ms.store(10, Ordering::SeqCst);
    tokio::spawn(dispatch::run(r.b.clone()));
    let jsonl: Vec<String> = (0..40)
        .map(|i| chat_line(&format!("g{i}"), &format!("q{i}")))
        .collect();
    let file = upload(&r.app, &jsonl.join("\n")).await;
    let id = create_openai(&r.app, &file, json!({})).await;
    wait_for(&r.db, &id, |r| r.succeeded >= 5).await;
    // health still answers; work no longer finishes
    r.fake.wedged.store(true, Ordering::SeqCst);
    let v = wait_state(&r.app, &id, "waiting_for_service").await;
    assert!(
        v["paddock"]["reason"]
            .as_str()
            .unwrap()
            .contains("stopped answering"),
        "{v}"
    );
    tokio::time::sleep(Duration::from_millis(800)).await;
    assert_eq!(r.db.batch_get(&id).unwrap().unwrap().errored, 0);
    r.fake.wedged.store(false, Ordering::SeqCst);
    let row = wait_terminal(&r.db, &id).await;
    assert_eq!(
        (row.status.as_str(), row.succeeded, row.errored),
        ("completed", 40, 0)
    );
    let _ = std::fs::remove_dir_all(&r.root);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_full_disk_pauses_the_batch_until_writes_succeed() {
    let r = rig(2).await;
    r.fake.delay_ms.store(10, Ordering::SeqCst);
    tokio::spawn(dispatch::run(r.b.clone()));
    let jsonl: Vec<String> = (0..40)
        .map(|i| chat_line(&format!("d{i}"), &format!("q{i}")))
        .collect();
    let expect: HashSet<String> = (0..40).map(|i| format!("d{i}")).collect();
    let file = upload(&r.app, &jsonl.join("\n")).await;
    let id = create_openai(&r.app, &file, json!({})).await;
    wait_for(&r.db, &id, |r| r.succeeded >= 5).await;
    files::FAIL_WRITES_UNDER
        .lock()
        .unwrap()
        .push(r.root.clone());
    let v = wait_state(&r.app, &id, "paused").await;
    assert!(
        v["paddock"]["reason"].as_str().unwrap().contains("disk"),
        "{v}"
    );
    // paused means paused: no requests re-run whose answers cannot be kept
    let before = r.fake.calls.lock().unwrap().values().sum::<u32>();
    tokio::time::sleep(Duration::from_millis(250)).await;
    let after = r.fake.calls.lock().unwrap().values().sum::<u32>();
    assert!(
        after - before <= 6,
        "{before} -> {after}: re-running into a full disk"
    );
    files::FAIL_WRITES_UNDER
        .lock()
        .unwrap()
        .retain(|d| d != &r.root);
    let row = wait_terminal(&r.db, &id).await;
    assert_eq!((row.status.as_str(), row.succeeded), ("completed", 40));
    let (_, v) = call(&r.app, get(&format!("/v1/batches/{id}"))).await;
    exactly_once(
        &file_lines(&r.app, v["output_file_id"].as_str()).await,
        &expect,
    );
    let _ = std::fs::remove_dir_all(&r.root);
}

async fn part(app: &axum::Router, upload: &str, data: &[u8]) -> String {
    let boundary = "XpartX";
    let mut body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"data\"; filename=\"blob\"\r\n\
         Content-Type: application/octet-stream\r\n\r\n"
    )
    .into_bytes();
    body.extend_from_slice(data);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    let req = Request::post(format!("/v1/uploads/{upload}/parts"))
        .header(
            "content-type",
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(Body::from(body))
        .unwrap();
    let (s, v) = call(app, req).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["object"], "upload.part");
    v["id"].as_str().unwrap().to_owned()
}

async fn new_upload(app: &axum::Router, bytes: usize) -> String {
    let (s, v) = call(
        app,
        post_json(
            "/v1/uploads",
            json!({"bytes": bytes, "filename": "in.jsonl", "mime_type": "application/jsonl",
                   "purpose": "batch"}),
        ),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(
        (v["object"].as_str(), v["status"].as_str()),
        (Some("upload"), Some("pending"))
    );
    v["id"].as_str().unwrap().to_owned()
}

fn md5_hex(b: &[u8]) -> String {
    use md5::Digest;
    md5::Md5::digest(b)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_resumable_upload_assembles_in_order_and_feeds_a_batch() {
    let r = rig(2).await;
    tokio::spawn(dispatch::run(r.b.clone()));
    let text: String = (0..30)
        .map(|i| chat_line(&format!("u{i}"), &format!("q{i}")) + "\n")
        .collect();
    let bytes = text.as_bytes();
    let (a, rest) = bytes.split_at(bytes.len() / 3);
    let (b, c) = rest.split_at(rest.len() / 2);
    let id = new_upload(&r.app, bytes.len()).await;
    // parts may arrive in any order; the completion names the order
    let pc = part(&r.app, &id, c).await;
    let pa = part(&r.app, &id, a).await;
    let pb = part(&r.app, &id, b).await;
    // wrong totals and repeats are refused, the upload stays pending
    for bad in [json!([pa, pb]), json!([pa, pa, pc])] {
        let (s, _) = call(
            &r.app,
            post_json(
                &format!("/v1/uploads/{id}/complete"),
                json!({"part_ids": bad}),
            ),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
    }
    let done = json!({"part_ids": [pa, pb, pc], "md5": md5_hex(bytes)});
    let (s, v) = call(
        &r.app,
        post_json(&format!("/v1/uploads/{id}/complete"), done.clone()),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["status"], "completed");
    assert_eq!(v["file"]["bytes"], bytes.len());
    let fid = v["file"]["id"].as_str().unwrap().to_owned();
    // completing again is the same answer
    let (_, again) = call(
        &r.app,
        post_json(&format!("/v1/uploads/{id}/complete"), done),
    )
    .await;
    assert_eq!(again["file"]["id"], fid.as_str());
    let (_, content) = raw(&r.app, get(&format!("/v1/files/{fid}/content"))).await;
    assert_eq!(content, bytes, "byte for byte, in the completion's order");
    let bid = create_openai(&r.app, &fid, json!({})).await;
    let row = wait_terminal(&r.db, &bid).await;
    assert_eq!((row.status.as_str(), row.succeeded), ("completed", 30));
    let _ = std::fs::remove_dir_all(&r.root);
}

#[tokio::test]
async fn an_interrupted_completion_resumes_where_it_stopped() {
    let r = rig(2).await;
    let data: Vec<u8> = (0..3000u32).flat_map(|i| i.to_le_bytes()).collect();
    let (a, b) = data.split_at(5000);
    let id = new_upload(&r.app, data.len()).await;
    let pa = part(&r.app, &id, a).await;
    let pb = part(&r.app, &id, b).await;
    // what a completion that died after its first part leaves behind: the
    // order fixed, part one appended and recorded, its file gone
    let dir = r.root.join("uploads").join(&id);
    let order = serde_json::to_string(&[&pa, &pb]).unwrap();
    assert!(r.db.upload_set_order(&id, &order).unwrap());
    std::fs::write(
        dir.join("assembly"),
        [a, b"torn tail past the record"].concat(),
    )
    .unwrap();
    r.db.upload_assembled(&id, 1, a.len() as i64).unwrap();
    std::fs::remove_file(dir.join(&pa)).unwrap();
    // a different order now is refused; the same one resumes
    let (s, _) = call(
        &r.app,
        post_json(
            &format!("/v1/uploads/{id}/complete"),
            json!({"part_ids": [pb, pa]}),
        ),
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    let (s, v) = call(
        &r.app,
        post_json(
            &format!("/v1/uploads/{id}/complete"),
            json!({"part_ids": [pa, pb]}),
        ),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    let fid = v["file"]["id"].as_str().unwrap();
    let (_, content) = raw(&r.app, get(&format!("/v1/files/{fid}/content"))).await;
    assert_eq!(content, data);
    assert!(!dir.exists(), "the parts are cleaned up");
    // a wrong checksum cancels: those bytes are not the ones the client meant
    let id2 = new_upload(&r.app, 4).await;
    let p = part(&r.app, &id2, b"abcd").await;
    let (s, v) = call(
        &r.app,
        post_json(
            &format!("/v1/uploads/{id2}/complete"),
            json!({"part_ids": [p], "md5": "00"}),
        ),
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert!(
        v["error"]["message"].as_str().unwrap().contains("md5"),
        "{v}"
    );
    assert_eq!(r.db.upload_get(&id2).unwrap().unwrap().status, "cancelled");
    let _ = std::fs::remove_dir_all(&r.root);
}
