//! HTTP capability, admission and validation checks; no GPU required.
#![allow(clippy::unwrap_used)]
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use paddock_engine::diarization::{Backend, Diarizer, Frontend};
use paddock_runner::{
    diarizations::DiarizationModel,
    routes::{AppState, router},
};
use serde_json::Value;
use std::sync::Arc;
use tower::ServiceExt;
struct Fake;
impl Backend for Fake {
    fn pre_encode(&mut self, f: &[f32]) -> Result<Vec<f32>, String> {
        Ok(vec![0.; f.len() / 1024 * 512])
    }
    fn predict(&mut self, x: &[f32], _valid: usize) -> Result<(Vec<[f32; 8]>, f64), String> {
        Ok((
            vec![[0.8, 0.9, 0., 0., 0., 0., 0., 0.]; x.len() / 512 * 8],
            0.001,
        ))
    }
    fn frontend(&self) -> Result<Frontend, String> {
        Frontend::new(&[1.; 400], &vec![0.; 128 * 257])
    }
    fn silence(&self) -> &[f32] {
        &[0.; 512]
    }
    fn weight_bytes(&self) -> u64 {
        0
    }
    fn workspace_bytes(&self) -> u64 {
        0
    }
}
fn app() -> (axum::Router, Arc<tokio::sync::Semaphore>) {
    let s = state();
    let admission = s.diarization.as_ref().unwrap().admission.clone();
    (router(s), admission)
}
fn state() -> Arc<AppState> {
    let mut s = AppState::for_tests(None);
    s.auth_key = Some("test-diarization-key".into());
    let admission = Arc::new(tokio::sync::Semaphore::new(2));
    s.diarization = Some(DiarizationModel {
        id: "diarizer".into(),
        service: Diarizer::spawn(|| Ok(Fake)).unwrap(),
        admission: admission.clone(),
    });
    Arc::new(s)
}
fn body(fields: &[(&str, &str)]) -> Vec<u8> {
    let mut out = Vec::new();
    for (name, value) in fields {
        out.extend_from_slice(
            format!(
                "--audio\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
            )
            .as_bytes(),
        );
    }
    let pcm = vec![0u8; 3200];
    let mut wav = Vec::new();
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36u32 + pcm.len() as u32).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&16000u32.to_le_bytes());
    wav.extend_from_slice(&32000u32.to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&(pcm.len() as u32).to_le_bytes());
    wav.extend(pcm);
    out.extend_from_slice(b"--audio\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\n");
    out.extend(wav);
    out.extend_from_slice(b"\r\n--audio--\r\n");
    out
}
async fn call(
    app: &axum::Router,
    path: &str,
    data: Option<Vec<u8>>,
    auth: bool,
) -> (StatusCode, Value) {
    let mut req = Request::builder()
        .method(if data.is_some() { "POST" } else { "GET" })
        .uri(path)
        .header("content-type", "multipart/form-data; boundary=audio");
    if auth {
        req = req.header("authorization", "Bearer test-diarization-key");
    }
    let result = app
        .clone()
        .oneshot(req.body(Body::from(data.unwrap_or_default())).unwrap())
        .await
        .unwrap();
    let status = result.status();
    let bytes = result.into_body().collect().await.unwrap().to_bytes();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}
#[tokio::test]
async fn discovery_auth_overlap_and_input_validation() {
    let (app, _) = app();
    let uri = "/v1/audio/diarizations";
    assert_eq!(
        call(&app, uri, Some(body(&[])), false).await.0,
        StatusCode::UNAUTHORIZED
    );
    let (status, out) = call(&app, uri, Some(body(&[("model", "diarizer")])), true).await;
    assert_eq!(status, StatusCode::OK, "{out}");
    assert_eq!(out["num_speakers"], 2);
    assert_eq!(out["frames"], 10);
    assert_eq!(out["segments"][0]["end"], 0.1);
    for fields in [
        vec![("threshold", "NaN")],
        vec![("threshold", "1")],
        vec![("preset", "guess")],
        vec![("stream", "true")],
        vec![("preset", "low"), ("preset", "low")],
    ] {
        assert_eq!(
            call(&app, uri, Some(body(&fields)), true).await.0,
            StatusCode::BAD_REQUEST
        );
    }
    assert_eq!(
        call(&app, uri, Some(body(&[("model", "other")])), true)
            .await
            .0,
        StatusCode::NOT_FOUND
    );
    let (_, models) = call(&app, "/v1/models", None, true).await;
    assert_eq!(models["data"].as_array().unwrap().len(), 1);
    assert_eq!(models["data"][0]["id"], "diarizer");
    let (_, server) = call(&app, "/api/server", None, true).await;
    assert_eq!(
        server["diarization"]["capabilities"]["transcription"],
        false
    );
}
#[tokio::test]
async fn word_attribution_retains_text_and_rejects_invalid_clocks() {
    let (app, _) = app();
    let uri = "/v1/audio/diarizations";
    let words = r#"[{"word":"Hej!","start":0,"end":0.1,"confidence":0.7},{"word":"?"}]"#;
    let (status, out) = call(&app, uri, Some(body(&[("words", words)])), true).await;
    assert_eq!(status, StatusCode::OK, "{out}");
    assert_eq!(out["attribution"], "time_overlap_v1");
    assert_eq!(out["words"][0]["word"], "Hej!");
    assert_eq!(out["words"][0]["confidence"], 0.7);
    assert_eq!(out["words"][0]["speakers"], serde_json::json!([0, 1]));
    assert!(out["words"][0]["speaker"].is_null());
    assert_eq!(out["words"][1]["speaker_status"], "untimed");
    for invalid in [
        r#"[{"word":"Hej","start":0,"end":1}]"#,
        r#"[{"word":"Hej","start":0}]"#,
        r#"{}"#,
    ] {
        assert_eq!(
            call(&app, uri, Some(body(&[("words", invalid)])), true)
                .await
                .0,
            StatusCode::BAD_REQUEST
        );
    }
}
#[tokio::test]
async fn backpressure_precedes_decoding() {
    let (app, admission) = app();
    let _a = admission.acquire().await.unwrap();
    let _b = admission.acquire().await.unwrap();
    assert_eq!(
        call(&app, "/v1/audio/diarizations", Some(body(&[])), true)
            .await
            .0,
        StatusCode::TOO_MANY_REQUESTS
    );
}

#[tokio::test]
async fn oversized_upload_is_413_not_a_decoder_error() {
    let (app, _) = app();
    let mut data =
        b"--audio\r\nContent-Disposition: form-data; name=\"file\"; filename=\"large.wav\"\r\n\r\n"
            .to_vec();
    data.resize(data.len() + 26 * 1024 * 1024, 0);
    data.extend_from_slice(b"\r\n--audio--\r\n");
    assert_eq!(
        call(&app, "/v1/audio/diarizations", Some(data), true)
            .await
            .0,
        StatusCode::PAYLOAD_TOO_LARGE
    );
}

use futures::{SinkExt, StreamExt};
use tokio_tungstenite::{
    connect_async,
    tungstenite::{Message, client::IntoClientRequest},
};
type Socket =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;
struct Server {
    url: String,
    task: tokio::task::JoinHandle<()>,
}
impl Drop for Server {
    fn drop(&mut self) {
        self.task.abort();
    }
}
async fn serve(state: Arc<AppState>) -> Server {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!(
        "ws://{}/v1/audio/diarizations/stream",
        listener.local_addr().unwrap()
    );
    let task = tokio::spawn(async move { axum::serve(listener, router(state)).await.unwrap() });
    Server { url, task }
}
async fn connect(url: &str) -> Result<Socket, tokio_tungstenite::tungstenite::Error> {
    let mut request = url.into_client_request().unwrap();
    request.headers_mut().insert(
        "authorization",
        "Bearer test-diarization-key".parse().unwrap(),
    );
    connect_async(request).await.map(|r| r.0)
}
async fn event(socket: &mut Socket) -> Value {
    let msg = tokio::time::timeout(std::time::Duration::from_secs(5), socket.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    serde_json::from_str(msg.to_text().unwrap()).unwrap()
}
fn status(e: tokio_tungstenite::tungstenite::Error) -> StatusCode {
    let tokio_tungstenite::tungstenite::Error::Http(r) = e else {
        panic!("unexpected: {e}")
    };
    r.status()
}

#[tokio::test]
async fn live_timeline_is_incremental_overlap_preserving_and_partition_invariant() {
    let s = state();
    let server = serve(s.clone()).await;
    for chunk in [2222, 32000] {
        let mut ws = connect(&format!("{}?model=diarizer&preset=ultra_low", server.url))
            .await
            .unwrap();
        assert_eq!(event(&mut ws).await["sequence_number"], 0);
        let mut completed = Vec::new();
        let pcm = vec![0u8; 16000 * 2 + 9600];
        let mut seq = 1;
        let mut previous_frame = 0;
        for bytes in pcm.chunks(chunk) {
            ws.send(Message::Binary(bytes.to_vec().into()))
                .await
                .unwrap();
            let e = event(&mut ws).await;
            assert_eq!(e["type"], "diarization.update");
            assert_eq!(e["sequence_number"], seq);
            assert!(e["frames"].as_u64().unwrap() >= previous_frame);
            previous_frame = e["frames"].as_u64().unwrap();
            completed.extend(e["completed"].as_array().unwrap().iter().cloned());
            seq += 1;
        }
        assert!(previous_frame > 0, "must produce output BEFORE finish");
        ws.send(Message::Text(r#"{"type":"finish"}"#.into()))
            .await
            .unwrap();
        let e = event(&mut ws).await;
        assert_eq!(e["type"], "diarization.done");
        assert_eq!(e["sequence_number"], seq);
        assert_eq!(e["received_samples"], 20800);
        assert_eq!(e["frames"], 130);
        assert_eq!(e["active"], serde_json::json!([]));
        completed.extend(e["completed"].as_array().unwrap().iter().cloned());
        assert_eq!(
            completed,
            serde_json::json!([
                {"speaker":0,"start":0.0,"end":1.3}, {"speaker":1,"start":0.0,"end":1.3}
            ])
            .as_array()
            .unwrap()
            .to_vec()
        );
        assert!(matches!(
            ws.next().await.unwrap().unwrap(),
            Message::Close(_)
        ));
    }
    assert!(
        s.drain
            .wait_drained(std::time::Duration::from_secs(2))
            .await
    );
}

#[tokio::test]
async fn live_admission_auth_drain_and_disconnect_release() {
    let s = state();
    let server = serve(s.clone()).await;
    assert_eq!(
        status(connect_async(&server.url).await.unwrap_err()),
        StatusCode::UNAUTHORIZED
    );
    for q in ["model=wrong", "preset=wrong", "threshold=NaN", "unknown=1"] {
        let st = status(connect(&format!("{}?{q}", server.url)).await.unwrap_err());
        assert_eq!(
            st,
            if q == "model=wrong" {
                StatusCode::NOT_FOUND
            } else {
                StatusCode::BAD_REQUEST
            }
        );
    }
    let mut a = connect(&server.url).await.unwrap();
    event(&mut a).await;
    let mut b = connect(&server.url).await.unwrap();
    event(&mut b).await;
    assert_eq!(s.drain.in_flight(), 2);
    assert_eq!(
        status(connect(&server.url).await.unwrap_err()),
        StatusCode::TOO_MANY_REQUESTS
    );
    let (code, _) = call(
        &router(s.clone()),
        "/v1/audio/diarizations",
        Some(body(&[])),
        true,
    )
    .await;
    assert_eq!(code, StatusCode::TOO_MANY_REQUESTS);
    drop(a);
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        while s
            .diarization
            .as_ref()
            .unwrap()
            .admission
            .available_permits()
            != 1
        {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let (code, _) = call(
        &router(s.clone()),
        "/v1/audio/diarizations",
        Some(body(&[])),
        true,
    )
    .await;
    assert_eq!(
        code,
        StatusCode::OK,
        "idle live session must not block a file"
    );
    s.drain.begin();
    assert_eq!(
        status(connect(&server.url).await.unwrap_err()),
        StatusCode::SERVICE_UNAVAILABLE
    );
    assert_eq!(
        call(
            &router(s.clone()),
            "/v1/audio/diarizations",
            Some(body(&[])),
            true
        )
        .await
        .0,
        StatusCode::SERVICE_UNAVAILABLE
    );
    drop(b);
    assert!(
        s.drain
            .wait_drained(std::time::Duration::from_secs(2))
            .await
    );
}

#[tokio::test]
async fn live_rejects_invalid_audio_and_control_without_leaking_slots() {
    let s = state();
    let server = serve(s.clone()).await;
    for bad in [
        Message::Binary(vec![0u8; 3].into()),
        Message::Binary(Vec::new().into()),
        Message::Text(r#"{"type":"finish","extra":1}"#.into()),
        Message::Text(r#"{"type":"finish"}"#.into()),
    ] {
        let mut ws = connect(&server.url).await.unwrap();
        event(&mut ws).await;
        ws.send(bad).await.unwrap();
        let e = event(&mut ws).await;
        assert_eq!(e["type"], "diarization.error");
        assert_eq!(e["sequence_number"], 1);
        assert!(matches!(
            ws.next().await.unwrap().unwrap(),
            Message::Close(_)
        ));
    }
    assert!(
        s.drain
            .wait_drained(std::time::Duration::from_secs(2))
            .await
    );
    assert_eq!(
        s.diarization
            .as_ref()
            .unwrap()
            .admission
            .available_permits(),
        2
    );
}

#[tokio::test]
async fn live_frame_limit_closes_before_inference_and_reclaims_session() {
    let s = state();
    let server = serve(s.clone()).await;
    let mut ws = connect(&server.url).await.unwrap();
    event(&mut ws).await;
    // One sample over the announced maximum; the WebSocket implementation
    // rejects this frame before PCM is copied or handed to the device.
    ws.send(Message::Binary(vec![0u8; 32002].into()))
        .await
        .unwrap();
    let end = tokio::time::timeout(std::time::Duration::from_secs(2), ws.next())
        .await
        .unwrap();
    assert!(matches!(
        end,
        None | Some(Err(_)) | Some(Ok(Message::Close(_)))
    ));
    drop(ws);
    assert!(
        s.drain
            .wait_drained(std::time::Duration::from_secs(2))
            .await
    );
    assert_eq!(
        s.diarization
            .as_ref()
            .unwrap()
            .admission
            .available_permits(),
        2
    );
}
