//! Native bounded speaker timeline transport, deliberately separate from the
//! OpenAI transcription protocol. Binary messages are mono 16-kHz PCM16LE;
//! {"type":"finish"} flushes once and closes the recording. Every accepted
//! audio message receives a numbered update (including lookahead-only input).
use crate::{diarizations, routes::AppState};
use axum::{
    Json,
    extract::{
        Query, State, WebSocketUpgrade,
        ws::{Message, WebSocket},
    },
    http::StatusCode,
    response::{IntoResponse, Response},
};
use paddock_engine::diarization::Session;
use paddock_models::diarization::{Preset, SAMPLE_RATE, SegmentTracker};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{sync::Arc, time::Duration};

const MAX_MESSAGE: usize = SAMPLE_RATE * 2;
fn low() -> Preset {
    Preset::Low
}
fn half() -> f32 {
    0.5
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Options {
    model: Option<String>,
    #[serde(default = "low")]
    preset: Preset,
    #[serde(default = "half")]
    threshold: f32,
}
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum Control {
    Finish,
}

pub async fn handle(
    State(state): State<Arc<AppState>>,
    Query(opts): Query<Options>,
    ws: WebSocketUpgrade,
) -> Response {
    let Some(m) = &state.diarization else {
        return diarizations::err(
            StatusCode::BAD_REQUEST,
            "This runner does not serve speaker diarization",
        );
    };
    if opts.model.as_ref().is_some_and(|id| id != &m.id) {
        return diarizations::err(
            StatusCode::NOT_FOUND,
            "Requested diarization model is not served by this endpoint",
        );
    }
    let tracker = match SegmentTracker::new(opts.threshold) {
        Ok(v) => v,
        Err(e) => return diarizations::err(StatusCode::BAD_REQUEST, e),
    };
    let Ok(permit) = m.admission.clone().try_acquire_owned() else {
        return diarizations::err(
            StatusCode::TOO_MANY_REQUESTS,
            "Diarization is busy; retry shortly",
        );
    };
    let session = match m.service.session(opts.preset) {
        Ok(s) => s,
        Err(e) => return diarizations::service_error(e),
    };
    // The HTTP upgrade body ends immediately, but the actual work does not.
    // Register before returning so shutdown never misses an upgraded socket.
    let guard = state.drain.guard();
    if state.drain.is_draining() {
        return diarizations::err(StatusCode::SERVICE_UNAVAILABLE, "This endpoint is draining");
    }
    let model = m.id.clone();
    ws.max_message_size(MAX_MESSAGE)
        .max_frame_size(MAX_MESSAGE)
        .write_buffer_size(4096)
        .max_write_buffer_size(2 * 1024 * 1024)
        .on_upgrade(move |socket| async move {
            let (_permit, _guard) = (permit, guard);
            // Absolute session expiry also covers endless pings or slow input.
            let _ = tokio::time::timeout(
                Duration::from_secs(900),
                run(socket, session, tracker, model, opts.preset),
            )
            .await;
        })
}

async fn send(socket: &mut WebSocket, value: Value) -> Result<(), ()> {
    tokio::time::timeout(
        Duration::from_secs(10),
        socket.send(Message::Text(value.to_string().into())),
    )
    .await
    .map_err(|_| ())?
    .map_err(|_| ())
}

async fn fail(socket: &mut WebSocket, message: impl Into<String>, seq: usize) {
    let _ = send(
        socket,
        json!({"type":"diarization.error", "sequence_number":seq,
        "error":{"type":"diarization_error","message":message.into()}}),
    )
    .await;
    let _ = tokio::time::timeout(Duration::from_secs(1), socket.send(Message::Close(None))).await;
}

async fn run(
    mut socket: WebSocket,
    mut session: Session,
    mut tracker: SegmentTracker,
    model: String,
    preset: Preset,
) {
    let g = preset.geometry();
    if send(
        &mut socket,
        json!({"type":"diarization.session", "sequence_number":0, "model":model,
        "preset":preset,"audio_format":"pcm_s16le","sample_rate":16000,"channels":1,
        "max_chunk_samples":16000,"max_duration_seconds":600,"max_speakers":8,
        "frame_seconds":0.01,"input_buffer_seconds":(g.chunk+g.right) as f64 * 0.08,
        "stft_lookahead_seconds":0.006,"transcription":false}),
    )
    .await
    .is_err()
    {
        return;
    }
    let mut seq = 1;
    let mut audio_deadline = tokio::time::Instant::now() + Duration::from_secs(60);
    loop {
        let msg = match tokio::time::timeout_at(audio_deadline, socket.recv()).await {
            Ok(Some(Ok(msg))) => msg,
            Err(_) => {
                fail(&mut socket, "No audio received for 60 seconds", seq).await;
                return;
            }
            _ => return,
        };
        let (pcm, finish) = match msg {
            Message::Binary(bytes)
                if !bytes.is_empty() && bytes.len() % 2 == 0 && bytes.len() <= MAX_MESSAGE =>
            {
                (
                    bytes
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|b| i16::from_le_bytes(*b) as f32 / 32768.)
                        .collect(),
                    false,
                )
            }
            Message::Text(text)
                if matches!(serde_json::from_str::<Control>(&text), Ok(Control::Finish)) =>
            {
                (Vec::new(), true)
            }
            Message::Ping(_) | Message::Pong(_) => continue,
            Message::Close(_) => return,
            _ => {
                fail(&mut socket, "Send mono 16-kHz PCM16LE binary messages (1–16000 samples) or {\"type\":\"finish\"}", seq).await;
                return;
            }
        };
        let batch = match session.feed(pcm, finish).await {
            Ok(b) => b,
            Err(e) => {
                fail(&mut socket, e.to_string(), seq).await;
                return;
            }
        };
        audio_deadline = tokio::time::Instant::now() + Duration::from_secs(60);
        let update = match tracker.push(&batch.probabilities, finish) {
            Ok(u) => u,
            Err(e) => {
                fail(&mut socket, e, seq).await;
                return;
            }
        };
        if send(
            &mut socket,
            json!({"type":if finish {"diarization.done"} else {"diarization.update"},
            "sequence_number":seq,"received_samples":batch.received_samples,
            "frames":batch.frames,"finalized_through_seconds":batch.frames as f64 * 0.01,
            "completed":update.completed,"active":update.active,"gpu_seconds":batch.gpu_seconds}),
        )
        .await
        .is_err()
        {
            return;
        }
        seq += 1;
        if finish {
            let _ = tokio::time::timeout(Duration::from_secs(1), socket.send(Message::Close(None)))
                .await;
            return;
        }
    }
}

/// A malformed query gets a machine-readable refusal, not a framework text body.
pub async fn route(
    state: State<Arc<AppState>>,
    query: Result<Query<Options>, axum::extract::rejection::QueryRejection>,
    ws: WebSocketUpgrade,
) -> Response {
    match query {
        Ok(q) => handle(state, q, ws).await,
        Err(_) => (
            StatusCode::BAD_REQUEST,
            Json(paddock_api::ErrorBody::new(
                "diarization_error",
                "Invalid model, preset or threshold query",
            )),
        )
            .into_response(),
    }
}
