//! OpenAI-compatible speech generation through Paddock's own Metal and CUDA
//! graphs.
use crate::routes::AppState;
use axum::{
    Json,
    body::{Body, Bytes},
    extract::State,
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};
use paddock_engine::phonon::{MAX_IN_FLIGHT, Options, Output, Speech};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{path::Path, sync::Arc};

pub struct SpeechModel {
    pub id: String,
    pub service: Speech,
    admission: Arc<tokio::sync::Semaphore>,
}
impl SpeechModel {
    pub fn capabilities(&self) -> Value {
        json!({"speech_generation":true,"audio_generation":true,"stream":true,"sample_rate":24000,"channels":1,
        "voices":["default"],"languages":["en"],"response_formats":["wav","pcm"],
        "max_input_bytes":4096,"max_text_tokens":512,"max_audio_seconds":60,
        "voice_cloning":false,"speed_control":false,"max_in_flight_requests":MAX_IN_FLIGHT})
    }
}
pub fn load(
    id: String,
    path: &Path,
    device: &str,
    gpu: usize,
    pack: Option<&Path>,
    budget: Option<u64>,
) -> Result<SpeechModel, String> {
    if device == "cuda" {
        let (path, pack) = (path.to_owned(), pack.map(Path::to_path_buf));
        let service = Speech::spawn(move || {
            let exec = paddock_engine::gpu::GpuExecutor::with_pack(gpu, pack.as_deref())
                .map_err(|e| e.to_string())?;
            crate::serving::note_device_cc(&exec);
            if let Some(b) = budget {
                exec.set_vram_budget(b);
            }
            let m = paddock_engine::gpu_model::phonon::GpuPhonon::load(Arc::new(exec), &path)
                .map_err(|e| e.to_string())?;
            tracing::info!(
                weight_bytes = m.weight_bytes(),
                workspace_bytes = m.workspace_bytes(),
                "Phonon loaded on cuda"
            );
            Ok(m)
        })?;
        return Ok(SpeechModel {
            id,
            service,
            admission: Arc::new(tokio::sync::Semaphore::new(MAX_IN_FLIGHT)),
        });
    }
    #[cfg(all(feature = "metal", target_os = "macos"))]
    if device == "metal" {
        let path = path.to_owned();
        let service = Speech::spawn(move || {
            paddock_metal::Phonon::load(&path, budget).map_err(|e| e.to_string())
        })?;
        return Ok(SpeechModel {
            id,
            service,
            admission: Arc::new(tokio::sync::Semaphore::new(MAX_IN_FLIGHT)),
        });
    }
    let _ = (id, path, budget);
    Err(format!(
        "Phonon needs the native cuda or metal backend; {device:?} is not supported"
    ))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    model: String,
    input: String,
    #[serde(default = "default_voice")]
    voice: String,
    #[serde(default = "default_format")]
    response_format: String,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    seed: u64,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_speed")]
    speed: f32,
}
fn default_voice() -> String {
    "default".into()
}
fn default_format() -> String {
    "wav".into()
}
fn default_temperature() -> f32 {
    0.3
}
fn default_speed() -> f32 {
    1.
}
fn error(status: StatusCode, message: impl Into<String>) -> Response {
    let mut r = (
        status,
        Json(paddock_api::ErrorBody::new("speech_error", message)),
    )
        .into_response();
    if status == StatusCode::TOO_MANY_REQUESTS {
        r.headers_mut().insert(
            header::RETRY_AFTER,
            axum::http::HeaderValue::from_static("1"),
        );
    }
    r
}
fn validate(r: &Request, id: &str) -> Result<(), String> {
    if r.model != id {
        return Err("requested model is not served by this endpoint".into());
    }
    if r.voice != "default" {
        return Err(
            "this checkpoint serves voice \"default\"; voice cloning is not supported".into(),
        );
    }
    if r.speed != 1. {
        return Err("Phonon supports speed=1 only".into());
    }
    if !matches!(r.response_format.as_str(), "wav" | "pcm") {
        return Err("response_format must be wav or pcm".into());
    }
    if r.stream && r.response_format != "pcm" {
        return Err(
            "streaming requires response_format=pcm (24 kHz mono signed 16-bit little endian)"
                .into(),
        );
    }
    paddock_engine::phonon::validate_text(&r.input)?;
    Options {
        seed: r.seed,
        temperature: r.temperature,
    }
    .validate()
}
pub async fn handle(
    State(state): State<Arc<AppState>>,
    body: Result<Json<Request>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let r = match body {
        Ok(Json(r)) => r,
        Err(e) => {
            return error(
                if e.status() == StatusCode::PAYLOAD_TOO_LARGE {
                    e.status()
                } else {
                    StatusCode::BAD_REQUEST
                },
                e.body_text(),
            );
        }
    };
    let Some(model) = &state.speech else {
        return error(
            StatusCode::BAD_REQUEST,
            "this endpoint does not serve speech generation",
        );
    };
    if let Err(e) = validate(&r, &model.id) {
        return error(StatusCode::BAD_REQUEST, e);
    }
    let Ok(permit) = model.admission.clone().try_acquire_owned() else {
        return error(
            StatusCode::TOO_MANY_REQUESTS,
            "speech worker is busy; retry shortly",
        );
    };
    let mut audio = match model.service.start(
        r.input,
        Options {
            seed: r.seed,
            temperature: r.temperature,
        },
    ) {
        Ok(a) => a,
        Err(e) => {
            use paddock_engine::phonon::StartError;
            let code = match &e {
                StartError::Invalid(_) => StatusCode::BAD_REQUEST,
                StartError::Busy => StatusCode::TOO_MANY_REQUESTS,
                StartError::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
            };
            return error(code, e.to_string());
        }
    };
    // Wait for the first frame before sending success headers. Checkpoint,
    // tokenization and GPU failures remain structured API errors.
    let first = match audio.recv().await {
        Some(Ok(Output::Audio(a))) => a,
        Some(Err(e)) => return error(StatusCode::UNPROCESSABLE_ENTITY, e),
        None | Some(Ok(Output::Finished)) => {
            return error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "speech worker stopped without audio",
            );
        }
    };
    let first = match paddock_engine::phonon::pcm16(&first) {
        Ok(p) => p,
        Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, e),
    };
    if r.stream {
        let stream = async_stream::stream! {
            let _permit=permit;
            yield Ok::<Bytes,std::io::Error>(Bytes::from(first));
            loop {
                let pcm=match audio.recv().await {
                    Some(Ok(Output::Audio(pcm)))=>pcm,
                    Some(Ok(Output::Finished))=>break,
                    Some(Err(e))=>{yield Err(std::io::Error::other(e));break;},
                    None=>{yield Err(std::io::Error::other("speech worker stopped before completion"));break;},
                };
                match paddock_engine::phonon::pcm16(&pcm) {Ok(bytes)=>yield Ok(Bytes::from(bytes)),Err(e)=>{yield Err(std::io::Error::other(e));break;}}
            }
        };
        return (
            [
                (header::CONTENT_TYPE, "audio/pcm"),
                (header::CACHE_CONTROL, "no-store"),
            ],
            Body::from_stream(stream),
        )
            .into_response();
    }
    let _permit = permit;
    let mut pcm = first;
    loop {
        let chunk = match audio.recv().await {
            Some(Ok(Output::Audio(c))) => c,
            Some(Ok(Output::Finished)) => break,
            Some(Err(e)) => return error(StatusCode::UNPROCESSABLE_ENTITY, e),
            None => {
                return error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "speech worker stopped before completion",
                );
            }
        };
        match paddock_engine::phonon::pcm16(&chunk) {
            Ok(c) => pcm.extend(c),
            Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, e),
        }
    }
    if r.response_format == "wav" {
        match paddock_engine::phonon::wav(&pcm) {
            Ok(w) => (
                [
                    (header::CONTENT_TYPE, "audio/wav"),
                    (header::CACHE_CONTROL, "no-store"),
                ],
                w,
            )
                .into_response(),
            Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, e),
        }
    } else {
        (
            [
                (header::CONTENT_TYPE, "audio/pcm"),
                (header::CACHE_CONTROL, "no-store"),
            ],
            pcm,
        )
            .into_response()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn no_ignored_speech_controls() {
        let mut r: Request =
            serde_json::from_value(json!({"model":"phonon","input":"Hello"})).unwrap();
        assert!(validate(&r, "phonon").is_ok());
        r.voice = "alloy".into();
        assert!(validate(&r, "phonon").is_err());
        r.voice = "default".into();
        r.stream = true;
        assert!(validate(&r, "phonon").is_err());
        r.response_format = "pcm".into();
        assert!(validate(&r, "phonon").is_ok());
        assert!(
            serde_json::from_value::<Request>(
                json!({"model":"phonon","input":"Hello","instructions":"whisper"})
            )
            .is_err()
        );
    }
}
