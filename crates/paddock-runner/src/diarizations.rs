//! Speaker activity, not speech-to-text. No transcript or named-person claims.
use crate::routes::AppState;
use axum::{
    Json,
    extract::{Multipart, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use paddock_engine::diarization::{Diarizer, Error};
use paddock_models::diarization::Preset;
use paddock_models::diarization::words::{self, Word};
use serde_json::{Value, json};
use std::{path::Path, sync::Arc, time::Instant};

pub struct DiarizationModel {
    pub id: String,
    pub service: Diarizer,
    pub admission: Arc<tokio::sync::Semaphore>,
}
impl DiarizationModel {
    pub fn capabilities(&self) -> Value {
        json!({"speaker_diarization":true,"max_speakers":8,"overlap":true,"frame_seconds":0.01,
        "sample_rate":16000,"max_duration_seconds":600,"transcription":false,"stream":false,
        "latency_presets":["offline","low","very_low","ultra_low"],
        "realtime_diarization":{"transport":"websocket","path":"/v1/audio/diarizations/stream",
            "audio_format":"pcm_s16le","sample_rate":16000,"channels":1,"max_chunk_samples":16000}})
    }
}
pub fn is_model(path: &Path, arch: &str) -> bool {
    arch == "sortformer"
        || (path.is_dir()
            && std::fs::read(path.join("config.json"))
                .ok()
                .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
                .is_some_and(|v| v["model_type"] == "nemotron_diarization"))
}
pub fn load(
    id: String,
    path: &Path,
    device: &str,
    gpu: usize,
    pack: Option<&Path>,
    budget: Option<u64>,
) -> Result<DiarizationModel, String> {
    let model = |service| DiarizationModel {
        id,
        service,
        admission: Arc::new(tokio::sync::Semaphore::new(2)),
    };
    #[cfg(all(feature = "metal", target_os = "macos"))]
    if device == "metal" {
        let path = path.to_owned();
        let service = Diarizer::spawn(move || {
            paddock_metal::Diarization::load(&path, budget).map_err(|e| e.to_string())
        })?;
        return Ok(model(service));
    }
    if device != "cuda" {
        return Err(format!(
            "Nemotron 3 Diarization needs the native cuda or metal backend (got {device:?})"
        ));
    }
    let (path, pack) = (path.to_owned(), pack.map(Path::to_path_buf));
    let service = Diarizer::spawn(move || {
        use paddock_engine::diarization::Backend;
        let exec = paddock_engine::gpu::GpuExecutor::with_pack(gpu, pack.as_deref())
            .map_err(|e| e.to_string())?;
        crate::serving::note_device_cc(&exec);
        if let Some(b) = budget {
            exec.set_vram_budget(b);
        }
        let mut m =
            paddock_engine::gpu_model::diarization::GpuDiarization::load(Arc::new(exec), &path)
                .map_err(|e| e.to_string())?;
        // One short window before the first request: every kernel's first
        // launch pays its module load, and that belongs to startup.
        let fe = m.frontend()?;
        let mut warm = paddock_engine::diarization::Stream::new(Preset::UltraLow);
        warm.feed(&mut m, &fe, &[0.; 4000], true)
            .map_err(|e| format!("Nemotron 3 Diarization warm-up: {e}"))?;
        tracing::info!(
            format = ?m.format(),
            weight_bytes = m.weight_bytes(),
            workspace_bytes = Backend::workspace_bytes(&m),
            "Nemotron 3 Diarization loaded on cuda"
        );
        Ok(m)
    })?;
    Ok(model(service))
}
pub(crate) fn err(code: StatusCode, message: impl Into<String>) -> Response {
    let mut response = (
        code,
        Json(paddock_api::ErrorBody::new("diarization_error", message)),
    )
        .into_response();
    if code == StatusCode::TOO_MANY_REQUESTS || code == StatusCode::SERVICE_UNAVAILABLE {
        response
            .headers_mut()
            .insert("retry-after", axum::http::HeaderValue::from_static("1"));
    }
    response
}
pub(crate) fn service_error(e: Error) -> Response {
    let status = match &e {
        Error::Invalid(_) => StatusCode::BAD_REQUEST,
        Error::Busy => StatusCode::TOO_MANY_REQUESTS,
        Error::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
        Error::Backend(_) => StatusCode::INTERNAL_SERVER_ERROR,
    };
    err(status, e.to_string())
}
pub async fn handle(State(state): State<Arc<AppState>>, mut form: Multipart) -> Response {
    let Some(m) = &state.diarization else {
        return err(
            StatusCode::BAD_REQUEST,
            "This runner does not serve speaker diarization",
        );
    };
    // Admit BEFORE consuming/decompressing audio; concurrent uploads cannot
    // grow an unbounded queue of decoded PCM behind the owning GPU thread.
    let Ok(_permit) = m.admission.clone().try_acquire_owned() else {
        return err(
            StatusCode::TOO_MANY_REQUESTS,
            "Diarization is busy; retry shortly",
        );
    };
    let started = Instant::now();
    let mut file = None;
    let mut model = None;
    let mut preset = Preset::Offline;
    let mut threshold = 0.5;
    let mut words: Option<Vec<Word>> = None;
    let mut seen = std::collections::HashSet::new();
    loop {
        let field = match form.next_field().await {
            Ok(Some(f)) => f,
            Ok(None) => break,
            Err(e) => return err(e.status(), e.to_string()),
        };
        let name = field.name().unwrap_or("").to_owned();
        if !seen.insert(name.clone()) {
            return err(StatusCode::BAD_REQUEST, format!("duplicate field {name}"));
        }
        match name.as_str() {
            "words" => {
                let bytes = match field.bytes().await {
                    Ok(b) if b.len() <= 2 * 1024 * 1024 => b,
                    _ => {
                        return err(
                            StatusCode::BAD_REQUEST,
                            "words must be a JSON array of at most 2 MiB",
                        );
                    }
                };
                words = match serde_json::from_slice::<Vec<Word>>(&bytes) {
                    Ok(w) if w.len() <= words::MAX_WORDS => Some(w),
                    _ => {
                        return err(
                            StatusCode::BAD_REQUEST,
                            "invalid words; at most 20000 word/start/end records",
                        );
                    }
                };
            }
            "file" => match field.bytes().await {
                Ok(b) => file = Some(b),
                Err(e) => return err(e.status(), e.to_string()),
            },
            "model" | "preset" | "threshold" => {
                let text = match field.text().await {
                    Ok(s) if s.len() <= 256 => s,
                    _ => return err(StatusCode::BAD_REQUEST, "invalid field value"),
                };
                match name.as_str() {
                    "model" => model = Some(text),
                    "preset" => {
                        preset = match text.as_str() {
                            "offline" => Preset::Offline,
                            "low" => Preset::Low,
                            "very_low" => Preset::VeryLow,
                            "ultra_low" => Preset::UltraLow,
                            _ => {
                                return err(
                                    StatusCode::BAD_REQUEST,
                                    "preset must be offline, low, very_low or ultra_low",
                                );
                            }
                        };
                    }
                    _ => {
                        threshold = match text.parse::<f32>() {
                            Ok(t) if t.is_finite() && (0.0..1.0).contains(&t) => t,
                            _ => return err(StatusCode::BAD_REQUEST, "threshold must be in [0,1)"),
                        };
                    }
                }
            }
            _ => {
                return err(
                    StatusCode::BAD_REQUEST,
                    format!(
                        "unknown field {name}; accepted: file, model, preset, threshold, words"
                    ),
                );
            }
        }
    }
    if model.as_ref().is_some_and(|id| id != &m.id) {
        return err(
            StatusCode::NOT_FOUND,
            "Requested diarization model is not served by this endpoint",
        );
    }
    let Some(file) = file else {
        return err(StatusCode::BAD_REQUEST, "file is required");
    };
    let pcm = match tokio::task::spawn_blocking(move || {
        let audio = paddock_engine::audio::decode::decode_audio_limited(&file, 600)?;
        if audio.samples.iter().any(|v| !v.is_finite()) {
            return Err("audio contains nonfinite samples".to_owned());
        }
        paddock_engine::audio::resample::resample(&audio.samples, audio.sample_rate, 16000)
    })
    .await
    {
        Ok(Ok(p)) => p,
        Ok(Err(e)) => return err(StatusCode::BAD_REQUEST, e),
        Err(_) => return err(StatusCode::INTERNAL_SERVER_ERROR, "Audio decoding failed"),
    };
    let duration = pcm.len() as f64 / 16000.;
    if let Some(w) = &words
        && let Err(e) = words::validate(w, duration)
    {
        return err(StatusCode::BAD_REQUEST, e);
    }
    match m.service.diarize(pcm, preset, threshold).await {
        Ok(out) => {
            let speakers: std::collections::HashSet<_> =
                out.segments.iter().map(|s| s.speaker).collect();
            let attributed = match words
                .map(|w| words::attribute(w, &out.segments, duration))
                .transpose()
            {
                Ok(w) => w,
                Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, e),
            };
            Json(json!({
                "model": m.id, "duration": duration, "frame_seconds": 0.01,
                "frames": out.frames, "preset": preset, "num_speakers": speakers.len(),
                "segments": out.segments, "processing_seconds": started.elapsed().as_secs_f64(),
                "gpu_seconds": out.gpu_seconds, "words": attributed,
                "attribution": "time_overlap_v1"
            }))
            .into_response()
        }
        Err(e) => service_error(e),
    }
}
