//! Whisper adapter for the platform-neutral runner residency controller.
use crate::{
    config::Config,
    residency::{Lease, LoadPolicy, Pool},
    serving::{AsrModel, ServeError},
};
use paddock_engine::{
    audio::MelFeatures,
    transcriber::{LanguageAsk, Progress, Transcriber, Transcript},
};
use paddock_models::{gguf::Value, mapped::MappedGguf};
use std::{
    path::Path,
    sync::{Arc, atomic::Ordering::Relaxed},
};

#[derive(Clone)]
pub enum Handle {
    Loaded(Transcriber),
    Resident(Pool<Transcriber>),
}
impl Handle {
    pub fn residency(&self) -> Option<crate::residency::Snapshot> {
        match self {
            Self::Resident(pool) => Some(pool.snapshot()),
            _ => None,
        }
    }
    pub async fn session_lease(&self) -> Result<Option<Lease<Transcriber>>, String> {
        match self {
            Self::Resident(pool) => pool.acquire().await.map(Some),
            _ => Ok(None),
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub async fn transcribe(
        &self,
        windows: Vec<MelFeatures>,
        speech: Vec<bool>,
        language: LanguageAsk,
        prompt: Vec<u32>,
        timestamps: bool,
        words: bool,
        max_tokens: usize,
        progress: Option<tokio::sync::mpsc::UnboundedSender<Progress>>,
    ) -> Result<Transcript, String> {
        let lease = self.session_lease().await?;
        let worker = match (&lease, self) {
            (Some(worker), _) => &**worker,
            (_, Self::Loaded(worker)) => worker,
            _ => unreachable!("resident worker requires a lease"),
        };
        worker
            .transcribe(
                windows, speech, language, prompt, timestamps, words, max_tokens, progress,
            )
            .await
    }
}

pub fn configure(
    cfg: &Config,
    id: String,
    path: &Path,
    gpu: usize,
    config_path: Option<&str>,
) -> Result<AsrModel, ServeError> {
    let map = MappedGguf::open(path).map_err(|e| ServeError::Open(path.into(), e.to_string()))?;
    let reservation = if cfg.device == "cuda" {
        paddock_engine::gpu_model::whisper::GpuWhisper::residency_bytes(
            &map,
            cfg.max_ctx,
            cfg.max_batch,
        )
        .map_err(ServeError::Engine)?
    } else {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            if cfg.device != "metal" {
                return Err(ServeError::Engine("Whisper requires CUDA or Metal".into()));
            }
            paddock_metal::Whisper::residency_bytes(&map, cfg.max_ctx, cfg.max_batch)
                .map_err(|e| ServeError::Engine(e.to_string()))?
        }
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        {
            return Err(ServeError::Engine(
                "This build requires CUDA for Whisper".into(),
            ));
        }
    };
    let tokenizer = Arc::new(
        paddock_tokenizer::GgufTokenizer::from_gguf(map.gguf())
            .map_err(|e| ServeError::Tokenizer(e.to_string()))?,
    );
    let u = |key: &str| {
        map.gguf()
            .arch_field(key)
            .and_then(Value::as_u64)
            .ok_or_else(|| ServeError::Engine(format!("Whisper residency: missing {key}")))
    };
    let language_json = map
        .gguf()
        .arch_field("lang_to_id_json")
        .and_then(Value::as_str)
        .ok_or_else(|| ServeError::Engine("Whisper residency: missing language map".into()))?;
    let entries: std::collections::BTreeMap<String, u32> =
        serde_json::from_str(language_json).map_err(|e| ServeError::Engine(e.to_string()))?;
    if entries.is_empty() {
        return Err(ServeError::Engine("Empty Whisper language map".into()));
    }
    let languages: Vec<String> = entries
        .keys()
        .map(|s| s.trim_start_matches("<|").trim_end_matches("|>").to_owned())
        .collect();
    let rate = u("mel.sampling_rate")?;
    if rate == 0 {
        return Err(ServeError::Engine("Invalid Whisper sampling rate".into()));
    }
    let scale = paddock_engine::whisper::TimeScale {
        begin: map
            .gguf()
            .arch_field("token.timestamp_begin")
            .and_then(Value::as_u64)
            .unwrap_or(u("token.no_timestamps")? + 1)
            .try_into()
            .map_err(|_| ServeError::Engine("Invalid timestamp token".into()))?,
        precision: 2.0 * u("mel.hop_length")? as f32 / rate as f32,
        window_s: u("mel.chunk_length_s")? as f32,
    };
    let metrics = Arc::new(paddock_engine::metrics::EngineMetrics::default());
    let original = std::fs::metadata(path).map_err(|e| ServeError::Engine(e.to_string()))?;
    let signature = (original.len(), original.modified().ok());
    let path = path.to_owned();
    let device = cfg.device.clone();
    let pack = cfg.kernel_pack.clone();
    let max_ctx = cfg.max_ctx;
    let max_batch = cfg.max_batch;
    let budget = cfg.vram_budget.map(|mib| mib << 20);
    let model_id = id.clone();
    let mut expected_languages = languages.clone();
    expected_languages.sort();
    let loaded_metrics = metrics.clone();
    let loaded_tokenizer = tokenizer.clone();
    let build = move || {
        // Serialize cold loads among cooperating runners. Lock acquisition has
        // a deadline; dropping the file releases it even after a failed load.
        let current = std::fs::metadata(&path).map_err(|e| e.to_string())?;
        if (current.len(), current.modified().ok()) != signature {
            return Err(
                "model_changed: checkpoint changed; restart the runner to review its new identity"
                    .into(),
            );
        }
        let model = crate::serving::load_asr_reusing(
            model_id.clone(),
            &path,
            &device,
            gpu,
            pack.as_deref(),
            max_ctx,
            max_batch,
            budget,
            loaded_tokenizer.clone(),
        )
        .map_err(|e| format!("model_load_failed: {e}"))?;
        let mut actual_languages = model.languages.clone();
        actual_languages.sort();
        let unchanged = std::fs::metadata(&path)
            .ok()
            .is_some_and(|m| (m.len(), m.modified().ok()) == signature);
        if !unchanged
            || actual_languages != expected_languages
            || model.time_scale.begin != scale.begin
            || model.time_scale.precision != scale.precision
            || model.time_scale.window_s != scale.window_s
            || !model.word_times
        {
            if let Handle::Loaded(worker) = model.transcriber {
                worker.close_and_wait();
            }
            return Err(
                "model_changed: loaded Whisper capabilities do not match its advertised metadata"
                    .into(),
            );
        }
        loaded_metrics
            .weights_mem_bytes
            .store(model.metrics.weights_mem_bytes.load(Relaxed), Relaxed);
        loaded_metrics
            .model_mem_bytes
            .store(model.metrics.model_mem_bytes.load(Relaxed), Relaxed);
        match model.transcriber {
            Handle::Loaded(worker) => Ok(worker),
            _ => unreachable!(),
        }
    };
    let initial = if cfg.residency.load == LoadPolicy::AtStartup {
        Some(build().map_err(ServeError::Engine)?)
    } else {
        None
    };
    let cleared_metrics = metrics.clone();
    let pool = Pool::new(cfg.residency.clone(), initial, build, move |worker| {
        worker.close_and_wait();
        cleared_metrics.weights_mem_bytes.store(0, Relaxed);
        cleared_metrics.model_mem_bytes.store(0, Relaxed);
    });
    if let Some(path) = config_path {
        pool.watch_policy(path.into());
    }
    Ok(AsrModel {
        residency_budget: Some(reservation),
        id,
        transcriber: Handle::Resident(pool),
        tokenizer,
        metrics,
        word_times: true,
        languages,
        time_scale: scale,
        max_tokens: max_ctx.min(448).saturating_sub(8).max(16),
    })
}

pub(crate) fn error_response(error: String) -> axum::response::Response {
    use axum::{
        Json,
        http::{StatusCode, header},
        response::IntoResponse,
    };
    let permanent = error.contains("model_budget_exceeded:")
        || error.contains("configured VRAM budget")
        || error.contains("model_changed:");
    // Only the residency's own refusals are temporary: the queue, a closing or
    // failed worker, a cold load still in flight. Anything else is the decode
    // failing on this request, which a retry repeats - advertising it as 503
    // with Retry-After cost one client three retries per call.
    let temporary = error.contains("model_unavailable")
        || error.contains("model_load_timeout:")
        || error.contains("insufficient_memory");
    let (status, code) = if permanent {
        (StatusCode::BAD_REQUEST, "model_configuration_error")
    } else if error.contains("model_load_timeout:") {
        (StatusCode::SERVICE_UNAVAILABLE, "model_load_timeout")
    } else if temporary {
        (StatusCode::SERVICE_UNAVAILABLE, "model_unavailable")
    } else {
        (StatusCode::INTERNAL_SERVER_ERROR, "internal_error")
    };
    let mut response = (
        status,
        Json(serde_json::json!({"error":{"type":code,"code":code,"message":error}})),
    )
        .into_response();
    if status == StatusCode::SERVICE_UNAVAILABLE {
        response.headers_mut().insert(
            header::RETRY_AFTER,
            axum::http::HeaderValue::from_static("2"),
        );
    }
    response
}
