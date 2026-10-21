//! Nemotron 3.5 ASR - NVIDIA's cache-aware streaming RNN-T recognizer -
//! behind `POST /v1/audio/transcriptions` (and the realtime session).
//!
//! The model streams natively: its encoder attends 56 frames back and a
//! chunk forward, its convolutions are causal, and an RNN-T emits each token
//! once, at the frame that produced it. So a file and a live stream are the
//! same computation (the engine proves it bit for bit) and a delta stream is
//! simply the tokens in the order they were emitted - nothing provisional,
//! nothing to retract, no agreement policy between passes.
//!
//! Wire choices, each one stated where a caller would trip over it:
//!   - `language` selects the checkpoint's language prompt: a bare code
//!     (`de`) or a locale (`de-DE`), from the prompt dictionary the file
//!     carries; absent = the model's own detection (`auto`). `languages`
//!     (a soft prior) has nothing to bias here and is refused.
//!   - `prompt` is refused: the model takes no text context.
//!   - `temperature` above 0 is refused: the decoder is greedy.
//!   - segments are the model's own sentences (it closes each with a
//!     language tag), words are its pieces joined at word boundaries, both
//!     timed by the 80 ms frames their tokens were emitted at.
//!   - `paddock_latency_ms` (80 | 160 | 320 | 560 | 1120) picks the
//!     attention chunk; a file defaults to 1120, the most accurate published
//!     point (latency is no constraint on a file).

use std::path::Path;
use std::sync::Arc;

use axum::extract::{Multipart, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use paddock_api::ErrorBody;
use paddock_engine::audio::{decode::decode_audio, resample::resample};
use paddock_engine::nemotron_asr::{Error, Recognizer, Update};
use paddock_models::nemotron_asr::{self as arch, Sentence, Word};
use serde_json::{Value, json};

use crate::routes::AppState;

/// Concurrent streams (files and live sessions) one worker admits. Each
/// holds ~12 MB of device state (the K/V and convolution rings).
const MAX_SESSIONS: usize = 32;
/// The longest file accepted: four hours (the session keeps only the PCM
/// its next window reads; the decoded upload is the bound that remains).
const MAX_SECONDS: usize = 4 * 3600;
/// Audio a streamed file feeds per step: two seconds of deltas at a time.
const STREAM_STEP: usize = 2 * arch::SAMPLE_RATE;

pub struct RnntAsrModel {
    pub id: String,
    pub recognizer: Recognizer,
    /// the prompt dictionary's keys (bare codes and locales), `auto` aside
    pub languages: Vec<String>,
}

/// The attention chunks the model publishes, in ms, with their lookahead.
pub fn latency_lookahead(ms: u32) -> Option<usize> {
    arch::LOOKAHEADS
        .iter()
        .copied()
        .find(|&r| arch::chunk_frames(r) as f64 * arch::FRAME_SECONDS * 1000.0 == ms as f64)
}

fn latencies_ms() -> Vec<u32> {
    arch::LOOKAHEADS
        .iter()
        .map(|&r| (arch::chunk_frames(r) as f64 * arch::FRAME_SECONDS * 1000.0).round() as u32)
        .collect()
}

impl RnntAsrModel {
    pub fn capabilities(&self) -> Value {
        json!({
            "transcription": true,
            "translation": false,
            "stream": true,
            "streaming": "native",
            "timestamp_granularities": ["segment", "word"],
            "language_detection": {"auto": true, "hints": false, "languages": self.languages},
            "latency_ms": latencies_ms(),
            "default_latency_ms": {"file": 1120, "realtime": 320},
            "frame_seconds": arch::FRAME_SECONDS,
            "sample_rate": arch::SAMPLE_RATE,
            "max_duration_seconds": MAX_SECONDS,
            "realtime_transcription": {"transport": "websocket", "path": "/v1/realtime?intent=transcription"},
        })
    }

    /// The prompt for a request's `language`, or a refusal naming the set.
    pub fn prompt_for(&self, language: Option<&str>) -> Result<u32, String> {
        let key = language.unwrap_or(arch::AUTO_PROMPT);
        let offered = key == arch::AUTO_PROMPT || self.languages.iter().any(|l| l == key);
        self.recognizer
            .prompt(key)
            .filter(|_| offered)
            .ok_or_else(|| {
                format!(
                    "language '{key}' is not in this checkpoint's prompt dictionary ({} codes; the \
                 list is on /v1/models as `language_detection.languages`)",
                    self.languages.len()
                )
            })
    }
}

pub fn is_model(arch: &str) -> bool {
    arch == "asr"
}

pub fn load(
    id: String,
    path: &Path,
    device: &str,
    gpu: usize,
    pack: Option<&Path>,
    budget: Option<u64>,
) -> Result<RnntAsrModel, String> {
    if device != "cuda" {
        return Err(format!(
            "Nemotron 3.5 ASR needs the native cuda backend (got {device:?})"
        ));
    }
    let (path, pack) = (path.to_owned(), pack.map(Path::to_path_buf));
    let recognizer = Recognizer::spawn(
        move || {
            let exec = paddock_engine::gpu::GpuExecutor::with_pack(gpu, pack.as_deref())
                .map_err(|e| e.to_string())?;
            crate::serving::note_device_cc(&exec);
            if let Some(b) = budget {
                exec.set_vram_budget(b);
            }
            let m = paddock_engine::gpu_model::nemotron_asr::GpuNemotronAsr::load(
                Arc::new(exec),
                &path,
            )
            .map_err(|e| e.to_string())?;
            tracing::info!(
                weight_bytes = m.weight_bytes(),
                workspace_bytes = m.workspace_bytes(),
                "Nemotron 3.5 ASR loaded on cuda"
            );
            Ok(m)
        },
        MAX_SESSIONS,
    )?;
    // the languages the checkpoint can actually write: its dictionary has 128
    // prompt slots, but only the locales its vocabulary has a tag for were
    // trained (the model card's 40) - a key is offered when its prompt is
    // one of those locales' (or their bare code's)
    let facts = recognizer.facts();
    let trained: Vec<u32> = facts
        .vocab
        .iter()
        .filter_map(|p| arch::tag_language(p))
        .filter_map(|loc| {
            let bare = loc.split('-').next().unwrap_or(loc);
            recognizer.prompt(loc).or_else(|| recognizer.prompt(bare))
        })
        .collect();
    let mut languages: Vec<String> = facts
        .prompts
        .iter()
        .filter(|(k, id)| k != arch::AUTO_PROMPT && trained.contains(id))
        .map(|(k, _)| k.clone())
        .collect();
    languages.sort();
    languages.dedup();
    Ok(RnntAsrModel {
        id,
        recognizer,
        languages,
    })
}

pub(crate) fn err(status: StatusCode, kind: &str, msg: impl Into<String>) -> Response {
    let mut r = (status, Json(ErrorBody::new(kind, msg))).into_response();
    if status == StatusCode::TOO_MANY_REQUESTS || status == StatusCode::SERVICE_UNAVAILABLE {
        r.headers_mut()
            .insert("retry-after", axum::http::HeaderValue::from_static("1"));
    }
    r
}

pub(crate) fn service_error(e: Error) -> Response {
    match e {
        Error::Invalid(m) => err(StatusCode::BAD_REQUEST, "invalid_request_error", m),
        Error::Busy => err(
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limit_error",
            Error::Busy.to_string(),
        ),
        Error::Unavailable => err(
            StatusCode::SERVICE_UNAVAILABLE,
            "server_error",
            Error::Unavailable.to_string(),
        ),
        Error::Backend(m) => err(StatusCode::INTERNAL_SERVER_ERROR, "server_error", m),
    }
}

/// A transcript as the wire shows it.
pub(crate) struct Transcript {
    pub text: String,
    pub words: Vec<Word>,
    pub sentences: Vec<Sentence>,
    /// what the model said it heard: the most frequent sentence tag
    pub language: Option<String>,
}

pub(crate) fn transcript(vocab: &[String], tokens: &[u32], frames: &[u32]) -> Transcript {
    let (words, sentences) = arch::render(vocab, tokens, frames);
    let mut counts: Vec<(&str, usize)> = Vec::new();
    for s in &sentences {
        if let Some(l) = s.language.as_deref() {
            match counts.iter_mut().find(|(k, _)| *k == l) {
                Some((_, n)) => *n += 1,
                None => counts.push((l, 1)),
            }
        }
    }
    let language = counts
        .iter()
        .max_by_key(|(_, n)| *n)
        .map(|(l, _)| (*l).to_owned());
    Transcript {
        text: arch::detokenize(vocab, tokens, false),
        words,
        sentences,
        language,
    }
}

pub(crate) fn verbose(t: &Transcript, duration: f64, segments: bool, words: bool) -> Value {
    let mut body = json!({
        "task": "transcribe",
        "language": t.language.as_deref().unwrap_or("unknown"),
        "duration": duration,
        "text": t.text,
    });
    if segments {
        body["segments"] = Value::Array(
            t.sentences
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    json!({
                        "id": i,
                        "seek": 0,
                        "start": s.start,
                        "end": s.end.min(duration),
                        "text": s.text,
                        "tokens": [],
                        "temperature": 0.0,
                        "avg_logprob": 0.0,
                        "compression_ratio": 0.0,
                        "no_speech_prob": 0.0,
                        "paddock_language": s.language,
                    })
                })
                .collect(),
        );
    }
    if words {
        body["words"] = Value::Array(
            t.words
                .iter()
                .map(|w| json!({"word": w.text, "start": w.start, "end": w.end.min(duration)}))
                .collect(),
        );
    }
    body
}

fn cues(t: &Transcript, duration: f64) -> Vec<crate::subtitles::Cue<'_>> {
    t.sentences
        .iter()
        .map(|s| crate::subtitles::Cue {
            start: s.start,
            end: s.end.min(duration),
            text: &s.text,
        })
        .collect()
}

fn usage(frames: usize, tokens: usize) -> Value {
    json!({
        "type": "tokens",
        "input_tokens": frames,
        "output_tokens": tokens,
        "total_tokens": frames + tokens,
        "input_token_details": {"audio_tokens": frames, "text_tokens": 0},
    })
}

/// `/v1/audio/transcriptions` on this lane.
pub async fn handle(State(state): State<Arc<AppState>>, mut mp: Multipart) -> Response {
    let Some(model) = state.rnnt_asr.as_ref() else {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "this runner does not serve the Nemotron 3.5 ASR lane",
        );
    };
    let mut file: Option<Vec<u8>> = None;
    let mut response_format = "json".to_owned();
    let mut language: Option<String> = None;
    let mut grans = (false, false); // (segment, word)
    let mut stream = false;
    let mut latency: Option<String> = None;
    let mut refusals: Vec<String> = Vec::new();
    let mut unknown: Vec<String> = Vec::new();
    loop {
        let field = match mp.next_field().await {
            Ok(Some(f)) => f,
            Ok(None) => break,
            Err(e) => {
                return err(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    e.to_string(),
                );
            }
        };
        let name = field.name().unwrap_or_default().to_owned();
        match name.trim_end_matches("[]") {
            "file" => match field.bytes().await {
                Ok(b) => file = Some(b.to_vec()),
                Err(e) => {
                    return err(
                        StatusCode::BAD_REQUEST,
                        "invalid_request_error",
                        e.to_string(),
                    );
                }
            },
            "model" => {
                let _ = field.bytes().await;
            }
            "response_format" => response_format = field.text().await.unwrap_or_default(),
            "language" => language = field.text().await.ok().filter(|s| !s.is_empty()),
            "stream" => {
                let v = field.text().await.unwrap_or_default();
                stream = matches!(v.trim(), "true" | "True" | "1");
            }
            "timestamp_granularities" => match field.text().await.unwrap_or_default().trim() {
                "segment" => grans.0 = true,
                "word" => grans.1 = true,
                "" => {}
                other => refusals.push(format!(
                    "unknown timestamp_granularities value '{other}' (segment | word)"
                )),
            },
            "paddock_latency_ms" => latency = field.text().await.ok(),
            "temperature" => {
                let t: f32 = field
                    .text()
                    .await
                    .ok()
                    .and_then(|t| t.trim().parse().ok())
                    .unwrap_or(0.0);
                if t != 0.0 {
                    refusals.push(
                        "`temperature` above 0: this model decodes greedily (an RNN-T's \
                         argmax walk) and has no sampler to temper"
                            .into(),
                    );
                }
            }
            "prompt" => {
                if !field.text().await.unwrap_or_default().trim().is_empty() {
                    refusals.push(
                        "`prompt`: this model takes no text context - its only conditioning \
                         is the language prompt, selected by `language`"
                            .into(),
                    );
                }
            }
            "languages" => {
                if !field.text().await.unwrap_or_default().trim().is_empty() {
                    refusals.push(
                        "`languages` biases a detector's posterior, and this model has none to \
                         bias - send `language` to select its language prompt, or leave it \
                         out for the model's own detection"
                            .into(),
                    );
                }
            }
            "include" => {
                let v = field.text().await.unwrap_or_default();
                if !matches!(v.trim(), "" | "false" | "null" | "[]") {
                    refusals.push(format!("`include` value {v:?} is not served on this model"));
                }
            }
            "chunking_strategy" => {
                let v = field.text().await.unwrap_or_default();
                if !matches!(v.trim().trim_matches('"'), "" | "auto" | "null") {
                    refusals.push(
                        "`chunking_strategy`: this model streams natively over any length, \
                         so there is nothing to chunk (\"auto\" is accepted)"
                            .into(),
                    );
                }
            }
            other => {
                unknown.push(other.to_owned());
                let _ = field.bytes().await;
            }
        }
    }
    if !unknown.is_empty() {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!(
                "Unrecognized request argument supplied: {}",
                unknown.join(", ")
            ),
        );
    }
    if !refusals.is_empty() {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            refusals.join("; "),
        );
    }
    let Some(file) = file else {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "missing `file` field",
        );
    };
    if !matches!(
        response_format.as_str(),
        "json" | "text" | "verbose_json" | "srt" | "vtt"
    ) {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!(
                "unsupported response_format '{response_format}' (json | text | verbose_json | srt | vtt)"
            ),
        );
    }
    let subtitles = matches!(response_format.as_str(), "srt" | "vtt");
    if stream && subtitles {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!(
                "`stream` cannot produce {response_format}; stream with verbose_json and render cues from `segments`"
            ),
        );
    }
    if (grans.0 || grans.1) && !subtitles && response_format != "verbose_json" {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "timestamp_granularities requires response_format=verbose_json",
        );
    }
    if grans.1 && subtitles {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!(
                "`word` timestamps have nowhere to go in {response_format}; ask for verbose_json"
            ),
        );
    }
    let lookahead = match latency.as_deref().map(str::trim) {
        None | Some("") => *arch::LOOKAHEADS
            .iter()
            .max()
            .unwrap_or(&arch::DEFAULT_LOOKAHEAD),
        Some(v) => match v.parse::<u32>().ok().and_then(latency_lookahead) {
            Some(r) => r,
            None => {
                return err(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    format!(
                        "paddock_latency_ms '{v}': the model publishes {:?}",
                        latencies_ms()
                    ),
                );
            }
        },
    };
    let prompt = match model.prompt_for(language.as_deref()) {
        Ok(p) => p,
        Err(e) => return err(StatusCode::BAD_REQUEST, "invalid_request_error", e),
    };
    let decoded = tokio::task::spawn_blocking(move || -> Result<Vec<f32>, String> {
        let wav = decode_audio(&file)?;
        if wav.samples.is_empty() {
            return Err("audio file holds no samples".into());
        }
        let pcm = resample(&wav.samples, wav.sample_rate, arch::SAMPLE_RATE as u32)?;
        if pcm.len() > MAX_SECONDS * arch::SAMPLE_RATE {
            return Err(format!("audio longer than {} hours", MAX_SECONDS / 3600));
        }
        Ok(pcm)
    })
    .await;
    let pcm = match decoded {
        Ok(Ok(p)) => p,
        Ok(Err(e)) => return err(StatusCode::BAD_REQUEST, "invalid_request_error", e),
        Err(e) => {
            return err(
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                e.to_string(),
            );
        }
    };
    let duration = pcm.len() as f64 / arch::SAMPLE_RATE as f64;
    let vocab = model.recognizer.facts().vocab.clone();
    let forced = language.map(|l| {
        // a bare code reports as itself; the tag the model writes is a locale
        l.trim().to_owned()
    });
    if stream {
        let session = match model.recognizer.session(lookahead, prompt).await {
            Ok(s) => s,
            Err(e) => return service_error(e),
        };
        let verbose_on = response_format == "verbose_json";
        return stream_file(session, pcm, vocab, duration, verbose_on, grans, forced)
            .into_response();
    }
    let u = match model.recognizer.transcribe(&pcm, lookahead, prompt).await {
        Ok(u) => u,
        Err(e) => return service_error(e),
    };
    let mut t = transcript(&vocab, &u.tokens, &u.frames);
    if t.language.is_none() {
        t.language = forced;
    }
    match response_format.as_str() {
        "text" => t.text.into_response(),
        "srt" => crate::subtitles::srt(&cues(&t, duration)).into_response(),
        "vtt" => crate::subtitles::vtt(&cues(&t, duration)).into_response(),
        "verbose_json" => {
            let mut b = verbose(&t, duration, grans.0 || !grans.1, grans.1);
            b["usage"] = json!({"type": "duration", "seconds": duration.ceil()});
            Json(b).into_response()
        }
        _ => Json(json!({"text": t.text, "usage": usage(u.encoded_frames, u.tokens.len())}))
            .into_response(),
    }
}

/// `stream=true`: feed the file two seconds at a time and send each step's
/// new text as a delta - the tokens in emission order, never revised.
fn stream_file(
    mut session: paddock_engine::nemotron_asr::RecognizerSession,
    pcm: Vec<f32>,
    vocab: Arc<Vec<String>>,
    duration: f64,
    verbose_on: bool,
    grans: (bool, bool),
    forced: Option<String>,
) -> axum::response::sse::Sse<
    impl futures::Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>>,
> {
    let (tx, rx) = tokio::sync::mpsc::channel::<String>(64);
    tokio::spawn(async move {
        let mut all = Update::default();
        let mut shown = String::new();
        let mut parts = pcm.chunks(STREAM_STEP).peekable();
        while let Some(part) = parts.next() {
            let last = parts.peek().is_none();
            let u = match session.feed(part.to_vec(), last).await {
                Ok(u) => u,
                Err(e) => {
                    let _ = tx
                        .send(
                            json!({"type": "error", "error": {"message": e.to_string()}})
                                .to_string(),
                        )
                        .await;
                    return;
                }
            };
            all.tokens.extend(&u.tokens);
            all.frames.extend(&u.frames);
            all.encoded_frames = u.encoded_frames;
            let text = arch::detokenize(&vocab, &all.tokens, false);
            if text.len() > shown.len() && text.starts_with(&shown) {
                let delta = text[shown.len()..].to_owned();
                shown = text;
                let ev = json!({"type": "transcript.text.delta", "delta": delta});
                if tx.send(ev.to_string()).await.is_err() {
                    return;
                }
            }
        }
        let mut t = transcript(&vocab, &all.tokens, &all.frames);
        if t.language.is_none() {
            t.language = forced;
        }
        let mut done = json!({
            "type": "transcript.text.done",
            "text": t.text,
            "usage": usage(all.encoded_frames, all.tokens.len()),
            "languages": [{"code": t.language.as_deref().unwrap_or("unknown")}],
        });
        if verbose_on {
            done["paddock_verbose"] = verbose(&t, duration, grans.0 || !grans.1, grans.1);
        }
        let _ = tx.send(done.to_string()).await;
    });
    let s = futures::stream::unfold(rx, |mut rx| async move {
        rx.recv()
            .await
            .map(|d| (Ok(axum::response::sse::Event::default().data(d)), rx))
    });
    axum::response::sse::Sse::new(s)
}
