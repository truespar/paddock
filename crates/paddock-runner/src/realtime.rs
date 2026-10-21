//! `GET /v1/realtime?intent=transcription` - the OpenAI Realtime TRANSCRIPTION
//! session over WebSocket: audio arrives while it is still being spoken and
//! text comes back as it is recognised (microphone half).
//!
//! ## Why this wire and not one of our own
//!
//! Streaming a FILE is `/v1/audio/transcriptions` with `stream=true` - one
//! request, one answer, deltas along the way. A microphone is the other
//! direction too: the request has no end until the speaker stops, so it cannot
//! be a POST body. OpenAI already defines that shape, and clients already
//! speak it, so conforming costs nothing and inventing costs everyone.
//!
//! Client -> server: `session.update`, `input_audio_buffer.append`,
//! `input_audio_buffer.commit`, `input_audio_buffer.clear`, and Paddock's
//! ordered Stop barrier `input_audio_buffer.drain` (advertised in capabilities).
//! Server -> client: `session.created` / `session.updated`,
//! `conversation.item.input_audio_transcription.delta` / `.completed`,
//! `input_audio_buffer.committed` / `.cleared`, and `error`.
//!
//! ## The policy: LocalAgreement-2, and why not AlignAtt
//!
//! Whisper's encoder is a FIXED 30 s window - you cannot feed it 200 ms and
//! ask for text. So a streaming policy is not optional plumbing, it is the
//! feature: something has to decide which words are safe to SHOW, given that
//! the model will happily invent an ending for audio that has not been spoken
//! yet, then change its mind when it has.
//!
//! Two policies own this literature.
//!
//! **LocalAgreement-2** (Liu et al. 2020; whisper_streaming, Macháček et al.
//! 2023) re-transcribes a growing buffer and commits only the longest prefix
//! that two CONSECUTIVE passes agree on. It needs nothing from the model but
//! its output, which is why it is what this file implements.
//!
//! **AlignAtt** (Papi et al. 2023; Simul-Whisper, Interspeech 2024;
//! SimulStreaming, MIT) is the better one, and is the 2025 state of the art:
//! it reads the decoder's CROSS-ATTENTION and stops emitting at the first
//! token whose most-attended audio frame lands within a threshold of the end
//! of the buffer - i.e. it detects "this word is guessing about audio I have
//! not heard" directly, instead of inferring it from two passes disagreeing.
//! It is both lower-latency and cheaper (one pass, not one per second).
//!
//! We ship LocalAgreement-2 KNOWINGLY, and the reason is worth writing down
//! because it decides the order of two tasks: AlignAtt needs materialized
//! cross-attention scores out of the decoder, and our flash-decoding kernel
//! does not produce them - which is the same engine door needs for
//! word-level timestamps (whisper's DTW alignment reads the same weights).
//! One door, two features. Until it opens, the second-best policy is the best
//! available one; when it opens, this file gets AlignAtt and the other gets its
//! timings from the same tensor.
//!
//! ## What `completed` is, and what it is not
//!
//! Two promises, and the second follows from the first. The deltas
//! CONCATENATE to the completed transcript - a client that appends every
//! delta ends up holding exactly what the terminal event says, the same
//! guarantee `/v1/audio/transcriptions?stream=true` makes, so neither
//! streaming surface needs a client to know which one it is talking to.
//!
//! Which means `completed` is not the same thing as transcribing the whole
//! utterance in one go. It is the committed prefix plus the final pass's
//! tail, and the committed prefix was decided from SHORTER buffers. When the
//! model's hypothesis changes as it hears more - and it does; see the language
//! note in the decode loop - the live answer keeps what it already showed
//! rather than rewriting it. Never retracting a word is what a delta stream
//! buys, and this is what it costs.
//!
//! Revision-aware clients may explicitly request `paddock_final_revision`
//! (advertised as `realtime_transcription.final_revision`). Their terminal
//! `transcript` is the final utterance hypothesis, with `paddock_revised` true
//! when it corrects the provisional deltas. Native/web Studio negotiate this
//! mode so stored text and final word timestamps describe the same decode.
//! The existing final pass is reused; no whole-recording second transcription.
//! Other clients retain the original append-only contract by default.
//!
//! ## The utterance is the unit of finality
//!
//! `completed` used to carry text and nothing else - no segment times, no word
//! times, no per-word confidence - so a turn committed straight from a live
//! session was a second-class record next to a dropped-file turn, and the
//! Studio worked around it by transcribing the recording a SECOND time through
//! the file endpoint and storing that instead. You watched a transcript form
//! and then it was replaced wholesale by a different one, with nothing saying
//! so.
//!
//! The fix is not a better whole-file pass at the end, it is noticing that the
//! boundary is already here. A closed utterance already runs a final pass - the
//! one whose answer is authoritative rather than a hypothesis - so that is
//! where the richer decode belongs. `paddock_verbose` on `completed` is the
//! exact object the file endpoint's `verbose_json` returns for that utterance's
//! audio, and it costs no extra decode.
//!
//! What that buys, and it is the whole argument for utterance-level finality:
//! cost scales with SPEECH rather than with recording length, so a forty-minute
//! session is never re-transcribed to settle, and it settles as it goes instead
//! of all at once at the end. whisper.cpp's `stream` reaches the same shape
//! from the other direction - its VAD mode emits a bounded block per utterance
//! and that block is the one that carries timestamps
//! (`params.no_timestamps = !use_vad`), while its sliding-window mode rewrites
//! a provisional line and asks for no times at all.
//!
//! Two things it is careful about. Timestamps are a SESSION property, not a
//! final-pass one: they change the decode prompt, and LocalAgreement compares
//! each pass against the one before it, so flipping them at the boundary would
//! make the final pass disagree with its own hypotheses for no reason but the
//! prompt. Word timing (a second forward pass per window, the DTW alignment) is
//! final-pass only - that is the cost this file refused to pay when it ran
//! every pass with `false, false`, and at a closed utterance it is paid once.
//!
//! It is opt-in because of what it costs and because of what it changes:
//! dictation never shows word times, and the timestamp prompt is exactly what
//! some fine-tunes condition their no-speech refusal on.
//!
//! ## Server VAD: who decides an utterance is over
//!
//! `turn_detection: {type: "server_vad"}` hands the turn boundary to the
//! server, which is what a hands-free microphone needs - nobody presses a
//! button to say they stopped talking. The detector itself is
//! `paddock_engine::audio::vad` (read its header for what it is and what the
//! state of the art is); this file owns what the API means by a turn:
//!
//! - `threshold` picks how far above the room a frame must sit,
//! - `prefix_padding_ms` is how much audio before the first speech frame the
//!   utterance keeps - and, while nobody is speaking, exactly how much audio
//!   this session holds at all,
//! - `silence_duration_ms` is how long the quiet has to last before the turn
//!   ends,
//! - `idle_timeout_ms` announces a session where nobody said anything.
//!
//! Two properties fall out of it that are worth naming. Passes only run while
//! someone is SPEAKING, so an open microphone in an empty room costs nothing
//! and - more importantly - whisper is never handed a buffer of silence to
//! hallucinate over. And the buffer stops being a cap-until-you-commit: audio
//! before speech is dropped down to the prefix padding, so `MAX_BUFFER_S` now
//! bounds one continuous utterance instead of one forgotten session.
//!
//! With `turn_detection: null` - still the default, and what OpenAI's own
//! whisper-backed realtime model requires - none of this runs and the client
//! ends utterances with `input_audio_buffer.commit`.
//!
//! ## Buffer trimming, and why it needs no timestamps
//!
//! Every pass re-transcribes the whole buffer, so a long utterance would pay
//! for its own past over and over - and worse, whisper's last window would
//! grow ever emptier, which is where it hallucinates.
//!
//! whisper_streaming solves this by trimming at a completed SEGMENT, which
//! means asking the model for timestamps and parsing them. We do not have to:
//! our passes cut the buffer into whisper's own 30 s windows from the start,
//! windows never attend across each other, and a COMPLETE window's mel is
//! byte-identical on every later pass. So the first window's transcript stops
//! being a hypothesis the moment a second window exists - a free, exact
//! boundary where the reference implementation has to hunt for one.
//!
//! When that happens the session retires that window's words (anything in
//! there still uncommitted goes out first - nothing can revise it once its
//! audio is gone), drops its 30 s of audio, and hands its TEXT to the next
//! pass as `<|startofprev|>` context. That second half is what keeps quality:
//! trimming alone would throw away the context that made the next window good.
//! The caller's own `prompt` rides in front of it, and the engine keeps the
//! tail when the two together outgrow whisper's half-context budget.
//!
//! ## Two lanes, one policy
//!
//! LocalAgreement-2 needs nothing from the model but its output, so the whole
//! session above is model-agnostic and only the pass differs: whisper decodes
//! fixed 30 s windows on its own thread, while Qwen3-ASR and granite-speech
//! take the whole buffer as one multimodal prompt on the ordinary engine.
//! Both run here.
//!
//! Two things follow from that difference, and both are the generative lanes'
//! shape rather than a gap in this file. There is no finished-window boundary
//! on them, so nothing to TRIM at - the buffer stays bounded by server VAD and
//! the session cap instead. And the caller's `prompt` reaches them as an
//! INSTRUCTION rather than a spelling hint (on those families the field is the
//! task), which is why the retired transcript is never handed to them: asking
//! a model to follow the words it just transcribed is not context, it is a
//! command.

use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use base64::Engine as _;
use paddock_api::ErrorBody;
use serde_json::{Value, json};

use crate::routes::AppState;

/// Don't start a pass for less than this much new audio. whisper_streaming's
/// own default; below it the passes cost more than they reveal.
const MIN_CHUNK_S: f32 = 1.0;

/// How much audio one session may hold. Trimming keeps a speaking session near
/// one window, so this is the backstop for the shapes trimming cannot reach -
/// a client that never commits under manual turns, mostly. Ten minutes is far
/// past any dictation and far short of a problem.
pub(crate) const MAX_BUFFER_S: f32 = 600.0;

/// File word timestamps do not imply live enrichment (Granite Plus emits
/// them under a different instruction). Both clients consume this explicit
/// contract instead of inferring it from the file endpoint's granularities.
pub(crate) fn capabilities(audio: bool, whisper: bool, max_clip_s: Option<f64>) -> Value {
    json!({
        "supported": audio,
        "enrichment": audio && whisper,
        "drain": audio,
        "final_revision": audio,
        "utterance_max_s": if audio { Some(max_clip_s.unwrap_or(MAX_BUFFER_S as f64).min(MAX_BUFFER_S as f64)) } else { None },
        "sample_rates": [16000, 24000, 44100, 48000],
        "turn_detection": ["server_vad"],
    })
}

/// Whisper's encoder window, from the engine's own mel geometry rather than
/// the usual hardcoded 30. Buffer trimming turns on this number: windows do
/// not attend across each other, so a COMPLETE one is finished forever.
const WINDOW_S: f32 =
    (paddock_engine::audio::PAD_SAMPLES / paddock_engine::audio::SAMPLE_RATE) as f32;

/// The rate OpenAI's realtime input is defined at. A client that sends
/// something else must SAY so (see `Session::rate`) - we resample rather than
/// refuse, because a browser's AudioContext hands out 44.1 or 48 kHz and
/// making every caller resample in JS is worse than doing it here.
pub(crate) const DEFAULT_RATE: u32 = 24000;

/// `server_vad`'s defaults, straight from the API reference - a client that
/// sends `{"type": "server_vad"}` and nothing else gets these.
const VAD_THRESHOLD: f32 = 0.5;
const VAD_PREFIX_MS: u32 = 300;
const VAD_SILENCE_MS: u32 = 500;

/// Consecutive speech frames before a turn is declared started. Three 20 ms
/// frames: long enough that a door slam or a keystroke does not open an
/// utterance, short enough to keep the first syllable.
const VAD_ONSET_FRAMES: usize = 3;

#[derive(serde::Deserialize)]
pub struct RealtimeQuery {
    #[serde(default)]
    intent: String,
    #[serde(default)]
    model: Option<String>,
}

fn err(status: StatusCode, kind: &str, msg: impl Into<String>) -> Response {
    (status, Json(ErrorBody::new(kind, msg))).into_response()
}

pub async fn handle(
    State(state): State<Arc<AppState>>,
    Query(q): Query<RealtimeQuery>,
    ws: WebSocketUpgrade,
) -> Response {
    // Refuse before the upgrade wherever we can: a 400 with a body is readable
    // by every client, and an error event on a socket that just opened is not.
    if q.intent != "transcription" {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!(
                "this endpoint serves transcription sessions only \
                 (`?intent=transcription`), not {:?} - paddock has no realtime \
                 speech-to-speech model",
                q.intent
            ),
        );
    }
    if state.rnnt_asr.is_some() {
        // the native streaming RNN-T lane: no agreement policy (rnnt_realtime.rs)
        return ws.on_upgrade(move |socket| crate::rnnt_realtime::run(socket, state));
    }
    let lane = match (state.asr.as_ref(), state.serving.as_ref()) {
        (Some(asr), _) => Lane {
            kind: LaneKind::Whisper {
                transcriber: asr.transcriber.clone(),
                scale: asr.time_scale,
                tok: asr.tokenizer.clone(),
                langs: Arc::new(asr.languages.clone()),
                max_tokens: asr.max_tokens,
            },
            model: asr.id.clone(),
            word_times: asr.word_times,
        },
        // A generative ASR model runs the same policy on the ordinary engine
        // It is not the whisper lane wearing a hat: the pass is a
        // multimodal prompt rather than a windowed encode, and everything the
        // session decides on top of it is unchanged, which is the point.
        (None, Some(m)) if m.supports_audio => Lane {
            kind: LaneKind::Generative {
                state: state.clone(),
            },
            model: m.id.clone(),
            word_times: false,
        },
        (None, Some(_)) => {
            return err(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "this model does not serve transcription",
            );
        }
        (None, None) => {
            return err(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "no model is loaded",
            );
        }
    };
    if let Some(m) = &q.model {
        tracing::debug!(asked = %m, serving = %lane.model, "realtime: model names a deployment");
    }
    let lease = if let LaneKind::Whisper { transcriber, .. } = &lane.kind {
        match transcriber.session_lease().await {
            Ok(lease) => lease,
            Err(error) => return crate::asr_residency::error_response(error),
        }
    } else {
        None
    };
    ws.on_upgrade(move |socket| async move {
        let _lease = lease;
        run(socket, lane).await;
    })
}

/// Everything a session needs from the loaded model, cloned so the socket task
/// owns it - a WebSocket outlives the handler that upgraded it.
#[derive(Clone)]
struct Lane {
    kind: LaneKind,
    model: String,
    word_times: bool,
}

/// The two shapes of ASR behind one session.
///
/// LocalAgreement-2 is model-agnostic by construction - it needs nothing from
/// the model but its output - so the policy above is shared and only the pass
/// differs: whisper decodes fixed 30 s windows on its own thread, a generative
/// model takes the whole buffer as one multimodal prompt on the ordinary
/// engine.
#[derive(Clone)]
enum LaneKind {
    Whisper {
        transcriber: crate::asr_residency::Handle,
        tok: Arc<paddock_tokenizer::GgufTokenizer>,
        /// the checkpoint's timestamp geometry - needed here only to tell a
        /// timestamp token from a text one when the marker rule reads a
        /// window's transcript
        scale: paddock_engine::gpu_model::whisper::TimeScale,
        /// the checkpoint's own language map - bounds the transcript's
        /// language check to what this model could have written.
        /// Shared rather than cloned per pass: it is 99 short strings and a
        /// live session runs a pass every turn.
        langs: Arc<Vec<String>>,
        max_tokens: usize,
    },
    Generative {
        state: Arc<AppState>,
    },
}

/// What a pass is given as context.
///
/// Two fields rather than one string because the lanes want different halves.
/// Whisper feeds both behind `<|startofprev|>`, where text is a spelling hint.
/// The generative families take only the INSTRUCTION: on those models the
/// field selects the task, so handing them the transcript so far would be
/// asking them to do what it says.
#[derive(Clone, Default)]
struct Context {
    asked: String,
    carried: String,
}

impl Context {
    fn joined(&self) -> String {
        format!("{} {}", self.asked, self.carried).trim().to_owned()
    }
}

pub(crate) fn id(prefix: &str) -> String {
    format!("{prefix}_{}", uuid::Uuid::new_v4().simple())
}

/// What a pass is asked to produce beyond words on a page.
///
/// Both are off unless the session opted into `paddock_verbose`, which keeps
/// the default lane exactly what it was: a re-transcribe-the-buffer pass that
/// pays for nothing it has no event to carry.
#[derive(Clone, Copy, Default)]
struct Times {
    /// Decode with timestamp tokens. Constant for the whole SESSION - see the
    /// module note on why this must not flip at the utterance boundary.
    segments: bool,
    /// Run the DTW alignment for word times. Final pass only: it is a second
    /// forward pass per window, and re-running it every second on a growing
    /// buffer is the waste this file was written to avoid.
    words: bool,
    /// Return final-pass metadata even when this backend cannot align words.
    verbose: bool,
}

impl Times {
    fn for_pass(verbose: bool, final_pass: bool, word_times: bool) -> Self {
        Self {
            segments: verbose,
            words: verbose && final_pass && word_times,
            verbose: verbose && final_pass,
        }
    }
}

/// A pass's answer.
struct PassOut {
    /// The text per WINDOW rather than joined, because whether a window is
    /// finished is the session's trimming rule (see `WINDOW_S`) and the join is
    /// one line at the call site.
    parts: Vec<String>,
    language: String,
    /// The `paddock_verbose` object - present only on a final pass of a session
    /// that asked for it, and only on the whisper lane.
    verbose: Option<Value>,
}

/// One transcription pass over the whole buffer: resample + mel off the async
/// threads, then the ordinary whisper decode.
async fn pass(
    lane: Lane,
    pcm: Vec<f32>,
    rate: u32,
    language: Option<String>,
    ctx: Context,
    times: Times,
) -> Result<PassOut, String> {
    // What this pass covers, in seconds of audio - the `duration` of the
    // utterance as far as the enriched object is concerned. Taken before the
    // resample, from the samples the session actually holds.
    let duration_s = pcm.len() as f64 / rate.max(1) as f64;
    match lane.kind {
        LaneKind::Whisper {
            transcriber,
            tok,
            scale,
            langs,
            max_tokens,
        } => {
            let prepared = tokio::task::spawn_blocking(move || -> Result<Vec<_>, String> {
                let samples = paddock_engine::audio::resample::resample(&pcm, rate, 16000)?;
                let mut windows = Vec::new();
                let mut off = 0usize;
                while off < samples.len().max(1) {
                    let end = (off + paddock_engine::audio::PAD_SAMPLES).min(samples.len());
                    windows.push(paddock_engine::audio::whisper_features(&samples[off..end])?);
                    off = end;
                }
                Ok(windows)
            })
            .await
            .map_err(|e| e.to_string())??;

            let context = ctx.joined();
            let prompt = if context.is_empty() {
                Vec::new()
            } else {
                // whisper's own convention: the context joins the marker with a
                // leading space, because byte-level BPE spells a word
                // differently at the start of a line than mid-sentence
                tok.encode(&format!(" {context}"))
                    .map_err(|e| e.to_string())?
            };
            // the session's own instruction, kept for the language report -
            // `transcribe` consumes it below
            let asked = language.clone();
            let mut out = transcriber
                // `times` is off on both counts unless the session asked for
                // `paddock_verbose`, which keeps the default lane what it has
                // always been: a re-transcribe-the-buffer pass that pays for
                // nothing the socket has no event to carry.
                // no window gate here: this socket already runs its own VAD for
                // turn detection, and the buffer it hands over has
                // been cut at the turn boundaries that detector found. Gating a
                // second time would only be able to disagree with the first.
                .transcribe(
                    prepared,
                    Vec::new(),
                    // A live session settles its language once, at session
                    // config, so a pass either forces it or detects - there is
                    // no per-utterance candidate set to hint with (the session
                    // would have to re-ask on every turn, which is the mid-clip
                    // flip the file lane refuses for the same reason).
                    paddock_engine::transcriber::LanguageAsk::forced(language),
                    prompt,
                    times.segments,
                    times.words,
                    max_tokens,
                    None,
                )
                .await?;
            // the same rule the file lane applies: a fine-tune that
            // TYPES its no-speech marker must not have that typed into the
            // user's live transcript either
            crate::transcriptions::suppress_marker_windows(&mut out, &tok, &scale);
            let mut parts = Vec::with_capacity(out.windows.len());
            for w in &out.windows {
                // Timestamp tokens are decode control and never reach a
                // transcript. Unconditional because it has to be: with
                // `times.segments` on they are in the stream, and a delta that
                // carried `<|4.20|>` into someone's composer would be the
                // opt-in quietly changing what the text is.
                let ids: Vec<u32> = w
                    .tokens
                    .iter()
                    .copied()
                    .filter(|&t| !scale.is_timestamp(t))
                    .collect();
                parts.push(tok.decode(&ids, true).map_err(|e| e.to_string())?);
            }
            // The enriched object describes this pass - see `live_verbose` on
            // why it is built from the final pass's own join rather than from
            // the transcript the deltas assembled.
            let verbose = times.verbose.then(|| {
                let text = paddock_engine::gpu_model::whisper::join_windows(&parts, &out.language);
                crate::transcriptions::live_verbose(
                    &tok,
                    &scale,
                    &langs,
                    asked.as_deref(),
                    &out,
                    &text,
                    duration_s,
                )
            });
            Ok(PassOut {
                parts,
                language: out.language,
                verbose,
            })
        }
        LaneKind::Generative { state } => {
            let Some(model) = state.serving.as_ref() else {
                return Err("no model is loaded".into());
            };
            let samples = tokio::task::spawn_blocking(move || {
                paddock_engine::audio::resample::resample(&pcm, rate, 16000)
            })
            .await
            .map_err(|e| e.to_string())??;
            let instruction = Some(ctx.asked.trim()).filter(|s| !s.is_empty());
            let (text, lang) = crate::transcriptions::generative_pass(
                model,
                samples,
                instruction,
                language.as_deref(),
                state.max_ctx,
            )
            .await?;
            // One "window": these families take the whole buffer as a single
            // prompt, so there is no finished-window boundary to trim at (see
            // the module note). The session's own cap and server VAD are what
            // bound the buffer here.
            //
            // No `verbose` either, and `session.update` refuses the opt-in on
            // this lane by name rather than accepting it and answering with
            // nothing: these families have no timestamp vocabulary at all
            // so there is no enriched pass to run.
            Ok(PassOut {
                parts: vec![text],
                language: lang.unwrap_or_default(),
                verbose: None,
            })
        }
    }
}

/// LocalAgreement-2's committed/pending bookkeeping.
///
/// The rule in one line: a word is shown once two consecutive passes over a
/// growing buffer produce it in the same place. `pending` is what the last
/// pass said beyond the commit point; the next pass's tail is compared against
/// it and their common prefix becomes committed.
#[derive(Default)]
struct Agreement {
    /// Words already sent AND whose audio has left the buffer (see
    /// `Live::trim`). They are no longer part of any hypothesis, so they take
    /// no part in the agreement - they only have to survive into the final
    /// transcript.
    retired: Vec<String>,
    /// words already sent as deltas, still inside the buffer's hypothesis
    committed: Vec<String>,
    /// the previous pass's words past `committed`, awaiting a second opinion
    pending: Vec<String>,
}

impl Agreement {
    /// Revision-capable clients replace provisional words with the final
    /// hypothesis whose timestamps we return. Retired windows remain intact.
    fn final_text(&self, words: &[String]) -> String {
        self.retired
            .iter()
            .chain(words)
            .cloned()
            .collect::<Vec<_>>()
            .join(" ")
    }
    /// Feed one pass's full hypothesis; returns the newly committed words.
    ///
    /// Indexing is POSITIONAL: `committed` is treated as a prefix of every
    /// later hypothesis. That is exact here because each pass re-transcribes
    /// the same buffer from its start (nothing is ever trimmed off the front),
    /// and it is the same structure whisper_streaming's HypothesisBuffer uses.
    /// A pass that comes back SHORTER than the commit point is a model that
    /// changed its mind about audio it already heard - the commit stands, and
    /// the short hypothesis contributes nothing.
    fn feed(&mut self, words: Vec<String>) -> Vec<String> {
        if words.len() <= self.committed.len() {
            self.pending.clear();
            return Vec::new();
        }
        let tail = &words[self.committed.len()..];
        let k = self
            .pending
            .iter()
            .zip(tail.iter())
            .take_while(|(a, b)| a == b)
            .count();
        let fresh: Vec<String> = tail[..k].to_vec();
        self.committed.extend(fresh.iter().cloned());
        self.pending = tail[k..].to_vec();
        fresh
    }

    /// The audio behind the first `n` words of this hypothesis is about to
    /// leave the buffer. Anything in there that was not committed yet is
    /// committed now - nothing can revise it once its audio is gone - and the
    /// lot moves to `retired`, which is what re-bases the positional indexing
    /// onto the shorter buffer the next pass will see.
    ///
    /// Returns the words that had not been sent yet, so the caller can.
    fn retire(&mut self, words: &[String], n: usize) -> Vec<String> {
        let n = n.min(words.len());
        let mut fresh = Vec::new();
        while self.committed.len() < n {
            let w = words[self.committed.len()].clone();
            self.committed.push(w.clone());
            fresh.push(w);
        }
        // `pending` is "this hypothesis beyond what is committed", and that is
        // the same set of WORDS before and after a retirement - only its index
        // moves. Recompute it against the commit point, not against `n`, or a
        // forced commit past it would put words back in the queue that have
        // already gone out.
        self.pending = words[self.committed.len().min(words.len())..].to_vec();
        self.retired.extend(self.committed.drain(..n));
        fresh
    }

    /// Everything shown so far, as text.
    fn text(&self) -> String {
        let mut out = self.retired.join(" ");
        if !self.committed.is_empty() {
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(&self.committed.join(" "));
        }
        out
    }
}

/// A finished pass, handed back to the socket loop.
struct Pass {
    result: Result<PassOut, String>,
    /// `Some(n)` on the FINAL pass of an utterance, whose answer is
    /// authoritative rather than a hypothesis: it covered `buf[..n]`, and that
    /// many samples leave the buffer when it lands. `None` on a hypothesis
    /// pass over everything buffered so far.
    cut: Option<usize>,
    /// Absolute index of this pass's first sample - where the utterance it
    /// closes begins in the session. Captured at kick rather than read back at
    /// landing: the buffer's origin is stable across a final pass today (the
    /// trims are all in the other branch, and `inflight` blocks the quiet one),
    /// but an item that cannot be placed in its session is not the kind of
    /// thing to make depend on that staying true.
    origin: usize,
    /// which buffer this pass was about (see `Live::epoch`)
    epoch: u64,
}

/// `server_vad`, as the session was told to run it. Held in the wire's own
/// units so `session.updated` can hand the client back exactly what it sent;
/// samples are derived against the session rate at the point of use.
#[derive(Clone, Copy)]
pub(crate) struct VadConfig {
    pub(crate) threshold: f32,
    pub(crate) prefix_ms: u32,
    pub(crate) silence_ms: u32,
    pub(crate) idle_ms: Option<u32>,
}

impl VadConfig {
    fn samples(ms: u32, rate: u32) -> usize {
        (ms as u64 * rate as u64 / 1000) as usize
    }
    pub(crate) fn prefix(&self, rate: u32) -> usize {
        Self::samples(self.prefix_ms, rate)
    }
}

/// What the detector's frame verdicts mean for a TURN.
///
/// The three rules are the three knobs: a turn opens after
/// `VAD_ONSET_FRAMES` of speech, closes when `silence` has passed since the
/// last speech frame, and - when nobody has said anything for `idle` - says so
/// without closing anything.
pub(crate) struct Turns {
    silence: usize,
    idle: Option<usize>,
    pub(crate) speaking: bool,
    /// consecutive speech frames seen while not yet speaking
    onset: usize,
    /// absolute sample index just past the last frame that held speech
    last_speech: usize,
    /// absolute index where the current quiet stretch began (a session start,
    /// a turn end, or the last idle announcement)
    quiet_since: usize,
}

/// What the session has to do about it. Positions are absolute sample indices
/// counted from the first sample of the session, the same space the detector's
/// frames use and the same one `audio_start_ms` is defined in.
#[derive(Debug, PartialEq)]
pub(crate) enum Turn {
    Started { at: usize },
    Stopped { at: usize },
    Idle { from: usize, to: usize },
}

impl Turns {
    fn new(cfg: &VadConfig, rate: u32, base: usize) -> Self {
        Self {
            silence: VadConfig::samples(cfg.silence_ms, rate),
            idle: cfg.idle_ms.map(|ms| VadConfig::samples(ms, rate)),
            speaking: false,
            onset: 0,
            last_speech: base,
            quiet_since: base,
        }
    }

    fn feed(&mut self, frames: &[paddock_engine::audio::vad::Frame]) -> Vec<Turn> {
        let mut out = Vec::new();
        for f in frames {
            if self.speaking {
                if f.speech {
                    self.last_speech = f.end;
                } else if f.end - self.last_speech >= self.silence {
                    // the turn ended where the speech did, not where we
                    // noticed: the silence we waited out belongs to neither
                    self.speaking = false;
                    self.onset = 0;
                    self.quiet_since = self.last_speech;
                    out.push(Turn::Stopped {
                        at: self.last_speech,
                    });
                }
                continue;
            }
            if f.speech {
                self.onset += 1;
                if self.onset >= VAD_ONSET_FRAMES {
                    // back-date the start to the run's first frame - the
                    // confirmation delay is ours, and the speaker should not
                    // lose a syllable to it
                    let at = f.end - self.onset * (f.end - f.start);
                    self.speaking = true;
                    self.last_speech = f.end;
                    out.push(Turn::Started { at });
                }
                continue;
            }
            self.onset = 0;
            if let Some(idle) = self.idle
                && f.end - self.quiet_since >= idle
            {
                out.push(Turn::Idle {
                    from: self.quiet_since,
                    to: f.end,
                });
                self.quiet_since = f.end;
            }
        }
        out
    }

    /// The client committed by hand while the detector thought someone was
    /// talking. The client wins - it knows things a microphone does not.
    pub(crate) fn yield_to_client(&mut self, at: usize) {
        self.speaking = false;
        self.onset = 0;
        self.last_speech = at;
        self.quiet_since = at;
    }
}

/// The detector, its policy, and where in the session it started.
///
/// `base` exists because a `session.update` can change the rate or the
/// thresholds mid-session, which means a new detector - and the frame
/// positions it counts from zero have to keep meaning the same thing as the
/// `audio_start_ms` the client has already been shown.
pub(crate) struct Detector {
    pub(crate) cfg: VadConfig,
    vad: paddock_engine::audio::vad::Vad,
    pub(crate) turns: Turns,
    base: usize,
}

impl Detector {
    pub(crate) fn new(cfg: VadConfig, rate: u32, base: usize) -> Self {
        Self {
            vad: paddock_engine::audio::vad::Vad::new(rate, cfg.threshold),
            turns: Turns::new(&cfg, rate, base),
            cfg,
            base,
        }
    }

    pub(crate) fn feed(&mut self, pcm: &[f32]) -> Vec<Turn> {
        let mut frames = self.vad.feed(pcm);
        for f in &mut frames {
            f.start += self.base;
            f.end += self.base;
        }
        self.turns.feed(&frames)
    }
}

/// Everything the socket loop keeps about the utterance in progress.
struct Live {
    /// audio that has not yet been turned into a completed item
    buf: Vec<f32>,
    /// absolute index - samples since the session began - of `buf[0]`
    origin: usize,
    /// how much of `buf` the last pass saw; the `MIN_CHUNK_S` throttle
    seen: usize,
    inflight: bool,
    /// `commit` is consumed when its pass starts; retain whether that pass
    /// still owes a completed item for the ordered drain acknowledgement.
    final_inflight: bool,
    /// set when a turn has ended: the utterance is `buf[..n]`, and the next
    /// pass over it is the final one
    commit: Option<usize>,
    agreed: Agreement,
    item: String,
    sent_since_commit: usize,
    /// Bumped whenever the buffer is thrown away under a running pass (a
    /// `clear`). A pass that comes back stamped with an older generation is
    /// answering a question nobody is asking any more, and its `cut` would
    /// drain samples that belong to the next utterance.
    epoch: u64,
}

impl Live {
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            origin: 0,
            seen: 0,
            inflight: false,
            final_inflight: false,
            commit: None,
            agreed: Agreement::default(),
            item: id("item"),
            sent_since_commit: 0,
            epoch: 0,
        }
    }

    /// Absolute index one past the last buffered sample.
    fn head(&self) -> usize {
        self.origin + self.buf.len()
    }

    fn pending_items(&self) -> Vec<String> {
        if self.commit.is_some() || self.final_inflight {
            vec![self.item.clone()]
        } else {
            Vec::new()
        }
    }

    /// Drop `n` samples off the front, keeping every absolute index honest.
    fn drop_front(&mut self, n: usize) {
        let n = n.min(self.buf.len());
        self.buf.drain(..n);
        self.origin += n;
        self.seen = self.seen.saturating_sub(n);
    }

    /// Start a new item: what happens after a completed transcript, and after
    /// a client throws the buffer away.
    fn restart(&mut self) {
        self.seen = 0;
        self.commit = None;
        self.final_inflight = false;
        self.agreed = Agreement::default();
        self.sent_since_commit = 0;
        self.item = id("item");
    }
}

async fn run(mut socket: WebSocket, lane: Lane) {
    let session_id = id("sess");
    let mut rate = DEFAULT_RATE;
    let mut language: Option<String> = None;
    let mut det: Option<Detector> = None;
    let mut live = Live::new();
    // the caller's `prompt`; the retired transcript joins it per pass
    let mut asked = String::new();
    // whether a closed utterance comes back enriched
    let mut verbose = false;
    let mut final_revision = false;

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Pass>();

    if send(
        &mut socket,
        session_event(
            "session.created",
            &session_id,
            &lane,
            rate,
            &language,
            det.as_ref(),
            &asked,
            verbose,
            final_revision,
        ),
    )
    .await
    .is_err()
    {
        return;
    }

    loop {
        tokio::select! {
            biased;
            // pass results first: they are what turns buffered audio into text,
            // and a client that is still uploading must not starve them
            Some(p) = rx.recv() => {
                live.inflight = false;
                live.final_inflight = false;
                if p.epoch != live.epoch {
                    // the audio this pass was about has been thrown away; its
                    // `cut` would drain samples belonging to the next utterance
                    kick(&lane, &mut live, rate, &language, &asked, &tx, hold(&det), verbose);
                    continue;
                }
                match p.result {
                    Ok(out) => {
                        let text = paddock_engine::gpu_model::whisper::join_windows(
                            &out.parts,
                            &out.language,
                        );
                        // The language LOCKS on the first pass that resolves
                        // one, and stays locked. Whisper detects per window, so
                        // letting every pass re-decide would flip the language
                        // mid-utterance - and words already committed cannot be
                        // retracted, so the flip would show as a transcript
                        // half in each language. The cost is that the decision
                        // is made on the first second or so of speech; a client
                        // that already knows should send `language` in
                        // `session.update` and skip the guess entirely.
                        //
                        // Measured on the English LibriSpeech
                        // fixture with the Swedish-tuned kb-whisper-large: the
                        // one-second detection said `en` where the whole-clip
                        // detection said `sv` and TRANSLATED. More audio is not
                        // automatically a better detection.
                        if language.is_none() && !out.language.is_empty() {
                            language = Some(out.language.clone());
                            // say so: the session's effective config changed,
                            // and a client that is not told cannot know which
                            // language it is reading
                            let up = session_event(
                                "session.updated", &session_id, &lane, rate, &language,
                                det.as_ref(), &asked, verbose, final_revision,
                            );
                            if send(&mut socket, up).await.is_err() {
                                return;
                            }
                        }
                        let words: Vec<String> =
                            text.split_whitespace().map(str::to_owned).collect();
                        if let Some(cut) = p.cut {
                            // The final pass is the answer: everything it says
                            // past what was already shown goes out as one last
                            // delta, so the deltas still concatenate to the
                            // completed transcript. It only covers what is in
                            // the buffer, so it is indexed against `committed`
                            // - retired words are audio this pass never saw.
                            let head = live.agreed.text();
                            let rest = words
                                .get(live.agreed.committed.len()..)
                                .unwrap_or_default()
                                .join(" ");
                            if !rest.is_empty()
                                && send(
                                    &mut socket,
                                    delta_event(&live.item, &rest, !head.is_empty()),
                                )
                                .await
                                .is_err()
                            {
                                return;
                            }
                            let full = match (head.is_empty(), rest.is_empty()) {
                                (true, _) => rest,
                                (false, true) => head,
                                (false, false) => format!("{head} {rest}"),
                            };
                            let canonical = live.agreed.final_text(&words);
                            let revised = final_revision && canonical != full;
                            let mut done = json!({
                                "type": "conversation.item.input_audio_transcription.completed",
                                "event_id": id("event"),
                                "item_id": live.item,
                                "content_index": 0,
                                "transcript": if final_revision { canonical } else { full },
                                "paddock_revised": revised,
                                // the duration variant, which is the honest one
                                // for a model billed by nothing: this is how
                                // much audio was heard, not a token price. The
                                // utterance is what was CUT, not what is
                                // buffered - audio that arrived while the final
                                // pass ran belongs to the next one.
                                "usage": {
                                    "type": "duration",
                                    "seconds": cut as f64 / rate.max(1) as f64,
                                },
                                // PADDOCK EXTENSION. Where this item sits in the
                                // session, in the same space `audio_start_ms`
                                // uses on the VAD events - because those events
                                // are the only other way to learn it, and under
                                // manual turns they never fire. An item that
                                // cannot be placed against the recording cannot
                                // be a row in a timeline or a seek target.
                                "paddock_audio_start_ms": ms(p.origin, rate),
                            });
                            // PADDOCK EXTENSION. The enriched
                            // object for this utterance: segments, word times,
                            // per-word confidence and any guard notices, in the
                            // exact shape the file endpoint's verbose_json
                            // returns. Absent unless the session asked.
                            if let Some(v) = out.verbose {
                                done["paddock_verbose"] = v;
                            }
                            if send(&mut socket, done).await.is_err() {
                                return;
                            }
                            live.drop_front(cut);
                            live.restart();
                        } else {
                            for w in live.agreed.feed(words.clone()) {
                                let lead = live.sent_since_commit > 0;
                                live.sent_since_commit += 1;
                                if send(&mut socket, delta_event(&live.item, &w, lead))
                                    .await
                                    .is_err()
                                {
                                    return;
                                }
                            }
                            // ---- buffer trimming ----
                            //
                            // Whisper's windows never attend across each other,
                            // so once the buffer holds a COMPLETE 30 s window
                            // that window's mel is byte-identical every pass and
                            // its transcript can no longer change. That is the
                            // boundary whisper_streaming has to hunt for with
                            // segment timestamps, sitting here for free: retire
                            // its words, drop its audio, and hand its text to
                            // the next pass as `<|startofprev|>` context, which
                            // is the half of this that makes the next window
                            // good rather than merely cheap.
                            if out.parts.len() > 1 {
                                let n = out.parts[0].split_whitespace().count();
                                for w in live.agreed.retire(&words, n) {
                                    let lead = live.sent_since_commit > 0;
                                    live.sent_since_commit += 1;
                                    if send(&mut socket, delta_event(&live.item, &w, lead))
                                        .await
                                        .is_err()
                                    {
                                        return;
                                    }
                                }
                                live.drop_front((WINDOW_S * rate as f32) as usize);
                            }
                        }
                    }
                    Err(e) => {
                        if send(&mut socket, error_event("internal_error", &e)).await.is_err() {
                            return;
                        }
                        live.commit = None;
                    }
                }
                kick(&lane, &mut live, rate, &language, &asked, &tx, hold(&det), verbose);
            }
            msg = socket.recv() => {
                let Some(Ok(msg)) = msg else { return };
                let text = match msg {
                    Message::Text(t) => t.to_string(),
                    // Binary frames are not part of this protocol - audio rides
                    // base64 inside `input_audio_buffer.append`, per the spec.
                    Message::Binary(_) => {
                        if send(&mut socket, error_event(
                            "invalid_request_error",
                            "binary frames are not part of the realtime protocol; send audio as \
                             base64 in `input_audio_buffer.append`",
                        )).await.is_err() {
                            return;
                        }
                        continue;
                    }
                    Message::Close(_) => return,
                    Message::Ping(_) | Message::Pong(_) => continue,
                };
                let ev: Value = match serde_json::from_str(&text) {
                    Ok(v) => v,
                    Err(e) => {
                        if send(&mut socket, error_event("invalid_request_error", &e.to_string()))
                            .await
                            .is_err()
                        {
                            return;
                        }
                        continue;
                    }
                };
                match ev.get("type").and_then(Value::as_str).unwrap_or_default() {
                    "session.update" => {
                        // Checked before `apply_update` touches anything: a
                        // refused update must not half-apply, and this is the
                        // one refusal that depends on the lane rather than on
                        // the object, so it cannot live inside the parser.
                        if matches!(lane.kind, LaneKind::Generative { .. })
                            && ev
                                .pointer("/session/audio/input/transcription/paddock_verbose")
                                .and_then(Value::as_bool)
                                == Some(true)
                        {
                            let msg = concat!(
                                "`paddock_verbose` is not served on this model: these families ",
                                "have no timestamp vocabulary, so a closed utterance has no ",
                                "segments and no word times to report",
                            );
                            if send(&mut socket, error_event("invalid_request_error", msg))
                                .await
                                .is_err()
                            {
                                return;
                            }
                            continue;
                        }
                        let mut vad = det.as_ref().map(|d| d.cfg);
                        let next_revision = match ev.pointer("/session/audio/input/transcription/paddock_final_revision") {
                            Some(Value::Bool(value)) => *value,
                            None | Some(Value::Null) => final_revision,
                            Some(_) => {
                                if send(&mut socket, error_event("invalid_request_error", "`paddock_final_revision` must be a boolean")).await.is_err() { return; }
                                continue;
                            }
                        };
                        match apply_update(
                            ev.get("session"), &mut rate, &mut language, &mut vad, &mut asked,
                            &mut verbose,
                        ) {
                            Ok(()) => {
                                final_revision = next_revision;
                                // both the rate and the thresholds change what
                                // a frame is, so the detector is rebuilt rather
                                // than adjusted - starting from here, not from
                                // the audio it never heard
                                det = vad.map(|cfg| Detector::new(cfg, rate, live.head()));
                                let up = session_event(
                                    "session.updated", &session_id, &lane, rate, &language,
                                    det.as_ref(), &asked, verbose, final_revision,
                                );
                                if send(&mut socket, up).await.is_err() {
                                    return;
                                }
                            }
                            Err(e) => {
                                if send(&mut socket, error_event("invalid_request_error", &e))
                                    .await
                                    .is_err()
                                {
                                    return;
                                }
                            }
                        }
                    }
                    "input_audio_buffer.append" => {
                        let b64 = ev.get("audio").and_then(Value::as_str).unwrap_or_default();
                        match decode_pcm16(b64) {
                            Ok(mut s) => {
                                // the detector reads the samples before they
                                // join the buffer, so a turn boundary can be
                                // expressed against a buffer that already holds
                                // the audio it refers to
                                let turns =
                                    det.as_mut().map(|d| d.feed(&s)).unwrap_or_default();
                                live.buf.append(&mut s);
                                for t in turns {
                                    // the server ended the turn, so the server
                                    // owes the client the same acknowledgement
                                    // its own `commit` would have earned
                                    let mut ending = false;
                                    let out = match t {
                                        Turn::Started { at } => json!({
                                            "type": "input_audio_buffer.speech_started",
                                            "event_id": id("event"),
                                            "item_id": live.item,
                                            "audio_start_ms": ms(at, rate),
                                        }),
                                        Turn::Stopped { at } => {
                                            // the utterance ends a padding past
                                            // the last speech: the same cushion
                                            // that precedes it, so a trailing
                                            // consonant is not clipped off
                                            let pad = det
                                                .as_ref()
                                                .map_or(0, |d| d.cfg.prefix(rate));
                                            let cut = (at + pad)
                                                .saturating_sub(live.origin)
                                                .min(live.buf.len());
                                            live.commit = (cut > 0).then_some(cut);
                                            ending = live.commit.is_some();
                                            json!({
                                                "type": "input_audio_buffer.speech_stopped",
                                                "event_id": id("event"),
                                                "item_id": live.item,
                                                "audio_end_ms": ms(at, rate),
                                            })
                                        }
                                        Turn::Idle { from, to } => json!({
                                            "type": "input_audio_buffer.timeout_triggered",
                                            "event_id": id("event"),
                                            "item_id": live.item,
                                            "audio_start_ms": ms(from, rate),
                                            "audio_end_ms": ms(to, rate),
                                        }),
                                    };
                                    if send(&mut socket, out).await.is_err() {
                                        return;
                                    }
                                    if ending && send(&mut socket, committed_event(&live.item))
                                        .await
                                        .is_err()
                                    {
                                        return;
                                    }
                                }
                                // While nobody is speaking, hold only the
                                // pre-roll: enough to give the next utterance
                                // its `prefix_padding_ms` and not one sample
                                // more. This is what turns MAX_BUFFER_S from a
                                // limit on a forgotten SESSION into a limit on
                                // one very long utterance. Never while a pass
                                // is running - its `cut` indexes this buffer.
                                if let Some(d) = det.as_ref()
                                    && !d.turns.speaking
                                    && live.commit.is_none()
                                    && !live.inflight
                                {
                                    let keep = d.cfg.prefix(rate) + d.vad.frame_len();
                                    if live.buf.len() > keep {
                                        let n = live.buf.len() - keep;
                                        live.drop_front(n);
                                    }
                                }
                                let cap = (MAX_BUFFER_S * rate as f32) as usize;
                                if live.buf.len() > cap {
                                    // Named, not silent: truncating a client's
                                    // audio without a word is the failure mode
                                    // the product principles ban.
                                    let how = if det.is_some() {
                                        "one utterance has run this long without a pause long \
                                         enough to end it"
                                    } else {
                                        "send `input_audio_buffer.commit` to close an utterance"
                                    };
                                    let e = format!(
                                        "the input audio buffer holds {:.0} s (limit {:.0} s) - \
                                         {how}",
                                        live.buf.len() as f32 / rate as f32,
                                        MAX_BUFFER_S,
                                    );
                                    let _ = send(&mut socket, error_event(
                                        "invalid_request_error", &e,
                                    )).await;
                                    return;
                                }
                            }
                            Err(e) => {
                                if send(&mut socket, error_event("invalid_request_error", &e))
                                    .await
                                    .is_err()
                                {
                                    return;
                                }
                                continue;
                            }
                        }
                        kick(&lane, &mut live, rate, &language, &asked, &tx, hold(&det), verbose);
                    }
                    "input_audio_buffer.drain" => {
                        // Ordered input barrier: every earlier append has
                        // reached VAD. Close active speech, not silence/pre-roll,
                        // and acknowledge before the client waits for remaining
                        // completed events. This removes Stop's race against
                        // delayed speech_started without retranscribing a file.
                        let speaking = det.as_ref().is_none_or(|d| d.turns.speaking);
                        if speaking && !live.buf.is_empty() {
                            if send(&mut socket, committed_event(&live.item)).await.is_err() { return; }
                            live.commit = Some(live.buf.len());
                            if let Some(d) = det.as_mut() { d.turns.yield_to_client(live.head()); }
                            kick(&lane, &mut live, rate, &language, &asked, &tx, hold(&det), verbose);
                        }
                        if send(&mut socket, json!({
                            "type": "input_audio_buffer.drained",
                            "event_id": id("event"),
                            "audio_end_ms": ms(live.head(), rate),
                            "pending_item_ids": live.pending_items(),
                        })).await.is_err() { return; }
                    }
                    "input_audio_buffer.commit" => {
                        if live.buf.is_empty() {
                            if send(&mut socket, error_event(
                                "invalid_request_error",
                                "the input audio buffer is empty - nothing to commit",
                            )).await.is_err() {
                                return;
                            }
                            continue;
                        }
                        if send(&mut socket, committed_event(&live.item)).await.is_err() {
                            return;
                        }
                        live.commit = Some(live.buf.len());
                        if let Some(d) = det.as_mut() {
                            d.turns.yield_to_client(live.head());
                        }
                        kick(&lane, &mut live, rate, &language, &asked, &tx, hold(&det), verbose);
                    }
                    "input_audio_buffer.clear" => {
                        // A pass may be in flight over audio that is about to
                        // stop existing; the generation bump is what makes its
                        // answer land in the bin instead of in the transcript.
                        live.origin = live.head();
                        live.buf.clear();
                        live.epoch += 1;
                        live.restart();
                        if let Some(d) = det.as_mut() {
                            d.turns.yield_to_client(live.origin);
                        }
                        let ack = json!({
                            "type": "input_audio_buffer.cleared",
                            "event_id": id("event"),
                        });
                        if send(&mut socket, ack).await.is_err() {
                            return;
                        }
                    }
                    other => {
                        // Every client event this session does not serve is
                        // refused by NAME - the response ones because there is
                        // no model to respond with, the conversation ones
                        // because a transcription session has no conversation.
                        let why = match other {
                            "response.create" | "response.cancel" => {
                                "a transcription session has no response model; \
                                 use /v1/chat/completions for that"
                            }
                            "conversation.item.create"
                            | "conversation.item.delete"
                            | "conversation.item.retrieve"
                            | "conversation.item.truncate"
                            | "output_audio_buffer.clear" => {
                                "a transcription session has no conversation and no output audio"
                            }
                            "" => "every event needs a `type`",
                            _ => "unknown event type",
                        };
                        let msg = format!("`{other}`: {why}");
                        if send(&mut socket, error_event("invalid_request_error", &msg))
                            .await
                            .is_err()
                        {
                            return;
                        }
                    }
                }
            }
        }
    }
}

/// Start a pass if one is warranted and none is running.
///
/// This is the whole throttle, and it is self-tuning: a pass covers whatever
/// has accumulated, so if the model takes three seconds the next pass simply
/// has three seconds more audio in it. Nothing queues up behind a slow model.
///
/// `hold` is server VAD's answer to "is anyone talking": while it is set, only
/// a pending commit gets a pass, so an open microphone in a quiet room costs
/// nothing and whisper is never asked what it hears in silence.
#[allow(clippy::too_many_arguments)]
fn kick(
    lane: &Lane,
    live: &mut Live,
    rate: u32,
    language: &Option<String>,
    asked: &str,
    tx: &tokio::sync::mpsc::UnboundedSender<Pass>,
    hold: bool,
    verbose: bool,
) {
    if live.inflight || live.buf.is_empty() {
        return;
    }
    let cut = live.commit.take();
    let upto = cut.unwrap_or(live.buf.len()).min(live.buf.len());
    if cut.is_none() {
        let min_new = (MIN_CHUNK_S * rate as f32) as usize;
        if hold || live.buf.len().saturating_sub(live.seen) < min_new {
            return;
        }
    }
    if upto == 0 {
        return;
    }
    live.seen = live.buf.len();
    live.inflight = true;
    live.final_inflight = cut.is_some();
    // The context is rebuilt from the session's own state every pass rather
    // than cached: `retired` only changes on a trim, and joining a few hundred
    // words once a second is not worth a second copy that can go stale.
    let ctx = Context {
        asked: asked.to_owned(),
        carried: live.agreed.retired.join(" "),
    };
    // Keep timestamp decoding constant across hypotheses, but only run final
    // alignment when the loaded backend supports it. Metal Whisper otherwise
    // fails exactly when finalizing an utterance, stranding its partial text.
    let times = Times::for_pass(verbose, cut.is_some(), lane.word_times);
    let origin = live.origin;
    let (lane, pcm, language, tx, epoch) = (
        lane.clone(),
        live.buf[..upto].to_vec(),
        language.clone(),
        tx.clone(),
        live.epoch,
    );
    tokio::spawn(async move {
        let result = pass(lane, pcm, rate, language, ctx, times).await;
        let _ = tx.send(Pass {
            result,
            cut,
            origin,
            epoch,
        });
    });
}

/// While server VAD is on and nobody is speaking, there is nothing to
/// transcribe.
fn hold(det: &Option<Detector>) -> bool {
    det.as_ref().is_some_and(|d| !d.turns.speaking)
}

/// An absolute sample index as the wire's milliseconds-since-session-start.
pub(crate) fn ms(samples: usize, rate: u32) -> u64 {
    samples as u64 * 1000 / rate.max(1) as u64
}

pub(crate) fn committed_event(item: &str) -> Value {
    json!({
        "type": "input_audio_buffer.committed",
        "event_id": id("event"),
        "item_id": item,
        "previous_item_id": Value::Null,
    })
}

pub(crate) async fn send(socket: &mut WebSocket, ev: Value) -> Result<(), ()> {
    socket
        .send(Message::Text(ev.to_string().into()))
        .await
        .map_err(|_| ())
}

/// `session.created` / `session.updated`. `type: "transcription"` is what
/// picks the transcription variant out of the SDK's session union; `id` and
/// `object` ride along as the API's own extra fields.
#[allow(clippy::too_many_arguments)]
fn session_event(
    kind: &str,
    session_id: &str,
    lane: &Lane,
    rate: u32,
    language: &Option<String>,
    det: Option<&Detector>,
    prompt: &str,
    verbose: bool,
    final_revision: bool,
) -> Value {
    json!({
        "type": kind,
        "event_id": id("event"),
        "session": {
            "id": session_id,
            "object": "realtime.transcription_session",
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": rate},
                    "noise_reduction": Value::Null,
                    // stated, not omitted: a null here is how the API spells
                    // "you drive the turns", and it is still the default
                    "turn_detection": det.map_or(Value::Null, |d| json!({
                        "type": "server_vad",
                        "threshold": d.cfg.threshold,
                        "prefix_padding_ms": d.cfg.prefix_ms,
                        "silence_duration_ms": d.cfg.silence_ms,
                        "idle_timeout_ms": d.cfg.idle_ms,
                    })),
                    "transcription": {
                        "model": lane.model,
                        "language": language,
                        "prompt": prompt,
                        // stated even when false, for the same reason
                        // `turn_detection` is: a session hands back the config
                        // it is actually running, and a caller who asked for
                        // this and reads it back missing cannot tell a refusal
                        // from an old build
                        "paddock_verbose": verbose,
                        "paddock_final_revision": final_revision,
                    },
                },
            },
        },
    })
}

fn delta_event(item: &str, delta: &str, lead_space: bool) -> Value {
    json!({
        "type": "conversation.item.input_audio_transcription.delta",
        "event_id": id("event"),
        "item_id": item,
        "content_index": 0,
        // The word boundary rides the DELTA, not the client's concatenation
        // logic: a consumer appends what it is given, so the space between two
        // committed words has to be in one of them.
        "delta": if lead_space { format!(" {delta}") } else { delta.to_owned() },
    })
}

pub(crate) fn error_event(kind: &str, message: &str) -> Value {
    json!({
        "type": "error",
        "event_id": id("event"),
        "error": {"type": kind, "message": message},
    })
}

/// Apply a `session.update`'s transcription settings, refusing by name what
/// this session does not serve.
fn apply_update(
    session: Option<&Value>,
    rate: &mut u32,
    language: &mut Option<String>,
    vad: &mut Option<VadConfig>,
    asked: &mut String,
    verbose: &mut bool,
) -> Result<(), String> {
    // A refused update must not change sample rate/language/VAD halfway.
    let (mut r, mut l, mut v, mut a, mut b) =
        (*rate, language.clone(), *vad, asked.clone(), *verbose);
    apply_update_inner(session, &mut r, &mut l, &mut v, &mut a, &mut b)?;
    *rate = r;
    *language = l;
    *vad = v;
    *asked = a;
    *verbose = b;
    Ok(())
}

fn apply_update_inner(
    session: Option<&Value>,
    rate: &mut u32,
    language: &mut Option<String>,
    vad: &mut Option<VadConfig>,
    asked: &mut String,
    verbose: &mut bool,
) -> Result<(), String> {
    let Some(s) = session else {
        return Err("`session.update` needs a `session` object".into());
    };
    if let Some(t) = s.get("type").and_then(Value::as_str)
        && t != "transcription"
    {
        return Err(format!(
            "this is a transcription session; `session.type` must be \"transcription\", not {t:?}"
        ));
    }
    let input = s.pointer("/audio/input");
    if let Some(td) = input.and_then(|i| i.get("turn_detection")) {
        *vad = if td.is_null() {
            None
        } else {
            Some(parse_vad(td)?)
        };
    }
    if let Some(nr) = input.and_then(|i| i.get("noise_reduction"))
        && !nr.is_null()
    {
        return Err("`noise_reduction` is not served; send the audio you want transcribed".into());
    }
    if let Some(f) = input.and_then(|i| i.get("format")) {
        match f.get("type").and_then(Value::as_str) {
            Some("audio/pcm") | None => {}
            Some(other) => {
                return Err(format!(
                    "audio format {other:?} is not served - send `audio/pcm` (little-endian \
                     signed 16-bit); the G.711 companding formats have no decoder here"
                ));
            }
        }
        if let Some(r) = f.get("rate").and_then(Value::as_u64) {
            // OpenAI pins this to 24000. We accept any sane rate and resample,
            // because a browser's AudioContext hands out 44.1 or 48 kHz and
            // refusing would make every caller resample in JS instead.
            if !(8000..=192_000).contains(&r) {
                return Err(format!("sample rate {r} is out of range (8000-192000)"));
            }
            *rate = r as u32;
        }
    }
    if let Some(t) = input.and_then(|i| i.get("transcription")) {
        match t.get("prompt") {
            Some(Value::String(p)) => *asked = p.clone(),
            Some(Value::Null) => asked.clear(),
            None => {}
            Some(_) => return Err("`prompt` must be a string".into()),
        }
        match t.get("language") {
            Some(Value::String(l)) if !l.is_empty() => *language = Some(l.clone()),
            Some(Value::Null) => *language = None,
            _ => {}
        }
        // PADDOCK EXTENSION. Opt-in rather than default because it
        // costs a DTW pass per closed utterance, and because the timestamp
        // prompt is what some fine-tunes condition their no-speech refusal on
        //  - turning it on silently would change what a checkpoint
        // says, not just what it reports.
        match t.get("paddock_verbose") {
            Some(Value::Bool(b)) => *verbose = *b,
            None | Some(Value::Null) => {}
            Some(_) => return Err("`paddock_verbose` must be a boolean".into()),
        }
    }
    Ok(())
}

/// One `turn_detection` object. Everything it can hold is either served or
/// refused by name - a client that asks for semantic turn-taking must not be
/// quietly given the energy detector instead.
pub(crate) fn parse_vad(td: &Value) -> Result<VadConfig, String> {
    match td.get("type").and_then(Value::as_str) {
        Some("server_vad") | None => {}
        Some("semantic_vad") => {
            return Err(
                "`semantic_vad` is not served: deciding a turn ended from what was SAID needs a \
                 turn-detection model, and this session has an acoustic detector. Use \
                 `server_vad`, or drive the turns yourself with `input_audio_buffer.commit`."
                    .into(),
            );
        }
        Some(other) => return Err(format!("unknown `turn_detection.type` {other:?}")),
    }
    for k in ["create_response", "interrupt_response"] {
        if td.get(k).and_then(Value::as_bool) == Some(true) {
            return Err(format!(
                "`{k}` belongs to a conversation session; this one transcribes and never responds"
            ));
        }
    }
    let num = |k: &str, d: u32| -> Result<u32, String> {
        match td.get(k) {
            None | Some(Value::Null) => Ok(d),
            Some(v) => v
                .as_u64()
                .filter(|n| *n <= 600_000)
                .map(|n| n as u32)
                .ok_or_else(|| format!("`{k}` must be a duration in milliseconds (0-600000)")),
        }
    };
    let threshold = match td.get("threshold") {
        None | Some(Value::Null) => VAD_THRESHOLD,
        Some(v) => v
            .as_f64()
            .filter(|f| (0.0..=1.0).contains(f))
            .map(|f| f as f32)
            .ok_or("`threshold` must be between 0 and 1")?,
    };
    Ok(VadConfig {
        threshold,
        prefix_ms: num("prefix_padding_ms", VAD_PREFIX_MS)?,
        silence_ms: num("silence_duration_ms", VAD_SILENCE_MS)?,
        idle_ms: match td.get("idle_timeout_ms") {
            None | Some(Value::Null) => None,
            Some(_) => Some(num("idle_timeout_ms", 0)?),
        },
    })
}

/// base64 -> little-endian PCM16 -> f32 in [-1, 1].
pub(crate) fn decode_pcm16(b64: &str) -> Result<Vec<f32>, String> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .map_err(|e| format!("`audio` is not valid base64: {e}"))?;
    if bytes.len() % 2 != 0 {
        return Err("`audio` is not whole 16-bit samples (odd byte count)".into());
    }
    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
        .collect())
}

#[cfg(test)]
mod tests {
    #[test]
    fn final_revision_replaces_stale_prefix_and_keeps_retired_windows() {
        let agreement = super::Agreement {
            retired: vec!["Earlier.".into()],
            committed: vec!["Wrong".into(), "extra".into()],
            ..Default::default()
        };
        assert_eq!(
            agreement.final_text(&["Correct.".into()]),
            "Earlier. Correct."
        );
        assert_eq!(
            agreement.text(),
            "Earlier. Wrong extra",
            "legacy append-only text remains unchanged"
        );
        assert_eq!(agreement.final_text(&[]), "Earlier.");
    }
    #[test]
    fn rejected_update_does_not_half_change_the_audio_format() {
        let mut rate = 16000;
        let mut language = Some("sv".to_owned());
        let mut vad = None;
        let mut asked = String::new();
        let mut verbose = false;
        let update = serde_json::json!({"audio":{"input":{
            "format":{"type":"audio/pcm","rate":48000},
            "transcription":{"language":"en","paddock_verbose":"invalid"}
        }}});
        assert!(
            super::apply_update(
                Some(&update),
                &mut rate,
                &mut language,
                &mut vad,
                &mut asked,
                &mut verbose
            )
            .is_err()
        );
        assert_eq!(rate, 16000);
        assert_eq!(language.as_deref(), Some("sv"));
        assert!(!verbose);
    }
    #[test]
    fn drain_tracks_pending_final_pass_but_not_probe_or_cleared_audio() {
        let mut live = super::Live::new();
        assert!(live.pending_items().is_empty());
        live.inflight = true;
        assert!(
            live.pending_items().is_empty(),
            "a probe alone owes no completed item"
        );
        live.commit = Some(123);
        assert_eq!(live.pending_items(), [live.item.clone()]);
        live.commit = None;
        live.final_inflight = true;
        assert_eq!(live.pending_items(), [live.item.clone()]);
        live.restart();
        assert!(
            live.pending_items().is_empty(),
            "clear invalidates the old final pass"
        );
    }
    #[test]
    fn realtime_capabilities_are_not_file_timestamp_capabilities() {
        let whisper = super::capabilities(true, true, None);
        assert_eq!(whisper["enrichment"], true);
        assert_eq!(whisper["drain"], true);
        assert_eq!(whisper["utterance_max_s"], 600.0);
        // Granite Plus file word times cannot opt a generative live lane into
        // the Whisper-specific pass. Qwen/Granite enforce the tower ceiling.
        let generative = super::capabilities(true, false, Some(120.0));
        assert_eq!(generative["enrichment"], false);
        assert_eq!(generative["supported"], true);
        assert_eq!(generative["utterance_max_s"], 120.0);
        assert_eq!(super::capabilities(false, false, None)["supported"], false);
    }
    use super::*;

    fn words(s: &str) -> Vec<String> {
        s.split_whitespace().map(str::to_owned).collect()
    }

    #[test]
    fn local_agreement_commits_only_what_two_passes_share() {
        let mut a = Agreement::default();
        // first pass has nothing to agree with, so it commits nothing - that is
        // the whole point of the -2 in LocalAgreement-2
        assert!(a.feed(words("the quick brown")).is_empty());
        // second pass agrees on "the quick", diverges after
        assert_eq!(a.feed(words("the quick red fox")), words("the quick"));
        assert_eq!(a.text(), "the quick");
        // third agrees with the second's tail "red fox"
        assert_eq!(a.feed(words("the quick red fox jumps")), words("red fox"));
        assert_eq!(a.text(), "the quick red fox");
    }

    #[test]
    fn a_hypothesis_that_shrinks_never_unsays_a_committed_word() {
        let mut a = Agreement::default();
        a.feed(words("one two three"));
        assert_eq!(a.feed(words("one two three")), words("one two three"));
        // the model changed its mind about audio it already heard; a delta
        // stream cannot take words back, so the commit stands
        assert!(a.feed(words("one")).is_empty());
        assert_eq!(a.text(), "one two three");
    }

    #[test]
    fn silence_then_speech_commits_nothing_early() {
        let mut a = Agreement::default();
        assert!(a.feed(Vec::new()).is_empty());
        assert!(a.feed(Vec::new()).is_empty());
        assert!(a.feed(words("hello")).is_empty());
        assert_eq!(a.feed(words("hello there")), words("hello"));
    }

    #[test]
    fn pcm16_round_trips_through_base64() {
        let src: Vec<i16> = vec![0, 32767, -32768, 1234];
        let raw: Vec<u8> = src.iter().flat_map(|v| v.to_le_bytes()).collect();
        let b64 = base64::engine::general_purpose::STANDARD.encode(&raw);
        let out = decode_pcm16(&b64).expect("decode");
        assert_eq!(out.len(), 4);
        assert!((out[0]).abs() < 1e-6);
        assert!((out[1] - 32767.0 / 32768.0).abs() < 1e-6);
        assert!((out[2] + 1.0).abs() < 1e-6);
        assert!(decode_pcm16("not base64!!").is_err());
        // an odd byte count is a truncated sample, not something to round off
        let odd = base64::engine::general_purpose::STANDARD.encode([1u8, 2, 3]);
        assert!(decode_pcm16(&odd).is_err());
    }

    #[test]
    fn the_update_refuses_what_it_cannot_serve_by_name() {
        let (mut rate, mut lang, mut vad, mut asked, mut verb) =
            (DEFAULT_RATE, None, None, String::new(), false);
        let semantic = json!({"audio": {"input": {"turn_detection": {"type": "semantic_vad"}}}});
        let e = apply_update(
            Some(&semantic),
            &mut rate,
            &mut lang,
            &mut vad,
            &mut asked,
            &mut verb,
        )
        .unwrap_err();
        assert!(
            e.contains("semantic_vad") && e.contains("server_vad"),
            "{e}"
        );
        assert!(vad.is_none(), "a refused update must not half-apply");

        let ulaw = json!({"audio": {"input": {"format": {"type": "audio/pcmu"}}}});
        let e = apply_update(
            Some(&ulaw),
            &mut rate,
            &mut lang,
            &mut vad,
            &mut asked,
            &mut verb,
        )
        .unwrap_err();
        assert!(e.contains("audio/pcmu") || e.contains("pcmu"), "{e}");

        // what it does serve
        let ok = json!({
            "type": "transcription",
            "audio": {"input": {
                "format": {"type": "audio/pcm", "rate": 48000},
                "turn_detection": null,
                "transcription": {"language": "sv"},
            }},
        });
        apply_update(
            Some(&ok),
            &mut rate,
            &mut lang,
            &mut vad,
            &mut asked,
            &mut verb,
        )
        .expect("accepted");
        assert_eq!(rate, 48000);
        assert_eq!(lang.as_deref(), Some("sv"));
        assert!(vad.is_none());
    }

    #[test]
    fn server_vad_takes_the_specs_defaults_and_hands_them_back() {
        let (mut rate, mut lang, mut vad, mut asked, mut verb) =
            (DEFAULT_RATE, None, None, String::new(), false);
        let bare = json!({"audio": {"input": {"turn_detection": {"type": "server_vad"}}}});
        apply_update(
            Some(&bare),
            &mut rate,
            &mut lang,
            &mut vad,
            &mut asked,
            &mut verb,
        )
        .expect("accepted");
        let cfg = vad.expect("enabled");
        assert_eq!(
            (cfg.threshold, cfg.prefix_ms, cfg.silence_ms, cfg.idle_ms),
            (0.5, 300, 500, None),
        );

        // ... and every knob is honoured, including the one that is optional
        let full = json!({"audio": {"input": {"turn_detection": {
            "type": "server_vad",
            "threshold": 0.8,
            "prefix_padding_ms": 120,
            "silence_duration_ms": 900,
            "idle_timeout_ms": 15000,
        }}}});
        apply_update(
            Some(&full),
            &mut rate,
            &mut lang,
            &mut vad,
            &mut asked,
            &mut verb,
        )
        .expect("accepted");
        let cfg = vad.expect("enabled");
        assert_eq!(
            (cfg.threshold, cfg.prefix_ms, cfg.silence_ms, cfg.idle_ms),
            (0.8, 120, 900, Some(15000)),
        );

        // a nonsense threshold is refused rather than clamped: silently
        // serving a different setting than the one asked for is the failure
        // mode the whole file is written against
        let bad = json!({"audio": {"input": {"turn_detection": {
            "type": "server_vad", "threshold": 7,
        }}}});
        let e = apply_update(
            Some(&bad),
            &mut rate,
            &mut lang,
            &mut vad,
            &mut asked,
            &mut verb,
        )
        .unwrap_err();
        assert!(e.contains("threshold"), "{e}");

        // conversation-only knobs are named, not ignored
        let resp = json!({"audio": {"input": {"turn_detection": {
            "type": "server_vad", "create_response": true,
        }}}});
        let e = apply_update(
            Some(&resp),
            &mut rate,
            &mut lang,
            &mut vad,
            &mut asked,
            &mut verb,
        )
        .unwrap_err();
        assert!(e.contains("create_response"), "{e}");
    }

    #[test]
    fn the_enriched_utterance_is_opt_in_and_read_back() {
        let (mut rate, mut lang, mut vad, mut asked, mut verb) =
            (DEFAULT_RATE, None, None, String::new(), false);
        // absent leaves it alone rather than defaulting it off, so an update
        // about something else cannot turn it off behind the caller's back
        let unrelated = json!({"audio": {"input": {"transcription": {"language": "sv"}}}});
        apply_update(
            Some(&unrelated),
            &mut rate,
            &mut lang,
            &mut vad,
            &mut asked,
            &mut verb,
        )
        .expect("accepted");
        assert!(!verb);

        let on = json!({"audio": {"input": {"transcription": {"paddock_verbose": true}}}});
        apply_update(
            Some(&on),
            &mut rate,
            &mut lang,
            &mut vad,
            &mut asked,
            &mut verb,
        )
        .expect("accepted");
        assert!(verb);
        let again = json!({"audio": {"input": {"transcription": {"prompt": "hi"}}}});
        apply_update(
            Some(&again),
            &mut rate,
            &mut lang,
            &mut vad,
            &mut asked,
            &mut verb,
        )
        .expect("accepted");
        assert!(verb, "an unrelated update must not clear it");

        let off = json!({"audio": {"input": {"transcription": {"paddock_verbose": false}}}});
        apply_update(
            Some(&off),
            &mut rate,
            &mut lang,
            &mut vad,
            &mut asked,
            &mut verb,
        )
        .expect("accepted");
        assert!(!verb);

        // refused rather than coerced, same as every other knob here
        let junk = json!({"audio": {"input": {"transcription": {"paddock_verbose": "yes"}}}});
        let e = apply_update(
            Some(&junk),
            &mut rate,
            &mut lang,
            &mut vad,
            &mut asked,
            &mut verb,
        )
        .unwrap_err();
        assert!(e.contains("paddock_verbose"), "{e}");
    }

    #[test]
    fn word_timing_is_paid_at_the_boundary_and_timestamps_are_not() {
        // The shape of the decision `kick` makes, asserted here because it is
        // the whole cost argument for: the DTW pass rides the FINAL
        // pass only, while the timestamp prompt is constant for the session so
        // LocalAgreement never compares one prompt's hypothesis against
        // another's.
        let times =
            |verbose: bool, cut: Option<usize>| Times::for_pass(verbose, cut.is_some(), true);
        let off = times(false, Some(16000));
        assert!(
            !off.segments && !off.words,
            "a session that did not ask pays nothing"
        );
        let hyp = times(true, None);
        assert!(
            hyp.segments && !hyp.words,
            "a hypothesis pass never runs the alignment"
        );
        let fin = times(true, Some(16000));
        assert!(
            fin.segments && fin.words,
            "the closed utterance is where it is paid"
        );
        assert!(fin.verbose);
        let metal = Times::for_pass(true, true, false);
        assert!(
            metal.segments && metal.verbose && !metal.words,
            "a segment-only backend must finalize without unavailable alignment"
        );
        let metal_hypothesis = Times::for_pass(true, false, false);
        assert!(metal_hypothesis.segments && !metal_hypothesis.verbose && !metal_hypothesis.words);
    }

    /// Frames at 20 ms, `n` of them, all speech or all not.
    fn frames(at: usize, n: usize, speech: bool) -> Vec<paddock_engine::audio::vad::Frame> {
        (0..n)
            .map(|i| paddock_engine::audio::vad::Frame {
                start: at + i * 320,
                end: at + (i + 1) * 320,
                speech,
                db: if speech { -20.0 } else { -70.0 },
            })
            .collect()
    }

    fn cfg(silence_ms: u32, idle_ms: Option<u32>) -> VadConfig {
        VadConfig {
            threshold: 0.5,
            prefix_ms: 300,
            silence_ms,
            idle_ms,
        }
    }

    #[test]
    fn a_turn_opens_on_a_run_of_speech_and_is_back_dated_to_its_start() {
        let c = cfg(500, None);
        let mut t = Turns::new(&c, 16000, 0);
        // two frames is not a turn yet - a click must not open one
        assert!(t.feed(&frames(0, 2, true)).is_empty());
        // the third confirms it, and the start is the first frame of the run
        assert_eq!(t.feed(&frames(640, 1, true)), vec![Turn::Started { at: 0 }]);
        assert!(t.speaking);
    }

    #[test]
    fn a_turn_ends_where_the_speech_did_not_where_the_silence_was_noticed() {
        let c = cfg(500, None);
        let mut t = Turns::new(&c, 16000, 0);
        t.feed(&frames(0, 3, true));
        // 500 ms of silence at 16 kHz is 8000 samples = 25 frames; the turn
        // ends at the last speech frame, not 500 ms later
        let out = t.feed(&frames(960, 25, false));
        assert_eq!(out, vec![Turn::Stopped { at: 960 }]);
        assert!(!t.speaking);
    }

    #[test]
    fn a_gap_shorter_than_the_silence_window_is_a_pause_not_a_turn_end() {
        let c = cfg(500, None);
        let mut t = Turns::new(&c, 16000, 0);
        t.feed(&frames(0, 3, true));
        // 400 ms of breath, then more words: one turn, not two
        assert!(t.feed(&frames(960, 20, false)).is_empty());
        assert!(t.feed(&frames(7360, 5, true)).is_empty());
        assert!(t.speaking);
    }

    #[test]
    fn the_idle_timeout_announces_a_room_where_nobody_spoke() {
        let c = cfg(500, Some(1000));
        let mut t = Turns::new(&c, 16000, 0);
        // 1 s at 16 kHz is 50 frames; the announcement lands on the frame that
        // crosses it and the clock restarts, so it repeats rather than fires
        // once and goes quiet
        let out = t.feed(&frames(0, 50, false));
        assert_eq!(out, vec![Turn::Idle { from: 0, to: 16000 }]);
        let out = t.feed(&frames(16000, 50, false));
        assert_eq!(
            out,
            vec![Turn::Idle {
                from: 16000,
                to: 32000
            }]
        );
    }

    #[test]
    fn a_hand_commit_ends_the_turn_the_detector_thought_was_running() {
        let c = cfg(500, None);
        let mut t = Turns::new(&c, 16000, 0);
        t.feed(&frames(0, 3, true));
        assert!(t.speaking);
        t.yield_to_client(960);
        assert!(!t.speaking);
        // and the next run of speech opens a new turn rather than resuming
        assert_eq!(
            t.feed(&frames(960, 3, true)),
            vec![Turn::Started { at: 960 }]
        );
    }

    #[test]
    fn retiring_a_finished_window_rebases_the_agreement_onto_the_shorter_buffer() {
        let mut a = Agreement::default();
        a.feed(words("the quick brown fox jumps"));
        assert_eq!(
            a.feed(words("the quick brown fox jumps over")),
            words("the quick brown fox jumps")
        );
        // the first three words' audio is leaving; they were committed, so
        // nothing new goes out, and what is left still lines up with the
        // hypothesis the next pass will produce
        let hyp = words("the quick brown fox jumps over");
        assert!(a.retire(&hyp, 3).is_empty());
        assert_eq!(a.committed, words("fox jumps"));
        assert_eq!(a.text(), "the quick brown fox jumps");
        // the next pass sees only the trimmed buffer, and agreement carries on
        assert_eq!(a.feed(words("fox jumps over the")), words("over"));
        assert_eq!(a.text(), "the quick brown fox jumps over");
    }

    #[test]
    fn retiring_past_the_commit_point_sends_the_words_it_takes_with_it() {
        let mut a = Agreement::default();
        a.feed(words("one two three four"));
        // only "one two" ever got a second opinion
        assert_eq!(a.feed(words("one two zebra")), words("one two"));
        // ... but the audio behind "one two zebra" is going, so "zebra" can
        // never be revised again: it goes out now rather than vanishing
        let hyp = words("one two zebra");
        assert_eq!(a.retire(&hyp, 3), words("zebra"));
        assert_eq!(a.text(), "one two zebra");
        assert!(a.committed.is_empty(), "everything retired");
    }

    #[test]
    fn dropping_the_front_keeps_absolute_positions_honest() {
        let mut live = Live::new();
        live.buf = vec![0.0; 1000];
        live.seen = 800;
        live.drop_front(300);
        assert_eq!((live.buf.len(), live.origin, live.seen), (700, 300, 500));
        assert_eq!(
            live.head(),
            1000,
            "the head is where the audio ends, always"
        );
        // dropping more than there is drops what there is
        live.drop_front(9999);
        assert_eq!((live.buf.len(), live.origin, live.seen), (0, 1000, 0));
    }
}
