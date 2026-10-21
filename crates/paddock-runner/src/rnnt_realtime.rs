//! The realtime transcription session (`GET /v1/realtime?intent=transcription`)
//! on the native streaming RNN-T lane. Same wire as the agreement lanes
//! (realtime.rs) - `session.update`, `input_audio_buffer.append` / `.commit`
//! / `.clear` / `.drain`, `server_vad` turns - with none of their policy: the
//! model is cache-aware, so audio goes to it as it arrives, and every token
//! it emits is final the moment it is emitted. A delta is new tokens, never a
//! guess that a later pass may revise; `completed` is the concatenation of
//! the deltas by construction.
//!
//! An utterance (item) is one recognizer stream: it opens on the first audio
//! (or the detector's speech onset, with its pre-roll), receives every
//! append, and is finished by a commit, a drain or the detector's turn end -
//! the model's last chunk then sees the recording's end, as a file's does.

use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket};
use paddock_engine::audio::resample::resample;
use paddock_engine::nemotron_asr::{RecognizerSession, Update};
use paddock_models::nemotron_asr as arch;
use serde_json::{Value, json};

use crate::realtime::{
    DEFAULT_RATE, Detector, Turn, VadConfig, committed_event, decode_pcm16, error_event, id, ms,
    parse_vad, send,
};
use crate::routes::AppState;

/// The attention chunk a live session starts at: 320 ms, the model card's
/// streaming default.
const DEFAULT_LATENCY_MS: u32 = 320;

/// Incremental resampling to 16 kHz that reproduces the whole-recording
/// resample sample for sample: an output is released once its filter
/// support lies inside what has arrived, computed over a window that starts
/// on a whole input period (so the polyphase phases line up), and the window
/// keeps only the support the next output reaches back to.
struct StreamResampler {
    from: u32,
    p: usize,
    q: usize,
    margin: usize,
    buf: Vec<f32>,
    /// absolute input index of buf[0], a multiple of q
    base: usize,
    /// next absolute output index to release
    next: usize,
}

impl StreamResampler {
    fn new(from: u32) -> Self {
        let to = arch::SAMPLE_RATE;
        let g = gcd(to, from as usize);
        let (p, q) = (to / g, from as usize / g);
        let stretch = (q as f64 / p as f64).max(1.0);
        // the resampler's half-width (32 lobes scaled by the decimation)
        let margin = (32.0 * stretch).ceil() as usize + 2;
        Self {
            from,
            p,
            q,
            margin,
            buf: Vec::new(),
            base: 0,
            next: 0,
        }
    }

    fn release(&mut self, last: bool) -> Result<Vec<f32>, String> {
        if self.from as usize == arch::SAMPLE_RATE {
            return Ok(std::mem::take(&mut self.buf));
        }
        if self.buf.is_empty() {
            return Ok(Vec::new());
        }
        let y = resample(&self.buf, self.from, arch::SAMPLE_RATE as u32)?;
        let off = self.base / self.q * self.p;
        let end_in = self.base + self.buf.len();
        let stop = if last {
            off + y.len()
        } else {
            // n's support ends near floor(n q / p) + margin
            let full = end_in.saturating_sub(self.margin);
            (full * self.p / self.q).min(off + y.len())
        };
        let out: Vec<f32> = (self.next.max(off)..stop).map(|n| y[n - off]).collect();
        self.next = self.next.max(stop);
        // keep what the next output's support reaches back to
        let keep = (self.next * self.q / self.p).saturating_sub(self.margin) / self.q * self.q;
        if keep > self.base {
            self.buf.drain(..(keep - self.base).min(self.buf.len()));
            self.base = keep;
        }
        Ok(out)
    }

    fn push(&mut self, x: &[f32]) -> Result<Vec<f32>, String> {
        self.buf.extend_from_slice(x);
        self.release(false)
    }

    fn finish(&mut self) -> Result<Vec<f32>, String> {
        self.release(true)
    }
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// One utterance in flight.
struct Item {
    id: String,
    stream: RecognizerSession,
    rs: StreamResampler,
    tokens: Vec<u32>,
    frames: Vec<u32>,
    shown: String,
    /// absolute input sample its first audio is
    origin: usize,
    /// input samples it has received
    samples: usize,
}

struct Settings {
    rate: u32,
    language: Option<String>,
    latency_ms: u32,
    verbose: bool,
    vad: Option<VadConfig>,
}

fn session_event(kind: &str, session_id: &str, model: &str, s: &Settings) -> Value {
    json!({
        "type": kind,
        "event_id": id("event"),
        "session": {
            "id": session_id,
            "object": "realtime.transcription_session",
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": s.rate},
                    "noise_reduction": Value::Null,
                    "turn_detection": s.vad.map_or(Value::Null, |d| json!({
                        "type": "server_vad",
                        "threshold": d.threshold,
                        "prefix_padding_ms": d.prefix_ms,
                        "silence_duration_ms": d.silence_ms,
                        "idle_timeout_ms": d.idle_ms,
                    })),
                    "transcription": {
                        "model": model,
                        "language": s.language,
                        "prompt": "",
                        "paddock_verbose": s.verbose,
                        "paddock_latency_ms": s.latency_ms,
                        // a native stream never revises: `completed` IS the
                        // deltas, so there is no final revision to opt into
                        "paddock_final_revision": false,
                    },
                },
            },
        },
    })
}

/// Apply a `session.update`, refusing by name what this session does not
/// serve; a refused update changes nothing.
fn apply_update(
    model: &crate::rnnt_asr::RnntAsrModel,
    ev: &Value,
    s: &mut Settings,
) -> Result<(), String> {
    let Some(sess) = ev.get("session") else {
        return Err("`session.update` needs a `session` object".into());
    };
    if let Some(t) = sess.get("type").and_then(Value::as_str)
        && t != "transcription"
    {
        return Err(format!(
            "this is a transcription session; `session.type` must be \"transcription\", not {t:?}"
        ));
    }
    let mut n = Settings {
        rate: s.rate,
        language: s.language.clone(),
        latency_ms: s.latency_ms,
        verbose: s.verbose,
        vad: s.vad,
    };
    let input = sess.pointer("/audio/input");
    if let Some(td) = input.and_then(|i| i.get("turn_detection")) {
        n.vad = if td.is_null() {
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
                    "audio format {other:?} is not served - send `audio/pcm` (little-endian signed 16-bit)"
                ));
            }
        }
        if let Some(r) = f.get("rate").and_then(Value::as_u64) {
            if !(8000..=192_000).contains(&r) {
                return Err(format!("sample rate {r} is out of range (8000-192000)"));
            }
            n.rate = r as u32;
        }
    }
    if let Some(t) = input.and_then(|i| i.get("transcription")) {
        if let Some(Value::String(p)) = t.get("prompt")
            && !p.trim().is_empty()
        {
            return Err(
                "`prompt`: this model takes no text context - its only conditioning is the \
                 language prompt, selected by `language`"
                    .into(),
            );
        }
        match t.get("language") {
            Some(Value::String(l)) if !l.is_empty() => {
                model.prompt_for(Some(l))?;
                n.language = Some(l.clone());
            }
            Some(Value::Null) => n.language = None,
            _ => {}
        }
        match t.get("paddock_latency_ms") {
            None | Some(Value::Null) => {}
            Some(v) => {
                let ms = v.as_u64().unwrap_or(0) as u32;
                if crate::rnnt_asr::latency_lookahead(ms).is_none() {
                    return Err(format!(
                        "paddock_latency_ms {v}: the model publishes 80, 160, 320, 560 and 1120"
                    ));
                }
                n.latency_ms = ms;
            }
        }
        match t.get("paddock_verbose") {
            Some(Value::Bool(b)) => n.verbose = *b,
            None | Some(Value::Null) => {}
            Some(_) => return Err("`paddock_verbose` must be a boolean".into()),
        }
    }
    *s = n;
    Ok(())
}

pub(crate) async fn run(mut socket: WebSocket, state: Arc<AppState>) {
    let Some(model) = state.rnnt_asr.as_ref() else {
        return;
    };
    let vocab = model.recognizer.facts().vocab.clone();
    let session_id = id("sess");
    let mut s = Settings {
        rate: DEFAULT_RATE,
        language: None,
        latency_ms: DEFAULT_LATENCY_MS,
        verbose: false,
        vad: None,
    };
    let mut det: Option<Detector> = None;
    let mut item: Option<Item> = None;
    // input samples seen this session (the detector's and the events' space)
    let mut head = 0usize;
    // with a detector: the pre-roll an onset carries into its utterance
    let mut pre: Vec<f32> = Vec::new();
    if send(
        &mut socket,
        session_event("session.created", &session_id, &model.id, &s),
    )
    .await
    .is_err()
    {
        return;
    }
    // a delta for whatever text an update completed
    async fn deltas(
        socket: &mut WebSocket,
        it: &mut Item,
        vocab: &[String],
        u: Update,
    ) -> Result<(), ()> {
        it.tokens.extend(u.tokens);
        it.frames.extend(u.frames);
        let text = arch::detokenize(vocab, &it.tokens, false);
        if text.len() > it.shown.len() && text.starts_with(&it.shown) {
            let delta = text[it.shown.len()..].to_owned();
            it.shown = text;
            send(
                socket,
                json!({
                    "type": "conversation.item.input_audio_transcription.delta",
                    "event_id": id("event"),
                    "item_id": it.id,
                    "content_index": 0,
                    "delta": delta,
                }),
            )
            .await?;
        }
        Ok(())
    }
    loop {
        let Some(Ok(msg)) = socket.recv().await else {
            return;
        };
        let text = match msg {
            Message::Text(t) => t.to_string(),
            Message::Binary(_) => {
                let e = "binary frames are not part of the realtime protocol; send audio as base64 in \
                         `input_audio_buffer.append`";
                if send(&mut socket, error_event("invalid_request_error", e))
                    .await
                    .is_err()
                {
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
                if send(
                    &mut socket,
                    error_event("invalid_request_error", &e.to_string()),
                )
                .await
                .is_err()
                {
                    return;
                }
                continue;
            }
        };
        match ev.get("type").and_then(Value::as_str).unwrap_or_default() {
            "session.update" => match apply_update(model, &ev, &mut s) {
                Ok(()) => {
                    det = s.vad.map(|cfg| Detector::new(cfg, s.rate, head));
                    if send(
                        &mut socket,
                        session_event("session.updated", &session_id, &model.id, &s),
                    )
                    .await
                    .is_err()
                    {
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
            },
            "input_audio_buffer.append" => {
                let b64 = ev.get("audio").and_then(Value::as_str).unwrap_or_default();
                let pcm = match decode_pcm16(b64) {
                    Ok(p) => p,
                    Err(e) => {
                        if send(&mut socket, error_event("invalid_request_error", &e))
                            .await
                            .is_err()
                        {
                            return;
                        }
                        continue;
                    }
                };
                let turns = det.as_mut().map(|d| d.feed(&pcm)).unwrap_or_default();
                let start = head;
                head += pcm.len();
                let mut ending = false;
                for t in turns {
                    let out = match t {
                        Turn::Started { at } => {
                            if item.is_none() {
                                match open(model, &s, at.saturating_sub(pre.len().min(at))).await {
                                    Ok(mut it) => {
                                        // the pre-roll goes first
                                        let roll = std::mem::take(&mut pre);
                                        it.samples += roll.len();
                                        let r = it.rs.push(&roll);
                                        match r {
                                            Ok(x) if !x.is_empty() => {
                                                if let Ok(u) = it.stream.feed(x, false).await
                                                    && deltas(&mut socket, &mut it, &vocab, u)
                                                        .await
                                                        .is_err()
                                                {
                                                    return;
                                                }
                                            }
                                            _ => {}
                                        }
                                        item = Some(it);
                                    }
                                    Err(e) => {
                                        if send(&mut socket, error_event("server_error", &e))
                                            .await
                                            .is_err()
                                        {
                                            return;
                                        }
                                    }
                                }
                            }
                            json!({
                                "type": "input_audio_buffer.speech_started",
                                "event_id": id("event"),
                                "item_id": item.as_ref().map(|i| i.id.clone()),
                                "audio_start_ms": ms(at, s.rate),
                            })
                        }
                        Turn::Stopped { at } => {
                            ending = item.is_some();
                            json!({
                                "type": "input_audio_buffer.speech_stopped",
                                "event_id": id("event"),
                                "item_id": item.as_ref().map(|i| i.id.clone()),
                                "audio_end_ms": ms(at, s.rate),
                            })
                        }
                        Turn::Idle { from, to } => json!({
                            "type": "input_audio_buffer.timeout_triggered",
                            "event_id": id("event"),
                            "item_id": item.as_ref().map(|i| i.id.clone()),
                            "audio_start_ms": ms(from, s.rate),
                            "audio_end_ms": ms(to, s.rate),
                        }),
                    };
                    if send(&mut socket, out).await.is_err() {
                        return;
                    }
                }
                // without a detector every append is speech
                if det.is_none() && item.is_none() {
                    match open(model, &s, start).await {
                        Ok(it) => item = Some(it),
                        Err(e) => {
                            if send(&mut socket, error_event("server_error", &e))
                                .await
                                .is_err()
                            {
                                return;
                            }
                            continue;
                        }
                    }
                }
                match item.as_mut() {
                    Some(it) => {
                        it.samples += pcm.len();
                        let x = match it.rs.push(&pcm) {
                            Ok(x) => x,
                            Err(e) => {
                                if send(&mut socket, error_event("server_error", &e))
                                    .await
                                    .is_err()
                                {
                                    return;
                                }
                                continue;
                            }
                        };
                        if !x.is_empty() {
                            match it.stream.feed(x, false).await {
                                Ok(u) => {
                                    if deltas(&mut socket, it, &vocab, u).await.is_err() {
                                        return;
                                    }
                                }
                                Err(e) => {
                                    if send(
                                        &mut socket,
                                        error_event("server_error", &e.to_string()),
                                    )
                                    .await
                                    .is_err()
                                    {
                                        return;
                                    }
                                    item = None;
                                    continue;
                                }
                            }
                        }
                    }
                    None => {
                        // silence between turns: keep the pre-roll only
                        let keep = det.as_ref().map_or(0, |d| d.cfg.prefix(s.rate));
                        pre.extend_from_slice(&pcm);
                        if pre.len() > keep {
                            pre.drain(..pre.len() - keep);
                        }
                    }
                }
                if ending && let Some(it) = item.take() {
                    if send(&mut socket, committed_event(&it.id)).await.is_err() {
                        return;
                    }
                    if complete(&mut socket, it, &vocab, &s).await.is_err() {
                        return;
                    }
                }
            }
            "input_audio_buffer.commit" | "input_audio_buffer.drain" => {
                let drain =
                    ev.get("type").and_then(Value::as_str) == Some("input_audio_buffer.drain");
                let mut pending = Vec::new();
                match item.take() {
                    Some(it) => {
                        if let Some(d) = det.as_mut() {
                            d.turns.yield_to_client(head);
                        }
                        if send(&mut socket, committed_event(&it.id)).await.is_err() {
                            return;
                        }
                        pending.push(it.id.clone());
                        if drain
                            && send(&mut socket, drained(head, s.rate, &pending))
                                .await
                                .is_err()
                        {
                            return;
                        }
                        if complete(&mut socket, it, &vocab, &s).await.is_err() {
                            return;
                        }
                    }
                    None if drain => {
                        if send(&mut socket, drained(head, s.rate, &pending))
                            .await
                            .is_err()
                        {
                            return;
                        }
                    }
                    None => {
                        let e = "the input audio buffer is empty - nothing to commit";
                        if send(&mut socket, error_event("invalid_request_error", e))
                            .await
                            .is_err()
                        {
                            return;
                        }
                    }
                }
            }
            "input_audio_buffer.clear" => {
                item = None;
                pre.clear();
                if send(
                    &mut socket,
                    json!({"type": "input_audio_buffer.cleared", "event_id": id("event")}),
                )
                .await
                .is_err()
                {
                    return;
                }
            }
            other => {
                let e = format!("unknown client event type {other:?}");
                if send(&mut socket, error_event("invalid_request_error", &e))
                    .await
                    .is_err()
                {
                    return;
                }
            }
        }
    }
}

/// Open an utterance's recognizer stream at the session's settings.
async fn open(
    model: &crate::rnnt_asr::RnntAsrModel,
    s: &Settings,
    origin: usize,
) -> Result<Item, String> {
    let lookahead =
        crate::rnnt_asr::latency_lookahead(s.latency_ms).unwrap_or(arch::DEFAULT_LOOKAHEAD);
    let prompt = model.prompt_for(s.language.as_deref())?;
    let stream = model
        .recognizer
        .session(lookahead, prompt)
        .await
        .map_err(|e| e.to_string())?;
    Ok(Item {
        id: id("item"),
        stream,
        rs: StreamResampler::new(s.rate),
        tokens: Vec::new(),
        frames: Vec::new(),
        shown: String::new(),
        origin,
        samples: 0,
    })
}

fn drained(head: usize, rate: u32, pending: &[String]) -> Value {
    json!({
        "type": "input_audio_buffer.drained",
        "event_id": id("event"),
        "audio_end_ms": ms(head, rate),
        "pending_item_ids": pending,
    })
}

/// Finish an utterance: its last chunk sees the recording's end, the final
/// tokens go out as one last delta, then `completed`.
async fn complete(
    socket: &mut WebSocket,
    mut it: Item,
    vocab: &[String],
    s: &Settings,
) -> Result<(), ()> {
    let rest = it.rs.finish().unwrap_or_default();
    match it.stream.feed(rest, true).await {
        Ok(u) => {
            it.tokens.extend(u.tokens);
            it.frames.extend(u.frames);
        }
        Err(e) => {
            send(socket, error_event("server_error", &e.to_string())).await?;
        }
    }
    let t = crate::rnnt_asr::transcript(vocab, &it.tokens, &it.frames);
    if t.text.len() > it.shown.len() && t.text.starts_with(&it.shown) {
        send(
            socket,
            json!({
                "type": "conversation.item.input_audio_transcription.delta",
                "event_id": id("event"),
                "item_id": it.id,
                "content_index": 0,
                "delta": &t.text[it.shown.len()..],
            }),
        )
        .await?;
    }
    let seconds = it.samples as f64 / s.rate.max(1) as f64;
    let mut done = json!({
        "type": "conversation.item.input_audio_transcription.completed",
        "event_id": id("event"),
        "item_id": it.id,
        "content_index": 0,
        "transcript": t.text,
        "paddock_revised": false,
        "usage": {"type": "duration", "seconds": seconds},
        "paddock_audio_start_ms": ms(it.origin, s.rate),
        "paddock_language": t.language,
    });
    if s.verbose {
        done["paddock_verbose"] = crate::rnnt_asr::verbose(&t, seconds, true, true);
    }
    send(socket, done).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_resampling_matches_the_whole_recording() {
        // a chirp at 24 kHz, fed in uneven pieces
        let x: Vec<f32> = (0..24_000 * 3)
            .map(|i| {
                let t = i as f32 / 24_000.0;
                (2.0 * std::f32::consts::PI * (200.0 + 900.0 * t) * t).sin() * 0.5
            })
            .collect();
        let whole = resample(&x, 24_000, 16_000).unwrap();
        for piece in [1, 977, 2_400, 24_001] {
            let mut r = StreamResampler::new(24_000);
            let mut out = Vec::new();
            for c in x.chunks(piece) {
                out.extend(r.push(c).unwrap());
            }
            out.extend(r.finish().unwrap());
            assert_eq!(out.len(), whole.len(), "piece {piece}");
            assert!(
                out.iter()
                    .zip(&whole)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "piece {piece}: not the whole recording's samples"
            );
        }
        // 16 kHz passes through
        let mut r = StreamResampler::new(16_000);
        assert_eq!(r.push(&[0.5, 0.25]).unwrap(), vec![0.5, 0.25]);
    }
}
