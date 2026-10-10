//! `POST /v1/audio/transcriptions` - OpenAI-compatible speech-to-text.
//! multipart/form-data: `file` (any container OpenAI accepts - see
//! `paddock_engine::audio::decode`), `model`, `response_format` (json |
//! text | verbose_json | srt | vtt), optional `language` (ISO 639-1),
//! `prompt`, `temperature` (default 0 = greedy),
//! `timestamp_granularities[]`. Spec parameters we do not serve refuse by
//! name; anything not in the spec at all is an unknown field and refused
//! the same way the JSON lanes refuse one. Every disposition here is
//! pinned by tests/spec/coverage.json (`audio_transcriptions`).
//!
//! Three lanes serve this one wire contract, and the form is parsed once so
//! the surface is identical whichever is loaded: the whisper family (an
//! encoder-decoder on its own thread), Qwen3-ASR, and granite-speech (both
//! generative - an audio tower feeding an ordinary LLM through the
//! multimodal splice).
//!
//! ## Qwen3-ASR
//!
//! Prompt construction mirrors vLLM's `get_generation_prompt` for this
//! model (the binding rival - same scaffolding, same forced-language
//! assistant prefix, same sanitizer semantics):
//!
//! ```text
//! [<|im_start|>system\n{prompt}<|im_end|>\n]          # only when given
//! <|im_start|>user\n<|audio_start|>...audio...<|audio_end|><|im_end|>\n
//! <|im_start|>assistant\n[language {Lang}<asr_text>]  # when language forced
//! ```
//!
//! The model emits `language {X}<asr_text>{transcript}`; the response strips
//! that envelope (and reports the detected language on `verbose_json`).
//!
//! ## granite-speech
//!
//! The checkpoint's own template, which is a plain `USER: ... ASSISTANT:`
//! envelope, with the audio marker first:
//!
//! ```text
//! USER: ...audio...{instruction}
//!  ASSISTANT:
//! ```
//!
//! The INSTRUCTION selects the task on this family (raw vs punctuated
//! transcript, keyword biasing, translation - the model card publishes the
//! set), so `prompt` carries it verbatim rather than being system context.
//! That is also how IBM's own llama-server example drives this endpoint.
//! `language` is an input HINT under the OpenAI spec and this model detects
//! the input language itself, so it does not enter the prompt; ask for
//! translation through `prompt` (`translate the speech to German.`).
//!
//! The audio itself rides `MmChunk::Audio` (16 kHz mono f32 - decoded and
//! resampled here); the engine's tower expands it to its token rows, so the
//! runner never has to agree with the engine about the audio token count.
//!
//! ## Timestamps
//!
//! `timestamp_granularities[]` selects them. Asking a model that cannot answer
//! one gets a refusal naming the model, never an empty `segments` array - and
//! which it can answer is a per-checkpoint fact the loader reads out of the
//! GGUF, published on `/v1/models` so a client never has to learn it from a
//! 400.
//!
//! Three unrelated mechanisms sit behind two granularity names (
//! landed the last two):
//!
//!   * `segment`, whisper - its own timestamp VOCABULARY. Asking for it CHANGES
//!     the DECODE: the prompt drops `<|notimestamps|>`, which is how the model
//!     is told to emit times, and the logits are constrained by whisper's
//!     timestamp grammar. That is why it is opt-in rather than always on - a
//!     plain transcription keeps the exact prompt every WER gate was measured
//!     against.
//!   * `word`, whisper - cross-attention DTW over a SECOND, teacher-forced pass
//!     per window, which is where OpenAI's "word timestamps incur additional
//!     latency" comes from. It does not change the transcript at all: the pass
//!     re-runs the tokens already decoded, under the canonical alignment
//!     prompt.
//!   * `word`, granite-speech-plus - the model is ASKED for the times and
//!     writes them into its answer as `[T:N]` tags, which the runner parses back
//!     out. So here the granularity does change the transcript, and visibly: a
//!     different instruction is a different task on this family, and IBM's card
//!     is explicit that the timestamp mode drops punctuation and capitalization.
//!     `segment` stays refused on it for that same reason - no punctuation means
//!     no sentence boundaries to cut cues on.
//!
//! On whisper the two granularities compose, giving OpenAI's own shape: a
//! top-level `words[]` of `{word, start, end}` plus `segments[]`. Everywhere,
//! the words are cut out of the same grouping the per-word confidences use, so
//! a word's time and its confidence can never be about different words.
//!
//! ## Streaming
//!
//! `stream=true` answers with OpenAI's `transcript.text.delta` events and one
//! `transcript.text.done`, on all three lanes. See the streaming section
//! below for the invariant it holds and why each piece is there. `srt`/`vtt`
//! refuse to stream - a partial subtitle document is not a partial answer.

/// granite-speech's default instruction: the model card's headline prompt,
/// and the one IBM's llama-server example passes. Used when the caller sends
/// no `prompt`.
const GRANITE_SPEECH_PROMPT: &str =
    "transcribe the speech with proper punctuation and capitalization.";

/// granite-speech-plus's word-timestamp instruction, verbatim from IBM's model
/// card (`TS_PROMPT`) minus the `<|audio|>` marker our envelope supplies
/// separately. Sent instead of the default when `word` granularity is asked
/// for - on this family the instruction is the task selector, so this is not
/// an addition to the prompt, it replaces it.
const GRANITE_SPEECH_TS_PROMPT: &str = "Timestamps: Transcribe the speech. After each word, add a \
     timestamp tag showing the end time in centiseconds, e.g. hello [T:45] world [T:82]";

/// How eager the window gate is, on `audio::vad`'s 0..1 scale.
///
/// Deliberately LOW. The two mistakes do not cost the same: a missed silent
/// window costs one wasted encoder pass, a wrongly-gated speech window costs
/// the user their words. 0.15 sits well under the live socket's own 0.5
/// turn-detection setting for exactly that reason - the question here is "is
/// there anything at all in this span", not "is someone talking to me now".
const VAD_GATE_THRESHOLD: f32 = 0.15;

/// What `language` says when nothing on this lane can answer it.
///
/// OpenAI types `TranscriptionVerbose.language` as a required string, so a null
/// there is not a modest answer - it is a body the official SDK raises on, and
/// "any OpenAI client just works" is the bar. granite-speech detects the input
/// language and does not report it, so when the caller sent no hint either
/// there is genuinely nothing to say; this says that, in a word no language is
/// named. Found by pointing the official SDK at this lane for the first time.
/// The field's remaining wart: it is an ISO code on whisper and an English
/// NAME on Qwen3-ASR.
const UNKNOWN_LANGUAGE: &str = "unknown";

/// What granite-speech-plus transcribes a SILENCE as. It carries a timestamp
/// tag like any word, which is the whole point of it: the gaps keep the tag
/// stream dense so consecutive times stay inside one wrap of the model's
/// three-digit clock. It is not a word and never reaches the transcript.
const GRANITE_SILENCE: &str = "_";

use std::sync::Arc;

use axum::extract::{Multipart, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use paddock_api::ErrorBody;
use paddock_engine::audio::guards::{self, Repetition};
use paddock_engine::audio::{decode::decode_audio, resample::resample};
use paddock_engine::sampler::SamplingParams;
use paddock_engine::service::{GenRequest, MmChunk, TokenEvent};
use tokio::sync::mpsc::unbounded_channel;

use crate::routes::AppState;

fn err(status: StatusCode, kind: &str, msg: impl Into<String>) -> Response {
    (status, Json(ErrorBody::new(kind, msg))).into_response()
}

use crate::language::{LanguageReport, Source, language_name};

/// Strip ChatML-like control tokens (`<|...|>`) and the `<asr_text>` delimiter
/// from user-controlled text, to a FIXPOINT - a single pass would let nested
/// payloads reconstruct a valid token (vLLM's sanitizer semantics).
fn sanitize_user_text(text: &str) -> String {
    let mut s = text.to_owned();
    loop {
        let before = s.clone();
        // shortest `<|...|>` spans without an inner '<'
        let mut out = String::with_capacity(s.len());
        let bytes = s.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i..].starts_with(b"<|")
                && let Some(end) = s[i + 2..].find("|>")
                && !s[i + 2..i + 2 + end].contains('<')
            {
                i += 2 + end + 2;
                continue;
            }
            let ch_len = s[i..].chars().next().map(char::len_utf8).unwrap_or(1);
            out.push_str(&s[i..i + ch_len]);
            i += ch_len;
        }
        s = out.replace("<asr_text>", "");
        if s == before {
            return s;
        }
    }
}

/// Whisper's `compression_ratio`: raw bytes over zlib-compressed bytes. It is
/// the reference's own repetition detector - a segment stuck in a loop
/// compresses far better than speech, and OpenAI's decoder rejects a window
/// above 2.4. faster-whisper acts on it too, at the same 2.4.
///
/// We do not, and that is a measurement rather than an argument. Over five
/// checkpoints, clean speech tops out at 1.93 - and the one real hallucination
/// the sweep captured, roest-v3 on quiet room tone, scored **2.13**. A 2.4
/// threshold would have MISSED it while leaving less headroom on clean speech
/// (+0.47 on granite) than the entropy test already has (+0.64). The
/// separation exists (2.13 vs 1.93) but the constant is in the wrong place for
/// our checkpoints, and 0.20 on a single sample is not something to ship blind.
///
/// So it stays reported, and the caller can apply their own threshold - which
/// they can now actually do, since `paddock_guard_stats` puts all four signals
/// on the wire instead of this one. Revisit with a real hallucination set
/// (HALAS) behind it.
fn compression_ratio(text: &str) -> f32 {
    use std::io::Write;
    if text.is_empty() {
        return 0.0;
    }
    let mut enc = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
    if enc.write_all(text.as_bytes()).is_err() {
        return 0.0;
    }
    match enc.finish() {
        Ok(z) if !z.is_empty() => text.len() as f32 / z.len() as f32,
        _ => 0.0,
    }
}

/// Which `timestamp_granularities[]` the caller asked for.
#[derive(Default, Clone, Copy)]
struct Granularities {
    segment: bool,
    word: bool,
}

impl Granularities {
    fn any(&self) -> bool {
        self.segment || self.word
    }
}

/// Split `language {X}<asr_text>{transcript}` into (detected language,
/// transcript); text without the envelope passes through whole.
fn parse_output(raw: &str) -> (Option<String>, String) {
    if let Some(idx) = raw.find("<asr_text>") {
        let head = &raw[..idx];
        let lang = head.strip_prefix("language ").map(|l| l.trim().to_owned());
        (lang, raw[idx + "<asr_text>".len()..].to_owned())
    } else {
        (None, raw.to_owned())
    }
}

/// One finished segment, before it becomes JSON or a subtitle cue. Both
/// consumers want the same three things plus the metrics, so the transcript is
/// turned into these once and rendered after.
struct Seg {
    start: f64,
    end: f64,
    text: String,
    tokens: Vec<u32>,
    /// the paddock extension the Studio marks by; see `word_confidences`
    words: Vec<WordConf>,
    avg_logprob: f32,
    no_speech_prob: f32,
    window: usize,
}

/// Group a segment's tokens into WORDS with a confidence each.
///
/// Whisper's BPE is sub-word - " Efter" arrives as one token but "grymheter"
/// as three - so per-TOKEN confidence renders as broken fragments in a UI. The
/// word is the unit a reader can act on ("the model was unsure of that name"),
/// and the boundary rule is whisper's own: a token whose text opens with a
/// space starts a new word.
///
/// Tokens are decoded as growing PREFIXES rather than one at a time. A single
/// byte-level BPE token can be half a multi-byte character, and decoding it
/// alone yields a replacement char; the difference between successive prefix
/// decodes is always valid text. Segments are tens of tokens, so the quadratic
/// cost is nothing.
/// One word, with what the model nearly said instead.
#[derive(Clone)]
pub(crate) struct WordConf {
    pub word: String,
    /// mean logprob over the word's tokens - the confidence the Studio marks by
    pub logprob: f32,
    /// The word's token span in the slice it was grouped from, INCLUSIVE both
    /// ends. This is what lets word times be attached to exactly the words the
    /// confidence was computed over: the engine returns a boundary
    /// per text token, and the alternative - grouping a second time on the
    /// timing side - is two groupings that can disagree about where a word
    /// starts.
    pub first: usize,
    pub last: usize,
    /// Seconds from the start of the CLIP, where the lane recovered them.
    /// Absent on a lane with no timing pass at all, and absent together - a
    /// word with a start and no end is not a word this endpoint reports.
    pub start: Option<f32>,
    pub end: Option<f32>,
    /// The RUNNER-UP at the word's first token, when one was asked for and it
    /// decodes to something a person can read.
    ///
    /// First token only, deliberately. A divergence there means the runner-up
    /// is a word-initial piece and reads as an alternative word ("vill" for
    /// "ville"); mid-word it is a BPE fragment ("lle" for "ll"), which is
    /// noise to show a human. Reconstructing a whole alternative word by
    /// substituting mid-stream would be worse than either: the model would
    /// have continued differently from there, so the reconstruction is a
    /// decode that never happened.
    pub alt: Option<String>,
    /// top1 - top2 in PROBABILITY space at that same token. Small = the model
    /// was choosing between two words, which is where errors live; a low
    /// top-1 probability with no close rival is a model that was merely
    /// diffuse, not torn.
    pub margin: Option<f32>,
}

/// Whether a runner-up is worth showing a reader as "what it nearly said".
///
/// Two rejections. One that decodes to nothing says nothing. And one that is a
/// PREFIX of the chosen word, or has the chosen word as a prefix, is the model
/// choosing a different TOKENIZATION of the same word rather than a different
/// word - on whisper's BPE that is the common case, not the corner. Measured
/// over one kb-whisper clip, most contested steps read " Fra" under " Frank",
/// " h" under " hennes", "ck" under "cka". Offering those as an alternative
/// reading tells the reader something untrue about what the model weighed.
///
/// The cost is real and accepted: a genuine pair like "vill"/"ville" is a
/// prefix too and gets dropped with them. There is no way to tell the two
/// apart from the first token alone - the model would have continued
/// differently after picking the runner-up, and that continuation is a decode
/// that never happened. Silence beats a confident wrong claim.
fn alt_reads(word: &str, alt: &str) -> bool {
    !alt.is_empty() && !word.starts_with(alt) && !alt.starts_with(word)
}

fn word_confidences(
    tok: &paddock_tokenizer::GgufTokenizer,
    tokens: &[u32],
    logprobs: &[f32],
    // per token, the runner-up `(id, logprob)`; empty when none was asked for
    runners: &[Option<(u32, f32)>],
) -> Result<Vec<WordConf>, String> {
    // (text, logprobs, first token index, last token index)
    let mut words: Vec<(String, Vec<f32>, usize, usize)> = Vec::new();
    let mut prev = String::new();
    for i in 0..tokens.len() {
        let whole = tok.decode(&tokens[..=i], true).map_err(|e| e.to_string())?;
        let piece = whole.strip_prefix(prev.as_str()).unwrap_or("").to_owned();
        prev = whole;
        let lp = logprobs.get(i).copied().unwrap_or(0.0);
        // a piece that is only whitespace belongs to the word it precedes, so
        // hold it for the next one rather than emitting a blank word
        let starts_word = piece.starts_with(char::is_whitespace) || words.is_empty();
        if starts_word && !piece.trim().is_empty() {
            words.push((piece.trim_start().to_owned(), vec![lp], i, i));
        } else if let Some(last) = words.last_mut() {
            last.0.push_str(&piece);
            last.1.push(lp);
            last.3 = i;
        }
    }
    let mut out = Vec::new();
    for (w, lps, first, last) in words {
        if w.trim().is_empty() {
            continue;
        }
        let n = lps.len().max(1) as f32;
        let (alt, margin) = match runners.get(first).copied().flatten() {
            Some((id, lp2)) => {
                let gap = logprobs.get(first).copied().unwrap_or(0.0).exp() - lp2.exp();
                let text = tok
                    .decode(&[id], true)
                    .unwrap_or_default()
                    .trim()
                    .to_owned();
                let show = alt_reads(w.trim_end(), &text);
                (show.then_some(text), Some(gap.max(0.0)))
            }
            None => (None, None),
        };
        out.push(WordConf {
            word: w.trim_end().to_owned(),
            logprob: lps.iter().sum::<f32>() / n,
            first,
            last,
            start: None,
            end: None,
            alt,
            margin,
        });
    }
    Ok(out)
}

/// Attach word times to words already grouped by `word_confidences`.
///
/// `bounds` is the window's boundary list - one per text token plus one, so
/// token `j` spans `[bounds[j], bounds[j+1])` - and `tok0` is where this
/// segment's tokens start in that window's text-token stream. A word therefore
/// runs from its first token's boundary to its last token's next one.
///
/// Words whose span falls outside the list keep no times rather than borrowing
/// a neighbour's: this only happens if the timing pass and the transcript
/// disagree about the token count, which is a bug, and a wrong time is worse
/// than a missing one.
fn attach_times(words: &mut [WordConf], bounds: &[f32], tok0: usize, duration_s: f64) {
    let clip_end = duration_s as f32;
    for w in words.iter_mut() {
        let (a, b) = (tok0 + w.first, tok0 + w.last + 1);
        let (Some(&s), Some(&e)) = (bounds.get(a), bounds.get(b)) else {
            continue;
        };
        // The last window is zero-padded to a full 30 s and the DTW path can
        // land on its final frame, so clamp for the same reason the segments
        // clamp: a word that seeks past the end of the file is wrong in the one
        // place a listener notices.
        let s = s.min(clip_end);
        w.start = Some(s);
        w.end = Some(e.max(s).min(clip_end));
    }
}

// ---- granite-speech-plus word times  ----
//
// A completely different mechanism from whisper's, and worth stating plainly
// because the wire hides the difference: whisper recovers times from
// cross-attention in a second pass that cannot change what was said, while
// granite-speech-plus is ASKED for them and writes them into its answer as
// text. So on this family the granularity CHANGES the TRANSCRIPT - a different
// instruction is a different task, and IBM's card is explicit that the
// timestamp mode drops the punctuation and capitalization the default prompt
// produces.
//
// The format (IBM's model card): `hello [T:45] world [T:82]`, where the number
// is when the word ENDED, in centiseconds - but only its last three digits, to
// save tokens. So the clock wraps every 10 s and the only way back is
// monotonicity: a tag that would land before the previous word ended must have
// wrapped. Silences ride as `_` with tags of their own, which is what keeps
// consecutive tags inside one wrap of each other and what gives the word after
// a pause an honest START rather than the end of the word before it.

/// One piece of a `[T:N]`-tagged transcript, in emission order.
enum TsPiece<'a> {
    Text(&'a str),
    /// centiseconds, mod 1000 - Not a time until the wrap is recovered
    Tag(u32),
}

/// Split a tagged string into text runs and tags.
///
/// A `[T:` that is not followed by digits and a `]` stays TEXT: the model is
/// writing a transcript, and swallowing characters on a near-miss would corrupt
/// the answer to save a malformed tag nobody can use anyway.
fn ts_pieces(s: &str) -> Vec<TsPiece<'_>> {
    let mut out = Vec::new();
    let mut rest = s;
    loop {
        let Some(i) = rest.find("[T:") else {
            if !rest.is_empty() {
                out.push(TsPiece::Text(rest));
            }
            return out;
        };
        if i > 0 {
            out.push(TsPiece::Text(&rest[..i]));
        }
        let after = &rest[i + 3..];
        let digits = after.bytes().take_while(u8::is_ascii_digit).count();
        if after.as_bytes().get(digits) == Some(&b']')
            && let Ok(v) = after[..digits].parse::<u32>()
        {
            out.push(TsPiece::Tag(v));
            rest = &after[digits + 1..];
            continue;
        }
        out.push(TsPiece::Text(&rest[i..i + 3]));
        rest = after;
    }
}

/// The transcript a reader wants: tags and silence markers out, one space
/// between words.
///
/// Whitespace is normalized rather than preserved because removing a tag from
/// `hello [T:45] world` otherwise leaves two spaces where the model wrote one.
/// This model emits space-separated words and no punctuation in this mode, so
/// there is no formatting to lose - and it makes the text agree exactly with
/// the words rejoined, which is the invariant the gate asserts.
fn granite_strip_ts(raw: &str) -> String {
    let mut text = String::new();
    for p in ts_pieces(raw) {
        if let TsPiece::Text(t) = p {
            text.push_str(t);
        }
    }
    text.split_whitespace()
        .filter(|w| *w != GRANITE_SILENCE)
        .collect::<Vec<_>>()
        .join(" ")
}

/// The same, for a transcript still being written.
///
/// A tag being TYPED must not reach a delta: `hello [` would render as a word
/// and then vanish two tokens later when `T:45]` completes the tag, and the one
/// thing the delta stream guarantees is that it only ever extends. An unclosed
/// `[` at the tail is a tag in progress - in this mode the model writes no
/// brackets of its own - so everything from it waits for the next token.
fn granite_stream_ts(raw: &str) -> String {
    let cut = raw
        .rfind('[')
        .filter(|&i| !raw[i..].contains(']'))
        .unwrap_or(raw.len());
    granite_strip_ts(&raw[..cut])
}

/// Attach granite-speech-plus's word times to words already grouped by
/// `word_confidences`.
///
/// Takes the GROUPS rather than the raw text for the same reason the whisper
/// lane does: there is one grouping, and it owns the word boundaries and the
/// confidences, so a word's time can never turn out to be about a different
/// word than the number printed beside it. A tag normally lands as a group of
/// its own (the space before `[` starts one), but the pieces are walked in
/// order within each group as well, so a tokenizer that glues `word [T:12]`
/// together still attributes the time to the word in front of it.
///
/// A word whose tag never arrived keeps no times rather than borrowing a
/// neighbour's - `verbose_body` then ships the whole array as `paddock_words`,
/// since a spec `words[]` with an invented `start` is a worse answer than an
/// honest extension one.
fn granite_timed_words(groups: &[WordConf], duration_s: f64) -> Vec<WordConf> {
    let clip_end = duration_s as f32;
    let mut out: Vec<WordConf> = Vec::new();
    // the running clock, in seconds from the start of the clip
    let mut last_end = 0.0f32;
    // how many whole 10 s wraps the three-digit tags have been through
    let mut offset = 0.0f32;
    // the word waiting for its end tag, and the time it started at
    let mut open: Option<(WordConf, f32)> = None;
    // a word we could not time is still a word; a silence is not, either way
    let close_untimed = |open: &mut Option<(WordConf, f32)>, out: &mut Vec<WordConf>| {
        if let Some((w, _)) = open.take()
            && w.word != GRANITE_SILENCE
        {
            out.push(w);
        }
    };
    for g in groups {
        for p in ts_pieces(&g.word) {
            match p {
                TsPiece::Text(t) => {
                    let t = t.trim();
                    if t.is_empty() {
                        continue;
                    }
                    close_untimed(&mut open, &mut out);
                    // the group's confidence and runner-up ride along: this is
                    // the same word, only with the tag text taken back out
                    open = Some((
                        WordConf {
                            word: t.to_owned(),
                            ..g.clone()
                        },
                        last_end,
                    ));
                }
                TsPiece::Tag(n) => {
                    let mut end = n as f32 / 100.0 + offset;
                    // IBM's own recovery loop, strict `<` and all: only the last
                    // three digits ride, so a tag landing before the previous
                    // word ended has wrapped. A silence longer than 10 s would
                    // defeat this, which is exactly why the model marks silences
                    // instead of leaving a hole.
                    while end < last_end {
                        offset += 10.0;
                        end += 10.0;
                    }
                    if let Some((mut w, start)) = open.take()
                        && w.word != GRANITE_SILENCE
                    {
                        // clamp for the reason every other lane clamps: a word
                        // that seeks past the end of the file is wrong in the
                        // one place a listener notices
                        let start = start.min(clip_end);
                        w.start = Some(start);
                        w.end = Some(end.max(start).min(clip_end));
                        out.push(w);
                    }
                    last_end = end;
                }
            }
        }
    }
    close_untimed(&mut open, &mut out);
    out
}

/// Where the TRANSCRIPT starts in a generative lane's token stream.
///
/// Qwen3-ASR prefixes its answer with `language {X}<asr_text>` when the
/// language was not forced. Those tokens are part of the decode and carry
/// logprobs like any other, so handing the whole stream to `word_confidences`
/// would report "language" and "swedish" as words of the transcript. The split
/// is found by decoding cumulatively, the same way `parse_output` finds it in
/// the finished string - there is no token id to match on, because the marker
/// is ordinary text to this tokenizer.
fn transcript_start(tok: &paddock_tokenizer::GgufTokenizer, ids: &[u32]) -> usize {
    for i in 0..ids.len() {
        if tok
            .decode(&ids[..=i], true)
            .is_ok_and(|s| s.contains("<asr_text>"))
        {
            return i + 1;
        }
    }
    0
}

/// Per-word confidence for a lane with no segments. `skip` drops
/// the envelope tokens; the words themselves are formed exactly as the whisper
/// lane forms them, so the two lanes' numbers mean the same thing.
fn generative_words(
    tok: &paddock_tokenizer::GgufTokenizer,
    ids: &[u32],
    lps: &[f32],
    runners: &[Option<(u32, f32)>],
    skip: usize,
) -> Option<Vec<WordConf>> {
    if lps.len() < ids.len() || skip > ids.len() {
        return None; // no logprobs asked for, or a stream that lost some
    }
    // the runner-up list is parallel to the tokens when present, absent when
    // top-2 was not asked for - either way it is sliced the same way
    let r = if runners.len() >= ids.len() {
        &runners[skip..]
    } else {
        &[][..]
    };
    word_confidences(tok, &ids[skip..], &lps[skip..], r)
        .ok()
        .filter(|w| !w.is_empty())
}

/// One word on the wire. `alt` and `margin` ride only where the lane could
/// answer them, so their ABSENCE is honest rather than a zero that reads as
/// "no close call".
fn word_json(w: &WordConf) -> serde_json::Value {
    let mut v = serde_json::json!({
        "word": w.word,
        "logprob": w.logprob,
        "confidence": w.logprob.exp(),
    });
    if let Some(a) = &w.alt {
        v["paddock_alt"] = serde_json::json!(a);
    }
    if let Some(m) = w.margin {
        v["paddock_margin"] = serde_json::json!(m);
    }
    v
}

/// The SPEC word object: `{word, start, end}` and nothing else that OpenAI
/// declares. Our per-word figures ride under `paddock_` names here - inside
/// `paddock_words` the container was already namespaced and `logprob` could sit
/// bare, but this array is spec, so an undeclared key on it is a claim about
/// the spec that is not true.
fn spec_word_json(w: &WordConf) -> serde_json::Value {
    let mut v = serde_json::json!({
        "word": w.word,
        "start": w.start.unwrap_or(0.0),
        "end": w.end.unwrap_or(0.0),
        "paddock_logprob": w.logprob,
        "paddock_confidence": w.logprob.exp(),
    });
    if let Some(a) = &w.alt {
        v["paddock_alt"] = serde_json::json!(a);
    }
    if let Some(m) = w.margin {
        v["paddock_margin"] = serde_json::json!(m);
    }
    v
}

/// Build OpenAI's `verbose_json` segment array from a whisper transcript.
///
/// The engine hands back a token stream per 30 s window with the timestamp
/// tokens still in it; `split_segments` cuts that into spans and this
/// detokenizes each one into the wire shape. Every field OpenAI declares is
/// filled - a segment object missing one is not a segment object as far as a
/// typed SDK is concerned.
fn segment_json(segs: &[Seg], window_s: f64, with_words: bool) -> Vec<serde_json::Value> {
    segs.iter()
        .enumerate()
        .map(|(i, s)| {
            let mut v = serde_json::json!({
                "id": i,
                // whisper's own unit: mel frames, 10 ms each, so a 30 s
                // window advances the seek by 3000
                "seek": (s.window as f64 * window_s * 100.0) as i64,
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "tokens": s.tokens,
                "temperature": 0.0,
                "avg_logprob": s.avg_logprob,
                "compression_ratio": compression_ratio(&s.text),
                "no_speech_prob": s.no_speech_prob,
            });
            // PADDOCK EXTENSION, named so it cannot be mistaken for spec.
            // OpenAI's segment has no per-word confidence anywhere and its
            // `word` object is {word, start, end}; the Studio's colour coding
            // needs the probability.
            //
            // It RETIRES when the body carries a real `words` array (
            // landed the times): the same confidences ride there, on objects
            // that also say when the word was said, and the same fact in two
            // places is two places to disagree.
            if !with_words {
                v["paddock_words"] =
                    serde_json::json!(s.words.iter().map(word_json).collect::<Vec<_>>());
            }
            v
        })
        .collect()
}

fn whisper_segments(
    tok: &paddock_tokenizer::GgufTokenizer,
    scale: &paddock_engine::gpu_model::whisper::TimeScale,
    out: &paddock_engine::transcriber::Transcript,
    duration_s: f64,
) -> Result<Vec<Seg>, String> {
    use paddock_engine::gpu_model::whisper::split_segments;
    let mut segments = Vec::new();
    for (wi, w) in out.windows.iter().enumerate() {
        // Where this segment's tokens start in the WINDOW's text-token stream,
        // which is the index space the engine's word-timing boundaries live in.
        // The segments partition that stream in order (a slice with no text at
        // all is skipped and contributes none), so a running cursor is the
        // whole mapping.
        let mut tok0 = 0usize;
        for seg in split_segments(&w.tokens, &w.logprobs, &w.runners, scale, wi) {
            let n_tok = seg.tokens.len();
            let at = tok0;
            tok0 += n_tok;
            let text = tok.decode(&seg.tokens, true).map_err(|e| e.to_string())?;
            let text = text.trim();
            if text.is_empty() {
                continue;
            }
            // The last window is zero-padded to a full 30 s, so whisper can
            // and does close a segment past where the audio actually stops.
            // Clamp to the clip - a caption that seeks past the end of the
            // file is wrong in the one place a listener would notice.
            let end = (seg.end as f64).min(duration_s).max(seg.start as f64);
            let mut words = word_confidences(tok, &seg.tokens, &seg.logprobs, &seg.runners)?;
            if !w.boundaries.is_empty() {
                attach_times(&mut words, &w.boundaries, at, duration_s);
            }
            segments.push(Seg {
                start: seg.start as f64,
                end,
                text: text.to_owned(),
                tokens: seg.tokens,
                words,
                avg_logprob: seg.avg_logprob,
                no_speech_prob: w.no_speech_prob,
                window: seg.window,
            });
        }
    }
    Ok(segments)
}

/// The clip's words, flat and in reading order, for a caller that asked for
/// `word` granularity.
///
/// Lifted straight out of the segments rather than regrouped: the segments hold
/// the one grouping - the boundary rule, the confidences, and now the times -
/// so a word's start can never disagree with the confidence printed beside it.
/// That was the phase-4 requirement, and re-splitting the text here would have
/// broken it silently on exactly the words BPE makes interesting.
fn whisper_words(segments: Option<&[Seg]>, grans: Granularities) -> Option<Vec<WordConf>> {
    let segs = segments.filter(|_| grans.word)?;
    Some(segs.iter().flat_map(|s| s.words.iter().cloned()).collect())
}

// ---- decode guards  ----
//
// An ASR decode can fail without erroring: whisper writes a fluent sentence
// over a window of silence, a model gets stuck repeating one token until the
// context bound stops it. `paddock_engine::audio::guards` catches both, and
// this is how the catch reaches the caller - because a guard that acts
// silently is the same silent failure with a smaller bill.
//
// There is no OpenAI field for this. Their API cannot express "this span of
// your audio produced nothing trustworthy" at all, so it rides as a namespaced
// extension on every JSON body, and it is ABSENT on the clips (all of them, in
// a clean battery) where nothing fired.

/// One notice about a span of audio whose decode did not go well.
struct Guard {
    start: f64,
    end: f64,
    /// `no_speech` | `repetition` | `length` | `context` | `language_mismatch`
    /// - see `reason_note`
    reason: &'static str,
    /// whether the span's text was DISCARDED. True only for `no_speech`,
    /// where there is a positive signal the audio held nothing; a cut decode
    /// keeps what it said up to the cut.
    dropped: bool,
    /// the numbers the verdict was made from, each present only where the
    /// lane could answer it
    no_speech_prob: Option<f32>,
    avg_logprob: Option<f32>,
    entropy: Option<f32>,
    /// Something the CALLER can do about it, where there is such a thing.
    /// Absent on the notices that are simply reports.
    ///
    /// Owned rather than `&'static str`: the language notice's
    /// remedy names the language the transcript actually came back in, and a
    /// hint that cannot say which language is barely a hint.
    hint: Option<String>,
}

/// What each reason means, in the response rather than in our docs. A caller
/// reading `"reason": "repetition"` should not have to look anything up to
/// know whether to retry, re-record, or accept the answer.
fn reason_note(reason: &str) -> &'static str {
    match reason {
        "no_speech" => {
            "the model reports no speech here and what it transcribed anyway was \
                        low-confidence, so the text was discarded"
        }
        "repetition" => {
            "the decode collapsed into a repeating loop and was cut; the text \
                         before the loop is kept"
        }
        "length" => "the decode hit the token ceiling for audio of this length and was cut",
        "context" => "the decode ran out of served context and was cut",
        "vad" => "no speech was detected here, so this span was skipped before the model saw it",
        "no_speech_marker" => concat!(
            "the model answered this span with its no-speech marker instead of with ",
            "words; it was not passed off as a transcript",
        ),
        // The one notice that is not about the decode being cut, and the one
        // this endpoint most needed. A whisper-family model given
        // the wrong language does not mislabel its output - it TRANSLATES,
        // and the result is fluent and grammatical with nothing to notice.
        // Nothing else here can catch that, because nothing else compares the
        // language asked for against the language written.
        "language_mismatch" => concat!(
            "the transcript is not written in the language this decode ran under; a ",
            "speech model told to expect the wrong language translates rather than ",
            "transcribes, so this text may be a translation of what was said",
        ),
        _ => "the decode was stopped",
    }
}

/// The language notice, when the transcript's own language contradicts the
/// one the decode ran under.
///
/// Spans the whole clip, unlike every other notice here, and honestly so: the
/// language is settled once per transcription, so there is no span of audio to
/// point at - the finding is about the answer, not about a stretch of the
/// recording.
///
/// It rides as a guard rather than living only in the verbose body because
/// `response_format=json` has no verbose body to look inside, and this is the
/// last failure on this endpoint that a plain-JSON caller should have to
/// discover by reading their transcript in a language they do not speak.
fn language_guard(rep: &LanguageReport, duration_s: f64) -> Option<Guard> {
    let found = rep.mismatch()?;
    let asked = rep.code.as_deref().unwrap_or_default().to_owned();
    let name = crate::language::display_name(&found.code);
    Some(Guard {
        start: 0.0,
        end: duration_s,
        reason: "language_mismatch",
        // The text stands. It is the caller's to judge - it may be a perfectly
        // good translation, and throwing it away would be this endpoint
        // deciding something it has no standing to decide.
        dropped: false,
        no_speech_prob: None,
        avg_logprob: None,
        entropy: None,
        hint: Some(format!(
            "this text reads as {name}; send `language={}` to transcribe it as {name}, or omit \
             `language` to let the model decide (it ran as {asked})",
            found.code,
        )),
    })
}

fn guard_json(g: &Guard) -> serde_json::Value {
    let mut v = serde_json::json!({
        "start": g.start,
        "end": g.end,
        "reason": g.reason,
        "text_dropped": g.dropped,
        "note": reason_note(g.reason),
    });
    if let Some(p) = g.no_speech_prob {
        v["no_speech_prob"] = serde_json::json!(p);
    }
    if let Some(l) = g.avg_logprob {
        v["avg_logprob"] = serde_json::json!(l);
    }
    // absent on a window too short for the tail window to mean anything -
    // an infinity on the wire says nothing a missing key does not
    if let Some(e) = g.entropy.filter(|e| e.is_finite()) {
        v["entropy"] = serde_json::json!(e);
    }
    if let Some(h) = &g.hint {
        v["hint"] = serde_json::json!(h);
    }
    v
}

/// The clip's notices, one per whisper window that tripped something.
///
/// Windows are the unit because whisper's guards are: each 30 s window is its
/// own decode with its own no-speech probability, and a clip where window 4 is
/// silence and the rest is speech has to say which four seconds to distrust.
fn whisper_guards(
    out: &paddock_engine::transcriber::Transcript,
    window_s: f64,
    duration_s: f64,
) -> Vec<Guard> {
    out.windows
        .iter()
        .enumerate()
        .filter_map(|(i, w)| {
            // both suppressions empty the window; they differ in how we know,
            // and a notice that claimed the wrong one would be a small lie
            let reason = match (w.suppressed, w.stop) {
                (true, paddock_engine::audio::guards::Stop::Marker) => Some("no_speech_marker"),
                (true, _) => Some("no_speech"),
                (false, st) => st.wire(),
            };
            let start = i as f64 * window_s;
            Some(Guard {
                start,
                end: ((i + 1) as f64 * window_s).min(duration_s).max(start),
                reason: reason?,
                dropped: w.suppressed,
                no_speech_prob: Some(w.no_speech_prob),
                avg_logprob: Some(w.avg_logprob),
                entropy: Some(w.entropy),
                // MEASURED: nb-whisper-large refuses
                // four battery clips under the `<|notimestamps|>` prompt and
                // transcribes every one of them without it. So where a
                // checkpoint answers with its marker and the caller did not
                // ask for times, there is something they can actually do -
                // and a notice that knows it should say so rather than leave
                // them with an empty transcript and a reason code.
                // MEASURED, and the warning is half the value.
                // Asking for times does get text out of a refusing checkpoint -
                // but on nb-whisper that text scored 100% WER on four of five
                // recovered clips, because what comes back is a TRANSLATION
                // into the model's own language rather than a transcript. So
                // the hint names the door and says what is behind it.
                hint: (reason? == "no_speech_marker" && !out.timestamps).then(|| {
                    concat!(
                        "this checkpoint's refusal is conditioned on the no-timestamps decode ",
                        "prompt; asking for timestamp_granularities[]=segment may return text ",
                        "instead - check what language it comes back in, since a monolingual ",
                        "fine-tune may answer with a translation rather than a transcript",
                    )
                    .to_owned()
                }),
            })
        })
        .collect()
}

/// Attach the notices to a body, or leave it exactly as it was.
///
/// Absent rather than empty: an empty array on every clean transcription is a
/// key every client has to learn to ignore, and this one means "look here".
fn with_guards(mut body: serde_json::Value, guards: &[Guard]) -> serde_json::Value {
    if !guards.is_empty() {
        body["paddock_guards"] =
            serde_json::json!(guards.iter().map(guard_json).collect::<Vec<_>>());
    }
    body
}

/// Every guard signal for one decode unit, whether or not anything fired
///
/// The notices above say what happened; these say how close it came, which is
/// the only way to tell a threshold that is holding from one that is about to
/// misfire. Opt-in, because on a clean clip it is a page of numbers nobody
/// asked for - and because the request that made it necessary is ours: the
/// thresholds are OpenAI's 2022 constants and had never been measured on a
/// checkpoint we serve.
///
/// It is also the honest answer to a promise `compression_ratio`'s comment has
/// been making since  - "so the caller can apply their own threshold".
/// A caller cannot do that on a number they can only see for one of the four.
struct GuardStats {
    start: f64,
    end: f64,
    tokens: usize,
    /// tail entropy in nats; absent where the lane does not run that test or
    /// the decode was too short to judge
    entropy: Option<f32>,
    avg_logprob: Option<f32>,
    no_speech_prob: Option<f32>,
    compression_ratio: Option<f32>,
    /// the wire name of how it ended, or "eot"
    stop: &'static str,
}

fn stats_json(s: &GuardStats) -> serde_json::Value {
    let mut v = serde_json::json!({
        "start": s.start,
        "end": s.end,
        "tokens": s.tokens,
        "stop": s.stop,
    });
    // Each number rides only where the lane can answer it, the same rule the
    // word objects follow: a null would read as "we measured zero".
    for (k, n) in [
        ("entropy", s.entropy.filter(|e| e.is_finite())),
        ("avg_logprob", s.avg_logprob),
        ("no_speech_prob", s.no_speech_prob),
        ("compression_ratio", s.compression_ratio),
    ] {
        if let Some(n) = n {
            v[k] = serde_json::json!(n);
        }
    }
    // The thresholds this build would apply, sent alongside the values so a
    // sweep does not have to hardcode our constants to compute a headroom -
    // and so a change to them shows up in the data rather than silently
    // shifting what an old artifact means.
    v["thresholds"] = serde_json::json!({
        "entropy": guards::ENTROPY_THOLD,
        "logprob": guards::LOGPROB_THOLD,
        "no_speech": guards::NO_SPEECH_THOLD,
    });
    v
}

fn with_stats(mut body: serde_json::Value, stats: &[GuardStats]) -> serde_json::Value {
    if !stats.is_empty() {
        body["paddock_guard_stats"] =
            serde_json::json!(stats.iter().map(stats_json).collect::<Vec<_>>());
    }
    body
}

// ---- no-speech MARKERS ----
//
// Some whisper fine-tunes answer non-speech by TYPING a marker instead of by
// raising `<|nospeech|>`'s probability. Measured on nb-whisper-large:
// 8 s of digital silence comes back as the seven ORDINARY TEXT tokens
// `[2627, 91, 1771, 496, 9799, 91, 29]`, which spell `<|nocaptions|>` - every
// one of them far below `<|endoftext|>` (50257), so no control-token filter can
// see them and the string lands in the user's transcript verbatim. It is a
// training artifact: that literal must have been in the fine-tune's own
// transcripts.
//
// The model is not wrong, it is ANSWERING - in the only vocabulary it has -
// and this is the same verdict the silence rule computes from statistics
// (`guards::is_no_speech`), arriving by a different route. So it gets the same
// treatment: the window is suppressed and the caller is told, rather than
// handed a marker to interpret. Worth pairing with, which found that
// on this very checkpoint the no-speech PROBABILITY is inert - the token is
// the only channel nb-whisper has left.
//
// A per-checkpoint literal, and named as one. The precedent is already here:
// `GRANITE_SILENCE`, the `<asr_text>` envelope, granite's `[T:N]` tags. The
// safety is that the whole trimmed window must equal a marker - a transcript
// that merely contains the characters is untouched, and a caller who genuinely
// dictates "<|nocaptions|>" and nothing else gets a notice explaining why they
// got nothing back.
const NO_SPEECH_MARKERS: &[&str] = &["<|nocaptions|>", "<|nospeech|>"];

fn is_no_speech_marker(text: &str) -> bool {
    NO_SPEECH_MARKERS.contains(&text.trim())
}

/// Is this the beginning of a marker being typed?
///
/// The streaming half of the rule, and the same trick `granite_stream_ts` uses
/// for a tag in progress: `<|noc` must not reach a delta, because two tokens
/// later the whole thing is withdrawn and a delta stream cannot take bytes
/// back. Text that is a PROPER prefix of a marker waits.
fn is_marker_prefix(text: &str) -> bool {
    let t = text.trim();
    !t.is_empty()
        && NO_SPEECH_MARKERS
            .iter()
            .any(|m| m.starts_with(t) && *m != t)
}

/// Should the delta stream sit on this window's text for now?
///
/// A marker being TYPED, and a marker that is COMPLETE - the second because a
/// finished marker is about to be suppressed, and emitting it one instant
/// before withdrawing it is the same broken promise as emitting half of one.
/// A window that carries on past the marker into real words stops matching and
/// flows normally.
fn marker_hold(text: &str) -> bool {
    is_marker_prefix(text) || is_no_speech_marker(text)
}

/// Suppress any window whose whole transcript is a no-speech marker.
///
/// Runs on the `Transcript` the moment it comes back, so every consumer
/// downstream - the joined text, the segments, the guard notices, the stats,
/// the live socket - sees the same thing without any of them knowing this rule
/// exists. That is the same reason the engine clears a suppressed window's
/// tokens itself rather than flagging them (see `transcriber::Window`): a rule
/// four consumers have to remember is a rule three of them will forget.
pub(crate) fn suppress_marker_windows(
    out: &mut paddock_engine::transcriber::Transcript,
    tok: &paddock_tokenizer::GgufTokenizer,
    scale: &paddock_engine::gpu_model::whisper::TimeScale,
) {
    for w in &mut out.windows {
        if w.tokens.is_empty() || w.suppressed {
            continue;
        }
        let text: Vec<u32> = w
            .tokens
            .iter()
            .copied()
            .filter(|&t| !scale.is_timestamp(t))
            .collect();
        if tok
            .decode(&text, true)
            .is_ok_and(|s| is_no_speech_marker(&s))
        {
            w.suppressed = true;
            w.stop = paddock_engine::audio::guards::Stop::Marker;
            w.tokens.clear();
            w.logprobs.clear();
            w.runners.clear();
        }
    }
}

/// Which 30 s windows hold speech.
///
/// The gate every shipping ASR system has and we did not: faster-whisper's
/// `vad_filter`, whisper.cpp's `--vad`, WhisperX's whole pipeline. A window the
/// encoder never sees cannot be hallucinated in, and on a real recording -
/// meetings, voicemail, a podcast with a music bed - it is also the cheapest
/// throughput there is.
///
/// MEASURED, not assumed: roest-v3 answers 8 s
/// of quiet room tone with "Ja det er det. Det er det. Det er det. Det er det",
/// which no decode guard caught - 16 tokens is under the entropy window and
/// the capitalised first phrase breaks the exact period. This is what would
/// have stopped it, before the encoder ran.
///
/// Ours is an ENERGY detector with a minimum-statistics noise floor, not
/// Silero: models run on the GPU here, so a neural VAD is not available to us
/// on the host (see `audio::vad`). That is a real difference in the noisy case
/// and the reason this is opt-in.
/// OpenAI `chunking_strategy`: the PER-REQUEST control for the
/// VAD window gate that until now only had a server-level `--vad-gate` flag.
///
/// `"auto"` and `{"type": "server_vad"}` both mean the same thing here - server
/// VAD is the only chunking this server does, so "let the server decide" and
/// "use server VAD" resolve identically. Returns the threshold to gate with,
/// or None to leave the server default alone.
///
/// `prefix_padding_ms` / `silence_duration_ms` are REFUSED rather than
/// swallowed. They shape where a live socket cuts an utterance; this endpoint
/// gates whole 30 s encoder windows on whether they contain speech at all, so
/// there is no boundary here for them to move. Accepting them would let a
/// caller believe they had tuned something.
fn parse_chunking_strategy(raw: &str) -> Result<Option<f32>, String> {
    let v: serde_json::Value = match serde_json::from_str(raw) {
        Ok(v) => v,
        // a bare word arrives unquoted from a multipart form
        Err(_) => serde_json::Value::String(raw.trim().to_owned()),
    };
    if v.as_str() == Some("auto") {
        return Ok(Some(VAD_GATE_THRESHOLD));
    }
    let Some(obj) = v.as_object() else {
        return Err(format!(
            "chunking_strategy must be \"auto\" or a server_vad object (got {raw})"
        ));
    };
    match obj.get("type").and_then(serde_json::Value::as_str) {
        Some("server_vad") => {}
        Some(other) => {
            return Err(format!(
                "unsupported chunking_strategy type {other:?} (this server does \"server_vad\")"
            ));
        }
        None => return Err("chunking_strategy needs a `type`".into()),
    }
    let mut threshold = VAD_GATE_THRESHOLD;
    for (k, val) in obj {
        match k.as_str() {
            "type" => {}
            "threshold" => match val.as_f64() {
                Some(t) if (0.0..=1.0).contains(&t) => threshold = t as f32,
                Some(t) => {
                    return Err(format!(
                        "chunking_strategy.threshold must be 0..=1 (got {t})"
                    ));
                }
                None => return Err("chunking_strategy.threshold must be a number".into()),
            },
            "prefix_padding_ms" | "silence_duration_ms" => {
                return Err(format!(
                    "chunking_strategy.{k} is not supported here: it moves an utterance boundary in a live session, and this endpoint gates whole encoder windows on whether they hold speech at all"
                ));
            }
            other => return Err(format!("unsupported chunking_strategy field {other:?}")),
        }
    }
    Ok(Some(threshold))
}

fn window_speech(samples: &[f32], rate: u32, threshold: f32) -> Vec<bool> {
    use paddock_engine::audio::PAD_SAMPLES;
    let mut vad = paddock_engine::audio::vad::Vad::new(rate, threshold);
    // one pass over the clip; the detector is causal, so its noise floor warms
    // up on the lead-in exactly as it would in a live session
    let mut speech = vec![false; samples.len().div_ceil(PAD_SAMPLES).max(1)];
    for f in vad.feed(samples) {
        if f.speech {
            // a frame straddling a window boundary marks both: a word cut in
            // half by the 30 s grid must not lose either half
            for w in [f.start / PAD_SAMPLES, f.end.saturating_sub(1) / PAD_SAMPLES] {
                if let Some(slot) = speech.get_mut(w) {
                    *slot = true;
                }
            }
        }
    }
    speech
}

/// Per-window guard signals for a whisper transcript.
///
/// The window is the unit for the same reason the notices use it: each 30 s
/// window is its own decode with its own no-speech probability. The text a
/// window produced is re-derived here only to compress it - a suppressed
/// window has no tokens left, and its ratio is honestly absent rather than
/// the 0.0 an empty string would give.
fn whisper_stats(
    tok: &paddock_tokenizer::GgufTokenizer,
    scale: &paddock_engine::gpu_model::whisper::TimeScale,
    out: &paddock_engine::transcriber::Transcript,
    duration_s: f64,
) -> Vec<GuardStats> {
    let window_s = scale.window_s as f64;
    out.windows
        .iter()
        .enumerate()
        .map(|(i, w)| {
            let text: Vec<u32> = w
                .tokens
                .iter()
                .copied()
                .filter(|&t| !scale.is_timestamp(t))
                .collect();
            let decoded = tok.decode(&text, true).unwrap_or_default();
            let start = i as f64 * window_s;
            GuardStats {
                start,
                end: ((i + 1) as f64 * window_s).min(duration_s).max(start),
                tokens: w.tokens.len(),
                entropy: Some(w.entropy),
                avg_logprob: Some(w.avg_logprob),
                no_speech_prob: Some(w.no_speech_prob),
                compression_ratio: (!decoded.trim().is_empty())
                    .then(|| compression_ratio(&decoded)),
                stop: if w.suppressed {
                    "no_speech"
                } else {
                    w.stop.wire().unwrap_or("eot")
                },
            }
        })
        .collect()
}

/// The generative lanes' half of the decode guards.
///
/// Whisper's live in the engine because its decode does. These families run on
/// the ORDINARY generator, whose loop serves chat as much as transcription -
/// and a repetition guard belongs nowhere near chat: a code block legitimately
/// repeats indentation for dozens of tokens, and an entropy floor would cut it
/// mid-function. So the guard sits out here, on the one lane whose output is
/// known to be speech.
///
/// There is no clip-relative span to report either: these models take the
/// whole buffer as a single prompt, so a notice covers the whole clip. That is
/// the honest granularity - the model never told us where in the audio it lost
/// the thread.
struct GenGuard {
    rep: Repetition,
    cut: Option<&'static str>,
}

impl GenGuard {
    /// `structured` = this decode has a GRAMMAR rather than being plain
    /// transcript text, which today means granite-speech-plus's timestamp mode
    /// (`word [T:452] word [T:498]`). It changes which tests run - see
    /// `Repetition`, and the measurement that forced the split.
    fn new(structured: bool) -> Self {
        let rep = if structured {
            Repetition::structured()
        } else {
            Repetition::text()
        };
        Self { rep, cut: None }
    }

    /// Feed a decoded token. `true` means stop consuming - the caller drops
    /// its receiver, which is how the engine learns to retire the slot (a
    /// closed event channel fails the next send and the slot goes back).
    fn token(&mut self, id: u32) -> bool {
        if self.cut.is_some() {
            return true;
        }
        if self.rep.push(id) {
            self.cut = Some("repetition");
            return true;
        }
        false
    }

    /// The engine's own verdict. `Length` here means the token ceiling cut the
    /// transcript, which used to be a 200 with a silently truncated answer.
    fn finished(&mut self, reason: paddock_engine::service::FinishReason) {
        if self.cut.is_none() && reason == paddock_engine::service::FinishReason::Length {
            self.cut = Some("length");
        }
    }

    /// One entry covering the whole clip - these families take the buffer as a
    /// single prompt, so there is no window to split on.
    fn stats(&self, duration_s: f64, tokens: usize, text: &str, lps: &[f32]) -> Vec<GuardStats> {
        vec![GuardStats {
            start: 0.0,
            end: duration_s,
            tokens,
            // absent in granite's timestamp mode: that lane runs the period
            // test only, so there is no entropy being tested to report
            entropy: Some(self.rep.value()),
            // only where the caller asked for logprobs - a decode without them
            // has nothing to average, and a fabricated 0.0 would read as
            // perfect confidence
            avg_logprob: (!lps.is_empty()).then(|| guards::avg_logprob(lps)),
            // no no-speech head on these families at all
            no_speech_prob: None,
            compression_ratio: (!text.trim().is_empty()).then(|| compression_ratio(text)),
            stop: self.cut.unwrap_or("eot"),
        }]
    }

    fn guards(&self, duration_s: f64) -> Vec<Guard> {
        self.cut
            .map(|reason| {
                vec![Guard {
                    start: 0.0,
                    end: duration_s,
                    reason,
                    dropped: false,
                    no_speech_prob: None,
                    avg_logprob: None,
                    entropy: Some(self.rep.value()),
                    hint: None,
                }]
            })
            .unwrap_or_default()
    }
}

/// The verbose_json body, built once so the streaming and non-streaming
/// answers cannot drift apart.
fn verbose_body(
    text: &str,
    // None where the lane cannot say - granite-speech detects the input
    // language and never reports it. It reaches the wire as
    // `UNKNOWN_LANGUAGE`, not as null: the SDK types this field as a required
    // string and raises on null.
    language: Option<&str>,
    duration_s: f64,
    segments: Option<&[Seg]>,
    window_s: f64,
    // Everything this transcription can say about LANGUAGE: where
    // the reported code came from, its probability and runners-up where a
    // detector produced them, the candidate set the caller hinted and whether
    // it changed the answer, and what language the finished text is actually
    // written in. Absent - not null - on a lane with nothing to say.
    //
    // It does not replace `language`, which stays exactly the spec field it
    // was: one string, the language of the transcription. This is the story
    // behind that string, under a namespaced key, because OpenAI's verbose
    // body has nowhere to put a probability or a contradiction.
    language_detail: Option<&LanguageReport>,
    // The clip's words, flat and in reading order.
    //
    // Which KEY they LAND under is DECIDED by whether they have times, because
    // that is exactly what separates the spec's object from ours. OpenAI's
    // `word` is `{word, start, end}` - a word with no times is not one, so a
    // lane that has logprobs and no timestamp vocabulary at all (Qwen3-ASR,
    // granite-speech) answers under `paddock_words` instead. Not a
    // one-segment array spanning the clip either: a segment is a TIME SPAN, and
    // inventing start=0/end=duration would put a seek affordance on every word
    // that could only ever jump to the beginning.
    words: Option<&[WordConf]>,
) -> serde_json::Value {
    let mut body = serde_json::json!({
        "task": "transcribe",
        "language": language.filter(|c| !c.is_empty()).unwrap_or(UNKNOWN_LANGUAGE),
        "duration": duration_s,
        "text": text,
    });
    // A window either got the timing pass or it did not, so this is all-or-
    // nothing in practice; `all` rather than `any` because the one shape that
    // must never ship is a spec `words` array with a `start` of 0.0 standing in
    // for "we do not know".
    let timed = words.is_some_and(|ws| ws.iter().all(|w| w.start.is_some() && w.end.is_some()));
    if let Some(detail) = language_detail.and_then(LanguageReport::json) {
        body["paddock_language"] = detail;
    }
    if let Some(segs) = segments {
        body["segments"] = serde_json::Value::Array(segment_json(segs, window_s, timed));
    }
    if let Some(ws) = words {
        body[if timed { "words" } else { "paddock_words" }] = serde_json::json!(
            ws.iter()
                .map(if timed { spec_word_json } else { word_json })
                .collect::<Vec<_>>()
        );
    }
    body
}

/// The same object, for one closed realtime utterance.
///
/// The live socket used to hand back text and nothing else, so a turn committed
/// straight from it had no word times, no confidence and no segments - and the
/// Studio worked around that by transcribing the recording a SECOND time
/// through the file endpoint and storing that instead. Deciding that the
/// utterance is the unit of finality removes the second decode: the pass that
/// closes an utterance already runs, so it may as well be the enriched one.
///
/// Deliberately routed through `verbose_body` and `whisper_guards` rather than
/// assembling its own shape: `paddock_verbose` is documented on the streaming
/// file lane as "the exact object the non-streaming verbose_json returns", and
/// a second assembly is a second thing to drift.
///
/// Times are UTTERANCE-LOCAL - this is the object the file endpoint would
/// return if you handed it this utterance's audio on its own, which is what
/// makes it composable. Where the item sits in the session rides beside it, on
/// the event.
pub(crate) fn live_verbose(
    tok: &paddock_tokenizer::GgufTokenizer,
    scale: &paddock_engine::gpu_model::whisper::TimeScale,
    // the checkpoint's own language map - bounds the transcript's language
    // check to what this model could have written (see `language`)
    langs: &[String],
    // the SESSION's forced language, if it set one. A live session settles
    // this once at config rather than per utterance, but it is still an
    // instruction and not a measurement - without it every forced session
    // would report `source: unknown` beside a perfectly definite code.
    asked: Option<&str>,
    out: &paddock_engine::transcriber::Transcript,
    // The FINAL PASS's own reading, not the transcript the deltas built. The
    // two can differ: the committed prefix was decided from shorter buffers and
    // a delta stream never retracts, so what was shown stands even when the
    // last pass changes its mind about it. Keeping this object self-consistent
    // - text, segments and words all from one decode - means the disagreement
    // is visible and countable where it happens, instead of a `words` array
    // silently indexing a string it was not produced from.
    text: &str,
    duration_s: f64,
) -> serde_json::Value {
    let mut guards = whisper_guards(out, scale.window_s as f64, duration_s);
    // Segments always; timed words only if the backend supplied alignment.
    // Without alignment these remain confidence-only `paddock_words`, never
    // fabricated word timestamps or a reason to fail final transcription.
    let segs = whisper_segments(tok, scale, out, duration_s).ok();
    let words = whisper_words(
        segs.as_deref(),
        Granularities {
            segment: true,
            word: true,
        },
    );
    // No candidate set here: a session hints once or not at all, and re-asking
    // per utterance would be the mid-clip language flip the file lane refuses
    // for the same reason.
    let rep = whisper_report(out, asked, &[], text, langs);
    guards.extend(language_guard(&rep, duration_s));
    with_guards(
        verbose_body(
            text,
            Some(&out.language),
            duration_s,
            segs.as_deref(),
            scale.window_s as f64,
            Some(&rep),
            words.as_deref(),
        ),
        &guards,
    )
}

/// The language report for a GENERATIVE lane (Qwen3-ASR, granite-speech).
///
/// No posterior exists on these lanes - there is no language head to read, so
/// `candidates` is empty and no probability is reported. What they can answer
/// differs by family and that difference is the whole content of `source`:
/// Qwen3-ASR NAMES its language inside its answer, granite-speech detects
/// internally and says nothing. The text check runs either way, and on
/// granite (which reports no language at all) it is the only language signal
/// the response carries.
fn generative_report(
    asked: Option<&str>,
    // whatever the lane could say - a code from `language`, an English name
    // from Qwen3-ASR's envelope, or nothing
    reported: Option<&str>,
    text: &str,
) -> LanguageReport {
    let source = match (asked, reported) {
        (Some(_), _) => Source::Asked,
        (None, Some(_)) => Source::Reported,
        (None, None) => Source::Unknown,
    };
    // Normalised to a CODE here rather than passed through: the model writes
    // "Swedish" and everything downstream compares codes (see
    // `language::to_code`). An unrecognised value keeps its own text - better
    // an unmatchable string than a wrong language.
    let code = asked
        .or(reported)
        .and_then(|v| crate::language::to_code(v).or_else(|| Some(v.to_owned())));
    LanguageReport::new(code, source, Vec::new(), None, Vec::new(), 0.0, text, &[])
}

/// The language report for a whisper transcription.
///
/// `asked` is the caller's forced code (None = they let it detect), `hints`
/// their candidate set, `langs` the checkpoint's own map. The engine already
/// decided the language and, where it detected one, left the posterior on the
/// transcript - this only says which of the four sources that was and runs the
/// text check.
fn whisper_report(
    out: &paddock_engine::transcriber::Transcript,
    asked: Option<&str>,
    hints: &[String],
    text: &str,
    langs: &[String],
) -> LanguageReport {
    let source = if asked.is_some() {
        Source::Asked
    } else if out.language_probs.is_empty() {
        // no window ever decoded (an empty clip, or every window VAD-gated),
        // so nothing detected anything and there is no code to stand behind
        Source::Unknown
    } else {
        Source::Detected
    };
    LanguageReport::new(
        (!out.language.is_empty()).then(|| out.language.clone()),
        source,
        out.language_probs.clone(),
        out.language_prior_moved.clone(),
        hints.to_vec(),
        paddock_engine::transcriber::DEFAULT_LANGUAGE_PRIOR,
        text,
        langs,
    )
}

// ---- streaming  ----
//
// `stream=true` answers with OpenAI's own event set for this endpoint -
// `transcript.text.delta` repeatedly, then one `transcript.text.done`. The
// reason to have it is not politeness to the SDK: whisper cuts a clip into
// 30 s windows and decodes each a token at a time, so on a 40-minute file
// the transcript exists progressively, and answering only at the end hides
// work that is already finished.
//
// The INVARIANT, and every piece below exists to hold it: the deltas
// concatenate to `done.text`, byte for byte. A client that appends each
// delta must end up with the same string as one that ignores them and reads
// the final event. The conformance gate asserts it, because a streaming
// transcript that quietly disagrees with the final one is worse than no
// streaming at all.
//
// Two hazards threaten it. Windows decode out of ORDER (they occupy slots in
// parallel), and a byte-level BPE token can be half a character.

/// The part of a growing detokenization that is safe to show.
///
/// Decoding a token prefix that ends mid-scalar yields U+FFFD, which the next
/// token replaces with the real glyph; and `join_windows` trims, so trailing
/// whitespace is not part of the answer yet either. Emitting either would put
/// bytes in the delta stream that the final text does not contain.
fn stable(s: &str) -> &str {
    s.trim_end_matches(|c: char| c == char::REPLACEMENT_CHARACTER || c.is_whitespace())
}

/// What a delta stream has already sent, as a byte cursor into the growing
/// text.
#[derive(Default)]
struct Emitted(usize);

impl Emitted {
    /// The not-yet-sent suffix. `text` only ever extends what came before -
    /// that is what `stable` buys - so a byte count is a valid cursor. The
    /// boundary check is belt and braces: skipping one event is recoverable
    /// (the next one carries the same bytes), emitting half a character is
    /// not.
    fn advance(&mut self, text: &str) -> Option<String> {
        if text.len() <= self.0 || !text.is_char_boundary(self.0) {
            return None;
        }
        let d = text[self.0..].to_owned();
        self.0 = text.len();
        Some(d)
    }
}

/// Reassembles the transcriber's out-of-order window progress into the single
/// growing string a reader can be shown.
///
/// Text may only be emitted for the run of windows that is complete from the
/// START of the clip, plus the first incomplete one. Anything past that gap
/// would have to be taken back when the gap fills in, and a delta stream has
/// no way to take anything back. `frontier` is where that run ends.
struct WhisperStream {
    tok: Arc<paddock_tokenizer::GgufTokenizer>,
    scale: paddock_engine::gpu_model::whisper::TimeScale,
    /// per window, the ids that carry TEXT - timestamp tokens are decode
    /// control and never reach a transcript
    ids: Vec<Vec<u32>>,
    /// `ids[w]` detokenized. Recomputed whole on each token rather than
    /// appended to, which is what makes a multi-byte character come out
    /// intact (the same growing-prefix trick as `word_confidences`).
    parts: Vec<String>,
    closed: Vec<bool>,
    /// Windows the silence guard might yet suppress. The verdict
    /// needs the finished decode, so an open window that could be dropped must
    /// not put bytes on the wire - the one thing this stream promises is that
    /// it never takes any back. Set from `Progress::WindowOpen`, which arrives
    /// before that window's first token, and true for almost nothing: it takes
    /// a no-speech probability past 0.6 to get here at all.
    provisional: Vec<bool>,
    frontier: usize,
    lang: String,
}

/// Which window texts may be shown right now - the one rule the delta stream
/// lives or dies by, kept as a function so it can be tested without a
/// tokenizer and a GPU.
///
/// Everything before `frontier` is closed and final. The window at the
/// frontier is still decoding, so only its stable prefix goes out, and only
/// while the silence guard has not marked it provisional - a window that may
/// yet be suppressed must stream nothing, because a delta cannot be recalled.
/// Anything past the frontier waits: emitting window 3 while window 2 is still
/// open would put the transcript out of order.
fn emittable<'a>(parts: &'a [String], frontier: usize, provisional: &[bool]) -> Vec<&'a str> {
    let mut v: Vec<&str> = parts[..frontier.min(parts.len())]
        .iter()
        .map(String::as_str)
        .collect();
    if let Some(open) = parts.get(frontier)
        && !provisional.get(frontier).copied().unwrap_or(false)
        && !marker_hold(stable(open))
    {
        v.push(stable(open));
    }
    v
}

impl WhisperStream {
    fn new(
        windows: usize,
        scale: paddock_engine::gpu_model::whisper::TimeScale,
        tok: Arc<paddock_tokenizer::GgufTokenizer>,
    ) -> Self {
        Self {
            tok,
            scale,
            ids: vec![Vec::new(); windows],
            parts: vec![String::new(); windows],
            closed: vec![false; windows],
            provisional: vec![false; windows],
            frontier: 0,
            lang: String::new(),
        }
    }

    fn apply(&mut self, p: paddock_engine::transcriber::Progress) {
        use paddock_engine::transcriber::Progress;
        match p {
            Progress::Language(code) => self.lang = code,
            Progress::WindowOpen {
                window,
                no_speech_prob,
            } => {
                if window < self.provisional.len() {
                    self.provisional[window] =
                        no_speech_prob > paddock_engine::audio::guards::NO_SPEECH_THOLD;
                }
            }
            Progress::Token { window, id, .. } => {
                if self.scale.is_timestamp(id) || window >= self.ids.len() {
                    return;
                }
                self.ids[window].push(id);
                // a decode failure here costs one delta, never the answer:
                // the final transcript is built from the same ids by the
                // authoritative path
                if let Ok(t) = self.tok.decode(&self.ids[window], true) {
                    self.parts[window] = t;
                }
            }
            Progress::WindowDone { window, suppressed } => {
                if window < self.closed.len() {
                    self.closed[window] = true;
                    self.provisional[window] = false;
                    // the guard's verdict landed: this window held no speech,
                    // so whatever the model wrote over it is not transcript.
                    // Nothing of it ever reached a delta (that is what
                    // `provisional` bought), so clearing it here is the only
                    // place it has to be undone.
                    if suppressed {
                        self.ids[window].clear();
                        self.parts[window].clear();
                    }
                }
                while self.frontier < self.closed.len() && self.closed[self.frontier] {
                    self.frontier += 1;
                }
            }
        }
    }

    /// Everything emittable right now, joined exactly as the final text will
    /// be - same function, same language, so the streamed prefix and the
    /// final answer cannot use different seam rules.
    fn view(&self) -> String {
        let v = emittable(&self.parts, self.frontier, &self.provisional);
        paddock_engine::gpu_model::whisper::join_windows(&v, &self.lang)
    }
}

fn sse_data(s: String) -> Result<axum::response::sse::Event, std::convert::Infallible> {
    Ok(axum::response::sse::Event::default().data(s))
}

fn ev_delta(delta: &str) -> String {
    serde_json::json!({ "type": "transcript.text.delta", "delta": delta }).to_string()
}

/// The terminal event. `languages` is the spec's own plural of `{code}`
/// objects; `usage` uses the spec's token variant with counts that are true
/// rather than convenient - "input" is the rows the model actually consumed
/// for the audio (whisper's encoder positions, the generative lanes' expanded
/// audio prompt rows), not a billing figure borrowed from someone else's
/// price list.
fn ev_done(
    text: &str,
    language: Option<&str>,
    audio_rows: usize,
    output: usize,
    verbose: Option<serde_json::Value>,
    // Also inside `paddock_verbose` when that is present, and deliberately: it
    // is documented as the exact object the non-streaming verbose_json
    // returns, and that object carries them. Here as well so a caller who
    // streamed plain `json` - no verbose body to look inside - still learns a
    // span of their audio was refused.
    guards: &[Guard],
    // Same reasoning, same placement: OpenAI's `languages` array is a bare
    // list of `{code}` objects with nowhere to put a probability, a candidate
    // set or a contradiction, so the detail rides beside it under our own name.
    language_detail: Option<&LanguageReport>,
) -> String {
    let mut v = serde_json::json!({
        "type": "transcript.text.done",
        "text": text,
        "usage": {
            "type": "tokens",
            "input_tokens": audio_rows,
            "output_tokens": output,
            "total_tokens": audio_rows + output,
            "input_token_details": {"audio_tokens": audio_rows, "text_tokens": 0},
        },
    });
    // Always present, for the same reason `verbose_body` always fills
    // `language`: a client that reads it should not have to handle the field
    // being there on one lane and gone on another. `UNKNOWN_LANGUAGE` where
    // nothing detected one.
    v["languages"] = serde_json::json!([
        { "code": language.filter(|c| !c.is_empty()).unwrap_or(UNKNOWN_LANGUAGE) }
    ]);
    // PADDOCK EXTENSION, named so it cannot be mistaken for spec. OpenAI's
    // done event carries `text` and nothing else about the transcript, so a
    // caller who asked for verbose_json AND streaming would otherwise have to
    // send the clip a second time to get its segments - a re-decode of the
    // whole file to learn something the server already computed. This is that
    // answer, the exact object the non-streaming verbose_json returns.
    if let Some(x) = verbose {
        v["paddock_verbose"] = x;
    }
    if let Some(detail) = language_detail.and_then(LanguageReport::json) {
        v["paddock_language"] = detail;
    }
    with_guards(v, guards).to_string()
}

/// Handles the whisper lane's `stream=true`. Owns clones of everything it
/// needs - an SSE body outlives the handler that built it, so nothing here
/// may borrow `AppState`.
fn stream_whisper(
    asr: &crate::serving::AsrModel,
    residency: Option<crate::residency::Lease<paddock_engine::transcriber::Transcriber>>,
    mel_windows: Vec<paddock_engine::audio::MelFeatures>,
    // per-window VAD gate; empty = ungated
    speech: Vec<bool>,
    language: paddock_engine::transcriber::LanguageAsk,
    prompt: Vec<u32>,
    grans: Granularities,
    verbose: bool,
    duration_s: f64,
) -> Response {
    use async_stream::stream;
    let transcriber = asr.transcriber.clone();
    let tok = asr.tokenizer.clone();
    // an SSE body outlives the handler, so the language facts are cloned in
    // rather than borrowed off `AppState` like everything else here
    let asked = language.forced.clone();
    let hints = language.hints.clone();
    let langs = asr.languages.clone();
    let scale = asr.time_scale;
    let max_tokens = asr.max_tokens;
    let n = mel_windows.len();
    // whisper's encoder turns each window into a fixed number of positions -
    // 30 s at 0.02 s a step is the 1500 every released checkpoint has, read
    // off the checkpoint's own geometry rather than hardcoded
    let rows_per_window = (scale.window_s / scale.precision).round().max(0.0) as usize;

    let sse = stream! {
        let _residency = residency;
        let (ptx, mut prx) = unbounded_channel();
        let fut = transcriber
            .transcribe(mel_windows, speech, language, prompt, grans.segment, grans.word, max_tokens, Some(ptx));
        tokio::pin!(fut);
        let mut st = WhisperStream::new(n, scale, tok.clone());
        let mut emitted = Emitted::default();
        let mut quiet = false;
        let out = loop {
            tokio::select! {
                // Biased so progress always wins a tie: the reply and the
                // last window's tokens land in the same instant, and taking
                // the reply first would drop deltas we already have.
                biased;
                p = prx.recv(), if !quiet => match p {
                    Some(p) => {
                        st.apply(p);
                        if let Some(d) = emitted.advance(&st.view()) {
                            yield sse_data(ev_delta(&d));
                        }
                    }
                    // the job retired and dropped its sender; stop polling a
                    // closed channel or this select spins
                    None => quiet = true,
                },
                r = &mut fut => break r,
            }
        };
        let mut out = match out {
            Ok(t) => t,
            Err(e) => {
                // A top-level `error` key is what the OpenAI SDKs raise on
                // mid-stream, and it is the only way to fail a response whose
                // 200 status went out with the first delta.
                yield sse_data(serde_json::to_string(&ErrorBody::new("internal_error", e))
                    .unwrap_or_default());
                yield sse_data("[DONE]".to_owned());
                return;
            }
        };

        let ts = &scale;
        suppress_marker_windows(&mut out, &tok, ts);
        let mut parts = Vec::with_capacity(out.windows.len());
        let mut decoded = 0usize;
        for w in &out.windows {
            decoded += w.tokens.len();
            let words: Vec<u32> =
                w.tokens.iter().copied().filter(|&t| !ts.is_timestamp(t)).collect();
            parts.push(tok.decode(&words, true).unwrap_or_default());
        }
        let text = paddock_engine::gpu_model::whisper::join_windows(&parts, &out.language);
        // The authoritative text can only EXTEND what the deltas carried
        // (same tokens, same join), but saying so is not the same as relying
        // on it: emit whatever is left so the concatenation invariant holds
        // even if a decode failed mid-stream and cost a delta.
        //
        // A SUPPRESSED window is what makes that "only extend" a real property
        // rather than a lucky one: the engine clears its tokens, so `parts`
        // above is already empty for it, and nothing of it ever reached a
        // delta because `WhisperStream` held a provisional window back.
        if let Some(d) = emitted.advance(&text) {
            yield sse_data(ev_delta(&d));
        }
        let mut guards = whisper_guards(&out, ts.window_s as f64, duration_s);
        let rep = whisper_report(&out, asked.as_deref(), &hints, &text, &langs);
        guards.extend(language_guard(&rep, duration_s));
        let verbose_json = verbose.then(|| {
            // built for either granularity - the segments are also the word
            // grouping, so a `word`-only request needs them and just does not
            // publish them (same rule as the non-streaming path)
            let segs = grans
                .any()
                .then(|| whisper_segments(&tok, ts, &out, duration_s).ok())
                .flatten();
            let words = whisper_words(segs.as_deref(), grans);
            let segs = grans.segment.then_some(segs).flatten();
            with_guards(
                verbose_body(
                    &text,
                    Some(&out.language),
                    duration_s,
                    segs.as_deref(),
                    ts.window_s as f64,
                    Some(&rep),
                    words.as_deref(),
                ),
                &guards,
            )
        });
        yield sse_data(ev_done(
            &text,
            Some(&out.language),
            n * rows_per_window,
            decoded,
            verbose_json,
            &guards,
            Some(&rep),
        ));
        yield sse_data("[DONE]".to_owned());
    };
    axum::response::Sse::new(sse)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}

/// One whole-clip transcription on a generative ASR lane, start to finish.
///
/// What the live session needs: the file endpoint does the same
/// work inline because it has streaming, verbose bodies and per-word
/// confidence wrapped around it, none of which a re-transcribe-the-buffer pass
/// wants. Returns the transcript and the language, which on Qwen3-ASR the
/// model itself names inside its answer.
pub(crate) async fn generative_pass(
    model: &crate::serving::ServingModel,
    samples: Vec<f32>,
    context: Option<&str>,
    language: Option<&str>,
    max_ctx: usize,
) -> Result<(String, Option<String>), String> {
    let frontend = model.audio_frontend;
    // the mel runs off the async threads for the same reason the file lane's
    // does: host DSP on a runtime worker stalls every other request on it
    let (samples, mel) = tokio::task::spawn_blocking(move || {
        let mel = frontend.features(&samples)?;
        Ok::<_, String>((samples, mel))
    })
    .await
    .map_err(|e| e.to_string())??;

    let (pre_ids, post_ids) = generative_prompt(model, context, language, false)?;
    let prompt_rows = pre_ids.len() + frontend.prompt_rows(samples.len()) + post_ids.len();
    if max_ctx > 0 && prompt_rows + 16 > max_ctx {
        return Err(format!(
            "audio prompt needs {prompt_rows} tokens but max_ctx is {max_ctx} - raise \
             --max-ctx or end the utterance sooner"
        ));
    }
    // the same honest ceiling the file lane uses; a live buffer is
    // bounded by the session's own cap and the VAD, but a degenerate pass
    // would still burn the whole remaining context before either noticed
    let duration_s = samples.len() as f64 / paddock_engine::audio::SAMPLE_RATE as f64;
    let max_tokens = (max_ctx.saturating_sub(prompt_rows))
        .min(guards::token_ceiling(duration_s))
        .max(16);
    let mut text_ids = pre_ids.clone();
    text_ids.extend_from_slice(&post_ids);
    let (tx, mut rx) = unbounded_channel();
    model.engine.submit(GenRequest {
        prompt: text_ids,
        max_tokens,
        sampler: SamplingParams {
            temperature: 0.0,
            ..Default::default()
        },
        stop_tokens: model.stop_tokens.clone(),
        events: tx,
        mm_chunks: Some(vec![
            MmChunk::Text(pre_ids),
            MmChunk::Audio {
                samples,
                mel: Some(mel),
            },
            MmChunk::Text(post_ids),
        ]),
        constraint: None,
        // no logprobs: a live pass has no wire to carry per-word confidence on,
        // and asking for them drops the slot out of the decode-overlap path
        logprobs: None,
        submitted: None,
        canvas_read: None,
        user_turn: false,
    })?;

    let mut ids = Vec::new();
    // a live pass never runs the timestamp instruction (see `generative_prompt`
    // above: `ts_mode` is false), so this is plain transcript text
    let mut guard = GenGuard::new(false);
    while let Some(ev) = rx.recv().await {
        match ev {
            TokenEvent::Token { id, .. } => {
                ids.push(id);
                if guard.token(id) {
                    break;
                }
            }
            TokenEvent::Done(reason, _) => {
                guard.finished(reason);
                break;
            }
            TokenEvent::Error(e) => return Err(e.message),
            TokenEvent::Prefilled { .. } => {}
        }
    }
    drop(rx);
    // A live session has no wire to carry a notice on - its events are
    // transcript text - and it does not need one: LocalAgreement-2 only
    // promotes what two consecutive passes agree on, so a degenerate pass
    // fails to confirm and disappears on the next one. The log is for the
    // operator, the cut is for the GPU.
    if let Some(reason) = guard.cut {
        tracing::warn!(
            reason,
            tokens = ids.len(),
            "live transcription pass cut by a decode guard"
        );
    }
    let raw = model
        .tokenizer
        .decode(&ids, true)
        .map_err(|e| e.to_string())?;
    // Qwen3-ASR wraps its answer in `language X<asr_text>...` unless the language
    // was forced, in which case the envelope is in the prompt and the answer is
    // bare.
    Ok(match frontend {
        crate::serving::AudioFrontend::Qwen3Asr | crate::serving::AudioFrontend::Qwen3AsrMetal
            if language.is_none() =>
        {
            let (lang, text) = parse_output(&raw);
            (text.trim().to_owned(), lang)
        }
        _ => (raw.trim().to_owned(), language.map(str::to_owned)),
    })
}

/// Build the prompt halves that surround the audio rows for a generative ASR
/// lane - the shape both the file endpoint and the live session need, and the
/// one place either of them may render an envelope.
///
/// `context` is the caller's `prompt`: a SYSTEM message on Qwen3-ASR, the task
/// INSTRUCTION on granite-speech (which is what selects raw vs punctuated vs
/// timed on that family). `ts_mode` picks granite's timestamp instruction and
/// is never set by a lane that cannot parse the tags back out.
pub(crate) fn generative_prompt(
    model: &crate::serving::ServingModel,
    context: Option<&str>,
    language: Option<&str>,
    ts_mode: bool,
) -> Result<(Vec<u32>, Vec<u32>), String> {
    Ok(match model.audio_frontend {
        // Qwen3-ASR: the vLLM-parity ChatML envelope (module doc). Written out
        // here rather than rendered, because the GGUF's embedded template is
        // the converter's generic ChatML fallback with no audio branch at all
        // (see `serving::load`) - the official one ships in a file GGUF
        // converters never read.
        crate::serving::AudioFrontend::Qwen3Asr | crate::serving::AudioFrontend::Qwen3AsrMetal => {
            let mut pre = String::new();
            if let Some(ctx) = context {
                let clean = sanitize_user_text(ctx);
                if !clean.is_empty() {
                    pre.push_str(&format!("<|im_start|>system\n{clean}<|im_end|>\n"));
                }
            }
            pre.push_str("<|im_start|>user\n<|audio_start|>");
            let mut post = "<|audio_end|><|im_end|>\n<|im_start|>assistant\n".to_owned();
            if let Some(code) = language {
                post.push_str(&format!("language {}<asr_text>", language_name(code)));
            }
            match (model.tokenizer.encode(&pre), model.tokenizer.encode(&post)) {
                (Ok(a), Ok(b)) => (a, b),
                (Err(e), _) | (_, Err(e)) => {
                    return Err(e.to_string());
                }
            }
        }
        // granite-speech: the CHECKPOINT'S own chat template, rendered exactly
        // the way the chat path renders it, with the model card's prompt order
        // (audio marker first, instruction after).
        //
        // Rendered rather than written out, because the two siblings ship
        // different templates: the base's is a bare `USER: ... ASSISTANT:` and
        // -plus's is the full granite-4 envelope with a system block. This
        // handler used to hardcode the base's for both, which cost -plus its
        // system block on every transcription - and with it the ability to
        // follow an instruction more complicated than "transcribe". Measured:
        // through the hardcoded envelope the timestamp prompt came
        // back as a plain 33-word transcript of the middle of the clip; through
        // the checkpoint's own it comes back fully tagged. The base sibling's
        // template renders byte-identically to the string that was hardcoded,
        // so nothing moved on that lane.
        //
        // The INSTRUCTION is what selects the task on this model, and the card
        // publishes the set: raw transcript, punctuated transcript, keyword
        // biasing, translation, and (on -plus) word timings and speaker
        // labels. So the OpenAI `prompt` field carries it verbatim - which is
        // also how IBM's own llama-server example drives this endpoint
        // (`-F "prompt=transcribe the speech with proper punctuation and
        // capitalization."`). Absent, we send the card's headline prompt.
        crate::serving::AudioFrontend::GraniteSpeech
        | crate::serving::AudioFrontend::GraniteSpeechMetal => {
            let instruction = if ts_mode {
                // `prompt` and this are mutually exclusive and the handler
                // already refused the pair, so nothing of the caller's is
                // being dropped here.
                GRANITE_SPEECH_TS_PROMPT.to_owned()
            } else {
                context
                    .map(sanitize_user_text)
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(|| GRANITE_SPEECH_PROMPT.to_owned())
            };
            let (Some(template), Some(marker), Some(pad)) = (
                model.chat_template.as_deref(),
                model.audio_inline_marker.as_deref(),
                model.audio_pad_id,
            ) else {
                return Err(
                    "granite-speech loaded without its chat template or audio marker".into(),
                );
            };
            // The marker goes into the user content, which is the shape this
            // family's templates are written against (`inline_audio_content`
            // does the same to a parts list on the chat path).
            let messages = [
                serde_json::json!({ "role": "user", "content": format!("{marker}{instruction}") }),
            ];
            let rendered = crate::chat_template::render(template, &messages, None, None)?;
            let mut ids = match model.tokenizer.encode(&rendered) {
                Ok(v) => v,
                Err(e) => return Err(e.to_string()),
            };
            // same BOS rule the chat path applies: templates emit text only,
            // the leading BOS is the tokenizer's business
            if let Some(bos) = model.bos
                && ids.first() != Some(&bos)
            {
                ids.insert(0, bos);
            }
            // Split at the marker's TOKEN, not at the string: the audio rows
            // take its place, so it must not survive into either half.
            let Some(at) = ids.iter().position(|&t| t == pad) else {
                return Err("granite-speech template dropped the audio marker".into());
            };
            (ids[..at].to_vec(), ids[at + 1..].to_vec())
        }
        crate::serving::AudioFrontend::None => {
            return Err("this model does not serve transcription".into());
        }
    })
}

pub async fn handle(State(state): State<Arc<AppState>>, mut mp: Multipart) -> Response {
    // Two lanes serve this endpoint with one wire contract: a dedicated
    // whisper-family model (encoder-decoder on its own thread) or the
    // generative Qwen3-ASR family (audio mmproj on the serving model). The
    // form is parsed once and the lane picked after it, so the OpenAI
    // surface is identical either way.
    //
    // These two refusals - and only these two - answer before draining the
    // body, deliberately: nothing this server can do with the audio depends on
    // reading it, and buffering a 100 MB upload to say "no model is loaded"
    // is worse than the reset a still-uploading client may see instead of the
    // 400. Every refusal that is about the form waits for the whole body.
    let whisper = state.asr.as_ref();
    if whisper.is_none() {
        let Some(model) = state.serving.as_ref() else {
            return err(
                StatusCode::SERVICE_UNAVAILABLE,
                "model_not_loaded",
                "no model is loaded; start paddock with a `model` in config",
            );
        };
        if !model.supports_audio {
            return err(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "this model does not serve transcription (load a whisper-family model, or an \
                 ASR model with its audio mmproj)",
            );
        }
    }

    let mut file: Option<Vec<u8>> = None;
    let mut response_format = "json".to_owned();
    // Some(threshold) once `chunking_strategy` has been read; None leaves the
    // server's own --vad-gate default in charge.
    let mut chunking: Option<f32> = None;
    let mut language: Option<String> = None;
    let mut languages: Vec<String> = Vec::new();
    let mut context: Option<String> = None;
    let mut temperature = 0.0f32;
    let mut grans = Granularities::default();
    let mut stream = false;
    let mut unknown_gran: Option<String> = None;
    let mut want_logprobs = false;
    let mut want_stats = false;
    let mut unknown_include: Option<String> = None;
    let mut unsupported: Vec<String> = Vec::new();
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
        // The SDKs post a repeated field as one part per value under `name[]`;
        // curl users write the bare name. Normalizing here means every arm
        // below is spelled once, and a spec array param cannot slip past the
        // named-refusal list just because the client used brackets.
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
            "chunking_strategy" => {
                let raw = field.text().await.unwrap_or_default();
                match parse_chunking_strategy(&raw) {
                    Ok(t) => chunking = t,
                    Err(e) => return err(StatusCode::BAD_REQUEST, "invalid_request_error", e),
                }
            }
            "response_format" => {
                response_format = field.text().await.unwrap_or_default();
            }
            "language" => language = field.text().await.ok().filter(|s| !s.is_empty()),
            // OpenAI's own plural: "Possible languages of the input audio, in
            // ISO-639-1 format." A HINT, not an instruction, and that is what
            // it does here - the codes become a soft prior over the language
            // posterior, biasing without forbidding. Both spellings
            // arrive: the SDKs post one part per value under `languages[]`,
            // curl users write a bare comma list.
            "languages" => {
                let sent = field.text().await.unwrap_or_default();
                languages.extend(
                    sent.split(',')
                        .map(|s| s.trim().trim_matches(|c| c == '[' || c == ']' || c == '"'))
                        .filter(|s| !s.is_empty() && *s != "null")
                        .map(str::to_owned),
                );
            }
            "prompt" => context = field.text().await.ok().filter(|s| !s.is_empty()),
            "temperature" => {
                temperature = field
                    .text()
                    .await
                    .ok()
                    .and_then(|t| t.parse().ok())
                    .unwrap_or(0.0);
            }
            "timestamp_granularities" => match field.text().await.unwrap_or_default().trim() {
                "segment" => grans.segment = true,
                "word" => grans.word = true,
                "" => {}
                other => unknown_gran = Some(other.to_owned()),
            },
            // httpx spells a Python bool as "true"/"false" in a form part,
            // which is what every SDK on this endpoint posts; a bare `1` is
            // the curl idiom. Anything else asks for nothing.
            "stream" => {
                let sent = field.text().await.unwrap_or_default();
                stream = matches!(sent.trim(), "true" | "True" | "1");
            }
            // PADDOCK EXTENSION on the REQUEST side, and the only one - every
            // other extension here is a response field. It asks for the decode
            // guards' raw signals whether or not any of them fired,
            // which is what turns "nothing tripped" into "nothing
            // tripped, and the closest approach was 0.79 nats". Namespaced so
            // it cannot be mistaken for spec, and opt-in because on a clean
            // clip it is a page of numbers nobody asked for.
            "paddock_guard_stats" => {
                let sent = field.text().await.unwrap_or_default();
                want_stats = matches!(sent.trim(), "true" | "True" | "1");
            }
            // `include` is OpenAI's own way to ask for extra response payload,
            // and `logprobs` is the value that means "tell me how sure you
            // were". That is exactly what the generative ASR lanes can answer
            // and could not be asked for: whisper delivers word
            // confidence on its SEGMENTS, and a model with no timestamp
            // vocabulary has no segments to hang it on. So this parameter is
            // the times-free way in, and it stays a REFUSAL on any lane that
            // cannot honour it rather than being quietly ignored.
            "include" => {
                let sent = field.text().await.unwrap_or_default();
                let asked = sent
                    .trim()
                    .trim_matches(|c| c == '[' || c == ']' || c == '"');
                match asked {
                    "" | "false" | "null" => {}
                    "logprobs" => want_logprobs = true,
                    other => unknown_include = Some(other.to_owned()),
                }
            }
            // Real spec parameters we do not serve. Every one of these used to
            // land in the catch-all below and be dropped without a word. The
            // set is what the pinned SDK enumerates (openai 2.53.0
            // `transcription_create_params`) minus what is handled above.
            "keywords" | "known_speaker_names" | "known_speaker_references" => {
                let sent = field.text().await.unwrap_or_default();
                // an explicitly false/empty value asks for nothing, so it is
                // not a refusal - only a caller actually REQUESTING the
                // feature gets told it is missing
                if !sent.is_empty() && sent != "false" && sent != "null" && sent != "[]" {
                    unsupported.push(name.trim_end_matches("[]").to_owned());
                }
            }
            // `model` names the deployment, and a runner serves exactly one -
            // so it is accepted and has nothing to select. Same as the chat
            // lanes, which answer with their own id whatever was asked for.
            "model" => {
                let _ = field.bytes().await;
            }
            // Anything else is refused by NAME, in OpenAI's own wording, for
            // the reason the JSON lanes deny unknown fields (extract.rs): a
            // form field that is read by nobody is a caller belief the server
            // never contradicts. Every decode-affecting field is handled
            // above, so what lands here really is unknown - a client posting
            // `word_timestamps` or `vad_filter` at this endpoint is asking for
            // something it will not get, and should hear so.
            //
            // Collected rather than returned on the spot: the fields arrive
            // before the file part (that is how the SDKs order the form), so
            // answering here closes the socket while the client is still
            // uploading and it sees a connection reset instead of the 400.
            // Every refusal on this endpoint waits for the body to drain.
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
    let Some(file) = file else {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "missing `file` field",
        );
    };
    if !unsupported.is_empty() {
        // Named individually, because "unsupported parameter" tells a caller
        // nothing about whether to wait for it or work around it.
        let why = |f: &str| match f {
            "keywords" => {
                "keyword biasing is not a separate parameter here - the \
                           granite-speech lane takes it through `prompt`"
            }
            "known_speaker_names" | "known_speaker_references" => {
                "speaker diarization is not served"
            }
            _ => "not served",
        };
        let list = unsupported
            .iter()
            .map(|f| format!("`{f}` ({})", why(f)))
            .collect::<Vec<_>>()
            .join("; ");
        return err(StatusCode::BAD_REQUEST, "invalid_request_error", list);
    }
    if let Some(other) = unknown_include {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!(
                "`include` value '{other}' is not served (logprobs) - `logprobs` asks for \
                 per-word confidence on `paddock_words`"
            ),
        );
    }
    // ---- language, and the candidate set  ----
    //
    // Both refusals here exist because the alternative is a field the server
    // reads and then ignores. A code this checkpoint has never heard of used
    // to reach the decode and fail there as a 500; `language` + `languages`
    // together used to drop the hint without a word.
    if !languages.is_empty() && language.is_some() {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "`language` and `languages` answer the same question two ways - `language` FORCES \
             one and skips detection entirely, `languages` biases detection toward a set. Send \
             one: the code if you know it, the set if you do not.",
        );
    }
    if !languages.is_empty() && whisper.is_none() {
        // Honest rather than accepted-and-dropped: these lanes have no
        // language posterior to bias. Qwen3-ASR names its own language inside
        // its answer and granite-speech detects the input language without
        // reporting it, so on both there is nothing a prior could weight.
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "`languages` biases this server's own language detection, and this model has none \
             to bias - it identifies the language internally and cannot be nudged. Send \
             `language` to force one, or load a whisper-family model \
             (/v1/models publishes `language_detection.hints` per model)",
        );
    }
    // The checkpoint's own map, never a baked list - a fine-tune ships the
    // full set, a converted checkpoint might not, and only the file knows.
    if let Some(asr) = whisper {
        let known = |c: &String| asr.languages.iter().any(|k| k == c);
        let bad: Vec<&String> = language
            .iter()
            .chain(languages.iter())
            .filter(|c| !known(c))
            .collect();
        if !bad.is_empty() && !asr.languages.is_empty() {
            let named = bad
                .iter()
                .map(|c| format!("'{c}'"))
                .collect::<Vec<_>>()
                .join(", ");
            return err(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!(
                    "language {named} is not in this checkpoint's map ({} languages; the full \
                     list is on /v1/models as `language_detection.languages`)",
                    asr.languages.len()
                ),
            );
        }
    }
    if !matches!(
        response_format.as_str(),
        "json" | "text" | "verbose_json" | "srt" | "vtt"
    ) {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!(
                "unsupported response_format '{response_format}' \
                 (json | text | verbose_json | srt | vtt)"
            ),
        );
    }
    // A subtitle file is timestamps - asking for one is asking for segments,
    // so the granularity is implied rather than something the caller has to
    // remember to send (OpenAI does not require it either).
    let subtitles = matches!(response_format.as_str(), "srt" | "vtt");
    // A subtitle DOCUMENT is not a delta stream - half an SRT file is not a
    // partial answer, it is a broken file, and the cue a caller is mid-way
    // through can still move when the window closes. The Studio's own export
    // renders cues client-side from verbose_json for exactly this reason, so
    // the refusal names the route that works.
    if stream && subtitles {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!(
                "`stream` cannot produce {response_format} (a partial subtitle file is not a \
                 partial answer); stream with response_format=verbose_json and render the cues \
                 from `segments`"
            ),
        );
    }
    if let Some(other) = unknown_gran {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!("unknown timestamp_granularities value '{other}' (segment | word)"),
        );
    }
    // OpenAI's own constraint: the granularities only have somewhere to go in
    // verbose_json, so asking for them alongside any other format is a
    // request that cannot be answered as written. srt/vtt are exempt - they
    // carry times by construction.
    if grans.any() && !subtitles && response_format != "verbose_json" {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "timestamp_granularities requires response_format=verbose_json",
        );
    }
    if subtitles {
        grans.segment = true;
    }
    // `include=logprobs` asks for per-word confidence. On the WHISPER lane that
    // confidence needs somewhere to ride, so asking for it turns segments on -
    // the same implication subtitles carry above. Without this, a caller who
    // asked for confidence and nothing else would be answered with a transcript
    // and no confidence anywhere, which is the silent no-op this endpoint
    // refuses to do. The generative lanes have no segments and answer it
    // top-level, so they must not be swept into the timestamp refusal below.
    //
    // `word` granularity already puts the words at the top level with their
    // confidence on them, so it does not need segments forced on top.
    if want_logprobs && whisper.is_some() && !grans.word {
        grans.segment = true;
    }
    // A deliberate deviation, named rather than silent. OpenAI puts `logprobs`
    // on the plain `json` body as a TOKEN-level array; we answer per WORD
    // (whisper's BPE splits "grymheter" into three fragments nobody can read a
    // number off) and per-word confidence only has somewhere to live in
    // verbose_json. The token-level `json` shape is not served yet - this
    // refusal names it so a caller learns which half exists.
    if want_logprobs && response_format != "verbose_json" {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "include=logprobs is served on response_format=verbose_json here, as per-word \
             confidence on `words`/`paddock_words` (OpenAI puts a token-level `logprobs` array \
             on `json`; that shape is not served yet)",
        );
    }
    // A subtitle file is a SEGMENT document - its cues are sentences, and one
    // cue per word is not a subtitle anyone can read. So `word` alongside
    // srt/vtt is a request that cannot be answered as written, and saying so
    // beats silently rendering the segments the caller did not ask for.
    if grans.word && subtitles {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!(
                "`word` timestamps have nowhere to go in {response_format} (its cues are \
                 sentences); ask for response_format=verbose_json, or `segment` here"
            ),
        );
    }
    // granite-speech-plus is the one generative lane that can time words: ask
    // it to and it writes `[T:N]` tags into its answer. Its base
    // sibling cannot, and neither can Qwen3-ASR, so this is a per-CHECKPOINT
    // capability the loader reads out of the GGUF rather than a family one.
    let can_time_words =
        whisper.is_none() && state.serving.as_ref().is_some_and(|s| s.audio_word_times);
    let ts_mode = grans.word && can_time_words;
    if grans.segment && whisper.is_none() {
        // No generative lane produces segments. granite-speech-plus times
        // WORDS, which is not the same thing - its timestamp mode emits no
        // punctuation at all, so there are no sentence boundaries to cut on
        // and inventing some would put made-up spans on a subtitle track.
        let asked = if subtitles {
            "subtitles need segment timestamps and this model produces none \
             (a single cue over the whole clip is not a subtitle file)"
        } else {
            "this model does not produce segment timestamps"
        };
        // The why is per-model, and keyed off the checkpoint rather than off
        // what was asked: a caller who sent only `segment` is exactly the one
        // who has not learned that this one times words, and the refusal is the
        // moment to say so.
        let why = if can_time_words {
            "its timestamp mode emits no punctuation, so there are no sentences to cut cues \
             on. It does time individual WORDS - ask for `word` granularity"
        } else {
            "whisper emits them as vocabulary tokens and the Qwen3-ASR and granite-speech \
             lanes have no equivalent"
        };
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!("{asked} - {why}"),
        );
    }
    if grans.word && whisper.is_some_and(|w| !w.word_times) {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "word timestamps are not implemented on this Whisper backend; segment timestamps are available",
        );
    }
    if grans.word && whisper.is_none() && !ts_mode {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "this model does not produce word timestamps - whisper recovers them from \
             cross-attention and granite-speech-plus writes them into its transcript; \
             neither Qwen3-ASR nor the base granite-speech has any way to answer",
        );
    }
    if ts_mode && context.is_some() {
        // On this family the instruction is the task selector, and there is one
        // instruction slot. Word times need IBM's timestamp prompt in it, so a
        // caller who also sent `prompt` has asked for two tasks in one pass -
        // silently dropping either half is the failure mode this endpoint does
        // not do.
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "`prompt` and `word` timestamps both set the instruction on granite-speech, and \
             it takes one - asking for word times sends the model card's timestamp prompt, \
             so send one or the other",
        );
    }
    // Word times ride on WORDS, and this lane has exactly one thing that cuts a
    // transcript into words: `word_confidences`, which needs the logprobs. So
    // asking for the times asks for them, the same way asking for subtitles asks
    // for segments above. The confidences that come back are real ones - the
    // alternative was a fabricated 1.0 beside every word, which is worse than
    // the decode-overlap fast path this costs.
    if ts_mode {
        want_logprobs = true;
    }

    // decode + resample + mel off the async threads (a 20-minute WAV is real
    // work). The mel runs here rather than engine-side: on the
    // batched engine every admission used to pay the host DSP on the engine
    // thread, serializing concurrent transcriptions behind each other's
    // frontend. Both lanes do it here now - the whisper lane's frontend is
    // per-30s-window, so this is where the clip is CUT into windows too, and
    // the transcriber thread receives features it can encode straight away
    let is_whisper = whisper.is_some();
    // the SERVED model's contract, never a default - Qwen3-ASR and
    // granite-speech share no mel geometry (see `AudioFrontend`)
    let frontend = state
        .serving
        .as_ref()
        .map(|m| m.audio_frontend)
        .unwrap_or_default();
    type Mels = Vec<paddock_engine::audio::MelFeatures>;
    type Decoded = (Vec<f32>, Option<paddock_engine::audio::MelFeatures>, Mels);
    let decoded = tokio::task::spawn_blocking(move || -> Result<Decoded, String> {
        let wav = decode_audio(&file)?;
        if wav.samples.is_empty() {
            return Err("audio file holds no samples".into());
        }
        let samples = resample(&wav.samples, wav.sample_rate, 16000)?;
        if !is_whisper {
            let mel = frontend.features(&samples)?;
            return Ok((samples, Some(mel), Vec::new()));
        }
        // whisper's encoder is a fixed 30 s window, so anything longer is a
        // sequence of them; `whisper_features` zero-pads the tail window
        let mut windows = Vec::new();
        let mut off = 0usize;
        while off < samples.len().max(1) {
            let end = (off + paddock_engine::audio::PAD_SAMPLES).min(samples.len());
            windows.push(paddock_engine::audio::whisper_features(&samples[off..end])?);
            off = end;
        }
        Ok((samples, None, windows))
    })
    .await;
    let (samples, mel, mel_windows) = match decoded {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => return err(StatusCode::BAD_REQUEST, "invalid_request_error", e),
        Err(e) => {
            return err(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                e.to_string(),
            );
        }
    };
    let duration_s = samples.len() as f64 / 16000.0;
    // Which windows the encoder will actually be given. Empty when
    // the gate is off, which is the default and which the engine reads as
    // "every window has speech" - so an ungated server behaves exactly as it
    // did before this existed.
    // The request's `chunking_strategy` decides for this clip; the server's
    // --vad-gate flag is the default when the request says nothing. A caller
    // who asked for server_vad on a lane with no windowed encoder is told so
    // rather than quietly given an ungated decode.
    if chunking.is_some() && !is_whisper {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "chunking_strategy is only served by the whisper family: the generative ASR lanes read the whole clip in one pass, so there are no windows to gate",
        );
    }
    let gate = chunking.or(state.vad_gate.then_some(VAD_GATE_THRESHOLD));
    let speech = match gate {
        Some(t) if is_whisper => window_speech(&samples, 16000, t),
        _ => Vec::new(),
    };

    if let Some(asr) = whisper {
        // `prompt` is whisper's `<|startofprev|>` context: text fed ahead of
        // `<|startoftranscript|>` that biases spelling and vocabulary without
        // being an instruction. It is tokenized here - the tokenizer lives on
        // this side with the detokenizer - and the engine adds the marker.
        let ctx_tokens = match context.as_deref() {
            // whisper's own convention (decoding.py): the prompt is joined to
            // the marker with a leading space, because a byte-level BPE spells
            // a word differently at the start of a line than mid-sentence
            Some(p) => match asr.tokenizer.encode(&format!(" {}", p.trim())) {
                Ok(ids) => {
                    // A tokenizer will happily turn a literal "<|notimestamps|>"
                    // in a caller's prompt into the real control token, which
                    // would rewrite the decode contract from inside a field
                    // that is supposed to be a spelling hint. Whisper's control
                    // block starts at `<|endoftext|>` and runs to the last
                    // timestamp, so one comparison covers all of it. (The
                    // engine refuses these too - this is the half that can
                    // answer 400 instead of 500.)
                    let first = asr
                        .tokenizer
                        .token_to_id("<|endoftext|>")
                        .unwrap_or(u32::MAX);
                    if let Some(&bad) = ids.iter().find(|&&t| t >= first) {
                        return err(
                            StatusCode::BAD_REQUEST,
                            "invalid_request_error",
                            format!(
                                "`prompt` contains whisper control token {bad}; a context \
                                 prompt is plain text"
                            ),
                        );
                    }
                    ids
                }
                Err(e) => {
                    return err(
                        StatusCode::BAD_REQUEST,
                        "invalid_request_error",
                        format!("`prompt` could not be tokenized: {e}"),
                    );
                }
            },
            None => Vec::new(),
        };
        // Gaps stated rather than ignored: temperature sampling is a real
        // upstream feature this lane does not serve yet, and silently dropping
        // a field the caller set is exactly the failure mode the product
        // principles forbid.
        if temperature != 0.0 {
            return err(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "the whisper lane decodes greedily; `temperature` must be 0",
            );
        }
        let ask = paddock_engine::transcriber::LanguageAsk {
            forced: language.clone(),
            hints: languages.clone(),
            strength: paddock_engine::transcriber::DEFAULT_LANGUAGE_PRIOR,
        };
        if stream {
            // Refuse a failed cold load before committing HTTP 200/SSE.
            // The stream owns the lease, including while the client is slow.
            let residency = match asr.transcriber.session_lease().await {
                Ok(lease) => lease,
                Err(error) => return crate::asr_residency::error_response(error),
            };
            return stream_whisper(
                asr,
                residency,
                mel_windows,
                speech,
                ask,
                ctx_tokens,
                grans,
                response_format == "verbose_json",
                duration_s,
            );
        }
        let out = asr
            .transcriber
            .transcribe(
                mel_windows,
                speech,
                ask,
                ctx_tokens,
                grans.segment,
                grans.word,
                asr.max_tokens,
                None,
            )
            .await;
        let mut out = match out {
            Ok(t) => t,
            Err(e) if asr.transcriber.residency().is_some() => {
                return crate::asr_residency::error_response(e);
            }
            Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, "internal_error", e),
        };
        // before anything reads the tokens
        suppress_marker_windows(&mut out, &asr.tokenizer, &asr.time_scale);
        // Timestamp tokens are decode CONTROL, not transcript - but they are
        // not marked special in whisper's tokenizer (`<|0.00|>` and its 1500
        // siblings ship as added tokens with "special": false), so
        // `skip_special_tokens` leaves them in and the text came back reading
        // `<|0.00|> Efter att...<|10.00|>`. Strip them by id here, which is also
        // what keeps `text` byte-identical whether or not times were asked
        // for.
        let ts = &asr.time_scale;
        let mut parts = Vec::with_capacity(out.windows.len());
        for w in &out.windows {
            let words: Vec<u32> = w
                .tokens
                .iter()
                .copied()
                .filter(|&t| !ts.is_timestamp(t))
                .collect();
            match asr.tokenizer.decode(&words, true) {
                Ok(t) => parts.push(t),
                Err(e) => {
                    return err(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "internal_error",
                        e.to_string(),
                    );
                }
            }
        }
        // per-window trim + one space at each seam (whisper's tokenizer
        // round-trips with a leading space by design, but the model does not
        // always emit one at a window's first token - see join_windows)
        let text = paddock_engine::gpu_model::whisper::join_windows(&parts, &out.language);
        // Segments are built whenever either granularity was asked for: they
        // are also how the words are grouped (`whisper_segments` runs the
        // grouping the confidences already use), so a `word`-only request needs
        // them internally and simply does not publish them.
        let segments = if grans.any() {
            match whisper_segments(&asr.tokenizer, &asr.time_scale, &out, duration_s) {
                Ok(s) => Some(s),
                Err(e) => {
                    return err(StatusCode::INTERNAL_SERVER_ERROR, "internal_error", e);
                }
            }
        } else {
            None
        };
        let words = whisper_words(segments.as_deref(), grans);
        let segments = grans.segment.then_some(segments).flatten();
        let mut guards = whisper_guards(&out, asr.time_scale.window_s as f64, duration_s);
        let rep = whisper_report(&out, language.as_deref(), &languages, &text, &asr.languages);
        guards.extend(language_guard(&rep, duration_s));
        let stats = if want_stats {
            whisper_stats(&asr.tokenizer, &asr.time_scale, &out, duration_s)
        } else {
            Vec::new()
        };
        return match response_format.as_str() {
            // `text`, `srt` and `vtt` have nowhere to put a notice, the same
            // way they have nowhere to put `duration`. A suppressed window is
            // simply absent from them - which is the right transcript, just
            // without the explanation the JSON formats carry.
            "text" => (StatusCode::OK, text).into_response(),
            "srt" | "vtt" => {
                let segs = segments.unwrap_or_default();
                let cues: Vec<crate::subtitles::Cue> = segs
                    .iter()
                    .map(|s| crate::subtitles::Cue {
                        start: s.start,
                        end: s.end,
                        text: &s.text,
                    })
                    .collect();
                let doc = if response_format == "srt" {
                    crate::subtitles::srt(&cues)
                } else {
                    crate::subtitles::vtt(&cues)
                };
                (StatusCode::OK, doc).into_response()
            }
            "verbose_json" => Json(with_stats(
                with_guards(
                    verbose_body(
                        &text,
                        Some(&out.language),
                        duration_s,
                        segments.as_deref(),
                        asr.time_scale.window_s as f64,
                        Some(&rep),
                        words.as_deref(),
                    ),
                    &guards,
                ),
                &stats,
            ))
            .into_response(),
            _ => Json(with_stats(
                with_guards(serde_json::json!({ "text": text }), &guards),
                &stats,
            ))
            .into_response(),
        };
    }
    let Some(model) = state.serving.as_ref() else {
        return err(
            StatusCode::SERVICE_UNAVAILABLE,
            "model_not_loaded",
            "no model is loaded",
        );
    };
    let mel = mel.expect("the generative lane computes mel");
    let frontend = model.audio_frontend;
    let audio_tokens = frontend.prompt_rows(samples.len());

    // Prompt scaffolding, split around the MmChunk so the engine expands the
    // audio's token rows itself. Each family gets its own official envelope:
    // rendering one model's through the other's template is fluent nonsense,
    // not an error.
    // A model with no audio frontend is the caller's mistake, not ours - every
    // other way `generative_prompt` can fail is a broken checkpoint.
    if matches!(frontend, crate::serving::AudioFrontend::None) {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "this model does not serve transcription",
        );
    }
    let (pre_ids, post_ids) =
        match generative_prompt(model, context.as_deref(), language.as_deref(), ts_mode) {
            Ok(p) => p,
            Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, "internal_error", e),
        };
    let prompt_rows = pre_ids.len() + audio_tokens + post_ids.len();
    if state.max_ctx > 0 && prompt_rows + 16 > state.max_ctx {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!(
                "audio prompt needs {prompt_rows} tokens but max_ctx is {} - raise --max-ctx \
                 or send shorter audio",
                state.max_ctx
            ),
        );
    }
    // Bounded by what the audio can honestly need, not by a round number
    // a 7.8 s clip that answers with 4000 tokens is degenerate
    // whatever the context window allows, and the old flat 4096 let it run for
    // 16 s of GPU before the context bound stopped it. `token_ceiling` still
    // clears the fastest speech anyone produces by ~2.5x.
    let max_tokens = (state.max_ctx.saturating_sub(prompt_rows))
        .min(guards::token_ceiling(duration_s))
        .max(16);

    let mut text_ids = pre_ids.clone();
    text_ids.extend_from_slice(&post_ids);
    let chunks = vec![
        MmChunk::Text(pre_ids),
        MmChunk::Audio {
            samples,
            mel: Some(mel),
        },
        MmChunk::Text(post_ids),
    ];

    let (tx, mut rx) = unbounded_channel();
    let gen_req = GenRequest {
        prompt: text_ids,
        max_tokens,
        sampler: SamplingParams {
            temperature,
            ..Default::default()
        },
        stop_tokens: model.stop_tokens.clone(),
        events: tx,
        mm_chunks: Some(chunks),
        constraint: None,
        // 2 = the chosen token's logprob AND the runner-up, which is what
        // turns "38%" into "it nearly said 'vill'" - a number a reader cannot
        // act on into an alternative they can judge at a glance. Only when
        // asked: a slot carrying
        // logprobs drops out of the decode-overlap fast path (service.rs
        // `overlap_ok`), so this is a cost the caller opts into rather than
        // one every transcription pays for a number most never read.
        logprobs: want_logprobs.then_some(2),
        submitted: None,
        canvas_read: None,
        user_turn: false,
    };
    if let Err(e) = model.engine.submit(gen_req) {
        return err(StatusCode::INTERNAL_SERVER_ERROR, "internal_error", e);
    }

    if stream {
        // The generative lanes stream for free - they are ordinary decode, so
        // the tokens already arrive one at a time on the engine's own event
        // channel. The only work here is turning a growing id list into a
        // growing STRING without ever emitting bytes the final answer will not
        // contain: same `stable` rule as the whisper lane.
        use async_stream::stream;
        let tok = model.tokenizer.clone();
        let forced = language.clone();
        let verbose = response_format == "verbose_json";
        let sse = stream! {
            let mut ids: Vec<u32> = Vec::new();
            let mut lps: Vec<f32> = Vec::new();
            let mut runners: Vec<Option<(u32, f32)>> = Vec::new();
            let mut emitted = Emitted::default();
            let mut detected: Option<String> = forced.clone();
            let mut guard = GenGuard::new(ts_mode);
            while let Some(ev) = rx.recv().await {
                match ev {
                    TokenEvent::Prefilled { .. } => {}
                    TokenEvent::Token { id, logprobs } => {
                        ids.push(id);
                        if let Some(lp) = logprobs {
                            lps.push(lp.chosen);
                            runners.push(lp.top.get(1).copied());
                        }
                        if guard.token(id) {
                            break;
                        }
                        let Ok(raw) = tok.decode(&ids, true) else { continue };
                        // Qwen3-ASR wraps its answer in `language X<asr_text>...`
                        // unless the language was forced (then the envelope is
                        // in the PROMPT). Splitting per token means the
                        // envelope never reaches a delta - while it is still
                        // being emitted the transcript half is empty.
                        let body = match frontend {
                            crate::serving::AudioFrontend::Qwen3Asr | crate::serving::AudioFrontend::Qwen3AsrMetal if forced.is_none() => {
                                let (lang, t) = parse_output(&raw);
                                if lang.is_some() {
                                    detected = lang;
                                }
                                t
                            }
                            // granite-speech-plus writes its word times into the
                            // transcript, so in timestamp mode the raw stream is
                            // not the answer - the tags and silence markers come
                            // out on the way to the delta, and a tag still being
                            // typed waits (see `granite_stream_ts`).
                            _ if ts_mode => granite_stream_ts(&raw),
                            _ => raw,
                        };
                        if let Some(d) = emitted.advance(stable(body.trim_start())) {
                            yield sse_data(ev_delta(&d));
                        }
                    }
                    TokenEvent::Done(reason, _) => {
                        guard.finished(reason);
                        break;
                    }
                    TokenEvent::Error(e) => {
                        yield sse_data(
                            serde_json::to_string(&ErrorBody::new("internal_error", e.message))
                                .unwrap_or_default(),
                        );
                        yield sse_data("[DONE]".to_owned());
                        return;
                    }
                }
            }
            // A cut leaves the engine still generating into a channel nobody
            // reads; dropping the receiver is what tells it to retire the slot
            // (its next send fails), and it is the entire saving.
            drop(rx);
            let mut guards = guard.guards(duration_s);
            let raw = tok.decode(&ids, true).unwrap_or_default();
            let text = match frontend {
                crate::serving::AudioFrontend::Qwen3Asr | crate::serving::AudioFrontend::Qwen3AsrMetal if forced.is_none() => {
                    let (lang, t) = parse_output(&raw);
                    if lang.is_some() {
                        detected = lang;
                    }
                    t
                }
                _ if ts_mode => granite_strip_ts(&raw),
                _ => raw,
            };
            let text = text.trim();
            if let Some(d) = emitted.advance(text) {
                yield sse_data(ev_delta(&d));
            }
            let words = want_logprobs
                .then(|| generative_words(&tok, &ids, &lps, &runners, transcript_start(&tok, &ids)))
                .flatten()
                .map(|w| if ts_mode { granite_timed_words(&w, duration_s) } else { w });
            let rep = generative_report(forced.as_deref(), detected.as_deref(), text);
            guards.extend(language_guard(&rep, duration_s));
            let verbose_json = verbose.then(|| {
                with_guards(
                    verbose_body(
                        text,
                        detected.as_deref(),
                        duration_s,
                        None,
                        0.0,
                        Some(&rep),
                        words.as_deref(),
                    ),
                    &guards,
                )
            });
            yield sse_data(ev_done(
                text,
                detected.as_deref(),
                audio_tokens,
                ids.len(),
                verbose_json,
                &guards,
                Some(&rep),
            ));
            yield sse_data("[DONE]".to_owned());
        };
        return axum::response::Sse::new(sse)
            .keep_alive(axum::response::sse::KeepAlive::default())
            .into_response();
    }

    let mut ids: Vec<u32> = Vec::new();
    let mut lps: Vec<f32> = Vec::new();
    let mut runners: Vec<Option<(u32, f32)>> = Vec::new();
    let mut guard = GenGuard::new(ts_mode);
    while let Some(ev) = rx.recv().await {
        match ev {
            TokenEvent::Prefilled { .. } => {}
            TokenEvent::Token { id, logprobs } => {
                ids.push(id);
                if let Some(lp) = logprobs {
                    lps.push(lp.chosen);
                    // top[0] is the chosen token on a greedy decode; top[1] is
                    // the road not taken, which is the useful half
                    runners.push(lp.top.get(1).copied());
                }
                if guard.token(id) {
                    break;
                }
            }
            TokenEvent::Done(reason, _) => {
                guard.finished(reason);
                break;
            }
            TokenEvent::Error(e) => {
                return err(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "internal_error",
                    e.message,
                );
            }
        }
    }
    // see the streaming lane: the drop is what retires the slot
    drop(rx);
    let mut guards = guard.guards(duration_s);
    let raw = match model.tokenizer.decode(&ids, true) {
        Ok(t) => t,
        Err(e) => {
            return err(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                e.to_string(),
            );
        }
    };
    let (detected, text) = match frontend {
        // Qwen3-ASR: the forced-language path already put the envelope head
        // in the PROMPT, so raw output is bare transcript there; otherwise
        // strip the `language {X}<asr_text>` envelope the model emits.
        crate::serving::AudioFrontend::Qwen3Asr | crate::serving::AudioFrontend::Qwen3AsrMetal
            if language.is_some() =>
        {
            (language.clone(), raw)
        }
        crate::serving::AudioFrontend::Qwen3Asr | crate::serving::AudioFrontend::Qwen3AsrMetal => {
            parse_output(&raw)
        }
        // granite-speech emits a bare transcript with no envelope, and it
        // detects the input language without reporting it. So `language` here
        // is the caller's own hint echoed back, never a detection we made -
        // and it is null when they sent none rather than a guess.
        //
        // In timestamp mode the transcript is not bare: the times are text, and
        // they come back out here so `text` reads as a transcript rather than
        // as a tag stream. The words keep them (`granite_timed_words`).
        crate::serving::AudioFrontend::GraniteSpeech
        | crate::serving::AudioFrontend::GraniteSpeechMetal
            if ts_mode =>
        {
            (language.clone(), granite_strip_ts(&raw))
        }
        crate::serving::AudioFrontend::GraniteSpeech
        | crate::serving::AudioFrontend::GraniteSpeechMetal => (language.clone(), raw),
        crate::serving::AudioFrontend::None => (None, raw),
    };
    let text = text.trim().to_owned();
    let stats = if want_stats {
        guard.stats(duration_s, ids.len(), &text, &lps)
    } else {
        Vec::new()
    };
    let rep = generative_report(language.as_deref(), detected.as_deref(), &text);
    guards.extend(language_guard(&rep, duration_s));

    match response_format.as_str() {
        "text" => (StatusCode::OK, text).into_response(),
        "verbose_json" => {
            let words = want_logprobs
                .then(|| {
                    generative_words(
                        &model.tokenizer,
                        &ids,
                        &lps,
                        &runners,
                        transcript_start(&model.tokenizer, &ids),
                    )
                })
                .flatten()
                .map(|w| {
                    if ts_mode {
                        granite_timed_words(&w, duration_s)
                    } else {
                        w
                    }
                });
            Json(with_stats(
                with_guards(
                    verbose_body(
                        &text,
                        detected.as_deref(),
                        duration_s,
                        None,
                        0.0,
                        Some(&rep),
                        words.as_deref(),
                    ),
                    &guards,
                ),
                &stats,
            ))
            .into_response()
        }
        _ => Json(with_stats(
            with_guards(serde_json::json!({ "text": text }), &guards),
            &stats,
        ))
        .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The alternative is what makes a mark ACTIONABLE - "nearly said vill"
    /// against "38%" - so the rules for when it is shown are worth pinning:
    /// first token only (mid-word the runner-up is a BPE fragment), and never
    /// when it decodes to the word we already have.
    #[test]
    fn word_json_carries_the_alternative_only_when_it_reads() {
        let w = WordConf {
            word: "ville".into(),
            logprob: -0.5,
            first: 0,
            last: 0,
            start: None,
            end: None,
            alt: Some("vill".into()),
            margin: Some(0.21),
        };
        let v = word_json(&w);
        assert_eq!(v["word"], "ville");
        assert_eq!(v["paddock_alt"], "vill");
        assert!((v["paddock_margin"].as_f64().unwrap() - 0.21).abs() < 1e-6);
        // exp(-0.5): the confidence stays the same transform it always was
        assert!((v["confidence"].as_f64().unwrap() - (-0.5f64).exp()).abs() < 1e-6);

        // a lane that could not answer sends neither key rather than a zero
        // margin, which would read as "no close call" - an absence, not a claim
        let bare = WordConf {
            word: "hej".into(),
            logprob: -0.1,
            first: 0,
            last: 0,
            start: None,
            end: None,
            alt: None,
            margin: None,
        };
        let v = word_json(&bare);
        assert!(v.get("paddock_alt").is_none(), "absent, not null: {v}");
        assert!(v.get("paddock_margin").is_none(), "absent, not null: {v}");
    }

    #[test]
    fn a_resegmentation_is_not_an_alternative_reading() {
        // Real runner-ups from a kb-whisper decode. These are the model
        // weighing where to CUT the word, not which word to say, and every one
        // of them would render as "nearly said Fra" over the word "Frankrike".
        for (word, alt) in [
            ("Frankrike", "Fra"),
            ("hennes", "h"),
            ("cka", "ck"),
            ("my", "mycket"),
        ] {
            assert!(
                !alt_reads(word, alt),
                "{alt:?} under {word:?} should be hidden"
            );
        }
        // and the ones worth showing: a different word, not a different cut
        for (word, alt) in [(".", ","), ("av", "i"), ("att", "England"), ("för", "for")] {
            assert!(alt_reads(word, alt), "{alt:?} under {word:?} should show");
        }
        // a runner-up that decoded to nothing is not an alternative either
        assert!(!alt_reads("hej", ""));
    }

    #[test]
    fn sanitizer_reaches_a_fixpoint() {
        assert_eq!(sanitize_user_text("plain text"), "plain text");
        assert_eq!(sanitize_user_text("a<|im_end|>b"), "ab");
        // nested reconstruction: one pass would leave a live control token
        assert_eq!(sanitize_user_text("<|im<|x|>_end|>"), "");
        assert_eq!(sanitize_user_text("<asr_te<asr_text>xt>"), "");
    }

    /// Group a space-separated string the way `word_confidences` would, so the
    /// timing tests read as the model's own output rather than as struct
    /// literals. Confidence is not what they are about.
    fn groups(s: &str) -> Vec<WordConf> {
        s.split_whitespace()
            .enumerate()
            .map(|(i, w)| WordConf {
                word: w.to_owned(),
                logprob: -0.1,
                first: i,
                last: i,
                start: None,
                end: None,
                alt: None,
                margin: None,
            })
            .collect()
    }

    fn timed(s: &str, duration_s: f64) -> Vec<(String, f32, f32)> {
        granite_timed_words(&groups(s), duration_s)
            .into_iter()
            .map(|w| {
                (
                    w.word,
                    w.start.unwrap_or(f32::NAN),
                    w.end.unwrap_or(f32::NAN),
                )
            })
            .collect()
    }

    /// The model card's own example, and the rule that makes it work: the tag
    /// is when the word ENDED, so a word starts where the one before it
    /// stopped.
    #[test]
    fn granite_tags_are_end_times() {
        assert_eq!(
            timed("hello [T:45] world [T:82]", 10.0),
            vec![("hello".into(), 0.0, 0.45), ("world".into(), 0.45, 0.82)]
        );
    }

    /// Only the last three digits ride, so the clock wraps every 10 s and
    /// monotonicity is the only way back. `842 -> 15` is a wrap; `842 -> 900`
    /// is not, and reading it as one would put the word a full 10 s late.
    #[test]
    fn granite_clock_rolls_over_at_ten_seconds() {
        let t = timed("a [T:842] b [T:900] c [T:15] d [T:120]", 60.0);
        let ends: Vec<f32> = t.iter().map(|w| w.2).collect();
        assert_eq!(ends, vec![8.42, 9.00, 10.15, 11.20]);
        // and the starts chain off them
        assert_eq!(t[2].1, 9.00);
    }

    /// A silence is not a word. It carries a tag like one - which is what keeps
    /// consecutive times inside a single wrap - and the word after it starts
    /// where the silence ENDED, not where the word before the silence stopped.
    #[test]
    fn granite_silences_advance_the_clock_without_becoming_words() {
        let t = timed("stop [T:120] _ [T:640] go [T:700]", 30.0);
        assert_eq!(
            t,
            vec![("stop".into(), 0.0, 1.20), ("go".into(), 6.40, 7.00)]
        );
    }

    /// A tag that never arrived costs its word the times, not its place in the
    /// transcript - `verbose_body` then ships the array as `paddock_words`
    /// rather than a spec `words[]` with an invented start.
    #[test]
    fn granite_word_without_a_tag_keeps_no_times() {
        let w = granite_timed_words(&groups("one [T:50] two three [T:90]"), 10.0);
        let words: Vec<&str> = w.iter().map(|w| w.word.as_str()).collect();
        assert_eq!(words, vec!["one", "two", "three"]);
        assert!(
            w[1].start.is_none() && w[1].end.is_none(),
            "{:?}",
            (w[1].start, w[1].end)
        );
        // the ones that did get a tag still have theirs
        assert_eq!(w[2].end, Some(0.90));
    }

    /// Times land inside the clip. The model counts in its own three-digit
    /// clock and nothing stops it running past the end of a short file; a word
    /// that seeks past the audio is wrong where a listener notices.
    #[test]
    fn granite_times_are_clamped_to_the_clip() {
        let t = timed("late [T:500]", 2.0);
        assert_eq!(t, vec![("late".into(), 0.0, 2.0)]);
    }

    #[test]
    fn granite_strips_tags_and_silences_from_the_transcript() {
        assert_eq!(
            granite_strip_ts("hello [T:45] _ [T:60] world [T:82]"),
            "hello world"
        );
        // a near-miss is TEXT: swallowing characters to salvage a malformed tag
        // corrupts the answer to save a number nobody can use
        assert_eq!(granite_strip_ts("a [T:] b [T:x] c"), "a [T:] b [T:x] c");
        // and the words rejoin to exactly that text, which is the invariant
        let g = groups("hello [T:45] _ [T:60] world [T:82]");
        let joined = granite_timed_words(&g, 10.0)
            .iter()
            .map(|w| w.word.clone())
            .collect::<Vec<_>>()
            .join(" ");
        assert_eq!(
            joined,
            granite_strip_ts("hello [T:45] _ [T:60] world [T:82]")
        );
    }

    /// The delta stream's one guarantee is that it only ever EXTENDS. A tag
    /// being typed would break it - `hello [` renders as a word and then
    /// vanishes two tokens later - so an unclosed bracket holds everything
    /// after it back.
    #[test]
    fn granite_stream_text_never_shrinks() {
        let full = "hello [T:45] world [T:82]";
        let mut prev = String::new();
        for i in 0..=full.len() {
            if !full.is_char_boundary(i) {
                continue;
            }
            let now = granite_stream_ts(&full[..i]);
            assert!(now.starts_with(&prev), "{prev:?} -> {now:?} at {i}");
            prev = now;
        }
        assert_eq!(granite_strip_ts(full), "hello world");
    }

    // ---- decode guards  ----

    fn win(no_speech: f32, avg: f32, stop: paddock_engine::audio::guards::Stop) -> WinOut {
        WinOut {
            no_speech_prob: no_speech,
            avg_logprob: avg,
            stop,
            suppressed: paddock_engine::audio::guards::is_no_speech(no_speech, avg),
            tokens: vec![1, 2, 3],
            ..Default::default()
        }
    }

    use paddock_engine::audio::guards::Stop;
    use paddock_engine::transcriber::{Transcript, Window as WinOut};

    fn clip(windows: Vec<WinOut>, _duration_s: f64) -> Transcript {
        Transcript {
            language: "sv".into(),
            windows,
            timestamps: false,
            language_probs: Vec::new(),
            language_prior_moved: None,
        }
    }

    /// A clean transcription must carry no notice at all - not an empty array.
    /// Every clip in the WER battery goes through here, and a key that is
    /// always present is a key every client learns to ignore.
    #[test]
    fn a_clean_clip_carries_no_notices() {
        let out = clip(
            vec![win(0.01, -0.2, Stop::Eot), win(0.03, -0.15, Stop::Eot)],
            45.0,
        );
        let g = whisper_guards(&out, 30.0, 45.0);
        assert!(g.is_empty(), "{} notices on a clean clip", g.len());
        let body = with_guards(serde_json::json!({ "text": "hej" }), &g);
        assert!(
            body.get("paddock_guards").is_none(),
            "absent, not empty: {body}"
        );
    }

    /// The silence arm, and the span is the point of it: a caller has to learn
    /// which seconds of their recording produced nothing, not just that
    /// something somewhere was dropped.
    #[test]
    fn a_suppressed_window_names_its_span_and_says_the_text_went() {
        let out = clip(
            vec![
                win(0.02, -0.2, Stop::Eot),
                win(0.91, -1.6, Stop::Eot),
                win(0.02, -0.3, Stop::Eot),
            ],
            72.0,
        );
        let g = whisper_guards(&out, 30.0, 72.0);
        assert_eq!(g.len(), 1);
        assert_eq!((g[0].start, g[0].end), (30.0, 60.0));
        assert_eq!(g[0].reason, "no_speech");
        assert!(g[0].dropped);
        let v = guard_json(&g[0]);
        assert_eq!(v["text_dropped"], true);
        assert!(v["note"].as_str().unwrap().contains("discarded"), "{v}");
        // the numbers the verdict was made from ride along, so a caller can
        // disagree with our threshold rather than take it on faith
        assert!((v["no_speech_prob"].as_f64().unwrap() - 0.91).abs() < 1e-5);
    }

    /// The last window is short - the clip ends inside it - and a notice that
    /// seeks past the end of the file is wrong in the one place a listener
    /// would check.
    #[test]
    fn the_last_windows_notice_is_clamped_to_the_clip() {
        let out = clip(
            vec![win(0.02, -0.2, Stop::Eot), win(0.95, -2.0, Stop::Eot)],
            41.7,
        );
        let g = whisper_guards(&out, 30.0, 41.7);
        assert_eq!(g.len(), 1);
        assert!((g[0].end - 41.7).abs() < 1e-9, "{}", g[0].end);
    }

    /// A cut is not a drop, and the difference is the whole shape of the
    /// answer: a looping window had real speech in front of the loop, so its
    /// text stays and the notice says where to stop trusting it.
    #[test]
    fn a_repetition_cut_keeps_its_text() {
        let out = clip(vec![win(0.02, -0.4, Stop::Repetition)], 30.0);
        let g = whisper_guards(&out, 30.0, 30.0);
        assert_eq!(g.len(), 1);
        assert_eq!(g[0].reason, "repetition");
        assert!(!g[0].dropped);
        assert!(guard_json(&g[0])["note"].as_str().unwrap().contains("kept"));
    }

    /// Truncation used to be a 200 with a silently short transcript on both
    /// generative lanes - the engine reported `Length` and nobody read it.
    #[test]
    fn the_generative_lane_reports_its_own_truncation() {
        use paddock_engine::service::FinishReason;
        let mut g = GenGuard::new(false);
        g.finished(FinishReason::Length);
        let out = g.guards(7.8);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].reason, "length");
        assert_eq!((out[0].start, out[0].end), (0.0, 7.8));

        // and a decode that simply stopped says nothing
        let mut clean = GenGuard::new(false);
        clean.finished(FinishReason::Stop);
        assert!(clean.guards(7.8).is_empty());
    }

    /// The shape: one character forever. The guard has to say stop, and
    /// the first stop is what saves the 4000 tokens.
    ///
    /// Both lane kinds, because the split between them is exactly what the
    /// granite regression forced (see `Repetition`) and a structured lane
    /// dropping the entropy test must not have dropped its teeth with it.
    #[test]
    fn the_generative_lane_cuts_a_loop() {
        for structured in [false, true] {
            let mut g = GenGuard::new(structured);
            let mut stopped_at = None;
            for i in 0..500u32 {
                if g.token(7) && stopped_at.is_none() {
                    stopped_at = Some(i);
                }
            }
            assert_eq!(
                stopped_at,
                Some(paddock_engine::audio::guards::PERIOD_MIN_TOKENS as u32 - 1),
                "structured={structured}"
            );
            assert_eq!(g.guards(7.8)[0].reason, "repetition");
        }
    }

    // ---- the delta stream's emit rule ----

    fn parts(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| (*s).to_owned()).collect()
    }

    /// The ordinary case, unchanged by the guards: closed windows in full,
    /// the open one's stable prefix, nothing past the gap.
    #[test]
    fn the_open_window_streams_its_stable_prefix() {
        let p = parts(&["one", "two ", "three"]);
        assert_eq!(emittable(&p, 1, &[false; 3]), vec!["one", "two"]);
        // window 2 exists but window 1 is still open, so it waits
        assert_eq!(emittable(&p, 0, &[false; 3]), vec!["one"]);
    }

    /// And the guard's half: a window that might yet be suppressed puts
    /// nothing on the wire. It is the only thing standing between a
    /// hallucination over silence and a delta stream that has to take bytes
    /// back, which it cannot do.
    #[test]
    fn a_provisional_window_streams_nothing() {
        let p = parts(&["real speech", "thank you for watching"]);
        assert_eq!(emittable(&p, 1, &[false, true]), vec!["real speech"]);
        // once it closes clean it flows normally - provisional is a hold, not
        // a verdict
        assert_eq!(
            emittable(&p, 2, &[false, false]),
            vec!["real speech", "thank you for watching"]
        );
    }

    // ---- no-speech markers  ----

    /// The measured bug: nb-whisper-large answers 8 s of digital silence with
    /// the seven ORDINARY TEXT tokens `[2627, 91, 1771, 496, 9799, 91, 29]`,
    /// which spell `<|nocaptions|>`. Every one is far below `<|endoftext|>`
    /// (50257), so the control-token filter this looked like a job for cannot
    /// see them at all - the model is typing the string, not emitting the id.
    #[test]
    fn a_typed_marker_is_a_no_speech_answer_not_a_transcript() {
        assert!(is_no_speech_marker("<|nocaptions|>"));
        assert!(is_no_speech_marker("  <|nospeech|> "));
    }

    /// The safety, and the reason the rule is whole-text rather than
    /// substring: a transcript that merely mentions the marker is a
    /// transcript.
    #[test]
    fn a_transcript_that_merely_contains_a_marker_survives() {
        assert!(!is_no_speech_marker(
            "the tag <|nocaptions|> appears in the file"
        ));
        assert!(!is_no_speech_marker("<|nocaptions|> and then she spoke"));
        assert!(!is_no_speech_marker(""));
        assert!(!is_no_speech_marker("Ja"));
    }

    /// The streaming half. `<|noc` must never reach a delta - two tokens later
    /// the whole thing is withdrawn, and a delta stream cannot take bytes
    /// back. Same rule `granite_stream_ts` uses for a tag being typed.
    #[test]
    fn a_marker_being_typed_never_reaches_a_delta() {
        let full = "<|nocaptions|>";
        for i in 1..=full.len() {
            if !full.is_char_boundary(i) {
                continue;
            }
            assert!(marker_hold(&full[..i]), "{:?} should be held", &full[..i]);
        }
        // ...and a window that carries on into real words stops matching
        assert!(!marker_hold("<|nocaptions|> and then she spoke"));
        assert!(!marker_hold("Ja det er det"));
    }

    /// The whole point, at the seam that matters: a held marker contributes
    /// nothing to the view, so `done.text` (which is built after suppression)
    /// and the deltas agree - the invariant the gate asserts.
    #[test]
    fn the_stream_emits_nothing_for_a_marker_window() {
        let p = parts(&["real speech", "<|nocaptions|>"]);
        assert_eq!(emittable(&p, 1, &[false; 2]), vec!["real speech"]);
        // even complete, because complete is exactly when it is about to be
        // suppressed
        assert_eq!(
            emittable(&p, 2, &[false; 2]),
            vec!["real speech", "<|nocaptions|>"]
        );
    }

    // ---- the VAD window gate  ----

    /// Deterministic broadband noise at `amp`, `secs` long at 16 kHz - a
    /// quiet room, or a loud one, with no RNG state to depend on.
    fn noise(secs: f32, amp: f32) -> Vec<f32> {
        let mut st: u32 = 12345;
        (0..(16000.0 * secs) as usize)
            .map(|_| {
                st = st.wrapping_mul(1103515245).wrapping_add(12345) & 0x7FFF_FFFF;
                amp * (st as f32 / 0x3FFF_FFFF as f32 - 1.0)
            })
            .collect()
    }

    /// The case the gate exists for. roest-v3 answers 8 s of quiet room tone
    /// with "Ja det er det. Det er det. Det er det. Det er det" and no decode
    /// guard catches it; a window marked speechless
    /// never reaches the encoder to say it.
    #[test]
    fn a_quiet_room_is_not_a_window_worth_decoding() {
        assert_eq!(
            window_speech(&noise(8.0, 0.004), 16000, VAD_GATE_THRESHOLD),
            vec![false]
        );
    }

    #[test]
    fn a_window_with_sound_in_it_is_kept() {
        assert_eq!(
            window_speech(&noise(8.0, 0.30), 16000, VAD_GATE_THRESHOLD),
            vec![true]
        );
    }

    /// The gate is per WINDOW, so a clip that is quiet for its first 30 s and
    /// speaks after must lose the first and keep the second - that asymmetry
    /// is the whole saving on a real recording.
    #[test]
    fn the_gate_is_per_window_not_per_clip() {
        let mut pcm = noise(30.0, 0.004);
        pcm.extend(noise(20.0, 0.30));
        assert_eq!(
            window_speech(&pcm, 16000, VAD_GATE_THRESHOLD),
            vec![false, true]
        );
    }

    /// An empty or sub-window clip still gets exactly one flag: the engine
    /// indexes this parallel to the mel windows, and a short read there would
    /// silently gate a window that was never judged.
    #[test]
    fn the_flags_are_parallel_to_the_windows() {
        assert_eq!(window_speech(&[], 16000, VAD_GATE_THRESHOLD).len(), 1);
        assert_eq!(
            window_speech(&noise(0.4, 0.30), 16000, VAD_GATE_THRESHOLD).len(),
            1
        );
        // 61 s is three windows even though the last is a sliver
        assert_eq!(
            window_speech(&noise(61.0, 0.30), 16000, VAD_GATE_THRESHOLD).len(),
            3
        );
    }

    // ---- the language contradiction  ----

    /// The whole point, end to end at the response layer: a forced language,
    /// a transcript in another one, and a notice that says so on every body
    /// shape - including plain `json`, which has no verbose object to look
    /// inside and is exactly where this failure was invisible.
    #[test]
    fn a_translated_transcript_gets_a_notice_on_every_body() {
        let german = "Hallo, ich teste und sehe, wie es funktioniert.";
        let rep = generative_report(Some("sv"), None, german);
        let g = language_guard(&rep, 6.0).expect("sv asked, German written");
        assert_eq!(g.reason, "language_mismatch");
        // the text STAYS - it may be a perfectly good translation, and that is
        // the caller's call to make, not this endpoint's
        assert!(!g.dropped);
        assert_eq!((g.start, g.end), (0.0, 6.0));
        let v = guard_json(&g);
        assert!(v["note"].as_str().unwrap().contains("translat"), "{v}");
        // the remedy names the language it actually came back in
        let hint = v["hint"].as_str().unwrap();
        assert!(hint.contains("language=de"), "{hint}");
        assert!(hint.contains("German"), "{hint}");
        // and the plain-json body carries it
        let body = with_guards(serde_json::json!({ "text": german }), &[g]);
        assert_eq!(body["paddock_guards"][0]["reason"], "language_mismatch");
    }

    /// A correct transcript carries nothing. Same rule as every other notice
    /// here: the key is absent, and a warning that fires on good output is
    /// worse than no warning at all.
    #[test]
    fn a_transcript_in_the_asked_language_carries_no_notice() {
        let swedish = "Jag testar transkriberingen och ser hur den fungerar i praktiken.";
        let rep = generative_report(Some("sv"), None, swedish);
        assert!(language_guard(&rep, 6.0).is_none());
        let body = with_guards(serde_json::json!({ "text": swedish }), &[]);
        assert!(body.get("paddock_guards").is_none(), "{body}");
    }

    /// Qwen3-ASR names its language as an English NAME inside its own answer
    /// ("language Swedish<asr_text>...") while everything else here speaks
    /// codes. The report normalises, or the check could never compare
    /// anything on this lane (is the standing question of what the
    /// spec `language` field itself should carry).
    #[test]
    fn a_model_that_names_its_language_is_normalised_to_a_code() {
        let german = "Hallo, ich teste und sehe, wie es funktioniert.";
        let rep = generative_report(None, Some("Swedish"), german);
        assert_eq!(rep.code.as_deref(), Some("sv"));
        let v = rep.json().unwrap();
        // "reported", not "detected": the model said so, we did not measure it
        assert_eq!(v["source"], "reported");
        assert!(
            v.get("probability").is_none(),
            "no number was ever produced: {v}"
        );
        assert!(language_guard(&rep, 6.0).is_some());
        // lowercase too - the same checkpoint has been seen writing both
        assert_eq!(
            generative_report(None, Some("swedish"), german)
                .code
                .as_deref(),
            Some("sv")
        );
    }

    /// granite-speech detects the input language and reports nothing. With no
    /// hint from the caller either there is genuinely nothing to say, and the
    /// response says exactly that rather than inventing a language.
    #[test]
    fn a_lane_that_reports_no_language_says_so() {
        let rep = generative_report(
            None,
            None,
            "Jag testar transkriberingen och ser hur den går.",
        );
        assert!(rep.code.is_none());
        assert!(language_guard(&rep, 6.0).is_none(), "nothing to contradict");
        let v = rep.json();
        // the text check still ran; whether it could ANSWER is a separate
        // question, and either way no language is asserted
        assert!(v.as_ref().is_none_or(|v| v.get("code").is_none()), "{v:?}");
    }

    #[test]
    fn output_envelope_parses() {
        let (l, t) = parse_output("language English<asr_text>hello world");
        assert_eq!(l.as_deref(), Some("English"));
        assert_eq!(t, "hello world");
        let (l, t) = parse_output("no envelope at all");
        assert!(l.is_none());
        assert_eq!(t, "no envelope at all");
    }

    /// `chunking_strategy` parsing. The param_surface probes cannot reach this:
    /// the endpoint needs an ASR model loaded before it looks at the form, so
    /// the parser is checked on its own OUTPUT here - the same lesson as
    /// output_config, where "parse it then discard it" survived every HTTP test.
    mod chunking {
        use super::super::{VAD_GATE_THRESHOLD, parse_chunking_strategy};

        #[test]
        fn auto_and_server_vad_both_mean_server_vad() {
            // server VAD is the only chunking this endpoint does, so "let the
            // server decide" and "use server VAD" resolve to the same thing
            for raw in ["auto", "\"auto\"", r#"{"type":"server_vad"}"#] {
                assert_eq!(
                    parse_chunking_strategy(raw).expect("ok"),
                    Some(VAD_GATE_THRESHOLD),
                    "{raw}"
                );
            }
        }

        #[test]
        fn threshold_is_a_real_dial() {
            let got = parse_chunking_strategy(r#"{"type":"server_vad","threshold":0.8}"#)
                .expect("ok")
                .expect("some");
            assert!((got - 0.8).abs() < 1e-6, "got {got}");
        }

        #[test]
        fn the_millisecond_dials_are_refused_rather_than_swallowed() {
            // They move where a live session cuts an utterance. This endpoint
            // gates whole 30 s encoder windows on whether they hold speech, so
            // there is no boundary for them to move - accepting them would let
            // a caller believe they had tuned something.
            for k in ["prefix_padding_ms", "silence_duration_ms"] {
                let raw = format!(r#"{{"type":"server_vad","{k}":300}}"#);
                let e = parse_chunking_strategy(&raw).expect_err("should refuse");
                assert!(e.contains(k), "{e}");
                assert!(e.contains("encoder windows"), "should say why: {e}");
            }
        }

        #[test]
        fn malformed_shapes_name_themselves() {
            for (raw, want) in [
                ("\"eager\"", "must be \"auto\""),
                (
                    r#"{"type":"semantic_vad"}"#,
                    "unsupported chunking_strategy type",
                ),
                (r#"{"threshold":0.5}"#, "needs a `type`"),
                (r#"{"type":"server_vad","threshold":3}"#, "must be 0..=1"),
                (
                    r#"{"type":"server_vad","threshold":"loud"}"#,
                    "must be a number",
                ),
                (
                    r#"{"type":"server_vad","nope":1}"#,
                    "unsupported chunking_strategy field",
                ),
            ] {
                let e = parse_chunking_strategy(raw).expect_err("should refuse");
                assert!(e.contains(want), "expected {want:?} for {raw}, got {e}");
            }
        }
    }
}
