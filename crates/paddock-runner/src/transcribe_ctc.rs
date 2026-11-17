//! Nemotron 3.5 Transcribe on `/v1/audio/transcriptions`: word times
//! (`timestamp_granularities[]=word` on `verbose_json`) and speaker-
//! attributed transcription (`response_format=diarized_json`, OpenAI's
//! shape), both read off the checkpoint's CTC timestamp head - the object
//! the chat lane returns as `ctc_timestamps` (see `ctc_timestamps.rs`).
//!
//! Word times keep the plain transcription prompt, so `text` is exactly the
//! transcript the same request without them gets. Each timed word also
//! carries the decode's confidence (with `include=logprobs`) and the speaker
//! the model's own speaker branch has active under it ([`timed_words`],
//! by time overlap - the shared attribution contract). Speaker LABELS in the
//! transcript need the model to write its `<spk:N>` turns, which it does
//! under the model card's own speaker-attribution instruction
//! ([`SPEAKER_PROMPT`]) - a different task, so a diarized transcript can
//! differ from a plain one.

use std::collections::BTreeMap;

use paddock_models::diarization::{Segment, words::Word};
use serde_json::{Value, json};

use crate::transcriptions::WordConf;

/// The model card's "who said what, and when" instruction, verbatim - the
/// request NVIDIA documents for word timestamps with speaker attribution.
pub(crate) const SPEAKER_PROMPT: &str = "Transcribe this audio with punctuation and \
    capitalization. Include all spoken words and verbal fillers. Write numbers in their \
    standard written form. Put speaker tags <spk:0> <spk:1> <spk:2> <spk:3> on the \
    transcription.";

/// What this lane refuses before any audio is decoded, or None.
pub(crate) fn refusal(
    diarized: bool,
    head: bool,
    granularities: bool,
    logprobs: bool,
    language: bool,
    prompt: bool,
) -> Option<&'static str> {
    if diarized && !head {
        return Some(
            "response_format=diarized_json labels speakers, which this server answers on \
             Nemotron 3.5 Transcribe (its CTC timestamp head); this model has no speaker labels",
        );
    }
    if diarized && granularities {
        return Some(
            "timestamp_granularities is not available with response_format=diarized_json \
             (OpenAI's contract): its segments carry the times",
        );
    }
    if diarized && logprobs {
        return Some("include=logprobs is not served with response_format=diarized_json");
    }
    if diarized && language {
        return Some(
            "`language` is not served with response_format=diarized_json: speaker attribution \
             runs the model card's own instruction, which has no language slot - the model \
             transcribes the language it hears",
        );
    }
    if diarized && prompt {
        return Some(
            "`prompt` is not served with response_format=diarized_json: speaker attribution \
             runs the model card's own instruction",
        );
    }
    None
}

/// A transcript decoded without its control tokens, whitespace collapsed
/// (the dropped `<spk:N>` turns leave doubled spaces behind).
pub(crate) fn plain_text(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// An aligned word that is an event tag (`<ev:laughter>`): a word of the
/// alignment, not of the transcript's text.
fn is_tag(w: &str) -> bool {
    w.starts_with('<') && w.ends_with('>')
}

/// The letters two spellings of one transcript are matched on: letters and
/// digits only, so the punctuation the alignment splits off or drops cannot
/// put the two out of step.
fn letters(w: &str) -> impl Iterator<Item = char> + '_ {
    w.chars().filter(|c| c.is_alphanumeric())
}

/// How far ahead a resync looks, in letters, on each side: the alignment's
/// copy of the transcript only ever LOSES text against the decode's (the
/// loop guard trims a runaway repeat), so the decode side gets the long reach.
const RESYNC_ALIGNED: usize = 48;
const RESYNC_DECODED: usize = 4096;
/// Letters that must agree for a resync to count.
const RESYNC_RUN: usize = 4;

/// For each aligned word, the decoded word owning its first letter, and
/// whether both words START on that letter (the decoded word's runner-up
/// then reads as an alternative to this word).
///
/// Both lists spell one transcript - the alignment reads the decode's own
/// text - but cut it differently: the alignment splits zh / ja into
/// characters and words on dashes, the decode groups tokens on whitespace.
/// So they are walked letter by letter; a stretch where the letters disagree
/// is skipped by a resync, and the aligned words inside it own nothing
/// rather than a neighbour's confidence.
fn owners(aligned: &[&str], decoded: &[&str]) -> Vec<Option<(usize, bool)>> {
    let a: Vec<(char, usize, bool)> = aligned
        .iter()
        .enumerate()
        .flat_map(|(k, w)| letters(w).enumerate().map(move |(i, c)| (c, k, i == 0)))
        .collect();
    let b: Vec<(char, usize, bool)> = decoded
        .iter()
        .enumerate()
        .flat_map(|(k, w)| letters(w).enumerate().map(move |(i, c)| (c, k, i == 0)))
        .collect();
    let agree = |x: usize, y: usize| {
        let n = RESYNC_RUN.min(a.len() - x).min(b.len() - y);
        // a short run counts only where it reaches the end of both
        (n == RESYNC_RUN || (x + n == a.len() && y + n == b.len()))
            && (0..n).all(|t| a[x + t].0 == b[y + t].0)
    };
    let mut out = vec![None; aligned.len()];
    let (mut i, mut j) = (0usize, 0usize);
    'walk: while i < a.len() && j < b.len() {
        if a[i].0 == b[j].0 {
            let slot = &mut out[a[i].1];
            if slot.is_none() {
                *slot = Some((b[j].1, a[i].2 && b[j].2));
            }
            i += 1;
            j += 1;
            continue;
        }
        // the nearest point both agree again, fewest letters skipped first
        for d in 1..=RESYNC_ALIGNED + RESYNC_DECODED {
            for da in 0..=d.min(RESYNC_ALIGNED) {
                let db = d - da;
                if db > RESYNC_DECODED || i + da >= a.len() || j + db >= b.len() {
                    continue;
                }
                if agree(i + da, j + db) {
                    i += da;
                    j += db;
                    continue 'walk;
                }
            }
        }
        break;
    }
    out
}

/// The alignment's words as OpenAI's `{word, start, end}`, in time order,
/// each with two `paddock_` extensions where they can be answered:
///
///   * the decode's confidence (`paddock_logprob` / `paddock_confidence`,
///     plus `paddock_alt` / `paddock_margin` on the word a runner-up was
///     recorded for) when `decoded` carries the request's logprobs;
///   * the speaker under the word (`paddock_speaker`, `paddock_speakers`,
///     `paddock_speaker_status`, `paddock_speaker_coverage`): time overlap
///     with the model's own speaker timeline (the alignment's `diarization`),
///     the same contract a separate diarization model answers with. Off
///     (`speakers = false`) on a live utterance: each pass numbers its own
///     speakers, so the labels would not agree across a session.
pub(crate) fn timed_words(
    ctc: &Value,
    decoded: Option<&[WordConf]>,
    duration_s: f64,
    speakers: bool,
) -> Vec<Value> {
    let aligned: Vec<&Value> = ctc["words"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|w| !w["word"].as_str().is_some_and(is_tag))
        .collect();
    let spelled: Vec<&str> = aligned
        .iter()
        .map(|w| w["word"].as_str().unwrap_or(""))
        .collect();
    let owned = decoded.map(|d| {
        let dw: Vec<&str> = d.iter().map(|w| w.word.as_str()).collect();
        owners(&spelled, &dw)
    });
    let words: Vec<Word> = aligned
        .iter()
        .enumerate()
        .map(|(k, w)| {
            let mut metadata = BTreeMap::new();
            if let (Some(d), Some(Some((j, first)))) = (decoded, owned.as_ref().map(|o| o[k])) {
                let c = &d[j];
                metadata.insert("paddock_logprob".to_owned(), json!(c.logprob));
                metadata.insert("paddock_confidence".to_owned(), json!(c.logprob.exp()));
                if first {
                    if let Some(alt) = &c.alt {
                        metadata.insert("paddock_alt".to_owned(), json!(alt));
                    }
                    if let Some(m) = c.margin {
                        metadata.insert("paddock_margin".to_owned(), json!(m));
                    }
                }
            }
            Word {
                word: spelled[k].to_owned(),
                start: w["start"].as_f64(),
                end: w["end"].as_f64(),
                metadata,
            }
        })
        .collect();
    // no speakers rather than a refusal: the times and the text stand alone
    let attributed = if speakers {
        paddock_models::diarization::words::attribute(words.clone(), &timeline(ctc), duration_s)
            .unwrap_or(words)
    } else {
        words
    };
    attributed
        .into_iter()
        .map(|mut w| {
            for k in ["speaker", "speakers", "speaker_status", "speaker_coverage"] {
                if let Some(v) = w.metadata.remove(k) {
                    w.metadata.insert(format!("paddock_{k}"), v);
                }
            }
            serde_json::to_value(w).unwrap_or(Value::Null)
        })
        .collect()
}

/// The model's own speaker timeline (its speaker branch's active spans).
fn timeline(ctc: &Value) -> Vec<Segment> {
    ctc["diarization"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|s| {
            Some(Segment {
                speaker: usize::try_from(s["speaker"].as_u64()?).ok()?,
                start: s["start"].as_f64()?,
                end: s["end"].as_f64()?,
            })
        })
        .collect()
}

/// `paddock_diarization` on `verbose_json`: the speaker timeline the words'
/// `paddock_speaker*` fields were attributed against, in the shape a
/// diarization model's own answer has (minus its words, which ride the spec
/// array). `preset` names where the timeline came from: the transcriber's
/// built-in speaker branch, not one of a diarization model's presets.
pub(crate) fn speaker_timeline(ctc: &Value, model: &str, duration_s: f64) -> Value {
    let segments = timeline(ctc);
    let speakers: std::collections::BTreeSet<usize> = segments.iter().map(|s| s.speaker).collect();
    json!({
        "model": model,
        "duration": duration_s,
        "segments": segments,
        "num_speakers": speakers.len(),
        "preset": "transcribe",
        "attribution": "time_overlap_v1",
    })
}

/// OpenAI's speaker labels: `A`, `B`, ... in order of first appearance in
/// the recording (the model's tag numbers carry no order of their own - a
/// transcript may open on `<spk:1>`); words before any tag are `unknown`.
fn label(seen: &mut Vec<String>, tag: &str) -> String {
    if tag.parse::<u32>().is_err() {
        return "unknown".into();
    }
    let k = seen.iter().position(|t| t == tag).unwrap_or_else(|| {
        seen.push(tag.to_owned());
        seen.len() - 1
    });
    match u8::try_from(k) {
        Ok(k) if k < 26 => char::from(b'A' + k).to_string(),
        _ => format!("S{k}"),
    }
}

/// Characters the alignment times one at a time (Chinese and Japanese):
/// they join without a space.
fn character_unit(w: &str) -> bool {
    let mut c = w.chars();
    let (Some(ch), None) = (c.next(), c.next()) else {
        return false;
    };
    matches!(ch as u32, 0x3040..=0x30FF | 0x31F0..=0x31FF | 0x3400..=0x4DBF | 0x4E00..=0x9FFF
        | 0xF900..=0xFAFF | 0x20000..=0x2FA1F)
}

/// OpenAI's diarized segments from the alignment: each run of consecutive
/// (time-ordered) words with one speaker tag, its span from the first
/// word's start to the last word's end, its text those words.
pub(crate) fn segments(ctc: &Value) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    let mut run: Option<(String, f64, f64, String, bool)> = None;
    let flush =
        |out: &mut Vec<Value>, seen: &mut Vec<String>, r: (String, f64, f64, String, bool)| {
            out.push(json!({
                "type": "transcript.text.segment",
                "id": format!("seg_{:03}", out.len()),
                "start": r.1,
                "end": r.2,
                "text": r.3,
                "speaker": label(seen, &r.0),
            }));
        };
    for w in ctc["words"].as_array().into_iter().flatten() {
        let (Some(word), Some(start), Some(end)) =
            (w["word"].as_str(), w["start"].as_f64(), w["end"].as_f64())
        else {
            continue;
        };
        if is_tag(word) {
            continue;
        }
        let speaker = w["speaker"].as_str().unwrap_or("None").to_owned();
        let unit = character_unit(word);
        match &mut run {
            Some(r) if r.0 == speaker => {
                if !(unit && r.4) {
                    r.3.push(' ');
                }
                r.3.push_str(word);
                r.2 = r.2.max(end);
                r.4 = unit;
            }
            _ => {
                if let Some(r) = run.take() {
                    flush(&mut out, &mut seen, r);
                }
                run = Some((speaker, start, end, word.to_owned(), unit));
            }
        }
    }
    if let Some(r) = run {
        flush(&mut out, &mut seen, r);
    }
    out
}

/// The `diarized_json` body (OpenAI's `TranscriptionDiarized`).
pub(crate) fn diarized_body(
    text: &str,
    duration_s: f64,
    ctc: &Value,
    input_tokens: usize,
    output_tokens: usize,
) -> Value {
    json!({
        "task": "transcribe",
        "duration": duration_s,
        "text": text,
        "segments": segments(ctc),
        "usage": {
            "type": "tokens",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_token_details": {"audio_tokens": input_tokens, "text_tokens": 0},
        },
    })
}

/// The alignment's error (`alignment_failed`, `multiple_audio`, ...), when
/// it had one: the body then carries a `timestamps` guard saying so, since an
/// empty `words` / `segments` with nothing saying why would read as silence.
pub(crate) fn error(ctc: &Value) -> Option<&str> {
    ctc["error"].as_str()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obj() -> Value {
        json!({
            "words": [
                {"word": "Hello,", "start": 0.32, "end": 0.611, "speaker": "0"},
                {"word": "<ev:laughter>", "start": 0.66, "end": 0.9, "speaker": "0"},
                {"word": "there.", "start": 0.7, "end": 1.0, "speaker": "0"},
                {"word": "Hi.", "start": 1.04, "end": 1.287, "speaker": "1"},
                {"word": "你", "start": 2.0, "end": 2.1, "speaker": "1"},
                {"word": "好", "start": 2.1, "end": 2.2, "speaker": "1"},
                {"word": "Yes.", "start": 2.5, "end": 2.9, "speaker": "0"},
            ],
            "diarization": [],
            "speaker_tag_to_diarization_speaker": {"0": 2, "1": 0},
            "error": null,
        })
    }

    #[test]
    fn segments_are_speaker_runs_in_time_order() {
        let s = segments(&obj());
        assert_eq!(s.len(), 3);
        assert_eq!(s[0]["text"], "Hello, there.");
        assert_eq!(
            (s[0]["start"].as_f64(), s[0]["end"].as_f64()),
            (Some(0.32), Some(1.0))
        );
        assert_eq!(s[0]["speaker"], "A");
        assert_eq!(s[1]["text"], "Hi. 你好");
        assert_eq!(s[1]["speaker"], "B");
        assert_eq!(s[2]["id"], "seg_002");
        assert_eq!(s[2]["type"], "transcript.text.segment");
    }

    #[test]
    fn words_drop_event_tags_and_the_transcripts_speaker_tags() {
        // no timeline, no logprobs: OpenAI's word and nothing that would
        // read as an answer (the speaker fields say "unassigned")
        let w = timed_words(&obj(), None, 3.0, true);
        assert_eq!(w.len(), 6);
        assert_eq!(
            (
                w[1]["word"].as_str(),
                w[1]["start"].as_f64(),
                w[1]["end"].as_f64()
            ),
            (Some("there."), Some(0.7), Some(1.0))
        );
        assert_eq!(w[1]["paddock_speaker_status"], "unassigned");
        assert!(w[1].get("speaker").is_none());
    }

    #[test]
    fn an_alignment_error_is_read_and_the_body_keeps_its_shape() {
        let mut o = obj();
        o["error"] = json!("alignment_failed");
        assert_eq!(error(&o), Some("alignment_failed"));
        assert_eq!(error(&obj()), None);
        let b = diarized_body("x", 1.0, &o, 10, 2);
        assert_eq!(b["usage"]["total_tokens"], 12);
        assert_eq!(b["segments"].as_array().map(Vec::len), Some(3));
    }

    fn conf(word: &str, logprob: f32, alt: Option<&str>) -> WordConf {
        WordConf {
            word: word.into(),
            logprob,
            first: 0,
            last: 0,
            start: None,
            end: None,
            alt: alt.map(Into::into),
            margin: alt.map(|_| 0.25),
        }
    }

    #[test]
    fn timed_words_carry_the_decodes_confidence_and_the_speaker_under_them() {
        let mut o = obj();
        o["diarization"] = json!([
            {"speaker": 0, "start": 0.3, "end": 1.02},
            {"speaker": 1, "start": 1.03, "end": 2.95},
        ]);
        // the decode groups on whitespace: one word for the two characters,
        // no event tag
        let d = [
            conf("Hello,", -0.1, Some("Hallo")),
            conf("there.", -0.2, None),
            conf("Hi.", -0.3, None),
            conf("你好", -0.4, Some("你")),
            conf("Yes.", -0.5, None),
        ];
        let w = timed_words(&o, Some(&d), 3.0, true);
        assert_eq!(w.len(), 6);
        assert_eq!(
            (w[0]["word"].as_str(), w[0]["start"].as_f64()),
            (Some("Hello,"), Some(0.32))
        );
        assert_eq!(w[0]["paddock_logprob"], json!(-0.1f32));
        assert_eq!(w[0]["paddock_alt"], "Hallo");
        assert_eq!(w[0]["paddock_speaker"], 0);
        assert_eq!(w[0]["paddock_speaker_status"], "assigned");
        assert_eq!(w[2]["paddock_speaker"], 1);
        // both characters own the decode's word; only the first starts it
        assert_eq!(w[3]["paddock_logprob"], json!(-0.4f32));
        assert_eq!(w[4]["paddock_logprob"], json!(-0.4f32));
        assert_eq!(w[3]["paddock_alt"], "你");
        assert!(w[4].get("paddock_alt").is_none());
        assert_eq!(w[5]["paddock_logprob"], json!(-0.5f32));
        // the spec keys are never shadowed by bare extension names
        assert!(
            w.iter()
                .all(|x| x.get("speaker").is_none() && x.get("confidence").is_none())
        );
        // without logprobs: times and speakers, no confidence
        let bare = timed_words(&o, None, 3.0, true);
        assert!(bare[0].get("paddock_confidence").is_none());
        assert_eq!(bare[5]["paddock_speaker"], 1);
        // a live utterance: times and confidence, no speaker fields at all
        let live = timed_words(&o, Some(&d), 3.0, false);
        assert_eq!(live[0]["paddock_logprob"], json!(-0.1f32));
        assert!(
            live.iter()
                .all(|x| x.get("paddock_speaker_status").is_none())
        );
        let t = speaker_timeline(&o, "m", 3.0);
        assert_eq!(t["num_speakers"], 2);
        assert_eq!(t["attribution"], "time_overlap_v1");
    }

    #[test]
    fn a_stretch_the_alignment_lost_owns_nothing_and_the_walk_resyncs() {
        // the loop guard trimmed a repeat from the alignment's copy
        let aligned = ["the", "end", "is", "near."];
        let decoded = ["the", "end", "the", "end", "the", "end", "is", "near."];
        let o = owners(&aligned, &decoded);
        assert_eq!(o[0], Some((0, true)));
        assert_eq!(o[1], Some((1, true)));
        assert_eq!(o[2], Some((6, true)));
        assert_eq!(o[3], Some((7, true)));
        // a dash the alignment split on: both halves own the decode's word
        let o = owners(&["well", "known"], &["well-known"]);
        assert_eq!(o, vec![Some((0, true)), Some((0, false))]);
        // nothing in common: nothing owned
        assert_eq!(owners(&["abc"], &["xyz"]), vec![None]);
    }

    #[test]
    fn refusals() {
        assert!(refusal(true, false, false, false, false, false).is_some());
        assert!(refusal(true, true, true, false, false, false).is_some());
        assert!(refusal(true, true, false, false, true, false).is_some());
        assert!(refusal(true, true, false, true, false, false).is_some());
        assert!(refusal(true, true, false, false, false, false).is_none());
        // word times with logprobs are joined now
        assert!(refusal(false, true, true, true, true, true).is_none());
        assert_eq!(plain_text(" Hello.  Hi. "), "Hello. Hi.");
        let mut seen = Vec::new();
        assert_eq!(label(&mut seen, "1"), "A");
        assert_eq!(label(&mut seen, "0"), "B");
        assert_eq!(label(&mut seen, "1"), "A");
        assert_eq!(label(&mut seen, "None"), "unknown");
    }
}
