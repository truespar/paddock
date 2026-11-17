//! `transcriptions.rs`'s unit tests, kept beside it to hold the handler file
//! under the length ceiling.

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
