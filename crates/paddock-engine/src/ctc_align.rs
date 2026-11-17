//! Word timestamps and speaker labels for Nemotron 3.5 Transcribe: the
//! alignment of a generated t-SOT transcript (`<spk:N>` turns) to the
//! checkpoint's CTC head, as NVIDIA's NeMo plugin publishes it (vLLM's
//! `ctc_timestamps` response object). NeMo's `MultiSpeakerSOTWordTimestamp
//! Aligner` is the reference; the algorithm is its, the code ours:
//!
//!   1. the transcript cleaned and split into words per speaker stream
//!      (em dashes separate words, `<ev:*>` tags are words of their own,
//!      `<sty:*>` tags dropped; zh / ja text aligns per lexical character)
//!   2. each stream tokenized with the head's SentencePiece and expanded to
//!      blank-interleaved CTC targets; the log-prob columns kept are the
//!      blank plus the transcript's tokens (`Prepared::columns`, blank first)
//!   3. speaker tags mapped to the diarizer's columns by matching each
//!      stream's token counts to the greedy CTC onsets heard while a column
//!      is active (IDF-weighted cosine, linear assignment)
//!   4. per stream a banded max-sum CTC Viterbi over the frames its speaker
//!      is active, with the log activity as a prior, the band centred on
//!      unique token-trigram anchors
//!   5. word edges from the path, a calibrated correction from the log-prob
//!      profile around each edge, then a refinement against the 10 ms
//!      speaker activity segments
//!
//! The arithmetic follows the reference's dtypes (float32 DP and profiles,
//! float64 times and similarities, NumPy's interpolation and rounding), so
//! the same log probabilities give the same words and times. A stream with
//! no legal path loses its words without an error, as the reference does.
//! Thai would need the reference's dictionary segmenter and is refused.

use serde_json::{Value, json};
use unicode_normalization::UnicodeNormalization;
use unicode_properties::{GeneralCategoryGroup, UnicodeGeneralCategory};

use paddock_tokenizer::spm::{SPACE, SpmBpe};

/// The diarizer's speaker columns.
pub const SPEAKERS: usize = 8;

/// The aligner's hyperparameters: the checkpoint's
/// `ctc_timestamps.alignment_config.parameters`.
#[derive(Clone, Debug)]
pub struct AlignParams {
    pub speaker_logprob_weight: f64,
    pub speaker_activity_threshold: f64,
    pub support_activity_threshold: f64,
    pub vad_threshold: f64,
    pub band_ratio: f64,
    pub band_min_states: usize,
    pub band_full_below_frames: usize,
    pub maximum_token_len: f64,
    /// in encoder frames (the refinement cap is twice it in CTC frames)
    pub max_word_frames: usize,
    pub onset_snap_seconds: f64,
    pub refinement_end_blend: f64,
    pub outside_word_duration: f64,
    pub max_correction: f64,
    pub min_onset_spacing: f64,
    pub min_word_duration: f64,
    pub grid_seconds: f64,
    pub profile_radius: usize,
    pub profile_points: usize,
    pub logprob_floor: f64,
    pub logprob_ceiling: f64,
    /// start / end edge weights over the 2 x points profile, and offsets
    pub interpolation: Option<([Vec<f64>; 2], [f64; 2])>,
    pub epsilon: f64,
}

impl AlignParams {
    /// From the checkpoint config's `ctc_timestamps` block.
    pub fn from_config(ctc: &Value) -> Result<Self, String> {
        let p = &ctc["alignment_config"]["parameters"];
        let f = |k: &str| {
            p[k].as_f64()
                .ok_or_else(|| format!("ctc_timestamps parameter {k} is missing"))
        };
        let u = |k: &str| f(k).map(|v| v as usize);
        let points = u("boundary_profile_points")?;
        let interpolation = if p["interpolation_enabled"].as_bool().unwrap_or(false) {
            let bi = &p["boundary_interpolation"];
            let coeffs: Vec<Vec<f64>> = serde_json::from_value(bi["coefficients"].clone())
                .map_err(|e| format!("boundary_interpolation.coefficients: {e}"))?;
            let offset: Vec<f64> = serde_json::from_value(bi["offset"].clone())
                .map_err(|e| format!("boundary_interpolation.offset: {e}"))?;
            if coeffs.len() != 2 || offset.len() != 2 {
                return Err("boundary_interpolation must have two edges".into());
            }
            Some((
                [
                    expand_weights(&coeffs[0], points)?,
                    expand_weights(&coeffs[1], points)?,
                ],
                [offset[0], offset[1]],
            ))
        } else {
            None
        };
        Ok(Self {
            speaker_logprob_weight: f("speaker_logprob_weight")?,
            speaker_activity_threshold: f("speaker_activity_threshold")?,
            support_activity_threshold: f("support_activity_threshold")?,
            vad_threshold: f("vad_threshold")?,
            band_ratio: f("band_ratio")?,
            band_min_states: u("band_min_states")?,
            band_full_below_frames: u("band_full_below_frames")?,
            maximum_token_len: f("maximum_token_len")?,
            max_word_frames: u("max_word_frames")?,
            onset_snap_seconds: f("onset_snap_seconds")?,
            refinement_end_blend: f("refinement_end_blend")?,
            outside_word_duration: f("boundary_outside_word_duration_seconds")?,
            max_correction: f("boundary_max_correction_seconds")?,
            min_onset_spacing: f("boundary_min_onset_spacing_seconds")?,
            min_word_duration: f("boundary_min_word_duration_seconds")?,
            grid_seconds: f("boundary_grid_seconds")?,
            profile_radius: u("boundary_profile_radius_frames")?,
            profile_points: points,
            logprob_floor: f("boundary_logprob_floor")?,
            logprob_ceiling: f("boundary_logprob_ceiling")?,
            interpolation,
            epsilon: 1e-6,
        })
    }
}

/// The compressed calibration (K DCT modes, channel sum and difference
/// interleaved) expanded to the 2 x points profile weights of one edge.
fn expand_weights(coeffs: &[f64], points: usize) -> Result<Vec<f64>, String> {
    if coeffs.is_empty() || coeffs.len() > 2 * points || coeffs.iter().any(|c| !c.is_finite()) {
        return Err("boundary_interpolation coefficients out of shape".into());
    }
    let n = points as f64;
    let temporal = |p: usize, freq: usize| {
        let mut v =
            (std::f64::consts::PI * (p as f64 + 0.5) * freq as f64 / n).cos() * (2.0 / n).sqrt();
        if freq == 0 {
            v /= 2f64.sqrt();
        }
        v
    };
    let r2 = 2f64.sqrt();
    // basis[row][col]: rows 0..points the token channel, points.. the blank
    let basis = |row: usize, col: usize| {
        let (p, freq, diff) = (row % points, col / 2, col % 2 == 1);
        let t = temporal(p, freq) / r2;
        if row >= points && diff { -t } else { t }
    };
    Ok((0..2 * points)
        .map(|row| {
            coeffs
                .iter()
                .enumerate()
                .map(|(k, c)| c * basis(row, k))
                .sum()
        })
        .collect())
}

/// One alignment unit.
#[derive(Clone, Debug)]
pub struct Word {
    pub word: String,
    pub speaker_tag: Option<u32>,
    character: bool,
    token_ids: Vec<u32>,
}

/// One speaker stream's targets.
#[derive(Clone, Debug)]
pub struct Stream {
    pub tag: Option<u32>,
    pub words: Vec<Word>,
    /// blank-interleaved targets as compact columns (blank = 0)
    labels: Vec<u32>,
    state_to_word: Vec<Option<usize>>,
}

/// A transcript ready for its log probabilities.
#[derive(Clone, Debug)]
pub struct Prepared {
    pub streams: Vec<Stream>,
    /// the CTC classes whose log probabilities the alignment reads, blank
    /// first, then the transcript's tokens ascending
    pub columns: Vec<u32>,
}

/// The decoder control tokens a transcript can still carry, replaced by a
/// space (speaker tags and event tags excepted, as the reference does).
pub fn clean(transcript: &str, special_tokens: &[String]) -> String {
    let mut t = transcript.to_owned();
    for tok in special_tokens {
        if tok.is_empty() || is_speaker_tag(tok) || is_added(tok) {
            continue;
        }
        t = t.replace(tok.as_str(), " ");
    }
    t.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn is_speaker_tag(s: &str) -> bool {
    let l = s.to_ascii_lowercase();
    l.strip_prefix("<spk:")
        .and_then(|r| r.strip_suffix('>'))
        .is_some_and(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()))
}

/// `<ev:*>` and not one of the excluded `<spk:*>` / `<sty:stt:*>` /
/// `<sty:end:*>` (case-insensitive globs).
fn is_added(tok: &str) -> bool {
    let l = tok.to_lowercase();
    let glob = |pfx: &str| l.starts_with(pfx) && l.ends_with('>');
    !(glob("<spk:") || glob("<sty:stt:") || glob("<sty:end:")) && glob("<ev:")
}

/// Every `<tag>` (no spaces or angle brackets inside) set apart by spaces.
fn isolate_tags(t: &str) -> String {
    let mut out = String::with_capacity(t.len() + 8);
    let mut rest = t;
    while let Some(i) = rest.find('<') {
        out.push_str(&rest[..i]);
        let tail = &rest[i..];
        let body = &tail[1..];
        let end = body.find(|c: char| c == '<' || c == '>' || c.is_whitespace());
        match end {
            Some(j) if j > 0 && body[j..].starts_with('>') => {
                out.push(' ');
                out.push_str(&tail[..j + 2]);
                out.push(' ');
                rest = &tail[j + 2..];
            }
            _ => {
                out.push('<');
                rest = body;
            }
        }
    }
    out.push_str(rest);
    out
}

/// `parse_sot_words`: words with their speaker tag (None before the first
/// tag of a tagged transcript, 0 throughout an untagged one) and turn.
pub fn parse(transcript: &str) -> Vec<Word> {
    let t = isolate_tags(&transcript.replace('\u{2014}', " "));
    // split on speaker tags (case-insensitive), keeping their numbers
    let mut parts: Vec<(Option<u32>, &str)> = Vec::new();
    let mut tagged = false;
    let mut cur: Option<u32> = None;
    let mut start = 0usize;
    let lower = t.to_ascii_lowercase();
    let mut search = 0usize;
    while let Some(i) = lower[search..].find("<spk:").map(|i| i + search) {
        let after = &t[i + 5..];
        let digits: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
        if !digits.is_empty() && after[digits.len()..].starts_with('>') {
            parts.push((cur, &t[start..i]));
            cur = digits.parse().ok();
            tagged = true;
            start = i + 5 + digits.len() + 1;
            search = start;
        } else {
            search = i + 1;
        }
    }
    parts.push((cur, &t[start..]));
    let mut words = Vec::new();
    for (k, (tag, part)) in parts.iter().enumerate() {
        let speaker_tag = if !tagged {
            Some(0)
        } else if k == 0 {
            None
        } else {
            *tag
        };
        for w in part.split_whitespace() {
            if w.starts_with('<') && w.ends_with('>') && !is_added(w) {
                continue;
            }
            words.push(Word {
                word: w.to_owned(),
                speaker_tag,
                character: false,
                token_ids: Vec::new(),
            });
        }
    }
    words
}

fn is_tag(w: &str) -> bool {
    w.len() >= 3
        && w.starts_with('<')
        && w.ends_with('>')
        && !w[1..w.len() - 1].contains(['<', '>'])
}

/// The scripts the reference recognises by Unicode name prefix (HANGUL,
/// HIRAGANA / KATAKANA, CJK UNIFIED / COMPATIBILITY IDEOGRAPH, THAI), as
/// the code point blocks those names cover.
fn script(c: char) -> Option<&'static str> {
    let u = c as u32;
    let hangul = matches!(u, 0x1100..=0x11FF | 0x3131..=0x318E | 0xA960..=0xA97C
        | 0xAC00..=0xD7A3 | 0xD7B0..=0xD7C6 | 0xD7CB..=0xD7FB);
    let kana = matches!(u, 0x3041..=0x3096 | 0x309D..=0x309F | 0x30A0..=0x30FF
        | 0x31F0..=0x31FF | 0x1B000..=0x1B122 | 0x1B132 | 0x1B150..=0x1B152
        | 0x1B155 | 0x1B164..=0x1B167);
    let han = matches!(u, 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAD9
        | 0x20000..=0x2A6DF | 0x2A700..=0x2B739 | 0x2B740..=0x2B81D | 0x2B820..=0x2CEA1
        | 0x2CEB0..=0x2EBE0 | 0x2EBF0..=0x2EE5D | 0x2F800..=0x2FA1D | 0x30000..=0x3134A
        | 0x31350..=0x323AF);
    let thai = matches!(u, 0x0E01..=0x0E3A | 0x0E3F..=0x0E5B);
    if hangul {
        Some("ko")
    } else if kana {
        Some("ja")
    } else if han {
        Some("zh")
    } else if thai {
        Some("th")
    } else {
        None
    }
}

fn lexical_chars(text: &str) -> Vec<char> {
    text.nfc()
        .filter(|c| {
            matches!(
                c.general_category_group(),
                GeneralCategoryGroup::Letter
                    | GeneralCategoryGroup::Mark
                    | GeneralCategoryGroup::Number
            )
        })
        .collect()
}

/// `_timestamp_units` with `auto` language detection.
fn units(words: Vec<Word>) -> Result<Vec<Word>, String> {
    let mut found = [false; 4];
    for w in words.iter().filter(|w| !is_tag(&w.word)) {
        for c in w.word.chars() {
            match script(c) {
                Some("ko") => found[0] = true,
                Some("ja") => found[1] = true,
                Some("zh") => found[2] = true,
                Some("th") => found[3] = true,
                _ => {}
            }
        }
    }
    // first match in the reference's order: ko, ja, zh, th
    if found[0] || !(found[1] || found[2] || found[3]) {
        return Ok(words);
    }
    if !found[1] && !found[2] {
        return Err("Thai timestamp units need a dictionary word segmenter".into());
    }
    let mut out = Vec::new();
    for w in words {
        if is_tag(&w.word) {
            out.push(w);
            continue;
        }
        for c in lexical_chars(&w.word) {
            out.push(Word {
                word: c.to_string(),
                character: true,
                ..w.clone()
            });
        }
    }
    Ok(out)
}

/// `_character_token_ids`: the character's own piece, else its encoding
/// without bare `▁` pieces, refusing what does not decode back.
fn character_ids(sp: &SpmBpe, c: &str, blank: u32) -> Result<Vec<u32>, String> {
    let id = sp.piece_to_id(c);
    let ids: Vec<u32> = if id != sp.unk_id() {
        vec![id]
    } else {
        sp.encode(c)
            .into_iter()
            .filter(|&t| sp.piece(t) != Some(SPACE))
            .collect()
    };
    let back: String = lexical_chars(&sp.decode(&ids)).into_iter().collect();
    if ids.is_empty() || ids.contains(&sp.unk_id()) || ids.iter().any(|&t| t >= blank) || back != c
    {
        return Err(format!("Unrepresentable character: {c:?}"));
    }
    Ok(ids)
}

/// Group in first-appearance order of the tag.
fn groups(words: &[Word]) -> Vec<(Option<u32>, Vec<usize>)> {
    let mut g: Vec<(Option<u32>, Vec<usize>)> = Vec::new();
    for (i, w) in words.iter().enumerate() {
        match g.iter_mut().find(|(t, _)| *t == w.speaker_tag) {
            Some((_, v)) => v.push(i),
            None => g.push((w.speaker_tag, vec![i])),
        }
    }
    g
}

/// Steps 1-2: the cleaned transcript's streams and columns. `ctc_frames` is
/// the CTC timeline's length (twice the encoder frames).
pub fn prepare(
    transcript: &str,
    sp: &SpmBpe,
    blank: u32,
    ctc_frames: usize,
) -> Result<Prepared, String> {
    let mut words = units(parse(transcript))?;
    for (_, idx) in groups(&words) {
        let ids = stream_ids(
            sp,
            &idx.iter().map(|&i| &words[i]).collect::<Vec<_>>(),
            blank,
        )?;
        for (i, t) in idx.into_iter().zip(ids) {
            words[i].token_ids = t;
        }
    }
    let mut streams = Vec::new();
    for (tag, idx) in groups(&words) {
        let ws: Vec<Word> = idx.iter().map(|&i| words[i].clone()).collect();
        let mut labels = vec![blank];
        let mut state_to_word = vec![None];
        for (wi, w) in ws.iter().enumerate() {
            for &t in &w.token_ids {
                labels.extend([t, blank]);
                state_to_word.extend([Some(wi), None]);
            }
        }
        let tokens: Vec<u32> = labels.iter().skip(1).step_by(2).copied().collect();
        let repeats = tokens.windows(2).filter(|p| p[0] == p[1]).count();
        if tokens.len() + repeats > ctc_frames {
            return Err(format!(
                "Speaker {tag:?} transcript is too long for the CTC timeline."
            ));
        }
        streams.push(Stream {
            tag,
            words: ws,
            labels,
            state_to_word,
        });
    }
    let mut columns: Vec<u32> = streams
        .iter()
        .flat_map(|s| s.labels.iter().copied())
        .filter(|&l| l != blank)
        .collect();
    columns.sort_unstable();
    columns.dedup();
    columns.insert(0, blank);
    let column_of: std::collections::HashMap<u32, u32> = columns
        .iter()
        .enumerate()
        .map(|(c, &l)| (l, c as u32))
        .collect();
    for s in &mut streams {
        for l in &mut s.labels {
            *l = column_of[l];
        }
    }
    Ok(Prepared { streams, columns })
}

/// One stream's token ids per word (`_tokenize_word_stream`).
fn stream_ids(sp: &SpmBpe, words: &[&Word], blank: u32) -> Result<Vec<Vec<u32>>, String> {
    if words.iter().any(|w| w.character) {
        return words
            .iter()
            .map(|w| {
                if w.character {
                    character_ids(sp, &w.word, blank)
                } else {
                    stream_ids(sp, &[*w], blank).map(|mut v| v.remove(0))
                }
            })
            .collect();
    }
    let text = words
        .iter()
        .map(|w| w.word.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let full = sp.encode(&text);
    if full.is_empty() || full.iter().any(|&t| t >= blank) {
        return Err("Tokenizer produced IDs outside the non-blank CTC vocabulary.".into());
    }
    let mut cache: std::collections::HashMap<&str, Vec<u32>> = std::collections::HashMap::new();
    let mut per = Vec::with_capacity(words.len());
    for w in words {
        let ids = cache
            .entry(w.word.as_str())
            .or_insert_with(|| sp.encode(&w.word))
            .clone();
        if ids.is_empty() {
            return Err("SentencePiece produced an empty word tokenization.".into());
        }
        per.push(ids);
    }
    if per.iter().flatten().copied().collect::<Vec<_>>() != full {
        return Err("Per-word and full-stream SentencePiece tokenization disagree.".into());
    }
    Ok(per)
}

/// What the alignment reads besides the transcript: the head's log
/// probabilities over `Prepared::columns`, the diarizer's 80 ms pooled
/// probabilities and its 10 ms activity labels.
pub struct Inputs<'a> {
    /// `[ctc_frames][columns]`
    pub logp: &'a [f32],
    pub ctc_frames: usize,
    /// `[encoder frames][8]`
    pub speaker80: &'a [[f32; SPEAKERS]],
    /// one bit per speaker a 10 ms frame
    pub labels10: &'a [u8],
    pub duration: f64,
}

/// A 10 ms activity segment.
#[derive(Clone, Copy, Debug)]
pub struct Segment {
    pub speaker: usize,
    pub start: f64,
    pub end: f64,
}

/// The labels' runs per speaker, sorted by (start, end, speaker).
pub fn segments(labels10: &[u8], duration: f64) -> Vec<Segment> {
    let mut out = Vec::new();
    for s in 0..SPEAKERS {
        let mut t = 0usize;
        while t < labels10.len() {
            if labels10[t] >> s & 1 == 1 {
                let a = t;
                while t < labels10.len() && labels10[t] >> s & 1 == 1 {
                    t += 1;
                }
                let start = a as f64 * 0.01;
                let end = (t as f64 * 0.01).min(duration);
                if end > start {
                    out.push(Segment {
                        speaker: s,
                        start,
                        end,
                    });
                }
            } else {
                t += 1;
            }
        }
    }
    out.sort_by(|a, b| {
        a.start
            .total_cmp(&b.start)
            .then(a.end.total_cmp(&b.end))
            .then(a.speaker.cmp(&b.speaker))
    });
    out
}

/// `torch.linspace(0, n - 1, frames)` in float32, then the linear
/// resampling of the (clamped) probabilities onto it.
fn resample(probs: &[[f32; SPEAKERS]], frames: usize) -> Vec<[f32; SPEAKERS]> {
    let n = probs.len();
    let clamp = |p: &[f32; SPEAKERS]| p.map(|v| v.clamp(0.0, 1.0));
    if n == frames {
        return probs.iter().map(clamp).collect();
    }
    if n == 1 {
        return vec![clamp(&probs[0]); frames];
    }
    let (start, end) = (0f32, (n - 1) as f32);
    let step = if frames > 1 {
        (end - start) / (frames - 1) as f32
    } else {
        0.0
    };
    let half = frames / 2;
    (0..frames)
        .map(|i| {
            let pos = if i < half {
                start + step * i as f32
            } else {
                end - step * (frames - i - 1) as f32
            };
            let lo = pos.floor() as usize;
            let hi = (pos.ceil() as usize).min(n - 1);
            let f = pos - lo as f32;
            let (a, b) = (clamp(&probs[lo.min(n - 1)]), clamp(&probs[hi]));
            let mut o = [0f32; SPEAKERS];
            for s in 0..SPEAKERS {
                o[s] = a[s] * (1.0 - f) + b[s] * f;
            }
            o
        })
        .collect()
}

/// scipy's rectangular `linear_sum_assignment` (minimisation), Crouse's
/// shortest augmenting path with the library's scan order and ties.
fn linear_sum_assignment(cost: &[Vec<f64>]) -> Vec<(usize, usize)> {
    let nr0 = cost.len();
    if nr0 == 0 {
        return Vec::new();
    }
    let nc0 = cost[0].len();
    let transposed = nc0 < nr0;
    let (nr, nc) = if transposed { (nc0, nr0) } else { (nr0, nc0) };
    let c = |i: usize, j: usize| if transposed { cost[j][i] } else { cost[i][j] };
    let mut u = vec![0f64; nr];
    let mut v = vec![0f64; nc];
    let mut shortest = vec![0f64; nc];
    let mut path = vec![usize::MAX; nc];
    let mut col4row = vec![usize::MAX; nr];
    let mut row4col = vec![usize::MAX; nc];
    let mut sr = vec![false; nr];
    let mut sc = vec![false; nc];
    let mut remaining = vec![0usize; nc];
    for cur in 0..nr {
        let mut min_val = 0f64;
        let mut i = cur;
        let mut num_remaining = nc;
        for (it, r) in remaining.iter_mut().enumerate() {
            *r = nc - it - 1;
        }
        sr.fill(false);
        sc.fill(false);
        shortest.fill(f64::INFINITY);
        let mut sink = usize::MAX;
        while sink == usize::MAX {
            let mut index = usize::MAX;
            let mut lowest = f64::INFINITY;
            sr[i] = true;
            for (it, &j) in remaining[..num_remaining].iter().enumerate() {
                let r = min_val + c(i, j) - u[i] - v[j];
                if r < shortest[j] {
                    path[j] = i;
                    shortest[j] = r;
                }
                if shortest[j] < lowest || (shortest[j] == lowest && row4col[j] == usize::MAX) {
                    lowest = shortest[j];
                    index = it;
                }
            }
            min_val = lowest;
            if min_val == f64::INFINITY || index == usize::MAX {
                return Vec::new(); // infeasible
            }
            let j = remaining[index];
            if row4col[j] == usize::MAX {
                sink = j;
            } else {
                i = row4col[j];
            }
            sc[j] = true;
            num_remaining -= 1;
            remaining[index] = remaining[num_remaining];
        }
        u[cur] += min_val;
        for ii in 0..nr {
            if sr[ii] && ii != cur {
                u[ii] += min_val - shortest[col4row[ii]];
            }
        }
        for jj in 0..nc {
            if sc[jj] {
                v[jj] -= min_val - shortest[jj];
            }
        }
        let mut j = sink;
        loop {
            let ii = path[j];
            row4col[j] = ii;
            std::mem::swap(&mut col4row[ii], &mut j);
            if ii == cur {
                break;
            }
        }
    }
    let mut pairs: Vec<(usize, usize)> = (0..nr).map(|i| (i, col4row[i])).collect();
    if transposed {
        pairs = pairs.into_iter().map(|(i, j)| (j, i)).collect();
        pairs.sort_unstable();
    }
    pairs
}

/// Step 3: stream tag -> diarizer column.
fn speaker_mapping(
    cols: usize,
    streams: &[Stream],
    speaker: &[[f32; SPEAKERS]],
    threshold: f32,
    greedy: &[usize],
) -> Vec<Option<usize>> {
    if streams.is_empty() {
        return Vec::new();
    }
    let t = greedy.len();
    let mut observed = vec![vec![0u64; cols]; SPEAKERS];
    for f in 0..t {
        let peak = greedy[f] != 0 && (f == 0 || greedy[f] != greedy[f - 1]);
        if !peak {
            continue;
        }
        for (s, row) in observed.iter_mut().enumerate() {
            if speaker[f][s] >= threshold {
                row[greedy[f]] += 1;
            }
        }
    }
    let expected: Vec<Vec<u64>> = streams
        .iter()
        .map(|s| {
            let mut v = vec![0u64; cols];
            for &l in s.labels.iter().skip(1).step_by(2) {
                v[l as usize] += 1;
            }
            v
        })
        .collect();
    let idf: Vec<f64> = (0..cols)
        .map(|c| {
            let present = observed.iter().filter(|r| r[c] > 0).count();
            (SPEAKERS as f64 / (1 + present) as f64).ln_1p()
        })
        .collect();
    let a: Vec<Vec<f64>> = expected
        .iter()
        .map(|r| {
            r.iter()
                .zip(&idf)
                .map(|(&e, w)| (e as f64 * w).sqrt())
                .collect()
        })
        .collect();
    let b: Vec<Vec<f64>> = observed
        .iter()
        .map(|r| {
            r.iter()
                .zip(&idf)
                .map(|(&o, w)| (o as f64 * w).sqrt())
                .collect()
        })
        .collect();
    let norm = |v: &Vec<f64>| v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let cost: Vec<Vec<f64>> = (0..streams.len())
        .map(|i| {
            (0..SPEAKERS)
                .map(|j| {
                    let dot: f64 = a[i].iter().zip(&b[j]).map(|(x, y)| x * y).sum();
                    let mut sim = dot / (norm(&a[i]) * norm(&b[j])).max(1e-8);
                    let ea = expected[i].iter().sum::<u64>() as f64 + 1.0;
                    let ob = observed[j].iter().sum::<u64>() as f64 + 1.0;
                    sim -= 0.1 * (ea / ob).ln().abs();
                    -sim
                })
                .collect()
        })
        .collect();
    let mut mapping = vec![None; streams.len()];
    for (r, c) in linear_sum_assignment(&cost) {
        mapping[r] = Some(c);
    }
    mapping
}

/// `np.rint`: round half to even.
fn rint(x: f64) -> f64 {
    x.round_ties_even()
}

/// NumPy's `interp` for increasing `xp` (slopes precomputed, an exact knot
/// returning its value).
fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let n = xp.len();
    if x <= xp[0] {
        return fp[0];
    }
    if x >= xp[n - 1] {
        return fp[n - 1];
    }
    let j = xp.partition_point(|&v| v <= x) - 1;
    if x == xp[j] {
        return fp[j];
    }
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    slope * (x - xp[j]) + fp[j]
}

/// Step 4's band: the first state of each searched frame's band.
fn band_starts(
    greedy: &[usize],
    target: &[u32],
    states: usize,
    half: usize,
    width: usize,
) -> Vec<usize> {
    let frames = greedy.len();
    let keep: Vec<usize> = (0..frames)
        .filter(|&i| greedy[i] != 0 && (i == 0 || greedy[i] != greedy[i - 1]))
        .collect();
    let observed: Vec<usize> = keep.iter().map(|&i| greedy[i]).collect();
    let trigrams = |v: &[usize]| {
        let mut m: std::collections::HashMap<[usize; 3], Vec<usize>> = Default::default();
        for i in 0..v.len().saturating_sub(2) {
            m.entry([v[i], v[i + 1], v[i + 2]]).or_default().push(i);
        }
        m
    };
    let tgt: Vec<usize> = target.iter().map(|&t| t as usize).collect();
    let (left, right) = (trigrams(&observed), trigrams(&tgt));
    let mut pairs: Vec<(usize, usize)> = left
        .iter()
        .filter(|(k, v)| v.len() == 1 && right.get(*k).is_some_and(|r| r.len() == 1))
        .map(|(k, v)| (keep[v[0] + 1], 2 * (right[k][0] + 1) + 1))
        .collect();
    pairs.sort_unstable();
    // longest chain increasing in state (patience sorting, bisect_left)
    let (mut tails, mut tail_ids, mut parent) = (
        Vec::<usize>::new(),
        Vec::<usize>::new(),
        Vec::<isize>::new(),
    );
    for (i, &(_, s)) in pairs.iter().enumerate() {
        let j = tails.partition_point(|&t| t < s);
        parent.push(if j > 0 { tail_ids[j - 1] as isize } else { -1 });
        if j == tails.len() {
            tails.push(s);
            tail_ids.push(i);
        } else {
            tails[j] = s;
            tail_ids[j] = i;
        }
    }
    let mut chosen = Vec::new();
    let mut k = tail_ids.last().map_or(-1, |&t| t as isize);
    while k >= 0 {
        chosen.push(pairs[k as usize]);
        k = parent[k as usize];
    }
    chosen.reverse();
    let mut xs = vec![0f64];
    let mut ys = vec![0f64];
    for &(t, s) in &chosen {
        if t > 0 && t + 1 < frames {
            xs.push(t as f64);
            ys.push(s as f64);
        }
    }
    xs.push((frames - 1) as f64);
    ys.push((states - 1) as f64);
    let hi = (states - width) as i64;
    let mut starts: Vec<i64> = (0..frames)
        .map(|f| (rint(interp(f as f64, &xs, &ys)) as i64 - half as i64).clamp(0, hi))
        .collect();
    // advance at most two states a frame, and reach the end in time
    let mut run = i64::MAX;
    for (f, s) in starts.iter_mut().enumerate() {
        run = run.min(*s - 2 * f as i64);
        *s = run + 2 * f as i64;
    }
    for (f, s) in starts.iter_mut().enumerate() {
        let floor = (states as i64 - width as i64 - 2 * (frames as i64 - 1 - f as i64)).max(0);
        *s = (*s).max(floor);
    }
    starts.into_iter().map(|s| s as usize).collect()
}

/// Step 4: the banded max-sum DP and its backtrace; the state of every
/// searched frame, or None.
fn banded_search(
    logp: &[f32],
    cols: usize,
    labels: &[u32],
    positions: &[usize],
    prior: &[f32],
    greedy: &[usize],
    p: &AlignParams,
) -> Option<Vec<usize>> {
    let (frames, states) = (positions.len(), labels.len());
    let half = if frames < p.band_full_below_frames {
        states
    } else {
        states.min((p.band_ratio * states as f64 + p.band_min_states as f64).ceil() as usize)
    };
    let width = states.min(2 * half + 1);
    let target: Vec<u32> = labels.iter().skip(1).step_by(2).copied().collect();
    let repeats = target.windows(2).filter(|w| w[0] == w[1]).count();
    if frames < target.len() + repeats || frames == 0 {
        return None;
    }
    let g: Vec<usize> = positions.iter().map(|&f| greedy[f]).collect();
    let starts = band_starts(&g, &target, states, half, width);
    let can_skip: Vec<bool> = (0..states)
        .map(|s| labels[s] != 0 && s >= 2 && labels[s] != labels[s - 2])
        .collect();
    let lp = |f: usize, l: u32| logp[f * cols + l as usize];
    let neg = f32::NEG_INFINITY;
    let mut prev = vec![neg; width];
    let first = positions[0];
    for j in 0..width.min(2) {
        prev[j] = lp(first, labels[j]);
        if labels[j] != 0 {
            prev[j] += prior[first];
        }
    }
    let mut cur = vec![neg; width];
    let mut ptr = vec![0u8; frames * width];
    for t in 1..frames {
        let start = starts[t];
        let delta = start - starts[t - 1];
        let pos = positions[t];
        for j in 0..width {
            let k = j + delta;
            let mut best = if k < width { prev[k] } else { neg };
            let mut mv = 0u8;
            let one = if k >= 1 && k - 1 < width {
                prev[k - 1]
            } else {
                neg
            };
            if one > best {
                best = one;
                mv = 1;
            }
            let two = if k >= 2 && k - 2 < width {
                prev[k - 2]
            } else {
                neg
            };
            if two > best && can_skip[start + j] {
                best = two;
                mv = 2;
            }
            ptr[t * width + j] = mv;
            let lab = labels[start + j];
            cur[j] = best + lp(pos, lab);
            if lab != 0 {
                cur[j] += prior[pos];
            }
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    let last = starts[frames - 1];
    let mut best: Option<(usize, f32)> = None;
    for e in [states - 1, states - 2] {
        if e >= last && e - last < width {
            let v = prev[e - last];
            if best.is_none_or(|(_, b)| v > b) {
                best = Some((e, v));
            }
        }
    }
    let (mut state, score) = best?;
    if !score.is_finite() {
        return None;
    }
    let mut path = vec![0usize; frames];
    for t in (0..frames).rev() {
        path[t] = state;
        if t > 0 {
            state -= ptr[t * width + state - starts[t]] as usize;
        }
    }
    Some(path)
}

/// An aligned word.
#[derive(Clone, Debug)]
struct Row {
    word: String,
    tag: Option<u32>,
    start: f64,
    end: f64,
    edge_cols: [u32; 2],
    outside: bool,
}

/// Steps 3-5 and the public object (`ctc_timestamps`).
pub fn align(prep: &Prepared, inp: &Inputs<'_>, p: &AlignParams) -> Result<Value, String> {
    let frames = inp.ctc_frames;
    let cols = prep.columns.len();
    if frames == 0 || inp.logp.len() != frames * cols {
        return Err("Padded sequence lengths must match the batch and time dimension.".into());
    }
    let step = inp.duration / frames as f64;
    let logp = inp.logp;
    let greedy: Vec<usize> = (0..frames)
        .map(|f| {
            let row = &logp[f * cols..(f + 1) * cols];
            let mut best = 0usize;
            for (c, &v) in row.iter().enumerate() {
                if v > row[best] {
                    best = c;
                }
            }
            best
        })
        .collect();
    let speaker = resample(inp.speaker80, frames);
    let mapping = speaker_mapping(
        cols,
        &prep.streams,
        &speaker,
        p.speaker_activity_threshold as f32,
        &greedy,
    );
    let mut rows: Vec<Row> = Vec::new();
    for (stream, column) in prep.streams.iter().zip(&mapping) {
        let activity: Vec<f32> = match column {
            Some(c) => speaker.iter().map(|s| s[*c]).collect(),
            None => vec![0.0; frames],
        };
        let support = p.support_activity_threshold as f32;
        let positions: Vec<usize> = (0..frames).filter(|&f| activity[f] >= support).collect();
        let (w, eps) = (p.speaker_logprob_weight as f32, p.epsilon as f32);
        let prior: Vec<f32> = activity.iter().map(|&a| w * a.max(eps).ln()).collect();
        let Some(path) = banded_search(logp, cols, &stream.labels, &positions, &prior, &greedy, p)
        else {
            continue; // the reference drops a stream without a legal path
        };
        let mut by_word: Vec<Vec<usize>> = vec![Vec::new(); stream.words.len()];
        let mut state_at = vec![usize::MAX; frames];
        for (k, &f) in positions.iter().enumerate() {
            state_at[f] = path[k];
            if let Some(wi) = stream.state_to_word[path[k]] {
                by_word[wi].push(f);
            }
        }
        for (w, fr) in stream.words.iter().zip(&by_word) {
            let (Some(&a), Some(&b)) = (fr.first(), fr.last()) else {
                return Err(format!("CTC path did not visit word {:?}.", w.word));
            };
            let start = a as f64 * step;
            let end = ((b + 1) as f64 * step).min(start + p.maximum_token_len);
            rows.push(Row {
                word: w.word.clone(),
                tag: stream.tag,
                start,
                end,
                edge_cols: [stream.labels[state_at[a]], stream.labels[state_at[b]]],
                outside: false,
            });
        }
    }
    let dia = segments(inp.labels10, inp.duration);
    let col_of = |tag: Option<u32>| {
        prep.streams
            .iter()
            .position(|s| s.tag == tag)
            .and_then(|i| mapping[i])
    };
    if let Some((weights, offs)) = &p.interpolation {
        interpolate(
            &mut rows,
            logp,
            cols,
            frames,
            step,
            inp.duration,
            &dia,
            &col_of,
            weights,
            offs,
            p,
        );
    }
    let refined = refine(rows, &dia, &col_of, step, p);
    let mut words: Vec<(f64, f64, Value)> = refined
        .into_iter()
        .map(|r| {
            let (s, e) = (round3(r.start), round3(r.end));
            let spk = r.tag.map_or_else(|| "None".to_owned(), |t| t.to_string());
            (
                s,
                e,
                json!({ "word": r.word, "start": s, "end": e, "speaker": spk }),
            )
        })
        .collect();
    words.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    let mut map = serde_json::Map::new();
    for (s, c) in prep.streams.iter().zip(&mapping) {
        let k = s.tag.map_or_else(|| "None".to_owned(), |t| t.to_string());
        map.insert(k, c.map_or(Value::Null, |c| json!(c)));
    }
    Ok(json!({
        "words": words.into_iter().map(|w| w.2).collect::<Vec<_>>(),
        "diarization": diarization_json(&dia),
        "speaker_tag_to_diarization_speaker": map,
        "error": Value::Null,
    }))
}

/// The public segments.
pub fn diarization_json(dia: &[Segment]) -> Value {
    Value::Array(
        dia.iter()
            .map(
                |s| json!({ "speaker": s.speaker, "start": round3(s.start), "end": round3(s.end) }),
            )
            .collect(),
    )
}

/// An object without words (an error, or a transcript with none).
pub fn empty(error: Option<&str>, dia: &[Segment]) -> Value {
    json!({
        "words": [],
        "diarization": diarization_json(dia),
        "speaker_tag_to_diarization_speaker": {},
        "error": error,
    })
}

/// Python's `round(x, 3)`: correctly rounded, ties to even on the exact
/// binary value (Rust's formatter rounds the same way).
fn round3(x: f64) -> f64 {
    format!("{x:.3}").parse().unwrap_or(x)
}

/// Step 5a: the calibrated edge correction (`_interpolate_word_boundaries`).
#[allow(clippy::too_many_arguments)]
fn interpolate(
    rows: &mut [Row],
    logp: &[f32],
    cols: usize,
    frames: usize,
    step: f64,
    duration: f64,
    dia: &[Segment],
    col_of: &dyn Fn(Option<u32>) -> Option<usize>,
    weights: &[Vec<f64>; 2],
    offs: &[f64; 2],
    p: &AlignParams,
) {
    if rows.is_empty() {
        return;
    }
    let upper = duration;
    for r in rows.iter_mut() {
        let (st, en): (Vec<f64>, Vec<f64>) = match col_of(r.tag) {
            Some(c) => dia
                .iter()
                .filter(|s| s.speaker == c)
                .map(|s| (s.start, s.end))
                .unzip(),
            None => (Vec::new(), Vec::new()),
        };
        let mut idx: Vec<usize> = (0..st.len()).collect();
        idx.sort_by(|&a, &b| st[a].total_cmp(&st[b]).then(en[a].total_cmp(&en[b])));
        let st: Vec<f64> = idx.iter().map(|&i| st[i]).collect();
        let en: Vec<f64> = idx.iter().map(|&i| en[i]).collect();
        let last = st.partition_point(|&s| s < r.end) as isize - 1;
        r.outside = last < 0 || en[last as usize] <= r.start;
        if r.outside {
            r.end = (r.start + p.outside_word_duration).min(upper);
        }
    }
    let boundaries: Vec<[f64; 2]> = rows.iter().map(|r| [r.start, r.end]).collect();
    let (radius, points) = (p.profile_radius, p.profile_points);
    let taps = 2 * radius + 2;
    let mut corr = vec![[0f64; 2]; rows.len()];
    for (wi, r) in rows.iter().enumerate() {
        for e in 0..2 {
            let center = boundaries[wi][e] / step;
            let base = center.floor();
            let frac = center - base;
            let mut profile = Vec::with_capacity(2 * points);
            for ch in 0..2 {
                let col = if ch == 0 { r.edge_cols[e] as usize } else { 0 };
                let samples: Vec<f64> = (0..taps)
                    .map(|i| {
                        let f = (base as i64 + i as i64 - radius as i64).clamp(0, frames as i64 - 1)
                            as usize;
                        f64::from(
                            logp[f * cols + col]
                                .clamp(p.logprob_floor as f32, p.logprob_ceiling as f32),
                        )
                    })
                    .collect();
                for m in 0..points {
                    let fine = (2 * radius) as f64 * m as f64 / (points - 1) as f64 + frac;
                    let v: f64 = samples
                        .iter()
                        .enumerate()
                        .map(|(i, s)| s * (1.0 - (i as f64 - fine).abs()).max(0.0))
                        .sum();
                    profile.push(v);
                }
            }
            corr[wi][e] = profile
                .iter()
                .zip(&weights[e])
                .map(|(a, b)| a * b)
                .sum::<f64>()
                + offs[e];
        }
    }
    let cap = p.max_correction;
    let grid = p.grid_seconds;
    let mut refined: Vec<[f64; 2]> = boundaries
        .iter()
        .zip(&corr)
        .map(|(b, c)| {
            [0, 1].map(|e| {
                let v = b[e] + c[e].clamp(-cap, cap);
                ((v / grid).round_ties_even() * grid).clamp(0.0, upper)
            })
        })
        .collect();
    let latest = (upper - p.min_word_duration).max(0.0);
    let mut seen: Vec<Option<u32>> = Vec::new();
    for r in rows.iter() {
        if !seen.contains(&r.tag) {
            seen.push(r.tag);
        }
    }
    for tag in seen {
        let idx: Vec<usize> = (0..rows.len()).filter(|&i| rows[i].tag == tag).collect();
        let n = idx.len();
        let spacing = p.min_onset_spacing.min(latest / (n.max(2) - 1) as f64);
        let mut run = f64::NEG_INFINITY;
        for (k, &i) in idx.iter().enumerate() {
            let g = k as f64 * spacing;
            run = run.max(refined[i][0] - g);
            refined[i][0] = (run + g).min(latest - (n - 1) as f64 * spacing + g);
        }
    }
    for (i, r) in refined.iter_mut().enumerate() {
        let b = boundaries[i];
        for e in 0..2 {
            r[e] = r[e].clamp(b[e] - cap, b[e] + cap).clamp(0.0, upper);
        }
        r[0] = r[0].min(latest);
        r[1] = upper.min((r[0] + p.min_word_duration).max(r[1]));
        r[1] = r[1].min((b[1] + cap).clamp(0.0, upper));
        if rows[i].outside {
            let seed = p.outside_word_duration;
            let lo = r[0] + p.min_word_duration.max(seed - cap);
            let hi = r[0] + p.min_word_duration.max(seed + cap);
            r[1] = upper.min(r[1].max(lo).min(hi));
        }
    }
    for (r, v) in rows.iter_mut().zip(refined) {
        r.start = v[0];
        r.end = v[1];
    }
}

/// Step 5b: the refinement against the speaker's activity segments
/// (`_refine_word_timestamps`), per tag in first-appearance order, each
/// tag's words sorted by start.
fn refine(
    rows: Vec<Row>,
    dia: &[Segment],
    col_of: &dyn Fn(Option<u32>) -> Option<usize>,
    step: f64,
    p: &AlignParams,
) -> Vec<Row> {
    let limit = (2 * p.max_word_frames) as f64 * step.min(0.08);
    let mut tags: Vec<Option<u32>> = Vec::new();
    for r in &rows {
        if !tags.contains(&r.tag) {
            tags.push(r.tag);
        }
    }
    let mut out = Vec::with_capacity(rows.len());
    for tag in tags {
        let mut words: Vec<Row> = rows.iter().filter(|r| r.tag == tag).cloned().collect();
        words.sort_by(|a, b| a.start.total_cmp(&b.start));
        let mut iv: Vec<(f64, f64)> = match col_of(tag) {
            Some(c) => dia
                .iter()
                .filter(|s| s.speaker == c && s.end > s.start)
                .map(|s| (s.start, s.end))
                .collect(),
            None => Vec::new(),
        };
        iv.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
        let starts: Vec<f64> = iv.iter().map(|v| v.0).collect();
        let member: Vec<Option<usize>> = words
            .iter()
            .map(|w| {
                let j = starts.partition_point(|&s| s <= w.start) as isize - 1;
                (!w.outside && j >= 0 && w.start < iv[j as usize].1).then_some(j as usize)
            })
            .collect();
        let mut seen = std::collections::HashSet::new();
        for i in 0..words.len() {
            let Some(m) = member[i] else { continue };
            if !seen.insert(m) {
                continue;
            }
            let mut onset = iv[m].0;
            if i > 0 {
                onset = onset.max(words[i].start.min(words[i - 1].start + p.min_onset_spacing));
            }
            let d = words[i].start - onset;
            if 0.0 < d && d <= p.onset_snap_seconds {
                words[i].start = onset;
            }
        }
        for i in 0..words.len() {
            let w = &words[i];
            let end = if w.outside {
                w.end
            } else {
                match member[i] {
                    None => words.get(i + 1).map_or(w.end, |n| n.start),
                    Some(m) if i + 1 < words.len() && member[i + 1] == Some(m) => {
                        words[i + 1].start
                    }
                    Some(m) => iv[m].1,
                }
            };
            let end = if p.refinement_end_blend != 1.0 {
                w.end + p.refinement_end_blend * (end - w.end)
            } else {
                end
            };
            let start = w.start;
            words[i].end = start.max(end.min(start + limit));
        }
        out.extend(words);
    }
    out
}
