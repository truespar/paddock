//! Everything the transcription lanes know about LANGUAGE: naming a code,
//! reading a language back out of finished text, and assembling the one
//! object every response reports it all through.
//!
//! ## Why this exists at all
//!
//! A wrong language is not a labelling mistake on this class of model - it is
//! a silent TRANSLATION. Told to expect the wrong language, a whisper-family
//! model stops transcribing and starts translating, and what comes back is
//! fluent, grammatical and wrong with no ragged edge to notice. Measured:
//! Swedish speech, `language` left to detect, and
//! Qwen3-ASR answered *"Hallo, ich teste und sehe, wie es funktioniert."* -
//! correct German for what was said in Swedish.
//!
//! Under the no-silent-failures principle that has to be caught and said out
//! loud, and the runner is the only party that can: it holds the language the
//! decode was told to expect AND the text that came out, at the same moment.
//! A client cannot do it without a second round trip, and doing it client-side
//! would mean the UI inventing a fact the API never returned.
//!
//! ## What is measured, and what is only reported
//!
//! Two different signals share this module and must not be confused:
//!
//!   * the AUDIO language - the model's own posterior over its language
//!     tokens (whisper), or the language a generative lane names in its
//!     answer. That is the engine's, and it arrives here already decided.
//!   * the TEXT language - what the finished transcript is actually written
//!     in, from character n-grams. That is this module's, and it is the only
//!     independent check available: it needs no second model, no audio and no
//!     GPU, and it works on lanes that have no language detection at all.
//!
//! The check REPORTS. It never rewrites `language`, never re-runs a decode,
//! and never relabels a transcript - a detector that silently corrects is a
//! second silent failure wearing the first one's clothes.

use paddock_engine::gpu_model::whisper::LangProb;
use whatlang::{Detector, Lang};

/// ISO 639-1 -> English language name, the exact map vLLM's transcription
/// surface uses (whisper's `ISO639_1_SUPPORTED_LANGS`); unknown codes pass
/// through verbatim, also matching vLLM.
///
/// WIRE COMPATIBILITY, not presentation. This feeds Qwen3-ASR's forced-language
/// envelope (`language {Lang}<asr_text>`), where the string is part of the
/// PROMPT and the arbiter's exact table is the point - vLLM passes an unknown
/// code through verbatim, so we do too, and a nicer name here would be a
/// divergence rather than an improvement. For anything a person reads, use
/// `display_name`.
pub(crate) fn language_name(code: &str) -> &str {
    NAMED
        .iter()
        .find(|(c, _)| *c == code)
        .map(|(_, n)| *n)
        .unwrap_or(code)
}

/// The same, for a HUMAN - whisper's whole set rather than the arbiter's 57.
///
/// Split from `language_name` because the two answer different questions and
/// only one of them is allowed to be pretty. The prompt table is vLLM's and
/// stays frozen; this one exists because `{"code": "am", "name": "am"}` went
/// out on the wire on the first live run of `paddock_language` -
/// Amharic, Maltese, Punjabi, Malayalam and Yiddish are all languages the
/// loaded checkpoint declares, and a `name` that echoes the code tells a
/// client nothing it did not already have.
///
/// Falls back to the code, which is honest for a checkpoint declaring
/// something outside whisper's set.
pub(crate) fn display_name(code: &str) -> &str {
    if let Some((_, n)) = NAMED.iter().find(|(c, _)| *c == code) {
        return n;
    }
    NAMED_REST
        .iter()
        .find(|(c, _)| *c == code)
        .map(|(_, n)| *n)
        .unwrap_or(code)
}

/// The languages whisper declares that vLLM's transcription table does not
/// name. Together with `NAMED` this covers all 100 of a large-v3 checkpoint's
/// map. Spellings follow whisper's own `LANGUAGES` dict, which is where these
/// codes come from - including `jw` for Javanese and `yue` for Cantonese,
/// neither of which is the ISO 639-1 form.
const NAMED_REST: &[(&str, &str)] = &[
    ("am", "Amharic"),
    ("as", "Assamese"),
    ("ba", "Bashkir"),
    ("bn", "Bengali"),
    ("bo", "Tibetan"),
    ("br", "Breton"),
    ("eu", "Basque"),
    ("fo", "Faroese"),
    ("gu", "Gujarati"),
    ("ha", "Hausa"),
    ("haw", "Hawaiian"),
    ("ht", "Haitian Creole"),
    ("jw", "Javanese"),
    ("ka", "Georgian"),
    ("km", "Khmer"),
    ("la", "Latin"),
    ("lb", "Luxembourgish"),
    ("ln", "Lingala"),
    ("lo", "Lao"),
    ("mg", "Malagasy"),
    ("ml", "Malayalam"),
    ("mn", "Mongolian"),
    ("mt", "Maltese"),
    ("my", "Burmese"),
    ("nn", "Norwegian Nynorsk"),
    ("oc", "Occitan"),
    ("pa", "Punjabi"),
    ("ps", "Pashto"),
    ("sa", "Sanskrit"),
    ("sd", "Sindhi"),
    ("si", "Sinhala"),
    ("sn", "Shona"),
    ("so", "Somali"),
    ("sq", "Albanian"),
    ("su", "Sundanese"),
    ("te", "Telugu"),
    ("tg", "Tajik"),
    ("tk", "Turkmen"),
    ("tt", "Tatar"),
    ("uz", "Uzbek"),
    ("yi", "Yiddish"),
    ("yo", "Yoruba"),
    ("yue", "Cantonese"),
];

const NAMED: &[(&str, &str)] = &[
    ("af", "Afrikaans"),
    ("ar", "Arabic"),
    ("az", "Azerbaijani"),
    ("be", "Belarusian"),
    ("bg", "Bulgarian"),
    ("bs", "Bosnian"),
    ("ca", "Catalan"),
    ("cs", "Czech"),
    ("cy", "Welsh"),
    ("da", "Danish"),
    ("de", "German"),
    ("el", "Greek"),
    ("en", "English"),
    ("es", "Spanish"),
    ("et", "Estonian"),
    ("fa", "Persian"),
    ("fi", "Finnish"),
    ("fr", "French"),
    ("gl", "Galician"),
    ("he", "Hebrew"),
    ("hi", "Hindi"),
    ("hr", "Croatian"),
    ("hu", "Hungarian"),
    ("hy", "Armenian"),
    ("id", "Indonesian"),
    ("is", "Icelandic"),
    ("it", "Italian"),
    ("ja", "Japanese"),
    ("kk", "Kazakh"),
    ("kn", "Kannada"),
    ("ko", "Korean"),
    ("lt", "Lithuanian"),
    ("lv", "Latvian"),
    ("mi", "Maori"),
    ("mk", "Macedonian"),
    ("mr", "Marathi"),
    ("ms", "Malay"),
    ("ne", "Nepali"),
    ("nl", "Dutch"),
    ("no", "Norwegian"),
    ("pl", "Polish"),
    ("pt", "Portuguese"),
    ("ro", "Romanian"),
    ("ru", "Russian"),
    ("sk", "Slovak"),
    ("sl", "Slovenian"),
    ("sr", "Serbian"),
    ("sv", "Swedish"),
    ("sw", "Swahili"),
    ("ta", "Tamil"),
    ("th", "Thai"),
    ("tl", "Tagalog"),
    ("tr", "Turkish"),
    ("uk", "Ukrainian"),
    ("ur", "Urdu"),
    ("vi", "Vietnamese"),
    ("zh", "Chinese"),
];

/// The reverse of `language_name`, and it takes either value space.
///
/// The lanes disagree about what a language is on the wire: whisper reports
/// the ISO code ("sv") while Qwen3-ASR writes an English name into its own
/// answer envelope (`language Swedish<asr_text>...`) - is the standing
/// question of which one the spec `language` field should settle on. Nothing
/// here can compare a language against another without first agreeing what a
/// language is, so `paddock_language` settles it for itself and always reports
/// the code.
///
/// Case-insensitive, because the model's capitalisation is the model's: the
/// same checkpoint has been seen writing both "Swedish" and "swedish".
pub fn to_code(value: &str) -> Option<String> {
    let v = value.trim();
    if v.is_empty() {
        return None;
    }
    let lower = v.to_lowercase();
    // already a code - checked first so "no" is Norwegian rather than a miss
    if CODES.iter().any(|(c, _)| *c == lower) {
        return Some(lower);
    }
    // an English name, from the same tables the report renders with, so the
    // two can never disagree about a pair - both of them, since a model that
    // names its own language can name any of whisper's, not just the 57 the
    // arbiter's prompt table covers
    NAMED
        .iter()
        .chain(NAMED_REST)
        .find(|(_, n)| n.to_lowercase() == lower)
        .map(|(c, _)| (*c).to_owned())
}

/// whatlang's ISO 639-3 enum ↔ whisper's bare codes.
///
/// Not derivable, and the exceptions are the reason: whisper spells Javanese
/// `jw` (ISO 639-1 is `jv`), Farsi `fa` where whatlang says `pes`, Mandarin
/// `zh` where whatlang says `cmn`, and Norwegian `no` where whatlang has only
/// Bokmål. Four of whatlang's 70 have no whisper code at all - Esperanto,
/// Odia, Akan, Zulu - and they are simply absent here, which is what makes
/// them unreportable rather than mapped to something close-ish.
const CODES: &[(&str, Lang)] = &[
    ("af", Lang::Afr),
    ("am", Lang::Amh),
    ("ar", Lang::Ara),
    ("az", Lang::Aze),
    ("be", Lang::Bel),
    ("bg", Lang::Bul),
    ("bn", Lang::Ben),
    ("ca", Lang::Cat),
    ("cs", Lang::Ces),
    ("cy", Lang::Cym),
    ("da", Lang::Dan),
    ("de", Lang::Deu),
    ("el", Lang::Ell),
    ("en", Lang::Eng),
    ("es", Lang::Spa),
    ("et", Lang::Est),
    ("fa", Lang::Pes),
    ("fi", Lang::Fin),
    ("fr", Lang::Fra),
    ("gu", Lang::Guj),
    ("he", Lang::Heb),
    ("hi", Lang::Hin),
    ("hr", Lang::Hrv),
    ("hu", Lang::Hun),
    ("hy", Lang::Hye),
    ("id", Lang::Ind),
    ("it", Lang::Ita),
    ("ja", Lang::Jpn),
    ("jw", Lang::Jav),
    ("ka", Lang::Kat),
    ("km", Lang::Khm),
    ("kn", Lang::Kan),
    ("ko", Lang::Kor),
    ("la", Lang::Lat),
    ("lt", Lang::Lit),
    ("lv", Lang::Lav),
    ("mk", Lang::Mkd),
    ("ml", Lang::Mal),
    ("mr", Lang::Mar),
    ("my", Lang::Mya),
    ("ne", Lang::Nep),
    ("nl", Lang::Nld),
    ("no", Lang::Nob),
    ("pa", Lang::Pan),
    ("pl", Lang::Pol),
    ("pt", Lang::Por),
    ("ro", Lang::Ron),
    ("ru", Lang::Rus),
    ("si", Lang::Sin),
    ("sk", Lang::Slk),
    ("sl", Lang::Slv),
    ("sn", Lang::Sna),
    ("sr", Lang::Srp),
    ("sv", Lang::Swe),
    ("ta", Lang::Tam),
    ("te", Lang::Tel),
    ("th", Lang::Tha),
    ("tk", Lang::Tuk),
    ("tl", Lang::Tgl),
    ("tr", Lang::Tur),
    ("uk", Lang::Ukr),
    ("ur", Lang::Urd),
    ("uz", Lang::Uzb),
    ("vi", Lang::Vie),
    ("yi", Lang::Yid),
    ("zh", Lang::Cmn),
];

fn whisper_code(lang: Lang) -> Option<&'static str> {
    CODES.iter().find(|(_, l)| *l == lang).map(|(c, _)| *c)
}

fn whatlang_lang(code: &str) -> Option<Lang> {
    CODES.iter().find(|(c, _)| *c == code).map(|(_, l)| *l)
}

/// Codes whose difference this check cannot honestly claim to see in written
/// text, so a disagreement inside a group is never reported.
///
/// Each group is a specific, checkable reason rather than a hedge:
///   * `no`/`nn` - Bokmål and Nynorsk are an ORTHOGRAPHIC choice, not a
///     different language, and upstream whisper rejected splitting the token
///     for exactly that reason. whatlang carries only Bokmål.
///   * `sr`/`hr`/`bs` - one language in the trigram sense; whatlang separates
///     Serbian and Croatian mostly by script, which a Latin-script transcript
///     does not provide.
///   * `id`/`ms` - Indonesian and Malay; whatlang has only Indonesian.
///   * `zh`/`yue` - Cantonese written in Chinese characters is the same
///     character stream as Mandarin to any n-gram model.
///
/// Hindi/Urdu are deliberately not a group: same spoken language, different
/// scripts, so the text genuinely does say which one it is.
const EQUIVALENT: &[&[&str]] = &[
    &["no", "nn"],
    &["sr", "hr", "bs"],
    &["id", "ms"],
    &["zh", "yue"],
];

fn same_language(a: &str, b: &str) -> bool {
    a == b || EQUIVALENT.iter().any(|g| g.contains(&a) && g.contains(&b))
}

/// The shortest transcript worth judging, in LETTERS (digits, punctuation and
/// whitespace do not count), plus a word floor.
///
/// Both are gates against a false alarm, which is the failure that matters
/// here: a warning that fires on a correct Swedish transcript teaches people
/// to ignore the warning, and then the real one is invisible too. The numbers
/// bracket the case this was written for - "Hallo, ich teste und sehe, wie es
/// funktioniert." is 39 letters and 8 words - while excluding the one-word
/// answers ("Ja", "Tack", "Okej") that no n-gram model can place.
///
/// PROVISIONAL, like every other threshold in this feature: Stage B measures
/// the accuracy-vs-length curve and these move with it.
const MIN_LETTERS: usize = 24;
const MIN_WORDS: usize = 4;

/// What language a finished transcript is written in.
#[derive(Clone, Debug, PartialEq)]
pub struct TextLanguage {
    /// whisper-style bare code
    pub code: String,
    /// A MARGIN, not a mass: how far the winner beat its nearest rival among
    /// the languages this check weighed. 1.0 means "no contest". Which
    /// languages those were depends on the stage that decided it - see
    /// `identify_text`, where the two-way case is the sharp one.
    pub confidence: f32,
}

/// Is there enough here to judge at all?
fn long_enough(text: &str) -> bool {
    let letters = text.chars().filter(|c| c.is_alphabetic()).count();
    let words = text
        .split_whitespace()
        .filter(|w| w.chars().any(char::is_alphabetic))
        .count();
    letters >= MIN_LETTERS && words >= MIN_WORDS
}

/// Read the language of a finished transcript, or abstain.
///
/// Two STAGES, and the second is what makes this usable at all. Measured,
/// one sentence each:
///
/// | text | 70-way best | 70-way conf | vs the asked language |
/// |---|---|---|---|
/// | the German that started this | Deu | 0.54 | **1.00** |
/// | correct Swedish | Swe | 0.56 | 1.00 vs de, 0.98 vs no |
/// | correct English | Eng | 0.89 | 1.00 vs nl |
/// | correct Dutch | **Afr** | 0.06 | 0.06 - abstains |
///
/// Open 70-way confidence is useless here: the right answers score 0.54 and
/// 0.56, under any threshold that would also reject a wrong one. But the
/// question this check actually asks is not "which of 70 languages is this",
/// it is "is this not the language the decode ran under" - a two-way test,
/// and two-way it separates cleanly. That is the same lever the LID
/// literature reports on audio: restricting a 107-language classifier to
/// three took its error 17.08% -> 1.47% (Valente, Interspeech 2024).
///
/// So: stage one NOMINATES a rival with no confidence gate at all, stage two
/// weighs exactly `{asked, rival}` and only that verdict is trusted.
///
/// **Measured** over a 10x10 matrix - one ordinary sentence in each
/// of sv/da/no/de/en/nl/fr/es/fi/it, each read as if every one of the ten had
/// been asked for, 100 verdicts:
///
///   * **0 wrong**: it never fired on a correct transcript and never agreed
///     with a wrong one. That is the number this design optimises for.
///   * 67 of the 90 real mismatches caught.
///   * 24 abstentions, concentrated exactly where the
///     hard split is - Danish and Norwegian text, whose trigram profiles this
///     library separates poorly (da-vs-no scored 0.11 on one sentence). The
///     remainder are near-pairs on short text: sv/da 0.80, fr/en 0.89.
///
/// The reliability threshold is whatlang's own (margin > 0.9) and stays
/// borrowed deliberately. Four of those misses sit at 0.80-0.89 and lowering the
/// bar would have caught them with no wrong answer on this MATRIX - which is
/// ten sentences, and tuning a threshold on ten sentences is not tuning
/// (nobody has published a calibration curve for this; faster-whisper's 0.5
/// has a mechanical PR rationale and no accuracy study). It moves when
/// something bigger than ten sentences says it should.
///
/// `allow` is the set of codes the LANE could plausibly have produced - a
/// whisper checkpoint's own language map, empty for a lane that publishes
/// none. A model cannot have written a language it was never trained on, so
/// scoring against languages outside its map only invites a confident
/// nonsense answer. It is not the caller's candidate set: restricting to that
/// would guarantee the check agrees with whatever was asked for, which is the
/// exact opposite of its job.
///
/// Abstains - None - on text too short to judge, on a pair too close to call
/// (Danish vs Bokmål on one sentence scores 0.59; Dutch vs Afrikaans 0.06),
/// and on any language this check cannot name. Every one of those is a
/// silence, and silence is right: a warning that fires on a correct
/// transcript teaches people to ignore the warning, and then the real one is
/// invisible too.
pub fn identify_text(text: &str, expected: Option<&str>, allow: &[String]) -> Option<TextLanguage> {
    if !long_enough(text) {
        return None;
    }
    let langs: Vec<Lang> = allow.iter().filter_map(|c| whatlang_lang(c)).collect();
    // An empty allowlist means "the lane did not say", not "allow nothing" -
    // `Detector::with_allowlist(vec![])` would detect nothing at all.
    let nominee = if langs.is_empty() {
        whatlang::detect(text)
    } else {
        Detector::with_allowlist(langs).detect(text)
    }?;
    let rival = whisper_code(nominee.lang())?;
    let conf = |c: f64| c as f32;

    // No language to test against - a lane that names none and a caller who
    // asked for none. There is no pair to weigh, so whatlang's own
    // reliability test is all there is, and it will usually say no.
    let Some(expected) = expected else {
        return nominee.is_reliable().then(|| TextLanguage {
            code: rival.to_owned(),
            confidence: conf(nominee.confidence()),
        });
    };
    // The nominee already agrees. Nothing is being contradicted, so the
    // 70-way margin is the honest number to report: it says how much the
    // agreement is worth, and nobody is being interrupted over it.
    if same_language(expected, rival) {
        return Some(TextLanguage {
            code: expected.to_owned(),
            confidence: conf(nominee.confidence()),
        });
    }
    // The two-way adjudication. A language this check cannot represent
    // (Icelandic, Nynorsk, Cantonese - whatlang has none of them) cannot be
    // one side of a pair, so the question goes unanswered rather than
    // answered against a stand-in.
    let (Some(a), Some(b)) = (whatlang_lang(expected), whatlang_lang(rival)) else {
        return None;
    };
    let verdict = Detector::with_allowlist(vec![a, b]).detect(text)?;
    if !verdict.is_reliable() {
        return None;
    }
    let code = whisper_code(verdict.lang())?;
    Some(TextLanguage {
        code: code.to_owned(),
        confidence: conf(verdict.confidence()),
    })
}

/// Where the reported language came from. Four genuinely different answers,
/// and a client that treats them alike will mislead someone: a forced code is
/// the caller's own instruction reflected back, a detection is a measurement
/// with a number behind it, and a model's self-report is neither.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Source {
    /// the caller forced it; no detection ran
    Asked,
    /// our LID pass chose it, and `candidates` says how confidently
    Detected,
    /// the model named it inside its own answer (Qwen3-ASR's envelope)
    Reported,
    /// nothing on this lane can say (granite-speech, with no hint given)
    Unknown,
}

impl Source {
    fn wire(self) -> &'static str {
        match self {
            Source::Asked => "asked",
            Source::Detected => "detected",
            Source::Reported => "reported",
            Source::Unknown => "unknown",
        }
    }
}

/// How many runners-up ride on the wire.
///
/// Not the whole 99: the tail below the top few is uniformly tiny and would
/// be a page of noise on every transcription. Five covers the finding that
/// motivates publishing any of it - the true language is in whisper's own
/// 10-best 98.6% of the time while its argmax scores 83.9% (arXiv 2409.18428)
/// - with room to see a two-way call for what it is.
const TOP_K: usize = 5;

/// Everything one transcription can say about language, assembled once so the
/// streaming, non-streaming and live-socket answers cannot drift apart.
pub struct LanguageReport {
    pub code: Option<String>,
    pub source: Source,
    /// the posterior after any candidate-set prior, best first; empty on
    /// every lane and every request where no detection ran
    pub candidates: Vec<LangProb>,
    /// what the audio alone preferred, when the caller's hints outranked it
    pub prior_moved: Option<LangProb>,
    pub hints: Vec<String>,
    pub hint_strength: f32,
    /// what the transcript turned out to be written in, where the check could
    /// run at all
    pub written: Option<TextLanguage>,
}

impl LanguageReport {
    /// Build the report for a finished transcription.
    ///
    /// `allow` is the lane's own language map (see `identify_text`).
    pub fn new(
        code: Option<String>,
        source: Source,
        candidates: Vec<LangProb>,
        prior_moved: Option<LangProb>,
        hints: Vec<String>,
        hint_strength: f32,
        text: &str,
        allow: &[String],
    ) -> Self {
        Self {
            // the language the decode ran under is what the text is weighed
            // against, whether that came from the caller or from detection
            written: identify_text(text, code.as_deref(), allow),
            code,
            source,
            candidates,
            prior_moved,
            hints,
            hint_strength,
        }
    }

    /// The probability the reported language carries, when it came from a
    /// measurement. Absent on a forced code and on a model's self-report -
    /// neither has a number, and inventing a 1.0 there would read as
    /// certainty nobody claimed.
    fn probability(&self) -> Option<f32> {
        let code = self.code.as_deref()?;
        if self.source != Source::Detected {
            return None;
        }
        self.candidates.iter().find(|c| c.code == code).map(|c| c.p)
    }

    /// Does the transcript's own language agree with the one the decode ran
    /// under? None where the check abstained or there is nothing to compare.
    pub fn agrees(&self) -> Option<bool> {
        let (code, written) = (self.code.as_deref()?, self.written.as_ref()?);
        Some(same_language(code, &written.code))
    }

    /// The language the transcript is actually in, when that is not the one
    /// the decode was told to expect. This is the contradiction - the thing
    /// worth interrupting a user over.
    pub fn mismatch(&self) -> Option<&TextLanguage> {
        (self.agrees() == Some(false))
            .then_some(self.written.as_ref())
            .flatten()
    }

    /// Absent rather than empty when there is nothing to say: a lane with no
    /// language at all, no hints and no readable text adds no key. Same rule
    /// the guard notices follow - a key that is always there is a key every
    /// client learns to ignore.
    pub fn json(&self) -> Option<serde_json::Value> {
        if self.code.is_none() && self.candidates.is_empty() && self.written.is_none() {
            return None;
        }
        let mut v = serde_json::json!({ "source": self.source.wire() });
        if let Some(c) = &self.code {
            v["code"] = serde_json::json!(c);
            v["name"] = serde_json::json!(display_name(c));
        }
        if let Some(p) = self.probability() {
            v["probability"] = serde_json::json!(p);
        }
        if !self.candidates.is_empty() {
            v["candidates"] = serde_json::json!(
                self.candidates
                    .iter()
                    .take(TOP_K)
                    .map(|c| serde_json::json!({
                        "code": c.code,
                        "name": display_name(&c.code),
                        "probability": c.p,
                    }))
                    .collect::<Vec<_>>()
            );
        }
        if !self.hints.is_empty() {
            let mut h = serde_json::json!({
                "languages": self.hints,
                "strength": self.hint_strength,
            });
            // The hint CHANGED the answer - the one thing about a soft prior
            // that must never be invisible.
            if let Some(m) = &self.prior_moved {
                h["outranked"] = serde_json::json!({
                    "code": m.code,
                    "name": display_name(&m.code),
                    "probability": m.p,
                });
            }
            v["hints"] = h;
        }
        if let Some(w) = &self.written {
            let mut written = serde_json::json!({
                "code": w.code,
                "name": display_name(&w.code),
                "confidence": w.confidence,
            });
            if let Some(a) = self.agrees() {
                written["agrees"] = serde_json::json!(a);
            }
            v["written"] = written;
        }
        Some(v)
    }
}

/// What this runner can do about language, published on `/api/server` and in
/// `/v1/models` capabilities.
///
/// COMPUTED from what is loaded, never declared - the same rule the rest of
/// that listing follows. A client that has to learn "this model cannot be
/// hinted" from a 400, or that offers a language picker with 99 entries when
/// the loaded checkpoint declares 30, has been told nothing useful.
///
/// `None` on a runner that does not transcribe at all, so the key is absent
/// rather than a set of falses nobody can act on.
/// The refusal for a forced `language` a lane was never trained on, or None:
/// Transcribe writes it into its prompt's Language slot, so a code outside
/// the card's set would decode as a guess. Shared by the file endpoint and
/// the realtime session.
pub(crate) fn unserved_language(
    model: &crate::serving::ServingModel,
    code: &str,
) -> Option<String> {
    let set = &paddock_models::transcribe::LANGUAGES;
    (model.audio_frontend == crate::serving::AudioFrontend::Transcribe && !set.contains(&code))
        .then(|| {
            format!(
                "language {code:?} is not one of this model's languages (ISO 639-1: {}); \
                 omit `language` to transcribe the language it hears",
                set.join(", ")
            )
        })
}

pub fn caps_json(
    asr: Option<&crate::serving::AsrModel>,
    serving: Option<&crate::serving::ServingModel>,
) -> Option<serde_json::Value> {
    use crate::serving::AudioFrontend;
    if let Some(a) = asr {
        // whisper: a TRAINED language-detection pass, its full posterior, and
        // a candidate set that biases it
        return Some(serde_json::json!({
            "supported": true,
            "probabilities": true,
            "hints": true,
            "languages": a.languages,
        }));
    }
    let s = serving.filter(|s| s.supports_audio)?;
    // The generative lanes split, and the split is the model's not the family's
    // wrapper: Qwen3-ASR NAMES the language inside its own answer envelope,
    // granite-speech detects the input language and reports nothing at all.
    // Neither exposes a distribution, and neither has anything a prior could
    // weight - `languages` is refused on both rather than accepted and dropped.
    let names_it = matches!(
        s.audio_frontend,
        AudioFrontend::Qwen3Asr | AudioFrontend::Qwen3AsrMetal
    );
    // Nemotron 3.5 Transcribe's card declares its set (40 languages): a
    // forced `language` outside it is refused, so a client offers exactly
    // these. The others carry no language map to read, so we genuinely do
    // not know their set and say so by not listing one - unpublished, not
    // empty-because-none.
    let languages: &[&str] = if s.audio_frontend == AudioFrontend::Transcribe {
        &paddock_models::transcribe::LANGUAGES
    } else {
        &[]
    };
    Some(serde_json::json!({
        "supported": names_it,
        "probabilities": false,
        "hints": false,
        "languages": languages,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The failure this whole module exists for, as it actually happened:
    /// Swedish spoken, German written, and the German is
    /// verbatim what Qwen3-ASR returned.
    ///
    /// It also pins why the two-stage test exists: unadjudicated, this text
    /// scores 0.54 against 70 languages - below any threshold that would also
    /// reject a wrong answer. Weighed against the language actually asked for
    /// it scores 1.0.
    #[test]
    fn the_measured_swedish_to_german_failure_is_caught() {
        let german = "Hallo, ich teste und sehe, wie es funktioniert.";
        let found = identify_text(german, Some("sv"), &[]).expect("German vs Swedish is decidable");
        assert_eq!(found.code, "de");
        assert!(
            found.confidence > 0.9,
            "two-way margin was {}",
            found.confidence
        );
        assert!(!same_language("sv", &found.code));
    }

    /// And the other half, which matters more: a correct transcript must not
    /// fire. A false alarm teaches people to ignore the warning, and then the
    /// real one is invisible too.
    #[test]
    fn a_correct_transcript_does_not_fire() {
        for (asked, text) in [
            (
                "sv",
                "Jag testar transkriberingen och ser hur den fungerar i praktiken.",
            ),
            (
                "en",
                "I am testing the transcription and seeing how it works in practice.",
            ),
            (
                "de",
                "Ich teste die Transkription und sehe, wie sie in der Praxis funktioniert.",
            ),
            (
                "da",
                "Jeg tester transskriptionen og ser hvordan den fungerer i praksis.",
            ),
            // whatlang's 70-way best for this one is AFRIKAANS at 0.06, and
            // Dutch-vs-Afrikaans stays too close to call two-way as well - so
            // the answer here is a silence, not a false alarm
            (
                "nl",
                "Ik test de transcriptie en kijk hoe die in de praktijk werkt.",
            ),
        ] {
            // an abstention is never a mismatch - that is the whole point
            // of returning None rather than a low-confidence guess
            if let Some(found) = identify_text(text, Some(asked), &[]) {
                assert!(
                    same_language(asked, &found.code),
                    "{asked} transcript read as {} ({})",
                    found.code,
                    found.confidence
                );
            }
        }
    }

    /// Short text abstains rather than guessing. This is the regime where
    /// every identifier is weakest and where a wrong answer would be most
    /// annoying - dictation is 2-4 seconds of speech.
    #[test]
    fn text_too_short_to_judge_says_nothing() {
        for t in [
            "Ja",
            "Tack så mycket",
            "Okej.",
            "Hej hur mår du",
            "42 42 42 42 42",
        ] {
            assert_eq!(
                identify_text(t, Some("sv"), &[]),
                None,
                "{t:?} should abstain"
            );
        }
    }

    /// The near-pairs abstain rather than firing, and this is the measured
    /// half: the Nordic languages cluster as
    /// {Danish, Bokmål} / {Swedish}, so da-vs-no is the hard split. One
    /// sentence is not enough to call it and the check says so.
    ///
    /// Both of these are RECALL losses - real mismatches this check does not
    /// report - and they are pinned rather than fixed because the alternative
    /// (dropping the threshold to catch them) buys recall with the one thing
    /// that must not be spent: a warning that can fire on a correct
    /// transcript. If Stage B's measurement moves the threshold, this test is
    /// where that shows up.
    #[test]
    fn a_pair_too_close_to_call_abstains() {
        let danish = "Jeg tester transskriptionen og ser hvordan den fungerer i praksis.";
        // asked as Norwegian, genuinely Danish: 0.59 two-way on one sentence,
        // so nothing is reported rather than interrupting over a coin flip
        assert_eq!(identify_text(danish, Some("no"), &[]), None);
        // Danish text asked as GERMAN also abstains, and that one is this
        // library's own weakness rather than the languages': the pair scores
        // 0.25 because whatlang's German profile matches Danish orthography
        // more than it should. Recorded because it is the shape of miss to
        // expect - anything READ as Nordic is hard to contradict here.
        assert_eq!(identify_text(danish, Some("de"), &[]), None);
        // ...and the reverse direction, which is the one that actually burned
        // us, is decisive: German text asked as a Nordic language fires.
        let german = "Hallo, ich teste und sehe, wie es funktioniert.";
        assert_eq!(
            identify_text(german, Some("no"), &[]).map(|f| f.code),
            Some("de".into())
        );
        assert_eq!(
            identify_text(german, Some("da"), &[]).map(|f| f.code),
            Some("de".into())
        );
    }

    /// A language this check cannot represent cannot be adjudicated. Icelandic
    /// and Nynorsk are whisper codes with no whatlang counterpart, and an
    /// answer against a stand-in would be worse than no answer.
    #[test]
    fn a_language_the_check_cannot_name_is_left_alone() {
        let icelandic = "Ég er að prófa umritunina og sé hvernig hún virkar í reynd.";
        assert_eq!(identify_text(icelandic, Some("is"), &[]), None);
        assert_eq!(whatlang_lang("is"), None);
        assert_eq!(whatlang_lang("nn"), None);
    }

    /// The orthography groups. `nn` and `no` are the same language written
    /// two ways, and reporting a "mismatch" between them would be noise on
    /// every Norwegian transcript.
    #[test]
    fn orthographic_variants_are_not_a_mismatch() {
        assert!(same_language("no", "nn"));
        assert!(same_language("nn", "no"));
        assert!(same_language("sr", "hr"));
        assert!(same_language("bs", "sr"));
        assert!(same_language("zh", "yue"));
        assert!(same_language("id", "ms"));
        // ...and the pairs that are a real finding stay one
        assert!(!same_language("sv", "da"));
        assert!(!same_language("sv", "no"));
        assert!(!same_language("hi", "ur"));
        assert!(!same_language("nl", "de"));
    }

    /// The allowlist is the LANE's map, and it keeps the answer inside what
    /// the checkpoint could have produced. Esperanto is whatlang's favourite
    /// wrong answer and whisper cannot write it at all.
    #[test]
    fn the_allowlist_keeps_the_answer_inside_the_checkpoints_map() {
        let esperanto = "Ĉiuj redaktantoj de Esperanta Vikipedio estas volontuloj kaj ili \
                         partoprenas en la kunlaborema komunumo sen estro.";
        // unrestricted, nothing to adjudicate against: whatlang knows
        // Esperanto, but whisper has no code for it, so there is nothing
        // honest to report
        assert_eq!(identify_text(esperanto, None, &[]), None);
        // restricted to a Nordic lane: whatever it nominates must at least be
        // a language that lane could have written
        let nordic: Vec<String> = ["sv", "da", "no", "en"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let out = identify_text(esperanto, Some("sv"), &nordic);
        assert!(out.is_none() || out.as_ref().is_some_and(|o| nordic.contains(&o.code)));
    }

    /// Every language a loaded checkpoint can declare has a NAME a person can
    /// read. Found live on the first real `paddock_language` body:
    /// the arbiter's prompt table covers 57 of whisper's 100, so Amharic,
    /// Maltese, Punjabi, Malayalam and Yiddish all came back as
    /// `{"code": "am", "name": "am"}` - a name that echoes the code is not a
    /// name.
    #[test]
    fn every_whisper_language_has_a_readable_name() {
        // large-v3's own map, which is what the served checkpoints declare
        const WHISPER: &[&str] = &[
            "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo", "br", "bs", "ca", "cs",
            "cy", "da", "de", "el", "en", "es", "et", "eu", "fa", "fi", "fo", "fr", "gl", "gu",
            "ha", "haw", "he", "hi", "hr", "ht", "hu", "hy", "id", "is", "it", "ja", "jw", "ka",
            "kk", "km", "kn", "ko", "la", "lb", "ln", "lo", "lt", "lv", "mg", "mi", "mk", "ml",
            "mn", "mr", "ms", "mt", "my", "ne", "nl", "nn", "no", "oc", "pa", "pl", "ps", "pt",
            "ro", "ru", "sa", "sd", "si", "sk", "sl", "sn", "so", "sq", "sr", "su", "sv", "sw",
            "ta", "te", "tg", "th", "tk", "tl", "tr", "tt", "uk", "ur", "uz", "vi", "yi", "yo",
            "yue", "zh",
        ];
        for c in WHISPER {
            let n = display_name(c);
            assert_ne!(n, *c, "{c} has no name");
            assert!(
                n.chars().next().is_some_and(char::is_uppercase),
                "{c} -> {n}"
            );
        }
        // the two tables partition, so a rename can never leave two spellings
        for (c, _) in NAMED_REST {
            assert!(!NAMED.iter().any(|(k, _)| k == c), "{c} is in both tables");
        }
        // ...and the PROMPT table is untouched by any of this: it is vLLM's, and
        // an unknown code passes through verbatim there deliberately
        assert_eq!(language_name("am"), "am");
        assert_eq!(language_name("sv"), "Swedish");
        // a model naming any of them round-trips back to a code
        assert_eq!(to_code("Cantonese").as_deref(), Some("yue"));
        assert_eq!(to_code("javanese").as_deref(), Some("jw"));
    }

    /// The code map has to round-trip, or a language identified is a language
    /// reported under the wrong name. whisper's own spellings are the trap:
    /// `jw`, `fa`, `zh`, `no`.
    #[test]
    fn the_code_map_round_trips() {
        for (code, lang) in CODES {
            assert_eq!(whatlang_lang(code), Some(*lang), "{code}");
            assert_eq!(whisper_code(*lang), Some(*code), "{code}");
        }
        assert_eq!(whatlang_lang("jw"), Some(Lang::Jav));
        assert_eq!(whisper_code(Lang::Pes), Some("fa"));
        assert_eq!(whisper_code(Lang::Cmn), Some("zh"));
        assert_eq!(whisper_code(Lang::Nob), Some("no"));
        // and the four whatlang languages whisper cannot name stay unmapped
        for l in [Lang::Epo, Lang::Ori, Lang::Aka, Lang::Zul] {
            assert_eq!(whisper_code(l), None, "{l:?}");
        }
    }

    fn report(code: &str, source: Source, text: &str) -> LanguageReport {
        LanguageReport::new(
            Some(code.to_owned()),
            source,
            Vec::new(),
            None,
            Vec::new(),
            0.0,
            text,
            &[],
        )
    }

    /// The report says what happened and does not correct it: `code` stays
    /// the language the decode ran under even when the text disagrees, and
    /// the disagreement rides beside it.
    #[test]
    fn the_report_never_relabels_the_transcript() {
        let r = report(
            "sv",
            Source::Asked,
            "Hallo, ich teste und sehe, wie es funktioniert.",
        );
        assert_eq!(r.code.as_deref(), Some("sv"));
        assert_eq!(r.agrees(), Some(false));
        assert_eq!(r.mismatch().map(|m| m.code.as_str()), Some("de"));
        let v = r.json().expect("a report with a language is never empty");
        assert_eq!(v["code"], "sv");
        assert_eq!(v["source"], "asked");
        assert_eq!(v["written"]["code"], "de");
        assert_eq!(v["written"]["agrees"], false);
        // a forced language has no probability to report, and a fabricated
        // one would read as certainty nobody claimed
        assert!(v.get("probability").is_none(), "{v}");
    }

    #[test]
    fn an_agreeing_transcript_reports_no_mismatch() {
        let r = report(
            "sv",
            Source::Detected,
            "Jag testar transkriberingen och ser hur den fungerar.",
        );
        assert_eq!(r.agrees(), Some(true));
        assert!(r.mismatch().is_none());
    }

    /// A transcript too short to judge leaves the question open - `agrees`
    /// is None, not `true`. Answering "true" would be claiming a check that
    /// never ran.
    #[test]
    fn an_unjudgeable_transcript_leaves_the_question_open() {
        let r = report("sv", Source::Asked, "Ja.");
        assert_eq!(r.agrees(), None);
        assert!(r.mismatch().is_none());
        assert!(r.json().expect("has a code").get("written").is_none());
    }

    /// The posterior rides top-k with its probability, and the reported
    /// language's own number comes off it rather than being carried twice.
    #[test]
    fn a_detected_language_carries_its_probability_and_runners_up() {
        let probs = |v: &[(&str, f32)]| -> Vec<LangProb> {
            v.iter()
                .enumerate()
                .map(|(i, (c, p))| LangProb {
                    code: (*c).to_owned(),
                    id: i as u32,
                    p: *p,
                })
                .collect()
        };
        let r = LanguageReport::new(
            Some("sv".into()),
            Source::Detected,
            probs(&[
                ("sv", 0.82),
                ("no", 0.09),
                ("da", 0.04),
                ("de", 0.02),
                ("nl", 0.01),
                ("en", 0.005),
                ("fi", 0.004),
            ]),
            None,
            Vec::new(),
            0.0,
            "Jag testar transkriberingen och ser hur den fungerar.",
            &[],
        );
        let v = r.json().unwrap();
        assert!((v["probability"].as_f64().unwrap() - 0.82).abs() < 1e-6);
        assert_eq!(v["candidates"].as_array().unwrap().len(), TOP_K);
        assert_eq!(v["candidates"][1]["code"], "no");
        assert_eq!(v["candidates"][1]["name"], "Norwegian");
    }

    /// A hint that overturned the audio's own answer says so. The prior is
    /// allowed to be strong; it is not allowed to be invisible.
    #[test]
    fn a_hint_that_changed_the_answer_is_reported() {
        let r = LanguageReport::new(
            Some("sv".into()),
            Source::Detected,
            vec![LangProb {
                code: "sv".into(),
                id: 0,
                p: 0.61,
            }],
            Some(LangProb {
                code: "de".into(),
                id: 1,
                p: 0.44,
            }),
            vec!["sv".into(), "en".into()],
            0.5,
            "Jag testar transkriberingen och ser hur den fungerar.",
            &[],
        );
        let v = r.json().unwrap();
        assert_eq!(v["hints"]["languages"][0], "sv");
        assert!((v["hints"]["strength"].as_f64().unwrap() - 0.5).abs() < 1e-6);
        assert_eq!(v["hints"]["outranked"]["code"], "de");
        assert_eq!(v["hints"]["outranked"]["name"], "German");
    }

    /// A lane that can say nothing at all adds no key - the granite-speech
    /// case, where the model detects the input language and never reports it.
    #[test]
    fn a_lane_with_nothing_to_say_adds_no_key() {
        let r = LanguageReport::new(
            None,
            Source::Unknown,
            Vec::new(),
            None,
            Vec::new(),
            0.0,
            "Ja.",
            &[],
        );
        assert!(r.json().is_none());
    }
}
