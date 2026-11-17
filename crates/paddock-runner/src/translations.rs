//! `POST /v1/audio/translations` - speech in, text in another language out.
//!
//! The same multipart surface as `/v1/audio/transcriptions` (it is the same
//! handler, run with [`Plan`] armed): `file`, `model`, `response_format`
//! (json | text | verbose_json) and `temperature` from OpenAI's spec, plus
//! vLLM's extensions `language` (the speech's language), `to_language` (the
//! text's, default English) and `stream`. OpenAI's own endpoint only ever
//! answers in English; the models here translate both ways through English,
//! which is what `to_language` reaches.
//!
//! Served on the lanes whose model card documents speech translation, with
//! the card's own instruction - the instruction IS the task on these
//! families, so it is never invented:
//!
//!   * Nemotron 3.5 Transcribe: `Please translate the given speech to
//!     <language>.` (reasoning off, greedy, empty system turn - the card's
//!     translation recipe), between English and the 31 languages its support
//!     matrix marks for translation.
//!   * granite-speech: `translate the speech to <language> with proper
//!     punctuation and capitalization.`, to the card's seven languages, to or
//!     from English.
//!   * Gemma 4 12B: the card's AST instruction, which names the speech's
//!     language - so `language` is required - and answers with the
//!     transcript, a newline, `<language>: ` and the translation; only the
//!     part after that marker is the answer.
//!
//! Elsewhere (Qwen3-ASR, the whisper checkpoints served here, the streaming
//! RNN-T model) a translation is refused by name rather than answered with a
//! transcript.

use std::sync::Arc;

use axum::extract::{Multipart, State};
use axum::response::Response;
use serde_json::{Value, json};

use crate::language::display_name;
use crate::routes::AppState;
use crate::serving::{AudioFrontend, ServingModel};

/// Nemotron 3.5 Transcribe's translation languages: English and the rows of
/// the card's support matrix with a mark under "Translation <-> English".
const TRANSCRIBE_TARGETS: [&str; 32] = [
    "ar", "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "he", "hi", "hr", "hu", "it",
    "ja", "ko", "lt", "lv", "mt", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "th", "uk", "zh",
];

/// granite-speech's `<language>` set, as the card's translation prompt
/// names them.
const GRANITE_TARGETS: [(&str, &str); 7] = [
    ("en", "English"),
    ("fr", "French"),
    ("de", "German"),
    ("es", "Spanish"),
    ("ja", "Japanese"),
    ("it", "Italian"),
    ("zh", "Mandarin"),
];

/// What one translation request asks for, checked against the lane.
#[derive(Clone, Debug)]
pub(crate) struct Plan {
    /// the text's language (ISO 639-1)
    pub to: String,
    /// the instruction the model is given
    pub instruction: String,
    /// Gemma 4's answer marker (`\n<language>: `): the translation is what
    /// follows it, the transcript what precedes it
    marker: Option<String>,
}

impl Plan {
    /// The plan for this lane, or why it cannot translate this way.
    pub(crate) fn new(
        frontend: AudioFrontend,
        from: Option<&str>,
        to: Option<&str>,
    ) -> Result<Self, String> {
        let to = to.unwrap_or("en");
        if from == Some(to) {
            return Err(format!(
                "the speech is already in {} - transcribe it (/v1/audio/transcriptions)",
                display_name(to)
            ));
        }
        // through English, both ways: the cards evaluate X -> en and en -> X
        let pivot = |targets: &[&str]| -> Result<(), String> {
            if !targets.contains(&to) {
                return Err(format!(
                    "this model does not translate to {to:?} (to_language, ISO 639-1: {})",
                    targets.join(", ")
                ));
            }
            if let Some(f) = from
                && !targets.contains(&f)
            {
                return Err(format!(
                    "this model does not translate from {f:?} (language, ISO 639-1: {})",
                    targets.join(", ")
                ));
            }
            if to != "en" && from.is_some_and(|f| f != "en") {
                return Err(
                    "this model translates to and from English - one side of the pair must be \
                     `en`"
                        .into(),
                );
            }
            Ok(())
        };
        match frontend {
            AudioFrontend::Transcribe => {
                pivot(&TRANSCRIBE_TARGETS)?;
                Ok(Self {
                    to: to.to_owned(),
                    instruction: format!(
                        "Please translate the given speech to {}.",
                        display_name(to)
                    ),
                    marker: None,
                })
            }
            AudioFrontend::GraniteSpeech | AudioFrontend::GraniteSpeechMetal => {
                let codes: Vec<&str> = GRANITE_TARGETS.iter().map(|(c, _)| *c).collect();
                pivot(&codes)?;
                let name = GRANITE_TARGETS
                    .iter()
                    .find(|(c, _)| *c == to)
                    .map_or(to, |(_, n)| *n);
                Ok(Self {
                    to: to.to_owned(),
                    instruction: format!(
                        "translate the speech to {name} with proper punctuation and capitalization."
                    ),
                    marker: None,
                })
            }
            AudioFrontend::Gemma4Raw => {
                let Some(from) = from else {
                    return Err(
                        "Gemma 4's translation instruction names the language spoken - send \
                         `language` (ISO 639-1)"
                            .into(),
                    );
                };
                // the names the instruction carries; a code we cannot name is
                // not one to put into a prompt
                let named = |c: &str| (display_name(c) != c).then(|| display_name(c).to_owned());
                let (Some(src), Some(dst)) = (named(from), named(to)) else {
                    return Err(format!("unknown language code in {from:?} -> {to:?}"));
                };
                Ok(Self {
                    to: to.to_owned(),
                    instruction: format!(
                        "Transcribe the following speech segment in {src}, then translate it \
                         into {dst}.\nWhen formatting the answer, first output the \
                         transcription in {src}, then one newline, then output the string \
                         '{dst}: ', then the translation in {dst}."
                    ),
                    marker: Some(format!("\n{dst}: ")),
                })
            }
            AudioFrontend::Qwen3Asr | AudioFrontend::Qwen3AsrMetal => Err(
                "Qwen3-ASR transcribes; it does not translate (its card documents no \
                 translation task)"
                    .into(),
            ),
            AudioFrontend::None => Err("this model does not serve audio".into()),
        }
    }

    /// The answer within the model's raw text: all of it, except on a lane
    /// that writes the transcript first (Gemma 4), where nothing shows until
    /// its marker has been written and then only what follows it - so a
    /// stream never sends the transcript as if it were the translation.
    pub(crate) fn visible<'a>(&self, raw: &'a str) -> &'a str {
        match &self.marker {
            None => raw,
            Some(m) => raw.find(m.as_str()).map_or("", |i| &raw[i + m.len()..]),
        }
    }

    /// Whether the model wrote the marker its lane promised (always true
    /// on a lane that has none).
    pub(crate) fn answered(&self, raw: &str) -> bool {
        self.marker
            .as_ref()
            .is_none_or(|m| raw.contains(m.as_str()))
    }
}

/// What `/api/server` and `/v1/models` say about translation on this lane:
/// the languages it reaches, that pairs go through English, and whether the
/// speech's language must be named. None on a lane that does not translate.
pub(crate) fn caps_json(model: &ServingModel) -> Option<Value> {
    if !model.supports_audio {
        return None;
    }
    match model.audio_frontend {
        AudioFrontend::Transcribe => Some(json!({
            "languages": TRANSCRIBE_TARGETS,
            "through_english": true,
            "language_required": false,
        })),
        AudioFrontend::GraniteSpeech | AudioFrontend::GraniteSpeechMetal => Some(json!({
            "languages": GRANITE_TARGETS.iter().map(|(c, _)| *c).collect::<Vec<_>>(),
            "through_english": true,
            "language_required": false,
        })),
        // the card names no language set for speech translation
        AudioFrontend::Gemma4Raw => Some(json!({
            "languages": null,
            "through_english": false,
            "language_required": true,
        })),
        _ => None,
    }
}

pub async fn handle(State(state): State<Arc<AppState>>, mp: Multipart) -> Response {
    crate::transcriptions::serve(state, mp, true).await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan_for(
        frontend: AudioFrontend,
        from: Option<&str>,
        to: Option<&str>,
    ) -> Result<Plan, String> {
        Plan::new(frontend, from, to)
    }

    #[test]
    fn transcribe_translates_through_english_with_the_cards_instruction() {
        let p = plan_for(AudioFrontend::Transcribe, None, None).unwrap();
        assert_eq!(p.to, "en");
        assert_eq!(
            p.instruction,
            "Please translate the given speech to English."
        );
        let p = plan_for(AudioFrontend::Transcribe, Some("en"), Some("de")).unwrap();
        assert_eq!(
            p.instruction,
            "Please translate the given speech to German."
        );
        // a pair without English, a language the matrix does not translate,
        // and a "translation" into the speech's own language are refused
        assert!(plan_for(AudioFrontend::Transcribe, Some("de"), Some("fr")).is_err());
        assert!(plan_for(AudioFrontend::Transcribe, Some("tr"), None).is_err());
        assert!(plan_for(AudioFrontend::Transcribe, None, Some("vi")).is_err());
        assert!(plan_for(AudioFrontend::Transcribe, Some("en"), Some("en")).is_err());
        assert_eq!(TRANSCRIBE_TARGETS.len(), 32);
        assert!(
            TRANSCRIBE_TARGETS
                .iter()
                .all(|c| paddock_models::transcribe::LANGUAGES.contains(c))
        );
    }

    #[test]
    fn granite_names_its_languages_as_its_card_does() {
        let p = plan_for(AudioFrontend::GraniteSpeech, None, Some("zh")).unwrap();
        assert_eq!(
            p.instruction,
            "translate the speech to Mandarin with proper punctuation and capitalization."
        );
        assert!(plan_for(AudioFrontend::GraniteSpeech, None, Some("sv")).is_err());
    }

    #[test]
    fn gemma4_needs_the_spoken_language_and_answers_after_its_marker() {
        assert!(plan_for(AudioFrontend::Gemma4Raw, None, Some("en")).is_err());
        let p = plan_for(AudioFrontend::Gemma4Raw, Some("sv"), None).unwrap();
        assert!(p.instruction.starts_with(
            "Transcribe the following speech segment in Swedish, then translate it into English."
        ));
        let raw = "Hej, hur mår du?\nEnglish: Hi, how are you?";
        assert_eq!(p.visible(raw), "Hi, how are you?");
        assert!(p.answered(raw));
        // still writing the transcript: nothing to show yet
        assert_eq!(p.visible("Hej, hur"), "");
        assert!(!p.answered("Hej, hur"));
    }

    #[test]
    fn transcription_only_lanes_refuse() {
        assert!(plan_for(AudioFrontend::Qwen3Asr, None, None).is_err());
        assert!(plan_for(AudioFrontend::None, None, None).is_err());
    }
}
