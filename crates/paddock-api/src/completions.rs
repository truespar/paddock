//! OpenAI `/v1/completions` (legacy text completion) wire types. No chat
//! template needed - raw text in, raw text out - so this is the first served
//! endpoint. Chat Completions and the Responses API build on the same engine.

use serde::{Deserialize, Serialize};

/// `prompt` accepts a string, an array of strings, an array of token ids or
/// an array of token-id arrays, per the OpenAI schema. Strings come first, so
/// `[]` stays the empty text prompt it always was.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Prompt {
    Single(String),
    Many(Vec<String>),
    Tokens(Vec<u32>),
    ManyTokens(Vec<Vec<u32>>),
}

impl Prompt {
    /// The text prompt ("" for a token prompt). The MVP serves a single
    /// prompt; arrays take element 0 (multi-prompt batching lands with the P2
    /// scheduler).
    pub fn first(&self) -> &str {
        match self {
            Prompt::Single(s) => s,
            Prompt::Many(v) => v.first().map(String::as_str).unwrap_or(""),
            Prompt::Tokens(_) | Prompt::ManyTokens(_) => "",
        }
    }

    /// The token ids when the caller sent ids instead of text (element 0 of a
    /// batch). They go to the model as sent: no tokenizing, no BOS.
    pub fn token_ids(&self) -> Option<&[u32]> {
        match self {
            Prompt::Tokens(t) => Some(t),
            Prompt::ManyTokens(v) => Some(v.first().map(Vec::as_slice).unwrap_or(&[])),
            Prompt::Single(_) | Prompt::Many(_) => None,
        }
    }
}

/// `stop` accepts a string or array of strings.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Stop {
    One(String),
    Many(Vec<String>),
}

impl Stop {
    pub fn to_vec(&self) -> Vec<String> {
        match self {
            Stop::One(s) => vec![s.clone()],
            Stop::Many(v) => v.clone(),
        }
    }
}

/// Unknown fields are REJECTED, matching the real API (see chat.rs) - spec
/// parameters we don't implement (best_of, echo, suffix, n) 400 loudly,
/// tracked in tests/spec/coverage.json.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: Prompt,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub repeat_penalty: Option<f32>,
    /// OpenAI presence penalty (-2..2, 0 = off).
    #[serde(default)]
    pub presence_penalty: f32,
    /// OpenAI frequency penalty (-2..2, 0 = off).
    #[serde(default)]
    pub frequency_penalty: f32,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stop: Option<Stop>,
    #[serde(default)]
    pub stream: bool,
    /// {"include_usage": true} - usage in a terminal streaming chunk.
    #[serde(default)]
    pub stream_options: Option<serde_json::Value>,
    /// {"token_id": bias} in -100..100, added to the logits before sampling.
    #[serde(default)]
    pub logit_bias: Option<std::collections::HashMap<String, f32>>,
    /// Legacy logprobs: top-N alternatives per emitted token (the array shape:
    /// tokens / token_logprobs / top_logprobs / text_offset). OpenAI caps the
    /// legacy endpoint at 5; we accept 0..=20 (superset, matches our chat cap).
    #[serde(default)]
    pub logprobs: Option<u8>,
    /// Telemetry hint: accepted with its spec semantics (no effect to have).
    #[serde(default)]
    pub user: Option<String>,
    /// How many completions to return (1..=8 here, as on chat).
    #[serde(default)]
    pub n: Option<usize>,
    /// Generate `best_of` candidates server-side and return the `n` with the
    /// highest log probability per TOKEN. Must be >= `n`, and OpenAI refuses
    /// it with `stream: true` (it cannot rank what it has already sent).
    #[serde(default)]
    pub best_of: Option<usize>,
    /// Prepend the prompt to the returned text. The legacy perplexity idiom:
    /// `echo` + `logprobs` + `max_tokens: 0` scores an existing string.
    #[serde(default)]
    pub echo: bool,
}

fn default_max_tokens() -> usize {
    128
}
#[derive(Debug, Clone, Serialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: u32,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    /// {"cached_tokens": N} - prompt tokens served from the prefix cache.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<serde_json::Value>,
}

impl Usage {
    /// The prompt-token details object: what the prefix cache served, and what
    /// this request had to prefill itself.
    ///
    /// `cache_write_tokens` is the counterpart of `cached_tokens` - the tokens
    /// that went into the cache rather than came out of it. OpenAI made it a
    /// required field of `/v1/responses`'s `input_tokens_details` (SDK 2.53.0;
    /// it stays optional on chat completions, but the accounting should read
    /// the same on every surface, so all three emit it).
    ///
    /// Honest caveat: checkpointing into the radix cache is page-granular, so
    /// the tokens actually retained can be up to one page short of this. What
    /// the number states exactly is "the part of this prompt that was not a
    /// reuse", which is the question a caller is asking.
    pub fn cached_details(prompt_tokens: usize, cached: usize) -> Option<serde_json::Value> {
        Some(serde_json::json!({
            "cached_tokens": cached,
            "cache_write_tokens": prompt_tokens.saturating_sub(cached),
        }))
    }

    /// Details with the MEDIA-row count alongside the cache numbers.
    ///
    /// `audio_tokens` is OpenAI's own field; `image_tokens` is a paddock
    /// extension (OpenAI's shape carries `cached_tokens`,
    /// `cache_write_tokens` and `audio_tokens` only). Both are omitted
    /// entirely for a text-only request, so the wire shape is unchanged
    /// unless media was actually sent - and the count goes under the field
    /// that names what was sent, since a clip reported as `image_tokens`
    /// is a number a client cannot act on.
    ///
    /// It is the only number that tells a caller what its `detail` choice (or
    /// its clip length) really cost: one `<image>` or `<|audio|>` placeholder
    /// is one prompt token to the tokenizer and hundreds of rows to prefill.
    pub fn media_details(
        prompt_tokens: usize,
        cached: usize,
        media_tokens: usize,
        audio_tokens: usize,
    ) -> Option<serde_json::Value> {
        let mut v = Self::cached_details(prompt_tokens, cached)?;
        // `audio_tokens` of the media rows are a clip's (all of them on the
        // audio-only families, which pass usize::MAX), the rest a picture's -
        // Gemma 4 12B takes both in one request
        let audio = audio_tokens.min(media_tokens);
        if audio > 0 {
            v["audio_tokens"] = serde_json::json!(audio);
        }
        if media_tokens > audio {
            v["image_tokens"] = serde_json::json!(media_tokens - audio);
        }
        Some(v)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

impl CompletionResponse {
    /// The streaming-chunk object shares this struct with `object` fixed to
    /// "text_completion" and per-chunk `text` deltas; the non-stream form
    /// carries the full text + usage.
    pub fn object_name() -> &'static str {
        "text_completion"
    }
}

#[cfg(test)]
mod tests {
    use super::Prompt;

    fn parse(v: &str) -> Prompt {
        serde_json::from_str(v).unwrap()
    }

    /// Every shape OpenAI documents for `prompt`, and `[]` still the empty
    /// text prompt.
    #[test]
    fn prompt_takes_text_and_token_ids() {
        assert_eq!(parse(r#""hi""#).first(), "hi");
        assert_eq!(parse(r#"["a", "b"]"#).first(), "a");
        assert_eq!(parse(r#"[]"#).token_ids(), None);
        let t = parse("[1, 2, 248046]");
        assert_eq!(t.token_ids(), Some(&[1, 2, 248046][..]));
        assert_eq!(t.first(), "");
        assert_eq!(parse("[[5, 6], [7]]").token_ids(), Some(&[5, 6][..]));
        assert!(serde_json::from_str::<Prompt>("[-1]").is_err());
    }
}
