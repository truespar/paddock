//! OpenAI `/v1/chat/completions` wire types.
//!
//! `messages` and `tools` are kept as `serde_json::Value` and passed straight
//! to the model's own Jinja chat template - that's how we support any model's
//! chat format, not just one hand-coded case. Sampling fields are typed.

use serde::{Deserialize, Serialize};

/// Unknown fields are REJECTED (serde deny_unknown_fields), matching the real
/// API's "Unrecognized request argument" behavior - an unimplemented spec
/// parameter is a loud 400 naming the field, never a silent ignore. Fields
/// beyond the OpenAI spec (top_k, min_p, repeat_penalty, chat_template_kwargs)
/// are deliberate local extensions, tracked in tests/spec/coverage.json.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatCompletionRequest {
    pub model: String,
    /// OpenAI message objects, verbatim (role/content/tool_calls/...).
    pub messages: Vec<serde_json::Value>,
    /// Tool definitions, verbatim ({type:function, function:{...}}).
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
    /// "auto" | "none" | "required" | {type:function,function:{name}} .
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    /// DEPRECATED pre-2023-11 spelling of `tools`: a bare list of
    /// `{name, description, parameters}` with no `type` wrapper. OpenAI still
    /// accepts it and a lot of pinned client code still emits it, so we
    /// translate rather than refuse - see `chat::adopt_legacy_functions`.
    #[serde(default)]
    pub functions: Option<Vec<serde_json::Value>>,
    /// Predicted Outputs. Accepted into the struct only so the refusal can say
    /// why - see `chat::refuse_unserved_options`.
    #[serde(default)]
    pub prediction: Option<serde_json::Value>,
    /// Server-executed web search. Same: parsed to be refused by name.
    #[serde(default)]
    pub web_search_options: Option<serde_json::Value>,
    /// DEPRECATED spelling of `tool_choice`: "none" | "auto" | {"name": "x"}.
    /// Note the forced shape is `{"name":...}`, not tool_choice's
    /// `{"type":"function","function":{"name":...}}`.
    #[serde(default)]
    pub function_call: Option<serde_json::Value>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    /// Deprecated spelling of the completion-token cap (kept for compat;
    /// `max_completion_tokens` is the current name - sending both is a 400).
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// Current name for the completion-token cap.
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    // Sampling fields are Option so the server can tell "omitted" from an
    // explicit value: request wins, else the server's --temp/--top-p/...
    // defaults, else the OpenAI-compat built-ins.
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
    /// llama.cpp-compat extension: penalty window (server default 64).
    #[serde(default)]
    pub repeat_last_n: Option<usize>,
    /// vLLM-compat bench extension: never stop on EOS/stop-token ids -
    /// generation runs to max_tokens (equal-length cross-model benching).
    #[serde(default)]
    pub ignore_eos: bool,
    /// OpenAI presence penalty (-2..2, 0 = off).
    #[serde(default)]
    pub presence_penalty: f32,
    /// OpenAI frequency penalty (-2..2, 0 = off).
    #[serde(default)]
    pub frequency_penalty: f32,
    /// number of choices to generate (1..=8 here).
    #[serde(default = "default_n")]
    pub n: u32,
    /// attach per-token logprobs to each choice.
    #[serde(default)]
    pub logprobs: bool,
    /// number of top alternatives per token (0..=20; needs `logprobs`).
    #[serde(default)]
    pub top_logprobs: Option<u8>,
    /// {"include_usage": true} - usage in a terminal streaming chunk.
    #[serde(default)]
    pub stream_options: Option<serde_json::Value>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stop: Option<crate::completions::Stop>,
    #[serde(default)]
    pub stream: bool,
    /// Extra chat-template context (vLLM-style), e.g. qwen3.5
    /// `{"enable_thinking": true}` or gpt-oss `{"reasoning_effort": "high"}`.
    #[serde(default)]
    pub chat_template_kwargs: Option<serde_json::Value>,
    /// {"type":"text"|"json_object"|"json_schema", "json_schema":{...}} -
    /// enforced by constrained decoding, not by prompting.
    #[serde(default)]
    pub response_format: Option<serde_json::Value>,
    /// {"token_id": bias} with bias in -100..100, added to the logits before
    /// sampling (a real effect, not accepted-and-ignored).
    #[serde(default)]
    pub logit_bias: Option<std::collections::HashMap<String, f32>>,
    /// Reasoning-model effort knob. The spec's vocabulary is seven values
    /// (none, minimal, low, medium, high, xhigh, max) and the served model's
    /// own chat template usually knows fewer, so a level lands on that
    /// template's nearest rung rather than 400ing a current SDK. `none` turns
    /// reasoning off where the model can be turned off, and clamps to the
    /// lowest rung where it cannot. `/v1/models` publishes this model's actual
    /// rungs under `capabilities.reasoning_effort`. An honest 400 on a model
    /// with no reasoning mode at all.
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    /// OpenRouter-shaped reasoning object. Chat completions has no official
    /// OpenAI field for a thinking budget, and OpenRouter's `reasoning:
    /// {"max_tokens": N}` is the de-facto shape agent stacks already speak -
    /// so that one key is accepted here (`effort` stays on its own
    /// `reasoning_effort` field above; any other key is a loud 400). N caps
    /// the reasoning tokens: at the cap the runner forces the model out of
    /// its think block with the dialect's budget-exhaustion recipe.
    #[serde(default)]
    pub reasoning: Option<serde_json::Value>,
    /// GPT-5-era response-length hint (`low` | `medium` | `high`, default
    /// medium). Validated for spec conformance; local models have no
    /// verbosity knob, so it does not change generation.
    #[serde(default)]
    pub verbosity: Option<String>,
    /// Local extension (deepseek2-ocr family only): the OCR request object -
    /// `{mode, grounding, crop, no_repeat_ngram_size, ngram_window}`, every
    /// field optional. Also accepted as `chat_template_kwargs.ocr` (what an
    /// unmodified SDK reaches via extra_body); this top-level form is the
    /// documented one and wins when both are sent. A loud 400 on any other
    /// model.
    #[serde(default)]
    pub ocr: Option<serde_json::Value>,
    /// vLLM-compat (paddleocr family): per-request multimodal processor
    /// kwargs - `{min_pixels, max_pixels}` smart-resize budget. The official
    /// paddleocr client sends it per block class (Spotting raises max to
    /// 1605632) through the SDK's extra_body. A loud 400 for unknown keys.
    #[serde(default)]
    pub mm_processor_kwargs: Option<serde_json::Value>,
    /// vLLM-compat: strip (`true`) or keep (`false`) special tokens in the
    /// generated text. Absent = the dialect's own behavior (unchanged). The
    /// paddleocr client sends `false` for Spotting - the `<|LOC_END|>` /
    /// `<|LOC_SEP|>` separators its parser needs are special tokens - and
    /// `true` for every other task.
    #[serde(default)]
    pub skip_special_tokens: Option<bool>,
    /// Local extension: document metadata (PDF Title/Author/dates) injected
    /// with extracted file content - "full" (the default) or "off".
    #[serde(default)]
    pub file_metadata: Option<String>,
    /// Local extension: cap the pages taken from every multi-page attachment
    /// (PDF pages rendered or extracted). Absent = the server's own limits.
    #[serde(default)]
    pub max_pages: Option<u32>,
    /// Local extension: how PDFs reach the model - "render" (pdfium page
    /// images, needs a vision model) or "text" (sift text extraction).
    /// Absent = auto: render where the model can see, text otherwise.
    #[serde(default)]
    pub pdf_mode: Option<String>,
    /// Local extension: run the forensic preprocessing pass this turn - "on"
    /// or "off". Absent = follow the endpoint's `[forensics] auto` default.
    /// An explicit "on" runs over every image/PDF present even when `auto`
    /// would not; "off" suppresses it regardless. Only effective on an
    /// endpoint whose `[forensics]` is enabled and a vision model (forensics is
    /// VLM-coupled - its findings are injected for the vision tower to examine).
    #[serde(default)]
    pub forensics: Option<String>,
    /// Telemetry / routing hints: accepted with their spec semantics (no
    /// server-side effect to have).
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub safety_identifier: Option<String>,
    #[serde(default)]
    pub prompt_cache_key: Option<String>,
    #[serde(default)]
    pub service_tier: Option<String>,
    /// Accepted only as ["text"] - no audio modality here.
    #[serde(default)]
    pub modalities: Option<Vec<String>>,
    /// Persistence knobs: completions are not stored on this server, so
    /// store:true and non-empty metadata are honest 400s.
    #[serde(default)]
    pub store: Option<bool>,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
}

fn default_n() -> u32 {
    1
}

/// One tool call in an assistant message.
#[derive(Debug, Clone, Serialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str, // "function"
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize)]
pub struct FunctionCall {
    pub name: String,
    /// JSON-encoded arguments string (OpenAI sends this as a string, not object).
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatMessage {
    pub role: &'static str,
    /// Always serialized, `null` when the turn produced no content (a pure
    /// tool call, or reasoning that ran out the token budget). The spec's
    /// `ChatCompletionResponseMessage` lists it required alongside `role` and
    /// `refusal`, so a client indexing `message["content"]` must never see a
    /// missing key - the Python SDK tolerates it, a stricter client would not.
    pub content: Option<String>,
    /// Required by the spec and always `null` here: refusals are a policy
    /// feature of the hosted models, and a local server that invented one
    /// would be lying about why it declined.
    pub refusal: Option<String>,
    /// gpt-oss analysis channel, surfaced like OpenAI's reasoning models.
    /// A de-facto extension (not in the OpenAI schema), so unlike the fields
    /// above it stays omitted when absent rather than emitted as null.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// The DEPRECATED response half of `functions`. A client that asked with
    /// `functions` cannot parse `tool_calls` - it reads `message.function_call`
    /// - so answering a legacy request in the modern shape is a silent failure
    ///   on the client side. Present only for such requests, and then
    ///   `tool_calls` is absent: the two shapes are alternatives, never both.
    #[serde(skip_serializing_if = "Option::is_none")]
    ///   Reuses `FunctionCall` above: the legacy shape is the inner `function`
    ///   object of a modern `ToolCall`, minus the id/type wrapper.
    pub function_call: Option<FunctionCall>,
}

impl ChatMessage {
    /// An assistant turn. `refusal` is fixed at null - see the field.
    pub fn assistant(
        content: Option<String>,
        reasoning_content: Option<String>,
        tool_calls: Vec<ToolCall>,
    ) -> ChatMessage {
        ChatMessage {
            role: "assistant",
            content,
            refusal: None,
            reasoning_content,
            tool_calls,
            function_call: None,
        }
    }

    /// Rewrite a modern assistant turn into the legacy `function_call` shape.
    /// The legacy protocol has no way to express more than one call per turn,
    /// so a parallel-call answer keeps its first call and drops the rest -
    /// which is why `adopt_legacy_functions` also pins parallel_tool_calls to
    /// false, so this branch never actually has to discard anything.
    pub fn into_legacy(mut self) -> ChatMessage {
        if let Some(first) = self.tool_calls.first() {
            self.function_call = Some(FunctionCall {
                name: first.function.name.clone(),
                arguments: first.function.arguments.clone(),
            });
            self.tool_calls.clear();
        }
        self
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    /// Required by the spec even when logprobs were not requested, where it
    /// is `null` - hence no skip_serializing_if.
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str, // "chat.completion"
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: crate::completions::Usage,
    /// Paddock extension, deepseek2-ocr family only: what the server actually
    /// resolved for this OCR request (mode/crop/pages/tiles/pass_through/
    /// ngram control) plus parsed grounding `regions` when armed. Absent on
    /// every other model - hence skipped, never null.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ocr: Option<serde_json::Value>,
    /// Nemotron 3.5 Transcribe's word timestamps and speaker labels, when
    /// the request opted in (`mm_processor_kwargs.capture_ctc_timestamps`,
    /// vLLM + NeMo's contract): `words`, `diarization`,
    /// `speaker_tag_to_diarization_speaker`, `error`. Absent otherwise.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ctc_timestamps: Option<serde_json::Value>,
}
