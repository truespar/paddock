//! `POST /v1/chat/completions` - renders the model's chat template, generates,
//! parses the model family's dialect (Harmony, Qwen XML) into content /
//! reasoning / tool calls. Streaming emits `reasoning_content` deltas (the
//! DeepSeek/vLLM convention) alongside `content` deltas; each tool call
//! streams ATOMICALLY as its block completes (fragmenting the arguments of a
//! non-JSON dialect is not prefix-stable, so no partial-argument deltas).
//! Supports n<=8 merged-SSE choices and logprobs over all sampled tokens.
//! Streams always end with a terminal usage chunk (server-exact counts -
//! benchmark clients undercount from visible text otherwise); the
//! stream_options field is accepted and validated for compatibility.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use async_stream::stream;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Json, Response};
use paddock_api::ErrorBody;
use paddock_api::chat::{
    ChatChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, FunctionCall, ToolCall,
};
use paddock_api::completions::Usage;
use paddock_engine::sampler::SamplingParams;
use paddock_engine::service::{
    EngineError, ErrorClass, FinishReason, GenRequest, MmChunk, TokenEvent, TokenLogprobs,
};
use tokio::sync::mpsc::{UnboundedReceiver, unbounded_channel};

use crate::completions::apply_stop_strings;
use crate::parsers::{Dialect, Parsed, ToolHints, holdback, parse, tool_hints};
use paddock_engine::sampler::TokenConstraint;

use crate::chat_template;
use crate::constrained::{
    CompiledSchema, DispatchMachine, Gate, GatedConstraint, JsonMachine, Machine, ToolMachine,
    ToolSet,
};
use crate::routes::AppState;
use crate::serving::ServingModel;

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn err(status: StatusCode, kind: &str, msg: impl Into<String>) -> Response {
    (status, Json(ErrorBody::new(kind, msg))).into_response()
}

/// Map a classified engine error to the OpenAI error envelope. The engine
/// decides whose fault a failure is (`ErrorClass`); this picks the status,
/// `error.type`, and `error.code`. Shared by every OpenAI-shape surface
/// (chat, completions, responses).
pub(crate) fn engine_err(e: &EngineError) -> Response {
    let (status, kind) = match e.class {
        ErrorClass::InvalidRequest => (StatusCode::BAD_REQUEST, "invalid_request_error"),
        ErrorClass::Overloaded => (StatusCode::SERVICE_UNAVAILABLE, "server_error"),
        ErrorClass::Internal => (StatusCode::INTERNAL_SERVER_ERROR, "internal_error"),
    };
    let mut body = ErrorBody::new(kind, &e.message);
    if let Some(code) = e.code {
        body = body.with_code(code);
    }
    (status, Json(body)).into_response()
}

/// Rows the engine actually admits for a prompt: the pad-free text stream
/// plus every image's vision rows - a pad token in `prompt_ids` is a
/// position marker, not a row. Returns `(total, image_rows, images)`.
///
/// Gating on the token stream alone let a 129-page fax TIFF through the
/// edge check and 20+ seconds later it died inside the engine ("batch size
/// 82275 exceeds max_batch 32768") as a mid-stream failure, after every
/// page had been decoded and encoded. The images are already
/// fitted to their `detail` by `decode_image_url`, so `tokens_for` on their
/// final dims is the row count prefill will see.
pub(crate) fn admitted_rows(
    model: &ServingModel,
    text_len: usize,
    mm_chunks: Option<&[MmChunk]>,
) -> (usize, usize, usize) {
    let Some(chunks) = mm_chunks else {
        return (text_len, 0, 0);
    };
    let budget = model.engine.vision_budget();
    let (mut rows, mut n) = (0usize, 0usize);
    for c in chunks {
        match c {
            MmChunk::Image { w, h, .. } => {
                n += 1;
                rows += budget
                    .as_ref()
                    .map_or(0, |b| b.tokens_for(*w as u32, *h as u32) as usize);
            }
            // audio rows are exact from the sample count (the same formula
            // the engine's row expansion uses); `n` counts only IMAGES so
            // the image-flavored overflow refusal never fires for audio -
            // an audio overflow takes the plain context message
            MmChunk::Audio { samples, .. } => {
                rows += model.audio_frontend.prompt_rows(samples.len());
            }
            // directives add no rows (a crop or pixel budget can only SHRINK
            // the per-image estimate above - admission stays conservative)
            MmChunk::Text(_) | MmChunk::OcrCrop(_) | MmChunk::VisionPixels { .. } => {}
        }
    }
    (text_len + rows, rows, n)
}

/// The edge context check every surface runs before submitting: prices what
/// prefill will actually see (text + image rows), and names the pages/detail
/// levers in the refusal when images carry the overflow.
pub(crate) fn context_gate(
    model: &ServingModel,
    text_len: usize,
    mm_chunks: Option<&[MmChunk]>,
    max_ctx: usize,
) -> Option<EngineError> {
    if max_ctx == 0 {
        return None;
    }
    let (total, image_rows, images) = admitted_rows(model, text_len, mm_chunks);
    if total <= max_ctx {
        return None;
    }
    Some(if image_rows > 0 {
        EngineError::context_overflow_images(total, max_ctx, image_rows, images)
    } else {
        EngineError::context_overflow(total, max_ctx)
    })
}

struct Prepared {
    /// The rendered prompt as tokenized, one placeholder per image. This is the
    /// length a request is ADMITTED against and the pre-prefill token count.
    prompt_ids: Vec<u32>,
    /// What the engine is given: `prompt_ids` minus the placeholders (see
    /// [`MmPrompt`]). Equal to `prompt_ids` when there are no images.
    engine_prompt: Vec<u32>,
    max_tokens: usize,
    sampler: SamplingParams,
    stop_tokens: Vec<u32>,
    stop_strings: Vec<String>,
    /// the rendered prompt ended inside an open `<think>` block (qwen
    /// thinking mode) - generated text is reasoning until `</think>`
    thinking_open: bool,
    /// request `parallel_tool_calls: false` - keep only the first parsed call
    single_tool_call: bool,
    /// None = no tools declared -> tool extraction disabled (see `tool_hints`)
    hints: Option<ToolHints>,
    /// interleaved text/image chunks when the request carries an image
    mm_chunks: Option<Vec<MmChunk>>,
    /// grammar to enforce (instantiated per choice - machines are stateful)
    constraint_spec: ConstraintSpec,
    gate: GateSpec,
    /// resolved thinking budget (reasoning.max_tokens) - instantiated per
    /// choice as the BudgetGated wrapper around the grammar, if any
    think_budget: Option<ThinkBudget>,
    /// number of choices (1..=8)
    n: usize,
    /// Some(top_k) when the request wants logprobs
    logprobs: Option<u8>,
    /// deepseek2-ocr: what the server resolved (echoed on the response;
    /// `ngram` already folded into `sampler`, `force_base` into `mm_chunks`)
    ocr: Option<crate::deepseek_ocr::OcrResolved>,
    /// request `skip_special_tokens` (vLLM-compat): None = the dialect's own
    /// decode behavior, Some = the caller decides (paddleocr spotting needs
    /// `false` so its `<|LOC_END|>`/`<|LOC_SEP|>` separators survive)
    skip_special: Option<bool>,
    /// the request arrived in the DEPRECATED `functions` spelling, so its
    /// answer has to go back in the matching `function_call` shape - a client
    /// that asked that way cannot read `tool_calls`. Set by
    /// `adopt_legacy_functions`, which leaves `req.functions` in place purely
    /// as this marker after translating it onto `tools`.
    legacy_functions: bool,
}

/// The compiled-but-uninstantiated constraint: shared immutable parts only,
/// so each of the `n` choices gets its own fresh machine. Shared with
/// /v1/responses (same grammar machinery, single choice).
pub(crate) enum ConstraintSpec {
    None,
    Json(std::sync::Arc<CompiledSchema>),
    Tool(std::sync::Arc<ToolSet>),
    /// `tool_choice: "auto"` with tools declared: the same tool grammar, armed
    /// as a re-armable dispatch so a call the model chooses to make cannot come
    /// out malformed. The bool is "one call only" (`parallel_tool_calls:
    /// false`), enforced by disarming rather than by dropping extra calls after
    /// the fact.
    Dispatch(std::sync::Arc<ToolSet>, bool),
}

#[derive(Clone, Copy)]
pub(crate) enum GateSpec {
    Immediate,
    AfterToken(u32),
    HarmonyFinal {
        channel: u32,
        message: u32,
    },
    MuseContent {
        start: u32,
        message: u32,
        preopened: bool,
    },
}

/// A resolved thinking budget: the request's cap on reasoning tokens plus
/// the dialect's forced-exit recipe, ready to instantiate per choice as a
/// [`crate::constrained::BudgetGated`] wrapper. Built once in `prepare` (the
/// tokenizer work happens there); the ids are cloned per choice like the
/// grammar machines are.
pub(crate) struct ThinkBudget {
    pub budget: usize,
    /// the dialect's think-close id - sampled naturally, it disarms the budget
    pub disarm: u32,
    /// exit-phrase ids ending at the close id
    pub exit_ids: Vec<u32>,
    /// tool-call open/close markers where the dialect writes calls inside its
    /// think block (qwen, laguna) - the injection defers past an open call
    pub call_markers: Option<(Vec<u8>, Vec<u8>)>,
}

/// The injected budget-exhaustion phrase - the Qwen3 technical report's own
/// recipe (arXiv:2505.09388 §"thinking budget"): the model is steered into
/// answering from what it has, instead of a bare close-tag slammed mid-word.
/// The leading newlines detach it from whatever the budget landed inside.
const THINK_EXIT_PHRASE: &str = "\n\nConsidering the limited time by the user, \
     I have to give the solution based on the thinking directly now.\n";

/// Resolve a requested thinking budget against this model. `knob` is the
/// surface's own spelling (`thinking.budget_tokens`, `reasoning.max_tokens`)
/// so the refusal quotes what the caller actually sent.
pub(crate) fn think_budget(
    model: &ServingModel,
    budget: usize,
    thinking_open: bool,
    knob: &str,
) -> Result<ThinkBudget, String> {
    use crate::parsers::Dialect;
    let close = match model.dialect {
        Dialect::QwenXml | Dialect::Laguna => "</think>",
        Dialect::GemmaChannel => crate::parsers::G_CLOSE,
        // Harmony and muse reason in channel structures whose exit is a
        // multi-token header interleaved with stop semantics - an injected
        // close needs its own recipe there, not this one. Honest refusal
        // beats a corrupted channel stream.
        _ => {
            return Err(format!(
                "{knob} is not supported on this model family yet (its reasoning is \
                 channel-structured, not a <think> block)"
            ));
        }
    };
    if !thinking_open {
        return Err(format!(
            "{knob} was sent but thinking is not enabled on this request \
             (the rendered prompt does not open a think block)"
        ));
    }
    let disarm = model
        .tokenizer
        .token_to_id(close)
        .ok_or_else(|| format!("this model's tokenizer has no single {close} token"))?;
    let mut exit_ids = model
        .tokenizer
        .encode(THINK_EXIT_PHRASE)
        .map_err(|e| e.to_string())?;
    exit_ids.push(disarm);
    let call_markers = match model.dialect {
        Dialect::QwenXml | Dialect::Laguna => {
            Some((b"<tool_call>".to_vec(), b"</tool_call>".to_vec()))
        }
        _ => None,
    };
    Ok(ThinkBudget {
        budget,
        disarm,
        exit_ids,
        call_markers,
    })
}

pub(crate) fn instantiate_constraint(
    spec: &ConstraintSpec,
    gate: GateSpec,
    model: &ServingModel,
    budget: Option<&ThinkBudget>,
) -> Option<Box<dyn TokenConstraint>> {
    let machine = match spec {
        ConstraintSpec::None => {
            // No grammar. A budget still needs the wrapper - with nothing
            // inside, it is a pure reasoning-token governor.
            return budget.map(|b| {
                Box::new(crate::constrained::BudgetGated::new(
                    model.vocab_bytes(),
                    None,
                    b.budget,
                    b.disarm,
                    b.exit_ids.clone(),
                    b.call_markers.clone(),
                )) as Box<dyn TokenConstraint>
            });
        }
        ConstraintSpec::Json(s) => Machine::Json(JsonMachine::new(s.clone())),
        // `forced`, not `new`: a forced call owns the region from the first
        // sampled token, which on muse includes the message header (and the
        // thought that precedes it) rather than starting at the call's tag
        ConstraintSpec::Tool(t) => Machine::Tool(ToolMachine::forced(t.clone())),
        ConstraintSpec::Dispatch(t, single) => {
            Machine::Dispatch(DispatchMachine::new(t.clone(), *single))
        }
    };
    let gate = match gate {
        GateSpec::Immediate => Gate::Immediate,
        GateSpec::AfterToken(t) => Gate::AfterToken(t),
        GateSpec::HarmonyFinal { channel, message } => Gate::HarmonyFinal {
            channel,
            message,
            collecting: false,
            header: Vec::new(),
        },
        // starts COLLECTING, unlike Harmony's: the generation prompt already
        // wrote `<|start|>assistant`, so the model's very first token is part
        // of the header this gate has to read. A pre-opened prompt
        // (muse::PREOPEN) already spelled the whole ` to=self<|message|>`
        // header, so the first token is reasoning BODY - collecting it would
        // buffer the entire thought as header bytes.
        GateSpec::MuseContent {
            start,
            message,
            preopened,
        } => Gate::MuseContent {
            start,
            message,
            collecting: !preopened,
            header: Vec::new(),
        },
    };
    // ids the grammar itself spells (muse's `<|start|>`/`<|message|>`/`<|eom|>`);
    // every other family leaves this empty and keeps the blanket refusal
    let preserved = model
        .dialect
        .grammar_specials()
        .iter()
        .filter_map(|t| model.tokenizer.token_to_id(t))
        .collect();
    let inner = GatedConstraint::new(model.vocab_bytes(), gate, machine, preserved);
    Some(match budget {
        None => Box::new(inner),
        Some(b) => Box::new(crate::constrained::BudgetGated::new(
            model.vocab_bytes(),
            Some(inner),
            b.budget,
            b.disarm,
            b.exit_ids.clone(),
            b.call_markers.clone(),
        )),
    })
}

/// The 400 for a forced `tool_choice` on a family we have no tool-call
/// grammar for. `knob` is the surface's own spelling of the setting, so the
/// message quotes back what the caller actually sent.
///
/// The reason differs per family and saying which one matters: gemma4 is not
/// "not done yet", it is structurally blocked until its tool channels parse,
/// and a Plain-dialect model has no tool syntax to force at all.
pub(crate) fn no_forced_tool_grammar(dialect: Dialect, knob: &str) -> String {
    let why = match dialect {
        Dialect::Harmony => "the gpt-oss Harmony channel grammar is a roadmap item",
        Dialect::GemmaChannel => {
            "gemma4 tool-call channels are not parsed yet, so a forced call could not \
             come back as tool_calls"
        }
        _ => "this model family has no tool-call syntax",
    };
    format!("tool_choice {knob} cannot be enforced on this model ({why}); use \"auto\" or \"none\"")
}

/// `tool_choice: "auto"` with tools declared - arm the dialect's tool grammar
/// as a dispatch, so a call the model decides to make is spelled correctly by
/// construction instead of repaired afterwards.
///
/// Every exit here is silent and unconstrained. That asymmetry with the forced
/// path is deliberate: `auto` is what every agent sends by default, so failing
/// a chat because one MCP tool has an unspellable name - or because this family
/// has no tool grammar yet - would trade a real request for a theoretical one.
/// The worst case is exactly today's behaviour, best-effort parsing.
/// Kill switch: `PADDOCK_NO_TOOL_DISPATCH=1` decodes auto tool calls free, the
/// way the server did before. It exists so the grammar's effect can
/// be MEASURED (same model, same seeds, one flag) rather than asserted, and as
/// the escape hatch if a family's real output ever disagrees with its grammar.
/// Read once - this sits in `prepare`, on the admission path.
fn no_tool_dispatch() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_NO_TOOL_DISPATCH").is_ok_and(|v| v != "0")
    })
}

pub(crate) fn auto_tool_dispatch(
    model: &ServingModel,
    tools: Option<&[serde_json::Value]>,
    single: bool,
) -> ConstraintSpec {
    if no_tool_dispatch() {
        return ConstraintSpec::None;
    }
    let Some(tools) = tools.filter(|t| !t.is_empty()) else {
        return ConstraintSpec::None;
    };
    let Some(syntax) = model.dialect.tool_syntax() else {
        return ConstraintSpec::None; // harmony / gemma4: no grammar yet
    };
    match ToolSet::compile(syntax, tools, None) {
        Ok(set) => ConstraintSpec::Dispatch(set, single),
        Err(why) => {
            tracing::debug!(
                why,
                "tool grammar not compiled - auto tool calls decode free"
            );
            ConstraintSpec::None
        }
    }
}

/// The dialect's content trigger for a constrained request: reasoning stays
/// free; the grammar takes over at the content boundary.
/// Where the grammar takes over. `forced_tool`: this is a `tool_choice:
/// "required"` / named-function machine, which OWNS the whole turn rather than
/// waiting for a content boundary - the distinction only matters on muse,
/// whose forced machine spells the message header (and the thought before it)
/// itself, so gating it would leave the grammar pointed at the wrong place.
pub(crate) fn content_gate(
    model: &ServingModel,
    thinking_open: bool,
    forced_tool: bool,
) -> Result<GateSpec, String> {
    if model.dialect == Dialect::MuseChannel {
        if forced_tool {
            return Ok(GateSpec::Immediate);
        }
        // reasoning is whatever the model addresses `to=self`; the grammar
        // wakes at the first message that goes anywhere else. `thinking_open`
        // here means the render pre-opened ` to=self<|message|>` (muse::PREOPEN)
        return Ok(GateSpec::MuseContent {
            start: model
                .tokenizer
                .token_to_id(crate::muse::START)
                .ok_or("muse-glimmer model has no <|start|> token")?,
            message: model
                .tokenizer
                .token_to_id(crate::muse::MESSAGE)
                .ok_or("muse-glimmer model has no <|message|> token")?,
            preopened: thinking_open,
        });
    }
    Ok(match (model.dialect, thinking_open) {
        // laguna gates identically: reasoning runs free, the grammar takes
        // over once the model closes the think block
        (Dialect::QwenXml | Dialect::Laguna, true) => GateSpec::AfterToken(
            model
                .tokenizer
                .token_to_id("</think>")
                .ok_or("thinking model has no </think> token")?,
        ),
        // gemma4 thinking: reasoning runs free until the thought channel
        // closes; the grammar takes over at <channel|> (a single special)
        (Dialect::GemmaChannel, true) => GateSpec::AfterToken(
            model
                .tokenizer
                .token_to_id("<channel|>")
                .ok_or("gemma4 model has no <channel|> token")?,
        ),
        (Dialect::Harmony, _) => GateSpec::HarmonyFinal {
            channel: model
                .tokenizer
                .token_to_id("<|channel|>")
                .ok_or("gpt-oss model has no <|channel|> token")?,
            message: model
                .tokenizer
                .token_to_id("<|message|>")
                .ok_or("gpt-oss model has no <|message|> token")?,
        },
        _ => GateSpec::Immediate,
    })
}

impl Prepared {
    fn make_constraint(&self, model: &ServingModel) -> Option<Box<dyn TokenConstraint>> {
        instantiate_constraint(
            &self.constraint_spec,
            self.gate,
            model,
            self.think_budget.as_ref(),
        )
    }
}

/// A decoded request image (interleaved RGB8).
#[derive(Debug)]
pub(crate) struct RequestImage {
    rgb: Vec<u8>,
    w: usize,
    h: usize,
}

impl RequestImage {
    /// The engine-side chunk this image becomes - the one constructor every
    /// surface uses, so the fields stay private to this module.
    pub(crate) fn into_chunk(self) -> MmChunk {
        MmChunk::Image {
            rgb: self.rgb,
            w: self.w,
            h: self.h,
        }
    }

    /// Decoded dimensions - what the OCR resolution plans its geometry from.
    pub(crate) fn size(&self) -> (usize, usize) {
        (self.w, self.h)
    }
}

/// OpenAI's `detail` knob on an image content part - how much of the model's
/// resolution this image is allowed to spend.
///
/// This is the spec's own field, not an invention of ours: chat completions
/// puts it at `image_url.detail`, the Responses API at `input_image.detail`.
/// The Anthropic Messages API has no equivalent, so its images are always
/// `Auto` - adding a paddock-specific field there would break the drop-in bar
/// for the sake of a knob its SDKs would never send.
///
/// The levels resolve against the ENDPOINT's budget, so `high` means "as much
/// as this tower published", not a fixed pixel count. On gemma4, whose token
/// cost per image is fixed at 280 whatever you send it, all three levels land
/// in the same place - correctly, and worth knowing before reading a UI that
/// offers the choice anyway.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum ImageDetail {
    #[default]
    Auto,
    Low,
    High,
}

impl ImageDetail {
    fn parse(s: &str) -> Result<Self, String> {
        match s {
            "auto" => Ok(Self::Auto),
            "low" => Ok(Self::Low),
            "high" => Ok(Self::High),
            other => Err(format!(
                "invalid image detail {other:?} (expected auto, low, or high)"
            )),
        }
    }

    /// Vision rows this level may spend on one image at `b`.
    pub(crate) fn token_cap(self, b: &paddock_engine::generator::VisionBudget) -> u32 {
        match self {
            // the model's published ceiling - the whole point of asking
            Self::High => b.max_tokens,
            // the smallest the tower will actually encode; going under it
            // discards detail without buying back a single row
            Self::Low => b.min_tokens,
            // the conservative default (see AUTO_MAX_TOKENS): what every
            // client was already getting before `detail` existed here
            Self::Auto => paddock_engine::generator::AUTO_MAX_TOKENS.min(b.max_tokens),
        }
    }
}

/// One image content part: where its bytes are, and how much resolution the
/// caller asked us to spend on them.
#[derive(Debug)]
pub(crate) struct ImageRef<'a> {
    pub url: std::borrow::Cow<'a, str>,
    pub detail: ImageDetail,
}

/// The `detail` on one image content part. Chat completions nests it inside
/// `image_url`; the Responses API puts it on the part itself. Both spellings
/// are accepted wherever they appear rather than per-surface, because the two
/// shapes cannot collide and a client that mixes them meant the same thing.
fn part_detail(part: &serde_json::Value) -> Result<ImageDetail, String> {
    let found = part
        .get("image_url")
        .and_then(|v| v.get("detail"))
        .or_else(|| part.get("detail"));
    match found {
        None | Some(serde_json::Value::Null) => Ok(ImageDetail::Auto),
        Some(v) => match v.as_str() {
            Some(s) => ImageDetail::parse(s),
            None => Err("image detail must be a string".into()),
        },
    }
}

/// Find image content parts, in render order. The detection condition
/// MIRRORS the qwen chat template's (`'image' in item or 'image_url' in item
/// or item.type == 'image'`) so the count always matches the `<|image_pad|>`
/// slots the template emits. Detection only - decoding happens after the
/// vision-capability check so a vision-less server gives the useful error.
pub(crate) fn find_images(messages: &[serde_json::Value]) -> Result<Vec<ImageRef<'_>>, String> {
    use std::borrow::Cow;
    let mut out = Vec::new();
    for msg in messages {
        let Some(parts) = msg.get("content").and_then(|c| c.as_array()) else {
            continue;
        };
        for part in parts {
            if part.get("video").is_some()
                || part.get("type").and_then(|t| t.as_str()) == Some("video")
            {
                return Err("video input is not supported".into());
            }
            let is_image = part.get("image").is_some()
                || part.get("image_url").is_some()
                || part.get("type").and_then(|t| t.as_str()) == Some("image");
            if !is_image {
                continue;
            }
            let detail = part_detail(part)?;
            // OpenAI chat / Responses: `image_url` (or `image`) is either the
            // data URI itself or `{url}`.
            if let Some(url) = ["image_url", "image"]
                .iter()
                .find_map(|k| part.get(k))
                .and_then(|v| v.as_str().or_else(|| v.get("url").and_then(|u| u.as_str())))
            {
                out.push(ImageRef {
                    url: Cow::Borrowed(url),
                    detail,
                });
                continue;
            }
            // Anthropic: `{"type":"image","source":{...}}`, which carries the
            // bytes as `{type:"base64", media_type, data}` - its only inline
            // shape, so /v1/messages could not take an image at all without
            // this. Reassembled into the same data URI the other surfaces send
            // so exactly one decoder sees images. `source.type == "url"` is
            // refused for the same SSRF/availability reason `decode_image_url`
            // refuses remote urls, but named here rather than as "no url".
            if let Some(src) = part.get("source") {
                let sty = src.get("type").and_then(|t| t.as_str()).unwrap_or("base64");
                if sty == "url" {
                    return Err(
                        "image source type \"url\" is not supported (the server does not \
                         fetch remote images); send type \"base64\" instead"
                            .into(),
                    );
                }
                if sty != "base64" {
                    return Err(format!("unsupported image source type {sty:?}"));
                }
                let media = src
                    .get("media_type")
                    .and_then(|m| m.as_str())
                    .ok_or("image source has no media_type")?;
                let data = src
                    .get("data")
                    .and_then(|d| d.as_str())
                    .ok_or("image source has no data")?;
                out.push(ImageRef {
                    url: Cow::Owned(format!("data:{media};base64,{data}")),
                    detail,
                });
                continue;
            }
            return Err("image content part has no url".into());
        }
    }
    Ok(out)
}

/// One audio content part: the base64 payload and the declared format (used
/// only for error messages - the decoder sniffs the bytes).
#[derive(Debug)]
pub(crate) struct AudioRef {
    pub b64: String,
    pub format: String,
}

/// Find audio content parts, in render order. Accepted shapes:
/// OpenAI chat's `{"type":"input_audio","input_audio":{"data":b64,"format":..}}`
/// and vLLM's `audio_url` extension carrying a
/// `data:audio/...` URI (string or `{url}`). Remote URLs are refused for the
/// same SSRF/availability reason images refuse them. Detection only - the
/// capability check and decode happen at the call site so an audio-less
/// server gives the useful error.
pub(crate) fn find_audio(messages: &[serde_json::Value]) -> Result<Vec<AudioRef>, String> {
    let mut out = Vec::new();
    for msg in messages {
        let Some(parts) = msg.get("content").and_then(|c| c.as_array()) else {
            continue;
        };
        for part in parts {
            if let Some(ia) = part.get("input_audio") {
                let data = ia
                    .get("data")
                    .and_then(|d| d.as_str())
                    .ok_or("input_audio part has no data")?;
                let format = ia
                    .get("format")
                    .and_then(|f| f.as_str())
                    .unwrap_or("wav")
                    .to_owned();
                out.push(AudioRef {
                    b64: data.to_owned(),
                    format,
                });
                continue;
            }
            if let Some(au) = part.get("audio_url") {
                let url = au
                    .as_str()
                    .or_else(|| au.get("url").and_then(|u| u.as_str()))
                    .ok_or("audio_url part has no url")?;
                let Some(rest) = url.strip_prefix("data:audio/") else {
                    return Err(
                        "only data:audio/... URIs are supported (the server does not fetch \
                         remote audio); inline the clip as base64"
                            .into(),
                    );
                };
                let (meta, b64) = rest.split_once(',').ok_or("malformed data:audio URI")?;
                let format = meta.split(';').next().unwrap_or("wav").to_owned();
                out.push(AudioRef {
                    b64: b64.to_owned(),
                    format,
                });
                continue;
            }
            // a bare `{"type":"audio"}` marker carries no payload - that shape
            // only exists POST-normalization; a client sending it meant to
            // send one of the two payload-bearing shapes
            if part.get("type").and_then(|t| t.as_str()) == Some("audio") {
                return Err("audio content part has no data (send `input_audio` or a \
                            data:audio `audio_url`)"
                    .into());
            }
        }
    }
    Ok(out)
}

/// Decode audio parts to 16 kHz mono f32 - the exact pipeline
/// `/v1/audio/transcriptions` runs. Every container OpenAI's transcription
/// endpoint accepts works here too, which is a deliberate SUPERSET of this
/// surface's own spec: OpenAI's chat `input_audio` part declares only wav and
/// mp3, and refusing a flac a caller already holds - on the one endpoint that
/// takes the same bytes as its sibling - would be a distinction with no reason
/// behind it. The declared `format` is recorded in the refusal but never
/// trusted: the container comes from the bytes.
pub(crate) fn decode_audio_parts(
    refs: Vec<AudioRef>,
    frontend: crate::serving::AudioFrontend,
) -> Result<Vec<MmChunk>, String> {
    use base64::Engine as _;
    refs.into_iter()
        .map(|r| {
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(r.b64.trim())
                .map_err(|e| format!("audio base64: {e}"))?;
            let wav = paddock_engine::audio::decode::decode_audio(&bytes)
                .map_err(|e| format!("audio decode (declared format {:?}): {e}", r.format))?;
            if wav.samples.is_empty() {
                return Err("audio part holds no samples".into());
            }
            let samples =
                paddock_engine::audio::resample::resample(&wav.samples, wav.sample_rate, 16000)?;
            // mel here, not engine-side: the request thread pays
            // the host DSP so concurrent requests' frontends run in parallel
            // instead of serializing on the engine thread. Which contract
            // runs is the served model's, never a default - see
            // `AudioFrontend`.
            let mel = frontend.features(&samples)?;
            Ok(MmChunk::Audio {
                samples,
                mel: Some(mel),
            })
        })
        .collect()
}

/// Decode a `data:` image URI into interleaved RGB8, fitted to what `detail`
/// allows on this endpoint. Accepted: png/jpeg/webp/gif (both vendor
/// contracts) plus bmp and tiff (local-first superset - the files users
/// actually have; pdfium's PDF rasterizer set the convert-for-the-user
/// precedent). The format comes from the bytes (magic sniff), never the
/// declared media type. Animated gif yields its first frame - OpenAI's own
/// gif scope is "non-animated". A MULTI-page tiff never reaches here on a
/// vision server: `expand_attachments` already replaced it with per-page
/// parts (`crate::tiffdoc`, document semantics); a single-page tiff decodes
/// on this lane like any picture. Alpha is dropped uncomposited, the same
/// treatment transparent PNGs have always had here. AVIF/HEIC are the remaining wild formats: both need native decoders
/// (rav1d/dav1d, libde265 - HEIC is patent-encumbered besides), so they are
/// a pdfium-style optional pack if ever, not a feature flag. Remote URLs are
/// an explicit non-feature: fetching them from the server is an SSRF hazard
/// and an availability dependency - clients inline base64 instead.
///
/// `budget` is the served tower's, or None on a model without one (then this
/// only decodes - a vision-less server has already refused the request by the
/// time we get here, and the standalone decode is what the tests exercise).
pub(crate) fn decode_image_url(
    url: &str,
    budget: Option<paddock_engine::generator::VisionBudget>,
    detail: ImageDetail,
) -> Result<RequestImage, String> {
    use base64::Engine as _;
    let Some(rest) = url.strip_prefix("data:") else {
        return Err(
            "only data: image URIs are supported (the server does not fetch remote URLs); \
             inline the image as base64"
                .into(),
        );
    };
    let (meta, b64) = rest.split_once(',').ok_or("malformed data: URI")?;
    if !meta.ends_with(";base64") {
        return Err("data: image URI must be base64-encoded".into());
    }
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64.trim())
        .map_err(|e| format!("image base64: {e}"))?;
    // an actionable refusal: the crate's error names the format it found
    // (or failed to sniff); we add what would work so the client can
    // convert instead of guessing. The HEIF pair is listed only when this
    // install can actually read it - promising a format we would then refuse
    // is worse than a short list.
    let fmt_err = |e: &dyn std::fmt::Display| {
        format!("image decode: {e} (accepted formats: png, jpeg, webp, gif, tiff, bmp, avif)")
    };

    // AVIF never reaches the `image` crate, which has no decoder for it - and
    // neither does HEIC, but for the opposite reason. AVIF is decoded here by
    // rav1d, linked in. HEIC is HEVC and is REFUSED, permanently: no HEVC
    // decoder can be embedded in a closed binary without publishing relinkable
    // object code. So the message names the format and the
    // codec rather than implying something is missing from the install.
    let mut rgb = if let Some(codec) = paddock_heif::sniff(&bytes) {
        let r = paddock_heif::decode(&bytes).map_err(|e| match e {
            paddock_heif::Error::NoDecoder { codec } => format!(
                "image decode: {} photos use HEVC, which this server cannot decode - \
                 convert to JPEG, PNG or AVIF",
                codec.label()
            ),
            other => format!("image decode: {} photo: {other}", codec.label()),
        })?;
        // No orientation pass here, unlike the branch below. paddock-heif has
        // already applied the container's own irot/imir transforms, so these
        // pixels are upright; applying an EXIF Orientation tag on top would
        // rotate a portrait photo twice.
        image::RgbImage::from_raw(r.width, r.height, r.rgb)
            .ok_or_else(|| "image decode: decoded plane is the wrong size".to_string())?
    } else {
        let reader = image::ImageReader::new(std::io::Cursor::new(&bytes))
            .with_guessed_format()
            .map_err(|e| fmt_err(&e))?;
        let mut decoder = reader.into_decoder().map_err(|e| fmt_err(&e))?;
        // EXIF orientation before anything downstream: phones store
        // sensor-native pixels plus a rotation flag, so a tower fed the raw
        // buffer sees sideways or mirrored images - and nothing downstream can
        // tell. Uprighting is correctness, not preprocessing; a flag that fails
        // to parse means "no transform", never a refused image.
        use image::ImageDecoder as _;
        let orientation = decoder
            .orientation()
            .unwrap_or(image::metadata::Orientation::NoTransforms);
        let mut img = image::DynamicImage::from_decoder(decoder).map_err(|e| fmt_err(&e))?;
        img.apply_orientation(orientation);
        img.to_rgb8()
    };
    // Down only, and only for `detail: low`. Low is the one level whose cap
    // can bind below the family's own budget resample; Auto's cap
    // (AUTO_MAX_TOKENS) and High's (the published max) are >= every family
    // budget in practice, so pre-shrinking for them made the ordinary path a
    // double resample (Triangle here, then the family's bit-exact bicubic) -
    // measured at ~100ms inline per A4-class request, and a
    // divergence from the reference processors, which resample exactly once
    // from the original. The family's own preprocessing is that single
    // resample; upsampling small images stays the tower's job (it knows its
    // alignment grid).
    if let (Some(b), ImageDetail::Low) = (budget, detail) {
        let (tw, th) = b.fit_tokens(rgb.width(), rgb.height(), detail.token_cap(&b));
        if tw < rgb.width() || th < rgb.height() {
            // Triangle, not Lanczos3: the `image` crate widens a filter's
            // support by the downscale ratio, so bilinear here is genuinely
            // area-averaged rather than 2-tap point sampling - real
            // antialiasing at roughly a third of Lanczos3's cost, and the same
            // filter family the engine's own resize uses. This runs inline on
            // the request thread, which is why the cheaper correct filter wins.
            rgb = image::imageops::resize(&rgb, tw, th, image::imageops::FilterType::Triangle);
        }
    }
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    Ok(RequestImage {
        rgb: rgb.into_raw(),
        w,
        h,
    })
}

/// vLLM's `mm_processor_kwargs`, restricted to the two keys any family
/// actually reads (`min_pixels`, `max_pixels` - the paddleocr smart-resize
/// budget). Unknown keys are a loud 400: a typo'd knob silently ignored is a
/// wrong answer with no witness. Returns None when absent or empty; the
/// family fills whichever half the caller left out with its own default.
pub(crate) fn parse_mm_processor_kwargs(
    v: Option<&serde_json::Value>,
) -> Result<Option<(Option<u64>, Option<u64>)>, String> {
    let Some(v) = v else { return Ok(None) };
    let obj = v
        .as_object()
        .ok_or("mm_processor_kwargs must be an object")?;
    let (mut min, mut max) = (None, None);
    for (k, val) in obj {
        let n = val
            .as_u64()
            .filter(|&n| n > 0)
            .ok_or_else(|| format!("mm_processor_kwargs.{k} must be a positive integer"))?;
        match k.as_str() {
            "min_pixels" => min = Some(n),
            "max_pixels" => max = Some(n),
            other => {
                return Err(format!(
                    "mm_processor_kwargs key {other:?} is not served \
                     (supported: min_pixels, max_pixels)"
                ));
            }
        }
    }
    Ok((min.is_some() || max.is_some()).then_some((min, max)))
}

/// Decode every image content part of a request, each fitted to its own
/// `detail`. Shared by all three API surfaces so one policy applies to
/// `/v1/chat/completions`, `/v1/responses` and `/v1/messages` alike.
pub(crate) fn decode_images(
    refs: Vec<ImageRef<'_>>,
    budget: Option<paddock_engine::generator::VisionBudget>,
) -> Result<Vec<RequestImage>, String> {
    refs.into_iter()
        .map(|r| decode_image_url(&r.url, budget, r.detail))
        .collect()
}

/// A multimodal prompt, split two ways - both halves come from one walk of the
/// rendered token stream, and a caller cannot get one without the other.
///
/// That pairing is the point. The placeholder is a position marker, not a
/// token the model ever sees: the engine splices the encoded vision rows in at
/// the slot. Anything downstream that treats the prompt as a token SEQUENCE -
/// generation history, repetition/presence/frequency penalties, the pre-prefill
/// token count - has to read the pad-free stream instead, and for a while two
/// of the three API surfaces read the wrong one because the filtering lived at
/// the call sites (six of them) rather than here.
#[derive(Debug)]
pub(crate) struct MmPrompt {
    /// Interleaved text/image chunks: what the engine prefills.
    pub chunks: Vec<MmChunk>,
    /// The same prompt with every placeholder removed: what the engine is given
    /// as `GenRequest.prompt`, and whose length is the text-token count.
    pub text_ids: Vec<u32>,
}

/// Split the rendered prompt at its media pad tokens (`<|image_pad|>` /
/// `<|audio_pad|>`) and interleave the decoded media chunks. Errors when the
/// template's slot count disagrees with the extracted media (a
/// template/extraction drift would silently drop content).
pub(crate) fn build_mm_chunks(
    prompt_ids: &[u32],
    pad_id: u32,
    media: Vec<MmChunk>,
    task: Option<&crate::chat_template::TaskTag>,
) -> Result<MmPrompt, String> {
    let n_media = media.len();
    let mut media = media.into_iter();
    let mut chunks = Vec::new();
    let mut text: Vec<u32> = Vec::new();
    let mut text_ids: Vec<u32> = Vec::with_capacity(prompt_ids.len());
    let mut n_pads = 0usize;
    for &t in prompt_ids {
        if t == pad_id {
            n_pads += 1;
            let Some(m) = media.next() else { continue };
            chunks.push(MmChunk::Text(std::mem::take(&mut text)));
            chunks.push(m);
        } else {
            text.push(t);
            text_ids.push(t);
        }
    }
    if n_pads != n_media {
        // The one case with a known cause gets a message that says what to
        // change: granite-vision's template rebuilds a task-tagged turn as
        // IBM's canned prompt around ONE image slot, so a multi-page document
        // (one image per page) sent with `<tables_json>` and friends renders
        // one slot for N pages. The tasks are single-image by design - IBM's
        // own document pipeline drives the model one page per request - and
        // that is the answer here too, not a silent drop of N-1 pages.
        if let Some(t) = task
            && n_pads < n_media
        {
            return Err(format!(
                "{} is a one-image task on this model: its prompt renders one image                  slot and this request carries {n_media} images (a multi-page document                  is one image per page) - send one page per request",
                t.tag
            ));
        }
        return Err(format!(
            "template rendered {n_pads} media slot(s) for {n_media} media item(s)"
        ));
    }
    chunks.push(MmChunk::Text(text));
    Ok(MmPrompt { chunks, text_ids })
}

/// The task tag whose canned instruction the rendered prompt carries, if any.
/// The template already swapped the tag for the instruction by the time we
/// look, so the instruction being present is what proves a tag fired - the
/// user's own text is not consulted, which keeps this true for every API
/// surface without each of them parsing content parts.
pub(crate) fn fired_task_tag<'a>(
    prompt: &str,
    tags: &'a [crate::chat_template::TaskTag],
) -> Option<&'a crate::chat_template::TaskTag> {
    tags.iter()
        .find(|t| !t.prompt.is_empty() && prompt.contains(&t.prompt))
}

/// Parse the OpenAI logit_bias map ({"token_id": -100..100}) into sampler
/// pairs, validating ids against the vocab. Shared with /v1/completions.
pub(crate) fn parse_logit_bias(
    bias: Option<&std::collections::HashMap<String, f32>>,
    vocab: usize,
) -> Result<Vec<(u32, f32)>, String> {
    let Some(m) = bias else { return Ok(Vec::new()) };
    let mut out = Vec::with_capacity(m.len());
    for (k, &b) in m {
        let id: u32 = k
            .parse()
            .map_err(|_| format!("logit_bias key {k:?} is not a token id"))?;
        if id as usize >= vocab {
            return Err(format!("logit_bias token id {id} is out of vocabulary"));
        }
        if !(-100.0..=100.0).contains(&b) {
            return Err(format!(
                "logit_bias for token {id} must be in -100..100 (got {b})"
            ));
        }
        out.push((id, b));
    }
    Ok(out)
}

/// Fold an OpenAI `reasoning_effort` into this model's template kwargs.
///
/// The spec's vocabulary is none|minimal|low|medium|high|xhigh|max, and `none`
/// officially means "do not reason". What a given checkpoint can do with that
/// is `model.reasoning` - measured from its own template at load, not inferred
/// from its family, because the family cannot answer it: Qwen3.5, 3.6 and 3.8
/// all report arch `qwen35` and dialect `QwenXml`, and 3.8 is the only one of
/// the three with a ladder (see `crate::reasoning`).
///
/// Three cases, and 3.8 is the first model to need the third:
///
/// - **A ladder and an off position** (Qwen3.8: low/medium/xhigh plus
///   `enable_thinking`). `none` turns thinking off - and sends no rung, since
///   an effort that grades a thought process the template is not rendering is
///   noise. Every other level sets the rung.
/// - **A ladder and no off position** (gpt-oss, Muse Glimmer - both render
///   their reasoning preamble unconditionally). `none` clamps to the lowest
///   rung rather than pretending to disable something the template always
///   writes.
/// - **An off position and no ladder** (Qwen3.5/3.6, gemma4, laguna): `none` ->
///   off, any other level -> on. Pretending a toggle model honors `xhigh`
///   differently from `low` would be exactly the plausible-but-wrong behavior
///   this project bans; what it can answer is the question a client asks with
///   `none`.
///
/// The vocabulary is validated on every path, so an invalid level is a 400 even
/// where the value then collapses.
///
/// An explicit `chat_template_kwargs` entry still wins: it is the lower-level
/// knob and a caller who set it meant it.
pub(crate) fn merge_reasoning_effort(
    caps: &crate::reasoning::ReasoningCaps,
    effort: &str,
    kwargs: Option<serde_json::Value>,
) -> Result<Option<serde_json::Value>, String> {
    // validate the seven-value vocabulary first, on every path
    let rank = reasoning_effort_rank(effort)?;
    if !caps.reasons() {
        return Err(
            "reasoning_effort is not settable on this model: its chat template has no \
             reasoning mode to constrain"
                .to_owned(),
        );
    }
    let (key, value): (&str, serde_json::Value) = if effort == "none" && caps.off {
        ("enable_thinking", serde_json::json!(false))
    } else if let Some(kw) = caps.kwarg {
        // `clamp` cannot be None here: a model with a kwarg has rungs
        let rung = caps
            .clamp(rank)
            .ok_or("this model grades reasoning effort but publishes no levels")?;
        (kw, serde_json::json!(rung))
    } else {
        ("enable_thinking", serde_json::json!(true))
    };
    let mut obj = kwargs
        .as_ref()
        .and_then(|k| k.as_object().cloned())
        .unwrap_or_default();
    obj.entry(key.to_owned()).or_insert(value);
    Ok(Some(serde_json::Value::Object(obj)))
}

/// Range-check one sampling knob against the vendor's documented bounds.
/// `None` (unset) always passes; NaN never does.
pub(crate) fn in_range(name: &str, v: Option<f32>, lo: f32, hi: f32) -> Result<(), String> {
    match v {
        Some(x) if !(lo..=hi).contains(&x) => {
            Err(format!("{name} must be between {lo} and {hi} (got {x})"))
        }
        _ => Ok(()),
    }
}

/// The sampling knobs both OpenAI surfaces share, checked against the ranges
/// the published schema declares.
///
/// These used to be accepted at any value. Mostly that is merely sloppy -
/// `top_p: 2.0` clamps to the same thing as 1.0 - but a NEGATIVE temperature
/// is not: the sampler divides logits by it, which inverts the distribution,
/// so the model starts confidently emitting its least likely token. A caller
/// who fat-fingers a minus sign should get a 400, not fluent nonsense. (The
/// observed behavior at temperature -1 was a normal-looking greeting, which is
/// exactly why nobody had noticed.)
///
/// `temp_max` differs by vendor: OpenAI documents 0..2, Anthropic 0..1.
pub(crate) fn validate_sampling(
    temperature: Option<f32>,
    temp_max: f32,
    top_p: Option<f32>,
    min_p: Option<f32>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
) -> Result<(), String> {
    in_range("temperature", temperature, 0.0, temp_max)?;
    in_range("top_p", top_p, 0.0, 1.0)?;
    // paddock extension, but a cumulative-probability floor outside 0..1 is
    // meaningless on its own terms
    in_range("min_p", min_p, 0.0, 1.0)?;
    in_range("frequency_penalty", frequency_penalty, -2.0, 2.0)?;
    in_range("presence_penalty", presence_penalty, -2.0, 2.0)?;
    Ok(())
}

/// Validate the spec's seven-value vocabulary and rank it. The API grew from
/// three values to seven (none, minimal, low, medium, high, xhigh, max); local
/// templates know fewer, so the outer values clamp to the nearest rung rather
/// than 400ing requests from current SDKs.
/// Is `effort` inside the ladder's vocabulary? Exists so the Anthropic side
/// can assert that every rung Anthropic publishes is one this server already
/// understood, without exposing the rank itself.
#[cfg(test)]
pub(crate) fn reasoning_effort_rank_is_valid(effort: &str) -> bool {
    reasoning_effort_rank(effort).is_ok()
}

fn reasoning_effort_rank(effort: &str) -> Result<usize, String> {
    // none/minimal collapse into low, and max into xhigh: neither end has a
    // template that spells them, and a request from a current SDK must not 400
    Ok(match effort {
        "none" | "minimal" | "low" => 0,
        "medium" => 1,
        "high" => 2,
        "xhigh" | "max" => 3,
        other => {
            return Err(format!(
                "invalid reasoning_effort {other:?} (expected one of none, minimal, low, \
                 medium, high, xhigh, max)"
            ));
        }
    })
}

/// Two OpenAI chat options this server does not serve, refused by NAME rather
/// than by serde's generic unknown-field message.
///
/// Both were on the "implementable, we have the subsystem" list until the code
/// was actually read. Having the subsystem turned out to be necessary and not
/// sufficient - each needs a REQUEST-SCOPED path into it that does not exist:
///
/// `prediction` (Predicted Outputs) hands the server a draft of the expected
/// answer and asks it to verify rather than generate. That is speculative
/// decoding with a caller-supplied draft, and `GenRequest` has no field for
/// one: this engine's spec lane is model-attached (MTP heads, a DFlash
/// drafter), so the draft comes from the checkpoint and never from the wire.
/// Wiring it is an engine feature - a per-request draft threaded through the
/// scheduler and the verify lane - not a parameter.
///
/// `web_search_options` asks the SERVER to run searches and fold the results
/// into one answer. `/v1/chat/completions` has no server-side tool loop at all
/// - that lives on `/v1/responses`, which already serves a `web_search` tool
///   against the same paddock-websearch providers. Accepting the option here
///   would return an answer that never searched.
///
/// Naming the reason and the surface that does serve it beats "Unrecognized
/// request argument", which is true and useless.
fn refuse_unserved_options(req: &ChatCompletionRequest) -> Result<(), String> {
    if req.prediction.is_some() {
        return Err(
            "prediction (Predicted Outputs) is not served: it needs speculative decoding from a caller-supplied draft, and this engine drafts from the model itself (MTP / DFlash heads). Send the request without it."
                .into(),
        );
    }
    if req.web_search_options.is_some() {
        return Err(
            "web_search_options is not served on /v1/chat/completions, which runs no server-side tool loop. Use /v1/responses with a `web_search` tool - same providers, same keys - or call a search tool yourself and pass the results in."
                .into(),
        );
    }
    Ok(())
}

/// Translate the DEPRECATED `functions` / `function_call` request fields onto
/// `tools` / `tool_choice`, in place, so everything downstream sees exactly one
/// spelling. Returns whether the request was in the legacy shape.
///
/// This is the pre-2023-11 OpenAI tool protocol. OpenAI still accepts it and a
/// lot of pinned client code - LangChain-era wrappers, vendored helper
/// libraries - still emits it, so refusing was costing us real drop-in
/// compatibility for no gain: the two shapes carry identical information.
///
/// Three details that are easy to get wrong:
///   - a legacy function is the INNER object; `tools` wants it wrapped as
///     `{"type":"function","function":{...}}`
///   - legacy forces a call with `{"name":"x"}`, not tool_choice's
///     `{"type":"function","function":{"name":"x"}}`
///   - the legacy protocol has one call per turn and no id, so parallel calls
///     are pinned off rather than silently truncated on the way out
fn adopt_legacy_functions(req: &mut ChatCompletionRequest) -> Result<(), String> {
    if req.functions.is_none() && req.function_call.is_none() {
        return Ok(());
    }
    // Mixing the two generations is ambiguous about which one the answer
    // should wear, so it is a 400 rather than a precedence rule.
    if req.functions.is_some() && req.tools.is_some() {
        return Err("specify either the deprecated `functions` or `tools`, not both".into());
    }
    if req.function_call.is_some() && req.tool_choice.is_some() {
        return Err(
            "specify either the deprecated `function_call` or `tool_choice`, not both".into(),
        );
    }
    if req.function_call.is_some() && req.functions.is_none() {
        return Err("`function_call` requires `functions`".into());
    }

    if let Some(fns) = req.functions.as_ref() {
        if fns.is_empty() {
            return Err("`functions` must not be empty".into());
        }
        let mut tools = Vec::with_capacity(fns.len());
        for f in fns {
            if f.get("name").and_then(|n| n.as_str()).is_none() {
                return Err("each entry of `functions` needs a string `name`".into());
            }
            tools.push(serde_json::json!({"type": "function", "function": f}));
        }
        req.tools = Some(tools);
        // One call per answer: see the doc comment.
        req.parallel_tool_calls = Some(false);
    }

    match req.function_call.take() {
        None => {}
        Some(v) if v.as_str() == Some("none") => req.tool_choice = Some(serde_json::json!("none")),
        Some(v) if v.as_str() == Some("auto") => req.tool_choice = Some(serde_json::json!("auto")),
        Some(v) if v.get("name").and_then(|n| n.as_str()).is_some() => {
            let name = v["name"].as_str().expect("guard").to_owned();
            req.tool_choice =
                Some(serde_json::json!({"type": "function", "function": {"name": name}}));
        }
        Some(other) => {
            return Err(format!(
                "invalid function_call {other} (expected \"none\", \"auto\", or {{\"name\": ...}})"
            ));
        }
    }
    Ok(())
}

fn prepare(
    model: &ServingModel,
    req: &ChatCompletionRequest,
    output_ceiling: Option<usize>,
    sd: &crate::routes::SamplingDefaults,
) -> Result<Prepared, String> {
    let template = model
        .chat_template
        .as_deref()
        .ok_or("this model has no chat template; use /v1/completions")?;

    // token cap: max_completion_tokens is the current spelling, max_tokens
    // the deprecated one; both at once is ambiguous (matches OpenAI's error)
    let max_tokens = match (req.max_tokens, req.max_completion_tokens) {
        (Some(_), Some(_)) => {
            return Err(
                "specify either max_completion_tokens or the deprecated max_tokens, not both"
                    .into(),
            );
        }
        (Some(m), None) | (None, Some(m)) => m,
        (None, None) => 1024,
    };
    // Server-wide output ceiling (PADDOCK_MAX_OUTPUT_CEILING) - the same
    // clamp /v1/responses applies; a request can't demand more than the
    // deployment allows.
    let max_tokens = output_ceiling.map_or(max_tokens, |c| max_tokens.min(c));

    // persistence / modality knobs this server truthfully does not have
    if req.store == Some(true) {
        return Err("completions are not stored on this server; omit store or pass false".into());
    }
    if let Some(md) = req.metadata.as_ref()
        && !(md.is_null() || md.as_object().is_some_and(|o| o.is_empty()))
    {
        return Err(
            "metadata requires stored completions, which this server does not support".into(),
        );
    }
    if let Some(m) = req.modalities.as_ref()
        && m.as_slice() != ["text"]
    {
        return Err("only the \"text\" modality is supported".into());
    }

    // The chat tools union is `function` | `custom`. A `custom` tool is a
    // freeform-text tool with no JSON schema, which no dialect grammar here
    // can express - and accepting it silently (which is what happened before)
    // tells a client the tool is available when the model can never call it.
    if let Some(ts) = req.tools.as_deref() {
        for t in ts {
            match t.get("type").and_then(|v| v.as_str()) {
                // absent `type` is the pre-2024 shape, still a function
                Some("function") | None => {}
                Some(other) => {
                    return Err(format!(
                        "unsupported tool type {other:?} (this server serves `function` \
                         tools; a custom tool has no schema to constrain)"
                    ));
                }
            }
        }
    }

    // tool_choice: "none" hides tools from the render; "auto" is the default;
    // "required" / named function forces a call via the tool-call grammar
    // (qwen dialect; the Harmony channel structure is a follow-up).
    let mut tools = req.tools.as_deref();
    // None = free; Some(None) = any tool; Some(Some(name)) = that tool
    let mut forced_tool: Option<Option<String>> = None;
    match req.tool_choice.as_ref() {
        None => {}
        Some(v) if v.as_str() == Some("auto") => {}
        Some(v) if v.as_str() == Some("none") => tools = None,
        Some(v) if v.as_str() == Some("required") => forced_tool = Some(None),
        Some(v)
            if v.get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .is_some() =>
        {
            let name = v["function"]["name"].as_str().expect("guard").to_owned();
            forced_tool = Some(Some(name));
        }
        Some(other) => return Err(format!("invalid tool_choice {other}")),
    }
    let tool_syntax =
        match forced_tool {
            None => None,
            Some(_) => Some(model.dialect.tool_syntax().ok_or_else(|| {
                no_forced_tool_grammar(model.dialect, "\"required\"/named function")
            })?),
        };

    // response_format: enforced with constrained decoding (never prompting)
    let rf_schema = match req.response_format.as_ref() {
        None => None,
        Some(v) => match v.get("type").and_then(|t| t.as_str()) {
            None | Some("text") => None,
            Some("json_object") => Some(CompiledSchema::any_json()),
            Some("json_schema") => {
                let schema = v
                    .get("json_schema")
                    .and_then(|j| j.get("schema"))
                    .ok_or("response_format.json_schema.schema is required")?;
                Some(CompiledSchema::compile(schema)?)
            }
            Some(other) => {
                return Err(format!("unsupported response_format type {other:?}"));
            }
        },
    };
    if rf_schema.is_some() && forced_tool.is_some() {
        return Err(
            "response_format cannot be combined with a forced tool_choice (the output \
             cannot be both a JSON answer and a tool call)"
                .into(),
        );
    }

    // sampling / choice-count / logprobs validation
    validate_sampling(
        req.temperature,
        2.0,
        req.top_p,
        req.min_p,
        Some(req.frequency_penalty),
        Some(req.presence_penalty),
    )?;
    if req.n == 0 || req.n > 8 {
        return Err(format!("n must be 1..=8 (got {})", req.n));
    }
    if let Some(k) = req.top_logprobs {
        if !req.logprobs {
            return Err("top_logprobs requires logprobs: true".into());
        }
        if k > 20 {
            return Err(format!("top_logprobs must be 0..=20 (got {k})"));
        }
    }
    let logprobs = req.logprobs.then(|| req.top_logprobs.unwrap_or(0));
    // stream_options is still VALIDATED (stream-only, matches OpenAI's
    // error) but no longer gates behavior: the terminal usage chunk is
    // always emitted (see stream_response's rationale).
    if req.stream_options.is_some() && !req.stream {
        return Err("stream_options requires stream: true".into());
    }

    // image content parts: capability-checked first (the useful error), then
    // decoded and fitted to each part's `detail`, injected at the template's
    // <|image_pad|> slots after tokenization
    let image_refs = find_images(&req.messages)?;
    if !image_refs.is_empty() && !model.supports_vision {
        return Err(
            "this model is not serving vision (a vision-capable model needs its `mmproj` \
             companion file set in the config to accept image input)"
                .into(),
        );
    }
    // The inverse gate: a document parser given no document free-runs
    // transcription-vocabulary noise to the token cap. Runs after
    // expand_attachments, so a PDF/TIFF attachment has already become page
    // images and passes.
    if image_refs.is_empty() && model.document_parser {
        return Err(
            "this model is a document parser - attach an image (or a PDF, which is sent \
             as page images) for it to read; it cannot answer text-only prompts"
                .into(),
        );
    }
    let t_img = std::time::Instant::now();
    let images = decode_images(image_refs, model.engine.vision_budget())?;
    if !images.is_empty() && paddock_models::dev_var_os!("PADDOCK_REQ_TRACE").is_some() {
        eprintln!(
            "req-trace: decode_images {:.1} ms ({} images), done at {}",
            t_img.elapsed().as_secs_f64() * 1e3,
            images.len(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_micros())
        );
    }

    // audio content parts: same capability-first split as images.
    // Decoded to 16 kHz mono here; the template renders one <|audio_pad|>
    // slot per part and build_mm_chunks interleaves the clips at the slots.
    let audio_refs = find_audio(&req.messages)?;
    if !audio_refs.is_empty() && !model.supports_audio {
        return Err(
            "this model is not serving audio (an ASR model needs its audio `mmproj` \
             companion file set in the config to accept audio input)"
                .into(),
        );
    }
    let audio = decode_audio_parts(audio_refs, model.audio_frontend)?;

    // verbosity: current-spec length hint, validated for conformance; local
    // models have no verbosity knob, so a valid value changes nothing
    if let Some(v) = req.verbosity.as_deref()
        && !matches!(v, "low" | "medium" | "high")
    {
        return Err(format!(
            "invalid verbosity {v:?} (expected low, medium, or high)"
        ));
    }

    // reasoning_effort: graded on gpt-oss, an on/off toggle everywhere else
    // that reasons at all (see merge_reasoning_effort)
    let mut kwargs = req.chat_template_kwargs.clone();
    if let Some(effort) = req.reasoning_effort.as_deref() {
        kwargs = merge_reasoning_effort(&model.reasoning, effort, kwargs)?;
    }
    // reasoning: {"max_tokens": N} - the OpenRouter-shaped thinking budget
    // (effort stays on reasoning_effort above). Resolved against the rendered
    // prompt further down, once thinking_open is known.
    let budget_req = match req.reasoning.as_ref() {
        None => None,
        Some(r) => {
            let obj = r.as_object().ok_or("`reasoning` must be a JSON object")?;
            for k in obj.keys() {
                if k != "max_tokens" {
                    return Err(format!(
                        "unsupported reasoning parameter {k:?} (effort goes on reasoning_effort)"
                    ));
                }
            }
            match obj.get("max_tokens") {
                None => None,
                Some(v) => Some(
                    v.as_u64()
                        .filter(|&n| n >= 1)
                        .ok_or("reasoning.max_tokens must be a positive integer")?
                        as usize,
                ),
            }
        }
    };

    // OpenAI arguments-strings -> objects, or templates drop them from history
    let trace = paddock_models::dev_var_os!("PADDOCK_REQ_TRACE").is_some();
    let t0 = std::time::Instant::now();
    // Name anything we don't serve before the template sees it - see
    // validate_content_parts for why a template is the wrong place to find out.
    chat_template::validate_roles(&req.messages)?;
    chat_template::validate_content_parts(&req.messages)?;
    let mut messages = chat_template::normalize_messages(&req.messages);
    if let Some(marker) = model.audio_inline_marker.as_deref() {
        chat_template::inline_audio_content(&mut messages, marker);
    }
    // deepseek2-ocr instruction mapping: resolve the `ocr`
    // request object + prompt vocabulary - mutates `messages` (canonical
    // task string / grounding token) and carries the family's sampling
    // default and crop override out through `Prepared`.
    if req.ocr.is_some() && !model.ocr && !model.paddleocr {
        return Err("the `ocr` request object is only served by document-parser models".into());
    }
    let ocr = if model.ocr {
        let opts = crate::deepseek_ocr::OcrOpts::from_request(req.ocr.as_ref(), kwargs.as_ref())?;
        let sizes: Vec<(usize, usize)> = images.iter().map(RequestImage::size).collect();
        let max_tiles = model
            .engine
            .vision_budget()
            .map_or(0, |b| (b.max_pixels / (640 * 640)) as usize);
        crate::deepseek_ocr::resolve(&mut messages, opts, &sizes, max_tiles)?
    } else if model.paddleocr {
        let mode = crate::paddle_ocr::opts_from_request(req.ocr.as_ref(), kwargs.as_ref())?;
        crate::paddle_ocr::resolve(&mut messages, mode, images.len())?
    } else {
        None
    };
    let mut prompt = chat_template::render(template, &messages, tools, kwargs.as_ref())?;
    if trace {
        tracing::info!(
            "req-trace: render {:.1} ms ({} chars)",
            t0.elapsed().as_secs_f64() * 1e3,
            prompt.len()
        );
    }
    // muse-glimmer: pre-open the (unconditional) reasoning message so the
    // token sampled from the prefill logits is already visible reasoning text
    // - see muse::PREOPEN for the measured cost of letting the model type it.
    // Not under a forced tool: that grammar arms at token 1 and emits its own
    // ` to=NAME<|message|>` header, which a pre-opened prompt would corrupt.
    if model.dialect == crate::parsers::Dialect::MuseChannel
        && forced_tool.is_none()
        && crate::muse::preopen()
    {
        prompt.push_str(crate::muse::PREOPEN);
    }
    // thinking-mode detection is dialect-shaped - see Dialect::thinking_open
    // (qwen pre-opens "<think>\n", laguna a bare "<think>", gemma4 pre-closes
    // when off)
    let thinking_open = model.dialect.thinking_open(&prompt);
    let think_budget = budget_req
        .map(|n| think_budget(model, n, thinking_open, "reasoning.max_tokens"))
        .transpose()?;
    // gemma4 thinking: pre-open the thought channel so the token sampled from
    // the prefill logits is already visible reasoning text (see g4_preopen)
    if thinking_open
        && model.dialect == crate::parsers::Dialect::GemmaChannel
        && crate::parsers::g4_preopen()
    {
        prompt.push_str(crate::parsers::G_THOUGHT);
    }
    // encode with specials: the rendered prompt contains <|start|> etc. which
    // must map to their single ids
    let t1 = std::time::Instant::now();
    let mut prompt_ids = model.tokenizer.encode(&prompt).map_err(|e| e.to_string())?;
    if trace {
        eprintln!(
            "req-trace: encode {:.1} ms ({} tokens), send at {}",
            t1.elapsed().as_secs_f64() * 1e3,
            prompt_ids.len(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_micros())
        );
    }
    // BOS-leading families (gemma4): chat templates emit text only - the
    // leading BOS is the tokenizer's job, same as the raw-completions path
    if let Some(bos) = model.bos
        && prompt_ids.first() != Some(&bos)
    {
        prompt_ids.insert(0, bos);
    }

    let (mut mm_chunks, engine_prompt) = if !images.is_empty() {
        let pad = model
            .image_pad_id
            .ok_or("model has no <|image_pad|> token")?;
        let media = images.into_iter().map(RequestImage::into_chunk).collect();
        let task = fired_task_tag(&prompt, &model.task_tags);
        let mm = build_mm_chunks(&prompt_ids, pad, media, task)?;
        (Some(mm.chunks), mm.text_ids)
    } else if !audio.is_empty() {
        let pad = model
            .audio_pad_id
            .ok_or("model has no <|audio_pad|> token")?;
        let mm = build_mm_chunks(&prompt_ids, pad, audio, None)?;
        (Some(mm.chunks), mm.text_ids)
    } else {
        (None, prompt_ids.clone())
    };
    // the resolved crop override rides as a directive chunk the OCR engine
    // consumes in its planning pass (single image forced to base sizing)
    if let (Some(o), Some(chunks)) = (&ocr, mm_chunks.as_mut())
        && o.force_base
    {
        chunks.insert(
            0,
            MmChunk::OcrCrop(paddock_engine::service::OcrCropMode::Base),
        );
    }
    // vLLM-compat per-request pixel budget (paddleocr family): parsed with
    // unknown keys refused, injected as a directive the family consumes in
    // its own bit-exact resize. Sending it without an image is a caller bug
    // worth a 400, not a silently dropped knob.
    if let Some((min_px, max_px)) = parse_mm_processor_kwargs(req.mm_processor_kwargs.as_ref())? {
        let Some(chunks) = mm_chunks.as_mut() else {
            return Err("mm_processor_kwargs was sent but the request carries no image".into());
        };
        chunks.insert(
            0,
            MmChunk::VisionPixels {
                min_pixels: min_px,
                max_pixels: max_px,
            },
        );
    }

    // assemble the constraint SPEC: grammar + the dialect's content trigger
    // (reasoning stays free; the machine takes over at the content boundary);
    // instantiated per choice in handle()
    let mut constraint_spec = match forced_tool {
        Some(only) => ConstraintSpec::Tool(ToolSet::compile(
            tool_syntax.expect("gated with forced_tool"),
            tools.unwrap_or(&[]),
            only.as_deref(),
        )?),
        None => match rf_schema {
            Some(s) => ConstraintSpec::Json(s),
            None => auto_tool_dispatch(model, tools, req.parallel_tool_calls == Some(false)),
        },
    };
    // A forced choice or a response_format must refuse when the dialect gives
    // us nowhere to arm; auto dispatch is opportunistic and degrades to
    // unconstrained decoding instead of 400-ing an ordinary chat.
    let opportunistic = matches!(constraint_spec, ConstraintSpec::Dispatch(..));
    let gate = if matches!(constraint_spec, ConstraintSpec::None) {
        GateSpec::Immediate
    } else {
        match content_gate(
            model,
            thinking_open,
            matches!(constraint_spec, ConstraintSpec::Tool(_)),
        ) {
            Ok(g) => g,
            Err(why) if opportunistic => {
                tracing::debug!(why, "no content gate here - auto tool calls decode free");
                constraint_spec = ConstraintSpec::None;
                GateSpec::Immediate
            }
            Err(why) => return Err(why),
        }
    };

    // Request field wins; else this model's elected defaults for this turn -
    // the qwen cards publish one set for thinking and another for instruct,
    // and `thinking_open` on the rendered prompt is what knows which we are
    //Document parsers are the one family exception: their
    // checkpoints ship greedy generation configs (extraction is
    // deterministic - vLLM applies the same model-default rule), so an
    // untouched request decodes greedy.
    let dflt = sd.resolve(thinking_open);
    let sampler = SamplingParams {
        temperature: req.temperature.unwrap_or(if model.document_parser {
            0.0
        } else {
            dflt.temp
        }),
        top_k: req.top_k.unwrap_or(dflt.top_k),
        top_p: req.top_p.unwrap_or(dflt.top_p),
        min_p: req.min_p.unwrap_or(dflt.min_p),
        repeat_penalty: req.repeat_penalty.unwrap_or(dflt.repeat_penalty),
        repeat_last_n: req.repeat_last_n.unwrap_or(sd.repeat_last_n),
        presence_penalty: req.presence_penalty,
        frequency_penalty: req.frequency_penalty,
        seed: sd.seed_or_now(req.seed),
        logit_bias: parse_logit_bias(req.logit_bias.as_ref(), model.tokenizer.vocab_size)?,
        // the OCR family's required repetition guard (reference parity),
        // 35/128 single page, 35/1024 multi-page, caller-overridable
        no_repeat_ngram: ocr.as_ref().map_or((0, 0), |o| o.ngram),
    };

    Ok(Prepared {
        prompt_ids,
        engine_prompt,
        max_tokens,
        sampler,
        stop_tokens: if req.ignore_eos {
            Vec::new()
        } else {
            model.stop_tokens.clone()
        },
        stop_strings: req.stop.as_ref().map(|s| s.to_vec()).unwrap_or_default(),
        thinking_open,
        single_tool_call: req.parallel_tool_calls == Some(false),
        hints: tool_hints(tools),
        mm_chunks,
        constraint_spec,
        gate,
        think_budget,
        n: req.n as usize,
        logprobs,
        ocr,
        skip_special: req.skip_special_tokens,
        legacy_functions: req.functions.is_some(),
    })
}

struct Meta {
    id: String,
    model_id: String,
    tokenizer: Arc<paddock_tokenizer::GgufTokenizer>,
    prompt_len: usize,
    /// TEXT prompt tokens only (image pads removed) - `prompt_len` grows to the
    /// engine's real prefill row count once the Prefilled event lands, and the
    /// difference between the two is the vision-row cost of the request.
    text_prompt_len: usize,
    dialect: Dialect,
    thinking_open: bool,
    stop_strings: Vec<String>,
    single_tool_call: bool,
    hints: Option<ToolHints>,
    /// logprobs requested - attach entries to responses/chunks
    logprobs: bool,
    /// answer in the DEPRECATED `function_call` shape (see Prepared)
    legacy_functions: bool,
    /// The media rows are a CLIP's, not a picture's - picks which usage field
    /// `media_tokens` reports under (`audio_tokens` is OpenAI's own; the
    /// image one is our extension). A request never mixes the two: the
    /// families that take audio take no images.
    media_is_audio: bool,
    /// deepseek2-ocr resolution - echoed as the response's `ocr` extension,
    /// with grounded `regions` parsed from the finished output when armed.
    ocr: Option<crate::deepseek_ocr::OcrResolved>,
    /// request `skip_special_tokens` - applied to the content decode when the
    /// caller sent it; None keeps the dialect's own behavior
    skip_special: Option<bool>,
    /// Event-record slots (§8.1); no-op unless the events middleware planted one.
    scope: crate::events::EventScope,
}

impl Meta {
    /// Prefill rows this request's media cost, 0 when there was none - the
    /// gap between what the tokenizer counted (one placeholder) and what
    /// prefill ran (the picture's or the clip's whole row run).
    fn media_tokens(&self) -> usize {
        self.prompt_len.saturating_sub(self.text_prompt_len)
    }

    /// The response's `ocr` extension: the resolution echo, plus grounded
    /// `regions` parsed from the finished output (choice 0) when armed. The
    /// parse runs on a decode that keeps special tokens - the markup rides on
    /// `<|ref|>`/`<|det|>` specials the content decode may strip.
    fn ocr_json(&self, first_choice_ids: &[u32]) -> Option<serde_json::Value> {
        let o = self.ocr.as_ref()?;
        let mut echo = o.echo();
        if let Ok(raw) = self.tokenizer.decode(first_choice_ids, false)
            && let Some(regions) = crate::deepseek_ocr::regions_json(&raw)
        {
            echo["regions"] = regions;
        }
        Some(echo)
    }

    fn parse(&self, ids: &[u32]) -> Parsed {
        // default keeps specials for the dialect parsers (their markup rides
        // on them); an explicit request flag wins - the paddleocr client
        // sends true for five tasks, false for Spotting's <|LOC_*|> markers
        let raw = self
            .tokenizer
            .decode(ids, self.skip_special.unwrap_or(false))
            .unwrap_or_default();
        self.parse_raw(&raw)
    }

    /// The parse half of [`Self::parse`], for callers that already hold the
    /// decoded text - the streaming loop decodes incrementally
    /// (StreamDecoder) instead of re-decoding the whole id run per token.
    fn parse_raw(&self, raw: &str) -> Parsed {
        let mut parsed = parse(self.dialect, raw, self.thinking_open, self.hints.as_ref());
        // parallel_tool_calls=false: we cannot force the model to emit one
        // call, but we keep the response coherent by dropping the extras (the
        // client re-sends history with one call + one result; the model can
        // re-issue the rest next turn)
        if self.single_tool_call && parsed.tool_calls.len() > 1 {
            parsed.tool_calls.truncate(1);
        }
        parsed
    }

    /// OpenAI logprobs object over the given (token, logprobs) run. Entries
    /// cover all sampled tokens of the turn - reasoning and tool-syntax
    /// included - since text-level parsing loses token/content alignment
    /// (llama.cpp-server behavior; documented deviation).
    fn logprobs_json(&self, ids: &[u32], lps: &[TokenLogprobs]) -> serde_json::Value {
        let entry = |id: u32, chosen: f32, top: &[(u32, f32)]| {
            let tok = self.tokenizer.decode(&[id], false).unwrap_or_default();
            serde_json::json!({
                "token": tok,
                "logprob": chosen,
                "bytes": tok.as_bytes(),
                "top_logprobs": top.iter().map(|&(tid, l)| {
                    let ts = self.tokenizer.decode(&[tid], false).unwrap_or_default();
                    serde_json::json!({"token": ts, "logprob": l, "bytes": ts.as_bytes()})
                }).collect::<Vec<_>>(),
            })
        };
        let content: Vec<serde_json::Value> = ids
            .iter()
            .zip(lps)
            .map(|(&id, lp)| entry(id, lp.chosen, &lp.top))
            .collect();
        serde_json::json!({"content": content, "refusal": null})
    }
}

fn to_wire_calls(parsed: &Parsed) -> Vec<ToolCall> {
    parsed
        .tool_calls
        .iter()
        .map(|tc| ToolCall {
            id: format!("call_{}", uuid::Uuid::new_v4().simple()),
            kind: "function",
            function: FunctionCall {
                name: tc.name.clone(),
                arguments: tc.arguments.clone(),
            },
        })
        .collect()
}

/// The `file_metadata` local extension, shared by all three surfaces:
/// document metadata (PDF Title/Author/dates) rides into the prompt with the
/// extracted content by default ("full"); "off" drops the block. Anything
/// else is a loud 400 - never accepted-and-ignored.
pub(crate) fn file_metadata_on(v: Option<&str>) -> Result<bool, String> {
    match v {
        None | Some("full") => Ok(true),
        Some("off") => Ok(false),
        Some(other) => Err(format!(
            "file_metadata must be \"full\" (default) or \"off\", got {other:?}"
        )),
    }
}

/// How PDFs reach the model - the `pdf_mode` local extension.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum PdfMode {
    /// pdfium page images (needs a vision model + the pdfium library).
    Render,
    /// sift text extraction (works on any model).
    Text,
}

/// The attachment local extensions of one request, validated together -
/// `file_metadata` / `max_pages` / `pdf_mode` - shared by all three surfaces
/// and count_tokens (which honors them so its number matches generation).
#[derive(Clone, Copy, Debug)]
pub(crate) struct AttachOpts {
    pub with_meta: bool,
    /// Caller's page cap for multi-page attachments; the server's own
    /// rendering limit still applies on top (min of the two).
    pub max_pages: Option<usize>,
    /// None = auto: render where the model can see, text otherwise.
    pub pdf_mode: Option<PdfMode>,
    /// Per-request forensics override: `None` = follow the endpoint's
    /// `[forensics] auto` default; `Some(true)` = run over every image/PDF this
    /// turn regardless of `auto`; `Some(false)` = suppress regardless. Never
    /// forces forensics on when `[forensics]` is disabled - availability is the
    /// endpoint's decision, this only steers a configured runtime per turn.
    pub forensics: Option<bool>,
}

pub(crate) fn attach_opts(
    file_metadata: Option<&str>,
    max_pages: Option<u32>,
    pdf_mode: Option<&str>,
    forensics: Option<&str>,
) -> Result<AttachOpts, String> {
    let with_meta = file_metadata_on(file_metadata)?;
    if max_pages == Some(0) {
        return Err("max_pages must be at least 1".into());
    }
    let pdf_mode = match pdf_mode {
        None => None,
        Some("render") => Some(PdfMode::Render),
        Some("text") => Some(PdfMode::Text),
        Some(other) => {
            return Err(format!(
                "pdf_mode must be \"render\" or \"text\", got {other:?}"
            ));
        }
    };
    let forensics = match forensics {
        None => None,
        Some("on") => Some(true),
        Some("off") => Some(false),
        Some(other) => {
            return Err(format!(
                "forensics must be \"on\" or \"off\", got {other:?}"
            ));
        }
    };
    Ok(AttachOpts {
        with_meta,
        max_pages: max_pages.map(|n| n as usize),
        pdf_mode,
        forensics,
    })
}

/// Expand any PDF content parts in `messages` in-place, on a blocking thread.
///
/// Two routes, one seam: a vision model with pdfium present gets rendered
/// page-image parts (the tower reads the pages); every other combination -
/// no mmproj, or no pdfium library - gets the sift TEXT path (`crate::doc`),
/// because any model can read a PDF's text layer. PDFs are therefore never
/// refused for lack of vision anymore; the one honest refusal left is a
/// text-less scanned PDF on the text route.
///
/// `Ok(())` when there were no PDFs or expansion succeeded; `Err((status,
/// message))` on decode / extraction / task error - the caller wraps it in its
/// own error envelope (OpenAI vs Anthropic). Shared by all three surfaces
/// (each reduces to a `Vec` of objects carrying a `content` array).
/// A file-shaped part (`file`/`input_file`/`document`) that survives every
/// extraction lane carries no inline bytes we can read - a `file_id` / URL
/// reference, or a malformed shape. Name it and refuse honestly; before this
/// guard the request died deep in the chat template with "Unexpected item
/// type in content. (in chat:33)" - a leak, not an
/// answer. (Unreadable-content refusals - binary data, encrypted PDFs - come
/// from the lanes themselves, which see the decoded bytes.)
fn unsupported_file_part(messages: &[serde_json::Value]) -> Option<String> {
    use serde_json::Value;
    for m in messages {
        let Some(parts) = m.get("content").and_then(Value::as_array) else {
            continue;
        };
        for p in parts {
            let t = p.get("type").and_then(Value::as_str).unwrap_or("");
            if matches!(t, "file" | "input_file" | "document") {
                let name = p
                    .get("file")
                    .and_then(|f| f.get("filename"))
                    .and_then(Value::as_str)
                    .or_else(|| p.get("filename").and_then(Value::as_str))
                    .or_else(|| p.get("title").and_then(Value::as_str))
                    .unwrap_or("unnamed file");
                let by_id = p
                    .get("file")
                    .and_then(|f| f.get("file_id"))
                    .or_else(|| p.get("file_id"))
                    .is_some();
                if by_id {
                    return Some(format!(
                        "file attachment {name:?} references a file_id - this server has no \
                         file storage; send the bytes inline as base64 `file_data`"
                    ));
                }
                if p.get("source")
                    .and_then(|s| s.get("type"))
                    .and_then(Value::as_str)
                    == Some("url")
                {
                    return Some(format!(
                        "file attachment {name:?} references a URL - this server does not \
                         fetch documents; send the bytes inline as a base64 `source`"
                    ));
                }
                return Some(format!(
                    "could not read file attachment {name:?} - expected inline base64 bytes \
                     (`file_data`, or a base64 `source` on the Anthropic surface)"
                ));
            }
        }
    }
    None
}

/// Expand attachment content parts in `messages` in-place, on a blocking
/// thread - the one seam every surface's attachments go through.
///
/// Photos first: with `file_metadata` on, every caller-sent image part gains
/// a `[Photo: taken ..., camera, GPS ...]` line read from the ORIGINAL bytes
/// (sift), before PDF expansion so rendered page images are never scanned.
/// Word documents and spreadsheets become text on every route (vision or
/// not). PDFs take one of two routes: a vision model with pdfium present
/// gets rendered page images; every other combination gets the sift TEXT
/// path. Whatever file-shaped part remains goes through the text-native
/// catch-all - inlined if it decodes as text, an honest refusal if binary.
pub(crate) async fn expand_attachments(
    state: &Arc<AppState>,
    model: &ServingModel,
    messages: &mut Vec<serde_json::Value>,
    opts: AttachOpts,
    // Context-enrichment output items (prebuilt JSON) ride out here - full file
    // metadata for every image/PDF (always-on) plus forensic reports when
    // forensics is on. Only the `/v1/responses` callers consume them (they become
    // `file_metadata` / `forensics` output items); `/v1/chat`, `/v1/messages`
    // and count_tokens pass a throwaway. Cleared then filled, so a caller can
    // reuse a buffer.
    enrichment_out: &mut Vec<serde_json::Value>,
) -> Result<Option<String>, (StatusCode, String)> {
    enrichment_out.clear();
    let with_meta = opts.with_meta;
    // deepseek2-ocr class: fixed prompt vocabulary. Injected framing text
    // ([Attached ...], [page k], [Photo: ...]) is off-vocabulary conditioning for
    // a document parser AND suppresses the derived canonical task string
    // (deepseek_ocr::resolve derives only on an empty body) - so the
    // multi-page lanes emit bare page images and the photo-meta pass is
    // skipped. See pdf::expand_in_messages for the ceiling-clip error this
    // route takes instead of the in-prompt disclosure note.
    // All document parsers, not just deepseek - paddle is task-prompt-
    // conditioned too, and framing text is off-vocabulary there as well.
    let plain_pages = model.document_parser;
    let has_pdfs = crate::pdf::has_pdf_parts(messages);
    let has_docx = crate::doc::has_docx_parts(messages);
    let has_sheets = crate::doc::has_sheet_parts(messages);
    let has_textfiles = crate::doc::has_textfile_parts(messages);
    let want_photo_meta = with_meta && !plain_pages && crate::doc::has_image_parts(messages);
    // Forensic preprocessing ([forensics] gate): images only, and - like the
    // photo-meta pass - never on a document-parser model, where injected framing
    // text is off-vocabulary. Cloned Arc rides into the blocking closure below.
    let forensics = state.forensics.clone();
    // Forensics is VLM-coupled: the findings are injected to make the VISION
    // model examine the pixels (confirm/contradict, read the sum/VAT). Without a
    // vision tower the directive is inert AND a raw image is refused upstream
    // (chat.rs `image_refs`/`supports_vision`), so forensics there is wasted
    // compute - never run it.
    let want_forensics = model.supports_vision
        && !plain_pages
        && forensics.as_ref().is_some_and(|r| {
            let has_image = crate::doc::has_image_parts(messages);
            let has_pdf = crate::pdf::has_pdf_parts(messages);
            match opts.forensics {
                // Explicit per-request "on": run over anything analyzable this
                // turn, even where `auto` alone would have skipped it.
                Some(true) => has_image || has_pdf,
                // Explicit "off": the caller opted this turn out.
                Some(false) => false,
                // No override: follow the endpoint's configured auto scope.
                None => (r.auto_images() && has_image) || (r.auto_pdfs() && has_pdf),
            }
        });
    // Full file metadata (paddock_filemeta) ships as a durable output item for
    // every image/PDF attachment - always-on context enrichment (the model-level
    // Intelligence section can't turn it off; a per-request toggle can). Cheap
    // (~20 ms) and read-only. Image metadata is skipped on a non-vision model
    // only because the image itself is refused there - a PDF's Info still ships.
    let want_filemeta = !plain_pages
        && ((model.supports_vision && crate::doc::has_image_parts(messages))
            || crate::pdf::has_pdf_parts(messages));
    // multi-page TIFF = a scanned document: on a vision model each page
    // becomes an image part (crate::tiffdoc). On a text-only model the image
    // part is already refused loudly downstream, so nothing is silently lost
    // by skipping the lane there.
    let has_tiffs = model.supports_vision && crate::tiffdoc::has_tiff_parts(messages);
    if !has_pdfs
        && !has_docx
        && !has_sheets
        && !has_textfiles
        && !want_photo_meta
        && !want_forensics
        && !want_filemeta
        && !has_tiffs
    {
        // nothing to expand - but a file-shaped part without inline data
        // (file_id / URL reference) still needs the honest refusal, not a
        // template error
        if let Some(msg) = unsupported_file_part(messages) {
            return Err((StatusCode::BAD_REQUEST, msg));
        }
        return Ok(None);
    }
    // PDF route: resolved per PART inside the walk (a flat pdf_mode/max_pages
    // on the part wins, then the request-level extension, then auto - each
    // file carries its own settings). The request-level "render"-impossible
    // case still 400s up front with the useful why; a part-level "render" the
    // server can't do errors from the walk naming the file. Never a silent
    // downgrade to text.
    let can_render = model.supports_vision && crate::pdf::available(&state.pdf);
    let no_render_why = if model.supports_vision {
        "PDF rendering (pdfium) is not available on this server"
    } else {
        "the loaded model has no vision tower"
    };
    if matches!(opts.pdf_mode, Some(PdfMode::Render)) && has_pdfs && !can_render {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "pdf_mode \"render\" is not possible here - {no_render_why}; use \"text\" \
                 (extracted text) or omit pdf_mode"
            ),
        ));
    }
    let cfg = state.pdf.clone();
    let req_max = opts.max_pages;
    let req_mode = opts.pdf_mode;
    let max_ctx = state.max_ctx;
    let taken = std::mem::take(messages);
    let joined = tokio::task::spawn_blocking(move || {
        let mut msgs = taken;
        let (photos, map_sample) = if want_photo_meta {
            let found = crate::doc::inject_image_meta(&mut msgs);
            (found.injected, found.map_sample)
        } else {
            (0, None)
        };
        // Context enrichment over the ORIGINAL image/PDF bytes, right after the
        // photo-meta pass (before any lane that replaces parts) so it reads the
        // untouched upload. All items ride out for the Responses output-item
        // surface (a persister stores them); forensics additionally injects its
        // findings as a text note per image so the vision model sees them.
        let mut enrichment_items: Vec<serde_json::Value> = Vec::new();
        // Full metadata first, on the pristine parts (read-only, no mutation).
        if want_filemeta {
            enrichment_items.extend(crate::doc::collect_file_metadata(&msgs));
        }
        if want_forensics && let Some(rt) = forensics.as_deref() {
            let fitems = crate::doc::inject_forensics(&mut msgs, rt);
            if !fitems.is_empty() {
                tracing::debug!(images = fitems.len(), "forensic preprocessing injected");
            }
            enrichment_items.extend(fitems.iter().map(|it| it.output_item()));
        }
        if has_docx {
            // Word docs are text on every route, vision or not (scriptor
            // resolves tracked changes to the final view)
            crate::doc::expand_docx_in_messages(&mut msgs, with_meta, max_ctx)?;
        }
        if has_sheets {
            crate::doc::expand_sheets_in_messages(&mut msgs, with_meta, max_ctx)?;
        }
        let mut summary = crate::pdf::PdfSummary::default();
        if has_tiffs {
            // after the photo pass (the [Photo: ...] line reads the ORIGINAL
            // TIFF's EXIF; the replacement page images are never scanned) -
            // same page cap as the PDF raster route, same cost class
            crate::tiffdoc::expand_in_messages(
                &mut msgs,
                cfg.max_pages,
                req_max,
                &mut summary,
                plain_pages,
            )?;
        }
        if has_pdfs {
            let pdfs;
            (msgs, pdfs) = crate::pdf::expand_in_messages(
                msgs,
                &cfg,
                with_meta,
                max_ctx,
                req_max,
                req_mode,
                can_render,
                no_render_why,
                plain_pages,
            )?;
            summary.absorb(&pdfs);
        }
        if has_textfiles {
            // last of the lanes: only leftovers reach the catch-all
            crate::doc::expand_textfiles_in_messages(&mut msgs, max_ctx)?;
        }
        Ok::<_, String>((msgs, summary, photos, map_sample, enrichment_items))
    })
    .await;
    match joined {
        Ok(Ok((expanded, summary, photos, map_sample, enrichment_items))) => {
            if summary.any() || photos > 0 {
                tracing::debug!(
                    pdfs = summary.pdfs,
                    tiffs = summary.tiffs,
                    pages = summary.rendered_pages,
                    truncated = summary.truncated,
                    photo_meta = photos,
                    can_render,
                    "expanded attachments"
                );
            }
            *messages = expanded;
            *enrichment_out = enrichment_items;
            // PDFs are gone now; anything file-shaped left is another format
            if let Some(msg) = unsupported_file_part(messages) {
                return Err((StatusCode::BAD_REQUEST, msg));
            }
            // The map capability rides out of here rather than being applied
            // in here: this function is handed a different array on every
            // dialect - chat messages here, Responses input items there - and
            // only the caller knows where its system turn actually is.
            Ok(map_sample)
        }
        Ok(Err(e)) => Err((StatusCode::BAD_REQUEST, e)),
        Err(_) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            "attachment expansion task panicked".to_owned(),
        )),
    }
}

/// `POST /v1/chat/completions`.
pub async fn handle(
    State(state): State<Arc<AppState>>,
    scope: Option<axum::Extension<crate::events::EventScope>>,
    crate::extract::OaiJson(mut req): crate::extract::OaiJson<ChatCompletionRequest>,
) -> Response {
    let scope = scope.map(|e| e.0).unwrap_or_default();
    // Normalise the deprecated tool spelling at the very edge, before the
    // model-availability check. It reads nothing but the request, and a
    // malformed request is malformed whether or not a model happens to be
    // loaded - answering 503 there would hide a 400 the caller must fix.
    if let Err(e) = adopt_legacy_functions(&mut req) {
        return err(StatusCode::BAD_REQUEST, "invalid_request_error", e);
    }
    if let Err(e) = refuse_unserved_options(&req) {
        return err(StatusCode::BAD_REQUEST, "invalid_request_error", e);
    }
    let Some(model) = state.serving.as_ref() else {
        return err(
            StatusCode::SERVICE_UNAVAILABLE,
            "model_not_loaded",
            "no model is loaded",
        );
    };
    scope.model(&model.id);
    scope.user(req.user.as_deref());
    // PDF attachments -> page images (vision+pdfium) or extracted text (sift),
    // blocking work off the async thread; the expanded parts flow through the
    // normal paths below.
    let opts = match attach_opts(
        req.file_metadata.as_deref(),
        req.max_pages,
        req.pdf_mode.as_deref(),
        req.forensics.as_deref(),
    ) {
        Ok(o) => o,
        Err(e) => return err(StatusCode::BAD_REQUEST, "invalid_request_error", e),
    };
    // /v1/chat/completions: forensics stays injection-only (spec-strict, no
    // extra output item), so the structured items are discarded here.
    match expand_attachments(&state, model, &mut req.messages, opts, &mut Vec::new()).await {
        Ok(Some(sample)) => crate::doc::add_map_capability(&mut req.messages, &sample),
        Ok(None) => {}
        Err((code, msg)) => {
            let kind = if code == StatusCode::BAD_REQUEST {
                "invalid_request_error"
            } else {
                "internal_error"
            };
            return err(code, kind, msg);
        }
    }
    let t_prep = std::time::Instant::now();
    let prepared = match prepare(model, &req, state.max_output_ceiling, &state.sampling) {
        Ok(p) => p,
        Err(e) => return err(StatusCode::BAD_REQUEST, "invalid_request_error", e),
    };
    scope.tokenized(t_prep.elapsed());
    // multimodal prompts feed the engine TEXT tokens only (the pad slot is
    // replaced by the image rows engine-side); history/penalties see text
    let prompt: Vec<u32> = prepared.engine_prompt.clone();
    let meta = Meta {
        id: format!("chatcmpl-{}", uuid::Uuid::new_v4().simple()),
        model_id: model.id.clone(),
        tokenizer: model.tokenizer.clone(),
        prompt_len: prepared.prompt_ids.len(),
        text_prompt_len: prompt.len(),
        dialect: model.dialect,
        thinking_open: prepared.thinking_open,
        stop_strings: prepared.stop_strings.clone(),
        single_tool_call: prepared.single_tool_call,
        hints: prepared.hints.clone(),
        logprobs: prepared.logprobs.is_some(),
        legacy_functions: prepared.legacy_functions,
        media_is_audio: model.supports_audio,
        ocr: prepared.ocr.clone(),
        skip_special: prepared.skip_special,
        scope,
    };

    // Reject an over-window prompt at the edge: a clean 400 for both stream and
    // non-stream (the engine also rejects on admit, but a streaming request has
    // committed its 200 SSE status by then). Prices image rows too - what
    // prefill will actually see, not just the text stream.
    if let Some(e) = context_gate(
        model,
        prompt.len(),
        prepared.mm_chunks.as_deref(),
        state.max_ctx,
    ) {
        return engine_err(&e);
    }

    // n choices = n engine sequences; per-choice seed offsets keep sampled
    // choices independent (greedy choices are identical, as OpenAI's are)
    let mut rxs = Vec::with_capacity(prepared.n);
    for i in 0..prepared.n {
        let (tx, rx) = unbounded_channel();
        let mut sampler = prepared.sampler.clone();
        sampler.seed = sampler.seed.wrapping_add(i as u64);
        let gen_req = GenRequest {
            prompt: prompt.clone(),
            max_tokens: prepared.max_tokens,
            sampler,
            stop_tokens: prepared.stop_tokens.clone(),
            events: tx,
            mm_chunks: prepared.mm_chunks.clone(),
            constraint: prepared.make_constraint(model),
            logprobs: prepared.logprobs,
            submitted: None, // stamped by Engine::submit
        };
        if let Err(e) = model.engine.submit(gen_req) {
            return err(StatusCode::INTERNAL_SERVER_ERROR, "internal_error", e);
        }
        rxs.push(rx);
    }

    if req.stream {
        stream_response(meta, rxs)
    } else {
        collect_response(meta, rxs).await
    }
}

async fn collect_response(mut meta: Meta, rxs: Vec<UnboundedReceiver<TokenEvent>>) -> Response {
    let mut choices = Vec::with_capacity(rxs.len());
    let mut completion_tokens = 0usize;
    let mut cached = 0usize;
    // choice 0's raw ids, kept for the OCR grounding-region parse
    let mut first_ids: Vec<u32> = Vec::new();
    for (index, mut rx) in rxs.into_iter().enumerate() {
        let mut ids = Vec::new();
        let mut lps: Vec<TokenLogprobs> = Vec::new();
        let mut reason = FinishReason::Length;
        while let Some(ev) = rx.recv().await {
            match ev {
                // `rows` is what the engine actually prefilled - on an image
                // request that is the picture's expanded row run, not the one
                // `<image>` token the prompt was tokenized to. Raise rather
                // than assign so a backend that reports nothing keeps the
                // tokenized length.
                TokenEvent::Prefilled { cached: c, rows } => {
                    cached = cached.max(c as usize);
                    meta.prompt_len = meta.prompt_len.max(rows as usize);
                }
                TokenEvent::Token { id, logprobs } => {
                    ids.push(id);
                    if let Some(lp) = logprobs {
                        lps.push(lp);
                    }
                }
                TokenEvent::Done(r, stats) => {
                    reason = r;
                    meta.scope.phases(&stats);
                    break;
                }
                TokenEvent::Error(e) => {
                    return engine_err(&e);
                }
            }
        }

        let mut parsed = meta.parse(&ids);
        // stop strings cut visible content only (reasoning/tool calls unaffected)
        let mut stop_hit = false;
        if let Some(content) = parsed.content.take() {
            let (trunc, hit) = apply_stop_strings(&content, &meta.stop_strings);
            stop_hit = hit;
            parsed.content = (!trunc.is_empty()).then_some(trunc);
        }

        let finish = if !parsed.tool_calls.is_empty() {
            if meta.legacy_functions {
                "function_call"
            } else {
                "tool_calls"
            }
        } else if stop_hit {
            "stop"
        } else {
            reason.as_str()
        };

        let tool_calls = to_wire_calls(&parsed);
        let message = ChatMessage::assistant(parsed.content, parsed.reasoning, tool_calls);
        // A legacy `functions` request cannot read `tool_calls` - it looks at
        // `message.function_call` - so answering in the modern shape would be
        // a 200 the client silently drops.
        let message = if meta.legacy_functions {
            message.into_legacy()
        } else {
            message
        };
        completion_tokens += ids.len();
        if index == 0 && meta.ocr.is_some() {
            first_ids = ids.clone();
        }
        meta.scope.finish(finish);
        choices.push(ChatChoice {
            index: index as u32,
            message,
            logprobs: meta.logprobs.then(|| meta.logprobs_json(&ids, &lps)),
            finish_reason: Some(finish.to_owned()),
        });
    }
    meta.scope.usage(meta.prompt_len, completion_tokens);
    meta.scope.cached(cached);

    // read before the struct literal moves the String fields out of `meta`
    let details = Usage::media_details(
        meta.prompt_len,
        cached,
        meta.media_tokens(),
        meta.media_is_audio,
    );
    let ocr = meta.ocr_json(&first_ids);
    Json(ChatCompletionResponse {
        id: meta.id,
        object: "chat.completion",
        created: now_secs(),
        model: meta.model_id,
        choices,
        usage: Usage {
            prompt_tokens: meta.prompt_len,
            completion_tokens,
            total_tokens: meta.prompt_len + completion_tokens,
            prompt_tokens_details: details,
        },
        ocr,
    })
    .into_response()
}

/// Largest prefix of `s` that is safe to emit: everything except a tail that
/// could be the start of a marker or stop string, rounded down to a char
/// boundary.
pub(crate) fn safe_emit_len(s: &str, dialect_markers: &[&str], stops: &[String]) -> usize {
    let mut markers: Vec<&str> = dialect_markers.to_vec();
    markers.extend(stops.iter().map(String::as_str));
    let mut n = s.len() - holdback(s, &markers);
    while n > 0 && !s.is_char_boundary(n) {
        n -= 1;
    }
    n
}

/// Per-choice streaming state for the merged n-way SSE loop.
struct ChoiceState {
    ids: Vec<u32>,
    /// incremental decode of `ids` - the per-token full re-decode was the
    /// O(n^2) long-stream collapse at high concurrency
    sd: paddock_tokenizer::StreamDecoder,
    lps: Vec<TokenLogprobs>,
    /// logprob entries already attached to an emitted chunk
    lp_flushed: usize,
    r_emitted: usize,
    c_emitted: usize,
    /// complete tool calls already streamed (atomic per-call deltas)
    calls_emitted: usize,
    stop_hit: bool,
    done: bool,
    finish_sent: bool,
    reason: FinishReason,
}

impl ChoiceState {
    fn new(sd: paddock_tokenizer::StreamDecoder) -> ChoiceState {
        ChoiceState {
            ids: Vec::new(),
            sd,
            lps: Vec::new(),
            lp_flushed: 0,
            r_emitted: 0,
            c_emitted: 0,
            calls_emitted: 0,
            stop_hit: false,
            done: false,
            finish_sent: false,
            reason: FinishReason::Length,
        }
    }
}

/// Streaming: `reasoning_content` / `content` deltas as they resolve, tool
/// calls streamed ATOMICALLY as each call's block completes (id + name +
/// full arguments in one delta - incremental argument fragments of a non-JSON
/// dialect are not prefix-stable), n choices merged into one SSE with per-
/// chunk `index`, optional terminal usage chunk (stream_options).
fn stream_response(mut meta: Meta, rxs: Vec<UnboundedReceiver<TokenEvent>>) -> Response {
    use futures::StreamExt as _;
    let created = now_secs();
    let n = rxs.len();
    let sse = stream! {
        // role announcement chunks (OpenAI sends role first, per choice)
        for i in 0..n {
            yield sse_data(&chunk(&meta, created, i, serde_json::json!({"role":"assistant"}), None, None));
        }

        let mut states: Vec<ChoiceState> = (0..n)
            .map(|_| ChoiceState::new(meta.tokenizer.stream_decoder(meta.skip_special.unwrap_or(false))))
            .collect();
        let mut merged = futures::stream::select_all(rxs.into_iter().enumerate().map(|(i, rx)| {
            futures::stream::unfold(rx, |mut rx| async move {
                rx.recv().await.map(|ev| (ev, rx))
            })
            .map(move |ev| (i, ev))
            .boxed()
        }));
        let mut open = n;
        let mut cached = 0usize;

        while open > 0 {
            let Some((i, ev)) = merged.next().await else { break };
            let cs = &mut states[i];
            if cs.done {
                continue; // stop-string hit earlier; drain silently
            }
            match ev {
                // `rows` is what the engine actually prefilled - on an image
                // request that is the picture's expanded row run, not the one
                // `<image>` token the prompt was tokenized to. Raise rather
                // than assign so a backend that reports nothing keeps the
                // tokenized length.
                TokenEvent::Prefilled { cached: c, rows } => {
                    cached = cached.max(c as usize);
                    meta.prompt_len = meta.prompt_len.max(rows as usize);
                }
                TokenEvent::Token { id, logprobs } => {
                    cs.ids.push(id);
                    if let Some(lp) = logprobs {
                        cs.lps.push(lp);
                    }
                    let raw = cs.sd.push(&meta.tokenizer, id);
                    let parsed = meta.parse_raw(&raw);

                    if let Some(r) = &parsed.reasoning {
                        let safe = safe_emit_len(r, meta.dialect.reasoning_markers(), &[]);
                        // same re-decode boundary hazard as the content path
                        while cs.r_emitted < r.len() && !r.is_char_boundary(cs.r_emitted) {
                            cs.r_emitted += 1;
                        }
                        if safe > cs.r_emitted {
                            let delta = r[cs.r_emitted..safe].to_owned();
                            cs.r_emitted = safe;
                            let lp = flush_lps(&meta, cs);
                            yield sse_data(&chunk(
                                &meta, created, i,
                                serde_json::json!({"reasoning_content": delta}),
                                None, lp,
                            ));
                        }
                    }
                    // tool calls stream as soon as their block CLOSES
                    let complete = parsed.complete_calls.min(parsed.tool_calls.len());
                    while cs.calls_emitted < complete {
                        let k = cs.calls_emitted;
                        let tc = &parsed.tool_calls[k];
                        let call = serde_json::json!([{
                            "index": k,
                            "id": format!("call_{}", uuid::Uuid::new_v4().simple()),
                            "type": "function",
                            "function": {"name": tc.name, "arguments": tc.arguments},
                        }]);
                        cs.calls_emitted += 1;
                        let lp = flush_lps(&meta, cs);
                        yield sse_data(&chunk(
                            &meta, created, i,
                            serde_json::json!({"tool_calls": call}),
                            None, lp,
                        ));
                    }
                    if let Some(c) = &parsed.content {
                        let (trunc, hit) = apply_stop_strings(c, &meta.stop_strings);
                        let mut safe = if hit {
                            trunc.len()
                        } else {
                            safe_emit_len(c, meta.dialect.content_markers(), &meta.stop_strings)
                        };
                        // clamp to a char boundary: safe_emit_len counts
                        // bytes and can land inside a multi-byte scalar
                        // (panicked on a sampled U+12000-block char)
                        while safe > 0 && !c.is_char_boundary(safe) {
                            safe -= 1;
                        }
                        // c is RE-DECODED each tick: a byte-fallback token
                        // completing a scalar can shift bytes so an old
                        // boundary lands mid-char - skip forward instead of
                        // panicking (worst case: one mangled char once)
                        while cs.c_emitted < c.len() && !c.is_char_boundary(cs.c_emitted) {
                            cs.c_emitted += 1;
                        }
                        if safe > cs.c_emitted {
                            let delta = c[cs.c_emitted..safe].to_owned();
                            cs.c_emitted = safe;
                            let lp = flush_lps(&meta, cs);
                            yield sse_data(&chunk(
                                &meta, created, i,
                                serde_json::json!({"content": delta}),
                                None, lp,
                            ));
                        }
                        if hit {
                            cs.stop_hit = true;
                            cs.done = true;
                            cs.reason = FinishReason::Stop;
                            open -= 1;
                            yield sse_data(&chunk(&meta, created, i, serde_json::json!({}), Some("stop"), None));
                            cs.finish_sent = true;
                            // n == 1: dropping the whole merged stream stops
                            // the engine slot; with n > 1 the stopped choice
                            // drains silently until its Done
                            if n == 1 {
                                break;
                            }
                        }
                    }
                }
                TokenEvent::Done(r, stats) => {
                    cs.reason = r;
                    cs.done = true;
                    open -= 1;
                    meta.scope.phases(&stats);
                }
                TokenEvent::Error(e) => {
                    // An engine failure used to close the stream as a clean
                    // `finish_reason: "stop"` with nothing logged, so a tick
                    // that died mid-generation reached the client as a short
                    // but valid answer - and a benchmark counted it as
                    // throughput. A whole lane once read as slow-but-working
                    // for hours that way: every sequence was erroring
                    // after ~1 token and being reported as a successful stop.
                    // Say so, loudly, and put it in the stream.
                    tracing::warn!(choice = i, error = %e, "chat stream: engine error mid-generation");
                    if !cs.finish_sent {
                        yield sse_data(&serde_json::json!({
                            "error": {"message": e.to_string(), "type": "engine_error"}
                        }).to_string());
                    }
                    cs.reason = FinishReason::Stop;
                    cs.done = true;
                    open -= 1;
                }
            }
        }

        // terminal per-choice: any tool calls whose block never closed
        // (finish at length / stop-token) + the finish chunk
        let mut completion_tokens = 0usize;
        for (i, cs) in states.iter_mut().enumerate() {
            completion_tokens += cs.ids.len();
            if cs.finish_sent {
                meta.scope.finish("stop"); // early stop-string hit
                continue;
            }
            let parsed = meta.parse(&cs.ids);
            // Flush what the marker/stop-string holdback kept mid-stream: at
            // end of turn an unresolved tail is plain text, and the
            // non-streaming path already returns it - streaming must not
            // disagree (it silently lost the last few tokens of any reply
            // whose tail prefix-matched a marker or stop string).
            if let Some(r) = &parsed.reasoning {
                while cs.r_emitted < r.len() && !r.is_char_boundary(cs.r_emitted) {
                    cs.r_emitted += 1;
                }
                if r.len() > cs.r_emitted {
                    let delta = r[cs.r_emitted..].to_owned();
                    cs.r_emitted = r.len();
                    let lp = flush_lps(&meta, cs);
                    yield sse_data(&chunk(
                        &meta, created, i,
                        serde_json::json!({"reasoning_content": delta}),
                        None, lp,
                    ));
                }
            }
            if let Some(c) = &parsed.content {
                // a stop string the holdback was still disambiguating can
                // complete exactly at the cut - it still truncates and stops
                let (trunc, hit) = apply_stop_strings(c, &meta.stop_strings);
                if hit {
                    cs.stop_hit = true;
                }
                while cs.c_emitted < trunc.len() && !trunc.is_char_boundary(cs.c_emitted) {
                    cs.c_emitted += 1;
                }
                if trunc.len() > cs.c_emitted {
                    let delta = trunc[cs.c_emitted..].to_owned();
                    cs.c_emitted = trunc.len();
                    let lp = flush_lps(&meta, cs);
                    yield sse_data(&chunk(
                        &meta, created, i,
                        serde_json::json!({"content": delta}),
                        None, lp,
                    ));
                }
            }
            let finish = if !parsed.tool_calls.is_empty() && !cs.stop_hit {
                if meta.legacy_functions { "function_call" } else { "tool_calls" }
            } else if cs.stop_hit {
                "stop"
            } else {
                cs.reason.as_str()
            };
            if !cs.stop_hit && parsed.tool_calls.len() > cs.calls_emitted {
                let calls: Vec<serde_json::Value> = parsed.tool_calls[cs.calls_emitted..]
                    .iter()
                    .enumerate()
                    .map(|(off, tc)| {
                        serde_json::json!({
                            "index": cs.calls_emitted + off,
                            "id": format!("call_{}", uuid::Uuid::new_v4().simple()),
                            "type": "function",
                            "function": {"name": tc.name, "arguments": tc.arguments},
                        })
                    })
                    .collect();
                let lp = flush_lps(&meta, cs);
                // Legacy clients read `delta.function_call`, and their protocol
                // has room for exactly one call - which is safe here because
                // adopt_legacy_functions pinned parallel_tool_calls off, so
                // `calls` is never longer than 1 on this branch.
                let delta = if meta.legacy_functions {
                    match calls.first() {
                        Some(c) => serde_json::json!({"function_call": c["function"]}),
                        None => serde_json::json!({}),
                    }
                } else {
                    serde_json::json!({"tool_calls": calls})
                };
                yield sse_data(&chunk(&meta, created, i, delta, None, lp));
            }
            meta.scope.finish(finish);
            yield sse_data(&chunk(&meta, created, i, serde_json::json!({}), Some(finish), None));
        }
        meta.scope.usage(meta.prompt_len, completion_tokens);
        meta.scope.cached(cached);

        // Terminal usage chunk, empty choices. Always emitted,
        // not just under stream_options.include_usage: benchmark clients
        // prefer server-reported counts and fall
        // back to client-side re-tokenization of VISIBLE text without them -
        // which drops reasoning_content deltas and undercounts badly under
        // BPE round-trip. llama.cpp
        // and most OAI-compatible servers volunteer this chunk; the OpenAI
        // SDKs parse it fine (chunk.usage is optional in the schema).
        {
            let mut usage = serde_json::json!({
                "id": meta.id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": meta.model_id,
                "choices": [],
                "usage": {
                    "prompt_tokens": meta.prompt_len,
                    "completion_tokens": completion_tokens,
                    "total_tokens": meta.prompt_len + completion_tokens,
                    "prompt_tokens_details":
                        Usage::media_details(meta.prompt_len, cached, meta.media_tokens(), meta.media_is_audio),
                },
            });
            // the OCR resolution echo (+ grounded regions) rides the terminal
            // chunk - the one place a streaming client gets whole-turn facts
            if let Some(o) = meta.ocr_json(states.first().map_or(&[], |s| &s.ids)) {
                usage["ocr"] = o;
            }
            yield sse_data(&usage.to_string());
        }
        yield Ok::<_, std::convert::Infallible>(Event::default().data("[DONE]"));
    };
    Sse::new(sse).into_response()
}

/// Logprob entries accumulated since the last emitted chunk for this choice.
fn flush_lps(meta: &Meta, cs: &mut ChoiceState) -> Option<serde_json::Value> {
    if !meta.logprobs || cs.lp_flushed >= cs.lps.len() {
        return None;
    }
    let ids = &cs.ids[cs.lp_flushed..cs.lps.len()];
    let lps = &cs.lps[cs.lp_flushed..];
    let out = meta.logprobs_json(ids, lps);
    cs.lp_flushed = cs.lps.len();
    Some(out)
}

fn sse_data(s: &str) -> Result<Event, std::convert::Infallible> {
    Ok(Event::default().data(s))
}

fn chunk(
    meta: &Meta,
    created: u64,
    index: usize,
    delta: serde_json::Value,
    finish: Option<&str>,
    logprobs: Option<serde_json::Value>,
) -> String {
    serde_json::json!({
        "id": meta.id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": meta.model_id,
        "choices": [{
            "index": index,
            "delta": delta,
            "finish_reason": finish,
            "logprobs": logprobs,
        }],
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A family whose tool-call syntax has no grammar refuses honestly, and the
    /// refusal says which reason applies - gemma4 is not "not done yet", it is
    /// blocked until its tool channels parse at all, and that distinction is
    /// the difference between "wait for a release" and "this cannot work".
    #[test]
    fn the_forced_tool_refusal_names_its_own_reason() {
        for (d, needle) in [
            (Dialect::GemmaChannel, "not parsed yet"),
            (Dialect::Harmony, "Harmony"),
            (Dialect::Plain, "no tool-call syntax"),
        ] {
            assert!(d.tool_syntax().is_none(), "{d:?} claims a grammar");
            let msg = no_forced_tool_grammar(d, "\"any\"");
            assert!(msg.contains(needle), "{d:?}: {msg:?}");
        }
        // and the four that can be forced
        assert!(Dialect::JsonToolCall.tool_syntax().is_some(), "granite");
        assert!(Dialect::Laguna.tool_syntax().is_some(), "laguna");
        assert!(Dialect::QwenXml.tool_syntax().is_some());
        assert!(Dialect::MuseChannel.tool_syntax().is_some(), "muse");
    }

    /// The two INTERPOLATING families use different template variables and
    /// different ladders, and getting either wrong is silent: the template
    /// renders an effort the model was never trained on, or drops a rung it
    /// has. Their templates validate nothing, so the cited table is the only
    /// place this can be checked - everyone else is measured
    /// (`crate::reasoning`).
    #[test]
    fn the_cited_ladders_stay_per_family() {
        let (kw, ladder) = Dialect::Harmony
            .effort_kwarg()
            .expect("gpt-oss grades effort");
        assert_eq!(kw, "reasoning_effort");
        assert_eq!(ladder, ["low", "medium", "high"]);

        // muse-glimmer's model card lists low/medium/high/xhigh, so `xhigh`
        // must reach the model rather than being flattened into `high`
        let (kw, ladder) = Dialect::MuseChannel
            .effort_kwarg()
            .expect("muse grades strength");
        assert_eq!(kw, "reasoning_strength");
        assert_eq!(ladder, ["low", "medium", "high", "xhigh"]);

        // the qwen dialect must not acquire a citation: 3.5, 3.6 and 3.8 all
        // parse as QwenXml and only 3.8 has rungs, so any answer here would be
        // wrong for two of the three
        assert!(Dialect::QwenXml.effort_kwarg().is_none());
    }

    #[test]
    fn the_seven_value_vocabulary_ranks_onto_a_three_rung_ladder() {
        let caps = crate::reasoning::ReasoningCaps {
            kwarg: Some("reasoning_effort"),
            levels: ["low", "medium", "high"]
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
            default_level: Some("medium".to_owned()),
            off: false,
            preserve: false,
            source: crate::reasoning::LadderSource::Card,
        };
        for (given, expect) in [
            ("none", "low"),
            ("minimal", "low"),
            ("low", "low"),
            ("medium", "medium"),
            ("high", "high"),
            ("xhigh", "high"),
            ("max", "high"),
        ] {
            let rank = reasoning_effort_rank(given).expect("valid");
            assert_eq!(caps.clamp(rank), Some(expect), "{given}");
        }
        assert!(reasoning_effort_rank("extreme").is_err());
    }

    fn caps(
        kwarg: Option<&'static str>,
        levels: &[&str],
        off: bool,
    ) -> crate::reasoning::ReasoningCaps {
        crate::reasoning::ReasoningCaps {
            kwarg,
            levels: levels.iter().map(|s| (*s).to_owned()).collect(),
            default_level: levels.last().map(|s| (*s).to_owned()),
            off,
            preserve: false,
            source: crate::reasoning::LadderSource::Template,
        }
    }

    /// Qwen3.8's shape, which nothing in the codebase could express before:
    /// three rungs AND an off position, on a dialect two other models share.
    #[test]
    fn a_ladder_with_an_off_position_uses_both() {
        let c = caps(Some("reasoning_effort"), &["low", "medium", "xhigh"], true);
        assert_eq!(c.style(), "effort");

        // a rung sets the rung, in the template's own spelling - `high` is the
        // model's alias for xhigh and must land there, not on a level it does
        // not have
        for (asked, rung) in [
            ("low", "low"),
            ("minimal", "low"),
            ("medium", "medium"),
            ("high", "xhigh"),
            ("xhigh", "xhigh"),
            ("max", "xhigh"),
        ] {
            let kw = merge_reasoning_effort(&c, asked, None)
                .expect("valid")
                .unwrap();
            assert_eq!(kw["reasoning_effort"], rung, "{asked}");
            assert!(
                kw.get("enable_thinking").is_none(),
                "{asked}: thinking is already on, saying so adds nothing"
            );
        }

        // `none` is the off request, not the bottom rung - and it sends no
        // effort, because grading a thought process the template is skipping
        // is noise the model would have to ignore
        let off = merge_reasoning_effort(&c, "none", None)
            .expect("valid")
            .unwrap();
        assert_eq!(off["enable_thinking"], false);
        assert!(off.get("reasoning_effort").is_none());

        assert!(merge_reasoning_effort(&c, "extreme", None).is_err());
    }

    /// The two older shapes must not move: this change is about 3.8 gaining a
    /// ladder, not about anything else changing what it sends.
    #[test]
    fn a_ladder_without_an_off_position_clamps_none_to_the_bottom() {
        // gpt-oss and muse render their reasoning preamble unconditionally
        let c = caps(Some("reasoning_effort"), &["low", "medium", "high"], false);
        let out = merge_reasoning_effort(&c, "none", None)
            .expect("valid")
            .unwrap();
        assert_eq!(out["reasoning_effort"], "low");
        assert!(out.get("enable_thinking").is_none());
    }

    #[test]
    fn a_switch_without_a_ladder_still_answers_on_off() {
        let c = caps(None, &[], true);
        assert_eq!(c.style(), "toggle");
        let on = merge_reasoning_effort(&c, "high", None)
            .expect("valid")
            .unwrap();
        assert_eq!(on["enable_thinking"], true);
        assert!(on.get("reasoning_effort").is_none());
        let off = merge_reasoning_effort(&c, "none", None)
            .expect("valid")
            .unwrap();
        assert_eq!(off["enable_thinking"], false);
        // the vocabulary is still checked where the value then collapses
        assert!(merge_reasoning_effort(&c, "extreme", None).is_err());
    }

    #[test]
    fn a_model_that_cannot_reason_refuses_instead_of_ignoring() {
        let c = caps(None, &[], false);
        assert!(merge_reasoning_effort(&c, "high", None).is_err());
    }

    #[test]
    fn an_explicit_template_kwarg_outranks_the_effort_field() {
        // chat_template_kwargs is the lower-level knob; a caller who set it
        // meant it, on both the ladder and the switch
        let c = caps(Some("reasoning_effort"), &["low", "medium", "xhigh"], true);
        let out = merge_reasoning_effort(
            &c,
            "low",
            Some(serde_json::json!({"reasoning_effort": "medium"})),
        )
        .expect("valid")
        .unwrap();
        assert_eq!(out["reasoning_effort"], "medium");

        let out = merge_reasoning_effort(
            &c,
            "none",
            Some(serde_json::json!({"enable_thinking": true})),
        )
        .expect("valid")
        .unwrap();
        assert_eq!(out["enable_thinking"], true);
    }

    /// 2x1 24-bit BMP (blue-ish, red-ish pixels), handwritten.
    fn tiny_bmp() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(b"BM");
        b.extend_from_slice(&(54u32 + 8).to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(&54u32.to_le_bytes());
        b.extend_from_slice(&40u32.to_le_bytes());
        b.extend_from_slice(&2i32.to_le_bytes());
        b.extend_from_slice(&1i32.to_le_bytes());
        b.extend_from_slice(&1u16.to_le_bytes());
        b.extend_from_slice(&24u16.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(&8u32.to_le_bytes());
        b.extend_from_slice(&2835u32.to_le_bytes());
        b.extend_from_slice(&2835u32.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(&[255, 0, 0, 0, 0, 255, 0, 0]); // BGR + row pad
        b
    }

    fn data_uri(bytes: &[u8]) -> String {
        use base64::Engine as _;
        format!(
            "data:image/bmp;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(bytes)
        )
    }

    #[test]
    fn extracts_and_decodes_data_uri_images() {
        let uri = data_uri(&tiny_bmp());
        let msgs = vec![serde_json::json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": {"url": uri}}
            ]
        })];
        let refs = find_images(&msgs).expect("find");
        assert_eq!(refs.len(), 1);
        let img = decode_image_url(&refs[0].url, None, refs[0].detail).expect("decode");
        assert_eq!((img.w, img.h), (2, 1));
        // BMP is BGR bottom-up; decoded RGB: pixel0 = (0,0,255), pixel1 = (255,0,0)
        assert_eq!(&img.rgb, &[0, 0, 255, 255, 0, 0]);
    }

    /// Every inline image shape the three APIs actually send. Anthropic's is
    /// the one that mattered: `/v1/messages` has no other way to carry an
    /// image, so before this it accepted the request shape and then failed
    /// with "image content part has no url" - the surface was advertised as
    /// vision-capable and could not take a picture.
    #[test]
    fn every_api_shape_yields_the_same_data_uri() {
        let uri = data_uri(&tiny_bmp());
        let b64 = uri.split(",").nth(1).expect("b64").to_owned();
        let msgs = vec![serde_json::json!({
            "role": "user",
            "content": [
                // OpenAI chat completions
                {"type": "image_url", "image_url": {"url": uri}},
                // Responses API: image_url is the URI itself
                {"type": "input_image", "image_url": uri},
                // Anthropic messages: base64 + media_type, no url anywhere
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/bmp", "data": b64}},
            ]
        })];
        let refs = find_images(&msgs).expect("find");
        assert_eq!(refs.len(), 3);
        for r in &refs {
            let img = decode_image_url(&r.url, None, r.detail).expect("decode");
            assert_eq!((img.w, img.h), (2, 1));
            assert_eq!(&img.rgb, &[0, 0, 255, 255, 0, 0]);
        }
    }

    /// A remote Anthropic source is refused by NAME, not as a missing url -
    /// same SSRF/availability stance `decode_image_url` takes, but the client
    /// gets told which knob it turned.
    #[test]
    fn anthropic_url_sources_are_refused_by_name() {
        let msgs = vec![serde_json::json!({
            "role": "user",
            "content": [{"type": "image", "source": {
                "type": "url", "url": "https://x.test/cat.png"}}]
        })];
        let err = find_images(&msgs).unwrap_err();
        assert!(err.contains("does not fetch remote"), "{err}");

        // and a part with no recognizable payload still says so plainly
        let msgs = vec![serde_json::json!({
            "role": "user", "content": [{"type": "image"}]
        })];
        assert!(find_images(&msgs).unwrap_err().contains("no url"));
    }

    #[test]
    fn remote_urls_and_videos_are_rejected() {
        assert!(
            decode_image_url("https://x.test/cat.png", None, ImageDetail::Auto)
                .unwrap_err()
                .contains("data:")
        );

        let video = vec![serde_json::json!({
            "role": "user",
            "content": [{"type": "video", "video": "data:video/mp4;base64,AAAA"}]
        })];
        assert!(find_images(&video).unwrap_err().contains("video"));
    }

    #[test]
    fn plain_string_content_has_no_images() {
        let msgs = vec![serde_json::json!({"role": "user", "content": "hello"})];
        assert!(find_images(&msgs).expect("find").is_empty());
    }

    #[test]
    fn attach_opts_validates_the_four_extensions() {
        let o = attach_opts(None, None, None, None).expect("defaults");
        assert!(o.with_meta && o.max_pages.is_none() && o.pdf_mode.is_none());
        assert!(o.forensics.is_none(), "no override -> follow endpoint auto");
        let o = attach_opts(Some("off"), Some(4), Some("text"), Some("on")).expect("explicit");
        assert!(!o.with_meta);
        assert_eq!(o.max_pages, Some(4));
        assert!(o.pdf_mode == Some(PdfMode::Text));
        assert_eq!(o.forensics, Some(true));
        assert_eq!(
            attach_opts(None, None, None, Some("off"))
                .expect("off")
                .forensics,
            Some(false)
        );
        assert!(
            attach_opts(None, None, Some("render"), None)
                .expect("render")
                .pdf_mode
                == Some(PdfMode::Render)
        );
        assert!(
            attach_opts(None, Some(0), None, None)
                .unwrap_err()
                .contains("max_pages")
        );
        assert!(
            attach_opts(None, None, Some("both"), None)
                .unwrap_err()
                .contains("pdf_mode")
        );
        assert!(
            attach_opts(Some("half"), None, None, None)
                .unwrap_err()
                .contains("file_metadata")
        );
        assert!(
            attach_opts(None, None, None, Some("yes"))
                .unwrap_err()
                .contains("forensics")
        );
    }

    /// File parts without inline bytes refuse with the caller's filename and
    /// what to do instead - never the chat template's "Unexpected item type"
    /// leak. (Parts with bytes are all claimed by the extraction lanes now:
    /// PDF, docx, sheets, and the text-native catch-all.)
    #[test]
    fn byref_file_parts_get_an_honest_refusal() {
        let by_id = serde_json::json!([{"role":"user","content":[
            {"type":"file","file":{"filename":"a.xlsx","file_id":"file-abc123"}}]}]);
        let msg = unsupported_file_part(by_id.as_array().unwrap()).expect("refused");
        assert!(msg.contains("a.xlsx"), "{msg}");
        assert!(msg.contains("file_id"), "{msg}");
        assert!(msg.contains("file_data"), "says what to do instead: {msg}");

        // Anthropic URL document sources: we never fetch
        let by_url = serde_json::json!([{"role":"user","content":[
            {"type":"document","source":{"type":"url","url":"https://x.test/a.pdf"},"title":"notes.pdf"}]}]);
        let msg = unsupported_file_part(by_url.as_array().unwrap()).expect("refused");
        assert!(msg.contains("notes.pdf"), "{msg}");
        assert!(msg.contains("URL"), "{msg}");

        // a data-carrying part is not this guard's business (lanes claim it),
        // and plain text and image parts pass untouched
        let ok = serde_json::json!([
            {"role":"user","content":"hi"},
            {"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}
        ]);
        assert!(unsupported_file_part(ok.as_array().unwrap()).is_none());
    }

    fn png_of(w: u32, h: u32) -> Vec<u8> {
        let img = image::RgbImage::from_fn(w, h, |x, y| {
            image::Rgb([(x % 251) as u8, (y % 241) as u8, 128])
        });
        let mut out = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut out, image::ImageFormat::Png)
            .expect("encode");
        out.into_inner()
    }

    /// webp and gif ride the same decode seam as png/jpeg - both are in the
    /// OpenAI AND Anthropic vision contracts (the anthropic SDK's media_type
    /// enum is exactly jpeg/png/gif/webp), so refusing them was a conformance
    /// gap. An animated gif must yield its first frame, not an error.
    #[test]
    fn webp_and_gif_decode_to_rgb() {
        use base64::Engine as _;
        let b64 = |bytes: &[u8]| base64::engine::general_purpose::STANDARD.encode(bytes);

        // lossless webp round-trips exact pixel values
        let rgb = image::RgbImage::from_pixel(20, 12, image::Rgb([255, 0, 0]));
        let mut webp = Vec::new();
        image::codecs::webp::WebPEncoder::new_lossless(&mut webp)
            .encode(rgb.as_raw(), 20, 12, image::ExtendedColorType::Rgb8)
            .expect("webp encode");
        let uri = format!("data:image/webp;base64,{}", b64(&webp));
        let img = decode_image_url(&uri, None, ImageDetail::Auto).expect("webp decode");
        assert_eq!((img.w, img.h), (20, 12));
        assert_eq!(&img.rgb[..3], &[255, 0, 0]);

        // two-frame animated gif, red then green: the decoder hands back the
        // red frame (and the alpha the encoder requires is dropped)
        let mut gif = Vec::new();
        let mut enc = image::codecs::gif::GifEncoder::new(&mut gif);
        for color in [[255u8, 0, 0, 255], [0, 255, 0, 255]] {
            let frame = image::Frame::new(image::RgbaImage::from_pixel(8, 8, image::Rgba(color)));
            enc.encode_frame(frame).expect("gif frame");
        }
        drop(enc);
        let uri = format!("data:image/gif;base64,{}", b64(&gif));
        let img = decode_image_url(&uri, None, ImageDetail::Auto).expect("gif decode");
        assert_eq!((img.w, img.h), (8, 8));
        assert_eq!(&img.rgb[..3], &[255, 0, 0]);
    }

    /// tiff rides the same seam as the contract formats (local-first
    /// superset: scanner/document files - the pdfium precedent). Encoded with
    /// the crate's own tiff encoder, decoded through the request path.
    #[test]
    fn tiff_decodes_like_the_rest() {
        use base64::Engine as _;
        let rgb = image::RgbImage::from_pixel(10, 6, image::Rgb([0, 0, 255]));
        let mut out = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(rgb)
            .write_to(&mut out, image::ImageFormat::Tiff)
            .expect("tiff encode");
        let uri = format!(
            "data:image/tiff;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(out.into_inner())
        );
        let img = decode_image_url(&uri, None, ImageDetail::Auto).expect("tiff decode");
        assert_eq!((img.w, img.h), (10, 6));
        assert_eq!(&img.rgb[..3], &[0, 0, 255]);
    }

    /// EXIF orientation is honored before the tower sees pixels: a phone JPEG
    /// stores sensor-native pixels + a rotation flag, and feeding the raw
    /// buffer shows the model a sideways image nothing downstream can detect.
    /// Orientation 6 (rotate 90 CW to display): a 4×2 red|blue strip must
    /// come out 2×4 with red on top.
    #[test]
    fn exif_orientation_uprights_the_image() {
        use base64::Engine as _;
        let rgb = image::RgbImage::from_fn(4, 2, |x, _| {
            if x < 2 {
                image::Rgb([255, 0, 0])
            } else {
                image::Rgb([0, 0, 255])
            }
        });
        let mut out = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(rgb)
            .write_to(&mut out, image::ImageFormat::Jpeg)
            .expect("jpeg encode");
        let jpeg = out.into_inner();
        // splice an EXIF APP1 with Orientation=6 right after SOI: little-endian
        // TIFF header + one IFD0 entry (tag 0x0112, SHORT, value 6)
        let tiff: &[u8] = &[
            0x49, 0x49, 0x2A, 0x00, 0x08, 0x00, 0x00, 0x00, // II*\0, IFD0 @8
            0x01, 0x00, // 1 entry
            0x12, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, // no next IFD
        ];
        let mut exif = vec![0xFF, 0xE1];
        let len = (2 + 6 + tiff.len()) as u16;
        exif.extend_from_slice(&len.to_be_bytes());
        exif.extend_from_slice(b"Exif\0\0");
        exif.extend_from_slice(tiff);
        let mut tagged = jpeg[..2].to_vec(); // SOI
        tagged.extend_from_slice(&exif);
        tagged.extend_from_slice(&jpeg[2..]);

        let uri = format!(
            "data:image/jpeg;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(&tagged)
        );
        let img = decode_image_url(&uri, None, ImageDetail::Auto).expect("jpeg decode");
        assert_eq!((img.w, img.h), (2, 4), "dimensions must swap");
        // top-left red, bottom-left blue (JPEG lossy -> tolerant thresholds)
        let top = &img.rgb[..3];
        let row3 = 3 * img.w * 3;
        let bottom = &img.rgb[row3..row3 + 3];
        assert!(
            top[0] > 128 && top[2] < 128,
            "top should be red, got {top:?}"
        );
        assert!(
            bottom[2] > 128 && bottom[0] < 128,
            "bottom should be blue, got {bottom:?}"
        );
    }

    /// An AVIF is decoded here, by rav1d, linked in. This used to be a
    /// deliberate refusal and the test asserted so; it flipped with
    /// rather than being deleted, because both directions are worth asserting
    /// at the moment they change.
    ///
    /// Fixture is paddock-heif's, reached across the crate boundary rather
    /// than copied (see that directory's README).
    #[test]
    fn an_avif_image_part_decodes() {
        use base64::Engine as _;
        let bytes = include_bytes!("../../paddock-heif/tests/data/avif32.heif");
        let uri = format!(
            "data:image/avif;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(bytes)
        );
        let img = decode_image_url(&uri, None, ImageDetail::Auto).expect("AVIF must decode");
        assert_eq!((img.w, img.h), (32, 32));
        assert!(img.rgb.iter().any(|&b| b != 0), "decoded to black");
    }

    /// A HEIC is refused, and the refusal has to be USEFUL. `doc.rs` sends a
    /// user here - "send it as an image part rather than a file part" - so this
    /// is where they land, and "unsupported format" would leave them with
    /// nowhere to go. It names the codec and says what to convert to.
    #[test]
    fn a_heic_image_part_is_refused_with_somewhere_to_go() {
        use base64::Engine as _;
        let bytes = include_bytes!("../../paddock-heif/tests/data/hevc32.heif");
        let uri = format!(
            "data:image/heic;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(bytes)
        );
        let e = decode_image_url(&uri, None, ImageDetail::Auto).expect_err("HEVC is not decodable");
        assert!(e.contains("HEIC") && e.contains("HEVC"), "{e}");
        assert!(e.contains("convert"), "a refusal must offer a way out: {e}");
    }

    /// A format nothing here reads must still fail as a clean request error
    /// naming the accepted set - never a panic, never a silent wrong answer.
    /// (JPEG 2000 in a HEIF container: sniffs as neither, decodes as neither.)
    #[test]
    fn a_format_we_read_by_no_route_errors_cleanly() {
        use base64::Engine as _;
        let junk = b"\x00\x00\x00\x14ftypj2ki\x00\x00\x00\x00j2ki";
        let uri = format!(
            "data:image/jp2;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(junk)
        );
        let err = decode_image_url(&uri, None, ImageDetail::Auto).unwrap_err();
        assert!(err.starts_with("image decode:"), "{err}");
        assert!(err.contains("accepted formats"), "{err}");
    }

    /// A shrunk-down stand-in for qwen's shape: a ceiling well above the 4096
    /// rows `auto` allows, so all three levels land somewhere different.
    fn test_budget() -> paddock_engine::generator::VisionBudget {
        paddock_engine::generator::VisionBudget {
            max_pixels: 262_144,
            min_pixels: 1_024,
            max_edge: None,
            pixels_per_token: 16,
            max_tokens: 16_384,
            min_tokens: 64,
        }
    }

    /// `detail` is read from both places the OpenAI specs put it - nested under
    /// `image_url` on chat completions, on the part itself on Responses - and
    /// an Anthropic image, whose schema has no such field, resolves to Auto.
    #[test]
    fn detail_is_read_from_both_api_spellings() {
        let uri = data_uri(&tiny_bmp());
        let b64 = uri.split(',').nth(1).expect("b64").to_owned();
        let msgs = vec![serde_json::json!({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": uri, "detail": "high"}},
                {"type": "input_image", "image_url": uri, "detail": "low"},
                {"type": "image_url", "image_url": {"url": uri}},
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/bmp", "data": b64}},
            ]
        })];
        let got: Vec<ImageDetail> = find_images(&msgs)
            .expect("find")
            .iter()
            .map(|r| r.detail)
            .collect();
        assert_eq!(
            got,
            [
                ImageDetail::High,
                ImageDetail::Low,
                ImageDetail::Auto,
                ImageDetail::Auto
            ]
        );
    }

    #[test]
    fn an_invalid_detail_is_refused_by_name() {
        let uri = data_uri(&tiny_bmp());
        let msgs = vec![serde_json::json!({
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": uri, "detail": "ultra"}}]
        })];
        let err = find_images(&msgs).unwrap_err();
        assert!(
            err.contains("ultra") && err.contains("auto, low, or high"),
            "{err}"
        );
    }

    /// `detail: low` is fitted here, and the other two levels are deliberately
    /// not - which is the narrowing `decode_image_url` documents and this test
    /// exists to hold.
    ///
    /// The reasoning, restated because it is what makes the asymmetry correct
    /// rather than an oversight: Low is the one level whose cap can bind below
    /// the family's own budget resample, so it has something to do. Auto's cap
    /// (AUTO_MAX_TOKENS) and High's (the published max) are at or above every
    /// family budget in practice, so shrinking for them made the ordinary path
    /// a double resample - Triangle here, then the family's bit-exact bicubic
    /// - which cost ~100 ms inline per A4-class request and diverged from the
    ///   reference processors, which resample exactly once from the original.
    ///
    /// This test asserted the old fit-every-level behaviour and went red when
    /// that narrowed; it was the test that was out of date, not the code.
    #[test]
    fn a_large_image_is_fitted_only_where_the_cap_can_bind() {
        let uri = data_uri(&png_of(900, 600));
        let b = test_budget();
        let dims = |d| {
            let i = decode_image_url(&uri, Some(b), d).expect("decode");
            (i.w, i.h)
        };
        // High and Auto reach the family's preprocessing at full size, and
        // byte-identically to a decode with no budget at all - that is what
        // "a single resample" means here.
        let plain = decode_image_url(&uri, None, ImageDetail::Auto).expect("decode");
        for d in [ImageDetail::High, ImageDetail::Auto] {
            let img = decode_image_url(&uri, Some(b), d).expect("decode");
            assert_eq!((img.w, img.h), (900, 600), "{d:?} pre-shrank");
            assert_eq!(img.rgb, plain.rgb, "{d:?} resampled without needing to");
        }
        // Low is the level with work to do: 64 rows x 16 px is 1024 pixels,
        // far under the source, so it fits and holds the aspect.
        let lo = dims(ImageDetail::Low);
        assert!(lo.0 * lo.1 <= 1_024, "{lo:?}");
        assert!(lo.0 < 900 && lo.1 < 600, "{lo:?}");
        let r = lo.0 as f64 / lo.1 as f64;
        assert!((r - 1.5).abs() < 0.05, "aspect drifted: {}x{}", lo.0, lo.1);
    }

    /// An image already inside the allowance is passed through byte-identical.
    /// This is what keeps the ordinary path a single resample (the family's own
    /// preprocessing) and leaves llama.cpp parity untouched.
    #[test]
    fn an_image_within_budget_is_not_resampled() {
        let uri = data_uri(&png_of(64, 48));
        let plain = decode_image_url(&uri, None, ImageDetail::Auto).expect("decode");
        for d in [ImageDetail::Auto, ImageDetail::High] {
            let fitted = decode_image_url(&uri, Some(test_budget()), d).expect("decode");
            assert_eq!((fitted.w, fitted.h), (64, 48));
            assert_eq!(fitted.rgb, plain.rgb, "pixels changed without a resize");
        }
    }

    /// And we never UPSAMPLE to reach a floor: a tiny image stays tiny, because
    /// growing it is the tower's job (it knows its own alignment grid) and
    /// guessing at that here would just be a second, worse resize.
    #[test]
    fn a_tiny_image_is_left_alone_rather_than_upsampled() {
        let uri = data_uri(&png_of(8, 8));
        let img = decode_image_url(&uri, Some(test_budget()), ImageDetail::High).expect("decode");
        assert_eq!((img.w, img.h), (8, 8));
    }

    #[test]
    fn mm_chunks_split_at_the_pad_token() {
        let img = MmChunk::Image {
            rgb: vec![0; 3],
            w: 1,
            h: 1,
        };
        let mm = build_mm_chunks(&[10, 11, 99, 12], 99, vec![img], None).expect("chunks");
        assert_eq!(mm.chunks.len(), 3);
        assert!(matches!(&mm.chunks[0], MmChunk::Text(t) if t == &vec![10, 11]));
        assert!(matches!(&mm.chunks[1], MmChunk::Image { w: 1, h: 1, .. }));
        assert!(matches!(&mm.chunks[2], MmChunk::Text(t) if t == &vec![12]));
    }

    /// The other half of the same walk: the placeholder is a POSITION, not a
    /// token, so the stream handed to the engine as generation history must not
    /// contain it. Two of the three API surfaces sent it anyway for as long as
    /// this filtering lived at the call sites - a stray id per image, silently
    /// skewing the repetition/presence/frequency penalties.
    #[test]
    fn the_engine_prompt_drops_every_placeholder() {
        let px = || MmChunk::Image {
            rgb: vec![0; 3],
            w: 1,
            h: 1,
        };
        let mm =
            build_mm_chunks(&[10, 99, 11, 99, 12], 99, vec![px(), px()], None).expect("chunks");
        assert_eq!(
            mm.text_ids,
            vec![10, 11, 12],
            "pads survived into the history stream"
        );
        // and it is exactly the concatenation of the text chunks, so the two
        // halves of MmPrompt can never describe different prompts
        let flat: Vec<u32> = mm
            .chunks
            .iter()
            .filter_map(|c| match c {
                MmChunk::Text(t) => Some(t.clone()),
                MmChunk::Image { .. }
                | MmChunk::Audio { .. }
                | MmChunk::OcrCrop(_)
                | MmChunk::VisionPixels { .. } => None,
            })
            .flatten()
            .collect();
        assert_eq!(flat, mm.text_ids);
    }

    /// A text-only prompt is untouched - the no-image path must not start
    /// filtering something out of an ordinary request.
    #[test]
    fn a_prompt_without_images_keeps_every_token() {
        let mm = build_mm_chunks(&[10, 11, 12], 99, vec![], None).expect("chunks");
        assert_eq!(mm.text_ids, vec![10, 11, 12]);
        assert_eq!(mm.chunks.len(), 1);
    }

    #[test]
    fn mm_chunk_count_mismatch_is_an_error() {
        let img = MmChunk::Image {
            rgb: vec![0; 3],
            w: 1,
            h: 1,
        };
        // no pad token in the prompt for the one image
        let err = build_mm_chunks(&[10, 11], 99, vec![img], None).unwrap_err();
        assert!(err.contains("1 media item(s)"), "{err}");
    }

    #[test]
    fn one_image_task_with_several_images_names_the_tag() {
        let px = || MmChunk::Image {
            rgb: vec![0; 3],
            w: 1,
            h: 1,
        };
        let tag = crate::chat_template::TaskTag {
            tag: "<tables_json>".into(),
            prompt: "Identify and extract the table schema".into(),
        };
        // one slot rendered (granite's tag path), four pages sent
        let err = build_mm_chunks(&[10, 99, 11], 99, vec![px(), px(), px(), px()], Some(&tag))
            .unwrap_err();
        assert!(
            err.starts_with("<tables_json> is a one-image task"),
            "{err}"
        );
        assert!(err.contains("4 images"), "{err}");
        // the fired tag is read off the RENDERED prompt, not the user's text
        let tags = [tag];
        assert!(fired_task_tag("...Identify and extract the table schema...", &tags).is_some());
        assert!(fired_task_tag("what is in this picture", &tags).is_none());
    }

    // ── audio content parts  ─────────────────────────────────────

    #[test]
    fn find_audio_takes_both_wire_shapes_in_order() {
        let msgs = [serde_json::json!({"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": "QUFB", "format": "wav"}},
            {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,QkJC"}},
            {"type": "audio_url", "audio_url": "data:audio/wav;base64,Q0ND"},
        ]})];
        let refs = find_audio(&msgs).expect("find");
        assert_eq!(refs.len(), 3);
        assert_eq!(refs[0].b64, "QUFB");
        assert_eq!(refs[1].b64, "QkJC");
        assert_eq!(refs[2].b64, "Q0ND");
    }

    #[test]
    fn remote_audio_urls_are_refused() {
        let msgs = [serde_json::json!({"role": "user", "content": [
            {"type": "audio_url", "audio_url": {"url": "https://example.com/a.wav"}},
        ]})];
        let err = find_audio(&msgs).unwrap_err();
        assert!(err.contains("does not fetch remote audio"), "{err}");
    }

    #[test]
    fn a_payloadless_audio_marker_is_refused() {
        let msgs = [serde_json::json!({"role": "user", "content": [
            {"type": "audio"},
        ]})];
        let err = find_audio(&msgs).unwrap_err();
        assert!(err.contains("no data"), "{err}");
    }

    #[test]
    fn audio_chunks_split_at_the_pad_token() {
        let clip = MmChunk::Audio {
            samples: vec![0.0; 160],
            mel: None,
        };
        let mm = build_mm_chunks(&[10, 99, 11], 99, vec![clip], None).expect("chunks");
        assert_eq!(mm.text_ids, vec![10, 11]);
        assert!(matches!(&mm.chunks[1], MmChunk::Audio { samples, .. } if samples.len() == 160));
    }

    #[test]
    fn decode_audio_parts_round_trips_wav() {
        // 16 kHz mono PCM16 WAV, 4 samples
        let mut wav: Vec<u8> = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36u32 + 8).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
        wav.extend_from_slice(&1u16.to_le_bytes()); // mono
        wav.extend_from_slice(&16000u32.to_le_bytes());
        wav.extend_from_slice(&32000u32.to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&8u32.to_le_bytes());
        for s in [0i16, 16384, -16384, 0] {
            wav.extend_from_slice(&s.to_le_bytes());
        }
        use base64::Engine as _;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&wav);
        let out = decode_audio_parts(
            vec![AudioRef {
                b64,
                format: "wav".into(),
            }],
            crate::serving::AudioFrontend::Qwen3Asr,
        )
        .expect("decode");
        assert_eq!(out.len(), 1);
        let MmChunk::Audio { samples, mel } = &out[0] else {
            panic!("not audio")
        };
        assert_eq!(samples.len(), 4);
        assert!((samples[1] - 0.5).abs() < 1e-3);
        // the runner-side frontend rode along
        assert!(mel.is_some());
    }

    /// The frontend is the SERVED MODEL's, never a default. The two contracts
    /// share no geometry, so the wrong one hands the tower a plausible plane
    /// of the wrong width - which is a width check away from being noise
    /// transcribed with a straight face.
    #[test]
    fn the_audio_frontend_is_the_models_own_contract() {
        use crate::serving::AudioFrontend;
        // 1 s of 16 kHz mono: long enough that both frontends produce real
        // frames (granite's pair-stacking gives 0 for a handful of samples)
        let mut wav = b"RIFF".to_vec();
        let n = 16000usize;
        wav.extend_from_slice(&(36 + 2 * n as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&16000u32.to_le_bytes());
        wav.extend_from_slice(&32000u32.to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(2 * n as u32).to_le_bytes());
        for i in 0..n {
            let v = ((i as f32 * 0.05).sin() * 12000.0) as i16;
            wav.extend_from_slice(&v.to_le_bytes());
        }
        use base64::Engine as _;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&wav);
        let mel_for = |fe: AudioFrontend| {
            let out = decode_audio_parts(
                vec![AudioRef {
                    b64: b64.clone(),
                    format: "wav".into(),
                }],
                fe,
            )
            .expect("decode");
            let MmChunk::Audio { mel, .. } = &out[0] else {
                panic!("not audio")
            };
            mel.clone().expect("frontend ran on the request thread")
        };

        // Qwen3-ASR: 128 Slaney mels at 100 frames/s, and the emitted plane
        // runs on to the conv stem's 100-frame chunk boundary (so its data is
        // WIDER than n_frames rows - that padding is the stem's, not audio).
        let q = mel_for(AudioFrontend::Qwen3Asr);
        assert_eq!(q.n_frames, paddock_engine::audio::real_frames(n));
        assert_eq!(q.data.len() % paddock_engine::audio::N_MEL, 0);
        assert!(q.data.len() >= q.n_frames * paddock_engine::audio::N_MEL);

        // granite-speech: 80 HTK mels pair-stacked to 160 wide at 50 frames/s,
        // exactly n_frames rows and nothing else.
        let g = mel_for(AudioFrontend::GraniteSpeech);
        assert_eq!(
            g.n_frames,
            paddock_engine::audio::granite::encoder_frames(n)
        );
        assert_eq!(
            g.data.len(),
            g.n_frames * paddock_engine::audio::granite::INPUT_DIM
        );
        assert_eq!(
            g.n_frames * 2,
            q.n_frames,
            "granite runs at half the frame rate"
        );

        // and the row-count rules the context gate quotes agree with them
        assert_eq!(
            AudioFrontend::GraniteSpeech.prompt_rows(n),
            g.n_frames
                .div_ceil(paddock_engine::audio::granite::WINDOW_SIZE)
                * 3
        );
        assert_eq!(AudioFrontend::None.prompt_rows(n), 0);
        assert!(AudioFrontend::None.features(&[0.0; 16]).is_err());
    }

    /// `max_clip_s` is `prompt_rows` run backwards, and /server publishes it
    /// so a client can split-or-skip before sending an hour of audio it was
    /// always going to refuse. The property that matters is the
    /// round trip: the answer FITS, and one second more does not.
    #[test]
    fn max_clip_s_is_the_exact_inverse_of_prompt_rows() {
        use crate::serving::AudioFrontend;
        let rate = paddock_engine::audio::SAMPLE_RATE;
        for fe in [AudioFrontend::Qwen3Asr, AudioFrontend::GraniteSpeech] {
            for rows in [1_024usize, 8_192, 32_768] {
                let s = fe
                    .max_clip_s(rows)
                    .expect("a generative lane has a ceiling");
                let at = (s * rate as f64) as usize;
                assert!(fe.prompt_rows(at) <= rows, "{fe:?}@{rows}: {s}s fits");
                assert!(
                    fe.prompt_rows(at + rate) > rows,
                    "{fe:?}@{rows}: {s}s is the LAST second that fits",
                );
            }
        }
        // Qwen3-ASR bills 13 rows a second, which is the whole reason this cap
        // is per-model: a 32k server hears ~42 minutes, not "an hour or so".
        let s = AudioFrontend::Qwen3Asr.max_clip_s(32_768).unwrap();
        assert!((2400.0..2600.0).contains(&s), "~42 min at 32k, got {s}s");
        // Whisper takes this path only via `None` - it windows, so it has no
        // context ceiling and publishes nothing rather than a made-up number.
        assert_eq!(AudioFrontend::None.max_clip_s(32_768), None);
        assert_eq!(AudioFrontend::Qwen3Asr.max_clip_s(0), None);
    }

    /// The legacy translation must produce the exact modern shape, not merely
    /// be accepted. Probing over HTTP only proves "not refused"; a mapping
    /// that dropped the `type` wrapper, or read tool_choice's nested
    /// `{"function":{"name"}}` where legacy sends a flat `{"name"}`, would
    /// pass that and then quietly declare no tools to the model.
    mod legacy_functions {
        use super::super::adopt_legacy_functions;
        use paddock_api::chat::ChatCompletionRequest;
        use serde_json::json;

        fn req(v: serde_json::Value) -> ChatCompletionRequest {
            let mut base = json!({
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "max_completion_tokens": 4
            });
            for (k, val) in v.as_object().expect("object") {
                base[k] = val.clone();
            }
            serde_json::from_value(base).expect("parses")
        }

        #[test]
        fn functions_become_type_wrapped_tools() {
            let mut r = req(json!({"functions": [
                {"name": "get_weather", "description": "d", "parameters": {"type": "object"}}
            ]}));
            adopt_legacy_functions(&mut r).expect("ok");
            let tools = r.tools.expect("tools set");
            assert_eq!(tools.len(), 1);
            assert_eq!(tools[0]["type"], "function");
            // the whole legacy object moves inside `function`, untouched
            assert_eq!(tools[0]["function"]["name"], "get_weather");
            assert_eq!(tools[0]["function"]["description"], "d");
            assert_eq!(tools[0]["function"]["parameters"]["type"], "object");
            // legacy has one call per turn and no id to correlate more
            assert_eq!(r.parallel_tool_calls, Some(false));
            // and the marker survives, so the ANSWER knows to go back legacy
            assert!(r.functions.is_some());
        }

        #[test]
        fn function_call_name_is_flat_but_tool_choice_is_nested() {
            let mut r = req(json!({
                "functions": [{"name": "f"}],
                "function_call": {"name": "f"}
            }));
            adopt_legacy_functions(&mut r).expect("ok");
            let tc = r.tool_choice.expect("tool_choice set");
            assert_eq!(tc["type"], "function");
            assert_eq!(tc["function"]["name"], "f");
            // consumed, so no downstream reader can see both spellings
            assert!(r.function_call.is_none());
        }

        #[test]
        fn none_and_auto_pass_straight_through() {
            for word in ["none", "auto"] {
                let mut r = req(json!({"functions": [{"name": "f"}], "function_call": word}));
                adopt_legacy_functions(&mut r).expect("ok");
                assert_eq!(r.tool_choice.expect("set"), json!(word));
            }
        }

        #[test]
        fn a_modern_request_is_left_completely_alone() {
            let mut r = req(json!({
                "tools": [{"type": "function", "function": {"name": "f"}}],
                "tool_choice": "auto"
            }));
            let before = (
                r.tools.clone(),
                r.tool_choice.clone(),
                r.parallel_tool_calls,
            );
            adopt_legacy_functions(&mut r).expect("ok");
            assert_eq!(
                (
                    r.tools.clone(),
                    r.tool_choice.clone(),
                    r.parallel_tool_calls
                ),
                before
            );
            assert!(
                !r.functions.is_some(),
                "must not mark a modern request legacy"
            );
        }
    }
}
