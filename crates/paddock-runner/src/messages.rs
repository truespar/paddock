//! `POST /v1/messages` (+ `/v1/messages/count_tokens`) - the Anthropic
//! Messages API, so Anthropic-SDK clients (Claude Code among them) can point
//! at Paddock. Reuses the chat pipeline: content blocks -> chat-template
//! messages -> generate -> dialect parse -> content blocks (thinking / text /
//! tool_use); the streaming path speaks the strict Anthropic event protocol
//! (message_start -> content_block_start/delta/stop -> message_delta ->
//! message_stop) that the SDK accumulator enforces.
//!
//! Semantics notes (documented, not silent): `thinking.budget_tokens` is
//! ENFORCED: at the budget the runner forces the model out of
//! its think block with the dialect's own exhaustion recipe (the Qwen3
//! report's published phrase + close token, as a forced token sequence
//! through the constraint seam - see `constrained::BudgetGated`); Anthropic's
//! >= 1024 floor and < max_tokens ceiling apply, and the budget composes with
//! > tool grammars; `thinking.type: "adaptive"` maps to enabled (locally the model
//! > always paces its own thinking); `thinking.display: "omitted"` returns
//! > thinking blocks with an empty `thinking` field and suppresses thinking
//! > deltas; gpt-oss reasons unconditionally, so it returns thinking blocks
//! > even without `thinking: enabled`; thinking blocks carry an empty
//! > `signature` (we do not sign reasoning). Streams emit one keep-alive `ping`
//! > right after `message_start`, like the live API.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_stream::stream;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Json, Response};
use paddock_api::messages::{CountTokensRequest, MessagesRequest};
use paddock_engine::sampler::SamplingParams;
use paddock_engine::service::{
    EngineError, ErrorClass, FinishReason, GenRequest, MmChunk, TokenEvent,
};
use serde_json::{Value, json};
use tokio::sync::mpsc::{UnboundedReceiver, unbounded_channel};

use crate::chat::{
    ConstraintSpec, GateSpec, build_mm_chunks, content_gate, decode_images, find_images,
    instantiate_constraint, no_forced_tool_grammar, safe_emit_len,
};
use crate::chat_template;
use crate::constrained::ToolSet;
use crate::parsers::{Dialect, Parsed, ToolHints, parse, tool_hints};
use crate::routes::AppState;
use crate::serving::ServingModel;

/// Anthropic error shape: {"type":"error","error":{"type":...,"message":...}}.
fn err(status: StatusCode, kind: &str, msg: impl Into<String>) -> Response {
    let body = json!({"type": "error", "error": {"type": kind, "message": msg.into()}});
    (status, Json(body)).into_response()
}

fn bad(msg: impl Into<String>) -> Response {
    err(StatusCode::BAD_REQUEST, "invalid_request_error", msg)
}

/// Anthropic error `type` for an engine error class (docs.anthropic.com/en/api/errors).
fn anthropic_kind(class: ErrorClass) -> &'static str {
    match class {
        ErrorClass::InvalidRequest => "invalid_request_error",
        ErrorClass::Overloaded => "overloaded_error",
        ErrorClass::Internal => "api_error",
    }
}

/// Map a classified engine error to the Anthropic error envelope: 400 for the
/// caller's fault, 529 `overloaded_error` for capacity, 500 `api_error` for ours.
fn engine_err(e: &EngineError) -> Response {
    let status = match e.class {
        ErrorClass::InvalidRequest => StatusCode::BAD_REQUEST,
        ErrorClass::Overloaded => StatusCode::from_u16(529).expect("529 is a valid status"),
        ErrorClass::Internal => StatusCode::INTERNAL_SERVER_ERROR,
    };
    err(status, anthropic_kind(e.class), &e.message)
}

/// Flatten a block `content` (string | [{type:text,...}]) to text.
fn block_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

/// An image block -> the chat-shaped image_url part the template and the
/// mm plumbing understand. `source.type: url` passes through so
/// decode_image_url can give the honest remote-fetch (SSRF) error.
fn image_part(source: &Value) -> Result<Value, String> {
    let url = match source.get("type").and_then(Value::as_str) {
        Some("base64") => {
            let media = source
                .get("media_type")
                .and_then(Value::as_str)
                .ok_or("image source needs media_type")?;
            let data = source
                .get("data")
                .and_then(Value::as_str)
                .ok_or("image source needs data")?;
            format!("data:{media};base64,{data}")
        }
        Some("url") => source
            .get("url")
            .and_then(Value::as_str)
            .ok_or("image source needs url")?
            .to_owned(),
        other => return Err(format!("unsupported image source type {other:?}")),
    };
    Ok(json!({"type": "image_url", "image_url": {"url": url}}))
}

/// Anthropic system + messages -> chat-template messages. Tool results become
/// role:"tool" messages (emitted before the rest of their user turn); image
/// blocks become chat image parts; assistant tool_use blocks become
/// tool_calls; assistant thinking text rides the `thinking` field (the
/// gpt-oss template renders it for tool-loop fidelity; qwen ignores it).
/// Fold a server-side block into Anthropic's `system`, which is a field on the
/// request rather than a message. String or content-array, both shapes kept as
/// they arrived; ours leads and the caller's text ends, matching the MCP
/// instruction merge further down and responses.rs's own.
fn merge_system(system: Option<Value>, block: &str) -> Value {
    match system {
        Some(Value::String(s)) if !s.trim().is_empty() => Value::String(format!(
            "{block}

{s}"
        )),
        Some(Value::Array(parts)) => {
            let mut out = vec![json!({"type": "text", "text": block})];
            out.extend(parts);
            Value::Array(out)
        }
        _ => Value::String(block.to_owned()),
    }
}

fn convert_messages(system: Option<&Value>, messages: &[Value]) -> Result<Vec<Value>, String> {
    let mut msgs = Vec::new();
    if let Some(sys) = system {
        msgs.push(json!({"role": "system", "content": block_text(sys)}));
    }
    for m in messages {
        let role = m
            .get("role")
            .and_then(Value::as_str)
            .ok_or("message needs a role")?;
        if role != "user" && role != "assistant" {
            return Err(format!("invalid message role {role:?}"));
        }
        let content = m.get("content").unwrap_or(&Value::Null);
        let blocks = match content {
            Value::String(s) => {
                msgs.push(json!({"role": role, "content": s}));
                continue;
            }
            Value::Array(blocks) => blocks,
            _ => return Err("message content must be a string or an array of blocks".into()),
        };

        let mut parts = Vec::new(); // text/image parts for a user message
        let mut text = String::new(); // assistant text
        let mut thinking = String::new();
        let mut tool_calls = Vec::new();
        // tool turns that must FOLLOW the assistant message (server tool
        // results ride inside the assistant content in Anthropic's history)
        let mut post_tools = Vec::new();
        let mut has_image = false;
        for b in blocks {
            match b.get("type").and_then(Value::as_str) {
                Some("text") => {
                    let t = b.get("text").and_then(Value::as_str).unwrap_or("");
                    parts.push(json!({"type": "text", "text": t}));
                    text.push_str(t);
                }
                Some("image") => {
                    if role != "user" {
                        return Err("image blocks belong in user messages".into());
                    }
                    has_image = true;
                    let source = b.get("source").ok_or("image block needs source")?;
                    parts.push(image_part(source)?);
                }
                // already chat-shaped image parts (e.g. a `document` block a PDF
                // was expanded into upstream) pass through as-is
                Some("image_url") => {
                    if role != "user" {
                        return Err("image content belongs in user messages".into());
                    }
                    has_image = true;
                    parts.push(b.clone());
                }
                Some("tool_result") => {
                    if role != "user" {
                        return Err("tool_result blocks belong in user messages".into());
                    }
                    let id = b.get("tool_use_id").and_then(Value::as_str).unwrap_or("");
                    let out = b.get("content").map(block_text).unwrap_or_default();
                    msgs.push(json!({"role": "tool", "content": out, "tool_call_id": id}));
                }
                Some("tool_use") => {
                    if role != "assistant" {
                        return Err("tool_use blocks belong in assistant messages".into());
                    }
                    let name = b.get("name").and_then(Value::as_str).unwrap_or("");
                    let args = b.get("input").cloned().unwrap_or_else(|| json!({}));
                    let id = b.get("id").and_then(Value::as_str).unwrap_or("");
                    tool_calls.push(json!({"id": id, "type": "function",
                                           "function": {"name": name, "arguments": args}}));
                }
                Some("thinking") => {
                    thinking.push_str(b.get("thinking").and_then(Value::as_str).unwrap_or(""));
                }
                Some("redacted_thinking") => {} // opaque; nothing to re-render
                // Server web-search history resent by the client: the call
                // becomes a tool_call, the result a following tool turn.
                Some("server_tool_use") => {
                    if role != "assistant" {
                        return Err("server_tool_use blocks belong in assistant messages".into());
                    }
                    let name = b.get("name").and_then(Value::as_str).unwrap_or("");
                    let args = b.get("input").cloned().unwrap_or_else(|| json!({}));
                    let id = b.get("id").and_then(Value::as_str).unwrap_or("");
                    tool_calls.push(json!({"id": id, "type": "function",
                                           "function": {"name": name, "arguments": args}}));
                }
                Some("web_search_tool_result") => {
                    if role != "assistant" {
                        return Err(
                            "web_search_tool_result blocks belong in assistant messages".into()
                        );
                    }
                    let id = b.get("tool_use_id").and_then(Value::as_str).unwrap_or("");
                    post_tools.push(json!({
                        "role": "tool",
                        "content": web_result_text(b.get("content").unwrap_or(&Value::Null)),
                        "tool_call_id": id,
                    }));
                }
                other => return Err(format!("unsupported content block type {other:?}")),
            }
        }

        if role == "user" {
            if !parts.is_empty() {
                // arrays only when an image forces part-shape; else flat text
                let content = if has_image {
                    Value::Array(parts)
                } else {
                    Value::String(text)
                };
                msgs.push(json!({"role": "user", "content": content}));
            }
        } else {
            let mut msg = json!({"role": "assistant"});
            if !text.is_empty() {
                msg["content"] = Value::String(text);
            }
            if !thinking.is_empty() {
                msg["thinking"] = Value::String(thinking);
            }
            if !tool_calls.is_empty() {
                msg["tool_calls"] = Value::Array(tool_calls);
            }
            msgs.push(msg);
        }
        msgs.extend(post_tools);
    }
    if msgs.iter().all(|m| m["role"] == "system") {
        return Err("no user input provided".into());
    }
    Ok(msgs)
}

/// A resent `web_search_tool_result` content -> the text the model sees for
/// that tool turn (result list, or the error line).
fn web_result_text(content: &Value) -> String {
    match content {
        Value::Array(arr) => {
            let mut s = String::from("Web search results:\n");
            for (i, r) in arr.iter().enumerate() {
                let title = r.get("title").and_then(Value::as_str).unwrap_or("");
                let url = r.get("url").and_then(Value::as_str).unwrap_or("");
                let body = r
                    .get("encrypted_content")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                s.push_str(&format!("\n{}. {title}\n   {url}\n   {body}\n", i + 1));
            }
            s
        }
        other => format!(
            "web search failed: {}",
            other
                .get("error_code")
                .and_then(Value::as_str)
                .unwrap_or("unavailable")
        ),
    }
}

/// Anthropic tools ({name, description?, input_schema}) -> chat-nested shape.
fn convert_tools(tools: &[Value]) -> Result<Vec<Value>, String> {
    // The server web-search tool ({"type":"web_search_20250305",...}) has no
    // input_schema - stand in its function def so count_tokens (and any path
    // that renders tools directly) sees what the model would.
    let web_def = crate::websearch::anthropic_tool_def();
    tools
        .iter()
        .map(|t| {
            let t = if t
                .get("type")
                .and_then(Value::as_str)
                .is_some_and(|s| s.starts_with("web_search"))
            {
                &web_def
            } else {
                t
            };
            let schema = t.get("input_schema").ok_or_else(|| {
                format!(
                    "unsupported tool type {:?} (only input_schema function tools)",
                    t.get("type").and_then(Value::as_str).unwrap_or("unknown")
                )
            })?;
            Ok(json!({
                "type": "function",
                "function": {
                    "name": t.get("name").cloned().unwrap_or(Value::Null),
                    "description": t.get("description").cloned().unwrap_or(Value::Null),
                    "parameters": schema.clone(),
                }
            }))
        })
        .collect()
}

struct Prepared {
    /// The rendered prompt as tokenized, one placeholder per image. This is the
    /// length a request is ADMITTED against and the reported input token count.
    prompt_ids: Vec<u32>,
    /// `prompt_ids` minus the image placeholders - what the engine is GIVEN
    /// (see `chat::MmPrompt`). A placeholder left in the history stream skews
    /// the repetition penalties against a token the model never sees.
    engine_prompt: Vec<u32>,
    sampler: SamplingParams,
    stop_tokens: Vec<u32>,
    max_tokens: usize,
    thinking_open: bool,
    hints: Option<ToolHints>,
    mm_chunks: Option<Vec<MmChunk>>,
    constraint_spec: ConstraintSpec,
    gate: GateSpec,
    /// resolved thinking budget (thinking.budget_tokens - ENFORCED)
    think_budget: Option<crate::chat::ThinkBudget>,
    single_tool_call: bool,
    stop_strings: Vec<String>,
    /// `{"applied_edits": [...]}` when the request configured context
    /// management - carried into the response (and message_delta) verbatim.
    context_management: Option<Value>,
    /// A fired compact_20260112: handle() routes to the two-iteration
    /// orchestration instead of submitting this render (whose prompt may be
    /// over-window - that is the point of compacting).
    compact: Option<CompactPlan>,
    /// deepseek2-ocr resolution (echoed on the response; `ngram` already
    /// folded into `sampler`, `force_base` into `mm_chunks`)
    ocr: Option<crate::deepseek_ocr::OcrResolved>,
}

/// What `run_compacting` needs from prepare: the fire's knobs plus the
/// effective conversation (resend-rewritten, clear-edited) to split.
struct CompactPlan {
    instructions: Option<String>,
    pause: bool,
    messages: Vec<Value>,
}

/// The conversion + render pipeline shared by generate and count_tokens.
/// Anthropic's `output_config: {effort, format}`, split into the two things
/// this server can act on. Both halves land on machinery
/// that already existed, and both correct a comment below that the spec has
/// since overtaken.
///
/// `effort` is Anthropic's own graded reasoning ladder. Until they published
/// it, `thinking.type` was the entire vocabulary this vendor had, so a model
/// with rungs (Qwen3.8) was only reachable on/off here and the note below said
/// so. Adopting it is not inventing a paddock-only field any more - it is the
/// vendor's, and its five values are a subset of the seven
/// `reasoning_effort_rank` already takes.
///
/// `format` is Anthropic's structured output. The note further down calls a
/// forced tool call "the Anthropic API's only structured-output mechanism -
/// it has no `response_format`"; `format` is that, so schema-shaped output no
/// longer needs the forced-tool detour, and reaches families whose dialect has
/// a JSON grammar but no tool syntax.
fn parse_output_config(v: Option<&Value>) -> Result<(Option<String>, Option<Value>), String> {
    let Some(cfg) = v else {
        return Ok((None, None));
    };
    let Some(obj) = cfg.as_object() else {
        return Err("output_config must be an object".into());
    };
    for k in obj.keys() {
        if k != "effort" && k != "format" {
            return Err(format!("unsupported output_config field {k:?}"));
        }
    }
    let effort = match obj.get("effort") {
        None | Some(Value::Null) => None,
        Some(Value::String(e)) => Some(e.clone()),
        Some(other) => {
            return Err(format!(
                "output_config.effort must be a string, got {other}"
            ));
        }
    };
    let format = match obj.get("format") {
        None | Some(Value::Null) => None,
        Some(f) => {
            match f.get("type").and_then(Value::as_str) {
                Some("json_schema") => {}
                Some(other) => {
                    return Err(format!(
                        "unsupported output_config.format type {other:?} (this server serves \"json_schema\")"
                    ));
                }
                None => return Err("output_config.format needs a `type`".into()),
            }
            // Anthropic puts the schema directly on `format`, not nested under
            // a second `json_schema` key the way OpenAI's response_format does.
            let schema = f
                .get("schema")
                .ok_or("output_config.format.schema is required")?;
            Some(schema.clone())
        }
    };
    Ok((effort, format))
}

fn render_prompt(
    model: &ServingModel,
    system: Option<&Value>,
    messages: &[Value],
    tools: Option<&[Value]>,
    thinking: Option<&Value>,
    // the `ocr` request object (deepseek2-ocr only) - resolved against the
    // normalized messages and returned so generation can echo it and
    // count_tokens counts the same injected text
    ocr_req: Option<&Value>,
    // `output_config.effort` - folded into the template kwargs beside
    // `enable_thinking`, exactly as the OpenAI surfaces fold reasoning_effort
    effort: Option<&str>,
) -> Result<
    (
        Vec<u32>,
        Vec<u32>,
        bool,
        Option<Vec<MmChunk>>,
        Option<crate::deepseek_ocr::OcrResolved>,
    ),
    String,
> {
    let template = model
        .chat_template
        .as_deref()
        .ok_or("this model has no chat template")?;
    let msgs = convert_messages(system, messages)?;
    // Extract before normalize_messages: convert_messages turns an
    // Anthropic `{"type":"image","source":{base64,media_type,data}}` block
    // into a chat image part, and normalize then rewrites every image part
    // down to a bare `{"type":"image"}` marker for the template. Running
    // normalize first DESTROYS the payload, and this surface rejected every
    // image with "image content part has no url" - Anthropic's API has no
    // other inline image shape, so /v1/messages could not take a picture
    // at all. The order is convert -> extract -> normalize.
    let image_refs = find_images(&msgs)?;

    // Audio parts are served on /v1/chat/completions and
    // /v1/audio/transcriptions only - refuse here rather than let the
    // template drop them silently now that `input_audio` passes part
    // validation.
    if !crate::chat::find_audio(&msgs)?.is_empty() {
        return Err(
            "audio input is served on /v1/chat/completions and /v1/audio/transcriptions, \
             not on this endpoint"
                .into(),
        );
    }
    if !image_refs.is_empty() && !model.supports_vision {
        return Err(
            "this model is not serving vision (a vision-capable model needs its `mmproj` \
             companion file set in the config to accept image input)"
                .into(),
        );
    }
    // The inverse gate, same as chat completions: a document
    // parser with no document free-runs noise, so refuse text-only requests
    // here too. expand_attachments already ran, so PDFs count as images.
    if image_refs.is_empty() && model.document_parser {
        return Err(
            "this model is a document parser - attach an image (or a PDF, which is sent \
             as page images) for it to read; it cannot answer text-only prompts"
                .into(),
        );
    }
    // Anthropic's schema has no `detail` field, so every image here resolves
    // to Auto. That is deliberate: inventing one would put a paddock-only key
    // in a request shape the Anthropic SDKs validate, and `auto` is the same
    // conservative size this surface was already serving.
    let images = decode_images(image_refs, model.engine.vision_budget())?;
    let mut msgs = chat_template::normalize_messages(&msgs);
    if let Some(marker) = model.audio_inline_marker.as_deref() {
        chat_template::inline_audio_content(&mut msgs, marker);
    }

    // thinking: explicit opt-in/out; the qwen3.5 9B template defaults off and
    // the Qwen3.6 27B template defaults on, so always pass the boolean.
    // budget_tokens is accepted, not enforced. gpt-oss and muse-glimmer have
    // no off switch at all (muse's template renders its reasoning preamble
    // unconditionally), so `disabled` cannot be honored there - `has_thinking_
    // toggle` is what says which families can answer this at all.
    // `adaptive` (the model decides) maps to enabled - local models always
    // pace their own thinking anyway.
    let enabled = match thinking.and_then(|t| t.get("type")).and_then(Value::as_str) {
        Some("enabled") | Some("adaptive") => true,
        None | Some("disabled") => false,
        Some(other) => return Err(format!("invalid thinking.type {other:?}")),
    };
    // Every toggle family reads the same `enable_thinking` name, and whether
    // this checkpoint reads it is measured from its template rather than kept
    // in a list. The list was wrong twice: it was QwenXml-only for
    // a while, which silently ignored `thinking: {"type": "disabled"}` on
    // gemma4 and laguna even though both templates were already reading the
    // flag. Ignoring a control the caller set is the failure mode this project
    // bans, so the question is now asked of the file we are serving.
    //
    // `thinking.type` used to be the whole vocabulary this vendor published, so
    // a model with a graded ladder (Qwen3.8) was only reachable on/off here and
    // this note said the ladder lived on the OpenAI surfaces. Anthropic has
    // since published `output_config.effort` (low|medium|high|xhigh|max), so
    // the ladder is reachable here too and honouring it invents nothing - see
    // `parse_output_config`. The boolean below still runs first; the rung is
    // folded on top of it.
    let mut kwargs = model
        .reasoning
        .off
        .then(|| json!({"enable_thinking": enabled}));
    // The ladder, when the caller asked for a rung. This runs after the
    // on/off boolean so an explicit `thinking: {"type": "disabled"}` plus an
    // effort still resolves the way the OpenAI surfaces resolve it, and it
    // reuses their validator so the two surfaces cannot drift on what a rung
    // means or on which models can be graded at all.
    if let Some(e) = effort {
        kwargs = crate::chat::merge_reasoning_effort(&model.reasoning, e, kwargs)?;
    }

    // deepseek2-ocr instruction mapping  - same seam as chat and
    // responses. This surface has no chat_template_kwargs channel, so the
    // top-level `ocr` field is the only form here.
    if ocr_req.is_some() && !model.ocr && !model.paddleocr {
        return Err("the `ocr` request object is only served by document-parser models".into());
    }
    let ocr = if model.ocr {
        let opts = ocr_req
            .map(crate::deepseek_ocr::OcrOpts::parse)
            .transpose()?;
        let sizes: Vec<(usize, usize)> =
            images.iter().map(crate::chat::RequestImage::size).collect();
        let max_tiles = model
            .engine
            .vision_budget()
            .map_or(0, |b| (b.max_pixels / (640 * 640)) as usize);
        crate::deepseek_ocr::resolve(&mut msgs, opts, &sizes, max_tiles)?
    } else if model.paddleocr {
        let mode = ocr_req
            .map(crate::paddle_ocr::parse_opts)
            .transpose()?
            .flatten();
        crate::paddle_ocr::resolve(&mut msgs, mode, images.len())?
    } else {
        None
    };

    let mut prompt = chat_template::render(template, &msgs, tools, kwargs.as_ref())?;
    // thinking-mode detection is dialect-shaped - see Dialect::thinking_open
    // (qwen pre-opens "<think>\n", laguna a bare "<think>", gemma4 pre-closes
    // when off)
    let thinking_open = model.dialect.thinking_open(&prompt);
    // gemma4 thinking: pre-open the thought channel so the token sampled from
    // the prefill logits is already visible reasoning text (see g4_preopen)
    if thinking_open
        && model.dialect == crate::parsers::Dialect::GemmaChannel
        && crate::parsers::g4_preopen()
    {
        prompt.push_str(crate::parsers::G_THOUGHT);
    }
    let mut prompt_ids = model.tokenizer.encode(&prompt).map_err(|e| e.to_string())?;
    // BOS-leading families (gemma4): chat templates emit text only - the
    // leading BOS is the tokenizer's job, same as the raw-completions path
    if let Some(bos) = model.bos
        && prompt_ids.first() != Some(&bos)
    {
        prompt_ids.insert(0, bos);
    }

    let (mut mm_chunks, engine_prompt) = if images.is_empty() {
        (None, prompt_ids.clone())
    } else {
        let pad = model
            .image_pad_id
            .ok_or("model has no <|image_pad|> token")?;
        let media = images
            .into_iter()
            .map(crate::chat::RequestImage::into_chunk)
            .collect();
        let task = crate::chat::fired_task_tag(&prompt, &model.task_tags);
        let mm = build_mm_chunks(&prompt_ids, pad, media, task)?;
        (Some(mm.chunks), mm.text_ids)
    };
    // resolved crop override -> directive chunk the OCR engine consumes
    if let (Some(o), Some(chunks)) = (&ocr, mm_chunks.as_mut())
        && o.force_base
    {
        chunks.insert(
            0,
            paddock_engine::service::MmChunk::OcrCrop(paddock_engine::service::OcrCropMode::Base),
        );
    }
    Ok((prompt_ids, engine_prompt, thinking_open, mm_chunks, ocr))
}

fn prepare(
    model: &ServingModel,
    req: &MessagesRequest,
    output_ceiling: Option<usize>,
    sd: &crate::routes::SamplingDefaults,
) -> Result<Prepared, String> {
    let mut tools = req.tools.as_ref().map(|ts| convert_tools(ts)).transpose()?;
    let (effort_owned, out_format) = parse_output_config(req.output_config.as_ref())?;
    let effort = effort_owned.as_deref();

    // tool_choice {type: auto|any|tool|none, disable_parallel_tool_use?}
    let mut forced_tool: Option<Option<String>> = None;
    let mut single_tool_call = false;
    if let Some(tc) = req.tool_choice.as_ref() {
        single_tool_call = tc
            .get("disable_parallel_tool_use")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        match tc.get("type").and_then(Value::as_str) {
            Some("auto") => {}
            Some("none") => tools = None,
            Some("any") => forced_tool = Some(None),
            Some("tool") => {
                let name = tc
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or("tool_choice type \"tool\" needs a name")?;
                forced_tool = Some(Some(name.to_owned()));
            }
            other => return Err(format!("invalid tool_choice type {other:?}")),
        }
    }
    // A forced call used to be the Anthropic API's only structured-output
    // mechanism, because the surface had no `response_format`. It has one now
    // (`output_config.format`), handled above and needing no tool syntax at
    // all - so this gate no longer decides whether schema-shaped output is
    // reachable, only whether a FORCED TOOL CALL is.
    let tool_syntax = match forced_tool {
        None => None,
        Some(_) => Some(
            model
                .dialect
                .tool_syntax()
                .ok_or_else(|| no_forced_tool_grammar(model.dialect, "\"any\"/\"tool\""))?,
        ),
    };

    // A round-tripped `compaction` block defines the effective conversation:
    // everything before it already collapsed into its summary. Active with or
    // without a context_management config - a request carrying our own
    // response's content must always be valid.
    let rewritten = crate::context_management::resend_rewrite(&req.messages);
    let messages: &[Value] = rewritten.as_deref().unwrap_or(&req.messages);

    // Server-side context management: parse before the render so
    // a malformed config is a 400 even when its trigger would not fire, apply
    // to the raw message array, and re-render only when an edit landed. The
    // common case (no cm, or trigger unmet) costs exactly one render.
    let cm_cfg = req
        .context_management
        .as_ref()
        .map(crate::context_management::parse)
        .transpose()?;
    let first = render_prompt(
        model,
        req.system.as_ref(),
        messages,
        tools.as_deref(),
        req.thinking.as_ref(),
        req.ocr.as_ref(),
        effort,
    )?;
    let mut cm_report = None;
    let mut compact = None;
    let (prompt_ids, engine_prompt, thinking_open, mut mm_chunks, ocr) = match &cm_cfg {
        None => first,
        Some(cfg) => {
            let count = |m: &[Value]| {
                render_prompt(
                    model,
                    req.system.as_ref(),
                    m,
                    tools.as_deref(),
                    req.thinking.as_ref(),
                    req.ocr.as_ref(),
                    effort,
                )
                .map(|(ids, _, _, _, _)| ids.len())
            };
            let (edited, applied) =
                crate::context_management::apply(cfg, messages, first.0.len(), count)?;
            cm_report = Some(json!({"applied_edits": applied.edits}));
            if let Some(fire) = applied.compact {
                // this render is not what will generate - run_compacting
                // re-renders both iterations; hand it the edited conversation
                compact = Some(CompactPlan {
                    instructions: fire.instructions,
                    pause: fire.pause,
                    messages: edited,
                });
                first
            } else if applied.edits.is_empty() {
                first
            } else {
                render_prompt(
                    model,
                    req.system.as_ref(),
                    &edited,
                    tools.as_deref(),
                    req.thinking.as_ref(),
                    req.ocr.as_ref(),
                    effort,
                )?
            }
        }
    };

    // vLLM-compat per-request pixel budget (paddleocr family) - same seam
    // and same refusal semantics as chat completions
    if let Some((min_px, max_px)) =
        crate::chat::parse_mm_processor_kwargs(req.mm_processor_kwargs.as_ref())?
    {
        let Some(chunks) = mm_chunks.as_mut() else {
            return Err("mm_processor_kwargs was sent but the request carries no image".into());
        };
        chunks.insert(
            0,
            paddock_engine::service::MmChunk::VisionPixels {
                min_pixels: min_px,
                max_pixels: max_px,
            },
        );
    }

    // `output_config.format` is Anthropic's structured output, and it is a
    // DIRECT one: unlike a forced tool call it needs no tool syntax from the
    // dialect, so schema-shaped output now reaches families that have a JSON
    // grammar but no tool grammar. Refusing the combination matches chat's
    // rule for response_format + forced tool_choice - the output cannot be
    // both a JSON answer and a tool call.
    if out_format.is_some() && forced_tool.is_some() {
        return Err(
            "output_config.format cannot be combined with a forced tool_choice (the output cannot be both a JSON answer and a tool call)"
                .into(),
        );
    }
    let mut constraint_spec = match forced_tool {
        Some(only) => ConstraintSpec::Tool(ToolSet::compile(
            tool_syntax.expect("gated with forced_tool"),
            tools.as_deref().unwrap_or(&[]),
            only.as_deref(),
        )?),
        // `tool_choice` absent or "auto": arm the dialect's tool grammar as a
        // dispatch so a call the model chooses to make is well formed by
        // construction. `disable_parallel_tool_use` disarms it
        // after the first call rather than dropping the extras afterwards.
        // A schema wins over the auto tool dispatch: the caller asked for JSON,
        // not for a tool the model might choose to call.
        None => match &out_format {
            Some(schema) => {
                ConstraintSpec::Json(crate::constrained::CompiledSchema::compile(schema)?)
            }
            None => crate::chat::auto_tool_dispatch(model, tools.as_deref(), single_tool_call),
        },
    };
    // Only the forced path may refuse; auto dispatch degrades to unconstrained.
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

    // Anthropic documents temperature 0..1, not OpenAI's 0..2 - the surfaces
    // genuinely differ here, so the ceiling is the vendor's own.
    crate::chat::validate_sampling(req.temperature, 1.0, req.top_p, req.min_p, None, None)?;

    // this model's elected defaults for this turn - see chat::prepare
    // thinking.budget_tokens - ENFORCED (this closes the module doc's old
    // "accepted, not enforced" note): at the budget the runner forces the
    // model out of its think block with the dialect's exhaustion recipe.
    // Anthropic's own floor and ceiling apply: >= 1024, < max_tokens.
    let think_budget = match req.thinking.as_ref().and_then(|t| t.get("budget_tokens")) {
        None => None,
        Some(v) => {
            let n = v
                .as_u64()
                .ok_or("thinking.budget_tokens must be an integer")?;
            if n < 1024 {
                return Err("thinking.budget_tokens must be at least 1024".into());
            }
            if n as usize >= req.max_tokens {
                return Err("`max_tokens` must be greater than `thinking.budget_tokens`".into());
            }
            Some(crate::chat::think_budget(
                model,
                n as usize,
                thinking_open,
                "thinking.budget_tokens",
            )?)
        }
    };
    let dflt = sd.resolve(thinking_open);
    let sampler = SamplingParams {
        // document parsers default greedy - same rule as chat completions
        temperature: req.temperature.unwrap_or(if model.document_parser {
            0.0
        } else {
            dflt.temp
        }),
        top_k: req.top_k.unwrap_or(dflt.top_k),
        top_p: req.top_p.unwrap_or(dflt.top_p),
        min_p: req.min_p.unwrap_or(dflt.min_p),
        // the Anthropic wire schema has no penalty knobs - the server-side
        // defaults are the only handle here
        repeat_penalty: dflt.repeat_penalty,
        repeat_last_n: sd.repeat_last_n,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        seed: sd.seed_or_now(req.seed),
        // the Anthropic API has no logit_bias knob
        logit_bias: Vec::new(),
        // the OCR family's repetition guard (reference parity), off elsewhere
        no_repeat_ngram: ocr.as_ref().map_or((0, 0), |o| o.ngram),
    };
    Ok(Prepared {
        prompt_ids,
        engine_prompt,
        sampler,
        stop_tokens: model.stop_tokens.clone(),
        max_tokens: output_ceiling.map_or(req.max_tokens, |c| req.max_tokens.min(c)),
        thinking_open,
        hints: tool_hints(tools.as_deref()),
        mm_chunks,
        constraint_spec,
        gate,
        think_budget,
        single_tool_call,
        stop_strings: req.stop_sequences.clone(),
        context_management: cm_report,
        compact,
        ocr,
    })
}

/// `thinking.display`: `summarized` is the upstream default - locally the
/// full thinking text is the returned block, since there is no separate
/// summarizer; `omitted` returns thinking blocks with an empty `thinking`
/// field and no thinking deltas. Unknown values are a 400.
fn thinking_omitted(thinking: Option<&Value>) -> Result<bool, String> {
    match thinking
        .and_then(|t| t.get("display"))
        .and_then(Value::as_str)
    {
        None | Some("summarized") => Ok(false),
        Some("omitted") => Ok(true),
        Some(other) => Err(format!("invalid thinking.display {other:?}")),
    }
}

struct Ctx {
    id: String,
    model_id: String,
    tokenizer: Arc<paddock_tokenizer::GgufTokenizer>,
    prompt_len: usize,
    dialect: Dialect,
    thinking_open: bool,
    hints: Option<ToolHints>,
    single_tool_call: bool,
    stop_strings: Vec<String>,
    /// thinking.display "omitted": withhold thinking text from the response.
    omit_thinking: bool,
    /// `{"applied_edits": [...]}` when context management ran - reported on
    /// the response body and the message_delta event, per the API contract.
    context_management: Option<Value>,
    /// compact_20260112 output riding this response: the summary (None = the
    /// pass produced nothing usable -> a null-content block, SDK-pinned) and
    /// the summarization pass's usage.iterations entry.
    compaction: Option<CompactionOut>,
    /// deepseek2-ocr resolution - the response's `ocr` extension, with
    /// grounded `regions` parsed from the finished output when armed.
    ocr: Option<crate::deepseek_ocr::OcrResolved>,
    /// Event-record slots (§8.1); no-op unless the events middleware planted one.
    scope: crate::events::EventScope,
}

struct CompactionOut {
    summary: Option<String>,
    /// `{"type": "compaction", ...}` iteration usage, SDK shape.
    usage: Value,
}

/// SDK-pinned block shape; serde turns a None summary into the null content
/// that marks a failed compaction (round-trips as a no-op).
fn compaction_block(summary: &Option<String>) -> Value {
    json!({"type": "compaction", "content": summary})
}

impl Ctx {
    /// The `ocr` extension object for this turn - the resolution echo plus
    /// grounded `regions` parsed from a decode that keeps special tokens
    /// (the markup rides on `<|ref|>`/`<|det|>` specials).
    fn ocr_json(&self, ids: &[u32]) -> Option<Value> {
        let o = self.ocr.as_ref()?;
        let mut echo = o.echo();
        if let Ok(raw) = self.tokenizer.decode(ids, false)
            && let Some(regions) = crate::deepseek_ocr::regions_json(&raw)
        {
            echo["regions"] = regions;
        }
        Some(echo)
    }

    fn parse(&self, ids: &[u32]) -> Parsed {
        let raw = self.tokenizer.decode(ids, false).unwrap_or_default();
        self.parse_raw(&raw)
    }

    /// The parse half of [`Self::parse`] for the streaming loop, which
    /// decodes incrementally (StreamDecoder) instead of re-decoding the
    /// whole id run per token (the O(n^2) long-stream collapse under
    /// concurrency).
    fn parse_raw(&self, raw: &str) -> Parsed {
        let mut parsed = parse(self.dialect, raw, self.thinking_open, self.hints.as_ref());
        if self.single_tool_call {
            parsed.tool_calls.truncate(1);
            parsed.complete_calls = parsed.complete_calls.min(1);
        }
        parsed
    }
}

/// Earliest stop-sequence hit: (byte index, which sequence).
fn find_stop<'a>(text: &str, stops: &'a [String]) -> Option<(usize, &'a str)> {
    let mut best: Option<(usize, &'a str)> = None;
    for s in stops {
        if !s.is_empty()
            && let Some(i) = text.find(s.as_str())
            && best.is_none_or(|(bi, _)| i < bi)
        {
            best = Some((i, s));
        }
    }
    best
}

/// The Anthropic server web-search tool (`web_search_20250305`), extracted from
/// the request's `tools` and resolved to the configured provider.
struct AnthWeb {
    spec: crate::websearch::WebSpec,
    /// searches allowed this request; 0 = unlimited.
    max_uses: usize,
}

/// Pull a `web_search_*` server tool out of `tools` (the agent loop injects the
/// model-facing function def instead). Declaring it on a server with no
/// provider configured is a clear 400.
fn extract_web_tool(
    state: &AppState,
    req: &mut MessagesRequest,
) -> Result<Option<AnthWeb>, String> {
    let Some(tools) = req.tools.as_mut() else {
        return Ok(None);
    };
    let Some(pos) = tools.iter().position(|t| {
        t.get("type")
            .and_then(Value::as_str)
            .is_some_and(|s| s.starts_with("web_search"))
    }) else {
        return Ok(None);
    };
    let t = tools.remove(pos);
    let Some(cfg) = state.live.snapshot().web_search.clone() else {
        return Err("web search is not set up on this runner; launch it with \
                    --web-search-provider/--web-search-api-key (or configure a \
                    provider in the manager's Studio Settings)"
            .into());
    };
    let mut opts = crate::websearch::SearchOpts::default();
    for (field, dst) in [
        ("allowed_domains", &mut opts.allowed_domains),
        ("blocked_domains", &mut opts.blocked_domains),
    ] {
        if let Some(arr) = t.get(field).and_then(Value::as_array) {
            *dst = arr
                .iter()
                .filter_map(|d| d.as_str().map(str::to_string))
                .collect();
        }
    }
    // the whole approximate location, not just the country: providers want
    // different shapes of it (see paddock_websearch::Location)
    opts.location = crate::websearch::Location::from_json(t.get("user_location"));
    let max_uses = t.get("max_uses").and_then(Value::as_u64).unwrap_or(0) as usize;
    Ok(Some(AnthWeb {
        spec: crate::websearch::WebSpec { cfg, opts },
        max_uses,
    }))
}

/// The Anthropic server web-FETCH tool (`web_fetch_*`), extracted from the
/// request's `tools`. Four versions exist upstream (20250910 -> 20260318) and
/// they differ only in options we either honour or have no use for, so the
/// prefix match accepts all of them rather than pinning one.
struct AnthFetch {
    spec: crate::websearch::FetchSpec,
    /// `citations.enabled` - off by default for fetch, unlike search
    citations: bool,
}

/// Pull a `web_fetch_*` server tool out of `tools`. Declaring it on a runner
/// with no provider is a clear 400, exactly like web search; declaring it on a
/// provider that only sells search is also a 400, and says which providers can.
fn extract_fetch_tool(
    state: &AppState,
    req: &mut MessagesRequest,
) -> Result<Option<AnthFetch>, String> {
    let Some(tools) = req.tools.as_mut() else {
        return Ok(None);
    };
    let Some(pos) = tools.iter().position(|t| {
        t.get("type")
            .and_then(Value::as_str)
            .is_some_and(|s| s.starts_with("web_fetch"))
    }) else {
        return Ok(None);
    };
    let t = tools.remove(pos);
    let Some(cfg) = state.live.snapshot().web_search.clone() else {
        return Err("web fetch is not set up on this runner; launch it with --web-search-provider/--web-search-api-key (or configure a provider in the manager's Studio Settings)"
            .into());
    };
    if !cfg.provider.can_fetch() {
        return Err(format!(
            "web fetch needs a provider that reads pages; this endpoint is on {}, which only searches. Use exa, tavily or firecrawl.",
            cfg.provider.label()
        ));
    }
    let mut opts = crate::websearch::FetchOpts::default()
        .content_tokens(t.get("max_content_tokens").and_then(Value::as_u64));
    for (field, dst) in [
        ("allowed_domains", &mut opts.allowed_domains),
        ("blocked_domains", &mut opts.blocked_domains),
    ] {
        if let Some(arr) = t.get(field).and_then(Value::as_array) {
            *dst = arr
                .iter()
                .filter_map(|d| d.as_str().map(str::to_string))
                .collect();
        }
    }
    let max_uses = t.get("max_uses").and_then(Value::as_u64).unwrap_or(0) as usize;
    let citations = t
        .pointer("/citations/enabled")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    Ok(Some(AnthFetch {
        spec: crate::websearch::FetchSpec {
            cfg,
            opts,
            max_uses,
        },
        citations,
    }))
}

/// Pull a `{"type":"forensics"}` server tool out of `tools` (paddock extension,
/// the same shape the Responses path serves). Off unless the runner has
/// `[forensics] tool = true`; a request asking for it on a runner that did not
/// enable it gets a clear 400, never a silent no-tool run. The agent loop injects
/// the model-facing function def instead, exactly like web search.
fn extract_forensics_tool(
    state: &AppState,
    req: &mut MessagesRequest,
) -> Result<Option<std::sync::Arc<crate::forensics::ForensicRuntime>>, String> {
    let Some(tools) = req.tools.as_mut() else {
        return Ok(None);
    };
    let Some(pos) = tools
        .iter()
        .position(|t| t.get("type").and_then(Value::as_str) == Some("forensics"))
    else {
        return Ok(None);
    };
    let Some(rt) = state.forensics.clone().filter(|r| r.tool) else {
        return Err("forensic analysis is not enabled on this runner; set \
                    [forensics] enabled = true and tool = true in its config"
            .into());
    };
    tools.remove(pos);
    Ok(Some(rt))
}

/// Pull a `{"type":"current_time"}` server tool out of `tools` (paddock
/// extension, same shape the Responses path serves). Always served - a clock
/// needs no provider or enablement; the declared timezone is validated here
/// so a junk zone is a 400 at request time, never a wrong clock answer.
fn extract_clock_tool(
    req: &mut MessagesRequest,
) -> Result<Option<crate::clock::ClockSpec>, String> {
    let Some(tools) = req.tools.as_mut() else {
        return Ok(None);
    };
    let Some(pos) = tools
        .iter()
        .position(|t| t.get("type").and_then(Value::as_str) == Some("current_time"))
    else {
        return Ok(None);
    };
    let spec = crate::clock::parse_spec(&tools[pos])?;
    tools.remove(pos);
    Ok(Some(spec))
}

/// Every URL the conversation has already put in front of the model.
///
/// This is the load-bearing half of web fetch's security model, and it lives
/// here because only the request knows the conversation. A model that can
/// fetch a URL it INVENTED can exfiltrate its own context by encoding it into
/// a hostname or a path - so a URL that has not been seen is refused, and
/// URLs discovered by a search this turn are added as they arrive.
fn known_urls(req: &MessagesRequest) -> std::collections::HashSet<String> {
    let mut out = std::collections::HashSet::new();
    for m in &req.messages {
        harvest_urls(&serde_json::to_value(m).unwrap_or(Value::Null), &mut out);
    }
    if let Some(s) = &req.system {
        harvest_urls(s, &mut out);
    }
    out
}

/// Walk any JSON and collect the http(s) URLs inside its strings. Deliberately
/// shape-blind: a URL is as legitimate in a tool result or a document block as
/// in a user's sentence, and enumerating the shapes would just be a list to
/// forget to update.
fn harvest_urls(v: &Value, out: &mut std::collections::HashSet<String>) {
    match v {
        Value::String(s) => {
            for (i, _) in s.match_indices("http") {
                let rest = &s[i..];
                if !(rest.starts_with("http://") || rest.starts_with("https://")) {
                    continue;
                }
                // a URL ends at whitespace or at the punctuation that
                // ordinarily follows one in prose
                let end = rest
                    .find(|c: char| {
                        c.is_whitespace() || matches!(c, '"' | '<' | '>' | '\\' | '|' | '`')
                    })
                    .unwrap_or(rest.len());
                let url = rest[..end].trim_end_matches(|c| {
                    matches!(c, '.' | ',' | ')' | ']' | '}' | ';' | ':' | '\'')
                });
                if url.len() > "https://".len() {
                    out.insert(url.to_string());
                }
            }
        }
        Value::Array(a) => a.iter().for_each(|x| harvest_urls(x, out)),
        Value::Object(o) => o.values().for_each(|x| harvest_urls(x, out)),
        _ => {}
    }
}

/// Run one Anthropic web fetch honoring `max_uses` and the seen-URL rule.
/// Returns the `web_fetch_tool_result` content and the model-facing feedback.
async fn run_anth_fetch(
    f: &AnthFetch,
    uses: &mut usize,
    requests: &mut usize,
    seen: &std::collections::HashSet<String>,
    url: &str,
) -> (Value, String) {
    let refuse = |code: &str, why: &str| {
        (
            crate::websearch::anthropic_fetch_error(code),
            format!("web fetch failed: {why}"),
        )
    };
    if url.trim().is_empty() {
        return refuse("invalid_tool_input", "no URL was given");
    }
    if f.spec.max_uses > 0 && *uses >= f.spec.max_uses {
        return refuse(
            "max_uses_exceeded",
            "the fetch limit for this request was reached",
        );
    }
    if !seen.iter().any(|s| crate::websearch::same_url(s, url)) {
        return refuse(
            "url_not_in_prior_context",
            "that URL has not appeared in this conversation, so it cannot be fetched",
        );
    }
    *uses += 1;
    *requests += 1;
    match crate::websearch::fetch(&f.spec.cfg, &f.spec.opts, url).await {
        Ok(got) => {
            crate::metrics::web_search_billed(&f.spec.cfg.provider, &got.usage);
            let at = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
            let feedback = format!(
                "Fetched {}{}:

{}",
                got.url,
                if got.title.is_empty() {
                    String::new()
                } else {
                    format!(" ({})", got.title)
                },
                got.content
            );
            (
                crate::websearch::anthropic_fetch_content(&got, &at, f.citations),
                feedback,
            )
        }
        Err(e) => (
            crate::websearch::anthropic_fetch_error(e.code),
            format!("web fetch failed: {e}"),
        ),
    }
}

/// Run one Anthropic web search honoring `max_uses`. Returns the
/// `web_search_tool_result` content and the model-facing feedback, bumping the
/// use/request counters.
async fn run_anth_web(
    w: &AnthWeb,
    uses: &mut usize,
    requests: &mut usize,
    query: &str,
) -> (Value, String) {
    if query.trim().is_empty() {
        return (
            json!({"type": "web_search_tool_result_error", "error_code": "invalid_tool_input"}),
            "web search failed: the search query was empty".into(),
        );
    }
    if w.max_uses > 0 && *uses >= w.max_uses {
        return (
            json!({"type": "web_search_tool_result_error", "error_code": "max_uses_exceeded"}),
            "web search failed: the search limit for this request was reached".into(),
        );
    }
    *uses += 1;
    *requests += 1;
    match crate::websearch::search(&w.spec.cfg, &w.spec.opts, query).await {
        Ok(found) => {
            crate::metrics::web_search_billed(&w.spec.cfg.provider, &found.usage);
            (
                crate::websearch::anthropic_result_content(&found.hits),
                crate::websearch::result_feedback(query, &found.hits),
            )
        }
        Err(e) => (
            json!({"type": "web_search_tool_result_error", "error_code": e.error_code()}),
            format!("web search failed: {e}"),
        ),
    }
}

pub async fn handle(
    State(state): State<Arc<AppState>>,
    scope: Option<axum::Extension<crate::events::EventScope>>,
    crate::extract::AnthJson(mut req): crate::extract::AnthJson<MessagesRequest>,
) -> Response {
    let scope = scope.map(|e| e.0).unwrap_or_default();
    // Shape-check `output_config` before the model-availability check. It reads
    // nothing but the request, and a malformed one is malformed whether or not
    // a model is loaded - answering 503 there hides a 400 the caller has to
    // fix. `prepare` parses it again for real; this call exists only so the
    // refusal is honest and reachable, and it is a few string compares.
    if let Err(e) = parse_output_config(req.output_config.as_ref()) {
        return err(StatusCode::BAD_REQUEST, "invalid_request_error", e);
    }
    let Some(model) = state.serving.as_ref() else {
        return err(
            StatusCode::SERVICE_UNAVAILABLE,
            "overloaded_error",
            "no model is loaded",
        );
    };
    scope.model(&model.id);
    // the Anthropic grouping key rides in metadata.user_id
    scope.user(
        req.metadata
            .as_ref()
            .and_then(|m| m.get("user_id"))
            .and_then(Value::as_str),
    );
    // PDF `document` blocks -> page images or extracted text (before the
    // MCP/agent branch and convert_messages, whose pass-through of text and
    // image_url parts preserves the expansion).
    let opts = match crate::chat::attach_opts(
        req.file_metadata.as_deref(),
        req.max_pages,
        req.pdf_mode.as_deref(),
        req.forensics.as_deref(),
    ) {
        Ok(o) => o,
        Err(e) => return bad(e),
    };
    // Anthropic /v1/messages: injection-only, no extra output item - discard.
    match crate::chat::expand_attachments(&state, model, &mut req.messages, opts, &mut Vec::new())
        .await
    {
        // Anthropic keeps `system` out of the message array, so the capability
        // merges into that field - same shape as the MCP instruction block
        // below, ours leading and the caller's text keeping the tail.
        Ok(Some(sample)) => req.system = Some(merge_system(req.system.take(), &sample)),
        Ok(None) => {}
        Err((code, msg)) => {
            let kind = if code == StatusCode::BAD_REQUEST {
                "invalid_request_error"
            } else {
                "api_error"
            };
            return err(code, kind, msg);
        }
    }
    // Server tools that trigger the agent loop: the web-search tool in `tools`,
    // and the Anthropic MCP connector's `mcp_servers` (beta).
    let fetch = match extract_fetch_tool(&state, &mut req) {
        Ok(f) => f,
        Err(msg) => return err(StatusCode::BAD_REQUEST, "invalid_request_error", msg),
    };
    // the seen-URL set is taken before any tool injection, from the
    // conversation as the caller sent it
    let seen_urls = known_urls(&req);
    let web = match extract_web_tool(&state, &mut req) {
        Ok(w) => w,
        Err(e) => return bad(e),
    };
    let forensics = match extract_forensics_tool(&state, &mut req) {
        Ok(f) => f,
        Err(e) => return bad(e),
    };
    let clock = match extract_clock_tool(&mut req) {
        Ok(c) => c,
        Err(e) => return bad(e),
    };
    let toolsets = match extract_mcp_toolsets(&mut req) {
        Ok(t) => t,
        Err(e) => return bad(e),
    };
    let (mcp_tools, routing, catalog, mcp_instructions) =
        if let Some(servers) = req.mcp_servers.clone().filter(|s| !s.is_empty()) {
            match gather_mcp_servers(&state, &servers, &toolsets).await {
                Ok(g) => g,
                Err(e) => return bad(e),
            }
        } else {
            (Vec::new(), HashMap::new(), Vec::new(), Vec::new())
        };
    // A server's own `instructions` describe how to use it; the spec means the
    // host to put them in front of the model. Appended to the caller's system
    // block, never replacing it.
    if !mcp_instructions.is_empty() {
        let block = mcp_instructions.join(
            "

",
        );
        // Ours leads, the caller's ends - see merge_instructions in
        // responses.rs for why the tail position belongs to the caller.
        let merged = match req.system.take() {
            Some(Value::String(s)) if !s.trim().is_empty() => Value::String(format!(
                "{block}

{s}"
            )),
            Some(Value::Array(parts)) => {
                let mut out = vec![json!({"type": "text", "text": block})];
                out.extend(parts);
                Value::Array(out)
            }
            _ => Value::String(block),
        };
        req.system = Some(merged);
    }
    if !routing.is_empty()
        || web.is_some()
        || fetch.is_some()
        || forensics.is_some()
        || clock.is_some()
    {
        // One tool set for the whole loop: the caller's client tools, the
        // discovered MCP tools, web search, and forensics. Merged here rather
        // than inside each loop because the round-0 context-management pass has
        // to render exactly the tools the loop will (cache alignment).
        let mut all = req.tools.take().unwrap_or_default();
        all.extend(mcp_tools);
        if web.is_some() {
            all.push(crate::websearch::anthropic_tool_def());
        }
        if fetch.is_some() {
            all.push(crate::websearch::anthropic_fetch_tool_def());
        }
        if forensics.is_some() {
            all.push(crate::forensics::anthropic_tool_def());
        }
        if clock.is_some() {
            all.push(crate::clock::anthropic_tool_def());
        }
        req.tools = Some(all);
        // compact_20260112 for the agent loops: runs once,
        // here, before the first round. The clear_* strategies keep applying
        // per round inside the loop, where they belong.
        let compaction = match precompact_agent(&state, &mut req, &scope).await {
            Ok(c) => c,
            Err(resp) => return resp,
        };
        return if req.stream {
            stream_mcp_agent(
                state.clone(),
                req,
                routing,
                catalog,
                web,
                fetch,
                forensics,
                clock,
                seen_urls,
                compaction,
                scope,
            )
        } else {
            run_mcp_agent(
                state.clone(),
                req,
                routing,
                catalog,
                web,
                fetch,
                forensics,
                clock,
                seen_urls,
                compaction,
                scope,
            )
            .await
        };
    }
    let t_prep = std::time::Instant::now();
    let mut prepared = match prepare(model, &req, state.max_output_ceiling, &state.sampling) {
        Ok(p) => p,
        Err(e) => return bad(e),
    };
    scope.tokenized(t_prep.elapsed());
    let omit_thinking = match thinking_omitted(req.thinking.as_ref()) {
        Ok(o) => o,
        Err(e) => return bad(e),
    };
    // compact_20260112 fired: hand off to the two-iteration orchestration
    // before the context gate - an over-window prompt is exactly what a
    // compaction exists to rescue (both iterations gate their own prompts).
    if let Some(plan) = prepared.compact.take() {
        return run_compacting(
            state.clone(),
            req,
            plan,
            prepared.context_management,
            omit_thinking,
            scope,
        )
        .await;
    }
    // Over-window prompt: clean 400 at the edge for stream and non-stream alike
    // (a streaming request has committed its 200 SSE status before the engine's
    // own admit check can answer). Prices image rows too - what prefill will
    // actually see, not just the token stream.
    if let Some(e) = crate::chat::context_gate(
        model,
        prepared.engine_prompt.len(),
        prepared.mm_chunks.as_deref(),
        state.max_ctx,
    ) {
        return engine_err(&e);
    }
    let ctx = Ctx {
        id: format!("msg_{}", uuid::Uuid::new_v4().simple()),
        model_id: model.id.clone(),
        tokenizer: model.tokenizer.clone(),
        prompt_len: prepared.prompt_ids.len(),
        dialect: model.dialect,
        thinking_open: prepared.thinking_open,
        hints: prepared.hints.clone(),
        single_tool_call: prepared.single_tool_call,
        stop_strings: prepared.stop_strings.clone(),
        omit_thinking,
        context_management: prepared.context_management.clone(),
        compaction: None,
        ocr: prepared.ocr.clone(),
        scope,
    };

    let constraint = instantiate_constraint(
        &prepared.constraint_spec,
        prepared.gate,
        model,
        prepared.think_budget.as_ref(),
    );
    let (tx, rx) = unbounded_channel();
    let gen_req = GenRequest {
        prompt: prepared.engine_prompt,
        max_tokens: prepared.max_tokens,
        sampler: prepared.sampler,
        stop_tokens: prepared.stop_tokens,
        events: tx,
        mm_chunks: prepared.mm_chunks,
        constraint,
        logprobs: None,
        submitted: None, // stamped by Engine::submit
    };
    if let Err(e) = model.engine.submit(gen_req) {
        return err(StatusCode::INTERNAL_SERVER_ERROR, "api_error", e);
    }

    if req.stream {
        stream_response(ctx, rx)
    } else {
        collect_response(ctx, rx).await
    }
}

/// Finished-turn content blocks + stop_reason/stop_sequence.
fn finish_blocks(
    ctx: &Ctx,
    parsed: &Parsed,
    finish: Option<FinishReason>,
) -> (Vec<Value>, &'static str, Value) {
    let mut blocks = Vec::new();
    if let Some(reasoning) = &parsed.reasoning {
        // display "omitted": the block stays, the text is withheld
        let text = if ctx.omit_thinking {
            ""
        } else {
            reasoning.as_str()
        };
        blocks.push(json!({"type": "thinking", "thinking": text, "signature": ""}));
    }
    let mut stop_seq = Value::Null;
    let mut stop_hit = false;
    if let Some(content) = &parsed.content {
        let text = match find_stop(content, &ctx.stop_strings) {
            Some((i, s)) => {
                stop_hit = true;
                stop_seq = json!(s);
                &content[..i]
            }
            None => content.as_str(),
        };
        if !text.is_empty() {
            blocks.push(json!({"type": "text", "text": text}));
        }
    }
    for tc in &parsed.tool_calls {
        let input = serde_json::from_str::<Value>(&tc.arguments).unwrap_or_else(|_| json!({}));
        blocks.push(json!({
            "type": "tool_use",
            "id": format!("toolu_{}", uuid::Uuid::new_v4().simple()),
            "name": tc.name,
            "input": input,
        }));
    }
    let stop_reason = if !parsed.tool_calls.is_empty() {
        "tool_use"
    } else if stop_hit {
        "stop_sequence"
    } else {
        match finish {
            Some(FinishReason::Length) => "max_tokens",
            _ => "end_turn",
        }
    };
    (blocks, stop_reason, stop_seq)
}

async fn collect_response(mut ctx: Ctx, mut rx: UnboundedReceiver<TokenEvent>) -> Response {
    let mut ids = Vec::new();
    let mut finish = None;
    let mut cached = 0usize;
    while let Some(ev) = rx.recv().await {
        match ev {
            // rows = what the engine actually prefilled; on an image request
            // that is the picture's expanded row run, not the single <image>
            // token the prompt tokenized to (see TokenEvent::Prefilled)
            TokenEvent::Prefilled { cached: c, rows } => {
                cached = c as usize;
                ctx.prompt_len = ctx.prompt_len.max(rows as usize);
            }
            TokenEvent::Token { id: t, .. } => {
                ids.push(t);
                // early exit on a stop sequence in visible content
                if !ctx.stop_strings.is_empty() {
                    let parsed = ctx.parse(&ids);
                    if let Some(content) = &parsed.content
                        && find_stop(content, &ctx.stop_strings).is_some()
                    {
                        break;
                    }
                }
            }
            TokenEvent::Done(r, stats) => {
                finish = Some(r);
                ctx.scope.phases(&stats);
                break;
            }
            TokenEvent::Error(e) => {
                return engine_err(&e);
            }
        }
    }
    let parsed = ctx.parse(&ids);
    let (mut content, stop_reason, stop_seq) = finish_blocks(&ctx, &parsed, finish);
    ctx.scope.usage(ctx.prompt_len, ids.len());
    ctx.scope.cached(cached);
    ctx.scope.finish(stop_reason);
    let mut usage = json!({"input_tokens": ctx.prompt_len, "output_tokens": ids.len()});
    if cached > 0 {
        // truthful: those prompt tokens were served from the prefix cache
        // (Paddock's caching is implicit - no cache_control needed)
        usage["cache_read_input_tokens"] = json!(cached);
    }
    if let Some(c) = &ctx.compaction {
        // the compaction block leads the content; top-level usage stays the
        // final message's (per spec), iterations carries both passes
        content.insert(0, compaction_block(&c.summary));
        usage["iterations"] = json!([c.usage, {
            "type": "message", "model": ctx.model_id,
            "input_tokens": ctx.prompt_len, "output_tokens": ids.len(),
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": cached,
        }]);
    }
    let mut body = json!({
        "id": ctx.id,
        "type": "message",
        "role": "assistant",
        "model": ctx.model_id,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": stop_seq,
        "usage": usage,
    });
    if let Some(cm) = &ctx.context_management {
        body["context_management"] = cm.clone();
    }
    if let Some(o) = ctx.ocr_json(&ids) {
        body["ocr"] = o;
    }
    Json(body).into_response()
}

/// Streaming: the Anthropic event protocol. Thinking and text stream as
/// deltas; tool_use blocks emit atomically on completion (one
/// input_json_delta with the full arguments - fragmenting a non-JSON
/// dialect's arguments is not prefix-stable, same policy as chat).
fn stream_response(mut ctx: Ctx, mut rx: UnboundedReceiver<TokenEvent>) -> Response {
    let start_input_tokens = ctx.prompt_len;
    let sse = stream! {
        yield ev("message_start", json!({"type": "message_start", "message": {
            "id": ctx.id, "type": "message", "role": "assistant",
            "model": ctx.model_id, "content": [],
            "stop_reason": null, "stop_sequence": null,
            "usage": {"input_tokens": ctx.prompt_len, "output_tokens": 0},
        }}));
        // keep-alive ping after message_start, like the live API's streams
        yield ev("ping", json!({"type": "ping"}));

        let mut index = 0usize;      // current block index
        if let Some(c) = &ctx.compaction {
            // SDK-pinned event shape: start, one compaction_delta carrying
            // the whole summary (the accumulator assigns, never appends), stop
            yield ev("content_block_start", json!({
                "type": "content_block_start", "index": index,
                "content_block": {"type": "compaction", "content": null}}));
            yield ev("content_block_delta", json!({
                "type": "content_block_delta", "index": index,
                "delta": {"type": "compaction_delta", "content": c.summary}}));
            yield ev("content_block_stop", json!({"type": "content_block_stop", "index": index}));
            index += 1;
        }
        let mut think_open = false;
        let mut text_open = false;
        let mut think_emitted = 0usize;
        let mut text_emitted = 0usize;
        let mut ids: Vec<u32> = Vec::new();
        // incremental decode of `ids` (the O(n^2) per-token full re-decode
        // was the long-stream collapse under concurrency)
        let mut sd = ctx.tokenizer.stream_decoder(false);
        let mut finish = None;
        let mut cached = 0usize;

        loop {
            match rx.recv().await {
                Some(TokenEvent::Prefilled { cached: c, rows }) => {
                    cached = c as usize;
                    ctx.prompt_len = ctx.prompt_len.max(rows as usize);
                }
                Some(TokenEvent::Token { id: t, .. }) => {
                    ids.push(t);
                    let raw = sd.push(&ctx.tokenizer, t);
                    let parsed = ctx.parse_raw(&raw);

                    if let Some(reasoning) = &parsed.reasoning {
                        let safe = safe_emit_len(reasoning, ctx.dialect.reasoning_markers(), &[]);
                        if safe > think_emitted {
                            if !think_open {
                                think_open = true;
                                yield ev("content_block_start", json!({
                                    "type": "content_block_start", "index": index,
                                    "content_block": {"type": "thinking", "thinking": "", "signature": ""}}));
                            }
                            let delta = reasoning[think_emitted..safe].to_owned();
                            think_emitted = safe;
                            // display "omitted": the block opens and closes,
                            // but no thinking text goes over the wire
                            if !ctx.omit_thinking {
                                yield ev("content_block_delta", json!({
                                    "type": "content_block_delta", "index": index,
                                    "delta": {"type": "thinking_delta", "thinking": delta}}));
                            }
                        }
                    }

                    if let Some(content) = &parsed.content {
                        let (cut, hit) = match find_stop(content, &ctx.stop_strings) {
                            Some((i, _)) => (i, true),
                            None => (
                                safe_emit_len(content, ctx.dialect.content_markers(), &ctx.stop_strings),
                                false,
                            ),
                        };
                        if cut > text_emitted || hit {
                            if think_open && !text_open {
                                yield ev("content_block_stop", json!({
                                    "type": "content_block_stop", "index": index}));
                                index += 1;
                            }
                            if !text_open {
                                text_open = true;
                                yield ev("content_block_start", json!({
                                    "type": "content_block_start", "index": index,
                                    "content_block": {"type": "text", "text": ""}}));
                            }
                            if cut > text_emitted {
                                let delta = content[text_emitted..cut].to_owned();
                                text_emitted = cut;
                                yield ev("content_block_delta", json!({
                                    "type": "content_block_delta", "index": index,
                                    "delta": {"type": "text_delta", "text": delta}}));
                            }
                        }
                        if hit {
                            break;
                        }
                    }
                }
                Some(TokenEvent::Done(r, stats)) => { finish = Some(r); ctx.scope.phases(&stats); break }
                None => break,
                Some(TokenEvent::Error(e)) => {
                    yield ev("error", json!({"type": "error",
                        "error": {"type": anthropic_kind(e.class), "message": e.message}}));
                    return;
                }
            }
        }

        let parsed = ctx.parse(&ids);
        let (_, stop_reason, stop_seq) = finish_blocks(&ctx, &parsed, finish);

        // close whichever streaming block is open
        if think_open && !text_open {
            yield ev("content_block_stop", json!({"type": "content_block_stop", "index": index}));
            index += 1;
        }
        if text_open {
            yield ev("content_block_stop", json!({"type": "content_block_stop", "index": index}));
            index += 1;
        }

        // tool_use blocks, atomically
        for tc in &parsed.tool_calls {
            let id = format!("toolu_{}", uuid::Uuid::new_v4().simple());
            yield ev("content_block_start", json!({
                "type": "content_block_start", "index": index,
                "content_block": {"type": "tool_use", "id": id, "name": tc.name, "input": {}}}));
            yield ev("content_block_delta", json!({
                "type": "content_block_delta", "index": index,
                "delta": {"type": "input_json_delta", "partial_json": tc.arguments}}));
            yield ev("content_block_stop", json!({"type": "content_block_stop", "index": index}));
            index += 1;
        }

        ctx.scope.usage(ctx.prompt_len, ids.len());
        ctx.scope.cached(cached);
        ctx.scope.finish(stop_reason);
        let mut usage = json!({"output_tokens": ids.len()});
        if cached > 0 {
            usage["cache_read_input_tokens"] = json!(cached);
        }
        // message_start went out before the engine had prefilled anything, so
        // its input_tokens could only be the TOKENIZED length - one `<image>`
        // per picture. Once the prefill reports its real row count, restate it
        // here: message_delta's usage is the cumulative one, and leaving the
        // stream's only input figure short by an image's ~500 rows is the kind
        // of silent under-report a billing or context-budget client acts on.
        if ctx.prompt_len > start_input_tokens {
            usage["input_tokens"] = json!(ctx.prompt_len);
        }
        if let Some(c) = &ctx.compaction {
            usage["iterations"] = json!([c.usage, {
                "type": "message", "model": ctx.model_id,
                "input_tokens": ctx.prompt_len, "output_tokens": ids.len(),
                "cache_creation_input_tokens": 0, "cache_read_input_tokens": cached,
            }]);
        }
        let mut delta_ev = json!({
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": stop_seq},
            "usage": usage,
        });
        if let Some(cm) = &ctx.context_management {
            // the API contract reports applied edits on the message_delta
            // event in streams, same shape as the non-streaming body
            delta_ev["context_management"] = cm.clone();
        }
        if let Some(o) = ctx.ocr_json(&ids) {
            delta_ev["ocr"] = o;
        }
        yield ev("message_delta", delta_ev);
        yield ev("message_stop", json!({"type": "message_stop"}));
    };
    Sse::new(sse).into_response()
}

/// One summarization generation over the compact span of `messages`, shared
/// by the plain two-iteration path and the agent loops' round-0 pass.
struct SummaryPass {
    /// None = failed compaction (the model produced nothing usable).
    summary: Option<String>,
    /// This pass's `usage.iterations` entry, SDK shape.
    usage: Value,
    input_tokens: usize,
    output_tokens: usize,
    cached: usize,
}

/// Render [same system, same tools, span..., instructions] and generate the
/// summary. Same-prefix-as-the-conversation is the whole point: the
/// summarization prefill then rides the radix cache of the very conversation
/// being compacted. Err is a ready-to-return error response - including the
/// over-window case, where compaction cannot rescue the conversation and the
/// loud error stands (the Anthropic dialect has no fail-open backstop; its
/// Responses twin does, hence `summary_pass(lenient_gate)` there).
// Err is axum's own Response by design: a ready-to-return body, not a boxed error
#[allow(clippy::result_large_err)]
async fn anth_summary_pass(
    state: &Arc<AppState>,
    model: &ServingModel,
    req: &MessagesRequest,
    messages: &[Value],
    instructions: Option<&str>,
    scope: &crate::events::EventScope,
) -> Result<SummaryPass, Response> {
    let tail_start = crate::context_management::compact_tail_start(messages);
    let mut msgs1: Vec<Value> = messages[..tail_start].to_vec();
    let instructions =
        instructions.unwrap_or(crate::context_management::DEFAULT_COMPACT_INSTRUCTIONS);
    msgs1.push(json!({"role": "user", "content": instructions}));
    // same tools in the render (minus a tool_choice "none") - anything else
    // and the span prefix diverges from the live conversation's, losing the
    // radix hit that makes the summarization prefill nearly free
    let tools = if req
        .tool_choice
        .as_ref()
        .and_then(|tc| tc.get("type"))
        .and_then(Value::as_str)
        == Some("none")
    {
        None
    } else {
        match req.tools.as_ref().map(|ts| convert_tools(ts)).transpose() {
            Ok(t) => t,
            Err(e) => return Err(bad(e)),
        }
    };
    // thinking off for the summary (suffix-only in every template family, so
    // the span prefix still matches): the budget goes to the summary itself,
    // and an effort rung is passed as None for the same reason - grading a
    // thought process this pass is not rendering would be incoherent
    let (p1_ids, p1_engine, p1_think_open, p1_mm, _) = match render_prompt(
        model,
        req.system.as_ref(),
        &msgs1,
        tools.as_deref(),
        None,
        None,
        None,
    ) {
        Ok(r) => r,
        Err(e) => return Err(bad(e)),
    };
    // even the span + instructions overflows: compaction cannot rescue this
    // conversation, and the loud over-window error stands
    if let Some(e) =
        crate::chat::context_gate(model, p1_engine.len(), p1_mm.as_deref(), state.max_ctx)
    {
        return Err(engine_err(&e));
    }
    // greedy-class: a deterministic summary, no sampling surprises. The
    // truncation knobs still come from the model's election - at temperature
    // 0 they cannot change the argmax, so this is about staying consistent
    // with the rest of the server rather than about the draw.
    let dflt = state.sampling.resolve(true);
    let sampler1 = SamplingParams {
        temperature: 0.0,
        top_k: dflt.top_k,
        top_p: dflt.top_p,
        min_p: dflt.min_p,
        repeat_penalty: dflt.repeat_penalty,
        repeat_last_n: state.sampling.repeat_last_n,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        seed: state.sampling.seed_or_now(req.seed),
        logit_bias: Vec::new(),
        no_repeat_ngram: (0, 0),
    };
    // bounded: a summary, not an essay (the pass is billed via usage.iterations)
    let max1 = state.max_output_ceiling.map_or(4096, |c| c.min(4096));
    let (tx, mut rx) = unbounded_channel();
    let gen1 = GenRequest {
        prompt: p1_engine,
        max_tokens: max1,
        sampler: sampler1,
        stop_tokens: model.stop_tokens.clone(),
        events: tx,
        mm_chunks: p1_mm,
        constraint: instantiate_constraint(&ConstraintSpec::None, GateSpec::Immediate, model, None),
        logprobs: None,
        submitted: None, // stamped by Engine::submit
    };
    if let Err(e) = model.engine.submit(gen1) {
        return Err(err(StatusCode::INTERNAL_SERVER_ERROR, "api_error", e));
    }
    let mut ids1: Vec<u32> = Vec::new();
    let mut cached1 = 0usize;
    let mut p1_len = p1_ids.len();
    while let Some(evt) = rx.recv().await {
        match evt {
            TokenEvent::Prefilled { cached: c, rows } => {
                cached1 = c as usize;
                p1_len = p1_len.max(rows as usize);
            }
            TokenEvent::Token { id: t, .. } => ids1.push(t),
            TokenEvent::Done(_, stats) => {
                scope.phases(&stats);
                break;
            }
            TokenEvent::Error(e) => return Err(engine_err(&e)),
        }
    }
    let raw = model.tokenizer.decode(&ids1, false).unwrap_or_default();
    let parsed = parse(model.dialect, &raw, p1_think_open, None);
    // None = failed compaction: the block reports null content (SDK pin) and
    // the request proceeds UNCOMPACTED - if that overflows, the loud
    // over-window error stands; never a silent trim
    let summary = parsed
        .content
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned);
    let usage = json!({
        "type": "compaction",
        "input_tokens": p1_len, "output_tokens": ids1.len(),
        "cache_creation_input_tokens": 0, "cache_read_input_tokens": cached1,
    });
    Ok(SummaryPass {
        summary,
        usage,
        input_tokens: p1_len,
        output_tokens: ids1.len(),
        cached: cached1,
    })
}

/// Round-0 context management for the MCP/web-search agent loops (
/// phase 5). The compact_20260112 orchestration runs once, here, on the
/// request's own prompt - the same measurement point a non-agent request uses
/// - and the compact edit is then dropped from the config so the loop's
///   per-round `prepare` cannot re-fire it. Three reasons that pin is right:
/// - the compact SPAN is fixed for the whole loop (the loop appends assistant
///   tool_use / user tool_result turns after the pending user message, which
///   is where the tail starts), so compacting at round 5 would remove exactly
///   what compacting here removes;
/// - the block then leads the content, which is where the spec'd resend
///   rewrite ("everything before the compaction block collapses into its
///   summary") is true of our summary - a block emitted mid-loop would tell
///   the client to drop tool round-trips the summary never saw;
/// - mid-turn tool bloat is `clear_tool_uses_20250919`'s job, and that one
///   does run every round.
///   Ok(Some) = compacted, `req.messages` replaced; Ok(None) = nothing to do;
///   Err = a ready-to-return response (an error, or the pause_after_compaction
///   answer, which is a complete response by definition).
// Err is axum's own Response by design: a ready-to-return body, not a boxed error
#[allow(clippy::result_large_err)]
async fn precompact_agent(
    state: &Arc<AppState>,
    req: &mut MessagesRequest,
    scope: &crate::events::EventScope,
) -> Result<Option<CompactionOut>, Response> {
    if req.context_management.is_none() {
        return Ok(None);
    }
    let Some(model) = state.serving.as_ref() else {
        return Ok(None); // the loop reports "no model is loaded" in its own dialect
    };
    // one extra render+tokenize, and only for a request that configured
    // context management: the loop's round 0 re-prepares over the result
    let mut prepared =
        prepare(model, req, state.max_output_ceiling, &state.sampling).map_err(bad)?;
    let compact = prepared.compact.take();
    drop_compact_edit(&mut req.context_management);
    let Some(plan) = compact else { return Ok(None) };
    if plan.pause {
        // "compact and stop" - no tool can contribute to a summary, so the
        // plain orchestration answers this verbatim (stream and non-stream)
        let omit = thinking_omitted(req.thinking.as_ref()).unwrap_or(false);
        let cm = prepared.context_management.clone();
        return Err(
            run_compacting(state.clone(), req.clone(), plan, cm, omit, scope.clone()).await,
        );
    }
    let pass = anth_summary_pass(
        state,
        model,
        req,
        &plan.messages,
        plan.instructions.as_deref(),
        scope,
    )
    .await?;
    // a failed summary (null content) leaves the conversation uncompacted and
    // the block reports it - the loop still runs, and an over-window prompt
    // stays the loud error it always was
    req.messages = match &pass.summary {
        Some(s) => crate::context_management::compacted_messages(&plan.messages, s),
        None => plan.messages,
    };
    Ok(Some(CompactionOut {
        summary: pass.summary,
        usage: pass.usage,
    }))
}

/// Remove the compact edit from a config after the round-0 pass, so the agent
/// loops' per-round `prepare` cannot fire the trigger mid-turn (see the pin on
/// `precompact_agent`). The clear_* strategies stay: they are per-round by
/// design, and their `applied_edits` report still rides the final response.
fn drop_compact_edit(cm: &mut Option<Value>) {
    if let Some(Value::Object(o)) = cm.as_mut()
        && let Some(Value::Array(edits)) = o.get_mut("edits")
    {
        edits.retain(|e| e.get("type").and_then(Value::as_str) != Some("compact_20260112"));
    }
}

/// `usage.iterations` for an agent turn that compacted: the summarization
/// pass, then the final message. Top-level usage stays the message's own, per
/// spec - iterations is where the extra generation is billed.
fn agent_iterations(
    c: &CompactionOut,
    model_id: &str,
    prompt_len: usize,
    out_tokens: usize,
    cached: usize,
) -> Value {
    json!([c.usage, {
        "type": "message", "model": model_id,
        "input_tokens": prompt_len, "output_tokens": out_tokens,
        "cache_creation_input_tokens": 0, "cache_read_input_tokens": cached,
    }])
}

/// The compact_20260112 orchestration (plan:
/// Two generations on the same
/// engine, sequentially: iteration 1 summarizes the compact span, iteration 2
/// is the real request over [framed summary + pending turn] - or, with
/// pause_after_compaction, the response stops at the block with
/// stop_reason "compaction". Cache economics are the design: iteration 1's
/// prefill rides the radix cache of the very conversation being compacted,
/// and the compacted conversation equals what a resend rewrites to, so every
/// later turn is a radix hit. Errors before the SSE stream starts return
/// clean HTTP errors even for stream=true (nothing has been committed yet).
async fn run_compacting(
    state: Arc<AppState>,
    req: MessagesRequest,
    plan: CompactPlan,
    cm_report: Option<Value>,
    omit_thinking: bool,
    scope: crate::events::EventScope,
) -> Response {
    let Some(model) = state.serving.as_ref() else {
        return err(
            StatusCode::SERVICE_UNAVAILABLE,
            "overloaded_error",
            "no model is loaded",
        );
    };

    // ── iteration 1: the summarization pass ─────────────────────────────
    let pass = match anth_summary_pass(
        &state,
        model,
        &req,
        &plan.messages,
        plan.instructions.as_deref(),
        &scope,
    )
    .await
    {
        Ok(p) => p,
        Err(r) => return r,
    };
    let SummaryPass {
        summary,
        usage: usage1,
        input_tokens: p1_len,
        output_tokens: out1,
        cached: cached1,
    } = pass;

    // ── pause_after_compaction: the block is the response ───────────────
    if plan.pause {
        scope.usage(p1_len, out1);
        scope.cached(cached1);
        scope.finish("compaction");
        let id = format!("msg_{}", uuid::Uuid::new_v4().simple());
        let block = compaction_block(&summary);
        if req.stream {
            let model_id = model.id.clone();
            let cm = cm_report.clone();
            let sse = stream! {
                yield ev("message_start", json!({"type": "message_start", "message": {
                    "id": id, "type": "message", "role": "assistant",
                    "model": model_id, "content": [],
                    "stop_reason": null, "stop_sequence": null,
                    "usage": {"input_tokens": p1_len, "output_tokens": 0},
                }}));
                yield ev("ping", json!({"type": "ping"}));
                yield ev("content_block_start", json!({
                    "type": "content_block_start", "index": 0,
                    "content_block": {"type": "compaction", "content": null}}));
                yield ev("content_block_delta", json!({
                    "type": "content_block_delta", "index": 0,
                    "delta": {"type": "compaction_delta", "content": summary}}));
                yield ev("content_block_stop", json!({"type": "content_block_stop", "index": 0}));
                let mut usage = json!({"output_tokens": out1, "iterations": [usage1]});
                if cached1 > 0 {
                    usage["cache_read_input_tokens"] = json!(cached1);
                }
                let mut delta_ev = json!({
                    "type": "message_delta",
                    "delta": {"stop_reason": "compaction", "stop_sequence": null},
                    "usage": usage,
                });
                if let Some(cm) = &cm {
                    delta_ev["context_management"] = cm.clone();
                }
                yield ev("message_delta", delta_ev);
                yield ev("message_stop", json!({"type": "message_stop"}));
            };
            return Sse::new(sse).into_response();
        }
        let mut usage = json!({
            "input_tokens": p1_len, "output_tokens": out1,
            "iterations": [usage1],
        });
        if cached1 > 0 {
            usage["cache_read_input_tokens"] = json!(cached1);
        }
        let mut body = json!({
            "id": id, "type": "message", "role": "assistant", "model": model.id,
            "content": [block], "stop_reason": "compaction", "stop_sequence": null,
            "usage": usage,
        });
        if let Some(cm) = &cm_report {
            body["context_management"] = cm.clone();
        }
        return Json(body).into_response();
    }

    // ── iteration 2: the real generation over the compacted turn ───────
    let msgs2 = match &summary {
        Some(s) => crate::context_management::compacted_messages(&plan.messages, s),
        None => plan.messages.clone(),
    };
    let mut req2 = req.clone();
    req2.messages = msgs2;
    req2.context_management = None;
    let prepared = match prepare(model, &req2, state.max_output_ceiling, &state.sampling) {
        Ok(p) => p,
        Err(e) => return bad(e),
    };
    if let Some(e) = crate::chat::context_gate(
        model,
        prepared.engine_prompt.len(),
        prepared.mm_chunks.as_deref(),
        state.max_ctx,
    ) {
        return engine_err(&e);
    }
    let ctx = Ctx {
        id: format!("msg_{}", uuid::Uuid::new_v4().simple()),
        model_id: model.id.clone(),
        tokenizer: model.tokenizer.clone(),
        prompt_len: prepared.prompt_ids.len(),
        dialect: model.dialect,
        thinking_open: prepared.thinking_open,
        hints: prepared.hints.clone(),
        single_tool_call: prepared.single_tool_call,
        stop_strings: prepared.stop_strings.clone(),
        omit_thinking,
        context_management: cm_report,
        compaction: Some(CompactionOut {
            summary,
            usage: usage1,
        }),
        ocr: prepared.ocr.clone(),
        scope,
    };
    let constraint = instantiate_constraint(
        &prepared.constraint_spec,
        prepared.gate,
        model,
        prepared.think_budget.as_ref(),
    );
    let (tx, rx) = unbounded_channel();
    let gen2 = GenRequest {
        prompt: prepared.engine_prompt,
        max_tokens: prepared.max_tokens,
        sampler: prepared.sampler,
        stop_tokens: prepared.stop_tokens,
        events: tx,
        mm_chunks: prepared.mm_chunks,
        constraint,
        logprobs: None,
        submitted: None, // stamped by Engine::submit
    };
    if let Err(e) = model.engine.submit(gen2) {
        return err(StatusCode::INTERNAL_SERVER_ERROR, "api_error", e);
    }
    if req.stream {
        stream_response(ctx, rx)
    } else {
        collect_response(ctx, rx).await
    }
}

// ── MCP connector (beta mcp-client-2025-11-20; the deprecated 2025-04-04
// inline `tool_configuration` shape keeps working) ──────────────────────────
//
// The round count, the per-round ceiling and the repeat ledger are shared with
// every other agent loop in `loop_budget` - this file used to keep its own
// MCP_MAX_ROUNDS, and four private copies of one number is how the dialects
// drift apart.

/// Tool configuration for one server from the current connector's
/// `mcp_toolset` entries in the `tools` array (beta mcp-client-2025-11-20).
#[derive(Default)]
struct McpToolset {
    default_enabled: Option<bool>,
    default_defer: Option<bool>,
    /// tool name -> (enabled, defer_loading) overrides.
    tools: HashMap<String, (Option<bool>, Option<bool>)>,
}

impl McpToolset {
    /// Per-tool resolution with the documented precedence: `configs` entry,
    /// then `default_config`, then the system defaults (enabled, not deferred).
    fn resolve(&self, tool: &str) -> (bool, bool) {
        let o = self.tools.get(tool);
        (
            o.and_then(|(e, _)| *e)
                .or(self.default_enabled)
                .unwrap_or(true),
            o.and_then(|(_, d)| *d)
                .or(self.default_defer)
                .unwrap_or(false),
        )
    }
}

/// Pull `mcp_toolset` entries out of the `tools` array - the 2025-11-20
/// connector puts per-server tool configuration there, not on the server
/// entry. A server referenced by no toolset serves all its tools, which also
/// keeps the deprecated inline `tool_configuration` requests working.
/// `cache_control` on a toolset is accepted; there is no prompt cache to hint.
fn extract_mcp_toolsets(req: &mut MessagesRequest) -> Result<HashMap<String, McpToolset>, String> {
    let has_servers = req.mcp_servers.as_ref().is_some_and(|s| !s.is_empty());
    let mut sets: HashMap<String, McpToolset> = HashMap::new();
    let emptied = {
        let Some(tools) = req.tools.as_mut() else {
            return Ok(sets);
        };
        let mut rest = Vec::with_capacity(tools.len());
        for t in tools.drain(..) {
            if t.get("type").and_then(Value::as_str) != Some("mcp_toolset") {
                rest.push(t);
                continue;
            }
            if !has_servers {
                return Err("an mcp_toolset tool requires mcp_servers".into());
            }
            let server = t
                .get("mcp_server_name")
                .and_then(Value::as_str)
                .ok_or("each mcp_toolset needs mcp_server_name")?
                .to_owned();
            let mut set = McpToolset::default();
            if let Some(d) = t.get("default_config") {
                set.default_enabled = d.get("enabled").and_then(Value::as_bool);
                set.default_defer = d.get("defer_loading").and_then(Value::as_bool);
            }
            if let Some(cfgs) = t.get("configs").and_then(Value::as_object) {
                for (tool, c) in cfgs {
                    set.tools.insert(
                        tool.clone(),
                        (
                            c.get("enabled").and_then(Value::as_bool),
                            c.get("defer_loading").and_then(Value::as_bool),
                        ),
                    );
                }
            }
            if sets.insert(server.clone(), set).is_some() {
                return Err(format!(
                    "mcp server {server:?} is referenced by more than one mcp_toolset"
                ));
            }
        }
        *tools = rest;
        tools.is_empty()
    };
    if emptied {
        req.tools = None;
    }
    Ok(sets)
}

/// Connect the request's `mcp_servers` (lazy HTTP) and gather their tools: the
/// Anthropic tool defs to inject, a routing map `tool_name -> (cfg, server_name)`,
/// and the searchable catalog (for progressive disclosure - see below).
/// Per-server tool config comes from the 2025-11-20 `mcp_toolset` entries in
/// `toolsets`, with the deprecated inline `tool_configuration` as fallback.
async fn gather_mcp_servers(
    state: &AppState,
    servers: &[Value],
    toolsets: &HashMap<String, McpToolset>,
) -> Result<
    (
        Vec<Value>,
        HashMap<String, (paddock_mcp::ServerConfig, String)>,
        Vec<crate::tool_search::CatalogTool>,
        // Each server's handshake `instructions` - spec-intended system-prompt
        // material this path also used to discard.
        Vec<String>,
    ),
    String,
> {
    use crate::tool_search::{self, CatalogTool};
    use paddock_mcp::{ServerConfig, Transport};
    // Kept per SERVER: disclosure is decided one server at a time (see below),
    // so these cannot be flattened until that decision is made.
    let mut per_server: Vec<(String, Vec<Value>)> = Vec::new();
    let mut routing: HashMap<String, (ServerConfig, String)> = HashMap::new();
    let mut catalog: Vec<CatalogTool> = Vec::new();
    let mut instructions: Vec<String> = Vec::new();
    let mut any_deferred = false;
    let mut seen: Vec<&str> = Vec::new();
    for s in servers {
        let name = s
            .get("name")
            .and_then(Value::as_str)
            .ok_or("each mcp_servers entry needs a name")?;
        seen.push(name);
        // A name-only entry resolves against the launch registry
        // (PADDOCK_MCP_SERVERS, expanded by the manager at spawn) - Anthropic
        // API callers get the box's registered servers exactly like the
        // Studio does. An explicit url always wins.
        let reg = state
            .live
            .snapshot()
            .mcp_servers
            .iter()
            .find(|e| e.get("server_label").and_then(Value::as_str) == Some(name))
            .cloned();
        let url = s
            .get("url")
            .and_then(Value::as_str)
            .or_else(|| {
                reg.as_ref()
                    .and_then(|e| e.get("server_url"))
                    .and_then(Value::as_str)
            })
            .map(String::from);
        let toolset = toolsets.get(name);
        let tool_cfg = s.get("tool_configuration");
        if tool_cfg
            .and_then(|c| c.get("enabled"))
            .and_then(Value::as_bool)
            == Some(false)
        {
            continue;
        }
        let allowed: Option<Vec<String>> = tool_cfg
            .and_then(|c| c.get("allowed_tools"))
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            });
        let mut headers = std::collections::HashMap::new();
        if let Some(tok) = s.get("authorization_token").and_then(Value::as_str) {
            headers.insert("Authorization".to_string(), format!("Bearer {tok}"));
        } else if let Some(h) = reg
            .as_ref()
            .and_then(|e| e.get("headers"))
            .and_then(Value::as_object)
        {
            for (k, v) in h {
                if let Some(vs) = v.as_str() {
                    headers.insert(k.clone(), vs.to_string());
                }
            }
        }
        // HTTP when a url exists (inline or registry); else a registry stdio
        // command; else an honest error.
        let transport = if let Some(url) = url {
            Transport::Http { url, headers }
        } else if let Some(command) = reg
            .as_ref()
            .and_then(|e| e.get("command"))
            .and_then(Value::as_str)
        {
            Transport::Stdio {
                command: command.to_string(),
                args: reg
                    .as_ref()
                    .and_then(|e| e.get("args"))
                    .and_then(Value::as_array)
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default(),
                env: crate::responses::str_map(reg.as_ref().and_then(|e| e.get("env"))),
            }
        } else {
            return Err(
                "each mcp_servers entry needs a url (or a name this runner's registry knows)"
                    .into(),
            );
        };
        let cfg = ServerConfig {
            id: format!("anthropic:{name}"),
            label: name.to_string(),
            transport,
        };
        let mtools = state
            .mcp
            .list_tools(&cfg)
            .await
            .map_err(|e| format!("mcp server {name:?}: {e}"))?;
        let mut server_defs: Vec<Value> = Vec::new();
        for mt in mtools {
            if let Some(allow) = &allowed
                && !allow.iter().any(|a| a == &mt.name)
            {
                continue;
            }
            let (enabled, defer) = toolset
                .map(|t| t.resolve(&mt.name))
                .unwrap_or((true, false));
            if !enabled {
                continue;
            }
            // deferred tools stay out of the direct injection but remain
            // discoverable through the search tools (catalog + routing)
            if !defer {
                server_defs.push(json!({
                    "name": mt.name,
                    "description": mt.description,
                    "input_schema": mt.input_schema,
                }));
            } else {
                any_deferred = true;
            }
            catalog.push(CatalogTool {
                name: mt.name.clone(),
                description: mt.description.clone().unwrap_or_default(),
                input_schema: mt.input_schema.clone(),
            });
            routing.insert(mt.name.clone(), (cfg.clone(), name.to_string()));
        }
        if let Some(instr) = state.mcp.instructions(&cfg).await {
            instructions.push(instr);
        }
        match per_server.iter_mut().find(|(l, _)| l == name) {
            Some((_, defs)) => defs.extend(server_defs),
            None => per_server.push((name.to_string(), server_defs)),
        }
    }
    if let Some(unknown) = toolsets.keys().find(|k| !seen.contains(&k.as_str())) {
        return Err(format!(
            "mcp_toolset references unknown mcp server {unknown:?}"
        ));
    }

    // Progressive disclosure, per SERVER (same policy as the Responses path):
    // spend the budget smallest-first and hide only the servers that are
    // actually big, so a small server keeps real schemas - and with them the
    // argument grammar, which cannot reach anything routed through
    // `mcp_call_tool`. Deferred tools are already out of `server_defs`, so they
    // go behind search whatever their server's size.
    let weights: Vec<tool_search::ServerWeight> = per_server
        .iter()
        .map(|(label, defs)| tool_search::ServerWeight {
            label: label.clone(),
            tools: defs.len(),
            chars: defs.iter().map(|d| d.to_string().chars().count()).sum(),
        })
        .collect();
    let shown = tool_search::disclose_servers(&weights, state.max_ctx);
    let mut tools: Vec<Value> = Vec::new();
    let mut hidden_labels: Vec<String> = Vec::new();
    let mut hidden_tools = 0usize;
    for (label, defs) in per_server {
        if shown.contains(&label) {
            tools.extend(defs);
        } else {
            hidden_tools += defs.len();
            hidden_labels.push(label);
        }
    }
    // Which notice depends on what actually happened to the schemas. Nothing
    // declared (every server hidden, or every tool deferred) is the all-hidden
    // text; a partial list names the servers that went missing.
    let _ = any_deferred;
    if !catalog.is_empty() {
        instructions.push(if tools.is_empty() {
            tool_search::SEARCH_MODE_INSTRUCTIONS.to_string()
        } else if hidden_labels.is_empty() {
            tool_search::SEARCH_AVAILABLE_INSTRUCTIONS.to_string()
        } else {
            tool_search::partial_mode_instructions(&hidden_labels, hidden_tools)
        });
    }
    // Always declare the search pair (see the twin comment in responses.rs):
    // searchability is not a mode, only the direct schemas are. This path used
    // to append it just for deferred tools, so the two dialects disagreed.
    if !catalog.is_empty() {
        tools.push(tool_search::search_tool_def_anthropic());
        tools.push(tool_search::call_tool_def_anthropic());
    }
    Ok((tools, routing, catalog, instructions))
}

/// A handled tool call resolved to what the Anthropic loops emit: the display
/// identity for the `mcp_tool_use` block plus the action to run.
enum AnthAction {
    Search {
        query: String,
        limit: usize,
    },
    Invoke {
        cfg: paddock_mcp::ServerConfig,
        real_name: String,
        real_args: Value,
    },
    Unknown {
        real_name: String,
    },
    /// Never dispatched: the call did not match its own schema, or the loop
    /// budget stopped a repeat. The message names what to fix, which is what
    /// makes the retry one round.
    Refuse {
        message: String,
    },
    /// Already run this turn with these exact arguments: the first
    /// result comes straight back and nothing is dispatched.
    Replay {
        output: String,
    },
}

struct AnthPlan {
    display_name: String,
    display_server: String,
    display_input: Value,
    action: AnthAction,
    /// Its slot in the turn's ledger - `Some` exactly when the call is going
    /// to run and its outcome has to be filed.
    sig: Option<crate::loop_budget::Signature>,
}

/// Take our tools out of the rendered set for the answer round: the
/// MCP tools, the two meta-tools, and web search. The caller's own tools stay
/// - handing a client tool call back ends the turn cleanly, and is a better
///   outcome than a forced answer.
fn strip_our_tools(
    req: &mut MessagesRequest,
    routing: &HashMap<String, (paddock_mcp::ServerConfig, String)>,
    web_on: bool,
    fetch_on: bool,
    forensics_on: bool,
    clock_on: bool,
) {
    // A FORCING tool_choice has to relax with them, whatever it named: `any`
    // over an emptied set compiles a grammar with nothing in it, and `tool`
    // naming an MCP tool we just removed cannot resolve at all - both 400 a
    // turn whose only remaining job is to answer. This round is already the
    // degraded path; ending it with an error would be the worst outcome of the
    // three.
    if matches!(
        req.tool_choice
            .as_ref()
            .and_then(|tc| tc.get("type"))
            .and_then(Value::as_str),
        Some("any" | "tool")
    ) {
        req.tool_choice = Some(json!({"type": "auto"}));
    }
    let Some(tools) = req.tools.take() else {
        return;
    };
    let kept: Vec<Value> = tools
        .into_iter()
        .filter(|t| {
            !is_handled_call(
                t.get("name").and_then(Value::as_str).unwrap_or(""),
                routing,
                web_on,
                fetch_on,
                forensics_on,
                clock_on,
            )
        })
        .collect();
    req.tools = if kept.is_empty() { None } else { Some(kept) };
}

/// Append instruction text to the conversation for the answer round.
///
/// It joins the last user turn when that turn is a block array - which after
/// a tool round it always is, holding the `tool_result` blocks - because two
/// consecutive user turns is a shape some chat templates render badly and
/// Anthropic's own wire rejects. Text after tool_result in the same turn is
/// exactly what their docs show.
fn append_user_text(messages: &mut Vec<Value>, text: &str) {
    if let Some(last) = messages.last_mut()
        && last.get("role").and_then(Value::as_str) == Some("user")
        && let Some(blocks) = last.get_mut("content").and_then(Value::as_array_mut)
    {
        blocks.push(json!({"type": "text", "text": text}));
        return;
    }
    messages.push(json!({"role": "user", "content": text}));
}

/// True when a tool call is one paddock handles internally (search / generic
/// executor / a routed MCP tool) rather than a client-side tool that ends the turn.
fn is_handled_call(
    name: &str,
    routing: &HashMap<String, (paddock_mcp::ServerConfig, String)>,
    web_on: bool,
    fetch_on: bool,
    forensics_on: bool,
    clock_on: bool,
) -> bool {
    name == crate::tool_search::SEARCH_TOOL
        || name == crate::tool_search::CALL_TOOL
        || (web_on && name == crate::websearch::TOOL_NAME)
        || (fetch_on && name == crate::websearch::FETCH_TOOL_NAME)
        || (forensics_on && name == crate::forensics::TOOL_NAME)
        || (clock_on && name == crate::clock::TOOL_NAME)
        || routing.contains_key(name)
}

/// Plan a handled call (no I/O): classify search / mcp_call_tool / direct and
/// resolve the routed server, so the loop can emit the `mcp_tool_use` block
/// before executing.
fn plan_anthropic_call(
    routing: &HashMap<String, (paddock_mcp::ServerConfig, String)>,
    catalog: &[crate::tool_search::CatalogTool],
    ledger: &mut crate::loop_budget::CallLedger,
    tc_name: &str,
    tc_args: &str,
) -> AnthPlan {
    use crate::tool_search::SEARCH_TOOL;
    // Drop a client-side namespace prefix ("functions.mcp_call_tool") unless the
    // routing already knows the name verbatim.
    let tc_name = if routing.contains_key(tc_name) {
        tc_name
    } else {
        crate::tool_search::strip_client_prefix(tc_name)
    };
    if tc_name == SEARCH_TOOL {
        let v: Value = serde_json::from_str(tc_args).unwrap_or(Value::Null);
        let query = v
            .get("query")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let limit = v
            .get("limit")
            .and_then(Value::as_u64)
            .unwrap_or(5)
            .clamp(1, 25) as usize;
        let plan = |action, sig| AnthPlan {
            display_name: SEARCH_TOOL.into(),
            display_server: "mcp".into(),
            display_input: v.clone(),
            action,
            sig,
        };
        // Discovery costs a whole round, so it has a budget of its own
        // on top of the repeat ledger: the ledger stops the same search twice,
        // this stops a fourth differently-worded one when every result has
        // already carried the complete index.
        if let Some(message) = ledger.search_budget_spent() {
            return plan(AnthAction::Refuse { message }, None);
        }
        // Normalized to what will actually run, so `{}` and `{"limit":5}` are
        // one query as far as the repeat ledger is concerned.
        return match ledger.check(
            SEARCH_TOOL,
            &json!({"query": query, "limit": limit}).to_string(),
        ) {
            (sig, crate::loop_budget::Verdict::Fresh) => {
                plan(AnthAction::Search { query, limit }, Some(sig))
            }
            (_, crate::loop_budget::Verdict::Replay(output)) => {
                plan(AnthAction::Replay { output }, None)
            }
            (_, crate::loop_budget::Verdict::Refuse(message)) => {
                plan(AnthAction::Refuse { message }, None)
            }
        };
    }
    // One seam, shared with the Responses dialect and the cloud loop: unwrap
    // the `mcp_call_tool` envelope when that is what this is, then check the
    // arguments against the target's schema before anything is dispatched.
    let (real_name, real_args) = match crate::tool_search::resolve_call(tc_name, tc_args, catalog) {
        crate::tool_search::Resolved::Call { name, arguments } => (
            name,
            serde_json::from_str::<Value>(&arguments).unwrap_or_else(|_| json!({})),
        ),
        crate::tool_search::Resolved::Refuse { name, message } => {
            let input = serde_json::from_str::<Value>(tc_args).unwrap_or_else(|_| json!({}));
            // A refusal never ran, so it stays out of the ledger.
            return AnthPlan {
                display_name: name,
                display_server: "mcp".into(),
                display_input: input,
                action: AnthAction::Refuse { message },
                sig: None,
            };
        }
    };
    // Judged on the RESOLVED identity: `mcp_call_tool{name:X,...}` and a direct
    // call to X are the same call, and a model alternating between the two
    // spellings while it loops is still looping. A replay keeps the server
    // it came from on its card, so the transcript reads as what happened.
    let routed = routing.get(&real_name).cloned();
    let display_server = routed
        .as_ref()
        .map_or_else(|| "mcp".to_string(), |(_, name)| name.clone());
    let (sig, verdict) = ledger.check(&real_name, &real_args.to_string());
    let (action, sig) = match verdict {
        crate::loop_budget::Verdict::Replay(output) => (AnthAction::Replay { output }, None),
        crate::loop_budget::Verdict::Refuse(message) => (AnthAction::Refuse { message }, None),
        crate::loop_budget::Verdict::Fresh => match routed {
            Some((cfg, _)) => (
                AnthAction::Invoke {
                    cfg,
                    real_name: real_name.clone(),
                    real_args: real_args.clone(),
                },
                Some(sig),
            ),
            None => (
                AnthAction::Unknown {
                    real_name: real_name.clone(),
                },
                Some(sig),
            ),
        },
    };
    AnthPlan {
        display_name: real_name,
        display_server,
        display_input: real_args,
        action,
        sig,
    }
}

/// Run a planned action, returning `(content_json, is_error, feedback_text)`.
async fn execute_anth_action(
    state: &AppState,
    catalog: &[crate::tool_search::CatalogTool],
    action: AnthAction,
) -> (Value, bool, String) {
    use crate::tool_search;
    match action {
        AnthAction::Search { query, limit } => {
            let hits = tool_search::search(catalog, &query, limit);
            let result = tool_search::search_result(&query, &hits, catalog);
            (json!(result.clone()), false, result)
        }
        AnthAction::Unknown { real_name } => {
            let m = format!(
                "unknown tool {real_name:?}; call {} to find available tools",
                tool_search::SEARCH_TOOL
            );
            (json!(m.clone()), true, m)
        }
        // Refused before dispatch - nothing reached a server, and the message
        // names what to fix or why the repeat stopped here.
        AnthAction::Refuse { message } => (json!(message.clone()), true, message),
        // The result of an identical call made earlier this turn.
        AnthAction::Replay { output } => (json!(output.clone()), false, output),
        AnthAction::Invoke {
            cfg,
            real_name,
            real_args,
        } => {
            match tokio::time::timeout(
                Duration::from_secs(60),
                state.mcp.call_tool(&cfg, &real_name, real_args),
            )
            .await
            {
                Ok(Ok(r)) => (r.content.clone(), r.is_error, r.content.to_string()),
                Ok(Err(e)) => {
                    let m = format!("tool error: {e}");
                    (json!(m.clone()), true, m)
                }
                Err(_) => {
                    let m = "the tool did not respond in time".to_string();
                    (json!(m.clone()), true, m)
                }
            }
        }
    }
}

/// An MCP tool result's content (JSON) -> an Anthropic `mcp_tool_result` content
/// list of text blocks (always valid `BetaTextBlock`).
fn mcp_result_content(content: &Value) -> Value {
    let mut blocks = Vec::new();
    match content {
        Value::Array(arr) => {
            for b in arr {
                if b.get("type").and_then(Value::as_str) == Some("text") {
                    let t = b.get("text").and_then(Value::as_str).unwrap_or_default();
                    blocks.push(json!({"type": "text", "text": t}));
                } else {
                    blocks.push(json!({"type": "text", "text": b.to_string()}));
                }
            }
        }
        Value::String(s) => blocks.push(json!({"type": "text", "text": s})),
        other => blocks.push(json!({"type": "text", "text": other.to_string()})),
    }
    if blocks.is_empty() {
        blocks.push(json!({"type": "text", "text": ""}));
    }
    Value::Array(blocks)
}

/// Non-streaming MCP agent loop: generate -> execute the model's MCP tool calls ->
/// feed results back -> repeat, accumulating `mcp_tool_use` / `mcp_tool_result`
/// content blocks. Non-MCP (client) tool calls end the turn for the caller.
async fn run_mcp_agent(
    state: Arc<AppState>,
    req: MessagesRequest,
    routing: HashMap<String, (paddock_mcp::ServerConfig, String)>,
    catalog: Vec<crate::tool_search::CatalogTool>,
    web: Option<AnthWeb>,
    fetch: Option<AnthFetch>,
    // the runner's forensic runtime when `{"type":"forensics"}` was requested and
    // `[forensics] tool = true` - the on-demand analyzer the loop executes
    forensics: Option<std::sync::Arc<crate::forensics::ForensicRuntime>>,
    // the builtin clock when `{"type":"current_time"}` was requested -
    // answered in-process, in the declared timezone
    clock: Option<crate::clock::ClockSpec>,
    // every URL the conversation has shown, grown as searches find more -
    // web fetch's seen-before rule
    mut seen_urls: std::collections::HashSet<String>,
    // A round-0 compaction (`precompact_agent`): its block leads the content
    // and `req.messages` already holds the compacted conversation.
    compaction: Option<CompactionOut>,
    scope: crate::events::EventScope,
) -> Response {
    // validated in handle() before the agent branch; the tool set was merged
    // there too (the round-0 pass had to render it)
    let omit_thinking = thinking_omitted(req.thinking.as_ref()).unwrap_or(false);
    let mut work = req;
    work.stream = false;

    let id = format!("msg_{}", uuid::Uuid::new_v4().simple());
    let mut blocks: Vec<Value> = Vec::new();
    let mut prompt_len = 0usize;
    let mut out_tokens = 0usize;
    let mut cached = 0usize;
    let mut model_id = String::new();
    // the clear_* strategies re-apply every round; the report describes the
    // render the last generation ran on
    let mut cm_report: Option<Value> = None;
    // web-search accounting: per-request uses (max_uses) + billed searches
    let (mut web_uses, mut web_requests) = (0usize, 0usize);
    let (mut fetch_uses, mut fetch_requests) = (0usize, 0usize);
    // The turn's budget: repeat ledger, per-round ceiling, and one
    // tools-off pass at the end that answers instead of stalling. Unbounded
    // by count deliberately - the Messages wire has no `max_tool_calls` (that is
    // a Responses field), so only our own two bounds apply here.
    let mut ledger = crate::loop_budget::CallLedger::new();
    let turn_cap = crate::loop_budget::turn_output_cap(work.max_tokens);
    let mut stop: Option<crate::loop_budget::Stop> = None;
    let mut announced = false;

    for round in 0..=crate::loop_budget::MAX_ROUNDS {
        let model = match state.serving.as_ref() {
            Some(m) => m,
            None => {
                return err(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "overloaded_error",
                    "no model is loaded",
                );
            }
        };
        if stop.is_none() && round == crate::loop_budget::MAX_ROUNDS {
            stop = Some(crate::loop_budget::Stop::Rounds(
                crate::loop_budget::MAX_ROUNDS,
            ));
        }
        // The answer round: our tools come out of the rendered set entirely
        // (the caller's own stay - a client tool call hands the turn back,
        // which is a clean ending), plus one instruction to answer with what
        // came back, plus a visible note saying why.
        let answering = stop.is_some();
        if answering && !announced {
            announced = true;
            let notice = stop.expect("answering means stopped").notice();
            blocks.push(json!({"type": "text", "text": notice}));
            strip_our_tools(
                &mut work,
                &routing,
                web.is_some(),
                fetch.is_some(),
                forensics.is_some(),
                clock.is_some(),
            );
            append_user_text(&mut work.messages, crate::loop_budget::ANSWER_ONLY_NUDGE);
        }
        model_id = model.id.clone();
        let dialect = model.dialect;
        let tokenizer = model.tokenizer.clone();
        let t_prep = std::time::Instant::now();
        let mut prepared = match prepare(model, &work, state.max_output_ceiling, &state.sampling) {
            Ok(p) => p,
            Err(e) => return bad(e),
        };
        // Lever 2: a tool round may not spend past what the turn's tool budget
        // has left; `ours` says whether a Length finish was that ceiling or the
        // caller's own max_tokens.
        let ours = !answering
            && crate::loop_budget::round_cap(prepared.max_tokens, out_tokens, turn_cap)
                < prepared.max_tokens;
        if !answering {
            prepared.max_tokens =
                crate::loop_budget::round_cap(prepared.max_tokens, out_tokens, turn_cap);
        }
        scope.tokenized(t_prep.elapsed());
        cm_report = prepared.context_management.clone();
        // over-window (can grow across tool rounds): reject cleanly
        if let Some(e) = crate::chat::context_gate(
            model,
            prepared.engine_prompt.len(),
            prepared.mm_chunks.as_deref(),
            state.max_ctx,
        ) {
            return engine_err(&e);
        }
        if round == 0 {
            prompt_len = prepared.prompt_ids.len();
        }
        let thinking_open = prepared.thinking_open;
        let hints = prepared.hints.clone();
        let single = prepared.single_tool_call;
        let stop_strings = prepared.stop_strings.clone();
        let constraint = instantiate_constraint(
            &prepared.constraint_spec,
            prepared.gate,
            model,
            prepared.think_budget.as_ref(),
        );
        let (tx, mut rx) = unbounded_channel();
        let gen_req = GenRequest {
            prompt: prepared.engine_prompt,
            max_tokens: prepared.max_tokens,
            sampler: prepared.sampler,
            stop_tokens: prepared.stop_tokens,
            events: tx,
            mm_chunks: prepared.mm_chunks,
            constraint,
            logprobs: None,
            submitted: None, // stamped by Engine::submit
        };
        if let Err(e) = model.engine.submit(gen_req) {
            return err(StatusCode::INTERNAL_SERVER_ERROR, "api_error", e);
        }
        let mut ids = Vec::new();
        let mut finish = None;
        while let Some(evt) = rx.recv().await {
            match evt {
                TokenEvent::Prefilled { cached: c, rows } => {
                    if round == 0 {
                        cached = c as usize;
                        // round 0's prompt is what input_tokens reports, and
                        // its image rows are part of it
                        prompt_len = prompt_len.max(rows as usize);
                    }
                }
                TokenEvent::Token { id: t, .. } => ids.push(t),
                TokenEvent::Done(r, stats) => {
                    finish = Some(r);
                    scope.phases(&stats);
                    break;
                }
                TokenEvent::Error(e) => {
                    return engine_err(&e);
                }
            }
        }
        out_tokens += ids.len();
        let raw = tokenizer.decode(&ids, false).unwrap_or_default();
        let mut parsed = parse(dialect, &raw, thinking_open, hints.as_ref());
        if single {
            parsed.tool_calls.truncate(1);
            parsed.complete_calls = parsed.complete_calls.min(1);
        }

        // reasoning + text blocks for this round
        if let Some(reasoning) = &parsed.reasoning {
            let text = if omit_thinking {
                ""
            } else {
                reasoning.as_str()
            };
            blocks.push(json!({"type": "thinking", "thinking": text, "signature": ""}));
        }
        let mut stop_seq = Value::Null;
        if let Some(content) = &parsed.content {
            let text = match find_stop(content, &stop_strings) {
                Some((i, s)) => {
                    stop_seq = json!(s);
                    &content[..i]
                }
                None => content.as_str(),
            };
            if !text.is_empty() {
                blocks.push(json!({"type": "text", "text": text}));
            }
        }

        // On the answer round nothing of ours may run: the tools were not
        // rendered, so anything matching one is a hallucination of a name the
        // model remembers. The caller's own calls still ride back.
        let mcp_calls: Vec<_> = if answering {
            Vec::new()
        } else {
            parsed
                .tool_calls
                .iter()
                .filter(|tc| {
                    is_handled_call(
                        &tc.name,
                        &routing,
                        web.is_some(),
                        fetch.is_some(),
                        forensics.is_some(),
                        clock.is_some(),
                    )
                })
                .collect()
        };
        let client_calls: Vec<_> = parsed
            .tool_calls
            .iter()
            .filter(|tc| {
                !is_handled_call(
                    &tc.name,
                    &routing,
                    web.is_some(),
                    fetch.is_some(),
                    forensics.is_some(),
                    clock.is_some(),
                )
            })
            .collect();

        // A round we cut short (the tool budget ran out mid-round) goes to the
        // answer round instead of returning its tail; the caller's own
        // max_tokens keeps reporting as before.
        if ours && matches!(finish, Some(FinishReason::Length)) {
            stop = Some(crate::loop_budget::Stop::Output);
            continue;
        }
        if mcp_calls.is_empty() || matches!(finish, Some(FinishReason::Length)) {
            for tc in &client_calls {
                let input =
                    serde_json::from_str::<Value>(&tc.arguments).unwrap_or_else(|_| json!({}));
                blocks.push(json!({
                    "type": "tool_use",
                    "id": format!("toolu_{}", uuid::Uuid::new_v4().simple()),
                    "name": tc.name,
                    "input": input,
                }));
            }
            let stop_reason = if !client_calls.is_empty() {
                "tool_use"
            } else if !stop_seq.is_null() {
                "stop_sequence"
            } else {
                match finish {
                    Some(FinishReason::Length) => "max_tokens",
                    _ => "end_turn",
                }
            };
            scope.usage(prompt_len, out_tokens);
            scope.cached(cached);
            scope.finish(stop_reason);
            let mut usage = json!({"input_tokens": prompt_len, "output_tokens": out_tokens});
            if cached > 0 {
                usage["cache_read_input_tokens"] = json!(cached);
            }
            if web.is_some() {
                usage["server_tool_use"] = json!({"web_search_requests": web_requests, "web_fetch_requests": fetch_requests});
            }
            if let Some(c) = &compaction {
                blocks.insert(0, compaction_block(&c.summary));
                usage["iterations"] =
                    agent_iterations(c, &model_id, prompt_len, out_tokens, cached);
            }
            let mut body = json!({
                "id": id,
                "type": "message",
                "role": "assistant",
                "model": model_id,
                "content": blocks,
                "stop_reason": stop_reason,
                "stop_sequence": stop_seq,
                "usage": usage,
            });
            if let Some(cm) = &cm_report {
                body["context_management"] = cm.clone();
            }
            return Json(body).into_response();
        }

        // execute MCP calls: emit mcp_tool_use + mcp_tool_result blocks, feed back
        let mut asst_content: Vec<Value> = Vec::new();
        if let Some(content) = &parsed.content
            && !content.is_empty()
        {
            asst_content.push(json!({"type": "text", "text": content}));
        }
        let mut user_results: Vec<Value> = Vec::new();
        for tc in &mcp_calls {
            // The assistant turn records the model's RAW call (a real tool, or the
            // mcp_search_tools / mcp_call_tool meta-tool) so the next prompt matches
            // what it generated; the client-facing block shows the resolved tool.
            let raw_input =
                serde_json::from_str::<Value>(&tc.arguments).unwrap_or_else(|_| json!({}));
            // Server web search: spec `server_tool_use` + `web_search_tool_result`
            // blocks, not the MCP pair.
            if let Some(w) = web
                .as_ref()
                .filter(|_| tc.name == crate::websearch::TOOL_NAME)
            {
                let tid = format!("srvtoolu_{}", uuid::Uuid::new_v4().simple());
                asst_content.push(
                    json!({"type": "tool_use", "id": tid, "name": tc.name, "input": raw_input}),
                );
                let query = raw_input
                    .get("query")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let (result_content, feedback) =
                    run_anth_web(w, &mut web_uses, &mut web_requests, &query).await;
                // a URL the model just saw in results becomes fetchable
                harvest_urls(&result_content, &mut seen_urls);
                blocks.push(json!({"type": "server_tool_use", "id": tid, "name": crate::websearch::TOOL_NAME, "input": {"query": query}}));
                blocks.push(json!({"type": "web_search_tool_result", "tool_use_id": tid, "content": result_content}));
                user_results
                    .push(json!({"type": "tool_result", "tool_use_id": tid, "content": feedback}));
                continue;
            }
            // Server web fetch: the same server_tool_use pair, reading one
            // named page instead of searching for pages.
            if let Some(f) = fetch
                .as_ref()
                .filter(|_| tc.name == crate::websearch::FETCH_TOOL_NAME)
            {
                let tid = format!("srvtoolu_{}", uuid::Uuid::new_v4().simple());
                asst_content.push(
                    json!({"type": "tool_use", "id": tid, "name": tc.name, "input": raw_input}),
                );
                let url = raw_input
                    .get("url")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let (result_content, feedback) =
                    run_anth_fetch(f, &mut fetch_uses, &mut fetch_requests, &seen_urls, &url).await;
                blocks.push(json!({"type": "server_tool_use", "id": tid, "name": crate::websearch::FETCH_TOOL_NAME, "input": {"url": url}}));
                blocks.push(json!({"type": "web_fetch_tool_result", "tool_use_id": tid, "content": result_content}));
                user_results
                    .push(json!({"type": "tool_result", "tool_use_id": tid, "content": feedback}));
                continue;
            }
            // Forensic tool (paddock extension): no native Anthropic block type
            // exists for it, so it rides the mcp_tool_use/mcp_tool_result pair
            // with server_name "forensics" - the same framing the Responses path
            // gives it (an `mcp_call` with server_label "forensics"). The result
            // is the structured JSON the model reads and weighs.
            if let Some(rt) = forensics
                .as_ref()
                .filter(|_| tc.name == crate::forensics::TOOL_NAME)
            {
                let tid = format!("mcptu_{}", uuid::Uuid::new_v4().simple());
                asst_content.push(json!({
                    "type": "tool_use", "id": tid, "name": tc.name, "input": raw_input.clone(),
                }));
                // Route through the repeat ledger, exactly like the MCP branch. A
                // forced tool_choice makes the model re-emit the same forensics
                // call every round; without dedup each round re-executes AND
                // re-appends the full report, growing the conversation until it
                // overflows the window (observed live: 16 identical calls). Fresh
                // runs once; a second identical call replays that result without
                // re-running; a third is refused - so the report lands at most
                // twice and the loop is bounded.
                let (sig, verdict) = ledger.check(crate::forensics::TOOL_NAME, &tc.arguments);
                let (content, is_err) = match verdict {
                    crate::loop_budget::Verdict::Fresh => {
                        let image_index = crate::forensics::parse_image_index(&tc.arguments);
                        let (c, _output, error, _status) =
                            crate::responses::run_forensics_tool(rt, &work.messages, image_index)
                                .await;
                        let e = error.is_some();
                        ledger.record(&sig, !e, &c);
                        (c, e)
                    }
                    crate::loop_budget::Verdict::Replay(msg) => (msg, false),
                    crate::loop_budget::Verdict::Refuse(msg) => (msg, true),
                };
                blocks.push(json!({
                    "type": "mcp_tool_use",
                    "id": tid,
                    "name": crate::forensics::TOOL_NAME,
                    "server_name": "forensics",
                    "input": raw_input,
                }));
                blocks.push(json!({
                    "type": "mcp_tool_result",
                    "tool_use_id": tid,
                    "is_error": is_err,
                    "content": mcp_result_content(&json!(content)),
                }));
                user_results
                    .push(json!({"type": "tool_result", "tool_use_id": tid, "content": content}));
                continue;
            }
            // Clock tool (paddock extension): the same mcp_tool_use/mcp_tool_result
            // framing with server_name "time" - answered in-process, no I/O.
            if let Some(spec) = clock.filter(|_| tc.name == crate::clock::TOOL_NAME) {
                let tid = format!("mcptu_{}", uuid::Uuid::new_v4().simple());
                asst_content.push(json!({
                    "type": "tool_use", "id": tid, "name": tc.name, "input": raw_input.clone(),
                }));
                let (sig, verdict) = ledger.check(crate::clock::TOOL_NAME, &tc.arguments);
                let (content, is_err) = match verdict {
                    crate::loop_budget::Verdict::Fresh => {
                        let (c, _output, error, _status) = crate::clock::run(spec, &tc.arguments);
                        let e = error.is_some();
                        ledger.record(&sig, !e, &c);
                        (c, e)
                    }
                    crate::loop_budget::Verdict::Replay(msg) => (msg, false),
                    crate::loop_budget::Verdict::Refuse(msg) => (msg, true),
                };
                blocks.push(json!({
                    "type": "mcp_tool_use",
                    "id": tid,
                    "name": crate::clock::TOOL_NAME,
                    "server_name": "time",
                    "input": raw_input,
                }));
                blocks.push(json!({
                    "type": "mcp_tool_result",
                    "tool_use_id": tid,
                    "is_error": is_err,
                    "content": mcp_result_content(&json!(content)),
                }));
                user_results
                    .push(json!({"type": "tool_result", "tool_use_id": tid, "content": content}));
                continue;
            }
            let tid = format!("mcptu_{}", uuid::Uuid::new_v4().simple());
            asst_content
                .push(json!({"type": "tool_use", "id": tid, "name": tc.name, "input": raw_input}));
            let plan =
                plan_anthropic_call(&routing, &catalog, &mut ledger, &tc.name, &tc.arguments);
            let (display_name, display_server, display_input) = (
                plan.display_name.clone(),
                plan.display_server.clone(),
                plan.display_input.clone(),
            );
            let sig = plan.sig;
            let (content_json, is_err, feedback) =
                execute_anth_action(&state, &catalog, plan.action).await;
            if let Some(sig) = &sig {
                ledger.record(sig, !is_err, &feedback);
            }
            blocks.push(json!({
                "type": "mcp_tool_use",
                "id": tid,
                "name": display_name,
                "server_name": display_server,
                "input": display_input,
            }));
            blocks.push(json!({
                "type": "mcp_tool_result",
                "tool_use_id": tid,
                "is_error": is_err,
                "content": mcp_result_content(&content_json),
            }));
            user_results
                .push(json!({"type": "tool_result", "tool_use_id": tid, "content": feedback}));
        }
        work.messages
            .push(json!({"role": "assistant", "content": asst_content}));
        work.messages
            .push(json!({"role": "user", "content": user_results}));
        // The whole turn's generation is bounded, not just each round.
        if out_tokens >= turn_cap {
            stop = Some(crate::loop_budget::Stop::Output);
        }
    }

    // Unreachable: the pass at MAX_ROUNDS answers with no tools of ours, so
    // its `mcp_calls` is empty and it returns above. Kept as the honest tail.
    scope.usage(prompt_len, out_tokens);
    scope.cached(cached);
    scope.finish("end_turn");
    let mut usage = json!({"input_tokens": prompt_len, "output_tokens": out_tokens});
    if web.is_some() {
        usage["server_tool_use"] =
            json!({"web_search_requests": web_requests, "web_fetch_requests": fetch_requests});
    }
    if let Some(c) = &compaction {
        blocks.insert(0, compaction_block(&c.summary));
        usage["iterations"] = agent_iterations(c, &model_id, prompt_len, out_tokens, cached);
    }
    let mut body = json!({
        "id": id, "type": "message", "role": "assistant", "model": model_id,
        "content": blocks, "stop_reason": "end_turn", "stop_sequence": null, "usage": usage,
    });
    if let Some(cm) = &cm_report {
        body["context_management"] = cm.clone();
    }
    Json(body).into_response()
}

/// Streaming MCP agent loop - the Anthropic event protocol, with `mcp_tool_use`
/// and `mcp_tool_result` blocks streamed as the loop executes tools.
fn stream_mcp_agent(
    state: Arc<AppState>,
    req: MessagesRequest,
    routing: HashMap<String, (paddock_mcp::ServerConfig, String)>,
    catalog: Vec<crate::tool_search::CatalogTool>,
    web: Option<AnthWeb>,
    fetch: Option<AnthFetch>,
    // the runner's forensic runtime when `{"type":"forensics"}` was requested and
    // `[forensics] tool = true` - the on-demand analyzer the loop executes
    forensics: Option<std::sync::Arc<crate::forensics::ForensicRuntime>>,
    // the builtin clock when `{"type":"current_time"}` was requested -
    // answered in-process, in the declared timezone
    clock: Option<crate::clock::ClockSpec>,
    // every URL the conversation has shown, grown as searches find more -
    // web fetch's seen-before rule
    mut seen_urls: std::collections::HashSet<String>,
    // A round-0 compaction (`precompact_agent`) - its block leads the stream.
    compaction: Option<CompactionOut>,
    scope: crate::events::EventScope,
) -> Response {
    let sse = stream! {
        // validated in handle() before the agent branch; the tool set was
        // merged there too (the round-0 pass had to render it)
        let omit_thinking = thinking_omitted(req.thinking.as_ref()).unwrap_or(false);
        let mut work = req;
        let (mut web_uses, mut web_requests) = (0usize, 0usize);
        let (mut fetch_uses, mut fetch_requests) = (0usize, 0usize);

        let model_id = match state.serving.as_ref() {
            Some(m) => m.id.clone(),
            None => {
                yield ev("error", json!({"type":"error","error":{"type":"overloaded_error","message":"no model is loaded"}}));
                return;
            }
        };
        let id = format!("msg_{}", uuid::Uuid::new_v4().simple());

        yield ev("message_start", json!({"type":"message_start","message":{
            "id": id, "type":"message", "role":"assistant", "model": model_id,
            "content": [], "stop_reason": null, "stop_sequence": null,
            "usage": {"input_tokens": 0, "output_tokens": 0}}}));
        // keep-alive ping after message_start, like the live API's streams
        yield ev("ping", json!({"type": "ping"}));

        let mut index = 0usize;
        let mut out_tokens = 0usize;
        let mut prompt_len = 0usize;
        let mut final_stop_reason = "end_turn";
        let mut final_stop_seq = Value::Null;
        // the clear_* strategies re-apply every round; the report describes
        // the render the last generation ran on
        let mut cm_report: Option<Value> = None;

        // A round-0 compaction leads the content: same SDK-pinned event shape
        // as the plain path (start, one compaction_delta carrying the whole
        // summary, stop), and index 0 is where a resend expects it.
        if let Some(c) = &compaction {
            yield ev("content_block_start", json!({
                "type": "content_block_start", "index": index,
                "content_block": {"type": "compaction", "content": null}}));
            yield ev("content_block_delta", json!({
                "type": "content_block_delta", "index": index,
                "delta": {"type": "compaction_delta", "content": c.summary}}));
            yield ev("content_block_stop", json!({"type": "content_block_stop", "index": index}));
            index += 1;
        }

        // The turn's budget - same three levers as the non-streamed loop.
        let mut ledger = crate::loop_budget::CallLedger::new();
        let turn_cap = crate::loop_budget::turn_output_cap(work.max_tokens);
        let mut stop: Option<crate::loop_budget::Stop> = None;
        let mut announced = false;

        'rounds: for round in 0..=crate::loop_budget::MAX_ROUNDS {
            let model = match state.serving.as_ref() { Some(m) => m, None => return };
            if stop.is_none() && round == crate::loop_budget::MAX_ROUNDS {
                stop = Some(crate::loop_budget::Stop::Rounds(crate::loop_budget::MAX_ROUNDS));
            }
            // The answer round: our tools leave the rendered set, the model is
            // told to answer with what came back, and a visible text block says
            // why the tool work stopped where it did.
            let answering = stop.is_some();
            if answering && !announced {
                announced = true;
                let notice = stop.expect("answering means stopped").notice();
                yield ev("content_block_start", json!({"type":"content_block_start","index":index,"content_block":{"type":"text","text":""}}));
                yield ev("content_block_delta", json!({"type":"content_block_delta","index":index,"delta":{"type":"text_delta","text":notice}}));
                yield ev("content_block_stop", json!({"type":"content_block_stop","index":index}));
                index += 1;
                strip_our_tools(
                    &mut work,
                    &routing,
                    web.is_some(),
                    fetch.is_some(),
                    forensics.is_some(),
                    clock.is_some(),
                );
                append_user_text(&mut work.messages, crate::loop_budget::ANSWER_ONLY_NUDGE);
            }
            let t_prep = std::time::Instant::now();
            let mut prepared = match prepare(model, &work, state.max_output_ceiling, &state.sampling) {
                Ok(p) => p,
                Err(e) => {
                    yield ev("error", json!({"type":"error","error":{"type":"invalid_request_error","message":e}}));
                    return;
                }
            };
            // Lever 2: a tool round may not spend past what the turn's tool
            // budget has left; `ours` says whether a Length finish was that
            // ceiling or the caller's own max_tokens.
            let ours = !answering
                && crate::loop_budget::round_cap(prepared.max_tokens, out_tokens, turn_cap)
                    < prepared.max_tokens;
            if !answering {
                prepared.max_tokens =
                    crate::loop_budget::round_cap(prepared.max_tokens, out_tokens, turn_cap);
            }
            scope.tokenized(t_prep.elapsed());
            cm_report = prepared.context_management.clone();
            // over-window (can grow across tool rounds): fail the stream cleanly
            if let Some(ge) = crate::chat::context_gate(
                model,
                prepared.engine_prompt.len(),
                prepared.mm_chunks.as_deref(),
                state.max_ctx,
            ) {
                yield ev("error", json!({"type":"error","error":{"type":anthropic_kind(ge.class),"message":ge.message}}));
                return;
            }
            if round == 0 { prompt_len = prepared.prompt_ids.len(); }
            let dialect = model.dialect;
            let tokenizer = model.tokenizer.clone();
            let thinking_open = prepared.thinking_open;
            let hints = prepared.hints.clone();
            let single = prepared.single_tool_call;
            let stop_strings = prepared.stop_strings.clone();
            let constraint = instantiate_constraint(&prepared.constraint_spec, prepared.gate, model, prepared.think_budget.as_ref());
            let (tx, mut rx) = unbounded_channel();
            let gen_req = GenRequest {
                prompt: prepared.engine_prompt, max_tokens: prepared.max_tokens, sampler: prepared.sampler,
                stop_tokens: prepared.stop_tokens, events: tx, mm_chunks: prepared.mm_chunks, constraint, logprobs: None,
                submitted: None };
            if let Err(e) = model.engine.submit(gen_req) {
                yield ev("error", json!({"type":"error","error":{"type":"api_error","message":e}}));
                return;
            }

            let mut think_open = false;
            let mut text_open = false;
            let mut think_emitted = 0usize;
            let mut text_emitted = 0usize;
            let mut ids: Vec<u32> = Vec::new();
            // incremental decode of `ids` (O(n^2) collapse fix)
            let mut sd = tokenizer.stream_decoder(false);
            let mut finish = None;

            loop {
                match rx.recv().await {
                    Some(TokenEvent::Prefilled { rows, .. }) => {
                        // round 0's prompt is the reported input_tokens; an
                        // image expands it well past the tokenized length
                        if round == 0 {
                            prompt_len = prompt_len.max(rows as usize);
                        }
                    }
                    Some(TokenEvent::Token { id: t, .. }) => {
                        ids.push(t);
                        let raw = sd.push(&tokenizer, t);
                        let mut parsed = parse(dialect, &raw, thinking_open, hints.as_ref());
                        if single { parsed.tool_calls.truncate(1); parsed.complete_calls = parsed.complete_calls.min(1); }
                        if let Some(reasoning) = &parsed.reasoning {
                            let safe = safe_emit_len(reasoning, dialect.reasoning_markers(), &[]);
                            if safe > think_emitted {
                                if !think_open {
                                    think_open = true;
                                    yield ev("content_block_start", json!({"type":"content_block_start","index":index,"content_block":{"type":"thinking","thinking":"","signature":""}}));
                                }
                                let delta = reasoning[think_emitted..safe].to_owned();
                                think_emitted = safe;
                                // display "omitted": no thinking text on the wire
                                if !omit_thinking {
                                    yield ev("content_block_delta", json!({"type":"content_block_delta","index":index,"delta":{"type":"thinking_delta","thinking":delta}}));
                                }
                            }
                        }
                        if let Some(content) = &parsed.content {
                            let (cut, hit) = match find_stop(content, &stop_strings) {
                                Some((i, _)) => (i, true),
                                None => (safe_emit_len(content, dialect.content_markers(), &stop_strings), false),
                            };
                            if cut > text_emitted || hit {
                                if think_open && !text_open {
                                    yield ev("content_block_stop", json!({"type":"content_block_stop","index":index}));
                                    index += 1; think_open = false;
                                }
                                if !text_open {
                                    text_open = true;
                                    yield ev("content_block_start", json!({"type":"content_block_start","index":index,"content_block":{"type":"text","text":""}}));
                                }
                                if cut > text_emitted {
                                    let delta = content[text_emitted..cut].to_owned();
                                    text_emitted = cut;
                                    yield ev("content_block_delta", json!({"type":"content_block_delta","index":index,"delta":{"type":"text_delta","text":delta}}));
                                }
                            }
                            if hit { break; }
                        }
                    }
                    Some(TokenEvent::Done(r, stats)) => { finish = Some(r); scope.phases(&stats); break; }
                    None => break,
                    Some(TokenEvent::Error(e)) => {
                        yield ev("error", json!({"type":"error","error":{"type":anthropic_kind(e.class),"message":e.message}}));
                        return;
                    }
                }
            }
            out_tokens += ids.len();
            let raw = tokenizer.decode(&ids, false).unwrap_or_default();
            let mut parsed = parse(dialect, &raw, thinking_open, hints.as_ref());
            if single { parsed.tool_calls.truncate(1); parsed.complete_calls = parsed.complete_calls.min(1); }

            if think_open && !text_open {
                yield ev("content_block_stop", json!({"type":"content_block_stop","index":index}));
                index += 1;
            }
            if text_open {
                yield ev("content_block_stop", json!({"type":"content_block_stop","index":index}));
                index += 1;
            }

            // On the answer round nothing of ours may run (see the twin comment
            // in run_mcp_agent); the caller's own calls still ride back.
            let mcp_calls: Vec<_> = if answering {
                Vec::new()
            } else {
                parsed.tool_calls.iter().filter(|tc| is_handled_call(&tc.name, &routing, web.is_some(), fetch.is_some(), forensics.is_some(), clock.is_some())).collect()
            };
            let client_calls: Vec<_> = parsed.tool_calls.iter().filter(|tc| !is_handled_call(&tc.name, &routing, web.is_some(), fetch.is_some(), forensics.is_some(), clock.is_some())).collect();

            // A round we cut (the tool budget ran out mid-round) goes to the
            // answer round rather than ending the turn on its tail; the
            // caller's own max_tokens reports as before.
            if ours && matches!(finish, Some(FinishReason::Length)) {
                stop = Some(crate::loop_budget::Stop::Output);
                continue 'rounds;
            }
            if mcp_calls.is_empty() || matches!(finish, Some(FinishReason::Length)) {
                for tc in &client_calls {
                    let tid = format!("toolu_{}", uuid::Uuid::new_v4().simple());
                    yield ev("content_block_start", json!({"type":"content_block_start","index":index,"content_block":{"type":"tool_use","id":tid,"name":tc.name,"input":{}}}));
                    yield ev("content_block_delta", json!({"type":"content_block_delta","index":index,"delta":{"type":"input_json_delta","partial_json":tc.arguments}}));
                    yield ev("content_block_stop", json!({"type":"content_block_stop","index":index}));
                    index += 1;
                }
                final_stop_reason = if !client_calls.is_empty() {
                    "tool_use"
                } else if let Some(content) = &parsed.content {
                    match find_stop(content, &stop_strings) {
                        Some((_, s)) => { final_stop_seq = json!(s); "stop_sequence" }
                        None => match finish { Some(FinishReason::Length) => "max_tokens", _ => "end_turn" },
                    }
                } else {
                    match finish { Some(FinishReason::Length) => "max_tokens", _ => "end_turn" }
                };
                break 'rounds;
            }

            let mut asst_content: Vec<Value> = Vec::new();
            if let Some(content) = &parsed.content && !content.is_empty() {
                asst_content.push(json!({"type":"text","text":content}));
            }
            let mut user_results: Vec<Value> = Vec::new();
            for tc in &mcp_calls {
                let raw_input = serde_json::from_str::<Value>(&tc.arguments).unwrap_or_else(|_| json!({}));
                // Server web search: spec `server_tool_use` + `web_search_tool_result`
                // block pair, streamed like Anthropic's.
                if let Some(w) = web.as_ref().filter(|_| tc.name == crate::websearch::TOOL_NAME) {
                    let tid = format!("srvtoolu_{}", uuid::Uuid::new_v4().simple());
                    let query = raw_input.get("query").and_then(Value::as_str).unwrap_or("").to_string();
                    yield ev("content_block_start", json!({"type":"content_block_start","index":index,"content_block":{"type":"server_tool_use","id":tid,"name":crate::websearch::TOOL_NAME,"input":{}}}));
                    yield ev("content_block_delta", json!({"type":"content_block_delta","index":index,"delta":{"type":"input_json_delta","partial_json":json!({"query": query}).to_string()}}));
                    yield ev("content_block_stop", json!({"type":"content_block_stop","index":index}));
                    index += 1;
                    asst_content.push(json!({"type":"tool_use","id":tid,"name":tc.name,"input":raw_input}));
                    let (result_content, feedback) = run_anth_web(w, &mut web_uses, &mut web_requests, &query).await;
                    // a URL the model just saw in results becomes fetchable
                    harvest_urls(&result_content, &mut seen_urls);
                    yield ev("content_block_start", json!({"type":"content_block_start","index":index,"content_block":{"type":"web_search_tool_result","tool_use_id":tid,"content":result_content}}));
                    yield ev("content_block_stop", json!({"type":"content_block_stop","index":index}));
                    index += 1;
                    user_results.push(json!({"type":"tool_result","tool_use_id":tid,"content":feedback}));
                    continue;
                }
                // Server web fetch: the same pair, reading one named page.
                if let Some(f) = fetch.as_ref().filter(|_| tc.name == crate::websearch::FETCH_TOOL_NAME) {
                    let tid = format!("srvtoolu_{}", uuid::Uuid::new_v4().simple());
                    let url = raw_input.get("url").and_then(Value::as_str).unwrap_or("").to_string();
                    yield ev("content_block_start", json!({"type":"content_block_start","index":index,"content_block":{"type":"server_tool_use","id":tid,"name":crate::websearch::FETCH_TOOL_NAME,"input":{}}}));
                    yield ev("content_block_delta", json!({"type":"content_block_delta","index":index,"delta":{"type":"input_json_delta","partial_json":json!({"url": url}).to_string()}}));
                    yield ev("content_block_stop", json!({"type":"content_block_stop","index":index}));
                    index += 1;
                    asst_content.push(json!({"type":"tool_use","id":tid,"name":tc.name,"input":raw_input}));
                    let (result_content, feedback) = run_anth_fetch(f, &mut fetch_uses, &mut fetch_requests, &seen_urls, &url).await;
                    yield ev("content_block_start", json!({"type":"content_block_start","index":index,"content_block":{"type":"web_fetch_tool_result","tool_use_id":tid,"content":result_content}}));
                    yield ev("content_block_stop", json!({"type":"content_block_stop","index":index}));
                    index += 1;
                    user_results.push(json!({"type":"tool_result","tool_use_id":tid,"content":feedback}));
                    continue;
                }
                // Forensic tool (paddock extension): streamed as the mcp_tool_use/
                // mcp_tool_result pair with server_name "forensics" - same framing
                // as the non-streamed path and the Responses surface.
                if let Some(rt) = forensics.as_ref().filter(|_| tc.name == crate::forensics::TOOL_NAME) {
                    let tid = format!("mcptu_{}", uuid::Uuid::new_v4().simple());
                    yield ev("content_block_start", json!({"type":"content_block_start","index":index,"content_block":{"type":"mcp_tool_use","id":tid,"name":crate::forensics::TOOL_NAME,"server_name":"forensics","input":{}}}));
                    yield ev("content_block_delta", json!({"type":"content_block_delta","index":index,"delta":{"type":"input_json_delta","partial_json":raw_input.to_string()}}));
                    yield ev("content_block_stop", json!({"type":"content_block_stop","index":index}));
                    index += 1;
                    asst_content.push(json!({"type":"tool_use","id":tid,"name":tc.name,"input":raw_input}));
                    // Repeat-ledger dedup (see the non-streamed twin): a forced
                    // tool_choice re-emits the same forensics call each round;
                    // replay the first result rather than re-running + re-appending
                    // the full report until the window overflows.
                    let (sig, verdict) = ledger.check(crate::forensics::TOOL_NAME, &tc.arguments);
                    let (content, is_err) = match verdict {
                        crate::loop_budget::Verdict::Fresh => {
                            let image_index = crate::forensics::parse_image_index(&tc.arguments);
                            let (c, _output, error, _status) = crate::responses::run_forensics_tool(rt, &work.messages, image_index).await;
                            let e = error.is_some();
                            ledger.record(&sig, !e, &c);
                            (c, e)
                        }
                        crate::loop_budget::Verdict::Replay(msg) => (msg, false),
                        crate::loop_budget::Verdict::Refuse(msg) => (msg, true),
                    };
                    yield ev("content_block_start", json!({"type":"content_block_start","index":index,"content_block":{"type":"mcp_tool_result","tool_use_id":tid,"is_error":is_err,"content":mcp_result_content(&json!(content))}}));
                    yield ev("content_block_stop", json!({"type":"content_block_stop","index":index}));
                    index += 1;
                    user_results.push(json!({"type":"tool_result","tool_use_id":tid,"content":content}));
                    continue;
                }
                // Clock tool: same framing, server_name "time", answered in-process.
                if let Some(spec) = clock.filter(|_| tc.name == crate::clock::TOOL_NAME) {
                    let tid = format!("mcptu_{}", uuid::Uuid::new_v4().simple());
                    yield ev("content_block_start", json!({"type":"content_block_start","index":index,"content_block":{"type":"mcp_tool_use","id":tid,"name":crate::clock::TOOL_NAME,"server_name":"time","input":{}}}));
                    yield ev("content_block_delta", json!({"type":"content_block_delta","index":index,"delta":{"type":"input_json_delta","partial_json":raw_input.to_string()}}));
                    yield ev("content_block_stop", json!({"type":"content_block_stop","index":index}));
                    index += 1;
                    asst_content.push(json!({"type":"tool_use","id":tid,"name":tc.name,"input":raw_input}));
                    let (sig, verdict) = ledger.check(crate::clock::TOOL_NAME, &tc.arguments);
                    let (content, is_err) = match verdict {
                        crate::loop_budget::Verdict::Fresh => {
                            let (c, _output, error, _status) = crate::clock::run(spec, &tc.arguments);
                            let e = error.is_some();
                            ledger.record(&sig, !e, &c);
                            (c, e)
                        }
                        crate::loop_budget::Verdict::Replay(msg) => (msg, false),
                        crate::loop_budget::Verdict::Refuse(msg) => (msg, true),
                    };
                    yield ev("content_block_start", json!({"type":"content_block_start","index":index,"content_block":{"type":"mcp_tool_result","tool_use_id":tid,"is_error":is_err,"content":mcp_result_content(&json!(content))}}));
                    yield ev("content_block_stop", json!({"type":"content_block_stop","index":index}));
                    index += 1;
                    user_results.push(json!({"type":"tool_result","tool_use_id":tid,"content":content}));
                    continue;
                }
                let tid = format!("mcptu_{}", uuid::Uuid::new_v4().simple());
                // Plan (no I/O) so the mcp_tool_use block can carry the resolved
                // tool identity before we execute; the template turn keeps the raw call.
                let plan = plan_anthropic_call(&routing, &catalog, &mut ledger, &tc.name, &tc.arguments);
                let (display_name, display_server, display_input) =
                    (plan.display_name.clone(), plan.display_server.clone(), plan.display_input.clone());
                let sig = plan.sig;
                yield ev("content_block_start", json!({"type":"content_block_start","index":index,"content_block":{"type":"mcp_tool_use","id":tid,"name":display_name,"server_name":display_server,"input":{}}}));
                yield ev("content_block_delta", json!({"type":"content_block_delta","index":index,"delta":{"type":"input_json_delta","partial_json":display_input.to_string()}}));
                yield ev("content_block_stop", json!({"type":"content_block_stop","index":index}));
                index += 1;
                asst_content.push(json!({"type":"tool_use","id":tid,"name":tc.name,"input":raw_input}));
                let (content_json, is_err, feedback) = execute_anth_action(&state, &catalog, plan.action).await;
                if let Some(sig) = &sig {
                    ledger.record(sig, !is_err, &feedback);
                }
                yield ev("content_block_start", json!({"type":"content_block_start","index":index,"content_block":{"type":"mcp_tool_result","tool_use_id":tid,"is_error":is_err,"content":mcp_result_content(&content_json)}}));
                yield ev("content_block_stop", json!({"type":"content_block_stop","index":index}));
                index += 1;
                user_results.push(json!({"type":"tool_result","tool_use_id":tid,"content":feedback}));
            }
            work.messages.push(json!({"role":"assistant","content":asst_content}));
            work.messages.push(json!({"role":"user","content":user_results}));
            // The whole turn's generation is bounded, not just each round.
            if out_tokens >= turn_cap {
                stop = Some(crate::loop_budget::Stop::Output);
            }
        }

        scope.usage(prompt_len, out_tokens);
        scope.finish(final_stop_reason);
        let mut usage = json!({"input_tokens": prompt_len, "output_tokens": out_tokens});
        if web.is_some() {
            usage["server_tool_use"] = json!({"web_search_requests": web_requests, "web_fetch_requests": fetch_requests});
        }
        if let Some(c) = &compaction {
            usage["iterations"] = agent_iterations(c, &model_id, prompt_len, out_tokens, 0);
        }
        let mut delta_ev = json!({"type":"message_delta","delta":{"stop_reason":final_stop_reason,"stop_sequence":final_stop_seq},"usage":usage});
        if let Some(cm) = &cm_report {
            // same contract as the plain stream: applied edits ride message_delta
            delta_ev["context_management"] = cm.clone();
        }
        yield ev("message_delta", delta_ev);
        yield ev("message_stop", json!({"type":"message_stop"}));
    };
    Sse::new(sse).into_response()
}

fn ev(name: &str, data: Value) -> Result<Event, std::convert::Infallible> {
    Ok(Event::default().event(name).data(data.to_string()))
}

/// `POST /v1/messages/count_tokens` - the same conversion + render pipeline,
/// stopping at tokenization. With an image, the count covers the text tokens
/// plus the single pad slot (an estimate, as the endpoint documents).
pub async fn count_tokens(
    State(state): State<Arc<AppState>>,
    scope: Option<axum::Extension<crate::events::EventScope>>,
    crate::extract::AnthJson(mut req): crate::extract::AnthJson<CountTokensRequest>,
) -> Response {
    let scope = scope.map(|e| e.0).unwrap_or_default();
    let Some(model) = state.serving.as_ref() else {
        return err(
            StatusCode::SERVICE_UNAVAILABLE,
            "overloaded_error",
            "no model is loaded",
        );
    };
    scope.model(&model.id);
    // The effort rung is a TEMPLATE kwarg on a graded model, so it can change
    // the rendered prompt and therefore the count. Parsed here for the same
    // reason the create call parses it, and refusing the same malformed shapes
    // - a counting client that would be 400ed for real must be 400ed here.
    let effort_owned = match parse_output_config(req.output_config.as_ref()) {
        Ok((e, _)) => e,
        Err(e) => return bad(e),
    };
    let effort = effort_owned.as_deref();
    // PDFs expand here exactly as on generate (images or extracted text), so
    // the count and the eventual bill agree for document-carrying requests.
    let opts = match crate::chat::attach_opts(
        req.file_metadata.as_deref(),
        req.max_pages,
        req.pdf_mode.as_deref(),
        req.forensics.as_deref(),
    ) {
        Ok(o) => o,
        Err(e) => return bad(e),
    };
    // Anthropic /v1/messages: injection-only, no extra output item - discard.
    match crate::chat::expand_attachments(&state, model, &mut req.messages, opts, &mut Vec::new())
        .await
    {
        // Anthropic keeps `system` out of the message array, so the capability
        // merges into that field - same shape as the MCP instruction block
        // below, ours leading and the caller's text keeping the tail.
        Ok(Some(sample)) => req.system = Some(merge_system(req.system.take(), &sample)),
        Ok(None) => {}
        Err((code, msg)) => {
            let kind = if code == StatusCode::BAD_REQUEST {
                "invalid_request_error"
            } else {
                "api_error"
            };
            return err(code, kind, msg);
        }
    }
    let tools = match req.tools.as_ref().map(|ts| convert_tools(ts)).transpose() {
        Ok(t) => t,
        Err(e) => return bad(e),
    };
    // same resend semantics as generation: a round-tripped compaction block
    // collapses everything before it, and the count must price that reality
    let rewritten = crate::context_management::resend_rewrite(&req.messages);
    let messages: &[Value] = rewritten.as_deref().unwrap_or(&req.messages);
    // context_management on the count: apply the same edits generation would,
    // so the number a budget-watching client acts on matches the bill. The
    // response carries original_input_tokens for the before/after picture.
    // A fired compact_20260112 is count-IGNORED (a count cannot run the
    // summarization generation), so the count reflects the clears only.
    let cm_cfg = match req
        .context_management
        .as_ref()
        .map(crate::context_management::parse)
    {
        Some(Ok(c)) => Some(c),
        Some(Err(e)) => return bad(e),
        None => None,
    };
    let mut counted_messages: Vec<Value> = messages.to_vec();
    let mut original: Option<usize> = None;
    if let Some(cfg) = &cm_cfg {
        let first = match render_prompt(
            model,
            req.system.as_ref(),
            messages,
            tools.as_deref(),
            req.thinking.as_ref(),
            req.ocr.as_ref(),
            effort,
        ) {
            Ok(r) => r,
            Err(e) => return bad(e),
        };
        let count = |m: &[Value]| {
            render_prompt(
                model,
                req.system.as_ref(),
                m,
                tools.as_deref(),
                req.thinking.as_ref(),
                req.ocr.as_ref(),
                effort,
            )
            .map(|(ids, _, _, _, _)| ids.len())
        };
        match crate::context_management::apply(cfg, messages, first.0.len(), count) {
            Ok((edited, applied)) => {
                original = Some(first.0.len());
                if !applied.edits.is_empty() {
                    counted_messages = edited;
                }
            }
            Err(e) => return bad(e),
        }
    }
    match render_prompt(
        model,
        req.system.as_ref(),
        &counted_messages,
        tools.as_deref(),
        req.thinking.as_ref(),
        req.ocr.as_ref(),
        effort,
    ) {
        // count_tokens reports the ADMITTED length - text rows plus every
        // image's vision rows (a pad placeholder is a position marker, not a
        // row). Counting placeholders as 1 made a 20-page scan price as 20
        // tokens while generation admitted ~80k; the estimate and the bill
        // must agree.
        Ok((_, engine_prompt, _, mm_chunks, _)) => {
            let (total, _, _) =
                crate::chat::admitted_rows(model, engine_prompt.len(), mm_chunks.as_deref());
            scope.usage(total, 0);
            let mut body = json!({"input_tokens": total});
            if let Some(orig) = original {
                body["context_management"] = json!({"original_input_tokens": orig});
            }
            Json(body).into_response()
        }
        Err(e) => bad(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_server_routing() -> HashMap<String, (paddock_mcp::ServerConfig, String)> {
        let cfg = paddock_mcp::ServerConfig {
            id: "test".into(),
            label: "artifacts".into(),
            transport: paddock_mcp::Transport::Http {
                url: "http://localhost/mcp".into(),
                headers: HashMap::new(),
            },
        };
        HashMap::from([(
            "artifacts__artifact_read".to_string(),
            (cfg, "artifacts".to_string()),
        )])
    }

    /// Lever 1 on the Anthropic lane: the same call twice comes off the
    /// ledger, a third time is refused, and the wrapper spelling is the same
    /// call as the direct one.
    #[test]
    fn a_repeated_anthropic_call_replays_then_is_refused() {
        let routing = one_server_routing();
        let catalog = vec![crate::tool_search::CatalogTool {
            name: "artifacts__artifact_read".into(),
            description: "read one".into(),
            input_schema: json!({"type":"object","properties":{"artifact_id":{"type":"string"}},
                                 "required":["artifact_id"]}),
        }];
        let mut ledger = crate::loop_budget::CallLedger::new();
        let args = r#"{"artifact_id":"a1"}"#;

        let p = plan_anthropic_call(
            &routing,
            &catalog,
            &mut ledger,
            "artifacts__artifact_read",
            args,
        );
        assert!(matches!(p.action, AnthAction::Invoke { .. }));
        ledger.record(&p.sig.expect("a call that runs is filed"), true, "the page");

        // ...through the mcp_call_tool envelope this time: same call.
        let wrapped =
            r#"{"name":"artifacts__artifact_read","arguments_json":"{\"artifact_id\":\"a1\"}"}"#;
        let p = plan_anthropic_call(
            &routing,
            &catalog,
            &mut ledger,
            crate::tool_search::CALL_TOOL,
            wrapped,
        );
        match p.action {
            AnthAction::Replay { output } => assert!(output.ends_with("the page"), "{output}"),
            _ => panic!("the wrapper spelling must hit the ledger"),
        }
        assert_eq!(
            p.display_server, "artifacts",
            "a replay keeps the server on its card"
        );

        let p = plan_anthropic_call(
            &routing,
            &catalog,
            &mut ledger,
            "artifacts__artifact_read",
            args,
        );
        match p.action {
            AnthAction::Refuse { message } => assert!(message.contains("twice"), "{message}"),
            _ => panic!("the third emission is a loop"),
        }
    }

    /// The answer round takes our tools out of the rendered set and leaves the
    /// caller's - a client tool call hands the turn back, which is a clean way
    /// for an over-budget turn to end.
    #[test]
    fn the_answer_round_strips_our_tools_and_keeps_the_callers() {
        let routing = one_server_routing();
        let mut req: MessagesRequest = serde_json::from_value(json!({
            "model": "m", "max_tokens": 16, "messages": [],
            "tools": [
                {"name": "get_weather", "input_schema": {"type": "object"}},
                {"name": "artifacts__artifact_read", "input_schema": {"type": "object"}},
                {"name": crate::tool_search::SEARCH_TOOL, "input_schema": {"type": "object"}},
                {"name": crate::tool_search::CALL_TOOL, "input_schema": {"type": "object"}},
            ],
        }))
        .expect("a valid request");
        strip_our_tools(&mut req, &routing, false, false, false, false);
        let names: Vec<&str> = req
            .tools
            .iter()
            .flatten()
            .filter_map(|t| t["name"].as_str())
            .collect();
        assert_eq!(names, ["get_weather"]);
    }

    /// A forcing tool_choice cannot survive the strip: `any` over an emptied
    /// set, or `tool` naming an MCP tool that just left, both 400 the round.
    #[test]
    fn a_forcing_tool_choice_relaxes_with_the_tools() {
        let routing = one_server_routing();
        for forced in [
            json!({"type":"any"}),
            json!({"type":"tool","name":"artifacts__artifact_read"}),
        ] {
            let mut req: MessagesRequest = serde_json::from_value(json!({
                "model": "m", "max_tokens": 16, "messages": [], "tool_choice": forced,
                "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
            }))
            .expect("a valid request");
            strip_our_tools(&mut req, &routing, false, false, false, false);
            assert_eq!(req.tool_choice, Some(json!({"type":"auto"})));
        }
        // "auto" and "none" are left exactly as the caller wrote them.
        let mut req: MessagesRequest = serde_json::from_value(json!({
            "model": "m", "max_tokens": 16, "messages": [], "tool_choice": {"type":"none"},
        }))
        .expect("a valid request");
        strip_our_tools(&mut req, &routing, false, false, false, false);
        assert_eq!(req.tool_choice, Some(json!({"type":"none"})));
    }

    #[test]
    fn stripping_the_only_tools_leaves_none_rather_than_an_empty_list() {
        let routing = one_server_routing();
        let mut req: MessagesRequest = serde_json::from_value(json!({
            "model": "m", "max_tokens": 16, "messages": [],
            "tools": [{"name": "artifacts__artifact_read", "input_schema": {"type": "object"}}],
        }))
        .expect("a valid request");
        strip_our_tools(&mut req, &routing, false, false, false, false);
        assert!(
            req.tools.is_none(),
            "an empty tools array is not the same as no tools"
        );
    }

    /// The Anthropic path serves the same `{"type":"forensics"}` tool the
    /// Responses path does: extraction pulls it out (gated on the runner's
    /// config), the loop recognizes its calls, and the answer round strips it.
    #[test]
    fn anthropic_forensics_tool_extracts_gates_and_is_handled() {
        let rt = crate::forensics::ForensicRuntime::build(&crate::config::ForensicsConfig {
            enabled: true,
            auto: crate::config::ForensicsAuto::Off,
            tool: true,
            device: None,
        })
        .expect("runtime");

        // Enabled runner: the forensics entry is pulled out, the client tool stays.
        let mut state = crate::routes::AppState::for_tests(None);
        state.forensics = Some(rt);
        let mut req: MessagesRequest = serde_json::from_value(json!({
            "model": "m", "max_tokens": 16, "messages": [],
            "tools": [
                {"name": "get_weather", "input_schema": {"type": "object"}},
                {"type": "forensics"},
            ],
        }))
        .expect("valid request");
        let got = extract_forensics_tool(&state, &mut req).expect("served");
        assert!(got.is_some(), "the runtime is returned when tool = true");
        let names: Vec<&str> = req
            .tools
            .iter()
            .flatten()
            .filter_map(|t| t["name"].as_str())
            .collect();
        assert_eq!(
            names,
            ["get_weather"],
            "forensics entry removed, client tool kept"
        );

        // The injected model-facing def + is_handled recognition.
        let def = crate::forensics::anthropic_tool_def();
        assert_eq!(def["name"], crate::forensics::TOOL_NAME);
        assert!(
            def["input_schema"].is_object(),
            "Anthropic def carries input_schema"
        );
        let routing = std::collections::HashMap::new();
        assert!(is_handled_call(
            crate::forensics::TOOL_NAME,
            &routing,
            false,
            false,
            true,
            false
        ));
        assert!(
            !is_handled_call(
                crate::forensics::TOOL_NAME,
                &routing,
                false,
                false,
                false,
                false
            ),
            "not handled when forensics is off"
        );

        // The answer round strips the forensics def (forensics_on).
        let mut req2: MessagesRequest = serde_json::from_value(json!({
            "model": "m", "max_tokens": 16, "messages": [],
            "tools": [{"name": crate::forensics::TOOL_NAME, "input_schema": {"type": "object"}}],
        }))
        .expect("valid request");
        strip_our_tools(&mut req2, &routing, false, false, true, false);
        assert!(
            req2.tools.is_none(),
            "forensics def leaves the rendered set on the answer round"
        );

        // Disabled runner: requesting it is a clean 400, never a silent no-op.
        let mut off = crate::routes::AppState::for_tests(None);
        off.forensics = None;
        let mut req3: MessagesRequest = serde_json::from_value(json!({
            "model": "m", "max_tokens": 16, "messages": [], "tools": [{"type": "forensics"}],
        }))
        .expect("valid request");
        assert!(
            extract_forensics_tool(&off, &mut req3).is_err(),
            "unset runner 400s"
        );

        // tool = false (auto-only runner) is also a 400 for the on-demand surface.
        let auto_only = crate::forensics::ForensicRuntime::build(&crate::config::ForensicsConfig {
            enabled: true,
            auto: crate::config::ForensicsAuto::Images,
            tool: false,
            device: None,
        })
        .expect("runtime");
        let mut auto_state = crate::routes::AppState::for_tests(None);
        auto_state.forensics = Some(auto_only);
        let mut req4: MessagesRequest = serde_json::from_value(json!({
            "model": "m", "max_tokens": 16, "messages": [], "tools": [{"type": "forensics"}],
        }))
        .expect("valid request");
        assert!(
            extract_forensics_tool(&auto_state, &mut req4).is_err(),
            "tool=false 400s"
        );
    }

    /// The nudge joins the tool-result turn rather than opening a second user
    /// turn: two user turns in a row is a shape Anthropic rejects and some
    /// chat templates render badly.
    #[test]
    fn the_answer_nudge_joins_the_tool_result_turn() {
        let mut msgs = vec![
            json!({"role":"user","content":"go"}),
            json!({"role":"assistant","content":[{"type":"tool_use","id":"t1"}]}),
            json!({"role":"user","content":[{"type":"tool_result","tool_use_id":"t1"}]}),
        ];
        append_user_text(&mut msgs, "answer now");
        assert_eq!(msgs.len(), 3, "no new turn");
        let blocks = msgs[2]["content"].as_array().expect("blocks");
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[1]["text"], "answer now");
    }

    #[test]
    fn the_answer_nudge_opens_a_turn_when_the_last_one_is_the_assistant() {
        let mut msgs = vec![json!({"role":"assistant","content":"hm"})];
        append_user_text(&mut msgs, "answer now");
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[1]["role"], "user");
    }

    /// The real granite-vision template, the same fixture the integration
    /// test renders. Lives under tests/ because that is where the rest of the
    /// template gates read it from; this is `cfg(test)` so it never ships.
    fn granite_template() -> &'static str {
        include_str!("../tests/fixtures/granite_vision_chat_template.jinja")
    }

    /// This surface's image path end to end, short of the engine: an
    /// Anthropic `{"type":"image","source":{base64,...}}` block through
    /// `convert_messages` -> `normalize_messages` -> the granite template.
    ///
    /// `convert_messages` is private, so this is the only place the Anthropic
    /// conversion can be pinned against a real template. It has
    /// been broken here before, and not subtly: with extraction ordered after
    /// normalization the payload was destroyed and /v1/messages rejected every
    /// image, because `source` is Anthropic's only inline image shape.
    ///
    /// The assertion is equivalence with what an OpenAI client's `image_url`
    /// part renders to - same picture, same question, same prompt.
    #[test]
    fn an_anthropic_image_block_renders_the_granite_prompt_the_chat_shape_does() {
        const URI: &str = "data:image/png;base64,iVBORw0KGgo=";
        let anthropic = [json!({
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}},
                {"type": "text", "text": "What is this?"}
            ]
        })];
        let converted = convert_messages(None, &anthropic).expect("convert");
        // the pixels have to still be reachable after conversion - this is the
        // ordering the surface got wrong (convert -> extract -> normalize)
        let urls = crate::chat::find_images(&converted).expect("find");
        assert_eq!(
            urls.iter().map(|r| r.url.as_ref()).collect::<Vec<_>>(),
            [URI]
        );

        let msgs = crate::chat_template::normalize_messages(&converted);
        let out =
            crate::chat_template::render(granite_template(), &msgs, None, None).expect("render");
        assert_eq!(out.matches("<image>").count(), 1, "rendered:\n{out}");
        assert!(
            out.contains("<image>\nWhat is this?"),
            "question dropped:\n{out}"
        );

        // byte-for-byte what an OpenAI client gets for the same picture
        let chat = [json!({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": URI}},
                {"type": "text", "text": "What is this?"}
            ]
        })];
        let chat_msgs = crate::chat_template::normalize_messages(&chat);
        let chat_out = crate::chat_template::render(granite_template(), &chat_msgs, None, None)
            .expect("render");
        assert_eq!(out, chat_out, "surfaces diverged");
    }

    /// Task tags ride as ordinary message text, so a surface that mangles text
    /// blocks silently disables the model's real interface rather than
    /// erroring. Anthropic's text blocks convert 1:1, and this pins it.
    #[test]
    fn a_task_tag_still_expands_through_an_anthropic_text_block() {
        let anthropic = [json!({
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}},
                {"type": "text", "text": "<chart2csv>"}
            ]
        })];
        let converted = convert_messages(None, &anthropic).expect("convert");
        let msgs = crate::chat_template::normalize_messages(&converted);
        let out =
            crate::chat_template::render(granite_template(), &msgs, None, None).expect("render");
        assert!(
            out.contains("extract the data into a CSV table"),
            "tag did not expand:\n{out}"
        );
        assert!(
            !out.contains("<chart2csv>"),
            "tag survived into the prompt:\n{out}"
        );
    }

    #[test]
    fn thinking_display_values() {
        assert!(!thinking_omitted(None).unwrap());
        assert!(!thinking_omitted(Some(&json!({"type": "enabled"}))).unwrap());
        assert!(!thinking_omitted(Some(&json!({"display": "summarized"}))).unwrap());
        assert!(thinking_omitted(Some(&json!({"display": "omitted"}))).unwrap());
        assert!(thinking_omitted(Some(&json!({"display": "raw"}))).is_err());
    }

    #[test]
    fn toolset_resolution_precedence() {
        // configs entry > default_config > system default (enabled, eager)
        let mut set = McpToolset::default();
        assert_eq!(set.resolve("t"), (true, false));
        set.default_enabled = Some(false);
        set.default_defer = Some(true);
        assert_eq!(set.resolve("t"), (false, true));
        set.tools.insert("t".into(), (Some(true), Some(false)));
        assert_eq!(set.resolve("t"), (true, false));
        assert_eq!(set.resolve("other"), (false, true));
    }

    #[test]
    fn mcp_toolsets_extract_and_validate() {
        // toolsets pull out of `tools`; plain function tools stay
        let mut req: MessagesRequest = serde_json::from_value(json!({
            "model": "m", "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
            "mcp_servers": [{"type": "url", "url": "https://x", "name": "srv"}],
            "tools": [
                {"name": "fn_tool", "description": "d", "input_schema": {"type": "object"}},
                {"type": "mcp_toolset", "mcp_server_name": "srv",
                 "default_config": {"enabled": false},
                 "configs": {"a": {"enabled": true, "defer_loading": true}}},
            ],
        }))
        .unwrap();
        let sets = extract_mcp_toolsets(&mut req).unwrap();
        assert_eq!(req.tools.as_ref().unwrap().len(), 1);
        assert_eq!(sets["srv"].resolve("a"), (true, true));
        assert_eq!(sets["srv"].resolve("b"), (false, false));

        // an mcp_toolset without mcp_servers is a 400
        let mut bare: MessagesRequest = serde_json::from_value(json!({
            "model": "m", "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "mcp_toolset", "mcp_server_name": "srv"}],
        }))
        .unwrap();
        assert!(extract_mcp_toolsets(&mut bare).is_err());
    }

    #[test]
    fn blocks_convert_to_chat_shapes() {
        let messages = vec![
            json!({"role": "user", "content": "hi"}),
            json!({"role": "assistant", "content": [
                {"type": "thinking", "thinking": "hmm", "signature": ""},
                {"type": "tool_use", "id": "toolu_1", "name": "f", "input": {"x": 1}},
            ]}),
            json!({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "42"},
                {"type": "text", "text": "thanks"},
            ]}),
        ];
        let out = convert_messages(Some(&json!("sys")), &messages).unwrap();
        assert_eq!(out[0]["role"], "system");
        assert_eq!(out[1]["content"], "hi");
        assert_eq!(out[2]["role"], "assistant");
        assert_eq!(out[2]["thinking"], "hmm");
        assert_eq!(out[2]["tool_calls"][0]["function"]["name"], "f");
        // tool_result becomes its own tool message before the user text
        assert_eq!(out[3]["role"], "tool");
        assert_eq!(out[3]["tool_call_id"], "toolu_1");
        assert_eq!(out[4]["role"], "user");
        assert_eq!(out[4]["content"], "thanks");
    }

    #[test]
    fn image_blocks_become_data_uris() {
        let messages = vec![json!({"role": "user", "content": [
            {"type": "text", "text": "what is this?"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/bmp", "data": "AAAA"}},
        ]})];
        let out = convert_messages(None, &messages).unwrap();
        assert!(out[0]["content"].is_array());
        let urls = crate::chat::find_images(&out).unwrap();
        assert_eq!(
            urls.iter().map(|r| r.url.as_ref()).collect::<Vec<_>>(),
            ["data:image/bmp;base64,AAAA"]
        );
    }

    #[test]
    fn find_stop_picks_earliest() {
        let stops = vec!["xx".to_owned(), "b".to_owned()];
        assert_eq!(find_stop("a b xx", &stops), Some((2, "b")));
        assert_eq!(find_stop("nothing", &stops), None);
    }

    /// The agent loops measure the compact trigger once,
    /// so the round-0 pass strips that edit - and only that edit, because the
    /// clear_* strategies are per-round by design.
    #[test]
    fn the_compact_edit_is_stripped_after_round_zero() {
        let mut cm = Some(json!({"edits": [
            {"type": "clear_thinking_20251015"},
            {"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 800}},
            {"type": "clear_tool_uses_20250919"},
        ]}));
        drop_compact_edit(&mut cm);
        let edits = cm.as_ref().unwrap()["edits"].as_array().unwrap();
        assert_eq!(edits.len(), 2);
        assert_eq!(edits[0]["type"], "clear_thinking_20251015");
        assert_eq!(edits[1]["type"], "clear_tool_uses_20250919");
        // idempotent, and a config without one is untouched
        drop_compact_edit(&mut cm);
        assert_eq!(cm.as_ref().unwrap()["edits"].as_array().unwrap().len(), 2);
        let mut none: Option<Value> = None;
        drop_compact_edit(&mut none);
        assert!(none.is_none());
    }

    #[test]
    fn only_urls_the_conversation_already_showed_are_fetchable() {
        // This is web fetch's security guard, so it is tested on the shapes a
        // real conversation carries: prose, a nested tool result, and a search
        // result block. A model that could fetch a URL it INVENTED could
        // exfiltrate its own context by encoding it into a hostname.
        let convo = json!([
            {"role": "user", "content": "compare https://example.com/a and <https://b.test/x?q=1>."},
            {"role": "user", "content": [
                {"type": "text", "text": "also (https://c.test/page)"},
                {"type": "tool_result", "content": [{"type": "text", "text": "see https://d.test/deep/path"}]}
            ]}
        ]);
        let mut seen = std::collections::HashSet::new();
        harvest_urls(&convo, &mut seen);
        for u in [
            "https://example.com/a",
            "https://b.test/x?q=1",
            "https://c.test/page",
            "https://d.test/deep/path",
        ] {
            assert!(seen.contains(u), "{u} was not harvested from {seen:?}");
        }
        // the punctuation that ordinarily follows a URL in prose is not part
        // of it, and the trailing period must not become a different page
        assert!(!seen.contains("https://example.com/a."));
        assert!(!seen.contains("https://c.test/page)"));

        // a URL the model made up is refused however plausible it looks
        let invented = "https://example.com/a/../../secret";
        assert!(
            !seen.iter().any(|s| crate::websearch::same_url(s, invented)),
            "an unseen URL must not match"
        );
        // and the seen ones match under the forgiving comparison
        assert!(
            seen.iter()
                .any(|s| crate::websearch::same_url(s, "https://example.com/a/")),
            "a trailing slash is not a different page"
        );
    }

    #[test]
    fn a_conversation_with_no_links_makes_nothing_fetchable() {
        let convo = json!([{"role": "user", "content": "no links here, just httpish words"}]);
        let mut seen = std::collections::HashSet::new();
        harvest_urls(&convo, &mut seen);
        assert!(seen.is_empty(), "{seen:?}");
    }

    /// `parse_output_config` splits Anthropic's `{effort, format}` into the two
    /// things this server acts on. The HTTP probes in tests/param_surface.rs
    /// only prove a shape is accepted or refused - a mutation that parsed the
    /// effort and then DISCARDED it passed every one of them - so the parse
    /// gets checked on its output here.
    ///
    /// The remaining untested link is the one-line call site that folds the
    /// parsed rung into the template kwargs: that needs a loaded model, so it
    /// is covered by the model-bearing SDK gate and by nothing cheaper.
    mod output_config {
        use super::super::parse_output_config;
        use serde_json::json;

        #[test]
        fn splits_both_halves_and_keeps_the_schema_verbatim() {
            let schema = json!({"type": "object", "properties": {"a": {"type": "string"}}});
            let cfg =
                json!({"effort": "xhigh", "format": {"type": "json_schema", "schema": schema}});
            let (effort, format) = parse_output_config(Some(&cfg)).expect("ok");
            assert_eq!(effort.as_deref(), Some("xhigh"));
            // the schema is lifted out of `format`, not left wrapped: Anthropic
            // puts it directly on `format` where OpenAI nests it one deeper
            assert_eq!(format.expect("format"), schema);
        }

        #[test]
        fn absent_and_empty_are_both_nothing() {
            assert_eq!(parse_output_config(None).expect("ok"), (None, None));
            assert_eq!(
                parse_output_config(Some(&json!({}))).expect("ok"),
                (None, None)
            );
            // explicit nulls are the SDK's way of spelling "unset"
            let cfg = json!({"effort": null, "format": null});
            assert_eq!(parse_output_config(Some(&cfg)).expect("ok"), (None, None));
        }

        #[test]
        fn every_malformed_shape_names_itself() {
            for (cfg, want) in [
                (json!("high"), "must be an object"),
                (json!({"effrot": "high"}), "unsupported output_config field"),
                (json!({"effort": 3}), "must be a string"),
                (
                    json!({"format": {"type": "regex"}}),
                    "unsupported output_config.format",
                ),
                (json!({"format": {"schema": {}}}), "needs a `type`"),
                (
                    json!({"format": {"type": "json_schema"}}),
                    "schema is required",
                ),
            ] {
                let e = parse_output_config(Some(&cfg)).expect_err("should refuse");
                assert!(e.contains(want), "expected {want:?} for {cfg}, got {e}");
            }
        }

        #[test]
        fn the_five_published_rungs_all_parse() {
            // Anthropic publishes low|medium|high|xhigh|max; every one is
            // inside the seven `reasoning_effort_rank` already takes, which is
            // why adopting their ladder needed no new vocabulary.
            for e in ["low", "medium", "high", "xhigh", "max"] {
                let cfg = json!({"effort": e});
                let (got, _) = parse_output_config(Some(&cfg)).expect("ok");
                assert_eq!(got.as_deref(), Some(e));
                assert!(
                    crate::chat::reasoning_effort_rank_is_valid(e),
                    "{e} unknown to the ladder"
                );
            }
        }
    }
}
