//! `POST /v1/responses` - the OpenAI Responses API.
//!
//! Reuses the chat pipeline: input items -> chat-template messages -> generate ->
//! dialect parse -> output items (reasoning / message / function_call).
//! Parallel tool calls fall out of the dialect parser producing multiple
//! function_call items. `text.format` (json_object / json_schema) and
//! `tool_choice` (none / required / named) ride the same grammar machinery as
//! chat (crate::chat::ConstraintSpec); image input items ride the same
//! multimodal plumbing. Truncation at `max_output_tokens` reports status
//! "incomplete" with incomplete_details, never a fake "completed".
//!
//! Deferred, documented (no-silent-failures): server-side conversation state
//! (`previous_response_id`, `store: true`) is rejected, not faked. MCP tools are
//! executed server-side: a request that enables an `mcp` server runs the agentic
//! loop below (generate -> execute -> feed back -> repeat), streaming `mcp_call`
//! (and, for approval-gated servers, `mcp_approval_request`) output items.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_stream::stream;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Json, Response};
use paddock_api::ErrorBody;
use paddock_api::responses::ResponsesRequest;
use paddock_engine::sampler::SamplingParams;
use paddock_engine::service::{FinishReason, GenRequest, MmChunk, TokenEvent, TokenLogprobs};
use serde_json::{Value, json};
use tokio::sync::mpsc::{UnboundedReceiver, unbounded_channel};

use crate::chat::{
    ConstraintSpec, GateSpec, build_mm_chunks, content_gate, decode_images, find_images,
    instantiate_constraint,
};
use crate::chat_template;
use crate::constrained::{CompiledSchema, ToolSet};
use crate::loop_budget;
use crate::parsers::{Dialect, Parsed, ToolHints, holdback, parse, tool_hints};
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

/// The one `include` value this server delivers (the spec's name for
/// per-token logprobs on the Responses surface).
const LOGPROBS_INCLUDE: &str = "message.output_text.logprobs";

/// `stream_options: {include_obfuscation}`.
///
/// OpenAI's hosted stream interleaves an `obfuscation` padding field so an
/// observer cannot read token lengths off TLS record sizes. This server emits
/// no such padding, which means `false` is a truthful description of what the
/// caller will get and `true` is a request for bytes that will never arrive.
/// Accepting `true` silently would be exactly the quiet-ignore this project
/// bans, so it is refused by name; `false` passes.
///
/// Stream-only, matching OpenAI and matching what chat already enforces.
fn check_stream_options(v: Option<&Value>, streaming: bool) -> Result<(), String> {
    let Some(so) = v else { return Ok(()) };
    if !streaming {
        return Err("stream_options requires stream: true".into());
    }
    let Some(obj) = so.as_object() else {
        return Err("stream_options must be an object".into());
    };
    for (k, val) in obj {
        match k.as_str() {
            "include_obfuscation" => match val.as_bool() {
                Some(false) => {}
                Some(true) => {
                    return Err(
                        "stream_options.include_obfuscation: true is not supported - this server emits no obfuscation padding, so the field can only be false"
                            .into(),
                    );
                }
                None => return Err("stream_options.include_obfuscation must be a boolean".into()),
            },
            other => return Err(format!("unsupported stream_options field {other:?}")),
        }
    }
    Ok(())
}

/// Whether a request asked for the logprobs include. Validated once at the
/// handler's entry; secondary lanes (truncation "auto") re-derive it from the
/// request instead of threading another parameter through.
fn lane_want_logprobs(req: &ResponsesRequest) -> bool {
    req.include
        .as_deref()
        .unwrap_or_default()
        .iter()
        .any(|v| v == LOGPROBS_INCLUDE)
}

/// Flatten an input `content` (string | array of {type,text}) to text.
fn content_text(v: &Value) -> String {
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

/// True when a content part is an image item (the same condition the chat
/// template and find_images use, plus the Responses `input_image` type).
fn is_image_part(p: &Value) -> bool {
    p.get("image").is_some()
        || p.get("image_url").is_some()
        || matches!(
            p.get("type").and_then(Value::as_str),
            Some("image") | Some("input_image")
        )
}

/// Normalize `instructions` + `input` into chat-template messages. Content
/// arrays that carry image parts pass through VERBATIM: the qwen template's
/// item conditions (`'image_url' in item`, `'text' in item`) match the
/// Responses part shapes directly, so the template emits the
/// `<|image_pad|>` slots; text-only arrays flatten to a string.
fn messages_from_input(instructions: Option<&str>, input: &Value) -> Result<Vec<Value>, String> {
    let mut msgs = Vec::new();
    if let Some(instr) = instructions {
        msgs.push(json!({"role": "system", "content": instr}));
    }
    // A `reasoning` item describes the assistant message that FOLLOWS it (that
    // is the order this server emits them in, and OpenAI's), so it is held here
    // until that message arrives rather than becoming a turn of its own.
    let mut pending_reasoning: Option<String> = None;
    match input {
        Value::String(s) => msgs.push(json!({"role": "user", "content": s})),
        Value::Array(items) => {
            for it in items {
                let ty = it.get("type").and_then(Value::as_str).unwrap_or("message");
                match ty {
                    // A prior turn's chain of thought, fed back. The documented
                    // multi-turn pattern for Responses is to echo the whole
                    // output array, which contains these - so refusing them
                    // 400s a CONFORMING client, which is what this server did
                    // before this arm existed. Whether the model is then shown the
                    // thinking is the template's call (`preserve_thinking`);
                    // accepting the item is ours.
                    "reasoning" => {
                        // full text first, summary as the fallback: a local
                        // model has the real thing, and only a provider that
                        // withholds it ships summaries alone.
                        let full = it.get("content").map(content_text).unwrap_or_default();
                        let text = if full.is_empty() {
                            it.get("summary").map(content_text).unwrap_or_default()
                        } else {
                            full
                        };
                        if !text.is_empty() {
                            pending_reasoning = Some(text);
                        }
                    }
                    "message" => {
                        let role = it.get("role").and_then(Value::as_str).unwrap_or("user");
                        let raw = it.get("content").unwrap_or(&Value::Null);
                        // Validate here, on the raw parts. Both branches below
                        // erase evidence: the image branch hands the array to a
                        // template that skips what it doesn't match, and
                        // `content_text` keeps only parts carrying `text`. So an
                        // unknown part reached neither the model nor the caller
                        // - a 200 with the content quietly gone.
                        if let Value::Array(parts) = raw {
                            crate::chat_template::validate_content_parts(&[json!({
                                "content": parts.clone()
                            })])?;
                        }
                        let content = match raw {
                            Value::Array(parts) if parts.iter().any(is_image_part) => raw.clone(),
                            other => Value::String(content_text(other)),
                        };
                        let mut m = json!({"role": role, "content": content});
                        // Templates read it off the assistant message
                        // (`message.reasoning_content`), which is also the
                        // chat-completions spelling - one shape reaches the
                        // template whichever API the caller used.
                        if role == "assistant"
                            && let Some(r) = pending_reasoning.take()
                            && let Some(o) = m.as_object_mut()
                        {
                            o.insert("reasoning_content".into(), json!(r));
                        }
                        msgs.push(m);
                    }
                    // prior tool result fed back for the next turn
                    "function_call_output" => {
                        let out = it.get("output").map(content_text).unwrap_or_default();
                        let id = it.get("call_id").and_then(Value::as_str).unwrap_or("");
                        msgs.push(json!({"role": "tool", "content": out, "tool_call_id": id}));
                    }
                    // prior assistant tool call, for conversation context
                    "function_call" => {
                        let name = it.get("name").and_then(Value::as_str).unwrap_or("");
                        let args = it.get("arguments").and_then(Value::as_str).unwrap_or("{}");
                        let id = it.get("call_id").and_then(Value::as_str).unwrap_or("");
                        msgs.push(json!({
                            "role": "assistant",
                            "tool_calls": [{"id": id, "type": "function",
                                            "function": {"name": name, "arguments": args}}]
                        }));
                    }
                    // Consumed elsewhere in the pipeline rather than as a
                    // conversation turn: resume_decisions reads these to
                    // release the approvals held against previous_response_id.
                    "mcp_approval_response" => {}
                    // Anything else was being dropped on the floor with a 200.
                    // An input item is a whole conversation turn: losing one
                    // silently is worse than losing a content part.
                    other => {
                        return Err(format!(
                            "unsupported input item type {other:?} (this server accepts \
                             message, reasoning, function_call, function_call_output \
                             and mcp_approval_response items)"
                        ));
                    }
                }
            }
        }
        _ => return Err("`input` must be a string or an array of input items".into()),
    }
    if msgs.iter().all(|m| m["role"] == "system") {
        return Err("no user input provided".into());
    }
    Ok(msgs)
}

/// Convert Responses-flat function tools to Chat-Completions-nested shape.
/// `mcp` and other hosted tool types are dropped for now (the model can't be
/// handed an MCP server we don't yet connect to - documented roadmap gap).
/// Tool types this surface actually serves. `web_search*` and `mcp` are
/// handled upstream in `gather_tools` (they never reach the template as
/// function defs), so they are dropped here but not ignored.
fn served_tool_type(ty: Option<&str>) -> bool {
    matches!(
        ty,
        Some("function") | Some("mcp") | Some("forensics") | Some("current_time") | None
    ) || ty.is_some_and(|s| s.starts_with("web_search"))
}

/// Flatten Responses tools into the nested chat shape the templates render.
///
/// Errors on a type we do not serve rather than dropping it. The old
/// `_ => None` filter meant a client declaring `file_search` or
/// `code_interpreter` got a clean 200 with the tool simply absent - the model
/// never saw it, the caller never heard about it. That is the silent-failure
/// shape this project bans, and it is invisible precisely because the request
/// "worked".
fn normalize_tools(tools: &[Value]) -> Result<Vec<Value>, String> {
    let mut out = Vec::new();
    for t in tools {
        let ty = t.get("type").and_then(Value::as_str);
        if !served_tool_type(ty) {
            return Err(format!(
                "unsupported tool type {:?} (this server serves `function`, `web_search`, \
                 `forensics`, `current_time` and `mcp` tools)",
                ty.unwrap_or("unknown")
            ));
        }
        if ty == Some("mcp")
            || ty == Some("forensics")
            || ty == Some("current_time")
            || ty.is_some_and(|s| s.starts_with("web_search"))
        {
            continue; // handled by gather_tools, never a function def
        }
        out.push(if t.get("function").is_some() {
            t.clone() // already nested
        } else {
            json!({
                "type": "function",
                "function": {
                    "name": t.get("name").cloned().unwrap_or(Value::Null),
                    "description": t.get("description").cloned().unwrap_or(Value::Null),
                    "parameters": t.get("parameters").cloned().unwrap_or(json!({})),
                }
            })
        });
    }
    Ok(out)
}

struct Prepared {
    /// The rendered prompt as tokenized, one placeholder per image. This is the
    /// length a request is ADMITTED against and the pre-prefill token count.
    prompt_ids: Vec<u32>,
    /// `prompt_ids` minus the image placeholders - what the engine is GIVEN
    /// (see `chat::MmPrompt`). Two things read it: `GenRequest.prompt`, because
    /// a placeholder in the history stream would skew the repetition penalties
    /// against a token the model never sees; and its length, because the engine
    /// reports the real prefill row count once prefill lands and the gap
    /// between the two is what the request's images cost - the only number
    /// that tells a caller what its `detail` choice actually bought.
    engine_prompt: Vec<u32>,
    sampler: SamplingParams,
    stop_tokens: Vec<u32>,
    max_tokens: usize,
    thinking_open: bool,
    hints: Option<ToolHints>,
    mm_chunks: Option<Vec<MmChunk>>,
    constraint_spec: ConstraintSpec,
    gate: GateSpec,
    /// resolved thinking budget (reasoning.max_tokens) - per-round like the
    /// grammar: each agent round's constraint gets a fresh wrapper
    think_budget: Option<crate::chat::ThinkBudget>,
    single_tool_call: bool,
    /// deepseek2-ocr resolution (echoed on the response; `ngram` already
    /// folded into `sampler`, `force_base` into `mm_chunks`)
    ocr: Option<crate::deepseek_ocr::OcrResolved>,
}

/// Build the per-round generation inputs. `messages` is the running transcript
/// (the caller owns it so the agent loop can append tool results and re-prepare);
/// `extra_tools` are already-nested function tools to add to the model's set
/// (the MCP agent injects discovered tools here).
fn prepare(
    model: &ServingModel,
    req: &ResponsesRequest,
    messages: &[Value],
    extra_tools: &[Value],
    // Hard ceiling on generated tokens (`AppState::max_output_ceiling`), or
    // `None` for no clamp. Bounds a request that asks for a huge generation.
    output_ceiling: Option<usize>,
    sd: &crate::routes::SamplingDefaults,
) -> Result<Prepared, String> {
    let template = model
        .chat_template
        .as_deref()
        .ok_or("this model has no chat template")?;
    // Image items are extracted before normalize_messages, which rewrites
    // every image part down to a bare `{"type":"image"}` marker for the
    // template - running that first DESTROYS the payload, and this surface
    // then rejected every image with "image content part has no url". Chat
    // completions has always had this order; /v1/responses and /v1/messages
    // had it inverted, so images never worked on either.
    let image_refs = find_images(messages)?;

    // Audio parts are served on /v1/chat/completions and
    // /v1/audio/transcriptions only - refuse here rather than let the
    // template drop them silently now that `input_audio` passes part
    // validation.
    if !crate::chat::find_audio(messages)?.is_empty() {
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
    // Responses spells `detail` on the input_image part itself rather than
    // nested under image_url; find_images accepts both spellings.
    let images = decode_images(image_refs, model.engine.vision_budget())?;

    // Name anything we don't serve before the template sees it. This surface
    // was the silent one: chat at least died inside the template, but
    // Responses passes content arrays through VERBATIM, so an unknown part
    // type reached a `'text' in item` template, matched nothing, and vanished
    // with a 200.
    chat_template::validate_content_parts(messages)?;
    // arguments-strings -> objects, or templates drop them from history
    let mut messages = chat_template::normalize_messages(messages);
    if let Some(marker) = model.audio_inline_marker.as_deref() {
        chat_template::inline_audio_content(&mut messages, marker);
    }

    // Responses function tools are FLAT ({type,name,parameters}); the chat
    // template expects the Chat-Completions NESTED shape ({type,function:{...}}).
    let mut tools = req
        .tools
        .as_ref()
        .map(|ts| normalize_tools(ts))
        .transpose()?;
    if !extra_tools.is_empty() {
        let mut v = tools.take().unwrap_or_default();
        v.extend(extra_tools.iter().cloned());
        tools = Some(v);
    }

    // tool_choice: "none" hides tools; "required" / named forces via grammar
    let mut forced_tool: Option<Option<String>> = None;
    match req.tool_choice.as_ref() {
        None => {}
        Some(v) if v.as_str() == Some("auto") => {}
        Some(v) if v.as_str() == Some("none") => tools = None,
        Some(v) if v.as_str() == Some("required") => forced_tool = Some(None),
        Some(v) if v.get("type").and_then(Value::as_str) == Some("function") => {
            // Responses-flat {"type":"function","name":...}; the chat-nested
            // form is accepted too
            let name = v
                .get("name")
                .or_else(|| v.get("function").and_then(|f| f.get("name")))
                .and_then(Value::as_str)
                .ok_or("tool_choice function needs a name")?;
            forced_tool = Some(Some(name.to_owned()));
        }
        Some(other) => return Err(format!("invalid tool_choice {other}")),
    }
    let tool_syntax = match forced_tool {
        None => None,
        Some(_) => Some(model.dialect.tool_syntax().ok_or_else(|| {
            crate::chat::no_forced_tool_grammar(model.dialect, "\"required\"/named function")
        })?),
    };

    // text.format: the Responses structured-output knob (schema is FLAT under
    // format, unlike chat's nested response_format.json_schema.schema)
    let rf_schema = match req.text.as_ref().and_then(|t| t.get("format")) {
        None => None,
        Some(f) => match f.get("type").and_then(Value::as_str) {
            None | Some("text") => None,
            Some("json_object") => Some(CompiledSchema::any_json()),
            Some("json_schema") => {
                let schema = f.get("schema").ok_or("text.format.schema is required")?;
                Some(CompiledSchema::compile(schema)?)
            }
            Some(other) => return Err(format!("unsupported text.format type {other:?}")),
        },
    };
    if rf_schema.is_some() && forced_tool.is_some() {
        return Err(
            "text.format cannot be combined with a forced tool_choice (the output \
             cannot be both a JSON answer and a tool call)"
                .into(),
        );
    }

    // chat_template_kwargs (vLLM-style): qwen3.5 `enable_thinking` opens the
    // <think> block; gpt-oss `reasoning_effort`. reasoning.effort below overlays
    // it for Harmony.
    let mut kwargs_obj = match req.chat_template_kwargs.as_ref() {
        None => serde_json::Map::new(),
        Some(k) => k
            .as_object()
            .cloned()
            .ok_or("chat_template_kwargs must be a JSON object")?,
    };
    // reasoning: {"effort": ...} - graded on gpt-oss, an on/off toggle on every
    // other family that reasons (see chat::merge_reasoning_effort). Any other
    // key would be silently ignored, so it is an honest error instead.
    let mut budget_req: Option<usize> = None;
    if let Some(r) = req.reasoning.as_ref() {
        let obj = r.as_object().ok_or("`reasoning` must be a JSON object")?;
        for k in obj.keys() {
            if k != "effort" && k != "max_tokens" {
                return Err(format!("unsupported reasoning parameter {k:?}"));
            }
        }
        if let Some(effort) = obj.get("effort").and_then(Value::as_str) {
            let merged = crate::chat::merge_reasoning_effort(
                &model.reasoning,
                effort,
                Some(Value::Object(kwargs_obj.clone())),
            )?;
            kwargs_obj = merged
                .and_then(|v| v.as_object().cloned())
                .unwrap_or(kwargs_obj);
        }
        // max_tokens: the thinking budget (OpenRouter's unified name for
        // Anthropic budget_tokens). Resolved against the rendered prompt
        // below, once thinking_open is known.
        if let Some(v) = obj.get("max_tokens") {
            budget_req = Some(
                v.as_u64()
                    .filter(|&n| n >= 1)
                    .ok_or("reasoning.max_tokens must be a positive integer")?
                    as usize,
            );
        }
    }
    // text.verbosity: current-spec length hint, validated for conformance;
    // local models have no verbosity knob, so a valid value changes nothing
    if let Some(v) = req
        .text
        .as_ref()
        .and_then(|t| t.get("verbosity"))
        .and_then(Value::as_str)
        && !matches!(v, "low" | "medium" | "high")
    {
        return Err(format!(
            "invalid text.verbosity {v:?} (expected low, medium, or high)"
        ));
    }
    let kwargs = if kwargs_obj.is_empty() {
        None
    } else {
        Some(Value::Object(kwargs_obj))
    };

    // deepseek2-ocr instruction mapping  - same seam as chat:
    // resolve the `ocr` object + prompt vocabulary against the normalized
    // messages, right before render (see crate::deepseek_ocr).
    if req.ocr.is_some() && !model.ocr && !model.paddleocr {
        return Err("the `ocr` request object is only served by document-parser models".into());
    }
    let ocr = if model.ocr {
        let opts = crate::deepseek_ocr::OcrOpts::from_request(req.ocr.as_ref(), kwargs.as_ref())?;
        let sizes: Vec<(usize, usize)> =
            images.iter().map(crate::chat::RequestImage::size).collect();
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

    let prompt = chat_template::render(template, &messages, tools.as_deref(), kwargs.as_ref())?;
    // thinking-mode detection is dialect-shaped - see Dialect::thinking_open.
    // (This also fixes a latent gemma4 bug: the old bare `ends_with("<think>\n")`
    // read gemma4-thinking as off here, so a response_format grammar clamped
    // from token 0 and the thought channel could never open.)
    let thinking_open = model.dialect.thinking_open(&prompt);
    let think_budget = budget_req
        .map(|n| crate::chat::think_budget(model, n, thinking_open, "reasoning.max_tokens"))
        .transpose()?;
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

    let was_forced = forced_tool.is_some();
    let mut constraint_spec = match forced_tool {
        Some(only) => ConstraintSpec::Tool(ToolSet::compile(
            tool_syntax.expect("gated with forced_tool"),
            tools.as_deref().unwrap_or(&[]),
            only.as_deref(),
        )?),
        None => match rf_schema {
            Some(s) => ConstraintSpec::Json(s),
            // auto tool choice: the dialect's grammar, armed as a re-armable
            // dispatch so a call the model makes cannot be malformed
            None => crate::chat::auto_tool_dispatch(
                model,
                tools.as_deref(),
                req.parallel_tool_calls == Some(false),
            ),
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
    // What the turn is actually constrained by, named out loud. The tool
    // grammar declining is currently a debug line inside `auto_tool_dispatch`,
    // which is how a whole class of "the model emitted a bare tool call" went
    // unexplained: from outside, an unconstrained turn and a constrained one
    // look identical until a model is weak enough to show the difference.
    // Cheap, once per round, and it is the first thing worth reading when a
    // call comes back malformed.
    tracing::debug!(
        constraint = match constraint_spec {
            ConstraintSpec::None => "none",
            ConstraintSpec::Tool(_) => "tool (forced)",
            ConstraintSpec::Dispatch(..) => "dispatch (auto)",
            ConstraintSpec::Json(_) => "json schema",
        },
        tools = tools.as_ref().map_or(0, Vec::len),
        forced = was_forced,
        "tool constraint for this round"
    );

    crate::chat::validate_sampling(
        req.temperature,
        2.0,
        req.top_p,
        req.min_p,
        req.frequency_penalty,
        req.presence_penalty,
    )?;

    // this model's elected defaults for this turn - see chat::prepare
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
        // the official Responses wire has no penalty knobs; ours takes them as
        // local extensions (the Studio's dials), else server defaults
        repeat_penalty: req.repeat_penalty.unwrap_or(dflt.repeat_penalty),
        repeat_last_n: sd.repeat_last_n,
        presence_penalty: req.presence_penalty.unwrap_or(0.0),
        frequency_penalty: req.frequency_penalty.unwrap_or(0.0),
        seed: sd.seed_or_now(req.seed),
        // the Responses API has no logit_bias knob
        logit_bias: Vec::new(),
        // the OCR family's repetition guard (reference parity), off elsewhere
        no_repeat_ngram: ocr.as_ref().map_or((0, 0), |o| o.ngram),
    };
    Ok(Prepared {
        prompt_ids,
        engine_prompt,
        sampler,
        stop_tokens: model.stop_tokens.clone(),
        // Clamp to the server ceiling so a request can't demand a huge (costly)
        // generation on an exposed instance; unset = honor the request as-is.
        max_tokens: output_ceiling.map_or(req.max_output_tokens, |c| req.max_output_tokens.min(c)),
        thinking_open,
        hints: tool_hints(tools.as_deref()),
        mm_chunks,
        constraint_spec,
        gate,
        think_budget,
        single_tool_call: req.parallel_tool_calls == Some(false),
        ocr,
    })
}

struct Meta {
    id: String,
    model_id: String,
    tokenizer: Arc<paddock_tokenizer::GgufTokenizer>,
    prompt_len: usize,
    /// text-only prompt length; `prompt_len - text_prompt_len` = media rows
    text_prompt_len: usize,
    /// those media rows are a clip's, not a picture's - picks the
    /// usage field they report under (see `Usage::media_details`).
    media_is_audio: bool,
    dialect: Dialect,
    thinking_open: bool,
    hints: Option<ToolHints>,
    single_tool_call: bool,
    // request echoes for the response object
    instructions: Option<String>,
    max_output_tokens: usize,
    /// The caller's tool-call ceiling, echoed back like every other request
    /// param the Response object carries a slot for. Only the agent loop can
    /// actually spend it (see `loop_budget::CallLedger::with_limit`).
    max_tool_calls: Option<usize>,
    temperature: f32,
    top_p: f32,
    tools: Value,
    tool_choice: Value,
    text_format: Value,
    /// Response-shaping extras only some paths set.
    ex: Extras,
    /// The logprobs include was asked for: text deltas/done and the final
    /// content part carry the per-token entries the engine computed.
    want_logprobs: bool,
    /// Event-record slots (§8.1); no-op unless the events middleware planted one.
    scope: crate::events::EventScope,
}

/// Defaults describe the plain single-shot request; the compaction and
/// truncation paths override.
#[derive(Default)]
struct Extras {
    /// Request echo: truncation "auto" was asked for.
    truncation_auto: bool,
    /// Messages removed by truncation "auto" - reported via the honest
    /// `truncation_dropped_items` extension field when > 0 (the official
    /// Response object has no slot for it; a superset beats silence).
    dropped: usize,
    /// A fired compaction: the prebuilt `compaction` item. Emitted as
    /// output[0] - non-stream inserts it, the stream emits its
    /// added/done pair before every other item and shifts indices by one.
    compaction: Option<Value>,
    /// Context-enrichment output items - `{type:"file_metadata"}` (always-on)
    /// and `{type:"forensics"}` (forensics on), one per image/PDF attachment.
    /// Lead the model's output, right after any compaction - the single-shot
    /// path's surface. The agent path carries the same items through `mcp_items`
    /// instead, so these stay empty there.
    enrichment: Vec<Value>,
    /// deepseek2-ocr resolution - echoed as the response's `ocr` extension;
    /// grounded `regions` are appended at the terminal points, where the
    /// finished output ids exist.
    ocr: Option<crate::deepseek_ocr::OcrResolved>,
}

impl Meta {
    /// One spec logprob entry - same shape as chat completions' (`token`,
    /// `logprob`, `bytes`, `top_logprobs`), which is also what the Responses
    /// events carry. Entries cover all sampled tokens of the turn in order -
    /// reasoning and marker tokens included - because text-level channel
    /// parsing loses token alignment (the chat surface's documented
    /// deviation, kept identical here). On the OCR lanes, which this include
    /// exists for, every sampled token is output text, so the array is exact.
    fn lp_entry(&self, id: u32, lp: &TokenLogprobs) -> Value {
        let tok = self.tokenizer.decode(&[id], false).unwrap_or_default();
        json!({
            "token": tok,
            "logprob": lp.chosen,
            "bytes": tok.as_bytes(),
            "top_logprobs": lp.top.iter().map(|&(tid, l)| {
                let ts = self.tokenizer.decode(&[tid], false).unwrap_or_default();
                json!({"token": ts, "logprob": l, "bytes": ts.as_bytes()})
            }).collect::<Vec<_>>(),
        })
    }

    /// Grounded regions for the finished output - parsed from a decode that
    /// keeps special tokens (the markup rides on `<|ref|>`/`<|det|>`
    /// specials). Attempted on every OCR-resolved decode: document-mode det
    /// records are regions too, and the parse itself is the truth test.
    fn attach_ocr_regions(&self, body: &mut Value, ids: &[u32]) {
        if let Some(_o) = &self.ex.ocr
            && let Ok(raw) = self.tokenizer.decode(ids, false)
            && let Some(regions) = crate::deepseek_ocr::regions_json(&raw)
        {
            body["ocr"]["regions"] = regions;
        }
    }
}

impl Meta {
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

pub async fn handle(
    State(state): State<Arc<AppState>>,
    scope: Option<axum::Extension<crate::events::EventScope>>,
    crate::extract::OaiJson(mut req): crate::extract::OaiJson<ResponsesRequest>,
) -> Response {
    let scope = scope.map(|e| e.0).unwrap_or_default();
    // Request-shape checks run before the model-availability check: a
    // malformed request is malformed whether or not a model is loaded, and a
    // 503 there hides a 400 the caller has to fix.
    if let Err(e) = check_stream_options(req.stream_options.as_ref(), req.stream) {
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
    // MCP approval continuation: a response paused on `mcp_approval_request`
    // resumes here via `previous_response_id` + `mcp_approval_response` input
    // items. (We do not otherwise persist responses - an unknown id 400s.)
    if let Some(prev_id) = req.previous_response_id.clone() {
        return resume_response(state.clone(), req, prev_id, scope).await;
    }
    if req.store == Some(true) {
        return err(
            StatusCode::BAD_REQUEST,
            "unsupported_parameter",
            "responses are not persisted on this server; pass store: false",
        );
    }
    if let Some(t) = req.truncation.as_deref()
        && !matches!(t, "disabled" | "auto")
    {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!("invalid truncation {t:?} (expected \"auto\" or \"disabled\")"),
        );
    }
    if let Some(md) = req.metadata.as_ref()
        && !(md.is_null() || md.as_object().is_some_and(|o| o.is_empty()))
    {
        return err(
            StatusCode::BAD_REQUEST,
            "unsupported_parameter",
            "metadata requires stored responses, which this server does not have",
        );
    }
    // `include`: logprobs is the one value we deliver; anything else 400s
    // rather than silently riding along undelivered.
    let mut want_logprobs = false;
    for v in req.include.as_deref().unwrap_or_default() {
        if v == LOGPROBS_INCLUDE {
            want_logprobs = true;
        } else {
            return err(
                StatusCode::BAD_REQUEST,
                "unsupported_parameter",
                format!("include value {v:?} is not supported (only \"{LOGPROBS_INCLUDE}\")"),
            );
        }
    }
    if let Some(k) = req.top_logprobs
        && k > 20
    {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!("top_logprobs must be 0..=20 (got {k})"),
        );
    }
    // spec semantics: `top_logprobs` shapes what the include delivers, and
    // logprobs ride the plain text lane - the agent loop's rounds are not a
    // message a client can align token-wise, so asking for both is refused
    // loudly below (at the loop branch) instead of dropped silently.
    let logprobs_k = want_logprobs.then(|| req.top_logprobs.unwrap_or(0));

    // Round-tripped compaction items rewrite the conversation first (active
    // on every request, config or not): everything before the last one
    // collapses into its summary, exactly reproducing what the compacting
    // request rendered - the radix-cache invariant. Then a compaction_trigger
    // (must be the final item) forces a compact-now of the whole input.
    let mut compact_trigger = false;
    if let Value::Array(items) = &mut req.input {
        if let Some(rw) = crate::context_management::oa_resend_rewrite(items) {
            *items = rw;
        }
        compact_trigger = match crate::context_management::oa_take_trigger(items) {
            Ok(t) => t,
            Err(e) => return err(StatusCode::BAD_REQUEST, "invalid_request_error", e),
        };
    }
    // `context_management: [{"type": "compaction", "compact_threshold": N}]`
    let compact_threshold = match req.context_management.as_deref() {
        None | Some([]) => None,
        Some(entries) => match crate::context_management::oa_parse(entries) {
            Ok(t) => t,
            Err(e) => return err(StatusCode::BAD_REQUEST, "invalid_request_error", e),
        },
    };
    // PDF attachments in the input items -> page images or extracted text,
    // before message assembly, so the expanded parts pass through
    // messages_from_input verbatim and ride the normal paths (single-shot +
    // agent loop both).
    let opts = match crate::chat::attach_opts(
        req.file_metadata.as_deref(),
        req.max_pages,
        req.pdf_mode.as_deref(),
        req.forensics.as_deref(),
    ) {
        Ok(o) => o,
        Err(e) => return err(StatusCode::BAD_REQUEST, "invalid_request_error", e),
    };
    // Carried out of expansion and applied after the input becomes messages:
    // on this API the system prompt is `instructions`, not an item, so there
    // is no system turn inside `items` to add it to (see add_map_capability).
    let mut map_sample: Option<String> = None;
    // Context enrichment: full file metadata (always-on) + forensic reports
    // (forensics on) ride out here as prebuilt output items, so a caller (incl.
    // the optional manager) reads them from the standard Responses output - not
    // just the model's injected context.
    let mut enrichment_items: Vec<Value> = Vec::new();
    req.input = match std::mem::take(&mut req.input) {
        Value::Array(mut items) => {
            match crate::chat::expand_attachments(
                &state,
                model,
                &mut items,
                opts,
                &mut enrichment_items,
            )
            .await
            {
                Ok(sample) => map_sample = sample,
                Err((code, msg)) => {
                    let kind = if code == StatusCode::BAD_REQUEST {
                        "invalid_request_error"
                    } else {
                        "internal_error"
                    };
                    return err(code, kind, msg);
                }
            }
            Value::Array(items)
        }
        other => other,
    };

    let mut messages = match messages_from_input(req.instructions.as_deref(), &req.input) {
        Ok(m) => m,
        Err(e) => return err(StatusCode::BAD_REQUEST, "invalid_request_error", e),
    };
    // Now the system turn exists (instructions became one), so the photo
    // capability has somewhere to go - and it goes into `req.instructions`,
    // the SYSTEM SLOT, not only into the message list. Same reason the MCP
    // servers' instructions do it below (see merge_instructions): every
    // COMPACTION on this path - precompact_agent, run_compacting_oa, the
    // /compact endpoint - throws the message list away and rebuilds it from
    // `req.instructions`, so a block living only in `messages` is gone the
    // first time the conversation compacts, while the photo line in the user
    // item is carried through as content and survives. A model left holding
    // the coordinates with no reason to draw them. The agent loop itself is
    // fine (it appends), but precompact_agent runs before the first
    // generation, so a long enough conversation loses it on answer one.
    // `messages` is patched too, since it was built at line 850, before this.
    if let Some(sample) = map_sample.as_deref() {
        let text = crate::doc::map_capability_text(sample);
        req.instructions = merge_instructions(req.instructions.take(), &[text]);
        crate::doc::add_map_capability(&mut messages, sample);
    }

    // compaction_trigger: summarize the whole conversation now (no tail
    // split - the caller is archiving, not asking) and answer with just the
    // compaction item. Declared server tools are deliberately not gathered
    // for this: no tool can contribute to a summary, and dialing MCP servers
    // to render tool defs the generation only needs for prefix alignment is
    // the wrong trade (the render simply misses them; same as
    // POST /v1/responses/compact).
    if compact_trigger {
        return run_compact_trigger(state.clone(), req, messages, scope).await;
    }

    // Server-run tools (MCP servers, web search): if the request enables any,
    // run the agentic loop instead of a single shot (gather tools -> generate ->
    // execute -> feed back -> repeat).
    match gather_mcp(&state, &req).await {
        Ok(g)
            if !g.routing.is_empty()
                || g.web_search.is_some()
                || g.forensics.is_some()
                || g.current_time.is_some() =>
        {
            if want_logprobs {
                return err(
                    StatusCode::BAD_REQUEST,
                    "unsupported_parameter",
                    "the logprobs include is not supported together with server-run tools \
                     (disable the server's MCP/web tools for this request, or drop the include)",
                );
            }
            // Fold into `req.instructions` - the SYSTEM slot - not just into
            // the message list. Compaction rebuilds the messages from
            // `req.instructions` (run_compacting_oa, and the resend paths),
            // so a block living only in `messages` survived the first pass and
            // then vanished the moment the conversation compacted, while the
            // user's own prompt carried on. The maintainer caught that. In the system
            // slot it is never summarized away and every rebuild re-renders
            // it. `messages` is patched too, since it was built before gather.
            req.instructions = merge_instructions(req.instructions.take(), &g.instructions);
            apply_server_instructions(&mut messages, &g.instructions);
            // The loop's context management: compaction
            // runs once, here, before the first generation. `truncation:"auto"`
            // stays live inside the loop, where the prompt grows.
            let item = match precompact_agent(
                &state,
                model,
                &req,
                &g.tools,
                &mut messages,
                compact_threshold,
                &scope,
            )
            .await
            {
                Ok(i) => i,
                Err(r) => return r,
            };
            // items are already prebuilt output-item Values (metadata + forensics)
            let fitems = enrichment_items;
            return if req.stream {
                stream_agent(
                    state.clone(),
                    req,
                    messages,
                    g,
                    Vec::new(),
                    item,
                    fitems,
                    scope,
                )
            } else {
                run_agent(
                    state.clone(),
                    req,
                    messages,
                    g,
                    Vec::new(),
                    item,
                    fitems,
                    scope,
                )
                .await
            };
        }
        Ok(_) => {}
        Err(e) => return err(StatusCode::BAD_REQUEST, "invalid_request_error", e),
    }

    let t_prep = std::time::Instant::now();
    let mut prepared = match prepare(
        model,
        &req,
        &messages,
        &[],
        state.max_output_ceiling,
        &state.sampling,
    ) {
        Ok(p) => p,
        Err(e) => return err(StatusCode::BAD_REQUEST, "invalid_request_error", e),
    };
    scope.tokenized(t_prep.elapsed());
    // Configured compaction fires on the RENDERED token count, and only over
    // a non-empty span (the whole prompt being the pending turn leaves
    // nothing to summarize). It runs before the gate: a prompt past the
    // threshold is usually also the one about to overflow.
    if let Some(threshold) = compact_threshold
        && prepared.prompt_ids.len() as u64 >= threshold
        && let Value::Array(items) = &req.input
        && crate::context_management::oa_tail_start(items) > 0
    {
        let items = items.clone();
        return run_compacting_oa(state.clone(), req, items, scope).await;
    }
    // Over-window prompt: clean 400 at the edge for stream and non-stream alike
    // (a streaming request has committed its 200 SSE status before the engine's
    // own admit check can answer). Prices image rows too - what prefill will
    // actually see, not just the token stream. With truncation "auto" the
    // spec'd fallback applies instead: drop conversation turns from the
    // BEGINNING (openai 2.53.0 semantics) until the prompt fits, reported
    // via the `truncation_dropped_items` extension field - never silent.
    let mut dropped = 0usize;
    while let Some(e) = crate::chat::context_gate(
        model,
        prepared.engine_prompt.len(),
        prepared.mm_chunks.as_deref(),
        state.max_ctx,
    ) {
        if req.truncation.as_deref() != Some("auto") {
            return crate::chat::engine_err(&e);
        }
        let n = drop_leading_turn(&mut messages);
        if n == 0 {
            // only the pending turn remains and it still does not fit
            return crate::chat::engine_err(&e);
        }
        dropped += n;
        prepared = match prepare(
            model,
            &req,
            &messages,
            &[],
            state.max_output_ceiling,
            &state.sampling,
        ) {
            Ok(p) => p,
            Err(e) => return err(StatusCode::BAD_REQUEST, "invalid_request_error", e),
        };
    }
    let meta = Meta {
        ex: Extras {
            truncation_auto: req.truncation.as_deref() == Some("auto"),
            dropped,
            compaction: None,
            enrichment: enrichment_items,
            ocr: prepared.ocr.clone(),
        },
        id: format!("resp_{}", uuid::Uuid::new_v4().simple()),
        model_id: model.id.clone(),
        tokenizer: model.tokenizer.clone(),
        prompt_len: prepared.prompt_ids.len(),
        text_prompt_len: prepared.engine_prompt.len(),
        media_is_audio: model.supports_audio,
        dialect: model.dialect,
        thinking_open: prepared.thinking_open,
        hints: prepared.hints.clone(),
        single_tool_call: prepared.single_tool_call,
        instructions: req.instructions.clone(),
        max_output_tokens: prepared.max_tokens,
        max_tool_calls: req.max_tool_calls,
        // echo what this turn was actually sampled at
        temperature: req
            .temperature
            .unwrap_or(state.sampling.resolve(prepared.thinking_open).temp),
        top_p: req
            .top_p
            .unwrap_or(state.sampling.resolve(prepared.thinking_open).top_p),
        tools: Value::Array(req.tools.clone().unwrap_or_default()),
        tool_choice: req.tool_choice.clone().unwrap_or_else(|| json!("auto")),
        text_format: req
            .text
            .as_ref()
            .and_then(|t| t.get("format"))
            .cloned()
            .unwrap_or_else(|| json!({"type": "text"})),
        want_logprobs,
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
        logprobs: logprobs_k,
        submitted: None, // stamped by Engine::submit
    };
    if let Err(e) = model.engine.submit(gen_req) {
        return err(StatusCode::INTERNAL_SERVER_ERROR, "internal_error", e);
    }

    if req.stream {
        stream_response(meta, rx)
    } else {
        collect_response(meta, rx).await
    }
}

/// Build the `output` array from a parsed assistant turn. `logprobs` rides
/// into the message's output_text part when the include asked for it (spec
/// shape: the part carries the full per-token array).
fn output_items(parsed: &Parsed, logprobs: Option<&[Value]>) -> Vec<Value> {
    let mut items = Vec::new();
    if let Some(reasoning) = &parsed.reasoning {
        items.push(json!({
            "type": "reasoning",
            "id": format!("rs_{}", uuid::Uuid::new_v4().simple()),
            "summary": [],
            "content": [{"type": "reasoning_text", "text": reasoning}],
        }));
    }
    if let Some(content) = &parsed.content {
        let mut part = json!({"type": "output_text", "text": content, "annotations": []});
        if let Some(lps) = logprobs {
            part["logprobs"] = json!(lps);
        }
        items.push(json!({
            "type": "message",
            "id": format!("msg_{}", uuid::Uuid::new_v4().simple()),
            "role": "assistant",
            "status": "completed",
            "content": [part],
        }));
    }
    for tc in &parsed.tool_calls {
        items.push(json!({
            "type": "function_call",
            "id": format!("fc_{}", uuid::Uuid::new_v4().simple()),
            "call_id": format!("call_{}", uuid::Uuid::new_v4().simple()),
            "name": tc.name,
            "arguments": tc.arguments,
            "status": "completed",
        }));
    }
    items
}

/// The full response object. `finish` None means still in progress (the
/// streaming `response.created` / `response.in_progress` snapshots).
fn response_object(
    meta: &Meta,
    status: &str,
    output: Vec<Value>,
    usage: Option<(usize, usize, usize)>,
    finish: Option<FinishReason>,
) -> Value {
    let incomplete = match finish {
        Some(FinishReason::Length) => json!({"reason": "max_output_tokens"}),
        _ => Value::Null,
    };
    let usage = match usage {
        None => Value::Null,
        Some((out_tokens, reasoning_tokens, cached)) => json!({
            "input_tokens": meta.prompt_len,
            // Same details object as chat completions' `prompt_tokens_details`
            // (one helper, so the accounting reads identically on every
            // surface) - `cache_write_tokens` is required here by the spec,
            // `audio_tokens` is the spec's own field and `image_tokens` is our
            // extension; both are omitted for a text-only request so the
            // documented shape is untouched.
            "input_tokens_details": paddock_api::completions::Usage::media_details(
                meta.prompt_len,
                cached,
                meta.prompt_len.saturating_sub(meta.text_prompt_len),
                meta.media_is_audio,
            ),
            "output_tokens": out_tokens,
            "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
            "total_tokens": meta.prompt_len + out_tokens,
        }),
    };
    let mut body = json!({
        "id": meta.id,
        "object": "response",
        "created_at": now_secs(),
        "status": status,
        "error": null,
        "incomplete_details": incomplete,
        "model": meta.model_id,
        "output": output,
        "instructions": meta.instructions,
        "max_output_tokens": meta.max_output_tokens,
        "max_tool_calls": meta.max_tool_calls,
        "metadata": {},
        "parallel_tool_calls": !meta.single_tool_call,
        "previous_response_id": null,
        "store": false,
        "temperature": meta.temperature,
        "text": {"format": meta.text_format},
        "tool_choice": meta.tool_choice,
        "tools": meta.tools,
        "top_p": meta.top_p,
        "truncation": if meta.ex.truncation_auto { "auto" } else { "disabled" },
        "usage": usage,
    });
    if meta.ex.dropped > 0 {
        body["truncation_dropped_items"] = json!(meta.ex.dropped);
    }
    // the OCR resolution echo (paddock extension, deepseek2-ocr only) - the
    // terminal points append grounded `regions` via `attach_ocr_regions`
    if let Some(o) = &meta.ex.ocr {
        body["ocr"] = o.echo();
    }
    body
}

/// Estimated token count of the reasoning text (same tokenizer, decoded
/// round trip) for usage.output_tokens_details.
fn reasoning_tokens(meta: &Meta, parsed: &Parsed) -> usize {
    parsed
        .reasoning
        .as_deref()
        .and_then(|r| meta.tokenizer.encode(r).ok())
        .map_or(0, |ids| ids.len())
}

/// Terminal status + event name for a finish reason.
fn terminal(finish: Option<FinishReason>) -> (&'static str, &'static str) {
    match finish {
        Some(FinishReason::Length) => ("incomplete", "response.incomplete"),
        _ => ("completed", "response.completed"),
    }
}

async fn collect_response(mut meta: Meta, mut rx: UnboundedReceiver<TokenEvent>) -> Response {
    let mut ids = Vec::new();
    let mut finish = None;
    let mut cached = 0usize;
    let mut lps: Vec<Value> = Vec::new();
    while let Some(ev) = rx.recv().await {
        match ev {
            // rows = what the engine actually prefilled; on an image request
            // that is the picture's expanded row run, not the single <image>
            // token the prompt tokenized to (see TokenEvent::Prefilled)
            TokenEvent::Prefilled { cached: c, rows } => {
                cached = c as usize;
                meta.prompt_len = meta.prompt_len.max(rows as usize);
            }
            TokenEvent::Token { id: t, logprobs } => {
                if meta.want_logprobs
                    && let Some(lp) = &logprobs
                {
                    lps.push(meta.lp_entry(t, lp));
                }
                ids.push(t);
            }
            TokenEvent::Done(r, stats) => {
                finish = Some(r);
                meta.scope.phases(&stats);
                break;
            }
            TokenEvent::Error(e) => {
                return crate::chat::engine_err(&e);
            }
        }
    }
    let parsed = meta.parse(&ids);
    let mut output = output_items(&parsed, meta.want_logprobs.then_some(lps.as_slice()));
    // Leading items, in order: a fired compaction, then always-on forensics
    // (they preprocessed the input), then the model's own output.
    let comp = meta.ex.compaction.is_some() as usize;
    if let Some(item) = &meta.ex.compaction {
        output.insert(0, item.clone());
    }
    for (k, item) in meta.ex.enrichment.iter().enumerate() {
        output.insert(comp + k, item.clone());
    }
    let rt = reasoning_tokens(&meta, &parsed);
    let (status, _) = terminal(finish);
    meta.scope.usage(meta.prompt_len, ids.len());
    meta.scope.cached(cached);
    meta.scope.finish(finish.map_or("stop", |f| f.as_str()));
    let mut body = response_object(&meta, status, output, Some((ids.len(), rt, cached)), finish);
    meta.attach_ocr_regions(&mut body, &ids);
    Json(body).into_response()
}

/// Streaming: emit the Responses event sequence for the text path plus
/// function_call items. Event names match the OpenAI vocabulary; each carries
/// a monotonic sequence_number.
fn stream_response(mut meta: Meta, mut rx: UnboundedReceiver<TokenEvent>) -> Response {
    let sse = stream! {
        let mut seq = 0u64;
        let mut next = || { let s = seq; seq += 1; s };

        let snapshot = response_object(&meta, "in_progress", vec![], None, None);
        yield ev("response.created", json!({"type":"response.created","sequence_number":next(),"response":snapshot}));
        let snapshot = response_object(&meta, "in_progress", vec![], None, None);
        yield ev("response.in_progress", json!({"type":"response.in_progress","sequence_number":next(),"response":snapshot}));

        // a fired compaction leads the stream: the item has no delta events
        // in the spec (opaque there, plaintext here) so it rides a complete
        // added/done pair at index 0, and every later item shifts by one
        let comp = meta.ex.compaction.is_some() as usize;
        // always-on forensics items follow the compaction (they preprocessed
        // the input, so they lead the model's own output), each a complete
        // added/done pair; reasoning/message then start after all of them.
        let base = comp + meta.ex.enrichment.len();
        if let Some(item) = &meta.ex.compaction {
            yield ev("response.output_item.added", json!({
                "type":"response.output_item.added","sequence_number":next(),
                "output_index":0,"item":item}));
            yield ev("response.output_item.done", json!({
                "type":"response.output_item.done","sequence_number":next(),
                "output_index":0,"item":item}));
        }
        for (k, item) in meta.ex.enrichment.iter().enumerate() {
            let idx = comp + k;
            yield ev("response.output_item.added", json!({
                "type":"response.output_item.added","sequence_number":next(),
                "output_index":idx,"item":item}));
            yield ev("response.output_item.done", json!({
                "type":"response.output_item.done","sequence_number":next(),
                "output_index":idx,"item":item}));
        }

        // reasoning item (index `base`) + message item stream lazily;
        // reasoning always precedes content in both dialects so its index is
        // `base` when present.
        let rs_id = format!("rs_{}", uuid::Uuid::new_v4().simple());
        let msg_id = format!("msg_{}", uuid::Uuid::new_v4().simple());
        let mut rs_open = false;
        let mut msg_open = false;
        let mut rs_emitted = 0usize;
        let mut emitted = 0usize;
        let mut ids: Vec<u32> = Vec::new();
        // incremental decode of `ids` (the O(n^2) per-token full re-decode
        // was the long-stream collapse under concurrency)
        let mut sd = meta.tokenizer.stream_decoder(false);
        let mut finish = None;
        let mut cached = 0usize;
        // logprob entries not yet carried by a text delta, and the full run -
        // concat(per-delta arrays) == the done event's array, always
        let mut lp_pending: Vec<Value> = Vec::new();
        let mut lp_all: Vec<Value> = Vec::new();

        loop {
            match rx.recv().await {
                Some(TokenEvent::Prefilled { cached: c, rows }) => {
                    cached = c as usize;
                    meta.prompt_len = meta.prompt_len.max(rows as usize);
                }
                Some(TokenEvent::Token { id: t, logprobs }) => {
                    if meta.want_logprobs
                        && let Some(lp) = &logprobs
                    {
                        let e = meta.lp_entry(t, lp);
                        lp_pending.push(e.clone());
                        lp_all.push(e);
                    }
                    ids.push(t);
                    let raw = sd.push(&meta.tokenizer, t);
                    let parsed = meta.parse_raw(&raw);

                    // reasoning channel -> reasoning item at index `base`
                    if let Some(reasoning) = &parsed.reasoning
                        && safe_len(reasoning, meta.dialect.reasoning_markers()) > rs_emitted {
                        if !rs_open {
                            rs_open = true;
                            yield ev("response.output_item.added", json!({
                                "type":"response.output_item.added","sequence_number":next(),
                                "output_index":base,
                                "item":{"type":"reasoning","id":rs_id,"summary":[],"content":[]}}));
                        }
                        let safe = safe_len(reasoning, meta.dialect.reasoning_markers());
                        let delta = reasoning[rs_emitted..safe].to_owned();
                        rs_emitted = safe;
                        yield ev("response.reasoning_text.delta", json!({
                            "type":"response.reasoning_text.delta","sequence_number":next(),
                            "item_id":rs_id,"output_index":base,"content_index":0,"delta":delta}));
                    }

                    // final channel -> message item
                    if let Some(content) = &parsed.content
                        && safe_len(content, meta.dialect.content_markers()) > emitted {
                        let msg_index = base + rs_open as usize;
                        if !msg_open {
                            msg_open = true;
                            yield ev("response.output_item.added", json!({
                                "type":"response.output_item.added","sequence_number":next(),
                                "output_index":msg_index,
                                "item":{"type":"message","id":msg_id,"role":"assistant",
                                        "status":"in_progress","content":[]}}));
                            yield ev("response.content_part.added", json!({
                                "type":"response.content_part.added","sequence_number":next(),
                                "item_id":msg_id,"output_index":msg_index,"content_index":0,
                                "part":{"type":"output_text","text":"","annotations":[]}}));
                        }
                        let safe = safe_len(content, meta.dialect.content_markers());
                        let delta = content[emitted..safe].to_owned();
                        emitted = safe;
                        // logprobs is required on text delta/done events in the
                        // current spec (empty when not requested); with the
                        // include set, a delta carries every entry sampled
                        // since the previous text delta
                        yield ev("response.output_text.delta", json!({
                            "type":"response.output_text.delta","sequence_number":next(),
                            "item_id":msg_id,"output_index":msg_index,"content_index":0,
                            "delta":delta,"logprobs":std::mem::take(&mut lp_pending)}));
                    }
                }
                Some(TokenEvent::Done(r, stats)) => { finish = Some(r); meta.scope.phases(&stats); break }
                None => break,
                Some(TokenEvent::Error(e)) => {
                    // honest terminal event; the SDK surfaces response.failed
                    let mut snapshot = response_object(&meta, "failed", vec![], None, None);
                    snapshot["error"] = json!({"code": e.code.unwrap_or("server_error"), "message": e.message});
                    yield ev("response.failed", json!({
                        "type":"response.failed","sequence_number":next(),"response":snapshot}));
                    return;
                }
            }
        }

        let parsed = meta.parse(&ids);

        // close the reasoning item
        if rs_open {
            let text = parsed.reasoning.clone().unwrap_or_default();
            yield ev("response.reasoning_text.done", json!({
                "type":"response.reasoning_text.done","sequence_number":next(),
                "item_id":rs_id,"output_index":base,"content_index":0,"text":text}));
            yield ev("response.output_item.done", json!({
                "type":"response.output_item.done","sequence_number":next(),"output_index":base,
                "item":{"type":"reasoning","id":rs_id,"summary":[],
                        "content":[{"type":"reasoning_text","text":text}]}}));
        }
        let msg_index = base + rs_open as usize;

        // close the message item; the done event carries the full logprobs
        // run, and the part itself carries it too when the include asked
        if msg_open {
            let text = parsed.content.clone().unwrap_or_default();
            yield ev("response.output_text.done", json!({
                "type":"response.output_text.done","sequence_number":next(),
                "item_id":msg_id,"output_index":msg_index,"content_index":0,
                "text":text,"logprobs":lp_all}));
            let mut part = json!({"type":"output_text","text":text,"annotations":[]});
            if meta.want_logprobs {
                part["logprobs"] = json!(lp_all);
            }
            yield ev("response.content_part.done", json!({
                "type":"response.content_part.done","sequence_number":next(),
                "item_id":msg_id,"output_index":msg_index,"content_index":0,
                "part":part}));
            yield ev("response.output_item.done", json!({
                "type":"response.output_item.done","sequence_number":next(),"output_index":msg_index,
                "item":{"type":"message","id":msg_id,"role":"assistant","status":"completed",
                        "content":[part]}}));
        }

        // function_call items (parallel-capable), after reasoning + message
        let fc_base = base + rs_open as usize + msg_open as usize;
        for (n, tc) in parsed.tool_calls.iter().enumerate() {
            let idx = fc_base + n;
            let fc_id = format!("fc_{}", uuid::Uuid::new_v4().simple());
            let call_id = format!("call_{}", uuid::Uuid::new_v4().simple());
            let item = json!({"type":"function_call","id":fc_id,"call_id":call_id,
                              "name":tc.name,"arguments":"","status":"in_progress"});
            yield ev("response.output_item.added", json!({
                "type":"response.output_item.added","sequence_number":next(),
                "output_index":idx,"item":item}));
            yield ev("response.function_call_arguments.delta", json!({
                "type":"response.function_call_arguments.delta","sequence_number":next(),
                "item_id":fc_id,"output_index":idx,"delta":tc.arguments}));
            yield ev("response.function_call_arguments.done", json!({
                "type":"response.function_call_arguments.done","sequence_number":next(),
                "item_id":fc_id,"output_index":idx,"arguments":tc.arguments}));
            yield ev("response.output_item.done", json!({
                "type":"response.output_item.done","sequence_number":next(),"output_index":idx,
                "item":{"type":"function_call","id":fc_id,"call_id":call_id,
                        "name":tc.name,"arguments":tc.arguments,"status":"completed"}}));
        }

        let mut output = output_items(&parsed, meta.want_logprobs.then_some(lp_all.as_slice()));
        if let Some(item) = &meta.ex.compaction {
            output.insert(0, item.clone());
        }
        let rt = reasoning_tokens(&meta, &parsed);
        let (status, event_name) = terminal(finish);
        meta.scope.usage(meta.prompt_len, ids.len());
        meta.scope.cached(cached);
        meta.scope.finish(finish.map_or("stop", |f| f.as_str()));
        let mut full = response_object(&meta, status, output, Some((ids.len(), rt, cached)), finish);
        meta.attach_ocr_regions(&mut full, &ids);
        yield ev(event_name, json!({
            "type":event_name,"sequence_number":next(),"response":full}));
    };
    Sse::new(sse).into_response()
}

fn ev(name: &str, data: Value) -> Result<Event, std::convert::Infallible> {
    Ok(Event::default().event(name).data(data.to_string()))
}

/// Emit-safe prefix length: hold back a partial marker tail, on a char
/// boundary.
fn safe_len(s: &str, markers: &[&str]) -> usize {
    let mut n = s.len() - holdback(s, markers);
    while n > 0 && !s.is_char_boundary(n) {
        n -= 1;
    }
    n
}

// ── context management  ──────────────────────────────────

/// truncation "auto": remove the leading conversation turn - the first
/// non-system message plus everything up to the next user message, so
/// tool-call/output pairs leave together and the conversation still opens
/// with a user turn. The PENDING TURN is never touched: dropping stops at the
/// last plain user message, the same boundary the compact span/tail split
/// uses. (It used to stop one short of the array's end, which on a
/// conversation ending in tool output - every agent-loop round - would have
/// drained the user's actual question and kept a dangling tool result.)
/// Returns how many messages were removed (0 = nothing droppable left).
fn drop_leading_turn(messages: &mut Vec<Value>) -> usize {
    let pending = crate::context_management::compact_tail_start(messages);
    let Some(first) = messages
        .iter()
        .position(|m| m.get("role").and_then(Value::as_str) != Some("system"))
    else {
        return 0;
    };
    if first + 1 > pending {
        return 0;
    }
    let mut end = first + 1;
    while end < pending && messages[end].get("role").and_then(Value::as_str) != Some("user") {
        end += 1;
    }
    messages.drain(first..end);
    end - first
}

/// The Responses compaction item (openai 2.53.0 `ResponseCompactionItem`).
/// `encrypted_content` is the required field the SDK types round-trip, so
/// the plaintext summary lives there - no encryption theater on a local box
/// (documented deviation; their servers encrypt because their state is
/// private, ours is the caller's own conversation).
fn compaction_item(summary: &str) -> Value {
    json!({"id": format!("cp_{}", uuid::Uuid::new_v4().simple()),
           "type": "compaction", "encrypted_content": summary})
}

/// A minimal ResponsesRequest for server-initiated passes (the standalone
/// compact endpoint has its own request type). Built through serde so the
/// field defaults stay defined in one place.
fn blank_request() -> ResponsesRequest {
    serde_json::from_value(json!({"model": "", "input": ""})).expect("minimal request parses")
}

/// Doctor a request clone for the summarization pass: greedy sampling, no
/// output constraint, a summary-sized output cap, thinking off (suffix-only
/// in every template family, so the span prefix still cache-matches the
/// conversation being compacted). Tools stay in the render for the same
/// prefix-match reason - unless the request hid them with tool_choice
/// "none"; a forced tool_choice must not force the SUMMARY, so it relaxes
/// to auto.
fn greedy_for_summary(mut r: ResponsesRequest) -> ResponsesRequest {
    r.temperature = Some(0.0);
    r.presence_penalty = Some(0.0);
    r.frequency_penalty = Some(0.0);
    r.text = None;
    r.max_output_tokens = 4096;
    if r.tool_choice
        .as_ref()
        .is_some_and(|tc| tc.as_str() != Some("none"))
    {
        r.tool_choice = None;
    }
    let mut kw = r
        .chat_template_kwargs
        .as_ref()
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    kw.insert("enable_thinking".into(), json!(false));
    r.chat_template_kwargs = Some(Value::Object(kw));
    r.stream = false;
    r.context_management = None;
    r
}

/// One summarization generation over `msgs1` (which must end with the
/// summarization instruction turn). Returns (summary, prompt_rows, cached,
/// output_tokens); Err is a ready-to-return error response. A None summary
/// is the failed-compaction case (the model produced nothing usable).
/// `lenient_gate`: an over-window span reads as a FAILED compaction (None)
/// instead of an error - in-create compaction is opportunistic and the
/// request must fall through to `truncation:"auto"` when it's armed; the
/// trigger and the standalone endpoint were ASKED to compact, so for them
/// the loud over-window error stands.
// Err is axum's own Response by design: a ready-to-return body, not a boxed error
#[allow(clippy::result_large_err)]
async fn summary_pass(
    state: &Arc<AppState>,
    model: &ServingModel,
    req1: &ResponsesRequest,
    msgs1: &[Value],
    // The agent loop's discovered MCP tools: the span render must carry the
    // same tools as the live conversation or its prefix diverges and the
    // summarization prefill loses the radix hit that makes it nearly free.
    extra_tools: &[Value],
    scope: &crate::events::EventScope,
    lenient_gate: bool,
) -> Result<(Option<String>, usize, usize, usize), Response> {
    let prepared = prepare(
        model,
        req1,
        msgs1,
        extra_tools,
        state.max_output_ceiling,
        &state.sampling,
    )
    .map_err(|e| err(StatusCode::BAD_REQUEST, "invalid_request_error", e))?;
    // even the span + instructions overflows: compaction cannot rescue this
    // conversation
    if let Some(e) = crate::chat::context_gate(
        model,
        prepared.engine_prompt.len(),
        prepared.mm_chunks.as_deref(),
        state.max_ctx,
    ) {
        if lenient_gate {
            return Ok((None, 0, 0, 0));
        }
        return Err(crate::chat::engine_err(&e));
    }
    let constraint = instantiate_constraint(
        &prepared.constraint_spec,
        prepared.gate,
        model,
        prepared.think_budget.as_ref(),
    );
    let (tx, mut rx) = unbounded_channel();
    let gen1 = GenRequest {
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
    if let Err(e) = model.engine.submit(gen1) {
        return Err(err(StatusCode::INTERNAL_SERVER_ERROR, "internal_error", e));
    }
    let mut ids: Vec<u32> = Vec::new();
    let mut cached = 0usize;
    let mut p_len = prepared.prompt_ids.len();
    while let Some(evt) = rx.recv().await {
        match evt {
            TokenEvent::Prefilled { cached: c, rows } => {
                cached = c as usize;
                p_len = p_len.max(rows as usize);
            }
            TokenEvent::Token { id: t, .. } => ids.push(t),
            TokenEvent::Done(_, stats) => {
                scope.phases(&stats);
                break;
            }
            TokenEvent::Error(e) => return Err(crate::chat::engine_err(&e)),
        }
    }
    let raw = model.tokenizer.decode(&ids, false).unwrap_or_default();
    let parsed = parse(model.dialect, &raw, prepared.thinking_open, None);
    let summary = parsed
        .content
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned);
    Ok((summary, p_len, cached, ids.len()))
}

/// Round-0 context management for the agent loop. The
/// compaction orchestration runs once, here, on the request's own prompt -
/// the same measurement point a non-agent request uses - and never again
/// inside the loop. Three reasons that is the right pin, not a shortcut:
/// - the compact SPAN is fixed for the whole loop (the loop only appends
///   after the pending user item), so compacting at round 5 would remove
///   exactly what compacting here removes;
/// - firing before the first generation keeps the compaction item at
///   output[0], which is the single-shot contract AND the only position
///   where the spec'd resend rewrite ("everything before the item collapses
///   into its summary") is true of our summary - an item emitted mid-loop
///   would tell the client to drop tool round-trips the summary never saw;
/// - mid-turn tool bloat is the Anthropic dialect's `clear_tool_uses` job,
///   and `truncation:"auto"` backstops it here (applied every round).
///   Returns the compaction item to lead the output with. None = no threshold,
///   trigger unmet, empty span, or a failed summarization - the absent item is
///   the OpenAI dialect's "compaction did not happen" signal. Err = a
///   ready-to-return error response.
// Err is axum's own Response by design: a ready-to-return body, not a boxed error
#[allow(clippy::too_many_arguments, clippy::result_large_err)]
async fn precompact_agent(
    state: &Arc<AppState>,
    model: &ServingModel,
    req: &ResponsesRequest,
    extra_tools: &[Value],
    messages: &mut Vec<Value>,
    threshold: Option<u64>,
    scope: &crate::events::EventScope,
) -> Result<Option<Value>, Response> {
    let Some(threshold) = threshold else {
        return Ok(None);
    };
    let Value::Array(items) = &req.input else {
        return Ok(None);
    };
    let tail_start = crate::context_management::oa_tail_start(items);
    if tail_start == 0 {
        return Ok(None); // the whole input is the pending turn: nothing to summarize
    }
    // one extra render+tokenize, and only for a request that armed the
    // threshold: the loop's round 0 re-prepares over the compacted messages
    let prepared = prepare(
        model,
        req,
        messages,
        extra_tools,
        state.max_output_ceiling,
        &state.sampling,
    )
    .map_err(|e| err(StatusCode::BAD_REQUEST, "invalid_request_error", e))?;
    if (prepared.prompt_ids.len() as u64) < threshold {
        return Ok(None);
    }
    let span = Value::Array(items[..tail_start].to_vec());
    let mut msgs1 = messages_from_input(req.instructions.as_deref(), &span)
        .map_err(|e| err(StatusCode::BAD_REQUEST, "invalid_request_error", e))?;
    msgs1.push(json!({"role": "user",
                      "content": crate::context_management::DEFAULT_COMPACT_INSTRUCTIONS}));
    let req1 = greedy_for_summary(req.clone());
    let (summary, ..) = summary_pass(state, model, &req1, &msgs1, extra_tools, scope, true).await?;
    // failed summary -> the loop runs uncompacted with no item, exactly as the
    // single-shot path does (and truncation "auto", if armed, still catches
    // an over-window prompt inside the loop)
    let Some(s) = summary else { return Ok(None) };
    let items2 = crate::context_management::oa_compacted_items(items, &s);
    *messages = messages_from_input(req.instructions.as_deref(), &Value::Array(items2))
        .map_err(|e| err(StatusCode::BAD_REQUEST, "invalid_request_error", e))?;
    Ok(Some(compaction_item(&s)))
}

/// The in-create compaction orchestration (`context_management` threshold
/// met): iteration 1 summarizes the span greedily, iteration 2 answers over
/// [framed summary + tail]. Same cache algebra as the /v1/messages twin:
/// iteration 1 renders [same system, same tools, span, instructions] so its
/// prefill rides the radix cache of the conversation being compacted, and
/// iteration 2's item list is exactly what a later resend rewrites to.
/// The official Response has no `usage.iterations` slot, so the pass's own
/// numbers stay off the wire; top-level usage covers the final generation.
async fn run_compacting_oa(
    state: Arc<AppState>,
    req: ResponsesRequest,
    items: Vec<Value>,
    scope: crate::events::EventScope,
) -> Response {
    let Some(model) = state.serving.as_ref() else {
        return err(
            StatusCode::SERVICE_UNAVAILABLE,
            "model_not_loaded",
            "no model is loaded",
        );
    };
    let tail_start = crate::context_management::oa_tail_start(&items);
    let span = Value::Array(items[..tail_start].to_vec());
    let mut msgs1 = match messages_from_input(req.instructions.as_deref(), &span) {
        Ok(m) => m,
        Err(e) => return err(StatusCode::BAD_REQUEST, "invalid_request_error", e),
    };
    msgs1.push(json!({"role": "user",
                      "content": crate::context_management::DEFAULT_COMPACT_INSTRUCTIONS}));
    let req1 = greedy_for_summary(req.clone());
    let (summary, _p1_len, _cached1, _out1) =
        match summary_pass(&state, model, &req1, &msgs1, &[], &scope, true).await {
            Ok(x) => x,
            Err(r) => return r,
        };

    // iteration 2: the real generation over [framed summary + tail]. A
    // failed summary proceeds UNCOMPACTED with no compaction item - the
    // OpenAI dialect has no failed-compaction shape, and the absent item is
    // the honest signal (documented pin); if the uncompacted prompt then
    // overflows, the loud gate error stands.
    let items2 = match &summary {
        Some(s) => crate::context_management::oa_compacted_items(&items, s),
        None => items,
    };
    let mut messages = match messages_from_input(req.instructions.as_deref(), &Value::Array(items2))
    {
        Ok(m) => m,
        Err(e) => return err(StatusCode::BAD_REQUEST, "invalid_request_error", e),
    };
    let mut prepared = match prepare(
        model,
        &req,
        &messages,
        &[],
        state.max_output_ceiling,
        &state.sampling,
    ) {
        Ok(p) => p,
        Err(e) => return err(StatusCode::BAD_REQUEST, "invalid_request_error", e),
    };
    // Same truncation:"auto" fallback as the plain path: an armed backstop
    // must catch the compaction-couldn't-run corner too (span over-window ->
    // failed compaction -> the uncompacted prompt still overflows), or the
    // combination the Studio sends would 400 exactly where it opted into
    // being saved. Without "auto" the loud gate error stands, as pinned.
    let mut dropped = 0usize;
    while let Some(e) = crate::chat::context_gate(
        model,
        prepared.engine_prompt.len(),
        prepared.mm_chunks.as_deref(),
        state.max_ctx,
    ) {
        if req.truncation.as_deref() != Some("auto") {
            return crate::chat::engine_err(&e);
        }
        let n = drop_leading_turn(&mut messages);
        if n == 0 {
            return crate::chat::engine_err(&e);
        }
        dropped += n;
        prepared = match prepare(
            model,
            &req,
            &messages,
            &[],
            state.max_output_ceiling,
            &state.sampling,
        ) {
            Ok(p) => p,
            Err(e) => return err(StatusCode::BAD_REQUEST, "invalid_request_error", e),
        };
    }
    let meta = Meta {
        id: format!("resp_{}", uuid::Uuid::new_v4().simple()),
        model_id: model.id.clone(),
        tokenizer: model.tokenizer.clone(),
        prompt_len: prepared.prompt_ids.len(),
        text_prompt_len: prepared.engine_prompt.len(),
        media_is_audio: model.supports_audio,
        dialect: model.dialect,
        thinking_open: prepared.thinking_open,
        hints: prepared.hints.clone(),
        single_tool_call: prepared.single_tool_call,
        instructions: req.instructions.clone(),
        max_output_tokens: prepared.max_tokens,
        max_tool_calls: req.max_tool_calls,
        // echo what this turn was actually sampled at
        temperature: req
            .temperature
            .unwrap_or(state.sampling.resolve(prepared.thinking_open).temp),
        top_p: req
            .top_p
            .unwrap_or(state.sampling.resolve(prepared.thinking_open).top_p),
        tools: Value::Array(req.tools.clone().unwrap_or_default()),
        tool_choice: req.tool_choice.clone().unwrap_or_else(|| json!("auto")),
        text_format: req
            .text
            .as_ref()
            .and_then(|t| t.get("format"))
            .cloned()
            .unwrap_or_else(|| json!({"type": "text"})),
        ex: Extras {
            truncation_auto: req.truncation.as_deref() == Some("auto"),
            dropped,
            compaction: summary.as_deref().map(compaction_item),
            enrichment: Vec::new(),
            ocr: prepared.ocr.clone(),
        },
        want_logprobs: lane_want_logprobs(&req),
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
        logprobs: lane_want_logprobs(&req).then(|| req.top_logprobs.unwrap_or(0)),
        submitted: None, // stamped by Engine::submit
    };
    if let Err(e) = model.engine.submit(gen_req) {
        return err(StatusCode::INTERNAL_SERVER_ERROR, "internal_error", e);
    }
    if req.stream {
        stream_response(meta, rx)
    } else {
        collect_response(meta, rx).await
    }
}

/// `compaction_trigger` (final input item): summarize the whole conversation
/// - no tail split, the caller is archiving - and answer with only the
///   compaction item. The Responses twin of `pause_after_compaction`; usage
///   covers the summarization pass, the only generation that ran.
async fn run_compact_trigger(
    state: Arc<AppState>,
    req: ResponsesRequest,
    mut messages: Vec<Value>,
    scope: crate::events::EventScope,
) -> Response {
    let Some(model) = state.serving.as_ref() else {
        return err(
            StatusCode::SERVICE_UNAVAILABLE,
            "model_not_loaded",
            "no model is loaded",
        );
    };
    messages.push(json!({"role": "user",
                         "content": crate::context_management::DEFAULT_COMPACT_ALL_INSTRUCTIONS}));
    let req1 = greedy_for_summary(req.clone());
    let (summary, p1_len, cached1, out1) =
        match summary_pass(&state, model, &req1, &messages, &[], &scope, false).await {
            Ok(x) => x,
            Err(r) => return r,
        };
    let Some(s) = summary else {
        return err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            "compaction produced an empty summary",
        );
    };
    let item = compaction_item(&s);
    let meta = Meta {
        id: format!("resp_{}", uuid::Uuid::new_v4().simple()),
        model_id: model.id.clone(),
        tokenizer: model.tokenizer.clone(),
        prompt_len: p1_len,
        text_prompt_len: p1_len,
        media_is_audio: model.supports_audio,
        dialect: model.dialect,
        thinking_open: false,
        hints: None,
        single_tool_call: req.parallel_tool_calls == Some(false),
        instructions: req.instructions.clone(),
        max_output_tokens: req.max_output_tokens,
        max_tool_calls: req.max_tool_calls,
        // the compaction pass runs with thinking closed, so it echoes the
        // instruct-side election
        temperature: req
            .temperature
            .unwrap_or(state.sampling.resolve(false).temp),
        top_p: req.top_p.unwrap_or(state.sampling.resolve(false).top_p),
        tools: Value::Array(req.tools.clone().unwrap_or_default()),
        tool_choice: req.tool_choice.clone().unwrap_or_else(|| json!("auto")),
        text_format: req
            .text
            .as_ref()
            .and_then(|t| t.get("format"))
            .cloned()
            .unwrap_or_else(|| json!({"type": "text"})),
        ex: Extras::default(),
        // a compaction answer has no message text for entries to ride on
        want_logprobs: false,
        scope,
    };
    meta.scope.usage(p1_len, out1);
    meta.scope.cached(cached1);
    meta.scope.finish("compaction");
    let full = response_object(
        &meta,
        "completed",
        vec![item.clone()],
        Some((out1, 0, cached1)),
        None,
    );
    if req.stream {
        let snap = response_object(&meta, "in_progress", vec![], None, None);
        let sse = stream! {
            let mut seq = 0u64;
            let mut next = || { let s = seq; seq += 1; s };
            yield ev("response.created", json!({
                "type":"response.created","sequence_number":next(),"response":snap}));
            yield ev("response.in_progress", json!({
                "type":"response.in_progress","sequence_number":next(),"response":snap}));
            yield ev("response.output_item.added", json!({
                "type":"response.output_item.added","sequence_number":next(),
                "output_index":0,"item":item}));
            yield ev("response.output_item.done", json!({
                "type":"response.output_item.done","sequence_number":next(),
                "output_index":0,"item":item}));
            yield ev("response.completed", json!({
                "type":"response.completed","sequence_number":next(),"response":full}));
        };
        return Sse::new(sse).into_response();
    }
    Json(full).into_response()
}

/// `POST /v1/responses/compact` (openai 2.53.0): the compaction executor
/// standalone. Returns their `CompactedResponse` shape - object
/// "response.compaction", output = the conversation's user message items
/// followed by one compaction item, usage = the summarization pass. The
/// summarization prefill rides the radix cache of the conversation being
/// compacted when the conversation was served tool-less with the same
/// instructions (this endpoint has no tools parameter on the wire).
pub async fn handle_compact(
    State(state): State<Arc<AppState>>,
    scope: Option<axum::Extension<crate::events::EventScope>>,
    crate::extract::OaiJson(req): crate::extract::OaiJson<paddock_api::responses::CompactRequest>,
) -> Response {
    let scope = scope.map(|e| e.0).unwrap_or_default();
    let Some(model) = state.serving.as_ref() else {
        return err(
            StatusCode::SERVICE_UNAVAILABLE,
            "model_not_loaded",
            "no model is loaded",
        );
    };
    scope.model(&model.id);
    if req.previous_response_id.is_some() {
        return err(
            StatusCode::BAD_REQUEST,
            "unsupported_parameter",
            "responses are not persisted on this server; send the full conversation as input",
        );
    }
    let mut items = match &req.input {
        Value::String(s) => vec![json!({"type": "message", "role": "user", "content": s})],
        Value::Array(a) => a.clone(),
        _ => {
            return err(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "`input` must be a string or an array of input items",
            );
        }
    };
    // an already-compacted conversation re-compacts over its rewritten form;
    // a stray compaction_trigger is redundant here (this endpoint is the
    // trigger) and is consumed, but still must be the final item
    if let Some(rw) = crate::context_management::oa_resend_rewrite(&items) {
        items = rw;
    }
    if let Err(e) = crate::context_management::oa_take_trigger(&mut items) {
        return err(StatusCode::BAD_REQUEST, "invalid_request_error", e);
    }
    if items.is_empty() {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "nothing to compact",
        );
    }
    // attachments expand exactly like create, so the summary covers file
    // CONTENT, not filenames
    let opts = match crate::chat::attach_opts(None, None, None, None) {
        Ok(o) => o,
        Err(e) => return err(StatusCode::BAD_REQUEST, "invalid_request_error", e),
    };
    // Internal compaction/summary pass - not a user-facing response, so no
    // forensics output item; injection into the summarized prompt is enough.
    let compact_map =
        match crate::chat::expand_attachments(&state, model, &mut items, opts, &mut Vec::new())
            .await
        {
            Ok(sample) => sample,
            Err((code, msg)) => {
                let kind = if code == StatusCode::BAD_REQUEST {
                    "invalid_request_error"
                } else {
                    "internal_error"
                };
                return err(code, kind, msg);
            }
        };
    let mut msgs1 =
        match messages_from_input(req.instructions.as_deref(), &Value::Array(items.clone())) {
            Ok(m) => m,
            Err(e) => return err(StatusCode::BAD_REQUEST, "invalid_request_error", e),
        };
    // A summary pass reads the same prompt the answer would, capability and
    // all - otherwise the two disagree about what the model was told, and the
    // prefix cache is aligned against a prompt that never ran.
    if let Some(sample) = compact_map.as_deref() {
        crate::doc::add_map_capability(&mut msgs1, sample);
    }
    msgs1.push(json!({"role": "user",
                      "content": crate::context_management::DEFAULT_COMPACT_ALL_INSTRUCTIONS}));
    let req1 = greedy_for_summary(blank_request());
    let (summary, p1_len, cached1, out1) =
        match summary_pass(&state, model, &req1, &msgs1, &[], &scope, false).await {
            Ok(x) => x,
            Err(r) => return r,
        };
    let Some(s) = summary else {
        return err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            "compaction produced an empty summary",
        );
    };
    scope.usage(p1_len, out1);
    scope.cached(cached1);
    scope.finish("compaction");
    // their documented output: "a list of all user messages, followed by a
    // single compaction item" - the user items echo verbatim (they are
    // already valid input items for the next request), with the "message"
    // discriminator made explicit so SDK unions parse the echo
    let mut output: Vec<Value> = items
        .iter()
        .filter(|it| crate::context_management::oa_user_message(it))
        .cloned()
        .map(|mut it| {
            if it.get("type").is_none() {
                it["type"] = json!("message");
            }
            it
        })
        .collect();
    output.push(compaction_item(&s));
    Json(json!({
        "id": format!("cr_{}", uuid::Uuid::new_v4().simple()),
        "created_at": now_secs(),
        "object": "response.compaction",
        "output": output,
        "usage": {
            "input_tokens": p1_len,
            "input_tokens_details": paddock_api::completions::Usage::media_details(p1_len, cached1, 0, false),
            "output_tokens": out1,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": p1_len + out1,
        },
    }))
    .into_response()
}

// ── MCP agent loop ──────────────────────────────────────────────────────────

/// Pending human-in-the-loop approvals, keyed by approval id. The streaming
/// agent loop registers a channel before it emits an `mcp_approval_request` and
/// parks on the receiver; the `/api/mcp-approvals/{id}` endpoint resolves it
/// (approve = true / deny = false). Lives in `AppState`, shared across requests.
#[derive(Default)]
pub struct ApprovalGate {
    pending: std::sync::Mutex<HashMap<String, tokio::sync::oneshot::Sender<bool>>>,
}

impl ApprovalGate {
    /// Arm an approval; returns the receiver the loop awaits.
    pub fn register(&self, id: String) -> tokio::sync::oneshot::Receiver<bool> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(id, tx);
        rx
    }
    /// Resolve a pending approval; returns false if the id is unknown (already
    /// decided, timed out, or never existed).
    pub fn resolve(&self, id: &str, approve: bool) -> bool {
        match self
            .pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(id)
        {
            Some(tx) => tx.send(approve).is_ok(),
            None => false,
        }
    }
    /// Drop an armed approval without deciding it (timeout / stream aborted).
    pub fn cancel(&self, id: &str) {
        self.pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(id);
    }
}

/// A gated MCP call awaiting the client's approve/deny decision (spec resubmit).
#[derive(Clone)]
pub struct PendingCall {
    pub approval_id: String,
    pub call_id: String,
    pub ns_name: String,
    pub args: String,
}

/// The loop state a response was paused in, awaiting MCP approvals. The client
/// resumes it with `previous_response_id` + `mcp_approval_response` input items.
pub struct PendingApproval {
    /// Transcript up to + including the assistant tool-calls turn (with results
    /// for any auto-run calls); the gated calls' results are filled on resume.
    pub messages: Vec<Value>,
    pub pending: Vec<PendingCall>,
    pub created_ms: u64,
}

/// A resolved approval to apply when resuming (execute if approved, else deny).
struct Decision {
    approval_id: String,
    call_id: String,
    ns_name: String,
    args: String,
    approved: bool,
}

/// In-memory store of responses paused on MCP approval, keyed by response id.
/// Bounded (oldest evicted); lost on restart - documented, fine for a local
/// single-user server. This is the OpenAI `previous_response_id` continuation.
#[derive(Default)]
pub struct ApprovalStore {
    map: std::sync::Mutex<HashMap<String, PendingApproval>>,
}

impl ApprovalStore {
    const CAP: usize = 128;
    pub fn insert(&self, id: String, p: PendingApproval) {
        let mut m = self
            .map
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if m.len() >= Self::CAP
            && let Some(oldest) = m
                .iter()
                .min_by_key(|(_, v)| v.created_ms)
                .map(|(k, _)| k.clone())
        {
            m.remove(&oldest);
        }
        m.insert(id, p);
    }
    pub fn take(&self, id: &str) -> Option<PendingApproval> {
        self.map
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(id)
    }
}

/// Wall-clock millis (for approval-cache eviction ordering).
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// One MCP server's discovered (and `allowed_tools`-filtered) tools, for the
/// spec `mcp_list_tools` output item emitted before generation.
struct Listing {
    server_label: String,
    tools: Vec<paddock_mcp::McpTool>,
}

/// Tools gathered from the request's MCP servers, ready to inject, plus a map
/// from the namespaced tool name the model sees back to (server, real name),
/// the set of namespaced tools whose calls require human approval, and the
/// per-server tool listings for `mcp_list_tools`.
struct Gathered {
    /// What actually goes into the model's prompt: the schemas of every server
    /// that fit the disclosure budget, plus the two synthetic search/call tools.
    tools: Vec<Value>,
    routing: HashMap<String, (paddock_mcp::ServerConfig, String)>,
    needs_approval: HashSet<String>,
    listings: Vec<Listing>,
    /// The full namespaced tool set, searchable via `mcp_search_tools` regardless
    /// of disclosure mode (also how `mcp_call_tool` resolves a discovered name).
    catalog: Vec<crate::tool_search::CatalogTool>,
    /// Set when the request carries a `web_search` tool and a provider is
    /// configured: the agent loop executes those calls server-side.
    web_search: Option<crate::websearch::WebSpec>,
    /// Set when the request carries a `{"type":"forensics"}` tool and the runner
    /// has `[forensics] tool = true`: the agent loop executes forensic analysis
    /// server-side over an image already in the conversation.
    forensics: Option<std::sync::Arc<crate::forensics::ForensicRuntime>>,
    /// Set when the request carries a `{"type":"current_time"}` tool: the agent
    /// loop answers clock calls itself, in the declared timezone.
    current_time: Option<crate::clock::ClockSpec>,
    /// Each connected server's handshake `instructions`, in declaration order.
    /// The MCP spec means these to reach the model like system-prompt text;
    /// `apply_server_instructions` folds them in.
    instructions: Vec<String>,
}

/// Fold the MCP servers' own `instructions` into the prompt, after the user's
/// system prompt rather than instead of it - a server describes how to use
/// itself, it does not get to overwrite what the user asked for.
///
/// Without this a server's guidance is collected at initialize and discarded,
/// which is exactly how "put pages in an artifact, not in your reply" failed
/// to reach a model that had the tool declared and connected.
/// Tool guidance first, the user's prompt LAST.
///
/// Order is not cosmetic. With the user's line leading and ~200 words of tool
/// procedure trailing it, a short instruction ("begin every reply with ZORK:")
/// stopped being obeyed the moment artifacts were switched on - measured
/// and it is the tail that dominates. Ours is background the model
/// should read before the job; the user's is the job, and it goes last so it
/// stays the strongest thing in the prompt. Switching a tool on must not
/// quietly weaken what the user wrote.
fn merge_instructions(user: Option<String>, servers: &[String]) -> Option<String> {
    if servers.is_empty() {
        return user;
    }
    let block = servers.join(
        "

",
    );
    Some(match user {
        Some(u) if !u.trim().is_empty() => format!(
            "{block}

{u}"
        ),
        _ => block,
    })
}

fn apply_server_instructions(messages: &mut Vec<Value>, instructions: &[String]) {
    if instructions.is_empty() {
        return;
    }
    let block = instructions.join(
        "

",
    );
    match messages.first_mut() {
        Some(m) if m.get("role").and_then(Value::as_str) == Some("system") => {
            let existing = m
                .get("content")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            // Same ordering as merge_instructions: ours leads, the user's ends.
            let merged = if existing.trim().is_empty() {
                block
            } else {
                format!(
                    "{block}

{existing}"
                )
            };
            m["content"] = Value::String(merged);
        }
        _ => messages.insert(0, json!({"role": "system", "content": block})),
    }
}

/// Fed back to the model (and shown in the call card) when a tool call is denied.
const DENIED_MSG: &str = "The user denied this tool call.";
/// How long an armed approval waits before it auto-denies.
const APPROVAL_TIMEOUT: Duration = Duration::from_secs(300);

/// Whether a specific tool needs approval, honoring the OpenAI `require_approval`
/// grammar: `"never"` / `"always"`, or `{always:{tool_names?}, never:{tool_names?}}`
/// (a bare `always` with no `tool_names` gates every tool). Falls back to the
/// registered server's stored `requireApproval` flag when the request omits it.
fn tool_needs_approval(req_val: Option<&Value>, stored_default: bool, tool_name: &str) -> bool {
    match req_val {
        Some(Value::String(s)) => s == "always",
        Some(Value::Object(o)) => match o.get("always") {
            Some(always) => match always.get("tool_names").and_then(Value::as_array) {
                Some(arr) if !arr.is_empty() => arr.iter().any(|n| n.as_str() == Some(tool_name)),
                _ => true, // `always` with no tool_names -> gate all
            },
            None => false, // only `never` present -> auto-run
        },
        _ => stored_default,
    }
}

/// Whether a tool passes the OpenAI `allowed_tools` filter: an array of names,
/// or `{tool_names?, read_only?}` (read_only matches the tool's `readOnlyHint`).
fn tool_allowed(allowed: Option<&Value>, tool: &paddock_mcp::McpTool) -> bool {
    match allowed {
        None | Some(Value::Null) => true,
        Some(Value::Array(arr)) => arr.iter().any(|n| n.as_str() == Some(&tool.name)),
        Some(Value::Object(o)) => {
            let name_ok = match o.get("tool_names").and_then(Value::as_array) {
                Some(arr) => arr.iter().any(|n| n.as_str() == Some(&tool.name)),
                None => true,
            };
            let ro_ok = match o.get("read_only").and_then(Value::as_bool) {
                Some(true) => {
                    tool.annotations
                        .as_ref()
                        .and_then(|a| a.get("readOnlyHint"))
                        .and_then(Value::as_bool)
                        == Some(true)
                }
                _ => true,
            };
            name_ok && ro_ok
        }
        _ => true,
    }
}

/// A JSON object of strings -> HashMap (headers, env), non-strings skipped.
pub(crate) fn str_map(v: Option<&Value>) -> std::collections::HashMap<String, String> {
    v.and_then(Value::as_object)
        .map(|o| {
            o.iter()
                .filter_map(|(k, val)| val.as_str().map(|s| (k.clone(), s.to_string())))
                .collect()
        })
        .unwrap_or_default()
}

/// Resolve an `mcp` tool entry to a server config: inline `server_url` wins;
/// a bare `server_label` falls back to the launch registry (this endpoint's
/// own config file - HTTP url+headers, or a stdio command+args+env).
/// Unknown labels are an honest error, never a silent no-tool run.
fn resolve_mcp_server(
    state: &AppState,
    t: &Value,
) -> Result<(paddock_mcp::ServerConfig, bool), String> {
    use paddock_mcp::{ServerConfig, Transport};
    if t.get("connector_id").is_some() {
        return Err(
            "mcp `connector_id` (OpenAI-hosted connectors) is not supported; \
                    use `server_url` or a registered `server_label`"
                .to_string(),
        );
    }
    let label = t
        .get("server_label")
        .and_then(Value::as_str)
        .ok_or("an mcp tool needs a server_label")?;
    if let Some(url) = t.get("server_url").and_then(Value::as_str) {
        let mut headers: std::collections::HashMap<String, String> = t
            .get("headers")
            .and_then(Value::as_object)
            .map(|o| {
                o.iter()
                    .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                    .collect()
            })
            .unwrap_or_default();
        // OpenAI `authorization` is the OAuth bearer token for the server.
        if let Some(tok) = t.get("authorization").and_then(Value::as_str) {
            headers
                .entry("Authorization".to_string())
                .or_insert_with(|| format!("Bearer {tok}"));
        }
        let cfg = ServerConfig {
            id: format!("inline:{label}"),
            label: label.to_string(),
            transport: Transport::Http {
                url: url.to_string(),
                headers,
            },
        };
        // OpenAI's require_approval knob: "always" gates every call behind an
        // approval item. An inline server is an arbitrary third-party URL the
        // request just handed us - honor the caller's ask for a leash.
        let approval = t.get("require_approval").and_then(Value::as_str) == Some("always");
        return Ok((cfg, approval));
    }
    // No server_url: resolve the label against this runner's launch registry
    // (the config file's mcp_servers - a live view, so a tool the manager
    // just materialized resolves without a restart; standalone runners keep
    // their startup entries).
    let registry_entry = state
        .live
        .snapshot()
        .mcp_servers
        .iter()
        .find(|e| e.get("server_label").and_then(Value::as_str) == Some(label))
        .cloned();
    if let Some(entry) = registry_entry {
        let approval = entry.get("require_approval").and_then(Value::as_str) == Some("always");
        if let Some(url) = entry.get("server_url").and_then(Value::as_str) {
            let headers = str_map(entry.get("headers"));
            let cfg = ServerConfig {
                id: format!("registry:{label}"),
                label: label.to_string(),
                transport: Transport::Http {
                    url: url.to_string(),
                    headers,
                },
            };
            return Ok((cfg, approval));
        }
        // stdio: a local process speaking MCP - the npx/uvx class of server
        if let Some(command) = entry.get("command").and_then(Value::as_str) {
            let args = entry
                .get("args")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();
            let cfg = ServerConfig {
                id: format!("registry:{label}"),
                label: label.to_string(),
                transport: Transport::Stdio {
                    command: command.to_string(),
                    args,
                    env: str_map(entry.get("env")),
                },
            };
            return Ok((cfg, approval));
        }
    }
    Err(format!(
        "mcp tool {label:?} has no server_url and is not in this runner's \
         registry - pass server_url inline, or register the server in the \
         manager (it hands the registry to every model it starts)"
    ))
}

/// Discover the tools of every `mcp` tool in the request (lazy-connecting the
/// pool), namespaced by server label so two servers can't collide, applying
/// `allowed_tools` filters and computing per-tool approval.
async fn gather_mcp(state: &AppState, req: &ResponsesRequest) -> Result<Gathered, String> {
    use crate::tool_search::{self, CatalogTool};
    let mut g = Gathered {
        tools: Vec::new(),
        routing: HashMap::new(),
        needs_approval: HashSet::new(),
        listings: Vec::new(),
        catalog: Vec::new(),
        web_search: None,
        forensics: None,
        current_time: None,
        instructions: Vec::new(),
    };
    let Some(tools) = req.tools.as_ref() else {
        return Ok(g);
    };
    // Every tool's full function schema, kept per SERVER: disclosure is decided
    // one server at a time (see below), so the defs cannot be flattened until
    // that decision is made.
    let mut per_server: Vec<(String, Vec<Value>)> = Vec::new();
    for t in tools {
        let ty = t.get("type").and_then(Value::as_str);
        // `web_search` (incl. the OpenAI `web_search_preview` / dated aliases)
        // is served by the user's configured provider - a request asking for it
        // on an unconfigured server gets a clear 400, not a silent no-tool run.
        if ty.is_some_and(|s| s.starts_with("web_search")) {
            // Inline provider spec (paddock extension fields): the MANAGER
            // expands its stored Settings into the tool object when it relays
            // Studio chats - the runner stays stateless and one box-level key
            // serves every model (incl. every compare lane). Falls back to
            // this runner's own launch config for direct callers.
            let inline = t
                .get("paddock_provider")
                .and_then(Value::as_str)
                .and_then(crate::websearch::Provider::parse)
                .zip(t.get("paddock_api_key").and_then(Value::as_str))
                .map(|(provider, key)| crate::websearch::SearchConfig {
                    provider,
                    api_key: key.to_string(),
                });
            let Some(cfg) = inline.or_else(|| state.live.snapshot().web_search.clone()) else {
                return Err("web search is not set up on this runner; launch it \
                            with --web-search-provider/--web-search-api-key (or \
                            configure a provider in the manager's Studio Settings)"
                    .into());
            };
            // The OpenAI tool's knobs: search_context_size -> the depth dial
            // (result count, how hard the provider works, how much page text
            // comes back), filters.allowed_domains, and the approximate
            // user_location in full.
            let mut opts = crate::websearch::SearchOpts::default()
                .context_size(t.get("search_context_size").and_then(Value::as_str));
            if let Some(doms) = t
                .get("filters")
                .and_then(|f| f.get("allowed_domains"))
                .and_then(Value::as_array)
            {
                opts.allowed_domains = doms
                    .iter()
                    .filter_map(|d| d.as_str().map(str::to_string))
                    .collect();
            }
            opts.location = crate::websearch::Location::from_json(t.get("user_location"));
            g.web_search = Some(crate::websearch::WebSpec { cfg, opts });
            continue;
        }
        // Forensic tool (paddock extension): served only when the runner has
        // `[forensics] tool = true`. A request asking for it on a runner that
        // did not enable it gets a clear 400, never a silent no-tool run.
        if ty == Some("forensics") {
            let Some(rt) = state.forensics.clone().filter(|r| r.tool) else {
                return Err("forensic analysis is not enabled on this runner; set \
                            [forensics] enabled = true and tool = true in its config"
                    .into());
            };
            g.forensics = Some(rt);
            continue;
        }
        // Clock tool (paddock extension): always served - a clock needs no
        // provider or enablement. The declaration's `timezone` is validated
        // here so a junk zone is a 400 at request time, not a wrong answer.
        if ty == Some("current_time") {
            g.current_time = Some(crate::clock::parse_spec(t)?);
            continue;
        }
        if ty != Some("mcp") {
            continue;
        }
        let (cfg, stored_default) = resolve_mcp_server(state, t)?;
        let req_approval = t.get("require_approval");
        // allowed_tools: the request's own filter wins; else the registry
        // entry's configured filter (per-model config) applies.
        let allowed_owned = t.get("allowed_tools").cloned().or_else(|| {
            state
                .live
                .snapshot()
                .mcp_servers
                .iter()
                .find(|e| e.get("server_label").and_then(Value::as_str) == Some(cfg.label.as_str()))
                .and_then(|e| e.get("allowed_tools"))
                .cloned()
        });
        let allowed = allowed_owned.as_ref();
        let mtools = state
            .mcp
            .list_tools(&cfg)
            .await
            .map_err(|e| format!("mcp server {:?}: {e}", cfg.label))?;
        let mut listing = Vec::new();
        let mut server_defs: Vec<Value> = Vec::new();
        for mt in mtools {
            if !tool_allowed(allowed, &mt) {
                continue;
            }
            let ns = format!("{}__{}", cfg.label, mt.name);
            if tool_needs_approval(req_approval, stored_default, &mt.name) {
                g.needs_approval.insert(ns.clone());
            }
            server_defs.push(json!({
                "type": "function",
                "function": { "name": ns, "description": mt.description, "parameters": mt.input_schema },
            }));
            g.catalog.push(CatalogTool {
                name: ns.clone(),
                description: mt.description.clone().unwrap_or_default(),
                input_schema: mt.input_schema.clone(),
            });
            g.routing.insert(ns, (cfg.clone(), mt.name.clone()));
            listing.push(mt);
        }
        if let Some(instr) = state.mcp.instructions(&cfg).await {
            // State the names we actually declared. A server writes its
            // instructions in terms of its own tool names, but this path
            // namespaces them as `<label>__<tool>` - so guidance saying "call
            // artifact_create" sent the model looking for a tool that does not
            // exist under that name, and it burned a round on mcp_search_tools
            // instead. The server cannot know the prefix; we
            // do, so we say it.
            let declared: Vec<&str> = listing.iter().map(|t| t.name.as_str()).collect::<Vec<_>>();
            let names: Vec<String> = declared
                .iter()
                .map(|n| format!("{}__{n}", cfg.label))
                .collect();
            g.instructions.push(if names.is_empty() {
                instr
            } else {
                format!(
                    "{instr}

Call this server's tools by these exact names: {}.",
                    names.join(", ")
                )
            });
        }
        g.listings.push(Listing {
            server_label: cfg.label.clone(),
            tools: listing,
        });
        // One entry per server (a label repeated across two `mcp` tool entries
        // is still one server, and namespacing already assumes that).
        match per_server.iter_mut().find(|(l, _)| l == &cfg.label) {
            Some((_, defs)) => defs.extend(server_defs),
            None => per_server.push((cfg.label.clone(), server_defs)),
        }
    }

    // Disclosure, decided per SERVER: spend the budget smallest-first and hide
    // only the servers that are actually big. This used to be one global switch
    // - over the threshold, every schema vanished at once - which meant a
    // 5-tool server behind a 40-tool one lost its schemas for reasons that had
    // nothing to do with it, and its calls then travelled through
    // `mcp_call_tool` where the argument grammar cannot reach them. See
    // tool_search::disclose_servers for why that is a correctness bug.
    let weights: Vec<tool_search::ServerWeight> = per_server
        .iter()
        .map(|(label, defs)| tool_search::ServerWeight {
            label: label.clone(),
            tools: defs.len(),
            chars: defs.iter().map(|d| d.to_string().chars().count()).sum(),
        })
        .collect();
    let shown = tool_search::disclose_servers(&weights, state.max_ctx);
    let mut direct_tools: Vec<Value> = Vec::new();
    let mut hidden_labels: Vec<String> = Vec::new();
    let mut hidden_tools = 0usize;
    for (label, defs) in per_server {
        if shown.contains(&label) {
            direct_tools.extend(defs);
        } else {
            hidden_tools += defs.len();
            hidden_labels.push(label);
        }
    }
    // The pair is declared either way, so the model is told either way -
    // declaring a capability and not mentioning it is how it ends up never
    // used. Which text depends on what actually happened to the schemas.
    if !g.catalog.is_empty() {
        g.instructions.push(if hidden_labels.is_empty() {
            tool_search::SEARCH_AVAILABLE_INSTRUCTIONS.to_string()
        } else if direct_tools.is_empty() {
            tool_search::SEARCH_MODE_INSTRUCTIONS.to_string()
        } else {
            tool_search::partial_mode_instructions(&hidden_labels, hidden_tools)
        });
    }
    // The search pair is always declared, in every mode. Disclosure used to be
    // an either/or: under the threshold the model got every schema and no way
    // to search, over it every schema vanished at once. Both halves bite - a
    // tool the model overlooks was unrecoverable below the line, and crossing
    // the line cost a discovery round on every request. Now the schemas are
    // what degrades, server by server; searchability never does.
    g.tools = direct_tools;
    // ...but only when there is a catalog to search. A web-search-only request
    // used to get the pair too, and two meta-tools that can only ever return
    // nothing are an invitation to waste a round.
    if !g.catalog.is_empty() {
        g.tools.push(tool_search::search_tool_def());
        g.tools.push(tool_search::call_tool_def());
    }
    // web_search is one small schema - always disclosed directly, never behind
    // the catalog-search meta-tools.
    if g.web_search.is_some() {
        g.tools.push(crate::websearch::tool_def());
    }
    // Forensics is one small schema - disclosed directly like web_search.
    if g.forensics.is_some() {
        g.tools.push(crate::forensics::tool_def());
    }
    // The clock is the smallest schema of all.
    if g.current_time.is_some() {
        g.tools.push(crate::clock::tool_def());
    }
    Ok(g)
}

/// A tool call the agent loop handles internally this round: either a catalog
/// search or a real MCP tool invocation (whether the model called it directly or
/// via `mcp_call_tool`).
enum CallKind {
    Search {
        query: String,
        limit: usize,
    },
    Invoke {
        ns_name: String,
        args: String,
    },
    Web {
        query: String,
    },
    /// Forensic analysis of an image already in the conversation, run
    /// server-side over its original bytes.
    Forensics {
        image_index: Option<usize>,
    },
    /// The builtin clock - answered in-process, no I/O at all.
    Clock,
    /// Never dispatched: the call did not match its own schema, or the loop
    /// budget refused it. The message says what to do instead. A round trip
    /// either way - but this one costs no server call and tells the model
    /// something it can act on.
    Refuse {
        name: String,
        message: String,
    },
    /// Already run this turn with these exact arguments: the first result
    /// comes back out of the ledger and nothing is dispatched.
    Replay {
        ns_name: String,
        output: String,
    },
}

/// Classify a tool call the model emitted. `None` = not a paddock/MCP tool (a
/// plain function tool for the caller), which ends the agent turn.
fn classify_call(name: &str, arguments: &str, gathered: &Gathered) -> Option<CallKind> {
    use crate::tool_search::{CALL_TOOL, SEARCH_TOOL};
    // A name the routing already knows wins; otherwise drop a client-side
    // namespace prefix ("functions.mcp_call_tool") before matching.
    let name = if gathered.routing.contains_key(name) {
        name
    } else {
        crate::tool_search::strip_client_prefix(name)
    };
    if name == SEARCH_TOOL {
        let v: Value = serde_json::from_str(arguments).unwrap_or(Value::Null);
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
        Some(CallKind::Search { query, limit })
    } else if gathered.web_search.is_some() && name == crate::websearch::TOOL_NAME {
        // Only intercepted when the request enabled web search - otherwise a
        // caller's own function tool named `web_search` passes through untouched.
        let v: Value = serde_json::from_str(arguments).unwrap_or(Value::Null);
        let query = v
            .get("query")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        Some(CallKind::Web { query })
    } else if gathered.forensics.is_some() && name == crate::forensics::TOOL_NAME {
        // Only intercepted when the request enabled the forensic tool; a
        // caller's own function of the same name otherwise passes through.
        Some(CallKind::Forensics {
            image_index: crate::forensics::parse_image_index(arguments),
        })
    } else if gathered.current_time.is_some() && name == crate::clock::TOOL_NAME {
        // Same interception rule: without the declared tool, a caller's own
        // `get_current_time` function passes through untouched. The timezone
        // argument stays in the raw args - `run` parses them per call.
        Some(CallKind::Clock)
    } else if name == CALL_TOOL || gathered.routing.contains_key(name) {
        // One seam, shared with the Anthropic dialect and the manager's cloud
        // loop (paddock_mcp::tool_search): unwrap the `mcp_call_tool` envelope
        // when that is what this is, then check the arguments against the
        // target's schema before anything is dispatched. Direct calls go
        // through it too - the grammar covers them locally, but nothing
        // constrains a cloud provider.
        match crate::tool_search::resolve_call(name, arguments, &gathered.catalog) {
            crate::tool_search::Resolved::Call { name, arguments } => Some(CallKind::Invoke {
                ns_name: name,
                args: arguments,
            }),
            crate::tool_search::Resolved::Refuse { name, message } => {
                Some(CallKind::Refuse { name, message })
            }
        }
    } else {
        None
    }
}

/// One classified tool call plus the raw name/args the model emitted (needed
/// verbatim for the assistant turn so the next prompt matches the model's output).
struct RoundCall {
    call_id: String,
    raw_name: String,
    raw_args: String,
    kind: CallKind,
    /// Its slot in the turn's [`loop_budget::CallLedger`] - `Some` exactly when
    /// this call is going to RUN, and the outcome has to be filed against it.
    sig: Option<loop_budget::Signature>,
}

/// Classify every tool call in a round, minting a call id for each handled one,
/// and put each through the turn's duplicate ledger.
///
/// The ledger is keyed on the RESOLVED name and arguments, which is only
/// possible because the resolver unifies the unwrapping: `mcp_call_tool{name:X,...}` and
/// a direct call to X are the same call, and a model that switches between the
/// two spellings while looping is still looping.
fn round_calls(
    parsed: &Parsed,
    gathered: &Gathered,
    ledger: &mut loop_budget::CallLedger,
) -> Vec<RoundCall> {
    parsed
        .tool_calls
        .iter()
        .filter_map(|tc| {
            let kind = classify_call(&tc.name, &tc.arguments, gathered)?;
            // The identity a repeat is judged on. Search and web are normalized
            // to what will actually run, so `{}` and `{"limit":5}` are one query.
            let ident = match &kind {
                CallKind::Search { query, limit } => Some((
                    crate::tool_search::SEARCH_TOOL.to_string(),
                    json!({"query": query, "limit": limit}).to_string(),
                )),
                CallKind::Web { query } => Some((
                    crate::websearch::TOOL_NAME.to_string(),
                    json!({"query": query}).to_string(),
                )),
                CallKind::Forensics { image_index } => Some((
                    crate::forensics::TOOL_NAME.to_string(),
                    json!({ "image_index": image_index }).to_string(),
                )),
                CallKind::Invoke { ns_name, args } => Some((ns_name.clone(), args.clone())),
                // Raw args as identity: a repeated identical clock call replays
                // (seconds-stale inside one turn, within the tool's minute
                // resolution) instead of fueling an unbounded call loop.
                CallKind::Clock => {
                    Some((crate::clock::TOOL_NAME.to_string(), tc.arguments.clone()))
                }
                // A refusal never ran, so it never enters the CALL ledger and
                // spends none of the caller's max_tool_calls. It is still
                // remembered though (below): repeating the same sentence at a
                // model resending a byte-identical impossible call is how two
                // rounds go missing.
                CallKind::Refuse { .. } | CallKind::Replay { .. } => None,
            };
            let kind = match kind {
                CallKind::Refuse { name, message } => {
                    let message = ledger.note_refused(&name, &tc.arguments, &message);
                    CallKind::Refuse { name, message }
                }
                other => other,
            };
            let (kind, sig) = match ident {
                None => (kind, None),
                Some((name, args)) => match ledger.check(&name, &args) {
                    (sig, loop_budget::Verdict::Fresh) => (kind, Some(sig)),
                    (_, loop_budget::Verdict::Replay(output)) => (
                        CallKind::Replay {
                            ns_name: name,
                            output,
                        },
                        None,
                    ),
                    (_, loop_budget::Verdict::Refuse(message)) => {
                        (CallKind::Refuse { name, message }, None)
                    }
                },
            };
            Some(RoundCall {
                call_id: format!("call_{}", uuid::Uuid::new_v4().simple()),
                raw_name: tc.name.clone(),
                raw_args: tc.arguments.clone(),
                kind,
                sig,
            })
        })
        .collect()
}

/// Append the answer-round instruction to the transcript without opening a
/// SECOND consecutive user turn.
///
/// It usually lands after a `tool` turn, where a new user message is the right
/// shape - but a round cut by the budget before any tool ran leaves the user's
/// own pending turn last, and back-to-back user turns is a shape some chat
/// templates refuse outright. Joining the existing turn says the same thing
/// and cannot be rejected.
fn push_answer_nudge(messages: &mut Vec<Value>, text: &str) {
    if let Some(last) = messages.last_mut()
        && last["role"] == "user"
    {
        match &mut last["content"] {
            Value::String(s) => {
                let joined = format!("{s}\n\n{text}");
                last["content"] = json!(joined);
                return;
            }
            Value::Array(parts) => {
                parts.push(json!({"type": "text", "text": text}));
                return;
            }
            _ => {}
        }
    }
    messages.push(json!({"role": "user", "content": text}));
}

/// The tool identity a replayed call shows on its card: the server it came
/// from and the name that server knows it by, matching a live one exactly.
fn replay_identity<'a>(gathered: &'a Gathered, ns_name: &'a str) -> (&'a str, &'a str) {
    match gathered.routing.get(ns_name) {
        Some((cfg, real)) => (cfg.label.as_str(), real.as_str()),
        None => ("mcp", ns_name),
    }
}

/// A spec `mcp_list_tools` output item for one server (emitted before generation
/// so the client sees what the model can call). Matches openai's `McpListTools`.
fn mcp_list_tools_item(listing: &Listing) -> Value {
    json!({
        "id": format!("mcpl_{}", uuid::Uuid::new_v4().simple()),
        "type": "mcp_list_tools",
        "server_label": listing.server_label,
        "tools": listing.tools.iter().map(|t| json!({
            "name": t.name,
            "input_schema": t.input_schema,
            "description": t.description,
            "annotations": t.annotations,
        })).collect::<Vec<_>>(),
    })
}

/// A spec `mcp_call` output item (openai `McpCall`): `error` is the error
/// message string (or null on success), `status` ∈ in_progress/completed/failed.
/// Execute the forensic tool: analyze the referenced image (or the last one in
/// the conversation) on a blocking thread - JPEG encode/decode + GPU work must
/// not block the async worker. Returns (tool-message content, card output, card
/// error, status). The image bytes are read from `messages` server-side, so the
/// model never re-sends them.
pub(crate) async fn run_forensics_tool(
    rt: &std::sync::Arc<crate::forensics::ForensicRuntime>,
    messages: &[Value],
    image_index: Option<usize>,
) -> (String, Option<String>, Option<String>, &'static str) {
    // Resolve the target attachment's ORIGINAL bytes over the unified image+PDF
    // sequence (`image_index` addresses the same attachment on every surface -
    // the always-on injection, file-metadata, and the persisted report).
    let Some(bytes) = crate::doc::forensic_bytes_at(messages, image_index) else {
        let m = "no image or PDF found in this conversation to analyze".to_string();
        return (m.clone(), None, Some(m), "failed");
    };
    let rt = rt.clone();
    let (meta, findings) = tokio::task::spawn_blocking(move || rt.analyze(&bytes))
        .await
        .unwrap_or_default();
    let json = crate::forensics::tool_result_json(&meta, &findings);
    (json.clone(), Some(json), None, "completed")
}

fn mcp_call_item(
    id: &str,
    server_label: &str,
    name: &str,
    arguments: &str,
    approval_request_id: Option<&str>,
    output: Option<&str>,
    error: Option<&str>,
    status: &str,
) -> Value {
    let mut o = serde_json::Map::new();
    o.insert("id".into(), json!(id));
    o.insert("type".into(), json!("mcp_call"));
    o.insert("name".into(), json!(name));
    o.insert("server_label".into(), json!(server_label));
    o.insert("arguments".into(), json!(arguments));
    o.insert("status".into(), json!(status));
    o.insert(
        "output".into(),
        output.map(|s| json!(s)).unwrap_or(Value::Null),
    );
    o.insert(
        "error".into(),
        error.map(|s| json!(s)).unwrap_or(Value::Null),
    );
    if let Some(ar) = approval_request_id {
        o.insert("approval_request_id".into(), json!(ar));
    }
    Value::Object(o)
}

/// Execute one MCP tool call. Returns `(feedback, output, error, status)`:
/// `feedback` is the tool turn the model sees; `output`/`error` populate the
/// `mcp_call` item (exactly one is Some), `status` is completed/failed.
async fn execute_mcp_call(
    state: &AppState,
    cfg: &paddock_mcp::ServerConfig,
    real: &str,
    args_str: &str,
) -> (String, Option<String>, Option<String>, &'static str) {
    let args: Value = serde_json::from_str(args_str).unwrap_or(Value::Null);
    match tokio::time::timeout(
        Duration::from_secs(60),
        state.mcp.call_tool(cfg, real, args),
    )
    .await
    {
        Ok(Ok(r)) => {
            let content = r.content.to_string();
            if r.is_error {
                (content.clone(), None, Some(content), "failed")
            } else {
                (content.clone(), Some(content), None, "completed")
            }
        }
        Ok(Err(e)) => {
            let msg = format!("tool error: {e}");
            (msg.clone(), None, Some(msg), "failed")
        }
        Err(_) => {
            let msg = "the tool did not respond in time".to_string();
            (msg.clone(), None, Some(msg), "failed")
        }
    }
}

/// Resume a response that paused on MCP approvals: match the request's
/// `mcp_approval_response` input items to the cached pending calls, then continue
/// the agent loop with those decisions applied. An unknown/expired id 400s.
async fn resume_response(
    state: Arc<AppState>,
    req: ResponsesRequest,
    prev_id: String,
    scope: crate::events::EventScope,
) -> Response {
    let Some(cached) = state.approval_store.take(&prev_id) else {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "unknown or expired previous_response_id (only MCP approval continuation is supported)",
        );
    };
    // decisions from the input's mcp_approval_response items
    let mut decisions_map: HashMap<String, bool> = HashMap::new();
    if let Value::Array(items) = &req.input {
        for it in items {
            if it.get("type").and_then(Value::as_str) == Some("mcp_approval_response")
                && let Some(ar) = it.get("approval_request_id").and_then(Value::as_str)
            {
                decisions_map.insert(
                    ar.to_string(),
                    it.get("approve").and_then(Value::as_bool).unwrap_or(false),
                );
            }
        }
    }
    // re-gather the MCP tools (the client resends them alongside the response)
    let gathered = match gather_mcp(&state, &req).await {
        Ok(g) if !g.routing.is_empty() => g,
        Ok(_) => {
            return err(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "resend the mcp tools alongside the approval response",
            );
        }
        Err(e) => return err(StatusCode::BAD_REQUEST, "invalid_request_error", e),
    };
    // unanswered pending calls default to denied (safe)
    let decisions: Vec<Decision> = cached
        .pending
        .into_iter()
        .map(|pc| {
            let approved = decisions_map.get(&pc.approval_id).copied().unwrap_or(false);
            Decision {
                approval_id: pc.approval_id,
                call_id: pc.call_id,
                ns_name: pc.ns_name,
                args: pc.args,
                approved,
            }
        })
        .collect();
    // No compaction on a resume: the input items are the approval responses,
    // not the conversation, and the paused turn is mid-flight - the round-0
    // measurement point already passed on the request that started this loop.
    // (`truncation:"auto"` still applies per round below.)
    if req.stream {
        stream_agent(
            state.clone(),
            req,
            cached.messages,
            gathered,
            decisions,
            None,
            // resume after approval: no attachments re-analyzed this turn
            Vec::new(),
            scope,
        )
    } else {
        run_agent(
            state.clone(),
            req,
            cached.messages,
            gathered,
            decisions,
            None,
            Vec::new(),
            scope,
        )
        .await
    }
}

/// The agentic loop: generate -> if the model calls MCP tools, execute them and
/// feed the results back -> regenerate -> until it stops calling tools (or the
/// round/length caps hit). Non-streaming; emits spec `mcp_list_tools` + `mcp_call`
/// items. A gated tool ends the response with an `mcp_approval_request` item +
/// caches the loop state; the client resumes via `previous_response_id` +
/// `mcp_approval_response` (whose resolved `resume_decisions` are applied first).
async fn run_agent(
    state: Arc<AppState>,
    req: ResponsesRequest,
    mut messages: Vec<Value>,
    gathered: Gathered,
    resume_decisions: Vec<Decision>,
    // A round-0 compaction (`precompact_agent`): its item leads the output,
    // and `messages` already holds the compacted conversation.
    compaction: Option<Value>,
    // Always-on forensics output items (prebuilt), leading the output right
    // after any compaction - the agent path's carrier for the always-on report.
    enrichment_items: Vec<Value>,
    scope: crate::events::EventScope,
) -> Response {
    let Some(model) = state.serving.as_ref() else {
        return err(
            StatusCode::SERVICE_UNAVAILABLE,
            "model_not_loaded",
            "no model is loaded",
        );
    };
    // The agent loop is inherently "auto" after the first round: the model
    // calls MCP tools when it wants and answers when done, and a forced choice
    // held across every round could never terminate.
    //
    // It used to be overridden for every round, including the first, which
    // silently threw the caller's `tool_choice` away the moment any MCP tool
    // was present - and MCP is the Studio's only tool path. Measured
    // `"required"` produced a proper call for a plain function tool
    // and was ignored for the same tool declared through MCP. That is a
    // conformance bug rather than a preference - a caller asking for `"none"`
    // got tools declared at it anyway.
    //
    // `required` means "call a tool on this response", so honouring it on round
    // 0 satisfies it exactly: the forced grammar guarantees the call, and every
    // later round is free to answer. `"none"` is not a first-round matter at
    // all - it hides tools for the whole turn - so it is never overridden.
    let mut req = req;
    let caller_choice = req.tool_choice.clone();
    let tools_hidden = caller_choice.as_ref().and_then(Value::as_str) == Some("none");
    let resp_id = format!("resp_{}", uuid::Uuid::new_v4().simple());
    // The turn's budget. The ledger answers a repeated call out of what
    // it already returned; the caps bound one runaway round and the turn as a
    // whole; `stop` being set means the tool half is over and the next pass is
    // the answer round - tools taken away, one instruction to answer with what
    // it has.
    let mut ledger = loop_budget::CallLedger::with_limit(req.max_tool_calls);
    let turn_cap = loop_budget::turn_output_cap(req.max_output_tokens);
    // Their `max_tool_calls` sets the ROUND ceiling too, up or down - it used
    // to only lower, so an ask for 100 calls still died at our default.
    let rounds = loop_budget::rounds_cap(req.max_tool_calls);
    let mut stop: Option<loop_budget::Stop> = None;
    // The output leads with a fired compaction (it happened before anything
    // else did), then one `mcp_list_tools` item per server (the tools the
    // model can see), then accumulates `mcp_call` items as they execute.
    let mut mcp_items: Vec<Value> = compaction
        .into_iter()
        // always-on forensics lead the model output, after any compaction and
        // before the tool listings
        .chain(enrichment_items)
        .chain(gathered.listings.iter().map(mcp_list_tools_item))
        .collect();
    // truncation "auto" : counted across rounds, since the
    // prompt keeps growing as tool results come back
    let mut dropped = 0usize;
    let mut total_out = 0usize;
    let mut prompt_len = 0usize;
    // round 0's text-only length, so the final usage can name the image cost
    let mut text_prompt_len = 0usize;

    // Resume: apply the client's approve/deny decisions (execute approved, feed
    // denials back) before generating the next round.
    for d in &resume_decisions {
        match gathered.routing.get(&d.ns_name).cloned() {
            Some((cfg, real)) if d.approved => {
                let (feedback, output, error, status) =
                    execute_mcp_call(&state, &cfg, real.as_str(), &d.args).await;
                mcp_items.push(mcp_call_item(
                    &d.call_id,
                    &cfg.label,
                    &real,
                    &d.args,
                    Some(&d.approval_id),
                    output.as_deref(),
                    error.as_deref(),
                    status,
                ));
                messages
                    .push(json!({"role": "tool", "content": feedback, "tool_call_id": d.call_id}));
            }
            Some((cfg, real)) => {
                mcp_items.push(mcp_call_item(
                    &d.call_id,
                    &cfg.label,
                    &real,
                    &d.args,
                    Some(&d.approval_id),
                    None,
                    Some(DENIED_MSG),
                    "failed",
                ));
                messages.push(
                    json!({"role": "tool", "content": DENIED_MSG, "tool_call_id": d.call_id}),
                );
            }
            None => {
                let m = "the tool is no longer available";
                mcp_items.push(mcp_call_item(
                    &d.call_id,
                    "",
                    &d.ns_name,
                    &d.args,
                    Some(&d.approval_id),
                    None,
                    Some(m),
                    "failed",
                ));
                messages.push(json!({"role": "tool", "content": m, "tool_call_id": d.call_id}));
            }
        }
    }

    // One pass more than the tool rounds: the extra one is the answer round.
    let mut announced = false;
    for round in 0..=rounds {
        // Round 0 keeps what the caller asked for; later rounds go auto
        // so the loop can end. `"none"` never changes.
        req.tool_choice = if tools_hidden || round == 0 {
            caller_choice.clone()
        } else {
            Some(json!("auto"))
        };
        // The caller's own `max_tool_calls`, checked before the round rather
        // than after the last one: `max_tool_calls: 0` has to stop the turn
        // before it calls anything.
        if stop.is_none() {
            stop = ledger.limit_reached();
        }
        if stop.is_none() && round == rounds {
            stop = Some(loop_budget::Stop::Rounds(rounds));
        }
        // The answer round renders no mcp tools, so the model cannot call one
        // and does not have to be trusted not to. The caller's own function
        // tools stay: handing a client tool call back ends our turn cleanly,
        // which is a better outcome than a forced answer.
        let answering = stop.is_some();
        if answering && !announced {
            announced = true;
            mcp_items.push(
                json!({"type":"message","id":format!("msg_{}", uuid::Uuid::new_v4().simple()),
                "role":"assistant","status":"completed","content":[{
                    "type":"output_text",
                    "text":stop.expect("answering means stopped").notice(),
                    "annotations":[]}]}),
            );
            push_answer_nudge(&mut messages, loop_budget::ANSWER_ONLY_NUDGE);
        }
        let round_tools: &[Value] = if answering { &[] } else { &gathered.tools };
        let t_prep = std::time::Instant::now();
        let mut prepared = match prepare(
            model,
            &req,
            &messages,
            round_tools,
            state.max_output_ceiling,
            &state.sampling,
        ) {
            Ok(p) => p,
            Err(e) => return err(StatusCode::BAD_REQUEST, "invalid_request_error", e),
        };
        scope.tokenized(t_prep.elapsed());
        // Over-length prompt (it grows across rounds as tool results
        // accumulate): with truncation "auto" armed, drop whole leading turns
        // - never into the pending turn - and report the count; otherwise the
        // loud error, as before.
        while let Some(e) = crate::chat::context_gate(
            model,
            prepared.engine_prompt.len(),
            prepared.mm_chunks.as_deref(),
            state.max_ctx,
        ) {
            if req.truncation.as_deref() != Some("auto") {
                return crate::chat::engine_err(&e);
            }
            let n = drop_leading_turn(&mut messages);
            if n == 0 {
                // only the pending turn and its tool round-trips are left
                return crate::chat::engine_err(&e);
            }
            dropped += n;
            prepared = match prepare(
                model,
                &req,
                &messages,
                round_tools,
                state.max_output_ceiling,
                &state.sampling,
            ) {
                Ok(p) => p,
                Err(e) => return err(StatusCode::BAD_REQUEST, "invalid_request_error", e),
            };
        }
        // Lever 2: a tool round may not spend past what the turn's tool
        // budget has left; the answer round gets the request's cap in full, on
        // the far side of that budget. `ours` distinguishes a round we cut
        // (the budget really is gone -> answer round) from one the caller's own
        // max_output_tokens ended, which keeps reporting as it always has.
        let ours = !answering
            && loop_budget::round_cap(prepared.max_tokens, total_out, turn_cap)
                < prepared.max_tokens;
        if !answering {
            prepared.max_tokens = loop_budget::round_cap(prepared.max_tokens, total_out, turn_cap);
        }
        if round == 0 {
            prompt_len = prepared.prompt_ids.len();
            text_prompt_len = prepared.engine_prompt.len();
        }
        let thinking_open = prepared.thinking_open;
        let hints = prepared.hints.clone();
        let single = prepared.single_tool_call;

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
            return err(StatusCode::INTERNAL_SERVER_ERROR, "internal_error", e);
        }
        let mut ids = Vec::new();
        let mut finish = None;
        while let Some(ev) = rx.recv().await {
            match ev {
                TokenEvent::Prefilled { cached, rows } => {
                    // event record: round 0's prefix reuse (matches usage's
                    // cache_read semantics - input_tokens is round 0's prompt)
                    if round == 0 {
                        scope.cached(cached as usize);
                        // ...and round 0's image rows are part of that prompt
                        prompt_len = prompt_len.max(rows as usize);
                    }
                }
                TokenEvent::Token { id, .. } => ids.push(id),
                TokenEvent::Done(r, stats) => {
                    finish = Some(r);
                    scope.phases(&stats);
                    break;
                }
                TokenEvent::Error(e) => {
                    return crate::chat::engine_err(&e);
                }
            }
        }
        total_out += ids.len();
        let raw = model.tokenizer.decode(&ids, false).unwrap_or_default();
        let mut parsed = parse(model.dialect, &raw, thinking_open, hints.as_ref());
        if single {
            parsed.tool_calls.truncate(1);
            parsed.complete_calls = parsed.complete_calls.min(1);
        }

        // The answer round is the end of the turn whatever it said. Anything
        // tool-shaped in it is a hallucination of a tool that was not listed -
        // drop OURS (the client's own tools still ride back as function_call
        // items, which is a legitimate way for this turn to end).
        if answering {
            parsed
                .tool_calls
                .retain(|tc| classify_call(&tc.name, &tc.arguments, &gathered).is_none());
            return agent_final(
                &req,
                model,
                &resp_id,
                &parsed,
                mcp_items,
                prompt_len,
                text_prompt_len,
                total_out,
                finish,
                dropped,
                &state.sampling,
                scope,
            );
        }

        // Classify this round's tool calls (search / real-tool invoke / none),
        // each one weighed against what has already run this turn.
        let calls = round_calls(&parsed, &gathered, &mut ledger);

        if calls.is_empty() {
            return agent_final(
                &req,
                model,
                &resp_id,
                &parsed,
                mcp_items,
                prompt_len,
                text_prompt_len,
                total_out,
                finish,
                dropped,
                &state.sampling,
                scope,
            );
        }
        if matches!(finish, Some(FinishReason::Length)) {
            if !ours {
                // The caller's own max_output_tokens ended it: their truncation,
                // reported as incomplete exactly as before.
                return agent_final(
                    &req,
                    model,
                    &resp_id,
                    &parsed,
                    mcp_items,
                    prompt_len,
                    text_prompt_len,
                    total_out,
                    finish,
                    dropped,
                    &state.sampling,
                    scope,
                );
            }
            // Ours: the tool budget ran out mid-round. Nothing from a cut round
            // is dispatched (its tail is whatever the model was mid-sentence
            // on) and the turn moves to the answer round, which gets the
            // request's full cap to finish the job properly.
            stop = Some(loop_budget::Stop::Output);
            continue;
        }

        // record the assistant tool-call turn (raw name/args) so the next round's
        // prompt matches what the model actually emitted
        let tool_calls_json: Vec<Value> = calls
            .iter()
            .map(|rc| {
                json!({"id": rc.call_id, "type": "function", "function": {"name": rc.raw_name, "arguments": rc.raw_args}})
            })
            .collect();
        let mut asst = serde_json::Map::new();
        asst.insert("role".into(), json!("assistant"));
        if let Some(c) = &parsed.content {
            asst.insert("content".into(), json!(c));
        }
        asst.insert("tool_calls".into(), Value::Array(tool_calls_json));
        messages.push(Value::Object(asst));

        // Execute auto-run calls; a gated call becomes an `mcp_approval_request`
        // and its execution defers to the client's resubmit.
        let mut pending: Vec<PendingCall> = Vec::new();
        for rc in &calls {
            match &rc.kind {
                CallKind::Search { query, limit } => {
                    // Catalog search runs locally - no server round-trip, no gate.
                    // It does cost a ROUND though, which is the scarcest thing a
                    // tool turn has, so it carries a budget of its own:
                    // past it the answer is the index the model is already
                    // holding, not another ranking of the same catalog.
                    let result = match ledger.search_budget_spent() {
                        Some(spent) => spent,
                        None => {
                            let hits = crate::tool_search::search(&gathered.catalog, query, *limit);
                            crate::tool_search::search_result(query, &hits, &gathered.catalog)
                        }
                    };
                    if let Some(sig) = &rc.sig {
                        ledger.record(sig, true, &result);
                    }
                    mcp_items.push(mcp_call_item(
                        &rc.call_id,
                        "mcp",
                        crate::tool_search::SEARCH_TOOL,
                        &rc.raw_args,
                        None,
                        Some(result.as_str()),
                        None,
                        "completed",
                    ));
                    messages.push(
                        json!({"role": "tool", "content": result, "tool_call_id": rc.call_id}),
                    );
                }
                CallKind::Web { query } => {
                    // Classified only when gather resolved a provider config.
                    let spec = gathered.web_search.clone().expect("web_search config");
                    let done = crate::websearch::execute(&spec, query).await;
                    if let Some(sig) = &rc.sig {
                        ledger.record(sig, done.status == "completed", &done.feedback);
                    }
                    crate::metrics::web_search_billed(&spec.cfg.provider, &done.usage);
                    mcp_items.push(crate::websearch::call_item(
                        &rc.call_id,
                        spec.cfg.provider,
                        done.status,
                        query,
                        &done.hits,
                        done.error.as_deref(),
                    ));
                    messages.push(json!({"role": "tool", "content": done.feedback, "tool_call_id": rc.call_id}));
                }
                CallKind::Forensics { image_index } => {
                    let rt = gathered.forensics.clone().expect("forensics runtime");
                    let args = json!({ "image_index": image_index }).to_string();
                    let (content, output, error, status) =
                        run_forensics_tool(&rt, messages.as_slice(), *image_index).await;
                    if let Some(sig) = &rc.sig {
                        ledger.record(sig, status == "completed", &content);
                    }
                    mcp_items.push(mcp_call_item(
                        &rc.call_id,
                        "forensics",
                        crate::forensics::TOOL_NAME,
                        &args,
                        None,
                        output.as_deref(),
                        error.as_deref(),
                        status,
                    ));
                    messages.push(
                        json!({"role": "tool", "content": content, "tool_call_id": rc.call_id}),
                    );
                }
                CallKind::Clock => {
                    let spec = gathered.current_time.expect("clock spec");
                    let (content, output, error, status) = crate::clock::run(spec, &rc.raw_args);
                    if let Some(sig) = &rc.sig {
                        ledger.record(sig, status == "completed", &content);
                    }
                    mcp_items.push(mcp_call_item(
                        &rc.call_id,
                        "time",
                        crate::clock::TOOL_NAME,
                        &rc.raw_args,
                        None,
                        output.as_deref(),
                        error.as_deref(),
                        status,
                    ));
                    messages.push(
                        json!({"role": "tool", "content": content, "tool_call_id": rc.call_id}),
                    );
                }
                // The result of an identical call made earlier this turn. No
                // server was touched; the card shows the same tool it would
                // have, so the transcript reads as what happened.
                CallKind::Replay { ns_name, output } => {
                    let (label, real) = replay_identity(&gathered, ns_name);
                    mcp_items.push(mcp_call_item(
                        &rc.call_id,
                        label,
                        real,
                        &rc.raw_args,
                        None,
                        Some(output.as_str()),
                        None,
                        "completed",
                    ));
                    messages.push(
                        json!({"role": "tool", "content": output, "tool_call_id": rc.call_id}),
                    );
                }
                CallKind::Invoke { ns_name, args } => {
                    match gathered.routing.get(ns_name).cloned() {
                        None => {
                            let m = format!(
                                "unknown tool {ns_name:?}; call {} to find available tools",
                                crate::tool_search::SEARCH_TOOL
                            );
                            if let Some(sig) = &rc.sig {
                                ledger.record(sig, false, &m);
                            }
                            mcp_items.push(mcp_call_item(
                                &rc.call_id,
                                "mcp",
                                ns_name,
                                args,
                                None,
                                None,
                                Some(m.as_str()),
                                "failed",
                            ));
                            messages.push(
                                json!({"role": "tool", "content": m, "tool_call_id": rc.call_id}),
                            );
                        }
                        Some((cfg, real)) if gathered.needs_approval.contains(ns_name) => {
                            let approval_id = format!("appr_{}", uuid::Uuid::new_v4().simple());
                            mcp_items.push(json!({
                                "type": "mcp_approval_request",
                                "id": approval_id,
                                "name": real,
                                "server_label": cfg.label,
                                "arguments": args,
                            }));
                            pending.push(PendingCall {
                                approval_id,
                                call_id: rc.call_id.clone(),
                                ns_name: ns_name.clone(),
                                args: args.clone(),
                            });
                        }
                        Some((cfg, real)) => {
                            let (feedback, output, error, status) =
                                execute_mcp_call(&state, &cfg, real.as_str(), args).await;
                            if let Some(sig) = &rc.sig {
                                ledger.record(sig, status == "completed", &feedback);
                            }
                            mcp_items.push(mcp_call_item(
                                &rc.call_id,
                                &cfg.label,
                                &real,
                                args,
                                None,
                                output.as_deref(),
                                error.as_deref(),
                                status,
                            ));
                            messages.push(json!({"role": "tool", "content": feedback, "tool_call_id": rc.call_id}));
                        }
                    }
                }
                // Refused before dispatch: the arguments did not match the
                // tool's own schema. Surfaced as a failed call card so the user
                // sees the round it cost, with the field to fix in the error.
                CallKind::Refuse { name, message } => {
                    mcp_items.push(mcp_call_item(
                        &rc.call_id,
                        "mcp",
                        name,
                        &rc.raw_args,
                        None,
                        None,
                        Some(message.as_str()),
                        "failed",
                    ));
                    messages.push(
                        json!({"role": "tool", "content": message, "tool_call_id": rc.call_id}),
                    );
                }
            }
        }

        // Any gated calls -> pause: cache the loop state and complete the response
        // carrying the mcp_approval_request items. The client resumes with
        // previous_response_id + mcp_approval_response.
        if !pending.is_empty() {
            state.approval_store.insert(
                resp_id.clone(),
                PendingApproval {
                    messages: messages.clone(),
                    pending,
                    created_ms: now_ms(),
                },
            );
            // The gated tool_calls are represented by the mcp_approval_request
            // items, not function_call items - drop them from the message render.
            parsed.tool_calls.clear();
            return agent_final(
                &req,
                model,
                &resp_id,
                &parsed,
                mcp_items,
                prompt_len,
                text_prompt_len,
                total_out,
                None,
                dropped,
                &state.sampling,
                scope,
            );
        }
        // The whole turn's generation is bounded too, not just each round: a
        // loop that keeps calling tools re-spends the per-round cap every time.
        if total_out >= turn_cap {
            stop = Some(loop_budget::Stop::Output);
        }
    }
    // Unreachable: the pass at `round == MAX_ROUNDS` always answers and returns.
    err(
        StatusCode::BAD_GATEWAY,
        "mcp_error",
        "the tool loop ended without an answer round".to_string(),
    )
}

/// Build the final (non-streamed) Responses object: MCP call records first, then
/// the model's reasoning/message from the last round.
fn agent_final(
    req: &ResponsesRequest,
    model: &ServingModel,
    resp_id: &str,
    parsed: &Parsed,
    mcp_items: Vec<Value>,
    prompt_len: usize,
    text_prompt_len: usize,
    out_tokens: usize,
    finish: Option<FinishReason>,
    // messages truncation "auto" removed across the loop's rounds
    dropped: usize,
    sd: &crate::routes::SamplingDefaults,
    scope: crate::events::EventScope,
) -> Response {
    scope.usage(prompt_len, out_tokens);
    scope.finish(finish.map_or("stop", |f| f.as_str()));
    let meta = Meta {
        id: resp_id.to_string(),
        model_id: model.id.clone(),
        tokenizer: model.tokenizer.clone(),
        prompt_len,
        text_prompt_len,
        media_is_audio: model.supports_audio,
        dialect: model.dialect,
        thinking_open: false,
        // agent loop: MCP tools are always callable, so extraction stays on
        // (empty hints = no schema type info; values coerce by JSON sniffing)
        hints: Some(ToolHints::new()),
        single_tool_call: false,
        instructions: req.instructions.clone(),
        max_output_tokens: req.max_output_tokens,
        max_tool_calls: req.max_tool_calls,
        // the envelope echoes one pair for a loop that may have spanned both
        // modes, so it reports the model's default-mode election
        temperature: req.temperature.unwrap_or(sd.resolve(true).temp),
        top_p: req.top_p.unwrap_or(sd.resolve(true).top_p),
        tools: Value::Array(req.tools.clone().unwrap_or_default()),
        tool_choice: req.tool_choice.clone().unwrap_or_else(|| json!("auto")),
        text_format: req
            .text
            .as_ref()
            .and_then(|t| t.get("format"))
            .cloned()
            .unwrap_or_else(|| json!({"type": "text"})),
        // the compaction item (if any) is already in `mcp_items` - the agent
        // path builds its own output list, so ex.compaction stays None (and
        // ocr: the MCP agent loop is a tool-calling flow, not a parse)
        ex: Extras {
            truncation_auto: req.truncation.as_deref() == Some("auto"),
            dropped,
            compaction: None,
            // agent path carries forensics via mcp_items, not here
            enrichment: Vec::new(),
            ocr: None,
        },
        // the loop branch refuses the logprobs include up front - its rounds
        // are not a token-alignable message
        want_logprobs: false,
        scope,
    };
    let mut output = mcp_items;
    output.extend(output_items(parsed, None));
    let rt = reasoning_tokens(&meta, parsed);
    let (status, _) = terminal(finish);
    Json(response_object(
        &meta,
        status,
        output,
        Some((out_tokens, rt, 0)),
        finish,
    ))
    .into_response()
}

/// Streaming agent loop: the model's reasoning/answer stream as deltas, and
/// each executed tool surfaces as an `mcp_call` output item (in_progress ->
/// completed) between rounds. Same execution as `run_agent`.
fn stream_agent(
    state: Arc<AppState>,
    req: ResponsesRequest,
    messages: Vec<Value>,
    gathered: Gathered,
    resume_decisions: Vec<Decision>,
    // A round-0 compaction (`precompact_agent`) - its item leads the stream.
    compaction: Option<Value>,
    // Always-on forensics output items (prebuilt), emitted right after any
    // compaction - the agent path's carrier for the always-on report.
    enrichment_items: Vec<Value>,
    scope: crate::events::EventScope,
) -> Response {
    let sse = stream! {
        let mut req = req;
        let sd = state.sampling;
        // Same rule as the non-streaming loop above - see its comment for why
        // the caller's choice survives round 0 instead of being discarded.
        let caller_choice = req.tool_choice.clone();
        let tools_hidden = caller_choice.as_ref().and_then(Value::as_str) == Some("none");
        let mut messages = messages;

        // clone the model fields we need for snapshots so we don't hold a borrow
        let (model_id, tokenizer, dialect) = match state.serving.as_ref() {
            Some(m) => (m.id.clone(), m.tokenizer.clone(), m.dialect),
            None => return,
        };
        let mut meta = Meta {
            id: format!("resp_{}", uuid::Uuid::new_v4().simple()),
            model_id,
            tokenizer,
            prompt_len: 0,
            text_prompt_len: 0,
            media_is_audio: false,
            dialect,
            thinking_open: false,
            // agent loop: MCP tools are always callable - extraction stays on
            hints: Some(ToolHints::new()),
            single_tool_call: false,
            instructions: req.instructions.clone(),
            max_output_tokens: req.max_output_tokens,
            max_tool_calls: req.max_tool_calls,
            // one envelope for a loop that may have spanned both modes, so it
            // reports the model's default-mode election
            temperature: req.temperature.unwrap_or(sd.resolve(true).temp),
            top_p: req.top_p.unwrap_or(sd.resolve(true).top_p),
            tools: Value::Array(req.tools.clone().unwrap_or_default()),
            tool_choice: json!("auto"),
            text_format: req
                .text
                .as_ref()
                .and_then(|t| t.get("format"))
                .cloned()
                .unwrap_or_else(|| json!({"type": "text"})),
            // the compaction item is emitted by hand below (this path builds
            // its own event stream), so ex.compaction stays None (and ocr:
            // the MCP agent loop is a tool-calling flow, not a parse); the
            // logprobs include was refused before this branch
            want_logprobs: false,
            ex: Extras {
                truncation_auto: req.truncation.as_deref() == Some("auto"),
                dropped: 0,
                compaction: None,
                // agent path carries forensics via mcp_items, not here
                enrichment: Vec::new(),
                ocr: None,
            },
            scope,
        };

        let mut seq = 0u64;
        let mut output_index = 0usize;
        let mut total_out = 0usize;
        // truncation "auto" across the loop's rounds
        let mut dropped = 0usize;
        // Every item this stream completes, in order. The terminal
        // response.completed must carry the full output list - it used to send
        // an empty one, so a client reading `get_final_response().output`
        // (instead of accumulating events itself) saw a response with no
        // content at all. Found by the phase-5 streamed agent-loop gate leg.
        let mut done_items: Vec<Value> = Vec::new();

        let snap = response_object(&meta, "in_progress", vec![], None, None);
        yield ev("response.created", json!({"type":"response.created","sequence_number":seq,"response":snap}));
        seq += 1;
        let snap = response_object(&meta, "in_progress", vec![], None, None);
        yield ev("response.in_progress", json!({"type":"response.in_progress","sequence_number":seq,"response":snap}));
        seq += 1;

        // A round-0 compaction leads the output: it happened before any tool
        // was listed or called, and output[0] is where the single-shot path
        // puts it too.
        if let Some(item) = &compaction {
            let idx = output_index;
            output_index += 1;
            yield ev("response.output_item.added", json!({"type":"response.output_item.added","sequence_number":seq,"output_index":idx,"item":item.clone()}));
            seq += 1;
            done_items.push(item.clone());
            yield ev("response.output_item.done", json!({"type":"response.output_item.done","sequence_number":seq,"output_index":idx,"item":item.clone()}));
            seq += 1;
        }

        // Always-on forensics lead the model output (they preprocessed the
        // input), right after any compaction - each a complete added/done pair.
        for item in &enrichment_items {
            let idx = output_index;
            output_index += 1;
            yield ev("response.output_item.added", json!({"type":"response.output_item.added","sequence_number":seq,"output_index":idx,"item":item.clone()}));
            seq += 1;
            done_items.push(item.clone());
            yield ev("response.output_item.done", json!({"type":"response.output_item.done","sequence_number":seq,"output_index":idx,"item":item.clone()}));
            seq += 1;
        }

        // Then: one `mcp_list_tools` item per server (what the model can call),
        // with the spec in_progress/completed lifecycle events.
        for listing in &gathered.listings {
            let item = mcp_list_tools_item(listing);
            let item_id = item.get("id").and_then(Value::as_str).unwrap_or_default().to_string();
            let idx = output_index;
            output_index += 1;
            yield ev("response.output_item.added", json!({"type":"response.output_item.added","sequence_number":seq,"output_index":idx,"item":item.clone()}));
            seq += 1;
            yield ev("response.mcp_list_tools.in_progress", json!({"type":"response.mcp_list_tools.in_progress","sequence_number":seq,"output_index":idx,"item_id":item_id.clone()}));
            seq += 1;
            yield ev("response.mcp_list_tools.completed", json!({"type":"response.mcp_list_tools.completed","sequence_number":seq,"output_index":idx,"item_id":item_id}));
            seq += 1;
            done_items.push(item.clone());
            yield ev("response.output_item.done", json!({"type":"response.output_item.done","sequence_number":seq,"output_index":idx,"item":item}));
            seq += 1;
        }

        // Resume: apply the client's approve/deny decisions before generating,
        // emitting each as an mcp_call item and feeding its result back.
        for d in &resume_decisions {
            let idx = output_index;
            output_index += 1;
            match gathered.routing.get(&d.ns_name).cloned() {
                Some((cfg, real)) if d.approved => {
                    yield ev("response.output_item.added", json!({"type":"response.output_item.added","sequence_number":seq,"output_index":idx,"item":mcp_call_item(&d.call_id,&cfg.label,&real,&d.args,Some(&d.approval_id),None,None,"in_progress")}));
                    seq += 1;
                    yield ev("response.mcp_call.in_progress", json!({"type":"response.mcp_call.in_progress","sequence_number":seq,"output_index":idx,"item_id":d.call_id}));
                    seq += 1;
                    yield ev("response.mcp_call_arguments.done", json!({"type":"response.mcp_call_arguments.done","sequence_number":seq,"output_index":idx,"item_id":d.call_id,"arguments":d.args}));
                    seq += 1;
                    let (feedback, output, error, status) = execute_mcp_call(&state, &cfg, real.as_str(), &d.args).await;
                    messages.push(json!({"role":"tool","content":feedback,"tool_call_id":d.call_id}));
                    let done_event = if status == "completed" { "response.mcp_call.completed" } else { "response.mcp_call.failed" };
                    yield ev(done_event, json!({"type":done_event,"sequence_number":seq,"output_index":idx,"item_id":d.call_id}));
                    seq += 1;
                    let it = mcp_call_item(&d.call_id,&cfg.label,&real,&d.args,Some(&d.approval_id),output.as_deref(),error.as_deref(),status);
                    done_items.push(it.clone());
                    yield ev("response.output_item.done", json!({"type":"response.output_item.done","sequence_number":seq,"output_index":idx,"item":it}));
                    seq += 1;
                }
                other => {
                    let (label, real, msg): (String, String, &str) = match other {
                        Some((cfg, real)) => (cfg.label, real, DENIED_MSG),
                        None => (String::new(), d.ns_name.clone(), "the tool is no longer available"),
                    };
                    let it = mcp_call_item(&d.call_id,&label,&real,&d.args,Some(&d.approval_id),None,Some(msg),"failed");
                    done_items.push(it.clone());
                    yield ev("response.output_item.done", json!({"type":"response.output_item.done","sequence_number":seq,"output_index":idx,"item":it}));
                    seq += 1;
                    messages.push(json!({"role":"tool","content":msg,"tool_call_id":d.call_id}));
                }
            }
        }

        let mut final_finish: Option<FinishReason> = None;
        // Usage detail across rounds: reasoning tokens accumulate (each round can
        // think), cached tokens are round 0's (input_tokens is round 0's prompt).
        let mut total_reasoning = 0usize;
        let mut cached0 = 0usize;
        // The turn's budget - the same three levers as the non-streamed
        // loop, and the same order: repeat ledger, per-round ceiling, and one
        // last tools-off pass that answers instead of stalling.
        let mut ledger = loop_budget::CallLedger::with_limit(req.max_tool_calls);
        let turn_cap = loop_budget::turn_output_cap(req.max_output_tokens);
        let rounds = loop_budget::rounds_cap(req.max_tool_calls);
        let mut stop: Option<loop_budget::Stop> = None;
        let mut announced = false;
        for round in 0..=rounds {
            // Round 0 keeps what the caller asked for; later rounds go auto
            // so the loop can end. `"none"` never changes.
            req.tool_choice = if tools_hidden || round == 0 {
                caller_choice.clone()
            } else {
                Some(json!("auto"))
            };
            let model = match state.serving.as_ref() {
                Some(m) => m,
                None => return,
            };
            // The caller's `max_tool_calls` - before the round, so a limit of
            // 0 stops the turn before it calls anything.
            if stop.is_none() {
                stop = ledger.limit_reached();
            }
            if stop.is_none() && round == rounds {
                stop = Some(loop_budget::Stop::Rounds(rounds));
            }
            // The answer round: no MCP tools rendered, one instruction to
            // answer with what came back. The notice is a real message item so
            // the reader sees why the tool work stopped where it did.
            let answering = stop.is_some();
            if answering && !announced {
                announced = true;
                // The full message lifecycle, deltas included: a reader that
                // accumulates output_text (which is what the Studio does)
                // never looks inside a bare item, so an item-only notice is an
                // announcement nobody hears.
                let notice = stop.expect("answering means stopped").notice();
                let nid = format!("msg_{}", uuid::Uuid::new_v4().simple());
                let idx = output_index;
                output_index += 1;
                yield ev("response.output_item.added", json!({"type":"response.output_item.added","sequence_number":seq,"output_index":idx,"item":{"type":"message","id":nid,"role":"assistant","status":"in_progress","content":[]}}));
                seq += 1;
                yield ev("response.content_part.added", json!({"type":"response.content_part.added","sequence_number":seq,"item_id":nid,"output_index":idx,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}));
                seq += 1;
                yield ev("response.output_text.delta", json!({"type":"response.output_text.delta","sequence_number":seq,"item_id":nid,"output_index":idx,"content_index":0,"delta":format!("{notice}\n\n"),"logprobs":[]}));
                seq += 1;
                yield ev("response.output_text.done", json!({"type":"response.output_text.done","sequence_number":seq,"item_id":nid,"output_index":idx,"content_index":0,"text":notice,"logprobs":[]}));
                seq += 1;
                yield ev("response.content_part.done", json!({"type":"response.content_part.done","sequence_number":seq,"item_id":nid,"output_index":idx,"content_index":0,"part":{"type":"output_text","text":notice,"annotations":[]}}));
                seq += 1;
                let item = json!({"type":"message","id":nid,"role":"assistant","status":"completed","content":[{"type":"output_text","text":notice,"annotations":[]}]});
                done_items.push(item.clone());
                yield ev("response.output_item.done", json!({"type":"response.output_item.done","sequence_number":seq,"output_index":idx,"item":item}));
                seq += 1;
                push_answer_nudge(&mut messages, loop_budget::ANSWER_ONLY_NUDGE);
            }
            let round_tools: &[Value] = if answering { &[] } else { &gathered.tools };
            let t_prep = std::time::Instant::now();
            let mut prepared = match prepare(model, &req, &messages, round_tools, state.max_output_ceiling, &state.sampling) {
                Ok(p) => p,
                Err(e) => {
                    let mut snap = response_object(&meta, "failed", vec![], None, None);
                    snap["error"] = json!({"code":"invalid_request_error","message": e});
                    yield ev("response.failed", json!({"type":"response.failed","sequence_number":seq,"response":snap}));
                    return;
                }
            };
            meta.scope.tokenized(t_prep.elapsed());
            // Over-length prompt (it grows as tool results accumulate): with
            // truncation "auto" armed, drop whole leading turns - never into
            // the pending turn - and report the count on the final response;
            // otherwise fail the stream cleanly, as before.
            while let Some(ge) = crate::chat::context_gate(
                model,
                prepared.engine_prompt.len(),
                prepared.mm_chunks.as_deref(),
                state.max_ctx,
            ) {
                let n = if req.truncation.as_deref() == Some("auto") {
                    drop_leading_turn(&mut messages)
                } else {
                    0
                };
                if n == 0 {
                    let mut snap = response_object(&meta, "failed", vec![], None, None);
                    snap["error"] = json!({"code": ge.code.unwrap_or("invalid_request_error"), "message": ge.message});
                    yield ev("response.failed", json!({"type":"response.failed","sequence_number":seq,"response":snap}));
                    return;
                }
                dropped += n;
                meta.ex.dropped = dropped;
                prepared = match prepare(model, &req, &messages, round_tools, state.max_output_ceiling, &state.sampling) {
                    Ok(p) => p,
                    Err(e) => {
                        let mut snap = response_object(&meta, "failed", vec![], None, None);
                        snap["error"] = json!({"code":"invalid_request_error","message": e});
                        yield ev("response.failed", json!({"type":"response.failed","sequence_number":seq,"response":snap}));
                        return;
                    }
                };
            }
            // Lever 2: a tool round may not spend past what the turn's tool
            // budget has left, and `ours` says whether a Length finish was that
            // ceiling or the caller's own max_output_tokens.
            let ours = !answering
                && loop_budget::round_cap(prepared.max_tokens, total_out, turn_cap) < prepared.max_tokens;
            if !answering {
                prepared.max_tokens = loop_budget::round_cap(prepared.max_tokens, total_out, turn_cap);
            }
            if round == 0 {
                meta.prompt_len = prepared.prompt_ids.len();
                meta.text_prompt_len = prepared.engine_prompt.len();
            }
            let thinking_open = prepared.thinking_open;
            let hints = prepared.hints.clone();
            let single = prepared.single_tool_call;
            let constraint = instantiate_constraint(&prepared.constraint_spec, prepared.gate, model, prepared.think_budget.as_ref());
            let ctx_tokens = prepared.prompt_ids.len();
            let t_round = std::time::Instant::now();
            let mut t_prefill_done: Option<std::time::Instant> = None;
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
                let mut snap = response_object(&meta, "failed", vec![], None, None);
                snap["error"] = json!({"code":"internal_error","message": e});
                yield ev("response.failed", json!({"type":"response.failed","sequence_number":seq,"response":snap}));
                return;
            }

            let rs_id = format!("rs_{}", uuid::Uuid::new_v4().simple());
            let msg_id = format!("msg_{}", uuid::Uuid::new_v4().simple());
            let (mut rs_open, mut msg_open) = (false, false);
            let (mut rs_index, mut msg_index) = (0usize, 0usize);
            let (mut rs_emitted, mut emitted) = (0usize, 0usize);
            let mut ids: Vec<u32> = Vec::new();
            // incremental decode of `ids` (O(n^2) collapse fix)
            let mut sd = meta.tokenizer.stream_decoder(false);
            let mut finish = None;

            loop {
                match rx.recv().await {
                    Some(TokenEvent::Prefilled { cached, rows }) => {
                        if round == 0 {
                            cached0 = cached as usize;
                            meta.prompt_len = meta.prompt_len.max(rows as usize);
                        }
                        t_prefill_done = Some(std::time::Instant::now());
                    }
                    Some(TokenEvent::Token { id, .. }) => {
                        if t_prefill_done.is_none() { t_prefill_done = Some(std::time::Instant::now()); }
                        ids.push(id);
                        let raw = sd.push(&meta.tokenizer, id);
                        let parsed = parse(dialect, &raw, thinking_open, hints.as_ref());
                        if let Some(reasoning) = &parsed.reasoning {
                            let safe = safe_len(reasoning, dialect.reasoning_markers());
                            if safe > rs_emitted {
                                if !rs_open {
                                    rs_open = true;
                                    rs_index = output_index;
                                    output_index += 1;
                                    yield ev("response.output_item.added", json!({"type":"response.output_item.added","sequence_number":seq,"output_index":rs_index,"item":{"type":"reasoning","id":rs_id,"summary":[],"content":[]}}));
                                    seq += 1;
                                }
                                let delta = reasoning[rs_emitted..safe].to_owned();
                                rs_emitted = safe;
                                yield ev("response.reasoning_text.delta", json!({"type":"response.reasoning_text.delta","sequence_number":seq,"item_id":rs_id,"output_index":rs_index,"content_index":0,"delta":delta}));
                                seq += 1;
                            }
                        }
                        if let Some(content) = &parsed.content {
                            let safe = safe_len(content, dialect.content_markers());
                            if safe > emitted {
                                if !msg_open {
                                    msg_open = true;
                                    msg_index = output_index;
                                    output_index += 1;
                                    yield ev("response.output_item.added", json!({"type":"response.output_item.added","sequence_number":seq,"output_index":msg_index,"item":{"type":"message","id":msg_id,"role":"assistant","status":"in_progress","content":[]}}));
                                    seq += 1;
                                    yield ev("response.content_part.added", json!({"type":"response.content_part.added","sequence_number":seq,"item_id":msg_id,"output_index":msg_index,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}));
                                    seq += 1;
                                }
                                let delta = content[emitted..safe].to_owned();
                                emitted = safe;
                                yield ev("response.output_text.delta", json!({"type":"response.output_text.delta","sequence_number":seq,"item_id":msg_id,"output_index":msg_index,"content_index":0,"delta":delta,"logprobs":[]}));
                                seq += 1;
                            }
                        }
                    }
                    Some(TokenEvent::Done(r, stats)) => {
                        finish = Some(r);
                        meta.scope.phases(&stats);
                        break;
                    }
                    Some(TokenEvent::Error(e)) => {
                        let mut snap = response_object(&meta, "failed", vec![], None, None);
                        snap["error"] = json!({"code": e.code.unwrap_or("internal_error"), "message": e.message});
                        yield ev("response.failed", json!({"type":"response.failed","sequence_number":seq,"response":snap}));
                        return;
                    }
                    None => break,
                }
            }
            total_out += ids.len();
            {
                let total_ms = t_round.elapsed().as_millis() as u64;
                let prefill_ms = t_prefill_done.map(|t| (t - t_round).as_millis() as u64).unwrap_or(0);
                let decode_ms = t_prefill_done.map(|t| t.elapsed().as_millis() as u64).unwrap_or(total_ms);
                let gen_tokens = ids.len();
                let decode_tps = if decode_ms > 0 { gen_tokens as f64 * 1000.0 / decode_ms as f64 } else { 0.0 };
                tracing::debug!(round, ctx_tokens, gen_tokens, prefill_ms, decode_ms, total_ms, decode_tps, "agent round");
            }
            let raw = meta.tokenizer.decode(&ids, false).unwrap_or_default();
            // Dev instrument: the round's RAW text, specials visible. Tool-call
            // bytes stream nothing, so when a round misbehaves this is the only
            // way to see what the model actually generated. Same switch as
            // the other request traces.
            if paddock_models::dev_var_os!("PADDOCK_REQ_TRACE").is_some() {
                eprintln!("req-trace: agent round {round} raw ({} tokens):
    {raw}", ids.len());
            }
            let mut parsed = parse(dialect, &raw, thinking_open, hints.as_ref());
            if single {
                parsed.tool_calls.truncate(1);
                parsed.complete_calls = parsed.complete_calls.min(1);
            }
            total_reasoning += reasoning_tokens(&meta, &parsed);

            if rs_open {
                let text = parsed.reasoning.clone().unwrap_or_default();
                yield ev("response.reasoning_text.done", json!({"type":"response.reasoning_text.done","sequence_number":seq,"item_id":rs_id,"output_index":rs_index,"content_index":0,"text":text}));
                seq += 1;
                let it = json!({"type":"reasoning","id":rs_id,"summary":[],"content":[{"type":"reasoning_text","text":text}]});
                done_items.push(it.clone());
                yield ev("response.output_item.done", json!({"type":"response.output_item.done","sequence_number":seq,"output_index":rs_index,"item":it}));
                seq += 1;
            }
            if msg_open {
                let text = parsed.content.clone().unwrap_or_default();
                yield ev("response.output_text.done", json!({"type":"response.output_text.done","sequence_number":seq,"item_id":msg_id,"output_index":msg_index,"content_index":0,"text":text,"logprobs":[]}));
                seq += 1;
                yield ev("response.content_part.done", json!({"type":"response.content_part.done","sequence_number":seq,"item_id":msg_id,"output_index":msg_index,"content_index":0,"part":{"type":"output_text","text":text,"annotations":[]}}));
                seq += 1;
                let it = json!({"type":"message","id":msg_id,"role":"assistant","status":"completed","content":[{"type":"output_text","text":text,"annotations":[]}]});
                done_items.push(it.clone());
                yield ev("response.output_item.done", json!({"type":"response.output_item.done","sequence_number":seq,"output_index":msg_index,"item":it}));
                seq += 1;
            }

            // The answer round ends the turn whatever it said - its text has
            // already streamed, and nothing tool-shaped in it may run.
            if answering {
                final_finish = finish;
                break;
            }
            let calls = round_calls(&parsed, &gathered, &mut ledger);
            if calls.is_empty() {
                final_finish = finish;
                break;
            }
            if matches!(finish, Some(FinishReason::Length)) {
                if !ours {
                    // The caller's own cap: their truncation, reported as it was.
                    final_finish = finish;
                    break;
                }
                // Ours: the tool budget ran out mid-round. Nothing from a cut
                // round is dispatched, and the answer round follows with the
                // request's full cap.
                stop = Some(loop_budget::Stop::Output);
                continue;
            }

            let tc_json: Vec<Value> = calls
                .iter()
                .map(|rc| json!({"id":rc.call_id,"type":"function","function":{"name":rc.raw_name,"arguments":rc.raw_args}}))
                .collect();
            let mut asst = serde_json::Map::new();
            asst.insert("role".into(), json!("assistant"));
            if let Some(c) = &parsed.content {
                asst.insert("content".into(), json!(c));
            }
            asst.insert("tool_calls".into(), Value::Array(tc_json));
            messages.push(Value::Object(asst));

            for rc in &calls {
                let call_id = rc.call_id.clone();
                let raw_args = &rc.raw_args;
                match &rc.kind {
                    // Catalog search: local, ungated. Surfaces as an mcp_call card
                    // (server_label "mcp") so the user sees the discovery step.
                    CallKind::Search { query, limit } => {
                        // Budgeted like the non-streamed lane: a fourth
                        // search returns the index it already has, not a fourth
                        // ranking of the same catalog.
                        let result = match ledger.search_budget_spent() {
                            Some(spent) => spent,
                            None => {
                                let hits = crate::tool_search::search(&gathered.catalog, query, *limit);
                                crate::tool_search::search_result(query, &hits, &gathered.catalog)
                            }
                        };
                        tracing::debug!(round, query = %query, searches = ledger.searches(), result_bytes = result.len(), "agent tool-search");
                        let idx = output_index;
                        output_index += 1;
                        yield ev("response.output_item.added", json!({"type":"response.output_item.added","sequence_number":seq,"output_index":idx,"item":mcp_call_item(&call_id,"mcp",crate::tool_search::SEARCH_TOOL,raw_args,None,None,None,"in_progress")}));
                        seq += 1;
                        yield ev("response.mcp_call.in_progress", json!({"type":"response.mcp_call.in_progress","sequence_number":seq,"output_index":idx,"item_id":call_id}));
                        seq += 1;
                        yield ev("response.mcp_call_arguments.delta", json!({"type":"response.mcp_call_arguments.delta","sequence_number":seq,"output_index":idx,"item_id":call_id,"delta":raw_args}));
                        seq += 1;
                        yield ev("response.mcp_call_arguments.done", json!({"type":"response.mcp_call_arguments.done","sequence_number":seq,"output_index":idx,"item_id":call_id,"arguments":raw_args}));
                        seq += 1;
                        if let Some(sig) = &rc.sig {
                            ledger.record(sig, true, &result);
                        }
                        messages.push(json!({"role":"tool","content":result,"tool_call_id":call_id}));
                        yield ev("response.mcp_call.completed", json!({"type":"response.mcp_call.completed","sequence_number":seq,"output_index":idx,"item_id":call_id}));
                        seq += 1;
                        let it = mcp_call_item(&call_id,"mcp",crate::tool_search::SEARCH_TOOL,raw_args,None,Some(result.as_str()),None,"completed");
                        done_items.push(it.clone());
                        yield ev("response.output_item.done", json!({"type":"response.output_item.done","sequence_number":seq,"output_index":idx,"item":it}));
                        seq += 1;
                    }
                    // Spec web_search_call lifecycle: added(in_progress) ->
                    // in_progress -> searching -> [execute] -> completed -> done
                    // (item status carries failed; there is no failed event name).
                    CallKind::Web { query } => {
                        let spec = gathered.web_search.clone().expect("web_search config");
                        let idx = output_index;
                        output_index += 1;
                        yield ev("response.output_item.added", json!({"type":"response.output_item.added","sequence_number":seq,"output_index":idx,"item":crate::websearch::call_item(&call_id, spec.cfg.provider, "in_progress", query, &[], None)}));
                        seq += 1;
                        yield ev("response.web_search_call.in_progress", json!({"type":"response.web_search_call.in_progress","sequence_number":seq,"output_index":idx,"item_id":call_id}));
                        seq += 1;
                        yield ev("response.web_search_call.searching", json!({"type":"response.web_search_call.searching","sequence_number":seq,"output_index":idx,"item_id":call_id}));
                        seq += 1;
                        let t_tool = std::time::Instant::now();
                        let done = crate::websearch::execute(&spec, query).await;
                        tracing::debug!(round, query = %query, hits = done.hits.len(), tool_ms = t_tool.elapsed().as_millis() as u64, status = %done.status, "agent web search");
                        if let Some(sig) = &rc.sig {
                            ledger.record(sig, done.status == "completed", &done.feedback);
                        }
                        crate::metrics::web_search_billed(&spec.cfg.provider, &done.usage);
                        messages.push(json!({"role":"tool","content":done.feedback,"tool_call_id":call_id}));
                        yield ev("response.web_search_call.completed", json!({"type":"response.web_search_call.completed","sequence_number":seq,"output_index":idx,"item_id":call_id}));
                        seq += 1;
                        let it = crate::websearch::call_item(&call_id, spec.cfg.provider, done.status, query, &done.hits, done.error.as_deref());
                        done_items.push(it.clone());
                        yield ev("response.output_item.done", json!({"type":"response.output_item.done","sequence_number":seq,"output_index":idx,"item":it}));
                        seq += 1;
                    }
                    // Forensics rides the generic mcp_call lifecycle (added ->
                    // in_progress -> completed|failed -> done) with a "forensics"
                    // label; there is no dedicated spec event family for it.
                    CallKind::Forensics { image_index } => {
                        let rt = gathered.forensics.clone().expect("forensics runtime");
                        let args = json!({ "image_index": image_index }).to_string();
                        let idx = output_index;
                        output_index += 1;
                        yield ev("response.output_item.added", json!({"type":"response.output_item.added","sequence_number":seq,"output_index":idx,"item":mcp_call_item(&call_id,"forensics",crate::forensics::TOOL_NAME,&args,None,None,None,"in_progress")}));
                        seq += 1;
                        yield ev("response.mcp_call.in_progress", json!({"type":"response.mcp_call.in_progress","sequence_number":seq,"output_index":idx,"item_id":call_id}));
                        seq += 1;
                        let t_tool = std::time::Instant::now();
                        let (content, output, error, status) = run_forensics_tool(&rt, messages.as_slice(), *image_index).await;
                        tracing::debug!(round, image_index = ?image_index, tool_ms = t_tool.elapsed().as_millis() as u64, status = %status, "agent forensics");
                        if let Some(sig) = &rc.sig {
                            ledger.record(sig, status == "completed", &content);
                        }
                        messages.push(json!({"role":"tool","content":content,"tool_call_id":call_id}));
                        let done_event = if status == "completed" { "response.mcp_call.completed" } else { "response.mcp_call.failed" };
                        yield ev(done_event, json!({"type":done_event,"sequence_number":seq,"output_index":idx,"item_id":call_id}));
                        seq += 1;
                        let it = mcp_call_item(&call_id,"forensics",crate::forensics::TOOL_NAME,&args,None,output.as_deref(),error.as_deref(),status);
                        done_items.push(it.clone());
                        yield ev("response.output_item.done", json!({"type":"response.output_item.done","sequence_number":seq,"output_index":idx,"item":it}));
                        seq += 1;
                    }
                    // The clock rides the same generic mcp_call lifecycle with a
                    // "time" label. It is answered in-process, so added and done
                    // land back-to-back - no in_progress beat needed between them.
                    CallKind::Clock => {
                        let spec = gathered.current_time.expect("clock spec");
                        let idx = output_index;
                        output_index += 1;
                        yield ev("response.output_item.added", json!({"type":"response.output_item.added","sequence_number":seq,"output_index":idx,"item":mcp_call_item(&call_id,"time",crate::clock::TOOL_NAME,&rc.raw_args,None,None,None,"in_progress")}));
                        seq += 1;
                        let (content, output, error, status) = crate::clock::run(spec, &rc.raw_args);
                        if let Some(sig) = &rc.sig {
                            ledger.record(sig, status == "completed", &content);
                        }
                        messages.push(json!({"role":"tool","content":content,"tool_call_id":call_id}));
                        let done_event = if status == "completed" { "response.mcp_call.completed" } else { "response.mcp_call.failed" };
                        yield ev(done_event, json!({"type":done_event,"sequence_number":seq,"output_index":idx,"item_id":call_id}));
                        seq += 1;
                        let it = mcp_call_item(&call_id,"time",crate::clock::TOOL_NAME,&rc.raw_args,None,output.as_deref(),error.as_deref(),status);
                        done_items.push(it.clone());
                        yield ev("response.output_item.done", json!({"type":"response.output_item.done","sequence_number":seq,"output_index":idx,"item":it}));
                        seq += 1;
                    }
                    CallKind::Invoke { ns_name, args } => {
                        let Some((cfg, real)) = gathered.routing.get(ns_name).cloned() else {
                            // Model called a name that isn't in the catalog.
                            let idx = output_index;
                            output_index += 1;
                            let m = format!("unknown tool {ns_name:?}; call {} to find available tools", crate::tool_search::SEARCH_TOOL);
                            if let Some(sig) = &rc.sig {
                                ledger.record(sig, false, &m);
                            }
                            yield ev("response.output_item.added", json!({"type":"response.output_item.added","sequence_number":seq,"output_index":idx,"item":mcp_call_item(&call_id,"mcp",ns_name,args,None,None,None,"in_progress")}));
                            seq += 1;
                            messages.push(json!({"role":"tool","content":m,"tool_call_id":call_id}));
                            yield ev("response.mcp_call.failed", json!({"type":"response.mcp_call.failed","sequence_number":seq,"output_index":idx,"item_id":call_id}));
                            seq += 1;
                            let it = mcp_call_item(&call_id,"mcp",ns_name,args,None,None,Some(m.as_str()),"failed");
                            done_items.push(it.clone());
                            yield ev("response.output_item.done", json!({"type":"response.output_item.done","sequence_number":seq,"output_index":idx,"item":it}));
                            seq += 1;
                            continue;
                        };

                        // Human-in-the-loop (Studio extension): a gated tool surfaces
                        // an `mcp_approval_request` and the loop parks until approve/
                        // deny (or timeout -> deny). `call_id` links the request to the
                        // Studio's call card.
                        if gathered.needs_approval.contains(ns_name) {
                            let approval_id = format!("appr_{}", uuid::Uuid::new_v4().simple());
                            let aidx = output_index;
                            output_index += 1;
                            yield ev("response.output_item.added", json!({"type":"response.output_item.added","sequence_number":seq,"output_index":aidx,"item":{"type":"mcp_approval_request","id":approval_id,"call_id":call_id,"server_label":cfg.label,"name":real,"arguments":args}}));
                            seq += 1;
                            let rx = state.approvals.register(approval_id.clone());
                            let approved = match tokio::time::timeout(APPROVAL_TIMEOUT, rx).await {
                                Ok(Ok(v)) => v,
                                _ => {
                                    state.approvals.cancel(&approval_id);
                                    false
                                }
                            };
                            let decision = if approved { "approved" } else { "denied" };
                            let it = json!({"type":"mcp_approval_request","id":approval_id,"call_id":call_id,"server_label":cfg.label,"name":real,"arguments":args,"status":decision});
                            done_items.push(it.clone());
                            yield ev("response.output_item.done", json!({"type":"response.output_item.done","sequence_number":seq,"output_index":aidx,"item":it}));
                            seq += 1;
                            if !approved {
                                messages.push(json!({"role":"tool","content":DENIED_MSG,"tool_call_id":call_id}));
                                continue;
                            }
                        }

                        // Spec mcp_call lifecycle: added(in_progress) -> in_progress ->
                        // arguments.delta -> arguments.done -> [execute] -> completed|failed -> done.
                        let idx = output_index;
                        output_index += 1;
                        yield ev("response.output_item.added", json!({"type":"response.output_item.added","sequence_number":seq,"output_index":idx,"item":mcp_call_item(&call_id,&cfg.label,&real,args,None,None,None,"in_progress")}));
                        seq += 1;
                        yield ev("response.mcp_call.in_progress", json!({"type":"response.mcp_call.in_progress","sequence_number":seq,"output_index":idx,"item_id":call_id}));
                        seq += 1;
                        yield ev("response.mcp_call_arguments.delta", json!({"type":"response.mcp_call_arguments.delta","sequence_number":seq,"output_index":idx,"item_id":call_id,"delta":args}));
                        seq += 1;
                        yield ev("response.mcp_call_arguments.done", json!({"type":"response.mcp_call_arguments.done","sequence_number":seq,"output_index":idx,"item_id":call_id,"arguments":args}));
                        seq += 1;
                        let t_tool = std::time::Instant::now();
                        let (feedback, output, error, status) = execute_mcp_call(&state, &cfg, real.as_str(), args).await;
                        tracing::debug!(round, tool = %real, result_bytes = feedback.len(), tool_ms = t_tool.elapsed().as_millis() as u64, status = %status, "agent tool call");
                        if let Some(sig) = &rc.sig {
                            ledger.record(sig, status == "completed", &feedback);
                        }
                        messages.push(json!({"role":"tool","content":feedback,"tool_call_id":call_id}));
                        let done_event = if status == "completed" { "response.mcp_call.completed" } else { "response.mcp_call.failed" };
                        yield ev(done_event, json!({"type":done_event,"sequence_number":seq,"output_index":idx,"item_id":call_id}));
                        seq += 1;
                        let it = mcp_call_item(&call_id,&cfg.label,&real,args,None,output.as_deref(),error.as_deref(),status);
                        done_items.push(it.clone());
                        yield ev("response.output_item.done", json!({"type":"response.output_item.done","sequence_number":seq,"output_index":idx,"item":it}));
                        seq += 1;
                    }
                    // Refused before dispatch: the arguments did not match the
                    // tool's own schema, or the loop budget stopped a repeat.
                    // Same failed-call lifecycle as an unknown name - nothing
                    // reached a server, and the message says what to do next.
                    CallKind::Refuse { name, message } => {
                        let idx = output_index;
                        output_index += 1;
                        yield ev("response.output_item.added", json!({"type":"response.output_item.added","sequence_number":seq,"output_index":idx,"item":mcp_call_item(&call_id,"mcp",name,raw_args,None,None,None,"in_progress")}));
                        seq += 1;
                        messages.push(json!({"role":"tool","content":message,"tool_call_id":call_id}));
                        yield ev("response.mcp_call.failed", json!({"type":"response.mcp_call.failed","sequence_number":seq,"output_index":idx,"item_id":call_id}));
                        seq += 1;
                        let it = mcp_call_item(&call_id,"mcp",name,raw_args,None,None,Some(message.as_str()),"failed");
                        done_items.push(it.clone());
                        yield ev("response.output_item.done", json!({"type":"response.output_item.done","sequence_number":seq,"output_index":idx,"item":it}));
                        seq += 1;
                    }
                    // Already run this turn with these exact arguments: the
                    // first result comes back out of the ledger, and the card
                    // carries the tool a live call would have shown.
                    CallKind::Replay { ns_name, output } => {
                        let (label, real) = replay_identity(&gathered, ns_name);
                        let idx = output_index;
                        output_index += 1;
                        yield ev("response.output_item.added", json!({"type":"response.output_item.added","sequence_number":seq,"output_index":idx,"item":mcp_call_item(&call_id,label,real,raw_args,None,None,None,"in_progress")}));
                        seq += 1;
                        messages.push(json!({"role":"tool","content":output,"tool_call_id":call_id}));
                        yield ev("response.mcp_call.completed", json!({"type":"response.mcp_call.completed","sequence_number":seq,"output_index":idx,"item_id":call_id}));
                        seq += 1;
                        let it = mcp_call_item(&call_id,label,real,raw_args,None,Some(output.as_str()),None,"completed");
                        done_items.push(it.clone());
                        yield ev("response.output_item.done", json!({"type":"response.output_item.done","sequence_number":seq,"output_index":idx,"item":it}));
                        seq += 1;
                    }
                }
            }
            // The whole turn's generation is bounded, not just each round.
            if total_out >= turn_cap {
                stop = Some(loop_budget::Stop::Output);
            }
        }

        let (status, event_name) = terminal(final_finish);
        meta.scope.usage(meta.prompt_len, total_out);
        meta.scope.cached(cached0);
        meta.scope.finish(final_finish.map_or("stop", |f| f.as_str()));
        let full = response_object(&meta, status, done_items, Some((total_out, total_reasoning, cached0)), final_finish);
        yield ev(event_name, json!({"type":event_name,"sequence_number":seq,"response":full}));
    };
    Sse::new(sse).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn forensics_runtime() -> std::sync::Arc<crate::forensics::ForensicRuntime> {
        crate::forensics::ForensicRuntime::build(&crate::config::ForensicsConfig {
            enabled: true,
            auto: crate::config::ForensicsAuto::Off,
            tool: true,
            device: None,
        })
        .expect("runtime builds when enabled")
    }

    /// A deterministic checkerboard|smooth PNG that fires ELA, as a data-URI
    /// chat message (the shape the tool resolves images from).
    fn image_message() -> Value {
        use base64::Engine as _;
        use image::{ExtendedColorType, ImageEncoder};
        let (w, h) = (512u32, 512u32);
        let mut rgb = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 3) as usize;
                let v: u8 = if x < 256 {
                    if (x + y) & 1 == 0 { 0 } else { 255 }
                } else {
                    (y * 255 / h) as u8
                };
                rgb[i] = v;
                rgb[i + 1] = v;
                rgb[i + 2] = v;
            }
        }
        // JPEG so ELA applies (ela is JPEG-only, per the reference should_skip).
        let mut jpg = std::io::Cursor::new(Vec::new());
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpg, 92)
            .write_image(&rgb, w, h, ExtendedColorType::Rgb8)
            .expect("encode jpeg");
        let b64 = base64::engine::general_purpose::STANDARD.encode(jpg.into_inner());
        json!({"role":"user","content":[
            {"type":"image_url","image_url":{"url":format!("data:image/jpeg;base64,{b64}")}}
        ]})
    }

    #[test]
    fn forensics_tool_type_is_served_and_dropped_from_function_defs() {
        // The gate the live smoke test caught: `{"type":"forensics"}` must pass
        // request validation (served) and be dropped from the function defs
        // (handled by gather, like web_search/mcp), not 400 as "unsupported".
        assert!(served_tool_type(Some("forensics")));
        let out = normalize_tools(&[json!({"type": "forensics"})]).expect("forensics is served");
        assert!(
            out.is_empty(),
            "forensics is handled by gather, not a function def"
        );
        // an unknown type still errors loudly (no silent drop)
        assert!(normalize_tools(&[json!({"type": "file_search"})]).is_err());
    }

    #[test]
    fn clock_tool_type_is_served_and_dropped_from_function_defs() {
        assert!(served_tool_type(Some("current_time")));
        let out =
            normalize_tools(&[json!({"type": "current_time", "timezone": "Europe/Stockholm"})])
                .expect("current_time is served");
        assert!(
            out.is_empty(),
            "current_time is handled by gather, not a function def"
        );
    }

    #[test]
    fn clock_tool_classifies_only_when_gathered() {
        let mut g = one_tool_gathered();
        g.current_time = Some(
            crate::clock::parse_spec(
                &json!({"type": "current_time", "timezone": "Europe/Stockholm"}),
            )
            .expect("valid zone"),
        );
        assert!(
            matches!(
                classify_call(crate::clock::TOOL_NAME, "{}", &g),
                Some(CallKind::Clock)
            ),
            "expected a Clock call"
        );
        // Not gathered -> a caller's own function of the same name passes through.
        g.current_time = None;
        assert!(classify_call(crate::clock::TOOL_NAME, "{}", &g).is_none());
    }

    #[test]
    fn forensics_tool_classifies_only_when_gathered() {
        let mut g = one_tool_gathered();
        g.forensics = Some(forensics_runtime());
        match classify_call(crate::forensics::TOOL_NAME, "{}", &g) {
            Some(CallKind::Forensics { image_index }) => assert_eq!(image_index, None),
            _ => panic!("expected a Forensics call"),
        }
        match classify_call(crate::forensics::TOOL_NAME, r#"{"image_index":2}"#, &g) {
            Some(CallKind::Forensics { image_index }) => assert_eq!(image_index, Some(2)),
            _ => panic!("expected a Forensics call carrying the index"),
        }
        // Not gathered -> a caller's own function of the same name passes through.
        g.forensics = None;
        assert!(classify_call(crate::forensics::TOOL_NAME, "{}", &g).is_none());
    }

    #[tokio::test]
    async fn forensics_tool_executes_over_conversation_image() {
        let rt = forensics_runtime();
        let messages = vec![image_message()];
        let (content, output, error, status) = run_forensics_tool(&rt, &messages, None).await;
        assert_eq!(status, "completed");
        assert!(error.is_none());
        assert!(output.is_some());
        assert!(
            content.contains("ela_block_outliers"),
            "tool result should carry the finding: {content}"
        );

        // No image in the conversation -> a clean failed status, never a panic.
        let (_c, _o, err, status) = run_forensics_tool(&rt, &[], None).await;
        assert_eq!(status, "failed");
        assert!(err.is_some());
    }

    /// A two-tool `Gathered` with no servers to talk to: enough for the
    /// classification and ledger decisions, which is all that runs here.
    fn one_tool_gathered() -> Gathered {
        let cfg = paddock_mcp::ServerConfig {
            id: "test".into(),
            label: "artifacts".into(),
            transport: paddock_mcp::Transport::Http {
                url: "http://localhost/mcp".into(),
                headers: HashMap::new(),
            },
        };
        let schema = json!({"type":"object","properties":{"artifact_id":{"type":"string"}},
                            "required":["artifact_id"]});
        Gathered {
            tools: vec![],
            routing: HashMap::from([(
                "artifacts__artifact_read".to_string(),
                (cfg, "artifact_read".to_string()),
            )]),
            needs_approval: HashSet::new(),
            listings: vec![],
            catalog: vec![crate::tool_search::CatalogTool {
                name: "artifacts__artifact_read".into(),
                description: "read one".into(),
                input_schema: schema,
            }],
            web_search: None,
            forensics: None,
            current_time: None,
            instructions: vec![],
        }
    }

    fn parsed_call(name: &str, arguments: &str) -> Parsed {
        Parsed {
            content: None,
            reasoning: None,
            tool_calls: vec![crate::parsers::ToolCallRaw {
                name: name.into(),
                arguments: arguments.into(),
            }],
            complete_calls: 1,
        }
    }

    /// Budget lever 1 through the Responses classifier: the same call twice in a
    /// turn comes back off the ledger, and a third time is refused.
    #[test]
    fn a_repeated_call_replays_then_is_refused() {
        let g = one_tool_gathered();
        let mut ledger = loop_budget::CallLedger::new();
        let p = parsed_call("artifacts__artifact_read", r#"{"artifact_id":"a1"}"#);

        let first = round_calls(&p, &g, &mut ledger);
        let sig = match &first[0].kind {
            CallKind::Invoke { .. } => first[0].sig.clone().expect("a call that runs is filed"),
            other => panic!("expected an invoke: {:?}", std::mem::discriminant(other)),
        };
        ledger.record(&sig, true, "the page");

        let second = round_calls(&p, &g, &mut ledger);
        match &second[0].kind {
            CallKind::Replay { ns_name, output } => {
                assert_eq!(ns_name, "artifacts__artifact_read");
                assert!(output.ends_with("the page"), "{output}");
            }
            _ => panic!("expected a replay"),
        }
        assert!(
            second[0].sig.is_none(),
            "a replay must not be filed as a run"
        );

        let third = round_calls(&p, &g, &mut ledger);
        match &third[0].kind {
            CallKind::Refuse { message, .. } => assert!(message.contains("twice"), "{message}"),
            _ => panic!("expected a refusal"),
        }
    }

    /// The wrapper and the direct call are one call: the resolver gives us the
    /// resolved identity, and the ledger is keyed on it, so a model that switches
    /// spelling mid-loop does not get a free round out of it.
    #[test]
    fn the_wrapper_and_the_direct_call_are_the_same_call() {
        let g = one_tool_gathered();
        let mut ledger = loop_budget::CallLedger::new();
        let direct = parsed_call("artifacts__artifact_read", r#"{"artifact_id":"a1"}"#);
        let sig = round_calls(&direct, &g, &mut ledger)[0]
            .sig
            .clone()
            .expect("filed");
        ledger.record(&sig, true, "the page");

        let wrapped = parsed_call(
            crate::tool_search::CALL_TOOL,
            r#"{"name":"artifacts__artifact_read","arguments_json":"{\"artifact_id\":\"a1\"}"}"#,
        );
        match &round_calls(&wrapped, &g, &mut ledger)[0].kind {
            CallKind::Replay { .. } => {}
            _ => panic!("the same call through mcp_call_tool must hit the ledger"),
        }
    }

    /// A round cut before any tool ran leaves the user's own turn last, and
    /// back-to-back user turns is a shape some chat templates refuse. The
    /// nudge joins that turn instead of opening another.
    #[test]
    fn the_answer_nudge_never_opens_a_second_user_turn() {
        let mut msgs = vec![json!({"role":"user","content":"go"})];
        push_answer_nudge(&mut msgs, "answer now");
        assert_eq!(msgs.len(), 1);
        assert!(
            msgs[0]["content"]
                .as_str()
                .expect("text")
                .ends_with("answer now")
        );

        // A multimodal user turn takes it as one more text part.
        let mut msgs = vec![json!({"role":"user","content":[{"type":"text","text":"go"}]})];
        push_answer_nudge(&mut msgs, "answer now");
        assert_eq!(msgs[0]["content"].as_array().expect("parts").len(), 2);

        // After a tool round it is a turn of its own, which is the usual case.
        let mut msgs = vec![json!({"role":"tool","content":"result","tool_call_id":"c1"})];
        push_answer_nudge(&mut msgs, "answer now");
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[1]["role"], "user");
    }

    /// The caller's own `max_tool_calls`, through the same classifier.
    /// The Nth+1 call is not dispatched - it comes back as a refusal naming
    /// their knob - and the loop reads `limit_reached` to take the answer round.
    #[test]
    fn the_callers_max_tool_calls_stops_dispatch_and_then_the_turn() {
        let g = one_tool_gathered();
        let mut ledger = loop_budget::CallLedger::with_limit(Some(1));
        let a = parsed_call("artifacts__artifact_read", r#"{"artifact_id":"a1"}"#);
        let sig = round_calls(&a, &g, &mut ledger)[0]
            .sig
            .clone()
            .expect("the first one runs");
        ledger.record(&sig, true, "the page");
        assert_eq!(
            ledger.limit_reached(),
            Some(loop_budget::Stop::ToolCalls(1))
        );

        // Different arguments, so this is a fresh call by every other rule -
        // only the caller's ceiling stops it.
        let b = parsed_call("artifacts__artifact_read", r#"{"artifact_id":"a2"}"#);
        let calls = round_calls(&b, &g, &mut ledger);
        match &calls[0].kind {
            CallKind::Refuse { message, .. } => {
                assert!(message.contains("max_tool_calls: 1"), "{message}");
            }
            _ => panic!("past the caller's limit nothing may be dispatched"),
        }
        assert!(calls[0].sig.is_none(), "a refusal is not a run");
    }

    /// A schema refusal never ran, so it must not consume a ledger slot -
    /// otherwise a model that fixes its arguments on the retry would find its
    /// corrected call counted as a repeat.
    #[test]
    fn a_schema_refusal_leaves_the_ledger_alone() {
        let g = one_tool_gathered();
        let mut ledger = loop_budget::CallLedger::new();
        let bad = parsed_call("artifacts__artifact_read", r#"{}"#);
        assert!(matches!(
            round_calls(&bad, &g, &mut ledger)[0].kind,
            CallKind::Refuse { .. }
        ));
        // ...and again, and it is still the SCHEMA refusal, not a repeat one.
        match &round_calls(&bad, &g, &mut ledger)[0].kind {
            CallKind::Refuse { message, .. } => {
                assert!(message.contains("did not match its schema"), "{message}");
            }
            _ => panic!("expected the schema refusal both times"),
        }
        // The corrected call still runs.
        let good = parsed_call("artifacts__artifact_read", r#"{"artifact_id":"a1"}"#);
        assert!(matches!(
            round_calls(&good, &g, &mut ledger)[0].kind,
            CallKind::Invoke { .. }
        ));
    }

    /// The invariant we asked for: the block lives in the SYSTEM slot, so a
    /// compaction that rebuilds the messages from `req.instructions` re-renders
    /// it. Putting it only in the message list meant the user's prompt survived
    /// compaction and ours did not.
    #[test]
    fn server_instructions_go_into_the_system_slot_so_compaction_keeps_them() {
        let merged = merge_instructions(Some("You are terse.".into()), &["Use artifacts.".into()]);
        let sys = merged.expect("a system prompt");
        // The user's line goes last: a short instruction stops being obeyed
        // when 200 words of tool procedure trail it.
        assert!(
            sys.ends_with("You are terse."),
            "the user's prompt must end it: {sys}"
        );
        assert!(
            sys.starts_with("Use artifacts."),
            "tool guidance leads: {sys}"
        );
        // Whatever a later rebuild does, it starts from this string.
        let rebuilt = messages_from_input(Some(&sys), &json!("hi")).unwrap();
        assert_eq!(rebuilt[0]["role"], "system");
        assert!(
            rebuilt[0]["content"]
                .as_str()
                .unwrap()
                .contains("Use artifacts.")
        );
    }

    #[test]
    fn merge_leaves_the_prompt_alone_when_no_server_speaks() {
        assert_eq!(
            merge_instructions(Some("mine".into()), &[]),
            Some("mine".into())
        );
        assert_eq!(merge_instructions(None, &[]), None);
    }

    #[test]
    fn merge_creates_the_prompt_when_the_user_set_none() {
        assert_eq!(
            merge_instructions(None, &["block".into()]),
            Some("block".into())
        );
        assert_eq!(
            merge_instructions(Some("   ".into()), &["block".into()]),
            Some("block".into())
        );
    }

    #[test]
    fn server_instructions_append_to_the_users_system_prompt() {
        // A server says how to use itself; it does not get to overwrite what
        // the user asked for. Order matters: the user's prompt leads.
        let mut msgs = vec![
            json!({"role": "system", "content": "You are terse."}),
            json!({"role": "user", "content": "hi"}),
        ];
        apply_server_instructions(&mut msgs, &["Put pages in artifacts.".to_string()]);
        assert_eq!(msgs.len(), 2, "no extra system message when one exists");
        let sys = msgs[0]["content"].as_str().unwrap();
        assert!(
            sys.ends_with("You are terse."),
            "the user's prompt must end it: {sys}"
        );
        assert!(sys.starts_with("Put pages in artifacts."), "{sys}");
    }

    /// The seam that broke twice. On this API the system prompt is
    /// `instructions`, not an item - so a capability applied inside attachment
    /// expansion, which only ever sees the items, lands as a SECOND system
    /// message after conversion. That is what shipped, and it is why a
    /// geotagged photo never got its map: the model was told twice-removed or
    /// not at all. The assertion that matters is the COUNT.
    #[test]
    fn the_map_capability_joins_the_instructions_system_turn() {
        let mut msgs = messages_from_input(
            Some("You are terse."),
            &json!([{"role": "user", "content": "hi"}]),
        )
        .expect("converts");
        crate::doc::add_map_capability(&mut msgs, "43.467448, 11.885127, Arezzo");
        assert_eq!(
            msgs.iter().filter(|m| m["role"] == "system").count(),
            1,
            "one system turn, never two: {msgs:?}"
        );
        let sys = msgs[0]["content"].as_str().expect("string content");
        assert!(sys.contains("```map"), "the block must be shown: {sys}");
        assert!(
            sys.contains("43.467448, 11.885127, Arezzo"),
            "ready to copy: {sys}"
        );
        assert!(
            sys.ends_with("You are terse."),
            "the caller keeps the tail: {sys}"
        );
        assert_eq!(msgs[1]["role"], "user", "the user turn stays after it");
    }

    /// ...and with no instructions at all, it is the system turn.
    #[test]
    fn the_map_capability_creates_the_system_turn_when_there_is_none() {
        let mut msgs = messages_from_input(None, &json!([{"role": "user", "content": "hi"}]))
            .expect("converts");
        crate::doc::add_map_capability(&mut msgs, "43.4, 11.8, Arezzo");
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[1]["role"], "user");
    }

    /// The third failure of the same seam, and the one `messages` alone cannot
    /// survive: every compaction rebuilds the message list from
    /// `req.instructions`, so a capability patched into `messages` is gone
    /// while the photo line, which lives in the user item, sails through - the
    /// model keeps the coordinates and loses the reason to draw them. The
    /// handler folds it into the instructions slot; this pins that a rebuild
    /// re-renders it.
    #[test]
    fn the_map_capability_survives_a_rebuild_from_instructions() {
        let text = crate::doc::map_capability_text("43.467157, 11.885395, Arezzo");
        let instructions =
            merge_instructions(Some("You are terse.".into()), &[text]).expect("some instructions");
        let rebuilt = messages_from_input(
            Some(&instructions),
            &json!([{"role": "user", "content": "hi"}]),
        )
        .expect("converts");
        assert_eq!(
            rebuilt.iter().filter(|m| m["role"] == "system").count(),
            1,
            "still one system turn: {rebuilt:?}"
        );
        let sys = rebuilt[0]["content"].as_str().expect("string content");
        assert!(
            sys.contains("```map"),
            "the block survived the rebuild: {sys}"
        );
        assert!(
            sys.ends_with("You are terse."),
            "and the caller still owns the tail: {sys}"
        );
    }

    #[test]
    fn server_instructions_create_a_system_prompt_when_there_is_none() {
        let mut msgs = vec![json!({"role": "user", "content": "hi"})];
        apply_server_instructions(&mut msgs, &["Use artifacts.".to_string()]);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "Use artifacts.");
        assert_eq!(msgs[1]["role"], "user", "the user turn must stay after it");
    }

    #[test]
    fn several_servers_are_joined_and_none_is_dropped() {
        let mut msgs = vec![json!({"role": "user", "content": "hi"})];
        apply_server_instructions(&mut msgs, &["First.".to_string(), "Second.".to_string()]);
        let sys = msgs[0]["content"].as_str().unwrap();
        assert!(sys.contains("First.") && sys.contains("Second."), "{sys}");
    }

    #[test]
    fn no_instructions_changes_nothing() {
        let before = vec![json!({"role": "user", "content": "hi"})];
        let mut msgs = before.clone();
        apply_server_instructions(&mut msgs, &[]);
        assert_eq!(
            msgs, before,
            "a server with no instructions must not add a system turn"
        );
    }

    #[test]
    fn image_bearing_content_arrays_pass_through() {
        let input = json!([{
            "type": "message", "role": "user",
            "content": [
                {"type": "input_text", "text": "What is this?"},
                {"type": "input_image", "image_url": "data:image/bmp;base64,AAAA"}
            ]
        }]);
        let msgs = messages_from_input(None, &input).unwrap();
        // array kept verbatim so the template renders the image_pad slot
        assert!(msgs[0]["content"].is_array(), "{:?}", msgs[0]);
        let urls = crate::chat::find_images(&msgs).unwrap();
        assert_eq!(
            urls.iter().map(|r| r.url.as_ref()).collect::<Vec<_>>(),
            ["data:image/bmp;base64,AAAA"]
        );
    }

    #[test]
    fn text_only_content_arrays_flatten() {
        let input = json!([{
            "type": "message", "role": "user",
            "content": [{"type": "input_text", "text": "hello "},
                        {"type": "input_text", "text": "world"}]
        }]);
        let msgs = messages_from_input(None, &input).unwrap();
        assert_eq!(msgs[0]["content"], "hello world");
    }

    /// Echoing the whole output array back is the documented multi-turn pattern
    /// for Responses, and that array contains `reasoning` items - so refusing
    /// them 400s a conforming client. It used to.
    #[test]
    fn a_reasoning_item_rides_on_the_assistant_message_it_precedes() {
        let input = json!([
            {"type": "message", "role": "user", "content": "q"},
            {"type": "reasoning", "id": "rs_1", "summary": [],
             "content": [{"type": "reasoning_text", "text": "let me think"}]},
            {"type": "message", "role": "assistant", "content": "a"},
            {"type": "message", "role": "user", "content": "q2"},
        ]);
        let msgs = messages_from_input(None, &input).unwrap();
        // three turns, not four: the reasoning is a FIELD, never a turn of its own
        assert_eq!(msgs.len(), 3, "reasoning must not become its own message");
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[1]["reasoning_content"], "let me think");
        // and it attaches to the message that FOLLOWS it, not the one before
        assert!(msgs[0].get("reasoning_content").is_none());
        assert!(msgs[2].get("reasoning_content").is_none());
    }

    /// A provider that withholds the chain and ships only a summary still has
    /// something to say; take it rather than drop the item silently.
    #[test]
    fn a_summary_only_reasoning_item_still_carries_its_text() {
        let input = json!([
            {"type": "reasoning", "id": "rs_1",
             "summary": [{"type": "summary_text", "text": "considered options"}]},
            {"type": "message", "role": "assistant", "content": "a"},
        ]);
        let msgs = messages_from_input(None, &input).unwrap();
        assert_eq!(msgs[0]["reasoning_content"], "considered options");
    }

    /// An empty one attaches nothing rather than an empty string, so a template
    /// that tests `is string` does not render a hollow think block for it.
    #[test]
    fn an_empty_reasoning_item_attaches_nothing() {
        let input = json!([
            {"type": "reasoning", "id": "rs_1", "summary": [], "content": []},
            {"type": "message", "role": "assistant", "content": "a"},
        ]);
        let msgs = messages_from_input(None, &input).unwrap();
        assert!(msgs[0].get("reasoning_content").is_none());
    }

    #[test]
    fn truncation_drops_whole_turns_from_the_front() {
        let mut msgs = vec![
            json!({"role": "system", "content": "sys"}),
            json!({"role": "user", "content": "q1"}),
            json!({"role": "assistant", "content": "a1"}),
            json!({"role": "user", "content": "q2"}),
            json!({"role": "assistant", "content": "a2"}),
            json!({"role": "user", "content": "pending"}),
        ];
        assert_eq!(drop_leading_turn(&mut msgs), 2, "q1 + a1 leave together");
        assert_eq!(
            msgs[0]["content"], "sys",
            "the system turn is never dropped"
        );
        assert_eq!(msgs[1]["content"], "q2");
        assert_eq!(drop_leading_turn(&mut msgs), 2);
        // only the pending turn is left: nothing droppable, the caller 400s
        assert_eq!(drop_leading_turn(&mut msgs), 0);
        assert_eq!(msgs.len(), 2);
    }

    #[test]
    fn truncation_never_eats_into_the_pending_turn() {
        // an agent-loop round: the pending question, then tool round-trips.
        // Dropping stops at the pending user message - draining into it would
        // leave the model a dangling tool result with no request.
        let mut msgs = vec![
            json!({"role": "user", "content": "old"}),
            json!({"role": "assistant", "content": "answer"}),
            json!({"role": "user", "content": "pending"}),
            json!({"role": "assistant", "tool_calls": [{"id": "c1"}]}),
            json!({"role": "tool", "content": "result", "tool_call_id": "c1"}),
        ];
        assert_eq!(
            drop_leading_turn(&mut msgs),
            2,
            "the finished exchange only"
        );
        assert_eq!(msgs[0]["content"], "pending");
        assert_eq!(
            drop_leading_turn(&mut msgs),
            0,
            "the turn in flight is untouchable"
        );
        assert_eq!(msgs.len(), 3);
    }
}
