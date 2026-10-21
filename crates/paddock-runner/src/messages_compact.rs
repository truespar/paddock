//! Server-side compaction for /v1/messages (`compact_20260112`): the
//! summarization pass, the agent loops' round-0 pass and the plain
//! two-iteration orchestration. A child of messages.rs (it shares the
//! parent's render/stream helpers); the pure halves - trigger math, span
//! split, resend rewrite - live in crate::context_management.

use super::*;

/// One summarization over the compact span of `messages`, shared by the
/// plain two-iteration path and the agent loops' round-0 pass. Several
/// generations when the span is summarized in parts; the numbers are summed.
struct SummaryPass {
    /// None = failed compaction (the model produced nothing usable).
    summary: Option<String>,
    /// This pass's `usage.iterations` entry, SDK shape.
    usage: Value,
    input_tokens: usize,
    output_tokens: usize,
    cached: usize,
}

fn summary_pass(summary: Option<String>, rows: usize, cached: usize, out: usize) -> SummaryPass {
    let mut usage = anth_usage(rows, cached, out);
    usage["type"] = json!("compaction");
    SummaryPass {
        summary,
        usage,
        input_tokens: rows,
        output_tokens: out,
        cached,
    }
}

/// A rendered prompt: ids, engine prompt, think block open, picture chunks.
type Rendered = (Vec<u32>, Vec<u32>, bool, Option<Vec<MmChunk>>);

/// Summarize the compact span: one pass over [same system, same tools,
/// span..., instructions] when it fits the window beside the summary's own
/// output, else a rolling pass over parts (context_management::
/// anth_span_parts) - part 1 is the span's prefix, every later part opens
/// with the user's own words and the summary so far, and updates it.
/// Same-prefix-as-the-conversation is the point of part 1 (or the single
/// pass): its prefill rides the radix cache of the very conversation being
/// compacted. Err is a ready-to-return error response - including a span
/// that cannot be shown to the summarizer even in parts, where the loud
/// over-window error stands (the Anthropic dialect has no fail-open
/// backstop; its Responses twin's in-create path does).
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
    use crate::context_management::{
        OA_PART_FIRST, anth_elide, anth_part_context, anth_part_next, anth_span_parts,
        anth_user_texts,
    };
    let span = &messages[..crate::context_management::compact_tail_start(messages)];
    let custom = instructions;
    let instructions =
        instructions.unwrap_or(crate::context_management::DEFAULT_COMPACT_INSTRUCTIONS);
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
    // tool removals from mid-conversation system messages too, folded over the
    // whole conversation exactly as the live render folds them - same reason
    let mut tools = tools;
    let controls = match crate::messages_system::system_controls(messages, req.tools.as_deref()) {
        Ok(c) => c,
        Err(e) => return Err(bad(e)),
    };
    crate::messages_system::drop_removed_tools(&mut tools, &controls.removed);
    // The live request's own thinking config and effort, not thinking off:
    // the template is NOT suffix-only there - Qwen3.8 writes its reasoning-
    // effort instruction into the SYSTEM block whenever thinking is on, so a
    // thinking-off render diverged a few tokens in and re-prefilled the whole
    // span (146K tokens, ~160 s, on an agent loop). An open think block is
    // closed at once by a one-token budget (anth_run_pass), so the tokens
    // still go to the summary itself.
    let effort_owned = match parse_output_config(req.output_config.as_ref()) {
        Ok((e, _)) => e,
        Err(e) => return Err(bad(e)),
    };
    let effort = controls.effort.as_deref().or(effort_owned.as_deref());
    let render = |msgs: &[Value], tools: Option<&[Value]>| -> Result<Rendered, Response> {
        render_prompt(
            model,
            req.system.as_ref(),
            msgs,
            tools,
            req.thinking.as_ref(),
            None,
            effort,
        )
        .map(|(ids, engine, open, mm, _)| (ids, engine, open, mm))
        .map_err(bad)
    };
    let gate = |r: &Rendered, window: usize| {
        crate::chat::context_gate(model, r.1.len(), r.3.as_deref(), window)
    };
    // bounded: a summary, not an essay (the pass is billed via usage.iterations)
    let max_out = state.max_output_ceiling.map_or(4096, |c| c.min(4096));
    // the summary's own tokens need room beside its prompt - its cap, but
    // never more than an eighth of the window (a small window must still be
    // able to compact)
    let reserve = max_out.min(state.max_ctx / 8).max(256);
    let budget = state.max_ctx.saturating_sub(reserve + 64);
    let mut msgs1 = span.to_vec();
    msgs1.push(json!({"role": "user", "content": instructions}));
    let whole = render(&msgs1, tools.as_deref())?;
    // a span that cannot be summarized even in parts: the loud over-window
    // error stands where the whole render is past the window, else the
    // compaction failed (null block) and the request proceeds uncompacted
    let give_up = |whole: &Rendered, rows, cached, out| match gate(whole, state.max_ctx) {
        Some(e) => Err(engine_err(&e)),
        None => Ok(summary_pass(None, rows, cached, out)),
    };
    if gate(&whole, budget).is_none() {
        let (s, rows, cached, out) =
            anth_run_pass(state, model, req, whole, max_out, scope).await?;
        return Ok(summary_pass(s, rows, cached, out));
    }

    // Parts. Cost per message from the full render's own tokens-per-char, so
    // dense records and prose are both priced by this tokenizer; every part
    // is still rendered and checked before it runs.
    let chars = |m: &Value| m.to_string().len();
    let total_chars = span.iter().map(chars).sum::<usize>().max(1);
    let ratio = whole.1.len() as f64 / total_chars as f64;
    let cost = |m: &Value| (chars(m) as f64 * ratio * 1.1) as usize + 8;
    // later parts carry the user's words and a summary so far beside the
    // system prompt and instructions; tools ride part 1 only
    let probe = render(
        &[json!({"role": "user", "content": anth_part_next(true)})],
        None,
    )?;
    // the user's words ride every later part, newest first within a quarter
    // of the budget: a long one (a pasted document) is still in the part
    // that holds it, and the summary carries it from there on
    let mut users = Vec::new();
    let mut users_cost = 0usize;
    for u in anth_user_texts(span).into_iter().rev() {
        let c = cost(&json!({"role": "user", "content": u}));
        if users_cost + c <= budget / 4 {
            users_cost += c;
            users.push(u);
        }
    }
    users.reverse();
    // the summary so far is at most one pass's output
    let overhead = probe.1.len() + users_cost + reserve + 256;
    let Some(part_budget) = budget.checked_sub(overhead).filter(|&b| b >= 1024) else {
        // the kept words alone fill the window: nothing a summary can do
        return give_up(&whole, 0, 0, 0);
    };
    let mut queue: std::collections::VecDeque<Vec<Value>> =
        anth_span_parts(span, &cost, part_budget).into();
    tracing::info!(
        parts = queue.len(),
        span_tokens = whole.1.len(),
        window = state.max_ctx,
        "compaction: the span outgrows the summarizer's window - summarizing it in parts"
    );
    let (mut rows, mut cached, mut out) = (0usize, 0usize, 0usize);
    let mut so_far: Option<String> = None;
    let mut tries = 0;
    while let Some(part) = queue.pop_front() {
        tries += 1;
        if tries > 64 {
            queue.push_front(part);
            break;
        }
        let first = so_far.is_none();
        let last = queue.is_empty();
        let mut msgs = match &so_far {
            None => part.clone(),
            Some(s) => {
                // a user message leading the part joins the context (one turn)
                let lead = part.first().filter(|m| m["role"] == "user");
                let mut v = vec![anth_part_context(&users, s, lead)];
                v.extend_from_slice(&part[usize::from(lead.is_some())..]);
                v
            }
        };
        let mut instruction = match (first, last) {
            (true, true) => instructions.to_owned(),
            (true, false) => OA_PART_FIRST.to_owned(),
            (false, last) => anth_part_next(last),
        };
        if let Some(c) = custom.filter(|_| last && !first) {
            // the caller's own instructions shape the summary that comes out
            instruction.push_str("\n\nWrite the summary to these instructions:\n\n");
            instruction.push_str(c);
        }
        msgs.push(json!({"role": "user", "content": instruction}));
        let r = render(&msgs, if first { tools.as_deref() } else { None })?;
        if gate(&r, budget).is_some() {
            // the estimate ran short: cut this part finer (or, when it cuts
            // no finer, its text down) and go again
            let half = part.iter().map(cost).sum::<usize>() / 2;
            let mut finer = anth_span_parts(&part, &cost, half.max(1));
            if finer == [part.clone()] {
                finer = vec![part.iter().map(|m| anth_elide(m, 0.6)).collect()];
            }
            for p in finer.into_iter().rev() {
                queue.push_front(p);
            }
            continue;
        }
        let (s, p, c, o) = anth_run_pass(state, model, req, r, max_out, scope).await?;
        rows += p;
        cached += c;
        out += o;
        match s {
            Some(s) => so_far = Some(s),
            None => return Ok(summary_pass(None, rows, cached, out)),
        }
    }
    if !queue.is_empty() {
        return give_up(&whole, rows, cached, out);
    }
    Ok(summary_pass(so_far, rows, cached, out))
}

/// One summarization generation over a rendered prompt (which ends with the
/// instruction turn). Returns (summary, prompt rows, cached, output tokens);
/// a None summary is a failed compaction.
#[allow(clippy::result_large_err)]
async fn anth_run_pass(
    state: &Arc<AppState>,
    model: &ServingModel,
    req: &MessagesRequest,
    (ids, engine, think_open, mm): Rendered,
    max_tokens: usize,
    scope: &crate::events::EventScope,
) -> Result<(Option<String>, usize, usize, usize), Response> {
    let close_think = crate::reasoning::summary_think_close(model, think_open);
    // greedy-class: a deterministic summary, no sampling surprises. The
    // truncation knobs still come from the model's election - at temperature
    // 0 they cannot change the argmax, so this is about staying consistent
    // with the rest of the server rather than about the draw.
    let dflt = state.sampling.resolve(true);
    let sampler = SamplingParams {
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
        no_repeat_ngram: (0, 0, false),
        reasoning: crate::reasoning::stop_policy(model, think_open),
    };
    let (tx, mut rx) = unbounded_channel();
    let gen1 = GenRequest {
        prompt: engine,
        max_tokens,
        sampler,
        stop_tokens: model.stop_tokens.clone(),
        events: tx,
        mm_chunks: mm,
        constraint: instantiate_constraint(
            &ConstraintSpec::None,
            GateSpec::Immediate,
            model,
            close_think.as_ref(),
        ),
        logprobs: None,
        submitted: None, // stamped by Engine::submit
        canvas_read: None,
        user_turn: false,
        background: false,
    };
    if let Err(e) = model.engine.submit(gen1) {
        return Err(err(StatusCode::INTERNAL_SERVER_ERROR, "api_error", e));
    }
    let mut out_ids: Vec<u32> = Vec::new();
    let mut cached = 0usize;
    let mut rows = ids.len();
    let mut terminal_tokens = 0;
    while let Some(evt) = next_event(&mut rx).await {
        match evt {
            TokenEvent::Prefilled { cached: c, rows: r } => {
                cached = c as usize;
                rows = rows.max(r as usize);
            }
            TokenEvent::Token { id: t, .. } => out_ids.push(t),
            TokenEvent::Done(_, stats) => {
                terminal_tokens = stats.terminal_tokens();
                scope.phases(&stats);
                break;
            }
            TokenEvent::Error(e) => return Err(engine_err(&e)),
        }
    }
    let raw = model.tokenizer.decode(&out_ids, false).unwrap_or_default();
    let parsed = parse(model.dialect, &raw, think_open, None);
    // None = failed compaction: the block reports null content (SDK pin) and
    // the request proceeds UNCOMPACTED - if that overflows, the loud
    // over-window error stands; never a silent trim
    let summary = parsed
        .content
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned);
    Ok((summary, rows, cached, out_ids.len() + terminal_tokens))
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
pub(super) async fn precompact_agent(
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
pub(super) fn drop_compact_edit(cm: &mut Option<Value>) {
    if let Some(Value::Object(o)) = cm.as_mut()
        && let Some(Value::Array(edits)) = o.get_mut("edits")
    {
        edits.retain(|e| e.get("type").and_then(Value::as_str) != Some("compact_20260112"));
    }
}

/// `usage.iterations` for an agent turn that compacted: the summarization
/// pass, then the final message. Top-level usage stays the message's own, per
/// spec - iterations is where the extra generation is billed.
pub(super) fn agent_iterations(
    c: &CompactionOut,
    model_id: &str,
    prompt_len: usize,
    out_tokens: usize,
    cached: usize,
) -> Value {
    json!([
        c.usage,
        anth_iteration("message", model_id, prompt_len, cached, out_tokens)
    ])
}

/// What the compaction orchestration produced.
enum Compacted {
    /// pause_after_compaction: the block is the whole response.
    Paused(SummaryPass),
    /// The real generation over the compacted turn, submitted.
    Run(Box<Ctx>, UnboundedReceiver<TokenEvent>),
}

/// The compact_20260112 orchestration. Two generations on the same engine,
/// sequentially: iteration 1 summarizes the compact span, iteration 2 is the
/// real request over [framed summary + pending turn] - or, with
/// pause_after_compaction, the response stops at the block with stop_reason
/// "compaction". Cache economics are the design: iteration 1's prefill rides
/// the radix cache of the very conversation being compacted, and the
/// compacted conversation equals what a resend rewrites to, so every later
/// turn is a radix hit.
///
/// A stream opens before iteration 1 and the summarization runs inside it,
/// ping keep-alives flowing meanwhile (anth_sse): a pass over a long span
/// takes minutes (a cold 290K-token span summarized in parts sat 450 s
/// before its first byte on the Responses twin) and agent clients run idle
/// watchdogs (Codex CLI's is 300 s). Past that point an error is the
/// stream's `error` event; a non-streaming request still answers with a
/// clean HTTP error.
pub(super) async fn run_compacting(
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
    let id = format!("msg_{}", uuid::Uuid::new_v4().simple());
    let model_id = model.id.clone();
    if !req.stream {
        return match compact(&state, &req, &plan, &cm_report, omit_thinking, &scope, &id).await {
            Ok(Compacted::Paused(pass)) => {
                paused_finish(&scope, &pass);
                let mut usage = anth_usage(pass.input_tokens, pass.cached, pass.output_tokens);
                usage["iterations"] = json!([pass.usage]);
                let mut body = json!({
                    "id": id, "type": "message", "role": "assistant", "model": model_id,
                    "content": [compaction_block(&pass.summary)],
                    "stop_reason": "compaction", "stop_sequence": null,
                    "usage": usage,
                });
                if let Some(cm) = &cm_report {
                    body["context_management"] = cm.clone();
                }
                Json(body).into_response()
            }
            Ok(Compacted::Run(ctx, rx)) => collect_response(*ctx, rx).await,
            Err(r) => r,
        };
    }
    // message_start can only carry the uncompacted prompt's tokenized
    // length; message_delta restates the real usage, as every stream does
    let start = message_start(&id, &model_id, plan.prompt_len);
    let sse = stream! {
        yield start;
        yield ev("ping", json!({"type": "ping"}));
        match compact(&state, &req, &plan, &cm_report, omit_thinking, &scope, &id).await {
            Ok(Compacted::Paused(pass)) => {
                paused_finish(&scope, &pass);
                yield ev("content_block_start", json!({
                    "type": "content_block_start", "index": 0,
                    "content_block": {"type": "compaction", "content": null}}));
                yield ev("content_block_delta", json!({
                    "type": "content_block_delta", "index": 0,
                    "delta": {"type": "compaction_delta", "content": pass.summary}}));
                yield ev("content_block_stop", json!({"type": "content_block_stop", "index": 0}));
                let mut usage = anth_usage(pass.input_tokens, pass.cached, pass.output_tokens);
                usage["iterations"] = json!([pass.usage]);
                let mut delta_ev = json!({
                    "type": "message_delta",
                    "delta": {"stop_reason": "compaction", "stop_sequence": null},
                    "usage": usage,
                });
                if let Some(cm) = &cm_report {
                    delta_ev["context_management"] = cm.clone();
                }
                yield ev("message_delta", delta_ev);
                yield ev("message_stop", json!({"type": "message_stop"}));
            }
            Ok(Compacted::Run(ctx, rx)) => {
                let body = stream_body(*ctx, rx);
                futures::pin_mut!(body);
                while let Some(e) = futures::StreamExt::next(&mut body).await {
                    yield e;
                }
            }
            Err(resp) => yield stream_error(resp).await,
        }
    };
    anth_sse(sse)
}

/// pause_after_compaction's scope bookkeeping: the pass is the only
/// generation that ran.
fn paused_finish(scope: &crate::events::EventScope, pass: &SummaryPass) {
    scope.usage(pass.input_tokens, pass.output_tokens);
    scope.cached(pass.cached);
    scope.finish("compaction");
}

/// A ready-to-return error response as the stream's `error` event (the
/// same `{"type": "error", "error": {...}}` body), for a failure after the
/// stream has opened.
async fn stream_error(resp: Response) -> Result<Event, std::convert::Infallible> {
    let status = resp.status();
    let body = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap_or_default();
    let v: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
    let e = v.get("error").cloned().unwrap_or_else(|| {
        let kind = if status.is_server_error() {
            "api_error"
        } else {
            "invalid_request_error"
        };
        json!({"type": kind, "message": "the request failed while compacting"})
    });
    ev("error", json!({"type": "error", "error": e}))
}

/// Iteration 1, then (unless paused) iteration 2 submitted; `id` is the
/// message id the response already carries or will carry.
#[allow(clippy::result_large_err)]
async fn compact(
    state: &Arc<AppState>,
    req: &MessagesRequest,
    plan: &CompactPlan,
    cm_report: &Option<Value>,
    omit_thinking: bool,
    scope: &crate::events::EventScope,
    id: &str,
) -> Result<Compacted, Response> {
    let Some(model) = state.serving.as_ref() else {
        return Err(err(
            StatusCode::SERVICE_UNAVAILABLE,
            "overloaded_error",
            "no model is loaded",
        ));
    };
    let pass = anth_summary_pass(
        state,
        model,
        req,
        &plan.messages,
        plan.instructions.as_deref(),
        scope,
    )
    .await?;
    if plan.pause {
        return Ok(Compacted::Paused(pass));
    }
    let msgs2 = match &pass.summary {
        Some(s) => crate::context_management::compacted_messages(&plan.messages, s),
        None => plan.messages.clone(),
    };
    let mut req2 = req.clone();
    req2.messages = msgs2;
    req2.context_management = None;
    let prepared = prepare(model, &req2, state.max_output_ceiling, &state.sampling).map_err(bad)?;
    if let Some(e) = crate::chat::context_gate(
        model,
        prepared.engine_prompt.len(),
        prepared.mm_chunks.as_deref(),
        state.max_ctx,
    ) {
        return Err(engine_err(&e));
    }
    let ctx = Ctx {
        id: id.to_owned(),
        model_id: model.id.clone(),
        tokenizer: model.tokenizer.clone(),
        prompt_len: prepared.prompt_ids.len(),
        dialect: model.dialect,
        thinking_open: prepared.thinking_open,
        hints: prepared.hints.clone(),
        single_tool_call: prepared.single_tool_call,
        stop_strings: prepared.stop_strings.clone(),
        omit_thinking,
        context_management: cm_report.clone(),
        compaction: Some(CompactionOut {
            summary: pass.summary,
            usage: pass.usage,
        }),
        ocr: prepared.ocr.clone(),
        scope: scope.clone(),
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
        canvas_read: None,
        user_turn: false,
        background: false,
    };
    if let Err(e) = model.engine.submit(gen2) {
        return Err(err(StatusCode::INTERNAL_SERVER_ERROR, "api_error", e));
    }
    Ok(Compacted::Run(Box::new(ctx), rx))
}
