//! Responses compaction executors: in-create compaction, the agent loop's
//! round-0 pass, `compaction_trigger`, the standalone `/v1/responses/compact`
//! endpoint, and the summarization pass they share (one pass, or a rolling
//! pass over parts when the span outgrows the window). A child of
//! responses.rs (it shares the parent's prepare/stream helpers); the pure
//! halves - plans, the v2 state, the part cuts - live in
//! crate::context_management.

use super::*;

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
/// output constraint, a summary-sized output cap. Thinking stays as the
/// conversation has it - the template is not suffix-only there (Qwen3.8
/// writes its reasoning-effort instruction into the system block), so a
/// thinking-off render re-prefilled the whole span - and `run_summary`
/// closes an open think block after one token instead. The caller's own
/// thinking budget is dropped for that reason. Tools stay in the render for
/// the same prefix-match reason - unless the request hid them with
/// tool_choice "none"; a forced tool_choice must not force the SUMMARY, so
/// it relaxes to auto.
fn greedy_for_summary(mut r: ResponsesRequest) -> ResponsesRequest {
    r.temperature = Some(0.0);
    r.presence_penalty = Some(0.0);
    r.frequency_penalty = Some(0.0);
    r.text = None;
    r.max_output_tokens = Some(4096);
    if r.tool_choice
        .as_ref()
        .is_some_and(|tc| tc.as_str() != Some("none"))
    {
        r.tool_choice = None;
    }
    if let Some(o) = r.reasoning.as_mut().and_then(Value::as_object_mut) {
        o.remove("max_tokens");
    }
    r.stream = false;
    r.context_management = None;
    r
}

/// The span a compaction summarizes, in the two shapes `summarize` needs.
struct SummarySpan<'a> {
    /// the whole span as chat messages, built exactly as the live request
    /// renders it (the single pass; the instruction turn is appended here)
    first: Vec<Value>,
    /// the same span as input items (cut into parts if it outgrows the window)
    items: &'a [Value],
    /// the user's own messages, verbatim, for the later parts' context
    users: Vec<Value>,
    /// the single-pass instruction (OA_COMPACT_INSTRUCTIONS / _ALL_)
    instructions: &'static str,
    /// a raw tool round stays after the summary (the last part says so)
    keeps_tail: bool,
}

/// The user messages a state keeps verbatim, the current request included.
fn kept_users(st: &crate::context_management::OaState) -> Vec<Value> {
    let lead = st
        .tail
        .iter()
        .take_while(|it| crate::context_management::oa_user_message(it));
    st.users.iter().chain(lead).cloned().collect()
}

/// Summarize a span: one pass when it fits the window beside the summary's
/// own output budget, else a rolling pass over parts (see
/// context_management::oa_span_parts) - part 1 is the span's prefix, so its
/// prefill still rides the radix cache of the conversation being compacted;
/// every later part opens with the user's messages verbatim and the summary
/// so far, and updates it. Returns (summary, prompt rows, cached, output
/// tokens) summed over the passes; Err is a ready-to-return error response.
/// A None summary is the failed-compaction case. `lenient_gate`: a span
/// that cannot be summarized even in parts reads as a FAILED compaction
/// (None) instead of an error - in-create compaction is opportunistic and
/// the request must fall through to `truncation:"auto"` when it's armed; the
/// trigger and the standalone endpoint were ASKED to compact, so for them
/// the loud over-window error stands.
// Err is axum's own Response by design: a ready-to-return body, not a boxed error
#[allow(clippy::result_large_err, clippy::too_many_arguments)]
async fn summarize(
    state: &Arc<AppState>,
    model: &ServingModel,
    req1: &ResponsesRequest,
    span: SummarySpan<'_>,
    // The agent loop's discovered MCP tools: the span render must carry the
    // same tools as the live conversation or its prefix diverges and the
    // summarization prefill loses the radix hit that makes it nearly free.
    extra_tools: &[Value],
    scope: &crate::events::EventScope,
    lenient_gate: bool,
) -> Result<(Option<String>, usize, usize, usize), Response> {
    use crate::context_management::{OA_PART_FIRST, oa_part_context, oa_part_next};
    let bad = |e: String| err(StatusCode::BAD_REQUEST, "invalid_request_error", e);
    let prep = |r: &ResponsesRequest, msgs: &[Value], tools: &[Value]| {
        prepare(model, r, msgs, tools, state.output_caps(), &state.sampling).map_err(bad)
    };
    let fits = |p: &Prepared, budget: usize| {
        crate::chat::context_gate(model, p.engine_prompt.len(), p.mm_chunks.as_deref(), budget)
            .is_none()
    };
    let too_long = |p: &Prepared| {
        let e = crate::chat::context_gate(
            model,
            p.engine_prompt.len(),
            p.mm_chunks.as_deref(),
            state.max_ctx,
        );
        e.map(|e| crate::chat::engine_err(&e))
    };
    let mut msgs1 = span.first;
    msgs1.push(json!({"role": "user", "content": span.instructions}));
    let whole = prep(req1, &msgs1, extra_tools)?;
    // the summary's own tokens need room beside its prompt - its cap, but
    // never more than an eighth of the window (a small window must still be
    // able to compact)
    let reserve = whole.max_tokens.min(state.max_ctx / 8).max(256);
    let budget = state.max_ctx.saturating_sub(reserve + 64);
    if fits(&whole, budget) {
        return run_summary(model, whole, scope).await;
    }

    // Parts. Cost per item from the full render's own tokens-per-char, so
    // dense records and prose are both priced by this tokenizer; every part
    // is still rendered and checked before it runs.
    let chars = |it: &Value| it.to_string().len();
    let total_chars = span.items.iter().map(chars).sum::<usize>().max(1);
    let ratio = whole.engine_prompt.len() as f64 / total_chars as f64;
    let cost = |it: &Value| (chars(it) as f64 * ratio * 1.1) as usize + 8;
    // later parts carry the user's messages and a summary so far beside
    // the system turn and instructions; tools only ride part 1
    let mut rest_req = req1.clone();
    rest_req.tools = None;
    rest_req.tool_choice = None;
    let probe = vec![json!({"role": "user", "content": oa_part_next(true, span.keeps_tail)})];
    let base = prep(
        &rest_req,
        &messages_from_input(rest_req.instructions.as_deref(), &Value::Array(probe))
            .map_err(bad)?,
        &[],
    )?;
    // the user's messages ride every later part, newest first within a
    // quarter of the budget: a long one (a pasted document) is still in the
    // part that holds it, and the summary carries it from there on
    let mut ctx_users = Vec::new();
    let mut users_cost = 0usize;
    for u in span.users.iter().rev() {
        let c = cost(u);
        if users_cost + c <= budget / 4 {
            users_cost += c;
            ctx_users.push(u.clone());
        }
    }
    ctx_users.reverse();
    let overhead = base.engine_prompt.len() + users_cost + reserve + 256;
    let Some(part_budget) = budget.checked_sub(overhead).filter(|&b| b >= 1024) else {
        // the kept messages alone fill the window: nothing a summary can do
        return match too_long(&whole) {
            Some(e) if !lenient_gate => Err(e),
            _ => Ok((None, 0, 0, 0)),
        };
    };
    let mut queue: std::collections::VecDeque<Vec<Value>> =
        crate::context_management::oa_span_parts(span.items, &cost, part_budget).into();
    tracing::info!(
        parts = queue.len(),
        span_tokens = whole.engine_prompt.len(),
        window = state.max_ctx,
        "compaction: the span outgrows the summarizer's window - summarizing it in parts"
    );
    let (mut rows, mut cached, mut out) = (0usize, 0usize, 0usize);
    let mut so_far: Option<String> = None;
    let mut tries = 0;
    while let Some(part) = queue.pop_front() {
        tries += 1;
        if tries > 64 {
            break;
        }
        let first = so_far.is_none();
        let last = queue.is_empty();
        let items: Vec<Value> = match &so_far {
            None => part.clone(),
            Some(s) => {
                let lead = part
                    .iter()
                    .take_while(|it| crate::context_management::oa_user_message(it))
                    .count();
                let mut v = vec![oa_part_context(&ctx_users, s, &part[..lead])];
                v.extend_from_slice(&part[lead..]);
                v
            }
        };
        let instruction = match (first, last) {
            (true, true) => span.instructions.to_owned(),
            (true, false) => OA_PART_FIRST.to_owned(),
            (false, _) => oa_part_next(last, span.keeps_tail),
        };
        let mut msgs =
            messages_from_input(req1.instructions.as_deref(), &Value::Array(items)).map_err(bad)?;
        msgs.push(json!({"role": "user", "content": instruction}));
        let (r, tools) = if first {
            (req1, extra_tools)
        } else {
            (&rest_req, &[][..])
        };
        let prepared = prep(r, &msgs, tools)?;
        if !fits(&prepared, budget) {
            // the estimate ran short: cut this part finer (or, one item
            // left, cut the item down) and go again
            let finer = if part.len() > 1 {
                let half = part.iter().map(cost).sum::<usize>() / 2;
                crate::context_management::oa_span_parts(&part, &cost, half.max(1))
            } else {
                vec![vec![crate::context_management::oa_elide(&part[0], 0.6)]]
            };
            for p in finer.into_iter().rev() {
                queue.push_front(p);
            }
            continue;
        }
        let (s, p, c, o) = run_summary(model, prepared, scope).await?;
        rows += p;
        cached += c;
        out += o;
        match s {
            Some(s) => so_far = Some(s),
            None => return Ok((None, rows, cached, out)),
        }
    }
    if !queue.is_empty() {
        return match too_long(&whole) {
            Some(e) if !lenient_gate => Err(e),
            _ => Ok((None, rows, cached, out)),
        };
    }
    Ok((so_far, rows, cached, out))
}

/// One summarization generation over a prepared prompt (which ends with the
/// summarization instruction turn). An open think block is closed after one
/// token (see reasoning::summary_think_close), so the output budget goes to
/// the summary. Returns (summary, prompt rows, cached, output tokens).
#[allow(clippy::result_large_err)]
async fn run_summary(
    model: &ServingModel,
    prepared: Prepared,
    scope: &crate::events::EventScope,
) -> Result<(Option<String>, usize, usize, usize), Response> {
    let close_think = crate::reasoning::summary_think_close(model, prepared.thinking_open);
    let constraint = instantiate_constraint(
        &prepared.constraint_spec,
        prepared.gate,
        model,
        close_think.as_ref(),
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
        canvas_read: None,
        user_turn: false,
        background: false,
    };
    if let Err(e) = model.engine.submit(gen1) {
        return Err(err(StatusCode::INTERNAL_SERVER_ERROR, "internal_error", e));
    }
    let mut ids: Vec<u32> = Vec::new();
    let mut cached = 0usize;
    let mut p_len = prepared.prompt_ids.len();
    let mut terminal_tokens = 0;
    while let Some(evt) = rx.recv().await {
        match evt {
            TokenEvent::Prefilled { cached: c, rows } => {
                cached = c as usize;
                p_len = p_len.max(rows as usize);
            }
            TokenEvent::Token { id: t, .. } => ids.push(t),
            TokenEvent::Done(_, stats) => {
                terminal_tokens = stats.terminal_tokens();
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
    Ok((summary, p_len, cached, ids.len() + terminal_tokens))
}

/// Round-0 context management for the server-run agent loop (MCP / web
/// tools). The compaction orchestration runs once, here, on the request's
/// own prompt - the same measurement point a non-agent request uses - and
/// never again inside the loop: the item leads the output at output[0], the
/// single-shot contract, and stands for the window as the request brought
/// it (the v2 state, see context_management::OaState). The loop's own
/// rounds come after it as output items the client appends, so a later
/// request compacts them like any other round. Inside one request,
/// `truncation:"auto"` backstops the growing prompt (applied every round).
/// `compact`: the threshold plus the window to plan over (the last state
/// and the items after it). Returns the compaction item to lead the output
/// with. None = no threshold, trigger unmet, nothing to summarize, or a
/// failed summarization - the absent item is the OpenAI dialect's
/// "compaction did not happen" signal. Err = a ready-to-return error
/// response.
// Err is axum's own Response by design: a ready-to-return body, not a boxed error
#[allow(clippy::too_many_arguments, clippy::result_large_err)]
pub(super) async fn precompact_agent(
    state: &Arc<AppState>,
    model: &ServingModel,
    req: &ResponsesRequest,
    extra_tools: &[Value],
    messages: &mut Vec<Value>,
    compact: Option<(u64, (Option<&crate::context_management::OaState>, &[Value]))>,
    scope: &crate::events::EventScope,
) -> Result<Option<Value>, Response> {
    let Some((threshold, (prev, after))) = compact else {
        return Ok(None);
    };
    // one extra render+tokenize, and only for a request that armed the
    // threshold: the loop's round 0 re-prepares over the compacted messages
    let prepared = prepare(
        model,
        req,
        messages,
        extra_tools,
        state.output_caps(),
        &state.sampling,
    )
    .map_err(|e| err(StatusCode::BAD_REQUEST, "invalid_request_error", e))?;
    if (prepared.prompt_ids.len() as u64) < threshold {
        return Ok(None);
    }
    let keep = crate::context_management::OaKeep::for_threshold(threshold);
    let Some(plan) = crate::context_management::oa_plan(prev, after, Some(keep)) else {
        return Ok(None); // nothing besides the kept messages to summarize
    };
    let first = messages_from_input(
        req.instructions.as_deref(),
        &Value::Array(plan.span.clone()),
    )
    .map_err(|e| err(StatusCode::BAD_REQUEST, "invalid_request_error", e))?;
    let span = SummarySpan {
        first,
        items: &plan.span,
        users: kept_users(&plan.state),
        instructions: crate::context_management::OA_COMPACT_INSTRUCTIONS,
        keeps_tail: plan
            .state
            .tail
            .iter()
            .any(|it| !crate::context_management::oa_user_message(it)),
    };
    let req1 = greedy_for_summary(req.clone());
    let (summary, ..) = summarize(state, model, &req1, span, extra_tools, scope, true).await?;
    // failed summary -> the loop runs uncompacted with no item, exactly as the
    // single-shot path does (and truncation "auto", if armed, still catches
    // an over-window prompt inside the loop)
    let Some(s) = summary else { return Ok(None) };
    let st = crate::context_management::OaState {
        summary: s,
        ..plan.state
    };
    let items2 = crate::context_management::oa_render(&st, &[]);
    *messages = messages_from_input(req.instructions.as_deref(), &Value::Array(items2))
        .map_err(|e| err(StatusCode::BAD_REQUEST, "invalid_request_error", e))?;
    Ok(Some(compaction_item(&st.encode())))
}

/// The in-create compaction orchestration (`context_management` threshold
/// met): iteration 1 summarizes the planned span greedily, iteration 2
/// answers over the compacted window (kept messages + framed summary +
/// raw tail). Same cache algebra as the /v1/messages twin: iteration 1
/// renders [same system, same tools, span, instructions] so its prefill
/// rides the radix cache of the conversation being compacted, and
/// iteration 2's item list is exactly what a later resend renders from the
/// compaction item.
/// The official Response has no `usage.iterations` slot, so the pass's own
/// numbers stay off the wire; top-level usage covers the final generation.
pub(super) async fn run_compacting_oa(
    state: Arc<AppState>,
    req: ResponsesRequest,
    plan: crate::context_management::OaPlan,
    scope: crate::events::EventScope,
) -> Response {
    if state.serving.is_none() {
        return err(
            StatusCode::SERVICE_UNAVAILABLE,
            "model_not_loaded",
            "no model is loaded",
        );
    }
    if !req.stream {
        return match compact_and_submit(state, req, plan, scope).await {
            Ok((meta, rx)) => collect_response(meta, rx).await,
            Err(r) => r,
        };
    }
    // Streaming: the 200 and the stream open NOW and the summarization runs
    // inside it, keep-alive comments flowing meanwhile - OpenAI compacts
    // inside the stream too. A pass over a long span takes minutes (a cold
    // 290K-token span summarized in parts sat 450 s before its first byte)
    // and agent clients run idle watchdogs (Codex CLI's is 300 s). Past this
    // point an error is the stream's own `error` event.
    let sse = stream! {
        match compact_and_submit(state, req, plan, scope).await {
            Ok((meta, rx)) => {
                let inner = response_events(meta, rx);
                futures::pin_mut!(inner);
                while let Some(e) = futures::StreamExt::next(&mut inner).await {
                    yield e;
                }
            }
            Err(resp) => yield stream_error(resp).await,
        }
    };
    streaming_response(sse)
}

/// A ready-to-return error response as the Responses stream `error` event
/// (`ResponseErrorEvent`: code, message, param) - for a failure after the
/// stream has opened.
async fn stream_error(resp: Response) -> Result<Event, std::convert::Infallible> {
    let status = resp.status();
    let body = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap_or_default();
    let v: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
    let e = v.get("error").unwrap_or(&v);
    let code = e
        .get("code")
        .and_then(Value::as_str)
        .or_else(|| e.get("type").and_then(Value::as_str))
        .unwrap_or(if status.is_server_error() {
            "server_error"
        } else {
            "invalid_request_error"
        });
    let message = e
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("the request failed while compacting");
    ev(
        "error",
        json!({"type": "error", "code": code, "message": message, "param": null,
               "sequence_number": 0}),
    )
}

/// The in-create compaction itself, up to the submitted generation:
/// iteration 1 summarizes the planned span, iteration 2 is submitted over
/// the compacted window; the caller streams or collects it.
#[allow(clippy::result_large_err)]
async fn compact_and_submit(
    state: Arc<AppState>,
    req: ResponsesRequest,
    plan: crate::context_management::OaPlan,
    scope: crate::events::EventScope,
) -> Result<(Meta, UnboundedReceiver<TokenEvent>), Response> {
    let Some(model) = state.serving.as_ref() else {
        return Err(err(
            StatusCode::SERVICE_UNAVAILABLE,
            "model_not_loaded",
            "no model is loaded",
        ));
    };
    let first = match messages_from_input(
        req.instructions.as_deref(),
        &Value::Array(plan.span.clone()),
    ) {
        Ok(m) => m,
        Err(e) => return Err(err(StatusCode::BAD_REQUEST, "invalid_request_error", e)),
    };
    let span = SummarySpan {
        first,
        items: &plan.span,
        users: kept_users(&plan.state),
        instructions: crate::context_management::OA_COMPACT_INSTRUCTIONS,
        keeps_tail: plan
            .state
            .tail
            .iter()
            .any(|it| !crate::context_management::oa_user_message(it)),
    };
    let req1 = greedy_for_summary(req.clone());
    let (summary, _p1_len, _cached1, _out1) =
        match summarize(&state, model, &req1, span, &[], &scope, true).await {
            Ok(x) => x,
            Err(r) => return Err(r),
        };

    // iteration 2: the real generation over the compacted window. A failed
    // summary proceeds UNCOMPACTED with no compaction item - the OpenAI
    // dialect has no failed-compaction shape, and the absent item is the
    // honest signal (documented pin); if the uncompacted prompt then
    // overflows, the loud gate error stands.
    let compacted = summary.map(|s| crate::context_management::OaState {
        summary: s,
        ..plan.state
    });
    let items2 = match &compacted {
        Some(st) => crate::context_management::oa_render(st, &[]),
        None => plan.window,
    };
    let mut messages = match messages_from_input(req.instructions.as_deref(), &Value::Array(items2))
    {
        Ok(m) => m,
        Err(e) => return Err(err(StatusCode::BAD_REQUEST, "invalid_request_error", e)),
    };
    let mut prepared = match prepare(
        model,
        &req,
        &messages,
        &[],
        state.output_caps(),
        &state.sampling,
    ) {
        Ok(p) => p,
        Err(e) => return Err(err(StatusCode::BAD_REQUEST, "invalid_request_error", e)),
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
            return Err(crate::chat::engine_err(&e));
        }
        let n = drop_leading_turn(&mut messages);
        if n == 0 {
            return Err(crate::chat::engine_err(&e));
        }
        dropped += n;
        prepared = match prepare(
            model,
            &req,
            &messages,
            &[],
            state.output_caps(),
            &state.sampling,
        ) {
            Ok(p) => p,
            Err(e) => return Err(err(StatusCode::BAD_REQUEST, "invalid_request_error", e)),
        };
    }
    let meta = Meta {
        id: format!("resp_{}", uuid::Uuid::new_v4().simple()),
        model_id: model.id.clone(),
        tokenizer: model.tokenizer.clone(),
        prompt_len: prepared.prompt_ids.len(),
        text_prompt_len: prepared.engine_prompt.len(),
        audio_rows: crate::chat::audio_rows(model, None),
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
            compaction: compacted.as_ref().map(|st| compaction_item(&st.encode())),
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
        canvas_read: None,
        user_turn: false,
        background: false,
    };
    if let Err(e) = model.engine.submit(gen_req) {
        return Err(err(StatusCode::INTERNAL_SERVER_ERROR, "internal_error", e));
    }
    Ok((meta, rx))
}

/// `compaction_trigger` (final input item): summarize the whole conversation
/// - no raw tail, the caller is archiving - and answer with only the
///   compaction item (its state keeps the user's messages verbatim). The
///   Responses twin of `pause_after_compaction`; usage covers the
///   summarization pass, the only generation that ran. `plan`: the archive
///   plan over the window `messages` renders (None = an empty window).
pub(super) async fn run_compact_trigger(
    state: Arc<AppState>,
    req: ResponsesRequest,
    messages: Vec<Value>,
    plan: Option<crate::context_management::OaPlan>,
    scope: crate::events::EventScope,
) -> Response {
    let Some(model) = state.serving.as_ref() else {
        return err(
            StatusCode::SERVICE_UNAVAILABLE,
            "model_not_loaded",
            "no model is loaded",
        );
    };
    let Some(plan) = plan else {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "nothing to compact",
        );
    };
    let span = SummarySpan {
        first: messages,
        items: &plan.span,
        users: kept_users(&plan.state),
        instructions: crate::context_management::OA_COMPACT_ALL_INSTRUCTIONS,
        keeps_tail: false,
    };
    let req1 = greedy_for_summary(req.clone());
    let (summary, p1_len, cached1, out1) =
        match summarize(&state, model, &req1, span, &[], &scope, false).await {
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
    let st = crate::context_management::OaState {
        summary: s,
        ..plan.state
    };
    let item = compaction_item(&st.encode());
    let meta = Meta {
        id: format!("resp_{}", uuid::Uuid::new_v4().simple()),
        model_id: model.id.clone(),
        tokenizer: model.tokenizer.clone(),
        prompt_len: p1_len,
        text_prompt_len: p1_len,
        audio_rows: crate::chat::audio_rows(model, None),
        dialect: model.dialect,
        thinking_open: false,
        hints: None,
        single_tool_call: req.parallel_tool_calls == Some(false),
        instructions: req.instructions.clone(),
        max_output_tokens: state.output_caps().resolve(req.max_output_tokens),
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
        return streaming_response(sse);
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
    // an already-compacted conversation re-compacts over its window (a v2
    // state carries its kept messages forward); a stray compaction_trigger
    // is redundant here (this endpoint is the trigger) and is consumed, but
    // still must be the final item
    let prev = crate::context_management::oa_take_state(&mut items);
    if prev.is_none()
        && let Some(rw) = crate::context_management::oa_resend_rewrite(&items)
    {
        items = rw;
    }
    if let Err(e) = crate::context_management::oa_take_trigger(&mut items) {
        return err(StatusCode::BAD_REQUEST, "invalid_request_error", e);
    }
    if items.is_empty() && prev.is_none() {
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
    let Some(plan) = crate::context_management::oa_plan(prev.as_ref(), &items, None) else {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "nothing to compact",
        );
    };
    let mut msgs1 = match messages_from_input(
        req.instructions.as_deref(),
        &Value::Array(plan.window.clone()),
    ) {
        Ok(m) => m,
        Err(e) => return err(StatusCode::BAD_REQUEST, "invalid_request_error", e),
    };
    // A summary pass reads the same prompt the answer would, capability and
    // all - otherwise the two disagree about what the model was told, and the
    // prefix cache is aligned against a prompt that never ran.
    if let Some(sample) = compact_map.as_deref() {
        crate::doc::add_map_capability(&mut msgs1, sample);
    }
    let span = SummarySpan {
        first: msgs1,
        items: &plan.window,
        users: kept_users(&plan.state),
        instructions: crate::context_management::OA_COMPACT_ALL_INSTRUCTIONS,
        keeps_tail: false,
    };
    let mut req1 = greedy_for_summary(blank_request());
    // later parts render `instructions` as the system turn too
    req1.instructions = req.instructions.clone();
    let (summary, p1_len, cached1, out1) =
        match summarize(&state, model, &req1, span, &[], &scope, false).await {
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
    // discriminator made explicit so SDK unions parse the echo. The item's
    // state keeps them too, so a resend renders the same window whether the
    // caller sends the echo back or not.
    let st = crate::context_management::OaState {
        summary: s,
        ..plan.state
    };
    let mut output: Vec<Value> = st
        .users
        .iter()
        .cloned()
        .map(|mut it| {
            if it.get("type").is_none() {
                it["type"] = json!("message");
            }
            it
        })
        .collect();
    output.push(compaction_item(&st.encode()));
    Json(json!({
        "id": format!("cr_{}", uuid::Uuid::new_v4().simple()),
        "created_at": now_secs(),
        "object": "response.compaction",
        "output": output,
        "usage": {
            "input_tokens": p1_len,
            "input_tokens_details": paddock_api::completions::Usage::media_details(p1_len, cached1, 0, 0),
            "output_tokens": out1,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": p1_len + out1,
        },
    }))
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A failure after the stream has opened (the compaction runs inside
    /// it) reaches the client as the typed Responses `error` event, code
    /// and message carried over from the error body it would have been.
    #[tokio::test]
    async fn a_failure_inside_the_stream_is_the_typed_error_event() {
        use http_body_util::BodyExt;
        let resp = crate::chat::engine_err(
            &paddock_engine::service::EngineError::context_overflow(300_000, 262_144),
        );
        let sse = streaming_response(futures::stream::once(stream_error(resp)));
        let body = sse.into_body().collect().await.expect("body").to_bytes();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(text.contains("event: error"), "{text}");
        let data = text
            .lines()
            .find_map(|l| l.strip_prefix("data: "))
            .expect("data line");
        let v: Value = serde_json::from_str(data).unwrap();
        assert_eq!(v["type"], "error");
        assert_eq!(v["code"], "context_length_exceeded");
        assert!(v["message"].as_str().unwrap().contains("262144"));
    }
}
