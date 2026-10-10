//! `POST /v1/completions` - non-streaming and SSE streaming.
//!
//! Incremental detokenization: we decode the whole id sequence each step and
//! emit the new UTF-8 suffix. Correct across multi-byte boundaries (never
//! splits a char); a byte-buffered streaming detokenizer is a perf refinement.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use async_stream::stream;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use paddock_api::ErrorBody;
use paddock_api::completions::{CompletionChoice, CompletionRequest, CompletionResponse, Usage};
use paddock_engine::sampler::SamplingParams;
use paddock_engine::service::{EngineError, FinishReason, GenRequest, TokenEvent, TokenLogprobs};
use tokio::sync::mpsc::{UnboundedReceiver, unbounded_channel};

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

/// Turn the request into prompt ids + a GenRequest (minus the events channel).
struct Prepared {
    prompt_ids: Vec<u32>,
    max_tokens: usize,
    sampler: SamplingParams,
    stop_tokens: Vec<u32>,
    /// stop strings for text-level truncation
    stop_strings: Vec<String>,
    /// Some(top_n) when the request wants legacy logprobs
    logprobs: Option<u8>,
    /// how many completions to RETURN (1..=8)
    n: usize,
    /// how many to GENERATE before ranking; >= n, and 1 unless best_of was
    /// asked for. When it exceeds n we rank and drop the losers.
    best_of: usize,
    /// prepend the prompt to each returned text
    echo: bool,
}

/// Rank a candidate by log probability per TOKEN, which is what OpenAI's
/// `best_of` selects on - a plain sum would just prefer the shortest
/// completion. An empty candidate has no evidence either way and sorts last.
fn mean_logprob(lps: &[TokenLogprobs]) -> f32 {
    if lps.is_empty() {
        return f32::NEG_INFINITY;
    }
    lps.iter().map(|l| l.chosen).sum::<f32>() / lps.len() as f32
}

/// `n` / `best_of`, resolved and checked. Model-independent, so the handler
/// runs it before the model-availability check - a malformed request is
/// malformed whether or not a model is loaded, and answering 503 there hides a
/// 400 the caller has to fix. `prepare` calls it again for the values.
fn choice_counts(req: &CompletionRequest) -> Result<(usize, usize), String> {
    // The ceiling matches chat's: n engine sequences is n slots.
    let n = req.n.unwrap_or(1);
    if !(1..=8).contains(&n) {
        return Err(format!("n must be 1..=8 (got {n})"));
    }
    let best_of = req.best_of.unwrap_or(n);
    if best_of < n {
        return Err(format!("best_of ({best_of}) must be >= n ({n})"));
    }
    if best_of > 8 {
        return Err(format!("best_of must be 1..=8 (got {best_of})"));
    }
    // OpenAI refuses this too, and not as a policy choice: ranking picks a
    // winner only once every candidate has finished, so there is nothing to
    // stream until the answer is already complete. Note the guard is
    // `best_of > n`, not `best_of.is_some()` - best_of == n is just n, and
    // refusing that would reject a legal request.
    if best_of > n && req.stream {
        return Err(
            "best_of > n cannot be combined with stream: true (the winner is only known once every candidate has finished)"
                .into(),
        );
    }
    Ok((n, best_of))
}

fn prepare(
    model: &ServingModel,
    req: &CompletionRequest,
    output_ceiling: Option<usize>,
    sd: &crate::routes::SamplingDefaults,
) -> Result<Prepared, String> {
    let mut prompt_ids = model
        .tokenizer
        .encode(req.prompt.first())
        .map_err(|e| e.to_string())?;
    // base-completion convention: lead with BOS when the model has one
    if let Some(bos) = model.bos
        && prompt_ids.first() != Some(&bos)
    {
        prompt_ids.insert(0, bos);
    }

    crate::chat::validate_sampling(
        req.temperature,
        2.0,
        req.top_p,
        req.min_p,
        Some(req.frequency_penalty),
        Some(req.presence_penalty),
    )?;
    // Request field wins; else this model's elected defaults. There is no
    // chat template on this route and so no thinking mode to be in - a raw
    // completion takes the model's default-mode profile, which is the same
    // thing its own generation_config.json would hand a plain
    // `model.generate()` call.
    let dflt = sd.resolve(true);
    let sampler = SamplingParams {
        temperature: req.temperature.unwrap_or(dflt.temp),
        top_k: req.top_k.unwrap_or(dflt.top_k),
        top_p: req.top_p.unwrap_or(dflt.top_p),
        min_p: req.min_p.unwrap_or(dflt.min_p),
        repeat_penalty: req.repeat_penalty.unwrap_or(dflt.repeat_penalty),
        repeat_last_n: sd.repeat_last_n,
        presence_penalty: req.presence_penalty,
        frequency_penalty: req.frequency_penalty,
        // no seed -> time-derived (OpenAI: seed absent = nondeterministic)
        seed: sd.seed_or_now(req.seed),
        logit_bias: crate::chat::parse_logit_bias(
            req.logit_bias.as_ref(),
            model.tokenizer.vocab_size,
        )?,
        // raw completions carry no images, so the OCR family's ngram
        // default never applies here
        no_repeat_ngram: (0, 0, false),
    };

    let stop_strings = req.stop.as_ref().map(|s| s.to_vec()).unwrap_or_default();

    if let Some(k) = req.logprobs
        && k > 20
    {
        return Err(format!("logprobs must be 0..=20 (got {k})"));
    }

    let (n, best_of) = choice_counts(req)?;

    Ok(Prepared {
        prompt_ids,
        max_tokens: output_ceiling.map_or(req.max_tokens, |c| req.max_tokens.min(c)),
        sampler,
        stop_tokens: model.stop_tokens.clone(),
        stop_strings,
        logprobs: req.logprobs,
        n,
        best_of,
        echo: req.echo,
    })
}

/// Truncate `text` at the earliest stop string, if any.
pub(crate) fn apply_stop_strings(text: &str, stops: &[String]) -> (String, bool) {
    let mut cut = text.len();
    let mut hit = false;
    for s in stops {
        if !s.is_empty()
            && let Some(idx) = text.find(s.as_str())
            && idx < cut
        {
            cut = idx;
            hit = true;
        }
    }
    (text[..cut].to_owned(), hit)
}

/// `POST /v1/completions` handler.
pub async fn handle(
    State(state): State<Arc<AppState>>,
    scope: Option<axum::Extension<crate::events::EventScope>>,
    crate::extract::OaiJson(req): crate::extract::OaiJson<CompletionRequest>,
) -> Response {
    let scope = scope.map(|e| e.0).unwrap_or_default();
    if let Err(e) = choice_counts(&req) {
        return err(StatusCode::BAD_REQUEST, "invalid_request_error", e);
    }
    let Some(model) = state.serving.as_ref() else {
        return err(
            StatusCode::SERVICE_UNAVAILABLE,
            "model_not_loaded",
            "no model is loaded; start paddock with a `model` in config",
        );
    };
    scope.model(&model.id);
    scope.user(req.user.as_deref());

    let t_prep = std::time::Instant::now();
    let prepared = match prepare(model, &req, state.max_output_ceiling, &state.sampling) {
        Ok(p) => p,
        Err(e) => return err(StatusCode::BAD_REQUEST, "invalid_request_error", e),
    };
    scope.tokenized(t_prep.elapsed());
    // Over-window prompt: clean 400 at the edge for stream and non-stream alike
    // (mirrors the engine's admit check).
    if state.max_ctx > 0 && prepared.prompt_ids.len() > state.max_ctx {
        return crate::chat::engine_err(&EngineError::context_overflow(
            prepared.prompt_ids.len(),
            state.max_ctx,
        ));
    }

    let meta = ResponseMeta {
        id: format!("cmpl-{}", uuid::Uuid::new_v4().simple()),
        model_id: model.id.clone(),
        tokenizer: model.tokenizer.clone(),
        prompt_len: prepared.prompt_ids.len(),
        // legacy text_offset counts characters over prompt + completion
        prompt_chars: req.prompt.first().chars().count(),
        logprobs: prepared.logprobs.is_some(),
        echo: prepared.echo.then(|| req.prompt.first().to_owned()),
        scope,
    };

    // best_of candidates (== n unless ranking was asked for), each its own
    // engine sequence. Per-choice seed offsets keep sampled candidates
    // independent, the way chat does it; greedy candidates come out identical,
    // which is also what OpenAI returns.
    //
    // Ranking needs the chosen-token logprob of every candidate, which the
    // engine only produces when asked. So when we are ranking we request
    // logprobs at k=0 internally even if the caller wanted none - k=0 carries
    // `chosen` and no top-k list, and `meta.logprobs` still decides whether
    // any of it is REPORTED.
    let gen_logprobs = if prepared.best_of > prepared.n {
        Some(prepared.logprobs.unwrap_or(0))
    } else {
        prepared.logprobs
    };
    let mut rxs = Vec::with_capacity(prepared.best_of);
    for i in 0..prepared.best_of {
        let (tx, rx) = unbounded_channel();
        let mut sampler = prepared.sampler.clone();
        sampler.seed = sampler.seed.wrapping_add(i as u64);
        let gen_req = GenRequest {
            prompt: prepared.prompt_ids.clone(),
            max_tokens: prepared.max_tokens,
            sampler,
            stop_tokens: prepared.stop_tokens.clone(),
            events: tx,
            mm_chunks: None,
            constraint: None,
            logprobs: gen_logprobs,
            submitted: None, // stamped by Engine::submit
            canvas_read: None,
            user_turn: false,
        };
        if let Err(e) = model.engine.submit(gen_req) {
            return err(StatusCode::INTERNAL_SERVER_ERROR, "internal_error", e);
        }
        rxs.push(rx);
    }

    let include_usage = match req.stream_options.as_ref() {
        None => false,
        Some(v) => {
            if !req.stream {
                return err(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    "stream_options requires stream: true",
                );
            }
            v.get("include_usage")
                .and_then(|b| b.as_bool())
                .unwrap_or(false)
        }
    };

    if req.stream {
        stream_response(meta, rxs, prepared.stop_strings, include_usage)
    } else {
        collect_response(meta, rxs, prepared).await
    }
}

/// Per-request metadata the response builders need.
pub struct ResponseMeta {
    pub id: String,
    pub model_id: String,
    pub tokenizer: Arc<paddock_tokenizer::GgufTokenizer>,
    pub prompt_len: usize,
    /// prompt text length in CHARACTERS - the legacy `text_offset` base
    /// (OpenAI counts offsets over prompt + completion)
    pub prompt_chars: usize,
    /// legacy logprobs requested - attach the array-shape object
    pub logprobs: bool,
    /// Some(prompt text) when `echo` was asked for. The prompt is prepended to
    /// every choice; `text_offset` already bases at `prompt_chars`, so the
    /// legacy logprobs offsets are correct for the echoed string as they
    /// stand. What is not included is a logprob entry per prompt token -
    /// nothing scores the prompt here, and inventing nulls for it would claim
    /// a measurement we never made.
    pub echo: Option<String>,
    /// Event-record slots (§8.1); no-op unless the events middleware planted one.
    pub scope: crate::events::EventScope,
}

/// The legacy `/v1/completions` logprobs object over a token run: parallel
/// `tokens` / `token_logprobs` / `top_logprobs` / `text_offset` arrays.
/// `off0` = character offset of the first token (prompt chars + completion
/// chars already emitted). The engine's `top` already includes the chosen
/// token (probability-descending), which is exactly the legacy dict shape.
/// Returns the object and the character offset after the run.
fn legacy_logprobs_json(
    tokenizer: &paddock_tokenizer::GgufTokenizer,
    ids: &[u32],
    lps: &[TokenLogprobs],
    off0: usize,
) -> (serde_json::Value, usize) {
    let mut tokens = Vec::with_capacity(ids.len());
    let mut token_logprobs = Vec::with_capacity(ids.len());
    let mut top_logprobs = Vec::with_capacity(ids.len());
    let mut text_offset = Vec::with_capacity(ids.len());
    let mut off = off0;
    for (&id, lp) in ids.iter().zip(lps) {
        let tok = tokenizer.decode(&[id], false).unwrap_or_default();
        text_offset.push(off);
        off += tok.chars().count();
        token_logprobs.push(lp.chosen);
        let mut map = serde_json::Map::new();
        for &(tid, l) in &lp.top {
            let ts = tokenizer.decode(&[tid], false).unwrap_or_default();
            map.insert(ts, serde_json::json!(l));
        }
        top_logprobs.push(serde_json::Value::Object(map));
        tokens.push(tok);
    }
    (
        serde_json::json!({
            "tokens": tokens,
            "token_logprobs": token_logprobs,
            "top_logprobs": top_logprobs,
            "text_offset": text_offset,
        }),
        off,
    )
}

async fn collect_response(
    mut meta: ResponseMeta,
    rxs: Vec<UnboundedReceiver<TokenEvent>>,
    prepared: Prepared,
) -> Response {
    // One drained candidate.
    struct Cand {
        ids: Vec<u32>,
        lps: Vec<TokenLogprobs>,
        finish: String,
    }

    let mut cands: Vec<Cand> = Vec::with_capacity(rxs.len());
    // Bill every generated candidate, including sampled stop IDs and the
    // candidates subsequently discarded by best_of ranking.
    let mut generated = 0usize;
    let mut cached = 0usize;
    for mut rx in rxs {
        let mut ids = Vec::new();
        let mut lps: Vec<TokenLogprobs> = Vec::new();
        let mut finish = "length".to_owned();
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
                    ids.push(t);
                    if let Some(lp) = logprobs {
                        lps.push(lp);
                    }
                }
                TokenEvent::Done(reason, stats) => {
                    finish = reason.as_str().to_owned();
                    generated += stats.terminal_tokens();
                    meta.scope.phases(&stats);
                    break;
                }
                TokenEvent::Error(e) => {
                    return crate::chat::engine_err(&e);
                }
            }
        }
        generated += ids.len();
        cands.push(Cand { ids, lps, finish });
    }

    // best_of: keep the n with the highest per-token log probability. The sort
    // is stable, so an untied field (greedy, or ranking off) keeps submission
    // order and choice `index` still reads as "the i-th completion".
    if prepared.best_of > prepared.n {
        cands.sort_by(|a, b| {
            mean_logprob(&b.lps)
                .partial_cmp(&mean_logprob(&a.lps))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        cands.truncate(prepared.n);
    }

    let mut choices = Vec::with_capacity(cands.len());
    let mut completion_tokens = 0usize;
    for (index, c) in cands.iter().enumerate() {
        let text = meta.tokenizer.decode(&c.ids, true).unwrap_or_default();
        let (text, stopped) = apply_stop_strings(&text, &prepared.stop_strings);
        let finish = if stopped {
            "stop".to_owned()
        } else {
            c.finish.clone()
        };

        // entries cover all sampled tokens, including any past a stop-string cut
        // (llama.cpp-server behavior; text-level truncation loses token alignment)
        let logprobs = (meta.logprobs && c.lps.len() == c.ids.len())
            .then(|| legacy_logprobs_json(&meta.tokenizer, &c.ids, &c.lps, meta.prompt_chars).0);

        // echo prepends the prompt to the completion - the whole point of the
        // flag, and why `text_offset` is based at prompt_chars already
        let text = match meta.echo.as_deref() {
            Some(p) => format!("{p}{text}"),
            None => text,
        };
        completion_tokens += c.ids.len();
        choices.push(CompletionChoice {
            text,
            index: index as u32,
            logprobs,
            finish_reason: Some(finish),
        });
    }
    // Usage counts what was GENERATED, which under best_of includes the
    // candidates that lost - the caller paid for those too, and OpenAI bills
    // them the same way.
    let _ = completion_tokens;
    meta.scope.finish(
        choices
            .first()
            .and_then(|c| c.finish_reason.as_deref())
            .unwrap_or("length"),
    );
    meta.scope.usage(meta.prompt_len, generated);
    meta.scope.cached(cached);
    Json(CompletionResponse {
        id: meta.id,
        object: CompletionResponse::object_name(),
        created: now_secs(),
        model: meta.model_id,
        choices,
        usage: Some(Usage {
            prompt_tokens: meta.prompt_len,
            completion_tokens: generated,
            total_tokens: meta.prompt_len + generated,
            prompt_tokens_details: Usage::cached_details(meta.prompt_len, cached),
        }),
    })
    .into_response()
}

fn stream_response(
    mut meta: ResponseMeta,
    rxs: Vec<UnboundedReceiver<TokenEvent>>,
    stop_strings: Vec<String>,
    include_usage: bool,
) -> Response {
    let created = now_secs();
    let n = rxs.len();
    let sse = stream! {
        // Per-choice stream state. `n` sequences are interleaved on one SSE
        // channel and told apart by the chunk's `index`, the same way chat
        // merges its choices; a client reassembles per index.
        struct ChoiceState {
            sd: paddock_tokenizer::StreamDecoder,
            emitted: usize,     // bytes of decoded text already sent
            ids: usize,         // tokens seen (usage only)
            lp_off: usize,      // legacy text_offset cursor
            echoed: bool,       // the prompt has been sent on this choice
            done: bool,
        }
        let mut cs: Vec<ChoiceState> = (0..n)
            .map(|_| ChoiceState {
                sd: meta.tokenizer.stream_decoder(true),
                emitted: 0,
                ids: 0,
                lp_off: meta.prompt_chars,
                echoed: false,
                done: false,
            })
            .collect();
        let mut cached = 0usize;
        let mut finished = 0usize;

        let mut merged = futures::stream::select_all(rxs.into_iter().enumerate().map(|(i, rx)| {
            Box::pin(async_stream::stream! {
                let mut rx = rx;
                while let Some(ev) = rx.recv().await {
                    yield (i, ev);
                }
            })
        }));

        while let Some((i, ev)) = futures::StreamExt::next(&mut merged).await {
            if cs[i].done {
                continue;
            }
            match ev {
                TokenEvent::Prefilled { cached: c, rows } => {
                    cached = c as usize;
                    meta.prompt_len = meta.prompt_len.max(rows as usize);
                }
                TokenEvent::Token { id: t, logprobs } => {
                    // echo rides out as the first delta on each choice, so a
                    // client concatenating deltas rebuilds prompt+completion
                    // exactly as the non-streaming body returns it
                    if !cs[i].echoed {
                        cs[i].echoed = true;
                        if let Some(prompt) = meta.echo.clone() {
                            yield Ok::<_, std::convert::Infallible>(Event::default().data(
                                chunk_json(&meta, created, &prompt, None, None, i),
                            ));
                        }
                    }
                    cs[i].ids += 1;
                    let full = cs[i].sd.push(&meta.tokenizer, t);
                    // text-level stop: truncate + finish
                    let (trunc, hit) = apply_stop_strings(&full, &stop_strings);
                    // per-token logprobs chunk (OpenAI streams one chunk per
                    // token): emitted even when the text delta is still empty
                    // (multi-byte char in flight), so entries never drop
                    let lp = logprobs.map(|lp| {
                        let (v, off) =
                            legacy_logprobs_json(&meta.tokenizer, &[t], &[lp], cs[i].lp_off);
                        cs[i].lp_off = off;
                        v
                    });
                    if trunc.len() > cs[i].emitted || lp.is_some() {
                        let e0 = cs[i].emitted;
                        let delta = full[e0..trunc.len().max(e0)].to_owned();
                        cs[i].emitted = trunc.len().max(e0);
                        yield Ok(Event::default().data(
                            chunk_json(&meta, created, &delta, lp, None, i),
                        ));
                    }
                    if hit {
                        cs[i].done = true;
                        finished += 1;
                        yield Ok(Event::default().data(
                            chunk_json(&meta, created, "", None, Some("stop"), i),
                        ));
                    }
                }
                TokenEvent::Done(reason, stats) => {
                    cs[i].ids += stats.terminal_tokens();
                    let f = match reason {
                        FinishReason::Stop | FinishReason::Repetition => "stop",
                        FinishReason::Length => "length",
                    };
                    meta.scope.phases(&stats);
                    cs[i].done = true;
                    finished += 1;
                    // a choice that produced nothing still owes its echo
                    if !cs[i].echoed && let Some(prompt) = meta.echo.clone() {
                        cs[i].echoed = true;
                        yield Ok(Event::default().data(
                            chunk_json(&meta, created, &prompt, None, None, i),
                        ));
                    }
                    yield Ok(Event::default().data(
                        chunk_json(&meta, created, "", None, Some(f), i),
                    ));
                }
                TokenEvent::Error(_) => {
                    cs[i].done = true;
                    finished += 1;
                    yield Ok(Event::default().data(
                        chunk_json(&meta, created, "", None, Some("stop"), i),
                    ));
                }
            }
            if finished == n {
                break;
            }
        }

        let total: usize = cs.iter().map(|c| c.ids).sum();
        meta.scope.finish("stop");
        meta.scope.usage(meta.prompt_len, total);
        meta.scope.cached(cached);
        // Always emitted, not just under include_usage - same
        // rationale as the chat stream: benchmark clients need
        // server counts or they re-tokenize visible text and undercount.
        {
            let _ = include_usage; // kept for the request-validation path
            let usage = serde_json::json!({
                "id": meta.id,
                "object": "text_completion",
                "created": created,
                "model": meta.model_id,
                "choices": [],
                "usage": {
                    "prompt_tokens": meta.prompt_len,
                    "completion_tokens": total,
                    "total_tokens": meta.prompt_len + total,
                    "prompt_tokens_details": Usage::cached_details(meta.prompt_len, cached),
                },
            });
            yield Ok(Event::default().data(usage.to_string()));
        }
        yield Ok(Event::default().data("[DONE]"));
    };

    // SSE comments while the model works in silence (a long cold prefill, a
    // tool call that is emitted whole): client idle timeouts count them,
    // parsers skip them - the Responses surface's rule (responses.rs)
    Sse::new(sse)
        .keep_alive(KeepAlive::default())
        .into_response()
}

fn chunk_json(
    meta: &ResponseMeta,
    created: u64,
    text: &str,
    logprobs: Option<serde_json::Value>,
    finish: Option<&str>,
    index: usize,
) -> String {
    let choice = serde_json::json!({
        "text": text,
        "index": index,
        "logprobs": logprobs,
        "finish_reason": finish,
    });
    serde_json::json!({
        "id": meta.id,
        "object": "text_completion",
        "created": created,
        "model": meta.model_id,
        "choices": [choice],
    })
    .to_string()
}
