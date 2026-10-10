//! The document read as a chat completion: the Markdown as the assistant's
//! content, usage summed over every region, and the `ocr` echo - the
//! resolution, the layout facts and every region. Streaming sends the
//! Markdown block by block in reading order, the echo on the terminal usage
//! chunk like every other OCR stream.

use std::sync::Arc;

use async_stream::stream;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json, Response};
use paddock_api::ErrorBody;
use paddock_api::chat::{ChatChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage};
use paddock_api::completions::Usage;
use paddock_engine::service::{FinishReason, MmChunk};
use serde_json::{Value, json};

use super::{DocEvent, Page, REGION_TOKENS, Summary, start};
use crate::deepseek_ocr::OcrResolved;
use crate::routes::AppState;

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn bad(msg: impl Into<String>) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(ErrorBody::new("invalid_request_error", msg)),
    )
        .into_response()
}

/// The echo: the resolution (a repetition stop fired if any region's did),
/// what the layout found and every region.
fn echo(ocr: &OcrResolved, sum: &Summary, pages: usize) -> Value {
    let fired = (sum.stopped > 0).then_some(FinishReason::Repetition);
    let mut e = ocr.echo_at(fired.or(Some(FinishReason::Stop)));
    let read = sum
        .regions
        .iter()
        .filter(|r| r.get("text").is_some())
        .count();
    e["layout"] = json!({
        "model": "PP-DocLayoutV3",
        "pages": pages,
        "regions": sum.regions.len(),
        "read": read,
        "repetition_stops": sum.stopped,
        "truncated": sum.truncated,
        "ms": (sum.layout_ms * 10.0).round() / 10.0,
    });
    e["regions"] = Value::Array(sum.regions.clone());
    e
}

/// `finish_reason`: a region cut at the token cap makes the document cut.
fn finish(sum: &Summary) -> &'static str {
    if sum.truncated > 0 { "length" } else { "stop" }
}

/// Serve one document-mode chat request over its decoded pages.
pub(crate) async fn respond(
    state: &Arc<AppState>,
    req: &ChatCompletionRequest,
    ocr: OcrResolved,
    chunks: Vec<MmChunk>,
) -> Response {
    if req.n > 1 {
        return bad("ocr.mode \"document\" reads one answer per page set (n must be 1)");
    }
    let pages: Vec<Page> = chunks
        .into_iter()
        .filter_map(|c| match c {
            MmChunk::Image { rgb, w, h } => Some(Page { rgb, w, h }),
            _ => None,
        })
        .collect();
    let n_pages = pages.len();
    // the caller's cap applies to each region; the pipeline's own otherwise
    let cap = req
        .max_completion_tokens
        .or(req.max_tokens)
        .unwrap_or(REGION_TOKENS);
    let model_id = state
        .serving
        .as_ref()
        .map(|m| m.id.clone())
        .unwrap_or_default();
    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());
    let created = now_secs();
    let mut rx = start(state.clone(), pages, cap);
    if !req.stream {
        let mut content = String::new();
        while let Some(ev) = rx.recv().await {
            match ev {
                DocEvent::Text(t) => content.push_str(&t),
                DocEvent::Error(e) => {
                    tracing::warn!(error = %e, "document read failed");
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(ErrorBody::new("engine_error", e)),
                    )
                        .into_response();
                }
                DocEvent::Done(sum) => {
                    return Json(ChatCompletionResponse {
                        id,
                        object: "chat.completion",
                        created,
                        model: model_id,
                        choices: vec![ChatChoice {
                            index: 0,
                            message: ChatMessage::assistant(Some(content), None, Vec::new()),
                            logprobs: None,
                            finish_reason: Some(finish(&sum).to_owned()),
                        }],
                        usage: Usage {
                            prompt_tokens: sum.prompt_tokens,
                            completion_tokens: sum.completion_tokens,
                            total_tokens: sum.prompt_tokens + sum.completion_tokens,
                            prompt_tokens_details: None,
                        },
                        ocr: Some(echo(&ocr, &sum, n_pages)),
                    })
                    .into_response();
                }
            }
        }
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorBody::new(
                "engine_error",
                "the document read ended early",
            )),
        )
            .into_response();
    }
    let chunk = move |delta: Value, finish: Option<&str>| {
        json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish, "logprobs": null}],
        })
    };
    let sse = stream! {
        let data = |v: Value| Ok::<_, std::convert::Infallible>(Event::default().data(v.to_string()));
        yield data(chunk(json!({"role": "assistant"}), None));
        while let Some(ev) = rx.recv().await {
            match ev {
                DocEvent::Text(t) => yield data(chunk(json!({"content": t}), None)),
                DocEvent::Error(e) => {
                    tracing::warn!(error = %e, "document stream failed");
                    yield data(json!({"error": {"message": e, "type": "engine_error"}}));
                    break;
                }
                DocEvent::Done(sum) => {
                    yield data(chunk(json!({}), Some(finish(&sum))));
                    let mut usage = chunk(json!({}), None);
                    usage["choices"] = json!([]);
                    usage["usage"] = json!({
                        "prompt_tokens": sum.prompt_tokens,
                        "completion_tokens": sum.completion_tokens,
                        "total_tokens": sum.prompt_tokens + sum.completion_tokens,
                        "ocr": echo(&ocr, &sum, n_pages),
                    });
                    yield data(usage);
                    break;
                }
            }
        }
        yield Ok(Event::default().data("[DONE]"));
    };
    Sse::new(sse)
        .keep_alive(KeepAlive::default())
        .into_response()
}
