//! Drain control (doc §5): the takeover/shutdown path. Draining is one-way -
//! new inference requests get 503 + `Retry-After` (retry-capable SDKs recover
//! unaided), in-flight requests finish under a bounded timeout, then the
//! process exits and the OS guarantees the VRAM back.
//!
//! In-flight accounting counts until the RESPONSE BODY completes, not until
//! the handler returns - a streaming completion is in flight for as long as
//! tokens are still flowing, which is exactly the work drain must wait for.
//! That's what `CountedBody` is: an RAII guard riding the response body.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

/// Shared drain state. One per runner, on `AppState`.
#[derive(Default)]
pub struct DrainCtl {
    draining: AtomicBool,
    in_flight: AtomicUsize,
}

impl DrainCtl {
    /// Flip into draining. Idempotent; there is no way back (a drained
    /// runner's next step is exit).
    pub fn begin(&self) {
        if !self.draining.swap(true, Ordering::SeqCst) {
            tracing::info!("drain started: refusing new inference requests (503 + Retry-After)");
        }
    }

    pub fn is_draining(&self) -> bool {
        self.draining.load(Ordering::SeqCst)
    }

    pub fn in_flight(&self) -> usize {
        self.in_flight.load(Ordering::SeqCst)
    }

    /// Register one in-flight request; the guard's Drop deregisters it.
    pub fn guard(self: &Arc<Self>) -> InFlightGuard {
        self.in_flight.fetch_add(1, Ordering::SeqCst);
        InFlightGuard { ctl: self.clone() }
    }

    /// Wait until nothing is in flight, up to `timeout`. Returns true if
    /// drained, false on timeout (requests still running).
    pub async fn wait_drained(&self, timeout: Duration) -> bool {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            if self.in_flight() == 0 {
                return true;
            }
            if tokio::time::Instant::now() >= deadline {
                return false;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }
}

/// Decrements the in-flight count when dropped - attached to the response
/// body so streaming responses count until their last byte.
pub struct InFlightGuard {
    ctl: Arc<DrainCtl>,
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.ctl.in_flight.fetch_sub(1, Ordering::SeqCst);
    }
}

/// A response body that holds an `InFlightGuard` until it is dropped (client
/// done reading, disconnected, or the response was discarded).
pub struct CountedBody {
    inner: axum::body::Body,
    _guard: InFlightGuard,
}

impl CountedBody {
    pub fn wrap(res: axum::response::Response, guard: InFlightGuard) -> axum::response::Response {
        let (parts, body) = res.into_parts();
        axum::response::Response::from_parts(
            parts,
            axum::body::Body::new(CountedBody {
                inner: body,
                _guard: guard,
            }),
        )
    }
}

impl http_body::Body for CountedBody {
    type Data = axum::body::Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Result<http_body::Frame<Self::Data>, Self::Error>>> {
        // axum's Body is Unpin (it boxes its inner stream), so this projection
        // needs no unsafe.
        let this = self.get_mut();
        std::pin::Pin::new(&mut this.inner).poll_frame(cx)
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.inner.size_hint()
    }
}

/// The endpoints drain gates: everything that starts inference work. Cheap
/// reads (/v1/models, /api/*) stay up so the manager and probes keep sight of
/// a draining runner. (Also the event-recorded set - one request, one record.)
pub(crate) fn is_inference_path(path: &str) -> bool {
    matches!(
        path,
        "/v1/completions"
            | "/v1/chat/completions"
            | "/v1/responses"
            | "/v1/messages"
            | "/v1/messages/count_tokens"
            | "/v1/embeddings"
            | "/v1/rerank"
            | "/v1/audio/transcriptions"
            | "/v1/audio/alignments"
            | "/v1/audio/diarizations"
            | "/v1/audio/diarizations/stream"
            | "/v1/segmentations"
            | "/v1/masks"
            | "/v1/images/generations"
            | "/v1/images/edits"
            | "/v1/tabular/predictions"
            | "/v1/tabular/contexts"
    ) || path.starts_with("/v1/masks/sessions")
}

/// The subset the admission cap refuses: real engine work. count_tokens is a
/// pure tokenize - keeping it un-refused means token budgeting stays alive on
/// a saturated endpoint (it is still drain-counted above, just never capped).
fn is_capped_path(path: &str) -> bool {
    is_inference_path(path) && path != "/v1/messages/count_tokens"
}

/// The Overloaded refusal in the dialect's own error shape: the Anthropic
/// envelope + 529 on /v1/messages (their documented overload status), the
/// OpenAI ErrorBody + 503 everywhere else. Both carry Retry-After, which
/// retry-capable SDKs honor unaided.
fn overloaded_response(
    path: &str,
    retry_after_secs: &'static str,
    msg: &str,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    let mut res = if path.starts_with("/v1/messages") {
        (
            axum::http::StatusCode::from_u16(529).expect("529 is a valid status"),
            axum::Json(serde_json::json!({
                "type": "error",
                "error": { "type": "overloaded_error", "message": msg }
            })),
        )
            .into_response()
    } else {
        (
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            axum::Json(paddock_api::ErrorBody::new("overloaded_error", msg)),
        )
            .into_response()
    };
    res.headers_mut().insert(
        axum::http::header::RETRY_AFTER,
        axum::http::HeaderValue::from_static(retry_after_secs),
    );
    res
}

/// Middleware: refuse new inference work while draining; enforce the explicit
/// admission cap (doc §13, llama-swap's `concurrencyLimit` lesson - queue
/// depth, distinct from `max_batch`'s compute width); count in-flight
/// inference requests (through their streaming bodies) otherwise.
pub async fn drain_mw(
    axum::extract::State(state): axum::extract::State<Arc<crate::routes::AppState>>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let path = req.uri().path();
    if !is_inference_path(path) {
        return next.run(req).await;
    }
    if state.drain.is_draining() {
        // Model-load timescale: the takeover successor is typically up within
        // this window.
        return overloaded_response(
            path,
            "15",
            "this endpoint is draining (model switch or shutdown in progress); retry shortly",
        );
    }
    let guard = state.drain.guard();
    // Cap check after our own increment, so concurrent arrivals can't all
    // read cap-1 and slip through together: whoever pushes the count past the
    // cap sees it and backs off (the guard's Drop releases the slot).
    if let Some(cap) = state.concurrency_limit
        && is_capped_path(path)
        && state.drain.in_flight() > cap
    {
        drop(guard);
        return overloaded_response(
            path,
            "1",
            &format!("endpoint at its admission cap ({cap} concurrent requests); retry shortly"),
        );
    }
    let res = next.run(req).await;
    CountedBody::wrap(res, guard)
}
