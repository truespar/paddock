//! How a sequence ended - the engine's half of the APIs' finish reasons. Every
//! request ends with a `Done` or an `Error`, shutdown included; a stream that
//! closes bare was cut off, and `next_event` hands it to the caller as one.

use tokio::sync::mpsc::UnboundedReceiver;

use super::admission::Inbox;
use super::{EngineError, GenRequest, Slot, TokenEvent};

/// Tokens of the vocabulary's first id in a row that mark a dead stream.
pub(super) const DEGENERATE_RUN: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// hit a stop token
    Stop,
    /// hit max_tokens
    Length,
    /// the sampler's repetition stop fired: the token picked would have
    /// repeated an n-gram already begun within its window (a degenerate
    /// loop), so the sequence ended before it. Reported as a stop on the
    /// wire; the document parsers' `ocr` echo says it fired.
    Repetition,
}

impl FinishReason {
    /// OpenAI `finish_reason` string.
    pub fn as_str(&self) -> &'static str {
        match self {
            FinishReason::Stop | FinishReason::Repetition => "stop",
            FinishReason::Length => "length",
        }
    }
}

impl EngineError {
    /// A request whose events stopped with neither a finish nor an error.
    pub fn unfinished() -> Self {
        EngineError::internal("the engine ended the request without finishing it")
    }
}

/// The next event of a request; a bare close becomes `EngineError::unfinished`
/// (never `None`). A closed stream used to read as a finished one, and every
/// surface reported the cut-off reply as a normal end of turn.
pub async fn next_event(rx: &mut UnboundedReceiver<TokenEvent>) -> Option<TokenEvent> {
    Some(
        rx.recv()
            .await
            .unwrap_or_else(|| TokenEvent::Error(EngineError::unfinished())),
    )
}

/// Shutdown: every request the scheduler still holds - live, preempted for
/// recompute, waiting for an exclusive multimodal run, or queued behind them -
/// ends with a retryable error. Dropped silently, its stream closed with no
/// finish, which every surface reported as a normal end of turn: a runner
/// stopped mid-reply (a model switch, a restart) handed its clients a cut-off
/// reply as complete, and Claude Code kept it.
pub(super) fn fail_held_requests(
    slots: &mut [Option<Slot>],
    preempted: &mut Vec<Slot>,
    mm_pending: Option<GenRequest>,
    rx: &mut Inbox<'_>,
) {
    let err = || {
        TokenEvent::Error(EngineError::overloaded(
            "the server is shutting down - retry the request",
        ))
    };
    for s in slots
        .iter_mut()
        .filter_map(Option::take)
        .chain(preempted.drain(..))
    {
        let _ = s.events.send(err());
    }
    for req in mm_pending.into_iter().chain(rx.drain()) {
        let _ = req.events.send(err());
    }
}

/// Did this stream just emit [`DEGENERATE_RUN`] id-0 tokens in a row? That is
/// what a non-finite logit row samples to: an argmax over NaN lands on index 0
/// (on Qwen's vocab, "!"). No real text spends 32 tokens on id 0 - repeated
/// punctuation merges into longer tokens - so the run is a numerical fault's
/// signature, not something a model writes.
pub(super) fn degenerate_run(history: &[u32], generated: usize) -> bool {
    generated >= DEGENERATE_RUN
        && history.len() >= DEGENERATE_RUN
        && history[history.len() - DEGENERATE_RUN..]
            .iter()
            .all(|&t| t == 0)
}

/// End a degenerate stream as an engine fault: a 500, so the caller (an SDK,
/// the batch run) retries it, rather than garbage streamed to max_tokens - a
/// 6000-token runaway holds a slot for minutes and still answers nothing.
pub(super) fn stop_degenerate(
    events: &tokio::sync::mpsc::UnboundedSender<TokenEvent>,
    pos: u32,
    generated: usize,
) {
    tracing::warn!(
        pos,
        generated,
        "degenerate stream: {DEGENERATE_RUN} consecutive id-0 tokens - non-finite logits; the \
         request is stopped as an engine fault"
    );
    let _ = events.send(TokenEvent::Error(EngineError::internal(
        "the engine produced an invalid token stream (non-finite logits) and stopped this \
         request; retrying it is safe",
    )));
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::super::{ErrorClass, admit};
    use super::*;
    use crate::metrics::EngineMetrics;
    use crate::sampler::SamplingParams;

    fn req(events: tokio::sync::mpsc::UnboundedSender<TokenEvent>) -> GenRequest {
        GenRequest {
            prompt: vec![1, 2, 3],
            max_tokens: 64,
            sampler: SamplingParams::default(),
            stop_tokens: Vec::new(),
            events,
            mm_chunks: None,
            constraint: None,
            logprobs: None,
            submitted: None,
            canvas_read: None,
            user_turn: false,
            background: false,
        }
    }

    fn overloaded(rx: &mut tokio::sync::mpsc::UnboundedReceiver<TokenEvent>) -> bool {
        let mut last = None;
        while let Ok(ev) = rx.try_recv() {
            last = Some(ev);
        }
        matches!(last, Some(TokenEvent::Error(e)) if e.class == ErrorClass::Overloaded)
    }

    /// A request whose events stop with no finish reads as an error, not as
    /// the end of the stream.
    #[tokio::test]
    async fn a_bare_close_reads_as_unfinished() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        tx.send(TokenEvent::Token {
            id: 7,
            logprobs: None,
        })
        .unwrap();
        drop(tx);
        assert!(matches!(
            next_event(&mut rx).await,
            Some(TokenEvent::Token { id: 7, .. })
        ));
        let Some(TokenEvent::Error(e)) = next_event(&mut rx).await else {
            panic!("a bare close must arrive as an error");
        };
        assert_eq!(e.class, ErrorClass::Internal);
        assert!(e.message.contains("without finishing"), "{}", e.message);
    }

    /// A live request and one still queued both end with a retryable error
    /// when the engine shuts down - not a bare close, which the surfaces
    /// reported as a finished reply.
    #[test]
    fn shutdown_fails_live_and_queued_requests_with_a_retryable_error() {
        let metrics = Arc::new(EngineMetrics::default());
        let (live_tx, mut live_rx) = tokio::sync::mpsc::unbounded_channel();
        let mut slots: Vec<Option<Slot>> = vec![None];
        struct Ctx;
        impl crate::generator::Generator for Ctx {
            fn reset(&mut self) {}
            fn forward(&mut self, _: u32) -> Result<Vec<f32>, crate::generator::GenError> {
                Ok(vec![0.])
            }
            fn vocab(&self) -> usize {
                1
            }
            fn max_context(&self) -> usize {
                4096
            }
        }
        assert!(admit(&mut slots, req(live_tx), &mut Ctx, &metrics));
        let (queue_tx, queue) = std::sync::mpsc::channel();
        let (waiting_tx, mut waiting_rx) = tokio::sync::mpsc::unbounded_channel();
        queue_tx.send(req(waiting_tx)).unwrap();
        fail_held_requests(&mut slots, &mut Vec::new(), None, &mut Inbox::new(&queue));
        assert!(slots.iter().all(Option::is_none), "the slot is released");
        assert!(overloaded(&mut live_rx), "the live request");
        assert!(overloaded(&mut waiting_rx), "the queued request");
    }
}
