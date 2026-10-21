//! Who gets a slot: live requests before background work.
//!
//! The manager's batch queue sends its lines with `x-paddock-origin: batch`
//! and the runner marks those requests `background` (`GenRequest`). Three
//! rules follow, all here so the scheduler loop only calls them:
//!
//! - admission order: a waiting live request is admitted before any waiting
//!   background one, whatever order they arrived in ([`Inbox`]);
//! - preemption: a live request that finds every slot taken takes a
//!   background slot, which goes back to the queue and is recomputed from its
//!   own history later ([`preempt_for_live`]) - the same recompute the
//!   pool-pressure path has always used, so its output does not change;
//! - pool pressure takes a background victim before a live one
//!   ([`pressure_victim`]).
//!
//! So a batch running flat out costs an interactive request at most a tick of
//! queueing, and the batch still gets every slot live traffic leaves idle.
//! What background work cannot do is starve: nothing here drops it, it only
//! waits while live requests are waiting.

use std::collections::{HashSet, VecDeque};
use std::sync::mpsc::{Receiver, RecvTimeoutError, TryRecvError};
use std::time::Duration;

use super::{EngineError, GenRequest, Generator, Slot, TokenEvent, abort_backend_chunks};

/// The scheduler's request queue: the engine channel drained into a live and
/// a background queue. `try_recv` / `recv_timeout` mirror the channel's own
/// methods (same error types), so the loop reads it exactly as it read the
/// channel; the only difference is which request comes out first.
pub(super) struct Inbox<'a> {
    rx: &'a Receiver<GenRequest>,
    live: VecDeque<GenRequest>,
    background: VecDeque<GenRequest>,
    closed: bool,
}

impl<'a> Inbox<'a> {
    pub(super) fn new(rx: &'a Receiver<GenRequest>) -> Self {
        Inbox {
            rx,
            live: VecDeque::new(),
            background: VecDeque::new(),
            closed: false,
        }
    }

    fn file(&mut self, req: GenRequest) {
        if req.background {
            self.background.push_back(req);
        } else {
            self.live.push_back(req);
        }
    }

    /// Move whatever the channel holds into the two queues.
    fn pump(&mut self) {
        loop {
            match self.rx.try_recv() {
                Ok(req) => self.file(req),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.closed = true;
                    break;
                }
            }
        }
    }

    fn pop(&mut self) -> Option<GenRequest> {
        self.live
            .pop_front()
            .or_else(|| self.background.pop_front())
    }

    /// `Receiver::try_recv`, live requests first. Disconnected only once the
    /// senders are gone AND nothing is left queued.
    pub(super) fn try_recv(&mut self) -> Result<GenRequest, TryRecvError> {
        self.pump();
        match self.pop() {
            Some(req) => Ok(req),
            None if self.closed => Err(TryRecvError::Disconnected),
            None => Err(TryRecvError::Empty),
        }
    }

    /// `Receiver::recv_timeout`, live requests first.
    pub(super) fn recv_timeout(
        &mut self,
        timeout: Duration,
    ) -> Result<GenRequest, RecvTimeoutError> {
        if let Ok(req) = self.try_recv() {
            return Ok(req);
        }
        if self.closed {
            return Err(RecvTimeoutError::Disconnected);
        }
        let first = self.rx.recv_timeout(timeout)?;
        self.file(first);
        // whatever landed in the same burst competes on class, not arrival
        self.pump();
        Ok(self.pop().expect("a request was just filed"))
    }

    /// Live requests waiting for a slot.
    pub(super) fn live_waiting(&mut self) -> usize {
        self.pump();
        self.live.len()
    }

    /// Everything still waiting, live first (shutdown fails them all).
    pub(super) fn drain(&mut self) -> impl Iterator<Item = GenRequest> + '_ {
        self.pump();
        self.live.drain(..).chain(self.background.drain(..))
    }
}

/// May this slot give its place to a live request? Background, no chunk in
/// flight, and either
/// - decoding text: a picture's rows are not in the token history, so a
///   recompute could not rebuild them - a picture slot keeps its place and
///   the live request takes the next slot that frees; or
/// - not started (its prompt or picture still untouched - held while live
///   work runs, see `hold_bg`): nothing to rebuild, it just queues again.
///
/// A prompt the picture encoder or a chunk already took is neither.
fn preemptible(s: &Slot, k: usize, chunking: &HashSet<usize>) -> bool {
    if !s.background || chunking.contains(&k) {
        return false;
    }
    if s.prefilled {
        s.pos as usize <= s.history.len()
    } else {
        !s.prompt.is_empty() || s.mm.is_some()
    }
}

/// A live request is waiting, no slot is free, and a background slot could
/// give it one: a pipelined segment (which never passes the tick top) should
/// drain so the tick top can make the swap.
pub(super) fn live_blocked(
    slots: &[Option<Slot>],
    chunking: &HashSet<usize>,
    inbox: &mut Inbox<'_>,
) -> bool {
    let free = slots.iter().filter(|s| s.is_none()).count();
    inbox.live_waiting() > free
        && slots
            .iter()
            .enumerate()
            .any(|(k, s)| s.as_ref().is_some_and(|s| preemptible(s, k, chunking)))
}

/// Make room for waiting live requests: while more of them wait than there
/// are free slots, take the background slot that is cheapest to rebuild
/// (fewest KV rows) and queue it for recompute behind them. Runs at the tick
/// top, where no pipelined tick is in flight; the reconcile right after
/// returns the victims' KV blocks before admission. Returns how many slots it
/// freed.
pub(super) fn preempt_for_live(
    slots: &mut [Option<Slot>],
    preempted: &mut Vec<Slot>,
    chunking: &HashSet<usize>,
    inbox: &mut Inbox<'_>,
) -> usize {
    let waiting = inbox.live_waiting();
    if waiting == 0 {
        return 0;
    }
    let free = slots.iter().filter(|s| s.is_none()).count();
    let mut freed = 0;
    while free + freed < waiting {
        let victim = (0..slots.len())
            .filter(|&k| {
                slots[k]
                    .as_ref()
                    .is_some_and(|s| preemptible(s, k, chunking))
            })
            .min_by_key(|&k| slots[k].as_ref().map_or(u32::MAX, |s| s.pos));
        let Some(k) = victim else { break };
        let need = slots[k].as_mut().expect("present").preempt_for_recompute();
        preempted.push(slots[k].take().expect("present"));
        freed += 1;
        tracing::debug!(
            "serve: background slot {k} yields to a live request ({need} blocks to recompute)"
        );
    }
    freed
}

/// Which preempted sequence resumes next: a live one first, then background
/// ones in queue order - but no background one while a live request is still
/// waiting, or the slot it just gave up would go straight back to it.
pub(super) fn next_resume(preempted: &[Slot], inbox: &mut Inbox<'_>) -> Option<usize> {
    if let Some(i) = preempted.iter().position(|s| !s.background) {
        return Some(i);
    }
    (!preempted.is_empty() && inbox.live_waiting() == 0).then_some(0)
}

/// The pool-pressure victim among `slots`: the NEWEST background sequence
/// that is decoding, else the newest of any (the long-standing rule).
/// Preempting the newest keeps recompute sequences, which re-admit into low
/// slots, off the target - no livelock; a background victim must be
/// prefilled for the same reason.
pub(super) fn pressure_victim(
    slots: &[Option<Slot>],
    eligible: impl Fn(usize) -> bool,
) -> Option<usize> {
    (0..slots.len())
        .rev()
        .find(|&k| {
            eligible(k)
                && slots[k]
                    .as_ref()
                    .is_some_and(|s| s.background && s.prefilled)
        })
        .or_else(|| {
            (0..slots.len())
                .rev()
                .find(|&k| eligible(k) && slots[k].is_some())
        })
}

/// G4b mixed-tick preemption: a mixed (chunked-prefill + decode) pass couldn't
/// grow the budget pool. Preempt a DECODE slot (`pressure_victim`) - never a
/// chunking slot, whose prefill is making progress - so the retried tick fits
/// with its blocks freed; `chunking` stays intact and the reconcile at the next
/// tick top returns the victim's blocks. No KV half-advanced: a mixed pass hits
/// PoolExhausted inside ensure_pool_rows, before run_layers writes anything. If
/// every active slot is mid-prefill (no decode victim to relieve the pool - a
/// watermark-should-prevent pathological case), fail the stuck chunking slots so
/// the server recovers instead of spinning.
pub(super) fn preempt_or_fail_mixed(
    generator: &mut dyn Generator,
    slots: &mut [Option<Slot>],
    preempted: &mut Vec<Slot>,
    chunking: &mut HashSet<usize>,
    pool_stats: bool,
) {
    match pressure_victim(slots, |k| !chunking.contains(&k)) {
        Some(k) => {
            let need = slots[k].as_mut().expect("present").preempt_for_recompute();
            preempted.push(slots[k].take().expect("present"));
            if pool_stats {
                tracing::warn!(
                    "pool: PREEMPTED slot {k} (mixed) for recompute ({need} blocks), {} queued",
                    preempted.len()
                );
            }
        }
        None => {
            let stuck: Vec<usize> = chunking.iter().copied().collect();
            tracing::warn!(
                "serve: mixed pool-exhausted with no decode victim; failing {} chunking slot(s)",
                stuck.len()
            );
            for k in stuck {
                if let Some(s) = slots[k].take() {
                    let _ = s.events.send(TokenEvent::Error(EngineError::overloaded(
                        "the KV cache pool is exhausted under the current load; retry shortly",
                    )));
                }
            }
            abort_backend_chunks(generator, chunking);
        }
    }
}
