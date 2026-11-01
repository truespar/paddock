//! Park/wake: the async restore flow.
//!
//! The interim tier restore BLOCKED admission in a bounded sleep loop -
//! correct, but it spent the batch's time waiting on IO. The park/wake
//! contract (KVFlow) is the opposite: a request whose prefix is `Loading`
//! is *skipped this tick* and the batch runs other work; the wake re-enters
//! admission. This module is that state machine, shared by every family:
//!
//! - The family's probe/elect front half (unchanged, family-flavored: aux
//!   boundaries, make-room pressure, truncation) produces a `TierHit` and
//!   optionally an [`AuxPlan`], then parks a [`RestoreFlow`] on the tier
//!   keyed by scheduler slot.
//! - The scheduler consults `Generator::tier_prefix_loading(slot, tokens)`
//!   before taking a prompt; a parked slot answers `Loading` and admission
//!   moves on. Reservation-first still holds: `begin_restore` seats the
//!   destination blocks before any IO starts.
//! - The per-pass `tier_pump` drives flows: the blocks round publishes into
//!   the radix, then (hybrid families) the aux round lands the state blob in
//!   a RESERVED checkpoint slot that attaches only once verified - the same
//!   two-round recipe the blocking path proved byte-exact, minus the sleeps.
//! - When the flow resolves, the next consult answers `Done`; the request
//!   re-enters admission and the family's normal resume path adopts the
//!   published prefix through an ordinary radix match. Nothing about the
//!   publication or attach contract changed - only who waits.
//!
//! Abandonment is graceful by construction: a flow past its park deadline
//! stops parking its request (recompute proceeds, bounded TTFT damage) but
//! keeps consuming its ticket's wake as a zombie so a late completion still
//! publishes for the next request and a reserved checkpoint slot is always
//! recycled. A slot reused by a different request (client cancelled while
//! parked) zombies the old flow the same way.

use std::time::Instant;

use cudarc::driver::CudaEvent;

use super::pool_tier::{AuxHit, PoolTier, RestoreWake, TicketId, TierHit, XferSink};
use crate::kv_pool::{BLOCK_TOKENS, KvPool};
use crate::paged_radix::PagedRadix;

/// How long a parked request waits before abandoning the restore and
/// recomputing. Generous versus the blocking path's `restore_deadline` -
/// nothing stalls while parked, so the bound only caps TTFT damage when the
/// IO path is sick (the breaker usually trips first).
pub fn park_deadline(est_us: f64) -> std::time::Duration {
    let us = (est_us * 8.0).clamp(20_000.0, 2_000_000.0);
    std::time::Duration::from_micros(us as u64)
}

/// The blocks round's deadline, bounded below by what abandoning would cost:
/// a flow that is abandoned recomputes the whole restorable span, so giving
/// up earlier than that recompute takes never saves time - it only turns a
/// late restore into a late restore PLUS a full prefill (3 of 20 restores
/// on a 9B, each a 26-32K prompt re-prefilled for 6-10 s, where the restore
/// was ~0.4 s from done; the estimate is 2-4x optimistic until it has
/// learned). Still capped, so a black-holed transport cannot hold a request
/// for a whole long prefill.
pub fn park_deadline_vs(est_us: f64, recompute_us: f64) -> std::time::Duration {
    let us = (est_us * 8.0)
        .max(recompute_us)
        .clamp(20_000.0, 10_000_000.0);
    std::time::Duration::from_micros(us as u64)
}

/// A zombie that never resolves (transport black-holed mid-op) is dropped
/// after this long; an aux round's reserved checkpoint slot leaks then -
/// bounded, logged, and strictly better than growing the zombie list.
const ZOMBIE_TTL: std::time::Duration = std::time::Duration::from_secs(30);

/// The hybrid round-two plan: which blob to restore and where checkpoint
/// slots live on the device (base + idx*stride), captured at flow start.
/// The device allocation outlives the flow - both live for the batch. A
/// family whose checkpoints live in pool pages leaves base and stride at 0:
/// the radix names each reserved checkpoint's pages instead.
pub struct AuxPlan {
    pub hit: AuxHit,
    pub state_base: u64,
    pub state_stride: u64,
}

enum FlowState {
    /// Round one in flight: runs restoring + publishing into the radix.
    Blocks { ticket: TicketId, deadline: Instant },
    /// Round two in flight: the state blob landing in reserved slot `cidx`.
    Aux {
        ticket: TicketId,
        cidx: u32,
        deadline: Instant,
    },
    /// Both rounds in flight at once: the blob loads into reserved slot
    /// `cidx` beside the runs and attaches once both have landed. Each round
    /// is a few scheduler pumps (submit, land, retire) whatever the disk
    /// does, so back to back they cost a restore twice the pump cadence -
    /// ~600 + ~350 ms on a Qwen3.8-27B session beside prefill.
    Both {
        blocks: TicketId,
        aux: TicketId,
        cidx: u32,
        deadline: Instant,
        blocks_wake: Option<RestoreWake>,
        aux_wake: Option<RestoreWake>,
    },
    /// Past deadline (or orphaned): unparked, still consuming the wake.
    Abandoned { ticket: TicketId, cidx: Option<u32> },
    /// A `Both` that will not attach: unparked, consuming whichever wakes
    /// are still out (None = consumed) before it returns slot `cidx` and the
    /// published path's pin - the blocks ticket pins its tip when it
    /// resolves, so letting go before it has would leak that pin.
    Draining {
        blocks: Option<TicketId>,
        aux: Option<TicketId>,
        cidx: u32,
    },
    /// Resolved. `ok` = the restored prefix is adoptable (published, and for
    /// hybrids the checkpoint attached); `at` GCs unclaimed entries.
    Done { ok: bool, at: Instant },
}

/// What a consult learns about its slot's flow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowStatus {
    /// No flow parked for this slot (or a stale one was zombied).
    None,
    /// Restore in flight - skip this request this tick (KVFlow).
    Loading,
    /// Resolved; the entry is consumed. Re-match the radix and proceed -
    /// on `ok` the prefix is there, otherwise recompute (never re-park).
    Done { ok: bool },
}

/// One parked restore: the two-round recipe as a pollable state machine.
pub struct RestoreFlow {
    /// The prefix under restore, through the boundary - flow identity for
    /// slot-reuse detection AND the attach path's radix walk.
    tokens: Vec<u32>,
    /// Success bar for round one: published depth must reach this.
    boundary_blocks: usize,
    aux: Option<AuxPlan>,
    state: FlowState,
    /// The blocks round's ticket - owner of the published path's pin (see
    /// [`PoolTier::hold_on_publish`]) from publication until the flow ends.
    blocks_ticket: TicketId,
    started: Instant,
    est_us: f64,
    bytes: u64,
    /// This restore reads mostly off disk - its completion trains the T2
    /// bandwidth EWMA, not the T1 one.
    from_nvme: bool,
}

impl RestoreFlow {
    /// Start round one (reservation-first - `begin_restore` seats the
    /// destination or refuses) and wrap it as a parked flow. `boundary` is
    /// `hit.end_block`; for hybrids the caller has already truncated the hit
    /// to the aux boundary.
    pub fn begin<T: XferSink>(
        tier: &mut PoolTier<T>,
        pool: &mut crate::kv_pool::KvPool,
        tokens: &[u32],
        hit: &TierHit,
        aux: Option<AuxPlan>,
        est_us: f64,
        after: Option<CudaEvent>,
    ) -> Option<RestoreFlow> {
        let ticket = match tier.begin_restore(hit, tokens, pool, after) {
            Some(t) => {
                tier.note_park(true);
                if aux.is_some() {
                    tier.hold_on_publish(t);
                }
                t
            }
            None => {
                // reservation-first refused: the block pool could not seat
                // the destination. A capacity story, and counted as one -
                // it must never read as an IO failure.
                tier.note_park(false);
                return None;
            }
        };
        let cut = (hit.end_block * BLOCK_TOKENS).min(tokens.len());
        // what the flow moves: the blob rides with the blocks
        let bytes = hit.bytes + aux.as_ref().map_or(0, |p| p.hit.bytes);
        let nvme = hit.nvme_bytes + aux.as_ref().map_or(0, |p| p.hit.nvme_bytes);
        let recompute_us = tier.cost.recompute_us(cut as u32);
        Some(RestoreFlow {
            tokens: tokens[..cut].to_vec(),
            boundary_blocks: hit.end_block,
            aux,
            state: FlowState::Blocks {
                ticket,
                deadline: Instant::now() + park_deadline_vs(est_us, recompute_us),
            },
            blocks_ticket: ticket,
            started: Instant::now(),
            est_us,
            bytes,
            // dominant source decides which tier's rate this completion
            // teaches - a mostly-disk restore is a disk measurement
            from_nvme: nvme * 2 > bytes,
        })
    }

    /// True while a consult should keep its request parked.
    fn loading(&self) -> bool {
        matches!(
            self.state,
            FlowState::Blocks { .. } | FlowState::Aux { .. } | FlowState::Both { .. }
        )
    }

    /// Reserve the checkpoint slot and start the blob round. None when no
    /// slot can be had or the round cannot start (the slot is returned).
    fn start_aux<T: XferSink>(
        &self,
        tier: &mut PoolTier<T>,
        pr: &mut PagedRadix,
        pool: Option<&mut KvPool>,
        after: &mut dyn FnMut() -> Option<CudaEvent>,
    ) -> Option<(TicketId, u32)> {
        let plan = self.aux.as_ref()?;
        // paged checkpoints draw their pages here; a flat pool (or a paged
        // one pumped without its pool) takes a slot
        let cidx = match pool {
            Some(pool) if pr.state_is_paged() => pr.reserve_state_slot_with_pool(pool),
            _ => pr.reserve_state_slot(),
        }?;
        // a blob coming off disk is seated back in T1 whole
        tier.expect_aux_fill(&plan.hit, &self.tokens);
        let started = if pr.state_is_paged() {
            let pages = pr.state_pages(cidx).to_vec();
            tier.begin_restore_aux_paged(&plan.hit, &pages, after())
        } else {
            let dst = plan.state_base + cidx as u64 * plan.state_stride;
            tier.begin_restore_aux(&plan.hit, dst, after())
        };
        match started {
            Some(tk) => Some((tk, cidx)),
            None => {
                pr.recycle_state(cidx);
                None
            }
        }
    }

    fn done(&self) -> Option<(bool, Instant)> {
        match self.state {
            FlowState::Done { ok, at } => Some((ok, at)),
            _ => None,
        }
    }

    /// Does this flow belong to a request whose prompt starts with the
    /// prefix under restore? (Slot reuse by an unrelated request fails this.)
    fn matches(&self, tokens: &[u32]) -> bool {
        tokens.len() >= self.tokens.len() && tokens[..self.tokens.len()] == self.tokens[..]
    }

    fn finish<T: XferSink>(&mut self, tier: &mut PoolTier<T>, ok: bool) {
        // attached (or given up): the checkpoint, if any, now keeps the path
        tier.release_hold(self.blocks_ticket);
        let el = self.started.elapsed().as_micros() as f64;
        tier.cost
            .observe_restore_from(self.bytes, el, self.est_us, self.from_nvme);
        tier.note_flow_end(ok, self.bytes, self.from_nvme);
        tracing::debug!(
            ok,
            boundary = self.boundary_blocks,
            elapsed_us = el,
            est_us = self.est_us,
            "tier flow resolved"
        );
        self.state = FlowState::Done {
            ok,
            at: Instant::now(),
        };
    }

    /// Advance one step. The caller has already run `pump_completions`;
    /// this only claims wakes and transitions. `after` records a compute-
    /// stream fence for the aux round's H2D.
    fn pump<T: XferSink>(
        &mut self,
        tier: &mut PoolTier<T>,
        pr: &mut PagedRadix,
        mut pool: Option<&mut KvPool>,
        after: &mut dyn FnMut() -> Option<CudaEvent>,
    ) {
        match self.state {
            FlowState::Blocks { ticket, deadline } => {
                // start the blob round now, beside the runs (`Both`); if no
                // slot can be had yet, the sequential path below retries
                // once the runs land
                if self.aux.is_some()
                    && Instant::now() < deadline
                    && let Some((aux, cidx)) = self.start_aux(tier, pr, pool.as_deref_mut(), after)
                {
                    self.state = FlowState::Both {
                        blocks: ticket,
                        aux,
                        cidx,
                        deadline,
                        blocks_wake: None,
                        aux_wake: None,
                    };
                    return;
                }
                // (the blob round below inherits this deadline: `est_us`
                // already prices the blob, so the bound stays per flow)
                if let Some(w) = tier.take_wake(ticket) {
                    if self.aux.is_none() {
                        // KV-only family (2.1-lite): any published depth is
                        // an adoptable prefix - the resume matches whatever
                        // landed and recomputes the rest. Only a zero-depth
                        // failure reports false.
                        self.finish(tier, w.ok && w.end_block > 0);
                        return;
                    }
                    if !(w.ok && w.end_block >= self.boundary_blocks) {
                        // hybrid: blocks below the boundary are unusable
                        // without their state blob - partial publication
                        // still serves later requests, but this flow's
                        // boundary was not reached
                        self.finish(tier, false);
                        return;
                    }
                    let Some(plan) = self.aux.as_ref() else {
                        unreachable!("aux checked above");
                    };
                    // the flow's own deadline, or the blob's time at the
                    // measured rate if the blocks round ate the budget. A
                    // fixed 16 GB/s guess here abandoned every Qwen3.8
                    // blob round once its checksums left the tick
                    let rate = tier.cost.rates_bpus().0.max(1.0);
                    let aux_deadline =
                        deadline.max(Instant::now() + park_deadline(plan.hit.bytes as f64 / rate));
                    match self.start_aux(tier, pr, pool, after) {
                        Some((tk, cidx)) => {
                            self.state = FlowState::Aux {
                                ticket: tk,
                                cidx,
                                deadline: aux_deadline,
                            };
                        }
                        None => self.finish(tier, false),
                    }
                } else if Instant::now() >= deadline {
                    tracing::debug!(ticket, "tier flow abandoned (blocks round deadline)");
                    tier.note_flow_end(false, self.bytes, self.from_nvme);
                    self.state = FlowState::Abandoned { ticket, cidx: None };
                }
            }
            FlowState::Aux {
                ticket,
                cidx,
                deadline,
            } => {
                if let Some(w) = tier.take_wake(ticket) {
                    let attached = w.ok
                        && pr.attach_state_at(
                            &self.tokens,
                            self.boundary_blocks * BLOCK_TOKENS,
                            cidx,
                        );
                    if !attached {
                        tracing::debug!(
                            ticket,
                            blob_landed = w.ok,
                            boundary = self.boundary_blocks,
                            "tier flow: checkpoint not attached"
                        );
                        pr.recycle_state(cidx);
                    }
                    self.finish(tier, attached);
                } else if Instant::now() >= deadline {
                    tracing::debug!(ticket, "tier flow abandoned (aux round deadline)");
                    tier.note_flow_end(false, self.bytes, self.from_nvme);
                    self.state = FlowState::Abandoned {
                        ticket,
                        cidx: Some(cidx),
                    };
                }
            }
            FlowState::Both {
                blocks,
                aux,
                cidx,
                deadline,
                mut blocks_wake,
                mut aux_wake,
            } => {
                if blocks_wake.is_none() {
                    blocks_wake = tier.take_wake(blocks);
                }
                if aux_wake.is_none() {
                    aux_wake = tier.take_wake(aux);
                }
                let boundary = self.boundary_blocks;
                let reached = |w: &RestoreWake| w.ok && w.end_block >= boundary;
                match (blocks_wake, aux_wake) {
                    (Some(b), Some(a)) => {
                        let attached = reached(&b)
                            && a.ok
                            && pr.attach_state_at(
                                &self.tokens,
                                self.boundary_blocks * BLOCK_TOKENS,
                                cidx,
                            );
                        if !attached {
                            tracing::debug!(
                                blocks_reached = reached(&b),
                                blob_landed = a.ok,
                                boundary = self.boundary_blocks,
                                "tier flow: checkpoint not attached"
                            );
                            pr.recycle_state(cidx);
                        }
                        self.finish(tier, attached);
                    }
                    // the runs fell short of the boundary: the blob has
                    // nothing to attach to - unpark now, and let its wake
                    // return the slot when it lands
                    (Some(b), None) if !reached(&b) => {
                        tier.note_flow_end(false, self.bytes, self.from_nvme);
                        self.state = FlowState::Draining {
                            blocks: None,
                            aux: Some(aux),
                            cidx,
                        };
                    }
                    (b, a) if Instant::now() >= deadline => {
                        tracing::debug!(blocks, aux, "tier flow abandoned (both rounds' deadline)");
                        tier.note_flow_end(false, self.bytes, self.from_nvme);
                        self.state = FlowState::Draining {
                            blocks: b.is_none().then_some(blocks),
                            aux: a.is_none().then_some(aux),
                            cidx,
                        };
                    }
                    _ => {
                        self.state = FlowState::Both {
                            blocks,
                            aux,
                            cidx,
                            deadline,
                            blocks_wake,
                            aux_wake,
                        };
                    }
                }
            }
            FlowState::Draining {
                mut blocks,
                mut aux,
                cidx,
            } => {
                if blocks.is_some_and(|t| tier.take_wake(t).is_some()) {
                    blocks = None;
                }
                if aux.is_some_and(|t| tier.take_wake(t).is_some()) {
                    aux = None;
                }
                if blocks.is_none() && aux.is_none() {
                    let el = self.started.elapsed().as_micros() as f64;
                    tier.cost
                        .observe_restore_from(self.bytes, el, self.est_us, self.from_nvme);
                    pr.recycle_state(cidx);
                    tier.release_hold(self.blocks_ticket);
                    self.state = FlowState::Done {
                        ok: false,
                        at: Instant::now(),
                    };
                } else if self.started.elapsed() > ZOMBIE_TTL {
                    tracing::warn!(
                        cidx,
                        "tier zombie flow dropped - checkpoint slot leaked (bounded)"
                    );
                    tier.release_hold(self.blocks_ticket);
                    self.state = FlowState::Done {
                        ok: false,
                        at: Instant::now(),
                    };
                } else {
                    self.state = FlowState::Draining { blocks, aux, cidx };
                }
            }
            FlowState::Abandoned { ticket, cidx } => {
                if tier.take_wake(ticket).is_some() {
                    // the restore did finish, just late: teach the cost
                    // model what it really took, or it keeps electing on
                    // the estimate that was blown
                    let el = self.started.elapsed().as_micros() as f64;
                    tier.cost
                        .observe_restore_from(self.bytes, el, self.est_us, self.from_nvme);
                    // a late blocks-round success already published via the
                    // ticket's own resolution; only the reservation needs us
                    if let Some(c) = cidx {
                        pr.recycle_state(c);
                    }
                    tier.release_hold(self.blocks_ticket);
                    self.state = FlowState::Done {
                        ok: false,
                        at: Instant::now(),
                    };
                } else if self.started.elapsed() > ZOMBIE_TTL {
                    if let Some(c) = cidx {
                        tracing::warn!(
                            ticket,
                            cidx = c,
                            "tier zombie flow dropped - checkpoint slot leaked (bounded)"
                        );
                    }
                    tier.release_hold(self.blocks_ticket);
                    self.state = FlowState::Done {
                        ok: false,
                        at: Instant::now(),
                    };
                }
            }
            FlowState::Done { .. } => {}
        }
    }
}

/// Flow parking on the tier - one entry per scheduler slot plus a zombie
/// list. Owned by [`PoolTier`] so families need no new storage.
#[derive(Default)]
pub(super) struct FlowPark {
    active: std::collections::HashMap<usize, RestoreFlow>,
    zombies: Vec<RestoreFlow>,
    /// Consults keeping their request parked without a flow, per slot.
    waits: std::collections::HashMap<usize, ConsultWait>,
}

/// How often a waiting consult tries again. The scheduler re-enters a
/// parked slot every pass - 100 us apart when nothing decodes - and each
/// try probes (and, waiting for room, presses retention) again.
const CONSULT_RETRY: std::time::Duration = std::time::Duration::from_millis(5);
/// The longest a consult waits, whatever the recompute costs.
const CONSULT_WAIT_CAP_US: f64 = 10_000_000.0;
/// The least a wait must be worth. Its tries come once a scheduler pass -
/// 0.1-0.2 s beside decode on a 27B, longer beside a prefill chunk - so a
/// wait that cannot outlast a few passes only adds one (live: a 128-token
/// gap's first retry came 144 ms in, past the 130 ms the gap was worth).
const CONSULT_WAIT_MIN_US: f64 = 500_000.0;
/// A wait no try has touched this long past its end is stale (its request
/// went another way - a probe that missed, a reused slot).
const CONSULT_WAIT_STALE: std::time::Duration = std::time::Duration::from_secs(1);

/// What a waiting consult waits for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WaitFor {
    /// Block-pool room for an elected restore ([`PoolTier::wait_for_pool`]).
    PoolRoom,
    /// Deeper state still on its way into T1 ([`PoolTier::wait_for_blob`]).
    BlobLanding,
}

struct ConsultWait {
    why: WaitFor,
    t_len: usize,
    since: Instant,
    until: Instant,
    next: Instant,
    /// The ledger before this try. A retry asks the same question again, so
    /// its lookup is the first try's, not another one.
    mark: super::accounting::TierDecisions,
}

/// What a consult does about a slot whose request is waiting (see
/// [`PoolTier::consult_try`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsultTry {
    /// No wait: an ordinary consult.
    Fresh,
    /// Too soon to try again - keep the request parked.
    Hold,
    /// Try again now, without blocking on the pool: the passes between
    /// tries are the wait.
    Retry,
}

impl<T: XferSink> PoolTier<T> {
    /// The consult's flow lookup. Consuming: `Done` removes the entry, so
    /// the caller must act on it (re-match and proceed, never re-park). A
    /// flow whose prefix does not match `tokens` (slot reused) is zombied
    /// and the answer is `None`.
    pub fn flow_status(&mut self, slot: usize, tokens: &[u32]) -> FlowStatus {
        let Some(f) = self.flows.active.get(&slot) else {
            return FlowStatus::None;
        };
        if !f.matches(tokens) {
            let f = self.flows.active.remove(&slot).expect("present");
            if f.loading()
                || matches!(
                    f.state,
                    FlowState::Abandoned { .. } | FlowState::Draining { .. }
                )
            {
                self.flows.zombies.push(f);
            }
            return FlowStatus::None;
        }
        match f.done() {
            Some((ok, _)) => {
                self.flows.active.remove(&slot);
                FlowStatus::Done { ok }
            }
            None if f.loading() => FlowStatus::Loading,
            // abandoned: unpark as a FAILED result - the request admits and
            // recomputes while the zombie keeps draining. Reporting `None`
            // here let the consult immediately elect a FRESH restore of the
            // same content every deadline period; the live probe stacked
            // four tickets' destination reservations that way and starved
            // the pool.
            None => {
                let f = self.flows.active.remove(&slot).expect("present");
                self.flows.zombies.push(f);
                FlowStatus::Done { ok: false }
            }
        }
    }

    /// Park a freshly begun flow for `slot`. An abandoned predecessor moves
    /// to the zombie list rather than being clobbered.
    pub fn park_flow(&mut self, slot: usize, flow: RestoreFlow) {
        if let Some(w) = self.flows.waits.remove(&slot) {
            // seated after waiting: one lookup, however many tries
            self.dec.rewind_lookup(&w.mark);
            tracing::debug!(
                slot,
                waited_ms = w.since.elapsed().as_millis() as u64,
                why = ?w.why,
                "tier: restore parked after its consult waited"
            );
        }
        if let Some(old) = self.flows.active.insert(slot, flow)
            && !matches!(old.state, FlowState::Done { .. })
        {
            self.flows.zombies.push(old);
        }
    }

    /// Drive every parked flow one step. Call right after `pump_completions`
    /// from the family's `tier_pump` - the per-pass slot the service loop
    /// already runs, which is what turns a wake into a re-admission.
    pub fn pump_flows(
        &mut self,
        pr: &mut PagedRadix,
        after: &mut dyn FnMut() -> Option<CudaEvent>,
    ) {
        self.pump_flows_in(pr, None, after);
    }

    /// [`Self::pump_flows`] for a family whose checkpoints live in pool pages
    /// (issue #33): an aux round reserves its checkpoint's pages from `pool`
    /// before the blob lands in them.
    pub fn pump_flows_with_pool(
        &mut self,
        pr: &mut PagedRadix,
        pool: &mut KvPool,
        after: &mut dyn FnMut() -> Option<CudaEvent>,
    ) {
        self.pump_flows_in(pr, Some(pool), after);
    }

    fn pump_flows_in(
        &mut self,
        pr: &mut PagedRadix,
        mut pool: Option<&mut KvPool>,
        after: &mut dyn FnMut() -> Option<CudaEvent>,
    ) {
        let mut park = std::mem::take(&mut self.flows);
        for f in park.active.values_mut() {
            f.pump(self, pr, pool.as_deref_mut(), after);
        }
        for f in park.zombies.iter_mut() {
            f.pump(self, pr, pool.as_deref_mut(), after);
        }
        park.zombies
            .retain(|f| !matches!(f.state, FlowState::Done { .. }));
        // a pin outlives its flow only when the flow went without finishing
        // (GC'd unclaimed, or dropped as a zombie before its wake): let go
        let owned = |t: &TicketId| {
            park.active
                .values()
                .chain(park.zombies.iter())
                .any(|f| f.blocks_ticket == *t && f.done().is_none())
        };
        let stray: Vec<TicketId> = self.holds.keys().filter(|t| !owned(t)).copied().collect();
        for t in stray {
            self.release_hold(t);
        }
        // GC resolved entries nobody consulted (requester cancelled while
        // parked) - normally the consult claims Done within a tick
        park.active.retain(|_, f| match f.done() {
            Some((_, at)) => at.elapsed() < std::time::Duration::from_secs(2),
            None => true,
        });
        self.flows = park;
    }

    /// The consult's next question after [`Self::flow_status`]: is `slot`'s
    /// request waiting ([`Self::wait_for_pool`], [`Self::wait_for_blob`])?
    pub fn consult_try(&mut self, slot: usize, t_len: usize) -> ConsultTry {
        let now = Instant::now();
        let dec = self.dec;
        match self.flows.waits.get_mut(&slot) {
            Some(w) if w.t_len == t_len && now < w.until + CONSULT_WAIT_STALE => {
                if now < w.next {
                    return ConsultTry::Hold;
                }
                w.mark = dec;
                ConsultTry::Retry
            }
            Some(_) => {
                self.flows.waits.remove(&slot);
                ConsultTry::Fresh
            }
            None => ConsultTry::Fresh,
        }
    }

    /// An elected restore the block pool cannot seat right now. Waiting beats
    /// recomputing while the wait plus the restore still costs less than the
    /// prefill it saves (`restore_tokens` of it): the pool is full of live
    /// sequences, the next ones to finish free it, and a recompute needs the
    /// same blocks - it only starts sooner, a chunk at a time. Refusing
    /// outright made a 19.5K-token turn whose state sat on the tier recompute
    /// in full, 14.6 s, because a restore beside it had just taken the free
    /// blocks. True: keep the request parked and consult again; false: give
    /// up (a park refusal) and recompute.
    pub fn wait_for_pool(
        &mut self,
        slot: usize,
        t_len: usize,
        est_us: f64,
        restore_tokens: usize,
    ) -> bool {
        let slack_us = self.cost.recompute_us(restore_tokens as u32) - est_us;
        if self.wait_while(slot, t_len, WaitFor::PoolRoom, slack_us) {
            return true;
        }
        self.refuse_park();
        false
    }

    /// Deeper state for this prompt is still being stored into T1
    /// (`blob_landing` found it, `gap_tokens` past what is restorable now,
    /// and started its stores). Waiting for it beats restoring the shallower
    /// boundary while the wait costs less than re-prefilling the gap. True:
    /// keep the request parked and consult again; false: stop waiting and
    /// restore what is there.
    pub fn wait_for_blob(&mut self, slot: usize, t_len: usize, gap_tokens: usize) -> bool {
        let slack_us = self.cost.recompute_us(gap_tokens as u32);
        self.wait_while(slot, t_len, WaitFor::BlobLanding, slack_us)
    }

    /// Keep `slot`'s request parked for `why` while `slack_us` (counted from
    /// the first try of this wait) has not run out. False once it has, or
    /// when the wait would never pay - and the wait is forgotten. A retry's
    /// lookup counts are handed back: the first try's stand.
    fn wait_while(&mut self, slot: usize, t_len: usize, why: WaitFor, slack_us: f64) -> bool {
        let now = Instant::now();
        let mut since = now;
        if let Some(w) = self.flows.waits.get_mut(&slot)
            && w.t_len == t_len
        {
            self.dec.rewind_lookup(&w.mark);
            since = w.since;
            if w.why == why {
                if now < w.until {
                    w.next = now + CONSULT_RETRY;
                    return true;
                }
                let waited_ms = w.since.elapsed().as_millis() as u64;
                self.flows.waits.remove(&slot);
                tracing::debug!(slot, waited_ms, why = ?why, "tier: consult stopped waiting");
                return false;
            }
            // the same request, now waiting on something else (the time
            // it waited already still counts as its wait)
            self.flows.waits.remove(&slot);
        }
        let slack_us = slack_us.min(CONSULT_WAIT_CAP_US);
        if slack_us < CONSULT_WAIT_MIN_US {
            return false;
        }
        tracing::debug!(
            slot,
            t_len,
            why = ?why,
            wait_up_to_ms = (slack_us / 1000.0) as u64,
            "tier: consult waits"
        );
        self.flows.waits.insert(
            slot,
            ConsultWait {
                why,
                t_len,
                since,
                until: now + std::time::Duration::from_micros(slack_us as u64),
                next: now + CONSULT_RETRY,
                mark: self.dec,
            },
        );
        true
    }

    /// Open flows (parked + zombies) - test/stat witness.
    pub fn flow_count(&self) -> usize {
        self.flows.active.len() + self.flows.zombies.len()
    }
}
