//! `TierCatalog` - replica records, reservations, single-flight promotions and
//! the per-tier byte ledger.
//!
//! This is a DETERMINISTIC transactional core: no threads, no async, no time.
//! The serving layer calls methods; the transport's completions are events fed
//! through [`TierCatalog::on_completion`]. Every method either commits a whole
//! transition or changes nothing, and `check_invariants` can prove the whole
//! structure consistent after any prefix of events - which is what the race
//! suite in `tests.rs` does, exhaustively, before any CUDA or disk exists.
//!
//! Replica states follow the plan's machine. `Reading(op, generation, pins)` is
//! represented as `Ready { pins > 0 }` with the reading ops in the op table -
//! same machine, but N concurrent readers stay representable and the op ids
//! live in one place:
//!
//! ```text
//! Absent ── reserve ──▶ Reserved ── begin_store ──▶ Writing(op, generation)
//!    ▲                     │ release                    │ complete: bytes+checksum verified
//!    │                     ▼                            ▼        (else -> Absent, counted)
//!    └── evict ──────── (Absent) ◀── cancel_store ── Ready(generation, loc) ⇄ pins via loads
//!                                                       │ integrity failure on read
//!                                                       ▼
//!                                                      Bad (bytes released, marker kept)
//! ```
//!
//! Hard rules the tests enforce:
//! - reserve the destination before any I/O starts; a load must never finish
//!   into an exhausted pool (reservation-first).
//! - publish `Ready` only after producer event + full byte count + integrity
//!   check.
//! - source stays pinned for the whole op.
//! - completions validate op id + generation - a late completion can never
//!   publish into a new owner.
//! - abort/completion are idempotent and release every reservation exactly once.
//! - duplicate loads are single-flight with a waiter list; each waiter may
//!   still independently cancel (elect recompute) without tearing down the op
//!   for the others.

use std::collections::HashMap;

use super::digest::{Checksum, LogicalKey};
use super::transport::{IoCompletion, IoJob, IoJobKind, IoOutcome};
use super::{Gen, LoadDst, Loc, OpId, Tier, WaiterId};

/// Capacity per tier, bytes. Zero disables a tier (every reserve fails).
#[derive(Debug, Clone, Copy)]
pub struct TierCatalogConfig {
    pub ram_capacity: u64,
    pub nvme_capacity: u64,
}

/// Byte ledger of one tier - the buckets, tracked separately so admission
/// and observability never have to infer them. `pinned` counts bytes of Ready
/// replicas currently pinned by at least one read (a subset of `ready`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Ledger {
    pub capacity: u64,
    pub free: u64,
    pub reserved: u64,
    pub in_flight: u64,
    pub ready: u64,
    pub pinned: u64,
}

/// Decision-accountability counters: every path that swallows an
/// event or degrades service counts it, so "0% hit rate for months" class
/// regressions are structurally impossible to miss.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Counters {
    /// Completion for an op no longer in the table (cancelled, already
    /// resolved, or duplicated). Harmless by design; a RATE of them is a
    /// transport bug.
    pub stale_completions: u64,
    /// Completion whose shape contradicts its op (a Load answered with
    /// StoreDone, a missing dst loc). Always a bug somewhere below us.
    pub protocol_errors: u64,
    pub short_writes: u64,
    /// Checksum mismatches (store-time or at-rest read).
    pub integrity_failures: u64,
    /// Transport-reported failures.
    pub io_failures: u64,
    /// Stores that found no room in the tier's memory (capacity, not faults).
    pub no_room: u64,
    pub single_flight_joins: u64,
    pub waiter_cancels: u64,
    pub op_cancels: u64,
    pub evictions: u64,
    pub bad_marks: u64,
}

#[derive(Debug)]
enum ReplicaState {
    /// Capacity held by admission; no bytes, no op yet.
    Reserved,
    /// Store (or load-into-this-tier) in flight.
    Writing { op: OpId, generation: Gen },
    /// Published: bytes at `loc`, integrity-checked, readable. `pins` = live
    /// read ops (the plan's `Reading` when > 0).
    Ready {
        generation: Gen,
        loc: Loc,
        checksum: Checksum,
        pins: u32,
    },
    /// At-rest corruption discovered; bytes already released. Kept as a
    /// marker (negative cache + diagnostics) until reserved over or evicted.
    Bad,
}

#[derive(Debug)]
struct Replica {
    /// Exact payload bytes (the reservation bound until `begin_store` trims
    /// it to the sealed size).
    bytes: u64,
    state: ReplicaState,
}

#[derive(Debug)]
enum OpKind {
    /// `expected: None` is the adopt-on-first-write mode (`begin_store_adopt`):
    /// the transport's reported checksum is published as the replica's
    /// integrity reference instead of being verified against a producer-side
    /// one. See `begin_store_adopt` for why that is the honest real-transport
    /// contract rather than a weakening.
    Store {
        key: LogicalKey,
        tier: Tier,
        generation: Gen,
        bytes: u64,
        expected: Option<Checksum>,
    },
    Load {
        key: LogicalKey,
        src: Tier,
        dst: LoadDst,
        generation: Gen,
        bytes: u64,
        /// The source replica's checksum at op creation - what the delivered
        /// bytes must hash to (end-to-end read integrity).
        checksum: Checksum,
        waiters: Vec<WaiterId>,
    },
}

/// How a load ended, delivered per waiter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadResult {
    Ok,
    /// Transport failure - source replica intact, retry or recompute.
    IoFailed,
    /// Delivered bytes failed the checksum: at-rest corruption. The source
    /// replica has been marked Bad; recompute is the only path.
    Integrity,
}

/// A waiter to wake after `on_completion` / teardown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Wake {
    pub waiter: WaiterId,
    pub result: LoadResult,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReserveError {
    /// Not enough free bytes - caller evicts (policy is the scheduler's) and
    /// retries, or declines admission.
    Insufficient { free: u64, wanted: u64 },
    /// Replica already exists in a live state on this tier.
    AlreadyPresent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeginStoreError {
    NotReserved,
    /// Sealed size exceeds the reservation bound - a codec bug; nothing moves.
    ExceedsReservation {
        reserved: u64,
        sealed: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadError {
    /// No replica on the source tier (or it is Bad / not yet Ready).
    NotReady,
    /// Destination tier already holds (or is acquiring) this key.
    DstOccupied,
    /// Destination tier out of free bytes.
    DstInsufficient { free: u64, wanted: u64 },
    /// src == dst.
    SameTier,
}

/// Outcome of `cancel_waiter`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancelOutcome {
    /// Other waiters remain; the shared op keeps running.
    OpContinues,
    /// This was the last waiter - the op is torn down here; the CALLER must
    /// now call `transport.cancel(op)`. Any late completion becomes stale.
    CancelIo,
    /// The op already resolved (waiter was - or will not be - woken).
    AlreadyDone,
}

/// Starting a load either creates the op (submit `job`) or joins in flight.
#[derive(Debug)]
pub enum LoadStart {
    Started { op: OpId, job: IoJob },
    Joined { op: OpId },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictError {
    NotFound,
    /// Reserved or Writing - cancel the op first.
    Busy,
    /// Ready but pinned by a live read.
    Pinned,
}

#[derive(Debug)]
struct TierState {
    ledger: Ledger,
    replicas: HashMap<LogicalKey, Replica>,
}

#[derive(Debug)]
pub struct TierCatalog {
    tiers: [TierState; 2],
    ops: HashMap<OpId, OpKind>,
    /// Single-flight: at most one live load per (key, destination class).
    promotions: HashMap<(LogicalKey, LoadDst), OpId>,
    next_op: u64,
    next_gen: u64,
    pub counters: Counters,
}

impl TierCatalog {
    pub fn new(cfg: TierCatalogConfig) -> Self {
        let mk = |capacity: u64| TierState {
            ledger: Ledger {
                capacity,
                free: capacity,
                ..Ledger::default()
            },
            replicas: HashMap::new(),
        };
        TierCatalog {
            tiers: [mk(cfg.ram_capacity), mk(cfg.nvme_capacity)],
            ops: HashMap::new(),
            promotions: HashMap::new(),
            next_op: 1,
            next_gen: 1,
            counters: Counters::default(),
        }
    }

    pub fn ledger(&self, tier: Tier) -> Ledger {
        self.tiers[tier.idx()].ledger
    }

    /// Ready-and-readable bytes for `key` on `tier` (probe building block).
    /// The physical location of a Ready replica - what the engine hands back
    /// to the transport (`free_extent`) right before `evict`, and the
    /// occupancy witnesses read. The catalog itself never interprets it.
    pub fn ready_loc(&self, key: &LogicalKey, tier: Tier) -> Option<Loc> {
        match self.tiers[tier.idx()].replicas.get(key) {
            Some(Replica {
                state: ReplicaState::Ready { loc, .. },
                ..
            }) => Some(*loc),
            _ => None,
        }
    }

    pub fn ready_bytes(&self, key: &LogicalKey, tier: Tier) -> Option<u64> {
        match self.tiers[tier.idx()].replicas.get(key) {
            Some(Replica {
                bytes,
                state: ReplicaState::Ready { .. },
            }) => Some(*bytes),
            _ => None,
        }
    }

    /// The in-flight promotion for `(key, dst)` if any - probes report it so
    /// a scheduler can JOIN instead of recompute-or-duplicate.
    pub fn promotion_of(&self, key: &LogicalKey, dst: LoadDst) -> Option<OpId> {
        self.promotions.get(&(*key, dst)).copied()
    }

    /// Whether an operation is still live (observability + tests).
    pub fn has_op(&self, op: OpId) -> bool {
        self.ops.contains_key(&op)
    }

    fn fresh_op(&mut self) -> OpId {
        let id = OpId(self.next_op);
        self.next_op += 1;
        id
    }

    fn fresh_gen(&mut self) -> Gen {
        let g = Gen(self.next_gen);
        self.next_gen += 1;
        g
    }

    // -- admission ----------------------------------------------------------

    /// Hold `bound` bytes on `tier` for `key` (reservation-first). The
    /// bound comes from `PayloadSchema::reserve_bytes`; `begin_store` trims to
    /// the sealed exact size.
    pub fn reserve(&mut self, key: LogicalKey, tier: Tier, bound: u64) -> Result<(), ReserveError> {
        let t = &mut self.tiers[tier.idx()];
        match t.replicas.get(&key) {
            None
            | Some(Replica {
                state: ReplicaState::Bad,
                ..
            }) => {}
            Some(_) => return Err(ReserveError::AlreadyPresent),
        }
        if t.ledger.free < bound {
            return Err(ReserveError::Insufficient {
                free: t.ledger.free,
                wanted: bound,
            });
        }
        t.ledger.free -= bound;
        t.ledger.reserved += bound;
        t.replicas.insert(
            key,
            Replica {
                bytes: bound,
                state: ReplicaState::Reserved,
            },
        );
        Ok(())
    }

    /// Admission changed its mind before I/O - release exactly once.
    pub fn release_reservation(&mut self, key: &LogicalKey, tier: Tier) -> bool {
        let t = &mut self.tiers[tier.idx()];
        match t.replicas.get(key) {
            Some(Replica {
                state: ReplicaState::Reserved,
                bytes,
            }) => {
                let bytes = *bytes;
                t.replicas.remove(key);
                t.ledger.reserved -= bytes;
                t.ledger.free += bytes;
                true
            }
            _ => false,
        }
    }

    /// Boot-time preload of a replica RECOVERED from a durable tier's store
    /// (restart persistence): inserts it Ready with a fresh
    /// generation, charging the ledger. False (and nothing changes) when the
    /// key is already live on that tier or the ledger cannot hold it - the
    /// caller counts skips; a preload must never wedge an open.
    pub fn preload_ready(
        &mut self,
        key: LogicalKey,
        tier: Tier,
        loc: Loc,
        checksum: Checksum,
        bytes: u64,
    ) -> bool {
        let t = &mut self.tiers[tier.idx()];
        if t.replicas.contains_key(&key) || t.ledger.free < bytes {
            return false;
        }
        let generation = self.fresh_gen();
        let t = &mut self.tiers[tier.idx()];
        t.ledger.free -= bytes;
        t.ledger.ready += bytes;
        t.replicas.insert(
            key,
            Replica {
                bytes,
                state: ReplicaState::Ready {
                    generation,
                    loc,
                    checksum,
                    pins: 0,
                },
            },
        );
        true
    }

    // -- store --------------------------------------------------------------

    /// Start the demote I/O for a sealed payload: `sealed` exact bytes,
    /// `expected` the producer's checksum of the packed bytes. Returns the job
    /// to submit; the surplus of the reservation bound returns to `free` now.
    pub fn begin_store(
        &mut self,
        key: LogicalKey,
        tier: Tier,
        sealed: u64,
        expected: Checksum,
    ) -> Result<(OpId, IoJob), BeginStoreError> {
        self.begin_store_inner(key, tier, sealed, Some(expected))
    }

    /// `begin_store` without a producer-side checksum: the transport's
    /// reported checksum is ADOPTED as the replica's integrity reference
    /// (trust-on-first-write). This is the honest contract for the real
    /// transports, not a weakening: the packed bytes first exist host-side
    /// after the DMA, so a producer-side checksum could only come from
    /// reading the same payload a second time - halving store bandwidth to
    /// re-verify a leg (device->host DMA) the link layer already CRCs. Every
    /// at-rest READ - the legs where silent corruption actually lives -
    /// still verifies against the adopted checksum end to end.
    pub fn begin_store_adopt(
        &mut self,
        key: LogicalKey,
        tier: Tier,
        sealed: u64,
    ) -> Result<(OpId, IoJob), BeginStoreError> {
        self.begin_store_inner(key, tier, sealed, None)
    }

    fn begin_store_inner(
        &mut self,
        key: LogicalKey,
        tier: Tier,
        sealed: u64,
        expected: Option<Checksum>,
    ) -> Result<(OpId, IoJob), BeginStoreError> {
        let bound = match self.tiers[tier.idx()].replicas.get(&key) {
            Some(Replica {
                state: ReplicaState::Reserved,
                bytes,
            }) => *bytes,
            _ => return Err(BeginStoreError::NotReserved),
        };
        if sealed > bound {
            return Err(BeginStoreError::ExceedsReservation {
                reserved: bound,
                sealed,
            });
        }
        let op = self.fresh_op();
        let generation = self.fresh_gen();
        let t = &mut self.tiers[tier.idx()];
        t.ledger.reserved -= bound;
        t.ledger.in_flight += sealed;
        t.ledger.free += bound - sealed;
        let r = t.replicas.get_mut(&key).expect("checked above");
        r.bytes = sealed;
        r.state = ReplicaState::Writing { op, generation };
        self.ops.insert(
            op,
            OpKind::Store {
                key,
                tier,
                generation,
                bytes: sealed,
                expected,
            },
        );
        Ok((
            op,
            IoJob {
                op,
                tier,
                key,
                bytes: sealed,
                kind: IoJobKind::Store { expected },
            },
        ))
    }

    /// Abort an in-flight store (pressure, shutdown). Idempotent; the caller
    /// must `transport.cancel(op)` afterwards - the late completion is stale.
    pub fn cancel_store(&mut self, op: OpId) -> bool {
        let Some(OpKind::Store {
            key, tier, bytes, ..
        }) = self.ops.get(&op)
        else {
            return false;
        };
        let (key, tier, bytes) = (*key, *tier, *bytes);
        let t = &mut self.tiers[tier.idx()];
        debug_assert!(matches!(
            t.replicas.get(&key).map(|r| &r.state),
            Some(ReplicaState::Writing { op: o, .. }) if *o == op
        ));
        t.replicas.remove(&key);
        t.ledger.in_flight -= bytes;
        t.ledger.free += bytes;
        self.ops.remove(&op);
        self.counters.op_cancels += 1;
        true
    }

    // -- load ---------------------------------------------------------------

    /// Start (or join) the promotion of `key` from `src` toward `dst`.
    /// Single-flight per (key, dst): a second caller joins the waiter list
    /// and may later still elect recompute via `cancel_waiter`. The GPU
    /// destination's block reservation is the pool/`kv_plan` arbiter's charge
    /// - made before calling this; a tier destination is charged here.
    pub fn begin_load(
        &mut self,
        key: LogicalKey,
        src: Tier,
        dst: LoadDst,
        waiter: WaiterId,
    ) -> Result<LoadStart, LoadError> {
        if let LoadDst::Tier(d) = dst
            && d == src
        {
            return Err(LoadError::SameTier);
        }
        if let Some(&op) = self.promotions.get(&(key, dst)) {
            match self.ops.get_mut(&op) {
                Some(OpKind::Load { waiters, .. }) => {
                    waiters.push(waiter);
                    self.counters.single_flight_joins += 1;
                    return Ok(LoadStart::Joined { op });
                }
                _ => unreachable!("promotion entry must reference a live load op"),
            }
        }
        // source must be Ready; record its identity before pinning
        let (generation, loc, checksum, bytes) = match self.tiers[src.idx()].replicas.get(&key) {
            Some(Replica {
                bytes,
                state:
                    ReplicaState::Ready {
                        generation,
                        loc,
                        checksum,
                        ..
                    },
            }) => (*generation, *loc, *checksum, *bytes),
            _ => return Err(LoadError::NotReady),
        };
        // destination charge, before any pin so failure changes nothing
        if let LoadDst::Tier(d) = dst {
            let t = &mut self.tiers[d.idx()];
            match t.replicas.get(&key) {
                None
                | Some(Replica {
                    state: ReplicaState::Bad,
                    ..
                }) => {}
                Some(_) => return Err(LoadError::DstOccupied),
            }
            if t.ledger.free < bytes {
                return Err(LoadError::DstInsufficient {
                    free: t.ledger.free,
                    wanted: bytes,
                });
            }
        }
        let op = self.fresh_op();
        // pin the source for the whole op
        {
            let t = &mut self.tiers[src.idx()];
            let r = t.replicas.get_mut(&key).expect("checked above");
            let ReplicaState::Ready { pins, .. } = &mut r.state else {
                unreachable!()
            };
            if *pins == 0 {
                t.ledger.pinned += bytes;
            }
            *pins += 1;
        }
        if let LoadDst::Tier(d) = dst {
            let t = &mut self.tiers[d.idx()];
            t.ledger.free -= bytes;
            t.ledger.in_flight += bytes;
            t.replicas.insert(
                key,
                Replica {
                    bytes,
                    state: ReplicaState::Writing { op, generation },
                },
            );
        }
        self.ops.insert(
            op,
            OpKind::Load {
                key,
                src,
                dst,
                generation,
                bytes,
                checksum,
                waiters: vec![waiter],
            },
        );
        self.promotions.insert((key, dst), op);
        Ok(LoadStart::Started {
            op,
            job: IoJob {
                op,
                tier: src,
                key,
                bytes,
                kind: IoJobKind::Load { loc, dst },
            },
        })
    }

    /// One waiter gives up (elects recompute). The op survives while any other
    /// waiter remains - cancellation is per-waiter, teardown is last-out.
    pub fn cancel_waiter(&mut self, op: OpId, waiter: WaiterId) -> CancelOutcome {
        let Some(OpKind::Load { waiters, .. }) = self.ops.get_mut(&op) else {
            return CancelOutcome::AlreadyDone;
        };
        let before = waiters.len();
        waiters.retain(|w| *w != waiter);
        if waiters.len() == before {
            return CancelOutcome::AlreadyDone; // unknown waiter - idempotent
        }
        let now_empty = waiters.is_empty();
        self.counters.waiter_cancels += 1;
        if !now_empty {
            return CancelOutcome::OpContinues;
        }
        self.teardown_load(op);
        self.counters.op_cancels += 1;
        CancelOutcome::CancelIo
    }

    /// Unpin source, release dst, drop promotion + op. Shared by cancel and
    /// the failure paths - the single place teardown happens, so it happens
    /// exactly once.
    fn teardown_load(&mut self, op: OpId) {
        let Some(OpKind::Load {
            key,
            src,
            dst,
            bytes,
            ..
        }) = self.ops.remove(&op)
        else {
            return;
        };
        self.unpin(&key, src, bytes);
        if let LoadDst::Tier(d) = dst {
            let t = &mut self.tiers[d.idx()];
            if matches!(
                t.replicas.get(&key).map(|r| &r.state),
                Some(ReplicaState::Writing { op: o, .. }) if *o == op
            ) {
                t.replicas.remove(&key);
                t.ledger.in_flight -= bytes;
                t.ledger.free += bytes;
            }
        }
        self.promotions.remove(&(key, dst));
    }

    fn unpin(&mut self, key: &LogicalKey, tier: Tier, bytes: u64) {
        let t = &mut self.tiers[tier.idx()];
        if let Some(Replica {
            state: ReplicaState::Ready { pins, .. },
            ..
        }) = t.replicas.get_mut(key)
        {
            debug_assert!(*pins > 0, "unpin under zero pins");
            *pins -= 1;
            if *pins == 0 {
                t.ledger.pinned -= bytes;
            }
        }
        // Ready may have become Bad through the integrity path while other
        // readers were live - Bad holds no pin accounting, nothing to do.
    }

    // -- completion ---------------------------------------------------------

    /// Feed one transport completion. Returns the waiters to wake (empty for
    /// stores). Unknown op ids are STALE - counted, ignored: that single rule
    /// makes duplicate and post-cancel completions harmless.
    pub fn on_completion(&mut self, c: IoCompletion) -> Vec<Wake> {
        match self.ops.get(&c.op) {
            None => {
                self.counters.stale_completions += 1;
                Vec::new()
            }
            Some(OpKind::Store { .. }) => {
                self.complete_store(c);
                Vec::new()
            }
            Some(OpKind::Load { .. }) => self.complete_load(c),
        }
    }

    fn complete_store(&mut self, c: IoCompletion) {
        let Some(OpKind::Store {
            key,
            tier,
            generation,
            bytes,
            expected,
        }) = self.ops.remove(&c.op)
        else {
            unreachable!("dispatched on kind")
        };
        let release = |t: &mut TierState| {
            t.replicas.remove(&key);
            t.ledger.in_flight -= bytes;
            t.ledger.free += bytes;
        };
        let t = &mut self.tiers[tier.idx()];
        // second belt beside the op table: the replica must still be this
        // op's Writing at this generation, or someone re-owned the slot
        let owner_ok = matches!(
            t.replicas.get(&key).map(|r| &r.state),
            Some(ReplicaState::Writing { op, generation: g }) if *op == c.op && *g == generation
        );
        if !owner_ok {
            self.counters.protocol_errors += 1;
            return;
        }
        match c.outcome {
            IoOutcome::StoreDone {
                loc,
                bytes: landed,
                checksum,
            } => {
                if landed != bytes {
                    self.counters.short_writes += 1;
                    release(t);
                    return;
                }
                if let Some(exp) = expected
                    && checksum != exp
                {
                    self.counters.integrity_failures += 1;
                    release(t);
                    return;
                }
                // publish: producer event + full byte count + integrity
                let r = t.replicas.get_mut(&key).expect("owner_ok");
                r.state = ReplicaState::Ready {
                    generation,
                    loc,
                    checksum,
                    pins: 0,
                };
                t.ledger.in_flight -= bytes;
                t.ledger.ready += bytes;
            }
            IoOutcome::Failed => {
                self.counters.io_failures += 1;
                release(t);
            }
            IoOutcome::NoRoom => {
                self.counters.no_room += 1;
                release(t);
            }
            IoOutcome::LoadDone { .. } => {
                self.counters.protocol_errors += 1;
                release(t);
            }
        }
    }

    fn complete_load(&mut self, c: IoCompletion) -> Vec<Wake> {
        let op_id = c.op;
        let Some(OpKind::Load {
            key,
            src,
            dst,
            generation,
            bytes,
            checksum,
            waiters,
        }) = self.ops.remove(&op_id)
        else {
            unreachable!("dispatched on kind")
        };
        self.promotions.remove(&(key, dst));
        self.unpin(&key, src, bytes);
        let _ = op_id; // silences nothing: used in release_dst below
        let release_dst = |tiers: &mut [TierState; 2]| {
            if let LoadDst::Tier(d) = dst {
                let t = &mut tiers[d.idx()];
                if matches!(
                    t.replicas.get(&key).map(|r| &r.state),
                    Some(ReplicaState::Writing { op, .. }) if *op == op_id
                ) {
                    t.replicas.remove(&key);
                    t.ledger.in_flight -= bytes;
                    t.ledger.free += bytes;
                }
            }
        };
        let wake = |result: LoadResult| {
            waiters
                .iter()
                .map(|w| Wake { waiter: *w, result })
                .collect()
        };
        match c.outcome {
            IoOutcome::LoadDone {
                bytes: delivered,
                checksum: got,
                dst_loc,
            } => {
                if delivered != bytes || got != checksum {
                    // the bytes at REST no longer hash to what was published:
                    // at-rest corruption. Source is poisoned - mark Bad.
                    self.counters.integrity_failures += 1;
                    self.mark_bad_internal(&key, src);
                    release_dst(&mut self.tiers);
                    return wake(LoadResult::Integrity);
                }
                if let LoadDst::Tier(d) = dst {
                    let Some(loc) = dst_loc else {
                        // transport didn't say where it landed - protocol bug
                        self.counters.protocol_errors += 1;
                        release_dst(&mut self.tiers);
                        return wake(LoadResult::IoFailed);
                    };
                    let t = &mut self.tiers[d.idx()];
                    if let Some(r) = t.replicas.get_mut(&key) {
                        // dst inherits the SOURCE generation - same content
                        r.state = ReplicaState::Ready {
                            generation,
                            loc,
                            checksum,
                            pins: 0,
                        };
                        t.ledger.in_flight -= bytes;
                        t.ledger.ready += bytes;
                    }
                }
                wake(LoadResult::Ok)
            }
            IoOutcome::Failed | IoOutcome::NoRoom => {
                self.counters.io_failures += 1;
                release_dst(&mut self.tiers);
                wake(LoadResult::IoFailed)
            }
            IoOutcome::StoreDone { .. } => {
                self.counters.protocol_errors += 1;
                release_dst(&mut self.tiers);
                wake(LoadResult::IoFailed)
            }
        }
    }

    // -- eviction / integrity ----------------------------------------------

    /// Evict a Ready, unpinned replica (policy is the caller's). Returns the
    /// bytes freed. Evicting a Bad marker returns 0 (its bytes were released
    /// at mark time).
    pub fn evict(&mut self, key: &LogicalKey, tier: Tier) -> Result<u64, EvictError> {
        let t = &mut self.tiers[tier.idx()];
        match t.replicas.get(key) {
            None => Err(EvictError::NotFound),
            Some(Replica {
                state: ReplicaState::Reserved | ReplicaState::Writing { .. },
                ..
            }) => Err(EvictError::Busy),
            Some(Replica {
                state: ReplicaState::Ready { pins, .. },
                ..
            }) if *pins > 0 => Err(EvictError::Pinned),
            Some(Replica {
                state: ReplicaState::Ready { .. },
                bytes,
            }) => {
                let bytes = *bytes;
                t.replicas.remove(key);
                t.ledger.ready -= bytes;
                t.ledger.free += bytes;
                self.counters.evictions += 1;
                Ok(bytes)
            }
            Some(Replica {
                state: ReplicaState::Bad,
                ..
            }) => {
                t.replicas.remove(key);
                Ok(0)
            }
        }
    }

    /// External integrity discovery (scrub). A pinned replica cannot be
    /// yanked mid-read - the read's own checksum will catch the damage.
    pub fn mark_bad(&mut self, key: &LogicalKey, tier: Tier) -> Result<(), EvictError> {
        match self.tiers[tier.idx()].replicas.get(key) {
            Some(Replica {
                state: ReplicaState::Ready { pins: 0, .. },
                ..
            }) => {
                self.mark_bad_internal(key, tier);
                Ok(())
            }
            Some(Replica {
                state: ReplicaState::Ready { .. },
                ..
            }) => Err(EvictError::Pinned),
            Some(_) => Err(EvictError::Busy),
            None => Err(EvictError::NotFound),
        }
    }

    /// Ready -> Bad regardless of pins (the integrity read path arrives here
    /// with its own pin already released). Bytes leave `ready`; live readers'
    /// pin accounting was already dropped by their own unpin path.
    fn mark_bad_internal(&mut self, key: &LogicalKey, tier: Tier) {
        let t = &mut self.tiers[tier.idx()];
        if let Some(r) = t.replicas.get_mut(key)
            && let ReplicaState::Ready { pins, .. } = &r.state
        {
            let (pins, bytes) = (*pins, r.bytes);
            if pins > 0 {
                // other concurrent readers still hold pins; their unpins
                // will find Bad and no-op. Drop the pinned accounting now.
                t.ledger.pinned -= bytes;
            }
            t.ledger.ready -= bytes;
            t.ledger.free += bytes;
            r.state = ReplicaState::Bad;
            self.counters.bad_marks += 1;
        }
    }

    // -- invariants (the race suite's oracle) -------------------------------

    /// Prove the whole structure consistent; panics with a description on the
    /// first violation. O(replicas + ops) - test instrument, also cheap
    /// enough for debug builds at checkpoints.
    pub fn check_invariants(&self) {
        for tier in Tier::ALL {
            let t = &self.tiers[tier.idx()];
            let l = &t.ledger;
            assert_eq!(
                l.free + l.reserved + l.in_flight + l.ready,
                l.capacity,
                "{tier:?}: ledger buckets must partition capacity: {l:?}"
            );
            let (mut reserved, mut in_flight, mut ready, mut pinned) = (0u64, 0u64, 0u64, 0u64);
            for (key, r) in &t.replicas {
                match &r.state {
                    ReplicaState::Reserved => reserved += r.bytes,
                    ReplicaState::Writing { op, .. } => {
                        in_flight += r.bytes;
                        assert!(
                            self.ops.contains_key(op),
                            "{tier:?}: Writing replica references dead op {op:?}"
                        );
                    }
                    ReplicaState::Ready { pins, .. } => {
                        ready += r.bytes;
                        if *pins > 0 {
                            pinned += r.bytes;
                            let live = self
                                .ops
                                .values()
                                .filter(|k| {
                                    matches!(k, OpKind::Load { key: k2, src, .. }
                                    if k2 == key && *src == tier)
                                })
                                .count() as u32;
                            assert_eq!(
                                live, *pins,
                                "{tier:?}: pins must equal live read ops for the key"
                            );
                        }
                    }
                    ReplicaState::Bad => {}
                }
            }
            assert_eq!(
                l.reserved, reserved,
                "{tier:?}: reserved bucket vs replicas"
            );
            assert_eq!(
                l.in_flight, in_flight,
                "{tier:?}: in_flight bucket vs replicas"
            );
            assert_eq!(l.ready, ready, "{tier:?}: ready bucket vs replicas");
            assert_eq!(l.pinned, pinned, "{tier:?}: pinned bucket vs replicas");
            assert!(
                l.pinned <= l.ready,
                "{tier:?}: pinned must be a subset of ready"
            );
        }
        for (op, kind) in &self.ops {
            match kind {
                OpKind::Store {
                    key,
                    tier,
                    generation,
                    ..
                } => {
                    let ok = matches!(
                        self.tiers[tier.idx()].replicas.get(key).map(|r| &r.state),
                        Some(ReplicaState::Writing { op: o, generation: g }) if o == op && g == generation
                    );
                    assert!(ok, "store {op:?} must own a Writing replica");
                }
                OpKind::Load {
                    key,
                    src,
                    dst,
                    waiters,
                    ..
                } => {
                    assert!(!waiters.is_empty(), "load {op:?} must have waiters");
                    assert_eq!(
                        self.promotions.get(&(*key, *dst)),
                        Some(op),
                        "load {op:?} must be the registered promotion"
                    );
                    let src_state = self.tiers[src.idx()].replicas.get(key).map(|r| &r.state);
                    let src_ok = matches!(src_state,
                        Some(ReplicaState::Ready { pins, .. }) if *pins > 0)
                        // a sibling reader's integrity failure may poison the
                        // source mid-read; this op's completion will fail too
                        || matches!(src_state, Some(ReplicaState::Bad));
                    assert!(
                        src_ok,
                        "load {op:?} source must be Ready+pinned (or poisoned Bad)"
                    );
                }
            }
        }
        for ((key, dst), op) in &self.promotions {
            let ok = matches!(self.ops.get(op), Some(OpKind::Load { key: k, dst: d, .. })
                if k == key && d == dst);
            assert!(
                ok,
                "promotion entry ({key:?},{dst:?}) must reference its live op"
            );
        }
    }
}
