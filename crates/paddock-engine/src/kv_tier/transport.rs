//! The data-plane seam and its deterministic test double.
//!
//! [`TierTransport`] is capability-driven: CUDA staging-ring DMA is one
//! implementation, platform NVMe IO (overlapped/IORing on Windows, io_uring +
//! O_DIRECT on Linux, POSIX on macOS) is another, and future direct paths
//! (DirectStorage, GDS, CXL) are electable behind the same trait without
//! re-architecting. The elected v1 path everywhere is the measured CPU-bounce.
//!
//! The trait is intentionally a COMMAND interface: submit / cancel / poll.
//! No callbacks, no async runtime - the catalog's transactional core stays a
//! deterministic state machine, and completions are events the driver loop
//! feeds it. That is also what makes the race suite possible: [`FakeTransport`]
//! implements the same interface but lets a test deliver, delay, drop,
//! duplicate, truncate or corrupt any completion in any order.

use std::collections::HashMap;

use super::digest::{Checksum, LogicalKey};
use super::{LoadDst, Loc, OpId, Tier};

/// What a transport can and cannot do - election input (the device probes), never
/// a runtime branch inside the catalog.
#[derive(Debug, Clone, Copy)]
pub struct TransportCaps {
    /// Can gather strided device spans itself (otherwise the pack kernel must
    /// stage into one contiguous run first).
    pub gather_scatter: bool,
    /// Required IO alignment for unbuffered/direct paths, if any. Discovered
    /// per device, never assumed (4 KiB alone does not satisfy
    /// every Windows unbuffered device).
    pub direct_io_align: Option<u32>,
    /// Completed writes survive process death once flushed (an NVMe tier is
    /// durable, a RAM tier is not).
    pub durable: bool,
    /// In-flight operations can be cancelled (best effort - a cancelled op
    /// may still complete; the catalog treats such completions as stale).
    pub cancellable: bool,
}

/// One submitted IO operation.
#[derive(Debug, Clone)]
pub struct IoJob {
    pub op: OpId,
    pub tier: Tier,
    pub key: LogicalKey,
    /// Exact payload bytes (manifest total for stores; replica bytes for loads).
    pub bytes: u64,
    pub kind: IoJobKind,
}

#[derive(Debug, Clone)]
pub enum IoJobKind {
    /// Pack + write toward the tier. `expected` is the producer's checksum of
    /// the packed bytes when the producer has one (the fake / any transport
    /// whose payload exists before submission); `None` is the adopt mode -
    /// the transport reports the checksum it computed over what it persisted
    /// and the catalog publishes that (`begin_store_adopt`).
    Store { expected: Option<Checksum> },
    /// Read + unpack from the tier at `loc` (round-tripped from the store
    /// completion via the replica record), delivering toward `dst`. A tier
    /// destination writes through to that tier's store in the same op (an
    /// NVMe->RAM promotion is one IO - T1 capacity is host memory), and the
    /// completion reports where the bytes landed.
    Load { loc: Loc, dst: LoadDst },
}

/// Completion event, fed to the catalog by the driver loop.
#[derive(Debug, Clone)]
pub struct IoCompletion {
    pub op: OpId,
    pub outcome: IoOutcome,
}

#[derive(Debug, Clone)]
pub enum IoOutcome {
    /// Store persisted. `bytes` is what actually landed (the catalog rejects
    /// short writes), `checksum` what the tier holds (verified against the
    /// producer's), `loc` where - round-tripped into future loads.
    StoreDone {
        loc: Loc,
        bytes: u64,
        checksum: Checksum,
    },
    /// Load delivered. `checksum` is over the DELIVERED bytes - end-to-end
    /// read integrity, verified against the replica record. `dst_loc` is where
    /// the bytes landed when the destination is a tier (`None` for GPU).
    LoadDone {
        bytes: u64,
        checksum: Checksum,
        dst_loc: Option<Loc>,
    },
    /// The transport failed the op (device error, cancellation acknowledged).
    Failed,
    /// A store found no room for its extent in the tier's memory - capacity,
    /// not a fault: the catalog releases it like `Failed`, but it must never
    /// count toward the circuit breaker, or a merely FULL tier goes offline.
    NoRoom,
}

/// The data-plane trait. Implementations own their queues, staging and
/// platform handles; they never touch catalog state.
pub trait TierTransport {
    fn caps(&self) -> TransportCaps;
    /// Enqueue. Rejection is immediate (queue full, alignment) - after an Ok
    /// the op will eventually produce at most one non-stale completion.
    fn submit(&mut self, job: IoJob) -> Result<(), SubmitError>;
    /// Best-effort cancel; the op may still complete (raced) or complete as
    /// `Failed` - the catalog is correct under all three outcomes.
    fn cancel(&mut self, op: OpId);
    /// Drain ready completions.
    fn poll(&mut self) -> Vec<IoCompletion>;

    /// Whether stores of these sizes, submitted in order now, would each find
    /// room in the tier's memory. Transports without an arena of their own
    /// answer true; the RAM transport answers for its slabs.
    fn can_place(&self, _extents: &[u64]) -> bool {
        true
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmitError {
    QueueFull,
    Misaligned,
    /// The engine did not register a device-span source/destination spec for
    /// this key before submitting (RamTransport `expect_store`/`expect_load`).
    SourceMissing,
    /// The packed record/extent exceeds the transport's staging capacity.
    TooLarge,
}

// ---------------------------------------------------------------------------
// The deterministic fake - the race suite's instrument.
// ---------------------------------------------------------------------------

/// In-memory tier store + fully test-controlled completion delivery. Payload
/// CONTENT is synthesized deterministically from `(key, bytes)`, so checksums
/// are real BLAKE3 end to end and a corruption test flips real bytes.
#[derive(Debug, Default)]
pub struct FakeTransport {
    /// Jobs submitted and not yet resolved by the test.
    pending: HashMap<OpId, IoJob>,
    /// Order of submission (tests that "deliver in order" / "deliver reversed").
    order: Vec<OpId>,
    /// The tier's at-rest bytes.
    stored: HashMap<Loc, Vec<u8>>,
    next_loc: u64,
    /// Ops the catalog asked to cancel - by default they complete `Failed`
    /// when delivered; a test may instead force the raced full completion.
    cancelled: Vec<OpId>,
    ready: Vec<IoCompletion>,
}

/// Deterministic payload content for a key: repeat the key digest to length.
/// Content is a function of the key alone so a load can be verified against
/// what an INDEPENDENT store of the same key would have produced.
pub fn synth_payload(key: &LogicalKey, bytes: u64) -> Vec<u8> {
    key.0.iter().copied().cycle().take(bytes as usize).collect()
}

impl FakeTransport {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn in_flight(&self) -> usize {
        self.pending.len()
    }

    /// Pending op ids in SUBMISSION order - deterministic, so seeded fuzz
    /// runs replay exactly.
    pub fn pending_ops(&self) -> Vec<OpId> {
        self.order.clone()
    }

    /// At-rest locations, sorted - deterministic for the same reason.
    pub fn locs(&self) -> Vec<Loc> {
        let mut v: Vec<Loc> = self.stored.keys().copied().collect();
        v.sort();
        v
    }

    fn take(&mut self, op: OpId) -> Option<IoJob> {
        self.order.retain(|o| *o != op);
        self.pending.remove(&op)
    }

    /// Deliver op's natural, successful completion.
    pub fn deliver(&mut self, op: OpId) {
        let Some(job) = self.take(op) else { return };
        let outcome = match job.kind {
            IoJobKind::Store { .. } => {
                let data = synth_payload(&job.key, job.bytes);
                let checksum = Checksum::of_payload(&data);
                let loc = Loc(self.next_loc);
                self.next_loc += 1;
                self.stored.insert(loc, data);
                IoOutcome::StoreDone {
                    loc,
                    bytes: job.bytes,
                    checksum,
                }
            }
            IoJobKind::Load { loc, dst } => match self.stored.get(&loc).cloned() {
                Some(data) => {
                    let dst_loc = match dst {
                        LoadDst::Gpu => None,
                        LoadDst::Tier(_) => {
                            let l = Loc(self.next_loc);
                            self.next_loc += 1;
                            self.stored.insert(l, data.clone());
                            Some(l)
                        }
                    };
                    IoOutcome::LoadDone {
                        bytes: data.len() as u64,
                        checksum: Checksum::of_payload(&data),
                        dst_loc,
                    }
                }
                None => IoOutcome::Failed,
            },
        };
        self.ready.push(IoCompletion { op, outcome });
    }

    /// Deliver every pending op in submission order.
    pub fn deliver_all(&mut self) {
        for op in std::mem::take(&mut self.order) {
            // take() re-checks pending, so this tolerates prior partial delivery
            if self.pending.contains_key(&op) {
                self.order.push(op); // restore for take()'s retain bookkeeping
                self.deliver(op);
            }
        }
    }

    /// Deliver a SHORT write: only `bytes` landed. The catalog must refuse to
    /// publish and release the reservation.
    pub fn deliver_short(&mut self, op: OpId, bytes: u64) {
        let Some(job) = self.take(op) else { return };
        let data = synth_payload(&job.key, bytes);
        let checksum = Checksum::of_payload(&data);
        let loc = Loc(self.next_loc);
        self.next_loc += 1;
        self.stored.insert(loc, data);
        self.ready.push(IoCompletion {
            op: job.op,
            outcome: IoOutcome::StoreDone {
                loc,
                bytes,
                checksum,
            },
        });
    }

    /// Deliver "no room in the tier's memory" for a store.
    pub fn deliver_no_room(&mut self, op: OpId) {
        if self.take(op).is_some() {
            self.ready.push(IoCompletion {
                op,
                outcome: IoOutcome::NoRoom,
            });
        }
    }

    /// Deliver failure (device error / acknowledged cancel).
    pub fn deliver_failed(&mut self, op: OpId) {
        if self.take(op).is_some() {
            self.ready.push(IoCompletion {
                op,
                outcome: IoOutcome::Failed,
            });
        }
    }

    /// Duplicate the last ready completion for `op` (a transport bug / retry
    /// race). The catalog must count the second one stale, change nothing.
    pub fn duplicate_last(&mut self) {
        if let Some(c) = self.ready.last().cloned() {
            self.ready.push(c);
        }
    }

    /// Silently lose the op - no completion will ever arrive. (Models a lost
    /// interrupt; catalog-level timeouts are a Phase-3 concern, the Phase-0
    /// contract is that nothing corrupts and cancel still cleans up.)
    pub fn drop_op(&mut self, op: OpId) {
        self.take(op);
    }

    /// Flip a byte of the at-rest content (silent media corruption). The next
    /// load of that loc delivers a checksum that no longer matches the
    /// replica record; the catalog must fail the load and mark the replica Bad.
    pub fn corrupt_at_rest(&mut self, loc: Loc) {
        if let Some(data) = self.stored.get_mut(&loc)
            && let Some(b) = data.first_mut()
        {
            *b ^= 0xff;
        }
    }

    /// Whether the given loc still holds bytes (eviction tests).
    pub fn holds(&self, loc: Loc) -> bool {
        self.stored.contains_key(&loc)
    }
}

impl TierTransport for FakeTransport {
    fn caps(&self) -> TransportCaps {
        TransportCaps {
            gather_scatter: true,
            direct_io_align: None,
            durable: false,
            cancellable: true,
        }
    }

    fn submit(&mut self, job: IoJob) -> Result<(), SubmitError> {
        self.order.push(job.op);
        self.pending.insert(job.op, job);
        Ok(())
    }

    fn cancel(&mut self, op: OpId) {
        // Best-effort semantics: mark it; the TEST decides whether the op
        // resolves as Failed (deliver_failed), completes anyway (deliver - the
        // race), or vanishes (drop_op).
        self.cancelled.push(op);
    }

    fn poll(&mut self) -> Vec<IoCompletion> {
        std::mem::take(&mut self.ready)
    }
}
