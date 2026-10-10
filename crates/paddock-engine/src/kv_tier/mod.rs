//! KV tier control plane - the transactional catalog behind RAM/NVMe KV
//! offloading. This module holds the CONTRACTS, built and race-tested against
//! a deterministic fake transport before any CUDA or disk wiring - the same
//! CPU-side-first discipline that made `kv_pool` testable.
//!
//! Four separated granularities - identity, payload, residency and
//! I/O size are different things and never share one struct:
//!
//! - **logical key** ([`LogicalKey`], `digest.rs`) - strong content-chain key
//!   per native 16-token block, rooted in a [`digest::CacheNamespace`]
//!   (identity digest × privacy scope). Preserves partial-prefix hits, dedup
//!   and sharing across sequences.
//! - **payload schema** (`payload.rs`) - a versioned descriptor of everything
//!   required to RESUME at a boundary, not just attention KV: cache groups,
//!   SWA ring + window metadata, recurrent/conv state at the same epoch,
//!   positional/multimodal metadata, drafter state or a re-warm plan. A hit is
//!   usable only when every required component is ready.
//! - **replica record** (`catalog.rs`) - per-tier residency with independent
//!   transactional state; GPU + RAM + disk may all hold valid copies at once.
//!   (GPU residency stays owned by `kv_pool`/`paged_radix`; the catalog tracks
//!   the OFF-GPU tiers and the in-flight traffic between all three.)
//! - **physical extent** - an adaptive multi-MiB aggregation of adjacent
//!   logical keys for pack/DMA/disk jobs, with an offset table so a partial
//!   hit reads only useful spans. Transfer contiguity must not dictate
//!   cache-key granularity. (Extent packing lands with the transports; the
//!   catalog already speaks per-key byte accounting.)
//!
//! The data plane sits behind [`transport::TierTransport`]; the elected v1
//! implementation everywhere is the measured CPU-bounce path (pinned staging
//! ring + async DMA / buffered-unbuffered platform IO). Direct paths
//! (DirectStorage, GDS, CXL) stay electable behind the same trait.

pub mod accounting;
pub mod catalog;
pub mod cold;
pub mod cost;
pub mod digest;
pub mod fingerprint;
#[cfg(feature = "cuda")]
mod hash_pool;
#[cfg(feature = "cuda")]
pub mod host;
pub mod io;
pub mod nvme_store;
pub mod payload;
#[cfg(feature = "cuda")]
pub mod pool_tier;
#[cfg(feature = "cuda")]
pub mod ram_transport;
#[cfg(feature = "cuda")]
pub mod restore_flow;
pub mod transport;

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests;

pub use accounting::{MissReason, TierDecisions, TierReport};
pub use catalog::{TierCatalog, TierCatalogConfig};
pub use cost::{CostModel, Election, HitShape};
pub use digest::{CacheNamespace, Checksum, IdentityDigest, LogicalKey, PrivacyScope};
pub use payload::{KvPayloadCodec, PayloadManifest, PayloadSchema};
#[cfg(feature = "cuda")]
pub use pool_tier::{PoolTier, RestoreWake, TierHit, XferSink};
#[cfg(feature = "cuda")]
pub use ram_transport::{PlaneDesc, RamTransport, XferSpec};
#[cfg(feature = "cuda")]
pub use restore_flow::{AuxPlan, FlowStatus, RestoreFlow};
pub use transport::{FakeTransport, IoCompletion, IoJob, TierTransport};

/// Backend-independent cache-tier telemetry consumed by the serving API.
#[derive(Debug, Clone, Copy, Default)]
pub struct TierStats {
    pub resident_runs: u64,
    pub ready_bytes: u64,
    pub in_flight_demotes: u64,
    pub open_tickets: u64,
    pub tripped: bool,
    pub single_flight_joins: u64,
    pub io_failures: u64,
    pub integrity_failures: u64,
    pub evictions: u64,
    pub stale_completions: u64,
    /// T2 payload bytes written this UTC day (endurance budget).
    pub t2_written_day_bytes: u64,
}

/// An off-GPU storage tier. v1 ships exactly these two; remote/cluster tiers
/// are a non-goal (plan, Non-goals). The catalog is written against this enum
/// rather than an open-ended id so ledger state can live in a fixed array and
/// an impossible tier is unrepresentable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tier {
    /// T1: host RAM capacity store. May be pageable/NUMA-aware memory; the
    /// pinned STAGING ring is transport property, not tier capacity.
    Ram,
    /// T2: local NVMe with the transactional on-disk format.
    Nvme,
}

impl Tier {
    pub const ALL: [Tier; 2] = [Tier::Ram, Tier::Nvme];

    #[inline]
    pub(crate) fn idx(self) -> usize {
        match self {
            Tier::Ram => 0,
            Tier::Nvme => 1,
        }
    }
}

/// Where a load delivers. The GPU destination's block reservation is charged
/// by the pool/`kv_plan` arbiter before the load starts (reservation-first,
/// reservation-first); a tier destination (NVMe->RAM prefetch) is charged inside the
/// catalog itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LoadDst {
    Gpu,
    Tier(Tier),
}

/// Monotonic operation id. Every store/load is one operation; completions,
/// cancels and waiter bookkeeping all key on it, and a completion whose op id
/// is no longer in the table is STALE by definition (counted, ignored) - that
/// single rule is what makes late/duplicate completions harmless.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OpId(pub u64);

/// Immutable content generation. Assigned when a store operation is created;
/// a replica publishes carrying the generation of the store that produced it,
/// and a tier-to-tier copy INHERITS the source generation (same bytes, same
/// content). Completions validate generation as a second belt beside the op
/// table, so a completion from before an evict+re-store cycle can never
/// publish into the new owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Gen(pub u64);

/// Opaque physical location handed back by a tier's transport at store
/// completion (arena offset for RAM, extent+offset for NVMe). The catalog
/// never interprets it - it round-trips it into subsequent load jobs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Loc(pub u64);

/// Identifies a scheduler-side waiter parked on an in-flight load (one
/// request slot, in practice). Waiters join single-flight loads and may
/// independently cancel (elect recompute) without tearing down the shared op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WaiterId(pub u64);

/// What a prefix probe hands the scheduler for one off-tier hit - the
/// probe-returns-a-plan contract. This fixes the SHAPE; the fields
/// are filled by the Phase-1a scheduler wiring (`generator`/`kv_plan` grow
/// probe / poll-wake / cancel / publish-sealed / reserve hooks - the current
/// synchronous `prefill_begin` contract cannot express an asynchronous hit).
#[derive(Debug, Clone)]
pub struct RestorePlan {
    /// Blocks already GPU-resident (radix hit) - charged as cached by
    /// admission (fixes TRT-LLM's reuse-blind scheduler).
    pub gpu_blocks: u32,
    /// Restorable suffix per tier, in blocks, with per-tier payload bytes.
    pub restorable: Vec<(Tier, u32, u64)>,
    /// Required payload components that are not ready on any tier (a hit is
    /// usable only when every required component is ready).
    pub missing_components: u32,
    /// Predicted ready time under CURRENT queues (cost model), in
    /// microseconds. u64::MAX until the cost model lands.
    pub est_ready_us: u64,
}
