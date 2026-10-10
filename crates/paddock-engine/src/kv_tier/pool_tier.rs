//! PoolTier - the seam between the transactional tier core and
//! the live paged pool: demote on eviction pressure, single-flight restore,
//! reservation-first accounting, all in `PagedRadix`/`KvPool` vocabulary so
//! families never touch catalog internals.
//!
//! ## The run model
//!
//! The demote/restore unit is a **run**: `run_blocks` adjacent chain blocks,
//! tiled from the chain root (blocks `[r*R, (r+1)*R)` form run `r`). One run
//! = one catalog entry keyed by its DEEPEST chain key = one physical extent
//! = one IO - which keeps ops 1:1 with transfers (no fragment tax) while
//! preserving partial-prefix hits at run granularity. `R` adapts to the
//! model: `clamp(16 MiB / record_stride, 1, 8)` blocks, so extents land in
//! the elected 2-16 MiB band whatever the family's KV geometry. A chain's
//! tail past the last full run boundary never demotes (it dies with
//! eviction); decode-page publication will revisit that.
//!
//! ## Demote (the `evict_lru` arm)
//!
//! [`PoolTier::pressure_demote`] replaces the families' plain evict-ahead
//! loop. It walks the same LRU-leaf order eviction always had, but when the
//! leaf being evicted CLOSES a run (depth % R == 0), the run's bytes are
//! captured first: every block of that run is an ancestor of the leaf, so
//! all are still alive - they are pinned (`pool.retain`), the store op is
//! submitted, and the pins release only at store COMPLETION (`pump`). The
//! pool sees the freed ids a few ms later than plain eviction would have
//! delivered them; nothing ever reads freed bytes (GPU blocks release only
//! after D2H completes). Content keys make re-demote free:
//! a run already resident (or in flight) reserves `AlreadyPresent` and is
//! skipped.
//!
//! ## Restore (probe -> elect -> single-flight ticket -> publish)
//!
//! [`PoolTier::probe`] extends a radix match: consecutive T1-Ready runs from
//! the run boundary at or below the GPU-resident depth. [`PoolTier::elect`]
//! prices restore vs recompute under the current queues.
//! [`PoolTier::begin_restore`] allocates fresh pool blocks per run (the
//! reservation, charged against the same pool `kv_plan` granted), starts one
//! load per run - joining any in-flight promotion of the same content
//! single-flight - and [`PoolTier::pump`] resolves tickets: each landed run
//! is PUBLISHED into the radix (`insert_extension`, which refuses to attach
//! across a hole if the prefix was evicted mid-flight), so every waiting
//! slot adopts the restored blocks zero-copy exactly like any other prefix
//! hit. A wake reports the published depth; the family re-matches and
//! resumes from whatever actually landed - a failed or partial restore
//! degrades to recompute, never to a wrong answer.
//!
//! ## Aux components (1b.3 hybrid payloads)
//!
//! Hybrid families (SWA rings, DeltaNet/Mamba state) can only RESUME at a
//! boundary whose checkpoint blob is present - restored blocks alone are
//! worthless there (the qwen3.8 c32 lesson). The tier moves those blobs as
//! **aux components**: a contiguous device blob shards into ≤16 MiB catalog
//! entries keyed `boundary_chain_key.child_bytes("aux", shard_i)`, riding
//! the same transport (a contiguous blob is a one-plane, one-block spec).
//! Demote: [`PoolTier::pressure_demote`] returns the checkpoints it claimed
//! off evicted paths ([`AuxTaken`], detached via `PagedRadix::take_state` so
//! the blob bytes outlive eviction); the family maps each to its device span
//! and calls [`PoolTier::demote_aux`]; the state-pool index recycles only
//! when every shard's store completed. Restore is two rounds: blocks first
//! (publication makes the boundary's node exist), then the family allocates
//! a checkpoint slot (`attach_state`) and calls
//! [`PoolTier::begin_restore_aux`] to land the blob into it - after which
//! its EXISTING hybrid resume path works untouched. A hybrid hit is usable
//! only to the deepest boundary whose every shard is Ready
//! ([`PoolTier::probe_aux`]) - the "every required component" rule.

use std::collections::HashMap;

use cudarc::driver::CudaEvent;

use super::catalog::{LoadResult, LoadStart, ReserveError, TierCatalog, TierCatalogConfig};
use super::cost::{CostModel, Election, HitShape};
use super::digest::{CacheNamespace, LogicalKey};
use super::ram_transport::{DEVICE_STAGING_EXTENT_BYTES, PlaneDesc, RamTransport, XferSpec};
use super::transport::{SubmitError, TierTransport};
use super::{LoadDst, Loc, OpId, Tier, WaiterId};
use crate::kv_pool::{BLOCK_TOKENS, BlockId, KvPool};
use crate::paged_radix::PagedRadix;

/// The SHIPPED tier election (1b.6): set once by the runner from its config
/// file's `[kv_offload]` section, before the model loads. This is the whole
/// user-facing surface - budgets only, per the no-tuning rule; everything
/// else stays elected.
static TIER_RAM: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();

/// Called by the runner exactly once at startup. Later calls are ignored
/// (the election is per-process, like the VRAM budget).
pub fn set_tier_ram_bytes(bytes: Option<u64>) {
    let _ = TIER_RAM.set(bytes.filter(|b| *b > 0));
}

/// The armed T2 election: `(store parent dir, quota bytes)` from
/// `[kv_offload] nvme_path/nvme_gb`. The family builds the namespace dir
/// How long an unused cache namespace may linger before the store retires it.
/// A namespace is keyed by model identity, so it is stranded the moment its
/// model's file, context size or KV dtype changes - correct, but unbounded
/// without a reaper. Two weeks is long enough that switching back to last
/// month's model still finds its cache warm, short enough that a box does not
/// accumulate dead caches forever. Elected, not configurable.
pub const STALE_NAMESPACE_TTL: std::time::Duration =
    std::time::Duration::from_secs(14 * 24 * 60 * 60);

/// Resolve the T2 store directory for a namespace, sweeping stale ones on the
/// way. Every family calls this rather than pairing `tier_nvme` with `dir_for`
/// itself, so the lifecycle has exactly one place to live - and so a family
/// cannot quietly forget to attach T2 at all, which laguna did.
pub fn nvme_dir_for(ns: &CacheNamespace) -> Option<(std::path::PathBuf, u64)> {
    let (root, quota) = tier_nvme()?;
    let dir = super::nvme_store::NvmeStore::dir_for(&root, ns);
    let (n, bytes) = super::nvme_store::NvmeStore::sweep_stale(&root, &dir, STALE_NAMESPACE_TTL);
    if n > 0 {
        tracing::info!(
            namespaces = n,
            reclaimed_mib = bytes / (1 << 20),
            "KV cache: retired stale namespaces at startup"
        );
    }
    Some((dir, quota))
}

/// under it via `NvmeStore::dir_for`.
static TIER_NVME: std::sync::OnceLock<Option<(std::path::PathBuf, u64)>> =
    std::sync::OnceLock::new();

pub fn set_tier_nvme(cfg: Option<(std::path::PathBuf, u64)>) {
    let _ = TIER_NVME.set(cfg.filter(|(_, q)| *q > 0));
}

pub fn tier_nvme() -> Option<(std::path::PathBuf, u64)> {
    TIER_NVME.get().cloned().flatten()
}

/// The armed T1 budget: the config file's `[kv_offload] ram_gb`, with the
/// dev flag (`PADDOCK_KV_TIER_RAM_GB`, compiled out of hardened builds) as
/// a dev-build override.
pub fn tier_ram_bytes() -> Option<u64> {
    if let Some(b) = paddock_models::dev_var!("PADDOCK_KV_TIER_RAM_GB")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|g| *g > 0.0)
        .map(|g| (g * (1u64 << 30) as f64) as u64)
    {
        return Some(b);
    }
    TIER_RAM.get().copied().flatten()
}

/// Bounded wait for the INTERIM synchronous restore, from the election's own
/// estimate. The async park/wake admission contract replaces this and is
/// the 1b ship gate; blocking is still strictly cheaper than the recompute
/// it replaces (the election proved it), and a timeout degrades to that
/// recompute with the ticket left running - its publication serves the next
/// request.
pub fn restore_deadline(est_us: f64) -> std::time::Duration {
    std::time::Duration::from_micros(((est_us * 4.0) as u64).clamp(5_000, 250_000))
}

/// Target extent size - the middle of the measured ≥2 MiB flat band, small enough
/// that a run is a fine-grained partial-hit unit.
const TARGET_EXTENT_BYTES: u64 = 16 << 20;
/// Ceiling on blocks per run: 8 blocks = 128 tokens of hit granularity.
const MAX_RUN_BLOCKS: usize = 8;

/// What a transport must offer beyond the catalog's command interface for
/// the pool tier to drive it: the device-geometry side channel and physical
/// extent reclamation. `RamTransport` implements it for real; tests
/// implement it over `FakeTransport` so every control-flow path here runs
/// deterministically without a GPU.
pub trait XferSink: TierTransport {
    /// Keys the transport's T2 cache evicted for quota since the last call -
    /// the pump drops their catalog Nvme references so no probe elects a
    /// load the disk can no longer serve. Default: none (no T2).
    fn take_t2_evictions(&mut self) -> Vec<[u8; 32]> {
        Vec::new()
    }

    /// Read-fill promotions: payloads a T2 load seated in the T1 slab on
    /// the way to the GPU - (key, loc, checksum, bytes). The pump adopts
    /// them into the catalog (next hit restores at RAM speed) or returns
    /// the extent. Default: none.
    fn take_t2_promotions(&mut self) -> Vec<([u8; 32], Loc, [u8; 32], u64)> {
        Vec::new()
    }

    /// Endurance telemetry: T2 payload bytes written this UTC day (3.3).
    fn t2_written_today(&self) -> u64 {
        0
    }

    /// The T2 device's probed read bandwidth in GB/s, when a store is
    /// attached - the cost model's seed for disk-sourced restores.
    /// Default: no T2, nothing to seed.
    fn t2_device_gbs(&self) -> Option<f64> {
        None
    }

    /// Durable writes deferred to read slack. Default: none.
    fn t2_pending_writes(&self) -> usize {
        0
    }

    /// The T2 device's measured geometry: (read GB/s, write GB/s,
    /// unbuffered). None when no store is attached.
    fn t2_device(&self) -> Option<(f64, f64, bool)> {
        None
    }

    /// A durable copy of `key` on disk, if one exists: (loc, bytes,
    /// checksum). Read at T1 eviction so the content survives in the tier
    /// rather than only on the device. Default: no T2, nothing durable.
    fn t2_entry(&self, _key: &[u8; 32]) -> Option<(Loc, u64, [u8; 32])> {
        None
    }

    /// Where the cost model's calibration persists (the T2 namespace
    /// directory), and the device tag it is keyed by. Default: nowhere.
    fn calibration_home(&self) -> Option<(std::path::PathBuf, String)> {
        None
    }

    /// The T1 arena's slabs and live extents, and the footprint it charges
    /// for `bytes` - eviction opens holes from these when the ledger has
    /// room but no hole fits ([`TierTransport::can_place`]). Default: no
    /// arena (placement always succeeds).
    fn arena_layout(&self) -> Vec<super::host::SlabLayout> {
        Vec::new()
    }
    fn arena_rounded(&self, bytes: u64) -> u64 {
        bytes
    }

    fn expect_store(&mut self, key: LogicalKey, spec: XferSpec) -> Result<(), SubmitError>;
    fn expect_load(&mut self, key: LogicalKey, spec: XferSpec) -> Result<(), SubmitError>;
    fn free_extent(&mut self, loc: Loc);
}

impl XferSink for RamTransport {
    fn take_t2_evictions(&mut self) -> Vec<[u8; 32]> {
        self.take_t2_evictions_inner()
    }
    fn t2_device_gbs(&self) -> Option<f64> {
        self.t2().map(|s| s.device().read_gbs)
    }
    fn t2_pending_writes(&self) -> usize {
        RamTransport::t2_pending_writes(self)
    }
    fn t2_device(&self) -> Option<(f64, f64, bool)> {
        self.t2().map(|s| {
            let d = s.device();
            (d.read_gbs, d.write_gbs, d.unbuffered)
        })
    }
    fn t2_entry(&self, key: &[u8; 32]) -> Option<(Loc, u64, [u8; 32])> {
        let (_gen, loc, len, sum) = self.t2()?.entry(key)?;
        Some((loc, len, sum))
    }
    fn calibration_home(&self) -> Option<(std::path::PathBuf, String)> {
        Some((self.t2()?.dir().to_path_buf(), self.device_tag()))
    }
    fn take_t2_promotions(&mut self) -> Vec<([u8; 32], Loc, [u8; 32], u64)> {
        self.take_t2_promotions_inner()
    }
    fn t2_written_today(&self) -> u64 {
        self.t2_written_today_inner()
    }
    fn arena_layout(&self) -> Vec<super::host::SlabLayout> {
        self.t1_layout()
    }
    fn arena_rounded(&self, bytes: u64) -> u64 {
        self.t1_rounded(bytes)
    }

    fn expect_store(&mut self, key: LogicalKey, spec: XferSpec) -> Result<(), SubmitError> {
        RamTransport::expect_store(self, key, spec)
    }
    fn expect_load(&mut self, key: LogicalKey, spec: XferSpec) -> Result<(), SubmitError> {
        RamTransport::expect_load(self, key, spec)
    }
    fn free_extent(&mut self, loc: Loc) {
        RamTransport::free_extent(self, loc)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum TierRefused {
    #[error(
        "one block's record ({record} bytes across {planes} planes) exceeds the \
         {staging}-byte staging extent - this geometry cannot tier"
    )]
    RecordTooLarge {
        record: u64,
        planes: usize,
        staging: usize,
    },
    #[error("plane geometry not 16-byte aligned - the pack kernels cannot move it")]
    Misaligned,
}

/// witness snapshot for /metrics and the Studio panel - decision
/// accountability, not just counters (each field names why bytes moved or
/// did not).
pub use super::TierStats;

/// A restorable extension found by [`PoolTier::probe`].
#[derive(Debug, Clone)]
pub struct TierHit {
    /// First block index (0-based) the restore covers - the run boundary at
    /// or below the GPU-resident depth (a sub-run overlap re-restores
    /// byte-identical content: priced, never wrong).
    pub start_block: usize,
    /// One past the last block the restore covers.
    pub end_block: usize,
    /// The run keys, in chain order.
    pub keys: Vec<LogicalKey>,
    /// Total payload bytes across the runs.
    pub bytes: u64,
    /// Of `bytes`, how many are sourced from T2 (disk). A chain often
    /// straddles the tiers - its cold tail demoted to disk while the head
    /// still sits in RAM - and the two read an order of magnitude apart, so
    /// the election prices each leg at its own measured rate.
    pub nvme_bytes: u64,
}

impl TierHit {
    /// Tokens of new prefix the restore delivers beyond what the GPU holds.
    pub fn new_tokens(&self, gpu_blocks: usize) -> u32 {
        (self.end_block.saturating_sub(gpu_blocks) * BLOCK_TOKENS) as u32
    }
}

/// A checkpoint the demote arm claimed off an evicted chain - the family
/// maps `state_idx` to its device blob span and either `demote_aux`es it or
/// recycles the index (every `AuxTaken` must take exactly one of the two
/// paths, or the state pool leaks a slot).
#[derive(Debug, Clone)]
pub struct AuxTaken {
    pub key: LogicalKey,
    pub end_block: usize,
    pub state_idx: u32,
    /// The run keys the checkpoint resumes over, root-first - bound to the
    /// blob once it stores (see `PoolTier::blob_runs`).
    pub runs: Vec<LogicalKey>,
}

/// Where an aux blob's bytes sit on device (issue #33).
///
/// `Flat` is one contiguous span - a fixed checkpoint pool at `base + idx *
/// stride`. `Pages` is a checkpoint that lives in pool pages
/// ([`PagedRadix::set_state_paged`]): those hold the blob in the KV planes'
/// own record layout, so it ships as page runs over the tier's planes exactly
/// like a KV run, and its RAM bytes are the flat blob's plus tail padding.
enum AuxSpan {
    Flat { base: u64, bytes: u64 },
    Pages(Vec<u32>),
}

/// A restorable aux boundary found by [`PoolTier::probe_aux`].
#[derive(Debug, Clone, Copy)]
pub struct AuxHit {
    pub key: LogicalKey,
    pub end_block: usize,
    pub bytes: u64,
    pub shards: usize,
    /// Of `bytes`, how many come off T2 (priced at the disk rate).
    pub nvme_bytes: u64,
}

/// Ticket for an in-flight restore. Resolved by [`PoolTier::pump`].
pub type TicketId = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RestoreWake {
    pub ticket: TicketId,
    /// true = the radix now holds this prefix through `end_block` - re-match
    /// to adopt. false = nothing new landed; recompute from the existing
    /// GPU depth.
    pub ok: bool,
    /// Chain depth (in blocks) the radix is published to for this prefix.
    pub end_block: usize,
}

struct DeferredStore {
    key: LogicalKey,
    blocks: Vec<BlockId>,
    /// A shard of an aux blob, not a block run: its completion must never
    /// enter the run inventory. Mirrored shards (no `aux_recycle`) used to
    /// fall through to the run arm - a customer's 32 GiB tier reported
    /// 135,280 "runs" of 8 MiB, every blob shard counted as one, and plain
    /// eviction took blob shards as if they were KV.
    shard: bool,
    /// Aux-shard store: the state index to recycle once the last shard of
    /// its blob completes (the countdown lives in `aux_pending`).
    aux_recycle: Option<u32>,
    /// True when this store's pins hold EVICTED blocks - their release
    /// frees. A 2.3 mirror store pins TREE-HELD blocks instead (release
    /// frees nothing), and counting those toward pressure targets made
    /// `pressure_demote` believe frees were coming and skip eviction -
    /// the pool wedged solid at free=0 with 1,176 evictable blocks in the
    /// radix (found on a nemotron probe).
    evicting: bool,
}

struct AuxMeta {
    bytes: u64,
    shards: usize,
    last_used: u64,
}

struct RunMeta {
    loc: Option<Loc>,
    last_used: u64,
}

struct TicketRun {
    op: OpId,
    key: LogicalKey,
    /// The tier-owned destination blocks - `None` when this run JOINED an
    /// in-flight op (the op's starter owns the blocks and publishes them).
    blocks: Option<Vec<BlockId>>,
    done: Option<bool>,
}

struct Ticket {
    /// tokens[..end_block*16] - the chain the publication attaches under.
    tokens: Vec<u32>,
    start_block: usize,
    runs: Vec<TicketRun>,
    /// Aux ticket: loads land in the family's checkpoint slot; no radix
    /// publication here (the family attaches the state itself on wake).
    aux: bool,
    /// Keep the published path's deepest block pinned until the flow lets
    /// go ([`PoolTier::release_hold`]). A hybrid's restored blocks carry no
    /// checkpoint until the blob round attaches one, so to the radix they
    /// are dead KV - the first thing any pressure trims, the blob round's
    /// own checkpoint reservation included (every live Qwen3.8 restore
    /// failed its attach that way).
    hold: bool,
}

/// See the module note. Generic over the transport so every control-flow
/// path runs under the deterministic fake in CPU tests.
pub struct PoolTier<T: XferSink> {
    pub catalog: TierCatalog,
    pub transport: T,
    pub cost: CostModel,
    ns_root: LogicalKey,
    planes: Vec<PlaneDesc>,
    record_stride: u64,
    run_blocks: usize,
    runs: HashMap<LogicalKey, RunMeta>,
    /// Aux inventory: boundary chain key -> blob geometry (catalog holds the
    /// per-shard entries; this holds what probe/restore need to find them).
    aux_meta: HashMap<LogicalKey, AuxMeta>,
    /// The runs each boundary's blob resumes over (root-first), and how many
    /// boundaries own each run. A hybrid restores only through consecutive
    /// runs from the chain head AND the blob, so T1 evicts runs no blob
    /// owns first and an owned run only together with its blobs. Plain LRU
    /// over runs and blobs alike broke chains at the head (the oldest
    /// stamp) and kept blobs whose KV was gone. In step with `aux_meta`,
    /// plus the claims of blobs being admitted.
    blob_runs: HashMap<LogicalKey, Vec<LogicalKey>>,
    run_owners: HashMap<LogicalKey, u32>,
    /// The family's blob (bytes, shards) - see `blobs.rs`.
    blob_geom: Option<(u64, usize)>,
    /// Disk-sourced blob restores seating their blob back in T1, by
    /// boundary, and each expected shard's boundary.
    fills: blobs::Fills,
    fill_of: HashMap<LogicalKey, LogicalKey>,
    /// state_idx -> outstanding shard stores (recycle at zero).
    aux_pending: HashMap<u32, usize>,
    deferred: HashMap<OpId, DeferredStore>,
    /// Circuit breaker: consecutive transport failures. At
    /// [`Self::BREAKER_TRIP`] the tier TRIPS - probes answer None and
    /// demotes evict plain - and serving continues on recompute, said once
    /// at WARN. Any success resets the count. Repeated IO or integrity
    /// failures must degrade the tier loudly, never wedge serving.
    breaker: u32,
    tripped: bool,
    /// Restore riders: op -> tickets waiting on it (single-flight - several
    /// tickets may ride one op).
    op_riders: HashMap<OpId, Vec<TicketId>>,
    tickets: HashMap<TicketId, Ticket>,
    /// Resolved-but-unclaimed restore wakes. Several code paths pump the
    /// transport (resume waits, insert margins, shed); whichever pump
    /// resolves a ticket parks the wake here and the ticket's own waiter
    /// claims it via `take_wake` - no wake is ever consumed by a bystander.
    resolved: HashMap<TicketId, RestoreWake>,
    /// park/wake: parked restore flows per scheduler slot (+ zombies).
    /// Methods live in `restore_flow.rs`.
    pub(super) flows: super::restore_flow::FlowPark,
    /// Published restores' pinned tips (see `Ticket::hold`), by ticket.
    pub(super) holds: HashMap<TicketId, BlockId>,
    /// Pins let go of, released at the next pump (which has the pool).
    unpins: Vec<BlockId>,
    next_ticket: TicketId,
    lru: u64,
    /// decision ledger - why the tier did what it did, not just what it
    /// holds. Written at every decision point below; read once per
    /// scheduler pass by [`Self::report`].
    pub(super) dec: super::accounting::TierDecisions,
    /// Recently evicted keys, so a miss on content we threw away is
    /// reported as such rather than as an ordinary cold miss.
    ghosts: super::accounting::GhostSet,
    /// When the cost model's calibration was last persisted, and what.
    cal_saved: Option<(std::time::Instant, super::cost::Calibration)>,
}

impl<T: XferSink> PoolTier<T> {
    pub fn new(
        ns: &CacheNamespace,
        planes: Vec<PlaneDesc>,
        ram_capacity: u64,
        transport: T,
    ) -> Result<Self, TierRefused> {
        let nvme = tier_nvme().map(|(_, q)| q).unwrap_or(0);
        Self::with_capacities(ns, planes, ram_capacity, nvme, transport)
    }

    /// [`Self::new`] with an explicit T2 ledger capacity (tests and callers
    /// that arm T2 without the process-global election).
    pub fn with_capacities(
        ns: &CacheNamespace,
        planes: Vec<PlaneDesc>,
        ram_capacity: u64,
        nvme_capacity: u64,
        transport: T,
    ) -> Result<Self, TierRefused> {
        if planes
            .iter()
            .any(|p| p.base % 16 != 0 || p.stride % 16 != 0 || p.bytes % 16 != 0)
        {
            return Err(TierRefused::Misaligned);
        }
        let record_stride: u64 = planes.iter().map(|p| p.bytes).sum();
        if record_stride == 0 || record_stride > DEVICE_STAGING_EXTENT_BYTES as u64 {
            return Err(TierRefused::RecordTooLarge {
                record: record_stride,
                planes: planes.len(),
                staging: DEVICE_STAGING_EXTENT_BYTES,
            });
        }
        let run_blocks = ((TARGET_EXTENT_BYTES / record_stride) as usize)
            .clamp(1, MAX_RUN_BLOCKS)
            .min((DEVICE_STAGING_EXTENT_BYTES as u64 / record_stride) as usize)
            .max(1);
        tracing::info!(
            run_blocks,
            record_mib = record_stride as f64 / (1u64 << 20) as f64,
            extent_mib = (run_blocks as u64 * record_stride) as f64 / (1u64 << 20) as f64,
            capacity_gib = ram_capacity as f64 / (1u64 << 30) as f64,
            "KV pool tier armed"
        );
        let mut tier = Self {
            catalog: TierCatalog::new(TierCatalogConfig {
                ram_capacity,
                // T2 capacity mirrors the armed quota so preloaded entries
                // fit the ledger; within-run demote-to-T2 promotion is the
                // Phase-3 follow-up (v1 T2 entries appear via preload +
                // transport write-through)
                nvme_capacity,
            }),
            cost: {
                // 3.2: seed the disk rate from the store's open-time probe,
                // so the first T2 restore is priced by this device instead
                // of by a constant chosen on someone else's hardware.
                let mut c = CostModel::new();
                if let Some(gbs) = transport.t2_device_gbs() {
                    c.seed_nvme(gbs);
                }
                c
            },
            transport,
            ns_root: ns.root(),
            planes,
            record_stride,
            run_blocks,
            runs: HashMap::new(),
            aux_meta: HashMap::new(),
            blob_runs: HashMap::new(),
            run_owners: HashMap::new(),
            blob_geom: None,
            fills: HashMap::new(),
            fill_of: HashMap::new(),
            aux_pending: HashMap::new(),
            breaker: 0,
            tripped: false,
            deferred: HashMap::new(),
            op_riders: HashMap::new(),
            tickets: HashMap::new(),
            resolved: HashMap::new(),
            flows: Default::default(),
            holds: HashMap::new(),
            unpins: Vec::new(),
            next_ticket: 1,
            lru: 0,
            dec: Default::default(),
            ghosts: Default::default(),
            cal_saved: None,
        };
        tier.load_calibration();
        Ok(tier)
    }

    /// Durable write-throughs waiting for disk read slack - an accounting
    /// gauge, and what proves the deferral is real rather than nominal.
    pub fn pending_durable_writes(&self) -> usize {
        self.transport.t2_pending_writes()
    }

    /// flow bookkeeping for the ledger - called by `restore_flow` at
    /// the three points a parked restore can end up.
    pub(super) fn note_park(&mut self, started: bool) {
        if started {
            self.dec.parked += 1;
        } else {
            self.dec.park_refused += 1;
        }
    }

    pub(super) fn note_flow_end(&mut self, ok: bool, bytes: u64, from_nvme: bool) {
        self.dec.moved_bytes += bytes;
        if ok {
            self.dec.resolved_ok += 1;
            self.dec.useful_bytes += bytes;
            if from_nvme {
                self.dec.served_from_nvme += 1;
            } else {
                self.dec.served_from_ram += 1;
            }
        } else {
            self.dec.abandoned += 1;
        }
    }

    /// The view: decisions, occupancy, model honesty, device truth.
    /// Assembled on demand - once per scheduler pass - and never cached, so
    /// it can never go stale behind a wedged tier.
    pub fn report(&self) -> super::accounting::TierReport {
        let t1 = self.catalog.ledger(Tier::Ram);
        let t2 = self.catalog.ledger(Tier::Nvme);
        let (ram_bpus, nvme_bpus) = self.cost.rates_bpus();
        let dev = self.transport.t2_device();
        super::accounting::TierReport {
            decisions: self.dec,
            t1_ready_bytes: t1.ready,
            t1_in_flight_bytes: t1.in_flight,
            t1_reserved_bytes: t1.reserved,
            t1_capacity_bytes: t1.capacity,
            t2_ready_bytes: t2.ready,
            t2_capacity_bytes: t2.capacity,
            resident_runs: self.runs.len() as u64,
            in_flight_demotes: self.deferred.len() as u64,
            open_tickets: self.tickets.len() as u64,
            pending_durable_writes: self.transport.t2_pending_writes() as u64,
            tripped: self.tripped,
            io_failures: self.catalog.counters.io_failures,
            integrity_failures: self.catalog.counters.integrity_failures,
            evictions: self.catalog.counters.evictions,
            single_flight_joins: self.catalog.counters.single_flight_joins,
            stale_completions: self.catalog.counters.stale_completions,
            rate_ram_bpus: ram_bpus,
            rate_nvme_bpus: nvme_bpus,
            prediction_error_pct: self.cost.prediction_error_pct(),
            device_read_gbs: dev.map(|d| d.0).unwrap_or(0.0),
            device_write_gbs: dev.map(|d| d.1).unwrap_or(0.0),
            device_unbuffered: dev.map(|d| d.2).unwrap_or(false),
            t2_written_day_bytes: self.transport.t2_written_today(),
            ghost_keys: self.ghosts.len() as u64,
        }
    }

    /// The namespace chain root - hand to `PagedRadix::set_tier_root` at
    /// pool setup, before any insert.
    pub fn tier_root(&self) -> LogicalKey {
        self.ns_root
    }

    /// Where a key is Ready right now - T1 preferred (faster), else T2.
    fn ready_on(&self, key: &LogicalKey) -> Option<(Tier, u64)> {
        if let Some(b) = self.catalog.ready_bytes(key, Tier::Ram) {
            return Some((Tier::Ram, b));
        }
        self.catalog
            .ready_bytes(key, Tier::Nvme)
            .map(|b| (Tier::Nvme, b))
    }

    pub fn run_blocks(&self) -> usize {
        self.run_blocks
    }

    fn tick(&mut self) -> u64 {
        self.lru += 1;
        self.lru
    }

    /// Consecutive transport failures before the tier trips (circuit
    /// breaker). One flaky transfer must not kill the tier; a persistently
    /// failing device must not keep eating work.
    const BREAKER_TRIP: u32 = 8;

    fn breaker_ok(&mut self) {
        self.breaker = 0;
    }

    fn breaker_fail(&mut self) {
        self.breaker += 1;
        if !self.tripped && self.breaker >= Self::BREAKER_TRIP {
            self.tripped = true;
            tracing::warn!(
                failures = self.breaker,
                "KV tier circuit breaker TRIPPED after repeated transport                  failures - the tier stops demoting and restoring; serving                  continues correct on recompute"
            );
        }
    }

    /// Whether the breaker has taken the tier offline.
    pub fn is_tripped(&self) -> bool {
        self.tripped
    }

    fn spec(&self, block_ids: Vec<u32>, after: Option<CudaEvent>) -> XferSpec {
        XferSpec {
            planes: self.planes.clone(),
            block_ids,
            after,
        }
    }

    // -- aux spans (see AuxSpan) --------------------------------------------

    /// Whole pages per aux shard: as close to `AUX_SHARD` as pages allow.
    fn pages_per_shard(&self) -> usize {
        (Self::AUX_SHARD / self.record_stride.max(1)).max(1) as usize
    }

    fn aux_bytes(&self, s: &AuxSpan) -> u64 {
        match s {
            AuxSpan::Flat { bytes, .. } => *bytes,
            AuxSpan::Pages(p) => p.len() as u64 * self.record_stride,
        }
    }

    fn aux_shards(&self, s: &AuxSpan) -> usize {
        match s {
            AuxSpan::Flat { bytes, .. } => bytes.div_ceil(Self::AUX_SHARD) as usize,
            AuxSpan::Pages(p) => p.len().div_ceil(self.pages_per_shard()),
        }
    }

    /// Shard `i`'s pages (empty for a flat span).
    fn aux_shard_pages<'a>(&self, s: &'a AuxSpan, i: usize) -> &'a [u32] {
        match s {
            AuxSpan::Flat { .. } => &[],
            AuxSpan::Pages(p) => {
                let pps = self.pages_per_shard();
                &p[i * pps..((i + 1) * pps).min(p.len())]
            }
        }
    }

    fn aux_shard_len(&self, s: &AuxSpan, i: usize) -> u64 {
        match s {
            AuxSpan::Flat { bytes, .. } => Self::AUX_SHARD.min(bytes - i as u64 * Self::AUX_SHARD),
            AuxSpan::Pages(_) => self.aux_shard_pages(s, i).len() as u64 * self.record_stride,
        }
    }

    fn aux_shard_spec(&self, s: &AuxSpan, i: usize, after: Option<CudaEvent>) -> XferSpec {
        match s {
            AuxSpan::Flat { base, .. } => XferSpec {
                planes: vec![PlaneDesc {
                    base: base + i as u64 * Self::AUX_SHARD,
                    stride: 16,
                    bytes: self.aux_shard_len(s, i),
                }],
                block_ids: vec![0],
                after,
            },
            AuxSpan::Pages(_) => self.spec(self.aux_shard_pages(s, i).to_vec(), after),
        }
    }

    fn aux_span_ok(&self, s: &AuxSpan) -> bool {
        match s {
            AuxSpan::Flat { base, bytes } => {
                *bytes > 0 && bytes.is_multiple_of(16) && base.is_multiple_of(16)
            }
            AuxSpan::Pages(p) => !p.is_empty(),
        }
    }

    fn run_bytes(&self) -> u64 {
        self.record_stride * self.run_blocks as u64
    }

    // -- demote --------------------------------------------------------------

    /// Evict cached-but-idle prefixes until `pool.free_blocks() >= target`,
    /// demoting each run whose closing leaf goes - the tier-aware form of
    /// the families' evict-ahead loop. `after` fences the first gather
    /// against the compute stream (later stores are stream-ordered behind it
    /// on the tier lane). Returns blocks evicted.
    pub fn pressure_demote(
        &mut self,
        radix: &mut PagedRadix,
        pool: &mut KvPool,
        target: usize,
        mut after: Option<CudaEvent>,
    ) -> (usize, Vec<AuxTaken>) {
        let mut evicted = 0;
        let mut taken: Vec<AuxTaken> = Vec::new();
        self.drop_stale_claims();
        loop {
            // demote pins DEFER their frees to store completion, so count
            // them toward the target or the loop eats the entire radix
            // before a single block physically frees (found live on the
            // first granite probe: one insert's margin call demoted every
            // cached chain - 60 runs - in one burst)
            let pending: usize = self
                .deferred
                .values()
                .filter(|d| d.evicting)
                .map(|d| d.blocks.len())
                .sum();
            if pool.free_blocks() + pending >= target {
                break;
            }
            let path = radix.lru_leaf_path();
            if path.is_empty() {
                break;
            }
            // claim checkpoints on the doomed path before eviction recycles
            // them - the family demotes (or recycles) each AuxTaken. Those
            // the tier can take bind their runs first, so the run demotes
            // below cannot evict this chain's own stored head to make room.
            let r = self.run_blocks;
            for e in &path {
                if e.state_blk.is_some()
                    && let Some(key) = e.tkey
                    && !self.aux_meta.contains_key(&key)
                    && let Some(idx) = radix.take_state(e.node)
                {
                    let runs = self.path_run_keys(&path, e.depth);
                    if e.depth % r == 0 {
                        self.own_runs(key, runs.clone());
                    }
                    taken.push(AuxTaken {
                        key,
                        end_block: e.depth,
                        state_idx: idx,
                        runs,
                    });
                }
            }
            // capture this chain's complete runs ROOT-FIRST: a probe can
            // only resume through consecutive runs from the chain head, so
            // the head must publish first (deepest-first submission left
            // every restore probing a not-yet-stored run 0 - found live)
            self.write_runs(&path, pool, &mut after, true);
            // evict the path bottom-up; a branch point (sibling chain alive)
            // stops the walk - its shared prefix stays serving
            for e in path.iter().rev() {
                match radix.evict_leaf(e.node, pool) {
                    Some(_) => evicted += 1,
                    None => break,
                }
            }
            // drain whatever completed so pins release as we go
            self.pump_completions(radix, pool);
        }
        (evicted, taken)
    }

    /// The exhaustion-cliff shed: press until `want` blocks are free, not
    /// merely promised. One `pressure_demote` + a 50ms drain is not enough
    /// when a parked restore's loads occupy the lane - load-first holds every
    /// demote store behind them, the pins cannot release, and the admitted
    /// request dies with PoolExhausted (found live on a nemotron probe).
    /// This loop pumps, RE-presses (a restore that resolves
    /// mid-wait publishes into the radix - instantly evictable retention),
    /// and only gives up when nothing is in flight anywhere or a hard 2s
    /// cap passes. Blocking the tick at the cliff is deliberate: the
    /// alternative is a killed request, and the real fix (live swap-out)
    /// is 2.4. `state` = (blob base, stride) for hybrid checkpoint blobs;
    /// None recycles boundary state plainly (KV-only families).
    pub fn make_room_blocking(
        &mut self,
        radix: &mut PagedRadix,
        pool: &mut KvPool,
        want: usize,
        state: Option<(u64, u64)>,
        after: &mut dyn FnMut() -> Option<CudaEvent>,
    ) -> bool {
        let hard = std::time::Instant::now() + std::time::Duration::from_secs(2);
        loop {
            self.press(radix, pool, want, state, after);
            self.pump_completions(radix, pool);
            if pool.free_blocks() >= want {
                return true;
            }
            if self.stats().2 == 0 && self.stats().3 == 0 {
                // nothing in flight, nothing evictable - genuinely dry
                let paths = radix.lru_leaf_paths(4);
                let shared: usize = paths
                    .iter()
                    .flatten()
                    .filter(|e| pool.refcount(e.block) > 1)
                    .count();
                let rc1 = (0..pool.capacity())
                    .filter(|&b| pool.refcount(b) == 1)
                    .count();
                let rc2 = (0..pool.capacity())
                    .filter(|&b| pool.refcount(b) > 1)
                    .count();
                tracing::warn!(
                    want,
                    free = pool.free_blocks(),
                    leaves = paths.len(),
                    leaf_blocks = paths.iter().map(|p| p.len()).sum::<usize>(),
                    leaf_shared = shared,
                    pool_rc1 = rc1,
                    pool_rc_multi = rc2,
                    "tier make-room: dry exit"
                );
                return false;
            }
            if std::time::Instant::now() >= hard {
                tracing::warn!(
                    want,
                    free = pool.free_blocks(),
                    "tier make-room: hard cap hit with work still in flight"
                );
                return false;
            }
            std::thread::sleep(std::time::Duration::from_micros(200));
        }
    }

    /// One make-room pass: pressure-demote toward `target` free blocks and
    /// route every claimed checkpoint - blob demote at run boundaries when
    /// `state` gives the pool geometry `(base, stride)`, plain recycle
    /// otherwise. No wait: pins release via later pumps. The insert-margin
    /// evict-ahead uses this directly (a PLAIN evict_lru there discards the
    /// checkpoint blobs and leaves the tier with unusable block runs -
    /// found live on the nemotron probe: probe hits, aux always None,
    /// every repeat recomputed); the cliff shed loops it.
    pub fn press(
        &mut self,
        radix: &mut PagedRadix,
        pool: &mut KvPool,
        target: usize,
        state: Option<(u64, u64)>,
        after: &mut dyn FnMut() -> Option<CudaEvent>,
    ) {
        if radix.state_is_paged() {
            self.press_paged_state(radix, pool, target, after);
        }
        let (_e, taken) = self.pressure_demote(radix, pool, target, after());
        let r = self.run_blocks;
        for a in taken {
            if a.end_block % r != 0 {
                radix.recycle_state(a.state_idx);
            } else if radix.state_is_paged() {
                self.demote_aux_paged(radix, pool, a, after());
            } else if let Some((base, stride)) = state {
                let blob = base + a.state_idx as u64 * stride;
                self.demote_aux(radix, a, blob, stride, after());
            } else {
                self.disown_runs(a.key);
                radix.recycle_state(a.state_idx);
            }
        }
    }

    /// The paged backing's pressure order, run before the LRU-path demotion
    /// (issue #33). Checkpoints hold pool pages there, and on a single
    /// conversation the LRU leaf path IS the live one: demoting it would
    /// ship the newest checkpoint along with the stale ones. So first dead
    /// KV goes (below the deepest checkpoint a hybrid recomputes anyway),
    /// then the stalest checkpoint by its own recency - and only what that
    /// cannot free falls through to the path walk.
    ///
    /// A checkpoint is only restorable from RAM together with the KV it
    /// resumes over (the restore flow's blocks round comes first), so its
    /// path's runs are written through beside the blob. They stay on the
    /// device - the path may still be serving - and so do not count as
    /// freed; once no checkpoint covers them they trim as dead KV, already
    /// safe in T1.
    fn press_paged_state(
        &mut self,
        radix: &mut PagedRadix,
        pool: &mut KvPool,
        target: usize,
        after: &mut dyn FnMut() -> Option<CudaEvent>,
    ) {
        let r = self.run_blocks;
        radix.reclaim(pool);
        loop {
            let pending = self.pending_evict_blocks();
            if pool.free_blocks() + pending >= target {
                return;
            }
            if radix.evict_dead_leaves(pool, target.saturating_sub(pending), 0) > 0 {
                continue;
            }
            let Some(e) = radix.stalest_state() else {
                return;
            };
            let Some(idx) = radix.take_state(e.node) else {
                return;
            };
            match e.tkey {
                Some(key) if e.depth % r == 0 => {
                    // the KV first: root-first, like pressure_demote, so a
                    // probe can walk the chain from its head. Even when the
                    // blob is already in T1 - the slack mirror may have
                    // shipped it bare, and the path is about to turn dead
                    // and trim, so this is the KV's last way out (stored
                    // runs dedup for free).
                    let path = radix.path_entries(e.node);
                    let runs = self.path_run_keys(&path, e.depth);
                    let stored = self.aux_meta.contains_key(&key);
                    if !stored {
                        self.own_runs(key, runs.clone());
                    }
                    let mut ev = after();
                    self.write_runs(&path, pool, &mut ev, false);
                    if stored {
                        radix.recycle_state(idx);
                    } else {
                        let t = AuxTaken {
                            key,
                            end_block: e.depth,
                            state_idx: idx,
                            runs,
                        };
                        self.demote_aux_paged(radix, pool, t, after());
                    }
                }
                // off a run boundary or unkeyed: drop it
                _ => radix.recycle_state(idx),
            }
            // a refused or dropped blob parked its pages; count them now
            radix.reclaim(pool);
        }
    }

    /// Blocks an evicting store will free when it lands - demote pins defer
    /// their frees to completion, so pressure passes count them as freed.
    fn pending_evict_blocks(&self) -> usize {
        self.deferred
            .values()
            .filter(|d| d.evicting)
            .map(|d| d.blocks.len())
            .sum()
    }

    /// 2.3 background write-through: pre-store LRU-retained chains during
    /// slack so a later eviction is a pure block release (the content is
    /// already T1-resident and `demote_run`'s dedup skips the IO). Rides
    /// the same demote machinery - pins release at store completion and the
    /// radix keeps serving throughout; the transport's kick holds stores
    /// whenever load traffic exists (load-first), so a restore is never behind more
    /// than the shallow store pipeline. Aux blobs still ship at eviction
    /// (v1). Bounded: at most `max_runs` submissions per call, LRU chains
    /// first (they evict first, so they are worth pre-storing first).
    /// Returns runs submitted.
    pub fn mirror_slack(
        &mut self,
        radix: &PagedRadix,
        pool: &mut KvPool,
        mut after: Option<CudaEvent>,
        max_runs: usize,
        state: Option<(u64, u64)>,
    ) -> usize {
        if self.tripped || !self.tickets.is_empty() {
            return 0; // never beside a restore; breaker owns the tier
        }
        let r = self.run_blocks;
        let mut submitted = 0usize;
        // blob write-through first: live checkpoints mirror (slot stays
        // attached), one blob per pass. Scanned off the state attachments,
        // not the LRU leaves - the checkpoint pool recycles oldest-first,
        // so the LRU chains are exactly the ones whose blobs are already
        // gone - and before the run scan, whose per-pass cap starved the
        // blob half for the entire busy phase (both found live on the
        // gemma4 pooled smoke). A hybrid's blocks are worthless without
        // their blob; the blob is the priority artifact.
        //
        // The blob goes WITH the KV it resumes over, root-first: the run scan
        // below reaches only LRU leaves, never a live conversation's path,
        // and that path's KV is trimmed as dead the moment its checkpoint
        // recycles - so a blob shipped alone sat in T1 with nothing under
        // it, and every repeat of the prompt missed cold (found live on
        // Qwen3.8: three 150 MB blobs per prompt in T1, none restorable).
        self.drop_stale_claims();
        if radix.state_is_paged() || state.is_some() {
            for (depth, key, idx, node) in radix.state_attachments() {
                if depth % r == 0
                    && let Some(key) = key
                    && !self.aux_meta.contains_key(&key)
                {
                    let path = radix.path_entries(node);
                    let runs = self.path_run_keys(&path, depth);
                    self.own_runs(key, runs.clone());
                    submitted += self.write_runs(&path, pool, &mut after, false);
                    let span = match state {
                        Some((base, stride)) if !radix.state_is_paged() => AuxSpan::Flat {
                            base: base + idx as u64 * stride,
                            bytes: stride,
                        },
                        _ => AuxSpan::Pages(radix.state_pages(idx).to_vec()),
                    };
                    let ev = after.take();
                    let pool = radix.state_is_paged().then_some(&mut *pool);
                    if self.mirror_aux_span(key, depth, span, pool, ev, runs) {
                        break;
                    }
                }
            }
        }
        for path in radix.lru_leaf_paths(4) {
            for lo in (0..(path.len() / r) * r).step_by(r) {
                if submitted >= max_runs {
                    return submitted;
                }
                let run = &path[lo..lo + r];
                if let Some(key) = run[r - 1].tkey
                    && run.iter().all(|e| e.tkey.is_some())
                    && self.demote_run(key, run, pool, &mut after, false)
                {
                    submitted += 1;
                }
            }
        }
        submitted
    }

    /// Store every complete run of `path` root-first - a probe resumes only
    /// through consecutive runs from the chain head, so the head publishes
    /// first. `evicting` = the path is being evicted (its pins count as
    /// freed); otherwise a write-through, what a checkpoint needs in T1
    /// beside its blob. Runs already stored or in flight dedup for free.
    /// Returns runs submitted.
    fn write_runs(
        &mut self,
        path: &[crate::paged_radix::LruPathEntry],
        pool: &mut KvPool,
        after: &mut Option<CudaEvent>,
        evicting: bool,
    ) -> usize {
        let r = self.run_blocks;
        let mut n = 0;
        for lo in (0..(path.len() / r) * r).step_by(r) {
            let run = &path[lo..lo + r];
            if let Some(rk) = run[r - 1].tkey
                && run.iter().all(|x| x.tkey.is_some())
                && self.demote_run(rk, run, pool, after, evicting)
            {
                n += 1;
            }
        }
        n
    }

    /// Capture one run into T1 (asynchronously). Pins the blocks; the pins
    /// release at store completion in `pump`. Every failure mode degrades to
    /// "not demoted" - eviction proceeds either way.
    /// `after` is taken only when a store actually submits - a dedup skip
    /// must leave the fence for the next run that does demote, or its gather
    /// would launch unfenced against the compute stream.
    fn demote_run(
        &mut self,
        key: LogicalKey,
        run: &[crate::paged_radix::LruPathEntry],
        pool: &mut KvPool,
        after: &mut Option<CudaEvent>,
        evicting: bool,
    ) -> bool {
        if self.tripped {
            return false; // eviction proceeds plain; the tier is offline
        }
        let bytes = self.run_bytes();
        // reservation-first; AlreadyPresent = dedup (already stored or in
        // flight - content keys make re-demote free)
        match self.catalog.reserve(key, Tier::Ram, bytes) {
            Ok(()) => {
                // the ledger had the bytes; the arena must have the hole
                if !self.transport.can_place(&[bytes]) && !self.evict_t1(0, &[bytes]) {
                    self.catalog.release_reservation(&key, Tier::Ram);
                    return false;
                }
            }
            Err(ReserveError::AlreadyPresent) => return false,
            Err(ReserveError::Insufficient { .. }) => {
                if !self.evict_t1(bytes, &[bytes])
                    || self.catalog.reserve(key, Tier::Ram, bytes).is_err()
                {
                    return false; // T1 genuinely full of busy/pinned entries
                }
            }
        }
        let ids: Vec<u32> = run.iter().map(|e| e.block).collect();
        let spec = self.spec(ids, after.take());
        if let Err(e) = self.transport.expect_store(key, spec) {
            tracing::warn!(err = ?e, "tier demote: spec rejected");
            self.catalog.release_reservation(&key, Tier::Ram);
            return false;
        }
        let (op, job) = self
            .catalog
            .begin_store_adopt(key, Tier::Ram, bytes)
            .expect("reserved exactly above");
        if let Err(e) = self.transport.submit(job) {
            tracing::warn!(err = ?e, "tier demote: submit rejected");
            self.catalog.cancel_store(op);
            return false;
        }
        // pin only once nothing below can fail - pump is the one release path
        for e in run {
            pool.retain(e.block);
        }
        tracing::debug!(
            op = op.0,
            end_block = run.last().map(|e| e.depth).unwrap_or(0),
            blocks = run.len(),
            bytes,
            "tier demote submitted"
        );
        self.deferred.insert(
            op,
            DeferredStore {
                key,
                blocks: run.iter().map(|e| e.block).collect(),
                shard: false,
                aux_recycle: None,
                evicting,
            },
        );
        true
    }

    /// Shard size for aux blobs - half the staging extent so a shard's spec
    /// always fits with headroom.
    const AUX_SHARD: u64 = 16 << 20;

    /// Demote a claimed checkpoint blob (contiguous device memory at `base`,
    /// `bytes` long, 16-aligned) under its boundary's chain key. The state
    /// index recycles via `pump` once every shard's store completes - or
    /// immediately on refusal, so the pool can never leak a slot.
    pub fn demote_aux(
        &mut self,
        radix: &mut PagedRadix,
        t: AuxTaken,
        base: u64,
        bytes: u64,
        after: Option<CudaEvent>,
    ) {
        self.demote_aux_span(radix, None, t, AuxSpan::Flat { base, bytes }, after);
    }

    /// [`Self::demote_aux`] for a checkpoint that lives in pool pages: the
    /// pages ship as runs over the KV planes and stay pinned until their
    /// shard lands, so they count toward a pressure pass's target like a
    /// demoted KV run's blocks do.
    pub fn demote_aux_paged(
        &mut self,
        radix: &mut PagedRadix,
        pool: &mut KvPool,
        t: AuxTaken,
        after: Option<CudaEvent>,
    ) {
        let pages = radix.state_pages(t.state_idx).to_vec();
        self.demote_aux_span(radix, Some(pool), t, AuxSpan::Pages(pages), after);
    }

    fn demote_aux_span(
        &mut self,
        radix: &mut PagedRadix,
        mut pool: Option<&mut KvPool>,
        t: AuxTaken,
        span: AuxSpan,
        mut after: Option<CudaEvent>,
    ) {
        if !self.aux_span_ok(&span) {
            tracing::warn!(
                bytes = self.aux_bytes(&span),
                "tier aux demote: blob not 16-aligned - recycled unstored"
            );
            self.disown_runs(t.key);
            radix.recycle_state(t.state_idx);
            return;
        }
        if self.aux_meta.contains_key(&t.key) {
            // dedup: this blob is already stored or in flight, bound as is
            radix.recycle_state(t.state_idx);
            return;
        }
        // bound before the shard reservations, which may evict
        self.own_runs(t.key, t.runs.clone());
        let bytes = self.aux_bytes(&span);
        let shards = self.aux_shards(&span);
        // all-or-nothing admission: reserve every shard first
        let mut reserved = 0usize;
        for i in 0..shards {
            let skey = t.key.child_bytes("aux", &(i as u32).to_le_bytes());
            let slen = self.aux_shard_len(&span, i);
            match self.catalog.reserve(skey, Tier::Ram, slen) {
                Ok(()) => reserved += 1,
                Err(ReserveError::AlreadyPresent) => {
                    // dedup: this blob is already stored/in flight
                    for j in 0..reserved {
                        let k = t.key.child_bytes("aux", &(j as u32).to_le_bytes());
                        self.catalog.release_reservation(&k, Tier::Ram);
                    }
                    self.disown_runs(t.key);
                    radix.recycle_state(t.state_idx);
                    return;
                }
                Err(ReserveError::Insufficient { .. }) => {
                    if self.evict_t1(slen, &[])
                        && self.catalog.reserve(skey, Tier::Ram, slen).is_ok()
                    {
                        reserved += 1;
                        continue;
                    }
                    for j in 0..reserved {
                        let k = t.key.child_bytes("aux", &(j as u32).to_le_bytes());
                        self.catalog.release_reservation(&k, Tier::Ram);
                    }
                    self.disown_runs(t.key);
                    radix.recycle_state(t.state_idx);
                    return;
                }
            }
        }
        // the ledger holds every shard; the arena must hold them too
        let lens: Vec<u64> = (0..shards).map(|i| self.aux_shard_len(&span, i)).collect();
        if !self.transport.can_place(&lens) && !self.evict_t1(0, &lens) {
            for j in 0..shards {
                let k = t.key.child_bytes("aux", &(j as u32).to_le_bytes());
                self.catalog.release_reservation(&k, Tier::Ram);
            }
            self.disown_runs(t.key);
            radix.recycle_state(t.state_idx);
            return;
        }
        let mut submitted = 0usize;
        for i in 0..shards {
            let skey = t.key.child_bytes("aux", &(i as u32).to_le_bytes());
            let slen = self.aux_shard_len(&span, i);
            let spec = self.aux_shard_spec(&span, i, after.take());
            let ok = self.transport.expect_store(skey, spec).is_ok();
            let started = if ok {
                self.catalog.begin_store_adopt(skey, Tier::Ram, slen).ok()
            } else {
                None
            };
            match started {
                Some((op, job)) => {
                    if self.transport.submit(job).is_ok() {
                        // a paged blob's pages stay pinned until this shard
                        // lands (a flat blob's slot is simply not recycled
                        // until then)
                        // (only pins actually taken are recorded: completion
                        // releases exactly `blocks`)
                        let pins = match pool.as_deref_mut() {
                            Some(pool) => {
                                let pins = self.aux_shard_pages(&span, i).to_vec();
                                for &b in &pins {
                                    pool.retain(b);
                                }
                                pins
                            }
                            None => Vec::new(),
                        };
                        self.deferred.insert(
                            op,
                            DeferredStore {
                                key: skey,
                                blocks: pins,
                                shard: true,
                                aux_recycle: Some(t.state_idx),
                                evicting: true,
                            },
                        );
                        submitted += 1;
                    } else {
                        self.catalog.cancel_store(op);
                    }
                }
                None => {
                    self.catalog.release_reservation(&skey, Tier::Ram);
                }
            }
        }
        if submitted == 0 {
            self.disown_runs(t.key);
            radix.recycle_state(t.state_idx);
            return;
        }
        self.aux_pending.insert(t.state_idx, submitted);
        let now = self.tick();
        self.aux_meta.insert(
            t.key,
            AuxMeta {
                bytes,
                shards,
                last_used: now,
            },
        );
        tracing::debug!(
            end_block = t.end_block,
            bytes,
            shards,
            "tier aux demote submitted"
        );
    }

    /// 2.3 for blobs: write a live checkpoint's state blob through to T1
    /// without taking or recycling its slot - the checkpoint keeps serving
    /// warm resumes while the copy makes its later eviction free AND its
    /// slot's recycling lossless. Found live on the gemma4 pooled smoke:
    /// the checkpoint pool is far smaller than the block pool, so slots
    /// recycle long before blocks evict and eviction-time blob demotes
    /// never see a blob to ship. Content-keyed dedup makes re-mirroring
    /// free; T1-full just skips (best effort, never churns T1 for it).
    pub fn mirror_aux(
        &mut self,
        key: LogicalKey,
        end_block: usize,
        base: u64,
        bytes: u64,
        after: Option<CudaEvent>,
        runs: Vec<LogicalKey>,
    ) -> bool {
        self.mirror_aux_span(
            key,
            end_block,
            AuxSpan::Flat { base, bytes },
            None,
            after,
            runs,
        )
    }

    fn mirror_aux_span(
        &mut self,
        key: LogicalKey,
        end_block: usize,
        span: AuxSpan,
        mut pool: Option<&mut KvPool>,
        mut after: Option<CudaEvent>,
        runs: Vec<LogicalKey>,
    ) -> bool {
        if self.tripped || !self.aux_span_ok(&span) || self.aux_meta.contains_key(&key) {
            return false;
        }
        // bound before the shard reservations, which may evict
        self.own_runs(key, runs);
        let bytes = self.aux_bytes(&span);
        let shards = self.aux_shards(&span);
        let mut reserved = 0usize;
        for i in 0..shards {
            let skey = key.child_bytes("aux", &(i as u32).to_le_bytes());
            let slen = self.aux_shard_len(&span, i);
            match self.catalog.reserve(skey, Tier::Ram, slen) {
                Ok(()) => reserved += 1,
                Err(ReserveError::Insufficient { .. })
                    if self.evict_t1(slen, &[])
                        && self.catalog.reserve(skey, Tier::Ram, slen).is_ok() =>
                {
                    // a live blob outranks cold block runs: a hybrid's
                    // blocks are worthless without it, and the blob is the
                    // scarcer artifact (the checkpoint pool recycles fast)
                    reserved += 1;
                }
                Err(_) => {
                    // dedup (already stored / in flight) or T1 truly full
                    for j in 0..reserved {
                        let k = key.child_bytes("aux", &(j as u32).to_le_bytes());
                        self.catalog.release_reservation(&k, Tier::Ram);
                    }
                    self.disown_runs(key);
                    return false;
                }
            }
        }
        // the ledger holds every shard; the arena must hold them too
        let lens: Vec<u64> = (0..shards).map(|i| self.aux_shard_len(&span, i)).collect();
        if !self.transport.can_place(&lens) && !self.evict_t1(0, &lens) {
            for j in 0..shards {
                let k = key.child_bytes("aux", &(j as u32).to_le_bytes());
                self.catalog.release_reservation(&k, Tier::Ram);
            }
            self.disown_runs(key);
            return false;
        }
        let mut submitted = 0usize;
        for i in 0..shards {
            let skey = key.child_bytes("aux", &(i as u32).to_le_bytes());
            let slen = self.aux_shard_len(&span, i);
            let spec = self.aux_shard_spec(&span, i, after.take());
            let ok = self.transport.expect_store(skey, spec).is_ok();
            let started = if ok {
                self.catalog.begin_store_adopt(skey, Tier::Ram, slen).ok()
            } else {
                None
            };
            match started {
                Some((op, job)) => {
                    if self.transport.submit(job).is_ok() {
                        // The checkpoint stays attached, so it can be
                        // evicted or stolen while this shard is still being
                        // read: a paged blob's pages are pinned so they
                        // cannot return to the pool mid-store. (A flat slot
                        // has no such guard - it can be refilled under a
                        // mirror in flight - one more reason to page them.)
                        let pins = match pool.as_deref_mut() {
                            Some(pool) => {
                                let pins = self.aux_shard_pages(&span, i).to_vec();
                                for &b in &pins {
                                    pool.retain(b);
                                }
                                pins
                            }
                            None => Vec::new(),
                        };
                        self.deferred.insert(
                            op,
                            DeferredStore {
                                key: skey,
                                blocks: pins,
                                shard: true,
                                aux_recycle: None, // the slot stays attached
                                evicting: false,
                            },
                        );
                        submitted += 1;
                    } else {
                        self.catalog.cancel_store(op);
                    }
                }
                None => {
                    self.catalog.release_reservation(&skey, Tier::Ram);
                }
            }
        }
        if submitted == 0 {
            self.disown_runs(key);
            return false;
        }
        let now = self.tick();
        self.aux_meta.insert(
            key,
            AuxMeta {
                bytes,
                shards,
                last_used: now,
            },
        );
        tracing::debug!(end_block, bytes, shards, "tier aux mirror submitted");
        true
    }

    /// Make room in T1: evict least-recently-probed Ready entries until the
    /// ledger has `need` bytes free, then - if the arena still has no hole
    /// for every extent in `place` - open holes window by window
    /// ([`Self::punch_hole`]). The ledger counts exact bytes; the RAM arena
    /// needs each rounded extent contiguous in one slab, and LRU frees bytes
    /// all over it. Evicting to the ledger alone left a full tier with no
    /// hole for the next 16 MiB shard: every store failed and the breaker
    /// took the tier offline; evicting LRU until a hole appeared drained it.
    fn evict_t1(&mut self, need: u64, place: &[u64]) -> bool {
        if self.catalog.ledger(Tier::Ram).free < need {
            // runs no blob owns first (all of them, for a KV-only family:
            // plain LRU), then blobs, dominated ones first (`blob_victims`)
            // - each turning the runs only it owned into the next victims,
            // deepest first, so a chain shrinks from its tail and never
            // loses the head everything else hangs on
            let mut orphans: Vec<(u64, LogicalKey)> = self
                .runs
                .iter()
                .filter(|(k, _)| !self.run_owners.contains_key(*k))
                .map(|(k, m)| (m.last_used, *k))
                .collect();
            orphans.sort_unstable_by_key(|v| v.0);
            let mut orphans: std::collections::VecDeque<LogicalKey> =
                orphans.into_iter().map(|v| v.1).collect();
            let mut blobs = self.blob_victims().into_iter();
            while self.catalog.ledger(Tier::Ram).free < need {
                if let Some(k) = orphans.pop_front() {
                    self.evict_run_t1(k);
                    continue;
                }
                let Some(b) = blobs.next() else { break };
                let owned = self.blob_runs.get(&b).cloned().unwrap_or_default();
                self.retire_aux(b);
                for k in owned.into_iter().rev() {
                    if !self.run_owners.contains_key(&k) && self.runs.contains_key(&k) {
                        orphans.push_back(k);
                    }
                }
            }
            if self.catalog.ledger(Tier::Ram).free < need {
                return false;
            }
        }
        self.make_placeable(place)
    }

    /// Evict one Ready run from T1. False when the catalog refused (pinned
    /// by a restore in flight).
    fn evict_run_t1(&mut self, key: LogicalKey) -> bool {
        let loc = self.catalog.ready_loc(&key, Tier::Ram);
        if self.catalog.evict(&key, Tier::Ram).is_err() {
            return false;
        }
        if let Some(l) = loc {
            self.transport.free_extent(l);
        }
        // T1 -> T2 promotion: the write-through already put these
        // bytes on disk, so publish the DURABLE copy as readable
        // instead of losing the content until the next restart.
        // Without this the tier writes gigabytes it can never
        // read back within the run - which is exactly what the
        // panel showed the first time it was pointed at a
        // thrashing workload (17.5 GB written, 0 readable).
        // `runs` is the T1 inventory (victim scan + extent
        // ownership), so a promoted key leaves it either way -
        // the catalog's Nvme replica is what keeps the content
        // findable, exactly as a boot-time preload does.
        self.runs.remove(&key);
        let promoted = match self.transport.t2_entry(&key.0) {
            Some((t2loc, bytes, sum)) => self.catalog.preload_ready(
                key,
                Tier::Nvme,
                t2loc,
                super::digest::Checksum(sum),
                bytes,
            ),
            None => false,
        };
        if promoted {
            self.dec.promoted_to_disk += 1;
        } else {
            // nothing durable: the content is gone, so remember
            // it - a later miss here is capacity, not cold
            self.ghosts.record_eviction(key.0);
        }
        true
    }

    /// Make the arena able to take every extent in `place`, in order: plan,
    /// on a copy of the arena, one hole per extent it cannot take as it
    /// stands - the window of Ready entries whose NEWEST entry is oldest
    /// (fewest bytes on a tie), so LRU order holds window by window - and
    /// evict only once the WHOLE plan exists. Evicting toward a plan that
    /// cannot complete frees bytes the store still cannot use: a full tier
    /// that chased holes that way drained itself to nothing during a long
    /// prefill, while every in-flight store blocked the windows it needed.
    fn make_placeable(&mut self, place: &[u64]) -> bool {
        if self.transport.can_place(place) {
            return true;
        }
        #[derive(Clone, Copy, PartialEq)]
        enum Owner {
            Run(LogicalKey),
            Aux(LogicalKey),
        }
        // Ready, evictable extents and who owns them; everything else in the
        // arena (stores in flight, pinned restores' sources) is immovable
        let mut owner: HashMap<Loc, (u64, Owner)> = HashMap::new();
        for (k, m) in &self.runs {
            if let Some(l) = m.loc {
                owner.insert(l, (m.last_used, Owner::Run(*k)));
            }
        }
        for (k, m) in &self.aux_meta {
            for i in 0..m.shards {
                let sk = k.child_bytes("aux", &(i as u32).to_le_bytes());
                if let Some(l) = self.catalog.ready_loc(&sk, Tier::Ram) {
                    owner.insert(l, (m.last_used, Owner::Aux(*k)));
                }
            }
        }
        // the simulated arena: per slab, (offset, len, owner and its last
        // use) - None is immovable (live and not evictable, or a planned hole)
        type Sim = Vec<(u64, Vec<(u64, u64, Option<(Owner, u64)>)>)>;
        let mut sim: Sim = self
            .transport
            .arena_layout()
            .into_iter()
            .map(|s| {
                let ex = s
                    .extents
                    .iter()
                    .map(|&(o, r, l)| (o, r, owner.get(&l).map(|&(used, w)| (w, used))))
                    .collect();
                (s.len, ex)
            })
            .collect();
        let mut plan: Vec<Owner> = Vec::new();
        for &bytes in place {
            let size = self.transport.arena_rounded(bytes);
            let mut best: Option<((u64, u64), usize, u64)> = None;
            for (si, (len, ex)) in sim.iter().enumerate() {
                let mut starts: Vec<u64> = std::iter::once(0)
                    .chain(ex.iter().flat_map(|&(o, r, _)| [o, o + r]))
                    .collect();
                starts.sort_unstable();
                starts.dedup();
                for p in starts {
                    if p + size > *len {
                        break;
                    }
                    let first = ex.partition_point(|&(o, r, _)| o + r <= p);
                    let (mut newest, mut freed) = (0u64, 0u64);
                    let mut clear = true;
                    for &(o, r, who) in &ex[first..] {
                        if o >= p + size {
                            break;
                        }
                        match who {
                            Some((_, used)) => {
                                newest = newest.max(used);
                                freed += r;
                            }
                            None => {
                                clear = false;
                                break;
                            }
                        }
                    }
                    if clear && best.as_ref().is_none_or(|(c, _, _)| (newest, freed) < *c) {
                        best = Some(((newest, freed), si, p));
                    }
                }
            }
            let Some((_, si, p)) = best else {
                return false; // no plan: evict nothing
            };
            // take the window: its owners go to the plan (an aux owner frees
            // every shard of its blob, wherever they sit), the hole is planned
            let ex = &sim[si].1;
            let taken: Vec<Owner> = ex
                .iter()
                .filter(|&&(o, r, _)| o < p + size && o + r > p)
                .filter_map(|&(_, _, w)| w.map(|(w, _)| w))
                .collect();
            for w in taken {
                if !plan.contains(&w) {
                    plan.push(w);
                }
                for (_, ex) in sim.iter_mut() {
                    ex.retain(|&(_, _, who)| who.map(|(o, _)| o) != Some(w));
                }
            }
            let ex = &mut sim[si].1;
            let at = ex.partition_point(|&(o, _, _)| o < p);
            ex.insert(at, (p, size, None));
        }
        // a hole through an owned run takes its blobs along - they cannot
        // restore without it
        let cut: std::collections::HashSet<LogicalKey> = plan
            .iter()
            .filter_map(|w| match w {
                Owner::Run(k) if self.run_owners.contains_key(k) => Some(*k),
                _ => None,
            })
            .collect();
        let broken: Vec<LogicalKey> = if cut.is_empty() {
            Vec::new()
        } else {
            self.blob_runs
                .iter()
                .filter(|(_, rs)| rs.iter().any(|r| cut.contains(r)))
                .map(|(b, _)| *b)
                .collect()
        };
        for b in broken {
            self.retire_aux(b);
        }
        for w in plan {
            match w {
                Owner::Run(k) => {
                    self.evict_run_t1(k);
                }
                Owner::Aux(k) => self.retire_aux(k),
            }
        }
        self.transport.can_place(place)
    }

    /// Retire the aux boundary owning shard key `skey` (any shard key maps
    /// back by prefix scan of the inventory - bounded by resident boundaries).
    fn retire_aux_of(&mut self, skey: &LogicalKey) {
        let owner = self.aux_meta.iter().find_map(|(k, m)| {
            (0..m.shards)
                .any(|i| k.child_bytes("aux", &(i as u32).to_le_bytes()) == *skey)
                .then_some(*k)
        });
        if let Some(k) = owner {
            self.retire_aux(k);
        }
    }

    /// Retire a run whose at-rest bytes failed integrity (the catalog marked
    /// the replica Bad and released its ledger bytes; this clears the marker
    /// and the physical extent so the key can store fresh later).
    fn retire_run(&mut self, key: LogicalKey) {
        let loc = self.runs.get(&key).and_then(|m| m.loc);
        let _ = self.catalog.evict(&key, Tier::Ram); // Bad marker -> Ok(0)
        if let Some(l) = loc {
            self.transport.free_extent(l);
        }
        self.runs.remove(&key);
    }

    // -- probe / elect ------------------------------------------------------

    /// Restorable extension beyond `gpu_blocks` GPU-resident blocks for this
    /// prompt: consecutive T1-Ready runs from the run boundary at or below
    /// the GPU depth. None = nothing restorable.
    pub fn probe(&mut self, tokens: &[u32], gpu_blocks: usize) -> Option<TierHit> {
        use super::accounting::MissReason;
        self.dec.lookups += 1;
        if self.tripped {
            self.dec.record_miss(MissReason::Tripped);
            return None;
        }
        let full = tokens.len().saturating_sub(1) / BLOCK_TOKENS;
        let r = self.run_blocks;
        let start_run = gpu_blocks / r;
        if (start_run + 1) * r > full {
            // no complete run beyond the GPU depth fits - a restore would
            // deliver nothing the pool does not already hold
            self.dec.record_miss(MissReason::NoNewTokens);
            return None;
        }
        // chain keys along the prompt at run boundaries
        let mut key = self.ns_root;
        let mut keys_at = Vec::new();
        for b in 0..(full / r) * r {
            key = key.child(&tokens[b * BLOCK_TOKENS..(b + 1) * BLOCK_TOKENS]);
            if (b + 1) % r == 0 {
                keys_at.push(key);
            }
        }
        let mut hit_keys = Vec::new();
        let mut bytes = 0u64;
        let mut nvme_bytes = 0u64;
        let mut end = start_run * r;
        for (ri, k) in keys_at.iter().enumerate().skip(start_run) {
            match self.ready_on(k) {
                Some((t, b)) => {
                    hit_keys.push(*k);
                    bytes += b;
                    if t == Tier::Nvme {
                        nvme_bytes += b;
                    }
                    end = (ri + 1) * r;
                }
                None => break, // runs must be consecutive to resume through
            }
        }
        if hit_keys.is_empty() || end <= gpu_blocks {
            // classify: did we hold the first run we wanted and throw it
            // away? That is the alarm, and it reads completely
            // differently from cold traffic on the same counter.
            let wanted = keys_at.get(start_run);
            let ghost = wanted.is_some_and(|k| self.ghosts.contains(&k.0));
            self.dec.record_miss(if ghost {
                MissReason::Ghost
            } else if hit_keys.is_empty() {
                MissReason::Cold
            } else {
                MissReason::NoNewTokens
            });
            return None;
        }
        self.dec.hits += 1;
        let now = self.tick();
        for k in &hit_keys {
            if let Some(m) = self.runs.get_mut(k) {
                m.last_used = now;
            }
        }
        tracing::debug!(
            gpu_blocks,
            start_block = start_run * r,
            end_block = end,
            runs = hit_keys.len(),
            bytes,
            "tier probe hit"
        );
        Some(TierHit {
            start_block: start_run * r,
            end_block: end,
            keys: hit_keys,
            bytes,
            nvme_bytes,
        })
    }

    /// Restore an aux blob into the family's freshly allocated checkpoint
    /// slot (`dst_base`, capacity >= blob bytes). Round two of a hybrid
    /// restore - call after the block publication wake, with the boundary's
    /// radix node present and `attach_state` done. The wake carries no radix
    /// publication; the family's existing resume path reads the slot.
    pub fn begin_restore_aux(
        &mut self,
        hit: &AuxHit,
        dst_base: u64,
        after: Option<CudaEvent>,
    ) -> Option<TicketId> {
        let span = AuxSpan::Flat {
            base: dst_base,
            bytes: hit.bytes,
        };
        self.begin_restore_aux_span(hit, span, after)
    }

    /// [`Self::begin_restore_aux`] into a reserved checkpoint's pool pages.
    /// `None` - and the caller recycles the reservation - when the pages
    /// cannot hold what was stored (a blob stored under another geometry).
    pub fn begin_restore_aux_paged(
        &mut self,
        hit: &AuxHit,
        pages: &[u32],
        after: Option<CudaEvent>,
    ) -> Option<TicketId> {
        let span = AuxSpan::Pages(pages.to_vec());
        if self.aux_bytes(&span) != hit.bytes || self.aux_shards(&span) != hit.shards {
            tracing::debug!(
                stored = hit.bytes,
                pages = pages.len(),
                "tier aux restore: stored blob does not match the checkpoint's pages"
            );
            return None;
        }
        self.begin_restore_aux_span(hit, span, after)
    }

    fn begin_restore_aux_span(
        &mut self,
        hit: &AuxHit,
        span: AuxSpan,
        mut after: Option<CudaEvent>,
    ) -> Option<TicketId> {
        let ticket = self.next_ticket;
        let mut runs: Vec<TicketRun> = Vec::with_capacity(hit.shards);
        for i in 0..hit.shards {
            let skey = hit.key.child_bytes("aux", &(i as u32).to_le_bytes());
            let src = self.ready_on(&skey).map(|(t, _)| t).unwrap_or(Tier::Ram);
            match self
                .catalog
                .begin_load(skey, src, LoadDst::Gpu, WaiterId(ticket))
            {
                Ok(LoadStart::Started { op, job }) => {
                    let spec = self.aux_shard_spec(&span, i, after.take());
                    if self.transport.expect_load(skey, spec).is_err()
                        || self.transport.submit(job).is_err()
                    {
                        self.catalog.cancel_waiter(op, WaiterId(ticket));
                        self.unwind_restore(&runs, ticket, &mut KvPool::with_blocks(0));
                        return None;
                    }
                    runs.push(TicketRun {
                        op,
                        key: skey,
                        blocks: None,
                        done: None,
                    });
                    self.op_riders.entry(op).or_default().push(ticket);
                }
                Ok(LoadStart::Joined { op }) => {
                    runs.push(TicketRun {
                        op,
                        key: skey,
                        blocks: None,
                        done: None,
                    });
                    self.op_riders.entry(op).or_default().push(ticket);
                }
                Err(e) => {
                    tracing::debug!(err = ?e, shard = i, "tier aux restore: load rejected");
                    self.unwind_restore(&runs, ticket, &mut KvPool::with_blocks(0));
                    return None;
                }
            }
        }
        if runs.is_empty() {
            return None;
        }
        self.next_ticket += 1;
        tracing::debug!(
            ticket,
            shards = runs.len(),
            end_block = hit.end_block,
            "tier aux restore started"
        );
        self.tickets.insert(
            ticket,
            Ticket {
                tokens: Vec::new(),
                start_block: hit.end_block,
                runs,
                aux: true,
                hold: false,
            },
        );
        Some(ticket)
    }

    /// The cost-model election for a probe hit under the current queues.
    pub fn elect(&mut self, hit: &TierHit, gpu_blocks: usize) -> Election {
        let e = self.cost.elect(HitShape {
            restore_bytes: hit.bytes,
            restore_tokens: hit.new_tokens(gpu_blocks),
            queued_bytes: self.catalog.ledger(Tier::Ram).in_flight,
            nvme_bytes: hit.nvme_bytes,
        });
        if e.is_restore() {
            self.dec.elected_restore += 1;
        } else {
            self.dec.elected_recompute += 1;
        }
        e
    }

    /// The election for a hybrid family's hit, which is usable only down to
    /// the deepest boundary whose state blob is resident (`aux`, from
    /// [`Self::probe_aux`] filtered by the family's resume rules). No such
    /// boundary turns the probe's hit into a [`MissReason::NoState`] miss -
    /// the blocks are there but nothing can resume on them. Otherwise the
    /// hit is cut to the boundary, priced with the blob, and the arm counted
    /// like [`Self::elect`]'s. `Some((cut hit, est_us))` when restore wins.
    ///
    /// [`MissReason::NoState`]: super::accounting::MissReason::NoState
    pub fn elect_hybrid(&mut self, hit: &TierHit, aux: Option<&AuxHit>) -> Option<(TierHit, f64)> {
        let Some(aux) = aux else {
            self.dec.hits = self.dec.hits.saturating_sub(1);
            self.dec.record_miss(super::accounting::MissReason::NoState);
            return None;
        };
        let r = self.run_blocks;
        let n_runs = aux.end_block / r;
        let per_run = hit.bytes / hit.keys.len().max(1) as u64;
        let cut = TierHit {
            start_block: 0,
            end_block: aux.end_block,
            bytes: per_run * n_runs as u64,
            keys: hit.keys[..n_runs.min(hit.keys.len())].to_vec(),
            // runs are equal-sized, so the cut hit keeps the same share of
            // disk-sourced bytes as the full one
            nvme_bytes: hit.nvme_bytes.min(per_run * n_runs as u64),
        };
        let e = self.cost.elect(HitShape {
            restore_bytes: cut.bytes + aux.bytes,
            restore_tokens: (aux.end_block * BLOCK_TOKENS) as u32,
            queued_bytes: self.catalog.ledger(Tier::Ram).in_flight,
            nvme_bytes: cut.nvme_bytes + aux.nvme_bytes,
        });
        match e {
            Election::Restore { est_us, .. } => {
                self.dec.elected_restore += 1;
                Some((cut, est_us))
            }
            Election::Recompute { .. } => {
                self.dec.elected_recompute += 1;
                None
            }
        }
    }

    /// An elected restore the block pool cannot seat, even after pressing
    /// retention into the tier - counted with the reservation refusals.
    pub fn refuse_park(&mut self) {
        self.dec.park_refused += 1;
    }

    // -- restore ------------------------------------------------------------

    /// Start (or join) restoring `hit`. Allocates the destination blocks per
    /// run from the pool now (reservation-first) - on exhaustion or any
    /// submit failure everything unwinds and the caller recomputes; nothing
    /// stays half-started. `after` fences the first H2D+scatter against the
    /// compute stream. Returns the ticket `pump` will wake.
    pub fn begin_restore(
        &mut self,
        hit: &TierHit,
        tokens: &[u32],
        pool: &mut KvPool,
        mut after: Option<CudaEvent>,
    ) -> Option<TicketId> {
        let ticket = self.next_ticket;
        let mut runs: Vec<TicketRun> = Vec::with_capacity(hit.keys.len());
        for key in &hit.keys {
            let src = self.ready_on(key).map(|(t, _)| t).unwrap_or(Tier::Ram);
            match self
                .catalog
                .begin_load(*key, src, LoadDst::Gpu, WaiterId(ticket))
            {
                Ok(LoadStart::Started { op, job }) => {
                    // the destination reservation for this run
                    let mut blocks = Vec::with_capacity(self.run_blocks);
                    let mut exhausted = false;
                    for _ in 0..self.run_blocks {
                        match pool.alloc() {
                            Ok(b) => blocks.push(b),
                            Err(_) => {
                                exhausted = true;
                                break;
                            }
                        }
                    }
                    if exhausted {
                        // the pool affords a PREFIX of the hit - take it.
                        // A working set slightly over VRAM is exactly where
                        // the tier lives; all-or-nothing here refused an
                        // 11-of-12-run restore on the first live probe.
                        for b in blocks {
                            pool.release(b);
                        }
                        self.catalog.cancel_waiter(op, WaiterId(ticket));
                        if runs.is_empty() {
                            return None; // not even one run fits - recompute
                        }
                        tracing::debug!(kept_runs = runs.len(), "tier restore truncated to fit");
                        break;
                    }
                    let spec = self.spec(blocks.clone(), after.take());
                    if self.transport.expect_load(*key, spec).is_err()
                        || self.transport.submit(job).is_err()
                    {
                        tracing::debug!(run = runs.len(), "tier begin_restore: submit rejected");
                        for b in blocks {
                            pool.release(b);
                        }
                        self.catalog.cancel_waiter(op, WaiterId(ticket));
                        self.unwind_restore(&runs, ticket, pool);
                        return None;
                    }
                    runs.push(TicketRun {
                        op,
                        key: *key,
                        blocks: Some(blocks),
                        done: None,
                    });
                    self.op_riders.entry(op).or_default().push(ticket);
                }
                Ok(LoadStart::Joined { op }) => {
                    // single-flight: the in-flight op's starter owns the
                    // destination blocks and publishes them for everyone
                    runs.push(TicketRun {
                        op,
                        key: *key,
                        blocks: None,
                        done: None,
                    });
                    self.op_riders.entry(op).or_default().push(ticket);
                }
                Err(e) => {
                    tracing::debug!(err = ?e, run = runs.len(), "tier begin_restore: load rejected");
                    self.unwind_restore(&runs, ticket, pool);
                    return None;
                }
            }
        }
        if runs.is_empty() {
            return None;
        }
        let end_block = hit.start_block + runs.len() * self.run_blocks;
        self.next_ticket += 1;
        tracing::debug!(
            ticket,
            runs = runs.len(),
            start_block = hit.start_block,
            end_block,
            "tier restore started"
        );
        self.tickets.insert(
            ticket,
            Ticket {
                tokens: tokens[..end_block * BLOCK_TOKENS].to_vec(),
                start_block: hit.start_block,
                runs,
                aux: false,
                hold: false,
            },
        );
        Some(ticket)
    }

    /// Roll back a half-built restore: release owned blocks, withdraw the
    /// ticket's waiter from every op it touched (ops other tickets ride
    /// continue; sole-waiter ops tear down).
    fn unwind_restore(&mut self, runs: &[TicketRun], ticket: TicketId, pool: &mut KvPool) {
        for r in runs {
            if let Some(blocks) = &r.blocks {
                for &b in blocks {
                    pool.release(b);
                }
            }
            self.catalog.cancel_waiter(r.op, WaiterId(ticket));
            if let Some(riders) = self.op_riders.get_mut(&r.op) {
                riders.retain(|t| *t != ticket);
                if riders.is_empty() {
                    self.op_riders.remove(&r.op);
                }
            }
        }
    }

    // -- pump ---------------------------------------------------------------

    /// Drive completions: release demote pins, publish finished restores
    /// into the radix, park ticket wakes for `take_wake`. Call from every
    /// serving-side touch point (resume, insert, shed) - any pump advances
    /// everyone's work; only `take_wake` hands a specific ticket's result
    /// to its waiter.
    pub fn pump_completions(&mut self, radix: &mut PagedRadix, pool: &mut KvPool) {
        self.save_calibration();
        for b in self.unpins.drain(..) {
            pool.release(b);
        }
        for k in self.transport.take_t2_evictions() {
            let _ = self.catalog.evict(&LogicalKey(k), Tier::Nvme);
            // the durable copy is gone now, so this is the moment the
            // content is truly lost (a T1 eviction that PROMOTED did not
            // lose anything and must not be counted as a capacity miss)
            self.ghosts.record_eviction(k);
        }
        for (k, loc, sum, len) in self.transport.take_t2_promotions() {
            let key = LogicalKey(k);
            if self.adopt_shard_promotion(key, loc, sum, len) {
                continue; // a blob shard: seated whole by `settle_fills`
            }
            self.lru += 1;
            if self
                .catalog
                .preload_ready(key, Tier::Ram, loc, super::digest::Checksum(sum), len)
            {
                // eviction walks the runs/aux inventories, so the promoted
                // entry must appear there or its bytes could never free
                self.runs.entry(key).or_insert(RunMeta {
                    loc: Some(loc),
                    last_used: self.lru,
                });
            } else {
                // logically full or already resident - hand the extent back
                self.transport.free_extent(loc);
            }
        }
        self.settle_fills();
        let completions = self.transport.poll();
        for c in completions {
            let op = c.op;
            let no_room = matches!(c.outcome, super::transport::IoOutcome::NoRoom);
            let catalog_wakes = self.catalog.on_completion(c);
            // demote resolution: release the pins; on success record the run
            if let Some(d) = self.deferred.remove(&op) {
                for &b in &d.blocks {
                    pool.release(b);
                }
                if d.shard {
                    // aux shard store: count down; recycle the state index
                    // once its whole blob is off the device's hands (a
                    // mirrored blob's slot stays attached - nothing to count)
                    if let Some(idx) = d.aux_recycle
                        && let Some(n) = self.aux_pending.get_mut(&idx)
                    {
                        *n -= 1;
                        if *n == 0 {
                            self.aux_pending.remove(&idx);
                            radix.recycle_state(idx);
                        }
                    }
                    if self.catalog.ready_bytes(&d.key, Tier::Ram).is_none() {
                        // a failed shard poisons the whole blob for probes -
                        // retire the boundary's inventory entry
                        tracing::debug!(op = op.0, "tier aux shard failed - boundary unusable");
                        self.retire_aux_of(&d.key);
                    }
                    continue;
                }
                if self.catalog.ready_bytes(&d.key, Tier::Ram).is_some() {
                    self.breaker_ok();
                    let now = self.tick();
                    let loc = self.catalog.ready_loc(&d.key, Tier::Ram);
                    self.runs.insert(
                        d.key,
                        RunMeta {
                            loc,
                            last_used: now,
                        },
                    );
                    tracing::debug!(
                        op = op.0,
                        resident_runs = self.runs.len(),
                        t1_ready_bytes = self.catalog.ledger(Tier::Ram).ready,
                        "tier run resident"
                    );
                } else if no_room {
                    // capacity, not a fault: the run is simply not kept
                    tracing::debug!(op = op.0, "tier demote found no room");
                } else {
                    self.breaker_fail();
                    tracing::debug!(op = op.0, "tier demote did not publish");
                }
                continue;
            }
            // restore resolution
            let riders = self.op_riders.remove(&op).unwrap_or_default();
            let mut integrity_key = None;
            let mut resolved = Vec::new();
            for w in catalog_wakes {
                let WaiterId(tid) = w.waiter;
                if !riders.contains(&tid) {
                    continue;
                }
                let Some(t) = self.tickets.get_mut(&tid) else {
                    continue;
                };
                let ok = w.result == LoadResult::Ok;
                if !ok {
                    tracing::debug!(op = op.0, result = ?w.result, "tier load wake non-ok");
                }
                if ok {
                    self.breaker = 0;
                } else {
                    self.breaker += 1;
                    if !self.tripped && self.breaker >= Self::BREAKER_TRIP {
                        self.tripped = true;
                        tracing::warn!(
                            "KV tier circuit breaker TRIPPED - serving continues on recompute"
                        );
                    }
                }
                for r in t.runs.iter_mut().filter(|r| r.op == op) {
                    r.done = Some(ok);
                    if w.result == LoadResult::Integrity {
                        integrity_key = Some(r.key);
                    }
                }
                if t.runs.iter().all(|r| r.done.is_some()) {
                    resolved.push(tid);
                }
            }
            if let Some(k) = integrity_key {
                self.retire_run(k);
            }
            for tid in resolved {
                let w = self.resolve_ticket(tid, radix, pool);
                self.resolved.insert(tid, w);
            }
        }
        // a paged checkpoint recycled above parked its pages; this is the
        // first moment with a pool at hand to give them back
        radix.reclaim(pool);
    }

    /// Claim a resolved ticket's wake (see `pump_completions`).
    pub fn take_wake(&mut self, ticket: TicketId) -> Option<RestoreWake> {
        self.resolved.remove(&ticket)
    }

    /// All of a ticket's ops finished - publish what landed, in chain order,
    /// stopping at the first failure or hole. The wake's `end_block` is the
    /// depth the radix actually holds; the family re-matches against that.
    fn resolve_ticket(
        &mut self,
        tid: TicketId,
        radix: &mut PagedRadix,
        pool: &mut KvPool,
    ) -> RestoreWake {
        let t = self.tickets.remove(&tid).expect("resolved while present");
        if t.aux {
            // aux blob: no publication - the family attaches the checkpoint
            // (the wake's end_block echoes the boundary)
            let ok = t.runs.iter().all(|x| x.done == Some(true));
            if let Some(first) = t.runs.first() {
                self.fill_resolved(&first.key);
            }
            return RestoreWake {
                ticket: tid,
                ok,
                end_block: if ok { t.start_block } else { 0 },
            };
        }
        let r = self.run_blocks;
        let mut depth = t.start_block;
        let mut chain_intact = true;
        for (i, run) in t.runs.iter().enumerate() {
            let start = t.start_block + i * r;
            let landed = run.done == Some(true);
            match (&run.blocks, landed && chain_intact) {
                (Some(blocks), true) => {
                    let _ = radix.insert_extension(&t.tokens, start, blocks, pool);
                    // the radix is the source of truth for what is reachable
                    // (insert_extension refuses holes; a hash collision can
                    // cut a publish short) - read it rather than assume
                    if radix.chain_depth(&t.tokens) >= start + r {
                        depth = start + r;
                    } else {
                        chain_intact = false; // prefix evicted mid-flight
                    }
                    for &b in blocks {
                        pool.release(b); // radix holds its own refs now
                    }
                }
                (Some(blocks), false) => {
                    for &b in blocks {
                        pool.release(b);
                    }
                    chain_intact = false;
                }
                (None, true) => {
                    // joined run: the starter publishes; if its ticket
                    // resolved first the chain is present, otherwise the
                    // family's re-match after its wake will see the depth -
                    // this wake only promises what is provably there now
                    if radix.chain_depth(&t.tokens) >= start + r {
                        depth = start + r;
                    } else {
                        chain_intact = false;
                    }
                }
                (None, false) => chain_intact = false,
            }
        }
        if t.hold
            && depth > t.start_block
            && let Some(b) = radix.block_at(&t.tokens, depth * BLOCK_TOKENS)
        {
            pool.retain(b);
            self.holds.insert(tid, b);
        }
        RestoreWake {
            ticket: tid,
            ok: depth > t.start_block,
            end_block: depth,
        }
    }

    /// Pin the path `ticket` publishes until [`Self::release_hold`] (see
    /// `Ticket::hold`) - for a restore that attaches a checkpoint after.
    pub fn hold_on_publish(&mut self, ticket: TicketId) {
        if let Some(t) = self.tickets.get_mut(&ticket) {
            t.hold = true;
        }
    }

    /// Let go of `ticket`'s pin, if it took one; the next pump releases it.
    pub fn release_hold(&mut self, ticket: TicketId) {
        if let Some(b) = self.holds.remove(&ticket) {
            self.unpins.push(b);
        }
    }

    /// The full snapshot for /metrics and the Studio panel.
    pub fn tier_stats(&self) -> TierStats {
        let c = &self.catalog.counters;
        TierStats {
            resident_runs: self.runs.len() as u64,
            ready_bytes: self.catalog.ledger(Tier::Ram).ready,
            in_flight_demotes: self.deferred.len() as u64,
            open_tickets: self.tickets.len() as u64,
            tripped: self.tripped,
            single_flight_joins: c.single_flight_joins,
            io_failures: c.io_failures,
            integrity_failures: c.integrity_failures,
            evictions: c.evictions,
            stale_completions: c.stale_completions,
            t2_written_day_bytes: self.transport.t2_written_today(),
        }
    }

    /// witnesses: (T1 ready runs, T1 ready bytes, in-flight demotes,
    /// open restore tickets).
    pub fn stats(&self) -> (usize, u64, usize, usize) {
        (
            self.runs.len(),
            self.catalog.ledger(Tier::Ram).ready,
            self.deferred.len(),
            self.tickets.len(),
        )
    }
}

impl PoolTier<RamTransport> {
    /// Restart persistence: preload every entry the T2 store recovered into the
    /// catalog as `Ready` on the Nvme tier - the restart-persistence hookup.
    /// Returns entries preloaded (skips count separately, loudly).
    pub fn preload_from_t2(&mut self) -> usize {
        let entries: Vec<([u8; 32], Loc, u64, [u8; 32])> = match self.transport.t2() {
            Some(t2) => t2
                .live_iter()
                .map(|(k, _g, loc, len, sum)| (*k, loc, len, sum))
                .collect(),
            None => return 0,
        };
        let mut loaded = 0usize;
        let mut skipped = 0usize;
        for (k, loc, len, sum) in entries.into_iter() {
            if self.catalog.preload_ready(
                super::digest::LogicalKey(k),
                Tier::Nvme,
                loc,
                super::digest::Checksum(sum),
                len,
            ) {
                loaded += 1;
            } else {
                skipped += 1;
            }
        }
        if skipped > 0 {
            tracing::warn!(skipped, "T2 preload: entries skipped (ledger/duplicates)");
        }
        tracing::info!(
            loaded,
            "T2 preload: recovered entries visible to the catalog"
        );
        loaded
    }
}

mod blobs;
mod calibration;

#[cfg(test)]
mod hybrid_tests;

#[cfg(test)]
mod tests {
    use super::super::digest::{IdentityDigest, IdentityFields, PrivacyScope};
    use super::super::transport::FakeTransport;
    use super::*;
    use crate::kv_pool::BlockTable;

    // The fake accepts specs and owns no physical extents - every control
    // path of the tier runs deterministically without a GPU; the byte-layout
    // truth is the GPU gate's job (gpu_kv_tier_roundtrip).
    impl XferSink for FakeTransport {
        fn expect_store(&mut self, _key: LogicalKey, _spec: XferSpec) -> Result<(), SubmitError> {
            Ok(())
        }
        fn expect_load(&mut self, _key: LogicalKey, _spec: XferSpec) -> Result<(), SubmitError> {
            Ok(())
        }
        fn free_extent(&mut self, _loc: Loc) {}
    }

    pub(super) fn ns() -> CacheNamespace {
        CacheNamespace {
            identity: IdentityDigest::compute(&IdentityFields {
                model_tensors: b"pool-tier-test",
                adapter: b"",
                architecture: b"synthetic",
                cache_schema: b"2x2MiB",
                layout_abi: 1,
                tokenizer: b"none",
            }),
            scope: PrivacyScope::Shared,
        }
    }

    /// 2 planes x 2 MiB per block -> record 4 MiB -> run_blocks = 4.
    pub(super) fn planes() -> Vec<PlaneDesc> {
        vec![
            PlaneDesc {
                base: 0,
                stride: 2 << 20,
                bytes: 2 << 20,
            },
            PlaneDesc {
                base: 1 << 30,
                stride: 2 << 20,
                bytes: 2 << 20,
            },
        ]
    }

    pub(super) fn tier(capacity: u64) -> PoolTier<FakeTransport> {
        PoolTier::new(&ns(), planes(), capacity, FakeTransport::new()).unwrap()
    }

    /// A chain of `n` full blocks (+1 tail token): prefill fresh pool blocks,
    /// insert into the radix, drop the slot's refs - tree-held, evictable.
    pub(super) fn cached_chain(
        radix: &mut PagedRadix,
        pool: &mut KvPool,
        seed: u32,
        n: usize,
    ) -> Vec<u32> {
        let tokens: Vec<u32> = (0..n * BLOCK_TOKENS)
            .map(|i| seed * 10_000 + i as u32)
            .chain([9])
            .collect();
        let mut t = BlockTable::new();
        t.ensure(n * BLOCK_TOKENS - 1, pool).unwrap();
        radix.insert(&tokens, t.blocks(), pool);
        t.clear(pool);
        tokens
    }

    pub(super) fn armed_radix(t: &PoolTier<FakeTransport>) -> PagedRadix {
        let mut r = PagedRadix::new();
        r.set_tier_root(t.tier_root());
        r
    }

    // -- paged checkpoints (issue #33) ---------------------------------------

    /// A checkpoint that lives in pool pages goes to T1 as page runs, with
    /// the KV it resumes over; its pages stay pinned until the store lands,
    /// come back to the pool after, and restore into a fresh reservation.
    #[test]
    fn a_paged_checkpoint_ships_with_its_kv_and_restores_into_fresh_pages() {
        let mut t = tier(256 << 20);
        let mut pool = KvPool::with_blocks(64);
        let mut radix = armed_radix(&t);
        // 6 pages x 4 MiB records = 24 MiB: shards of 4 pages, so 2 (4 + 2)
        radix.set_state_paged(4, 6);
        let tokens = cached_chain(&mut radix, &mut pool, 1, 4);
        let s0 = radix
            .attach_state_with_pool(&tokens, 4 * BLOCK_TOKENS, &mut pool)
            .expect("checkpoint");
        let pages = radix.state_pages(s0).to_vec();
        assert_eq!(pages.len(), 6);
        t.press(&mut radix, &mut pool, 64, None, &mut || None);
        for &p in &pages {
            assert_eq!(pool.refcount(p), 2, "owned by the index and pinned");
        }
        t.transport.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        assert_eq!(
            pool.free_blocks(),
            64,
            "the pages and the chain all came back"
        );
        let hit = t.probe(&tokens, 0).expect("the KV went to T1 with it");
        assert_eq!(hit.end_block, 4);
        let aux = t.probe_aux(&tokens, 4).expect("blob resident");
        assert_eq!((aux.end_block, aux.shards, aux.bytes), (4, 2, 24 << 20));
        // round two into a fresh reservation's pages
        let c = radix
            .reserve_state_slot_with_pool(&mut pool)
            .expect("reserve");
        let fresh = radix.state_pages(c).to_vec();
        assert!(
            t.begin_restore_aux_paged(&aux, &fresh[..5], None).is_none(),
            "a checkpoint of another size never loads"
        );
        let ticket = t
            .begin_restore_aux_paged(&aux, &fresh, None)
            .expect("aux ticket");
        t.transport.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        assert_eq!(t.take_wake(ticket).map(|w| w.ok), Some(true));
        t.catalog.check_invariants();
    }

    /// The write-through mirror copies a checkpoint that stays attached, so it
    /// can be evicted mid-store: its pages must not reach the free list until
    /// the store has read them.
    #[test]
    fn a_mirrored_paged_checkpoint_keeps_its_pages_until_the_store_lands() {
        let mut t = tier(256 << 20);
        let mut pool = KvPool::with_blocks(64);
        let mut radix = armed_radix(&t);
        radix.set_state_paged(4, 3);
        let tokens = cached_chain(&mut radix, &mut pool, 1, 4);
        let s0 = radix
            .attach_state_with_pool(&tokens, 4 * BLOCK_TOKENS, &mut pool)
            .expect("checkpoint");
        let pages = radix.state_pages(s0).to_vec();
        t.mirror_slack(&radix, &mut pool, None, 0, None);
        // the checkpoint is evicted with its chain while the store is open
        while radix.evict_lru(&mut pool).is_some() {}
        for &p in &pages {
            assert_eq!(pool.refcount(p), 1, "the mirror's pin holds it");
        }
        t.transport.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        for &p in &pages {
            assert_eq!(pool.refcount(p), 0, "free once the store read it");
        }
        assert!(t.probe_aux(&tokens, 4).is_some(), "and the copy is whole");
    }

    #[test]
    fn run_tiling_adapts_to_record_size() {
        assert_eq!(
            tier(256 << 20).run_blocks(),
            4,
            "16 MiB target / 4 MiB record"
        );
        // a 20 MiB record still tiers (1-block runs)...
        let fat = vec![PlaneDesc {
            base: 0,
            stride: 20 << 20,
            bytes: 20 << 20,
        }];
        let t = PoolTier::new(&ns(), fat, 256 << 20, FakeTransport::new()).unwrap();
        assert_eq!(t.run_blocks(), 1);
        // ...a 40 MiB one cannot (exceeds the staging extent) and refuses
        let huge = vec![PlaneDesc {
            base: 0,
            stride: 40 << 20,
            bytes: 40 << 20,
        }];
        assert!(matches!(
            PoolTier::new(&ns(), huge, 256 << 20, FakeTransport::new()),
            Err(TierRefused::RecordTooLarge { .. })
        ));
    }

    #[test]
    fn pressure_demote_captures_closing_runs_and_defers_release() {
        let mut t = tier(256 << 20);
        let mut pool = KvPool::with_blocks(64);
        let mut radix = armed_radix(&t);
        let tokens = cached_chain(&mut radix, &mut pool, 1, 8);
        assert_eq!(pool.free_blocks(), 56);
        let (evicted, aux) = t.pressure_demote(&mut radix, &mut pool, 64, None);
        assert_eq!(evicted, 8, "whole chain evicted");
        assert!(aux.is_empty(), "no checkpoints in this radix");
        // both runs (blocks 0..4 and 4..8) demoted; their pins hold all 8
        // blocks until the stores complete - free unchanged, tree empty
        assert_eq!(pool.free_blocks(), 56, "demote pins defer the release");
        assert_eq!(t.stats().2, 2, "two stores in flight");
        assert_eq!(radix.cached_blocks(), 0);
        t.transport.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        assert_eq!(t.stats().3, 0, "stores open no tickets");
        assert_eq!(pool.free_blocks(), 64, "pins released at completion");
        assert_eq!(t.stats().0, 2, "two runs resident in T1");
        t.catalog.check_invariants();
        // and the content is probeable
        let hit = t.probe(&tokens, 0).expect("both runs restorable");
        assert_eq!((hit.start_block, hit.end_block), (0, 8));
        assert_eq!(hit.keys.len(), 2);
    }

    #[test]
    fn redemote_of_resident_content_is_free() {
        let mut t = tier(256 << 20);
        let mut pool = KvPool::with_blocks(64);
        let mut radix = armed_radix(&t);
        let _ = cached_chain(&mut radix, &mut pool, 1, 4);
        t.pressure_demote(&mut radix, &mut pool, 64, None);
        t.transport.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        assert_eq!(t.stats().0, 1);
        // the same content cached and evicted again: dedup - no second store
        let _ = cached_chain(&mut radix, &mut pool, 1, 4);
        t.pressure_demote(&mut radix, &mut pool, 64, None);
        assert_eq!(t.stats().2, 0, "AlreadyPresent skipped the store");
        assert_eq!(pool.free_blocks(), 64, "no pins - release was immediate");
        t.catalog.check_invariants();
    }

    #[test]
    fn a_chain_tail_past_the_last_run_boundary_evicts_plain() {
        let mut t = tier(256 << 20);
        let mut pool = KvPool::with_blocks(64);
        let mut radix = armed_radix(&t);
        let _ = cached_chain(&mut radix, &mut pool, 1, 6); // 1 run + 2 tail
        t.pressure_demote(&mut radix, &mut pool, 64, None);
        assert_eq!(t.stats().2, 1, "only the complete run stored");
        // 2 tail blocks released immediately, 4 pinned
        assert_eq!(pool.free_blocks(), 60);
        t.transport.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        assert_eq!(pool.free_blocks(), 64);
    }

    #[test]
    fn restore_publishes_into_the_radix_and_serves_adoption() {
        let mut t = tier(256 << 20);
        t.cost.set_force_restore(true);
        let mut pool = KvPool::with_blocks(64);
        let mut radix = armed_radix(&t);
        let tokens = cached_chain(&mut radix, &mut pool, 1, 8);
        t.pressure_demote(&mut radix, &mut pool, 64, None);
        t.transport.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        assert_eq!(
            radix.cached_blocks(),
            0,
            "GPU copy gone - the miss this fixes"
        );
        let hit = t.probe(&tokens, 0).expect("hit");
        assert!(t.elect(&hit, 0).is_restore());
        let ticket = t
            .begin_restore(&hit, &tokens, &mut pool, None)
            .expect("ticket");
        assert_eq!(
            pool.free_blocks(),
            56,
            "8 destination blocks reserved up front"
        );
        t.transport.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        assert_eq!(
            t.take_wake(ticket),
            Some(RestoreWake {
                ticket,
                ok: true,
                end_block: 8
            })
        );
        // published: the radix serves the prefix like any other hit, holding
        // its own references; the tier's are gone
        assert_eq!(radix.chain_depth(&tokens), 8);
        assert_eq!(radix.match_prefix(&tokens).len(), 8);
        assert_eq!(pool.free_blocks(), 56, "radix retains exactly the 8 blocks");
        assert_eq!(t.stats().3, 0, "no open tickets");
        t.catalog.check_invariants();
    }

    #[test]
    fn concurrent_restores_share_one_flight() {
        let mut t = tier(256 << 20);
        t.cost.set_force_restore(true);
        let mut pool = KvPool::with_blocks(64);
        let mut radix = armed_radix(&t);
        let tokens = cached_chain(&mut radix, &mut pool, 1, 4);
        t.pressure_demote(&mut radix, &mut pool, 64, None);
        t.transport.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        let hit = t.probe(&tokens, 0).expect("hit");
        let t1 = t
            .begin_restore(&hit, &tokens, &mut pool, None)
            .expect("first");
        let free_after_first = pool.free_blocks();
        let hit2 = t.probe(&tokens, 0).expect("hit again");
        let t2 = t
            .begin_restore(&hit2, &tokens, &mut pool, None)
            .expect("join");
        assert_eq!(
            pool.free_blocks(),
            free_after_first,
            "the join allocates NOTHING"
        );
        assert_eq!(t.catalog.counters.single_flight_joins, 1);
        t.transport.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        assert_eq!(
            t.take_wake(t1),
            Some(RestoreWake {
                ticket: t1,
                ok: true,
                end_block: 4
            })
        );
        assert_eq!(
            t.take_wake(t2),
            Some(RestoreWake {
                ticket: t2,
                ok: true,
                end_block: 4
            })
        );
        assert_eq!(radix.chain_depth(&tokens), 4);
        t.catalog.check_invariants();
    }

    #[test]
    fn partial_failure_publishes_the_landed_prefix_only() {
        let mut t = tier(256 << 20);
        t.cost.set_force_restore(true);
        let mut pool = KvPool::with_blocks(64);
        let mut radix = armed_radix(&t);
        let tokens = cached_chain(&mut radix, &mut pool, 1, 8);
        t.pressure_demote(&mut radix, &mut pool, 64, None);
        t.transport.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        let hit = t.probe(&tokens, 0).expect("hit");
        let ticket = t
            .begin_restore(&hit, &tokens, &mut pool, None)
            .expect("ticket");
        // run 1 lands, run 2's IO fails
        let ops = t.transport.pending_ops();
        assert_eq!(ops.len(), 2);
        t.transport.deliver(ops[0]);
        t.transport.deliver_failed(ops[1]);
        t.pump_completions(&mut radix, &mut pool);
        assert_eq!(
            t.take_wake(ticket),
            Some(RestoreWake {
                ticket,
                ok: true,
                end_block: 4
            })
        );
        assert_eq!(radix.chain_depth(&tokens), 4, "landed prefix published");
        assert_eq!(
            pool.free_blocks(),
            60,
            "failed run's blocks released, 4 held by radix"
        );
        // the failed run's T1 copy SURVIVES (IO failure, not integrity)
        assert!(t.probe(&tokens, 4).is_some(), "run 2 still restorable");
        t.catalog.check_invariants();
    }

    #[test]
    fn integrity_failure_retires_the_run_and_recomputes() {
        let mut t = tier(256 << 20);
        t.cost.set_force_restore(true);
        let mut pool = KvPool::with_blocks(64);
        let mut radix = armed_radix(&t);
        let tokens = cached_chain(&mut radix, &mut pool, 1, 4);
        t.pressure_demote(&mut radix, &mut pool, 64, None);
        t.transport.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        assert_eq!(t.stats().0, 1);
        // silent media corruption at rest
        let loc = t.transport.locs()[0];
        t.transport.corrupt_at_rest(loc);
        let hit = t.probe(&tokens, 0).expect("hit");
        let ticket = t
            .begin_restore(&hit, &tokens, &mut pool, None)
            .expect("ticket");
        t.transport.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        assert_eq!(
            t.take_wake(ticket),
            Some(RestoreWake {
                ticket,
                ok: false,
                end_block: 0
            })
        );
        assert_eq!(pool.free_blocks(), 64, "destination blocks released");
        assert_eq!(t.stats().0, 0, "poisoned run retired");
        assert!(t.probe(&tokens, 0).is_none(), "never offered again");
        assert_eq!(t.catalog.counters.integrity_failures, 1);
        t.catalog.check_invariants();
    }

    #[test]
    fn t1_capacity_pressure_evicts_lru_runs() {
        // room for exactly one run (16 MiB)
        let mut t = tier(16 << 20);
        let mut pool = KvPool::with_blocks(64);
        let mut radix = armed_radix(&t);
        let a = cached_chain(&mut radix, &mut pool, 1, 4);
        t.pressure_demote(&mut radix, &mut pool, 64, None);
        t.transport.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        assert!(t.probe(&a, 0).is_some());
        // a second chain demotes: T1 must evict A to admit B
        let b = cached_chain(&mut radix, &mut pool, 2, 4);
        t.pressure_demote(&mut radix, &mut pool, 64, None);
        t.transport.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        assert!(t.probe(&a, 0).is_none(), "A evicted from T1");
        assert!(t.probe(&b, 0).is_some(), "B resident");
        assert_eq!(t.stats().0, 1);
        t.catalog.check_invariants();
    }

    #[test]
    fn restore_truncates_to_the_pool_space_that_fits() {
        let mut t = tier(256 << 20);
        t.cost.set_force_restore(true);
        let mut pool = KvPool::with_blocks(16); // room for chain + ~1 run dst
        let mut radix = armed_radix(&t);
        let tokens = cached_chain(&mut radix, &mut pool, 1, 8);
        t.pressure_demote(&mut radix, &mut pool, 16, None);
        t.transport.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        assert_eq!(pool.free_blocks(), 16, "all evicted + pins released");
        // occupy the pool so only one run's destination fits
        let held: Vec<_> = (0..10).map(|_| pool.alloc().unwrap()).collect();
        let hit = t.probe(&tokens, 0).expect("both runs restorable");
        assert_eq!(hit.keys.len(), 2);
        let ticket = t
            .begin_restore(&hit, &tokens, &mut pool, None)
            .expect("truncated ticket");
        t.transport.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        assert_eq!(
            t.take_wake(ticket),
            Some(RestoreWake {
                ticket,
                ok: true,
                end_block: 4
            }),
            "one run restored, the rest recomputes"
        );
        assert_eq!(radix.chain_depth(&tokens), 4);
        for b in held {
            pool.release(b);
        }
        t.catalog.check_invariants();
    }

    /// Hybrid aux lifecycle: a checkpointed chain evicts under pressure, the
    /// claimed blob demotes as shards, the state index recycles at store
    /// completion, the boundary probes only when every shard is Ready, and
    /// the aux restore ticket resolves without touching the radix.
    #[test]
    fn aux_demote_probe_restore_roundtrip() {
        let mut t = tier(256 << 20);
        let mut pool = KvPool::with_blocks(64);
        let mut radix = armed_radix(&t);
        radix.set_state_capacity(1);
        let tokens = cached_chain(&mut radix, &mut pool, 1, 4);
        let s0 = radix
            .attach_state(&tokens, 4 * BLOCK_TOKENS)
            .expect("checkpoint");
        let (evicted, aux) = t.pressure_demote(&mut radix, &mut pool, 64, None);
        assert_eq!(evicted, 4);
        assert_eq!(aux.len(), 1, "checkpoint claimed off the evicted path");
        assert_eq!((aux[0].end_block, aux[0].state_idx), (4, s0));
        // blob: 20 MiB -> 2 shards (16 + 4)
        t.demote_aux(&mut radix, aux[0].clone(), 4096, 20 << 20, None);
        // the state index is not recycled until the stores complete: a new
        // chain cannot checkpoint yet (capacity 1, index in flight)
        let b = cached_chain(&mut radix, &mut pool, 2, 4);
        assert!(
            radix.attach_state(&b, 4 * BLOCK_TOKENS).is_none(),
            "index still claimed"
        );
        // one shard landing is not enough for a probe (every component)
        let ops = t.transport.pending_ops();
        let aux_ops: Vec<_> = ops.iter().copied().skip(ops.len() - 2).collect();
        t.transport.deliver(aux_ops[0]);
        t.pump_completions(&mut radix, &mut pool);
        assert!(
            t.probe_aux(&tokens, 4).is_none(),
            "partial blob must not probe"
        );
        t.transport.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        // recycled: the new chain can checkpoint now
        assert!(
            radix.attach_state(&b, 4 * BLOCK_TOKENS).is_some(),
            "index recycled"
        );
        let hit = t.probe_aux(&tokens, 4).expect("blob resident");
        assert_eq!((hit.end_block, hit.shards, hit.bytes), (4, 2, 20 << 20));
        // round two: restore into a fresh checkpoint slot
        let ticket = t.begin_restore_aux(&hit, 8192, None).expect("aux ticket");
        t.transport.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        assert_eq!(
            t.take_wake(ticket),
            Some(RestoreWake {
                ticket,
                ok: true,
                end_block: 4
            })
        );
        t.catalog.check_invariants();
    }

    #[test]
    fn aux_shard_failure_recycles_and_never_probes() {
        let mut t = tier(256 << 20);
        let mut pool = KvPool::with_blocks(64);
        let mut radix = armed_radix(&t);
        radix.set_state_capacity(1);
        let tokens = cached_chain(&mut radix, &mut pool, 1, 4);
        radix
            .attach_state(&tokens, 4 * BLOCK_TOKENS)
            .expect("checkpoint");
        let (_e, aux) = t.pressure_demote(&mut radix, &mut pool, 64, None);
        t.demote_aux(&mut radix, aux[0].clone(), 4096, 20 << 20, None);
        let ops = t.transport.pending_ops();
        let aux_ops: Vec<_> = ops.iter().copied().skip(ops.len() - 2).collect();
        t.transport.deliver(aux_ops[0]);
        t.transport.deliver_failed(aux_ops[1]);
        // drain the block stores too
        t.transport.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        assert!(t.probe_aux(&tokens, 4).is_none(), "half a blob is unusable");
        // the index recycled even through the failure
        let b = cached_chain(&mut radix, &mut pool, 2, 4);
        assert!(
            radix.attach_state(&b, 4 * BLOCK_TOKENS).is_some(),
            "recycled on failure"
        );
        t.catalog.check_invariants();
    }

    #[test]
    fn repeated_transport_failures_trip_the_breaker_and_serving_degrades_clean() {
        let mut t = tier(256 << 20);
        let mut pool = KvPool::with_blocks(64);
        let mut radix = armed_radix(&t);
        for i in 0..PoolTier::<FakeTransport>::BREAKER_TRIP {
            let _ = cached_chain(&mut radix, &mut pool, 1, 4); // same content
            t.pressure_demote(&mut radix, &mut pool, 64, None);
            let ops = t.transport.pending_ops();
            assert_eq!(ops.len(), 1, "iteration {i}: one store in flight");
            t.transport.deliver_failed(ops[0]);
            t.pump_completions(&mut radix, &mut pool);
        }
        assert!(t.is_tripped(), "8 consecutive failures must trip");
        // tripped: probes answer None, demotes are skipped, eviction still
        // works plain - serving degrades to recompute, never wedges
        let tokens = cached_chain(&mut radix, &mut pool, 2, 4);
        assert!(t.probe(&tokens, 0).is_none());
        let (evicted, _aux) = t.pressure_demote(&mut radix, &mut pool, 64, None);
        assert_eq!(evicted, 4, "eviction proceeds without the tier");
        assert_eq!(t.stats().2, 0, "no store submitted while tripped");
        assert_eq!(pool.free_blocks(), 64, "no pins - release was immediate");
        t.catalog.check_invariants();
    }

    /// A FULL tier is not a broken one. Stores that find no room in the
    /// arena complete NoRoom; however many arrive in a row, the breaker must
    /// stay closed and the next store that lands must still be kept - the
    /// tier used to count these as transport failures and go offline for
    /// good the moment it filled.
    #[test]
    fn stores_without_room_never_trip_the_breaker() {
        let mut t = tier(256 << 20);
        let mut pool = KvPool::with_blocks(64);
        let mut radix = armed_radix(&t);
        for i in 0..3 * PoolTier::<FakeTransport>::BREAKER_TRIP {
            let _ = cached_chain(&mut radix, &mut pool, 1, 4); // same content
            t.pressure_demote(&mut radix, &mut pool, 64, None);
            let ops = t.transport.pending_ops();
            assert_eq!(ops.len(), 1, "iteration {i}: one store in flight");
            t.transport.deliver_no_room(ops[0]);
            t.pump_completions(&mut radix, &mut pool);
        }
        assert!(!t.is_tripped(), "no-room stores are capacity, not faults");
        assert_eq!(
            t.catalog.counters.no_room,
            3 * PoolTier::<FakeTransport>::BREAKER_TRIP as u64
        );
        let tokens = cached_chain(&mut radix, &mut pool, 1, 4);
        t.pressure_demote(&mut radix, &mut pool, 64, None);
        t.transport.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        assert!(
            t.probe(&tokens, 0).is_some(),
            "the tier still keeps what lands"
        );
        assert_eq!(pool.free_blocks(), 64, "every pin released");
        t.catalog.check_invariants();
    }

    /// The ledger must explain a miss, not just count it. Content we
    /// held and evicted reads as a GHOST - the alarm that separates "this
    /// workload is cold" from "this tier is too small for this workload" -
    /// while a prefix we never saw stays an ordinary cold miss.
    #[test]
    fn a_miss_on_evicted_content_is_reported_as_a_ghost() {
        use super::super::accounting::MissReason;
        let mut t = tier(256 << 20);
        let mut pool = KvPool::with_blocks(64);
        let mut radix = armed_radix(&t);
        let a = cached_chain(&mut radix, &mut pool, 1, 8);
        t.pressure_demote(&mut radix, &mut pool, 64, None);
        t.transport.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        assert!(
            t.probe(&a, 0).is_some(),
            "resident chain must probe as a hit"
        );

        // force it out of T1 and back off the tier entirely
        let need = t.catalog.ledger(Tier::Ram).capacity;
        t.evict_t1(need, &[]);
        let before = t.dec;
        assert!(t.probe(&a, 0).is_none(), "evicted chain cannot restore");
        let after = t.dec;
        assert_eq!(
            after.miss_ghost - before.miss_ghost,
            1,
            "a miss on what we evicted must be a ghost, not a cold miss"
        );
        assert_eq!(after.miss_cold, before.miss_cold);

        // a prefix the tier never held stays cold - the alarm must not fire
        // on traffic that was never cacheable in the first place
        let unseen: Vec<u32> = (0..8 * BLOCK_TOKENS as u32).map(|i| 90_000 + i).collect();
        let c0 = t.dec.miss_cold;
        assert!(t.probe(&unseen, 0).is_none());
        assert_eq!(t.dec.miss_cold, c0 + 1, "unseen prefix must read as cold");
        assert_eq!(t.dec.miss_ghost, after.miss_ghost);

        // and the report carries it out
        let r = t.report();
        assert_eq!(r.decisions.miss_ghost, 1);
        assert!(r.ghost_keys >= 1, "the evicted key is remembered");
        assert!(!r.decisions.ghost_alarm(), "one ghost is not an alarm");
        let _ = MissReason::Ghost;
    }

    #[test]
    fn make_room_frees_retention_after_a_resolved_restore() {
        use super::super::restore_flow::{FlowStatus, RestoreFlow};
        // the nemotron cliff, miniaturized: retention fills the pool, a
        // restore resolves into the last free blocks, then an admission
        // needs a full table's worth - make-room must free the old chains
        let mut t = tier(256 << 20);
        let mut pool = KvPool::with_blocks(64);
        let mut radix = armed_radix(&t);
        // chain A cached then demoted out (it will be the restore)
        let a = cached_chain(&mut radix, &mut pool, 1, 8);
        t.pressure_demote(&mut radix, &mut pool, 64, None);
        t.transport.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        assert_eq!(pool.free_blocks(), 64);
        // retention: chains B..G fill 48 blocks; mirror them (as live would)
        let mut others = Vec::new();
        for seed in 2..8 {
            others.push(cached_chain(&mut radix, &mut pool, seed, 8));
        }
        t.mirror_slack(&radix, &mut pool, None, 64, None);
        t.transport.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        assert_eq!(pool.free_blocks(), 16);
        // restore A: destination takes 8 of the last 16
        let hit = t.probe(&a, 0).expect("hit");
        let flow = RestoreFlow::begin(&mut t, &mut pool, &a, &hit, None, 1000.0, None).expect("f");
        t.park_flow(0, flow);
        t.transport.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        t.pump_flows(&mut radix, &mut || None);
        assert_eq!(t.flow_status(0, &a), FlowStatus::Done { ok: true });
        assert_eq!(pool.free_blocks(), 8);
        // the admission now needs 24 blocks - make-room must evict the
        // mirrored retention (pure release) and deliver
        let ok = t.make_room_blocking(&mut radix, &mut pool, 24, None, &mut || None);
        // drain everything the press may have put in flight
        t.transport.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        assert!(
            ok,
            "make-room must free retention (free={})",
            pool.free_blocks()
        );
        assert!(pool.free_blocks() >= 24, "free={}", pool.free_blocks());
        t.catalog.check_invariants();
        let _ = others;
    }

    #[test]
    fn in_flight_mirror_pins_never_mask_evictable_retention() {
        // The nemotron wedge: mirror stores in flight pin TREE-HELD blocks
        // (their release frees nothing), and counting them toward pressure
        // targets made pressure_demote skip eviction - free=0 with a full
        // radix. The pending count must only see EVICTION pins.
        let mut t = tier(256 << 20);
        let mut pool = KvPool::with_blocks(64);
        let mut radix = armed_radix(&t);
        for seed in 1..9 {
            let _ = cached_chain(&mut radix, &mut pool, seed, 8);
        }
        assert_eq!(pool.free_blocks(), 0, "retention fills the pool");
        // mirror submits (2 runs, 16 pinned tree-held blocks) - Deliberately
        // not delivered: the wedge needs them in flight
        assert_eq!(t.mirror_slack(&radix, &mut pool, None, 2, None), 2);
        assert!(t.stats().2 > 0, "mirror stores in flight");
        // one press must EVICT despite the pins (the old pending count
        // saw the mirror's 8 pinned blocks, believed frees were coming,
        // and skipped eviction entirely)
        t.press(&mut radix, &mut pool, 8, None, &mut || None);
        t.transport.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        assert!(pool.free_blocks() >= 8, "free={}", pool.free_blocks());
        t.catalog.check_invariants();
    }

    #[test]
    fn mirror_slack_prestores_without_evicting_and_makes_eviction_free() {
        let mut t = tier(256 << 20);
        let mut pool = KvPool::with_blocks(64);
        let mut radix = armed_radix(&t);
        let tokens = cached_chain(&mut radix, &mut pool, 1, 8);
        let free_before = pool.free_blocks();
        // slack pass: stores submit, radix keeps serving, nothing evicts
        let n = t.mirror_slack(&radix, &mut pool, None, 8, None);
        assert_eq!(n, 2, "both complete runs mirrored");
        assert_eq!(radix.match_prefix(&tokens).len(), 8, "radix still serves");
        t.transport.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        assert_eq!(
            pool.free_blocks(),
            free_before,
            "mirror pins released, blocks still cached"
        );
        assert_eq!(t.stats().0, 2, "runs T1-resident");
        // a second pass is pure dedup - no new submissions
        assert_eq!(t.mirror_slack(&radix, &mut pool, None, 8, None), 0);
        // eviction is now free: no new stores, frees are immediate
        let (evicted, _aux) = t.pressure_demote(&mut radix, &mut pool, 64, None);
        assert_eq!(evicted, 8);
        assert_eq!(
            t.stats().2,
            0,
            "no store in flight - content was pre-mirrored"
        );
        assert_eq!(
            pool.free_blocks(),
            64,
            "release immediate, no deferred pins"
        );
        // and the content restores like any demoted chain
        assert!(t.probe(&tokens, 0).is_some());
        t.catalog.check_invariants();
    }

    #[test]
    fn mirror_slack_ships_live_checkpoint_blobs() {
        // the gemma4 lesson: the checkpoint pool recycles slots long before
        // blocks evict, so blobs must write through WHILE live - the slot
        // stays attached and eviction-time demote_aux dedups to a recycle
        let mut t = tier(256 << 20);
        let mut pool = KvPool::with_blocks(64);
        let mut radix = armed_radix(&t);
        radix.set_state_capacity(2);
        let tokens = cached_chain(&mut radix, &mut pool, 1, 8);
        radix
            .attach_state(&tokens, 8 * BLOCK_TOKENS)
            .expect("checkpoint");
        // slack pass with state geometry: blocks AND the blob mirror
        let n = t.mirror_slack(&radix, &mut pool, None, 8, Some((4096, 20 << 20)));
        assert_eq!(n, 2, "runs mirrored");
        t.transport.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        // the blob is T1-resident while the checkpoint still SERVES warm
        assert!(
            t.probe_aux(&tokens, 8).is_some(),
            "blob probeable without eviction"
        );
        assert!(
            radix.match_full(&tokens).ckpt.is_some(),
            "checkpoint still attached"
        );
        // eviction: block dedup + blob dedup -> recycle; everything frees
        let (_e, aux) = t.pressure_demote(&mut radix, &mut pool, 64, None);
        for a in aux {
            t.demote_aux(&mut radix, a, 4096, 20 << 20, None);
        }
        t.transport.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        assert_eq!(pool.free_blocks(), 64);
        // the full hybrid is restorable from T1
        assert!(t.probe(&tokens, 0).is_some());
        assert!(t.probe_aux(&tokens, 8).is_some());
        // and the state slot recycled through the dedup path
        let b = cached_chain(&mut radix, &mut pool, 2, 4);
        assert!(
            radix.attach_state(&b, 4 * BLOCK_TOKENS).is_some(),
            "slot recycled"
        );
        t.catalog.check_invariants();
    }

    #[test]
    fn mirror_slack_defers_to_restores() {
        use super::super::restore_flow::RestoreFlow;
        let mut t = tier(256 << 20);
        let mut pool = KvPool::with_blocks(64);
        let mut radix = armed_radix(&t);
        let a = cached_chain(&mut radix, &mut pool, 1, 4);
        t.pressure_demote(&mut radix, &mut pool, 64, None);
        t.transport.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        // an open restore ticket suspends mirroring (never beside a restore)
        let hit = t.probe(&a, 0).expect("hit");
        let flow = RestoreFlow::begin(&mut t, &mut pool, &a, &hit, None, 1000.0, None).expect("f");
        t.park_flow(0, flow);
        let _b = cached_chain(&mut radix, &mut pool, 2, 4);
        assert_eq!(
            t.mirror_slack(&radix, &mut pool, None, 8, None),
            0,
            "restore in flight"
        );
        t.transport.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        t.pump_flows(&mut radix, &mut || None);
        assert!(
            t.mirror_slack(&radix, &mut pool, None, 8, None) > 0,
            "slack again after resolve"
        );
        t.catalog.check_invariants();
    }

    #[test]
    fn flow_parks_pumps_and_publishes() {
        use super::super::restore_flow::{FlowStatus, RestoreFlow};
        let mut t = tier(256 << 20);
        let mut pool = KvPool::with_blocks(64);
        let mut radix = armed_radix(&t);
        let tokens = cached_chain(&mut radix, &mut pool, 1, 8);
        t.pressure_demote(&mut radix, &mut pool, 64, None);
        t.transport.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        let hit = t.probe(&tokens, 0).expect("hit");
        let flow =
            RestoreFlow::begin(&mut t, &mut pool, &tokens, &hit, None, 1000.0, None).expect("flow");
        t.park_flow(3, flow);
        assert_eq!(t.flow_status(3, &tokens), FlowStatus::Loading, "parked");
        t.transport.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        t.pump_flows(&mut radix, &mut || None);
        assert_eq!(t.flow_status(3, &tokens), FlowStatus::Done { ok: true });
        assert_eq!(
            t.flow_status(3, &tokens),
            FlowStatus::None,
            "Done is consumed"
        );
        assert_eq!(radix.chain_depth(&tokens), 8, "published for adoption");
        assert_eq!(t.flow_count(), 0);
        t.catalog.check_invariants();
    }

    #[test]
    fn kv_only_flow_partial_publication_is_a_win() {
        use super::super::restore_flow::{FlowStatus, RestoreFlow};
        let mut t = tier(256 << 20);
        let mut pool = KvPool::with_blocks(64);
        let mut radix = armed_radix(&t);
        let tokens = cached_chain(&mut radix, &mut pool, 1, 8);
        t.pressure_demote(&mut radix, &mut pool, 64, None);
        t.transport.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        // crowd the pool so only one run's destination fits (truncate-to-fit)
        let hog: Vec<_> = (0..60).map(|_| pool.alloc().unwrap()).collect();
        let hit = t.probe(&tokens, 0).expect("hit");
        let flow =
            RestoreFlow::begin(&mut t, &mut pool, &tokens, &hit, None, 1000.0, None).expect("f");
        t.park_flow(0, flow);
        t.transport.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        t.pump_flows(&mut radix, &mut || None);
        // 4 of 8 blocks published - an adoptable prefix, reported as a win
        assert_eq!(t.flow_status(0, &tokens), FlowStatus::Done { ok: true });
        assert_eq!(radix.chain_depth(&tokens), 4, "partial prefix adoptable");
        for b in hog {
            pool.release(b);
        }
        t.catalog.check_invariants();
    }

    #[test]
    fn flow_two_round_hybrid_attaches_the_checkpoint() {
        use super::super::restore_flow::{AuxPlan, FlowStatus, RestoreFlow};
        let mut t = tier(256 << 20);
        let mut pool = KvPool::with_blocks(64);
        let mut radix = armed_radix(&t);
        radix.set_state_capacity(2);
        let tokens = cached_chain(&mut radix, &mut pool, 1, 4);
        radix
            .attach_state(&tokens, 4 * BLOCK_TOKENS)
            .expect("checkpoint");
        let (_e, aux) = t.pressure_demote(&mut radix, &mut pool, 64, None);
        t.demote_aux(&mut radix, aux[0].clone(), 4096, 20 << 20, None);
        t.transport.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        let hit = t.probe(&tokens, 0).expect("hit");
        let aux_hit = t.probe_aux(&tokens, hit.end_block).expect("aux hit");
        assert_eq!(aux_hit.end_block, 4);
        let plan = AuxPlan {
            hit: aux_hit,
            state_base: 4096,
            state_stride: 20 << 20,
        };
        let flow = RestoreFlow::begin(&mut t, &mut pool, &tokens, &hit, Some(plan), 1000.0, None)
            .expect("flow");
        t.park_flow(0, flow);
        assert_eq!(t.flow_status(0, &tokens), FlowStatus::Loading);
        // round one lands; the flow pump starts round two on its own
        t.transport.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        t.pump_flows(&mut radix, &mut || None);
        assert_eq!(
            t.flow_status(0, &tokens),
            FlowStatus::Loading,
            "aux round in flight"
        );
        // round two lands; the checkpoint attaches only once verified
        t.transport.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        t.pump_flows(&mut radix, &mut || None);
        assert_eq!(t.flow_status(0, &tokens), FlowStatus::Done { ok: true });
        let m = radix.match_full(&tokens);
        assert_eq!(m.blocks.len(), 4);
        assert!(m.ckpt.is_some(), "flow attached the checkpoint");
        t.catalog.check_invariants();
    }

    #[test]
    fn flow_slot_reuse_zombies_the_old_flow_and_still_publishes() {
        use super::super::restore_flow::{FlowStatus, RestoreFlow};
        let mut t = tier(256 << 20);
        let mut pool = KvPool::with_blocks(64);
        let mut radix = armed_radix(&t);
        let tokens_a = cached_chain(&mut radix, &mut pool, 1, 8);
        t.pressure_demote(&mut radix, &mut pool, 64, None);
        t.transport.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        let hit = t.probe(&tokens_a, 0).expect("hit");
        let flow = RestoreFlow::begin(&mut t, &mut pool, &tokens_a, &hit, None, 1000.0, None)
            .expect("flow");
        t.park_flow(0, flow);
        // the requester cancelled; an unrelated request lands in slot 0
        let tokens_b: Vec<u32> = (0..64).map(|i| 999_000 + i).collect();
        assert_eq!(
            t.flow_status(0, &tokens_b),
            FlowStatus::None,
            "unrelated prompt never parks"
        );
        assert_eq!(t.flow_count(), 1, "old flow zombied, still draining");
        t.transport.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        t.pump_flows(&mut radix, &mut || None);
        assert_eq!(t.flow_count(), 0, "zombie resolved and retired");
        assert_eq!(
            radix.chain_depth(&tokens_a),
            8,
            "late publication still lands"
        );
        t.catalog.check_invariants();
    }

    #[test]
    fn flow_deadline_abandons_and_a_late_completion_still_publishes() {
        use super::super::restore_flow::{FlowStatus, RestoreFlow};
        let mut t = tier(256 << 20);
        let mut pool = KvPool::with_blocks(64);
        let mut radix = armed_radix(&t);
        let tokens = cached_chain(&mut radix, &mut pool, 1, 8);
        t.pressure_demote(&mut radix, &mut pool, 64, None);
        t.transport.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        let hit = t.probe(&tokens, 0).expect("hit");
        // est 1000us -> park_deadline clamps to its 20ms floor
        let flow =
            RestoreFlow::begin(&mut t, &mut pool, &tokens, &hit, None, 1000.0, None).expect("flow");
        t.park_flow(0, flow);
        std::thread::sleep(std::time::Duration::from_millis(25));
        t.pump_flows(&mut radix, &mut || None);
        assert_eq!(
            t.flow_status(0, &tokens),
            FlowStatus::Done { ok: false },
            "past deadline: unparked as FAILED - the request recomputes, and              the consult must never elect a fresh restore of the same content"
        );
        assert_eq!(
            t.flow_status(0, &tokens),
            FlowStatus::None,
            "consumed; zombie drains"
        );
        assert_eq!(t.flow_count(), 1, "zombie still consuming the wake");
        // the IO finally lands: the zombie consumes its wake and the
        // publication serves the next request for the same content
        t.transport.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        t.pump_flows(&mut radix, &mut || None);
        assert_eq!(radix.chain_depth(&tokens), 8, "late publication landed");
        assert_eq!(t.flow_count(), 0, "zombie retired");
        t.catalog.check_invariants();
    }

    #[test]
    fn prefix_evicted_mid_flight_refuses_to_publish_into_a_hole() {
        let mut t = tier(256 << 20);
        t.cost.set_force_restore(true);
        let mut pool = KvPool::with_blocks(64);
        let mut radix = armed_radix(&t);
        let tokens = cached_chain(&mut radix, &mut pool, 1, 8);
        t.pressure_demote(&mut radix, &mut pool, 64, None);
        t.transport.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        // GPU holds blocks 0..4 again (restore run 1 first)
        let hit1 = t.probe(&tokens, 0).expect("hit");
        let hit1 = TierHit {
            end_block: 4,
            keys: hit1.keys[..1].to_vec(),
            bytes: 16 << 20,
            ..hit1
        };
        let tk1 = t
            .begin_restore(&hit1, &tokens, &mut pool, None)
            .expect("t1");
        t.transport.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        assert_eq!(
            t.take_wake(tk1),
            Some(RestoreWake {
                ticket: tk1,
                ok: true,
                end_block: 4
            })
        );
        // now restore run 2 (attaches at depth 4)... but evict the prefix
        // while the load is in flight
        let hit2 = t.probe(&tokens, 4).expect("run 2");
        let tk2 = t
            .begin_restore(&hit2, &tokens, &mut pool, None)
            .expect("t2");
        while radix.evict_lru(&mut pool).is_some() {}
        assert_eq!(radix.chain_depth(&tokens), 0, "prefix gone mid-flight");
        t.transport.deliver_all();
        t.pump_completions(&mut radix, &mut pool);
        assert_eq!(
            t.take_wake(tk2),
            Some(RestoreWake {
                ticket: tk2,
                ok: false,
                end_block: 4
            })
        );
        assert_eq!(
            radix.chain_depth(&tokens),
            0,
            "nothing attached across the hole"
        );
        assert_eq!(
            pool.free_blocks(),
            64,
            "restored blocks released, not leaked"
        );
        t.catalog.check_invariants();
    }
}
