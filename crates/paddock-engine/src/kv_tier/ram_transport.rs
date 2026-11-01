//! The RAM (T1) transport - the first real [`TierTransport`].
//!
//! Data plane, per the host-memory elections:
//!
//! - **Demote**: gather kernel packs scattered pool blocks into a page-first
//!   contiguous extent in device staging (on-die, ~order-of-magnitude above
//!   bus rate), then one `cuMemcpyDtoHAsync` moves the whole extent into a
//!   registered T1 slab at the 26.7 GB/s bus ceiling. No per-page transfers,
//!   ever (8 KiB fragments measure 5% of the bus).
//! - **Restore**: one H2D into device staging, scatter kernel back into the
//!   (freshly reserved) pool blocks.
//! - Both run on this transport's own forked lane, event-fenced against the
//!   compute stream via the seal event the engine passes in [`XferSpec`] -
//!   serving kernels never wait on tier traffic.
//! - When slab registration fell back to pageable (loud, see `host.rs`), the
//!   DMA bounces through the cached-pinned staging ring instead - slower,
//!   still correct, still one contiguous bus transfer per extent.
//!
//! Direction arbitration (a v1 stub of the full arbiter): there is no
//! PCIe duplex under WDDM - a demote running beside a restore halves the
//! restore. So stores start only when no load is queued or in flight;
//! restores never wait on demotes. The full contention-priced arbiter
//! arrives later.
//!
//! Everything runs on the engine thread (submit/poll/drop - same contract as
//! `host.rs`). `poll()` is non-blocking: it advances queues, checks events,
//! computes checksums for finished ops and returns their completions.

use std::collections::{HashMap, HashSet, VecDeque};

use cudarc::driver::{CudaEvent, CudaSlice, DevicePtr, DevicePtrMut};

use super::digest::{Checksum, LogicalKey};
use super::hash_pool::HashPool;
use super::host::{HostStore, PinMode, StagingRing};
use super::nvme_store::{NvmeStore, StoreError, StoreItem};
use super::t2_reads::{ReadDone, ReadJob, T2Readers};
use super::transport::{
    IoCompletion, IoJob, IoJobKind, IoOutcome, SubmitError, TierTransport, TransportCaps,
};
use super::{LoadDst, Loc, OpId};
use crate::gpu::{GpuError, GpuExecutor};

/// One device span source/destination: a paged-pool plane. `base` must be
/// the plane's device address, `stride` the byte distance between block
/// records in it, `bytes` one block's bytes in this plane. All multiples of
/// 16 (validated at [`RamTransport::expect_store`] / `expect_load`).
#[derive(Debug, Clone, Copy)]
pub struct PlaneDesc {
    pub base: u64,
    pub stride: u64,
    pub bytes: u64,
}

/// The engine-side description of where a payload's bytes live on device -
/// registered per key right before `begin_store`/`begin_load` submission
/// (the catalog's [`IoJob`] deliberately carries no device geometry; the
/// data plane's schema knowledge stays out of the transactional core).
pub struct XferSpec {
    /// Plane order defines the record layout (dst offsets are the running
    /// sum of plane bytes). Must be identical between a key's store and any
    /// later restore - the payload schema's canonical plane order.
    pub planes: Vec<PlaneDesc>,
    /// Pool block ids, one per 16-token block in chain order.
    pub block_ids: Vec<u32>,
    /// Producer fence: the compute-stream event after which the source
    /// blocks are sealed (stores) / the destination blocks are reserved and
    /// quiescent (loads). None = already synchronized by the caller.
    pub after: Option<CudaEvent>,
}

impl XferSpec {
    fn record_stride(&self) -> u64 {
        self.planes.iter().map(|p| p.bytes).sum()
    }

    fn total(&self) -> u64 {
        self.record_stride() * self.block_ids.len() as u64
    }

    fn max_plane_bytes(&self) -> u64 {
        self.planes.iter().map(|p| p.bytes).max().unwrap_or(0)
    }

    fn aligned(&self) -> bool {
        self.planes
            .iter()
            .all(|p| p.base % 16 == 0 && p.stride % 16 == 0 && p.bytes % 16 == 0 && p.bytes > 0)
    }

    /// The device u64 descriptor image: {base, stride, bytes, dst_off} per
    /// plane, dst_off = running sum of plane bytes.
    fn desc_image(&self) -> Vec<u64> {
        let mut v = Vec::with_capacity(self.planes.len() * 4);
        let mut off = 0u64;
        for p in &self.planes {
            v.extend_from_slice(&[p.base, p.stride, p.bytes, off]);
            off += p.bytes;
        }
        v
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dir {
    Store,
    Load,
}

/// An op that has its GPU work enqueued and is waiting on its fence event.
struct Flight {
    op: OpId,
    dir: Dir,
    key: LogicalKey,
    /// T1 extent (stores / T1 loads) or the T2 loc (Nvme loads - opaque
    /// here; the length comes from `spec_total`).
    host_loc: Loc,
    /// Payload bytes per the spec.
    spec_total: usize,
    done: CudaEvent,
    ring_slot: Option<usize>,
    /// This load's source is the T2 store (host_loc is not a T1 extent).
    from_t2: bool,
    /// A disk read that landed in pinned memory (`host_loc` names it).
    landed: Option<Landed>,
    /// This flight's descriptor buffers - flights PIPELINE on the lane
    /// stream, so a shared buffer would be overwritten by flight N+1's
    /// upload before flight N's kernel reads it. Held until the event
    /// retires; the drop is a stream-ordered mempool free. Never read: it
    /// exists to be held.
    #[allow(dead_code)]
    descs: (CudaSlice<u64>, CudaSlice<u32>),
}

/// Where a disk read landed, and what it verified against.
#[derive(Debug, Clone, Copy)]
struct Landed {
    /// In the read arena (`t2_stage`), not a T1 extent.
    stage: bool,
    checksum: [u8; 32],
}

/// A disk read in flight on the reader pool.
struct Landing {
    q: Queued,
    loc: Loc,
    landed: Landed,
    len: u64,
}

/// Disk reads in flight at once, in bytes. Submission happens at pumps, so
/// the window is what one scheduler tick can put in front of the device: a
/// 256 MiB window read a 700 MB restore at ~1.9 GB/s (ticks of ~100 ms
/// beside prefill), a quarter of the drive. Reads landing in T1 are bounded
/// by T1's room; the arena below bounds the rest.
const T2_READ_WINDOW: u64 = 1 << 30;
/// The pinned arena disk reads land in when T1 has no room for a read-fill
/// copy (a full T1 is the steady state): 16 MiB shards across every reader
/// twice over, a fixed cost beside the ring when a disk tier is armed.
const T2_STAGE_BYTES: u64 = 256 << 20;

/// What `start_t2_read` did with a disk load.
enum T2Start {
    Reading,
    /// The read window is full - retry once reads in flight complete.
    Wait(Queued),
    /// Read it on the ring path instead.
    Ring(Queued),
}

/// A submitted op waiting for its direction's staging extent (or, for
/// stores, for load traffic to drain - the no-duplex rule).
struct Queued {
    job: IoJob,
    spec: XferSpec,
    /// Stores allocate their T1 extent at submit (the catalog admitted the
    /// bytes; the allocation is that admission made physical).
    host_loc: Loc,
}

/// Device staging: one extent per direction so a restore never waits for a
/// demote's extent (they still serialize on the lane's stream - the bus
/// would serialize them anyway).
pub const DEVICE_STAGING_EXTENT_BYTES: usize =
    paddock_models::kv_tier_geom::STAGING_EXTENT_BYTES as usize;

/// VRAM the tier claims for staging - the number families charge as a named
/// `Reserve` in their `kv_plan::Demand` when the tier is enabled (that is
/// what "staging accounted in kv_plan" means; the reserve wiring lands with
/// the serving integration).
pub const fn device_staging_bytes() -> u64 {
    // one definition, shared with the manager's fit estimate - the engine
    // reserves exactly what the estimate subtracts, or the two answers drift
    paddock_models::kv_tier_geom::device_staging_bytes()
}

/// Days since the UNIX epoch - the endurance budget's rollover clock.
fn utc_day() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() / 86_400)
        .unwrap_or(0)
}

/// Staging-ring geometry (cached-pinned host bounce): 2 extents matching the
/// device staging size - one per direction in the pageable fallback.
pub const RING_EXTENTS: usize = paddock_models::kv_tier_geom::STAGING_EXTENTS as usize;

/// A finished flight whose payload digest is still being computed (see
/// `hash_pool`): its completion waits for the digest.
enum Hashing {
    Store {
        key: LogicalKey,
        loc: Loc,
        bytes: u64,
    },
    Load {
        bytes: u64,
    },
}

pub struct RamTransport {
    /// Forked lane: `exec.stream` is the tier stream. Shares the context,
    /// mempool and kernel table with the serving executor.
    exec: GpuExecutor,
    /// Checksum workers. Declared before `store` on purpose: fields drop in
    /// order, and the workers read slab bytes until they are joined.
    hasher: HashPool,
    /// Disk readers (`t2_reads`) - before `store` and `t2_stage` for the
    /// same reason: they write into slab bytes until joined.
    t2_readers: Option<T2Readers>,
    /// Finished flights waiting on their digest, by op.
    hashing: HashMap<OpId, Hashing>,
    store: HostStore,
    /// The pinned arena disk reads land in when T1 has no room
    /// ([`T2_STAGE_BYTES`]).
    t2_stage: Option<HostStore>,
    /// Disk reads in flight, by op, and their bytes.
    t2_reads: HashMap<OpId, Landing>,
    t2_read_bytes: u64,
    /// T2 durable store: when attached, every successful T1 store
    /// writes through to disk (restart persistence), and loads whose job
    /// names `Tier::Nvme` read from it through the pinned ring. v1
    /// write-through is synchronous in `poll` - the full arbiter moves it to
    /// read-slack windows, per the measured contention law.
    t2: Option<NvmeStore>,
    ring: StagingRing,
    dev: [CudaSlice<u8>; 2],
    /// (descriptor buffers are per-flight - see `Flight::descs`)
    expect_store: HashMap<LogicalKey, XferSpec>,
    expect_load: HashMap<LogicalKey, XferSpec>,
    q_store: VecDeque<Queued>,
    /// Queued stores a restore waits on (`expedite_inner`): they launch at
    /// the next kick, loads or not.
    urgent: HashSet<OpId>,
    /// Stores whose T1 extent could not be carved while the slab supply is
    /// still delivering the budget: retried at every kick, never completed
    /// NoRoom until the supply is done (a dropped store is lost content).
    q_held: VecDeque<(IoJob, XferSpec)>,
    q_load: VecDeque<Queued>,
    /// Quota-eviction outbox: keys `make_room` retired, awaiting the pump's
    /// catalog invalidation.
    t2_evicted: Vec<[u8; 32]>,
    /// Read-fill outbox: (key, T1 loc, checksum, bytes) promoted off a T2
    /// load, awaiting the pump's catalog adoption (or extent return).
    t2_promoted: Vec<([u8; 32], Loc, [u8; 32], u64)>,
    /// 3.3 endurance budget: payload bytes written through to T2 in the
    /// current UTC day. The write-through STOPS at the budget (serving and
    /// T1 continue; only durability degrades, loudly once) so the mirror
    /// pass can never wear an SSD unboundedly. Elected, not tunable - a
    /// quarter of the store's volume a day, at most 1 TiB (`NvmeStore::
    /// day_budget`); `PADDOCK_KV_T2_DAY_GB` is a dev-build override,
    /// compiled out of hardened builds.
    t2_day: u64,
    t2_written_today: u64,
    t2_budget: u64,
    t2_budget_warned: bool,
    flights: Vec<Flight>,
    cancelled: HashSet<OpId>,
    ready: Vec<IoCompletion>,
}

impl RamTransport {
    /// Build the T1 transport: fork a lane off the serving executor, claim
    /// the device staging extents, arm the host store and ring.
    /// `ram_capacity` is the catalog's T1 capacity (the `[kv_offload]
    /// ram_gb` budget minus the ring - the caller runs that arithmetic and
    /// reports it, kv_plan-style).
    pub fn new(serving: &GpuExecutor, ram_capacity: u64) -> Result<Self, GpuError> {
        Self::new_inner(serving, ram_capacity, None)
    }

    /// [`Self::new`] with a durable T2 store attached (restart
    /// persistence): `dir` is the namespace's store directory
    /// (`NvmeStore::dir_for`), `quota` its byte budget. Recovery is logged;
    /// preloading the recovered entries into the catalog is the tier
    /// layer's job (`PoolTier::preload_from_t2`).
    pub fn with_t2(
        serving: &GpuExecutor,
        ram_capacity: u64,
        dir: &std::path::Path,
        quota: u64,
    ) -> Result<Self, GpuError> {
        Self::new_inner(serving, ram_capacity, Some((dir.to_path_buf(), quota)))
    }

    fn new_inner(
        serving: &GpuExecutor,
        ram_capacity: u64,
        t2cfg: Option<(std::path::PathBuf, u64)>,
    ) -> Result<Self, GpuError> {
        let t2 = match t2cfg {
            Some((dir, quota)) => match NvmeStore::open(&dir, quota) {
                Ok((st, report)) => {
                    tracing::info!(
                        dir = %dir.display(),
                        recovered = report.recovered_entries,
                        replayed = report.replayed_wal_records,
                        discarded = report.discarded_tail_records,
                        orphaned = report.orphaned_bytes,
                        reset = report.reset_after_corruption,
                        ms = report.recovery_ms,
                        day_budget_gib = st.day_budget() >> 30,
                        "KV T2 store recovered"
                    );
                    super::census::ev("boot", &[0; 32], report.recovered_entries, None);
                    Some(st)
                }
                Err(e) => {
                    tracing::warn!(err = %e, "KV T2 store declined - RAM tier only");
                    None
                }
            },
            None => None,
        };
        if !serving.has_kv_tier_xfer() {
            return Err(GpuError::Unsupported(
                "this kernel pack has no kv tier gather/scatter (slots 479/480) - \
                 rebuild the pack (packs/cuda/build.ps1 / build.sh) to enable KV \
                 offload; serving continues without the tier"
                    .into(),
            ));
        }
        let exec = serving.fork_stream()?;
        let dev = [
            exec.alloc_u8(DEVICE_STAGING_EXTENT_BYTES)?,
            exec.alloc_u8(DEVICE_STAGING_EXTENT_BYTES)?,
        ];
        let mut store = HostStore::new(ram_capacity, true);
        let ring = StagingRing::new(RING_EXTENTS, DEVICE_STAGING_EXTENT_BYTES);
        // Self-test: PROVE device->slab->readback on this machine before the
        // tier claims to exist. A tier that would fail its first real demote
        // must refuse here, loudly, not corrupt or wedge later.
        {
            let n = 256 * 1024usize;
            let pattern: Vec<u8> = (0..n).map(|i| (i * 31 + 7) as u8).collect();
            let mut probe_dev = exec.alloc_u8(n).map_err(|e| {
                GpuError::Unsupported(format!("kv tier self-test alloc failed: {e}"))
            })?;
            exec.upload_u8(&pattern, &mut probe_dev)?;
            let loc = store.alloc(n as u64).map_err(|e| {
                GpuError::Unsupported(format!("kv tier self-test slab alloc failed: {e}"))
            })?;
            let (hp, hl) = store.resolve(loc).expect("just allocated");
            debug_assert_eq!(hl, n as u64);
            // SAFETY: freshly allocated extent, exclusively ours.
            let dst = unsafe { std::slice::from_raw_parts_mut(hp, n) };
            Self::dtoh(&exec, &probe_dev, dst)?;
            exec.synchronize()?;
            let ok = dst == pattern.as_slice();
            let _ = store.free(loc);
            if !ok {
                return Err(GpuError::Unsupported(
                    "kv tier self-test FAILED: bytes read back from the T1 slab                      do not match what the device sent - refusing to arm the                      tier on this machine (serving continues untiered)"
                        .into(),
                ));
            }
            tracing::info!(mode = ?store.mode(), "KV tier self-test passed (device -> slab -> verify)");
        }
        // the first slab is proven: the rest of the budget builds off the
        // engine thread from here (see host.rs `SlabSupply`)
        store.start_supply(Some(exec.stream.context().clone()));
        let t2_budget = paddock_models::dev_var_os!("PADDOCK_KV_T2_DAY_GB")
            .and_then(|v| v.to_str().and_then(|t| t.parse::<u64>().ok()))
            .map(|gb| gb << 30)
            .unwrap_or_else(|| t2.as_ref().map_or(0, NvmeStore::day_budget));
        // disk restores: readers at depth, landing in pinned memory. The
        // arena's one slab is built now - building it at the first restore
        // would put its populate + registration on that restore's clock.
        let (t2_readers, t2_stage) = match t2.as_ref() {
            Some(st) => {
                let mut stage = HostStore::new(T2_STAGE_BYTES, true);
                if let Ok(l) = stage.alloc(1) {
                    let _ = stage.free(l);
                }
                (Some(T2Readers::new(st.reader_io())), Some(stage))
            }
            None => (None, None),
        };
        tracing::info!(
            capacity_gib = ram_capacity as f64 / (1u64 << 30) as f64,
            staging_mib = device_staging_bytes() >> 20,
            "KV RAM tier transport ready"
        );
        Ok(Self {
            exec,
            hasher: HashPool::new(),
            t2_readers,
            hashing: HashMap::new(),
            store,
            t2_stage,
            t2_reads: HashMap::new(),
            t2_read_bytes: 0,
            t2,
            ring,
            dev,
            expect_store: HashMap::new(),
            expect_load: HashMap::new(),
            q_store: VecDeque::new(),
            urgent: HashSet::new(),
            q_held: VecDeque::new(),
            q_load: VecDeque::new(),
            t2_evicted: Vec::new(),
            t2_promoted: Vec::new(),
            t2_day: utc_day(),
            t2_written_today: 0,
            t2_budget,
            t2_budget_warned: false,
            flights: Vec::new(),
            cancelled: HashSet::new(),
            ready: Vec::new(),
        })
    }

    /// Register the device geometry for a store of `key`, validated now so
    /// `submit` cannot half-start. Call right before `begin_store`.
    pub fn expect_store(&mut self, key: LogicalKey, spec: XferSpec) -> Result<(), SubmitError> {
        Self::validate(&spec)?;
        self.expect_store.insert(key, spec);
        Ok(())
    }

    /// Register the destination geometry for a load of `key` (the freshly
    /// reserved pool blocks the scatter writes). Call right before
    /// `begin_load`.
    pub fn expect_load(&mut self, key: LogicalKey, spec: XferSpec) -> Result<(), SubmitError> {
        Self::validate(&spec)?;
        self.expect_load.insert(key, spec);
        Ok(())
    }

    fn validate(spec: &XferSpec) -> Result<(), SubmitError> {
        if !spec.aligned() || !spec.record_stride().is_multiple_of(16) {
            return Err(SubmitError::Misaligned);
        }
        if spec.planes.is_empty() || spec.block_ids.is_empty() {
            return Err(SubmitError::SourceMissing);
        }
        if spec.total() > DEVICE_STAGING_EXTENT_BYTES as u64 {
            return Err(SubmitError::TooLarge);
        }
        Ok(())
    }

    /// Free a T1 extent the catalog evicted (`TierCatalog::evict` returns the
    /// ledger bytes; this returns the physical ones). The engine wiring calls
    /// both - the catalog does not know about physical locations.
    pub fn free_extent(&mut self, loc: Loc) {
        if let Err(e) = self.store.free(loc) {
            tracing::error!(err = %e, "KV tier: freeing an evicted extent failed (leak)");
        }
    }

    /// The lane the transport runs on - tests use it to synchronize.
    pub fn lane(&self) -> &GpuExecutor {
        &self.exec
    }

    pub fn host_mode(&self) -> PinMode {
        self.store.mode()
    }

    /// Whether the whole T1 budget is carved yet (the slab supply has
    /// delivered its last slab), and `(slabs, pinned slabs)` so far.
    pub fn t1_slabs(&mut self) -> (bool, usize, usize) {
        let done = self.store.supply_done();
        let (n, pinned) = self.store.slab_counts();
        (done, n, pinned)
    }

    /// The footprint the T1 arena charges for `bytes`.
    pub fn t1_rounded(&self, bytes: u64) -> u64 {
        self.store.rounded(bytes)
    }

    /// The T1 arena's occupancy (see [`HostStore::layout`]).
    pub fn t1_layout(&self) -> Vec<super::host::SlabLayout> {
        self.store.layout()
    }

    /// Physical T1 bytes currently allocated (rounded extents) - gates and
    /// the occupancy split read this beside the catalog's logical ledger.
    pub fn t1_allocated(&self) -> u64 {
        self.store.allocated()
    }

    /// Stores submitted and not yet launched - a gate witness.
    pub fn stores_queued(&self) -> usize {
        self.q_store.len()
    }

    /// Keys the T2 cache evicted for quota - the tier's pump drains these
    /// and drops the catalog's Nvme references (see `XferSink`).
    pub fn take_t2_evictions_inner(&mut self) -> Vec<[u8; 32]> {
        std::mem::take(&mut self.t2_evicted)
    }

    /// Read-fill promotions off completed T2 loads (see `XferSink`).
    pub fn take_t2_promotions_inner(&mut self) -> Vec<([u8; 32], Loc, [u8; 32], u64)> {
        std::mem::take(&mut self.t2_promoted)
    }

    /// Endurance telemetry: T2 payload bytes written this UTC day.
    pub fn t2_written_today_inner(&self) -> u64 {
        self.t2_written_today
    }

    /// ... and the day's budget (0 with no T2 store attached).
    pub fn t2_day_budget_inner(&self) -> u64 {
        if self.t2.is_some() { self.t2_budget } else { 0 }
    }

    /// The attached T2 store, when one is (restart preload + tombstones).
    pub fn t2(&self) -> Option<&NvmeStore> {
        self.t2.as_ref()
    }

    /// The serving device's tag (see `GpuExecutor::device_tag`).
    pub fn device_tag(&self) -> String {
        self.exec.device_tag()
    }

    pub fn t2_mut(&mut self) -> Option<&mut NvmeStore> {
        self.t2.as_mut()
    }

    // -- launch paths -------------------------------------------------------

    fn fail(&mut self, op: OpId) {
        self.ready.push(IoCompletion {
            op,
            outcome: IoOutcome::Failed,
        });
    }

    /// Enqueue the GPU work for a store: fence, gather, DMA to the T1
    /// extent, fence event. Any launch error completes the op Failed and
    /// releases what it held.
    fn launch_store(&mut self, q: Queued) {
        let op = q.job.op;
        match self.launch_store_inner(&q) {
            Ok((done, ring_slot, descs)) => {
                self.flights.push(Flight {
                    op,
                    dir: Dir::Store,
                    key: q.job.key,
                    host_loc: q.host_loc,
                    spec_total: q.spec.total() as usize,
                    done,
                    ring_slot,
                    from_t2: false,
                    landed: None,
                    descs,
                });
            }
            Err(e) => {
                tracing::warn!(err = %e, "KV tier store launch failed - completing Failed");
                self.free_extent(q.host_loc);
                self.fail(op);
            }
        }
    }

    fn launch_store_inner(
        &mut self,
        q: &Queued,
    ) -> Result<(CudaEvent, Option<usize>, (CudaSlice<u64>, CudaSlice<u32>)), GpuError> {
        let spec = &q.spec;
        let total = spec.total() as usize;
        let descs = self.make_descs(spec)?;
        if let Some(ev) = &spec.after {
            self.exec.wait_event(ev)?;
        }
        // split-borrow the store-direction extent out of self so exec (also
        // &self) can launch into it
        let (exec, dev) = (&self.exec, &mut self.dev[Dir::Store as usize]);
        exec.kv_gather_blocks(
            &descs.0,
            &descs.1,
            dev,
            spec.record_stride(),
            spec.max_plane_bytes(),
            spec.planes.len(),
            spec.block_ids.len(),
        )?;
        // DMA: direct into the registered slab, or bounce through the ring
        let (host_ptr, payload) = self
            .store
            .resolve(q.host_loc)
            .map_err(|e| GpuError::Unsupported(format!("host extent vanished: {e}")))?;
        debug_assert_eq!(payload as usize, total);
        let ring_slot = if self.store.is_pinned(q.host_loc) {
            // SAFETY: extent pointer valid until free_extent (post-completion);
            // slice length equals the DMA length; registered => DMA-safe.
            let dst = unsafe { std::slice::from_raw_parts_mut(host_ptr, total) };
            Self::dtoh(exec, dev, dst)?;
            None
        } else {
            let slot = self.ring.acquire().ok_or_else(|| {
                GpuError::Unsupported("staging ring exhausted (pageable fallback)".into())
            })?;
            let (rp, rl) = self.ring.ptr(slot);
            debug_assert!(total <= rl);
            // SAFETY: ring extent is address-stable and exclusively ours
            // while the slot is busy.
            let dst = unsafe { std::slice::from_raw_parts_mut(rp, total) };
            Self::dtoh(exec, dev, dst)?;
            Some(slot)
        };
        let done = exec.record_event()?;
        Ok((done, ring_slot, descs))
    }

    /// `landed`: a disk read already sitting in pinned memory at
    /// `q.host_loc` - it launches like a T1 load from there.
    fn launch_load(&mut self, q: Queued, landed: Option<Landed>) {
        let op = q.job.op;
        match self.launch_load_inner(&q, landed) {
            Ok((done, ring_slot, descs)) => {
                self.flights.push(Flight {
                    op,
                    dir: Dir::Load,
                    key: q.job.key,
                    host_loc: q.host_loc,
                    spec_total: q.spec.total() as usize,
                    done,
                    ring_slot,
                    from_t2: q.job.tier == super::Tier::Nvme,
                    landed,
                    descs,
                });
            }
            Err(e) => {
                tracing::warn!(err = %e, "KV tier load launch failed - completing Failed");
                if let Some(l) = landed {
                    self.free_landing(q.host_loc, l);
                }
                self.fail(op);
            }
        }
    }

    fn free_landing(&mut self, loc: Loc, l: Landed) {
        let freed = match (l.stage, self.t2_stage.as_mut()) {
            (true, Some(stage)) => stage.free(loc),
            _ => self.store.free(loc),
        };
        if let Err(e) = freed {
            tracing::error!(err = %e, "KV tier: freeing a disk read's landing failed (leak)");
        }
    }

    fn launch_load_inner(
        &mut self,
        q: &Queued,
        landed: Option<Landed>,
    ) -> Result<(CudaEvent, Option<usize>, (CudaSlice<u64>, CudaSlice<u32>)), GpuError> {
        let spec = &q.spec;
        let total = spec.total() as usize;
        let descs = self.make_descs(spec)?;
        if let Some(ev) = &spec.after {
            self.exec.wait_event(ev)?;
        }
        // T2 source without a landing (no reader pool, or nowhere aligned
        // to land it): read it synchronously (VERIFIED against its commit
        // record) into a pinned ring slot, then the same H2D + scatter as
        // any load.
        if q.job.tier == super::Tier::Nvme && landed.is_none() {
            // Claim the staging slot first and read the extent straight into
            // it: the slot is pinned host memory, so a T2 restore costs
            // disk -> pinned -> GPU with no heap bounce in between.
            // `read_into` verifies the commit checksum in place and falls
            // back to the allocating path itself if the slot cannot satisfy
            // the device's alignment contract.
            let slot = self
                .ring
                .acquire()
                .ok_or_else(|| GpuError::Unsupported("staging ring exhausted (T2 load)".into()))?;
            let (rp, rl) = self.ring.ptr(slot);
            let t2 = match self.t2.as_mut() {
                Some(t) => t,
                None => {
                    self.ring.release(slot);
                    return Err(GpuError::Unsupported("Nvme load without a T2 store".into()));
                }
            };
            // SAFETY: ring extent exclusively ours while the slot is busy.
            let dst = unsafe { std::slice::from_raw_parts_mut(rp, rl) };
            let read = t2.read_into(&q.job.key.0, dst);
            let len = match read {
                Ok((_gen, len)) => len as usize,
                Err(e) => {
                    self.ring.release(slot);
                    return Err(GpuError::Unsupported(format!("T2 read failed: {e}")));
                }
            };
            if len != total {
                self.ring.release(slot);
                return Err(GpuError::Unsupported(format!(
                    "T2 payload {len} bytes vs spec {total} - schema drift"
                )));
            }
            // SAFETY: same extent, now filled and length-checked.
            let src = unsafe { std::slice::from_raw_parts(rp, total) };
            let (exec, dev) = (&self.exec, &mut self.dev[Dir::Load as usize]);
            Self::htod(exec, src, dev)?;
            exec.kv_scatter_blocks(
                &descs.0,
                &descs.1,
                dev,
                spec.record_stride(),
                spec.max_plane_bytes(),
                spec.planes.len(),
                spec.block_ids.len(),
            )?;
            let done = exec.record_event()?;
            return Ok((done, Some(slot), descs));
        }
        let source = match (landed, self.t2_stage.as_ref()) {
            (Some(Landed { stage: true, .. }), Some(stage)) => stage,
            _ => &self.store,
        };
        let (host_ptr, payload) = source
            .resolve(q.host_loc)
            .map_err(|e| GpuError::Unsupported(format!("host extent vanished: {e}")))?;
        let pinned = source.is_pinned(q.host_loc);
        if payload as usize != total {
            return Err(GpuError::Unsupported(format!(
                "load spec total {total} != stored payload {payload} - spec/plane \
                 order mismatch between store and restore"
            )));
        }
        let (exec, dev) = (&self.exec, &mut self.dev[Dir::Load as usize]);
        let ring_slot = if pinned {
            // SAFETY: extent valid until free; length checked above.
            let src = unsafe { std::slice::from_raw_parts(host_ptr, total) };
            Self::htod(exec, src, dev)?;
            None
        } else {
            let slot = self.ring.acquire().ok_or_else(|| {
                GpuError::Unsupported("staging ring exhausted (pageable fallback)".into())
            })?;
            let (rp, _rl) = self.ring.ptr(slot);
            // host-side bounce copy: pageable slab -> pinned ring
            // SAFETY: both regions valid and non-overlapping by construction.
            unsafe { std::ptr::copy_nonoverlapping(host_ptr, rp, total) };
            let src = unsafe { std::slice::from_raw_parts(rp, total) };
            Self::htod(exec, src, dev)?;
            Some(slot)
        };
        exec.kv_scatter_blocks(
            &descs.0,
            &descs.1,
            dev,
            spec.record_stride(),
            spec.max_plane_bytes(),
            spec.planes.len(),
            spec.block_ids.len(),
        )?;
        let done = exec.record_event()?;
        Ok((done, ring_slot, descs))
    }

    fn make_descs(&self, spec: &XferSpec) -> Result<(CudaSlice<u64>, CudaSlice<u32>), GpuError> {
        let img = spec.desc_image();
        let mut planes = self.exec.alloc_u64(img.len())?;
        self.exec.upload_u64(&img, &mut planes)?;
        let blocks = self.exec.to_device_u32(&spec.block_ids)?;
        Ok((planes, blocks))
    }

    /// One contiguous D2H at extent granularity - the only shape the bus leg
    /// is allowed to have.
    fn dtoh(exec: &GpuExecutor, dev: &CudaSlice<u8>, dst: &mut [u8]) -> Result<(), GpuError> {
        let (dp, _g) = dev.device_ptr(&exec.stream);
        // SAFETY: dp addresses >= dst.len() bytes (extent capacity checked at
        // validate); stream-ordered after the gather on the same lane.
        unsafe { cudarc::driver::result::memcpy_dtoh_async(dst, dp as _, exec.stream.cu_stream()) }
            .map_err(|e| GpuError::Unsupported(format!("tier D2H failed: {e}")))
    }

    fn htod(exec: &GpuExecutor, src: &[u8], dev: &mut CudaSlice<u8>) -> Result<(), GpuError> {
        let (dp, _g) = dev.device_ptr_mut(&exec.stream);
        // SAFETY: symmetric to dtoh.
        unsafe { cudarc::driver::result::memcpy_htod_async(dp as _, src, exec.stream.cu_stream()) }
            .map_err(|e| GpuError::Unsupported(format!("tier H2D failed: {e}")))
    }

    /// Hand a disk load to the reader pool: land it in a fresh T1 extent
    /// when T1 has the room (that extent is then the read-fill copy, no
    /// second memcpy), else in the read arena.
    fn start_t2_read(&mut self, q: Queued) -> T2Start {
        let (Some(t2), Some(readers)) = (self.t2.as_ref(), self.t2_readers.as_ref()) else {
            return T2Start::Ring(q);
        };
        // an unknown key or a schema drift fails on the ring path, loudly
        let Some(plan) = t2.read_plan(&q.job.key.0) else {
            return T2Start::Ring(q);
        };
        if plan.len != q.spec.total() || plan.padded > self.store.rounded(plan.len) {
            return T2Start::Ring(q);
        }
        if !self.t2_reads.is_empty() && self.t2_read_bytes + plan.len > T2_READ_WINDOW {
            return T2Start::Wait(q);
        }
        let aligned = |p: *mut u8| (p as usize).is_multiple_of(plan.align as usize);
        let mut landing = None;
        if let Ok(loc) = self.store.alloc(plan.len) {
            match self.store.resolve(loc) {
                Ok((p, _)) if aligned(p) && self.store.is_pinned(loc) => {
                    landing = Some((loc, p, false))
                }
                _ => {
                    let _ = self.store.free(loc);
                }
            }
        }
        if landing.is_none()
            && let Some(stage) = self.t2_stage.as_mut()
        {
            match stage.alloc(plan.len) {
                Ok(loc) => match stage.resolve(loc) {
                    Ok((p, _)) if aligned(p) => landing = Some((loc, p, true)),
                    _ => {
                        let _ = stage.free(loc);
                    }
                },
                // the arena is busy with reads in flight: they free it
                Err(_) if !self.t2_reads.is_empty() => return T2Start::Wait(q),
                Err(_) => {}
            }
        }
        let Some((loc, dst, stage)) = landing else {
            return T2Start::Ring(q);
        };
        let job = ReadJob {
            op: q.job.op,
            path: plan.path,
            off: plan.off,
            dst,
            padded: plan.padded as usize,
            len: plan.len as usize,
            expect: plan.checksum,
        };
        // SAFETY: the landing is freshly allocated, aligned, at least
        // `padded` bytes (`rounded`, checked above), and stays allocated and untouched
        // until this op's completion is collected in `land_t2_reads`; the
        // reader pool drops (joining every read) before either store.
        if !unsafe { readers.submit(job) } {
            let l = Landed {
                stage,
                checksum: plan.checksum,
            };
            self.free_landing(loc, l);
            return T2Start::Ring(q);
        }
        self.t2_read_bytes += plan.len;
        self.t2_reads.insert(
            q.job.op,
            Landing {
                q,
                loc,
                landed: Landed {
                    stage,
                    checksum: plan.checksum,
                },
                len: plan.len,
            },
        );
        T2Start::Reading
    }

    /// Disk reads that finished: launch each from where it landed.
    fn land_t2_reads(&mut self) {
        let Some(readers) = self.t2_readers.as_ref() else {
            return;
        };
        for done in readers.collect() {
            let (op, outcome) = match done {
                ReadDone::Ok(op) => (op, Ok(())),
                ReadDone::Corrupt(op) => (op, Err(None)),
                ReadDone::Failed(op, e) => (op, Err(Some(e))),
            };
            let Some(l) = self.t2_reads.remove(&op) else {
                continue;
            };
            self.t2_read_bytes -= l.len;
            if self.cancelled.remove(&op) {
                self.free_landing(l.loc, l.landed);
                self.fail(op);
                continue;
            }
            match outcome {
                Ok(()) => {
                    let q = Queued {
                        job: l.q.job,
                        spec: l.q.spec,
                        host_loc: l.loc,
                    };
                    self.launch_load(q, Some(l.landed));
                }
                Err(e) => {
                    if e.is_none()
                        && let Some(t2) = self.t2.as_mut()
                    {
                        t2.mark_corrupt(&l.q.job.key.0);
                    } else if let Some(e) = e {
                        tracing::warn!(err = %e, "KV T2 read failed - completing Failed");
                    }
                    self.free_landing(l.loc, l.landed);
                    self.fail(op);
                }
            }
        }
    }

    /// Is the DISK busy with reads? Disk writes wait for this to be false.
    /// The device probes are why: on this class of device a concurrent
    /// writer collapses reads to 3% retention (11.54 -> 0.39 GB/s), which
    /// would turn every restore that overlaps a write burst into a loser
    /// against recompute - the same no-duplex law the PCIe lane has.
    fn t2_reads_active(&self) -> bool {
        !self.t2_reads.is_empty()
            || self.q_load.iter().any(|q| q.job.tier == super::Tier::Nvme)
            || self.flights.iter().any(|f| f.dir == Dir::Load && f.from_t2)
    }

    /// Whether a [`Self::persist_batch_inner`] could run now - see
    /// `XferSink::disk_ready`. The no-duplex law: no disk read queued or in
    /// flight.
    pub fn disk_ready_inner(&self) -> bool {
        self.disk_writable_inner() && !self.t2_reads_active()
    }

    /// A disk attached, its writes not paused, the day's budget not spent -
    /// reads or not (`XferSink::disk_writable`).
    pub fn disk_writable_inner(&self) -> bool {
        let Some(t2) = self.t2.as_ref() else {
            return false;
        };
        let spent = if utc_day() == self.t2_day {
            self.t2_written_today
        } else {
            0
        };
        !t2.writes_blocked() && spent < self.t2_budget
    }

    /// Write T1 extents to disk now, in order, as one group commit (see
    /// `NvmeStore::store_many`). What gets written, and in what order, is the
    /// pool tier's durability plan (`pool_tier/durable.rs`) - this used to
    /// write every T1 store through from a queue that dropped its oldest
    /// entries, chain heads among them. `items` carry the checksum T1
    /// computed off the engine thread, so nothing is hashed here. Stops at
    /// the first extent gone from T1: the ones after it stay unwritten.
    pub fn persist_batch_inner(
        &mut self,
        items: &[(LogicalKey, Loc, [u8; 32])],
    ) -> Vec<super::pool_tier::Persist> {
        use super::pool_tier::Persist;
        let mut out = vec![Persist::NoDisk; items.len()];
        let Some(t2) = self.t2.as_ref() else {
            return out;
        };
        // (index, payload, length, key, checksum) of what still needs writing
        let mut batch: Vec<(usize, *const u8, u64, LogicalKey, [u8; 32])> = Vec::new();
        for (i, &(key, loc, sum)) in items.iter().enumerate() {
            // content keys: already on disk means the same bytes
            if t2.contains(&key.0) {
                super::census::ev("wd", &key.0, 0, None);
                out[i] = Persist::Durable;
                continue;
            }
            let Ok((ptr, len)) = self.store.resolve(loc) else {
                out[i] = Persist::Gone;
                break;
            };
            batch.push((i, ptr as *const u8, len, key, sum));
        }
        if batch.is_empty() {
            return out;
        }
        let total: u64 = batch.iter().map(|b| b.2).sum();
        let day = utc_day();
        if day != self.t2_day {
            self.t2_day = day;
            self.t2_written_today = 0;
            self.t2_budget_warned = false;
        }
        if self.t2_written_today + total > self.t2_budget {
            for &(i, _, len, key, _) in &batch {
                super::census::ev("wb", &key.0, len, None);
                out[i] = Persist::Declined;
            }
            if !self.t2_budget_warned {
                self.t2_budget_warned = true;
                tracing::warn!(
                    written_gib = self.t2_written_today as f64 / (1u64 << 30) as f64,
                    "T2 daily write budget reached - durability pauses until the day \
                     rolls over (serving and T1 continue)"
                );
            }
            return out;
        }
        let store_items: Vec<StoreItem<'_>> = batch
            .iter()
            .map(|&(_, ptr, len, key, sum)| StoreItem {
                key: key.0,
                generation: 0,
                schema_version: 1,
                // SAFETY: T1 extents stay valid until freed, and nothing
                // frees one during this call; len is the payload length
                payload: unsafe { std::slice::from_raw_parts(ptr, len as usize) },
                checksum: Some(sum),
            })
            .collect();
        let Some(t2) = self.t2.as_mut() else {
            return out;
        };
        let mut r = t2.store_many(&store_items);
        if matches!(
            r,
            Err(StoreError::QuotaExhausted { .. } | StoreError::VolumeFull { .. })
        ) {
            match t2.make_room(total) {
                Ok(keys) => {
                    for k in &keys {
                        super::census::ev("x2", k, 0, None);
                    }
                    self.t2_evicted.extend(keys);
                    r = t2.store_many(&store_items);
                }
                Err(e) => tracing::warn!(err = %e, "T2 make_room failed"),
            }
        }
        let landed = match r {
            Ok(_) => {
                self.t2_written_today += total;
                Persist::Written
            }
            // the store logs its own state changes (pause, resume, the volume
            // reaching its reserve) once each - a line per key was a log
            // flood on a full disk
            Err(e) if e.is_capacity() => Persist::Declined,
            Err(e) => {
                tracing::debug!(err = %e, "T2 write failed - RAM-only for these keys");
                Persist::Failed
            }
        };
        for &(i, _, len, key, _) in &batch {
            let ev = if landed == Persist::Written {
                "w"
            } else {
                "wx"
            };
            super::census::ev(ev, &key.0, len, None);
            out[i] = landed;
        }
        out
    }

    /// Stores held for the slab supply: carve what fits now, complete the
    /// rest NoRoom once the supply is done and the arena still has no hole.
    fn retry_held(&mut self) {
        let n = self.q_held.len();
        for _ in 0..n {
            let Some((job, spec)) = self.q_held.pop_front() else {
                break;
            };
            match self.store.alloc(job.bytes) {
                Ok(host_loc) => self.q_store.push_back(Queued {
                    job,
                    spec,
                    host_loc,
                }),
                Err(_) if !self.store.supply_done() => self.q_held.push_back((job, spec)),
                Err(e) => {
                    tracing::debug!(err = %e, "KV tier store: no room in T1 (after hold)");
                    self.ready.push(IoCompletion {
                        op: job.op,
                        outcome: IoOutcome::NoRoom,
                    });
                }
            }
        }
    }

    /// Start whatever the queues and the no-duplex rule allow.
    /// [`XferSink::expedite`]: mark the queued ones urgent and start them.
    pub fn expedite_inner(&mut self, ops: &[OpId]) {
        let queued: HashSet<OpId> = self.q_store.iter().map(|q| q.job.op).collect();
        let before = self.urgent.len();
        self.urgent
            .extend(ops.iter().copied().filter(|op| queued.contains(op)));
        if self.urgent.len() > before {
            self.kick();
        }
    }

    fn kick(&mut self) {
        self.retry_held();
        // Loads first, always - and drain the whole queue onto the lane
        // stream: the lane serializes staging-extent reuse, and per-flight
        // done events let one pump retire them all. One-flight-per-pump
        // gated the lane on the SERVICE TICK cadence - a 24-run restore
        // took 24 ticks and always outlived its park deadline (found on a
        // live probe). Only a ring-bouncing flight (T2 source or the
        // pageable fallback) waits for a slot; a retire re-kicks. A disk
        // load goes to the reader pool instead (`start_t2_read`) and
        // launches from where it landed once read.
        while let Some(q) = self.q_load.front() {
            let disk = q.job.tier == super::Tier::Nvme;
            if disk && self.t2_readers.is_some() {
                let q = self.q_load.pop_front().expect("front exists");
                if self.cancelled.remove(&q.job.op) {
                    self.fail(q.job.op);
                    continue;
                }
                match self.start_t2_read(q) {
                    T2Start::Reading => continue,
                    T2Start::Wait(q) => {
                        // the window is full: reads in flight free it
                        self.q_load.push_front(q);
                        break;
                    }
                    // nowhere aligned to land it: the ring bounce below
                    T2Start::Ring(q) => {
                        if !self.ring.available() {
                            self.q_load.push_front(q);
                            break;
                        }
                        self.launch_load(q, None);
                        continue;
                    }
                }
            }
            let needs_ring = disk || !self.store.is_pinned(q.host_loc);
            if needs_ring && !self.ring.available() {
                break;
            }
            let q = self.q_load.pop_front().expect("front exists");
            if self.cancelled.remove(&q.job.op) {
                self.fail(q.job.op);
                continue;
            }
            self.launch_load(q, None);
        }
        // stores a restore waits on go now, beside loads: their content is
        // the restore's (a consult found the state it needs still on its
        // way into T1), so holding them only delays that restore
        if !self.urgent.is_empty() {
            let mut i = 0;
            while i < self.q_store.len() {
                if !self.urgent.contains(&self.q_store[i].job.op) {
                    i += 1;
                    continue;
                }
                if !self.store.is_pinned(self.q_store[i].host_loc) && !self.ring.available() {
                    break;
                }
                let q = self.q_store.remove(i).expect("index in range");
                self.urgent.remove(&q.job.op);
                if self.cancelled.remove(&q.job.op) {
                    self.free_extent(q.host_loc);
                    self.fail(q.job.op);
                    continue;
                }
                self.launch_store(q);
            }
            // what is not queued any more has launched (or finished)
            let queued: HashSet<OpId> = self.q_store.iter().map(|q| q.job.op).collect();
            self.urgent.retain(|op| queued.contains(op));
        }
        // stores only in load-slack (no duplex - a demote beside a
        // restore halves the restore) - and only so many bytes deep: the
        // lane stream is FIFO, so a restore that arrives next tick waits
        // behind every store already on the stream. The window refills
        // once per pump, i.e. once per service tick, so its depth IS the
        // write-through rate: four 4 MiB flights a tick was 16 MB/s under
        // a long prefill (~1 s ticks) against ~0.8 GB of KV and state a
        // 10K-token Qwen3.8 prompt produces - the queue never drained,
        // its pins starved the pool, and repeats probed runs still
        // queued. 256 MiB keeps a restore's wait near 10 ms at PCIe rate.
        // (a store still hashing holds its extent and its pool pins just the
        // same, so it counts until its completion is out)
        const STORE_INFLIGHT_BYTES: usize = 256 << 20;
        let open = |fl: &[Flight], h: &HashMap<OpId, Hashing>| -> usize {
            let flying: usize = fl
                .iter()
                .filter(|f| f.dir == Dir::Store)
                .map(|f| f.spec_total)
                .sum();
            let hashing: usize = h
                .values()
                .map(|x| match x {
                    Hashing::Store { bytes, .. } => *bytes as usize,
                    Hashing::Load { .. } => 0,
                })
                .sum();
            flying + hashing
        };
        while self.q_load.is_empty()
            && !self.flights.iter().any(|f| f.dir == Dir::Load)
            && open(&self.flights, &self.hashing) < STORE_INFLIGHT_BYTES
        {
            let Some(q) = self.q_store.front() else { break };
            let needs_ring = !self.store.is_pinned(q.host_loc);
            if needs_ring && !self.ring.available() {
                break;
            }
            let q = self.q_store.pop_front().expect("front exists");
            if self.cancelled.remove(&q.job.op) {
                self.free_extent(q.host_loc);
                self.fail(q.job.op);
                continue;
            }
            self.launch_store(q);
        }
    }
}

impl TierTransport for RamTransport {
    fn caps(&self) -> TransportCaps {
        TransportCaps {
            gather_scatter: true,
            direct_io_align: None,
            durable: false,
            cancellable: true,
        }
    }

    fn submit(&mut self, job: IoJob) -> Result<(), SubmitError> {
        match &job.kind {
            IoJobKind::Store { .. } => {
                let spec = self
                    .expect_store
                    .remove(&job.key)
                    .ok_or(SubmitError::SourceMissing)?;
                if spec.total() != job.bytes {
                    tracing::error!(
                        spec_total = spec.total(),
                        job_bytes = job.bytes,
                        "KV tier store: sealed bytes disagree with the device spec"
                    );
                    return Err(SubmitError::SourceMissing);
                }
                // the T1 extent is allocated now. The pump evicts until
                // `can_place` holds, so no room here is a race, not the
                // steady state - and it completes NoRoom (a capacity miss
                // the breaker ignores), never Failed: counting it as a fault
                // took a merely full tier offline for good
                let host_loc = match self.store.alloc(job.bytes) {
                    Ok(l) => l,
                    Err(e) if !self.store.supply_done() => {
                        // the budget is still being carved off-thread: hold
                        // the store and retry it at the next kick rather
                        // than dropping content the GPU has already shed
                        tracing::debug!(err = %e, "KV tier store: held for the slab supply");
                        self.q_held.push_back((job, spec));
                        return Ok(());
                    }
                    Err(e) => {
                        tracing::debug!(err = %e, "KV tier store: no room in T1");
                        self.ready.push(IoCompletion {
                            op: job.op,
                            outcome: IoOutcome::NoRoom,
                        });
                        return Ok(());
                    }
                };
                self.q_store.push_back(Queued {
                    job,
                    spec,
                    host_loc,
                });
            }
            IoJobKind::Load { loc, dst } => {
                if job.tier == super::Tier::Nvme && self.t2.is_none() {
                    tracing::warn!("KV transport: Nvme load without a T2 store");
                    self.fail(job.op);
                    return Ok(());
                }
                if !matches!(dst, LoadDst::Gpu) {
                    // NVMe->RAM promotion lands with T2; the catalog can express
                    // it but this transport cannot serve it yet.
                    tracing::warn!("KV RAM transport: tier-destination load unsupported (v1)");
                    self.fail(job.op);
                    return Ok(());
                }
                let spec = self
                    .expect_load
                    .remove(&job.key)
                    .ok_or(SubmitError::SourceMissing)?;
                let host_loc = *loc;
                self.q_load.push_back(Queued {
                    job,
                    spec,
                    host_loc,
                });
            }
        }
        self.kick();
        Ok(())
    }

    fn can_place(&self, extents: &[u64]) -> bool {
        self.store.fits(extents)
    }

    fn cancel(&mut self, op: OpId) {
        self.cancelled.insert(op);
    }

    fn poll(&mut self) -> Vec<IoCompletion> {
        // sweep finished flights (order irrelevant - each has its own event)
        let mut i = 0;
        while i < self.flights.len() {
            if !self.exec.event_done(&self.flights[i].done) {
                i += 1;
                continue;
            }
            let f = self.flights.swap_remove(i);
            if let Some(slot) = f.ring_slot {
                if f.dir == Dir::Store {
                    // pageable fallback: land the bounced bytes in the slab
                    if let Ok((dst, len)) = self.store.resolve(f.host_loc) {
                        let (rp, _) = self.ring.ptr(slot);
                        // SAFETY: distinct regions, lengths equal by launch.
                        unsafe { std::ptr::copy_nonoverlapping(rp, dst, len as usize) };
                    }
                }
                self.ring.release(slot);
            }
            let was_cancelled = self.cancelled.remove(&f.op);
            match (f.dir, was_cancelled) {
                (Dir::Store, true) => {
                    self.free_extent(f.host_loc);
                    self.fail(f.op);
                }
                (Dir::Store, false) => match self.store.resolve(f.host_loc) {
                    // the digest is computed off the tick; the completion
                    // goes out when it is back (see `hash_pool`)
                    Ok((ptr, len)) => {
                        // SAFETY: the extent is freed only after this op's
                        // completion, which waits for the digest.
                        unsafe {
                            self.hasher
                                .submit(f.op, ptr as *const u8, len as usize, false)
                        };
                        self.hashing.insert(
                            f.op,
                            Hashing::Store {
                                key: f.key,
                                loc: f.host_loc,
                                bytes: len,
                            },
                        );
                    }
                    Err(_) => {
                        self.free_extent(f.host_loc);
                        self.ready.push(IoCompletion {
                            op: f.op,
                            outcome: IoOutcome::Failed,
                        });
                    }
                },
                (Dir::Load, true) => {
                    if let Some(l) = f.landed {
                        self.free_landing(f.host_loc, l);
                    }
                    self.fail(f.op);
                }
                (Dir::Load, false) if f.landed.is_some() => {
                    super::census::ev("r2", &f.key.0, f.spec_total as u64, None);
                    let l = f.landed.expect("matched");
                    // read-fill: a landing in T1 IS the copy the next hit
                    // restores from at RAM speed (the pump adopts it or hands
                    // it back); one in the arena goes back to the arena
                    if l.stage {
                        self.free_landing(f.host_loc, l);
                    } else {
                        self.t2_promoted.push((
                            f.key.0,
                            f.host_loc,
                            l.checksum,
                            f.spec_total as u64,
                        ));
                    }
                    // verified against its commit record on the reader - the
                    // catalog's reference for this replica is that checksum
                    self.ready.push(IoCompletion {
                        op: f.op,
                        outcome: IoOutcome::LoadDone {
                            bytes: f.spec_total as u64,
                            checksum: Checksum(l.checksum),
                            dst_loc: None,
                        },
                    });
                }
                (Dir::Load, false) if f.from_t2 => {
                    super::census::ev("r2", &f.key.0, f.spec_total as u64, None);
                    // T2 load: the disk bytes verified against their commit
                    // record at launch; the ring slot still holds exactly
                    // what was delivered - hash those for the catalog's
                    // end-to-end check. (Slot release above is safe: reuse
                    // only happens at kick(), which runs after this sweep.)
                    let outcome = match f.ring_slot {
                        Some(slot) => {
                            let (rp, _rl) = self.ring.ptr(slot);
                            // SAFETY: no reuse before kick() at sweep end.
                            let bytes = unsafe {
                                std::slice::from_raw_parts(rp as *const u8, f.spec_total)
                            };
                            // `read_into` verified these bytes against their
                            // commit record: that checksum is the catalog's
                            // reference, so hashing them again proved nothing
                            let checksum = self
                                .t2
                                .as_ref()
                                .and_then(|t| t.commit_checksum(&f.key.0))
                                .map_or_else(|| Checksum::of_payload(bytes), Checksum);
                            // read-fill promotion: the payload bounced
                            // through the pinned ring anyway - seat a copy
                            // in the T1 slab so the next hit restores at
                            // RAM speed. A full slab just skips (the pump
                            // may also refuse on ledger grounds and hand
                            // the extent back via free_extent).
                            if let Ok(loc) = self.store.alloc(f.spec_total as u64) {
                                if let Ok((dst, _)) = self.store.resolve(loc) {
                                    // SAFETY: fresh extent, exclusively
                                    // ours; lengths equal by alloc.
                                    unsafe {
                                        std::ptr::copy_nonoverlapping(
                                            rp as *const u8,
                                            dst,
                                            f.spec_total,
                                        )
                                    };
                                    self.t2_promoted.push((
                                        f.key.0,
                                        loc,
                                        checksum.0,
                                        f.spec_total as u64,
                                    ));
                                } else {
                                    let _ = self.store.free(loc);
                                }
                            }
                            IoOutcome::LoadDone {
                                bytes: f.spec_total as u64,
                                checksum,
                                dst_loc: None,
                            }
                        }
                        None => IoOutcome::Failed,
                    };
                    self.ready.push(IoCompletion { op: f.op, outcome });
                }
                (Dir::Load, false) => {
                    super::census::ev("r1", &f.key.0, f.spec_total as u64, None);
                    // checksum over the SOURCE extent - at-rest integrity is
                    // the leg the checksum exists for; the H2D+scatter leg
                    // rides link-layer CRC like every other transfer in the
                    // engine
                    match self.store.resolve(f.host_loc) {
                        Ok((ptr, len)) => {
                            // SAFETY: the catalog pins a load's source until
                            // its completion, which waits for the digest.
                            unsafe {
                                self.hasher
                                    .submit(f.op, ptr as *const u8, len as usize, true)
                            };
                            self.hashing.insert(f.op, Hashing::Load { bytes: len });
                        }
                        Err(_) => self.ready.push(IoCompletion {
                            op: f.op,
                            outcome: IoOutcome::Failed,
                        }),
                    }
                }
            }
        }
        for (op, checksum) in self.hasher.collect() {
            let Some(h) = self.hashing.remove(&op) else {
                continue;
            };
            let cancelled = self.cancelled.remove(&op);
            match h {
                Hashing::Store { loc, .. } if cancelled => {
                    self.free_extent(loc);
                    self.fail(op);
                }
                Hashing::Store { key, loc, bytes } => {
                    // whether it also reaches the disk is the pool tier's
                    // durability plan, not every store's
                    super::census::ev("s", &key.0, bytes, None);
                    self.ready.push(IoCompletion {
                        op,
                        outcome: IoOutcome::StoreDone {
                            loc,
                            bytes,
                            checksum,
                        },
                    });
                }
                Hashing::Load { .. } if cancelled => self.fail(op),
                Hashing::Load { bytes } => self.ready.push(IoCompletion {
                    op,
                    outcome: IoOutcome::LoadDone {
                        bytes,
                        checksum,
                        dst_loc: None,
                    },
                }),
            }
        }
        self.land_t2_reads();
        self.kick();
        std::mem::take(&mut self.ready)
    }
}
