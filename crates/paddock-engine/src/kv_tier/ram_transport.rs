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
use super::nvme_store::NvmeStore;
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
    /// This flight's descriptor buffers - flights PIPELINE on the lane
    /// stream, so a shared buffer would be overwritten by flight N+1's
    /// upload before flight N's kernel reads it. Held until the event
    /// retires; the drop is a stream-ordered mempool free. Never read: it
    /// exists to be held.
    #[allow(dead_code)]
    descs: (CudaSlice<u64>, CudaSlice<u32>),
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

/// Deferred write-through backlog cap. Each entry is 40 bytes and names a
/// live T1 extent, so the memory cost is nothing; the cap exists so a long
/// read-heavy phase cannot accumulate a write burst that then fires all at
/// once when the reads finally stop.
const T2_PENDING_MAX: usize = 512;

/// The elected T2 daily write budget (see the field note): 1 TiB.
const T2_DAY_BUDGET: u64 = 1 << 40;

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
    /// Finished flights waiting on their digest, by op.
    hashing: HashMap<OpId, Hashing>,
    store: HostStore,
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
    q_load: VecDeque<Queued>,
    /// Quota-eviction outbox: keys `make_room` retired, awaiting the pump's
    /// catalog invalidation.
    t2_evicted: Vec<[u8; 32]>,
    /// 3.2 write deferral: durable write-throughs waiting for DISK read
    /// slack. The device probes are the reason this queue exists - on this class of
    /// device a concurrent writer collapses reads to 3% retention (11.54 ->
    /// 0.39 GB/s), which would turn every restore that overlaps a demote
    /// burst into a loser against recompute. So the disk gets the same
    /// no-duplex law the PCIe lane already has: writes drain only while no
    /// T2 read is queued or in flight.
    t2_pending: VecDeque<(LogicalKey, Loc)>,
    /// Read-fill outbox: (key, T1 loc, checksum, bytes) promoted off a T2
    /// load, awaiting the pump's catalog adoption (or extent return).
    t2_promoted: Vec<([u8; 32], Loc, [u8; 32], u64)>,
    /// 3.3 endurance budget: payload bytes written through to T2 in the
    /// current UTC day. The write-through STOPS at the budget (serving and
    /// T1 continue; only durability degrades, loudly once) so the mirror
    /// pass can never wear an SSD unboundedly. Elected, not tunable -
    /// 1 TiB/day is < 1 DWPD on a 2 TB drive; `PADDOCK_KV_T2_DAY_GB` is a
    /// dev-build override, compiled out of hardened builds.
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
                        "KV T2 store recovered"
                    );
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
        tracing::info!(
            capacity_gib = ram_capacity as f64 / (1u64 << 30) as f64,
            staging_mib = device_staging_bytes() >> 20,
            "KV RAM tier transport ready"
        );
        Ok(Self {
            exec,
            hasher: HashPool::new(),
            hashing: HashMap::new(),
            store,
            t2,
            ring,
            dev,
            expect_store: HashMap::new(),
            expect_load: HashMap::new(),
            q_store: VecDeque::new(),
            q_load: VecDeque::new(),
            t2_evicted: Vec::new(),
            t2_pending: VecDeque::new(),
            t2_promoted: Vec::new(),
            t2_day: utc_day(),
            t2_written_today: 0,
            t2_budget: paddock_models::dev_var_os!("PADDOCK_KV_T2_DAY_GB")
                .and_then(|v| v.to_str().and_then(|t| t.parse::<u64>().ok()))
                .map(|gb| gb << 30)
                .unwrap_or(T2_DAY_BUDGET),
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

    /// Keys the T2 cache evicted for quota - the tier's pump drains these
    /// and drops the catalog's Nvme references (see `XferSink`).
    pub fn take_t2_evictions_inner(&mut self) -> Vec<[u8; 32]> {
        std::mem::take(&mut self.t2_evicted)
    }

    /// Read-fill promotions off completed T2 loads (see `XferSink`).
    pub fn take_t2_promotions_inner(&mut self) -> Vec<([u8; 32], Loc, [u8; 32], u64)> {
        std::mem::take(&mut self.t2_promoted)
    }

    /// Durable writes deferred to read slack right now - the
    /// export, and the observable that makes the deferral law testable.
    pub fn t2_pending_writes(&self) -> usize {
        self.t2_pending.len()
    }

    /// Endurance telemetry: T2 payload bytes written this UTC day.
    pub fn t2_written_today_inner(&self) -> u64 {
        self.t2_written_today
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

    fn launch_load(&mut self, q: Queued) {
        let op = q.job.op;
        match self.launch_load_inner(&q) {
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
                    descs,
                });
            }
            Err(e) => {
                tracing::warn!(err = %e, "KV tier load launch failed - completing Failed");
                self.fail(op);
            }
        }
    }

    fn launch_load_inner(
        &mut self,
        q: &Queued,
    ) -> Result<(CudaEvent, Option<usize>, (CudaSlice<u64>, CudaSlice<u32>)), GpuError> {
        let spec = &q.spec;
        let total = spec.total() as usize;
        let descs = self.make_descs(spec)?;
        if let Some(ev) = &spec.after {
            self.exec.wait_event(ev)?;
        }
        // T2 source: the payload lives on disk. Read it (VERIFIED against
        // its commit record) into a pinned ring slot, then the same H2D +
        // scatter as any load - the ring is exactly the CPU-bounce the plan
        // elected for every disk path.
        if q.job.tier == super::Tier::Nvme {
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
        let (host_ptr, payload) = self
            .store
            .resolve(q.host_loc)
            .map_err(|e| GpuError::Unsupported(format!("host extent vanished: {e}")))?;
        if payload as usize != total {
            return Err(GpuError::Unsupported(format!(
                "load spec total {total} != stored payload {payload} - spec/plane \
                 order mismatch between store and restore"
            )));
        }
        let (exec, dev) = (&self.exec, &mut self.dev[Dir::Load as usize]);
        let ring_slot = if self.store.is_pinned(q.host_loc) {
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

    /// Is the DISK busy with reads? Deferred write-through waits for this
    /// to be false (a writer beside a reader costs 97% of the
    /// read bandwidth on consumer-RAID-class storage).
    fn t2_reads_active(&self) -> bool {
        self.q_load.iter().any(|q| q.job.tier == super::Tier::Nvme)
            || self.flights.iter().any(|f| f.dir == Dir::Load && f.from_t2)
    }

    /// Flush every deferred write, ignoring read slack. Called at teardown:
    /// deferral is a scheduling choice about when durability happens, not
    /// whether - a runner that exits with a full pending queue would lose
    /// exactly the warm sessions the restart-persistence gate exists to
    /// keep. Nothing is reading the device at this point anyway.
    pub fn flush_t2(&mut self) {
        while !self.t2_pending.is_empty() {
            let before = self.t2_pending.len();
            self.drain_t2_writes_inner(usize::MAX);
            if self.t2_pending.len() == before {
                break; // budget exhausted or store gone: nothing more will land
            }
        }
    }

    /// Drain deferred durable writes in read slack. Bounded per pass so one
    /// drain can never hold the pump while a restore arrives - the next pass
    /// picks up where this one stopped, and an arriving T2 read stops the
    /// drain at the next entry.
    fn drain_t2_writes(&mut self) {
        const PER_PASS: usize = 4;
        if self.t2_reads_active() {
            return; // a restore wants the device; it gets the device
        }
        self.drain_t2_writes_inner(PER_PASS)
    }

    fn drain_t2_writes_inner(&mut self, per_pass: usize) {
        if self.t2.is_none() || self.t2_pending.is_empty() {
            return;
        }
        let day = utc_day();
        if day != self.t2_day {
            self.t2_day = day;
            self.t2_written_today = 0;
            self.t2_budget_warned = false;
        }
        for _ in 0..per_pass {
            let Some((key, loc)) = self.t2_pending.pop_front() else {
                return;
            };
            // the T1 extent may have been evicted while the write waited -
            // that is a lost durability opportunity, never a correctness
            // problem (the tier is a cache; the next store re-offers it)
            let Ok((ptr, len)) = self.store.resolve(loc) else {
                continue;
            };
            if self.t2_written_today + len > self.t2_budget {
                if !self.t2_budget_warned {
                    self.t2_budget_warned = true;
                    tracing::warn!(
                        written_gib = self.t2_written_today as f64 / (1u64 << 30) as f64,
                        "T2 daily write budget reached - durability pauses until the day \
                         rolls over (serving and T1 continue)"
                    );
                }
                self.t2_pending.clear(); // nothing else will fit today either
                return;
            }
            // SAFETY: extent valid until free; len is the payload length.
            let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, len as usize) };
            let Some(t2) = self.t2.as_mut() else { return };
            let mut r = t2.store(key.0, 0, 1, bytes);
            if matches!(r, Err(super::nvme_store::StoreError::QuotaExhausted { .. })) {
                match t2.make_room(len) {
                    Ok(keys) => {
                        self.t2_evicted.extend(keys);
                        r = t2.store(key.0, 0, 1, bytes);
                    }
                    Err(e) => tracing::warn!(err = %e, "T2 make_room failed"),
                }
            }
            match r {
                Ok(_) => self.t2_written_today += len,
                Err(e) => {
                    tracing::warn!(err = %e, "T2 write-through failed - RAM-only for this key")
                }
            }
        }
    }

    /// Start whatever the queues and the no-duplex rule allow.
    fn kick(&mut self) {
        // Loads first, always - and drain the whole queue onto the lane
        // stream: the lane serializes staging-extent reuse, and per-flight
        // done events let one pump retire them all. One-flight-per-pump
        // gated the lane on the SERVICE TICK cadence - a 24-run restore
        // took 24 ticks and always outlived its park deadline (found on a
        // live probe). Only a ring-bouncing flight (T2 source or the
        // pageable fallback) waits for a slot; a retire re-kicks.
        while let Some(q) = self.q_load.front() {
            let needs_ring = q.job.tier == super::Tier::Nvme || !self.store.is_pinned(q.host_loc);
            if needs_ring && !self.ring.available() {
                break;
            }
            let q = self.q_load.pop_front().expect("front exists");
            if self.cancelled.remove(&q.job.op) {
                self.fail(q.job.op);
                continue;
            }
            self.launch_load(q);
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
                (Dir::Load, true) => self.fail(f.op),
                (Dir::Load, false) if f.from_t2 => {
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
                            let checksum = Checksum::of_payload(bytes);
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
                    // Restart write-through, DEFERRED: the extent is
                    // durable-eligible now, but the disk write waits for
                    // read slack (see `t2_pending`). Ordering is irrelevant
                    // to correctness here - the T1 copy already serves, and
                    // the durable copy only has to exist before the next
                    // restart.
                    if self.t2.is_some() {
                        self.t2_pending.push_back((key, loc));
                        if self.t2_pending.len() > T2_PENDING_MAX {
                            // a cache, not a journal: the oldest deferred
                            // write is the one whose T1 extent is likeliest
                            // to be gone anyway
                            self.t2_pending.pop_front();
                        }
                    }
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
        self.kick();
        // durable write-through rides the read slack the kick just left
        self.drain_t2_writes();
        std::mem::take(&mut self.ready)
    }
}

impl Drop for RamTransport {
    fn drop(&mut self) {
        // last call for durability - see `flush_t2`
        if self.t2.is_some() && !self.t2_pending.is_empty() {
            let n = self.t2_pending.len();
            self.flush_t2();
            tracing::debug!(
                deferred = n,
                "KV T2: flushed deferred write-through at teardown"
            );
        }
    }
}
