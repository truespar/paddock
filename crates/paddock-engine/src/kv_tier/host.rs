//! T1 host memory: capacity slabs + the pinned staging ring.
//!
//! Two kinds of host memory with different jobs ("storage capacity
//! ≠ pinned transport pool"), both elected by the host-memory probe.
//!
//! - **Capacity slabs** ([`HostStore`]): where demoted payload extents live.
//!   Allocated as ordinary pageable memory, touched, then registered once via
//!   `cuMemHostRegister` - 5.9 ms/GiB vs 72-151 ms for
//!   `cuMemHostAlloc`, and registered ≡ pinned ≡ 26.7 GB/s on the bus, so a
//!   large T1 costs milliseconds to arm and DMA runs straight into it (no
//!   bounce). Registration failing (WDDM working-set limits, exotic hosts) is
//!   the loud fallback: the store keeps working pageable and transfers bounce
//!   through the staging ring - slower, said once at WARN, never silent.
//! - **Staging ring** ([`StagingRing`]): a small bounded pool of
//!   **cached-pinned** extents (`cuMemHostAlloc`, no write-combining - WC
//!   reads collapse 389x, and checksums/NVMe writes READ this memory;
//!   cudarc's `alloc_pinned` is WC and stays banned here). Used as the DMA
//!   bounce when slabs are pageable, and by the NVMe path.
//!
//! Everything here must run on the engine thread with the CUDA context
//! current (allocation, registration, and Drop) - the same contract every
//! other device-adjacent resource in this crate already lives under.

use std::collections::BTreeMap;

use super::Loc;

/// Allocation granularity inside a slab. Extents are 2-16 MiB,
/// so 64 KiB rounding wastes < 3% worst-case and keeps the free-range math
/// trivial; it is also a multiple of every direct-IO alignment the device
/// probe discovered,
/// so a Phase-3 NVMe path can reuse slab-resident extents in place.
pub const ALLOC_ALIGN: u64 = 64 * 1024;

/// Default slab size. Big enough that registration cost (5.9 ms/GiB) is paid
/// in coarse strides, small enough that the last slab's rounding never
/// strands gigabytes of a small budget.
pub const DEFAULT_SLAB_BYTES: u64 = 1 << 30;

#[derive(Debug, thiserror::Error)]
pub enum HostMemError {
    #[error("host tier exhausted: {wanted} bytes wanted, {free} free")]
    Exhausted { wanted: u64, free: u64 },
    #[error("unknown host location {0:?} (double free or foreign loc)")]
    UnknownLoc(Loc),
    #[error("host allocation failed: {0}")]
    Alloc(String),
}

/// What the store's transfer path actually got - read by the transport to
/// decide direct-DMA vs ring-bounce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PinMode {
    /// Slabs are page-locked (registered): DMA moves directly between device
    /// staging and slab extents at full bus rate.
    Registered,
    /// Registration unavailable: slabs are plain pageable memory; transfers
    /// bounce through the pinned staging ring (or run pageable if even the
    /// ring could not pin - still correct, announced loudly).
    Pageable,
}

/// Write one byte per page so registration locks committed pages, not
/// zero-page COW mappings - the probe measured register-of-touched at
/// 5.9 ms/GiB and that is the contract we arm.
fn touch_pages(buf: &mut [u8]) {
    for i in (0..buf.len()).step_by(4096) {
        // volatile so the write is not optimized away over a fresh zeroed Vec
        // SAFETY: i < buf.len(); one-byte write inside the slice
        unsafe { std::ptr::write_volatile(buf.as_mut_ptr().add(i), 0) };
    }
}

struct Slab {
    mem: Box<[u8]>,
    registered: bool,
    /// Free ranges (offset, len), sorted by offset, coalesced on free.
    free: Vec<(u64, u64)>,
}

impl Slab {
    fn new(bytes: u64, want_register: bool) -> (Self, bool) {
        let mut mem = vec![0u8; bytes as usize].into_boxed_slice();
        let mut registered = false;
        if want_register {
            touch_pages(&mut mem);
            // SAFETY: mem is a live allocation of exactly `bytes`; the flag
            // set is PORTABLE (usable from any context), never WC.
            let rc = unsafe {
                cudarc::driver::sys::cuMemHostRegister_v2(
                    mem.as_mut_ptr().cast(),
                    bytes as usize,
                    cudarc::driver::sys::CU_MEMHOSTREGISTER_PORTABLE,
                )
            }
            .result();
            registered = rc.is_ok();
            if let Err(e) = rc {
                tracing::warn!(
                    bytes,
                    err = %e,
                    "cuMemHostRegister failed for a KV tier slab - this slab \
                     serves PAGEABLE (transfers bounce through the staging \
                     ring; correctness unaffected, bandwidth reduced)"
                );
            }
        }
        let free = vec![(0, bytes)];
        (
            Self {
                mem,
                registered,
                free,
            },
            registered,
        )
    }

    fn alloc(&mut self, rounded: u64) -> Option<u64> {
        // first-fit over the sorted range list - extents are large and few,
        // the scan is nothing
        for i in 0..self.free.len() {
            let (off, len) = self.free[i];
            if len >= rounded {
                if len == rounded {
                    self.free.remove(i);
                } else {
                    self.free[i] = (off + rounded, len - rounded);
                }
                return Some(off);
            }
        }
        None
    }

    fn free_range(&mut self, off: u64, len: u64) {
        let at = self.free.partition_point(|(o, _)| *o < off);
        self.free.insert(at, (off, len));
        // coalesce with the neighbor on each side
        if at + 1 < self.free.len() {
            let (no, nl) = self.free[at + 1];
            if off + len == no {
                self.free[at].1 += nl;
                self.free.remove(at + 1);
            }
        }
        if at > 0 {
            let (po, pl) = self.free[at - 1];
            if po + pl == off {
                self.free[at - 1].1 += self.free[at].1;
                self.free.remove(at);
            }
        }
    }
}

impl Drop for Slab {
    fn drop(&mut self) {
        if self.registered {
            // SAFETY: registered exactly once on this pointer; engine-thread
            // drop contract (context current). Failure is unrecoverable and
            // harmless at teardown - the process is releasing everything.
            let _ =
                unsafe { cudarc::driver::sys::cuMemHostUnregister(self.mem.as_mut_ptr().cast()) };
        }
    }
}

/// One slab's occupancy (see [`HostStore::layout`]).
pub struct SlabLayout {
    pub len: u64,
    /// `(offset, rounded length, loc)` of each live extent, by offset.
    pub extents: Vec<(u64, u64, Loc)>,
}

#[derive(Debug, Clone, Copy)]
struct AllocInfo {
    slab: usize,
    off: u64,
    rounded: u64,
    payload: u64,
}

/// T1 capacity: lazily-grown, register-once slabs with a coalescing
/// range allocator. Hands out [`Loc`]s the catalog round-trips; the transport
/// resolves them back to pointers for DMA/checksum.
///
/// Capacity accounting note: the CATALOG's byte ledger admits exact sealed
/// bytes; this store rounds each extent up to [`ALLOC_ALIGN`] and slabs are
/// carved in [`DEFAULT_SLAB_BYTES`] strides, so physical exhaustion can
/// arrive slightly before the ledger's. That is handled honestly: `alloc`
/// fails, the transport completes the op `Failed`, the catalog releases the
/// reservation and serving recomputes - never a wedge, never silent (the
/// failure counts in `Counters::io_failures` and logs).
pub struct HostStore {
    slabs: Vec<Slab>,
    /// Total budget in bytes - slabs never grow past it.
    capacity: u64,
    slab_bytes: u64,
    /// Whether new slabs should attempt registration. Latches false on the
    /// first failure so the store does not retry a losing syscall per slab.
    try_register: bool,
    mode: PinMode,
    allocs: BTreeMap<u64, AllocInfo>,
    next_loc: u64,
    allocated: u64,
}

// SAFETY: the store is used from the engine thread only (creation, alloc,
// free, drop all happen there - the transport that owns it lives on that
// thread). Send is required because the engine/generator that owns the
// transport is itself moved across threads at startup.
unsafe impl Send for HostStore {}

impl HostStore {
    /// A store with `capacity` bytes of budget. `register` = attempt the
    /// The elected page-lock path (false only in unit tests without a GPU).
    pub fn new(capacity: u64, register: bool) -> Self {
        Self {
            slabs: Vec::new(),
            capacity,
            slab_bytes: DEFAULT_SLAB_BYTES.min(capacity.max(ALLOC_ALIGN)),
            try_register: register,
            mode: if register {
                PinMode::Registered
            } else {
                PinMode::Pageable
            },
            allocs: BTreeMap::new(),
            next_loc: 0,
            allocated: 0,
        }
    }

    /// The transfer mode transfers should assume right now. Registered until
    /// a slab registration fails, then pageable for the rest of the process
    /// (mixed-mode transfers are not worth their complexity - one failure
    /// says the OS is short on lockable pages, stop asking).
    pub fn mode(&self) -> PinMode {
        self.mode
    }

    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    /// Bytes currently allocated to extents (rounded sizes - the physical truth).
    pub fn allocated(&self) -> u64 {
        self.allocated
    }

    /// Allocate an extent for `payload` bytes. The rounded footprint is what
    /// the store charges; the payload length is what `resolve` reports back.
    pub fn alloc(&mut self, payload: u64) -> Result<Loc, HostMemError> {
        let rounded = self.rounded(payload);
        // try existing slabs first
        for (si, slab) in self.slabs.iter_mut().enumerate() {
            if let Some(off) = slab.alloc(rounded) {
                return Ok(self.record(si, off, rounded, payload));
            }
        }
        // grow: a new slab, budget permitting. The last slab may be smaller
        // than the stride so a budget that is not a slab multiple is still
        // fully usable.
        let used: u64 = self.slabs.iter().map(|s| s.mem.len() as u64).sum();
        let room = self.capacity.saturating_sub(used);
        let want = self.slab_bytes.min(room).max(rounded.min(room));
        if want < rounded {
            return Err(HostMemError::Exhausted {
                wanted: payload,
                free: self.capacity.saturating_sub(self.allocated),
            });
        }
        let (mut slab, registered) = Slab::new(want, self.try_register);
        if self.try_register && !registered {
            self.try_register = false;
            self.mode = PinMode::Pageable;
        }
        let off = slab
            .alloc(rounded)
            .expect("fresh slab fits its own trigger");
        self.slabs.push(slab);
        Ok(self.record(self.slabs.len() - 1, off, rounded, payload))
    }

    /// The footprint `alloc` charges for `payload` bytes.
    pub fn rounded(&self, payload: u64) -> u64 {
        payload.div_ceil(ALLOC_ALIGN) * ALLOC_ALIGN
    }

    /// Every carved slab's length and live extents `(offset, rounded, loc)`,
    /// sorted by offset - what eviction needs to open a hole where one fits.
    pub fn layout(&self) -> Vec<SlabLayout> {
        let mut out: Vec<SlabLayout> = self
            .slabs
            .iter()
            .map(|s| SlabLayout {
                len: s.mem.len() as u64,
                extents: Vec::new(),
            })
            .collect();
        for (&loc, a) in &self.allocs {
            out[a.slab].extents.push((a.off, a.rounded, Loc(loc)));
        }
        for s in &mut out {
            s.extents.sort_unstable_by_key(|e| e.0);
        }
        out
    }

    /// Whether `alloc` would place every extent in `payloads`, in order,
    /// right now - the same first-fit over the same free ranges, and the same
    /// slab growth. The catalog's ledger counts exact bytes; this arena needs
    /// each rounded extent CONTIGUOUS inside one slab, so a ledger with room
    /// can still sit on an arena whose free bytes are holes too small for the
    /// next extent. Eviction asks this, not the ledger, when it is done.
    pub fn fits(&self, payloads: &[u64]) -> bool {
        let mut free: Vec<Vec<(u64, u64)>> = self.slabs.iter().map(|s| s.free.clone()).collect();
        let mut used: u64 = self.slabs.iter().map(|s| s.mem.len() as u64).sum();
        'next: for &p in payloads {
            let rounded = self.rounded(p);
            for ranges in free.iter_mut() {
                if let Some(i) = ranges.iter().position(|&(_, len)| len >= rounded) {
                    let (off, len) = ranges[i];
                    if len == rounded {
                        ranges.remove(i);
                    } else {
                        ranges[i] = (off + rounded, len - rounded);
                    }
                    continue 'next;
                }
            }
            let room = self.capacity.saturating_sub(used);
            let want = self.slab_bytes.min(room).max(rounded.min(room));
            if want < rounded {
                return false;
            }
            used += want;
            free.push(if want > rounded {
                vec![(rounded, want - rounded)]
            } else {
                Vec::new()
            });
        }
        true
    }

    fn record(&mut self, slab: usize, off: u64, rounded: u64, payload: u64) -> Loc {
        let loc = Loc(self.next_loc);
        self.next_loc += 1;
        self.allocs.insert(
            loc.0,
            AllocInfo {
                slab,
                off,
                rounded,
                payload,
            },
        );
        self.allocated += rounded;
        loc
    }

    /// Release an extent. Unknown locs are an error, not a shrug - a double
    /// free here is a transport bug the tests must see.
    pub fn free(&mut self, loc: Loc) -> Result<(), HostMemError> {
        let info = self
            .allocs
            .remove(&loc.0)
            .ok_or(HostMemError::UnknownLoc(loc))?;
        self.slabs[info.slab].free_range(info.off, info.rounded);
        self.allocated -= info.rounded;
        Ok(())
    }

    /// Pointer + payload length for an extent - the DMA/checksum target. The
    /// pointer stays valid until `free(loc)`: slabs never move or shrink.
    pub fn resolve(&self, loc: Loc) -> Result<(*mut u8, u64), HostMemError> {
        let info = self
            .allocs
            .get(&loc.0)
            .ok_or(HostMemError::UnknownLoc(loc))?;
        // SAFETY: offset is inside the slab by construction
        let ptr = unsafe { self.slabs[info.slab].mem.as_ptr().add(info.off as usize) as *mut u8 };
        Ok((ptr, info.payload))
    }

    /// Whether the extent behind `loc` sits in page-locked memory (per-slab:
    /// early slabs may be registered while later ones fell back).
    pub fn is_pinned(&self, loc: Loc) -> bool {
        self.allocs
            .get(&loc.0)
            .map(|i| self.slabs[i.slab].registered)
            .unwrap_or(false)
    }
}

/// One cached-pinned extent, RAII over `cuMemHostAlloc`/`cuMemFreeHost`.
/// Falls back to a plain heap allocation when pinning fails (`pinned()`
/// reports which) - the ring stays functional either way.
struct PinnedExtent {
    ptr: *mut u8,
    len: usize,
    /// Some = heap fallback storage; None = driver-owned pinned pages.
    fallback: Option<Box<[u8]>>,
}

impl PinnedExtent {
    fn new(len: usize) -> Self {
        // CACHED pinned: PORTABLE only, never CU_MEMHOSTALLOC_WRITECOMBINED.
        // SAFETY: len > 0; the returned pointer is owned here until Drop.
        match unsafe {
            cudarc::driver::result::malloc_host(len, cudarc::driver::sys::CU_MEMHOSTALLOC_PORTABLE)
        } {
            Ok(p) => Self {
                ptr: p.cast(),
                len,
                fallback: None,
            },
            Err(e) => {
                tracing::warn!(
                    len,
                    err = %e,
                    "cuMemHostAlloc failed for a staging-ring extent - heap \
                     fallback (DMA through it degrades to staged copies)"
                );
                let mut b = vec![0u8; len].into_boxed_slice();
                Self {
                    ptr: b.as_mut_ptr(),
                    len,
                    fallback: Some(b),
                }
            }
        }
    }

    fn pinned(&self) -> bool {
        self.fallback.is_none()
    }
}

impl Drop for PinnedExtent {
    fn drop(&mut self) {
        if self.fallback.is_none() {
            // SAFETY: ptr came from malloc_host and is freed exactly once;
            // engine-thread drop contract (context current).
            let _ = unsafe { cudarc::driver::result::free_host(self.ptr.cast()) };
        }
    }
}

/// Bounded pool of address-stable cached-pinned extents. Pooled at startup
/// because pinned allocation costs milliseconds per call -
/// never allocated per-op.
pub struct StagingRing {
    extents: Vec<PinnedExtent>,
    busy: Vec<bool>,
}

// SAFETY: engine-thread ownership, same contract as HostStore.
unsafe impl Send for StagingRing {}

impl StagingRing {
    pub fn new(n: usize, extent_bytes: usize) -> Self {
        let extents: Vec<PinnedExtent> = (0..n).map(|_| PinnedExtent::new(extent_bytes)).collect();
        let pinned = extents.iter().filter(|e| e.pinned()).count();
        tracing::info!(
            extents = n,
            extent_mib = extent_bytes / (1 << 20),
            pinned,
            "KV tier staging ring ready"
        );
        let busy = vec![false; n];
        Self { extents, busy }
    }

    pub fn extent_bytes(&self) -> usize {
        self.extents.first().map(|e| e.len).unwrap_or(0)
    }

    /// Claim a free extent slot, or None (caller queues the op).
    /// A free slot exists - the kick loop's gate for ring-bouncing flights
    /// (pipelined flights each hold a slot until their event retires).
    pub fn available(&self) -> bool {
        self.busy.iter().any(|b| !b)
    }

    pub fn acquire(&mut self) -> Option<usize> {
        let i = self.busy.iter().position(|b| !*b)?;
        self.busy[i] = true;
        Some(i)
    }

    pub fn release(&mut self, slot: usize) {
        debug_assert!(self.busy[slot], "release of a free staging slot");
        self.busy[slot] = false;
    }

    /// The slot's stable pointer. Valid for the ring's lifetime.
    pub fn ptr(&self, slot: usize) -> (*mut u8, usize) {
        (self.extents[slot].ptr, self.extents[slot].len)
    }

    pub fn is_pinned(&self, slot: usize) -> bool {
        self.extents[slot].pinned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // All host-store tests run register=false: allocator logic is identical
    // and the suite must pass on GPU-less CI. The registered path is proven
    // by the GPU-gated round-trip gates in `ram_transport`.

    #[test]
    fn alloc_free_recycles_and_coalesces() {
        let mut s = HostStore::new(4 << 20, false);
        let a = s.alloc(1 << 20).unwrap();
        let b = s.alloc(1 << 20).unwrap();
        let c = s.alloc(1 << 20).unwrap();
        assert_eq!(s.allocated(), 3 << 20);
        // free the middle, then the first - coalescing must let a 2 MiB
        // extent land in the hole
        s.free(b).unwrap();
        s.free(a).unwrap();
        let d = s.alloc(2 << 20).unwrap();
        assert_eq!(s.allocated(), (1 << 20) + (2 << 20));
        s.free(c).unwrap();
        s.free(d).unwrap();
        assert_eq!(s.allocated(), 0);
    }

    /// The arena the customer's tier died on: free bytes enough for the next
    /// extent but no hole that large. `fits` must say no exactly where
    /// `alloc` fails, and yes again once a neighbor frees.
    #[test]
    fn fits_answers_for_holes_not_free_bytes() {
        let mut s = HostStore::new(8 << 20, false);
        let x: Vec<_> = (0..8).map(|_| s.alloc(1 << 20).unwrap()).collect();
        // free every other MiB: 4 MiB free, no 2 MiB hole
        for i in (0..8).step_by(2) {
            s.free(x[i]).unwrap();
        }
        assert_eq!(s.allocated(), 4 << 20);
        assert!(s.fits(&[1 << 20]));
        assert!(
            !s.fits(&[2 << 20]),
            "4 MiB free in 1 MiB holes is no room for 2 MiB"
        );
        assert!(s.alloc(2 << 20).is_err());
        // four 1 MiB extents fit, a fifth does not
        assert!(s.fits(&[1 << 20; 4]));
        assert!(!s.fits(&[1 << 20; 5]));
        // free a neighbor: the 3 MiB hole now takes the 2 MiB extent
        s.free(x[1]).unwrap();
        assert!(s.fits(&[2 << 20]));
        assert!(s.alloc(2 << 20).is_ok());
        // fits never allocates
        let before = s.allocated();
        let _ = s.fits(&[1 << 20, 1 << 20]);
        assert_eq!(s.allocated(), before);
    }

    #[test]
    fn rounding_charges_the_rounded_size() {
        let mut s = HostStore::new(1 << 20, false);
        let a = s.alloc(1000).unwrap();
        assert_eq!(s.allocated(), ALLOC_ALIGN);
        let (_, payload) = s.resolve(a).unwrap();
        assert_eq!(payload, 1000, "payload length survives the rounding");
        s.free(a).unwrap();
    }

    #[test]
    fn exhaustion_is_an_error_not_a_wedge() {
        let mut s = HostStore::new(1 << 20, false);
        let _a = s.alloc(1 << 20).unwrap();
        match s.alloc(1) {
            Err(HostMemError::Exhausted { .. }) => {}
            other => panic!("expected Exhausted, got {other:?}"),
        }
    }

    #[test]
    fn double_free_is_loud() {
        let mut s = HostStore::new(1 << 20, false);
        let a = s.alloc(4096).unwrap();
        s.free(a).unwrap();
        assert!(matches!(s.free(a), Err(HostMemError::UnknownLoc(_))));
    }

    #[test]
    fn budget_not_a_slab_multiple_is_fully_usable() {
        // 1.5 slabs of budget: the second slab must be carved at the
        // remainder, not skipped.
        let mut s = HostStore::new(3 << 19, false); // 1.5 MiB
        s.slab_bytes = 1 << 20;
        let a = s.alloc(1 << 20).unwrap();
        let b = s.alloc(1 << 19).unwrap(); // fits only in a remainder slab
        s.free(a).unwrap();
        s.free(b).unwrap();
    }

    #[test]
    fn resolve_pointers_are_distinct_and_stable() {
        let mut s = HostStore::new(4 << 20, false);
        let a = s.alloc(1 << 20).unwrap();
        let b = s.alloc(1 << 20).unwrap();
        let (pa, _) = s.resolve(a).unwrap();
        let (pb, _) = s.resolve(b).unwrap();
        assert_ne!(pa, pb);
        // write through a, read back through a fresh resolve - same memory
        unsafe { std::ptr::write(pa, 0xAB) };
        let (pa2, _) = s.resolve(a).unwrap();
        assert_eq!(pa, pa2);
        assert_eq!(unsafe { std::ptr::read(pa2) }, 0xAB);
    }
}
