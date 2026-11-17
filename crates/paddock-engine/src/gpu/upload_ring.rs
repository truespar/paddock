//! Pinned staging for host->device uploads: large one-shot ones - a page's
//! RGB canvas, megabytes a request - see [`GpuExecutor::to_device_u8_staged`],
//! and a pass's small tables - see [`GpuExecutor::htod`].

use super::*;
use cudarc::driver::{CudaEvent, CudaSlice, DevicePtrMut, DeviceRepr, DeviceSlice};
use std::collections::VecDeque;

/// Uploads below this stay pageable: the driver stages one whole and returns.
const STAGED_MIN: usize = 1 << 20;
/// Staging buffers per executor; with all of them in flight an upload waits
/// for the first one's DMA.
const RING: usize = 4;

/// One cached-pinned staging buffer and the event after its last DMA.
pub(super) struct Stage {
    ptr: *mut u8,
    len: usize,
    done: Option<CudaEvent>,
}

// SAFETY: the pointer is a driver allocation this value alone owns, and every
// access goes through the executor's mutex.
unsafe impl Send for Stage {}

impl Stage {
    const EMPTY: Self = Self {
        ptr: std::ptr::null_mut(),
        len: 0,
        done: None,
    };

    /// `len` bytes of cached pinned memory - PORTABLE, never write-combined
    /// (see `kv_tier/host.rs`). `None` when the driver refuses.
    fn new(len: usize) -> Option<Self> {
        // SAFETY: len > 0; the pointer is owned here until Drop.
        let ptr = unsafe {
            cudarc::driver::result::malloc_host(len, cudarc::driver::sys::CU_MEMHOSTALLOC_PORTABLE)
        }
        .ok()?;
        Some(Self {
            ptr: ptr.cast(),
            len,
            done: None,
        })
    }

    fn idle(&self) -> bool {
        self.done.as_ref().is_none_or(CudaEvent::is_complete)
    }
}

impl Drop for Stage {
    fn drop(&mut self) {
        if let Some(ev) = self.done.take() {
            let _ = ev.synchronize(); // never free pages a DMA still reads
        }
        if !self.ptr.is_null() {
            // SAFETY: ptr came from malloc_host and is freed exactly once.
            let _ = unsafe { cudarc::driver::result::free_host(self.ptr.cast()) };
        }
    }
}

impl GpuExecutor {
    /// The concatenation of `parts` in one fresh device buffer, through a
    /// small ring of pinned staging buffers. A pageable copy this size cannot
    /// be staged whole, so the driver holds the calling thread until the
    /// stream reaches it - behind every span pass already queued: ~11 ms a
    /// page on the B200 recover load (1.1 s of each 20 s, engine thread),
    /// against 0.3 ms of DMA. Here the host copies into pinned pages and
    /// returns; the DMA runs in stream order, and a buffer is reused once the
    /// event after its DMA has fired. Small uploads, and a refused pinned
    /// allocation, take the pageable path.
    pub fn to_device_u8_staged(&self, parts: &[&[u8]]) -> Result<CudaSlice<u8>, GpuError> {
        let n: usize = parts.iter().map(|p| p.len()).sum();
        let pageable = || match parts {
            [one] => self.to_device_u8(one),
            _ => self.to_device_u8(&parts.concat()),
        };
        if n < STAGED_MIN {
            return pageable();
        }
        let mut ring = self
            .upload_ring
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // an idle buffer that fits, else any idle one (regrown), else a new
        // one, else wait out the first
        let i = match ring
            .iter()
            .position(|s| s.len >= n && s.idle())
            .or_else(|| ring.iter().position(Stage::idle))
        {
            Some(i) => i,
            None if ring.len() < RING => {
                ring.push(Stage::EMPTY);
                ring.len() - 1
            }
            None => 0,
        };
        let s = &mut ring[i];
        if let Some(ev) = s.done.take() {
            ev.synchronize().map_err(drv)?;
        }
        if s.len < n {
            *s = Stage::EMPTY; // free the small one before pinning the big one
            match Stage::new(n.max(8 << 20).next_multiple_of(1 << 20)) {
                Some(big) => *s = big,
                None => {
                    drop(ring);
                    return pageable();
                }
            }
        }
        let mut off = 0;
        for p in parts {
            // SAFETY: s.ptr holds s.len >= n bytes and no DMA reads them (done
            // was waited above); the parts are host memory disjoint from it.
            unsafe { std::ptr::copy_nonoverlapping(p.as_ptr(), s.ptr.add(off), p.len()) };
            off += p.len();
        }
        // SAFETY: uninitialized device bytes, fully written by the copy below.
        let mut dst = unsafe { self.stream.alloc::<u8>(n) }.map_err(drv)?;
        {
            let (dptr, _g) = dst.device_ptr_mut(&self.stream);
            // SAFETY: n pinned bytes at s.ptr, kept from reuse until `done`
            // fires - after this DMA, in stream order.
            unsafe {
                cudarc::driver::result::memcpy_htod_async(
                    dptr,
                    std::slice::from_raw_parts(s.ptr, n),
                    self.stream.cu_stream(),
                )
            }
            .map_err(drv)?;
        }
        s.done = Some(self.stream.record_event(None).map_err(drv)?);
        Ok(dst)
    }
}

/// The small-upload ring's pinned bytes, and the largest upload it stages.
const SMALL_RING: usize = 16 << 20;
const SMALL_MAX: usize = 2 << 20;
const SMALL_ALIGN: usize = 256;

/// A bump ring of cached pinned memory for a pass's uploads (block tables,
/// run tables, copy descriptors, token / position rows). Each upload takes
/// the next span, its DMA is queued in stream order, and the span comes back
/// once the event recorded after the DMA has fired - in practice long before
/// the ring wraps onto it.
pub(super) struct SmallRing {
    ptr: *mut u8,
    head: usize,
    /// in-flight spans in allocation order: (start, end, event after the DMA)
    busy: VecDeque<(usize, usize, CudaEvent)>,
    /// fired events, re-recorded instead of created (an event create took
    /// up to 13 ms on a busy GB10)
    spare: Vec<CudaEvent>,
}

// SAFETY: the pointer is a driver allocation this value alone owns, and every
// access goes through the executor's mutex.
unsafe impl Send for SmallRing {}

impl SmallRing {
    fn new() -> Option<Self> {
        // SAFETY: the pointer is owned here until Drop.
        let ptr = unsafe {
            cudarc::driver::result::malloc_host(
                SMALL_RING,
                cudarc::driver::sys::CU_MEMHOSTALLOC_PORTABLE,
            )
        }
        .ok()?;
        Some(Self {
            ptr: ptr.cast(),
            head: 0,
            busy: VecDeque::new(),
            spare: Vec::new(),
        })
    }

    /// A span of `n` bytes no DMA still reads: fired spans retire from the
    /// front, and a span the new one would overlap is waited out.
    fn reserve(&mut self, n: usize) -> Result<usize, GpuError> {
        let n = n.next_multiple_of(SMALL_ALIGN);
        while let Some((_, _, ev)) = self.busy.front() {
            if !ev.is_complete() {
                break;
            }
            let (_, _, ev) = self.busy.pop_front().expect("front");
            self.spare.push(ev);
        }
        if self.head + n > SMALL_RING {
            self.head = 0;
        }
        let (a, b) = (self.head, self.head + n);
        // any span the new one overlaps - not only the front: after a wrap
        // the front can be a span from the lap before, past the old head,
        // while this lap's spans at the ring's start are still in flight.
        // The events fire in stream order, so waiting on the newest
        // overlapping span retires every older one with it.
        while let Some(i) = self.busy.iter().rposition(|&(s, e, _)| s < b && a < e) {
            self.busy[i].2.synchronize().map_err(drv)?;
            for _ in 0..=i {
                let (_, _, ev) = self.busy.pop_front().expect("front");
                self.spare.push(ev);
            }
        }
        self.head = b;
        Ok(a)
    }
}

impl Drop for SmallRing {
    fn drop(&mut self) {
        for (_, _, ev) in self.busy.drain(..) {
            let _ = ev.synchronize(); // never free pages a DMA still reads
        }
        if !self.ptr.is_null() {
            // SAFETY: ptr came from malloc_host and is freed exactly once.
            let _ = unsafe { cudarc::driver::result::free_host(self.ptr.cast()) };
        }
    }
}

impl GpuExecutor {
    /// `src` into `dst` (its first `src.len()` elements) on the compute
    /// stream, staged through the pinned small-upload ring: the call copies
    /// into pinned pages and returns, the DMA runs in stream order. A
    /// pageable copy is staged by the driver, and on GB10 that held the
    /// engine thread ~1 ms for a 512 KB block table (14 us of DMA) and, with
    /// a second pageable copy queued behind it, until the stream reached it;
    /// the GPU then idled while the thread caught up (5-9 ms between a
    /// Transcribe round's passes). Large uploads and a refused pinned
    /// allocation take the pageable path. Never inside a graph capture.
    pub fn htod<T: DeviceRepr, D: DevicePtrMut<T> + DeviceSlice<T>>(
        &self,
        src: &[T],
        dst: &mut D,
    ) -> Result<(), GpuError> {
        let n = std::mem::size_of_val(src);
        if src.is_empty() {
            return Ok(());
        }
        if src.len() > dst.len() {
            return Err(GpuError::Driver(format!(
                "htod: {} elements into {}",
                src.len(),
                dst.len()
            )));
        }
        let pageable = |dst: &mut D| {
            let (dptr, _g) = dst.device_ptr_mut(&self.stream);
            // SAFETY: dst holds src.len() elements (checked above); the
            // driver stages the pageable bytes before returning.
            unsafe { cudarc::driver::result::memcpy_htod_async(dptr, src, self.stream.cu_stream()) }
                .map_err(drv)
        };
        if n > SMALL_MAX || paddock_models::dev_var_os!("PADDOCK_NO_HTOD_RING").is_some() {
            return pageable(dst);
        }
        let mut ring = self
            .small_ring
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if ring.is_none() {
            *ring = SmallRing::new();
        }
        let Some(r) = ring.as_mut() else {
            drop(ring);
            return pageable(dst);
        };
        let off = r.reserve(n)?;
        // SAFETY: [off, off + n) lies in the ring and no DMA reads it
        // (reserve waited out any span it overlaps); src is host memory
        // disjoint from the pinned pages.
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr().cast::<u8>(), r.ptr.add(off), n) };
        {
            let (dptr, _g) = dst.device_ptr_mut(&self.stream);
            // SAFETY: n pinned bytes at ptr + off, kept from reuse until the
            // event below fires - after this DMA, in stream order.
            unsafe {
                cudarc::driver::result::memcpy_htod_async(
                    dptr,
                    std::slice::from_raw_parts(r.ptr.add(off), n),
                    self.stream.cu_stream(),
                )
            }
            .map_err(drv)?;
        }
        let ev = match r.spare.pop() {
            Some(ev) => {
                ev.record(&self.stream).map_err(drv)?;
                ev
            }
            None => self.stream.record_event(None).map_err(drv)?,
        };
        r.busy
            .push_back((off, off + n.next_multiple_of(SMALL_ALIGN), ev));
        Ok(())
    }
}
