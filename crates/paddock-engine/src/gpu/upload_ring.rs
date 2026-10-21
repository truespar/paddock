//! Pinned staging for large one-shot host->device uploads - a page's RGB
//! canvas, megabytes a request - see [`GpuExecutor::to_device_u8_staged`].

use super::*;
use cudarc::driver::{CudaEvent, CudaSlice, DevicePtrMut};

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
