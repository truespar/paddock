//! host<->device transfer, events, allocation.

use super::error::*;
use super::types::narrow_to_f16;
use super::*;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut, DeviceRepr};
use half::f16;

impl GpuExecutor {
    /// Device-to-device copy of `len` elements from src[offset..] into dst.
    pub fn copy_slice<T: DeviceRepr>(
        &self,
        src: &CudaSlice<T>,
        offset: usize,
        len: usize,
        dst: &mut CudaSlice<T>,
    ) -> Result<(), GpuError> {
        let view = src
            .try_slice(offset..offset + len)
            .ok_or_else(|| oob("copy_slice: range out of bounds"))?;
        self.stream.memcpy_dtod(&view, dst).map_err(drv)
    }

    /// Device-to-device copy of `len` elements from `src[src_off..]` to
    /// `dst[dst_off..]` (both sub-ranges). Used to shuttle cached prefix KV in
    /// and out of a session's slot.
    pub fn copy_region<T: DeviceRepr>(
        &self,
        src: &CudaSlice<T>,
        src_off: usize,
        dst: &mut CudaSlice<T>,
        dst_off: usize,
        len: usize,
    ) -> Result<(), GpuError> {
        let view = src
            .try_slice(src_off..src_off + len)
            .ok_or_else(|| oob("copy_region: src range out of bounds"))?;
        let mut dview = dst
            .try_slice_mut(dst_off..dst_off + len)
            .ok_or_else(|| oob("copy_region: dst range out of bounds"))?;
        self.stream.memcpy_dtod(&view, &mut dview).map_err(drv)
    }

    pub fn to_host(&self, buf: &CudaSlice<f32>) -> Result<Vec<f32>, GpuError> {
        self.stream.synchronize().map_err(drv)?;
        self.stream.clone_dtoh(buf).map_err(drv)
    }

    /// Device -> host copy of `n` bytes (weight-prep self-checks read bf16
    /// planes, which live as byte slices).
    pub fn to_host_u8_len(&self, buf: &CudaSlice<u8>, n: usize) -> Result<Vec<u8>, GpuError> {
        self.stream.synchronize().map_err(drv)?;
        self.stream.clone_dtoh(&buf.slice(0..n)).map_err(drv)
    }

    /// Sync then copy the first `n` elements of a device buffer to host -
    /// for scratch planes sized to a high-water mark, where copying the whole
    /// allocation would move megabytes of stale tail per call.
    pub fn to_host_len(&self, buf: &CudaSlice<f32>, n: usize) -> Result<Vec<f32>, GpuError> {
        let view = buf
            .try_slice(0..n)
            .ok_or_else(|| oob("to_host_len: n out of range"))?;
        self.stream.synchronize().map_err(drv)?;
        self.stream.clone_dtoh(&view).map_err(drv)
    }

    /// Sync then copy `n` bytes at `offset` from a device byte buffer - for
    /// pulling a couple of weight rows out of a resident tensor without
    /// moving the whole plane.
    pub fn to_host_range_u8(
        &self,
        buf: &CudaSlice<u8>,
        offset: usize,
        n: usize,
    ) -> Result<Vec<u8>, GpuError> {
        let view = buf
            .try_slice(offset..offset + n)
            .ok_or_else(|| oob("to_host_range_u8: range out of bounds"))?;
        self.stream.synchronize().map_err(drv)?;
        self.stream.clone_dtoh(&view).map_err(drv)
    }

    /// Sync then copy a u32 device buffer to host (decode-loop token readback).
    /// Sync then copy the first `n` u32 of a device buffer to host - a run
    /// list read out of a plane sized for the worst case.
    pub fn to_host_u32_len(&self, buf: &CudaSlice<u32>, n: usize) -> Result<Vec<u32>, GpuError> {
        let view = buf
            .try_slice(0..n)
            .ok_or_else(|| oob("to_host_u32_len: n out of range"))?;
        self.stream.synchronize().map_err(drv)?;
        self.stream.clone_dtoh(&view).map_err(drv)
    }

    pub fn to_host_u32(&self, buf: &CudaSlice<u32>) -> Result<Vec<u32>, GpuError> {
        self.stream.synchronize().map_err(drv)?;
        self.stream.clone_dtoh(buf).map_err(drv)
    }

    /// Block the host until all prior work on this stream completes. For timing
    /// probes that need a barrier between phases.
    pub fn synchronize(&self) -> Result<(), GpuError> {
        self.stream.synchronize().map_err(drv)
    }

    /// Record an event on the compute stream marking everything enqueued so far.
    pub fn record_event(&self) -> Result<cudarc::driver::CudaEvent, GpuError> {
        self.stream.record_event(None).map_err(drv)
    }

    /// Make the COMPUTE stream wait on `ev` device-side (no host block) -
    /// the join half of cross-lane overlap (the prefill chunk walk runs
    /// decode-row attention on a forked lane and joins here).
    pub fn wait_event(&self, ev: &cudarc::driver::CudaEvent) -> Result<(), GpuError> {
        self.stream.wait(ev).map_err(drv)
    }

    /// Non-blocking: has this event fired (all work recorded before it done)?
    pub fn event_done(&self, ev: &cudarc::driver::CudaEvent) -> bool {
        // SAFETY: valid live event handle; cuEventQuery has no side effects
        unsafe { cudarc::driver::result::event::query(ev.cu_event()).is_ok() }
    }

    /// Read the first `n` elements of a device buffer after `ev` fires,
    /// via the side stream - later work already enqueued on the compute
    /// stream keeps running; only the event's producers are waited on.
    pub fn to_host_len_after(
        &self,
        ev: &cudarc::driver::CudaEvent,
        buf: &CudaSlice<f32>,
        n: usize,
    ) -> Result<Vec<f32>, GpuError> {
        let view = buf
            .try_slice(0..n)
            .ok_or_else(|| oob("to_host_len_after: n out of range"))?;
        self.copy_stream.wait(ev).map_err(drv)?;
        self.copy_stream.synchronize().map_err(drv)?;
        self.copy_stream.clone_dtoh(&view).map_err(drv)
    }

    /// Read `n` u32s at `off` after `ev` fires, via the side stream - the
    /// compute stream keeps running whatever was enqueued after the event
    /// (the pipelined decode reads tick N's ids while tick N+1 executes).
    pub fn to_host_u32_after(
        &self,
        ev: &cudarc::driver::CudaEvent,
        buf: &CudaSlice<u32>,
        off: usize,
        n: usize,
    ) -> Result<Vec<u32>, GpuError> {
        let view = buf
            .try_slice(off..off + n)
            .ok_or_else(|| oob("to_host_u32_after: range out of bounds"))?;
        self.copy_stream.wait(ev).map_err(drv)?;
        self.copy_stream.synchronize().map_err(drv)?;
        self.copy_stream.clone_dtoh(&view).map_err(drv)
    }

    pub fn to_device(&self, host: &[f32]) -> Result<CudaSlice<f32>, GpuError> {
        self.stream.clone_htod(host).map_err(drv)
    }

    /// Upload a u32 host slice (position/index buffers).
    pub fn to_device_u32(&self, host: &[u32]) -> Result<CudaSlice<u32>, GpuError> {
        self.stream.clone_htod(host).map_err(drv)
    }

    /// Upload an i32 host slice (trtllm repack index arrays).
    pub fn to_device_i32(&self, host: &[i32]) -> Result<CudaSlice<i32>, GpuError> {
        self.stream.clone_htod(host).map_err(drv)
    }

    /// Upload a u16 host slice (fp16-as-bits planes, e.g. the RS q-store).
    pub fn to_device_u16(&self, host: &[u16]) -> Result<CudaSlice<u16>, GpuError> {
        self.stream.clone_htod(host).map_err(drv)
    }

    /// Read back a u16 device buffer (fp16-as-bits planes).
    /// Read back an f16 buffer's first `n` values. Kernel gates compare f16
    /// landings bit for bit, and `f16` is not `u16` to the type system even
    /// though it is to the device.
    pub fn to_host_f16_len(&self, buf: &CudaSlice<f16>, n: usize) -> Result<Vec<f16>, GpuError> {
        self.stream.synchronize().map_err(drv)?;
        let view = buf
            .try_slice(0..n)
            .ok_or_else(|| oob("to_host_f16_len: range past the buffer"))?;
        self.stream.clone_dtoh(&view).map_err(drv)
    }

    /// Read the first `n` f16 values out of a BYTE plane. The KV caches are
    /// `CudaSlice<u8>` sized by the element width (one allocation that means
    /// f16 or fp8 depending on the dtype flag), so a gate that wants to check
    /// what an f16-width cache actually holds has to reinterpret.
    pub fn to_host_f16_from_u8(&self, buf: &CudaSlice<u8>, n: usize) -> Result<Vec<f16>, GpuError> {
        let bytes = self.to_host_range_u8(buf, 0, n * 2)?;
        Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f16::from_bits(u16::from_le_bytes(*c)))
            .collect())
    }

    pub fn to_host_u16(&self, buf: &CudaSlice<u16>) -> Result<Vec<u16>, GpuError> {
        self.stream.clone_dtoh(buf).map_err(drv)
    }

    /// Upload a bf16 host slice (the QSA compressed-key caches are bf16).
    pub fn to_device_bf16(&self, host: &[half::bf16]) -> Result<CudaSlice<half::bf16>, GpuError> {
        self.stream.clone_htod(host).map_err(drv)
    }

    /// Read back a bf16 device buffer.
    pub fn to_host_bf16(&self, buf: &CudaSlice<half::bf16>) -> Result<Vec<half::bf16>, GpuError> {
        self.stream.synchronize().map_err(drv)?;
        self.stream.clone_dtoh(buf).map_err(drv)
    }

    /// Upload into the FRONT of an existing device buffer (no allocation).
    /// The serving hot path reuses persistent upload buffers: per-encode
    /// transient allocs are stream-ordered pool operations whose cross-
    /// stream reuse ordering SERIALIZES the dual-lane streams.
    pub fn upload_u32(&self, host: &[u32], dst: &mut CudaSlice<u32>) -> Result<(), GpuError> {
        let mut view = dst
            .try_slice_mut(0..host.len())
            .ok_or_else(|| oob("upload_u32: host longer than buffer"))?;
        self.stream.memcpy_htod(host, &mut view).map_err(drv)
    }

    /// u8-at-offset twin: fills `dst[off..off + host.len()]`. Lets a loader
    /// stream a plane bigger than any single host buffer it is willing to
    /// allocate (see the PartitionAlloc note at the qwen35 head build).
    pub fn upload_u8_at(
        &self,
        host: &[u8],
        dst: &mut CudaSlice<u8>,
        off: usize,
    ) -> Result<(), GpuError> {
        let mut view = dst
            .try_slice_mut(off..off + host.len())
            .ok_or_else(|| oob("upload_u8_at: range out of bounds"))?;
        self.stream.memcpy_htod(host, &mut view).map_err(drv)
    }

    /// f32 twin of [`Self::upload_u32`] - feeds a per-request host plane into
    /// the FRONT of a resident buffer (whisper's mel window lands this way,
    /// so the pad row past it keeps the zeros `alloc_zeros` gave it).
    ///
    /// Pageable-source cost: the driver stages the copy and blocks the host
    /// for it (~1.4 ms measured for a 10 MB tower chunk). A pinned
    /// bounce buffer was tried there and FALSIFIED - our memcpy into the slab
    /// costs the same as the driver's staging. Big recurring feeds should
    /// shrink the bytes instead (the OCR tower now ships u8 pixels).
    pub fn upload_f32(&self, host: &[f32], dst: &mut CudaSlice<f32>) -> Result<(), GpuError> {
        let mut view = dst
            .try_slice_mut(0..host.len())
            .ok_or_else(|| oob("upload_f32: host longer than buffer"))?;
        self.stream.memcpy_htod(host, &mut view).map_err(drv)
    }

    /// [`Self::upload_f32`] landing at an element offset - the batched
    /// whisper admission stages each audio's mel at its own base.
    pub fn upload_f32_at(
        &self,
        host: &[f32],
        dst: &mut CudaSlice<f32>,
        off: usize,
    ) -> Result<(), GpuError> {
        let mut view = dst
            .try_slice_mut(off..off + host.len())
            .ok_or_else(|| oob("upload_f32_at: range outside buffer"))?;
        self.stream.memcpy_htod(host, &mut view).map_err(drv)
    }

    /// u8 twin of [`Self::upload_f32`] (the OCR tower's pixel views).
    pub fn upload_u8(&self, host: &[u8], dst: &mut CudaSlice<u8>) -> Result<(), GpuError> {
        let mut view = dst
            .try_slice_mut(0..host.len())
            .ok_or_else(|| oob("upload_u8: host longer than buffer"))?;
        self.stream.memcpy_htod(host, &mut view).map_err(drv)
    }

    /// Zero `n` elements starting at `off` - re-arms the always-zero pad rows
    /// (a window-partition pad source, a conv gather's zero tap) inside a
    /// persistent slab that `alloc_zeros` used to guarantee back when the
    /// buffer was allocated per call.
    pub fn zero_region(
        &self,
        buf: &mut CudaSlice<f32>,
        off: usize,
        n: usize,
    ) -> Result<(), GpuError> {
        let mut view = buf
            .try_slice_mut(off..off + n)
            .ok_or_else(|| oob("zero_region: range past the buffer"))?;
        self.stream.memset_zeros(&mut view).map_err(drv)
    }

    /// f16 twin of [`Self::copy_region`] (the f16 SSM arena's rollback copy).
    pub fn copy_region_f16(
        &self,
        src: &CudaSlice<f16>,
        src_off: usize,
        dst: &mut CudaSlice<f16>,
        dst_off: usize,
        n: usize,
    ) -> Result<(), GpuError> {
        let sv = src
            .try_slice(src_off..src_off + n)
            .ok_or_else(|| oob("copy_region_f16: src range past the buffer"))?;
        let mut dv = dst
            .try_slice_mut(dst_off..dst_off + n)
            .ok_or_else(|| oob("copy_region_f16: dst range past the buffer"))?;
        self.stream.memcpy_dtod(&sv, &mut dv).map_err(drv)
    }

    /// f16 twin of [`Self::zero_region`] - the f16 SSM-state arena's slot
    /// admission zeroes through this (a fresh sequence must start from S = 0).
    pub fn zero_region_f16(
        &self,
        buf: &mut CudaSlice<f16>,
        off: usize,
        n: usize,
    ) -> Result<(), GpuError> {
        let mut view = buf
            .try_slice_mut(off..off + n)
            .ok_or_else(|| oob("zero_region_f16: range past the buffer"))?;
        self.stream.memset_zeros(&mut view).map_err(drv)
    }

    /// u64 twin of [`Self::upload_u32`] (batched-copy descriptor uploads).
    pub fn upload_u64(&self, host: &[u64], dst: &mut CudaSlice<u64>) -> Result<(), GpuError> {
        let mut view = dst
            .try_slice_mut(0..host.len())
            .ok_or_else(|| oob("upload_u64: host longer than buffer"))?;
        self.stream.memcpy_htod(host, &mut view).map_err(drv)
    }

    pub fn alloc_u64(&self, len: usize) -> Result<CudaSlice<u64>, GpuError> {
        // SAFETY: contents written before any read (upload_u64)
        unsafe { self.stream.alloc(len) }.map_err(drv)
    }

    /// Device-resident table of the given buffers' base addresses, for
    /// kernels that walk a set of per-layer planes in one launch (
    /// layer-batched cross-K/V store). Valid as long as every source buffer
    /// stays alive - the caller owns that pairing, same as any recorded
    /// graph node.
    pub fn pointer_table(&self, bufs: &[CudaSlice<u8>]) -> Result<CudaSlice<u64>, GpuError> {
        let ptrs: Vec<u64> = bufs
            .iter()
            .map(|b| {
                let (p, _g) = b.device_ptr(&self.stream);
                p
            })
            .collect();
        let mut t = self.alloc_u64(ptrs.len())?;
        self.upload_u64(&ptrs, &mut t)?;
        Ok(t)
    }

    pub fn to_device_u8(&self, host: &[u8]) -> Result<CudaSlice<u8>, GpuError> {
        self.stream.clone_htod(host).map_err(drv)
    }

    pub fn to_host_i8(&self, buf: &CudaSlice<i8>) -> Result<Vec<i8>, GpuError> {
        self.stream.synchronize().map_err(drv)?;
        self.stream.clone_dtoh(buf).map_err(drv)
    }

    pub fn alloc(&self, n: usize) -> Result<CudaSlice<f32>, GpuError> {
        self.stream.alloc_zeros(n).map_err(drv)
    }

    /// Disable cudarc's per-buffer CUDA-event tracking for this context. The engine
    /// drives everything on a single stream, so ops are already stream-ordered and
    /// the cross-stream event bookkeeping is pure overhead - and, critically, its
    /// `cuStreamWaitEvent` on pre-capture events makes CUDA-graph capture illegal.
    /// Call once, before allocating any buffers that a graph will touch.
    pub fn disable_event_tracking(&self) {
        // SAFETY: single-stream executor - no concurrent cross-stream buffer use.
        unsafe { self.ctx.disable_event_tracking() }
    }

    /// Allocate a zeroed f16 device buffer - the KV cache stores post-rope K/V as
    /// f16 to halve the decode-attention read bandwidth.
    pub fn alloc_f16(&self, n: usize) -> Result<CudaSlice<f16>, GpuError> {
        self.stream.alloc_zeros(n).map_err(drv)
    }

    /// bf16 twin of `alloc_f16` (TGV activation staging).
    pub fn stream_alloc_bf16(&self, n: usize) -> Result<CudaSlice<half::bf16>, GpuError> {
        self.stream.alloc_zeros(n).map_err(drv)
    }

    /// Upload host f32 values as an f16 device plane, checked for overflow.
    ///
    /// The counterpart to [`Self::upload_f16`] for weights a loader BUILDS
    /// rather than reads straight from the file - a conv permuted into im2row
    /// order, or one third of a fused QKV tensor. Those never exist as a named
    /// GGUF tensor, so they cannot go through the map-based path, but they want
    /// exactly the same storage class. `what` names the plane in the overflow
    /// error, since the caller's slice has no name of its own.
    /// Upload an f16 slice a loader has already narrowed. `to_device_f16`
    /// widens through f32 first, which is four minutes of load on a 3.2G-element
    /// dense set; a caller that converts bit-level hands the result straight in.
    pub fn f16_to_device(&self, host: &[f16]) -> Result<CudaSlice<f16>, GpuError> {
        self.stream.clone_htod(host).map_err(drv)
    }

    pub fn to_device_f16(&self, host: &[f32], what: &str) -> Result<CudaSlice<f16>, GpuError> {
        let h = narrow_to_f16(host, what)?;
        self.stream.clone_htod(&h).map_err(drv)
    }

    /// Convert `n` f32 elements into an f16 destination (`dst[i] = f16(src[i])`).
    /// Used to write one post-rope K/V row into the fp16 cache on the single-stream
    /// path (the batched path converts inside kv_append_batch).
    /// f32 -> bf16 twin (slot 548): stages activations for the TGV GEMM.
    pub fn convert_f32_bf16(
        &self,
        src: &CudaSlice<f32>,
        dst: &mut CudaSlice<half::bf16>,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .convert_f32_bf16
            .ok_or(GpuError::MissingOp("convert_f32_bf16"))?;
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (dp, _g2) = dst.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; both buffers hold >= n elements
        check(unsafe { f(sp as *const _, dp as *mut _, n as u64, self.stream_ptr()) })
    }

    pub fn convert_f32_f16(
        &self,
        src: &CudaSlice<f32>,
        dst: &mut CudaSlice<f16>,
        n: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .convert_f32_f16
            .ok_or(GpuError::MissingOp("convert_f32_f16"))?;
        let (sp, _g1) = src.device_ptr(&self.stream);
        let (dp, _g2) = dst.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; both buffers hold >= n elements
        check(unsafe { f(sp as *const _, dp as *mut _, n as u64, self.stream_ptr()) })
    }

    /// OCR tower patch stem: u8 interleaved-RGB views to
    /// normalized f16 patch rows in the SAM conv stem's im2row order, one
    /// gather - bit-identical to host normalize + im2row + the f32->f16
    /// convert, at a quarter of the upload bytes.
    #[allow(clippy::too_many_arguments)]
    pub fn ocr_patches_u8(
        &self,
        pixels: &CudaSlice<u8>,
        out: &mut CudaSlice<f16>,
        mean: [f32; 3],
        std: [f32; 3],
        views: usize,
        px: usize,
        patch: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .ocr_patches_u8
            .ok_or(GpuError::MissingOp("ocr_patches_u8"))?;
        let n = views * px * px * 3;
        if pixels.len() < n || out.len() < n {
            return Err(oob("ocr_patches_u8: buffers under the view geometry"));
        }
        let (sp, _g1) = pixels.device_ptr(&self.stream);
        let (dp, _g2) = out.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; both buffers hold >= views*px*px*3 elements
        check(unsafe {
            f(
                sp as *const _,
                dp as *mut _,
                mean[0],
                mean[1],
                mean[2],
                std[0],
                std[1],
                std[2],
                views as u32,
                px as u32,
                patch as u32,
                self.stream_ptr(),
            )
        })
    }
}
