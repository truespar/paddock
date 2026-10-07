//! The recurrent SSM state arena and its numeric class (scan rung).
//!
//! State is STORED in one of two widths and COMPUTED in f32 either way - the
//! same shape `--kv-cache-dtype` has for the KV cache. f32 is the checkpoint's
//! own declaration (`mamba_ssm_cache_dtype: float32`) and stays the default;
//! f16 halves that traffic and is worth a few percent at c32, where the
//! state is 32 slots x 2 MiB per layer streamed in and out every step.
//!
//! The dispatch lives here rather than at the ~8 call sites in the layer
//! walk. That is deliberate: a match at every site is how one arm silently
//! keeps the old width through a refactor, and the state is the one plane
//! where that corrupts a sequence instead of erroring.
//!
//! The prefix-cache checkpoint pool, its staging blobs and the radix
//! accounting keep their f32 `[state | win]` layout untouched - that is the
//! restore path, and re-laying it out to chase a byte is a bad trade. The
//! f16 arm converts at the boundary instead, which round-trips bit-for-bit
//! because the blob only ever receives values that came from f16.

use cudarc::driver::{CudaSlice, DevicePtr};
use half::f16;

use crate::gpu::{GpuError, GpuExecutor};

/// Storage width for the recurrent state. Arithmetic is f32 in both arms.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SsmDtype {
    F32,
    F16,
}

impl SsmDtype {
    pub fn bytes(self) -> usize {
        match self {
            SsmDtype::F32 => 4,
            SsmDtype::F16 => 2,
        }
    }
}

/// A per-layer recurrent-state arena in one of the two classes.
pub enum SsmArena {
    F32(CudaSlice<f32>),
    F16(CudaSlice<f16>),
}

impl SsmArena {
    pub fn alloc(e: &GpuExecutor, elems: usize, dt: SsmDtype) -> Result<Self, GpuError> {
        // alloc() zeroes; a fresh sequence must start from S = 0
        Ok(match dt {
            SsmDtype::F32 => SsmArena::F32(e.alloc(elems)?),
            SsmDtype::F16 => SsmArena::F16(e.alloc_f16(elems)?),
        })
    }

    /// Device address of element `off` and the element size in bytes - a
    /// batched byte copy's descriptor end (the verify rollback). Same stream
    /// as every other user of the arena, so no cross-stream guard is held.
    pub fn addr(&self, e: &GpuExecutor, off: usize) -> (u64, usize) {
        match self {
            SsmArena::F32(s) => {
                let (p, _g) = s.device_ptr(&e.stream);
                (p + (off * 4) as u64, 4)
            }
            SsmArena::F16(s) => {
                let (p, _g) = s.device_ptr(&e.stream);
                (p + (off * 2) as u64, 2)
            }
        }
    }

    pub fn zero_region(&mut self, e: &GpuExecutor, off: usize, n: usize) -> Result<(), GpuError> {
        match self {
            SsmArena::F32(s) => e.zero_region(s, off, n),
            SsmArena::F16(s) => e.zero_region_f16(s, off, n),
        }
    }

    /// Batched single-token decode step over every active slot.
    #[allow(clippy::too_many_arguments)]
    pub fn scan_step_batch(
        &mut self,
        e: &GpuExecutor,
        xbc: &CudaSlice<f32>,
        dt: &CudaSlice<f32>,
        dt_off: usize,
        dt_stride: usize,
        slots: &CudaSlice<u32>,
        a: &CudaSlice<f32>,
        d: &CudaSlice<f32>,
        dt_bias: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        batch: usize,
        n_heads: usize,
        head_dim: usize,
        d_state: usize,
        n_groups: usize,
    ) -> Result<(), GpuError> {
        match self {
            SsmArena::F32(s) => e.mamba2_scan_step_batch(
                s, xbc, dt, dt_off, dt_stride, slots, a, d, dt_bias, y, batch, n_heads, head_dim,
                d_state, n_groups,
            ),
            SsmArena::F16(s) => e.mamba2_scan_step_batch_f16(
                s, xbc, dt, dt_off, dt_stride, slots, a, d, dt_bias, y, batch, n_heads, head_dim,
                d_state, n_groups,
            ),
        }
    }

    /// Span walk at a slot offset (prefill / chunk runs).
    #[allow(clippy::too_many_arguments)]
    pub fn scan_seq_at(
        &mut self,
        e: &GpuExecutor,
        state_off: usize,
        xbc: &CudaSlice<f32>,
        xbc_off: usize,
        dt: &CudaSlice<f32>,
        dt_off: usize,
        dt_stride: usize,
        a: &CudaSlice<f32>,
        d: &CudaSlice<f32>,
        dt_bias: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        y_off: usize,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
        d_state: usize,
        n_groups: usize,
    ) -> Result<(), GpuError> {
        match self {
            SsmArena::F32(s) => e.mamba2_scan_seq_at(
                s, state_off, xbc, xbc_off, dt, dt_off, dt_stride, a, d, dt_bias, y, y_off,
                n_tokens, n_heads, head_dim, d_state, n_groups,
            ),
            SsmArena::F16(s) => e.mamba2_scan_seq_at_f16(
                s, state_off, xbc, xbc_off, dt, dt_off, dt_stride, a, d, dt_bias, y, y_off,
                n_tokens, n_heads, head_dim, d_state, n_groups,
            ),
        }
    }

    /// Span walk + per-row snapshots for spec rollback. `snap` must be the
    /// same class as the live arena: a partial accept rolls back by copying a
    /// snap row over the state, so a mixed pair would re-round every rollback.
    #[allow(clippy::too_many_arguments)]
    pub fn scan_seq_snap_at(
        &mut self,
        e: &GpuExecutor,
        state_off: usize,
        xbc: &CudaSlice<f32>,
        xbc_off: usize,
        dt: &CudaSlice<f32>,
        dt_off: usize,
        dt_stride: usize,
        a: &CudaSlice<f32>,
        d: &CudaSlice<f32>,
        dt_bias: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        y_off: usize,
        snap: &mut SsmArena,
        snap_off: usize,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
        d_state: usize,
        n_groups: usize,
    ) -> Result<(), GpuError> {
        match (self, snap) {
            (SsmArena::F32(s), SsmArena::F32(sn)) => e.mamba2_scan_seq_snap_at(
                s, state_off, xbc, xbc_off, dt, dt_off, dt_stride, a, d, dt_bias, y, y_off, sn,
                snap_off, n_tokens, n_heads, head_dim, d_state, n_groups,
            ),
            (SsmArena::F16(s), SsmArena::F16(sn)) => e.mamba2_scan_seq_snap_at_f16(
                s, state_off, xbc, xbc_off, dt, dt_off, dt_stride, a, d, dt_bias, y, y_off, sn,
                snap_off, n_tokens, n_heads, head_dim, d_state, n_groups,
            ),
            _ => Err(GpuError::MissingOp(
                "ssm snap class differs from the live arena",
            )),
        }
    }

    /// The verify walk without snapshots (f16 class only, slot 746): the
    /// state is read and left pre-round, each row's scan inputs land in
    /// `keep` at `keep_off` for the commit's replay.
    #[allow(clippy::too_many_arguments)]
    pub fn scan_seq_keep_at(
        &self,
        e: &GpuExecutor,
        state_off: usize,
        xbc: &CudaSlice<f32>,
        xbc_off: usize,
        dt: &CudaSlice<f32>,
        dt_off: usize,
        dt_stride: usize,
        a: &CudaSlice<f32>,
        d: &CudaSlice<f32>,
        dt_bias: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        y_off: usize,
        keep: &mut CudaSlice<f32>,
        keep_off: usize,
        n_tokens: usize,
        n_heads: usize,
        head_dim: usize,
        d_state: usize,
        n_groups: usize,
    ) -> Result<(), GpuError> {
        match self {
            SsmArena::F16(s) => e.mamba2_scan_seq_keep_at_f16(
                s, state_off, xbc, xbc_off, dt, dt_off, dt_stride, a, d, dt_bias, y, y_off, keep,
                keep_off, n_tokens, n_heads, head_dim, d_state, n_groups,
            ),
            SsmArena::F32(_) => Err(GpuError::MissingOp(
                "rollback by replay is the f16 state class's",
            )),
        }
    }

    /// Copy a slot's state into the f32 checkpoint blob.
    pub fn save_to_blob(
        &self,
        e: &GpuExecutor,
        off: usize,
        blob: &mut CudaSlice<f32>,
        blob_off: usize,
        n: usize,
    ) -> Result<(), GpuError> {
        match self {
            SsmArena::F32(s) => e.copy_region(s, off, blob, blob_off, n),
            SsmArena::F16(s) => e.ssm_state_widen(s, off, blob, blob_off, n),
        }
    }

    /// Restore a slot's state from the f32 checkpoint blob.
    pub fn restore_from_blob(
        &mut self,
        e: &GpuExecutor,
        blob: &CudaSlice<f32>,
        blob_off: usize,
        off: usize,
        n: usize,
    ) -> Result<(), GpuError> {
        match self {
            SsmArena::F32(s) => e.copy_region(blob, blob_off, s, off, n),
            SsmArena::F16(s) => e.ssm_state_narrow(blob, blob_off, s, off, n),
        }
    }

    /// Copy a state region from another arena of the same class - the spec
    /// partial-accept rollback (snapshot row -> live slot). Same class means
    /// this is a byte copy with no re-rounding, which is why the snap plane
    /// is allocated in the arena's class rather than always f32.
    pub fn copy_region_from(
        &mut self,
        e: &GpuExecutor,
        src: &SsmArena,
        src_off: usize,
        dst_off: usize,
        n: usize,
    ) -> Result<(), GpuError> {
        match (src, self) {
            (SsmArena::F32(a), SsmArena::F32(b)) => e.copy_region(a, src_off, b, dst_off, n),
            (SsmArena::F16(a), SsmArena::F16(b)) => e.copy_region_f16(a, src_off, b, dst_off, n),
            _ => Err(GpuError::MissingOp("ssm rollback across differing classes")),
        }
    }

    /// Host copy of one slot's state, widened to f32 so callers (tests, the
    /// state dump) see one representation regardless of class.
    pub fn dump_slot(&self, e: &GpuExecutor, off: usize, n: usize) -> Result<Vec<f32>, GpuError> {
        match self {
            SsmArena::F32(s) => {
                let v = s
                    .try_slice(off..off + n)
                    .ok_or(GpuError::MissingOp("state view"))?;
                e.stream
                    .clone_dtoh(&v)
                    .map_err(|err| GpuError::Driver(err.to_string()))
            }
            SsmArena::F16(s) => {
                let v = s
                    .try_slice(off..off + n)
                    .ok_or(GpuError::MissingOp("state view"))?;
                let h: Vec<f16> = e
                    .stream
                    .clone_dtoh(&v)
                    .map_err(|err| GpuError::Driver(err.to_string()))?;
                Ok(h.into_iter().map(|x| x.to_f32()).collect())
            }
        }
    }
}

/// Elected storage class for the recurrent state.
///
/// f16 is the ELECTED DEFAULT. The quality gate (examples/nemotron_ppl.rs,
/// wikitext-2 teacher-forced through the batch path, kv8) measured f16 ppl
/// 6.38046 vs f32 6.42970 - slightly better - with top-1 agreement 96.4% and
/// a symmetric tail, i.e. quality-neutral by the same bar the fp4 and f8t
/// classes were judged against, and it is worth several percent end to end at
/// c32. Arithmetic is f32 in both classes - this elects storage.
///
/// f32 is the checkpoint's own conservative declaration
/// (`mamba_ssm_cache_dtype: float32`). It
/// stays reachable only in dev builds, via PADDOCK_SSM_DTYPE=f32, as the
/// reference class for the NLL gate and class A/Bs - a measurement deviation,
/// never a serving recommendation. Hardened builds always serve f16.
pub fn ssm_dtype_from_env() -> SsmDtype {
    match paddock_models::dev_var!("PADDOCK_SSM_DTYPE").as_deref() {
        Ok("f32") | Ok("float32") | Ok("fp32") => SsmDtype::F32,
        _ => SsmDtype::F16,
    }
}
