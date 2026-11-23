//! Draft-tree verify wrappers - pack slots 975-976: one same-slot run of a
//! tree's verify rows on the masked pf7rp rung, and the KV compaction that
//! moves an accepted path's rows down to their true positions.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use super::error::check;
use super::{GpuError, GpuExecutor, KvDtype};

impl GpuExecutor {
    /// Slots 975 + 976 are both in the pack.
    pub fn has_spec_tree(&self) -> bool {
        self.kernels.attn_prefill_tree.is_some() && self.kernels.kv_tree_move.is_some()
    }

    /// Tree rows `[row_off, row_off + rows)` of one slot: `positions` are
    /// the rows' cache slots (the causal bound and the appends' targets),
    /// `win_pos` their logical positions (the window floor), `tmask` two u32
    /// a row (ancestor mask over the tree's keys, the tree's base slot).
    /// Ok(false) = the rung declines the shape.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_prefill_tree_at(
        &self,
        q: &CudaSlice<f32>,
        pool_k: &CudaSlice<u8>,
        pool_v: &CudaSlice<u8>,
        sinks: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        (positions, win_pos, tmask): (&CudaSlice<u32>, &CudaSlice<u32>, &CudaSlice<u32>),
        slots: &CudaSlice<u32>,
        row_off: usize,
        (block_tables, blocks_per_slot): (&CudaSlice<u32>, usize),
        (n_heads, n_kv_heads, head_dim, kv_dim): (usize, usize, usize, usize),
        swa_window: usize,
        rows: usize,
        scale: f32,
        kv_dtype: KvDtype,
    ) -> Result<bool, GpuError> {
        let f = self
            .kernels
            .attn_prefill_tree
            .ok_or(GpuError::MissingOp("attn_prefill_tree"))?;
        let q_row = n_heads * head_dim;
        let end = row_off + rows;
        if end * q_row > q.len()
            || end * q_row > out.len()
            || end > positions.len()
            || end > win_pos.len()
            || 2 * end > tmask.len()
            || end > slots.len()
        {
            return Err(GpuError::Unsupported(format!(
                "attn_prefill_tree: rows {row_off}..{end} past a plane"
            )));
        }
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = pool_k.device_ptr(&self.stream);
        let (vp, _g3) = pool_v.device_ptr(&self.stream);
        let (sp, _g4) = sinks.device_ptr(&self.stream);
        let (op, _g5) = out.device_ptr_mut(&self.stream);
        let (pp, _g6) = positions.device_ptr(&self.stream);
        let (wp, _g7) = win_pos.device_ptr(&self.stream);
        let (tp, _g8) = tmask.device_ptr(&self.stream);
        let (slp, _g9) = slots.device_ptr(&self.stream);
        let (btp, _g10) = block_tables.device_ptr(&self.stream);
        let qo = (row_off * q_row * 4) as u64;
        let ro = (row_off * 4) as u64;
        // SAFETY: ABI contract (slot 975); every plane checked to cover the
        // run's rows above, the pools are the layer's own
        let rc = unsafe {
            f(
                (qp + qo) as *const _,
                kp as *const _,
                vp as *const _,
                sp as *const _,
                (op + qo) as *mut _,
                (pp + ro) as *const _,
                (wp + ro) as *const _,
                (tp + 2 * ro) as *const _,
                btp as *const _,
                blocks_per_slot as u32,
                (slp + ro) as *const _,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                kv_dim as u32,
                swa_window as u32,
                rows as u32,
                scale,
                kv_dtype as u32,
                self.stream_ptr(),
            )
        };
        if rc == -2 {
            return Ok(false);
        }
        check(rc).map(|()| true)
    }

    /// Slot 976: move every layer's accepted tree rows (`plan`: `n_groups`
    /// records of `stride` u32, `[slot, n, (src, dst) x n]` in depth order);
    /// `kp` / `vp` the layers' pool bases (u64 device pointers), `kind` a
    /// layer's table (0 = `bt0`, 1 = `bt1`), `row_bytes` a pool row's bytes.
    #[allow(clippy::too_many_arguments)]
    pub fn kv_tree_move(
        &self,
        (kp, vp, kind): (&CudaSlice<u64>, &CudaSlice<u64>, &CudaSlice<u32>),
        layers: usize,
        (bt0, bps0): (&CudaSlice<u32>, usize),
        (bt1, bps1): (&CudaSlice<u32>, usize),
        plan: &CudaSlice<u32>,
        n_groups: usize,
        stride: usize,
        row_bytes: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .kv_tree_move
            .ok_or(GpuError::MissingOp("kv_tree_move"))?;
        if kp.len() < layers || vp.len() < layers || kind.len() < layers {
            return Err(GpuError::Unsupported(
                "kv_tree_move: pointer planes under the layer count".into(),
            ));
        }
        if plan.len() < n_groups * stride {
            return Err(GpuError::Unsupported(
                "kv_tree_move: plan under its groups".into(),
            ));
        }
        let (kpp, _g1) = kp.device_ptr(&self.stream);
        let (vpp, _g2) = vp.device_ptr(&self.stream);
        let (kdp, _g3) = kind.device_ptr(&self.stream);
        let (b0, _g4) = bt0.device_ptr(&self.stream);
        let (b1, _g5) = bt1.device_ptr(&self.stream);
        let (pl, _g6) = plan.device_ptr(&self.stream);
        // SAFETY: ABI contract (slot 976); the pointer planes name the
        // layers' live pools, the plan's positions sit inside each slot's
        // mapped blocks (the caller backed the round's rows)
        check(unsafe {
            f(
                kpp as *const _,
                vpp as *const _,
                kdp as *const _,
                layers as u32,
                b0 as *const _,
                bps0 as u32,
                b1 as *const _,
                bps1 as u32,
                pl as *const _,
                n_groups as u32,
                stride as u32,
                row_bytes as u32,
                self.stream_ptr(),
            )
        })
    }
}
