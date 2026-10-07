//! the rows partials: split-K attention over runs of one slot's query rows -
//! the spec verify and block-drafter walks and nemotron's W16 decode class
//! (slots 679 shared span, 683 fixed splits, 744 pow2 splits, 745 two warps a
//! row with its TILE law).

use super::error::*;
use super::*;
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

impl GpuExecutor {
    /// True when the pack carries the multi-row split partial (slot 679).
    pub fn has_attn_rows_partial(&self) -> bool {
        self.kernels.attn_rows_partial.is_some()
    }

    /// Split-K decode attention over GROUPS of one slot's consecutive query
    /// rows (slot 679): `groups` is `[n_groups][2]` u32 of (first row, rows
    /// <= 8). Each row attends keys up to its own position (a verify chunk's
    /// causal tail; a block drafter's rows share the block end), and each K/V
    /// tile is read once per group - the decode partial reads it once per
    /// ROW. `pool_k/v` is the paged block pool (`block_tables` Some) or a
    /// dense f16 cache `[slots, max_ctx, kv_dim]` (None). `window` > 0 bounds
    /// each row to its last `window` keys (0 = full). hd128, G <= 16.
    /// Partials pair with [`Self::attn_combine_batch`] at `batch = n_rows`:
    /// `out_o` >= n_heads * n_rows * n_splits * 128 floats, `out_ml` twice
    /// n_heads * n_rows * n_splits.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_rows_partial(
        &self,
        q: &CudaSlice<f32>,
        pool_k: &CudaSlice<u8>,
        pool_v: &CudaSlice<u8>,
        out_o: &mut CudaSlice<f32>,
        out_ml: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        slots: &CudaSlice<u32>,
        groups: &CudaSlice<u32>,
        n_groups: usize,
        block_tables: Option<(&CudaSlice<u32>, usize)>,
        max_ctx: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_dim: usize,
        n_rows: usize,
        n_splits: usize,
        window: usize,
        scale: f32,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .attn_rows_partial
            .ok_or(GpuError::MissingOp("attn_rows_partial"))?;
        self.attn_rows_partial_go(
            AttnRowsFn::Shared(f),
            q,
            pool_k,
            pool_v,
            out_o,
            out_ml,
            positions,
            slots,
            groups,
            n_groups,
            block_tables,
            max_ctx,
            n_heads,
            n_kv_heads,
            head_dim,
            kv_dim,
            n_rows,
            n_splits,
            window,
            scale,
            kv_dtype,
        )
    }

    /// True when the pack carries the fixed-split rows partial (slot 683).
    pub fn has_attn_rows_partial_fixed(&self) -> bool {
        self.kernels.attn_rows_partial_fixed.is_some()
    }

    /// True when the pack carries slot 744's pow2 split law
    /// ([`Self::attn_rows_partial_fixed`] with `split_keys` 0).
    pub fn has_attn_rows_partial_pow2(&self) -> bool {
        self.kernels.attn_rows_partial_pow2.is_some()
    }

    /// True when the pack carries slot 745's two-warps-a-row kernel
    /// ([`Self::attn_rows_partial_fixed`] with `kh_rows`).
    pub fn has_attn_rows_partial_kh(&self) -> bool {
        self.kernels.attn_rows_partial_kh.is_some()
    }

    /// [`Self::attn_rows_partial`] on FIXED key splits: split s holds keys
    /// `[s * split_keys, (s + 1) * split_keys)` (a multiple of 32), so a row's
    /// partials depend on its own keys alone - the W16 decode class runs
    /// decode ticks (one group per row) and verify rounds through one law and
    /// one split count. Splits past a group's keys write m = -inf only.
    /// Full attention only for that property: under a window a split starts
    /// at the group's earliest in-window key, which its other rows move.
    /// `split_keys` 0 selects slot 744's law: the split size follows each
    /// row's own key count n - max(256, next_pow2(ceil(n / n_splits))) - so
    /// a shallow context fills the die; groups must sit in one size bucket.
    /// `kh_rows` = Some(largest group's rows, <= 6) runs slot 745 instead:
    /// the same laws with every row's keys split across two warps - a
    /// different fold, so decode ticks and verify rounds must BOTH take it
    /// (e4m3 paged pools only) - plus its TILE law, `split_keys` =
    /// 0x4000_0000 | budget (745 only).
    #[allow(clippy::too_many_arguments)]
    pub fn attn_rows_partial_fixed(
        &self,
        q: &CudaSlice<f32>,
        pool_k: &CudaSlice<u8>,
        pool_v: &CudaSlice<u8>,
        out_o: &mut CudaSlice<f32>,
        out_ml: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        slots: &CudaSlice<u32>,
        groups: &CudaSlice<u32>,
        n_groups: usize,
        block_tables: Option<(&CudaSlice<u32>, usize)>,
        max_ctx: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_dim: usize,
        n_rows: usize,
        n_splits: usize,
        window: usize,
        scale: f32,
        kv_dtype: KvDtype,
        split_keys: usize,
        kh_rows: Option<usize>,
    ) -> Result<(), GpuError> {
        let f = if let Some(m) = kh_rows {
            // 745 reads 744's law from the split_keys flag (budget = n_splits)
            let law = if split_keys == 0 {
                0x8000_0000 | n_splits as u32
            } else {
                split_keys as u32
            };
            AttnRowsFn::Kh(
                self.kernels
                    .attn_rows_partial_kh
                    .ok_or(GpuError::MissingOp("attn_rows_partial_kh"))?,
                law,
                m as u32,
            )
        } else if split_keys & 0xc000_0000 != 0 {
            return Err(GpuError::Unsupported(format!(
                "attn_rows_partial_fixed: law word {split_keys:#x} is slot 745's (pass kh_rows)"
            )));
        } else if split_keys == 0 {
            // 744 takes 683's arguments less split_keys - the shared shape
            AttnRowsFn::Shared(
                self.kernels
                    .attn_rows_partial_pow2
                    .ok_or(GpuError::MissingOp("attn_rows_partial_pow2"))?,
            )
        } else {
            AttnRowsFn::Fixed(
                self.kernels
                    .attn_rows_partial_fixed
                    .ok_or(GpuError::MissingOp("attn_rows_partial_fixed"))?,
                split_keys as u32,
            )
        };
        self.attn_rows_partial_go(
            f,
            q,
            pool_k,
            pool_v,
            out_o,
            out_ml,
            positions,
            slots,
            groups,
            n_groups,
            block_tables,
            max_ctx,
            n_heads,
            n_kv_heads,
            head_dim,
            kv_dim,
            n_rows,
            n_splits,
            window,
            scale,
            kv_dtype,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn attn_rows_partial_go(
        &self,
        f: AttnRowsFn,
        q: &CudaSlice<f32>,
        pool_k: &CudaSlice<u8>,
        pool_v: &CudaSlice<u8>,
        out_o: &mut CudaSlice<f32>,
        out_ml: &mut CudaSlice<f32>,
        positions: &CudaSlice<u32>,
        slots: &CudaSlice<u32>,
        groups: &CudaSlice<u32>,
        n_groups: usize,
        block_tables: Option<(&CudaSlice<u32>, usize)>,
        max_ctx: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_dim: usize,
        n_rows: usize,
        n_splits: usize,
        window: usize,
        scale: f32,
        kv_dtype: KvDtype,
    ) -> Result<(), GpuError> {
        let need = n_heads * n_rows * n_splits;
        if out_o.len() < need * head_dim || out_ml.len() < need * 2 || groups.len() < n_groups * 2 {
            return Err(GpuError::Unsupported(format!(
                "attn_rows_partial: partial planes under {n_heads} heads x {n_rows} rows x \
                 {n_splits} splits (or groups under {n_groups})"
            )));
        }
        let (qp, _g1) = q.device_ptr(&self.stream);
        let (kp, _g2) = pool_k.device_ptr(&self.stream);
        let (vp, _g3) = pool_v.device_ptr(&self.stream);
        let (op, _g4) = out_o.device_ptr_mut(&self.stream);
        let (mp, _g5) = out_ml.device_ptr_mut(&self.stream);
        let (pp, _g6) = positions.device_ptr(&self.stream);
        let (slp, _g7) = slots.device_ptr(&self.stream);
        let (gp, _g8) = groups.device_ptr(&self.stream);
        let bt_guard = block_tables.map(|(b, bps)| (b.device_ptr(&self.stream), bps));
        let (btp, bps) = match &bt_guard {
            Some(((p, _), bps)) => (*p as *const core::ffi::c_void, *bps),
            None => (std::ptr::null(), 0),
        };
        let paged = u32::from(bt_guard.is_some());
        // SAFETY: ABI contract; planes sized above, the pool [n_blocks, 16,
        // kv_dim] (paged) or [slots, max_ctx, kv_dim] f16 (dense)
        check(unsafe {
            match f {
                AttnRowsFn::Shared(f) => f(
                    qp as *const _,
                    kp as *const _,
                    vp as *const _,
                    op as *mut _,
                    mp as *mut _,
                    pp as *const _,
                    slp as *const _,
                    gp as *const _,
                    n_groups as u32,
                    btp,
                    bps as u32,
                    max_ctx as u32,
                    n_heads as u32,
                    n_kv_heads as u32,
                    head_dim as u32,
                    kv_dim as u32,
                    n_rows as u32,
                    n_splits as u32,
                    window as u32,
                    scale,
                    kv_dtype as u32,
                    paged,
                    self.stream_ptr(),
                ),
                AttnRowsFn::Fixed(f, split_keys) => f(
                    qp as *const _,
                    kp as *const _,
                    vp as *const _,
                    op as *mut _,
                    mp as *mut _,
                    pp as *const _,
                    slp as *const _,
                    gp as *const _,
                    n_groups as u32,
                    btp,
                    bps as u32,
                    max_ctx as u32,
                    n_heads as u32,
                    n_kv_heads as u32,
                    head_dim as u32,
                    kv_dim as u32,
                    n_rows as u32,
                    n_splits as u32,
                    window as u32,
                    scale,
                    kv_dtype as u32,
                    paged,
                    split_keys,
                    self.stream_ptr(),
                ),
                AttnRowsFn::Kh(f, split_keys, max_rows) => f(
                    qp as *const _,
                    kp as *const _,
                    vp as *const _,
                    op as *mut _,
                    mp as *mut _,
                    pp as *const _,
                    slp as *const _,
                    gp as *const _,
                    n_groups as u32,
                    btp,
                    bps as u32,
                    max_ctx as u32,
                    n_heads as u32,
                    n_kv_heads as u32,
                    head_dim as u32,
                    kv_dim as u32,
                    n_rows as u32,
                    n_splits as u32,
                    window as u32,
                    scale,
                    kv_dtype as u32,
                    paged,
                    split_keys,
                    max_rows,
                    self.stream_ptr(),
                ),
            }
        })
    }
}

/// The rows partial's entries: the shared-span split law (slot 679), fixed
/// key splits (slot 683, with the split size), or slot 745's two warps a row.
#[derive(Clone, Copy)]
enum AttnRowsFn {
    Shared(paddock_kernels::abi::AttnRowsPartialFn),
    Fixed(paddock_kernels::abi::AttnRowsPartialFixedFn, u32),
    /// slot 745: the split law word and the block's row capacity
    Kh(paddock_kernels::abi::AttnRowsPartialKhFn, u32, u32),
}
