//! Growing a slot's paged KV table, and the VRAM plan's graph-pools reserve -
//! split out of `batch.rs`, which carries the rest of the batched lane.

use super::*;
use crate::gpu::GpuError;

/// Pages a cache eviction frees beyond a growth's shortfall (see
/// `ensure_slot_blocks`): 1 KiB tokens of the coldest cache.
const GROW_SLACK: usize = 64;

impl GpuQwen35 {
    /// P5 budget pool: grow slot `slot`'s block table to back logical position
    /// `upto_pos`, allocating physical blocks from the shared free-list and
    /// re-uploading the (small) block table to the device. No-op in identity
    /// mode (`pool` is `None`). The device upload is outside any captured graph
    /// (callers invoke this before the capture/replay), and the paged kernels
    /// read the table by pointer at replay - so growth is graph-safe.
    pub(super) fn ensure_slot_blocks(
        &mut self,
        slot: usize,
        upto_pos: usize,
    ) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        if self.batch.as_ref().expect("batch").pool.is_none() {
            return Ok(());
        }
        let before = self.batch.as_ref().expect("batch enabled").tables[slot]
            .blocks()
            .len();
        // Grow the slot's table from the pool. On exhaustion, evict an LRU cached
        // prefix - its pages return to the pool (blocks a live slot still holds
        // survive on refcount) - and retry; when the tree is dry, surface
        // PoolExhausted for the scheduler to preempt.
        loop {
            let grew = {
                let bs = self.batch.as_mut().expect("batch enabled");
                let pool = bs.pool.as_mut().expect("pool checked above");
                bs.tables[slot].ensure(upto_pos, pool).is_ok()
            };
            if grew {
                break;
            }
            let evicted = {
                let exec = self.exec.clone();
                let bs = self.batch.as_mut().expect("batch enabled");
                let pool = bs.pool.as_mut().expect("pool checked above");
                // the whole remainder in one pass: a page at a time cost a walk
                // of the tree (dead leaves, then checkpoint victims) per page,
                // and a full pool sends every span and decode growth here
                let short = (upto_pos / BLOCK_TOKENS + 1)
                    .saturating_sub(bs.tables[slot].blocks().len())
                    .max(1);
                let want = pool.free_blocks() + short;
                match (bs.tier.as_mut(), bs.paged_prefix.as_mut()) {
                    (Some(tier), Some(pr)) => {
                        // tier-aware shed: closing runs (and their DeltaNet
                        // checkpoint blobs) demote to T1 before eviction -
                        // the cliff-grade press (see `make_room_blocking`:
                        // one pressure pass + 50ms was not enough while a
                        // parked restore's loads held the lane). The
                        // checkpoints are pool pages, which the tier reads off
                        // the radix, so there is no blob base to pass.
                        tier.make_room_blocking(pr, pool, want, None, &mut || {
                            exec.record_event().ok()
                        })
                        .then_some(0u32)
                    }
                    // dead KV, then the stalest checkpoint's pages, then LRU
                    // KV - context a slot was promised outranks all cache
                    // - plus GROW_SLACK pages, so the next decode growths find
                    // them free instead of walking the tree one page apiece
                    (None, Some(pr)) => (pr.make_room(pool, want + GROW_SLACK, 0)
                        || pool.free_blocks() >= short)
                        .then_some(0u32),
                    _ => None,
                }
            };
            if evicted.is_none() {
                return Err(GpuModelError::PoolExhausted);
            }
        }
        let bs = self.batch.as_mut().expect("batch enabled");
        let pool = bs.pool.as_mut().expect("pool checked above");
        let (free, cap) = (pool.free_blocks(), pool.capacity());
        let blocks = bs.tables[slot].blocks().to_vec();
        if blocks.len() > before {
            if paddock_models::dev_var_os!("PADDOCK_POOL_STATS").is_some() {
                tracing::info!(
                    "pool: slot {slot} grew {before}->{} blocks  ({free}/{cap} free)",
                    blocks.len()
                );
            }
            let (base, bps) = (slot * bs.blocks_per_slot, bs.blocks_per_slot);
            for (j, &blk) in blocks.iter().enumerate() {
                bs.block_table_host[base + j] = blk;
            }
            // THIS slot's row only. A whole-table upload is a snapshot of every
            // slot's row at enqueue time, and the two execution lanes both grow
            // tables: a main-lane upload queued behind a long pass landed after
            // a decode-lane slot's newer one and put back that slot's old row -
            // its next token's KV went to a stale page and the stream read a
            // block nobody wrote (fp8 NaN, then id-0 forever; 25 of 100 A4
            // pages at 32 slots). Rows are disjoint, so order no longer matters.
            let dst = bs
                .d_block_tables
                .as_mut()
                .expect("pool implies device table");
            exec.stream
                .memcpy_htod(
                    &bs.block_table_host[base..base + bps],
                    &mut dst.slice_mut(base..base + bps),
                )
                .map_err(|e| GpuError::Driver(e.to_string()))?;
        }
        Ok(())
    }
}

/// The `graph pools + headroom` reserve: what the plan cannot meet in the
/// ledger before it runs and what no other term names - CUDA graph
/// instantiation for every captured width, the decode arena, per-width sampler
/// planes, per-image vision transients, and allocator retained-not-live slack.
///
/// This is the residual of the old 3 GiB "graph/prefill scratch" guess once
/// its two large tenants got honest accounting: the prefill scratch is now
/// profiled at load (`enable_batch_sized` allocates it before the grant is
/// read) and the wave-pass buffers are computed (`pf_bufs_bytes`). The
/// operator's `graph_scratch_mib` override still applies, to this term.
///
/// Measured 2026-09-06 on the RTX PRO 6000, Qwen3.8-27B Q8_0, max_batch 32,
/// spec on, vision, after the smoke sweep: see the audit log
/// (`qwen35 VRAM plan audit`) - the constant is the measured residual with
/// headroom, and the audit line is how to re-check it after a change.
const GRAPH_POOLS_HEADROOM_MIB: u64 = 768;

pub(super) fn graph_pools_headroom_bytes() -> u64 {
    crate::kv_plan::graph_scratch_override_mib().unwrap_or(GRAPH_POOLS_HEADROOM_MIB) << 20
}
