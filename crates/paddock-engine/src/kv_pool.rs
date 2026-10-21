//! Paged KV pool - the block allocator + per-sequence block tables that decouple
//! KV memory from `max_ctx × max_batch`. See.
//!
//! **P1 (this module) is the CPU-side bookkeeping only:** which physical block
//! backs which `(sequence, logical position)`, a free-list + refcount so a block
//! shared by several sequences / radix nodes is freed only at refcount 0,
//! copy-on-write for zero-copy prefix sharing, and a sliding ring for windowed
//! (SWA) layers whose allocation is capped at the window instead of `max_ctx`.
//!
//! It owns **no device memory** and touches no kernel - a `BlockId` is just an
//! index into the per-layer `[n_blocks, BLOCK_TOKENS, kv_dim]` GPU store that P2
//! adds, and the paged attention kernels that read that store land in P2+. Keeping
//! this layer pure makes the allocator fully unit-testable before any GPU work,
//! which is where subtle refcount/CoW/ring bugs would otherwise hide.

/// A physical block id: indexes the per-layer `[n_blocks, BLOCK_TOKENS, kv_dim]`
/// pool store (the same id addresses every paged layer - a combined block table,
/// which vLLM notes is free vs. per-layer tables and cuts metadata).
pub type BlockId = u32;

/// Tokens per block (page granularity). 16 keeps each block an internally
/// contiguous, tensor-core-tile-aligned run, so the tuned attention inner loop is
/// reused unchanged over each page - the FlashInfer "≥16" rule that avoids
/// PagedAttention's 20-26% kernel tax. (Matches `prefix_cache::BLOCK_TOKENS`.)
pub const BLOCK_TOKENS: usize = 16;

/// The pool had no free block to satisfy an allocation. The scheduler's response
/// is to preempt a victim sequence (recompute on re-admit) - vAttention's
/// `step() -> fail -> preempt` contract, portable to unified-memory targets where
/// "swap" is meaningless.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolExhausted;

/// Fixed-capacity pool of physical KV blocks shared by every active sequence and
/// (later) the radix prefix cache. Refcounted: a block referenced by K sequences
/// or radix nodes has refcount K; only refcount-0 blocks are free.
#[derive(Debug)]
pub struct KvPool {
    /// refcount per block; 0 = free (present in `free`).
    refcount: Vec<u32>,
    /// LIFO free-list of refcount-0 blocks. LIFO keeps recently-freed blocks hot
    /// and makes deferred-reclamation (hand a finished seq's blocks to the next
    /// arrival) a natural fast path.
    free: Vec<BlockId>,
    /// allocation generation per block, bumped by every `alloc` (0 = never
    /// allocated). A side store addressed by block id - rows a drafter keeps
    /// beside the pool's own - stamps the generation it wrote under, and a
    /// page whose generation moved since has had another tenant: freed and
    /// handed out again, whether to a sequence or to a tier restore that
    /// refills only the pool's own planes.
    tenant: Vec<u32>,
    /// Blocks the prefix cache holds a reference to ([`Self::retain_cached`]),
    /// and how many of those nobody else references (refcount 1) - what the
    /// cache could free right now. Counted on every refcount move so the
    /// admission figure is O(1): it was a walk of the whole radix tree,
    /// several times a scheduler pass and once a pipelined decode tick.
    cached: Vec<bool>,
    lone_cached: usize,
}

impl KvPool {
    /// A pool of exactly `n_blocks` blocks, all free.
    pub fn with_blocks(n_blocks: u32) -> Self {
        Self {
            refcount: vec![0; n_blocks as usize],
            // free-list high-id-first so `alloc` hands out 0,1,2,... in order (nicer
            // for tests + locality); order is otherwise immaterial.
            free: (0..n_blocks).rev().collect(),
            tenant: vec![0; n_blocks as usize],
            cached: vec![false; n_blocks as usize],
            lone_cached: 0,
        }
    }

    /// Size the pool from a VRAM budget. `block_bytes` is the bytes one block id
    /// consumes across all paged layers (`n_paged_layers × 2(K,V) × BLOCK_TOKENS ×
    /// kv_dim × kv_bytes`). This is the vLLM `--gpu-memory-utilization` knob: KV
    /// capacity follows the budget, not `max_ctx × max_batch`.
    pub fn with_budget(budget_bytes: u64, block_bytes: u64) -> Self {
        let n = budget_bytes.checked_div(block_bytes).unwrap_or(0) as u32;
        Self::with_blocks(n)
    }

    /// Total blocks in the pool.
    pub fn capacity(&self) -> u32 {
        self.refcount.len() as u32
    }

    /// Blocks currently free (available to `alloc`).
    pub fn free_blocks(&self) -> usize {
        self.free.len()
    }

    /// Current refcount of a block (0 = free).
    pub fn refcount(&self, b: BlockId) -> u32 {
        self.refcount[b as usize]
    }

    /// Allocate a fresh block (refcount -> 1). `Err` = pool exhausted.
    pub fn alloc(&mut self) -> Result<BlockId, PoolExhausted> {
        let b = self.free.pop().ok_or(PoolExhausted)?;
        // Hard invariants, not debug_asserts: a violated one hands a block to
        // two owners and the serve keeps running on corrupted KV - the failure
        // then surfaces as wrong tokens far from the cause. A panic here names
        // the caller instead.
        assert_eq!(
            self.refcount[b as usize], 0,
            "kv pool: free block {b} had nonzero refcount"
        );
        self.refcount[b as usize] = 1;
        let g = &mut self.tenant[b as usize];
        // wraps past 0 so a stamp of 0 always reads as "never written"
        *g = g.wrapping_add(1).max(1);
        Ok(b)
    }

    /// The block's allocation generation (see `tenant`).
    pub fn generation(&self, b: BlockId) -> u32 {
        self.tenant[b as usize]
    }

    /// Add a reference to an already-allocated block (refcount++). Used when a new
    /// sequence adopts a shared prefix block.
    pub fn retain(&mut self, b: BlockId) {
        assert!(
            self.refcount[b as usize] > 0,
            "kv pool: retain of free block {b}"
        );
        if self.cached[b as usize] && self.refcount[b as usize] == 1 {
            self.lone_cached -= 1;
        }
        self.refcount[b as usize] += 1;
    }

    /// Drop a reference (refcount--). At 0 the block returns to the free-list.
    pub fn release(&mut self, b: BlockId) {
        let rc = &mut self.refcount[b as usize];
        assert!(*rc > 0, "kv pool: double release of block {b}");
        *rc -= 1;
        if self.cached[b as usize] {
            assert!(*rc > 0, "kv pool: block {b} freed under the prefix cache");
            if *rc == 1 {
                self.lone_cached += 1;
            }
        }
        if *rc == 0 {
            self.free.push(b);
        }
    }

    /// [`Self::retain`] on behalf of the prefix cache: from now on the block
    /// counts toward [`Self::lone_cached`] whenever the cache is its only
    /// holder. One cache reference per block.
    pub fn retain_cached(&mut self, b: BlockId) {
        assert!(!self.cached[b as usize], "kv pool: block {b} cached twice");
        self.retain(b);
        self.cached[b as usize] = true;
    }

    /// The prefix cache dropping the reference [`Self::retain_cached`] took.
    pub fn release_cached(&mut self, b: BlockId) {
        assert!(self.cached[b as usize], "kv pool: block {b} not cached");
        if self.refcount[b as usize] == 1 {
            self.lone_cached -= 1;
        }
        self.cached[b as usize] = false;
        self.release(b);
    }

    /// Blocks only the prefix cache references - freed the moment it evicts them.
    pub fn lone_cached(&self) -> usize {
        self.lone_cached
    }

    /// Copy-on-write before writing to a block you reference. If `b` is unshared
    /// (refcount 1) you own it: returns `(b, false)` - write in place. If shared
    /// (refcount > 1): allocates a fresh block, **moves your reference** off `b`
    /// (release) onto the new block, and returns `(new, true)` - the caller must
    /// device-copy `b -> new` before writing, and repoint its block-table entry.
    /// `Err` only when a copy is needed but the pool is exhausted.
    pub fn cow(&mut self, b: BlockId) -> Result<(BlockId, bool), PoolExhausted> {
        if self.refcount[b as usize] <= 1 {
            return Ok((b, false));
        }
        let new = self.alloc()?;
        self.release(b);
        Ok((new, true))
    }
}

/// Per-sequence logical->physical block map for a `Paged` (full-attention) layer.
/// Grows by whole blocks as the sequence decodes; entry `i` backs logical tokens
/// `[i*BLOCK_TOKENS, (i+1)*BLOCK_TOKENS)`.
#[derive(Debug, Default)]
pub struct BlockTable {
    blocks: Vec<BlockId>,
}

impl BlockTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// The physical block ids, in logical order (fed to the paged kernel in P2).
    pub fn blocks(&self) -> &[BlockId] {
        &self.blocks
    }

    /// How many logical tokens the table currently backs.
    pub fn token_capacity(&self) -> usize {
        self.blocks.len() * BLOCK_TOKENS
    }

    /// Ensure the table backs logical position `pos` (0-based), allocating blocks
    /// from `pool` as needed. Cheap no-op once capacity is reached. `Err` = pool
    /// exhausted mid-grow (the already-grown blocks stay; the scheduler preempts).
    pub fn ensure(&mut self, pos: usize, pool: &mut KvPool) -> Result<(), PoolExhausted> {
        let need = pos / BLOCK_TOKENS + 1;
        while self.blocks.len() < need {
            self.blocks.push(pool.alloc()?);
        }
        Ok(())
    }

    /// Physical `(block, offset_in_block)` for logical position `pos`. `pos` must
    /// be within `token_capacity()` (call `ensure` first).
    pub fn locate(&self, pos: usize) -> (BlockId, usize) {
        (self.blocks[pos / BLOCK_TOKENS], pos % BLOCK_TOKENS)
    }

    /// Adopt a shared prefix: append `shared` blocks to this (typically empty)
    /// table, retaining each in the pool. This is the zero-copy radix reuse
    /// mechanism - the slot's leading logical blocks point at physical blocks that
    /// other sequences / radix nodes also hold. The first subsequent write into a
    /// shared block goes through `KvPool::cow`.
    pub fn share_prefix(&mut self, shared: &[BlockId], pool: &mut KvPool) {
        for &b in shared {
            pool.retain(b);
            self.blocks.push(b);
        }
    }

    /// Copy-on-write the block backing `pos` if it is shared, updating the table
    /// entry. Returns `Some((src, dst))` if a device copy `src -> dst` is owed,
    /// `None` if the block was already private. Call before writing token `pos`.
    pub fn cow_at(
        &mut self,
        pos: usize,
        pool: &mut KvPool,
    ) -> Result<Option<(BlockId, BlockId)>, PoolExhausted> {
        let i = pos / BLOCK_TOKENS;
        let old = self.blocks[i];
        let (nb, copied) = pool.cow(old)?;
        if copied {
            self.blocks[i] = nb;
            Ok(Some((old, nb)))
        } else {
            Ok(None)
        }
    }

    /// Release every block back to the pool (sequence finished or preempted).
    pub fn clear(&mut self, pool: &mut KvPool) {
        for &b in &self.blocks {
            pool.release(b);
        }
        self.blocks.clear();
    }
}

/// A fixed-size ring of blocks for a sliding-window (SWA / banded) layer. Only the
/// last `window` tokens are ever read, so allocation is capped at
/// `ceil(window/BLOCK_TOKENS) + 1` blocks (the `+1` gives the block being written
/// headroom so it never clobbers a block still inside the read window) and reused
/// cyclically - instead of `max_ctx` blocks per slot.
#[derive(Debug)]
pub struct WindowRing {
    window: usize,
    ring: Vec<BlockId>,
}

impl WindowRing {
    /// Blocks a ring for `window` needs: `ceil(window/BLOCK_TOKENS) + 1`.
    pub fn blocks_for(window: usize) -> usize {
        window.div_ceil(BLOCK_TOKENS) + 1
    }

    /// Allocate a ring for a `window`-sized SWA layer. `Err` (and no leak - any
    /// partially-allocated blocks are released) if the pool can't fund the ring.
    pub fn new(window: usize, pool: &mut KvPool) -> Result<Self, PoolExhausted> {
        let n = Self::blocks_for(window);
        let mut ring = Vec::with_capacity(n);
        for _ in 0..n {
            match pool.alloc() {
                Ok(b) => ring.push(b),
                Err(e) => {
                    for &b in &ring {
                        pool.release(b);
                    }
                    return Err(e);
                }
            }
        }
        Ok(Self { window, ring })
    }

    pub fn window(&self) -> usize {
        self.window
    }

    pub fn ring_blocks(&self) -> usize {
        self.ring.len()
    }

    /// Physical `(block, offset)` for absolute position `pos`. Logical block
    /// `pos/BLOCK_TOKENS` maps cyclically onto the ring; the read side bounds the
    /// window (`first_pos = pos + 1 - window`), exactly as the current SWA kernels
    /// do - the ring only recycles storage the window has moved past.
    pub fn locate(&self, pos: usize) -> (BlockId, usize) {
        (
            self.ring[(pos / BLOCK_TOKENS) % self.ring.len()],
            pos % BLOCK_TOKENS,
        )
    }

    /// Release the ring back to the pool.
    pub fn clear(&mut self, pool: &mut KvPool) {
        for &b in &self.ring {
            pool.release(b);
        }
        self.ring.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_until_exhausted_then_release_recycles() {
        let mut pool = KvPool::with_blocks(3);
        assert_eq!(pool.capacity(), 3);
        assert_eq!(pool.free_blocks(), 3);
        let a = pool.alloc().unwrap();
        let b = pool.alloc().unwrap();
        let c = pool.alloc().unwrap();
        assert_eq!(pool.free_blocks(), 0);
        assert_eq!(pool.alloc(), Err(PoolExhausted));
        assert_eq!((a, b, c), (0, 1, 2)); // in-order hand-out
        pool.release(b);
        assert_eq!(pool.free_blocks(), 1);
        assert_eq!(pool.alloc().unwrap(), b); // recycled
    }

    #[test]
    fn generation_moves_with_every_tenant() {
        let mut pool = KvPool::with_blocks(2);
        let a = pool.alloc().unwrap();
        let g = pool.generation(a);
        assert!(g > 0, "an allocated block has a nonzero generation");
        pool.retain(a);
        pool.release(a);
        assert_eq!(pool.generation(a), g, "sharing is the same tenant");
        pool.release(a);
        assert_eq!(pool.alloc().unwrap(), a);
        assert_ne!(pool.generation(a), g, "a re-issued block is a new tenant");
    }

    #[test]
    fn refcount_frees_only_at_zero() {
        let mut pool = KvPool::with_blocks(2);
        let a = pool.alloc().unwrap();
        pool.retain(a); // rc 2
        pool.retain(a); // rc 3
        assert_eq!(pool.refcount(a), 3);
        pool.release(a); // 2
        pool.release(a); // 1
        assert_eq!(pool.free_blocks(), 1); // the other block; `a` still held
        pool.release(a); // 0 -> freed
        assert_eq!(pool.refcount(a), 0);
        assert_eq!(pool.free_blocks(), 2);
    }

    #[test]
    fn cow_private_writes_in_place_shared_copies() {
        let mut pool = KvPool::with_blocks(4);
        let a = pool.alloc().unwrap();
        // unshared: write in place, no copy.
        assert_eq!(pool.cow(a).unwrap(), (a, false));
        // share it (a second owner), then CoW: fresh block + copy owed, our ref moved.
        pool.retain(a); // rc 2 (two owners)
        let (nb, copied) = pool.cow(a).unwrap();
        assert!(copied && nb != a);
        assert_eq!(pool.refcount(a), 1); // the other owner remains
        assert_eq!(pool.refcount(nb), 1);
    }

    #[test]
    fn cow_reports_exhaustion() {
        let mut pool = KvPool::with_blocks(1);
        let a = pool.alloc().unwrap();
        pool.retain(a); // shared, pool now full
        assert_eq!(pool.cow(a), Err(PoolExhausted));
    }

    #[test]
    fn with_budget_sizes_by_bytes() {
        // 10 blocks worth of budget, 1 spare byte ignored.
        let pool = KvPool::with_budget(1024 * 10 + 1, 1024);
        assert_eq!(pool.capacity(), 10);
        assert_eq!(KvPool::with_budget(500, 1024).capacity(), 0);
    }

    #[test]
    fn block_table_grows_and_locates() {
        let mut pool = KvPool::with_blocks(8);
        let mut bt = BlockTable::new();
        assert_eq!(bt.token_capacity(), 0);
        // position 0 needs 1 block.
        bt.ensure(0, &mut pool).unwrap();
        assert_eq!(bt.blocks().len(), 1);
        assert_eq!(bt.locate(0), (bt.blocks()[0], 0));
        // position 20 needs 2 blocks (block 1 covers tokens 16..31).
        bt.ensure(20, &mut pool).unwrap();
        assert_eq!(bt.blocks().len(), 2);
        assert_eq!(bt.locate(20), (bt.blocks()[1], 4));
        assert_eq!(bt.token_capacity(), 32);
        // idempotent within capacity.
        let before = pool.free_blocks();
        bt.ensure(31, &mut pool).unwrap();
        assert_eq!(pool.free_blocks(), before);
        // clear returns both blocks.
        bt.clear(&mut pool);
        assert_eq!(pool.free_blocks(), 8);
    }

    #[test]
    fn block_table_ensure_reports_exhaustion() {
        let mut pool = KvPool::with_blocks(1);
        let mut bt = BlockTable::new();
        // needs 2 blocks for pos 16, only 1 available.
        assert_eq!(bt.ensure(16, &mut pool), Err(PoolExhausted));
        assert_eq!(bt.blocks().len(), 1); // grew as far as it could
    }

    #[test]
    fn share_prefix_retains_then_cow_privatizes() {
        let mut pool = KvPool::with_blocks(8);
        // "producer" builds a 2-block prefix.
        let mut producer = BlockTable::new();
        producer.ensure(BLOCK_TOKENS, &mut pool).unwrap(); // 2 blocks
        let shared: Vec<BlockId> = producer.blocks().to_vec();
        // "consumer" adopts it (zero copy) - refcounts go to 2.
        let mut consumer = BlockTable::new();
        consumer.share_prefix(&shared, &mut pool);
        assert_eq!(pool.refcount(shared[0]), 2);
        assert_eq!(pool.refcount(shared[1]), 2);
        // consumer writes into the first shared block -> CoW gives it a private copy.
        let owed = consumer.cow_at(0, &mut pool).unwrap();
        assert!(owed.is_some());
        let (src, dst) = owed.unwrap();
        assert_eq!(src, shared[0]);
        assert_ne!(consumer.blocks()[0], shared[0]); // repointed
        assert_eq!(consumer.blocks()[0], dst);
        assert_eq!(pool.refcount(shared[0]), 1); // producer keeps its copy
        // CoW is per-block: the SECOND logical block was untouched by the first's
        // CoW - still shared. Writing into it now privatizes it too.
        assert_eq!(consumer.blocks()[1], shared[1]);
        assert_eq!(pool.refcount(shared[1]), 2);
        assert!(consumer.cow_at(BLOCK_TOKENS, &mut pool).unwrap().is_some());
        assert_eq!(pool.refcount(shared[1]), 1);
    }

    #[test]
    fn window_ring_caps_allocation_and_cycles() {
        let mut pool = KvPool::with_blocks(64);
        let window = 100; // ceil(100/16)+1 = 7+1 = 8 blocks, regardless of max_ctx
        assert_eq!(WindowRing::blocks_for(window), 8);
        let ring = WindowRing::new(window, &mut pool).unwrap();
        assert_eq!(ring.ring_blocks(), 8);
        assert_eq!(pool.free_blocks(), 64 - 8);
        assert_eq!(ring.locate(3).1, 3); // offset within block
        // logical block 0 and logical block 8 (== ring_len) map to the same slot.
        assert_eq!(ring.locate(0).0, ring.locate(8 * BLOCK_TOKENS).0);
        assert_ne!(ring.locate(0).0, ring.locate(BLOCK_TOKENS).0); // adjacent differ
        let mut ring = ring;
        ring.clear(&mut pool);
        assert_eq!(pool.free_blocks(), 64);
    }

    #[test]
    fn window_ring_exhaustion_leaks_nothing() {
        let mut pool = KvPool::with_blocks(3); // need 8, only 3
        assert!(WindowRing::new(100, &mut pool).is_err());
        assert_eq!(pool.free_blocks(), 3); // partial alloc rolled back
    }
}
