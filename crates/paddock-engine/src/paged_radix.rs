//! Zero-copy radix prefix cache over the paged `KvPool` (P5c).
//!
//! **This module is the CPU-side bookkeeping only** (the same discipline as
//! `kv_pool` / P1): it maps block-aligned token prefixes to `KvPool` block ids so
//! a new sequence that shares a prefix ADOPTS the cached blocks (refcount++, via
//! [`crate::kv_pool::BlockTable::share_prefix`]) instead of recomputing or copying
//! them. It owns no device memory and touches no kernel - the blocks it names are
//! the same `KvPool` blocks the slots write, which is the whole point of
//! zero-copy sharing (vs. `prefix_cache::RadixKvCache`, which keeps its own store
//! and COPIES).
//!
//! Refcount lifecycle (all through `KvPool`):
//! - `insert` RETAINS each new cached block -> the tree holds one reference.
//! - a sharing slot RETAINS via `BlockTable::share_prefix` -> +1 per slot.
//! - a slot finishing RELEASES via `BlockTable::clear` (free-on-completion, P5b).
//! - [`PagedRadix::evict_lru`] RELEASES the tree's reference on the LRU leaf.
//! - a block returns to the free-list only at refcount 0 (no tree node, no slot).
//!
//! Only **full 16-token blocks** are cached: a block-aligned prefix means the
//! adopting slot never writes into a shared block (its own writes start at the
//! next block boundary), so **no copy-on-write is needed** for this path. The
//! token-granular partial-tail reuse (needs `BlockTable::cow_at`) and the DeltaNet
//! recurrent-state checkpoints (hybrid resume) are the device-wiring follow-up
//! (P5c-P2); this module is the allocator/tree bookkeeping they build on.

use crate::kv_pool::{BLOCK_TOKENS, BlockId, KvPool};
use crate::kv_tier::LogicalKey;
use std::collections::HashMap;

/// FNV-1a over a block's tokens - the child key under its parent. (Was
/// mirrored from the dense cache's hash_block; that cache is gone and this
/// is the one definition of block identity now.) Exact tokens are
/// also compared on match, so a collision costs a miss, never a wrong reuse.
fn hash_block(tokens: &[u32]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &t in tokens {
        h ^= t as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

struct Node {
    parent: u32,
    /// the `KvPool` block this node caches (retained while the node is alive).
    block: BlockId,
    /// hash of this node's 16 tokens (its key in `parent.children`).
    key: u64,
    /// the exact tokens, for the hash-collision check on match.
    tokens: Vec<u32>,
    children: HashMap<u64, u32>,
    last_used: u64,
    alive: bool,
    /// This prefix has been MATCHED again after being cached, i.e. it recurs.
    ///
    /// The signal is the page match, deliberately not "a checkpoint was handed
    /// to a resume". Keying on successful resumes is chicken-and-egg: a prefix
    /// can only prove itself by hitting, and thrash is precisely what stops it
    /// hitting. Measured on a c32 leg with the resume-keyed
    /// version: 158 requests per leg matched 176 tokens of pages and found
    /// `ckpt None` - exactly the prefixes that deserved protection, and every
    /// one of them invisible to the flag, so admission control never fired once.
    ///
    /// Survives the checkpoint being stolen: the proof is about the PREFIX, not
    /// the state blob. See `protect_proven`.
    recurred: bool,
    /// P5c hybrid resume: index of the DeltaNet recurrent-state checkpoint for
    /// the prefix ending at this node (a block-boundary position), or `None`.
    /// The device state blob lives in the model's paged state pool at this index;
    /// this is CPU-side bookkeeping only.
    state_blk: Option<u32>,
    /// When this node's checkpoint was last written or handed to a resume -
    /// the checkpoint's OWN recency, separate from `last_used`. A match bumps
    /// every node on its path, so on a long conversation the checkpoints of
    /// turns long past look as fresh as the latest one and path recency cannot
    /// tell them apart. The paged backing steals by this instead (SGLang's
    /// MambaRadixCache keeps the same two lists: KV leaf-to-root, states from
    /// any node).
    ckpt_used: u64,
    /// KV tier content-chain key for the prefix ending at this node
    /// `parent.tkey.child(tokens)`, rooted in the cache
    /// namespace via [`PagedRadix::set_tier_root`]. `None` when the tier is
    /// off or the node predates arming - such nodes simply never demote.
    tkey: Option<LogicalKey>,
}

/// The prefix-cache hit for a prompt: the shared KV block ids to adopt, plus the
/// deepest DeltaNet state checkpoint along the matched path (its block-boundary
/// position and the state-pool index) for a hybrid resume.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct PagedMatch {
    pub blocks: Vec<BlockId>,
    pub ckpt: Option<(usize, u32)>,
    /// The deepest matched node (0 = nothing matched): the one childless node
    /// on the matched path, so sparing it spares the path (see
    /// [`PagedRadix::evict_lru_sparing`]).
    pub tail: u32,
}

/// One block on the LRU leaf's root-to-leaf path (see
/// [`PagedRadix::lru_leaf_path`]). `depth` is 1-based: the prefix ending at
/// this block spans `depth` full 16-token blocks.
#[derive(Debug, Clone, Copy)]
pub struct LruPathEntry {
    pub node: u32,
    pub depth: usize,
    pub block: BlockId,
    pub tkey: Option<LogicalKey>,
    /// State-checkpoint index at this boundary (hybrid families) - what the
    /// tier's demote arm captures before eviction would recycle it.
    pub state_blk: Option<u32>,
}

/// A radix tree of block-aligned token prefixes over the shared `KvPool`. Node 0
/// is the dummy root (no block).
pub struct PagedRadix {
    nodes: Vec<Node>,
    free_nodes: Vec<u32>,
    clock: u64,
    /// CPU free-list of DeltaNet state-checkpoint indices (into the model's paged
    /// state pool). Empty until `set_state_capacity` - text-only / non-hybrid
    /// models never checkpoint state.
    state_free: Vec<u32>,
    /// Admission control on the state pool: once a checkpoint belongs to a
    /// prefix that has recurred, stop stealing it - let the pool fill and hold.
    ///
    /// Plain LRU steal is pathological exactly where this cache lives. When the
    /// distinct-prefix working set is a little LARGER than the pool and requests
    /// cycle through it, every arrival evicts the entry that would have been the
    /// next hit - the classic cyclic-thrash degenerate case, hit rate ~0 rather
    /// than the ~capacity/working-set you would expect. Measured on the qwen3.8
    /// c32 leg: 128 distinct prefixes, 88 checkpoint slots,
    /// **0 usable resumes out of 224 requests** on the first leg, while 180
    /// later requests matched 176 tokens of KV pages and found `ckpt None` -
    /// their state had been stolen before they came back. On a hybrid model
    /// matched pages without the recurrent state are worthless: those requests
    /// re-prefill anyway, having paid the match and the adopt.
    ///
    /// Holding the recurring set makes it stick, so the cache serves a stable
    /// `capacity/working-set` fraction instead of churning every entry out just
    /// before its next hit; the refused admissions also skip their state
    /// snapshot, which is the dominant write cost (~170 MiB per checkpoint at
    /// 27B - 48 GDN layers of state + conv window).
    ///
    /// The trade is adaptivity: a resident set that has all recurred never
    /// yields to a newly hot prefix. That is the right trade only where the
    /// pool is smaller than the working set, which is why this is opt-in -
    /// qwen35 arms it from `PADDOCK_CKPT_PROTECT`. Sizing the pool to the
    /// working set is the better fix where the memory exists.
    protect_proven: bool,
    /// Counters for the admission policy, read by the engine's prefix-stats
    /// witness: (state writes, LRU steals, refused admissions).
    st_writes: u64,
    st_steals: u64,
    st_refused: u64,
    /// `Some` when checkpoints live in the KV pool's own pages instead of a
    /// fixed device pool (issue #33, see [`PagedState`]).
    paged: Option<PagedState>,
}

/// Checkpoints as cache, not reservation (issue #33).
///
/// A fixed checkpoint pool is carved out of the grant before KV is sized, so
/// its floor alone can refuse a context that would otherwise fit (16 x ~150
/// MiB on a 27B). Paged, each state index owns `pages_per_ckpt` blocks of the
/// KV pool itself: pages a live context is not using hold checkpoints, and
/// the context takes them back as it grows. This is vLLM's hybrid KV cache
/// manager (one pool, one eviction order for KV and mamba state) rather than
/// SGLang's separate-pool-plus-elastic-resize - with paddock's batched copies
/// a checkpoint scattered over pages costs a longer descriptor list, not a
/// new kernel.
///
/// The index API stays what callers already use; the pool enters where pages
/// are drawn (`*_with_pool`). An index returned without a pool at hand (the
/// tier's completion paths) parks its pages in `released` until the next call
/// that has one.
struct PagedState {
    pages_per_ckpt: usize,
    /// Pool pages each state index owns (empty = it holds none).
    pages: Vec<Vec<BlockId>>,
    /// Pages of indices recycled without the pool; returned by [`PagedRadix::reclaim`].
    released: Vec<BlockId>,
}

impl Default for PagedRadix {
    fn default() -> Self {
        Self::new()
    }
}

impl PagedRadix {
    pub fn new() -> Self {
        Self {
            nodes: vec![Node {
                parent: 0,
                block: 0,
                key: 0,
                tokens: Vec::new(),
                children: HashMap::new(),
                last_used: 0,
                alive: true,
                recurred: false,
                state_blk: None,
                ckpt_used: 0,
                tkey: None,
            }],
            free_nodes: Vec::new(),
            clock: 0,
            state_free: Vec::new(),
            protect_proven: false,
            st_writes: 0,
            st_steals: 0,
            st_refused: 0,
            paged: None,
        }
    }

    /// Enable checkpoints that live in pool pages (see [`PagedState`]): up to
    /// `max_ckpts` indices, each drawing `pages_per_ckpt` blocks from the pool
    /// when it is written. `max_ckpts` only bounds the bookkeeping - how many
    /// exist at once is whatever the pool can spare.
    pub fn set_state_paged(&mut self, max_ckpts: u32, pages_per_ckpt: usize) {
        self.state_free = (0..max_ckpts).rev().collect();
        self.paged = Some(PagedState {
            pages_per_ckpt: pages_per_ckpt.max(1),
            pages: vec![Vec::new(); max_ckpts as usize],
            released: Vec::new(),
        });
    }

    /// True when checkpoints live in pool pages.
    pub fn state_is_paged(&self) -> bool {
        self.paged.is_some()
    }

    /// Pool pages one checkpoint draws (0 for the flat backing).
    pub fn pages_per_ckpt(&self) -> usize {
        self.paged.as_ref().map_or(0, |p| p.pages_per_ckpt)
    }

    /// The pool pages holding checkpoint `idx`, in blob order - empty for the
    /// flat backing, where the blob lives at a fixed offset instead.
    pub fn state_pages(&self, idx: u32) -> &[BlockId] {
        self.paged
            .as_ref()
            .and_then(|p| p.pages.get(idx as usize))
            .map_or(&[], Vec::as_slice)
    }

    /// Return the pages of indices recycled without a pool (see
    /// [`Self::recycle_state`]). Every pool-holding entry point drains first;
    /// callers under pressure call it before counting free blocks.
    pub fn reclaim(&mut self, pool: &mut KvPool) -> usize {
        let Some(p) = self.paged.as_mut() else {
            return 0;
        };
        let n = p.released.len();
        for b in p.released.drain(..) {
            pool.release(b);
        }
        n
    }

    /// Arm state-pool admission control (see `protect_proven`). Call at pool
    /// setup, next to `set_state_capacity`.
    pub fn set_protect_proven(&mut self, on: bool) {
        self.protect_proven = on;
    }

    /// (state writes, LRU steals, refused admissions) since boot.
    pub fn state_stats(&self) -> (u64, u64, u64) {
        (self.st_writes, self.st_steals, self.st_refused)
    }

    /// Enable DeltaNet state checkpoints (hybrid models): `n` state-pool indices
    /// become available for `attach_state`. Idempotent-ish - resets the free-list
    /// to `0..n` (call once at pool setup, alongside the device state pool alloc).
    pub fn set_state_capacity(&mut self, n: u32) {
        self.state_free = (0..n).rev().collect();
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    /// The block ids of the longest block-aligned prefix of `tokens` already
    /// cached (one per 16-token block, in order). Bumps LRU on the matched path.
    /// Keeps at least one token unmatched to prefill, so a prompt equal to a
    /// cached sequence still has a token to run.
    pub fn match_prefix(&mut self, tokens: &[u32]) -> Vec<BlockId> {
        let cap = tokens.len().saturating_sub(1);
        let full = cap / BLOCK_TOKENS;
        let mut node = 0u32;
        let mut blocks = Vec::new();
        for bi in 0..full {
            let chunk = &tokens[bi * BLOCK_TOKENS..(bi + 1) * BLOCK_TOKENS];
            let h = hash_block(chunk);
            let Some(&child) = self.nodes[node as usize].children.get(&h) else {
                break;
            };
            if self.nodes[child as usize].tokens != chunk {
                break; // hash collision - treat as a miss
            }
            blocks.push(self.nodes[child as usize].block);
            let t = self.tick();
            self.nodes[child as usize].last_used = t;
            node = child;
        }
        blocks
    }

    /// The full prefix-cache hit for `tokens`: the shared KV block ids AND the
    /// deepest DeltaNet state checkpoint along the matched path. A hybrid resume
    /// needs both - the KV blocks to adopt and the recurrent state at that
    /// block-boundary position. LRU-bumped like `match_prefix`.
    pub fn match_full(&mut self, tokens: &[u32]) -> PagedMatch {
        let cap = tokens.len().saturating_sub(1);
        let full = cap / BLOCK_TOKENS;
        let mut node = 0u32;
        let mut blocks = Vec::new();
        let mut ckpt = None;
        let mut ckpt_node = 0u32;
        for bi in 0..full {
            let chunk = &tokens[bi * BLOCK_TOKENS..(bi + 1) * BLOCK_TOKENS];
            let h = hash_block(chunk);
            let Some(&child) = self.nodes[node as usize].children.get(&h) else {
                break;
            };
            if self.nodes[child as usize].tokens != chunk {
                break;
            }
            blocks.push(self.nodes[child as usize].block);
            if let Some(sb) = self.nodes[child as usize].state_blk {
                ckpt = Some(((bi + 1) * BLOCK_TOKENS, sb));
                ckpt_node = child;
            }
            // Reaching this node at all means the prefix came back - mark it
            // whether or not its state survived. See `Node::recurred`.
            self.nodes[child as usize].recurred = true;
            let t = self.tick();
            self.nodes[child as usize].last_used = t;
            node = child;
        }
        // Only the checkpoint a resume would take counts as used; the older
        // ones the path passed through keep aging.
        if ckpt.is_some() {
            let t = self.tick();
            self.nodes[ckpt_node as usize].ckpt_used = t;
        }
        PagedMatch {
            blocks,
            ckpt,
            tail: node,
        }
    }

    /// Claim a free state-pool index without attaching it - the tier's aux
    /// restore lands the blob first and attaches after (an attach-then-fill
    /// order would leave a garbage checkpoint visible on failure). Same
    /// steal/protect policy as `attach_state`. Undo with
    /// [`Self::recycle_state`].
    pub fn reserve_state_slot(&mut self) -> Option<u32> {
        if self.paged.is_some() {
            return None; // pages come from the pool: reserve_state_slot_with_pool
        }
        if self.state_free.is_empty() && self.count_state() == 0 {
            return None; // state capacity never enabled
        }
        self.alloc_state()
    }

    /// [`Self::reserve_state_slot`] for either backing: under the paged one the
    /// index's pages are drawn from `pool` here, before the blob lands.
    pub fn reserve_state_slot_with_pool(&mut self, pool: &mut KvPool) -> Option<u32> {
        if self.paged.is_none() {
            return self.reserve_state_slot();
        }
        self.alloc_state_paged(pool, 0)
    }

    /// Detach the checkpoint at boundary `pos` of `tokens` (the reverse of
    /// `attach_state_at`) and return its pool index for the caller to
    /// recycle. `None` if the node does not exist or holds no checkpoint.
    /// The node and its KV page stay.
    pub fn detach_state_at(&mut self, tokens: &[u32], pos: usize) -> Option<u32> {
        let node = self.node_at(tokens, pos)?;
        self.nodes[node as usize].state_blk.take()
    }

    /// Detach the checkpoint at boundary `pos` of `tokens` only if it is
    /// `idx`; true when it was (the caller recycles `idx`). How a rolling
    /// reply checkpoint is dropped: since it was attached the pool may have
    /// stolen it and handed the index to another node, and another slot may
    /// have checkpointed the node again - a plain detach would take THAT
    /// checkpoint off the node, and the caller, holding the other index,
    /// could only leak it.
    pub fn detach_state_if(&mut self, tokens: &[u32], pos: usize, idx: u32) -> bool {
        let Some(node) = self.node_at(tokens, pos) else {
            return false;
        };
        let n = &mut self.nodes[node as usize];
        if n.state_blk != Some(idx) {
            return false;
        }
        n.state_blk = None;
        true
    }

    /// The checkpoint at exactly boundary `pos` of `tokens`, if one is
    /// attached there, marked used - for a resume that must start at that
    /// boundary rather than at the deepest checkpoint a match offers (an
    /// exact re-send repeating its first run's walk).
    pub fn resume_state_at(&mut self, tokens: &[u32], pos: usize) -> Option<u32> {
        let node = self.node_at(tokens, pos)?;
        let idx = self.nodes[node as usize].state_blk?;
        let t = self.tick();
        self.nodes[node as usize].ckpt_used = t;
        Some(idx)
    }

    /// The cached node ending exactly at block boundary `pos` of `tokens`.
    /// Unlike a match, which keeps a token back for the prefill to run and
    /// so never reaches the node at `tokens.len()`, this walks all the way.
    /// No LRU bump.
    fn node_at(&self, tokens: &[u32], pos: usize) -> Option<u32> {
        let want = pos / BLOCK_TOKENS;
        if want == 0 || !pos.is_multiple_of(BLOCK_TOKENS) || tokens.len() < pos {
            return None;
        }
        let mut node = 0u32;
        for bi in 0..want {
            let chunk = &tokens[bi * BLOCK_TOKENS..(bi + 1) * BLOCK_TOKENS];
            let h = hash_block(chunk);
            let child = *self.nodes[node as usize].children.get(&h)?;
            if self.nodes[child as usize].tokens != chunk {
                return None;
            }
            node = child;
        }
        Some(node)
    }

    /// Attach a RESERVED state index to the cached node ending at block
    /// boundary `pos`. False - and the caller recycles the index - if the
    /// node is missing (evicted between publication and now) or already
    /// checkpointed.
    pub fn attach_state_at(&mut self, tokens: &[u32], pos: usize, idx: u32) -> bool {
        let want = pos / BLOCK_TOKENS;
        if want == 0 || !pos.is_multiple_of(BLOCK_TOKENS) {
            return false;
        }
        let mut node = 0u32;
        for bi in 0..want {
            let chunk = &tokens[bi * BLOCK_TOKENS..(bi + 1) * BLOCK_TOKENS];
            let h = hash_block(chunk);
            let Some(&child) = self.nodes[node as usize].children.get(&h) else {
                return false;
            };
            if self.nodes[child as usize].tokens != chunk {
                return false;
            }
            node = child;
        }
        if node == 0 || self.nodes[node as usize].state_blk.is_some() {
            return false;
        }
        self.nodes[node as usize].state_blk = Some(idx);
        let t = self.tick();
        self.nodes[node as usize].ckpt_used = t;
        true
    }

    /// Attach a DeltaNet state checkpoint to the cached node ending at block-
    /// boundary `pos` (a `BLOCK_TOKENS` multiple), returning the state-pool index
    /// for the model to write the state blob into. `None` if `pos` isn't a cached
    /// node, already has a checkpoint, or state capacity is off. Steals the LRU
    /// node's checkpoint when the state pool is exhausted (that node + its KV page
    /// stay - only its checkpoint moves).
    pub fn attach_state(&mut self, tokens: &[u32], pos: usize) -> Option<u32> {
        if self.paged.is_some() {
            return None; // pages come from the pool: attach_state_with_pool
        }
        if self.state_free.is_empty() && self.count_state() == 0 {
            return None; // state capacity never enabled
        }
        let node = self.checkpointable_node(tokens, pos)?;
        let sb = self.alloc_state()?;
        self.nodes[node as usize].state_blk = Some(sb);
        let t = self.tick();
        self.nodes[node as usize].ckpt_used = t;
        Some(sb)
    }

    /// [`Self::attach_state`] for either backing: under the paged one the
    /// checkpoint's pages are drawn from `pool` - out of free pages, then dead
    /// KV, then the stalest resident checkpoint - and the node being
    /// checkpointed is never the one evicted to make that room.
    pub fn attach_state_with_pool(
        &mut self,
        tokens: &[u32],
        pos: usize,
        pool: &mut KvPool,
    ) -> Option<u32> {
        if self.paged.is_none() {
            return self.attach_state(tokens, pos);
        }
        let node = self.checkpointable_node(tokens, pos)?;
        let sb = self.alloc_state_paged(pool, node)?;
        self.nodes[node as usize].state_blk = Some(sb);
        let t = self.tick();
        self.nodes[node as usize].ckpt_used = t;
        Some(sb)
    }

    /// The cached node ending at block boundary `pos` of `tokens`, if it
    /// exists and holds no checkpoint yet.
    fn checkpointable_node(&self, tokens: &[u32], pos: usize) -> Option<u32> {
        let want = pos / BLOCK_TOKENS;
        if want == 0 || !pos.is_multiple_of(BLOCK_TOKENS) {
            return None;
        }
        let mut node = 0u32;
        for bi in 0..want {
            let chunk = &tokens[bi * BLOCK_TOKENS..(bi + 1) * BLOCK_TOKENS];
            let h = hash_block(chunk);
            let child = *self.nodes[node as usize].children.get(&h)?;
            if self.nodes[child as usize].tokens != chunk {
                return None;
            }
            node = child;
        }
        if node == 0 || self.nodes[node as usize].state_blk.is_some() {
            return None;
        }
        Some(node)
    }

    /// A state index with its pages drawn from `pool` (paged backing). Pages
    /// come from free blocks first, then dead KV (see
    /// [`Self::evict_dead_leaves`]), then by stealing the stalest resident
    /// checkpoint - whose pages go back to the pool rather than being
    /// overwritten in place, because the tier may still be reading them
    /// under a pin. `spare` is the node the caller is about to checkpoint.
    fn alloc_state_paged(&mut self, pool: &mut KvPool, spare: u32) -> Option<u32> {
        self.reclaim(pool);
        let k = self.paged.as_ref().map_or(1, |p| p.pages_per_ckpt);
        loop {
            if pool.free_blocks() >= k
                && let Some(idx) = self.state_free.pop()
            {
                let pages: Vec<BlockId> = (0..k)
                    .map(|_| pool.alloc().expect("free blocks counted above"))
                    .collect();
                if let Some(p) = self.paged.as_mut() {
                    p.pages[idx as usize] = pages;
                }
                self.st_writes += 1;
                return Some(idx);
            }
            if pool.free_blocks() < k && self.evict_dead_leaves(pool, k, spare) > 0 {
                continue;
            }
            // Out of pages or out of indices: take a resident checkpoint.
            let Some(victim) = self.state_victim(spare) else {
                self.st_refused += 1;
                return None;
            };
            if self.protect_proven && self.nodes[victim].recurred {
                // see alloc_state: hold a fully-recurred resident set
                self.st_refused += 1;
                return None;
            }
            self.log_steal(victim, "checkpoint alloc", pool);
            let idx = self.nodes[victim]
                .state_blk
                .take()
                .expect("victims hold a checkpoint");
            self.drop_state_pages(idx, pool);
            self.state_free.push(idx);
            self.st_steals += 1;
        }
    }

    /// Debug witness for a checkpoint steal: which boundary lost its state,
    /// how stale it was, and why. Depth is a parent walk, so it is only
    /// computed when someone is listening.
    fn log_steal(&self, victim: usize, why: &str, pool: &KvPool) {
        if !tracing::enabled!(tracing::Level::DEBUG) {
            return;
        }
        let mut depth = 0usize;
        let mut node = victim as u32;
        while node != 0 {
            depth += 1;
            node = self.nodes[node as usize].parent;
        }
        tracing::debug!(
            "ckpt steal ({why}): boundary {} tok, idle {} ticks, recurred {}, free idx {}, \
             free pages {}",
            depth * BLOCK_TOKENS,
            self.clock.saturating_sub(self.nodes[victim].ckpt_used),
            self.nodes[victim].recurred,
            self.state_free.len(),
            pool.free_blocks()
        );
    }

    /// The checkpoint to give up first: never-recurred first under
    /// `protect_proven`, then least recently written or resumed. The paged
    /// backing orders by the checkpoint's own recency (`ckpt_used`); the
    /// flat one keeps the node's path recency it has always used.
    fn state_victim(&self, spare: u32) -> Option<usize> {
        let protect = self.protect_proven;
        let paged = self.paged.is_some();
        self.nodes
            .iter()
            .enumerate()
            .filter(|(i, n)| *i != 0 && *i as u32 != spare && n.alive && n.state_blk.is_some())
            .min_by_key(|(_, n)| {
                (
                    protect && n.recurred,
                    if paged { n.ckpt_used } else { n.last_used },
                )
            })
            .map(|(i, _)| i)
    }

    /// Give index `idx`'s pages back to the pool (paged backing; no-op flat).
    fn drop_state_pages(&mut self, idx: u32, pool: &mut KvPool) {
        if let Some(p) = self.paged.as_mut() {
            for b in p.pages[idx as usize].drain(..) {
                pool.release(b);
            }
        }
    }

    /// Evict dead KV until `pool` has `want` free blocks, returning how many
    /// nodes went. Dead means a childless node with no checkpoint whose page
    /// only the tree holds: on a hybrid a prefix resumes only AT a
    /// checkpoint, so KV below the deepest one on its path is recomputed on
    /// every resume anyway - it is the cheapest thing in the pool to lose.
    /// Each eviction walks up while the parent turns dead too, so a
    /// conversation whose checkpoints are gone is trimmed in one pass instead
    /// of one rescan per node. `spare` (and so its path) is never taken.
    pub fn evict_dead_leaves(&mut self, pool: &mut KvPool, want: usize, spare: u32) -> usize {
        let mut gone = 0;
        while pool.free_blocks() < want {
            let mut dead: Vec<(u64, u32)> = self
                .nodes
                .iter()
                .enumerate()
                .filter(|(i, _)| self.is_dead_leaf(*i as u32, pool, spare))
                .map(|(i, n)| (n.last_used, i as u32))
                .collect();
            if dead.is_empty() {
                break;
            }
            dead.sort_unstable();
            for (_, leaf) in dead {
                let mut v = leaf;
                while pool.free_blocks() < want && self.is_dead_leaf(v, pool, spare) {
                    let parent = self.nodes[v as usize].parent;
                    self.evict_node(v, pool);
                    gone += 1;
                    v = parent;
                }
                if pool.free_blocks() >= want {
                    break;
                }
            }
        }
        gone
    }

    fn is_dead_leaf(&self, v: u32, pool: &KvPool, spare: u32) -> bool {
        let Some(n) = self.nodes.get(v as usize) else {
            return false;
        };
        v != 0
            && v != spare
            && n.alive
            && n.children.is_empty()
            && n.state_blk.is_none()
            && pool.refcount(n.block) == 1
    }

    /// Make `want` blocks free for a LIVE context, in the paged backing's
    /// order: released pages, dead KV, the stalest checkpoint (whose path then
    /// turns dead and trims next round), and only then an LRU leaf. Unlike a
    /// checkpoint's own allocation this ignores `protect_proven` - context a
    /// running request was promised always outranks cache. False when nothing
    /// left in the tree can free a page (everything is held by live slots).
    pub fn make_room(&mut self, pool: &mut KvPool, want: usize, spare: u32) -> bool {
        self.reclaim(pool);
        loop {
            if pool.free_blocks() >= want {
                return true;
            }
            if self.evict_dead_leaves(pool, want, spare) > 0 {
                continue;
            }
            if let Some(v) = self.state_victim_any(spare) {
                self.log_steal(v, "live context needs pages", pool);
                let idx = self.nodes[v].state_blk.take().expect("victim holds one");
                self.drop_state_pages(idx, pool);
                self.state_free.push(idx);
                self.st_steals += 1;
                continue;
            }
            if self.evict_lru_sparing(spare, pool).is_none() {
                return false;
            }
            tracing::debug!(
                "make_room: LRU KV leaf evicted for a live context (free pages {}, want {want})",
                pool.free_blocks()
            );
        }
    }

    /// Like [`Self::state_victim`] but blind to `protect_proven`.
    fn state_victim_any(&self, spare: u32) -> Option<usize> {
        let paged = self.paged.is_some();
        self.nodes
            .iter()
            .enumerate()
            .filter(|(i, n)| *i != 0 && *i as u32 != spare && n.alive && n.state_blk.is_some())
            .min_by_key(|(_, n)| if paged { n.ckpt_used } else { n.last_used })
            .map(|(i, _)| i)
    }

    /// The stalest resident checkpoint as a path entry (node, depth, tier key,
    /// index) - what the tier's pressure pass demotes before it touches any
    /// hot path. `None` when no checkpoint is resident.
    pub fn stalest_state(&self) -> Option<LruPathEntry> {
        let v = self.state_victim_any(0)? as u32;
        let mut depth = 0usize;
        let mut node = v;
        while node != 0 {
            depth += 1;
            node = self.nodes[node as usize].parent;
        }
        let n = &self.nodes[v as usize];
        Some(LruPathEntry {
            node: v,
            depth,
            block: n.block,
            tkey: n.tkey,
            state_blk: n.state_blk,
        })
    }

    fn count_state(&self) -> usize {
        self.nodes.iter().filter(|n| n.state_blk.is_some()).count()
    }

    /// A free state-pool index, stealing the LRU checkpointed node's if exhausted
    /// (that node + its KV page survive; only the checkpoint is reclaimed).
    ///
    /// Under `protect_proven`, victims are ordered never-recurred first and a
    /// recurred checkpoint is not stolen at all - the pool fills, then holds.
    /// A refused admission also skips the caller's state snapshot, which is the
    /// dominant write cost, so refusing is cheaper than serving.
    fn alloc_state(&mut self) -> Option<u32> {
        if let Some(b) = self.state_free.pop() {
            self.st_writes += 1;
            return Some(b);
        }
        // Plain LRU unless `protect_proven`: the never-recurred-first order
        // is the opt-in policy's ("hold the proven set"), and applied by
        // default it steals the NEWEST useful checkpoint - an agentic
        // session's just-committed cut has not recurred yet, its previous
        // turn's stale cut has - so a full pool cycled every session back
        // to the shared prefix (GB10 2026-09-11).
        let protect = self.protect_proven;
        let victim = self
            .nodes
            .iter()
            .enumerate()
            .filter(|(i, n)| *i != 0 && n.alive && n.state_blk.is_some())
            .min_by_key(|(_, n)| (protect && n.recurred, n.last_used))
            .map(|(i, _)| i)?;
        if self.protect_proven && self.nodes[victim].recurred {
            // Every resident checkpoint belongs to a prefix that came back, so
            // stealing one only moves the miss around - the cyclic-thrash
            // trade LRU makes by default. Hold the resident set instead.
            self.st_refused += 1;
            return None;
        }
        self.st_writes += 1;
        self.st_steals += 1;
        self.nodes[victim].state_blk.take()
    }

    /// Cache `tokens`' full 16-token blocks from a slot's `blocks` (logical block
    /// `i` backs tokens `[i*16, i*16+16)`). A block not already in the tree gets a
    /// node and is RETAINED in `pool` (the tree's reference); a block already
    /// cached just bumps LRU (its node keeps the earlier block - the caller's
    /// duplicate can be released by the slot as usual). `blocks.len()` must cover
    /// `tokens.len()/16` full blocks.
    pub fn insert(&mut self, tokens: &[u32], blocks: &[BlockId], pool: &mut KvPool) {
        self.reclaim(pool);
        let full = (tokens.len() / BLOCK_TOKENS).min(blocks.len());
        let mut node = 0u32;
        for bi in 0..full {
            let chunk = &tokens[bi * BLOCK_TOKENS..(bi + 1) * BLOCK_TOKENS];
            let h = hash_block(chunk);
            if let Some(&child) = self.nodes[node as usize].children.get(&h) {
                if self.nodes[child as usize].tokens == chunk {
                    let t = self.tick();
                    self.nodes[child as usize].last_used = t;
                    node = child;
                    continue;
                }
                // hash collision with different tokens: don't cache further (the
                // trie can't disambiguate) - stop extending this path.
                break;
            }
            pool.retain(blocks[bi]); // the tree now holds a reference
            let t = self.tick();
            let nid = self.new_node(Node {
                parent: node,
                block: blocks[bi],
                key: h,
                tokens: chunk.to_vec(),
                children: HashMap::new(),
                last_used: t,
                alive: true,
                recurred: false,
                state_blk: None,
                ckpt_used: 0,
                tkey: self.nodes[node as usize].tkey.map(|k| k.child(chunk)),
            });
            self.nodes[node as usize].children.insert(h, nid);
            node = nid;
        }
    }

    /// Arm KV-tier chain keys: every node inserted from now
    /// on carries `parent.tkey.child(tokens)`, rooted here. Call once at pool
    /// setup, before any insert - nodes created unarmed never demote.
    pub fn set_tier_root(&mut self, root: LogicalKey) {
        self.nodes[0].tkey = Some(root);
    }

    /// The current LRU childless leaf's full root-to-leaf path - what the
    /// tier's demote arm inspects before eviction (it needs every block of a
    /// run alive while the gather reads it; `evict_lru` would already have
    /// released the leaf). Entries are root-first; `depth` is 1-based (=
    /// the number of 16-token blocks the prefix ending here spans).
    /// The `max` least-recently-used leaf paths, oldest first - the
    /// write-through mirror pass walks these: the chains
    /// eviction would pick first are the ones worth pre-storing.
    pub fn lru_leaf_paths(&self, max: usize) -> Vec<Vec<LruPathEntry>> {
        let mut leaves: Vec<(u64, u32)> = self
            .nodes
            .iter()
            .enumerate()
            .filter(|(i, n)| *i != 0 && n.alive && n.children.is_empty())
            .map(|(i, n)| (n.last_used, i as u32))
            .collect();
        leaves.sort_unstable();
        leaves
            .into_iter()
            .take(max)
            .map(|(_, leaf)| self.path_to(leaf))
            .collect()
    }

    /// Every live checkpoint attachment: (depth-blocks, tier chain key,
    /// state index). Bounded by the state-pool capacity, so the blob
    /// write-through can scan it every pass - the LRU leaves are exactly
    /// the chains whose checkpoints have already recycled.
    pub fn state_attachments(&self) -> Vec<(usize, Option<LogicalKey>, u32)> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(i, n)| *i != 0 && n.alive && n.state_blk.is_some())
            .map(|(i, n)| {
                let mut d = 0usize;
                let mut node = i as u32;
                while node != 0 {
                    d += 1;
                    node = self.nodes[node as usize].parent;
                }
                (d, n.tkey, n.state_blk.expect("filtered"))
            })
            .collect()
    }

    pub fn lru_leaf_path(&self) -> Vec<LruPathEntry> {
        let Some(leaf) = self
            .nodes
            .iter()
            .enumerate()
            .filter(|(i, n)| *i != 0 && n.alive && n.children.is_empty())
            .min_by_key(|(_, n)| n.last_used)
            .map(|(i, _)| i as u32)
        else {
            return Vec::new();
        };
        self.path_to(leaf)
    }

    /// Root-first path entries ending at `node` (depth 1-based) - the KV a
    /// checkpoint at `node` resumes over, which is what the tier has to hold
    /// beside the blob for that checkpoint to restore from RAM.
    pub fn path_entries(&self, node: u32) -> Vec<LruPathEntry> {
        self.path_to(node)
    }

    /// Root-first path entries for `leaf` (depth 1-based) - shared by the
    /// single-victim LRU walk and the mirror pass's multi-leaf variant.
    fn path_to(&self, leaf: u32) -> Vec<LruPathEntry> {
        let mut path = Vec::new();
        let mut node = leaf;
        while node != 0 {
            let n = &self.nodes[node as usize];
            path.push(LruPathEntry {
                node,
                depth: 0,
                block: n.block,
                tkey: n.tkey,
                state_blk: n.state_blk,
            });
            node = n.parent;
        }
        path.reverse();
        for (i, e) in path.iter_mut().enumerate() {
            e.depth = i + 1;
        }
        path
    }

    /// How many full blocks of `tokens` are currently cached (read-only - no
    /// LRU bump, no recurrence marking). The tier's restore publication uses
    /// it to verify a chain is attachable / already published.
    pub fn chain_depth(&self, tokens: &[u32]) -> usize {
        let full = tokens.len() / BLOCK_TOKENS;
        let mut node = 0u32;
        let mut depth = 0;
        for bi in 0..full {
            let chunk = &tokens[bi * BLOCK_TOKENS..(bi + 1) * BLOCK_TOKENS];
            let h = hash_block(chunk);
            let Some(&child) = self.nodes[node as usize].children.get(&h) else {
                break;
            };
            if self.nodes[child as usize].tokens != chunk {
                break;
            }
            depth = bi + 1;
            node = child;
        }
        depth
    }

    /// Publish a restored run: attach `blocks` as chain
    /// blocks `[start_block, start_block + blocks.len())` of `tokens`,
    /// RETAINING each in `pool` exactly like `insert`. Returns `false` -
    /// and touches nothing - unless the chain is present through
    /// `start_block` (the prefix may have been evicted while the restore
    /// was in flight; publishing into a hole would attach content at wrong
    /// positions). Positions already cached keep their existing blocks
    /// (the caller releases its surplus copies, same as `insert`).
    pub fn insert_extension(
        &mut self,
        tokens: &[u32],
        start_block: usize,
        blocks: &[BlockId],
        pool: &mut KvPool,
    ) -> bool {
        if self.chain_depth(tokens) < start_block {
            return false;
        }
        let end = (start_block + blocks.len()).min(tokens.len() / BLOCK_TOKENS);
        // walk to start_block (present per the check), then insert onward -
        // same node lifecycle as `insert`
        let mut node = 0u32;
        for bi in 0..end {
            let chunk = &tokens[bi * BLOCK_TOKENS..(bi + 1) * BLOCK_TOKENS];
            let h = hash_block(chunk);
            if let Some(&child) = self.nodes[node as usize].children.get(&h) {
                if self.nodes[child as usize].tokens == chunk {
                    let t = self.tick();
                    self.nodes[child as usize].last_used = t;
                    node = child;
                    continue;
                }
                return bi > start_block; // collision - stop; partial publish stands
            }
            debug_assert!(bi >= start_block, "chain_depth guaranteed presence");
            pool.retain(blocks[bi - start_block]);
            let t = self.tick();
            let tkey = self.nodes[node as usize].tkey.map(|k| k.child(chunk));
            let nid = self.new_node(Node {
                parent: node,
                block: blocks[bi - start_block],
                key: h,
                tokens: chunk.to_vec(),
                children: HashMap::new(),
                last_used: t,
                alive: true,
                recurred: false,
                state_blk: None,
                ckpt_used: 0,
                tkey,
            });
            self.nodes[node as usize].children.insert(h, nid);
            node = nid;
        }
        true
    }

    /// Detach a node's state checkpoint without recycling its pool index -
    /// the tier's demote arm claims it so the blob's bytes survive until the
    /// store's gather has read them; the index returns via
    /// [`Self::recycle_state`] at store completion. `None` if the node has
    /// no checkpoint (or was already claimed).
    pub fn take_state(&mut self, node: u32) -> Option<u32> {
        self.nodes.get_mut(node as usize)?.state_blk.take()
    }

    /// Return a state index claimed by [`Self::take_state`] to the free list
    /// (the tier calls this once the demote's store completed - or failed;
    /// either way the blob region is no longer read).
    pub fn recycle_state(&mut self, idx: u32) {
        // Paged: the tier recycles from completion paths that hold no pool,
        // so the pages wait in `released` for the next `reclaim`.
        if let Some(p) = self.paged.as_mut()
            && let Some(pages) = p.pages.get_mut(idx as usize)
        {
            p.released.append(pages);
        }
        self.state_free.push(idx);
    }

    /// Evict a SPECIFIC live childless leaf (the tier's demote arm walks the
    /// LRU path itself and evicts bottom-up as it goes). Returns the released
    /// block, `None` if `node` is not currently evictable.
    pub fn evict_leaf(&mut self, node: u32, pool: &mut KvPool) -> Option<BlockId> {
        let n = self.nodes.get(node as usize)?;
        if node == 0 || !n.alive || !n.children.is_empty() {
            return None;
        }
        Some(self.evict_node(node, pool))
    }

    /// Evict the least-recently-used childless leaf, RELEASING its block back to
    /// `pool` (the tree's reference; the block frees only if no slot still holds
    /// it). Returns the evicted block id, or `None` if the tree has no leaf. Call
    /// on pool exhaustion to reclaim cached-but-idle prefixes.
    pub fn evict_lru(&mut self, pool: &mut KvPool) -> Option<BlockId> {
        let victim = self
            .nodes
            .iter()
            .enumerate()
            .filter(|(i, n)| *i != 0 && n.alive && n.children.is_empty())
            .min_by_key(|(_, n)| n.last_used)
            .map(|(i, _)| i as u32)?;
        Some(self.evict_node(victim, pool))
    }

    /// `evict_lru`, never taking node `spare` - for a caller about to EXTEND
    /// the path a match just returned (`PagedMatch::tail`). Sparing the tail
    /// spares the whole path: every other node on it has the next one as a
    /// child, so none of them is ever a leaf while the tail stands. Where the
    /// tree holds the only reference to a page (a side store copied in and
    /// out, not pages a live sequence also holds), evicting the caller's own
    /// tail would free pages its match still names - the next alloc hands
    /// them back out, and re-inserting the match maps two nodes onto one page.
    pub fn evict_lru_sparing(&mut self, spare: u32, pool: &mut KvPool) -> Option<BlockId> {
        let victim = self
            .nodes
            .iter()
            .enumerate()
            .filter(|(i, n)| *i != 0 && *i as u32 != spare && n.alive && n.children.is_empty())
            .min_by_key(|(_, n)| n.last_used)
            .map(|(i, _)| i as u32)?;
        Some(self.evict_node(victim, pool))
    }

    /// Shared teardown for `evict_lru` / `evict_leaf`. Caller guarantees the
    /// node is a live childless non-root leaf.
    fn evict_node(&mut self, victim: u32, pool: &mut KvPool) -> BlockId {
        let (parent, key, blk) = {
            let n = &self.nodes[victim as usize];
            (n.parent, n.key, n.block)
        };
        self.nodes[parent as usize].children.remove(&key);
        self.nodes[victim as usize].alive = false;
        self.nodes[victim as usize].children = HashMap::new();
        if let Some(sb) = self.nodes[victim as usize].state_blk.take() {
            self.drop_state_pages(sb, pool);
            self.state_free.push(sb); // reclaim the checkpoint index
        }
        self.free_nodes.push(victim);
        pool.release(blk);
        blk
    }

    /// Number of cached blocks (alive non-root nodes).
    pub fn cached_blocks(&self) -> usize {
        self.nodes.iter().skip(1).filter(|n| n.alive).count()
    }

    /// Blocks the tree could reclaim under pressure: alive nodes whose block
    /// only the tree references (refcount 1 - not shared with a live slot).
    /// Admission accounting adds this to the pool's free count so the prefix
    /// cache behaves as reclaimable capacity, not a reservation - otherwise a
    /// retention-heavy workload (salted benches, many one-shot prompts) drives
    /// `free` to ~0 and the admission watermark serializes the whole server
    /// behind slot completions (found live: gemma4 c8 TTFT 3.3 s -> 52 s).
    pub fn evictable_blocks(&self, pool: &KvPool) -> usize {
        let kv = self
            .nodes
            .iter()
            .skip(1)
            .filter(|n| n.alive && pool.refcount(n.block) == 1)
            .count();
        // Paged checkpoints are cache too: every page an attached checkpoint
        // holds alone frees when `make_room` takes it, and released pages
        // are free the moment anyone reclaims them.
        let ckpt = self.paged.as_ref().map_or(0, |p| {
            let attached: usize = self
                .nodes
                .iter()
                .skip(1)
                .filter(|n| n.alive)
                .filter_map(|n| n.state_blk)
                .map(|i| {
                    p.pages[i as usize]
                        .iter()
                        .filter(|&&b| pool.refcount(b) == 1)
                        .count()
                })
                .sum();
            attached + p.released.len()
        });
        kv + ckpt
    }

    fn new_node(&mut self, n: Node) -> u32 {
        if let Some(id) = self.free_nodes.pop() {
            self.nodes[id as usize] = n;
            id
        } else {
            self.nodes.push(n);
            (self.nodes.len() - 1) as u32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_pool::{BlockTable, KvPool};

    // 16-token block of a constant value (distinct per block seed).
    fn block_toks(seed: u32) -> Vec<u32> {
        (0..BLOCK_TOKENS as u32).map(|i| seed * 100 + i).collect()
    }

    /// A slot prefills `n_blocks` fresh blocks (alloc from the pool) and returns
    /// its block table.
    fn prefill(pool: &mut KvPool, n_blocks: usize) -> BlockTable {
        let mut t = BlockTable::new();
        t.ensure(n_blocks * BLOCK_TOKENS - 1, pool).expect("alloc");
        t
    }

    #[test]
    fn empty_match_is_empty() {
        let mut r = PagedRadix::new();
        assert!(r.match_prefix(&block_toks(1)).is_empty());
    }

    #[test]
    fn insert_then_match_hits_and_tree_holds_a_ref() {
        let mut pool = KvPool::with_blocks(16);
        let mut r = PagedRadix::new();
        // slot prefills 2 blocks, then inserts them
        let table = prefill(&mut pool, 2);
        let mut toks: Vec<u32> = block_toks(1);
        toks.extend(block_toks(2));
        toks.push(999); // +1 so 2 full blocks are cacheable (cap keeps 1 token)
        r.insert(&toks, table.blocks(), &mut pool);
        assert_eq!(r.cached_blocks(), 2);
        // both blocks now held by the slot AND the tree
        for &b in table.blocks() {
            assert_eq!(pool.refcount(b), 2, "slot + tree");
        }
        // a fresh prompt with the same prefix matches both blocks
        let got = r.match_prefix(&toks);
        assert_eq!(got, table.blocks());
    }

    #[test]
    fn match_keeps_at_least_one_token() {
        let mut pool = KvPool::with_blocks(16);
        let mut r = PagedRadix::new();
        let table = prefill(&mut pool, 2);
        let toks: Vec<u32> = [block_toks(1), block_toks(2)].concat(); // exactly 32
        r.insert(&toks, table.blocks(), &mut pool);
        // matching the same 32 tokens keeps the last block unmatched (cap=31 ->
        // 1 full block), so there is always a token left to prefill.
        assert_eq!(r.match_prefix(&toks).len(), 1);
    }

    #[test]
    fn branching_prefix_shares_the_common_block() {
        let mut pool = KvPool::with_blocks(16);
        let mut r = PagedRadix::new();
        // seq A: block1, block2
        let ta = prefill(&mut pool, 2);
        let a: Vec<u32> = [block_toks(1), block_toks(2), vec![7]].concat();
        r.insert(&a, ta.blocks(), &mut pool);
        // seq B shares block1 then diverges to block3
        let b_prefix: Vec<u32> = [block_toks(1), block_toks(3), vec![7]].concat();
        let shared = r.match_prefix(&b_prefix);
        assert_eq!(
            shared,
            &ta.blocks()[..1],
            "shares only the common first block"
        );
        // B adopts the shared block into its own table (refcount++)
        let mut tb = BlockTable::new();
        tb.share_prefix(&shared, &mut pool);
        assert_eq!(pool.refcount(shared[0]), 3, "A-slot + tree + B-slot");
    }

    #[test]
    fn evict_lru_releases_tree_ref_and_frees_when_unshared() {
        let mut pool = KvPool::with_blocks(4);
        let mut r = PagedRadix::new();
        let mut table = prefill(&mut pool, 2); // blocks used by the slot
        let toks: Vec<u32> = [block_toks(1), block_toks(2), vec![7]].concat();
        r.insert(&toks, table.blocks(), &mut pool);
        let free_after_insert = pool.free_blocks();
        // the slot finishes: its refs drop (free-on-completion). Blocks stay
        // pinned by the tree (refcount 1), so still not free.
        table.clear(&mut pool);
        assert_eq!(
            pool.free_blocks(),
            free_after_insert,
            "tree still pins them"
        );
        // evict both leaves -> their blocks return to the pool
        assert!(r.evict_lru(&mut pool).is_some());
        assert!(r.evict_lru(&mut pool).is_some());
        assert_eq!(r.cached_blocks(), 0);
        assert_eq!(pool.free_blocks(), 4, "all blocks back");
    }

    #[test]
    fn evict_lru_targets_the_least_recently_used() {
        let mut pool = KvPool::with_blocks(8);
        let mut r = PagedRadix::new();
        // two independent single-block prefixes
        let mut t1 = BlockTable::new();
        t1.ensure(BLOCK_TOKENS - 1, &mut pool).unwrap();
        let a: Vec<u32> = [block_toks(1), vec![7]].concat();
        r.insert(&a, t1.blocks(), &mut pool);
        let mut t2 = BlockTable::new();
        t2.ensure(BLOCK_TOKENS - 1, &mut pool).unwrap();
        let b: Vec<u32> = [block_toks(2), vec![7]].concat();
        r.insert(&b, t2.blocks(), &mut pool);
        // touch A (more recent) -> B is the LRU and must be evicted first
        let _ = r.match_prefix(&a);
        let evicted = r.evict_lru(&mut pool).unwrap();
        assert_eq!(evicted, t2.blocks()[0], "LRU (B) evicted, not A");
    }

    #[test]
    fn detach_state_at_frees_the_index_and_the_match_loses_the_checkpoint() {
        let mut pool = KvPool::with_blocks(16);
        let mut r = PagedRadix::new();
        r.set_state_capacity(1);
        let table = prefill(&mut pool, 3);
        let toks: Vec<u32> = [block_toks(1), block_toks(2), block_toks(3), vec![9]].concat();
        r.insert(&toks, table.blocks(), &mut pool);
        let idx = r.reserve_state_slot().expect("one index");
        assert!(r.attach_state_at(&toks, 2 * BLOCK_TOKENS, idx));
        assert_eq!(r.match_full(&toks).ckpt, Some((2 * BLOCK_TOKENS, idx)));
        // not a boundary / not a node: None, nothing changes
        assert_eq!(r.detach_state_at(&toks, 2 * BLOCK_TOKENS + 1), None);
        assert_eq!(r.detach_state_at(&[1, 2, 3], BLOCK_TOKENS), None);
        assert_eq!(r.detach_state_at(&toks, 2 * BLOCK_TOKENS), Some(idx));
        assert!(
            r.match_full(&toks).ckpt.is_none(),
            "checkpoint gone, page stays"
        );
        assert_eq!(r.match_full(&toks).blocks, table.blocks());
        // the pool was exhausted (capacity 1); recycling makes the index reusable
        r.recycle_state(idx);
        assert_eq!(r.reserve_state_slot(), Some(idx));
    }

    #[test]
    fn detach_state_if_takes_only_the_named_checkpoint() {
        let mut pool = KvPool::with_blocks(16);
        let mut r = PagedRadix::new();
        r.set_state_capacity(2);
        let table = prefill(&mut pool, 3);
        let toks: Vec<u32> = [block_toks(1), block_toks(2), block_toks(3), vec![9]].concat();
        r.insert(&toks, table.blocks(), &mut pool);
        let ours = r.reserve_state_slot().expect("index");
        let theirs = r.reserve_state_slot().expect("index");
        // the node was re-checkpointed under another index since ours
        assert!(r.attach_state_at(&toks, 2 * BLOCK_TOKENS, theirs));
        // a reply's own stream ends AT its checkpoint, which a match never
        // reaches - the drop must still find the node
        let reply = &toks[..2 * BLOCK_TOKENS];
        assert!(r.match_full(reply).ckpt.is_none());
        assert!(!r.detach_state_if(reply, 2 * BLOCK_TOKENS, ours));
        assert_eq!(
            r.match_full(&toks).ckpt,
            Some((2 * BLOCK_TOKENS, theirs)),
            "a foreign checkpoint stays attached"
        );
        assert!(
            !r.detach_state_if(&toks, BLOCK_TOKENS, theirs),
            "none there"
        );
        assert!(!r.detach_state_if(&toks, 2 * BLOCK_TOKENS + 1, theirs));
        assert!(r.detach_state_if(reply, 2 * BLOCK_TOKENS, theirs));
        assert!(r.match_full(&toks).ckpt.is_none());
        assert!(
            !r.detach_state_if(reply, 2 * BLOCK_TOKENS, theirs),
            "already gone"
        );
        assert_eq!(r.match_full(&toks).blocks, table.blocks(), "pages stay");
    }

    #[test]
    fn attach_state_and_match_full_returns_the_deepest_checkpoint() {
        let mut pool = KvPool::with_blocks(16);
        let mut r = PagedRadix::new();
        r.set_state_capacity(4);
        let table = prefill(&mut pool, 3);
        let toks: Vec<u32> = [block_toks(1), block_toks(2), block_toks(3), vec![9]].concat();
        r.insert(&toks, table.blocks(), &mut pool);
        // no state yet
        assert!(r.match_full(&toks).ckpt.is_none());
        // checkpoint at the 2-block boundary (pos 32)
        let sb = r.attach_state(&toks, 2 * BLOCK_TOKENS).expect("attach");
        // re-attaching the same node is a no-op
        assert!(r.attach_state(&toks, 2 * BLOCK_TOKENS).is_none());
        // match now reports the checkpoint (pos 32, index sb); blocks intact
        let m = r.match_full(&toks);
        assert_eq!(m.ckpt, Some((2 * BLOCK_TOKENS, sb)));
        assert_eq!(m.blocks, table.blocks());
    }

    #[test]
    fn no_state_capacity_means_no_checkpoints() {
        let mut pool = KvPool::with_blocks(8);
        let mut r = PagedRadix::new(); // state capacity not enabled
        let table = prefill(&mut pool, 2);
        let toks: Vec<u32> = [block_toks(1), block_toks(2), vec![9]].concat();
        r.insert(&toks, table.blocks(), &mut pool);
        assert!(r.attach_state(&toks, BLOCK_TOKENS).is_none());
        assert!(r.match_full(&toks).ckpt.is_none());
    }

    #[test]
    fn state_pool_exhaustion_steals_the_lru_checkpoint() {
        let mut pool = KvPool::with_blocks(16);
        let mut r = PagedRadix::new();
        r.set_state_capacity(1); // room for one checkpoint
        // two independent 1-block prefixes, each checkpointed at pos 16
        let mut t1 = BlockTable::new();
        t1.ensure(BLOCK_TOKENS - 1, &mut pool).unwrap();
        let a: Vec<u32> = [block_toks(1), vec![9]].concat();
        r.insert(&a, t1.blocks(), &mut pool);
        let s1 = r.attach_state(&a, BLOCK_TOKENS).expect("a state");
        let mut t2 = BlockTable::new();
        t2.ensure(BLOCK_TOKENS - 1, &mut pool).unwrap();
        let b: Vec<u32> = [block_toks(2), vec![9]].concat();
        r.insert(&b, t2.blocks(), &mut pool);
        // pool exhausted -> steals A's checkpoint (LRU), reusing the same index
        let s2 = r.attach_state(&b, BLOCK_TOKENS).expect("b steals a");
        assert_eq!(s1, s2, "reused the stolen index");
        // A no longer has a checkpoint; B does
        assert!(r.match_full(&a).ckpt.is_none());
        assert_eq!(r.match_full(&b).ckpt, Some((BLOCK_TOKENS, s2)));
    }

    /// Two 1-block prefixes and room for one checkpoint. `on_a` decides whether
    /// A is resumed (which marks it proven) before B asks for the slot.
    /// Returns (radix, a, b, a's original state index, B's attach result).
    /// Two 1-block prefixes and room for one checkpoint. `recur_a` decides
    /// whether A is matched again (which marks it recurred) before B asks for
    /// the slot. Returns (radix, a, b, a's state index, B's attach result).
    fn two_prefixes_one_slot(
        protect: bool,
        recur_a: bool,
    ) -> (PagedRadix, Vec<u32>, Vec<u32>, u32, Option<u32>) {
        let mut pool = KvPool::with_blocks(16);
        let mut r = PagedRadix::new();
        r.set_state_capacity(1);
        r.set_protect_proven(protect);
        let mut t1 = BlockTable::new();
        t1.ensure(BLOCK_TOKENS - 1, &mut pool).unwrap();
        let a: Vec<u32> = [block_toks(1), vec![9]].concat();
        r.insert(&a, t1.blocks(), &mut pool);
        let s1 = r.attach_state(&a, BLOCK_TOKENS).expect("a state");
        if recur_a {
            r.match_full(&a); // A comes back - this is what marks it recurred
        }
        let mut t2 = BlockTable::new();
        t2.ensure(BLOCK_TOKENS - 1, &mut pool).unwrap();
        let b: Vec<u32> = [block_toks(2), vec![9]].concat();
        r.insert(&b, t2.blocks(), &mut pool);
        let sb = r.attach_state(&b, BLOCK_TOKENS);
        (r, a, b, s1, sb)
    }

    #[test]
    fn protect_proven_refuses_to_evict_a_recurring_prefix() {
        let (mut r, a, b, s1, sb) = two_prefixes_one_slot(true, true);
        // A has come back once, so its checkpoint is the one about to be hit
        // again - under plain LRU B evicts exactly that. Refuse instead.
        assert!(
            sb.is_none(),
            "must not evict a recurring prefix's checkpoint"
        );
        assert_eq!(
            r.match_full(&a).ckpt,
            Some((BLOCK_TOKENS, s1)),
            "A survives"
        );
        assert!(r.match_full(&b).ckpt.is_none(), "B got no checkpoint");
        let (_, _, refused) = r.state_stats();
        assert_eq!(refused, 1);
    }

    #[test]
    fn protect_proven_still_steals_from_a_prefix_that_never_came_back() {
        // A was checkpointed on its first and only sighting, so it has shown no
        // recurrence and carries no protection: B takes its slot as before.
        let (mut r, a, b, s1, sb) = two_prefixes_one_slot(true, false);
        assert_eq!(sb, Some(s1), "reused the stolen index");
        assert!(r.match_full(&a).ckpt.is_none());
        assert_eq!(r.match_full(&b).ckpt, Some((BLOCK_TOKENS, s1)));
        let (_, steals, refused) = r.state_stats();
        assert_eq!((steals, refused), (1, 0));
    }

    #[test]
    fn protect_proven_is_off_by_default_and_steals() {
        // The unarmed path must behave exactly as it did before the policy.
        let (mut r, a, _b, s1, sb) = two_prefixes_one_slot(false, true);
        assert_eq!(
            sb,
            Some(s1),
            "default policy still steals from a recurring A"
        );
        assert!(r.match_full(&a).ckpt.is_none());
    }

    #[test]
    fn a_page_match_marks_recurrence_even_when_the_state_was_stolen() {
        // The regression that made the first version of this policy inert: a
        // prefix whose checkpoint is gone still matches its PAGES, and that is
        // the signal. Keyed on successful resumes instead, it stays invisible.
        let mut pool = KvPool::with_blocks(16);
        let mut r = PagedRadix::new();
        r.set_state_capacity(1);
        let mut t1 = BlockTable::new();
        t1.ensure(BLOCK_TOKENS - 1, &mut pool).unwrap();
        let a: Vec<u32> = [block_toks(1), vec![9]].concat();
        r.insert(&a, t1.blocks(), &mut pool);
        r.attach_state(&a, BLOCK_TOKENS).expect("a state");
        // B steals A's checkpoint (policy off), so A's pages survive but its
        // state does not.
        let mut t2 = BlockTable::new();
        t2.ensure(BLOCK_TOKENS - 1, &mut pool).unwrap();
        let b: Vec<u32> = [block_toks(2), vec![9]].concat();
        r.insert(&b, t2.blocks(), &mut pool);
        r.attach_state(&b, BLOCK_TOKENS).expect("b steals");
        let m = r.match_full(&a);
        assert!(m.ckpt.is_none(), "state gone");
        assert_eq!(
            m.blocks.len(),
            1,
            "pages still match - the recurrence signal"
        );
        // Armed, the roles now invert on that evidence alone: A has come back
        // and B never has, so A reclaims the slot from the one-hit-wonder.
        r.set_protect_proven(true);
        assert!(
            r.attach_state(&a, BLOCK_TOKENS).is_some(),
            "A reclaims from non-recurring B"
        );
        assert!(
            r.match_full(&b).ckpt.is_none(),
            "B lost the slot it never earned"
        );
        // ...and now that B has been matched too, A is protected from it.
        assert!(
            r.attach_state(&b, BLOCK_TOKENS).is_none(),
            "B refused against recurring A"
        );
        let (_, _, refused) = r.state_stats();
        assert_eq!(refused, 1);
    }

    #[test]
    fn evict_lru_reclaims_the_checkpoint_index() {
        let mut pool = KvPool::with_blocks(8);
        let mut r = PagedRadix::new();
        r.set_state_capacity(1);
        let table = prefill(&mut pool, 1);
        let toks: Vec<u32> = [block_toks(1), vec![9]].concat();
        r.insert(&toks, table.blocks(), &mut pool);
        r.attach_state(&toks, BLOCK_TOKENS).expect("state");
        // evicting the node reclaims its state index (free-list refilled), so a
        // fresh prefix can checkpoint again.
        assert!(r.evict_lru(&mut pool).is_some());
        let mut t2 = BlockTable::new();
        t2.ensure(BLOCK_TOKENS - 1, &mut pool).unwrap();
        let b: Vec<u32> = [block_toks(2), vec![9]].concat();
        r.insert(&b, t2.blocks(), &mut pool);
        assert!(
            r.attach_state(&b, BLOCK_TOKENS).is_some(),
            "index reclaimed"
        );
    }

    #[test]
    fn reinsert_of_cached_prefix_does_not_double_retain() {
        let mut pool = KvPool::with_blocks(8);
        let mut r = PagedRadix::new();
        let table = prefill(&mut pool, 1);
        let toks: Vec<u32> = [block_toks(1), vec![7]].concat();
        r.insert(&toks, table.blocks(), &mut pool);
        let rc = pool.refcount(table.blocks()[0]);
        // a second request with the same block re-inserts: the node already
        // exists, so no extra retain (the tree holds exactly one reference).
        r.insert(&toks, table.blocks(), &mut pool);
        assert_eq!(pool.refcount(table.blocks()[0]), rc, "no double-retain");
        assert_eq!(r.cached_blocks(), 1);
    }

    // -- paged checkpoints (issue #33) -------------------------------------

    /// Cache `n` blocks of `seed`'s chain from a slot that then finishes, so
    /// the tree holds the only reference to every page (what a finished
    /// conversation leaves behind). Returns the tokens (+1 so all `n` cache).
    fn cached(r: &mut PagedRadix, pool: &mut KvPool, seed: u32, n: usize) -> Vec<u32> {
        let mut t = prefill(pool, n);
        let toks: Vec<u32> = (0..n as u32)
            .flat_map(|i| block_toks(seed * 1000 + i))
            .chain([7])
            .collect();
        r.insert(&toks, t.blocks(), pool);
        t.clear(pool);
        toks
    }

    #[test]
    fn a_paged_checkpoint_owns_pool_pages_and_eviction_returns_them() {
        let mut pool = KvPool::with_blocks(16);
        let mut r = PagedRadix::new();
        r.set_state_paged(4, 3);
        let a = cached(&mut r, &mut pool, 1, 2);
        assert_eq!(pool.free_blocks(), 14);
        // the pool-less form cannot draw pages
        assert!(r.attach_state(&a, BLOCK_TOKENS).is_none());
        assert!(r.reserve_state_slot().is_none());
        let idx = r
            .attach_state_with_pool(&a, 2 * BLOCK_TOKENS, &mut pool)
            .expect("pages free");
        assert_eq!(r.state_pages(idx).len(), 3);
        assert_eq!(pool.free_blocks(), 11, "three pages drawn");
        assert_eq!(r.match_full(&a).ckpt, Some((2 * BLOCK_TOKENS, idx)));
        // evicting the checkpointed leaf gives back its page AND its 3 pages
        assert!(r.evict_lru(&mut pool).is_some());
        assert_eq!(pool.free_blocks(), 15);
        assert!(r.state_pages(idx).is_empty());
    }

    #[test]
    fn checkpoint_pages_come_from_dead_kv_before_any_checkpoint() {
        // 9 pages: A holds 2 KV + a 3-page checkpoint, B is 4 blocks of dead
        // KV. C's own block takes one of B's, its checkpoint the other three.
        let mut pool = KvPool::with_blocks(9);
        let mut r = PagedRadix::new();
        r.set_state_paged(4, 3);
        let a = cached(&mut r, &mut pool, 1, 2);
        let ia = r
            .attach_state_with_pool(&a, 2 * BLOCK_TOKENS, &mut pool)
            .expect("a");
        let _b = cached(&mut r, &mut pool, 2, 4);
        assert_eq!(pool.free_blocks(), 0);
        // C needs 3 pages: B's dead chain goes, A's checkpoint stays
        let c = cached_after_room(&mut r, &mut pool, 3, 1);
        let ic = r
            .attach_state_with_pool(&c, BLOCK_TOKENS, &mut pool)
            .expect("c");
        assert_eq!(r.match_full(&a).ckpt, Some((2 * BLOCK_TOKENS, ia)));
        assert_eq!(r.match_full(&c).ckpt, Some((BLOCK_TOKENS, ic)));
        assert_eq!(r.state_stats().1, 0, "nothing stolen");
    }

    /// `cached`, making room for the chain's own KV first the way a family's
    /// exhaustion loop would.
    fn cached_after_room(r: &mut PagedRadix, pool: &mut KvPool, seed: u32, n: usize) -> Vec<u32> {
        assert!(r.make_room(pool, n, 0), "room for the chain");
        cached(r, pool, seed, n)
    }

    #[test]
    fn the_stalest_checkpoint_goes_first_not_the_one_on_the_coldest_path() {
        // One conversation, two turns: turn 1's checkpoint sits INSIDE the
        // path the later match keeps bumping, so by path recency it would
        // look as fresh as turn 2's. By its own recency it is the older.
        let mut pool = KvPool::with_blocks(16);
        let mut r = PagedRadix::new();
        r.set_state_paged(4, 2);
        let conv = cached(&mut r, &mut pool, 1, 4);
        let old = r
            .attach_state_with_pool(&conv, BLOCK_TOKENS, &mut pool)
            .expect("turn 1");
        let new = r
            .attach_state_with_pool(&conv, 4 * BLOCK_TOKENS, &mut pool)
            .expect("turn 2");
        let _other = cached(&mut r, &mut pool, 2, 1);
        let other_ck = r
            .attach_state_with_pool(&_other, BLOCK_TOKENS, &mut pool)
            .expect("other");
        let _ = r.match_full(&conv); // the conversation comes back: turn 2 resumes
        // a live context needs pages: turn 1's checkpoint is the stalest
        let before = pool.free_blocks();
        assert!(r.make_room(&mut pool, before + 2, 0));
        assert_eq!(r.match_full(&conv).ckpt, Some((4 * BLOCK_TOKENS, new)));
        assert!(r.state_pages(old).is_empty(), "turn 1 gave its pages up");
        assert_eq!(
            r.state_pages(other_ck).len(),
            2,
            "the other chain kept its checkpoint"
        );
    }

    #[test]
    fn a_live_context_outranks_protected_checkpoints() {
        let mut pool = KvPool::with_blocks(8);
        let mut r = PagedRadix::new();
        r.set_state_paged(4, 4);
        r.set_protect_proven(true);
        let a = cached(&mut r, &mut pool, 1, 2);
        let ia = r
            .attach_state_with_pool(&a, 2 * BLOCK_TOKENS, &mut pool)
            .expect("a");
        let _ = r.match_full(&a); // recurred: protected against other checkpoints
        assert_eq!(pool.free_blocks(), 2);
        // ...but not against context a slot was promised
        assert!(r.make_room(&mut pool, 6, 0));
        assert!(r.state_pages(ia).is_empty());
    }

    #[test]
    fn a_recycled_index_parks_its_pages_until_a_pool_is_at_hand() {
        let mut pool = KvPool::with_blocks(16);
        let mut r = PagedRadix::new();
        r.set_state_paged(4, 3);
        let a = cached(&mut r, &mut pool, 1, 1);
        r.attach_state_with_pool(&a, BLOCK_TOKENS, &mut pool)
            .expect("a");
        let node = r.match_full(&a).tail;
        let idx = r.take_state(node).expect("tier claims it");
        let free = pool.free_blocks();
        r.recycle_state(idx); // completion path: no pool
        assert_eq!(pool.free_blocks(), free, "not yet");
        assert_eq!(
            r.evictable_blocks(&pool),
            1 + 3,
            "the page and the parked three"
        );
        assert_eq!(r.reclaim(&mut pool), 3);
        assert_eq!(pool.free_blocks(), free + 3);
    }

    #[test]
    fn a_stolen_checkpoints_pinned_pages_are_never_reused() {
        let mut pool = KvPool::with_blocks(8);
        let mut r = PagedRadix::new();
        r.set_state_paged(4, 3);
        let a = cached(&mut r, &mut pool, 1, 1);
        let ia = r
            .attach_state_with_pool(&a, BLOCK_TOKENS, &mut pool)
            .expect("a");
        let pinned: Vec<BlockId> = r.state_pages(ia).to_vec();
        for &b in &pinned {
            pool.retain(b); // a write-through store is reading them
        }
        let b = cached(&mut r, &mut pool, 2, 1);
        // 3 free: B's checkpoint takes those; a third chain must steal A's,
        // whose pages stay out of reach until the pin drops
        r.attach_state_with_pool(&b, BLOCK_TOKENS, &mut pool)
            .expect("b");
        let c = cached_after_room(&mut r, &mut pool, 3, 1);
        let ic = r.attach_state_with_pool(&c, BLOCK_TOKENS, &mut pool);
        for page in ic.map(|i| r.state_pages(i).to_vec()).unwrap_or_default() {
            assert!(!pinned.contains(&page), "reused a page under a pin");
        }
        for &p in &pinned {
            assert!(pool.refcount(p) >= 1, "the pin still holds it");
        }
    }

    #[test]
    fn making_room_for_a_checkpoint_never_evicts_the_node_being_checkpointed() {
        let mut pool = KvPool::with_blocks(4);
        let mut r = PagedRadix::new();
        r.set_state_paged(2, 2);
        // the target is dead KV (childless, no checkpoint, tree-only page) -
        // the one thing the dead-first order would reach for
        let a = cached(&mut r, &mut pool, 1, 1);
        let _b = cached(&mut r, &mut pool, 2, 1);
        assert_eq!(pool.free_blocks(), 2);
        let _ = r.match_full(&_b); // B is fresher, A the LRU dead leaf
        let _c = cached(&mut r, &mut pool, 3, 1);
        assert_eq!(pool.free_blocks(), 1);
        let ia = r
            .attach_state_with_pool(&a, BLOCK_TOKENS, &mut pool)
            .expect("room from another dead leaf");
        assert_eq!(r.match_full(&a).ckpt, Some((BLOCK_TOKENS, ia)));
    }

    /// A side store (the tree holds each page's only reference) extended past
    /// its capacity must never map two nodes onto one page. The publish
    /// pattern: match, make room, alloc the missing pages, insert match + new.
    #[test]
    fn extending_a_path_past_capacity_never_aliases_a_page() {
        let publish = |r: &mut PagedRadix, pool: &mut KvPool, toks: &[u32], sparing: bool| {
            let full = toks.len() / BLOCK_TOKENS;
            let m = r.match_full(toks);
            let want = full - m.blocks.len();
            while pool.free_blocks() < want {
                let freed = if sparing {
                    r.evict_lru_sparing(m.tail, pool)
                } else {
                    r.evict_lru(pool)
                };
                if freed.is_none() {
                    break;
                }
            }
            let need = want.min(pool.free_blocks());
            let new: Vec<BlockId> = (0..need).map(|_| pool.alloc().expect("free")).collect();
            let mut all = m.blocks.clone();
            all.extend_from_slice(&new);
            r.insert(toks, &all, pool);
            for b in new {
                pool.release(b);
            }
        };
        let alive_blocks = |r: &PagedRadix| -> Vec<BlockId> {
            r.nodes
                .iter()
                .skip(1)
                .filter(|n| n.alive)
                .map(|n| n.block)
                .collect()
        };
        // one conversation growing 3 blocks a turn through a 4-page store
        let conv: Vec<u32> = (0..9).flat_map(block_toks).collect();
        for sparing in [false, true] {
            let mut pool = KvPool::with_blocks(4);
            let mut r = PagedRadix::new();
            for turn in 1..=3 {
                publish(&mut r, &mut pool, &conv[..turn * 3 * BLOCK_TOKENS], sparing);
            }
            let blocks = alive_blocks(&r);
            let mut distinct = blocks.clone();
            distinct.sort_unstable();
            distinct.dedup();
            if sparing {
                assert_eq!(
                    distinct.len(),
                    blocks.len(),
                    "a page mapped twice: {blocks:?}"
                );
                assert!(blocks.len() <= 4, "more nodes than pages: {blocks:?}");
                // what it kept is the conversation's head, prefix-closed
                assert_eq!(r.match_full(&conv).blocks.len(), blocks.len());
            } else {
                // the unspared form evicts its own tail and re-adopts the ids
                assert!(
                    distinct.len() < blocks.len(),
                    "expected the aliasing: {blocks:?}"
                );
            }
        }
    }
}
