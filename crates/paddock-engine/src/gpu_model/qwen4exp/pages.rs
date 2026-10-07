//! Flash-Next's live KV on a planned block pool.
//!
//! Every attention layer's K and V planes, and the QSA indexer's compressed
//! key plane beside them, are `[pool blocks, rows, width]` planes that one
//! set of block ids addresses: a 16-token page of slot `s` is pool block
//! `table[s][page]` in each of them (16 K/V rows, 16/cr = 4 index rows). The
//! walk's attention kernels read through the table (the paged twins of the
//! dense-strip kernels, bit-identical over the same keys - pack slots
//! 688-696 and the older twins), so nothing about a page's contents depends
//! on where it sits.
//!
//! The prefix cache lives on the same pool (the qwen35 / nemotron design):
//! the radix holds a sequence's pages by refcount, a resume adopts them
//! (`adopt`: no copy), and the recurrent-state checkpoints a resume restores
//! are pool pages too (`PagedRadix::set_state_paged`). The plan guarantees
//! full context for every slot plus the live turns' checkpoints, and buys
//! the rest of the checkpoints it wants while the grant affords them; pages
//! no live sequence holds are cache, taken back (`back`, via the radix's
//! `make_room`: dead KV, then the stalest checkpoint, then LRU KV) when a
//! live sequence needs them. An idle slot's pages go back to the pool
//! (`release`) - the radix keeps what it filed.

use cudarc::driver::CudaSlice;

use crate::gpu::GpuExecutor;
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::kv_plan;
use crate::kv_pool::{BLOCK_TOKENS, BlockId, BlockTable, KvPool};
use crate::paged_radix::PagedRadix;

/// Graph pools and the allocations the plan does not see, beside the pool.
/// qwen35's measured residual (768 MiB, its `graph pools + headroom`); the
/// operator's `graph_scratch_mib` override applies here too. Everything
/// else this lane allocates is either in the ledger before the grant is read
/// (weights, per-slot state, walk scratch, the prefix side store) or sizes
/// itself from what is left afterwards (the MTP head, the expert slot cache).
const GRAPH_POOLS_HEADROOM_MIB: u64 = 768;

/// Pool rows a page holds in the QSA index plane: one compressed key per
/// `QSA_BLOCK` tokens.
pub(super) const IDX_ROWS_PER_PAGE: usize = BLOCK_TOKENS / crate::gpu::qsa::QSA_BLOCK;

/// Blocks a slot's table spans at this context.
pub(super) fn blocks_per_slot(max_tokens: usize) -> usize {
    max_tokens.div_ceil(BLOCK_TOKENS)
}

/// Recurrent-state checkpoints the plan wants per slot beyond the live
/// turns' (qwen35's six: two prompt cuts and a reply cut, two turns deep).
const CKPTS_PER_SLOT: u64 = 6;

/// What the plan decided for the pool.
pub(super) struct PoolPlan {
    /// blocks to allocate, the spare included
    pub(super) blocks: usize,
    /// (checkpoint index space, pages one checkpoint draws) - None with the
    /// prefix cache off
    pub(super) ckpt: Option<(u32, usize)>,
}

/// Plan the pool against the grant: full context for every slot and, with
/// the prefix cache on (`ckpt_bytes`: one checkpoint's size), the live turns'
/// checkpoint pages - refusing a config that cannot back those, with the
/// arithmetic - plus the checkpoints it wants beyond them while the grant
/// affords them, and one spare block.
pub(super) fn plan_pool(
    e: &GpuExecutor,
    max_tokens: usize,
    slots: usize,
    n_attn: usize,
    kv_row_bytes: usize,
    idx_row_bytes: usize,
    ckpt_bytes: Option<u64>,
) -> Result<PoolPlan, GpuModelError> {
    let bps = blocks_per_slot(max_tokens);
    // one block across every attention layer: its K and V pages and its
    // index rows - which is also one page's checkpoint payload
    let block_bytes =
        (n_attn * (2 * BLOCK_TOKENS * kv_row_bytes + IDX_ROWS_PER_PAGE * idx_row_bytes)) as u64;
    let graph_headroom =
        kv_plan::graph_scratch_override_mib().unwrap_or(GRAPH_POOLS_HEADROOM_MIB) << 20;
    let mut reserves = vec![
        kv_plan::Reserve::new("graph pools + headroom", graph_headroom),
        kv_plan::Reserve::new("the pool's spare block", block_bytes),
    ];
    let (mut ckpt_blocks, mut retention, mut ppc) = (0, 0, None);
    if let Some(cb) = ckpt_bytes {
        let p = cb.div_ceil(block_bytes.max(1)) as usize;
        let (must, want) = crate::ckpt_pages::page_demand(slots, p, CKPTS_PER_SLOT);
        (ckpt_blocks, retention, ppc) = (must, want, Some(p));
        // a walk's in-walk cuts stage flat (STAGED_CUTS a walk) before they
        // commit into pages
        let staged = super::prefix::STAGED_CUTS as u64;
        reserves.push(kv_plan::Reserve::new("checkpoint staging", staged * cb));
    }
    let demand = kv_plan::Demand {
        family: "qwen4exp",
        max_ctx: max_tokens,
        slots,
        blocks_per_slot: bps,
        block_bytes,
        ckpt_blocks,
        retention_blocks: retention,
        reserves,
        // the tables are sized for full context and a slot keeps its pages
        // between requests - a pool short of that would queue invisibly
        when_short: kv_plan::WhenShort::Refuse,
        ..Default::default()
    };
    // a missing reading is an error, never permission
    let grant = e.vram_headroom().ok_or_else(|| {
        GpuModelError::Config(
            "qwen4exp: the driver did not report free VRAM, so the KV pool cannot be planned"
                .into(),
        )
    })?;
    let plan = demand
        .plan(grant)
        .map_err(|w| GpuModelError::WontFit(w.message))?;
    plan.report(&demand, grant);
    // The index space is bookkeeping, not memory: how many checkpoints exist
    // at once is whatever pages the live contexts leave free, so it allows as
    // many as the pool could ever hold (the qwen35 / nemotron sizing). It
    // used to be the plan's guaranteed + wanted count - 16 at two slots -
    // and that cap, not memory, is what bound: a 2 x 262K agentic soak stole
    // a checkpoint on every allocation with ~20K pool pages (68 checkpoints'
    // worth) free, and a conversation's newest turns lost the cut their next
    // turn resumes from (2026-09-30: 61 steals, 0 free indices each time).
    let ckpt = ppc.map(|p| ((plan.pool_blocks / p).max(1) as u32, p));
    Ok(PoolPlan {
        blocks: plan.pool_blocks + 1,
        ckpt,
    })
}

/// The pool's bookkeeping and the table the walks read.
pub(super) struct KvPages {
    pool: KvPool,
    tables: Vec<BlockTable>,
    /// host mirror of `d_tab`, `[slots * bps]`. An entry a slot has not
    /// backed names the pool's spare block - a block no slot owns, allocated
    /// first and never freed. No kernel of this lane reads a table entry past
    /// the pages under a row's keys; the spare keeps every entry a real
    /// address anyway.
    host: Vec<BlockId>,
    /// the table every trunk attention launch reads, `[slots * bps]` u32.
    /// Address-stable for the whole serve: captured decode graphs bake the
    /// pointer and read the contents at replay, so a table that grew is
    /// uploaded before the walk or replay that needs it (`sync`).
    pub(super) d_tab: CudaSlice<u32>,
    pub(super) bps: usize,
    dirty: bool,
    /// the entry an unbacked table slot names (see `host`)
    spare: BlockId,
    /// the prefix radix over this pool's pages, its checkpoints paged; None
    /// with the prefix cache off
    pub(super) radix: Option<PagedRadix>,
}

impl KvPages {
    pub(super) fn new(
        e: &GpuExecutor,
        slots: usize,
        max_tokens: usize,
        plan: &PoolPlan,
    ) -> Result<Self, GpuModelError> {
        let bps = blocks_per_slot(max_tokens);
        let mut pool = KvPool::with_blocks(plan.blocks as u32);
        let spare = pool.alloc().map_err(|_| GpuModelError::PoolExhausted)?;
        let host = vec![spare; slots * bps];
        let d_tab = e.to_device_u32(&host)?;
        let radix = plan.ckpt.map(|(n, ppc)| {
            let mut r = PagedRadix::new();
            r.set_state_paged(n, ppc);
            r
        });
        Ok(Self {
            pool,
            tables: (0..slots).map(|_| BlockTable::new()).collect(),
            host,
            d_tab,
            bps,
            dirty: false,
            spare,
            radix,
        })
    }

    /// Back `slot`'s positions `[0, upto)`. Tables only grow: a slot's pages
    /// outlive its request (see the module note).
    pub(super) fn back(&mut self, slot: usize, upto: usize) -> Result<(), GpuModelError> {
        if upto == 0 {
            return Ok(());
        }
        if upto > self.bps * BLOCK_TOKENS {
            // past the table's stride it would back the next slot's entries
            return Err(GpuModelError::ContextExceeded {
                got: upto,
                max: self.bps * BLOCK_TOKENS,
            });
        }
        let had = self.tables[slot].blocks().len();
        loop {
            match self.tables[slot].ensure(upto - 1, &mut self.pool) {
                Ok(()) => break,
                // a dry pool: what the radix holds is cache - dead KV, then
                // the stalest checkpoint, then LRU KV (promised context
                // outranks cache; the plan backs full context, so this
                // always finds room)
                Err(_) => {
                    let want = self.pool.free_blocks() + 1;
                    let shed = self
                        .radix
                        .as_mut()
                        .is_some_and(|r| r.make_room(&mut self.pool, want, 0));
                    if !shed {
                        return Err(GpuModelError::PoolExhausted);
                    }
                }
            }
        }
        let now = self.tables[slot].blocks().len();
        if now > had {
            let base = slot * self.bps;
            self.host[base + had..base + now]
                .copy_from_slice(&self.tables[slot].blocks()[had..now]);
            self.dirty = true;
        }
        Ok(())
    }

    /// Point `slot`'s table at `shared` - a cached prefix's pages, retained,
    /// not copied - after releasing whatever it held.
    pub(super) fn adopt(&mut self, slot: usize, shared: &[BlockId]) {
        self.tables[slot].clear(&mut self.pool);
        self.tables[slot].share_prefix(shared, &mut self.pool);
        self.rewrite_row(slot);
    }

    /// Give `slot`'s pages back (the radix keeps the ones it filed).
    pub(super) fn release(&mut self, slot: usize) {
        if self.tables[slot].blocks().is_empty() {
            return;
        }
        self.tables[slot].clear(&mut self.pool);
        self.rewrite_row(slot);
    }

    /// `slot`'s mirror row from its table, the unbacked tail naming the spare.
    fn rewrite_row(&mut self, slot: usize) {
        let base = slot * self.bps;
        let row = &mut self.host[base..base + self.bps];
        let b = self.tables[slot].blocks();
        row[..b.len()].copy_from_slice(b);
        row[b.len()..].fill(self.spare);
        self.dirty = true;
    }

    /// The radix and the pool it retains from, together (every radix call
    /// that files, adopts, attaches or evicts takes the pool).
    pub(super) fn radix_pool(&mut self) -> Option<(&mut PagedRadix, &mut KvPool)> {
        let pool = &mut self.pool;
        self.radix.as_mut().map(|r| (r, pool))
    }

    /// Upload the table if a `back` grew it. Called by every walk's staging,
    /// after its `back`s and before the walk or replay reads the table.
    pub(super) fn sync(&mut self, e: &GpuExecutor) -> Result<(), GpuModelError> {
        if self.dirty {
            e.upload_u32(&self.host, &mut self.d_tab)?;
            self.dirty = false;
        }
        Ok(())
    }

    /// `slot`'s pool blocks in page order - the pages its positions
    /// `[0, 16 * len)` live in.
    pub(super) fn blocks(&self, slot: usize) -> &[BlockId] {
        self.tables[slot].blocks()
    }
}

/// An identity table for a dense `[slots * bps * 16, width]` plane - the MTP
/// head's own KV, which the same paged kernels read: page `j` of slot `s` is
/// block `s * bps + j`, i.e. row `s * bps * 16 + pos`.
pub(super) fn identity_table(
    e: &GpuExecutor,
    slots: usize,
    bps: usize,
) -> Result<CudaSlice<u32>, GpuModelError> {
    let ids: Vec<u32> = (0..(slots * bps).max(1) as u32).collect();
    Ok(e.to_device_u32(&ids)?)
}
