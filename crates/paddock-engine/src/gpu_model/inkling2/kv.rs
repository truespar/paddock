//! What every layer visit keeps per slot: its K / V rows and the four short
//! convolutions' carried inputs.
//!
//! The 8 full-attention visits (layers 5 and 23 once, 11 and 17 in each of
//! the loop's passes) page their rows in 16-row blocks through one block
//! table a slot - their positions are the same, so the table is shared. The
//! 40 sliding visits read only the 1024 newest keys and keep a ring a slot:
//! a power of two at least the window plus a prompt chunk, so a chunk's first
//! row still finds its whole window once the chunk's own rows landed. Blocks
//! come from one refcounted pool: a slot's table references them, and so
//! does the prefix cache's radix for a prompt it keeps (prefix.rs) - a block
//! goes back to the free list when nobody does.

use cudarc::driver::CudaSlice;
use paddock_models::inkling2 as ik;

use super::KV;
use super::draft::{FEAT_RING, NFEAT};
use crate::gpu::{GpuExecutor, INKLING2_KV_F32};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::kv_pool::{BlockTable, KvPool};

/// A visit's K and V rows.
pub(crate) struct Store {
    pub k: CudaSlice<u8>,
    pub v: CudaSlice<u8>,
}

/// The conv states of a visit, [slots][3][C] f32 each: K, V (C = 1024), the
/// attention branch and the MLP branch (C = 2304).
pub(crate) const CONV_K: usize = 0;
pub(crate) const CONV_V: usize = 1;
pub(crate) const CONV_ATTN: usize = 2;
pub(crate) const CONV_MLP: usize = 3;

pub(crate) struct Kv {
    /// Row class: 0 f16, 1 fp8 e4m3, 2 f32.
    pub dtype: u32,
    pub slots: usize,
    /// Rows a sliding visit's ring holds a slot.
    pub ring: usize,
    /// Blocks a slot may hold (max_ctx / 16, rounded up).
    pub bps: usize,
    pub stores: Vec<Store>,
    pub conv: Vec<[CudaSlice<f32>; 4]>,
    /// [slots][bps] block ids, and the host's copy.
    pub bt: CudaSlice<u32>,
    pub bt_host: Vec<u32>,
    /// The full visits' blocks, refcounted (slot tables + the prefix cache),
    /// and each slot's table.
    pub pool: KvPool,
    pub tables: Vec<BlockTable>,
    /// Rows each slot holds.
    pub lens: Vec<usize>,
    /// With the DSpark draft: its three layers' K / V (bf16; the full
    /// layer paged through `bt`, the local ones in rings) and each slot's
    /// last FEAT_RING rows of features.
    pub draft: Vec<Store>,
    pub feat_ring: Option<CudaSlice<f32>>,
    /// With the draft: per visit the 4 convs' verify stashes [slots][11][C]
    /// (the carried rows, then the run's inputs), and per slot its last
    /// verify run (first position, rows) not yet settled.
    pub stash: Vec<[CudaSlice<f32>; 4]>,
    pub spec_run: Vec<Option<(usize, usize)>>,
    /// A rollback's (slot, kept rows) table.
    pub spec_items: Option<CudaSlice<u32>>,
}

/// Whether visit `v` attends over the full context.
pub(crate) fn full_visit(v: usize) -> bool {
    ik::full_attention(ik::visit(v).0)
}

/// Bytes a K / V element takes in `dtype`.
fn elem_bytes(dtype: u32) -> usize {
    match dtype {
        0 => 2,
        1 => 1,
        _ => 4,
    }
}

/// Rows a sliding visit's ring holds a slot for prompt chunks of `chunk`.
pub(crate) fn ring_rows(chunk: usize) -> usize {
    (ik::WINDOW + chunk).next_power_of_two()
}

/// Bytes one pool block costs across the full visits (and the draft's
/// full layer).
pub(crate) fn block_bytes(dtype: u32, draft: bool) -> u64 {
    let full = (0..ik::VISITS).filter(|&v| full_visit(v)).count();
    let d = if draft { 16 * 2 * KV * 2 } else { 0 };
    (full * 16 * 2 * KV * elem_bytes(dtype) + d) as u64
}

/// Bytes one slot holds outside the pool: the sliding rings, the conv
/// states and its block-table row (and the draft's local rings and feature
/// ring).
pub(crate) fn slot_bytes(dtype: u32, chunk: usize, bps: usize, draft: bool) -> u64 {
    let sliding = (0..ik::VISITS).filter(|&v| !full_visit(v)).count();
    let rings = sliding * ring_rows(chunk) * 2 * KV * elem_bytes(dtype);
    let conv = ik::VISITS * 3 * (2 * KV + 2 * ik::WIDTH) * 4;
    let d = if draft {
        2 * ring_rows(chunk) * 2 * KV * 2
            + FEAT_RING * NFEAT * ik::WIDTH * 4
            + ik::VISITS * 11 * (2 * KV + 2 * ik::WIDTH) * 4
    } else {
        0
    };
    (rings + conv + bps * 4 + d) as u64
}

impl Kv {
    /// Stores for `slots` sequences of up to `max_ctx` rows, prompts in
    /// chunks of up to `chunk` rows, `pool_blocks` 16-row blocks shared by
    /// the full visits.
    pub fn alloc(
        exec: &GpuExecutor,
        slots: usize,
        max_ctx: usize,
        chunk: usize,
        dtype: u32,
        pool_blocks: usize,
        draft: bool,
    ) -> Result<Self, GpuModelError> {
        if dtype > INKLING2_KV_F32 {
            return Err(GpuModelError::Unsupported(format!(
                "inkling2: KV class {dtype}"
            )));
        }
        let eb = elem_bytes(dtype);
        let ring = ring_rows(chunk);
        let bps = max_ctx.div_ceil(16);
        let n_blocks = pool_blocks;
        let w = ik::WIDTH;
        let mut stores = Vec::with_capacity(ik::VISITS);
        let mut conv = Vec::with_capacity(ik::VISITS);
        for v in 0..ik::VISITS {
            let rows = if full_visit(v) {
                n_blocks * 16
            } else {
                slots * ring
            };
            stores.push(Store {
                k: exec.alloc_u8(rows * KV * eb)?,
                v: exec.alloc_u8(rows * KV * eb)?,
            });
            let z = |c: usize| exec.to_device(&vec![0.0f32; slots * 3 * c]);
            conv.push([z(KV)?, z(KV)?, z(w)?, z(w)?]);
        }
        // the draft's layers: 0 and 2 local (rings), 1 full (paged)
        let mut draft_stores = Vec::new();
        let mut feat_ring = None;
        if draft {
            for local in [true, false, true] {
                let rows = if local { slots * ring } else { n_blocks * 16 };
                draft_stores.push(Store {
                    k: exec.alloc_u8(rows * KV * 2)?,
                    v: exec.alloc_u8(rows * KV * 2)?,
                });
            }
            feat_ring = Some(exec.alloc(slots * FEAT_RING * NFEAT * w)?);
        }
        let stash = if draft {
            (0..ik::VISITS)
                .map(|_| {
                    let z = |c: usize| exec.alloc(slots * 11 * c);
                    Ok([z(KV)?, z(KV)?, z(w)?, z(w)?])
                })
                .collect::<Result<Vec<_>, crate::gpu::GpuError>>()?
        } else {
            Vec::new()
        };
        Ok(Self {
            dtype,
            slots,
            ring,
            bps,
            stores,
            conv,
            draft: draft_stores,
            feat_ring,
            stash,
            spec_run: vec![None; slots],
            spec_items: if draft {
                Some(exec.alloc_u32(2 * slots)?)
            } else {
                None
            },
            bt: exec.to_device_u32(&vec![0u32; slots * bps])?,
            bt_host: vec![0; slots * bps],
            pool: KvPool::with_blocks(n_blocks as u32),
            tables: (0..slots).map(|_| BlockTable::new()).collect(),
            lens: vec![0; slots],
        })
    }

    /// Back `slot`'s positions 0..upto with blocks (its table row uploaded
    /// when it grew). PoolExhausted when the pool has too few left.
    pub fn ensure(
        &mut self,
        exec: &GpuExecutor,
        slot: usize,
        upto: usize,
    ) -> Result<(), GpuModelError> {
        let want = upto.div_ceil(16);
        if want > self.bps {
            return Err(GpuModelError::Unsupported(format!(
                "inkling2: {upto} rows past the slot's {} positions",
                self.bps * 16
            )));
        }
        let have = self.tables[slot].blocks().len();
        if want <= have {
            return Ok(());
        }
        if self.pool.free_blocks() < want - have {
            return Err(GpuModelError::PoolExhausted);
        }
        self.tables[slot]
            .ensure(upto - 1, &mut self.pool)
            .map_err(|_| GpuModelError::PoolExhausted)?;
        self.upload_row(exec, slot)
    }

    /// Mirror `slot`'s table into its block-table row (host and device).
    pub fn upload_row(&mut self, exec: &GpuExecutor, slot: usize) -> Result<(), GpuModelError> {
        let blocks = self.tables[slot].blocks();
        let base = slot * self.bps;
        self.bt_host[base..base + blocks.len()].copy_from_slice(blocks);
        let row = &self.bt_host[base..base + self.bps];
        let mut view = self
            .bt
            .try_slice_mut(slot * self.bps..(slot + 1) * self.bps)
            .ok_or_else(|| GpuModelError::Unsupported("inkling2: block table row".into()))?;
        exec.stream.memcpy_htod(row, &mut view).map_err(|e| {
            GpuModelError::Unsupported(format!("inkling2: block table upload: {e}"))
        })?;
        Ok(())
    }

    /// Drop `slot`'s references to its blocks (the ones the prefix cache
    /// keeps stay) and forget its rows.
    pub fn release(&mut self, slot: usize) {
        self.tables[slot].clear(&mut self.pool);
        self.lens[slot] = 0;
        if let Some(r) = self.spec_run.get_mut(slot) {
            *r = None;
        }
    }

    /// Restart `slot`'s conv states at zero (a new sequence).
    pub fn zero_conv(&mut self, exec: &GpuExecutor, slot: usize) -> Result<(), GpuModelError> {
        for planes in &mut self.conv {
            for (i, p) in planes.iter_mut().enumerate() {
                let c = if i < CONV_ATTN { KV } else { ik::WIDTH };
                exec.zero_region(p, slot * 3 * c, 3 * c)?;
            }
        }
        Ok(())
    }

    /// Device bytes the stores and states hold.
    pub fn bytes(&self) -> u64 {
        let s: usize = self
            .stores
            .iter()
            .chain(&self.draft)
            .map(|s| s.k.len() + s.v.len())
            .sum::<usize>()
            + self.feat_ring.as_ref().map_or(0, |r| r.len() * 4);
        let c: usize = self
            .conv
            .iter()
            .chain(&self.stash)
            .flat_map(|p| p.iter().map(|x| x.len() * 4))
            .sum();
        (s + c + self.bt_host.len() * 4) as u64
    }
}
