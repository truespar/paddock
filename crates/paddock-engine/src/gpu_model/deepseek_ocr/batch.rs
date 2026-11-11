//! DeepSeek-OCR continuous batching: paged KV, batched decode, chunked
//! prefill, and the R-SWA ring on the pool - built on granite's lane (the
//! simplest in the repo) with laguna's MoE blocks.
//!
//! ## The ring on paged KV: three row streams, no new kernels
//!
//! Every pass carries three position streams instead of granite's one, and
//! each kernel class reads exactly one of them:
//!
//!   d_pos   true position   -> rope (absolute forever, the reference's rule)
//!   d_wpos  write slot      -> KV append (`pf + (pos-pf) % W` in steady state)
//!   d_apos  attention bound -> attention (`pf + W - 1` once the ring is full:
//!                             the kernel derives n_kv = apos+1, windowless)
//!
//! During prefill and warmup all three are equal, so those phases are
//! byte-identical to an ordinary causal model. `ensure_rows` is driven by
//! d_wpos, so a slot's pool footprint PINS at `prefill + W` blocks no matter
//! how long the output runs - the family's whole point, and what the fit
//! estimate prices.
//!
//! ## Why the radix prefix cache stays on
//!
//! The safety argument:
//! - adoption and insertion are both full-16-row-block granular
//!   (`paged_radix::insert` takes `tokens.len()/16` blocks; a match covering
//!   the block that contains `prefill_len` would need more rows than the
//!   prompt has), so the boundary block is always slot-private;
//! - ring rows live at `[pf, pf+W)` - the boundary block plus fresh blocks -
//!   and therefore never touch a shared block;
//! - prefill rows are never rewritten after the ring engages.
//!   So the vision+prompt prefix is a plain shareable prefix, ring pages
//!   are slot-private scratch, and nothing new is needed. Only the RING is
//!   uncacheable, which costs nothing: its contents are generation-specific.
//!
//! Decode ticks are captured into per-r CUDA graphs (granite's recipe: all
//! planes allocated at enable, loop bounds read from device buffers at
//! replay, host work outside the capture). The ring's divergence lives
//! entirely in the d_wpos/d_apos uploads, which happen before replay - the
//! captured body is ring-agnostic.
//!
//! Deliberately not here yet (perf rungs): the W4A8 flat-mmq prefill rung
//! and the WMMA-prefill fp8-KV arm (G=1 has no fp8 tile - f16 KV rides the
//! wmma path), the decode pipe, spec.
//!
//! FlashDecoding splits are in, and the bring-up note above was wrong to park
//! them as "decode is launch-bound -> graphs first". The serial
//! `pd_attn_decode_batch_paged` walk is work-bound: a
//! (10 heads × ≤8 rows) = ≤80-block grid cannot fill a 188-SM die, and it
//! runs at ~23% of the DRAM roof.

use std::collections::HashMap;

use cudarc::driver::CudaSlice;
use cudarc::driver::sys::CUstreamCaptureMode;

use crate::gpu::{DASH_D, GpuError, KvDtype};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::granite::batch::pf_rows;
use crate::gpu_model::qwen35::{gemv_any, mmq_pre_any, prefill_mm_pre_any, prefill_quant};
use crate::kv_plan;
use crate::kv_pool::{BlockTable, KvPool};

use super::load::{DsFfn, GpuDeepseekOcr};

/// VRAM slack the slot-fit math leaves untouched (graph/scratch churn).
const VRAM_HEADROOM: usize = 1 << 30;

/// FlashDecoding split ceiling (partial-scratch sizing).
const MAX_ATTN_SPLITS: usize = 16;

/// KV splits for the batched decode attention - position-INDEPENDENT so a
/// captured per-r graph can bake it (qwen3-asr's formula). This family is
/// 10q/10kv = group 1, so the GQA-fused arm never applies: the unfused
/// partial walks one q-head per block and the only question is grid fill.
/// At c8 the unsplit grid is 80 blocks on a 188-SM die; 16 splits make it
/// 1280 (~65-key chunks at the ~1035-token R-SWA depth).
fn attn_splits_for(n_heads: usize, batch: usize, sm_count: usize) -> usize {
    if paddock_models::dev_var_os!("PADDOCK_NO_ATTN_SPLIT").is_some() {
        return 1;
    }
    if n_heads * batch >= 2 * 3 * sm_count {
        return 1; // die already saturated
    }
    if let Some(n) = paddock_models::dev_var!("PADDOCK_ATTN_SPLITS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n >= 1)
    {
        return n.min(MAX_ATTN_SPLITS);
    }
    MAX_ATTN_SPLITS
}

pub(crate) struct LayerKv {
    pub k: CudaSlice<u8>,
    pub v: CudaSlice<u8>,
}

/// Per-slot R-SWA state. `prefill_len` arms the ring; None (mid-prefill, or a
/// family member without R-SWA) means ordinary causal growth.
#[derive(Clone, Copy, Default)]
pub(crate) struct RingState {
    pub prefill_len: Option<usize>,
}

/// Batched-lane scratch, sized once at enable for `cap`-row passes.
pub(crate) struct BatchScratch {
    pub x: CudaSlice<f32>,
    pub xn: CudaSlice<f32>,
    pub xq: CudaSlice<i8>,
    pub xs: CudaSlice<f32>,
    pub ssums: CudaSlice<f32>,
    /// mma_ks K-split partials
    pub part: CudaSlice<f32>,
    /// flat-mmq activations for the r-invariant prefill GEMM ladder
    pub yq: CudaSlice<u8>,
    /// per-128-block sums off yq (the W4A8 mu term; Q8_0 leaves it unread)
    pub xsums: CudaSlice<f32>,
    /// Q8_0 mmq stream-k fixup plane
    pub skfix: CudaSlice<f32>,
    pub q: CudaSlice<f32>,
    pub k: CudaSlice<f32>,
    pub v: CudaSlice<f32>,
    pub attn: CudaSlice<f32>,
    /// FlashDecoding partial planes: per-(head, decode-row, split) o and
    /// (m, l). Decode rows never exceed `n_slots`, so they size by slots,
    /// not `cap` - prefill rows go through the prefill kernels.
    pub attn_o: CudaSlice<f32>,
    pub attn_ml: CudaSlice<f32>,
    pub proj: CudaSlice<f32>,
    /// no-op sinks [n_heads] - -inf, the softmax identity (lesson)
    pub sinks: CudaSlice<f32>,
    pub ffn_gate: CudaSlice<f32>,
    pub ffn_up: CudaSlice<f32>,
    // MoE lane
    pub moe_logits: CudaSlice<f32>,
    pub moe_idx: CudaSlice<u32>,
    pub moe_w: CudaSlice<f32>,
    pub moe_fused: CudaSlice<f32>,
    pub moe_fq: CudaSlice<i8>,
    pub moe_fs: CudaSlice<f32>,
    // sorted-MoE lane (moe_align layout): the routed pair's prefill arm.
    // srow/sslot map padded sorted rows back to (token, slot); bexp names
    // each block's expert; part holds per-(token, slot) down partials for
    // the fixed-order fold (bit-reproducible - atomic scatter flips
    // near-tie greedy tokens, the b9895 lesson).
    pub moe_srow: CudaSlice<u32>,
    pub moe_sslot: CudaSlice<u32>,
    pub moe_bexp: CudaSlice<u32>,
    pub moe_part: CudaSlice<f32>,
    pub sh_out: CudaSlice<f32>,
    // row streams - the ring's three position views plus tokens and slots
    pub d_toks: CudaSlice<u32>,
    pub d_pos: CudaSlice<u32>,
    pub d_wpos: CudaSlice<u32>,
    pub d_apos: CudaSlice<u32>,
    pub d_slots: CudaSlice<u32>,
    pub head_logits: CudaSlice<f32>,
    // device sampling: rows × {inv_t, u, mode, pad} params and
    // the picked ids - head rows argmax on card instead of the [rows, vocab]
    // readback + host scan that made sampling 10.3 % of the c8 tick wall
    pub d_par: CudaSlice<u32>,
    pub d_out: CudaSlice<u32>,
    ///  uniq-routing diagnostic (PADDOCK_MOE_UNIQ=path): raw non-pool
    /// accumulator armed at enable, 0 = unarmed. See gemma4's
    /// `g4_moe_uniq_arm` for the layout + dumper thread.
    pub moe_uniq_dev: u64,
}

/// The whole batching state: pool + tables + ring + scratch.
pub(crate) struct BatchState {
    pub n_slots: usize,
    /// scratch row capacity (pf_rows + slots) - mixed ticks size by it
    pub cap: usize,
    pub bps: usize,
    pub pool: KvPool,
    pub tables: Vec<BlockTable>,
    pub bt_host: Vec<u32>,
    pub d_bt: CudaSlice<u32>,
    pub kv: Vec<LayerKv>,
    pub ring: Vec<RingState>,
    pub sc: BatchScratch,
    pub prefix: Option<crate::paged_radix::PagedRadix>,
    pub kv_bytes: u64,
    pub graphs: HashMap<usize, super::SendGraph>,
    /// Slot 0's decode cursor for the serial `Generator::forward` surface
    /// (warmup and any serial-path caller) - set by whichever prefill entry
    /// ran last, advanced per decoded token. qwen3_asr's pattern.
    pub slot0_pos: usize,
    /// Rows the last `prefix_resume` adopted, per slot - `take_prefill_reused`
    /// telemetry (the usage report's `cached` field).
    pub last_reused: Vec<usize>,
    /// Unlimited-OCR-2's router state (None on the R-SWA members).
    pub dash: Option<super::dash::DashDev>,
}

/// Contiguous same-slot runs over a chunk's prefill rows - an attention
/// launch never mixes two slots' query rows. (`pub(super)` so the multimodal
/// splice can drive the same walk.)
pub(super) struct PfCuts {
    pub(super) dec: usize,
    pub(super) runs: Vec<(usize, usize)>,
}

fn drv(e: cudarc::driver::DriverError) -> GpuError {
    crate::gpu::from_driver(e)
}

/// Triage: FNV-1a over raw f32 bits (PADDOCK_OCR_BATCH_DUMP legs).
fn fnv_f32(h: &[f32]) -> u64 {
    let mut acc = 0xcbf29ce484222325u64;
    for v in h {
        for b in v.to_bits().to_le_bytes() {
            acc ^= b as u64;
            acc = acc.wrapping_mul(0x100000001b3);
        }
    }
    acc
}

impl GpuDeepseekOcr {
    /// Allocate the paged-KV + scratch state for up to `max_batch` slots.
    pub fn enable_batch(&mut self, max_batch: usize) -> Result<usize, GpuModelError> {
        if !self.exec.has_paged_kv() {
            return Ok(1);
        }
        // the serial dense KV makes way for the paged stores
        self.decode = None;
        self.scratch = None;
        self.batch = None;
        self.exec.trim_mem_pool();

        let hp = &self.hp;
        let (embd, nh, n_kv, hd) = (hp.n_embd, hp.n_head, hp.n_head_kv, hp.head_dim);
        let kv_dim = n_kv * hd;
        let kvb = self.kv_dtype.bytes();
        let bps = self.max_ctx.div_ceil(16);
        let n_layer = hp.n_layer;
        let block_bytes = n_layer * 16 * kv_dim * 2 * kvb;
        let cap = pf_rows() + max_batch;
        let fused_len = hp.n_expert_used * hp.n_ff_exp;
        let wide = hp.n_ff.max(fused_len).max(hp.shexp_ff()).max(embd);
        // sorted-MoE sizing: padded rows at the worst block width (each
        // touched expert wastes up to bm-1 PAD rows; bm=64 pads more rows,
        // bm=32 makes more blocks - size each buffer by its own worst case)
        let pairs_cap = cap * hp.n_expert_used;
        let blocks32_cap = (pairs_cap + hp.n_expert * 31).div_ceil(32);
        let blocks64_cap = (pairs_cap + hp.n_expert * 63).div_ceil(64);
        let rows_pad_cap = (blocks32_cap * 32).max(blocks64_cap * 64);

        let scratch_est = (cap * (2 * hp.n_ff + 4 * embd + 2 * fused_len) * 4)
            + (cap * wide * 4)
            + (8 * 64 * wide * 4)
            + (max_batch * hp.n_vocab * 4)
            + (pairs_cap * embd * 4)
            + (rows_pad_cap * hp.n_ff_exp)
            + (256 << 20);
        // the router lane keeps per-block summaries beside the KV; a resumed
        // prefix would need them restored too, so prefix reuse is off there
        let px_on = !super::prefix::prefix_disabled() && hp.dash.is_none();
        if hp.dash.is_some() && (!self.exec.has_dash() || !matches!(self.kv_dtype, KvDtype::Fp16)) {
            return Err(GpuModelError::Unsupported(
                "unlimited-ocr-2 needs the pack's SR-SWA kernels (slots 935-939) and an f16 KV"
                    .into(),
            ));
        }
        let retain = if px_on {
            super::prefix::retention_blocks()
        } else {
            0
        };
        // One arbiter sizes the KV store: crate::kv_plan.
        let grant = self
            .exec
            .vram_headroom()
            .ok_or_else(|| GpuError::Driver("no free-VRAM reading".into()))?;
        let demand = kv_plan::Demand {
            family: "deepseek-ocr",
            max_ctx: self.max_ctx,
            slots: max_batch,
            // R-SWA changes the per-slot ceiling: a slot can never address more
            // than its prompt + the ring, but the PROMPT bound is max_ctx, so the
            // hard addressing cap stays slots × bps. The ring is why the pool
            // will in practice sit far below it - pin the savings, don't
            // pre-spend them.
            blocks_per_slot: bps,
            block_bytes: block_bytes as u64,
            // one block id addresses every layer, so no KV is per-slot here
            per_slot_bytes: 0,
            retention_blocks: retain,
            // every slot must at least hold a full chunk's worth of prompt, or
            // admission deadlocks on its own first chunk
            floor_blocks_per_slot: pf_rows().div_ceil(16),
            floor_blocks_min: 256,
            reserves: vec![
                kv_plan::Reserve::new("graph/scratch slack", VRAM_HEADROOM as u64),
                kv_plan::Reserve::new("prefill scratch", scratch_est as u64),
            ],
            ..Default::default()
        };
        let plan = demand
            .plan(grant)
            .map_err(|e| GpuModelError::WontFit(e.message))?;
        plan.report(&demand, grant);
        let pool_blocks = plan.pool_blocks;
        let slots = plan.slots;

        let e = &self.exec;
        let mut kv = Vec::with_capacity(n_layer);
        let mut kv_bytes = 0u64;
        for _ in 0..n_layer {
            let bytes = pool_blocks * 16 * kv_dim * kvb;
            kv_bytes += 2 * bytes as u64;
            kv.push(LayerKv {
                k: e.alloc_u8(bytes)?,
                v: e.alloc_u8(bytes)?,
            });
        }

        let sc = BatchScratch {
            x: e.alloc(cap * embd)?,
            xn: e.alloc(cap * embd)?,
            xq: e.alloc_i8(cap * wide)?,
            xs: e.alloc(cap * wide / 32)?,
            ssums: e.alloc(cap * wide / 16)?,
            part: e.alloc(8 * 64 * wide)?,
            yq: e.alloc_u8(wide.div_ceil(128) * cap.next_multiple_of(128) * 144)?,
            xsums: e.alloc(wide.div_ceil(128) * cap.next_multiple_of(128) * 4)?,
            skfix: e.alloc(256 * 128 * 128 + 256)?,
            q: e.alloc(cap * embd)?,
            k: e.alloc(cap * kv_dim)?,
            v: e.alloc(cap * kv_dim)?,
            attn: e.alloc(cap * embd)?,
            attn_o: e.alloc(nh * slots * MAX_ATTN_SPLITS * hp.head_dim)?,
            attn_ml: e.alloc(nh * slots * MAX_ATTN_SPLITS * 2)?,
            proj: e.alloc(cap * embd)?,
            sinks: e.alloc_no_sinks(nh)?,
            ffn_gate: e.alloc(cap * hp.n_ff.max(hp.shexp_ff()))?,
            ffn_up: e.alloc(cap * hp.n_ff.max(hp.shexp_ff()))?,
            moe_logits: e.alloc(cap * hp.n_expert)?,
            moe_idx: e.alloc_u32(cap * hp.n_expert_used)?,
            moe_w: e.alloc(cap * hp.n_expert_used)?,
            moe_fused: e.alloc(cap * fused_len)?,
            // fq/fs serve both shapes: token-batched [cap, fused_len] and
            // sorted [rows_pad, n_ff_exp] - rows_pad_cap covers the pad
            moe_fq: e.alloc_i8(rows_pad_cap * hp.n_ff_exp)?,
            moe_fs: e.alloc(rows_pad_cap * hp.n_ff_exp / 32)?,
            moe_srow: e.alloc_u32(rows_pad_cap)?,
            moe_sslot: e.alloc_u32(rows_pad_cap)?,
            moe_bexp: e.alloc_u32(blocks32_cap.max(blocks64_cap))?,
            moe_part: e.alloc(pairs_cap * embd)?,
            sh_out: e.alloc(cap * embd)?,
            d_toks: e.alloc_u32(cap)?,
            d_pos: e.alloc_u32(cap)?,
            d_wpos: e.alloc_u32(cap)?,
            d_apos: e.alloc_u32(cap)?,
            d_slots: e.alloc_u32(cap)?,
            head_logits: e.alloc(slots.max(1) * hp.n_vocab)?,
            d_par: e.alloc_u32(slots.max(1) * 4)?,
            d_out: e.alloc_u32(slots.max(1))?,
            // diagnostic only - a failed arm logs and serves unarmed rather
            // than failing enable_batch
            moe_uniq_dev: if hp.n_expert != 0
                && paddock_models::dev_var_os!("PADDOCK_MOE_UNIQ").is_some()
            {
                match crate::gpu_model::gemma4::g4_moe_uniq_arm(e) {
                    Ok(p) => p,
                    Err(err) => {
                        tracing::warn!("moe_uniq arm failed, serving unarmed: {err}");
                        0
                    }
                }
            } else {
                0
            },
        };

        self.batch = Some(BatchState {
            n_slots: slots,
            cap,
            bps,
            pool: KvPool::with_blocks(pool_blocks as u32),
            tables: (0..slots).map(|_| BlockTable::new()).collect(),
            bt_host: vec![0u32; slots * bps],
            d_bt: e.alloc_u32(slots * bps)?,
            kv,
            ring: vec![RingState::default(); slots],
            sc,
            prefix: px_on.then(crate::paged_radix::PagedRadix::new),
            kv_bytes,
            graphs: HashMap::new(),
            slot0_pos: 0,
            last_reused: vec![0; slots],
            dash: None,
        });
        if let Some(d) = self.hp.dash {
            let dd = super::dash::DashDev::new(&self.exec, &d, n_layer, slots, cap, self.max_ctx)?;
            tracing::info!(
                "unlimited-ocr-2 router state: {:.1} MiB ({} blocks a slot)",
                dd.bytes() as f64 / (1u64 << 20) as f64,
                dd.kbar_blocks
            );
            self.batch.as_mut().expect("just built").dash = Some(dd);
        }
        tracing::info!(
            "deepseek-ocr batch: {slots} slots, {n_layer}-layer pool {pool_blocks} blocks \
             ({:.2} GiB, {} tokens), {} rows/chunk, R-SWA {:?} - KV per slot pins at \
             prefill+{} rows",
            (pool_blocks * block_bytes) as f64 / (1u64 << 30) as f64,
            pool_blocks * 16,
            pf_rows(),
            self.hp.rswa_window,
            self.hp.rswa_window.unwrap_or(0),
        );
        Ok(slots)
    }

    /// The ring's row mapping for one (slot, true position): (write slot,
    /// attention-bound position). Ordinary causal until the boundary is
    /// marked and the warmup is over.
    fn ring_map(&self, slot: usize, pos: usize) -> (usize, usize) {
        let bs = self.batch.as_ref().expect("batch enabled");
        // the router lane's ring is its local window (its attention reads the
        // window by true position; the bound stream is unused there)
        let window = self.hp.rswa_window.or(self.hp.dash.map(|d| d.local_window));
        match (bs.ring[slot].prefill_len, window) {
            (Some(pf), Some(w)) if pos >= pf + w => (pf + (pos - pf) % w, pf + w - 1),
            _ => (pos, pos),
        }
    }

    /// Back every `(slot, WRITE position)` with a physical pool block. Driven
    /// by the ring-mapped write slot, which is what pins a ring sequence's
    /// footprint. PoolExhausted sheds radix retention first.
    pub(super) fn ensure_rows(&mut self, slots: &[u32], wpos: &[u32]) -> Result<(), GpuModelError> {
        let bs = self.batch.as_mut().expect("batch enabled");
        let mut grew = false;
        for (i, &s) in slots.iter().enumerate() {
            let s = s as usize;
            let before = bs.tables[s].blocks().len();
            loop {
                match bs.tables[s].ensure(wpos[i] as usize, &mut bs.pool) {
                    Ok(()) => break,
                    Err(_) => {
                        let shed = bs
                            .prefix
                            .as_mut()
                            .and_then(|r| r.evict_lru(&mut bs.pool))
                            .is_some();
                        if !shed {
                            return Err(GpuModelError::PoolExhausted);
                        }
                    }
                }
            }
            let now = bs.tables[s].blocks().len();
            // A table past its bt stripe would write the NEIGHBOR slot's
            // entries (silent cross-slot corruption) before the flat indexing
            // even panics on the last slot. The service clamps generation to
            // the window rows-based at prefill finish, so this firing means
            // that contract broke - refuse loudly instead of spilling
            // (first seen on the paddleocr twin of this loop).
            if now > bs.bps {
                return Err(GpuModelError::ContextExceeded {
                    got: now * 16,
                    max: bs.bps * 16,
                });
            }
            if now > before {
                grew = true;
                let base = s * bs.bps;
                for j in before..now {
                    bs.bt_host[base + j] = bs.tables[s].blocks()[j];
                }
            }
        }
        if grew {
            self.exec
                .stream
                .memcpy_htod(&bs.bt_host, &mut bs.d_bt)
                .map_err(drv)?;
        }
        Ok(())
    }

    pub fn release_inactive_slots(&mut self, occupied: &[bool]) {
        let Some(bs) = self.batch.as_mut() else {
            return;
        };
        for (s, occ) in occupied.iter().enumerate() {
            if !occ && s < bs.tables.len() && !bs.tables[s].blocks().is_empty() {
                bs.tables[s].clear(&mut bs.pool);
                bs.ring[s] = RingState::default();
                if let Some(dd) = bs.dash.as_mut() {
                    dd.slots[s] = None;
                }
            }
        }
    }

    pub fn pool_active(&self) -> bool {
        self.batch.is_some()
    }

    pub fn pool_free_blocks(&self) -> Option<usize> {
        self.batch
            .as_ref()
            .map(|b| b.pool.free_blocks() + self.prefix_evictable())
    }

    pub fn kv_mem_bytes(&self) -> Option<u64> {
        self.batch.as_ref().map(|b| b.kv_bytes)
    }

    /// Pool blocks `slot` currently owns - what the ring-pinning gate reads:
    /// after any number of generated tokens this must be exactly
    /// `ceil((prefill + W) / 16)` on a ring sequence.
    pub fn pool_slot_blocks(&self, slot: usize) -> Option<usize> {
        self.batch.as_ref().map(|b| b.tables[slot].blocks().len())
    }

    // ── prefill ─────────────────────────────────────────────────────────────

    /// Prefill a whole prompt into `slot` (chunked at `pf_rows`), mark the
    /// R-SWA boundary, publish the prefix, and return the last token's logits.
    pub fn forward_prefill(
        &mut self,
        slot: usize,
        tokens: &[u32],
    ) -> Result<Vec<f32>, GpuModelError> {
        self.admit(slot, tokens.len())?;
        self.dash_admit(slot, tokens);
        let mut base = self.prefill_resume_rows(slot, tokens, tokens.len())?;
        let mut last_len = 0usize;
        for chunk in tokens[base..].chunks(pf_rows()) {
            let rows: Vec<(u32, u32, u32)> = chunk
                .iter()
                .enumerate()
                .map(|(j, &t)| (slot as u32, (base + j) as u32, t))
                .collect();
            self.rows_pass_body(&rows, 0)?;
            base += chunk.len();
            last_len = chunk.len();
        }
        self.prefix_insert(slot, tokens);
        // Everything appended so far is the globally-visible prefix; the ring
        // covers only what decode appends after this.
        let base = self.ring_base(slot, tokens.len());
        self.batch.as_mut().expect("batch enabled").ring[slot].prefill_len = Some(base);
        self.dash_head_guard(slot)?;
        self.head_row(last_len - 1)
    }

    /// Admission checks + slot reset for an `n_rows`-row prompt (row count,
    /// not token count: the multimodal path admits image rows too).
    pub(super) fn admit(&mut self, slot: usize, n_rows: usize) -> Result<(), GpuModelError> {
        let n_slots = self.batch.as_ref().expect("batch enabled").n_slots;
        if slot >= n_slots {
            return Err(GpuModelError::Unsupported(format!(
                "slot {slot} >= enabled {n_slots}"
            )));
        }
        if n_rows == 0 {
            return Err(GpuModelError::Unsupported("empty prompt".into()));
        }
        if n_rows > self.max_ctx {
            return Err(GpuModelError::Unsupported(format!(
                "prompt {n_rows} rows > max_ctx {}",
                self.max_ctx
            )));
        }
        {
            let bs = self.batch.as_mut().expect("batch enabled");
            bs.tables[slot].clear(&mut bs.pool);
            bs.ring[slot] = RingState::default();
        }
        self.ensure_rows(&[slot as u32], &[(n_rows - 1) as u32])
    }

    /// Numeric class of a resume, stated where it happens: the recomputed
    /// TAIL is not bitwise-identical to the cold pass. The engine's norm
    /// elects its reduction width at the 64-row boundary (pd_rmsnorm_batch:
    /// "the >=64 prefill lanes change reduction grouping = the sanctioned
    /// near-tie class"), so a short tail norms 1024-wide where the cold chunk
    /// normed 256-wide - last-ulp differences the MoE router can amplify into
    /// a flipped near-tie expert (measured: max|Δ| 0.20 on one logit of a
    /// garbage prompt, argmax unchanged). Same class the realign gates
    /// arbitrate everywhere else; bitwise resume would need an r-pinned norm
    /// width, tracked as a follow-up if OCR realign ever flips on real
    /// documents (peaked distributions make it unlikely).
    pub(super) fn prefill_resume_rows(
        &mut self,
        slot: usize,
        keys: &[u32],
        n_rows: usize,
    ) -> Result<usize, GpuModelError> {
        let start = self.prefix_resume(slot, keys)?;
        self.batch.as_mut().expect("batch enabled").last_reused[slot] = start;
        if start > 0 {
            self.ensure_rows(&[slot as u32], &[(n_rows - 1) as u32])?;
        }
        Ok(start)
    }

    /// One pass over `chunk` rows (slot, true position, token). The leading
    /// `dec` rows are the decode band. Prefill rows have wpos == apos == pos
    /// by construction (the ring only diverges after the boundary is marked,
    /// and a prefill pass runs before its own mark). Rows belonging to a
    /// QUEUED image prompt have their placeholder embeddings overwritten from
    /// the entry's feature planes by (slot, position) - see chunked.rs.
    pub(super) fn rows_pass_body(
        &mut self,
        chunk: &[(u32, u32, u32)],
        dec: usize,
    ) -> Result<(), GpuModelError> {
        let r = chunk.len();
        let toks: Vec<u32> = chunk.iter().map(|x| x.2).collect();
        let slots_v: Vec<u32> = chunk.iter().map(|x| x.0).collect();
        let pos: Vec<u32> = chunk.iter().map(|x| x.1).collect();
        let mut wpos = Vec::with_capacity(r);
        let mut apos = Vec::with_capacity(r);
        for x in chunk {
            let (w, a) = self.ring_map(x.0 as usize, x.1 as usize);
            wpos.push(w as u32);
            apos.push(a as u32);
        }
        let mut runs: Vec<(usize, usize)> = Vec::new();
        for (i, x) in chunk.iter().enumerate().skip(dec) {
            match runs.last_mut() {
                Some((off, n)) if chunk[*off].0 == x.0 => *n += 1,
                _ => runs.push((i, 1)),
            }
        }
        self.ensure_rows(&slots_v, &wpos)?;
        self.upload_rows(&toks, &pos, &wpos, &apos, &slots_v, dec)?;
        self.embed_rows(r)?;
        self.splice_queued_rows(chunk, dec)?;
        // triage (PADDOCK_OCR_CHUNK_DUMP=1): bit-exact hash of the spliced
        // input and of the walk output - the cold-vs-resume divergence
        // bisector. Synchronizes.
        let dump = paddock_models::dev_var_os!("PADDOCK_OCR_CHUNK_DUMP").is_some();
        if dump {
            let h = self.dump_hash_x(r)?;
            let b0 = self.dump_hash_blk0()?;
            eprintln!(
                "ocr chunk-pass r={r} dec={dec} pos0={} xin={h:016x} blk0={b0:016x}",
                chunk[dec].1
            );
        }
        self.layer_walk(r, Some(&PfCuts { dec, runs }))?;
        if dump {
            let h = self.dump_hash_x(r)?;
            let b0 = self.dump_hash_blk0()?;
            eprintln!(
                "ocr chunk-pass r={r} dec={dec} pos0={} xout={h:016x} blk0={b0:016x}",
                chunk[dec].1
            );
        }
        Ok(())
    }

    /// Triage: hash of PHYSICAL pool block 0's layer-0 K+V rows.
    pub(super) fn dump_hash_blk0(&mut self) -> Result<u64, GpuModelError> {
        let kv_dim = self.hp.n_head_kv * self.hp.head_dim;
        let row_bytes = kv_dim * self.kv_dtype.bytes();
        let bs = self.batch.as_ref().expect("batch enabled");
        let mut acc = 0xcbf29ce484222325u64;
        for plane in [&bs.kv[0].k, &bs.kv[0].v] {
            if let Some(v) = plane.try_slice(0..16 * row_bytes) {
                for byte in self.exec.stream.clone_dtoh(&v).map_err(drv)? {
                    acc ^= byte as u64;
                    acc = acc.wrapping_mul(0x100000001b3);
                }
            }
        }
        Ok(acc)
    }

    /// Triage helper: FNV-1a over the raw f32 bits of residual rows 0..r.
    pub(super) fn dump_hash_x(&mut self, r: usize) -> Result<u64, GpuModelError> {
        let embd = self.hp.n_embd;
        let bs = self.batch.as_ref().expect("batch enabled");
        let h = self.exec.to_host_len(&bs.sc.x, r * embd)?;
        Ok(fnv_f32(&h))
    }

    // ── decode ──────────────────────────────────────────────────────────────

    /// One batched decode step; row i drives `slots[i]` at true position
    /// `positions[i]`. Leaves [r, vocab] logits in head_logits.
    pub fn batch_step_slots(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        slots: &[u32],
    ) -> Result<(), GpuModelError> {
        let r = tokens.len();
        assert_eq!(r, positions.len());
        assert_eq!(r, slots.len());
        let n_slots = self.batch.as_ref().expect("batch enabled").n_slots;
        assert!(r <= n_slots, "rows {r} > enabled {n_slots}");
        let mut wpos = Vec::with_capacity(r);
        let mut apos = Vec::with_capacity(r);
        for i in 0..r {
            let (w, a) = self.ring_map(slots[i] as usize, positions[i] as usize);
            wpos.push(w as u32);
            apos.push(a as u32);
        }
        // grown by the WRITE slot: once the ring engages this never grows again
        self.ensure_rows(slots, &wpos)?;
        self.upload_rows(tokens, positions, &wpos, &apos, slots, r)?;
        self.step_replay(r)
    }

    pub fn batch_step(&mut self, tokens: &[u32], positions: &[u32]) -> Result<(), GpuModelError> {
        let ident: Vec<u32> = (0..tokens.len() as u32).collect();
        self.batch_step_slots(tokens, positions, &ident)
    }

    fn step_body(&mut self, r: usize) -> Result<(), GpuModelError> {
        self.embed_rows(r)?;
        self.layer_walk(r, None)?;
        self.head_rows(r)
    }

    fn capture_body(
        &mut self,
        body: impl FnOnce(&mut Self) -> Result<(), GpuModelError>,
        what: &str,
    ) -> Result<super::SendGraph, GpuModelError> {
        let exec = self.exec.clone();
        exec.stream
            .synchronize()
            .map_err(|e| GpuError::Driver(format!("{what} pre-capture sync: {e}")))?;
        exec.stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|e| GpuError::Driver(format!("{what} begin_capture: {e}")))?;
        let rec = body(self);
        let graph = crate::gpu::end_capture_no_flags(&exec.stream)
            .map_err(|e| GpuError::Driver(format!("{what} end_capture: {e}")));
        rec?;
        let graph =
            graph?.ok_or_else(|| GpuError::Driver(format!("{what} capture produced no graph")))?;
        Ok(super::SendGraph(graph))
    }

    fn step_replay(&mut self, r: usize) -> Result<(), GpuModelError> {
        if !self
            .batch
            .as_ref()
            .expect("batch enabled")
            .graphs
            .contains_key(&r)
        {
            let g = self.capture_body(|s| s.step_body(r), "decode")?;
            self.batch
                .as_mut()
                .expect("batch enabled")
                .graphs
                .insert(r, g);
        }
        self.batch.as_ref().expect("batch enabled").graphs[&r]
            .0
            .launch()
            .map_err(|e| GpuError::Driver(format!("decode graph launch: {e}")))?;
        Ok(())
    }

    /// Upload one pass's row streams; the leading `dec` rows are the decode
    /// band (holes included), the rest prompt rows.
    pub(super) fn upload_rows(
        &mut self,
        tokens: &[u32],
        pos: &[u32],
        wpos: &[u32],
        apos: &[u32],
        slots: &[u32],
        dec: usize,
    ) -> Result<(), GpuModelError> {
        let bs = self.batch.as_mut().expect("batch enabled");
        self.exec.upload_u32(tokens, &mut bs.sc.d_toks)?;
        self.exec.upload_u32(pos, &mut bs.sc.d_pos)?;
        self.exec.upload_u32(wpos, &mut bs.sc.d_wpos)?;
        self.exec.upload_u32(apos, &mut bs.sc.d_apos)?;
        self.exec.upload_u32(slots, &mut bs.sc.d_slots)?;
        // every pass uploads its rows exactly once, here: the router lane's
        // decode state machine advances with them
        self.dash_prepare(tokens, pos, slots, dec)
    }

    pub(super) fn embed_rows(&mut self, r: usize) -> Result<(), GpuModelError> {
        let embd = self.hp.n_embd;
        let bs = self.batch.as_mut().expect("batch enabled");
        self.exec
            .embed_gather_batch_q8(&self.tok_embd, &bs.sc.d_toks, &mut bs.sc.x, embd, r)?;
        Ok(())
    }

    pub(super) fn layer_walk(
        &mut self,
        r: usize,
        cuts: Option<&PfCuts>,
    ) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let hp = &self.hp;
        let (embd, nh, n_kv, hd) = (hp.n_embd, hp.n_head, hp.n_head_kv, hp.head_dim);
        let kv_dim = n_kv * hd;
        let eps = hp.rms_eps;
        let scale = 1.0 / (hd as f32).sqrt();
        let (n_expert, n_active, moe_ff, shexp_ff) =
            (hp.n_expert, hp.n_expert_used, hp.n_ff_exp, hp.shexp_ff());
        let rope = self.rope_params();
        let kv_dtype = self.kv_dtype;
        let dash_p = self.hp.dash;
        let bs = self.batch.as_mut().expect("batch enabled");
        let bps = bs.bps;
        let pf = cuts.is_some();
        let r1 = r == 1 && !pf;
        // Decode glue folds: the split chains spent ~0.2 ms/step in tiny
        // norm/quant/activation launches. The fused
        // kernels write the same planes the split chains did (xn stays for the
        // router/gemv readers), values identical per each kernel's contract,
        // so every downstream consumer is untouched. Prefill keeps the
        // r-invariant ladder (its bytes-equality contract is with itself).
        let glue_fuse = !pf
            && exec.has_rmsnorm_quant_q8()
            && exec.has_add_rmsnorm_quant_q8()
            && exec.has_swiglu_quant_q8()
            && paddock_models::dev_var_os!("PADDOCK_OCR_NO_GLUE_FUSE").is_none();
        // Ring-aware rope+append fold: the granite fold finally
        // reaches this family - the ring twin reads two position streams
        // (d_pos turns the rope, d_wpos lands the appends), which is exactly
        // why the one-stream fusions stayed off here. Decode passes only:
        // mixed chunks rope prefill rows too and keep the batch kernels.
        let rope_fuse = cuts.is_none()
            && exec.has_rope_qk_append_paged_ring()
            && paddock_models::dev_var_os!("PADDOCK_NO_ROPE_FUSE").is_none();
        // WMMA prefill attention - hd128 f16-KV rides the tensor-core class.
        // (G=1 has no fp8 prefill tile; fp8 KV falls through to the tiled or
        // decode-class kernels below, correct at a worse speed.)
        let wmma_pf = cuts.is_some()
            && hd == 128
            && matches!(kv_dtype, KvDtype::Fp16)
            && exec.has_attn_prefill_f16_paged()
            && paddock_models::dev_var_os!("PADDOCK_OCR_NO_WMMA").is_none();

        for (li, layer) in self.layers.iter().enumerate() {
            let sc = &mut bs.sc;
            if glue_fuse && !r1 {
                // norm + quantize in one pass - xn still lands (mmq reads
                // xq/xs; nothing else reads xn before the next writer)
                exec.rmsnorm_quant_q8_batch(
                    &sc.x,
                    &layer.attn_norm.buf,
                    &mut sc.xn,
                    &mut sc.xq,
                    &mut sc.xs,
                    embd,
                    eps,
                    r,
                )?;
            } else {
                exec.rmsnorm_batch(&sc.x, &layer.attn_norm.buf, &mut sc.xn, embd, eps, r)?;
            }
            if pf {
                // r-INVARIANT prefill ladder (granite's law): a radix-resumed
                // tail must reproduce the cold chunk's bytes exactly, and the
                // decode-side mmq rungs elect by r. Int8 dots are exact, so
                // one ladder = one bit pattern at any row count - this is what
                // makes "resumed head logits == cold head logits" a bytes
                // claim instead of a tolerance (measured before the switch:
                // one flipped near-tie router expert, 0.84 on a logit).
                prefill_quant(&exec, &mut sc.xq, &mut sc.xs, &mut sc.yq, &sc.xn, embd, r)?;
                prefill_mm_pre_any(
                    &exec,
                    &layer.wq,
                    &sc.xq,
                    &sc.xs,
                    &sc.yq,
                    &mut sc.xsums,
                    &mut sc.ssums,
                    &mut sc.skfix,
                    &mut sc.q,
                    r,
                )?;
                prefill_mm_pre_any(
                    &exec,
                    &layer.wk,
                    &sc.xq,
                    &sc.xs,
                    &sc.yq,
                    &mut sc.xsums,
                    &mut sc.ssums,
                    &mut sc.skfix,
                    &mut sc.k,
                    r,
                )?;
                prefill_mm_pre_any(
                    &exec,
                    &layer.wv,
                    &sc.xq,
                    &sc.xs,
                    &sc.yq,
                    &mut sc.xsums,
                    &mut sc.ssums,
                    &mut sc.skfix,
                    &mut sc.v,
                    r,
                )?;
            } else if r1 {
                gemv_any(&exec, &layer.wq, &sc.xn, &mut sc.q)?;
                gemv_any(&exec, &layer.wk, &sc.xn, &mut sc.k)?;
                gemv_any(&exec, &layer.wv, &sc.xn, &mut sc.v)?;
            } else {
                if !glue_fuse {
                    exec.quantize_q8(&sc.xn, &mut sc.xq, &mut sc.xs, r * embd)?;
                }
                mmq_pre_any(
                    &exec,
                    &layer.wq,
                    &sc.xq,
                    &sc.xs,
                    &mut sc.ssums,
                    &mut sc.part,
                    &mut sc.q,
                    r,
                )?;
                mmq_pre_any(
                    &exec,
                    &layer.wk,
                    &sc.xq,
                    &sc.xs,
                    &mut sc.ssums,
                    &mut sc.part,
                    &mut sc.k,
                    r,
                )?;
                mmq_pre_any(
                    &exec,
                    &layer.wv,
                    &sc.xq,
                    &sc.xs,
                    &mut sc.ssums,
                    &mut sc.part,
                    &mut sc.v,
                    r,
                )?;
            }
            // NEOX rope on the true positions; append at the WRITE slots. On
            // decode passes the ring twin folds all four launches into one
            // (rope by d_pos, append at d_wpos - the two-stream form the
            // one-stream fusions couldn't serve); roped k goes straight to
            // the pool (nothing reads sc.k after the append).
            let kvs = &mut bs.kv[li];
            if let (Some(d), Some(rt)) = (dash_p, layer.router.as_ref()) {
                // SR-SWA: the router's [q_r | k_r] off the same normed rows,
                // its norms + the page-2D rope on q / k / q_r / k_r, the K / V
                // appends, then this layer's block summaries, the per-row
                // route / select and the sparse attention into sc.attn
                let dd = bs
                    .dash
                    .as_mut()
                    .expect("router state enabled with the lane");
                exec.matvec_f32_batch(&rt.wqk, &sc.xn, &mut dd.rqk, r)?;
                exec.dash_rope2d(
                    &mut sc.q,
                    &mut sc.k,
                    &mut dd.rqk,
                    DASH_D,
                    2 * DASH_D,
                    &rt.q_norm1,
                    &rt.k_norm1,
                    eps,
                    &dd.d_coords,
                    &dd.inv_tok,
                    r,
                    nh,
                )?;
                for (src, dst) in [(&sc.k, &mut kvs.k), (&sc.v, &mut kvs.v)] {
                    exec.kv_append_batch_paged(
                        src,
                        dst,
                        &sc.d_wpos,
                        Some(&sc.d_slots),
                        &bs.d_bt,
                        bps,
                        kv_dim,
                        r,
                        kv_dtype,
                    )?;
                }
                exec.dash_summary(
                    &dd.rqk,
                    DASH_D,
                    2 * DASH_D,
                    &mut dd.staging[li],
                    &mut dd.kbar[li],
                    &rt.head_cls.buf,
                    &dd.d_items,
                    dd.n_items,
                    dd.kbar_blocks,
                )?;
                exec.dash_route(
                    &dd.rqk,
                    2 * DASH_D,
                    &dd.kbar[li],
                    dd.kbar_blocks,
                    &dd.d_rinfo,
                    &mut dd.sel_blk,
                    &mut dd.sel_bias,
                    &mut dd.sel_n,
                    r,
                    d.local_blocks,
                    d.entmax_window,
                    d.topk_blocks,
                    d.score_scale,
                    d.floor,
                    1.0 / d.sigma,
                )?;
                exec.dash_attn_f16(
                    &sc.q,
                    &kvs.k,
                    &kvs.v,
                    &mut sc.attn,
                    &bs.d_bt,
                    bps,
                    &dd.d_ainfo,
                    &dd.sel_blk,
                    &dd.sel_bias,
                    &dd.sel_n,
                    r,
                    nh,
                    scale,
                    d.local_window,
                )?;
            } else {
                if rope_fuse {
                    exec.rope_qk_append_paged_ring(
                        &mut sc.q,
                        &mut sc.k,
                        &sc.v,
                        &mut kvs.k,
                        &mut kvs.v,
                        &sc.d_pos,
                        &sc.d_wpos,
                        Some(&sc.d_slots),
                        &bs.d_bt,
                        bps,
                        nh,
                        n_kv,
                        hd,
                        rope,
                        r,
                        true,
                        kv_dtype,
                    )?;
                } else {
                    exec.rope_yarn_batch(&mut sc.q, &sc.d_pos, nh, hd, rope, r)?;
                    exec.rope_yarn_batch(&mut sc.k, &sc.d_pos, n_kv, hd, rope, r)?;
                    exec.kv_append_batch_paged(
                        &sc.k,
                        &mut kvs.k,
                        &sc.d_wpos,
                        Some(&sc.d_slots),
                        &bs.d_bt,
                        bps,
                        kv_dim,
                        r,
                        kv_dtype,
                    )?;
                    exec.kv_append_batch_paged(
                        &sc.v,
                        &mut kvs.v,
                        &sc.d_wpos,
                        Some(&sc.d_slots),
                        &bs.d_bt,
                        bps,
                        kv_dim,
                        r,
                        kv_dtype,
                    )?;
                }
                // Attention bounds come from d_apos - clamped once the ring is
                // full, equal to d_pos everywhere else.
                match cuts {
                    Some(c) => {
                        let all: &[(usize, usize)] = if c.runs.len() == 1 && c.dec == 0 {
                            &[(0, r)]
                        } else {
                            &c.runs
                        };
                        if c.dec > 0 {
                            // decode rows sit at [0, c.dec) so the batch-shaped
                            // partial covers them exactly (row_off is always 0)
                            let ns = attn_splits_for(nh, c.dec, exec.sm_count());
                            if ns > 1 && exec.has_attn_partial_batch_paged() {
                                exec.attn_partial_batch_paged(
                                    &sc.q,
                                    &kvs.k,
                                    &kvs.v,
                                    &mut sc.attn_o,
                                    &mut sc.attn_ml,
                                    &sc.d_apos,
                                    Some(&sc.d_slots),
                                    &bs.d_bt,
                                    bps,
                                    nh,
                                    n_kv,
                                    hd,
                                    kv_dim,
                                    0,
                                    ns,
                                    c.dec,
                                    scale,
                                    kv_dtype,
                                )?;
                                exec.attn_combine_batch(
                                    &sc.attn_o,
                                    &sc.attn_ml,
                                    &sc.sinks,
                                    &mut sc.attn,
                                    nh,
                                    hd,
                                    ns,
                                    c.dec,
                                )?;
                            } else {
                                exec.attn_decode_batch_rows_paged(
                                    &sc.q,
                                    &kvs.k,
                                    &kvs.v,
                                    &sc.sinks,
                                    &mut sc.attn,
                                    &sc.d_apos,
                                    Some(&sc.d_slots),
                                    &bs.d_bt,
                                    bps,
                                    nh,
                                    n_kv,
                                    hd,
                                    kv_dim,
                                    0,
                                    0,
                                    c.dec,
                                    scale,
                                    kv_dtype,
                                )?;
                            }
                        }
                        for &(off, len) in all {
                            if off < c.dec {
                                continue;
                            }
                            if wmma_pf {
                                exec.attn_prefill_f16_paged_at(
                                    &sc.q,
                                    &kvs.k,
                                    &kvs.v,
                                    &sc.sinks,
                                    &mut sc.attn,
                                    &sc.d_apos,
                                    &sc.d_slots,
                                    off,
                                    &bs.d_bt,
                                    bps,
                                    nh,
                                    n_kv,
                                    hd,
                                    kv_dim,
                                    0,
                                    len,
                                    scale,
                                    kv_dtype,
                                )?;
                            } else {
                                exec.attn_decode_batch_rows_paged(
                                    &sc.q,
                                    &kvs.k,
                                    &kvs.v,
                                    &sc.sinks,
                                    &mut sc.attn,
                                    &sc.d_apos,
                                    Some(&sc.d_slots),
                                    &bs.d_bt,
                                    bps,
                                    nh,
                                    n_kv,
                                    hd,
                                    kv_dim,
                                    0,
                                    off,
                                    len,
                                    scale,
                                    kv_dtype,
                                )?;
                            }
                        }
                    }
                    None => {
                        let ns = attn_splits_for(nh, r, exec.sm_count());
                        if ns > 1 && exec.has_attn_partial_batch_paged() {
                            exec.attn_partial_batch_paged(
                                &sc.q,
                                &kvs.k,
                                &kvs.v,
                                &mut sc.attn_o,
                                &mut sc.attn_ml,
                                &sc.d_apos,
                                Some(&sc.d_slots),
                                &bs.d_bt,
                                bps,
                                nh,
                                n_kv,
                                hd,
                                kv_dim,
                                0,
                                ns,
                                r,
                                scale,
                                kv_dtype,
                            )?;
                            exec.attn_combine_batch(
                                &sc.attn_o,
                                &sc.attn_ml,
                                &sc.sinks,
                                &mut sc.attn,
                                nh,
                                hd,
                                ns,
                                r,
                            )?;
                        } else {
                            exec.attn_decode_batch_paged(
                                &sc.q,
                                &kvs.k,
                                &kvs.v,
                                &sc.sinks,
                                &mut sc.attn,
                                &sc.d_apos,
                                Some(&sc.d_slots),
                                &bs.d_bt,
                                bps,
                                nh,
                                n_kv,
                                hd,
                                kv_dim,
                                0,
                                r,
                                scale,
                                kv_dtype,
                            )?;
                        }
                    }
                }
            }
            if pf {
                prefill_quant(&exec, &mut sc.xq, &mut sc.xs, &mut sc.yq, &sc.attn, embd, r)?;
                prefill_mm_pre_any(
                    &exec,
                    &layer.wo,
                    &sc.xq,
                    &sc.xs,
                    &sc.yq,
                    &mut sc.xsums,
                    &mut sc.ssums,
                    &mut sc.skfix,
                    &mut sc.proj,
                    r,
                )?;
            } else if r1 {
                gemv_any(&exec, &layer.wo, &sc.attn, &mut sc.proj)?;
            } else {
                exec.quantize_q8(&sc.attn, &mut sc.xq, &mut sc.xs, r * embd)?;
                mmq_pre_any(
                    &exec,
                    &layer.wo,
                    &sc.xq,
                    &sc.xs,
                    &mut sc.ssums,
                    &mut sc.part,
                    &mut sc.proj,
                    r,
                )?;
            }
            // FFN pre-norm: the fused arm also emits the dp4a xq/xs the MoE
            // ops (any r) and the dense/shexp mmq ladder (r>1) eat - the
            // separate quantize below disappears. Dense at r1 rides gemv on
            // xn alone, so the fold would write unread planes there.
            let ffn_nq = glue_fuse && (matches!(&layer.ffn, DsFfn::Moe(_)) || !r1);
            if ffn_nq {
                exec.add_rmsnorm_quant_q8_batch(
                    &mut sc.x,
                    &sc.proj,
                    &layer.ffn_norm.buf,
                    &mut sc.xn,
                    &mut sc.xq,
                    &mut sc.xs,
                    embd,
                    eps,
                    r,
                )?;
            } else {
                exec.add_rmsnorm_batch(
                    &mut sc.x,
                    &sc.proj,
                    &layer.ffn_norm.buf,
                    &mut sc.xn,
                    embd,
                    eps,
                    r,
                )?;
            }

            match &layer.ffn {
                DsFfn::Dense { gate, up, down } => {
                    let n_ff = gate.dims()[1];
                    if pf {
                        prefill_quant(&exec, &mut sc.xq, &mut sc.xs, &mut sc.yq, &sc.xn, embd, r)?;
                        prefill_mm_pre_any(
                            &exec,
                            gate,
                            &sc.xq,
                            &sc.xs,
                            &sc.yq,
                            &mut sc.xsums,
                            &mut sc.ssums,
                            &mut sc.skfix,
                            &mut sc.ffn_gate,
                            r,
                        )?;
                        prefill_mm_pre_any(
                            &exec,
                            up,
                            &sc.xq,
                            &sc.xs,
                            &sc.yq,
                            &mut sc.xsums,
                            &mut sc.ssums,
                            &mut sc.skfix,
                            &mut sc.ffn_up,
                            r,
                        )?;
                        exec.swiglu(&mut sc.ffn_gate, &sc.ffn_up, r * n_ff)?;
                        prefill_quant(
                            &exec,
                            &mut sc.xq,
                            &mut sc.xs,
                            &mut sc.yq,
                            &sc.ffn_gate,
                            n_ff,
                            r,
                        )?;
                        prefill_mm_pre_any(
                            &exec,
                            down,
                            &sc.xq,
                            &sc.xs,
                            &sc.yq,
                            &mut sc.xsums,
                            &mut sc.ssums,
                            &mut sc.skfix,
                            &mut sc.proj,
                            r,
                        )?;
                    } else if r1 {
                        gemv_any(&exec, gate, &sc.xn, &mut sc.ffn_gate)?;
                        gemv_any(&exec, up, &sc.xn, &mut sc.ffn_up)?;
                        exec.swiglu(&mut sc.ffn_gate, &sc.ffn_up, n_ff)?;
                        gemv_any(&exec, down, &sc.ffn_gate, &mut sc.proj)?;
                    } else {
                        if !ffn_nq {
                            exec.quantize_q8(&sc.xn, &mut sc.xq, &mut sc.xs, r * embd)?;
                        }
                        mmq_pre_any(
                            &exec,
                            gate,
                            &sc.xq,
                            &sc.xs,
                            &mut sc.ssums,
                            &mut sc.part,
                            &mut sc.ffn_gate,
                            r,
                        )?;
                        mmq_pre_any(
                            &exec,
                            up,
                            &sc.xq,
                            &sc.xs,
                            &mut sc.ssums,
                            &mut sc.part,
                            &mut sc.ffn_up,
                            r,
                        )?;
                        if glue_fuse {
                            exec.swiglu_quant_q8(
                                &sc.ffn_gate,
                                &sc.ffn_up,
                                &mut sc.xq,
                                &mut sc.xs,
                                r * n_ff,
                            )?;
                        } else {
                            exec.swiglu(&mut sc.ffn_gate, &sc.ffn_up, r * n_ff)?;
                            exec.quantize_q8(&sc.ffn_gate, &mut sc.xq, &mut sc.xs, r * n_ff)?;
                        }
                        mmq_pre_any(
                            &exec,
                            down,
                            &sc.xq,
                            &sc.xs,
                            &mut sc.ssums,
                            &mut sc.part,
                            &mut sc.proj,
                            r,
                        )?;
                    }
                }
                DsFfn::Moe(w) => {
                    // router in f32 off the UNQUANTIZED normed rows, then the
                    // DeepSeek-greedy epilogue (full-softmax weights, slot 381)
                    exec.matvec_f32_batch(&w.router_w, &sc.xn, &mut sc.moe_logits, r)?;
                    exec.moe_topk_softmax_all_batch(
                        &sc.moe_logits,
                        n_expert,
                        n_active,
                        &mut sc.moe_idx,
                        &mut sc.moe_w,
                        r,
                    )?;
                    //  diagnostic: real uniq-experts-per-(tick,layer)
                    // histogram - the number that prices the decode dp4a
                    // pair's true weight bytes. Launch-only (~2us), baked
                    // into captured graphs, off unless PADDOCK_MOE_UNIQ.
                    if sc.moe_uniq_dev != 0 {
                        exec.moe_uniq_hist(&sc.moe_idx, r * n_active, n_expert, sc.moe_uniq_dev)?;
                    }
                    // the MoE ops eat the dp4a-class xq/xs regardless of the
                    // GEMM ladder; per-32-group int8 quantize is r-invariant,
                    // so this stays one numeric class across cold and resume.
                    // On decode passes the fused pre-norm above already wrote
                    // xq/xs (identical values).
                    if !ffn_nq {
                        exec.quantize_q8(&sc.xn, &mut sc.xq, &mut sc.xs, r * embd)?;
                    }
                    // Routed pair election: PREFILL ticks take the sorted
                    // int8-MMA pair over the moe_align layout - each touched
                    // expert's weights read once per pass - while decode
                    // ticks keep the token-batched dp4a pair (sorted grids
                    // are mostly PAD at decode width). Before this arm
                    // existed the dp4a pair ran at prefill width: 6.4 ms/
                    // launch at r=907, 49% of all gpu busy on the c4 OCR
                    // probe. The split is by PHASE, not pair count: a
                    // pairs-floor election put an 11-row radix RESUME on
                    // dp4a while the cold 907-row pass ran sorted, and the
                    // classes disagree in the last f32 bit - cold vs cached
                    // greedy flipped a box coordinate (68->70) on the battery
                    // page. Same page must parse identically with and
                    // without the cache, so every prefill row rides one
                    // class (the ladder is r-invariant by construction).
                    // Kill-switches are the shared QMOE family
                    // (PADDOCK_NO_SORTED_QMOE / PADDOCK_NO_QMOE_MMA /
                    // PADDOCK_NO_QMOE_BM64).
                    let pairs = r * n_active;
                    let mma = exec
                        .kernels()
                        .map(|k| {
                            k.q8_0_moe_gate_up_mma.is_some()
                                && k.q8_0_moe_down_mma.is_some()
                                && k.moe_align.is_some()
                                && k.moe_slot_combine.is_some()
                        })
                        .unwrap_or(false)
                        && exec.compute_capability().0 >= 8
                        && paddock_models::dev_var_os!("PADDOCK_NO_QMOE_MMA").is_none();
                    // Read-once at decode was measured closed at c8 - three
                    // dedup shapes, all lost to this per-pair streaming. The
                    // uniq histogram prices c8 decode at ~31 uniq experts
                    // under 48 pairs - only 1.55x redundant bytes - and every
                    // dedup shape paid more than that in shape taxes: (a)
                    // sorted mma at decode width (pad-block waste + align
                    // fixed costs); (b) a bit-exact dp4a grouped pair, one
                    // block per (out-row, expert), row staged to shared -
                    // gate_up med 160.9 us / down 111.7 vs this pair band's
                    // 74.1 + 36.9 (the per-CTA scan->stage->dot chain
                    // dominates at 896-1360 B rows, ~1.55 tokens/expert); (c)
                    // the walk repair, out-row tiles with register-hoisted
                    // rows - 150.9 / 47.9 (block-wide bit-exact reductions
                    // serialize (row x pair) rounds; even the barrier-free
                    // warp-per-row down reached only ~55% of roof vs this
                    // kernel's ~100%).
                    //
                    // WIDE decode is the other side of that bar (2026-11-10,
                    // GB10, Unlimited-OCR-2 battery, end to end): at c32 the
                    // pairs are 192 over at most 64 experts and the dp4a pair
                    // re-streams each expert ~3x - 1.04 + 0.53 ms a layer,
                    // the step's largest cost. The sorted pair (a) reads each
                    // touched expert once: pages/s c32 +57%, c16 +25%, c8
                    // +4%, while at c4 (24 pairs) it LOSES 9% - its pad and
                    // align costs are fixed. So decode takes it from 3/4 of a
                    // pair per expert up. Numerically that makes a decode
                    // step's MoE class depend on the batch width (prefill
                    // rows stay one class, the cold-vs-cached rule above);
                    // this family's decode is not row-count invariant anyway.
                    let wide_pairs: usize =
                        paddock_models::dev_var!("PADDOCK_QMOE_DECODE_SORTED_PAIRS")
                            .ok()
                            .and_then(|v| v.parse().ok())
                            .unwrap_or((3 * n_expert).div_ceil(4));
                    let sorted = mma
                        && (pf || pairs >= wide_pairs)
                        && paddock_models::dev_var_os!("PADDOCK_NO_SORTED_QMOE").is_none();
                    if sorted {
                        // BM=64 halves the weight re-reads once blocks
                        // populate (fill bar from the qwen measurement:
                        // pairs >= n_expert * fill); PAD rows are zeros
                        // either way, so BM only regroups tokens.
                        let bm64_fill: usize = paddock_models::dev_var!("PADDOCK_QMOE_BM64_FILL")
                            .ok()
                            .and_then(|v| v.parse().ok())
                            .unwrap_or(64);
                        let bm64 = exec
                            .kernels()
                            .map(|k| k.moe_align_bm.is_some())
                            .unwrap_or(false)
                            && paddock_models::dev_var_os!("PADDOCK_NO_QMOE_BM64").is_none()
                            && pairs >= n_expert * bm64_fill;
                        let bm = if bm64 { 64usize } else { 32usize };
                        let max_blocks = (pairs + n_expert * (bm - 1)).div_ceil(bm);
                        if bm == 64 {
                            exec.moe_align_bm(
                                &sc.moe_idx,
                                &mut sc.moe_srow,
                                &mut sc.moe_sslot,
                                &mut sc.moe_bexp,
                                r,
                                n_active,
                                n_expert,
                                bm,
                                max_blocks,
                            )?;
                        } else {
                            exec.moe_align(
                                &sc.moe_idx,
                                &mut sc.moe_srow,
                                &mut sc.moe_sslot,
                                &mut sc.moe_bexp,
                                r,
                                n_active,
                                n_expert,
                                max_blocks,
                            )?;
                        }
                        // mma gate_up quantizes its SwiGLU output in registers
                        // (fq/fs direct - the separate quantize pass disappears)
                        exec.q8_0_moe_gate_up_mma(
                            &w.gate_exps,
                            &w.up_exps,
                            &sc.moe_srow,
                            &sc.moe_bexp,
                            &sc.xq,
                            &sc.xs,
                            &mut sc.moe_fq,
                            &mut sc.moe_fs,
                            max_blocks,
                            bm,
                        )?;
                        exec.q8_0_moe_down_mma(
                            &w.down_exps,
                            &sc.moe_srow,
                            &sc.moe_sslot,
                            &sc.moe_bexp,
                            &sc.moe_w,
                            &sc.moe_fq,
                            &sc.moe_fs,
                            &mut sc.moe_part,
                            n_active,
                            max_blocks,
                            bm,
                        )?;
                        // fixed-order fold over slots into a zeroed proj -
                        // bit-reproducible where an atomic scatter is not
                        exec.stream.memset_zeros(&mut sc.proj).map_err(drv)?;
                        exec.moe_slot_combine(&sc.moe_part, &mut sc.proj, embd, n_active, r)?;
                    } else {
                        exec.q8_0_moe_gate_up(
                            &w.gate_exps,
                            &w.up_exps,
                            &sc.moe_idx,
                            &sc.xq,
                            &sc.xs,
                            &mut sc.moe_fused,
                            n_active,
                            r,
                        )?;
                        exec.quantize_q8(
                            &sc.moe_fused,
                            &mut sc.moe_fq,
                            &mut sc.moe_fs,
                            r * n_active * moe_ff,
                        )?;
                        exec.q8_0_moe_down(
                            &w.down_exps,
                            &sc.moe_idx,
                            &sc.moe_w,
                            &sc.moe_fq,
                            &sc.moe_fs,
                            &mut sc.proj,
                            n_active,
                            r,
                        )?;
                    }
                    // shared expert: same normed rows, ungated, always added.
                    if pf {
                        prefill_quant(&exec, &mut sc.xq, &mut sc.xs, &mut sc.yq, &sc.xn, embd, r)?;
                        prefill_mm_pre_any(
                            &exec,
                            &w.shexp_gate,
                            &sc.xq,
                            &sc.xs,
                            &sc.yq,
                            &mut sc.xsums,
                            &mut sc.ssums,
                            &mut sc.skfix,
                            &mut sc.ffn_gate,
                            r,
                        )?;
                        prefill_mm_pre_any(
                            &exec,
                            &w.shexp_up,
                            &sc.xq,
                            &sc.xs,
                            &sc.yq,
                            &mut sc.xsums,
                            &mut sc.ssums,
                            &mut sc.skfix,
                            &mut sc.ffn_up,
                            r,
                        )?;
                        exec.swiglu(&mut sc.ffn_gate, &sc.ffn_up, r * shexp_ff)?;
                        prefill_quant(
                            &exec,
                            &mut sc.xq,
                            &mut sc.xs,
                            &mut sc.yq,
                            &sc.ffn_gate,
                            shexp_ff,
                            r,
                        )?;
                        prefill_mm_pre_any(
                            &exec,
                            &w.shexp_down,
                            &sc.xq,
                            &sc.xs,
                            &sc.yq,
                            &mut sc.xsums,
                            &mut sc.ssums,
                            &mut sc.skfix,
                            &mut sc.sh_out,
                            r,
                        )?;
                    } else if r1 {
                        gemv_any(&exec, &w.shexp_gate, &sc.xn, &mut sc.ffn_gate)?;
                        gemv_any(&exec, &w.shexp_up, &sc.xn, &mut sc.ffn_up)?;
                        exec.swiglu(&mut sc.ffn_gate, &sc.ffn_up, shexp_ff)?;
                        gemv_any(&exec, &w.shexp_down, &sc.ffn_gate, &mut sc.sh_out)?;
                    } else {
                        mmq_pre_any(
                            &exec,
                            &w.shexp_gate,
                            &sc.xq,
                            &sc.xs,
                            &mut sc.ssums,
                            &mut sc.part,
                            &mut sc.ffn_gate,
                            r,
                        )?;
                        mmq_pre_any(
                            &exec,
                            &w.shexp_up,
                            &sc.xq,
                            &sc.xs,
                            &mut sc.ssums,
                            &mut sc.part,
                            &mut sc.ffn_up,
                            r,
                        )?;
                        if glue_fuse {
                            exec.swiglu_quant_q8(
                                &sc.ffn_gate,
                                &sc.ffn_up,
                                &mut sc.xq,
                                &mut sc.xs,
                                r * shexp_ff,
                            )?;
                        } else {
                            exec.swiglu(&mut sc.ffn_gate, &sc.ffn_up, r * shexp_ff)?;
                            exec.quantize_q8(&sc.ffn_gate, &mut sc.xq, &mut sc.xs, r * shexp_ff)?;
                        }
                        mmq_pre_any(
                            &exec,
                            &w.shexp_down,
                            &sc.xq,
                            &sc.xs,
                            &mut sc.ssums,
                            &mut sc.part,
                            &mut sc.sh_out,
                            r,
                        )?;
                    }
                    exec.add(&mut sc.proj, &sc.sh_out, r * embd)?;
                }
            }
            exec.add(&mut sc.x, &sc.proj, r * embd)?;
            // triage: PADDOCK_OCR_BATCH_DUMP=1 prints the residual sum per
            // layer, PREFILL passes only (a readback inside the decode graph
            // capture wedges the capture). Synchronizes - never on in serving.
            if pf && paddock_models::dev_var_os!("PADDOCK_OCR_BATCH_DUMP").is_some() {
                let h = exec.to_host_len(&sc.x, r * embd)?;
                let s: f64 = h.iter().map(|&v| v as f64).sum();
                eprintln!("ocr-batch layer-{li} r={r}: {s:.6}");
            }
        }
        Ok(())
    }

    pub(super) fn head_rows(&mut self, rows: usize) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let (embd, eps) = (self.hp.n_embd, self.hp.rms_eps);
        let bs = self.batch.as_mut().expect("batch enabled");
        let sc = &mut bs.sc;
        if rows == 1 {
            exec.rmsnorm_batch(&sc.x, &self.output_norm.buf, &mut sc.xn, embd, eps, rows)?;
            gemv_any(&exec, &self.lm_head, &sc.xn, &mut sc.head_logits)?;
        } else if exec.has_rmsnorm_quant_q8()
            && paddock_models::dev_var_os!("PADDOCK_OCR_NO_GLUE_FUSE").is_none()
        {
            exec.rmsnorm_quant_q8_batch(
                &sc.x,
                &self.output_norm.buf,
                &mut sc.xn,
                &mut sc.xq,
                &mut sc.xs,
                embd,
                eps,
                rows,
            )?;
            mmq_pre_any(
                &exec,
                &self.lm_head,
                &sc.xq,
                &sc.xs,
                &mut sc.ssums,
                &mut sc.part,
                &mut sc.head_logits,
                rows,
            )?;
        } else {
            exec.rmsnorm_batch(&sc.x, &self.output_norm.buf, &mut sc.xn, embd, eps, rows)?;
            exec.quantize_q8(&sc.xn, &mut sc.xq, &mut sc.xs, rows * embd)?;
            mmq_pre_any(
                &exec,
                &self.lm_head,
                &sc.xq,
                &sc.xs,
                &mut sc.ssums,
                &mut sc.part,
                &mut sc.head_logits,
                rows,
            )?;
        }
        // the page-count guard on every head row, before any sampler sees it
        // (modes uploaded per pass; a prompt's first head sets row 0)
        if let (Some(d), Some(dd)) = (self.hp.dash, bs.dash.as_ref()) {
            exec.dash_guard(
                &mut sc.head_logits,
                &dd.d_guard,
                rows,
                self.hp.n_vocab,
                d.eos_token_id,
                d.page_token_id,
            )?;
        }
        Ok(())
    }

    fn head_row_at(&mut self, row: usize) -> Result<(), GpuModelError> {
        let n_embd = self.hp.n_embd;
        if row > 0 {
            let exec = self.exec.clone();
            let bs = self.batch.as_mut().expect("batch enabled");
            let sc = &mut bs.sc;
            let src =
                sc.x.try_slice(row * n_embd..(row + 1) * n_embd)
                    .ok_or_else(|| GpuError::Driver("x row slice".into()))?;
            let mut dst = sc
                .proj
                .try_slice_mut(0..n_embd)
                .ok_or_else(|| GpuError::Driver("proj row slice".into()))?;
            exec.stream.memcpy_dtod(&src, &mut dst).map_err(drv)?;
            let ps = sc
                .proj
                .try_slice(0..n_embd)
                .ok_or_else(|| GpuError::Driver("proj src slice".into()))?;
            let mut xd =
                sc.x.try_slice_mut(0..n_embd)
                    .ok_or_else(|| GpuError::Driver("x dst slice".into()))?;
            exec.stream.memcpy_dtod(&ps, &mut xd).map_err(drv)?;
        }
        self.head_rows(1)
    }

    pub(super) fn head_row(&mut self, row: usize) -> Result<Vec<f32>, GpuModelError> {
        let n_vocab = self.hp.n_vocab;
        self.head_row_at(row)?;
        let bs = self.batch.as_ref().expect("batch enabled");
        let v = bs
            .sc
            .head_logits
            .try_slice(0..n_vocab)
            .ok_or_else(|| GpuError::Driver("head row slice".into()))?;
        Ok(self.exec.stream.clone_dtoh(&v).map_err(drv)?)
    }

    /// Read the [rows, vocab] logits back to the host.
    pub fn read_batch_logits(&mut self, rows: usize) -> Result<Vec<f32>, GpuModelError> {
        let vocab = self.hp.n_vocab;
        let bs = self.batch.as_ref().expect("batch enabled");
        let v = bs
            .sc
            .head_logits
            .try_slice(0..rows * vocab)
            .ok_or_else(|| GpuError::Driver("batch logits slice".into()))?;
        // tickseg caveat: clone_dtoh syncs, so this span is GPU-tail-wait +
        // copy, an upper bound on the pure PCIe cost - split it with an
        // explicit pre-sync only if the ledger makes this segment the front
        if crate::tickseg::on() {
            let t = std::time::Instant::now();
            let out = self.exec.stream.clone_dtoh(&v).map_err(drv)?;
            crate::tickseg::rb(t.elapsed(), out.len() * 4);
            return Ok(out);
        }
        Ok(self.exec.stream.clone_dtoh(&v).map_err(drv)?)
    }

    // ── device sampling (the c8 idle ledger's smp segment) ────────
    //
    // qwen3_asr's recipe verbatim: mode-1/2 `sample_rows` on head_logits, ids
    // come back as r u32s instead of the [r, vocab] plane (517 KB/row). This
    // family needs nothing new kernel-side - its always-armed no-repeat-ngram
    // guard is handled at PLAN level: the scheduler grants Device(Greedy)
    // only on ticks where the guard would ban nothing (a no-op mask leaves
    // raw logits = the device argmax input, bit-exact), and ban-live rows
    // arrive as Host rows and keep the exact readback path.

    /// Pack per-row sampler params (inv_t, u, mode, pad). Host/Hole rows stay
    /// mode 0 = untouched. RsVerify is a spec-verify plan and never reaches a
    /// dense sampled tick.
    fn pack_samp_par(plans: &[crate::generator::RowSample]) -> Vec<u32> {
        use crate::generator::RowSample;
        use crate::sampler::DevicePlan;
        let mut par = vec![0u32; plans.len() * 4];
        for (i, p) in plans.iter().enumerate() {
            match p {
                RowSample::Hole | RowSample::Host => {}
                RowSample::Device(DevicePlan::Greedy) => par[i * 4 + 2] = 1,
                RowSample::Device(DevicePlan::Categorical { inv_t, u }) => {
                    par[i * 4] = inv_t.to_bits();
                    par[i * 4 + 1] = u.to_bits();
                    par[i * 4 + 2] = 2;
                }
                // P65 TruncCat is qwen35-only (supports_host_head); skip-safe
                RowSample::Device(DevicePlan::TruncCat { .. }) => {}
                RowSample::Device(DevicePlan::RsVerify { .. })
                | RowSample::Device(DevicePlan::RsTrunc { .. }) => {}
            }
        }
        par
    }

    pub(crate) fn supports_device_sampling_impl(&self) -> bool {
        self.batch.is_some() && self.exec.has_sample_rows()
    }

    /// Sample head_logits rows 0..r on device with `plans`; only Host-plan
    /// rows pay a vocab-row readback. Assumes the head has already run.
    pub(super) fn sample_head_rows(
        &mut self,
        r: usize,
        plans: &[crate::generator::RowSample],
    ) -> Result<crate::generator::SampledStep, GpuModelError> {
        use crate::generator::{RowSample, SampledStep};
        assert_eq!(plans.len(), r, "one plan per row");
        let exec = self.exec.clone();
        let vocab = self.hp.n_vocab;
        let par = Self::pack_samp_par(plans);
        {
            let sc = &mut self.batch.as_mut().expect("batch enabled").sc;
            let mut v = sc
                .d_par
                .try_slice_mut(0..r * 4)
                .ok_or_else(|| GpuError::Driver("d_par slice".into()))?;
            exec.stream.memcpy_htod(&par, &mut v).map_err(drv)?;
            exec.sample_rows_at(&sc.head_logits, &sc.d_par, 0, &mut sc.d_out, 0, r, vocab)?;
        }
        let sc = &self.batch.as_ref().expect("batch enabled").sc;
        let ids_view = sc
            .d_out
            .try_slice(0..r)
            .ok_or_else(|| GpuError::Driver("d_out slice".into()))?;
        let ids = exec.stream.clone_dtoh(&ids_view).map_err(drv)?;
        let mut host_rows = Vec::new();
        for (i, p) in plans.iter().enumerate() {
            if matches!(p, RowSample::Host) {
                let v = sc
                    .head_logits
                    .try_slice(i * vocab..(i + 1) * vocab)
                    .ok_or_else(|| GpuError::Driver("host row slice".into()))?;
                host_rows.push((i, exec.stream.clone_dtoh(&v).map_err(drv)?));
            }
        }
        Ok(SampledStep { ids, host_rows })
    }

    /// Device-sampled decode tick: the dense batch step + sample_rows, ids
    /// come back as r u32s instead of the [r, vocab] logits plane.
    pub(crate) fn forward_batch_sampled_impl(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        plans: &[crate::generator::RowSample],
    ) -> Result<crate::generator::SampledStep, GpuModelError> {
        self.batch_step(tokens, positions)?;
        self.sample_head_rows(tokens.len(), plans)
    }

    /// Plain NEOX rope params, θ from the header - shared with forward.rs.
    pub(crate) fn rope_params(&self) -> (f32, f32, f32, f32, f32, f32) {
        use paddock_kernels::reference::ops::YarnRope;
        YarnRope::new(
            self.hp.head_dim,
            self.hp.rope_freq_base,
            1.0,
            self.hp.n_ctx_train,
            0.0,
            1.0,
            32.0,
            1.0,
        )
        .kernel_params()
    }
}
