//! Gemma 4 prefix cache, pooled edition: a `PagedRadix` over the GLOBAL
//! layers' budget-pool blocks (ZERO-COPY - a prefix hit adopts the cached
//! blocks by refcount, no page copies), plus SWA-WINDOW checkpoints as the
//! "recurrent state" analog - an SWA layer can only resume where its whole
//! 1024-token window was snapshotted, exactly like qwen35's DeltaNet states
//! (and gpt-oss P5c). Policy tuned to the agentic pattern (conversations
//! grow monotonically): one checkpoint per cached prompt, at its last full
//! page boundary - the position the next turn resumes from.
//!
//! Sharing is safe without COW because checkpoints sit on block boundaries:
//! a resumed slot re-prefills from the (block-aligned) cut, so adopted
//! blocks are never written; decode appends land beyond the prompt in
//! blocks the slot allocated itself.
//!
//! Sizes (31B, window 1024): a window checkpoint is 50×2×64 blocks×16×4096
//! ×2 = ~839 MB - big but flat, so the pool defaults small (2) and the
//! radix recycles indices on eviction. Checkpoints land STRAIGHT from the
//! slot's rings into the pool at insert time, count-sized (only the valid
//! window blocks, not the reserved blob) - safe because the ring carries a
//! whole sub-span of slack past the window (ring = (span + img + window)/16
//! + 1 ≈ 211 blocks vs win 64) and only the ≤16-token tail appends between
//!   the cut and the insert. The old two-hop design (stage blob at the cut,
//!   full-blob land at insert) moved ~10x the bytes on the compute stream and
//!   was 55% of the 128x128c32 idle; an event-gated side-stream
//!   variant measured worse (the per-item fill->land chain through
//!   one shared blob ping-pongs across streams, and the default-priority side
//!   stream starves under decode).
//!
//! Exactness: resume + tail re-prefill is the same kernel walk a cold
//! prefill runs over the same values - the long-prompt oracle gate holds
//! with reuse engaged (see gemma4_prefix_check).

use cudarc::driver::{CudaSlice, DevicePtr};

use crate::gpu::GpuError;
use crate::gpu_model::prefix_cache::BLOCK_TOKENS;
use crate::kv_tier::digest::{IdentityDigest, IdentityFields, PrivacyScope};
use crate::kv_tier::pool_tier::tier_ram_bytes;
use crate::kv_tier::{CacheNamespace, PlaneDesc, PoolTier, RamTransport};
use crate::paged_radix::PagedRadix;

use super::GpuGemma4;

/// The gemma4 tier instance: global-layer pool runs + SWA window checkpoint
/// blobs as aux components (kv-offload 1b.3 - laguna's recipe, on the
/// family it was copied from).
pub(crate) type Gemma4Tier = PoolTier<RamTransport>;

/// Don't bother resuming prefixes shorter than this (restore overhead wins).
const MIN_CACHE_PREFIX: usize = 64;
/// Don't checkpoint prompts shorter than this.
const MIN_SNAPSHOT_LEN: usize = 4 * BLOCK_TOKENS;

/// SWA-window checkpoints the pool holds, per served slot. The flat 2 this
/// replaced was LRU across the whole radix, so a cohort of agentic sessions
/// (each caching its own growing conversation) never kept more than two
/// checkpoints total - at 8 sessions no seat found its checkpoint on the
/// next turn and re-prefilled the entire history (c8 cached_tokens=0 every
/// turn, measured on GB10). One per active session is the bare minimum;
/// four leaves the LRU a few turns of slack as sessions interleave.
const PREFIX_CKPTS_PER_SLOT: usize = 4;

/// Checkpoint pool size: `PADDOCK_GEMMA4_PREFIX_CKPTS` wins outright (0 drops
/// the pool); else `PREFIX_CKPTS_PER_SLOT` per slot, capped by the
/// `PADDOCK_GEMMA4_PREFIX_STATE_MB` byte budget (8192) and never below 2.
/// One 31B window checkpoint at fp8 KV is ~210 MB, so 8 slots reserve
/// ~6.7 GB here; Muse's is ~41 MB, the 26B-A4B's ~105 MB.
fn ckpt_slots_for(slots: usize, state_bytes: usize) -> usize {
    if let Some(n) = paddock_models::dev_var!("PADDOCK_GEMMA4_PREFIX_CKPTS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        return n;
    }
    let budget_mb: usize = paddock_models::dev_var!("PADDOCK_GEMMA4_PREFIX_STATE_MB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8192);
    let by_budget = (budget_mb << 20) / state_bytes.max(1);
    (slots * PREFIX_CKPTS_PER_SLOT).min(by_budget).max(2)
}

/// Free-margin the pool keeps under retention (compaction): with
/// the pool retention-full, every admission allocates via scattered
/// on-demand evictions - cold roaming block IDs instead of the free list's
/// LIFO reuse of the same hot pages (~2.5% at salted 128x128c32,
/// presence-triggered). Evict-ahead at insert keeps this many blocks free
/// so admissions draw from the free list again. 0 disables (the old
/// behavior); sized well under the agentic retained set (~8k blocks) so
/// the moat's prefixes never feel it.
fn px_margin() -> usize {
    paddock_models::dev_var!("PADDOCK_G4_PX_MARGIN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2048)
}

pub(crate) struct Gemma4Prefix {
    pub radix: PagedRadix,
    /// T1 tier over the global-layer pool + checkpoint blobs (1b.3).
    pub tier: Option<Gemma4Tier>,
    /// SWA window checkpoint pool: [n_states, state_bytes] raw bytes
    d_ckpt: CudaSlice<f32>,
    /// checkpoint bytes = swa_layers × K+V × win_blocks × 16 × kv_dim × 2
    state_bytes: usize,
    /// window in blocks (64 at window 1024)
    win_blocks: usize,
    /// tokens reused per slot since the last `take_prefill_reused`
    pub last_reused: Vec<usize>,
}

impl GpuGemma4 {
    /// SWA checkpoint state size in bytes (0 = prefix cache not applicable).
    fn prefix_state_bytes(&self) -> usize {
        let kv_dim_swa = self
            .layers
            .iter()
            .find(|l| l.is_swa)
            .map(|l| l.n_kv_heads * l.head_dim)
            .unwrap_or(0);
        let n_swa = self.layers.iter().filter(|l| l.is_swa).count();
        if kv_dim_swa == 0 || !self.hp.swa_window.is_multiple_of(BLOCK_TOKENS) {
            return 0;
        }
        let win_blocks = self.hp.swa_window / BLOCK_TOKENS;
        // element size follows the SWA cache dtype (1 under KV8, 2 for f16)
        let eb = self
            .kv
            .iter()
            .zip(self.layers.iter())
            .find(|(_, l)| l.is_swa)
            .map(|(k, _)| k.dtype.bytes())
            .unwrap_or(2);
        n_swa * 2 * win_blocks * BLOCK_TOKENS * kv_dim_swa * eb
    }

    /// VRAM `build_prefix` will claim (the checkpoint pool) - the reserve
    /// enable_batch carves out of the slot-fit budget.
    pub(crate) fn prefix_vram_estimate(&self, slots: usize) -> usize {
        if paddock_models::dev_var_os!("PADDOCK_NO_PREFIX_CACHE").is_some() {
            return 0;
        }
        let state_bytes = self.prefix_state_bytes();
        ckpt_slots_for(slots, state_bytes) * state_bytes
    }

    /// Build the prefix cache (called from enable_batch; POOL mode only -
    /// prefix hits adopt shared pool blocks, so dense globals can't share).
    pub(crate) fn build_prefix(&mut self, slots: usize) -> Result<(), GpuError> {
        self.prefix = None;
        if paddock_models::dev_var_os!("PADDOCK_NO_PREFIX_CACHE").is_some()
            || self.paging.is_none()
            || self.gpool.is_none()
        {
            return Ok(());
        }
        let state_bytes = self.prefix_state_bytes();
        if state_bytes == 0 {
            return Ok(());
        }
        let win_blocks = self.hp.swa_window / BLOCK_TOKENS;
        let n_states = ckpt_slots_for(slots, state_bytes);
        let mut radix = PagedRadix::new();
        radix.set_state_capacity(n_states as u32);
        let d_ckpt = self
            .exec
            .stream
            .alloc_zeros::<f32>(n_states * state_bytes.div_ceil(4))
            .map_err(|e| GpuError::Driver(e.to_string()))?;
        // KV tier (1b.3, dev flag): global-layer pool planes; SWA window
        // blobs ride as aux components. Loud decline - serving continues
        // untiered on any failure.
        let tier = match tier_ram_bytes() {
            Some(ram) => {
                let mut planes = Vec::new();
                for (lw, kvl) in self.layers.iter().zip(self.kv.iter()) {
                    if lw.is_swa {
                        continue;
                    }
                    let stride = (16 * kvl.kv_dim * kvl.dtype.bytes()) as u64;
                    for plane in [&kvl.k, &kvl.v] {
                        let (pp, _g) = plane.device_ptr(&self.exec.stream);
                        planes.push(PlaneDesc {
                            base: pp,
                            stride,
                            bytes: stride,
                        });
                    }
                }
                let content_id = self.content_id;
                let architecture = format!(
                    "gemma4 v1 global_layers={} max_ctx={} swa_win={} state_bytes={}",
                    planes.len() / 2,
                    self.max_ctx,
                    self.hp.swa_window,
                    state_bytes,
                );
                let ns = CacheNamespace {
                    identity: IdentityDigest::compute(&IdentityFields {
                        model_tensors: &content_id.0,
                        adapter: b"",
                        architecture: architecture.as_bytes(),
                        cache_schema: b"pool-planes k/v interleaved + swa-ckpt aux v1",
                        layout_abi: 1,
                        tokenizer: &content_id.1,
                    }),
                    scope: PrivacyScope::Shared,
                };
                let transport = match crate::kv_tier::pool_tier::nvme_dir_for(&ns) {
                    Some((dir, quota)) => RamTransport::with_t2(&self.exec, ram, &dir, quota),
                    None => RamTransport::new(&self.exec, ram),
                };
                match transport
                    .map_err(|e| e.to_string())
                    .and_then(|t| PoolTier::new(&ns, planes, ram, t).map_err(|e| e.to_string()))
                {
                    Ok(mut t) => {
                        t.preload_from_t2();
                        radix.set_tier_root(t.tier_root());
                        Some(t)
                    }
                    Err(e) => {
                        tracing::warn!(err = %e, "gemma4 KV tier declined");
                        None
                    }
                }
            }
            None => None,
        };
        self.prefix = Some(Gemma4Prefix {
            radix,
            tier,
            d_ckpt,
            state_bytes,
            win_blocks,
            last_reused: vec![0; slots],
        });
        Ok(())
    }

    /// Try to resume `tokens` in `slot` from the cache: adopt the cached
    /// global-layer blocks (zero copy, refcounted) + restore the SWA window
    /// checkpoint, return the resume position (0 = cold). Only positions
    /// with a checkpoint are resumable.
    pub(crate) fn prefix_resume(&mut self, slot: usize, tokens: &[u32]) -> Result<usize, GpuError> {
        // DFlash coverage across a restore. The ring holds features this
        // slot actually walked, and a feature at position i is a function of
        // tokens[..=i] alone - so whatever prefix of the incoming sequence
        // this slot already walked is still VALID, exactly like the target
        // KV blocks the restore adopts below. Clearing unconditionally (the
        // bring-up rule) turned every prefix hit into a cold drafter: a
        // second turn resumed at pos with coverage starting at pos, which
        // can never reach a window back, so spec declined every round for
        // the whole request - and prefix-heavy agentic serving is the
        // tier-1 workload.
        // llama.cpp keeps its per-slot drafter context across a cached
        // prompt for the same reason. Truncate to the proven-equal span.
        self.dflash_trim_slot(slot, tokens);
        let Some(pf) = self.prefix.as_mut() else {
            return Ok(0);
        };
        pf.last_reused[slot] = 0;
        let mut m = pf.radix.match_full(tokens);
        // TIER (D5 park/wake): the restore is consulted and PARKED at
        // admission (`tier_prefix_loading`); an elected restore has already
        // published + attached by the time prefill runs. Pump for freshness
        // and re-match, so paths that skip the consult still pick up
        // published prefixes.
        if let (Some(tier), None, Some(gp)) = (pf.tier.as_mut(), m.ckpt, self.gpool.as_mut()) {
            tier.pump_completions(&mut pf.radix, &mut gp.pool);
            m = pf.radix.match_full(tokens);
        }
        let stats = paddock_models::dev_var_os!("PADDOCK_PREFIX_STATS").is_some();
        if stats {
            tracing::info!(
                "gemma4-prefix: slot {slot} len {} matched {} ckpt {:?}",
                tokens.len(),
                m.blocks.len() * BLOCK_TOKENS,
                m.ckpt.map(|(p, _)| p)
            );
        }
        let Some((pos, cidx)) = m.ckpt else {
            return Ok(0);
        };
        if pos < MIN_CACHE_PREFIX || pos >= tokens.len() {
            return Ok(0);
        }
        // adopt the global blocks up to the checkpoint (beyond-ckpt blocks
        // are useless - the tail re-prefill covers those positions, and
        // adopting them would let it WRITE shared blocks)
        {
            let gp = self.gpool.as_mut().expect("prefix requires the pool");
            let nb = pos / BLOCK_TOKENS;
            gp.tables[slot].clear(&mut gp.pool);
            gp.tables[slot].share_prefix(&m.blocks[..nb], &mut gp.pool);
            let base = slot * gp.bps;
            for j in 0..nb {
                gp.bt_host[base + j] = gp.tables[slot].blocks()[j];
            }
            self.exec
                .stream
                .memcpy_htod(&gp.bt_host, &mut gp.d_bt)
                .map_err(|e| GpuError::Driver(e.to_string()))?;
        }
        // SWA window: checkpoint blob -> ring blocks for logical pages
        // [pos/16 - win, pos/16), each at ring slot (j % ring)
        let pg = self.paging.as_ref().expect("prefix requires paging");
        let count = (pos / BLOCK_TOKENS).min(pf.win_blocks);
        let first = pos / BLOCK_TOKENS - count;
        let mut descs: Vec<u64> = Vec::new();
        let (cp, _g) = pf.d_ckpt.device_ptr(&self.exec.stream);
        let mut src = cp + (cidx as usize * pf.state_bytes) as u64;
        for (lw, kvl) in self.layers.iter().zip(self.kv.iter()) {
            if !lw.is_swa {
                continue;
            }
            let kv_dim = lw.n_kv_heads * lw.head_dim;
            let bt = (BLOCK_TOKENS * kv_dim * kvl.dtype.bytes()) as u64;
            for plane in [&kvl.k, &kvl.v] {
                let (pp, _g2) = plane.device_ptr(&self.exec.stream);
                for i in 0..count {
                    let j = first + i;
                    let dst_blk = slot * pg.ring + (j % pg.ring);
                    descs.extend([src + i as u64 * bt, pp + dst_blk as u64 * bt, bt]);
                }
                // blob layout reserves win_blocks slots even when count < win
                src += pf.win_blocks as u64 * bt;
            }
        }
        let d = self
            .exec
            .stream
            .clone_htod(&descs)
            .map_err(|e| GpuError::Driver(e.to_string()))?;
        self.exec.batched_copy(&d, descs.len() / 3)?;
        pf.last_reused[slot] = pos;
        Ok(pos)
    }

    /// After the tail prefill: cache the prompt's global blocks in the radix
    /// (retained in the pool, shared zero-copy with later prompts) and land
    /// the SWA window checkpoint at `ckpt_pos` (if any).
    pub(crate) fn prefix_insert(
        &mut self,
        slot: usize,
        tokens: &[u32],
        ckpt_pos: Option<usize>,
    ) -> Result<(), GpuError> {
        if self.prefix.is_none() {
            return Ok(());
        }
        // DIAGNOSTIC: keep the prefix machinery but skip the
        // whole insert side - no retention, no eviction pressure, no
        // checkpoint landing. Isolates the insert-side cost from the
        // resume walks in salted A/Bs. Never default; hits need inserts.
        if paddock_models::dev_var_os!("PADDOCK_G4_PX_NORETAIN").is_some() {
            return Ok(());
        }
        let (pf, gp) = (
            self.prefix.as_mut().expect("prefix checked above"),
            self.gpool.as_mut().expect("prefix requires the pool"),
        );
        let blocks = gp.tables[slot].blocks().to_vec();
        pf.radix.insert(tokens, &blocks, &mut gp.pool);
        // evict-ahead to the free margin (see px_margin): same eviction
        // count at steady state, moved off the admission path - and the
        // freed IDs go through the free list's LIFO reuse. Tier-aware: a
        // run whose closing leaf goes is captured into T1 first, and the
        // victims' checkpoint blobs demote too.
        let margin = crate::gpu_model::prefix_cache::evict_ahead_margin(
            px_margin(),
            gp.pool.capacity() as usize,
        );
        if let Some(tier) = pf.tier.as_mut() {
            if margin > 0 && gp.pool.free_blocks() < margin {
                let exec = self.exec.clone();
                let state = {
                    let (cp, _g) = pf.d_ckpt.device_ptr(&exec.stream);
                    Some((cp, pf.state_bytes as u64))
                };
                tier.press(&mut pf.radix, &mut gp.pool, margin, state, &mut || {
                    exec.record_event().ok()
                });
            }
            tier.pump_completions(&mut pf.radix, &mut gp.pool);
        } else if margin > 0 {
            while gp.pool.free_blocks() < margin {
                if pf.radix.evict_lru(&mut gp.pool).is_none() {
                    break;
                }
            }
        }
        if paddock_models::dev_var_os!("PADDOCK_PREFIX_STATS").is_some() {
            tracing::info!(
                "gemma4-prefix: insert slot {slot} len {} ckpt {:?}",
                tokens.len(),
                ckpt_pos
            );
        }
        if let Some(pos) = ckpt_pos
            && let Some(cidx) = pf.radix.attach_state(tokens, pos)
        {
            // land the window ending at `pos` STRAIGHT from the slot's
            // rings into its checkpoint-pool slot, count-sized - one
            // in-order copy instead of the old stage-blob two-hop, and
            // only the VALID blocks (the blob reserves win_blocks per
            // plane but restore reads exactly `count`; the full-blob
            // land moved ~9x the meaningful bytes at 128-token cells).
            // Ring safety: ring carries a sub-span of slack past the
            // window (~211 blocks vs win 64) and only the ≤16-token
            // tail appends between the cut and this insert, so the
            // window is still resident. Decode appends run after this
            // copy in stream order.
            let pg = self.paging.as_ref().expect("prefix requires paging");
            let count = (pos / BLOCK_TOKENS).min(pf.win_blocks);
            let first = pos / BLOCK_TOKENS - count;
            let (cp, _g) = pf.d_ckpt.device_ptr(&self.exec.stream);
            let mut dst = cp + (cidx as usize * pf.state_bytes) as u64;
            let mut descs: Vec<u64> = Vec::new();
            for (lw, kvl) in self.layers.iter().zip(self.kv.iter()) {
                if !lw.is_swa {
                    continue;
                }
                let kv_dim = lw.n_kv_heads * lw.head_dim;
                let bt = (BLOCK_TOKENS * kv_dim * kvl.dtype.bytes()) as u64;
                for plane in [&kvl.k, &kvl.v] {
                    let (pp, _g2) = plane.device_ptr(&self.exec.stream);
                    for i in 0..count {
                        let j = first + i;
                        let src_blk = slot * pg.ring + (j % pg.ring);
                        descs.extend([pp + src_blk as u64 * bt, dst + i as u64 * bt, bt]);
                    }
                    // blob layout reserves win_blocks slots per plane
                    // (the shape prefix_resume reads)
                    dst += pf.win_blocks as u64 * bt;
                }
            }
            let d = self
                .exec
                .stream
                .clone_htod(&descs)
                .map_err(|e| GpuError::Driver(e.to_string()))?;
            self.exec.batched_copy(&d, descs.len() / 3)?;
        }
        Ok(())
    }

    /// The checkpoint cut for a prompt: its last full page boundary (always
    /// < len, so the tail chunk after the cut is never empty).
    pub(crate) fn prefix_cut(&self, t_len: usize, start: usize) -> Option<usize> {
        if self.prefix.is_none() || t_len < MIN_SNAPSHOT_LEN {
            return None;
        }
        // tiered: align to the tier's run size (runs are the restore
        // granularity; an unaligned boundary could never resume through T1)
        let step = match self.prefix.as_ref().and_then(|pf| pf.tier.as_ref()) {
            Some(t) => t.run_blocks() * BLOCK_TOKENS,
            None => BLOCK_TOKENS,
        };
        let cut = (t_len - 1) / step * step;
        (cut > start).then_some(cut)
    }

    /// The per-tick tier pump (see `Generator::tier_pump`).
    /// The D5 admission consult (park/wake): probe + elect the hybrid
    /// two-round restore and, when elected, START it and PARK the request -
    /// `true` tells the scheduler to skip this slot this tick; the per-pass
    /// `tier_pump` drives the flow (blocks publish, then the window blob
    /// lands in a RESERVED checkpoint slot that attaches only once
    /// verified) and the request re-enters admission when it resolves.
    /// A hybrid restore is only worth anything through a BOUNDARY, so the
    /// hit truncates to the deepest affordable one; when retention crowds
    /// the destination it is pressure-demoted first (the prefix cache is
    /// reclaimable capacity).
    pub(crate) fn tier_consult_impl(&mut self, slot: usize, tokens: &[u32]) -> bool {
        use crate::kv_tier::FlowStatus;
        let exec = self.exec.clone();
        let (Some(pf), Some(gp)) = (self.prefix.as_mut(), self.gpool.as_mut()) else {
            return false;
        };
        let Some(tier) = pf.tier.as_mut() else {
            return false;
        };
        let state_bytes = pf.state_bytes as u64;
        tier.pump_completions(&mut pf.radix, &mut gp.pool);
        {
            let exec2 = exec.clone();
            tier.pump_flows(&mut pf.radix, &mut || exec2.record_event().ok());
        }
        match tier.flow_status(slot, tokens) {
            FlowStatus::Loading => return true,
            FlowStatus::Done { .. } => return false,
            FlowStatus::None => {}
        }
        // a resident usable checkpoint makes the tier moot for this prompt
        if pf.radix.match_full(tokens).ckpt.is_some() {
            return false;
        }
        // the blob's geometry lets the probe find it on any tier
        tier.declare_blob_flat(state_bytes);
        // probe counts the lookup (and the miss when nothing is held)
        let Some(hit) = tier.probe(tokens, 0) else {
            return false;
        };
        let r = tier.run_blocks();
        let deepest = tier
            .probe_aux(tokens, hit.end_block)
            .filter(|a| a.end_block * BLOCK_TOKENS >= MIN_CACHE_PREFIX && a.end_block % r == 0);
        let Some((hit, est_us)) = tier.elect_hybrid(&hit, deepest.as_ref()) else {
            return false;
        };
        let aux = deepest.expect("an elected hybrid hit has its boundary");
        // the destination: the restored blocks (the blob lands in a fixed
        // checkpoint slot, not in pool pages)
        let need = aux.end_block;
        let afford =
            |pool: &crate::kv_pool::KvPool| pool.free_blocks().saturating_sub(2 * r) / r * r;
        if afford(&gp.pool) < need {
            // retention crowds the destination: pressure-demote it (the
            // prefix cache is reclaimable capacity)
            let want = need + 2 * r;
            let after = exec.record_event().ok();
            let (_e, taken) = tier.pressure_demote(&mut pf.radix, &mut gp.pool, want, after);
            let (cp, _g) = pf.d_ckpt.device_ptr(&exec.stream);
            for t in taken {
                if t.end_block % r == 0 {
                    let base = cp + t.state_idx as u64 * state_bytes;
                    let ev = exec.record_event().ok();
                    tier.demote_aux(&mut pf.radix, t, base, state_bytes, ev);
                } else {
                    pf.radix.recycle_state(t.state_idx);
                }
            }
            let deadline = std::time::Instant::now() + std::time::Duration::from_millis(50);
            while gp.pool.free_blocks() < want && tier.stats().2 > 0 {
                tier.pump_completions(&mut pf.radix, &mut gp.pool);
                if std::time::Instant::now() >= deadline {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_micros(200));
            }
        }
        tracing::debug!(
            free = gp.pool.free_blocks(),
            need,
            boundary = aux.end_block,
            "gemma4 tier gate"
        );
        if afford(&gp.pool) < need {
            tier.refuse_park();
            return false;
        }
        let after = exec.record_event().ok();
        let (cp, _g) = pf.d_ckpt.device_ptr(&exec.stream);
        let plan = crate::kv_tier::AuxPlan {
            hit: aux,
            state_base: cp,
            state_stride: state_bytes,
        };
        match crate::kv_tier::RestoreFlow::begin(
            tier,
            &mut gp.pool,
            tokens,
            &hit,
            Some(plan),
            est_us,
            after,
        ) {
            Some(flow) => {
                tier.park_flow(slot, flow);
                tracing::debug!(
                    slot,
                    boundary = hit.end_block,
                    "gemma4 tier: restore parked (D5)"
                );
                true
            }
            None => false,
        }
    }

    pub(crate) fn tier_pump_impl(&mut self) {
        let exec = self.exec.clone();
        let (Some(pf), Some(gp)) = (self.prefix.as_mut(), self.gpool.as_mut()) else {
            return;
        };
        let state = Some(pf.tier_state_geom(&exec.stream));
        let Some(tier) = pf.tier.as_mut() else { return };
        tier.pump_completions(&mut pf.radix, &mut gp.pool);
        tier.pump_flows(&mut pf.radix, &mut || exec.record_event().ok());
        // 2.3 write-through: retained chains AND live window blobs
        // pre-store in slack so eviction (and ckpt-slot recycling) is free
        tier.mirror_slack(&pf.radix, &mut gp.pool, exec.record_event().ok(), 2, state);
    }
}

impl Gemma4Prefix {
    /// The checkpoint-pool geometry for blob demotes: (device base, stride
    /// bytes). Batch-side eviction arms need it and the fields are module-
    /// private by design.
    pub(crate) fn tier_state_geom(&self, stream: &cudarc::driver::CudaStream) -> (u64, u64) {
        use cudarc::driver::DevicePtr;
        let (cp, _g) = self.d_ckpt.device_ptr(stream);
        (cp, self.state_bytes as u64)
    }
}
