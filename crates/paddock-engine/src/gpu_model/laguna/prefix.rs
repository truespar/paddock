//! Laguna prefix cache: a `PagedRadix` over the full layers' budget-pool
//! blocks (ZERO-COPY - a hit adopts cached blocks by refcount) plus
//! SWA-WINDOW checkpoints as the resume state for the 30 ring-backed layers
//! (the gemma4 scheme; gemma4/prefix.rs is the annotated original). A prompt
//! can only resume at a position whose whole 512-token SWA window was
//! snapshotted; policy = two checkpoints per cached prompt: its last full
//! page boundary - the position the next agentic turn resumes from - and the
//! back-off boundary ~256 tokens behind it, where a prompt whose tail was
//! rewritten (the same document, another question) resumes.
//!
//! Sharing is safe without COW because checkpoints sit on block boundaries:
//! the tail re-prefill starts at the (block-aligned) cut, so adopted full-
//! layer blocks are never written.
//!
//! Sizes (XS-2.1, window 512): a checkpoint is 30 layers × K+V × 32 blocks
//! × 16 × 1024 × 2 B ≈ 63 MB - flat, so the pool defaults small and the
//! radix recycles state indices on eviction. Checkpoints land STRAIGHT from
//! the slot's rings at insert (count-sized): the ring carries a whole
//! SWA_SPAN of slack past the window (65 blocks vs win 32) and only the
//! ≤16-token tail appends between the cut and the insert, so the window is
//! still resident (the gemma4 Phase-46 one-hop design).

use cudarc::driver::{CudaSlice, DevicePtr};

use crate::gpu::GpuError;
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::prefix_cache::BLOCK_TOKENS;
use crate::kv_tier::digest::{IdentityDigest, IdentityFields, PrivacyScope};
use crate::kv_tier::pool_tier::tier_ram_bytes;
use crate::kv_tier::{CacheNamespace, Election, PlaneDesc, PoolTier, RamTransport};
use crate::paged_radix::PagedRadix;

use super::GpuLaguna;

/// The laguna tier instance: full-attention pool runs + SWA-window
/// checkpoint blobs as aux components (kv-offload 1b.3 - the first hybrid
/// family). Restored blocks alone are worthless here (resume needs the
/// window), so a tier hit is usable only to a boundary whose blob is
/// resident - exactly what `probe_aux` answers.
pub(crate) type LagunaTier = PoolTier<RamTransport>;

/// Don't bother resuming prefixes shorter than this (restore overhead wins).
const MIN_CACHE_PREFIX: usize = 64;
/// Don't checkpoint prompts shorter than this.
const MIN_SNAPSHOT_LEN: usize = 4 * BLOCK_TOKENS;

/// Window checkpoint slots. A prompt now lands two (its trailing cut and the
/// back-off cut behind it), so where a checkpoint is small - Kolibri's 40
/// fp8 SWA layers are ~21 MB - the pool holds 8; Laguna's ~63 MB f16
/// windows keep 4. PADDOCK_LAGUNA_PREFIX_CKPTS overrides.
fn ckpt_slots(state_bytes: usize) -> usize {
    paddock_models::dev_var!("PADDOCK_LAGUNA_PREFIX_CKPTS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(if state_bytes <= 32 << 20 { 8 } else { 4 })
}

/// Free-margin kept by evict-ahead at insert (see gemma4's px_margin: LIFO
/// free-list reuse beats scattered on-demand eviction on the admission path).
fn px_margin() -> usize {
    paddock_models::dev_var!("PADDOCK_LAGUNA_PX_MARGIN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1024)
}

pub(crate) struct LagunaPrefix {
    pub radix: PagedRadix,
    /// T1 tier over the full-attention pool + checkpoint blobs (1b.3).
    /// None unless the dev flag arms it and pack + host memory cooperate.
    pub tier: Option<LagunaTier>,
    /// SWA window checkpoint pool: [n_states, state_bytes] raw bytes
    d_ckpt: CudaSlice<f32>,
    /// checkpoint bytes = swa_layers × K+V × win_blocks × 16 × kv_dim × 2
    state_bytes: usize,
    /// window in blocks (32 at window 512)
    win_blocks: usize,
    /// tokens reused per slot since the last `take_prefill_reused`
    pub last_reused: Vec<usize>,
}

impl GpuLaguna {
    /// Blocks a resume needs from before a block-aligned cut: the window
    /// counts the current token, so a row at the cut reads `window - 1`
    /// earlier keys - 32 blocks for Laguna's 512 and Kolibri's 513 alike.
    fn win_blocks(&self) -> usize {
        self.hp.swa_window.saturating_sub(1).div_ceil(BLOCK_TOKENS)
    }

    /// One SWA checkpoint's size in bytes (0 = prefix cache not applicable).
    pub(crate) fn prefix_state_bytes(&self) -> usize {
        let n_swa = self.layers.iter().filter(|l| l.is_swa).count();
        if n_swa == 0 || self.hp.swa_window == 0 {
            return 0;
        }
        let kv_dim = self.hp.n_kv_heads * self.hp.head_dim;
        n_swa * 2 * self.win_blocks() * BLOCK_TOKENS * kv_dim * self.kv_dtype.bytes()
    }

    /// VRAM `build_prefix` will claim - the reserve enable_batch carves out.
    pub(crate) fn prefix_vram_estimate(&self) -> usize {
        if paddock_models::dev_var_os!("PADDOCK_NO_PREFIX_CACHE").is_some() {
            return 0;
        }
        ckpt_slots(self.prefix_state_bytes()) * self.prefix_state_bytes()
    }

    /// Build the prefix cache (end of enable_batch - the d_ckpt blob
    /// allocates dead last so every decode-path buffer keeps the addresses
    /// the prefix-off config gets, as gemma4 does).
    pub(crate) fn build_prefix(&mut self, slots: usize) -> Result<(), GpuError> {
        if paddock_models::dev_var_os!("PADDOCK_NO_PREFIX_CACHE").is_some() || self.batch.is_none()
        {
            return Ok(());
        }
        let state_bytes = self.prefix_state_bytes();
        if state_bytes == 0 {
            return Ok(());
        }
        let win_blocks = self.win_blocks();
        let n_states = ckpt_slots(state_bytes);
        let mut radix = PagedRadix::new();
        radix.set_state_capacity(n_states as u32);
        let d_ckpt = self
            .exec
            .stream
            .alloc_zeros::<f32>(n_states * state_bytes.div_ceil(4))
            .map_err(|e| GpuError::Driver(e.to_string()))?;
        // KV tier (1b.3, dev flag): full-attention pool planes; the SWA
        // checkpoint blobs ride as aux components. Loud decline on any
        // failure - serving continues untiered.
        let tier = match tier_ram_bytes() {
            Some(ram) => {
                let bs = self.batch.as_ref().expect("checked");
                let kv_dim = self.hp.n_kv_heads * self.hp.head_dim;
                let stride = (16 * kv_dim * self.kv_dtype.bytes()) as u64;
                let mut planes = Vec::new();
                for (lw, kvl) in self.layers.iter().zip(bs.kv.iter()) {
                    if lw.is_swa {
                        continue; // rings are per-slot state, not pool blocks
                    }
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
                    "laguna v1 full_layers={} kv_dim={} dtype={:?} max_ctx={} swa_win={} state_bytes={}",
                    planes.len() / 2,
                    kv_dim,
                    self.kv_dtype,
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
                // T2 attaches here exactly as it does for every other
                // family - laguna was the one that never did, so a config
                // asking for disk offload got RAM only and said nothing.
                let transport = match crate::kv_tier::pool_tier::nvme_dir_for(&ns) {
                    Some((dir, quota)) => RamTransport::with_t2(&self.exec, ram, &dir, quota),
                    None => RamTransport::new(&self.exec, ram),
                };
                match transport
                    .map_err(|e| e.to_string())
                    .and_then(|t| PoolTier::new(&ns, planes, ram, t).map_err(|e| e.to_string()))
                {
                    Ok(t) => {
                        radix.set_tier_root(t.tier_root());
                        Some(t)
                    }
                    Err(e) => {
                        tracing::warn!(err = %e, "laguna KV tier declined");
                        None
                    }
                }
            }
            None => None,
        };
        self.batch.as_mut().expect("checked").prefix = Some(LagunaPrefix {
            radix,
            tier,
            d_ckpt,
            state_bytes,
            win_blocks,
            last_reused: vec![0; slots],
        });
        Ok(())
    }

    /// Try to resume `tokens` in `slot`: adopt the cached full-layer blocks
    /// (zero copy, refcounted) + restore the SWA window checkpoint into the
    /// slot's rings. Returns the resume position (0 = cold).
    pub(crate) fn prefix_resume(
        &mut self,
        slot: usize,
        tokens: &[u32],
    ) -> Result<usize, GpuModelError> {
        let kv_dim = self.hp.n_kv_heads * self.hp.head_dim;
        let bs = self.batch.as_mut().expect("batch enabled");
        let Some(pf) = bs.prefix.as_mut() else {
            return Ok(0);
        };
        pf.last_reused[slot] = 0;
        let mut m = pf.radix.match_full(tokens);
        // TIER (D5 park/wake): the restore is consulted and PARKED at
        // admission (`tier_prefix_loading`); an elected restore has already
        // published + attached by the time prefill runs. Pump for freshness
        // and re-match, so paths that skip the consult still pick up
        // published prefixes.
        if let (Some(tier), None) = (pf.tier.as_mut(), m.ckpt) {
            tier.pump_completions(&mut pf.radix, &mut bs.pool);
            m = pf.radix.match_full(tokens);
        }
        if paddock_models::dev_var_os!("PADDOCK_PREFIX_STATS").is_some() {
            tracing::info!(
                "laguna-prefix: slot {slot} len {} matched {} ckpt {:?}",
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
        // adopt the full-layer blocks up to the checkpoint (beyond-ckpt
        // blocks would be written by the tail re-prefill - never adopt them)
        {
            let nb = pos / BLOCK_TOKENS;
            bs.tables[slot].clear(&mut bs.pool);
            bs.tables[slot].share_prefix(&m.blocks[..nb], &mut bs.pool);
            let base = slot * bs.bps;
            for j in 0..nb {
                bs.bt_host[base + j] = bs.tables[slot].blocks()[j];
            }
            self.exec
                .stream
                .memcpy_htod(&bs.bt_host, &mut bs.d_bt)
                .map_err(|e| GpuError::Driver(e.to_string()))?;
        }
        // SWA windows: checkpoint blob -> ring blocks for logical pages
        // [pos/16 - win, pos/16), each at ring slot (j % ring)
        let count = (pos / BLOCK_TOKENS).min(pf.win_blocks);
        let first = pos / BLOCK_TOKENS - count;
        let bt = (BLOCK_TOKENS * kv_dim * self.kv_dtype.bytes()) as u64;
        let mut descs: Vec<u64> = Vec::new();
        let (cp, _g) = pf.d_ckpt.device_ptr(&self.exec.stream);
        let mut src = cp + (cidx as usize * pf.state_bytes) as u64;
        for (lw, kvl) in self.layers.iter().zip(bs.kv.iter()) {
            if !lw.is_swa {
                continue;
            }
            for plane in [&kvl.k, &kvl.v] {
                let (pp, _g2) = plane.device_ptr(&self.exec.stream);
                for i in 0..count {
                    let j = first + i;
                    let dst_blk = slot * bs.ring + (j % bs.ring);
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

    /// After the tail prefill: cache the prompt's full-layer blocks in the
    /// radix, attach the windows `staged` at the scheduler's hints during the
    /// prefill, and land an SWA window checkpoint at each of `cuts`.
    pub(crate) fn prefix_insert(
        &mut self,
        slot: usize,
        tokens: &[u32],
        cuts: &[usize],
        staged: &[(usize, u32)],
    ) -> Result<(), GpuModelError> {
        let bs = self.batch.as_mut().expect("batch enabled");
        let Some(pf) = bs.prefix.as_mut() else {
            return Ok(());
        };
        let blocks = bs.tables[slot].blocks().to_vec();
        pf.radix.insert(tokens, &blocks, &mut bs.pool);
        // evict-ahead to the free margin: same steady-state eviction count,
        // moved off the admission path; freed IDs reuse via the LIFO list
        let margin = crate::gpu_model::prefix_cache::evict_ahead_margin(
            px_margin(),
            bs.pool.capacity() as usize,
        );
        if let Some(tier) = pf.tier.as_mut() {
            if margin > 0 && bs.pool.free_blocks() < margin {
                let after = self.exec.record_event().ok();
                let (_e, aux) = tier.pressure_demote(&mut pf.radix, &mut bs.pool, margin, after);
                // demote each claimed checkpoint blob as aux shards; a
                // boundary the tier cannot take (unaligned pre-alignment
                // checkpoints) just recycles
                let (cp, _g) = pf.d_ckpt.device_ptr(&self.exec.stream);
                for a in aux {
                    if a.end_block % tier.run_blocks() == 0 {
                        let base = cp + a.state_idx as u64 * pf.state_bytes as u64;
                        let ev = self.exec.record_event().ok();
                        tier.demote_aux(&mut pf.radix, a, base, pf.state_bytes as u64, ev);
                    } else {
                        pf.radix.recycle_state(a.state_idx);
                    }
                }
            }
            tier.pump_completions(&mut pf.radix, &mut bs.pool);
        } else if margin > 0 {
            while bs.pool.free_blocks() < margin {
                if pf.radix.evict_lru(&mut bs.pool).is_none() {
                    break;
                }
            }
        }
        if paddock_models::dev_var_os!("PADDOCK_PREFIX_STATS").is_some() {
            tracing::info!(
                "laguna-prefix: insert slot {slot} len {} ckpts {cuts:?}",
                tokens.len()
            );
        }
        // checkpoints staged mid-prefill at the scheduler's hints: their
        // windows already sit in reserved slots; attach (or hand back)
        for (pos, idx) in staged {
            if !pf.radix.attach_state_at(tokens, *pos, *idx) {
                pf.radix.recycle_state(*idx);
            }
        }
        let landed: Vec<(usize, u32)> = cuts
            .iter()
            .filter_map(|&pos| pf.radix.attach_state(tokens, pos).map(|c| (pos, c)))
            .collect();
        // land each window ending at its cut STRAIGHT from the slot's rings,
        // count-sized. Ring safety: 65 blocks of ring vs 32 of window, and
        // `prefix_cuts` keeps every cut whose window the ring still holds
        // (the trailing cut is <= 16 tokens back, the back-off cut ~17
        // blocks); decode appends run after this copy in stream order.
        for (pos, cidx) in landed {
            self.land_window(slot, pos, cidx)?;
        }
        Ok(())
    }

    /// The reply's first tool call just opened (`Generator::reply_pin_at`),
    /// and `history[..pos]` is what the slot's cache holds: the prompt and
    /// the reply so far - its reasoning. File that under the radix up to the
    /// last page boundary at or before `pos` and land the SWA window ending
    /// there straight from the slot's ring (written within the last 15
    /// tokens; the ring keeps a whole window of slack), so the next agentic
    /// turn - which re-sends this reasoning ahead of the call and its result -
    /// resumes past it instead of re-prefilling it from this prompt's end.
    /// The call itself stays out: a client re-serializes tool calls, so the
    /// bytes past its opening token are not the model's. Laguna caches no
    /// rolling reply checkpoint (qwen35's per-page snapshots): a Kolibri-class
    /// template drops a turn's reasoning once a new user message follows, so
    /// the end of a final answer is never re-sent as generated - the opening
    /// of a call is the one boundary a next request shares. One checkpoint a
    /// reply; the radix's LRU owns it afterwards.
    pub(crate) fn reply_pin_impl(
        &mut self,
        slot: usize,
        history: &[u32],
        pos: u32,
    ) -> Result<(), GpuModelError> {
        // Laguna's own template renders every earlier assistant turn with an
        // empty think block - a reply's reasoning never comes back, so its pin
        // could only take a window slot from prompts that do resume
        if self.hp.flavor != super::Flavor::Kolibri {
            return Ok(());
        }
        let cut = pos as usize / BLOCK_TOKENS * BLOCK_TOKENS;
        let idx = {
            let Some(bs) = self.batch.as_mut() else {
                return Ok(());
            };
            let Some(pf) = bs.prefix.as_mut() else {
                return Ok(());
            };
            if cut < MIN_SNAPSHOT_LEN
                || history.len() < cut
                || bs
                    .tables
                    .get(slot)
                    .is_none_or(|t| t.blocks().len() < cut / BLOCK_TOKENS)
            {
                return Ok(());
            }
            let blocks = bs.tables[slot].blocks()[..cut / BLOCK_TOKENS].to_vec();
            pf.radix.insert(&history[..cut], &blocks, &mut bs.pool);
            // None: the boundary is already checkpointed (a prompt cut) or
            // the state pool is off - nothing to add
            let Some(idx) = pf.radix.attach_state(&history[..cut], cut) else {
                return Ok(());
            };
            idx
        };
        if paddock_models::dev_var_os!("PADDOCK_PREFIX_STATS").is_some() {
            tracing::info!("laguna-reply-pin: slot {slot} pos {pos} cut {cut} idx {idx}");
        }
        self.land_window(slot, cut, idx)?;
        Ok(())
    }

    /// Copy the SWA window ending at `pos` (block-aligned) out of `slot`'s
    /// rings into checkpoint slot `cidx`, stream-ordered after the walk that
    /// wrote it. The caller keeps the window ring-resident: the ring holds
    /// the slot's last `ring` blocks, the window `win_blocks`.
    pub(crate) fn land_window(&self, slot: usize, pos: usize, cidx: u32) -> Result<(), GpuError> {
        let kv_dim = self.hp.n_kv_heads * self.hp.head_dim;
        let bs = self.batch.as_ref().expect("batch enabled");
        let Some(pf) = bs.prefix.as_ref() else {
            return Ok(());
        };
        let count = (pos / BLOCK_TOKENS).min(pf.win_blocks);
        let first = pos / BLOCK_TOKENS - count;
        let bt = (BLOCK_TOKENS * kv_dim * self.kv_dtype.bytes()) as u64;
        let (cp, _g) = pf.d_ckpt.device_ptr(&self.exec.stream);
        let mut dst = cp + (cidx as usize * pf.state_bytes) as u64;
        let mut descs: Vec<u64> = Vec::new();
        for (lw, kvl) in self.layers.iter().zip(bs.kv.iter()) {
            if !lw.is_swa {
                continue;
            }
            for plane in [&kvl.k, &kvl.v] {
                let (pp, _g2) = plane.device_ptr(&self.exec.stream);
                for i in 0..count {
                    let j = first + i;
                    let src_blk = slot * bs.ring + (j % bs.ring);
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
        self.exec.batched_copy(&d, descs.len() / 3)
    }

    /// Stage a hint checkpoint mid-prefill: reserve a state slot and land the
    /// window ending at `pos` (the chunk just landed there, so it is
    /// ring-resident). None when the cache is off or no slot can be had.
    pub(crate) fn stage_window(
        &mut self,
        slot: usize,
        pos: usize,
    ) -> Result<Option<u32>, GpuError> {
        let idx = {
            let bs = self.batch.as_mut().expect("batch enabled");
            let Some(pf) = bs.prefix.as_mut() else {
                return Ok(None);
            };
            match pf.radix.reserve_state_slot() {
                Some(i) => i,
                None => return Ok(None),
            }
        };
        self.land_window(slot, pos, idx)?;
        Ok(Some(idx))
    }

    /// Hand back the slots a dropped prefill had staged.
    pub(crate) fn recycle_staged(&mut self, staged: &[(usize, u32)]) {
        if let Some(pf) = self.batch.as_mut().and_then(|b| b.prefix.as_mut()) {
            for &(_, idx) in staged {
                pf.radix.recycle_state(idx);
            }
        }
    }

    /// The scheduler's shared-prefix floor (see `Generator::prefix_share_floor`):
    /// prompts sharing at least this much with an in-flight one wait for it
    /// and resume off the checkpoint its hints stage. None without a cache.
    pub(crate) fn prefix_share_floor_impl(&self) -> Option<usize> {
        self.batch.as_ref()?.prefix.as_ref()?;
        Some(MIN_CACHE_PREFIX.max(2 * BLOCK_TOKENS))
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
        let Some(bs) = self.batch.as_mut() else {
            return false;
        };
        let Some(pf) = bs.prefix.as_mut() else {
            return false;
        };
        let Some(tier) = pf.tier.as_mut() else {
            return false;
        };
        let state_bytes = pf.state_bytes as u64;
        tier.pump_completions(&mut pf.radix, &mut bs.pool);
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
        let hit = tier.probe(tokens, 0);
        let afford = |pool: &crate::kv_pool::KvPool, r: usize| {
            pool.free_blocks().saturating_sub(2 * r) / r * r
        };
        let r = tier.run_blocks();
        let mut afford_blocks = afford(&bs.pool, r);
        let deepest = hit
            .as_ref()
            .and_then(|h| tier.probe_aux(tokens, h.end_block))
            .filter(|a| a.end_block * BLOCK_TOKENS >= MIN_CACHE_PREFIX && a.end_block % r == 0);
        if let Some(a) = &deepest
            && afford_blocks < a.end_block
        {
            let want = a.end_block + 2 * r;
            let after = exec.record_event().ok();
            let (_e, taken) = tier.pressure_demote(&mut pf.radix, &mut bs.pool, want, after);
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
            // demote pins defer the frees - drain briefly (the restore
            // itself no longer waits; only this make-room drain is bounded-
            // synchronous, and only when the destination was crowded)
            let deadline = std::time::Instant::now() + std::time::Duration::from_millis(50);
            while bs.pool.free_blocks() < want && tier.stats().2 > 0 {
                tier.pump_completions(&mut pf.radix, &mut bs.pool);
                if std::time::Instant::now() >= deadline {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_micros(200));
            }
            afford_blocks = afford(&bs.pool, r);
        }
        let aux = deepest.filter(|a| a.end_block <= afford_blocks);
        let (Some(hit), Some(aux)) = (hit, aux) else {
            return false;
        };
        let n_runs = aux.end_block / r;
        let per_run = hit.bytes / hit.keys.len().max(1) as u64;
        let hit = crate::kv_tier::TierHit {
            start_block: 0,
            end_block: aux.end_block,
            bytes: per_run * n_runs as u64,
            keys: hit.keys[..n_runs.min(hit.keys.len())].to_vec(),
            // runs are equal-sized, so the truncated hit keeps the same
            // share of disk-sourced bytes as the full one
            nvme_bytes: hit.nvme_bytes.min(per_run * n_runs as u64),
        };
        let shape = crate::kv_tier::HitShape {
            restore_bytes: hit.bytes + aux.bytes,
            restore_tokens: (aux.end_block * BLOCK_TOKENS) as u32,
            queued_bytes: tier.catalog.ledger(crate::kv_tier::Tier::Ram).in_flight,
            nvme_bytes: hit.nvme_bytes,
        };
        let Election::Restore { est_us, .. } = tier.cost.elect(shape) else {
            return false;
        };
        let after = exec.record_event().ok();
        let (cp, _g) = pf.d_ckpt.device_ptr(&exec.stream);
        let plan = crate::kv_tier::AuxPlan {
            hit: aux,
            state_base: cp,
            state_stride: state_bytes,
        };
        match crate::kv_tier::RestoreFlow::begin(
            tier,
            &mut bs.pool,
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
                    "laguna tier: restore parked (D5)"
                );
                true
            }
            None => false,
        }
    }

    pub(crate) fn tier_pump_impl(&mut self) {
        let exec = self.exec.clone();
        let Some(bs) = self.batch.as_mut() else {
            return;
        };
        let Some(pf) = bs.prefix.as_mut() else { return };
        let state = Some(pf.tier_state_geom(&exec.stream));
        let Some(tier) = pf.tier.as_mut() else { return };
        tier.pump_completions(&mut pf.radix, &mut bs.pool);
        tier.pump_flows(&mut pf.radix, &mut || exec.record_event().ok());
        // 2.3 write-through: retained chains AND live window blobs
        // pre-store in slack so eviction (and ckpt-slot recycling) is free
        tier.mirror_slack(&pf.radix, &mut bs.pool, exec.record_event().ok(), 2, state);
    }

    /// The checkpoint cuts for a prompt, ascending: its last full page
    /// boundary (always < len, so the tail chunk after the cut is never
    /// empty) and the BACK-OFF boundary behind it (`backoff_cut`, ~256
    /// tokens back) - where a prompt whose tail was rewritten (the same
    /// document with another question, an edited last instruction) resumes,
    /// as the hybrids' does. Both land straight from the slot's rings at
    /// insert, so the back-off cut is kept only while its whole window is
    /// still ring-resident then.
    pub(crate) fn prefix_cuts(&self, t_len: usize, start: usize) -> Vec<usize> {
        let Some(cut) = self.prefix_cut(t_len, start) else {
            return Vec::new();
        };
        let Some(bs) = self.batch.as_ref() else {
            return vec![cut];
        };
        let step = match bs.prefix.as_ref().and_then(|pf| pf.tier.as_ref()) {
            Some(t) => t.run_blocks() * BLOCK_TOKENS,
            None => BLOCK_TOKENS,
        };
        // the ring holds the last `ring` blocks of the prompt written so far
        let last = (t_len - 1) / BLOCK_TOKENS;
        let resident = |c: usize| c / BLOCK_TOKENS + bs.ring >= last + 1 + self.win_blocks();
        match crate::gpu_model::prefix_cache::backoff_cut(cut, step) {
            Some(b) if b > start && b >= MIN_SNAPSHOT_LEN && resident(b) => vec![b, cut],
            _ => vec![cut],
        }
    }

    /// The trailing checkpoint cut alone (see [`Self::prefix_cuts`]).
    fn prefix_cut(&self, t_len: usize, start: usize) -> Option<usize> {
        let has = self.batch.as_ref().is_some_and(|b| b.prefix.is_some());
        if !has || t_len < MIN_SNAPSHOT_LEN {
            return None;
        }
        // tiered: align the cut to the tier's run size so a demoted
        // boundary's blocks are exactly restorable (runs are the tier's
        // restore granularity; an unaligned boundary could never resume
        // through T1)
        let step = match self.batch.as_ref().and_then(|b| b.prefix.as_ref()) {
            Some(pf) => match pf.tier.as_ref() {
                Some(t) => t.run_blocks() * BLOCK_TOKENS,
                None => BLOCK_TOKENS,
            },
            None => BLOCK_TOKENS,
        };
        let cut = (t_len - 1) / step * step;
        (cut > start).then_some(cut)
    }
}

impl LagunaPrefix {
    /// The checkpoint-pool geometry for blob demotes: (device base, stride
    /// bytes). Batch-side eviction arms need it and the fields are module-
    /// private by design.
    pub(crate) fn tier_state_geom(&self, stream: &cudarc::driver::CudaStream) -> (u64, u64) {
        use cudarc::driver::DevicePtr;
        let (cp, _g) = self.d_ckpt.device_ptr(stream);
        (cp, self.state_bytes as u64)
    }
}
