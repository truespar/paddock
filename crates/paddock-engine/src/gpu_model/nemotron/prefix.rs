//! Nemotron radix prefix cache - stage D.
//!
//! The attention half is granite's shape: a [`PagedRadix`] over the 6-layer
//! block pool, a hit ADOPTS blocks by refcount, nothing copies. The mamba
//! half is the qwen35 hybrid precedent: recurrent state is not per-block
//! sharable - a sequence can only resume at a position whose 23-layer
//! (SSM state + conv window) snapshot was CHECKPOINTED, so the resume point
//! is the deepest checkpoint under the block match (`PagedMatch::ckpt`),
//! never the raw match length. Checkpoints land at `ckpt_cuts` - the last
//! two page boundaries of a prompt (qwen35's two-boundary law: a re-rendered
//! next turn diverges inside the trailing generation header, which ~5/16 of
//! the time crosses the last boundary; a checkpoint only there is
//! unreachable and reuse deterministically drops to 0%).
//!
//! Snapshots are STAGED during the pass, never by splitting it: the mamba
//! run walk in `layer_walk` pauses its conv/scan advance at a break row,
//! copies that layer's slot state into a staging blob, and continues - the
//! GEMM passes and the tick structure stay whole (splitting a chunk at a
//! cut would re-stream all 20 GiB of weights per split; qwen35's
//! d_ckpt_stage exists for exactly this reason). After the pass, the blob
//! copies into the checkpoint's pool pages under the radix node: since
//! issue #33 a checkpoint lives in the attention pool's own pages
//! (`PagedRadix::set_state_paged`), so it is cache that a growing context
//! takes back, not a fixed reserve. Live-state snapshots and restores pass
//! through one flat bounce blob, because the SSM arena may be f16 and
//! widens/narrows only into a contiguous f32 span.

use crate::gpu::GpuError;
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::kv_pool::BLOCK_TOKENS;

use super::{GpuNemotron, Mixer};
use paddock_models::nemotron::NemotronBlock;

/// Don't resume prefixes shorter than this (granite's floor - the restore
/// here also pays 23 state copies, so trivial prompts aren't worth churn).
pub(super) const MIN_CACHE_PREFIX: usize = 32;

/// Staging blobs per pass: two per slot (every prompt has two trailing cuts
/// and a coalesced tick can carry every slot's prompt), clamped 4..=32. A
/// pass stages one blob per checkpoint cut that lands inside it; cuts beyond
/// the count are skipped (reuse loss only, never an error). It was a flat 4
/// ("a full admission wave's trailing cuts"), which is two prompts: an
/// 8-session agentic wave (its turns arrive together and share one tick)
/// left six of the eight sessions without a checkpoint every turn, and
/// each of them re-prefilled from the shared system prompt's cut on the
/// next turn (GB10 2026-09-11, cached_tokens 1872 at every c8 turn).
/// `PADDOCK_NEMO_CKPT_STAGES` pins a count (dev; 4 = the old flat value).
pub(super) fn ckpt_stages(slots: usize) -> usize {
    paddock_models::dev_var!("PADDOCK_NEMO_CKPT_STAGES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or((2 * slots).clamp(4, 32))
}

/// Blocks kept in reserve for radix retention when sizing the pool. Cheap
/// here - a nemotron block-set is 96 KiB (6 attention layers), so the 512
/// default is ~48 MB, not granite-30b's 2 GiB.
pub(crate) fn retention_blocks() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_NEMO_PREFIX_BLOCKS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(512)
    })
}

/// Engine-wide off switch, honoured by every family.
pub(crate) fn prefix_disabled() -> bool {
    paddock_models::dev_var_os!("PADDOCK_NO_PREFIX_CACHE").is_some()
}

pub(crate) use crate::gpu_model::prefix_cache::reply_ckpt_disabled;

/// The checkpoint boundaries for a prompt: its last two full page
/// boundaries, ascending (0 entries collapse when the prompt is short).
/// Keeps at least one token to prefill, matching the radix matcher.
///
/// No back-off cut here (`prefix_cache::backoff_cut`, which qwen35 and
/// qwen4exp take): the scan PAUSES at every staged cut and resumes from the
/// slot arena, which is f16 by default, so each cut is one more f16 round
/// trip of the recurrent state. At the trailing pair that touches the last
/// page; 256 tokens back it moved the prompt-end logits enough that the
/// KV8 smoke's fp8-vs-f16 greedy-8 fell 8/8 -> 4/8 (GB10 2026-10-05). The
/// f32-state families stage without rounding. Nemotron's prefill is also
/// the cheapest to redo (~8K tok/s on GB10).
pub(super) fn ckpt_cuts(t_len: usize, step: usize) -> [usize; 2] {
    // `step` = the tier's run span when armed (both boundaries must sit at
    // run granularity or their blobs cannot demote - qwen35's precedent),
    // BLOCK_TOKENS otherwise (the historical behavior, unchanged).
    let step = step.max(BLOCK_TOKENS);
    let b1 = if t_len > 1 {
        (t_len - 1) / BLOCK_TOKENS * BLOCK_TOKENS
    } else {
        0
    };
    let b1 = b1 / step * step;
    [b1.saturating_sub(step), b1]
}

impl GpuNemotron {
    /// Match `keys` against the radix; on a hit with a reachable checkpoint,
    /// adopt the KV blocks and restore the mamba state snapshot. Then back
    /// every row the prompt still writes (the whole prompt on a miss) and
    /// return the resume position, 0 = cold (admission already zeroed the
    /// arenas). Called right after `admit_rows`, which backs nothing: the
    /// adopted blocks are the slot's before the tail allocation can make the
    /// pool shed retention, so the tail never evicts the prefix it extends.
    pub(super) fn prefix_resume_rows(
        &mut self,
        slot: usize,
        keys: &[u32],
        n_rows: usize,
    ) -> Result<usize, GpuModelError> {
        self.last_reused[slot] = 0;
        let pos = self.prefix_adopt(slot, keys)?;
        self.ensure_rows(&[slot as u32], &[(n_rows - 1) as u32])?;
        // the drafter's rows came with the adopted pages wherever a live
        // span completed them (see dflash_adopt_slot)
        self.dflash_adopt_slot(slot, pos);
        // the in-file MTP's rows ride the same pages (see mtp_adopt_slot)
        self.mtp_adopt_slot(slot, pos);
        if pos > 0 {
            self.last_reused[slot] = pos;
        }
        Ok(pos)
    }

    /// The adopting half of `prefix_resume_rows`: the deepest checkpointed
    /// match of `keys`, its blocks shared into the slot's table and its mamba
    /// state restored. Returns the resume position, 0 on a miss.
    fn prefix_adopt(&mut self, slot: usize, keys: &[u32]) -> Result<usize, GpuModelError> {
        let m = {
            let bs = self.batch.as_mut().expect("batch enabled");
            let Some(radix) = bs.prefix.as_mut() else {
                return Ok(0);
            };
            let mut m = radix.match_full(keys);
            // TIER (D5 park/wake): the restore is consulted and PARKED at
            // admission (`tier_prefix_loading`); an elected restore has
            // already published + attached by the time prefill runs. Pump
            // for freshness and re-match, so paths that skip the consult
            // still pick up published prefixes.
            if m.ckpt.is_none()
                && let Some(tier) = bs.tier.as_mut()
            {
                tier.pump_completions(radix, &mut bs.pool);
                m = radix.match_full(keys);
            }
            m
        };
        // hybrid law: resume only where state was snapshotted - the deepest
        // checkpoint under the match, never the raw block-match length
        let Some((pos, idx)) = m.ckpt else {
            return Ok(0);
        };
        if pos < MIN_CACHE_PREFIX || pos >= keys.len() {
            return Ok(0);
        }
        {
            let bs = self.batch.as_mut().expect("batch enabled");
            // adopt the shared blocks (admission left the table empty)
            bs.tables[slot].clear(&mut bs.pool);
            bs.tables[slot].share_prefix(&m.blocks[..pos / BLOCK_TOKENS], &mut bs.pool);
            let base = slot * bs.bps;
            for (j, &b) in bs.tables[slot].blocks().iter().enumerate() {
                bs.bt_host[base + j] = b;
            }
            self.exec
                .stream
                .memcpy_htod(&bs.bt_host, &mut bs.d_bt)
                .map_err(|e| GpuError::Driver(e.to_string()))?;
        }
        self.restore_state(slot, idx)?;
        Ok(pos)
    }

    /// Publish a finished prompt's full pages into the radix (idempotent for
    /// pages the checkpoint commits already inserted), then evict down to a
    /// free margin so the next admission does not pay for it.
    pub(super) fn prefix_insert(&mut self, slot: usize, keys: &[u32]) {
        let bs = self.batch.as_mut().expect("batch enabled");
        let Some(radix) = bs.prefix.as_mut() else {
            return;
        };
        let blocks = bs.tables[slot].blocks().to_vec();
        radix.insert(keys, &blocks, &mut bs.pool);
        self.prefix_press_margin();
    }

    /// Evict (or tier-demote) radix retention down to the admission margin.
    fn prefix_press_margin(&mut self) {
        let bs = self.batch.as_mut().expect("batch enabled");
        if bs.prefix.is_none() {
            return;
        }
        let margin =
            crate::gpu_model::prefix_cache::evict_ahead_margin(256, bs.pool.capacity() as usize);
        if margin > 0 && bs.pool.free_blocks() < margin {
            match (bs.tier.as_mut(), bs.prefix.as_mut()) {
                (Some(tier), Some(radix)) => {
                    // tier-aware evict-ahead: closing runs AND their mamba
                    // checkpoint blobs demote before eviction - a plain
                    // evict_lru here discarded the blobs and left the tier
                    // restore-blind (probe hit, aux None, every repeat
                    // recomputed)
                    // The blobs are pool pages; the tier reads them off the radix.
                    let exec = self.exec.clone();
                    tier.press(radix, &mut bs.pool, margin, None, &mut || {
                        exec.record_event().ok()
                    });
                    tier.pump_completions(radix, &mut bs.pool);
                }
                (None, Some(radix)) => {
                    radix.make_room(&mut bs.pool, margin, 0);
                }
                _ => {}
            }
        }
    }

    /// Commit one staged checkpoint after its pass: insert the pages up to
    /// `cut`, attach a state index under the radix node (drawing its pool
    /// pages), and copy the staging blob into those pages. No-ops (reuse loss
    /// only) when the cache is off or the node can't take a checkpoint.
    pub(super) fn commit_stage(&mut self, stage: usize, slot: usize, keys: &[u32], cut: usize) {
        use cudarc::driver::DevicePtr;
        let exec = self.exec.clone();
        let bs = self.batch.as_mut().expect("batch enabled");
        let Some(radix) = bs.prefix.as_mut() else {
            return;
        };
        let blocks: Vec<u32> = match bs.tables[slot].blocks().get(..cut / BLOCK_TOKENS) {
            Some(b) => b.to_vec(),
            None => return,
        };
        radix.insert(&keys[..cut], &blocks, &mut bs.pool);
        let Some(idx) = radix.attach_state_with_pool(keys, cut, &mut bs.pool) else {
            return;
        };
        let (Some(layout), Some(radix), Some(db)) = (
            bs.ckpt_layout.as_ref(),
            bs.prefix.as_ref(),
            bs.d_ckpt_desc.as_mut(),
        ) else {
            return;
        };
        let (sp, _g) = bs.d_ckpt_stage[stage].device_ptr(&exec.stream);
        let mut descs = Vec::new();
        layout.push_copy(
            radix.state_pages(idx),
            0,
            sp,
            (bs.state_ckpt_f32 * 4) as u64,
            crate::ckpt_pages::Dir::ToPages,
            &mut descs,
        );
        if let Err(e) = exec.batched_copy_upload(db, &descs) {
            tracing::warn!("nemotron ckpt commit failed (stage {stage}): {e}");
        }
    }

    /// Move checkpoint `idx` between the bounce blob and its pool pages:
    /// `ToPages` files the snapshot the bounce holds, `FromPages` brings one
    /// back for a restore. The pages hold the flat blob's layout.
    pub(super) fn ckpt_pages_copy(
        &mut self,
        idx: u32,
        dir: crate::ckpt_pages::Dir,
    ) -> Result<(), GpuModelError> {
        use cudarc::driver::DevicePtr;
        let exec = self.exec.clone();
        let bs = self.batch.as_mut().expect("batch enabled");
        let (Some(layout), Some(radix), Some(bounce), Some(db)) = (
            bs.ckpt_layout.as_ref(),
            bs.prefix.as_ref(),
            bs.d_ckpt_bounce.as_ref(),
            bs.d_ckpt_desc.as_mut(),
        ) else {
            return Err(GpuModelError::Unsupported(
                "checkpoint copy without paged checkpoints".into(),
            ));
        };
        let pages = radix.state_pages(idx);
        if pages.is_empty() {
            return Err(GpuModelError::Unsupported(
                "checkpoint index holds no pages".into(),
            ));
        }
        let (bp, _g) = bounce.device_ptr(&exec.stream);
        let mut descs = Vec::new();
        layout.push_copy(
            pages,
            0,
            bp,
            (bs.state_ckpt_f32 * 4) as u64,
            dir,
            &mut descs,
        );
        exec.batched_copy_upload(db, &descs)?;
        Ok(())
    }

    /// Restore `slot`'s mamba state from checkpoint `idx` - the reverse of
    /// the staged snapshot: the checkpoint's pages -> the bounce blob -> each
    /// mamba layer's slot arena and window, in the layer order it was written.
    fn restore_state(&mut self, slot: usize, idx: u32) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let hp = self.hp.clone();
        let state_elems = hp.mamba_heads * hp.mamba_head_dim * hp.d_state;
        let win_elems = (hp.d_conv - 1) * hp.conv_dim();
        self.ckpt_pages_copy(idx, crate::ckpt_pages::Dir::FromPages)?;
        let bs = self.batch.as_mut().expect("batch enabled");
        let Some(sp) = bs.d_ckpt_bounce.as_ref() else {
            return Err(GpuModelError::Unsupported(
                "resume without a checkpoint bounce blob".into(),
            ));
        };
        let mut boff = 0usize;
        for li in 0..hp.n_layer {
            let Some(s) = bs.ssm[li].as_mut() else {
                continue;
            };
            s.restore_from_blob(&exec, sp, boff, slot * state_elems, state_elems)?;
            boff += state_elems;
            let w = bs.conv_win[li].as_mut().expect("mamba layer has window");
            exec.copy_region(sp, boff, w, slot * win_elems, win_elems)?;
            boff += win_elems;
        }
        Ok(())
    }

    /// Blocks the radix could give back - added to the free count for
    /// admission accounting (the cache is reclaimable capacity, not a
    /// reservation - the gemma4 c8 lesson).
    /// The D5 admission consult (park/wake): probe + elect the hybrid
    /// two-round restore and, when elected, START it and PARK the request -
    /// qwen35's recipe verbatim with the mamba state blob in place of the
    /// DeltaNet one. `true` = skip this slot this tick; the per-pass
    /// `tier_pump` drives the flow and the wake re-enters admission.
    pub(crate) fn tier_consult_impl(&mut self, slot: usize, tokens: &[u32]) -> bool {
        use crate::kv_tier::FlowStatus;
        let exec = self.exec.clone();
        let Some(bs) = self.batch.as_mut() else {
            return false;
        };
        if bs.ckpt_layout.is_none() {
            return false; // no checkpoints, so nothing a hybrid can resume at
        }
        let (Some(tier), Some(pr)) = (bs.tier.as_mut(), bs.prefix.as_mut()) else {
            return false;
        };
        tier.pump_completions(pr, &mut bs.pool);
        {
            let exec2 = exec.clone();
            // an aux round reserves its checkpoint's pages from the pool
            tier.pump_flows_with_pool(pr, &mut bs.pool, &mut || exec2.record_event().ok());
        }
        match tier.flow_status(slot, tokens) {
            FlowStatus::Loading => return true,
            FlowStatus::Done { .. } => return false,
            FlowStatus::None => {}
        }
        // a resident usable checkpoint makes the tier moot for this prompt
        if pr.match_full(tokens).ckpt.is_some() {
            return false;
        }
        // the blob's geometry lets the probe find it on any tier
        tier.declare_blob_pages(pr.pages_per_ckpt());
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
        // the destination: the restored blocks, then the checkpoint's own
        // pages - the blob round draws those from the pool too
        let need = aux.end_block + pr.pages_per_ckpt();
        let afford =
            |pool: &crate::kv_pool::KvPool| pool.free_blocks().saturating_sub(2 * r) / r * r;
        if afford(&bs.pool) < need {
            // retention crowds the destination: pressure-demote it (the
            // prefix cache is reclaimable capacity)
            let want = need + 2 * r;
            let after = exec.record_event().ok();
            let (_e, taken) = tier.pressure_demote(&mut *pr, &mut bs.pool, want, after);
            for t in taken {
                if t.end_block % r == 0 {
                    let ev = exec.record_event().ok();
                    tier.demote_aux_paged(pr, &mut bs.pool, t, ev);
                } else {
                    pr.recycle_state(t.state_idx);
                }
            }
            pr.reclaim(&mut bs.pool);
            let deadline = std::time::Instant::now() + std::time::Duration::from_millis(50);
            while bs.pool.free_blocks() < want && tier.stats().2 > 0 {
                tier.pump_completions(&mut *pr, &mut bs.pool);
                if std::time::Instant::now() >= deadline {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_micros(200));
            }
        }
        tracing::debug!(
            free = bs.pool.free_blocks(),
            need,
            boundary = aux.end_block,
            "nemotron tier gate"
        );
        if afford(&bs.pool) < need {
            tier.refuse_park();
            return false;
        }
        let after = exec.record_event().ok();
        // the blob lands in pool pages the flow reserves; the radix names
        // them, so there is no slot base to hand over
        let plan = crate::kv_tier::AuxPlan {
            hit: aux,
            state_base: 0,
            state_stride: 0,
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
                    "nemotron tier: restore parked (D5)"
                );
                true
            }
            None => false,
        }
    }

    /// The per-tick tier pump (see `Generator::tier_pump`).
    pub(crate) fn tier_pump_impl(&mut self) {
        let exec = self.exec.clone();
        let Some(bs) = self.batch.as_mut() else {
            return;
        };
        let (Some(tier), Some(pr)) = (bs.tier.as_mut(), bs.prefix.as_mut()) else {
            return;
        };
        tier.pump_completions(pr, &mut bs.pool);
        tier.pump_flows_with_pool(pr, &mut bs.pool, &mut || exec.record_event().ok());
        // 2.3 write-through: retained chains AND live state blobs
        // pre-store in slack so eviction (and ckpt-slot recycling) is free.
        // The blobs are pool pages; the tier reads them off the radix.
        tier.mirror_slack(pr, &mut bs.pool, exec.record_event().ok(), 2, None);
    }

    pub(crate) fn tier_stats_impl(&self) -> Option<crate::kv_tier::TierStats> {
        self.batch.as_ref()?.tier.as_ref().map(|t| t.tier_stats())
    }

    /// The checkpoint step: the tier's run span when armed (boundaries must
    /// sit at run granularity to demote), BLOCK_TOKENS otherwise.
    pub(crate) fn tier_ckpt_step(&self) -> usize {
        self.batch
            .as_ref()
            .and_then(|b| b.tier.as_ref())
            .map(|t| t.run_blocks() * BLOCK_TOKENS)
            .unwrap_or(BLOCK_TOKENS)
    }

    pub(crate) fn prefix_evictable(&self) -> usize {
        self.batch
            .as_ref()
            .and_then(|bs| bs.prefix.as_ref().map(|r| r.evictable_blocks(&bs.pool)))
            .unwrap_or(0)
    }
}

// ── stage F: the reply checkpoint ──────────────────────────────────────────
//
// The prefix cache's mamba checkpoints landed only at a PROMPT's last two
// page boundaries, so an agentic turn N+1 resumed at turn N's prompt and
// re-prefilled turn N's whole reply plus the new message (~230 rows per
// turn; 105 ms at one row on the GB10 against a ~65 ms floor for the new
// message alone, and eight of them in one tick at c8). Now every 16-token
// boundary a reply crosses snapshots the slot's live mamba state straight
// from the arena into the pool and files the reply's pages under the radix,
// one live reply checkpoint per slot (the previous one is detached and its
// index recycled), so the next turn resumes at the END of the reply.
//
// The sequence is tracked per slot: the prompt's keys at admission, then
// every decode token a tick FEEDS and every row a spec round commits (so the
// tracked length always equals the next position; any gap - a tier restore,
// a recompute - ends tracking for the slot until its next admission). The decode pipe enqueues
// the copy the moment the tick is launched (stream-ordered behind it) and
// files the pages when the ids reach the host.
impl GpuNemotron {
    /// Admission: start tracking `slot`'s sequence at the prompt's keys.
    pub(super) fn reply_track_admit(&mut self, slot: usize, tokens: &[u32]) {
        self.reply_release(slot);
        let Some(bs) = self.batch.as_mut() else {
            return;
        };
        if reply_ckpt_disabled()
            || bs.prefix.is_none()
            || bs.ckpt_layout.is_none()
            || slot >= bs.seq.len()
        {
            return;
        }
        bs.seq[slot] = tokens.to_vec();
    }

    /// A tick fed `tok` at `pos` for `slot`.
    pub(super) fn reply_feed(&mut self, slot: usize, pos: u32, tok: u32) {
        let Some(bs) = self.batch.as_mut() else {
            return;
        };
        let Some(seq) = bs.seq.get_mut(slot) else {
            return;
        };
        if seq.is_empty() {
            return;
        }
        if seq.len() != pos as usize {
            seq.clear();
            return;
        }
        seq.push(tok);
    }

    /// After a decode pass over these rows: snapshot every row whose position
    /// closes a page, then file whatever the host has the ids for.
    pub(super) fn reply_after_rows(
        &mut self,
        slots: &[u32],
        positions: &[u32],
    ) -> Result<(), GpuModelError> {
        for (i, &s) in slots.iter().enumerate() {
            let cut = positions[i] as usize + 1;
            if cut.is_multiple_of(BLOCK_TOKENS) {
                self.reply_snapshot(s as usize, cut)?;
            }
        }
        self.reply_resolve_pending();
        Ok(())
    }

    /// A spec round committed `accepts[i].2` rows of request `i`: feed them
    /// and checkpoint every page the commit closes, at the exact closing
    /// row. The verify keeps a state snapshot per row, so the cut need not
    /// fall on the round's end - at c committed rows a round, a round end
    /// lands on a given page edge about 1/c of the time. Runs before the rollback:
    /// a row's conv window is the pre-round window followed by the round's
    /// conv-input rows, and the rollback overwrites the former.
    pub(super) fn reply_after_spec(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
        accepts: &[(usize, usize, usize)],
    ) -> Result<(), GpuModelError> {
        for (&(slot, pos0, ref chunk), &(_, off, acc)) in reqs.iter().zip(accepts) {
            for (j, &tok) in chunk.iter().enumerate().take(acc) {
                self.reply_feed(slot, (pos0 + j) as u32, tok);
                let cut = pos0 + j + 1;
                if cut.is_multiple_of(BLOCK_TOKENS) {
                    self.reply_snapshot_row(slot, cut, off + j, j + 1)?;
                }
            }
        }
        self.reply_resolve_pending();
        Ok(())
    }

    /// `reply_snapshot` for a verify row: the state is the row's snapshot,
    /// the conv window the last k-1 rows of [pre-round window ∥ the round's
    /// first `n_new` conv-input rows], assembled in the bounce blob and then
    /// copied into the checkpoint's pages.
    fn reply_snapshot_row(
        &mut self,
        slot: usize,
        cut: usize,
        row: usize,
        n_new: usize,
    ) -> Result<(), GpuModelError> {
        if !self.reply_tracking(slot) {
            return Ok(());
        }
        let exec = self.exec.clone();
        let hp = self.hp.clone();
        let state_elems = hp.mamba_heads * hp.mamba_head_dim * hp.d_state;
        let conv_dim = hp.conv_dim();
        let km1 = hp.d_conv - 1;
        let win_elems = km1 * conv_dim;
        let bs = self.batch.as_mut().expect("batch enabled");
        let Some(radix) = bs.prefix.as_mut() else {
            return Ok(());
        };
        let Some(idx) = radix.reserve_state_slot_with_pool(&mut bs.pool) else {
            return Ok(());
        };
        let (Some(sp), Some(vp)) = (bs.d_ckpt_bounce.as_mut(), bs.verify.as_mut()) else {
            radix.recycle_state(idx);
            return Ok(());
        };
        let keep_old = km1.saturating_sub(n_new);
        let take_new = km1 - keep_old;
        if vp.rescan {
            // rollback by replay keeps no per-row states: replay the round's
            // first n_new rows from the (still pre-round) live state into the
            // replay target, one launch for every mamba layer
            use cudarc::driver::DevicePtr;
            let keep_row = crate::gpu::mamba2_keep_row(
                hp.mamba_heads,
                hp.mamba_head_dim,
                hp.d_state,
                hp.n_groups,
            );
            let (tp, _gt) = vp
                .d_rs_tmp
                .as_ref()
                .expect("replay target")
                .device_ptr(&exec.stream);
            let mut d = Vec::new();
            for (li, layer) in self.layers.iter().enumerate() {
                let Mixer::Mamba(w) = &layer.mixer else {
                    continue;
                };
                let (st, eb) = bs.ssm[li].as_ref().expect("ssm arena").addr(&exec, 0);
                let (kp, _g1) = vp.keep[li].as_ref().expect("keep").device_ptr(&exec.stream);
                let (ap, _g2) = w.a.device_ptr(&exec.stream);
                let (bp, _g3) = w.dt_bias.device_ptr(&exec.stream);
                let m = (d.len() / 6) as u64;
                d.extend([
                    st + (slot * state_elems * eb) as u64,
                    tp + m * (state_elems * 2) as u64,
                    kp + ((row + 1 - n_new) * keep_row * 4) as u64,
                    ap,
                    bp,
                    n_new as u64,
                ]);
            }
            exec.mamba2_rescan_upload(
                &mut vp.d_rs_desc,
                &d,
                hp.mamba_heads,
                hp.mamba_head_dim,
                hp.d_state,
                hp.n_groups,
            )?;
        }
        let mut boff = 0usize;
        let mut m = 0usize;
        for li in 0..hp.n_layer {
            if !matches!(hp.blocks[li], NemotronBlock::Mamba) {
                continue;
            }
            if vp.rescan {
                let tmp = vp.d_rs_tmp.as_ref().expect("replay target");
                exec.ssm_state_widen(tmp, m * state_elems, sp, boff, state_elems)?;
            } else {
                let snap = vp.snap[li].as_ref().expect("snap");
                snap.save_to_blob(&exec, row * state_elems, sp, boff, state_elems)?;
            }
            m += 1;
            boff += state_elems;
            let w = bs.conv_win[li].as_ref().expect("mamba layer has window");
            let xbc = vp.xbc[li].as_ref().expect("mamba layer has xbc rows");
            if keep_old > 0 {
                exec.copy_region(
                    w,
                    slot * win_elems + (km1 - keep_old) * conv_dim,
                    sp,
                    boff,
                    keep_old * conv_dim,
                )?;
            }
            exec.copy_region(
                xbc,
                (row + 1 - take_new) * conv_dim,
                sp,
                boff + keep_old * conv_dim,
                take_new * conv_dim,
            )?;
            boff += win_elems;
        }
        self.reply_file_bounce(slot, cut, idx)
    }

    /// Copy the snapshot the bounce blob now holds into reserved checkpoint
    /// `idx`'s pages and queue it for filing - or give the index back if the
    /// copy failed.
    fn reply_file_bounce(
        &mut self,
        slot: usize,
        cut: usize,
        idx: u32,
    ) -> Result<(), GpuModelError> {
        if let Err(e) = self.ckpt_pages_copy(idx, crate::ckpt_pages::Dir::ToPages) {
            if let Some(radix) = self.batch.as_mut().and_then(|b| b.prefix.as_mut()) {
                radix.recycle_state(idx);
            }
            return Err(e);
        }
        let bs = self.batch.as_mut().expect("batch enabled");
        bs.reply_pending.push((slot, cut, idx));
        Ok(())
    }

    /// The pipe's ids for tick `k - 1` arrived: they are the tokens fed at
    /// `pos0 + k`. Feed them and file the snapshots they complete.
    pub(super) fn reply_pipe_ids(
        &mut self,
        ids: &[u32],
        pos0: &[u32],
        slots: Option<&[u32]>,
        k: usize,
    ) {
        for (i, &t) in ids.iter().enumerate() {
            let slot = slots.map_or(i, |s| s[i] as usize);
            self.reply_feed(slot, pos0[i] + k as u32, t);
        }
        self.reply_resolve_pending();
    }

    fn reply_tracking(&self, slot: usize) -> bool {
        self.batch
            .as_ref()
            .is_some_and(|bs| bs.seq.get(slot).is_some_and(|s| !s.is_empty()))
    }

    /// Copy `slot`'s live state into a reserved checkpoint (the reverse of
    /// `restore_state`, through the bounce blob) and queue it for filing.
    fn reply_snapshot(&mut self, slot: usize, cut: usize) -> Result<(), GpuModelError> {
        if !self.reply_tracking(slot) {
            return Ok(());
        }
        let exec = self.exec.clone();
        let hp = self.hp.clone();
        let state_elems = hp.mamba_heads * hp.mamba_head_dim * hp.d_state;
        let win_elems = (hp.d_conv - 1) * hp.conv_dim();
        let bs = self.batch.as_mut().expect("batch enabled");
        let Some(radix) = bs.prefix.as_mut() else {
            return Ok(());
        };
        let Some(idx) = radix.reserve_state_slot_with_pool(&mut bs.pool) else {
            return Ok(());
        };
        let Some(sp) = bs.d_ckpt_bounce.as_mut() else {
            radix.recycle_state(idx);
            return Ok(());
        };
        let mut boff = 0usize;
        for li in 0..hp.n_layer {
            let Some(s) = bs.ssm[li].as_ref() else {
                continue;
            };
            s.save_to_blob(&exec, slot * state_elems, sp, boff, state_elems)?;
            boff += state_elems;
            let w = bs.conv_win[li].as_ref().expect("mamba layer has window");
            exec.copy_region(w, slot * win_elems, sp, boff, win_elems)?;
            boff += win_elems;
        }
        self.reply_file_bounce(slot, cut, idx)
    }

    /// File every pending snapshot whose ids have arrived: the reply's pages
    /// up to the cut go under the radix, the state attaches at the cut, the
    /// slot's previous reply checkpoint is detached and its index recycled.
    pub(super) fn reply_resolve_pending(&mut self) {
        let mut filed_any = false;
        {
            let Some(bs) = self.batch.as_mut() else {
                return;
            };
            let Some(radix) = bs.prefix.as_mut() else {
                return;
            };
            let mut i = 0;
            while i < bs.reply_pending.len() {
                let (slot, cut, idx) = bs.reply_pending[i];
                let seq = &bs.seq[slot];
                if seq.is_empty() {
                    // tracking ended before the ids came - orphaned blob
                    radix.recycle_state(idx);
                    bs.reply_pending.swap_remove(i);
                    continue;
                }
                if seq.len() < cut {
                    i += 1;
                    continue;
                }
                let filed = match bs.tables[slot].blocks().get(..cut / BLOCK_TOKENS) {
                    Some(blocks) => {
                        let blocks = blocks.to_vec();
                        radix.insert(&seq[..cut], &blocks, &mut bs.pool);
                        radix.attach_state_at(seq, cut, idx)
                    }
                    None => false,
                };
                if filed {
                    filed_any = true;
                    if let Some((old_cut, old_idx)) = bs.reply_ckpt[slot].replace((cut, idx))
                        && radix.detach_state_if(seq, old_cut, old_idx)
                    {
                        radix.recycle_state(old_idx);
                    }
                } else {
                    radix.recycle_state(idx);
                }
                bs.reply_pending.swap_remove(i);
            }
        }
        if filed_any {
            self.prefix_press_margin();
        }
    }

    /// The reply just started its first tool call (see
    /// `Generator::reply_pin`): the live checkpoint becomes the held one and
    /// the next filed snapshot opens a new live one instead of replacing it.
    /// Once per reply. A snapshot still waiting for its ids precedes the call
    /// too; it files as the new live one.
    /// `Generator::anchor_at`: hold the user turn's prompt-end checkpoint.
    pub(crate) fn anchor_at(&mut self, tokens: &[u32], upto: usize) {
        if let Some(radix) = self.batch.as_mut().and_then(|bs| bs.prefix.as_mut())
            && let Some(at) = radix.mark_anchor(tokens, upto)
        {
            tracing::debug!(
                "nemotron anchor: user-turn checkpoint at {at} (prompt {})",
                upto + 1
            );
        }
    }

    pub(crate) fn reply_pin(&mut self, slot: usize) {
        let Some(bs) = self.batch.as_mut() else {
            return;
        };
        if slot >= bs.reply_pinned.len() || bs.reply_pinned[slot].is_some() {
            return;
        }
        bs.reply_pinned[slot] = bs.reply_ckpt[slot].take();
    }

    /// The slot went idle (or is being re-admitted): stop tracking. Its last
    /// reply checkpoint - and the one held at its first tool call - STAY in
    /// the radix for the next turn; the pool's LRU owns them now; snapshots
    /// whose ids never arrived are given back.
    pub(super) fn reply_release(&mut self, slot: usize) {
        let Some(bs) = self.batch.as_mut() else {
            return;
        };
        if slot >= bs.seq.len() {
            return;
        }
        bs.seq[slot].clear();
        bs.reply_ckpt[slot] = None;
        bs.reply_pinned[slot] = None;
        if let Some(radix) = bs.prefix.as_mut() {
            let mut i = 0;
            while i < bs.reply_pending.len() {
                if bs.reply_pending[i].0 == slot {
                    radix.recycle_state(bs.reply_pending[i].2);
                    bs.reply_pending.swap_remove(i);
                } else {
                    i += 1;
                }
            }
        }
    }
}
