//! Qwen3.5/3.6 per-slot prefill + radix prefix cache resume/insert/snapshot.

use super::*;
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::prefix_cache::BLOCK_TOKENS;
use cudarc::driver::DevicePtr;

impl GpuQwen35 {
    /// The D5 admission consult (park/wake): probe + elect the hybrid
    /// two-round restore and, when elected, START it and PARK the request -
    /// `true` tells the scheduler to skip this slot this tick and run other
    /// work; the per-pass `tier_pump` drives the flow (blocks publish, then
    /// the DeltaNet blob lands in a RESERVED checkpoint slot that attaches
    /// only once verified) and the request re-enters admission when it
    /// resolves. Failure of any kind degrades to recompute; the family's
    /// normal resume path adopts whatever published via an ordinary match.
    pub(crate) fn tier_consult_impl(&mut self, slot: usize, tokens: &[u32]) -> bool {
        use crate::kv_tier::{Election, FlowStatus};
        use cudarc::driver::DevicePtr;
        let exec = self.exec.clone();
        let Some(bs) = self.batch.as_mut() else {
            return false;
        };
        let state_bytes = (bs.state_ckpt_f32 * 4) as u64;
        let (Some(tier), Some(pr), Some(pool)) =
            (bs.tier.as_mut(), bs.paged_prefix.as_mut(), bs.pool.as_mut())
        else {
            return false;
        };
        let Some(sp) = bs.d_state_pool.as_ref() else {
            return false;
        };
        tier.pump_completions(pr, pool);
        {
            let exec2 = exec.clone();
            tier.pump_flows(pr, &mut || exec2.record_event().ok());
        }
        match tier.flow_status(slot, tokens) {
            FlowStatus::Loading => return true,
            // resolved: the resume path re-matches; never re-park
            FlowStatus::Done { .. } => return false,
            FlowStatus::None => {}
        }
        // a resident usable checkpoint makes the tier moot for this prompt
        if pr.match_full(tokens).ckpt.is_some() {
            return false;
        }
        let hit = tier.probe(tokens, 0);
        let afford = |pool: &crate::kv_pool::KvPool, r: usize| {
            pool.free_blocks().saturating_sub(2 * r) / r * r
        };
        let r = tier.run_blocks();
        let mut afford_blocks = afford(pool, r);
        let deepest = hit
            .as_ref()
            .and_then(|h| tier.probe_aux(tokens, h.end_block))
            .filter(|a| a.end_block * crate::kv_pool::BLOCK_TOKENS >= 32 && a.end_block % r == 0);
        if let Some(a) = &deepest
            && afford_blocks < a.end_block
        {
            // retention crowds the destination: pressure-demote it (the
            // prefix cache is reclaimable capacity)
            let want = a.end_block + 2 * r;
            let after = exec.record_event().ok();
            let (_e, taken) = tier.pressure_demote(pr, pool, want, after);
            let (cp, _g) = sp.device_ptr(&exec.stream);
            for t in taken {
                if t.end_block % r == 0 {
                    let blob = cp + t.state_idx as u64 * state_bytes;
                    let ev = exec.record_event().ok();
                    tier.demote_aux(pr, t, blob, state_bytes, ev);
                } else {
                    pr.recycle_state(t.state_idx);
                }
            }
            let deadline = std::time::Instant::now() + std::time::Duration::from_millis(50);
            while pool.free_blocks() < want && tier.stats().2 > 0 {
                tier.pump_completions(pr, pool);
                if std::time::Instant::now() >= deadline {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_micros(200));
            }
            afford_blocks = afford(pool, r);
        }
        let aux = deepest.filter(|a| a.end_block <= afford_blocks);
        tracing::debug!(
            free = pool.free_blocks(),
            afford_blocks,
            hit_end = hit.as_ref().map(|h| h.end_block),
            aux_end = aux.as_ref().map(|a| a.end_block),
            "qwen35 tier gate"
        );
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
            restore_tokens: (aux.end_block * crate::kv_pool::BLOCK_TOKENS) as u32,
            queued_bytes: tier.catalog.ledger(crate::kv_tier::Tier::Ram).in_flight,
            nvme_bytes: hit.nvme_bytes,
        };
        let Election::Restore { est_us, .. } = tier.cost.elect(shape) else {
            return false;
        };
        let after = exec.record_event().ok();
        let (cp, _g) = sp.device_ptr(&exec.stream);
        let plan = crate::kv_tier::AuxPlan {
            hit: aux,
            state_base: cp,
            state_stride: state_bytes,
        };
        match crate::kv_tier::RestoreFlow::begin(
            tier,
            pool,
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
                    "qwen35 tier: restore parked (D5)"
                );
                true
            }
            None => false,
        }
    }

    /// Prefill `tokens` into batch slot `slot`. Fresh sequences zero the
    /// slot's recurrent state + conv window; a prompt whose prefix hits a
    /// prefix-cache checkpoint RESUMES there instead (full-attn KV pages copy
    /// in, DeltaNet states + windows restore, only the remainder is
    /// prefilled). The prefill snapshots checkpoints at the last two page
    /// boundaries - the multi-turn resume points, since a re-rendered history
    /// diverges only inside the trailing page (generation header) - and
    /// inserts this prompt's pages + checkpoints into the cache after.
    pub fn forward_prefill_slot(
        &mut self,
        slot: usize,
        tokens: &[u32],
    ) -> Result<Vec<f32>, GpuModelError> {
        let t_len = tokens.len();
        // Empty prompts are rejected upstream; an over-length prompt returns a
        // clean error rather than panicking the engine thread (the scheduler
        // also guards this before admission - belt and suspenders).
        debug_assert!(t_len > 0, "empty prompt should be rejected before prefill");
        if t_len > self.max_ctx {
            return Err(GpuModelError::ContextExceeded {
                got: t_len,
                max: self.max_ctx,
            });
        }
        assert!(self.batch.is_some(), "enable_batch first");
        assert!(slot < self.batch.as_ref().expect("batch").max_batch);
        // match+restore the cached prefix (DeltaNet state + KV pages), then prefill
        // only the divergent tail [start, t_len). One match/restore implementation,
        // shared with the chunked prefill_begin / advance_chunks paths.
        let start = self.prefix_resume_begin(slot, tokens)?;
        self.prefill_slot_tail(slot, tokens, start)
    }

    /// Match `tokens` against the prefix cache and, if a resumable DeltaNet
    /// checkpoint exists under the shared prefix, adopt the cached KV pages
    /// (zero-copy refcount in pool mode / dense copy otherwise) and restore the
    /// DeltaNet recurrent state + causal-conv window into `slot`. Returns the
    /// resume position (block-aligned; 0 = fresh, state zeroed). A hybrid model
    /// can only resume where state was snapshotted, so the resume point is the
    /// deepest checkpoint under the match, not the raw prefix length. Shared by
    /// `forward_prefill_slot` and the chunked `prefill_begin`.
    pub(super) fn prefix_resume_begin(
        &mut self,
        slot: usize,
        tokens: &[u32],
    ) -> Result<usize, GpuModelError> {
        let t_len = tokens.len();
        // text sequence: llama-position == kv position
        self.batch.as_mut().expect("batch").mrope_delta[slot] = 0;
        // P5c: pool mode with the zero-copy radix cache ADOPTS cached KV pages by
        // refcount (no copy) + restores DeltaNet state from the paged state pool;
        // dense mode copies pages out of the dense RadixKvCache.
        let paged = self.batch.as_ref().expect("batch").paged_prefix.is_some();
        let mut state_idx: Option<u32> = None;
        let start = if paged {
            let mut m = {
                let bs = self.batch.as_mut().expect("batch");
                bs.paged_prefix
                    .as_mut()
                    .expect("prefix checked above")
                    .match_full(tokens)
            };
            // TIER (D5 park/wake): the restore is consulted and PARKED at
            // admission (`tier_prefix_loading`) - by the time prefill runs,
            // any elected restore has already published + attached. Pump for
            // freshness and re-match, so admission paths that skip the
            // consult (classic/mm) still pick up published prefixes.
            if m.ckpt.is_none() {
                let bs = self.batch.as_mut().expect("batch");
                if let (Some(tier), Some(pr), Some(pool)) =
                    (bs.tier.as_mut(), bs.paged_prefix.as_mut(), bs.pool.as_mut())
                {
                    tier.pump_completions(pr, pool);
                    m = pr.match_full(tokens);
                }
            }
            let mut start = 0usize;
            if paddock_models::dev_var_os!("PADDOCK_PREFIX_STATS").is_some() {
                tracing::info!(
                    "qwen35-resume: t_len {t_len} matched_blocks {} ({} tok) ckpt {:?} \
                     min {}",
                    m.blocks.len(),
                    m.blocks.len() * BLOCK_TOKENS,
                    m.ckpt.map(|(p, _)| p),
                    super::min_cache_prefix()
                );
            }
            if let Some((pos, idx)) = m.ckpt {
                // A resume is worth its per-slot cost either when the prefill
                // it skips is big (>= min_cache_prefix) or when few slots are
                // live, so the cost is not paid 32 times in one tick. See
                // resume_live_max() for the measurement that sets the gate.
                // The serve's CONFIGURED slot count, not the instantaneous
                // live count: during a cohort ramp the live count is briefly
                // small, so an instantaneous gate lets the first arrivals
                // resume and the cell goes bimodal anyway (measured: c32
                // 2568/2527/2392 with a live gate at 12). The slot count is
                // fixed for the serve and is exactly "how many could resume
                // in one tick".
                let slots_cfg = self.batch.as_ref().expect("batch").tables.len();
                let worth_it =
                    pos >= super::min_cache_prefix() || slots_cfg <= super::resume_live_max();
                if worth_it && pos >= 32 && pos < t_len {
                    {
                        let bs = self.batch.as_mut().expect("batch");
                        let pool = bs.pool.as_mut().expect("prefix requires the pool");
                        // release the slot's previous sequence, then point its
                        // table at the shared physical pages (retain in the pool).
                        bs.tables[slot].clear(pool);
                        bs.tables[slot].share_prefix(&m.blocks[..pos / BLOCK_TOKENS], pool);
                    }
                    self.restore_paged_state(slot, idx)?;
                    start = pos;
                    state_idx = Some(idx);
                }
            }
            start
        } else {
            // dense KV mode (explicit PADDOCK_DENSE_KV A/B or max_batch<=1)
            // has no prefix cache - the paged pool's zero-copy radix is the
            // only cache, so dense prefills cold.
            0
        };
        if start == 0 {
            // fresh sequence: zero DeltaNet state (prefill_slot_chunk(start=0)
            // clears + reallocates the KV table from the pool).
            self.zero_slot_state(slot)?;
        }
        // Drafter re-sync on resume. The drafter's KV rides the pool as a
        // stripe addressed by the same block tables (mtp_block_pass_b), so the
        // ADOPTED pages already carry its rows whenever the writer's warm
        // chain covered them - recorded per state checkpoint in `mtp_cover`.
        // Covered => the slot keeps drafting at full fidelity, including on
        // CROSS-SLOT radix hits, the common case in agentic serving and in
        // any repeated-prompt benchmark. Without this they go cold for the
        // whole session and spec throughput more than halves from the second
        // repeat on. Uncovered => dense, the old behavior for a span
        // the writer never warmed (long prompt past warm_max, spec off).
        // The seam row `start` pairs with a stale pending_h (old h, not
        // h[start-1], which the radix skip never computed) - one noisy row in
        // the DRAFTER's attention history only; drafts are class-free, the
        // verify numerics decide the emitted stream.
        let covered =
            state_idx.is_some_and(|i| self.batch.as_ref().expect("batch").mtp_cover.contains(&i));
        if let Some(sb) = self.spec_batch.as_mut()
            && start > 0
            && slot < sb.alloc_batch
        {
            if covered {
                sb.pos[slot] = start;
                sb.mtp_toks[slot].clear();
                sb.mtp_toks[slot].extend_from_slice(&tokens[..start]);
                sb.mtp_warm[slot] = true;
            } else {
                sb.mtp_warm[slot] = false;
            }
        }
        // DFlash re-sync on resume, paged stripe mode: the feature rows ride
        // the pool like the MTP stripe, so an adopted span whose checkpoint
        // is in `dflash_cover` restores coverage with the pages - set the
        // host span to match and the block drafter resumes warm. Without it
        // every prefix-restored slot is structurally cold below
        // pos ~window+start and repeated prompts silently serve the MTP
        // fallback at a large throughput cost; relaxing the warm check
        // instead was measured and REJECTED (~70% per-draft acceptance).
        // Uncovered - or
        // dense-ring mode - trims to the longest provable prefix as before.
        let dfl_covered = state_idx.is_some_and(|i| {
            self.batch
                .as_ref()
                .expect("batch")
                .dflash_cover
                .contains(&i)
        });
        if dfl_covered && start > 0 {
            self.dflash_restore_slot(slot, &tokens[..start]);
        } else {
            self.dflash_trim_slot(slot, tokens);
        }
        self.last_reused[slot] = start;
        if start > 0 && paddock_models::dev_var_os!("PADDOCK_POOL_STATS").is_some() {
            tracing::warn!("prefix: slot {slot} resumed at {start}/{t_len} (KV + DeltaNet state)");
        }
        Ok(start)
    }

    /// Prefill `tokens[start..]` into `slot` (state already seeded by
    /// `prefix_resume_begin`), snapshotting the DeltaNet state at trailing page
    /// boundaries and inserting this prompt's full pages into the prefix cache so
    /// the next turn resumes here. Bit-identical to a cold prefill of the same
    /// tokens (the parity gate). Shared by `forward_prefill_slot` and `advance_chunks`.
    pub(super) fn prefill_slot_tail(
        &mut self,
        slot: usize,
        tokens: &[u32],
        start: usize,
    ) -> Result<Vec<f32>, GpuModelError> {
        if self.batch.as_ref().expect("batch").paged_prefix.is_some() {
            self.prefill_slot_tail_paged(slot, tokens, start)
        } else {
            self.prefill_slot_tail_dense(slot, tokens, start)
        }
    }

    fn prefill_slot_tail_dense(
        &mut self,
        slot: usize,
        tokens: &[u32],
        start: usize,
    ) -> Result<Vec<f32>, GpuModelError> {
        // dense mode has no prefix cache, so there are no checkpoint cuts to
        // stage - one plain chunk over the whole tail
        self.prefill_slot_chunk(slot, &tokens[start..], start)
    }

    /// P5c paged tail: the zero-copy-radix variant of `prefill_slot_tail`. State
    /// is already seeded by `prefix_resume_begin`; this prefills `[start, t_len)`,
    /// snapshots the DeltaNet state at the last two full page boundaries (see
    /// `ckpt_cuts` - the re-rendered next turn diverges inside the trailing
    /// generation header, which can straddle the last boundary), and inserts this
    /// prompt's full pages so the next turn (this history + more) resumes here.
    /// Bit-identical to a cold prefill (the gate) - chunk splits don't change math.
    fn prefill_slot_tail_paged(
        &mut self,
        slot: usize,
        tokens: &[u32],
        start: usize,
    ) -> Result<Vec<f32>, GpuModelError> {
        let t_len = tokens.len();
        let mut pos = start;
        if paddock_models::dev_var_os!("PADDOCK_PREFIX_STATS").is_some() {
            tracing::info!(
                "qwen35-slot-tail(SPLITS at cuts): t_len {t_len} start {start} cuts {:?}",
                ckpt_cuts(t_len, self.tier_ckpt_step())
            );
        }
        for c in ckpt_cuts(t_len, self.tier_ckpt_step()) {
            if c <= pos || c >= t_len {
                continue;
            }
            // prefill up to the boundary, snapshot the state there before the
            // following rows advance it.
            self.prefill_slot_chunk(slot, &tokens[pos..c], pos)?;
            let blocks: Vec<u32> = self.batch.as_ref().expect("batch").tables[slot].blocks()
                [..c / BLOCK_TOKENS]
                .to_vec();
            {
                let bs = self.batch.as_mut().expect("batch");
                let pool = bs.pool.as_mut().expect("prefix requires the pool");
                bs.paged_prefix
                    .as_mut()
                    .expect("paged tail: prefix cache on")
                    .insert(&tokens[..c], &blocks, pool);
            }
            let idx = {
                let bs = self.batch.as_mut().expect("batch");
                bs.paged_prefix
                    .as_mut()
                    .expect("paged tail: prefix cache on")
                    .attach_state(tokens, c)
            };
            if let Some(idx) = idx {
                self.snapshot_paged_state(slot, idx)?;
                // the chunk warm ran inside prefill_slot_chunk, so the
                // drafter's coverage through `c` is decided by now
                self.record_mtp_cover(idx, slot, c, tokens);
                self.record_dflash_cover(idx, slot, c, tokens);
            }
            pos = c;
        }
        let logits = self.prefill_slot_chunk(slot, &tokens[pos..], pos)?;

        // cache every full page of this prompt (idempotent for those inserted at
        // the checkpoint above) so a longer continuation resumes past b1.
        let full = t_len / BLOCK_TOKENS;
        if full > 0 {
            let blocks: Vec<u32> =
                self.batch.as_ref().expect("batch").tables[slot].blocks()[..full].to_vec();
            let bs = self.batch.as_mut().expect("batch");
            let pool = bs.pool.as_mut().expect("prefix requires the pool");
            bs.paged_prefix
                .as_mut()
                .expect("paged tail: prefix cache on")
                .insert(&tokens[..full * BLOCK_TOKENS], &blocks, pool);
        }
        Ok(logits)
    }

    /// Zero `slot`'s DeltaNet recurrent states + conv windows (fresh sequence).
    ///
    /// One table-driven batched_copy launch, not a per-layer copy_region loop:
    /// this runs on every admission, and the loop form was ~96-128 individual
    /// cuMemcpyDtoDAsync issues whose host issue time left the GPU idle
    /// between ticks - profiled at ~11 ms of pre-embed gap per admitted
    /// request, with the host inside it doing exactly these copies (x192 per
    /// slot, x1954 on a multi-slot admission wave). Same desc format as
    /// snapshot_paged_state: (src, dst, len_bytes) triples.
    pub(super) fn zero_slot_state(&mut self, slot: usize) -> Result<(), GpuModelError> {
        self.dflash_clear_slot(slot);
        let exec = self.exec.clone();
        let state_elems = self.n_v_heads * self.state_size * self.state_size;
        let win_elems = (self.conv_k - 1) * self.conv_dim;
        let esz = crate::gpu::GpuExecutor::dn_state_esz();
        let bs = self.batch.as_mut().expect("batch");
        let (zsp, _gz) = bs.d_zero_state.device_ptr(&exec.stream);
        let (zwp, _gw) = bs.d_zero_win.device_ptr(&exec.stream);
        let mut descs: Vec<u64> = Vec::new();
        for li in 0..self.n_layers {
            if let Some(r) = bs.recur[li].as_ref() {
                let (rp, _g1) = r.device_ptr(&exec.stream);
                descs.extend([
                    zsp,
                    rp + slot as u64 * state_elems as u64 * esz,
                    state_elems as u64 * esz,
                ]);
            }
            if let Some(wn) = bs.conv_win[li].as_ref() {
                let (wp, _g2) = wn.device_ptr(&exec.stream);
                descs.extend([
                    zwp,
                    wp + (slot * win_elems * 4) as u64,
                    (win_elems * 4) as u64,
                ]);
            }
        }
        if descs.is_empty() {
            return Ok(());
        }
        let d = exec
            .stream
            .clone_htod(&descs)
            .map_err(|e| crate::gpu::GpuError::Driver(e.to_string()))?;
        exec.batched_copy(&d, descs.len() / 3)?;
        Ok(())
    }

    /// Multimodal prefill into batch slot `slot`: text + each image's
    /// embeddings (any number interleaved), each image block given equal-t
    /// mutual visibility via the attention bound, and the slot's mrope delta
    /// recorded so the batched decode/spec steps carry the diverged
    /// llama-position. Returns the last row's logits and the total row count
    /// (the engine's KV position for this slot).
    ///
    /// Prefix-cached like the text path - see
    /// [`Self::forward_prefill_slot_mm_encoded`].
    pub fn forward_prefill_slot_mm(
        &mut self,
        slot: usize,
        chunks: &[crate::service::MmChunk],
    ) -> Result<(Vec<f32>, usize), GpuModelError> {
        if self.vision.is_none() {
            return Err(GpuModelError::Unsupported(
                "qwen35 was loaded without an mmproj - configure `mmproj` to enable image input"
                    .into(),
            ));
        }
        // owned VisionOutputs (cache-served or freshly encoded), so the borrow
        // ends before the &mut self prefill work below
        let images = self.encode_all_images(chunks)?;
        self.forward_prefill_slot_mm_encoded(slot, chunks, images)
    }

    /// The slot mm prefill with the vision outputs already encoded - the seam
    /// the batched multi-request encode feeds (one tower pass for the whole
    /// admission wave, then per-slot prefills consume their outputs here).
    ///
    /// PREFIX CACHED, keyed on content rather than on row
    /// tokens: `build_mm_layout` gives every image row the same `0` placeholder
    /// id, so a radix keyed on `ids` would treat two different pictures as the
    /// same prefix and serve one image's KV for the other. Text rows key as
    /// themselves and image rows key off the picture's content hash - see
    /// [`crate::gpu_model::prefix_cache::image_key_row`]. That is what makes the
    /// document workload work: same page, many questions, and every turn after
    /// the first resumes past the whole picture instead of re-prefilling its
    /// 1400-odd soft-token rows.
    ///
    /// qwen35 is the hard family, and granite is the easy one for contrast
    /// (`granite/prefix.rs` explains why). Its DeltaNet layers hold recurrent
    /// state that cannot be rolled back to an arbitrary position, so a resume is
    /// only possible where state was SNAPSHOTTED - the same `m.ckpt` gate the
    /// text path uses, the same two-boundary cut rule (`ckpt_cuts`), and the
    /// same window-extended conv for a span that starts mid-sequence. On top of
    /// that, qwen35 gives every row of one picture equal mRoPE `t` and an
    /// attention bound pointing at the span's last row, so a cut landing inside
    /// a picture has gemma4v's non-causal hazard; `cut_outside_image_spans`
    /// makes such a cut - and therefore such a resume - unreachable.
    pub fn forward_prefill_slot_mm_encoded(
        &mut self,
        slot: usize,
        chunks: &[crate::service::MmChunk],
        images: Vec<crate::gpu_model::qwen35::vision::VisionOutput>,
    ) -> Result<(Vec<f32>, usize), GpuModelError> {
        assert!(self.batch.is_some(), "enable_batch first");
        assert!(slot < self.batch.as_ref().expect("batch").max_batch);
        // token ids (image spans are `0` placeholders), the mRoPE grid, and the
        // equal-t image visibility bound - one ordered walk, any number of images
        let grids: Vec<(usize, usize)> = images.iter().map(|v| (v.nx, v.ny)).collect();
        let lay = build_mm_layout(chunks, &grids)?;
        let t_len = lay.t_len;
        assert!(t_len > 0);
        if t_len > self.max_ctx {
            return Err(GpuModelError::BatchTooLarge {
                got: t_len,
                max: self.max_ctx,
            });
        }
        let keys = mm_radix_keys(&lay, &mm_image_hashes(chunks));
        let img_spans: Vec<(usize, usize)> =
            lay.splices.iter().map(|&(off, n)| (off, off + n)).collect();

        // Same admission shape as the text path: match + restore, then grow the
        // table to cover the whole prompt. `start` is a block-aligned row count
        // already resident in KV (and whose DeltaNet state has been restored),
        // 0 on a cold prompt.
        //
        // P5 budget pool: the slot's block table must back this prompt's KV
        // before the paged appends/attention below read it. Without this the mm
        // prefill wrote DENSE slot*max_ctx offsets into the pool store while
        // decode read through the block table - correct only by the fresh-pool
        // slot-0 coincidence, cross-slot KV corruption under any concurrency
        // (found live: of two concurrent image requests, the blue-image slot
        // answered "red").
        let start = self.mm_prefix_resume(slot, &keys)?;
        if self.batch.as_ref().expect("batch").pool.is_some() {
            self.ensure_slot_blocks(slot, t_len - 1)?;
        }

        // Prefill [start, t_len) in spans that end at the checkpoint cuts, so
        // the DeltaNet state can be snapshotted at each boundary before the
        // following rows advance it - the paged text tail's shape exactly.
        let cuts = self.mm_ckpt_cuts(t_len, start, &img_spans);
        let mut pos = start;
        for c in cuts {
            self.mm_prefill_span(slot, &lay, &images, pos, c)?;
            self.mm_prefix_publish(slot, &keys, c, true)?;
            pos = c;
        }
        let logits = self.mm_prefill_span(slot, &lay, &images, pos, t_len)?;
        // cache every full page of this prompt (idempotent for those inserted
        // at a cut above) so a longer continuation resumes past the last one
        self.mm_prefix_publish(slot, &keys, t_len / BLOCK_TOKENS * BLOCK_TOKENS, false)?;

        self.batch.as_mut().expect("batch").mrope_delta[slot] =
            lay.final_mrope_pos as i64 - t_len as i64;
        Ok((logits, t_len))
    }

    /// Match `keys` against the paged radix and, if a resumable DeltaNet
    /// checkpoint sits under the match, adopt the cached KV pages (zero-copy
    /// refcount) and restore the recurrent state + conv windows into `slot`.
    /// Returns the resume row, 0 for a cold prompt (whose state it zeroes).
    ///
    /// No MTP re-sync, unlike the text path's `prefix_resume_begin`: image
    /// prompts never warm the spec shadow at all (the placeholder ids would
    /// poison it - see the batch tick's note), so there is no cursor to rewind.
    fn mm_prefix_resume(&mut self, slot: usize, keys: &[u32]) -> Result<usize, GpuModelError> {
        let paged = self.batch.as_ref().expect("batch").paged_prefix.is_some();
        let mut start = 0usize;
        if paged {
            let m = {
                let bs = self.batch.as_mut().expect("batch");
                bs.paged_prefix
                    .as_mut()
                    .expect("prefix checked above")
                    .match_full(keys)
            };
            if paddock_models::dev_var_os!("PADDOCK_PREFIX_STATS").is_some() {
                tracing::info!(
                    "qwen35-mm-prefix: slot {slot} rows {} matched {} ckpt {:?} min {}",
                    keys.len(),
                    m.blocks.len() * BLOCK_TOKENS,
                    m.ckpt.map(|(p, _)| p),
                    super::min_cache_prefix()
                );
            }
            if let Some((pos, idx)) = m.ckpt
                && pos >= super::min_cache_prefix()
                && pos < keys.len()
            {
                {
                    let bs = self.batch.as_mut().expect("batch");
                    let pool = bs.pool.as_mut().expect("prefix requires the pool");
                    bs.tables[slot].clear(pool);
                    bs.tables[slot].share_prefix(&m.blocks[..pos / BLOCK_TOKENS], pool);
                }
                self.restore_paged_state(slot, idx)?;
                start = pos;
                // multimodal prefills don't run the drafter warm today, so a
                // resumed mm slot must not keep a stale warm flag from its
                // previous sequence (it would draft against alien pool rows)
                if let Some(sb) = self.spec_batch.as_mut()
                    && slot < sb.alloc_batch
                {
                    sb.mtp_warm[slot] = false;
                }
            }
        }
        if start == 0 {
            // fresh sequence: release the slot's previous blocks and zero the
            // DeltaNet state, so the first span convolves against a zero window
            if self.batch.as_ref().expect("batch").pool.is_some() {
                let bs = self.batch.as_mut().expect("batch");
                let pool = bs.pool.as_mut().expect("pool checked above");
                bs.tables[slot].clear(pool);
            }
            self.zero_slot_state(slot)?;
        }
        self.last_reused[slot] = start;
        Ok(start)
    }

    /// The checkpoint cuts for a multimodal prompt: [`ckpt_cuts`]'s two
    /// boundaries, each walked back out of any image span it lands inside, then
    /// filtered to the ones this prefill can actually reach.
    fn mm_ckpt_cuts(&self, t_len: usize, start: usize, img_spans: &[(usize, usize)]) -> Vec<usize> {
        if self.batch.as_ref().expect("batch").paged_prefix.is_none() {
            return Vec::new();
        }
        let mut out: Vec<usize> = Vec::new();
        for c in ckpt_cuts(t_len, self.tier_ckpt_step()) {
            let c = crate::gpu_model::prefix_cache::cut_outside_image_spans(c, img_spans);
            if c > start && c < t_len && !out.contains(&c) {
                out.push(c);
            }
        }
        out
    }

    /// Publish `slot`'s full pages up to row `upto` into the radix, and (when
    /// `checkpoint`) snapshot the DeltaNet state there so the next turn can
    /// resume at it. A no-op below one page or outside pool mode.
    fn mm_prefix_publish(
        &mut self,
        slot: usize,
        keys: &[u32],
        upto: usize,
        checkpoint: bool,
    ) -> Result<(), GpuModelError> {
        if upto < BLOCK_TOKENS || self.batch.as_ref().expect("batch").paged_prefix.is_none() {
            return Ok(());
        }
        let nb = upto / BLOCK_TOKENS;
        let blocks: Vec<u32> =
            self.batch.as_ref().expect("batch").tables[slot].blocks()[..nb].to_vec();
        let idx = {
            let bs = self.batch.as_mut().expect("batch");
            let pool = bs.pool.as_mut().expect("prefix requires the pool");
            let radix = bs.paged_prefix.as_mut().expect("prefix checked above");
            radix.insert(&keys[..nb * BLOCK_TOKENS], &blocks, pool);
            checkpoint.then(|| radix.attach_state(keys, upto)).flatten()
        };
        if let Some(idx) = idx {
            self.snapshot_paged_state(slot, idx)?;
            // multimodal prefills don't run the drafter warm today, so this
            // records honestly-uncovered (and clears a recycled idx's bit)
            self.record_mtp_cover(idx, slot, upto, keys);
        }
        Ok(())
    }

    /// One prefill pass over rows [a, b) of a multimodal prompt.
    ///
    /// Rows and mRoPE stay ABSOLUTE (the attention bound indexes the whole
    /// prompt, and an image row's bound points at its span's last row, which may
    /// sit in an earlier span), so only the host slices move. Every image lies
    /// entirely inside one span - the cut rule guarantees no boundary falls
    /// inside a picture - so a span either splices a picture whole or not at all.
    fn mm_prefill_span(
        &mut self,
        slot: usize,
        lay: &MmLayout,
        images: &[crate::gpu_model::qwen35::vision::VisionOutput],
        a: usize,
        b: usize,
    ) -> Result<Vec<f32>, GpuModelError> {
        let t_len = lay.t_len;
        debug_assert!(a < b && b <= t_len);
        self.ensure_scratch(b - a)?;

        let exec = self.exec.clone();
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);

        let r = b - a;
        let mut d_tokens = exec.alloc_u32(r)?;
        let mut d_rows = exec.alloc_u32(r)?;
        let mut d_slots = exec.alloc_u32(r)?;
        let mut d_bound = exec.alloc_u32(r)?;
        let mut d_mrope = exec.alloc_u32(4 * r)?;
        // ABSOLUTE rows: these are the sequence's KV positions, and the paged
        // append/attn index the block table by them.
        let rows_host: Vec<u32> = (a as u32..b as u32).collect();
        let slot_host = vec![slot as u32; r];
        // the mRoPE grid is axis-major [4, t_len], so a span takes the same
        // window out of each of the four sections
        let mrope_host: Vec<u32> = (0..4)
            .flat_map(|s| lay.mrope[s * t_len + a..s * t_len + b].iter().copied())
            .collect();
        exec.stream
            .memcpy_htod(&lay.ids[a..b], &mut d_tokens)
            .map_err(drv)?;
        exec.stream
            .memcpy_htod(&rows_host, &mut d_rows)
            .map_err(drv)?;
        exec.stream
            .memcpy_htod(&slot_host, &mut d_slots)
            .map_err(drv)?;
        exec.stream
            .memcpy_htod(&lay.bound[a..b], &mut d_bound)
            .map_err(drv)?;
        exec.stream
            .memcpy_htod(&mrope_host, &mut d_mrope)
            .map_err(drv)?;

        let (embd, n_heads, n_kv_heads, head_dim) =
            (self.embd, self.n_heads, self.n_kv_heads, self.head_dim);
        let (state_size, n_k_heads, n_v_heads, conv_k) =
            (self.state_size, self.n_k_heads, self.n_v_heads, self.conv_k);
        let (conv_dim, ff, max_ctx) = (self.conv_dim, self.ff, self.max_ctx);
        let (n_rot, sections, yarn, eps) =
            (self.n_rot, self.sections, self.yarn_params, self.rms_eps);
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let state_elems = n_v_heads * state_size * state_size;
        // A span starting at row 0 convolves against the zeroed window
        // `mm_prefix_resume` just wrote, so the fused conv's implicit
        // zero-padding is exactly right and it stays on the fast arm. Any later
        // span - after a resume, or after a checkpoint cut - carries real
        // history in the slot's window and takes the window-extended build,
        // same seam as `prefill_slot_chunk`.
        let fresh = a == 0;

        let sinks = &self.sinks;
        let layers = &self.layers;
        let tok_embd = &self.tok_embd;
        let bs_f8ffn_p = &self.bs_f8ffn;
        let bs_f8row_p = &self.bs_f8row_ffn;
        let w8_min = w8_min_batch();
        // PROJECTION e4m3 planes. mm_prefill_span is the VISION twin of the
        // text prefill walk below and had no w8 arm at all -- w8_min was read
        // here only for the FFN. Invisible while both residencies exist, and
        // the one reader that survived the projection REPLACE audit: a stubbed
        // DeltaNet in_qkv [5120, 10240] reaching prefill_mm_pre. Found by the
        // shared-helper guard, which names the tensor; per-call-site guards
        // cannot catch a site nobody had listed.
        let bs_w8_all = &self.bs_w8;
        let sc = self.scratch.as_mut().expect("scratch");
        let bs = self.batch.as_mut().expect("batch");

        // The window-extended conv stages [window | span] through the shared
        // ext buffers, which are sized for a typical resumed tail. A wider one
        // GROWS them rather than asserting: a follow-up turn can legitimately
        // add a whole second picture, and an image's rows cannot be split
        // across passes (they attend to their span's last row, which would not
        // be in KV yet). Growth persists, so it happens at most once per size,
        // and no captured graph holds these - decode replays read the conv
        // WINDOW, never this staging.
        if !fresh {
            let need = (conv_k - 1 + r) * conv_dim;
            if bs.d_conv_ext.len() < need {
                bs.d_conv_ext = exec.alloc(need)?;
                bs.d_conv_out = exec.alloc(need)?;
            }
        }

        embed_any(&exec, tok_embd, &d_tokens, &mut sc.d_x, embd, r)?;
        // inject each image's embeddings over its placeholder rows. Pictures
        // wholly below `a` are already in KV; the cut rule means none can
        // straddle the boundary.
        for (k, &(off, n)) in lay.splices.iter().enumerate() {
            if off >= b || off + n <= a {
                continue;
            }
            debug_assert!(
                off >= a && off + n <= b,
                "image rows [{off}, {}) straddle the prefill span [{a}, {b})",
                off + n
            );
            exec.copy_region(&images[k].embd, 0, &mut sc.d_x, (off - a) * embd, n * embd)?;
        }

        for (li, layer) in layers.iter().enumerate() {
            // the w8 arms quantize from xn, so it must be materialized
            let lw8 = bs_w8_all.get(li).filter(|_| r > w8_min);
            let keep_xn = matches!(&layer.mixer, Mixer::Linear(_)) || lw8.is_some();
            prefill_add_norm_quant(
                &exec,
                &mut sc.d_x,
                None,
                false,
                &layer.attn_norm.buf,
                &mut sc.d_xn,
                keep_xn,
                &mut sc.d_pxq,
                &mut sc.d_pxs,
                &mut sc.d_yq,
                embd,
                r,
                eps,
            )?;
            let mixer_b16 = false;
            match &layer.mixer {
                Mixer::Full(w) => {
                    if let Some(p8) = lw8.and_then(|l| l.wq.as_ref()) {
                        exec.quantize_e4m3(&sc.d_xn, &mut sc.d_pxq, &mut sc.d_exs, r * embd)?;
                        exec.f8_gemm_w8(
                            p8,
                            0,
                            &sc.d_pxq,
                            &sc.d_exs,
                            &mut sc.d_qg,
                            w.wq.dims()[0],
                            w.wq.dims()[1],
                            r,
                        )?;
                        exec.split_qg(&sc.d_qg, &mut sc.d_q, &mut sc.d_gate, r, n_heads, head_dim)?;
                        exec.f8_gemm_w8(
                            p8,
                            w.wq.dims()[1],
                            &sc.d_pxq,
                            &sc.d_exs,
                            &mut sc.d_k,
                            w.wk.dims()[0],
                            w.wk.dims()[1],
                            r,
                        )?;
                        exec.f8_gemm_w8(
                            p8,
                            w.wq.dims()[1] + w.wk.dims()[1],
                            &sc.d_pxq,
                            &sc.d_exs,
                            &mut sc.d_v,
                            w.wv.dims()[0],
                            w.wv.dims()[1],
                            r,
                        )?;
                    } else {
                        prefill_mm_pre_any(
                            &exec,
                            &w.wq,
                            &sc.d_pxq,
                            &sc.d_pxs,
                            &sc.d_yq,
                            &mut sc.d_xsums,
                            &mut sc.d_ssums,
                            &mut sc.d_skfix,
                            &mut sc.d_qg,
                            r,
                        )?;
                        exec.split_qg(&sc.d_qg, &mut sc.d_q, &mut sc.d_gate, r, n_heads, head_dim)?;
                        prefill_mm_pre_any(
                            &exec,
                            &w.wk,
                            &sc.d_pxq,
                            &sc.d_pxs,
                            &sc.d_yq,
                            &mut sc.d_xsums,
                            &mut sc.d_ssums,
                            &mut sc.d_skfix,
                            &mut sc.d_k,
                            r,
                        )?;
                        prefill_mm_pre_any(
                            &exec,
                            &w.wv,
                            &sc.d_pxq,
                            &sc.d_pxs,
                            &sc.d_yq,
                            &mut sc.d_xsums,
                            &mut sc.d_ssums,
                            &mut sc.d_skfix,
                            &mut sc.d_v,
                            r,
                        )?;
                    }
                    exec.rmsnorm_batch(
                        &sc.d_q,
                        &w.q_norm.buf,
                        &mut sc.d_qn,
                        head_dim,
                        eps,
                        r * n_heads,
                    )?;
                    exec.rmsnorm_batch(
                        &sc.d_k,
                        &w.k_norm.buf,
                        &mut sc.d_kn,
                        head_dim,
                        eps,
                        r * n_kv_heads,
                    )?;
                    exec.mrope(
                        &mut sc.d_qn,
                        &d_mrope,
                        r,
                        n_heads,
                        head_dim,
                        n_rot,
                        yarn,
                        sections,
                    )?;
                    exec.mrope(
                        &mut sc.d_kn,
                        &d_mrope,
                        r,
                        n_kv_heads,
                        head_dim,
                        n_rot,
                        yarn,
                        sections,
                    )?;
                    // paged appends/attn through the slot's block table, exactly
                    // as prefill_slot_chunk - the mm rows' positions are d_rows
                    // (fresh sequence, 0..t_len) and the visibility bound stays
                    // the image-extended d_bound
                    if bs.paged {
                        let bt = bs.d_block_tables.as_ref().expect("paged block tables");
                        let bps = bs.blocks_per_slot;
                        exec.kv_append_batch_paged(
                            &sc.d_kn,
                            bs.kv_k[li].as_mut().expect("full-attn layer KV"),
                            &d_rows,
                            Some(&d_slots),
                            bt,
                            bps,
                            kv_dim,
                            r,
                            self.kv_dtype,
                        )?;
                        exec.kv_append_batch_paged(
                            &sc.d_v,
                            bs.kv_v[li].as_mut().expect("full-attn layer KV"),
                            &d_rows,
                            Some(&d_slots),
                            bt,
                            bps,
                            kv_dim,
                            r,
                            self.kv_dtype,
                        )?;
                    } else {
                        exec.kv_append_batch(
                            &sc.d_kn,
                            bs.kv_k[li].as_mut().expect("full-attn layer KV"),
                            &d_rows,
                            Some(&d_slots),
                            kv_dim,
                            max_ctx,
                            r,
                            self.kv_dtype,
                        )?;
                        exec.kv_append_batch(
                            &sc.d_v,
                            bs.kv_v[li].as_mut().expect("full-attn layer KV"),
                            &d_rows,
                            Some(&d_slots),
                            kv_dim,
                            max_ctx,
                            r,
                            self.kv_dtype,
                        )?;
                    }
                    prefill_attn(
                        &exec,
                        &sc.d_qn,
                        bs.kv_k[li].as_ref().expect("full-attn layer KV"),
                        bs.kv_v[li].as_ref().expect("full-attn layer KV"),
                        sinks,
                        &mut sc.d_attn,
                        &d_bound,
                        &d_slots,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        max_ctx,
                        kv_dim,
                        r,
                        scale,
                        self.kv_dtype,
                        bs.d_block_tables
                            .as_ref()
                            .filter(|_| bs.paged)
                            .map(|bt| (bt, bs.blocks_per_slot)),
                        Some((&mut sc.d_attn_o, &mut sc.d_attn_ml)),
                    )?;
                    exec.mul_sigmoid(&mut sc.d_attn, &sc.d_gate, r * q_dim)?;
                    if let Some(p8) = lw8.and_then(|l| l.wo.as_ref()) {
                        exec.quantize_e4m3(
                            &sc.d_attn,
                            &mut sc.d_pxq,
                            &mut sc.d_exs,
                            r * w.wo.dims()[0],
                        )?;
                        exec.f8_gemm_w8(
                            p8,
                            0,
                            &sc.d_pxq,
                            &sc.d_exs,
                            &mut sc.d_proj,
                            w.wo.dims()[0],
                            w.wo.dims()[1],
                            r,
                        )?;
                    } else {
                        prefill_mm_any(
                            &exec,
                            &w.wo,
                            &mut sc.d_pxq,
                            &mut sc.d_pxs,
                            &mut sc.d_yq,
                            &mut sc.d_xsums,
                            &mut sc.d_ssums,
                            &mut sc.d_skfix,
                            &sc.d_attn,
                            &mut sc.d_proj,
                            r,
                        )?;
                    }
                }
                Mixer::Linear(w) => {
                    if let Some(p8) = lw8.and_then(|l| l.in_qkv.as_ref()) {
                        exec.quantize_e4m3(&sc.d_xn, &mut sc.d_pxq, &mut sc.d_exs, r * embd)?;
                        exec.f8_gemm_w8(
                            p8,
                            0,
                            &sc.d_pxq,
                            &sc.d_exs,
                            &mut sc.d_mixed,
                            w.in_qkv.dims()[0],
                            w.in_qkv.dims()[1],
                            r,
                        )?;
                    } else {
                        prefill_mm_pre_any(
                            &exec,
                            &w.in_qkv,
                            &sc.d_pxq,
                            &sc.d_pxs,
                            &sc.d_yq,
                            &mut sc.d_xsums,
                            &mut sc.d_ssums,
                            &mut sc.d_skfix,
                            &mut sc.d_mixed,
                            r,
                        )?;
                    }
                    let km1 = conv_k - 1;
                    let woff = slot * km1 * conv_dim;
                    // vb16 only on the fresh arm: the window-extended path
                    // below produces an f32 d_dv, and the recurrence must be
                    // told which it got.
                    let vb16 = fresh && dn_vb16(&exec, r, state_size);
                    if !fresh {
                        // window-extended conv: rows = [slot's window | this
                        // span], conv over km1+r, keep the last r. The span
                        // starts mid-sequence, so the previous rows' influence
                        // has to come from the window rather than from implicit
                        // zero-padding.
                        {
                            let win = bs.conv_win[li].as_ref().expect("DeltaNet layer conv");
                            debug_assert!((km1 + r) * conv_dim <= bs.d_conv_ext.len());
                            exec.copy_region(win, woff, &mut bs.d_conv_ext, 0, km1 * conv_dim)?;
                        }
                        exec.copy_region(
                            &sc.d_mixed,
                            0,
                            &mut bs.d_conv_ext,
                            km1 * conv_dim,
                            r * conv_dim,
                        )?;
                        exec.causal_conv1d_silu(
                            &bs.d_conv_ext,
                            &w.conv_w.buf,
                            &mut bs.d_conv_out,
                            km1 + r,
                            conv_dim,
                            conv_k,
                        )?;
                        exec.copy_region(
                            &bs.d_conv_out,
                            km1 * conv_dim,
                            &mut sc.d_conv,
                            0,
                            r * conv_dim,
                        )?;
                        // window after this span = the last km1 extended rows
                        // (correct even for r < km1: old window rows carry over)
                        {
                            let win = bs.conv_win[li].as_mut().expect("DeltaNet layer conv");
                            exec.copy_region(
                                &bs.d_conv_ext,
                                r * conv_dim,
                                win,
                                woff,
                                km1 * conv_dim,
                            )?;
                        }
                        exec.deltanet_split_gqa_norm(
                            &sc.d_conv,
                            &mut sc.d_dq,
                            &mut sc.d_dk,
                            &mut sc.d_dv,
                            r,
                            n_k_heads,
                            n_v_heads,
                            state_size,
                        )?;
                    } else {
                        if vb16 {
                            // v-bf16 arm of the fused conv (the DN bf16-operand
                            // chain's severable slice - q/k stay f32)
                            exec.causal_conv1d_silu_qkv_b16_at(
                                &sc.d_mixed,
                                &w.conv_w.buf,
                                &mut sc.d_dq,
                                &mut sc.d_dk,
                                &mut sc.d_dv,
                                0,
                                0,
                                r,
                                n_k_heads,
                                n_v_heads,
                                state_size,
                                conv_k,
                            )?;
                        } else if exec.has_conv_silu_qkv() {
                            // fused conv+split+norm (bit-exact composition) -
                            // d_conv never materializes on the fresh bulk path
                            exec.causal_conv1d_silu_qkv_at(
                                &sc.d_mixed,
                                &w.conv_w.buf,
                                &mut sc.d_dq,
                                &mut sc.d_dk,
                                &mut sc.d_dv,
                                0,
                                0,
                                r,
                                n_k_heads,
                                n_v_heads,
                                state_size,
                                conv_k,
                            )?;
                        } else {
                            exec.causal_conv1d_silu(
                                &sc.d_mixed,
                                &w.conv_w.buf,
                                &mut sc.d_conv,
                                r,
                                conv_dim,
                                conv_k,
                            )?;
                            exec.deltanet_split_gqa_norm(
                                &sc.d_conv,
                                &mut sc.d_dq,
                                &mut sc.d_dk,
                                &mut sc.d_dv,
                                r,
                                n_k_heads,
                                n_v_heads,
                                state_size,
                            )?;
                        }
                        // the fresh arm's window is the tail of its own rows
                        let win = bs.conv_win[li].as_mut().expect("DeltaNet layer conv");
                        if r >= km1 {
                            exec.copy_region(
                                &sc.d_mixed,
                                (r - km1) * conv_dim,
                                win,
                                woff,
                                km1 * conv_dim,
                            )?;
                        } else {
                            exec.copy_region(
                                &sc.d_mixed,
                                0,
                                win,
                                woff + (km1 - r) * conv_dim,
                                r * conv_dim,
                            )?;
                        }
                    }
                    if let Some(ab) = w
                        .ab_f32
                        .as_ref()
                        .filter(|_| r >= ab_f32_min_rows() || w.alpha_w.is_none())
                    {
                        // x2-v3: one f32-plane decay GEMM (64-col tile, x read once) +
                        // fused-layout gate; same values, tiled order (PPL-gated opt-in)
                        ab_gate(
                            &exec,
                            ab,
                            &sc.d_xn,
                            &mut sc.d_ab,
                            &w.ssm_a.buf,
                            &w.dt_bias.buf,
                            &mut sc.d_g,
                            &mut sc.d_beta,
                            r,
                            n_v_heads,
                        )?;
                    } else {
                        if exec.has_q8_0_gemm_repacked_x2() {
                            // fused pair: x staged once for both decay projections
                            // (bit-exact per output vs the two separate calls)
                            exec.q8_0_gemm_repacked_x2(
                                w.alpha_w.as_ref().expect("Q8 alpha (x2 path)"),
                                w.beta_w.as_ref().expect("Q8 beta (x2 path)"),
                                &sc.d_xn,
                                &mut sc.d_a,
                                &mut sc.d_b,
                                r,
                            )?;
                        } else {
                            exec.q8_0_gemm_repacked(
                                w.alpha_w.as_ref().expect("Q8 alpha"),
                                None,
                                &sc.d_xn,
                                &mut sc.d_a,
                                r,
                            )?;
                            exec.q8_0_gemm_repacked(
                                w.beta_w.as_ref().expect("Q8 beta"),
                                None,
                                &sc.d_xn,
                                &mut sc.d_b,
                                r,
                            )?;
                        }
                        exec.delta_gate(
                            &sc.d_a,
                            &sc.d_b,
                            &w.ssm_a.buf,
                            &w.dt_bias.buf,
                            &mut sc.d_g,
                            &mut sc.d_beta,
                            r,
                            n_v_heads,
                        )?;
                    }
                    prefill_delta_recurrent(
                        &exec,
                        sc,
                        bs.recur[li].as_mut().expect("DeltaNet layer state"),
                        slot * state_elems,
                        r,
                        n_v_heads,
                        state_size,
                        vb16,
                    )?;
                    if let Some(p8) = lw8.and_then(|l| l.in_qkv.as_ref()) {
                        // fused in_qkv|gate_w: gate_w rows start at conv_dim
                        exec.quantize_e4m3(&sc.d_xn, &mut sc.d_pxq, &mut sc.d_exs, r * embd)?;
                        exec.f8_gemm_w8(
                            p8,
                            w.in_qkv.dims()[1],
                            &sc.d_pxq,
                            &sc.d_exs,
                            &mut sc.d_z,
                            w.gate_w.dims()[0],
                            w.gate_w.dims()[1],
                            r,
                        )?;
                    } else {
                        prefill_mm_pre_any(
                            &exec,
                            &w.gate_w,
                            &sc.d_pxq,
                            &sc.d_pxs,
                            &sc.d_yq,
                            &mut sc.d_xsums,
                            &mut sc.d_ssums,
                            &mut sc.d_skfix,
                            &mut sc.d_z,
                            r,
                        )?;
                    }
                    exec.gated_rmsnorm(
                        &sc.d_dattn,
                        &sc.d_z,
                        &w.ssm_norm.buf,
                        &mut sc.d_core,
                        r * n_v_heads,
                        state_size,
                        eps,
                    )?;
                    if let Some(p8) = lw8.and_then(|l| l.out_w.as_ref()) {
                        exec.quantize_e4m3(
                            &sc.d_core,
                            &mut sc.d_pxq,
                            &mut sc.d_exs,
                            r * w.out_w.dims()[0],
                        )?;
                        exec.f8_gemm_w8(
                            p8,
                            0,
                            &sc.d_pxq,
                            &sc.d_exs,
                            &mut sc.d_proj,
                            w.out_w.dims()[0],
                            w.out_w.dims()[1],
                            r,
                        )?;
                    } else {
                        prefill_mm_any(
                            &exec,
                            &w.out_w,
                            &mut sc.d_pxq,
                            &mut sc.d_pxs,
                            &mut sc.d_yq,
                            &mut sc.d_xsums,
                            &mut sc.d_ssums,
                            &mut sc.d_skfix,
                            &sc.d_core,
                            &mut sc.d_proj,
                            r,
                        )?;
                    }
                }
            }
            let mut proj_is_b16 = false;
            match &layer.ffn {
                Ffn::Dense { gate, up, down } => {
                    // prefill-FFN f8 arm: the W8 prefill class extended to
                    // the FFN, because ~70% of prefill bytes were still
                    // running through int8-mmq. f8_gemm_w8 measures
                    // 1.27-1.85x best-q8 at M >= 512.
                    // Same e4m3 planes the decode lane built; same w8_min gate.
                    let f8f = bs_f8ffn_p.get(li).and_then(|o| o.as_ref()).filter(|_| {
                        r > super::f8_ffn_pf_min()
                            && paddock_models::dev_var_os!("PADDOCK_F8_ROWSCALE").is_none()
                    });
                    let f8r = bs_f8row_p.get(li).and_then(|o| o.as_ref());
                    // f8 arm: one fused add+norm+e4m3 (xn still written for
                    // alpha/beta) replaces the mmq-quant helper (whose yq is
                    // dead here) + the standalone e4m3 re-read of xn - two
                    // full r x embd passes and a launch saved per layer-tick.
                    let norm_fused = f8f.is_some()
                        && if mixer_b16 {
                            exec.has_add_rmsnorm_e4m3_xn_b16()
                        } else {
                            exec.has_add_rmsnorm_e4m3_xn()
                        };
                    if norm_fused {
                        if mixer_b16 {
                            exec.add_rmsnorm_e4m3_xn_b16(
                                &mut sc.d_x,
                                &sc.d_proj,
                                &layer.post_norm.buf,
                                &mut sc.d_xn,
                                &mut sc.d_pxq,
                                &mut sc.d_exs,
                                embd,
                                r,
                                eps,
                            )?;
                        } else {
                            exec.add_rmsnorm_e4m3_xn(
                                &mut sc.d_x,
                                Some(&sc.d_proj),
                                &layer.post_norm.buf,
                                &mut sc.d_xn,
                                &mut sc.d_pxq,
                                &mut sc.d_exs,
                                embd,
                                r,
                                eps,
                            )?;
                        }
                    } else {
                        prefill_add_norm_quant(
                            &exec,
                            &mut sc.d_x,
                            Some(&sc.d_proj),
                            mixer_b16,
                            &layer.post_norm.buf,
                            &mut sc.d_xn,
                            f8r.is_some() || f8f.is_some(),
                            &mut sc.d_pxq,
                            &mut sc.d_pxs,
                            &mut sc.d_yq,
                            embd,
                            r,
                            eps,
                        )?;
                    }
                    if let Some(p) = f8r {
                        super::ops::ffn_f8row_rows(
                            &exec,
                            p,
                            &sc.d_xn,
                            &mut sc.d_f8t_q,
                            &mut sc.d_f8t_rs,
                            &mut sc.d_ffn_gate,
                            &mut sc.d_ffn_up,
                            &mut sc.d_proj,
                            r,
                        )?;
                    } else if let Some([gu8, d8]) = f8f {
                        // fused plane, row-sliced: gate = rows [0,ff), up =
                        // rows [ff,2ff) - byte-identical to the old separate
                        // planes (same repack stream, offset math only)
                        let ffh = gu8.2 / 2;
                        if !norm_fused {
                            exec.quantize_e4m3(&sc.d_xn, &mut sc.d_pxq, &mut sc.d_exs, r * gu8.1)?;
                        }
                        // bf16 epilogue pair when the pack ships it: halves
                        // the gate/up store traffic (the rival's cutlass
                        // writes bf16; ours wrote f32) and the fused quant
                        // reads bf16 - else the f32 chain below.
                        static O16: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                        let o16 = *O16.get_or_init(|| {
                            paddock_models::dev_var_os!("PADDOCK_NO_F8W8_TMA").is_none()
                                && paddock_models::dev_var_os!("PADDOCK_NO_O16").is_none()
                        });
                        if o16 && exec.has_f8_o16() {
                            if exec.has_swiglu_b16_gu() {
                                // One fused gate|up GEMM - bit-exact vs the
                                // sliced pair (see batch.rs); d_ffn_gate's
                                // r*ffh f32 capacity holds r*2ffh bf16 exactly
                                exec.f8_gemm_w8_o16(
                                    &gu8.0,
                                    0,
                                    &sc.d_pxq,
                                    &sc.d_exs,
                                    &mut sc.d_ffn_gate,
                                    gu8.1,
                                    gu8.2,
                                    r,
                                )?;
                                exec.quantize_e4m3_swiglu_b16_gu(
                                    &sc.d_ffn_gate,
                                    &mut sc.d_pxq,
                                    &mut sc.d_exs,
                                    r * d8.1,
                                    ffh,
                                )?;
                            } else {
                                exec.f8_gemm_w8_o16(
                                    &gu8.0,
                                    0,
                                    &sc.d_pxq,
                                    &sc.d_exs,
                                    &mut sc.d_ffn_gate,
                                    gu8.1,
                                    ffh,
                                    r,
                                )?;
                                exec.f8_gemm_w8_o16(
                                    &gu8.0,
                                    ffh,
                                    &sc.d_pxq,
                                    &sc.d_exs,
                                    &mut sc.d_ffn_up,
                                    gu8.1,
                                    ffh,
                                    r,
                                )?;
                                exec.quantize_e4m3_swiglu_b16(
                                    &sc.d_ffn_gate,
                                    &sc.d_ffn_up,
                                    &mut sc.d_pxq,
                                    &mut sc.d_exs,
                                    r * d8.1,
                                )?;
                            }
                        } else {
                            exec.f8_gemm_w8(
                                &gu8.0,
                                0,
                                &sc.d_pxq,
                                &sc.d_exs,
                                &mut sc.d_ffn_gate,
                                gu8.1,
                                ffh,
                                r,
                            )?;
                            exec.f8_gemm_w8(
                                &gu8.0,
                                ffh,
                                &sc.d_pxq,
                                &sc.d_exs,
                                &mut sc.d_ffn_up,
                                gu8.1,
                                ffh,
                                r,
                            )?;
                            // fused swiglu+e4m3-quant: one pass instead of
                            // swiglu-write + quant-read (286 MB/layer-tick of f32
                            // round-trip at r=2048 - the bf16-activations gap
                            // vs the engines that write bf16 epilogues, closed at
                            // the seam that matters)
                            exec.quantize_e4m3_swiglu(
                                &sc.d_ffn_gate,
                                &sc.d_ffn_up,
                                &mut sc.d_pxq,
                                &mut sc.d_exs,
                                r * d8.1,
                            )?;
                        }
                        if o16 && exec.has_f8_o16() && exec.has_add_b16() {
                            // bf16 down out (halves the last 42 MB/layer-tick
                            // f32 store of the FFN) - the tail add reads bf16
                            exec.f8_gemm_w8_o16(
                                &d8.0,
                                0,
                                &sc.d_pxq,
                                &sc.d_exs,
                                &mut sc.d_proj,
                                d8.1,
                                d8.2,
                                r,
                            )?;
                            proj_is_b16 = true;
                        } else {
                            exec.f8_gemm_w8(
                                &d8.0,
                                0,
                                &sc.d_pxq,
                                &sc.d_exs,
                                &mut sc.d_proj,
                                d8.1,
                                d8.2,
                                r,
                            )?;
                        }
                    } else {
                        prefill_mm_pre_any(
                            &exec,
                            gate,
                            &sc.d_pxq,
                            &sc.d_pxs,
                            &sc.d_yq,
                            &mut sc.d_xsums,
                            &mut sc.d_ssums,
                            &mut sc.d_skfix,
                            &mut sc.d_ffn_gate,
                            r,
                        )?;
                        prefill_mm_pre_any(
                            &exec,
                            up,
                            &sc.d_pxq,
                            &sc.d_pxs,
                            &sc.d_yq,
                            &mut sc.d_xsums,
                            &mut sc.d_ssums,
                            &mut sc.d_skfix,
                            &mut sc.d_ffn_up,
                            r,
                        )?;
                        prefill_ffn_down_any(
                            &exec,
                            down,
                            &mut sc.d_pxq,
                            &mut sc.d_pxs,
                            &mut sc.d_yq,
                            &mut sc.d_xsums,
                            &mut sc.d_ssums,
                            &mut sc.d_skfix,
                            &mut sc.d_ffn_gate,
                            &sc.d_ffn_up,
                            &mut sc.d_proj,
                            ff,
                            r,
                        )?;
                    }
                }
                Ffn::Nvf4Dense { gate, up, down } => {
                    // Above the row band the chain takes the same f8w arm the
                    // Dense lane runs, off planes load.rs builds from the
                    // NVFP4 checkpoint's own values. Without it this path
                    // falls to nvf4_ffn, whose W4A16 tcp kernel is a
                    // DECODE-band arm and costs ~7x the wide-prefill chain.
                    let f8f = bs_f8ffn_p.get(li).and_then(|o| o.as_ref()).filter(|_| {
                        r > nvf4_f8w_min_rows(w8_min)
                            && paddock_models::dev_var_os!("PADDOCK_F8_ROWSCALE").is_none()
                    });
                    // xn stays written: alpha/beta read it either way.
                    prefill_add_norm_quant(
                        &exec,
                        &mut sc.d_x,
                        Some(&sc.d_proj),
                        mixer_b16,
                        &layer.post_norm.buf,
                        &mut sc.d_xn,
                        true,
                        &mut sc.d_pxq,
                        &mut sc.d_pxs,
                        &mut sc.d_yq,
                        embd,
                        r,
                        eps,
                    )?;
                    if let Some([gu8, d8]) = f8f {
                        proj_is_b16 = prefill_ffn_f8w(
                            &exec,
                            gu8,
                            d8,
                            &sc.d_xn,
                            &mut sc.d_pxq,
                            &mut sc.d_exs,
                            &mut sc.d_ffn_gate,
                            &mut sc.d_ffn_up,
                            &mut sc.d_proj,
                            r,
                        )?;
                    } else {
                        nvf4_ffn(
                            &exec,
                            gate,
                            up,
                            down,
                            &sc.d_xn,
                            &mut sc.d_pxq,
                            &mut sc.d_nvs,
                            &mut sc.d_nv4part,
                            &mut sc.d_ffn_gate,
                            &mut sc.d_ffn_up,
                            &mut sc.d_proj,
                            ff,
                            r,
                        )?;
                    }
                }
                Ffn::Moe(w) => {
                    // MoE needs the f32 xn (router + shared expert)
                    prefill_add_norm_quant(
                        &exec,
                        &mut sc.d_x,
                        Some(&sc.d_proj),
                        mixer_b16,
                        &layer.post_norm.buf,
                        &mut sc.d_xn,
                        true,
                        &mut sc.d_pxq,
                        &mut sc.d_pxs,
                        &mut sc.d_yq,
                        embd,
                        r,
                        eps,
                    )?;
                    moe_ffn(
                        &exec,
                        w,
                        self.moe.expect("moe dims"),
                        embd,
                        r,
                        true,
                        &sc.d_xn,
                        &mut sc.d_moe_xq,
                        &mut sc.d_moe_xs,
                        &mut sc.d_ssums,
                        &mut sc.d_moe_xs8,
                        &mut sc.d_moe_fs8,
                        &mut sc.d_moe_logits,
                        &sc.d_zero_bias,
                        &mut sc.d_moe_idx,
                        &mut sc.d_moe_w,
                        &mut sc.d_moe_fused,
                        &mut sc.d_moe_fq,
                        &mut sc.d_moe_fs,
                        &mut sc.d_moe_srow,
                        &mut sc.d_moe_sslot,
                        &mut sc.d_moe_bexp,
                        &mut sc.d_moe_part,
                        &mut sc.d_pxq,
                        &mut sc.d_pxs,
                        &mut sc.d_yq,
                        &mut sc.d_skfix,
                        &mut sc.d_ffn_gate,
                        &mut sc.d_ffn_up,
                        &mut sc.d_mixed,
                        &mut sc.d_proj,
                    )?;
                }
            }
            if proj_is_b16 {
                exec.add_b16(&mut sc.d_x, &sc.d_proj, r * embd)?;
            } else {
                exec.add(&mut sc.d_x, &sc.d_proj, r * embd)?;
            }
        }

        exec.rmsnorm_batch(&sc.d_x, &self.out_norm.buf, &mut sc.d_h, embd, eps, r)?;
        exec.copy_region(&sc.d_h, (r - 1) * embd, &mut sc.d_xn, 0, embd)?;
        // f8 head when the floor elects it: this is a ONE-ROW head call on the
        // prefill path, and it was a direct Q8_0 reader - the site the head
        // reclaim's first audit missed (the FFN arms in this file were checked,
        // the head was not; the stub gave an illegal access, loudly).
        if let Some(p) = super::head_f8(self.out_f8.as_ref(), 1) {
            super::head_f8_gemm(
                &exec,
                p,
                &sc.d_xn,
                &mut sc.d_pxq,
                &mut sc.d_exs,
                &mut bs.d_ks_part,
                &mut sc.d_logits,
                1,
            )?;
        } else {
            super::stub_guard(&self.output, "prefix.rs prefill head")?;
            gemv_any(&exec, &self.output, &sc.d_xn, &mut sc.d_logits)?;
        }
        // only the last span's logits are the prompt's; the caller drops the
        // rest. Computing them for every span costs one gemv over the vocab per
        // cut - two extra on a checkpointed prompt, next to 40 layers of work.
        let logits = exec.to_host(&sc.d_logits)?;
        Ok(logits)
    }

    /// One prefill pass over `tokens` at positions [start, start+len) of
    /// `slot`'s sequence. The slot's DeltaNet states/windows carry across
    /// calls, so the orchestrator can split a prompt at checkpoint cuts or
    /// resume behind a restored checkpoint. One cuBLAS-tensor-core pass;
    /// returns the last row's logits.
    pub(super) fn prefill_slot_chunk(
        &mut self,
        slot: usize,
        tokens: &[u32],
        start: usize,
    ) -> Result<Vec<f32>, GpuModelError> {
        let t_len = tokens.len();
        assert!(t_len > 0 && start + t_len <= self.max_ctx);
        self.ensure_scratch(t_len)?;
        let exec = self.exec.clone();
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);

        let mut d_tokens = exec.alloc_u32(t_len)?;
        let mut d_positions = exec.alloc_u32(t_len)?;
        let mut d_slots = exec.alloc_u32(t_len)?;
        let mut d_mrope = exec.alloc_u32(4 * t_len)?;
        let pos_host: Vec<u32> = (start as u32..(start + t_len) as u32).collect();
        let mrope_host: Vec<u32> = (0..4).flat_map(|_| pos_host.iter().copied()).collect();
        let slot_host = vec![slot as u32; t_len];
        exec.stream
            .memcpy_htod(tokens, &mut d_tokens)
            .map_err(drv)?;
        exec.stream
            .memcpy_htod(&pos_host, &mut d_positions)
            .map_err(drv)?;
        exec.stream
            .memcpy_htod(&slot_host, &mut d_slots)
            .map_err(drv)?;
        exec.stream
            .memcpy_htod(&mrope_host, &mut d_mrope)
            .map_err(drv)?;

        // P5 budget pool: (re)allocate this slot's blocks for [start, start+t_len).
        // A fresh prompt (start == 0) returns the slot's previous blocks first
        // (slot reuse across requests). The paged prefill append/attn below then
        // read the freshly grown block table. No-op in identity/dense mode.
        if self.batch.as_ref().expect("batch").pool.is_some() {
            if start == 0 {
                let bs = self.batch.as_mut().expect("batch");
                let pool = bs.pool.as_mut().expect("pool checked above");
                bs.tables[slot].clear(pool);
            }
            self.ensure_slot_blocks(slot, start + t_len - 1)?;
        }

        let (embd, n_heads, n_kv_heads, head_dim) =
            (self.embd, self.n_heads, self.n_kv_heads, self.head_dim);
        let (state_size, n_k_heads, n_v_heads, conv_k) =
            (self.state_size, self.n_k_heads, self.n_v_heads, self.conv_k);
        let (conv_dim, ff, max_ctx) = (self.conv_dim, self.ff, self.max_ctx);
        let (n_rot, sections, yarn, eps) =
            (self.n_rot, self.sections, self.yarn_params, self.rms_eps);
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let r = t_len;
        let state_elems = n_v_heads * state_size * state_size;

        let sinks = &self.sinks;
        let layers = &self.layers;
        let tok_embd = &self.tok_embd;
        // b1: fp8 W8A8 dense-proj planes (empty unless PADDOCK_QWEN35_W8). Consulted
        // only above w8_min for large (prefill) batches; else the exact Q8_0 path.
        let bs_w8_all = &self.bs_w8;
        let w8_min = w8_min_batch();
        // nvf4 W4A4 dense-proj planes (empty unless PADDOCK_QWEN35_PROJ_NV4).
        let bs_nv4_all = &self.bs_nv4;
        let bs_f8ffn_p = &self.bs_f8ffn;
        let bs_f8row_p = &self.bs_f8row_ffn;
        let nv4_min = proj_nv4_min_batch();
        let sc = self.scratch.as_mut().expect("scratch");
        let bs = self.batch.as_mut().expect("batch");

        embed_any(&exec, tok_embd, &d_tokens, &mut sc.d_x, embd, r)?;

        for (li, layer) in layers.iter().enumerate() {
            // W8 planes for this layer, only when the batch is large enough to win.
            let lw8 = bs_w8_all.get(li).filter(|_| r > w8_min);
            if li == 0 && paddock_models::dev_var_os!("PADDOCK_W8_TRACE").is_some() {
                tracing::info!(
                    rows = r,
                    w8_min,
                    planes = bs_w8_all.len(),
                    lw8 = lw8.is_some(),
                    "qwen35 prefill W8 consult"
                );
            }
            // nv4 planes take priority over W8 when both are loaded (nv4 is the
            // 2×-MMA win; W8 was ~parity). Only above the large-batch threshold.
            let lnv4 = bs_nv4_all.get(li).filter(|_| r > nv4_min);
            // attn_norm fused with the qkv/in_qkv quantize (P6k); xn only
            // materializes for Linear mixers (alpha/beta still read it) - but the
            // W8/nv4 paths also need the f32 normed hidden as their quant source,
            // so force it there too.
            let keep_xn =
                matches!(&layer.mixer, Mixer::Linear(_)) || lw8.is_some() || lnv4.is_some();
            // W8-taken layers: one fused norm+e4m3 replaces the mmq-quant
            // helper (yq dead on the w8 arms) + the standalone e4m3 quant of
            // xn below - same seam as the FFN post-norm cut. xn still
            // written (alpha/beta + reuse comments hold).
            let entry_fused = lnv4.is_none() && lw8.is_some() && exec.has_add_rmsnorm_e4m3_xn();
            if entry_fused {
                exec.add_rmsnorm_e4m3_xn(
                    &mut sc.d_x,
                    None,
                    &layer.attn_norm.buf,
                    &mut sc.d_xn,
                    &mut sc.d_pxq,
                    &mut sc.d_exs,
                    embd,
                    r,
                    eps,
                )?;
            } else {
                prefill_add_norm_quant(
                    &exec,
                    &mut sc.d_x,
                    None,
                    false,
                    &layer.attn_norm.buf,
                    &mut sc.d_xn,
                    keep_xn,
                    &mut sc.d_pxq,
                    &mut sc.d_pxs,
                    &mut sc.d_yq,
                    embd,
                    r,
                    eps,
                )?;
            }
            let mut mixer_b16 = false;
            match &layer.mixer {
                Mixer::Full(w) => {
                    if let Some(l4) = lnv4 {
                        // One nvf4 quant of the normed hidden feeds wq/wk/wv (W4A4).
                        exec.quantize_nvf4(&sc.d_xn, &mut sc.d_pxq, &mut sc.d_nvs, r * embd)?;
                        exec.mxfp4_gemm_nv4(
                            l4.wq.as_ref().expect("Full mixer nv4 plane"),
                            &sc.d_pxq,
                            &sc.d_nvs,
                            &mut sc.d_qg,
                            w.wq.dims()[0],
                            w.wq.dims()[1],
                            r,
                        )?;
                        exec.split_qg(&sc.d_qg, &mut sc.d_q, &mut sc.d_gate, r, n_heads, head_dim)?;
                        exec.mxfp4_gemm_nv4(
                            l4.wk.as_ref().expect("Full mixer nv4 plane"),
                            &sc.d_pxq,
                            &sc.d_nvs,
                            &mut sc.d_k,
                            w.wk.dims()[0],
                            w.wk.dims()[1],
                            r,
                        )?;
                        exec.mxfp4_gemm_nv4(
                            l4.wv.as_ref().expect("Full mixer nv4 plane"),
                            &sc.d_pxq,
                            &sc.d_nvs,
                            &mut sc.d_v,
                            w.wv.dims()[0],
                            w.wv.dims()[1],
                            r,
                        )?;
                    } else if let Some(l8) = lw8 {
                        // One e4m3 quant of the normed hidden feeds wq/wk/wv.
                        if !entry_fused {
                            exec.quantize_e4m3(&sc.d_xn, &mut sc.d_pxq, &mut sc.d_exs, r * embd)?;
                        }
                        exec.f8_gemm_w8(
                            l8.wq.as_ref().expect("Full mixer W8 plane"),
                            0,
                            &sc.d_pxq,
                            &sc.d_exs,
                            &mut sc.d_qg,
                            w.wq.dims()[0],
                            w.wq.dims()[1],
                            r,
                        )?;
                        exec.split_qg(&sc.d_qg, &mut sc.d_q, &mut sc.d_gate, r, n_heads, head_dim)?;
                        exec.f8_gemm_w8(
                            l8.wq.as_ref().expect("Full mixer W8 plane"),
                            w.wq.dims()[1],
                            &sc.d_pxq,
                            &sc.d_exs,
                            &mut sc.d_k,
                            w.wk.dims()[0],
                            w.wk.dims()[1],
                            r,
                        )?;
                        exec.f8_gemm_w8(
                            l8.wq.as_ref().expect("Full mixer W8 plane"),
                            w.wq.dims()[1] + w.wk.dims()[1],
                            &sc.d_pxq,
                            &sc.d_exs,
                            &mut sc.d_v,
                            w.wv.dims()[0],
                            w.wv.dims()[1],
                            r,
                        )?;
                    } else {
                        prefill_mm_pre_any(
                            &exec,
                            &w.wq,
                            &sc.d_pxq,
                            &sc.d_pxs,
                            &sc.d_yq,
                            &mut sc.d_xsums,
                            &mut sc.d_ssums,
                            &mut sc.d_skfix,
                            &mut sc.d_qg,
                            r,
                        )?;
                        exec.split_qg(&sc.d_qg, &mut sc.d_q, &mut sc.d_gate, r, n_heads, head_dim)?;
                        prefill_mm_pre_any(
                            &exec,
                            &w.wk,
                            &sc.d_pxq,
                            &sc.d_pxs,
                            &sc.d_yq,
                            &mut sc.d_xsums,
                            &mut sc.d_ssums,
                            &mut sc.d_skfix,
                            &mut sc.d_k,
                            r,
                        )?;
                        prefill_mm_pre_any(
                            &exec,
                            &w.wv,
                            &sc.d_pxq,
                            &sc.d_pxs,
                            &sc.d_yq,
                            &mut sc.d_xsums,
                            &mut sc.d_ssums,
                            &mut sc.d_skfix,
                            &mut sc.d_v,
                            r,
                        )?;
                    }
                    exec.rmsnorm_batch(
                        &sc.d_q,
                        &w.q_norm.buf,
                        &mut sc.d_qn,
                        head_dim,
                        eps,
                        r * n_heads,
                    )?;
                    exec.rmsnorm_batch(
                        &sc.d_k,
                        &w.k_norm.buf,
                        &mut sc.d_kn,
                        head_dim,
                        eps,
                        r * n_kv_heads,
                    )?;
                    exec.mrope(
                        &mut sc.d_qn,
                        &d_mrope,
                        r,
                        n_heads,
                        head_dim,
                        n_rot,
                        yarn,
                        sections,
                    )?;
                    exec.mrope(
                        &mut sc.d_kn,
                        &d_mrope,
                        r,
                        n_kv_heads,
                        head_dim,
                        n_rot,
                        yarn,
                        sections,
                    )?;
                    if bs.paged {
                        let bt = bs.d_block_tables.as_ref().expect("paged block tables");
                        let bps = bs.blocks_per_slot;
                        exec.kv_append_batch_paged(
                            &sc.d_kn,
                            bs.kv_k[li].as_mut().expect("full-attn layer KV"),
                            &d_positions,
                            Some(&d_slots),
                            bt,
                            bps,
                            kv_dim,
                            r,
                            self.kv_dtype,
                        )?;
                        exec.kv_append_batch_paged(
                            &sc.d_v,
                            bs.kv_v[li].as_mut().expect("full-attn layer KV"),
                            &d_positions,
                            Some(&d_slots),
                            bt,
                            bps,
                            kv_dim,
                            r,
                            self.kv_dtype,
                        )?;
                    } else {
                        exec.kv_append_batch(
                            &sc.d_kn,
                            bs.kv_k[li].as_mut().expect("full-attn layer KV"),
                            &d_positions,
                            Some(&d_slots),
                            kv_dim,
                            max_ctx,
                            r,
                            self.kv_dtype,
                        )?;
                        exec.kv_append_batch(
                            &sc.d_v,
                            bs.kv_v[li].as_mut().expect("full-attn layer KV"),
                            &d_positions,
                            Some(&d_slots),
                            kv_dim,
                            max_ctx,
                            r,
                            self.kv_dtype,
                        )?;
                    }
                    prefill_attn(
                        &exec,
                        &sc.d_qn,
                        bs.kv_k[li].as_ref().expect("full-attn layer KV"),
                        bs.kv_v[li].as_ref().expect("full-attn layer KV"),
                        sinks,
                        &mut sc.d_attn,
                        &d_positions,
                        &d_slots,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        max_ctx,
                        kv_dim,
                        r,
                        scale,
                        self.kv_dtype,
                        bs.d_block_tables
                            .as_ref()
                            .filter(|_| bs.paged)
                            .map(|bt| (bt, bs.blocks_per_slot)),
                        Some((&mut sc.d_attn_o, &mut sc.d_attn_ml)),
                    )?;
                    exec.mul_sigmoid(&mut sc.d_attn, &sc.d_gate, r * q_dim)?;
                    if let Some(l4) = lnv4 {
                        exec.quantize_nvf4(
                            &sc.d_attn,
                            &mut sc.d_pxq,
                            &mut sc.d_nvs,
                            r * w.wo.dims()[0],
                        )?;
                        exec.mxfp4_gemm_nv4(
                            l4.wo.as_ref().expect("Full mixer nv4 plane"),
                            &sc.d_pxq,
                            &sc.d_nvs,
                            &mut sc.d_proj,
                            w.wo.dims()[0],
                            w.wo.dims()[1],
                            r,
                        )?;
                    } else if let Some(l8) = lw8 {
                        exec.quantize_e4m3(
                            &sc.d_attn,
                            &mut sc.d_pxq,
                            &mut sc.d_exs,
                            r * w.wo.dims()[0],
                        )?;
                        // mixer bf16 seam: wo writes bf16, the post_norm
                        // entry (ABI-247 consumer) reads it back
                        static MO16: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                        let mo16 = *MO16.get_or_init(|| {
                            paddock_models::dev_var_os!("PADDOCK_NO_F8W8_TMA").is_none()
                                && paddock_models::dev_var_os!("PADDOCK_NO_O16").is_none()
                        });
                        if mo16 && exec.has_f8_o16() {
                            exec.f8_gemm_w8_o16(
                                l8.wo.as_ref().expect("Full mixer W8 plane"),
                                0,
                                &sc.d_pxq,
                                &sc.d_exs,
                                &mut sc.d_proj,
                                w.wo.dims()[0],
                                w.wo.dims()[1],
                                r,
                            )?;
                            mixer_b16 = true;
                        } else {
                            exec.f8_gemm_w8(
                                l8.wo.as_ref().expect("Full mixer W8 plane"),
                                0,
                                &sc.d_pxq,
                                &sc.d_exs,
                                &mut sc.d_proj,
                                w.wo.dims()[0],
                                w.wo.dims()[1],
                                r,
                            )?;
                        }
                    } else {
                        prefill_mm_any(
                            &exec,
                            &w.wo,
                            &mut sc.d_pxq,
                            &mut sc.d_pxs,
                            &mut sc.d_yq,
                            &mut sc.d_xsums,
                            &mut sc.d_ssums,
                            &mut sc.d_skfix,
                            &sc.d_attn,
                            &mut sc.d_proj,
                            r,
                        )?;
                    }
                }
                Mixer::Linear(w) => {
                    // input quantized by the fused attn_norm above (P6k)
                    if let Some(l4) = lnv4 {
                        // nvf4-quant the normed hidden once; it feeds in_qkv AND
                        // (unchanged since) gate_w below.
                        exec.quantize_nvf4(&sc.d_xn, &mut sc.d_pxq, &mut sc.d_nvs, r * embd)?;
                        exec.mxfp4_gemm_nv4(
                            l4.in_qkv.as_ref().expect("Linear mixer nv4 plane"),
                            &sc.d_pxq,
                            &sc.d_nvs,
                            &mut sc.d_mixed,
                            w.in_qkv.dims()[0],
                            w.in_qkv.dims()[1],
                            r,
                        )?;
                    } else if let Some(l8) = lw8 {
                        // e4m3-quant the normed hidden once; it feeds in_qkv AND
                        // (unchanged since) gate_w below.
                        if !entry_fused {
                            exec.quantize_e4m3(&sc.d_xn, &mut sc.d_pxq, &mut sc.d_exs, r * embd)?;
                        }
                        exec.f8_gemm_w8(
                            l8.in_qkv.as_ref().expect("Linear mixer W8 plane"),
                            0,
                            &sc.d_pxq,
                            &sc.d_exs,
                            &mut sc.d_mixed,
                            w.in_qkv.dims()[0],
                            w.in_qkv.dims()[1],
                            r,
                        )?;
                    } else {
                        prefill_mm_pre_any(
                            &exec,
                            &w.in_qkv,
                            &sc.d_pxq,
                            &sc.d_pxs,
                            &sc.d_yq,
                            &mut sc.d_xsums,
                            &mut sc.d_ssums,
                            &mut sc.d_skfix,
                            &mut sc.d_mixed,
                            r,
                        )?;
                    }
                    // window-extended conv (the spec-path pattern): rows =
                    // [slot's window | this chunk], conv over km1+r, keep the
                    // last r. Bit-identical to implicit zero-padding when the
                    // window is zeroed (fresh sequence) and seamless when the
                    // chunk starts mid-sequence (resume / checkpoint cut).
                    let km1 = conv_k - 1;
                    let woff = slot * km1 * conv_dim;
                    {
                        let win = bs.conv_win[li].as_ref().expect("DeltaNet layer conv");
                        assert!(
                            (km1 + r) * conv_dim <= bs.d_conv_ext.len(),
                            "resume chunk {r} rows outgrew the conv ext staging"
                        );
                        exec.copy_region(win, woff, &mut bs.d_conv_ext, 0, km1 * conv_dim)?;
                    }
                    exec.copy_region(
                        &sc.d_mixed,
                        0,
                        &mut bs.d_conv_ext,
                        km1 * conv_dim,
                        r * conv_dim,
                    )?;
                    exec.causal_conv1d_silu(
                        &bs.d_conv_ext,
                        &w.conv_w.buf,
                        &mut bs.d_conv_out,
                        km1 + r,
                        conv_dim,
                        conv_k,
                    )?;
                    exec.copy_region(
                        &bs.d_conv_out,
                        km1 * conv_dim,
                        &mut sc.d_conv,
                        0,
                        r * conv_dim,
                    )?;
                    // window after this chunk = the last km1 extended rows
                    // (correct even for r < km1: old window rows carry over)
                    {
                        let win = bs.conv_win[li].as_mut().expect("DeltaNet layer conv");
                        exec.copy_region(&bs.d_conv_ext, r * conv_dim, win, woff, km1 * conv_dim)?;
                    }
                    exec.deltanet_split_gqa_norm(
                        &sc.d_conv,
                        &mut sc.d_dq,
                        &mut sc.d_dk,
                        &mut sc.d_dv,
                        r,
                        n_k_heads,
                        n_v_heads,
                        state_size,
                    )?;
                    // alpha/beta stay on the exact f32 path, matching `prefill`:
                    // they feed g, the decay that multiplies the whole recurrent
                    // state every token - f16 staging here would bake a different
                    // numeric class into the slot's state than the single path
                    // (found as a 38-token-in greedy flip in the spec-batch gate).
                    if let Some(ab) = w
                        .ab_f32
                        .as_ref()
                        .filter(|_| r >= ab_f32_min_rows() || w.alpha_w.is_none())
                    {
                        // x2-v3: one f32-plane decay GEMM (64-col tile, x read once) +
                        // fused-layout gate; same values, tiled order (PPL-gated opt-in)
                        ab_gate(
                            &exec,
                            ab,
                            &sc.d_xn,
                            &mut sc.d_ab,
                            &w.ssm_a.buf,
                            &w.dt_bias.buf,
                            &mut sc.d_g,
                            &mut sc.d_beta,
                            r,
                            n_v_heads,
                        )?;
                    } else {
                        if exec.has_q8_0_gemm_repacked_x2() {
                            // fused pair: x staged once for both decay projections
                            // (bit-exact per output vs the two separate calls)
                            exec.q8_0_gemm_repacked_x2(
                                w.alpha_w.as_ref().expect("Q8 alpha (x2 path)"),
                                w.beta_w.as_ref().expect("Q8 beta (x2 path)"),
                                &sc.d_xn,
                                &mut sc.d_a,
                                &mut sc.d_b,
                                r,
                            )?;
                        } else {
                            exec.q8_0_gemm_repacked(
                                w.alpha_w.as_ref().expect("Q8 alpha"),
                                None,
                                &sc.d_xn,
                                &mut sc.d_a,
                                r,
                            )?;
                            exec.q8_0_gemm_repacked(
                                w.beta_w.as_ref().expect("Q8 beta"),
                                None,
                                &sc.d_xn,
                                &mut sc.d_b,
                                r,
                            )?;
                        }
                        exec.delta_gate(
                            &sc.d_a,
                            &sc.d_b,
                            &w.ssm_a.buf,
                            &w.dt_bias.buf,
                            &mut sc.d_g,
                            &mut sc.d_beta,
                            r,
                            n_v_heads,
                        )?;
                    }
                    prefill_delta_recurrent(
                        &exec,
                        sc,
                        bs.recur[li].as_mut().expect("DeltaNet layer state"),
                        slot * state_elems,
                        r,
                        n_v_heads,
                        state_size,
                        false,
                    )?;
                    // d_xn/d_pxq/d_exs (or d_yq on the Q8 path) untouched since
                    // in_qkv's quant: reuse the same e4m3 activations for gate_w.
                    if let Some(l4) = lnv4 {
                        // reuses the in_qkv nvf4 activation (d_pxq/d_nvs untouched since).
                        exec.mxfp4_gemm_nv4(
                            l4.gate_w.as_ref().expect("Linear mixer nv4 plane"),
                            &sc.d_pxq,
                            &sc.d_nvs,
                            &mut sc.d_z,
                            w.gate_w.dims()[0],
                            w.gate_w.dims()[1],
                            r,
                        )?;
                    } else if let Some(l8) = lw8 {
                        exec.f8_gemm_w8(
                            l8.in_qkv.as_ref().expect("Linear mixer W8 plane"),
                            w.in_qkv.dims()[1],
                            &sc.d_pxq,
                            &sc.d_exs,
                            &mut sc.d_z,
                            w.gate_w.dims()[0],
                            w.gate_w.dims()[1],
                            r,
                        )?;
                    } else {
                        prefill_mm_pre_any(
                            &exec,
                            &w.gate_w,
                            &sc.d_pxq,
                            &sc.d_pxs,
                            &sc.d_yq,
                            &mut sc.d_xsums,
                            &mut sc.d_ssums,
                            &mut sc.d_skfix,
                            &mut sc.d_z,
                            r,
                        )?;
                    }
                    // DN out_proj glue: gated norm + e4m3 quant in one pass
                    // on the w8 arm (scale math bit-matches the standalone
                    // quantize; f32 core still written for fallbacks).
                    let gr_fused = lnv4.is_none() && lw8.is_some() && exec.has_gated_rmsnorm_e4m3();
                    if gr_fused {
                        exec.gated_rmsnorm_e4m3(
                            &sc.d_dattn,
                            &sc.d_z,
                            &w.ssm_norm.buf,
                            Some(&mut sc.d_core),
                            &mut sc.d_pxq,
                            &mut sc.d_exs,
                            r * n_v_heads,
                            state_size,
                            eps,
                        )?;
                    } else {
                        exec.gated_rmsnorm(
                            &sc.d_dattn,
                            &sc.d_z,
                            &w.ssm_norm.buf,
                            &mut sc.d_core,
                            r * n_v_heads,
                            state_size,
                            eps,
                        )?;
                    }
                    if let Some(l4) = lnv4 {
                        exec.quantize_nvf4(
                            &sc.d_core,
                            &mut sc.d_pxq,
                            &mut sc.d_nvs,
                            r * w.out_w.dims()[0],
                        )?;
                        exec.mxfp4_gemm_nv4(
                            l4.out_w.as_ref().expect("Linear mixer nv4 plane"),
                            &sc.d_pxq,
                            &sc.d_nvs,
                            &mut sc.d_proj,
                            w.out_w.dims()[0],
                            w.out_w.dims()[1],
                            r,
                        )?;
                    } else if let Some(l8) = lw8 {
                        if !gr_fused {
                            exec.quantize_e4m3(
                                &sc.d_core,
                                &mut sc.d_pxq,
                                &mut sc.d_exs,
                                r * w.out_w.dims()[0],
                            )?;
                        }
                        static MO16: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                        let mo16 = *MO16.get_or_init(|| {
                            paddock_models::dev_var_os!("PADDOCK_NO_F8W8_TMA").is_none()
                                && paddock_models::dev_var_os!("PADDOCK_NO_O16").is_none()
                        });
                        if mo16 && exec.has_f8_o16() {
                            exec.f8_gemm_w8_o16(
                                l8.out_w.as_ref().expect("Linear mixer W8 plane"),
                                0,
                                &sc.d_pxq,
                                &sc.d_exs,
                                &mut sc.d_proj,
                                w.out_w.dims()[0],
                                w.out_w.dims()[1],
                                r,
                            )?;
                            mixer_b16 = true;
                        } else {
                            exec.f8_gemm_w8(
                                l8.out_w.as_ref().expect("Linear mixer W8 plane"),
                                0,
                                &sc.d_pxq,
                                &sc.d_exs,
                                &mut sc.d_proj,
                                w.out_w.dims()[0],
                                w.out_w.dims()[1],
                                r,
                            )?;
                        }
                    } else {
                        prefill_mm_any(
                            &exec,
                            &w.out_w,
                            &mut sc.d_pxq,
                            &mut sc.d_pxs,
                            &mut sc.d_yq,
                            &mut sc.d_xsums,
                            &mut sc.d_ssums,
                            &mut sc.d_skfix,
                            &sc.d_core,
                            &mut sc.d_proj,
                            r,
                        )?;
                    }
                }
            }
            // residual add + post_norm + gate/up quantize in one pass (P6k);
            // xn skipped - the ffn quantize is its only consumer here
            let mut proj_is_b16 = false;
            match &layer.ffn {
                Ffn::Dense { gate, up, down } => {
                    // prefill-FFN f8 arm: the W8 prefill class extended to
                    // the FFN, because ~70% of prefill bytes were still
                    // running through int8-mmq. f8_gemm_w8 measures
                    // 1.27-1.85x best-q8 at M >= 512.
                    // Same e4m3 planes the decode lane built; same w8_min gate.
                    let f8f = bs_f8ffn_p.get(li).and_then(|o| o.as_ref()).filter(|_| {
                        r > super::f8_ffn_pf_min()
                            && paddock_models::dev_var_os!("PADDOCK_F8_ROWSCALE").is_none()
                    });
                    let f8r = bs_f8row_p.get(li).and_then(|o| o.as_ref());
                    // f8 arm: one fused add+norm+e4m3 (xn still written for
                    // alpha/beta) replaces the mmq-quant helper (whose yq is
                    // dead here) + the standalone e4m3 re-read of xn - two
                    // full r x embd passes and a launch saved per layer-tick.
                    let norm_fused = f8f.is_some()
                        && if mixer_b16 {
                            exec.has_add_rmsnorm_e4m3_xn_b16()
                        } else {
                            exec.has_add_rmsnorm_e4m3_xn()
                        };
                    if norm_fused {
                        if mixer_b16 {
                            exec.add_rmsnorm_e4m3_xn_b16(
                                &mut sc.d_x,
                                &sc.d_proj,
                                &layer.post_norm.buf,
                                &mut sc.d_xn,
                                &mut sc.d_pxq,
                                &mut sc.d_exs,
                                embd,
                                r,
                                eps,
                            )?;
                        } else {
                            exec.add_rmsnorm_e4m3_xn(
                                &mut sc.d_x,
                                Some(&sc.d_proj),
                                &layer.post_norm.buf,
                                &mut sc.d_xn,
                                &mut sc.d_pxq,
                                &mut sc.d_exs,
                                embd,
                                r,
                                eps,
                            )?;
                        }
                    } else {
                        prefill_add_norm_quant(
                            &exec,
                            &mut sc.d_x,
                            Some(&sc.d_proj),
                            mixer_b16,
                            &layer.post_norm.buf,
                            &mut sc.d_xn,
                            f8r.is_some() || f8f.is_some(),
                            &mut sc.d_pxq,
                            &mut sc.d_pxs,
                            &mut sc.d_yq,
                            embd,
                            r,
                            eps,
                        )?;
                    }
                    if let Some(p) = f8r {
                        super::ops::ffn_f8row_rows(
                            &exec,
                            p,
                            &sc.d_xn,
                            &mut sc.d_f8t_q,
                            &mut sc.d_f8t_rs,
                            &mut sc.d_ffn_gate,
                            &mut sc.d_ffn_up,
                            &mut sc.d_proj,
                            r,
                        )?;
                    } else if let Some([gu8, d8]) = f8f {
                        // fused plane, row-sliced: gate = rows [0,ff), up =
                        // rows [ff,2ff) - byte-identical to the old separate
                        // planes (same repack stream, offset math only)
                        let ffh = gu8.2 / 2;
                        if !norm_fused {
                            exec.quantize_e4m3(&sc.d_xn, &mut sc.d_pxq, &mut sc.d_exs, r * gu8.1)?;
                        }
                        // bf16 epilogue pair when the pack ships it: halves
                        // the gate/up store traffic (the rival's cutlass
                        // writes bf16; ours wrote f32) and the fused quant
                        // reads bf16 - else the f32 chain below.
                        static O16: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                        let o16 = *O16.get_or_init(|| {
                            paddock_models::dev_var_os!("PADDOCK_NO_F8W8_TMA").is_none()
                                && paddock_models::dev_var_os!("PADDOCK_NO_O16").is_none()
                        });
                        if o16 && exec.has_f8_o16() {
                            if exec.has_swiglu_b16_gu() {
                                // One fused gate|up GEMM - bit-exact vs the
                                // sliced pair (see batch.rs); d_ffn_gate's
                                // r*ffh f32 capacity holds r*2ffh bf16 exactly
                                exec.f8_gemm_w8_o16(
                                    &gu8.0,
                                    0,
                                    &sc.d_pxq,
                                    &sc.d_exs,
                                    &mut sc.d_ffn_gate,
                                    gu8.1,
                                    gu8.2,
                                    r,
                                )?;
                                exec.quantize_e4m3_swiglu_b16_gu(
                                    &sc.d_ffn_gate,
                                    &mut sc.d_pxq,
                                    &mut sc.d_exs,
                                    r * d8.1,
                                    ffh,
                                )?;
                            } else {
                                exec.f8_gemm_w8_o16(
                                    &gu8.0,
                                    0,
                                    &sc.d_pxq,
                                    &sc.d_exs,
                                    &mut sc.d_ffn_gate,
                                    gu8.1,
                                    ffh,
                                    r,
                                )?;
                                exec.f8_gemm_w8_o16(
                                    &gu8.0,
                                    ffh,
                                    &sc.d_pxq,
                                    &sc.d_exs,
                                    &mut sc.d_ffn_up,
                                    gu8.1,
                                    ffh,
                                    r,
                                )?;
                                exec.quantize_e4m3_swiglu_b16(
                                    &sc.d_ffn_gate,
                                    &sc.d_ffn_up,
                                    &mut sc.d_pxq,
                                    &mut sc.d_exs,
                                    r * d8.1,
                                )?;
                            }
                        } else {
                            exec.f8_gemm_w8(
                                &gu8.0,
                                0,
                                &sc.d_pxq,
                                &sc.d_exs,
                                &mut sc.d_ffn_gate,
                                gu8.1,
                                ffh,
                                r,
                            )?;
                            exec.f8_gemm_w8(
                                &gu8.0,
                                ffh,
                                &sc.d_pxq,
                                &sc.d_exs,
                                &mut sc.d_ffn_up,
                                gu8.1,
                                ffh,
                                r,
                            )?;
                            // fused swiglu+e4m3-quant: one pass instead of
                            // swiglu-write + quant-read (286 MB/layer-tick of f32
                            // round-trip at r=2048 - the bf16-activations gap
                            // vs the engines that write bf16 epilogues, closed at
                            // the seam that matters)
                            exec.quantize_e4m3_swiglu(
                                &sc.d_ffn_gate,
                                &sc.d_ffn_up,
                                &mut sc.d_pxq,
                                &mut sc.d_exs,
                                r * d8.1,
                            )?;
                        }
                        if o16 && exec.has_f8_o16() && exec.has_add_b16() {
                            // bf16 down out (halves the last 42 MB/layer-tick
                            // f32 store of the FFN) - the tail add reads bf16
                            exec.f8_gemm_w8_o16(
                                &d8.0,
                                0,
                                &sc.d_pxq,
                                &sc.d_exs,
                                &mut sc.d_proj,
                                d8.1,
                                d8.2,
                                r,
                            )?;
                            proj_is_b16 = true;
                        } else {
                            exec.f8_gemm_w8(
                                &d8.0,
                                0,
                                &sc.d_pxq,
                                &sc.d_exs,
                                &mut sc.d_proj,
                                d8.1,
                                d8.2,
                                r,
                            )?;
                        }
                    } else {
                        prefill_mm_pre_any(
                            &exec,
                            gate,
                            &sc.d_pxq,
                            &sc.d_pxs,
                            &sc.d_yq,
                            &mut sc.d_xsums,
                            &mut sc.d_ssums,
                            &mut sc.d_skfix,
                            &mut sc.d_ffn_gate,
                            r,
                        )?;
                        prefill_mm_pre_any(
                            &exec,
                            up,
                            &sc.d_pxq,
                            &sc.d_pxs,
                            &sc.d_yq,
                            &mut sc.d_xsums,
                            &mut sc.d_ssums,
                            &mut sc.d_skfix,
                            &mut sc.d_ffn_up,
                            r,
                        )?;
                        prefill_ffn_down_any(
                            &exec,
                            down,
                            &mut sc.d_pxq,
                            &mut sc.d_pxs,
                            &mut sc.d_yq,
                            &mut sc.d_xsums,
                            &mut sc.d_ssums,
                            &mut sc.d_skfix,
                            &mut sc.d_ffn_gate,
                            &sc.d_ffn_up,
                            &mut sc.d_proj,
                            ff,
                            r,
                        )?;
                    }
                }
                Ffn::Nvf4Dense { gate, up, down } => {
                    // Above the row band the chain takes the same f8w arm the
                    // Dense lane runs, off planes load.rs builds from the
                    // NVFP4 checkpoint's own values. Without it this path
                    // falls to nvf4_ffn, whose W4A16 tcp kernel is a
                    // DECODE-band arm and costs ~7x the wide-prefill chain.
                    let f8f = bs_f8ffn_p.get(li).and_then(|o| o.as_ref()).filter(|_| {
                        r > nvf4_f8w_min_rows(w8_min)
                            && paddock_models::dev_var_os!("PADDOCK_F8_ROWSCALE").is_none()
                    });
                    // xn stays written: alpha/beta read it either way.
                    prefill_add_norm_quant(
                        &exec,
                        &mut sc.d_x,
                        Some(&sc.d_proj),
                        mixer_b16,
                        &layer.post_norm.buf,
                        &mut sc.d_xn,
                        true,
                        &mut sc.d_pxq,
                        &mut sc.d_pxs,
                        &mut sc.d_yq,
                        embd,
                        r,
                        eps,
                    )?;
                    if let Some([gu8, d8]) = f8f {
                        proj_is_b16 = prefill_ffn_f8w(
                            &exec,
                            gu8,
                            d8,
                            &sc.d_xn,
                            &mut sc.d_pxq,
                            &mut sc.d_exs,
                            &mut sc.d_ffn_gate,
                            &mut sc.d_ffn_up,
                            &mut sc.d_proj,
                            r,
                        )?;
                    } else {
                        nvf4_ffn(
                            &exec,
                            gate,
                            up,
                            down,
                            &sc.d_xn,
                            &mut sc.d_pxq,
                            &mut sc.d_nvs,
                            &mut sc.d_nv4part,
                            &mut sc.d_ffn_gate,
                            &mut sc.d_ffn_up,
                            &mut sc.d_proj,
                            ff,
                            r,
                        )?;
                    }
                }
                Ffn::Moe(w) => {
                    // MoE needs the f32 xn (router + shared expert)
                    prefill_add_norm_quant(
                        &exec,
                        &mut sc.d_x,
                        Some(&sc.d_proj),
                        mixer_b16,
                        &layer.post_norm.buf,
                        &mut sc.d_xn,
                        true,
                        &mut sc.d_pxq,
                        &mut sc.d_pxs,
                        &mut sc.d_yq,
                        embd,
                        r,
                        eps,
                    )?;
                    moe_ffn(
                        &exec,
                        w,
                        self.moe.expect("moe dims"),
                        embd,
                        r,
                        true,
                        &sc.d_xn,
                        &mut sc.d_moe_xq,
                        &mut sc.d_moe_xs,
                        &mut sc.d_ssums,
                        &mut sc.d_moe_xs8,
                        &mut sc.d_moe_fs8,
                        &mut sc.d_moe_logits,
                        &sc.d_zero_bias,
                        &mut sc.d_moe_idx,
                        &mut sc.d_moe_w,
                        &mut sc.d_moe_fused,
                        &mut sc.d_moe_fq,
                        &mut sc.d_moe_fs,
                        &mut sc.d_moe_srow,
                        &mut sc.d_moe_sslot,
                        &mut sc.d_moe_bexp,
                        &mut sc.d_moe_part,
                        &mut sc.d_pxq,
                        &mut sc.d_pxs,
                        &mut sc.d_yq,
                        &mut sc.d_skfix,
                        &mut sc.d_ffn_gate,
                        &mut sc.d_ffn_up,
                        &mut sc.d_mixed,
                        &mut sc.d_proj,
                    )?;
                }
            }
            if proj_is_b16 {
                exec.add_b16(&mut sc.d_x, &sc.d_proj, r * embd)?;
            } else {
                exec.add(&mut sc.d_x, &sc.d_proj, r * embd)?;
            }
        }

        exec.rmsnorm_batch(&sc.d_x, &self.out_norm.buf, &mut sc.d_h, embd, eps, r)?;
        exec.copy_region(&sc.d_h, (r - 1) * embd, &mut sc.d_xn, 0, embd)?;
        // f8 head when the floor elects it: this is a ONE-ROW head call on the
        // prefill path, and it was a direct Q8_0 reader - the site the head
        // reclaim's first audit missed (the FFN arms in this file were checked,
        // the head was not; the stub gave an illegal access, loudly).
        if let Some(p) = super::head_f8(self.out_f8.as_ref(), 1) {
            super::head_f8_gemm(
                &exec,
                p,
                &sc.d_xn,
                &mut sc.d_pxq,
                &mut sc.d_exs,
                &mut bs.d_ks_part,
                &mut sc.d_logits,
                1,
            )?;
        } else {
            super::stub_guard(&self.output, "prefix.rs prefill head")?;
            gemv_any(&exec, &self.output, &sc.d_xn, &mut sc.d_logits)?;
        }
        let logits = exec.to_host(&sc.d_logits)?;

        // Phase-A MTP warm hook: keep the slot's draft-head KV/h in lockstep
        // with the serial prefill so serving spec rounds can draft immediately
        // (~1/41 of the chunk's cost; env-gated). Fresh sequences zero the h
        // carry; later chunks chain from the previous chunk's warm. A chunk
        // arriving out of sequence (prefix-cache resume seeded `start` past
        // the warm position) leaves the slot COLD - it serves dense until its
        // next fresh prefill. Placed after the logits readback: the warm pass
        // clobbers sc.d_x/d_xn but reads sc.d_h, which still holds this
        // chunk's post-out_norm rows. spec_warm_wanted = the scheduler's
        // live-count hint (live > spec cap -> warming is pure cost).
        if self.spec_warm_wanted && self.serve_spec_on() && self.batch.is_some() {
            self.ensure_serve_spec()?;
            let embd = self.embd;
            // long-prompt guard: the warm pass is eager (launch-bound at
            // WARM_CHUNK granularity) and measured +30% TTFT on 4096-token
            // pf8 prompts for a decode phase too short to repay it. Prompts
            // past the cap skip warming and serve dense.
            let warm_max: usize = std::env::var("PADDOCK_QWEN35_SPEC_WARM_MAX")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2048);
            let (in_range, chain_ok) = {
                let sb = self.spec_batch.as_ref().expect("spec batch");
                let ir = slot < sb.alloc_batch;
                let within = start + t_len <= warm_max;
                (
                    ir,
                    ir && within && (start == 0 || (sb.mtp_warm[slot] && sb.pos[slot] == start)),
                )
            };
            if paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some() {
                tracing::info!(
                    "[spec-warm-hook] slot={slot} start={start} len={t_len} in_range={in_range} chain_ok={chain_ok}"
                );
            }
            if chain_ok {
                if start == 0 {
                    let zeros = vec![0f32; embd];
                    let sb = self.spec_batch.as_mut().expect("spec batch");
                    let mut v = sb.pending_h.slice_mut(slot * embd..(slot + 1) * embd);
                    exec.stream.memcpy_htod(&zeros, &mut v).map_err(drv)?;
                }
                self.mtp_warm_slot(slot, tokens, start, 0)?;
                let sb = self.spec_batch.as_mut().expect("spec batch");
                sb.pos[slot] = start + t_len;
                sb.mtp_warm[slot] = true;
                sb.mtp_toks[slot].truncate(start);
                sb.mtp_toks[slot].extend_from_slice(tokens);
            } else if in_range {
                self.spec_batch.as_mut().expect("spec batch").mtp_warm[slot] = false;
            }
        }
        Ok(logits)
    }
}
