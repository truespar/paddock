//! Picture prompts in STEPS - the multimodal half of the stall-free lane.
//!
//! A picture prompt used to prefill in one blocking call: its tower passes and
//! every planned prefill pass ran back to back inside one scheduler tick, and
//! every other live stream waited it out. On the 27B (GB10, 2026-09-29) a
//! 3-picture prompt held a decoding session for 13.5 s - three 2048x2048
//! towers at 2.06 s each, then 12,351 prefill rows at ~1.4K rows/s.
//!
//! Text prompts ride the chunked queue; picture rows cannot join it. A
//! picture's rows attend to its own last row (`MmLayout::bound`), so a picture
//! crosses every layer together, and the unified walk carries neither picture
//! rows nor that bound. So the backend runs the prefill itself, one planned
//! UNIT per `Generator::encode_step`, and the scheduler's ticks run between
//! the units; `MmAdmit::Prefilled` then finishes the slot exactly as a blocking
//! prefill would.
//!
//! - A SLOT job (a resumable prompt, a large one, a lone cold one) alternates
//!   two units: the next pass's pictures (one tower call), then the pass
//!   itself (`mm_pass_ends`: at most `step_rows`, never inside a picture).
//! - A GROUP job batches small cold prompts exactly as the blocking wave does -
//!   one batched tower call plus one batched pass per group - so concurrent
//!   image requests still share a pass (the vi8 fix).
//!
//! The blocking entries (`forward_prefill_slot_mm`, `forward_prefill_mm_wave`)
//! run the same units back to back, so the two paths cannot drift apart. A
//! unit's floor is one whole picture; that, not the step budget, is what bounds
//! the stall for a large picture.

use std::collections::VecDeque;

use super::batch::MmShareCtx;
use super::*;
use crate::generator::MmAdmit;
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::prefix_cache::BLOCK_TOKENS;
use crate::service::MmChunk;

/// One prompt's prefill in planned passes, resumable between units.
pub(super) struct MmSlotJob {
    slot: usize,
    lay: MmLayout,
    keys: Vec<u32>,
    img_spans: Vec<(usize, usize)>,
    /// (end row, checkpoint here) for every pass still to run
    passes: VecDeque<(usize, bool)>,
    /// rows [0, pos) are in the slot's KV and recurrent state
    pos: usize,
    /// the current pass's pictures, encoded by the unit before it
    pictures: Vec<Option<super::pictures::Picture>>,
    /// whether the current pass's pictures have been encoded
    encoded: bool,
}

/// Small cold prompts batched into groups, one batched pass per group.
pub(super) struct MmGroupJob {
    /// (slot, placeholder ids) per prompt, `prefill_batch_pass`'s items
    pub(super) items: Vec<(usize, Vec<u32>)>,
    ctxs: Vec<MmShareCtx>,
    /// item indices per group still to run
    pub(super) groups: VecDeque<Vec<usize>>,
}

/// A stepped admission the backend holds between `encode_step` calls, with
/// the prompts it prefills (the chunks are the requests' own, moved in).
pub(super) enum MmStepJob {
    Slot {
        job: MmSlotJob,
        chunks: Vec<MmChunk>,
    },
    Group {
        job: MmGroupJob,
        prompts: Vec<Vec<MmChunk>>,
    },
}

/// Every picture of a prompt as a tower source, in order.
fn picture_sources(chunks: &[MmChunk]) -> Vec<(&[u8], usize, usize)> {
    chunks
        .iter()
        .filter_map(|c| match c {
            MmChunk::Image { rgb, w, h } => Some((rgb.as_slice(), *w, *h)),
            _ => None,
        })
        .collect()
}

fn failed(e: GpuModelError) -> MmAdmit {
    MmAdmit::Failed(crate::generator::GenError::Backend(e.to_string()))
}

impl GpuQwen35 {
    /// Whether picture prompts take the stepped lane: vision attached,
    /// batching on, and the chunked text lane on too - with that pinned off
    /// (PADDOCK_NO_CHUNKED_PREFILL) every prompt takes a blocking pass, and so
    /// do these. PADDOCK_QWEN35_NO_MM_STEPS pins the blocking wave for A/B.
    pub(crate) fn mm_steps_supported(&self) -> bool {
        static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let off = *OFF
            .get_or_init(|| paddock_models::dev_var_os!("PADDOCK_QWEN35_NO_MM_STEPS").is_some());
        !off && self.has_vision() && self.batch.is_some() && self.supports_chunked_prefill()
    }

    /// Rows a stepped unit may prefill: 2048 - about 1.5 s on the 27B at the
    /// ~1.4K rows/s a picture prompt prefills at on GB10, which sits beside
    /// the paced text band - while staying past the ~1500 rows a pass needs to
    /// amortize its fixed cost (see `service::mixed_tick_budget`). Never past
    /// the planned pass. PADDOCK_QWEN35_MM_STEP_ROWS overrides it (development).
    fn mm_step_rows(&self) -> usize {
        static ROWS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        let rows = *ROWS.get_or_init(|| {
            paddock_models::dev_var!("PADDOCK_QWEN35_MM_STEP_ROWS")
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|&n: &usize| n >= 64)
                .unwrap_or(2048)
        });
        rows.min(self.prefill_chunk_rows)
    }

    /// Begin one prompt's pass plan on `slot`: the layout, the prefix resume
    /// (which adopts the cached pages and restores the recurrent state), the
    /// slot's blocks, and the passes - at most `pass_rows` apart, never inside
    /// a picture, ending on the checkpoint cuts. Nothing is prefilled yet.
    pub(super) fn mm_slot_begin(
        &mut self,
        slot: usize,
        chunks: &[MmChunk],
        pass_rows: usize,
    ) -> Result<MmSlotJob, GpuModelError> {
        assert!(self.batch.is_some(), "enable_batch first");
        assert!(slot < self.batch.as_ref().expect("batch").max_batch);
        // token ids (image spans are `0` placeholders), the mRoPE grid, and the
        // per-row causal bound - one ordered walk, any number of images
        let grids = self.picture_grids(chunks)?;
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

        // Prefill [start, t_len) in passes that end at the checkpoint cuts, so
        // the DeltaNet state can be snapshotted at each boundary before the
        // following rows advance it - the paged text tail's shape exactly -
        // and at most `pass_rows` apart, so no pass outgrows the serving
        // scratch (`mm_pass_ends`).
        let cuts = self.mm_ckpt_cuts(t_len, start, &img_spans);
        let passes = crate::gpu_model::prefix_cache::mm_pass_ends(
            start, t_len, &cuts, &img_spans, pass_rows,
        );
        Ok(MmSlotJob {
            slot,
            pictures: vec![None; img_spans.len()],
            lay,
            keys,
            img_spans,
            passes: passes.into_iter().collect(),
            pos: start,
            encoded: false,
        })
    }

    /// One unit of a slot job: the next pass's pictures (one tower call) when
    /// they are not encoded yet, else the pass itself. Returns the last row's
    /// logits and the prompt's row count once the final pass has run.
    pub(super) fn mm_slot_unit(
        &mut self,
        job: &mut MmSlotJob,
        chunks: &[MmChunk],
    ) -> Result<Option<(Vec<f32>, usize)>, GpuModelError> {
        let (end, checkpoint) = *job
            .passes
            .front()
            .expect("a slot job with no pass left has already finished");
        if !job.encoded {
            job.encoded = true;
            // exactly the pictures this pass splices (the cut rule keeps each
            // one whole inside a single pass), released when it ends
            let need: Vec<usize> = (0..job.img_spans.len())
                .filter(|&k| job.img_spans[k].0 >= job.pos && job.img_spans[k].1 <= end)
                .collect();
            if !need.is_empty() {
                let sources = picture_sources(chunks);
                let got =
                    self.encode_pictures(&need.iter().map(|&k| sources[k]).collect::<Vec<_>>())?;
                for (k, p) in need.into_iter().zip(got) {
                    job.pictures[k] = Some(p);
                }
                return Ok(None);
            }
        }
        let logits = self.mm_prefill_span(job.slot, &job.lay, &job.pictures, job.pos, end)?;
        if checkpoint {
            self.mm_prefix_publish(job.slot, &job.keys, end, true)?;
        }
        job.pictures.iter_mut().for_each(|p| *p = None);
        job.pos = end;
        job.passes.pop_front();
        job.encoded = false;
        if !job.passes.is_empty() {
            return Ok(None);
        }
        let t_len = job.lay.t_len;
        // cache every full page of this prompt (idempotent for those inserted
        // at a cut above) so a longer continuation resumes past the last one
        self.mm_prefix_publish(
            job.slot,
            &job.keys,
            t_len / BLOCK_TOKENS * BLOCK_TOKENS,
            false,
        )?;
        self.batch.as_mut().expect("batch").mrope_delta[job.slot] =
            job.lay.final_mrope_pos as i64 - t_len as i64;
        Ok(Some((logits, t_len)))
    }

    /// How a wave routes one request: `None` = alone in planned passes (a
    /// resumable prompt, or one longer than `cap`), `Some(grids)` = a cold
    /// prompt small enough to batch.
    pub(super) fn mm_route(
        &mut self,
        slot: usize,
        chunks: &[MmChunk],
        cap: usize,
    ) -> Result<Option<Vec<(usize, usize)>>, GpuModelError> {
        let max_batch = self
            .batch
            .as_ref()
            .ok_or(GpuModelError::BatchDisabled)?
            .max_batch;
        if slot >= max_batch {
            return Err(GpuModelError::BatchTooLarge {
                got: slot + 1,
                max: max_batch,
            });
        }
        let grids = self.picture_grids(chunks)?;
        // The batched pass is fresh-only - a request with a cached prefix would
        // re-prefill a picture the radix already holds, which is the reuse the
        // content-keyed cache exists for (a document conversation is a hit by
        // construction). And it takes each prompt whole, so one longer than
        // the cap would outgrow what the pass may hold.
        if self.mm_prefix_would_resume(chunks, &grids)? || mm_rows(chunks, &grids) > cap {
            Ok(None)
        } else {
            Ok(Some(grids))
        }
    }

    /// Batch a wave's cold small prompts: their layouts, then groups of at
    /// most `cap` rows (an oversized single prompt is its own group).
    pub(super) fn mm_group_plan(
        &self,
        cold: Vec<(Vec<MmChunk>, usize, Vec<(usize, usize)>)>,
        cap: usize,
    ) -> Result<(MmGroupJob, Vec<Vec<MmChunk>>), GpuModelError> {
        let mut items: Vec<(usize, Vec<u32>)> = Vec::with_capacity(cold.len());
        let mut ctxs: Vec<MmShareCtx> = Vec::with_capacity(cold.len());
        let mut prompts: Vec<Vec<MmChunk>> = Vec::with_capacity(cold.len());
        for (chunks, slot, grids) in cold {
            let MmLayout {
                ids,
                mrope,
                bound,
                splices,
                t_len,
                final_mrope_pos,
            } = build_mm_layout(&chunks, &grids)?;
            if t_len == 0 || t_len > self.max_ctx {
                return Err(GpuModelError::BatchTooLarge {
                    got: t_len,
                    max: self.max_ctx,
                });
            }
            items.push((slot, ids));
            ctxs.push(MmShareCtx {
                mrope,
                bound,
                splices,
                images: Vec::new(),
                final_mrope_pos,
            });
            prompts.push(chunks);
        }
        let mut groups: VecDeque<Vec<usize>> = VecDeque::new();
        let mut group: Vec<usize> = Vec::new();
        let mut rows = 0usize;
        for (i, item) in items.iter().enumerate() {
            let tl = item.1.len();
            if rows + tl > cap && !group.is_empty() {
                groups.push_back(std::mem::take(&mut group));
                rows = 0;
            }
            // an oversized single request just runs as its own pass (same
            // kernels as the solo path; no serial special case needed)
            group.push(i);
            rows += tl;
        }
        if !group.is_empty() {
            groups.push_back(group);
        }
        Ok((
            MmGroupJob {
                items,
                ctxs,
                groups,
            },
            prompts,
        ))
    }

    /// One unit of a group job: the next group's pictures in one batched tower
    /// call, borrowed for this pass only, then its batched pass. Returns
    /// (item index, last-row logits) for every prompt of the group.
    pub(super) fn mm_group_unit(
        &mut self,
        job: &mut MmGroupJob,
        prompts: &[Vec<MmChunk>],
    ) -> Result<Vec<(usize, Vec<f32>)>, GpuModelError> {
        // left queued until its pass succeeds: a failure fails exactly the
        // groups not reported yet, this one included
        let group = job
            .groups
            .front()
            .expect("a group job with no group left has already finished")
            .clone();
        let want: Vec<(&[u8], usize, usize)> = group
            .iter()
            .flat_map(|&j| picture_sources(&prompts[j]))
            .collect();
        let mut got = self.encode_pictures(&want)?.into_iter();
        for &j in &group {
            let n = job.ctxs[j].splices.len();
            job.ctxs[j].images = got.by_ref().take(n).collect();
        }
        let mut lout: Vec<Vec<f32>> = vec![Vec::new(); job.items.len()];
        let r = self.prefill_batch_pass(&job.items, &group, &mut lout, Some(&job.ctxs));
        for &j in &group {
            job.ctxs[j].images.clear();
        }
        r?;
        job.groups.pop_front();
        Ok(group
            .iter()
            .map(|&j| (j, std::mem::take(&mut lout[j])))
            .collect())
    }

    /// `Generator::set_page_reader`.
    pub(crate) fn set_page_reader(&mut self, on: bool) {
        self.page_reader = on;
    }

    /// `Generator::prefill_begin_multimodal`: plan the wave exactly as the
    /// blocking one routes it, with the step budget as the cap, and hold every
    /// planned slot until its units have run. A request that cannot be planned
    /// fails alone.
    ///
    /// On a page reader a wave of several pages groups up to the elected
    /// prefill chunk instead: a page (~2.9K rows at LightOnOCR-3's 2048-px
    /// edge) is past the step budget, so each went alone, two units and two
    /// decode ticks apiece. Grouped two to a unit, 64 pages in flight on GB10
    /// reached their first token 18-19% sooner (p50 39.8 -> 32.6 s, p95 72.9 ->
    /// 59.0 s). The group pass publishes nothing to the prefix cache, which is
    /// why only an endpoint whose pictures are never continued groups this wide.
    pub(crate) fn mm_steps_begin(
        &mut self,
        reqs: Vec<(usize, Vec<MmChunk>)>,
    ) -> Vec<(usize, MmAdmit)> {
        let step = if self.page_reader && reqs.len() > 1 {
            self.mm_step_rows().max(self.prefill_chunk_rows)
        } else {
            self.mm_step_rows()
        };
        let mut out: Vec<(usize, MmAdmit)> = Vec::with_capacity(reqs.len());
        let mut cold: Vec<(Vec<MmChunk>, usize, Vec<(usize, usize)>)> = Vec::new();
        let mut alone: Vec<(usize, Vec<MmChunk>)> = Vec::new();
        for (slot, chunks) in reqs {
            match self.mm_route(slot, &chunks, step) {
                Err(e) => out.push((slot, failed(e))),
                Ok(None) => alone.push((slot, chunks)),
                Ok(Some(grids)) => cold.push((chunks, slot, grids)),
            }
        }
        // a cohort of one is a slot job, and it also gets to publish its pages
        if cold.len() == 1 {
            let (chunks, slot, _) = cold.pop().expect("len 1");
            alone.push((slot, chunks));
        }
        for (slot, chunks) in alone {
            match self.mm_slot_begin(slot, &chunks, step) {
                Ok(job) => {
                    self.mm_steps.push_back(MmStepJob::Slot { job, chunks });
                    out.push((slot, MmAdmit::Encoding));
                }
                Err(e) => out.push((slot, failed(e))),
            }
        }
        if !cold.is_empty() {
            let slots: Vec<usize> = cold.iter().map(|c| c.1).collect();
            match self.mm_group_plan(cold, step) {
                Ok((job, prompts)) => {
                    self.mm_steps.push_back(MmStepJob::Group { job, prompts });
                    out.extend(slots.into_iter().map(|k| (k, MmAdmit::Encoding)));
                }
                Err(e) => {
                    let msg = e.to_string();
                    out.extend(slots.into_iter().map(|k| {
                        (
                            k,
                            MmAdmit::Failed(crate::generator::GenError::Backend(msg.clone())),
                        )
                    }));
                }
            }
        }
        out
    }

    /// `Generator::encode_step`: run ONE unit of the oldest held job and
    /// report the slots it finished. A failed unit fails every slot its job
    /// still held.
    pub(crate) fn mm_steps_run(&mut self) -> Vec<(usize, MmAdmit)> {
        let Some(mut held) = self.mm_steps.pop_front() else {
            return Vec::new();
        };
        let (out, more) = match &mut held {
            MmStepJob::Slot { job, chunks } => match self.mm_slot_unit(job, chunks) {
                Ok(None) => (Vec::new(), true),
                Ok(Some((logits, rows))) => {
                    (vec![(job.slot, MmAdmit::Prefilled { logits, rows })], false)
                }
                Err(e) => (vec![(job.slot, failed(e))], false),
            },
            MmStepJob::Group { job, prompts } => match self.mm_group_unit(job, prompts) {
                Ok(done) => {
                    let out = done
                        .into_iter()
                        .map(|(j, logits)| {
                            let (slot, ids) = &job.items[j];
                            (
                                *slot,
                                MmAdmit::Prefilled {
                                    logits,
                                    rows: ids.len(),
                                },
                            )
                        })
                        .collect();
                    (out, !job.groups.is_empty())
                }
                Err(e) => {
                    // the failed group is still queued: fail it and every
                    // group after it, never one already reported
                    let msg = e.to_string();
                    let out = job
                        .groups
                        .iter()
                        .flatten()
                        .map(|&j| {
                            (
                                job.items[j].0,
                                MmAdmit::Failed(crate::generator::GenError::Backend(msg.clone())),
                            )
                        })
                        .collect();
                    (out, false)
                }
            },
        };
        if more {
            self.mm_steps.push_front(held);
        }
        out
    }

    /// `Generator::encoding_pending`.
    pub(crate) fn mm_steps_pending(&self) -> bool {
        !self.mm_steps.is_empty()
    }

    /// Drop a held slot job whose client hung up. A member of a GROUP job is
    /// refused (false) until its group has run: the group is one planned
    /// batched pass, and dropping one prompt out of it would leave the pass
    /// writing a slot the scheduler may already have handed to a new request.
    /// The scheduler retries next tick; the finished slot is then reported and
    /// released like any other whose client is gone.
    pub(crate) fn mm_steps_abort(&mut self, slot: usize) -> bool {
        let at = self
            .mm_steps
            .iter()
            .position(|h| matches!(h, MmStepJob::Slot { job, .. } if job.slot == slot));
        match at {
            Some(i) => {
                self.mm_steps.remove(i);
                true
            }
            None => false,
        }
    }
}
