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
//!   itself (`mm_pass_ends`: at most `step_rows`, never inside a picture). It
//!   plans when it reaches the front, so it resumes from whatever the jobs
//!   ahead of it published.
//! - A GROUP job batches small cold prompts exactly as the blocking wave does -
//!   one batched tower call plus one batched pass per group - so concurrent
//!   image requests still share a pass (the vi8 fix).
//!
//! The blocking entries (`forward_prefill_slot_mm`, `forward_prefill_mm_wave`)
//! run the same units back to back, so the two paths cannot drift apart. A
//! unit's floor is one whole picture; that, not the step budget, is what bounds
//! the stall for a large picture.

use std::collections::VecDeque;

use super::mm_ride::MmShareCtx;
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

impl MmSlotJob {
    pub(super) fn slot(&self) -> usize {
        self.slot
    }

    /// Take what the unified queue needs (`mm_ride_enqueue`): the layout, the
    /// radix keys, the encoded pictures, the resume row and the planned
    /// passes. The job is spent afterwards.
    #[allow(clippy::type_complexity)]
    pub(super) fn take_ride_parts(
        &mut self,
    ) -> (
        MmLayout,
        Vec<u32>,
        Vec<Option<super::pictures::Picture>>,
        usize,
        Vec<(usize, bool)>,
    ) {
        let lay = MmLayout {
            ids: std::mem::take(&mut self.lay.ids),
            mrope: std::mem::take(&mut self.lay.mrope),
            bound: std::mem::take(&mut self.lay.bound),
            splices: std::mem::take(&mut self.lay.splices),
            t_len: self.lay.t_len,
            final_mrope_pos: self.lay.final_mrope_pos,
        };
        (
            lay,
            std::mem::take(&mut self.keys),
            std::mem::take(&mut self.pictures),
            self.pos,
            std::mem::take(&mut self.passes).into_iter().collect(),
        )
    }
}

/// What one unit of a slot job produced.
pub(super) enum MmUnit {
    /// more units to run
    More,
    /// the prompt is prefilled: its last row's logits and its row count
    Prefilled(Vec<f32>, usize),
    /// the prompt was handed to the unified queue (`mm_ride`); it finishes as
    /// a unified finisher, and this job is spent
    Riding,
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
    /// A slot job not planned yet: it plans (prefix match, resume, passes)
    /// when it reaches the front. Jobs run in order, so every job ahead of it
    /// has published by then, and a prompt sharing their text head resumes
    /// there instead of prefilling it again - pages of one document pipeline
    /// arriving a tick apart, or a burst's followers (`shared_head_split`).
    Deferred {
        slot: usize,
        chunks: Vec<MmChunk>,
        step: usize,
    },
}

/// A prompt's text before its first picture, as the radix keys it (a picture
/// row is keyed by the picture, so nothing past it can be shared).
fn text_head(chunks: &[MmChunk], grids: &[(usize, usize)]) -> Vec<u32> {
    build_mm_layout(chunks, grids).map_or_else(
        |_| Vec::new(),
        |lay| {
            let end = lay.splices.first().map_or(lay.t_len, |&(off, _)| off);
            lay.ids[..end].to_vec()
        },
    )
}

/// A cold burst's leader and the prompts that can resume from it: the first
/// prompt whose head is long enough to checkpoint (`first_picture_cut`), and
/// every other one that shares that whole head. A group pass publishes
/// nothing, so a burst of pages behind the same static instructions used to
/// leave the cache as cold as it found it. `None` when no two prompts share
/// a checkpointable head.
fn shared_head_split(heads: &[Vec<u32>], min: usize) -> Option<(usize, Vec<usize>)> {
    heads.iter().enumerate().find_map(|(lead, head)| {
        let cut =
            crate::gpu_model::prefix_cache::first_picture_cut(0, &[(head.len(), head.len())], min)?;
        let followers: Vec<usize> = (0..heads.len())
            .filter(|&i| i != lead && heads[i].len() >= cut && heads[i][..cut] == head[..cut])
            .collect();
        (!followers.is_empty()).then_some((lead, followers))
    })
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

/// Riding slot jobs carried onto the unified queue per encode step
/// (`mm_steps_run`). PADDOCK_MM_RIDE_JOBS overrides; 0 restores one unit a
/// tick.
fn mm_ride_jobs() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_MM_RIDE_JOBS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4)
    })
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
        ride: bool,
    ) -> Result<MmUnit, GpuModelError> {
        let (end, checkpoint) = *job
            .passes
            .front()
            .expect("a slot job with no pass left has already finished");
        // riding the unified tick (`ride`, the stepped lane's choice; the
        // blocking entry never rides), the whole remaining prompt queues at
        // once, so this unit encodes every picture from the resume point on
        if !job.encoded {
            job.encoded = true;
            // exactly the pictures this pass splices (the cut rule keeps each
            // one whole inside a single pass), released when it ends
            let need: Vec<usize> = (0..job.img_spans.len())
                .filter(|&k| job.img_spans[k].0 >= job.pos && (ride || job.img_spans[k].1 <= end))
                .collect();
            if !need.is_empty() {
                let sources = picture_sources(chunks);
                let got =
                    self.encode_pictures(&need.iter().map(|&k| sources[k]).collect::<Vec<_>>())?;
                for (k, p) in need.into_iter().zip(got) {
                    job.pictures[k] = Some(p);
                }
                return Ok(MmUnit::More);
            }
        }
        if ride {
            self.mm_ride_enqueue(job)?;
            return Ok(MmUnit::Riding);
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
            return Ok(MmUnit::More);
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
        Ok(MmUnit::Prefilled(logits, t_len))
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
        // A burst sharing a text head before its pictures (a document
        // pipeline's static instructions): the leader runs alone and
        // checkpoints the head, the followers wait for it and resume there,
        // and only the rest are grouped.
        let mut deferred: Vec<(usize, Vec<MmChunk>)> = Vec::new();
        if cold.len() > 1 {
            let heads: Vec<Vec<u32>> = cold.iter().map(|(c, _, g)| text_head(c, g)).collect();
            if let Some((lead, followers)) = shared_head_split(&heads, super::min_cache_prefix()) {
                let mut rest = Vec::with_capacity(cold.len());
                for (i, (chunks, slot, grids)) in cold.into_iter().enumerate() {
                    if i == lead {
                        alone.push((slot, chunks));
                    } else if followers.contains(&i) {
                        deferred.push((slot, chunks));
                    } else {
                        rest.push((chunks, slot, grids));
                    }
                }
                cold = rest;
            }
        }
        // a cohort of one is a slot job, and it also gets to publish its pages
        if cold.len() == 1 {
            let (chunks, slot, _) = cold.pop().expect("len 1");
            alone.push((slot, chunks));
        }
        // slot jobs plan at the front (see `MmStepJob::Deferred`), the burst's
        // followers right behind their leader
        for (slot, chunks) in alone.into_iter().chain(deferred) {
            self.mm_steps
                .push_back(MmStepJob::Deferred { slot, chunks, step });
            out.push((slot, MmAdmit::Encoding));
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

    /// `Generator::encode_step`. Riding the unified tick, the front slot jobs
    /// go all the way onto the unified queue here - plan, encode, enqueue -
    /// up to `mm_ride_jobs()` of them; otherwise ONE unit of the oldest job.
    ///
    /// One unit per tick made a riding page cost three ticks (plan, tower,
    /// enqueue) and capped picture admissions at a third of the tick rate:
    /// 3.8 pages/s at ~11.5 ticks/s on the B200 recover load, with ~40 of 128
    /// slots waiting here and the decode lane idle around them. A riding
    /// job's pass work runs later on the unified queue, so the stall this
    /// pacing bounds is just the tower launch (one picture per tower pass).
    /// A burst follower still waits for its leader: `ride_head_pending` sees
    /// the leader in the chunk queue as soon as it is enqueued, so the
    /// follower stays Deferred and ends the run. A group at the front keeps
    /// its own pacing - one planned batched pass a step.
    pub(crate) fn mm_steps_run(&mut self) -> Vec<(usize, MmAdmit)> {
        if mm_ride_jobs() == 0 || !self.mm_ride_on() {
            return self.mm_steps_unit();
        }
        let out = self.mm_steps_ride();
        if out.is_empty() && matches!(self.mm_steps.front(), Some(MmStepJob::Group { .. })) {
            return self.mm_steps_unit();
        }
        out
    }

    /// `Generator::encode_step_overlap`: only the riding half of
    /// `mm_steps_run` - never a group's blocking pass - so the scheduler can
    /// run it beside an in-flight span, in the host time a queued decode tick
    /// leaves (the loop-top step idled the decode lane ~20 ms a call).
    pub(crate) fn mm_steps_run_overlap(&mut self) -> Vec<(usize, MmAdmit)> {
        if mm_ride_jobs() == 0 || !self.mm_ride_on() {
            return Vec::new();
        }
        self.mm_steps_ride()
    }

    /// Carry up to `mm_ride_jobs()` riding slot jobs at the front onto the
    /// unified queue; stops at a group, an empty queue, or a follower still
    /// waiting on its leader.
    fn mm_steps_ride(&mut self) -> Vec<(usize, MmAdmit)> {
        let jobs = mm_ride_jobs();
        let mut out = Vec::new();
        let mut done = 0;
        // a slot job takes at most three units (plan, encode, enqueue)
        for _ in 0..3 * jobs {
            let stalled_at = match self.mm_steps.front() {
                Some(MmStepJob::Deferred { slot, .. }) => Some(*slot),
                Some(MmStepJob::Slot { .. }) => None,
                // a group is one planned batched pass a unit: its own pacing
                Some(MmStepJob::Group { .. }) | None => break,
            };
            let before = self.mm_steps.len();
            out.extend(self.mm_steps_unit());
            if self.mm_steps.len() < before {
                done += 1;
                if done >= jobs {
                    break;
                }
            } else if stalled_at.is_some()
                && matches!(self.mm_steps.front(),
                    Some(MmStepJob::Deferred { slot, .. }) if Some(*slot) == stalled_at)
            {
                break; // waiting on a leader's head checkpoint
            }
        }
        out
    }

    /// One unit of the oldest held job; reports the slots it finished. A
    /// failed unit fails every slot its job still held.
    fn mm_steps_unit(&mut self) -> Vec<(usize, MmAdmit)> {
        let Some(held) = self.mm_steps.pop_front() else {
            return Vec::new();
        };
        // a slot job plans now that every job ahead of it has published
        let mut held = match held {
            MmStepJob::Deferred { slot, chunks, step } => {
                if self.ride_head_pending(&chunks) {
                    self.mm_steps
                        .push_front(MmStepJob::Deferred { slot, chunks, step });
                    return Vec::new();
                }
                return match self.mm_slot_begin(slot, &chunks, step) {
                    Ok(job) => {
                        self.mm_steps.push_front(MmStepJob::Slot { job, chunks });
                        Vec::new()
                    }
                    Err(e) => vec![(slot, failed(e))],
                };
            }
            other => other,
        };
        let (out, more) = match &mut held {
            MmStepJob::Deferred { .. } => unreachable!("deferred jobs are planned above"),
            MmStepJob::Slot { job, chunks } => {
                match self.mm_slot_unit(job, chunks, self.mm_ride_on()) {
                    Ok(MmUnit::More) => (Vec::new(), true),
                    Ok(MmUnit::Prefilled(logits, rows)) => {
                        (vec![(job.slot, MmAdmit::Prefilled { logits, rows })], false)
                    }
                    // queued on the unified tick: the scheduler marks the slot
                    // chunking and finishes it from the fused tick's finishers
                    Ok(MmUnit::Riding) => (vec![(job.slot, MmAdmit::Queued)], false),
                    Err(e) => (vec![(job.slot, failed(e))], false),
                }
            }
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

    /// Riding the unified tick, the job ahead only QUEUED its prefill, so a
    /// job reaching the front would plan before that prompt published
    /// anything - a burst's second page planned cold beside its leader. Wait
    /// (one encode step per tick, so the leader's spans run meanwhile) while
    /// a riding prompt that shares this one's text head has not yet landed
    /// the checkpoint before its first picture.
    fn ride_head_pending(&self, chunks: &[MmChunk]) -> bool {
        let Ok(grids) = self.picture_grids(chunks) else {
            return false;
        };
        let head = text_head(chunks, &grids);
        self.chunked.iter().any(|c| {
            let Some(mm) = c.mm.as_ref() else {
                return false;
            };
            let first = mm.splices.first().map_or(mm.t_len, |sp| sp.0);
            let Some(cut) = mm.cuts.iter().copied().filter(|&x| x <= first).max() else {
                return false;
            };
            c.done < cut && head.len() >= cut && head[..cut] == c.tokens[..cut]
        })
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
        let at = self.mm_steps.iter().position(|h| match h {
            MmStepJob::Slot { job, .. } => job.slot == slot,
            MmStepJob::Deferred { slot: s, .. } => *s == slot,
            MmStepJob::Group { .. } => false,
        });
        match at {
            Some(i) => {
                self.mm_steps.remove(i);
                true
            }
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::shared_head_split;

    fn head(fill: u32, n: usize) -> Vec<u32> {
        vec![fill; n]
    }

    #[test]
    fn a_burst_behind_the_same_instructions_gets_a_leader_and_followers() {
        // three pages behind 3,300 tokens of the same static text, one other
        let mut other = head(7, 3300);
        other[100] = 9;
        let heads = vec![head(7, 3300), head(7, 3300), other, head(7, 3310)];
        let (lead, followers) = shared_head_split(&heads, 512).expect("a shared head");
        assert_eq!(lead, 0);
        // the leader checkpoints at 3296; the 3310-token head shares all of it
        assert_eq!(followers, vec![1, 3]);
    }

    #[test]
    fn short_or_unshared_heads_stay_grouped() {
        assert_eq!(shared_head_split(&[head(7, 200), head(7, 200)], 512), None);
        let mut b = head(7, 3300);
        b[10] = 8;
        assert_eq!(shared_head_split(&[head(7, 3300), b], 512), None);
        assert_eq!(shared_head_split(&[head(7, 3300)], 512), None);
    }
}
