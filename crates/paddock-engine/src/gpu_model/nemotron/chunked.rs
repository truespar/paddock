//! Prefill through the batch lane: whole prompts and chunks, the chunk
//! planner and its commits, the checkpoint stages a pass breaks at, and the
//! mixed tick (a decode band fused on top of a prefill chunk). Split out of
//! `batch.rs` (which had crossed the 2,500-line ceiling).

use crate::gpu_model::gpt_oss::GpuModelError;

use super::batch::*;
use super::*;

impl GpuNemotron {
    /// The checkpoint plan for rows covering positions `[base, base+len)` of
    /// a prompt of `t_len` rows resumed at `start`: which pass rows (offset
    /// by `row0`, the rows' base index within the whole pass) end at a
    /// `ckpt_cuts` boundary. Returns (pass-row breaks for PfCuts, and the
    /// (stage, cut) commits to run after the pass). `stage0` threads the
    /// stage counter across multiple prompts sharing one pass.
    pub(super) fn stage_plan(
        t_len: usize,
        start: usize,
        base: usize,
        len: usize,
        row0: usize,
        stage0: &mut usize,
        step: usize,
        max_stages: usize,
    ) -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
        let mut breaks = Vec::new();
        let mut after = Vec::new();
        for cut in super::prefix::ckpt_cuts(t_len, step) {
            if cut > start.max(base) && cut <= base + len && *stage0 < max_stages {
                breaks.push((row0 + (cut - base), *stage0));
                after.push((*stage0, cut));
                *stage0 += 1;
            }
        }
        (breaks, after)
    }

    /// Prefill a whole prompt into `slot` (chunked at `prefill_chunk`) and
    /// return the last token's logits. Trailing-boundary checkpoints stage
    /// during the passes and commit between chunks (stage D).
    pub(crate) fn forward_prefill_impl(
        &mut self,
        slot: usize,
        tokens: &[u32],
    ) -> Result<Vec<f32>, GpuModelError> {
        self.admit_rows(slot, tokens.len())?;
        let start = self.prefix_resume_rows(slot, tokens, tokens.len())?;
        self.reply_track_admit(slot, tokens);
        let mut base = start;
        let mut last_len = 0usize;
        for chunk in tokens[base..].chunks(self.prefill_chunk) {
            let rows: Vec<(u32, u32, u32)> = chunk
                .iter()
                .enumerate()
                .map(|(j, &t)| (slot as u32, (base + j) as u32, t))
                .collect();
            let mut stage = 0usize;
            let (breaks, after) = if !self.has_ckpt_stages() {
                (Vec::new(), Vec::new())
            } else {
                Self::stage_plan(
                    tokens.len(),
                    start,
                    base,
                    chunk.len(),
                    0,
                    &mut stage,
                    self.tier_ckpt_step(),
                    self.ckpt_stage_count(),
                )
            };
            self.rows_pass_body(&rows, 0, breaks)?;
            for (st, cut) in after {
                self.commit_stage(st, slot, tokens, cut);
            }
            base += chunk.len();
            last_len = chunk.len();
        }
        self.prefix_insert(slot, tokens);
        self.head_row(last_len - 1)
    }

    /// COALESCED multi-prompt prefill: every pending prompt's rows
    /// concatenate into shared chunks - one weight-amortized pass over the
    /// wave (granite's shape; run isolation keeps attention AND the
    /// recurrent advance per slot).
    pub(crate) fn forward_prefill_batch_impl(
        &mut self,
        items: &[(usize, Vec<u32>)],
    ) -> Result<Vec<Vec<f32>>, GpuModelError> {
        if items.len() == 1 || paddock_models::dev_var_os!("PADDOCK_NO_COALESCED_PREFILL").is_some()
        {
            return items
                .iter()
                .map(|(slot, toks)| self.forward_prefill_impl(*slot, toks))
                .collect();
        }
        let mut starts = vec![0usize; items.len()];
        for (it, (slot, tokens)) in items.iter().enumerate() {
            self.admit_rows(*slot, tokens.len())?;
            starts[it] = self.prefix_resume_rows(*slot, tokens, tokens.len())?;
            self.reply_track_admit(*slot, tokens);
        }
        let mut rows: Vec<(u32, u32, u32)> = Vec::new();
        let mut last_row = vec![0usize; items.len()];
        for (it, (slot, toks)) in items.iter().enumerate() {
            for (j, &t) in toks.iter().enumerate().skip(starts[it]) {
                rows.push((*slot as u32, j as u32, t));
            }
            last_row[it] = rows.len() - 1;
        }
        // per-item global row base within the wave stream, for the cut plan
        let mut item_base = vec![0usize; items.len()];
        {
            let mut acc = 0usize;
            for (it, (_, toks)) in items.iter().enumerate() {
                item_base[it] = acc;
                acc += toks.len() - starts[it];
            }
        }
        let mut out: Vec<Vec<f32>> = vec![Vec::new(); items.len()];
        let step = self.tier_ckpt_step();
        let mut base = 0usize;
        for chunk in rows.chunks(self.prefill_chunk) {
            let r = chunk.len();
            // finishers whose last row landed in this chunk read inside the
            // pass - the next chunk's embed overwrites d_x. Ascending by row
            // because head_row bounces its row through x[0].
            let mut fin: Vec<(usize, usize)> = last_row
                .iter()
                .enumerate()
                .filter(|&(_, &lr)| lr >= base && lr < base + r)
                .map(|(it, &lr)| (lr - base, it))
                .collect();
            fin.sort_unstable();
            // checkpoint cuts of every item whose boundary rows land in this
            // chunk; global row of item position p = item_base + (p - start)
            let mut breaks: Vec<(usize, usize)> = Vec::new();
            let mut after: Vec<(usize, usize, usize)> = Vec::new();
            let mut stage = 0usize;
            let max_stages = self.ckpt_stage_count();
            for (it, (_, toks)) in items.iter().enumerate() {
                for cut in super::prefix::ckpt_cuts(toks.len(), step) {
                    if cut <= starts[it] || stage >= max_stages {
                        continue;
                    }
                    let grow = item_base[it] + (cut - starts[it]);
                    if grow > base && grow <= base + r {
                        breaks.push((grow - base, stage));
                        after.push((stage, it, cut));
                        stage += 1;
                    }
                }
            }
            breaks.sort_unstable();
            self.rows_pass_body(chunk, 0, breaks)?;
            for (st, it, cut) in after {
                let (slot, toks) = &items[it];
                let keys = toks.clone();
                self.commit_stage(st, *slot, &keys, cut);
            }
            for (row, it) in fin {
                out[it] = self.head_row(row)?;
            }
            base += r;
        }
        for (slot, toks) in items {
            self.prefix_insert(*slot, toks);
        }
        Ok(out)
    }

    /// Queue a prompt for STALL-FREE chunked prefill (Sarathi shape). Does
    /// the whole admission prologue now, so a mixed tick only moves rows.
    pub(crate) fn prefill_begin_impl(
        &mut self,
        slot: usize,
        tokens: Vec<u32>,
    ) -> Result<(), GpuModelError> {
        // a queued entry for this slot is stale (the old request died and the
        // slot was reused): evict rather than wedge the slot
        self.chunked.retain(|c| c.slot != slot);
        self.admit_rows(slot, tokens.len())?;
        let cursor = self.prefix_resume_rows(slot, &tokens, tokens.len())?;
        if paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG_IDS").is_some() {
            tracing::info!("[spec2a-ids] admit slot={slot} cursor={cursor} prompt={tokens:?}");
        }
        self.reply_track_admit(slot, &tokens);
        // a live prompt queues ahead of batch ones (rows are picked in order)
        let at = (self.batch.as_ref().and_then(|b| b.prefix.as_ref()))
            .and_then(|r| r.chunk_queue_at(slot, self.chunked.iter().map(|c| c.slot)))
            .unwrap_or(self.chunked.len());
        self.chunked.insert(
            at,
            ChunkedPrefill {
                slot,
                keys: tokens.clone(),
                tokens,
                cursor,
                media: None,
            },
        );
        Ok(())
    }

    /// Drop slot's in-flight prefill (client hung up mid-prompt).
    pub(crate) fn prefill_abort_impl(&mut self, slot: usize) -> bool {
        let n = self.chunked.len();
        self.chunked.retain(|c| c.slot != slot);
        self.chunked.len() != n
    }

    /// Pick this tick's chunk rows: FIFO over the queue, up to `budget`
    /// rows, splitting the last prompt if it does not fit. The tick's width
    /// is the WHOLE-PROMPT rule: `PREFILL_TICK_BASE` rows, or the first
    /// queued prompt's remaining rows when that is more, never past the
    /// scratch cap - a cohort of short prompts ramps through base-width
    /// ticks (the median first token comes earlier that way), a long prompt
    /// streams the experts once instead of once per base width (GB10
    /// 2026-09-11: 2048 / 4096 / 8192 rows per tick measured on the 8x1k
    /// cohort and on 4k / 8k prompts).
    /// `wait`: a pass with nothing else to run takes the first queued prompt
    /// even when its encode has not landed (lending waits for it) rather than
    /// running empty; a round's pass never waits.
    pub(super) fn plan_chunk(
        &self,
        budget: usize,
        wait: bool,
    ) -> (Vec<(u32, u32, u32)>, Vec<(usize, usize, bool)>) {
        let mut rows: Vec<(u32, u32, u32)> = Vec::new();
        let mut take: Vec<(usize, usize, bool)> = Vec::new();
        // PADDOCK_TR_SKIP_UNLANDED (dev): a recording prompt whose encode is
        // still on the speech lane sits the pass out instead of the pass
        // waiting for it. Measured worse (GB10 2026-11-14, FLEURS c32 RTFx
        // 137.3 -> 134.1): the wait costs the die ~10 ms a round, a skipped
        // prompt costs its slot the whole round. Off: every queued prompt is
        // eligible and lending waits on the encode.
        let skip = paddock_models::dev_var_os!("PADDOCK_TR_SKIP_UNLANDED").is_some();
        let mut landed: Vec<usize> = (0..self.chunked.len())
            .filter(|&qi| {
                !skip
                    || match self.chunked[qi]
                        .media
                        .as_ref()
                        .and_then(|m| m.encode.as_ref())
                    {
                        Some((sh, _)) => sh
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .landed(&self.exec),
                        None => true,
                    }
            })
            .collect();
        if landed.is_empty() && wait {
            landed.push(0);
        }
        let Some(&q0) = landed.first() else {
            return (rows, take);
        };
        let first_rem = self.chunked[q0].tokens.len() - self.chunked[q0].cursor;
        let width = super::forward::PREFILL_TICK_BASE.max(first_rem);
        let cap = budget.clamp(1, self.prefill_chunk).min(width.max(1));
        // Prompt-aligned ticks (GB10, 2026-09-11):
        // a prompt after the first joins the tick only whole - within the
        // cap, or within a quarter-base overshoot the scratch can take - and
        // the tick ends at the previous prompt's boundary otherwise. A prompt
        // cut at the tick edge costs its owner a whole extra tick of
        // first-token latency and one more mamba segment split: the 8 x 1k
        // cohort ran as 2048-row cuts (five ticks, the 4th and 5th first
        // tokens on the 3rd tick) and now runs as 2-prompt ticks (four). The
        // first prompt keeps the whole-prompt rule above; on the big die the
        // scratch cap binds first and nothing changes.
        // PADDOCK_NO_PROMPT_ALIGN=1 restores the row-exact cut.
        let align = paddock_models::dev_var_os!("PADDOCK_NO_PROMPT_ALIGN").is_none();
        let slack_cap = if align {
            (cap + super::forward::PREFILL_TICK_BASE / 4)
                .min(budget.max(1))
                .min(self.prefill_chunk)
                .max(cap)
        } else {
            cap
        };
        for (k, &qi) in landed.iter().enumerate() {
            let c = &self.chunked[qi];
            if rows.len() >= cap {
                break;
            }
            let remaining = c.tokens.len() - c.cursor;
            let limit = if k == 0 { cap } else { slack_cap };
            if align && k > 0 && rows.len() + remaining > limit {
                break;
            }
            let n = remaining.min(limit - rows.len()).max(1);
            for j in 0..n {
                let p = c.cursor + j;
                rows.push((c.slot as u32, p as u32, c.tokens[p]));
            }
            take.push((qi, n, n == remaining));
        }
        (rows, take)
    }

    /// Advance cursors and drop finished prompts from the queue.
    pub(super) fn commit_chunk(
        &mut self,
        take: &[(usize, usize, bool)],
        finished_raw: Vec<(usize, crate::generator::FinishSample)>,
    ) -> Vec<(usize, crate::generator::FinishSample, usize)> {
        for &(qi, n, _) in take {
            self.chunked[qi].cursor += n;
        }
        let mut out = Vec::new();
        for (qi, fs) in finished_raw {
            let slot = self.chunked[qi].slot;
            let toks = std::mem::take(&mut self.chunked[qi].tokens);
            let keys = std::mem::take(&mut self.chunked[qi].keys);
            // a recording prompt's placeholder ids key nothing (mm.rs)
            if self.chunked[qi].media.is_none() {
                self.prefix_insert(slot, &keys);
            }
            out.push((slot, fs, toks.len()));
        }
        self.chunked.retain(|c| !c.tokens.is_empty());
        out
    }

    /// Build the fused tick's row stream: decode rows first (one band), then
    /// as much of the prefill queue as scratch capacity allows.
    /// Whether checkpoint staging buffers exist (prefix cache armed with a
    /// state pool); the checkpoint planners emit no cuts without them.
    pub(super) fn has_ckpt_stages(&self) -> bool {
        self.ckpt_stage_count() > 0
    }

    /// Staging blobs this serve allocated (`prefix::ckpt_stages` at enable;
    /// 0 with the cache off) - the per-pass cap every stage plan honours.
    pub(super) fn ckpt_stage_count(&self) -> usize {
        self.batch.as_ref().map_or(0, |b| b.d_ckpt_stage.len())
    }

    pub(super) fn fuse_rows(
        &self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
    ) -> (
        Vec<(u32, u32, u32)>,
        usize,
        Vec<(usize, usize)>,
        Vec<(usize, usize, bool)>,
    ) {
        let mut rows: Vec<(u32, u32, u32)> =
            decodes.iter().map(|&(s, t, p)| (s as u32, p, t)).collect();
        let dec_n = rows.len();
        let room = self
            .batch
            .as_ref()
            .expect("batch enabled")
            .cap
            .saturating_sub(dec_n);
        let (chunk_rows, take) = if room == 0 {
            (Vec::new(), Vec::new())
        } else {
            self.plan_chunk(budget.min(room), dec_n == 0)
        };
        rows.extend_from_slice(&chunk_rows);
        let mut fin: Vec<(usize, usize)> = Vec::new();
        let mut off = dec_n;
        for &(qi, n, done) in &take {
            if done {
                fin.push((off + n - 1, qi));
            }
            off += n;
        }
        (rows, dec_n, fin, take)
    }

    /// The mixed tick's checkpoint plan: cuts of any queued prompt whose
    /// boundary rows land inside this tick's take. Returns (PfCuts breaks,
    /// (stage, queue index, cut) commits for after the pass).
    pub(super) fn mixed_stage_plan(
        &self,
        take: &[(usize, usize, bool)],
        dec_n: usize,
    ) -> (Vec<(usize, usize)>, Vec<(usize, usize, usize)>) {
        let mut breaks = Vec::new();
        let mut after = Vec::new();
        // No staging buffers (prefix cache off, or no checkpoint pool fit):
        // nothing to stage into, so no cuts. Without this the walk indexed
        // `d_ckpt_stage[stg]` on an empty vec (PADDOCK_NO_PREFIX_CACHE=1
        // panicked on the first prompt past a page boundary, GB10 2026-09-11).
        if !self.has_ckpt_stages() {
            return (breaks, after);
        }
        let mut stage = 0usize;
        let mut row_base = dec_n;
        let step = self.tier_ckpt_step();
        let max_stages = self.ckpt_stage_count();
        for &(qi, n, _) in take {
            let c = &self.chunked[qi];
            if c.media.is_some() {
                // a recording prompt never feeds the cache (mm.rs)
                row_base += n;
                continue;
            }
            for cut in super::prefix::ckpt_cuts(c.tokens.len(), step) {
                if cut > c.cursor && cut <= c.cursor + n && stage < max_stages {
                    breaks.push((row_base + (cut - c.cursor), stage));
                    after.push((stage, qi, cut));
                    stage += 1;
                }
            }
            row_base += n;
        }
        (breaks, after)
    }

    /// Run the staged checkpoint commits after a mixed tick's pass.
    pub(super) fn mixed_stage_commit(&mut self, after: Vec<(usize, usize, usize)>) {
        for (st, qi, cut) in after {
            let slot = self.chunked[qi].slot;
            let keys = self.chunked[qi].keys.clone();
            self.commit_stage(st, slot, &keys, cut);
        }
    }

    /// One FUSED mixed tick: decode rows and the prefill chunk in a single
    /// weight-amortized pass, decode rows device-sampled (granite's shape).
    pub(crate) fn forward_mixed_sampled_impl(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
        plans: &[crate::generator::RowSample],
        fin_plans: &[(usize, crate::generator::RowSample)],
    ) -> Result<
        (
            crate::generator::SampledStep,
            Vec<(usize, crate::generator::FinishSample, usize)>,
        ),
        GpuModelError,
    > {
        use crate::generator::SampledStep;
        // Nothing queued -> a plain decode tick on the captured graph.
        if self.chunked.is_empty() {
            let step = if decodes.is_empty() {
                SampledStep {
                    ids: Vec::new(),
                    host_rows: Vec::new(),
                }
            } else {
                let toks: Vec<u32> = decodes.iter().map(|d| d.1).collect();
                let pos: Vec<u32> = decodes.iter().map(|d| d.2).collect();
                let slots: Vec<u32> = decodes.iter().map(|d| d.0 as u32).collect();
                self.forward_batch_sampled_slots(&toks, &pos, Some(&slots), plans)?
            };
            return Ok((step, Vec::new()));
        }
        let (rows, dec_n, fin, take) = self.fuse_rows(decodes, budget);
        if dec_n > 0 {
            let slots: Vec<u32> = decodes.iter().map(|d| d.0 as u32).collect();
            let pos: Vec<u32> = decodes.iter().map(|d| d.2).collect();
            self.ensure_rows(&slots, &pos)?;
        }
        let (breaks, after) = self.mixed_stage_plan(&take, dec_n);
        for &(slot, tok, pos) in decodes {
            self.reply_feed(slot, pos, tok);
        }
        let lent = self.lend_media(&take, dec_n)?;
        let pass = self.rows_pass_body(&rows, dec_n, breaks);
        self.return_media(lent);
        pass?;
        self.mixed_stage_commit(after);
        {
            let dslots: Vec<u32> = decodes.iter().map(|d| d.0 as u32).collect();
            let dpos: Vec<u32> = decodes.iter().map(|d| d.2).collect();
            self.reply_after_rows(&dslots, &dpos)?;
        }
        // Decode rows first: one bulk head over rows 0..dec_n, then device
        // sampling - it must precede the finisher heads because head_row
        // bounces through x[0] and rewrites head_logits[0..vocab].
        let step = if dec_n > 0 {
            self.head_rows(dec_n)?;
            self.sample_head_rows(dec_n, plans, true)?
        } else {
            SampledStep {
                ids: Vec::new(),
                host_rows: Vec::new(),
            }
        };
        let finished_raw = self.finish_prompts(&fin, fin_plans)?;
        let finished = self.commit_chunk(&take, finished_raw);
        Ok((step, finished))
    }

    /// The prompts a pass finished: each one's tail row `(pass row, queue
    /// index)` through the head, sampled on device where its plan allows,
    /// else its logits to the host.
    pub(super) fn finish_prompts(
        &mut self,
        fin: &[(usize, usize)],
        fin_plans: &[(usize, crate::generator::RowSample)],
    ) -> Result<Vec<(usize, crate::generator::FinishSample)>, GpuModelError> {
        use crate::generator::FinishSample;
        let mut finished_raw = Vec::with_capacity(fin.len());
        if fin.len() > 1 && self.w16_class_probe(fin.len()) {
            // the tails in one head call: a cohort of short prompts (a
            // recording wave) finishes several a tick, and each one-row head
            // re-streamed the whole vocab plane (4 x 2.7 ms a tick, GB10
            // 2026-11-14). The W16 head scores a row alike at any count.
            use crate::generator::RowSample;
            let rows: Vec<usize> = fin.iter().map(|&(row, _)| row).collect();
            let plans: Vec<RowSample> = fin
                .iter()
                .map(|&(_, qi)| {
                    let slot = self.chunked[qi].slot;
                    match fin_plans.iter().find(|(s, _)| *s == slot).map(|(_, p)| *p) {
                        Some(p @ RowSample::Device(_)) => p,
                        _ => RowSample::Host,
                    }
                })
                .collect();
            self.head_rows_at(&rows)?;
            let s = self.sample_head_rows(rows.len(), &plans, false)?;
            let mut host = s.host_rows.into_iter();
            for (i, &(_, qi)) in fin.iter().enumerate() {
                let fs = match plans[i] {
                    RowSample::Device(_) => FinishSample::Sampled(s.ids[i]),
                    _ => FinishSample::Logits(host.next().expect("host row").1),
                };
                finished_raw.push((qi, fs));
            }
        } else {
            for &(row, qi) in fin {
                let slot = self.chunked[qi].slot;
                let plan = fin_plans.iter().find(|(s, _)| *s == slot).map(|(_, p)| *p);
                let fs = match plan {
                    Some(p @ crate::generator::RowSample::Device(_)) => {
                        self.head_row_at(row)?;
                        let s = self.sample_head_rows(1, std::slice::from_ref(&p), false)?;
                        FinishSample::Sampled(s.ids[0])
                    }
                    _ => FinishSample::Logits(self.head_row(row)?),
                };
                finished_raw.push((qi, fs));
            }
        }
        Ok(finished_raw)
    }

    /// Speculative mixed tick: the decode rows' verify round, then the
    /// prefill chunk as a span of its own. Without it the trait's decline
    /// sent every mixed tick to the plain one AFTER the service had drafted
    /// (the drafts were thrown away) and its decode rows advanced one token
    /// each - at c32 on a recording cohort that was every other tick (GB10
    /// 2026-11-14: a 128-row round, then 3 wasted MTP steps and a 1-token
    /// band beside a ~700-row chunk). Two passes rather than one walk: the
    /// verify rows stand for decode steps and keep their class (W16), the
    /// chunk keeps the block-scaled lanes, and the two cannot share a
    /// launch - the band recompute a fused walk would need re-streams the
    /// experts anyway. The span carries no decode rows, so no band either.
    /// The round goes first: a decline leaves the chunk untouched, which is
    /// the service's fallback contract.
    pub(crate) fn forward_mixed_spec_impl(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
        budget: usize,
        plans: &[crate::sampler::DevicePlan],
        fin_plans: &[(usize, crate::generator::RowSample)],
    ) -> Result<
        (
            Option<Vec<u32>>,
            Vec<(usize, crate::generator::FinishSample, usize)>,
        ),
        GpuModelError,
    > {
        if self.fused_rounds() {
            // the round and the chunk in one pass (fused.rs), the plans'
            // checks as the round alone takes them
            use crate::sampler::DevicePlan;
            let total: usize = reqs.iter().map(|r| r.2.len()).sum();
            let runnable = plans.iter().all(|p| match p {
                DevicePlan::Greedy | DevicePlan::Categorical { .. } => true,
                DevicePlan::TruncCat { .. } => self.device_trunc_supported(),
                DevicePlan::RsVerify { .. } | DevicePlan::RsTrunc { .. } => false,
            });
            if plans.len() != total || !runnable || !self.supports_device_sampling_impl() {
                return Ok((None, Vec::new()));
            }
            return Ok(
                match self.forward_fused_round(
                    reqs,
                    budget,
                    super::spec::VerifyPick::Plans(plans),
                    fin_plans,
                )? {
                    Some((picks, finished)) => (Some(picks), finished),
                    None => (None, Vec::new()),
                },
            );
        }
        let Some(picks) = self.forward_spec_batch_plans_impl(reqs, plans)? else {
            return Ok((None, Vec::new()));
        };
        let (_, finished) = self.forward_mixed_sampled_impl(&[], budget, &[], fin_plans)?;
        Ok((Some(picks), finished))
    }

    /// The unsampled mixed tick (full logits readback).
    pub(crate) fn forward_mixed_impl(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
    ) -> Result<(Vec<f32>, Vec<(usize, Vec<f32>, usize)>), GpuModelError> {
        if self.chunked.is_empty() {
            if decodes.is_empty() {
                return Ok((Vec::new(), Vec::new()));
            }
            let toks: Vec<u32> = decodes.iter().map(|d| d.1).collect();
            let pos: Vec<u32> = decodes.iter().map(|d| d.2).collect();
            let slots: Vec<u32> = decodes.iter().map(|d| d.0 as u32).collect();
            self.batch_step_slots(&toks, &pos, &slots)?;
            return Ok((self.read_batch_logits(decodes.len())?, Vec::new()));
        }
        let (rows, dec_n, fin, take) = self.fuse_rows(decodes, budget);
        if dec_n > 0 {
            let slots: Vec<u32> = decodes.iter().map(|d| d.0 as u32).collect();
            let pos: Vec<u32> = decodes.iter().map(|d| d.2).collect();
            self.ensure_rows(&slots, &pos)?;
        }
        let (breaks, after) = self.mixed_stage_plan(&take, dec_n);
        for &(slot, tok, pos) in decodes {
            self.reply_feed(slot, pos, tok);
        }
        let lent = self.lend_media(&take, dec_n)?;
        let pass = self.rows_pass_body(&rows, dec_n, breaks);
        self.return_media(lent);
        pass?;
        self.mixed_stage_commit(after);
        {
            let dslots: Vec<u32> = decodes.iter().map(|d| d.0 as u32).collect();
            let dpos: Vec<u32> = decodes.iter().map(|d| d.2).collect();
            self.reply_after_rows(&dslots, &dpos)?;
        }
        let mut dec_logits = Vec::new();
        if dec_n > 0 {
            self.head_rows(dec_n)?;
            dec_logits = self.read_batch_logits(dec_n)?;
        }
        let mut finished_raw = Vec::with_capacity(fin.len());
        for &(row, qi) in &fin {
            finished_raw.push((
                qi,
                crate::generator::FinishSample::Logits(self.head_row(row)?),
            ));
        }
        let finished = self
            .commit_chunk(&take, finished_raw)
            .into_iter()
            .map(|(slot, fs, n)| match fs {
                crate::generator::FinishSample::Logits(l) => (slot, l, n),
                crate::generator::FinishSample::Sampled(_) => unreachable!("unsampled mixed tick"),
            })
            .collect();
        Ok((dec_logits, finished))
    }
}
