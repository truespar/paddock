//! Transcribe's fused round: a spec verify round and the prompt chunk beside
//! it in ONE pass. As two passes (the round, then the chunk as a span) every
//! tick that admitted a recording streamed the experts twice; at c32 on a
//! recording cohort the spans held ~35% of the wall (GB10 2026-11-14), the
//! second expert stream most of it. Here the verify rows run first in a
//! prompt walk - every lane the prompt class's, which no row's bits depend
//! on the rows beside it in - and only their mamba advance is the verify
//! walk's (`PfCuts::vruns`): the live state stays pre-round and the commit
//! replays the accepted rows. Rounds without a chunk go the same way, so a
//! round scores in one class whether or not a prompt rides it.

use super::batch::PfCuts;
use super::spec::VerifyPick;
use super::*;

impl GpuNemotron {
    /// This model's rounds run fused: Transcribe (the MTP drafter, the f16
    /// replay class and the run-table kernels; PADDOCK_TR_NO_FUSED keeps the
    /// verify walk for the A/B).
    pub(crate) fn fused_rounds(&self) -> bool {
        self.hp.transcribe
            && self.dflash.is_none()
            && self.mtp.is_some()
            && self.exec.has_mamba_verify_runs()
            && paddock_models::dev_var_os!("PADDOCK_TR_NO_FUSED").is_none()
    }

    /// One fused round: `reqs` as a verify round's (`[pending, drafts...]`
    /// per slot), picked by `pick`, and up to `chunk_budget` rows of the
    /// queued prompts (0 = none) in the same pass. Returns the round's picks
    /// and the prompts it finished; `None` = declined before anything moved
    /// (the caller's fallback contract).
    pub(super) fn forward_fused_round(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
        chunk_budget: usize,
        pick: VerifyPick<'_>,
        fin_plans: &[(usize, crate::generator::RowSample)],
    ) -> Result<
        Option<(
            Vec<u32>,
            Vec<(usize, crate::generator::FinishSample, usize)>,
        )>,
        GpuModelError,
    > {
        let total: usize = reqs.iter().map(|r| r.2.len()).sum();
        if total == 0
            || total > self.spec_rows()
            || !self.spec_verify_ready()
            || self.batch.is_none()
            || reqs.iter().all(|r| r.2.len() < 2)
        {
            return Ok(None);
        }
        self.pipe_b_abort();
        self.ensure_verify_planes()?;
        if !self
            .batch
            .as_ref()
            .and_then(|b| b.verify.as_ref())
            .is_some_and(|v| v.rescan)
        {
            return Ok(None);
        }
        // the round's rows: one run a request
        let mut rows: Vec<(u32, u32, u32)> = Vec::with_capacity(total);
        let mut runs: Vec<(usize, usize, u32)> = Vec::with_capacity(reqs.len());
        let mut run_pos: Vec<u32> = Vec::with_capacity(reqs.len());
        for &(slot, pos, ref chunk) in reqs {
            runs.push((rows.len(), chunk.len(), slot as u32));
            run_pos.push(pos as u32);
            for (i, &t) in chunk.iter().enumerate() {
                rows.push((slot as u32, (pos + i) as u32, t));
            }
        }
        {
            let slots: Vec<u32> = rows.iter().map(|r| r.0).collect();
            let pos: Vec<u32> = rows.iter().map(|r| r.1).collect();
            self.ensure_rows(&slots, &pos)?;
        }
        let vruns = runs.len();
        // the prompt chunk beside them, as many rows as the pass has room
        let room = self
            .batch
            .as_ref()
            .expect("batch enabled")
            .cap
            .saturating_sub(total);
        let (chunk_rows, take) = if chunk_budget == 0 || room == 0 || self.chunked.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            self.plan_chunk(chunk_budget.min(room), false)
        };
        for (i, x) in chunk_rows.iter().enumerate() {
            let row = total + i;
            let prompt_run = runs.len() > vruns;
            match runs.last_mut() {
                Some((off, n, s)) if prompt_run && *s == x.0 && *off + *n == row => *n += 1,
                _ => {
                    runs.push((row, 1, x.0));
                    run_pos.push(x.1);
                }
            }
        }
        rows.extend_from_slice(&chunk_rows);
        let (breaks, after) = self.mixed_stage_plan(&take, total);
        let mut fin: Vec<(usize, usize)> = Vec::new();
        let mut off = total;
        for &(qi, n, done) in &take {
            if done {
                fin.push((off + n - 1, qi));
            }
            off += n;
        }
        // the prompt rows' drafter spans (slot, first pos, end pos, row)
        let mut spans: Vec<(usize, usize, usize, usize)> = Vec::new();
        for (i, x) in chunk_rows.iter().enumerate() {
            let (s, p) = (x.0 as usize, x.1 as usize);
            match spans.last_mut() {
                Some((ls, _, le, _)) if *ls == s && *le == p => *le += 1,
                _ => spans.push((s, p, p + 1, total + i)),
            }
        }
        let lent = self.lend_media(&take, total)?;
        let walk = (|| -> Result<(), GpuModelError> {
            let toks: Vec<u32> = rows.iter().map(|r| r.2).collect();
            let pos: Vec<u32> = rows.iter().map(|r| r.1).collect();
            let slots: Vec<u32> = rows.iter().map(|r| r.0).collect();
            self.upload_rows(&toks, &pos, &slots)?;
            self.embed_rows(rows.len())?;
            self.layer_walk(
                rows.len(),
                Some(&PfCuts {
                    runs,
                    run_pos,
                    dec: 0,
                    breaks,
                    vruns,
                }),
                false,
            )
        })();
        self.return_media(lent);
        walk?;
        self.mixed_stage_commit(after);
        // the round: head over its rows, picks, accept, rollback
        self.verify_head(total)?;
        let (picks, accepts) = self.verify_pick_commit(reqs, total, pick, true)?;
        // the drafter over every row of the pass (its block walks rows 0..r
        // in one go), coverage through the round's accepted rows and the
        // prompts' whole spans
        if self.mtp.as_ref().is_some_and(|m| m.state.is_some()) {
            let mut mruns: Vec<(usize, usize, usize)> = reqs
                .iter()
                .zip(&accepts)
                .map(|(r, a)| (r.0, a.1, r.2.len()))
                .collect();
            mruns.extend(spans.iter().map(|&(s, a, b, row)| (s, row, b - a)));
            self.mtp_append_rows(&mruns)?;
            let mut adv: Vec<(usize, usize, usize, usize)> = reqs
                .iter()
                .zip(&accepts)
                .map(|(&(slot, pos0, _), &(_, off, acc))| (slot, pos0, pos0 + acc, off + acc - 1))
                .collect();
            adv.extend(
                spans
                    .iter()
                    .map(|&(s, a, b, row)| (s, a, b, row + (b - a) - 1)),
            );
            self.mtp_advance_many(&adv)?;
        }
        // the prompts the pass finished (their heads rewrite x's first rows,
        // so after everything above has read them)
        let finished_raw = self.finish_prompts(&fin, fin_plans)?;
        let finished = self.commit_chunk(&take, finished_raw);
        Ok(Some((picks, finished)))
    }
}
