//! The `Generator` face of the Flash Next lane - the engine's scheduler
//! calls in here; everything it forwards to lives in `forward.rs` and its
//! siblings.

use super::*;
use crate::generator::{RowSample, SampledStep};

impl crate::generator::Generator for Qwen4ExpGpu {
    fn release_idle_memory(&mut self) {
        self.exec.trim_mem_pool();
    }

    /// The resident-weight line `/api/stats` publishes and the catalog's
    /// shape generator measures from (see `weights_mem_bytes` above).
    fn weights_mem_bytes(&self) -> Option<u64> {
        Self::weights_mem_bytes(self)
    }

    /// The live mempool counter every family publishes as its device ledger
    /// (`/api/stats` `model_mem`): without it the manager had nothing to
    /// reconcile this lane's NVML figure against.
    fn device_mem_used(&self) -> Option<u64> {
        self.exec.process_mem_used()
    }

    fn reply_pin(&mut self, slot: usize) {
        Qwen4ExpGpu::reply_pin(self, slot);
    }
    fn anchor_at(&mut self, _slot: usize, tokens: &[u32], upto: usize) {
        self.pages.anchor(tokens, upto);
    }
    fn set_slot_background(&mut self, slot: usize, on: bool) {
        self.pages.set_slot_low(slot, on);
    }

    fn release_inactive_slots(&mut self, occupied: &[bool]) {
        Qwen4ExpGpu::release_inactive_slots(self, occupied);
    }
    fn reset(&mut self) {
        // trait returns unit; a state-clear failure here would surface on the
        // next forward as a driver error rather than being swallowed silently
        if let Err(e) = Qwen4ExpGpu::reset(self) {
            tracing::warn!("qwen4exp reset: {e}");
        }
    }

    fn forward(&mut self, token: u32) -> Result<Vec<f32>, crate::generator::GenError> {
        self.decode_step(token).map_err(q4x_gen_err)
    }
    fn vocab(&self) -> usize {
        self.cfg.vocab
    }
    fn max_context(&self) -> usize {
        self.max_tokens
    }

    /// Slots are allocated at LOAD (the GDN recurrent state, both conv windows
    /// and every scratch plane are sized by them), so this reports what the
    /// instance already carries rather than allocating. `serving.rs` passes the
    /// serve width into `load_with_slots` for exactly this reason.
    fn enable_batch(&mut self, max_batch: usize) -> Result<usize, crate::generator::GenError> {
        Ok(self.slots.min(max_batch.max(1)).max(1))
    }

    fn forward_prefill(
        &mut self,
        slot: usize,
        tokens: &[u32],
    ) -> Result<Vec<f32>, crate::generator::GenError> {
        self.prefill_slot(slot, tokens).map_err(q4x_gen_err)
    }

    /// The scheduler's whole admitted wave in one walk. The trait default
    /// prefills one prompt at a time, which at c32 is a 1.66 s blocking
    /// prefill tick against a 14.6 ms decode tick.
    fn forward_prefill_batch(
        &mut self,
        items: &[(usize, Vec<u32>)],
    ) -> Result<Vec<Vec<f32>>, crate::generator::GenError> {
        if !super::super::prefill_wave_enabled() || !self.exec.has_gated_delta_recurrent_runs() {
            return items
                .iter()
                .map(|(s, t)| self.forward_prefill(*s, t))
                .collect();
        }
        // A wave wider than the walk's scratch is SPLIT, not abandoned: at
        // imax the prompts are 1024 tokens, so a 32-wide admission is 32768
        // rows against a 4096-row walk and the whole cell would otherwise fall
        // back to one prompt at a time. Four per sub-wave still fills the
        // recurrence grid four times over.
        let mut out = Vec::with_capacity(items.len());
        let mut lo = 0usize;
        while lo < items.len() {
            let mut hi = lo;
            let mut rows = 0usize;
            while hi < items.len() && (hi == lo || rows + items[hi].1.len() <= self.walk_rows) {
                rows += items[hi].1.len();
                hi += 1;
            }
            if rows > self.walk_rows {
                // a single prompt longer than the walk: the serial entry owns
                // that case (it is the one the chunked lane would take)
                let (s, t) = &items[lo];
                out.push(self.forward_prefill(*s, t)?);
            } else {
                out.extend(self.prefill_slots(&items[lo..hi]).map_err(q4x_gen_err)?);
            }
            lo = hi;
        }
        Ok(out)
    }

    fn forward_prefill_stream(
        &mut self,
        tokens: &[u32],
    ) -> Result<Vec<f32>, crate::generator::GenError> {
        self.forward_prompt(tokens).map_err(q4x_gen_err)
    }

    /// Row i drives slot i. The scheduler passes positions explicitly while
    /// this model tracks them per slot; they are CHECKED rather than trusted,
    /// so a desync fails loudly here instead of silently decoding at the wrong
    /// position (the failure mode a wrong-position KV read would give is
    /// plausible text, which no gate would catch).
    fn forward_batch(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
    ) -> Result<Vec<f32>, crate::generator::GenError> {
        if tokens.len() != positions.len() {
            return Err(crate::generator::GenError::Backend(format!(
                "forward_batch: {} tokens vs {} positions",
                tokens.len(),
                positions.len()
            )));
        }
        // The scheduler ticks its whole occupied PREFIX, not just the live
        // rows: a slot that finished (or is occupied but not yet prefilled)
        // rides along as a HOLE feeding (token 0, position 0), and the rows
        // are only meaningful where it is not. `forward_batch` has no plans
        // to read, so position 0 is the hole marker - a decode row is always
        // at position >= 1 because it follows a prompt.
        let rows: Vec<(usize, u32)> = tokens
            .iter()
            .copied()
            .enumerate()
            .filter(|&(i, _)| positions[i] != 0)
            .collect();
        Self::check_positions(&self.pos, &rows, positions)?;
        let per_row = self.decode_step_batch(&rows).map_err(q4x_gen_err)?;
        self.mtp_flush().map_err(q4x_gen_err)?;
        // hand back a full [rows, vocab] plane: hole rows keep their zeros,
        // which is what the caller's own Hole arm discards
        let mut out = vec![0f32; tokens.len() * self.cfg.vocab];
        for ((i, _), row) in rows.iter().copied().zip(per_row) {
            out[i * self.cfg.vocab..(i + 1) * self.cfg.vocab].copy_from_slice(&row);
        }
        Ok(out)
    }

    /// The service checks this before drawing per-row uniforms, so answering
    /// truthfully is what keeps a slot's seed stream from paying for a path
    /// that will not run.
    fn take_prefill_reused(&mut self, slot: usize) -> usize {
        self.prefix.as_mut().map_or(0, |p| p.take_reused(slot))
    }
    fn supports_device_sampling(&self) -> bool {
        self.exec.has_sample_rows()
    }

    fn supports_device_trunc(&self) -> bool {
        self.exec.has_sample_rows_t() && self.exec.has_sample_rows_p()
    }

    fn supports_stop_mask(&self) -> bool {
        self.exec.has_stop_mask_rows() && self.exec.has_sample_rows()
    }

    fn set_stop_masks(&mut self, masks: Vec<Option<crate::sampler::StopMask>>) {
        self.stop.set(masks);
    }

    fn take_stop_pressure(&mut self) -> Vec<bool> {
        self.stop.take(&self.exec)
    }

    /// Device-sampled decode tick: the walk lands `[rows, vocab]` in `d_out`,
    /// the sampler reduces it on device, and only `Host`-plan rows read a
    /// vocab row back. This is the method whose absence made the first serving
    /// measurement 27.6 ms/tok against the engine's 7.9 - without it the
    /// service reads 0.99 MB of logits per token at c1 (31.8 MB/step at c32)
    /// and samples on the host.
    fn forward_batch_sampled(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        plans: &[RowSample],
    ) -> Result<SampledStep, crate::generator::GenError> {
        if tokens.len() != positions.len() || plans.len() != tokens.len() {
            return Err(crate::generator::GenError::Backend(format!(
                "forward_batch_sampled: {} tokens, {} positions, {} plans",
                tokens.len(),
                positions.len(),
                plans.len()
            )));
        }
        // Hole rows are the scheduler's own convention for a slot inside the
        // occupied prefix that must not decode this tick (finished, or
        // occupied with no KV behind it yet) and they feed (0, 0). Ticking
        // them anyway is what a first cut did, and the position check then
        // failed the whole tick: 269 requests died as "slot 0: scheduler says
        // position 0, model is at 309" across the first serve ladder.
        let rows: Vec<(usize, u32)> = tokens
            .iter()
            .copied()
            .enumerate()
            .filter(|&(i, _)| !matches!(plans[i], RowSample::Hole))
            .collect();
        Self::check_positions(&self.pos, &rows, positions)?;
        if rows.is_empty() {
            return Ok(SampledStep {
                ids: vec![0u32; tokens.len()],
                host_rows: Vec::new(),
            });
        }
        self.decode_batch_walk(&rows).map_err(q4x_gen_err)?;
        let step = self
            .sample_rows_from_logits(&rows, plans)
            .map_err(q4x_gen_err)?;
        // after the sample: feeding the head reuses the walk's scratch
        self.mtp_flush().map_err(q4x_gen_err)?;
        Ok(step)
    }

    // ---- chunked prefill: the mixed tick (forward/chunked.rs) ----------

    fn supports_chunked_prefill(&self) -> bool {
        self.chunked_supported()
    }

    fn prefill_begin(
        &mut self,
        slot: usize,
        tokens: Vec<u32>,
    ) -> Result<(), crate::generator::GenError> {
        self.prefill_begin_impl(slot, tokens).map_err(q4x_gen_err)
    }

    fn prefill_abort(&mut self, slot: usize) -> bool {
        self.prefill_abort_impl(slot)
    }

    fn prefill_queue(&self) -> Vec<(usize, usize, usize)> {
        self.prefill_queue_impl()
    }

    fn prefill_tick_cap(&self, decode_rows: usize) -> usize {
        self.prefill_tick_cap_impl(decode_rows)
    }

    /// The mixed tick with host sampling: the decode rows' `[nd, vocab]`
    /// logits in `decodes` order, and each finished prompt's last logits.
    fn forward_mixed(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
    ) -> Result<(Vec<f32>, Vec<(usize, Vec<f32>, usize)>), crate::generator::GenError> {
        let w = self.mixed_walk(decodes, budget).map_err(q4x_gen_err)?;
        let logits = if w.nd > 0 {
            self.exec
                .to_host_len(&self.sc.d_out, w.nd * self.cfg.vocab)
                .map_err(|e| q4x_gen_err(e.into()))?
        } else {
            Vec::new()
        };
        let fins: Vec<(usize, usize, usize)> = w.finishers(self).collect();
        let mut finished = Vec::with_capacity(fins.len());
        for (slot, row, rows) in fins {
            finished.push((slot, self.out_row(row).map_err(q4x_gen_err)?, rows));
        }
        self.mixed_commit(w).map_err(q4x_gen_err)?;
        Ok((logits, finished))
    }

    /// The mixed tick, decode rows sampled on device (plans in `decodes`
    /// order); a finishing prompt hands back its last logits.
    fn forward_mixed_sampled(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
        plans: &[RowSample],
        _fin_plans: &[(usize, RowSample)],
    ) -> Result<
        (
            SampledStep,
            Vec<(usize, crate::generator::FinishSample, usize)>,
        ),
        crate::generator::GenError,
    > {
        if plans.len() != decodes.len() {
            return Err(crate::generator::GenError::Backend(format!(
                "forward_mixed_sampled: {} decode rows, {} plans",
                decodes.len(),
                plans.len()
            )));
        }
        let w = self.mixed_walk(decodes, budget).map_err(q4x_gen_err)?;
        let at: Vec<usize> = (0..w.nd).collect();
        let (ids, host_rows) = self
            .sample_out_rows(&plans[..w.nd], &at)
            .map_err(q4x_gen_err)?;
        let fins: Vec<(usize, usize, usize)> = w.finishers(self).collect();
        let mut finished = Vec::with_capacity(fins.len());
        for (slot, row, rows) in fins {
            let logits = self.out_row(row).map_err(q4x_gen_err)?;
            finished.push((slot, crate::generator::FinishSample::Logits(logits), rows));
        }
        // after the reads: seeding the drafter reuses the walk's scratch
        self.mixed_commit(w).map_err(q4x_gen_err)?;
        Ok((SampledStep { ids, host_rows }, finished))
    }

    /// A drafter exists once a head GGUF is attached (`attach_mtp`); the
    /// service's policy resolution handles --spec / --no-spec on top.
    fn spec_capable(&self) -> bool {
        self.mtp.is_some() && self.exec.has_argmax_rows()
    }

    /// Per-tick: keep feeding the draft head only while speculation is the
    /// plan for this width. Stashing every decode row and draining it through
    /// a head pass is most of the cost of having a drafter attached at all -
    /// at 8 live slots it was 19 ms of a 45 ms tick that no round consumed.
    fn spec_fuse_hint(&mut self, on: bool) {
        if let Some(m) = self.mtp.as_mut() {
            m.feed = on;
        }
    }

    /// Same question for the eager prefill seed.
    fn spec_warm_hint(&mut self, on: bool) {
        if let Some(m) = self.mtp.as_mut() {
            m.warm_prefills = on;
        }
    }

    fn forward_spec_batch(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
    ) -> Result<Option<Vec<u32>>, crate::generator::GenError> {
        self.verify_round(reqs).map_err(q4x_gen_err)
    }

    /// The sampled twin: every verify row drawn on device from its own plan.
    /// Without it a temperature > 0 serve never speculated at all - see
    /// `verify_round_plans` for the exactness argument and what it measured.
    fn forward_spec_batch_plans(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
        plans: &[crate::sampler::DevicePlan],
    ) -> Result<Option<Vec<u32>>, crate::generator::GenError> {
        self.verify_round_plans(reqs, plans).map_err(q4x_gen_err)
    }

    /// The host-sampled round - the one constrained slots, i.e. every
    /// tool-carrying request, speculate through: raw row logits out, the
    /// service samples them through the slot's sampler and grammar, then
    /// `spec_commit` keeps what it accepted (see `verify_open`).
    fn forward_spec_verify(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
    ) -> Result<Option<Vec<f32>>, crate::generator::GenError> {
        self.verify_open(reqs).map_err(q4x_gen_err)
    }

    fn spec_commit(&mut self, committed: &[u32]) -> Result<(), crate::generator::GenError> {
        self.verify_close(committed).map_err(q4x_gen_err)
    }

    fn spec_draft_batch(
        &mut self,
        pendings: &[(usize, u32)],
        k: usize,
    ) -> Result<Option<Vec<Vec<u32>>>, crate::generator::GenError> {
        if self.mtp.is_none() {
            return Ok(None);
        }
        let mut out = Vec::with_capacity(pendings.len());
        for &(slot, tok) in pendings {
            out.push(self.mtp_draft(slot, tok, k).map_err(q4x_gen_err)?);
        }
        Ok(Some(out))
    }

    /// No re-warm path: every walk feeds the head's stash, so a cold slot is
    /// one whose rows outran what it tracks - it drafts nothing until its next
    /// prompt, and the verify still serves it one token a round.
    fn spec_ensure_warm(
        &mut self,
        slot: usize,
        _committed: &[u32],
        want_pos: u32,
    ) -> Result<bool, crate::generator::GenError> {
        Ok(self.mtp_warm(slot, want_pos as usize + 1))
    }

    /// Depth 3, elected in serving - the row budget asks for 7 at one live
    /// slot and this family cannot repay it.
    ///
    /// Every extra verify row costs a full per-row pass over the routed
    /// experts it touches, while the tokens it returns grow only with
    /// acceptance. GB10, code prompt, greedy, MTP head attached, decode-only
    /// tok/s with TTFT excluded (2026-09-19):
    ///
    ///   depth 1   29.95 median of 5   (acceptance 90.2%)
    ///   depth 3   33.95 median of 3   (acceptance 81.2%)   ELECTED
    ///
    /// Depth 7 - what the row budget asks for at one live slot - was capped
    /// here without ever being measured until 2026-09-20, when the line above
    /// said only that this family "cannot repay it". It does not: on
    /// syn_128x128_c1, 3 reps each, depth 3 medians 30.45 (30.45/32.12/30.01)
    /// against depth 7's 26.70 (26.70/27.23/23.73), -12.3%, losing every rep.
    /// A/B it with `PADDOCK_Q38FN_SPEC_DEPTH`.
    ///
    /// Why it loses is worth keeping, because the original reasoning here was
    /// wrong in a way that invites re-litigation. It used to say each extra
    /// row "re-streams" its experts; it does not - the rows of a verify batch
    /// share experts ~10x more than chance (26.3 distinct of 35.5 rows,
    /// measured in serving) and L2 banks ~41% of that for free, so the bytes
    /// are largely not re-read. The cost is per-row unpack on the i-quant
    /// seat, which no routing trick removes (priced on the seat formats,
    /// 2026-09-20), and on top of it the dense side is not flat in n either:
    /// GDN's per-token recurrence, attention over the wider batch and the K
    /// sequential MTP draft passes all grow with depth. Two terms growing
    /// against an acceptance curve that decays is why deeper does not pay.
    ///
    /// Elect this in serving, not in the harness. `examples/q38fn_spec_bench`
    /// prices the same depths at 31.60 (d1) against 28.09 (d3) - it inverts
    /// the ordering, and an earlier cut of this election took its answer and
    /// shipped depth 1. The harness is a bare loop over draft + verify and
    /// says so itself ("its numbers are never a serving cell"); a serving
    /// round also carries the state save, the rollback bookkeeping, the accept
    /// walk and the stream, and those are per-ROUND costs that a deeper draft
    /// amortizes over more committed tokens. Acceptance moves the other way
    /// (a single draft is easier to hit) and is not the thing to maximize -
    /// tokens per round against the round's real cost is.
    ///
    /// Re-measure in serving when the verify row's cost moves (the
    /// routed-expert seat, the expert byte count, the walk's kernel election).
    fn spec_depth_cap(&self) -> Option<usize> {
        Some(super::super::spec_depth_override().unwrap_or(3))
    }

    /// The drafter decides warmth per slot (a cold slot gets an empty draft
    /// list and rides the verify as a one-row chunk).
    fn spec_draft_per_slot_warm(&self) -> bool {
        self.mtp.is_some()
    }

    /// Canonical rejection sampling: armed only when the head's RS buffers
    /// exist, which needs PADDOCK_SPEC_RS *and* a pack carrying both kernels
    /// (`mtp::RsBufs`). Answering truthfully is load-bearing - the service
    /// stashes chain draws exactly when this is true, and a backend that
    /// claimed the arm without the buffers would take sampled rounds it cannot
    /// resolve and emit the wrong distribution.
    fn supports_spec_rs(&self) -> bool {
        self.mtp.as_ref().is_some_and(|m| m.rs.is_some())
    }

    fn spec_rs_stash(&mut self, draws: Vec<crate::generator::SpecRsDraw>) {
        self.spec_rs_draws = Some(draws);
    }
}
