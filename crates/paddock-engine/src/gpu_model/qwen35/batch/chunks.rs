//! The chunked-prefill advance: one tick's worth of chunks for the prompts in
//! flight (`advance_chunks`), and the batched weight-amortized pass over fresh
//! chunked prompts it can take (`advance_chunks_batched`).

use super::*;

impl GpuQwen35 {
    /// Chunked prefill is available once enable_batch has run. OPT-IN
    /// (PADDOCK_CHUNKED_PREFILL, DEFAULT-ON via apply_default_stack;
    /// kill: PADDOCK_NO_CHUNKED_PREFILL - the kill pins the classic
    /// blocking prefill),
    /// which an early sweep showed beats the chunked/unified path on the
    /// prefill-heavy concurrency configs (c8/c32/pf8). Enable it (with
    /// PADDOCK_UNIFIED=1 for the fused tick) for decode-dominated workloads where
    /// it wins (dc4/c1). Reverted from default-on after that measurement.
    pub fn supports_chunked_prefill(&self) -> bool {
        self.batch.is_some() && std::env::var_os("PADDOCK_CHUNKED_PREFILL").is_some()
    }

    /// `Generator::prefill_queue`: every tick shape here spends its budget
    /// FIFO over `chunked`, from each prompt's `done`.
    pub(crate) fn prefill_queue(&self) -> Vec<(usize, usize, usize)> {
        self.chunked
            .iter()
            .map(|c| (c.slot, c.done, c.tokens.len() - c.done))
            .collect()
    }

    /// `Generator::prefill_tick_cap`: a long prompt advances at most one
    /// unified span per tick - fused, overlapped, spec-in-mixed, and the split
    /// tick's bounded head alike (a short tail the split tick prefills whole
    /// is shallow-or-short, never a paced tick's shape).
    pub(crate) fn prefill_tick_cap(&self) -> usize {
        super::unified_prefill_rows()
    }

    /// Split-tick prefill (the two-forward mixed tick: deep decode batches
    /// past PADDOCK_UNIFIED_DEC_MAX, or PADDOCK_NO_UNIFIED): run in-flight
    /// prompts this tick, FIFO (earliest admission first -> its first token
    /// sooner), up to a ~`budget` row soft cap but always at least one. A tail
    /// that FITS the cap is prefilled whole from `done` (the resume point
    /// `prefill_begin` restored) via `prefill_slot_tail` - bit-identical to a
    /// standalone prefill of the same tokens. This staggers the cohort's first
    /// tokens over several ticks and lets decode run between them (the TTFT
    /// win) without splitting a short prompt mid-tick.
    ///
    /// A tail LONGER than the cap never runs whole. It used to - "always at
    /// least one" meant a 150K-token prompt at the head of the queue was one
    /// prefill call, and every decode row of the serve waited minutes for its
    /// next token (measured on GB10, qwen3.8-27b). Every serving engine bounds
    /// the per-step prefill instead (vLLM's max_num_batched_tokens, SGLang's
    /// chunked_prefill_size), and so does every other path here: such a head
    /// advances by one prefill-only unified span per tick - the same span the
    /// fused, overlapped and spec-in-mixed ticks run, cut at the budget, its
    /// DeltaNet state and conv window resumed in the slot across ticks - and
    /// the decode pass follows it. The scheduler's budget then bounds the tick
    /// for long prompts too, which is what its tick pacer (crate::pacing)
    /// relies on. A long tail behind a short head waits until it heads the
    /// queue. Invariant this also keeps: the whole-tail path never sees a
    /// tail longer than the elected chunk - its resume walk stages the conv
    /// window for at most that many rows, and a 19,920-row tail against a
    /// 1024-row election tripped that assert and took the engine thread down
    /// (GB10, 16 slots under an 80 GiB budget, 2026-09-23). Finishers
    /// sample per `fin_plans` on the span path (Logits without a plan), and
    /// read logits back on the whole-tail path.
    /// Returns `(slot, finish, prompt rows)` per finished prompt.
    pub(super) fn advance_chunks(
        &mut self,
        budget: usize,
        fin_plans: &[(usize, crate::generator::RowSample)],
    ) -> Result<Vec<(usize, crate::generator::FinishSample, usize)>, GpuModelError> {
        use crate::generator::FinishSample;
        let cap = budget.min(self.prefill_chunk_rows).max(1);
        if self
            .chunked
            .first()
            .is_some_and(|c| c.mm.is_some() || c.tokens.len() - c.done > cap)
        {
            return if self.unified_span_launch(cap, fin_plans)? {
                self.unified_span_finish()
            } else {
                Ok(Vec::new())
            };
        }
        let logits = |v: Vec<(usize, Vec<f32>, usize)>| {
            v.into_iter()
                .map(|(k, l, r)| (k, FinishSample::Logits(l), r))
                .collect()
        };
        // With a DFlash drafter armed the soft cap is HARD past prompt 0. The
        // drafter's fusion accumulator is sized for chunk_tick_rows() prefill
        // rows (+ the verify/decode share), and the soft cap's overshoot -
        // admit the next tail whenever `used < cap` - put 8 x 1075-token
        // tails (8601 rows) in one tick against an 8544-row cap. tap_band
        // then marks the state stale and the append wipes every ring in the
        // tick, decoding slots included - which costs a wide leg roughly
        // three quarters of its throughput. Deferring that
        // one tail by a tick costs nothing the TTFT stagger doesn't already
        // pay; prompt 0 keeps its always-admitted guarantee (a single tail
        // longer than the cap still wipes, and now says so in the log).
        let hard = self.dflash_armed().then(chunk_tick_rows);
        // Lever 2: BATCH this tick's prefill spans into one weight-amortized pass
        // (PADDOCK_QWEN35_CHUNK_BATCH, DEFAULT-ON via apply_default_stack;
        // kill: PADDOCK_NO_CHUNK_BATCH - the kill pins the serial path, which
        // prefills the
        // chunked prompts one at a time -> re-reads the 256-expert MoE per prompt,
        // which tanks decode throughput on every mixed tick. Batching keeps
        // the chunk path's low TTFT while restoring decode.
        if std::env::var_os("PADDOCK_QWEN35_CHUNK_BATCH").is_some() {
            return self.advance_chunks_batched(cap, hard).map(logits);
        }
        let mut finished = Vec::new();
        let mut used = 0usize;
        while !self.chunked.is_empty() {
            // budget the tick, but guarantee forward progress on prompt 0
            if used > 0 && used >= cap {
                break;
            }
            let (slot, toks, done) = {
                let c = &self.chunked[0];
                (c.slot, c.tokens.clone(), c.done)
            };
            let len = toks.len();
            if used > 0 && hard.is_some_and(|h| used + len > h) {
                break; // the drafter's fusion cap: this tail goes next tick
            }
            if len - done > cap {
                break; // a long tail waits to head the queue (spans, above)
            }
            // Resume from `done` (prefill_begin already restored the prefix state
            // + KV); prefill only the tail, snapshot + insert. Bit-identical to an
            // un-chunked prefill - chunked prefill changes only when prompts run.
            let logits = self.prefill_slot_tail(slot, &toks, done)?;
            finished.push((slot, logits, len));
            self.chunked.remove(0);
            used += len;
        }
        Ok(logits(finished))
    }

    /// Lever 2 batched tick prefill: pop fresh (done==0) chunked prompts up to the
    /// row `cap` and prefill them in one weight-amortized pass (`prefill_batch_pass`),
    /// instead of the serial per-prompt loop. Resumed (done>0, prefix-cache-hit)
    /// prompts still go serial (they need the resume + snapshot path; rare on the
    /// salted concurrency workload this targets). v1: batched prompts are not
    /// re-inserted into the prefix cache (matches `prefill_batch_pass`); caching in
    /// the batched path is the same v2 follow-up as Lever 1. NOTE: this
    /// lane is not reachable under the default unified
    /// stack (verified live - the kill-switch A/B changed nothing); if it is ever
    /// re-defaulted, the missing insert means every prompt through it loses
    /// multi-turn prefix reuse, so the v2 follow-up becomes mandatory then.
    fn advance_chunks_batched(
        &mut self,
        cap: usize,
        hard: Option<usize>,
    ) -> Result<Vec<(usize, Vec<f32>, usize)>, GpuModelError> {
        let mut finished = Vec::new();
        // collect this tick's spans front-to-back up to the row cap. Fresh
        // spans are PEEKED (removed only after their pass succeeds) so a
        // fallible pass can't lose queue entries - the gemma4 destructive
        // take zombied its chunking slots on PoolExhausted.
        // Resumed spans keep their commit-after-success serial removal, so
        // after the walk the peeked fresh spans sit at queue positions
        // [0, batch.len()).
        let mut batch: Vec<(usize, Vec<u32>)> = Vec::new();
        let mut lens: Vec<usize> = Vec::new();
        let mut used = 0usize;
        let mut i = 0usize;
        while i < self.chunked.len() {
            if used > 0 && used >= cap {
                break;
            }
            let (slot, toks, done) = {
                let c = &self.chunked[i];
                (c.slot, c.tokens.clone(), c.done)
            };
            let len = toks.len();
            if used > 0 && hard.is_some_and(|h| used + len > h) {
                break; // the drafter's fusion cap: this tail goes next tick
            }
            if len - done > cap {
                break; // a long tail waits to head the queue (advance_chunks)
            }
            if done != 0 {
                // resumed span: keep the serial resume+snapshot path (correctness
                // over amortization for the rare cache-hit-mid-prompt case).
                let logits = self.prefill_slot_tail(slot, &toks, done)?;
                finished.push((slot, logits, len));
                self.chunked.remove(i);
                used += len;
                continue;
            }
            batch.push((slot, toks));
            lens.push(len);
            i += 1;
            used += len;
        }
        if !batch.is_empty() {
            // A lone admission rides the group pass when the f8t chunk arm
            // would engage there: the serial tail ran the q8/f8bs routes at
            // ~18.6 ms per c32 admission where the group pass does the same
            // rows on the decode tile plane (tc5r at 65+). Costs the v1
            // prefix-cache insert, like every grouped span here.
            let f8t_solo = batch.len() == 1
                && lens[0] <= f8t_chunk_rmax()
                && f8t_unified_on()
                && self.bs_f8t_attn.iter().any(Option::is_some);
            if batch.len() == 1 && !f8t_solo {
                // one fresh span: serial (its own tail already amortizes its
                // read) - and that path publishes on its own.
                let (slot, toks) = &batch[0];
                let logits = self.prefill_slot_tail(*slot, toks, 0)?;
                finished.push((*slot, logits, lens[0]));
                self.chunked.remove(0);
            } else {
                // Make the wave PUBLISH. `prefill_batch_pass`
                // prefills "fresh whole prompts, KV pos 0..take" and never
                // touches the radix, so every prompt riding it loses prefix
                // reuse - the v1 gap documented above. That gap went live when
                // PADDOCK_QWEN35_CHUNK_BATCH became default-on, which makes
                // the older "not reachable" note stale: instrumented, 96 of a
                // 112-request c16 leg ride this pass and their
                // `prefix_resume_begin` reads `ckpt None` forever (only the 16
                // warmup requests, which take the unified span path, resume).
                //
                // Teaching the pass per-share START positions would pull in the
                // resumed-span machinery `unified_launch_core` already owns.
                // Instead stop the wave at each prompt's DeltaNet checkpoint
                // boundary: after a pass over [0, cut) the slot's state sits
                // exactly on the boundary, so it snapshots exactly the way
                // `prefill_slot_tail_paged` does. The <=16-row remainder keeps
                // `done = cut` in the queue and rides the next tick, where the
                // unified tick fuses it with decode rather than paying a pass.
                // Splitting is bit-exact by the same argument the serial tail
                // makes: chunk splits don't change the math.
                //
                // One cut, not `ckpt_cuts`' two - a second stop costs a second
                // wave pass, and one resumable boundary is what a repeated
                // prompt needs. `>= MIN_CACHE_PREFIX` because a shallower
                // checkpoint is one `prefix_resume_begin` would refuse anyway.
                //
                // OPT-IN, DEFAULT off - this arm is MEASURED A NET LOSS and is
                // kept only as scaffolding for the real fix (below). Over 3
                // legs at c16 it costs both TTFT and throughput badly.
                // It works - legs 2-3 resume all 112 requests at `ckpt Some(176)`
                // and that saves ~126 ms of prefill - but splitting leaves a
                // <=16-row tail per prompt, and those tails are picked up by the
                // `done != 0` branch above, i.e. SERIALLY: 16 prompts x a full
                // weight pass ~= +850 ms, which swamps the win.
                //
                // So the split is the wrong shape. The fix has to leave no tail:
                // stage the DeltaNet state mid-pass at the boundary (the
                // `d_ckpt_stage` pattern `unified_finish_core` already uses for
                // its fused share, via `snapshot_staged_pool`) so the wave still
                // runs each prompt whole. That fights the wave's packed varlen
                // walk (one stage1+walk pair per layer), which is why it is a
                // separate piece of work rather than a widening of this one.
                // The alternative - per-share START positions so resumed spans
                // can ride the wave in a batch - is the `v2 follow-up` named
                // above and is the same machinery `unified_launch_core` owns.
                // NOTE: with publishing live the checkpoint pool BINDS (96
                // publishes/leg vs a 94-slot pool -> `attach_state` returns None
                // and nothing resumes); pool 256 is what made legs 2-3 resume.
                // That is why the earlier pool A/B read NULL: nothing published.
                let publish = self
                    .batch
                    .as_ref()
                    .is_some_and(|b| b.paged_prefix.is_some())
                    && paddock_models::dev_var_os!("PADDOCK_WAVE_PUBLISH").is_some();
                let cuts: Vec<Option<usize>> = lens
                    .iter()
                    .map(|&len| {
                        let c = ckpt_pos(len);
                        (publish && c >= min_cache_prefix() && c < len).then_some(c)
                    })
                    .collect();
                for (e, cut) in batch.iter_mut().zip(&cuts) {
                    if let Some(c) = *cut {
                        e.1.truncate(c);
                    }
                }
                let mut out: Vec<Vec<f32>> = vec![Vec::new(); batch.len()];
                let group: Vec<usize> = (0..batch.len()).collect();
                self.prefill_batch_pass(&batch, &group, &mut out, None)?;
                for i in 0..batch.len() {
                    let slot = batch[i].0;
                    match cuts[i] {
                        // ran whole: its last-token logits finish the prompt
                        None => finished.push((slot, std::mem::take(&mut out[i]), lens[i])),
                        // ran to the boundary: publish it, tail rides next tick
                        Some(c) => {
                            let toks = std::mem::take(&mut batch[i].1);
                            self.wave_publish_ckpt(slot, &toks, c)?;
                        }
                    }
                }
                // Finished prompts leave the queue; the split ones keep their
                // tail. Descending so the removals don't shift later indices.
                for i in (0..batch.len()).rev() {
                    match cuts[i] {
                        Some(c) => self.chunked[i].done = c,
                        None => {
                            self.chunked.remove(i);
                        }
                    }
                }
            }
        }
        Ok(finished)
    }
}
