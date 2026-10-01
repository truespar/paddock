//! Chunked prefill for Flash-Next: the scheduler's MIXED tick.
//!
//! Before this the lane served admissions through the classic blocking
//! prefill (`forward_prefill_batch`): every live stream froze until the whole
//! admitted wave had walked. Now a prompt is queued at `prefill_begin` and
//! advanced a budgeted span per tick in the SAME walk as the live decode rows
//! (Sarathi-Serve's stall-free batching; one weight stream per tick, which on
//! a 512-routed-expert model is most of a tick's cost).
//!
//! The walk is the prefill wave's (`Phase::PrefillRuns`): every row-parallel
//! op runs once over all rows. The decode rows lead - one token each for
//! distinct slots, at their own positions - and `walk_lead` routes their
//! three carried-state ops (GDN conv step, GDN recurrence, PLE ring step)
//! and their attention to the batched DECODE entries, one launch for all of
//! them and the class a decode tick runs; the prompt spans behind them keep
//! the per-run arms, continuing each prompt's carried state from its cursor
//! (the `row0` arms a prefix-cache resume already uses). The head gives one
//! row per run, so decode logits sit at rows `[0, nd)`.
//!
//! Span geometry: FIFO over the queue under the tick's row budget. A span
//! stops at its prompt's next checkpoint cut (`prefix::ckpt_cuts`, the last
//! two page boundaries) so the carried state can be snapshotted there - the
//! same chunk geometry the wave and the single-slot walks leave, which is
//! what makes a later resume bit-identical to the cold run - and the next
//! prompt takes the rows it leaves. A tick with no decode rows and one span
//! is the single-slot walk (captured, fork-enabled, the fastest prefill
//! shape), and when that span finishes its prompt the cuts inside it are
//! taken IN the walk (`walk_cuts`), exactly as `prefill_from` does: an idle
//! server's lone prompt runs the walk it always did. That walk starts where
//! an exact re-send can start it again (`walk_origin`): the resume point, or
//! a checkpoint filed at the walk boundary it starts on.
//!
//! The row budget comes from the scheduler, which is also where the
//! long-context tick pacer (crate::pacing) sizes it; `prefill_queue` and
//! `prefill_tick_cap` are its view.

use super::{Phase, Qwen4ExpGpu, Run};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::prefix_cache::BLOCK_TOKENS;

/// One prompt queued for chunked prefill.
pub(super) struct ChunkedPrefill {
    slot: usize,
    tokens: Vec<u32>,
    /// next row to walk; starts at the prefix-cache resume point
    cursor: usize,
}

/// One prompt's span in a walk.
#[derive(Clone, Copy)]
pub(super) struct Span {
    slot: usize,
    from: usize,
    to: usize,
    /// the walk row its first token sits at
    off: usize,
    /// its row in `d_out` (the head's one row per run)
    out_row: usize,
    /// it finishes its prompt
    finishes: bool,
    /// its end is a checkpoint cut (snapshot there)
    at_cut: bool,
}

/// A walked mixed tick, between the walk and its commit: `d_out` rows
/// `[0, nd)` hold the decode rows' logits and each finishing span's logits
/// sit at its `out_row`. Read what you need, then `mixed_commit`.
pub(super) struct MixedWalk {
    pub nd: usize,
    spans: Vec<Span>,
    /// in-walk checkpoints reserved for a single-slot finishing span
    reserved: Vec<(usize, u32)>,
}

impl MixedWalk {
    /// (slot, `d_out` row, prompt rows) per span that finished its prompt.
    pub(super) fn finishers<'a>(
        &'a self,
        g: &'a Qwen4ExpGpu,
    ) -> impl Iterator<Item = (usize, usize, usize)> + 'a {
        self.spans.iter().filter(|s| s.finishes).map(move |s| {
            let len = g
                .chunked
                .iter()
                .find(|c| c.slot == s.slot)
                .map_or(s.to, |c| c.tokens.len());
            (s.slot, s.out_row, len)
        })
    }
}

impl Qwen4ExpGpu {
    /// Chunked prefill is served once the pack carries the runs recurrence
    /// the mixed walk's prompt spans need. `PADDOCK_Q38FN_NO_CHUNKED=1` keeps
    /// the classic blocking wave (A/B).
    pub(super) fn chunked_supported(&self) -> bool {
        super::super::chunked_prefill_enabled() && self.exec.has_gated_delta_recurrent_runs()
    }

    /// Queue `tokens` for chunked prefill on `slot`. The admission prologue
    /// runs now - the prefix-cache consult restores the slot's carried state
    /// to the resume point (or the slot is reset), the drafter and the
    /// n-gram stream are primed - so a tick only has to move rows.
    pub(super) fn prefill_begin_impl(
        &mut self,
        slot: usize,
        tokens: Vec<u32>,
    ) -> Result<(), GpuModelError> {
        if slot >= self.slots {
            return Err(GpuModelError::Unsupported(format!(
                "slot {slot} but this instance carries {}",
                self.slots
            )));
        }
        let n = tokens.len();
        if n == 0 || n > self.max_tokens {
            return Err(GpuModelError::Unsupported(format!(
                "prompt of {n} tokens; this lane is sized for 1..={}",
                self.max_tokens
            )));
        }
        // a queued entry for this slot is STALE (the scheduler keeps one
        // prompt per live slot - a duplicate means the old request died and
        // the slot was reused): evict it rather than wedge the slot
        self.chunked.retain(|c| c.slot != slot);
        if self.chunked.len() >= crate::service::max_chunks_inflight() {
            return Err(GpuModelError::Unsupported(
                "chunked prefill queue is full".into(),
            ));
        }
        let start = self.prefix_resume(slot, &tokens)?;
        if start == 0 {
            self.reset_slot(slot)?;
        }
        self.mtp_begin(slot, start);
        self.stream[slot] = vec![self.cfg.bos_id as i64; 2];
        self.stream[slot].extend(tokens.iter().map(|&i| i as i64));
        self.pos[slot] = start;
        self.chunked.push(ChunkedPrefill {
            slot,
            tokens,
            cursor: start,
        });
        Ok(())
    }

    /// Drop slot `slot`'s queued prompt (the client hung up). Nothing is in
    /// flight between ticks here, so this never has to defer; the slot's
    /// carried state is reset or restored by its next admission.
    pub(super) fn prefill_abort_impl(&mut self, slot: usize) -> bool {
        let n = self.chunked.len();
        self.chunked.retain(|c| c.slot != slot);
        if self.chunked.len() == n {
            return false;
        }
        self.mtp_clear(slot);
        true
    }

    /// `Generator::prefill_queue`: FIFO, each prompt from its cursor.
    pub(super) fn prefill_queue_impl(&self) -> Vec<(usize, usize, usize)> {
        self.chunked
            .iter()
            .map(|c| (c.slot, c.cursor, c.tokens.len() - c.cursor))
            .collect()
    }

    /// `Generator::prefill_tick_cap`: the prompt rows one tick takes. With
    /// decode rows aboard, the lane's elected span (`chunk_rows`): each rider
    /// waits the whole tick for its token. With none, the whole walk - an
    /// idle tick has no one to stall, and a burst or a lone prompt walks as
    /// the wave and the single-slot walk always did.
    pub(super) fn prefill_tick_cap_impl(&self, decode_rows: usize) -> usize {
        // a walk carries at most walk_rows rows (the scratch's size); with
        // a context longer than that, the riderless tick walks it in pieces
        let room = self.walk_rows.saturating_sub(decode_rows);
        if decode_rows == 0 {
            room
        } else {
            room.min(super::super::chunk_rows())
        }
    }

    /// Where `slot`'s span may end this tick: at most `room` rows past its
    /// cursor, and never past its next checkpoint cut - unless `inwalk`, when
    /// a span that FINISHES the prompt takes the cuts inside itself. Only
    /// from a page boundary: the walk's start is where a re-send starts it
    /// again (`walk_origin`), which takes a checkpoint there; a cursor that
    /// budgeted ticks beside decode rows left mid-page stops at the cuts.
    fn span_end(&self, qi: usize, room: usize, inwalk: bool) -> (usize, bool) {
        let c = &self.chunked[qi];
        let len = c.tokens.len();
        let inwalk = inwalk && c.cursor.is_multiple_of(BLOCK_TOKENS);
        // and never across an absolute multiple of walk_rows: the single-slot
        // walk splits there, so both walk a long prompt in the same pieces
        let bound = (c.cursor / self.walk_rows + 1) * self.walk_rows;
        let to = (c.cursor + room).min(len).min(bound);
        if inwalk && to == len {
            return (to, false);
        }
        let cut = if self.prefix.is_some() {
            super::super::prefix::ckpt_cuts(len)
                .into_iter()
                .filter(|&x| x > c.cursor && x < len)
                .min()
        } else {
            None
        };
        match cut {
            Some(ct) if ct <= to => (ct, true),
            _ => (to, false),
        }
    }

    /// Walk one mixed tick: `decodes` (slot, token, position) plus up to
    /// `budget` prompt rows, FIFO. Leaves the logits in `d_out` (see
    /// [`MixedWalk`]); the caller reads them and then calls
    /// [`Self::mixed_commit`].
    pub(super) fn mixed_walk(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
    ) -> Result<MixedWalk, GpuModelError> {
        let nd = decodes.len();
        let mut seen = vec![false; self.slots];
        for &(sl, _, p) in decodes {
            if sl >= self.slots {
                return Err(GpuModelError::Unsupported(format!(
                    "slot {sl} but this instance carries {}",
                    self.slots
                )));
            }
            if std::mem::replace(&mut seen[sl], true) {
                return Err(GpuModelError::Unsupported(format!(
                    "slot {sl} appears twice in one mixed tick"
                )));
            }
            if self.chunked.iter().any(|c| c.slot == sl) {
                return Err(GpuModelError::Unsupported(format!(
                    "slot {sl} decodes while its prompt is still queued"
                )));
            }
            // checked rather than trusted, as in `forward_batch`: a wrong-
            // position KV read gives plausible text no gate would catch
            if self.pos[sl] == 0 || self.pos[sl] != p as usize || self.pos[sl] >= self.max_tokens {
                return Err(GpuModelError::Unsupported(format!(
                    "slot {sl}: scheduler says position {p}, model is at {} (max {})",
                    self.pos[sl], self.max_tokens
                )));
            }
        }
        let room = budget.min(self.prefill_tick_cap_impl(nd));
        // a lone prompt on an idle tick walks alone, and may take its cuts in
        // the walk; a burst walks together (the wave)
        let inwalk = nd == 0
            && self.chunked.len() == 1
            && self.prefix.is_some()
            && super::super::inwalk_ckpt_enabled()
            && self.exec.has_gated_delta_recurrent_pn()
            && super::super::gdn_pn_enabled();
        let mut spans: Vec<Span> = Vec::new();
        let mut off = nd;
        let mut left = room;
        for qi in 0..self.chunked.len() {
            if left == 0 {
                break;
            }
            let (to, at_cut) = self.span_end(qi, left, inwalk);
            let c = &self.chunked[qi];
            let len = to - c.cursor;
            spans.push(Span {
                slot: c.slot,
                from: c.cursor,
                to,
                off,
                out_row: nd + spans.len(),
                finishes: to == c.tokens.len(),
                at_cut,
            });
            off += len;
            left -= len;
            // a span that ran out of budget ends the tick's prompt rows; one
            // that stopped at a cut or finished leaves the rest to the next
            if !at_cut && to < c.tokens.len() {
                break;
            }
        }
        if spans.is_empty() {
            // no prompt rows this tick (the scheduler can park finished
            // prompts in its chunking set a tick longer): the decode tick,
            // captured graph and all
            if nd > 0 {
                let rows: Vec<(usize, u32)> = decodes.iter().map(|&(s, t, _)| (s, t)).collect();
                self.decode_batch_walk(&rows)?;
            }
            return Ok(MixedWalk {
                nd,
                spans,
                reserved: Vec::new(),
            });
        }
        if nd == 0 && spans.len() == 1 {
            return self.mixed_walk_single(spans[0], inwalk);
        }

        // the runs walk: decode rows first (length-1 runs at their slots'
        // positions), then the prompt spans
        let mut runs: Vec<Run> = Vec::with_capacity(nd + spans.len());
        let mut ids: Vec<u32> = Vec::with_capacity(off);
        for (i, &(sl, t, _)) in decodes.iter().enumerate() {
            runs.push(Run {
                slot: sl,
                off: i,
                len: 1,
                row0: self.pos[sl],
            });
            ids.push(t);
            // the n-gram hash reads the row's own token off its stream
            self.stream[sl].push(t as i64);
        }
        for s in &spans {
            runs.push(Run {
                slot: s.slot,
                off: s.off,
                len: s.to - s.from,
                row0: s.from,
            });
            let c = self
                .chunked
                .iter()
                .find(|c| c.slot == s.slot)
                .expect("span of a queued prompt");
            ids.extend_from_slice(&c.tokens[s.from..s.to]);
        }
        let rows = off;
        let walked = self.stage_inputs_runs_ids(&ids, &runs, nd).and_then(|_| {
            self.cur_slots = runs
                .iter()
                .flat_map(|r| std::iter::repeat_n(r.slot, r.len))
                .collect();
            self.walk_qsa = self.qsa_for_runs(&runs);
            self.cur_runs = runs;
            self.walk_lead = nd;
            let w = self.device_walk(rows, Phase::PrefillRuns);
            self.walk_lead = 0;
            self.cur_runs.clear();
            w
        });
        if let Err(e) = walked {
            // put the streams back: no position moved
            for &(sl, _, _) in decodes {
                self.stream[sl].pop();
            }
            return Err(e);
        }
        let rows_dec: Vec<(usize, u32)> = decodes.iter().map(|&(s, t, _)| (s, t)).collect();
        for &(sl, _) in &rows_dec {
            self.pos[sl] += 1;
        }
        // before any prompt seeds the drafter: a head pass rewrites `d_h`
        // from row 0, where the decode rows' streams sit
        self.mtp_note_rows(&rows_dec)?;
        self.reply_after_rows(&rows_dec)?;
        Ok(MixedWalk {
            nd,
            spans,
            reserved: Vec::new(),
        })
    }

    /// The single-slot walk (`Phase::Prefill`) over one span, continuing the
    /// slot's carried state from the span's first row; a finishing span
    /// takes its prompt's cuts in the walk when `inwalk`.
    fn mixed_walk_single(&mut self, s: Span, inwalk: bool) -> Result<MixedWalk, GpuModelError> {
        let qi = self
            .chunked
            .iter()
            .position(|c| c.slot == s.slot)
            .expect("span of a queued prompt");
        let len = self.chunked[qi].tokens.len();
        let mut reserved: Vec<(usize, u32)> = Vec::new();
        // span_end's test: a span took its cuts inside only from a page
        let inwalk = inwalk && s.from.is_multiple_of(BLOCK_TOKENS);
        if inwalk
            && s.finishes
            && let Some(pc) = self.prefix.as_mut()
        {
            for c in super::super::prefix::ckpt_cuts(len) {
                if c > s.from
                    && c < len
                    && let Some(idx) = pc.reserve_ckpt(&mut self.pages)
                {
                    reserved.push((c, idx));
                }
            }
        }
        // each cut into its staging blob, committed into its checkpoint's
        // pages right after the walk
        self.walk_cuts = reserved
            .iter()
            .enumerate()
            .map(|(blob, &(c, _))| (c - s.from, blob as u32))
            .collect();
        self.walk_ckpts = reserved.iter().map(|&(_, idx)| idx).collect();
        let ids = std::mem::take(&mut self.chunked[qi].tokens);
        let walked = if reserved.is_empty() {
            Ok(())
        } else {
            self.walk_origin(s.slot, &ids, s.from)
        }
        .and_then(|()| self.walk_span_dev(s.slot, &ids, s.from, s.to));
        self.chunked[qi].tokens = ids;
        self.walk_cuts.clear();
        let walked = walked.and_then(|()| self.commit_walk_cuts());
        if let Err(e) = walked {
            self.walk_ckpts.clear();
            self.recycle_reserved(&reserved);
            return Err(e);
        }
        Ok(MixedWalk {
            nd: 0,
            spans: vec![Span {
                off: 0,
                out_row: 0,
                ..s
            }],
            reserved,
        })
    }

    /// Commit a walked tick once its logits have been read: feed the
    /// drafter the prompts' rows, file each span's pages and checkpoints,
    /// advance the cursors and retire the finished prompts, then drain the
    /// drafter's stash (its head passes reuse the walk's scratch).
    pub(super) fn mixed_commit(&mut self, w: MixedWalk) -> Result<(), GpuModelError> {
        for s in &w.spans {
            let qi = self
                .chunked
                .iter()
                .position(|c| c.slot == s.slot)
                .expect("span of a queued prompt");
            let ids = std::mem::take(&mut self.chunked[qi].tokens);
            let res = (|| {
                self.mtp_seed(s.slot, s.off, s.from, s.to, &ids)?;
                self.pos[s.slot] = s.to;
                if s.finishes {
                    self.prefix_publish(s.slot, &ids, s.to, false)?;
                    self.attach_cuts(s.slot, &ids, &w.reserved);
                    self.reply_track_admit(s.slot);
                } else if s.at_cut {
                    self.prefix_cut(s.slot, &ids, s.to)?;
                }
                Ok::<(), GpuModelError>(())
            })();
            self.chunked[qi].tokens = ids;
            self.chunked[qi].cursor = s.to;
            res?;
        }
        self.chunked.retain(|c| c.cursor < c.tokens.len());
        self.mtp_flush()
    }
}
