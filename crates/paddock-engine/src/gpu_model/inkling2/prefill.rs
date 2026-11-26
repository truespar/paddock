//! Chunked prefill: an admission queues here and advances a budget of rows
//! a mixed tick, its chunk riding the live decode rows in the same pass
//! (one weight read for both), instead of every stream waiting out the
//! whole prompt. The queue runs in admission order; a prompt's chunk is a
//! run of its own slot like any other, so the pass needs nothing special.

use paddock_models::inkling2 as ik;

use super::GpuInkling2;
use super::forward::{Logits, Seg};
use crate::gpu_model::gpt_oss::GpuModelError;

/// A queued prompt: `done` of its `tokens` are in the slot. A picture prompt
/// carries its pictures' rows and the prompt row each starts at.
pub(crate) struct Pending {
    pub slot: usize,
    pub tokens: Vec<u32>,
    pub done: usize,
    pub pics: Vec<(usize, std::sync::Arc<super::vision::PictureRows>)>,
}

/// What a mixed tick hands back: the decode rows' logits (flat, input
/// order) and `(slot, last logits, prompt rows)` for every prompt it
/// finished.
pub(crate) type Mixed = (Vec<f32>, Vec<(usize, Vec<f32>, usize)>);

impl GpuInkling2 {
    /// Queue `tokens` as `slot`'s whole sequence: the slot starts over
    /// (blocks back, conv states zeroed) and fills over the next ticks.
    pub(crate) fn prefill_begin_impl(
        &mut self,
        slot: usize,
        tokens: Vec<u32>,
    ) -> Result<(), GpuModelError> {
        self.prefill_begin_cached(slot, tokens, &[], false)
            .map(|_| ())
    }

    /// [`Self::prefill_begin_impl`] through the prefix cache (`cache`, a
    /// text prompt): resume from its deepest checkpoint and checkpoint its
    /// cuts on the way (the scheduler's divergence `hints` among them). The
    /// rows served from the cache.
    pub(crate) fn prefill_begin_cached(
        &mut self,
        slot: usize,
        tokens: Vec<u32>,
        hints: &[usize],
        cache: bool,
    ) -> Result<usize, GpuModelError> {
        if slot >= self.kv.slots || tokens.is_empty() {
            return Err(GpuModelError::Unsupported(format!(
                "inkling2: prefill of {} tokens on slot {slot} ({} seated)",
                tokens.len(),
                self.kv.slots
            )));
        }
        if tokens.len() > self.max_ctx {
            return Err(GpuModelError::ContextExceeded {
                got: tokens.len(),
                max: self.max_ctx,
            });
        }
        self.queue.retain(|p| p.slot != slot);
        if self.queue.len() >= crate::service::max_chunks_inflight() {
            return Err(GpuModelError::Unsupported(
                "inkling2: chunked prefill queue is full".into(),
            ));
        }
        self.kv.release(slot);
        if let Some(p) = self.prefix.as_mut() {
            p.drop_staged(slot);
            p.reply_release(slot);
        }
        let resume = if cache {
            self.prefix_resume(slot, &tokens)?
        } else {
            0
        };
        if resume == 0 {
            self.kv.zero_conv(&self.exec, slot)?;
        }
        if let Some(p) = self.prefix.as_mut().filter(|_| cache) {
            p.cuts[slot] = super::prefix::Prefix::cuts_for(tokens.len(), hints)
                .into_iter()
                .filter(|&c| c > resume)
                .collect();
        }
        self.queue.push_back(Pending {
            slot,
            tokens,
            done: resume,
            pics: Vec::new(),
        });
        Ok(resume)
    }

    /// Drop `slot`'s queued prompt and its rows.
    pub(crate) fn prefill_abort_impl(&mut self, slot: usize) -> bool {
        let before = self.queue.len();
        self.queue.retain(|p| p.slot != slot);
        if self.queue.len() != before {
            self.kv.release(slot);
        }
        if let Some(p) = self.prefix.as_mut() {
            p.drop_staged(slot);
            p.reply_release(slot);
        }
        true
    }

    /// `(slot, depth, remaining)` a queued prompt, in the order a tick
    /// spends its rows.
    pub(crate) fn prefill_queue_impl(&self) -> Vec<(usize, usize, usize)> {
        self.queue
            .iter()
            .map(|p| (p.slot, p.done, p.tokens.len() - p.done))
            .collect()
    }

    /// Prompt rows a tick can carry beside `decode_rows` decode rows.
    pub(crate) fn prefill_tick_cap_impl(&self, decode_rows: usize) -> usize {
        self.scratch.rows.saturating_sub(decode_rows)
    }

    /// One mixed tick: `decodes[i] = (slot, token, pos)` plus up to
    /// `budget` prompt rows from the queue, in order, in one pass.
    pub(crate) fn forward_mixed_impl(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
    ) -> Result<Mixed, GpuModelError> {
        let want: Vec<(usize, usize)> = decodes.iter().map(|d| (d.0, d.2 as usize)).collect();
        self.sync_slots(&want)?;
        for &(slot, _, p) in decodes {
            if self.kv.lens.get(slot).copied() != Some(p as usize) {
                return Err(GpuModelError::Unsupported(format!(
                    "inkling2: slot {slot} decodes at {p} but holds {} rows",
                    self.kv.lens.get(slot).copied().unwrap_or(0)
                )));
            }
        }
        // the queue's chunks this tick (copied out: the pass borrows the
        // lane), each with whether it finishes its prompt
        let mut left = budget.min(self.prefill_tick_cap_impl(decodes.len()));
        let mut chunks: Vec<(usize, Vec<u32>, bool)> = Vec::new();
        // the chunks' picture rows, at their rows of the pass (decode rows
        // first, then the chunks in order)
        let mut overlay = Vec::new();
        // (slot, first position, rows, first pass row) for the prefix
        // cache's mid-pass captures
        let mut runs = Vec::new();
        let mut row = decodes.len();
        for p in &self.queue {
            if left == 0 {
                break;
            }
            let take = (p.tokens.len() - p.done).min(left);
            let end = p.done + take;
            overlay.extend(super::mm::spans(&p.pics, p.done, end, row));
            runs.push((p.slot, p.done, take, row));
            chunks.push((
                p.slot,
                p.tokens[p.done..end].to_vec(),
                end == p.tokens.len(),
            ));
            row += take;
            left -= take;
        }
        let toks: Vec<u32> = decodes.iter().map(|d| d.1).collect();
        let mut segs: Vec<Seg<'_>> = decodes
            .iter()
            .zip(&toks)
            .map(|(d, t)| Seg {
                slot: d.0,
                ids: std::slice::from_ref(t),
                want: Logits::Last,
            })
            .collect();
        segs.extend(chunks.iter().map(|(slot, ids, last)| Seg {
            slot: *slot,
            ids,
            want: if *last { Logits::Last } else { Logits::None },
        }));
        let v = ik::VOCAB_REAL;
        let nd = decodes.len();
        let mut dec = vec![0.0f32; nd * v];
        let mut fin: Vec<Option<Vec<f32>>> = vec![None; chunks.len()];
        self.overlay = overlay;
        let landing = self.prefix_arm(&runs)?;
        let r = self.forward_segs(&segs, &mut |si, _, l| {
            if si < nd {
                dec[si * v..(si + 1) * v].copy_from_slice(l);
            } else {
                fin[si - nd] = Some(l.to_vec());
            }
        });
        self.overlay.clear();
        if r.is_err()
            && let Some(p) = self.prefix.as_mut()
        {
            p.cap_n = 0;
            for &(_, _, idx) in &landing {
                p.radix.recycle_state(idx);
            }
        }
        r?;
        self.prefix_land(&landing)?;
        // advance the queue; the finished leave it
        let mut finished = Vec::new();
        for ((slot, ids, _), logits) in chunks.iter().zip(fin) {
            let p = self
                .queue
                .iter_mut()
                .find(|p| p.slot == *slot)
                .expect("a chunk's prompt is queued");
            p.done += ids.len();
            if let Some(l) = logits {
                finished.push((*slot, l, p.tokens.len()));
            }
        }
        // a landed text prompt goes into the prefix cache
        let landed: Vec<(usize, Vec<u32>)> = self
            .queue
            .iter()
            .filter(|p| p.done >= p.tokens.len() && p.pics.is_empty())
            .map(|p| (p.slot, p.tokens.clone()))
            .collect();
        for (slot, tokens) in landed {
            self.prefix_insert(slot, &tokens);
        }
        self.queue.retain(|p| p.done < p.tokens.len());
        Ok((dec, finished))
    }
}
