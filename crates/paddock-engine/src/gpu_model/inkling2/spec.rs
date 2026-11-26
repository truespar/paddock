//! Speculative rounds with the DSpark draft: the verify pass, and the one
//! piece of state a rejected draft leaves behind.
//!
//! K / V, the draft's context and the feature rings are kept by position,
//! so rows past an accepted prefix are rewritten before anything reads them.
//! The short convs are not: their carried rows advance with every row of a
//! verify run. A verify pass therefore stashes each conv's carried rows and
//! the run's inputs (slot 1000), and a slot asked for at a position inside
//! its last verify run takes the carried rows after the accepted prefix
//! (slot 1001) - the greedy round learns that position from the slot's next
//! call, the sampled round from `spec_commit`.

use paddock_models::inkling2 as ik;

use super::GpuInkling2;
use super::draft::BLOCK;
use super::forward::{Logits, PassOut, Seg};
use super::kv::{CONV_ATTN, CONV_K, CONV_MLP, CONV_V};
use super::{KV, MAX_VERIFY};
use crate::gpu_model::gpt_oss::GpuModelError;

impl GpuInkling2 {
    /// Bring each `(slot, pos)` to hold exactly `pos` rows: equal is a no-op,
    /// a position inside the slot's last verify run rolls its convs back to
    /// the accepted prefix; anything else is an error (the convs cannot
    /// rewind further).
    pub(crate) fn sync_slots(&mut self, want: &[(usize, usize)]) -> Result<(), GpuModelError> {
        let mut items = Vec::new();
        // (slot, rows) a verify run settled at, for the prefix cache once
        // the convs are back there
        let mut settled = Vec::new();
        for &(slot, pos) in want {
            let len = self.kv.lens[slot];
            if len == pos {
                if self.kv.spec_run[slot].take().is_some() {
                    settled.push((slot, pos));
                }
                continue;
            }
            match self.kv.spec_run.get(slot).copied().flatten() {
                Some((p0, l)) if len == p0 + l && pos >= p0 && pos < len => {
                    items.extend([slot as u32, (pos - p0) as u32]);
                    self.kv.lens[slot] = pos;
                    self.kv.spec_run[slot] = None;
                    settled.push((slot, pos));
                }
                _ => {
                    return Err(GpuModelError::Unsupported(format!(
                        "inkling2: slot {slot} asked at {pos} but holds {len} rows"
                    )));
                }
            }
        }
        if items.is_empty() {
            for (slot, pos) in settled {
                self.reply_settled(slot, pos)?;
            }
            return Ok(());
        }
        let exec = self.exec.clone();
        let n = items.len() / 2;
        let kv = &mut self.kv;
        let buf = kv
            .spec_items
            .as_mut()
            .ok_or_else(|| GpuModelError::Unsupported("inkling2: no verify stash".into()))?;
        exec.upload_u32(&items, buf)?;
        for (planes, stash) in kv.conv.iter_mut().zip(&kv.stash) {
            for c in [CONV_K, CONV_V, CONV_ATTN, CONV_MLP] {
                let width = if c < CONV_ATTN { KV } else { ik::WIDTH };
                exec.inkling2_conv_restore(&stash[c], &mut planes[c], width, (buf, n))?;
            }
        }
        for (slot, pos) in settled {
            self.reply_settled(slot, pos)?;
        }
        Ok(())
    }

    /// The verify pass of a round: `reqs[i] = (slot, pos, chunk)` with the
    /// slot synced to `pos` first; every chunk row's logits, flat in request
    /// order. The runs stay open until the slots are next asked for.
    pub(crate) fn verify(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
    ) -> Result<Vec<f32>, GpuModelError> {
        let want: Vec<(usize, usize)> = reqs.iter().map(|r| (r.0, r.1)).collect();
        self.sync_slots(&want)?;
        if let Some(r) = reqs
            .iter()
            .find(|r| r.2.is_empty() || r.2.len() > MAX_VERIFY)
        {
            return Err(GpuModelError::Unsupported(format!(
                "inkling2: a verify chunk of {} rows (1..={MAX_VERIFY})",
                r.2.len()
            )));
        }
        let v = ik::VOCAB_REAL;
        let mut base = Vec::with_capacity(reqs.len());
        let mut rows = 0;
        for r in reqs {
            base.push(rows);
            rows += r.2.len();
        }
        let mut out = vec![0.0f32; rows * v];
        let segs: Vec<Seg<'_>> = reqs
            .iter()
            .map(|r| Seg {
                slot: r.0,
                ids: &r.2,
                want: Logits::All,
            })
            .collect();
        // a round's rows ride a pass a chunk at a time
        let chunk = self.scratch.rows;
        let mut i = 0;
        while i < segs.len() {
            let mut j = i;
            let mut n = 0;
            while j < segs.len() && n + segs[j].ids.len() <= chunk {
                n += segs[j].ids.len();
                j += 1;
            }
            let part = &segs[i..j];
            let offs = &base[i..j];
            self.forward_segs_as(part, true, &mut |si, r, l| {
                let at = (offs[si] + r) * v;
                out[at..at + v].copy_from_slice(l);
            })?;
            i = j;
        }
        for r in reqs {
            self.kv.spec_run[r.0] = Some((r.1, r.2.len()));
        }
        self.last_verify = reqs.iter().map(|r| (r.0, r.1)).collect();
        Ok(out)
    }

    /// [`Self::verify`] with every row sampled on the device by its plan
    /// (`params` rows x 4 words, sample_rows'), only the ids read back: a
    /// round's rows of 200K logits never cross to the host.
    pub(crate) fn verify_picks(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
        params: &[u32],
    ) -> Result<Vec<u32>, GpuModelError> {
        let want: Vec<(usize, usize)> = reqs.iter().map(|r| (r.0, r.1)).collect();
        self.sync_slots(&want)?;
        if let Some(r) = reqs
            .iter()
            .find(|r| r.2.is_empty() || r.2.len() > MAX_VERIFY)
        {
            return Err(GpuModelError::Unsupported(format!(
                "inkling2: a verify chunk of {} rows (1..={MAX_VERIFY})",
                r.2.len()
            )));
        }
        let segs: Vec<Seg<'_>> = reqs
            .iter()
            .map(|r| Seg {
                slot: r.0,
                ids: &r.2,
                want: Logits::All,
            })
            .collect();
        let mut ids = Vec::with_capacity(params.len() / 4);
        let chunk = self.scratch.rows;
        let mut i = 0;
        while i < segs.len() {
            let mut j = i;
            let mut n = 0;
            while j < segs.len() && n + segs[j].ids.len() <= chunk {
                n += segs[j].ids.len();
                j += 1;
            }
            self.forward_segs_out(&segs[i..j], true, &mut PassOut::Picks(params, &mut ids))?;
            i = j;
        }
        for r in reqs {
            self.kv.spec_run[r.0] = Some((r.1, r.2.len()));
        }
        self.last_verify = reqs.iter().map(|r| (r.0, r.1)).collect();
        Ok(ids)
    }

    /// Drafts for `pendings[i] = (slot, pending token)` (each slot synced to
    /// its pending position), at most `k` each; slots too short or too close
    /// to the context's end get none.
    pub(crate) fn draft_for(
        &mut self,
        pendings: &[(usize, u32)],
        k: usize,
    ) -> Result<Vec<Vec<u32>>, GpuModelError> {
        let mut out = vec![Vec::new(); pendings.len()];
        let mut items = Vec::new();
        let mut which = Vec::new();
        for (i, &(slot, tok)) in pendings.iter().enumerate() {
            let p = self.kv.lens.get(slot).copied().unwrap_or(0);
            if p == 0 || p + BLOCK > self.max_ctx || self.kv.spec_run[slot].is_some() {
                continue;
            }
            items.push((slot, tok, p));
            which.push(i);
        }
        if items.is_empty() || k == 0 {
            return Ok(out);
        }
        let drafts = self.draft_round(&items, k)?;
        for (d, i) in drafts.into_iter().zip(which) {
            out[i] = d;
        }
        Ok(out)
    }
}
