//! The lane behind the service's `Generator`. The single stream is slot 0;
//! `enable_batch` seats as many slots as the plan backs, and a decode tick
//! is one forward over a row a live slot. Prompts run whole on their own
//! slot, or through the chunked queue riding decode ticks (prefill.rs) -
//! the conv states carry a sequence forward and cannot rewind, so every row
//! must continue its slot exactly where it stands.

use paddock_models::inkling2 as ik;

use super::GpuInkling2;
use super::forward::{Logits, Seg};
use super::kv::{self, Kv};
use crate::generator::{GenError, Generator};
use crate::gpu::GpuError;
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::kv_plan;

fn gen_err(e: GpuModelError) -> GenError {
    match e {
        GpuModelError::PoolExhausted => GenError::PoolExhausted,
        GpuModelError::Config(msg) => GenError::Config(msg),
        other => GenError::Backend(other.to_string()),
    }
}

/// Fixed charges the plan keeps out of the pool: the CUDA graph / stream
/// pool's working room.
const VRAM_HEADROOM: u64 = 1 << 30;

impl GpuInkling2 {
    /// Seat up to `max_batch` slots: the KV store is re-planned against the
    /// grant (sliding rings and conv states per slot, the full visits'
    /// blocks from one pool). Returns the slots seated.
    pub fn enable_batch_impl(&mut self, max_batch: usize) -> Result<usize, GpuModelError> {
        if max_batch == 0 {
            return Err(GpuModelError::Config("inkling2: max_batch 0".into()));
        }
        if max_batch > self.scratch.rows {
            return Err(GpuModelError::BatchTooLarge {
                got: max_batch,
                max: self.scratch.rows,
            });
        }
        let dtype = self.kv.dtype;
        let bps = self.max_ctx.div_ceil(16);
        // the single-slot store goes first: the plan reads what is free
        // without it (and the tick graphs baked its addresses)
        self.graphs.clear();
        self.graph_seen.clear();
        self.queue.clear();
        let draft = self.draft.is_some();
        self.kv = Kv::alloc(&self.exec, 1, 16, self.chunk, dtype, 1, draft)?;
        self.exec.synchronize()?;
        self.exec.trim_mem_pool();
        let grant = self
            .exec
            .vram_headroom()
            .ok_or_else(|| GpuError::Driver("no free-VRAM reading".into()))?;
        let demand = kv_plan::Demand {
            family: "inkling2",
            max_ctx: self.max_ctx,
            slots: max_batch,
            blocks_per_slot: bps,
            block_bytes: kv::block_bytes(dtype, draft),
            per_slot_bytes: kv::slot_bytes(dtype, self.chunk, bps, draft),
            // every slot must hold a whole prompt chunk, or admission
            // deadlocks on its own first chunk
            floor_blocks_per_slot: self.chunk.div_ceil(16),
            floor_blocks_min: 256,
            reserves: vec![
                kv_plan::Reserve::new("graph/scratch slack", VRAM_HEADROOM),
                kv_plan::Reserve::new(
                    "prefix checkpoints",
                    super::prefix::reserve_bytes(&self.exec, max_batch, dtype, draft),
                ),
            ],
            ..Default::default()
        };
        let plan = match demand.plan(grant) {
            Ok(p) => p,
            Err(e) => {
                // the single stream keeps a whole-context store
                self.kv = Kv::alloc(&self.exec, 1, self.max_ctx, self.chunk, dtype, bps, draft)?;
                return Err(GpuModelError::WontFit(e.message));
            }
        };
        plan.report(&demand, grant);
        self.kv = Kv::alloc(
            &self.exec,
            plan.slots,
            self.max_ctx,
            self.chunk,
            dtype,
            plan.pool_blocks,
            draft,
        )?;
        if let Some(d) = self.draft.as_mut() {
            d.alloc_scratch(&self.exec, plan.slots, self.chunk)?;
        }
        self.prefix = super::prefix::Prefix::build(&self.exec, plan.slots, dtype, draft)?;
        tracing::info!(
            "inkling2 batch: {} slots, {} pool blocks ({} tokens), KV class {dtype}, {:.2} GiB",
            plan.slots,
            plan.pool_blocks,
            plan.token_capacity,
            self.kv.bytes() as f64 / (1u64 << 30) as f64
        );
        Ok(plan.slots)
    }

    /// Prefill `tokens` as `slot`'s whole sequence (from position 0) in
    /// chunks; the last row's logits.
    pub fn prefill_slot(&mut self, slot: usize, tokens: &[u32]) -> Result<Vec<f32>, GpuModelError> {
        let mut out = self.prefill_slots(&[(slot, tokens)])?;
        Ok(out.pop().expect("one prompt"))
    }

    /// Prefill several `(slot, prompt)` as whole sequences, packed into
    /// shared passes of up to a chunk of rows in order (a long prompt
    /// continues in the next pass); each prompt's last logits, in order.
    pub fn prefill_slots(
        &mut self,
        items: &[(usize, &[u32])],
    ) -> Result<Vec<Vec<f32>>, GpuModelError> {
        let mut seen = vec![false; self.kv.slots];
        for &(slot, tokens) in items {
            if slot >= self.kv.slots || std::mem::replace(&mut seen[slot], true) {
                return Err(GpuModelError::Unsupported(format!(
                    "inkling2: slot {slot} twice or past the {} seated",
                    self.kv.slots
                )));
            }
            if tokens.is_empty() {
                return Err(GpuModelError::Unsupported("inkling2: empty prompt".into()));
            }
            if tokens.len() > self.max_ctx {
                return Err(GpuModelError::ContextExceeded {
                    got: tokens.len(),
                    max: self.max_ctx,
                });
            }
        }
        for &(slot, _) in items {
            self.queue.retain(|p| p.slot != slot);
            self.kv.release(slot);
            self.kv.zero_conv(&self.exec, slot)?;
        }
        let rows = self.scratch.rows;
        let mut done = vec![0usize; items.len()];
        let mut out = vec![Vec::new(); items.len()];
        loop {
            let mut left = rows;
            let mut segs = Vec::new();
            let mut which = Vec::new();
            for (i, &(slot, tokens)) in items.iter().enumerate() {
                if left == 0 {
                    break;
                }
                let take = (tokens.len() - done[i]).min(left);
                if take == 0 {
                    continue;
                }
                let end = done[i] + take;
                segs.push(Seg {
                    slot,
                    ids: &tokens[done[i]..end],
                    want: if end == tokens.len() {
                        Logits::Last
                    } else {
                        Logits::None
                    },
                });
                which.push(i);
                left -= take;
            }
            if segs.is_empty() {
                return Ok(out);
            }
            self.forward_segs(&segs, &mut |si, _, l| out[which[si]] = l.to_vec())?;
            for (si, &i) in which.iter().enumerate() {
                done[i] += segs[si].ids.len();
            }
        }
    }

    /// One decode tick: row i feeds `tokens[i]` at `positions[i]` to slot
    /// i. A row at position 0 is a hole (an empty or not yet prefilled
    /// slot): it runs nothing and its logits row stays zero.
    pub fn decode_tick(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
    ) -> Result<Vec<f32>, GpuModelError> {
        let v = ik::VOCAB_REAL;
        let mut out = vec![0.0f32; tokens.len() * v];
        let mut segs = Vec::with_capacity(tokens.len());
        let mut rows = Vec::with_capacity(tokens.len());
        // a slot asked inside its last verify run rolls back to it first
        let want: Vec<(usize, usize)> = positions
            .iter()
            .enumerate()
            .filter(|&(i, &p)| p != 0 && i < self.kv.slots)
            .map(|(i, &p)| (i, p as usize))
            .collect();
        self.sync_slots(&want)?;
        for (i, (t, &p)) in tokens.iter().zip(positions).enumerate() {
            if p == 0 {
                continue;
            }
            if i >= self.kv.slots || self.kv.lens[i] != p as usize {
                return Err(GpuModelError::Unsupported(format!(
                    "inkling2: slot {i} decodes at {p} but holds {} rows",
                    self.kv.lens.get(i).copied().unwrap_or(0)
                )));
            }
            segs.push(Seg {
                slot: i,
                ids: std::slice::from_ref(t),
                want: Logits::Last,
            });
            rows.push(i);
        }
        if segs.is_empty() {
            return Ok(out);
        }
        self.forward_segs(&segs, &mut |si, _, l| {
            let r = rows[si];
            out[r * v..(r + 1) * v].copy_from_slice(l);
        })?;
        Ok(out)
    }

    fn check_room(&self, n: usize) -> Result<(), GenError> {
        if self.len() + n > self.max_ctx {
            return Err(GenError::Backend(format!(
                "context full: {} + {n} tokens at max_ctx {}",
                self.len(),
                self.max_ctx
            )));
        }
        Ok(())
    }
}

impl Generator for GpuInkling2 {
    fn device_mem_used(&self) -> Option<u64> {
        self.exec.process_mem_used()
    }

    fn weights_mem_bytes(&self) -> Option<u64> {
        Some(self.weights_bytes)
    }

    fn kv_mem_bytes(&self) -> Option<u64> {
        Some(self.kv.bytes())
    }

    fn release_idle_memory(&mut self) {
        self.exec.trim_mem_pool();
    }

    fn reset(&mut self) {
        // a failure here leaves the next forward on stale conv taps, so say
        // it loudly rather than serving them
        if let Err(e) = GpuInkling2::reset(self) {
            tracing::error!("inkling2: reset failed: {e}");
        }
    }

    fn forward(&mut self, token: u32) -> Result<Vec<f32>, GenError> {
        self.check_room(1)?;
        self.forward_last(&[token]).map_err(gen_err)
    }

    fn vocab(&self) -> usize {
        ik::VOCAB_REAL
    }

    fn max_context(&self) -> usize {
        self.max_ctx
    }

    fn enable_batch(&mut self, max_batch: usize) -> Result<usize, GenError> {
        self.enable_batch_impl(max_batch).map_err(gen_err)
    }

    fn forward_batch(&mut self, tokens: &[u32], positions: &[u32]) -> Result<Vec<f32>, GenError> {
        self.decode_tick(tokens, positions).map_err(gen_err)
    }

    fn forward_prefill(&mut self, slot: usize, tokens: &[u32]) -> Result<Vec<f32>, GenError> {
        self.prefill_slot(slot, tokens).map_err(gen_err)
    }

    fn forward_prefill_batch(
        &mut self,
        items: &[(usize, Vec<u32>)],
    ) -> Result<Vec<Vec<f32>>, GenError> {
        let items: Vec<(usize, &[u32])> = items.iter().map(|(s, t)| (*s, t.as_slice())).collect();
        self.prefill_slots(&items).map_err(gen_err)
    }

    fn forward_prefill_stream(&mut self, tokens: &[u32]) -> Result<Vec<f32>, GenError> {
        Generator::forward_prefill(self, 0, tokens)
    }

    fn release_inactive_slots(&mut self, occupied: &[bool]) {
        for slot in 0..self.kv.slots {
            if !occupied.get(slot).copied().unwrap_or(false) {
                self.queue.retain(|p| p.slot != slot);
                if !self.kv.tables[slot].blocks().is_empty() {
                    self.kv.release(slot);
                }
                if let Some(p) = self.prefix.as_mut() {
                    p.drop_staged(slot);
                    p.reply_release(slot);
                }
            }
        }
    }

    fn supports_chunked_prefill(&self) -> bool {
        true
    }

    fn prefill_queue(&self) -> Vec<(usize, usize, usize)> {
        self.prefill_queue_impl()
    }

    fn prefill_tick_cap(&self, decode_rows: usize) -> usize {
        self.prefill_tick_cap_impl(decode_rows)
    }

    fn prefill_begin(&mut self, slot: usize, tokens: Vec<u32>) -> Result<(), GenError> {
        self.prefill_begin_cached(slot, tokens, &[], true)
            .map(|_| ())
            .map_err(gen_err)
    }

    // the prefix cache (prefix.rs)
    fn prefill_begin_hinted(
        &mut self,
        slot: usize,
        tokens: Vec<u32>,
        hints: &[usize],
    ) -> Result<usize, GenError> {
        self.prefill_begin_cached(slot, tokens, hints, true)
            .map_err(gen_err)
    }

    fn prefix_share_floor(&self) -> Option<usize> {
        self.prefix
            .as_ref()
            .map(|_| super::prefix::MIN_CACHE_PREFIX.max(32))
    }

    fn take_prefill_reused(&mut self, slot: usize) -> usize {
        self.prefix
            .as_mut()
            .and_then(|p| p.last_reused.get_mut(slot))
            .map_or(0, std::mem::take)
    }

    fn anchor_at(&mut self, _slot: usize, tokens: &[u32], upto: usize) {
        if let Some(p) = self.prefix.as_mut() {
            p.radix.mark_anchor(tokens, upto);
        }
    }

    fn set_slot_background(&mut self, slot: usize, on: bool) {
        if let Some(p) = self.prefix.as_mut() {
            p.radix.set_slot_low(slot, on);
        }
    }

    fn reply_pin(&mut self, slot: usize) {
        self.reply_pin_impl(slot);
    }

    fn prefill_abort(&mut self, slot: usize) -> bool {
        self.prefill_abort_impl(slot)
    }

    // pictures (mm.rs): the tower at admission, then the chunked queue
    fn supports_mm_slots(&self) -> bool {
        self.vision.is_some()
    }

    fn forward_prefill_multimodal(
        &mut self,
        slot: usize,
        chunks: &[crate::service::MmChunk],
    ) -> Result<(Vec<f32>, usize), GenError> {
        self.prefill_mm_slot_impl(slot, chunks).map_err(gen_err)
    }

    fn forward_multimodal(
        &mut self,
        chunks: &[crate::service::MmChunk],
    ) -> Result<Option<(Vec<f32>, usize)>, GenError> {
        if self.vision.is_none() {
            return Ok(None);
        }
        self.prefill_mm_slot_impl(0, chunks)
            .map(Some)
            .map_err(gen_err)
    }

    fn supports_chunked_multimodal(&self) -> bool {
        self.vision.is_some()
    }

    fn prefill_begin_multimodal(
        &mut self,
        items: Vec<(usize, Vec<crate::service::MmChunk>)>,
    ) -> Vec<(usize, crate::generator::MmAdmit)> {
        self.prefill_begin_mm_impl(items)
    }

    fn vision_budget(&self) -> Option<crate::generator::VisionBudget> {
        self.vision_budget_impl()
    }

    fn forward_mixed(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
    ) -> Result<(Vec<f32>, Vec<(usize, Vec<f32>, usize)>), GenError> {
        self.forward_mixed_impl(decodes, budget).map_err(gen_err)
    }

    fn pool_free_blocks(&self) -> Option<usize> {
        Some(self.pool_free_impl())
    }

    // ---- speculative rounds with the release's DSpark draft (spec.rs) ----

    fn spec_capable(&self) -> bool {
        self.draft_ready()
    }

    // the draft filters slots itself (a slot it cannot draft gets none)
    fn spec_draft_per_slot_warm(&self) -> bool {
        self.draft_ready()
    }

    // Measured on GB10: every verify row past the first adds its 8 routed
    // experts' planes (~1.5 GB a row at decode widths), so a round pays
    // only while one stream holds the tick - at two or four live slots the
    // rounds ran 15% / 7% behind plain decode ticks, at one they break even
    // or win with the text.
    fn spec_live_cap(&self) -> usize {
        1
    }

    fn spec_ensure_warm(
        &mut self,
        slot: usize,
        _committed: &[u32],
        want_pos: u32,
    ) -> Result<bool, GenError> {
        // `want_pos` is the last committed position: the slot holds one more
        if !self.draft_ready() || slot >= self.kv.slots {
            return Ok(false);
        }
        Ok(self.sync_slots(&[(slot, want_pos as usize + 1)]).is_ok())
    }

    fn spec_draft_batch(
        &mut self,
        pendings: &[(usize, u32)],
        k: usize,
    ) -> Result<Option<Vec<Vec<u32>>>, GenError> {
        if !self.draft_ready() {
            return Ok(None);
        }
        self.draft_for(pendings, k).map(Some).map_err(gen_err)
    }

    fn forward_spec_batch(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
    ) -> Result<Option<Vec<u32>>, GenError> {
        if !self.draft_ready() {
            return Ok(None);
        }
        // every row greedy, on the device (the host scan's lowest-index tie)
        let rows: usize = reqs.iter().map(|r| r.2.len()).sum();
        let mut params = vec![0u32; rows * 4];
        for p in params.as_chunks_mut::<4>().0 {
            p[2] = 1;
        }
        self.verify_picks(reqs, &params).map(Some).map_err(gen_err)
    }

    // The verify with every row sampled by its pre-drawn plan (greedy, or
    // temperature-only: the u-quantile of softmax(logits x inv_t) walking
    // the vocab in index order - the dense sampler's distribution), on the
    // device: only the ids come back (a round's 200K-wide logit rows
    // crossing to the host cost ~5 ms of an ~80 ms round at a 22K context).
    fn forward_spec_batch_plans(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
        plans: &[crate::sampler::DevicePlan],
    ) -> Result<Option<Vec<u32>>, GenError> {
        use crate::sampler::DevicePlan;
        let rows: usize = reqs.iter().map(|r| r.2.len()).sum();
        if !self.draft_ready()
            || plans.len() != rows
            || plans
                .iter()
                .any(|p| !matches!(p, DevicePlan::Greedy | DevicePlan::Categorical { .. }))
        {
            return Ok(None);
        }
        let mut params = vec![0u32; rows * 4];
        for (p, plan) in params.as_chunks_mut::<4>().0.iter_mut().zip(plans) {
            match *plan {
                DevicePlan::Categorical { inv_t, u } => {
                    p[0] = inv_t.to_bits();
                    p[1] = u.to_bits();
                    p[2] = 2;
                }
                _ => p[2] = 1,
            }
        }
        self.verify_picks(reqs, &params).map(Some).map_err(gen_err)
    }

    fn forward_spec_verify(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
    ) -> Result<Option<Vec<f32>>, GenError> {
        if !self.draft_ready() {
            return Ok(None);
        }
        self.verify(reqs).map(Some).map_err(gen_err)
    }

    fn spec_commit(&mut self, committed: &[u32]) -> Result<(), GenError> {
        let want: Vec<(usize, usize)> = self
            .last_verify
            .iter()
            .zip(committed)
            .map(|(&(slot, pos), &c)| (slot, pos + c as usize))
            .collect();
        self.sync_slots(&want).map_err(gen_err)
    }
}
