//! The serving contract: the batched paged-KV lane (batch.rs) with the
//! stall-free chunked queue and the encoder budget (chunked.rs). There is no
//! separate serial spine - the serial surface (`forward`, warm-up) decodes
//! through slot 0 of the batch lane, and the exclusive multimodal path
//! prefills slot 0 the same way.

use crate::generator::{FinishSample, GenError, Generator, MmAdmit, RowSample, SampledStep};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::service::MmChunk;

use super::load::GpuNemotronParse;

/// Keep the scheduler's typed signals: pool exhaustion preempts, a
/// provably infeasible config fails the spawn.
fn gen_err(e: GpuModelError) -> GenError {
    match e {
        GpuModelError::PoolExhausted => GenError::PoolExhausted,
        GpuModelError::Config(msg) => GenError::Config(msg),
        other => GenError::Backend(other.to_string()),
    }
}

impl GpuNemotronParse {
    fn ensure_batch(&mut self) -> Result<(), GpuModelError> {
        if self.batch.is_none() {
            self.enable_batch(1)?;
        }
        Ok(())
    }
}

impl Generator for GpuNemotronParse {
    fn device_mem_used(&self) -> Option<u64> {
        self.exec.process_mem_used()
    }

    fn release_idle_memory(&mut self) {
        self.exec.trim_mem_pool();
    }

    fn reset(&mut self) {
        if let Some(bs) = self.batch.as_mut() {
            bs.slot0_pos = 0;
        }
        // KV needs no clearing: every attention read is position-bounded
    }

    fn forward(&mut self, token: u32) -> Result<Vec<f32>, GenError> {
        self.ensure_batch().map_err(gen_err)?;
        let pos = self.batch.as_ref().expect("batch").slot0_pos;
        if pos >= self.max_ctx {
            return Err(GenError::Backend(format!(
                "context full: {pos} rows at max_ctx {}",
                self.max_ctx
            )));
        }
        self.batch_step_slots(&[token], &[pos as u32], &[0])
            .map_err(gen_err)?;
        self.batch.as_mut().expect("batch").slot0_pos = pos + 1;
        self.read_batch_logits(1).map_err(gen_err)
    }

    fn vocab(&self) -> usize {
        self.cfg.vocab
    }

    fn max_context(&self) -> usize {
        self.max_ctx
    }

    fn weights_mem_bytes(&self) -> Option<u64> {
        Some(self.weights_bytes)
    }

    fn kv_mem_bytes(&self) -> Option<u64> {
        Some(self.batch.as_ref().map_or(0, |b| b.kv_bytes))
    }

    fn enable_batch(&mut self, max_batch: usize) -> Result<usize, GenError> {
        GpuNemotronParse::enable_batch(self, max_batch).map_err(gen_err)
    }

    fn forward_batch(&mut self, tokens: &[u32], positions: &[u32]) -> Result<Vec<f32>, GenError> {
        let slots: Vec<u32> = (0..tokens.len() as u32).collect();
        self.batch_step_slots(tokens, positions, &slots)
            .map_err(gen_err)?;
        self.read_batch_logits(tokens.len()).map_err(gen_err)
    }

    fn forward_prefill(&mut self, slot: usize, tokens: &[u32]) -> Result<Vec<f32>, GenError> {
        self.ensure_batch().map_err(gen_err)?;
        self.prefill_now(slot, &[MmChunk::Text(tokens.to_vec())])
            .map(|(l, _)| l)
            .map_err(gen_err)
    }

    fn forward_prefill_stream(&mut self, tokens: &[u32]) -> Result<Vec<f32>, GenError> {
        Generator::forward_prefill(self, 0, tokens)
    }

    fn release_inactive_slots(&mut self, occupied: &[bool]) {
        GpuNemotronParse::release_inactive_slots(self, occupied);
        self.chunk_release(occupied);
    }

    fn pool_free_blocks(&self) -> Option<usize> {
        GpuNemotronParse::pool_free_blocks(self)
    }

    // ── chunked prefill + the encoder budget ───────────────────────────────

    fn supports_chunked_prefill(&self) -> bool {
        self.batch.is_some()
    }

    fn prefill_queue(&self) -> Vec<(usize, usize, usize)> {
        self.chunked
            .iter()
            .map(|c| (c.slot, c.cursor, c.rows() - c.cursor))
            .collect()
    }

    fn prefill_tick_cap(&self, decode_rows: usize) -> usize {
        self.batch.as_ref().map_or(0, |bs| {
            crate::gpu_model::granite::batch::pf_rows().min(bs.cap.saturating_sub(decode_rows))
        })
    }

    fn prefill_begin(&mut self, slot: usize, tokens: Vec<u32>) -> Result<(), GenError> {
        self.prefill_begin_impl(slot, tokens).map_err(gen_err)
    }

    fn prefill_abort(&mut self, slot: usize) -> bool {
        self.prefill_abort_impl(slot)
    }

    fn forward_mixed(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
    ) -> Result<(Vec<f32>, Vec<(usize, Vec<f32>, usize)>), GenError> {
        self.forward_mixed_impl(decodes, budget).map_err(gen_err)
    }

    fn supports_device_sampling(&self) -> bool {
        self.supports_device_sampling_impl()
    }

    fn forward_batch_sampled(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        plans: &[RowSample],
    ) -> Result<SampledStep, GenError> {
        let slots: Vec<u32> = (0..tokens.len() as u32).collect();
        self.batch_step_slots(tokens, positions, &slots)
            .map_err(gen_err)?;
        self.sample_head_rows(tokens.len(), plans).map_err(gen_err)
    }

    fn forward_mixed_sampled(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
        plans: &[RowSample],
        _fin_plans: &[(usize, RowSample)],
    ) -> Result<(SampledStep, Vec<(usize, FinishSample, usize)>), GenError> {
        self.forward_mixed_sampled_impl(decodes, budget, plans)
            .map_err(gen_err)
    }

    fn supports_chunked_multimodal(&self) -> bool {
        self.batch.is_some() && paddock_models::dev_var_os!("PADDOCK_MM_NO_CHUNK").is_none()
    }

    fn prefill_begin_multimodal(
        &mut self,
        items: Vec<(usize, Vec<MmChunk>)>,
    ) -> Vec<(usize, MmAdmit)> {
        self.prefill_begin_multimodal_impl(items)
    }

    fn encode_step(&mut self) -> Vec<(usize, MmAdmit)> {
        self.encode_step_impl()
    }

    fn encoding_pending(&self) -> bool {
        !self.enc.is_empty() || self.flight.is_some()
    }

    fn supports_mm_slots(&self) -> bool {
        self.batch.is_some()
    }

    fn forward_prefill_multimodal(
        &mut self,
        slot: usize,
        chunks: &[MmChunk],
    ) -> Result<(Vec<f32>, usize), GenError> {
        self.ensure_batch().map_err(gen_err)?;
        self.prefill_now(slot, chunks).map_err(gen_err)
    }

    fn forward_prefill_multimodal_batch(
        &mut self,
        items: Vec<(usize, Vec<MmChunk>)>,
    ) -> Vec<(usize, Result<(Vec<f32>, usize), GenError>)> {
        items
            .into_iter()
            .map(|(slot, chunks)| (slot, self.prefill_now(slot, &chunks).map_err(gen_err)))
            .collect()
    }

    // ── one-token MTP (spec.rs) ─────────────────────────────────────────────

    fn spec_capable(&self) -> bool {
        self.spec_ready()
    }

    /// The checkpoint trains one head: a draft of one, every round.
    fn spec_fixed_draft_depth(&self) -> Option<usize> {
        self.spec_ready().then_some(1)
    }

    fn spec_depth_cap(&self) -> Option<usize> {
        self.spec_ready().then_some(1)
    }

    /// The head drafts from the slot's own last hidden row, never by
    /// replaying tokens - so a slot whose KV cursor sits past its token
    /// history (the picture rows) is warm the moment it has a pending token.
    fn spec_draft_kv_space(&self) -> bool {
        self.spec_ready()
    }

    /// Verify rows are two a slot; the head-row scratch is sized for every
    /// slot's pair.
    fn spec_live_cap(&self) -> usize {
        self.batch.as_ref().map_or(0, |b| b.n_slots)
    }

    fn spec_draft_batch(
        &mut self,
        pendings: &[(usize, u32)],
        _k: usize,
    ) -> Result<Option<Vec<Vec<u32>>>, GenError> {
        if !self.spec_ready() {
            return Ok(None);
        }
        let d = self.spec_draft_impl(pendings).map_err(gen_err)?;
        Ok(Some(d.into_iter().map(|t| vec![t]).collect()))
    }

    /// `h` is kept after every head pass, so a slot is warm whenever the
    /// stored row is its last committed one - nothing to re-sync.
    fn spec_ensure_warm(
        &mut self,
        slot: usize,
        _committed: &[u32],
        want_pos: u32,
    ) -> Result<bool, GenError> {
        Ok(self.spec_ready() && self.spec_warm(slot, want_pos))
    }

    fn forward_spec_batch(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
    ) -> Result<Option<Vec<u32>>, GenError> {
        if !self.spec_ready() {
            return Ok(None);
        }
        self.spec_verify_impl(reqs).map(Some).map_err(gen_err)
    }

    fn forward_multimodal(
        &mut self,
        chunks: &[MmChunk],
    ) -> Result<Option<(Vec<f32>, usize)>, GenError> {
        self.ensure_batch().map_err(gen_err)?;
        self.prefill_now(0, chunks).map(Some).map_err(gen_err)
    }
}

impl GpuNemotronParse {
    /// One page decoded greedily on `slot` - the oracle gates' and the
    /// bench's direct path (no service). `forced` teacher-forces the inputs
    /// (the oracle's own ids) and returns this engine's pick at every
    /// position; free-running stops at EOS or `max_new`.
    pub fn decode_page_greedy(
        &mut self,
        slot: usize,
        chunks: &[MmChunk],
        max_new: usize,
        forced: Option<&[u32]>,
    ) -> Result<Vec<u32>, GpuModelError> {
        self.ensure_batch()?;
        let (logits, n_rows) = self.prefill_now(slot, chunks)?;
        // the finisher's logits arrive penalized; argmax, lowest id on ties
        let mut pick = logits
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |b, (i, &v)| {
                if v > b.1 { (i, v) } else { b }
            })
            .0 as u32;
        let mut out = vec![pick];
        let mut pos = n_rows;
        let steps = forced.map_or(max_new, |f| f.len());
        while out.len() < steps {
            let input = forced.map_or(pick, |f| f[out.len() - 1]);
            if forced.is_none() && input == self.cfg.eos_id {
                break;
            }
            if pos >= self.max_ctx {
                break;
            }
            self.batch_step_slots(&[input], &[pos as u32], &[slot as u32])?;
            pick = self.read_picks(1)?[0];
            out.push(pick);
            pos += 1;
        }
        if forced.is_none() && out.last() == Some(&self.cfg.eos_id) {
            // keep the EOS: the oracle's id stream ends with it too
        }
        Ok(out)
    }
}

impl GpuNemotronParse {
    /// The tower, for the phase bench and the gates.
    pub fn tower_mut(&mut self) -> &mut super::vision::RadioTower {
        &mut self.tower
    }

    /// The phase bench's decode probe: one batched decode tick of `r` rows
    /// (slots 0..r at `pos`), synchronized - device time of the tick.
    pub fn bench_decode_tick(&mut self, r: usize, pos: u32) -> Result<f64, GpuModelError> {
        let slots: Vec<u32> = (0..r as u32).collect();
        let toks = vec![self.cfg.bos_id; r];
        let p = vec![pos; r];
        self.batch_step_slots(&toks, &p, &slots)?;
        self.exec
            .stream
            .synchronize()
            .map_err(crate::gpu::from_driver)?;
        let t = std::time::Instant::now();
        for _ in 0..10 {
            self.batch_step_slots(&toks, &p, &slots)?;
        }
        self.exec
            .stream
            .synchronize()
            .map_err(crate::gpu::from_driver)?;
        Ok(t.elapsed().as_secs_f64() * 1e3 / 10.0)
    }

    /// One verify round over `r` slots (2 rows each) - device time.
    pub fn bench_verify_round(&mut self, r: usize, pos: usize) -> Result<f64, GpuModelError> {
        let reqs: Vec<(usize, usize, Vec<u32>)> = (0..r)
            .map(|s| (s, pos, vec![self.cfg.bos_id, self.cfg.bos_id]))
            .collect();
        self.spec_verify_impl(&reqs)?;
        let t = std::time::Instant::now();
        for _ in 0..10 {
            self.spec_verify_impl(&reqs)?;
        }
        self.exec
            .stream
            .synchronize()
            .map_err(crate::gpu::from_driver)?;
        Ok(t.elapsed().as_secs_f64() * 1e3 / 10.0)
    }
}
