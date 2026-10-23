//! One-token speculation with the checkpoint's own MTP head.
//!
//! The head (2.1M params, trained with the release) drafts the token after
//! next from the decoder's final-normed hidden row `h` of the position that
//! produced the pending token and the pending token's embedding:
//! `draft = lm_head(extra_head(h + extra_proj(32 * E[pending])))` - no
//! attention, no norm. NVIDIA's vLLM proposer wires it exactly so (it swaps in
//! the target's tied, x32-scaled embedding and its LM head); the F64 oracle
//! reproduces the release's own acceptance (93-98% on the fixtures) with this
//! wiring and no other.
//!
//! A round: `spec_draft_batch` drafts one token per live slot from each
//! slot's stored `h`; `forward_spec_batch` runs `[pending, draft]` as two
//! decode rows per slot through the ordinary layer walk (KV for both rows
//! lands before either attends, so row 1 sees row 0) and the head. The
//! penalty is exact on both rows: row 0's set is the slot's committed ids
//! (the pending token is committed - it is marked seen as the row runs); row
//! 1 adds its own uncommitted input, the draft, as a per-row extra - what the
//! reference would have had in `input_ids` at that position. After the
//! picks come back the family applies the service's own accept rule
//! (greedy: accept while the draft equals the pick) to mark the accepted
//! draft seen and to keep `h` for the slot's next pending token.
//!
//! The draft is the head's plain argmax. Verification decides every emitted
//! token, so the draft's rule only moves acceptance, never output.

use crate::gpu::GpuError;
use crate::gpu_model::gpt_oss::GpuModelError;

use super::batch::N_EXTRA;
use super::load::GpuNemotronParse;

/// A captured spec shape: the draft over n pendings, or the verify over
/// (rows, committed row-0 inputs, and the attention shape: the invariant
/// walk's group count, else the chunk width of the shared KV walk).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum GraphKey {
    Draft(usize),
    Verify(usize, usize, usize),
}

fn drv(e: cudarc::driver::DriverError) -> GpuError {
    crate::gpu::from_driver(e)
}

impl GpuNemotronParse {
    pub(crate) fn spec_ready(&self) -> bool {
        self.mtp.is_some()
            && self.batch.is_some()
            && paddock_models::dev_var_os!("PADDOCK_NP_NO_MTP").is_none()
    }

    /// Keep head row `row`'s normed hidden as `slot`'s MTP input; `pos` is
    /// that row's KV position.
    pub(super) fn keep_h(
        &mut self,
        row: usize,
        slot: usize,
        pos: u32,
    ) -> Result<(), GpuModelError> {
        let dm = self.cfg.d_model;
        let exec = self.exec.clone();
        let bs = self.batch.as_mut().expect("batch enabled");
        let sc = &mut bs.sc;
        exec.copy_region(&sc.hid, row * dm, &mut sc.mtp_h, slot * dm, dm)?;
        bs.h_pos[slot] = pos;
        Ok(())
    }

    /// Warm = the stored `h` is the hidden of the slot's last committed row
    /// (the one that produced its pending token).
    pub(crate) fn spec_warm(&self, slot: usize, want_pos: u32) -> bool {
        self.batch
            .as_ref()
            .is_some_and(|b| b.h_pos.get(slot) == Some(&want_pos))
    }

    /// Draft one token for each `(slot, pending)`.
    pub(crate) fn spec_draft_impl(
        &mut self,
        pendings: &[(usize, u32)],
    ) -> Result<Vec<u32>, GpuModelError> {
        let n = pendings.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let exec = self.exec.clone();
        let bs = self.batch.as_mut().expect("batch enabled");
        if n > bs.hcap {
            return Err(GpuModelError::BatchTooLarge {
                got: n,
                max: bs.hcap,
            });
        }
        let sc = &mut bs.sc;
        let toks: Vec<u32> = pendings.iter().map(|p| p.1).collect();
        let slots: Vec<u32> = pendings.iter().map(|p| p.0 as u32).collect();
        exec.upload_u32(&toks, &mut sc.d_toks)?;
        exec.upload_u32(&slots, &mut sc.d_rslot)?;
        self.replay_or_capture(GraphKey::Draft(n), |m| m.draft_body(n))?;
        let sc = &self.batch.as_ref().expect("batch enabled").sc;
        let v = sc
            .d_out
            .try_slice(0..n)
            .ok_or_else(|| GpuError::Driver("draft picks slice".into()))?;
        Ok(exec.stream.clone_dtoh(&v).map_err(drv)?)
    }

    /// The draft's device work over `n` pendings whose tokens and slots are
    /// already in `d_toks` / `d_rslot` - graph-safe (every slot-dependent
    /// address is read on the device: `h` is gathered by `d_rslot`).
    fn draft_body(&mut self, n: usize) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let dm = self.cfg.d_model;
        let vocab = self.cfg.vocab;
        let mtp = self.mtp.as_ref().expect("spec_ready checked");
        let bs = self.batch.as_mut().expect("batch enabled");
        let words = bs.words;
        let sc = &mut bs.sc;
        // 32 E[pending] -> extra_proj (+bias), + h
        exec.embed_gather_batch(&self.embed32, &sc.d_toks, &mut sc.xe, dm, n)?;
        exec.convert_f32_f16(&sc.xe, &mut sc.xn16, n * dm)?;
        super::batch::lin(&exec, self.w16, &mtp.proj, &sc.xn16, &mut sc.q, n)?;
        exec.bias_add(&mut sc.q, &mtp.proj_b, n, dm)?;
        exec.embed_gather_batch(&sc.mtp_h, &sc.d_rslot, &mut sc.attn, dm, n)?;
        exec.add(&mut sc.q, &sc.attn, n * dm)?;
        // extra_head (+bias) -> the tied head (its e4m3 copy where built:
        // the input rounds to f16 exactly as the f16 head's convert does) ->
        // argmax
        exec.convert_f32_f16(&sc.q, &mut sc.xn16, n * dm)?;
        super::batch::lin(&exec, self.w16, &mtp.head, &sc.xn16, &mut sc.attn, n)?;
        exec.bias_add(&mut sc.attn, &mtp.head_b, n, dm)?;
        match &mtp.head8 {
            Some(h8) => exec.dense_w16_e4m3(
                h8,
                dm,
                vocab,
                &sc.attn,
                &mut sc.xn16,
                &mut sc.head_logits,
                vocab,
                n,
            )?,
            None => {
                exec.convert_f32_f16(&sc.attn, &mut sc.xn16, n * dm)?;
                super::batch::lin(
                    &exec,
                    self.w16,
                    &self.head,
                    &sc.xn16,
                    &mut sc.head_logits,
                    n,
                )?;
            }
        }
        // penalty 1.0 = the plain greedy pick (the kernel skips nothing else)
        exec.np_pen_argmax(
            &mut sc.head_logits,
            &sc.seen,
            &sc.d_rslot,
            &sc.d_extra,
            0,
            1.0,
            vocab,
            words,
            n,
            false,
            &mut sc.d_out,
        )?;
        Ok(())
    }

    /// The verify's device work over `r` uploaded rows: the committed row-0
    /// inputs marked seen, embed + norm, the layer walk, the penalized head.
    fn verify_body(&mut self, r: usize, n_or: usize, k1: usize) -> Result<(), GpuModelError> {
        {
            let exec = self.exec.clone();
            let bs = self.batch.as_mut().expect("batch enabled");
            let words = bs.words;
            let sc = &mut bs.sc;
            exec.np_seen_or(&mut sc.seen, &sc.d_or_s, &sc.d_or_t, n_or, words)?;
        }
        self.embed_rows(r)?;
        self.embed_norm(r)?;
        self.layer_walk_k(r, None, k1)?;
        self.head_rows(r, None, N_EXTRA)
    }

    /// Replay the graph captured for `key`, capturing `body` on first use.
    /// Every input the body reads lives in device buffers the caller has
    /// already uploaded, so one graph per shape serves every round of it.
    /// A new shape runs `body` eagerly once before its capture: a launcher's
    /// one-time setup (the f16 GEMM's K-split scratch is a lazy cudaMalloc)
    /// would invalidate the capture if it first happened inside it. The
    /// bodies are idempotent - seen bits OR, KV lands the same values at the
    /// same positions, the head recomputes its logits - so the warm run
    /// changes nothing the replay does not redo.
    fn replay_or_capture(
        &mut self,
        key: GraphKey,
        body: impl Fn(&mut Self) -> Result<(), GpuModelError>,
    ) -> Result<(), GpuModelError> {
        if paddock_models::dev_var_os!("PADDOCK_NP_NO_GRAPH").is_some() {
            return body(self);
        }
        if !self
            .batch
            .as_ref()
            .expect("batch enabled")
            .spec_graphs
            .contains_key(&key)
        {
            body(self)?;
            let exec = self.exec.clone();
            exec.stream
                .synchronize()
                .map_err(|e| GpuError::Driver(format!("spec pre-capture sync: {e}")))?;
            exec.stream
                .begin_capture(
                    cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                )
                .map_err(|e| GpuError::Driver(format!("spec begin_capture: {e}")))?;
            let rec = body(self);
            let graph = crate::gpu::end_capture_no_flags(&exec.stream)
                .map_err(|e| GpuError::Driver(format!("spec end_capture: {e}")));
            rec?;
            let g =
                graph?.ok_or_else(|| GpuError::Driver("spec capture produced no graph".into()))?;
            self.batch
                .as_mut()
                .expect("batch enabled")
                .spec_graphs
                .insert(key, super::SendGraph(g));
        }
        self.batch.as_ref().expect("batch enabled").spec_graphs[&key]
            .0
            .launch()
            .map_err(|e| GpuError::Driver(format!("spec graph launch: {e}")))?;
        Ok(())
    }

    /// Verify ragged chunks `(slot, start_pos, [pending, drafts..])`; the
    /// greedy pick of every row, flat in request order.
    pub(crate) fn spec_verify_impl(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
    ) -> Result<Vec<u32>, GpuModelError> {
        let mut toks = Vec::new();
        let mut pos = Vec::new();
        let mut slots = Vec::new();
        // per row: the extra penalized id (its own input when uncommitted)
        let mut extra = Vec::new();
        // the committed row 0 of each request marks its input seen
        let mut or_rows = Vec::new();
        for &(slot, start, ref chunk) in reqs {
            for (j, &t) in chunk.iter().enumerate() {
                if j == 0 {
                    or_rows.push(toks.len());
                }
                toks.push(t);
                pos.push((start + j) as u32);
                slots.push(slot as u32);
                extra.push(if j == 0 { u32::MAX } else { t });
            }
            if chunk.len() > 1 + N_EXTRA {
                return Err(GpuModelError::Unsupported(format!(
                    "nemotron-parse verifies {N_EXTRA} draft a slot, got {}",
                    chunk.len() - 1
                )));
            }
        }
        let r = toks.len();
        {
            let bs = self.batch.as_ref().expect("batch enabled");
            if r > bs.hcap {
                return Err(GpuModelError::BatchTooLarge {
                    got: r,
                    max: bs.hcap,
                });
            }
        }
        self.ensure_rows(&slots, &pos)?;
        self.upload_rows(&toks, &pos, &slots)?;
        let exec = self.exec.clone();
        {
            let sc = &mut self.batch.as_mut().expect("batch enabled").sc;
            let or_s: Vec<u32> = or_rows.iter().map(|&i| slots[i]).collect();
            let or_t: Vec<u32> = or_rows.iter().map(|&i| toks[i]).collect();
            exec.upload_u32(&or_s, &mut sc.d_or_s)?;
            exec.upload_u32(&or_t, &mut sc.d_or_t)?;
            let mut v = sc
                .d_extra
                .try_slice_mut(0..r * N_EXTRA)
                .ok_or_else(|| GpuError::Driver("extra slice".into()))?;
            exec.stream.memcpy_htod(&extra, &mut v).map_err(drv)?;
        }
        // every chunk carries its draft (the usual round): one KV walk a
        // slot - WIDTH-GATED. Measured GB10 (f16 KV, ~3K-row prefixes), verify
        // round shared vs per-row: 16 slots 8.93 vs 11.30 ms, 32 15.98 vs
        // 17.72; but 1 slot 3.35 vs 3.11, 8 6.19 vs 6.05 - below ~12 chunks
        // the slots' KV is L2-light and the fused walk only serializes.
        let full = reqs.iter().all(|q| q.2.len() == 1 + N_EXTRA);
        let k1 = if full && reqs.len() >= 12 {
            1 + N_EXTRA
        } else {
            1
        };
        let n_or = or_rows.len();
        // the invariant walk attends by the uploaded groups (a chunk a slot,
        // split at the split law's bucket edges): their count is the shape
        let shape = if self.inv {
            self.upload_band_groups(&slots, &pos)?
        } else {
            k1
        };
        self.replay_or_capture(GraphKey::Verify(r, n_or, shape), |m| {
            m.verify_body(r, n_or, k1)
        })?;
        let picks = self.read_picks(r)?;
        // the extras must not leak into the next head pass
        {
            let bs = self.batch.as_mut().expect("batch enabled");
            let none = vec![u32::MAX; r * N_EXTRA];
            let mut v = bs
                .sc
                .d_extra
                .try_slice_mut(0..r * N_EXTRA)
                .ok_or_else(|| GpuError::Driver("extra slice".into()))?;
            exec.stream.memcpy_htod(&none, &mut v).map_err(drv)?;
        }
        // the service's greedy accept rule, replayed: accepted drafts are
        // committed ids now, and h follows the row whose pick is the slot's
        // next pending token
        let mut base = 0usize;
        let (mut acc_s, mut acc_t) = (Vec::new(), Vec::new());
        let start_of = |slot: usize| reqs.iter().find(|r| r.0 == slot).map_or(0, |r| r.1);
        for &(slot, _, ref chunk) in reqs {
            let mut a = 0usize;
            while a + 1 < chunk.len() && chunk[a + 1] == picks[base + a] {
                acc_s.push(slot as u32);
                acc_t.push(chunk[a + 1]);
                a += 1;
            }
            self.keep_h(base + a, slot, (start_of(slot) + a) as u32)?;
            base += chunk.len();
        }
        if !acc_s.is_empty() {
            let bs = self.batch.as_mut().expect("batch enabled");
            let words = bs.words;
            let sc = &mut bs.sc;
            exec.upload_u32(&acc_s, &mut sc.d_or_s)?;
            exec.upload_u32(&acc_t, &mut sc.d_or_t)?;
            exec.np_seen_or(&mut sc.seen, &sc.d_or_s, &sc.d_or_t, acc_s.len(), words)?;
        }
        Ok(picks)
    }
}

/// What [`GpuNemotronParse::decode_page_spec`] measured.
pub struct SpecRun {
    pub tokens: Vec<u32>,
    /// rounds run and drafts accepted
    pub rounds: usize,
    pub accepted: usize,
}

impl GpuNemotronParse {
    /// One page decoded with MTP rounds on `slot`, the service's greedy accept
    /// rule applied here (the gates' and the bench's direct path).
    pub fn decode_page_spec(
        &mut self,
        slot: usize,
        chunks: &[crate::service::MmChunk],
        max_new: usize,
    ) -> Result<SpecRun, GpuModelError> {
        if self.batch.is_none() {
            self.enable_batch(1)?;
        }
        let (logits, n_rows) = self.prefill_now(slot, chunks)?;
        let mut pending = logits
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |b, (i, &v)| {
                if v > b.1 { (i, v) } else { b }
            })
            .0 as u32;
        let mut out = vec![pending];
        let (mut pos, mut rounds, mut accepted) = (n_rows, 0usize, 0usize);
        let eos = self.cfg.eos_id;
        while out.len() < max_new && pending != eos && pos + 2 <= self.max_ctx {
            let draft = self.spec_draft_impl(&[(slot, pending)])?[0];
            let picks = self.spec_verify_impl(&[(slot, pos, vec![pending, draft])])?;
            rounds += 1;
            if picks[0] == draft {
                accepted += 1;
                out.push(draft);
                pos += 1;
                if draft == eos || out.len() >= max_new {
                    break;
                }
                out.push(picks[1]);
                pending = picks[1];
            } else {
                out.push(picks[0]);
                pending = picks[0];
            }
            pos += 1;
        }
        Ok(SpecRun {
            tokens: out,
            rounds,
            accepted,
        })
    }
}
