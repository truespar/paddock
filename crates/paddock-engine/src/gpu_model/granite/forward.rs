//! Granite serial forward - batch-1 decode + token-by-token prefill (the
//! Generator bring-up trio: reset/forward/vocab). Op order follows
//! llama.cpp's `src/models/granite.cpp` one-for-one so the greedy-parity diff
//! has no structural difference to explain:
//!
//! ```text
//! embd  = tok_embd[t]; embd *= embedding_scale        (build_inp_embd, llama-graph.cpp ~2302)
//! layer:  cur  = Attn(RMSNorm(x))                      kq_scale = attention_scale
//!         cur *= residual_scale;  x += cur             (granite.cpp ~241)
//!         cur  = SwiGLU(RMSNorm(x))
//!         cur *= residual_scale;  x += cur             (granite.cpp ~301)
//! head:   logits = lm_head · RMSNorm(x);  logits /= logit_scale  (granite.cpp ~188)
//! ```
//!
//! `scale_add` (x += w·y) does the residual multiplier and the add in one
//! launch, which is the same arithmetic as llama.cpp's ggml_scale-then-add.
//! Rope is the NORM-convention entry point - see mod.rs.
//!
//! Plain launches, no graph capture yet - correctness first (greedy match vs
//! the newest llama.cpp release binary on the identical GGUF); the capture and
//! batch lanes follow, the order gemma4 and laguna took.

use cudarc::driver::CudaSlice;

use crate::generator::{FinishSample, GenError, Generator, RowSample, SampledStep};
use crate::gpu::KvDtype;
use crate::gpu_model::gpt_oss::GpuModelError;

use super::*;

fn gen_err(e: GpuModelError) -> GenError {
    match e {
        GpuModelError::PoolExhausted => GenError::PoolExhausted,
        other => GenError::Backend(other.to_string()),
    }
}

/// Per-sequence decode state: dense per-layer KV. Granite is all-full
/// attention - every layer caches and every layer reads the whole prefix, so
/// there are no SWA rings to size differently.
pub(crate) struct DecodeState {
    pub kv_k: Vec<CudaSlice<u8>>,
    pub kv_v: Vec<CudaSlice<u8>>,
    pub pos: usize,
    pub d_token: CudaSlice<u32>,
    pub d_pos: CudaSlice<u32>,
    /// constant [0] - slot 0.
    pub d_slots: CudaSlice<u32>,
}

/// Decode-step scratch, allocated once.
pub(crate) struct Scratch {
    pub d_x: CudaSlice<f32>,
    pub d_xn: CudaSlice<f32>,
    pub d_q: CudaSlice<f32>,
    pub d_k: CudaSlice<f32>,
    pub d_v: CudaSlice<f32>,
    pub d_attn: CudaSlice<f32>,
    pub d_proj: CudaSlice<f32>,
    /// Sink logits [n_heads] from `alloc_no_sinks` - the very-negative no-op
    /// value, because granite has no attention sinks. It is the identity for
    /// the softmax denominator (exp(-1e30-m)=0); zero is not - it injects a
    /// phantom exp(0-max) term that steals real probability mass from the
    /// actual positions. Granite is unusually exposed: its KQ scale is
    /// attention_scale = 1/128 rather than 1/sqrt(128), so scores are ~7x
    /// smaller and exp(0-max) is not negligible. Measured on the 8b: an
    /// all-zero sink buffer left the top-1/top-2 gap at 2.64 nats against the
    /// oracle's 6.21 on a single-token prompt (worst case - one position, so
    /// the phantom term carries the most weight).
    pub d_sinks: CudaSlice<f32>,
    pub d_ffn_gate: CudaSlice<f32>,
    /// [2*n_ff] landing pad for the merged gate|up GEMV; `swiglu_fused`
    /// packs it down into `d_ffn_gate`.
    pub d_ffn_gu: CudaSlice<f32>,
    pub d_ffn_up: CudaSlice<f32>,
    pub d_logits: CudaSlice<f32>,
}

impl GpuGranite {
    fn ensure_decode(&mut self) -> Result<(), GpuModelError> {
        if self.decode.is_some() && self.scratch.is_some() {
            return Ok(());
        }
        let e = &self.exec;
        let hp = &self.hp;
        let kv_dim = hp.n_kv_heads * hp.head_dim;
        let kv_dtype = self.kv_dtype;
        let kv_bytes = kv_dtype.bytes();

        let (mut kv_k, mut kv_v) = (
            Vec::with_capacity(hp.n_layer),
            Vec::with_capacity(hp.n_layer),
        );
        for _ in 0..hp.n_layer {
            kv_k.push(e.alloc_u8(self.max_ctx * kv_dim * kv_bytes)?);
            kv_v.push(e.alloc_u8(self.max_ctx * kv_dim * kv_bytes)?);
        }
        self.decode = Some(DecodeState {
            kv_k,
            kv_v,
            pos: 0,
            d_token: e.alloc_u32(1)?,
            d_pos: e.alloc_u32(1)?,
            d_slots: e.alloc_u32(1)?, // zeroed -> slot 0
        });
        self.scratch = Some(Scratch {
            d_x: e.alloc(hp.n_embd)?,
            d_xn: e.alloc(hp.n_embd)?,
            d_q: e.alloc(hp.n_heads * hp.head_dim)?,
            d_k: e.alloc(kv_dim)?,
            d_v: e.alloc(kv_dim)?,
            d_attn: e.alloc(hp.n_heads * hp.head_dim)?,
            d_proj: e.alloc(hp.n_embd)?,
            d_sinks: e.alloc_no_sinks(hp.n_heads)?,
            d_ffn_gate: e.alloc(hp.n_ff)?,
            d_ffn_gu: e.alloc(2 * hp.n_ff)?,
            d_ffn_up: e.alloc(hp.n_ff)?,
            d_logits: e.alloc(hp.n_vocab)?,
        });
        Ok(())
    }

    /// One token through the whole stack; returns the full logits row.
    fn forward_one(&mut self, token: u32) -> Result<Vec<f32>, GpuModelError> {
        // The serial lane has no DeepStack path: it walks one token at a time
        // with no row/position table, so there is nowhere to address vision
        // rows from. Refuse loudly rather than return fluent text that ignored
        // the image - this is exactly the silent-failure shape the project
        // principles forbid. Vision requests go through the batch lane.
        if !self.media.is_empty() {
            return Err(GpuModelError::Unsupported(
                "granite: images are registered but this request is on the serial batch-1 lane, \
                 which cannot inject DeepStack features - enable the batch lane for vision"
                    .into(),
            ));
        }
        self.ensure_decode()?;
        let exec = self.exec.clone();
        let hp = &self.hp;
        let (embd, n_heads, n_kv_heads, head_dim) =
            (hp.n_embd, hp.n_heads, hp.n_kv_heads, hp.head_dim);
        let kv_dim = n_kv_heads * head_dim;
        let (eps, n_ff) = (hp.eps, hp.n_ff);
        // Granite's own KQ scale, not 1/sqrt(head_dim).
        let scale = hp.attention_scale;
        let res_s = hp.residual_scale;
        let kv_dtype = self.kv_dtype;
        let sc = self.scratch.as_mut().expect("scratch");
        let ds = self.decode.as_mut().expect("decode");

        if ds.pos >= self.max_ctx {
            return Err(GpuModelError::Unsupported(format!(
                "context full: {} tokens at max_ctx {}",
                ds.pos, self.max_ctx
            )));
        }
        let pos = ds.pos as u32;
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
        exec.stream
            .memcpy_htod(&[token], &mut ds.d_token)
            .map_err(drv)?;
        exec.stream
            .memcpy_htod(&[pos], &mut ds.d_pos)
            .map_err(drv)?;

        super::ops::embed_gather(&exec, &self.tok_embd, &ds.d_token, &mut sc.d_x, embd, 1)?;
        // embedding_multiplier - applied right after the gather, as
        // llama.cpp does inside build_inp_embd.
        exec.scale(&mut sc.d_x, hp.embedding_scale, embd)?;

        for (li, layer) in self.layers.iter().enumerate() {
            exec.rmsnorm_batch(&sc.d_x, &layer.attn_norm.buf, &mut sc.d_xn, embd, eps, 1)?;
            if let Some(qkv) = &layer.qkv_nv4 {
                // fused plane: one GEMV, then the three segments peel off into
                // the spine's own q/k/v buffers. This serial path is not the
                // serving band, so three tiny copies keep it on its unfused
                // rope/append below rather than growing a paged twin here.
                exec.nvf4_gemv(qkv, &sc.d_xn, &mut sc.d_ffn_gu, None)?;
                let qd = n_heads * head_dim;
                exec.stream
                    .memcpy_dtod(&sc.d_ffn_gu.slice(0..qd), &mut sc.d_q)
                    .map_err(drv)?;
                exec.stream
                    .memcpy_dtod(&sc.d_ffn_gu.slice(qd..qd + kv_dim), &mut sc.d_k)
                    .map_err(drv)?;
                exec.stream
                    .memcpy_dtod(
                        &sc.d_ffn_gu.slice(qd + kv_dim..qd + 2 * kv_dim),
                        &mut sc.d_v,
                    )
                    .map_err(drv)?;
            } else {
                super::ops::gemv(&exec, &layer.wq, &sc.d_xn, &mut sc.d_q)?;
                super::ops::gemv(&exec, &layer.wk, &sc.d_xn, &mut sc.d_k)?;
                super::ops::gemv(&exec, &layer.wv, &sc.d_xn, &mut sc.d_v)?;
            }
            // NORM convention (interleaved pairs) - granite's rope type.
            exec.rope_yarn_batch_norm(&mut sc.d_q, &ds.d_pos, n_heads, head_dim, hp.rope, 1)?;
            exec.rope_yarn_batch_norm(&mut sc.d_k, &ds.d_pos, n_kv_heads, head_dim, hp.rope, 1)?;
            exec.kv_append_batch(
                &sc.d_k,
                &mut ds.kv_k[li],
                &ds.d_pos,
                Some(&ds.d_slots),
                kv_dim,
                self.max_ctx,
                1,
                kv_dtype,
            )?;
            exec.kv_append_batch(
                &sc.d_v,
                &mut ds.kv_v[li],
                &ds.d_pos,
                Some(&ds.d_slots),
                kv_dim,
                self.max_ctx,
                1,
                kv_dtype,
            )?;
            // window 0 = full attention on every layer
            exec.attn_decode_batch(
                &sc.d_q,
                &ds.kv_k[li],
                &ds.kv_v[li],
                &sc.d_sinks,
                &mut sc.d_attn,
                &ds.d_pos,
                Some(&ds.d_slots),
                n_heads,
                n_kv_heads,
                head_dim,
                self.max_ctx,
                kv_dim,
                0,
                1,
                scale,
                kv_dtype,
            )?;
            super::ops::gemv(&exec, &layer.wo, &sc.d_attn, &mut sc.d_proj)?;
            // residual_multiplier on the attention branch: x += 0.22·proj
            exec.scale_add(&mut sc.d_x, &sc.d_proj, res_s, embd)?;

            exec.rmsnorm_batch(&sc.d_x, &layer.ffn_norm.buf, &mut sc.d_xn, embd, eps, 1)?;
            match (&layer.gate_up, &layer.gate, &layer.up) {
                // merged [2*n_ff, embd]: one GEMV, then the packing epilogue
                (Some(gu), _, _) => {
                    super::ops::gemv(&exec, gu, &sc.d_xn, &mut sc.d_ffn_gu)?;
                    if matches!(gu, GraniteW::Nvf4(p) if p.gu_pairs) {
                        exec.swiglu_fused_il(&sc.d_ffn_gu, &mut sc.d_ffn_gate, n_ff, 1)?;
                    } else {
                        exec.swiglu_fused(&sc.d_ffn_gu, &mut sc.d_ffn_gate, n_ff, 1)?;
                    }
                }
                (None, Some(g), Some(u)) => {
                    super::ops::gemv(&exec, g, &sc.d_xn, &mut sc.d_ffn_gate)?;
                    super::ops::gemv(&exec, u, &sc.d_xn, &mut sc.d_ffn_up)?;
                    exec.swiglu(&mut sc.d_ffn_gate, &sc.d_ffn_up, n_ff)?;
                }
                _ => {
                    return Err(GpuModelError::Unsupported(
                        "granite ffn planes missing".into(),
                    ));
                }
            }
            super::ops::gemv(&exec, &layer.down, &sc.d_ffn_gate, &mut sc.d_proj)?;
            // residual_multiplier again on the FFN branch
            exec.scale_add(&mut sc.d_x, &sc.d_proj, res_s, embd)?;
        }

        exec.rmsnorm_batch(&sc.d_x, &self.output_norm.buf, &mut sc.d_xn, embd, eps, 1)?;
        super::ops::gemv(&exec, &self.lm_head, &sc.d_xn, &mut sc.d_logits)?;
        // logits_scaling: llama.cpp scales by 1/f_logit_scale. This must stay
        // on the device logits rather than folding into sampling - argmax is
        // invariant to it, but logprobs and temperature are not.
        exec.scale(&mut sc.d_logits, 1.0 / hp.logit_scale, hp.n_vocab)?;
        ds.pos += 1;
        let logits = exec.stream.clone_dtoh(&sc.d_logits).map_err(drv)?;
        Ok(logits)
    }
}

impl GpuGranite {
    /// Select the KV cache element type (default [`KvDtype::Fp16`], greedy-exact).
    /// [`KvDtype::Fp8E4m3`] is a lossy opt-in throughput/memory mode. Drops any
    /// existing decode/batch state so the caches re-allocate at the new element
    /// size on the next `forward_one`/`enable_batch`; call it before serving.
    pub fn set_kv_dtype(&mut self, dtype: KvDtype) {
        self.kv_dtype = dtype;
        self.decode = None;
        self.scratch = None;
        self.batch = None;
        self.chunked.clear();
    }
}

impl Generator for GpuGranite {
    /// The live mempool counter every family publishes as its device ledger
    /// (`/api/stats` `model_mem`): without it the manager had nothing to
    /// reconcile this model's NVML figure against.
    fn device_mem_used(&self) -> Option<u64> {
        self.exec.process_mem_used()
    }

    fn release_idle_memory(&mut self) {
        self.exec.trim_mem_pool();
    }

    fn reset(&mut self) {
        self.pipe_abort();
        if let Some(ds) = self.decode.as_mut() {
            ds.pos = 0;
        }
        // KV needs no clearing: every attention read is position-bounded.
    }

    fn forward(&mut self, token: u32) -> Result<Vec<f32>, GenError> {
        self.forward_one(token).map_err(gen_err)
    }

    fn vocab(&self) -> usize {
        self.hp.n_vocab
    }

    fn max_context(&self) -> usize {
        self.max_ctx
    }

    fn weights_mem_bytes(&self) -> Option<u64> {
        Some(self.weights_bytes)
    }

    fn kv_mem_bytes(&self) -> Option<u64> {
        self.kv_mem_bytes_impl()
    }

    // ── the batched serving lane (batch.rs) ─────────────────────────────────

    fn enable_batch(&mut self, max_batch: usize) -> Result<usize, GenError> {
        self.enable_batch_impl(max_batch).map_err(gen_err)
    }

    /// granite's drafter (`spec.rs`'s model-free `NgramDraft`, prompt-lookup
    /// decoding - see `forward_spec_batch` above) needs no attached file and
    /// no env-var arm, unlike laguna's DFlash/MTP check: it's always
    /// structurally available once loaded. True unconditionally so service.rs
    /// routes a single-user (`--max-batch 1`) granite session through the
    /// batched loop, which is also the only place the tuned W4A8 decode GEMV
    /// and prefix cache live (see service.rs's routing note).
    fn spec_capable(&self) -> bool {
        true
    }

    fn forward_batch(&mut self, tokens: &[u32], positions: &[u32]) -> Result<Vec<f32>, GenError> {
        self.batch_step(tokens, positions).map_err(gen_err)?;
        self.read_batch_logits(tokens.len()).map_err(gen_err)
    }

    fn supports_device_sampling(&self) -> bool {
        self.supports_device_sampling_impl()
    }

    fn supports_device_trunc(&self) -> bool {
        self.device_trunc_supported()
    }

    fn forward_batch_sampled(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        plans: &[RowSample],
    ) -> Result<SampledStep, GenError> {
        self.forward_batch_sampled_impl(tokens, positions, plans)
            .map_err(gen_err)
    }
    fn supports_decode_pipe(&self) -> bool {
        self.supports_decode_pipe_impl()
    }
    fn decode_pipe_begin(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        plans: &[crate::generator::RowSample],
    ) -> Result<(), crate::generator::GenError> {
        self.decode_pipe_begin_impl(tokens, positions, None, plans)
            .map_err(|e| crate::generator::GenError::Backend(e.to_string()))
    }
    fn decode_pipe_begin_slots(
        &mut self,
        slots: &[u32],
        tokens: &[u32],
        positions: &[u32],
        plans: &[crate::generator::RowSample],
    ) -> Result<(), crate::generator::GenError> {
        self.decode_pipe_begin_impl(tokens, positions, Some(slots), plans)
            .map_err(|e| crate::generator::GenError::Backend(e.to_string()))
    }
    fn decode_pipe_next(
        &mut self,
        plans: &[crate::generator::RowSample],
    ) -> Result<Vec<u32>, crate::generator::GenError> {
        self.decode_pipe_next_impl(plans)
            .map_err(|e| crate::generator::GenError::Backend(e.to_string()))
    }
    fn decode_pipe_drain(&mut self) -> Result<Vec<u32>, crate::generator::GenError> {
        self.decode_pipe_drain_impl()
            .map_err(|e| crate::generator::GenError::Backend(e.to_string()))
    }

    fn forward_prefill(&mut self, slot: usize, tokens: &[u32]) -> Result<Vec<f32>, GenError> {
        self.forward_prefill_impl(slot, tokens).map_err(gen_err)
    }

    fn forward_prefill_batch(
        &mut self,
        items: &[(usize, Vec<u32>)],
    ) -> Result<Vec<Vec<f32>>, GenError> {
        self.forward_prefill_batch_impl(items).map_err(gen_err)
    }

    /// granite has no MTP head, so every draft the service hands here came
    /// from its model-free n-gram drafter (prompt-lookup decoding) - see
    /// `spec_policy.rs`/`spec.rs`. Verifying that draft is just a tiny
    /// chunked-prefill pass (see `forward_spec_batch_impl`), so this is the
    /// same weight-read the batch would have paid anyway, now yielding
    /// 1+accepted tokens per pass instead of 1 - the memory-wall lever a
    /// single-stream dense decode has no other way around.
    fn forward_spec_batch(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
    ) -> Result<Option<Vec<u32>>, GenError> {
        if self.batch.is_none() {
            return Ok(None);
        }
        self.forward_spec_batch_impl(reqs)
            .map(Some)
            .map_err(gen_err)
    }

    /// Single-stream bulk prefill. Without this the default feeds one token
    /// per `forward` call, so a long prompt prefills at DECODE speed -
    /// roughly 45 tokens/s on the 8b before the batch lane existed.
    fn forward_prefill_stream(&mut self, tokens: &[u32]) -> Result<Vec<f32>, GenError> {
        // batched: slot 0 is this stream's own; serial: the token-by-token
        // spine is still the only path (and still the parity reference)
        if self.batch.is_some() {
            return self.forward_prefill_impl(0, tokens).map_err(gen_err);
        }
        let mut logits = Vec::new();
        for &t in tokens {
            logits = self.forward(t)?;
        }
        Ok(logits)
    }

    fn supports_chunked_prefill(&self) -> bool {
        self.batch.is_some()
    }

    // the mixed tick's FIFO over the queue, row-exact from each cursor
    fn prefill_queue(&self) -> Vec<(usize, usize, usize)> {
        self.chunked
            .iter()
            .map(|c| (c.slot, c.cursor, c.tokens.len() - c.cursor))
            .collect()
    }

    // plan_chunk's cap under the pass's row capacity (the decode rows share it)
    fn prefill_tick_cap(&self, decode_rows: usize) -> usize {
        self.batch.as_ref().map_or(0, |bs| {
            super::batch::pf_rows().min(bs.cap.saturating_sub(decode_rows))
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

    fn forward_mixed_sampled(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
        plans: &[RowSample],
        fin_plans: &[(usize, RowSample)],
    ) -> Result<(SampledStep, Vec<(usize, FinishSample, usize)>), GenError> {
        self.forward_mixed_sampled_impl(decodes, budget, plans, fin_plans)
            .map_err(gen_err)
    }

    fn release_inactive_slots(&mut self, occupied: &[bool]) {
        self.release_inactive_slots_impl(occupied);
        // A wave mid-encode holds its own copy of the slot list, so it has to
        // hear about a death too - otherwise it would queue a prefill for a
        // slot the scheduler has already handed to somebody else.
        self.wave_release(occupied);
    }

    fn tier_pump(&mut self) {
        self.tier_pump_impl();
    }
    fn tier_prefix_loading(&mut self, slot: usize, tokens: &[u32]) -> bool {
        self.tier_consult_impl(slot, tokens)
    }
    fn tier_observe_prefill(&mut self, tokens: u32, wall_us: f64) {
        if let Some(t) = self.batch.as_mut().and_then(|b| b.tier.as_mut()) {
            t.cost.observe_prefill(tokens, wall_us);
        }
    }
    fn tier_stats(&self) -> Option<crate::kv_tier::TierStats> {
        self.batch.as_ref()?.tier.as_ref().map(|t| t.tier_stats())
    }
    fn tier_report(&self) -> Option<crate::kv_tier::TierReport> {
        self.batch
            .as_ref()?
            .tier
            .as_ref()
            .map(crate::kv_tier::PoolTier::report)
    }
    fn take_prefill_reused(&mut self, slot: usize) -> usize {
        self.last_reused.get_mut(slot).map_or(0, std::mem::take)
    }

    // ── vision (multimodal.rs) ──────────────────────────────────────────────

    /// granite-vision prefills into batch slots like gemma4/qwen35 do, never
    /// through the exclusive drain-the-server path: an image request is an
    /// ordinary prompt whose rows happen to carry projected pixels.
    fn supports_mm_slots(&self) -> bool {
        (self.vision.is_some() || self.audio.is_some()) && self.batch.is_some()
    }

    fn vision_budget(&self) -> Option<crate::generator::VisionBudget> {
        self.vision.as_ref().map(|v| v.budget())
    }

    fn forward_prefill_multimodal(
        &mut self,
        slot: usize,
        chunks: &[crate::service::MmChunk],
    ) -> Result<(Vec<f32>, usize), GenError> {
        self.multimodal_prefill_slot(slot, chunks).map_err(gen_err)
    }

    /// One tower pass for the whole admission wave, then per-slot prefill.
    ///
    /// The default here is a serial per-request loop, which made N concurrent
    /// image requests each run their own 27-block tower + 8 Q-Formers before
    /// any of them prefilled - the vi8 TTFT staircase. Encoding is
    /// the expensive half: ~206 ms for one 640x440 chart on an A6000.
    ///
    /// The prefill half is still per-slot. That is deliberate rather than
    /// forgotten: batching it properly means the mm lane enqueueing a ROW PLAN
    /// into the stall-free chunked queue the text path uses, which is
    /// - and doing it there gets cross-slot coalescing AND stops an image
    ///   prompt holding the tick, instead of only the first.
    fn forward_prefill_multimodal_batch(
        &mut self,
        items: Vec<(usize, Vec<crate::service::MmChunk>)>,
    ) -> Vec<(usize, Result<(Vec<f32>, usize), GenError>)> {
        let witness = paddock_models::dev_var_os!("PADDOCK_ROUTE_WITNESS").is_some();
        let t0 = std::time::Instant::now();
        let n_req = items.len();
        // granite-SPEECH has nothing for `encode_wave` to do: it batches IMAGE
        // tiles, and a clip contributes none, so a speech prompt came back
        // with an empty feature vector and then failed the one-per-media-chunk
        // check in `mm_admit` - "0 pre-encoded images for a prompt with 1
        // image chunks", a 500 on a few percent of requests at c32.
        // This path is reachable for speech whenever
        // the chunked lane leaves a slot behind (its in-flight bound is a
        // fixed room count, and c32 saturates it), so it is not a
        // vision-only path however much its wave machinery is. Each clip
        // encodes with its own prompt, which is what the audio tower costs
        // anyway: ~4 ms for 3.5 s of speech against 270-330 ms for one
        // max-grid picture, and the whole reason speech admits inline.
        if self.audio.is_some() {
            let out: Vec<_> = items
                .into_iter()
                .map(|(k, chunks)| (k, self.multimodal_prefill_slot(k, &chunks).map_err(gen_err)))
                .collect();
            if witness {
                eprintln!(
                    "pd route: speech batch-prefill {n_req} reqs in {:.1}ms",
                    t0.elapsed().as_secs_f64() * 1e3
                );
            }
            return out;
        }
        // A/B pin: restores the per-request loop this replaced, so the batched
        // pass can be measured against it without a rebuild - and so a wave
        // that ever answers wrong can be bisected against the serial path.
        if paddock_models::dev_var_os!("PADDOCK_MM_SERIAL_ENCODE").is_some() {
            let out: Vec<_> = items
                .into_iter()
                .map(|(k, chunks)| {
                    let r = self.multimodal_prefill_slot(k, &chunks).map_err(gen_err);
                    (k, r)
                })
                .collect();
            if witness {
                eprintln!(
                    "pd route: mm serial {n_req} reqs (encode+prefill) in {:.1}ms",
                    t0.elapsed().as_secs_f64() * 1e3
                );
            }
            return out;
        }
        let waves = {
            let refs: Vec<&[crate::service::MmChunk]> =
                items.iter().map(|(_, c)| c.as_slice()).collect();
            self.encode_wave(&refs)
        };
        if witness {
            eprintln!(
                "pd route: mm encode {n_req} reqs in {:.1}ms",
                t0.elapsed().as_secs_f64() * 1e3
            );
        }
        let waves = match waves {
            Ok(w) => w,
            // A batched-encode failure is systemic (alloc/driver), not one
            // request's fault: report it on every pending slot rather than
            // half-serving the wave.
            Err(e) => {
                let msg = e.to_string();
                return items
                    .into_iter()
                    .map(|(k, _)| (k, Err(GenError::Backend(msg.clone()))))
                    .collect();
            }
        };
        let t1 = std::time::Instant::now();
        let out: Vec<_> = items
            .into_iter()
            .zip(waves)
            .map(|((k, chunks), feats)| {
                let r = self
                    .multimodal_prefill_slot_pre(k, &chunks, Some(&feats))
                    .map_err(gen_err);
                (k, r)
            })
            .collect();
        if witness {
            eprintln!(
                "pd route: mm prefill {n_req} reqs in {:.1}ms",
                t1.elapsed().as_secs_f64() * 1e3
            );
        }
        out
    }

    /// Image prompts ride the same stall-free queue text does - the tick is
    /// never held for a whole picture. Gated on the batch lane for
    /// the same reason `supports_mm_slots` is: the serial lane has no row
    /// table to address DeepStack features from.
    fn supports_chunked_multimodal(&self) -> bool {
        // A/B pin: PADDOCK_MM_NO_CHUNK=1 restores the blocking whole-prompt
        // pass, so the stall this removes can be measured rather than asserted.
        (self.vision.is_some() || self.audio.is_some())
            && self.batch.is_some()
            && paddock_models::dev_var_os!("PADDOCK_MM_NO_CHUNK").is_none()
    }

    /// Register the wave with the ENCODER BUDGET (encode.rs) and
    /// spend its first group.
    ///
    /// Earlier work made the queueing half free and left the ENCODE as the whole stall -
    /// 270-330 ms of one max-grid picture with no decode row moving, which no
    /// amount of PREFILL chunking touches because none of it is prefill. So the
    /// encode is now budgeted in tile groups across ticks, and this call returns
    /// as soon as the first group is done: `Queued` when that finished the wave
    /// (an everyday photo is 2-5 tiles, so usually), `Encoding` when it did not.
    fn prefill_begin_multimodal(
        &mut self,
        items: Vec<(usize, Vec<crate::service::MmChunk>)>,
    ) -> Vec<(usize, crate::generator::MmAdmit)> {
        let witness = paddock_models::dev_var_os!("PADDOCK_ROUTE_WITNESS").is_some();
        let t0 = std::time::Instant::now();
        let n_req = items.len();
        // granite-SPEECH admits directly: there is no tile budget to spend
        // because a clip is not tiled - the conformer's conv module is
        // CENTERED over the whole plane, so a slice of a clip is not the rows
        // that slice would have contributed (the exact property that makes a
        // vision tile stack divisible). Nor does it need one: the tower is
        // ~4 ms for a 3.5 s clip and ~19 ms for 46 s, against 270-330 ms for
        // one max-grid picture. So encode inline and queue the row stream,
        // which is what qwen3-asr's lane does too.
        if self.audio.is_some() {
            let out: Vec<_> = items
                .into_iter()
                .map(|(k, chunks)| {
                    let r = match self.multimodal_prefill_begin(k, &chunks, None) {
                        Ok(_rows) => crate::generator::MmAdmit::Queued,
                        Err(e) => {
                            crate::generator::MmAdmit::Failed(GenError::Backend(e.to_string()))
                        }
                    };
                    (k, r)
                })
                .collect();
            if witness {
                eprintln!(
                    "pd route: speech admit {n_req} reqs (encode+queue) in {:.1}ms",
                    t0.elapsed().as_secs_f64() * 1e3
                );
            }
            return out;
        }
        let out = self.wave_begin(items);
        if witness {
            // The three numbers that matter, kept apart deliberately: how long
            // the tick was actually held, whether the wave finished in it, and
            // how much of a hold is queueing rather than encoding.
            let done = out
                .iter()
                .all(|(_, a)| !matches!(a, crate::generator::MmAdmit::Encoding));
            eprintln!(
                "pd route: mm chunked-admit {n_req} reqs - first group {:.1}ms, complete={done}",
                t0.elapsed().as_secs_f64() * 1e3
            );
        }
        out
    }

    fn encode_step(&mut self) -> Vec<(usize, crate::generator::MmAdmit)> {
        let witness = paddock_models::dev_var_os!("PADDOCK_ROUTE_WITNESS").is_some();
        let t0 = std::time::Instant::now();
        let out = self.wave_step();
        if witness {
            eprintln!(
                "pd route: mm encode group {:.1}ms, finished {} reqs",
                t0.elapsed().as_secs_f64() * 1e3,
                out.len()
            );
        }
        out
    }

    fn encoding_pending(&self) -> bool {
        self.wave_pending()
    }

    /// The exclusive path, reached only when `supports_mm_slots` is false. The
    /// default returns `Ok(None)`, which the service reports as "this model does
    /// not support image input" - true for a text-only granite, misleading for a
    /// vision one that merely has no batch lane. Say which it is.
    fn forward_multimodal(
        &mut self,
        _chunks: &[crate::service::MmChunk],
    ) -> Result<Option<(Vec<f32>, usize)>, GenError> {
        if self.vision.is_none() && self.audio.is_none() {
            return Ok(None);
        }
        Err(GenError::Backend(
            "granite multimodal needs the batch lane (max_batch > 1): the serial batch-1 lane \
             has no row/position table to address media features from"
                .into(),
        ))
    }

    fn pool_free_blocks(&self) -> Option<usize> {
        self.pool_free_blocks_impl()
    }
}
