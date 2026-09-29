//! DeepSeek-OCR serial forward - batch-1 decode + token-by-token prefill, the
//! laguna bring-up trio with this family's deltas applied and the R-SWA ring
//! implemented exactly as designed:
//! zero new kernels - the ring is a write-slot redirect plus a clamped
//! attention-bound position, both host arithmetic.
//!
//! Deltas from laguna's step, all simplifications except the router:
//!   - no per-head QK norms, no softplus gate, no YaRN/partial rotary - plain
//!     full-rotary NEOX rope θ=10000 on every layer (`rope_yarn_batch` with
//!     ext_factor 0, laguna's own SWA-layer arm);
//!   - the router is `moe_topk_softmax_all_batch`: softmax over all 64
//!     logits, greedy top-6, weights = the full-distribution probs with no
//!     renormalization (slot 381). Laguna's sigmoid+bias+renorm kernel picks
//!     the same experts and is silently wrong here - see the slot comment.
//!
//! ## The ring, in two variables
//!
//! Every step carries a true position (rope, bookkeeping) and derives:
//!
//!   write slot     = pf + (pos - pf) % W    once pos ≥ pf + W, else pos
//!   attention pos  = pf + W - 1             once pos ≥ pf + W, else pos
//!
//! where `pf` is the recorded prefill length and W the ring window. The write
//! slot makes steady-state append an in-place overwrite of the oldest ring
//! row; the attention position bounds visibility at `pf + W` rows (the decode
//! kernel derives `n_kv = position + 1` and takes q pre-roped, so the clamp
//! cannot touch rope). Warmup (pos < pf + W) degenerates to ordinary causal
//! append - the reference's phase 2. The caller marks the prefill/decode
//! boundary with [`GpuDeepseekOcr::note_prefill_end`]; without it the model
//! is an ordinary causal decoder, which is exactly the reference's behavior
//! when the ring is disarmed.
//!
//! Plain launches, no capture - correctness first (stage sums + logits + 200
//! greedy steps vs the reference oracle, tests/gpu_deepseek_ocr_forward.rs).

use cudarc::driver::CudaSlice;

use paddock_kernels::reference::ops::YarnRope;

use crate::generator::GenError;
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::qwen35::gemv_any;

use super::load::{DsFfn, GpuDeepseekOcr};

/// Per-sequence decode state: dense per-layer KV sized `max_ctx` rows. The
/// serial lane never needs more than `prefill + W` of it once the ring engages
/// - the paged pool (batch.rs) is where that saving becomes real; here the
///   dense buffer just holds the same rows the pool would.
pub(crate) struct DecodeState {
    pub kv_k: Vec<CudaSlice<u8>>,
    pub kv_v: Vec<CudaSlice<u8>>,
    /// True position of the next token - never clamped, never wrapped.
    pub pos: usize,
    /// Set by [`GpuDeepseekOcr::note_prefill_end`]; arms the ring.
    pub prefill_len: Option<usize>,
    pub d_token: CudaSlice<u32>,
    /// true position (rope)
    pub d_pos: CudaSlice<u32>,
    /// write slot (KV append) - diverges from d_pos in ring steady state
    pub d_wpos: CudaSlice<u32>,
    /// clamped position (attention visibility bound)
    pub d_apos: CudaSlice<u32>,
    /// constant [0] - slot 0
    pub d_slots: CudaSlice<u32>,
    /// plain NEOX rope params (θ 10000, no scaling)
    pub rope: (f32, f32, f32, f32, f32, f32),
    /// `PADDOCK_OCR_DEC_DUMP=1`: running f64 sum of each layer's output rows
    /// across every forward - comparable to the oracle's per-layer prefill
    /// stage sums when accumulated over exactly the prompt.
    pub layer_sums: Vec<f64>,
}

/// Decode-step scratch, one token wide.
pub(crate) struct Scratch {
    pub d_x: CudaSlice<f32>,
    pub d_xn: CudaSlice<f32>,
    pub d_q: CudaSlice<f32>,
    pub d_k: CudaSlice<f32>,
    pub d_v: CudaSlice<f32>,
    pub d_attn: CudaSlice<f32>,
    pub d_proj: CudaSlice<f32>,
    /// no-op sinks - this family ships none, so -inf (the softmax identity)
    pub d_sinks: CudaSlice<f32>,
    pub d_ffn_gate: CudaSlice<f32>,
    pub d_ffn_up: CudaSlice<f32>,
    pub d_moe_xq: CudaSlice<i8>,
    pub d_moe_xs: CudaSlice<f32>,
    pub d_moe_logits: CudaSlice<f32>,
    pub d_moe_idx: CudaSlice<u32>,
    pub d_moe_w: CudaSlice<f32>,
    pub d_moe_fused: CudaSlice<f32>,
    pub d_moe_fq: CudaSlice<i8>,
    pub d_moe_fs: CudaSlice<f32>,
    pub d_sh_gate: CudaSlice<f32>,
    pub d_sh_up: CudaSlice<f32>,
    pub d_sh_out: CudaSlice<f32>,
    pub d_logits: CudaSlice<f32>,
}

fn dump_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_OCR_DEC_DUMP").is_some())
}

impl GpuDeepseekOcr {
    fn ensure_decode(&mut self) -> Result<(), GpuModelError> {
        if self.decode.is_some() && self.scratch.is_some() {
            return Ok(());
        }
        let e = &self.exec;
        let hp = &self.hp;
        let kv_dim = hp.n_head_kv * hp.head_dim;
        let kv_bytes = self.kv_dtype.bytes();

        let (mut kv_k, mut kv_v) = (
            Vec::with_capacity(hp.n_layer),
            Vec::with_capacity(hp.n_layer),
        );
        for _ in 0..hp.n_layer {
            kv_k.push(e.alloc_u8(self.max_ctx * kv_dim * kv_bytes)?);
            kv_v.push(e.alloc_u8(self.max_ctx * kv_dim * kv_bytes)?);
        }
        // Plain rope: full head_dim rotary, θ from the header (10000),
        // freq_scale 1, ext_factor 0 (no YaRN ramp) - laguna's SWA-layer arm.
        let rope = YarnRope::new(
            hp.head_dim,
            hp.rope_freq_base,
            1.0,
            hp.n_ctx_train,
            0.0,
            1.0,
            32.0,
            1.0,
        )
        .kernel_params();
        self.decode = Some(DecodeState {
            kv_k,
            kv_v,
            pos: 0,
            prefill_len: None,
            d_token: e.alloc_u32(1)?,
            d_pos: e.alloc_u32(1)?,
            d_wpos: e.alloc_u32(1)?,
            d_apos: e.alloc_u32(1)?,
            d_slots: e.alloc_u32(1)?, // zeroed -> slot 0
            rope,
            layer_sums: vec![0.0; hp.n_layer],
        });

        let ff_max = self
            .layers
            .iter()
            .map(|l| match &l.ffn {
                DsFfn::Dense { gate, .. } => gate.dims()[1],
                DsFfn::Moe(_) => hp.n_ff_exp,
            })
            .max()
            .unwrap_or(hp.n_ff)
            .max(hp.shexp_ff());
        let fused_len = hp.n_expert_used * hp.n_ff_exp;
        self.scratch = Some(Scratch {
            d_x: e.alloc(hp.n_embd)?,
            d_xn: e.alloc(hp.n_embd)?,
            d_q: e.alloc(hp.n_embd)?,
            d_k: e.alloc(kv_dim)?,
            d_v: e.alloc(kv_dim)?,
            d_attn: e.alloc(hp.n_embd)?,
            d_proj: e.alloc(hp.n_embd)?,
            d_sinks: e.alloc_no_sinks(hp.n_head)?,
            d_ffn_gate: e.alloc(ff_max)?,
            d_ffn_up: e.alloc(ff_max)?,
            d_moe_xq: e.alloc_i8(hp.n_embd)?,
            d_moe_xs: e.alloc(hp.n_embd / 32)?,
            d_moe_logits: e.alloc(hp.n_expert)?,
            d_moe_idx: e.alloc_u32(hp.n_expert_used)?,
            d_moe_w: e.alloc(hp.n_expert_used)?,
            d_moe_fused: e.alloc(fused_len)?,
            d_moe_fq: e.alloc_i8(fused_len)?,
            d_moe_fs: e.alloc(fused_len / 32)?,
            d_sh_gate: e.alloc(hp.shexp_ff())?,
            d_sh_up: e.alloc(hp.shexp_ff())?,
            d_sh_out: e.alloc(hp.n_embd)?,
            d_logits: e.alloc(hp.n_vocab)?,
        });
        Ok(())
    }

    /// Mark the prefill/decode boundary: everything appended so far is the
    /// globally-visible prefix; the ring covers only what comes after. The
    /// reference records this at its first decode step - here the caller
    /// (test harness now, the serve path later) knows the boundary exactly.
    pub fn note_prefill_end(&mut self) {
        if let Some(ds) = self.decode.as_mut() {
            ds.prefill_len = Some(ds.pos);
        }
    }

    /// Drop all sequence state (weights stay resident).
    pub fn reset(&mut self) {
        if let Some(ds) = self.decode.as_mut() {
            ds.pos = 0;
            ds.prefill_len = None;
            ds.layer_sums.iter_mut().for_each(|s| *s = 0.0);
        }
    }

    /// Accumulated per-layer output sums (dump mode only) - the oracle's
    /// prefill stage sums, when accumulated over exactly the prompt tokens.
    pub fn layer_sums(&self) -> Option<&[f64]> {
        self.decode.as_ref().map(|d| d.layer_sums.as_slice())
    }

    /// One token through the whole stack; returns the full logits row.
    pub fn forward_one(&mut self, token: u32) -> Result<Vec<f32>, GpuModelError> {
        self.ensure_decode()?;
        let exec = self.exec.clone();
        let hp = &self.hp;
        let (embd, n_heads, n_kv_heads, head_dim) =
            (hp.n_embd, hp.n_head, hp.n_head_kv, hp.head_dim);
        let kv_dim = n_kv_heads * head_dim;
        let eps = hp.rms_eps;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let (n_expert, n_active, moe_ff, shexp_ff) =
            (hp.n_expert, hp.n_expert_used, hp.n_ff_exp, hp.shexp_ff());
        let kv_dtype = self.kv_dtype;
        let dump = dump_on();
        let sc = self.scratch.as_mut().expect("scratch");
        let ds = self.decode.as_mut().expect("decode");

        let pos = ds.pos;
        // The ring: write slot + attention bound diverge from the true
        // position once steady state is reached. Both are ordinary causal
        // before that (warmup) or when the boundary was never marked.
        let (wslot, apos) = match (ds.prefill_len, hp.rswa_window) {
            (Some(pf), Some(w)) if pos >= pf + w => (pf + (pos - pf) % w, pf + w - 1),
            _ => (pos, pos),
        };
        if wslot >= self.max_ctx {
            return Err(GpuModelError::Unsupported(format!(
                "context full: row {wslot} at max_ctx {}",
                self.max_ctx
            )));
        }
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
        exec.stream
            .memcpy_htod(&[token], &mut ds.d_token)
            .map_err(drv)?;
        exec.stream
            .memcpy_htod(&[pos as u32], &mut ds.d_pos)
            .map_err(drv)?;
        exec.stream
            .memcpy_htod(&[wslot as u32], &mut ds.d_wpos)
            .map_err(drv)?;
        exec.stream
            .memcpy_htod(&[apos as u32], &mut ds.d_apos)
            .map_err(drv)?;

        exec.embed_gather_batch_q8(&self.tok_embd, &ds.d_token, &mut sc.d_x, embd, 1)?;

        for (li, layer) in self.layers.iter().enumerate() {
            exec.rmsnorm_batch(&sc.d_x, &layer.attn_norm.buf, &mut sc.d_xn, embd, eps, 1)?;
            gemv_any(&exec, &layer.wq, &sc.d_xn, &mut sc.d_q)?;
            gemv_any(&exec, &layer.wk, &sc.d_xn, &mut sc.d_k)?;
            gemv_any(&exec, &layer.wv, &sc.d_xn, &mut sc.d_v)?;
            // rope on the true position - the clamp must never reach here
            exec.rope_yarn_batch(&mut sc.d_q, &ds.d_pos, n_heads, head_dim, ds.rope, 1)?;
            exec.rope_yarn_batch(&mut sc.d_k, &ds.d_pos, n_kv_heads, head_dim, ds.rope, 1)?;
            // append at the WRITE slot - in steady state this overwrites the
            // oldest ring row in place, the reference's phase 3
            exec.kv_append_batch(
                &sc.d_k,
                &mut ds.kv_k[li],
                &ds.d_wpos,
                Some(&ds.d_slots),
                kv_dim,
                self.max_ctx,
                1,
                kv_dtype,
            )?;
            exec.kv_append_batch(
                &sc.d_v,
                &mut ds.kv_v[li],
                &ds.d_wpos,
                Some(&ds.d_slots),
                kv_dim,
                self.max_ctx,
                1,
                kv_dtype,
            )?;
            // attention bounded by the CLAMPED position: n_kv = apos + 1 rows,
            // no window (visibility is prefix + the whole ring - no mask)
            exec.attn_decode_batch(
                &sc.d_q,
                &ds.kv_k[li],
                &ds.kv_v[li],
                &sc.d_sinks,
                &mut sc.d_attn,
                &ds.d_apos,
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
            gemv_any(&exec, &layer.wo, &sc.d_attn, &mut sc.d_proj)?;
            exec.add_rmsnorm_batch(
                &mut sc.d_x,
                &sc.d_proj,
                &layer.ffn_norm.buf,
                &mut sc.d_xn,
                embd,
                eps,
                1,
            )?;

            match &layer.ffn {
                DsFfn::Dense { gate, up, down } => {
                    gemv_any(&exec, gate, &sc.d_xn, &mut sc.d_ffn_gate)?;
                    gemv_any(&exec, up, &sc.d_xn, &mut sc.d_ffn_up)?;
                    exec.swiglu(&mut sc.d_ffn_gate, &sc.d_ffn_up, gate.dims()[1])?;
                    gemv_any(&exec, down, &sc.d_ffn_gate, &mut sc.d_proj)?;
                }
                DsFfn::Moe(w) => {
                    exec.quantize_q8(&sc.d_xn, &mut sc.d_moe_xq, &mut sc.d_moe_xs, embd)?;
                    // router in f32 off the UNQUANTIZED normed input - the
                    // reference's x.float() @ W.float()
                    exec.matvec_f32_batch(&w.router_w, &sc.d_xn, &mut sc.d_moe_logits, 1)?;
                    exec.moe_topk_softmax_all_batch(
                        &sc.d_moe_logits,
                        n_expert,
                        n_active,
                        &mut sc.d_moe_idx,
                        &mut sc.d_moe_w,
                        1,
                    )?;
                    exec.q8_0_moe_gate_up(
                        &w.gate_exps,
                        &w.up_exps,
                        &sc.d_moe_idx,
                        &sc.d_moe_xq,
                        &sc.d_moe_xs,
                        &mut sc.d_moe_fused,
                        n_active,
                        1,
                    )?;
                    exec.quantize_q8(
                        &sc.d_moe_fused,
                        &mut sc.d_moe_fq,
                        &mut sc.d_moe_fs,
                        n_active * moe_ff,
                    )?;
                    exec.q8_0_moe_down(
                        &w.down_exps,
                        &sc.d_moe_idx,
                        &sc.d_moe_w,
                        &sc.d_moe_fq,
                        &sc.d_moe_fs,
                        &mut sc.d_proj,
                        n_active,
                        1,
                    )?;
                    // shared expert: plain SwiGLU off the same normed input,
                    // ungated, always added
                    gemv_any(&exec, &w.shexp_gate, &sc.d_xn, &mut sc.d_sh_gate)?;
                    gemv_any(&exec, &w.shexp_up, &sc.d_xn, &mut sc.d_sh_up)?;
                    exec.swiglu(&mut sc.d_sh_gate, &sc.d_sh_up, shexp_ff)?;
                    gemv_any(&exec, &w.shexp_down, &sc.d_sh_gate, &mut sc.d_sh_out)?;
                    exec.add(&mut sc.d_proj, &sc.d_sh_out, embd)?;
                }
            }
            exec.add(&mut sc.d_x, &sc.d_proj, embd)?;
            if dump {
                let h = exec.to_host(&sc.d_x)?;
                ds.layer_sums[li] += h.iter().map(|&v| v as f64).sum::<f64>();
            }
        }

        exec.rmsnorm_batch(&sc.d_x, &self.output_norm.buf, &mut sc.d_xn, embd, eps, 1)?;
        gemv_any(&exec, &self.lm_head, &sc.d_xn, &mut sc.d_logits)?;
        ds.pos += 1;
        let logits = exec.to_host(&sc.d_logits)?;
        Ok(logits)
    }
}

// ── the serving surface  ─────────────────────────────────────────

fn gen_err(e: GpuModelError) -> crate::generator::GenError {
    crate::generator::GenError::Backend(e.to_string())
}

/// The Generator arm: qwen3_asr's minimal batched + mm-slots contract. The
/// batched lane is the lane (paged pool, R-SWA footprint pinning, radix
/// prefix, tuned prefill ladder); the serial spine stays the parity
/// reference and the no-paged-kv fallback. Chunked prefill / mixed ticks /
/// device sampling / spec are listed perf rungs, not prerequisites.
impl crate::generator::Generator for GpuDeepseekOcr {
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
        self.reset();
        if let Some(bs) = self.batch.as_mut() {
            bs.slot0_pos = 0;
        }
        // KV needs no clearing: every attention read is position-bounded.
    }

    fn forward(&mut self, token: u32) -> Result<Vec<f32>, crate::generator::GenError> {
        // Batched engine: serial-surface callers (warmup, serial loop) decode
        // through slot 0 of the batch lane - the ring state set by the last
        // prefill stays armed, so a long generation still pins its KV.
        if self.batch.is_some() {
            let pos = self.batch.as_ref().expect("batch").slot0_pos;
            if pos >= self.max_ctx {
                return Err(crate::generator::GenError::Backend(format!(
                    "context full: {pos} rows at max_ctx {}",
                    self.max_ctx
                )));
            }
            self.batch_step_slots(&[token], &[pos as u32], &[0])
                .map_err(gen_err)?;
            self.batch.as_mut().expect("batch").slot0_pos = pos + 1;
            return self.read_batch_logits(1).map_err(gen_err);
        }
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
        if let Some(bs) = self.batch.as_ref() {
            return Some(bs.kv_bytes);
        }
        let kv_dim = self.hp.n_head_kv * self.hp.head_dim;
        Some((self.hp.n_layer * 2 * self.max_ctx * kv_dim * self.kv_dtype.bytes()) as u64)
    }

    // ── the batched serving lane (batch.rs) ─────────────────────────────────

    fn enable_batch(&mut self, max_batch: usize) -> Result<usize, crate::generator::GenError> {
        self.enable_batch(max_batch).map_err(gen_err)
    }

    fn forward_batch(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
    ) -> Result<Vec<f32>, crate::generator::GenError> {
        self.batch_step(tokens, positions).map_err(gen_err)?;
        self.read_batch_logits(tokens.len()).map_err(gen_err)
    }

    fn forward_prefill(
        &mut self,
        slot: usize,
        tokens: &[u32],
    ) -> Result<Vec<f32>, crate::generator::GenError> {
        let logits = self.forward_prefill(slot, tokens).map_err(gen_err)?;
        if slot == 0 {
            self.batch.as_mut().expect("batch enabled").slot0_pos = tokens.len();
        }
        Ok(logits)
    }

    /// Single-stream bulk prefill. Batched: slot 0 (which also ARMS the
    /// ring at the prompt boundary). Serial: the token-by-token spine plus
    /// the explicit boundary mark - without the mark the ring never engages
    /// and a serial generation grows its KV like an ordinary causal model.
    fn forward_prefill_stream(
        &mut self,
        tokens: &[u32],
    ) -> Result<Vec<f32>, crate::generator::GenError> {
        if self.batch.is_some() {
            return crate::generator::Generator::forward_prefill(self, 0, tokens);
        }
        let mut logits = Vec::new();
        for &t in tokens {
            logits = self.forward_one(t).map_err(gen_err)?;
        }
        self.note_prefill_end();
        Ok(logits)
    }

    fn release_inactive_slots(&mut self, occupied: &[bool]) {
        self.release_inactive_slots(occupied);
        // the chunked queue and the encode queue hold their own slot lists,
        // so they have to hear about a death too - a dead encode entry still
        // REPORTS at its turn (the scheduler's encoding set only clears on
        // encode_step reports), it just queues nothing
        self.chunk_release(occupied);
    }

    // ── stall-free chunked prefill + the encoder budget (chunked.rs) ────────

    fn supports_chunked_prefill(&self) -> bool {
        self.batch.is_some()
    }

    // the mixed tick's FIFO over the queue, row-exact from each cursor
    fn prefill_queue(&self) -> Vec<(usize, usize, usize)> {
        self.chunked
            .iter()
            .map(|c| (c.slot, c.cursor, c.rows.len() - c.cursor))
            .collect()
    }

    // plan_chunk's cap under the pass's row capacity (the decode rows share it)
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

    /// This family's requests always arm the no-repeat-ngram guard - the
    /// reason it sat on the full-readback host sampling path (10.3 % of the
    /// c8 tick wall, the idle ledger's smp segment). The scheduler grants
    /// Device(Greedy) per tick only when the guard would ban nothing, and
    /// there is no decode-pipe lookahead here for that check to go stale on.
    fn device_greedy_ngram_ok(&self) -> bool {
        true
    }

    fn forward_batch_sampled(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        plans: &[crate::generator::RowSample],
    ) -> Result<crate::generator::SampledStep, GenError> {
        self.forward_batch_sampled_impl(tokens, positions, plans)
            .map_err(gen_err)
    }

    fn forward_mixed_sampled(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
        plans: &[crate::generator::RowSample],
        fin_plans: &[(usize, crate::generator::RowSample)],
    ) -> Result<
        (
            crate::generator::SampledStep,
            Vec<(usize, crate::generator::FinishSample, usize)>,
        ),
        GenError,
    > {
        self.forward_mixed_sampled_impl(decodes, budget, plans, fin_plans)
            .map_err(gen_err)
    }

    /// Image prompts ride the same stall-free queue text does - the tick is
    /// never held for a whole page (granite's shape), and the tower
    /// encode spends one call per tick under the encoder budget.
    fn supports_chunked_multimodal(&self) -> bool {
        // A/B pin: PADDOCK_MM_NO_CHUNK=1 restores the blocking wave path, so
        // the stall this removes can be measured rather than asserted (the
        // same pin granite carries).
        self.vision.is_some()
            && self.batch.is_some()
            && paddock_models::dev_var_os!("PADDOCK_MM_NO_CHUNK").is_none()
    }

    fn prefill_begin_multimodal(
        &mut self,
        items: Vec<(usize, Vec<crate::service::MmChunk>)>,
    ) -> Vec<(usize, crate::generator::MmAdmit)> {
        let witness = paddock_models::dev_var_os!("PADDOCK_ROUTE_WITNESS").is_some();
        let t0 = std::time::Instant::now();
        let n_req = items.len();
        let out = self.prefill_begin_multimodal_impl(items);
        if witness {
            let queued = out
                .iter()
                .filter(|(_, a)| matches!(a, crate::generator::MmAdmit::Queued))
                .count();
            tracing::info!(
                "pd route: ocr chunked-admit {n_req} reqs in {:.1}ms ({queued} queued, rest encoding)",
                t0.elapsed().as_secs_f64() * 1e3,
            );
        }
        out
    }

    fn encode_step(&mut self) -> Vec<(usize, crate::generator::MmAdmit)> {
        let witness = paddock_models::dev_var_os!("PADDOCK_ROUTE_WITNESS").is_some();
        let t0 = std::time::Instant::now();
        let out = self.encode_step_impl();
        if witness {
            tracing::info!(
                "pd route: ocr encode step {:.1}ms, finished {} reqs",
                t0.elapsed().as_secs_f64() * 1e3,
                out.len()
            );
        }
        out
    }

    fn encoding_pending(&self) -> bool {
        !self.enc.is_empty()
    }

    fn pool_free_blocks(&self) -> Option<usize> {
        self.pool_free_blocks()
    }

    fn take_prefill_reused(&mut self, slot: usize) -> usize {
        self.take_prefill_reused(slot)
    }

    // ── vision (multimodal.rs) ──────────────────────────────────────────────

    /// Image requests ride ordinary batch slots - never the exclusive
    /// drain-the-server path (granite/qwen35 precedent). On this family the
    /// prompt is mostly picture, so slot-riding plus the content-keyed radix
    /// is the whole serving story.
    fn supports_mm_slots(&self) -> bool {
        self.vision.is_some() && self.batch.is_some()
    }

    fn vision_budget(&self) -> Option<crate::generator::VisionBudget> {
        self.vision_budget_impl()
    }

    fn forward_prefill_multimodal(
        &mut self,
        slot: usize,
        chunks: &[crate::service::MmChunk],
    ) -> Result<(Vec<f32>, usize), crate::generator::GenError> {
        let out = self
            .multimodal_prefill_slot(slot, chunks)
            .map_err(gen_err)?;
        if slot == 0 {
            self.batch.as_mut().expect("batch enabled").slot0_pos = out.1;
        }
        Ok(out)
    }

    /// The scheduler's whole admission wave in one call - pipelined (door B
    /// every slot's host preprocessing runs on worker threads while
    /// the GPU executes the earlier slots, instead of each request waiting
    /// out the strangers ahead of it in line.
    fn forward_prefill_multimodal_batch(
        &mut self,
        items: Vec<(usize, Vec<crate::service::MmChunk>)>,
    ) -> Vec<(usize, Result<(Vec<f32>, usize), crate::generator::GenError>)> {
        let out: Vec<_> = self
            .multimodal_prefill_wave(items)
            .into_iter()
            .map(|(k, r)| (k, r.map_err(gen_err)))
            .collect();
        for (k, r) in &out {
            if *k == 0
                && let Ok((_, n)) = r
            {
                self.batch.as_mut().expect("batch enabled").slot0_pos = *n;
            }
        }
        out
    }

    /// The exclusive-path entry, reached only by serial-path callers: route
    /// through slot 0 of the batch lane. The serial spine has no splice -
    /// an OCR server without the paged pool cannot serve images, and says so.
    fn forward_multimodal(
        &mut self,
        chunks: &[crate::service::MmChunk],
    ) -> Result<Option<(Vec<f32>, usize)>, crate::generator::GenError> {
        if self.batch.is_some() {
            let out = crate::generator::Generator::forward_prefill_multimodal(self, 0, chunks)?;
            return Ok(Some(out));
        }
        Err(crate::generator::GenError::Backend(
            "deepseek-ocr image serving needs the batched lane (paged KV pack)".into(),
        ))
    }
}
