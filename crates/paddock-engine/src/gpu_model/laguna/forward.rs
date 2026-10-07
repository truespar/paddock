//! Laguna serial forward - batch-1 decode + token-by-token prefill (the
//! Generator bring-up trio: reset/forward/vocab). Op sequence pinned in
//! qwen35's `record_step` is the
//! template, with the four Laguna deltas: separate per-head softplus gate,
//! sigmoid router with selection-only bias, always-on ungated shared expert,
//! and per-layer-type rope (partial YaRN on full layers / plain on SWA).
//!
//! Plain launches, no graph capture yet - correctness first (greedy match vs
//! the newest llama.cpp release binary), the capture/batch lanes follow.

use cudarc::driver::CudaSlice;

use crate::generator::{GenError, Generator};
use crate::gpu::KvDtype;
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::qwen35::gemv_any;

use super::*;

fn gen_err(e: GpuModelError) -> GenError {
    match e {
        // the scheduler preempts on pool exhaustion instead of failing the batch
        GpuModelError::PoolExhausted => GenError::PoolExhausted,
        other => GenError::Backend(other.to_string()),
    }
}

/// Per-sequence decode state: dense per-layer KV (every layer holds cache -
/// full layers read all of it, SWA layers read the trailing 512 via the
/// kernel's window arg; ring/paged pools land with the batch engine).
pub(crate) struct DecodeState {
    pub kv_k: Vec<CudaSlice<u8>>,
    pub kv_v: Vec<CudaSlice<u8>>,
    pub pos: usize,
    pub d_token: CudaSlice<u32>,
    pub d_pos: CudaSlice<u32>,
    /// [4] all-equal text positions - the partial-YaRN rope rides the mrope
    /// kernel (its n_rot seam is the partial rotary; equal axes collapse it
    /// to plain rope, reference-tested).
    pub d_mrope: CudaSlice<u32>,
    /// constant [0] - slot 0.
    pub d_slots: CudaSlice<u32>,
}

/// Decode-step scratch, allocated once for the LARGEST per-layer geometry
/// (SWA layers' 64-head Q width).
pub(crate) struct Scratch {
    pub d_x: CudaSlice<f32>,
    pub d_xn: CudaSlice<f32>,
    pub d_q: CudaSlice<f32>,
    pub d_qn: CudaSlice<f32>,
    pub d_k: CudaSlice<f32>,
    pub d_kn: CudaSlice<f32>,
    pub d_v: CudaSlice<f32>,
    /// per-head gate, pre-softplus [n_heads_max]
    pub d_gate_h: CudaSlice<f32>,
    pub d_attn: CudaSlice<f32>,
    pub d_proj: CudaSlice<f32>,
    /// no-op sinks [n_heads_max] - Laguna ships no attention sinks, so this is
    /// -inf (the softmax-denominator identity), not zero. See
    /// `GpuExecutor::alloc_no_sinks`.
    pub d_sinks: CudaSlice<f32>,
    pub d_ffn_gate: CudaSlice<f32>,
    pub d_ffn_up: CudaSlice<f32>,
    // MoE lane (batch-1 token-batched class)
    pub d_moe_xq: CudaSlice<i8>,
    pub d_moe_xs: CudaSlice<f32>,
    pub d_ssums: CudaSlice<f32>,
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

impl GpuLaguna {
    fn ensure_decode(&mut self) -> Result<(), GpuModelError> {
        if self.decode.is_some() && self.scratch.is_some() {
            return Ok(());
        }
        let e = &self.exec;
        let hp = &self.hp;
        let kv_dim = hp.n_kv_heads * hp.head_dim;
        let kv_dtype = self.kv_dtype;
        let kv_bytes = kv_dtype.bytes();
        let n_heads_max = hp.n_heads.iter().copied().max().unwrap_or(hp.n_heads[0]);
        let q_max = n_heads_max * hp.head_dim;
        let m = &hp.moe;

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
            d_mrope: e.alloc_u32(4)?,
            d_slots: e.alloc_u32(1)?, // zeroed -> slot 0
        });

        // Widest FFN row: dense layer 0 (8192) vs the routed fused rows.
        let ff_max = self
            .layers
            .iter()
            .map(|l| match &l.ffn {
                Ffn::Dense { gate, .. } => gate.dims()[1],
                Ffn::Moe(_) | Ffn::MoeNv(_) => m.moe_ff,
            })
            .max()
            .unwrap_or(m.moe_ff);
        let fused_len = m.n_active * m.moe_ff;
        self.scratch = Some(Scratch {
            d_x: e.alloc(hp.n_embd)?,
            d_xn: e.alloc(hp.n_embd)?,
            d_q: e.alloc(q_max)?,
            d_qn: e.alloc(q_max)?,
            d_k: e.alloc(kv_dim)?,
            d_kn: e.alloc(kv_dim)?,
            d_v: e.alloc(kv_dim)?,
            d_gate_h: e.alloc(n_heads_max)?,
            d_attn: e.alloc(q_max)?,
            d_proj: e.alloc(hp.n_embd)?,
            d_sinks: e.alloc_no_sinks(n_heads_max)?,
            d_ffn_gate: e.alloc(ff_max)?,
            d_ffn_up: e.alloc(ff_max)?,
            d_moe_xq: e.alloc_i8(hp.n_embd)?,
            d_moe_xs: e.alloc(hp.n_embd / 32)?,
            // per-16 sums, reused across both mu stages: xq rows (embd/16)
            // then fq rows (n_active * moe_ff / 16)
            d_ssums: e.alloc((hp.n_embd / 16).max(fused_len / 16))?,
            d_moe_logits: e.alloc(m.n_expert)?,
            d_moe_idx: e.alloc_u32(m.n_active)?,
            d_moe_w: e.alloc(m.n_active)?,
            d_moe_fused: e.alloc(fused_len)?,
            d_moe_fq: e.alloc_i8(fused_len)?,
            d_moe_fs: e.alloc(fused_len / 32)?,
            d_sh_gate: e.alloc(m.shexp_ff)?,
            d_sh_up: e.alloc(m.shexp_ff)?,
            d_sh_out: e.alloc(hp.n_embd)?,
            d_logits: e.alloc(hp.n_vocab)?,
        });
        Ok(())
    }

    /// One token through the whole stack; returns the full logits row.
    fn forward_one(&mut self, token: u32) -> Result<Vec<f32>, GpuModelError> {
        self.ensure_decode()?;
        let exec = self.exec.clone();
        let hp = &self.hp;
        let (embd, n_kv_heads, head_dim) = (hp.n_embd, hp.n_kv_heads, hp.head_dim);
        let kv_dim = n_kv_heads * head_dim;
        let eps = hp.eps;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let sections = [hp.n_rot as u32 / 2, 0, 0, 0];
        let moe_dims = hp.moe;
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
        exec.stream
            .memcpy_htod(&[pos; 4], &mut ds.d_mrope)
            .map_err(drv)?;

        match &self.tok_embd {
            TokEmbd::Q8(t) => exec.embed_gather_batch_q8(t, &ds.d_token, &mut sc.d_x, embd, 1)?,
            TokEmbd::Kq(t) => exec.kquant_gather(t, &ds.d_token, &mut sc.d_x, embd, 1)?,
            TokEmbd::Bf16(t) => {
                exec.embed_gather_bf16(t, &ds.d_token, &mut sc.d_x, embd, 1, 1.0)?
            }
        }

        for (li, layer) in self.layers.iter().enumerate() {
            let n_heads = layer.n_heads;
            exec.rmsnorm_batch(&sc.d_x, &layer.attn_norm.buf, &mut sc.d_xn, embd, eps, 1)?;
            match &layer.proj {
                Proj::Quant(q) => {
                    gemv_any(&exec, &q.wq, &sc.d_xn, &mut sc.d_q)?;
                    gemv_any(&exec, &q.wk, &sc.d_xn, &mut sc.d_k)?;
                    gemv_any(&exec, &q.wv, &sc.d_xn, &mut sc.d_v)?;
                    if let Some(g) = &q.g_proj {
                        gemv_any(&exec, g, &sc.d_xn, &mut sc.d_gate_h)?;
                    }
                }
                Proj::Bf16 { wqkv, .. } => super::head::bf16_qkv(
                    &exec,
                    wqkv,
                    &sc.d_xn,
                    None,
                    false,
                    &mut sc.d_q,
                    &mut sc.d_k,
                    &mut sc.d_v,
                    n_heads * head_dim,
                    kv_dim,
                    1,
                )?,
            }
            exec.rmsnorm_batch(
                &sc.d_q,
                &layer.q_norm.buf,
                &mut sc.d_qn,
                head_dim,
                eps,
                n_heads,
            )?;
            exec.rmsnorm_batch(
                &sc.d_k,
                &layer.k_norm.buf,
                &mut sc.d_kn,
                head_dim,
                eps,
                n_kv_heads,
            )?;
            if layer.is_swa {
                // plain full-rotary rope, θ 10k (ext_factor 0 in the params)
                exec.rope_yarn_batch(&mut sc.d_qn, &ds.d_pos, n_heads, head_dim, hp.rope_swa, 1)?;
                exec.rope_yarn_batch(
                    &mut sc.d_kn,
                    &ds.d_pos,
                    n_kv_heads,
                    head_dim,
                    hp.rope_swa,
                    1,
                )?;
            } else if !layer.nope {
                // partial-rotary YaRN over n_rot of head_dim - the mrope
                // kernel's n_rot seam with all-equal text positions
                exec.mrope(
                    &mut sc.d_qn,
                    &ds.d_mrope,
                    1,
                    n_heads,
                    head_dim,
                    hp.n_rot,
                    hp.rope_full,
                    sections,
                )?;
                exec.mrope(
                    &mut sc.d_kn,
                    &ds.d_mrope,
                    1,
                    n_kv_heads,
                    head_dim,
                    hp.n_rot,
                    hp.rope_full,
                    sections,
                )?;
            }
            exec.kv_append_batch(
                &sc.d_kn,
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
            let window = if layer.is_swa { hp.swa_window } else { 0 };
            exec.attn_decode_batch(
                &sc.d_qn,
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
                window,
                1,
                scale,
                kv_dtype,
            )?;
            if layer.has_gate() {
                exec.mul_softplus_head(&mut sc.d_attn, &sc.d_gate_h, n_heads, head_dim, 1)?;
            }
            match &layer.proj {
                Proj::Quant(q) => gemv_any(&exec, &q.wo, &sc.d_attn, &mut sc.d_proj)?,
                Proj::Bf16 { wo, .. } => exec.bf16_gemm(wo, None, &sc.d_attn, &mut sc.d_proj, 1)?,
            }
            if let Some(post) = &layer.post_attn_norm {
                // sandwich: x += rmsnorm(attn_out) * post, then the pre-FFN norm
                exec.rmsnorm_add_scale(&mut sc.d_x, &sc.d_proj, &post.buf, embd, eps, 1.0, 1)?;
                exec.rmsnorm_batch(&sc.d_x, &layer.ffn_norm.buf, &mut sc.d_xn, embd, eps, 1)?;
            } else {
                exec.add_rmsnorm_batch(
                    &mut sc.d_x,
                    &sc.d_proj,
                    &layer.ffn_norm.buf,
                    &mut sc.d_xn,
                    embd,
                    eps,
                    1,
                )?;
            }

            match &layer.ffn {
                Ffn::Dense { gate, up, down } => {
                    gemv_any(&exec, gate, &sc.d_xn, &mut sc.d_ffn_gate)?;
                    gemv_any(&exec, up, &sc.d_xn, &mut sc.d_ffn_up)?;
                    exec.swiglu(&mut sc.d_ffn_gate, &sc.d_ffn_up, gate.dims()[1])?;
                    gemv_any(&exec, down, &sc.d_ffn_gate, &mut sc.d_proj)?;
                }
                Ffn::MoeNv(w) => {
                    // W4A16 routed GEMVs off f32 + the BF16 shared expert
                    exec.matvec_f32_batch(&w.router_w, &sc.d_xn, &mut sc.d_moe_logits, 1)?;
                    moe_dims.route(
                        &exec,
                        &sc.d_moe_logits,
                        &w.probs_bias,
                        &mut sc.d_moe_idx,
                        &mut sc.d_moe_w,
                        1,
                    )?;
                    let k = moe_dims.n_active;
                    exec.q4x_moe_gu_swiglu(
                        &w.gate,
                        &w.up,
                        &sc.d_moe_idx,
                        &sc.d_xn,
                        &mut sc.d_moe_fused,
                        k,
                        1,
                    )?;
                    exec.nvf4_moe_down_acc(
                        &w.down,
                        &sc.d_moe_idx,
                        &sc.d_moe_w,
                        &sc.d_moe_fused,
                        &mut sc.d_proj,
                        None,
                        k,
                        1,
                        false,
                    )?;
                    exec.bf16_gemm(&w.sh_gate, None, &sc.d_xn, &mut sc.d_sh_gate, 1)?;
                    exec.bf16_gemm(&w.sh_up, None, &sc.d_xn, &mut sc.d_sh_up, 1)?;
                    exec.swiglu(&mut sc.d_sh_gate, &sc.d_sh_up, moe_dims.shexp_ff)?;
                    exec.bf16_gemm(&w.sh_down, None, &sc.d_sh_gate, &mut sc.d_sh_out, 1)?;
                    exec.add(&mut sc.d_proj, &sc.d_sh_out, embd)?;
                }
                Ffn::Moe(w) => {
                    exec.quantize_q8(&sc.d_xn, &mut sc.d_moe_xq, &mut sc.d_moe_xs, embd)?;
                    exec.matvec_f32_batch(&w.router_w, &sc.d_xn, &mut sc.d_moe_logits, 1)?;
                    moe_dims.route(
                        &exec,
                        &sc.d_moe_logits,
                        &w.probs_bias,
                        &mut sc.d_moe_idx,
                        &mut sc.d_moe_w,
                        1,
                    )?;
                    match (&w.gate_exps, &w.up_exps) {
                        (ExpW::Kq(g), ExpW::Kq(u)) => {
                            let needs =
                                crate::gpu::kq_needs_sums(g.ty) || crate::gpu::kq_needs_sums(u.ty);
                            if needs {
                                exec.q8_sums_strided(&sc.d_moe_xq, &mut sc.d_ssums, embd, 1)?;
                            }
                            exec.kquant_moe_gate_up(
                                g,
                                u,
                                &sc.d_moe_idx,
                                &sc.d_moe_xq,
                                &sc.d_moe_xs,
                                needs.then_some(&sc.d_ssums),
                                &mut sc.d_moe_fused,
                                moe_dims.n_active,
                                1,
                            )?;
                        }
                        _ => {
                            let g8 = match &w.gate_exps {
                                ExpW::Q8(g) => g,
                                ExpW::Kq(_) => unreachable!("loader pairs gate/up residency"),
                            };
                            let u8_ = match &w.up_exps {
                                ExpW::Q8(u) => u,
                                ExpW::Kq(_) => unreachable!("loader pairs gate/up residency"),
                            };
                            exec.q8_0_moe_gate_up(
                                g8,
                                u8_,
                                &sc.d_moe_idx,
                                &sc.d_moe_xq,
                                &sc.d_moe_xs,
                                &mut sc.d_moe_fused,
                                moe_dims.n_active,
                                1,
                            )?;
                        }
                    }
                    exec.quantize_q8(
                        &sc.d_moe_fused,
                        &mut sc.d_moe_fq,
                        &mut sc.d_moe_fs,
                        moe_dims.n_active * moe_dims.moe_ff,
                    )?;
                    match &w.down_exps {
                        ExpW::Kq(d) => {
                            let needs = crate::gpu::kq_needs_sums(d.ty);
                            if needs {
                                exec.q8_sums_strided(
                                    &sc.d_moe_fq,
                                    &mut sc.d_ssums,
                                    moe_dims.moe_ff,
                                    moe_dims.n_active,
                                )?;
                            }
                            exec.kquant_moe_down(
                                d,
                                &sc.d_moe_idx,
                                &sc.d_moe_w,
                                &sc.d_moe_fq,
                                &sc.d_moe_fs,
                                needs.then_some(&sc.d_ssums),
                                &mut sc.d_proj,
                                moe_dims.n_active,
                                1,
                            )?;
                        }
                        ExpW::Q8(d) => {
                            exec.q8_0_moe_down(
                                d,
                                &sc.d_moe_idx,
                                &sc.d_moe_w,
                                &sc.d_moe_fq,
                                &sc.d_moe_fs,
                                &mut sc.d_proj,
                                moe_dims.n_active,
                                1,
                            )?;
                        }
                    }
                    // Shared expert: plain SwiGLU, always on - added without
                    // any gate (the Laguna delta vs qwen35moe's sigmoid
                    // scalar).
                    gemv_any(&exec, &w.shexp_gate, &sc.d_xn, &mut sc.d_sh_gate)?;
                    gemv_any(&exec, &w.shexp_up, &sc.d_xn, &mut sc.d_sh_up)?;
                    exec.swiglu(&mut sc.d_sh_gate, &sc.d_sh_up, moe_dims.shexp_ff)?;
                    gemv_any(&exec, &w.shexp_down, &sc.d_sh_gate, &mut sc.d_sh_out)?;
                    exec.add(&mut sc.d_proj, &sc.d_sh_out, embd)?;
                }
            }
            match &layer.post_ffn_norm {
                Some(post) => {
                    exec.rmsnorm_add_scale(&mut sc.d_x, &sc.d_proj, &post.buf, embd, eps, 1.0, 1)?
                }
                None => exec.add(&mut sc.d_x, &sc.d_proj, embd)?,
            }
        }

        exec.rmsnorm_batch(&sc.d_x, &self.output_norm.buf, &mut sc.d_xn, embd, eps, 1)?;
        match &self.lm_head {
            Head::Quant(q) => gemv_any(&exec, q, &sc.d_xn, &mut sc.d_logits)?,
            Head::Bf16(w) => exec.bf16_gemm(w, None, &sc.d_xn, &mut sc.d_logits, 1)?,
        }
        ds.pos += 1;
        let logits = exec.stream.clone_dtoh(&sc.d_logits).map_err(drv)?;
        Ok(logits)
    }
}

impl GpuLaguna {
    /// Select the KV cache element type (default [`KvDtype::Fp16`], greedy-exact).
    /// [`KvDtype::Fp8E4m3`] is a lossy opt-in throughput/memory mode. Drops any
    /// existing decode/batch/pipe state so the caches re-allocate at the new
    /// element size on the next `forward_one`/`enable_batch`; call it before
    /// serving. The DFlash drafter's own aux-feature KV (dflash.rs) is
    /// independent of this and stays pinned to f16 - see dflash.rs's own
    /// numeric-class notes.
    pub fn set_kv_dtype(&mut self, dtype: KvDtype) {
        self.pipe_abort();
        self.kv_dtype = dtype;
        self.decode = None;
        self.scratch = None;
        self.batch = None;
        self.chunked.clear();
    }
}

impl Generator for GpuLaguna {
    fn release_idle_memory(&mut self) {
        self.exec.trim_mem_pool();
    }

    fn reset(&mut self) {
        self.pipe_abort(); // a leftover pipe must never survive into a fresh serve
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

    // ── continuous-batching lanes (batch.rs) ────────────────────────────────

    fn enable_batch(&mut self, max_batch: usize) -> Result<usize, GenError> {
        self.enable_batch_impl(max_batch).map_err(gen_err)
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

    // ---- stall-free batching (Sarathi-Serve Algorithm 3) ------------------
    // Laguna served admissions through the CLASSIC path at first: every
    // pending prompt prefilled to completion before any decode row moved, so
    // an admission wave froze every live stream (the paper's "generation
    // stall" - a flat ~9 s cohort TTFT at 1k×1k c32). These three methods
    // put it on the mixed path.

    /// The serial prefill API, routed through the batched lane once one
    /// exists (granite's shape). Without this the trait default loops
    /// `forward(t)`, whose serial state is a DENSE max_ctx KV for all 40
    /// layers - 20 GiB at 131k on the XS - built by the service's post-enable
    /// warm-up and then held for the life of a server that never touches the
    /// serial path again (measured 2026-09-06: 51.6 GiB with, 31.6 without).
    fn forward_prefill_stream(&mut self, tokens: &[u32]) -> Result<Vec<f32>, GenError> {
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
        // a pass with no decode band takes the wide chunk (Kolibri); one with
        // decode rows keeps pf_rows - every band row waits on the whole pass
        self.batch.as_ref().map_or(0, |bs| {
            let rows = if decode_rows == 0 {
                bs.pf_wide
            } else {
                super::batch::pf_rows()
            };
            rows.min(bs.cap.saturating_sub(decode_rows))
        })
    }

    fn prefill_begin(&mut self, slot: usize, tokens: Vec<u32>) -> Result<(), GenError> {
        self.prefill_begin_impl(slot, tokens, &[])
            .map(|_| ())
            .map_err(gen_err)
    }

    fn prefill_begin_hinted(
        &mut self,
        slot: usize,
        tokens: Vec<u32>,
        hints: &[usize],
    ) -> Result<usize, GenError> {
        self.prefill_begin_impl(slot, tokens, hints)
            .map_err(gen_err)
    }

    fn prefix_share_floor(&self) -> Option<usize> {
        self.prefix_share_floor_impl()
    }

    fn prefill_abort(&mut self, slot: usize) -> bool {
        self.prefill_abort_impl(slot)
    }

    fn forward_mixed(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
    ) -> Result<(Vec<f32>, Vec<(usize, Vec<f32>, usize)>), GenError> {
        // One fused pass - decode rows and the chunk share a single weight
        // stream. The two-pass form (prefill_advance then batch_step_slots)
        // doubled the tick's 18.9 GB expert read and cost -30% on c8.
        self.forward_mixed_fused(decodes, budget).map_err(gen_err)
    }

    fn forward_mixed_sampled(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
        plans: &[crate::generator::RowSample],
        _fin_plans: &[(usize, crate::generator::RowSample)],
    ) -> Result<
        (
            crate::generator::SampledStep,
            Vec<(usize, crate::generator::FinishSample, usize)>,
        ),
        GenError,
    > {
        use crate::generator::FinishSample;
        // The tick the scheduler actually takes (device sampling is on for
        // laguna), so the fused single-pass form has to be here - putting it
        // only behind `forward_mixed` left the two-pass weight tax in place
        // on every served tick.
        let (step, finished) = self
            .forward_mixed_fused_sampled(decodes, budget, plans)
            .map_err(gen_err)?;
        let finished: Vec<(usize, FinishSample, usize)> = finished
            .into_iter()
            .map(|(slot, logits, rows)| (slot, FinishSample::Logits(logits), rows))
            .collect();
        Ok((step, finished))
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

    fn supports_decode_pipe(&self) -> bool {
        self.supports_decode_pipe_impl()
    }

    fn decode_pipe_begin(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        plans: &[crate::generator::RowSample],
    ) -> Result<(), GenError> {
        self.decode_pipe_begin_impl(None, tokens, positions, plans)
            .map_err(gen_err)
    }

    fn decode_pipe_begin_slots(
        &mut self,
        slots: &[u32],
        tokens: &[u32],
        positions: &[u32],
        plans: &[crate::generator::RowSample],
    ) -> Result<(), GenError> {
        self.decode_pipe_begin_impl(Some(slots), tokens, positions, plans)
            .map_err(gen_err)
    }

    fn decode_pipe_next(
        &mut self,
        plans: &[crate::generator::RowSample],
    ) -> Result<Vec<u32>, GenError> {
        self.decode_pipe_next_impl(plans).map_err(gen_err)
    }

    fn decode_pipe_drain(&mut self) -> Result<Vec<u32>, GenError> {
        self.decode_pipe_drain_impl().map_err(gen_err)
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

    fn spec_capable(&self) -> bool {
        // routes single-user serving through the batched loop, where the
        // DFlash spec rounds run (attach-time facts only - stable before
        // enable_batch)
        self.serve_spec_on()
    }

    fn spec_ensure_warm(
        &mut self,
        slot: usize,
        _committed: &[u32],
        want_pos: u32,
    ) -> Result<bool, GenError> {
        // watermark check only - no token-replay re-warm (features flow from
        // every batched forward while armed, see dflash.rs)
        Ok(self.spec_ensure_warm_impl(slot, want_pos))
    }

    fn spec_draft_batch(
        &mut self,
        pendings: &[(usize, u32)],
        k: usize,
    ) -> Result<Option<Vec<Vec<u32>>>, GenError> {
        self.spec_draft_batch_impl(pendings, k).map_err(gen_err)
    }

    fn forward_spec_batch(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
    ) -> Result<Option<Vec<u32>>, GenError> {
        self.forward_spec_batch_impl(reqs).map_err(gen_err)
    }

    fn forward_spec_batch_plans(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
        plans: &[crate::sampler::DevicePlan],
    ) -> Result<Option<Vec<u32>>, GenError> {
        self.forward_spec_batch_plans_impl(reqs, plans)
            .map_err(gen_err)
    }

    fn tier_pump(&mut self) {
        self.tier_pump_impl();
    }
    fn tier_prefix_loading(&mut self, slot: usize, tokens: &[u32]) -> bool {
        self.tier_consult_impl(slot, tokens)
    }
    fn tier_observe_prefill(&mut self, tokens: u32, wall_us: f64) {
        if let Some(t) = self
            .batch
            .as_mut()
            .and_then(|b| b.prefix.as_mut())
            .and_then(|p| p.tier.as_mut())
        {
            t.cost.observe_prefill(tokens, wall_us);
        }
    }
    fn tier_stats(&self) -> Option<crate::kv_tier::TierStats> {
        self.batch
            .as_ref()?
            .prefix
            .as_ref()?
            .tier
            .as_ref()
            .map(|t| t.tier_stats())
    }
    fn tier_report(&self) -> Option<crate::kv_tier::TierReport> {
        self.batch
            .as_ref()?
            .prefix
            .as_ref()?
            .tier
            .as_ref()
            .map(crate::kv_tier::PoolTier::report)
    }
    fn release_inactive_slots(&mut self, occupied: &[bool]) {
        self.release_inactive_slots_impl(occupied);
    }

    fn take_prefill_reused(&mut self, slot: usize) -> usize {
        self.batch
            .as_mut()
            .and_then(|b| b.prefix.as_mut())
            .and_then(|pf| {
                let n = *pf.last_reused.get(slot)?;
                pf.last_reused[slot] = 0;
                Some(n)
            })
            .unwrap_or(0)
    }

    fn kv_mem_bytes(&self) -> Option<u64> {
        self.kv_mem_bytes_impl()
    }

    fn pool_free_blocks(&self) -> Option<usize> {
        self.pool_free_blocks_impl()
    }

    fn device_mem_used(&self) -> Option<u64> {
        self.exec.process_mem_used()
    }
}
