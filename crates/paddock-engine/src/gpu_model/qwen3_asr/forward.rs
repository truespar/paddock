//! Qwen3-ASR serial lane - batch-1 decode + one batched-row multimodal
//! prefill (the granite bring-up shape: plain launches, correctness first;
//! the batch/chunked/prefix lanes follow after the parity gate). Op order
//! per layer, matching upstream Qwen3 dense exactly:
//!
//! ```text
//! x   = tok_embd[t]            (audio rows overwritten by tower embeds)
//! h   = RMSNorm(x); q,k,v = Wq·h, Wk·h, Wv·h
//! q,k = rope_neox(rms_head_norm(q), rms_head_norm(k))   theta 1e6
//! x  += Wo · Attn(q, K, V)     causal, scale 1/sqrt(128), no sinks
//! x  += Wdown · SwiGLU(Wgate·RMSNorm(x), Wup·RMSNorm(x))
//! head: logits = lm_head · RMSNorm(x)   (tied to tok_embd)
//! ```
//!
//! The prefill runs all prompt rows in one pass through the tuned Q8_0
//! prefill stack (qwen35::ops `prefill_*` helpers - the qwen3.rs precedent)
//! with per-row causal attention via `attn_decode_batch` (each row reads KV
//! up to its own position; K/V for the whole prompt are appended before the
//! attention reads them, which is causal within slot 0).

use cudarc::driver::CudaSlice;

use crate::generator::{GenError, Generator};
use crate::gpu::KvDtype;
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::qwen35::{
    gemv_any, prefill_add_norm_quant, prefill_ffn_down_any, prefill_mm_pre_any,
};
use crate::service::MmChunk;

use super::{AudioOutput, GpuQwen3Asr};

fn gen_err(e: GpuModelError) -> GenError {
    match e {
        GpuModelError::PoolExhausted => GenError::PoolExhausted,
        other => GenError::Backend(other.to_string()),
    }
}

/// A vision tower's output spliced over its `<|image_pad|>` rows of a
/// prompt: `embd` replaces rows `off .. off + n_tokens`, and `deepstack[i]`
/// (same shape) is added to the residual at those rows after decoder layer
/// `i`. What qwen-image's text encoder (Qwen3-VL-8B) reads a reference
/// picture through.
pub(crate) struct VisionSplice<'a> {
    pub off: usize,
    pub n_tokens: usize,
    pub embd: &'a CudaSlice<f32>,
    pub deepstack: &'a [CudaSlice<f32>],
}

/// Per-sequence decode state: dense per-layer KV at slot 0 (full attention
/// on every layer - no rings, no recurrent state).
pub(crate) struct DecodeState {
    pub kv_k: Vec<CudaSlice<u8>>,
    pub kv_v: Vec<CudaSlice<u8>>,
    pub pos: usize,
    pub d_token: CudaSlice<u32>,
    pub d_pos: CudaSlice<u32>,
    /// constant [0] - slot 0.
    pub d_slots: CudaSlice<u32>,
}

/// Decode-step scratch (r = 1), allocated once.
pub(crate) struct Scratch {
    pub d_x: CudaSlice<f32>,
    pub d_xn: CudaSlice<f32>,
    pub d_q: CudaSlice<f32>,
    pub d_qn: CudaSlice<f32>,
    pub d_k: CudaSlice<f32>,
    pub d_kn: CudaSlice<f32>,
    pub d_v: CudaSlice<f32>,
    pub d_attn: CudaSlice<f32>,
    pub d_proj: CudaSlice<f32>,
    /// no-op sink logits (Qwen3 has none) - the -inf identity.
    pub d_sinks: CudaSlice<f32>,
    pub d_ffn_gate: CudaSlice<f32>,
    pub d_ffn_up: CudaSlice<f32>,
    pub d_logits: CudaSlice<f32>,
}

/// Batched-row prefill scratch, sized to the prompt high-water mark.
pub(crate) struct PrefillScratch {
    pub rows: usize,
    pub d_x: CudaSlice<f32>,
    pub d_xn: CudaSlice<f32>,
    pub d_q: CudaSlice<f32>,
    pub d_qn: CudaSlice<f32>,
    pub d_k: CudaSlice<f32>,
    pub d_kn: CudaSlice<f32>,
    pub d_v: CudaSlice<f32>,
    pub d_attn: CudaSlice<f32>,
    pub d_proj: CudaSlice<f32>,
    pub d_ffn_gate: CudaSlice<f32>,
    pub d_ffn_up: CudaSlice<f32>,
    pub up_tokens: CudaSlice<u32>,
    pub up_pos: CudaSlice<u32>,
    pub up_slots: CudaSlice<u32>,
    // quantize staging for the tuned prefill helpers
    pub d_pxq: CudaSlice<i8>,
    pub d_pxs: CudaSlice<f32>,
    pub d_yq: CudaSlice<u8>,
    pub d_skfix: CudaSlice<f32>,
    pub xsums: CudaSlice<f32>,
    pub ssums: CudaSlice<f32>,
}

impl GpuQwen3Asr {
    fn ensure_decode(&mut self) -> Result<(), GpuModelError> {
        if self.decode.is_some() && self.scratch.is_some() {
            return Ok(());
        }
        let e = &self.exec;
        let hp = &self.hp;
        let kv_dim = hp.n_kv_heads * hp.head_dim;
        let kv_bytes = self.kv_dtype.bytes();
        let (mut kv_k, mut kv_v) = (
            Vec::with_capacity(hp.n_layer),
            Vec::with_capacity(hp.n_layer),
        );
        for _ in 0..hp.n_layer {
            kv_k.push(e.alloc_u8(self.max_ctx * kv_dim * kv_bytes)?);
            kv_v.push(e.alloc_u8(self.max_ctx * kv_dim * kv_bytes)?);
        }
        // Announce the class the pool was actually sized for, not the one
        // that was asked for - the stamp has to come off the allocation.
        tracing::info!(
            kv = if kv_bytes == 1 {
                "fp8-e4m3 (default)"
            } else {
                "f16 (--kv-cache-dtype f16)"
            },
            mib = (2 * hp.n_layer * self.max_ctx * kv_dim * kv_bytes) as u64 / (1 << 20),
            "qwen3-asr kv cache resident"
        );
        self.decode = Some(DecodeState {
            kv_k,
            kv_v,
            pos: 0,
            d_token: e.alloc_u32(1)?,
            d_pos: e.alloc_u32(1)?,
            d_slots: e.alloc_u32(1)?, // zeroed -> slot 0
        });
        let q_dim = hp.n_heads * hp.head_dim;
        self.scratch = Some(Scratch {
            d_x: e.alloc(hp.n_embd)?,
            d_xn: e.alloc(hp.n_embd)?,
            d_q: e.alloc(q_dim)?,
            d_qn: e.alloc(q_dim)?,
            d_k: e.alloc(kv_dim)?,
            d_kn: e.alloc(kv_dim)?,
            d_v: e.alloc(kv_dim)?,
            d_attn: e.alloc(q_dim)?,
            d_proj: e.alloc(hp.n_embd)?,
            d_sinks: e.alloc_no_sinks(hp.n_heads)?,
            d_ffn_gate: e.alloc(hp.n_ff)?,
            d_ffn_up: e.alloc(hp.n_ff)?,
            d_logits: e.alloc(hp.n_vocab)?,
        });
        Ok(())
    }

    fn ensure_prefill(&mut self, rows: usize) -> Result<(), GpuModelError> {
        if self.prefill.as_ref().is_some_and(|p| p.rows >= rows) {
            return Ok(());
        }
        self.prefill = None;
        let e = &self.exec;
        let hp = &self.hp;
        let q_dim = hp.n_heads * hp.head_dim;
        let kv_dim = hp.n_kv_heads * hp.head_dim;
        // widest input any prefill GEMM quantizes (hidden or swiglu output)
        let wide = hp.n_embd.max(hp.n_ff);
        self.prefill = Some(PrefillScratch {
            rows,
            d_x: e.alloc(rows * hp.n_embd)?,
            d_xn: e.alloc(rows * hp.n_embd)?,
            d_q: e.alloc(rows * q_dim)?,
            d_qn: e.alloc(rows * q_dim)?,
            d_k: e.alloc(rows * kv_dim)?,
            d_kn: e.alloc(rows * kv_dim)?,
            d_v: e.alloc(rows * kv_dim)?,
            d_attn: e.alloc(rows * q_dim)?,
            d_proj: e.alloc(rows * hp.n_embd)?,
            d_ffn_gate: e.alloc(rows * hp.n_ff)?,
            d_ffn_up: e.alloc(rows * hp.n_ff)?,
            up_tokens: e.alloc_u32(rows)?,
            up_pos: e.alloc_u32(rows)?,
            up_slots: e.alloc_u32(rows)?, // zeroed -> slot 0 for every row
            d_pxq: e.alloc_i8(rows * wide)?,
            d_pxs: e.alloc(rows * wide / 32)?,
            d_yq: e.alloc_u8(wide.div_ceil(128) * rows.next_multiple_of(128) * 144)?,
            d_skfix: e.alloc(256 * 128 * 128 + 256)?,
            xsums: e.alloc(wide.div_ceil(128) * rows.next_multiple_of(128) * 4)?,
            ssums: e.alloc(rows * wide / 16)?,
        });
        Ok(())
    }

    /// Batched-row prefill of the whole prompt at slot 0, with audio-tower
    /// embeddings spliced over their placeholder rows. Returns the last
    /// row's logits; leaves `decode.pos` at the prompt length.
    fn prefill_rows(
        &mut self,
        ids: &[u32],
        splices: &[(usize, AudioOutput)],
    ) -> Result<Vec<f32>, GpuModelError> {
        let r = ids.len();
        self.prefill_body(ids, splices, &[], None)?;
        // last-row head
        let exec = self.exec.clone();
        let (embd, eps) = (self.hp.n_embd, self.hp.eps);
        let sc = self.prefill.as_mut().expect("prefill scratch");
        let dsc = self.scratch.as_mut().expect("scratch");
        exec.copy_region(&sc.d_x, (r - 1) * embd, &mut dsc.d_x, 0, embd)?;
        exec.rmsnorm_batch(&dsc.d_x, &self.output_norm.buf, &mut dsc.d_xn, embd, eps, 1)?;
        gemv_any(&exec, &self.lm_head, &dsc.d_xn, &mut dsc.d_logits)?;
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
        let logits = exec.stream.clone_dtoh(&dsc.d_logits).map_err(drv)?;
        Ok(logits)
    }

    /// The prefill's layer stack alone: runs every prompt row through the
    /// decoder at slot 0, leaves the FINAL hidden states in the prefill
    /// scratch's `d_x` and advances `decode.pos`. The head is the caller's -
    /// the serving prefill reads the last row, the aligner (aligner.rs) reads
    /// the `<timestamp>` rows, qwen-image's text encoder reads them all.
    ///
    /// `vision` splices a vision tower's output over its `<|image_pad|>`
    /// rows the way `splices` does an audio tower's, and adds each DeepStack
    /// stream to the residual after the decoder layer of its index; `mrope`
    /// (axis-major `[4][ids.len()]`, t / h / w / e) then rotates every row by
    /// its own multi-axis position with the interleaved sections - text rows
    /// with equal axes come out exactly as the plain rope they get otherwise.
    pub(crate) fn prefill_body(
        &mut self,
        ids: &[u32],
        splices: &[(usize, AudioOutput)],
        vision: &[VisionSplice<'_>],
        mrope: Option<&[u32]>,
    ) -> Result<(), GpuModelError> {
        let r = ids.len();
        assert!(r > 0);
        self.ensure_decode()?;
        if self.decode.as_ref().expect("decode").pos + r > self.max_ctx {
            return Err(GpuModelError::Unsupported(format!(
                "context full: prompt of {r} rows at max_ctx {}",
                self.max_ctx
            )));
        }
        self.ensure_prefill(r)?;
        let exec = self.exec.clone();
        let hp = &self.hp;
        let (embd, n_heads, n_kv_heads, head_dim, n_ff) =
            (hp.n_embd, hp.n_heads, hp.n_kv_heads, hp.head_dim, hp.n_ff);
        let kv_dim = n_kv_heads * head_dim;
        let (eps, rope, kv_dtype) = (hp.eps, hp.rope, self.kv_dtype);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let ds = self.decode.as_mut().expect("decode");
        let sc = self.prefill.as_mut().expect("prefill scratch");
        let start = ds.pos as u32;
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);

        let positions: Vec<u32> = (0..r as u32).map(|i| start + i).collect();
        exec.stream
            .memcpy_htod(ids, &mut sc.up_tokens)
            .map_err(drv)?;
        exec.stream
            .memcpy_htod(&positions, &mut sc.up_pos)
            .map_err(drv)?;

        match &self.tok_embd {
            super::TokEmbd::Q8(t) => {
                exec.embed_gather_batch_q8(t, &sc.up_tokens, &mut sc.d_x, embd, r)?
            }
            super::TokEmbd::Kq(t) => exec.kquant_gather(t, &sc.up_tokens, &mut sc.d_x, embd, r)?,
        }
        // audio / vision splice: tower embeddings overwrite their placeholder
        // rows
        for (off, out) in splices {
            exec.copy_region(&out.embd, 0, &mut sc.d_x, off * embd, out.n_tokens * embd)?;
        }
        for v in vision {
            exec.copy_region(v.embd, 0, &mut sc.d_x, v.off * embd, v.n_tokens * embd)?;
        }
        let d_mrope = match mrope {
            Some(m) => {
                assert_eq!(m.len(), 4 * r, "mrope positions are [4][rows]");
                Some(exec.to_device_u32(m)?)
            }
            None => None,
        };
        let (n_rot, sections) = (hp.n_rot, hp.sections);

        for (li, layer) in self.layers.iter().enumerate() {
            prefill_add_norm_quant(
                &exec,
                &mut sc.d_x,
                None,
                false,
                &layer.attn_norm.buf,
                &mut sc.d_xn,
                false,
                &mut sc.d_pxq,
                &mut sc.d_pxs,
                &mut sc.d_yq,
                embd,
                r,
                eps,
            )?;
            prefill_mm_pre_any(
                &exec,
                &layer.wq,
                &sc.d_pxq,
                &sc.d_pxs,
                &sc.d_yq,
                &mut sc.xsums,
                &mut sc.ssums,
                &mut sc.d_skfix,
                &mut sc.d_q,
                r,
            )?;
            prefill_mm_pre_any(
                &exec,
                &layer.wk,
                &sc.d_pxq,
                &sc.d_pxs,
                &sc.d_yq,
                &mut sc.xsums,
                &mut sc.ssums,
                &mut sc.d_skfix,
                &mut sc.d_k,
                r,
            )?;
            prefill_mm_pre_any(
                &exec,
                &layer.wv,
                &sc.d_pxq,
                &sc.d_pxs,
                &sc.d_yq,
                &mut sc.xsums,
                &mut sc.ssums,
                &mut sc.d_skfix,
                &mut sc.d_v,
                r,
            )?;
            // per-head q/k RMS norm, then plain NEOX rope
            exec.rmsnorm_batch(
                &sc.d_q,
                &layer.q_norm.buf,
                &mut sc.d_qn,
                head_dim,
                eps,
                r * n_heads,
            )?;
            exec.rmsnorm_batch(
                &sc.d_k,
                &layer.k_norm.buf,
                &mut sc.d_kn,
                head_dim,
                eps,
                r * n_kv_heads,
            )?;
            match &d_mrope {
                Some(pos) => {
                    exec.imrope(
                        &mut sc.d_qn,
                        pos,
                        r,
                        n_heads,
                        head_dim,
                        n_rot,
                        rope,
                        sections,
                    )?;
                    exec.imrope(
                        &mut sc.d_kn,
                        pos,
                        r,
                        n_kv_heads,
                        head_dim,
                        n_rot,
                        rope,
                        sections,
                    )?;
                }
                None => {
                    exec.rope_yarn_batch(&mut sc.d_qn, &sc.up_pos, n_heads, head_dim, rope, r)?;
                    exec.rope_yarn_batch(&mut sc.d_kn, &sc.up_pos, n_kv_heads, head_dim, rope, r)?;
                }
            }
            exec.kv_append_batch(
                &sc.d_kn,
                &mut ds.kv_k[li],
                &sc.up_pos,
                Some(&sc.up_slots),
                kv_dim,
                self.max_ctx,
                r,
                kv_dtype,
            )?;
            exec.kv_append_batch(
                &sc.d_v,
                &mut ds.kv_v[li],
                &sc.up_pos,
                Some(&sc.up_slots),
                kv_dim,
                self.max_ctx,
                r,
                kv_dtype,
            )?;
            // causal per-row attention: each row reads slot 0 KV up to its
            // position (all K/V rows are appended above before any read)
            let sinks = &self.scratch.as_ref().expect("scratch").d_sinks;
            exec.attn_decode_batch(
                &sc.d_qn,
                &ds.kv_k[li],
                &ds.kv_v[li],
                sinks,
                &mut sc.d_attn,
                &sc.up_pos,
                Some(&sc.up_slots),
                n_heads,
                n_kv_heads,
                head_dim,
                self.max_ctx,
                kv_dim,
                0,
                r,
                scale,
                kv_dtype,
            )?;
            crate::gpu_model::qwen35::prefill_mm_any(
                &exec,
                &layer.wo,
                &mut sc.d_pxq,
                &mut sc.d_pxs,
                &mut sc.d_yq,
                &mut sc.xsums,
                &mut sc.ssums,
                &mut sc.d_skfix,
                &sc.d_attn,
                &mut sc.d_proj,
                r,
            )?;

            prefill_add_norm_quant(
                &exec,
                &mut sc.d_x,
                Some(&sc.d_proj),
                false,
                &layer.ffn_norm.buf,
                &mut sc.d_xn,
                false,
                &mut sc.d_pxq,
                &mut sc.d_pxs,
                &mut sc.d_yq,
                embd,
                r,
                eps,
            )?;
            prefill_mm_pre_any(
                &exec,
                &layer.gate,
                &sc.d_pxq,
                &sc.d_pxs,
                &sc.d_yq,
                &mut sc.xsums,
                &mut sc.ssums,
                &mut sc.d_skfix,
                &mut sc.d_ffn_gate,
                r,
            )?;
            prefill_mm_pre_any(
                &exec,
                &layer.up,
                &sc.d_pxq,
                &sc.d_pxs,
                &sc.d_yq,
                &mut sc.xsums,
                &mut sc.ssums,
                &mut sc.d_skfix,
                &mut sc.d_ffn_up,
                r,
            )?;
            prefill_ffn_down_any(
                &exec,
                &layer.down,
                &mut sc.d_pxq,
                &mut sc.d_pxs,
                &mut sc.d_yq,
                &mut sc.xsums,
                &mut sc.ssums,
                &mut sc.d_skfix,
                &mut sc.d_ffn_gate,
                &sc.d_ffn_up,
                &mut sc.d_proj,
                n_ff,
                r,
            )?;
            exec.add(&mut sc.d_x, &sc.d_proj, r * embd)?;
            // DeepStack: stream `li` joins the residual at the image rows
            // after this layer, before the next layer's norm reads it
            for v in vision {
                if let Some(stream) = v.deepstack.get(li) {
                    exec.add_at(&mut sc.d_x, v.off * embd, stream, 0, v.n_tokens * embd)?;
                }
            }
        }

        let ds = self.decode.as_mut().expect("decode");
        ds.pos += r;
        Ok(())
    }

    /// One token through the whole stack; returns the full logits row.
    fn forward_one(&mut self, token: u32) -> Result<Vec<f32>, GpuModelError> {
        self.ensure_decode()?;
        let exec = self.exec.clone();
        let hp = &self.hp;
        let (embd, n_heads, n_kv_heads, head_dim, n_ff) =
            (hp.n_embd, hp.n_heads, hp.n_kv_heads, hp.head_dim, hp.n_ff);
        let kv_dim = n_kv_heads * head_dim;
        let (eps, rope, kv_dtype) = (hp.eps, hp.rope, self.kv_dtype);
        let scale = 1.0 / (head_dim as f32).sqrt();
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

        match &self.tok_embd {
            super::TokEmbd::Q8(t) => {
                exec.embed_gather_batch_q8(t, &ds.d_token, &mut sc.d_x, embd, 1)?
            }
            super::TokEmbd::Kq(t) => exec.kquant_gather(t, &ds.d_token, &mut sc.d_x, embd, 1)?,
        }

        for (li, layer) in self.layers.iter().enumerate() {
            exec.rmsnorm_batch(&sc.d_x, &layer.attn_norm.buf, &mut sc.d_xn, embd, eps, 1)?;
            gemv_any(&exec, &layer.wq, &sc.d_xn, &mut sc.d_q)?;
            gemv_any(&exec, &layer.wk, &sc.d_xn, &mut sc.d_k)?;
            gemv_any(&exec, &layer.wv, &sc.d_xn, &mut sc.d_v)?;
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
            exec.rope_yarn_batch(&mut sc.d_qn, &ds.d_pos, n_heads, head_dim, rope, 1)?;
            exec.rope_yarn_batch(&mut sc.d_kn, &ds.d_pos, n_kv_heads, head_dim, rope, 1)?;
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
                0,
                1,
                scale,
                kv_dtype,
            )?;
            gemv_any(&exec, &layer.wo, &sc.d_attn, &mut sc.d_proj)?;
            exec.add(&mut sc.d_x, &sc.d_proj, embd)?;

            exec.rmsnorm_batch(&sc.d_x, &layer.ffn_norm.buf, &mut sc.d_xn, embd, eps, 1)?;
            gemv_any(&exec, &layer.gate, &sc.d_xn, &mut sc.d_ffn_gate)?;
            gemv_any(&exec, &layer.up, &sc.d_xn, &mut sc.d_ffn_up)?;
            exec.swiglu(&mut sc.d_ffn_gate, &sc.d_ffn_up, n_ff)?;
            gemv_any(&exec, &layer.down, &sc.d_ffn_gate, &mut sc.d_proj)?;
            exec.add(&mut sc.d_x, &sc.d_proj, embd)?;
        }

        exec.rmsnorm_batch(&sc.d_x, &self.output_norm.buf, &mut sc.d_xn, embd, eps, 1)?;
        gemv_any(&exec, &self.lm_head, &sc.d_xn, &mut sc.d_logits)?;
        ds.pos += 1;
        let logits = exec.stream.clone_dtoh(&sc.d_logits).map_err(drv)?;
        Ok(logits)
    }

    /// Select the KV cache element type (default fp16, greedy-exact). Drops
    /// decode state so the caches re-allocate at the new element size.
    pub fn set_kv_dtype(&mut self, dtype: KvDtype) {
        self.kv_dtype = dtype;
        self.decode = None;
        self.scratch = None;
        self.prefill = None;
        self.batch = None;
        self.chunked.clear();
    }
}

impl Generator for GpuQwen3Asr {
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
        if let Some(ds) = self.decode.as_mut() {
            ds.pos = 0;
        }
        if let Some(bs) = self.batch.as_mut() {
            bs.slot0_pos = 0;
        }
        // KV needs no clearing: every attention read is position-bounded.
    }

    fn forward(&mut self, token: u32) -> Result<Vec<f32>, GenError> {
        // Batched engine: the serial Generator surface (warmup, any serial-
        // path caller) decodes through the batch lane at slot 0 instead of
        // resurrecting the dense serial KV next to the pool - the slot-0
        // cursor is set by whichever prefill entry ran last.
        if self.batch.is_some() {
            let pos = self.batch.as_ref().expect("batch").slot0_pos;
            if pos >= self.max_ctx {
                return Err(GenError::Backend(format!(
                    "context full: {pos} tokens at max_ctx {}",
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
        if let Some(b) = self.batch_kv_mem_bytes() {
            return Some(b);
        }
        let kv_dim = self.hp.n_kv_heads * self.hp.head_dim;
        Some((self.hp.n_layer * 2 * self.max_ctx * kv_dim * self.kv_dtype.bytes()) as u64)
    }

    // ── the batched serving lane (batch.rs) ──────────────────────

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

    fn forward_batch_sampled(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        plans: &[crate::generator::RowSample],
    ) -> Result<crate::generator::SampledStep, GenError> {
        self.forward_batch_sampled_impl(tokens, positions, plans)
            .map_err(gen_err)
    }

    fn forward_prefill(&mut self, slot: usize, tokens: &[u32]) -> Result<Vec<f32>, GenError> {
        self.forward_prefill_impl(slot, tokens).map_err(gen_err)
    }

    // ── stall-free chunked prefill + mixed ticks  ─────────

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

    /// Audio prompts join the chunked queue too: tower encodes run at
    /// admission (a few ms - no encoder budget needed at ASR scale) and the
    /// rows advance inside mixed ticks, so an admission never freezes the
    /// live streams. This is the c32 serial-admission fix.
    fn supports_chunked_multimodal(&self) -> bool {
        self.tower.is_some()
            && self.batch.is_some()
            && paddock_models::dev_var_os!("PADDOCK_MM_NO_CHUNK").is_none()
    }

    fn prefill_begin(&mut self, slot: usize, tokens: Vec<u32>) -> Result<(), GenError> {
        self.prefill_begin_impl(slot, tokens).map_err(gen_err)
    }

    fn prefill_begin_multimodal(
        &mut self,
        items: Vec<(usize, Vec<MmChunk>)>,
    ) -> Vec<(usize, crate::generator::MmAdmit)> {
        self.prefill_begin_multimodal_impl(items)
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

    /// Single-stream bulk prefill. Batched: slot 0 is this stream's own.
    /// Serial: the same batched-row pass the multimodal path takes (splice-
    /// free) - without this override the trait default feeds one token per
    /// `forward` call, so a text prompt prefills at decode speed.
    fn forward_prefill_stream(&mut self, tokens: &[u32]) -> Result<Vec<f32>, GenError> {
        if self.batch.is_some() {
            return self.forward_prefill_impl(0, tokens).map_err(gen_err);
        }
        self.prefill_rows(tokens, &[]).map_err(gen_err)
    }

    fn release_inactive_slots(&mut self, occupied: &[bool]) {
        self.release_inactive_slots_impl(occupied);
    }

    fn pool_free_blocks(&self) -> Option<usize> {
        self.pool_free_blocks_impl()
    }

    // ── audio (the mm-slot lane) ────────────────────────────────────────────

    /// Audio requests ride ordinary batch slots  - never the
    /// exclusive drain-the-server path - once the batch lane is up. This is
    /// the whisper.cpp-mutex differentiator: N concurrent transcriptions
    /// admit and decode together.
    fn supports_mm_slots(&self) -> bool {
        self.tower.is_some() && self.batch.is_some()
    }

    fn forward_prefill_multimodal(
        &mut self,
        slot: usize,
        chunks: &[MmChunk],
    ) -> Result<(Vec<f32>, usize), GenError> {
        self.forward_prefill_multimodal_impl(slot, chunks)
            .map_err(gen_err)
    }

    /// Whole-prompt multimodal prefill: text tokens embed normally, each
    /// audio chunk runs the mel frontend + tower and its embeddings splice
    /// over the placeholder rows. Returns (last-row logits, total rows) -
    /// rows is what the request bills as prompt_tokens.
    fn forward_multimodal(
        &mut self,
        chunks: &[MmChunk],
    ) -> Result<Option<(Vec<f32>, usize)>, GenError> {
        // Batched engine: the exclusive path is only ever reached by serial-
        // path callers (mm requests normally ride slots via supports_mm_slots)
        // - route it through slot 0 of the batch lane rather than the dead
        // serial planes.
        if self.batch.is_some() {
            return self
                .forward_prefill_multimodal_impl(0, chunks)
                .map(Some)
                .map_err(gen_err);
        }
        let mut ids: Vec<u32> = Vec::new();
        let mut splices: Vec<(usize, AudioOutput)> = Vec::new();
        for ch in chunks {
            match ch {
                MmChunk::Text(t) => ids.extend_from_slice(t),
                MmChunk::Audio { samples, mel } => {
                    let tower = self.tower.as_mut().ok_or_else(|| {
                        GenError::Backend(
                            "audio request but no audio tower attached (--mmproj)".into(),
                        )
                    })?;
                    let owned;
                    let mel = match mel {
                        Some(m) => m,
                        None => {
                            owned = crate::audio::qwen3_asr_features(samples)
                                .map_err(GenError::Backend)?;
                            &owned
                        }
                    };
                    let out = tower
                        .encode(mel)
                        .map_err(|e| GenError::Backend(e.to_string()))?;
                    let off = ids.len();
                    // placeholder rows (overwritten by the splice; id 0 is
                    // fine - the embed row is never read)
                    ids.extend(std::iter::repeat_n(0u32, out.n_tokens));
                    splices.push((off, out));
                }
                MmChunk::Image { .. } => {
                    return Err(GenError::Backend(
                        "qwen3-asr serves audio, not images".into(),
                    ));
                }
                MmChunk::OcrCrop(_) => {
                    return Err(GenError::Backend(
                        "OCR crop directive on qwen3-asr - routing bug".into(),
                    ));
                }
                MmChunk::VisionPixels { .. } => {
                    return Err(GenError::Backend(
                        "pixel-budget directive on qwen3-asr - routing bug".into(),
                    ));
                }
            }
        }
        if ids.is_empty() {
            return Err(GenError::Backend("empty multimodal prompt".into()));
        }
        let rows = ids.len();
        let logits = self.prefill_rows(&ids, &splices).map_err(gen_err)?;
        Ok(Some((logits, rows)))
    }
}
