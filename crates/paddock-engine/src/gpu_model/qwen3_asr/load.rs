//! Qwen3-ASR decoder loader - stock Qwen3 dense metadata (the same fields
//! qwen3.rs reads for the embeddings encoder), granite's QuantW/TokEmbd
//! resident classes, tied-head fallback. The routing layer must only send
//! arch `qwen3` here when an audio mmproj rides along (the bare arch is the
//! embeddings encoder's).

use std::sync::Arc;

use paddock_kernels::reference::ops::YarnRope;
use paddock_models::gguf::Value;
use paddock_models::mapped::MappedGguf;

use crate::gpu::{GpuExecutor, KvDtype};
use crate::gpu_model::gpt_oss::GpuModelError;

use super::{AsrLayer, GpuQwen3Asr, Hparams, TokEmbd};

impl GpuQwen3Asr {
    pub fn load(
        exec: Arc<GpuExecutor>,
        map: &MappedGguf,
        max_ctx: usize,
    ) -> Result<Self, GpuModelError> {
        exec.vram_load_gate(map.total_len(), "qwen3-asr")
            .map_err(GpuModelError::WontFit)?;
        // Single-stream engine: cudarc's cross-stream event tracking is pure
        // overhead and blocks CUDA-graph capture. Must precede all allocs.
        exec.disable_event_tracking();

        let u = |k: &str| {
            map.gguf()
                .arch_field(k)
                .and_then(Value::as_u64)
                .ok_or_else(|| GpuModelError::MissingMeta(k.to_owned()))
        };
        let f = |k: &str| map.gguf().arch_field(k).and_then(Value::as_f32);

        let n_layer = u("block_count")? as usize;
        let n_embd = u("embedding_length")? as usize;
        let n_heads = u("attention.head_count")? as usize;
        let n_kv_heads = u("attention.head_count_kv")? as usize;
        // Qwen3 stamps key_length (head_dim 128 ≠ embd/heads on this family).
        let head_dim = u("attention.key_length")? as usize;
        let n_ff = u("feed_forward_length")? as usize;
        let eps = f("attention.layer_norm_rms_epsilon").unwrap_or(1e-6);
        let ctx_train = u("context_length")? as usize;
        if n_kv_heads == 0 || !n_heads.is_multiple_of(n_kv_heads) {
            return Err(GpuModelError::Unsupported(format!(
                "qwen3-asr: head_count {n_heads} not a multiple of head_count_kv {n_kv_heads}"
            )));
        }
        // Full rotary, plain NEOX rope; refuse stamped-but-unhandled scaling
        // (no-silent-failures - same stance as granite).
        //
        // The conversion stamps `rope.dimension_sections` (the qwen3vl class
        // carries M-RoPE) - but both upstream implementations feed every
        // axis the same sequential position for text and audio segments
        // (vLLM qwen3_asr `get_mrope_input_positions`), and M-RoPE with
        // equal axes is exactly plain 1D rope. Validate the sections cover
        // the full rotary width so a future genuinely-multiaxis checkpoint
        // cannot slip through this degeneracy argument silently.
        let n_rot = map
            .gguf()
            .arch_field("rope.dimension_count")
            .and_then(Value::as_u64)
            .map(|v| v as usize)
            .unwrap_or(head_dim);
        let mut sections = [(n_rot / 2) as u32, 0, 0, 0];
        if let Some(Value::Array(secs)) = map.gguf().arch_field("rope.dimension_sections") {
            let sum: u64 = secs.iter().filter_map(Value::as_u64).sum();
            if sum as usize * 2 != n_rot {
                return Err(GpuModelError::Unsupported(format!(
                    "qwen3-asr: rope.dimension_sections sum {sum}*2 != rotary width {n_rot} - \
                     not the equal-axis M-RoPE geometry this family serves as plain rope"
                )));
            }
            for (slot, v) in sections.iter_mut().zip(secs) {
                *slot = v.as_u64().unwrap_or(0) as u32;
            }
        }
        if map.gguf().arch_field("rope.scaling.type").is_some()
            || f("rope.scaling.factor").is_some_and(|v| v != 1.0)
        {
            return Err(GpuModelError::Unsupported(
                "qwen3-asr: rope scaling is stamped but unhandled - refusing to ignore it".into(),
            ));
        }
        if max_ctx > ctx_train {
            return Err(GpuModelError::Unsupported(format!(
                "qwen3-asr: requested context {max_ctx} exceeds the trained {ctx_train}"
            )));
        }
        let rope_base = f("rope.freq_base").unwrap_or(1e6);
        let rope =
            YarnRope::new(n_rot, rope_base, 1.0, ctx_train, 0.0, 1.0, 32.0, 1.0).kernel_params();

        let te_ty = map
            .tensor_info("token_embd.weight")
            .map(|t| t.ggml_type)
            .ok_or_else(|| GpuModelError::MissingMeta("token_embd.weight".into()))?;
        let (tok_embd, n_vocab) = if crate::gpu::kq_params(te_ty).is_some() {
            let t = exec.repack_kquant(map, "token_embd.weight")?;
            let vocab = t.dims[1];
            (TokEmbd::Kq(t), vocab)
        } else {
            let t = exec.upload_raw(map, "token_embd.weight")?;
            if t.ty != paddock_models::ggml_type::GgmlType::Q8_0 {
                return Err(GpuModelError::Unsupported(format!(
                    "qwen3-asr: token_embd quant {:?} has no resident gather path - quantize \
                     the conversion to Q8_0 (the served class) or a k-quant",
                    t.ty
                )));
            }
            let vocab = t.dims[1];
            (TokEmbd::Q8(t), vocab)
        };

        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let dt = |name: &str| exec.upload(map, &format!("blk.{i}.{name}"));
            let qt = |name: &str| exec.load_quantw(map, &format!("blk.{i}.{name}"));
            layers.push(AsrLayer {
                attn_norm: dt("attn_norm.weight")?,
                wq: qt("attn_q.weight")?,
                wk: qt("attn_k.weight")?,
                wv: qt("attn_v.weight")?,
                q_norm: dt("attn_q_norm.weight")?,
                k_norm: dt("attn_k_norm.weight")?,
                wo: qt("attn_output.weight")?,
                ffn_norm: dt("ffn_norm.weight")?,
                gate: qt("ffn_gate.weight")?,
                up: qt("ffn_up.weight")?,
                down: qt("ffn_down.weight")?,
            });
        }
        let output_norm = exec.upload(map, "output_norm.weight")?;
        let lm_head = if map.tensor_info("output.weight").is_some() {
            exec.load_quantw(map, "output.weight")?
        } else {
            exec.load_quantw(map, "token_embd.weight")?
        };

        let layer_bytes: u64 = layers
            .iter()
            .map(|l| {
                l.wq.bytes()
                    + l.wk.bytes()
                    + l.wv.bytes()
                    + l.wo.bytes()
                    + l.gate.bytes()
                    + l.up.bytes()
                    + l.down.bytes()
            })
            .sum();
        let weights_bytes = layer_bytes + tok_embd.resident_bytes() as u64 + lm_head.bytes();
        tracing::info!(
            layers = n_layer,
            vocab = n_vocab,
            weight_gib = weights_bytes / (1 << 30),
            "qwen3-asr decoder loaded (Qwen3 dense, tied head)"
        );

        Ok(Self {
            exec,
            hp: Hparams {
                n_layer,
                n_embd,
                n_heads,
                n_kv_heads,
                head_dim,
                n_ff,
                n_vocab,
                eps,
                rope,
                n_rot,
                sections,
            },
            max_ctx,
            // f16, and MEASURED to be the right default here. fp8-e4m3 was
            // tried on the same day whisper adopted it and is a REGRESSION
            // for this family - a couple of percent down at c1, c8 and c32
            // alike. Two reasons, and they are the mirror image of whisper's:
            //   - this decoder's KV is an ordinary causal cache over a SHORT
            //     transcript, not whisper's 1500-frame cross planes, so KV
            //     bytes are nowhere near the binding cost (whisper's cross
            //     attention was 27% of all GPU time; here it is noise), and
            //   - the hd128 fp8 prefill tile has no GQA ratio-2 arm, so this
            //     geometry drops off the tensor-core tile at fp8 and pays
            //     dequant for bytes it did not need to save.
            // The switch still works (`--kv-cache-dtype fp8_e4m3`) for anyone
            // who wants the VRAM back; it just must not be the default. Revisit
            // if the ratio-2 fp8 prefill arm ever lands.
            // KV stays f16 - the exception to the KV8 default (2026-10-04):
            // the fp8 prefill tile has no group-2 arm (16q/8kv; batch.rs
            // wmma_pf), and measured at fp8 the family is exact (11/11) but
            // 2-6% slower (measured 2026-10-04), so f16 is the faster class.
            kv_dtype: KvDtype::Fp16,
            tok_embd,
            layers,
            output_norm,
            lm_head,
            tower: None,
            decode: None,
            scratch: None,
            prefill: None,
            batch: None,
            chunked: Vec::new(),
            weights_bytes,
        })
    }
}
