//! Loading the text backbone from the GGUF: geometry checked against what the
//! graph in `forward.rs` implements (any drift is refused, not approximated),
//! Q8_0 projections repacked for the mmq tile (q|k|v and gate|up each fused
//! into one plane),
//! the PLE projection kept as its bf16 bytes, norms widened to f32.

use std::sync::Arc;

use paddock_models::ggml_type::GgmlType;
use paddock_models::gguf::Value;
use paddock_models::mapped::MappedGguf;

use super::*;

fn bad(msg: impl Into<String>) -> GpuModelError {
    GpuModelError::MissingMeta(msg.into())
}

impl GpuEmbeddingGemma2 {
    /// Load the text encoder. `context` caps a sequence (1..=8192, the
    /// trained budget); the pass row budget is `max(context, 512)`.
    pub fn load(
        exec: Arc<GpuExecutor>,
        map: &MappedGguf,
        context: usize,
    ) -> Result<Self, GpuModelError> {
        if context == 0 || context > CONTEXT {
            return Err(bad(format!(
                "EmbeddingGemma 2 serves 1..={CONTEXT} input tokens (its trained budget; the \
                 rope tables reach further, the training did not)"
            )));
        }
        if !exec.has_embedding_gemma2() || !exec.has_q8_0_gemm_mmq() {
            return Err(bad(
                "this kernel pack predates the EmbeddingGemma 2 lane (slots 806-812) - update it",
            ));
        }
        let g = map.gguf();
        if g.architecture() != Some("gemma-embedding2") {
            return Err(bad("not an EmbeddingGemma 2 GGUF"));
        }
        let u = |k: &str| g.arch_field(k).and_then(Value::as_u64);
        let f = |k: &str| g.arch_field(k).and_then(Value::as_f32);
        let list = |k: &str| match g.arch_field(k) {
            Some(Value::Array(v)) => Some(v.clone()),
            _ => None,
        };
        let kv_heads: Option<Vec<u64>> =
            list("attention.head_count_kv").and_then(|v| v.iter().map(Value::as_u64).collect());
        let pattern: Option<Vec<bool>> = list("attention.sliding_window_pattern").and_then(|v| {
            v.iter()
                .map(|x| match x {
                    Value::Bool(b) => Some(*b),
                    _ => None,
                })
                .collect()
        });
        let geometry_ok = u("block_count") == Some(LAYERS as u64)
            && u("embedding_length") == Some(WIDTH as u64)
            && u("feed_forward_length") == Some(FF as u64)
            && u("attention.head_count") == Some(4)
            && u("embedding_length_out") == Some(DIM as u64)
            && u("embedding_length_per_layer_input") == Some(WIDTH as u64)
            && u("attention.key_length") == Some(512)
            && u("attention.value_length") == Some(512)
            && u("attention.key_length_swa") == Some(256)
            && u("attention.value_length_swa") == Some(256)
            && u("rope.dimension_count") == Some(512)
            && u("rope.dimension_count_swa") == Some(256)
            && u("attention.sliding_window") == Some(1024)
            && f("rope.freq_base") == Some(1_000_000.0)
            && f("rope.freq_base_swa") == Some(10_000.0)
            && matches!(g.arch_field("attention.causal"), Some(Value::Bool(false)))
            && u("pooling_type") == Some(1)
            && kv_heads.as_deref().is_some_and(|v| {
                v.len() == LAYERS
                    && v.iter()
                        .enumerate()
                        .all(|(i, &n)| n == if i % 6 == 5 { 1 } else { 2 })
            })
            && pattern.as_deref().is_some_and(|v| {
                v.len() == LAYERS && v.iter().enumerate().all(|(i, &s)| s == (i % 6 != 5))
            });
        if !geometry_ok {
            return Err(bad("unsupported EmbeddingGemma 2 GGUF geometry"));
        }
        let eps = f("attention.layer_norm_rms_epsilon")
            .ok_or_else(|| bad("missing attention.layer_norm_rms_epsilon"))?;
        let before = exec.process_mem_used().unwrap_or(0);

        let ty = |name: &str| map.tensor_info(name).map(|t| t.ggml_type);
        let norm = |name: &str, n: usize| -> Result<CudaSlice<f32>, GpuModelError> {
            let t = exec.upload(map, name)?;
            if t.element_count() != n {
                return Err(bad(format!("{name}: expected {n} values")));
            }
            Ok(t.buf)
        };
        let q8 = |name: &str, k: usize, n: usize| -> Result<RepackedQ8, GpuModelError> {
            if ty(name) != Some(GgmlType::Q8_0) {
                return Err(bad(format!("{name}: the CUDA lane serves the Q8_0 GGUF")));
            }
            let w = exec.repack_q8(map, name)?;
            if w.dims != [k, n] {
                return Err(bad(format!(
                    "{name}: expected [{k}, {n}], got {:?}",
                    w.dims
                )));
            }
            Ok(w)
        };

        let embd = exec.upload_raw(map, "token_embd.weight")?;
        if embd.ty != GgmlType::Q8_0 || embd.dims != [WIDTH, VOCAB] {
            return Err(bad("token_embd.weight: expected Q8_0 [512, 262144]"));
        }
        let ple_proj = exec.upload_raw(map, "per_layer_model_proj.weight")?;
        if ple_proj.ty != GgmlType::Bf16 || ple_proj.dims != [WIDTH, WIDTH * LAYERS] {
            return Err(bad(
                "per_layer_model_proj.weight: expected BF16 [512, 12288]",
            ));
        }
        let ple_norm = norm("per_layer_proj_norm.weight", WIDTH)?;
        let out_norm = norm("output_norm.weight", WIDTH)?;
        let output = q8("output.weight", WIDTH, DIM)?;

        let mut layers = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let hd = if i % 6 == 5 { 512 } else { 256 };
            let n = |s: &str| format!("blk.{i}.{s}.weight");
            let q = q8(&n("attn_q"), WIDTH, 4 * hd)?;
            let k = q8(&n("attn_k"), WIDTH, 512)?;
            let v = q8(&n("attn_v"), WIDTH, 512)?;
            let qkv = exec.concat_q8(&[&q, &k, &v])?;
            drop((q, k, v));
            let scale = {
                let (info, bytes) = map
                    .tensor_bytes(&n("layer_output_scale"))
                    .map_err(|e| bad(e.to_string()))?;
                if info.ggml_type != GgmlType::F32 || bytes.len() != 4 {
                    return Err(bad("layer_output_scale: expected one f32"));
                }
                f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
            };
            layers.push(Layer {
                hd,
                qkv,
                o: q8(&n("attn_output"), 4 * hd, WIDTH)?,
                gate_up: {
                    let (g, u) = (q8(&n("ffn_gate"), WIDTH, FF)?, q8(&n("ffn_up"), WIDTH, FF)?);
                    exec.concat_q8(&[&g, &u])?
                },
                down: q8(&n("ffn_down"), FF, WIDTH)?,
                inp_gate: q8(&n("inp_gate"), WIDTH, WIDTH)?,
                proj: q8(&n("proj"), WIDTH, WIDTH)?,
                attn_norm: norm(&n("attn_norm"), WIDTH)?,
                post_attn: norm(&n("post_attention_norm"), WIDTH)?,
                ffn_norm: norm(&n("ffn_norm"), WIDTH)?,
                post_ffw: norm(&n("post_ffw_norm"), WIDTH)?,
                post_norm: norm(&n("post_norm"), WIDTH)?,
                q_norm: norm(&n("attn_q_norm"), hd)?,
                k_norm: norm(&n("attn_k_norm"), hd)?,
                scale,
            });
        }
        exec.synchronize()?;
        exec.release_staging();
        let weights_bytes = exec.process_mem_used().unwrap_or(0).saturating_sub(before);
        let capacity = context.max(512);
        tracing::info!(
            context,
            capacity,
            weights_bytes,
            "EmbeddingGemma 2 text backbone loaded (CUDA, Q8_0)"
        );
        Ok(Self {
            exec,
            embd,
            ple_proj,
            ple_norm,
            out_norm,
            output,
            layers,
            eps,
            window: 512,
            // the reference's host powf(freq_base, -2 / n_dims), per geometry
            theta_scale: [
                10_000f32.powf(-2.0 / 256.0),
                1_000_000f32.powf(-2.0 / 512.0),
            ],
            context,
            capacity,
            scratch: None,
            pool: [None, None],
            flip: 0,
            weights_bytes,
            images: None,
            audio: None,
        })
    }
}
