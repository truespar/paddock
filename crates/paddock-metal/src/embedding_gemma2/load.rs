use super::*;
use paddock_models::{
    mapped::MappedGguf,
    safetensors::{ShardedSafetensors, StDtype},
};

enum Source {
    Mlx(ShardedSafetensors),
    Gguf(MappedGguf),
}
impl Source {
    fn get(
        &self,
        d: &MetalDevice,
        hf: &str,
        gguf: &str,
        k: usize,
        n: usize,
        vector: bool,
    ) -> Result<Weight> {
        match self {
            Self::Gguf(map) => {
                let shape = [k, n];
                Weight::load(d, map, gguf, if vector { &shape[..1] } else { &shape })
            }
            Self::Mlx(map) => {
                let name = format!("language_model.{hf}");
                let (info, data) = map
                    .bytes(&name)
                    .ok_or_else(|| error(format!("missing {name}")))?;
                if info.dtype == StDtype::U32 && !vector {
                    let base = name
                        .strip_suffix(".weight")
                        .ok_or_else(|| error("invalid packed tensor"))?;
                    let bits = if info.shape == [n, k / 4] {
                        8
                    } else if info.shape == [n, k / 8] {
                        4
                    } else {
                        return Err(error(format!("{name}: invalid affine shape")));
                    };
                    let mut parts = vec![data];
                    for suffix in ["scales", "biases"] {
                        let key = format!("{base}.{suffix}");
                        let (t, b) = map
                            .bytes(&key)
                            .ok_or_else(|| error(format!("missing {key}")))?;
                        if t.dtype != StDtype::Bf16 || t.shape != [n, k / 64] {
                            return Err(error(format!("{key}: expected BF16 group-64 parameters")));
                        }
                        if b.chunks_exact(2)
                            .any(|v| u16::from_le_bytes([v[0], v[1]]) & 0x7f80 == 0x7f80)
                        {
                            return Err(error(format!("{key}: nonfinite scale/bias")));
                        }
                        parts.push(b);
                    }
                    return Ok(Weight {
                        buffer: d.upload_parts(&parts)?,
                        ty: if bits == 8 { 0x108 } else { 0x100 },
                        k,
                        n,
                    });
                }
                let shape = if vector { vec![k] } else { vec![n, k] };
                if info.shape != shape || !matches!(info.dtype, StDtype::Bf16 | StDtype::F32) {
                    return Err(error(format!(
                        "{name}: unsupported dtype/shape {:?} {:?}",
                        info.dtype, info.shape
                    )));
                }
                let src = d.upload(data)?;
                let dst = d.alloc(k * n * 4)?;
                let bad = d.upload(&0u32.to_le_bytes())?;
                let cmd = d.begin()?;
                cmd.dispatch(
                    "vis_cast",
                    &[&src, &dst, &bad],
                    &[
                        (k * n) as u32,
                        if info.dtype == StDtype::Bf16 { 30 } else { 0 },
                        0,
                    ],
                    [(k * n).div_ceil(256), 1, 1],
                    256,
                );
                cmd.finish()?;
                if unsafe { bad.read_u32(1)[0] } != 0 {
                    return Err(error(format!("{name}: nonfinite weights")));
                }
                Ok(Weight {
                    // Raw matrices stay in their checkpoint dtype. Widening
                    // all BF16 weights would double resident model memory.
                    buffer: if vector { dst } else { src },
                    ty: if !vector && info.dtype == StDtype::Bf16 {
                        30
                    } else {
                        0
                    },
                    k,
                    n,
                })
            }
        }
    }
}

impl EmbeddingGemma2 {
    pub fn load(path: &Path, context: usize, budget: Option<u64>) -> Result<Self> {
        if context == 0 || context > CONTEXT {
            return Err(error(
                "EmbeddingGemma 2 supports 1..8192 input tokens (the rotary ceiling is not its trained context)",
            ));
        }
        let source = if path.is_dir() {
            let config_path = path.join("config.json");
            if std::fs::metadata(&config_path)
                .map_err(|e| error(e.to_string()))?
                .len()
                > 1 << 20
            {
                return Err(error("EmbeddingGemma 2 config exceeds 1 MiB"));
            }
            let cfg: serde_json::Value = serde_json::from_slice(
                &std::fs::read(config_path).map_err(|e| error(e.to_string()))?,
            )
            .map_err(|e| error(e.to_string()))?;
            let t = &cfg["text_config"];
            if cfg["model_type"] != "embedding_gemma2"
                || t["hidden_size"] != WIDTH
                || t["num_hidden_layers"] != LAYERS
                || t["intermediate_size"] != FF
                || t["vocab_size"] != VOCAB
                || t["embedding_dim"] != DIM
                || t["num_attention_heads"] != 4
                || t["num_key_value_heads"] != 2
                || t["head_dim"] != 256
                || t["hidden_size_per_layer_input"] != WIDTH
                || t["sliding_window"] != 512
                || t["hidden_activation"] != "gelu_pytorch_tanh"
                || t["attention_bias"] != false
                || t["rms_norm_eps"].as_f64() != Some(1e-6)
                || cfg["dtype"] != "bfloat16"
                || t["dtype"] != "bfloat16"
                || t["attention_dropout"].as_f64() != Some(0.)
                || t["layer_types"].as_array().map(Vec::len) != Some(LAYERS)
                || t["per_layer_config"].as_object().map(serde_json::Map::len) != Some(4)
            {
                return Err(error("unsupported EmbeddingGemma 2 text geometry"));
            }
            for i in 0..LAYERS {
                let slide = i % 6 != 5;
                if t["layer_types"][i]
                    != if slide {
                        "sliding_attention"
                    } else {
                        "full_attention"
                    }
                {
                    return Err(error("unsupported attention pattern"));
                }
                if !slide
                    && (t["per_layer_config"][format!("{i:02}")]["head_dim"] != 512
                        || t["per_layer_config"][format!("{i:02}")]["num_key_value_heads"] != 1
                        || t["per_layer_config"][format!("{i:02}")]
                            .as_object()
                            .map(serde_json::Map::len)
                            != Some(2))
                {
                    return Err(error("unsupported global attention geometry"));
                }
            }
            for (kind, theta) in [
                ("full_attention", 1_000_000.),
                ("sliding_attention", 10_000.),
            ] {
                if t["rope_parameters"][kind]["rope_theta"].as_f64() != Some(theta)
                    || t["rope_parameters"][kind]["rope_type"] != "default"
                    || t["rope_parameters"][kind]
                        .as_object()
                        .map(serde_json::Map::len)
                        != Some(2)
                {
                    return Err(error("unsupported EmbeddingGemma 2 rotary parameters"));
                }
            }
            if !cfg["quantization"].is_null()
                && (cfg["quantization"]["mode"] != "affine"
                    || cfg["quantization"]["group_size"] != 64
                    || !matches!(cfg["quantization"]["bits"].as_u64(), Some(4 | 8)))
            {
                return Err(error(
                    "EmbeddingGemma 2 MLX supports BF16 or affine 4/8-bit group-64, not FP16",
                ));
            }
            Source::Mlx(ShardedSafetensors::open_dir(path).map_err(|e| error(e.to_string()))?)
        } else {
            let map = MappedGguf::open(path).map_err(|e| error(e.to_string()))?;
            let g = map.gguf();
            let u = |key| {
                g.arch_field(key)
                    .and_then(paddock_models::gguf::Value::as_u64)
            };
            if g.architecture() != Some("gemma-embedding2")
                || u("block_count") != Some(24)
                || u("embedding_length") != Some(512)
                || u("feed_forward_length") != Some(2048)
                || u("attention.head_count") != Some(4)
                || u("embedding_length_out") != Some(768)
                || u("attention.sliding_window") != Some(1024)
            {
                return Err(error("unsupported EmbeddingGemma 2 GGUF geometry"));
            }
            Source::Gguf(map)
        };
        let mlx = matches!(source, Source::Mlx(_));
        let device = MetalDevice::new(budget)?;
        let w = |hf, gg, k, n| source.get(&device, hf, gg, k, n, false);
        let v = |hf, gg, k| source.get(&device, hf, gg, k, 1, true);
        let embedding = w("embed_tokens.weight", "token_embd.weight", WIDTH, VOCAB)?;
        let ple = w(
            "ple.per_layer_model_projection.weight",
            "per_layer_model_proj.weight",
            WIDTH,
            WIDTH * LAYERS,
        )?;
        let ple_norm = v(
            "ple.per_layer_projection_norm.weight",
            "per_layer_proj_norm.weight",
            WIDTH,
        )?;
        let norm = v("norm.weight", "output_norm.weight", WIDTH)?;
        let output = w("embedding_projection.weight", "output.weight", WIDTH, DIM)?;
        let mut layers = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let hd = if i % 6 == 5 { 512 } else { 256 };
            let w = |hf, gg, k, n| {
                source.get(
                    &device,
                    &format!("layers.{i}.{hf}.weight"),
                    &format!("blk.{i}.{gg}.weight"),
                    k,
                    n,
                    false,
                )
            };
            let v = |hf, gg, k| {
                source.get(
                    &device,
                    &format!("layers.{i}.{hf}.weight"),
                    &format!("blk.{i}.{gg}.weight"),
                    k,
                    1,
                    true,
                )
            };
            layers.push(Layer {
                pre: v("input_layernorm", "attn_norm", WIDTH)?,
                post: v("post_attention_layernorm", "post_attention_norm", WIDTH)?,
                ff_pre: v("pre_feedforward_layernorm", "ffn_norm", WIDTH)?,
                ff_post: v("post_feedforward_layernorm", "post_ffw_norm", WIDTH)?,
                q: w("self_attn.q_proj", "attn_q", WIDTH, 4 * hd)?,
                k: w("self_attn.k_proj", "attn_k", WIDTH, 512)?,
                v: w("self_attn.v_proj", "attn_v", WIDTH, 512)?,
                o: w("self_attn.o_proj", "attn_output", 4 * hd, WIDTH)?,
                qn: v("self_attn.q_norm", "attn_q_norm", hd)?,
                kn: v("self_attn.k_norm", "attn_k_norm", hd)?,
                gate: w("mlp.gate_proj", "ffn_gate", WIDTH, FF)?,
                up: w("mlp.up_proj", "ffn_up", WIDTH, FF)?,
                down: w("mlp.down_proj", "ffn_down", FF, WIDTH)?,
                ple_gate: w("ple_block.per_layer_input_gate", "inp_gate", WIDTH, WIDTH)?,
                ple_out: w("ple_block.per_layer_projection", "proj", WIDTH, WIDTH)?,
                ple_norm: v("ple_block.post_per_layer_input_norm", "post_norm", WIDTH)?,
                scalar: source.get(
                    &device,
                    &format!("layers.{i}.layer_scalar"),
                    &format!("blk.{i}.layer_output_scale.weight"),
                    1,
                    1,
                    true,
                )?,
            });
        }
        let weight_bytes = device.allocated_bytes();
        let capacity = context.max(512);
        tracing::info!(
            context,
            capacity,
            mlx,
            weight_bytes,
            allocated = device.allocated_bytes(),
            "native EmbeddingGemma 2 text backbone loaded"
        );
        Ok(Self {
            device,
            identity: std::rc::Rc::new(()),
            embedding,
            ple,
            ple_norm,
            norm,
            output,
            layers,
            scratch: None,
            context,
            capacity,
            mlx,
            weight_bytes,
            vision: None,
            audio: None,
        })
    }
}
