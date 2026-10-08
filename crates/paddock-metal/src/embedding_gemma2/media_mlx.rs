//! Bundled MLX towers read their own tensors, never GGUF companion substitutes.
//! Convolution layout conversion runs once on Metal; matrices stay BF16 and
//! the audio embedder's affine-8 projection remains packed.
use super::*;
use paddock_models::safetensors::{ShardedSafetensors, StDtype};

fn key(name: &str) -> Result<String> {
    let direct = match name {
        "v.patch_embd.weight" => "vision_tower.patch_embedder.input_proj.weight",
        "v.position_embd.weight" => "vision_tower.patch_embedder.position_embedding_table",
        "mm.input_projection.weight" => "embed_vision.embedding_projection.weight",
        "mm.a.input_projection.weight" => "embed_audio.embedding_projection.weight",
        "a.input_projection.weight" => {
            "audio_tower.subsample_conv_projection.input_proj_linear.weight"
        }
        "a.pre_encode.out.weight" => "audio_tower.output_proj.weight",
        "a.pre_encode.out.bias" => "audio_tower.output_proj.bias",
        "a.conv1d.0.weight" => "audio_tower.subsample_conv_projection.layer0.conv.weight",
        "a.conv1d.1.weight" => "audio_tower.subsample_conv_projection.layer1.conv.weight",
        "a.conv1d.0.norm.weight" => "audio_tower.subsample_conv_projection.layer0.norm.weight",
        "a.conv1d.1.norm.weight" => "audio_tower.subsample_conv_projection.layer1.norm.weight",
        _ => "",
    };
    if !direct.is_empty() {
        return Ok(direct.into());
    }
    for (prefix, target, pairs) in [
        (
            "v.blk.",
            "vision_tower.encoder.layers.",
            &[
                ("ln1", "input_layernorm"),
                ("ln2", "pre_feedforward_layernorm"),
                ("attn_q", "self_attn.q_proj.linear"),
                ("attn_k", "self_attn.k_proj.linear"),
                ("attn_v", "self_attn.v_proj.linear"),
                ("attn_out", "self_attn.o_proj.linear"),
                ("attn_q_norm", "self_attn.q_norm"),
                ("attn_k_norm", "self_attn.k_norm"),
                ("attn_post_norm", "post_attention_layernorm"),
                ("ffn_post_norm", "post_feedforward_layernorm"),
                ("ffn_gate", "mlp.gate_proj.linear"),
                ("ffn_up", "mlp.up_proj.linear"),
                ("ffn_down", "mlp.down_proj.linear"),
            ][..],
        ),
        (
            "a.blk.",
            "audio_tower.layers.",
            &[
                ("ffn_norm", "feed_forward1.pre_layer_norm"),
                ("ffn_post_norm", "feed_forward1.post_layer_norm"),
                ("ffn_norm_1", "feed_forward2.pre_layer_norm"),
                ("ffn_post_norm_1", "feed_forward2.post_layer_norm"),
                ("ffn_up", "feed_forward1.ffw_layer_1"),
                ("ffn_down", "feed_forward1.ffw_layer_2"),
                ("ffn_up_1", "feed_forward2.ffw_layer_1"),
                ("ffn_down_1", "feed_forward2.ffw_layer_2"),
                ("attn_q", "self_attn.q_proj"),
                ("attn_k", "self_attn.k_proj"),
                ("attn_v", "self_attn.v_proj"),
                ("attn_out", "self_attn.post"),
                ("attn_k_rel", "self_attn.relative_k_proj"),
                ("attn_pre_norm", "norm_pre_attn"),
                ("attn_post_norm", "norm_post_attn"),
                ("conv_norm", "lconv1d.pre_layer_norm"),
                ("conv_pw1", "lconv1d.linear_start"),
                ("conv_pw2", "lconv1d.linear_end"),
                ("conv_dw", "lconv1d.depthwise_conv1d"),
                ("norm_conv", "lconv1d.conv_norm"),
                ("ln2", "norm_out"),
                ("per_dim_scale", "self_attn.per_dim_scale"),
            ][..],
        ),
    ] {
        if let Some(rest) = name.strip_prefix(prefix) {
            let (layer, rest) = rest
                .split_once('.')
                .ok_or_else(|| error("invalid media layer"))?;
            let (field, suffix) = rest
                .split_once('.')
                .ok_or_else(|| error("invalid media field"))?;
            let mapped = pairs
                .iter()
                .find(|(from, _)| *from == field)
                .ok_or_else(|| error(format!("unmapped media weight {name}")))?
                .1;
            let suffix = if field == "per_dim_scale" {
                String::new()
            } else if prefix == "a.blk."
                && suffix == "weight"
                && (field.starts_with("ffn_up")
                    || field.starts_with("ffn_down")
                    || matches!(
                        field,
                        "attn_q" | "attn_k" | "attn_v" | "attn_out" | "conv_pw1" | "conv_pw2"
                    ))
            {
                ".linear.weight".into()
            } else {
                format!(".{suffix}")
            };
            return Ok(format!("{target}{layer}.{mapped}{suffix}"));
        }
    }
    Err(error(format!("unmapped media tensor {name}")))
}

fn weight(
    d: &MetalDevice,
    m: &ShardedSafetensors,
    name: &str,
    shape: &[usize],
    ty: u32,
) -> Result<Weight> {
    let mapped = key(name)?;
    let (info, bytes) = m
        .bytes(&mapped)
        .ok_or_else(|| error(format!("missing {mapped}")))?;
    if info.dtype == StDtype::U32 && name == "mm.a.input_projection.weight" {
        let (k, n) = (shape[0], shape[1]);
        let mut parts = Vec::new();
        for (suffix, dtype, expected) in [
            ("weight", StDtype::U32, [n, k / 4]),
            ("scales", StDtype::Bf16, [n, k / 64]),
            ("biases", StDtype::Bf16, [n, k / 64]),
        ] {
            let key = format!("embed_audio.embedding_projection.{suffix}");
            let (info, data) = m
                .bytes(&key)
                .ok_or_else(|| error(format!("missing {key}")))?;
            if info.dtype != dtype
                || info.shape != expected
                || (dtype == StDtype::Bf16
                    && data
                        .chunks_exact(2)
                        .any(|v| u16::from_le_bytes([v[0], v[1]]) & 0x7f80 == 0x7f80))
            {
                return Err(error(format!("{key}: invalid affine-8/group-64 parameter")));
            }
            parts.push(data);
        }
        return Ok(Weight {
            buffer: d.upload_parts(&parts)?,
            ty: 0x108,
            k,
            n,
        });
    }
    let expected = match name {
        "v.patch_embd.weight" => vec![768, 768],
        "a.conv1d.0.weight" => vec![128, 3, 3, 1],
        "a.conv1d.1.weight" => vec![32, 3, 3, 128],
        n if n.ends_with("conv_dw.weight") => vec![1024, 5, 1],
        n if n.ends_with("_min") || n.ends_with("_max") => vec![],
        _ => shape.iter().rev().copied().collect(),
    };
    if info.shape != expected || info.dtype != StDtype::Bf16 {
        return Err(error(format!(
            "{mapped}: expected BF16 {expected:?}, got {:?} {:?}",
            info.dtype, info.shape
        )));
    }
    if bytes
        .chunks_exact(2)
        .any(|v| u16::from_le_bytes([v[0], v[1]]) & 0x7f80 == 0x7f80)
    {
        return Err(error(format!("{mapped}: nonfinite weights")));
    }
    let n: usize = shape.iter().product();
    let target = if ty != 0 || name == "v.patch_embd.weight" {
        30
    } else {
        0
    };
    let src = d.upload(bytes)?;
    let out = if target == 30 {
        src
    } else {
        let dst = d.alloc(n * 4)?;
        let c = d.begin()?;
        c.dispatch(
            "eg2_media_mlx_cast",
            &[&src, &dst],
            &[
                n as u32,
                if name == "a.conv1d.1.weight" { 128 } else { 1 },
                name.ends_with("per_dim_scale.weight") as u32,
            ],
            [n.div_ceil(256), 1, 1],
            256,
        );
        c.finish()?;
        dst
    };
    Ok(Weight {
        buffer: out,
        ty: target,
        k: shape[0],
        n: *shape.get(1).unwrap_or(&1),
    })
}

impl EmbeddingGemma2 {
    pub fn attach_mlx_media(
        &mut self,
        path: &Path,
        budget: usize,
        image: bool,
        audio: bool,
    ) -> Result<()> {
        use paddock_engine::encoder::embedding_gemma2::IMAGE_TOKEN_BUDGETS;
        if !self.mlx
            || self.vision.is_some()
            || self.audio.is_some()
            || !IMAGE_TOKEN_BUDGETS.contains(&budget)
        {
            return Err(error("invalid or duplicate MLX media attachment"));
        }
        let config = path.join("config.json");
        if std::fs::metadata(&config)
            .map_err(|e| error(e.to_string()))?
            .len()
            > 1 << 20
        {
            return Err(error("media config exceeds 1 MiB"));
        }
        let cfg: serde_json::Value =
            serde_json::from_slice(&std::fs::read(config).map_err(|e| error(e.to_string()))?)
                .map_err(|e| error(e.to_string()))?;
        let v = &cfg["vision_config"];
        let a = &cfg["audio_config"];
        if image
            && (v["hidden_size"] != 768
                || v["num_hidden_layers"] != 16
                || v["intermediate_size"] != 3072
                || v["num_attention_heads"] != 12
                || v["num_key_value_heads"] != 12
                || v["head_dim"] != 64
                || v["dtype"] != "bfloat16"
                || v["attention_bias"] != false
                || v["patch_size"] != 16
                || v["pooling_kernel_size"] != 3
                || v["position_embedding_size"] != 10240
                || v["standardize"] != false
                || v["use_clipped_linears"] != false
                || v["hidden_activation"] != "gelu_pytorch_tanh"
                || v["rms_norm_eps"].as_f64() != Some(1e-6)
                || v["rope_parameters"]["rope_theta"] != 100.0
                || v["rope_parameters"]["rope_type"] != "axial")
        {
            return Err(error("unsupported MLX image geometry"));
        }
        if audio
            && (a["hidden_size"] != 1024
                || a["num_hidden_layers"] != 12
                || a["num_attention_heads"] != 8
                || a["output_proj_dims"] != 1536
                || a["conv_kernel_size"] != 5
                || a["attention_context_left"] != 13
                || a["attention_context_right"] != 0
                || a["attention_chunk_size"] != 12
                || a["attention_logit_cap"] != 50.0
                || a["residual_weight"] != 0.5
                || a["dtype"] != "bfloat16"
                || a["gradient_clipping"] != 1e10
                || a["rms_norm_eps"].as_f64() != Some(1e-6)
                || a["subsampling_conv_channels"] != serde_json::json!([128, 32])
                || a["hidden_act"] != "silu"
                || a["use_clipped_linears"] != true)
        {
            return Err(error("unsupported MLX audio geometry"));
        }
        let map = ShardedSafetensors::open_dir(path).map_err(|e| error(e.to_string()))?;
        let before = self.device.allocated_bytes();
        let load = |name: &str, shape: &[usize], ty| weight(&self.device, &map, name, shape, ty);
        let vision = image
            .then(|| vision::Vision::from_weights(&self.device, budget, true, load))
            .transpose()?;
        let audio = audio
            .then(|| audio::Audio::from_weights(&self.device, true, load))
            .transpose()?;
        self.vision = vision;
        self.audio = audio;
        self.weight_bytes += self.device.allocated_bytes() - before;
        Ok(())
    }
}
