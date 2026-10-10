//! Validated metadata for elected MLX Qwen dense 27B and tied 0.8B/4B checkpoints.
//! This is a storage lane, not a dependency on the MLX inference runtime.
use crate::safetensors::StError;
use serde_json::Value;
use std::path::Path;

#[path = "mlx_multimodal.rs"]
mod multimodal;
pub use multimodal::{MultimodalConfig, MultimodalFamily};
mod diffusion;
pub use diffusion::DiffusionConfig;
#[path = "mlx_llama.rs"]
mod llama;
pub use llama::MiniCpmConfig;

#[derive(Clone, Debug)]
pub struct QwenConfig {
    pub width: usize,
    pub ff: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub value_heads: usize,
    pub tied: bool,
    pub context: usize,
    pub eps: f32,
    pub rope: f32,
}

impl QwenConfig {
    pub fn read(dir: &Path) -> Result<Self, StError> {
        let path = dir.join("config.json");
        if std::fs::metadata(&path)?.len() > 1 << 20 {
            return Err(StError::Header("MLX config exceeds 1 MiB".into()));
        }
        let v: Value = serde_json::from_slice(&std::fs::read(path)?)
            .map_err(|e| StError::Header(e.to_string()))?;
        Self::parse(&v)
    }

    pub fn parse(v: &Value) -> Result<Self, StError> {
        let bad = |s: &str| StError::Header(format!("MLX Qwen: {s}"));
        if v["model_type"] != "qwen3_5" {
            return Err(bad("requires dense qwen3_5"));
        }
        let quant = v
            .get("quantization")
            .or_else(|| v.get("quantization_config"))
            .ok_or_else(|| bad("missing quantization metadata"))?;
        for q in std::iter::once(quant).chain(v.get("quantization_config")) {
            if q["bits"] != 4
                || q["group_size"] != 64
                || q["mode"] != "affine"
                || q.as_object().is_none_or(|o| o.len() != 3)
            {
                return Err(bad("requires uniform affine 4-bit/group-64 quantization"));
            }
        }
        let t = &v["text_config"];
        let config = Self::parse_text(t)?;
        if v["tie_word_embeddings"].as_bool() != Some(config.tied) {
            return Err(bad("conflicting tied embedding metadata"));
        }
        if config.tied {
            config.validate_vision(&v["vision_config"])?;
        }
        Ok(config)
    }

    pub(crate) fn parse_text(t: &Value) -> Result<Self, StError> {
        let bad = |s: &str| StError::Header(format!("MLX Qwen: {s}"));
        let (width, ff, layers, heads, kv_heads, value_heads, tied) =
            match t["hidden_size"].as_u64() {
                Some(1024) => (1024, 3584, 24, 8, 2, 16, true),
                Some(2560) => (2560, 9216, 32, 16, 4, 32, true),
                Some(5120) => (5120, 17408, 64, 24, 4, 48, false),
                _ => return Err(bad("unsupported hidden_size")),
            };
        for (key, expected) in [
            ("hidden_size", width),
            ("intermediate_size", ff),
            ("num_hidden_layers", layers),
            ("num_attention_heads", heads),
            ("num_key_value_heads", kv_heads),
            ("head_dim", 256),
            ("vocab_size", 248320),
            ("full_attention_interval", 4),
            ("linear_conv_kernel_dim", 4),
            ("linear_key_head_dim", 128),
            ("linear_value_head_dim", 128),
            ("linear_num_key_heads", 16),
            ("linear_num_value_heads", value_heads),
        ] {
            if t[key].as_u64() != Some(expected) {
                return Err(bad(&format!("unsupported {key}")));
            }
        }
        if t["hidden_act"] != "silu"
            || t["dtype"] != "bfloat16"
            || t["attention_bias"] != false
            || t["tie_word_embeddings"].as_bool() != Some(tied)
            || t["attn_output_gate"] != true
            || (t["output_gate_type"] != "swish" && !(tied && t.get("output_gate_type").is_none()))
            || t["mamba_ssm_dtype"] != "float32"
        {
            return Err(bad("unsupported arithmetic/attention configuration"));
        }
        if tied && t["mtp_num_hidden_layers"] != 0 {
            return Err(bad("small tied checkpoints require zero MTP layers"));
        }
        let ordering = t["layer_types"]
            .as_array()
            .ok_or_else(|| bad("missing layer_types"))?;
        if ordering.len() != layers as usize
            || ordering.iter().enumerate().any(|(i, l)| {
                l != if (i + 1) % 4 == 0 {
                    "full_attention"
                } else {
                    "linear_attention"
                }
            })
        {
            return Err(bad("unsupported layer ordering"));
        }
        let r = &t["rope_parameters"];
        if r["rope_type"] != "default"
            || r["mrope_interleaved"] != true
            || r["partial_rotary_factor"] != 0.25
            || t["partial_rotary_factor"] != 0.25
            || r["mrope_section"] != serde_json::json!([11, 11, 10])
        {
            return Err(bad("unsupported rotary configuration"));
        }
        let positive = |v: &Value, name: &str| -> Result<f32, StError> {
            v.as_f64()
                .map(|v| v as f32)
                .filter(|v| v.is_finite() && *v > 0.)
                .ok_or_else(|| bad(&format!("invalid {name}")))
        };
        let context = t["max_position_embeddings"]
            .as_u64()
            .filter(|&n| n > 0 && n <= u32::MAX as u64)
            .ok_or_else(|| bad("invalid trained context"))? as usize;
        Ok(Self {
            width: width as usize,
            ff: ff as usize,
            layers: layers as usize,
            heads: heads as usize,
            kv_heads: kv_heads as usize,
            value_heads: value_heads as usize,
            tied,
            context,
            eps: positive(&t["rms_norm_eps"], "rms_norm_eps")?,
            rope: positive(&r["rope_theta"], "rope_theta")?,
        })
    }

    fn validate_vision(&self, v: &Value) -> Result<(), StError> {
        let (width, ff, layers, heads) = match self.width {
            1024 => (768, 3072, 12, 12),
            2560 => (1024, 4096, 24, 16),
            _ => return Err(StError::Header("unsupported MLX OCR tower".into())),
        };
        for (key, expected) in [
            ("hidden_size", width),
            ("intermediate_size", ff),
            ("depth", layers),
            ("num_heads", heads),
            ("out_hidden_size", self.width as u64),
            ("patch_size", 16),
            ("temporal_patch_size", 2),
            ("spatial_merge_size", 2),
            ("num_position_embeddings", 2304),
            ("in_channels", 3),
        ] {
            if v[key].as_u64() != Some(expected) {
                return Err(StError::Header(format!("MLX OCR tower: unsupported {key}")));
            }
        }
        if v["dtype"] != "bfloat16"
            || v["hidden_act"] != "gelu_pytorch_tanh"
            || v["deepstack_visual_indexes"] != serde_json::json!([])
        {
            return Err(StError::Header(
                "unsupported MLX OCR tower arithmetic/DeepStack".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    #[test]
    fn small_tied_mlx_requires_matching_tower_and_text_geometry() {
        for (width, ff, layers, heads, kv, values, vw, vf, depth, vh) in [
            (1024, 3584, 24, 8, 2, 16, 768, 3072, 12, 12),
            (2560, 9216, 32, 16, 4, 32, 1024, 4096, 24, 16),
        ] {
            let mut v = config();
            v["tie_word_embeddings"] = true.into();
            let t = &mut v["text_config"];
            for (key, value) in [
                ("hidden_size", width),
                ("intermediate_size", ff),
                ("num_hidden_layers", layers),
                ("num_attention_heads", heads),
                ("num_key_value_heads", kv),
                ("linear_num_value_heads", values),
                ("mtp_num_hidden_layers", 0),
            ] {
                t[key] = value.into();
            }
            t["tie_word_embeddings"] = true.into();
            t.as_object_mut().unwrap().remove("output_gate_type");
            t["layer_types"] = serde_json::json!(
                (0..layers)
                    .map(|i| if i % 4 == 3 {
                        "full_attention"
                    } else {
                        "linear_attention"
                    })
                    .collect::<Vec<_>>()
            );
            v["vision_config"] = serde_json::json!({"hidden_size":vw,"intermediate_size":vf,
                "depth":depth,"num_heads":vh,"out_hidden_size":width,"patch_size":16,
                "temporal_patch_size":2,"spatial_merge_size":2,"num_position_embeddings":2304,
                "in_channels":3,"dtype":"bfloat16","hidden_act":"gelu_pytorch_tanh",
                "deepstack_visual_indexes":[]});
            let cfg = QwenConfig::parse(&v).unwrap();
            assert_eq!(
                (cfg.width, cfg.layers, cfg.tied),
                (width as usize, layers as usize, true)
            );
            for (path, wrong) in [
                ("/tie_word_embeddings", serde_json::json!(false)),
                ("/vision_config/out_hidden_size", serde_json::json!(5120)),
                (
                    "/vision_config/deepstack_visual_indexes",
                    serde_json::json!([2]),
                ),
                ("/vision_config/hidden_act", serde_json::json!("gelu")),
                ("/text_config/mtp_num_hidden_layers", serde_json::json!(1)),
                ("/text_config/linear_num_key_heads", serde_json::json!(8)),
                ("/text_config/attn_output_gate", serde_json::json!(false)),
                ("/quantization/bits", serde_json::json!(8)),
            ] {
                let mut bad = v.clone();
                *bad.pointer_mut(path).unwrap() = wrong;
                assert!(QwenConfig::parse(&bad).is_err(), "accepted {path}");
            }
        }
    }
    pub(crate) fn config() -> Value {
        serde_json::json!({
            "model_type": "qwen3_5", "tie_word_embeddings": false,
            "quantization": {"bits":4,"group_size":64,"mode":"affine"},
            "text_config": {
                "hidden_size":5120,"intermediate_size":17408,"num_hidden_layers":64,
                "num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,"vocab_size":248320,
                "full_attention_interval":4,"linear_conv_kernel_dim":4,"linear_key_head_dim":128,
                "linear_value_head_dim":128,"linear_num_key_heads":16,"linear_num_value_heads":48,
                "hidden_act":"silu","dtype":"bfloat16","attention_bias":false,
                "tie_word_embeddings":false,"attn_output_gate":true,"output_gate_type":"swish",
                "mamba_ssm_dtype":"float32", "max_position_embeddings":262144,"rms_norm_eps":1e-6,
                "layer_types":(0..64).map(|i|if (i+1)%4==0{"full_attention"}else{"linear_attention"}).collect::<Vec<_>>(),
                "partial_rotary_factor":0.25,
                "rope_parameters":{"rope_type":"default","mrope_interleaved":true,
                    "partial_rotary_factor":0.25,"mrope_section":[11,11,10],"rope_theta":10000000}
            }
        })
    }
    #[test]
    fn accepts_only_supported_geometry_and_arithmetic() {
        let good = config();
        let parsed = QwenConfig::parse(&good).unwrap();
        assert_eq!(parsed.context, 262144);
        assert_eq!(parsed.eps, 1e-6);
        assert_eq!(parsed.rope, 10000000.);
        for (pointer, value) in [
            ("/model_type", serde_json::json!("qwen3_5_moe")),
            ("/text_config/hidden_size", serde_json::json!(4096)),
            ("/text_config/dtype", serde_json::json!("float16")),
            ("/text_config/attention_bias", serde_json::json!(true)),
            (
                "/text_config/layer_types/3",
                serde_json::json!("linear_attention"),
            ),
            ("/text_config/rms_norm_eps", serde_json::json!(0)),
            (
                "/text_config/rope_parameters/mrope_interleaved",
                serde_json::json!(false),
            ),
            (
                "/text_config/rope_parameters/rope_theta",
                serde_json::json!(-1),
            ),
        ] {
            let mut v = good.clone();
            *v.pointer_mut(pointer).unwrap() = value;
            assert!(QwenConfig::parse(&v).is_err(), "accepted {pointer}");
        }
    }
    #[test]
    fn rejects_conflicting_and_per_tensor_quantization() {
        let mut v = config();
        v["quantization_config"] = serde_json::json!({"bits":8,"group_size":64,"mode":"affine"});
        assert!(QwenConfig::parse(&v).is_err());
        v.as_object_mut().unwrap().remove("quantization_config");
        v["quantization"]["lm_head"] = serde_json::json!({"bits":8});
        assert!(QwenConfig::parse(&v).is_err());
        v = config();
        v["quantization_config"] = v["quantization"].clone();
        v.as_object_mut().unwrap().remove("quantization");
        assert!(QwenConfig::parse(&v).is_ok());
    }
}
