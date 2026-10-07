//! Kolibri-1's checkpoint contracts: the elected MLX mixed affine-4/8 build
//! (Metal) and the compressed-tensors NVFP4 build (CUDA: routed experts W4A4,
//! everything else BF16). Text only. The GGUF lane reads its own header.
use serde_json::{Value, json};
use std::path::Path;

pub const WIDTH: usize = 2560;
pub const LAYERS: usize = 50;
pub const HEADS: usize = 48;
pub const KV_HEADS: usize = 4;
pub const HEAD_DIM: usize = 128;
pub const EXPERTS: usize = 384;
pub const ACTIVE: usize = 6;
pub const FF: usize = 512;
pub const VOCAB: usize = 128000;
pub const WINDOW: usize = 513;
pub const MAX_CONTEXT: usize = 262144;

#[derive(Debug)]
pub struct KolibriConfig {
    value: Value,
}

impl KolibriConfig {
    pub fn read(dir: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(dir.join("config.json")).map_err(|e| e.to_string())?;
        Self::parse(serde_json::from_slice(&bytes).map_err(|e| e.to_string())?)
    }

    pub fn parse(value: Value) -> Result<Self, String> {
        let this = Self::parse_geometry(value)?;
        this.quantization("", 4)?;
        Ok(this)
    }

    /// The compressed-tensors NVFP4 build (`nvfp4-pack-quantized`): only the
    /// routed experts' gate/up/down projections are 4-bit float (group 16,
    /// activations 4-bit too); attention, shared experts, router, norms,
    /// embeddings and head stay BF16 - which is what the CUDA lane loads.
    pub fn read_nvfp4(dir: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(dir.join("config.json")).map_err(|e| e.to_string())?;
        let this =
            Self::parse_geometry(serde_json::from_slice(&bytes).map_err(|e| e.to_string())?)?;
        let q = this
            .value
            .get("quantization_config")
            .ok_or("Kolibri NVFP4 requires a quantization_config")?;
        if q.get("quant_method") != Some(&json!("compressed-tensors"))
            || q.get("format") != Some(&json!("nvfp4-pack-quantized"))
        {
            return Err("Kolibri NVFP4 requires compressed-tensors nvfp4-pack-quantized".into());
        }
        let groups = q
            .get("config_groups")
            .and_then(Value::as_object)
            .ok_or("Kolibri NVFP4: config_groups missing")?;
        for (name, g) in groups {
            let w = &g["weights"];
            if w.get("num_bits") != Some(&json!(4))
                || w.get("type") != Some(&json!("float"))
                || w.get("group_size") != Some(&json!(16))
            {
                return Err(format!(
                    "Kolibri NVFP4 {name}: weights are not 4-bit float / group 16"
                ));
            }
            // every target must be a routed-expert projection; a quantized
            // attention or shared-expert plane is a different build
            let targets = g
                .get("targets")
                .and_then(Value::as_array)
                .ok_or("targets missing")?;
            if targets.iter().any(|t| {
                !t.as_str()
                    .is_some_and(|t| t.contains("mlp\\.experts") || t.contains("mlp.experts"))
            }) {
                return Err(format!(
                    "Kolibri NVFP4 {name}: quantizes more than the routed experts"
                ));
            }
        }
        Ok(this)
    }

    /// The architecture contract shared by every Kolibri build.
    pub fn parse_geometry(value: Value) -> Result<Self, String> {
        for (key, expected) in [
            ("model_type", json!("kolibri1")),
            ("hidden_size", json!(WIDTH)),
            ("num_hidden_layers", json!(LAYERS)),
            ("num_attention_heads", json!(HEADS)),
            ("num_key_value_heads", json!(KV_HEADS)),
            ("head_dim", json!(HEAD_DIM)),
            ("num_experts", json!(EXPERTS)),
            ("num_experts_per_tok", json!(ACTIVE)),
            ("moe_intermediate_size", json!(FF)),
            ("shared_expert_intermediate_size", json!(FF)),
            ("vocab_size", json!(VOCAB)),
            ("sliding_window", json!(WINDOW)),
            ("max_position_embeddings", json!(MAX_CONTEXT)),
            ("norm_topk_prob", json!(false)),
            ("tie_word_embeddings", json!(false)),
            ("attention_bias", json!(false)),
            ("hidden_act", json!("silu")),
        ] {
            if value.get(key) != Some(&expected) {
                return Err(format!("Kolibri requires {key}={expected}"));
            }
        }
        for (key, expected) in [("rms_norm_eps", 1e-6), ("rope_theta", 10000.0)] {
            if value.get(key).and_then(Value::as_f64) != Some(expected) {
                return Err(format!("Kolibri requires {key}={expected}"));
            }
        }
        let layers = (0..LAYERS)
            .map(|i| {
                if Self::sliding(i) {
                    "sliding_attention"
                } else {
                    "full_attention"
                }
            })
            .collect::<Vec<_>>();
        if value.get("layer_types") != Some(&json!(layers)) {
            return Err("Kolibri requires four sliding layers followed by one full layer".into());
        }
        Ok(Self { value })
    }

    pub fn sliding(layer: usize) -> bool {
        !(layer + 1).is_multiple_of(5)
    }

    /// Check every projection's override, not just the default quantization.
    pub fn quantization(&self, name: &str, bits: usize) -> Result<(), String> {
        let root = self
            .value
            .get("quantization")
            .ok_or("missing MLX quantization")?;
        let q = root.get(name).unwrap_or(root);
        if q.get("bits") != Some(&json!(bits))
            || q.get("group_size") != Some(&json!(64))
            || q.get("mode") != Some(&json!("affine"))
        {
            return Err(format!("{name}: requires affine-{bits}/group-64"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn layer_pattern_and_window_include_current_token() {
        assert_eq!(
            (0..LAYERS).filter(|&l| KolibriConfig::sliding(l)).count(),
            40
        );
        assert!(KolibriConfig::sliding(0));
        assert!(!KolibriConfig::sliding(4));
        assert!(!KolibriConfig::sliding(49));
        assert_eq!(WINDOW - 1, 512);
    }
    #[test]
    fn reject_incomplete_or_other_architecture() {
        assert!(KolibriConfig::parse(json!({"model_type":"qwen3_moe"})).is_err());
        assert!(KolibriConfig::parse(json!({"model_type":"kolibri1"})).is_err());
    }
}
