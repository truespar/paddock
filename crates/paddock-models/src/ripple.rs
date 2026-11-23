//! Ricursive's Ripple-65B checkpoint contract: the official NVFP4 build
//! (ModelOpt `NVFP4`, the MLP linears 4-bit, everything else BF16) - what
//! the CUDA lane loads. Text only.
//!
//! Every switch the engine hard-codes is checked here, so a sibling
//! checkpoint with another geometry or recipe fails at load instead of
//! decoding with the wrong window, gate or clamp.
use serde_json::{Value, json};
use std::path::Path;

pub const WIDTH: usize = 5120;
pub const LAYERS: usize = 80;
pub const HEADS: usize = 80;
pub const KV_HEADS: usize = 8;
pub const HEAD_DIM: usize = 128;
pub const FF: usize = 40960;
pub const VOCAB: usize = 100352;
/// Keys a sliding layer sees, the current one included (the reference's
/// mask keeps `q - (window - 1) <= k <= q`).
pub const WINDOW: usize = 1024;
pub const EPS: f32 = 1e-6;
pub const ROPE_THETA: f32 = 10000.0;
/// The FFN's clamp (`ffn_clamp_threshold`): gate <= it, up in [-it, it].
pub const FFN_CLAMP: f32 = 20.0;
/// `<|endoftext|>` (bos) and `<|im_end|>` - the config's two eos ids.
pub const BOS: u32 = 100257;
pub const IM_END: u32 = 100265;
/// The draft block's mask token (the DFlash2 drafter's `mask_token_id`).
pub const DRAFT_MASK: u32 = 100278;

#[derive(Debug)]
pub struct RippleConfig {
    value: Value,
    /// `embed_scale` (sqrt(width) on the shipped build), applied to the
    /// gathered embedding rows.
    pub embed_scale: f32,
    /// `max_position_embeddings` - the positions the checkpoint declares.
    pub max_positions: usize,
}

impl RippleConfig {
    /// The NVFP4 build's `config.json`: the architecture contract plus the
    /// quantization recipe (ModelOpt NVFP4, group 16, the MLP linears only).
    pub fn read_nvfp4(dir: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(dir.join("config.json")).map_err(|e| e.to_string())?;
        let this =
            Self::parse_geometry(serde_json::from_slice(&bytes).map_err(|e| e.to_string())?)?;
        this.check_nvfp4()?;
        Ok(this)
    }

    /// The architecture contract shared by every Ripple-65B build.
    pub fn parse_geometry(value: Value) -> Result<Self, String> {
        for (key, expected) in [
            ("model_type", json!("ricursive")),
            ("hidden_size", json!(WIDTH)),
            ("num_hidden_layers", json!(LAYERS)),
            ("num_attention_heads", json!(HEADS)),
            ("num_key_value_heads", json!(KV_HEADS)),
            ("head_dim", json!(HEAD_DIM)),
            ("intermediate_size", json!(FF)),
            ("vocab_size", json!(VOCAB)),
            ("sliding_window", json!(WINDOW)),
            ("hidden_act", json!("silu")),
            ("tie_word_embeddings", json!(false)),
            ("is_hybrid_swa", json!(true)),
            ("use_gated_attention", json!(true)),
            ("gate_full_precision", json!(true)),
            ("use_valres", json!(true)),
            ("bos_token_id", json!(BOS)),
            ("eos_token_id", json!([IM_END, BOS])),
        ] {
            if value.get(key) != Some(&expected) {
                return Err(format!("Ripple requires {key}={expected}"));
            }
        }
        // f64 literals: the f32 constants widened would not equal the JSON's
        for (key, expected) in [
            ("rms_norm_eps", 1e-6),
            ("rope_theta", 10000.0),
            ("ffn_clamp_threshold", 20.0),
        ] {
            if value.get(key).and_then(Value::as_f64) != Some(expected) {
                return Err(format!("Ripple requires {key}={expected}"));
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
            return Err("Ripple requires four sliding layers followed by one full layer".into());
        }
        // a null scale means one baked into the table (the reference's
        // "folded" checkpoints) - not this build
        let embed_scale = value
            .get("embed_scale")
            .and_then(Value::as_f64)
            .ok_or("Ripple requires a numeric embed_scale")? as f32;
        if (embed_scale - (WIDTH as f32).sqrt()).abs() > 1e-3 {
            return Err(format!(
                "Ripple embed_scale {embed_scale} is not sqrt({WIDTH})"
            ));
        }
        let max_positions = value
            .get("max_position_embeddings")
            .and_then(Value::as_u64)
            .ok_or("Ripple requires max_position_embeddings")? as usize;
        Ok(Self {
            value,
            embed_scale,
            max_positions,
        })
    }

    /// Sliding layers: four of every five; layers 4, 9, ..., 79 attend in full.
    pub fn sliding(layer: usize) -> bool {
        !(layer + 1).is_multiple_of(5)
    }

    /// ModelOpt `NVFP4`: one config group, 4-bit float weights and inputs at
    /// group 16 (static), targeting the Linear modules, with every attention
    /// module, the embeddings and the head left out - so exactly the MLP
    /// linears are quantized, which is what the loader reads.
    fn check_nvfp4(&self) -> Result<(), String> {
        let q = self
            .value
            .get("quantization_config")
            .ok_or("Ripple NVFP4 requires a quantization_config")?;
        if q.get("quant_method") != Some(&json!("modelopt"))
            || q.get("quant_algo") != Some(&json!("NVFP4"))
        {
            return Err("Ripple NVFP4 requires ModelOpt quant_algo NVFP4".into());
        }
        let groups = q
            .get("config_groups")
            .and_then(Value::as_object)
            .ok_or("Ripple NVFP4: config_groups missing")?;
        if groups.len() != 1 {
            return Err(format!(
                "Ripple NVFP4: {} config groups, expected one",
                groups.len()
            ));
        }
        for (name, g) in groups {
            for side in ["weights", "input_activations"] {
                let s = &g[side];
                if s.get("num_bits") != Some(&json!(4))
                    || s.get("type") != Some(&json!("float"))
                    || s.get("group_size") != Some(&json!(16))
                {
                    return Err(format!(
                        "Ripple NVFP4 {name}: {side} are not 4-bit float / group 16"
                    ));
                }
            }
            if g.get("targets") != Some(&json!(["Linear"])) {
                return Err(format!("Ripple NVFP4 {name}: targets are not [\"Linear\"]"));
            }
        }
        let ignore: std::collections::HashSet<&str> = q
            .get("ignore")
            .and_then(Value::as_array)
            .ok_or("Ripple NVFP4: ignore list missing")?
            .iter()
            .filter_map(Value::as_str)
            .collect();
        let mut want: Vec<String> = vec!["lm_head".into(), "model.embed_tokens".into()];
        want.extend((0..LAYERS).map(|i| format!("model.layers.{i}.self_attn*")));
        if ignore.len() != want.len() || want.iter().any(|w| !ignore.contains(w.as_str())) {
            return Err(
                "Ripple NVFP4: the ignore list is not exactly the head, the embeddings and \
                 every attention module"
                    .into(),
            );
        }
        Ok(())
    }
}

/// The DFlash2 drafter (`ricursiveAI/Ripple-65B-Preview-DFlash2`): five
/// qwen3-style layers over the target's width - four sliding (4096 keys) and
/// a NoPE full one - with the grouped dynamic convolutions and the candidate
/// selector, conditioned on the target's post-block residuals at `taps`.
pub mod draft {
    use super::*;

    pub const LAYERS: usize = 5;
    pub const BLOCK: usize = 16;
    pub const MASK: u32 = DRAFT_MASK;
    pub const TAPS: [usize; 5] = [1, 20, 39, 58, 77];
    pub const CONV_TAPS: usize = 4;
    pub const CONV_GROUP: usize = 16;
    pub const SEL_TOP_K: usize = 32;
    pub const SEL_RANK: usize = 256;
    /// The sliding layers' keys, the current one included.
    pub const WINDOW: usize = 4096;
    /// The full layer's reach: the draft's training rows (12288 of context +
    /// 4096 predicted) - what the reference serves it at.
    pub const FULL_REACH: usize = 16384;
    /// The NoPE layer (`no_rope_layers`).
    pub const NOPE_LAYER: usize = 4;

    /// The drafter's `config.json` against everything the engine hard-codes.
    pub fn check(dir: &Path) -> Result<(), String> {
        let bytes = std::fs::read(dir.join("config.json")).map_err(|e| e.to_string())?;
        check_value(&serde_json::from_slice(&bytes).map_err(|e| e.to_string())?)
    }

    pub fn check_value(v: &Value) -> Result<(), String> {
        let layers: Vec<&str> = (0..LAYERS)
            .map(|i| {
                if i == NOPE_LAYER {
                    "full_attention"
                } else {
                    "sliding_attention"
                }
            })
            .collect();
        for (key, expected) in [
            ("architectures", json!(["DFlash2DraftModel"])),
            ("hidden_size", json!(WIDTH)),
            ("num_hidden_layers", json!(LAYERS)),
            ("num_attention_heads", json!(HEADS)),
            ("num_key_value_heads", json!(KV_HEADS)),
            ("head_dim", json!(HEAD_DIM)),
            ("intermediate_size", json!(FF)),
            ("vocab_size", json!(VOCAB)),
            ("draft_vocab_size", json!(VOCAB)),
            ("hidden_act", json!("silu")),
            ("attention_bias", json!(false)),
            ("is_causal", json!(false)),
            ("sliding_window", json!(WINDOW)),
            ("layer_types", json!(layers)),
            ("no_rope_layers", json!([NOPE_LAYER])),
            ("num_target_layers", json!(super::LAYERS)),
        ] {
            if v.get(key) != Some(&expected) {
                return Err(format!("Ripple DFlash2 requires {key}={expected}"));
            }
        }
        for (key, expected) in [("rms_norm_eps", 1e-6), ("rope_theta", 10000.0)] {
            if v.get(key).and_then(Value::as_f64) != Some(expected) {
                return Err(format!("Ripple DFlash2 requires {key}={expected}"));
            }
        }
        let d = v
            .get("dflash_config")
            .ok_or("Ripple DFlash2 requires a dflash_config")?;
        for (key, expected) in [
            ("block_size", json!(BLOCK)),
            ("mask_token_id", json!(MASK)),
            ("target_layer_ids", json!(TAPS)),
            ("conv_kernel_size", json!(CONV_TAPS)),
            ("conv_group_size", json!(CONV_GROUP)),
            ("selector_top_k", json!(SEL_TOP_K)),
            ("selector_rank", json!(SEL_RANK)),
        ] {
            if d.get(key) != Some(&expected) {
                return Err(format!(
                    "Ripple DFlash2 requires dflash_config.{key}={expected}"
                ));
            }
        }
        // the block-norm recipe the reference loads (pre-norm only)
        if d.get("block_norm").is_some_and(|b| b != &json!("pre")) {
            return Err("Ripple DFlash2: only the pre-norm block is implemented".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shipped() -> Value {
        let layers: Vec<&str> = (0..LAYERS)
            .map(|i| {
                if RippleConfig::sliding(i) {
                    "sliding_attention"
                } else {
                    "full_attention"
                }
            })
            .collect();
        let mut ignore = vec![json!("lm_head"), json!("model.embed_tokens")];
        ignore.extend((0..LAYERS).map(|i| json!(format!("model.layers.{i}.self_attn*"))));
        json!({
            "model_type": "ricursive", "hidden_size": 5120, "num_hidden_layers": 80,
            "num_attention_heads": 80, "num_key_value_heads": 8, "head_dim": 128,
            "intermediate_size": 40960, "vocab_size": 100352, "sliding_window": 1024,
            "hidden_act": "silu", "tie_word_embeddings": false, "is_hybrid_swa": true,
            "use_gated_attention": true, "gate_full_precision": true, "use_valres": true,
            "bos_token_id": 100257, "eos_token_id": [100265, 100257],
            "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "ffn_clamp_threshold": 20.0,
            "embed_scale": 71.55417527999327, "max_position_embeddings": 2097152,
            "layer_types": layers,
            "quantization_config": {
                "quant_method": "modelopt", "quant_algo": "NVFP4",
                "config_groups": {"group_0": {
                    "weights": {"num_bits": 4, "type": "float", "group_size": 16, "dynamic": false},
                    "input_activations": {"num_bits": 4, "type": "float", "group_size": 16,
                                          "dynamic": false},
                    "targets": ["Linear"]}},
                "ignore": ignore
            }
        })
    }

    #[test]
    fn the_shipped_drafter_contract_holds() {
        let layers = json!([
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention"
        ]);
        let mut v = json!({
            "architectures": ["DFlash2DraftModel"], "hidden_size": 5120,
            "num_hidden_layers": 5, "num_attention_heads": 80, "num_key_value_heads": 8,
            "head_dim": 128, "intermediate_size": 40960, "vocab_size": 100352,
            "draft_vocab_size": 100352, "hidden_act": "silu", "attention_bias": false,
            "is_causal": false, "sliding_window": 4096, "layer_types": layers,
            "no_rope_layers": [4], "num_target_layers": 80, "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "dflash_config": {"block_size": 16, "mask_token_id": 100278,
                "target_layer_ids": [1, 20, 39, 58, 77], "conv_kernel_size": 4,
                "conv_group_size": 16, "selector_top_k": 32, "selector_rank": 256}
        });
        draft::check_value(&v).unwrap();
        v["dflash_config"]["block_size"] = json!(8);
        assert!(draft::check_value(&v).is_err());
    }

    #[test]
    fn the_shipped_contract_parses_and_drift_is_refused() {
        let c = RippleConfig::parse_geometry(shipped()).unwrap();
        c.check_nvfp4().unwrap();
        assert!((c.embed_scale - 71.554_18).abs() < 1e-4);
        assert_eq!(c.max_positions, 2_097_152);
        assert!(
            RippleConfig::sliding(0) && !RippleConfig::sliding(4) && !RippleConfig::sliding(79)
        );
        let mut v = shipped();
        v["sliding_window"] = json!(4096);
        assert!(RippleConfig::parse_geometry(v).is_err());
        let mut v = shipped();
        v["ffn_clamp_threshold"] = json!(30.0);
        assert!(RippleConfig::parse_geometry(v).is_err());
        // a build that also quantizes an attention module is another recipe
        let mut v = shipped();
        v["quantization_config"]["ignore"]
            .as_array_mut()
            .unwrap()
            .retain(|s| s != "model.layers.7.self_attn*");
        let c = RippleConfig::parse_geometry(v).unwrap();
        assert!(c.check_nvfp4().is_err());
    }
}
