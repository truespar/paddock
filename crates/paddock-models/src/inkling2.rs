//! Thinking Machines' Inkling 2 Mini checkpoint contract
//! (`thinkingmachines/Inkling-2-Mini`, model type `inkling2_loop`): the
//! official release, routed experts NVFP4 (layers 0-19) and MXFP8 (20-23),
//! everything else BF16 - what the CUDA lane loads.
//!
//! Every switch the engine hard-codes is checked here - the loop's span and
//! passes, the attention pattern, the relative-bias extent, the clips, the
//! softcaps, the router, the experts' quantization recipe - so a sibling
//! checkpoint with another geometry fails at load instead of decoding with
//! the wrong loop or window.
use serde_json::{Value, json};
use std::path::Path;

pub const WIDTH: usize = 2304;
pub const LAYERS: usize = 24;
pub const HEADS: usize = 24;
pub const KV_HEADS: usize = 8;
pub const HEAD_DIM: usize = 128;
/// Per-head width of the relative states mixed into the position bias.
pub const D_REL: usize = 16;
/// Backward distances the relative bias covers (0..REL_EXTENT, zero past).
pub const REL_EXTENT: usize = 1024;
/// Keys a sliding layer sees, the current one included.
pub const WINDOW: usize = 1024;
/// The table's rows; the unembed's first VOCAB_REAL rows are real.
pub const VOCAB: usize = 201024;
pub const VOCAB_REAL: usize = 200058;
pub const EXPERTS: usize = 128;
pub const TOP_K: usize = 8;
pub const FF: usize = 1152;
/// The looped core: layers LOOP_START..LOOP_END run PASSES times.
pub const LOOP_START: usize = 6;
pub const LOOP_END: usize = 18;
pub const PASSES: usize = 3;
/// Layer visits a token: the loop's passes unrolled.
pub const VISITS: usize = LAYERS + (PASSES - 1) * (LOOP_END - LOOP_START);
/// Rank of the loop controllers' gate projections.
pub const GATE_RANK: usize = 64;
/// Layers summed into one attention-residual block (boundaries at 0, 6, 12, 18).
pub const ATTN_RES_BLOCK: usize = 6;
/// Taps of every short convolution (K, V, post-attention, post-MLP).
pub const SCONV_K: usize = 4;
/// Layers 20..LAYERS run their experts MXFP8; the rest NVFP4.
pub const MXFP8_FROM: usize = 20;
pub const EPS: f32 = 1e-6;
/// The query / key L2 clip (16 * sqrt(2)) and the relative states' clip.
pub const QK_CLIP: f32 = 22.627_417;
pub const REL_CLIP: f32 = 8.0;
/// Full layers scale queries and relative states by
/// `1 + LOG_ALPHA * ln(max((pos + 1) / LOG_FLOOR, 1))`.
pub const LOG_ALPHA: f32 = 0.1;
pub const LOG_FLOOR: usize = 8192;
pub const GATE_SOFTCAP: f32 = 4.0;
pub const UP_SOFTCAP: f32 = 25.0;
pub const LOGIT_SOFTCAP: f32 = 30.0;
/// The router's top-k weights, renormalized, are scaled by this.
pub const ROUTE_SCALE: f32 = 8.0;
/// `<|content_model_end_sampling|>` - the eos.
pub const EOS: u32 = 200006;
/// `<|message_model|>`, `<|content_thinking|>`, `<|end_message|>`.
pub const MESSAGE_MODEL: u32 = 200001;
pub const CONTENT_THINKING: u32 = 200008;
pub const END_MESSAGE: u32 = 200010;
/// The image placeholder (`<|unused_200054|>`), one per image feature.
pub const IMAGE_TOKEN: u32 = 200054;

/// Whether layer `i` is one of the four full-attention layers (5, 11, 17,
/// 23); the other twenty slide over WINDOW keys.
pub fn full_attention(i: usize) -> bool {
    i % ATTN_RES_BLOCK == ATTN_RES_BLOCK - 1
}

/// Whether layer `i` is in the looped core.
pub fn looped(i: usize) -> bool {
    (LOOP_START..LOOP_END).contains(&i)
}

/// The layers of visit `v` (0..VISITS) and its loop pass (0 outside the
/// core): prelude 0-5, then the core's three passes, then the coda.
pub fn visit(v: usize) -> (usize, usize) {
    let core = LOOP_END - LOOP_START;
    if v < LOOP_START {
        (v, 0)
    } else if v < LOOP_START + PASSES * core {
        let k = v - LOOP_START;
        (LOOP_START + k % core, k / core)
    } else {
        (v - (PASSES - 1) * core, 0)
    }
}

#[derive(Debug)]
pub struct Inkling2Config {
    value: Value,
    /// `model_max_length` - the positions the checkpoint declares (1Mi).
    pub max_positions: usize,
}

impl Inkling2Config {
    /// `config.json` + `hf_quant_config.json` of the official release.
    pub fn read(dir: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(dir.join("config.json")).map_err(|e| e.to_string())?;
        let this = Self::parse(serde_json::from_slice(&bytes).map_err(|e| e.to_string())?)?;
        let bytes = std::fs::read(dir.join("hf_quant_config.json")).map_err(|e| e.to_string())?;
        check_quant(&serde_json::from_slice(&bytes).map_err(|e| e.to_string())?)?;
        Ok(this)
    }

    /// The architecture contract (`config.json`).
    pub fn parse(value: Value) -> Result<Self, String> {
        if value.get("model_type") != Some(&json!("inkling2_loop")) {
            return Err("Inkling 2 requires model_type=inkling2_loop".into());
        }
        if value.get("eos_token_id") != Some(&json!(EOS)) {
            return Err(format!("Inkling 2 requires eos_token_id={EOS}"));
        }
        let t = value
            .get("text_config")
            .ok_or("Inkling 2 requires a text_config")?;
        for (key, expected) in [
            ("model_type", json!("inkling2_loop_text")),
            ("hidden_size", json!(WIDTH)),
            ("num_hidden_layers", json!(LAYERS)),
            ("vocab_size", json!(VOCAB)),
            ("unpadded_vocab_size", json!(VOCAB_REAL)),
            ("num_attention_heads", json!(HEADS)),
            ("num_key_value_heads", json!(KV_HEADS)),
            ("head_dim", json!(HEAD_DIM)),
            ("swa_num_attention_heads", json!(HEADS)),
            ("swa_num_key_value_heads", json!(KV_HEADS)),
            ("swa_head_dim", json!(HEAD_DIM)),
            ("sliding_window_size", json!(WINDOW)),
            ("d_rel", json!(D_REL)),
            ("rel_extent", json!(REL_EXTENT)),
            ("q_bias", json!(false)),
            ("o_bias", json!(false)),
            ("log_scaling_n_floor", json!(LOG_FLOOR)),
            ("use_value_norm_no_gamma", json!(true)),
            ("sconv_kernel_size", json!(SCONV_K)),
            ("n_layers_per_attn_res_block", json!(ATTN_RES_BLOCK)),
            ("attn_res_normalize_values", json!(true)),
            ("n_routed_experts", json!(EXPERTS)),
            ("num_experts_per_tok", json!(TOP_K)),
            ("n_shared_experts", json!(0)),
            ("intermediate_size", json!(FF)),
            ("use_gate_bias", json!(true)),
        ] {
            if t.get(key) != Some(&expected) {
                return Err(format!("Inkling 2 requires text_config.{key}={expected}"));
            }
        }
        // f64 literals: the f32 constants widened would not equal the JSON's
        for (key, expected) in [
            ("rms_norm_eps", 1e-6),
            ("log_scaling_alpha", 0.1),
            ("qk_l2_clip_threshold", 22.627416997969522),
            ("rel_l2_clip_threshold", 8.0),
            ("final_logit_softcapping", 30.0),
            ("mlp_gate_softcap", 4.0),
            ("mlp_up_softcap", 25.0),
            ("route_scale", 8.0),
        ] {
            if t.get(key).and_then(Value::as_f64) != Some(expected) {
                return Err(format!("Inkling 2 requires text_config.{key}={expected}"));
            }
        }
        let local: Vec<usize> = (0..LAYERS).filter(|&i| !full_attention(i)).collect();
        if t.get("local_layer_ids") != Some(&json!(local)) {
            return Err("Inkling 2 requires full attention at layers 5, 11, 17, 23 only".into());
        }
        let lc = t
            .get("loop_config")
            .ok_or("Inkling 2 requires a loop_config")?;
        for (key, expected) in [
            ("core_start", json!(LOOP_START)),
            ("core_end", json!(LOOP_END)),
            ("num_passes", json!(PASSES)),
            ("gate_rank", json!(GATE_RANK)),
            ("is_varloop", json!(false)),
        ] {
            if lc.get(key) != Some(&expected) {
                return Err(format!("Inkling 2 requires loop_config.{key}={expected}"));
            }
        }
        // the checkpoint's own scoring rule, not a default worth guessing
        if let Some(f) = t.get("scoring_func")
            && f != &json!("sigmoid")
        {
            return Err("Inkling 2 requires sigmoid router scores".into());
        }
        let max_positions =
            t.get("model_max_length")
                .and_then(Value::as_u64)
                .ok_or("Inkling 2 requires text_config.model_max_length")? as usize;
        Ok(Self {
            value,
            max_positions,
        })
    }

    /// The raw `config.json` (the vision and drafter sections).
    pub fn value(&self) -> &Value {
        &self.value
    }
}

/// The picture tower (`vision_config`): a 23-block pre-LN ViT, 1024 wide,
/// 16 heads of 64 on an axial rope, LayerScale on both residual branches, a
/// class token per picture, a learned 32 x 32 position field resampled to
/// the picture; then the 2 x 2 merge adapter 4096 -> 4096 -> the language
/// model's 2304. The processor never resizes: one language-model token per
/// 40 x 40 cell of the picture padded to whole cells.
pub mod vision {
    use serde_json::{Value, json};

    pub const WIDTH: usize = 1024;
    pub const LAYERS: usize = 23;
    pub const HEADS: usize = 16;
    pub const HEAD_DIM: usize = WIDTH / HEADS;
    pub const FF: usize = 4096;
    /// Native patch side, and the 2 x 2 merge into one token.
    pub const PATCH: usize = 20;
    pub const MERGE: usize = 2;
    /// Pixels a side one language-model token covers.
    pub const CELL: usize = PATCH * MERGE;
    pub const POS_GRID: usize = 32;
    pub const PROJ: usize = 4096;
    pub const EPS: f32 = 1e-5;
    pub const ROPE_THETA: f32 = 10000.0;
    /// Tokens one picture may cost on this lane (the tower's workspace is
    /// sized for it): a 4K screen is 96 x 54 = 5184.
    pub const MAX_TOKENS: usize = 8192;

    /// A `w x h` picture's token grid: (rows, columns) of 40-pixel cells.
    pub fn grid(w: usize, h: usize) -> (usize, usize) {
        (h.div_ceil(CELL), w.div_ceil(CELL))
    }

    /// The tower contract (`config.json`'s `vision_config`) and the
    /// processor's (`processor_config.json`): `Ok(false)` when the release
    /// carries no tower.
    pub fn check(config: &Value, processor: Option<&Value>) -> Result<bool, String> {
        let Some(v) = config.get("vision_config") else {
            return Ok(false);
        };
        for (key, expected) in [
            ("model_type", json!("inkling2_vision")),
            ("hidden_size", json!(WIDTH)),
            ("num_hidden_layers", json!(LAYERS)),
            ("num_attention_heads", json!(HEADS)),
            ("native_patch_size", json!(PATCH)),
            ("adapter_merge_size", json!(MERGE)),
            ("projection_hidden_dim", json!(PROJ)),
            ("projector_bias", json!(false)),
            ("posemb_grid_size", json!(POS_GRID)),
            ("text_hidden_size", json!(super::WIDTH)),
        ] {
            if v.get(key) != Some(&expected) {
                return Err(format!("Inkling 2 requires vision_config.{key}={expected}"));
            }
        }
        for (key, expected) in [("mlp_ratio", 4.0), ("layer_norm_eps", 1e-5)] {
            if v.get(key).and_then(Value::as_f64) != Some(expected) {
                return Err(format!("Inkling 2 requires vision_config.{key}={expected}"));
            }
        }
        // the axial rope at its default theta and the exact GELU: a config
        // that names either is a different tower
        for key in ["rope_parameters", "rope_theta", "hidden_act"] {
            if v.get(key).is_some() {
                return Err(format!("Inkling 2's tower: unexpected vision_config.{key}"));
            }
        }
        let p = processor
            .and_then(|p| p.get("image_processor"))
            .ok_or("Inkling 2's tower needs processor_config.json")?;
        let half = json!([0.5, 0.5, 0.5]);
        if p.get("size") != Some(&json!({"height": CELL, "width": CELL}))
            || p.get("image_mean") != Some(&half)
            || p.get("image_std") != Some(&half)
            || p.get("rescale_factor").and_then(Value::as_f64) != Some(1.0 / 255.0)
        {
            return Err("Inkling 2's processor: 40 x 40 cells, mean = std = 0.5, 1/255".into());
        }
        if processor.and_then(|p| p.get("image_token")) != Some(&json!("<|unused_200054|>")) {
            return Err("Inkling 2's processor: image token <|unused_200054|>".into());
        }
        Ok(true)
    }
}

/// `hf_quant_config.json`: the routed experts' recipe per layer - NVFP4 with
/// per-token "fouroversix_mae" activations on layers 0..MXFP8_FROM, MXFP8
/// W8A8 on the rest - and nothing else quantized.
pub fn check_quant(q: &Value) -> Result<(), String> {
    if q.get("quant_method") != Some(&json!("inkling2")) {
        return Err("Inkling 2 requires quant_method=inkling2".into());
    }
    let groups = q
        .pointer("/model/parameter_groups")
        .and_then(Value::as_array)
        .ok_or("Inkling 2 quant config has no model.parameter_groups")?;
    let mut nvfp4 = Vec::new();
    let mut mxfp8 = Vec::new();
    for g in groups {
        let dtype = g.get("compute_dtype").and_then(Value::as_str).unwrap_or("");
        let tensors: Vec<&str> = g
            .get("tensors")
            .and_then(Value::as_array)
            .map(|v| v.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        match dtype {
            "nvfp4" => {
                let a = g.get("required_activation_quantization");
                for (key, expected) in [
                    ("format", json!("nvfp4_e2m1")),
                    ("amax_type", json!("per_token")),
                    ("block_size", json!(16)),
                    ("scale_dtype", json!("float8_e4m3fn")),
                    ("scale_selection_rule", json!("fouroversix_mae")),
                    ("rounding", json!("nearest")),
                ] {
                    if a.and_then(|a| a.get(key)) != Some(&expected) {
                        return Err(format!(
                            "Inkling 2 NVFP4 activations require {key}={expected}"
                        ));
                    }
                }
                if g.get("weight_block_shape") != Some(&json!([1, 16]))
                    || g.get("weight_scale_format") != Some(&json!("amax_denominator"))
                {
                    return Err(
                        "Inkling 2 NVFP4 weights require 1 x 16 blocks over an amax denominator"
                            .into(),
                    );
                }
                nvfp4.extend(tensors);
            }
            "mxfp8" => {
                let a = g.get("required_activation_quantization");
                for (key, expected) in [
                    ("format", json!("mxfp8_e4m3")),
                    ("block_size", json!(32)),
                    ("scale_dtype", json!("ue8m0")),
                    ("rounding", json!("nearest")),
                ] {
                    if a.and_then(|a| a.get(key)) != Some(&expected) {
                        return Err(format!(
                            "Inkling 2 MXFP8 activations require {key}={expected}"
                        ));
                    }
                }
                if g.get("weight_block_shape") != Some(&json!([32, 32])) {
                    return Err("Inkling 2 MXFP8 weights require 32 x 32 blocks".into());
                }
                mxfp8.extend(tensors);
            }
            "bfloat16" | "float32" => {
                if let Some(t) = tensors.iter().find(|t| t.contains("mlp.experts")) {
                    return Err(format!(
                        "Inkling 2 expects every expert quantized, {t} is not"
                    ));
                }
            }
            other => {
                return Err(format!(
                    "Inkling 2 quant config: unknown compute_dtype {other}"
                ));
            }
        }
    }
    let experts = |range: std::ops::Range<usize>| -> Vec<String> {
        range
            .flat_map(|i| {
                [
                    format!("model.llm.layers.{i}.mlp.experts.w13_weight"),
                    format!("model.llm.layers.{i}.mlp.experts.w2_weight"),
                ]
            })
            .collect()
    };
    let same = |got: &[&str], want: Vec<String>| {
        let mut got: Vec<&str> = got.to_vec();
        got.sort_unstable();
        let mut want = want;
        want.sort_unstable();
        got == want.iter().map(String::as_str).collect::<Vec<_>>()
    };
    if !same(&nvfp4, experts(0..MXFP8_FROM)) {
        return Err(format!(
            "Inkling 2 requires NVFP4 experts on layers 0..{MXFP8_FROM} exactly"
        ));
    }
    if !same(&mxfp8, experts(MXFP8_FROM..LAYERS)) {
        return Err(format!(
            "Inkling 2 requires MXFP8 experts on layers {MXFP8_FROM}..{LAYERS} exactly"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visits_unroll_the_loop() {
        let layers: Vec<usize> = (0..VISITS).map(|v| visit(v).0).collect();
        let mut want: Vec<usize> = (0..LOOP_START).collect();
        for _ in 0..PASSES {
            want.extend(LOOP_START..LOOP_END);
        }
        want.extend(LOOP_END..LAYERS);
        assert_eq!(layers, want);
        assert_eq!(VISITS, 48);
        assert_eq!(visit(LOOP_START + 12), (LOOP_START, 1));
        assert_eq!(visit(VISITS - 1), (LAYERS - 1, 0));
        let full: Vec<usize> = (0..LAYERS).filter(|&i| full_attention(i)).collect();
        assert_eq!(full, vec![5, 11, 17, 23]);
    }

    #[test]
    fn the_official_config_passes_and_drift_fails() {
        let mut t: Value = serde_json::from_str(
            r#"{
            "model_type": "inkling2_loop_text", "hidden_size": 2304, "num_hidden_layers": 24,
            "vocab_size": 201024, "unpadded_vocab_size": 200058, "num_attention_heads": 24,
            "num_key_value_heads": 8, "head_dim": 128, "swa_num_attention_heads": 24,
            "swa_num_key_value_heads": 8, "swa_head_dim": 128, "sliding_window_size": 1024,
            "d_rel": 16, "rel_extent": 1024, "q_bias": false, "o_bias": false,
            "log_scaling_n_floor": 8192, "use_value_norm_no_gamma": true, "sconv_kernel_size": 4,
            "n_layers_per_attn_res_block": 6, "attn_res_normalize_values": true,
            "n_routed_experts": 128, "num_experts_per_tok": 8, "n_shared_experts": 0,
            "intermediate_size": 1152, "use_gate_bias": true, "rms_norm_eps": 1e-6,
            "log_scaling_alpha": 0.1, "qk_l2_clip_threshold": 22.627416997969522,
            "rel_l2_clip_threshold": 8.0, "final_logit_softcapping": 30.0,
            "mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0, "route_scale": 8.0,
            "local_layer_ids": [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22],
            "loop_config": {"core_start": 6, "core_end": 18, "num_passes": 3, "gate_rank": 64,
                            "is_varloop": false},
            "model_max_length": 1048576
        }"#,
        )
        .expect("json");
        let cfg = |t: &Value| json!({"model_type": "inkling2_loop", "eos_token_id": 200006, "text_config": t});
        let c = Inkling2Config::parse(cfg(&t)).expect("the official geometry");
        assert_eq!(c.max_positions, 1 << 20);
        t["loop_config"]["num_passes"] = json!(2);
        assert!(
            Inkling2Config::parse(cfg(&t)).is_err(),
            "a two-pass sibling must fail"
        );
    }

    /// The downloaded release (`INKLING2_DIR`, else the data root's copy),
    /// config and quant recipe; skipped where it is not on disk.
    #[test]
    fn the_downloaded_release_reads() {
        let dir = std::env::var_os("INKLING2_DIR")
            .map(std::path::PathBuf::from)
            .or_else(|| {
                std::env::var_os("HOME")
                    .map(|h| std::path::Path::new(&h).join("paddock/models/Inkling-2-Mini"))
            });
        let Some(dir) = dir.filter(|d| d.join("config.json").exists()) else {
            eprintln!("SKIP: no Inkling-2-Mini on disk");
            return;
        };
        Inkling2Config::read(&dir).expect("the official release");
    }
}
