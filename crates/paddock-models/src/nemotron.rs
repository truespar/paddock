//! NVIDIA Nemotron 3.5 Lightning (`nemotron_h_moe`) checkpoint config.
//!
//! Two sources for the same geometry: the safetensors-primary NVFP4 lane
//! reads the HF checkpoint directory (`read` - config.json + the
//! generation_config eos SET), and the unsloth Q8_0 GGUF lane
//! reads the same facts out of the GGUF header (`from_gguf`). The parsed
//! struct is identical either way - one forward graph serves both.
//!
//! Reference facts: 52 homogeneous
//! single-residual blocks - each layer is one mixer (mamba-2 | attention |
//! moe), never an attn+ffn pair. The interleave is an explicit per-layer
//! list in the file, not a stride; we parse it rather than re-deriving it
//! so a future sibling with a different pattern loads unchanged.

use std::path::Path;

use crate::gguf::Value;
use crate::safetensors::StError;

/// One trunk layer's mixer kind, from `layers_block_type`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NemotronBlock {
    Mamba,
    Attention,
    Moe,
}

/// Parsed nemotron_h_moe `config.json` (+ eos set from generation_config).
/// Field names follow the checkpoint's own vocabulary where it is sane.
#[derive(Debug, Clone)]
pub struct NemotronConfig {
    pub hidden: usize,  // 2688
    pub n_layer: usize, // 52
    pub blocks: Vec<NemotronBlock>,
    pub vocab: usize,   // 131072
    pub max_pos: usize, // 1048576
    pub eps: f32,       // layer_norm_epsilon 1e-5 (RMS everywhere)

    // attention (NoPE - no rope; kq_scale = 1/sqrt(head_dim))
    pub n_heads: usize,    // 32
    pub n_kv_heads: usize, // 2
    pub head_dim: usize,   // 128

    // mamba-2 (d_inner = heads * head_dim, not expand * hidden)
    pub mamba_heads: usize,    // 64
    pub mamba_head_dim: usize, // 64
    pub d_state: usize,        // 128
    pub d_conv: usize,         // 4
    pub n_groups: usize,       // 8
    pub chunk: usize,          // 128

    // MoE (sigmoid + f32 correction bias, top-6 of 128, renorm, x2.5 routed)
    pub n_expert: usize,   // 128
    pub n_active: usize,   // 6
    pub moe_ff: usize,     // 1856 (squared-relu, no gate matrix)
    pub shared_ff: usize,  // 3712 (one shared expert, unscaled, parallel)
    pub routed_scale: f32, // 2.5
    /// LatentMoE (Super): the routed experts run on a `latent`-wide copy of
    /// the hidden - `fc1_latent_proj` down before them, `fc2_latent_proj` up
    /// after the weighted sum - while the router and the shared expert read
    /// the full hidden. None: experts on the hidden (Lightning).
    pub latent: Option<usize>,
    /// The checkpoint wraps the language model in the vision-language
    /// `nemotron_h_omni` (tensors under `language_model.`, a RADIO tower
    /// and projector beside it).
    pub omni: bool,
    /// MTP blocks the checkpoint carries (`num_nextn_predict_layers`).
    pub mtp_layers: usize,

    /// generation_config eos set - [2, 11]; decode stops on any of these.
    pub eos_ids: Vec<u32>,
    pub bos_id: u32, // 1 (<s>), informational: add_bos_token is false
}

impl NemotronConfig {
    /// derived: mamba d_inner (4096)
    pub fn d_inner(&self) -> usize {
        self.mamba_heads * self.mamba_head_dim
    }
    /// derived: conv channel count = d_inner + 2 * n_groups * d_state (6144)
    pub fn conv_dim(&self) -> usize {
        self.d_inner() + 2 * self.n_groups * self.d_state
    }
    /// derived: in_proj out rows = d_inner (z) + conv_dim (x|B|C) + heads (dt)
    /// = 10304, in the checkpoint's row order [z | x B C | dt].
    pub fn in_proj_rows(&self) -> usize {
        self.d_inner() + self.conv_dim() + self.mamba_heads
    }

    pub fn read(dir: &Path) -> Result<Self, StError> {
        let top: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)
                .map_err(|e| StError::Header(e.to_string()))?;
        // the vision-language wrapper carries the language model's config
        // whole, under `llm_config`
        let omni = top.get("model_type").and_then(|x| x.as_str()) == Some("nemotron_h_omni");
        let v = if omni {
            top.get("llm_config").cloned().ok_or_else(|| {
                StError::Header("nemotron_h_omni config.json: missing llm_config".into())
            })?
        } else {
            top
        };
        let miss = |k: &str| StError::Header(format!("nemotron config.json: missing {k}"));
        let u = |k: &str| {
            v.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| miss(k))
        };
        let f = |k: &str| {
            v.get(k)
                .and_then(|x| x.as_f64())
                .map(|x| x as f32)
                .ok_or_else(|| miss(k))
        };

        let model_type = v.get("model_type").and_then(|x| x.as_str()).unwrap_or("");
        if model_type != "nemotron_h" {
            return Err(StError::Header(format!(
                "not a nemotron_h checkpoint (model_type {model_type:?})"
            )));
        }

        let blocks: Vec<NemotronBlock> = v
            .get("layers_block_type")
            .and_then(|x| x.as_array())
            .ok_or_else(|| miss("layers_block_type"))?
            .iter()
            .map(|b| match b.as_str() {
                Some("mamba") => Ok(NemotronBlock::Mamba),
                Some("attention") => Ok(NemotronBlock::Attention),
                Some("moe") => Ok(NemotronBlock::Moe),
                other => Err(StError::Header(format!(
                    "layers_block_type: unknown entry {other:?}"
                ))),
            })
            .collect::<Result<_, _>>()?;

        // the omni wrapper's llm_config leaves the count to the block list
        let n_layer = if v.get("num_hidden_layers").is_some() {
            u("num_hidden_layers")?
        } else {
            blocks.len()
        };
        if blocks.len() != n_layer {
            return Err(StError::Header(format!(
                "layers_block_type has {} entries for {} layers",
                blocks.len(),
                n_layer
            )));
        }

        // eos is a SET ([2, 11]) in generation_config.json; config.json's
        // scalar eos_token_id (2) alone would under-stop chat turns.
        let mut eos_ids: Vec<u32> = Vec::new();
        if let Ok(raw) = std::fs::read(dir.join("generation_config.json"))
            && let Ok(g) = serde_json::from_slice::<serde_json::Value>(&raw)
        {
            match g.get("eos_token_id") {
                Some(serde_json::Value::Array(a)) => {
                    eos_ids = a
                        .iter()
                        .filter_map(|x| x.as_u64())
                        .map(|x| x as u32)
                        .collect();
                }
                Some(serde_json::Value::Number(n)) => {
                    if let Some(x) = n.as_u64() {
                        eos_ids = vec![x as u32];
                    }
                }
                _ => {}
            }
        }
        if eos_ids.is_empty() {
            eos_ids = vec![
                v.get("eos_token_id")
                    .and_then(|x| x.as_u64())
                    .ok_or_else(|| miss("eos_token_id"))? as u32,
            ];
        }

        Ok(Self {
            hidden: u("hidden_size")?,
            n_layer,
            blocks,
            vocab: u("vocab_size")?,
            max_pos: u("max_position_embeddings")?,
            eps: f("layer_norm_epsilon")?,
            n_heads: u("num_attention_heads")?,
            n_kv_heads: u("num_key_value_heads")?,
            head_dim: u("head_dim")?,
            mamba_heads: u("mamba_num_heads")?,
            mamba_head_dim: u("mamba_head_dim")?,
            d_state: u("ssm_state_size")?,
            d_conv: u("conv_kernel")?,
            n_groups: u("n_groups")?,
            chunk: u("chunk_size")?,
            n_expert: u("n_routed_experts")?,
            n_active: u("num_experts_per_tok")?,
            moe_ff: u("moe_intermediate_size")?,
            shared_ff: u("moe_shared_expert_intermediate_size")?,
            routed_scale: f("routed_scaling_factor")?,
            latent: v
                .get("moe_latent_size")
                .and_then(|x| x.as_u64())
                .map(|x| x as usize),
            omni,
            mtp_layers: v
                .get("num_nextn_predict_layers")
                .and_then(|x| x.as_u64())
                .unwrap_or(0) as usize,
            eos_ids,
            bos_id: v.get("bos_token_id").and_then(|x| x.as_u64()).unwrap_or(1) as u32,
        })
    }

    /// The width the routed experts compute in: the latent for LatentMoE,
    /// else the hidden.
    pub fn expert_in(&self) -> usize {
        self.latent.unwrap_or(self.hidden)
    }
}

impl NemotronConfig {
    /// Parse the same config out of a `nemotron_h_moe` GGUF header
    /// (the unsloth Q8_0 second lane). The block interleave arrives as
    /// two per-layer arrays instead of `layers_block_type`: a layer with
    /// `attention.head_count_kv[i] > 0` is attention, one with
    /// `feed_forward_length[i] > 0` is moe, everything else is mamba -
    /// exactly llama.cpp's own classification (`is_recr` / `n_ff(il) == 0`
    /// in models/nemotron-h.cpp). `block_count` counts the MTP block too
    /// (53); `nextn_predict_layers` says how many trailing blocks are MTP.
    pub fn from_gguf(g: &crate::gguf::GgufFile) -> Result<Self, StError> {
        let bad = |m: String| StError::Header(m);
        let miss = |k: &str| StError::Header(format!("nemotron gguf: missing {k}"));
        if g.architecture() != Some("nemotron_h_moe") {
            return Err(bad(format!(
                "not a nemotron_h_moe gguf (architecture {:?})",
                g.architecture()
            )));
        }
        let u = |k: &str| {
            g.arch_field(k)
                .and_then(Value::as_u64)
                .map(|x| x as usize)
                .ok_or_else(|| miss(k))
        };
        let f = |k: &str| {
            g.arch_field(k)
                .and_then(Value::as_f32)
                .ok_or_else(|| miss(k))
        };
        let u_arr = |k: &str| -> Result<Vec<usize>, StError> {
            match g.arch_field(k) {
                Some(Value::Array(items)) => items
                    .iter()
                    .map(|x| {
                        x.as_u64().map(|v| v as usize).ok_or_else(|| {
                            bad(format!("nemotron gguf: {k} holds a non-integer entry"))
                        })
                    })
                    .collect(),
                Some(_) => Err(bad(format!("nemotron gguf: {k} is not an array"))),
                None => Err(miss(k)),
            }
        };

        let block_count = u("block_count")?;
        // trailing MTP blocks are not trunk layers; the trunk loader skips
        // them and the spec lane reads them by name
        let nextn = g
            .arch_field("nextn_predict_layers")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize;
        if nextn >= block_count {
            return Err(bad(format!(
                "nemotron gguf: nextn_predict_layers {nextn} >= block_count {block_count}"
            )));
        }
        let n_layer = block_count - nextn;

        let kv_arr = u_arr("attention.head_count_kv")?;
        let ff_arr = u_arr("feed_forward_length")?;
        if kv_arr.len() != block_count || ff_arr.len() != block_count {
            return Err(bad(format!(
                "nemotron gguf: per-layer arrays have {} / {} entries for block_count {block_count}",
                kv_arr.len(),
                ff_arr.len()
            )));
        }
        let blocks: Vec<NemotronBlock> = (0..n_layer)
            .map(|i| match (kv_arr[i], ff_arr[i]) {
                (kv, _) if kv > 0 => NemotronBlock::Attention,
                (_, ff) if ff > 0 => NemotronBlock::Moe,
                _ => NemotronBlock::Mamba,
            })
            .collect();
        // one consistent kv width across the attention layers, else the file
        // is not the geometry this graph serves
        let kv_vals: Vec<usize> = kv_arr[..n_layer]
            .iter()
            .copied()
            .filter(|&v| v > 0)
            .collect();
        let n_kv_heads = *kv_vals
            .first()
            .ok_or_else(|| bad("nemotron gguf: no attention layers".into()))?;
        if kv_vals.iter().any(|&v| v != n_kv_heads) {
            return Err(bad(
                "nemotron gguf: attention.head_count_kv varies across attention layers".into(),
            ));
        }

        // grouped expert routing is a different router - refuse, never ignore
        if let Some(gc) = g.arch_field("expert_group_count").and_then(Value::as_u64)
            && gc > 1
        {
            return Err(bad(format!(
                "nemotron gguf: expert_group_count {gc} unsupported"
            )));
        }
        match g.arch_field("expert_weights_norm") {
            Some(Value::Bool(true)) => {}
            other => {
                return Err(bad(format!(
                    "nemotron gguf: expert_weights_norm must be true (got {other:?}) - the top-k renorm is baked into the router kernel"
                )));
            }
        }

        let head_dim = u("attention.key_length")?;
        if u("attention.value_length")? != head_dim {
            return Err(bad("nemotron gguf: key_length != value_length".into()));
        }

        let mamba_heads = u("ssm.time_step_rank")?; // llama.cpp: n_ssm_head
        let d_inner = u("ssm.inner_size")?;
        if mamba_heads == 0 || d_inner % mamba_heads != 0 {
            return Err(bad(format!(
                "nemotron gguf: ssm.inner_size {d_inner} not divisible by time_step_rank {mamba_heads}"
            )));
        }

        // eos SET: the gguf stamps only <|im_end|> (11); the HF
        // generation_config also lists </s> (2). Resolve both from the vocab
        // by string, never by magic number - decode must stop on either.
        let mut eos_ids: Vec<u32> = vec![
            g.metadata
                .get("tokenizer.ggml.eos_token_id")
                .and_then(Value::as_u64)
                .ok_or_else(|| miss("tokenizer.ggml.eos_token_id"))? as u32,
        ];
        if let Some(Value::Array(toks)) = g.metadata.get("tokenizer.ggml.tokens") {
            for (i, t) in toks.iter().enumerate() {
                if matches!(t.as_str(), Some("</s>") | Some("<|im_end|>"))
                    && !eos_ids.contains(&(i as u32))
                {
                    eos_ids.push(i as u32);
                }
            }
        }
        eos_ids.sort_unstable();

        Ok(Self {
            hidden: u("embedding_length")?,
            n_layer,
            blocks,
            vocab: u("vocab_size")?,
            max_pos: u("context_length")?,
            eps: f("attention.layer_norm_rms_epsilon")?,
            n_heads: u("attention.head_count")?,
            n_kv_heads,
            head_dim,
            mamba_heads,
            mamba_head_dim: d_inner / mamba_heads,
            d_state: u("ssm.state_size")?,
            d_conv: u("ssm.conv_kernel")?,
            n_groups: u("ssm.group_count")?,
            // not stamped in GGUF; informational only (nothing in the engine
            // consumes it - the scan kernels stride by row, not chunk)
            chunk: 128,
            n_expert: u("expert_count")?,
            n_active: u("expert_used_count")?,
            moe_ff: u("expert_feed_forward_length")?,
            shared_ff: u("expert_shared_feed_forward_length")?,
            routed_scale: f("expert_weights_scale")?,
            latent: None,
            omni: false,
            mtp_layers: nextn,
            eos_ids,
            bos_id: g
                .metadata
                .get("tokenizer.ggml.bos_token_id")
                .and_then(Value::as_u64)
                .unwrap_or(1) as u32,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn ckpt() -> Option<PathBuf> {
        let p = std::env::var("NEMOTRON_NVFP4_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from("/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4")
            });
        p.join("config.json").exists().then_some(p)
    }

    #[test]
    fn nemotron_config_parses_the_shipped_checkpoint() {
        let Some(dir) = ckpt() else {
            eprintln!("skip: no nemotron checkpoint present");
            return;
        };
        let c = NemotronConfig::read(&dir).expect("config parses");
        assert_eq!(c.hidden, 2688);
        assert_eq!(c.n_layer, 52);
        assert_eq!(c.d_inner(), 4096);
        assert_eq!(c.conv_dim(), 6144);
        assert_eq!(c.in_proj_rows(), 10304);
        assert_eq!(c.blocks.len(), 52);
        // the pinned interleave: 23 mamba / 23 moe / 6 attention at 5,12,19,26,33,42
        let attn: Vec<usize> = c
            .blocks
            .iter()
            .enumerate()
            .filter(|(_, b)| **b == NemotronBlock::Attention)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(attn, vec![5, 12, 19, 26, 33, 42]);
        assert_eq!(
            c.blocks
                .iter()
                .filter(|b| **b == NemotronBlock::Mamba)
                .count(),
            23
        );
        assert_eq!(
            c.blocks
                .iter()
                .filter(|b| **b == NemotronBlock::Moe)
                .count(),
            23
        );
        assert_eq!(c.eos_ids, vec![2, 11]);
        assert_eq!(
            (c.n_expert, c.n_active, c.moe_ff, c.shared_ff),
            (128, 6, 1856, 3712)
        );
        assert!((c.routed_scale - 2.5).abs() < 1e-6);
        assert_eq!((c.n_groups, c.d_state, c.chunk), (8, 128, 128));
    }

    /// The vision-language wrapper (Super VL): the language model's config
    /// arrives whole under `llm_config`, with no `num_hidden_layers` (the
    /// block list is the count) and a LatentMoE width.
    #[test]
    fn omni_wrapper_unwraps_the_language_model_config() {
        // Super VL's interleave as shipped (M mamba, E moe, * attention):
        // 40 / 40 / 8, attention at 7, 16, 25, 36, 47, 58, 69, 78
        let blocks: Vec<&str> = "MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME"
            .chars()
            .map(|c| match c {
                'M' => "mamba",
                'E' => "moe",
                _ => "attention",
            })
            .collect();
        let llm = serde_json::json!({
            "model_type": "nemotron_h", "hidden_size": 4096,
            "layers_block_type": blocks, "vocab_size": 131072,
            "max_position_embeddings": 1048576, "layer_norm_epsilon": 1e-5,
            "num_attention_heads": 32, "num_key_value_heads": 2, "head_dim": 128,
            "mamba_num_heads": 128, "mamba_head_dim": 64, "ssm_state_size": 128,
            "conv_kernel": 4, "n_groups": 8, "chunk_size": 128,
            "n_routed_experts": 512, "num_experts_per_tok": 22,
            "moe_intermediate_size": 2688, "moe_shared_expert_intermediate_size": 5376,
            "moe_latent_size": 1024, "routed_scaling_factor": 5.0,
            "num_nextn_predict_layers": 1, "eos_token_id": 11, "bos_token_id": 1
        });
        let top = serde_json::json!({"model_type": "nemotron_h_omni", "llm_config": llm});
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("config.json"), top.to_string()).expect("write");
        let c = NemotronConfig::read(dir.path()).expect("omni config parses");
        assert!(c.omni);
        assert_eq!((c.n_layer, c.hidden), (88, 4096));
        let attn: Vec<usize> = (0..c.n_layer)
            .filter(|&i| c.blocks[i] == NemotronBlock::Attention)
            .collect();
        assert_eq!(attn, vec![7, 16, 25, 36, 47, 58, 69, 78]);
        assert_eq!((c.n_expert, c.n_active, c.latent), (512, 22, Some(1024)));
        assert_eq!(c.expert_in(), 1024);
        assert_eq!(
            (c.mamba_heads, c.d_inner(), c.in_proj_rows()),
            (128, 8192, 18560)
        );
        assert_eq!(c.mtp_layers, 1);
        assert_eq!(c.eos_ids, vec![11]);
        // a plain nemotron_h config has no latent and no wrapper
        let dir = tempfile::tempdir().expect("tempdir");
        let mut flat = llm.clone();
        flat["moe_latent_size"] = serde_json::Value::Null;
        flat["num_hidden_layers"] = serde_json::json!(88);
        std::fs::write(dir.path().join("config.json"), flat.to_string()).expect("write");
        let c = NemotronConfig::read(dir.path()).expect("flat config parses");
        assert!(!c.omni && c.latent.is_none());
        assert_eq!(c.expert_in(), 4096);
    }

    /// The two official drafters' configs as NVIDIA ships them (quantization
    /// block elided): DFlash = full, non-causal, yarn; DSpark = causal,
    /// sliding window 1024 with sinks, plain rope, the rank-512 Markov head.
    #[test]
    fn drafter_configs_parse_both_official_checkpoints() {
        let parse = |json: &str| {
            let dir = tempfile::tempdir().expect("tempdir");
            std::fs::write(dir.path().join("config.json"), json).expect("write");
            NemotronDflashConfig::read(dir.path())
        };
        let df = parse(DFLASH_CONFIG).expect("dflash");
        assert!(!df.causal && df.window == 0 && !df.sinks && df.markov_rank.is_none());
        assert_eq!(
            (df.rope_theta, df.rope_factor, df.rope_orig),
            (10000.0, 128.0, 8192)
        );
        let ds = parse(DSPARK_CONFIG).expect("dspark");
        assert!(ds.causal && ds.sinks);
        assert_eq!(
            (ds.window, ds.markov_rank, ds.block_size),
            (1024, Some(512), Some(8))
        );
        assert_eq!((ds.rope_theta, ds.rope_factor), (10000.0, 1.0));
        for c in [&df, &ds] {
            assert_eq!(c.target_layers, vec![1, 5, 19, 29, 41, 51]);
            assert_eq!(
                (c.n_layers, c.hidden, c.n_heads, c.n_kv_heads),
                (6, 2688, 32, 2)
            );
            assert_eq!(c.mask_token, 990);
        }
        // the anchor-as-first layout is another model
        let anchor = DSPARK_CONFIG.replace(
            "\"sample_from_anchor\": false",
            "\"sample_from_anchor\": true",
        );
        assert!(parse(&anchor).is_err());
    }

    const DFLASH_CONFIG: &str = r#"{
 "architectures": [
  "DFlashDraftModel"
 ],
 "attention_bias": false,
 "attention_sink_bias": false,
 "dflash_config": {
  "attention_sink_bias": false,
  "causal": false,
  "mask_token_id": 990,
  "target_layer_ids": [
   1,
   5,
   19,
   29,
   41,
   51
  ],
  "use_swa": false,
  "sample_from_anchor": false
 },
 "dflash_query_causal": false,
 "draft_vocab_size": null,
 "dspark_fixup_head_type": null,
 "dspark_markov_rank": null,
 "dtype": "bfloat16",
 "eagle_aux_hidden_state_layer_ids": [
  2,
  6,
  20,
  30,
  42,
  52
 ],
 "has_embed_tokens": true,
 "has_lm_head": false,
 "head_dim": 128,
 "hidden_act": "silu",
 "hidden_size": 2688,
 "intermediate_size": 6144,
 "layer_types": [
  "full_attention",
  "full_attention",
  "full_attention",
  "full_attention",
  "full_attention",
  "full_attention"
 ],
 "mask_token_id": 990,
 "max_position_embeddings": 1048576,
 "model_type": "qwen3",
 "num_attention_heads": 32,
 "num_aux_layers": 6,
 "num_hidden_layers": 6,
 "num_key_value_heads": 2,
 "pard_token": 990,
 "rms_norm_eps": 1e-06,
 "rope_parameters": {
  "factor": 128.0,
  "original_max_position_embeddings": 8192,
  "rope_theta": 10000,
  "rope_type": "yarn"
 },
 "rope_theta": 10000,
 "sliding_window": null,
 "target_layer_ids": [
  1,
  5,
  19,
  29,
  41,
  51
 ],
 "train_embed_tokens": false,
 "transformers_version": "5.5.3",
 "use_cache": false,
 "vocab_size": 131072,
 "sample_from_anchor": false
}"#;
    const DSPARK_CONFIG: &str = r#"{
 "architectures": [
  "Qwen3DSparkModel"
 ],
 "attention_bias": false,
 "attention_sink_bias": true,
 "block_size": 8,
 "dflash_config": {
  "attention_sink_bias": true,
  "causal": true,
  "mask_token_id": 990,
  "swa_window_size": 1024,
  "target_layer_ids": [
   1,
   5,
   19,
   29,
   41,
   51
  ],
  "use_swa": true,
  "sample_from_anchor": false
 },
 "dflash_query_causal": true,
 "draft_vocab_size": null,
 "dspark_bonus_anchor": true,
 "dspark_fixup_head_type": "markov",
 "dspark_markov_rank": 512,
 "dtype": "bfloat16",
 "eagle_aux_hidden_state_layer_ids": [
  2,
  6,
  20,
  30,
  42,
  52
 ],
 "has_embed_tokens": true,
 "has_lm_head": false,
 "head_dim": 128,
 "hidden_act": "silu",
 "hidden_size": 2688,
 "intermediate_size": 6144,
 "layer_types": [
  "sliding_attention",
  "sliding_attention",
  "sliding_attention",
  "sliding_attention",
  "sliding_attention",
  "sliding_attention"
 ],
 "markov_head_type": "vanilla",
 "markov_rank": 512,
 "mask_token_id": 990,
 "max_position_embeddings": 1048576,
 "model_type": "qwen3",
 "num_attention_heads": 32,
 "num_aux_layers": 6,
 "num_hidden_layers": 6,
 "num_key_value_heads": 2,
 "pard_token": 990,
 "rms_norm_eps": 1e-06,
 "rope_parameters": null,
 "rope_theta": 10000,
 "sliding_window": 1024,
 "target_layer_ids": [
  1,
  5,
  19,
  29,
  41,
  51
 ],
 "train_embed_tokens": false,
 "transformers_version": "5.5.3",
 "use_cache": false,
 "vocab_size": 131072,
 "sample_from_anchor": false
}"#;
}

/// The official block drafters' config (C2): `nvidia/...-NVFP4-DFlash` - a
/// 6-layer qwen3-class dense GQA drafter with yarn rope and NVFP4 MLPs - and
/// its DSpark sibling `...-NVFP4-DSpark` (`Qwen3DSparkModel`): the same
/// backbone and taps with a CAUSAL block, sliding-window attention with
/// per-head sinks, plain rope, and a low-rank Markov head that biases each
/// draft by the token drafted before it. Every consumed key is validated
/// present.
#[derive(Debug, Clone)]
pub struct NemotronDflashConfig {
    pub n_layers: usize,
    pub hidden: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub inter: usize,
    pub eps: f32,
    pub mask_token: u32,
    pub target_layers: Vec<usize>,
    /// yarn: (theta, factor, original_max_position_embeddings); plain rope
    /// reads as factor 1 (yarn's interpolation and mscale are then identity)
    pub rope_theta: f32,
    pub rope_factor: f32,
    pub rope_orig: usize,
    /// the query block attends causally (DSpark) or as a whole (DFlash)
    pub causal: bool,
    /// sliding-window width of every layer's attention, 0 = full
    pub window: usize,
    /// per-head attention sink logits ship with each layer
    pub sinks: bool,
    /// the Markov fixup head's rank (DSpark), None = drafts are the base
    /// logits' argmaxes
    pub markov_rank: Option<usize>,
    /// the trained block (anchor + masks), when the config states it
    pub block_size: Option<usize>,
}

impl NemotronDflashConfig {
    pub fn read(dir: &Path) -> Result<Self, StError> {
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)
            .map_err(|e| StError::Header(e.to_string()))?;
        let miss = |k: &str| StError::Header(format!("dflash config: missing {k}"));
        let u = |k: &str| {
            v.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| miss(k))
        };
        let arch = v["architectures"][0].as_str();
        let dspark = match arch {
            Some("DFlashDraftModel") => false,
            Some("Qwen3DSparkModel") => true,
            _ => {
                return Err(StError::Header(format!(
                    "not a DFlash/DSpark drafter checkpoint (architectures {:?})",
                    v.get("architectures")
                )));
            }
        };
        let dc = &v["dflash_config"];
        // the bonus-anchor layout: row 0 is the committed token, mask row j
        // predicts the token AT its own position. The paper's anchor-as-first
        // layout (row j-1 predicts position j) is another model.
        if dc["sample_from_anchor"].as_bool().unwrap_or(false)
            || v["sample_from_anchor"].as_bool().unwrap_or(false)
        {
            return Err(StError::Header(
                "drafter samples from its anchor row - only the bonus-anchor layout is served"
                    .into(),
            ));
        }
        let rp = &v["rope_parameters"];
        let (rope_theta, rope_factor, rope_orig) = if rp.is_null() {
            // plain rope (DSpark ships rope_parameters null + rope_theta)
            let theta = v["rope_theta"].as_f64().ok_or_else(|| miss("rope_theta"))? as f32;
            (theta, 1.0, u("max_position_embeddings")?)
        } else {
            match rp["rope_type"].as_str() {
                Some("yarn") => (
                    rp["rope_theta"]
                        .as_f64()
                        .ok_or_else(|| miss("rope_theta"))? as f32,
                    rp["factor"].as_f64().ok_or_else(|| miss("rope factor"))? as f32,
                    rp["original_max_position_embeddings"]
                        .as_u64()
                        .ok_or_else(|| miss("rope original_max_position_embeddings"))?
                        as usize,
                ),
                Some("default") => (
                    rp["rope_theta"]
                        .as_f64()
                        .ok_or_else(|| miss("rope_theta"))? as f32,
                    1.0,
                    u("max_position_embeddings")?,
                ),
                t => {
                    return Err(StError::Header(format!(
                        "drafter rope_type {t:?} unsupported (expected yarn or default)"
                    )));
                }
            }
        };
        let causal = dc["causal"].as_bool().unwrap_or(false);
        let window = if dc["use_swa"].as_bool().unwrap_or(false) {
            dc["swa_window_size"]
                .as_u64()
                .or_else(|| v["sliding_window"].as_u64())
                .ok_or_else(|| miss("dflash_config.swa_window_size"))? as usize
        } else {
            0
        };
        let sinks = dc["attention_sink_bias"].as_bool().unwrap_or(false);
        let markov_rank = if dspark {
            if v["dspark_fixup_head_type"].as_str() != Some("markov")
                || v["markov_head_type"]
                    .as_str()
                    .is_some_and(|t| t != "vanilla")
            {
                return Err(StError::Header(format!(
                    "dspark fixup head {:?}/{:?} unsupported (expected the vanilla markov head)",
                    v.get("dspark_fixup_head_type"),
                    v.get("markov_head_type")
                )));
            }
            Some(
                v["dspark_markov_rank"]
                    .as_u64()
                    .or_else(|| v["markov_rank"].as_u64())
                    .ok_or_else(|| miss("dspark_markov_rank"))? as usize,
            )
        } else {
            None
        };
        Ok(Self {
            n_layers: u("num_hidden_layers")?,
            hidden: u("hidden_size")?,
            n_heads: u("num_attention_heads")?,
            n_kv_heads: u("num_key_value_heads")?,
            head_dim: u("head_dim")?,
            inter: u("intermediate_size")?,
            eps: v
                .get("rms_norm_eps")
                .and_then(|x| x.as_f64())
                .ok_or_else(|| miss("rms_norm_eps"))? as f32,
            mask_token: v["dflash_config"]["mask_token_id"]
                .as_u64()
                .ok_or_else(|| miss("dflash_config.mask_token_id"))? as u32,
            target_layers: v["dflash_config"]["target_layer_ids"]
                .as_array()
                .ok_or_else(|| miss("dflash_config.target_layer_ids"))?
                .iter()
                .map(|x| {
                    x.as_u64()
                        .map(|n| n as usize)
                        .ok_or_else(|| miss("target_layer_ids entry"))
                })
                .collect::<Result<_, _>>()?,
            rope_theta,
            rope_factor,
            rope_orig,
            causal,
            window,
            sinks,
            markov_rank,
            block_size: v["block_size"].as_u64().map(|b| b as usize),
        })
    }
}
