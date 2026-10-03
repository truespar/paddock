//! Clef (Cloudflare's decision models, `Cloudflare/clef-flash`): a Qwen3.5
//! backbone (`Qwen3_5ForConditionalGeneration`, BF16 sharded safetensors)
//! under a joint schema head (`joint_head.safetensors` +
//! `joint_head_config.json`). This reads and validates both configs; the
//! engine reads the tensors.
//!
//! The vision tower (`vision_config`, `model.visual.*` in the same shards)
//! and the image processor (`processor_config.json`) are read too: images
//! are a capability of a checkpoint that ships both, validated here against
//! what the engine's image lane implements.

use std::path::Path;

/// Files a Clef checkpoint directory holds beside its shards.
pub const HEAD_CONFIG: &str = "joint_head_config.json";
pub const HEAD_WEIGHTS: &str = "joint_head.safetensors";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClefBlock {
    /// Gated DeltaNet (`linear_attention`)
    Gdn,
    /// gated full attention (`full_attention`)
    Attention,
}

/// The joint schema head (`JointSchemaHead(**joint_head_config)`).
#[derive(Clone, Debug, PartialEq)]
pub struct ClefHeadConfig {
    pub width: usize,
    pub routing_layers: usize,
    pub layers: usize,
    pub heads: usize,
    pub feedforward: usize,
}

/// The Qwen3.5 vision tower (`vision_config`).
#[derive(Clone, Debug, PartialEq)]
pub struct ClefVisionConfig {
    pub depth: usize,
    pub hidden: usize,
    pub heads: usize,
    pub ffn: usize,
    /// the learned position grid's side (`sqrt(num_position_embeddings)`)
    pub pos_side: usize,
    /// the merger's output width (the backbone's hidden)
    pub out_hidden: usize,
    pub patch: usize,
    pub temporal_patch: usize,
    pub merge: usize,
    /// `<|image_pad|>`, `<|vision_start|>`, `<|vision_end|>`
    pub image_token: u32,
    pub vision_start: u32,
    pub vision_end: u32,
}

impl ClefVisionConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden / self.heads
    }
}

/// The image processor (`processor_config.json` `image_processor`): a
/// Qwen2-VL processor's pixel budget, the rest pinned to what the lane
/// implements (bicubic, rescale 1/255, mean = std = 0.5, RGB).
#[derive(Clone, Debug, PartialEq)]
pub struct ClefImageConfig {
    /// `size.shortest_edge` / `size.longest_edge`: the pixel-count bounds
    /// smart_resize keeps an image within
    pub min_pixels: u64,
    pub max_pixels: u64,
}

/// Qwen2-VL's `smart_resize`, as Transformers computes it in Python floats:
/// both sides to the nearest multiple of `factor` (ties to even), then the
/// pixel count brought inside `[min_pixels, max_pixels]` by one common
/// scale - floor when shrinking, ceil when growing. Errors as the reference
/// does on an aspect ratio past 200.
pub fn smart_resize(
    height: u64,
    width: u64,
    factor: u64,
    min_pixels: u64,
    max_pixels: u64,
) -> Result<(u64, u64), String> {
    if height == 0 || width == 0 || factor == 0 {
        return Err("image has no pixels".into());
    }
    let (h, w, f) = (height as f64, width as f64, factor as f64);
    if h.max(w) / h.min(w) > 200.0 {
        return Err(format!(
            "absolute aspect ratio must be smaller than 200, got {}",
            h.max(w) / h.min(w)
        ));
    }
    let mut hb = (h / f).round_ties_even() as u64 * factor;
    let mut wb = (w / f).round_ties_even() as u64 * factor;
    if hb * wb > max_pixels {
        let beta = (h * w / max_pixels as f64).sqrt();
        hb = factor.max((h / beta / f).floor() as u64 * factor);
        wb = factor.max((w / beta / f).floor() as u64 * factor);
    } else if hb * wb < min_pixels {
        let beta = (min_pixels as f64 / (h * w)).sqrt();
        hb = (h * beta / f).ceil() as u64 * factor;
        wb = (w * beta / f).ceil() as u64 * factor;
    }
    Ok((hb, wb))
}

#[derive(Clone, Debug, PartialEq)]
pub struct ClefConfig {
    pub hidden: usize,
    pub blocks: Vec<ClefBlock>,
    pub vocab: usize,
    pub ffn: usize,
    pub eps: f32,
    // full attention
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// rotated dims per head (`partial_rotary_factor * head_dim`)
    pub n_rot: usize,
    pub rope_theta: f32,
    /// rotary pairs per position axis (t, h, w); interleaved in the model
    pub mrope_sections: [u32; 3],
    // Gated DeltaNet
    pub gdn_k_heads: usize,
    pub gdn_v_heads: usize,
    pub gdn_k_dim: usize,
    pub gdn_v_dim: usize,
    pub gdn_conv: usize,
    pub head: ClefHeadConfig,
    /// the vision tower and its processor - None on a checkpoint without
    /// them (images are then refused)
    pub vision: Option<(ClefVisionConfig, ClefImageConfig)>,
}

impl ClefConfig {
    /// in_proj_qkv rows: q | k | v.
    pub fn gdn_qkv_rows(&self) -> usize {
        2 * self.gdn_k_heads * self.gdn_k_dim + self.gdn_v_heads * self.gdn_v_dim
    }
    /// z (output gate) rows = the value plane's width.
    pub fn gdn_v_width(&self) -> usize {
        self.gdn_v_heads * self.gdn_v_dim
    }
    /// q_proj rows: per head a query then its output gate.
    pub fn attn_q_rows(&self) -> usize {
        2 * self.n_heads * self.head_dim
    }
    pub fn q_width(&self) -> usize {
        self.n_heads * self.head_dim
    }
    pub fn kv_width(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    /// Whether `dir` looks like a Clef checkpoint (the head's two files beside
    /// a sharded backbone). Cheap - no parsing.
    pub fn is_clef_dir(dir: &Path) -> bool {
        dir.join(HEAD_CONFIG).is_file()
            && dir.join(HEAD_WEIGHTS).is_file()
            && dir.join("config.json").is_file()
            && dir.join("model.safetensors.index.json").is_file()
    }

    pub fn read(dir: &Path) -> Result<Self, String> {
        let json = |name: &str| -> Result<serde_json::Value, String> {
            let bytes = std::fs::read(dir.join(name)).map_err(|e| format!("{name}: {e}"))?;
            serde_json::from_slice(&bytes).map_err(|e| format!("{name}: {e}"))
        };
        let v = json("config.json")?;
        let arch_ok = v
            .get("architectures")
            .and_then(|a| a.as_array())
            .is_some_and(|a| {
                a.iter()
                    .any(|x| x.as_str() == Some("Qwen3_5ForConditionalGeneration"))
            });
        if !arch_ok || v.get("model_type").and_then(|x| x.as_str()) != Some("qwen3_5") {
            return Err(
                "config.json: not a Qwen3_5ForConditionalGeneration (model_type qwen3_5) backbone"
                    .into(),
            );
        }
        let tc = v.get("text_config").ok_or("config.json: no text_config")?;
        let miss = |k: &str| format!("config.json text_config: missing or invalid {k}");
        let u = |k: &str| {
            tc.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| miss(k))
        };
        let b = |k: &str| tc.get(k).and_then(|x| x.as_bool());
        let n_layer = u("num_hidden_layers")?;
        let blocks = tc
            .get("layer_types")
            .and_then(|x| x.as_array())
            .ok_or_else(|| miss("layer_types"))?
            .iter()
            .map(|t| match t.as_str() {
                Some("linear_attention") => Ok(ClefBlock::Gdn),
                Some("full_attention") => Ok(ClefBlock::Attention),
                other => Err(format!("config.json: unknown layer type {other:?}")),
            })
            .collect::<Result<Vec<_>, _>>()?;
        if blocks.len() != n_layer {
            return Err(format!(
                "config.json: layer_types has {} entries for {n_layer} layers",
                blocks.len()
            ));
        }
        if b("attn_output_gate") != Some(true) {
            return Err("config.json: the Clef lane reads a gated-output attention".into());
        }
        if b("tie_word_embeddings") == Some(true)
            || v.get("tie_word_embeddings").and_then(|x| x.as_bool()) == Some(true)
        {
            return Err("config.json: tied embeddings - the head reads its own lm_head".into());
        }
        if tc.get("hidden_act").and_then(|x| x.as_str()) != Some("silu") {
            return Err("config.json: hidden_act must be silu".into());
        }
        let rp = tc
            .get("rope_parameters")
            .ok_or_else(|| miss("rope_parameters"))?;
        let rope_theta = rp
            .get("rope_theta")
            .and_then(|x| x.as_f64())
            .ok_or_else(|| miss("rope_parameters.rope_theta"))? as f32;
        if rp
            .get("rope_type")
            .and_then(|x| x.as_str())
            .unwrap_or("default")
            != "default"
        {
            return Err("config.json: only the default rope type is read".into());
        }
        if rp.get("mrope_interleaved").and_then(|x| x.as_bool()) != Some(true) {
            return Err("config.json: Qwen3.5 rotates interleaved mrope sections".into());
        }
        let sec: Vec<u32> = rp
            .get("mrope_section")
            .and_then(|x| x.as_array())
            .ok_or_else(|| miss("rope_parameters.mrope_section"))?
            .iter()
            .map(|x| x.as_u64().map(|v| v as u32))
            .collect::<Option<_>>()
            .ok_or_else(|| miss("rope_parameters.mrope_section"))?;
        let head_dim = u("head_dim")?;
        let partial = rp
            .get("partial_rotary_factor")
            .or_else(|| tc.get("partial_rotary_factor"))
            .and_then(|x| x.as_f64())
            .ok_or_else(|| miss("partial_rotary_factor"))?;
        let n_rot = (head_dim as f64 * partial).round() as usize;
        if sec.len() != 3 || sec.iter().sum::<u32>() as usize * 2 != n_rot {
            return Err(format!(
                "config.json: mrope_section {sec:?} does not cover the {n_rot} rotated dims"
            ));
        }
        let eps = tc
            .get("rms_norm_eps")
            .and_then(|x| x.as_f64())
            .ok_or_else(|| miss("rms_norm_eps"))? as f32;

        let hv = json(HEAD_CONFIG)?;
        let hu = |k: &str| {
            hv.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| format!("{HEAD_CONFIG}: missing or invalid {k}"))
        };
        let head = ClefHeadConfig {
            width: hu("width")?,
            routing_layers: hu("routing_layers")?,
            layers: hu("layers")?,
            heads: hu("heads")?,
            feedforward: hu("feedforward")?,
        };
        let mut cfg = ClefConfig {
            hidden: u("hidden_size")?,
            blocks,
            vocab: u("vocab_size")?,
            ffn: u("intermediate_size")?,
            eps,
            n_heads: u("num_attention_heads")?,
            n_kv_heads: u("num_key_value_heads")?,
            head_dim,
            n_rot,
            rope_theta,
            mrope_sections: [sec[0], sec[1], sec[2]],
            gdn_k_heads: u("linear_num_key_heads")?,
            gdn_v_heads: u("linear_num_value_heads")?,
            gdn_k_dim: u("linear_key_head_dim")?,
            gdn_v_dim: u("linear_value_head_dim")?,
            gdn_conv: u("linear_conv_kernel_dim")?,
            head,
            vision: None,
        };
        if v.get("vision_config").is_some() && dir.join("processor_config.json").is_file() {
            cfg.vision = Some((
                read_vision(&v, cfg.hidden)?,
                read_image_processor(&json("processor_config.json")?)?,
            ));
        }
        if hu("hidden_size")? != cfg.hidden {
            return Err(format!(
                "{HEAD_CONFIG}: hidden_size {} is not the backbone's {}",
                hu("hidden_size")?,
                cfg.hidden
            ));
        }
        if cfg.head.heads == 0 || !cfg.head.width.is_multiple_of(cfg.head.heads) {
            return Err(format!("{HEAD_CONFIG}: width is not a multiple of heads"));
        }
        if cfg.n_kv_heads == 0
            || !cfg.n_heads.is_multiple_of(cfg.n_kv_heads)
            || cfg.gdn_k_heads == 0
            || !cfg.gdn_v_heads.is_multiple_of(cfg.gdn_k_heads)
        {
            return Err("config.json: head counts do not group".into());
        }
        Ok(cfg)
    }
}

/// The vision companion a Clef GGUF is served with (ggml-org's conversions
/// carry the backbone and head only): the official checkpoint's
/// `model.visual.*` tensors byte for byte in one safetensors file, with the
/// checkpoint's `config.json` and `processor_config.json` verbatim in its
/// header metadata under these keys, and where it was cut from.
pub const COMPANION_CONFIG: &str = "clef.config.json";
pub const COMPANION_PROCESSOR: &str = "clef.processor_config.json";
pub const COMPANION_SOURCE: &str = "clef.source";

/// A tensor's dims as stored (`ne[0]` first).
fn gguf_dims<'a>(g: &'a crate::gguf::GgufFile, name: &str) -> Result<&'a [u64], String> {
    g.tensors
        .iter()
        .find(|t| t.name == name)
        .map(|t| t.dims.as_slice())
        .ok_or_else(|| format!("GGUF: tensor {name} missing"))
}

impl ClefConfig {
    /// Whether `g` is a Clef GGUF (`general.architecture = "clef"`, the
    /// schema llama.cpp's converter writes).
    pub fn is_clef_gguf(g: &crate::gguf::GgufFile) -> bool {
        g.architecture() == Some("clef")
    }

    /// Read a Clef GGUF's backbone and head from its metadata (llama.cpp's
    /// `clef` schema, `conversion/clef.py`), and the vision tower and image
    /// processor from the companion's metadata when one is given - checked
    /// against the GGUF, so a companion cut from another Clef is refused.
    pub fn from_gguf(
        g: &crate::gguf::GgufFile,
        companion: Option<&std::collections::HashMap<String, String>>,
    ) -> Result<Self, String> {
        if !Self::is_clef_gguf(g) {
            return Err("GGUF: not a Clef model (general.architecture is not clef)".into());
        }
        let field = |k: &str| {
            g.arch_field(k)
                .ok_or_else(|| format!("GGUF: clef.{k} missing"))
        };
        let u = |k: &str| {
            field(k)?
                .as_u64()
                .map(|x| x as usize)
                .ok_or_else(|| format!("GGUF: clef.{k} is not an integer"))
        };
        let f = |k: &str| {
            field(k)?
                .as_f32()
                .ok_or_else(|| format!("GGUF: clef.{k} is not a float"))
        };
        let arr = |k: &str| match field(k)? {
            crate::gguf::Value::Array(a) => Ok(a.as_slice()),
            _ => Err(format!("GGUF: clef.{k} is not an array")),
        };
        if field("decision.type")?.as_str() != Some("clef") {
            return Err("GGUF: clef.decision.type is not clef".into());
        }
        let blocks = arr("attention.recurrent_layers")?
            .iter()
            .map(|b| match b {
                crate::gguf::Value::Bool(true) => Ok(ClefBlock::Gdn),
                crate::gguf::Value::Bool(false) => Ok(ClefBlock::Attention),
                _ => Err("GGUF: clef.attention.recurrent_layers holds a non-bool".to_string()),
            })
            .collect::<Result<Vec<_>, _>>()?;
        if blocks.len() != u("block_count")? {
            return Err(format!(
                "GGUF: {} recurrent-layer flags for {} blocks",
                blocks.len(),
                u("block_count")?
            ));
        }
        let sec: Vec<u32> = arr("rope.dimension_sections")?
            .iter()
            .map(|x| x.as_u64().map(|v| v as u32))
            .collect::<Option<_>>()
            .ok_or("GGUF: clef.rope.dimension_sections is not integers")?;
        let n_rot = u("rope.dimension_count")?;
        // [t, h, w, 0]: the converter pads Qwen3.5's three sections to four
        if sec.len() != 4 || sec[3] != 0 || sec[..3].iter().sum::<u32>() as usize * 2 != n_rot {
            return Err(format!(
                "GGUF: rope sections {sec:?} do not cover the {n_rot} rotated dims"
            ));
        }
        let head_dim = u("attention.key_length")?;
        if u("attention.value_length")? != head_dim {
            return Err("GGUF: key and value head widths differ".into());
        }
        let (v_heads, inner) = (u("ssm.time_step_rank")?, u("ssm.inner_size")?);
        if v_heads == 0 || !inner.is_multiple_of(v_heads) {
            return Err("GGUF: clef.ssm.inner_size is not a whole number of value heads".into());
        }
        // the head's LayerNorms are torch's default (1e-5), the one eps the
        // engine's head carries
        if (f("attention.layer_norm_epsilon")? - 1e-5).abs() > 1e-12 {
            return Err("GGUF: the head's LayerNorm eps is not torch's default 1e-5".into());
        }
        let width = gguf_dims(g, "decision.proj_memory.weight")?[1] as usize;
        let head = ClefHeadConfig {
            width,
            routing_layers: u("decision.routing_block_count")?,
            layers: u("decision.block_count")?,
            heads: u("decision.head_count")?,
            feedforward: gguf_dims(g, "dec.blk.0.ffn_up.weight")?[1] as usize,
        };
        let hidden = u("embedding_length")?;
        let mut cfg = ClefConfig {
            hidden,
            blocks,
            vocab: gguf_dims(g, "token_embd.weight")?[1] as usize,
            ffn: u("feed_forward_length")?,
            eps: f("attention.layer_norm_rms_epsilon")?,
            n_heads: u("attention.head_count")?,
            n_kv_heads: u("attention.head_count_kv")?,
            head_dim,
            n_rot,
            rope_theta: f("rope.freq_base")?,
            mrope_sections: [sec[0], sec[1], sec[2]],
            gdn_k_heads: u("ssm.group_count")?,
            gdn_v_heads: v_heads,
            gdn_k_dim: u("ssm.state_size")?,
            gdn_v_dim: inner / v_heads,
            gdn_conv: u("ssm.conv_kernel")?,
            head,
            vision: None,
        };
        if let Some(meta) = companion {
            let json = |k: &str| -> Result<serde_json::Value, String> {
                let text = meta
                    .get(k)
                    .ok_or_else(|| format!("vision companion: no {k} in its metadata"))?;
                serde_json::from_str(text).map_err(|e| format!("vision companion {k}: {e}"))
            };
            let v = json(COMPANION_CONFIG)?;
            // the companion's backbone must be this GGUF's
            let tc = &v["text_config"];
            let tu = |k: &str| tc.get(k).and_then(|x| x.as_u64()).map(|x| x as usize);
            if tu("hidden_size") != Some(cfg.hidden)
                || tu("num_hidden_layers") != Some(cfg.blocks.len())
                || tu("vocab_size") != Some(cfg.vocab)
                || tu("intermediate_size") != Some(cfg.ffn)
            {
                return Err(
                    "vision companion: cut from a different Clef than this GGUF (its backbone's \
                     hidden / layers / vocab / ffn differ)"
                        .into(),
                );
            }
            cfg.vision = Some((
                read_vision(&v, cfg.hidden)?,
                read_image_processor(&json(COMPANION_PROCESSOR)?)?,
            ));
        }
        cfg.check_groups()?;
        Ok(cfg)
    }

    fn check_groups(&self) -> Result<(), String> {
        if self.head.heads == 0 || !self.head.width.is_multiple_of(self.head.heads) {
            return Err("the head's width is not a multiple of its heads".into());
        }
        if self.n_kv_heads == 0
            || !self.n_heads.is_multiple_of(self.n_kv_heads)
            || self.gdn_k_heads == 0
            || !self.gdn_v_heads.is_multiple_of(self.gdn_k_heads)
        {
            return Err("head counts do not group".into());
        }
        Ok(())
    }
}

fn read_vision(v: &serde_json::Value, text_hidden: usize) -> Result<ClefVisionConfig, String> {
    let vc = &v["vision_config"];
    let miss = |k: &str| format!("config.json vision_config: missing or invalid {k}");
    let u = |k: &str| {
        vc.get(k)
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .ok_or_else(|| miss(k))
    };
    let id = |k: &str| {
        v.get(k)
            .and_then(|x| x.as_u64())
            .map(|x| x as u32)
            .ok_or_else(|| format!("config.json: missing or invalid {k}"))
    };
    if vc.get("hidden_act").and_then(|x| x.as_str()) != Some("gelu_pytorch_tanh") {
        return Err("config.json vision_config: hidden_act must be gelu_pytorch_tanh".into());
    }
    if vc
        .get("deepstack_visual_indexes")
        .and_then(|x| x.as_array())
        .is_some_and(|a| !a.is_empty())
    {
        return Err("config.json vision_config: deepstack taps are not read by this lane".into());
    }
    let n_pos = u("num_position_embeddings")?;
    let side = (n_pos as f64).sqrt() as usize;
    let c = ClefVisionConfig {
        depth: u("depth")?,
        hidden: u("hidden_size")?,
        heads: u("num_heads")?,
        ffn: u("intermediate_size")?,
        pos_side: side,
        out_hidden: u("out_hidden_size")?,
        patch: u("patch_size")?,
        temporal_patch: u("temporal_patch_size")?,
        merge: u("spatial_merge_size")?,
        image_token: id("image_token_id")?,
        vision_start: id("vision_start_token_id")?,
        vision_end: id("vision_end_token_id")?,
    };
    // the kernels' geometry: 72-wide heads, 16 x 16 patches of 3 channels
    // over 2 frames, 2 x 2 merge windows, a square position grid
    if side * side != n_pos
        || c.heads == 0
        || c.hidden != c.heads * 72
        || c.patch != 16
        || c.temporal_patch != 2
        || c.merge != 2
        || u("in_channels")? != 3
        || c.out_hidden != text_hidden
    {
        return Err(format!(
            "config.json vision_config: geometry the image lane does not implement \
             (hidden {} over {} heads, patch {}, temporal {}, merge {}, {} positions, out {})",
            c.hidden, c.heads, c.patch, c.temporal_patch, c.merge, n_pos, c.out_hidden
        ));
    }
    Ok(c)
}

fn read_image_processor(p: &serde_json::Value) -> Result<ClefImageConfig, String> {
    let ip = &p["image_processor"];
    let bad = |what: &str| format!("processor_config.json image_processor: {what}");
    let f = |k: &str| ip.get(k).and_then(|x| x.as_f64());
    let b = |k: &str| ip.get(k).and_then(|x| x.as_bool());
    let three = |k: &str, want: f64| {
        ip.get(k)
            .and_then(|x| x.as_array())
            .is_some_and(|a| a.len() == 3 && a.iter().all(|x| x.as_f64() == Some(want)))
    };
    if ip.get("image_processor_type").and_then(|x| x.as_str()) != Some("Qwen2VLImageProcessor") {
        return Err(bad("not a Qwen2VLImageProcessor"));
    }
    // the lane's pixels: bicubic resize, (x / 255 - 0.5) / 0.5, RGB
    if f("resample") != Some(3.0)
        || b("do_resize") == Some(false)
        || b("do_rescale") == Some(false)
        || b("do_normalize") == Some(false)
        || b("do_convert_rgb") == Some(false)
        || f("rescale_factor").is_none_or(|r| (r * 255.0 - 1.0).abs() > 1e-12)
        || !three("image_mean", 0.5)
        || !three("image_std", 0.5)
        || f("patch_size") != Some(16.0)
        || f("merge_size") != Some(2.0)
        || f("temporal_patch_size") != Some(2.0)
    {
        return Err(bad(
            "preprocessing the image lane does not implement (it reads bicubic, \
             rescale 1/255, mean = std = 0.5, patch 16, merge 2, temporal 2)",
        ));
    }
    let size = &ip["size"];
    let px = |k: &str| {
        size.get(k)
            .and_then(|x| x.as_u64())
            .ok_or_else(|| bad(&format!("size.{k} missing")))
    };
    Ok(ClefImageConfig {
        min_pixels: px("shortest_edge")?,
        max_pixels: px("longest_edge")?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reads the real checkpoint when `CLEF_DIR` points at one.
    #[test]
    fn reads_the_release_config() {
        let Some(dir) = std::env::var_os("CLEF_DIR").map(std::path::PathBuf::from) else {
            eprintln!("SKIP: CLEF_DIR is not set");
            return;
        };
        assert!(ClefConfig::is_clef_dir(&dir));
        let c = ClefConfig::read(&dir).expect("config");
        assert_eq!(c.hidden, 4096);
        assert_eq!(c.blocks.len(), 32);
        assert_eq!(
            c.blocks
                .iter()
                .filter(|b| **b == ClefBlock::Attention)
                .count(),
            8
        );
        assert_eq!(
            (c.n_heads, c.n_kv_heads, c.head_dim, c.n_rot),
            (16, 4, 256, 64)
        );
        assert_eq!(c.gdn_qkv_rows(), 8192);
        assert_eq!(c.head.width, 1024);
        let (v, ip) = c.vision.expect("the release ships its vision tower");
        assert_eq!((v.depth, v.hidden, v.heads, v.ffn), (27, 1152, 16, 4304));
        assert_eq!((v.pos_side, v.out_hidden, v.head_dim()), (48, 4096, 72));
        assert_eq!(
            (v.image_token, v.vision_start, v.vision_end),
            (248056, 248053, 248054)
        );
        assert_eq!((ip.min_pixels, ip.max_pixels), (65536, 16777216));
    }

    /// A Clef GGUF (`CLEF_GGUF`, ggml-org's Flash or 27B) reads to the
    /// release's own config: the 9B's or the 27B's dims, the head, the
    /// tiling the converter wrote. With `CLEF_COMPANION` its tower and
    /// processor come from the companion's header, and a companion cut from
    /// the OTHER Clef (`CLEF_OTHER_COMPANION`) is refused.
    #[test]
    fn reads_a_clef_gguf_and_its_companion() {
        let Some(path) = std::env::var_os("CLEF_GGUF").map(std::path::PathBuf::from) else {
            eprintln!("SKIP: CLEF_GGUF is not set");
            return;
        };
        let g = crate::mapped::MappedGguf::open(&path).expect("the GGUF");
        assert!(ClefConfig::is_clef_gguf(g.gguf()));
        let c = ClefConfig::from_gguf(g.gguf(), None).expect("config");
        assert!(c.vision.is_none());
        let flash = c.hidden == 4096;
        let (layers, attn, heads, ffn, vh) = if flash {
            (32, 8, 16, 12288, 32)
        } else {
            (64, 16, 24, 17408, 48)
        };
        assert_eq!(c.blocks.len(), layers);
        assert_eq!(
            c.blocks
                .iter()
                .filter(|b| **b == ClefBlock::Attention)
                .count(),
            attn
        );
        assert_eq!(
            (c.n_heads, c.n_kv_heads, c.head_dim, c.n_rot),
            (heads, 4, 256, 64)
        );
        assert_eq!(
            (c.ffn, c.vocab, c.gdn_v_heads, c.gdn_k_heads),
            (ffn, 248320, vh, 16)
        );
        assert_eq!((c.gdn_k_dim, c.gdn_v_dim, c.gdn_conv), (128, 128, 4));
        assert_eq!(c.mrope_sections, [11, 11, 10]);
        assert_eq!(c.rope_theta, 1e7);
        assert_eq!(
            c.head,
            ClefHeadConfig {
                width: 1024,
                routing_layers: 2,
                layers: 4,
                heads: 16,
                feedforward: 4096
            }
        );
        let meta = |var: &str| {
            std::env::var_os(var).map(|p| {
                crate::safetensors::SafetensorsFile::open(std::path::Path::new(&p))
                    .expect("the companion")
                    .metadata
            })
        };
        if let Some(m) = meta("CLEF_COMPANION") {
            let c = ClefConfig::from_gguf(g.gguf(), Some(&m)).expect("config with the tower");
            let (v, ip) = c.vision.expect("the companion's tower");
            assert_eq!((v.depth, v.hidden, v.out_hidden), (27, 1152, c.hidden));
            assert_eq!((ip.min_pixels, ip.max_pixels), (65536, 16777216));
        }
        if let Some(m) = meta("CLEF_OTHER_COMPANION") {
            let e = ClefConfig::from_gguf(g.gguf(), Some(&m)).expect_err("another Clef's tower");
            assert!(e.contains("different Clef"), "{e}");
        }
    }

    /// Values Transformers' `smart_resize` gives (factor 32, Clef's bounds).
    #[test]
    fn smart_resize_is_the_references() {
        let r = |h, w| smart_resize(h, w, 32, 65536, 16777216).unwrap();
        assert_eq!(r(100, 100), (256, 256));
        assert_eq!(r(480, 640), (480, 640));
        assert_eq!(r(1080, 1920), (1088, 1920));
        assert_eq!(r(5000, 5000), (4096, 4096));
        // ties to even: 1200 / 32 = 37.5 -> 38, 900 / 32 = 28.125 -> 28
        assert_eq!(r(900, 1200), (896, 1216));
        // grown to the floor: 30 x 50 -> 224 x 352 (the oracle's tiny.png)
        assert_eq!(r(30, 50), (224, 352));
        assert_eq!(r(760, 420), (768, 416));
        assert_eq!(
            smart_resize(900, 1200, 32, 65536, 262144).unwrap(),
            (416, 576)
        );
        assert!(smart_resize(10, 2100, 32, 65536, 16777216).is_err());
    }
}
