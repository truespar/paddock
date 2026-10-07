//! `config.json` of Meta's SAM 3 ("Segment Anything with Concepts") as
//! `facebook/sam3` ships it: the transformers layout, `architectures =
//! ["Sam3VideoModel"]` - one file holding the detector (image encoder, text
//! encoder, fusion encoder, DETR decoder, heads) and the SAM 2-lineage
//! tracker. This module reads what the engine's image encoder consumes today
//! (the ViT backbone and the two FPN necks) and grows with the lanes.
//!
//! The parity reference is Meta's own implementation (facebookresearch/sam3),
//! not transformers, and the two disagree on values this file states. Where
//! they do, Meta's wins and the field is NOT read - see [`META_LN_EPS`]. Every
//! other value the graph depends on is validated present and refused by name
//! when it is something the graph does not build.

use std::path::Path;

use crate::safetensors::StError;

/// The `architectures[0]` this parser answers to.
pub const SAM3_ARCH: &str = "Sam3VideoModel";

/// LayerNorm eps of the ViT (all 65 norms). Meta builds them as
/// `nn.LayerNorm(eps=1e-5)`; transformers' port - and so this config's
/// `layer_norm_eps` - says 1e-6. The checkpoint was trained and is served by
/// Meta's code, so 1e-5 is the model and the config field is ignored.
pub const META_LN_EPS: f32 = 1e-5;

/// The image encoder, as the engine consumes it.
#[derive(Debug, Clone)]
pub struct Sam3VisionConfig {
    /// input side in pixels (1008); the picture is resized to a square, aspect
    /// ratio NOT kept, the way Meta's processor does it
    pub image_size: usize,
    pub patch: usize,
    pub channels: usize,
    pub hidden: usize,
    pub n_layer: usize,
    pub n_heads: usize,
    pub intermediate: usize,
    /// side of an attention window, in patches (24)
    pub window: usize,
    /// blocks that attend over the whole grid instead of inside windows
    pub global_blocks: Vec<usize>,
    pub rope_theta: f32,
    /// side of the learned absolute position table, in patches (336 / 14 =
    /// 24). Tiled across the grid, never interpolated.
    pub pos_side: usize,
    /// channels of every FPN level (256)
    pub fpn_dim: usize,
}

impl Sam3VisionConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden / self.n_heads
    }
    /// patch-grid side (72)
    pub fn grid(&self) -> usize {
        self.image_size / self.patch
    }
    /// tokens per picture (5184)
    pub fn tokens(&self) -> usize {
        self.grid() * self.grid()
    }
    /// tokens per window (576)
    pub fn window_tokens(&self) -> usize {
        self.window * self.window
    }
    pub fn is_global(&self, block: usize) -> bool {
        self.global_blocks.contains(&block)
    }

    /// Whether `dir` holds a SAM 3 checkpoint (its config.json names the
    /// `facebook/sam3` layout) - the runner's family test, cheap and silent.
    pub fn is_ours(dir: &Path) -> bool {
        std::fs::read(dir.join("config.json"))
            .ok()
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
            .and_then(|v| {
                v.get("architectures")?
                    .as_array()?
                    .first()?
                    .as_str()
                    .map(str::to_owned)
            })
            .is_some_and(|a| a == SAM3_ARCH)
    }

    pub fn read(dir: &Path) -> Result<Self, StError> {
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)
            .map_err(|e| StError::Header(e.to_string()))?;
        Self::from_value(&v)
    }

    pub fn from_value(v: &serde_json::Value) -> Result<Self, StError> {
        let bad = |m: String| StError::Header(format!("sam3 config.json: {m}"));
        let arch = v
            .get("architectures")
            .and_then(|a| a.as_array())
            .and_then(|a| a.first())
            .and_then(|a| a.as_str())
            .unwrap_or_default();
        if arch != SAM3_ARCH {
            return Err(bad(format!(
                "architectures[0] '{arch}' (want {SAM3_ARCH}, the facebook/sam3 layout)"
            )));
        }
        let at = |path: &[&str]| -> Result<&serde_json::Value, StError> {
            let mut o = v;
            for k in path {
                o = o
                    .get(*k)
                    .ok_or_else(|| bad(format!("missing {}", path.join("."))))?;
            }
            Ok(o)
        };
        let vision = at(&["detector_config", "vision_config"])?;
        let bb = at(&["detector_config", "vision_config", "backbone_config"])?;
        let getu = |o: &serde_json::Value, scope: &str, k: &str| {
            o.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| bad(format!("missing {scope}.{k}")))
        };
        let b = "backbone_config";
        let cfg = Self {
            image_size: getu(bb, b, "image_size")?,
            patch: getu(bb, b, "patch_size")?,
            channels: getu(bb, b, "num_channels")?,
            hidden: getu(bb, b, "hidden_size")?,
            n_layer: getu(bb, b, "num_hidden_layers")?,
            n_heads: getu(bb, b, "num_attention_heads")?,
            intermediate: getu(bb, b, "intermediate_size")?,
            window: getu(bb, b, "window_size")?,
            global_blocks: bb
                .get("global_attn_indexes")
                .and_then(|a| a.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_u64())
                        .map(|x| x as usize)
                        .collect()
                })
                .ok_or_else(|| bad("missing backbone_config.global_attn_indexes".into()))?,
            rope_theta: bb
                .get("rope_theta")
                .and_then(|x| x.as_f64())
                .ok_or_else(|| bad("missing backbone_config.rope_theta".into()))?
                as f32,
            pos_side: getu(bb, b, "pretrain_image_size")? / getu(bb, b, "patch_size")?.max(1),
            fpn_dim: getu(vision, "vision_config", "fpn_hidden_size")?,
        };

        // ---- what the graph builds, refused by name otherwise ----
        if bb.get("qkv_bias").and_then(|x| x.as_bool()) != Some(true) {
            return Err(bad("backbone_config.qkv_bias must be true".into()));
        }
        if bb.get("hidden_act").and_then(|x| x.as_str()) != Some("gelu") {
            return Err(bad("backbone_config.hidden_act must be gelu".into()));
        }
        let scales: Vec<f64> = vision
            .get("scale_factors")
            .and_then(|a| a.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_f64()).collect())
            .unwrap_or_default();
        // the 0.5 level exists in the checkpoint but Meta's backbone drops it
        // (scalp = 1), so the graph serves 4x, 2x and 1x
        if scales != [4.0, 2.0, 1.0, 0.5] {
            return Err(bad(format!(
                "vision_config.scale_factors {scales:?} (the necks build [4, 2, 1, 0.5])"
            )));
        }
        if cfg.channels != 3 {
            return Err(bad(format!("{} input channels (want 3)", cfg.channels)));
        }
        if cfg.patch == 0 || !cfg.image_size.is_multiple_of(cfg.patch) {
            return Err(bad(format!(
                "image_size {} is not a whole number of {}-px patches",
                cfg.image_size, cfg.patch
            )));
        }
        if cfg.window == 0 || !cfg.grid().is_multiple_of(cfg.window) {
            return Err(bad(format!(
                "a {}-patch grid is not a whole number of {}-patch windows (the padded \
                 partition is not built)",
                cfg.grid(),
                cfg.window
            )));
        }
        // the window-major row order puts one copy of the position table in
        // each window; that only works when the tile period is the window side
        if cfg.pos_side != cfg.window {
            return Err(bad(format!(
                "position table side {} != window side {}",
                cfg.pos_side, cfg.window
            )));
        }
        if cfg.n_heads == 0 || !cfg.hidden.is_multiple_of(cfg.n_heads) {
            return Err(bad(format!(
                "hidden {} over {} heads",
                cfg.hidden, cfg.n_heads
            )));
        }
        let hd = cfg.head_dim();
        // the half attention's geometry, and an even head for the rope pairs
        if !hd.is_multiple_of(8) || !(16..=128).contains(&hd) {
            return Err(bad(format!(
                "head_dim {hd} (want a multiple of 8 in 16..=128)"
            )));
        }
        if !cfg.hidden.is_multiple_of(8) || !cfg.intermediate.is_multiple_of(8) {
            return Err(bad(format!(
                "hidden {} / intermediate {} (the f16 GEMM stages 8-wide units)",
                cfg.hidden, cfg.intermediate
            )));
        }
        if cfg.global_blocks.iter().any(|&i| i >= cfg.n_layer) {
            return Err(bad(format!(
                "global_attn_indexes {:?} past {} blocks",
                cfg.global_blocks, cfg.n_layer
            )));
        }
        if cfg.fpn_dim == 0 || !cfg.fpn_dim.is_multiple_of(8) || !cfg.hidden.is_multiple_of(4) {
            return Err(bad(format!("fpn_hidden_size {}", cfg.fpn_dim)));
        }
        Ok(cfg)
    }
}

/// The text tower (Meta's CLIP-architecture `TextTransformer`) and the
/// resizer that brings its tokens to the detector's width.
#[derive(Debug, Clone)]
pub struct Sam3TextConfig {
    pub vocab: usize,
    /// prompt length the tower reads, markers included (32)
    pub context: usize,
    pub hidden: usize,
    pub n_layer: usize,
    pub n_heads: usize,
    pub intermediate: usize,
    /// the detector's width the resizer lands on (256)
    pub d_model: usize,
}

impl Sam3TextConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden / self.n_heads
    }

    pub fn read(dir: &Path) -> Result<Self, StError> {
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)
            .map_err(|e| StError::Header(e.to_string()))?;
        Self::from_value(&v)
    }

    pub fn from_value(v: &serde_json::Value) -> Result<Self, StError> {
        let bad = |m: String| StError::Header(format!("sam3 config.json: {m}"));
        let t = v
            .get("detector_config")
            .and_then(|d| d.get("text_config"))
            .ok_or_else(|| bad("missing detector_config.text_config".into()))?;
        let getu = |k: &str| {
            t.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| bad(format!("missing text_config.{k}")))
        };
        let d_model = v
            .get("detector_config")
            .and_then(|d| d.get("vision_config"))
            .and_then(|d| d.get("fpn_hidden_size"))
            .and_then(|x| x.as_u64())
            .ok_or_else(|| bad("missing vision_config.fpn_hidden_size".into()))?
            as usize;
        let cfg = Self {
            vocab: getu("vocab_size")?,
            context: getu("max_position_embeddings")?,
            hidden: getu("hidden_size")?,
            n_layer: getu("num_hidden_layers")?,
            n_heads: getu("num_attention_heads")?,
            intermediate: getu("intermediate_size")?,
            d_model,
        };
        // Meta's text tower is nn.GELU (exact) with LayerNorm eps 1e-5; here
        // the config agrees with it, and anything else is a different tower
        if t.get("hidden_act").and_then(|x| x.as_str()) != Some("gelu") {
            return Err(bad("text_config.hidden_act must be gelu".into()));
        }
        let eps = t.get("layer_norm_eps").and_then(|x| x.as_f64());
        if eps.is_none_or(|e| (e - META_LN_EPS as f64).abs() > 1e-12) {
            return Err(bad(format!(
                "text_config.layer_norm_eps {eps:?} (Meta's tower is 1e-5)"
            )));
        }
        // the causal attention kernel holds a prompt in shared memory
        if cfg.context != 32 || cfg.n_heads == 0 || !cfg.hidden.is_multiple_of(cfg.n_heads) {
            return Err(bad(format!(
                "text tower: context {} / {} heads over {}",
                cfg.context, cfg.n_heads, cfg.hidden
            )));
        }
        let hd = cfg.head_dim();
        if hd > 64 || !hd.is_multiple_of(2) {
            return Err(bad(format!(
                "text head_dim {hd} (the causal kernel takes <= 64)"
            )));
        }
        if !cfg.hidden.is_multiple_of(8) || !cfg.intermediate.is_multiple_of(8) {
            return Err(bad("text tower widths must be multiples of 8".into()));
        }
        Ok(cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn good() -> serde_json::Value {
        serde_json::json!({
            "architectures": ["Sam3VideoModel"],
            "detector_config": {
                "vision_config": {
                    "fpn_hidden_size": 256,
                    "scale_factors": [4.0, 2.0, 1.0, 0.5],
                    "backbone_config": {
                        "global_attn_indexes": [7, 15, 23, 31],
                        "hidden_act": "gelu",
                        "hidden_size": 1024,
                        "image_size": 1008,
                        "intermediate_size": 4736,
                        "layer_norm_eps": 1e-6,
                        "num_attention_heads": 16,
                        "num_channels": 3,
                        "num_hidden_layers": 32,
                        "patch_size": 14,
                        "pretrain_image_size": 336,
                        "qkv_bias": true,
                        "rope_theta": 10000.0,
                        "window_size": 24
                    }
                }
            }
        })
    }

    #[test]
    fn reads_the_published_shape() {
        let c = Sam3VisionConfig::from_value(&good()).unwrap();
        assert_eq!((c.grid(), c.tokens(), c.window_tokens()), (72, 5184, 576));
        assert_eq!((c.head_dim(), c.pos_side, c.fpn_dim), (64, 24, 256));
        assert!(c.is_global(7) && c.is_global(31) && !c.is_global(8));
    }

    #[test]
    fn refuses_what_the_graph_does_not_build() {
        let mut v = good();
        v["architectures"][0] = "Sam3Model".into();
        assert!(Sam3VisionConfig::from_value(&v).is_err());

        let mut v = good();
        v["detector_config"]["vision_config"]["backbone_config"]["window_size"] = 16.into();
        let e = Sam3VisionConfig::from_value(&v).unwrap_err().to_string();
        assert!(e.contains("windows") || e.contains("window side"), "{e}");

        let mut v = good();
        v["detector_config"]["vision_config"]["backbone_config"]["qkv_bias"] = false.into();
        assert!(Sam3VisionConfig::from_value(&v).is_err());
    }
}
