//! The checkpoint contract of NVIDIA's Nemotron 3.5 Parse
//! (`architectures = ["RadioMBartPrefixForCausalLM"]`, `model_type =
//! "nemotron_parse"`): a C-RADIOv4-SO400M picture tower whose neck output is
//! the causal prefix of a 10-layer mBART decoder (no cross-attention), plus one
//! trained multi-token-prediction head.
//!
//! Three files make the contract and all three are read here:
//! `config.json` (geometry), `preprocessor_config.json` (the pixel path) and
//! `generation_config.json` (the decoding the checkpoint was evaluated with).
//! Same stance as every safetensors family: a field the engine consumes is
//! validated present, and a value the graph does not implement is refused by
//! NAME. This family needs it more than most - the release ships several
//! switches (`legacy_causal_vision_prefix`, `navit_position_embedding_mode`,
//! `navit_padding_mode`, ...) whose other settings load every tensor and
//! return fluent garbage.
//!
//! What is NOT in the files and is an architecture constant of the release's
//! own modeling code, pinned here so it has one home:
//! - the tower's LayerNorm eps (1e-6, timm's `vit_so400m_patch16_224` blocks),
//!   the decoder's (1e-5, `nn.LayerNorm`'s default);
//! - exact-erf GELU in both (timm `nn.GELU()` and mBART's `gelu`);
//! - the learned position grid (128 x 128) is read off the tensor shape by the
//!   loader, never assumed.

use std::path::Path;

use crate::safetensors::StError;

/// The `architectures[0]` this parser answers to.
pub const ARCH: &str = "RadioMBartPrefixForCausalLM";
/// `model_type`, checked beside the architecture.
pub const MODEL_TYPE: &str = "nemotron_parse";
/// The only tower this release ships (`encoder.args.model`).
pub const TOWER: &str = "vit_so400m_patch16_224";

/// timm's ViT block norm (`partial(nn.LayerNorm, eps=1e-6)` in the release's
/// modeling file) - not carried by config.json.
pub const TOWER_LN_EPS: f32 = 1e-6;
/// The neck's three LayerNorms are built with eps 1e-6 too (`RadioWithNeck`).
pub const NECK_LN_EPS: f32 = 1e-6;
/// mBART's `nn.LayerNorm(d_model)` - PyTorch's default eps.
pub const DECODER_LN_EPS: f32 = 1e-5;

/// The generation contract NVIDIA publishes for this checkpoint (the
/// release's serving notes): the same prompt in Transformers, vLLM and
/// evaluation.
pub const CONTRACT_PROMPT: &str =
    "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>";
/// The contract's `max_tokens` (`docs/serving.md`, the README's quick start
/// and `run_transformers_inference.py` all use it; generation_config.json's
/// 9000 is the export default, not the evaluated setting).
pub const CONTRACT_MAX_TOKENS: usize = 8744;

/// Everything the engine and the runner consume, validated.
#[derive(Debug, Clone)]
pub struct NemotronParseConfig {
    // ---- tower (config.encoder) ----
    pub patch: usize,
    /// width of the SO400M tower (1152); the loader checks it against the
    /// tensors, the config's own `vision_spatial_dim` must agree
    pub tower_dim: usize,
    /// `vision_summary_dim` - the two summary tokens concatenated (2304)
    pub summary_dim: usize,
    /// `encoder.args.register_multiple` - CLS + register tokens parked ahead
    /// of the patches in the reference (10); the loader checks the tensor
    pub n_prefix_tokens: usize,
    /// `encoder.max_resolution` (2048) - the position field's source extent
    pub max_resolution: usize,
    // ---- neck ----
    /// patches merged across by the (1, 4) stride-4 conv (`neck_compression`)
    pub neck_compression: usize,
    // ---- decoder (config.decoder) ----
    pub d_model: usize,
    pub n_layer: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub ffn: usize,
    /// embedding / LM-head rows (72512: 72505 real ids padded to a multiple of 64)
    pub vocab: usize,
    /// `scale_embedding`: text embeddings x sqrt(d_model); the visual prefix is not scaled
    pub embed_scale: f32,
    /// `max_sequence_length` - prefix + prompt + output, the reference refuses past it
    pub max_ctx: usize,
    /// `num_extra_heads` - trained one-step MTP heads (1)
    pub n_mtp_heads: usize,
    // ---- ids ----
    pub bos_id: u32,
    pub pad_id: u32,
    pub eos_id: u32,
    /// the visual placeholder (`image_token_index`, 1 = `<pad>`)
    pub image_token_id: u32,
    /// first `<class_*>` id (`class_token_start_idx`)
    pub class_token_start: u32,
    // ---- pixels (preprocessor_config.json) ----
    /// (height, width) caps of the aspect-preserving downscale (2048, 1664)
    pub max_hw: (usize, usize),
    pub mean: [f32; 3],
    pub std: [f32; 3],
    /// width padding unit = patch x neck_compression (64): the neck's merge
    /// must see whole groups of patches across
    pub width_unit: usize,
    // ---- decoding (generation_config.json) ----
    pub repetition_penalty: f32,
    pub max_new_tokens: usize,
}

impl NemotronParseConfig {
    /// Tower attention heads: SO400M is 16 x 72 (a release constant - the
    /// width is 1152 and timm builds it with 16 heads).
    pub fn tower_heads(&self) -> usize {
        16
    }
    pub fn tower_head_dim(&self) -> usize {
        self.tower_dim / self.tower_heads()
    }
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }

    /// Whether `dir/config.json` names this family at all - the runner's
    /// cheap dispatch probe. Not ours -> false; ours but malformed is left for
    /// [`Self::read`] to refuse loudly.
    pub fn is_ours(dir: &Path) -> bool {
        std::fs::read(dir.join("config.json"))
            .ok()
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
            .is_some_and(|v| {
                v.get("model_type").and_then(|m| m.as_str()) == Some(MODEL_TYPE)
                    && v.get("architectures")
                        .and_then(|a| a.as_array())
                        .and_then(|a| a.first())
                        .and_then(|a| a.as_str())
                        == Some(ARCH)
            })
    }

    pub fn read(dir: &Path) -> Result<Self, StError> {
        let load = |name: &str| -> Result<serde_json::Value, StError> {
            serde_json::from_slice(&std::fs::read(dir.join(name))?)
                .map_err(|e| StError::Header(format!("{name}: {e}")))
        };
        Self::from_json(
            &load("config.json")?,
            &load("preprocessor_config.json")?,
            &load("generation_config.json")?,
        )
    }

    /// The three documents already parsed - split from [`Self::read`] so the
    /// refusals are testable without a checkpoint on disk.
    pub fn from_json(
        cfg: &serde_json::Value,
        pre: &serde_json::Value,
        gen_cfg: &serde_json::Value,
    ) -> Result<Self, StError> {
        let bad = |m: String| StError::Header(format!("nemotron-parse: {m}"));
        let obj = |v: &serde_json::Value, k: &str, scope: &str| {
            v.get(k)
                .cloned()
                .ok_or_else(|| bad(format!("{scope} missing {k}")))
        };
        let u = |v: &serde_json::Value, k: &str, scope: &str| {
            v.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| bad(format!("{scope} missing integer {k}")))
        };
        let f = |v: &serde_json::Value, k: &str, scope: &str| {
            v.get(k)
                .and_then(|x| x.as_f64())
                .ok_or_else(|| bad(format!("{scope} missing number {k}")))
        };
        let s = |v: &serde_json::Value, k: &str, scope: &str| {
            v.get(k)
                .and_then(|x| x.as_str())
                .map(str::to_owned)
                .ok_or_else(|| bad(format!("{scope} missing string {k}")))
        };
        let b = |v: &serde_json::Value, k: &str, scope: &str| {
            v.get(k)
                .and_then(|x| x.as_bool())
                .ok_or_else(|| bad(format!("{scope} missing bool {k}")))
        };
        // a switch whose other settings this graph does not implement
        let want_str = |v: &serde_json::Value, k: &str, scope: &str, want: &str| {
            let got = s(v, k, scope)?;
            if got != want {
                return Err(bad(format!(
                    "{scope}.{k} = {got:?} (only {want:?} is implemented)"
                )));
            }
            Ok(())
        };
        let want_true = |v: &serde_json::Value, k: &str, scope: &str| {
            if !b(v, k, scope)? {
                return Err(bad(format!(
                    "{scope}.{k} = false (only true is implemented)"
                )));
            }
            Ok(())
        };

        // ---- identity ----
        let arch = cfg
            .get("architectures")
            .and_then(|a| a.as_array())
            .and_then(|a| a.first())
            .and_then(|a| a.as_str())
            .unwrap_or_default();
        if arch != ARCH {
            return Err(bad(format!("architectures[0] {arch:?} (want {ARCH:?})")));
        }
        want_str(cfg, "model_type", "config", MODEL_TYPE)?;
        // The release's modeling code refuses anything else; so do we. The
        // encoder-decoder (cross-attention) Parse v1.x/2.0 graphs share the
        // model_type but are a different network.
        want_true(cfg, "legacy_causal_vision_prefix", "config")?;
        want_true(cfg, "navit_packing", "config")?;
        want_str(
            cfg,
            "navit_position_embedding_mode",
            "config",
            "content_normalized_2d_interpolate",
        )?;
        want_str(
            cfg,
            "navit_coordinate_frame",
            "config",
            "content_normalized",
        )?;
        if u(cfg, "cross_attention_kv_heads", "config")? != 0 {
            return Err(bad(
                "config.cross_attention_kv_heads != 0 - the cross-attention graph".into(),
            ));
        }
        if !b(cfg, "tie_word_embeddings", "config")? {
            return Err(bad(
                "config.tie_word_embeddings = false - this graph ties the LM head".into(),
            ));
        }

        // ---- tower ----
        let enc = obj(cfg, "encoder", "config")?;
        let args = obj(&enc, "args", "config.encoder")?;
        want_str(&args, "model", "config.encoder.args", TOWER)?;
        let patch = u(&enc, "patch_size", "config.encoder")?;
        let tower_dim = u(cfg, "vision_spatial_dim", "config")?;
        let summary_dim = u(cfg, "vision_summary_dim", "config")?;
        let n_prefix_tokens = u(&args, "register_multiple", "config.encoder.args")?;
        let max_resolution = u(&enc, "max_resolution", "config.encoder")?;
        if let Some(w) = enc.get("vitdet_window_size")
            && !w.is_null()
        {
            return Err(bad(
                "config.encoder.vitdet_window_size set - windowed attention is not this release"
                    .into(),
            ));
        }
        if summary_dim != 2 * tower_dim {
            return Err(bad(format!(
                "vision_summary_dim {summary_dim} != 2 x vision_spatial_dim {tower_dim} \
                 (the summary is two concatenated tokens)"
            )));
        }
        if tower_dim != 1152 {
            return Err(bad(format!(
                "vision_spatial_dim {tower_dim} - SO400M is 1152 wide"
            )));
        }
        let neck_compression = cfg
            .get("neck_compression")
            .and_then(|x| x.as_u64())
            .map_or(4, |x| x as usize);
        if neck_compression != 4 {
            return Err(bad(format!(
                "neck_compression {neck_compression} - the shipped neck conv is (1, 4)"
            )));
        }

        // ---- decoder ----
        let dec = obj(cfg, "decoder", "config")?;
        let d_model = u(&dec, "d_model", "config.decoder")?;
        let n_layer = u(&dec, "decoder_layers", "config.decoder")?;
        let n_heads = u(&dec, "decoder_attention_heads", "config.decoder")?;
        let n_kv_heads = u(&dec, "self_attention_kv_heads", "config.decoder")?;
        let ffn = u(&dec, "decoder_ffn_dim", "config.decoder")?;
        let vocab = u(&dec, "vocab_size", "config.decoder")?;
        want_str(&dec, "activation_function", "config.decoder", "gelu")?;
        if dec.get("add_cross_attention").and_then(|x| x.as_bool()) != Some(false) {
            return Err(bad("config.decoder.add_cross_attention is not false".into()));
        }
        let embed_scale = if b(&dec, "scale_embedding", "config.decoder")? {
            (d_model as f32).sqrt()
        } else {
            1.0
        };
        if n_kv_heads == 0 || n_heads % n_kv_heads != 0 || d_model % n_heads != 0 {
            return Err(bad(format!(
                "decoder heads {n_heads} / kv {n_kv_heads} / width {d_model} do not tile"
            )));
        }
        let max_ctx = u(cfg, "max_sequence_length", "config")?;
        let n_mtp_heads = cfg
            .get("num_extra_heads")
            .and_then(|x| x.as_u64())
            .map_or(0, |x| x as usize);
        if n_mtp_heads > 1 {
            return Err(bad(format!(
                "num_extra_heads {n_mtp_heads} - one trained MTP head is implemented"
            )));
        }

        // ---- ids ----
        let id = |v: &serde_json::Value, k: &str, scope: &str| u(v, k, scope).map(|x| x as u32);
        let bos_id = id(cfg, "bos_token_id", "config")?;
        let pad_id = id(cfg, "pad_token_id", "config")?;
        let eos_id = id(cfg, "eos_token_id", "config")?;
        let image_token_id = id(cfg, "image_token_index", "config")?;
        let class_token_start = id(cfg, "class_token_start_idx", "config")?;

        // ---- pixels ----
        want_true(pre, "navit_packing", "preprocessor_config")?;
        want_str(
            pre,
            "navit_padding_mode",
            "preprocessor_config",
            "top_left_white",
        )?;
        want_str(
            pre,
            "navit_coordinate_frame",
            "preprocessor_config",
            "content_normalized",
        )?;
        want_true(pre, "do_normalize", "preprocessor_config")?;
        let fs = pre
            .get("final_size")
            .and_then(|x| x.as_array())
            .filter(|a| a.len() == 2)
            .ok_or_else(|| bad("preprocessor_config missing final_size [h, w]".into()))?;
        let dim = |i: usize| {
            fs[i]
                .as_u64()
                .map(|x| x as usize)
                .ok_or_else(|| bad("final_size not integers".into()))
        };
        let max_hw = (dim(0)?, dim(1)?);
        let rescale = f(pre, "rescale_factor", "preprocessor_config")?;
        if (rescale - 1.0 / 255.0).abs() > 1e-12 {
            return Err(bad(format!(
                "rescale_factor {rescale} (the processor divides by 255)"
            )));
        }
        let three = |k: &str| -> Result<[f32; 3], StError> {
            let a = pre
                .get(k)
                .and_then(|x| x.as_array())
                .filter(|a| a.len() == 3)
                .ok_or_else(|| bad(format!("preprocessor_config missing {k}[3]")))?;
            let mut o = [0f32; 3];
            for (i, v) in a.iter().enumerate() {
                o[i] = v.as_f64().ok_or_else(|| bad(format!("{k} not numbers")))? as f32;
            }
            Ok(o)
        };
        let mean = three("image_mean")?;
        let std = three("image_std")?;
        if max_hw.0 % patch != 0 || max_hw.1 % (patch * neck_compression) != 0 {
            return Err(bad(format!(
                "final_size {max_hw:?} not on the {patch} x {} grid",
                patch * neck_compression
            )));
        }
        if max_hw.0.max(max_hw.1) > max_resolution {
            return Err(bad(format!(
                "final_size {max_hw:?} past the tower's max_resolution {max_resolution}"
            )));
        }

        // ---- decoding ----
        let repetition_penalty = gen_cfg
            .get("repetition_penalty")
            .and_then(|x| x.as_f64())
            .map_or(1.0, |x| x as f32);
        // NaN fails too: only a positive, finite factor is a penalty
        if !repetition_penalty.is_finite() || repetition_penalty <= 0.0 {
            return Err(bad(format!("repetition_penalty {repetition_penalty}")));
        }
        if gen_cfg.get("do_sample").and_then(|x| x.as_bool()) == Some(true) {
            return Err(bad(
                "generation_config do_sample = true - the contract is greedy".into(),
            ));
        }

        Ok(Self {
            patch,
            tower_dim,
            summary_dim,
            n_prefix_tokens,
            max_resolution,
            neck_compression,
            d_model,
            n_layer,
            n_heads,
            n_kv_heads,
            ffn,
            vocab,
            embed_scale,
            max_ctx,
            n_mtp_heads,
            bos_id,
            pad_id,
            eos_id,
            image_token_id,
            class_token_start,
            max_hw,
            mean,
            std,
            width_unit: patch * neck_compression,
            repetition_penalty,
            max_new_tokens: CONTRACT_MAX_TOKENS,
        })
    }

    /// The content size the processor resizes a `w` x `h` page to - its
    /// `LongestMaxSizeHW`, reproduced line for line (downscale only; height
    /// first, then width; `int()` truncation of the float products).
    pub fn content_hw(&self, w: usize, h: usize) -> (usize, usize) {
        let (max_h, max_w) = self.max_hw;
        let aspect = w as f64 / h as f64;
        let (mut nh, mut nw) = (h, w);
        if h > max_h {
            nh = max_h;
            nw = (nh as f64 * aspect) as usize;
        }
        if nw > max_w {
            nw = max_w;
            nh = (nw as f64 / aspect) as usize;
        }
        (nh.max(1), nw.max(1))
    }

    /// The padded canvas (`_pad_to_navit_grid`): height up to the patch,
    /// width up to patch x neck_compression, white bottom/right.
    pub fn canvas_hw(&self, content: (usize, usize)) -> (usize, usize) {
        (
            content.0.div_ceil(self.patch) * self.patch,
            content.1.div_ceil(self.width_unit) * self.width_unit,
        )
    }

    /// Decoder rows one page's picture becomes: (H/16)(W/64) merged patch
    /// rows + the summary row.
    pub fn prefix_rows(&self, canvas: (usize, usize)) -> usize {
        (canvas.0 / self.patch) * (canvas.1 / self.width_unit) + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shipped files (the revision the catalog pins), trimmed to the keys
    /// this reader consumes plus the switches it refuses on.
    fn shipped() -> (serde_json::Value, serde_json::Value, serde_json::Value) {
        let cfg = serde_json::json!({
            "architectures": ["RadioMBartPrefixForCausalLM"],
            "model_type": "nemotron_parse",
            "bos_token_id": 0, "pad_token_id": 1, "eos_token_id": 2,
            "image_token_index": 1, "class_token_start_idx": 52315,
            "max_sequence_length": 16384, "tie_word_embeddings": true,
            "legacy_causal_vision_prefix": true, "navit_packing": true,
            "navit_position_embedding_mode": "content_normalized_2d_interpolate",
            "navit_coordinate_frame": "content_normalized",
            "cross_attention_kv_heads": 0, "num_extra_heads": 1,
            "vision_spatial_dim": 1152, "vision_summary_dim": 2304,
            "encoder": {
                "patch_size": 16, "max_resolution": 2048, "vitdet_window_size": null,
                "args": {"model": "vit_so400m_patch16_224", "register_multiple": 10}
            },
            "decoder": {
                "d_model": 1024, "decoder_layers": 10, "decoder_attention_heads": 16,
                "self_attention_kv_heads": 8, "decoder_ffn_dim": 4096, "vocab_size": 72512,
                "activation_function": "gelu", "add_cross_attention": false,
                "scale_embedding": true
            }
        });
        let pre = serde_json::json!({
            "navit_packing": true, "navit_padding_mode": "top_left_white",
            "navit_coordinate_frame": "content_normalized", "do_normalize": true,
            "final_size": [2048, 1664], "rescale_factor": 0.00392156862745098,
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_std": [0.26862954, 0.26130258, 0.27577711]
        });
        let gen_cfg = serde_json::json!({"do_sample": false, "repetition_penalty": 1.1});
        (cfg, pre, gen_cfg)
    }

    #[test]
    fn the_shipped_contract_reads() {
        let (c, p, g) = shipped();
        let cfg = NemotronParseConfig::from_json(&c, &p, &g).expect("reads");
        assert_eq!(
            (cfg.tower_dim, cfg.tower_head_dim(), cfg.head_dim()),
            (1152, 72, 64)
        );
        assert_eq!(cfg.embed_scale, 32.0);
        assert_eq!(cfg.width_unit, 64);
        assert_eq!(cfg.repetition_penalty, 1.1);
        // a full page and an A4 at the cap
        assert_eq!(cfg.prefix_rows((2048, 1664)), 3329);
        assert_eq!(cfg.prefix_rows(cfg.canvas_hw((2048, 1448))), 2945);
    }

    /// `LongestMaxSizeHW` truncates, height first, and never upscales.
    #[test]
    fn content_size_follows_the_processor() {
        let (c, p, g) = shipped();
        let cfg = NemotronParseConfig::from_json(&c, &p, &g).expect("reads");
        assert_eq!(cfg.content_hw(1240, 1754), (1754, 1240)); // fits: untouched
        assert_eq!(cfg.content_hw(1448, 2048), (2048, 1448));
        // too tall: h -> 2048, w = int(2048 * 2480/3508) = 1447
        assert_eq!(cfg.content_hw(2480, 3508), (2048, 1447));
        // too wide after the height pass: w -> 1664, h = int(1664 / aspect)
        assert_eq!(cfg.content_hw(4000, 2000), (832, 1664));
        assert_eq!(cfg.canvas_hw((1754, 1240)), (1760, 1280));
    }

    #[test]
    fn graph_switches_it_does_not_implement_are_refused_by_name() {
        for (path, val) in [
            ("legacy_causal_vision_prefix", serde_json::json!(false)),
            (
                "navit_position_embedding_mode",
                serde_json::json!("legacy_virtual_canvas"),
            ),
            ("cross_attention_kv_heads", serde_json::json!(8)),
        ] {
            let (mut c, p, g) = shipped();
            c[path] = val;
            let e = NemotronParseConfig::from_json(&c, &p, &g).expect_err("refused");
            assert!(e.to_string().contains(path), "{e}");
        }
        let (c, mut p, g) = shipped();
        p["navit_padding_mode"] = serde_json::json!("center_white");
        let e = NemotronParseConfig::from_json(&c, &p, &g).expect_err("refused");
        assert!(e.to_string().contains("navit_padding_mode"), "{e}");
    }
}
