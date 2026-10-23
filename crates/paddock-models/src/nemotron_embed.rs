//! The checkpoint contract of NVIDIA's Nemotron 3.5 Embed 8B and Rerank 8B
//! (`nvidia/Nemotron-3.5-Embed-8B-BF16`, `architectures =
//! ["Mistral3Model"]`; `nvidia/Nemotron-3.5-Rerank-8B-BF16`,
//! `["Mistral3ForSequenceClassification"]`, both `model_type = "mistral3"`):
//! a Pixtral picture tower and Mistral 3's projector in front of a
//! bidirectional Ministral 3 8B body, mean-pooled over every token.
//!
//! - the embedder L2-normalizes the mean (a 4096-wide unit vector, or its
//!   Matryoshka prefix re-normalized);
//! - the reranker scores it: one `score` row (4096 -> 1, no bias) on the f32
//!   mean, the raw logit being the relevance (higher = more relevant; the
//!   release's Sentence Transformers head is `Identity`).
//!
//! Four files make the contract and all four are read here: `config.json`
//! (geometry), `processor_config.json` (the pixel path),
//! `sentence_bert_config.json` (the context: `max_seq_length`) and
//! `1_Pooling/config.json` (mean over all tokens, prompt included). Same
//! stance as every safetensors family: a field the engine consumes is
//! validated present, and a value the graph does not implement is refused by
//! NAME.
//!
//! What is NOT in the files and is an architecture constant of the release's
//! modeling code (Transformers' `pixtral`, `mistral3`, `ministral3`), pinned
//! here so it has one home:
//! - Pixtral's two norms a block and `ln_pre` are RMSNorm at eps 1e-5
//!   (`PixtralRMSNorm(.., eps=1e-5)`), no final norm; the projector's norm
//!   takes the TEXT config's `rms_norm_eps`;
//! - Pixtral's 2D rope: the 32 frequencies of a 64-wide head split even /
//!   odd, the even ones turned by the patch row and the odd ones by its
//!   column, rotate-half over the whole head ([`Self::vision_rope_angles`]);
//! - the projector's patch merger is `F.unfold` over 2 x 2 blocks, so its
//!   input column is `c * 4 + (dy * 2 + dx)` ([`MERGE_COL`]).

use std::path::Path;

use crate::safetensors::StError;

/// `architectures[0]` of the embedder.
pub const ARCH_EMBED: &str = "Mistral3Model";
/// `architectures[0]` of the reranker (its own `model.py`).
pub const ARCH_RERANK: &str = "Mistral3ForSequenceClassification";
/// `model_type`, checked beside the architecture.
pub const MODEL_TYPE: &str = "mistral3";

/// Pixtral's RMSNorm eps (`PixtralRMSNorm(.., eps=1e-5)` in every block and
/// `ln_pre`) - not carried by the vision config.
pub const VISION_RMS_EPS: f32 = 1e-5;

/// The patch merger's input column for channel `c` of block position
/// (`dy`, `dx`): `F.unfold`'s (C, kh, kw) order with C slowest.
pub const fn merge_col(c: usize, dy: usize, dx: usize) -> usize {
    c * 4 + dy * 2 + dx
}
/// [`merge_col`] as a value (the docs point here).
pub const MERGE_COL: fn(usize, usize, usize) -> usize = merge_col;

/// Embedder or reranker: the same graph up to the pooled mean.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Embed,
    Rerank,
}

/// YaRN as Transformers' `_compute_yarn_parameters` reads it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Yarn {
    pub theta: f64,
    pub factor: f64,
    pub original_max_pos: usize,
    pub beta_fast: f64,
    pub beta_slow: f64,
    /// the cos / sin multiplier (`attention_factor`, else derived from
    /// `mscale` / `mscale_all_dim`) - 1.0 for these checkpoints
    pub attention_factor: f64,
    pub truncate: bool,
}

/// Everything the engine and the runner consume, validated.
#[derive(Debug, Clone)]
pub struct NemotronEmbedConfig {
    pub kind: Kind,
    /// `model.` for the reranker (its body sits under the sequence-
    /// classification wrapper), empty for the embedder
    pub prefix: &'static str,
    // ---- text body (text_config) ----
    pub d_model: usize,
    pub n_layer: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub vocab: usize,
    pub rms_eps: f32,
    pub rope: Yarn,
    /// Llama 4's query scaling: q x (1 + beta ln(1 + floor(pos / orig)))
    pub l4_beta: f32,
    pub l4_orig: usize,
    /// `sentence_bert_config.max_seq_length` - text and picture tokens
    pub max_ctx: usize,
    // ---- picture tower (vision_config) ----
    pub v_dim: usize,
    pub v_layers: usize,
    pub v_heads: usize,
    pub v_head_dim: usize,
    pub v_ffn: usize,
    pub patch: usize,
    pub v_theta: f64,
    /// 2 x 2 patches a picture token
    pub merge: usize,
    /// `[IMG]`
    pub image_token_id: u32,
    // ---- pixels (processor_config.image_processor) ----
    pub longest_edge: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub rescale: f32,
}

impl NemotronEmbedConfig {
    /// Whether `dir/config.json` names this family - the runner's dispatch
    /// probe. Ours but malformed is left for [`Self::read`] to refuse.
    pub fn is_ours(dir: &Path) -> bool {
        std::fs::read(dir.join("config.json"))
            .ok()
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
            .is_some_and(|v| {
                let arch = v
                    .get("architectures")
                    .and_then(|a| a.as_array())
                    .and_then(|a| a.first())
                    .and_then(|a| a.as_str());
                v.get("model_type").and_then(|m| m.as_str()) == Some(MODEL_TYPE)
                    && matches!(arch, Some(ARCH_EMBED | ARCH_RERANK))
                    && v.pointer("/text_config/model_type")
                        .and_then(|m| m.as_str())
                        == Some("ministral3")
            })
    }

    pub fn read(dir: &Path) -> Result<Self, StError> {
        let load = |name: &str| -> Result<serde_json::Value, StError> {
            serde_json::from_slice(&std::fs::read(dir.join(name))?)
                .map_err(|e| StError::Header(format!("{name}: {e}")))
        };
        Self::from_json(
            &load("config.json")?,
            &load("processor_config.json")?,
            &load("sentence_bert_config.json")?,
            &load("1_Pooling/config.json")?,
        )
    }

    /// The four documents already parsed - split from [`Self::read`] so the
    /// refusals are testable without a checkpoint on disk.
    pub fn from_json(
        cfg: &serde_json::Value,
        proc: &serde_json::Value,
        st: &serde_json::Value,
        pool: &serde_json::Value,
    ) -> Result<Self, StError> {
        let bad = |m: String| StError::Header(format!("nemotron-embed: {m}"));
        let get = |v: &serde_json::Value, k: &str, scope: &str| {
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
        let want_str = |v: &serde_json::Value, k: &str, scope: &str, want: &str| {
            let got = s(v, k, scope)?;
            if got != want {
                return Err(bad(format!(
                    "{scope}.{k} = {got:?} (only {want:?} is implemented)"
                )));
            }
            Ok(())
        };
        let want_bool = |v: &serde_json::Value, k: &str, scope: &str, want: bool| match v
            .get(k)
            .and_then(|x| x.as_bool())
        {
            Some(b) if b == want => Ok(()),
            Some(b) => Err(bad(format!(
                "{scope}.{k} = {b} (only {want} is implemented)"
            ))),
            None => Err(bad(format!("{scope} missing bool {k}"))),
        };

        // ---- identity ----
        let arch = cfg
            .get("architectures")
            .and_then(|a| a.as_array())
            .and_then(|a| a.first())
            .and_then(|a| a.as_str())
            .unwrap_or_default();
        let kind = match arch {
            ARCH_EMBED => Kind::Embed,
            ARCH_RERANK => Kind::Rerank,
            other => {
                return Err(bad(format!(
                    "architectures[0] {other:?} (want {ARCH_EMBED:?} or {ARCH_RERANK:?})"
                )));
            }
        };
        want_str(cfg, "model_type", "config", MODEL_TYPE)?;
        // a causal checkpoint of the same graph is a different model: the
        // retrieval releases attend both ways (Transformers' mask builder
        // reads this flag)
        want_bool(cfg, "is_causal", "config", false)?;
        want_str(cfg, "projector_hidden_act", "config", "gelu")?;
        want_bool(cfg, "multimodal_projector_bias", "config", false)?;
        if cfg.get("vision_feature_layer").and_then(|x| x.as_i64()) != Some(-1) {
            return Err(bad(
                "config.vision_feature_layer is not -1 (only the last tower layer is implemented)"
                    .into(),
            ));
        }
        let merge = u(cfg, "spatial_merge_size", "config")?;
        if merge != 2 {
            return Err(bad(format!(
                "config.spatial_merge_size {merge} (the 2 x 2 merger is implemented)"
            )));
        }
        let image_token_id = u(cfg, "image_token_index", "config")? as u32;
        if kind == Kind::Rerank {
            let labels = cfg
                .get("id2label")
                .and_then(|x| x.as_object())
                .map_or(0, |m| m.len());
            if labels != 1 {
                return Err(bad(format!(
                    "config.id2label has {labels} labels (the one-logit score head is implemented)"
                )));
            }
        }

        // ---- text body ----
        let tc = get(cfg, "text_config", "config")?;
        let ts = "config.text_config";
        want_str(&tc, "model_type", ts, "ministral3")?;
        want_bool(&tc, "is_causal", ts, false)?;
        want_str(&tc, "hidden_act", ts, "silu")?;
        if tc.get("sliding_window").is_some_and(|w| !w.is_null()) {
            return Err(bad(format!(
                "{ts}.sliding_window is set (full attention is implemented)"
            )));
        }
        let d_model = u(&tc, "hidden_size", ts)?;
        let n_layer = u(&tc, "num_hidden_layers", ts)?;
        let n_heads = u(&tc, "num_attention_heads", ts)?;
        let n_kv_heads = u(&tc, "num_key_value_heads", ts)?;
        let head_dim = u(&tc, "head_dim", ts)?;
        let ffn = u(&tc, "intermediate_size", ts)?;
        let vocab = u(&tc, "vocab_size", ts)?;
        let rms_eps = f(&tc, "rms_norm_eps", ts)? as f32;
        if n_kv_heads == 0 || !n_heads.is_multiple_of(n_kv_heads) || head_dim != 128 {
            return Err(bad(format!(
                "{ts}: heads {n_heads} / kv {n_kv_heads} at head_dim {head_dim} (GQA at 128 is \
                 implemented)"
            )));
        }
        if !d_model.is_multiple_of(64) || !ffn.is_multiple_of(64) {
            return Err(bad(format!(
                "{ts}: width {d_model} / ffn {ffn} not multiples of 64"
            )));
        }
        let rp = get(&tc, "rope_parameters", ts)?;
        let rs = "config.text_config.rope_parameters";
        want_str(&rp, "rope_type", rs, "yarn")?;
        if rp
            .get("partial_rotary_factor")
            .and_then(|x| x.as_f64())
            .is_some_and(|p| p != 1.0)
        {
            return Err(bad(format!("{rs}.partial_rotary_factor != 1")));
        }
        let factor = f(&rp, "factor", rs)?;
        let mscale = rp.get("mscale").and_then(|x| x.as_f64());
        let mscale_all = rp.get("mscale_all_dim").and_then(|x| x.as_f64());
        // Transformers' get_mscale
        let gm = |scale: f64, m: f64| {
            if scale <= 1.0 {
                1.0
            } else {
                0.1 * m * scale.ln() + 1.0
            }
        };
        let attention_factor = match rp.get("attention_factor").and_then(|x| x.as_f64()) {
            Some(a) => a,
            None => match (mscale, mscale_all) {
                (Some(m), Some(ma)) if m != 0.0 && ma != 0.0 => gm(factor, m) / gm(factor, ma),
                _ => gm(factor, 1.0),
            },
        };
        let rope = Yarn {
            theta: f(&rp, "rope_theta", rs)?,
            factor,
            original_max_pos: u(&rp, "original_max_position_embeddings", rs)?,
            // `get("beta_fast") or 32`: a 0 also falls back
            beta_fast: rp
                .get("beta_fast")
                .and_then(|x| x.as_f64())
                .filter(|&b| b != 0.0)
                .unwrap_or(32.0),
            beta_slow: rp
                .get("beta_slow")
                .and_then(|x| x.as_f64())
                .filter(|&b| b != 0.0)
                .unwrap_or(1.0),
            attention_factor,
            truncate: rp.get("truncate").and_then(|x| x.as_bool()).unwrap_or(true),
        };
        // The query scaling: Ministral 3's attention reads these two keys of
        // rope_parameters (`llama_4_scaling_beta`, the original window)
        let l4_beta = f(&rp, "llama_4_scaling_beta", rs)? as f32;
        let l4_orig = rope.original_max_pos;

        // ---- picture tower ----
        let vc = get(cfg, "vision_config", "config")?;
        let vs = "config.vision_config";
        want_str(&vc, "model_type", vs, "pixtral")?;
        want_str(&vc, "hidden_act", vs, "silu")?;
        if u(&vc, "num_channels", vs)? != 3 {
            return Err(bad(format!("{vs}.num_channels != 3")));
        }
        let v_dim = u(&vc, "hidden_size", vs)?;
        let v_heads = u(&vc, "num_attention_heads", vs)?;
        let v_head_dim = u(&vc, "head_dim", vs)?;
        if v_heads * v_head_dim != v_dim || v_head_dim != 64 {
            return Err(bad(format!(
                "{vs}: {v_heads} heads of {v_head_dim} for width {v_dim} (hd 64 is implemented)"
            )));
        }
        let vrp = get(&vc, "rope_parameters", vs)?;
        let v_theta = f(&vrp, "rope_theta", "config.vision_config.rope_parameters")?;
        let patch = u(&vc, "patch_size", vs)?;

        // ---- pixels ----
        let ip = get(proc, "image_processor", "processor_config")?;
        let is = "processor_config.image_processor";
        want_str(&ip, "image_processor_type", is, "PixtralImageProcessor")?;
        want_str(
            proc,
            "processor_class",
            "processor_config",
            "PixtralProcessor",
        )?;
        if u(&ip, "resample", is)? != 3 {
            return Err(bad(format!("{is}.resample is not 3 (bicubic)")));
        }
        for k in ["do_resize", "do_rescale", "do_normalize", "do_convert_rgb"] {
            want_bool(&ip, k, is, true)?;
        }
        if u(&ip, "patch_size", is)? != patch || u(proc, "patch_size", "processor_config")? != patch
        {
            return Err(bad(format!(
                "{is}.patch_size disagrees with the tower's {patch}"
            )));
        }
        if u(proc, "spatial_merge_size", "processor_config")? != merge {
            return Err(bad("processor_config.spatial_merge_size disagrees".into()));
        }
        for (k, want) in [
            ("image_token", "[IMG]"),
            ("image_break_token", "[IMG_BREAK]"),
            ("image_end_token", "[IMG_END]"),
        ] {
            want_str(proc, k, "processor_config", want)?;
        }
        let longest_edge = ip
            .pointer("/size/longest_edge")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .ok_or_else(|| bad(format!("{is}.size missing longest_edge")))?;
        let three = |k: &str| -> Result<[f32; 3], StError> {
            let a = ip
                .get(k)
                .and_then(|x| x.as_array())
                .filter(|a| a.len() == 3)
                .ok_or_else(|| bad(format!("{is} missing 3-vector {k}")))?;
            let mut out = [0f32; 3];
            for (o, v) in out.iter_mut().zip(a) {
                *o = v
                    .as_f64()
                    .ok_or_else(|| bad(format!("{is}.{k} not numeric")))?
                    as f32;
            }
            Ok(out)
        };
        let mean = three("image_mean")?;
        let std = three("image_std")?;
        let rescale = f(&ip, "rescale_factor", is)? as f32;

        // ---- pooling + context ----
        if !pool
            .get("pooling_mode_mean_tokens")
            .or(pool
                .get("pooling_mode")
                .filter(|m| m.as_str() == Some("mean")))
            .is_some_and(|m| m.as_bool() == Some(true) || m.as_str() == Some("mean"))
        {
            return Err(bad("1_Pooling: only mean pooling is implemented".into()));
        }
        want_bool(pool, "include_prompt", "1_Pooling", true)?;
        let max_ctx = u(st, "max_seq_length", "sentence_bert_config")?;

        Ok(Self {
            kind,
            prefix: if kind == Kind::Rerank { "model." } else { "" },
            d_model,
            n_layer,
            n_heads,
            n_kv_heads,
            head_dim,
            ffn,
            vocab,
            rms_eps,
            rope,
            l4_beta,
            l4_orig,
            max_ctx,
            v_dim,
            v_layers: u(&vc, "num_hidden_layers", vs)?,
            v_heads,
            v_head_dim,
            v_ffn: u(&vc, "intermediate_size", vs)?,
            patch,
            v_theta,
            merge,
            image_token_id,
            longest_edge,
            mean,
            std,
            rescale,
        })
    }

    /// The text body's YaRN inverse frequencies, `head_dim / 2` of them, as
    /// Transformers computes them: f32 tensors (`torch.float`) end to end,
    /// the correction range in Python floats. Each step is one f32 op in the
    /// reference's order, so the values are its values.
    pub fn yarn_inv_freq(&self) -> Vec<f32> {
        let y = &self.rope;
        let dim = self.head_dim;
        let half = dim / 2;
        let corr = |rot: f64| {
            (dim as f64 * (y.original_max_pos as f64 / (rot * 2.0 * std::f64::consts::PI)).ln())
                / (2.0 * y.theta.ln())
        };
        let (mut low, mut high) = (corr(y.beta_fast), corr(y.beta_slow));
        if y.truncate {
            low = low.floor();
            high = high.ceil();
        }
        let low = low.max(0.0);
        let high = high.min((dim - 1) as f64);
        // linear_ramp_factor: `min == max` nudges max by 0.001
        let high = if low == high { high + 0.001 } else { high };
        (0..half)
            .map(|i| {
                let pos_freq = y.theta.powf((2 * i) as f64 / dim as f64) as f32;
                let extra = 1.0f32 / pos_freq;
                let inter = 1.0f32 / (y.factor as f32 * pos_freq);
                let ramp = ((i as f32 - low as f32) / (high as f32 - low as f32)).clamp(0.0, 1.0);
                let ef = 1.0f32 - ramp;
                inter * (1.0f32 - ef) + extra * ef
            })
            .collect()
    }

    /// The text rope's (cos, sin) tables `[positions][head_dim / 2]`: the
    /// angle is the reference's f32 product `pos * inv_freq`, its cosine and
    /// sine evaluated in f64 and rounded once (torch's f32 `cos` / `sin` sit
    /// within an ulp of that), times the attention factor.
    pub fn text_rope_tables(&self, positions: usize) -> (Vec<f32>, Vec<f32>) {
        let inv = self.yarn_inv_freq();
        let af = self.rope.attention_factor;
        let n = positions * inv.len();
        let (mut cos, mut sin) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for p in 0..positions {
            for &f in &inv {
                let a = (p as f32 * f) as f64;
                cos.push((a.cos() * af) as f32);
                sin.push((a.sin() * af) as f32);
            }
        }
        (cos, sin)
    }

    /// Pixtral's inverse frequencies re-ordered for the 2D rope: the even
    /// ones (turned by the patch row) then the odd ones (by the column),
    /// `v_head_dim / 2` in all.
    pub fn vision_inv_freq(&self) -> Vec<f32> {
        let dim = self.v_head_dim;
        let inv: Vec<f32> = (0..dim / 2)
            .map(|i| 1.0f32 / self.v_theta.powf((2 * i) as f64 / dim as f64) as f32)
            .collect();
        inv.iter()
            .step_by(2)
            .chain(inv.iter().skip(1).step_by(2))
            .copied()
            .collect()
    }

    /// One patch's rope angles at grid (`row`, `col`): `v_head_dim / 2` of
    /// them, the first half from the row, the second from the column - the
    /// layout `PixtralVisionRotaryEmbedding.recomposition_frequencies` builds,
    /// each the reference's f32 product.
    pub fn vision_rope_angles(&self, inv: &[f32], row: usize, col: usize, out: &mut [f32]) {
        let q = inv.len() / 2;
        for (k, o) in out.iter_mut().enumerate().take(inv.len()) {
            let p = if k < q { row } else { col };
            *o = p as f32 * inv[k];
        }
    }

    /// Llama 4's query factor at text position `pos`.
    pub fn l4_scale(&self, pos: usize) -> f32 {
        let fl = (pos / self.l4_orig) as f64;
        (1.0 + self.l4_beta as f64 * (1.0 + fl).ln()) as f32
    }

    /// The processor's resize target for a `w` x `h` picture, as (width,
    /// height): the floor-ratio downscale to the longest edge, then each
    /// side rounded UP to whole picture tokens (patch x merge) - Pixtral's
    /// `get_resize_output_image_size` with the processor's merged patch.
    pub fn picture_target(&self, w: usize, h: usize) -> Result<(usize, usize), String> {
        if w == 0 || h == 0 || w > 16384 || h > 16384 {
            return Err(format!(
                "a {w} x {h} picture: sides must be within 1..16384"
            ));
        }
        let le = self.longest_edge as f64;
        let ratio = (h as f64 / le).max(w as f64 / le);
        let (mut hh, mut ww) = (h, w);
        if ratio > 1.0 {
            hh = (h as f64 / ratio).floor() as usize;
            ww = (w as f64 / ratio).floor() as usize;
        }
        let unit = self.patch * self.merge;
        // `(x - 1) // unit + 1` tokens a side; a 0 side (a sliver) becomes one
        let side = |x: usize| (x.max(1) - 1) / unit + 1;
        Ok((side(ww) * unit, side(hh) * unit))
    }

    /// Picture tokens a `w` x `h` picture takes in a prompt: per merged row
    /// its tokens plus one `[IMG_BREAK]` (the last one `[IMG_END]`).
    pub fn picture_tokens(&self, w: usize, h: usize) -> Result<usize, String> {
        let (tw, th) = self.picture_target(w, h)?;
        let unit = self.patch * self.merge;
        Ok((tw / unit + 1) * (th / unit))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shipped files (the Embed and Rerank revisions the catalog pins),
    /// trimmed to the keys this reader consumes.
    fn shipped(arch: &str) -> [serde_json::Value; 4] {
        let mut cfg = serde_json::json!({
            "architectures": [arch], "model_type": "mistral3", "is_causal": false,
            "image_token_index": 10, "multimodal_projector_bias": false,
            "projector_hidden_act": "gelu", "spatial_merge_size": 2, "vision_feature_layer": -1,
            "text_config": {
                "model_type": "ministral3", "is_causal": false, "hidden_act": "silu",
                "hidden_size": 4096, "num_hidden_layers": 34, "num_attention_heads": 32,
                "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 14336,
                "vocab_size": 131072, "rms_norm_eps": 1e-5, "sliding_window": null,
                "rope_parameters": {
                    "apply_yarn_scaling": false, "beta_fast": 32.0, "beta_slow": 1.0,
                    "factor": 16.0, "llama_4_scaling_beta": 0.1, "mscale": 1.0,
                    "mscale_all_dim": 1.0, "original_max_position_embeddings": 16384,
                    "rope_theta": 1000000.0, "rope_type": "yarn", "type": "yarn"
                }
            },
            "vision_config": {
                "model_type": "pixtral", "hidden_act": "silu", "hidden_size": 1024,
                "num_attention_heads": 16, "head_dim": 64, "intermediate_size": 4096,
                "num_hidden_layers": 24, "num_channels": 3, "patch_size": 14,
                "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}
            }
        });
        if arch == ARCH_RERANK {
            cfg["id2label"] = serde_json::json!({"0": "LABEL_0"});
        }
        let proc = serde_json::json!({
            "image_break_token": "[IMG_BREAK]", "image_end_token": "[IMG_END]",
            "image_token": "[IMG]", "patch_size": 14, "processor_class": "PixtralProcessor",
            "spatial_merge_size": 2,
            "image_processor": {
                "do_convert_rgb": true, "do_normalize": true, "do_rescale": true,
                "do_resize": true, "image_mean": [0.48145466, 0.4578275, 0.40821073],
                "image_std": [0.26862954, 0.26130258, 0.27577711],
                "image_processor_type": "PixtralImageProcessor", "patch_size": 14,
                "resample": 3, "rescale_factor": 0.00392156862745098,
                "size": {"longest_edge": 1120}
            }
        });
        let st = serde_json::json!({"max_seq_length": 32768});
        let pool = serde_json::json!({"pooling_mode_mean_tokens": true, "include_prompt": true});
        [cfg, proc, st, pool]
    }

    fn read(v: &[serde_json::Value; 4]) -> Result<NemotronEmbedConfig, StError> {
        NemotronEmbedConfig::from_json(&v[0], &v[1], &v[2], &v[3])
    }

    #[test]
    fn shipped_configs_read_and_name_their_kind() {
        let e = read(&shipped(ARCH_EMBED)).unwrap();
        assert_eq!(e.kind, Kind::Embed);
        assert_eq!(e.prefix, "");
        assert_eq!(
            (e.d_model, e.n_layer, e.n_heads, e.n_kv_heads),
            (4096, 34, 32, 8)
        );
        assert_eq!(e.rope.attention_factor, 1.0);
        assert_eq!(e.max_ctx, 32768);
        let r = read(&shipped(ARCH_RERANK)).unwrap();
        assert_eq!(r.kind, Kind::Rerank);
        assert_eq!(r.prefix, "model.");
        // the ST pooling file of the reranker spells the mode as a string
        let mut v = shipped(ARCH_RERANK);
        v[3] = serde_json::json!({"pooling_mode": "mean", "include_prompt": true});
        assert!(read(&v).is_ok());
    }

    #[test]
    fn unimplemented_switches_are_refused_by_name() {
        let cases: [(&str, fn(&mut [serde_json::Value; 4])); 6] = [
            ("is_causal", |v| v[0]["is_causal"] = true.into()),
            ("sliding_window", |v| {
                v[0]["text_config"]["sliding_window"] = 4096.into()
            }),
            ("rope_type", |v| {
                v[0]["text_config"]["rope_parameters"]["rope_type"] = "default".into()
            }),
            ("resample", |v| {
                v[1]["image_processor"]["resample"] = 2.into()
            }),
            ("include_prompt", |v| v[3]["include_prompt"] = false.into()),
            ("vision_feature_layer", |v| {
                v[0]["vision_feature_layer"] = (-2).into()
            }),
        ];
        for (name, edit) in cases {
            let mut v = shipped(ARCH_EMBED);
            edit(&mut v);
            let e = read(&v).unwrap_err().to_string();
            assert!(e.contains(name), "{name}: {e}");
        }
    }

    #[test]
    fn yarn_matches_the_reference_ramp() {
        let c = read(&shipped(ARCH_EMBED)).unwrap();
        let inv = c.yarn_inv_freq();
        assert_eq!(inv.len(), 64);
        // correction range [20, 37]: below it the original frequency, above
        // it the frequency over the factor (16, so exact)
        let orig = |i: usize| 1.0f32 / 1e6f64.powf(i as f64 / 64.0) as f32;
        for (i, &f) in inv.iter().enumerate().take(21) {
            assert_eq!(f, orig(i), "{i}");
        }
        for (i, &f) in inv.iter().enumerate().skip(37) {
            assert_eq!(f, 1.0f32 / (16.0 * (1e6f64.powf(i as f64 / 64.0) as f32)));
        }
        assert!(inv[28] < orig(28) && inv[28] > orig(28) / 16.0);
        assert_eq!(c.l4_scale(16383), 1.0);
        assert_eq!(c.l4_scale(16384), (1.0 + 0.1 * 2f64.ln()) as f32);
    }

    #[test]
    fn picture_geometry_is_the_processors() {
        let c = read(&shipped(ARCH_EMBED)).unwrap();
        // ratio > 1: floor both sides, then up to whole 28-pixel tokens
        assert_eq!(c.picture_target(2480, 3508).unwrap(), (812, 1120));
        assert_eq!(c.picture_target(1000, 700).unwrap(), (1008, 700));
        // 1120 x 812 -> 40 tokens + a break a row, 29 rows
        assert_eq!(c.picture_tokens(1120, 812).unwrap(), 41 * 29);
        assert_eq!(c.picture_tokens(1, 1).unwrap(), 2);
        assert!(c.picture_target(0, 5).is_err());
        let inv = c.vision_inv_freq();
        assert_eq!(inv.len(), 32);
        assert_eq!(inv[0], 1.0);
        assert_eq!(inv[16], 1.0f32 / 1e4f64.powf(2.0 / 64.0) as f32);
    }
}
