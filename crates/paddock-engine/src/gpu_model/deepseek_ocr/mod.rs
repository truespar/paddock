//! DeepSeek-OCR family - `DeepSeek-OCR`, `DeepSeek-OCR-2`, `baidu/Unlimited-OCR`.
//! GGUF arch string `deepseek2-ocr`. Design and evidence:
//!
//! All three checkpoints share an identical language config - 12 layers, width
//! 1280, 10 q / 10 kv heads, 64 routed + 2 shared experts, top-6, vocab 129280 -
//! so one decoder covers the family. They differ on exactly four axes: the
//! second vision tower (CLIP-L for OCR/Unlimited, a qwen2-0.5b tower for OCR-2),
//! the projector input width that follows from it (2048 with the SAM concat vs
//! 896 without), the tile budget (6 vs 32), and whether R-SWA is on.
//!
//! Two things about this family are genuinely unlike anything else we serve, and
//! both are recorded on [`Hparams`] rather than left to the reader:
//!
//! 1. **It is not MLA.** The config says `use_mla: false` with
//!    `qk_nope_head_dim == qk_rope_head_dim == 0`, which makes the reference pick
//!    a plain Llama MHA rather than the DeepSeek-V2 attention its filename
//!    suggests. Everything here is ordinary q/k/v/o with RoPE.
//! 2. **R-SWA**, the reason "Unlimited" is in the name - see [`Hparams::rswa_window`].
//!
//! `baidu/Unlimited-OCR-2` keeps v1's tower (byte-identical) and decoder body and
//! replaces R-SWA with **SR-SWA** (its code calls the router DASH): every layer's
//! queries read a 128-token local window plus at most 12 router-selected 32-token
//! blocks with an additive log-prior bias, in prefill and decode alike, under a
//! page-2D rope (a token axis counted within each page, a page-parity axis). See
//! [`Dash`] for the parameters and their trained values.

pub mod batch;
pub mod chunked;
pub mod dash;
pub mod encode;
pub mod forward;
pub mod load;
pub mod multimodal;
pub mod prefix;
pub mod preprocess;
pub mod resample;
pub mod tiling;
pub mod vision;

pub use load::GpuDeepseekOcr;

/// A captured decode tick. Single-threaded on the engine's thread (the same
/// argument every other family's SendGraph makes).
pub(crate) struct SendGraph(pub(crate) crate::gpu::CapturedGraph);
// SAFETY: never accessed from two threads at once; see above.
unsafe impl Send for SendGraph {}

use paddock_models::gguf::Value;
use paddock_models::mapped::MappedGguf;

/// The GGUF `general.architecture` every member of this family converts to.
/// llama.cpp folds all three checkpoints onto it (`MODEL_ARCH.DEEPSEEK2OCR`),
/// so it is the routing key in the runner's `build_generator` match too.
pub const ARCH: &str = "deepseek2-ocr";

// No `Eq`: rms_eps / rope_freq_base are f32.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hparams {
    pub n_layer: usize,
    /// Model width (`embedding_length`), 1280 across the family.
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    /// Always derived as `n_embd / n_head`, never read from the file - see
    /// [`Hparams::from_gguf`] for why that is not a stylistic preference.
    pub head_dim: usize,
    pub n_vocab: usize,
    pub n_ctx_train: usize,
    /// Dense FFN width, used by the `first_k_dense` leading layers only.
    pub n_ff: usize,
    /// Per-routed-expert FFN width (896).
    pub n_ff_exp: usize,
    /// Layers `[0, first_k_dense)` are dense; the rest are MoE. 1 in practice.
    pub first_k_dense: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    /// Number of shared experts as the config counts them (2). Note the GGUF
    /// stores them PRE-MERGED into one `ffn_*_shexp` plane of width
    /// `n_shared_expert * n_ff_exp` (1792), so the graph runs one dense MLP and
    /// never iterates this - it is kept for the fit/VRAM estimate and for
    /// asserting the merged width is what we think it is.
    pub n_expert_shared: usize,
    pub rms_eps: f32,
    pub rope_freq_base: f32,
    /// R-SWA window, 128 on Unlimited-OCR and absent on the DeepSeek-OCR pair.
    ///
    /// This is **not** the sliding window gemma4 and laguna implement. There,
    /// the window slides over everything and old tokens scroll out. Here the
    /// prefill region - vision tokens plus prompt - stays GLOBALLY VISIBLE
    /// forever, and only the tokens the model *generates* live in a ring of
    /// `rswa_window` slots appended after that prefix, overwritten in place. So
    /// total KV per sequence is pinned at `prefill_len + rswa_window` no matter
    /// how long the output runs, which is what lets the model parse an
    /// arbitrarily long document at constant memory.
    ///
    /// Consequence worth stating where it will be read: the prefix region is a
    /// perfectly ordinary shareable prefix, so prefix caching stays on here.
    pub rswa_window: Option<usize>,
    /// SR-SWA (Unlimited-OCR-2): present iff the file carries the router keys.
    pub dash: Option<Dash>,
}

/// Unlimited-OCR-2's SR-SWA router ("DASH" in its code) and page-2D rope, as the
/// checkpoint's inference path resolves them. The file stores them verbatim from
/// the checkpoint's config (never the router module's rewritten live state).
///
/// Per query: the router scores every candidate key block `c` (32 keys) by
/// `q_r . kbar_c` - `q_r` / `kbar_c` being the router's own 128-wide projections,
/// `kbar_c` a learned-softmax summary of the block's router keys - takes the top
/// `entmax_window` (16), runs 1.5-entmax over them, and reads the first
/// `topk_blocks` (12) with bias `log(max(p, floor)) - mean` (over the read ones,
/// / `sigma`). The query attends to those blocks plus its `local_window` (128)
/// latest keys; the window wins where the two overlap. Remote blocks are those at
/// or before the query's block minus `local_blocks`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dash {
    pub block_size: usize,
    pub local_blocks: usize,
    pub local_window: usize,
    pub index_dim: usize,
    pub topk_blocks: usize,
    pub entmax_window: usize,
    pub sigma: f32,
    /// The router's entmax input scale `scale / sqrt(index_dim)`; one value for
    /// every layer (the loader refuses a file whose per-layer values differ).
    pub score_scale: f32,
    /// Prior floor before the log.
    pub floor: f32,
    /// Remote candidates are limited to image blocks (decode: the response
    /// page's; prefill: the query's own page, falling back to every block).
    pub image_only: bool,
    pub image_token_id: u32,
    pub page_token_id: u32,
    /// The page-count guard's stop token (the tokenizer's EOS).
    pub eos_token_id: u32,
    pub per_page_image_tokens: usize,
    /// Page-2D rope: token pairs `inv[j] = tok_theta^(-j / n_tok)` (`n_tok =
    /// head_dim/2 - page_pairs`) on the in-page token count, one page pair
    /// turning by `(page mod page_mod) * page_theta^0` radians.
    pub tok_theta: f32,
    pub page_theta: f32,
    pub page_mod: u32,
}

/// Every key this family reads, so a stale or foreign file fails by NAME.
mod key {
    pub const BLOCK_COUNT: &str = "deepseek2-ocr.block_count";
    pub const EMBD: &str = "deepseek2-ocr.embedding_length";
    pub const HEAD: &str = "deepseek2-ocr.attention.head_count";
    pub const HEAD_KV: &str = "deepseek2-ocr.attention.head_count_kv";
    pub const VOCAB: &str = "deepseek2-ocr.vocab_size";
    pub const CTX: &str = "deepseek2-ocr.context_length";
    pub const FF: &str = "deepseek2-ocr.feed_forward_length";
    pub const FF_EXP: &str = "deepseek2-ocr.expert_feed_forward_length";
    pub const LEAD_DENSE: &str = "deepseek2-ocr.leading_dense_block_count";
    pub const N_EXPERT: &str = "deepseek2-ocr.expert_count";
    pub const N_EXPERT_USED: &str = "deepseek2-ocr.expert_used_count";
    pub const N_EXPERT_SHARED: &str = "deepseek2-ocr.expert_shared_count";
    pub const RMS_EPS: &str = "deepseek2-ocr.attention.layer_norm_rms_epsilon";
    pub const ROPE_BASE: &str = "deepseek2-ocr.rope.freq_base";
    pub const SLIDING: &str = "deepseek2-ocr.attention.sliding_window";
    pub const DASH_BLOCK: &str = "deepseek2-ocr.dash.block_size";
    pub const DASH_LOCAL_BLOCKS: &str = "deepseek2-ocr.dash.local_blocks";
    pub const DASH_LOCAL_WINDOW: &str = "deepseek2-ocr.dash.local_window";
    pub const DASH_INDEX_DIM: &str = "deepseek2-ocr.dash.index_dim";
    pub const DASH_TOPK: &str = "deepseek2-ocr.dash.topk_blocks";
    pub const DASH_WINDOW: &str = "deepseek2-ocr.dash.entmax_window";
    pub const DASH_STAGE0: &str = "deepseek2-ocr.dash.stage0";
    pub const DASH_SIGMA: &str = "deepseek2-ocr.dash.sigma";
    pub const DASH_SCALE: &str = "deepseek2-ocr.dash.scale_per_layer";
    pub const DASH_FLOOR: &str = "deepseek2-ocr.dash.floor_per_layer";
    pub const DASH_LAYERS: &str = "deepseek2-ocr.dash.layers";
    pub const DASH_IMAGE_ONLY: &str = "deepseek2-ocr.dash.image_only";
    pub const IMAGE_TOKEN: &str = "deepseek2-ocr.image_token_id";
    pub const PAGE_TOKEN: &str = "deepseek2-ocr.page_token_id";
    pub const PAGE_TOKENS: &str = "deepseek2-ocr.per_page_image_tokens";
    pub const ROPE_TOK_THETA: &str = "deepseek2-ocr.page_rope.tok_theta";
    pub const ROPE_PAGE_THETA: &str = "deepseek2-ocr.page_rope.page_theta";
    pub const ROPE_PAGE_PAIRS: &str = "deepseek2-ocr.page_rope.page_pairs";
    pub const ROPE_PAGE_MOD: &str = "deepseek2-ocr.page_rope.page_mod";
    pub const ROPE_TOK_MODE: &str = "deepseek2-ocr.page_rope.tok_freq_mode";
    pub const EOS: &str = "tokenizer.ggml.eos_token_id";
}

#[derive(Debug, thiserror::Error)]
pub enum HparamsError {
    #[error("deepseek-ocr: missing GGUF key {0}")]
    MissingKey(String),
    #[error("deepseek-ocr: GGUF key {0} has the wrong type")]
    BadKey(String),
    #[error("deepseek-ocr: {0}")]
    Geometry(String),
}

fn u64_key(map: &MappedGguf, k: &str) -> Result<u64, HparamsError> {
    map.gguf()
        .metadata
        .get(k)
        .ok_or_else(|| HparamsError::MissingKey(k.to_owned()))?
        .as_u64()
        .ok_or_else(|| HparamsError::BadKey(k.to_owned()))
}

fn usize_key(map: &MappedGguf, k: &str) -> Result<usize, HparamsError> {
    Ok(u64_key(map, k)? as usize)
}

fn f32_key(map: &MappedGguf, k: &str, default: Option<f32>) -> Result<f32, HparamsError> {
    match map.gguf().metadata.get(k) {
        Some(Value::F32(f)) => Ok(*f),
        Some(Value::F64(f)) => Ok(*f as f32),
        Some(_) => Err(HparamsError::BadKey(k.to_owned())),
        None => default.ok_or_else(|| HparamsError::MissingKey(k.to_owned())),
    }
}

/// One f32 per layer, all equal (the [`Dash`] contract).
fn uniform_f32_array(map: &MappedGguf, k: &str, n: usize) -> Result<f32, HparamsError> {
    let Some(Value::Array(a)) = map.gguf().metadata.get(k) else {
        return Err(HparamsError::MissingKey(k.to_owned()));
    };
    let v: Option<Vec<f32>> = a.iter().map(Value::as_f32).collect();
    let v = v.ok_or_else(|| HparamsError::BadKey(k.to_owned()))?;
    if v.len() != n || v.iter().any(|&x| x != v[0]) {
        return Err(HparamsError::Geometry(format!(
            "{k}: {v:?} - one equal value per layer is what this lane serves"
        )));
    }
    Ok(v[0])
}

/// The file is Unlimited-OCR-2 (it carries the SR-SWA router keys). The
/// runner's prompt layout (`<PAGE><image>` a page) and request vocabulary key
/// off this before any weight loads; [`Hparams::from_gguf`] then validates the
/// keys in full.
pub fn is_unlimited_ocr2(gguf: &paddock_models::gguf::GgufFile) -> bool {
    gguf.metadata.contains_key(key::DASH_BLOCK)
}

impl Dash {
    /// Present iff the file has the router keys; every one of them is then
    /// required, and the shapes this lane implements are checked by name.
    fn from_gguf(
        map: &MappedGguf,
        n_layer: usize,
        head_dim: usize,
    ) -> Result<Option<Self>, HparamsError> {
        let md = &map.gguf().metadata;
        if !md.contains_key(key::DASH_BLOCK) {
            return Ok(None);
        }
        let geo = |m: String| Err(HparamsError::Geometry(m));
        if md.get(key::DASH_STAGE0).and_then(Value::as_str) != Some("head_cls") {
            return geo("dash.stage0 must be head_cls (the block-summary router)".into());
        }
        if md.get(key::ROPE_TOK_MODE).and_then(Value::as_str) != Some("resample") {
            return geo("page_rope.tok_freq_mode must be resample".into());
        }
        if usize_key(map, key::ROPE_PAGE_PAIRS)? != 1 {
            return geo("page_rope.page_pairs must be 1".into());
        }
        match md.get(key::DASH_LAYERS) {
            Some(Value::Array(a))
                if a.len() == n_layer && a.iter().all(|v| v.as_u64() == Some(1)) => {}
            _ => return geo("dash.layers: every layer must be a router layer".into()),
        }
        let index_dim = usize_key(map, key::DASH_INDEX_DIM)?;
        if index_dim != head_dim {
            return geo(format!(
                "dash.index_dim {index_dim} != head_dim {head_dim} (shared rope)"
            ));
        }
        let block_size = usize_key(map, key::DASH_BLOCK)?;
        let local_blocks = usize_key(map, key::DASH_LOCAL_BLOCKS)?;
        let local_window = usize_key(map, key::DASH_LOCAL_WINDOW)?;
        if local_window != local_blocks * block_size {
            return geo(format!(
                "dash.local_window {local_window} != {local_blocks} blocks x {block_size}"
            ));
        }
        let scale = uniform_f32_array(map, key::DASH_SCALE, n_layer)?;
        Ok(Some(Dash {
            block_size,
            local_blocks,
            local_window,
            index_dim,
            topk_blocks: usize_key(map, key::DASH_TOPK)?,
            entmax_window: usize_key(map, key::DASH_WINDOW)?,
            sigma: f32_key(map, key::DASH_SIGMA, None)?,
            score_scale: scale / (index_dim as f32).sqrt(),
            floor: uniform_f32_array(map, key::DASH_FLOOR, n_layer)?,
            image_only: matches!(md.get(key::DASH_IMAGE_ONLY), Some(Value::Bool(true))),
            image_token_id: u64_key(map, key::IMAGE_TOKEN)? as u32,
            page_token_id: u64_key(map, key::PAGE_TOKEN)? as u32,
            eos_token_id: u64_key(map, key::EOS)? as u32,
            per_page_image_tokens: usize_key(map, key::PAGE_TOKENS)?,
            tok_theta: f32_key(map, key::ROPE_TOK_THETA, None)?,
            page_theta: f32_key(map, key::ROPE_PAGE_THETA, None)?,
            page_mod: u64_key(map, key::ROPE_PAGE_MOD)? as u32,
        }))
    }
}

impl Hparams {
    pub fn from_gguf(map: &MappedGguf) -> Result<Self, HparamsError> {
        let n_embd = usize_key(map, key::EMBD)?;
        let n_head = usize_key(map, key::HEAD)?;
        if n_head == 0 || n_embd % n_head != 0 {
            return Err(HparamsError::Geometry(format!(
                "embedding_length {n_embd} is not divisible by head_count {n_head}"
            )));
        }
        // head_dim is COMPUTED, and that is load-bearing rather than tidy.
        // llama.cpp's converter writes `deepseek2-ocr.rope.dimension_count = 0`
        // for this family because it derives that key from the config's
        // `qk_rope_head_dim`, which is 0 precisely because the model is not MLA.
        // Trusting the file there yields a rope dimension of zero - a silently
        // dead RoPE. The real value is 1280/10 = 128.
        let head_dim = n_embd / n_head;

        let n_expert = usize_key(map, key::N_EXPERT)?;
        let n_expert_used = usize_key(map, key::N_EXPERT_USED)?;
        if n_expert_used == 0 || n_expert_used > n_expert {
            return Err(HparamsError::Geometry(format!(
                "expert_used_count {n_expert_used} out of range for expert_count {n_expert}"
            )));
        }

        let hp = Hparams {
            n_layer: usize_key(map, key::BLOCK_COUNT)?,
            n_embd,
            n_head,
            n_head_kv: usize_key(map, key::HEAD_KV)?,
            head_dim,
            n_vocab: usize_key(map, key::VOCAB)?,
            n_ctx_train: usize_key(map, key::CTX)?,
            n_ff: usize_key(map, key::FF)?,
            n_ff_exp: usize_key(map, key::FF_EXP)?,
            first_k_dense: usize_key(map, key::LEAD_DENSE)?,
            n_expert,
            n_expert_used,
            n_expert_shared: usize_key(map, key::N_EXPERT_SHARED)?,
            rms_eps: f32_key(map, key::RMS_EPS, Some(1e-6))?,
            // The family's configs declare no rope theta; the reference's
            // DeepseekV2RotaryEmbedding default (10000) is the trained value.
            rope_freq_base: f32_key(map, key::ROPE_BASE, Some(10_000.0))?,
            // Absent on the DeepSeek-OCR pair (`sliding_window_size: null`),
            // 128 on Unlimited-OCR. Absent means no ring: plain causal decode.
            rswa_window: match map.gguf().metadata.get(key::SLIDING) {
                Some(v) => match v.as_u64() {
                    Some(0) | None => None,
                    Some(w) => Some(w as usize),
                },
                None => None,
            },
            dash: None,
        };
        let hp = Hparams {
            dash: Dash::from_gguf(map, hp.n_layer, hp.head_dim)?,
            ..hp
        };
        if hp.dash.is_some() && hp.rswa_window.is_some() {
            return Err(HparamsError::Geometry(
                "a router file must not also declare the v1 R-SWA ring".into(),
            ));
        }
        if hp.first_k_dense > hp.n_layer {
            return Err(HparamsError::Geometry(format!(
                "leading_dense_block_count {} exceeds block_count {}",
                hp.first_k_dense, hp.n_layer
            )));
        }
        Ok(hp)
    }

    /// Width of the merged shared-expert plane the GGUF actually stores
    /// (`ffn_*_shexp`): the two shared experts arrive pre-merged, so the graph
    /// runs one dense MLP of this width rather than looping experts.
    pub fn shexp_ff(&self) -> usize {
        self.n_expert_shared * self.n_ff_exp
    }

    /// Is this layer a MoE layer? Layers below `first_k_dense` are plain dense
    /// MLPs of width `n_ff`; the rest route.
    pub fn is_moe_layer(&self, layer: usize) -> bool {
        layer >= self.first_k_dense && self.n_expert > 0
    }

    /// Per-sequence KV rows once decoding is in steady state. R-SWA is the whole
    /// point of the family: the answer does not grow with the number of tokens
    /// generated, only with how much was prefilled. Used by the fit estimate.
    pub fn kv_rows(&self, prefill_len: usize, max_new: usize) -> usize {
        if let Some(d) = &self.dash {
            // SR-SWA keeps the prompt's blocks (up to the end of the block that
            // straddles the prompt end) plus a ring the size of the local window
            let keep = prefill_len.next_multiple_of(d.block_size);
            return keep + d.local_window.min(max_new);
        }
        match self.rswa_window {
            Some(w) => prefill_len + w.min(max_new),
            None => prefill_len + max_new,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hp() -> Hparams {
        Hparams {
            n_layer: 12,
            n_embd: 1280,
            n_head: 10,
            n_head_kv: 10,
            head_dim: 128,
            n_vocab: 129_280,
            n_ctx_train: 32_768,
            n_ff: 6848,
            n_ff_exp: 896,
            first_k_dense: 1,
            n_expert: 64,
            n_expert_used: 6,
            n_expert_shared: 2,
            rms_eps: 1e-6,
            rope_freq_base: 10_000.0,
            rswa_window: Some(128),
            dash: None,
        }
    }

    #[test]
    fn shared_experts_are_one_merged_plane() {
        // The GGUF stores ffn_*_shexp at 1792 = 2 x 896. If this ever stops
        // matching the tensor's real width the graph is looping the wrong shape.
        assert_eq!(hp().shexp_ff(), 1792);
    }

    #[test]
    fn layer_zero_is_dense_and_the_rest_route() {
        let h = hp();
        assert!(
            !h.is_moe_layer(0),
            "first_k_dense_replace=1 makes layer 0 dense"
        );
        for l in 1..h.n_layer {
            assert!(h.is_moe_layer(l), "layer {l} must route");
        }
    }

    /// The property the whole family exists for: output length does not move KV.
    #[test]
    fn rswa_pins_kv_regardless_of_output_length() {
        let h = hp();
        let prefill = 907; // the reference A4 test page's prompt length
        assert_eq!(
            h.kv_rows(prefill, 8),
            prefill + 8,
            "below the ring, KV still grows"
        );
        assert_eq!(h.kv_rows(prefill, 128), prefill + 128);
        assert_eq!(
            h.kv_rows(prefill, 100_000),
            prefill + 128,
            "ring must cap it"
        );

        // Without a window (the DeepSeek-OCR pair) it is ordinary causal growth.
        let plain = Hparams {
            rswa_window: None,
            ..h
        };
        assert_eq!(plain.kv_rows(prefill, 100_000), prefill + 100_000);
    }

    #[test]
    fn head_dim_is_computed_not_read() {
        // 1280/10 = 128, while the file's rope.dimension_count says 0.
        assert_eq!(hp().n_embd / hp().n_head, 128);
    }
}
