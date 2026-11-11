//! Whisper (encoder-decoder ASR, the whisper-large-v3 geometry) - the Nordic
//! lane's family.
//! KB-Whisper (sv), NB-Whisper (no) and Røst-whisper (da) are all fine-tunes
//! of this one architecture, served from our GGUF conversion
//! (our whisper converter - llama.cpp's converter cannot do
//! whisper, so the schema is paddock's own; tensor names keep the HF spelling
//! minus the "model." prefix, which is the vocabulary this loader speaks).
//!
//! Geometry (whisper-large-v3 class, all stamped in metadata): 128-bin
//! log-mel -> conv stem (k3/s1 then k3/s2, both +bias+GELU) -> sinusoid
//! position table (stored, [1500, 1280]) -> 32-layer full-attention encoder
//! (d1280, 20 heads × hd64, erf-GELU 5120 MLP, pre-LN with biases) ->
//! 32-layer decoder: causal self-attention over ≤448 LEARNED positions plus
//! CROSS-ATTENTION over the encoder output - the one mechanism no served
//! family has yet. Cross K/V are computed once per request from the ≤1500
//! encoder frames and stay static for the whole decode (attention with a
//! fixed-length precomputed KV and no causal mask). Whisper quirks the
//! structs pin: k_proj has no bias anywhere (q/v/out all do), and the head
//! (`proj_out`) is tied to `embed_tokens` upstream - KB-Whisper materializes
//! the duplicate plane, NB-Whisper and Røst omit it, so the loader reads the
//! head from whichever the file carries (the tie makes them the same bytes).
//!
//! This module is the loader + resident-weights skeleton: every plane
//! device-resident at the file's own f16 (activations will accumulate in
//! f32 - the qwen3_asr audio-tower chassis), hparams/mel/decode-contract
//! tokens parsed and validated. The encoder graph is, the decoder
//! + transcription serving.
//!
//! The decoder prompt contract the ids below serve is
//! `<|startoftranscript|><|{lang}|><|transcribe|><|notimestamps|>`; the
//! runner owns prompt construction (its tokenizer resolves `<|sv|>`
//! directly, and the authoritative checkpoint map rides the GGUF as
//! `whisper.lang_to_id_json`).

use std::sync::Arc;

use cudarc::driver::CudaSlice;

use crate::gpu::{GpuExecutor, HalfTensor};
use crate::gpu_model::gpt_oss::GpuModelError;

pub mod align;
mod backend;
mod decoder;
mod encoder;
mod load;
pub mod segments;
mod timing;

pub use decoder::DecodeBatch;
pub(crate) use encoder::EncScratch;
pub use encoder::EncoderOutput;
pub use segments::{Segment, split_segments, ts_state};

use crate::whisper::posterior_over;
pub use crate::whisper::{LangProb, TimeScale, join_windows};

/// A captured decode graph. `CapturedGraph` is not `Send` by construction
/// (CUDA handles are context-bound), but the transcriber owns the model on
/// one dedicated thread for exactly that reason - same rationale as
/// granite/gemma4/qwen35.
pub(crate) struct SendGraph(pub(crate) crate::gpu::CapturedGraph);
// SAFETY: the graph never leaves the transcriber thread that captured it.
unsafe impl Send for SendGraph {}

/// LayerNorm parameters - whisper is classic LN with bias everywhere.
pub(crate) struct LayerNormP {
    pub w: CudaSlice<f32>,
    pub b: CudaSlice<f32>,
}

/// The decoder's cross-attention planes (its pre-LN rides along - every
/// whisper attention is pre-normed). Only q and o live here: the per-layer
/// cross wk/wv/bv are consumed at load into the model-level LAYER-BATCHED
/// planes (`cross_wk_all`/`cross_wv_all`/`cross_bv_all`) - every
/// layer's cross K/V projection reads the same encoder states, so admission
/// runs them as one GEMM per plane set and the per-layer copies would be
/// 210 MB of dead weight.
pub(crate) struct AttnP {
    pub ln: LayerNormP,
    pub wq: HalfTensor,
    pub bq: CudaSlice<f32>,
    pub wo: HalfTensor,
    pub bo: CudaSlice<f32>,
}

/// The erf-GELU MLP (fc1 -> gelu -> fc2) with its pre-LN
/// (`final_layer_norm` in HF spelling).
pub(crate) struct MlpP {
    pub ln: LayerNormP,
    pub fc1_w: HalfTensor,
    pub fc1_b: CudaSlice<f32>,
    pub fc2_w: HalfTensor,
    pub fc2_b: CudaSlice<f32>,
}

/// A self-attention block whose q/k/v planes are MERGED into one
/// `[d_model, 3*d_model]` GEMM plane at load (decoder,
/// encoder).
///
/// Decode runs this projection once per token per layer at one row, where
/// three separate 1280-out GEMVs are three weight streams the memory system
/// sees as three unrelated 3.3 MB reads. Concatenating on the output axis is
/// free - GGUF planes are row-major over out_dim, so the merge is three
/// copies at load - and it turns them into one 9.8 MB stream plus one split
/// kernel that also appends K/V to the slot cache. The encoder merges for a
/// different reason (measured): at 1500x1280 each split GEMM makes
/// so few 256-row tiles that half the tc5p clusters idle - fused, the three
/// run in 19.09us where separate they cost 3x12.60.
pub(crate) struct SelfAttnP {
    pub ln: LayerNormP,
    /// q|k|v concatenated on the out axis, dims [d_model, 3*d_model]
    pub wqkv: HalfTensor,
    pub bq: CudaSlice<f32>,
    pub bv: CudaSlice<f32>,
    pub wo: HalfTensor,
    pub bo: CudaSlice<f32>,
}

pub(crate) struct EncLayer {
    pub attn: SelfAttnP,
    pub mlp: MlpP,
}

pub(crate) struct DecLayer {
    pub self_attn: SelfAttnP,
    pub cross_attn: AttnP,
    pub mlp: MlpP,
}

impl LayerNormP {
    fn bytes(&self) -> usize {
        (self.w.len() + self.b.len()) * 4
    }
}

impl AttnP {
    fn bytes(&self) -> usize {
        self.ln.bytes() + self.wq.bytes() + self.wo.bytes() + (self.bq.len() + self.bo.len()) * 4
    }
}

impl MlpP {
    fn bytes(&self) -> usize {
        self.ln.bytes()
            + self.fc1_w.bytes()
            + self.fc2_w.bytes()
            + (self.fc1_b.len() + self.fc2_b.len()) * 4
    }
}

impl SelfAttnP {
    fn bytes(&self) -> usize {
        self.ln.bytes()
            + self.wqkv.bytes()
            + self.wo.bytes()
            + (self.bq.len() + self.bv.len() + self.bo.len()) * 4
    }
}

impl EncLayer {
    fn bytes(&self) -> usize {
        self.attn.bytes() + self.mlp.bytes()
    }
}

impl DecLayer {
    fn bytes(&self) -> usize {
        self.self_attn.bytes() + self.cross_attn.bytes() + self.mlp.bytes()
    }
}

// The geometry/mel fields are consumed by the encoder graph and the
// decoder; until those land the loader validating + holding them is
// the whole story.
#[allow(dead_code)]
pub(crate) struct Hparams {
    pub d_model: usize,
    pub n_enc_heads: usize,
    pub enc_ffn: usize,
    pub n_dec_heads: usize,
    pub dec_ffn: usize,
    pub head_dim: usize,
    pub n_vocab: usize,
    /// Encoder frame capacity - 1500 = 30 s at the conv stem's 2× time
    /// downsample (`max_source_positions`).
    pub n_audio_ctx: usize,
    /// Decoder position capacity actually being served: the trained learned
    /// table (`max_target_positions`, 448) possibly narrowed by --max-ctx.
    pub n_text_ctx: usize,
    /// LayerNorm epsilon. The HF config declares none (torch nn.LayerNorm
    /// default) - 1e-5, pinned in the loader.
    pub eps: f32,
}

/// Mel-extractor parameters, from the checkpoint's own preprocessor config
/// (never hardcoded - rides the GGUF). Consumed by the earlier frontend
/// parameterization of `crate::audio`.
#[allow(dead_code)]
pub(crate) struct MelSpec {
    pub bins: usize,
    pub n_fft: usize,
    pub hop: usize,
    /// window length in seconds (30) - mel frames per window = chunk_s*sr/hop
    pub chunk_s: usize,
    pub sr: usize,
}

/// Decode-contract token ids, from the checkpoint's own generation config
/// (fine-tunes could re-map them; these are stamped per-file, never assumed).
/// Prompt construction consumes the task/timestamp ids.
#[allow(dead_code)]
pub(crate) struct DecodeTokens {
    pub sot: u32,
    pub eot: u32,
    pub no_timestamps: u32,
    pub transcribe: u32,
    /// absent on transcribe-only fine-tunes
    pub translate: Option<u32>,
    /// `<|nospeech|>` - the token whose probability at the first decode step
    /// is OpenAI's `no_speech_prob`.
    pub nospeech: u32,
    /// `<|startofprev|>` - opens the CONTEXT pre-roll that can precede
    /// `<|startoftranscript|>`. Whisper was trained with the previous
    /// segment's text there, which is what makes it a vocabulary/style hint
    /// (the API's `prompt`) rather than an instruction.
    pub sot_prev: u32,
    /// `<|startoflm|>` - a prompt-side token with no decode role here; it is
    /// read so it can be suppressed like the rest of the prompt block.
    pub sot_lm: u32,
    /// `<|0.00|>`, the first of whisper's 1501 timestamp tokens. Everything at
    /// or above it is a timestamp; `(id - here) * time_precision` is when.
    pub timestamp_begin: u32,
}

impl DecodeTokens {
    /// The ids greedy decoding may never emit, suppressed before the softmax at
    /// every sampled step: the reference's `SuppressTokens` (openai/whisper
    /// `decoding.py` `_get_suppress_tokens`: transcribe, translate, sot,
    /// sot_prev, sot_lm, no_speech) plus `<|notimestamps|>`, which whisper.cpp
    /// kills unconditionally and the timestamp grammar kills in its own mode.
    /// vLLM applies none of this and lets a near-empty window come back as a
    /// confident `<|nospeech|>` token - which is not transcript, and what the
    /// word-timing pass rightly refuses. Text, timestamps and `<|eot|>` are the
    /// only things a transcript may hold.
    pub fn suppressed(&self) -> Vec<u32> {
        let mut ids = vec![
            self.sot,
            self.sot_lm,
            self.sot_prev,
            self.nospeech,
            self.no_timestamps,
            self.transcribe,
        ];
        ids.extend(self.translate);
        ids.sort_unstable();
        ids.dedup();
        ids
    }
}

// `exec`/`mel` are held for the encoder graph - the loader skeleton
// only builds and ledgers them.
#[allow(dead_code)]
pub struct GpuWhisper {
    pub(crate) exec: Arc<GpuExecutor>,
    pub(crate) hp: Hparams,
    pub(crate) mel: MelSpec,
    pub(crate) tokens: DecodeTokens,
    // ---- encoder ----
    /// conv1: k3/s1/p1, mel bins -> d_model, as a [3*bins, d_model] GEMM
    /// plane (k-axis permuted to tap-major at load - see `load::conv1d`).
    pub(crate) conv1_w: HalfTensor,
    pub(crate) conv1_b: CudaSlice<f32>,
    /// conv2: k3/s2/p1, d_model -> d_model - halves time: 3000 mel frames ->
    /// 1500 encoder positions. Same [3*d_model, d_model] plane shape.
    pub(crate) conv2_w: HalfTensor,
    pub(crate) conv2_b: CudaSlice<f32>,
    /// stored sinusoid table [n_audio_ctx, d_model], resident f32 - added
    /// whole to the conv output (unlike qwen3_asr's 13-row per-chunk reset,
    /// every request reads up to all 1500 rows).
    pub(crate) enc_pos: CudaSlice<f32>,
    pub(crate) enc_layers: Vec<EncLayer>,
    pub(crate) enc_ln: LayerNormP,
    // ---- decoder ----
    /// token embedding table, row-copy source ([d_model, n_vocab]). Held at
    /// f32 (not the f16 the file ships) because the decode step reads one
    /// row per token straight into the activation buffer - a widen op on the
    /// way would cost a kernel per token to save 133 MB on a 3 GB model.
    pub(crate) tok_embd: crate::gpu::DeviceTensor,
    /// LEARNED positions [n_text_ctx(≤448), d_model], resident f32.
    pub(crate) dec_pos: CudaSlice<f32>,
    pub(crate) dec_layers: Vec<DecLayer>,
    /// LAYER-BATCHED cross-attention K/V projections: every
    /// decoder layer's cross wk (and wv) reads the same encoder states, so
    /// admission runs one `[d_model, n_layer*d_model]` GEMM per plane set
    /// instead of 64 launches that each idle half the tc5p clusters
    /// (measured 806us -> 270 per window). Layer li's rows sit at
    /// `[li*d_model, (li+1)*d_model)`; `cross_bv_all` is the matching
    /// concatenated v bias (k has none - architecture).
    pub(crate) cross_wk_all: HalfTensor,
    pub(crate) cross_wv_all: HalfTensor,
    pub(crate) cross_bv_all: CudaSlice<f32>,
    pub(crate) dec_ln: LayerNormP,
    /// lm head (`proj_out.weight` - the file's own plane; tied upstream).
    pub(crate) head: HalfTensor,
    /// (bare code, token id) for every language the checkpoint declares,
    /// from its own `lang_to_id` map - language forcing and whisper's own
    /// detection both resolve through this, never through a baked table.
    pub(crate) langs: Vec<(String, u32)>,
    /// How far into a window the first timestamp may land, as an offset from
    /// `<|0.00|>`. OpenAI's `max_initial_timestamp` is 1.0 s and this is that
    /// in timestamp steps (50 at the usual 0.02 s). It exists because a model
    /// left free to open a window at 12 s has effectively dropped the first
    /// twelve seconds without saying so.
    pub(crate) max_initial_ts: u32,
    /// KV cache element type for both the cross and self planes. f16 is the
    /// default and the reference's class; `set_kv_dtype` flips it before
    /// `prepare_batch` sizes the pool.
    pub(crate) kv_dtype: crate::gpu::KvDtype,
    /// One window's encoder working set, allocated on the first encode and
    /// reused forever after (it used to be ~135 MB of
    /// `cudaMalloc`/free per window, on the serving thread).
    pub(crate) enc: Option<EncScratch>,
    /// The decode slot pool, once `prepare_batch` has sized it.
    pub(crate) decode: Option<DecodeBatch>,
    pub(crate) weights_bytes: u64,
}

impl GpuWhisper {
    /// The honest VRAM ledger: every resident plane, table and vector this
    /// struct holds (f16 planes at 2 B/elt, f32 vectors/tables at 4).
    fn resident_bytes(&self) -> u64 {
        let enc: usize = self.enc_layers.iter().map(EncLayer::bytes).sum();
        let dec: usize = self.dec_layers.iter().map(DecLayer::bytes).sum();
        (enc + dec
            + self.conv1_w.bytes()
            + self.conv2_w.bytes()
            + (self.conv1_b.len() + self.conv2_b.len()) * 4
            + (self.enc_pos.len() + self.dec_pos.len()) * 4
            + self.cross_wk_all.bytes()
            + self.cross_wv_all.bytes()
            + self.cross_bv_all.len() * 4
            + self.enc_ln.bytes()
            + self.dec_ln.bytes()
            + self.tok_embd.element_count() * 4
            + self.head.bytes()) as u64
    }

    pub fn vocab(&self) -> usize {
        self.hp.n_vocab
    }

    pub fn weights_bytes(&self) -> u64 {
        self.weights_bytes
    }

    /// Device bytes this process holds live (weights + the encoder/decoder
    /// state `prepare_batch` allocates) - the `model_mem` line of
    /// `/api/stats`. `weights_bytes` above stays exact across it because it
    /// is a per-tensor sum, not a snapshot.
    pub fn device_mem_used(&self) -> Option<u64> {
        self.exec.process_mem_used()
    }

    pub fn n_layers(&self) -> (usize, usize) {
        (self.enc_layers.len(), self.dec_layers.len())
    }

    /// (sot, eot) - the ids the serving loop starts and stops on.
    pub fn contract_tokens(&self) -> (u32, u32) {
        (self.tokens.sot, self.tokens.eot)
    }

    /// `<|nospeech|>` - the token whose probability at the first decode step
    /// is `no_speech_prob`. Exposed so a probe can check the value against
    /// the raw logits row.
    pub fn nospeech_token(&self) -> u32 {
        self.tokens.nospeech
    }

    /// `<|startofprev|>` - what a context prompt is fed behind.
    pub fn sot_prev_token(&self) -> u32 {
        self.tokens.sot_prev
    }

    /// Timestamp-token geometry for this checkpoint (see `TimeScale`).
    pub fn time_scale(&self) -> TimeScale {
        TimeScale {
            begin: self.tokens.timestamp_begin,
            // the conv stem halves time, so one encoder frame - and one
            // timestamp step - is two mel hops
            precision: 2.0 * self.mel.hop as f32 / self.mel.sr as f32,
            window_s: self.mel.chunk_s as f32,
        }
    }

    /// The decoder context actually served - one window's output can never
    /// exceed it, and the scheduler stops a slot one step short of it.
    pub fn text_ctx(&self) -> usize {
        self.hp.n_text_ctx
    }

    /// Pick the KV cache element type. Must be called before `prepare_batch`
    /// - the pool is sized in bytes for the width chosen here, and the
    ///   captured admission graph addresses those exact buffers.
    pub fn set_kv_dtype(&mut self, dtype: crate::gpu::KvDtype) {
        if self.decode.is_some() && dtype != self.kv_dtype {
            // sizing already happened; force a rebuild rather than
            // reinterpret half-width planes as full-width ones
            self.decode = None;
            if let Some(sc) = self.enc.as_mut() {
                sc.forget_graph();
            }
        }
        self.kv_dtype = dtype;
    }

    /// Encoder geometry: (frames, d_model) per 30 s window.
    pub fn encoder_shape(&self) -> (usize, usize) {
        (self.hp.n_audio_ctx, self.hp.d_model)
    }

    /// Copy encoder states back to the host - what the oracle gate and the
    /// encode example read; serving keeps them on device.
    pub fn states_to_host(&self, out: &EncoderOutput) -> Result<Vec<f32>, GpuModelError> {
        Ok(self
            .exec
            .to_host_len(&out.states, out.n_frames * self.hp.d_model)?)
    }
}

#[cfg(test)]
mod tests {
    use super::join_windows;

    #[test]
    fn windows_join_with_exactly_one_space() {
        // the Danish gate's failure: Røst ends one window on "fødselsdag"
        // and opens the next with "Øerne" carrying no leading space
        let parts = ["Jesu fødselsdag", "Øerne ligger i havet"];
        assert_eq!(
            join_windows(&parts, "da"),
            "Jesu fødselsdag Øerne ligger i havet"
        );
        // and the opposite case: whisper's usual leading space must not
        // become a double space
        let spaced = [" Jesu fødselsdag", " Øerne ligger i havet"];
        assert_eq!(
            join_windows(&spaced, "da"),
            "Jesu fødselsdag Øerne ligger i havet"
        );
    }

    #[test]
    fn space_free_scripts_join_bare() {
        let parts = ["\u{4eca}\u{5929}", "\u{5929}\u{6c14}\u{5f88}\u{597d}"];
        assert_eq!(
            join_windows(&parts, "zh"),
            "\u{4eca}\u{5929}\u{5929}\u{6c14}\u{5f88}\u{597d}"
        );
        assert_eq!(
            join_windows(&parts, "sv"),
            "\u{4eca}\u{5929} \u{5929}\u{6c14}\u{5f88}\u{597d}"
        );
    }

    #[test]
    fn empty_windows_leave_no_seam() {
        // a silent tail window decodes to nothing - it must not leave a
        // trailing space or a double separator behind
        let parts = ["hej", "   ", "då"];
        assert_eq!(join_windows(&parts, "sv"), "hej då");
        assert_eq!(join_windows::<&str>(&[], "sv"), "");
    }
}

#[cfg(test)]
mod posterior_tests {
    use super::*;

    fn langs(codes: &[&str]) -> Vec<(String, u32)> {
        // ids deliberately not 0..n and not in code order: a checkpoint's
        // language tokens sit wherever its vocab put them, and indexing the
        // logits by position instead of by id is the bug this guards
        codes
            .iter()
            .enumerate()
            .map(|(i, c)| ((*c).to_owned(), (50259 + i) as u32))
            .collect()
    }

    fn logits(at: &[(u32, f32)], vocab: usize) -> Vec<f32> {
        let mut v = vec![f32::NEG_INFINITY; vocab];
        for (id, l) in at {
            v[*id as usize] = *l;
        }
        v
    }

    /// The distribution is over the LANGUAGE tokens only - whisper's own
    /// normalisation - so it sums to 1 whatever the rest of the vocab holds.
    #[test]
    fn the_posterior_normalises_over_the_language_tokens_alone() {
        let l = langs(&["en", "sv", "de"]);
        // a huge logit somewhere else in the vocab must not enter the sum
        let mut row = logits(&[(50259, 1.0), (50260, 3.0), (50261, 2.0)], 51865);
        row[7] = 99.0;
        let post = posterior_over(&l, &row).expect("three languages");
        let total: f32 = post.iter().map(|p| p.p).sum();
        assert!((total - 1.0).abs() < 1e-5, "sums to {total}");
        assert_eq!(post[0].code, "sv");
        assert_eq!(post[1].code, "de");
        assert_eq!(post[2].code, "en");
        // softmax over (1,3,2): e^0/(e^0+e^-1+e^-2) for the winner
        let want = 1.0 / (1.0 + (-1.0f32).exp() + (-2.0f32).exp());
        assert!((post[0].p - want).abs() < 1e-5, "{} vs {want}", post[0].p);
    }

    /// Whisper's logits reach ±30 and a bare exp() there flushes the tail to
    /// zero - the max has to come out first or a confident window's runners-up
    /// all read as exactly 0.
    #[test]
    fn a_confident_row_keeps_its_tail() {
        let l = langs(&["en", "sv"]);
        let row = logits(&[(50259, -40.0), (50260, 40.0)], 51865);
        let post = posterior_over(&l, &row).unwrap();
        assert_eq!(post[0].code, "sv");
        assert!(post[0].p > 0.999);
        assert!(post[1].p > 0.0, "the tail flushed to zero");
        assert!(post[1].p.is_finite());
    }

    /// A logits row too short for the checkpoint's own token ids is refused
    /// rather than scored against a truncated denominator - that would put
    /// probabilities on the wire that do not describe any distribution.
    #[test]
    fn a_short_row_is_refused_not_scored() {
        let l = langs(&["en", "sv"]);
        assert!(posterior_over(&l, &vec![0.0; 100]).is_err());
        assert!(posterior_over(&[], &vec![0.0; 51865]).is_err());
    }
}
