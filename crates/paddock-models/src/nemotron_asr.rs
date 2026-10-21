//! Nemotron 3.5 ASR: NVIDIA's cache-aware streaming FastConformer encoder
//! with an RNN-T decoder and language-ID prompt conditioning, as its own GGUF
//! conversion ships it (`general.architecture = "asr"`, `asr.head_type =
//! "rnnt"`). The architecture contract, the tensor inventory, the prompt
//! dictionary and the vocabulary are checked here before a backend allocates
//! anything; a checkpoint off the contract is refused by name, not
//! approximated.
//!
//! The encoder is fixed by the contract: a x8 causal convolution subsampler
//! (a 3x3 stride-2 stem, then two depthwise-separable 3x3 stride-2 stages,
//! frequency padded (2, 1) and time causal) into 24 conformer blocks of 1024
//! with Transformer-XL relative-position attention under a `chunked_limited`
//! mask (56 frames of left context, chunks of `lookahead + 1` frames) and a
//! causal kernel-9 depthwise convolution. The prompt projector folds a
//! one-hot language vector into every frame; a two-layer LSTM prediction
//! network and a ReLU joint score the 13087-piece vocabulary plus the blank.

use crate::mapped::MappedGguf;
use half::f16;
use std::path::Path;

pub const SAMPLE_RATE: usize = 16_000;
pub const HOP: usize = 160;
pub const N_FFT: usize = 512;
pub const WINDOW: usize = 400;
pub const MEL: usize = 128;
pub const BINS: usize = N_FFT / 2 + 1;
pub const SUBSAMPLING: usize = 8;
pub const D_MODEL: usize = 1024;
pub const HEADS: usize = 8;
pub const HEAD_DIM: usize = D_MODEL / HEADS;
pub const D_FF: usize = 4096;
pub const LAYERS: usize = 24;
pub const CONV_KERNEL: usize = 9;
/// Subsampler channels and the frequency bins after each stride-2 stage
/// (128 -> 65 -> 33 -> 17, each `bins / 2 + 1` under the (2, 1) padding).
pub const SUB_CHANNELS: usize = 256;
pub const SUB_BINS: [usize; 3] = [65, 33, 17];
pub const SUB_OUT: usize = SUB_CHANNELS * SUB_BINS[2];
/// Encoder frames of left attention context (`sliding_window - 1`).
pub const LEFT_CONTEXT: usize = 56;
/// The right contexts NVIDIA publishes (80 ms frames: chunks of 80, 160,
/// 320, 560 and 1120 ms); 3 is the checkpoint's offline default.
pub const LOOKAHEADS: [usize; 5] = [0, 1, 3, 6, 13];
pub const DEFAULT_LOOKAHEAD: usize = 3;
pub const PRED: usize = 640;
pub const PRED_LAYERS: usize = 2;
/// Joint outputs: the vocabulary plus the blank, which is the last index.
pub const VOCAB: usize = 13_088;
pub const BLANK: u32 = 13_087;
pub const MAX_SYMBOLS: usize = 10;
pub const PROMPTS: usize = 128;
pub const PROMPT_HIDDEN: usize = 2048;
pub const AUTO_PROMPT: &str = "auto";
/// The stored relative-position table: row `PE_CENTER - d` is distance `d`.
pub const PE_ROWS: usize = 9_999;
pub const PE_CENTER: usize = 4_999;

/// GGML type ids as the GGUF stores them.
pub const F32: u32 = 0;
pub const F16: u32 = 1;
pub const Q8_0: u32 = 8;

/// Encoder frames a chunk of right context `lookahead` holds.
pub fn chunk_frames(lookahead: usize) -> usize {
    lookahead + 1
}

/// Mel frames of a stream's first chunk and of every later one: the causal
/// x8 subsampler turns `1 + 8 r` frames into `r + 1` and each `8 (r + 1)`
/// after them into `r + 1` more (NeMo's `setup_streaming_params`).
pub fn chunk_mel_frames(lookahead: usize, first: bool) -> usize {
    if first {
        1 + SUBSAMPLING * lookahead
    } else {
        SUBSAMPLING * (lookahead + 1)
    }
}

/// Valid mel frames of `samples` of audio: the centered STFT's frames whose
/// hop lies inside the recording (`floor(samples / hop)`).
pub fn mel_frames(samples: usize) -> usize {
    samples / HOP
}

/// Encoder frames of `mel` valid mel frames: each causal stride-2 stage
/// keeps `n / 2 + 1` (left padding 2, right 1).
pub fn encoder_frames(mel: usize) -> usize {
    if mel == 0 {
        return 0;
    }
    (0..3).fold(mel, |n, _| n / 2 + 1)
}

/// One stored tensor: its GGML type id and raw little-endian bytes.
pub struct Raw<'a> {
    pub ty: u32,
    pub bytes: &'a [u8],
}

/// What a tensor may be stored as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// a projection: Q8_0 blocks along K
    Projection,
    /// an F16 convolution or embedding, widened exactly
    Half,
    /// F32 as stored: norms, biases, the prompt projector, tables
    Full,
}

/// The checked inventory: (GGUF name, shape row-major as a backend reads it,
/// stored kind). A shape's last axis is the GGUF's fastest one.
pub fn inventory() -> Vec<(String, Vec<usize>, Kind)> {
    use Kind::*;
    let mut out: Vec<(String, Vec<usize>, Kind)> = Vec::new();
    let mut push = |n: String, s: &[usize], k: Kind| out.push((n, s.to_vec(), k));
    push("encoder.pos_enc.pe".into(), &[PE_ROWS, D_MODEL], Full);
    push("preprocessor.fb".into(), &[MEL, BINS], Full);
    let c = SUB_CHANNELS;
    push(
        "encoder.pre_encode.conv.0.weight".into(),
        &[c, 1, 3, 3],
        Half,
    );
    for (dw, pw) in [(2, 3), (5, 6)] {
        push(
            format!("encoder.pre_encode.conv.{dw}.weight"),
            &[c, 1, 3, 3],
            Half,
        );
        push(
            format!("encoder.pre_encode.conv.{pw}.weight"),
            &[c, c, 1, 1],
            Half,
        );
    }
    for i in [0, 2, 3, 5, 6] {
        push(format!("encoder.pre_encode.conv.{i}.bias"), &[c], Full);
    }
    push(
        "encoder.pre_encode.out.weight".into(),
        &[D_MODEL, SUB_OUT],
        Projection,
    );
    push("encoder.pre_encode.out.bias".into(), &[D_MODEL], Full);
    for i in 0..LAYERS {
        let p = format!("encoder.layers.{i}");
        for n in [
            "norm_feed_forward1",
            "norm_self_att",
            "norm_conv",
            "conv.batch_norm",
            "norm_feed_forward2",
            "norm_out",
        ] {
            push(format!("{p}.{n}.weight"), &[D_MODEL], Full);
            push(format!("{p}.{n}.bias"), &[D_MODEL], Full);
        }
        for f in ["feed_forward1", "feed_forward2"] {
            push(
                format!("{p}.{f}.linear1.weight"),
                &[D_FF, D_MODEL],
                Projection,
            );
            push(
                format!("{p}.{f}.linear2.weight"),
                &[D_MODEL, D_FF],
                Projection,
            );
        }
        for n in [
            "linear_q",
            "linear_k",
            "linear_v",
            "linear_out",
            "linear_pos",
        ] {
            push(
                format!("{p}.self_attn.{n}.weight"),
                &[D_MODEL, D_MODEL],
                Projection,
            );
        }
        for n in ["pos_bias_u", "pos_bias_v"] {
            push(format!("{p}.self_attn.{n}"), &[HEADS, HEAD_DIM], Full);
        }
        push(
            format!("{p}.conv.pointwise_conv1.weight"),
            &[2 * D_MODEL, D_MODEL],
            Projection,
        );
        push(
            format!("{p}.conv.depthwise_conv.weight"),
            &[D_MODEL, 1, CONV_KERNEL],
            Half,
        );
        push(
            format!("{p}.conv.pointwise_conv2.weight"),
            &[D_MODEL, D_MODEL],
            Projection,
        );
    }
    push(
        "decoder.prediction.embed.weight".into(),
        &[VOCAB, PRED],
        Half,
    );
    for l in 0..PRED_LAYERS {
        for g in ["ih", "hh"] {
            push(
                format!("decoder.prediction.dec_rnn.lstm.{g}_l{l}.weight"),
                &[4 * PRED, PRED],
                Projection,
            );
            push(
                format!("decoder.prediction.dec_rnn.lstm.{g}_l{l}.bias"),
                &[4 * PRED],
                Full,
            );
        }
    }
    for (n, k, o) in [
        ("joint.pred", PRED, PRED),
        ("joint.enc", D_MODEL, PRED),
        ("joint.joint_net.2", PRED, VOCAB),
    ] {
        push(format!("{n}.weight"), &[o, k], Projection);
        push(format!("{n}.bias"), &[o], Full);
    }
    push(
        "prompt_kernel.0.weight".into(),
        &[PROMPT_HIDDEN, D_MODEL + PROMPTS],
        Full,
    );
    push("prompt_kernel.0.bias".into(), &[PROMPT_HIDDEN], Full);
    push(
        "prompt_kernel.2.weight".into(),
        &[D_MODEL, PROMPT_HIDDEN],
        Full,
    );
    push("prompt_kernel.2.bias".into(), &[D_MODEL], Full);
    out
}

/// The served checkpoint: NVIDIA's GGUF, validated.
pub struct Checkpoint {
    g: MappedGguf,
    /// (dictionary key, prompt index) in the file's order
    prompts: Vec<(String, u32)>,
    vocab: Vec<String>,
}

fn metadata(g: &MappedGguf) -> Result<(), String> {
    use crate::gguf::Value as G;
    let m = &g.gguf().metadata;
    let str_of = |k: &str| m.get(k).and_then(|v| v.as_str());
    for (key, expected) in [
        ("general.architecture", "asr"),
        ("asr.head_type", "rnnt"),
        ("asr.encoder.att_context_style", "chunked_limited"),
        ("asr.encoder.conv_context", "causal"),
        ("asr.encoder.conv_norm", "layer_norm"),
        ("asr.preprocessor.normalize", "NA"),
        ("asr.tokenizer.type", "sentencepiece_bpe"),
    ] {
        if str_of(key) != Some(expected) {
            return Err(format!(
                "unsupported {key} {:?} (want {expected:?})",
                str_of(key)
            ));
        }
    }
    for (key, expected) in [
        ("asr.encoder.d_model", D_MODEL),
        ("asr.encoder.n_layers", LAYERS),
        ("asr.encoder.n_heads", HEADS),
        ("asr.encoder.d_ff", D_FF),
        ("asr.encoder.conv_kernel_size", CONV_KERNEL),
        ("asr.encoder.subsampling_factor", SUBSAMPLING),
        ("asr.encoder.subsampling_conv_channels", SUB_CHANNELS),
        ("asr.encoder.feat_in", MEL),
        ("asr.encoder.train_left_ctx", LEFT_CONTEXT),
        ("asr.preprocessor.sample_rate", SAMPLE_RATE),
        ("asr.preprocessor.n_fft", N_FFT),
        ("asr.preprocessor.features", MEL),
        ("asr.rnnt.vocab_size", VOCAB),
        ("asr.rnnt.blank_id", BLANK as usize),
        ("asr.rnnt.pred_embed_dim", PRED),
        ("asr.rnnt.pred_hidden", PRED),
        ("asr.rnnt.pred_num_layers", PRED_LAYERS),
        ("asr.rnnt.joint_dim", PRED),
        ("asr.rnnt.max_symbols_per_step", MAX_SYMBOLS),
        ("asr.rnnt.num_prompts", PROMPTS),
    ] {
        let got = m.get(key).and_then(|v| v.as_u64());
        if got != Some(expected as u64) {
            return Err(format!("unsupported {key} {got:?} (want {expected})"));
        }
    }
    for (key, expected) in [
        ("asr.encoder.offline_left_ctx", LEFT_CONTEXT as i64),
        ("asr.encoder.offline_right_ctx", DEFAULT_LOOKAHEAD as i64),
    ] {
        let got = m.get(key).and_then(|v| v.as_i64());
        if got != Some(expected) {
            return Err(format!("unsupported {key} {got:?} (want {expected})"));
        }
    }
    for (key, expected) in [
        ("asr.encoder.xscaling", false),
        ("asr.encoder.use_bias", false),
        ("asr.encoder.cache_supported", true),
        ("asr.preprocessor.hann_periodic", false),
        ("asr.preprocessor.stft_center_window", true),
    ] {
        if m.get(key) != Some(&G::Bool(expected)) {
            return Err(format!("unsupported {key} (want {expected})"));
        }
    }
    for (key, expected) in [
        ("asr.preprocessor.window_size", 0.025f32),
        ("asr.preprocessor.window_stride", 0.01),
        ("asr.preprocessor.preemph", 0.97),
    ] {
        if m.get(key).and_then(|v| v.as_f32()) != Some(expected) {
            return Err(format!("unsupported {key} (want {expected})"));
        }
    }
    Ok(())
}

fn strings(g: &MappedGguf, key: &str) -> Result<Vec<String>, String> {
    use crate::gguf::Value as G;
    match g.gguf().metadata.get(key) {
        Some(G::Array(items)) => items
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| format!("{key}: not a string array"))
            })
            .collect(),
        _ => Err(format!("missing {key}")),
    }
}

impl Checkpoint {
    /// Open and check the metadata, the prompt dictionary and the
    /// vocabulary; [`Self::validate`] checks every tensor.
    pub fn open(path: &Path) -> Result<Self, String> {
        let g = MappedGguf::open(path).map_err(|e| e.to_string())?;
        metadata(&g)?;
        let mut prompts = Vec::new();
        for entry in strings(&g, "asr.rnnt.prompt_dictionary")? {
            let (key, id) = entry
                .rsplit_once(':')
                .ok_or_else(|| format!("prompt dictionary entry {entry:?}"))?;
            let id: u32 = id
                .parse()
                .map_err(|_| format!("prompt dictionary entry {entry:?}"))?;
            if key.is_empty() || id as usize >= PROMPTS {
                return Err(format!("prompt dictionary entry {entry:?}"));
            }
            prompts.push((key.to_owned(), id));
        }
        if !prompts.iter().any(|(k, _)| k == AUTO_PROMPT) {
            return Err("prompt dictionary has no auto entry".into());
        }
        let vocab = strings(&g, "asr.tokenizer.vocab")?;
        if vocab.len() != BLANK as usize {
            return Err(format!(
                "vocabulary of {} pieces (want {BLANK} + the blank)",
                vocab.len()
            ));
        }
        Ok(Self { g, prompts, vocab })
    }

    /// The stored tensor behind a name, checked against its row-major shape.
    pub fn raw(&self, name: &str, shape: &[usize]) -> Result<Raw<'_>, String> {
        let (info, bytes) = self.g.tensor_bytes(name).map_err(|e| e.to_string())?;
        let expected: Vec<u64> = shape.iter().rev().map(|&n| n as u64).collect();
        let n: usize = shape.iter().product();
        let len = match info.raw_type {
            F32 => n * 4,
            F16 => n * 2,
            Q8_0 if shape.last().is_some_and(|k| k % 32 == 0) => n / 32 * 34,
            _ => usize::MAX,
        };
        if info.dims != expected || bytes.len() != len {
            return Err(format!(
                "{name}: unsupported shape/type {:?}/{}",
                info.dims, info.raw_type
            ));
        }
        Ok(Raw {
            ty: info.raw_type,
            bytes,
        })
    }

    /// F32 values of a tensor, row-major: F16 widens exactly, Q8_0
    /// dequantizes exactly (an f16 scale times an int8 fits F32's mantissa).
    pub fn values(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        let v = decode(&self.raw(name, shape)?)?;
        if v.iter().any(|x| !x.is_finite()) {
            return Err(format!("{name}: nonfinite weight"));
        }
        Ok(v)
    }

    /// Every inventory tensor present with its shape and an allowed stored
    /// type; the bytes a backend keeping projections as stored (Q8_0) and
    /// everything else in F32 holds.
    pub fn validate(&self) -> Result<u64, String> {
        let mut bytes = 0u64;
        for (name, shape, kind) in inventory() {
            let raw = self.raw(&name, &shape)?;
            let ok = match kind {
                Kind::Projection => raw.ty == Q8_0,
                Kind::Half => raw.ty == F16,
                Kind::Full => raw.ty == F32,
            };
            if !ok {
                return Err(format!("{name}: stored as type {} (want {kind:?})", raw.ty));
            }
            let n = shape.iter().product::<usize>() as u64;
            bytes += if kind == Kind::Projection {
                raw.bytes.len() as u64
            } else {
                4 * n
            };
        }
        Ok(bytes)
    }

    /// The prompt index of a dictionary key: a locale (`de-DE`), a bare code
    /// (`de`) or `auto`.
    pub fn prompt(&self, key: &str) -> Option<u32> {
        self.prompts
            .iter()
            .find(|(k, _)| k == key)
            .map(|&(_, id)| id)
    }

    /// The dictionary as stored: (key, prompt index).
    pub fn prompts(&self) -> &[(String, u32)] {
        &self.prompts
    }

    /// The vocabulary's pieces, index = token id (the blank is not one).
    pub fn vocab(&self) -> &[String] {
        &self.vocab
    }

    /// The analysis window, 400 taps: the symmetric Hann the metadata names,
    /// F64 evaluated and rounded once.
    pub fn window(&self) -> Vec<f32> {
        (0..WINDOW)
            .map(|i| {
                (0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (WINDOW - 1) as f64).cos()))
                    as f32
            })
            .collect()
    }
}

/// F32 values of a stored tensor (row-major as stored).
pub fn decode(raw: &Raw<'_>) -> Result<Vec<f32>, String> {
    let b = raw.bytes;
    Ok(match raw.ty {
        F32 => b
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        F16 => b
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f16::from_le_bytes(*c).to_f32())
            .collect(),
        Q8_0 => b
            .as_chunks::<34>()
            .0
            .iter()
            .flat_map(|blk| {
                let d = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                blk[2..].iter().map(move |&q| d * f32::from(q as i8))
            })
            .collect(),
        t => return Err(format!("unsupported stored type {t}")),
    })
}

/// A special piece: `<unk>` or a `<xx-XX>` language tag.
pub fn is_special(piece: &str) -> bool {
    piece.len() > 2 && piece.starts_with('<') && piece.ends_with('>')
}

/// The language a tag piece names (`<de-DE>` -> `de-DE`).
pub fn tag_language(piece: &str) -> Option<&str> {
    let inner = piece.strip_prefix('<')?.strip_suffix('>')?;
    (inner.len() >= 5 && inner.as_bytes()[2] == b'-' && inner != "unk").then_some(inner)
}

/// Text of a token run: pieces joined, the word-boundary mark as a space,
/// special pieces dropped (`keep_tags` keeps language tags), runs of spaces a
/// dropped tag leaves folded into one, ends trimmed.
pub fn detokenize(vocab: &[String], tokens: &[u32], keep_tags: bool) -> String {
    let mut out = String::new();
    for &t in tokens {
        let Some(p) = vocab.get(t as usize) else {
            continue;
        };
        if is_special(p) && !(keep_tags && tag_language(p).is_some()) {
            continue;
        }
        for ch in p.chars() {
            let ch = if ch == '\u{2581}' { ' ' } else { ch };
            if ch == ' ' && (out.is_empty() || out.ends_with(' ')) {
                continue;
            }
            out.push(ch);
        }
    }
    out.trim_end().to_owned()
}

/// Seconds an encoder frame spans (8 hops of 10 ms).
pub const FRAME_SECONDS: f64 = (SUBSAMPLING * HOP) as f64 / SAMPLE_RATE as f64;

/// A word of a transcript and the encoder frames its tokens were emitted at:
/// `start` the first token's frame, `end` one frame past the last one's.
#[derive(Clone, Debug, PartialEq)]
pub struct Word {
    pub text: String,
    pub start: f64,
    pub end: f64,
}

/// A sentence: the model closes one with a language tag after its terminal
/// punctuation (in every prompt mode), and the tag names the language it
/// heard. Text is the detokenized run without the tag.
#[derive(Clone, Debug, PartialEq)]
pub struct Sentence {
    pub text: String,
    pub start: f64,
    pub end: f64,
    pub language: Option<String>,
    /// the sentence's words, `words[first..first + count]`
    pub first: usize,
    pub count: usize,
}

/// Words and sentences of a token run with the frames they were emitted at.
/// A piece that starts with the word-boundary mark starts a word; any other
/// piece (punctuation, a continuation) extends the current one. A language
/// tag closes the sentence it follows.
pub fn render(vocab: &[String], tokens: &[u32], frames: &[u32]) -> (Vec<Word>, Vec<Sentence>) {
    let mut words: Vec<Word> = Vec::new();
    let mut sentences = Vec::new();
    let mut first = 0usize;
    let mut sentence_tokens: Vec<u32> = Vec::new();
    // a bare boundary piece (the mark alone) starts the next piece's word
    let mut boundary = false;
    let mut close =
        |words: &Vec<Word>, toks: &mut Vec<u32>, first: &mut usize, lang: Option<String>| {
            let text = detokenize(vocab, toks, false);
            if !text.is_empty() && *first < words.len() {
                sentences.push(Sentence {
                    text,
                    start: words[*first].start,
                    end: words[words.len() - 1].end,
                    language: lang,
                    first: *first,
                    count: words.len() - *first,
                });
            }
            *first = words.len();
            toks.clear();
        };
    for (&t, &f) in tokens.iter().zip(frames) {
        let Some(p) = vocab.get(t as usize) else {
            continue;
        };
        if let Some(lang) = tag_language(p) {
            close(
                &words,
                &mut sentence_tokens,
                &mut first,
                Some(lang.to_owned()),
            );
            continue;
        }
        if is_special(p) {
            continue;
        }
        sentence_tokens.push(t);
        let (start, end) = (f as f64 * FRAME_SECONDS, (f + 1) as f64 * FRAME_SECONDS);
        let piece = p.replace('\u{2581}', " ");
        let begins = boundary || p.starts_with('\u{2581}') || words.len() == first;
        let text = piece.trim_start();
        if text.is_empty() {
            boundary = true;
            continue;
        }
        boundary = false;
        match words.last_mut() {
            Some(w) if !begins => {
                w.text.push_str(text);
                w.end = end;
            }
            _ => words.push(Word {
                text: text.to_owned(),
                start,
                end,
            }),
        }
    }
    close(&words, &mut sentence_tokens, &mut first, None);
    (words, sentences)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_follows_the_causal_subsampler() {
        // 10 s: 1000 mel frames -> 501 -> 251 -> 126 encoder frames
        assert_eq!(encoder_frames(mel_frames(160_000)), 126);
        assert_eq!(encoder_frames(0), 0);
        assert_eq!(encoder_frames(1), 1);
        // a first chunk of 1 + 8 r frames is r + 1 encoder frames, and every
        // later chunk of 8 (r + 1) adds r + 1
        for r in LOOKAHEADS {
            let first = chunk_mel_frames(r, true);
            assert_eq!(encoder_frames(first), chunk_frames(r));
            let two = first + chunk_mel_frames(r, false);
            assert_eq!(encoder_frames(two), 2 * chunk_frames(r));
            // left context is whole chunks at every published lookahead
            assert_eq!(LEFT_CONTEXT % chunk_frames(r), 0);
        }
        assert_eq!(SUB_OUT, 4352);
    }

    #[test]
    fn q8_dequant_is_exact() {
        // d = 0.5 (f16 0x3800), q = -128, 1, 127, ...
        let mut blk = vec![0x00u8, 0x38];
        blk.extend((0..32).map(|i| [128u8, 1, 127][i % 3]));
        let v = decode(&Raw {
            ty: Q8_0,
            bytes: &blk,
        })
        .unwrap();
        assert_eq!(&v[..3], &[-64.0, 0.5, 63.5]);
    }

    #[test]
    fn detokenize_drops_tags_and_folds_spaces() {
        let vocab: Vec<String> = [
            "<unk>",
            "\u{2581}Hello",
            ".",
            "\u{2581}",
            "<en-US>",
            "\u{2581}World",
        ]
        .iter()
        .map(|s| (*s).to_owned())
        .collect();
        assert_eq!(detokenize(&vocab, &[1, 2, 3, 4, 5], false), "Hello. World");
        assert_eq!(
            detokenize(&vocab, &[1, 2, 3, 4, 5], true),
            "Hello. <en-US> World"
        );
        assert_eq!(tag_language("<en-US>"), Some("en-US"));
        assert_eq!(tag_language("<unk>"), None);
    }

    /// The published GGUF, when present: every tensor and the dictionary.
    #[test]
    fn published_checkpoint_validates() {
        let Some(path) = std::env::var_os("NEMOTRON_ASR_GGUF") else {
            return;
        };
        let c = Checkpoint::open(Path::new(&path)).unwrap();
        let bytes = c.validate().unwrap();
        assert!(bytes > 600_000_000, "{bytes}");
        assert_eq!(c.prompt("en-US"), Some(0));
        assert_eq!(c.prompt("de"), Some(9));
        assert_eq!(c.prompt(AUTO_PROMPT), Some(101));
        assert_eq!(c.vocab()[2947], "<en-US>");
        let fb = c.values("preprocessor.fb", &[MEL, BINS]).unwrap();
        assert!(fb.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn render_cuts_words_and_sentences_on_tags() {
        let vocab: Vec<String> = [
            "<unk>",
            "\u{2581}Hel",
            "lo",
            ".",
            "<en-US>",
            "\u{2581}Wor",
            "ld",
            "!",
        ]
        .iter()
        .map(|s| (*s).to_owned())
        .collect();
        let (words, sentences) = render(&vocab, &[1, 2, 3, 4, 5, 6, 7], &[0, 1, 1, 2, 4, 5, 5]);
        let texts: Vec<&str> = words.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(texts, ["Hello.", "World!"]);
        // the bare mark is a boundary before a piece without one
        let mut v2 = vocab.clone();
        v2.push("\u{2581}".into());
        v2.push("there".into());
        let (w2, _) = render(&v2, &[1, 2, 8, 9], &[0, 0, 1, 1]);
        let t2: Vec<&str> = w2.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(t2, ["Hello", "there"]);
        assert_eq!(words[0].start, 0.0);
        assert!((words[0].end - 0.16).abs() < 1e-12);
        assert_eq!(sentences.len(), 2);
        assert_eq!(sentences[0].text, "Hello.");
        assert_eq!(sentences[0].language.as_deref(), Some("en-US"));
        assert_eq!((sentences[1].first, sentences[1].count), (1, 1));
        assert_eq!(sentences[1].language, None);
        assert!((sentences[1].start - 0.32).abs() < 1e-12);
    }

    #[test]
    fn inventory_counts() {
        let inv = inventory();
        let proj = inv
            .iter()
            .filter(|(_, _, k)| *k == Kind::Projection)
            .count();
        // 11 a layer (2 + 2 FF, 5 attention, 2 conv) + the subsampler's
        // linear + 4 LSTM + 3 joint
        assert_eq!(proj, 11 * LAYERS + 1 + 4 + 3);
        // every GGUF tensor of the published file: 657
        assert_eq!(inv.len(), 657);
    }
}
