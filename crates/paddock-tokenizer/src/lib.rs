//! Tokenizers built from GGUF-embedded vocabularies.
//!
//! SOTA position: we do not hand-roll BPE. The `tokenizers` crate is HF's
//! reference implementation (the thing tokenizer.json semantics are defined
//! against); our job is faithful *construction* from GGUF metadata - vocab,
//! merges, special tokens, and the per-family pre-tokenizer regex. Where
//! llama.cpp rewrites regexes to dodge its engine's missing features, we run
//! the original tokenizer.json patterns verbatim (fancy-regex handles (?i:)
//! and lookahead).
//!
//! Ultimate correctness gate: the transformers parity oracle - one wrong token
//! id shows up as a logit mismatch on the first forward pass.

mod pre_tokenizer;
pub mod sam3;

use paddock_models::gguf::{GgufFile, Value};
use tokenizers::models::bpe::BPE;
use tokenizers::{AddedToken, Tokenizer};

#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error("GGUF is missing tokenizer metadata key {0}")]
    MissingKey(&'static str),
    #[error(
        "tokenizer model {0:?} not supported yet (supported: \"gpt2\" byte-level BPE, \
         \"gemma4\" SPM-style BPE)"
    )]
    UnsupportedModel(String),
    #[error(
        "pre-tokenizer {0:?} is not in the registry - add its regex (verified against \
         the model's tokenizer.json) to pre_tokenizer.rs rather than guessing a default"
    )]
    UnknownPreTokenizer(String),
    #[error("merge entry {0:?} is malformed (expected \"left right\")")]
    BadMerge(String),
    #[error(
        "this model's GGUF ships an SPM vocab without merges; its real tokenizer \
         lives in the checkpoint's tokenizer.json - place that file next to the \
         weights (looked for {0})"
    )]
    MissingSidecar(String),
    #[error("tokenizers library error: {0}")]
    Library(String),
}

// llama.cpp token-type ids (tokenizer.ggml.token_type)
const TOKEN_TYPE_CONTROL: u64 = 3;
const TOKEN_TYPE_USER_DEFINED: u64 = 4;

/// A ready-to-use tokenizer plus the special ids generation cares about.
#[derive(Debug)]
pub struct GgufTokenizer {
    inner: Tokenizer,
    // HF permits more terminals than the GGUF eos/eot convenience pair.
    extra_eos_ids: Vec<u32>,
    pub bos_id: Option<u32>,
    pub eos_id: Option<u32>,
    /// GGUF `tokenizer.ggml.eot_token_id` - the END-OF-TURN token, which is
    /// not always the eos: muse-glimmer's eos (`<|end_of_text|>`) never
    /// appears mid-conversation, and `<|eot|>` is what actually closes an
    /// assistant turn. Serving adds it to the stop set; a file that declares
    /// none leaves this None (most do - only laguna and muse ship one here).
    pub eot_id: Option<u32>,
    pub pad_id: Option<u32>,
    /// GGUF `tokenizer.ggml.add_bos_token` - whether callers should lead with BOS.
    pub add_bos: bool,
    /// GGUF `tokenizer.chat_template` (Jinja), if present.
    pub chat_template: Option<String>,
    pub vocab_size: usize,
}

impl GgufTokenizer {
    /// Complete checkpoint stop set, preserving upstream order and removing
    /// duplicates. Gemma 4 uses three ids; never truncate that to eos/eot.
    pub fn stop_ids(&self) -> Vec<u32> {
        let mut ids = Vec::new();
        for id in self
            .eos_id
            .into_iter()
            .chain(self.eot_id)
            .chain(self.extra_eos_ids.iter().copied())
        {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
        ids
    }
    pub fn from_gguf(f: &GgufFile) -> Result<Self, TokenizerError> {
        // Whisper family: our own GGUF schema (our whisper converter)
        // embeds the HF tokenizer.json whole - whisper's GPT-2 BPE plus its
        // ~1600 special tokens (sot/eot, tasks, languages, timestamps) are
        // exactly what the `tokenizers` crate parses natively, so construction
        // is a straight parse instead of a lossy tokenizer.ggml.* rebuild.
        if let Some(json) = f
            .metadata
            .get("whisper.tokenizer_json")
            .and_then(Value::as_str)
        {
            return Self::from_embedded_json(f, json);
        }
        let model = get_str(f, "tokenizer.ggml.model")?;
        let tokens = get_str_array(f, "tokenizer.ggml.tokens")?;
        let mut inner = match model {
            "gpt2" => build_gpt2_bpe(f, &tokens)?,
            "gemma4" => build_spm_bpe(f, &tokens)?,
            other => return Err(TokenizerError::UnsupportedModel(other.to_owned())),
        };

        // control + user-defined tokens must never be split by the pre-tokenizer
        // (chat markers like <|return|> are the difference between a stop and a
        // runaway generation)
        if let Some(Value::Array(types)) = f.metadata.get("tokenizer.ggml.token_type") {
            let special: Vec<AddedToken> = types
                .iter()
                .enumerate()
                .filter_map(|(i, ty)| {
                    let ty = ty.as_u64()?;
                    if ty == TOKEN_TYPE_CONTROL || ty == TOKEN_TYPE_USER_DEFINED {
                        let content = tokens.get(i)?;
                        Some(
                            AddedToken::from((*content).to_owned(), ty == TOKEN_TYPE_CONTROL)
                                .normalized(false),
                        )
                    } else {
                        None
                    }
                })
                .collect();
            if !special.is_empty() {
                inner
                    .add_tokens(special)
                    .map_err(|e| TokenizerError::Library(e.to_string()))?;
            }
        }

        let id_meta = |key: &str| {
            f.metadata
                .get(key)
                .and_then(Value::as_u64)
                .and_then(|v| u32::try_from(v).ok())
        };

        let add_bos = match f.metadata.get("tokenizer.ggml.add_bos_token") {
            // gemma4 always leads with BOS regardless of the GGUF flag - some
            // early conversions shipped `false`; llama.cpp force-overrides too
            // (llama-vocab.cpp, "workaround for Gemma 4", PR #21500)
            Some(Value::Bool(b)) => *b || model == "gemma4",
            // default: llama SPM adds BOS, gpt2-BPE families generally don't
            _ => model == "gemma4",
        };

        let chat_template = f
            .metadata
            .get("tokenizer.chat_template")
            .and_then(Value::as_str)
            .map(str::to_owned);

        // Kolibri 1's generation_config stops on <|im_end|> (the GGUF eos)
        // AND <|endoftext|>; its converter carries only the first
        let extra_eos_ids = match f.metadata.get("tokenizer.ggml.pre").and_then(Value::as_str) {
            Some("kolibri1") => tokens
                .iter()
                .position(|t| *t == "<|endoftext|>")
                .map(|i| vec![i as u32])
                .unwrap_or_default(),
            _ => Vec::new(),
        };

        Ok(Self {
            inner,
            bos_id: id_meta("tokenizer.ggml.bos_token_id"),
            eos_id: id_meta("tokenizer.ggml.eos_token_id"),
            extra_eos_ids,
            eot_id: id_meta("tokenizer.ggml.eot_token_id"),
            pad_id: id_meta("tokenizer.ggml.padding_token_id"),
            add_bos,
            chat_template,
            vocab_size: tokens.len(),
        })
    }

    /// Build from a complete HF tokenizer.json embedded as GGUF metadata
    /// (the whisper family's schema). The decode-contract ids ride separate
    /// `whisper.token.*` keys stamped from the checkpoint's own generation
    /// config: generation ends at `<|endoftext|>` (eot); there is no
    /// text-side BOS - whisper prompting is explicit special tokens
    /// (sot/lang/task) built by the transcription path, so `add_bos` is
    /// permanently false and no chat template exists.
    fn from_embedded_json(f: &GgufFile, json: &str) -> Result<Self, TokenizerError> {
        let inner = Tokenizer::from_bytes(json.as_bytes())
            .map_err(|e| TokenizerError::Library(e.to_string()))?;
        let vocab_size = inner.get_vocab_size(true);
        let eos_id = f
            .metadata
            .get("whisper.token.eot")
            .and_then(Value::as_u64)
            .and_then(|v| u32::try_from(v).ok());
        Ok(Self {
            inner,
            bos_id: None,
            eos_id,
            extra_eos_ids: Vec::new(),
            eot_id: None,
            pad_id: None,
            add_bos: false,
            chat_template: None,
            vocab_size,
        })
    }

    /// [`Self::from_gguf`], with a sidecar fallback for SPM-class files.
    ///
    /// A `tokenizer.ggml.model = "llama"` GGUF carries vocab + scores but no
    /// merges, and its scores are converter-synthesized ranks (0, -1, -2, ...),
    /// not SentencePiece log-probs - there is no faithful rebuild from the
    /// GGUF alone. The tokenizer the family's arbiter (vLLM) actually runs is
    /// the checkpoint's converted-BPE tokenizer.json, so that file is the
    /// source of truth: read it from next to the weights (paddleocr-vl is the
    /// first family in this class). Decode-contract ids, add_bos, and the
    /// chat template still come from the GGUF's own metadata; vocab_size is
    /// the GGUF token count (the model's logit width includes padding rows
    /// past the HF vocab, and those must stay decodable-to-empty rather than
    /// out of range).
    pub fn from_gguf_with_sidecar(
        f: &GgufFile,
        weights_dir: &std::path::Path,
    ) -> Result<Self, TokenizerError> {
        match Self::from_gguf(f) {
            Err(TokenizerError::UnsupportedModel(model)) if model == "llama" => {
                let path = weights_dir.join("tokenizer.json");
                let bytes = std::fs::read(&path)
                    .map_err(|_| TokenizerError::MissingSidecar(path.display().to_string()))?;
                let inner = Tokenizer::from_bytes(&bytes)
                    .map_err(|e| TokenizerError::Library(e.to_string()))?;
                let id_meta = |key: &str| {
                    f.metadata
                        .get(key)
                        .and_then(Value::as_u64)
                        .and_then(|v| u32::try_from(v).ok())
                };
                let add_bos = matches!(
                    f.metadata.get("tokenizer.ggml.add_bos_token"),
                    Some(Value::Bool(true))
                );
                Ok(Self {
                    inner,
                    bos_id: id_meta("tokenizer.ggml.bos_token_id"),
                    eos_id: id_meta("tokenizer.ggml.eos_token_id"),
                    extra_eos_ids: Vec::new(),
                    eot_id: id_meta("tokenizer.ggml.eot_token_id"),
                    pad_id: id_meta("tokenizer.ggml.padding_token_id"),
                    add_bos,
                    chat_template: f
                        .metadata
                        .get("tokenizer.chat_template")
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                    vocab_size: get_str_array(f, "tokenizer.ggml.tokens")?.len(),
                })
            }
            other => other,
        }
    }

    /// Build straight from an HF checkpoint directory - the safetensors-primary
    /// path (nemotron_h_moe and the forced aligner; the
    /// aligner ignores the generative-contract fields and looks its special
    /// ids up by literal token text). No GGUF exists in this
    /// lane, so the checkpoint's own files are the only source of truth:
    ///
    /// - `tokenizer.json` - parsed natively (this is the exact tokenizer the
    ///   vLLM arbiter runs, byte-level BPE with the added-token overlay).
    /// - `tokenizer_config.json` - `add_bos_token` and the bos string.
    /// - `generation_config.json` - the eos SET. Nemotron stops on [2, 11]
    ///   (`</s>` and `<|im_end|>`); the first id lands in `eos_id`, the second
    ///   in `eot_id`. `stop_ids()` returns the complete set, including models
    ///   such as Gemma 4 which publish more than two terminal ids.
    /// - `chat_template.jinja` - the family template comes from this file
    ///   (tokenizer_config carries none for this family).
    pub fn from_hf_dir(dir: &std::path::Path) -> Result<Self, TokenizerError> {
        let tok_path = dir.join("tokenizer.json");
        let bytes = std::fs::read(&tok_path)
            .map_err(|_| TokenizerError::MissingSidecar(tok_path.display().to_string()))?;
        let inner =
            Tokenizer::from_bytes(&bytes).map_err(|e| TokenizerError::Library(e.to_string()))?;

        let json = |name: &str| -> Option<serde_json::Value> {
            serde_json::from_slice(&std::fs::read(dir.join(name)).ok()?).ok()
        };
        let tc = json("tokenizer_config.json");
        // bos/eos in tokenizer_config are token STRINGS (or {content: ...});
        // resolve through the vocab rather than trusting any numeric field.
        let tok_str = |v: Option<&serde_json::Value>| -> Option<String> {
            match v? {
                serde_json::Value::String(s) => Some(s.clone()),
                serde_json::Value::Object(o) => {
                    o.get("content").and_then(|c| c.as_str()).map(str::to_owned)
                }
                _ => None,
            }
        };
        let bos_id = tok_str(tc.as_ref().and_then(|t| t.get("bos_token")))
            .and_then(|s| inner.token_to_id(&s));
        let pad_id = tok_str(tc.as_ref().and_then(|t| t.get("pad_token")))
            .and_then(|s| inner.token_to_id(&s));
        let add_bos = tc
            .as_ref()
            .and_then(|t| t.get("add_bos_token"))
            .and_then(|b| b.as_bool())
            .unwrap_or(false);

        let mut eos_ids: Vec<u32> = Vec::new();
        if let Some(g) = json("generation_config.json") {
            match g.get("eos_token_id") {
                Some(serde_json::Value::Array(a)) => {
                    eos_ids = a
                        .iter()
                        .map(|x| {
                            x.as_u64()
                                .and_then(|x| u32::try_from(x).ok())
                                .filter(|&x| inner.id_to_token(x).is_some())
                                .ok_or_else(|| {
                                    TokenizerError::Library(
                                        "invalid generation_config terminal id".into(),
                                    )
                                })
                        })
                        .collect::<Result<_, _>>()?;
                }
                Some(serde_json::Value::Number(n)) => {
                    let id = n
                        .as_u64()
                        .and_then(|x| u32::try_from(x).ok())
                        .filter(|&id| inner.id_to_token(id).is_some())
                        .ok_or_else(|| {
                            TokenizerError::Library("invalid generation_config terminal id".into())
                        })?;
                    eos_ids = vec![id];
                }
                _ => {}
            }
        }
        if eos_ids.is_empty() {
            // fall back to the tokenizer_config eos string
            eos_ids.extend(
                tok_str(tc.as_ref().and_then(|t| t.get("eos_token")))
                    .and_then(|s| inner.token_to_id(&s)),
            );
        }

        let chat_template = std::fs::read_to_string(dir.join("chat_template.jinja"))
            .ok()
            .or_else(|| {
                tc.as_ref()
                    .and_then(|t| t.get("chat_template"))
                    .and_then(|c| c.as_str())
                    .map(str::to_owned)
            });

        let vocab_size = inner.get_vocab_size(true);
        Ok(Self {
            eos_id: eos_ids.first().copied(),
            eot_id: eos_ids.get(1).copied(),
            extra_eos_ids: eos_ids.iter().skip(2).copied().collect(),
            inner,
            bos_id,
            pad_id,
            add_bos,
            chat_template,
            vocab_size,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        Ok(self
            .inner
            .encode(text, false)
            .map_err(|e| TokenizerError::Library(e.to_string()))?
            .get_ids()
            .to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String, TokenizerError> {
        let s = self
            .inner
            .decode(ids, skip_special)
            .map_err(|e| TokenizerError::Library(e.to_string()))?;
        // Byte-level BPE can split one multi-byte character across tokens, so a
        // sequence whose final token(s) end mid-character decodes to a trailing
        // U+FFFD (replacement char). In streaming that char completes on the next
        // token, and a truncated final char shouldn't be shown at all - so drop a
        // trailing U+FFFD rather than surface a stray `�` (matches llama.cpp, which
        // buffers incomplete UTF-8). Only reallocates when one is actually present.
        if s.ends_with('\u{FFFD}') {
            Ok(s.trim_end_matches('\u{FFFD}').to_string())
        } else {
            Ok(s)
        }
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Ids the vocabulary spans, added tokens included - the range a random
    /// canvas id is drawn from.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    /// Incremental decoder for the streaming handlers - see [`StreamDecoder`].
    pub fn stream_decoder(&self, skip_special: bool) -> StreamDecoder {
        StreamDecoder {
            flushed: String::new(),
            pending: Vec::new(),
            skip_special,
        }
    }
}

/// Amortized per-token decoding for token streams.
///
/// Every streaming handler used to re-decode its full accumulated id
/// sequence on every sampled token - O(n²) per request, and a ~4.5x collapse
/// in long-stream throughput at high concurrency (streamed vs unstreamed, for
/// identical engine work). This keeps a
/// `flushed` text prefix plus a short `pending` id tail: each push decodes
/// only the tail, and the tail folds into the prefix every `FLUSH_AT`
/// tokens at a seam verified clean of split UTF-8 scalars.
///
/// Exactness: both tokenizer families here decode as position-independent
/// per-token surfaces (byte-level BPE's ByteLevel decoder; the SPM lane's
/// ByteFallback + Fuse + global ▁->space Replace, with no prefix-space
/// first-token special-casing) - so decode(a) + decode(b) == decode(a ++ b)
/// at the byte level whenever the seam does not split a multi-byte scalar.
/// A split scalar surfaces as U+FFFD at the slice edge, which is exactly
/// what the seam back-off below detects; real fallback runs are ≤ 4 tokens,
/// so backing off a few ids always finds a clean seam within the tail.
pub struct StreamDecoder {
    flushed: String,
    pending: Vec<u32>,
    skip_special: bool,
}

/// Tail ids kept un-flushed so an in-flight multi-token scalar never
/// reaches a flush seam.
const STREAM_LAG: usize = 8;
/// Fold the tail into the prefix once it grows past this many ids.
const STREAM_FLUSH_AT: usize = 64;

impl StreamDecoder {
    /// Push the next sampled id and return the full text so far. Matches
    /// `GgufTokenizer::decode(&all_ids, skip_special)` byte-for-byte,
    /// including its trailing-U+FFFD trim of an in-flight partial scalar.
    pub fn push(&mut self, tok: &GgufTokenizer, id: u32) -> String {
        self.pending.push(id);
        if self.pending.len() >= STREAM_FLUSH_AT {
            // pick a seam that does not split a scalar: back off up to 8 ids
            // (real byte-fallback runs are <= 4). If none verifies, skip this
            // fold - pending keeps growing and the next push retries, so the
            // pathological case degrades to the old full-decode behavior
            // instead of ever emitting wrong bytes.
            let mut cut = self.pending.len() - STREAM_LAG;
            let floor = cut.saturating_sub(8);
            while cut > floor {
                let folded = tok
                    .decode(&self.pending[..cut], self.skip_special)
                    .unwrap_or_default();
                // decode() trims a trailing U+FFFD, so a split scalar shows
                // as the RAW inner decode ending mid-char; detect it by
                // re-checking the untrimmed length parity instead: fold only
                // when re-decoding the remainder reproduces the whole.
                let rest = tok
                    .decode(&self.pending[cut..], self.skip_special)
                    .unwrap_or_default();
                let whole = tok
                    .decode(&self.pending, self.skip_special)
                    .unwrap_or_default();
                if format!("{folded}{rest}") == whole {
                    self.flushed.push_str(&folded);
                    self.pending.drain(..cut);
                    break;
                }
                cut -= 1;
            }
        }
        let tail = tok
            .decode(&self.pending, self.skip_special)
            .unwrap_or_default();
        let mut out = String::with_capacity(self.flushed.len() + tail.len());
        out.push_str(&self.flushed);
        out.push_str(&tail);
        out
    }
}

/// Byte-level BPE (GPT-2 lineage): regex pre-split from the family registry,
/// then BPE over the GPT-2 byte alphabet.
fn build_gpt2_bpe(f: &GgufFile, tokens: &[&str]) -> Result<Tokenizer, TokenizerError> {
    let pre = f
        .metadata
        .get("tokenizer.ggml.pre")
        .and_then(Value::as_str)
        .unwrap_or("default");
    // The `granite-docling` GGUF label is reused for different published
    // tokenizers. Granite 4.1 Vision uses the explicit llama3-like split; the
    // Granite 4.2 text family ships a plain ByteLevel(use_regex=true), i.e.
    // GPT-2 split. Changing the registry entry globally would break Vision.
    // Use the checkpoint's canonical basename, never its filename/display
    // alias, to resolve this converter ambiguity. Verified 2026-09-06 against
    // https://huggingface.co/ibm-granite/granite-4.2-8b/blob/main/tokenizer.json
    // and the same-GGUF released llama-tokenize. No weights are rewritten.
    // Speech 4.1 repeats the ambiguity within the family: base is llama3-like,
    // Plus is plain ByteLevel(use_regex=true). Verified against both official
    // tokenizer.json files on 2026-09-11. Resolve from file metadata, never a
    // user-chosen path. This tokenizer correction is backend-independent.
    let basename = f.metadata.get("general.basename").and_then(Value::as_str);
    let speech_plus = basename == Some("granite-speech-4.1")
        && ["general.finetune", "general.name"].iter().any(|key| {
            f.metadata
                .get(*key)
                .and_then(Value::as_str)
                .is_some_and(|v| v.to_lowercase().ends_with("plus"))
        });
    let pre = if pre == "granite-docling" && (basename == Some("granite-4.2") || speech_plus) {
        "gpt-2"
    } else {
        pre
    };

    let bpe = BPE::builder()
        .vocab_and_merges(build_vocab(tokens), parse_merges(f, false)?)
        .build()
        .map_err(|e| TokenizerError::Library(e.to_string()))?;

    let mut inner = Tokenizer::new(bpe);
    inner.with_pre_tokenizer(Some(pre_tokenizer::build(pre)?));
    inner.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
    Ok(inner)
}

/// SPM-style BPE (Gemma 4 lineage). Construction verified against llama.cpp's
/// llama-vocab.cpp (LLAMA_VOCAB_PRE_TYPE_GEMMA4):
/// - the normalizer replaces spaces with ▁ (U+2581); no prefix space
///   (`add_space_prefix = false` in the GGUF)
/// - BPE merges run over raw UTF-8 (not the GPT-2 byte alphabet), with byte
///   fallback to the `<0xXX>` vocab entries for unknown sequences
/// - the only pre-split is on newline runs ("[^\n]+|[\n]+"), and newline runs
///   present verbatim in the vocab are looked up whole (llama.cpp PR #21343);
///   `ignore_merges` gives the same whole-piece-lookup-first semantics
fn build_spm_bpe(f: &GgufFile, tokens: &[&str]) -> Result<Tokenizer, TokenizerError> {
    use tokenizers::SplitDelimiterBehavior;
    use tokenizers::decoders::byte_fallback::ByteFallback;
    use tokenizers::decoders::fuse::Fuse;
    use tokenizers::decoders::sequence::Sequence as DecoderSequence;
    use tokenizers::normalizers::replace::Replace as NormReplace;
    use tokenizers::pre_tokenizers::split::{Split, SplitPattern};

    let unk = f
        .metadata
        .get("tokenizer.ggml.unknown_token_id")
        .and_then(Value::as_u64)
        .and_then(|id| tokens.get(id as usize).map(|t| (*t).to_owned()));

    let mut builder = BPE::builder()
        .vocab_and_merges(build_vocab(tokens), parse_merges(f, true)?)
        .byte_fallback(true)
        .ignore_merges(true);
    if let Some(unk) = unk {
        builder = builder.unk_token(unk);
    }
    let bpe = builder
        .build()
        .map_err(|e| TokenizerError::Library(e.to_string()))?;

    let mut inner = Tokenizer::new(bpe);
    inner
        .with_normalizer(Some(
            NormReplace::new(" ", "\u{2581}")
                .map_err(|e| TokenizerError::Library(e.to_string()))?,
        ))
        .map_err(|e| TokenizerError::Library(e.to_string()))?;
    inner.with_pre_tokenizer(Some(
        Split::new(
            SplitPattern::Regex("\n+".to_owned()),
            SplitDelimiterBehavior::Isolated,
            false,
        )
        .map_err(|e| TokenizerError::Library(e.to_string()))?,
    ));
    inner.with_decoder(Some(DecoderSequence::new(vec![
        tokenizers::DecoderWrapper::Replace(
            tokenizers::normalizers::replace::Replace::new("\u{2581}", " ")
                .map_err(|e| TokenizerError::Library(e.to_string()))?,
        ),
        tokenizers::DecoderWrapper::ByteFallback(ByteFallback::new()),
        tokenizers::DecoderWrapper::Fuse(Fuse::new()),
    ])));
    Ok(inner)
}

fn build_vocab(tokens: &[&str]) -> tokenizers::models::bpe::Vocab {
    tokens
        .iter()
        .enumerate()
        .map(|(i, t)| (t.to_string(), i as u32))
        .collect()
}

/// Parse `tokenizer.ggml.merges`. With `skip_first`, the split point is the
/// first space at OR after byte 1 - SPM-style vocabs can open a merge side
/// with characters that make a position-0 space ambiguous; mirrors llama.cpp's
/// `word.find(' ', 1)` for the gemma4 path.
fn parse_merges(f: &GgufFile, skip_first: bool) -> Result<Vec<(String, String)>, TokenizerError> {
    let merges_raw = get_str_array(f, "tokenizer.ggml.merges")?;
    merges_raw
        .iter()
        .map(|m| {
            // byte-wise search: ' ' is ASCII so the found index is always a
            // char boundary, and `skip(1)` must count BYTES to mirror
            // llama.cpp's `word.find(' ', 1)` (merges open with multi-byte ▁)
            let skip = usize::from(skip_first);
            let pos = m
                .as_bytes()
                .iter()
                .skip(skip)
                .position(|&b| b == b' ')
                .map(|p| p + skip);
            pos.map(|p| (m[..p].to_owned(), m[p + 1..].to_owned()))
                .ok_or_else(|| TokenizerError::BadMerge((*m).to_owned()))
        })
        .collect()
}

fn get_str<'a>(f: &'a GgufFile, key: &'static str) -> Result<&'a str, TokenizerError> {
    f.metadata
        .get(key)
        .and_then(Value::as_str)
        .ok_or(TokenizerError::MissingKey(key))
}

fn get_str_array<'a>(f: &'a GgufFile, key: &'static str) -> Result<Vec<&'a str>, TokenizerError> {
    match f.metadata.get(key) {
        Some(Value::Array(items)) => Ok(items.iter().filter_map(Value::as_str).collect()),
        _ => Err(TokenizerError::MissingKey(key)),
    }
}

#[cfg(test)]
mod tests;
