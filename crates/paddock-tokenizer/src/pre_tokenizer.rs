//! Per-family pre-tokenizer registry.
//!
//! Each entry is the ORIGINAL split regex from the family's tokenizer.json -
//! source-verified, never paraphrased (a subtly different regex tokenizes
//! differently and poisons logit parity). Unknown `tokenizer.ggml.pre` values
//! are a hard error by design: silently falling back to the GPT-2 default is
//! exactly the class of quiet wrongness we don't ship.

use tokenizers::SplitDelimiterBehavior;
use tokenizers::pre_tokenizers::PreTokenizerWrapper;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};

use crate::TokenizerError;

/// o200k (GPT-4o / gpt-oss family). Verified against llama.cpp llama-vocab.cpp
/// ("original regex from tokenizer.json", GPT4O case).
const O200K: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Classic GPT-2 pattern - the "default" pre for older byte-level BPE files.
const GPT2: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

/// Qwen3.5/3.6 family. Its own llama.cpp pre-type (LLAMA_VOCAB_PRE_TYPE_QWEN35),
/// distinct from qwen2 by folding Unicode marks \p{M} into the letter runs -
/// original regex from tokenizer.json, verified against llama-vocab.cpp.
const QWEN35: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Qwen2 / Qwen2.5 / Qwen3-BASE family (LLAMA_VOCAB_PRE_TYPE_QWEN2). The Qwen3
/// base tokenizer - used by Qwen3-Embedding and Qwen3-Reranker - declares this
/// pre-type in its GGUF. Same shape as QWEN35 but letter runs are plain \p{L}+
/// (no \p{M} mark-folding) and the punctuation class excludes \p{M} likewise.
/// The standard llama.cpp QWEN2 regex; keep bit-identical to llama-vocab.cpp.
const QWEN2: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// The llama3 split, which llama.cpp also serves to `dbrx` and `smaug-bpe`
/// (one shared case, llama-vocab.cpp DBRX/SMAUG at ~301). IBM Granite 4.1
/// declares `tokenizer.ggml.pre = "dbrx"` with a gpt2 byte-level BPE over a
/// 100352 vocab, so this is the entry granite loads through. Contractions are
/// spelled as explicit [sS] classes rather than (?i:) - that is llama.cpp's
/// own spelling of the llama3 pattern, kept verbatim so our split matches the
/// same-weights oracle token for token.
const LLAMA3: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// IBM Granite-Vision 4.1 (`tokenizer.ggml.pre = "granite-docling"`) - the
/// original regex from the model's own tokenizer.json. Same 100352-entry
/// byte-level BPE as the granite text models, and the same split as LLAMA3
/// above except the contractions keep their `(?i:)` spelling.
///
/// Deliberately not what llama.cpp does. Upstream routes
/// LLAMA_VOCAB_PRE_TYPE_GRANITE_DOCLING onto its GPT2 case
/// (`'s|...| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)`), which differs on
/// three counts: unbounded digit runs instead of `\p{N}{1,3}`, digits allowed
/// to carry a leading space, and no `\r\n` / trailing `\s+` handling.
///
/// The vocab decides it. It holds exactly 1000 three-digit tokens - the
/// complete 000-999 set - and zero tokens of four or more digits, which only
/// happens when the BPE was trained behind `\p{N}{1,3}`. Under llama.cpp's
/// regex a run like "12345" becomes one pre-token that no vocab entry can
/// cover; under this one it splits "123"+"45" the way the merges expect.
///
/// So for this model the same-weights oracle is wrong, and greedy text parity
/// against llama.cpp will legitimately diverge on multi-digit numbers and
/// newline runs. That matters more than usual here: granite-vision's whole
/// job is emitting CSV/JSON full of numbers. Believe our split.
/// (Checked against llama-vocab.cpp and tokenizer.json.)
const GRANITE_DOCLING: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Laguna (poolside) main split - stage 2 of its two-stage pre-tokenizer.
/// Verbatim from poolside/Laguna-XS-2.1 tokenizer.json;
/// llama.cpp's LAGUNA case spells the (?i:) contractions as explicit [sS]
/// classes - same semantics, we keep the original.
const LAGUNA_MAIN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Laguna stage 1: newline runs MERGE with the following token (tokenizer.json
/// `Split((?:\r?\n)+(?!\r?\n), MergedWithNext)`). Not an isolated newline
/// split - llama.cpp approximates this with "[^\n]+|[\n]+"; we reproduce the
/// original exactly.
const LAGUNA_NL: &str = r"(?:\r?\n)+(?!\r?\n)";

/// DeepSeek-V3 family (`tokenizer.ggml.pre = "deepseek-v3"` - the DeepSeek-OCR
/// checkpoints declare it; vocab 129280). A THREE-stage split sequence,
/// reproduced 1:1 from baidu/Unlimited-OCR's tokenizer.json:
/// digit runs capped at 3 first, then CJK/kana runs, then the main pattern,
/// then ByteLevel. The main pattern is unusual - punctuation-prefixed ASCII
/// words get their own branch, and the "letter run" class keys on \p{P}\p{S}
/// rather than \p{N} - so no existing single-stage entry can stand in for it.
const DEEPSEEK3_NUM: &str = r"\p{N}{1,3}";
const DEEPSEEK3_CJK: &str = "[\u{4e00}-\u{9fa5}\u{3040}-\u{309f}\u{30a0}-\u{30ff}]+";
const DEEPSEEK3_MAIN: &str = r##"[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"##;

/// MiniCPM5 (`tokenizer.ggml.pre = "minicpm5"`; openbmb/MiniCPM5-2B, vocab
/// 130560, served under arch `llama`). Two-stage, reproduced 1:1 from the
/// model's own tokenizer.json: digit runs are capped at 3 FIRST, then a
/// llama3-shaped main split whose digit branch is the unbounded `\p{N}+` -
/// which is only correct because stage 1 already cut every run. llama.cpp's
/// MINICPM5 case carries the same pair (contractions spelled as [sS]
/// classes there; we keep the original `(?i:)`).
const MINICPM5_NUM: &str = r"\p{N}{1,3}";
const MINICPM5_MAIN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}+| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

pub fn build(pre: &str) -> Result<PreTokenizerWrapper, TokenizerError> {
    if pre == "minicpm5" {
        let mut stages = Vec::with_capacity(3);
        for pat in [MINICPM5_NUM, MINICPM5_MAIN] {
            let s = Split::new(
                SplitPattern::Regex(pat.to_owned()),
                SplitDelimiterBehavior::Isolated,
                false,
            )
            .map_err(|e| TokenizerError::Library(e.to_string()))?;
            stages.push(PreTokenizerWrapper::Split(s));
        }
        let mut byte_level = ByteLevel::default();
        byte_level.add_prefix_space = false;
        byte_level.use_regex = false;
        stages.push(PreTokenizerWrapper::ByteLevel(byte_level));
        return Ok(Sequence::new(stages).into());
    }
    if pre == "deepseek-v3" {
        let mut stages = Vec::with_capacity(4);
        for pat in [DEEPSEEK3_NUM, DEEPSEEK3_CJK, DEEPSEEK3_MAIN] {
            let s = Split::new(
                SplitPattern::Regex(pat.to_owned()),
                SplitDelimiterBehavior::Isolated,
                false,
            )
            .map_err(|e| TokenizerError::Library(e.to_string()))?;
            stages.push(PreTokenizerWrapper::Split(s));
        }
        let mut byte_level = ByteLevel::default();
        byte_level.add_prefix_space = false;
        byte_level.use_regex = false;
        stages.push(PreTokenizerWrapper::ByteLevel(byte_level));
        return Ok(Sequence::new(stages).into());
    }
    // Laguna is the one two-stage entry: Sequence[Split(nl, MergedWithNext),
    // Split(main, Isolated), ByteLevel] - the tokenizer.json structure
    // reproduced 1:1 rather than flattened into a single pattern.
    if pre == "laguna" {
        let nl = Split::new(
            SplitPattern::Regex(LAGUNA_NL.to_owned()),
            SplitDelimiterBehavior::MergedWithNext,
            false,
        )
        .map_err(|e| TokenizerError::Library(e.to_string()))?;
        let main = Split::new(
            SplitPattern::Regex(LAGUNA_MAIN.to_owned()),
            SplitDelimiterBehavior::Isolated,
            false,
        )
        .map_err(|e| TokenizerError::Library(e.to_string()))?;
        let mut byte_level = ByteLevel::default();
        byte_level.add_prefix_space = false;
        byte_level.use_regex = false;
        return Ok(Sequence::new(vec![
            PreTokenizerWrapper::Split(nl),
            PreTokenizerWrapper::Split(main),
            PreTokenizerWrapper::ByteLevel(byte_level),
        ])
        .into());
    }
    let pattern = match pre {
        // llama.cpp maps all of these onto the same o200k regex (verified in
        // llama-vocab.cpp: gpt-4o, llama4, minimax-m2 share the GPT4O case)
        "gpt-4o" | "llama4" | "minimax-m2" => O200K,
        "qwen35" => QWEN35,
        // Kolibri 1's tokenizer.json split regex is qwen2's verbatim
        // (`\p{N}{1}` == `\p{N}`); its converter stamps pre="kolibri1"
        "qwen2" | "kolibri1" => QWEN2,
        // granite 4.1 ships pre="dbrx"; llama.cpp routes dbrx to its llama3
        // case. smaug-bpe shares that case upstream but stays unlisted until a
        // model we serve declares it - an unknown pre must keep erroring.
        // nemotron 3.5's gguf ships pre="pixtral", which llama.cpp also
        // routes to LLAMA3 (with ignore_merges - our BPE sets that globally).
        "dbrx" | "pixtral" => LLAMA3,
        // granite-vision 4.1 ships pre="granite-docling"; we take its
        // tokenizer.json regex, not llama.cpp's GPT2 routing - see the const.
        "granite-docling" => GRANITE_DOCLING,
        "default" | "gpt-2" => GPT2,
        other => return Err(TokenizerError::UnknownPreTokenizer(other.to_owned())),
    };

    let split = Split::new(
        SplitPattern::Regex(pattern.to_owned()),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .map_err(|e| TokenizerError::Library(e.to_string()))?;

    // the split regex has already done the segmentation, so ByteLevel only
    // byte-maps (no prefix space, no second regex pass)
    let mut byte_level = ByteLevel::default();
    byte_level.add_prefix_space = false;
    byte_level.use_regex = false;

    Ok(Sequence::new(vec![
        PreTokenizerWrapper::Split(split),
        PreTokenizerWrapper::ByteLevel(byte_level),
    ])
    .into())
}

#[cfg(test)]
mod tests {
    use tokenizers::PreTokenizer;
    use tokenizers::tokenizer::{OffsetReferential, OffsetType, PreTokenizedString};

    fn segments(pre: &str, text: &str) -> Vec<String> {
        let p = super::build(pre).unwrap();
        let mut s = PreTokenizedString::from(text);
        p.pre_tokenize(&mut s).unwrap();
        s.get_splits(OffsetReferential::Original, OffsetType::Byte)
            .into_iter()
            .map(|(seg, _, _)| seg.to_owned())
            .collect()
    }

    /// The three-stage deepseek-v3 split vs segments captured from HF
    /// tokenizers running the checkpoint's own tokenizer.json.
    /// Pins the two family quirks: digit runs cap at 3 before the main
    /// pattern sees them, and punctuation-prefixed ASCII words ("(hello")
    /// stay one segment.
    #[test]
    fn deepseek_v3_splits_like_the_reference() {
        assert_eq!(
            segments("deepseek-v3", "Revenue reached 12.4 million (up 1234%)."),
            [
                "Revenue",
                "Ġreached",
                "Ġ",
                "12",
                ".",
                "4",
                "Ġmillion",
                "Ġ(",
                "up",
                "Ġ",
                "123",
                "4",
                "%)."
            ]
        );
        assert_eq!(
            segments("deepseek-v3", "文档解析123テスト"),
            ["æĸĩæ¡£è§£æŀĲ", "123", "ãĥĨãĤ¹ãĥĪ"]
        );
        assert_eq!(
            segments("deepseek-v3", "(hello) .World  x\ny"),
            ["(hello", ")", "Ġ.", "World", "Ġ", "Ġx", "Ċ", "y"]
        );
    }

    /// The two-stage minicpm5 split vs segments captured from HF tokenizers
    /// (0.22.1) running openbmb/MiniCPM5-2B's own tokenizer.json. Pins the
    /// 3-digit cap landing BEFORE the main pattern ("2026" -> "202","6"), the
    /// `(?i:)` contractions, and the llama3-shaped punctuation run that
    /// swallows a following newline pair (".\n\n" is one segment).
    #[test]
    fn minicpm5_splits_like_the_reference() {
        assert_eq!(
            segments("minicpm5", "Revenue reached 12.4 million (up 1234%)."),
            [
                "Revenue",
                "Ġreached",
                "Ġ",
                "12",
                ".",
                "4",
                "Ġmillion",
                "Ġ(",
                "up",
                "Ġ",
                "123",
                "4",
                "%)."
            ]
        );
        assert_eq!(
            segments("minicpm5", "I'm sure it's 2026-09-22.\n\nNext line."),
            [
                "I", "'m", "Ġsure", "Ġit", "'s", "Ġ", "202", "6", "-", "09", "-", "22", ".ĊĊ",
                "Next", "Ġline", "."
            ]
        );
        assert_eq!(
            segments("minicpm5", "文档解析123テスト"),
            ["æĸĩæ¡£è§£æŀĲ", "123", "ãĥĨãĤ¹ãĥĪ"]
        );
    }
}
