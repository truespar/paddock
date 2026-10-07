//! SAM 3's text tokenizer: CLIP byte-level BPE, the way Meta's
//! `sam3/model/tokenizer_ve.py::SimpleTokenizer` prepares a concept prompt.
//!
//! The BPE itself is `facebook/sam3`'s own `tokenizer.json` run by HF's
//! tokenizers - checked against Meta's tokenizer on 36 prompts (accents, CJK,
//! emoji, contractions, punctuation, a 40-character run, collapsed and
//! stripped whitespace): every id equal, once the one step the JSON does not
//! carry is done here first. Meta's cleaning is `ftfy.fix_text`, then
//! `html.unescape` TWICE, then whitespace collapse and lowercase; the JSON's
//! normalizer does the last two, and the double HTML5 unescape is done here
//! (htmlize, the same character-reference rules Python's `html.unescape`
//! follows).
//!
//! Known gap, stated rather than hidden: ftfy's mojibake repair is not
//! reproduced, so text that arrives already double-encoded (UTF-8 read as
//! Latin-1 and re-encoded) tokenizes as what it literally is.
//!
//! Layout is Meta's: `[SOT] + bpe + [EOT]`, zero-padded to 32. Meta truncates
//! a longer prompt and forces EOT into the last slot - silently. Paddock does
//! not truncate silently: a prompt past 30 BPE tokens is refused with what it
//! measured.

use std::path::Path;

use tokenizers::Tokenizer;

/// Tokens the text tower reads, start and end markers included.
pub const SAM3_CONTEXT: usize = 32;
/// CLIP's start-of-text and end-of-text ids.
pub const SAM3_SOT: u32 = 49406;
pub const SAM3_EOT: u32 = 49407;

#[derive(Debug, thiserror::Error)]
pub enum Sam3TokenizeError {
    #[error("sam3 tokenizer.json: {0}")]
    Load(String),
    #[error("sam3 tokenize: {0}")]
    Encode(String),
    #[error(
        "the prompt is {got} tokens; SAM 3 reads at most {max} between its start and end \
         markers - shorten it (concept prompts are short noun phrases)"
    )]
    TooLong { got: usize, max: usize },
}

/// One prompt, as the text tower consumes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sam3Tokens {
    /// `[SOT, bpe.., EOT, 0..]`, always [`SAM3_CONTEXT`] long
    pub ids: [u32; SAM3_CONTEXT],
    /// SOT through EOT; the rest is padding every consumer masks out
    pub valid: usize,
}

pub struct Sam3Tokenizer {
    inner: Tokenizer,
}

impl Sam3Tokenizer {
    pub fn from_file(path: &Path) -> Result<Self, Sam3TokenizeError> {
        let inner = Tokenizer::from_file(path)
            .map_err(|e| Sam3TokenizeError::Load(format!("{}: {e}", path.display())))?;
        Ok(Self { inner })
    }

    /// The BPE ids of a prompt, cleaned as Meta cleans it (no markers).
    pub fn bpe(&self, text: &str) -> Result<Vec<u32>, Sam3TokenizeError> {
        let once = htmlize::unescape(text);
        let twice = htmlize::unescape(once.as_ref());
        let enc = self
            .inner
            .encode(twice.as_ref(), false)
            .map_err(|e| Sam3TokenizeError::Encode(e.to_string()))?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn encode(&self, text: &str) -> Result<Sam3Tokens, Sam3TokenizeError> {
        let bpe = self.bpe(text)?;
        let max = SAM3_CONTEXT - 2;
        if bpe.len() > max {
            return Err(Sam3TokenizeError::TooLong {
                got: bpe.len(),
                max,
            });
        }
        let mut ids = [0u32; SAM3_CONTEXT];
        ids[0] = SAM3_SOT;
        ids[1..=bpe.len()].copy_from_slice(&bpe);
        ids[bpe.len() + 1] = SAM3_EOT;
        Ok(Sam3Tokens {
            ids,
            valid: bpe.len() + 2,
        })
    }
}
