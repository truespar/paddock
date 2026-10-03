//! Clef's one joint sequence per request, token for token the reference's
//! `encode_record`, because the ids are the model's input and the spans are
//! where its head reads:
//!
//! ```text
//! <|im_start|>system\n<SYSTEM_PROMPT><|im_end|>\n<|im_start|>user\nSTATE:\n
//! <state>
//! \n\nSCHEMA FIELDS:\n
//!   per question: \nFIELD <n>\nID: <id>\nTYPE: <type>\nINSTRUCTION: <instruction>
//!                 \nALLOWED OPTIONS:\n
//!                 per option: OPTION <m>: <option text>\n
//!                 END FIELD\n
//! \n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:
//! ```
//!
//! Every piece is tokenized on its own (`tokenizer(text,
//! add_special_tokens=False)`), so no BPE merge crosses a piece boundary -
//! that is what makes the instruction and option spans exact token ranges.
//!
//! Images go after `STATE:\n`, before the state: the reference's processor
//! tokenizes `<|vision_start|><|image_pad|><|vision_end|>` once per image
//! plus a newline as one piece, each pad expanded to the image's tokens
//! (`media_ids`).
//!
//! One departure: the reference cuts a state that does not fit
//! `max_length` silently. Here that is refused, with the counts, unless the
//! caller asked for the cut with `max_state_tokens` - then it is the
//! caller's cut, and the response says how many tokens were read.

use paddock_tokenizer::GgufTokenizer;

use super::question::Question;

/// The reference's `max_length` default.
pub const MAX_LENGTH: usize = 16384;

const SYSTEM_PROMPT: &str = "Read the complete state and schema. Decide every field jointly. \
                             Each answer must be exactly one of that field's allowed options.";

/// One question's place in the sequence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Spans {
    /// the instruction's tokens, `[start, end)`
    pub question: (usize, usize),
    /// each option's text tokens, encoding order
    pub options: Vec<(usize, usize)>,
}

#[derive(Clone, Debug)]
pub struct Encoded {
    pub ids: Vec<u32>,
    /// per question, request order
    pub spans: Vec<Spans>,
    /// the state's tokens before any cut, and how many the sequence carries
    pub state_tokens: usize,
    pub state_kept: usize,
    /// per image, the row of its first `<|image_pad|>`
    pub image_rows: Vec<usize>,
}

/// The media piece: `<|vision_start|><|image_pad|><|vision_end|>` per image
/// and a newline, tokenized as one text (the processor's), each pad
/// expanded to its image's `tokens`. Returns the ids and, per image, the
/// offset of its first pad inside them. Empty for no images.
pub fn media_ids(
    tok: &ClefTok,
    tokens: &[usize],
    pad: u32,
) -> Result<(Vec<u32>, Vec<usize>), String> {
    if tokens.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    checked_media_length(tokens.len(), tokens)?;
    let text = "<|vision_start|><|image_pad|><|vision_end|>".repeat(tokens.len()) + "\n";
    let piece = tok.encode(&text)?;
    if piece.iter().filter(|&&t| t == pad).count() != tokens.len() {
        return Err("the tokenizer does not carry the image placeholder tokens".into());
    }
    let length = checked_media_length(piece.len() - tokens.len(), tokens)?;
    let mut ids = Vec::with_capacity(length);
    let mut at = Vec::with_capacity(tokens.len());
    let mut k = 0;
    for t in piece {
        if t == pad {
            at.push(ids.len());
            ids.extend(std::iter::repeat_n(pad, tokens[k]));
            k += 1;
        } else {
            ids.push(t);
        }
    }
    Ok((ids, at))
}

/// Reject oversized/overflowing expansions before constructing the token Vec.
fn checked_media_length(overhead: usize, tokens: &[usize]) -> Result<usize, String> {
    tokens.iter().try_fold(overhead, |n, &t| {
        n.checked_add(t)
            .filter(|&n| t > 0 && n <= MAX_LENGTH)
            .ok_or_else(|| format!("images: image tokens exceed the {MAX_LENGTH}-token context"))
    })
}

pub struct ClefTok {
    tok: GgufTokenizer,
}

impl ClefTok {
    pub fn load(dir: &std::path::Path) -> Result<Self, String> {
        let tok = GgufTokenizer::from_hf_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        Ok(Self { tok })
    }

    /// The same BPE from a Clef GGUF's own vocabulary and merges (its
    /// `qwen35` pre-tokenizer) - token for token the checkpoint's
    /// tokenizer.json on the gate fixtures.
    pub fn from_gguf(g: &paddock_models::gguf::GgufFile) -> Result<Self, String> {
        let tok = GgufTokenizer::from_gguf(g).map_err(|e| format!("GGUF tokenizer: {e}"))?;
        Ok(Self { tok })
    }

    /// `tokenizer(text, add_special_tokens=False).input_ids`
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        self.tok.encode(text).map_err(|e| e.to_string())
    }

    pub fn vocab(&self) -> usize {
        self.tok.vocab_size()
    }
}

/// Build the sequence. `state` is the state's text (`render(state)`);
/// `media` the images' piece and each image's offset in it ([`media_ids`]);
/// `max_state_tokens` is the caller's own cut, if any.
pub fn encode(
    tok: &ClefTok,
    questions: &[Question],
    state: &str,
    media: (&[u32], &[usize]),
    max_length: usize,
    max_state_tokens: Option<usize>,
) -> Result<Encoded, String> {
    let mut schema = tok.encode("\n\nSCHEMA FIELDS:\n")?;
    let mut spans = Vec::with_capacity(questions.len());
    for (qi, q) in questions.iter().enumerate() {
        schema.extend(tok.encode(&format!(
            "\nFIELD {}\nID: {}\nTYPE: {}\nINSTRUCTION: ",
            qi + 1,
            q.id,
            q.kind.name()
        ))?);
        let qs = schema.len();
        schema.extend(tok.encode(&q.instruction)?);
        let question = (qs, schema.len());
        schema.extend(tok.encode("\nALLOWED OPTIONS:\n")?);
        let mut options = Vec::with_capacity(q.options.len());
        for (oi, o) in q.options.iter().enumerate() {
            schema.extend(tok.encode(&format!("OPTION {}: ", oi + 1))?);
            let os = schema.len();
            schema.extend(tok.encode(&o.text)?);
            options.push((os, schema.len()));
            schema.extend(tok.encode("\n")?);
        }
        schema.extend(tok.encode("END FIELD\n")?);
        if question.0 == question.1 || options.iter().any(|(s, e)| s == e) {
            // a span with no tokens has no mean - the reference reads NaN
            return Err(format!(
                "question {:?}: its instruction or an option tokenizes to nothing",
                q.id
            ));
        }
        spans.push(Spans { question, options });
    }
    let mut prefix = tok.encode(&format!(
        "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"
    ))?;
    let image_rows = media.1.iter().map(|o| prefix.len() + o).collect();
    prefix.extend_from_slice(media.0);
    let suffix = tok.encode(
        "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:",
    )?;
    let mut state_ids = tok.encode(state)?;
    let state_tokens = state_ids.len();
    if let Some(cap) = max_state_tokens {
        state_ids.truncate(cap);
    }
    let fixed = prefix.len() + schema.len() + suffix.len();
    if fixed > max_length {
        return Err(format!(
            "the schema and images need {fixed} tokens before the state; the most is \
             {max_length}"
        ));
    }
    let room = max_length - fixed;
    if state_ids.len() > room {
        if max_state_tokens.is_some() {
            state_ids.truncate(room);
        } else {
            return Err(format!(
                "state too long: {state_tokens} tokens with {fixed} for the schema is more than \
                 {max_length}; at most {room} state tokens fit (send max_state_tokens to read \
                 only the first ones)"
            ));
        }
    }
    let off = prefix.len() + state_ids.len();
    for s in &mut spans {
        s.question = (s.question.0 + off, s.question.1 + off);
        for o in &mut s.options {
            *o = (o.0 + off, o.1 + off);
        }
    }
    let state_kept = state_ids.len();
    let mut ids = prefix;
    ids.extend(state_ids);
    ids.extend(schema);
    ids.extend(suffix);
    Ok(Encoded {
        ids,
        spans,
        state_tokens,
        state_kept,
        image_rows,
    })
}

#[cfg(test)]
mod admission_tests {
    use super::*;
    #[test]
    fn media_expansion_is_checked_before_allocation() {
        assert_eq!(checked_media_length(5, &[64, 256]).unwrap(), 325);
        for tokens in [&[usize::MAX][..], &[MAX_LENGTH, 1], &[0], &[MAX_LENGTH - 4]] {
            assert!(checked_media_length(5, tokens).is_err());
        }
        assert_eq!(
            checked_media_length(4, &[MAX_LENGTH - 4]).unwrap(),
            MAX_LENGTH
        );
    }
}
