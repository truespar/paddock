//! How a sequence ended - the engine's half of the APIs' finish reasons.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// hit a stop token
    Stop,
    /// hit max_tokens
    Length,
    /// the sampler's repetition stop fired: the token picked would have
    /// repeated an n-gram already begun within its window (a degenerate
    /// loop), so the sequence ended before it. Reported as a stop on the
    /// wire; the document parsers' `ocr` echo says it fired.
    Repetition,
}

impl FinishReason {
    /// OpenAI `finish_reason` string.
    pub fn as_str(&self) -> &'static str {
        match self {
            FinishReason::Stop | FinishReason::Repetition => "stop",
            FinishReason::Length => "length",
        }
    }
}
