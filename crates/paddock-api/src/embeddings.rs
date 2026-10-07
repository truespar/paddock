//! Embedding + rerank wire types.
//!
//! `/v1/embeddings` is the OpenAI shape (input: string | [string] | [int] |
//! [[int]]; output: a list of {embedding, index}). `/v1/rerank` follows the
//! de-facto Cohere/Jina shape (query + documents -> scored, sorted results),
//! which has no OpenAI equivalent but is what RAG clients speak.

use serde::{Deserialize, Serialize};

/// `POST /v1/embeddings`. Unknown fields rejected (spec-honesty, like chat).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingRequest {
    pub model: String,
    /// string | array of strings | array of token ids | array of token-id arrays.
    pub input: serde_json::Value,
    /// "float" (default) or "base64" little-endian f32 bytes.
    #[serde(default)]
    pub encoding_format: Option<String>,
    /// Matryoshka prefix width; accepted only by models trained for it.
    #[serde(default)]
    pub dimensions: Option<usize>,
    /// Optional published task prompt, for task-aware embedding models.
    /// Omitted means raw input: callers can supply their own exact prefix.
    #[serde(default)]
    pub task: Option<String>,
    /// Telemetry hint, accepted and unused.
    #[serde(default)]
    pub user: Option<String>,
}

/// An embedding on the wire: a JSON float array (`encoding_format` "float",
/// default) or a base64 string of the little-endian f32 bytes ("base64" - the
/// OpenAI-standard compact form: ~2.3x smaller than the float text and far
/// cheaper to serialize/transfer/parse for RAG-scale batches).
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum EmbeddingValue {
    Floats(Vec<f32>),
    Base64(String),
}

#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingData {
    pub object: &'static str, // "embedding"
    pub embedding: EmbeddingValue,
    pub index: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingResponse {
    pub object: &'static str, // "list"
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

/// `POST /v1/rerank` (Cohere/Jina-compatible).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RerankRequest {
    pub model: String,
    pub query: String,
    pub documents: Vec<String>,
    /// Return only the top-N results (default: all).
    #[serde(default)]
    pub top_n: Option<usize>,
    /// Echo each document's text back in the results.
    #[serde(default)]
    pub return_documents: bool,
    /// Optional task instruction for the relevance judge.
    #[serde(default)]
    pub instruction: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RerankResult {
    pub index: usize,
    pub relevance_score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RerankResponse {
    pub model: String,
    pub results: Vec<RerankResult>,
    pub usage: EmbeddingUsage,
}
