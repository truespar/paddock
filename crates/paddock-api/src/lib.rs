//! Wire-format types for Paddock's public APIs.
//!
//! Conformance is a correctness feature (north star): every struct here
//! must serialize to exactly what the official OpenAI / Anthropic specs describe.
//! The conformance suites under `crates/paddock-runner/tests` run the official
//! SDKs against these shapes - if a field name or nesting differs from the
//! spec, that's a bug here, not in the client.

pub mod chat;
pub mod completions;
pub mod embeddings;
pub mod error;
pub mod images;
pub mod masks;
pub mod messages;
pub mod models;
pub mod responses;
pub mod segmentation;

pub use completions::{CompletionChoice, CompletionRequest, CompletionResponse, Prompt, Usage};
pub use error::{ApiError, ErrorBody};
pub use models::{ModelList, ModelObject};
