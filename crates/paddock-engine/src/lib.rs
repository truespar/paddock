//! The native inference engine.
//!
//! Design contracts:
//! - engine core owns its own loop and all mutable state (vLLM-V1 pattern);
//!   the server talks to it over channels only
//! - KV allocator supports per-layer cache kinds from day one:
//!   paged full-attention / sliding-window ring / recurrent state slots
//! - kernels come from versioned kernel packs (paddock-kernels), all in-house
//!
//! The first cut ships the skeleton: the backend trait and placeholder types,
//! so the server and harness can compile against stable seams while the
//! implementations fill in behind them.
pub mod align;
pub mod audio;
pub mod backend;
pub mod ckpt_pages;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod decision;
pub mod encoder;
pub mod envset;
pub mod generator;
#[cfg(feature = "cuda")]
pub mod gpu;
#[cfg(feature = "cuda")]
pub mod gpu_model;
pub mod granite_layout;
pub mod host_memory;
pub mod image;
pub mod kv_plan;
pub mod kv_pool;
pub mod kv_tier;
pub mod metrics;
pub mod pacing;
pub mod paged_radix;
#[cfg(feature = "cuda")]
pub mod reference;
pub mod sampler;
#[cfg(feature = "cuda")]
pub mod segment;
pub mod service;
pub mod spec;
pub mod spec_policy;
pub mod tabular;
pub mod tickseg;
pub mod transcriber;
pub mod whisper;

pub use backend::{Backend, BackendInfo};
