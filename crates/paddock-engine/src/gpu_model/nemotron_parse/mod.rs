//! Nemotron 3.5 Parse - NVIDIA's 0.63B document parser
//! (`nvidia/Nemotron-3.5-Parse`, OpenMDW-1.1): a page picture in, the page
//! out as boxed, classified, reading-ordered elements
//! (`<x_..><y_..>text<x_..><y_..><class_..>`, tables as LaTeX `tabular`).
//!
//! Two parts, both served from the release's own F32 safetensors:
//! - [`vision::RadioTower`]: C-RADIOv4-SO400M (27 global-attention ViT blocks,
//!   1152 wide, 16 x 72 heads) with a content-normalized position field and
//!   the release's neck, turning one page into `(H/16)(W/64) + 1` prefix rows;
//! - the decoder: 10 mBART layers used as a causal decoder-only LM over
//!   `[picture prefix | prompt | output]` - no position embeddings of any kind,
//!   GQA 16 q / 8 kv at hd 64, LM head tied to the embedding, plus one trained
//!   MTP head for one-token speculation.
//!
//! Numeric class: f16 operands, f32 accumulation, residual and norms (closer
//! to the F32 masters than the vendor's BF16 serving - the gate holds us to
//! that). Reference: the release's `hf_nemotron_parse_modeling.py`,
//! `hf_prefix_geometry.py` and `hf_nemotron_parse_processor.py`, run in F64.

pub mod batch;
pub mod chunked;
pub mod forward;
pub mod load;
pub mod preprocess;
pub mod spec;
pub mod vision;
pub mod walk;

pub use crate::gpu_model::gpt_oss::GpuModelError;
pub use load::GpuNemotronParse;
pub use preprocess::{PagePrep, prep_page};
pub use vision::{PrefixPlane, RadioTower, TOWER_PASS_ROWS};

/// A captured decode tick. Single-threaded on the engine's thread (the same
/// argument every other family's SendGraph makes).
pub(crate) struct SendGraph(pub(crate) crate::gpu::CapturedGraph);
// SAFETY: never accessed from two threads at once; see above.
unsafe impl Send for SendGraph {}
