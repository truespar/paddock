//! On-GPU model graphs, assembled from the verified GpuExecutor building
//! blocks. Acceptance bar for every family: greedy-decode token parity with
//! the newest llama.cpp release binary serving the identical GGUF
//! (same-weights reference; no CPU references).

pub mod clef;
pub mod deepseek_ocr;
pub mod diarization;
pub mod dinov3;
pub mod doclayout;
pub mod embedding_gemma2;
pub mod gemma4;
pub mod gpt_oss;
pub mod granite;
pub mod kumo;
pub mod laguna;
pub mod laya;
pub mod nemotron;
pub mod paddleocr_vl;
pub mod picture_store;
// host-side and backend-neutral, so it lives at the crate root (the runner
// fits LightOnOCR pages with it on every backend); re-exported here for the
// towers that already name it by this path
pub use crate::pillow;
pub mod prefix_cache;
pub mod qwen3;
pub mod qwen35;
pub mod qwen3_asr;
pub mod qwen4exp;
pub mod qwen_image;
pub mod sam3;
pub(crate) mod st_load;
pub mod whisper;
