//! On-GPU model graphs, assembled from the verified GpuExecutor building
//! blocks. Acceptance bar for every family: greedy-decode token parity with
//! the newest llama.cpp release binary serving the identical GGUF
//! (same-weights reference; no CPU references).

pub mod deepseek_ocr;
pub mod dinov3;
pub mod gemma4;
pub mod gpt_oss;
pub mod granite;
pub mod kumo;
pub mod laguna;
pub mod laya;
pub mod nemotron;
pub mod paddleocr_vl;
pub mod picture_store;
pub mod pillow;
pub mod prefix_cache;
pub mod qwen3;
pub mod qwen35;
pub mod qwen3_asr;
pub mod qwen4exp;
pub mod qwen_image;
pub(crate) mod st_load;
pub mod whisper;
