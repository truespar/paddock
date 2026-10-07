//! Native Metal execution for Paddock. Model state stays on its engine thread;
//! generation, encoder and Whisper contracts share the native serving boundary.
#[cfg(target_os = "macos")]
mod embedding_gemma2;
#[cfg(target_os = "macos")]
pub use embedding_gemma2::EmbeddingGemma2;
#[cfg(target_os = "macos")]
mod kolibri;
#[cfg(target_os = "macos")]
pub use kolibri::Kolibri;
#[cfg(target_os = "macos")]
mod clef;
#[cfg(target_os = "macos")]
pub use clef::Clef;
#[cfg(target_os = "macos")]
mod diarization;
#[cfg(target_os = "macos")]
pub use diarization::Diarization;
#[cfg(target_os = "macos")]
mod kumo;
#[cfg(target_os = "macos")]
pub use kumo::{Kumo, KumoOutput};
#[cfg(target_os = "macos")]
mod laya;
#[cfg(target_os = "macos")]
pub use laya::Laya;
#[cfg(target_os = "macos")]
mod whisper;
#[cfg(target_os = "macos")]
pub use whisper::Whisper;
#[cfg(target_os = "macos")]
mod affine;
#[cfg(all(test, target_os = "macos"))]
mod affine_few_tests;
#[cfg(all(test, target_os = "macos"))]
mod affine_stable_tests;
#[cfg(all(test, target_os = "macos"))]
mod affine_tests;
#[cfg(target_os = "macos")]
mod device;
#[cfg(target_os = "macos")]
mod telemetry;
#[cfg(target_os = "macos")]
pub use telemetry::telemetry_snapshot;
#[cfg(target_os = "macos")]
mod offload;
#[cfg(target_os = "macos")]
mod paged_offload;
#[cfg(all(test, target_os = "macos"))]
mod paged_offload_tests;
#[cfg(target_os = "macos")]
mod splash;
#[cfg(target_os = "macos")]
pub use offload::{KvOffloadConfig, configure_kv_offload, kv_offload_config};
#[cfg(all(test, target_os = "macos"))]
mod mlx_operation_tests;
#[cfg(target_os = "macos")]
mod nemotron;
#[cfg(target_os = "macos")]
mod paddleocr;
#[cfg(target_os = "macos")]
mod unlimited_ocr;
#[cfg(target_os = "macos")]
pub use paddleocr::PaddleOcr;
#[cfg(target_os = "macos")]
pub use unlimited_ocr::UnlimitedOcr;
#[cfg(target_os = "macos")]
mod qwen3_asr;
#[cfg(target_os = "macos")]
pub use qwen3_asr::Qwen3Asr;
#[cfg(target_os = "macos")]
pub use qwen3_asr::aligner::Qwen3Aligner;
#[cfg(target_os = "macos")]
mod iquant;
#[cfg(target_os = "macos")]
mod qwen4exp;
#[cfg(target_os = "macos")]
pub use qwen4exp::{FlashNext, FlashNextMlxPlan, FlashNextPlan};
#[cfg(target_os = "macos")]
mod projection;
#[cfg(target_os = "macos")]
mod weights;
#[cfg(target_os = "macos")]
pub use device::*;
#[cfg(target_os = "macos")]
pub use nemotron::Nemotron;
#[cfg(target_os = "macos")]
mod granite;
#[cfg(all(test, target_os = "macos"))]
mod granite_attention_tests;
#[cfg(target_os = "macos")]
mod schedule;
#[cfg(target_os = "macos")]
pub use granite::Granite;
#[cfg(target_os = "macos")]
mod qwen35;
#[cfg(target_os = "macos")]
pub use qwen35::Qwen35;
#[cfg(target_os = "macos")]
mod gemma4;
#[cfg(target_os = "macos")]
pub use gemma4::Gemma4;
#[cfg(target_os = "macos")]
mod qwen3;
#[cfg(target_os = "macos")]
mod qwen_image;
#[cfg(target_os = "macos")]
pub use qwen_image::QwenImage;
#[cfg(target_os = "macos")]
pub use qwen3::Qwen3Encoder;
#[cfg(target_os = "macos")]
mod gpt_oss;
#[cfg(target_os = "macos")]
pub use gpt_oss::GptOss;
#[cfg(target_os = "macos")]
mod laguna;
#[cfg(target_os = "macos")]
pub use laguna::Laguna;

#[cfg(all(test, target_os = "macos"))]
mod tests;
