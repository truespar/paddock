//! EmbeddingGemma 2's bidirectional text backbone. Ragged requests share the
//! projections, not attention domains. No generative KV cache, causal mask,
//! padded-token pooling or host model math. Multimodal towers feed this same
//! graph; a text-only load deliberately does not allocate their weights.
//!
//! Bring-up status: same-Q8 GGUF text checks pass. Affine8 MLX text passes the
//! strict single-input reference through 8K; native batching is bit-stable.
//! The separately retained upstream batched-reference gate still fails:
//! upstream changes its own arithmetic with batch shape. GGUF picture/audio
//! towers pass the same-weight media
//! references (audio compares the HF frontend separately from llama.cpp's
//! different log floor). MLX image/audio and sampled-video checks pass the
//! media tolerance, not bit parity. Retrieval-quality benchmarks and broad
//! hardware/rival qualification remain open.
mod audio;
mod forward;
mod load;
mod media;
mod media_mlx;
mod scratch;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_attention;
#[cfg(test)]
mod tests_media;
#[cfg(test)]
mod tests_projection;
mod vision;

use crate::device::{Buffer, Completion, MetalDevice, MetalError, Result};
use crate::weights::Weight;
use paddock_engine::encoder::EncoderBackend;
use std::path::Path;

const WIDTH: usize = 512;
const FF: usize = 2048;
const LAYERS: usize = 24;
const VOCAB: usize = 262144;
const DIM: usize = 768;
const CONTEXT: usize = 8192;

fn error(s: impl Into<String>) -> MetalError {
    MetalError::Model(s.into())
}

struct Layer {
    pre: Weight,
    post: Weight,
    ff_pre: Weight,
    ff_post: Weight,
    q: Weight,
    k: Weight,
    v: Weight,
    o: Weight,
    qn: Weight,
    kn: Weight,
    gate: Weight,
    up: Weight,
    down: Weight,
    ple_gate: Weight,
    ple_out: Weight,
    ple_norm: Weight,
    scalar: Weight,
}
struct Scratch {
    rows: usize,
    x: Buffer,
    norm: Buffer,
    delta: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    attn: Buffer,
    gate: Buffer,
    up: Buffer,
    ple: Buffer,
    token_out: Buffer,
    scores: Buffer,
}
pub struct EmbeddingGemma2 {
    device: MetalDevice,
    identity: std::rc::Rc<()>,
    embedding: Weight,
    ple: Weight,
    ple_norm: Weight,
    norm: Weight,
    output: Weight,
    layers: Vec<Layer>,
    scratch: Option<std::rc::Rc<Scratch>>,
    context: usize,
    capacity: usize,
    mlx: bool,
    weight_bytes: u64,
    vision: Option<vision::Vision>,
    audio: Option<audio::Audio>,
}
pub struct PendingEmbedding {
    // Fence before freeing result/input allocations even on cancellation.
    completion: Completion,
    identity: std::rc::Rc<()>,
    output: Buffer,
    count: usize,
    dimensions: usize,
    _inputs: Vec<Buffer>,
    // Retain the exact scratch generation until this command completes. A
    // larger enqueue-ahead request can allocate a replacement safely.
    _scratch: std::rc::Rc<Scratch>,
    #[cfg(test)]
    trace: Option<Buffer>,
}

impl EncoderBackend for EmbeddingGemma2 {
    type Pending = PendingEmbedding;
    fn weights_mem_bytes(&self) -> Option<u64> {
        Some(self.weight_bytes)
    }
    fn device_mem_used(&self) -> Option<u64> {
        Some(self.device.allocated_bytes())
    }
    fn idle_reclaim_after(&self) -> Option<std::time::Duration> {
        (self.scratch.is_some()
            || self
                .vision
                .as_ref()
                .is_some_and(vision::Vision::has_workspace))
        .then(|| std::time::Duration::from_secs(2))
    }
    fn reclaim_idle(&mut self) {
        self.scratch = None;
        if let Some(v) = &mut self.vision {
            v.reclaim();
        }
    }
    fn media_kinds(&self) -> (bool, bool, bool) {
        (
            self.vision.is_some(),
            self.audio.is_some(),
            self.vision.is_some(),
        )
    }
    fn validate_media(
        &self,
        seqs: &[Vec<u32>],
        media: &[Vec<paddock_engine::service::MmChunk>],
    ) -> std::result::Result<(), String> {
        self.check_media(seqs, media)
    }
    fn embed_submit_media(
        &mut self,
        seqs: &[Vec<u32>],
        media: &[Vec<paddock_engine::service::MmChunk>],
        lane: usize,
        dim: Option<usize>,
    ) -> std::result::Result<Self::Pending, String> {
        self.check_media(seqs, media)?;
        self.validate_dimensions(dim)?;
        if lane != 0 {
            return Err("invalid embedding lane".into());
        }
        // Media need no text workspace until their features are ready.
        // Small generations already fit beside the tower, but drop
        // a large text cache before the media stage. At <=3072 rows the
        // <=282 MiB cache + the largest 1120-token tower/resize + retained
        // media features fit the catalog's workspace envelope. Audio groups
        // are bounded at 4096 projection rows and reuse frontend scratch. In-flight
        // work still owns its generation through PendingEmbedding.
        if self.scratch.as_ref().is_some_and(|s| s.rows > 3072)
            && media.iter().any(|m| !m.is_empty())
        {
            self.scratch = None;
        }
        let inputs = self.encode_media(seqs, media).map_err(|e| e.to_string())?;
        self.submit_with_media(seqs, dim.unwrap_or(DIM), inputs)
            .map_err(|e| e.to_string())
    }
    fn coalesce_row_budget(&self) -> usize {
        self.capacity
    }
    fn lanes(&mut self) -> usize {
        1
    }
    fn pool_ready(&self, p: &Self::Pending) -> bool {
        p.completion.ready()
    }
    fn validate(&self, seqs: &[Vec<u32>]) -> std::result::Result<(), String> {
        let rows = seqs.iter().try_fold(0usize, |n, s| n.checked_add(s.len()));
        if seqs.is_empty()
            || rows.is_none_or(|n| n > self.capacity)
            || seqs.iter().any(|s| {
                s.is_empty()
                    || s.len() > self.context
                    || s.iter()
                        .any(|&t| t as usize >= VOCAB || (258880..=258884).contains(&t))
            })
        {
            return Err(format!(
                "EmbeddingGemma 2 needs nonempty text sequences within {} tokens and {} batch rows; media placeholders require their matching media input",
                self.context, self.capacity
            ));
        }
        Ok(())
    }
    fn embed_submit(
        &mut self,
        seqs: &[Vec<u32>],
        lane: usize,
    ) -> std::result::Result<Self::Pending, String> {
        self.validate(seqs)?;
        if lane != 0 {
            return Err("invalid embedding lane".into());
        }
        self.submit(seqs, DIM).map_err(|e| e.to_string())
    }
    fn validate_dimensions(&self, dim: Option<usize>) -> std::result::Result<(), String> {
        if matches!(dim, None | Some(128 | 256 | 512 | 768)) {
            Ok(())
        } else {
            Err("EmbeddingGemma 2 dimensions must be 128, 256, 512 or 768".into())
        }
    }
    fn embed_submit_dimensions(
        &mut self,
        seqs: &[Vec<u32>],
        lane: usize,
        dim: Option<usize>,
    ) -> std::result::Result<Self::Pending, String> {
        self.validate(seqs)?;
        self.validate_dimensions(dim)?;
        if lane != 0 {
            return Err("invalid embedding lane".into());
        }
        self.submit(seqs, dim.unwrap_or(DIM))
            .map_err(|e| e.to_string())
    }
    fn embed_collect(&mut self, p: &Self::Pending) -> std::result::Result<Vec<Vec<f32>>, String> {
        if !std::rc::Rc::ptr_eq(&self.identity, &p.identity) {
            return Err("foreign embedding completion".into());
        }
        p.completion.wait().map_err(|e| e.to_string())?;
        let values = unsafe { p.output.read_f32(0, p.count * p.dimensions) };
        if values.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite embedding output".into());
        }
        Ok(values
            .chunks_exact(p.dimensions)
            .map(<[f32]>::to_vec)
            .collect())
    }
    fn rerank_submit(
        &mut self,
        _: &[Vec<u32>],
        _: u32,
        _: u32,
        _: usize,
    ) -> std::result::Result<Self::Pending, String> {
        Err("EmbeddingGemma 2 is an embedder, not a yes/no reranker".into())
    }
    fn rerank_collect(
        &mut self,
        _: &Self::Pending,
        _: u32,
        _: u32,
    ) -> std::result::Result<Vec<f32>, String> {
        Err("EmbeddingGemma 2 is an embedder, not a yes/no reranker".into())
    }
}
