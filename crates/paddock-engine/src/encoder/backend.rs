//! Backend-owned completion handles keep CUDA events and Metal command buffers
//! out of the request scheduler. Construction and all calls stay on its thread;
//! neither the backend nor its pending handles need to be Send/Sync.

pub trait EncoderBackend {
    type Pending;
    fn weights_mem_bytes(&self) -> Option<u64>;
    fn device_mem_used(&self) -> Option<u64>;
    /// Opt-in scratch reclamation after this much queue-and-device idle time.
    /// Return None once reclaimed to avoid waking a completely idle runner.
    fn idle_reclaim_after(&self) -> Option<std::time::Duration> {
        None
    }
    /// Called only with no queued or uncollected work. Never evict weights.
    fn reclaim_idle(&mut self) {}
    fn coalesce_row_budget(&self) -> usize;
    /// How long an IDLE encoder holds a lone job after a merged batch, to
    /// catch its clients' turnaround burst: (wait for a second job, trailing
    /// wait per further arrival). None launches at once. The default suits
    /// a pass with a large fixed cost; a backend whose small pass is cheap
    /// does better launching and letting the in-flight merge absorb the rest.
    fn burst_windows(&self) -> Option<(std::time::Duration, std::time::Duration)> {
        Some((
            std::time::Duration::from_millis(20),
            std::time::Duration::from_millis(5),
        ))
    }
    /// Validate each request before merging, so one malformed caller cannot
    /// fail unrelated callers sharing its GPU batch. No device work here.
    fn validate(&self, _seqs: &[Vec<u32>]) -> Result<(), String> {
        Ok(())
    }
    fn lanes(&mut self) -> usize;
    fn inflight_capacity(&mut self) -> usize {
        2 * self.lanes()
    }
    fn pool_ready(&self, pending: &Self::Pending) -> bool;
    fn embed_submit(&mut self, seqs: &[Vec<u32>], lane: usize) -> Result<Self::Pending, String>;
    fn validate_dimensions(&self, dimensions: Option<usize>) -> Result<(), String> {
        if dimensions.is_some() {
            Err("this encoder does not support Matryoshka dimensions".into())
        } else {
            Ok(())
        }
    }
    fn embed_submit_dimensions(
        &mut self,
        seqs: &[Vec<u32>],
        lane: usize,
        dimensions: Option<usize>,
    ) -> Result<Self::Pending, String> {
        self.validate_dimensions(dimensions)?;
        self.embed_submit(seqs, lane)
    }
    /// Admission for a sequence batch that may carry media (`media[i]` is
    /// sequence i's pictures / clips, placeholder-run order). A text-only
    /// backend accepts only empty media lists.
    fn validate_media(
        &self,
        seqs: &[Vec<u32>],
        media: &[Vec<crate::service::MmChunk>],
    ) -> Result<(), String> {
        if media.iter().any(|m| !m.is_empty()) {
            return Err("this encoder embeds text only".into());
        }
        self.validate(seqs)
    }
    /// The media this backend embeds: (pictures, audio). Known once built
    /// (an mmproj may carry a tower the pack cannot run).
    fn media_kinds(&self) -> (bool, bool) {
        (false, false)
    }
    /// Submit with media; a text-only backend takes the text path.
    fn embed_submit_media(
        &mut self,
        seqs: &[Vec<u32>],
        media: &[Vec<crate::service::MmChunk>],
        lane: usize,
        dimensions: Option<usize>,
    ) -> Result<Self::Pending, String> {
        if media.iter().any(|m| !m.is_empty()) {
            return Err("this encoder embeds text only".into());
        }
        self.embed_submit_dimensions(seqs, lane, dimensions)
    }
    fn embed_collect(&mut self, pending: &Self::Pending) -> Result<Vec<Vec<f32>>, String>;
    fn rerank_submit(
        &mut self,
        seqs: &[Vec<u32>],
        yes: u32,
        no: u32,
        lane: usize,
    ) -> Result<Self::Pending, String>;
    fn rerank_collect(
        &mut self,
        pending: &Self::Pending,
        yes: u32,
        no: u32,
    ) -> Result<Vec<f32>, String>;
    fn block_scale_calibration(&self) -> bool {
        false
    }
    fn calibrate_bs(
        &mut self,
        _seqs: &[Vec<u32>],
        _n_docs: usize,
        _rel: &[usize],
    ) -> Result<&'static str, String> {
        Err("this encoder has no block-scale calibration".into())
    }
    fn calibrate_bs_rerank(
        &mut self,
        _seqs: &[Vec<u32>],
        _yes: u32,
        _no: u32,
        _group: usize,
        _rel: &[usize],
    ) -> Result<&'static str, String> {
        Err("this encoder has no block-scale calibration".into())
    }
    fn import_smooth(&mut self, _bytes: &[u8]) -> Result<bool, String> {
        Ok(false)
    }
    fn apply_bs_profile(&mut self, _profile: &str) -> Result<bool, String> {
        Ok(false)
    }
    fn export_smooth(&self) -> Result<Option<Vec<u8>>, String> {
        Ok(None)
    }
}

#[cfg(feature = "cuda")]
impl EncoderBackend for crate::gpu_model::embedding_gemma2::GpuEmbeddingGemma2 {
    type Pending = crate::gpu_model::embedding_gemma2::PendingEmbedding;
    fn burst_windows(&self) -> Option<(std::time::Duration, std::time::Duration)> {
        self.burst_windows()
    }
    fn media_kinds(&self) -> (bool, bool) {
        (self.serves_images(), self.serves_audio())
    }
    fn weights_mem_bytes(&self) -> Option<u64> {
        self.weights_mem_bytes()
    }
    fn device_mem_used(&self) -> Option<u64> {
        self.device_mem_used()
    }
    fn idle_reclaim_after(&self) -> Option<std::time::Duration> {
        self.idle_reclaim_after()
    }
    fn reclaim_idle(&mut self) {
        self.reclaim_idle();
    }
    fn coalesce_row_budget(&self) -> usize {
        self.coalesce_row_budget()
    }
    fn validate(&self, seqs: &[Vec<u32>]) -> Result<(), String> {
        self.validate(seqs)
    }
    fn lanes(&mut self) -> usize {
        1
    }
    fn pool_ready(&self, p: &Self::Pending) -> bool {
        self.pool_ready(p)
    }
    fn embed_submit(&mut self, s: &[Vec<u32>], lane: usize) -> Result<Self::Pending, String> {
        EncoderBackend::embed_submit_dimensions(self, s, lane, None)
    }
    fn validate_dimensions(&self, dims: Option<usize>) -> Result<(), String> {
        crate::gpu_model::embedding_gemma2::GpuEmbeddingGemma2::validate_dimensions(dims)
    }
    fn embed_submit_dimensions(
        &mut self,
        s: &[Vec<u32>],
        lane: usize,
        dims: Option<usize>,
    ) -> Result<Self::Pending, String> {
        if lane != 0 {
            return Err("invalid embedding lane".into());
        }
        self.embed_submit_dimensions(s, dims)
    }
    fn validate_media(
        &self,
        seqs: &[Vec<u32>],
        media: &[Vec<crate::service::MmChunk>],
    ) -> Result<(), String> {
        self.validate_media(seqs, media)
    }
    fn embed_submit_media(
        &mut self,
        seqs: &[Vec<u32>],
        media: &[Vec<crate::service::MmChunk>],
        lane: usize,
        dims: Option<usize>,
    ) -> Result<Self::Pending, String> {
        if lane != 0 {
            return Err("invalid embedding lane".into());
        }
        self.embed_submit_media(seqs, media, dims)
    }
    fn embed_collect(&mut self, p: &Self::Pending) -> Result<Vec<Vec<f32>>, String> {
        self.embed_collect(p)
    }
    fn rerank_submit(
        &mut self,
        _: &[Vec<u32>],
        _: u32,
        _: u32,
        _: usize,
    ) -> Result<Self::Pending, String> {
        Err("EmbeddingGemma 2 is an embedder, not a yes/no reranker".into())
    }
    fn rerank_collect(&mut self, _: &Self::Pending, _: u32, _: u32) -> Result<Vec<f32>, String> {
        Err("EmbeddingGemma 2 is an embedder, not a yes/no reranker".into())
    }
}

#[cfg(feature = "cuda")]
impl EncoderBackend for crate::gpu_model::qwen3::GpuQwen3 {
    type Pending = crate::gpu_model::qwen3::PendingPooled;
    fn weights_mem_bytes(&self) -> Option<u64> {
        self.weights_mem_bytes()
    }
    fn device_mem_used(&self) -> Option<u64> {
        self.device_mem_used()
    }
    fn coalesce_row_budget(&self) -> usize {
        self.coalesce_row_budget()
    }
    fn lanes(&mut self) -> usize {
        self.lanes()
    }
    fn pool_ready(&self, p: &Self::Pending) -> bool {
        self.pool_ready(p)
    }
    fn embed_submit(&mut self, s: &[Vec<u32>], lane: usize) -> Result<Self::Pending, String> {
        self.embed_submit(s, lane).map_err(|e| e.to_string())
    }
    fn embed_collect(&mut self, p: &Self::Pending) -> Result<Vec<Vec<f32>>, String> {
        self.embed_collect(p).map_err(|e| e.to_string())
    }
    fn rerank_submit(
        &mut self,
        s: &[Vec<u32>],
        y: u32,
        n: u32,
        lane: usize,
    ) -> Result<Self::Pending, String> {
        self.rerank_submit(s, y, n, lane).map_err(|e| e.to_string())
    }
    fn rerank_collect(&mut self, p: &Self::Pending, y: u32, n: u32) -> Result<Vec<f32>, String> {
        self.rerank_collect(p, y, n).map_err(|e| e.to_string())
    }
    fn block_scale_calibration(&self) -> bool {
        true
    }
    fn calibrate_bs(
        &mut self,
        s: &[Vec<u32>],
        n: usize,
        r: &[usize],
    ) -> Result<&'static str, String> {
        self.calibrate_bs(s, n, r).map_err(|e| e.to_string())
    }
    fn calibrate_bs_rerank(
        &mut self,
        s: &[Vec<u32>],
        y: u32,
        n: u32,
        g: usize,
        r: &[usize],
    ) -> Result<&'static str, String> {
        self.calibrate_bs_rerank(s, y, n, g, r)
            .map_err(|e| e.to_string())
    }
    fn import_smooth(&mut self, b: &[u8]) -> Result<bool, String> {
        self.import_smooth(b).map_err(|e| e.to_string())
    }
    fn apply_bs_profile(&mut self, p: &str) -> Result<bool, String> {
        self.apply_bs_profile(p).map_err(|e| e.to_string())
    }
    fn export_smooth(&self) -> Result<Option<Vec<u8>>, String> {
        self.export_smooth().map_err(|e| e.to_string())
    }
}
