//! EmbeddingGemma 2 on CUDA - the text encoder of Google's multimodal
//! embedder (740M total; this is its 270M text backbone). Nothing is
//! generated: a sequence goes in, a 768-wide unit vector (or its 128 / 256 /
//! 512 Matryoshka prefix, re-normalized) comes out.
//!
//! The serving unit is a PASS of packed sequences - every queued request's
//! token rows back to back, `cu[s]..cu[s + 1]`, no padding (the Laya
//! encoder's layout). Every op is row-batched over the packed rows; attention
//! is the one that knows where a sequence ends (`gpu::eg2_attn`, per-sequence
//! tile lists).
//!
//!   ids -> x0 = E[ids] * sqrt(512)  (also the residual x)
//!   24 x { [this layer's PLE: rmsnorm(x0 . P_l / sqrt(512)) * w_ple]
//!          xn = attn_norm(x) -> q|k|v GEMM -> q/k rmsnorm + rope, v rmsnorm
//!          -> bidirectional attention (sliding |i - j| <= 512 at hd 256,
//!          full at hd 512) -> o GEMM -> x += post_norm(o); ffn_norm
//!          -> gate, up GEMMs -> gelu(gate) * up -> down GEMM
//!          -> x += post_ffw_norm(down) -> inp_gate GEMM on x
//!          -> gelu(g) * PLE_l -> proj GEMM -> x = (x + post_norm(proj)) * s_l }
//!   output_norm -> 512 -> 768 GEMM per token -> mean pool -> L2 normalize
//!
//! Reference: llama.cpp's `gemma-embedding2` graph on the identical GGUF
//! (black box over HTTP, the newest prebuilt release; never built). Precision
//! class: the reference's own on CUDA - Q8_0 weights against int8-quantized
//! activations (32-value blocks) on the mmq tensor cores, the BF16 PLE
//! projection on the bf16 tile, f16 q/k/v and P in attention with f32
//! accumulate, f32 residual stream and norms.
//!
//! Batch invariance: a vector does not depend on what else shared its pass.
//! The mmq GEMM runs plain-tiled (no stream-K fixup, whose split follows the
//! tile count), the PLE tile never splits K at in = 512 and is called
//! directly rather than through the band dispatcher, the encoder's own row
//! kernels keep one block shape whatever the row count, attention work per
//! (sequence, query tile, kv head) is laid from the sequence start, and the
//! pool sums in row order. Gated in `tests/gpu_embedding_gemma2.rs`.
//!
//! Media: pictures through Gemma 4's vision tower when the endpoint loads the
//! mmproj (`media.rs`); the audio Conformer and video frames are not served
//! yet, and their placeholders are refused at admission rather than embedded
//! as text.

pub mod audio;
mod forward;
mod load;
pub mod media;

use std::sync::Arc;

use cudarc::driver::CudaSlice;
use half::f16;

use crate::gpu::{GpuExecutor, QuantTensor, RepackedQ8};

pub use crate::gpu_model::gpt_oss::GpuModelError;

const WIDTH: usize = crate::gpu::EG2_WIDTH;
const LAYERS: usize = crate::gpu::EG2_LAYERS;
const FF: usize = 2048;
const VOCAB: usize = 262144;
/// The pooled output width (the projection's out dim).
pub const DIM: usize = 768;
/// The trained input budget; the rope tables go further, the training did not.
pub const CONTEXT: usize = 8192;
/// Token ids of the media placeholders (<|image|>, <|audio|>, <|video|> and
/// their delimiters): text-only serving refuses them instead of embedding a
/// placeholder as if it were text.
const MEDIA_IDS: std::ops::RangeInclusive<u32> = 258880..=258884;
/// Passes up to this many rows project every layer's PLE in one GEMM up
/// front (24 x 512 floats a row, twice); larger ones project each layer's
/// slice before the layer. Both land bit-identical values (the bf16 tile's
/// configs move ownership, never the K order, and K = 512 never splits).
const PLE_UPFRONT_ROWS: usize = 256;
/// Passes up to this many rows take the small-pass GEMM (GEMV shaped, a
/// lane a token) instead of the mmq tile - bit-identical outputs, so the
/// switch is pure speed. One 32-token group: past it the GEMV re-streams the
/// weights a group (GB10 pass times, 18 / 84 / 144 rows: mmq everywhere
/// 4.23 / 4.31 / 4.77 ms, GEMV to 32 rows 2.97 / 4.28 / 4.73, to 128 rows
/// 2.74 / 6.05 / 4.69).
const GEMV_ROWS: usize = 32;
/// The same band when the pack carries the medium-row mmq tile (slot 828):
/// that tile's pass time is flat across the band while the GEMV's grows with
/// the rows (GB10 pass times, 4 / 8 / 12 / 16 / 18 / 32 rows: GEMV 2.31 /
/// 2.38 / 2.50-2.61 / 2.60-2.64 / 2.84-2.96 / 3.49-3.61 ms, the tile 2.63 /
/// 2.68 / 2.68-2.73 / 2.65-2.69 / 2.73 / 2.73), so the GEMV keeps the
/// passes up to 16 rows.
const GEMV_ROWS_WITH_TILE: usize = 16;
/// Seconds of complete idleness before the scratch is dropped.
const IDLE_RECLAIM: std::time::Duration = std::time::Duration::from_secs(2);

struct Layer {
    /// 256 on the sliding layers, 512 on every sixth (full) one.
    hd: usize,
    qkv: RepackedQ8,
    o: RepackedQ8,
    /// gate | up concatenated: one GEMM lands [rows][2 ff], the GEGLU reads
    /// the halves in place
    gate_up: RepackedQ8,
    down: RepackedQ8,
    inp_gate: RepackedQ8,
    proj: RepackedQ8,
    attn_norm: CudaSlice<f32>,
    post_attn: CudaSlice<f32>,
    ffn_norm: CudaSlice<f32>,
    post_ffw: CudaSlice<f32>,
    post_norm: CudaSlice<f32>,
    q_norm: CudaSlice<f32>,
    k_norm: CudaSlice<f32>,
    /// `layer_output_scale` - one f32, read once at load.
    scale: f32,
}

/// Per-pass planes, sized for `rows_cap` packed rows and `seq_cap` sequences.
/// Grows with admitted work, dropped after [`IDLE_RECLAIM`] of idleness.
struct Scratch {
    rows_cap: usize,
    seq_cap: usize,
    ids: CudaSlice<u32>,
    pos: CudaSlice<u32>,
    cu: CudaSlice<u32>,
    tiles: [CudaSlice<u32>; 2],
    x0: CudaSlice<f32>,
    x: CudaSlice<f32>,
    delta: CudaSlice<f32>,
    g: CudaSlice<f32>,
    /// the PLE projection's landing: one layer's [rows][512], or (small
    /// passes) every layer's [rows][24][512]
    ple_raw: CudaSlice<f32>,
    /// the normalized PLE: [rows][512], or layer-major [24][rows][512]
    ple: CudaSlice<f32>,
    qkv: CudaSlice<f32>,
    attn: CudaSlice<f32>,
    gate_up: CudaSlice<f32>,
    tok: CudaSlice<f32>,
    q16: CudaSlice<f16>,
    k16: CudaSlice<f16>,
    v16: CudaSlice<f16>,
    yq: CudaSlice<u8>,
}

impl Scratch {
    fn tiles_for(rows: usize, seqs: usize, hd_rows: usize) -> usize {
        rows / hd_rows + seqs
    }

    fn new(exec: &GpuExecutor, rows_cap: usize, seq_cap: usize) -> Result<Self, GpuModelError> {
        let [r64, r16] = crate::gpu::EG2_ATTN_ROWS;
        let ple = (rows_cap * WIDTH).max(rows_cap.min(PLE_UPFRONT_ROWS) * WIDTH * LAYERS);
        Ok(Self {
            rows_cap,
            seq_cap,
            ids: exec.alloc_u32(rows_cap)?,
            pos: exec.alloc_u32(rows_cap)?,
            cu: exec.alloc_u32(seq_cap + 1)?,
            tiles: [
                exec.alloc_u32(Self::tiles_for(rows_cap, seq_cap, r64))?,
                exec.alloc_u32(Self::tiles_for(rows_cap, seq_cap, r16))?,
            ],
            x0: exec.alloc(rows_cap * WIDTH)?,
            x: exec.alloc(rows_cap * WIDTH)?,
            delta: exec.alloc(rows_cap * WIDTH)?,
            g: exec.alloc(rows_cap * WIDTH)?,
            ple_raw: exec.alloc(ple)?,
            ple: exec.alloc(ple)?,
            qkv: exec.alloc(rows_cap * (4 * 512 + 1024))?,
            attn: exec.alloc(rows_cap * 4 * 512)?,
            gate_up: exec.alloc(rows_cap * 2 * FF)?,
            tok: exec.alloc(rows_cap * DIM)?,
            q16: exec.alloc_f16(rows_cap * 4 * 512)?,
            k16: exec.alloc_f16(rows_cap * 512)?,
            v16: exec.alloc_f16(rows_cap * 512)?,
            // widest mmq input is 2048 (down, the full layers' o)
            yq: exec.alloc_u8(crate::gpu::mmq_bytes(2048, rows_cap))?,
        })
    }
}

/// A submitted pass: its pooled vectors land in `pool[flip]`, ready when
/// `ev` fires.
pub struct PendingEmbedding {
    ev: cudarc::driver::CudaEvent,
    flip: usize,
    count: usize,
    dims: usize,
}

pub struct GpuEmbeddingGemma2 {
    exec: Arc<GpuExecutor>,
    embd: QuantTensor,
    ple_proj: QuantTensor,
    ple_norm: CudaSlice<f32>,
    out_norm: CudaSlice<f32>,
    output: RepackedQ8,
    layers: Vec<Layer>,
    eps: f32,
    /// |i - j| bound of the sliding layers (the GGUF's window / 2).
    window: usize,
    /// rope `base^(-2 / hd)` for hd 256 and hd 512, computed on the host as
    /// the reference does.
    theta_scale: [f32; 2],
    context: usize,
    capacity: usize,
    scratch: Option<Scratch>,
    /// Two pooled-output buffers: the scheduler keeps at most two passes in
    /// flight, and a pass's output must outlive the next one's submit.
    pool: [Option<CudaSlice<f32>>; 2],
    flip: usize,
    weights_bytes: u64,
    /// The picture tower, when the endpoint loaded the mmproj.
    images: Option<media::ImageTower>,
    audio: Option<audio::AudioTower>,
}

impl GpuEmbeddingGemma2 {
    /// Resident weight bytes (the encoder's `weights_mem` line).
    pub fn weights_mem_bytes(&self) -> Option<u64> {
        Some(self.weights_bytes)
    }

    /// Device bytes this process holds live (weights + scratch).
    pub fn device_mem_used(&self) -> Option<u64> {
        self.exec.process_mem_used()
    }

    /// How long a fully idle encoder keeps its scratch; None once dropped.
    pub fn idle_reclaim_after(&self) -> Option<std::time::Duration> {
        self.scratch.as_ref().map(|_| IDLE_RECLAIM)
    }

    /// Drop the scratch (never the weights). The scheduler calls this only
    /// with nothing queued or uncollected, and frees are stream-ordered.
    pub fn reclaim_idle(&mut self) {
        self.scratch = None;
        self.pool = [None, None];
        // the picture tower's kept scratch goes with it (a picture pass
        // always builds `scratch`, so this runs after any picture)
        if let Some(t) = &mut self.images {
            t.release();
        }
    }

    /// The idle-burst merge windows (see `EncoderBackend::burst_windows`):
    /// none. A lone query pass is ~3 ms, and whatever arrives while a pass is
    /// in flight is merged into the next one anyway, so holding an idle GPU
    /// for a client's turnaround only idles it. Measured over HTTP on GB10
    /// (32 clients, one 18-token query a request): the shared 20 / 5 ms
    /// windows 1241 req/s with the GPU 31% busy; 5 / 1 ms 1641; 2 / 0.5 ms
    /// 1914; none 1920. Document and single-client cells moved within noise.
    pub fn burst_windows(&self) -> Option<(std::time::Duration, std::time::Duration)> {
        None
    }

    /// Most packed rows one pass takes.
    pub fn coalesce_row_budget(&self) -> usize {
        self.capacity
    }

    /// Admission, before any device work: one malformed caller must not fail
    /// the others that would have shared its pass.
    pub fn validate(&self, seqs: &[Vec<u32>]) -> Result<(), String> {
        let rows = seqs.iter().try_fold(0usize, |n, s| n.checked_add(s.len()));
        if seqs.is_empty()
            || rows.is_none_or(|n| n > self.capacity)
            || seqs.iter().any(|s| {
                s.is_empty()
                    || s.len() > self.context
                    || s.iter()
                        .any(|&t| t as usize >= VOCAB || MEDIA_IDS.contains(&t))
            })
        {
            return Err(format!(
                "EmbeddingGemma 2 needs nonempty text sequences within {} tokens and {} batch \
                 rows; media placeholders require their matching media input",
                self.context, self.capacity
            ));
        }
        Ok(())
    }

    /// Load the media towers from the companion mmproj (EmbeddingGemma 2's
    /// carries Gemma 4's `gemma4v` picture and `gemma4a` audio towers);
    /// `budget` is the soft-token budget a picture resizes to (one of
    /// [`media::IMAGE_TOKEN_BUDGETS`]). Audio needs the pack's slots 813-819;
    /// an older pack serves pictures only, and says so. `audio = false`
    /// leaves an audio tower in the file unloaded. Call once a file: the
    /// catalog ships the two towers as two files.
    pub fn attach_mmproj(
        &mut self,
        map: &paddock_models::mapped::MappedGguf,
        budget: usize,
        audio: bool,
    ) -> Result<(), GpuModelError> {
        let g = map.gguf();
        let s = |k: &str| {
            g.metadata
                .get(k)
                .and_then(paddock_models::gguf::Value::as_str)
        };
        let vision = s("clip.vision.projector_type") == Some("gemma4v");
        let has_audio = s("clip.audio.projector_type") == Some("gemma4a");
        if g.architecture() != Some("clip") || !(vision || has_audio) {
            return Err(GpuModelError::MissingMeta(
                "not an EmbeddingGemma 2 mmproj (a clip file with Gemma 4's gemma4v / gemma4a towers)"
                    .into(),
            ));
        }
        let before = self.exec.process_mem_used().unwrap_or(0);
        if vision {
            let tower =
                crate::gpu_model::gemma4::vision::VisionModel::load(self.exec.clone(), map, None)?;
            if tower.llm_embd() != WIDTH {
                return Err(GpuModelError::MissingMeta(format!(
                    "the mmproj projects to {} wide, the backbone is {WIDTH}",
                    tower.llm_embd()
                )));
            }
            self.images = Some(media::ImageTower::new(self.exec.clone(), tower, budget)?);
            tracing::info!(budget, "EmbeddingGemma 2 picture tower attached");
        }
        let audio = has_audio && audio;
        if audio && self.exec.has_eg2_audio() {
            self.audio = Some(audio::AudioTower::load(self.exec.clone(), map)?);
            tracing::info!("EmbeddingGemma 2 audio tower attached");
        } else if audio {
            tracing::warn!(
                "this kernel pack has no EmbeddingGemma 2 audio tower (slots 813-819); audio is not served"
            );
        }
        self.exec.synchronize()?;
        self.weights_bytes += self
            .exec
            .process_mem_used()
            .unwrap_or(0)
            .saturating_sub(before);
        Ok(())
    }

    /// The audio tower, when the mmproj carried one and the pack runs it.
    pub fn audio_tower(&mut self) -> Option<&mut audio::AudioTower> {
        self.audio.as_mut()
    }

    /// Whether audio clips are served.
    pub fn serves_audio(&self) -> bool {
        self.audio.is_some()
    }

    /// The picture tower, when the mmproj is attached.
    pub fn image_tower(&mut self) -> Option<&mut media::ImageTower> {
        self.images.as_mut()
    }

    /// Whether pictures are served (the mmproj is attached).
    pub fn serves_images(&self) -> bool {
        self.images.is_some()
    }

    /// Soft tokens a `w` x `h` picture takes on this endpoint - the run of
    /// [`media::IMAGE_TOKEN`] its sequence must carry.
    pub fn image_tokens(&self, w: usize, h: usize) -> Result<usize, String> {
        self.images
            .as_ref()
            .ok_or("this endpoint serves no pictures (no mmproj attached)")?
            .tokens_for(w, h)
    }

    /// Admission for sequences with media: every picture's soft-token run
    /// matches its picture in order and size, nothing is left over, and the
    /// text rules of [`Self::validate`] hold for everything else.
    pub fn validate_media(
        &self,
        seqs: &[Vec<u32>],
        media: &[Vec<crate::service::MmChunk>],
    ) -> Result<(), String> {
        use crate::service::MmChunk;
        if media.iter().all(Vec::is_empty) {
            return self.validate(seqs);
        }
        let rows = seqs.iter().try_fold(0usize, |n, s| n.checked_add(s.len()));
        if rows.is_none_or(|n| n > self.capacity) {
            return Err(format!("the batch is past {} rows", self.capacity));
        }
        for (i, seq) in seqs.iter().enumerate() {
            let items = media.get(i).map_or(&[][..], Vec::as_slice);
            if seq.is_empty() || seq.len() > self.context {
                return Err(format!(
                    "each input must contain 1..{} tokens, media included",
                    self.context
                ));
            }
            let runs = media::placeholder_runs(seq);
            if runs.len() != items.len() {
                return Err(format!(
                    "{} media placeholder runs for {} media inputs",
                    runs.len(),
                    items.len()
                ));
            }
            for (run, item) in runs.iter().zip(items) {
                match (run.token, item) {
                    (media::IMAGE_TOKEN, MmChunk::Image { w, h, rgb }) => {
                        if rgb.len() != w * h * 3 {
                            return Err("a picture's bytes do not match its size".into());
                        }
                        let want = self.image_tokens(*w, *h)?;
                        if run.len != want {
                            return Err(format!(
                                "a {w} x {h} picture takes {want} soft tokens, its run has {}",
                                run.len
                            ));
                        }
                    }
                    (media::AUDIO_TOKEN, MmChunk::Audio { samples, .. }) => {
                        if self.audio.is_none() {
                            return Err("this endpoint serves no audio (no audio tower)".into());
                        }
                        let want = audio::audio_tokens(samples.len())?;
                        if run.len != want {
                            return Err(format!(
                                "a clip of {} samples takes {want} soft tokens, its run has {}",
                                samples.len(),
                                run.len
                            ));
                        }
                    }
                    _ => return Err("a media input does not match its placeholder".into()),
                }
            }
            let text_ok = seq.iter().all(|&t| {
                (t as usize) < VOCAB
                    && (!MEDIA_IDS.contains(&t)
                        || matches!(
                            t,
                            media::IMAGE_TOKEN
                                | media::EOI_TOKEN
                                | media::AUDIO_TOKEN
                                | media::EOA_TOKEN
                        ))
            });
            if !text_ok {
                return Err("input holds an out-of-vocabulary or unmatched media token".into());
            }
        }
        Ok(())
    }

    /// Validate, then submit a pass whose sequences may carry media.
    pub fn embed_submit_media(
        &mut self,
        seqs: &[Vec<u32>],
        media: &[Vec<crate::service::MmChunk>],
        dims: Option<usize>,
    ) -> Result<PendingEmbedding, String> {
        self.validate_media(seqs, media)?;
        Self::validate_dimensions(dims)?;
        self.submit(seqs, media, dims.unwrap_or(DIM))
            .map_err(|e| e.to_string())
    }

    /// The Matryoshka sizes it was trained for.
    pub fn validate_dimensions(dims: Option<usize>) -> Result<(), String> {
        if matches!(dims, None | Some(128 | 256 | 512 | 768)) {
            Ok(())
        } else {
            Err("EmbeddingGemma 2 dimensions must be 128, 256, 512 or 768".into())
        }
    }

    /// Non-blocking: has a submitted pass completed?
    pub fn pool_ready(&self, p: &PendingEmbedding) -> bool {
        self.exec.event_done(&p.ev)
    }

    /// Read a pass's vectors (blocks only on its own event).
    pub fn embed_collect(&mut self, p: &PendingEmbedding) -> Result<Vec<Vec<f32>>, String> {
        let buf = self.pool[p.flip]
            .as_ref()
            .ok_or("embedding output was reclaimed before collection")?;
        let flat = self
            .exec
            .to_host_len_after(&p.ev, buf, p.count * p.dims)
            .map_err(|e| e.to_string())?;
        if flat.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite embedding output".into());
        }
        Ok(flat.chunks_exact(p.dims).map(<[f32]>::to_vec).collect())
    }

    /// Validate, then submit one pass at `dims` output components.
    pub fn embed_submit_dimensions(
        &mut self,
        seqs: &[Vec<u32>],
        dims: Option<usize>,
    ) -> Result<PendingEmbedding, String> {
        self.validate(seqs)?;
        Self::validate_dimensions(dims)?;
        self.submit(seqs, &[], dims.unwrap_or(DIM))
            .map_err(|e| e.to_string())
    }

    /// Blocking convenience for tests and probes: one pass, collected.
    pub fn embed(
        &mut self,
        seqs: &[Vec<u32>],
        dims: Option<usize>,
    ) -> Result<Vec<Vec<f32>>, String> {
        let p = self.embed_submit_dimensions(seqs, dims)?;
        self.embed_collect(&p)
    }

    /// [`Self::embed`] for sequences with media.
    pub fn embed_media(
        &mut self,
        seqs: &[Vec<u32>],
        media: &[Vec<crate::service::MmChunk>],
        dims: Option<usize>,
    ) -> Result<Vec<Vec<f32>>, String> {
        let p = self.embed_submit_media(seqs, media, dims)?;
        self.embed_collect(&p)
    }
}
