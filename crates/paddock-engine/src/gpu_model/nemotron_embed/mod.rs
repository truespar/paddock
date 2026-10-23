//! Nemotron 3.5 Embed 8B and Rerank 8B on CUDA - NVIDIA's multimodal
//! retrieval pair (`nvidia/Nemotron-3.5-Embed-8B-BF16`,
//! `nvidia/Nemotron-3.5-Rerank-8B-BF16`). Nothing is generated: a sequence
//! goes in, one vector (the embedder: 4096 wide, unit length, or its
//! Matryoshka prefix re-normalized) or one relevance logit (the reranker)
//! comes out.
//!
//! The graph, both checkpoints:
//!
//!   pictures -> Pixtral tower + Mistral 3 projector -> rows of `[IMG]`
//!   ids -> E[ids] (bf16, widened exactly), `[IMG]` rows from the pictures
//!   34 x { xn = rms(x) -> q|k|v GEMM -> YaRN rope + Llama 4's q factor
//!          -> bidirectional GQA attention (32 / 8, hd 128) -> x += o GEMM;
//!          xn = rms(x) -> gate|up GEMM with silu(gate) * up landed
//!          -> x += down GEMM }
//!   rms(x) -> mean over every row of the sequence (prompt included)
//!   -> unit vector (Embed) | mean . score (Rerank)
//!
//! The serving unit is a PASS of packed sequences, `cu[s]..cu[s + 1]`, no
//! padding (the EmbeddingGemma 2 and Laya encoders' layout); attention is the
//! one op that knows where a sequence ends. A pass holds at least one
//! full-context sequence: bidirectional attention needs the whole sequence
//! at once, so a sequence is never chunked.
//!
//! Numeric class: f16 operands and f32 accumulation on every GEMM (the bf16
//! weights widened to f16 exactly; activations rounded once to f16 into each
//! GEMM - three more mantissa bits than the vendor's bf16 serving), f32
//! residual stream, norms and softmax, f64 pooling. Reference: the release's
//! own Transformers code (`Mistral3Model` / its `model.py`) in float64 on the
//! same weights (the repo's nemotron-embed oracle script).
//!
//! Batch invariance: a vector or a logit does not depend on what shared its
//! pass. Every GEMM rides the mma rings (no GEMV band, no K-split - a row's
//! K order is the same at any row count), the row kernels keep one block
//! shape whatever the row count, attention blocks own (sequence, query tile,
//! kv head) with key tiles from the sequence start, pictures attend only
//! within themselves, and the pool reads a sequence's rows in order in one
//! block. Gated in `tests/gpu_nemotron_embed.rs`.

mod forward;
pub mod load;
pub mod vision;

use std::sync::Arc;

use cudarc::driver::CudaSlice;
use half::f16;
use paddock_models::nemotron_embed::{Kind, NemotronEmbedConfig};

use crate::gpu::{GpuExecutor, HalfTensor};
pub use crate::gpu_model::gpt_oss::GpuModelError;
pub use vision::{Picture, TOWER_PATCHES};

/// Seconds of complete idleness before the pass planes are dropped.
const IDLE_RECLAIM: std::time::Duration = std::time::Duration::from_secs(5);
/// Rows a pass takes at least (a full context always fits).
const PASS_ROWS_MIN: usize = 8192;

pub(crate) struct TextLayer {
    pub ln1: CudaSlice<f32>,
    /// q | k | v stacked, `[d -> (nh + 2 nkv) hd]`
    pub qkv: HalfTensor,
    pub o: HalfTensor,
    pub ln2: CudaSlice<f32>,
    /// gate | up in 16-row blocks, `[d -> 2 ffn]`
    pub gu: HalfTensor,
    pub down: HalfTensor,
}

/// Per-pass planes, sized for `rows_cap` packed rows, `seq_cap` sequences
/// and `pic_cap` picture rows; grown with admitted work, dropped after
/// [`IDLE_RECLAIM`].
struct Scratch {
    rows_cap: usize,
    seq_cap: usize,
    pic_cap: usize,
    src: CudaSlice<u32>,
    pos: CudaSlice<u32>,
    cu: CudaSlice<u32>,
    tiles: CudaSlice<u32>,
    x: CudaSlice<f32>,
    /// a projection's plain f32 landing, added into `x` by the next seam
    proj: CudaSlice<f32>,
    xn16: CudaSlice<f16>,
    qkv16: CudaSlice<f16>,
    attn16: CudaSlice<f16>,
    ff16: CudaSlice<f16>,
    pic: CudaSlice<f32>,
    mean: CudaSlice<f32>,
}

/// What a pass hands back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Output {
    /// unit vectors over the first `dims` components
    Embed { dims: usize },
    /// the reranker's logit
    Score,
    /// the mean over each sequence's rows, before any head (the gates')
    Mean,
}

/// A submitted pass: its outputs land in `out[flip]`, ready when `ev` fires.
pub struct PendingNemotron {
    ev: cudarc::driver::CudaEvent,
    flip: usize,
    count: usize,
    width: usize,
}

pub struct GpuNemotronEmbed {
    pub(crate) exec: Arc<GpuExecutor>,
    pub cfg: NemotronEmbedConfig,
    /// `[vocab][d]` bf16 bytes
    pub(crate) emb: CudaSlice<u8>,
    pub(crate) layers: Vec<TextLayer>,
    pub(crate) norm: CudaSlice<f32>,
    /// the reranker's score row, f32
    pub(crate) score: Option<CudaSlice<f32>>,
    pub(crate) tower: Option<vision::Tower>,
    /// YaRN (cos, sin) `[context][hd / 2]`
    pub(crate) rope_cos: CudaSlice<f32>,
    pub(crate) rope_sin: CudaSlice<f32>,
    pub(crate) context: usize,
    scratch: Option<Scratch>,
    /// Two output planes: the scheduler keeps at most two passes in flight,
    /// and a pass's output must outlive the next one's submit.
    out: [Option<CudaSlice<f32>>; 2],
    flip: usize,
    pub(crate) weights_bytes: u64,
}

impl GpuNemotronEmbed {
    pub fn kind(&self) -> Kind {
        self.cfg.kind
    }

    /// Resident weight bytes (the encoder's `weights_mem` line).
    pub fn weights_mem_bytes(&self) -> Option<u64> {
        Some(self.weights_bytes)
    }

    /// Device bytes this process holds live (weights + scratch).
    pub fn device_mem_used(&self) -> Option<u64> {
        self.exec.process_mem_used()
    }

    /// How long a fully idle encoder keeps its pass planes; None once dropped.
    pub fn idle_reclaim_after(&self) -> Option<std::time::Duration> {
        self.scratch.as_ref().map(|_| IDLE_RECLAIM)
    }

    /// Drop the pass planes (never the weights). The scheduler calls this
    /// only with nothing queued or uncollected; frees are stream-ordered.
    pub fn reclaim_idle(&mut self) {
        self.scratch = None;
        self.out = [None, None];
        if let Some(t) = &mut self.tower {
            t.release();
        }
    }

    /// Most packed rows one pass takes - never under one full context.
    pub fn coalesce_row_budget(&self) -> usize {
        self.context.max(PASS_ROWS_MIN)
    }

    /// The tokens a sequence may hold, pictures included.
    pub fn context(&self) -> usize {
        self.context
    }

    /// Whether pictures are served (the tower is loaded).
    pub fn serves_images(&self) -> bool {
        self.tower.is_some()
    }

    /// `[IMG]` rows a `w` x `h` picture takes.
    pub fn picture_rows(&self, w: usize, h: usize) -> Result<usize, String> {
        let (tw, th) = self.cfg.picture_target(w, h)?;
        let u = self.cfg.patch * self.cfg.merge;
        Ok((tw / u) * (th / u))
    }

    /// Admission for text-only sequences, before any device work.
    pub fn validate(&self, seqs: &[Vec<u32>]) -> Result<(), String> {
        self.validate_media(seqs, &[])
    }

    /// Admission for sequences that may carry pictures (`media[i]` is
    /// sequence i's, in order): every `[IMG]` is filled by a picture, in
    /// order, and nothing is left over. One malformed caller must not fail
    /// the others that would have shared its pass.
    pub fn validate_media(
        &self,
        seqs: &[Vec<u32>],
        media: &[Vec<crate::service::MmChunk>],
    ) -> Result<(), String> {
        use crate::service::MmChunk;
        let rows = seqs.iter().try_fold(0usize, |n, s| n.checked_add(s.len()));
        if seqs.is_empty() || rows.is_none_or(|n| n > self.coalesce_row_budget()) {
            return Err(format!(
                "a pass holds 1..{} sequences' rows",
                self.coalesce_row_budget()
            ));
        }
        let img = self.cfg.image_token_id;
        for (i, seq) in seqs.iter().enumerate() {
            if seq.is_empty() || seq.len() > self.context {
                return Err(format!(
                    "each input must contain 1..{} tokens, pictures included",
                    self.context
                ));
            }
            if seq.iter().any(|&t| t as usize >= self.cfg.vocab) {
                return Err("input contains a token id outside this model's vocabulary".into());
            }
            let want = seq.iter().filter(|&&t| t == img).count();
            let mut have = 0usize;
            for m in media.get(i).map_or(&[][..], Vec::as_slice) {
                match m {
                    MmChunk::Image { rgb, w, h } => {
                        if self.tower.is_none() {
                            return Err("this endpoint serves no pictures".into());
                        }
                        if rgb.len() != 3 * w * h {
                            return Err("a picture's bytes do not match its size".into());
                        }
                        if *w > vision::MAX_SIDE || *h > vision::MAX_SIDE {
                            return Err(format!(
                                "a {w} x {h} picture: sides must be within {}",
                                vision::MAX_SIDE
                            ));
                        }
                        have += self.picture_rows(*w, *h)?;
                    }
                    _ => return Err("only pictures are served as media".into()),
                }
            }
            if have != want {
                return Err(format!(
                    "{want} picture tokens for pictures that take {have} - each picture's \
                     [IMG] rows must match its size"
                ));
            }
        }
        Ok(())
    }

    /// Matryoshka: any prefix of the 4096 components, re-normalized (the
    /// model card's truncation), on the embedder only.
    pub fn validate_dimensions(&self, dims: Option<usize>) -> Result<(), String> {
        match (self.cfg.kind, dims) {
            (_, None) => Ok(()),
            (Kind::Embed, Some(d)) if (1..=self.cfg.d_model).contains(&d) => Ok(()),
            (Kind::Embed, Some(_)) => Err(format!(
                "Nemotron 3.5 Embed dimensions must be within 1..={}",
                self.cfg.d_model
            )),
            (Kind::Rerank, Some(_)) => Err("a reranker has no embedding dimensions".into()),
        }
    }

    /// Validate, then submit one pass.
    pub fn submit_media(
        &mut self,
        seqs: &[Vec<u32>],
        media: &[Vec<crate::service::MmChunk>],
        output: Output,
    ) -> Result<PendingNemotron, String> {
        self.validate_media(seqs, media)?;
        match (output, self.cfg.kind) {
            (Output::Embed { dims }, Kind::Embed) => self.validate_dimensions(Some(dims))?,
            (Output::Score, Kind::Rerank) | (Output::Mean, _) => {}
            (Output::Embed { .. }, Kind::Rerank) => {
                return Err("Nemotron 3.5 Rerank scores pairs; it is not an embedder".into());
            }
            (Output::Score, Kind::Embed) => {
                return Err("Nemotron 3.5 Embed is an embedder, not a reranker".into());
            }
        }
        self.submit(seqs, media, output).map_err(|e| e.to_string())
    }

    /// Non-blocking: has a submitted pass completed?
    pub fn pool_ready(&self, p: &PendingNemotron) -> bool {
        self.exec.event_done(&p.ev)
    }

    /// Read a pass's outputs, one row of its width a sequence (blocks only on
    /// its own event).
    pub fn collect(&mut self, p: &PendingNemotron) -> Result<Vec<Vec<f32>>, String> {
        let buf = self.out[p.flip]
            .as_ref()
            .ok_or("the output was reclaimed before collection")?;
        let flat = self
            .exec
            .to_host_len_after(&p.ev, buf, p.count * p.width)
            .map_err(|e| e.to_string())?;
        if flat.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite encoder output".into());
        }
        Ok(flat.chunks_exact(p.width).map(<[f32]>::to_vec).collect())
    }

    /// Blocking convenience for tests and probes: one pass, collected.
    pub fn run(
        &mut self,
        seqs: &[Vec<u32>],
        media: &[Vec<crate::service::MmChunk>],
        output: Output,
    ) -> Result<Vec<Vec<f32>>, String> {
        let p = self.submit_media(seqs, media, output)?;
        self.collect(&p)
    }

    /// One picture's patch rows as the processor and the stem make them
    /// (`[patches][592]` f16), the gates' probe of the pixel path.
    pub fn picture_patch_rows(&mut self, pic: &Picture<'_>) -> Result<Vec<f16>, GpuModelError> {
        let exec = self.exec.clone();
        self.tower
            .as_mut()
            .ok_or_else(|| GpuModelError::Unsupported("no picture tower loaded".into()))?
            .patch_rows(&exec, pic)
    }

    /// The tower's output rows for `pics` (`[rows][d_model]`), the gates'
    /// probe of the picture path alone.
    pub fn picture_features(&mut self, pics: &[Picture<'_>]) -> Result<Vec<f32>, GpuModelError> {
        let rows: usize = pics
            .iter()
            .map(|p| self.picture_rows(p.w, p.h))
            .collect::<Result<Vec<_>, _>>()
            .map_err(GpuModelError::Unsupported)?
            .iter()
            .sum();
        let exec = self.exec.clone();
        let mut out = exec.alloc(rows.max(1) * self.cfg.d_model)?;
        let tower = self
            .tower
            .as_mut()
            .ok_or_else(|| GpuModelError::Unsupported("no picture tower loaded".into()))?;
        tower.encode(&exec, pics, &mut out, 0)?;
        exec.synchronize()?;
        Ok(exec.to_host(&out)?[..rows * self.cfg.d_model].to_vec())
    }
}
