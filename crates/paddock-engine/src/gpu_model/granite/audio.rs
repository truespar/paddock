//! Granite Speech audio tower (`clip` GGUF, `projector_type ==
//! "granite_speech"`) - a conformer CTC encoder feeding a
//! BLIP-2 Q-Former projector. References: transformers 5.6
//! `modeling_granite_speech.py` (the upstream graph and the correctness
//! oracle) and llama.cpp b10330 `tools/mtmd/models/granite-speech.cpp`.
//!
//! Dataflow per clip:
//!
//!   stacked log-mel [n_frames, 160] (host, `crate::audio::granite`)
//!   -> input_linear 160->1024
//!   -> 16 conformer blocks, each
//!       x += 0.5 * ff1(LN x)                        macaron half-step
//!       x +=       attn(LN x)                       blockwise + Shaw RPE
//!       x +=       conv(LN x)                       GLU / dw-k15 / BN / SiLU
//!       x += 0.5 * ff2(LN x)                        macaron half-step
//!       x  = LN(x)                                  post-norm
//!     with a dual-head CTC branch spliced into the residual after layer 8:
//!       x += out_mid(softmax(out(x)))               1024->348->1024
//!   -> pad to whole 15-frame windows, 3 learned queries per window through
//!     2 post-LN Q-Former layers (self-attn -> cross-attn onto the window ->
//!     GELU FFN, eps 1e-12)
//!   -> linear 1024->2048
//!   -> [3 * ceil(n_frames/15), 2048] LLM-space audio embeddings, i.e. exactly
//!     **10 tokens per second** of speech.
//!
//! Chassis follows qwen3_asr/audio.rs: f32 activations over f16 weight
//! planes, resident scratch (never allocate inside `encode` - the one
//! exception is the output span, which outlives the call), cuBLAS for every
//! GEMM, and `crate::gpu::asr`'s fused epilogues so the launch train stays
//! short. New pack kernels live in `packs/cuda/src/asr/granite_speech.cuh`.
//!
//! Two shape facts drive the layout. The attention is BLOCKWISE over 200
//! frames and blocks never attend across, so nothing here needs a mask
//! tensor - the last block is simply short, which is what the reference's
//! pad-and-mask computes. And the conv module is CENTERED (k15, pad 7 both
//! ways), so it reads the whole clip and cannot be chunked without changing
//! the answer at every chunk boundary.
//!
//! The `-plus` variant adds one thing: `cat_hidden_layers` (GGUF
//! `clip.audio.feature_layer`, `[3]` as shipped) concatenates the named
//! blocks' outputs with the final one on the feature axis, so the Q-Former's
//! cross-attention reads a 2048-wide K/V input instead of 1024. Everything
//! else - geometry, block count, projector, LLM width - is identical, which
//! is why it rides this same tower rather than a second one.

use std::sync::Arc;

use cudarc::driver::CudaSlice;
use half::f16;
use paddock_models::gguf::Value;
use paddock_models::mapped::MappedGguf;

use crate::audio::MelFeatures;
use crate::audio::granite as mel;
use crate::gpu::{GpuExecutor, HalfTensor};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::qwen35::vision::host_f32;

/// Rows the planes grow in. A 30 s clip is 1500 encoder frames, so a stock
/// transcription lands inside the first step and never reallocates.
const ROWS_STEP: usize = 512;

/// The most encoder frames one pass takes (~4.4 min at 50 fps): a clip that
/// fits runs in one pass exactly as it always did, and a longer one runs in
/// SEGMENTS whose encoded range never exceeds this - so the planes stop
/// growing here, where they used to grow to the longest clip ever served
/// (~12 GB for a clip filling a 32k context) and keep it.
///
/// Segmenting is exact, not an approximation, by construction. Attention is
/// blockwise and never crosses a 200-frame block; the only thing that does
/// is the centered k15 conv, 7 frames a layer. A frame computed with the
/// wrong neighbours at a segment's edge therefore spreads by at most one
/// block per layer (7 frames across the edge, then the next layer's
/// attention over that whole block), so after 16 layers the error is 16
/// blocks deep. Each segment is encoded with SEGMENT_HALO frames of real
/// context on both sides (18 blocks) and keeps only its middle.
const SEGMENT_CAP: usize = SEGMENT_KEEP + 2 * SEGMENT_HALO;
/// Frames a segment keeps. Kept ranges start on multiples of 600 frames -
/// whole attention blocks AND whole 15-frame projector windows.
const SEGMENT_KEEP: usize = 10 * 600;
/// Context each side of a kept range: 18 blocks, covering the 16-block error
/// depth, and a multiple of both the block (200) and the window (15) so the
/// encoded range starts on a block and the kept rows on a window.
const SEGMENT_HALO: usize = 6 * 600;

/// One conformer block. Names follow the GGUF, which follows llama.cpp's
/// converter: `ffn_*` is the first macaron half and `ffn_*_1` the second,
/// `ln1` is the attention pre-norm and `ln2` the block's post-norm,
/// `norm_conv` is the conv module's LayerNorm while `conv_norm` is its
/// **folded BatchNorm** (the converter folds the running stats into an
/// affine at conversion time - there is no running mean/var at runtime).
struct ConfBlock {
    ff_norm_w: CudaSlice<f32>,
    ff_norm_b: CudaSlice<f32>,
    ff_up_w: HalfTensor,
    ff_up_b: CudaSlice<f32>,
    ff_down_w: HalfTensor,
    ff_down_b: CudaSlice<f32>,
    ln1_w: CudaSlice<f32>,
    ln1_b: CudaSlice<f32>,
    /// q|k|v concatenated on the output axis at load - one GEMM per layer
    /// instead of three, and the attention kernel reads the merged landing
    /// with a 3*d row stride.
    qkv_w: HalfTensor,
    /// Shaw relative position table `[2*max_pos+1, head_dim]`, shared by all
    /// heads of this layer.
    rel: CudaSlice<f32>,
    attn_out_w: HalfTensor,
    attn_out_b: CudaSlice<f32>,
    norm_conv_w: CudaSlice<f32>,
    norm_conv_b: CudaSlice<f32>,
    conv_pw1_w: HalfTensor,
    conv_pw1_b: CudaSlice<f32>,
    /// depthwise taps, TRANSPOSED at load to tap-major `[k][d]` so the
    /// kernel's per-tap read is contiguous across the warp
    conv_dw_w: CudaSlice<f32>,
    conv_bn_w: CudaSlice<f32>,
    conv_bn_b: CudaSlice<f32>,
    conv_pw2_w: HalfTensor,
    conv_pw2_b: CudaSlice<f32>,
    ff_norm1_w: CudaSlice<f32>,
    ff_norm1_b: CudaSlice<f32>,
    ff_up1_w: HalfTensor,
    ff_up1_b: CudaSlice<f32>,
    ff_down1_w: HalfTensor,
    ff_down1_b: CudaSlice<f32>,
    ln2_w: CudaSlice<f32>,
    ln2_b: CudaSlice<f32>,
}

/// One Q-Former layer: self-attention over the 3 queries, cross-attention
/// onto the 15-frame window, then a GELU FFN - each followed by its own
/// LayerNorm on the residual (post-LN, BLIP-2's contract).
struct QFormerLayer {
    sq_w: HalfTensor,
    sq_b: CudaSlice<f32>,
    sk_w: HalfTensor,
    sk_b: CudaSlice<f32>,
    sv_w: HalfTensor,
    sv_b: CudaSlice<f32>,
    so_w: HalfTensor,
    so_b: CudaSlice<f32>,
    sn_w: CudaSlice<f32>,
    sn_b: CudaSlice<f32>,
    cq_w: HalfTensor,
    cq_b: CudaSlice<f32>,
    /// Cross-attention K/V weights, one band per encoder SOURCE: the base
    /// model has a single 1024-wide source (one band = the whole plane),
    /// `-plus` has `[captured..., final]` and the file's K axis is their
    /// concatenation. Splitting at load lets the GEMMs accumulate band by
    /// band instead of staging a 2048-wide plane per encode.
    ck_w: Vec<HalfTensor>,
    ck_b: CudaSlice<f32>,
    cv_w: Vec<HalfTensor>,
    cv_b: CudaSlice<f32>,
    co_w: HalfTensor,
    co_b: CudaSlice<f32>,
    cn_w: CudaSlice<f32>,
    cn_b: CudaSlice<f32>,
    fu_w: HalfTensor,
    fu_b: CudaSlice<f32>,
    fd_w: HalfTensor,
    fd_b: CudaSlice<f32>,
    fn_w: CudaSlice<f32>,
    fn_b: CudaSlice<f32>,
}

/// The encoded clip: audio-token embeddings ready for LLM injection.
pub struct AudioOutput {
    /// [n_tokens, llm_embd] device-resident audio embeddings.
    pub embd: CudaSlice<f32>,
    pub n_tokens: usize,
}

/// Every device plane `encode` touches, allocated once and grown in
/// [`ROWS_STEP`] steps. Sizing is driven by the encoder frame count; the
/// window-padded count and the query count both derive from it.
struct Scratch {
    cap: usize,
    /// mel upload landing [cap, 160]
    md: CudaSlice<f32>,
    /// residual stream [pad_cap, embd] - sized for the PADDED row count
    /// because the projector reads whole windows
    x: CudaSlice<f32>,
    /// branch output [cap, embd]
    n: CudaSlice<f32>,
    /// f16 staging for every GEMM input; widest use is [cap, 4*embd]
    s16: CudaSlice<f16>,
    /// FFN / conv-up landing [cap, 4*embd]
    up: CudaSlice<f32>,
    /// GLU output [cap, 2*embd]
    glu: CudaSlice<f32>,
    /// merged q|k|v landing [cap, 3*embd]
    qkv: CudaSlice<f32>,
    /// attention output, f16 (feeds the output projection directly)
    a16: CudaSlice<f16>,
    /// CTC logits [cap, ctc_dim] and their softmax at f16
    ctc: CudaSlice<f32>,
    ctc16: CudaSlice<f16>,
    /// encoder output at f16, read by both Q-Former layers' cross K/V
    enc16: CudaSlice<f16>,
    /// `-plus` only: one [pad_cap, embd] f16 plane per `cat_hidden_layers`
    /// entry, holding that block's captured output. Empty on the base model.
    cat16: Vec<CudaSlice<f16>>,
    /// zero rows, copied over the window padding (planes are reused across
    /// encodes, so the pad has to be written, not merely never touched)
    zero: CudaSlice<f32>,
    /// the same for the f16 capture planes, once a segment's kept rows move
    zero16: CudaSlice<f16>,
    /// one segment's projector output, copied into the clip's span
    seg_out: CudaSlice<f32>,
    /// the LayerNorm'd query template repeated per window - constant for a
    /// given block count, so it is built on grow and only copied per encode
    qtpl: CudaSlice<f32>,
    /// Q-Former residual stream [q_cap, embd]
    qz: CudaSlice<f32>,
    pq: CudaSlice<f32>,
    /// cross K/V are window-sized, not query-sized
    pk: CudaSlice<f32>,
    pv: CudaSlice<f32>,
    pat: CudaSlice<f32>,
    pn: CudaSlice<f32>,
}

impl Scratch {
    fn new(
        exec: &GpuExecutor,
        cap: usize,
        embd: usize,
        ffn: usize,
        conv_inner: usize,
        ctc_dim: usize,
        in_dim: usize,
        win: usize,
        queries: usize,
        n_cat: usize,
        out_dim: usize,
        qnorm: &[f32],
    ) -> Result<Self, GpuModelError> {
        let pad_cap = cap.div_ceil(win) * win;
        let q_cap = pad_cap / win * queries;
        // one template row set per window, repeated - the query parameter is
        // shared by every window (upstream repeats the same [1, 3, d])
        let mut tpl = Vec::with_capacity(q_cap * embd);
        for _ in 0..pad_cap / win {
            tpl.extend_from_slice(qnorm);
        }
        Ok(Self {
            cap,
            md: exec.alloc(cap * in_dim)?,
            x: exec.alloc(pad_cap * embd)?,
            n: exec.alloc(cap * embd)?,
            s16: exec.alloc_f16(cap * ffn)?,
            up: exec.alloc(cap * ffn)?,
            glu: exec.alloc(cap * conv_inner)?,
            qkv: exec.alloc(cap * 3 * embd)?,
            a16: exec.alloc_f16(cap * embd)?,
            ctc: exec.alloc(cap * ctc_dim)?,
            ctc16: exec.alloc_f16(cap * ctc_dim)?,
            enc16: exec.alloc_f16(pad_cap * embd)?,
            cat16: (0..n_cat)
                .map(|_| exec.alloc_f16(pad_cap * embd))
                .collect::<Result<_, _>>()?,
            zero: exec.alloc(win * embd)?,
            zero16: exec.alloc_f16(win * embd)?,
            seg_out: exec.alloc(q_cap * out_dim)?,
            qtpl: exec.to_device(&tpl)?,
            qz: exec.alloc(q_cap * embd)?,
            pq: exec.alloc(q_cap * embd)?,
            pk: exec.alloc(pad_cap * embd)?,
            pv: exec.alloc(pad_cap * embd)?,
            pat: exec.alloc(q_cap * embd)?,
            pn: exec.alloc(q_cap * embd)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn bytes(
        cap: usize,
        n_cat: usize,
        embd: usize,
        ffn: usize,
        conv_inner: usize,
        ctc_dim: usize,
        in_dim: usize,
        win: usize,
        queries: usize,
        out_dim: usize,
    ) -> usize {
        let pad_cap = cap.div_ceil(win) * win;
        let q_cap = pad_cap / win * queries;
        let f32s = cap * in_dim
            + pad_cap * embd
            + cap * embd
            + cap * ffn
            + cap * conv_inner
            + cap * 3 * embd
            + cap * ctc_dim
            + win * embd
            + q_cap * embd * 5
            + q_cap * out_dim
            + pad_cap * embd * 2;
        let f16s =
            cap * ffn + cap * embd + cap * ctc_dim + win * embd + pad_cap * embd * (1 + n_cat);
        f32s * 4 + f16s * 2
    }
}

pub struct SpeechTower {
    exec: Arc<GpuExecutor>,
    embd: usize,
    n_heads: usize,
    head_dim: usize,
    ffn: usize,
    /// conv module inner width = embd * conv_expansion_factor
    conv_inner: usize,
    conv_k: usize,
    /// attention block length in encoder frames (`context_size`)
    ctx: usize,
    max_pos: usize,
    eps: f32,
    ctc_dim: usize,
    win: usize,
    queries: usize,
    proj_heads: usize,
    /// `-plus` feature concat: block indices (1-based, matching upstream's
    /// `cat_hidden_layers`) whose output is captured and concatenated ahead of
    /// the final encoder output. Empty on the base model. Index 0 means the
    /// input projection's output, which upstream also allows.
    cat_layers: Vec<usize>,
    /// LLM embedding width the projector emits.
    pub out_dim: usize,
    inp_w: HalfTensor,
    inp_b: CudaSlice<f32>,
    blocks: Vec<ConfBlock>,
    ctc_out_w: HalfTensor,
    ctc_out_b: CudaSlice<f32>,
    ctc_mid_w: HalfTensor,
    ctc_mid_b: CudaSlice<f32>,
    qf: Vec<QFormerLayer>,
    proj_w: HalfTensor,
    proj_b: CudaSlice<f32>,
    /// LayerNorm'd query parameter, host side - repeated per window whenever
    /// the scratch grows.
    qnorm: Vec<f32>,
    sc: Scratch,
    /// Resident weight bytes (tallied at load) - the VRAM ledger line.
    weight_bytes: usize,
}

/// Q-Former LayerNorm epsilon. Not in the GGUF (llama.cpp hard-codes it too):
/// it comes from the BLIP-2 Q-Former config, which is a different module from
/// the conformer and uses a different eps - 1e-12 against the encoder's 1e-5.
const PROJ_EPS: f32 = 1e-12;

impl SpeechTower {
    pub fn load(exec: Arc<GpuExecutor>, map: &MappedGguf) -> Result<Self, GpuModelError> {
        let meta = &map.gguf().metadata;
        let proj = meta
            .get("clip.projector_type")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        if proj != "granite_speech" {
            return Err(GpuModelError::MissingMeta(format!(
                "mmproj is not a granite-speech audio tower (projector_type '{proj}', \
                 want 'granite_speech')"
            )));
        }
        if !exec.has_granite_speech() {
            return Err(GpuModelError::Unsupported(
                "the loaded CUDA pack has no granite-speech conformer kernels - rebuild it \
                 (packs/cuda/build.sh); a .cuh newer than the .so means yesterday's pack"
                    .into(),
            ));
        }
        let u = |k: &str| {
            meta.get(k)
                .and_then(Value::as_u64)
                .ok_or_else(|| GpuModelError::MissingMeta(k.to_owned()))
        };
        let n_layers = u("clip.audio.block_count")? as usize;
        let embd = u("clip.audio.embedding_length")? as usize;
        let n_heads = u("clip.audio.attention.head_count")? as usize;
        let ffn = u("clip.audio.feed_forward_length")? as usize;
        let out_dim = u("clip.audio.projection_dim")? as usize;
        let in_dim = u("clip.audio.num_mel_bins")? as usize;
        let ctx = u("clip.audio.chunk_size")? as usize;
        let conv_k = u("clip.audio.conv_kernel_size")? as usize;
        let max_pos = u("clip.audio.max_pos_emb")? as usize;
        let win = u("clip.audio.projector.window_size")? as usize;
        let down = u("clip.audio.projector.downsample_rate")? as usize;
        let proj_heads = u("clip.audio.projector.head_count")? as usize;
        let eps = meta
            .get("clip.audio.attention.layer_norm_epsilon")
            .and_then(Value::as_f32)
            .unwrap_or(1e-5);
        if in_dim != mel::INPUT_DIM || win != mel::WINDOW_SIZE || down != mel::DOWNSAMPLE_RATE {
            return Err(GpuModelError::Unsupported(format!(
                "granite-speech mmproj geometry ({in_dim} mel bins, window {win}, downsample \
                 {down}) disagrees with the host frontend ({}, {}, {})",
                mel::INPUT_DIM,
                mel::WINDOW_SIZE,
                mel::DOWNSAMPLE_RATE
            )));
        }
        // `context_size <= max_pos_emb` is an upstream config invariant, and
        // the attention kernel relies on it: it stages a contiguous run of
        // relative-position rows and would walk off the table otherwise.
        if ctx > max_pos {
            return Err(GpuModelError::Unsupported(format!(
                "granite-speech context_size {ctx} exceeds max_pos_emb {max_pos}"
            )));
        }
        let queries = win / down;

        // `-plus` feature concat. The key is a plain array of block indices;
        // upstream numbers them from 1 (`enumerate(self.layers, start=1)`),
        // with 0 meaning the input projection's output, and llama.cpp's
        // `is_feature_layer(il + 1)` agrees. An index past the last block
        // would capture nothing, so it is a file error rather than a no-op.
        let mut cat_layers: Vec<usize> = match meta.get("clip.audio.feature_layer") {
            None => Vec::new(),
            Some(Value::Array(vs)) => vs
                .iter()
                .map(|v| {
                    v.as_u64().map(|x| x as usize).ok_or_else(|| {
                        GpuModelError::Unsupported(
                            "clip.audio.feature_layer holds a non-integer entry".into(),
                        )
                    })
                })
                .collect::<Result<_, _>>()?,
            Some(v) => {
                return Err(GpuModelError::Unsupported(format!(
                    "clip.audio.feature_layer is {v:?}, expected an array of block indices"
                )));
            }
        };
        cat_layers.sort_unstable();
        if let Some(&worst) = cat_layers.last()
            && worst > n_layers
        {
            return Err(GpuModelError::Unsupported(format!(
                "clip.audio.feature_layer names block {worst} but the encoder has {n_layers}"
            )));
        }

        // Resident weight bytes, tallied at the upload seams rather than by
        // walking the struct's fields afterwards - a field walk silently
        // under-reports the day somebody adds a plane, and this number feeds
        // the VRAM ledger the fit chart quotes.
        let tally = std::rc::Rc::new(std::cell::Cell::new(0usize));

        let e = exec.clone();
        let t1 = tally.clone();
        let dt = |name: &str| -> Result<HalfTensor, GpuModelError> {
            let t = e.upload_f16(map, name)?;
            t1.set(t1.get() + t.buf.len() * 2);
            Ok(t)
        };
        let e2 = exec.clone();
        let t2 = tally.clone();
        let vec1 = |name: &str| -> Result<CudaSlice<f32>, GpuModelError> {
            let (host, _) = host_f32(map, name)?;
            t2.set(t2.get() + host.len() * 4);
            Ok(e2.to_device(&host)?)
        };

        // q|k|v share a K axis and differ only in output rows, so the file's
        // three [K, N] planes concatenate on N by plain appending - the GGUF
        // layout is output-major.
        let e3 = exec.clone();
        let t3 = tally.clone();
        let merge_qkv = move |i: usize| -> Result<HalfTensor, GpuModelError> {
            let mut all = Vec::new();
            let mut k = 0usize;
            for part in ["attn_q", "attn_k", "attn_v"] {
                let (w, dims) = host_f32(map, &format!("a.blk.{i}.{part}.weight"))?;
                k = dims[0];
                all.extend_from_slice(&w);
            }
            let n = all.len() / k;
            t3.set(t3.get() + all.len() * 2);
            Ok(HalfTensor {
                buf: e3.to_device_f16(&all, "a.blk.qkv")?,
                dims: vec![k, n],
            })
        };
        // depthwise taps ship channel-major [k][c]-per-channel; the kernel
        // wants tap-major so each tap read is one coalesced line.
        let e4 = exec.clone();
        let t4 = tally.clone();
        let dw_taps = move |i: usize| -> Result<CudaSlice<f32>, GpuModelError> {
            let (w, dims) = host_f32(map, &format!("a.blk.{i}.conv_dw.weight"))?;
            let (k, d) = (dims[0], dims[1]);
            let mut out = vec![0f32; k * d];
            for c in 0..d {
                for j in 0..k {
                    out[j * d + c] = w[c * k + j];
                }
            }
            t4.set(t4.get() + out.len() * 4);
            Ok(e4.to_device(&out)?)
        };

        // Cross-attention K/V weights, split into one band per encoder source.
        // The GGUF layout is output-major (a run of K per output row), so a
        // band is a per-row slice rather than a contiguous prefix - hence the
        // gather. On the base model there is exactly one source and the plane
        // goes up untouched, which keeps that (already gated) path bit-identical.
        let n_src = cat_layers.len() + 1;
        let e5 = exec.clone();
        let t5 = tally.clone();
        let cross_bands = move |name: &str| -> Result<Vec<HalfTensor>, GpuModelError> {
            if n_src == 1 {
                let t = e5.upload_f16(map, name)?;
                t5.set(t5.get() + t.buf.len() * 2);
                return Ok(vec![t]);
            }
            let (w, dims) = host_f32(map, name)?;
            let (k, n) = (dims[0], w.len() / dims[0]);
            if k != embd * n_src {
                return Err(GpuModelError::Unsupported(format!(
                    "{name} reads {k} inputs but the encoder offers {n_src} x {embd} - the \
                     mmproj's clip.audio.feature_layer disagrees with its own projector weights"
                )));
            }
            let mut out = Vec::with_capacity(n_src);
            for b in 0..n_src {
                let mut band = Vec::with_capacity(embd * n);
                for row in 0..n {
                    band.extend_from_slice(&w[row * k + b * embd..row * k + (b + 1) * embd]);
                }
                t5.set(t5.get() + band.len() * 2);
                out.push(HalfTensor {
                    buf: e5.to_device_f16(&band, name)?,
                    dims: vec![embd, n],
                });
            }
            Ok(out)
        };

        let mut blocks = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let t = |s: &str| format!("a.blk.{i}.{s}");
            blocks.push(ConfBlock {
                ff_norm_w: vec1(&t("ffn_norm.weight"))?,
                ff_norm_b: vec1(&t("ffn_norm.bias"))?,
                ff_up_w: dt(&t("ffn_up.weight"))?,
                ff_up_b: vec1(&t("ffn_up.bias"))?,
                ff_down_w: dt(&t("ffn_down.weight"))?,
                ff_down_b: vec1(&t("ffn_down.bias"))?,
                ln1_w: vec1(&t("ln1.weight"))?,
                ln1_b: vec1(&t("ln1.bias"))?,
                qkv_w: merge_qkv(i)?,
                rel: vec1(&t("attn_rel_pos_emb"))?,
                attn_out_w: dt(&t("attn_out.weight"))?,
                attn_out_b: vec1(&t("attn_out.bias"))?,
                norm_conv_w: vec1(&t("norm_conv.weight"))?,
                norm_conv_b: vec1(&t("norm_conv.bias"))?,
                conv_pw1_w: dt(&t("conv_pw1.weight"))?,
                conv_pw1_b: vec1(&t("conv_pw1.bias"))?,
                conv_dw_w: dw_taps(i)?,
                conv_bn_w: vec1(&t("conv_norm.weight"))?,
                conv_bn_b: vec1(&t("conv_norm.bias"))?,
                conv_pw2_w: dt(&t("conv_pw2.weight"))?,
                conv_pw2_b: vec1(&t("conv_pw2.bias"))?,
                ff_norm1_w: vec1(&t("ffn_norm_1.weight"))?,
                ff_norm1_b: vec1(&t("ffn_norm_1.bias"))?,
                ff_up1_w: dt(&t("ffn_up_1.weight"))?,
                ff_up1_b: vec1(&t("ffn_up_1.bias"))?,
                ff_down1_w: dt(&t("ffn_down_1.weight"))?,
                ff_down1_b: vec1(&t("ffn_down_1.bias"))?,
                ln2_w: vec1(&t("ln2.weight"))?,
                ln2_b: vec1(&t("ln2.bias"))?,
            });
        }

        let mut qf = Vec::new();
        for i in 0.. {
            let t = |s: &str| format!("a.proj_blk.{i}.{s}");
            if map.tensor_info(&t("self_attn_q.weight")).is_none() {
                break;
            }
            qf.push(QFormerLayer {
                sq_w: dt(&t("self_attn_q.weight"))?,
                sq_b: vec1(&t("self_attn_q.bias"))?,
                sk_w: dt(&t("self_attn_k.weight"))?,
                sk_b: vec1(&t("self_attn_k.bias"))?,
                sv_w: dt(&t("self_attn_v.weight"))?,
                sv_b: vec1(&t("self_attn_v.bias"))?,
                so_w: dt(&t("self_attn_out.weight"))?,
                so_b: vec1(&t("self_attn_out.bias"))?,
                sn_w: vec1(&t("self_attn_norm.weight"))?,
                sn_b: vec1(&t("self_attn_norm.bias"))?,
                cq_w: dt(&t("cross_attn_q.weight"))?,
                cq_b: vec1(&t("cross_attn_q.bias"))?,
                ck_w: cross_bands(&t("cross_attn_k.weight"))?,
                ck_b: vec1(&t("cross_attn_k.bias"))?,
                cv_w: cross_bands(&t("cross_attn_v.weight"))?,
                cv_b: vec1(&t("cross_attn_v.bias"))?,
                co_w: dt(&t("cross_attn_out.weight"))?,
                co_b: vec1(&t("cross_attn_out.bias"))?,
                cn_w: vec1(&t("cross_attn_norm.weight"))?,
                cn_b: vec1(&t("cross_attn_norm.bias"))?,
                fu_w: dt(&t("ffn_up.weight"))?,
                fu_b: vec1(&t("ffn_up.bias"))?,
                fd_w: dt(&t("ffn_down.weight"))?,
                fd_b: vec1(&t("ffn_down.bias"))?,
                fn_w: vec1(&t("ffn_norm.weight"))?,
                fn_b: vec1(&t("ffn_norm.bias"))?,
            });
        }
        if qf.is_empty() {
            return Err(GpuModelError::MissingMeta("a.proj_blk.0.*".into()));
        }

        // The query parameter's LayerNorm is the Q-Former's input embedding
        // step and depends on nothing else, so it runs once here rather than
        // per encode.
        let (qraw, qdims) = host_f32(map, "a.proj_query")?;
        if qdims[0] != embd || qraw.len() != embd * queries {
            return Err(GpuModelError::Unsupported(format!(
                "a.proj_query is {qdims:?}, expected [{embd}, {queries}]"
            )));
        }
        let (qn_w, _) = host_f32(map, "a.proj_norm.weight")?;
        let (qn_b, _) = host_f32(map, "a.proj_norm.bias")?;
        let mut qnorm = vec![0f32; qraw.len()];
        for q in 0..queries {
            let row = &qraw[q * embd..(q + 1) * embd];
            let mean = row.iter().map(|&v| v as f64).sum::<f64>() / embd as f64;
            let var = row
                .iter()
                .map(|&v| (v as f64 - mean) * (v as f64 - mean))
                .sum::<f64>()
                / embd as f64;
            let inv = 1.0 / (var + PROJ_EPS as f64).sqrt();
            for d in 0..embd {
                qnorm[q * embd + d] = (((row[d] as f64 - mean) * inv) as f32) * qn_w[d] + qn_b[d];
            }
        }

        let ctc_out_w = dt("a.enc_ctc_out.weight")?;
        let ctc_dim = ctc_out_w.dims[1];
        let conv_pw1_out = blocks[0].conv_pw1_w.dims[1];
        let conv_inner = conv_pw1_out / 2;

        // Segments are exact only while the halo covers the error depth: one
        // attention block per conformer layer (see SEGMENT_CAP). A tower
        // deeper, or with wider blocks, than this one would need a wider
        // halo - refuse it rather than transcribe long clips inexactly.
        if blocks.len() * ctx > SEGMENT_HALO || !SEGMENT_HALO.is_multiple_of(ctx) {
            return Err(GpuModelError::Unsupported(format!(
                "granite-speech: {} layers of {ctx}-frame attention blocks outrun the \
                 {SEGMENT_HALO}-frame segment halo that keeps long clips exact",
                blocks.len()
            )));
        }
        let sc = Scratch::new(
            &exec,
            ROWS_STEP,
            embd,
            ffn,
            conv_inner,
            ctc_dim,
            in_dim,
            win,
            queries,
            cat_layers.len(),
            out_dim,
            &qnorm,
        )?;
        let me = Self {
            exec,
            embd,
            n_heads,
            head_dim: embd / n_heads,
            ffn,
            conv_inner,
            conv_k,
            ctx,
            max_pos,
            eps,
            ctc_dim,
            win,
            queries,
            proj_heads,
            cat_layers,
            out_dim,
            inp_w: dt("a.input_projection.weight")?,
            inp_b: vec1("a.input_projection.bias")?,
            blocks,
            ctc_out_w,
            ctc_out_b: vec1("a.enc_ctc_out.bias")?,
            ctc_mid_w: dt("a.enc_ctc_out_mid.weight")?,
            ctc_mid_b: vec1("a.enc_ctc_out_mid.bias")?,
            qf,
            proj_w: dt("a.proj_linear.weight")?,
            proj_b: vec1("a.proj_linear.bias")?,
            qnorm,
            sc,
            weight_bytes: tally.get(),
        };
        tracing::info!(
            layers = n_layers,
            qformer = me.qf.len(),
            ctx,
            ctc_dim,
            cat_layers = ?me.cat_layers,
            weights_mib = me.weight_bytes / (1 << 20),
            scratch_mib = me.scratch_bytes(me.sc.cap) / (1 << 20),
            "granite-speech conformer tower resident at f16 (f32 accumulate)"
        );
        Ok(me)
    }

    /// Resident bytes: weights plus the scratch planes allocated so far. The
    /// scratch grows with the longest clip served, up to [`SEGMENT_CAP`]
    /// frames, so this rises once and settles - it is read for the ledger
    /// after load, when it is just the initial [`ROWS_STEP`] step.
    pub fn resident_bytes(&self) -> usize {
        self.weight_bytes + self.scratch_bytes(self.sc.cap)
    }

    /// What the scratch may still grow by: the plan reserves it, since the
    /// growth is on demand and bounded at [`SEGMENT_CAP`] frames.
    pub fn scratch_growth_bytes(&self) -> usize {
        self.scratch_bytes(SEGMENT_CAP)
            .saturating_sub(self.scratch_bytes(self.sc.cap))
    }

    fn scratch_bytes(&self, cap: usize) -> usize {
        Scratch::bytes(
            cap,
            self.cat_layers.len(),
            self.embd,
            self.ffn,
            self.conv_inner,
            self.ctc_dim,
            mel::INPUT_DIM,
            self.win,
            self.queries,
            self.out_dim,
        )
    }

    /// Audio tokens this tower emits for a clip of `frames` encoder frames.
    pub fn tokens_for_frames(&self, frames: usize) -> usize {
        frames.div_ceil(self.win) * self.queries
    }

    /// Audio tokens a clip of `len` 16 kHz samples will occupy in the prompt.
    /// Pure arithmetic over the host frontend's frame rule - call it while
    /// BUILDING the prompt, before a sample moves (mirrors `image_tokens`).
    pub fn tokens_for_samples(&self, len: usize) -> usize {
        self.tokens_for_frames(mel::encoder_frames(len))
    }

    fn grow(&mut self, rows: usize) -> Result<(), GpuModelError> {
        debug_assert!(rows <= SEGMENT_CAP, "a pass never exceeds SEGMENT_CAP");
        if rows <= self.sc.cap {
            return Ok(());
        }
        let cap = (rows.div_ceil(ROWS_STEP) * ROWS_STEP).min(SEGMENT_CAP.max(rows));
        // release the old planes first: building the new set while the old
        // was still held made every growth cost both at once
        self.sc = Scratch::new(
            &self.exec,
            1,
            self.embd,
            self.ffn,
            self.conv_inner,
            self.ctc_dim,
            mel::INPUT_DIM,
            self.win,
            self.queries,
            self.cat_layers.len(),
            self.out_dim,
            &self.qnorm[..self.queries * self.embd],
        )?;
        self.exec.trim_mem_pool();
        self.sc = Scratch::new(
            &self.exec,
            cap,
            self.embd,
            self.ffn,
            self.conv_inner,
            self.ctc_dim,
            mel::INPUT_DIM,
            self.win,
            self.queries,
            self.cat_layers.len(),
            self.out_dim,
            &self.qnorm,
        )?;
        Ok(())
    }

    /// Encode one clip's stacked log-mel features into LLM-space audio
    /// embeddings.
    pub fn encode(&mut self, feats: &MelFeatures) -> Result<AudioOutput, GpuModelError> {
        let t = feats.n_frames;
        if t == 0 {
            return Err(GpuModelError::Unsupported("empty audio features".into()));
        }
        // `MelFeatures` is the container every family's frontend fills, so the
        // one thing that distinguishes granite's plane from whisper's is its
        // width. Check it: a 128-wide whisper plane would otherwise upload
        // fine, read as 0.8 of a clip, and transcribe noise.
        if feats.data.len() != t * mel::INPUT_DIM {
            return Err(GpuModelError::Unsupported(format!(
                "granite-speech wants {t}x{} stacked log-mel features, got {} floats - \
                 this plane came from a different frontend",
                mel::INPUT_DIM,
                feats.data.len()
            )));
        }
        let exec = self.exec.clone();
        let n_tokens = self.tokens_for_frames(t);
        // the one allocation per encode: the span outlives the call
        let mut embd_out = exec.alloc(n_tokens * self.out_dim)?;
        let od = self.out_dim;
        if t <= SEGMENT_CAP {
            self.grow(t)?;
            let nq = self.encode_range(feats, 0, t, 0, t)?;
            exec.copy_region(&self.sc.seg_out, 0, &mut embd_out, 0, nq * od)?;
        } else {
            // overlapped segments (see SEGMENT_CAP): encode [x0, x1), keep
            // [k0, k1), land its tokens at the clip's token offset for k0
            self.grow(SEGMENT_CAP)?;
            let mut k0 = 0usize;
            while k0 < t {
                let k1 = (k0 + SEGMENT_KEEP).min(t);
                let (x0, x1) = (k0.saturating_sub(SEGMENT_HALO), (k1 + SEGMENT_HALO).min(t));
                let nq = self.encode_range(feats, x0, x1, k0, k1)?;
                let at = k0 / self.win * self.queries;
                exec.copy_region(&self.sc.seg_out, 0, &mut embd_out, at * od, nq * od)?;
                k0 = k1;
            }
        }
        let nq = n_tokens;

        // Debug tap for the encoder oracle gate: appends one
        // [n_tokens, out_dim] f32 block per encode.
        if let Ok(path) = paddock_models::dev_var!("PADDOCK_GS_DUMP_EMBD") {
            let host = exec.to_host_len(&embd_out, nq * self.out_dim)?;
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .map_err(|e| crate::gpu::GpuError::Driver(format!("embd dump {path}: {e}")))?;
            let bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
            f.write_all(&bytes)
                .map_err(|e| crate::gpu::GpuError::Driver(format!("embd dump {path}: {e}")))?;
            tracing::info!(
                n_tokens = nq,
                path,
                "dumped granite-speech tower embeddings"
            );
        }

        Ok(AudioOutput {
            embd: embd_out,
            n_tokens: nq,
        })
    }

    /// Input projection, the 16 conformer blocks and the Q-Former projector
    /// over encoder frames `[x0, x1)`, keeping `[k0, k1)`: the kept frames'
    /// tokens land in `sc.seg_out`, and their count is returned. A single
    /// pass is `x0 == k0 == 0, x1 == k1 == t` and runs exactly as the tower
    /// always did.
    fn encode_range(
        &mut self,
        feats: &MelFeatures,
        x0: usize,
        x1: usize,
        k0: usize,
        k1: usize,
    ) -> Result<usize, GpuModelError> {
        let exec = self.exec.clone();
        let sc = &mut self.sc;
        let (e, ffn, ci) = (self.embd, self.ffn, self.conv_inner);
        let eps = self.eps;
        let t = x1 - x0;
        let padded = t.div_ceil(self.win) * self.win;

        // ---- input projection: [t, 160] -> [t, embd] ----
        exec.upload_f32(
            &feats.data[x0 * mel::INPUT_DIM..x1 * mel::INPUT_DIM],
            &mut sc.md,
        )?;
        exec.convert_f32_f16(&sc.md, &mut sc.s16, t * mel::INPUT_DIM)?;
        exec.matvec_batch_f16(&self.inp_w, &sc.s16, &mut sc.x, t)?;
        exec.bias_add(&mut sc.x, &self.inp_b, t, e)?;

        // The window padding is real encoder input to the cross-attention
        // (upstream pads with zeros after the encoder), and these planes are
        // reused across encodes, so the pad rows have to be written, not
        // merely left alone. Zeroing them here rather than after the block
        // loop is what lets a `-plus` capture convert whole padded rows: the
        // loop only ever writes rows [0, t), so a pad row zeroed now stays
        // zeroed, and the arithmetic on the real rows is untouched either way.
        if padded > t {
            exec.copy_region(&sc.zero, 0, &mut sc.x, t * e, (padded - t) * e)?;
        }
        if let Some(slot) = self.cat_layers.iter().position(|&l| l == 0) {
            exec.convert_f32_f16(&sc.x, &mut sc.cat16[slot], padded * e)?;
        }

        // ---- 16 conformer blocks ----
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let ctc_at = self.blocks.len() / 2;
        for (li, blk) in self.blocks.iter().enumerate() {
            // ff1, folded in at half weight (the macaron half-step)
            exec.whisper_ln_f16(
                &sc.x,
                &blk.ff_norm_w,
                &blk.ff_norm_b,
                &mut sc.s16,
                t,
                e,
                eps,
            )?;
            exec.matvec_batch_f16(&blk.ff_up_w, &sc.s16, &mut sc.up, t)?;
            exec.gs_bias_silu_f16(&sc.up, &blk.ff_up_b, &mut sc.s16, t, ffn)?;
            exec.matvec_batch_f16(&blk.ff_down_w, &sc.s16, &mut sc.n, t)?;
            exec.gs_res_ln_f16(
                &mut sc.x,
                &sc.n,
                &blk.ff_down_b,
                Some((&blk.ln1_w, &blk.ln1_b, &mut sc.s16)),
                t,
                e,
                0.5,
                eps,
            )?;

            // blockwise attention with Shaw relative positions
            exec.matvec_batch_f16(&blk.qkv_w, &sc.s16, &mut sc.qkv, t)?;
            exec.gs_conf_attn(
                &sc.qkv,
                &mut sc.a16,
                &blk.rel,
                t,
                self.ctx,
                self.n_heads,
                self.head_dim,
                self.max_pos,
                scale,
            )?;
            exec.matvec_batch_f16(&blk.attn_out_w, &sc.a16, &mut sc.n, t)?;
            exec.gs_res_ln_f16(
                &mut sc.x,
                &sc.n,
                &blk.attn_out_b,
                Some((&blk.norm_conv_w, &blk.norm_conv_b, &mut sc.s16)),
                t,
                e,
                1.0,
                eps,
            )?;

            // conv module: pointwise-up -> GLU -> centered depthwise ->
            // folded BN -> SiLU -> pointwise-down
            exec.matvec_batch_f16(&blk.conv_pw1_w, &sc.s16, &mut sc.up, t)?;
            exec.gs_bias_glu(&sc.up, &blk.conv_pw1_b, &mut sc.glu, t, ci)?;
            exec.gs_dwconv_bn_silu_f16(
                &sc.glu,
                &blk.conv_dw_w,
                &blk.conv_bn_w,
                &blk.conv_bn_b,
                &mut sc.s16,
                t,
                ci,
                self.conv_k,
            )?;
            exec.matvec_batch_f16(&blk.conv_pw2_w, &sc.s16, &mut sc.n, t)?;
            exec.gs_res_ln_f16(
                &mut sc.x,
                &sc.n,
                &blk.conv_pw2_b,
                Some((&blk.ff_norm1_w, &blk.ff_norm1_b, &mut sc.s16)),
                t,
                e,
                1.0,
                eps,
            )?;

            // ff2 + the block's post-norm. The f16 landing is only wanted
            // where the CTC branch consumes it - every other layer's next
            // read is another LayerNorm, not a GEMM.
            exec.matvec_batch_f16(&blk.ff_up1_w, &sc.s16, &mut sc.up, t)?;
            exec.gs_bias_silu_f16(&sc.up, &blk.ff_up1_b, &mut sc.s16, t, ffn)?;
            exec.matvec_batch_f16(&blk.ff_down1_w, &sc.s16, &mut sc.n, t)?;
            let ctc_here = li + 1 == ctc_at;
            exec.gs_post_ln_f16(
                &mut sc.x,
                &sc.n,
                &blk.ff_down1_b,
                (&blk.ln2_w, &blk.ln2_b, ctc_here.then_some(&mut sc.s16)),
                t,
                e,
                0.5,
                eps,
            )?;

            // `-plus` feature concat: this block's post-norm output is one of
            // the sources the projector's cross-attention reads. Captured
            // before the CTC splice, which is where both references take it
            // (transformers captures at `idx` and runs CTC at `idx ==
            // num_layers // 2`; llama.cpp's `is_feature_layer(il + 1)` sits
            // above its own CTC block) - with the shipped [3] against a CTC at
            // 8 the two never coincide, but the order is the reference's.
            if let Some(slot) = self.cat_layers.iter().position(|&l| l == li + 1) {
                exec.convert_f32_f16(&sc.x, &mut sc.cat16[slot], padded * e)?;
            }

            // dual-head CTC branch, spliced back into the residual
            if ctc_here {
                exec.matvec_batch_f16(&self.ctc_out_w, &sc.s16, &mut sc.ctc, t)?;
                exec.gs_bias_softmax_f16(&sc.ctc, &self.ctc_out_b, &mut sc.ctc16, t, self.ctc_dim)?;
                exec.matvec_batch_f16(&self.ctc_mid_w, &sc.ctc16, &mut sc.n, t)?;
                exec.gs_res_ln_f16(&mut sc.x, &sc.n, &self.ctc_mid_b, None, t, e, 1.0, eps)?;
            }
        }

        // ---- the kept frames, moved to the front for the projector ----
        // A segment keeps the middle of what it encoded; its first kept frame
        // is a whole number of 15-frame windows in (see SEGMENT_HALO), so the
        // projector runs unchanged over the kept rows once they sit at row 0.
        let (off, kf) = (k0 - x0, k1 - k0);
        let padded_k = kf.div_ceil(self.win) * self.win;
        if off != 0 || kf != t {
            exec.copy_region(&sc.x, off * e, &mut sc.n, 0, kf * e)?;
            exec.copy_region(&sc.n, 0, &mut sc.x, 0, kf * e)?;
            if padded_k > kf {
                exec.copy_region(&sc.zero, 0, &mut sc.x, kf * e, (padded_k - kf) * e)?;
            }
            for plane in sc.cat16.iter_mut() {
                exec.copy_region(plane, off * e, &mut sc.s16, 0, kf * e)?;
                exec.copy_region(&sc.s16, 0, plane, 0, kf * e)?;
                if padded_k > kf {
                    exec.copy_region(&sc.zero16, 0, plane, kf * e, (padded_k - kf) * e)?;
                }
            }
        }
        let padded = padded_k;
        let nb = padded / self.win;
        let nq = nb * self.queries;
        // ---- Q-Former projector ----
        exec.convert_f32_f16(&sc.x, &mut sc.enc16, padded * e)?;
        exec.copy_region(&sc.qtpl, 0, &mut sc.qz, 0, nq * e)?;
        exec.convert_f32_f16(&sc.qz, &mut sc.s16, nq * e)?;

        let pd_head = e / self.proj_heads;
        let pscale = 1.0 / (pd_head as f32).sqrt();
        for pl in &self.qf {
            // self-attention over the window's own queries
            exec.matvec_batch_f16(&pl.sq_w, &sc.s16, &mut sc.pq, nq)?;
            exec.bias_add(&mut sc.pq, &pl.sq_b, nq, e)?;
            exec.matvec_batch_f16(&pl.sk_w, &sc.s16, &mut sc.pk, nq)?;
            exec.bias_add(&mut sc.pk, &pl.sk_b, nq, e)?;
            exec.matvec_batch_f16(&pl.sv_w, &sc.s16, &mut sc.pv, nq)?;
            exec.bias_add(&mut sc.pv, &pl.sv_b, nq, e)?;
            exec.vision_attn_x(
                &sc.pq,
                &sc.pk,
                &sc.pv,
                &mut sc.pat,
                self.queries,
                self.queries,
                self.proj_heads,
                pd_head,
                nb,
                pscale,
            )?;
            exec.convert_f32_f16(&sc.pat, &mut sc.s16, nq * e)?;
            exec.matvec_batch_f16(&pl.so_w, &sc.s16, &mut sc.pn, nq)?;
            exec.gs_post_ln_f16(
                &mut sc.qz,
                &sc.pn,
                &pl.so_b,
                (&pl.sn_w, &pl.sn_b, Some(&mut sc.s16)),
                nq,
                e,
                1.0,
                PROJ_EPS,
            )?;

            // Cross-attention onto this window's 15 encoder frames. On `-plus`
            // the K/V input is the CONCATENATION of the captured hidden states
            // with the final one, and a concat along the contracted axis is a
            // sum of per-source GEMMs - so each band accumulates into the same
            // landing (the DFlash fusion runs its aux planes the same way)
            // rather than staging a 2048-wide plane every encode. The base
            // model has one band and this is the single GEMM it always was.
            for (b, w) in pl.ck_w.iter().enumerate() {
                let src = sc.cat16.get(b).unwrap_or(&sc.enc16);
                if b == 0 {
                    exec.matvec_batch_f16(w, src, &mut sc.pk, padded)?;
                } else {
                    exec.gemm_f16_f32_acc(&w.buf, src, &mut sc.pk, e, e, padded)?;
                }
            }
            exec.bias_add(&mut sc.pk, &pl.ck_b, padded, e)?;
            for (b, w) in pl.cv_w.iter().enumerate() {
                let src = sc.cat16.get(b).unwrap_or(&sc.enc16);
                if b == 0 {
                    exec.matvec_batch_f16(w, src, &mut sc.pv, padded)?;
                } else {
                    exec.gemm_f16_f32_acc(&w.buf, src, &mut sc.pv, e, e, padded)?;
                }
            }
            exec.bias_add(&mut sc.pv, &pl.cv_b, padded, e)?;
            exec.matvec_batch_f16(&pl.cq_w, &sc.s16, &mut sc.pq, nq)?;
            exec.bias_add(&mut sc.pq, &pl.cq_b, nq, e)?;
            exec.vision_attn_x(
                &sc.pq,
                &sc.pk,
                &sc.pv,
                &mut sc.pat,
                self.queries,
                self.win,
                self.proj_heads,
                pd_head,
                nb,
                pscale,
            )?;
            exec.convert_f32_f16(&sc.pat, &mut sc.s16, nq * e)?;
            exec.matvec_batch_f16(&pl.co_w, &sc.s16, &mut sc.pn, nq)?;
            exec.gs_post_ln_f16(
                &mut sc.qz,
                &sc.pn,
                &pl.co_b,
                (&pl.cn_w, &pl.cn_b, Some(&mut sc.s16)),
                nq,
                e,
                1.0,
                PROJ_EPS,
            )?;

            // GELU FFN (exact erf - BLIP-2's `hidden_act: "gelu"`)
            exec.matvec_batch_f16(&pl.fu_w, &sc.s16, &mut sc.up, nq)?;
            exec.whisper_bias_gelu_f16(&sc.up, &pl.fu_b, &mut sc.s16, nq, ffn)?;
            exec.matvec_batch_f16(&pl.fd_w, &sc.s16, &mut sc.pn, nq)?;
            exec.gs_post_ln_f16(
                &mut sc.qz,
                &sc.pn,
                &pl.fd_b,
                (&pl.fn_w, &pl.fn_b, Some(&mut sc.s16)),
                nq,
                e,
                1.0,
                PROJ_EPS,
            )?;
        }

        exec.matvec_batch_f16(&self.proj_w, &sc.s16, &mut sc.seg_out, nq)?;
        exec.bias_add(&mut sc.seg_out, &self.proj_b, nq, self.out_dim)?;
        Ok(nq)
    }
}

impl super::GpuGranite {
    /// Attach the granite-speech audio tower from the companion mmproj GGUF.
    ///
    /// Refuses everything the two files can silently disagree on. A wrong
    /// pairing here is not a crash later - it is fluent text about audio the
    /// model never heard, which is why each check names its reason:
    ///
    /// - a VISION mmproj (or any other projector) is rejected by
    ///   [`SpeechTower::load`] on `clip.projector_type`;
    /// - a projector that emits a different width than the decoder embeds
    ///   means the two files are from different models;
    /// - a vocab with no `<|audio|>` has nowhere for the template to render a
    ///   clip or for the prompt to splice one, so the pair is mismatched;
    /// - a checkpoint that already carries DeepStack is granite-VISION, and
    ///   an audio tower has no business on it.
    pub fn attach_audio(
        &mut self,
        mmproj: &paddock_models::mapped::MappedGguf,
    ) -> Result<(), GpuModelError> {
        if self.hp.has_deepstack() {
            return Err(GpuModelError::Unsupported(
                "this granite checkpoint carries a deepstack_mapping - it is granite-VISION, \
                 and an audio mmproj cannot be attached to it"
                    .into(),
            ));
        }
        if self.audio_pad_id.is_none() {
            return Err(GpuModelError::Unsupported(
                "granite-speech: the model's vocab has no `<|audio|>` token, so there is no \
                 placeholder for the chat template to render or for the prompt to splice at \
                 - the mmproj and the text model are not a matching pair"
                    .into(),
            ));
        }
        let tower = SpeechTower::load(self.exec.clone(), mmproj)?;
        if tower.out_dim != self.hp.n_embd {
            return Err(GpuModelError::Unsupported(format!(
                "granite-speech: the audio tower projects to {} but the decoder embeds {} - \
                 mismatched files",
                tower.out_dim, self.hp.n_embd
            )));
        }
        self.weights_bytes += tower.resident_bytes() as u64;
        self.audio = Some(tower);
        Ok(())
    }

    pub fn has_audio(&self) -> bool {
        self.audio.is_some()
    }

    /// Prompt rows a clip of `len` 16 kHz samples will occupy. Pure host
    /// arithmetic - the admission path calls it before a sample moves, and
    /// the runner calls the same rule (`crate::audio::granite`) to size the
    /// request against the context window.
    pub fn audio_tokens(&self, len: usize) -> Result<usize, GpuModelError> {
        let t = self.audio.as_ref().ok_or_else(|| {
            GpuModelError::Unsupported(
                "granite was loaded without an audio mmproj - set `mmproj` in the config to \
                 enable audio input"
                    .into(),
            )
        })?;
        Ok(t.tokens_for_samples(len))
    }

    /// Encode one clip and register it at `pos` in `slot`'s prompt, returning
    /// the rows it covers.
    ///
    /// Rides the same [`super::deepstack::MediaRegistry`] images use: the
    /// clip's embeddings are one stream, which `apply_embed` copies over the
    /// `<|audio|>` rows before layer 0 - the exact operation DeepStack's
    /// stream 0 performs for a picture. Nothing else in the walk changes,
    /// which is what gives audio prompts the paged KV, prefix cache, chunked
    /// prefill and captured decode graphs for free.
    ///
    /// No cache (unlike images): a transcription's clip is asked about once,
    /// and holding tens of MB of encoder output per turn would evict nothing
    /// useful. If multi-turn audio chat shows repeats, `img_cache`'s shape
    /// ports over directly.
    pub(crate) fn place_audio(
        &mut self,
        slot: usize,
        pos: usize,
        feats: &MelFeatures,
    ) -> Result<usize, GpuModelError> {
        let tower = self.audio.as_mut().ok_or_else(|| {
            GpuModelError::Unsupported(
                "audio request but no audio tower attached (--mmproj)".into(),
            )
        })?;
        let out = tower.encode(feats)?;
        let width = tower.out_dim;
        Ok(self.place_feats(
            slot,
            pos,
            std::sync::Arc::new(crate::gpu_model::granite::vision::MediaFeatures {
                streams: vec![out.embd],
                tokens: out.n_tokens,
                width,
            }),
        ))
    }
}
