//! Gemma 4 vision tower (`clip` GGUF, projector `gemma4v`) - the mmproj
//! encoder. Reference: llama.cpp b10058 `tools/mtmd/models/gemma4v.cpp` +
//! `clip.cpp` (hparams: rope_theta 100, n_merge 3, kq_scale 1.0, token
//! limits 40..280) + `mtmd-image.cpp` (dyn-size smart_resize, align 48).
//!
//!   pixels (bilinear smart-resize, ×2-1) -> 16×16 conv (no bias, as GEMM
//!   over im2row [c][ky][kx]-ordered patches) -> +tbl_x[pos_x] +tbl_y[pos_y]
//!   (factorized learned pos tables) -> 27 × [ RMS ln1 -> QKV -> per-head RMS
//!   q/k norms -> 2D NEOX rope (x half by pos_x, y half by pos_y, θ=100) ->
//!   weightless V RMS -> bidirectional attn (scale 1.0) -> out-proj ->
//!   attn_post_norm -> +res -> RMS ln2 -> GEGLU (gate/up/down) ->
//!   ffn_post_norm -> +res ] -> 3×3 avg-pool ×√embd -> (h-std_bias)·std_scale
//!   -> weightless RMS -> mm.input_projection -> [n_out, 5376].
//!
//! f32 activations over f16 weight planes: every GEMM operand
//! is staged to f16 and accumulated in f32, so the tower is resident at the
//! mmproj file's own byte count (1.20 GB, not the 2.40 GB the bf16->f32 widen
//! used to cost) and the projections land on tensor cores instead of cuBLAS
//! SGEMM. bf16->f16 is exact for every weight whose exponent fits - see
//! `narrow_to_f16`, which refuses the ones that don't rather than shipping an
//! `inf`. The pooler/std/rms tail runs host-side (one row per output token,
//! 280 of them by default - encode is once per image). Correctness gate:
//! end-to-end token parity vs llama-mtmd-cli after the splice lands.

use std::sync::Arc;

use cudarc::driver::CudaSlice;
use paddock_models::gguf::Value;
use paddock_models::mapped::MappedGguf;

use crate::gpu::{GpuError, GpuExecutor, HalfTensor};

/// patch_size × n_merge: resize aligns to whole OUTPUT tokens (48 px each).
const ALIGN: usize = 48;
const N_MERGE: usize = 3;
/// Output-token ceiling, and it is GOOGLE'S own, not a downstream guess:
/// `google/gemma-4-31B-it` config.json states `vision_soft_tokens_per_image:
/// 280` (and processor_config `max_soft_tokens: 280`) with `patch_size: 16`,
/// `pooling_kernel_size: 3` - hence ALIGN 48 and 280 × 48² source pixels.
/// llama.cpp's `set_limit_image_tokens(40, 280)` agrees on the ceiling.
const MAX_TOKENS: usize = 280;
/// Resolve the per-image soft-token ceiling for one served endpoint:
/// `max_image_tokens` from its config, else Google's published 280.
///
/// Why it is settable at all: 280 tokens is 280 x 48^2 = 645 kpx, so an A4
/// page arrives at 672x912 and small print is gone before the encoder sees
/// it - measured on a generated page, where the published cap read 1 of 12
/// amounts and 0 of 12 references correctly against 12/12 at 1120. Nothing
/// in the tower is built around 280: every buffer in `encode` is sized from
/// `n`, the host tail walks `n` rows, and the position tables carry
/// `pos_size` entries per axis (10240 on gemma-4-31B-it's mmproj, about
/// 163k px a side). The cap is a processor convention, not a capability.
///
/// It arrives as an explicit argument, never from the environment: this is
/// product config, and the rule is the one `load_with` already states. The
/// clamp is the model's own - a value the position tables cannot address is
/// refused down to what they can, and said out loud.
///
/// Not free, which the endpoint's owner is choosing knowingly: attention
/// over patches is quadratic, so 4x the tokens is 16x the tower's attention
/// math (once per image), transient encode memory goes ~0.2 -> ~0.8 GB, and
/// the page then occupies ~1100 prompt tokens instead of ~270.
fn resolved_max_tokens(pos_size: usize, asked: Option<usize>) -> usize {
    let Some(want) = asked else {
        return MAX_TOKENS;
    };
    // what the file's own tables can address for a square image, in tokens
    let axis_tokens = pos_size / N_MERGE;
    let ceiling = axis_tokens.saturating_mul(axis_tokens).max(MAX_TOKENS);
    let got = want.clamp(MIN_TOKENS, ceiling);
    if got != want {
        tracing::warn!(
            asked = want,
            using = got,
            floor = MIN_TOKENS,
            ceiling,
            "max_image_tokens out of range for this tower - clamped"
        );
    }
    if got != MAX_TOKENS {
        tracing::info!(
            tokens = got,
            published = MAX_TOKENS,
            pixels = got * ALIGN * ALIGN,
            "gemma4 image budget set by config (the checkpoint publishes 280)"
        );
    }
    got
}
/// The FLOOR is llama.cpp's, not Google's: its comment says "the model
/// performs quite poor with small images, we need to bump minimum image
/// tokens to 40". Google's config states no minimum. Kept because it only ever
/// UPSAMPLES a tiny image, which cannot lose detail - but labelled, so nobody
/// later reads it as a published number the way our Qwen cap was.
const MIN_TOKENS: usize = 40;
const MIN_PIXELS: usize = MIN_TOKENS * ALIGN * ALIGN;
const ROPE_THETA: f32 = 100.0;

/// One tower block. Projections are [`HalfTensor`] (the GEMM operand class);
/// the RMS norm weights stay f32 because their consumers are the f32
/// elementwise ops, and they are a few KB each.
struct VBlock {
    ln1: CudaSlice<f32>,
    wq: HalfTensor,
    wk: HalfTensor,
    wv: HalfTensor,
    q_norm: CudaSlice<f32>,
    k_norm: CudaSlice<f32>,
    wo: HalfTensor,
    attn_post: CudaSlice<f32>,
    ln2: CudaSlice<f32>,
    gate: HalfTensor,
    up: HalfTensor,
    down: HalfTensor,
    ffn_post: CudaSlice<f32>,
}

/// Encoded image: [n_tokens, llm_embd] embeddings ready for splice.
pub struct VisionOutput {
    pub embd: CudaSlice<f32>,
    pub n_tokens: usize,
}

pub struct VisionModel {
    exec: Arc<GpuExecutor>,
    #[allow(dead_code)] // tower geometry record (layer count is implied by weights.len())
    n_layers: usize,
    embd: usize,
    n_heads: usize,
    head_dim: usize,
    patch: usize,
    eps: f32,
    /// factorized pos tables, host: [2][pos_size][embd] (x table then y)
    pos_tbl: Vec<f32>,
    pos_size: usize,
    /// Output-token ceiling in force for this instance: `MAX_TOKENS` unless
    /// the dev override raised it (see `resolved_max_tokens`).
    max_tokens: usize,
    conv: HalfTensor, // [3*patch*patch, embd] flattened conv-as-GEMM
    blocks: Vec<VBlock>,
    std_bias: Vec<f32>,
    std_scale: Vec<f32>,
    mm_proj: HalfTensor, // [embd, llm_embd]
    ones_hd: CudaSlice<f32>,
}

fn key_u32(map: &MappedGguf, key: &str) -> Result<usize, GpuError> {
    map.gguf()
        .metadata
        .get(key)
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .ok_or_else(|| GpuError::Driver(format!("mmproj missing {key}")))
}

fn host_f32(map: &MappedGguf, name: &str) -> Result<(Vec<f32>, Vec<usize>), GpuError> {
    use paddock_models::ggml_type::GgmlType;
    let (info, bytes) = map.tensor_bytes(name)?;
    let dims: Vec<usize> = info.dims.iter().map(|&d| d as usize).collect();
    let host = match info.ggml_type {
        GgmlType::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        GgmlType::Bf16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
            .collect(),
        ty => return Err(GpuError::Driver(format!("{name}: unhandled type {ty:?}"))),
    };
    Ok((host, dims))
}

impl VisionModel {
    /// `max_image_tokens` is the endpoint's config value (None = the
    /// checkpoint's published 280) - see [`resolved_max_tokens`].
    pub fn load(
        exec: Arc<GpuExecutor>,
        map: &MappedGguf,
        max_image_tokens: Option<usize>,
    ) -> Result<Self, GpuError> {
        let n_layers = key_u32(map, "clip.vision.block_count")?;
        let embd = key_u32(map, "clip.vision.embedding_length")?;
        let n_heads = key_u32(map, "clip.vision.attention.head_count")?;
        let patch = key_u32(map, "clip.vision.patch_size")?;
        let head_dim = embd / n_heads;
        let eps = match map
            .gguf()
            .metadata
            .get("clip.vision.attention.layer_norm_epsilon")
        {
            Some(Value::F32(f)) => *f,
            _ => 1e-6,
        };

        // conv [patch, patch, 3, embd] (kx fastest) -> GEMM weight rows in
        // im2row order idx = c*patch² + ky*patch + kx, [3*patch², embd]
        let (cw, cd) = host_f32(map, "v.patch_embd.weight")?;
        if cd != [patch, patch, 3, embd] {
            return Err(GpuError::Driver(format!("conv dims {cd:?}")));
        }
        let pp = patch * patch;
        let mut flat = vec![0f32; 3 * pp * embd];
        for oc in 0..embd {
            for c in 0..3 {
                for ky in 0..patch {
                    for kx in 0..patch {
                        let src = ((oc * 3 + c) * patch + ky) * patch + kx;
                        flat[oc * 3 * pp + c * pp + ky * patch + kx] = cw[src];
                    }
                }
            }
        }
        // built host-side (im2row permutation), so it takes the slice-based
        // narrow rather than `upload_f16`'s map-based one
        let conv = HalfTensor {
            buf: exec.to_device_f16(&flat, "v.patch_embd.weight")?,
            dims: vec![3 * pp, embd],
        };

        let (pos_tbl, pd) = host_f32(map, "v.position_embd.weight")?;
        if pd.len() != 3 || pd[0] != embd || pd[2] != 2 {
            return Err(GpuError::Driver(format!("pos_embd dims {pd:?}")));
        }
        let pos_size = pd[1];

        let (std_bias, _) = host_f32(map, "v.std_bias")?;
        let (std_scale, _) = host_f32(map, "v.std_scale")?;

        // `dt` = f16 GEMM plane (the file's bf16 narrowed, checked); `vf` = f32
        // norm vector. Everything this tower multiplies goes through `dt`.
        let dt = |name: String| -> Result<HalfTensor, GpuError> { exec.upload_f16(map, &name) };
        let vf =
            |name: String| -> Result<CudaSlice<f32>, GpuError> { Ok(exec.upload(map, &name)?.buf) };
        let mut blocks = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            blocks.push(VBlock {
                ln1: vf(format!("v.blk.{i}.ln1.weight"))?,
                wq: dt(format!("v.blk.{i}.attn_q.weight"))?,
                wk: dt(format!("v.blk.{i}.attn_k.weight"))?,
                wv: dt(format!("v.blk.{i}.attn_v.weight"))?,
                q_norm: vf(format!("v.blk.{i}.attn_q_norm.weight"))?,
                k_norm: vf(format!("v.blk.{i}.attn_k_norm.weight"))?,
                wo: dt(format!("v.blk.{i}.attn_out.weight"))?,
                attn_post: vf(format!("v.blk.{i}.attn_post_norm.weight"))?,
                ln2: vf(format!("v.blk.{i}.ln2.weight"))?,
                gate: dt(format!("v.blk.{i}.ffn_gate.weight"))?,
                up: dt(format!("v.blk.{i}.ffn_up.weight"))?,
                down: dt(format!("v.blk.{i}.ffn_down.weight"))?,
                ffn_post: vf(format!("v.blk.{i}.ffn_post_norm.weight"))?,
            });
        }
        let mm_proj = dt("mm.input_projection.weight".to_owned())?;
        let ones_hd = exec.to_device(&vec![1.0f32; head_dim])?;

        let me = Self {
            exec,
            n_layers,
            embd,
            n_heads,
            head_dim,
            patch,
            eps,
            pos_tbl,
            pos_size,
            max_tokens: resolved_max_tokens(pos_size, max_image_tokens),
            conv,
            blocks,
            std_bias,
            std_scale,
            mm_proj,
            ones_hd,
        };
        tracing::info!(
            weight_mib = me.weight_bytes() / (1 << 20),
            "gemma4 mmproj resident at f16 (f32 accumulate)"
        );
        Ok(me)
    }

    /// Device bytes the f16 weight planes hold - everything the GEMMs read,
    /// equal to the mmproj file's own weight bytes rather than
    /// twice them. Norms and the pos tables are excluded: a few MB of f32, and
    /// not what the estimator was getting wrong.
    pub fn weight_bytes(&self) -> usize {
        let blk: usize = self
            .blocks
            .iter()
            .map(|b| {
                b.wq.bytes()
                    + b.wk.bytes()
                    + b.wv.bytes()
                    + b.wo.bytes()
                    + b.gate.bytes()
                    + b.up.bytes()
                    + b.down.bytes()
            })
            .sum();
        blk + self.conv.bytes() + self.mm_proj.bytes()
    }

    /// llm-side embedding width (the projector's output).
    pub fn llm_embd(&self) -> usize {
        self.mm_proj.dims[1]
    }

    /// The largest image this tower can use. Gemma 4 is the tightest of our
    /// three families by a wide margin - 280 tokens per image whatever you send
    /// it - so a client that sizes to this budget is sending far less than for
    /// qwen or granite, correctly. Reported from the instance, so a raised cap
    /// reaches the API's detail levels and the Studio's picker by the one path
    /// they already read.
    pub fn budget(&self) -> crate::generator::VisionBudget {
        crate::generator::VisionBudget {
            max_pixels: self.max_pixels() as u64,
            min_pixels: MIN_PIXELS as u64,
            // still None: the table bound is ~163k px per side, which no
            // client can hit and every client would have to carry. It is
            // enforced where it can actually bite, inside `resize_target`.
            max_edge: None,
            pixels_per_token: (ALIGN * ALIGN) as u64,
            max_tokens: self.max_tokens as u32,
            min_tokens: MIN_TOKENS as u32,
        }
    }

    /// Source pixels the cap allows.
    fn max_pixels(&self) -> usize {
        self.max_tokens * ALIGN * ALIGN
    }

    /// The longest edge the POSITION TABLES can address, in pixels, rounded
    /// down to a whole output token. `encode` indexes `pos_tbl` by patch
    /// column and row, so a picture wider than this in patches would read
    /// past the x table and into the y one - silently, since the two are one
    /// tensor. A 1:1 image can never reach it; a pathological 200:1 strip can.
    fn max_edge(&self) -> usize {
        (self.pos_size / N_MERGE) * ALIGN
    }

    /// smart_resize (llama.cpp calc_size_preserved_ratio): align to 48, keep
    /// pixels within [MIN, MAX] preserving aspect, then hold every edge inside
    /// what the position tables can address.
    pub fn resize_target(&self, w: usize, h: usize) -> (usize, usize) {
        resize_target_px(w, h, self.max_pixels(), self.max_edge())
    }

    /// Soft tokens a `w`x`h` picture encodes to - what `encode` will report,
    /// from the resize target alone (patch grid, then the 3x3 pool).
    pub fn tokens_for(&self, w: usize, h: usize) -> usize {
        let (tw, th) = self.resize_target(w, h);
        (tw / self.patch / N_MERGE) * (th / self.patch / N_MERGE)
    }

    /// Full preprocessing: bilinear resize to the smart target, then im2row
    /// patches with the graph's ×2-1 scaling folded in. Returns (patches
    /// [n_patches, 3·patch²], grid_w, grid_h).
    pub fn preprocess_rgb(&self, rgb: &[u8], w: usize, h: usize) -> (Vec<f32>, usize, usize) {
        let (tw, th) = self.resize_target(w, h);
        let resized = resize_bilinear_u8(rgb, w, h, tw, th);
        let (gw, gh) = (tw / self.patch, th / self.patch);
        let pp = self.patch * self.patch;
        let mut out = vec![0f32; gw * gh * 3 * pp];
        for py in 0..gh {
            for px in 0..gw {
                let base = (py * gw + px) * 3 * pp;
                for c in 0..3 {
                    for ky in 0..self.patch {
                        for kx in 0..self.patch {
                            let sy = py * self.patch + ky;
                            let sx = px * self.patch + kx;
                            let v = resized[(sy * tw + sx) * 3 + c] as f32 / 255.0;
                            out[base + c * pp + ky * self.patch + kx] = 2.0 * v - 1.0;
                        }
                    }
                }
            }
        }
        (out, gw, gh)
    }

    /// Encode preprocessed patches -> projected [n_out, llm_embd] embeddings.
    pub fn encode(&self, patches: &[f32], gw: usize, gh: usize) -> Result<VisionOutput, GpuError> {
        let exec = &self.exec;
        let n = gw * gh;
        let (embd, heads, hd) = (self.embd, self.n_heads, self.head_dim);
        let eps = self.eps;

        // One f16 staging buffer for every GEMM's activations, sized by the
        // widest row this tower ever feeds a GEMM, and rewritten in sequence.
        // The conversions are the price of an f32 elementwise chain driving
        // f16 tensor-core GEMMs: each is one streaming pass over rows the GEMM
        // then reads `out_dim` times over.
        let ffn_dim = self.blocks[0].gate.dims[1];
        let stage = n * ffn_dim.max(embd).max(self.conv.dims[0]);
        let mut s16 = exec.alloc_f16(stage)?;

        // conv-as-GEMM + factorized pos add (host-gathered per image)
        let d_patches = exec.to_device(patches)?;
        let mut x = exec.stream.alloc_zeros::<f32>(n * embd).map_err(gerr)?;
        exec.convert_f32_f16(&d_patches, &mut s16, n * self.conv.dims[0])?;
        exec.matvec_batch_f16(&self.conv, &s16, &mut x, n)?;
        let mut pos_add = vec![0f32; n * embd];
        let mut pos_x = vec![0u32; n];
        let mut pos_y = vec![0u32; n];
        for i in 0..n {
            let (cx, cy) = (i % gw, i / gw);
            pos_x[i] = cx as u32;
            pos_y[i] = cy as u32;
            let tx = &self.pos_tbl[cx * embd..(cx + 1) * embd];
            let ty = &self.pos_tbl[self.pos_size * embd + cy * embd..][..embd];
            for e in 0..embd {
                pos_add[i * embd + e] = tx[e] + ty[e];
            }
        }
        let d_pos_add = exec.to_device(&pos_add)?;
        exec.add(&mut x, &d_pos_add, n * embd)?;
        let d_pos_x = exec.to_device_u32(&pos_x)?;
        let d_pos_y = exec.to_device_u32(&pos_y)?;

        let ts = ROPE_THETA.powf(-2.0 / (hd / 2) as f32);
        let mut normed = exec.stream.alloc_zeros::<f32>(n * embd).map_err(gerr)?;
        let mut q = exec.stream.alloc_zeros::<f32>(n * embd).map_err(gerr)?;
        let mut k = exec.stream.alloc_zeros::<f32>(n * embd).map_err(gerr)?;
        let mut v = exec.stream.alloc_zeros::<f32>(n * embd).map_err(gerr)?;
        let mut qn = exec.stream.alloc_zeros::<f32>(n * embd).map_err(gerr)?;
        let mut kn = exec.stream.alloc_zeros::<f32>(n * embd).map_err(gerr)?;
        let mut vn = exec.stream.alloc_zeros::<f32>(n * embd).map_err(gerr)?;
        let mut attn = exec.stream.alloc_zeros::<f32>(n * embd).map_err(gerr)?;
        let mut gate = exec.stream.alloc_zeros::<f32>(n * ffn_dim).map_err(gerr)?;
        let mut up = exec.stream.alloc_zeros::<f32>(n * ffn_dim).map_err(gerr)?;
        let mut proj = exec.stream.alloc_zeros::<f32>(n * embd).map_err(gerr)?;

        for b in &self.blocks {
            exec.rmsnorm_batch(&x, &b.ln1, &mut normed, embd, eps, n)?;
            // q, k and v all read the same normed rows - stage them once
            exec.convert_f32_f16(&normed, &mut s16, n * embd)?;
            exec.matvec_batch_f16(&b.wq, &s16, &mut q, n)?;
            exec.matvec_batch_f16(&b.wk, &s16, &mut k, n)?;
            exec.matvec_batch_f16(&b.wv, &s16, &mut v, n)?;
            exec.rmsnorm_batch(&q, &b.q_norm, &mut qn, hd, eps, n * heads)?;
            exec.rmsnorm_batch(&k, &b.k_norm, &mut kn, hd, eps, n * heads)?;
            exec.rope2d_neox(&mut qn, &d_pos_x, &d_pos_y, n, heads, hd, ts)?;
            exec.rope2d_neox(&mut kn, &d_pos_x, &d_pos_y, n, heads, hd, ts)?;
            exec.rmsnorm_batch(&v, &self.ones_hd, &mut vn, hd, eps, n * heads)?;
            exec.vision_attn(&qn, &kn, &vn, &mut attn, n, heads, hd, 1.0)?;
            exec.convert_f32_f16(&attn, &mut s16, n * embd)?;
            exec.matvec_batch_f16(&b.wo, &s16, &mut proj, n)?;
            exec.rmsnorm_batch(&proj, &b.attn_post, &mut normed, embd, eps, n)?;
            exec.add(&mut x, &normed, n * embd)?;

            exec.rmsnorm_batch(&x, &b.ln2, &mut normed, embd, eps, n)?;
            exec.convert_f32_f16(&normed, &mut s16, n * embd)?;
            exec.matvec_batch_f16(&b.gate, &s16, &mut gate, n)?;
            exec.matvec_batch_f16(&b.up, &s16, &mut up, n)?;
            // GELU is the TOWER's activation, not the decoder's - deliberately
            // not Hparams::glu_act. gemma4's SigLIP tower is GELU; muse-
            // glimmer's ViT-G/14 Perception Encoder states its own,
            // and reading it off the text FFN would be a category error.
            exec.geglu(&mut gate, &up, n * ffn_dim)?;
            exec.convert_f32_f16(&gate, &mut s16, n * ffn_dim)?;
            exec.matvec_batch_f16(&b.down, &s16, &mut proj, n)?;
            exec.rmsnorm_batch(&proj, &b.ffn_post, &mut normed, embd, eps, n)?;
            exec.add(&mut x, &normed, n * embd)?;
        }

        // host tail: 3×3 avg pool ×√embd -> std -> weightless RMS (≤280 rows)
        let hx = exec.to_host_len(&x, n * embd)?;
        let (ow, oh) = (gw / N_MERGE, gh / N_MERGE);
        let n_out = ow * oh;
        let scale = (embd as f32).sqrt();
        let mut pooled = vec![0f32; n_out * embd];
        for oy in 0..oh {
            for ox in 0..ow {
                let dst = &mut pooled[(oy * ow + ox) * embd..][..embd];
                for ky in 0..N_MERGE {
                    for kx in 0..N_MERGE {
                        let src = ((oy * N_MERGE + ky) * gw + (ox * N_MERGE + kx)) * embd;
                        for e in 0..embd {
                            dst[e] += hx[src + e];
                        }
                    }
                }
                let inv = scale / (N_MERGE * N_MERGE) as f32;
                for (e, d) in dst.iter_mut().enumerate() {
                    // pooled·√embd -> (h - std_bias)·std_scale
                    *d = (*d * inv - self.std_bias[e]) * self.std_scale[e];
                }
                // weightless RMS norm
                let ms = dst.iter().map(|v| v * v).sum::<f32>() / embd as f32;
                let r = 1.0 / (ms + eps).sqrt();
                for d in dst.iter_mut() {
                    *d *= r;
                }
            }
        }

        let d_pooled = exec.to_device(&pooled)?;
        let mut out = exec
            .stream
            .alloc_zeros::<f32>(n_out * self.llm_embd())
            .map_err(gerr)?;
        // n_out = n/9, so the staging buffer sized for n rows covers this
        exec.convert_f32_f16(&d_pooled, &mut s16, n_out * embd)?;
        exec.matvec_batch_f16(&self.mm_proj, &s16, &mut out, n_out)?;
        Ok(VisionOutput {
            embd: out,
            n_tokens: n_out,
        })
    }
}

fn gerr<E: std::fmt::Display>(e: E) -> GpuError {
    GpuError::Driver(e.to_string())
}

/// Plain bilinear u8 RGB resize (identical formula to the qwen35 vision
/// preprocessor's - llama.cpp img_tool::resize BILINEAR class).
fn resize_bilinear_u8(src: &[u8], sw: usize, sh: usize, tw: usize, th: usize) -> Vec<u8> {
    if sw == tw && sh == th {
        return src.to_vec();
    }
    let mut out = vec![0u8; tw * th * 3];
    for y in 0..th {
        let fy = (y as f32 + 0.5) * sh as f32 / th as f32 - 0.5;
        let y0 = fy.floor().max(0.0) as usize;
        let y1 = (y0 + 1).min(sh - 1);
        let dy = (fy - y0 as f32).clamp(0.0, 1.0);
        for x in 0..tw {
            let fx = (x as f32 + 0.5) * sw as f32 / tw as f32 - 0.5;
            let x0 = fx.floor().max(0.0) as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let dx = (fx - x0 as f32).clamp(0.0, 1.0);
            for c in 0..3 {
                let p00 = src[(y0 * sw + x0) * 3 + c] as f32;
                let p01 = src[(y0 * sw + x1) * 3 + c] as f32;
                let p10 = src[(y1 * sw + x0) * 3 + c] as f32;
                let p11 = src[(y1 * sw + x1) * 3 + c] as f32;
                let v = p00 * (1.0 - dx) * (1.0 - dy)
                    + p01 * dx * (1.0 - dy)
                    + p10 * (1.0 - dx) * dy
                    + p11 * dx * dy;
                out[(y * tw + x) * 3 + c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// The resize itself, free of the tower so it can be checked without a GPU:
/// align to whole output tokens, fit inside `max_pixels` preserving aspect,
/// then hold every edge inside what the position tables address.
fn resize_target_px(w: usize, h: usize, max_pixels: usize, edge_cap: usize) -> (usize, usize) {
    let f = ALIGN as f32;
    let round_by = |x: f32| ((x / f).round() * f) as usize;
    let ceil_by = |x: f32| ((x / f).ceil() * f) as usize;
    let floor_by = |x: f32| ((x / f).floor() * f) as usize;
    let mut wb = round_by(w as f32).max(ALIGN);
    let mut hb = round_by(h as f32).max(ALIGN);
    if wb * hb > max_pixels {
        let beta = ((w * h) as f32 / max_pixels as f32).sqrt();
        wb = floor_by(w as f32 / beta).max(ALIGN);
        hb = floor_by(h as f32 / beta).max(ALIGN);
    } else if wb * hb < MIN_PIXELS {
        let beta = (MIN_PIXELS as f32 / (w * h) as f32).sqrt();
        wb = ceil_by(w as f32 * beta);
        hb = ceil_by(h as f32 * beta);
    }
    // the table bound, applied last: clamping an edge only ever drops rows
    // the tower could not have addressed anyway
    (wb.min(edge_cap).max(ALIGN), hb.min(edge_cap).max(ALIGN))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The measured table width on gemma-4-31B-it's mmproj
    /// (v.position_embd.weight = [1152, 10240, 2]): 10240 patch positions per
    /// axis, so ~163k px per side. Nowhere near binding for a real picture.
    const POS_SIZE: usize = 10240;
    fn edge_cap() -> usize {
        (POS_SIZE / N_MERGE) * ALIGN
    }
    fn tokens(t: (usize, usize)) -> usize {
        (t.0 / ALIGN) * (t.1 / ALIGN)
    }

    /// What the published cap does to an A4 page scanned at 300 dpi - the
    /// shape behind the digit errors on small print.
    #[test]
    fn the_published_cap_renders_an_a4_page_at_roughly_680x912() {
        let (w, h) = (2480usize, 3508);
        let t = resize_target_px(w, h, MAX_TOKENS * ALIGN * ALIGN, edge_cap());
        assert_eq!(t, (672, 912), "A4 at 300 dpi under the 280-token cap");
        assert!(tokens(t) <= MAX_TOKENS, "{} tokens", tokens(t));
    }

    /// ...and what raising it to 1120 buys: twice the linear resolution, for
    /// four times the tokens. Same aspect, same alignment, still inside cap.
    #[test]
    fn a_1120_token_cap_doubles_the_linear_resolution() {
        let (w, h) = (2480usize, 3508);
        let lo = resize_target_px(w, h, MAX_TOKENS * ALIGN * ALIGN, edge_cap());
        let hi = resize_target_px(w, h, 1120 * ALIGN * ALIGN, edge_cap());
        assert!(tokens(hi) <= 1120, "{} tokens", tokens(hi));
        assert!(
            hi.0 >= 2 * lo.0 - ALIGN && hi.1 >= 2 * lo.1 - ALIGN,
            "expected ~2x per edge, got {lo:?} -> {hi:?}"
        );
        // aspect preserved to within one output token
        let (ar_src, ar_hi) = (w as f32 / h as f32, hi.0 as f32 / hi.1 as f32);
        assert!((ar_src - ar_hi).abs() < 0.02, "aspect {ar_src} vs {ar_hi}");
    }

    /// A pathological strip: the area fits, but one edge would run past the
    /// x table and read the y table's rows. The clamp is the only thing
    /// between that and silently wrong position embeddings.
    #[test]
    fn a_long_strip_is_held_inside_the_position_tables() {
        let cap = edge_cap();
        let t = resize_target_px(cap * 4, ALIGN, 1120 * ALIGN * ALIGN, cap);
        assert!(t.0 <= cap, "width {} over the table bound {cap}", t.0);
        assert!(t.1 >= ALIGN);
    }

    /// The floor still upsamples a tiny image rather than starving the tower.
    #[test]
    fn a_tiny_image_is_still_brought_up_to_the_floor() {
        let t = resize_target_px(64, 48, MAX_TOKENS * ALIGN * ALIGN, edge_cap());
        assert!(tokens(t) >= MIN_TOKENS, "{} tokens from 64x48", tokens(t));
    }
}
