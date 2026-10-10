//! Qwen3.5/3.6 vision tower (`clip` GGUF, projector `qwen3vl_merger`) - the P4
//! multimodal encoder. Reference: b9895 `tools/mtmd/models/qwen3vl.cpp` (graph)
//! + `clip.cpp` (positions / preprocessing). Dataflow:
//!
//!   patches (16×16, MERGED 2×2-block order) -> conv0+conv1 (+bias)
//!   -> +learned pos-embd (bilinear-resized to the grid, same merged order)
//!   -> 27 × [ LN1 -> QKV(+bias) -> vision M-RoPE(q,k) -> bidirectional attn
//!   -> out-proj(+bias) -> +res -> LN2 -> up(+bias) -> GELU -> down(+bias) -> +res ]
//!   -> post-LN -> reshape [N/4, 4·embd] -> mm0(+bias) -> GELU -> mm2(+bias)
//!   -> [N/4, llm_embd] image embeddings (this GGUF has no deepstack layers).
//!
//! f32 activations over f16 weight planes: every GEMM operand is
//! staged to f16 and accumulated in f32, so the tower is resident at the mmproj
//! file's own byte count (~0.9 GB, not the 1.8 GB the widen used to cost) and
//! the projections run on tensor cores instead of cuBLAS SGEMM. The mmproj
//! ships BF16; bf16->f16 is exact for every weight whose exponent fits - see
//! `narrow_to_f16`, which refuses the ones that don't rather than shipping an
//! `inf`. Norms and biases stay f32: their consumers are the f32 elementwise
//! ops, and they are a few KB each. The encoder runs once per image -
//! correctness first, the token-level gate vs llama-mtmd-cli is the arbiter.

use std::sync::Arc;

use cudarc::driver::CudaSlice;
use half::f16;
use paddock_models::ggml_type::GgmlType;
use paddock_models::gguf::Value;
use paddock_models::mapped::MappedGguf;

use crate::gpu::{GpuExecutor, HalfTensor};
use crate::gpu_model::gpt_oss::GpuModelError;

struct VBlock {
    ln1_w: CudaSlice<f32>,
    ln1_b: CudaSlice<f32>,
    // QKV split into three [embd, embd] projections at load (host slice of the
    // fused [embd, 3*embd] tensor - GGUF rows are per-output, so rows 0..embd = q).
    wq: HalfTensor,
    wk: HalfTensor,
    wv: HalfTensor,
    bq: CudaSlice<f32>,
    bk: CudaSlice<f32>,
    bv: CudaSlice<f32>,
    wo: HalfTensor,
    bo: CudaSlice<f32>,
    ln2_w: CudaSlice<f32>,
    ln2_b: CudaSlice<f32>,
    up_w: HalfTensor,
    up_b: CudaSlice<f32>,
    down_w: HalfTensor,
    down_b: CudaSlice<f32>,
}

/// A DeepStack merger (Qwen3-VL's `deepstack_merger_list`): the tower's
/// residual after one of its blocks, read as [N/4, 4·embd] (post-shuffle
/// norm), LayerNorm -> fc1 -> GELU -> fc2 into LLM space. Qwen3-VL-8B taps
/// blocks 8, 16, 24 for LLM layers 0, 1, 2; the Qwen3.5 mmproj carries none.
struct Merger {
    ln_w: CudaSlice<f32>,
    ln_b: CudaSlice<f32>,
    fc1: HalfTensor,
    fc1_b: CudaSlice<f32>,
    fc2: HalfTensor,
    fc2_b: CudaSlice<f32>,
}

/// The attention's planes for one tower pass: q/k/v as halves feeding
/// `vision_attn_h` (whose output is the wo GEMM's staging plane), or, on a
/// pack without slot 833, the f32 kernel's output plane.
enum AttnPlanes {
    Half([CudaSlice<f16>; 3]),
    F32(CudaSlice<f32>),
}

/// The encoded image: merged-grid embeddings ready for LLM injection.
/// Patch rows one batched tower pass takes before it splits (a picture
/// larger than this runs alone, at its own size): ~80 KB of activations per
/// row on the 27B's tower, so ~0.65 GB at this width.
pub(crate) const TOWER_PASS_ROWS: usize = 8192;

pub struct VisionOutput {
    /// [n_tokens, llm_embd] device-resident image embeddings.
    pub embd: CudaSlice<f32>,
    /// Output grid (post 2×2 merge): the LLM M-RoPE h/w extents.
    pub nx: usize,
    pub ny: usize,
    /// DeepStack streams, one `[n_tokens, llm_embd]` plane per tap in tap
    /// order, ADDED to the LLM's residual at the image rows after decoder
    /// layer 0, 1, 2, ... Empty for a tower without them.
    pub deepstack: Vec<CudaSlice<f32>>,
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
    /// learned position embeddings kept on host [n_side*n_side, embd] (row-major
    /// grid) - bilinearly resized + merge-reordered per image, then uploaded.
    pos_embd: Vec<f32>,
    n_side: usize,
    /// Resize the pos-embd grid with `align_corners` (on) or half-pixel
    /// centres (off). On since 2026-10-09, because that is the reference:
    /// transformers' Qwen3.5 tower samples `linspace(0, side - 1, n)`
    /// (`vision_utils::get_vision_bilinear_indices_and_weights`) and
    /// llama.cpp's qwen3vl graph resizes with `GGML_SCALE_FLAG_ALIGN_CORNERS`.
    /// The lane was first gated at half-pixel while its rotary was also wrong
    /// (contiguous instead of interleaved sections, see the loader), and plain
    /// answers hid both; LightOnOCR-3's grounding boxes exposed them - with
    /// both fixed its box streams agree with llama.cpp to near-tie flips.
    align_corners: bool,
    /// (tower block index the tap follows, its merger), in tap order
    deepstack: Vec<(usize, Merger)>,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    /// Source pixels smart-resize may keep - Qwen's spec, or the file's own
    /// limits when it states them. Not llama.cpp's quarter-cap.
    budget: PixelBudget,

    patch_w0: HalfTensor,
    patch_w1: HalfTensor,
    patch_bias: CudaSlice<f32>,
    blocks: Vec<VBlock>,
    post_ln_w: CudaSlice<f32>,
    post_ln_b: CudaSlice<f32>,
    mm0: HalfTensor,
    mm0_b: CudaSlice<f32>,
    mm2: HalfTensor,
    mm2_b: CudaSlice<f32>,
}

/// `PADDOCK_VIS_PHASES=1` prints a per-encode phase breakdown.
/// Read once - an encode is not a hot loop, but neither is it a place to pay
/// for an env lookup per image.
fn phase_timing() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_VIS_PHASES").is_some())
}

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// Read a GGUF tensor (F32 / F16 / BF16) as host f32. Shared with the other
/// mmproj towers (qwen3_asr's audio tower loads through the same classes).
pub(crate) fn host_f32(
    map: &MappedGguf,
    name: &str,
) -> Result<(Vec<f32>, Vec<usize>), GpuModelError> {
    let (info, bytes) = map.tensor_bytes(name).map_err(crate::gpu::GpuError::from)?;
    let dims: Vec<usize> = info.dims.iter().map(|&d| d as usize).collect();
    let data = match info.ggml_type {
        GgmlType::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        GgmlType::F16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f16::from_le_bytes(*c).to_f32())
            .collect(),
        // bf16 = the top half of the f32 bit pattern - exact widening
        // (unsloth ships mmproj-BF16.gguf for the 9B/qwen3.5 family)
        GgmlType::Bf16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
            .collect(),
        t => panic!("vision tensor {name}: unsupported type {t:?}"),
    };
    Ok((data, dims))
}

impl VisionModel {
    pub fn load(exec: Arc<GpuExecutor>, map: &MappedGguf) -> Result<Self, GpuModelError> {
        let u = |k: &str| {
            map.gguf()
                .metadata
                .get(k)
                .and_then(Value::as_u64)
                .unwrap_or_else(|| panic!("missing vision meta {k}"))
        };
        let n_layers = u("clip.vision.block_count") as usize;
        let embd = u("clip.vision.embedding_length") as usize;
        let n_heads = u("clip.vision.attention.head_count") as usize;
        let patch = u("clip.vision.patch_size") as usize;
        let head_dim = embd / n_heads;
        let eps = map
            .gguf()
            .metadata
            .get("clip.vision.attention.layer_norm_epsilon")
            .and_then(Value::as_f32)
            .unwrap_or(1e-6);
        let arr3 = |k: &str| -> [f32; 3] {
            match map.gguf().metadata.get(k) {
                Some(Value::Array(a)) => {
                    let v: Vec<f32> = a.iter().filter_map(Value::as_f32).collect();
                    [v[0], v[1], v[2]]
                }
                _ => [0.5, 0.5, 0.5],
            }
        };

        // device upload helpers. `dt` = f16 GEMM plane (the file's F16/BF16
        // kept at 16 bits rather than widened); `vec1` = f32 norm/bias vector.
        // Cloned Arc so the closures don't hold a borrow across the struct move.
        let e = exec.clone();
        let dt =
            move |name: &str| -> Result<HalfTensor, GpuModelError> { Ok(e.upload_f16(map, name)?) };
        let e = exec.clone();
        let vec1 = move |name: &str| -> Result<CudaSlice<f32>, GpuModelError> {
            let (host, _) = host_f32(map, name)?;
            Ok(e.to_device(&host)?)
        };

        // patch conv weights [16,16,3,1152] -> flatten to [768, 1152] (in, out):
        // GGUF rows are per-output-channel with kw fastest - exactly the im2col
        // order this module builds (c-major, then ky, kx... see encode()).
        let mut patch_w0 = dt("v.patch_embd.weight")?;
        patch_w0.dims = vec![patch * patch * 3, embd];
        let mut patch_w1 = dt("v.patch_embd.weight.1")?;
        patch_w1.dims = vec![patch * patch * 3, embd];

        let (pos_host, pos_dims) = host_f32(map, "v.position_embd.weight")?;
        let n_side = (pos_dims[1] as f64).sqrt() as usize;
        assert_eq!(n_side * n_side, pos_dims[1], "non-square pos-embd grid");

        let mut blocks = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let t = |s: &str| format!("v.blk.{i}.{s}");
            // fused qkv [embd, 3*embd]: rows 0..embd = q, embd..2embd = k, rest v
            let (qkv_w, _) = host_f32(map, &t("attn_qkv.weight"))?;
            let (qkv_b, _) = host_f32(map, &t("attn_qkv.bias"))?;
            let n = embd * embd;
            // host slices of the fused QKV tensor: no named GGUF tensor to
            // upload, so these take the slice-based checked narrow
            let mk = |h: &[f32], which: &str| -> Result<HalfTensor, GpuModelError> {
                Ok(HalfTensor {
                    buf: exec.to_device_f16(h, &format!("v.blk.{i}.attn_qkv.weight[{which}]"))?,
                    dims: vec![embd, embd],
                })
            };
            blocks.push(VBlock {
                ln1_w: vec1(&t("ln1.weight"))?,
                ln1_b: vec1(&t("ln1.bias"))?,
                wq: mk(&qkv_w[..n], "q")?,
                wk: mk(&qkv_w[n..2 * n], "k")?,
                wv: mk(&qkv_w[2 * n..], "v")?,
                bq: exec.to_device(&qkv_b[..embd])?,
                bk: exec.to_device(&qkv_b[embd..2 * embd])?,
                bv: exec.to_device(&qkv_b[2 * embd..])?,
                wo: dt(&t("attn_out.weight"))?,
                bo: vec1(&t("attn_out.bias"))?,
                ln2_w: vec1(&t("ln2.weight"))?,
                ln2_b: vec1(&t("ln2.bias"))?,
                up_w: dt(&t("ffn_up.weight"))?,
                up_b: vec1(&t("ffn_up.bias"))?,
                down_w: dt(&t("ffn_down.weight"))?,
                down_b: vec1(&t("ffn_down.bias"))?,
            });
        }

        // DeepStack taps: `clip.vision.is_deepstack_layers` flags the tower
        // blocks a merger follows; their tensors are named by that ABSOLUTE
        // block index
        let mut deepstack = Vec::new();
        if let Some(Value::Array(flags)) =
            map.gguf().metadata.get("clip.vision.is_deepstack_layers")
        {
            for (i, flag) in flags.iter().enumerate() {
                if !matches!(flag, Value::Bool(true)) {
                    continue;
                }
                let t = |s: &str| format!("v.deepstack.{i}.{s}");
                deepstack.push((
                    i,
                    Merger {
                        ln_w: vec1(&t("norm.weight"))?,
                        ln_b: vec1(&t("norm.bias"))?,
                        fc1: dt(&t("fc1.weight"))?,
                        fc1_b: vec1(&t("fc1.bias"))?,
                        fc2: dt(&t("fc2.weight"))?,
                        fc2_b: vec1(&t("fc2.bias"))?,
                    },
                ));
            }
        }

        let me = Self {
            exec,
            n_layers,
            embd,
            n_heads,
            head_dim,
            patch,
            eps,
            pos_embd: pos_host,
            n_side,
            align_corners: true,
            deepstack,
            image_mean: arr3("clip.vision.image_mean"),
            image_std: arr3("clip.vision.image_std"),
            budget: PixelBudget::from_gguf(map, patch),
            patch_w0,
            patch_w1,
            patch_bias: vec1("v.patch_embd.bias")?,
            blocks,
            post_ln_w: vec1("v.post_ln.weight")?,
            post_ln_b: vec1("v.post_ln.bias")?,
            mm0: dt("mm.0.weight")?,
            mm0_b: vec1("mm.0.bias")?,
            mm2: dt("mm.2.weight")?,
            mm2_b: vec1("mm.2.bias")?,
        };
        tracing::info!(
            weight_mib = me.weight_bytes() / (1 << 20),
            deepstack_taps = me.deepstack.len(),
            "qwen35 mmproj resident at f16 (f32 accumulate)"
        );
        Ok(me)
    }

    /// Resize the learned pos-embd grid with `align_corners` from now on
    /// (Qwen3-VL's tower; see the field).
    pub fn set_align_corners(&mut self, on: bool) {
        self.align_corners = on;
    }

    /// How many DeepStack streams an encode returns.
    pub fn deepstack_taps(&self) -> usize {
        self.deepstack.len()
    }

    /// Device bytes the f16 weight planes hold - everything the GEMMs read,
    /// equal to the mmproj file's own weight bytes rather than
    /// twice them. Biases, norms and the pos table are excluded: a few MB of
    /// f32, and not what the estimator was getting wrong.
    pub fn weight_bytes(&self) -> usize {
        let blk: usize = self
            .blocks
            .iter()
            .map(|b| {
                b.wq.bytes()
                    + b.wk.bytes()
                    + b.wv.bytes()
                    + b.wo.bytes()
                    + b.up_w.bytes()
                    + b.down_w.bytes()
            })
            .sum();
        let taps: usize = self
            .deepstack
            .iter()
            .map(|(_, m)| m.fc1.bytes() + m.fc2.bytes())
            .sum();
        blk + self.patch_w0.bytes()
            + self.patch_w1.bytes()
            + self.mm0.bytes()
            + self.mm2.bytes()
            + taps
    }

    /// Bilinearly resize the learned pos-embd grid to (ph, pw), then reorder into
    /// the merged 2×2-block patch order. Identity gather when the grid matches.
    fn pos_embd_for(&self, pw: usize, ph: usize) -> Vec<f32> {
        let (e, s) = (self.embd, self.n_side);
        // source grid row-major [s, s, e] -> target [ph, pw, e]
        let sample = |gy: usize, gx: usize| -> &[f32] {
            let idx = gy * s + gx;
            &self.pos_embd[idx * e..(idx + 1) * e]
        };
        let mut grid = vec![0f32; ph * pw * e];
        if pw == s && ph == s {
            grid.copy_from_slice(&self.pos_embd);
        } else {
            // bilinear: half-pixel centers (ggml_interpolate bilinear
            // semantics), or corner-aligned when the tower says so
            let src = |i: usize, n: usize| -> f32 {
                if self.align_corners {
                    if n > 1 {
                        i as f32 * (s - 1) as f32 / (n - 1) as f32
                    } else {
                        0.0
                    }
                } else {
                    ((i as f32 + 0.5) * s as f32 / n as f32 - 0.5).clamp(0.0, (s - 1) as f32)
                }
            };
            for y in 0..ph {
                let sy = src(y, ph);
                let y0 = sy.floor() as usize;
                let y1 = (y0 + 1).min(s - 1);
                let fy = sy - y0 as f32;
                for x in 0..pw {
                    let sx = src(x, pw);
                    let x0 = sx.floor() as usize;
                    let x1 = (x0 + 1).min(s - 1);
                    let fx = sx - x0 as f32;
                    let (a, b, c, d) = (
                        sample(y0, x0),
                        sample(y0, x1),
                        sample(y1, x0),
                        sample(y1, x1),
                    );
                    let out = &mut grid[(y * pw + x) * e..(y * pw + x + 1) * e];
                    for j in 0..e {
                        let top = a[j] + (b[j] - a[j]) * fx;
                        let bot = c[j] + (d[j] - c[j]) * fx;
                        out[j] = top + (bot - top) * fy;
                    }
                }
            }
        }
        // merged 2×2-block order
        let mut out = vec![0f32; ph * pw * e];
        let mut ptr = 0usize;
        for yb in (0..ph).step_by(2) {
            for xb in (0..pw).step_by(2) {
                for dy in 0..2 {
                    for dx in 0..2 {
                        let src = ((yb + dy) * pw + (xb + dx)) * e;
                        out[ptr * e..(ptr + 1) * e].copy_from_slice(&grid[src..src + e]);
                        ptr += 1;
                    }
                }
            }
        }
        out
    }

    /// Encode a normalized planar image ([3][h][w] f32, already (v-mean)/std) into
    /// LLM-space embeddings. `w`/`h` must be multiples of 2·patch (32).
    pub fn encode(&self, img: &[f32], w: usize, h: usize) -> Result<VisionOutput, GpuModelError> {
        let mut out = self.encode_batch(&[(img, w, h)])?;
        Ok(out.pop().expect("batch of one"))
    }

    /// Encode B same-size images in one tower pass (rows = B·n). Every tower op
    /// is row-independent except attention, which runs per image over its own
    /// row window (a group per image in `vision_attn_h`, or `vision_attn_at`
    /// per image on the f32 chain), so an image's rows are BITWISE independent
    /// of where it sits in the batch and of what else is in it - both gated in
    /// `batched_encode_matches_serial`. The tower weights are read once for the
    /// batch instead of B times.
    ///
    /// What is not bitwise is the comparison against B serial `encode` calls:
    /// cuBLAS picks its kernel off the row count, so the K-reduction order
    /// changes at B·n rows (measured rel 1.55e-6 for one 1280×1280 GEMM at 308
    /// vs 616 rows), and the tower rounds its activations to f16 between GEMMs,
    /// which parks that seed on the f16 quantization floor within a few layers
    /// - rel ~1e-3 out of 27 layers, and flat across wildly different images
    ///   because it is a floor, not propagation. That is below the tower's own
    ///   f16 resolution; the semantic arbiters are the serving greedy gates.
    ///   (The old "rel ~4e-6" figure here predates moving the tower off f32.)
    ///   This is the concurrent-image serving lever: a serial per-request
    ///   encode is what makes concurrent-image TTFT climb like a staircase.
    ///   Mixed-size batches are grouped by the CALLER; this asserts uniform
    ///   (w, h).
    pub fn encode_batch(
        &self,
        imgs: &[(&[f32], usize, usize)],
    ) -> Result<Vec<VisionOutput>, GpuModelError> {
        let b = imgs.len();
        assert!(b > 0);
        let (_, w, h) = imgs[0];
        let (patch, e) = (self.patch, self.embd);
        assert!(
            w % (patch * 2) == 0 && h % (patch * 2) == 0,
            "image must be 32-aligned"
        );
        for (img, iw, ih) in imgs {
            assert_eq!((*iw, *ih), (w, h), "encode_batch group must share dims");
            assert_eq!(img.len(), 3 * w * h);
        }
        let (pw, ph) = (w / patch, h / patch);
        let n = pw * ph;
        let rows = b * n;
        let exec = &self.exec;

        // host im2col in the merged 2×2-block order (+ the vision rope positions,
        // axis-major [4, n] per image = [y, x, y, x] - b9895 clip.cpp's exact
        // fill), images concatenated row-wise; pos/pos-embd repeat per image
        let t_phase = std::time::Instant::now();
        let k2 = patch * patch;
        let mut patches = vec![0f32; rows * 3 * k2];
        let mut pos = vec![0u32; 4 * rows];
        for (bi, (img, _, _)) in imgs.iter().enumerate() {
            let mut ptr = 0usize;
            let base = bi * n;
            for yb in (0..ph).step_by(2) {
                for xb in (0..pw).step_by(2) {
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let (py, px) = (yb + dy, xb + dx);
                            let row = base + ptr;
                            let dst = &mut patches[row * 3 * k2..(row + 1) * 3 * k2];
                            for c in 0..3 {
                                for ky in 0..patch {
                                    let src = c * w * h + (py * patch + ky) * w + px * patch;
                                    dst[c * k2 + ky * patch..c * k2 + ky * patch + patch]
                                        .copy_from_slice(&img[src..src + patch]);
                                }
                            }
                            // axis-major over the whole batch ([4, rows]): the
                            // mrope kernel reads y at [row] and x at [rows+row]
                            // with the call's total row count as the stride
                            pos[base + ptr] = py as u32;
                            pos[rows + base + ptr] = px as u32;
                            pos[2 * rows + base + ptr] = py as u32;
                            pos[3 * rows + base + ptr] = px as u32;
                            ptr += 1;
                        }
                    }
                }
            }
        }

        let t_im2col = t_phase.elapsed();

        let t_phase = std::time::Instant::now();
        let pe = self.pos_embd_for(pw, ph);
        let mut pe_rep = Vec::with_capacity(b * pe.len());
        for _ in 0..b {
            pe_rep.extend_from_slice(&pe);
        }
        let t_pos = t_phase.elapsed();

        let t_phase = std::time::Instant::now();
        let d_patches = exec.to_device(&patches)?;
        let d_pos = exec.to_device_u32(&pos)?;
        let d_pe = exec.to_device(&pe_rep)?;
        let t_upload = t_phase.elapsed();
        let t_phase = std::time::Instant::now();

        // One f16 staging buffer for every GEMM's activations, sized by the
        // widest row this tower feeds a GEMM (the merger's 4·embd) and
        // rewritten in sequence. The conversions are the price of an f32
        // elementwise chain driving f16 tensor-core GEMMs: each is one
        // streaming pass over rows the GEMM then reads `out_dim` times over.
        let ffn = self.blocks[0].up_w.dims[1];
        // the merger consumes 4 tower rows per row, so its two GEMMs are sized
        // off rows/4 rather than rows - priced separately instead of folded
        // into the per-row max, which would over-allocate 4x for the 4·embd
        // input plane
        let stage = (rows * ffn.max(e).max(self.patch_w0.dims[0]))
            .max((rows / 4) * self.mm0.dims[0].max(self.mm0.dims[1]));
        let mut s16 = exec.alloc_f16(stage)?;

        // dual patch conv (still image: both convs on the same pixels, summed)
        let mut d_x = exec.alloc(rows * e)?;
        let mut d_t = exec.alloc(rows * e)?;
        exec.convert_f32_f16(&d_patches, &mut s16, rows * self.patch_w0.dims[0])?;
        exec.matvec_batch_f16(&self.patch_w0, &s16, &mut d_x, rows)?;
        exec.matvec_batch_f16(&self.patch_w1, &s16, &mut d_t, rows)?;
        exec.add(&mut d_x, &d_t, rows * e)?;
        exec.bias_add(&mut d_x, &self.patch_bias, rows, e)?;
        exec.add(&mut d_x, &d_pe, rows * e)?;

        // scratch
        let mut d_n = exec.alloc(rows * e)?;
        let mut d_q = exec.alloc(rows * e)?;
        let mut d_k = exec.alloc(rows * e)?;
        let mut d_v = exec.alloc(rows * e)?;
        // Attention on halves where the pack can (slots 833 + 620): q/k/v land
        // as f16 in one pass off the projection planes and the attention
        // writes the wo GEMM's f16 staging plane itself, so the f32 attention
        // plane and the convert behind it are gone - and the f32 kernel already
        // rounded q/k/v to f16 into its fragments, so it is the same math on
        // narrower planes. Bit-identical to the f32 chain, which an older pack
        // keeps.
        let mut attn = if exec.has_vision_qkv_h() {
            AttnPlanes::Half([
                exec.alloc_f16(rows * e)?,
                exec.alloc_f16(rows * e)?,
                exec.alloc_f16(rows * e)?,
            ])
        } else {
            AttnPlanes::F32(exec.alloc(rows * e)?)
        };
        let mut d_up = exec.alloc(rows * ffn)?;
        // The FFN's seams ride its GEMMs where the pack can (slots 834/835):
        // the up GEMM lands bias + GELU as halves and wo / down add onto the
        // residual stream, so the f32 up and projection planes never land -
        // bit-identical, d_up / d_n staying only as the pack's fallback scratch.
        // LN2 then stages into the attention's q plane, free until the next
        // layer's rope, because the up GEMM writes the staging plane.
        let fused_ffn = matches!(attn, AttnPlanes::Half(_)) && exec.has_tower_ffn_fused();
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let theta_scale = 10000f32.powf(-2.0 / (self.head_dim / 2) as f32);

        // Per-op-class accounting for `PADDOCK_VIS_PHASES`. It
        // SYNCHRONIZES after each class, which serializes the layer - that is
        // the point (a launch queue tells you nothing about where the time
        // went) and it costs ~27 x 6 syncs against a multi-second encode.
        // Never on by default; the un-gated path launches exactly as before.
        let mut acc = [0f64; 6];
        let mut mark_t = std::time::Instant::now();
        macro_rules! mark {
            ($i:expr) => {
                if phase_timing() {
                    exec.stream
                        .synchronize()
                        .map_err(|e| crate::gpu::GpuError::Driver(e.to_string()))?;
                    acc[$i] += ms(mark_t.elapsed());
                    mark_t = std::time::Instant::now();
                }
            };
        }
        // DeepStack streams, image-major: every tap's merger runs on the
        // residual after its block, the same [rows/4, 4e] view + two GEMMs
        // as the final merger, and the result is split per image like `embd`
        let n4 = rows / 4;
        let mut ds_per_image: Vec<Vec<CudaSlice<f32>>> = (0..b).map(|_| Vec::new()).collect();
        // The elementwise chain between the GEMMs runs FUSED, as on the
        // PaddleOCR-VL tower: LN writes the f16 staging plane directly, the
        // q/k biases ride the rope load, the o/down biases ride the residual
        // add, and the FFN's bias+GELU+convert is one pass. Each fusion keeps
        // the unfused chain's IEEE order and its one f16 round (vision.cuh's
        // fusion header), so the tower's output is bit-identical. Unfused, a
        // 1440x2048 page (11520 rows) re-streamed ~1.3 GB a layer between the
        // GEMMs - about a fifth of the tower's time on GB10.
        for (bi, blk) in self.blocks.iter().enumerate() {
            // q, k and v all read the same normed rows - staged once, in f16
            exec.layernorm_f16(&d_x, &blk.ln1_w, &blk.ln1_b, &mut s16, rows, e, self.eps)?;
            mark!(0);
            exec.matvec_batch_f16(&blk.wq, &s16, &mut d_q, rows)?;
            exec.matvec_batch_f16(&blk.wk, &s16, &mut d_k, rows)?;
            exec.matvec_batch_f16(&blk.wv, &s16, &mut d_v, rows)?;
            // positions are axis-major over the whole pass ([4, rows]), so one
            // batched call reads every row's (y, x) as the serial call did
            match &mut attn {
                AttnPlanes::Half([q16, k16, v16]) => {
                    mark!(1);
                    exec.mrope_vision_qkv_h(
                        (&d_q, &d_k, &d_v),
                        (&blk.bq, &blk.bk, &blk.bv),
                        &d_pos,
                        (&mut *q16, &mut *k16, &mut *v16),
                        rows,
                        self.n_heads,
                        self.head_dim,
                        theta_scale,
                        scale,
                    )?;
                    mark!(2);
                    // the pass's pictures share their dims, so one launch
                    // covers them all, a group per picture over its own rows
                    exec.vision_attn_h(
                        q16,
                        k16,
                        v16,
                        &mut s16,
                        n,
                        n,
                        self.n_heads,
                        self.head_dim,
                        b,
                    )?;
                    mark!(3);
                }
                AttnPlanes::F32(d_a) => {
                    exec.bias_add(&mut d_v, &blk.bv, rows, e)?;
                    mark!(1);
                    exec.mrope_vision_bias(
                        &mut d_q,
                        &blk.bq,
                        &d_pos,
                        rows,
                        self.n_heads,
                        self.head_dim,
                        theta_scale,
                    )?;
                    exec.mrope_vision_bias(
                        &mut d_k,
                        &blk.bk,
                        &d_pos,
                        rows,
                        self.n_heads,
                        self.head_dim,
                        theta_scale,
                    )?;
                    mark!(2);
                    for bi in 0..b {
                        exec.vision_attn_at(
                            &d_q,
                            &d_k,
                            &d_v,
                            d_a,
                            bi * n,
                            n,
                            self.n_heads,
                            self.head_dim,
                            scale,
                        )?;
                    }
                    mark!(3);
                    exec.convert_f32_f16(d_a, &mut s16, rows * e)?;
                }
            }
            match (&mut attn, fused_ffn) {
                (AttnPlanes::Half([ln16, _, _]), true) => {
                    exec.matvec_batch_f16_bias_res(
                        &blk.wo, &s16, &mut d_x, &blk.bo, &mut d_n, rows,
                    )?;
                    mark!(4);
                    exec.layernorm_f16(&d_x, &blk.ln2_w, &blk.ln2_b, ln16, rows, e, self.eps)?;
                    // gelu_pytorch_tanh
                    exec.matvec_batch_f16_gelu_tanh(
                        &blk.up_w, ln16, &mut s16, &blk.up_b, &mut d_up, rows,
                    )?;
                    exec.matvec_batch_f16_bias_res(
                        &blk.down_w,
                        &s16,
                        &mut d_x,
                        &blk.down_b,
                        &mut d_n,
                        rows,
                    )?;
                }
                _ => {
                    exec.matvec_batch_f16(&blk.wo, &s16, &mut d_n, rows)?;
                    exec.add_bias_res(&mut d_x, &d_n, &blk.bo, rows, e)?;
                    mark!(4);
                    exec.layernorm_f16(&d_x, &blk.ln2_w, &blk.ln2_b, &mut s16, rows, e, self.eps)?;
                    exec.matvec_batch_f16(&blk.up_w, &s16, &mut d_up, rows)?;
                    exec.gelu_bias_f16(&d_up, &blk.up_b, &mut s16, rows, ffn)?; // gelu_pytorch_tanh
                    exec.matvec_batch_f16(&blk.down_w, &s16, &mut d_n, rows)?;
                    exec.add_bias_res(&mut d_x, &d_n, &blk.down_b, rows, e)?;
                }
            }
            mark!(5);
            if let Some((_, m)) = self.deepstack.iter().find(|(at, _)| *at == bi) {
                let (mid, out_dim) = (m.fc1.dims[1], m.fc2.dims[1]);
                exec.layernorm(&d_x, &m.ln_w, &m.ln_b, &mut d_n, n4, 4 * e, self.eps)?;
                exec.convert_f32_f16(&d_n, &mut s16, n4 * 4 * e)?;
                let mut d_m = exec.alloc(n4 * mid)?;
                exec.matvec_batch_f16(&m.fc1, &s16, &mut d_m, n4)?;
                exec.bias_add(&mut d_m, &m.fc1_b, n4, mid)?;
                exec.gelu(&mut d_m, n4 * mid)?;
                exec.convert_f32_f16(&d_m, &mut s16, n4 * mid)?;
                let mut d_o = exec.alloc(n4 * out_dim)?;
                exec.matvec_batch_f16(&m.fc2, &s16, &mut d_o, n4)?;
                exec.bias_add(&mut d_o, &m.fc2_b, n4, out_dim)?;
                let per = (n / 4) * out_dim;
                for (img, streams) in ds_per_image.iter_mut().enumerate() {
                    let mut one = exec.alloc(per)?;
                    exec.copy_region(&d_o, img * per, &mut one, 0, per)?;
                    streams.push(one);
                }
            }
        }
        if phase_timing() {
            tracing::info!(
                "qwen35-vis: layers x{}  ln1 {:.0}  qkv {:.0}  mrope {:.0}  attn {:.0}  \
                 out {:.0}  ffn {:.0}  (ms, GPU, summed over layers)",
                self.blocks.len(),
                acc[0],
                acc[1],
                acc[2],
                acc[3],
                acc[4],
                acc[5]
            );
        }

        // merger: consecutive 2x2-block rows are contiguous -> [rows/4, 4e] is a
        // view, and image boundaries land on 4-row multiples (n % 4 == 0); the
        // post-LN lands in the f16 staging plane directly (n4 x 4e == rows x e)
        exec.layernorm_f16(
            &d_x,
            &self.post_ln_w,
            &self.post_ln_b,
            &mut s16,
            rows,
            e,
            self.eps,
        )?;
        let n4 = rows / 4;
        let mid = self.mm0.dims[1];
        let mut d_m = exec.alloc(n4 * mid)?;
        exec.matvec_batch_f16(&self.mm0, &s16, &mut d_m, n4)?;
        exec.gelu_bias_f16(&d_m, &self.mm0_b, &mut s16, n4, mid)?;
        let out_dim = self.mm2.dims[1];
        let mut d_out = exec.alloc(n4 * out_dim)?;
        exec.matvec_batch_f16(&self.mm2, &s16, &mut d_out, n4)?;
        exec.bias_add(&mut d_out, &self.mm2_b, n4, out_dim)?;

        // split into per-image owned outputs
        let per = (n / 4) * out_dim;
        let mut outs = Vec::with_capacity(b);
        for (bi, deepstack) in ds_per_image.into_iter().enumerate() {
            let mut embd = exec.alloc(per)?;
            exec.copy_region(&d_out, bi * per, &mut embd, 0, per)?;
            outs.push(VisionOutput {
                embd,
                nx: pw / 2,
                ny: ph / 2,
                deepstack,
            });
        }
        if phase_timing() {
            // the tower phase is launches, not work, until the stream drains -
            // synchronize before reading the clock or the GPU column is a lie
            // and every millisecond it hides lands on whoever syncs next
            exec.stream
                .synchronize()
                .map_err(|e| crate::gpu::GpuError::Driver(e.to_string()))?;
            tracing::info!(
                "qwen35-vis: encode b={b} {w}x{h} rows={rows}  im2col {:.1} ms  pos-embd {:.1} ms  \
                 upload {:.1} ms  tower {:.1} ms",
                ms(t_im2col),
                ms(t_pos),
                ms(t_upload),
                ms(t_phase.elapsed())
            );
        }
        Ok(outs)
    }

    /// Normalize interleaved RGB u8 pixels into the planar f32 layout `encode`
    /// takes: out[c][y][x] = (px/255 - mean[c]) / std[c].
    pub fn normalize_rgb(&self, rgb: &[u8], w: usize, h: usize) -> Vec<f32> {
        assert_eq!(rgb.len(), 3 * w * h);
        let mut out = vec![0f32; 3 * w * h];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    let v = rgb[(y * w + x) * 3 + c] as f32 / 255.0;
                    out[c * w * h + y * w + x] = (v - self.image_mean[c]) / self.image_std[c];
                }
            }
        }
        out
    }

    /// Grid geometry for a given pixel size (pre-merge patch grid).
    pub fn patch_grid(&self, w: usize, h: usize) -> (usize, usize) {
        (w / self.patch, h / self.patch)
    }

    /// The merged token grid `(nx, ny)` a `w`x`h` picture encodes to - what
    /// `encode` will report, known from the resize target alone, so a prompt's
    /// layout can be built before any of its pictures is encoded.
    pub fn merged_grid(&self, w: usize, h: usize) -> (usize, usize) {
        let (tw, th) = smart_resize_target(w, h, self.patch, self.budget);
        let (pw, ph) = self.patch_grid(tw, th);
        (pw / 2, ph / 2)
    }

    /// Device bytes one `encode_batch` pass over `rows` patch rows allocates,
    /// by the same arithmetic the pass allocates with: the patch upload,
    /// positions and position embeddings, the six `embd`-wide f32 activation
    /// planes and the attention's three f16 ones, the FFN plane and the f16
    /// staging plane, then the merger's two
    /// planes and the per-picture outputs it hands back (plus DeepStack's
    /// streams where the tower has taps). The plan reserves this for the
    /// largest pass the tower can be asked for, since the pass allocates on
    /// demand and hands everything back when it ends.
    pub fn pass_bytes(&self, rows: usize) -> u64 {
        let e = self.embd;
        let ffn = self.blocks[0].up_w.dims[1];
        let k_in = self.patch_w0.dims[0];
        let (mid, out) = (self.mm0.dims[1], self.mm2.dims[1]);
        let n4 = rows / 4;
        let stage = (rows * ffn.max(e).max(k_in)).max(n4 * self.mm0.dims[0].max(mid));
        // six embd-wide f32 planes plus the half attention's three f16 ones
        // (an older pack's f32 attention plane instead is half a plane less)
        let f32s = rows * (k_in + e + 6 * e + ffn) + n4 * (mid + 2 * out);
        let taps = self.deepstack.len();
        let deepstack = if taps == 0 {
            0
        } else {
            let widest = self
                .deepstack
                .iter()
                .map(|(_, m)| m.fc1.dims[1])
                .max()
                .unwrap_or(0);
            taps * n4 * out + n4 * (widest + out)
        };
        (4 * (f32s + deepstack) + 4 * 4 * rows + 2 * stage + 2 * 3 * rows * e) as u64
    }

    /// The largest image this tower can use. One vision token is a 2·patch
    /// square block, so the pixel budget divides straight into a token count.
    pub fn budget(&self) -> crate::generator::VisionBudget {
        let align = (self.patch * 2) as u64;
        let per_token = align * align;
        crate::generator::VisionBudget {
            max_pixels: self.budget.max_pixels as u64,
            min_pixels: self.budget.min_pixels as u64,
            // area-bounded only: smart_resize has no per-edge cap, so a very
            // long strip is legal as long as it fits the area
            max_edge: None,
            pixels_per_token: per_token,
            max_tokens: (self.budget.max_pixels as u64 / per_token) as u32,
            min_tokens: (self.budget.min_pixels as u64 / per_token) as u32,
        }
    }

    /// Cap one picture at `tokens` merged tokens by lowering the pixel budget
    /// smart-resize serves at (and so the budget every API surface reports).
    /// Returns `(was, now)` in tokens when it lowered it.
    ///
    /// The batched lane prefills a picture's rows in ONE pass - they attend to
    /// the picture's last row - so a picture longer than the planned prefill
    /// pass would regrow the serving scratch past the plan. Qwen's spec allows
    /// 16384 tokens (16.8 MP); the 27B's planned 8192-row pass still takes
    /// twice llama.cpp's 4096 cap, and a `detail: auto` picture is 4096.
    pub fn cap_image_tokens(&mut self, tokens: usize) -> Option<(usize, usize)> {
        let per_token = (self.patch * 2) * (self.patch * 2);
        let was = self.budget.max_pixels / per_token;
        if tokens >= was {
            return None;
        }
        self.budget.max_pixels = tokens * per_token;
        self.budget.min_pixels = self.budget.min_pixels.min(self.budget.max_pixels);
        Some((was, tokens))
    }

    /// llama.cpp's qwen3vl `dyn_size` preprocessing for an arbitrary-size RGB
    /// image: smart-resize target, aspect-preserving Pillow bicubic with a
    /// black letterbox (PAD_CEIL, [`resize_pad_black`]), then mean/std
    /// normalization. Returns the planar f32 image `encode` takes plus its
    /// (32-aligned) pixel dims.
    pub fn preprocess_rgb(&self, rgb: &[u8], w: usize, h: usize) -> (Vec<f32>, usize, usize) {
        let t0 = std::time::Instant::now();
        let (tw, th) = smart_resize_target(w, h, self.patch, self.budget);
        let (canvas, t_resize) = if (tw, th) == (w, h) {
            (None, t0.elapsed())
        } else {
            let c = resize_pad_black(rgb, w, h, tw, th);
            (Some(c), t0.elapsed())
        };
        let t1 = std::time::Instant::now();
        let img = match &canvas {
            Some(c) => self.normalize_rgb(c, tw, th),
            None => self.normalize_rgb(rgb, w, h),
        };
        if phase_timing() {
            tracing::info!(
                "qwen35-vis: preprocess {w}x{h} -> {tw}x{th}  resize {:.1} ms  normalize {:.1} ms",
                ms(t_resize),
                ms(t1.elapsed())
            );
        }
        (img, tw, th)
    }
}

/// How many source pixels smart-resize may keep. Separate from the resize
/// ALGORITHM deliberately - the algorithm is llama.cpp-parity material, the
/// budget is a policy input, and conflating them is how we shipped a quarter
/// of Qwen's real resolution without noticing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PixelBudget {
    pub min_pixels: usize,
    pub max_pixels: usize,
}

impl PixelBudget {
    /// QWEN'S own SPEC, and what we serve at. `preprocessor_config.json` for
    /// Qwen3.5-9B and Qwen3.6-27B both carry:
    ///
    /// ```json
    /// "size": { "longest_edge": 16777216, "shortest_edge": 65536 },
    /// "patch_size": 16, "merge_size": 2
    /// ```
    ///
    /// Those two numbers are PIXEL AREAS despite the `edge` names (the
    /// Qwen2VLImageProcessorFast convention that replaced `max_pixels` /
    /// `min_pixels`) - 16777216 = 16384 tokens x 32², 65536 = 64 x 32².
    pub const QWEN_SPEC: Self = Self {
        min_pixels: 65_536,
        max_pixels: 16_777_216,
    };

    /// What llama.cpp's mtmd uses: `set_limit_image_tokens(8, 4096)`, i.e. a
    /// QUARTER of the spec ceiling, in a branch whose own comment cites the
    /// config that permits 16384. Kept only so the preprocessing parity test
    /// can state which budget it is comparing under; never used to serve.
    pub const LLAMACPP: Self = Self {
        min_pixels: 8 * 1024,
        max_pixels: 4096 * 1024,
    };

    /// The budget for a tower with this patch size, read from the file when it
    /// says (llama.cpp defines `clip.vision.image_{min,max}_pixels` and writes
    /// them for some architectures) and falling back to Qwen's published spec
    /// when it does not - which is the case for the mmproj files we serve.
    pub fn from_gguf(map: &MappedGguf, patch: usize) -> Self {
        let px = |k: &str| {
            map.gguf()
                .metadata
                .get(k)
                .and_then(Value::as_u64)
                .map(|v| v as usize)
        };
        let _ = patch; // spec values are already absolute pixel areas
        Self {
            min_pixels: px("clip.vision.image_min_pixels").unwrap_or(Self::QWEN_SPEC.min_pixels),
            max_pixels: px("clip.vision.image_max_pixels").unwrap_or(Self::QWEN_SPEC.max_pixels),
        }
    }
}

/// Port of llama.cpp b9895 `img_tool::calc_size_preserved_ratio` (transformers
/// "smart_resize"): round both edges to the 2·patch grid, then floor-rescale
/// over the pixel budget / ceil-rescale under it.
///
/// The BUDGET is a parameter rather than a constant so the divergence from
/// llama.cpp is visible at every call site: we serve at [`PixelBudget::QWEN_SPEC`]
/// and the parity test compares under [`PixelBudget::LLAMACPP`].
pub fn smart_resize_target(
    w: usize,
    h: usize,
    patch: usize,
    budget: PixelBudget,
) -> (usize, usize) {
    let align = patch * 2;
    let PixelBudget {
        min_pixels,
        max_pixels,
    } = budget;

    let f = align as f32;
    let round_by = |x: f32| ((x / f).round() * f) as usize;
    let ceil_by = |x: f32| ((x / f).ceil() * f) as usize;
    let floor_by = |x: f32| ((x / f).floor() * f) as usize;

    let mut h_bar = round_by(h as f32).max(align);
    let mut w_bar = round_by(w as f32).max(align);
    if h_bar * w_bar > max_pixels {
        let beta = ((h * w) as f32 / max_pixels as f32).sqrt();
        h_bar = floor_by(h as f32 / beta).max(align);
        w_bar = floor_by(w as f32 / beta).max(align);
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f32 / (h * w) as f32).sqrt();
        h_bar = ceil_by(h as f32 * beta);
        w_bar = ceil_by(w as f32 * beta);
    }
    (w_bar, h_bar)
}

/// llama.cpp's `img_tool::resize` with the qwen3vl settings (b11516:
/// RESIZE_ALGO_BICUBIC through `resize_pillow`, PAD_CEIL, black): scale = min
/// over both axes, new dims ceil-clamped, Pillow's exact bicubic (our
/// [`crate::pillow`] port), centered composite (floor offsets). Until
/// 2026-10-09 this was b9895's align-corners BILINEAR - llama.cpp has since
/// moved to Pillow bicubic, which is also the filter Qwen's own processor
/// names (`resample: 3`), and the parity gate is the newest release.
pub use crate::pillow::resize_pad_black;

#[cfg(test)]
mod tests {
    use super::*;

    /// The resize ALGORITHM, held against llama.cpp's budget - the shape math
    /// is what mtmd parity is about, and it must not drift.
    #[test]
    fn smart_resize_matches_llama_semantics() {
        let b = PixelBudget::LLAMACPP;
        // aligned in-budget image: identity (the historical gate's no-op case)
        assert_eq!(smart_resize_target(768, 768, 16, b), (768, 768));
        // 600 rounds up to 608 (19 blocks); width already aligned
        assert_eq!(smart_resize_target(800, 600, 16, b), (800, 608));
        // over budget: floor-by-factor after beta = sqrt(6e6 / 4194304)
        assert_eq!(smart_resize_target(3000, 2000, 16, b), (2496, 1664));
        // under budget: ceil-by-factor up to >= 8 tokens
        assert_eq!(smart_resize_target(50, 40, 16, b), (128, 96));
        // degenerate tiny input clamps to one block before the budget check
        let (w, h) = smart_resize_target(1, 1, 16, b);
        assert!(w >= 32 && h >= 32 && w * h >= b.min_pixels);
    }

    /// The budget we actually SERVE at is Qwen's, and it is 4x llama.cpp's.
    ///
    /// This is the bug the split exists for: `preprocessor_config.json` on
    /// Qwen3.5-9B and Qwen3.6-27B both say `size.longest_edge = 16777216`,
    /// while llama.cpp's mtmd calls `set_limit_image_tokens(8, 4096)` in a
    /// branch whose own comment cites that config. We had inherited the cap,
    /// so a 3000x2000 photo lost three quarters of its pixels before the tower
    /// ever saw it.
    #[test]
    fn the_served_budget_is_qwens_spec_not_llamacpps_quarter() {
        let (spec, llama) = (PixelBudget::QWEN_SPEC, PixelBudget::LLAMACPP);
        assert_eq!(
            spec.max_pixels,
            4 * llama.max_pixels,
            "spec is 4x mtmd's cap"
        );
        // 16777216 = 16384 tokens x (2*16)^2, straight off the config
        assert_eq!(spec.max_pixels, 16_384 * 32 * 32);

        // a 3000x2000 photo: mtmd cuts it, Qwen's spec keeps every pixel
        // (6e6 < 16.7e6), so the only rounding is the 32-grid alignment
        assert_eq!(smart_resize_target(3000, 2000, 16, llama), (2496, 1664));
        assert_eq!(smart_resize_target(3000, 2000, 16, spec), (3008, 2016));

        // and the ceiling still binds when it should
        let (w, h) = smart_resize_target(8000, 6000, 16, spec);
        assert!(w * h <= spec.max_pixels, "{w}x{h} = {} over budget", w * h);
        assert!(w * h > llama.max_pixels, "should exceed the old cap");
    }

    /// `detail: auto` lands exactly on the cap qwen was serving under before
    /// the spec fix - the whole reason the default is a token count and the
    /// number is 4096.
    ///
    /// If someone widens AUTO_MAX_TOKENS, this fails and they have to decide
    /// consciously that every existing client's images just got bigger. That is
    /// the property worth pinning; the arithmetic is incidental.
    #[test]
    fn detail_auto_is_the_resolution_clients_already_had() {
        // one vision token = a (2*patch)^2 = 32x32 block at patch 16
        let per_token = (2 * 16u64) * (2 * 16);
        assert_eq!(
            crate::generator::AUTO_MAX_TOKENS as u64 * per_token,
            PixelBudget::LLAMACPP.max_pixels as u64,
            "auto must reproduce the cap clients already had, not merely approximate it"
        );
    }

    #[test]
    fn resize_pad_letterboxes_the_short_axis() {
        // 100x50 -> target 128x64 (smart target for the aspect): scale = min(1.28)
        // = 1.28 exactly both ways here, so no padding; use a skewed target
        let src = vec![255u8; 3 * 100 * 50];
        let out = resize_pad_black(&src, 100, 50, 128, 96);
        // scale = min(1.28, 1.92) = 1.28 -> new = 128 x ceil(64) = 128x64,
        // centered vertically at offset (96-64)/2 = 16
        let px = |x: usize, y: usize| out[(y * 128 + x) * 3];
        assert_eq!(px(0, 0), 0, "top pad row must be black");
        assert_eq!(px(64, 15), 0, "last pad row above the image");
        assert_eq!(px(64, 16), 255, "first image row");
        assert_eq!(px(64, 79), 255, "last image row");
        assert_eq!(px(64, 80), 0, "first pad row below the image");
    }

    #[test]
    fn resize_is_identity_when_same_size() {
        let src: Vec<u8> = (0..3 * 8 * 4).map(|i| (i * 7 % 251) as u8).collect();
        assert_eq!(resize_pad_black(&src, 8, 4, 8, 4), src);
    }

    #[test]
    fn dump_500x300_canvas_probes() {
        // cross-checked against an independent float32 Python replication of
        // the b9895 C++ (scratchpad ref_resize.py) - prints for manual diff
        let (w, h) = (500usize, 300usize);
        let mut rgb = vec![0u8; 3 * w * h];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 3;
                rgb[i] = ((x * 255) / w) as u8;
                rgb[i + 1] = ((y * 255) / h) as u8;
                rgb[i + 2] = (((x / 64 + y / 64) % 2) * 200 + 25) as u8;
            }
        }
        let canvas = resize_pad_black(&rgb, w, h, 512, 288);
        let mut hash = 2166136261u32;
        for &b in &canvas {
            hash = (hash ^ b as u32).wrapping_mul(16777619);
        }
        tracing::info!("fnv {hash:#x}");
        for (x, y) in [
            (0, 0),
            (16, 0),
            (17, 0),
            (100, 100),
            (256, 144),
            (495, 287),
            (300, 7),
            (300, 8),
        ] {
            let i = (y * 512 + x) * 3;
            tracing::info!(
                "canvas[{x},{y}] = ({}, {}, {})",
                canvas[i],
                canvas[i + 1],
                canvas[i + 2]
            );
        }
    }
}
