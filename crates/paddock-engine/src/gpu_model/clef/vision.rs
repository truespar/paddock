//! Clef's vision tower (Qwen3.5's, `model.visual.*`): an image as the
//! reference's processor and tower make it, into the backbone rows its
//! `<|image_pad|>` tokens hold.
//!
//! Preprocessing is the processor's own (Transformers' Qwen2-VL image
//! processor, torchvision backend), on the GPU: torch's uint8 antialiased
//! bicubic resize to the smart_resize target - byte for byte, integer
//! weights and sums - then `(x - 127.5) / 127.5` into 16 x 16 patches of
//! both temporal slots, rows in 2 x 2 merge-window order (slots 735-737).
//!
//! The tower: patch embed (Conv3d as a GEMM over the 1536-wide rows), the
//! learned 48 x 48 position grid read bilinearly at the patch grid's
//! linspace points (738), 27 pre-norm blocks - LayerNorm, fused qkv, the 2D
//! rope (739), attention within each image (740), projection, LayerNorm,
//! the tanh-GELU MLP - then the merger: LayerNorm per patch, the 2 x 2
//! window's four rows as one 4608-wide row, erf-GELU MLP to the backbone's
//! width. Precision is the backbone's: every projection on its GEMM (733,
//! the activation split two ways against the exact BF16 weight), the
//! attention 3xTF32, norms, residuals and softmax F32.
//!
//! The MLP's 4304 hidden channels run as 4320 - zero weight rows, zero
//! bias, so the extra channels are GELU(0) = 0 and their (zero) columns in
//! the down projection add nothing: the GEMM's k tile is 32 wide.
//!
//! Memory: the tower runs before the backbone in a pass, while the
//! backbone's planes are idle, so its planes ARE the backbone's - the
//! workspace sizes three of them for both (`TowerPlanes`). A pass's
//! patches are at most four times its rows (one image token is a 2 x 2
//! window), which is what those planes are sized for.

use cudarc::driver::CudaSlice;
use paddock_models::clef::ClefVisionConfig;
use paddock_models::ggml_type::GgmlType;
use paddock_models::safetensors::ShardedSafetensors;

use crate::clef_decision::ClefImage;
use crate::gpu::{ClefPlanMem, DiarEpi, GpuExecutor, QuantTensor};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::st_load::{bf16_bytes, f32_tensor};

/// Three planes an image's bytes pass through in preprocessing (see
/// [`Vision::preprocess`]).
pub(super) struct Stage<'a> {
    pub raw: &'a mut CudaSlice<f32>,
    pub mid: &'a mut CudaSlice<f32>,
    pub fin: &'a mut CudaSlice<f32>,
}

/// LayerNorm eps of every norm in the tower (`nn.LayerNorm(.., eps=1e-6)`).
const LN_EPS: f32 = 1e-6;
/// The vision rope's frequency base (`Qwen3_5VisionRotaryEmbedding`).
const ROPE_THETA: f32 = 10000.0;
/// Positions the rope table covers: an image's patch grid side. A pass's
/// patches are at most 4 x 16384 at an aspect of at most 200, so a side
/// stays under ~3700.
pub(super) const ROPE_POSITIONS: usize = 4096;
/// One patch's pixel row: 3 channels x 2 frames x 16 x 16.
const PATCH_IN: usize = 1536;
/// The longest side an image may have before resizing (its pixels are
/// capped well below what a side this long implies at the reference's
/// aspect limit) and after (the rope table's patches).
const PLAN_MAX_IN: usize = 1 << 17;
const PLAN_MAX_OUT: usize = ROPE_POSITIONS * 16;

/// A LayerNorm's weight and bias.
struct Ln {
    w: CudaSlice<f32>,
    b: CudaSlice<f32>,
}

/// A linear layer: BF16 rows as stored (`[n][k]`), F32 bias.
struct Lin {
    w: QuantTensor,
    b: CudaSlice<f32>,
}

struct Block {
    norm1: Ln,
    qkv: Lin,
    proj: Lin,
    norm2: Ln,
    /// `[ffn_pad][hidden]`, rows past ffn zero
    fc1: Lin,
    /// `[hidden][ffn_pad]`, columns past ffn zero
    fc2: Lin,
}

pub(super) struct Vision {
    pub(super) cfg: ClefVisionConfig,
    patch: Lin,
    /// the learned position grid, BF16 `[side * side][hidden]` as stored
    pos: CudaSlice<u8>,
    blocks: Vec<Block>,
    merge_norm: Ln,
    merge_fc1: Lin,
    merge_fc2: Lin,
    /// `(cos, sin)` per (position, frequency), `[ROPE_POSITIONS][18]`
    rope: CudaSlice<f32>,
    /// the resize's two axis plans (width, height), resident
    plans: [ClefPlanMem; 2],
    pub(super) bytes: u64,
}

/// The resident resize plans' bytes (two axes).
pub(super) fn plan_bytes() -> u64 {
    2 * ClefPlanMem::bytes(PLAN_MAX_IN, PLAN_MAX_OUT) as u64
}

/// The MLP's hidden width rounded up to the GEMM's 32-wide k tile.
pub(super) fn ffn_pad(cfg: &ClefVisionConfig) -> usize {
    cfg.ffn.div_ceil(32) * 32
}

/// Per patch, the widths of the tower's three planes: the residual stream,
/// the normed rows (and the attention's output), and the wide plane (the
/// pixel rows, the fused qkv, the MLP's hidden, the merger's hidden).
pub(super) fn plane_widths(cfg: &ClefVisionConfig) -> (usize, usize, usize) {
    let h = cfg.hidden;
    let wide = PATCH_IN.max(3 * h).max(ffn_pad(cfg)).max(h);
    (h, h, wide)
}

/// The vision rope's `(cos, sin)` for every position below `positions` and
/// each of the head's `dim / 4` frequencies, `[positions][freqs][2]`:
/// `inv_freq[k] = 1 / theta^(2k / (dim / 2))` formed as Transformers forms
/// it in F32 (the exponent an F32 quotient, the power correctly rounded),
/// the angle one F32 product, cos / sin of it in F64 rounded once - as the
/// backbone's table (`super::load::rope_table`).
fn rope_table(head_dim: usize, positions: usize) -> Vec<f32> {
    let dim = head_dim / 2;
    let freqs = dim / 2;
    let inv: Vec<f32> = (0..freqs)
        .map(|k| {
            let e = (2 * k) as f32 / dim as f32;
            1.0f32 / f64::from(ROPE_THETA).powf(f64::from(e)) as f32
        })
        .collect();
    let mut t = Vec::with_capacity(positions * freqs * 2);
    for p in 0..positions {
        for &f in &inv {
            let angle = f64::from(p as f32 * f);
            t.push(angle.cos() as f32);
            t.push(angle.sin() as f32);
        }
    }
    t
}

impl Vision {
    /// The tower's tensors off the shards (BF16, exact) - `None` when the
    /// checkpoint has no vision config. `up_f32` / `plane` count bytes.
    pub(super) fn load(
        exec: &GpuExecutor,
        st: &ShardedSafetensors,
        cfg: &ClefVisionConfig,
    ) -> Result<Self, GpuModelError> {
        let mut bytes = 0u64;
        let mut up = |v: &[f32]| -> Result<CudaSlice<f32>, GpuModelError> {
            bytes += 4 * v.len() as u64;
            Ok(exec.to_device(v)?)
        };
        let (h, f, fp) = (cfg.hidden, cfg.ffn, ffn_pad(cfg));
        let v = |n: &str| format!("model.visual.{n}");
        let mut planes = 0u64;
        let mut plane = |raw: &[u8], n: usize, k: usize| -> Result<QuantTensor, GpuModelError> {
            planes += raw.len() as u64;
            Ok(QuantTensor {
                bytes: exec.to_device_u8(raw)?,
                ty: GgmlType::Bf16,
                dims: vec![k, n],
            })
        };
        let ln =
            |name: &str, up: &mut dyn FnMut(&[f32]) -> Result<CudaSlice<f32>, GpuModelError>| {
                Ok::<_, GpuModelError>(Ln {
                    w: up(&f32_tensor(st, &format!("{name}.weight"), h)?)?,
                    b: up(&f32_tensor(st, &format!("{name}.bias"), h)?)?,
                })
            };
        // the Conv3d kernel [hidden][3][2][16][16] is a [hidden][1536] row
        // plane in the pixel rows' own column order (c, t, y, x)
        let patch = Lin {
            w: plane(
                bf16_bytes(st, &v("patch_embed.proj.weight"), h * PATCH_IN)?,
                h,
                PATCH_IN,
            )?,
            b: up(&f32_tensor(st, &v("patch_embed.proj.bias"), h)?)?,
        };
        let side = cfg.pos_side;
        let pos_raw = bf16_bytes(st, &v("pos_embed.weight"), side * side * h)?;
        let pos = exec.to_device_u8(pos_raw)?;
        let pos_bytes = pos_raw.len() as u64;
        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            let p = |n: &str| v(&format!("blocks.{i}.{n}"));
            // fc1 rows padded to ffn_pad with zero rows (and zero bias), fc2
            // columns with zero columns
            let fc1w = bf16_bytes(st, &p("mlp.linear_fc1.weight"), f * h)?;
            let mut fc1 = fc1w.to_vec();
            fc1.resize(fp * h * 2, 0);
            let mut fc1b = f32_tensor(st, &p("mlp.linear_fc1.bias"), f)?;
            fc1b.resize(fp, 0.0);
            let fc2w = bf16_bytes(st, &p("mlp.linear_fc2.weight"), h * f)?;
            let mut fc2 = vec![0u8; h * fp * 2];
            for (r, src) in fc2w.chunks_exact(f * 2).enumerate() {
                fc2[r * fp * 2..r * fp * 2 + f * 2].copy_from_slice(src);
            }
            blocks.push(Block {
                norm1: ln(&p("norm1"), &mut up)?,
                qkv: Lin {
                    w: plane(bf16_bytes(st, &p("attn.qkv.weight"), 3 * h * h)?, 3 * h, h)?,
                    b: up(&f32_tensor(st, &p("attn.qkv.bias"), 3 * h)?)?,
                },
                proj: Lin {
                    w: plane(bf16_bytes(st, &p("attn.proj.weight"), h * h)?, h, h)?,
                    b: up(&f32_tensor(st, &p("attn.proj.bias"), h)?)?,
                },
                norm2: ln(&p("norm2"), &mut up)?,
                fc1: Lin {
                    w: plane(&fc1, fp, h)?,
                    b: up(&fc1b)?,
                },
                fc2: Lin {
                    w: plane(&fc2, h, fp)?,
                    b: up(&f32_tensor(st, &p("mlp.linear_fc2.bias"), h)?)?,
                },
            });
        }
        let mw = 4 * h;
        let merge_norm = ln(&v("merger.norm"), &mut up)?;
        let merge_fc1 = Lin {
            w: plane(
                bf16_bytes(st, &v("merger.linear_fc1.weight"), mw * mw)?,
                mw,
                mw,
            )?,
            b: up(&f32_tensor(st, &v("merger.linear_fc1.bias"), mw)?)?,
        };
        let out = cfg.out_hidden;
        let merge_fc2 = Lin {
            w: plane(
                bf16_bytes(st, &v("merger.linear_fc2.weight"), out * mw)?,
                out,
                mw,
            )?,
            b: up(&f32_tensor(st, &v("merger.linear_fc2.bias"), out)?)?,
        };
        let rope = up(&rope_table(cfg.head_dim(), ROPE_POSITIONS))?;
        let plan = || -> Result<ClefPlanMem, GpuModelError> {
            Ok(ClefPlanMem {
                idx: exec.alloc_u32(2 * PLAN_MAX_OUT + 1)?,
                w: exec.alloc_u8(2 * (4 * PLAN_MAX_IN + 5 * PLAN_MAX_OUT))?,
            })
        };
        let plans = [plan()?, plan()?];
        Ok(Self {
            cfg: cfg.clone(),
            patch,
            pos,
            blocks,
            merge_norm,
            merge_fc1,
            merge_fc2,
            rope,
            plans,
            bytes: bytes + planes + pos_bytes,
        })
    }

    /// The processor alone: `images` (a pass's, in order) resized and
    /// patchified into `wide` (`[patches][1536]`, image after image).
    /// `stage` holds an image's bytes on the way - three idle planes read as
    /// bytes: the decoded image, the width pass's output, the resized
    /// image - so no picture allocates. Returns the patch rows' (patch row,
    /// patch column, grid rows, grid columns) `[patches][4]`, each image's
    /// first row (and the end), and the largest image's patches.
    pub(super) fn preprocess(
        &mut self,
        e: &GpuExecutor,
        images: &[&ClefImage],
        wide: &mut CudaSlice<f32>,
        stage: Stage<'_>,
    ) -> Result<(Vec<u32>, Vec<u32>, usize), GpuModelError> {
        let bad = |m: String| GpuModelError::Unsupported(format!("Clef image: {m}"));
        // per patch row: (patch row, patch column, grid rows, grid columns),
        // merge-window order; per image its first row
        let mut info = Vec::new();
        let mut cu = vec![0u32];
        let mut max_len = 0usize;
        for im in images {
            let (rh, rw) = im.resized;
            let (gh, gw) = (rh / 16, rw / 16);
            if gh.max(gw) > ROPE_POSITIONS {
                return Err(bad(format!(
                    "a {rw} x {rh} image is {gw} x {gh} patches; the tower reads sides up to \
                     {ROPE_POSITIONS}"
                )));
            }
            for bh in 0..gh / 2 {
                for bw in 0..gw / 2 {
                    for mh in 0..2 {
                        for mw in 0..2 {
                            info.extend([
                                (2 * bh + mh) as u32,
                                (2 * bw + mw) as u32,
                                gh as u32,
                                gw as u32,
                            ]);
                        }
                    }
                }
            }
            max_len = max_len.max(gh * gw);
            cu.push((info.len() / 4) as u32);
        }
        let p = info.len() / 4;
        if wide.len() < p * PATCH_IN {
            return Err(bad(format!("{p} patches overflow the pass's planes")));
        }

        // the processor: resize (the axes that change, width first), then
        // the pixel rows into `wide`
        let Stage { raw, mid, fin } = stage;
        let bytes = |p: &mut CudaSlice<f32>| p.len() * 4;
        let (cap_raw, cap_mid, cap_fin) = (bytes(raw), bytes(mid), bytes(fin));
        let mut row = 0usize;
        for im in images {
            let (rh, rw) = im.resized;
            let (w0, h0) = (im.width, im.height);
            let (n_raw, n_mid, n_fin) = (h0 * w0 * 3, h0 * rw * 3, rh * rw * 3);
            if n_raw > cap_raw || n_mid > cap_mid || n_fin > cap_fin || w0.max(h0) > PLAN_MAX_IN {
                return Err(bad(format!(
                    "a {w0} x {h0} image resized to {rw} x {rh} is past the staging planes \
                     ({cap_raw} / {cap_mid} / {cap_fin} bytes, sides up to {PLAN_MAX_IN})"
                )));
            }
            // SAFETY: the three planes are F32 scratch at this point of the
            // pass (nothing live in them), read as bytes and never as floats
            // until the tower writes them again; each view is inside its plane
            let (mut raw_b, mut mid_b, mut fin_b) = unsafe {
                (
                    raw.transmute_mut::<u8>(n_raw)
                        .ok_or(bad("staging".into()))?,
                    mid.transmute_mut::<u8>(n_mid)
                        .ok_or(bad("staging".into()))?,
                    fin.transmute_mut::<u8>(n_fin)
                        .ok_or(bad("staging".into()))?,
                )
            };
            e.upload_u8_into(&im.rgb, &mut raw_b)?;
            // the width pass, then the height pass - each only when its axis
            // changes, the reference's order; `at` names the plane the image
            // is in after them (0 raw, 1 mid, 2 fin)
            let [pw, ph] = &mut self.plans;
            let mut at = 0u8;
            if rw != w0 {
                let plan = e.clef_resample_plan(w0, rw, pw)?;
                e.clef_resample(&raw_b, &mut mid_b, h0, true, &plan, pw)?;
                at = 1;
            }
            if rh != h0 {
                let plan = e.clef_resample_plan(h0, rh, ph)?;
                if at == 1 {
                    e.clef_resample(&mid_b, &mut fin_b, rw, false, &plan, ph)?;
                } else {
                    e.clef_resample(&raw_b, &mut fin_b, rw, false, &plan, ph)?;
                }
                at = 2;
            }
            match at {
                0 => e.clef_patchify(&raw_b, rh, rw, wide, row)?,
                1 => e.clef_patchify(&mid_b, rh, rw, wide, row)?,
                _ => e.clef_patchify(&fin_b, rh, rw, wide, row)?,
            }
            row += (rh / 16) * (rw / 16);
        }
        Ok((info, cu, max_len))
    }

    /// Run `images` (a pass's, in order) through the processor and the
    /// tower; image i's merged rows land back to back in `out`
    /// (`[tokens][out_hidden]`, image order). `x`, `xn` and `wide` are the
    /// tower's planes (see [`plane_widths`]), sized for the pass's patches;
    /// `x` keeps the last block's rows. `spare` is a plane idle for the
    /// whole encode (the backbone's residual stream), staging only.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode(
        &mut self,
        e: &GpuExecutor,
        images: &[&ClefImage],
        x: &mut CudaSlice<f32>,
        xn: &mut CudaSlice<f32>,
        wide: &mut CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        spare: &mut CudaSlice<f32>,
    ) -> Result<(), GpuModelError> {
        // preprocessing stages through the backbone's residual plane (idle
        // until the embedding lands) and the tower's own x / xn (idle until
        // the patch embedding)
        let (info, cu, max_len) = self.preprocess(
            e,
            images,
            wide,
            Stage {
                raw: spare,
                mid: x,
                fin: xn,
            },
        )?;
        let c = &self.cfg;
        let (h, heads, fp) = (c.hidden, c.heads, ffn_pad(c));
        let p = info.len() / 4;
        if p == 0 {
            return Ok(());
        }
        let (wx, wn, ww) = plane_widths(c);
        if x.len() < p * wx
            || xn.len() < p * wn
            || wide.len() < p * ww
            || out.len() < p / 4 * c.out_hidden
        {
            return Err(GpuModelError::Unsupported(format!(
                "Clef image: {p} patches overflow the pass's planes"
            )));
        }
        let d_info = e.to_device_u32(&info)?;
        let d_cu = e.to_device_u32(&cu)?;

        e.clef_gemm_bias(
            wide,
            &self.patch.w,
            Some(&self.patch.b),
            x,
            p,
            DiarEpi::Store,
        )?;
        e.clef_vpos(x, &self.pos, &d_info, p, c.pos_side, h)?;
        let scale = (c.head_dim() as f64).powf(-0.5) as f32;
        for b in &self.blocks {
            e.clef_norm(x, &b.norm1.w, &b.norm1.b, xn, h, p, LN_EPS)?;
            e.clef_gemm_bias(xn, &b.qkv.w, Some(&b.qkv.b), wide, p, DiarEpi::Store)?;
            e.clef_vrope(wide, heads, p, &d_info, &self.rope)?;
            e.clef_vattn(wide, xn, &d_cu, images.len(), p, max_len, heads, scale)?;
            e.clef_gemm_bias(xn, &b.proj.w, Some(&b.proj.b), x, p, DiarEpi::Resid)?;
            e.clef_norm(x, &b.norm2.w, &b.norm2.b, xn, h, p, LN_EPS)?;
            e.clef_gemm_bias(xn, &b.fc1.w, Some(&b.fc1.b), wide, p, DiarEpi::GeluTanh)?;
            debug_assert_eq!(b.fc2.w.dims[0], fp);
            e.clef_gemm_bias(wide, &b.fc2.w, Some(&b.fc2.b), x, p, DiarEpi::Resid)?;
        }
        // the merger: a patch's LayerNorm, then a window's four rows as one
        e.clef_norm(x, &self.merge_norm.w, &self.merge_norm.b, xn, h, p, LN_EPS)?;
        let tokens = p / 4;
        e.clef_gemm_bias(
            xn,
            &self.merge_fc1.w,
            Some(&self.merge_fc1.b),
            wide,
            tokens,
            DiarEpi::Gelu,
        )?;
        e.clef_gemm_bias(
            wide,
            &self.merge_fc2.w,
            Some(&self.merge_fc2.b),
            out,
            tokens,
            DiarEpi::Store,
        )?;
        Ok(())
    }
}
