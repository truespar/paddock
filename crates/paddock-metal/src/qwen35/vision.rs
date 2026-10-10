//! Native Qwen3.5/3.6/3.8 and LightOnOCR-3 ViT, independent of CUDA execution.
//! RGB -> patches -> fused-QKV transformer -> merger. The elected LightOn
//! towers share exact Pillow page resizing with CUDA, then GPU patchification;
//! existing larger towers retain their separately qualified GPU resize path.
//! Tensor contractions run on M5
//! MPP, with F32 activation storage and native BF16 weights. MLX patch
//! convolution preserves its BF16 operands and reference SIMD reduction;
//! other families retain their qualified F16 patch path.
//! Vision contractions use MPP's reduced-multiplication-precision mode, as
//! does the current same-weights M5 reference. Accumulation/residuals remain
//! F32. Strict contractions remain independently tested; F32 storage alone
//! does not imply strict F32 multiplication in the elected vision path.
//! Encoder jobs yield between blocks so active decoding can run during a
//! cold image admission.
use super::*;
use paddock_engine::generator::VisionBudget;
use paddock_models::{gguf::Value, mapped::MappedGguf};

#[cfg(test)]
const E: usize = 1152;
const PATCH: usize = 768;
#[path = "vision_geometry.rs"]
mod geometry;
use geometry::TowerGeometry;
#[path = "vision_mlx.rs"]
mod mlx;
#[cfg(test)]
#[path = "vision_mlx_trace_tests.rs"]
mod mlx_trace_tests;
#[cfg(test)]
#[path = "vision_tests.rs"]
mod tests;

struct Block {
    ln1: Weight,
    ln1b: Weight,
    qkv: Weight,
    qkvb: Weight,
    out: Weight,
    outb: Weight,
    ln2: Weight,
    ln2b: Weight,
    up: Weight,
    upb: Weight,
    down: Weight,
    downb: Weight,
}

pub(super) struct Vision {
    geometry: TowerGeometry,
    /// Small MLX graph: Pillow grid, BF16 boundaries, and a full NHWC patch
    /// contraction. Other exports retain their independently tested graph.
    mlx_bf16: bool,
    patch0: Weight,
    patch1: Option<Weight>,
    bias: Weight,
    pos: Weight,
    blocks: Vec<Block>,
    post: Weight,
    postb: Weight,
    mm0: Weight,
    mm0b: Weight,
    mm2: Weight,
    mm2b: Weight,
    eps: f32,
    mean: [f32; 3],
    std: [f32; 3],
    pub(super) budget: VisionBudget,
}

pub(super) struct Output {
    pub(super) embd: Buffer,
    pub(super) nx: usize,
    pub(super) ny: usize,
}

pub(super) struct Job {
    rows: usize,
    grids: Vec<(usize, usize)>,
    x: Buffer,
    stage: Buffer,
    qkv: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    attn: Buffer,
    xy: Buffer,
    tiles: Buffer,
    tile_count: usize,
    layer: usize,
    pub(super) block_cost: std::time::Duration,
    pub(super) submits: usize,
    pub(super) gpu_seconds: f64,
}

fn error(s: impl Into<String>) -> MetalError {
    MetalError::Model(s.into())
}

/// Geometry only; pixel arithmetic lives in vis_patches. Same explicit
/// area budget as CUDA and the checkpoint, not llama.cpp's smaller default.
pub(super) fn resize(w: usize, h: usize, budget: VisionBudget) -> Result<(usize, usize)> {
    if w == 0
        || h == 0
        || w.checked_mul(h).and_then(|n| n.checked_mul(3)).is_none()
        || w > u32::MAX as usize
        || h > u32::MAX as usize
    {
        return Err(error("invalid vision image dimensions"));
    }
    let grid = |v: f32, mode: i32| {
        let v = v / 32.0;
        (match mode {
            -1 => v.floor(),
            1 => v.ceil(),
            _ => v.round(),
        } as usize)
            .max(1)
            * 32
    };
    let (mut tw, mut th) = (grid(w as f32, 0), grid(h as f32, 0));
    if tw as u64 * th as u64 > budget.max_pixels {
        let beta = ((w * h) as f32 / budget.max_pixels as f32).sqrt();
        tw = grid(w as f32 / beta, -1);
        th = grid(h as f32 / beta, -1);
    } else if (tw as u64 * th as u64) < budget.min_pixels {
        let beta = (budget.min_pixels as f32 / (w * h) as f32).sqrt();
        tw = grid(w as f32 * beta, 1);
        th = grid(h as f32 * beta, 1);
    }
    if tw
        .checked_mul(th)
        .is_none_or(|n| n as u64 > budget.max_pixels)
    {
        return Err(error(
            "image aspect ratio cannot fit the vision pixel budget",
        ));
    }
    Ok((tw, th))
}

impl Vision {
    pub(super) fn dimensions(&self, w: usize, h: usize) -> Result<(usize, usize)> {
        if !self.mlx_bf16 {
            return resize(w, h, self.budget);
        }
        // MLX-VLM/HF smart_resize uses Python's ties-to-even grid rounding,
        // then stretches the RGB page to that grid (not GGUF's PAD_CEIL).
        if w == 0
            || h == 0
            || w.max(h) as f64 / w.min(h) as f64 > 200.0
            || w > u32::MAX as usize
            || h > u32::MAX as usize
        {
            return Err(error("invalid MLX image dimensions/aspect ratio"));
        }
        let (mut tw, mut th) = (
            ((w as f64 / 32.).round_ties_even() * 32.) as usize,
            ((h as f64 / 32.).round_ties_even() * 32.) as usize,
        );
        let area = w as f64 * h as f64;
        if tw as u64 * th as u64 > self.budget.max_pixels {
            let beta = (area / self.budget.max_pixels as f64).sqrt();
            tw = ((w as f64 / beta / 32.).floor() as usize).max(1) * 32;
            th = ((h as f64 / beta / 32.).floor() as usize).max(1) * 32;
        } else if (tw as u64 * th as u64) < self.budget.min_pixels {
            let beta = (self.budget.min_pixels as f64 / area).sqrt();
            tw = (w as f64 * beta / 32.).ceil() as usize * 32;
            th = (h as f64 * beta / 32.).ceil() as usize * 32;
        }
        if tw == 0 || th == 0 || tw as u64 * th as u64 > self.budget.max_pixels {
            return Err(error("MLX image cannot fit pixel budget"));
        }
        Ok((tw, th))
    }

    pub(super) fn load(device: &MetalDevice, path: &Path, width: usize) -> Result<Self> {
        let g = TowerGeometry::for_decoder(width)?;
        if path.join("hadamard.json").is_file() {
            paddock_models::bonsai::BonsaiConfig::read(path).map_err(|e| error(e.to_string()))?;
            let source = paddock_models::safetensors::ShardedSafetensors::open_dir(path)
                .map_err(|e| error(e.to_string()))?;
            let (min_pixels, max_pixels) = mlx::vision_budget(path)?;
            return Self::from_planes(
                width,
                1e-6,
                [0.5; 3],
                [0.5; 3],
                min_pixels,
                max_pixels,
                None,
                |name, dims, half| mlx::vision_weight(device, &source, name, dims, half),
            );
        }
        if path.join("manifest.json").is_file() {
            if width != 5120 {
                return Err(error("Splash vision requires dense 27B geometry"));
            }
            let source =
                paddock_models::splash::Vision::open(path).map_err(|e| error(e.to_string()))?;
            let load = |name: &str, dims: &[usize], half: bool| -> Result<Weight> {
                let buffer = device.upload_with(
                    dims.iter().product::<usize>() * if half { 2 } else { 4 },
                    |out| {
                        source
                            .copy(name, dims, half, out)
                            .map_err(|e| error(e.to_string()))
                    },
                )?;
                Ok(Weight {
                    buffer,
                    ty: if half { 30 } else { 0 },
                    k: dims[0],
                    n: *dims.get(1).unwrap_or(&1),
                })
            };
            return Self::from_planes(width, 1e-6, [0.5; 3], [0.5; 3], 65536, 4194304, None, load);
        }
        if path.is_dir() {
            let cfg =
                paddock_models::mlx::QwenConfig::read(path).map_err(|e| error(e.to_string()))?;
            if !cfg.tied || cfg.width != width {
                return Err(error("MLX OCR tower/decoder mismatch"));
            }
            let source = paddock_models::safetensors::ShardedSafetensors::open_dir(path)
                .map_err(|e| error(e.to_string()))?;
            let (min_pixels, max_pixels) = mlx::vision_budget(path)?;
            mlx::validate_processor(path, (min_pixels, max_pixels))?;
            return Self::from_planes(
                width,
                1e-6,
                [0.5; 3],
                [0.5; 3],
                min_pixels,
                max_pixels,
                Some(mlx::patch_weight(device, &source, g.width)?),
                |name, dims, half| {
                    mlx::vision_weight_typed(
                        device,
                        &source,
                        name,
                        dims,
                        half,
                        paddock_models::safetensors::StDtype::Bf16,
                    )
                },
            );
        }
        let map = MappedGguf::open(path).map_err(|e| error(e.to_string()))?;
        if paddock_models::hadamard::HadamardSpec::from_gguf(map.gguf())
            .map_err(|e| error(e.to_string()))?
            .is_some()
        {
            return Err(error(
                "Qwen vision companion must not carry a language-model Hadamard rotation",
            ));
        }
        let meta = &map.gguf().metadata;
        let u = |k: &str| meta.get(k).and_then(Value::as_u64);
        if map.gguf().architecture() != Some("clip")
            || meta.get("clip.projector_type").and_then(Value::as_str) != Some("qwen3vl_merger")
            || u("clip.vision.block_count") != Some(g.layers as u64)
            || u("clip.vision.embedding_length") != Some(g.width as u64)
            || u("clip.vision.feed_forward_length") != Some(g.ff as u64)
            || u("clip.vision.attention.head_count") != Some(g.heads as u64)
            || u("clip.vision.patch_size") != Some(16)
            || map
                .gguf()
                .tensors
                .iter()
                .any(|t| t.name.contains("deepstack"))
        {
            return Err(error(
                "Metal vision requires a matching elected Qwen qwen3vl_merger tower without DeepStack",
            ));
        }
        let eps = meta
            .get("clip.vision.attention.layer_norm_epsilon")
            .and_then(Value::as_f32)
            .unwrap_or(1e-6);
        if !eps.is_finite() || eps <= 0.0 {
            return Err(error("invalid vision norm epsilon"));
        }
        let arr = |k: &str| -> Result<[f32; 3]> {
            let Some(Value::Array(v)) = meta.get(k) else {
                return Err(error(format!("missing {k}")));
            };
            let v = v
                .iter()
                .map(Value::as_f32)
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| error(format!("invalid {k}")))?;
            if v.len() != 3 || v.iter().any(|v| !v.is_finite()) {
                return Err(error(format!("invalid {k}")));
            }
            Ok([v[0], v[1], v[2]])
        };
        let mean = arr("clip.vision.image_mean")?;
        let std = arr("clip.vision.image_std")?;
        if std.iter().any(|&v| v <= 0.0) {
            return Err(error("invalid vision image standard deviation"));
        }
        let min_pixels = u("clip.vision.image_min_pixels").unwrap_or(65_536);
        let max_pixels = u("clip.vision.image_max_pixels").unwrap_or(16_777_216);
        if min_pixels == 0 || min_pixels > max_pixels || max_pixels > 16_777_216 {
            return Err(error("unsupported vision pixel budget"));
        }
        // Convert on GPU once, preserving native BF16 matrix weights and
        // their 16-bit footprint. F32 patch weights use F16 operands. Reject
        // nonfinite/overflowing planes instead of accepting a poisoned tower.
        let load = |name: &str, dims: &[usize], half: bool| -> Result<Weight> {
            let source = Weight::load(device, &map, name, dims)?;
            if !matches!(source.ty, 0 | 1 | 30) {
                return Err(error(format!("{name}: mmproj must be F32/F16/BF16")));
            }
            if half && dims.len() == 2 && source.ty != 30 {
                return Err(error(format!(
                    "{name}: Metal Qwen vision requires the canonical BF16 matrix companion"
                )));
            }
            let count: usize = dims.iter().product();
            let ty = if half && source.ty == 30 {
                30
            } else {
                u32::from(half)
            };
            let buffer = device.alloc(count * if half { 2 } else { 4 })?;
            let bad = device.upload(&0u32.to_le_bytes())?;
            let cmd = device.begin()?;
            cmd.dispatch(
                "vis_cast",
                &[&source.buffer, &buffer, &bad],
                &[count as u32, source.ty, ty],
                [count.div_ceil(256), 1, 1],
                256,
            );
            cmd.finish()?;
            // SAFETY: the cast has completed.
            if unsafe { bad.read_u32(1)[0] } != 0 {
                return Err(error(format!(
                    "{name}: nonfinite or F16-overflowing vision weights"
                )));
            }
            Ok(Weight {
                buffer,
                ty,
                k: dims[0],
                n: *dims.get(1).unwrap_or(&1),
            })
        };
        Self::from_planes(width, eps, mean, std, min_pixels, max_pixels, None, load)
    }

    #[allow(clippy::too_many_arguments)]
    fn from_planes(
        width: usize,
        eps: f32,
        mean: [f32; 3],
        std: [f32; 3],
        min_pixels: u64,
        max_pixels: u64,
        mlx_patch: Option<Weight>,
        load: impl Fn(&str, &[usize], bool) -> Result<Weight>,
    ) -> Result<Self> {
        let geometry = TowerGeometry::for_decoder(width)?;
        let e = geometry.width;
        let f = geometry.ff;
        let mlx_bf16 = mlx_patch.is_some();
        let (patch0, patch1) = if let Some(patch) = mlx_patch {
            (patch, None)
        } else {
            let mut a = load("v.patch_embd.weight", &[16, 16, 3, e], true)?;
            let mut b = load("v.patch_embd.weight.1", &[16, 16, 3, e], true)?;
            a.k = PATCH;
            a.n = e;
            b.k = PATCH;
            b.n = e;
            (a, Some(b))
        };
        let mut blocks = Vec::new();
        for i in 0..geometry.layers {
            let w =
                |s: &str, dims: &[usize], half: bool| load(&format!("v.blk.{i}.{s}"), dims, half);
            blocks.push(Block {
                ln1: w("ln1.weight", &[e], false)?,
                ln1b: w("ln1.bias", &[e], false)?,
                qkv: w("attn_qkv.weight", &[e, e * 3], true)?,
                qkvb: w("attn_qkv.bias", &[e * 3], false)?,
                out: w("attn_out.weight", &[e, e], true)?,
                outb: w("attn_out.bias", &[e], false)?,
                ln2: w("ln2.weight", &[e], false)?,
                ln2b: w("ln2.bias", &[e], false)?,
                up: w("ffn_up.weight", &[e, f], true)?,
                upb: w("ffn_up.bias", &[f], false)?,
                down: w("ffn_down.weight", &[f, e], true)?,
                downb: w("ffn_down.bias", &[e], false)?,
            });
        }
        Ok(Self {
            geometry,
            mlx_bf16,
            patch0,
            patch1,
            blocks,
            eps,
            mean,
            std,
            bias: load("v.patch_embd.bias", &[e], false)?,
            pos: load("v.position_embd.weight", &[e, 2304], false)?,
            post: load("v.post_ln.weight", &[e], false)?,
            postb: load("v.post_ln.bias", &[e], false)?,
            mm0: load("mm.0.weight", &[e * 4, e * 4], true)?,
            mm0b: load("mm.0.bias", &[e * 4], false)?,
            mm2: load("mm.2.weight", &[e * 4, width], true)?,
            mm2b: load("mm.2.bias", &[width], false)?,
            budget: VisionBudget {
                min_pixels,
                max_pixels,
                max_edge: None,
                pixels_per_token: 1024,
                min_tokens: (min_pixels / 1024) as u32,
                max_tokens: (max_pixels / 1024) as u32,
            },
        })
    }

    fn mm<const RELAXED: bool, const HALF_INPUT: bool>(
        cmd: &Commands<'_>,
        w: &Weight,
        x: &Buffer,
        out: &Buffer,
        bias: &Weight,
        rows: usize,
        epilogue: u32,
    ) {
        let tile = if rows < 128 { 32 } else { 64 };
        cmd.dispatch(
            match (w.ty, tile) {
                (30, 32) => {
                    if RELAXED {
                        "vis_bmm_fast32"
                    } else {
                        "vis_bmm32"
                    }
                }
                (30, _) => {
                    if RELAXED {
                        "vis_bmm_fast64"
                    } else {
                        "vis_bmm64"
                    }
                }
                (_, 32) if HALF_INPUT => "vis_mm32",
                (_, _) if HALF_INPUT => "vis_mm64",
                (_, 32) => "vis_hmm32",
                _ => "vis_hmm64",
            },
            &[&w.buffer, x, out, &bias.buffer],
            &[w.k as u32, w.n as u32, rows as u32, epilogue],
            [w.n.div_ceil(64), rows.div_ceil(tile), 1],
            128,
        );
    }

    fn ln(
        &self,
        cmd: &Commands<'_>,
        x: &Buffer,
        w: &Weight,
        b: &Weight,
        stage: &Buffer,
        rows: usize,
    ) {
        cmd.dispatch(
            if self.mlx_bf16 {
                "qmlx_vis_norm"
            } else {
                "vis_ln"
            },
            &[x, &w.buffer, &b.buffer, stage],
            &[self.geometry.width as u32, self.eps.to_bits()],
            [rows, 1, 1],
            if self.mlx_bf16 {
                self.geometry.width.div_ceil(256) * 32
            } else {
                256
            },
        );
    }

    fn project(
        &self,
        cmd: &Commands<'_>,
        w: &Weight,
        x: &Buffer,
        out: &Buffer,
        bias: &Weight,
        rows: usize,
        epilogue: u32,
    ) {
        if self.mlx_bf16 {
            cmd.dispatch(
                "qmlx_vis_mm",
                &[&w.buffer, x, out, &bias.buffer],
                &[w.k as u32, w.n as u32, rows as u32, epilogue],
                [w.n.div_ceil(64), rows.div_ceil(64), 1],
                128,
            );
        } else {
            Self::mm::<true, false>(cmd, w, x, out, bias, rows, epilogue);
        }
    }

    /// Ragged batch: all rowwise projections share one weight walk; the tile
    /// descriptors bound attention to each image, with no padding to max size.
    pub(super) fn start(
        &self,
        device: &MetalDevice,
        images: &[(&[u8], usize, usize)],
    ) -> Result<Job> {
        if images.is_empty() {
            return Err(error("empty vision batch"));
        }
        let grids = images
            .iter()
            .map(|(rgb, w, h)| {
                let size = self.dimensions(*w, *h)?;
                if rgb.len() != w * h * 3 {
                    return Err(error("RGB byte count does not match dimensions"));
                }
                Ok((size.0 / 16, size.1 / 16))
            })
            .collect::<Result<Vec<_>>>()?;
        let rows: usize = grids.iter().map(|(w, h)| w * h).sum();
        let g = self.geometry;
        let e = g.width;
        if rows > 65_536 {
            return Err(MetalError::Memory(
                "vision wave exceeds 65536 patch rows; split admission wave".into(),
            ));
        }
        let mut xy = Vec::with_capacity(rows * 2);
        let mut tiles = Vec::new();
        let mut first = 0;
        for &(pw, ph) in &grids {
            let n = pw * ph;
            for row in 0..n {
                xy.extend([
                    ((row / 4 % (pw / 2)) * 2 + row % 2) as u32,
                    ((row / 4 / (pw / 2)) * 2 + row % 4 / 2) as u32,
                ]);
            }
            let query_tile = if self.mlx_bf16 { 64 } else { 32 };
            for r in (0..n).step_by(query_tile) {
                tiles.extend([
                    (first + r) as u32,
                    (n - r).min(query_tile) as u32,
                    first as u32,
                    n as u32,
                ]);
            }
            first += n;
        }
        let job = Job {
            rows,
            grids,
            x: device.alloc(rows * e * 4)?,
            stage: device.alloc(rows * g.ff.max(e) * 4)?,
            qkv: device.alloc(rows * e * 3 * 4)?,
            q: device.alloc((rows + 64) * g.padded_width() * 2)?,
            k: device.alloc((rows + 64) * g.padded_width() * 2)?,
            v: device.alloc((rows + 64) * g.padded_width() * 2)?,
            attn: device.alloc(rows * e * 4)?,
            xy: device.upload(&xy.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?,
            tiles: device.upload(
                &tiles
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?,
            tile_count: tiles.len() / 4,
            layer: 0,
            block_cost: std::time::Duration::ZERO,
            submits: 1,
            gpu_seconds: 0.0,
        };
        let temporal = self
            .patch1
            .as_ref()
            .map(|_| device.alloc(rows * e * 4))
            .transpose()?;
        let sources = images
            .iter()
            .zip(&job.grids)
            .map(|(&(rgb, w, h), &(pw, ph))| {
                if self.mlx_bf16 {
                    device.upload(&paddock_engine::pillow::resize_rgb8(
                        rgb,
                        w,
                        h,
                        pw * 16,
                        ph * 16,
                        paddock_engine::pillow::Filter::Bicubic,
                    ))
                } else if g.pillow_resize() {
                    device.upload(&paddock_engine::pillow::resize_pad_black(
                        rgb,
                        w,
                        h,
                        pw * 16,
                        ph * 16,
                    ))
                } else {
                    device.upload(rgb)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let cmd = device.begin()?;
        let mut first = 0;
        for (((_, w, h), &(pw, ph)), source) in images.iter().zip(&job.grids).zip(&sources) {
            let params = [
                if g.pillow_resize() {
                    (pw * 16) as u32
                } else {
                    *w as u32
                },
                if g.pillow_resize() {
                    (ph * 16) as u32
                } else {
                    *h as u32
                },
                (pw * 16) as u32,
                (ph * 16) as u32,
                first as u32,
                self.mean[0].to_bits(),
                self.mean[1].to_bits(),
                self.mean[2].to_bits(),
                self.std[0].to_bits(),
                self.std[1].to_bits(),
                self.std[2].to_bits(),
            ];
            cmd.dispatch(
                if self.mlx_bf16 {
                    "qmlx_vis_patches"
                } else {
                    "vis_patches"
                },
                &[source, &job.stage],
                &params,
                [
                    (pw * ph * if self.mlx_bf16 { 1536 } else { PATCH }).div_ceil(256),
                    1,
                    1,
                ],
                256,
            );
            first += pw * ph;
        }
        if let Some(patch1) = &self.patch1 {
            Self::mm::<true, true>(&cmd, &self.patch0, &job.stage, &job.x, &self.bias, rows, 0);
            Self::mm::<true, true>(
                &cmd,
                patch1,
                &job.stage,
                temporal
                    .as_ref()
                    .ok_or_else(|| error("missing temporal patch workspace"))?,
                &self.bias,
                rows,
                0,
            );
        } else {
            cmd.dispatch(
                "qmlx_vis_patch_mm",
                &[&self.patch0.buffer, &job.stage, &job.x],
                &[e as u32, rows as u32],
                [e.div_ceil(64), rows.div_ceil(64), 1],
                128,
            );
        }
        first = 0;
        for &(pw, ph) in &job.grids {
            cmd.dispatch(
                if self.mlx_bf16 {
                    "qmlx_vis_position"
                } else {
                    "vis_position"
                },
                &[
                    &job.x,
                    temporal.as_ref().unwrap_or(&job.x),
                    &self.bias.buffer,
                    &self.pos.buffer,
                ],
                &[pw as u32, ph as u32, first as u32, e as u32, 48],
                [(pw * ph * e).div_ceil(256), 1, 1],
                256,
            );
            first += pw * ph;
        }
        let mut job = job;
        job.gpu_seconds += cmd.finish()?;
        Ok(job)
    }

    #[cfg(test)]
    fn step(&self, device: &MetalDevice, job: &mut Job) -> Result<Option<Vec<Output>>> {
        // Tensor diagnostics inspect the final block before merger scratch
        // reuse. This also supplies the unfused GPU comparison for tests.
        self.step_blocks(device, job, 1, false)
    }

    pub(super) fn step_budget(
        &self,
        device: &MetalDevice,
        job: &mut Job,
        budget: std::time::Duration,
    ) -> Result<Option<Vec<Output>>> {
        let count = if job.block_cost.is_zero() {
            1
        } else {
            (budget.as_secs_f64() / (job.block_cost.as_secs_f64() * 1.1)).floor() as usize
        }
        .clamp(1, self.blocks.len());
        self.step_blocks(device, job, count, true)
    }

    fn step_blocks(
        &self,
        device: &MetalDevice,
        job: &mut Job,
        count: usize,
        finish_last_block: bool,
    ) -> Result<Option<Vec<Output>>> {
        let started = std::time::Instant::now();
        let cmd = device.begin()?;
        let rows = job.rows;
        let end = (job.layer + count).min(self.blocks.len());
        let attention_kernel = if self.mlx_bf16 {
            match self.geometry.heads {
                12 => "qmlx_vis_attention_12",
                _ => "qmlx_vis_attention_16",
            }
        } else {
            self.geometry.attention_kernel()
        };
        #[cfg(test)]
        let attention_kernel =
            if self.mlx_bf16 && std::env::var_os("PADDOCK_METAL_ATTENTION_CANDIDATE").is_some() {
                match self.geometry.heads {
                    12 => "qmlx_vis_attention_candidate_12",
                    _ => "qmlx_vis_attention_candidate_16",
                }
            } else {
                attention_kernel
            };
        for b in &self.blocks[job.layer..end] {
            self.ln(&cmd, &job.x, &b.ln1, &b.ln1b, &job.stage, rows);
            self.project(&cmd, &b.qkv, &job.stage, &job.qkv, &b.qkvb, rows, 1);
            cmd.dispatch(
                if self.mlx_bf16 {
                    if self.geometry.heads == 12 {
                        "qmlx_vis_qkv_12"
                    } else {
                        "qmlx_vis_qkv_16"
                    }
                } else {
                    self.geometry.qkv_kernel()
                },
                &[&job.qkv, &job.xy, &job.q, &job.k, &job.v],
                &[rows as u32],
                [
                    ((rows + 64) * self.geometry.padded_width()).div_ceil(256),
                    1,
                    1,
                ],
                256,
            );
            cmd.dispatch(
                attention_kernel,
                &[&job.q, &job.k, &job.v, &job.attn, &job.tiles],
                &[if self.mlx_bf16 { 31 } else { 0 }],
                [self.geometry.heads, job.tile_count, 1],
                if self.mlx_bf16 { 128 } else { 64 },
            );
            self.project(&cmd, &b.out, &job.attn, &job.x, &b.outb, rows, 2);
            self.ln(&cmd, &job.x, &b.ln2, &b.ln2b, &job.attn, rows);
            self.project(&cmd, &b.up, &job.attn, &job.stage, &b.upb, rows, 3);
            self.project(&cmd, &b.down, &job.stage, &job.x, &b.downb, rows, 2);
        }
        if job.layer < end && (end < self.blocks.len() || !finish_last_block) {
            job.gpu_seconds += cmd.finish()?;
            job.submits += 1;
            let cost = started.elapsed() / (end - job.layer) as u32;
            // Same-shape blocks use the latest completed cost. Keeping a
            // decaying high-water mark after a transient stall fragmented
            // a 27-block encoder into 21 submissions behind live decodes.
            // Admission still reserves 10% and checks the wall quantum.
            job.block_cost = cost;
            job.layer = end;
            return Ok(None);
        }
        // The merger follows the final block in the same command buffer.
        // Existing dispatch barriers preserve all dependencies; no scheduler
        // round is spent waiting merely to publish completed image features.
        self.ln(&cmd, &job.x, &self.post, &self.postb, &job.attn, rows);
        self.project(
            &cmd,
            &self.mm0,
            &job.attn,
            &job.stage,
            &self.mm0b,
            rows / 4,
            if self.mlx_bf16 { 4 } else { 3 },
        );
        let merged = device.alloc(rows / 4 * self.mm2.n * 4)?;
        self.project(
            &cmd,
            &self.mm2,
            &job.stage,
            &merged,
            &self.mm2b,
            rows / 4,
            1,
        );
        let bad = device.upload(&0u32.to_le_bytes())?;
        cmd.dispatch(
            "vis_finite",
            &[&merged, &bad],
            &[(rows / 4 * self.mm2.n) as u32],
            [(rows / 4 * self.mm2.n).div_ceil(256), 1, 1],
            256,
        );
        let mut outputs = Vec::new();
        let mut first = 0;
        for &(pw, ph) in &job.grids {
            let count = pw * ph / 4 * self.mm2.n;
            let embd = device.alloc(count * 4)?;
            cmd.dispatch(
                "spec_copy",
                &[&merged, &embd],
                &[first as u32, 0, count as u32],
                [count.div_ceil(256), 1, 1],
                256,
            );
            first += count;
            outputs.push(Output {
                embd,
                nx: pw / 2,
                ny: ph / 2,
            });
        }
        job.gpu_seconds += cmd.finish()?;
        job.submits += 1;
        job.layer = end;
        // SAFETY: output validation completed with the merger submission.
        if unsafe { bad.read_u32(1)[0] } != 0 {
            return Err(error("vision tower produced nonfinite image embeddings"));
        }
        Ok(Some(outputs))
    }
}
