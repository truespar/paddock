//! The 768-wide Gemma 4 picture tower. Resize, patches, axial positions,
//! split-query online attention, pooling and projection stay on Metal.
//! It is NOT the 1152-wide chat tower: there is no standardization tail.
use super::*;
use paddock_engine::encoder::embedding_gemma2::image_target;
use paddock_models::{gguf::Value, mapped::MappedGguf};

const E: usize = 768;
const F: usize = 3072;
struct Block {
    pre: Weight,
    qkv: Weight,
    qn: Weight,
    kn: Weight,
    out: Weight,
    post: Weight,
    ff_pre: Weight,
    gate_up: Weight,
    down: Weight,
    ff_post: Weight,
}
pub(super) struct Vision {
    mlx: bool,
    patch: Weight,
    pos: Weight,
    projection: Weight,
    blocks: Vec<Block>,
    resize: crate::clef::ImageResizeCache,
    pub(super) budget: usize,
}

/// GGUF tensor conversion is GPU-only, checked before publishing weights.
/// Matrix storage remains BF16; small vectors/convolutions retain F32.
pub(super) fn weight(
    d: &MetalDevice,
    m: &MappedGguf,
    name: &str,
    shape: &[usize],
    ty: u32,
) -> Result<Weight> {
    let source = Weight::load(d, m, name, shape)?;
    if !matches!(source.ty, 0 | 1 | 30) {
        return Err(error(format!("{name}: unsupported media weight type")));
    }
    let n: usize = shape.iter().product();
    let out = d.alloc(n * if ty == 0 { 4 } else { 2 })?;
    let bad = d.upload(&0u32.to_le_bytes())?;
    let c = d.begin()?;
    c.dispatch(
        "vis_cast",
        &[&source.buffer, &out, &bad],
        &[n as u32, source.ty, ty],
        [n.div_ceil(256), 1, 1],
        256,
    );
    c.finish()?;
    if unsafe { bad.read_u32(1)[0] } != 0 {
        return Err(error(format!("{name}: nonfinite media weight")));
    }
    Ok(Weight {
        buffer: out,
        ty,
        k: shape[0],
        n: *shape.get(1).unwrap_or(&1),
    })
}
pub(super) fn mm(c: &crate::device::Commands<'_>, w: &Weight, x: &Buffer, y: &Buffer, rows: usize) {
    c.dispatch(
        if w.ty == 30 {
            "vis_bmm_fast64"
        } else {
            "gv_patch_project"
        },
        &[&w.buffer, x, y, &w.buffer],
        &[w.k as u32, w.n as u32, rows as u32, 0],
        [w.n.div_ceil(64), rows.div_ceil(64), 1],
        128,
    );
}
pub(super) fn point(
    c: &crate::device::Commands<'_>,
    name: &str,
    b: &[&Buffer],
    p: &[u32],
    n: usize,
) {
    c.dispatch(name, b, p, [n.div_ceil(256), 1, 1], 256);
}
pub(super) fn norm(
    c: &crate::device::Commands<'_>,
    x: &Buffer,
    w: &Weight,
    y: &Buffer,
    width: usize,
    rows: usize,
    weighted: bool,
) {
    c.dispatch(
        "gv_rms",
        &[x, &w.buffer, y],
        &[width as u32, 1e-6f32.to_bits(), weighted as u32],
        [rows, 1, 1],
        256,
    );
}
impl Vision {
    pub(super) fn has_workspace(&self) -> bool {
        !self.resize.is_empty()
    }
    pub(super) fn load(d: &MetalDevice, m: &MappedGguf, budget: usize) -> Result<Self> {
        let u = |k| m.gguf().metadata.get(k).and_then(Value::as_u64);
        if u("clip.vision.embedding_length") != Some(E as u64)
            || u("clip.vision.feed_forward_length") != Some(F as u64)
            || u("clip.vision.block_count") != Some(16)
            || u("clip.vision.attention.head_count") != Some(12)
            || u("clip.vision.patch_size") != Some(16)
            || u("clip.vision.projection_dim") != Some(WIDTH as u64)
            || m.gguf()
                .tensors
                .iter()
                .any(|t| t.name.starts_with("v.std_") || t.name.ends_with(".input_min"))
        {
            return Err(error(
                "EmbeddingGemma 2 requires its 16-layer 768-wide gemma4v companion",
            ));
        }
        Self::from_weights(d, budget, false, |name, shape, ty| {
            weight(d, m, name, shape, ty)
        })
    }
    pub(super) fn from_weights(
        d: &MetalDevice,
        budget: usize,
        mlx: bool,
        load: impl Fn(&str, &[usize], u32) -> Result<Weight>,
    ) -> Result<Self> {
        let mut patch = load("v.patch_embd.weight", &[16, 16, 3, E], 0)?;
        patch.k = E;
        patch.n = E;
        let join = |weights: Vec<Weight>| -> Result<Weight> {
            let k = weights[0].k;
            let n = weights.iter().map(|w| w.n).sum::<usize>();
            if weights.iter().any(|w| w.ty != 30 || w.k != k) {
                return Err(error("vision projection bundle must be BF16"));
            }
            let buffer = d.alloc(k * n * 2)?;
            let c = d.begin()?;
            let mut at = 0;
            for w in &weights {
                let count = w.k * w.n / 2;
                point(
                    &c,
                    "spec_copy",
                    &[&w.buffer, &buffer],
                    &[0, at as u32, count as u32],
                    count,
                );
                // Keep the original planes alive until the copy fence.
                at += count;
            }
            c.finish()?;
            Ok(Weight {
                buffer,
                ty: 30,
                k,
                n,
            })
        };
        let mut blocks = Vec::new();
        for i in 0..16 {
            let w =
                |s: &str, shape: &[usize], ty| load(&format!("v.blk.{i}.{s}.weight"), shape, ty);
            blocks.push(Block {
                pre: w("ln1", &[E], 0)?,
                qkv: join(vec![
                    w("attn_q", &[E, E], 30)?,
                    w("attn_k", &[E, E], 30)?,
                    w("attn_v", &[E, E], 30)?,
                ])?,
                qn: w("attn_q_norm", &[64], 0)?,
                kn: w("attn_k_norm", &[64], 0)?,
                out: w("attn_out", &[E, E], 30)?,
                post: w("attn_post_norm", &[E], 0)?,
                ff_pre: w("ln2", &[E], 0)?,
                gate_up: join(vec![w("ffn_gate", &[E, F], 30)?, w("ffn_up", &[E, F], 30)?])?,
                down: w("ffn_down", &[F, E], 30)?,
                ff_post: w("ffn_post_norm", &[E], 0)?,
            });
        }
        Ok(Self {
            mlx,
            patch,
            blocks,
            budget,
            resize: Default::default(),
            pos: load("v.position_embd.weight", &[E, 10240, 2], 0)?,
            projection: load("mm.input_projection.weight", &[E, WIDTH], 30)?,
        })
    }
    pub(super) fn reclaim(&mut self) {
        self.resize = Default::default();
    }
    pub(super) fn encode_budget(
        &mut self,
        d: &MetalDevice,
        rgb: &[u8],
        w: usize,
        h: usize,
        budget: Option<usize>,
    ) -> Result<Buffer> {
        let (tw, th) = image_target(w, h, budget.unwrap_or(self.budget)).map_err(error)?;
        let pixels = self.resize.pixels_raw(d, rgb, w, h, (th, tw), false)?;
        let (pw, ph) = (tw / 16, th / 16);
        let rows = pw * ph;
        let a = |n| d.alloc(rows * n * 4);
        let (x, stage, qkv, attn, delta, gate_up) =
            (a(E)?, a(F)?, a(3 * E)?, a(E)?, a(E)?, a(2 * F)?);
        let (qh, kh, vh) = (
            d.alloc(rows * E * 2)?,
            d.alloc(rows * E * 2)?,
            d.alloc(rows * E * 2)?,
        );
        let tiles = (0..rows)
            .step_by(32)
            .flat_map(|i| [i as u32, (rows - i).min(32) as u32, 0, rows as u32])
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        let tiles = d.upload(&tiles)?;
        let output = d.alloc(rows / 9 * WIDTH * 4)?;
        let c = d.begin()?;
        point(
            &c,
            if self.mlx {
                "eg2v_mlx_patches"
            } else {
                "eg2v_patches"
            },
            &[&pixels, &stage],
            &[tw as u32, th as u32],
            rows * E,
        );
        let mm =
            |c: &crate::device::Commands<'_>, w: &Weight, x: &Buffer, y: &Buffer, rows: usize| {
                if self.mlx {
                    c.dispatch(
                        "gmlx_vmm64",
                        &[&w.buffer, x, y, &w.buffer],
                        &[w.k as u32, w.n as u32, rows as u32, 0],
                        [w.n.div_ceil(64), rows.div_ceil(64), 1],
                        128,
                    );
                } else {
                    mm(c, w, x, y, rows);
                }
            };
        let norm = |c: &crate::device::Commands<'_>,
                    x: &Buffer,
                    w: &Weight,
                    y: &Buffer,
                    width: usize,
                    rows: usize,
                    weighted: bool| {
            if self.mlx {
                c.dispatch(
                    "gmlx_norm",
                    &[x, &w.buffer, y],
                    &[
                        width as u32,
                        if weighted { 0 } else { 2 },
                        if self.mlx { 1e-6f32.to_bits() } else { 0 },
                    ],
                    [rows, 1, 1],
                    width.div_ceil(128) * 32,
                );
            } else {
                norm(c, x, w, y, width, rows, weighted);
            }
        };
        mm(&c, &self.patch, &stage, &x, rows);
        point(
            &c,
            if self.mlx {
                "eg2v_mlx_position"
            } else {
                "eg2v_position"
            },
            &[&x, &self.pos.buffer],
            &[pw as u32, ph as u32],
            rows * E,
        );
        for b in &self.blocks {
            norm(&c, &x, &b.pre, &stage, E, rows, true);
            mm(&c, &b.qkv, &stage, &qkv, rows);
            c.dispatch_at(
                if self.mlx { "eg2v_mlx_qkv" } else { "eg2v_qkv" },
                &[&qkv, &qkv, &qkv, &b.qn.buffer, &b.kn.buffer, &qh, &kh, &vh],
                &[0, E * 4, E * 8, 0, 0, 0, 0, 0],
                &[pw as u32, 3 * E as u32],
                [12, rows, 1],
                32,
            );
            c.dispatch(
                if self.mlx {
                    "eg2v_mlx_attention"
                } else {
                    "eg2v_attention"
                },
                &[&qh, &kh, &vh, &attn, &tiles],
                &[if self.mlx { 31 } else { 0 }],
                [12, rows.div_ceil(32), 1],
                64,
            );
            mm(&c, &b.out, &attn, &delta, rows);
            let sandwich = |post: &Weight, pre: &Weight| {
                c.dispatch(
                    if self.mlx {
                        "gmlx_sandwich"
                    } else {
                        "gemma_sandwich"
                    },
                    &[&x, &delta, &post.buffer, &pre.buffer, &stage],
                    &[
                        E as u32,
                        0,
                        if self.mlx { 1e-6f32.to_bits() } else { 0 },
                        1e-6f32.to_bits(),
                        1f32.to_bits(),
                    ],
                    [rows, 1, 1],
                    if self.mlx { E.div_ceil(128) * 32 } else { 256 },
                )
            };
            sandwich(&b.post, &b.ff_pre);
            mm(&c, &b.gate_up, &stage, &gate_up, rows);
            point(
                &c,
                "eg2v_gate_up",
                &[&gate_up, &stage],
                &[(rows * F) as u32, self.mlx as u32],
                rows * F,
            );
            mm(&c, &b.down, &stage, &delta, rows);
            sandwich(&b.ff_post, &b.pre);
        }
        point(
            &c,
            if self.mlx {
                "eg2v_mlx_pool"
            } else {
                "eg2v_pool"
            },
            &[&x, &attn],
            &[pw as u32, ph as u32],
            rows / 9 * E,
        );
        norm(&c, &attn, &self.pos, &stage, E, rows / 9, false);
        mm(&c, &self.projection, &stage, &output, rows / 9);
        c.finish()?;
        Ok(output)
    }
}
