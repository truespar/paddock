//! Clef's own image contract, not the chat GGUF processor. Native BF16 tower
//! weights and F32 activations; bounded online attention, no score matrix.
use super::*;
use paddock_engine::clef_decision::ClefImage;
use paddock_models::{
    clef::ClefVisionConfig,
    safetensors::{ShardedSafetensors, StDtype},
};
pub(super) mod resize;
const E: usize = 1152;
const W: usize = 4608;
const P: usize = 1536;

struct Block {
    n1: Norm,
    qkv: Linear,
    out: Linear,
    n2: Norm,
    up: Linear,
    down: Linear,
}
pub(super) struct Vision {
    pub(super) mlx: bool,
    resize: std::cell::RefCell<resize::Cache>,
    cfg: ClefVisionConfig,
    patch: Linear,
    pos: Buffer,
    blocks: Vec<Block>,
    norm: Norm,
    up: Linear,
    down: Linear,
    rope: Buffer,
}
/// Conservative peak: one image, at most four patches per decision token,
/// plus bounded decoded/resize staging and output retained for all images.
pub(super) fn workspace_bytes(width: usize) -> u64 {
    (MAX_ROWS * 4 * (E + P + W + 3 * P) * 4 + MAX_ROWS * width * 4 + (576 << 20)) as u64
}
impl Vision {
    pub fn load(
        d: &MetalDevice,
        s: &ShardedSafetensors,
        cfg: &ClefVisionConfig,
        mlx: bool,
    ) -> Result<Self> {
        if (
            cfg.depth,
            cfg.hidden,
            cfg.heads,
            cfg.ffn,
            cfg.patch,
            cfg.temporal_patch,
            cfg.merge,
            cfg.pos_side,
        ) != (27, E, 16, 4304, 16, 2, 2, 48)
        {
            return Err(error("unsupported Clef vision geometry"));
        }
        let prefix = if mlx { "vision_tower" } else { "model.visual" };
        let raw = |name: &str, shape: &[usize]| -> Result<&[u8]> {
            let key = format!("{prefix}.{name}");
            let (t, b) = s
                .bytes(&key)
                .ok_or_else(|| error(format!("missing {key}")))?;
            if t.dtype != StDtype::Bf16 || t.shape != shape {
                return Err(error(format!(
                    "{key}: expected BF16 {shape:?}, got {:?} {:?}",
                    t.dtype, t.shape
                )));
            }
            Ok(b)
        };
        let vector = |name: &str, n| -> Result<Buffer> {
            let values = raw(name, &[n])?
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16))
                .collect::<Vec<_>>();
            if values.iter().any(|v| !v.is_finite()) {
                return Err(error("nonfinite vision vector"));
            }
            upload(d, &values)
        };
        let linear = |name: &str, k, n| -> Result<Linear> {
            Ok(Linear {
                weight: d.upload(raw(&format!("{name}.weight"), &[n, k])?)?.into(),
                bias: Some(vector(&format!("{name}.bias"), n)?),
                k,
                n,
            })
        };
        let norm = |name: &str| -> Result<Norm> {
            Ok(Norm {
                weight: vector(&format!("{name}.weight"), E)?,
                bias: Some(vector(&format!("{name}.bias"), E)?),
                width: E,
                eps: 1e-6,
            })
        };
        let shape = if mlx {
            vec![E, 2, 16, 16, 3]
        } else {
            vec![E, 3, 2, 16, 16]
        };
        let original = d.upload(raw("patch_embed.proj.weight", &shape)?)?;
        let patch_weight = if mlx {
            let out = d.alloc(E * P * 2)?;
            let cmd = d.begin()?;
            point(
                &cmd,
                "clef_vis_patch_weight",
                &[&original, &out],
                &[(E * P) as u32],
                E * P,
            );
            cmd.finish()?;
            out
        } else {
            original
        };
        let patch = Linear {
            weight: patch_weight.into(),
            bias: Some(vector("patch_embed.proj.bias", E)?),
            k: P,
            n: E,
        };
        let pos = d.upload(raw("pos_embed.weight", &[2304, E])?)?;
        let mut blocks = Vec::new();
        for i in 0..27 {
            let n = |s: &str| format!("blocks.{i}.{s}");
            blocks.push(Block {
                n1: norm(&n("norm1"))?,
                qkv: linear(&n("attn.qkv"), E, 3 * E)?,
                out: linear(&n("attn.proj"), E, E)?,
                n2: norm(&n("norm2"))?,
                up: linear(&n("mlp.linear_fc1"), E, 4304)?,
                down: linear(&n("mlp.linear_fc2"), 4304, E)?,
            });
        }
        let mut angles = Vec::with_capacity(4096 * 36);
        for pos in 0..4096 {
            for pair in 0..18 {
                let freq = 1f32 / 10000f64.powf(f64::from(pair as f32 / 18f32)) as f32;
                let a = f64::from(pos as f32 * freq);
                angles.extend([a.cos() as f32, a.sin() as f32]);
            }
        }
        Ok(Self {
            mlx,
            resize: Default::default(),
            cfg: cfg.clone(),
            patch,
            pos,
            blocks,
            norm: norm("merger.norm")?,
            up: linear("merger.linear_fc1", W, W)?,
            down: linear("merger.linear_fc2", W, cfg.out_hidden)?,
            rope: upload(d, &angles)?,
        })
    }
    pub fn encode(&self, d: &MetalDevice, im: &ClefImage) -> Result<Buffer> {
        let (h, w) = im.resized;
        let rows = (h / 16) * (w / 16);
        if rows > MAX_ROWS * 4 || h.max(w) / 16 >= 4096 {
            return Err(error("image exceeds vision workspace"));
        }
        let x = d.alloc(rows * E * 4)?;
        let stage = d.alloc(rows * P * 4)?;
        let wide = d.alloc(rows * W * 4)?;
        let qkv = d.alloc(rows * 3 * P * 4)?;
        let out = d.alloc(rows / 4 * self.cfg.out_hidden * 4)?;
        let pixels = self.resize.borrow_mut().pixels(d, im, self.mlx)?;
        let mut info = Vec::<u32>::new();
        for y in 0..h / 32 {
            for x in 0..w / 32 {
                for dy in 0..2 {
                    for dx in 0..2 {
                        info.extend([
                            (2 * y + dy) as u32,
                            (2 * x + dx) as u32,
                            (h / 16) as u32,
                            (w / 16) as u32,
                        ]);
                    }
                }
            }
        }
        let info = d.upload(
            &info
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let mut tiles = Vec::<u32>::new();
        for i in (0..rows).step_by(16) {
            tiles.extend([i as u32, (rows - i).min(16) as u32, 0, rows as u32]);
        }
        let tiles = d.upload(
            &tiles
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let cmd = d.begin()?;
        point(
            &cmd,
            "clef_vis_patches",
            &[&pixels, &wide],
            &[w as u32, h as u32],
            rows * P,
        );
        self.patch.run(&cmd, &wide, &x, rows, 0);
        point(
            &cmd,
            "clef_vis_position",
            &[&x, &self.pos, &info],
            &[rows as u32],
            rows * E,
        );
        cmd.finish()?;
        // Block-level submissions bound command lifetime and permit other
        // runners to make progress between image-transformer blocks.
        for b in &self.blocks {
            let cmd = d.begin()?;
            b.n1.run(&cmd, &x, &stage, rows);
            b.qkv.run(&cmd, &stage, &wide, rows, 0);
            point(
                &cmd,
                "clef_vis_qkv",
                &[&wide, &info, &self.rope, &qkv],
                &[rows as u32],
                rows * P,
            );
            cmd.dispatch(
                "clef_vision_attention",
                &[&qkv, &qkv, &qkv, &stage, &tiles],
                &[
                    16,
                    16,
                    P as u32,
                    P as u32,
                    P as u32,
                    0,
                    (rows * P) as u32,
                    (2 * rows * P) as u32,
                ],
                [16, rows.div_ceil(16), 1],
                128,
            );
            point(
                &cmd,
                "clef_vis_unpad",
                &[&stage, &wide],
                &[rows as u32],
                rows * E,
            );
            b.out.run(&cmd, &wide, &x, rows, 1);
            b.n2.run(&cmd, &x, &stage, rows);
            b.up.run(&cmd, &stage, &wide, rows, 4);
            b.down.run(&cmd, &wide, &x, rows, 1);
            cmd.finish()?;
        }
        let cmd = d.begin()?;
        self.norm.run(&cmd, &x, &stage, rows);
        self.up.run(&cmd, &stage, &wide, rows / 4, 2);
        self.down.run(&cmd, &wide, &out, rows / 4, 0);
        cmd.finish()?;
        Ok(out)
    }
}
