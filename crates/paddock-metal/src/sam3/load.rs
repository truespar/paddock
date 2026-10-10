use super::*;
use paddock_models::safetensors::{SafetensorsFile, StDtype};
#[cfg(test)]
#[path = "load_tests.rs"]
mod tests;

const ROOT: &str = "detector_model.vision_encoder.backbone";
pub(super) struct Reader<'a> {
    pub(super) d: &'a MetalDevice,
    pub(super) st: &'a SafetensorsFile,
}
impl Reader<'_> {
    fn checked_bytes(&self, name: &str, shape: &[usize]) -> Result<&[u8]> {
        let (t, b) = self
            .st
            .bytes(name)
            .ok_or_else(|| error(format!("SAM 3 missing {name}")))?;
        if t.dtype != StDtype::F32 || t.shape != shape {
            return Err(error(format!(
                "SAM 3 {name}: expected F32 {shape:?}, got {:?} {:?}",
                t.dtype, t.shape
            )));
        }
        if b.as_chunks::<4>()
            .0
            .iter()
            .any(|x| !f32::from_le_bytes(*x).is_finite())
        {
            return Err(error(format!("SAM 3 nonfinite {name}")));
        }
        Ok(b)
    }
    pub(super) fn values(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        Ok(self
            .checked_bytes(name, shape)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect())
    }
    pub(super) fn f32_buffer(&self, name: &str, shape: &[usize]) -> Result<Buffer> {
        // A 193 MiB token table is already mapped in the checkpoint. Validate
        // it there and upload once, without two additional host-sized copies.
        self.d.upload(self.checked_bytes(name, shape)?)
    }
    pub(super) fn vector(&self, name: &str, n: usize) -> Result<Buffer> {
        self.f32_buffer(name, &[n])
    }
    pub(super) fn norm(&self, prefix: &str, n: usize) -> Result<Norm> {
        Ok(Norm {
            w: self.vector(&format!("{prefix}.weight"), n)?,
            b: self.vector(&format!("{prefix}.bias"), n)?,
        })
    }
    pub(super) fn matrix_values(&self, v: &[f32], k: usize, n: usize) -> Result<Matrix> {
        if k == 0 || n == 0 || k.checked_mul(n) != Some(v.len()) || !k.is_multiple_of(8) {
            return Err(error("SAM 3 invalid matrix dimensions"));
        }
        // Weight conversion/overflow validation stays on the GPU. Permutations
        // below only rearrange checkpoint elements, never perform model math.
        let w = self.d.alloc(v.len() * 2)?;
        let bad = self.d.upload(&0u32.to_le_bytes())?;
        // Bound temporary GPU residency to 256 KiB + the error word. Loading
        // a late text MLP must fit the admitted model, not require a second
        // full FP32 copy of its largest matrix. Each chunk is fenced before
        // releasing staging, and the error flag is sticky across chunks.
        const CHUNK: usize = 65536;
        for (i, values) in v.chunks(CHUNK).enumerate() {
            let source = upload(self.d, values)?;
            let c = self.d.begin()?;
            c.dispatch_at(
                "vis_cast",
                &[&source, &w, &bad],
                &[0, i * CHUNK * 2, 0],
                &[values.len() as u32, 0, 1],
                [values.len().div_ceil(256), 1, 1],
                256,
            );
            c.finish()?;
        }
        if unsafe { bad.read_u32(1)[0] } != 0 {
            return Err(error("SAM 3 weight exceeds finite F16 range"));
        }
        Ok(Matrix { w, k, n })
    }
    pub(super) fn conv(&self, name: &str, shape: &[usize]) -> Result<Conv> {
        let values = self.values(&format!("{name}.weight"), shape)?;
        let n = shape[0];
        let k = values.len() / n;
        Ok(Conv {
            w: self.matrix_values(&values, k, n)?,
            b: self.vector(&format!("{name}.bias"), n)?,
        })
    }
    fn conv3(&self, name: &str) -> Result<Conv> {
        let w = self.values(&format!("{name}.weight"), &[256, 256, 3, 3])?;
        let mut v = vec![0.; w.len()];
        for out in 0..256 {
            for input in 0..256 {
                for tap in 0..9 {
                    v[(out * 9 + tap) * 256 + input] = w[(out * 256 + input) * 9 + tap];
                }
            }
        }
        Ok(Conv {
            w: self.matrix_values(&v, 9 * 256, 256)?,
            b: self.vector(&format!("{name}.bias"), 256)?,
        })
    }
    fn convt(&self, name: &str, cin: usize, cout: usize) -> Result<Conv> {
        let w = self.values(&format!("{name}.weight"), &[cin, cout, 2, 2])?;
        let mut v = vec![0.; w.len()];
        for i in 0..cin {
            for o in 0..cout {
                for t in 0..4 {
                    v[(t * cout + o) * cin + i] = w[(i * cout + o) * 4 + t];
                }
            }
        }
        Ok(Conv {
            w: self.matrix_values(&v, cin, 4 * cout)?,
            b: self.vector(&format!("{name}.bias"), cout)?,
        })
    }
    fn neck(&self, root: &str) -> Result<Neck> {
        let name = |i, s| format!("{root}.fpn_layers.{i}.{s}");
        Ok(Neck {
            up4: [
                self.convt(&name(0, "scale_layers.0"), 1024, 512)?,
                self.convt(&name(0, "scale_layers.2"), 512, 256)?,
            ],
            up2: self.convt(&name(1, "scale_layers.0"), 1024, 512)?,
            proj1: [
                self.conv(&name(0, "proj1"), &[256, 256, 1, 1])?,
                self.conv(&name(1, "proj1"), &[256, 512, 1, 1])?,
                self.conv(&name(2, "proj1"), &[256, 1024, 1, 1])?,
            ],
            proj2: [
                self.conv3(&name(0, "proj2"))?,
                self.conv3(&name(1, "proj2"))?,
                self.conv3(&name(2, "proj2"))?,
            ],
        })
    }
}

// Shapes/positions only. Same F32 table-building operations and rotate-half
// coordinate order as CUDA's RopeTable; compare the uploaded bits in fixtures.
fn rope(d: &MetalDevice, global: bool) -> Result<Buffer> {
    let rows = if global { 5184 } else { 576 };
    let mut values = Vec::with_capacity(rows * 64);
    let f: Vec<_> = (0..16)
        .map(|i| 1f32 / 10000f32.powf((4 * i) as f32 / 64.))
        .collect();
    for row in 0..rows {
        let (x, y) = if global {
            (
                (row / 576 % 3 * 24 + row % 24) as f32 * (1. / 3.),
                (row / 576 / 3 * 24 + row % 576 / 24) as f32 * (1. / 3.),
            )
        } else {
            ((row % 24) as f32, (row / 24) as f32)
        };
        for pos in [x, y] {
            for freq in &f {
                let a = pos * freq;
                values.extend([a.cos(), a.sin()]);
            }
        }
    }
    upload(d, &values)
}
pub(super) fn tiles(d: &MetalDevice, global: bool) -> Result<Buffer> {
    let group = if global { 5184 } else { 576 };
    let mut out = Vec::<u32>::new();
    for first in (0..5184).step_by(group) {
        for row in (first..first + group).step_by(32) {
            out.extend([row as u32, 32, first as u32, group as u32]);
        }
    }
    d.upload(&out.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>())
}

impl Workspace {
    pub(super) fn new(d: &MetalDevice) -> Result<Self> {
        let alloc = |n| d.alloc(n);
        let levels = || -> Result<[Buffer; 3]> {
            Ok([
                alloc(288 * 288 * 256 * 4)?,
                alloc(144 * 144 * 256 * 4)?,
                alloc(72 * 72 * 256 * 4)?,
            ])
        };
        Ok(Self {
            pixels: alloc(1008 * 1008 * 3)?,
            patches: alloc(5184 * 592 * 2)?,
            x: alloc(5184 * 1024 * 4)?,
            norm: alloc(5184 * 1024 * 2)?,
            projection: alloc(5184 * 1024 * 2)?,
            qkv: alloc(5184 * 3072 * 4)?,
            q: alloc(5184 * 1024 * 2)?,
            k: alloc(5184 * 1024 * 2)?,
            v: alloc(5184 * 1024 * 2)?,
            attention: alloc(5184 * 1024 * 2)?,
            wide: alloc(5184 * 4736 * 2)?,
            raster: alloc(5184 * 1024 * 2)?,
            temp: alloc(144 * 144 * 1024 * 4)?,
            half_a: alloc(144 * 144 * 512 * 2)?,
            half_b: alloc(288 * 288 * 256 * 2)?,
            det: levels()?,
            trk: levels()?,
            s0: alloc(288 * 288 * 32 * 4)?,
            s1: alloc(144 * 144 * 64 * 4)?,
        })
    }
}
impl Sam3Vision {
    pub fn load(dir: &Path, budget: Option<u64>) -> Result<Self> {
        let cfg = Sam3VisionConfig::read(dir).map_err(|e| error(e.to_string()))?;
        if (
            cfg.image_size,
            cfg.patch,
            cfg.hidden,
            cfg.n_layer,
            cfg.n_heads,
            cfg.intermediate,
            cfg.window,
            cfg.fpn_dim,
        ) != (1008, 14, 1024, 32, 16, 4736, 24, 256)
            || cfg.global_blocks != [7, 15, 23, 31]
            || cfg.rope_theta != 10000.
        {
            return Err(error(
                "SAM 3 Metal image encoder requires the official 1008/ViT-L geometry",
            ));
        }
        let st = SafetensorsFile::open(&dir.join("model.safetensors"))
            .map_err(|e| error(e.to_string()))?;
        let device = MetalDevice::new_planned(budget, Self::resident_bytes_required())?;
        let r = Reader {
            d: &device,
            st: &st,
        };
        let pw = r.values(
            &format!("{ROOT}.embeddings.patch_embeddings.projection.weight"),
            &[1024, 3, 14, 14],
        )?;
        if st
            .bytes(&format!(
                "{ROOT}.embeddings.patch_embeddings.projection.bias"
            ))
            .is_some()
        {
            return Err(error("SAM 3 patch bias is unsupported"));
        }
        let mut padded = vec![0.; 1024 * 592];
        for i in 0..1024 {
            padded[i * 592..i * 592 + 588].copy_from_slice(&pw[i * 588..(i + 1) * 588]);
        }
        let patch = r.matrix_values(&padded, 592, 1024)?;
        let pos = upload(
            &device,
            &r.values(
                &format!("{ROOT}.embeddings.position_embeddings"),
                &[1, 576, 1024],
            )?,
        )?;
        let norm = r.norm(&format!("{ROOT}.layer_norm"), 1024)?;
        let mut blocks = Vec::new();
        for i in 0..32 {
            let name = |s: &str| format!("{ROOT}.layers.{i}.{s}");
            let mut weights = Vec::new();
            let mut biases = Vec::new();
            for axis in ["q", "k", "v"] {
                let w = r.values(
                    &name(&format!("attention.{axis}_proj.weight")),
                    &[1024, 1024],
                )?;
                let b = r.values(&name(&format!("attention.{axis}_proj.bias")), &[1024])?;
                for row in 0..1024 {
                    let from = if axis == "v" {
                        row
                    } else {
                        row / 64 * 64 + (row % 32) * 2 + (row % 64) / 32
                    };
                    weights.extend_from_slice(&w[from * 1024..(from + 1) * 1024]);
                    biases.push(b[from]);
                }
            }
            blocks.push(Block {
                n1: r.norm(&name("layer_norm1"), 1024)?,
                n2: r.norm(&name("layer_norm2"), 1024)?,
                qkv: Conv {
                    w: r.matrix_values(&weights, 1024, 3072)?,
                    b: upload(&device, &biases)?,
                },
                out: r.conv(&name("attention.o_proj"), &[1024, 1024])?,
                up: r.conv(&name("mlp.fc1"), &[4736, 1024])?,
                down: r.conv(&name("mlp.fc2"), &[1024, 4736])?,
            });
        }
        let det = r.neck("detector_model.vision_encoder.neck")?;
        let trk = r.neck("tracker_neck")?;
        let s0 = r.conv("tracker_model.mask_decoder.conv_s0", &[32, 256, 1, 1])?;
        let s1 = r.conv("tracker_model.mask_decoder.conv_s1", &[64, 256, 1, 1])?;
        let rope = [rope(&device, false)?, rope(&device, true)?];
        let tiles = [tiles(&device, false)?, tiles(&device, true)?];
        let weight_bytes = device.allocated_bytes();
        if weight_bytes != Self::planned_weight_bytes() {
            return Err(error(format!(
                "SAM 3 weight reservation drift: planned {}, loaded {weight_bytes}",
                Self::planned_weight_bytes()
            )));
        }
        let ws = Workspace::new(&device)?;
        let workspace_bytes = device.allocated_bytes() - weight_bytes;
        if workspace_bytes != Workspace::required_bytes() {
            return Err(error("SAM 3 workspace reservation drift"));
        }
        Ok(Self {
            device,
            cfg,
            patch,
            pos,
            norm,
            blocks,
            det,
            trk,
            s0,
            s1,
            rope,
            tiles,
            ws,
            weight_bytes,
            workspace_bytes,
            encoded: false,
            tracker: false,
            input: Default::default(),
        })
    }
}
