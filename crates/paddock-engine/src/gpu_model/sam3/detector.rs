//! SAM 3's detector: the concept-prompt path from image features + prompt
//! tokens to scored boxes and masks (Meta's `Sam3Image.forward_grounding`).
//!
//!   geometry encoder: [boxes..., CLS] -> geometry tokens        (fusion.rs)
//!   prompt = [valid text tokens, geometry tokens]
//!   fusion encoder: 5184 image tokens x prompt -> memory E      (fusion.rs)
//!   DETR decoder: 200 queries + presence over E and the prompt  (decoder.rs)
//!   scorer + segmentation head -> logits, boxes, masks           (decoder.rs, seg.rs)
//!
//! Every attention over the prompt reads only its VALID tokens - the text
//! tokens SOT..EOT and every geometry token - compacted into one plane. That
//! is exactly what Meta's key-padding mask computes (masked keys are
//! -inf before the softmax), with no mask to carry.
//!
//! transformers' names, Meta's modules: geometry `vision_layer_norm` /
//! `prompt_layer_norm` / `output_layer_norm` are `img_pre_norm` / `norm` /
//! `encode_norm`; decoder `self_attn_layer_norm` / `text_cross_attn_layer_norm`
//! / `vision_cross_attn_layer_norm` / `mlp_layer_norm` are `norm2` /
//! `catext_norm` / `norm1` / `norm3`, `output_layer_norm` the decoder's
//! output norm; scorer `query_proj` / `text_proj` / `text_mlp` are `hs_proj` /
//! `prompt_proj` / `prompt_mlp`; mask decoder `instance_projection` /
//! `semantic_projection` / `mask_embedder` are `instance_seg_head` /
//! `semantic_seg_head` / `mask_predictor.mask_embed`.
//!
//! Each attention's 1/sqrt(32) is folded into its q projection at load (the
//! half attention takes q pre-scaled): one f32 multiply on the weights before
//! their one round to f16, where Meta scales inside its SDPA.

use std::path::Path;
use std::sync::Arc;

use cudarc::driver::CudaSlice;
use half::f16;

use super::load::Reader;
use super::{Conv, GpuModelError, Norm};
use crate::gpu::{GpuExecutor, HalfTensor};

pub(super) const DET: &str = "detector_model";
/// The detector's LayerNorms are nn.LayerNorm defaults.
pub(super) const LN_EPS: f32 = 1e-5;
/// GroupNorm eps (nn.GroupNorm default) in the pixel decoder.
pub(super) const GN_EPS: f32 = 1e-5;
/// Key splits for the decoder's tensor-core box attention: the 36 row pairs
/// of the 72x72 memory in runs of 6, 192 blocks on the 4 x 8 (query block,
/// head) grid - one resident wave on an 84-SM die.
pub(super) const BOX_SPLITS: usize = 6;

/// Fixed geometry of SAM 3's detector (Meta's builder; the config does not
/// carry most of it, so it is checked against the tensors at load).
#[derive(Debug, Clone, Copy)]
pub struct DetGeom {
    pub d: usize,
    pub heads: usize,
    pub ffn: usize,
    pub queries: usize,
    /// side of the image-token grid (72)
    pub grid: usize,
    pub enc_layers: usize,
    pub dec_layers: usize,
    pub geo_layers: usize,
    /// text tokens the tower emits per prompt (32)
    pub text_tokens: usize,
    /// most exemplar boxes one prompt may carry
    pub max_boxes: usize,
}

impl DetGeom {
    pub fn hd(&self) -> usize {
        self.d / self.heads
    }
    pub fn tokens(&self) -> usize {
        self.grid * self.grid
    }
    /// decoder rows: the queries, then the presence token as the LAST row
    /// (self-attention, the norms and the FFN are per-row or permutation-
    /// equivariant, so where it rides is free; last keeps the queries
    /// contiguous from row 0 for every head that reads only them)
    pub fn dec_rows(&self) -> usize {
        self.queries + 1
    }
    pub fn max_prompt(&self) -> usize {
        self.text_tokens + self.max_boxes + 1
    }
}

/// Unbiased attention of `nq` rows over the 72x72 image memory - the
/// geometry encoder's cross-attention, the fusion encoder's self-attention.
/// The tensor-core box kernel at nbias 0 when it takes the shape, with its
/// keys split as finely as `part` holds when the rows alone cannot fill the
/// die (a few prompt rows are eight blocks otherwise); the plain vision
/// kernel when it does not. `tables` are any valid f32 planes - nothing reads
/// them at nbias 0.
#[allow(clippy::too_many_arguments)]
pub(super) fn memory_attn(
    exec: &GpuExecutor,
    g: &DetGeom,
    q: &CudaSlice<f16>,
    k: &CudaSlice<f16>,
    v: &CudaSlice<f16>,
    tables: (&CudaSlice<f32>, &CudaSlice<f32>),
    part: &mut CudaSlice<f32>,
    out: &mut CudaSlice<f16>,
    nq: usize,
) -> Result<(), GpuModelError> {
    let (heads, hd, t) = (g.heads, g.hd(), g.tokens());
    if !exec.sam3_box_attn_mma_fits(hd, g.grid, g.grid) {
        exec.vision_attn_h(q, k, v, out, nq, t, heads, hd, 1)?;
        return Ok(());
    }
    // 64 rows a block: past ~1k rows the row blocks fill the die themselves
    let nsplit = if nq >= 1024 {
        1
    } else {
        (g.grid / 2)
            .min(part.len() / (nq.max(1) * heads * (hd + 2)))
            .max(1)
    };
    exec.sam3_box_attn_mma(
        q, k, v, tables.0, tables.1, part, out, nq, heads, hd, g.grid, g.grid, 0, nsplit,
    )?;
    Ok(())
}

/// One multi-head attention's projections (q pre-scaled).
pub(super) struct Attn {
    pub q: Conv,
    pub k: Conv,
    pub v: Conv,
    pub o: Conv,
}

/// A pre-norm encoder layer (Meta's TransformerEncoderLayer, the geometry
/// encoder's and the fusion encoder's - they differ only in where positions
/// enter).
pub(super) struct EncLayer {
    pub norm1: Norm,
    pub sa: Attn,
    pub norm2: Norm,
    pub ca: Attn,
    pub norm3: Norm,
    pub fc1: Conv,
    pub fc2: Conv,
}

/// A post-norm DETR decoder layer.
pub(super) struct DecLayer {
    pub sa: Attn,
    pub sa_norm: Norm,
    pub ca_text: Attn,
    pub ca_text_norm: Norm,
    pub ca_img: Attn,
    pub ca_img_norm: Norm,
    pub fc1: Conv,
    pub fc2: Conv,
    pub mlp_norm: Norm,
}

/// One axis of the box-bias MLP (2 -> 256 -> heads), f32 as the kernel reads it.
pub(super) struct RpbMlp {
    pub w1: CudaSlice<f32>,
    pub b1: CudaSlice<f32>,
    pub w2: CudaSlice<f32>,
    pub b2: CudaSlice<f32>,
}

pub(super) struct Geometry {
    /// learned [label 0, label 1] rows, f32 [2][d]
    pub label_embed: CudaSlice<f32>,
    pub cls: CudaSlice<f32>,
    /// Linear(4 -> d), K padded to 8
    pub box_direct: Conv,
    /// the 7x7 valid conv over RoIAlign's output, as a (d*49 -> d) GEMM
    pub box_pool: Conv,
    /// Linear(258 -> d), K padded to 264
    pub box_pos: Conv,
    pub final_proj: Conv,
    pub prompt_norm: Norm,
    pub vision_norm: Norm,
    pub layers: Vec<EncLayer>,
    pub out_norm: Norm,
}

pub(super) struct Decoder {
    pub query_embed: CudaSlice<f32>,
    /// the initial reference boxes, sigmoid(reference_points) [queries][4]
    /// (a parameter transform, done once at load in f32)
    pub ref_init: CudaSlice<f32>,
    pub presence_token: CudaSlice<f32>,
    pub layers: Vec<DecLayer>,
    pub out_norm: Norm,
    pub box_head: [Conv; 3],
    pub ref_head: [Conv; 2],
    pub rpb_x: RpbMlp,
    pub rpb_y: RpbMlp,
    pub presence_head: [Conv; 3],
    pub presence_norm: Norm,
}

pub(super) struct Scorer {
    pub mlp1: Conv,
    pub mlp2: Conv,
    pub mlp_norm: Norm,
    pub text_proj: Conv,
    pub query_proj: Conv,
}

pub(super) struct SegHead {
    pub ca: Attn,
    pub ca_norm: Norm,
    pub convs: [Conv; 2],
    pub norms: [Norm; 2],
    pub instance: Conv,
    pub semantic: Conv,
    pub mask_embed: [Conv; 3],
}

/// Every buffer the detector touches, sized once.
pub(super) struct DetWorkspace {
    // the prompt, compacted to its valid rows
    pub prompt: CudaSlice<f32>,
    pub prompt16: CudaSlice<f16>,
    pub n_prompt: usize,
    // image-token planes [tokens][d]
    pub x: CudaSlice<f32>,
    pub h16: CudaSlice<f16>,
    pub hq16: CudaSlice<f16>,
    pub q16: CudaSlice<f16>,
    pub k16: CudaSlice<f16>,
    pub v16: CudaSlice<f16>,
    pub att16: CudaSlice<f16>,
    pub proj16: CudaSlice<f16>,
    pub ff16: CudaSlice<f16>,
    /// the fusion encoder's output E, f32 [tokens][d]
    pub enc: CudaSlice<f32>,
    pub mem16: CudaSlice<f16>,
    pub memq16: CudaSlice<f16>,
    // geometry / prompt-side planes [max_prompt][...]
    pub gx: CudaSlice<f32>,
    pub g16: CudaSlice<f16>,
    pub gq16: CudaSlice<f16>,
    pub gk16: CudaSlice<f16>,
    pub gv16: CudaSlice<f16>,
    pub gatt16: CudaSlice<f16>,
    pub gproj16: CudaSlice<f16>,
    pub gff16: CudaSlice<f16>,
    pub g32: CudaSlice<f32>,
    pub boxes: CudaSlice<f32>,
    pub boxes_xyxy: CudaSlice<f32>,
    pub box_in16: CudaSlice<f16>,
    pub roi16: CudaSlice<f16>,
    pub sine16: CudaSlice<f16>,
    pub vnorm: CudaSlice<f32>,
    // decoder planes [dec_rows][...]
    pub tgt: CudaSlice<f32>,
    pub t16: CudaSlice<f16>,
    pub tq16: CudaSlice<f16>,
    pub dq16: CudaSlice<f16>,
    pub dk16: CudaSlice<f16>,
    pub dv16: CudaSlice<f16>,
    pub datt16: CudaSlice<f16>,
    pub dproj16: CudaSlice<f16>,
    pub dff16: CudaSlice<f16>,
    pub reference: CudaSlice<f32>,
    pub qsine16: CudaSlice<f16>,
    pub qhid16: CudaSlice<f16>,
    pub qpos: CudaSlice<f32>,
    pub rpb_x: CudaSlice<f32>,
    pub rpb_y: CudaSlice<f32>,
    /// the tensor-core box attention's per-split partials
    pub box_part: CudaSlice<f32>,
    pub hs: CudaSlice<f32>,
    pub hs16: CudaSlice<f16>,
    pub mlp_a16: CudaSlice<f16>,
    pub mlp_b16: CudaSlice<f16>,
    pub box_delta: CudaSlice<f32>,
    pub presence: CudaSlice<f32>,
    pub pres32: CudaSlice<f32>,
    pub pres16: CudaSlice<f16>,
    // scorer
    pub sp: CudaSlice<f32>,
    pub pool_idx: CudaSlice<u32>,
    pub pooled: CudaSlice<f32>,
    pub pooled16: CudaSlice<f16>,
    pub pp: CudaSlice<f32>,
    pub hp: CudaSlice<f32>,
    pub logits: CudaSlice<f32>,
    pub probs: CudaSlice<f32>,
    // segmentation head
    pub segx: CudaSlice<f32>,
    pub seg16: CudaSlice<f16>,
    pub up16: CudaSlice<f16>,
    pub col16: CudaSlice<f16>,
    pub conv32: CudaSlice<f32>,
    pub gn_part: CudaSlice<f32>,
    pub gn_stat: CudaSlice<f32>,
    pub pix16: CudaSlice<f16>,
    pub inst16: CudaSlice<f16>,
    pub semantic: CudaSlice<f32>,
    /// the queries' mask embeddings, landed straight into a GEMM weight
    /// `[in d][out queries]` - the einsum against the pixel plane is a GEMM
    pub me: HalfTensor,
    pub masks: CudaSlice<f32>,
    pub bytes: u64,
}

/// SAM 3's detector, resident.
pub struct GpuSam3Detector {
    pub(super) exec: Arc<GpuExecutor>,
    pub(super) g: DetGeom,
    /// the 72x72 sine position table, f32 raster [tokens][d]
    pub(super) pos: CudaSlice<f32>,
    pub(super) geo: Geometry,
    pub(super) fusion: Vec<EncLayer>,
    pub(super) dec: Decoder,
    pub(super) scorer: Scorer,
    pub(super) seg: SegHead,
    pub(super) ws: DetWorkspace,
    weight_bytes: u64,
    /// SAM 3.1's detector folds presence into its logits
    /// (`supervise_joint_box_scores`): the score goes through Meta's
    /// `inverse_sigmoid` and +-10 clamp before its sigmoid (see
    /// [`GpuSam3Detector::picture_scores`])
    joint: bool,
}

/// Meta's `PositionEmbeddingSine(256, normalize=True)` over an n x n grid, as
/// a raster [n*n][d] f32 table: coordinate (i + 1) / (n + 1e-6) * 2pi,
/// frequency T^(2 floor(k/2) / (d/2)), sin on even k and cos on odd, the y
/// half then the x half - the f32 order of Meta's buffers (it computes them
/// on the GPU at build; the gate holds this table to them).
pub(crate) fn sine_pos_table(n: usize, d: usize, temperature: f32) -> Vec<f32> {
    let npf = d / 2;
    let scale = (2.0 * std::f64::consts::PI) as f32;
    let dim_t: Vec<f32> = (0..npf)
        .map(|k| temperature.powf((2 * (k / 2)) as f32 / npf as f32))
        .collect();
    let coord = |i: usize| (i as f32 + 1.0) / (n as f32 + 1e-6) * scale;
    let mut out = vec![0f32; n * n * d];
    for y in 0..n {
        for x in 0..n {
            let row = &mut out[(y * n + x) * d..(y * n + x + 1) * d];
            for (half, c) in [(0, coord(y)), (npf, coord(x))] {
                for k in 0..npf {
                    let u = c / dim_t[k];
                    row[half + k] = if k % 2 == 0 { u.sin() } else { u.cos() };
                }
            }
        }
    }
    out
}

impl Reader<'_> {
    /// An attention's four projections; q's weight and bias scaled by `qs`.
    fn attn(&mut self, prefix: &str, d: usize, qs: f32) -> Result<Attn, GpuModelError> {
        let lin = |r: &mut Self, name: &str, scale: f32| -> Result<Conv, GpuModelError> {
            let mut w = r.f32s(&format!("{prefix}.{name}.weight"), &[d, d])?;
            let mut b = r.f32s(&format!("{prefix}.{name}.bias"), &[d])?;
            if scale != 1.0 {
                w.iter_mut().for_each(|v| *v *= scale);
                b.iter_mut().for_each(|v| *v *= scale);
            }
            Ok(Conv {
                w: r.plane(&w, d, d, &format!("{prefix}.{name}"))?,
                b: r.dev(&b)?,
            })
        };
        Ok(Attn {
            q: lin(self, "q_proj", qs)?,
            k: lin(self, "k_proj", 1.0)?,
            v: lin(self, "v_proj", 1.0)?,
            o: lin(self, "o_proj", 1.0)?,
        })
    }

    /// `nn.Linear` [out, in] -> a GEMM plane (K zero-padded to `kp`) + bias.
    fn lin(
        &mut self,
        prefix: &str,
        out: usize,
        inp: usize,
        kp: usize,
    ) -> Result<Conv, GpuModelError> {
        let w = self.f32s(&format!("{prefix}.weight"), &[out, inp])?;
        let w = if kp == inp {
            w
        } else {
            let mut p = vec![0f32; out * kp];
            for o in 0..out {
                p[o * kp..o * kp + inp].copy_from_slice(&w[o * inp..(o + 1) * inp]);
            }
            p
        };
        Ok(Conv {
            w: self.plane(&w, kp, out, prefix)?,
            b: self.vec(&format!("{prefix}.bias"), out)?,
        })
    }

    fn enc_layer(
        &mut self,
        prefix: &str,
        d: usize,
        ffn: usize,
        qs: f32,
    ) -> Result<EncLayer, GpuModelError> {
        Ok(EncLayer {
            norm1: self.norm(&format!("{prefix}.layer_norm1"), d)?,
            sa: self.attn(&format!("{prefix}.self_attn"), d, qs)?,
            norm2: self.norm(&format!("{prefix}.layer_norm2"), d)?,
            ca: self.attn(&format!("{prefix}.cross_attn"), d, qs)?,
            norm3: self.norm(&format!("{prefix}.layer_norm3"), d)?,
            fc1: self.lin(&format!("{prefix}.mlp.fc1"), ffn, d, d)?,
            fc2: self.lin(&format!("{prefix}.mlp.fc2"), d, ffn, ffn)?,
        })
    }

    /// A box-bias MLP, f32 [out][in] as the table kernel reads it.
    fn rpb(&mut self, prefix: &str, hid: usize, heads: usize) -> Result<RpbMlp, GpuModelError> {
        let w1 = self.f32s(&format!("{prefix}.layer1.weight"), &[hid, 2])?;
        let w2 = self.f32s(&format!("{prefix}.layer2.weight"), &[heads, hid])?;
        Ok(RpbMlp {
            w1: self.dev(&w1)?,
            b1: self.vec(&format!("{prefix}.layer1.bias"), hid)?,
            w2: self.dev(&w2)?,
            b2: self.vec(&format!("{prefix}.layer2.bias"), heads)?,
        })
    }
}

impl GpuSam3Detector {
    pub fn geom(&self) -> DetGeom {
        self.g
    }
    pub fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }
    pub fn workspace_bytes(&self) -> u64 {
        self.ws.bytes
    }

    pub fn load_dir(
        exec: Arc<GpuExecutor>,
        dir: &Path,
        max_boxes: usize,
    ) -> Result<Self, GpuModelError> {
        let st = super::checkpoint::open(dir)?;
        let joint = super::checkpoint::is_multiplex(&*st);
        if !exec.has_sam3_detector() {
            return Err(GpuModelError::Unsupported(
                "this kernel pack predates SAM 3's detector (slots 760-769) - rebuild or update \
                 the pack"
                    .into(),
            ));
        }
        let g = DetGeom {
            d: 256,
            heads: 8,
            ffn: 2048,
            queries: 200,
            grid: 72,
            enc_layers: 6,
            dec_layers: 6,
            geo_layers: 3,
            text_tokens: 32,
            max_boxes: max_boxes.clamp(1, 64),
        };
        let (d, ffn, nq) = (g.d, g.ffn, g.queries);
        // the graph's fixed shape, read off the tensors that carry it
        let shape = |name: &str| st.tensor(name).map(|(t, _)| t.shape.clone());
        let checks = [
            (
                format!("{DET}.detr_decoder.query_embed.weight"),
                vec![nq, d],
            ),
            (
                format!(
                    "{DET}.detr_encoder.layers.{}.mlp.fc1.weight",
                    g.enc_layers - 1
                ),
                vec![ffn, d],
            ),
            (
                format!(
                    "{DET}.detr_decoder.layers.{}.mlp.fc1.weight",
                    g.dec_layers - 1
                ),
                vec![ffn, d],
            ),
            (
                format!(
                    "{DET}.geometry_encoder.layers.{}.mlp.fc1.weight",
                    g.geo_layers - 1
                ),
                vec![ffn, d],
            ),
        ];
        for (name, want) in checks {
            if shape(&name).as_deref() != Some(&want[..]) {
                return Err(GpuModelError::Unsupported(format!(
                    "sam3 detector: {name} is {:?}, the graph builds {want:?}",
                    shape(&name)
                )));
            }
        }
        let ws_bytes = DetWorkspace::bytes_for(&g);
        exec.vram_load_gate(ws_bytes + (60u64 << 20), "sam3 detector")
            .map_err(GpuModelError::WontFit)?;

        let qs = 1.0 / (g.hd() as f32).sqrt();
        let mut r = Reader {
            st: &*st,
            exec: &exec,
            bytes: 0,
        };
        let pos = r.dev(&sine_pos_table(g.grid, d, 10000.0))?;

        // ---- geometry encoder ----
        let gp = format!("{DET}.geometry_encoder");
        let pool_w = r.f32s(&format!("{gp}.boxes_pool_project.weight"), &[d, d, 7, 7])?;
        let geo = Geometry {
            label_embed: {
                let v = r.f32s(&format!("{gp}.label_embed.weight"), &[2, d])?;
                r.dev(&v)?
            },
            cls: {
                let v = r.f32s(&format!("{gp}.cls_embed.weight"), &[1, d])?;
                r.dev(&v)?
            },
            box_direct: r.lin(&format!("{gp}.boxes_direct_project"), d, 4, 8)?,
            box_pool: Conv {
                // [out][c][ky][kx] flattens to RoIAlign's [c][py][px] output order
                w: r.plane(&pool_w, d * 49, d, "sam3 boxes_pool_project")?,
                b: r.vec(&format!("{gp}.boxes_pool_project.bias"), d)?,
            },
            box_pos: r.lin(
                &format!("{gp}.boxes_pos_enc_project"),
                d,
                d + 2,
                (d + 2).next_multiple_of(8),
            )?,
            final_proj: r.lin(&format!("{gp}.final_proj"), d, d, d)?,
            prompt_norm: r.norm(&format!("{gp}.prompt_layer_norm"), d)?,
            vision_norm: r.norm(&format!("{gp}.vision_layer_norm"), d)?,
            layers: (0..g.geo_layers)
                .map(|i| r.enc_layer(&format!("{gp}.layers.{i}"), d, ffn, qs))
                .collect::<Result<_, _>>()?,
            out_norm: r.norm(&format!("{gp}.output_layer_norm"), d)?,
        };

        // ---- fusion encoder ----
        let fusion = (0..g.enc_layers)
            .map(|i| r.enc_layer(&format!("{DET}.detr_encoder.layers.{i}"), d, ffn, qs))
            .collect::<Result<Vec<_>, _>>()?;

        // ---- decoder ----
        let dp = format!("{DET}.detr_decoder");
        let mut layers = Vec::with_capacity(g.dec_layers);
        for i in 0..g.dec_layers {
            let l = |s: &str| format!("{dp}.layers.{i}.{s}");
            layers.push(DecLayer {
                sa: r.attn(&l("self_attn"), d, qs)?,
                sa_norm: r.norm(&l("self_attn_layer_norm"), d)?,
                ca_text: r.attn(&l("text_cross_attn"), d, qs)?,
                ca_text_norm: r.norm(&l("text_cross_attn_layer_norm"), d)?,
                ca_img: r.attn(&l("vision_cross_attn"), d, qs)?,
                ca_img_norm: r.norm(&l("vision_cross_attn_layer_norm"), d)?,
                fc1: r.lin(&l("mlp.fc1"), ffn, d, d)?,
                fc2: r.lin(&l("mlp.fc2"), d, ffn, ffn)?,
                mlp_norm: r.norm(&l("mlp_layer_norm"), d)?,
            });
        }
        let dec = Decoder {
            query_embed: {
                let v = r.f32s(&format!("{dp}.query_embed.weight"), &[nq, d])?;
                r.dev(&v)?
            },
            ref_init: {
                let v = r.f32s(&format!("{dp}.reference_points.weight"), &[nq, 4])?;
                let sig: Vec<f32> = v.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
                r.dev(&sig)?
            },
            presence_token: {
                let v = r.f32s(&format!("{dp}.presence_token.weight"), &[1, d])?;
                r.dev(&v)?
            },
            layers,
            out_norm: r.norm(&format!("{dp}.output_layer_norm"), d)?,
            box_head: [
                r.lin(&format!("{dp}.box_head.layer1"), d, d, d)?,
                r.lin(&format!("{dp}.box_head.layer2"), d, d, d)?,
                r.lin(&format!("{dp}.box_head.layer3"), 4, d, d)?,
            ],
            ref_head: [
                r.lin(&format!("{dp}.ref_point_head.layer1"), d, 2 * d, 2 * d)?,
                r.lin(&format!("{dp}.ref_point_head.layer2"), d, d, d)?,
            ],
            rpb_x: r.rpb(&format!("{dp}.box_rpb_embed_x"), d, g.heads)?,
            rpb_y: r.rpb(&format!("{dp}.box_rpb_embed_y"), d, g.heads)?,
            presence_head: [
                r.lin(&format!("{dp}.presence_head.layer1"), d, d, d)?,
                r.lin(&format!("{dp}.presence_head.layer2"), d, d, d)?,
                r.lin(&format!("{dp}.presence_head.layer3"), 1, d, d)?,
            ],
            presence_norm: r.norm(&format!("{dp}.presence_layer_norm"), d)?,
        };

        // ---- scorer ----
        let sp = format!("{DET}.dot_product_scoring");
        let scorer = Scorer {
            mlp1: r.lin(&format!("{sp}.text_mlp.layer1"), ffn, d, d)?,
            mlp2: r.lin(&format!("{sp}.text_mlp.layer2"), d, ffn, ffn)?,
            mlp_norm: r.norm(&format!("{sp}.text_mlp_out_norm"), d)?,
            text_proj: r.lin(&format!("{sp}.text_proj"), d, d, d)?,
            query_proj: r.lin(&format!("{sp}.query_proj"), d, d, d)?,
        };

        // ---- segmentation head ----
        let mp = format!("{DET}.mask_decoder");
        let conv3 = |r: &mut Reader, i: usize| -> Result<Conv, GpuModelError> {
            let name = format!("{mp}.pixel_decoder.conv_layers.{i}.weight");
            let w = r.f32s(&name, &[d, d, 3, 3])?;
            let mut out = vec![0f32; w.len()];
            for o in 0..d {
                for c in 0..d {
                    for t in 0..9 {
                        out[(o * 9 + t) * d + c] = w[(o * d + c) * 9 + t];
                    }
                }
            }
            Ok(Conv {
                w: r.plane(&out, 9 * d, d, &name)?,
                b: r.vec(&format!("{mp}.pixel_decoder.conv_layers.{i}.bias"), d)?,
            })
        };
        let lin4 = |r: &mut Reader, name: &str, out: usize| -> Result<Conv, GpuModelError> {
            let w = r.f32s(&format!("{name}.weight"), &[out, d, 1, 1])?;
            Ok(Conv {
                w: r.plane(&w, d, out, name)?,
                b: r.vec(&format!("{name}.bias"), out)?,
            })
        };
        let seg = SegHead {
            ca: r.attn(&format!("{mp}.prompt_cross_attn"), d, qs)?,
            ca_norm: r.norm(&format!("{mp}.prompt_cross_attn_norm"), d)?,
            convs: [conv3(&mut r, 0)?, conv3(&mut r, 1)?],
            norms: [
                r.norm(&format!("{mp}.pixel_decoder.norms.0"), d)?,
                r.norm(&format!("{mp}.pixel_decoder.norms.1"), d)?,
            ],
            instance: lin4(&mut r, &format!("{mp}.instance_projection"), d)?,
            semantic: lin4(&mut r, &format!("{mp}.semantic_projection"), 1)?,
            mask_embed: [
                r.lin(&format!("{mp}.mask_embedder.layers.0"), d, d, d)?,
                r.lin(&format!("{mp}.mask_embedder.layers.1"), d, d, d)?,
                r.lin(&format!("{mp}.mask_embedder.layers.2"), d, d, d)?,
            ],
        };
        let weight_bytes = r.bytes;
        let ws = DetWorkspace::new(&exec, &g)?;
        tracing::info!(
            weights_mib = weight_bytes >> 20,
            workspace_mib = ws.bytes >> 20,
            max_boxes = g.max_boxes,
            "sam3 detector resident"
        );
        Ok(Self {
            exec,
            g,
            pos,
            geo,
            fusion,
            dec,
            scorer,
            seg,
            ws,
            weight_bytes,
            joint,
        })
    }

    /// The picture's scores from `dets`, as the checkpoint's own detector
    /// makes them. SAM 3 (Meta's image model): sigmoid(class) x presence, the
    /// probabilities the decoder already holds. SAM 3.1 (its detector is
    /// built with `supervise_joint_box_scores`): that product is the decoder's
    /// logit - `inverse_sigmoid` (eps 1e-3) clamped to +-10 - and the score is
    /// its sigmoid (Meta's `pred_logits.sigmoid()`), which only moves a
    /// score past 0.99995 or under 4.5e-5.
    pub fn picture_scores(&self, dets: &Sam3Detections) -> Vec<f32> {
        if self.joint {
            dets.probs.iter().map(|&p| joint_score(p)).collect()
        } else {
            dets.probs.clone()
        }
    }

    /// Whether this detector scores the SAM 3.1 way (`picture_scores`).
    pub fn joint_scores(&self) -> bool {
        self.joint
    }

    /// The 72x72 position table on the device, f32 raster [tokens][d].
    pub fn pos_table(&self) -> &CudaSlice<f32> {
        &self.pos
    }
}

impl DetWorkspace {
    /// (f32 elements, f16 elements) - the admission estimate, kept beside the
    /// allocation it estimates.
    fn plan(g: &DetGeom) -> (usize, usize) {
        // (f32 elements, f16 elements)
        let (d, t, ffn, q, p) = (g.d, g.tokens(), g.ffn, g.dec_rows(), g.max_prompt());
        let px4 = 16 * t;
        let px2 = 4 * t;
        let f32s = p * d          // prompt
            + 2 * t * d           // x, enc
            + 2 * p * d           // gx, g32
            + 8 * g.max_boxes     // boxes, boxes_xyxy
            + t * d               // vnorm
            + q * d               // tgt
            + 4 * q               // reference
            + q * d               // qpos
            + 2 * q * g.grid * g.heads // rpb tables
            + GpuExecutor::sam3_box_attn_mma_part_len(q, g.heads, g.hd(), BOX_SPLITS)
            + q * d               // hs
            + q * 8               // box delta
            + 2 + d               // presence, pres32
            + p * d + 3 * d + q * d + 2 * q // scorer
            + t * d               // segx
            + px4 * d             // conv32
            + px4.div_ceil(64) * d + 2 * 8 // gn scratch
            + px4                 // semantic
            + px4 * g.queries; // masks
        let f16s = p * d          // prompt16
            + 8 * t * d           // h16 hq16 q16 k16 v16 att16 proj16 + mem16
            + t * d               // memq16
            + t * ffn             // ff16
            + 6 * p * d + p * ffn // geometry planes
            + g.max_boxes * (8 + d * 49 + 264) // box inputs
            + 7 * q * d + q * ffn // decoder planes
            + q * 2 * d           // qsine16
            + q * d               // qhid16
            + 2 * q * d + d       // mlp a/b, pres16
            + d                   // pooled16
            + t * d               // seg16
            + px2 * d             // up16 (the 144 stage)
            + px4 * 9 * d         // col16
            + px4 * d * 2         // pix16, inst16
            + g.queries * d; // me
        (f32s, f16s)
    }

    fn bytes_for(g: &DetGeom) -> u64 {
        let (a, b) = Self::plan(g);
        (a * 4 + b * 2) as u64
    }

    fn new(exec: &GpuExecutor, g: &DetGeom) -> Result<Self, GpuModelError> {
        let (d, t, ffn, q, p) = (g.d, g.tokens(), g.ffn, g.dec_rows(), g.max_prompt());
        let px4 = 16 * t;
        let px2 = 4 * t;
        let f = |n: usize| exec.alloc(n);
        let h = |n: usize| exec.alloc_f16(n);
        Ok(Self {
            prompt: f(p * d)?,
            prompt16: h(p * d)?,
            n_prompt: 0,
            x: f(t * d)?,
            h16: h(t * d)?,
            hq16: h(t * d)?,
            q16: h(t * d)?,
            k16: h(t * d)?,
            v16: h(t * d)?,
            att16: h(t * d)?,
            proj16: h(t * d)?,
            ff16: h(t * ffn)?,
            enc: f(t * d)?,
            mem16: h(t * d)?,
            memq16: h(t * d)?,
            gx: f(p * d)?,
            g16: h(p * d)?,
            gq16: h(p * d)?,
            gk16: h(t * d)?,
            gv16: h(t * d)?,
            gatt16: h(p * d)?,
            gproj16: h(p * d)?,
            gff16: h(p * ffn)?,
            g32: f(p * d)?,
            boxes: f(4 * g.max_boxes)?,
            boxes_xyxy: f(4 * g.max_boxes)?,
            box_in16: h(8 * g.max_boxes)?,
            roi16: h(g.max_boxes * d * 49)?,
            sine16: h(g.max_boxes * 264)?,
            vnorm: f(t * d)?,
            tgt: f(q * d)?,
            t16: h(q * d)?,
            tq16: h(q * d)?,
            dq16: h(q * d)?,
            dk16: h(t * d)?,
            dv16: h(t * d)?,
            datt16: h(q * d)?,
            dproj16: h(q * d)?,
            dff16: h(q * ffn)?,
            reference: f(4 * g.queries)?,
            qsine16: h(g.queries * 2 * d)?,
            qhid16: h(g.queries * d)?,
            qpos: f(g.queries * d)?,
            rpb_x: f(g.queries * g.grid * g.heads)?,
            rpb_y: f(g.queries * g.grid * g.heads)?,
            box_part: f(GpuExecutor::sam3_box_attn_mma_part_len(
                q,
                g.heads,
                g.hd(),
                BOX_SPLITS,
            ))?,
            hs: f(q * d)?,
            hs16: h(q * d)?,
            mlp_a16: h(q * d)?,
            mlp_b16: h(q * d)?,
            box_delta: f(q * 8)?,
            presence: f(1)?,
            pres32: f(d)?,
            pres16: h(d)?,
            sp: f(p * d)?,
            pool_idx: {
                let mut idx = exec.alloc_u32(p)?;
                exec.upload_u32(&(0..p as u32).collect::<Vec<_>>(), &mut idx)?;
                idx
            },
            pooled: f(d)?,
            pooled16: h(d)?,
            pp: f(d)?,
            hp: f(q * d)?,
            logits: f(g.queries)?,
            probs: f(g.queries)?,
            segx: f(t * d)?,
            seg16: h(t * d)?,
            up16: h(px2 * d)?,
            col16: h(px4 * 9 * d)?,
            conv32: f(px4 * d)?,
            gn_part: f(GpuExecutor::dp_group_norm_part_len(1, px4, d))?,
            gn_stat: f(2 * 8)?,
            pix16: h(px4 * d)?,
            inst16: h(px4 * d)?,
            semantic: f(px4)?,
            me: HalfTensor {
                buf: h(g.queries * d)?,
                dims: vec![d, g.queries],
            },
            masks: f(px4 * g.queries)?,
            bytes: Self::bytes_for(g),
        })
    }
}

/// What the detector decided for one picture and prompt, read back.
#[derive(Debug, Clone)]
pub struct Sam3Detections {
    /// per query: sigmoid(logit) * sigmoid(presence) - the processor's score
    pub probs: Vec<f32>,
    pub logits: Vec<f32>,
    /// per query: cxcywh, normalized to the picture
    pub boxes: Vec<[f32; 4]>,
    pub presence_logit: f32,
}

impl GpuSam3Detector {
    /// Copy a decoder-side result out (the gate's view): `hs` the last layer's
    /// normed queries [queries][d], the boxes, the logits, the presence logit.
    pub fn read_detections(&self) -> Result<Sam3Detections, GpuModelError> {
        let nq = self.g.queries;
        let probs = self.exec.to_host_len(&self.ws.probs, nq)?;
        let logits = self.exec.to_host_len(&self.ws.logits, nq)?;
        let b = self.exec.to_host_len(&self.ws.reference, nq * 4)?;
        let presence_logit = self.exec.to_host_len(&self.ws.presence, 1)?[0];
        Ok(Sam3Detections {
            probs,
            logits,
            boxes: b.as_chunks::<4>().0.to_vec(),
            presence_logit,
        })
    }

    /// The last decoder layer's normed queries, [queries][d] f32 (the decoder
    /// keeps the presence token as its LAST row, so these are rows 0..).
    pub fn read_queries(&self) -> Result<Vec<f32>, GpuModelError> {
        let (d, nq) = (self.g.d, self.g.queries);
        Ok(self.exec.to_host_len(&self.ws.hs, nq * d)?)
    }

    /// The fusion encoder's output E, [tokens][d] f32 raster.
    pub fn read_memory(&self) -> Result<Vec<f32>, GpuModelError> {
        Ok(self
            .exec
            .to_host_len(&self.ws.enc, self.g.tokens() * self.g.d)?)
    }

    /// The compacted prompt, [n_prompt][d] f32 (text rows, then geometry rows).
    pub fn read_prompt(&self) -> Result<Vec<f32>, GpuModelError> {
        Ok(self
            .exec
            .to_host_len(&self.ws.prompt, self.ws.n_prompt * self.g.d)?)
    }

    /// The mask logits of the last [`Self::segment`], on the device: pixel-major
    /// `[288^2][queries]` f32 (what the mask upsample reads in place).
    pub fn masks_plane(&self) -> &CudaSlice<f32> {
        &self.ws.masks
    }

    /// All queries' mask logits at 288^2, pixel-major [288^2][queries] f32.
    pub fn read_masks(&self) -> Result<Vec<f32>, GpuModelError> {
        Ok(self
            .exec
            .to_host_len(&self.ws.masks, 16 * self.g.tokens() * self.g.queries)?)
    }

    /// The semantic logit plane at 288^2.
    pub fn read_semantic(&self) -> Result<Vec<f32>, GpuModelError> {
        Ok(self
            .exec
            .to_host_len(&self.ws.semantic, 16 * self.g.tokens())?)
    }
}

/// Meta's joint score: a probability through the decoder's `inverse_sigmoid`
/// (eps 1e-3) and its +-10 clamp, then sigmoid again - SAM 3's video
/// detector and SAM 3.1's detector score this way.
pub(super) fn joint_score(p: f32) -> f32 {
    let x = p.clamp(0.0, 1.0);
    let l = (x.max(1e-3) / (1.0 - x).max(1e-3)).ln().clamp(-10.0, 10.0);
    1.0 / (1.0 + (-l).exp())
}
