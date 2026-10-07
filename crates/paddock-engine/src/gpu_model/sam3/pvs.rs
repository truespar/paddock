//! SAM 3's interactive single-object segmentation on one picture (PVS) -
//! Meta's `SAM3InteractiveImagePredictor`: clicks (positive and negative), a
//! box, and optionally the previous answer's low-resolution logits in; up to
//! three candidate masks and their predicted IoU out. The SAM 2 lineage head
//! over the TRACKER neck:
//!
//!   sparse = point pe + label embedding   (box corners first, then clicks,
//!                                           then one padding point)
//!   dense  = no_mask_embed, or the mask prompt through mask_embed
//!   keys   = trk72 + no_memory_embedding + dense       5184 x 256
//!   tokens = [obj_score, iou, mask x4] + sparse        T x 256
//!   2 x TwoWayAttentionBlock (post-norm; layer 0's self-attention has no
//!   position and no residual), then a final token->image attention + LN
//!   up     = GELU(LN(convT(keys) + conv_s1)), GELU(convT(.) + conv_s0)
//!   masks  = hyper(mask tokens) @ up                   4 x 288^2 logits
//!   iou    = MLP(iou token), through a sigmoid for SAM 3 (`iou_prediction_use_sigmoid`);
//!            SAM 3.1 trains it without one, so its scores are unbounded
//!
//! then Meta's output choice (multimask: masks 1..3; single: mask 0 unless
//! its stability score is under 0.98, then the best of 1..3) and its hole
//! fill (background components of at most 256 px at 288^2 become foreground).
//! The weights carry transformers' names (`tracker_model.*`); the math is
//! Meta's.
//!
//! The same heads track an object through a video (`_forward_sam_heads`):
//! the keys are then the memory-conditioned feature (`PvsFeatures`), the
//! prompt may be empty, there is no hole fill, and two more outputs matter -
//! the object pointer (`object_pointer_proj` over the chosen mask's token:
//! the best of 1..3 with three candidates, mask 0's with one, whatever mask
//! the stability check picked; `no_object_pointer` when the object score is
//! not positive) and the masks forced to -1024 where the object is gone.
//!
//! Precision is the detector's class: f16 GEMM operands, f32 residuals,
//! norms and heads. The decoder's q projections carry 1/sqrt(hd) (32-wide
//! heads for self-attention, 16-wide for both cross-attentions, which run at
//! 128 internal width).

use std::path::Path;
use std::sync::Arc;

use cudarc::driver::CudaSlice;
use half::f16;

use super::load::Reader;
use super::{Conv, ConvT, GpuModelError, GpuSam3Vision, Norm};
use crate::gpu::{GpuExecutor, HalfTensor, Mlp3, Sam3MaskDownOut};

const PE: &str = "tracker_model.prompt_encoder";
const DEC: &str = "tracker_model.mask_decoder";
/// transformer width, heads, the cross-attentions' internal width, MLP
const D: usize = 256;
const HEADS: usize = 8;
const CA: usize = 128;
const MLP: usize = 2048;
/// the 72 x 72 image tokens and the 288 x 288 mask grid
const GRID: usize = 72;
const SIDE: usize = 288;
/// [obj_score, iou, mask x 4] - the decoder's own tokens, before the prompt
const OWN: usize = 6;
const MASKS: usize = 4;
/// most clicks one prompt carries
pub const PVS_MAX_POINTS: usize = 32;
/// box corners + clicks + the padding point
const MAX_SPARSE: usize = 2 + PVS_MAX_POINTS + 1;
const MAX_T: usize = OWN + MAX_SPARSE;
/// LayerNorm eps: nn.LayerNorm's in the transformer, LayerNorm2d's elsewhere
const LN_EPS: f32 = 1e-5;
const LN2D_EPS: f32 = 1e-6;
/// Meta's predictor: stability band and threshold, hole area, re-fed clamp
const STABILITY_DELTA: f32 = 0.05;
const STABILITY_THRESH: f64 = 0.98;
const MAX_HOLE_AREA: usize = 256;
const MASK_CLAMP: f32 = 32.0;

struct Attn {
    q: Conv,
    k: Conv,
    v: Conv,
    o: Conv,
}

struct Layer {
    sa: Attn,
    n1: Norm,
    t2i: Attn,
    n2: Norm,
    fc1: Conv,
    fc2: Conv,
    n3: Norm,
    i2t: Attn,
    n4: Norm,
}

/// A small MLP's planes, f32 `[out][in]` - several sets end to end when each
/// row of the launch has its own.
struct Mlp3W {
    w1: CudaSlice<f32>,
    b1: CudaSlice<f32>,
    w2: CudaSlice<f32>,
    b2: CudaSlice<f32>,
    w3: CudaSlice<f32>,
    b3: CudaSlice<f32>,
    out: usize,
    per_row: bool,
}

impl Mlp3W {
    fn view(&self) -> Mlp3<'_> {
        Mlp3 {
            w1: &self.w1,
            b1: &self.b1,
            w2: &self.w2,
            b2: &self.b2,
            w3: &self.w3,
            b3: &self.b3,
            inp: D,
            hid: D,
            out: self.out,
            per_row: self.per_row,
        }
    }
}

/// The mask-prompt path: two Conv2d k2 s2 -> LN2d -> GELU stages, then a 1x1.
struct MaskEmbed {
    w1: CudaSlice<f32>,
    b1: CudaSlice<f32>,
    n1: Norm,
    w2: CudaSlice<f32>,
    b2: CudaSlice<f32>,
    n2: Norm,
    c3: Conv,
}

struct Workspace {
    xy: CudaSlice<f32>,
    labels: CudaSlice<u32>,
    sparse: CudaSlice<f32>,
    /// the tokens as they entered (the decoder's query position) and the
    /// residual stream, f32 `[T][256]`, with their f16 views
    qpe: CudaSlice<f32>,
    q: CudaSlice<f32>,
    q16: CudaSlice<f16>,
    qq16: CudaSlice<f16>,
    keys: CudaSlice<f32>,
    k16: CudaSlice<f16>,
    kq16: CudaSlice<f16>,
    dense: CudaSlice<f32>,
    pq16: CudaSlice<f16>,
    pk16: CudaSlice<f16>,
    pv16: CudaSlice<f16>,
    att16: CudaSlice<f16>,
    proj16: CudaSlice<f16>,
    mlp16: CudaSlice<f16>,
    up1: CudaSlice<f32>,
    up1h: CudaSlice<f16>,
    up2: CudaSlice<f32>,
    up2h: CudaSlice<f16>,
    tok: CudaSlice<f32>,
    hyper: CudaSlice<f32>,
    hyper16: HalfTensor,
    iou: CudaSlice<f32>,
    obj: CudaSlice<f32>,
    ptr: CudaSlice<f32>,
    md1: CudaSlice<f32>,
    md2: CudaSlice<f16>,
    mask_in: CudaSlice<f32>,
    counts: CudaSlice<u32>,
    lab: CudaSlice<u32>,
    area: CudaSlice<u32>,
    /// the decoder's raw logits `[288^2][4]` - the next refinement's prompt
    logits: CudaSlice<f32>,
    /// the same after the hole fill - what the masks are cut from
    post: CudaSlice<f32>,
    bytes: u64,
}

/// One click prompt, its coordinates already Meta's: the point in the
/// 1008-pixel input frame plus 0.5, divided by 1008.
#[derive(Debug, Clone, Copy)]
pub struct PvsPoint {
    pub x: f32,
    pub y: f32,
    pub positive: bool,
}

/// Where a mask prompt comes from.
#[derive(Debug, Clone, Copy)]
pub enum PvsMask<'a> {
    /// column `k` of the previous call's raw logits (refining that answer)
    Last(usize),
    /// a `288 x 288` logits plane from the caller (the gate feeds Meta's own)
    Host(&'a [f32]),
    /// a `288 x 288` plane already on the device, from element `off` (the
    /// video's mask-as-output pass: a detection's downsampled mask)
    Device(&'a CudaSlice<f32>, usize),
}

#[derive(Debug, Clone)]
pub struct PvsPrompt<'a> {
    pub points: Vec<PvsPoint>,
    /// x0, y0, x1, y1, normalized as the points are
    pub bbox: Option<[f32; 4]>,
    pub mask: Option<PvsMask<'a>>,
    /// three candidates (a single click is ambiguous) or one
    pub multimask: bool,
}

/// What the heads read: the 72^2 key feature (the picture's tracker level,
/// or a video frame's memory-conditioned one), whether Meta's
/// `no_memory_embedding` still has to be added to it (a picture, and a
/// video frame with no memory yet), and the two upscaling skips (conv_s0
/// over the 288^2 level, conv_s1 over the 144^2 one). `track` is the video
/// tracker's use: an empty prompt is a propagation, the hole fill is the
/// caller's, and an absent object's masks become -1024.
#[derive(Clone, Copy)]
pub struct PvsFeatures<'a> {
    pub feat: &'a CudaSlice<f32>,
    pub no_memory: bool,
    pub s0: &'a CudaSlice<f32>,
    pub s1: &'a CudaSlice<f32>,
    pub track: bool,
}

impl<'a> PvsFeatures<'a> {
    /// The picture `vision` last encoded with its tracker neck.
    pub fn picture(vision: &'a GpuSam3Vision) -> Self {
        Self {
            feat: vision.trk_level(2),
            no_memory: true,
            s0: vision.trk_s0(),
            s1: vision.trk_s1(),
            track: false,
        }
    }
}

/// Meta's "no object" logit (`NO_OBJ_SCORE`).
const NO_OBJ_LOGIT: f32 = -1024.0;

/// What a call chose: which of the 4 masks, best first, with its predicted
/// IoU. The masks themselves stay on the device ([`GpuSam3Pvs::masks`]).
#[derive(Debug, Clone)]
pub struct PvsResult {
    pub candidates: Vec<(usize, f32)>,
    /// all four predicted IoUs (the gate's view)
    pub iou: [f32; 4],
    /// sigmoid of the object-score head: is there an object under the
    /// prompt at all (the tracker's occlusion logit; the image predictor
    /// itself does not use it)
    pub object_score: f32,
    /// the same head's logit, the tracker's view (> 0 = the object is there)
    pub object_logit: f32,
    /// the mask whose token made the object pointer
    pub pointer_mask: usize,
}

/// SAM 3's tracker heads on one picture, resident.
pub struct GpuSam3Pvs {
    exec: Arc<GpuExecutor>,
    gauss: CudaSlice<f32>,
    point_embed: CudaSlice<f32>,
    not_a_point: CudaSlice<f32>,
    no_mask: CudaSlice<f32>,
    no_mem: CudaSlice<f32>,
    mask_embed: MaskEmbed,
    dense_pe: CudaSlice<f32>,
    own_tokens: CudaSlice<f32>,
    layers: Vec<Layer>,
    final_attn: Attn,
    final_norm: Norm,
    up1: ConvT,
    up_norm: Norm,
    up2: ConvT,
    hyper: Mlp3W,
    iou_head: Mlp3W,
    /// SAM 3 puts the IoU head through a sigmoid, SAM 3.1 does not
    iou_sigmoid: bool,
    obj_head: Mlp3W,
    ptr_head: Mlp3W,
    /// SAM 3's fixed no-object pointer (None for SAM 3.1)
    no_obj_ptr: Option<CudaSlice<f32>>,
    /// -1024 over the four masks, copied in where the object is gone
    gone: CudaSlice<f32>,
    ws: Workspace,
    weight_bytes: u64,
}

impl Reader<'_> {
    /// `[out, in]` Linear -> GEMM plane + bias, both scaled by `scale`.
    fn pvs_lin(
        &mut self,
        prefix: &str,
        out: usize,
        inp: usize,
        scale: f32,
    ) -> Result<Conv, GpuModelError> {
        let mut w = self.f32s(&format!("{prefix}.weight"), &[out, inp])?;
        let mut b = self.f32s(&format!("{prefix}.bias"), &[out])?;
        if scale != 1.0 {
            w.iter_mut().for_each(|v| *v *= scale);
            b.iter_mut().for_each(|v| *v *= scale);
        }
        Ok(Conv {
            w: self.plane(&w, inp, out, prefix)?,
            b: self.dev(&b)?,
        })
    }

    fn pvs_attn(&mut self, prefix: &str, inner: usize) -> Result<Attn, GpuModelError> {
        let qs = 1.0 / ((inner / HEADS) as f32).sqrt();
        Ok(Attn {
            q: self.pvs_lin(&format!("{prefix}.q_proj"), inner, D, qs)?,
            k: self.pvs_lin(&format!("{prefix}.k_proj"), inner, D, 1.0)?,
            v: self.pvs_lin(&format!("{prefix}.v_proj"), inner, D, 1.0)?,
            o: self.pvs_lin(&format!("{prefix}.o_proj"), D, inner, 1.0)?,
        })
    }

    /// One three-layer MLP (`proj_in`, `layers.0`, `proj_out`) per prefix,
    /// the sets laid end to end.
    fn pvs_mlp3(&mut self, prefixes: &[String], out: usize) -> Result<Mlp3W, GpuModelError> {
        let (mut w1, mut b1, mut w2, mut b2, mut w3, mut b3) =
            (vec![], vec![], vec![], vec![], vec![], vec![]);
        for p in prefixes {
            w1.extend(self.f32s(&format!("{p}.proj_in.weight"), &[D, D])?);
            b1.extend(self.f32s(&format!("{p}.proj_in.bias"), &[D])?);
            w2.extend(self.f32s(&format!("{p}.layers.0.weight"), &[D, D])?);
            b2.extend(self.f32s(&format!("{p}.layers.0.bias"), &[D])?);
            w3.extend(self.f32s(&format!("{p}.proj_out.weight"), &[out, D])?);
            b3.extend(self.f32s(&format!("{p}.proj_out.bias"), &[out])?);
        }
        Ok(Mlp3W {
            w1: self.dev(&w1)?,
            b1: self.dev(&b1)?,
            w2: self.dev(&w2)?,
            b2: self.dev(&b2)?,
            w3: self.dev(&w3)?,
            b3: self.dev(&b3)?,
            out,
            per_row: prefixes.len() > 1,
        })
    }
}

impl Workspace {
    fn new(exec: &GpuExecutor) -> Result<Self, GpuModelError> {
        let t = GRID * GRID;
        let px2 = 4 * t;
        let px4 = 16 * t;
        let f = |n: usize| exec.alloc(n);
        let h = |n: usize| exec.alloc_f16(n);
        let f32s = 2 * MAX_SPARSE + MAX_SPARSE * D + 2 * MAX_T * D
            + 2 * t * D // keys, dense
            + t * D // up1
            + px2 * 4 * 32 // up2
            + MASKS * D + MASKS * 32 + MASKS
            + px2 * 4 // md1
            + px4 // mask_in
            + 2 * px4 * MASKS;
        let f16s = 2 * MAX_T * D
            + 2 * t * D
            + 5 * t * D
            + MAX_T * MLP
            + px2 * 64
            + px4 * 32
            + MASKS * 32
            + t * 16;
        Ok(Self {
            xy: f(2 * MAX_SPARSE)?,
            labels: exec.alloc_u32(MAX_SPARSE)?,
            sparse: f(MAX_SPARSE * D)?,
            qpe: f(MAX_T * D)?,
            q: f(MAX_T * D)?,
            q16: h(MAX_T * D)?,
            qq16: h(MAX_T * D)?,
            keys: f(t * D)?,
            k16: h(t * D)?,
            kq16: h(t * D)?,
            dense: f(t * D)?,
            pq16: h(t * D)?,
            pk16: h(t * D)?,
            pv16: h(t * D)?,
            att16: h(t * D)?,
            proj16: h(t * D)?,
            mlp16: h(MAX_T * MLP)?,
            up1: f(t * D)?,
            up1h: h(px2 * 64)?,
            up2: f(px2 * 4 * 32)?,
            up2h: h(px4 * 32)?,
            tok: f(MASKS * D)?,
            hyper: f(MASKS * 32)?,
            hyper16: HalfTensor {
                buf: h(MASKS * 32)?,
                dims: vec![32, MASKS],
            },
            iou: f(MASKS)?,
            obj: f(1)?,
            ptr: f(D)?,
            md1: f(px2 * 4)?,
            md2: h(t * 16)?,
            mask_in: f(px4)?,
            counts: exec.alloc_u32(2)?,
            lab: exec.alloc_u32(px4)?,
            area: exec.alloc_u32(px4)?,
            logits: f(px4 * MASKS)?,
            post: f(px4 * MASKS)?,
            bytes: (f32s * 4 + f16s * 2 + 2 * px4 * 4) as u64,
        })
    }
}

impl GpuSam3Pvs {
    /// Load the tracker heads from a `facebook/sam3` directory.
    pub fn load_dir(exec: Arc<GpuExecutor>, dir: &Path) -> Result<Self, GpuModelError> {
        if !exec.has_sam3_tracker() {
            return Err(GpuModelError::Unsupported(
                "this kernel pack predates SAM 3's click prompts (slots 778-783) - rebuild or \
                 update the pack"
                    .into(),
            ));
        }
        let st = super::checkpoint::open(dir)?;
        exec.vram_load_gate(64u64 << 20, "sam3 click prompts")
            .map_err(GpuModelError::WontFit)?;
        let mut r = Reader {
            st: &*st,
            exec: &exec,
            bytes: 0,
        };

        // ---- prompt encoder ----
        let gauss = {
            let v = r.f32s(
                &format!("{PE}.shared_embedding.positional_embedding"),
                &[2, D / 2],
            )?;
            r.dev(&v)?
        };
        let point_embed = {
            let v = r.f32s(&format!("{PE}.point_embed.weight"), &[4, D])?;
            r.dev(&v)?
        };
        let not_a_point = {
            let v = r.f32s(&format!("{PE}.not_a_point_embed.weight"), &[1, D])?;
            r.dev(&v)?
        };
        let no_mask = {
            let v = r.f32s(&format!("{PE}.no_mask_embed.weight"), &[1, D])?;
            r.dev(&v)?
        };
        let no_mem = {
            let v = r.f32s("tracker_model.no_memory_embedding", &[1, 1, D])?;
            r.dev(&v)?
        };
        let me = format!("{PE}.mask_embed");
        let mask_embed = MaskEmbed {
            w1: {
                let v = r.f32s(&format!("{me}.conv1.weight"), &[4, 1, 2, 2])?;
                r.dev(&v)?
            },
            b1: r.vec(&format!("{me}.conv1.bias"), 4)?,
            n1: r.norm(&format!("{me}.layer_norm1"), 4)?,
            w2: {
                let v = r.f32s(&format!("{me}.conv2.weight"), &[16, 4, 2, 2])?;
                r.dev(&v)?
            },
            b2: r.vec(&format!("{me}.conv2.bias"), 16)?,
            n2: r.norm(&format!("{me}.layer_norm2"), 16)?,
            c3: r.conv1(&format!("{me}.conv3"), 16, D)?,
        };

        // ---- decoder ----
        let own = {
            let mut v = r.f32s(&format!("{DEC}.obj_score_token.weight"), &[1, D])?;
            v.extend(r.f32s(&format!("{DEC}.iou_token.weight"), &[1, D])?);
            v.extend(r.f32s(&format!("{DEC}.mask_tokens.weight"), &[MASKS, D])?);
            r.dev(&v)?
        };
        let tp = format!("{DEC}.transformer");
        let mut layers = Vec::with_capacity(2);
        for i in 0..2 {
            let l = |s: &str| format!("{tp}.layers.{i}.{s}");
            layers.push(Layer {
                sa: r.pvs_attn(&l("self_attn"), D)?,
                n1: r.norm(&l("layer_norm1"), D)?,
                t2i: r.pvs_attn(&l("cross_attn_token_to_image"), CA)?,
                n2: r.norm(&l("layer_norm2"), D)?,
                fc1: r.pvs_lin(&l("mlp.proj_in"), MLP, D, 1.0)?,
                fc2: r.pvs_lin(&l("mlp.proj_out"), D, MLP, 1.0)?,
                n3: r.norm(&l("layer_norm3"), D)?,
                i2t: r.pvs_attn(&l("cross_attn_image_to_token"), CA)?,
                n4: r.norm(&l("layer_norm4"), D)?,
            });
        }
        let final_attn = r.pvs_attn(&format!("{tp}.final_attn_token_to_image"), CA)?;
        let final_norm = r.norm(&format!("{tp}.layer_norm_final_attn"), D)?;
        let up1 = r.convt2(&format!("{DEC}.upscale_conv1"), D, D / 4)?;
        let up_norm = r.norm(&format!("{DEC}.upscale_layer_norm"), D / 4)?;
        let up2 = r.convt2(&format!("{DEC}.upscale_conv2"), D / 4, D / 8)?;
        let hyper = r.pvs_mlp3(
            &(0..MASKS)
                .map(|i| format!("{DEC}.output_hypernetworks_mlps.{i}"))
                .collect::<Vec<_>>(),
            D / 8,
        )?;
        let iou_head = r.pvs_mlp3(&[format!("{DEC}.iou_prediction_head")], MASKS)?;
        let iou_sigmoid = !super::checkpoint::is_multiplex(r.st);
        let obj_head = r.pvs_mlp3(&[format!("{DEC}.pred_obj_score_head")], 1)?;
        let ptr_head = r.pvs_mlp3(&["tracker_model.object_pointer_proj".to_string()], D)?;
        // SAM 3's fixed vector; SAM 3.1 has none (it maps the pointer itself
        // through `no_obj_ptr_linear`), which only tracking reads
        let no_obj_ptr = match r.st.tensor("tracker_model.no_object_pointer") {
            Some(_) => {
                let v = r.f32s("tracker_model.no_object_pointer", &[1, D])?;
                Some(r.dev(&v)?)
            }
            None => None,
        };
        let gone = r.dev(&vec![NO_OBJ_LOGIT; SIDE * SIDE * MASKS])?;
        let weight_bytes = r.bytes;

        let ws = Workspace::new(&exec)?;
        // the dense image pe: the 72^2 pixel centres through the point
        // features with no label - a constant, built once
        let t = GRID * GRID;
        let centres: Vec<f32> = (0..t)
            .flat_map(|i| {
                let (y, x) = (i / GRID, i % GRID);
                [
                    (x as f32 + 0.5) / GRID as f32,
                    (y as f32 + 0.5) / GRID as f32,
                ]
            })
            .collect();
        let xy = exec.to_device(&centres)?;
        let mut dense_pe = exec.alloc(t * D)?;
        exec.sam3_point_pe(
            &xy,
            None,
            &gauss,
            &point_embed,
            &not_a_point,
            &mut dense_pe,
            t,
            D / 2,
        )?;
        exec.synchronize()?;
        drop(xy);
        tracing::info!(
            weights_mib = weight_bytes >> 20,
            workspace_mib = ws.bytes >> 20,
            "sam3 click prompts resident"
        );
        Ok(Self {
            exec,
            gauss,
            point_embed,
            not_a_point,
            no_mask,
            no_mem,
            mask_embed,
            dense_pe,
            own_tokens: own,
            layers,
            final_attn,
            final_norm,
            up1,
            up_norm,
            up2,
            hyper,
            iou_head,
            iou_sigmoid,
            obj_head,
            ptr_head,
            no_obj_ptr,
            gone,
            ws,
            weight_bytes,
        })
    }

    pub fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }
    pub fn workspace_bytes(&self) -> u64 {
        self.ws.bytes
    }

    /// The last call's hole-filled logits, `[288^2][4]` f32: the masks are
    /// its columns (the request path upsamples them to the picture).
    pub fn masks(&self) -> &CudaSlice<f32> {
        &self.ws.post
    }
    /// The last call's raw logits, `[288^2][4]` f32 - in tracking, with an
    /// absent object's at -1024: what the video's memory and outputs read.
    pub fn logits(&self) -> &CudaSlice<f32> {
        &self.ws.logits
    }
    /// The last call's object pointer, `[256]` f32.
    pub fn pointer(&self) -> &CudaSlice<f32> {
        &self.ws.ptr
    }
    /// Meta's `no_object_pointer`, `[256]` f32 (what an empty mask-as-output
    /// pass hands the bank). SAM 3.1 has no fixed one.
    pub fn no_object_pointer(&self) -> Result<&CudaSlice<f32>, GpuModelError> {
        self.no_obj_ptr.as_ref().ok_or_else(|| {
            GpuModelError::Unsupported(
                "SAM 3.1 has no fixed no-object pointer (its tracker is not in this build)".into(),
            )
        })
    }
    /// The last call's raw logits, `[288^2][4]` (the gate's view).
    pub fn read_logits(&self) -> Result<Vec<f32>, GpuModelError> {
        Ok(self
            .exec
            .to_host_len(&self.ws.logits, SIDE * SIDE * MASKS)?)
    }

    /// One attention of the two-way transformer: `nq` rows of `q_in` over
    /// `nk` rows of `k_in` / `v_in` (f16 `[rows][256]`), projected to `inner`
    /// width, landing `proj16 = o_proj(attention)` without its bias (the seam
    /// after adds it).
    #[allow(clippy::too_many_arguments)]
    fn attend(
        exec: &GpuExecutor,
        a: &Attn,
        q_in: &CudaSlice<f16>,
        k_in: &CudaSlice<f16>,
        v_in: &CudaSlice<f16>,
        s: Scratch<'_>,
        nq: usize,
        nk: usize,
        inner: usize,
    ) -> Result<(), GpuModelError> {
        exec.matvec_batch_f16_h_bias(&a.q.w, q_in, s.pq, Some(&a.q.b), nq)?;
        exec.matvec_batch_f16_h_bias(&a.k.w, k_in, s.pk, Some(&a.k.b), nk)?;
        exec.matvec_batch_f16_h_bias(&a.v.w, v_in, s.pv, Some(&a.v.b), nk)?;
        exec.vision_attn_h(s.pq, s.pk, s.pv, s.att, nq, nk, HEADS, inner / HEADS, 1)?;
        exec.matvec_batch_f16_h(&a.o.w, s.att, s.proj, nq)?;
        Ok(())
    }

    /// Run the heads for one prompt over the picture `vision` last encoded
    /// with its tracker neck.
    pub fn predict(
        &mut self,
        vision: &GpuSam3Vision,
        prompt: &PvsPrompt<'_>,
    ) -> Result<PvsResult, GpuModelError> {
        if prompt.points.is_empty() && prompt.bbox.is_none() && prompt.mask.is_none() {
            return Err(GpuModelError::Unsupported(
                "sam3: a click prompt needs a click, a box or a mask".into(),
            ));
        }
        self.predict_on(PvsFeatures::picture(vision), prompt)
    }

    /// [`Self::predict`] over explicit features - a video frame's, in
    /// tracking (see [`PvsFeatures`]).
    pub fn predict_on(
        &mut self,
        feats: PvsFeatures<'_>,
        prompt: &PvsPrompt<'_>,
    ) -> Result<PvsResult, GpuModelError> {
        let exec = self.exec.clone();
        if prompt.points.len() > PVS_MAX_POINTS {
            return Err(GpuModelError::Unsupported(format!(
                "sam3: {} clicks (at most {PVS_MAX_POINTS})",
                prompt.points.len()
            )));
        }
        let t0 = GRID * GRID;
        if feats.feat.len() < t0 * D
            || feats.s0.len() < SIDE * SIDE * D / 8
            || feats.s1.len() < (SIDE / 2) * (SIDE / 2) * D / 4
        {
            return Err(GpuModelError::Unsupported(
                "sam3: tracker features under the heads' geometry".into(),
            ));
        }
        let t = GRID * GRID;
        let px4 = SIDE * SIDE;
        let ws = &mut self.ws;

        // ---- sparse prompt: box corners (2, 3), clicks (1 / 0), one pad (-1) ----
        let mut xy = Vec::with_capacity(2 * MAX_SPARSE);
        let mut labels = Vec::with_capacity(MAX_SPARSE);
        if let Some([x0, y0, x1, y1]) = prompt.bbox {
            xy.extend([x0, y0, x1, y1]);
            labels.extend([2u32, 3]);
        }
        for p in &prompt.points {
            xy.extend([p.x, p.y]);
            labels.push(u32::from(p.positive));
        }
        // the padding point sits at (0 + 0.5) / 1008 like Meta's: its pe is
        // replaced anyway, the coordinate only has to be finite. A tracker
        // call with no clicks and no box carries TWO: `_forward_sam_heads`
        // stands in a dummy point labelled -1, and the prompt encoder then
        // pads it like any click list
        let pads = if feats.track && prompt.points.is_empty() && prompt.bbox.is_none() {
            2
        } else {
            1
        };
        for _ in 0..pads {
            xy.extend([0.5 / 1008.0, 0.5 / 1008.0]);
            labels.push(u32::MAX); // -1
        }
        let ns = labels.len();
        let tt = OWN + ns;
        exec.upload_f32(&xy, &mut ws.xy)?;
        exec.upload_u32(&labels, &mut ws.labels)?;
        exec.sam3_point_pe(
            &ws.xy,
            Some(&ws.labels),
            &self.gauss,
            &self.point_embed,
            &self.not_a_point,
            &mut ws.sparse,
            ns,
            D / 2,
        )?;

        // ---- the mask prompt (before the logits it may read are rewritten) ----
        let dense_from_mask = match prompt.mask {
            None => false,
            Some(m) => {
                let (src, stride, off): (&CudaSlice<f32>, usize, usize) = match m {
                    PvsMask::Last(k) => (&ws.logits, MASKS, k.min(MASKS - 1)),
                    PvsMask::Host(plane) => {
                        if plane.len() != px4 {
                            return Err(GpuModelError::Unsupported(format!(
                                "sam3: a {} value mask prompt (want {px4})",
                                plane.len()
                            )));
                        }
                        exec.upload_f32(plane, &mut ws.mask_in)?;
                        (&ws.mask_in, 1, 0)
                    }
                    PvsMask::Device(plane, off) => {
                        if plane.len() < off + px4 {
                            return Err(GpuModelError::Unsupported(
                                "sam3: a device mask prompt under 288^2".into(),
                            ));
                        }
                        exec.copy_region(plane, off, &mut ws.mask_in, 0, px4)?;
                        (&ws.mask_in, 1, 0)
                    }
                };
                let me = &self.mask_embed;
                exec.sam3_mask_down(
                    src,
                    &me.w1,
                    &me.b1,
                    (&me.n1.w, &me.n1.b),
                    Sam3MaskDownOut::F32(&mut ws.md1),
                    SIDE,
                    SIDE,
                    1,
                    4,
                    stride,
                    off,
                    MASK_CLAMP,
                    LN2D_EPS,
                )?;
                exec.sam3_mask_down(
                    &ws.md1,
                    &me.w2,
                    &me.b2,
                    (&me.n2.w, &me.n2.b),
                    Sam3MaskDownOut::F16(&mut ws.md2),
                    SIDE / 2,
                    SIDE / 2,
                    4,
                    16,
                    4,
                    0,
                    0.0,
                    LN2D_EPS,
                )?;
                exec.matvec_batch_f16(&me.c3.w, &ws.md2, &mut ws.dense, t)?;
                exec.bias_add(&mut ws.dense, &me.c3.b, t, D)?;
                true
            }
        };

        // ---- tokens: [own 6] + sparse; the entering tokens are the query pe ----
        exec.copy_region(&self.own_tokens, 0, &mut ws.qpe, 0, OWN * D)?;
        exec.copy_region(&ws.sparse, 0, &mut ws.qpe, OWN * D, ns * D)?;
        exec.copy_region(&ws.qpe, 0, &mut ws.q, 0, tt * D)?;
        exec.sam3_seam_h(
            &mut ws.q,
            None,
            None,
            None,
            Some((&ws.qpe, tt, tt)),
            Some(&mut ws.q16),
            Some(&mut ws.qq16),
            tt,
            D,
            LN_EPS,
            false,
        )?;

        // ---- keys: the 72^2 feature (+ no_memory) + the dense prompt ----
        exec.copy_slice(feats.feat, 0, t * D, &mut ws.keys)?;
        if feats.no_memory {
            exec.add_rows_bcast(&mut ws.keys, &self.no_mem, t, 1, D)?;
        }
        if dense_from_mask {
            exec.add_rows_bcast(&mut ws.keys, &ws.dense, t, t, D)?;
        } else {
            exec.add_rows_bcast(&mut ws.keys, &self.no_mask, t, 1, D)?;
        }
        exec.sam3_seam_h(
            &mut ws.keys,
            None,
            None,
            None,
            Some((&self.dense_pe, t, t)),
            Some(&mut ws.k16),
            Some(&mut ws.kq16),
            t,
            D,
            LN_EPS,
            false,
        )?;

        // ---- the two-way transformer ----
        macro_rules! scratch {
            () => {
                Scratch {
                    pq: &mut ws.pq16,
                    pk: &mut ws.pk16,
                    pv: &mut ws.pv16,
                    att: &mut ws.att16,
                    proj: &mut ws.proj16,
                }
            };
        }
        for (li, l) in self.layers.iter().enumerate() {
            // self-attention; layer 0 without positions and without residual
            if li == 0 {
                Self::attend(
                    &exec,
                    &l.sa,
                    &ws.q16,
                    &ws.q16,
                    &ws.q16,
                    scratch!(),
                    tt,
                    tt,
                    D,
                )?;
                exec.zero_region(&mut ws.q, 0, tt * D)?;
            } else {
                Self::attend(
                    &exec,
                    &l.sa,
                    &ws.qq16,
                    &ws.qq16,
                    &ws.q16,
                    scratch!(),
                    tt,
                    tt,
                    D,
                )?;
            }
            Self::seam_q(&exec, ws, &l.sa.o.b, &l.n1, tt)?;
            // tokens -> image
            Self::attend(
                &exec,
                &l.t2i,
                &ws.qq16,
                &ws.kq16,
                &ws.k16,
                scratch!(),
                tt,
                t,
                CA,
            )?;
            Self::seam_q(&exec, ws, &l.t2i.o.b, &l.n2, tt)?;
            // MLP
            exec.matvec_batch_f16_h_relu(&l.fc1.w, &ws.q16, &mut ws.mlp16, Some(&l.fc1.b), tt)?;
            exec.matvec_batch_f16_h(&l.fc2.w, &ws.mlp16, &mut ws.proj16, tt)?;
            Self::seam_q(&exec, ws, &l.fc2.b, &l.n3, tt)?;
            // image -> tokens: the 5184 image rows attend to the tokens
            Self::attend(
                &exec,
                &l.i2t,
                &ws.kq16,
                &ws.qq16,
                &ws.q16,
                scratch!(),
                t,
                tt,
                CA,
            )?;
            exec.sam3_seam_h(
                &mut ws.keys,
                Some(&ws.proj16),
                Some(&l.i2t.o.b),
                Some((&l.n4.w, &l.n4.b)),
                Some((&self.dense_pe, t, t)),
                Some(&mut ws.k16),
                Some(&mut ws.kq16),
                t,
                D,
                LN_EPS,
                true,
            )?;
        }
        Self::attend(
            &exec,
            &self.final_attn,
            &ws.qq16,
            &ws.kq16,
            &ws.k16,
            scratch!(),
            tt,
            t,
            CA,
        )?;
        Self::seam_q(&exec, ws, &self.final_attn.o.b, &self.final_norm, tt)?;

        // ---- upscaling: 72 -> 144 (+ conv_s1, LN2d, GELU) -> 288 (+ conv_s0, GELU) ----
        exec.matvec_batch_f16(&self.up1.w, &ws.k16, &mut ws.up1, t)?;
        exec.sam3_up_skip(
            &ws.up1,
            &self.up1.b,
            feats.s1,
            Some((&self.up_norm.w, &self.up_norm.b)),
            &mut ws.up1h,
            GRID,
            GRID,
            D / 4,
            LN2D_EPS,
        )?;
        exec.matvec_batch_f16(&self.up2.w, &ws.up1h, &mut ws.up2, 4 * t)?;
        exec.sam3_up_skip(
            &ws.up2,
            &self.up2.b,
            feats.s0,
            None,
            &mut ws.up2h,
            2 * GRID,
            2 * GRID,
            D / 8,
            LN2D_EPS,
        )?;

        // ---- heads: hypernetwork rows -> the mask GEMM's weight; IoU ----
        exec.copy_region(&ws.q, 2 * D, &mut ws.tok, 0, MASKS * D)?;
        exec.sam3_mlp3_rows(
            &ws.tok,
            &self.hyper.view(),
            &mut ws.hyper,
            Some(&mut ws.hyper16.buf),
            MASKS,
            false,
        )?;
        exec.matvec_batch_f16(&ws.hyper16, &ws.up2h, &mut ws.logits, px4)?;
        exec.copy_region(&ws.q, D, &mut ws.tok, 0, D)?;
        exec.sam3_mlp3_rows(
            &ws.tok,
            &self.iou_head.view(),
            &mut ws.iou,
            None,
            1,
            self.iou_sigmoid,
        )?;
        exec.copy_region(&ws.q, 0, &mut ws.tok, 0, D)?;
        exec.sam3_mlp3_rows(&ws.tok, &self.obj_head.view(), &mut ws.obj, None, 1, false)?;
        let iou_v = exec.to_host_len(&ws.iou, MASKS)?;
        let iou = [iou_v[0], iou_v[1], iou_v[2], iou_v[3]];
        let object_logit = exec.to_host_len(&ws.obj, 1)?[0];
        let object_score = 1.0 / (1.0 + (-object_logit).exp());

        // ---- Meta's output choice ----
        let best_multi = (1..MASKS)
            .max_by(|&a, &b| iou[a].total_cmp(&iou[b]).then(b.cmp(&a)))
            .unwrap_or(1);
        let mut candidates: Vec<(usize, f32)> = if prompt.multimask {
            (1..MASKS).map(|k| (k, iou[k])).collect()
        } else {
            exec.sam3_mask_stats(&ws.logits, &mut ws.counts, px4, MASKS, 0, STABILITY_DELTA)?;
            let c = exec.to_host_u32(&ws.counts)?;
            let stability = if c[1] > 0 {
                c[0] as f64 / c[1] as f64
            } else {
                1.0
            };
            if stability >= STABILITY_THRESH {
                vec![(0, iou[0])]
            } else {
                vec![(best_multi, iou[best_multi])]
            }
        };
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));

        // ---- the object pointer: the token of the best of 1..3 with three
        // candidates, mask 0's with one (Meta's decoder hands back token 0
        // whatever mask the stability check chose) ----
        let pointer_mask = if prompt.multimask { best_multi } else { 0 };
        let present = object_logit > 0.0;
        if present {
            exec.copy_region(&ws.q, (2 + pointer_mask) * D, &mut ws.tok, 0, D)?;
            exec.sam3_mlp3_rows(&ws.tok, &self.ptr_head.view(), &mut ws.ptr, None, 1, false)?;
        } else if let Some(no) = &self.no_obj_ptr {
            exec.copy_region(no, 0, &mut ws.ptr, 0, D)?;
        } else if feats.track {
            // a picture never reads the pointer; tracking with SAM 3.1 is 7c
            return Err(GpuModelError::Unsupported(
                "SAM 3.1's tracking pointer is not in this build".into(),
            ));
        }
        if feats.track {
            if !present {
                exec.copy_region(&self.gone, 0, &mut ws.logits, 0, px4 * MASKS)?;
            }
            return Ok(PvsResult {
                candidates,
                iou,
                object_score,
                object_logit,
                pointer_mask,
            });
        }

        // ---- the hole fill, on a copy (the raw logits are the next prompt) ----
        exec.copy_slice(&ws.logits, 0, px4 * MASKS, &mut ws.post)?;
        for &(k, _) in &candidates {
            exec.sam3_fill_holes(
                &mut ws.post,
                &mut ws.lab,
                &mut ws.area,
                SIDE,
                MASKS,
                k,
                MAX_HOLE_AREA,
            )?;
        }
        Ok(PvsResult {
            candidates,
            iou,
            object_score,
            object_logit,
            pointer_mask,
        })
    }

    /// The token-side seam after an attention or the MLP (post-norm): the
    /// residual stream, its f16 view and the view plus the query pe.
    fn seam_q(
        exec: &GpuExecutor,
        ws: &mut Workspace,
        bias: &CudaSlice<f32>,
        n: &Norm,
        tt: usize,
    ) -> Result<(), GpuModelError> {
        exec.sam3_seam_h(
            &mut ws.q,
            Some(&ws.proj16),
            Some(bias),
            Some((&n.w, &n.b)),
            Some((&ws.qpe, tt, tt)),
            Some(&mut ws.q16),
            Some(&mut ws.qq16),
            tt,
            D,
            LN_EPS,
            true,
        )?;
        Ok(())
    }
}

/// The planes one attention writes: its three projections, the attention
/// and the output projection.
struct Scratch<'a> {
    pq: &'a mut CudaSlice<f16>,
    pk: &'a mut CudaSlice<f16>,
    pv: &'a mut CudaSlice<f16>,
    att: &'a mut CudaSlice<f16>,
    proj: &'a mut CudaSlice<f16>,
}
