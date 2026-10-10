//! Query selection, the six-layer deformable decoder and the heads.
//!
//! - Memory: the three encoder levels through `decoder_input_proj` (1 x 1 +
//!   BatchNorm), flattened row-major and concatenated level-major (100^2,
//!   50^2, 25^2 -> 13125 rows).
//! - Selection: `enc_output` (Linear + LayerNorm) over the memory times the
//!   anchors' valid mask, the shared class head over it, the 300 rows of
//!   highest best-class score (host top-k over 13125 maxima - selection
//!   logic, like sampling) gathered as the decoder's first hidden states.
//! - Initial boxes: the box of each selected query's mask (`mask_query_head`
//!   of its `decoder_norm`, dotted with the 32 prototypes, pixels > 0) -
//!   RT-DETR's mask-enhanced initialisation; the encoder box head is unused.
//! - Six post-norm layers: self-attention (q = k = h + pos(ref), v = h), the
//!   deformable cross-attention over the memory, FFN; then the shared box head
//!   refines ref = sigmoid(delta + inverse_sigmoid(ref)).
//! - Heads, last layer only: classes and masks off `decoder_norm(h)`; the
//!   reading order in the exported graph's form - `decoder_order_head.5` and
//!   the global pointer over `h` BEFORE the norm, P = sigmoid(s - s^T), votes
//!   down the columns.

use cudarc::driver::CudaSlice;

use super::load::{Names, Reader};
use super::{ConvBn, EncoderOut, GpuModelError, Plane};
use crate::gpu::{GpuExecutor, HalfTensor};

const D: usize = 256;
const HEADS: usize = 8;
const FFN: usize = 1024;
const CLASSES: usize = 25;
const PROTOS: usize = 32;
/// queries the decoder keeps
pub const QUERIES: usize = 300;
const LN_EPS: f32 = 1e-5;
const LEVELS: [(usize, usize); 3] = [(100, 100), (50, 50), (25, 25)];
const MEM: usize = 100 * 100 + 50 * 50 + 25 * 25;

type Lin = (HalfTensor, CudaSlice<f32>);
type Ln = (CudaSlice<f32>, CudaSlice<f32>);

struct Layer {
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    ln1: Ln,
    off: Lin,
    aw: Lin,
    value: Lin,
    out: Lin,
    ln2: Ln,
    fc1: Lin,
    fc2: Lin,
    ln3: Ln,
}

pub(super) struct Decoder {
    in_proj: [ConvBn; 3],
    /// the anchors' valid mask, 0 / 1 per memory row
    valid: CudaSlice<f32>,
    enc_out: Lin,
    enc_ln: Ln,
    score: Lin,
    bbox: [Lin; 3],
    mask_q: [Lin; 3],
    dec_norm: Ln,
    qpos: [Lin; 2],
    layers: Vec<Layer>,
    order_head: Lin,
    /// the global pointer split into its query and key halves, the query
    /// half times 1/sqrt(64) (exact: a power of two)
    ptr_q: Lin,
    ptr_k: Lin,
}

/// What the network hands the postprocess, plus the taps the gate reads.
pub struct DecoderOut {
    /// memory rows the selection kept, in rank order
    pub topk: Vec<u32>,
    /// f32 [300][4] (cx, cy, w, h) in [0, 1], the last layer's boxes
    pub boxes: Vec<f32>,
    /// f32 [300][25] class logits
    pub logits: Vec<f32>,
    /// f32 [300] reading-order votes (smaller is earlier)
    pub votes: Vec<f32>,
    /// f32 [300][200 * 200] mask logits, on the device
    pub masks: CudaSlice<f32>,
    /// the initial reference boxes, f32 [300][4]
    pub init_ref: Vec<f32>,
    /// each layer's output hidden states, f32 [300][256]
    pub layers: Vec<Vec<f32>>,
}

/// HF's anchor valid mask: per level cell (cx, cy) = ((x, y) + 0.5) / size
/// and w = h = 0.05 * 2^l in f32; valid where all four lie strictly in
/// (f32 0.01, f32 0.99) - 12,533 of the 13,125 rows at 800 x 800.
fn valid_mask() -> Vec<f32> {
    let (lo, hi) = (0.01f32, 1.0f32 - 0.01f32);
    let mut v = Vec::with_capacity(MEM);
    for (l, &(h, w)) in LEVELS.iter().enumerate() {
        let wh = 0.05f32 * 2f32.powi(l as i32);
        for y in 0..h {
            for x in 0..w {
                let cx = (x as f32 + 0.5) / w as f32;
                let cy = (y as f32 + 0.5) / h as f32;
                let ok = [cx, cy, wh, wh].iter().all(|&a| a > lo && a < hi);
                v.push(if ok { 1.0 } else { 0.0 });
            }
        }
    }
    v
}

impl Decoder {
    pub(super) fn load(rd: &mut Reader) -> Result<Self, GpuModelError> {
        let lin = |rd: &mut Reader, p: &str, io: (usize, usize)| rd.linear(p, io, 1.0);
        let mlp3 = |rd: &mut Reader, p: &str, out: usize| -> Result<[Lin; 3], GpuModelError> {
            Ok([
                lin(rd, &format!("{p}.layers.0"), (D, D))?,
                lin(rd, &format!("{p}.layers.1"), (D, D))?,
                lin(rd, &format!("{p}.layers.2"), (D, out))?,
            ])
        };
        let qscale = 1.0 / ((D / HEADS) as f32).sqrt();
        let mut layers = Vec::new();
        for i in 0..6 {
            let p = format!("model.decoder.layers.{i}");
            layers.push(Layer {
                q: rd.linear(&format!("{p}.self_attn.q_proj"), (D, D), qscale)?,
                k: lin(rd, &format!("{p}.self_attn.k_proj"), (D, D))?,
                v: lin(rd, &format!("{p}.self_attn.v_proj"), (D, D))?,
                o: lin(rd, &format!("{p}.self_attn.out_proj"), (D, D))?,
                ln1: rd.norm(&format!("{p}.self_attn_layer_norm"), D)?,
                off: lin(rd, &format!("{p}.encoder_attn.sampling_offsets"), (D, 192))?,
                aw: lin(rd, &format!("{p}.encoder_attn.attention_weights"), (D, 96))?,
                value: lin(rd, &format!("{p}.encoder_attn.value_proj"), (D, D))?,
                out: lin(rd, &format!("{p}.encoder_attn.output_proj"), (D, D))?,
                ln2: rd.norm(&format!("{p}.encoder_attn_layer_norm"), D)?,
                fc1: lin(rd, &format!("{p}.fc1"), (D, FFN))?,
                fc2: lin(rd, &format!("{p}.fc2"), (FFN, D))?,
                ln3: rd.norm(&format!("{p}.final_layer_norm"), D)?,
            });
        }
        let (ptr_q, ptr_k) = rd.pointer("model.decoder_global_pointer.dense", D, 64)?;
        Ok(Self {
            in_proj: [
                rd.conv_bn("model.decoder_input_proj.0", Names::Seq, (D, D, 1, 1))?,
                rd.conv_bn("model.decoder_input_proj.1", Names::Seq, (D, D, 1, 1))?,
                rd.conv_bn("model.decoder_input_proj.2", Names::Seq, (D, D, 1, 1))?,
            ],
            valid: rd.dev(&valid_mask())?,
            enc_out: lin(rd, "model.enc_output.0", (D, D))?,
            enc_ln: rd.norm("model.enc_output.1", D)?,
            score: lin(rd, "model.enc_score_head", (D, CLASSES))?,
            bbox: mlp3(rd, "model.enc_bbox_head", 4)?,
            mask_q: mlp3(rd, "model.mask_query_head", PROTOS)?,
            dec_norm: rd.norm("model.decoder_norm", D)?,
            qpos: [
                rd.linear_padded("model.decoder.query_pos_head.layers.0", (4, 512), 8)?,
                lin(rd, "model.decoder.query_pos_head.layers.1", (512, D))?,
            ],
            layers,
            order_head: lin(rd, "model.decoder_order_head.5", (D, D))?,
            ptr_q,
            ptr_k,
        })
    }

    pub(super) fn forward(
        &self,
        exec: &GpuExecutor,
        enc: &EncoderOut,
    ) -> Result<DecoderOut, GpuModelError> {
        let q = QUERIES;
        // memory: the three levels projected and stacked level-major
        let mut mem = exec.alloc_f16(MEM * D)?;
        let mut at = 0;
        for (lvl, proj) in enc.levels.iter().zip(&self.in_proj) {
            let p = proj.apply(exec, lvl, super::Act::None, 0)?;
            let n = p.h * p.w * D;
            exec.copy_region_f16(&p.data, 0, &mut mem, at, n)?;
            at += n;
        }
        // selection over the valid-masked memory
        let mut masked = exec.alloc_f16(MEM * D)?;
        exec.copy_region_f16(&mem, 0, &mut masked, 0, MEM * D)?;
        exec.dl_rowscale_h(&mut masked, &self.valid, MEM, D)?;
        let om = lin_ln(exec, &masked, &self.enc_out, &self.enc_ln, MEM)?;
        let om16 = to16(exec, &om, MEM * D)?;
        let cls = lin32(exec, &om16, &self.score, MEM)?;
        let topk = top_rows(&exec.to_host(&cls)?, CLASSES, q);
        let idx = exec.to_device_u32(&topk)?;
        let mut h = exec.alloc(q * D)?;
        exec.dl_gather_rows_f32(&om, &mut h, &idx, q, D)?;
        // initial boxes from the selected queries' masks
        let (_, masks0) = self.masks(exec, &h, &enc.mask_feat)?;
        let mut ref32 = exec.alloc(q * 4)?;
        exec.dl_mask_ref(&masks0, &mut ref32, q, (enc.mask_feat.h, enc.mask_feat.w))?;
        let init_ref = exec.to_host(&ref32)?;
        let mut ref16 = exec.alloc_f16(q * 8)?;
        exec.dl_ref_step(&mut ref32, None, &mut ref16, q)?;
        let mut taps = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            // query position from the current boxes
            let mut p1 = exec.alloc_f16(q * 512)?;
            exec.matvec_batch_f16_h_relu(
                &self.qpos[0].0,
                &ref16,
                &mut p1,
                Some(&self.qpos[0].1),
                q,
            )?;
            let pos = lin32(exec, &p1, &self.qpos[1], q)?;
            h = layer.forward(exec, &h, &pos, &mem, &ref32)?;
            taps.push(exec.to_host(&h)?);
            // refine the boxes with the shared box head
            let h16 = to16(exec, &h, q * D)?;
            let delta = mlp3(exec, &h16, &self.bbox, q)?;
            exec.dl_ref_step(&mut ref32, Some(&delta), &mut ref16, q)?;
        }
        // heads: classes and masks off decoder_norm(h)
        let (oq16, masks) = self.masks(exec, &h, &enc.mask_feat)?;
        let logits = exec.to_host(&lin32(exec, &oq16, &self.score, q)?)?;
        // reading order off h itself (the exported graph), not the normed copy
        let h16 = to16(exec, &h, q * D)?;
        let mut z = exec.alloc_f16(q * D)?;
        exec.matvec_batch_f16_h_bias(
            &self.order_head.0,
            &h16,
            &mut z,
            Some(&self.order_head.1),
            q,
        )?;
        let mut pq = exec.alloc_f16(q * 64)?;
        let mut pk = exec.alloc_f16(q * 64)?;
        exec.matvec_batch_f16_h_bias(&self.ptr_q.0, &z, &mut pq, Some(&self.ptr_q.1), q)?;
        exec.matvec_batch_f16_h_bias(&self.ptr_k.0, &z, &mut pk, Some(&self.ptr_k.1), q)?;
        let kt = HalfTensor {
            buf: pk,
            dims: vec![64, q],
        };
        let mut s = exec.alloc(q * q)?;
        exec.matvec_batch_f16(&kt, &pq, &mut s, q)?;
        let mut votes = exec.alloc(q)?;
        exec.dl_order_votes(&s, &mut votes, q)?;
        Ok(DecoderOut {
            topk,
            boxes: exec.to_host(&ref32)?,
            logits,
            votes: exec.to_host(&votes)?,
            masks,
            init_ref,
            layers: taps,
        })
    }

    /// `decoder_norm(h)` as halves, and its masks: the mask head's 32
    /// coefficients dotted with every prototype pixel, f32 [q][h * w].
    fn masks(
        &self,
        exec: &GpuExecutor,
        h: &CudaSlice<f32>,
        mf: &Plane,
    ) -> Result<(CudaSlice<half::f16>, CudaSlice<f32>), GpuModelError> {
        let q = QUERIES;
        let mut n = exec.alloc(q * D)?;
        exec.layernorm(h, &self.dec_norm.0, &self.dec_norm.1, &mut n, q, D, LN_EPS)?;
        let n16 = to16(exec, &n, q * D)?;
        let mut a = exec.alloc_f16(q * D)?;
        exec.matvec_batch_f16_h_relu(&self.mask_q[0].0, &n16, &mut a, Some(&self.mask_q[0].1), q)?;
        let mut b = exec.alloc_f16(q * D)?;
        exec.matvec_batch_f16_h_relu(&self.mask_q[1].0, &a, &mut b, Some(&self.mask_q[1].1), q)?;
        let mut mq = exec.alloc_f16(q * PROTOS)?;
        exec.matvec_batch_f16_h_bias(&self.mask_q[2].0, &b, &mut mq, Some(&self.mask_q[2].1), q)?;
        // the prototype plane [pixels][32] is a GEMM weight [out][in] as it lies
        let protos = HalfTensor {
            buf: mf.data.clone(),
            dims: vec![PROTOS, mf.h * mf.w],
        };
        let mut m = exec.alloc(q * mf.h * mf.w)?;
        exec.matvec_batch_f16(&protos, &mq, &mut m, q)?;
        Ok((n16, m))
    }
}

impl Layer {
    fn forward(
        &self,
        exec: &GpuExecutor,
        h: &CudaSlice<f32>,
        pos: &CudaSlice<f32>,
        mem: &CudaSlice<half::f16>,
        ref32: &CudaSlice<f32>,
    ) -> Result<CudaSlice<f32>, GpuModelError> {
        let q = QUERIES;
        let n = q * D;
        // self-attention, q = k = h + pos, v = h
        let hp16 = add16(exec, h, pos, n)?;
        let h16 = to16(exec, h, n)?;
        let (mut qq, mut kk, mut vv) = (exec.alloc_f16(n)?, exec.alloc_f16(n)?, exec.alloc_f16(n)?);
        exec.matvec_batch_f16_h_bias(&self.q.0, &hp16, &mut qq, Some(&self.q.1), q)?;
        exec.matvec_batch_f16_h_bias(&self.k.0, &hp16, &mut kk, Some(&self.k.1), q)?;
        exec.matvec_batch_f16_h_bias(&self.v.0, &h16, &mut vv, Some(&self.v.1), q)?;
        let mut a = exec.alloc_f16(n)?;
        exec.vision_attn_h(&qq, &kk, &vv, &mut a, q, q, HEADS, D / HEADS, 1)?;
        let h1 = res_ln(exec, &a, &self.o, h, &self.ln1, q)?;
        // deformable cross-attention, query h + pos
        let hq16 = add16(exec, &h1, pos, n)?;
        let off = lin32(exec, &hq16, &self.off, q)?;
        let aw = lin32(exec, &hq16, &self.aw, q)?;
        let mut value = exec.alloc_f16(MEM * D)?;
        exec.matvec_batch_f16_h_bias(&self.value.0, mem, &mut value, Some(&self.value.1), MEM)?;
        let mut m = exec.alloc_f16(n)?;
        exec.dl_msda(&value, &off, &aw, ref32, &mut m, q, LEVELS)?;
        let h2 = res_ln(exec, &m, &self.out, &h1, &self.ln2, q)?;
        // FFN
        let h16 = to16(exec, &h2, n)?;
        let mut f = exec.alloc_f16(q * FFN)?;
        exec.matvec_batch_f16_h_relu(&self.fc1.0, &h16, &mut f, Some(&self.fc1.1), q)?;
        res_ln(exec, &f, &self.fc2, &h2, &self.ln3, q)
    }
}

/// `LayerNorm(residual + (x W + b))`, f32 out.
fn res_ln(
    exec: &GpuExecutor,
    x16: &CudaSlice<half::f16>,
    w: &Lin,
    residual: &CudaSlice<f32>,
    ln: &Ln,
    rows: usize,
) -> Result<CudaSlice<f32>, GpuModelError> {
    let mut y = lin32(exec, x16, w, rows)?;
    exec.add(&mut y, residual, rows * D)?;
    let mut out = exec.alloc(rows * D)?;
    exec.layernorm(&y, &ln.0, &ln.1, &mut out, rows, D, LN_EPS)?;
    Ok(out)
}

/// `LayerNorm(x W + b)` over f16 rows, f32 out.
fn lin_ln(
    exec: &GpuExecutor,
    x16: &CudaSlice<half::f16>,
    w: &Lin,
    ln: &Ln,
    rows: usize,
) -> Result<CudaSlice<f32>, GpuModelError> {
    let y = lin32(exec, x16, w, rows)?;
    let mut out = exec.alloc(rows * D)?;
    exec.layernorm(&y, &ln.0, &ln.1, &mut out, rows, D, LN_EPS)?;
    Ok(out)
}

/// `x W + b` with an f32 landing.
fn lin32(
    exec: &GpuExecutor,
    x16: &CudaSlice<half::f16>,
    w: &Lin,
    rows: usize,
) -> Result<CudaSlice<f32>, GpuModelError> {
    let out = w.0.dims[1];
    let mut y = exec.alloc(rows * out)?;
    exec.matvec_batch_f16(&w.0, x16, &mut y, rows)?;
    exec.bias_add(&mut y, &w.1, rows, out)?;
    Ok(y)
}

/// The three-layer ReLU MLP (`MLPPredictionHead`), f32 out.
fn mlp3(
    exec: &GpuExecutor,
    x16: &CudaSlice<half::f16>,
    w: &[Lin; 3],
    rows: usize,
) -> Result<CudaSlice<f32>, GpuModelError> {
    let mut a = exec.alloc_f16(rows * w[0].0.dims[1])?;
    exec.matvec_batch_f16_h_relu(&w[0].0, x16, &mut a, Some(&w[0].1), rows)?;
    let mut b = exec.alloc_f16(rows * w[1].0.dims[1])?;
    exec.matvec_batch_f16_h_relu(&w[1].0, &a, &mut b, Some(&w[1].1), rows)?;
    lin32(exec, &b, &w[2], rows)
}

fn to16(
    exec: &GpuExecutor,
    x: &CudaSlice<f32>,
    n: usize,
) -> Result<CudaSlice<half::f16>, GpuModelError> {
    let mut y = exec.alloc_f16(n)?;
    exec.convert_f32_f16(x, &mut y, n)?;
    Ok(y)
}

/// `f16(a + b)` of two f32 planes.
fn add16(
    exec: &GpuExecutor,
    a: &CudaSlice<f32>,
    b: &CudaSlice<f32>,
    n: usize,
) -> Result<CudaSlice<half::f16>, GpuModelError> {
    let mut s = exec.alloc(n)?;
    exec.copy_region(a, 0, &mut s, 0, n)?;
    exec.add(&mut s, b, n)?;
    to16(exec, &s, n)
}

/// The `k` rows of highest best-column score, descending - `torch.topk` of
/// the per-row maxima; ties keep the lower row first.
fn top_rows(scores: &[f32], cols: usize, k: usize) -> Vec<u32> {
    let best: Vec<f32> = scores
        .chunks_exact(cols)
        .map(|r| r.iter().copied().fold(f32::NEG_INFINITY, f32::max))
        .collect();
    let mut idx: Vec<u32> = (0..best.len() as u32).collect();
    idx.sort_by(|&a, &b| {
        best[b as usize]
            .total_cmp(&best[a as usize])
            .then(a.cmp(&b))
    });
    idx.truncate(k);
    idx
}
