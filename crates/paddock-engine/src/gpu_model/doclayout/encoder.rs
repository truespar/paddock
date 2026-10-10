//! The hybrid encoder and the mask branch.
//!
//! - Input projections: 1 x 1 conv + BatchNorm (no activation) of backbone
//!   stages 2-4 to 256 channels: P3 (100^2), P4 (50^2), P5 (25^2).
//! - AIFI: one post-norm transformer layer on P5's 625 tokens - q = k =
//!   x + pos (the 2-D sine-cosine table, baked at load), v = x, 8 heads of
//!   32, exact GELU FFN 256 -> 1024 -> 256.
//! - CCFF: FPN top-down (lateral 1 x 1 + SiLU, nearest 2x up, concat
//!   [up, lateral], CSP-RepVGG block) then PAN bottom-up (3 x 3 stride-2 conv
//!   + SiLU, concat [down, lateral], CSP-RepVGG block) -> N3 / N4 / N5.
//! - Mask branch: per-level scale heads (3 x 3 + SiLU, bilinear 2x) summed at
//!   stride 8, an output conv, bilinear 2x to stride 4 plus the stage-1
//!   feature's lateral conv, then the 32-prototype head: `mask_feat`
//!   (200 x 200 x 32).
//!
//! A CSP-RepVGG block: `conv3(rep(rep(rep(conv1(x)))) + conv2(x))`, conv1 /
//! conv2 1 x 1 512 -> 256 + SiLU, each RepVGG its 3 x 3 + 1 x 1 pair fused
//! into one 3 x 3 + SiLU at load, conv3 the identity at this width.

use cudarc::driver::CudaSlice;

use super::load::{Names, Reader};
use super::{Act, BackboneOut, ConvBn, GpuModelError, Plane};
use crate::gpu::{GpuExecutor, HalfTensor};

const D: usize = 256;
const HEADS: usize = 8;
const FFN: usize = 1024;
const LN_EPS: f32 = 1e-5;

struct Csp {
    conv1: ConvBn,
    conv2: ConvBn,
    reps: [ConvBn; 3],
}

struct Aifi {
    q: (HalfTensor, CudaSlice<f32>),
    k: (HalfTensor, CudaSlice<f32>),
    v: (HalfTensor, CudaSlice<f32>),
    o: (HalfTensor, CudaSlice<f32>),
    ln1: (CudaSlice<f32>, CudaSlice<f32>),
    fc1: (HalfTensor, CudaSlice<f32>),
    fc2: (HalfTensor, CudaSlice<f32>),
    ln2: (CudaSlice<f32>, CudaSlice<f32>),
    /// the 25 x 25 sine-cosine table, [625][256]
    pos: CudaSlice<f32>,
}

pub(super) struct Encoder {
    proj: [ConvBn; 3],
    aifi: Aifi,
    lateral: [ConvBn; 2],
    fpn: [Csp; 2],
    down: [ConvBn; 2],
    pan: [Csp; 2],
    // mask branch
    head8: ConvBn,
    head16: ConvBn,
    head32: [ConvBn; 2],
    mask_out: ConvBn,
    mask_lateral: ConvBn,
    mask_base: ConvBn,
    prototypes: ConvBn,
}

/// The encoder's levels and the mask prototypes.
pub struct EncoderOut {
    /// N3 (100^2), N4 (50^2), N5 (25^2), 256 channels
    pub levels: [Plane; 3],
    /// 200 x 200 x 32
    pub mask_feat: Plane,
}

/// HF's `build_2d_sinusoidal_position_embedding`: f64 frequencies, cast to
/// f32, `[sin_h | cos_h | sin_w | cos_w]` per row-major token.
fn sincos_2d(h: usize, w: usize, dim: usize, temperature: f64) -> Vec<f32> {
    let pd = dim / 4;
    let omega: Vec<f64> = (0..pd)
        .map(|k| 1.0 / temperature.powf(k as f64 / pd as f64))
        .collect();
    let mut v = Vec::with_capacity(h * w * dim);
    for y in 0..h {
        for x in 0..w {
            for f in [f64::sin, f64::cos] {
                v.extend(omega.iter().map(|o| f(y as f64 * o) as f32));
            }
            for f in [f64::sin, f64::cos] {
                v.extend(omega.iter().map(|o| f(x as f64 * o) as f32));
            }
        }
    }
    v
}

impl Encoder {
    pub(super) fn load(rd: &mut Reader) -> Result<Self, GpuModelError> {
        let csp = |rd: &mut Reader, p: &str| -> Result<Csp, GpuModelError> {
            Ok(Csp {
                conv1: rd.conv_bn(&format!("{p}.conv1"), Names::ConvNorm, (2 * D, D, 1, 1))?,
                conv2: rd.conv_bn(&format!("{p}.conv2"), Names::ConvNorm, (2 * D, D, 1, 1))?,
                reps: [
                    rd.repvgg(&format!("{p}.bottlenecks.0"), D)?,
                    rd.repvgg(&format!("{p}.bottlenecks.1"), D)?,
                    rd.repvgg(&format!("{p}.bottlenecks.2"), D)?,
                ],
            })
        };
        let proj = [
            rd.conv_bn("model.encoder_input_proj.0", Names::Seq, (512, D, 1, 1))?,
            rd.conv_bn("model.encoder_input_proj.1", Names::Seq, (1024, D, 1, 1))?,
            rd.conv_bn("model.encoder_input_proj.2", Names::Seq, (2048, D, 1, 1))?,
        ];
        let a = "model.encoder.encoder.0.layers.0";
        let qscale = 1.0 / ((D / HEADS) as f32).sqrt();
        let aifi = Aifi {
            q: rd.linear(&format!("{a}.self_attn.q_proj"), (D, D), qscale)?,
            k: rd.linear(&format!("{a}.self_attn.k_proj"), (D, D), 1.0)?,
            v: rd.linear(&format!("{a}.self_attn.v_proj"), (D, D), 1.0)?,
            o: rd.linear(&format!("{a}.self_attn.out_proj"), (D, D), 1.0)?,
            ln1: rd.norm(&format!("{a}.self_attn_layer_norm"), D)?,
            fc1: rd.linear(&format!("{a}.fc1"), (D, FFN), 1.0)?,
            fc2: rd.linear(&format!("{a}.fc2"), (FFN, D), 1.0)?,
            ln2: rd.norm(&format!("{a}.final_layer_norm"), D)?,
            pos: rd.dev(&sincos_2d(25, 25, D, 10000.0))?,
        };
        let e = "model.encoder";
        let cn = Names::ConvNorm;
        let cvn = Names::ConvolutionNormalization;
        let mh = format!("{e}.mask_feature_head");
        Ok(Self {
            proj,
            aifi,
            lateral: [
                rd.conv_bn(&format!("{e}.lateral_convs.0"), cn, (D, D, 1, 1))?,
                rd.conv_bn(&format!("{e}.lateral_convs.1"), cn, (D, D, 1, 1))?,
            ],
            fpn: [
                csp(rd, &format!("{e}.fpn_blocks.0"))?,
                csp(rd, &format!("{e}.fpn_blocks.1"))?,
            ],
            down: [
                rd.conv_bn(&format!("{e}.downsample_convs.0"), cn, (D, D, 3, 2))?,
                rd.conv_bn(&format!("{e}.downsample_convs.1"), cn, (D, D, 3, 2))?,
            ],
            pan: [
                csp(rd, &format!("{e}.pan_blocks.0"))?,
                csp(rd, &format!("{e}.pan_blocks.1"))?,
            ],
            head8: rd.conv_bn(&format!("{mh}.scale_heads.0.layers.0"), cvn, (D, 64, 3, 1))?,
            head16: rd.conv_bn(&format!("{mh}.scale_heads.1.layers.0"), cvn, (D, 64, 3, 1))?,
            head32: [
                rd.conv_bn(&format!("{mh}.scale_heads.2.layers.0"), cvn, (D, 64, 3, 1))?,
                rd.conv_bn(&format!("{mh}.scale_heads.2.layers.2"), cvn, (64, 64, 3, 1))?,
            ],
            mask_out: rd.conv_bn(&format!("{mh}.output_conv"), cvn, (64, 64, 3, 1))?,
            mask_lateral: rd.conv_bn(&format!("{e}.encoder_mask_lateral"), cvn, (128, 64, 3, 1))?,
            mask_base: rd.conv_bn(
                &format!("{e}.encoder_mask_output.base_conv"),
                cvn,
                (64, 64, 3, 1),
            )?,
            prototypes: rd.conv_bias(&format!("{e}.encoder_mask_output.conv"), (64, 32, 1))?,
        })
    }

    pub(super) fn forward(
        &self,
        exec: &GpuExecutor,
        bb: &BackboneOut,
    ) -> Result<EncoderOut, GpuModelError> {
        let p3 = self.proj[0].apply(exec, &bb.stages[1], Act::None, 0)?;
        let p4 = self.proj[1].apply(exec, &bb.stages[2], Act::None, 0)?;
        let p5 = self.aifi.forward(exec, &self.proj[2], &bb.stages[3])?;
        // FPN, top-down
        let f5 = self.lateral[0].apply(exec, &p5, Act::Silu, 0)?;
        let f4 = self.fpn[0].forward(exec, &cat2(exec, &up2(exec, &f5, false)?, &p4)?)?;
        let f4l = self.lateral[1].apply(exec, &f4, Act::Silu, 0)?;
        let n3 = self.fpn[1].forward(exec, &cat2(exec, &up2(exec, &f4l, false)?, &p3)?)?;
        // PAN, bottom-up
        let d3 = self.down[0].apply(exec, &n3, Act::Silu, 0)?;
        let n4 = self.pan[0].forward(exec, &cat2(exec, &d3, &f4l)?)?;
        let d4 = self.down[1].apply(exec, &n4, Act::Silu, 0)?;
        let n5 = self.pan[1].forward(exec, &cat2(exec, &d4, &f5)?)?;
        // mask branch: every scale head lands at stride 8
        let h8 = self.head8.apply(exec, &n3, Act::Silu, 0)?;
        let h16 = up2(exec, &self.head16.apply(exec, &n4, Act::Silu, 0)?, true)?;
        let t = up2(exec, &self.head32[0].apply(exec, &n5, Act::Silu, 0)?, true)?;
        let h32 = up2(exec, &self.head32[1].apply(exec, &t, Act::Silu, 0)?, true)?;
        let n = h8.h * h8.w * h8.c;
        let mut sum = exec.alloc_f16(n)?;
        exec.dl_add_h(&h8.data, &h16.data, &mut sum, n, 0)?;
        let mut sum2 = exec.alloc_f16(n)?;
        exec.dl_add_h(&sum, &h32.data, &mut sum2, n, 0)?;
        let sum = Plane {
            data: sum2,
            h: h8.h,
            w: h8.w,
            c: h8.c,
        };
        let m = self.mask_out.apply(exec, &sum, Act::Silu, 0)?;
        let lat = self.mask_lateral.apply(exec, &bb.stages[0], Act::Silu, 0)?;
        let mut up = exec.alloc_f16(4 * m.h * m.w * m.c)?;
        exec.dl_up2_h(&m.data, &mut up, Some(&lat.data), (m.h, m.w, m.c), true)?;
        let fused = Plane {
            data: up,
            h: 2 * m.h,
            w: 2 * m.w,
            c: m.c,
        };
        let base = self.mask_base.apply(exec, &fused, Act::Silu, 0)?;
        let mask_feat = self.prototypes.apply(exec, &base, Act::None, 0)?;
        Ok(EncoderOut {
            levels: [n3, n4, n5],
            mask_feat,
        })
    }
}

impl Aifi {
    /// P5's projection (in f32 straight off the GEMM, the residual stream)
    /// through the encoder layer, back out as the f16 plane CCFF reads.
    fn forward(
        &self,
        exec: &GpuExecutor,
        proj: &ConvBn,
        s4: &Plane,
    ) -> Result<Plane, GpuModelError> {
        let rows = s4.h * s4.w;
        let n = rows * D;
        let mut x = exec.alloc(n)?;
        exec.matvec_batch_f16(&proj.w, &s4.data, &mut x, rows)?;
        exec.bias_add(&mut x, &proj.b, rows, D)?;
        let mut hp = exec.alloc(n)?;
        exec.copy_region(&x, 0, &mut hp, 0, n)?;
        exec.add(&mut hp, &self.pos, n)?;
        let mut hp16 = exec.alloc_f16(n)?;
        exec.convert_f32_f16(&hp, &mut hp16, n)?;
        let mut x16 = exec.alloc_f16(n)?;
        exec.convert_f32_f16(&x, &mut x16, n)?;
        let (mut q, mut k, mut v) = (exec.alloc_f16(n)?, exec.alloc_f16(n)?, exec.alloc_f16(n)?);
        exec.matvec_batch_f16_h_bias(&self.q.0, &hp16, &mut q, Some(&self.q.1), rows)?;
        exec.matvec_batch_f16_h_bias(&self.k.0, &hp16, &mut k, Some(&self.k.1), rows)?;
        exec.matvec_batch_f16_h_bias(&self.v.0, &x16, &mut v, Some(&self.v.1), rows)?;
        let mut a = exec.alloc_f16(n)?;
        exec.vision_attn_h(&q, &k, &v, &mut a, rows, rows, HEADS, D / HEADS, 1)?;
        let mut o = exec.alloc(n)?;
        exec.matvec_batch_f16(&self.o.0, &a, &mut o, rows)?;
        exec.bias_add(&mut o, &self.o.1, rows, D)?;
        exec.add(&mut o, &x, n)?;
        let mut h1 = exec.alloc(n)?;
        exec.layernorm(&o, &self.ln1.0, &self.ln1.1, &mut h1, rows, D, LN_EPS)?;
        let mut h16 = exec.alloc_f16(n)?;
        exec.convert_f32_f16(&h1, &mut h16, n)?;
        let mut f1 = exec.alloc_f16(rows * FFN)?;
        exec.matvec_batch_f16_h_gelu(&self.fc1.0, &h16, &mut f1, &self.fc1.1, rows)?;
        let mut f2 = exec.alloc(n)?;
        exec.matvec_batch_f16(&self.fc2.0, &f1, &mut f2, rows)?;
        exec.bias_add(&mut f2, &self.fc2.1, rows, D)?;
        exec.add(&mut f2, &h1, n)?;
        let mut h2 = exec.alloc(n)?;
        exec.layernorm(&f2, &self.ln2.0, &self.ln2.1, &mut h2, rows, D, LN_EPS)?;
        let mut out = exec.alloc_f16(n)?;
        exec.convert_f32_f16(&h2, &mut out, n)?;
        Ok(Plane {
            data: out,
            h: s4.h,
            w: s4.w,
            c: D,
        })
    }
}

impl Csp {
    fn forward(&self, exec: &GpuExecutor, x: &Plane) -> Result<Plane, GpuModelError> {
        let mut y1 = self.conv1.apply(exec, x, Act::Silu, 0)?;
        for r in &self.reps {
            y1 = r.apply(exec, &y1, Act::Silu, 0)?;
        }
        let y2 = self.conv2.apply(exec, x, Act::Silu, 0)?;
        let n = y1.h * y1.w * y1.c;
        let mut out = exec.alloc_f16(n)?;
        exec.dl_add_h(&y1.data, &y2.data, &mut out, n, 0)?;
        Ok(Plane {
            data: out,
            h: y1.h,
            w: y1.w,
            c: y1.c,
        })
    }
}

fn up2(exec: &GpuExecutor, x: &Plane, bilinear: bool) -> Result<Plane, GpuModelError> {
    let mut y = exec.alloc_f16(4 * x.h * x.w * x.c)?;
    exec.dl_up2_h(&x.data, &mut y, None, (x.h, x.w, x.c), bilinear)?;
    Ok(Plane {
        data: y,
        h: 2 * x.h,
        w: 2 * x.w,
        c: x.c,
    })
}

/// channel concat `[a, b]` of two planes of one spatial shape
fn cat2(exec: &GpuExecutor, a: &Plane, b: &Plane) -> Result<Plane, GpuModelError> {
    debug_assert_eq!((a.h, a.w), (b.h, b.w));
    let rows = a.h * a.w;
    let c = a.c + b.c;
    let mut y = exec.alloc_f16(rows * c)?;
    exec.dl_concat_h(&a.data, &mut y, rows, a.c, (c, 0))?;
    exec.dl_concat_h(&b.data, &mut y, rows, b.c, (c, a.c))?;
    Ok(Plane {
        data: y,
        h: a.h,
        w: a.w,
        c,
    })
}
