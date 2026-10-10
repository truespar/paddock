//! HGNetV2-L, PP-DocLayoutV3's backbone (Transformers `HGNetV2Backbone`, the
//! "L" preset): a stem down to stride 4, then four stages of HG blocks - six
//! conv layers each (3 x 3, or the light 1 x 1 + depthwise 5 x 5 in stages 3
//! and 4), their outputs concatenated with the block input and squeezed by
//! two 1 x 1 convs, a residual on every block after a stage's first. Stages 2
//! to 4 open on a depthwise 3 x 3 stride-2 downsample. All four stage outputs
//! leave: the first feeds the mask branch, the last three the encoder.

use super::load::{Names, Reader};
use super::{Act, BackboneOut, ConvBn, DwConv, GpuModelError, INPUT, Plane};
use crate::gpu::GpuExecutor;

const BB: &str = "model.backbone.model";

/// (in, mid, out, blocks, downsample, light, kernel) per stage - the preset.
const STAGES: [(usize, usize, usize, usize, bool, bool, usize); 4] = [
    (48, 48, 128, 1, false, false, 3),
    (128, 96, 512, 1, true, false, 3),
    (512, 192, 1024, 3, true, true, 5),
    (1024, 384, 2048, 1, true, true, 5),
];
/// conv layers per HG block
const LAYERS: usize = 6;

/// One HG block layer: a plain k x k conv, or the light pair.
enum Layer {
    Conv(ConvBn),
    Light(ConvBn, DwConv),
}

struct Block {
    layers: Vec<Layer>,
    squeeze: ConvBn,
    excite: ConvBn,
    residual: bool,
}

struct Stage {
    down: Option<DwConv>,
    blocks: Vec<Block>,
}

pub(super) struct Backbone {
    stem1: ConvBn,
    stem2a: ConvBn,
    stem2b: ConvBn,
    stem3: ConvBn,
    stem4: ConvBn,
    stages: Vec<Stage>,
}

impl Backbone {
    pub(super) fn load(rd: &mut Reader) -> Result<Self, GpuModelError> {
        let cn = Names::ConvolutionNormalization;
        let e = |n: &str| format!("{BB}.embedder.{n}");
        let stem1 = rd.conv_bn(&e("stem1"), cn, (3, 32, 3, 2))?;
        let stem2a = rd.conv_bn(&e("stem2a"), cn, (32, 16, 2, 1))?;
        let stem2b = rd.conv_bn(&e("stem2b"), cn, (16, 32, 2, 1))?;
        let stem3 = rd.conv_bn(&e("stem3"), cn, (64, 32, 3, 2))?;
        let stem4 = rd.conv_bn(&e("stem4"), cn, (32, 48, 1, 1))?;
        let mut stages = Vec::new();
        for (s, &(cin, mid, cout, nblocks, down, light, k)) in STAGES.iter().enumerate() {
            let sp = format!("{BB}.encoder.stages.{s}");
            let down = if down {
                Some(rd.dw_bn(&format!("{sp}.downsample"), (cin, 3, 2))?)
            } else {
                None
            };
            let mut blocks = Vec::new();
            for b in 0..nblocks {
                let bp = format!("{sp}.blocks.{b}");
                let bin = if b == 0 { cin } else { cout };
                let mut layers = Vec::new();
                for l in 0..LAYERS {
                    let lin = if l == 0 { bin } else { mid };
                    let lp = format!("{bp}.layers.{l}");
                    layers.push(if light {
                        Layer::Light(
                            rd.conv_bn(&format!("{lp}.conv1"), cn, (lin, mid, 1, 1))?,
                            rd.dw_bn(&format!("{lp}.conv2"), (mid, k, 1))?,
                        )
                    } else {
                        Layer::Conv(rd.conv_bn(&lp, cn, (lin, mid, k, 1))?)
                    });
                }
                let total = bin + LAYERS * mid;
                blocks.push(Block {
                    layers,
                    squeeze: rd.conv_bn(
                        &format!("{bp}.aggregation.0"),
                        cn,
                        (total, cout / 2, 1, 1),
                    )?,
                    excite: rd.conv_bn(
                        &format!("{bp}.aggregation.1"),
                        cn,
                        (cout / 2, cout, 1, 1),
                    )?,
                    residual: b != 0,
                });
            }
            stages.push(Stage { down, blocks });
        }
        Ok(Self {
            stem1,
            stem2a,
            stem2b,
            stem3,
            stem4,
            stages,
        })
    }

    pub(super) fn forward(
        &self,
        exec: &GpuExecutor,
        rgb800: &[u8],
    ) -> Result<BackboneOut, GpuModelError> {
        let n = 3 * INPUT * INPUT;
        let src = exec.to_device_u8(rgb800)?;
        let mut x = exec.alloc_f16(n)?;
        exec.dl_u8_to_h(&src, &mut x, n)?;
        let img = Plane {
            data: x,
            h: INPUT,
            w: INPUT,
            c: 3,
        };
        // stem: 3x3 s2, then the padded two-branch 2x2 pair beside a 2x2 pool
        let s1 = self.stem1.apply(exec, &img, Act::Relu, 0)?;
        let s2a = self.stem2a.apply(exec, &s1, Act::Relu, 1)?;
        let s2b = self.stem2b.apply(exec, &s2a, Act::Relu, 1)?;
        let mut pooled = exec.alloc_f16(s1.h * s1.w * s1.c)?;
        exec.dl_maxpool2_h(&s1.data, &mut pooled, (s1.h, s1.w, s1.c))?;
        let rows = s1.h * s1.w;
        let mut cat = exec.alloc_f16(rows * 64)?;
        exec.dl_concat_h(&pooled, &mut cat, rows, 32, (64, 0))?;
        exec.dl_concat_h(&s2b.data, &mut cat, rows, 32, (64, 32))?;
        let cat = Plane {
            data: cat,
            h: s1.h,
            w: s1.w,
            c: 64,
        };
        let s3 = self.stem3.apply(exec, &cat, Act::Relu, 0)?;
        let mut x = self.stem4.apply(exec, &s3, Act::Relu, 0)?;
        let mut outs = Vec::with_capacity(4);
        for stage in &self.stages {
            if let Some(d) = &stage.down {
                x = d.apply(exec, &x, Act::None)?;
            }
            for block in &stage.blocks {
                x = block.forward(exec, x)?;
            }
            outs.push(Plane {
                data: x.data.clone(),
                h: x.h,
                w: x.w,
                c: x.c,
            });
        }
        let stages: [Plane; 4] = outs
            .try_into()
            .map_err(|_| GpuModelError::Unsupported("hgnetv2: four stages".into()))?;
        Ok(BackboneOut { stages })
    }
}

impl Block {
    fn forward(&self, exec: &GpuExecutor, input: Plane) -> Result<Plane, GpuModelError> {
        let rows = input.h * input.w;
        let total = self.squeeze.cin;
        let mut cat = exec.alloc_f16(rows * total)?;
        exec.dl_concat_h(&input.data, &mut cat, rows, input.c, (total, 0))?;
        let mut off = input.c;
        let mut h = None::<Plane>;
        for layer in &self.layers {
            let src = h.as_ref().unwrap_or(&input);
            let y = match layer {
                Layer::Conv(c) => c.apply(exec, src, Act::Relu, 0)?,
                Layer::Light(c1, dw) => {
                    let t = c1.apply(exec, src, Act::None, 0)?;
                    dw.apply(exec, &t, Act::Relu)?
                }
            };
            exec.dl_concat_h(&y.data, &mut cat, rows, y.c, (total, off))?;
            off += y.c;
            h = Some(y);
        }
        let cat = Plane {
            data: cat,
            h: input.h,
            w: input.w,
            c: total,
        };
        let sq = self.squeeze.apply(exec, &cat, Act::Relu, 0)?;
        let mut out = self.excite.apply(exec, &sq, Act::Relu, 0)?;
        if self.residual {
            let n = rows * out.c;
            let mut sum = exec.alloc_f16(n)?;
            exec.dl_add_h(&out.data, &input.data, &mut sum, n, 0)?;
            out.data = sum;
        }
        Ok(out)
    }
}

impl ConvBn {
    /// The conv + folded BN + `act` over `x`. Padding is (k - 1) / 2 a side,
    /// plus `extra` zero rows / columns past the bottom / right edge (the
    /// HGNetV2 stem's `F.pad(x, (0, 1, 0, 1))` before its 2 x 2 convs).
    pub(super) fn apply(
        &self,
        exec: &GpuExecutor,
        x: &Plane,
        act: Act,
        extra: usize,
    ) -> Result<Plane, GpuModelError> {
        debug_assert_eq!(x.c, self.cin);
        let p = (self.k - 1) / 2;
        let oh = (x.h + 2 * p + extra - self.k) / self.stride + 1;
        let ow = (x.w + 2 * p + extra - self.k) / self.stride + 1;
        let rows = oh * ow;
        let im2row;
        let input = if self.k == 1 && self.stride == 1 {
            &x.data
        } else {
            let mut buf = exec.alloc_f16(rows * self.kpad)?;
            exec.dl_im2row_h(
                &x.data,
                &mut buf,
                (x.h, x.w, x.c),
                (oh, ow),
                (self.k, self.k),
                self.stride,
                (p as i32, p as i32),
                self.kpad,
            )?;
            im2row = buf;
            &im2row
        };
        let mut y = exec.alloc_f16(rows * self.cout)?;
        match act {
            Act::Relu => {
                exec.matvec_batch_f16_h_relu(&self.w, input, &mut y, Some(&self.b), rows)?
            }
            Act::Silu => {
                exec.matvec_batch_f16_h_silu(&self.w, input, &mut y, Some(&self.b), rows)?
            }
            Act::None => {
                exec.matvec_batch_f16_h_bias(&self.w, input, &mut y, Some(&self.b), rows)?
            }
        }
        Ok(Plane {
            data: y,
            h: oh,
            w: ow,
            c: self.cout,
        })
    }
}

impl DwConv {
    pub(super) fn apply(
        &self,
        exec: &GpuExecutor,
        x: &Plane,
        act: Act,
    ) -> Result<Plane, GpuModelError> {
        debug_assert_eq!(x.c, self.c);
        let p = (self.k - 1) / 2;
        let oh = (x.h + 2 * p - self.k) / self.stride + 1;
        let ow = (x.w + 2 * p - self.k) / self.stride + 1;
        let mut y = exec.alloc_f16(oh * ow * self.c)?;
        exec.dl_dwconv_h(
            &x.data,
            &mut y,
            &self.w,
            &self.b,
            (x.h, x.w, x.c),
            (oh, ow),
            self.k,
            self.stride,
            u32::from(act == Act::Relu),
        )?;
        Ok(Plane {
            data: y,
            h: oh,
            w: ow,
            c: self.c,
        })
    }
}
