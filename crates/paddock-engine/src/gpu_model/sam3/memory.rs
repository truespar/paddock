//! SAM 3's memory encoder (Meta's `SimpleMaskEncoder`): one memory an object
//! a frame, what the tracker's memory attention reads on the frames after.
//!
//!   m   = mask logits [n][s][s] -> sigmoid (> 0 on a point-prompted frame)
//!         * 20 - 10, bilinear to 1152^2
//!   m   = 4 x [Conv2d(k3, s2, p1) -> LayerNorm2d -> GELU]   1 -> 4 -> 16 ->
//!         64 -> 256 channels, 1152^2 -> 72^2, then a 1x1 (final_conv)
//!   x   = 1x1(trk72) + m    the frame's own tracker feature, the one the
//!                           memory attention has NOT conditioned
//!   x   = 2 x ConvNeXt block: x += scale * pw2(GELU(pw1(LN2d(dwconv7(x)))))
//!   mem = 1x1 to 64 channels, + the occlusion embedding where the object is
//!         gone                                              [5184][64]
//!   pos = the 64-channel sine table over the 72^2 grid, a constant
//!
//! The first two stages are the direct kernel (4 and 16 channels; the first
//! reads the logits and never writes the 1152^2 plane), the two wide ones the
//! f16 ring's implicit conv at stride 2, the 1x1s and the MLPs the dense
//! lane's GEMMs. Every object of a frame runs in one pass, as Meta batches a
//! tracker state's objects; the frame's own 1x1 is computed once and
//! broadcast.
//!
//! Precision is the tracker's class: f16 GEMM operands, f32 everything else.
//! The block's layer scale is folded into pw2's f32 weights and bias before
//! their one round to f16 (`scale * (W h + b)` = `(scale W) h + scale b`), so
//! the residual is one fused add. Meta then stores the memory in bf16; the
//! bank does that, this returns the f32.

use std::path::Path;
use std::sync::Arc;

use cudarc::driver::CudaSlice;
use half::f16;

use super::detector::sine_pos_table;
use super::load::Reader;
use super::{Conv, GpuModelError, Norm};
use crate::gpu::{GpuExecutor, Sam3MaskDownOut, Sam3MemDownIn};

const ME: &str = "tracker_model.memory_encoder";

/// The memory grid side (1008 / 14) and the planes' widths.
const GRID: usize = 72;
const D: usize = 256;
pub const MEM_DIM: usize = 64;
const MLP: usize = 1024;
/// The downsampler's input side (Meta's `interpol_size`).
const IN_SIDE: usize = 1152;
/// LayerNorm2d's eps everywhere in the encoder.
const LN2D_EPS: f32 = 1e-6;
/// Every downsampler plane holds this many values an object: 576^2 x 4 =
/// 288^2 x 16 = 144^2 x 64 = 72^2 x 256, so one f32 and one f16 scratch an
/// object carry the whole chain.
const PLANE: usize = GRID * GRID * D;

/// A small stage's Conv2d, kept f32 `[cout][cin][3][3]` for the direct kernel.
struct SmallConv {
    w: CudaSlice<f32>,
    b: CudaSlice<f32>,
    ln: Norm,
}

/// A wide stage: the tap-major f16 plane the implicit conv eats.
struct WideConv {
    conv: Conv,
    ln: Norm,
}

struct CxBlock {
    /// depthwise taps, tap-major `[49][256]`
    dw_w: CudaSlice<f32>,
    dw_b: CudaSlice<f32>,
    ln: Norm,
    pw1: Conv,
    /// pw2 with the layer scale folded in
    pw2: Conv,
}

struct Workspace {
    cap: usize,
    a32: CudaSlice<f32>,
    a16: CudaSlice<f16>,
    x: CudaSlice<f32>,
    mlp: CudaSlice<f16>,
    pix16: CudaSlice<f16>,
    pix: CudaSlice<f32>,
    out: CudaSlice<f32>,
}

/// The memory encoder, resident, for up to `cap` objects a pass.
pub struct GpuSam3MemEnc {
    exec: Arc<GpuExecutor>,
    small: [SmallConv; 2],
    wide: [WideConv; 2],
    final_conv: Conv,
    feat_proj: Conv,
    blocks: [CxBlock; 2],
    proj: Conv,
    /// the occlusion embedding repeated over the grid, `[5184][64]`
    occl: CudaSlice<f32>,
    pos: CudaSlice<f32>,
    ws: Workspace,
    weight_bytes: u64,
}

/// `[out][in][3][3]` -> `[out][ky][kx][in]`, the implicit conv's tap-outer K.
fn conv3_taps(w: &[f32], cin: usize, cout: usize) -> Vec<f32> {
    let mut out = vec![0f32; w.len()];
    for o in 0..cout {
        for i in 0..cin {
            for t in 0..9 {
                out[(o * 9 + t) * cin + i] = w[(o * cin + i) * 9 + t];
            }
        }
    }
    out
}

impl GpuSam3MemEnc {
    /// Load the encoder from a `facebook/sam3` directory, sized for `cap`
    /// objects a pass.
    pub fn load_dir(exec: Arc<GpuExecutor>, dir: &Path, cap: usize) -> Result<Self, GpuModelError> {
        if !exec.has_sam3_memory() {
            return Err(GpuModelError::Unsupported(
                "this kernel pack predates SAM 3's memory encoder (slots 784-787) - rebuild or \
                 update the pack"
                    .into(),
            ));
        }
        let cap = cap.max(1);
        let st = super::checkpoint::open(dir)?;
        let ws_bytes = (cap * (PLANE * (4 + 2 + 4) + GRID * GRID * (MLP * 2 + MEM_DIM * 4))
            + GRID * GRID * D * 6) as u64;
        exec.vram_load_gate(ws_bytes + (8u64 << 20), "sam3 memory encoder")
            .map_err(GpuModelError::WontFit)?;
        let mut r = Reader {
            st: &*st,
            exec: &exec,
            bytes: 0,
        };
        let down = |i: usize, s: &str| format!("{ME}.mask_downsampler.layers.{i}.{s}");
        let mut small = Vec::with_capacity(2);
        for (i, (cin, cout)) in [(1usize, 4usize), (4, 16)].into_iter().enumerate() {
            let w = r.f32s(&down(i, "conv.weight"), &[cout, cin, 3, 3])?;
            small.push(SmallConv {
                w: r.dev(&w)?,
                b: r.vec(&down(i, "conv.bias"), cout)?,
                ln: r.norm(&down(i, "layer_norm"), cout)?,
            });
        }
        let mut wide = Vec::with_capacity(2);
        for (i, (cin, cout)) in [(2usize, (16usize, 64usize)), (3, (64, 256))] {
            let name = down(i, "conv.weight");
            let w = r.f32s(&name, &[cout, cin, 3, 3])?;
            wide.push(WideConv {
                conv: Conv {
                    w: r.plane(&conv3_taps(&w, cin, cout), 9 * cin, cout, &name)?,
                    b: r.vec(&down(i, "conv.bias"), cout)?,
                },
                ln: r.norm(&down(i, "layer_norm"), cout)?,
            });
        }
        let final_conv = r.conv1(&format!("{ME}.mask_downsampler.final_conv"), D, D)?;
        let feat_proj = r.conv1(&format!("{ME}.feature_projection"), D, D)?;
        let mut blocks = Vec::with_capacity(2);
        for i in 0..2 {
            let p = |s: &str| format!("{ME}.memory_fuser.layers.{i}.{s}");
            let dw = r.f32s(&p("depthwise_conv.weight"), &[D, 1, 7, 7])?;
            let mut taps = vec![0f32; dw.len()];
            for c in 0..D {
                for t in 0..49 {
                    taps[t * D + c] = dw[c * 49 + t];
                }
            }
            let pw1 = Conv {
                w: r.linear(&p("pointwise_conv1.weight"), &[MLP, D])?,
                b: r.vec(&p("pointwise_conv1.bias"), MLP)?,
            };
            let scale = r.f32s(&p("scale"), &[D])?;
            let name = p("pointwise_conv2.weight");
            let mut w2 = r.f32s(&name, &[D, MLP])?;
            let mut b2 = r.f32s(&p("pointwise_conv2.bias"), &[D])?;
            for (o, s) in scale.iter().enumerate() {
                w2[o * MLP..(o + 1) * MLP].iter_mut().for_each(|v| *v *= s);
                b2[o] *= s;
            }
            blocks.push(CxBlock {
                dw_w: r.dev(&taps)?,
                dw_b: r.vec(&p("depthwise_conv.bias"), D)?,
                ln: r.norm(&p("layer_norm"), D)?,
                pw1,
                pw2: Conv {
                    w: r.plane(&w2, MLP, D, &name)?,
                    b: r.dev(&b2)?,
                },
            });
        }
        let proj = r.conv1(&format!("{ME}.projection"), D, MEM_DIM)?;
        let occl = {
            let v = r.f32s(
                "tracker_model.occlusion_spatial_embedding_parameter",
                &[1, MEM_DIM],
            )?;
            let plane: Vec<f32> = (0..GRID * GRID).flat_map(|_| v.iter().copied()).collect();
            r.dev(&plane)?
        };
        let pos = r.dev(&sine_pos_table(GRID, MEM_DIM, 10000.0))?;
        let weight_bytes = r.bytes;

        let f = |n: usize| exec.alloc(n);
        let h = |n: usize| exec.alloc_f16(n);
        let ws = Workspace {
            cap,
            a32: f(cap * PLANE)?,
            a16: h(cap * PLANE)?,
            x: f(cap * PLANE)?,
            mlp: h(cap * GRID * GRID * MLP)?,
            pix16: h(GRID * GRID * D)?,
            pix: f(GRID * GRID * D)?,
            out: f(cap * GRID * GRID * MEM_DIM)?,
        };
        let small: [SmallConv; 2] = small.try_into().map_err(|_| unreachable_load())?;
        let wide: [WideConv; 2] = wide.try_into().map_err(|_| unreachable_load())?;
        let blocks: [CxBlock; 2] = blocks.try_into().map_err(|_| unreachable_load())?;
        Ok(Self {
            exec,
            small,
            wide,
            final_conv,
            feat_proj,
            blocks,
            proj,
            occl,
            pos,
            ws,
            weight_bytes,
        })
    }

    pub fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }

    /// Objects one pass takes.
    pub fn capacity(&self) -> usize {
        self.ws.cap
    }

    /// The memories of `appearing.len()` objects on one frame. `pix` is the
    /// frame's tracker 72^2 feature, `[5184][256]` f32 (the neck's, before
    /// any memory conditioning); `masks` the objects' `[n][side][side]`
    /// logits, made the encoder's input by sigmoid or, `binarize` (a frame
    /// where the objects came from points or a given mask), by > 0. An
    /// object not `appearing` gets the occlusion embedding. The result is
    /// [`Self::memory`].
    pub fn encode(
        &mut self,
        pix: &CudaSlice<f32>,
        masks: &CudaSlice<f32>,
        side: usize,
        binarize: bool,
        appearing: &[bool],
    ) -> Result<(), GpuModelError> {
        let n = appearing.len();
        if n == 0 || n > self.ws.cap {
            return Err(GpuModelError::BatchTooLarge {
                got: n,
                max: self.ws.cap,
            });
        }
        if side == 0 || masks.len() < n * side * side || pix.len() < GRID * GRID * D {
            return Err(GpuModelError::Unsupported(format!(
                "sam3 memory: {n} masks of {side}^2 and a {}-value feature",
                pix.len()
            )));
        }
        let exec = self.exec.clone();
        let ws = &mut self.ws;
        let px = GRID * GRID;

        // ---- the mask downsampler ----
        let s0 = &self.small[0];
        exec.sam3_mem_down3(
            Sam3MemDownIn::Mask {
                logits: masks,
                sh: side,
                sw: side,
                binarize,
            },
            &s0.w,
            &s0.b,
            (&s0.ln.w, &s0.ln.b),
            Sam3MaskDownOut::F32(&mut ws.a32),
            n,
            IN_SIDE,
            IN_SIDE,
            1,
            4,
            LN2D_EPS,
        )?;
        let s1 = &self.small[1];
        exec.sam3_mem_down3(
            Sam3MemDownIn::Plane(&ws.a32),
            &s1.w,
            &s1.b,
            (&s1.ln.w, &s1.ln.b),
            Sam3MaskDownOut::F16(&mut ws.a16),
            n,
            IN_SIDE / 2,
            IN_SIDE / 2,
            4,
            16,
            LN2D_EPS,
        )?;
        // 288^2 x 16 -> 144^2 x 64 -> 72^2 x 256, each landing f32 + bias,
        // then LN2d + GELU back to f16 for the next
        for (wc, (side_out, cin)) in self.wide.iter().zip([(144usize, 16usize), (GRID, 64)]) {
            exec.f16_conv3s2_gemm(
                &wc.conv.w,
                &ws.a16,
                &mut ws.a32,
                Some(&wc.conv.b),
                n,
                side_out,
                side_out,
                cin,
                PLANE / cin,
            )?;
            let cout = wc.conv.w.dims[1];
            let rows = n * side_out * side_out;
            exec.sam3_ln_gelu_h(
                &ws.a32,
                (&wc.ln.w, &wc.ln.b),
                &mut ws.a16,
                rows,
                cout,
                LN2D_EPS,
            )?;
        }
        exec.matvec_batch_f16(&self.final_conv.w, &ws.a16, &mut ws.x, n * px)?;
        exec.bias_add(&mut ws.x, &self.final_conv.b, n * px, D)?;

        // ---- + the frame's own feature, once for every object ----
        exec.convert_f32_f16(pix, &mut ws.pix16, px * D)?;
        exec.matvec_batch_f16(&self.feat_proj.w, &ws.pix16, &mut ws.pix, px)?;
        exec.bias_add(&mut ws.pix, &self.feat_proj.b, px, D)?;
        exec.add_rows_bcast(&mut ws.x, &ws.pix, n * px, px, D)?;

        // ---- the fuser ----
        for blk in &self.blocks {
            exec.sam3_dwconv7_ln_h(
                &ws.x,
                &blk.dw_w,
                &blk.dw_b,
                (&blk.ln.w, &blk.ln.b),
                &mut ws.a16,
                n,
                GRID,
                GRID,
                D,
                LN2D_EPS,
            )?;
            exec.matvec_batch_f16_h_gelu(&blk.pw1.w, &ws.a16, &mut ws.mlp, &blk.pw1.b, n * px)?;
            exec.matvec_batch_f16(&blk.pw2.w, &ws.mlp, &mut ws.a32, n * px)?;
            exec.add_bias_res(&mut ws.x, &ws.a32, &blk.pw2.b, n * px, D)?;
        }

        // ---- to 64 channels, and the occlusion embedding ----
        exec.convert_f32_f16(&ws.x, &mut ws.a16, n * PLANE)?;
        exec.matvec_batch_f16(&self.proj.w, &ws.a16, &mut ws.out, n * px)?;
        exec.bias_add(&mut ws.out, &self.proj.b, n * px, MEM_DIM)?;
        for (i, _) in appearing.iter().enumerate().filter(|(_, a)| !**a) {
            exec.add_at(&mut ws.out, i * px * MEM_DIM, &self.occl, 0, px * MEM_DIM)?;
        }
        Ok(())
    }

    /// The last [`Self::encode`]'s memories, `[n][5184][64]` f32.
    pub fn memory(&self) -> &CudaSlice<f32> {
        &self.ws.out
    }

    /// The memory's position table, `[5184][64]` f32 - the same every frame
    /// and every object.
    pub fn pos(&self) -> &CudaSlice<f32> {
        &self.pos
    }
}

fn unreachable_load() -> GpuModelError {
    GpuModelError::Unsupported("sam3 memory: a stage list came out the wrong length".into())
}
