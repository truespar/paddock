//! SAM 3's memory attention (Meta's `TransformerEncoderCrossAttention` over
//! `TransformerDecoderLayerv2`, the tracker's `transformer.encoder`): the
//! frame's tracker 72^2 feature conditioned on one object's memory bank.
//!
//!   x = trk72 + 0.1 * pos                                   5184 x 256
//!   4 x pre-norm layer:
//!     x += o(SA(rope(q(LN1 x)), rope(k(LN1 x)), v(LN1 x)))   one 256 head
//!     x += o(CA(rope(q(LN2 x)), rope'(k(M + Mpos)), v(M)))   M the bank
//!     x += lin2(ReLU(lin1(LN3 x)))
//!   out = LN(x)
//!
//! The bank `M` is `[nk][64]`: 7-10 memory frames of 5184 tokens (the
//! memory encoder's output, stored bf16 by Meta) then 4 tokens a past object
//! pointer; `Mpos` its position (the sine table plus each frame's temporal
//! encoding, the pointers' own). rope' rotates the memory tokens' keys by
//! their grid position (Meta repeats the table once a frame) and leaves the
//! pointer tokens alone.
//!
//! Two foldings at load, both exact algebra: rope's interleaved complex pairs
//! become rotate-half pairs by permuting q's and k's projection rows (the
//! ViT's `rope_head_perm`, at 256), and the cross-attention's value and
//! output projections fold into one 64 -> 256 GEMM behind a 64-wide value
//! (`o(sum p v(m)) = (W_o W_v) sum p m + W_o b_v + b_o`, a row's weights
//! summing to one) - which is what lets the attention over ~40k memory keys
//! run its value side at 64. The self-attention's values are a full 256 and
//! run as two 128-wide halves over the same scores.
//!
//! One object a call: an object's cross-attention alone (~56 GMAC a layer
//! at a full bank) fills the die, and the scratch is then one object's.
//! Precision is the tracker's class: f16 GEMM and attention operands, f32
//! residual, norms and accumulators.

use std::path::Path;
use std::sync::Arc;

use cudarc::driver::CudaSlice;
use half::f16;

use super::detector::sine_pos_table;
use super::load::Reader;
use super::{Conv, GpuModelError, Norm, rope_head_perm};
use crate::gpu::GpuExecutor;

const MA: &str = "tracker_model.memory_attention";

const GRID: usize = 72;
const TOKENS: usize = GRID * GRID;
const D: usize = 256;
const KV_IN: usize = 64;
const FFN: usize = 2048;
const LAYERS: usize = 4;
const LN_EPS: f32 = 1e-5;
/// The biggest bank Meta builds: 4 conditioning frames + 6 previous ones,
/// and 19 object pointers of 4 tokens (the 4 conditioning frames' and 15
/// more).
pub const MEMATTN_MAX_KEYS: usize = 10 * TOKENS + 19 * 4;

struct Layer {
    ln1: Norm,
    ln2: Norm,
    ln3: Norm,
    /// q | k | v, q and k rope-permuted
    sa_qkv: Conv,
    sa_o: Conv,
    /// rope-permuted
    ca_q: Conv,
    /// 64 -> 256, rope-permuted
    ca_k: Conv,
    /// W_o W_v, 64 -> 256, with W_o b_v + b_o
    ca_ov: Conv,
    lin1: Conv,
    lin2: Conv,
}

struct Workspace {
    x: CudaSlice<f32>,
    h16: CudaSlice<f16>,
    qkv: CudaSlice<f32>,
    q16: CudaSlice<f16>,
    k16: CudaSlice<f16>,
    v16: CudaSlice<f16>,
    o16: CudaSlice<f16>,
    proj: CudaSlice<f32>,
    ffn: CudaSlice<f16>,
    k32: CudaSlice<f32>,
    kc16: CudaSlice<f16>,
    o64: CudaSlice<f16>,
    out: CudaSlice<f32>,
}

/// The memory attention, resident.
pub struct GpuSam3MemAttn {
    exec: Arc<GpuExecutor>,
    layers: Vec<Layer>,
    norm: Norm,
    /// axial rope over the grid, `[5184][128]` - x pairs then y pairs
    cs: CudaSlice<f32>,
    sn: CudaSlice<f32>,
    /// 0.1 x the tracker neck's sine table, `[5184][256]`
    pos01: CudaSlice<f32>,
    ws: Workspace,
    weight_bytes: u64,
}

/// Rows of an `[out][in]` matrix in `perm` order (new row n = old row
/// perm[n]); the bias the same way.
fn permute_rows(w: &[f32], b: &[f32], perm: &[usize], cols: usize) -> (Vec<f32>, Vec<f32>) {
    let mut wo = vec![0f32; w.len()];
    let mut bo = vec![0f32; b.len()];
    for (n, &p) in perm.iter().enumerate() {
        wo[n * cols..(n + 1) * cols].copy_from_slice(&w[p * cols..(p + 1) * cols]);
        bo[n] = b[p];
    }
    (wo, bo)
}

/// Meta's `compute_axial_cis(256, 72, 72, 10000)` as two `[5184][128]`
/// tables, in its f32 recipe: frequencies `1 / 10000^(4j / 256)` for the
/// 64 x pairs and again for the 64 y pairs, positions `t % 72` and `t / 72`.
pub(super) fn axial_rope_tables() -> (Vec<f32>, Vec<f32>) {
    let half = D / 2;
    let freqs: Vec<f32> = (0..half / 2)
        .map(|j| 1.0f32 / 10000f32.powf((4 * j) as f32 / D as f32))
        .collect();
    let mut cs = vec![0f32; TOKENS * half];
    let mut sn = vec![0f32; TOKENS * half];
    for t in 0..TOKENS {
        let (tx, ty) = ((t % GRID) as f32, (t / GRID) as f32);
        for j in 0..half {
            let a = if j < half / 2 {
                tx * freqs[j]
            } else {
                ty * freqs[j - half / 2]
            };
            cs[t * half + j] = a.cos();
            sn[t * half + j] = a.sin();
        }
    }
    (cs, sn)
}

impl GpuSam3MemAttn {
    pub fn load_dir(exec: Arc<GpuExecutor>, dir: &Path) -> Result<Self, GpuModelError> {
        if !exec.has_sam3_memory_attention() {
            return Err(GpuModelError::Unsupported(
                "this kernel pack predates SAM 3's memory attention (slots 788-789) - rebuild or \
                 update the pack"
                    .into(),
            ));
        }
        let st = super::checkpoint::open(dir)?;
        let ws_bytes = (TOKENS * (D * 4 * 3 + D * 2 * 5 + 3 * D * 4 + FFN * 2 + KV_IN * 2)
            + MEMATTN_MAX_KEYS * D * 6) as u64;
        exec.vram_load_gate(ws_bytes + (16u64 << 20), "sam3 memory attention")
            .map_err(GpuModelError::WontFit)?;
        let mut r = Reader {
            st: &*st,
            exec: &exec,
            bytes: 0,
        };
        let perm = rope_head_perm(D);
        let mut layers = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let p = |s: &str| format!("{MA}.layers.{i}.{s}");
            let lin = |r: &Reader,
                       s: &str,
                       out: usize,
                       inp: usize|
             -> Result<(Vec<f32>, Vec<f32>), GpuModelError> {
                Ok((
                    r.f32s(&p(&format!("{s}.weight")), &[out, inp])?,
                    r.f32s(&p(&format!("{s}.bias")), &[out])?,
                ))
            };
            // self-attention: q and k permuted for rope, then q | k | v
            let (qw, qb) = lin(&r, "self_attn.q_proj", D, D)?;
            let (kw, kb) = lin(&r, "self_attn.k_proj", D, D)?;
            let (vw, vb) = lin(&r, "self_attn.v_proj", D, D)?;
            let (qw, qb) = permute_rows(&qw, &qb, &perm, D);
            let (kw, kb) = permute_rows(&kw, &kb, &perm, D);
            let qkv_w = [qw, kw, vw].concat();
            let qkv_b = [qb, kb, vb].concat();
            let sa_qkv = Conv {
                w: r.plane(&qkv_w, D, 3 * D, &p("self_attn.qkv"))?,
                b: r.dev(&qkv_b)?,
            };
            let (ow, ob) = lin(&r, "self_attn.o_proj", D, D)?;
            let sa_o = Conv {
                w: r.plane(&ow, D, D, &p("self_attn.o_proj"))?,
                b: r.dev(&ob)?,
            };
            // cross-attention: q (256 -> 256) and k (64 -> 256) permuted;
            // v and o folded into one 64 -> 256
            let (cqw, cqb) = lin(&r, "cross_attn_image.q_proj", D, D)?;
            let (cqw, cqb) = permute_rows(&cqw, &cqb, &perm, D);
            let ca_q = Conv {
                w: r.plane(&cqw, D, D, &p("cross_attn_image.q_proj"))?,
                b: r.dev(&cqb)?,
            };
            let (ckw, ckb) = lin(&r, "cross_attn_image.k_proj", D, KV_IN)?;
            let (ckw, ckb) = permute_rows(&ckw, &ckb, &perm, KV_IN);
            let ca_k = Conv {
                w: r.plane(&ckw, KV_IN, D, &p("cross_attn_image.k_proj"))?,
                b: r.dev(&ckb)?,
            };
            let (cvw, cvb) = lin(&r, "cross_attn_image.v_proj", D, KV_IN)?;
            let (cow, cob) = lin(&r, "cross_attn_image.o_proj", D, D)?;
            let mut ovw = vec![0f32; D * KV_IN];
            let mut ovb = vec![0f32; D];
            for o in 0..D {
                let row = &cow[o * D..(o + 1) * D];
                for i2 in 0..KV_IN {
                    let s: f64 = (0..D)
                        .map(|m| row[m] as f64 * cvw[m * KV_IN + i2] as f64)
                        .sum();
                    ovw[o * KV_IN + i2] = s as f32;
                }
                let s: f64 = (0..D).map(|m| row[m] as f64 * cvb[m] as f64).sum();
                ovb[o] = (s + cob[o] as f64) as f32;
            }
            let ca_ov = Conv {
                w: r.plane(&ovw, KV_IN, D, &p("cross_attn_image.ov"))?,
                b: r.dev(&ovb)?,
            };
            layers.push(Layer {
                ln1: r.norm(&p("layer_norm1"), D)?,
                ln2: r.norm(&p("layer_norm2"), D)?,
                ln3: r.norm(&p("layer_norm3"), D)?,
                sa_qkv,
                sa_o,
                ca_q,
                ca_k,
                ca_ov,
                lin1: Conv {
                    w: r.linear(&p("linear1.weight"), &[FFN, D])?,
                    b: r.vec(&p("linear1.bias"), FFN)?,
                },
                lin2: Conv {
                    w: r.linear(&p("linear2.weight"), &[D, FFN])?,
                    b: r.vec(&p("linear2.bias"), D)?,
                },
            });
        }
        let norm = r.norm(&format!("{MA}.layer_norm"), D)?;
        let (cs, sn) = axial_rope_tables();
        let cs = r.dev(&cs)?;
        let sn = r.dev(&sn)?;
        let pos01: Vec<f32> = sine_pos_table(GRID, D, 10000.0)
            .into_iter()
            .map(|v| 0.1f32 * v)
            .collect();
        let pos01 = r.dev(&pos01)?;
        let weight_bytes = r.bytes;

        let f = |n: usize| exec.alloc(n);
        let h = |n: usize| exec.alloc_f16(n);
        let ws = Workspace {
            x: f(TOKENS * D)?,
            h16: h(TOKENS * D)?,
            qkv: f(TOKENS * 3 * D)?,
            q16: h(TOKENS * D)?,
            k16: h(TOKENS * D)?,
            v16: h(TOKENS * D)?,
            o16: h(TOKENS * D)?,
            proj: f(TOKENS * D)?,
            ffn: h(TOKENS * FFN)?,
            k32: f(MEMATTN_MAX_KEYS * D)?,
            kc16: h(MEMATTN_MAX_KEYS * D)?,
            o64: h(TOKENS * KV_IN)?,
            out: f(TOKENS * D)?,
        };
        Ok(Self {
            exec,
            layers,
            norm,
            cs,
            sn,
            pos01,
            ws,
            weight_bytes,
        })
    }

    pub fn weight_bytes(&self) -> u64 {
        self.weight_bytes
    }

    /// Condition the frame's tracker feature `src` (`[5184][256]` f32, the
    /// neck's) on one object's bank: `kin` = `f16(M + Mpos)` and `vmem` =
    /// `f16(M)`, both `[nk][64]`, the first `nrope` rows memory tokens (rope
    /// on their keys), the rest pointer tokens. The result is
    /// [`Self::output`], `[5184][256]` f32.
    pub fn run(
        &mut self,
        src: &CudaSlice<f32>,
        kin: &CudaSlice<f16>,
        vmem: &CudaSlice<f16>,
        nk: usize,
        nrope: usize,
    ) -> Result<(), GpuModelError> {
        if nk == 0 || nk > MEMATTN_MAX_KEYS || nrope > nk || !nrope.is_multiple_of(TOKENS) {
            return Err(GpuModelError::Unsupported(format!(
                "sam3 memory attention: {nk} keys, {nrope} of them memory tokens (at most \
                 {MEMATTN_MAX_KEYS}, whole frames of {TOKENS})"
            )));
        }
        if src.len() < TOKENS * D || kin.len() < nk * KV_IN || vmem.len() < nk * KV_IN {
            return Err(GpuModelError::Unsupported(
                "sam3 memory attention: inputs under the geometry".into(),
            ));
        }
        let exec = self.exec.clone();
        let ws = &mut self.ws;
        let rope = Some((&self.cs, &self.sn));
        let qscale = 1.0 / (D as f32).sqrt();
        exec.copy_region(src, 0, &mut ws.x, 0, TOKENS * D)?;
        exec.add_rows_bcast(&mut ws.x, &self.pos01, TOKENS, TOKENS, D)?;
        for l in &self.layers {
            // ---- self-attention ----
            exec.layernorm_f16(&ws.x, &l.ln1.w, &l.ln1.b, &mut ws.h16, TOKENS, D, LN_EPS)?;
            exec.matvec_batch_f16(&l.sa_qkv.w, &ws.h16, &mut ws.qkv, TOKENS)?;
            exec.bias_add(&mut ws.qkv, &l.sa_qkv.b, TOKENS, 3 * D)?;
            for (dst, off, rot, scale) in [
                (&mut ws.q16, 0, rope, qscale),
                (&mut ws.k16, D, rope, 1.0),
                (&mut ws.v16, 2 * D, None, 1.0),
            ] {
                exec.sam3_rope_rows_h(
                    &ws.qkv,
                    dst,
                    rot,
                    TOKENS,
                    3 * D,
                    off,
                    D,
                    TOKENS,
                    TOKENS,
                    TOKENS,
                    scale,
                )?;
            }
            for half in [0, D / 2] {
                exec.sam3_mem_attn_h(
                    &ws.q16,
                    &ws.k16,
                    &ws.v16,
                    &mut ws.o16,
                    TOKENS,
                    TOKENS,
                    1,
                    D / 2,
                    (D, half),
                    (D, half),
                )?;
            }
            exec.matvec_batch_f16(&l.sa_o.w, &ws.o16, &mut ws.proj, TOKENS)?;
            exec.add_bias_res(&mut ws.x, &ws.proj, &l.sa_o.b, TOKENS, D)?;

            // ---- cross-attention over the bank ----
            exec.layernorm_f16(&ws.x, &l.ln2.w, &l.ln2.b, &mut ws.h16, TOKENS, D, LN_EPS)?;
            exec.matvec_batch_f16(&l.ca_q.w, &ws.h16, &mut ws.proj, TOKENS)?;
            exec.bias_add(&mut ws.proj, &l.ca_q.b, TOKENS, D)?;
            exec.sam3_rope_rows_h(
                &ws.proj,
                &mut ws.q16,
                rope,
                TOKENS,
                D,
                0,
                D,
                TOKENS,
                TOKENS,
                TOKENS,
                qscale,
            )?;
            exec.matvec_batch_f16(&l.ca_k.w, kin, &mut ws.k32, nk)?;
            exec.bias_add(&mut ws.k32, &l.ca_k.b, nk, D)?;
            exec.sam3_rope_rows_h(
                &ws.k32,
                &mut ws.kc16,
                rope,
                nk,
                D,
                0,
                D,
                nk,
                nrope,
                TOKENS,
                1.0,
            )?;
            exec.sam3_mem_attn_h(
                &ws.q16,
                &ws.kc16,
                vmem,
                &mut ws.o64,
                TOKENS,
                nk,
                1,
                KV_IN,
                (KV_IN, 0),
                (KV_IN, 0),
            )?;
            exec.matvec_batch_f16(&l.ca_ov.w, &ws.o64, &mut ws.proj, TOKENS)?;
            exec.add_bias_res(&mut ws.x, &ws.proj, &l.ca_ov.b, TOKENS, D)?;

            // ---- the MLP ----
            exec.layernorm_f16(&ws.x, &l.ln3.w, &l.ln3.b, &mut ws.h16, TOKENS, D, LN_EPS)?;
            exec.matvec_batch_f16_h_relu(&l.lin1.w, &ws.h16, &mut ws.ffn, Some(&l.lin1.b), TOKENS)?;
            exec.matvec_batch_f16(&l.lin2.w, &ws.ffn, &mut ws.proj, TOKENS)?;
            exec.add_bias_res(&mut ws.x, &ws.proj, &l.lin2.b, TOKENS, D)?;
        }
        exec.layernorm(
            &ws.x,
            &self.norm.w,
            &self.norm.b,
            &mut ws.out,
            TOKENS,
            D,
            LN_EPS,
        )?;
        Ok(())
    }

    /// The last [`Self::run`]'s conditioned feature, `[5184][256]` f32.
    pub fn output(&self) -> &CudaSlice<f32> {
        &self.ws.out
    }

    /// 0.1 x the tracker neck's position table the input adds, `[5184][256]`
    /// (the gate holds it to Meta's).
    pub fn pos01(&self) -> &CudaSlice<f32> {
        &self.pos01
    }
}
