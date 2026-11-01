//! FlowLM's passes: the wide transformer over the prompt (and, at load, the
//! voice prompt) on the stored-weight tensor-core GEMM, Mimi's transformer on
//! the 3xTF32 one, and the one-row frame step - six layers of GEMVs and the
//! flow head - that the batch graphs replay.

use super::{
    BLOCK_MOD, CODEC_RING, CODEC_SPLITS, CODEC_WINDOW, COND_STRIDE, D, FF, FLOW_CACHE, FLOW_EPS,
    FLOW_SPLITS, GpuPhonon, LN_EPS, Layer, Tx,
};
use crate::gpu::{
    DiarEpi, DiarWeights, GpuError, GpuExecutor, KumoEpi, PhEpi, PhExtra, PhHint, PhPro, PhRope,
    PhWeight,
};

/// A wide pass's product on the GEMM its plane's class takes: 722 for
/// FlowLM's stored BF16 rows, 698 for Mimi's F32.
#[allow(clippy::too_many_arguments)]
fn gemm(
    exec: &GpuExecutor,
    x: &cudarc::driver::CudaSlice<f32>,
    w: &PhWeight,
    y: &mut cudarc::driver::CudaSlice<f32>,
    (k, n, m): (usize, usize, usize),
    resid: bool,
    gelu: bool,
) -> Result<(), GpuError> {
    match w {
        PhWeight::Bf16(b) => {
            let epi = match (resid, gelu) {
                (true, _) => DiarEpi::Resid,
                (_, true) => DiarEpi::Gelu,
                _ => DiarEpi::Store,
            };
            exec.diar_gemm((x, 0), DiarWeights::Bf16(b), None, y, (k, n, m), epi, None)
        }
        PhWeight::F32(f) => {
            let epi = match (resid, gelu) {
                (true, _) => KumoEpi::Resid,
                (_, true) => KumoEpi::Gelu,
                _ => KumoEpi::Store,
            };
            exec.kumo_gemm((x, 0), (f, 0), None, y, None, (k, n, m), epi, 1, (0, 0, 0))
        }
    }
}

/// One wide transformer stack in place over `rows` rows of `tx.x`, row r at
/// position `*tx.pos + r` (FlowLM: causal over its linear cache; Mimi: the
/// 251-row window over its ring, layer-scaled residuals).
fn wide(
    exec: &GpuExecutor,
    layers: &[Layer],
    caches: &mut [super::Cache],
    tx: &mut Tx,
    codec: bool,
    rows: usize,
) -> Result<(), GpuError> {
    let (cap, window, splits) = if codec {
        (CODEC_RING, CODEC_WINDOW, CODEC_SPLITS)
    } else {
        (FLOW_CACHE, 0, FLOW_SPLITS)
    };
    for (l, c) in layers.iter().zip(caches.iter_mut()) {
        let pos = &tx.pos;
        exec.ph_layernorm(&tx.x, &l.norm1.0, &l.norm1.1, &mut tx.xn, rows, D, LN_EPS)?;
        gemm(
            exec,
            &tx.xn,
            &l.in_proj,
            &mut tx.qkv,
            (D, 3 * D, rows),
            false,
            false,
        )?;
        exec.ph_rope_kv(&tx.qkv, &mut tx.q, &mut c.k, &mut c.v, pos, 0, rows, cap)?;
        exec.ph_attention(
            &tx.q,
            &c.k,
            &c.v,
            &mut tx.part,
            &mut tx.att,
            &mut tx.ctr,
            pos,
            0,
            (rows, cap, window, splits),
        )?;
        // x += out_proj(att) (Mimi: x += scale1 * out_proj(att), the
        // product staged in q, free once attention is done)
        match &l.scale {
            None => gemm(
                exec,
                &tx.att,
                &l.out_proj,
                &mut tx.x,
                (D, D, rows),
                true,
                false,
            )?,
            Some((s1, _)) => {
                gemm(
                    exec,
                    &tx.att,
                    &l.out_proj,
                    &mut tx.q,
                    (D, D, rows),
                    false,
                    false,
                )?;
                exec.ph_point(Some(&tx.q), 0, Some(s1), &mut tx.x, 0, rows * D, D, 5, 1.0)?;
            }
        }
        exec.ph_layernorm(&tx.x, &l.norm2.0, &l.norm2.1, &mut tx.xn, rows, D, LN_EPS)?;
        gemm(
            exec,
            &tx.xn,
            &l.linear1,
            &mut tx.wide,
            (D, FF, rows),
            false,
            true,
        )?;
        match &l.scale {
            None => gemm(
                exec,
                &tx.wide,
                &l.linear2,
                &mut tx.x,
                (FF, D, rows),
                true,
                false,
            )?,
            Some((_, s2)) => {
                gemm(
                    exec,
                    &tx.wide,
                    &l.linear2,
                    &mut tx.q,
                    (FF, D, rows),
                    false,
                    false,
                )?;
                exec.ph_point(Some(&tx.q), 0, Some(s2), &mut tx.x, 0, rows * D, D, 5, 1.0)?;
            }
        }
    }
    Ok(())
}

impl GpuPhonon {
    /// FlowLM over `rows` rows of its `x` at its position (the prompt).
    pub(super) fn wide_flow(&mut self, rows: usize) -> Result<(), GpuError> {
        let f = &mut self.flow;
        wide(
            &self.exec,
            &f.layers,
            &mut f.caches,
            &mut self.ws.tx,
            false,
            rows,
        )
    }

    /// Mimi's transformer over `rows` rows of its own `x`, on the codec lane.
    pub(super) fn wide_codec(&mut self, rows: usize) -> Result<(), GpuError> {
        let m = &mut self.mimi;
        wide(
            &self.codec,
            &m.layers,
            &mut m.caches,
            &mut self.cw.tx,
            true,
            rows,
        )
    }

    /// One frame's FlowLM row: the six layers on FlowLM's `x` row 0 at
    /// position `*pos + i`.
    fn frame_layers(&mut self, i: usize) -> Result<(), GpuError> {
        let exec = &self.exec;
        let w = &mut self.ws.tx;
        let stream = || PhExtra {
            hint: PhHint::Stream,
            ..PhExtra::default()
        };
        for (l, c) in self.flow.layers.iter().zip(self.flow.caches.iter_mut()) {
            // qkv with the rope and the cache writes in its epilogue
            exec.ph_linear(
                &w.x,
                0,
                PhPro::Ln(&l.norm1.0, &l.norm1.1),
                &l.in_proj,
                None,
                PhEpi::None,
                &mut w.q,
                0,
                (1, D, 3 * D),
                LN_EPS,
                PhExtra {
                    rope: Some(PhRope {
                        kc: &mut c.k,
                        vc: &mut c.v,
                        pos: &w.pos,
                        pos0: i,
                        cap: FLOW_CACHE,
                    }),
                    hint: PhHint::Stream,
                    ..PhExtra::default()
                },
            )?;
            exec.ph_attention(
                &w.q,
                &c.k,
                &c.v,
                &mut w.part,
                &mut w.att,
                &mut w.ctr,
                &w.pos,
                i,
                (1, FLOW_CACHE, 0, FLOW_SPLITS),
            )?;
            exec.ph_linear(
                &w.att,
                0,
                PhPro::None,
                &l.out_proj,
                None,
                PhEpi::Res(None),
                &mut w.x,
                0,
                (1, D, D),
                LN_EPS,
                stream(),
            )?;
            exec.ph_linear(
                &w.x,
                0,
                PhPro::Ln(&l.norm2.0, &l.norm2.1),
                &l.linear1,
                None,
                PhEpi::Gelu,
                &mut w.wide,
                0,
                (1, D, FF),
                LN_EPS,
                stream(),
            )?;
            exec.ph_linear(
                &w.wide,
                0,
                PhPro::None,
                &l.linear2,
                None,
                PhEpi::Res(None),
                &mut w.x,
                0,
                (1, FF, D),
                LN_EPS,
                stream(),
            )?;
        }
        Ok(())
    }

    /// The flow head on FlowLM's `x` row 0 for batch slot i: the EOS logit and
    /// the conditioning row into `ws.cond[i]`, the flow state `ws.flow[i]`
    /// (input_proj of the slot's noise, made for the whole batch first)
    /// through the six res blocks, the latent into `ws.latents[i]`. Every
    /// plane here is read again next frame and stays in L2.
    fn flow_head(&mut self, i: usize) -> Result<(), GpuError> {
        let exec = &self.exec;
        let f = &self.flow;
        let w = &mut self.ws;
        let keep = || PhExtra {
            hint: PhHint::Keep,
            ..PhExtra::default()
        };
        // [silu(cond_embed(h) + time) ; eos(h)], h = out_norm(x)
        exec.ph_linear(
            &w.tx.x,
            0,
            PhPro::Ln(&f.out_norm.0, &f.out_norm.1),
            &f.cond_eos.w,
            f.cond_eos.b.as_ref(),
            PhEpi::Silu,
            &mut w.cond,
            i * COND_STRIDE,
            (1, D, D + 1),
            LN_EPS,
            PhExtra {
                esplit: Some(D),
                hint: PhHint::Keep,
                ..PhExtra::default()
            },
        )?;
        exec.ph_linear(
            &w.cond,
            i * COND_STRIDE,
            PhPro::None,
            &f.modulation.w,
            f.modulation.b.as_ref(),
            PhEpi::None,
            &mut w.modulation,
            0,
            (1, D, BLOCK_MOD * 6 + 2 * D),
            LN_EPS,
            keep(),
        )?;
        for (bi, b) in f.blocks.iter().enumerate() {
            let m = bi * BLOCK_MOD;
            exec.ph_linear(
                &w.flow,
                i * D,
                PhPro::LnMod(&b.ln.0, &b.ln.1, &w.modulation, m),
                &b.mlp0.w,
                b.mlp0.b.as_ref(),
                PhEpi::Silu,
                &mut w.tmp,
                0,
                (1, D, D),
                FLOW_EPS,
                keep(),
            )?;
            exec.ph_linear(
                &w.tmp,
                0,
                PhPro::None,
                &b.mlp2.w,
                b.mlp2.b.as_ref(),
                PhEpi::Gated(None, &w.modulation, m + 2 * D),
                &mut w.flow,
                i * D,
                (1, D, D),
                FLOW_EPS,
                keep(),
            )?;
        }
        // latent = noise + final_linear(LN(x) * (1 + scale) + shift)
        exec.ph_linear(
            &w.flow,
            i * D,
            PhPro::PlainMod(&w.modulation, 6 * BLOCK_MOD),
            &f.final_linear.w,
            f.final_linear.b.as_ref(),
            PhEpi::Res(Some(&w.noise)),
            &mut w.latents,
            i * 32,
            (1, D, 32),
            FLOW_EPS,
            keep(),
        )
    }

    /// Frames into batch slots 0..n from `ws.noise`, frame i at FlowLM
    /// position `*pos + i`. `first`: the request's first frame (n = 1),
    /// whose transformer row the reset already ran.
    pub(super) fn sample_pass(&mut self, n: usize, first: bool) -> Result<(), GpuError> {
        {
            // every slot's flow state: input_proj(noise), one launch a batch
            let f = &self.flow;
            let w = &mut self.ws;
            self.exec.ph_linear(
                &w.noise,
                0,
                PhPro::None,
                &f.input_proj.w,
                f.input_proj.b.as_ref(),
                PhEpi::None,
                &mut w.flow,
                0,
                (n, 32, D),
                LN_EPS,
                PhExtra {
                    hint: PhHint::Keep,
                    ..PhExtra::default()
                },
            )?;
        }
        for i in 0..n {
            if !first {
                // x = input_linear(previous latent) + condition
                let w = &mut self.ws;
                let lin = &self.flow.input_linear;
                let (src, off) = if i == 0 {
                    (&w.prev, 0)
                } else {
                    (&w.latents, (i - 1) * 32)
                };
                self.exec.ph_linear(
                    src,
                    off,
                    PhPro::None,
                    &lin.w,
                    lin.b.as_ref(),
                    PhEpi::None,
                    &mut w.tx.x,
                    0,
                    (1, 32, D),
                    LN_EPS,
                    PhExtra {
                        hint: PhHint::Keep,
                        ..PhExtra::default()
                    },
                )?;
                self.frame_layers(i)?;
            }
            self.flow_head(i)?;
        }
        self.exec
            .copy_region(&self.ws.latents, (n - 1) * 32, &mut self.ws.prev, 0, 32)
    }
}
