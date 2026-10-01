//! One transformer block of the Kumo graph, two ways.
//!
//! The served block is fused: the RMSNorms ride the projections' operand
//! staging (the norm statistic is one pass shared by every norm of a plane),
//! the per-head rope / norm / log scaling ride the q and k projections'
//! epilogue, and the gated scaling is one pass. The reference block is the op
//! train one kernel a step - norm planes, raw projections, the gate plane. It
//! stays as the instrument the fused block is held to: every fused kernel
//! replays the arithmetic of the passes it removes, so the two agree to the
//! bit (the reference gate checks every case both ways).

use cudarc::driver::CudaSlice;

use super::{GpuKumo, GpuModelError, Kv};
use crate::gpu::{KumoEpi, KumoFuse, KumoHeads};

/// A block's use of the fitted context.
pub(super) enum Cache<'a> {
    Write(&'a mut Kv),
    Read(&'a Kv),
}

/// A block's geometry: `batch` independent sequences of `qlen` query and
/// `klen` key rows, keys gathered `kvstride` rows apart, `d` channels over
/// `heads` heads; `scaling` 0 none, 1 log, 2 gated log with rope; `qkvh`
/// the Test-GQA KV heads of query rows (0: none). `shared`: one set of `qlen`
/// query rows serves every batch (the inducing points).
#[derive(Clone, Copy)]
pub(super) struct Geo {
    pub batch: usize,
    pub qlen: usize,
    pub klen: usize,
    pub kvstride: usize,
    pub d: usize,
    pub heads: usize,
    pub scaling: u32,
    pub qkvh: usize,
    pub shared: bool,
}

/// Planes only the reference block uses - the normalized copies, the raw
/// projections and the gate plane - allocated only while it is on.
pub(super) struct RefPlanes {
    pub nq: CudaSlice<f32>,
    pub nk: CudaSlice<f32>,
    pub q: CudaSlice<f32>,
    pub k: CudaSlice<f32>,
    pub gate: CudaSlice<f32>,
}

/// Every block's working planes (`wide` twice a block plane) and the MLP
/// norm's per-row statistic; the fused path's rope tables.
pub(super) struct Scratch {
    pub qh: CudaSlice<f32>,
    pub kh: CudaSlice<f32>,
    pub v: CudaSlice<f32>,
    pub att: CudaSlice<f32>,
    pub tmp: CudaSlice<f32>,
    pub wide: CudaSlice<f32>,
    pub inv: CudaSlice<f32>,
    /// `[table][rope_seq][hd / 2]` (cos, sin) pairs (kumo.cuh 713), filled
    /// for rows of `rope_seq` tokens (0: not yet)
    pub rope: CudaSlice<f32>,
    pub rope_seq: usize,
    pub r: Option<RefPlanes>,
}

/// Rows a fused block reads: the plane, each of its rows' RMSNorm statistic,
/// and the gather that picks the block's rows out of it (`(len, stride)`:
/// row m is plane row `m / len * stride + m % len`).
#[derive(Clone, Copy)]
pub(super) struct Rows<'a> {
    pub x: &'a CudaSlice<f32>,
    pub inv: &'a CudaSlice<f32>,
    pub gather: Option<(usize, usize)>,
}

pub(super) const NO_STRIDES: (usize, usize, usize) = (0, 0, 0);

impl GpuKumo {
    /// The fused block. `q` / `kv` are the query and key/value rows (with
    /// their statistics); `resid` the query rows as the residual addend (one
    /// shared set, broadcast, when `g.shared`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn block_fused(
        &self,
        s: &mut Scratch,
        p: &str,
        q: Rows<'_>,
        kv: Rows<'_>,
        resid: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        g: Geo,
        mut cache: Option<Cache<'_>>,
    ) -> Result<(), GpuModelError> {
        let e = &*self.exec;
        let Geo {
            batch,
            qlen,
            klen,
            d,
            heads,
            scaling,
            shared,
            ..
        } = g;
        let hd = d / heads;
        let qrows = batch * qlen;
        let qin = if shared { qlen } else { qrows };
        let krows = batch * klen;
        let replay = matches!(cache, Some(Cache::Read(_)));
        let w = |n: &str| self.w(&format!("{p}.{n}"));
        let (qkv_w, qkv_b) = (w("attn.qkv_lin.weight"), w("attn.qkv_lin.bias"));
        let rope = scaling == 2;
        if rope && qlen.max(klen) > s.rope_seq {
            return Err(GpuModelError::Unsupported(
                "Kumo: the rope tables hold fewer positions than the row".into(),
            ));
        }
        // the rope table an inv_freq weight reads (see GpuKumo::rope)
        let table = |which: &str| self.rope[&format!("{p}.attn.{which}_transform.0.inv_freq")];
        let rope_stride = s.rope_seq * hd;
        // blocks without query scaling (the column output blocks) carry none
        let head_scale = (scaling > 0).then(|| w("attn.sdpa.query_scaling.head_scale"));
        // A one-key softmax is exactly one: the query side cannot change the
        // output, so it is not computed (the attention's one-key path copies v).
        if klen > 1 {
            // q = heads(norm(query) . Wq + bq), log-scaled here unless gated
            e.kumo_gemm_fused(
                q.x,
                (qkv_w, 0),
                Some((qkv_b, 0)),
                &mut s.qh,
                None,
                (d, d, qin),
                KumoEpi::Store,
                KumoFuse {
                    norm: Some((q.inv, w("query_norm.weight"))),
                    gather: q.gather,
                    heads: Some(KumoHeads {
                        hd,
                        seq: qlen,
                        rope: rope.then(|| (&s.rope, table("query") * rope_stride)),
                        scale: head_scale.filter(|_| scaling == 1).map(|hs| (hs, klen)),
                    }),
                },
            )?;
        }
        if !replay {
            // k | v: rows d..3d of the fused projection, heads on the k half
            e.kumo_gemm_fused(
                kv.x,
                (qkv_w, d * d),
                Some((qkv_b, d)),
                &mut s.kh,
                Some(&mut s.v),
                (d, 2 * d, krows),
                KumoEpi::Split,
                KumoFuse {
                    norm: Some((kv.inv, w("key_value_norm.weight"))),
                    gather: kv.gather,
                    heads: Some(KumoHeads {
                        hd,
                        seq: klen,
                        rope: rope.then(|| (&s.rope, table("key") * rope_stride)),
                        scale: None,
                    }),
                },
            )?;
        }
        if let Some(Cache::Write(kvc)) = &mut cache {
            // the first kvc.heads heads of every key row: all four for the
            // column blocks, the Test-GQA pair for the medium/large ICL
            let width = kvc.heads * hd;
            e.kumo_copy(&s.kh, &mut kvc.key, krows, width, d, width, krows)?;
            e.kumo_copy(&s.v, &mut kvc.value, krows, width, d, width, krows)?;
        }
        if let (true, Some(head_scale), true) = (rope, head_scale, klen > 1) {
            e.kumo_qgate(
                &mut s.qh,
                (
                    w("attn.sdpa.query_scaling.gate.0.weight"),
                    w("attn.sdpa.query_scaling.gate.0.bias"),
                    w("attn.sdpa.query_scaling.gate.2.weight"),
                    w("attn.sdpa.query_scaling.gate.2.bias"),
                ),
                head_scale,
                qin * heads,
                hd,
                heads,
                klen,
            )?;
        }
        self.attend(s, g, cache.as_ref())?;
        e.kumo_gemm(
            (&s.att, 0),
            (w("attn.out_lin.weight"), 0),
            Some((w("attn.out_lin.bias"), 0)),
            &mut s.tmp,
            None,
            (d, d, qrows),
            KumoEpi::Store,
            1,
            NO_STRIDES,
        )?;
        // out = query + attention; the MLP norm's statistic beside it
        e.kumo_stats(
            resid,
            Some((&s.tmp, if shared { qlen } else { qrows }, &mut *out)),
            &mut s.inv,
            d,
            qrows,
        )?;
        e.kumo_gemm_fused(
            out,
            (w("mlp.1.weight"), 0),
            Some((w("mlp.1.bias"), 0)),
            &mut s.wide,
            None,
            (d, 2 * d, qrows),
            KumoEpi::Gelu,
            KumoFuse {
                norm: Some((&s.inv, w("mlp.0.weight"))),
                ..KumoFuse::default()
            },
        )?;
        e.kumo_gemm(
            (&s.wide, 0),
            (w("mlp.3.weight"), 0),
            Some((w("mlp.3.bias"), 0)),
            out,
            None,
            (2 * d, d, qrows),
            KumoEpi::Resid,
            1,
            NO_STRIDES,
        )?;
        Ok(())
    }

    /// The reference block: the op train one kernel a step (see the module
    /// note). `query` is read as-is (one shared set when `g.shared`), `kv`
    /// gathered `g.kvstride` rows apart.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn block_reference(
        &self,
        s: &mut Scratch,
        p: &str,
        query: &CudaSlice<f32>,
        kv: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        g: Geo,
        mut cache: Option<Cache<'_>>,
    ) -> Result<(), GpuModelError> {
        let e = &*self.exec;
        let Geo {
            batch,
            qlen,
            klen,
            kvstride,
            d,
            heads,
            scaling,
            shared,
            ..
        } = g;
        let hd = d / heads;
        let qrows = batch * qlen;
        let qin = if shared { qlen } else { qrows };
        let krows = batch * klen;
        let replay = matches!(cache, Some(Cache::Read(_)));
        let w = |n: &str| self.w(&format!("{p}.{n}"));
        let r = s.r.as_mut().ok_or_else(|| {
            GpuModelError::Unsupported("Kumo-Tabular: reference planes not allocated".into())
        })?;
        if klen > 1 {
            e.kumo_norm(
                query,
                (w("query_norm.weight"), 0),
                &mut r.nq,
                d,
                qin,
                qin,
                qin,
            )?;
        }
        if !replay {
            e.kumo_norm(
                kv,
                (w("key_value_norm.weight"), 0),
                &mut r.nk,
                d,
                krows,
                klen,
                kvstride,
            )?;
        }
        let (qkv_w, qkv_b) = (w("attn.qkv_lin.weight"), w("attn.qkv_lin.bias"));
        if klen > 1 {
            e.kumo_gemm(
                (&r.nq, 0),
                (qkv_w, 0),
                Some((qkv_b, 0)),
                &mut r.q,
                None,
                (d, d, qin),
                KumoEpi::Store,
                1,
                NO_STRIDES,
            )?;
        }
        if !replay {
            // k | v: rows d..3d of the fused projection, one launch
            e.kumo_gemm(
                (&r.nk, 0),
                (qkv_w, d * d),
                Some((qkv_b, d)),
                &mut r.k,
                Some(&mut s.v),
                (d, 2 * d, krows),
                KumoEpi::Split,
                1,
                NO_STRIDES,
            )?;
        }
        let rope = scaling == 2;
        if klen > 1 {
            let freq = rope.then(|| w("attn.query_transform.0.inv_freq"));
            e.kumo_heads(&r.q, freq, &mut s.qh, heads, hd, qlen, qin * heads)?;
        }
        if !replay {
            let freq = rope.then(|| w("attn.key_transform.0.inv_freq"));
            e.kumo_heads(&r.k, freq, &mut s.kh, heads, hd, klen, krows * heads)?;
        }
        if let Some(Cache::Write(kvc)) = &mut cache {
            let width = kvc.heads * hd;
            e.kumo_copy(&s.kh, &mut kvc.key, krows, width, d, width, krows)?;
            e.kumo_copy(&s.v, &mut kvc.value, krows, width, d, width, krows)?;
        }
        if scaling > 0 && klen > 1 {
            if scaling == 2 {
                e.kumo_gemm(
                    (&s.qh, 0),
                    (w("attn.sdpa.query_scaling.gate.0.weight"), 0),
                    Some((w("attn.sdpa.query_scaling.gate.0.bias"), 0)),
                    &mut r.gate,
                    None,
                    (hd, 64, qin * heads),
                    KumoEpi::Gelu,
                    1,
                    NO_STRIDES,
                )?;
                e.kumo_gemm(
                    (&r.gate, 0),
                    (w("attn.sdpa.query_scaling.gate.2.weight"), 0),
                    Some((w("attn.sdpa.query_scaling.gate.2.bias"), 0)),
                    &mut r.q,
                    None,
                    (64, hd, qin * heads),
                    KumoEpi::Store,
                    1,
                    NO_STRIDES,
                )?;
            }
            e.kumo_scale(
                &mut s.qh,
                w("attn.sdpa.query_scaling.head_scale"),
                rope.then_some(&r.q),
                qin * d,
                hd,
                heads,
                klen,
            )?;
        }
        self.attend(s, g, cache.as_ref())?;
        let r = s.r.as_mut().expect("checked above");
        e.kumo_gemm(
            (&s.att, 0),
            (w("attn.out_lin.weight"), 0),
            Some((w("attn.out_lin.bias"), 0)),
            &mut s.tmp,
            None,
            (d, d, qrows),
            KumoEpi::Store,
            1,
            NO_STRIDES,
        )?;
        e.kumo_add_norm(
            query,
            &s.tmp,
            (w("mlp.0.weight"), 0),
            out,
            &mut r.nq,
            d,
            qrows,
            if shared { qlen } else { qrows },
        )?;
        e.kumo_gemm(
            (&r.nq, 0),
            (w("mlp.1.weight"), 0),
            Some((w("mlp.1.bias"), 0)),
            &mut s.wide,
            None,
            (d, 2 * d, qrows),
            KumoEpi::Gelu,
            1,
            NO_STRIDES,
        )?;
        e.kumo_gemm(
            (&s.wide, 0),
            (w("mlp.3.weight"), 0),
            Some((w("mlp.3.bias"), 0)),
            out,
            None,
            (2 * d, d, qrows),
            KumoEpi::Resid,
            1,
            NO_STRIDES,
        )?;
        Ok(())
    }

    /// Attention of the block's (scaled) queries `s.qh` over its keys - the
    /// block's own `s.kh` / `s.v`, or a fitted context's - into `s.att`.
    fn attend(
        &self,
        s: &mut Scratch,
        g: Geo,
        cache: Option<&Cache<'_>>,
    ) -> Result<(), GpuModelError> {
        let e = &*self.exec;
        let hd = g.d / g.heads;
        let replay = matches!(cache, Some(Cache::Read(_)));
        let (keys, values, kvh) = match cache {
            Some(Cache::Read(c)) => (&c.key, &c.value, c.heads),
            _ => (&s.kh, &s.v, g.heads),
        };
        let q_brows = if g.shared { 0 } else { g.qlen };
        let shape = (g.batch, g.qlen, g.klen, kvh);
        let heads = (g.heads, hd);
        if g.qkvh > 0 && g.qkvh < g.heads && replay {
            // every replayed row is a query row
            e.kumo_attention(
                &s.qh,
                keys,
                values,
                &mut s.att,
                heads,
                shape,
                q_brows,
                g.qkvh,
                (0, g.qlen),
            )?;
        } else if g.qkvh > 0 && g.qkvh < g.heads {
            // context rows on every head, query rows on the Test-GQA heads:
            // one launch each side of the boundary, never a mixed tile
            let split = g.klen.min(g.qlen);
            e.kumo_attention(
                &s.qh,
                keys,
                values,
                &mut s.att,
                heads,
                shape,
                q_brows,
                0,
                (0, split),
            )?;
            e.kumo_attention(
                &s.qh,
                keys,
                values,
                &mut s.att,
                heads,
                shape,
                q_brows,
                g.qkvh,
                (split, g.qlen - split),
            )?;
        } else {
            e.kumo_attention(
                &s.qh,
                keys,
                values,
                &mut s.att,
                heads,
                shape,
                q_brows,
                0,
                (0, g.qlen),
            )?;
        }
        Ok(())
    }
}
