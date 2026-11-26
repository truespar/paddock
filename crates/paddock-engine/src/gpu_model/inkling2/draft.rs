//! The release's DSpark draft (`mtp.safetensors`), the speculative decode
//! drafter the checkpoint was trained with.
//!
//! Every target pass keeps four features a row - the residual stream going
//! into visits 5, 23 and 41 and the final normed output - and turns them,
//! per draft layer, into that layer's context K / V at the row's position
//! (a softmax mix of the taps, RMSNorm, the context projections, k's norm
//! and RoPE). A draft round runs a block of 8 rows a slot - the pending
//! token's normed embedding, then 7 masks - through the three dense layers
//! (bidirectional inside the block over the context before it, the local
//! layers windowed to 1024 keys), takes the LM head's logits of the 7 mask
//! rows, and lets a rank-256 recurrent head correct them step by step, each
//! step fed the previous step's pick; the picks are the drafts.
//!
//! The draft's K / V live beside the target's (kv.rs: class 3, bf16, the
//! full layer paged through the same block tables, the local ones in
//! rings); a slot's last 16 rows of features sit in a ring for the
//! recurrent head's context.

use std::path::Path;

use cudarc::driver::CudaSlice;
use paddock_models::ggml_type::GgmlType;
use paddock_models::inkling2 as ik;
use paddock_models::safetensors::{SafetensorsFile, StDtype};

use super::forward::Seg;
use super::{GpuInkling2, KV};
use crate::gpu::{GpuExecutor, Inkling2KvLayout, QuantTensor};
use crate::gpu_model::gpt_oss::GpuModelError;

/// Visits whose input is a feature (the fourth is the final normed output).
pub(crate) const TAPS: [usize; 3] = [5, 23, 41];
/// Feature planes a row: the taps and the final output.
pub(crate) const NFEAT: usize = TAPS.len() + 1;
/// A draft block: the anchor (pending token) and its masks.
pub(crate) const BLOCK: usize = 8;
pub(crate) const DRAFTS: usize = BLOCK - 1;
/// Rows of features a slot keeps (a verify chunk's worth and then some).
pub(crate) const FEAT_RING: usize = 16;
/// Draft class: bf16 K / V (the target's clipped fp8 class does not fit the
/// draft's unnormalized values).
pub(crate) const DRAFT_KV: u32 = 3;
const LAYERS: usize = 3;
const LOCAL: [bool; LAYERS] = [true, false, true];
const FFN: usize = 9216;
const Q: usize = ik::HEADS * ik::HEAD_DIM;
const QKV: usize = Q + 2 * KV;
const RANK: usize = 256;
const WINDOW: usize = 1024;
const ROPE_THETA: f32 = 1.0e6;
/// Key splits of a draft block attention: a full draft layer at a 22K
/// context walked ~1.4K keys a CTA at 16 (GB10: 1.4 ms a call).
const MAX_SPLITS: usize = 64;

pub(crate) struct DraftLayer {
    pub in_norm: CudaSlice<f32>,
    pub post_norm: CudaSlice<f32>,
    /// bf16: the mix's norm (context features of this layer).
    pub dflare: CudaSlice<u8>,
    /// q | k | v [5120][2304]; the context's k | v [2048][2304].
    pub qkv: QuantTensor,
    pub kv_ctx: QuantTensor,
    pub o: QuantTensor,
    pub gate: QuantTensor,
    pub up: QuantTensor,
    pub down: QuantTensor,
    /// bf16 [128]: the block's q / k norms and the context k's.
    pub q_norm: CudaSlice<u8>,
    pub k_norm: CudaSlice<u8>,
    pub k_norm_ctx: CudaSlice<u8>,
}

pub(crate) struct Draft {
    pub layers: Vec<DraftLayer>,
    pub norm: CudaSlice<f32>,
    pub mask: CudaSlice<f32>,
    /// softmax(fuse_weight) rows: one a draft layer, the last the head's.
    pub alpha: Vec<CudaSlice<f32>>,
    pub markov_norm: CudaSlice<u8>,
    /// The recurrent head: token table [vocab][256], the projection's
    /// [state | token] and [hidden | context] columns, its bias, and the LM
    /// head folded through `hidden_out` ([vocab][256] + [vocab]).
    pub w1: QuantTensor,
    pub rnn_s: QuantTensor,
    pub rnn_c: QuantTensor,
    pub rnn_bias: Vec<f32>,
    pub fold_w: QuantTensor,
    pub fold_b: CudaSlice<u8>,
    pub inv_freq: CudaSlice<f32>,
    pub sc: Option<DraftScratch>,
}

/// A round's planes for up to `slots` blocks.
pub(crate) struct DraftScratch {
    pub slots: usize,
    pub x: CudaSlice<f32>,
    pub res: CudaSlice<f32>,
    pub h: CudaSlice<f32>,
    pub qkv: CudaSlice<f32>,
    pub att: CudaSlice<f32>,
    pub g: CudaSlice<f32>,
    pub u: CudaSlice<f32>,
    pub part_o: CudaSlice<f32>,
    pub part_ml: CudaSlice<f32>,
    pub blk: CudaSlice<u32>,
    pub dpos: CudaSlice<u32>,
    pub dslot: CudaSlice<u32>,
    pub anchors: CudaSlice<u32>,
    pub emb: CudaSlice<f32>,
    pub hid: CudaSlice<f32>,
    pub base: CudaSlice<f32>,
    pub cat: CudaSlice<f32>,
    pub ctxm: CudaSlice<f32>,
    pub bias: CudaSlice<f32>,
    pub fgather: CudaSlice<f32>,
    pub frnn: CudaSlice<f32>,
    pub cats: CudaSlice<f32>,
    pub w1rows: CudaSlice<f32>,
    pub raw: CudaSlice<f32>,
    pub latent: CudaSlice<f32>,
    pub gemm: CudaSlice<f32>,
    pub best: CudaSlice<u8>,
    pub ids: CudaSlice<u32>,
    pub drafts: CudaSlice<u32>,
    /// The context store's planes, a prompt chunk's rows.
    pub ctx: CudaSlice<f32>,
    pub kvc: CudaSlice<f32>,
}

fn err(s: String) -> GpuModelError {
    GpuModelError::Unsupported(s)
}

/// A tensor's bf16 bytes after checking its shape.
fn bf16_raw<'a>(
    st: &'a SafetensorsFile,
    name: &str,
    shape: &[usize],
) -> Result<&'a [u8], GpuModelError> {
    let (t, b) = st
        .bytes(name)
        .ok_or_else(|| err(format!("inkling2 draft: {name} missing")))?;
    if t.dtype != StDtype::Bf16 || t.shape != shape {
        return Err(err(format!(
            "inkling2 draft: {name} is {:?} {:?}, expected bf16 {shape:?}",
            t.dtype, t.shape
        )));
    }
    Ok(b)
}

fn bf16_f32(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<2>()
        .0
        .iter()
        .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
        .collect()
}

fn plane(
    exec: &GpuExecutor,
    bytes: &[u8],
    n: usize,
    k: usize,
) -> Result<QuantTensor, GpuModelError> {
    Ok(QuantTensor {
        bytes: exec.to_device_u8(bytes)?,
        ty: GgmlType::Bf16,
        dims: vec![k, n],
    })
}

/// Columns [c0, c1) of a row-major bf16 [n][k] matrix.
fn columns(b: &[u8], n: usize, k: usize, c0: usize, c1: usize) -> Vec<u8> {
    (0..n)
        .flat_map(|r| b[2 * (r * k + c0)..2 * (r * k + c1)].iter().copied())
        .collect()
}

impl Draft {
    /// The draft of the release at `dir`, or None when the release has no
    /// `mtp.safetensors`, the pack lacks the draft's ops, or speculation is
    /// off (`PADDOCK_NO_SPEC`). `unembed` is the
    /// target's LM head ([vocab][2304] bf16), folded into the recurrent head.
    pub fn load(
        exec: &GpuExecutor,
        dir: &Path,
        unembed: &QuantTensor,
    ) -> Result<Option<Self>, GpuModelError> {
        let path = dir.join("mtp.safetensors");
        if !path.exists() || !exec.has_inkling2_draft() {
            return Ok(None);
        }
        // spec "off" loads no drafter: the planes, the pool's draft layer and
        // the draft rings all go back to the KV pool
        if std::env::var_os("PADDOCK_NO_SPEC").is_some() {
            tracing::info!("inkling2: PADDOCK_NO_SPEC - the DSpark draft left unloaded");
            return Ok(None);
        }
        let st = SafetensorsFile::open(&path).map_err(|e| err(format!("inkling2 draft: {e}")))?;
        let w = ik::WIDTH;
        let f32v = |name: &str, n: usize| -> Result<CudaSlice<f32>, GpuModelError> {
            Ok(exec.to_device(&bf16_f32(bf16_raw(&st, name, &[n])?))?)
        };
        let u8v = |name: &str, n: usize| -> Result<CudaSlice<u8>, GpuModelError> {
            Ok(exec.to_device_u8(bf16_raw(&st, name, &[n])?)?)
        };
        let mut layers = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let p = format!("layers.{i}");
            let a = format!("{p}.self_attn");
            let qkv: Vec<u8> = [
                bf16_raw(&st, &format!("{a}.q_proj.weight"), &[Q, w])?,
                bf16_raw(&st, &format!("{a}.k_proj.weight"), &[KV, w])?,
                bf16_raw(&st, &format!("{a}.v_proj.weight"), &[KV, w])?,
            ]
            .concat();
            let kv_ctx: Vec<u8> = [
                bf16_raw(&st, &format!("{a}.k_proj_ctx.weight"), &[KV, w])?,
                bf16_raw(&st, &format!("{a}.v_proj_ctx.weight"), &[KV, w])?,
            ]
            .concat();
            layers.push(DraftLayer {
                in_norm: f32v(&format!("{p}.input_layernorm.weight"), w)?,
                post_norm: f32v(&format!("{p}.post_attention_layernorm.weight"), w)?,
                dflare: u8v(&format!("{p}.dflare_norm.weight"), w)?,
                qkv: plane(exec, &qkv, QKV, w)?,
                kv_ctx: plane(exec, &kv_ctx, 2 * KV, w)?,
                o: plane(
                    exec,
                    bf16_raw(&st, &format!("{a}.o_proj.weight"), &[w, Q])?,
                    w,
                    Q,
                )?,
                gate: plane(
                    exec,
                    bf16_raw(&st, &format!("{p}.mlp.gate_proj.weight"), &[FFN, w])?,
                    FFN,
                    w,
                )?,
                up: plane(
                    exec,
                    bf16_raw(&st, &format!("{p}.mlp.up_proj.weight"), &[FFN, w])?,
                    FFN,
                    w,
                )?,
                down: plane(
                    exec,
                    bf16_raw(&st, &format!("{p}.mlp.down_proj.weight"), &[w, FFN])?,
                    w,
                    FFN,
                )?,
                q_norm: u8v(&format!("{a}.q_norm.weight"), ik::HEAD_DIM)?,
                k_norm: u8v(&format!("{a}.k_norm.weight"), ik::HEAD_DIM)?,
                k_norm_ctx: u8v(&format!("{a}.k_norm_ctx.weight"), ik::HEAD_DIM)?,
            });
        }
        // softmax over the taps, a row per draft layer and the head's last
        let fuse = bf16_f32(bf16_raw(&st, "fuse_weight", &[LAYERS + 1, NFEAT])?);
        let alpha = fuse
            .chunks(NFEAT)
            .map(|r| {
                let m = r.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f32> = r.iter().map(|v| (v - m).exp()).collect();
                let s: f32 = e.iter().sum();
                exec.to_device(&e.iter().map(|v| v / s).collect::<Vec<_>>())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mh = "markov_head";
        let rnn = bf16_raw(
            &st,
            &format!("{mh}.rnn_proj.weight"),
            &[3 * RANK, 2 * RANK + 2 * w],
        )?;
        let kr = 2 * RANK + 2 * w;
        let rnn_s = plane(
            exec,
            &columns(rnn, 3 * RANK, kr, 0, 2 * RANK),
            3 * RANK,
            2 * RANK,
        )?;
        let rnn_c = plane(
            exec,
            &columns(rnn, 3 * RANK, kr, 2 * RANK, kr),
            3 * RANK,
            2 * w,
        )?;
        let rnn_bias = bf16_f32(bf16_raw(&st, &format!("{mh}.rnn_proj.bias"), &[3 * RANK])?);
        let w1 = plane(
            exec,
            bf16_raw(&st, &format!("{mh}.markov_w1.weight"), &[ik::VOCAB, RANK])?,
            ik::VOCAB,
            RANK,
        )?;
        // neox RoPE frequencies, the reference's f32 rule
        let inv: Vec<f32> = (0..ik::HEAD_DIM / 2)
            .map(|i| 1.0 / ROPE_THETA.powf((2 * i) as f32 / ik::HEAD_DIM as f32))
            .collect();
        let (fold_w, fold_b) = fold(exec, &st, unembed)?;
        Ok(Some(Self {
            layers,
            norm: f32v("norm.weight", w)?,
            mask: f32v("mask_embed", w)?,
            alpha,
            markov_norm: u8v(&format!("{mh}.dflare_norm.weight"), w)?,
            w1,
            rnn_s,
            rnn_c,
            rnn_bias,
            fold_w,
            fold_b,
            inv_freq: exec.to_device(&inv)?,
            sc: None,
        }))
    }

    /// Planes for rounds of up to `slots` blocks and context stores of up
    /// to `rows` rows.
    pub fn alloc_scratch(
        &mut self,
        exec: &GpuExecutor,
        slots: usize,
        rows: usize,
    ) -> Result<(), GpuModelError> {
        let (w, v) = (ik::WIDTH, ik::VOCAB_REAL);
        let r8 = BLOCK * slots;
        let r7 = DRAFTS * slots;
        let a = |n: usize| exec.alloc(n.max(1)).map_err(GpuModelError::from);
        // the head's context bias, one row a (step, slot)
        let bias: Vec<f32> = (0..r7)
            .flat_map(|_| self.rnn_bias.iter().copied())
            .collect();
        self.sc = Some(DraftScratch {
            slots,
            x: a(r8 * w)?,
            res: a(r8 * w)?,
            h: a(r8 * w)?,
            qkv: a(r8 * QKV)?,
            att: a(r8 * Q)?,
            g: a(r8 * FFN)?,
            u: a(r8 * FFN)?,
            part_o: a(r8 * ik::HEADS * MAX_SPLITS * ik::HEAD_DIM)?,
            part_ml: a(r8 * ik::HEADS * MAX_SPLITS * 2)?,
            blk: exec.alloc_u32(2 * slots)?,
            dpos: exec.alloc_u32(r8)?,
            dslot: exec.alloc_u32(r8)?,
            anchors: exec.alloc_u32(slots)?,
            emb: a(slots * w)?,
            hid: a(r7 * w)?,
            base: a(r7 * v)?,
            cat: a(r7 * 2 * w)?,
            ctxm: a(r7 * 3 * RANK)?,
            bias: exec.to_device(&bias)?,
            fgather: a(slots * NFEAT * w)?,
            frnn: a(slots * w)?,
            cats: a(slots * 2 * RANK)?,
            w1rows: a(slots * RANK)?,
            raw: a(slots * 3 * RANK)?,
            latent: a(slots * RANK)?,
            gemm: a(slots * v)?,
            best: exec.alloc_u8(8 * slots)?,
            ids: exec.alloc_u32(slots)?,
            drafts: exec.alloc_u32(r7)?,
            ctx: a(rows * w)?,
            kvc: a(rows * 2 * KV)?,
        });
        Ok(())
    }
}

/// The LM head folded through the recurrent head's `hidden_out`:
/// `fold_w[v] = head[v] . W` ([vocab][256]), `fold_b[v] = head[v] . b`,
/// rounded to bf16 - on the device, 4096 head rows a pass.
fn fold(
    exec: &GpuExecutor,
    st: &SafetensorsFile,
    unembed: &QuantTensor,
) -> Result<(QuantTensor, CudaSlice<u8>), GpuModelError> {
    let (w, v) = (ik::WIDTH, ik::VOCAB_REAL);
    let proj = bf16_raw(st, "markov_head.hidden_out.weight", &[w, RANK])?;
    // W^T as a [256][2304] plane, the bias as a [1][2304] one
    let mut pt = vec![0u8; proj.len()];
    for d in 0..w {
        for j in 0..RANK {
            let (s, t) = (2 * (d * RANK + j), 2 * (j * w + d));
            pt[t..t + 2].copy_from_slice(&proj[s..s + 2]);
        }
    }
    let pt = plane(exec, &pt, RANK, w)?;
    let pb = plane(
        exec,
        bf16_raw(st, "markov_head.hidden_out.bias", &[w])?,
        1,
        w,
    )?;
    let mut fold_w = exec.alloc_u8(2 * v * RANK)?;
    let mut fold_b = exec.alloc_u8(2 * v)?;
    const CHUNK: usize = 4096;
    let mut x = exec.alloc(CHUNK * w)?;
    let mut yw = exec.alloc(CHUNK * RANK)?;
    let mut yb = exec.alloc(CHUNK)?;
    let mut ids = exec.alloc_u32(CHUNK)?;
    for v0 in (0..v).step_by(CHUNK) {
        let k = CHUNK.min(v - v0);
        let rows: Vec<u32> = (v0..v0 + k).map(|r| r as u32).collect();
        exec.upload_u32(&rows, &mut ids)?;
        exec.embed_gather_bf16(unembed, &ids, &mut x, w, k, 1.0)?;
        exec.bf16_gemm_rows(&pt, 0, RANK, &x, &mut yw, k)?;
        exec.bf16_gemm_rows(&pb, 0, 1, &x, &mut yb, k)?;
        exec.inkling2_copy2d_bf16((&yw, 0, RANK), (&mut fold_w, v0 * RANK, RANK), RANK, k)?;
        exec.inkling2_copy2d_bf16((&yb, 0, 1), (&mut fold_b, v0, 1), 1, k)?;
    }
    exec.synchronize()?;
    Ok((
        QuantTensor {
            bytes: fold_w,
            ty: GgmlType::Bf16,
            dims: vec![RANK, v],
        },
        fold_b,
    ))
}

/// Key splits of a draft block attention over `span` keys.
fn draft_splits(span: usize) -> usize {
    span.div_ceil(256).clamp(1, MAX_SPLITS)
}

impl GpuInkling2 {
    /// Whether the release's DSpark draft is attached.
    pub fn has_draft(&self) -> bool {
        self.draft.is_some()
    }

    /// One draft round for `slot` (holding every position before `pos`)
    /// anchored at token `anchor` at `pos` - the gate's probe.
    #[doc(hidden)]
    pub fn draft_probe(
        &mut self,
        slot: usize,
        anchor: u32,
        pos: usize,
    ) -> Result<[u32; DRAFTS], GpuModelError> {
        let d = self.draft_round(&[(slot, anchor, pos)], DRAFTS)?;
        Ok(std::array::from_fn(|i| d[0][i]))
    }

    /// Gates only: from now on every draft round keeps each step's pick
    /// margin a block (the softcapped logits' top-1 - top-2, read back to
    /// the host - slow); [`Self::draft_margins`] hands the last round's.
    #[doc(hidden)]
    pub fn trace_draft_margins(&mut self) {
        self.draft_margins = Some(Vec::new());
    }

    /// The last traced draft round's margins, `[block][step]`.
    #[doc(hidden)]
    pub fn draft_margins(&self) -> Option<&[Vec<f32>]> {
        self.draft_margins.as_deref()
    }

    /// Whether the draft is attached and its planes are seated.
    pub fn draft_ready(&self) -> bool {
        self.draft.as_ref().is_some_and(|d| d.sc.is_some()) && !self.kv.draft.is_empty()
    }

    /// After a pass over `segs` (its rows' features in `scratch.feat`, its
    /// row tables uploaded): each draft layer's context K / V at the rows'
    /// positions, and each slot's feature ring.
    pub(crate) fn draft_after_pass(
        &mut self,
        segs: &[Seg<'_>],
        n: usize,
    ) -> Result<(), GpuModelError> {
        let (Some(d), false) = (self.draft.as_mut(), self.kv.draft.is_empty()) else {
            return Ok(());
        };
        let Some(dsc) = d.sc.as_mut() else {
            return Ok(());
        };
        let exec = self.exec.clone();
        let e: &GpuExecutor = &exec;
        let w = ik::WIDTH;
        let sc = &mut self.scratch;
        let kv = &mut self.kv;
        let fs = NFEAT * w;
        // the context, a prompt chunk's rows at a time (the call's rows)
        for (i, ly) in d.layers.iter().enumerate() {
            e.inkling2_mix_norm(
                (&sc.feat, fs),
                NFEAT,
                &d.alpha[i],
                &ly.dflare,
                &mut dsc.ctx,
                n,
                w,
                ik::EPS,
            )?;
            e.bf16_gemm_rows(&ly.kv_ctx, 0, 2 * KV, &dsc.ctx, &mut dsc.kvc, n)?;
            e.inkling2_qk_rope(
                (&mut dsc.kvc, 2 * KV, 0),
                n,
                ik::KV_HEADS,
                Some(&ly.k_norm_ctx),
                &sc.pos,
                &d.inv_freq,
                ik::EPS,
            )?;
            let layout = if LOCAL[i] {
                Inkling2KvLayout::Ring(kv.ring)
            } else {
                Inkling2KvLayout::Paged(&kv.bt, kv.bps)
            };
            let store = &mut kv.draft[i];
            e.inkling2_kv_store(
                (&dsc.kvc, 0, 2 * KV),
                (&sc.slots, &sc.pos),
                &mut store.k,
                layout,
                DRAFT_KV,
                KV,
                n,
            )?;
            e.inkling2_kv_store(
                (&dsc.kvc, KV, 2 * KV),
                (&sc.slots, &sc.pos),
                &mut store.v,
                layout,
                DRAFT_KV,
                KV,
                n,
            )?;
        }
        // each segment's last rows into its slot's feature ring (positions
        // already advanced: the segment ends at lens)
        let ring = kv.feat_ring.as_mut().expect("feature ring with the draft");
        let mut base = 0;
        for s in segs {
            let len = s.ids.len();
            let end = kv.lens[s.slot];
            let keep = len.min(FEAT_RING);
            let mut p = end - keep;
            let mut r = base + len - keep;
            while p < end {
                let at = p % FEAT_RING;
                let k = (FEAT_RING - at).min(end - p);
                e.inkling2_copy2d(
                    (&sc.feat, r * fs, fs),
                    (ring, (s.slot * FEAT_RING + at) * fs, fs),
                    fs,
                    k,
                )?;
                p += k;
                r += k;
            }
            base += len;
        }
        Ok(())
    }

    /// One draft round: for each `(slot, anchor token, anchor position)` -
    /// the slot holds every position before the anchor - its first `depth`
    /// (<= 7) drafted tokens (the block always runs whole; the LM head and
    /// the recurrent steps only as deep as asked).
    pub(crate) fn draft_round(
        &mut self,
        items: &[(usize, u32, usize)],
        depth: usize,
    ) -> Result<Vec<Vec<u32>>, GpuModelError> {
        let depth = depth.clamp(1, DRAFTS);
        let n = items.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let exec = self.exec.clone();
        let e: &GpuExecutor = &exec;
        let (w, v) = (ik::WIDTH, ik::VOCAB_REAL);
        let Some(d) = self.draft.as_mut() else {
            return Err(err("inkling2: no draft attached".into()));
        };
        let dsc =
            d.sc.as_mut()
                .ok_or_else(|| err("inkling2: draft planes not seated".into()))?;
        if n > dsc.slots {
            return Err(err(format!(
                "inkling2 draft: {n} blocks past {}",
                dsc.slots
            )));
        }
        let kv = &mut self.kv;
        for &(slot, _, p) in items {
            if slot >= kv.slots || kv.lens[slot] < p || p == 0 || p + BLOCK > self.max_ctx {
                return Err(err(format!(
                    "inkling2 draft: slot {slot} at {p} holds {}",
                    kv.lens.get(slot).copied().unwrap_or(0)
                )));
            }
            // the block's own rows may reach a block the slot does not hold yet
            super::prefix::ensure_rows(e, kv, self.prefix.as_mut(), slot, p + BLOCK)?;
        }
        let r8 = BLOCK * n;
        let r7 = depth * n;
        let mut blk = Vec::with_capacity(2 * n);
        let (mut dpos, mut dslot, mut anchors) = (
            Vec::with_capacity(r8),
            Vec::with_capacity(r8),
            Vec::with_capacity(n),
        );
        let mut span_full = 0;
        for &(slot, tok, p) in items {
            blk.extend([slot as u32, p as u32]);
            dpos.extend((p..p + BLOCK).map(|q| q as u32));
            dslot.extend(std::iter::repeat_n(slot as u32, BLOCK));
            anchors.push(tok);
            span_full = span_full.max(p + BLOCK);
        }
        e.upload_u32(&blk, &mut dsc.blk)?;
        e.upload_u32(&dpos, &mut dsc.dpos)?;
        e.upload_u32(&dslot, &mut dsc.dslot)?;
        e.upload_u32(&anchors, &mut dsc.anchors)?;
        // the block: masks, then each anchor's normed embedding
        e.embed_gather_bf16(&self.embed, &dsc.anchors, &mut dsc.emb, w, n, 1.0)?;
        e.rmsnorm_batch(&dsc.emb, &self.embed_norm, &mut dsc.h, w, ik::EPS, n)?;
        e.inkling2_copy2d((&d.mask, 0, 0), (&mut dsc.x, 0, w), w, r8)?;
        e.inkling2_copy2d((&dsc.h, 0, w), (&mut dsc.x, 0, BLOCK * w), w, n)?;
        e.copy_region(&dsc.x, 0, &mut dsc.res, 0, r8 * w)?;
        let scale = 1.0 / (ik::HEAD_DIM as f32).sqrt();
        for (i, ly) in d.layers.iter().enumerate() {
            if i > 0 {
                e.add(&mut dsc.res, &dsc.x, r8 * w)?;
            }
            e.rmsnorm_batch(&dsc.res, &ly.in_norm, &mut dsc.h, w, ik::EPS, r8)?;
            e.bf16_gemm_rows(&ly.qkv, 0, QKV, &dsc.h, &mut dsc.qkv, r8)?;
            e.inkling2_qk_rope(
                (&mut dsc.qkv, QKV, 0),
                r8,
                ik::HEADS,
                Some(&ly.q_norm),
                &dsc.dpos,
                &d.inv_freq,
                ik::EPS,
            )?;
            e.inkling2_qk_rope(
                (&mut dsc.qkv, QKV, Q),
                r8,
                ik::KV_HEADS,
                Some(&ly.k_norm),
                &dsc.dpos,
                &d.inv_freq,
                ik::EPS,
            )?;
            let (layout, window, span) = if LOCAL[i] {
                (Inkling2KvLayout::Ring(kv.ring), WINDOW, WINDOW + BLOCK)
            } else {
                (Inkling2KvLayout::Paged(&kv.bt, kv.bps), 0, span_full)
            };
            {
                let store = &mut kv.draft[i];
                e.inkling2_kv_store(
                    (&dsc.qkv, Q, QKV),
                    (&dsc.dslot, &dsc.dpos),
                    &mut store.k,
                    layout,
                    DRAFT_KV,
                    KV,
                    r8,
                )?;
                e.inkling2_kv_store(
                    (&dsc.qkv, Q + KV, QKV),
                    (&dsc.dslot, &dsc.dpos),
                    &mut store.v,
                    layout,
                    DRAFT_KV,
                    KV,
                    r8,
                )?;
            }
            let splits = draft_splits(span);
            let part = (splits > 1).then_some((&mut dsc.part_o, &mut dsc.part_ml));
            let store = &kv.draft[i];
            e.inkling2_draft_attn(
                (&dsc.qkv, 0, QKV),
                (&dsc.blk, n),
                (&store.k, &store.v),
                layout,
                DRAFT_KV,
                (&mut dsc.att, Q),
                part,
                (ik::HEADS, ik::KV_HEADS),
                window,
                splits,
                scale,
            )?;
            e.bf16_gemm_rows(&ly.o, 0, w, &dsc.att, &mut dsc.x, r8)?;
            e.add(&mut dsc.res, &dsc.x, r8 * w)?;
            e.rmsnorm_batch(&dsc.res, &ly.post_norm, &mut dsc.h, w, ik::EPS, r8)?;
            e.bf16_gemm_rows(&ly.gate, 0, FFN, &dsc.h, &mut dsc.g, r8)?;
            e.bf16_gemm_rows(&ly.up, 0, FFN, &dsc.h, &mut dsc.u, r8)?;
            e.swiglu(&mut dsc.g, &dsc.u, r8 * FFN)?;
            e.bf16_gemm_rows(&ly.down, 0, w, &dsc.g, &mut dsc.x, r8)?;
        }
        e.add(&mut dsc.res, &dsc.x, r8 * w)?;
        e.rmsnorm_batch(&dsc.res, &d.norm, &mut dsc.h, w, ik::EPS, r8)?;
        // the mask rows, step-major (row step * n + block), and their logits
        for b in 0..n {
            e.inkling2_copy2d(
                (&dsc.h, (b * BLOCK + 1) * w, w),
                (&mut dsc.hid, b * w, n * w),
                w,
                depth,
            )?;
        }
        e.bf16_gemm_rows(&self.unembed, 0, v, &dsc.hid, &mut dsc.base, r7)?;
        // the head's context: each block's last committed row's features
        let ring = kv.feat_ring.as_ref().expect("feature ring with the draft");
        let fs = NFEAT * w;
        for (b, &(slot, _, p)) in items.iter().enumerate() {
            let at = (slot * FEAT_RING + (p - 1) % FEAT_RING) * fs;
            e.inkling2_copy2d((ring, at, fs), (&mut dsc.fgather, b * fs, fs), fs, 1)?;
        }
        e.inkling2_mix_norm(
            (&dsc.fgather, fs),
            NFEAT,
            &d.alpha[LAYERS],
            &d.markov_norm,
            &mut dsc.frnn,
            n,
            w,
            ik::EPS,
        )?;
        e.inkling2_copy2d((&dsc.hid, 0, w), (&mut dsc.cat, 0, 2 * w), w, r7)?;
        for b in 0..n {
            e.inkling2_copy2d(
                (&dsc.frnn, b * w, 0),
                (&mut dsc.cat, b * 2 * w + w, n * 2 * w),
                w,
                depth,
            )?;
        }
        e.bf16_gemm_rows(&d.rnn_c, 0, 3 * RANK, &dsc.cat, &mut dsc.ctxm, r7)?;
        e.add(&mut dsc.ctxm, &dsc.bias, r7 * 3 * RANK)?;
        let trace = self.draft_margins.is_some();
        let mut margins = vec![Vec::new(); if trace { n } else { 0 }];
        // the recurrence: state (cats[:, :256]) from zero, the anchor first
        e.zero_region(&mut dsc.cats, 0, n * 2 * RANK)?;
        e.copy_region(&dsc.anchors, 0, &mut dsc.ids, 0, n)?;
        for step in 0..depth {
            e.embed_gather_bf16(&d.w1, &dsc.ids, &mut dsc.w1rows, RANK, n, 1.0)?;
            e.inkling2_copy2d(
                (&dsc.w1rows, 0, RANK),
                (&mut dsc.cats, RANK, 2 * RANK),
                RANK,
                n,
            )?;
            e.bf16_gemm_rows(&d.rnn_s, 0, 3 * RANK, &dsc.cats, &mut dsc.raw, n)?;
            e.inkling2_markov_gate(
                &dsc.raw,
                (&dsc.ctxm, step * n * 3 * RANK),
                (&mut dsc.cats, 2 * RANK),
                &mut dsc.latent,
                n,
                RANK,
            )?;
            e.bf16_gemm_rows(&d.fold_w, 0, v, &dsc.latent, &mut dsc.gemm, n)?;
            if trace {
                // the pick's operands as markov_argmax rounds them
                let base = e.to_host_len(&dsc.base, (step + 1) * n * v)?;
                let gm = e.to_host_len(&dsc.gemm, n * v)?;
                let fb: Vec<f32> = e
                    .to_host_u8_len(&d.fold_b, 2 * v)?
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| half::bf16::from_le_bytes(*c).to_f32())
                    .collect();
                let bf = |x: f32| half::bf16::from_f32(x).to_f32();
                for (b, m) in margins.iter_mut().enumerate() {
                    let (mut t1, mut t2) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
                    for i in 0..v {
                        let x = bf(bf(base[(step * n + b) * v + i]) + bf(gm[b * v + i] + fb[i]));
                        let y = bf(ik::LOGIT_SOFTCAP * (x / ik::LOGIT_SOFTCAP).tanh());
                        if y > t1 {
                            t2 = t1;
                            t1 = y;
                        } else if y > t2 {
                            t2 = y;
                        }
                    }
                    m.push(t1 - t2);
                }
            }
            e.inkling2_markov_argmax(
                (&dsc.base, step * n * v),
                &dsc.gemm,
                &d.fold_b,
                &mut dsc.best,
                &mut dsc.ids,
                n,
                (v, v),
                ik::LOGIT_SOFTCAP,
            )?;
            e.copy_region(&dsc.ids, 0, &mut dsc.drafts, step * n, n)?;
        }
        let host = e.to_host_u32_len(&dsc.drafts, r7)?;
        if trace {
            self.draft_margins = Some(margins);
        }
        Ok((0..n)
            .map(|b| (0..depth).map(|step| host[step * n + b]).collect())
            .collect())
    }
}
