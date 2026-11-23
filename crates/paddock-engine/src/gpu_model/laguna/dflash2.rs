//! Ripple's DFlash2 block-diffusion drafter
//! (`ricursiveAI/Ripple-65B-Preview-DFlash2`, BF16 safetensors, 4.04B) on
//! the laguna body. The model card and the reference's serving of it
//! (SGLang's `models/dflash.py` + the Spark bundle's draft patches) are the
//! study material; the kernels are the qwen35 / muse DFlash2 lane's.
//!
//! One drafter pass drafts a whole block per slot: rows = [committed token,
//! 15 x mask 100278] at positions p..p+15, embedded through the TARGET's
//! table at scale 1 (the target's own sqrt(width) lives in its forward, not
//! the table the drafter reads), five qwen3-style layers - per-head QK norm,
//! plain q|k|v, rope theta 1e4 on layers 0-3, layer 4 NoPE - each sublayer
//! wrapped in DFlash2's grouped dynamic convolution (`prepare` on its input,
//! `finish` on its output, both from one projection of the input), the
//! drafter's final norm, the target's BF16 head over the block's rows, and
//! the rank-256 candidate selector walking the top-32 candidates as a path.
//! Rows 1.. are the drafts.
//!
//! The block is NON-CAUSAL (`is_causal: false`): every block row attends
//! through the block end (the muse splice: attention position p + 15, rope
//! at the row's own position). Each row's sliding floor derives from that
//! bound - up to 15 of the window's oldest keys a row would not have read -
//! on drafts the verify judges anyway. The sliding layers see 4096 keys; the
//! NoPE full layer reaches 16384 back, the draft's training rows and what
//! the reference serves it at.
//!
//! Conditioning: z = hidden_norm(fc(concat h_i)), h_i the target's residual
//! LEAVING layers 1, 20, 39, 58, 77 (the layer walk's aux taps), and per
//! drafter layer k / v of z (k normed, roped unless NoPE) in POOL STRIPES -
//! one more full-layer-shaped store beside the target's, addressed by its
//! block tables, so a prefix-cache resume finds the drafter's rows with the
//! pages (nemotron's stripe, with its per-page generation stamp: a page
//! re-issued since - another tenant, or a tier restore that refills only the
//! target's planes - reads stale). With a 16K reach a per-slot ring could
//! never re-warm inside an agentic turn after a prefix hit.
//!
//! Numeric class: every drafter GEMM on per-row e4m3 twins of the BF16
//! planes (`bf16_to_f8row` at attach, 4.0 GB resident for the 8.1 GB file),
//! activations quantized per row; stripes at the target's KV dtype. Drafts
//! only move acceptance - the verify judges every token.

use std::collections::HashMap;
use std::path::Path;

use cudarc::driver::CudaSlice;

use crate::gpu::{F8RowPlane, GpuError, GpuExecutor, KvDtype, QuantTensor};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::nemotron::batch::{rows_groups, rows_partial_cap, rows_split};
use paddock_models::ggml_type::GgmlType;
use paddock_models::ripple::{self, draft};
use paddock_models::safetensors::{ShardedSafetensors, StDtype};

use super::batch::SPEC_ROWS;
use super::*;

/// Drafts a block can carry (the trained block minus its committed row).
pub(crate) const MAX_DRAFT: usize = draft::BLOCK - 1;

/// Rows one committed-row append carries at most (`d2_commit_rows`): the
/// size of its gather planes. A round's accepted rows rarely pass it; a
/// round that does commits in more chunks.
const COMMIT_ROWS: usize = 64;

fn drv(e: cudarc::driver::DriverError) -> GpuError {
    crate::gpu::from_driver(e)
}

/// A drafter plane: per-row e4m3 twin of the checkpoint's BF16 `[n, k]`.
pub(crate) struct D2Plane {
    w: F8RowPlane,
    k: usize,
    n: usize,
}

/// `y = W x` over `rows` rows whose e4m3 twin already sits in `q` / `rs`.
fn mm(
    exec: &GpuExecutor,
    w: &D2Plane,
    q: &CudaSlice<i8>,
    rs: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    rows: usize,
) -> Result<(), GpuError> {
    exec.f8row_gemm(&w.w, q, rs, y, w.k, w.n, rows)
}

/// DFlash2's grouped dynamic convolution around one sublayer: the base
/// kernel `[2, taps, embd]` (input side, output side) and the projection
/// to the per-group deltas of both sides.
pub(crate) struct D2Conv {
    base: CudaSlice<f32>,
    proj: D2Plane,
}

pub(crate) struct D2Layer {
    in_norm: CudaSlice<f32>,
    post_norm: CudaSlice<f32>,
    q_norm: CudaSlice<f32>,
    k_norm: CudaSlice<f32>,
    wq: D2Plane,
    wk: D2Plane,
    wv: D2Plane,
    wo: D2Plane,
    gate: D2Plane,
    up: D2Plane,
    down: D2Plane,
    attn_conv: D2Conv,
    mlp_conv: D2Conv,
    nope: bool,
    /// keys a block row reaches back, its own included
    window: usize,
}

/// The candidate selector: `score[e, p, c] = unary[e, c] + <pred[cand(e-1,
/// p)] * hidden(h_e), succ[cand(e, c)]>`, walked greedily slot to slot.
pub(crate) struct D2Selector {
    hidden: D2Plane,
    /// bf16 `[vocab, rank]` codebooks, gathered by candidate id
    pred: QuantTensor,
    succ: QuantTensor,
}

pub(crate) struct Dflash2 {
    layers: Vec<D2Layer>,
    /// fc split into one `[embd -> embd]` band per tap (concat order)
    fc: Vec<D2Plane>,
    hidden_norm: CudaSlice<f32>,
    final_norm: CudaSlice<f32>,
    sel: D2Selector,
    pub(crate) taps: Vec<usize>,
    eps: f32,
    /// the drafter's NEOX rope (theta 1e4, full rotary) as yarn params
    rope: (f32, f32, f32, f32, f32, f32),
    ff: usize,
    pub(crate) state: Option<D2State>,
    /// the draft-tree rounds' armed trees and compaction planes (`tree`)
    pub(super) tree: super::tree::TreeState,
}

/// Serving-time state (built at enable_batch).
pub(crate) struct D2State {
    /// per-layer feature K / V as pool stripes `[pool_blocks, 16, kv_dim]`
    /// at the target's KV dtype
    kv_k: Vec<CudaSlice<u8>>,
    kv_v: Vec<CudaSlice<u8>>,
    kv_dtype: KvDtype,
    /// per-slot live feature span [start, end)
    pub(crate) feat: Vec<(u32, u32)>,
    /// per pool block: the generation under which its sixteen drafter rows
    /// were completed by a live span (0 = never)
    page_gen: Vec<u32>,
    /// aux bands, `[taps][band][embd]` - the layer walk copies each tapped
    /// residual's rows here (band = the batch scratch's row capacity)
    pub(crate) aux: CudaSlice<f32>,
    pub(crate) band: usize,
    ctx: CudaSlice<f32>,
    ctxn: CudaSlice<f32>,
    acc: CudaSlice<f32>,
    kp: CudaSlice<f32>,
    kn: CudaSlice<f32>,
    vp: CudaSlice<f32>,
    /// per-row e4m3 activations + row scales, for every drafter GEMM
    f8q: CudaSlice<i8>,
    f8rs: CudaSlice<f32>,
    /// blocks one draft pass carries
    pub(crate) draft_slots: usize,
    d_toks: CudaSlice<u32>,
    d_pos: CudaSlice<u32>,
    d_apos: CudaSlice<u32>,
    d_slots: CudaSlice<u32>,
    d_groups: CudaSlice<u32>,
    x: CudaSlice<f32>,
    xn: CudaSlice<f32>,
    cvx: CudaSlice<f32>,
    cvc: CudaSlice<f32>,
    q: CudaSlice<f32>,
    qn: CudaSlice<f32>,
    k: CudaSlice<f32>,
    kn_r: CudaSlice<f32>,
    v: CudaSlice<f32>,
    attn: CudaSlice<f32>,
    attn_o: CudaSlice<f32>,
    attn_ml: CudaSlice<f32>,
    proj: CudaSlice<f32>,
    g: CudaSlice<f32>,
    u: CudaSlice<f32>,
    sinks: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    sel_params: CudaSlice<u32>,
    sel_topk: CudaSlice<u32>,
    sel_ids: CudaSlice<u32>,
    sel_pred: CudaSlice<f32>,
    sel_succ: CudaSlice<f32>,
    sel_hs: CudaSlice<f32>,
    d_out: CudaSlice<u32>,
    /// the selector's whole edge lattice of the last pass (slot 974): scores
    /// `[blocks * 15, K, K]`, candidate ids `[blocks * 15, K]` - a draft tree
    /// reads every predecessor's edges, the greedy walk one
    lat_scores: CudaSlice<f32>,
    lat_ids: CudaSlice<u32>,
    /// the committed rows' tap gather: its row map and landing plane
    /// (`[taps][COMMIT_ROWS][embd]`)
    cgidx: CudaSlice<u32>,
    cgtmp: CudaSlice<f32>,
    /// captured draft passes keyed by block count
    graphs: HashMap<usize, super::SendGraph>,
}

/// A slot's feature span after rows [start, start + n) commit: contiguous
/// growth, a rewrite of the tail from a row inside it, or - past a gap - a
/// new span from `start` (nemotron's rule).
fn span_note((s, e): (usize, usize), start: usize, n: usize) -> (usize, usize) {
    let s = if start >= s && start <= e { s } else { start };
    (s, start + n)
}

/// Warm for a block at `pos`: the span ends there and reaches back as far as
/// the block's furthest read - row 0, or the full layer's reach.
fn span_warm((s, e): (usize, usize), pos: usize) -> bool {
    e == pos && (s == 0 || s + draft::FULL_REACH <= pos + 1)
}

fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
        .collect()
}

impl GpuLaguna {
    /// Attach the DFlash2 drafter (its safetensors dir, or the file in it).
    /// Validates the config contract and every tensor against the target,
    /// builds the e4m3 twins; serving state builds at enable_batch.
    pub fn attach_dflash2(&mut self, path: &Path) -> Result<(), GpuModelError> {
        let fam = "ripple-dflash2";
        if self.hp.flavor != Flavor::Ripple {
            return Err(GpuModelError::Unsupported(format!(
                "{}: the DFlash2 drafter is Ripple's",
                self.hp.flavor.name()
            )));
        }
        let dir = if path.is_dir() {
            path
        } else {
            path.parent().unwrap_or(path)
        };
        draft::check(dir).map_err(|e| GpuModelError::Unsupported(format!("{fam} config: {e}")))?;
        let exec = self.exec.clone();
        let needs = [
            ("bf16_to_f8row", exec.has_bf16_to_f8row()),
            ("f8row_gemm", exec.has_f8row_gemm()),
            ("dflash_conv", exec.has_dflash_conv()),
            ("dflash_select", exec.has_dflash_select()),
            ("topk_rows", exec.has_topk_rows()),
            ("attn_rows_partial", exec.has_attn_rows_partial()),
            ("argmax_rows", exec.has_argmax_rows()),
            ("embed_gather_bf16", exec.has_embed_gather_bf16()),
        ];
        if let Some((op, _)) = needs.iter().find(|(_, ok)| !ok) {
            return Err(GpuModelError::Unsupported(format!(
                "{fam}: the kernel pack lacks {op}"
            )));
        }
        if !matches!(self.tok_embd, TokEmbd::Bf16(_)) || !matches!(self.lm_head, Head::Bf16(_)) {
            return Err(GpuModelError::Unsupported(format!(
                "{fam}: drafts embed and score through the target's BF16 table and head"
            )));
        }
        let st = ShardedSafetensors::open_dir(dir)
            .map_err(|e| GpuModelError::Unsupported(format!("{fam} shards: {e}")))?;
        let n_tensors = st.names().count();
        let want = draft::LAYERS * 15 + 6;
        if n_tensors != want {
            return Err(GpuModelError::Unsupported(format!(
                "{fam}: {n_tensors} tensors, the DFlash2 schema has {want}"
            )));
        }
        // planes land at half the file (e4m3), the codebooks at their bf16
        let file: u64 = st
            .names()
            .filter_map(|n| st.bytes(n).map(|(_, b)| b.len() as u64))
            .sum();
        exec.vram_load_gate(file / 2 + 4 * (ripple::VOCAB * draft::SEL_RANK) as u64, fam)
            .map_err(GpuModelError::WontFit)?;

        let (embd, hd, ff) = (ripple::WIDTH, ripple::HEAD_DIM, ripple::FF);
        let (q_dim, kv_dim) = (ripple::HEADS * hd, ripple::KV_HEADS * hd);
        let conv_dim = 2 * draft::CONV_TAPS * (embd / draft::CONV_GROUP);
        let raw = |name: &str, shape: &[usize]| -> Result<&[u8], GpuModelError> {
            let (t, b) = st
                .bytes(name)
                .ok_or_else(|| GpuModelError::Unsupported(format!("{fam} {name}: missing")))?;
            if t.dtype != StDtype::Bf16 || t.shape != shape {
                return Err(GpuModelError::Unsupported(format!(
                    "{fam} {name}: {:?} {:?}, expected BF16 {shape:?}",
                    t.dtype, t.shape
                )));
            }
            Ok(b)
        };
        let plane = |name: &str, n: usize, k: usize| -> Result<D2Plane, GpuModelError> {
            let w = exec.bf16_to_f8row(raw(name, &[n, k])?, k, n)?;
            Ok(D2Plane { w, k, n })
        };
        let vec_f32 = |name: &str, shape: &[usize]| -> Result<CudaSlice<f32>, GpuModelError> {
            Ok(exec.to_device(&bf16_to_f32(raw(name, shape)?))?)
        };
        let conv = |p: &str| -> Result<D2Conv, GpuModelError> {
            Ok(D2Conv {
                base: vec_f32(&format!("{p}.base_kernel"), &[2, draft::CONV_TAPS, embd])?,
                proj: plane(&format!("{p}.kernel_projection.weight"), conv_dim, embd)?,
            })
        };
        let mut layers = Vec::with_capacity(draft::LAYERS);
        for i in 0..draft::LAYERS {
            let p = format!("layers.{i}");
            let a = |s: &str| format!("{p}.self_attn.{s}");
            let full = i == draft::NOPE_LAYER;
            layers.push(D2Layer {
                in_norm: vec_f32(&format!("{p}.input_layernorm.weight"), &[embd])?,
                post_norm: vec_f32(&format!("{p}.post_attention_layernorm.weight"), &[embd])?,
                q_norm: vec_f32(&a("q_norm.weight"), &[hd])?,
                k_norm: vec_f32(&a("k_norm.weight"), &[hd])?,
                wq: plane(&a("q_proj.weight"), q_dim, embd)?,
                wk: plane(&a("k_proj.weight"), kv_dim, embd)?,
                wv: plane(&a("v_proj.weight"), kv_dim, embd)?,
                wo: plane(&a("o_proj.weight"), embd, q_dim)?,
                gate: plane(&format!("{p}.mlp.gate_proj.weight"), ff, embd)?,
                up: plane(&format!("{p}.mlp.up_proj.weight"), ff, embd)?,
                down: plane(&format!("{p}.mlp.down_proj.weight"), embd, ff)?,
                attn_conv: conv(&format!("{p}.attention_conv"))?,
                mlp_conv: conv(&format!("{p}.mlp_conv"))?,
                nope: full,
                window: if full {
                    draft::FULL_REACH
                } else {
                    draft::WINDOW
                },
            });
        }
        // fc [embd, taps * embd] -> one [embd, embd] band per tap: band i is
        // columns [i * embd, (i + 1) * embd) of every row
        let n_taps = draft::TAPS.len();
        let fc_raw = raw("fc.weight", &[embd, n_taps * embd])?;
        let mut fc = Vec::with_capacity(n_taps);
        for i in 0..n_taps {
            let mut band = Vec::with_capacity(embd * embd * 2);
            for row in 0..embd {
                let at = (row * n_taps * embd + i * embd) * 2;
                band.extend_from_slice(&fc_raw[at..at + embd * 2]);
            }
            fc.push(D2Plane {
                w: exec.bf16_to_f8row(&band, embd, embd)?,
                k: embd,
                n: embd,
            });
        }
        let codebook = |name: &str| -> Result<QuantTensor, GpuModelError> {
            Ok(QuantTensor {
                bytes: exec.to_device_u8(raw(name, &[ripple::VOCAB, draft::SEL_RANK])?)?,
                ty: GgmlType::Bf16,
                dims: vec![draft::SEL_RANK, ripple::VOCAB],
            })
        };
        let sel = D2Selector {
            hidden: plane(
                "candidate_selector.hidden_projection.weight",
                draft::SEL_RANK,
                embd,
            )?,
            pred: codebook("candidate_selector.predecessor_codebook")?,
            succ: codebook("candidate_selector.successor_codebook")?,
        };
        let d2 = Dflash2 {
            layers,
            fc,
            hidden_norm: vec_f32("hidden_norm.weight", &[embd])?,
            final_norm: vec_f32("norm.weight", &[embd])?,
            sel,
            taps: draft::TAPS.to_vec(),
            eps: ripple::EPS,
            rope: self.hp.rope_swa,
            ff,
            state: None,
            tree: Default::default(),
        };
        let bytes: u64 = d2
            .layers
            .iter()
            .flat_map(|l| {
                [
                    &l.wq,
                    &l.wk,
                    &l.wv,
                    &l.wo,
                    &l.gate,
                    &l.up,
                    &l.down,
                    &l.attn_conv.proj,
                    &l.mlp_conv.proj,
                ]
            })
            .chain(d2.fc.iter())
            .chain(std::iter::once(&d2.sel.hidden))
            .map(|p| (p.w.data.len() + 4 * p.w.scale.len()) as u64)
            .sum::<u64>()
            + (d2.sel.pred.bytes.len() + d2.sel.succ.bytes.len()) as u64;
        self.weights_bytes += bytes;
        tracing::info!(
            "{fam} drafter attached: {} layers (layer {} NoPE, reach {}), block {}, taps {:?}, \
             selector top-{} rank {}, {:.2} GB as e4m3 twins",
            draft::LAYERS,
            draft::NOPE_LAYER,
            draft::FULL_REACH,
            draft::BLOCK,
            draft::TAPS,
            draft::SEL_TOP_K,
            draft::SEL_RANK,
            bytes as f64 / 1e9
        );
        // the round's rows: the lane's verify cap (the depth itself is the
        // backend's, `spec_fixed_draft_depth`; explicit env wins). SAFETY:
        // model load runs before the serving threads spawn.
        if std::env::var_os("PADDOCK_SPEC_MAX_ROWS").is_none() {
            crate::envset::set_env("PADDOCK_SPEC_MAX_ROWS", &SPEC_ROWS.to_string());
        }
        self.dflash2 = Some(d2);
        Ok(())
    }

    /// Bytes the drafter's stripes add to every pool block (its layers' K
    /// and V over the block's 16 rows) - the KV plan prices a block at the
    /// target's layers plus this.
    pub(crate) fn d2_stripe_bytes(&self) -> usize {
        let kv_dim = self.hp.n_kv_heads * self.hp.head_dim;
        self.dflash2.as_ref().map_or(0, |_| {
            draft::LAYERS * 2 * 16 * kv_dim * self.kv_dtype.bytes()
        })
    }

    /// Build the drafter's serving state against the batch lane's pool and
    /// row capacity.
    pub(crate) fn d2_ensure_state(&mut self) -> Result<(), GpuModelError> {
        let Some(d2) = self.dflash2.as_mut() else {
            return Ok(());
        };
        let bs = self.batch.as_ref().expect("batch enabled");
        let (n_slots, band, pool_blocks) = (bs.n_slots, bs.cap, bs.pool.capacity() as usize);
        let e = &self.exec;
        let hp = &self.hp;
        let (embd, hd, vocab) = (hp.n_embd, hp.head_dim, hp.n_vocab);
        let (nh, n_kv) = (ripple::HEADS, hp.n_kv_heads);
        let (q_dim, kv_dim) = (nh * hd, n_kv * hd);
        let conv_dim = 2 * draft::CONV_TAPS * (embd / draft::CONV_GROUP);
        // a verify round takes SPEC_ROWS rows, a pending one per slot at
        // least - at most that many over a block's rows can draft
        let draft_slots = n_slots.clamp(1, SPEC_ROWS / draft::BLOCK);
        let rows = draft_slots * draft::BLOCK;
        let (top_k, rank) = (draft::SEL_TOP_K, draft::SEL_RANK);
        let kvb = self.kv_dtype.bytes();
        let stripe = pool_blocks * 16 * kv_dim * kvb;
        let mut kv_k = Vec::with_capacity(draft::LAYERS);
        let mut kv_v = Vec::with_capacity(draft::LAYERS);
        for _ in 0..draft::LAYERS {
            kv_k.push(e.alloc_u8(stripe)?);
            kv_v.push(e.alloc_u8(stripe)?);
        }
        let rows_cap = rows_partial_cap(n_kv, e.sm_count(), rows);
        let wide = embd.max(q_dim).max(d2.ff).max(conv_dim);
        let st = D2State {
            kv_k,
            kv_v,
            kv_dtype: self.kv_dtype,
            feat: vec![(0, 0); n_slots],
            page_gen: vec![0; pool_blocks],
            aux: e.alloc(d2.taps.len() * band * embd)?,
            band,
            ctx: e.alloc(band * embd)?,
            ctxn: e.alloc(band * embd)?,
            acc: e.alloc(band * embd)?,
            kp: e.alloc(band * kv_dim)?,
            kn: e.alloc(band * kv_dim)?,
            vp: e.alloc(band * kv_dim)?,
            f8q: e.alloc_i8((band * embd).max(rows * wide))?,
            f8rs: e.alloc(band.max(rows))?,
            draft_slots,
            d_toks: e.alloc_u32(rows)?,
            d_pos: e.alloc_u32(rows)?,
            d_apos: e.alloc_u32(rows)?,
            d_slots: e.alloc_u32(rows)?,
            d_groups: e.alloc_u32(2 * draft_slots * draft::BLOCK.div_ceil(8))?,
            x: e.alloc(rows * embd)?,
            xn: e.alloc(rows * embd)?,
            cvx: e.alloc(rows * embd)?,
            cvc: e.alloc(rows * conv_dim)?,
            q: e.alloc(rows * q_dim)?,
            qn: e.alloc(rows * q_dim)?,
            k: e.alloc(rows * kv_dim)?,
            kn_r: e.alloc(rows * kv_dim)?,
            v: e.alloc(rows * kv_dim)?,
            attn: e.alloc(rows * q_dim)?,
            attn_o: e.alloc(nh * rows_cap * hd)?,
            attn_ml: e.alloc(nh * rows_cap * 2)?,
            proj: e.alloc(rows * embd)?,
            g: e.alloc(rows * d2.ff)?,
            u: e.alloc(rows * d2.ff)?,
            sinks: e.alloc_no_sinks(nh)?,
            logits: e.alloc(rows * vocab)?,
            // static mode-4 row table for topk_rows (the qwen35 lane's)
            sel_params: e
                .to_device_u32(&(0..rows).flat_map(|_| [0, 0, 4, 0]).collect::<Vec<_>>())?,
            sel_topk: e.alloc_u32(rows * top_k * 2)?,
            sel_ids: e.alloc_u32(rows * top_k + draft_slots)?,
            sel_pred: e.alloc((rows * top_k + draft_slots) * rank)?,
            sel_succ: e.alloc(rows * top_k * rank)?,
            sel_hs: e.alloc(rows * rank)?,
            d_out: e.alloc_u32(rows)?,
            lat_scores: e.alloc(draft_slots * (draft::BLOCK - 1) * top_k * top_k)?,
            lat_ids: e.alloc_u32(draft_slots * (draft::BLOCK - 1) * top_k)?,
            cgidx: e.alloc_u32(d2.taps.len() * COMMIT_ROWS)?,
            cgtmp: e.alloc(d2.taps.len() * COMMIT_ROWS * embd)?,
            graphs: HashMap::new(),
        };
        tracing::info!(
            "ripple-dflash2 state: {n_slots} slots, stripes {:.2} GB ({} blocks), {draft_slots} \
             blocks a draft pass",
            (2 * draft::LAYERS * stripe) as f64 / 1e9,
            pool_blocks
        );
        d2.state = Some(st);
        Ok(())
    }

    /// A slot's new sequence: its pages went back to the pool, and with them
    /// whatever span described them.
    pub(crate) fn d2_reset_slot(&mut self, slot: usize) {
        if let Some(st) = self.dflash2.as_mut().and_then(|d| d.state.as_mut())
            && let Some(f) = st.feat.get_mut(slot)
        {
            *f = (0, 0);
        }
    }

    /// A prefix resume at `pos` over the pages the radix just shared into the
    /// slot's table: the span is the trailing run of adopted pages whose
    /// stripe rows are still the ones a live span completed.
    pub(crate) fn d2_adopt_slot(&mut self, slot: usize, pos: usize) {
        let Some(bs) = self.batch.as_ref() else {
            return;
        };
        let Some(st) = self.dflash2.as_mut().and_then(|d| d.state.as_mut()) else {
            return;
        };
        if slot >= st.feat.len() {
            return;
        }
        let blocks = bs.tables[slot].blocks();
        let full = (pos / 16).min(blocks.len());
        let mut first = full;
        while first > 0 {
            let b = blocks[first - 1];
            let g = st.page_gen[b as usize];
            if g == 0 || g != bs.pool.generation(b) {
                break;
            }
            first -= 1;
        }
        let s = if full * 16 < pos { pos } else { first * 16 };
        st.feat[slot] = (s as u32, pos as u32);
    }

    /// Can the drafter draft a block for `slot` at `pos`?
    pub(crate) fn d2_warm(&self, slot: usize, pos: usize) -> bool {
        self.dflash2
            .as_ref()
            .and_then(|d| d.state.as_ref())
            .and_then(|st| st.feat.get(slot))
            .is_some_and(|&(s, e)| span_warm((s as usize, e as usize), pos))
    }

    /// Rows [start, start + n) of `slot` carry their features now: grow the
    /// span and stamp every page it covers whole with its pool generation.
    fn d2_note_rows(&mut self, slot: usize, start: usize, n: usize) {
        let Some(bs) = self.batch.as_ref() else {
            return;
        };
        let Some(st) = self.dflash2.as_mut().and_then(|d| d.state.as_mut()) else {
            return;
        };
        if slot >= st.feat.len() || n == 0 {
            return;
        }
        let (s, e) = span_note(
            (st.feat[slot].0 as usize, st.feat[slot].1 as usize),
            start,
            n,
        );
        st.feat[slot] = (s as u32, e as u32);
        let blocks = bs.tables[slot].blocks();
        let done = (e / 16).min(blocks.len());
        let from = (start / 16).max(s.div_ceil(16)).min(done);
        for &b in &blocks[from..done] {
            st.page_gen[b as usize] = bs.pool.generation(b);
        }
    }

    /// Fuse the walk's tapped rows into context states and append their K/V
    /// to every drafter layer's stripes. `positions` / `slots` are the rows'
    /// host mirrors (their device twins still sit in the batch scratch);
    /// `spans` = only those row ranges (a verify commit's accepted rows).
    pub(crate) fn d2_append(
        &mut self,
        positions: &[u32],
        slots: &[u32],
        spans: Option<&[(usize, usize)]>,
    ) -> Result<(), GpuModelError> {
        let r = positions.len();
        if r == 0 {
            return Ok(());
        }
        let exec = self.exec.clone();
        let hp = &self.hp;
        let (embd, hd, n_kv) = (hp.n_embd, hp.head_dim, hp.n_kv_heads);
        let kv_dim = n_kv * hd;
        let Some(d2) = self.dflash2.as_mut() else {
            return Ok(());
        };
        let Some(st) = d2.state.as_mut() else {
            return Ok(());
        };
        let bs = self.batch.as_ref().expect("batch enabled");
        assert!(r <= st.band, "feature append past the aux band");
        let (sc, d_bt, bps) = (&bs.sc, &bs.d_bt, bs.bps);
        // z = hidden_norm(sum_i fc_i(h_i)): one band GEMM a tap
        for (i, w) in d2.fc.iter().enumerate() {
            exec.copy_region(&st.aux, i * st.band * embd, &mut st.acc, 0, r * embd)?;
            exec.quantize_e4m3_row(&st.acc, &mut st.f8q, &mut st.f8rs, embd, r)?;
            if i == 0 {
                mm(&exec, w, &st.f8q, &st.f8rs, &mut st.ctx, r)?;
            } else {
                mm(&exec, w, &st.f8q, &st.f8rs, &mut st.ctxn, r)?;
                exec.add(&mut st.ctx, &st.ctxn, r * embd)?;
            }
        }
        exec.rmsnorm_batch(&st.ctx, &d2.hidden_norm, &mut st.ctxn, embd, d2.eps, r)?;
        exec.quantize_e4m3_row(&st.ctxn, &mut st.f8q, &mut st.f8rs, embd, r)?;
        let dt = st.kv_dtype;
        for (l, ly) in d2.layers.iter().enumerate() {
            mm(&exec, &ly.wk, &st.f8q, &st.f8rs, &mut st.kp, r)?;
            mm(&exec, &ly.wv, &st.f8q, &st.f8rs, &mut st.vp, r)?;
            exec.rmsnorm_batch(&st.kp, &ly.k_norm, &mut st.kn, hd, d2.eps, r * n_kv)?;
            if !ly.nope {
                exec.rope_yarn_batch(&mut st.kn, &sc.d_pos, n_kv, hd, d2.rope, r)?;
            }
            let (pk, pv) = (&mut st.kv_k[l], &mut st.kv_v[l]);
            match spans {
                None => {
                    let (p, s) = (&sc.d_pos, Some(&sc.d_slots));
                    exec.kv_append_batch_paged(&st.kn, pk, p, s, d_bt, bps, kv_dim, r, dt)?;
                    exec.kv_append_batch_paged(&st.vp, pv, p, s, d_bt, bps, kv_dim, r, dt)?;
                }
                Some(sp) => {
                    for &(off, len) in sp {
                        let (p, s) = (&sc.d_pos, Some(&sc.d_slots));
                        exec.kv_append_batch_paged_rows(
                            &st.kn, pk, p, s, d_bt, bps, kv_dim, off, len, dt,
                        )?;
                        exec.kv_append_batch_paged_rows(
                            &st.vp, pv, p, s, d_bt, bps, kv_dim, off, len, dt,
                        )?;
                    }
                }
            }
        }
        // coverage: the same-slot runs the rows (or the spans) make
        let runs: Vec<(usize, usize)> = match spans {
            Some(sp) => sp.to_vec(),
            None => {
                let mut runs = Vec::new();
                let mut i = 0;
                while i < r {
                    let mut j = i + 1;
                    while j < r && slots[j] == slots[i] && positions[j] == positions[j - 1] + 1 {
                        j += 1;
                    }
                    runs.push((i, j - i));
                    i = j;
                }
                runs
            }
        };
        for (row, len) in runs {
            self.d2_note_rows(slots[row] as usize, positions[row] as usize, len);
        }
        Ok(())
    }

    /// The service's spec draft: every warm slot's block in one drafter pass
    /// (more slots than a pass carries take another). `pendings` are (slot,
    /// committed token); a slot drafts at its span's end, the pending token's
    /// position. Cold slots, a slot at the context's end or one whose block
    /// rows the pool cannot back draft nothing this round.
    pub(crate) fn d2_draft_batch(
        &mut self,
        pendings: &[(usize, u32)],
        k: usize,
    ) -> Result<Option<Vec<Vec<u32>>>, GpuModelError> {
        let Some(st) = self.dflash2.as_ref().and_then(|d| d.state.as_ref()) else {
            return Ok(None);
        };
        let (feat, cap) = (st.feat.clone(), st.draft_slots);
        let k = k.min(MAX_DRAFT);
        if k == 0 || pendings.is_empty() {
            return Ok(None);
        }
        let mut reqs: Vec<(usize, usize, usize, u32)> = Vec::new();
        for (i, &(slot, tok)) in pendings.iter().enumerate() {
            let Some(&(_, e)) = feat.get(slot) else {
                continue;
            };
            let p = e as usize;
            if p == 0 || p + draft::BLOCK > self.max_ctx || !self.d2_warm(slot, p) {
                continue;
            }
            // the block's own K/V land in the slot's pages past its committed
            // end - back them (the verify backs only its depth)
            match self.ensure_full_rows(&[slot as u32], &[(p + draft::BLOCK - 1) as u32]) {
                Ok(()) => reqs.push((i, slot, p, tok)),
                Err(GpuModelError::PoolExhausted) => {}
                Err(e) => return Err(e),
            }
        }
        if reqs.is_empty() {
            return Ok(None);
        }
        // a verify round takes SPEC_ROWS rows, one pending row per slot
        let kr = k.min(SPEC_ROWS.saturating_sub(pendings.len()) / reqs.len());
        if kr == 0 {
            return Ok(None);
        }
        let mut out = vec![Vec::new(); pendings.len()];
        for chunk in reqs.chunks(cap) {
            let picks = self.d2_draft_pass(chunk)?;
            for (b, &(i, ..)) in chunk.iter().enumerate() {
                out[i] = picks[b * draft::BLOCK + 1..b * draft::BLOCK + 1 + kr].to_vec();
            }
        }
        Ok(Some(out))
    }

    /// A verify round's committed rows into the drafter: `rows` are (the
    /// row in the round, its true position, its slot), `spans` the slots'
    /// runs over `rows`. The rows go to the front of the aux bands in
    /// chunks of at most COMMIT_ROWS (whole spans; a chunk already at the
    /// front skips the gather) and append at their positions.
    pub(crate) fn d2_commit_rows(
        &mut self,
        rows: &[(u32, u32, u32)],
        spans: &[(usize, usize)],
    ) -> Result<(), GpuModelError> {
        let mut i = 0;
        while i < spans.len() {
            let first = spans[i].0;
            let mut j = i;
            while j < spans.len() && spans[j].0 + spans[j].1 - first <= COMMIT_ROWS {
                j += 1;
            }
            let j = j.max(i + 1);
            let chunk = &rows[first..spans[j - 1].0 + spans[j - 1].1];
            let sp: Vec<(usize, usize)> =
                spans[i..j].iter().map(|&(o, n)| (o - first, n)).collect();
            self.d2_commit_chunk(chunk, &sp)?;
            i = j;
        }
        Ok(())
    }

    fn d2_commit_chunk(
        &mut self,
        rows: &[(u32, u32, u32)],
        spans: &[(usize, usize)],
    ) -> Result<(), GpuModelError> {
        let n = rows.len();
        let pos: Vec<u32> = rows.iter().map(|r| r.1).collect();
        let slots: Vec<u32> = rows.iter().map(|r| r.2).collect();
        let exec = self.exec.clone();
        let embd = self.hp.n_embd;
        if rows.iter().enumerate().any(|(k, r)| r.0 as usize != k) {
            let d2 = self.dflash2.as_mut().expect("drafter");
            let taps = d2.taps.len();
            let st = d2.state.as_mut().expect("state");
            assert!(n <= COMMIT_ROWS, "a commit chunk past its planes");
            let band = st.band;
            let idx: Vec<u32> = (0..taps)
                .flat_map(|t| rows.iter().map(move |r| (t * band) as u32 + r.0))
                .collect();
            let mut v = st
                .cgidx
                .try_slice_mut(0..idx.len())
                .ok_or_else(|| GpuError::Driver("commit gather map".into()))?;
            exec.stream.memcpy_htod(&idx, &mut v).map_err(drv)?;
            exec.hrow_gather(&st.aux, &st.cgidx, &mut st.cgtmp, taps * n, embd)?;
            for t in 0..taps {
                exec.copy_region(
                    &st.cgtmp,
                    t * n * embd,
                    &mut st.aux,
                    t * st.band * embd,
                    n * embd,
                )?;
            }
        }
        let sc = &mut self.batch.as_mut().expect("batch enabled").sc;
        let mut v = sc
            .d_pos
            .try_slice_mut(0..n)
            .ok_or_else(|| GpuError::Driver("d_pos".into()))?;
        exec.stream.memcpy_htod(&pos, &mut v).map_err(drv)?;
        let mut v = sc
            .d_slots
            .try_slice_mut(0..n)
            .ok_or_else(|| GpuError::Driver("d_slots".into()))?;
        exec.stream.memcpy_htod(&slots, &mut v).map_err(drv)?;
        self.d2_append(&pos, &slots, Some(spans))
    }

    /// The selector lattice of the last draft pass over its first `n`
    /// blocks (slot 974), read back: per block `(ids [15 * K], scores [15 *
    /// K * K])`, position-major, the predecessor row before the candidate.
    pub(crate) fn d2_lattice(
        &mut self,
        n: usize,
    ) -> Result<Vec<(Vec<u32>, Vec<f32>)>, GpuModelError> {
        let exec = self.exec.clone();
        let st = self
            .dflash2
            .as_mut()
            .and_then(|d| d.state.as_mut())
            .expect("dflash2 state");
        let (k, e) = (draft::SEL_TOP_K, draft::BLOCK - 1);
        exec.dflash_lattice(
            &st.sel_topk,
            &st.sel_pred,
            &st.sel_succ,
            &st.sel_hs,
            &mut st.lat_scores,
            &mut st.lat_ids,
            (1.0, 0.0),
            draft::SEL_RANK,
            k,
            draft::BLOCK,
            n * draft::BLOCK,
        )?;
        let (ids, scores) = (
            exec.to_host_u32(&st.lat_ids)?,
            exec.to_host(&st.lat_scores)?,
        );
        Ok((0..n)
            .map(|b| {
                (
                    ids[b * e * k..(b + 1) * e * k].to_vec(),
                    scores[b * e * k * k..(b + 1) * e * k * k].to_vec(),
                )
            })
            .collect())
    }

    /// Stage `act`'s blocks (request index, slot, position, committed token)
    /// and run the pass - one captured replay per block count; returns the
    /// selector's picks, `[blocks, 16]`.
    pub(super) fn d2_draft_pass(
        &mut self,
        act: &[(usize, usize, usize, u32)],
    ) -> Result<Vec<u32>, GpuModelError> {
        let n = act.len();
        let rows = n * draft::BLOCK;
        let exec = self.exec.clone();
        {
            let st = self
                .dflash2
                .as_mut()
                .and_then(|d| d.state.as_mut())
                .expect("dflash2 state");
            let mut toks = Vec::with_capacity(rows);
            let mut pos = Vec::with_capacity(rows);
            let mut apos = Vec::with_capacity(rows);
            let mut slots = Vec::with_capacity(rows);
            for &(_, slot, p, tok) in act {
                toks.push(tok);
                toks.extend(std::iter::repeat_n(draft::MASK, draft::BLOCK - 1));
                pos.extend(p as u32..(p + draft::BLOCK) as u32);
                // the non-causal block: every row attends through its end
                apos.extend(std::iter::repeat_n(
                    (p + draft::BLOCK - 1) as u32,
                    draft::BLOCK,
                ));
                slots.extend(std::iter::repeat_n(slot as u32, draft::BLOCK));
            }
            let groups = rows_groups((0..n).map(|i| (i * draft::BLOCK, draft::BLOCK)));
            for (host, dev) in [
                (&toks, &mut st.d_toks),
                (&pos, &mut st.d_pos),
                (&apos, &mut st.d_apos),
                (&slots, &mut st.d_slots),
                (&groups, &mut st.d_groups),
            ] {
                let mut v = dev
                    .try_slice_mut(0..host.len())
                    .ok_or_else(|| GpuError::Driver("dflash2 row stage".into()))?;
                exec.stream.memcpy_htod(host, &mut v).map_err(drv)?;
            }
        }
        // one replay per block count: every per-row input is staged device
        // data and the launches key on n alone (the laguna v1 discipline:
        // first sight runs eagerly, then records the same launch stream)
        let have = self
            .dflash2
            .as_ref()
            .and_then(|d| d.state.as_ref())
            .is_some_and(|st| st.graphs.contains_key(&n));
        if paddock_models::dev_var_os!("PADDOCK_SPEC_NOGRAPH").is_some() {
            self.d2_draft_body(n)?;
        } else if !have {
            self.d2_draft_body(n)?;
            let g = self.capture_body(|s| s.d2_draft_body(n), "dflash2 draft")?;
            self.dflash2
                .as_mut()
                .and_then(|d| d.state.as_mut())
                .expect("dflash2 state")
                .graphs
                .insert(n, g);
        } else {
            self.dflash2
                .as_ref()
                .and_then(|d| d.state.as_ref())
                .expect("dflash2 state")
                .graphs[&n]
                .0
                .launch()
                .map_err(|e| GpuError::Driver(format!("dflash2 draft graph launch: {e}")))?;
        }
        let st = self
            .dflash2
            .as_ref()
            .and_then(|d| d.state.as_ref())
            .expect("dflash2 state");
        let v = st
            .d_out
            .try_slice(0..rows)
            .ok_or_else(|| GpuError::Driver("dflash2 picks".into()))?;
        Ok(exec.stream.clone_dtoh(&v).map_err(drv)?)
    }

    /// The draft pass's device body (capture-safe): `n` staged blocks
    /// through the five layers, the final norm, the target head and the
    /// selector into `d_out`.
    fn d2_draft_body(&mut self, n: usize) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let hp = &self.hp;
        let (embd, hd, vocab, max_ctx) = (hp.n_embd, hp.head_dim, hp.n_vocab, self.max_ctx);
        let (nh, n_kv) = (ripple::HEADS, hp.n_kv_heads);
        let (q_dim, kv_dim) = (nh * hd, n_kv * hd);
        let scale = 1.0 / (hd as f32).sqrt();
        let (taps, gsz) = (draft::CONV_TAPS, draft::CONV_GROUP);
        let ng = embd / gsz;
        let blk = draft::BLOCK;
        let r = n * blk;
        let n_groups = n * blk.div_ceil(8);
        let ns = rows_split(n_kv, n_groups, exec.sm_count());
        let (TokEmbd::Bf16(table), Head::Bf16(head)) = (&self.tok_embd, &self.lm_head) else {
            return Err(GpuModelError::Unsupported(
                "dflash2: drafts read the target's BF16 table and head".into(),
            ));
        };
        let bs = self.batch.as_ref().expect("batch enabled");
        let (d_bt, bps) = (&bs.d_bt, bs.bps);
        let d2 = self.dflash2.as_mut().expect("dflash2");
        let ff = d2.ff;
        let st = d2.state.as_mut().expect("dflash2 state");
        let dt = st.kv_dtype;
        // the committed token and the masks through the target's table, raw
        exec.embed_gather_bf16(table, &st.d_toks, &mut st.x, embd, r, 1.0)?;
        // `prepare`: the projection off a sublayer's input lands both sides'
        // deltas in cvc, side 0 convolves the input into cvx
        let prepare = |st: &mut D2State, cv: &D2Conv| -> Result<(), GpuError> {
            exec.quantize_e4m3_row(&st.xn, &mut st.f8q, &mut st.f8rs, embd, r)?;
            mm(&exec, &cv.proj, &st.f8q, &st.f8rs, &mut st.cvc, r)?;
            exec.dflash_conv(
                &st.xn,
                &mut st.cvx,
                &cv.base,
                &st.cvc,
                0,
                embd,
                taps,
                ng,
                gsz,
                blk,
                r,
            )?;
            exec.quantize_e4m3_row(&st.cvx, &mut st.f8q, &mut st.f8rs, embd, r)
        };
        // `finish`: side 1 over the sublayer's output, onto the residual
        let finish = |st: &mut D2State, cv: &D2Conv| -> Result<(), GpuError> {
            exec.dflash_conv(
                &st.proj,
                &mut st.cvx,
                &cv.base,
                &st.cvc,
                1,
                embd,
                taps,
                ng,
                gsz,
                blk,
                r,
            )?;
            exec.add(&mut st.x, &st.cvx, r * embd)
        };
        for (l, ly) in d2.layers.iter().enumerate() {
            exec.rmsnorm_batch(&st.x, &ly.in_norm, &mut st.xn, embd, d2.eps, r)?;
            prepare(st, &ly.attn_conv)?;
            mm(&exec, &ly.wq, &st.f8q, &st.f8rs, &mut st.q, r)?;
            mm(&exec, &ly.wk, &st.f8q, &st.f8rs, &mut st.k, r)?;
            mm(&exec, &ly.wv, &st.f8q, &st.f8rs, &mut st.v, r)?;
            exec.rmsnorm_batch(&st.q, &ly.q_norm, &mut st.qn, hd, d2.eps, r * nh)?;
            exec.rmsnorm_batch(&st.k, &ly.k_norm, &mut st.kn_r, hd, d2.eps, r * n_kv)?;
            if !ly.nope {
                exec.rope_yarn_batch(&mut st.qn, &st.d_pos, nh, hd, d2.rope, r)?;
                exec.rope_yarn_batch(&mut st.kn_r, &st.d_pos, n_kv, hd, d2.rope, r)?;
            }
            // the block's K/V land in the slots' pages at their true
            // positions (rewritten by the real features as they commit;
            // never inside a page the radix shares)
            let (pk, pv) = (&mut st.kv_k[l], &mut st.kv_v[l]);
            let (p, s) = (&st.d_pos, Some(&st.d_slots));
            exec.kv_append_batch_paged(&st.kn_r, pk, p, s, d_bt, bps, kv_dim, r, dt)?;
            exec.kv_append_batch_paged(&st.v, pv, p, s, d_bt, bps, kv_dim, r, dt)?;
            // every block row through the block end, one group per 8 rows
            // reading its slot's context once
            exec.attn_rows_partial(
                &st.qn,
                &st.kv_k[l],
                &st.kv_v[l],
                &mut st.attn_o,
                &mut st.attn_ml,
                &st.d_apos,
                &st.d_slots,
                &st.d_groups,
                n_groups,
                Some((d_bt, bps)),
                max_ctx,
                nh,
                n_kv,
                hd,
                kv_dim,
                r,
                ns,
                ly.window,
                scale,
                dt,
            )?;
            exec.attn_combine_batch(
                &st.attn_o,
                &st.attn_ml,
                &st.sinks,
                &mut st.attn,
                nh,
                hd,
                ns,
                r,
            )?;
            exec.quantize_e4m3_row(&st.attn, &mut st.f8q, &mut st.f8rs, q_dim, r)?;
            mm(&exec, &ly.wo, &st.f8q, &st.f8rs, &mut st.proj, r)?;
            finish(st, &ly.attn_conv)?;
            exec.rmsnorm_batch(&st.x, &ly.post_norm, &mut st.xn, embd, d2.eps, r)?;
            prepare(st, &ly.mlp_conv)?;
            mm(&exec, &ly.gate, &st.f8q, &st.f8rs, &mut st.g, r)?;
            mm(&exec, &ly.up, &st.f8q, &st.f8rs, &mut st.u, r)?;
            exec.swiglu(&mut st.g, &st.u, r * ff)?;
            exec.quantize_e4m3_row(&st.g, &mut st.f8q, &mut st.f8rs, ff, r)?;
            mm(&exec, &ly.down, &st.f8q, &st.f8rs, &mut st.proj, r)?;
            finish(st, &ly.mlp_conv)?;
        }
        exec.rmsnorm_batch(&st.x, &d2.final_norm, &mut st.xn, embd, d2.eps, r)?;
        // the target's head over every block row, then the selector: top-32
        // candidates a row, their codebook rows, the hidden projection, the
        // greedy path walk
        exec.bf16_gemm(head, None, &st.xn, &mut st.logits, r)?;
        let (top_k, rank) = (draft::SEL_TOP_K, draft::SEL_RANK);
        exec.topk_rows(
            &st.logits,
            &st.sel_params,
            &mut st.sel_topk,
            r,
            vocab,
            top_k,
        )?;
        exec.dflash_cand_ids(
            &st.sel_topk,
            &st.d_toks,
            &mut st.sel_ids,
            top_k,
            blk,
            r,
            vocab,
        )?;
        let sel = &d2.sel;
        exec.embed_gather_bf16(
            &sel.pred,
            &st.sel_ids,
            &mut st.sel_pred,
            rank,
            r * top_k + n,
            1.0,
        )?;
        exec.embed_gather_bf16(
            &sel.succ,
            &st.sel_ids,
            &mut st.sel_succ,
            rank,
            r * top_k,
            1.0,
        )?;
        exec.quantize_e4m3_row(&st.xn, &mut st.f8q, &mut st.f8rs, embd, r)?;
        mm(&exec, &sel.hidden, &st.f8q, &st.f8rs, &mut st.sel_hs, r)?;
        exec.dflash_select(
            &st.sel_topk,
            &st.sel_pred,
            &st.sel_succ,
            &st.sel_hs,
            &mut st.d_out,
            1.0,
            0.0,
            rank,
            top_k,
            blk,
            r,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans_and_warmth() {
        assert_eq!(span_note((0, 0), 0, 700), (0, 700), "a cold prompt");
        assert_eq!(span_note((0, 700), 700, 1), (0, 701), "a tick");
        assert_eq!(span_note((0, 701), 690, 3), (0, 693), "parted inside");
        assert_eq!(span_note((0, 0), 512, 40), (512, 552), "past a gap");
        assert!(span_warm((0, 900), 900), "from row 0");
        assert!(!span_warm((0, 900), 901), "must end at the block");
        // past a gap the full layer's reach must be covered again
        assert!(!span_warm((100, 9000), 9000));
        assert!(span_warm((100, 16483), 16483));
        assert!(!span_warm((100, 16482), 16482), "one row short");
    }
}
