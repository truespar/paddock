//! DFlash drafter for nemotron (C2) - the official
//! `nvidia/...-NVFP4-DFlash` checkpoint: a 6-layer qwen3-class dense GQA
//! model (hidden 2688, 32/2 heads at hd 128, QK-norm, yarn rope θ=1e4
//! factor 128, NVFP4 W4A16 MLPs + fc, bf16 attention planes, no lm_head -
//! the TARGET's head runs on top). vLLM's `qwen3_dflash.py` is the
//! reference: committed positions become per-layer K/V from one fused
//! context state per row, `hidden_norm(fc(concat_i aux_i))`, where aux_i
//! are the target's post-block residuals at `target_layer_ids`
//! [1,5,19,29,41,51]; a draft round embeds `[committed, k × mask(990)]`
//! rows and attends NON-CAUSALLY over [context ∥ block] - expressed on the
//! causal kernels by giving every block row the block-end position (the
//! muse splice).
//!
//! The fc plane loads SPLIT into 6 per-aux column bands so the fusion runs
//! as 6 NVFP4 GEMMs over the contiguous aux planes (no interleave copy):
//! fc(concat_i aux_i) = Σ_i fc_band_i(aux_i).

use std::path::Path;

use cudarc::driver::CudaSlice;

use crate::gpu::{GpuError, KvDtype, Nvf4Plane, QuantTensor};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::qwen35::{prefill_mm_pre_any, prefill_quant};
use paddock_kernels::reference::ops::YarnRope;
use paddock_models::ggml_type::GgmlType;
use paddock_models::modelopt::nvfp4_view;
use paddock_models::nemotron::NemotronDflashConfig;
use paddock_models::safetensors::{ShardedSafetensors, StDtype};

use super::*;

/// A drafter NVFP4 plane (MLP or fc band) over a round's or a walk's rows.
/// Past four rows the scalar multi-row class turns issue-bound again on GB10
/// while the tensor-core tile streams these planes ~15-20% faster (b8:
/// gate/up 138 -> 166 GB/s, down 123 -> 141) and owns every prefill-sized
/// batch outright. The tile casts activations to bf16 - the class vLLM runs
/// the drafter in, and a draft's numerics are the verify's to judge.
fn df_mlp(
    exec: &crate::gpu::GpuExecutor,
    w: &Nvf4Plane,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    rows: usize,
) -> Result<(), GpuError> {
    if rows > 4 && w.in_dim.is_multiple_of(128) && exec.has_nvf4_gemm_tc() {
        exec.nvf4_gemm_tc(w, x, y, None, rows)
    } else {
        exec.nvf4_gemv_batch(w, x, y, None, rows)
    }
}

/// Draft block cap: 1 committed row + up to `MAX_DRAFT` masks per round.
pub(crate) const MAX_DRAFT: usize = 15;

/// The block the drafter was trained at: every round embeds `[committed, 7
/// masks]` whatever depth the round verifies. The block's rows attend each
/// other (the non-causal splice), so its width is part of every position's
/// prediction - measured per position at a 4-draft verify (GB10 2026-09-26,
/// greedy code/essay + sampled tool calls + 24K code):
///   block 5: 0.65/0.40/0.26/0.14   block 8: 0.80/0.58/0.42/0.29
///   block 12: 0.73/0.45/0.20/0.08  block 16: 0.73/0.44/0.21/0.08
/// - 8 is the trained width (vLLM's k=7 runs it), and narrower blocks lose
///   at every position. The service's per-slot chain ramp used to resize the
///   block every round (2..8 rows at a pinned k=7).
pub(crate) const BLOCK: usize = 8;

/// Drafts a round verifies, of the block's seven. Each verify row costs this
/// box its own routed experts (the round's MoE streams ~5.6 MB per distinct
/// expert per layer, at the bandwidth roof), so the depth is a trade against
/// the survival curve above, elected on the same legs (tok/s, block 8):
///   depth  code  essay  tools(wall)  24K code
///     3   114.3   93.3     109.4      109.5
///     4   105.4  106.0     112.5      110.9
///     5   107.9   82.6      88.9       92.6
///     7    96.3   79.2      98.4       87.6
///   (no drafter: 83.2 / 83.6 / 79.9 / 78.7)
/// Depth 4 holds the tool-call and long-context legs - the agent's traffic.
pub(crate) const VERIFY_DEPTH: usize = 4;

/// DSpark's verify depth. Its block is causal, so the first N drafts of any
/// block ARE the trained block's (no width to elect - the per-position curve
/// is one curve: 0.87/0.75/0.64/0.52/0.43 at depth 5). Each verify row costs
/// ~2.3 ms of routed experts a round (25.6 / 28.2 / 30.5 / 34.8 ms at depth
/// 3 / 4 / 5 / 7). Elected on GB10 2026-09-26, 131K serve, tok/s (three
/// tool-carrying sampled pairs / 24K code / 60K code, decode):
///   depth 3: 121.4 / 105.6 / 104.6   depth 4: 125.9 / 103.7 / 113.1
///   depth 5: 130.3 / 119.8 / 104.5   (no drafter 80.1 / 78.6 / 72.6)
/// and depth 7 below 5 on the 32K legs (a 120 vs 125 tok/s round model).
/// NVIDIA's Spark recipe verifies 3 on its own round cost.
pub(crate) const DSPARK_DEPTH: usize = 5;

impl DflashDrafter {
    /// Drafts a round verifies with this drafter (its elected depth).
    pub(crate) fn verify_depth(&self) -> usize {
        if self.markov.is_some() {
            DSPARK_DEPTH
        } else {
            VERIFY_DEPTH
        }
    }

    /// Drafts a round verifies per slot at `live` live slots (0 = the round
    /// does not speculate): the elected depth, never more rows than the
    /// verify's snapshot planes hold (so 3 at 7-8 live, 2 at 9-10, 1 to 16).
    /// A round's cost grows with its rows (more tokens touch more experts)
    /// while the per-position survival curve does not move, so the depth
    /// falls as the round widens. Elected on GB10 2026-09-26 with DSpark,
    /// Claude Code-shaped traffic (tool-carrying, the model's own sampling),
    /// aggregate tok/s at 2 / 3 / 4 / 6 / 8 live (two waves each):
    ///   depth 1: 132 / 160 / 173 / 208 / 234   depth 2: 153 / 171 / 202 / 231 / 252
    ///   depth 3: 153 / 190 / 193 / 228 / 271   depth 4: 163 / 190 / 213 / 229 / 274
    ///   depth 5: 133 / 190 / 200 / 225 / 264   (no drafter 120 / 137 / 167 / 191 / 215)
    /// (depth 4 and 5 at 6-8 live ran at what the rows fit). One live slot
    /// keeps the drafter's own elected depth (DSPARK_DEPTH / VERIFY_DEPTH).
    /// `PADDOCK_NEMO_SPEC_DEPTH` pins one depth at every width (dev: the
    /// election's sweep).
    pub(crate) fn depth_for_live(&self, live: usize) -> usize {
        static PIN: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
        let pin = *PIN.get_or_init(|| {
            paddock_models::dev_var!("PADDOCK_NEMO_SPEC_DEPTH")
                .ok()
                .and_then(|v| v.parse().ok())
        });
        if live == 0 {
            return 0;
        }
        let fit = (super::spec::SPEC_ROWS_NEMO / live).saturating_sub(1);
        let elected = if live == 1 {
            self.verify_depth()
        } else {
            self.verify_depth().min(4)
        };
        pin.unwrap_or(elected).min(fit).min(MAX_DRAFT)
    }

    /// Bytes the drafter's stripes add to every pool block: its layers' K
    /// and V over the block's 16 rows, f16 - the pool plan prices a block at
    /// the target's layers plus this, so the stripes are never unplanned.
    pub(crate) fn stripe_bytes(&self, kv_dim: usize) -> usize {
        self.n_layers * 2 * 16 * kv_dim * 2
    }
}

pub(crate) struct DfLayer {
    pub in_norm: CudaSlice<f32>,
    pub post_norm: CudaSlice<f32>,
    pub q_norm: CudaSlice<f32>,
    pub k_norm: CudaSlice<f32>,
    /// the checkpoint's bf16 attention planes as shipped (fused q|k|v + o)
    /// - the class vLLM runs the drafter in, at half the f32 widening's
    ///   bytes on every round and every context append
    pub attn: AttnBf16,
    /// per-head attention sink logits (DSpark): an extra softmax column
    /// with a raw, unscaled logit and a zero value, folded by the combine
    pub sinks: Option<CudaSlice<f32>>,
    pub gate: Nvf4Plane,
    pub up: Nvf4Plane,
    pub down: Nvf4Plane,
}

pub(crate) struct DflashDrafter {
    pub n_layers: usize,
    pub target_layers: Vec<usize>,
    pub inter: usize,
    /// yarn kernel params (neox convention - qwen3)
    pub rope: (f32, f32, f32, f32, f32, f32),
    pub eps: f32,
    /// fc split into one [hidden -> hidden] NVFP4 band per aux stream
    pub fc_bands: Vec<Nvf4Plane>,
    pub hidden_norm: CudaSlice<f32>,
    pub final_norm: CudaSlice<f32>,
    /// the TRAINED mask embedding - the drafter's embed_tokens row at the
    /// config's mask token (byte-probed: the table is identical to the
    /// target's EXCEPT this row; reusing the target's row 990 collapsed
    /// acceptance from ~2.3 to ~1.3 at k=7)
    pub mask_embd: CudaSlice<f32>,
    pub layers: Vec<DfLayer>,
    /// DSpark: the block attends causally at its rows' own positions;
    /// DFlash: every row attends through the block end (the splice)
    pub causal: bool,
    /// sliding-window width of the drafter's attention, 0 = full
    pub window: usize,
    /// DSpark's low-rank Markov fixup head
    pub markov: Option<MarkovHead>,
    pub state: Option<DflashState>,
}

/// DSpark's Markov head: draft j's logits are the base logits (the target's
/// head over mask row j) plus `B(prev) = w2 . w1[prev]`, a rank-r bias that
/// depends only on the token drafted before it (the anchor for the first) -
/// so the drafts are taken left to right, each one feeding the next.
pub(crate) struct MarkovHead {
    /// `markov_w1` bf16 [vocab, rank] - a gather table, row per token
    pub w1: QuantTensor,
    /// `markov_w2` NVFP4 [vocab, rank]
    pub w2: Nvf4Plane,
    pub rank: usize,
}

/// Serving-time drafter state (built at enable_batch when attached).
pub(crate) struct DflashState {
    /// per-drafter-layer context K/V as POOL STRIPES [pool_blocks, 16,
    /// kv_dim] f16, addressed by the target's own block tables: a page the
    /// radix hands to another slot carries the drafter's rows with it (the
    /// qwen35 stripe). The dense per-slot cache this replaced could only be
    /// reused by the slot that wrote it, so a conversation re-admitted into
    /// another slot - after its slot served someone else - drafted nothing
    /// for the rest of the turn.
    pub kv_k: Vec<CudaSlice<u8>>,
    pub kv_v: Vec<CudaSlice<u8>>,
    /// per-slot live feature span [start, end): rows of the slot's current
    /// sequence whose drafter K/V describe it. It grows as walks commit rows;
    /// a gap restarts it at the new row, which a windowed drafter recovers
    /// from once the span spans its window (see `dflash_warm`).
    pub feat: Vec<(u32, u32)>,
    /// per pool block: the pool generation under which the block's sixteen
    /// drafter rows were completed by a live span (0 = never). A page is
    /// drafter-valid for whoever adopts it iff this still equals the pool's
    /// generation - a page freed and re-issued since (another sequence, or a
    /// tier restore that refills only the target's planes) reads stale.
    pub page_gen: Vec<u32>,
    /// aux bands, one [band_rows, embd] plane per target layer (band_rows
    /// == the batch scratch row capacity - walk rows index straight in)
    pub aux: Vec<CudaSlice<f32>>,
    /// fused context rows [band, embd]: raw fc sum + the normed rows
    pub d_ctx: CudaSlice<f32>,
    pub d_ctxn: CudaSlice<f32>,
    pub d_acc: CudaSlice<f32>,
    /// per-append K/V projections [band, kv_dim]
    pub d_kp: CudaSlice<f32>,
    pub d_vp: CudaSlice<f32>,
    // draft-round planes: every live slot's block in ONE pass (see
    // dflash_draft_batch) - backbone rows for `draft_slots` blocks of up to
    // MAX_DRAFT + 1 rows, head rows for what a verify round can take
    pub draft_slots: usize,
    /// per backbone row: its position, attention position and slot
    pub d_pos: CudaSlice<u32>,
    pub d_apos: CudaSlice<u32>,
    pub d_slots: CudaSlice<u32>,
    /// the embedding each backbone row starts from: an index into `d_emb`
    /// ([n committed tokens' rows ∥ the trained mask row])
    pub d_eidx: CudaSlice<u32>,
    pub d_emb: CudaSlice<f32>,
    pub d_commit: CudaSlice<u32>,
    /// the head's rows, step-major (draft j of every slot adjacent - each
    /// Markov step reads one contiguous band): backbone row indices and the
    /// gathered normed rows
    pub d_hidx: CudaSlice<u32>,
    pub d_hx: CudaSlice<f32>,
    pub d_x: CudaSlice<f32>,
    pub d_xn: CudaSlice<f32>,
    pub d_q: CudaSlice<f32>,
    pub d_qn: CudaSlice<f32>,
    pub d_k: CudaSlice<f32>,
    pub d_kn: CudaSlice<f32>,
    pub d_v: CudaSlice<f32>,
    pub d_attn: CudaSlice<f32>,
    pub d_proj: CudaSlice<f32>,
    pub d_g: CudaSlice<f32>,
    pub d_u: CudaSlice<f32>,
    pub d_sinks: CudaSlice<f32>,
    pub d_logits: CudaSlice<f32>,
    pub d_picks: CudaSlice<u32>,
    /// the blocks' attention through the multi-row split partial: one group
    /// list over every slot's rows (<= 8 rows a group) and partial planes
    pub d_groups: CudaSlice<u32>,
    pub d_attn_o: CudaSlice<f32>,
    pub d_attn_ml: CudaSlice<f32>,
    /// DSpark Markov walk, one row per drafting slot: the previous draft,
    /// its w1 row, the bias and the biased logits of the step
    pub d_mk_prev: CudaSlice<u32>,
    pub d_mk_e: CudaSlice<f32>,
    pub d_mk_b: CudaSlice<f32>,
    pub d_mk_l: CudaSlice<f32>,
}

/// A slot's feature span after rows [start, start + n) commit: contiguous
/// growth, a rewrite of the tail from a row inside it (the sequence parted
/// there), or - past a gap - a new span from `start`.
fn span_note((s, e): (usize, usize), start: usize, n: usize) -> (usize, usize) {
    let s = if start >= s && start <= e { s } else { start };
    (s, start + n)
}

/// A span is warm for a block at `pos` when it ends there and reaches back as
/// far as the block's row at `pos` reads: row 0 under full attention, `pos +
/// 1 - window` under a sliding window.
fn span_warm((s, e): (usize, usize), pos: usize, window: usize) -> bool {
    e == pos && (s == 0 || (window > 0 && s + window <= pos + 1))
}

fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
        .collect()
}

impl GpuNemotron {
    /// Attach the official DFlash drafter (safetensors dir). Validates the
    /// geometry against the target and splits the fc plane into per-aux
    /// NVFP4 bands. `state` builds at enable_batch.
    pub fn attach_dflash(&mut self, path: &Path) -> Result<(), GpuModelError> {
        let dir = if path.is_dir() {
            path.to_path_buf()
        } else {
            path.parent()
                .ok_or_else(|| {
                    GpuModelError::Unsupported(format!(
                        "dflash path has no parent directory: {}",
                        path.display()
                    ))
                })?
                .to_path_buf()
        };
        let cfg = NemotronDflashConfig::read(&dir)
            .map_err(|e| GpuModelError::Unsupported(format!("dflash config: {e}")))?;
        let hp = &self.hp;
        if cfg.hidden != hp.hidden
            || cfg.n_heads != hp.n_heads
            || cfg.n_kv_heads != hp.n_kv_heads
            || cfg.head_dim != hp.head_dim
        {
            return Err(GpuModelError::Unsupported(
                "dflash geometry disagrees with the target".into(),
            ));
        }
        let (n_layers, inter, eps, mask_token) = (cfg.n_layers, cfg.inter, cfg.eps, cfg.mask_token);
        let target_layers = cfg.target_layers.clone();
        if target_layers.iter().any(|&l| l >= hp.n_layer)
            || !target_layers.windows(2).all(|w| w[0] < w[1])
        {
            return Err(GpuModelError::Unsupported(format!(
                "dflash target_layer_ids {target_layers:?} do not index the target"
            )));
        }
        // yarn (DFlash, factor 128), or plain rope as yarn's identity case
        // (DSpark): factor 1 with no ramp and no mscale
        let yarn = (cfg.rope_factor - 1.0).abs() > f32::EPSILON;
        let rope = YarnRope::new(
            hp.head_dim,
            cfg.rope_theta,
            1.0 / cfg.rope_factor,
            cfg.rope_orig,
            if yarn { 1.0 } else { 0.0 },
            1.0,
            32.0,
            1.0,
        )
        .kernel_params();

        let st = ShardedSafetensors::open_dir(&dir)
            .map_err(|e| GpuModelError::Unsupported(format!("dflash shards: {e}")))?;
        let exec = self.exec.clone();

        let f32t = |name: &str, want: usize| -> Result<CudaSlice<f32>, GpuModelError> {
            let (t, bytes) = st
                .bytes(name)
                .ok_or_else(|| GpuModelError::Unsupported(format!("{name}: missing")))?;
            if t.dtype != StDtype::Bf16 {
                return Err(GpuModelError::Unsupported(format!(
                    "{name}: expected bf16, got {:?}",
                    t.dtype
                )));
            }
            let v = bf16_to_f32(bytes);
            if v.len() != want {
                return Err(GpuModelError::Unsupported(format!(
                    "{name}: {} elems, expected {want}",
                    v.len()
                )));
            }
            exec.to_device(&v).map_err(GpuModelError::from)
        };
        let bf16raw = |name: &str, want: usize| -> Result<&[u8], GpuModelError> {
            let (t, bytes) = st
                .bytes(name)
                .ok_or_else(|| GpuModelError::Unsupported(format!("{name}: missing")))?;
            if t.dtype != StDtype::Bf16 || bytes.len() != want * 2 {
                return Err(GpuModelError::Unsupported(format!(
                    "{name}: expected {want} bf16 elems, got {:?} x {} bytes",
                    t.dtype,
                    bytes.len()
                )));
            }
            Ok(bytes)
        };
        if !exec.has_bf16_dense() {
            return Err(GpuModelError::Unsupported(
                "dflash: the kernel pack lacks the bf16 dense lane".into(),
            ));
        }
        let nvf4 = |name: &str, out: usize, inn: usize| -> Result<Nvf4Plane, GpuModelError> {
            let v = nvfp4_view(&st, name)
                .map_err(|e| GpuModelError::Unsupported(format!("{name}: {e}")))?;
            if (v.n, v.k) != (out, inn) {
                return Err(GpuModelError::Unsupported(format!(
                    "{name} is [{}, {}], expected [{out}, {inn}]",
                    v.n, v.k
                )));
            }
            exec.nvf4_upload(v.packed, v.scales, v.scale2, v.n, v.k)
                .map_err(GpuModelError::from)
        };

        // fc [hidden, n_aux*hidden] NVFP4 -> per-aux bands (packed bytes and
        // scales split per row on host; scale2 shared)
        let n_aux = target_layers.len();
        let fcv =
            nvfp4_view(&st, "fc").map_err(|e| GpuModelError::Unsupported(format!("fc: {e}")))?;
        if (fcv.n, fcv.k) != (hp.hidden, n_aux * hp.hidden) {
            return Err(GpuModelError::Unsupported(format!(
                "fc is [{}, {}], expected [{}, {}]",
                fcv.n,
                fcv.k,
                hp.hidden,
                n_aux * hp.hidden
            )));
        }
        let (pb, sb) = (hp.hidden / 2, hp.hidden / 16); // packed/scale bytes per band-row
        let (pk, sk) = (fcv.k / 2, fcv.k / 16);
        let mut fc_bands = Vec::with_capacity(n_aux);
        for ai in 0..n_aux {
            let mut p = Vec::with_capacity(fcv.n * pb);
            let mut s = Vec::with_capacity(fcv.n * sb);
            for row in 0..fcv.n {
                p.extend_from_slice(&fcv.packed[row * pk + ai * pb..row * pk + (ai + 1) * pb]);
                s.extend_from_slice(&fcv.scales[row * sk + ai * sb..row * sk + (ai + 1) * sb]);
            }
            fc_bands.push(exec.nvf4_upload(&p, &s, fcv.scale2, fcv.n, hp.hidden)?);
        }

        let q_dim = hp.n_heads * hp.head_dim;
        let kv_dim = hp.n_kv_heads * hp.head_dim;
        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let p = format!("layers.{l}");
            layers.push(DfLayer {
                in_norm: f32t(&format!("{p}.input_layernorm.weight"), hp.hidden)?,
                post_norm: f32t(&format!("{p}.post_attention_layernorm.weight"), hp.hidden)?,
                q_norm: f32t(&format!("{p}.self_attn.q_norm.weight"), hp.head_dim)?,
                k_norm: f32t(&format!("{p}.self_attn.k_norm.weight"), hp.head_dim)?,
                attn: {
                    let a = |n: &str, want: usize| {
                        bf16raw(&format!("{p}.self_attn.{n}_proj.weight"), want)
                    };
                    let mut fused = Vec::with_capacity((q_dim + 2 * kv_dim) * hp.hidden * 2);
                    fused.extend_from_slice(a("q", q_dim * hp.hidden)?);
                    fused.extend_from_slice(a("k", kv_dim * hp.hidden)?);
                    fused.extend_from_slice(a("v", kv_dim * hp.hidden)?);
                    AttnBf16 {
                        wqkv: QuantTensor {
                            bytes: exec.to_device_u8(&fused).map_err(GpuModelError::from)?,
                            ty: GgmlType::Bf16,
                            dims: vec![hp.hidden, q_dim + 2 * kv_dim],
                        },
                        q_dim,
                        kv_dim,
                        wo: QuantTensor {
                            bytes: exec
                                .to_device_u8(a("o", hp.hidden * q_dim)?)
                                .map_err(GpuModelError::from)?,
                            ty: GgmlType::Bf16,
                            dims: vec![q_dim, hp.hidden],
                        },
                    }
                },
                sinks: if cfg.sinks {
                    Some(f32t(
                        &format!("{p}.self_attn.attention_sink_bias"),
                        hp.n_heads,
                    )?)
                } else {
                    None
                },
                gate: nvf4(&format!("{p}.mlp.gate_proj"), inter, hp.hidden)?,
                up: nvf4(&format!("{p}.mlp.up_proj"), inter, hp.hidden)?,
                down: nvf4(&format!("{p}.mlp.down_proj"), hp.hidden, inter)?,
            });
        }
        let markov = match cfg.markov_rank {
            Some(rank) => Some(MarkovHead {
                w1: QuantTensor {
                    bytes: exec
                        .to_device_u8(bf16raw("markov_head.markov_w1.weight", hp.vocab * rank)?)
                        .map_err(GpuModelError::from)?,
                    ty: GgmlType::Bf16,
                    dims: vec![rank, hp.vocab],
                },
                w2: nvf4("markov_head.markov_w2", hp.vocab, rank)?,
                rank,
            }),
            None => None,
        };
        // embed_tokens is untrained and identical to the target's table
        // EXCEPT the mask row (byte-probed) - the rounds embed via the
        // target's arms and then overwrite the mask rows with this vector
        let mask_embd = {
            let (t, bytes) = st
                .bytes("embed_tokens.weight")
                .ok_or_else(|| GpuModelError::Unsupported("embed_tokens.weight: missing".into()))?;
            if t.dtype != StDtype::Bf16 {
                return Err(GpuModelError::Unsupported(format!(
                    "embed_tokens.weight: expected bf16, got {:?}",
                    t.dtype
                )));
            }
            let row = mask_token as usize * hp.hidden * 2;
            let v = bf16_to_f32(&bytes[row..row + hp.hidden * 2]);
            exec.to_device(&v)?
        };
        self.dflash = Some(DflashDrafter {
            n_layers,
            target_layers,
            inter,
            rope,
            eps,
            fc_bands,
            hidden_norm: f32t("hidden_norm.weight", hp.hidden)?,
            final_norm: f32t("norm.weight", hp.hidden)?,
            mask_embd,
            layers,
            causal: cfg.causal,
            window: cfg.window,
            markov,
            state: None,
        });
        tracing::info!(
            layers = n_layers,
            aux = n_aux,
            mask = mask_token,
            causal = cfg.causal,
            window = cfg.window,
            markov_rank = cfg.markov_rank.unwrap_or(0),
            "nemotron {} drafter attached",
            if cfg.markov_rank.is_some() {
                "DSpark"
            } else {
                "DFlash"
            }
        );
        // an explicit sideload wins the drafter seat - drop the in-file
        // nextn block so its weights and walk hooks don't ride for free
        if self.mtp.take().is_some() {
            tracing::info!("nemotron: in-file MTP block released (DFlash attached)");
        }
        Ok(())
    }

    /// Build (or rebuild) the drafter's serving state - called from
    /// enable_batch so the aux taps are live from the first walk.
    pub(crate) fn dflash_ensure_state(&mut self) -> Result<(), GpuModelError> {
        let Some(df) = self.dflash.as_mut() else {
            return Ok(());
        };
        let (n_slots, band) = {
            let bs = self.batch.as_ref().expect("batch enabled");
            (bs.n_slots, bs.cap)
        };
        let hp = &self.hp;
        let kv_dim = hp.n_kv_heads * hp.head_dim;
        let q_dim = hp.n_heads * hp.head_dim;
        let e = &self.exec;
        let n_aux = df.target_layers.len();
        // a verify round takes at most SPEC_ROWS rows, one pending row per
        // live slot at least: no more than half of them can draft
        let draft_slots = n_slots.clamp(1, super::spec::SPEC_ROWS_NEMO / 2);
        let rows = draft_slots * (MAX_DRAFT + 1);
        let head_rows = super::spec::SPEC_ROWS_NEMO;
        let markov = df.markov.as_ref().map(|m| m.rank);
        let mk = |w: usize| if markov.is_some() { draft_slots * w } else { 1 };
        let rows_cap = super::batch::rows_partial_cap(hp.n_kv_heads, e.sm_count(), rows);
        // the plan priced the stripes into every pool block (stripe_bytes)
        let pool_blocks = self.batch.as_ref().expect("batch enabled").pool.capacity() as usize;
        let stripe = pool_blocks * 16 * kv_dim * 2;
        let mut kv_k = Vec::with_capacity(df.n_layers);
        let mut kv_v = Vec::with_capacity(df.n_layers);
        for _ in 0..df.n_layers {
            kv_k.push(e.alloc_u8(stripe)?);
            kv_v.push(e.alloc_u8(stripe)?);
        }
        df.state = Some(DflashState {
            kv_k,
            kv_v,
            feat: vec![(0, 0); n_slots],
            page_gen: vec![0; pool_blocks],
            aux: (0..n_aux)
                .map(|_| e.alloc(band * hp.hidden))
                .collect::<Result<Vec<_>, _>>()?,
            d_ctx: e.alloc(band * hp.hidden)?,
            d_ctxn: e.alloc(band * hp.hidden)?,
            d_acc: e.alloc(band * hp.hidden)?,
            d_kp: e.alloc(band * kv_dim)?,
            d_vp: e.alloc(band * kv_dim)?,
            draft_slots,
            d_pos: e.alloc_u32(rows)?,
            d_apos: e.alloc_u32(rows)?,
            d_slots: e.alloc_u32(rows)?,
            d_eidx: e.alloc_u32(rows)?,
            d_emb: e.alloc((draft_slots + 1) * hp.hidden)?,
            d_commit: e.alloc_u32(draft_slots)?,
            d_hidx: e.alloc_u32(head_rows)?,
            d_hx: e.alloc(head_rows * hp.hidden)?,
            d_x: e.alloc(rows * hp.hidden)?,
            d_xn: e.alloc(rows * hp.hidden)?,
            d_q: e.alloc(rows * q_dim)?,
            d_qn: e.alloc(rows * q_dim)?,
            d_k: e.alloc(rows * kv_dim)?,
            d_kn: e.alloc(rows * kv_dim)?,
            d_v: e.alloc(rows * kv_dim)?,
            d_attn: e.alloc(rows * q_dim)?,
            d_proj: e.alloc(rows * hp.hidden)?,
            d_g: e.alloc(rows * df.inter)?,
            d_u: e.alloc(rows * df.inter)?,
            d_sinks: e.alloc_no_sinks(hp.n_heads)?,
            d_logits: e.alloc(head_rows * hp.vocab)?,
            d_picks: e.alloc_u32(head_rows)?,
            d_groups: e
                .alloc_u32(2 * draft_slots * (MAX_DRAFT + 1).div_ceil(super::batch::ROWS_GROUP))?,
            d_attn_o: e.alloc(hp.n_heads * rows_cap * hp.head_dim)?,
            d_attn_ml: e.alloc(hp.n_heads * rows_cap * 2)?,
            d_mk_prev: e.alloc_u32(draft_slots)?,
            d_mk_e: e.alloc(mk(markov.unwrap_or(1)))?,
            d_mk_b: e.alloc(mk(hp.vocab))?,
            d_mk_l: e.alloc(mk(hp.vocab))?,
        });
        Ok(())
    }

    /// A slot's new sequence resumes at `pos` from the prefix cache (0 =
    /// a cold prompt) over the pages the radix just shared into its table:
    /// its drafter span is the trailing run of adopted pages whose stripe
    /// rows are still the ones a live span completed (`page_gen`) - the
    /// whole resume for any page this server walked, in whichever slot. A
    /// stale page (re-issued since, or refilled by a tier restore without
    /// the drafter's rows) cuts the run there; a windowed drafter only needs
    /// its window of it.
    pub(crate) fn dflash_adopt_slot(&mut self, slot: usize, pos: usize) {
        let Some(bs) = self.batch.as_ref() else {
            return;
        };
        let Some(st) = self.dflash.as_mut().and_then(|d| d.state.as_mut()) else {
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
        // pages [first, full) are live; a table short of the resume point
        // (never, at a page-edge resume) would leave rows unfilled
        let s = if full * 16 < pos { pos } else { first * 16 };
        st.feat[slot] = (s as u32, pos as u32);
    }

    /// Admission: the slot's pages went back to the pool, and with them
    /// whatever span described them.
    pub(crate) fn dflash_reset_slot(&mut self, slot: usize) {
        if let Some(st) = self.dflash.as_mut().and_then(|d| d.state.as_mut())
            && slot < st.feat.len()
        {
            st.feat[slot] = (0, 0);
        }
    }

    /// Coverage-warm: the span ends at `pos` and reaches back as far as the
    /// block's context reads - to row 0 for DFlash's full attention, to the
    /// window for DSpark (its row at `pos` reads keys from `pos + 1 -
    /// window`), so a windowed drafter re-warms a window after any gap.
    pub(crate) fn dflash_warm(&self, slot: usize, pos: usize) -> bool {
        self.dflash.as_ref().is_some_and(|d| {
            d.state.as_ref().is_some_and(|st| {
                let (s, e) = st.feat[slot];
                span_warm((s as usize, e as usize), pos, d.window)
            })
        })
    }

    /// Fuse the tapped aux rows into context states and append their K/V to
    /// every drafter layer's cache. `rows` are the batch walk's rows
    /// (positions/slots still live in the batch scratch's d_pos/d_slots);
    /// `runs` gives contiguous same-slot spans for the coverage bookkeeping.
    pub(crate) fn dflash_append_features(&mut self, r: usize) -> Result<(), GpuModelError> {
        let hp = self.hp.clone();
        let exec = self.exec.clone();
        let kv_dim = hp.n_kv_heads * hp.head_dim;
        let Some(df) = self.dflash.as_mut() else {
            return Ok(());
        };
        let Some(st) = df.state.as_mut() else {
            return Ok(());
        };
        let bs = self.batch.as_ref().expect("batch enabled");
        let sc = &bs.sc;
        let bps = bs.bps;

        // fused context state: Σ_ai fc_band_ai(aux_ai), then hidden_norm
        // a prefill chunk's hundreds of rows are tensor-core work - the
        // scalar multi-row class added ~0.6 s to a 24K prefill's TTFT (GB10)
        for (ai, band) in df.fc_bands.iter().enumerate() {
            if ai == 0 {
                df_mlp(&exec, band, &st.aux[0], &mut st.d_ctx, r)?;
            } else {
                df_mlp(&exec, band, &st.aux[ai], &mut st.d_acc, r)?;
                exec.add(&mut st.d_ctx, &st.d_acc, r * hp.hidden)?;
            }
        }
        exec.rmsnorm_batch(
            &st.d_ctx,
            &df.hidden_norm,
            &mut st.d_ctxn,
            hp.hidden,
            df.eps,
            r,
        )?;

        for l in 0..df.n_layers {
            let ly = &df.layers[l];
            let a = &ly.attn;
            exec.bf16_gemm_rows(&a.wqkv, a.q_dim, kv_dim, &st.d_ctxn, &mut st.d_kp, r)?;
            exec.bf16_gemm_rows(
                &a.wqkv,
                a.q_dim + kv_dim,
                kv_dim,
                &st.d_ctxn,
                &mut st.d_vp,
                r,
            )?;
            exec.rmsnorm_batch(
                &st.d_kp,
                &ly.k_norm,
                &mut st.d_acc,
                hp.head_dim,
                df.eps,
                r * hp.n_kv_heads,
            )?;
            exec.rope_yarn_batch(
                &mut st.d_acc,
                &sc.d_pos,
                hp.n_kv_heads,
                hp.head_dim,
                df.rope,
                r,
            )?;
            exec.kv_append_batch_paged(
                &st.d_acc,
                &mut st.kv_k[l],
                &sc.d_pos,
                Some(&sc.d_slots),
                &bs.d_bt,
                bps,
                kv_dim,
                r,
                KvDtype::Fp16,
            )?;
            exec.kv_append_batch_paged(
                &st.d_vp,
                &mut st.kv_v[l],
                &sc.d_pos,
                Some(&sc.d_slots),
                &bs.d_bt,
                bps,
                kv_dim,
                r,
                KvDtype::Fp16,
            )?;
        }
        Ok(())
    }

    /// Host-side coverage bookkeeping after an append: rows [start, start +
    /// n) of `slot` now carry their features. The span grows contiguously; a
    /// write inside it rewrites the tail from there (the sequence parted at
    /// `start`); a write past a gap restarts it. Every page the span now
    /// covers whole is stamped with its pool generation (`page_gen`).
    pub(crate) fn dflash_note_rows(&mut self, slot: usize, start: usize, n: usize) {
        let Some(bs) = self.batch.as_ref() else {
            return;
        };
        let Some(st) = self.dflash.as_mut().and_then(|d| d.state.as_mut()) else {
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
        // the pages this note completed: whole inside the span, touched by
        // rows [start, e)
        let blocks = bs.tables[slot].blocks();
        let done = (e / 16).min(blocks.len());
        let from = (start / 16).max(s.div_ceil(16)).min(done);
        for &b in &blocks[from..done] {
            st.page_gen[b as usize] = bs.pool.generation(b);
        }
    }

    /// Coverage for a decode tick's rows - one per slot at its position, the
    /// step graph having appended their features on device. Every tick path
    /// notes here: the device-sampled tick and the decode pipe once did not,
    /// so a sampled request went drafter-cold at its first dense tick and
    /// every later round verified the pending token alone.
    pub(crate) fn dflash_note_ticks(&mut self, slots: &[u32], positions: &[u32]) {
        if self.dflash.as_ref().is_some_and(|d| d.state.is_some()) {
            for (&s, &p) in slots.iter().zip(positions) {
                self.dflash_note_rows(s as usize, p as usize, 1);
            }
        }
    }

    /// Draft every live slot's block in ONE drafter pass. `reqs` are (slot,
    /// pending position, committed token); each block is rows [committed,
    /// `kg` masks] at positions pos..=pos+kg - DSpark's rows attend causally
    /// at their own positions, DFlash's all through the block end (the
    /// non-causal splice) - and each slot gets back its first `kr` drafts.
    /// A slot at the context's end, or one whose block rows the pool cannot
    /// back, drafts nothing this round (a guess is not worth preempting a
    /// sequence for).
    ///
    /// One pass, not a pass per slot: the round streams the drafter's
    /// planes, the target's 198 MB head and each Markov step's w2 once for
    /// every slot, where the per-slot loop streamed them once per slot and
    /// synced the host after each (5.9 ms a slot at DSpark depth 5 - 22 ms of
    /// a four-slot round, GB10 2026-09-26).
    pub(crate) fn dflash_draft_batch(
        &mut self,
        reqs: &[(usize, usize, u32)],
        kg: usize,
        kr: usize,
    ) -> Result<Vec<Vec<u32>>, GpuModelError> {
        assert!((1..=MAX_DRAFT).contains(&kg) && (1..=kg).contains(&kr));
        let mut out = vec![Vec::new(); reqs.len()];
        // back every block's rows in the slots' own pages past their
        // committed ends (the verify backs only its depth)
        let mut act: Vec<(usize, usize, usize, u32)> = Vec::with_capacity(reqs.len());
        for (i, &(slot, pos, tok)) in reqs.iter().enumerate() {
            if pos + kg >= self.max_ctx {
                continue;
            }
            match self.ensure_rows(&[slot as u32], &[(pos + kg) as u32]) {
                Ok(()) => act.push((i, slot, pos, tok)),
                Err(GpuModelError::PoolExhausted) => {}
                Err(e) => return Err(e),
            }
        }
        let cap = self
            .dflash
            .as_ref()
            .and_then(|d| d.state.as_ref())
            .map_or(0, |st| {
                st.draft_slots.min(super::spec::SPEC_ROWS_NEMO / kr).max(1)
            });
        for chunk in act.chunks(cap) {
            let picks = self.dflash_draft_pass(chunk, kg, kr)?;
            let n = chunk.len();
            for (ci, &(i, ..)) in chunk.iter().enumerate() {
                out[i] = (0..kr).map(|j| picks[j * n + ci]).collect();
            }
        }
        Ok(out)
    }

    /// The pass behind `dflash_draft_batch` over `act` (request index, slot,
    /// position, committed token), every block backed. Returns the picks
    /// step-major: draft j of slot i at `j * n + i`.
    fn dflash_draft_pass(
        &mut self,
        act: &[(usize, usize, usize, u32)],
        kg: usize,
        kr: usize,
    ) -> Result<Vec<u32>, GpuModelError> {
        let hp = self.hp.clone();
        let exec = self.exec.clone();
        let max_ctx = self.max_ctx;
        let embd = hp.hidden;
        let kv_dim = hp.n_kv_heads * hp.head_dim;
        let scale = 1.0 / (hp.head_dim as f32).sqrt();
        let n = act.len();
        let blk = kg + 1;
        let rows = n * blk;
        let hrows = n * kr;
        let drv = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
        let groups = super::batch::rows_groups((0..n).map(|i| (i * blk, blk)));
        let n_groups = groups.len() / 2;
        let rows_ns = (hp.head_dim == 128
            && hp.n_heads / hp.n_kv_heads.max(1) <= 16
            && exec.has_attn_rows_partial())
        .then(|| super::batch::rows_split(hp.n_kv_heads, n_groups, exec.sm_count()));

        // stage the row streams
        {
            let df = self.dflash.as_mut().expect("dflash");
            let causal = df.causal;
            let st = df.state.as_mut().expect("dflash state");
            let mut positions = Vec::with_capacity(rows);
            let mut apos = Vec::with_capacity(rows);
            let mut slots = Vec::with_capacity(rows);
            let mut eidx = Vec::with_capacity(rows);
            for (i, &(_, slot, pos, _)) in act.iter().enumerate() {
                for r in 0..blk {
                    positions.push((pos + r) as u32);
                    // DFlash's splice: every row's ATTENTION position is the
                    // block end (the block attends as a whole)
                    apos.push(if causal { pos + r } else { pos + kg } as u32);
                    slots.push(slot as u32);
                    // the committed row embeds its token, the rest the
                    // drafter's TRAINED mask row (d_emb row n)
                    eidx.push(if r == 0 { i } else { n } as u32);
                }
            }
            let hidx: Vec<u32> = (1..=kr)
                .flat_map(|j| (0..n).map(move |i| (i * blk + j) as u32))
                .collect();
            let commit: Vec<u32> = act.iter().map(|a| a.3).collect();
            let stm = &self.exec.stream;
            for (host, dev, what) in [
                (&positions, &mut st.d_pos, "pos"),
                (&apos, &mut st.d_apos, "apos"),
                (&slots, &mut st.d_slots, "slots"),
                (&eidx, &mut st.d_eidx, "eidx"),
                (&hidx, &mut st.d_hidx, "hidx"),
                (&commit, &mut st.d_commit, "commit"),
                (&groups, &mut st.d_groups, "groups"),
            ] {
                let mut v = dev
                    .try_slice_mut(0..host.len())
                    .ok_or_else(|| GpuError::Driver(format!("draft {what}")))?;
                stm.memcpy_htod(host, &mut v).map_err(drv)?;
            }
            // DSpark's Markov walk starts from each slot's committed token
            if df.markov.is_some() {
                exec.copy_region(&st.d_commit, 0, &mut st.d_mk_prev, 0, n)?;
            }
        }

        // embed: the committed tokens through the target's table (the
        // drafter's embed is untrained) into d_emb rows 0..n, the trained mask
        // row after them, then one gather lays every backbone row out
        {
            let bs = self.batch.as_ref().expect("batch enabled");
            let (d_bt, bps) = (&bs.d_bt, bs.bps);
            let df = self.dflash.as_mut().expect("dflash");
            let st = df.state.as_mut().expect("dflash state");
            match &self.tok_embd {
                TokEmbd::F32(tab) => {
                    exec.embed_gather_batch(tab, &st.d_commit, &mut st.d_emb, embd, n)?
                }
                TokEmbd::Bf16(tab) => {
                    exec.embed_gather_bf16(tab, &st.d_commit, &mut st.d_emb, embd, n, 1.0)?
                }
                TokEmbd::Q8(tab) => {
                    exec.embed_gather_batch_q8(tab, &st.d_commit, &mut st.d_emb, embd, n)?
                }
            }
            exec.copy_region(&df.mask_embd, 0, &mut st.d_emb, n * embd, embd)?;
            exec.embed_gather_batch(&st.d_emb, &st.d_eidx, &mut st.d_x, embd, rows)?;
            for l in 0..df.n_layers {
                let ly = &df.layers[l];
                exec.rmsnorm_batch(&st.d_x, &ly.in_norm, &mut st.d_xn, embd, df.eps, rows)?;
                super::attn_qkv_batch(
                    &exec,
                    &ly.attn,
                    &st.d_xn,
                    &mut st.d_q,
                    &mut st.d_k,
                    &mut st.d_v,
                    rows,
                )?;
                exec.rmsnorm_batch(
                    &st.d_q,
                    &ly.q_norm,
                    &mut st.d_qn,
                    hp.head_dim,
                    df.eps,
                    rows * hp.n_heads,
                )?;
                exec.rmsnorm_batch(
                    &st.d_k,
                    &ly.k_norm,
                    &mut st.d_kn,
                    hp.head_dim,
                    df.eps,
                    rows * hp.n_kv_heads,
                )?;
                exec.rope_yarn_batch(
                    &mut st.d_qn,
                    &st.d_pos,
                    hp.n_heads,
                    hp.head_dim,
                    df.rope,
                    rows,
                )?;
                exec.rope_yarn_batch(
                    &mut st.d_kn,
                    &st.d_pos,
                    hp.n_kv_heads,
                    hp.head_dim,
                    df.rope,
                    rows,
                )?;
                // block K/V land in the slots' pages at their true positions
                // (overwritten by real context rows on commit; never inside a
                // page the radix shares - those end at or below the
                // committed end)
                exec.kv_append_batch_paged(
                    &st.d_kn,
                    &mut st.kv_k[l],
                    &st.d_pos,
                    Some(&st.d_slots),
                    d_bt,
                    bps,
                    kv_dim,
                    rows,
                    KvDtype::Fp16,
                )?;
                exec.kv_append_batch_paged(
                    &st.d_v,
                    &mut st.kv_v[l],
                    &st.d_pos,
                    Some(&st.d_slots),
                    d_bt,
                    bps,
                    kv_dim,
                    rows,
                    KvDtype::Fp16,
                )?;
                // every block's rows attend through their attention
                // positions, DSpark's inside its window: one group per 8 rows
                // of a slot reads that slot's context once, split across the
                // die - the generic per-(row, head) walk took 6.3 ms a layer
                // at a 24K context (GB10 2026-09-26). DSpark's sinks fold in
                // the combine (raw logit, zero value).
                let sinks = ly.sinks.as_ref().unwrap_or(&st.d_sinks);
                if let Some(ns) = rows_ns {
                    exec.attn_rows_partial(
                        &st.d_qn,
                        &st.kv_k[l],
                        &st.kv_v[l],
                        &mut st.d_attn_o,
                        &mut st.d_attn_ml,
                        &st.d_apos,
                        &st.d_slots,
                        &st.d_groups,
                        n_groups,
                        Some((d_bt, bps)),
                        max_ctx,
                        hp.n_heads,
                        hp.n_kv_heads,
                        hp.head_dim,
                        kv_dim,
                        rows,
                        ns,
                        df.window,
                        scale,
                        KvDtype::Fp16,
                    )?;
                    exec.attn_combine_batch(
                        &st.d_attn_o,
                        &st.d_attn_ml,
                        sinks,
                        &mut st.d_attn,
                        hp.n_heads,
                        hp.head_dim,
                        ns,
                        rows,
                    )?;
                } else {
                    exec.attn_decode_batch_paged(
                        &st.d_qn,
                        &st.kv_k[l],
                        &st.kv_v[l],
                        sinks,
                        &mut st.d_attn,
                        &st.d_apos,
                        Some(&st.d_slots),
                        d_bt,
                        bps,
                        hp.n_heads,
                        hp.n_kv_heads,
                        hp.head_dim,
                        kv_dim,
                        df.window,
                        rows,
                        scale,
                        KvDtype::Fp16,
                    )?;
                }
                exec.bf16_gemm(&ly.attn.wo, None, &st.d_attn, &mut st.d_proj, rows)?;
                exec.add(&mut st.d_x, &st.d_proj, rows * embd)?;
                exec.rmsnorm_batch(&st.d_x, &ly.post_norm, &mut st.d_xn, embd, df.eps, rows)?;
                df_mlp(&exec, &ly.gate, &st.d_xn, &mut st.d_g, rows)?;
                df_mlp(&exec, &ly.up, &st.d_xn, &mut st.d_u, rows)?;
                exec.swiglu(&mut st.d_g, &st.d_u, rows * df.inter)?;
                df_mlp(&exec, &ly.down, &st.d_g, &mut st.d_proj, rows)?;
                exec.add(&mut st.d_x, &st.d_proj, rows * embd)?;
            }
            // the head reads only the drafts a round returns: gather those
            // rows step-major, then the final norm over them
            exec.embed_gather_batch(&st.d_x, &st.d_hidx, &mut st.d_xn, embd, hrows)?;
            exec.rmsnorm_batch(&st.d_xn, &df.final_norm, &mut st.d_hx, embd, df.eps, hrows)?;
        }

        // the TARGET's head over the drafts' rows, once for every slot
        {
            let df = self.dflash.as_mut().expect("dflash");
            let st = df.state.as_mut().expect("dflash state");
            match &self.lm_head {
                // the target's own head election: past one row the
                // tensor-core tile streams the 198 MB plane at ~225 GB/s where
                // the multi-row GEMV managed ~70 (GB10 2026-09-26)
                HeadW::Nvf4(h) => {
                    super::head_nvf4_batch(&exec, h, &st.d_hx, &mut st.d_logits, hrows, false)?
                }
                HeadW::F8(_) => {
                    return Err(GpuModelError::Unsupported(
                        "DFlash drafting is wired for the NVFP4 / GGUF heads".into(),
                    ));
                }
                HeadW::Qw(q) => {
                    let bs = self.batch.as_mut().expect("batch enabled");
                    let s8 = bs.sc.q8.as_mut().expect("q8 batch scratch");
                    prefill_quant(
                        &exec, &mut s8.xq, &mut s8.xs, &mut s8.yq, &st.d_hx, embd, hrows,
                    )?;
                    prefill_mm_pre_any(
                        &exec,
                        q,
                        &s8.xq,
                        &s8.xs,
                        &s8.yq,
                        &mut s8.xsums,
                        &mut s8.ssums,
                        &mut s8.skfix,
                        &mut st.d_logits,
                        hrows,
                    )?;
                }
            }
            // true block diffusion: mask@p+i predicts its own position's
            // token, so drafts are the mask rows' picks. (Measured: with the
            // wrong mask embedding the rows degenerate into next-token
            // predictors - the earlier shifted read was chasing that bug.)
            match df.markov.as_ref() {
                None => exec.argmax_rows(&st.d_logits, &mut st.d_picks, hrows, hp.vocab)?,
                Some(mk) => {
                    // DSpark: left to right, draft j = argmax(base_j + w2 .
                    // w1[draft j-1]) from the anchor - every slot's step j in
                    // one band (the step-major layout), each step's argmax the
                    // next step's gather index, all on device
                    let vocab = hp.vocab;
                    for j in 0..kr {
                        exec.embed_gather_bf16(
                            &mk.w1,
                            &st.d_mk_prev,
                            &mut st.d_mk_e,
                            mk.rank,
                            n,
                            1.0,
                        )?;
                        exec.nvf4_gemv_batch(&mk.w2, &st.d_mk_e, &mut st.d_mk_b, None, n)?;
                        exec.copy_region(
                            &st.d_logits,
                            j * n * vocab,
                            &mut st.d_mk_l,
                            0,
                            n * vocab,
                        )?;
                        exec.add(&mut st.d_mk_l, &st.d_mk_b, n * vocab)?;
                        exec.argmax_rows(&st.d_mk_l, &mut st.d_mk_prev, n, vocab)?;
                        exec.copy_region(&st.d_mk_prev, 0, &mut st.d_picks, j * n, n)?;
                    }
                }
            }
            let view = st
                .d_picks
                .try_slice(0..hrows)
                .ok_or_else(|| GpuError::Driver("picks view".into()))?;
            Ok(self
                .exec
                .stream
                .clone_dtoh(&view)
                .map_err(|e| GpuError::Driver(e.to_string()))?)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans_grow_part_and_restart() {
        assert_eq!(span_note((0, 0), 0, 700), (0, 700), "a cold prompt");
        assert_eq!(span_note((0, 700), 700, 1), (0, 701), "a tick");
        assert_eq!(span_note((0, 701), 690, 3), (0, 693), "parted inside");
        assert_eq!(span_note((0, 0), 512, 40), (512, 552), "past a gap");
        assert_eq!(span_note((512, 552), 552, 5), (512, 557));
        assert_eq!(span_note((512, 557), 100, 5), (100, 105), "below it");
    }

    #[test]
    fn warmth_reads_the_window() {
        // full attention needs the span from row 0
        assert!(span_warm((0, 900), 900, 0));
        assert!(!span_warm((16, 900), 900, 0));
        assert!(!span_warm((0, 900), 901, 0), "must end at the block");
        // a 1024 window: the row at 1123 reads keys from 100
        assert!(span_warm((100, 1123), 1123, 1024));
        assert!(!span_warm((100, 1122), 1122, 1024), "one row short");
        assert!(span_warm((0, 5), 5, 1024));
    }
}
