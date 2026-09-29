//! Gemma 4 (`gemma4`) - dense decoder with 5:1 interleaved sliding-window /
//! global attention. Paddock's third text family (after gpt-oss and qwen35)
//! and the first with two attention geometries in one model:
//!
//! - 50 SWA layers: window 1024, GQA n_head/16 KV heads, head_dim 256,
//!   RoPE base 10k over the full head_dim
//! - 10 global layers (every 6th): GQA n_head/4 KV heads, head_dim 512,
//!   RoPE base 1M with `rope_freqs` factors - the 1e30 entries freeze those
//!   pairs, i.e. partial rotary - and no V projection: V is the K projection
//!   output through a WEIGHTLESS per-head RMS norm (K diverges via its
//!   learned norm + rope), so V still needs its own cache plane
//!
//! Other family quirks (all verified against llama.cpp b10058
//! `src/models/gemma4.cpp`, the same-weights parity oracle):
//! - token embeddings scaled by sqrt(n_embd) on input; LM head is the TIED
//!   embedding matrix; final logits soft-capped (30·tanh(l/30))
//! - attention score scale is 1.0 (not 1/sqrt(d))
//! - Every KV layer's V passes the weightless per-head RMS norm (not just
//!   the V-less global layers)
//! - sandwich norms: attn_norm -> attn -> attn_post_norm -> +residual, same
//!   for FFN; then the whole stream is multiplied by the per-layer
//!   `layer_output_scale` scalar (0.036..0.99 in the 31B - load-bearing)
//! - FFN is parallel GEGLU: gelu_tanh(gate(x)) * up(x)
//!
//! Acceptance bar: greedy-decode token parity with the same-weights
//! llama.cpp oracle on the identical GGUF.
//! This first cut is the correctness milestone: batch-1 decode, prefill via
//! the trait's token-by-token default. Batched prefill / decode / serving
//! lanes come after parity locks.

mod batch;
mod chunked;
pub mod dflash;
pub mod diffusion;
mod forward;
pub(crate) mod load;
///  uniq-routing diagnostic arm - shared with the other MoE families
/// (deepseek_ocr was the first borrower); the accumulator layout and dumper
/// are family-agnostic.
pub(crate) use load::g4_moe_uniq_arm;
mod multimodal;
pub mod muse_vision;
pub(crate) mod planes;
mod prefix;
mod scratch;
mod spec;
pub mod vision;

use cudarc::driver::CudaSlice;
use std::sync::Arc;

pub(crate) use planes::{EmbdTable, kq_rows, kq_stage};

/// A captured decode tick. The model is single-threaded on the engine's
/// thread (same argument as qwen35's SendGraph).
pub(crate) struct SendGraph(pub crate::gpu::CapturedGraph);
// SAFETY: never accessed from two threads at once; see above.
unsafe impl Send for SendGraph {}

use crate::generator::{GenError, Generator, RowSample, SampledStep};
use crate::gpu::{GluAct, GpuExecutor, QuantTensor, RepackedKQ, RepackedQ8};

/// Rung B2 candidate pipe shape (captured by a strip round).
#[derive(Clone, Copy)]
pub(crate) struct SpecPipeCfg {
    pub n: usize,
    pub rr: usize,
    pub k_use: usize,
    pub gkey: (usize, usize, bool, bool, usize),
}

/// slot ids for the armed pipe (reqs order) - kept beside the Copy cfg.
#[derive(Clone, Default)]
pub(crate) struct SpecPipeSlots(pub Vec<u32>);

/// Rung B2 armed pipeline state.
pub(crate) struct SpecPipe {
    pub n: usize,
    pub rr: usize,
    pub k_use: usize,
    pub gkey: (usize, usize, bool, bool, usize),
    pub slot_ids: Vec<u32>,
    pub stride: usize,
    /// strip double-buffer half the next round writes
    pub flip: usize,
    pub events: [Option<cudarc::driver::CudaEvent>; 2],
}

/// Armed async spec round: what the begin half of the split
/// drafter leaves behind for the device token assembly and the deferred
/// drafts fetch. `keep` maps chain rows back to the begin call's pendings
/// order (the fetch contract); `chain_slot` is each chain row's KV slot id
/// (the assembly contract - verify reqs may interleave cold slots the
/// pendings list never saw, so rows are matched by slot, not index).
pub(crate) struct SpecAsyncPlan {
    pub keep: Vec<usize>,
    pub chain_slot: Vec<u32>,
    pub r: usize,
    pub rr: usize,
    pub k_use: usize,
    /// Drafts read back post-verify (the picks dtoh already drained the
    /// stream). Filled by the impl's h re-point - which must replay the
    /// accept rule on real draft values, not the service's placeholders -
    /// and handed to the service's spec_draft_fetch.
    pub fetched: Option<Vec<Vec<u32>>>,
}

/// One weight plane in whatever class the FILE ships it in.
///
/// UD quant files are MIXED - muse-glimmer's UD-Q8_K_XL keeps `attn_k` and
/// `attn_v` (and `token_embd`/`output`) at bf16 next to Q8_0 everything else,
/// because the quantizer judged those planes worth the bytes; the A4B's
/// Q4_K_M holds Q4_K attention beside Q6_K `attn_v` on half the layers and
/// Q5_0 / Q8_0 on the rows a 256-block format cannot encode. The project's
/// rule for that is per-TENSOR dispatch rather than a per-model switch, and
/// the correctness spine is same-weights parity on the identical GGUF, so
/// requantizing a plane into another class at load is out on both counts.
/// The class travels with the plane; `q8()` is how an arm that can only eat
/// Q8 (the int8-mma and mmq rungs, every fp8 converter) asks, and `kq()` is
/// the k-quant lanes' question (`planes.rs` holds the dispatch).
pub(crate) enum Plane {
    Q8(RepackedQ8),
    /// bf16 bytes exactly as the file holds them, `dims` = `[in, out]`.
    Bf16(QuantTensor),
    /// A k-quant (Q4_K / Q5_K / Q6_K / IQ4_XS) or flat 32-weight (Q5_0 /
    /// Q5_1 / Q8_0-as-flat) plane on the repacked W4A8 streams.
    Kq(RepackedKQ),
}

/// Per-layer weights. `wv == None` on the V-less global layers.
pub(crate) struct LayerWeights {
    pub attn_norm: CudaSlice<f32>,
    pub wq: Plane,
    pub wk: Plane,
    pub wv: Option<Plane>,
    pub wo: Plane,
    /// Muse Glimmer's attention OUTPUT gate [n_embd -> n_head*head_dim].
    /// Fed by the post-attn_norm hidden state (not the raw residual), then
    /// `attn_out *= sigmoid(gate)` before o_proj. Presence of the tensor is
    /// the discriminator - gemma4 files have no `attn_gate` and leave this
    /// None, so no flag is needed to tell the two graphs apart.
    pub attn_gate: Option<Plane>,
    pub q_norm: CudaSlice<f32>,
    pub k_norm: CudaSlice<f32>,
    pub attn_post_norm: CudaSlice<f32>,
    pub ffn_norm: CudaSlice<f32>,
    pub ffn_gate: Plane,
    pub ffn_up: Plane,
    pub ffn_down: Plane,
    pub ffn_post_norm: CudaSlice<f32>,
    /// `layer_output_scale` - host scalar, multiplies the residual stream as
    /// the layer's last op. 1.0 when the tensor is absent.
    pub out_scale: f32,
    /// Sliding-window layer? Decides head geometry, rope base and window.
    pub is_swa: bool,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// fp8 (W8A8-e4m3) FFN plane variants for the >1024-row prefill lane
    /// (PADDOCK_G4_F8): the TMA block-scale GEMM runs ~1.43x the q8 pipe at
    /// gate/2048 (411 vs 259 TF at locked clocks). Lossy -
    /// quality-gated like fp8-KV. Q8 originals stay for every other lane;
    /// ~20.8GB extra VRAM which the global-pool sizing absorbs (it takes
    /// what is free after load).
    pub f8_gate: Option<crate::gpu::RepackedMxfp4>,
    pub f8_up: Option<crate::gpu::RepackedMxfp4>,
    pub f8_down: Option<crate::gpu::RepackedMxfp4>,
    /// verify-GEMM dedup: gate|up CONCATENATED along out into
    /// one e4m3 plane [2*n_ff][n_embd] - the r<=64 verify band runs one
    /// GEMM instead of two (launch/tail-wave overhead was ~1/3 of the twin
    /// lane's time over its streaming floor at c8). +238MB VRAM next to the
    /// separate planes (the >64 prefill arms keep those); values identical.
    pub f8_gu: Option<crate::gpu::RepackedMxfp4>,
    /// f8_gu is the INTERLEAVED lin plane (f8w_repack_lin_gui): gate/up pair
    /// p at tile rows (p>>3)*16+(p&7) / +8. Lin GEMMs are layout-blind, but
    /// every geglu over their output must use quantize_e4m3_geglu2i, and the
    /// r>=32 band fuses geglu+quant into the GEMM (f8_gemm_lin_gu).
    pub gu_il: bool,
    /// PC lane (PADDOCK_G4_PC + fp8-native source): f8_gu's bytes
    /// are quantized on a per-ROW pow2 grid whose exponent also fills the
    /// per-32 strip - every existing consumer dequantizes identically, and
    /// the chunk band routes the scale-free kt4a twin with these f32 scales
    /// ([2*n_ff]: gate half then up half).
    pub gu_ws: Option<CudaSlice<f32>>,
    /// PC scales for the fused qkv plane ([q_dim + 2*kv_dim], segment order
    /// q|k|v) and wo ([n_embd]) - same pow2-coexistence contract as gu_ws.
    pub qkv_ws: Option<CudaSlice<f32>>,
    pub wo_ws: Option<CudaSlice<f32>>,
    pub down_ws: Option<CudaSlice<f32>>,
    /// fp4-weight-class QUALITY probe (PADDOCK_G4_FP4_PROBE): mxfp4 twin of
    /// f8_gu, prefill-lane only - perf-irrelevant (the bs kernel is the old
    /// class); exists to answer "does 31B survive 4-bit FFN weights" before
    /// any fp4-TMA kernel investment.
    pub fp4_gu: Option<crate::gpu::RepackedMxfp4>,
    /// Per-ROW-scaled e4m3 planes for PREFILL-shaped rows (r >= 65) -
    /// DUPLICATES next to the q8 originals, which keep serving every decode/
    /// verify rung (exact class there is untouched). Motivation is sm_100:
    /// legacy int8 mma runs ~1.1P TOPS on B200 vs ~7.5P for e4m3
    /// and the per-32 block-scale GEMM
    /// has no hardware fold below sm_120a (a software fold is a regression),
    /// so prefill rides the fold-free rowwise class instead. ~31 GB extra on
    /// gemma4-31B (attn + FFN) - a datacenter-die tradeoff, default-on only
    /// at cc 10.x.
    /// F8A (Front B / F8R phase 2, sm_120): attn projections on the
    /// per-32 block-scale e4m3 class, REPLACE design (q8 planes -> stubs,
    /// VRAM-flat) - the hardware ue8m0 fold makes this the right class on
    /// sm_120a, unlike cc 10 where the rowwise planes below apply.
    /// qkv-concat plane: wq|wk(|wv) fused along out - the
    /// r>1 verify lane runs one GEMM (out 16384 SWA / 18432 global) and the
    /// nra2s epilogue reads the concat rows strided; r==1 and prefill ride
    /// row-offset sub-views (plain out-row-major plane). VRAM-flat: when
    /// this is built the separate f8a_wq/wk/wv are not.
    pub f8a_wqkv: Option<crate::gpu::RepackedMxfp4>,
    pub f8a_wq: Option<crate::gpu::RepackedMxfp4>,
    pub f8a_wk: Option<crate::gpu::RepackedMxfp4>,
    pub f8a_wv: Option<crate::gpu::RepackedMxfp4>,
    pub f8a_wo: Option<crate::gpu::RepackedMxfp4>,
    pub f8_wq: Option<crate::gpu::F8RowPlane>,
    pub f8_wk: Option<crate::gpu::F8RowPlane>,
    pub f8_wv: Option<crate::gpu::F8RowPlane>,
    pub f8_wo: Option<crate::gpu::F8RowPlane>,
    pub f8r_gate: Option<crate::gpu::F8RowPlane>,
    pub f8r_up: Option<crate::gpu::F8RowPlane>,
    pub f8r_down: Option<crate::gpu::F8RowPlane>,
    // v4 decode planes: rowwise e4m3 with the SW128 tile image
    // pre-baked - the r<=64 FFN band streams these via 1D bulk on cc 10
    // fused gate|up tile plane (336 tiles): one decode GEMM feeds geglu;
    // built instead of the split planes (VRAM-flat, the guf lesson)
    pub f8t_gu: Option<crate::gpu::F8TilePlane>,
    pub f8t_gate: Option<crate::gpu::F8TilePlane>,
    pub f8t_up: Option<crate::gpu::F8TilePlane>,
    pub f8t_down: Option<crate::gpu::F8TilePlane>,
    // attn twin of the v4 decode planes (qkv + wo); separately killable
    // because attn-weight e4m3 is the riskier quality move
    // qkv-concat on the tile route (cc-10 twin of f8a_wqkv): [q|k(|v)] tile
    // streams concatenated into one plane (128 tiles SWA / 144 global), fed
    // to the strided nra2s epilogue; built instead of the split wq/wk/wv
    pub f8t_qkv: Option<crate::gpu::F8TilePlane>,
    pub f8t_wq: Option<crate::gpu::F8TilePlane>,
    pub f8t_wk: Option<crate::gpu::F8TilePlane>,
    pub f8t_wv: Option<crate::gpu::F8TilePlane>,
    pub f8t_wo: Option<crate::gpu::F8TilePlane>,
    /// muse-glimmer attention OUTPUT-GATE e4m3 tile plane (cc-10 decode lever,
    /// The o-gate GEMV was the last per-layer projection stuck on
    /// the crippled int8 dp4a path (~52 us/layer vs e4m3's ~18 us - 17% of the
    /// muse decode tick). When built, the r==1 arm rides this instead of
    /// `q8_0_gemv_repacked`. The Q8 `attn_gate` is kept (prefill still rides
    /// it; the reclaim never stubs it). Opt-in PADDOCK_MUSE_OGATE_F8T,
    /// quality-gated (the gate is more quant-sensitive than wo - see the
    /// `attn_gate_apply` doc).
    pub f8t_attn_gate: Option<crate::gpu::F8TilePlane>,
    /// Per-32 (f8w) attn planes for the prefill lane - the tcgen05
    /// block-scale route made per-32 BEAT per-row at every shape (async-SF
    /// v2, 947/864/527 TF vs 932/849/400), so cc-10 defaults here; the
    /// rowwise planes stay as the env fallback. FFN per-32 planes reuse
    /// f8_gate/f8_up/f8_down.
    pub f8w_wq: Option<crate::gpu::RepackedMxfp4>,
    pub f8w_wk: Option<crate::gpu::RepackedMxfp4>,
    pub f8w_wv: Option<crate::gpu::RepackedMxfp4>,
    pub f8w_wo: Option<crate::gpu::RepackedMxfp4>,
    /// 26B-A4B: the routed-expert group of the hybrid FFN (None on dense
    /// models). The shared branch reuses ffn_gate/up/down above.
    pub moe: Option<MoeWeights>,
}

impl LayerWeights {
    /// May this layer take a FUSED attn-norm arm - the ones that write only
    /// the e4m3 planes (`rmsnorm_e4m3*`, `addnorm_e4m3*`) and never
    /// materialize f32 `normed`/`pf_normed`?
    ///
    /// No, if the layer carries a sigmoid output gate: the gate GEMM eats the
    /// f32 attn-norm activations, and those arms would leave it reading a
    /// STALE buffer from the previous layer - silently wrong output, not a
    /// crash. Always true on gemma4 (no gate planes), so nothing there moves.
    ///
    /// Relaxable once the gate rides the fused q|k|v concat plane (see
    /// `forward::attn_gate_apply`): then it consumes the same quantized
    /// activations the QKV GEMM does and needs no f32 form at all.
    pub(crate) fn fused_norm_ok(&self) -> bool {
        self.attn_gate.is_none()
    }

    /// May this layer's attention output plane (`pf_attn`) hold f16 (the a16
    /// stack)? No, if gated: `mul_sigmoid` is f32-in-place, and the f16 arms
    /// exist only because the e4m3 quantizers are pf_attn's sole readers
    /// there. A gated layer adds an f32 reader, so that premise breaks.
    ///
    /// Relaxable by an f16 `mul_sigmoid` twin, or by the same concat-plane
    /// rung, whichever the measurement says is worth it.
    pub(crate) fn f16_attn_ok(&self) -> bool {
        self.attn_gate.is_none()
    }

    /// Are the K/V projections in the Q8 class? Only then may this layer take
    /// the arms that consume a `&RepackedQ8` directly - the int8-mma (`mma_ks`)
    /// and mmq rungs, and the q|k|v concat plane, which need one operand class
    /// across all three segments.
    ///
    /// A muse-glimmer layer answers false (its k/v ship bf16), and lands on the
    /// per-plane `Plane::gemv`/`gemm` dispatch instead. Always true on gemma4.
    ///
    /// Relaxable per arm, not globally: the fp8 converters are the ones worth
    /// teaching bf16 first (a bf16 source reaches e4m3 in one step instead of
    /// the Q8_0 double hop gemma4 pays), which puts the concat plane and the
    /// whole f8a/f8t ladder back in reach for these layers.
    pub(crate) fn kv_q8(&self) -> bool {
        self.wk.q8().is_some() && self.wv.as_ref().is_none_or(|v| v.q8().is_some())
    }
}

/// The routed experts' three planes, one class per layer: the loader keeps
/// the trio on ONE lane family because the gate+up's int8 output feeds the
/// down through the same fq/fs handshake, and a Q8_0 down beside Q4_K
/// gate/up (the A4B Q4_K_M's 14 such layers) rides the k-quant lanes as the
/// flat Q8_0 seat rather than crossing families mid-layer.
pub(crate) enum ExpertPlanes {
    /// flat rows (e*ff_exp + o); `down` is `ffn_down_exps` [ff_exp, n_embd, n_expert]
    Q8 {
        gate: RepackedQ8,
        up: RepackedQ8,
        down: RepackedQ8,
    },
    /// the same three on the repacked k-quant streams; `down` a flat
    /// 32-weight plane at the expert's own width when the file's is
    Kq {
        gate: RepackedKQ,
        up: RepackedKQ,
        down: RepackedKQ,
    },
}

/// Routed-expert weights for the 26B-A4B hybrid FFN. Load-time folds keep
/// the reference math on existing ops:
/// the router's unweighted-rms + 1/sqrt(d) + scale chain collapses into one
/// weighted rmsnorm (gamma pre-folded); the fused gate_up_exps is split
/// exactly into separate gate/up repacks (row copy - Q8 blocks run along
/// the input dim); the per-expert down scale folds into the top-k combine.
pub(crate) struct MoeWeights {
    /// `ffn_gate_inp` [n_embd, n_expert] F32 - the router (f32 GEMV route).
    pub router_w: crate::gpu::DeviceTensor,
    /// pre-folded router norm gamma = ffn_gate_inp.scale / sqrt(n_embd).
    pub router_gamma: CudaSlice<f32>,
    /// the split halves of `ffn_gate_up_exps` and `ffn_down_exps`, in the
    /// class the file ships them
    pub experts: ExpertPlanes,
    /// `ffn_down_exps.scale` f32 [n_expert] - folded into combine weights.
    pub down_scale: CudaSlice<f32>,
    /// pre_ffw_norm_2 / post_ffw_norm_1 / post_ffw_norm_2 - the MoE branch's
    /// pre-norm and the two branch post-norms (shared, routed).
    pub pre_norm2: CudaSlice<f32>,
    pub post_norm1: CudaSlice<f32>,
    pub post_norm2: CudaSlice<f32>,
    /// tcgen05 e4m3 expert planes: the
    /// FUSED [gate|up] stream (rows at e*2ff + r - 1408/expert = 11 exact
    /// 128-tiles) and the K-PADDED down stream (704 -> 768 zero blocks).
    /// None off cc-10 / when the pack lacks the family - the s8-mma sorted
    /// pair keeps serving. Precision-class change; gates arbitrate.
    pub gu_f8: Option<crate::gpu::RepackedMxfp4>,
    pub dn_f8: Option<crate::gpu::RepackedMxfp4>,
    /// Flat-scale e4m3 expert planes: the SPLIT gate/up halves
    /// requantized to e4m3 with one power-of-2 scale per output row, flat
    /// rows (e*ff_exp + o) exactly like `gate_exps`/`up_exps`. Built only
    /// under PADDOCK_MOE_F8ROW - this is a lossy precision class on trial,
    /// and it is a DUPLICATE: the q8 originals keep serving the decode band,
    /// which has no flat-scale kernel yet. That duplication costs **+14.2 GiB
    /// on A4B** (472 MiB of gate+up per layer x 30 layers - measured
    /// 39.95 vs 25.75 GiB of weights), which is fine for an A/B on a 96 GB
    /// card and not shippable: making this the default means REPLACING the q8
    /// gate/up planes, i.e. porting the dec2 decode band first.
    pub gate_f8r: Option<crate::gpu::F8RowPlane>,
    pub up_f8r: Option<crate::gpu::F8RowPlane>,
    /// down half of the same lane, rows (e*n_embd + o), K = ff_exp. Adds a
    /// further +7.1 GiB on A4B (236 MiB x 30 layers) on top of gate+up.
    pub down_f8r: Option<crate::gpu::F8RowPlane>,
}

/// Which architecture the loaded file is. This folder serves the gemma4
/// lineage AND Meta's Muse Glimmer, which shares gemma4's whole skeleton
/// (sliding/full interleave, the four-norm sandwich, QK-norm, final logit
/// softcap) and differs only in the handful of constants below. Splitting it
/// into a parallel family would have meant duplicating the tuned serving lane
/// - WindowRing SWA paging, continuous batching, chunked prefill, captured
///   decode graphs - which is where this engine's speed actually lives.
///
/// This is not a tuning knob: it is picked from `general.architecture` and
/// every field it selects is either in the file or fixed by that arch's
/// reference implementation.
/// # Adding a variant here is a checklist, not a one-liner
///
/// This folder serves several architectures off one graph, and the things that
/// differ between them are ARCHITECTURE CONSTANTS: they appear in no GGUF
/// metadata key, they can only be read out of the reference implementation
/// (`llama.cpp/src/models/<arch>.cpp`), and every one of them is silent when
/// wrong - the model stays fluent for a few tokens and then wanders.
///
/// So every per-arch accessor below matches EXHAUSTIVELY, with no `_` arm. That
/// is deliberate and load-bearing: adding a variant must fail to compile until
/// somebody has opened the reference graph and decided each of these. A
/// catch-all would hand the newcomer gemma4's constants and say nothing.
///
/// The muse-glimmer bring-up is the cautionary tale - it needed
/// EIGHT of them, and every one of the four that got missed produced
/// plausible text, so each cost a long live-debugging hunt:
///
///   `attn_scale`   gemma4 scores UNSCALED (its query scale is folded into
///                  attn_q_norm at conversion); muse passes 1/sqrt(head_dim)
///                  on top of its own q-norm weights
///   `v_norm`       gemma4 RMS-norms V weightlessly; muse does not touch V
///   `rope_neox`    NEOX half-split pairs vs ROPE_TYPE_NORM interleaved
///   `glu_act`      GeGLU vs SwiGLU
///   `post_norm_eps`, `embd_scale`/`embd_rmsnorm`, `logit_scale` +
///   `final_softcap`, and whether the full-attention layers rope at all
///   (muse's are NoPE - carried as freq_scale 0 in `rope_global`)
///
/// And the constants alone are not enough: check which shared CUDA CARRIER
/// bakes each one in. Two of muse's four defects were kernels that never took
/// the argument (`pd_kv_nra_rows` had no pair layout, `pd_gemma_qkv_nra` had no
/// freq_scale), which no amount of Rust-side care would have caught.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Arch {
    Gemma4,
    MuseGlimmer,
    /// DiffusionGemma 26B-A4B (Google, 2026-06): the Gemma 4 26B-A4B body
    /// verbatim - same geometry, norms, MoE, softcap, rope - generating by
    /// block diffusion instead of next-token decode. The file adds a
    /// self-conditioning MLP and `diffusion.canvas_length`; its
    /// `attention.causal = false` describes the CANVAS pass, the prompt is
    /// still encoded causally. Every constant below that gemma4 answers,
    /// this arch answers the same way; what differs is the generation loop
    /// (`diffusion.rs`), not the layer walk.
    DiffusionGemma,
}

impl Arch {
    /// GGUF metadata key prefix - every `<arch>.foo` header key hangs off this.
    pub(crate) fn key(self) -> &'static str {
        match self {
            Arch::Gemma4 => "gemma4",
            Arch::MuseGlimmer => "muse-glimmer",
            Arch::DiffusionGemma => "diffusion-gemma",
        }
    }

    /// Whether this arch generates by block diffusion (a canvas denoised in
    /// place, committed as a block) rather than one token per step.
    pub(crate) fn is_diffusion(self) -> bool {
        matches!(self, Arch::DiffusionGemma)
    }
}

/// Geometry + rope constants derived from the GGUF header at load.
pub(crate) struct Hparams {
    pub arch: Arch,
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_vocab: usize,
    pub eps: f32,
    /// Epsilon for the POST-attention and POST-FFN norms. gemma4 uses the one
    /// rms eps everywhere, so this equals `eps` there. Muse Glimmer's
    /// reference graph hardcodes 1e-8 for both post-norms while keeping 1e-5
    /// for the pre-norms (muse-glimmer.cpp: `const float post_norm_eps = 1e-8f`)
    /// - it is in no metadata key, so it can only come from reading their code.
    pub post_norm_eps: f32,
    pub swa_window: usize,
    /// (theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale) -
    /// plain rope (no yarn), one tuple per geometry.
    ///
    /// NoPE (Muse Glimmer's full-attention layers get no positional encoding
    /// at all) rides this same tuple as `freq_scale = 0`. With ext_factor 0 the
    /// yarn ramp is 0, so the kernels reduce to `angle = freq_scale * theta`;
    /// at freq_scale 0 that is cos 0 = 1, sin 0 = 0 - a BIT-EXACT identity, not
    /// an approximation. Costs a few wasted sin/cos inside an already-running
    /// fused epilogue; skipping the rotation outright on global layers is the
    /// SOTA form and belongs in the perf pass, not the correctness milestone.
    pub rope_swa: (f32, f32, f32, f32, f32, f32),
    pub rope_global: (f32, f32, f32, f32, f32, f32),
    pub final_softcap: f32,
    /// Multiplier applied to the logits before the softcap. 1.0 on gemma4;
    /// Muse Glimmer ships `muse-glimmer.logit_scale` = 1/sqrt(26) = 0.196116.
    pub logit_scale: f32,
    /// Muse Glimmer RMS-normalizes the embeddings with no weight before layer
    /// 0; gemma4 scales them by sqrt(n_embd) instead. Different preambles, so
    /// the file picks one.
    pub embd_rmsnorm: bool,
    /// MoE geometry (26B-A4B: 128/8/704; all zero on dense models).
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub ff_exp: usize,
    /// Block-diffusion canvas width (`diffusion.canvas_length`, 256 on
    /// DiffusionGemma); 0 on the next-token archs.
    pub canvas_len: usize,
}

/// The arch's gated-FFN activation. Free function (not just the `Hparams`
/// method) because the loader has to elect fp8 plane classes long before it
/// builds the Hparams, and both must read the same rule - a loader that
/// elected a GELU-only lane for a SwiGLU file would serve silent nonsense.
pub(crate) fn glu_act_of(arch: Arch) -> GluAct {
    // EXHAUSTIVE on PURPOSE - no `_` arm. See Arch's note: a catch-all here is
    // how the next arch silently inherits gemma4's nonlinearity.
    match arch {
        Arch::Gemma4 | Arch::DiffusionGemma => GluAct::Gelu,
        Arch::MuseGlimmer => GluAct::Silu,
    }
}

impl Hparams {
    /// Scale folded into the embedding rows before layer 0.
    ///
    /// gemma4 multiplies by sqrt(n_embd) (ggml: `inpL = get_rows(...) *
    /// sqrtf(n_embd)`). muse-glimmer does no scale at all - it RMS-normalizes
    /// the rows instead (`build_norm(inpL, nullptr, nullptr, LLM_NORM_RMS,
    /// -1)`, i.e. No weight, at `f_norm_rms_eps`). Two different preambles
    /// with the same shape, so one number plus one flag covers both and no
    /// call site has to know which arch it is serving.
    pub fn embd_scale(&self) -> f32 {
        if self.embd_rmsnorm {
            1.0
        } else {
            (self.n_embd as f32).sqrt()
        }
    }

    /// Which nonlinearity the gated FFN folds into its gate half. gemma4's
    /// reference graph builds `LLM_FFN_GELU + LLM_FFN_PAR`; muse-glimmer's
    /// builds `LLM_FFN_SILU + LLM_FFN_PAR`. Like `post_norm_eps`, this lives
    /// in no metadata key - it can only come from reading the reference
    /// graph, so the arch is the only thing that carries it.
    ///
    /// Every FFN carrier kernel ships both instantiations (one `pd_glu_act`
    /// template, two ABI slots), so this is a plane-class choice made once at
    /// the call site, not a route that can silently degrade.
    pub(crate) fn glu_act(&self) -> GluAct {
        glu_act_of(self.arch)
    }

    /// RoPE pair layout: true = NEOX's half-split `(k, k+half)`, false =
    /// `ROPE_TYPE_NORM`'s interleaved `(2k, 2k+1)`.
    ///
    /// llama.cpp's `llama_model_rope_type` puts `LLM_ARCH_MUSE_GLIMMER` in
    /// the NORM bucket (next to granite and the llama lineage) while
    /// `LLM_ARCH_GEMMA4` is NEOX. Nothing in the GGUF metadata says which -
    /// like `post_norm_eps` and the FFN activation, it is carried by the
    /// architecture and can only come from reading the reference graph.
    ///
    /// Getting this wrong is the nastiest failure class in the family: the
    /// model still reads fluent for a few tokens (content-driven layers are
    /// unaffected) and then degrades, because only the roped layers see
    /// scrambled positions. It never errors.
    pub(crate) fn rope_neox(&self) -> bool {
        // EXHAUSTIVE on PURPOSE - no `_` arm (see Arch).
        match self.arch {
            Arch::Gemma4 | Arch::DiffusionGemma => true,
            Arch::MuseGlimmer => false,
        }
    }

    /// Whether V gets the weightless per-head RMS norm before it enters the
    /// cache. gemma4's reference graph does it (`gemma4.cpp`: `Vcur =
    /// ggml_rms_norm(ctx0, Vcur, hparams.f_norm_rms_eps)`, no weight, right
    /// next to the learned K norm); muse-glimmer's does not - it norms Q and
    /// K only and hands the raw `Vcur` to `build_attn`.
    ///
    /// Third member of the same family as `rope_neox` and `glu_act`: carried
    /// by the architecture, absent from every metadata key, and silent when
    /// wrong. Normalizing V anyway flattens each head's value magnitude to
    /// one, which is exactly the signal attention's weighted sum is supposed
    /// to preserve - the model still emits plausible tokens for a few steps
    /// and then wanders.
    pub(crate) fn v_norm(&self) -> bool {
        // EXHAUSTIVE on PURPOSE - no `_` arm (see Arch).
        match self.arch {
            Arch::Gemma4 | Arch::DiffusionGemma => true,
            Arch::MuseGlimmer => false,
        }
    }

    /// The scale applied to the QK scores, per head geometry.
    ///
    /// gemma4 pins `f_attention_scale = 1.0` (its own comment: "Gemma4 uses
    /// self.scaling = 1.0 (no pre-attn scaling)") because the query scale is
    /// folded into the `attn_q_norm` weights at conversion time - so this
    /// engine passed a literal 1.0 at every attention call site.
    /// muse-glimmer does not do that: `muse-glimmer.cpp` computes
    /// `kq_scale = 1.0f / sqrtf(n_embd_head)` and passes it to `build_attn`
    /// ALONGSIDE its own synthesized q-norm weights (which absorb a different
    /// constant, `qk_scale_factor`). Dropping it leaves the scores sqrt(128)
    /// = 11.3x too large, which drives softmax to a near one-hot pick - the
    /// single loudest of this arch's silent-wrongness modes.
    pub(crate) fn attn_scale(&self, head_dim: usize) -> f32 {
        // EXHAUSTIVE on PURPOSE - no `_` arm (see Arch).
        match self.arch {
            Arch::Gemma4 | Arch::DiffusionGemma => 1.0,
            Arch::MuseGlimmer => 1.0 / (head_dim as f32).sqrt(),
        }
    }
}

/// The mtmd image-splice markers: the tokens that bracket an image's soft
/// tokens in the prompt stream. Per-arch and not derivable from anything in
/// either file - llama.cpp's `mtmd.cpp` sets `img_beg`/`img_end` from the
/// PROJECTOR_TYPE, and the two towers this module serves disagree.
///
/// EXHAUSTIVE on PURPOSE - no `_` arm (see [`Arch`]). Inheriting gemma4's
/// markers would tokenize to `<unk>`-ish garbage around every picture.
pub(crate) fn image_markers(arch: Arch) -> (&'static str, &'static str) {
    match arch {
        // DiffusionGemma's processor is Gemma4Processor - the same markers
        Arch::Gemma4 | Arch::DiffusionGemma => ("<|image>", "<image|>"),
        Arch::MuseGlimmer => ("<|image_start|>", "<|image_end|>"),
    }
}

impl Hparams {
    /// LM-head epilogue, host side: the pre-softcap logit scale, then the
    /// tanh softcap. ORDER MATTERS and is not commutative - the reference
    /// does `scale(logit_scale)` and only then `tanh(x/C)*C`, so scaling
    /// after the cap would squash a different range and change which token
    /// wins near the cap. gemma4 has logit_scale 1.0 and is unaffected.
    ///
    /// The two are deliberately not folded into one op: softcap(s*x) is
    /// `C*tanh(s*x/C)`, which no single choice of C reproduces, because C is
    /// both the divisor and the outer multiplier.
    /// May the FUSED add+post-norm+pre-norm kernels (the `addnorm_e4m3_*`
    /// family) be used?
    ///
    /// Those kernels do the post-attention norm and the following pre-FFN norm
    /// in one pass and take a single epsilon for both. That is sound only when
    /// the two norms actually share one - true for gemma4, false for Muse
    /// Glimmer (post 1e-8, pre 1e-5). Phrased as a property of the epsilons
    /// rather than of the architecture, so a future file that happens to share
    /// them keeps the fast path automatically and one that doesn't cannot
    /// silently take it.
    ///
    /// SOTA target: give the addnorm family a second eps parameter so the
    /// split-epsilon case keeps the fusion. Until then the unfused
    /// `rmsnorm_add_scale` + `rmsnorm` pair is the correct-but-slower route.
    pub(crate) fn fused_two_norm_ok(&self) -> bool {
        self.post_norm_eps == self.eps
    }

    pub(crate) fn logit_epilogue(&self, logits: &mut [f32]) {
        if self.logit_scale != 1.0 {
            for l in logits.iter_mut() {
                *l *= self.logit_scale;
            }
        }
        if self.final_softcap != 0.0 {
            for l in logits.iter_mut() {
                *l = self.final_softcap * (*l / self.final_softcap).tanh();
            }
        }
    }
}

/// Device twin of [`Hparams::logit_epilogue`] - The one implementation every
/// device head must call. Takes the two scalars rather than `&Hparams` so a
/// caller holding a `&mut` borrow of `self.batch_logits` can still reach it.
///
/// It is a free function, and every site calls it, because the split version
/// is what broke Muse Glimmer: `batch.rs`'s three heads each open-coded the
/// `softcap` half and silently dropped the `scale` half, while `forward.rs`
/// did both. gemma4 has `logit_scale` 1.0 so the two agreed there and nothing
/// showed up; muse ships `logit_scale` 1/sqrt(26) with a cap of 20, so the
/// batched decode path fed tanh logits 5.1x too large and saturated it.
/// tanh is MONOTONIC, so the argmax survived - greedy output stayed exactly
/// right and every greedy-parity run passed - but the magnitudes
/// collapsed to +/-20 and anything sampling at temperature drew from a
/// near-uniform distribution. Measured against llama.cpp on the same GGUF:
/// top-20 spread 0.24-0.36 nats where the reference had 7.0-8.4.
pub(super) fn logit_epilogue_dev(
    exec: &GpuExecutor,
    logits: &mut CudaSlice<f32>,
    len: usize,
    logit_scale: f32,
    final_softcap: f32,
) -> Result<(), crate::gpu::GpuError> {
    // Order matters and does not commute - see `logit_epilogue` above.
    if logit_scale != 1.0 {
        exec.scale(logits, logit_scale, len)?;
    }
    if final_softcap != 0.0 {
        exec.softcap(logits, len, final_softcap)?;
    }
    Ok(())
}

/// Per-layer KV cache (f16). Global layers: dense [slots, max_ctx, kv_dim]
/// planes. SWA layers under paging: WindowRing block POOLS
/// [slots * ring, 16, kv_dim] addressed through the shared ring table.
pub(crate) struct LayerKv {
    pub k: CudaSlice<u8>,
    pub v: CudaSlice<u8>,
    /// dim-major twin V pool (vdim[block][kv_dim][16 keys]) for
    /// the v9q VD arm - SWA fp8 layers under PADDOCK_VDIM only; every
    /// legacy reader keeps `v`
    pub vdim: Option<CudaSlice<u8>>,
    pub kv_dim: usize,
    /// per-layer cache element type. fp8-e4m3 is the DEFAULT for pooled
    /// serving: 74.1 GiB of cache where f16 needs 110.6, at near-identical
    /// throughput, and the fp8 kernel arms cover decode, spec verify and
    /// prefill.
    /// PADDOCK_G4_KV16=1 switches back to f16; see alloc_kv in batch.rs.
    pub dtype: crate::gpu::KvDtype,
}

/// SWA WindowRing paging state (gpt-oss G3 scheme, static tables): logical
/// block j of slot s lives at pool block `s*ring + j%ring`. Sound because
/// windowed attention never reads a block older than window+chunk behind the
/// newest append - exactly what `ring` is sized to hold. One table serves all
/// SWA layers (same geometry/mapping; pools differ per layer).
pub(crate) struct SwaPaging {
    pub bt: CudaSlice<u32>,
    /// logical blocks per slot (max_ctx/16) - the table's slot stride
    pub bps: usize,
    /// pool blocks per slot: (PF_ROWS + window)/16 + 1, capped at bps
    pub ring: usize,
}

/// Budget-pool paging for the 10 GLOBAL layers (enable_batch, gpt-oss G4a
/// shape): their KV lives in a shared free-list of 16-token blocks instead
/// of dense [slots, max_ctx] planes (671 MB/slot at ctx 8192 - 90% waste on
/// ~800-token prompts). One logical block table serves every global layer;
/// each layer's k/v buffer is sized `pool` blocks. Tables grow on demand
/// (`ensure_global_rows`) and free on completion; the prefix cache shares
/// blocks zero-copy via refcounts.
pub(crate) struct GlobalPaging {
    pub pool: crate::kv_pool::KvPool,
    pub tables: Vec<crate::kv_pool::BlockTable>,
    /// host mirror of `d_bt` ([slots, bps]) - re-uploaded when tables grow
    pub bt_host: Vec<u32>,
    pub d_bt: CudaSlice<u32>,
    /// logical blocks per slot (max_ctx/16)
    pub bps: usize,
}

/// Decode-step scratch, allocated once at load for the LARGEST geometry.
pub(crate) struct Scratch {
    pub x: CudaSlice<f32>,      // residual stream [n_embd]
    pub normed: CudaSlice<f32>, // pre-attn / pre-ffn norm out [n_embd]
    pub q: CudaSlice<f32>,      // [n_head * 512]
    pub k: CudaSlice<f32>,      // [16 * 256] == [4 * 512] * 2 (max)
    pub v: CudaSlice<f32>,      // same
    pub kn: CudaSlice<f32>,     // normed k
    pub vn: CudaSlice<f32>,     // normed v
    pub qn: CudaSlice<f32>,     // normed+roped q
    pub attn: CudaSlice<f32>,   // attention out [n_head * 512]
    /// muse-glimmer attention output gate, pre-sigmoid [n_head * head_dim].
    /// 1-elem stub on ungated families (gemma4) - nothing reads it there.
    pub agate: CudaSlice<f32>,
    pub proj: CudaSlice<f32>,          // wo / ffn_down out [n_embd]
    pub gate: CudaSlice<f32>,          // [n_ff]
    pub up: CudaSlice<f32>,            // [n_ff]
    pub logits: CudaSlice<f32>,        // [n_vocab]
    pub stream_tmp: CudaSlice<f32>,    // out_scale staging [n_embd]
    pub pos: CudaSlice<u32>,           // [1] current position
    pub ones: CudaSlice<f32>,          // [512] unit weight for the weightless V norm
    pub neg_inf_sinks: CudaSlice<f32>, // [n_head] - this family has no sinks

    // ── batched-prefill lane: PF_ROWS-row chunks through the same graph.
    // All [PF_ROWS, dim] row-major; same names as the decode twins.
    pub pf_x: CudaSlice<f32>,
    pub pf_tmp: CudaSlice<f32>, // out_scale staging + embedding-row assembly
    pub pf_normed: CudaSlice<f32>,
    pub pf_q: CudaSlice<f32>,
    pub pf_qn: CudaSlice<f32>,
    pub pf_k: CudaSlice<f32>,
    pub pf_v: CudaSlice<f32>,
    pub pf_kn: CudaSlice<f32>,
    pub pf_vn: CudaSlice<f32>,
    pub pf_attn: CudaSlice<f32>,
    /// batched twin of `agate` [PF_ROWS, n_head * head_dim]; stub when ungated
    pub pf_agate: CudaSlice<f32>,
    pub pf_proj: CudaSlice<f32>,
    pub pf_gate: CudaSlice<f32>,
    pub pf_up: CudaSlice<f32>,
    pub pf_row: CudaSlice<f32>, // [n_embd] single-token dequant staging
    pub pf_pos: CudaSlice<u32>, // [PF_ROWS] chunk positions
    /// [PF_ROWS] chunk token ids - feeds the one-kernel embed gather (the
    /// per-row dequant_slice+copy_region loop was 2 host launches per row:
    /// up to 4096 cudaLaunchKernel/tick, ~8s of host launch time per c32
    /// window)
    pub pf_toks: CudaSlice<u32>,
    ///  batched-runs prefill attention: device prefix array of run
    /// row offsets ([n_runs+1] u32), armed per coalesced pass via slot 376
    pub pf_runs: CudaSlice<u32>,
    /// [8 x n_vocab] staged head logits for prompts finishing mid-pass -
    /// the per-prompt blocking to_host (384 x ~1.5ms cuStreamSynchronize
    /// stalls MID-pipeline per c32 window) becomes one deferred sync at
    /// tick end. 8 mirrors CHUNK_MAX_INFLIGHT.
    pub pf_fin: CudaSlice<f32>,
    /// [PF_ROWS] attention BOUNDS - == pf_pos except multimodal image rows,
    /// which carry their span end (non-causal within the image)
    pub pf_attn_pos: CudaSlice<u32>,
    pub pf_slots: CudaSlice<u32>, // [PF_ROWS] all-zero (single-sequence slot)
    /// mmq-quantized activation tile (int8 activations - the same numeric
    /// class as llama.cpp's Q8 prefill, which also int8-quantizes)
    pub pf_yq: CudaSlice<u8>,
    /// mmq split-tile fixup plane (sizing convention from qwen35)
    pub pf_skfix: CudaSlice<f32>,
    /// strided int8 quant + per-32-block scales for the mma GEMM class -
    /// the r>1 decode rung ([64 rows x n_ff] ceiling, mma_ks caps at 64)
    pub pf_xq: CudaSlice<i8>,
    pub pf_xs: CudaSlice<f32>,
    /// e4m3 activation plane + per-32 scales for the f8 prefill FFN lane
    /// (empty when PADDOCK_G4_F8 is off)
    pub pf_e4q: CudaSlice<i8>,
    pub pf_e4s: CudaSlice<u8>,
    /// per-ROW f32 activation scales for the rowwise-e4m3 prefill class
    /// (empty unless the f8row planes were built)
    pub pf_e4rs: CudaSlice<f32>,
    /// P54: constant ONES xrs vector for the fin-e4s wo GEMM
    /// (static-scale e4m3 fin store). Separate from pf_e4rs deliberately -
    /// that vector is rewritten by every other row-quantize in the tick.
    pub pf_fae4rs: CudaSlice<f32>,
    /// fused-gu-epilogue landing: e4m3 ff activations + per-32 scales from
    /// pd_f8_gemm_lin_gu. Separate from pf_e4q deliberately - the fused GEMM
    /// reads pf_e4q via TMA for its whole runtime while storing its output,
    /// so writing back into pf_e4q would be a read-under-write race (the
    /// 2-launch chain only got away with it across the launch boundary).
    /// [PF_ROWS x n_ff] / /32; 32-byte stubs off the f8 class.
    pub pf_ffq: CudaSlice<i8>,
    pub pf_ffs: CudaSlice<u8>,

    // ── k-quant plane staging (1-elem stubs on a Q8/bf16 file; see
    // planes.rs). The decode row: int8 + per-32 scales + per-16 sums of the
    // widest input the walk quantizes (n_embd, n_ff, the wo input).
    pub kq_xq: CudaSlice<i8>,
    pub kq_xs: CudaSlice<f32>,
    pub kq_ssums: CudaSlice<f32>,
    /// per-16 sums beside `pf_xq` (the 2..=64-row rungs' mu operand)
    pub pf_ssums: CudaSlice<f32>,
    /// per-32 sums off the mmq tile (`mmq_sums`; the tile GEMM's mu operand)
    pub pf_xsums: CudaSlice<f32>,
    /// one-slot id buffer for the single-row k-quant gather
    pub embd_id: CudaSlice<u32>,
    /// k-quant expert seats: the per-16 sums over `moe_xq` (rows of n_embd)
    /// and then over `moe_fq` (pairs of ff_exp), and the expert-grouped CSR
    /// at the 8/16-row groups the grouped kernels take (the `moe_srow` set
    /// is sized for BM >= 32 and holds fewer blocks)
    pub moe_ssums: CudaSlice<f32>,
    pub kq_srow: CudaSlice<u32>,
    pub kq_sslot: CudaSlice<u32>,
    pub kq_bexp: CudaSlice<u32>,
    /// the tensor-core gate+up's SORTED int8 rows + per-32 scales (the
    /// BM=32 `moe_align` layout, `[mb32 * 32][ff_exp]`): read in place by
    /// the expert-major tensor-core down, else unsorted into `moe_fq` /
    /// `moe_fs` for the grouped down
    pub kq_sfq: CudaSlice<i8>,
    pub kq_sfs: CudaSlice<f32>,
    /// the expert-major tensor-core down's per-expert `[first, end)` over
    /// those sorted rows (`[2 * n_expert]` u32; the pack fills it per launch)
    pub moe_emap: CudaSlice<u32>,

    // ── 26B-A4B hybrid-MoE lane (1-elem stubs on dense models). Sized for
    // PF_ROWS like the pf planes - every lane (prefill, batched decode, b=1
    // step) runs through the same g4_moe_tail helper.
    /// router-norm / pre_ffw_norm_2 output, then reused as the down-GEMM
    /// destination (both uses are sequential) [PF_ROWS, n_embd]
    pub moe_xn: CudaSlice<f32>,
    /// post_ffw_norm_2(moe_down) - the finished MoE branch [PF_ROWS, n_embd]
    pub moe_out: CudaSlice<f32>,
    /// router logits [PF_ROWS, n_expert]
    pub moe_logits: CudaSlice<f32>,
    /// top-k expert ids / renormalized weights [PF_ROWS, n_expert_used]
    pub moe_idx: CudaSlice<u32>,
    pub moe_w: CudaSlice<f32>,
    /// strided int8 quant of the expert input (the pf_xq planes are sized
    /// 192*n_ff - too small for PF_ROWS*n_embd, so the MoE lane owns its own)
    pub moe_xq: CudaSlice<i8>,
    pub moe_xs: CudaSlice<f32>,
    /// e4m3 twin of moe_xq/moe_xs for the flat-scale expert lane.
    /// Same layout (per-32 f32 scale), different encoding - kept separate
    /// rather than aliasing moe_xq so the q8 planes stay valid for whatever
    /// else reads them in the same tick. 101 KB at PF_ROWS.
    pub moe_x8q: CudaSlice<u8>,
    pub moe_x8s: CudaSlice<f32>,
    /// routed gate_up+geglu output [PF_ROWS * n_expert_used, ff_exp] + its quant
    pub moe_fused: CudaSlice<f32>,
    pub moe_fq: CudaSlice<i8>,
    pub moe_fs: CudaSlice<f32>,
    /// all-zeros router bias [n_expert] (the topk kernel folds a bias plane;
    /// gemma4 has none)
    pub moe_zbias: CudaSlice<f32>,
    /// sorted (moe_align) lane: sorted (token,slot) pair maps + per-block
    /// expert ids + per-(token,slot) down partials for the deterministic
    /// slot combine. Sized for the BM=128 pad superset (tc5 blocks).
    pub moe_srow: CudaSlice<u32>,
    pub moe_sslot: CudaSlice<u32>,
    pub moe_bexp: CudaSlice<u32>,
    /// second CSR (bm32) for the prefill dn hybrid - the f8s branch holds
    /// the bm128 layout in the primary set while the v2 down reads this one
    pub moe_srow2: CudaSlice<u32>,
    pub moe_sslot2: CudaSlice<u32>,
    pub moe_bexp2: CudaSlice<u32>,
    /// pair->bm32-row map for the prefill dn hybrid. Its own buffer: the
    /// first cut aliased moe_logits and PDL-armed neighbors make
    /// same-stream adjacency non-serializing - the intermittent
    /// ILLEGAL_ADDRESS class.
    pub moe_pairmap: CudaSlice<f32>,
    pub moe_part: CudaSlice<f32>,
    /// tc5 f8 expert lane: e4m3 quant of the MoE input + the sorted gather
    /// planes + the fused gate_up f32 output + its padded-stride e4m3 quant
    /// (moe_fq8's K-tail [ff_exp, ff_pad) is a STANDING zero region - only
    /// quantize_e4m3_geglu2_pad writes the plane, and only [0, ff_exp)).
    pub moe_e4q: CudaSlice<i8>,
    pub moe_e4s: CudaSlice<u8>,
    pub moe_xg: CudaSlice<u8>,
    pub moe_sg: CudaSlice<u8>,
    pub moe_gu: CudaSlice<f32>,
    pub moe_fq8: CudaSlice<u8>,
    pub moe_fs8: CudaSlice<u8>,
    ///  uniq-routing diagnostic (armed by PADDOCK_MOE_UNIQ=path):
    /// RAW cuMemAlloc accumulator (0 = unarmed), written by the
    /// pd_moe_uniq_hist kernel per (tick, layer) and read by a detached
    /// 5s dumper thread. Two constraints forced this exact shape, both
    /// measured here: (1) serving decode ticks replay captured
    /// (r, k1) graphs, so host-side per-invocation instrumentation counts
    /// once at capture and goes blind - the counter must be a device
    /// kernel baked into the graph; (2) a first-decode init sweep (a
    /// >96MB memset + float spray) overwrote every mempool placement
    /// > tried (mid-scratch, tail, tail + 96MB guard) - serving survives it
    /// > only because scratch is rewritten-before-read, so a persistent
    /// > passive buffer must live outside the pool. Leaked on model drop
    /// > (diagnostic, process lifetime).
    pub moe_uniq_dev: u64,
}

pub struct GpuGemma4 {
    pub(crate) exec: Arc<GpuExecutor>,
    pub(crate) hp: Hparams,
    pub(crate) layers: Vec<LayerWeights>,
    /// The raw embedding table for row gathers - Q8_0 or bf16 as the file
    /// holds it. None on a k-quant embedding: the repacked head below is the
    /// table then (`planes::EmbdTable` is the one seam every gather uses).
    pub(crate) token_embd: Option<QuantTensor>,
    /// The LM head plane. gemma4 TIES its head to the embedding (the file has
    /// no `output.weight`, so this is the repacked token_embd - the raw plane
    /// above stays for row gathers); muse-glimmer ships its own
    /// `output.weight` (6656 x 202048 bf16, its own 1.3 G of parameters), and
    /// tying them there would be silently wrong rather than merely slower.
    /// The loader picks off the TENSOR, never the arch, and the plane carries
    /// its own quant class, so every head call site stays arch-agnostic.
    pub(crate) head: Plane,
    /// All-ones weight vector for the UNWEIGHTED embedding RMSNorm - Some
    /// only when `Hparams::embd_rmsnorm`. The rmsnorm kernels are
    /// `x * inv_rms * w`, so ones make them exactly the reference's
    /// weightless norm without a second kernel that would then have to be
    /// kept in step with this one.
    pub(crate) embd_ones: Option<CudaSlice<f32>>,
    /// f8t TILE plane for the head (r <= 64 decode band): vocab/128 = 2048
    /// row-tiles rides the wmma route like qwen35's out_f8t. None off the
    /// f8t lane, under PADDOCK_NO_F8T_LMHEAD, or when `head` is not Q8.
    pub(crate) head_f8t: Option<crate::gpu::F8TilePlane>,
    /// Per-ROW e4m3 head plane - the same e4m3 weights as `head_f8t` without
    /// the tile repack, because `f8t_gemm`/`f8_repack_tiles` are NULLed off
    /// cc 10.0 exactly (tc5p SASS is sm_100a-only) while `f8row_gemm` runs
    /// from sm_89. Built only for a BF16 head, which is the case that has no
    /// int8 rung at all: the Q8 ladder needs a repacked Q8 source, so
    /// muse-glimmer's head fell to the plain bf16 kernel everywhere.
    /// Consumed by the DFlash draft round today (drafts are proposals the
    /// verify re-derives, so this needs no quality gate); putting the
    /// TARGET's own head on it is a separate, gated change.
    pub(crate) head_f8row: Option<crate::gpu::F8RowPlane>,
    pub(crate) output_norm: CudaSlice<f32>,
    /// rope_freqs factors [head_dim_global/2] for the global layers.
    pub(crate) rope_factors: CudaSlice<f32>,
    pub(crate) kv: Vec<LayerKv>,
    /// Device bytes the weight planes hold (all serving classes: Q8 originals
    /// kept + f8w prefill + f8t decode, after reclaim) - snapshotted by the
    /// loader at the weights/KV boundary for the memory-breakdown API.
    pub(crate) weights_bytes: Option<u64>,
    /// Content identity of the loaded weights and tokenizer, captured at
    /// load - the cache namespace's answer to "are these the same bytes?".
    /// Geometry alone stopped being a sufficient key when the tier gained a
    /// store that survives restarts (see `kv_tier::fingerprint`).
    pub(crate) content_id: ([u8; 32], [u8; 32]),
    pub(crate) scratch: Scratch,
    /// What `scratch` is sized from, kept so `enable_batch` can rebuild the
    /// planes at a narrower chunk without re-deriving them off the weights.
    pub(crate) scratch_dims: scratch::ScratchDims,
    pub(crate) max_ctx: usize,
    /// Rows the prefill scratch was allocated for, and therefore the chunk
    /// size every prefill lane splits at (see `forward::pf_rows`). Read it
    /// rather than `PF_ROWS` at any site that bounds rows - the constant is
    /// the ceiling, this is the allocation.
    ///
    /// Not fixed at load any more: `batch::set_pf_rows` re-allocates the
    /// scratch when the KV plan cannot seat the configured server beside it.
    pub(crate) pf_rows: usize,
    /// The SWA sub-span this server prefills in, and the span the WindowRing
    /// was sized to absorb. One value for both or the ring aliases blocks the
    /// window still needs. `enable_batch` re-elects it down
    /// `forward::SWA_SPAN_LADDER` when the widest rung would leave no room to
    /// batch; an operator pin (`PADDOCK_G4_SWA_SPAN`) freezes it.
    pub(crate) swa_span: usize,
    /// Rows the ring holds past `swa_span` for one picture longer than a
    /// sub-span (`picture_overshoot` when the ring was built): a picture is
    /// never split, so such a picture is a sub-span on its own.
    pub(crate) swa_overshoot: usize,
    pub(crate) pos: usize,
    /// KV slots allocated (1 until `enable_batch`)
    pub(crate) n_slots: usize,
    /// SWA WindowRing paging; None = dense planes (pack lacks paged kernels
    /// or PADDOCK_NO_PAGED_KV pins the escape hatch)
    pub(crate) paging: Option<SwaPaging>,
    /// explicit `--kv-cache-dtype` request (`set_kv_dtype`, the same setter
    /// contract every other family exposes). None = gemma4's own default
    /// logic (fp8-e4m3 when pooled, f16 otherwise - see `alloc_kv`). Kept
    /// ALONGSIDE the historical `PADDOCK_G4_KV16` env switch, which stays
    /// honored for A/B scripts; the env wins the f16 direction when both
    /// are set.
    pub(crate) kv_dtype_pref: Option<crate::gpu::KvDtype>,
    /// DiffusionGemma's block-diffusion lane (self-conditioning MLP, the
    /// transposed embedding, the sampler constants) - Some exactly when the
    /// file's arch is the diffusion one. See `diffusion.rs`.
    pub(crate) diffusion: Option<diffusion::DiffusionLane>,
    /// the canvases the batched diffusion loop holds open, by handle
    /// (`Generator::canvas_open`); None = a released handle, reused first
    pub(crate) canvases: Vec<Option<diffusion::CanvasState>>,
    /// global-layer budget pool (enable_batch + paging mode only; None =
    /// dense global planes - single-stream loads and the
    /// PADDOCK_NO_GLOBAL_POOL escape hatch)
    pub(crate) gpool: Option<GlobalPaging>,
    /// radix prefix cache (paging mode + enable_batch only)
    pub(crate) prefix: Option<prefix::Gemma4Prefix>,
    /// vision tower (attach_vision) + the <|image> / <image|> marker ids
    pub(crate) vision: Option<multimodal::VisionTower>,
    /// Encoded pictures, the only owner of the tower's outputs, bounded in
    /// bytes by the plan (`picture_budget_bytes`): a re-sent image skips
    /// preprocess + tower, and a prefill borrows a picture instead of copying
    /// it. Pictures are encoded as the pass that splices them comes up.
    pub(crate) pictures: crate::gpu_model::picture_store::PictureStore<multimodal::EncodedImage>,
    pub(crate) img_beg_id: Option<u32>,
    pub(crate) img_end_id: Option<u32>,
    /// [n_slots, n_vocab] device logits for the batched head
    pub(crate) batch_logits: Option<CudaSlice<f32>>,
    /// device sampler buffers: ([slots*4] packed params, [slots] out ids)
    pub(crate) samp: Option<(CudaSlice<u32>, CudaSlice<u32>)>,
    /// finisher sampler buffers: ([8*4] packed params, [8] out
    /// ids) - sample_rows over the staged pf_fin prefix replaces the
    /// [finishers, vocab] pageable logits readback on device-plannable
    /// finishers (the 1-8MB dtoh + host pick_next the 128x128 boundary
    /// census attributed 2.2ms/boundary to)
    pub(crate) fin_samp: Option<(CudaSlice<u32>, CudaSlice<u32>)>,
    /// [slots*4] u32 {k, top_p bits, min_p bits, pad} side plane for
    /// mode-5 rows - pd_sample_rows_t draws truncation rows fully on
    /// device (gemma4's election is 1.0/k64/p0.95, so every un-dialled
    /// request is one). Allocated with `samp`; read only for mode-5 rows.
    pub(crate) samp_tpar: Option<CudaSlice<u32>>,
    /// fin twin of `samp_tpar` ([64*4], the pf_fin staging cap)
    pub(crate) fin_tpar: Option<CudaSlice<u32>>,
    /// device-resident decode-tick token ids [n_slots]
    pub(crate) d_tokens: Option<CudaSlice<u32>>,
    /// device-resident decode-tick row->slot map [n_slots] (identity for the
    /// slot-dense tick, explicit for the compacted mixed tick). Dedicated
    /// buffer - pf_slots belongs to the prefill lanes, whose single-stream
    /// convention (all-zeros) a decode upload must not disturb.
    pub(crate) d_slots: Option<CudaSlice<u32>>,
    /// captured decode ticks keyed by (row count, spec chunk k1; k1=1 for
    /// dense) - the spec-verify attention arm bakes k1-deep kernels, so the
    /// same row count with a different chunking is a different graph
    pub(crate) decode_graphs:
        std::collections::HashMap<(usize, usize, bool, bool, usize), SendGraph>,
    /// gkeys sighted once (seen-once lazy capture): first sight
    /// runs the tick eagerly; only a RECURRING gkey pays the ~6.4ms
    /// capture (wave transitions mint one-off row counts)
    pub(crate) graph_seen: std::collections::HashSet<(usize, usize, bool, bool, usize)>,
    /// captured PREFILL passes (PADDOCK_PF_CAP): the c32 mixed tick
    /// is [eager forward_prefill_batch] + [captured decode], and the eager
    /// prefill pass's ~hundreds of per-layer launches are the steady-state
    /// tax the fine-chunk Pareto could never shed. Capture embed+prefill_layers
    /// of a SINGLE-RUN chunk (one prompt) keyed exactly on (r, pmax) - pmax is
    /// baked so a graph never replays over a different KV extent. htod of
    /// pos/slots/toks and the head+dtoh stay outside (block tables by pointer,
    /// gpt-oss G4a shape; ensure_global_rows re-uploads the device table).
    pub(crate) prefill_graphs: std::collections::HashMap<(usize, usize), SendGraph>,
    /// (r, pmax) sighted once - first sight eager, a recurring shape captures
    /// (128x128 c32 mints one shape, r=prompt_len pmax=len-1, that recurs).
    pub(crate) prefill_graph_seen: std::collections::HashSet<(usize, usize)>,
    /// FlashDecoding split-K partials: (o [heads*slots*splits, hd], ml ×2)
    pub(crate) attn_scratch: Option<(CudaSlice<f32>, CudaSlice<f32>)>,
    /// LCO arrival tickets: per-(kvh, chunk) counters the krs
    /// spec-FA merge epilogue bumps; wraps in-kernel, zeroed once at alloc.
    pub(crate) lco_tickets: Option<CudaSlice<u32>>,
    /// chunked-prefill queue: prompts advancing FIFO through mixed ticks
    /// (prefill_begin pushes, forward_mixed_sampled drains under budget)
    pub(crate) chunked: Vec<chunked::ChunkedPrefill>,
    /// MTP drafter (attach_mtp) - the separate gemma4-assistant model
    pub(crate) mtp: Option<spec::MtpDrafter>,
    /// Muse Glimmer's DFlash block-diffusion drafter  - the
    /// other drafter class this family serves. Mutually exclusive with
    /// `mtp` in practice: a checkpoint ships one or the other, and every
    /// spec hook picks DFlash first when it is attached.
    pub(crate) dflash: Option<dflash::DflashDrafter>,
    /// per-slot (pos, pf_normed row) of the last sampled batch tick - the
    /// drafter's h source. Cleared by any pass that overwrites pf_normed
    /// without recording (prefill/unified walks); see spec.rs header.
    pub(crate) spec_rows: Vec<Option<(u32, u32)>>,
    /// Issue-ahead: the mixed round in flight between
    /// forward_mixed_spec_launch and _wait (the service runs the previous
    /// round's deferred finish work between the two).
    pub(crate) mix_inflight: Option<batch::MixInflight>,
    /// begin()'s stashed fallback result (decline / pure-verify picks) -
    /// consumed by the next forward_mixed_spec_plans call.
    pub(crate) mix_fallback: Option<(
        Option<Vec<u32>>,
        Vec<(usize, crate::generator::FinishSample, usize)>,
    )>,
    /// PADDOCK_SPEC_STATS acceptance-economics counters (probe).
    pub(crate) spec_stats: spec::SpecStats,
    /// Armed async spec round: spec_draft_begin enqueued the
    /// chain and this carries what fetch/assembly need. Every armed round
    /// is paired with spec_draft_fetch before the next begin.
    pub(crate) spec_async: Option<SpecAsyncPlan>,
    /// forward_spec_rows assembled this round's verify tokens on device -
    /// batch_upload skips its host token copy (one-shot, self-clearing).
    pub(crate) toks_dev: bool,
    /// rung B1: strip round in flight - forward_batch_sampled_rows
    /// skips the sampled-ids dtoh (the strip carries everything the round
    /// consumes). One-shot, self-clearing.
    pub(crate) ids_skip: bool,
    /// rung B1: the device accept's parsed flat strip (flat u32s, n, stride),
    /// stashed by forward_spec_rows_impl for the strip entry point.
    pub(crate) spec_strip: Option<(Vec<u32>, usize, usize)>,
    /// Rejection sampling (PADDOCK_SPEC_RS): the service's per-slot chain draws for
    /// the round about to be drafted (consumed by spec_draft_begin).
    pub(crate) spec_rs_draws: Option<Vec<crate::generator::SpecRsDraw>>,
    /// the last chain's (chain_slot, rr, k_use) - the q-store /
    /// m.out layout the RS verify resolve maps verify rows onto. Overwritten
    /// per chain; outlives the async plan (which clears at fetch).
    pub(crate) spec_rs_chain: Option<(Vec<u32>, usize, usize)>,
    /// rung B1: the strip entry point armed this call (one-shot).
    pub(crate) want_strip: bool,
    /// One-shot flag around a spec VERIFY tick: the DFlash feature append
    /// must not run off the raw row set there, because rejected rows are
    /// not what the sequence committed. The verify caller clears it and
    /// calls `dflash_spec_commit` with the accepted spans instead. Same
    /// shape as `ids_skip` above.
    pub(crate) dflash_defer: bool,
    /// Per-tick gate on the DFlash tap: false at widths the scheduler will not
    /// speculate at, where fusing feeds a ring nothing will read. Default true
    /// so any path that never sets it keeps the old behaviour.
    pub(crate) dflash_fuse_wanted: bool,
    /// rung B2: candidate pipe shape captured by the last strip
    /// round (all slots kept, uniform k1, r/k1/long = the verify graph key).
    pub(crate) spec_pipe_cfg: Option<SpecPipeCfg>,
    /// slot ids beside the cfg (reqs order).
    pub(crate) spec_pipe_slots: SpecPipeSlots,
    /// rung B2: the ARMED one-ahead pipeline (None = classic rounds).
    pub(crate) spec_pipe: Option<SpecPipe>,
    /// rung B2: side stream for event-gated strip readbacks.
    pub(crate) pipe_copy: Option<GpuExecutor>,
    /// In-flight pipelined-decode state: depth-2 tick pipeline -
    /// tick N+1's inputs advance on device (token = d_out, pf_pos += 1) so
    /// the ~1ms/tick host turnaround (commit + SSE + graph-launch API)
    /// overlaps GPU work instead of gapping it.
    pub(crate) pipe: Option<G4Pipe>,
    /// [2 x vrows] sampled-id ring for the pipe's deferred readbacks.
    pub(crate) d_pipe_out: Option<CudaSlice<u32>>,
    /// verify-tick hint: rows are padded slot-major chunks of this k1 -
    /// batch_step_body routes attention through the k1-deep spec kernel.
    /// Set/cleared by the verify wrapper around its sampled-rows call.
    pub(crate) spec_k1: Option<usize>,
    /// Verify-tick span band (door 3): true when this tick's max position
    /// clears the FIN floor (default 0 - fin whenever the chunk floor
    /// holds). Part of the decode-graph key: host-side kernel election is
    /// FROZEN into a captured graph, so any dynamic route gate must ride
    /// the key - (rows, k1, band) lets both variants capture and replay
    /// correctly as contexts grow.
    pub(crate) spec_long: bool,
    /// host mirror of the live max KV position: feeds the
    /// KV-aware attn_splits_kv clamp; 0 = unknown (election keeps the
    /// full formula). Set by batch_upload / the pipe pos mirror.
    pub(crate) attn_pos_max: usize,
    /// Pos-thresholded LCO election: this pure tick's max
    /// position is below PADDOCK_SPEC_LCO_POS - shallow-walk band, the
    /// krs arms merge splits in-kernel. Rides the decode-graph key.
    pub(crate) spec_shallow: bool,
    /// captured drafter-chain steps keyed by row count (chain steps are
    /// shape-identical; rope positions update via htod between replays)
    pub(crate) mtp_graphs: std::collections::HashMap<usize, SendGraph>,
    /// Forked execution lane for the eager prefill chunk walk, to cut churn:
    /// decode-row attention and the k/v projection
    /// GEMMs are independent of the chunk-row attention / q GEMM on the main
    /// lane (disjoint rows, disjoint KV pages) and overlap into its tail
    /// waves. Event-joined per layer; never used under graph capture (the
    /// chunk walk is eager by construction - host memcpys precede it).
    pub(crate) pf_side: Option<GpuExecutor>,
}

/// GpuError -> GenError, preserving pool exhaustion so the scheduler can
/// preempt instead of failing the batch.
fn gen_err(e: crate::gpu::GpuError) -> GenError {
    match e {
        crate::gpu::GpuError::PoolExhausted => GenError::PoolExhausted,
        e => GenError::Backend(e.to_string()),
    }
}

impl Generator for GpuGemma4 {
    fn release_idle_memory(&mut self) {
        self.exec.trim_mem_pool();
    }

    fn reset(&mut self) {
        self.pos = 0;
    }

    fn device_mem_used(&self) -> Option<u64> {
        self.exec.process_mem_used()
    }

    fn weights_mem_bytes(&self) -> Option<u64> {
        // the diffusion lane's planes (self-cond MLP + the transposed
        // embedding) are weights too - resident for the process, priced with
        // the body so the fit chart and the catalog row see them
        self.weights_bytes
            .map(|b| b + self.diffusion.as_ref().map_or(0, |l| l.bytes))
    }

    fn kv_mem_bytes(&self) -> Option<u64> {
        // every KV plane (serial, SWA ring, global pool) lives in self.kv -
        // alloc_kv sized them, the slice lens are the ground truth
        Some(self.kv.iter().map(|l| (l.k.len() + l.v.len()) as u64).sum())
    }

    fn pool_free_blocks(&self) -> Option<usize> {
        let gp = self.gpool.as_ref()?;
        // free + prefix-reclaimable: the radix is a CACHE (ensure evicts LRU
        // leaves on demand), so admission must not treat retained blocks as
        // spoken for - that watermark-starves the server (c8 52 s TTFT).
        let evictable = self
            .prefix
            .as_ref()
            .map_or(0, |pf| pf.radix.evictable_blocks(&gp.pool));
        Some(gp.pool.free_blocks() + evictable)
    }

    fn tier_pump(&mut self) {
        self.tier_pump_impl();
    }
    fn tier_prefix_loading(&mut self, slot: usize, tokens: &[u32]) -> bool {
        self.tier_consult_impl(slot, tokens)
    }
    fn tier_observe_prefill(&mut self, tokens: u32, wall_us: f64) {
        if let Some(t) = self.prefix.as_mut().and_then(|p| p.tier.as_mut()) {
            t.cost.observe_prefill(tokens, wall_us);
        }
    }
    fn tier_stats(&self) -> Option<crate::kv_tier::TierStats> {
        self.prefix.as_ref()?.tier.as_ref().map(|t| t.tier_stats())
    }
    fn tier_report(&self) -> Option<crate::kv_tier::TierReport> {
        self.prefix
            .as_ref()?
            .tier
            .as_ref()
            .map(crate::kv_tier::PoolTier::report)
    }
    fn release_inactive_slots(&mut self, occupied: &[bool]) {
        self.release_inactive_slots_impl(occupied);
    }

    fn forward(&mut self, token: u32) -> Result<Vec<f32>, GenError> {
        self.step(token).map_err(gen_err)
    }

    fn forward_prefill_stream(&mut self, tokens: &[u32]) -> Result<Vec<f32>, GenError> {
        self.prefill_stream(tokens).map_err(gen_err)
    }

    // ── block diffusion (DiffusionGemma only; see diffusion.rs) ──
    fn canvas_width(&self) -> usize {
        self.hp.canvas_len
    }

    fn canvas_block(
        &mut self,
        base: usize,
        w: usize,
        temperature: Option<f32>,
        seed: u64,
    ) -> Result<(Vec<u32>, u32), GenError> {
        self.canvas_block_impl(base, w, temperature, seed)
            .map_err(gen_err)
    }

    fn canvas_commit(&mut self, base: usize, ids: &[u32]) -> Result<(), GenError> {
        self.canvas_commit_at(0, base, ids).map_err(gen_err)
    }

    fn canvas_read(
        &mut self,
        base: usize,
        canvas: &[u32],
        label_ids: &[u32],
    ) -> Result<crate::generator::CanvasReadOut, GenError> {
        self.canvas_read_impl(base, canvas, label_ids)
            .map_err(gen_err)
    }

    fn canvas_read_steps(
        &mut self,
        base: usize,
        canvas: &[u32],
        label_ids: &[u32],
        steps: u32,
        pinned: &[u32],
        seed: u64,
    ) -> Result<crate::generator::CanvasReadOut, GenError> {
        self.canvas_read_steps_impl(base, canvas, label_ids, steps, pinned, seed)
            .map_err(gen_err)
    }

    fn canvas_max_steps(&self) -> u32 {
        self.diffusion_config().map_or(0, |c| c.max_steps)
    }

    fn canvas_tick_max(&self) -> usize {
        self.pf_rows.checked_div(self.hp.canvas_len).unwrap_or(0)
    }

    fn canvas_open(&mut self, w: usize) -> Result<usize, GenError> {
        self.canvas_open_impl(w).map_err(gen_err)
    }

    fn canvas_set(&mut self, h: usize, ids: &[u32]) -> Result<(), GenError> {
        self.canvas_set_impl(h, ids).map_err(gen_err)
    }

    fn canvas_pin(&mut self, h: usize, positions: &[u32], ids: &[u32]) -> Result<(), GenError> {
        self.canvas_pin_impl(h, positions, ids).map_err(gen_err)
    }

    fn canvas_pins(&self) -> bool {
        self.diffusion.is_some()
    }

    fn canvas_close(&mut self, h: usize) {
        if let Some(c) = self.canvases.get_mut(h) {
            *c = None;
        }
    }

    fn canvas_noise(&self, w: usize, seed: u64, offset: u32) -> Vec<u32> {
        self.random_canvas(w, seed, offset)
    }

    fn canvas_tick(
        &mut self,
        ticks: &[crate::generator::CanvasTickReq],
    ) -> Result<Vec<crate::gpu::CanvasStatus>, GenError> {
        self.canvas_tick_impl(ticks).map_err(gen_err)
    }

    fn canvas_result(
        &self,
        h: usize,
        label_ids: &[u32],
    ) -> Result<crate::generator::CanvasReadOut, GenError> {
        self.canvas_result_impl(h, label_ids).map_err(gen_err)
    }

    fn canvas_commit_slot(
        &mut self,
        slot: usize,
        base: usize,
        ids: &[u32],
    ) -> Result<(), GenError> {
        self.canvas_commit_at(slot, base, ids).map_err(gen_err)
    }

    fn enable_batch(&mut self, max_batch: usize) -> Result<usize, GenError> {
        self.enable_batch_impl(max_batch)
            .map_err(|e| GenError::Backend(e.to_string()))
    }

    fn forward_batch(&mut self, tokens: &[u32], positions: &[u32]) -> Result<Vec<f32>, GenError> {
        self.forward_batch_host(tokens, positions)
            .map_err(|e| GenError::Backend(e.to_string()))
    }

    fn supports_device_sampling(&self) -> bool {
        true
    }

    fn forward_batch_sampled(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        plans: &[RowSample],
    ) -> Result<SampledStep, GenError> {
        self.forward_batch_sampled_impl(tokens, positions, plans)
            .map_err(gen_err)
    }

    fn forward_prefill_batch(
        &mut self,
        items: &[(usize, Vec<u32>)],
    ) -> Result<Vec<Vec<f32>>, GenError> {
        self.forward_prefill_batch_impl(items).map_err(gen_err)
    }

    // ── serving spec rounds: ragged verify rides the standard sampled tick
    // (positions are service-owned - no rollback state); drafts come from
    // the attached gemma4-assistant MTP drafter when its h is fresh, else
    // the service's n-gram fallback verifies just as well.
    fn forward_spec_batch(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
    ) -> Result<Option<Vec<u32>>, GenError> {
        let plans: Vec<RowSample> = reqs
            .iter()
            .flat_map(|(_, _, c)| {
                c.iter()
                    .map(|_| RowSample::Device(crate::sampler::DevicePlan::Greedy))
            })
            .collect();
        self.forward_spec_rows_impl(reqs, &plans).map_err(gen_err)
    }

    fn forward_spec_batch_plans(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
        plans: &[crate::sampler::DevicePlan],
    ) -> Result<Option<Vec<u32>>, GenError> {
        let rows: Vec<RowSample> = plans.iter().map(|p| RowSample::Device(*p)).collect();
        self.forward_spec_rows_impl(reqs, &rows).map_err(gen_err)
    }

    fn spec_draft_batch(
        &mut self,
        pendings: &[(usize, u32)],
        k: usize,
    ) -> Result<Option<Vec<Vec<u32>>>, GenError> {
        // DFlash first when it is attached: one forward drafts the whole
        // block for every warm slot, where the MTP drafter chains k times.
        // A checkpoint carries one drafter or the other, never both.
        if self.dflash.is_some() {
            return self.dflash_draft_batch(pendings, k).map_err(gen_err);
        }
        self.spec_draft_batch_impl(pendings, k).map_err(gen_err)
    }

    fn spec_draft_begin(
        &mut self,
        pendings: &[(usize, u32)],
        k: usize,
    ) -> Result<Option<(usize, Vec<bool>)>, GenError> {
        // the async round needs the device token assembly to exist
        if !self.exec.has_spec_toks() {
            return Ok(None);
        }
        // DFlash has no chain to overlap - its whole round is one forward
        // and one readback, so it declines the split-phase path and the
        // service falls through to the synchronous spec_draft_batch.
        if self.dflash.is_some() {
            return Ok(None);
        }
        self.spec_draft_begin_impl(pendings, k).map_err(gen_err)
    }

    fn spec_draft_fetch(&mut self) -> Result<Option<Vec<Vec<u32>>>, GenError> {
        self.spec_draft_fetch_impl().map_err(gen_err)
    }

    fn supports_spec_rs(&self) -> bool {
        // armed at drafter load (PADDOCK_SPEC_RS + both kernels +
        // u32_addk); rs buffers existing is the arm
        self.mtp.as_ref().is_some_and(|m| m.rs.is_some())
    }

    fn spec_rs_stash(&mut self, draws: Vec<crate::generator::SpecRsDraw>) {
        self.spec_rs_draws = Some(draws);
    }

    fn spec_pipe_arm(&mut self) -> bool {
        // OPT-IN: the pipe measures neutral everywhere we have looked (any
        // apparent win sat inside the run-to-run noise band), so it stays
        // opt-in infrastructure rather than a default. Enable:
        // PADDOCK_SPEC_PIPE=1.
        // The pipe replays chain graphs without the RS per-round state
        // (uplane/step), so RS rounds decline it.
        paddock_models::dev_var_os!("PADDOCK_SPEC_PIPE").is_some()
            && !self.supports_spec_rs()
            && self.spec_pipe_arm_impl()
    }

    fn spec_pipe_round(&mut self, par: &[u32]) -> Result<(), GenError> {
        self.spec_pipe_round_impl(par).map_err(gen_err)
    }

    fn spec_pipe_strip(
        &mut self,
        half: usize,
    ) -> Result<Vec<crate::generator::SpecAccepted>, GenError> {
        let flat = self.spec_pipe_strip_impl(half).map_err(gen_err)?;
        let (n, stride) = {
            let cfg = self
                .spec_pipe
                .as_ref()
                .ok_or_else(|| GenError::Backend("pipe not armed".into()))?;
            (cfg.n, cfg.stride)
        };
        let slot_ids = self.spec_pipe.as_ref().expect("armed").slot_ids.clone();
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let o = &flat[i * stride..(i + 1) * stride];
            let acc = o[0] as usize;
            // heal the h map as we go - the next classic round's warmth
            // check reads it (device h is already gathered by the pipe)
            if let Some(&sid) = slot_ids.get(i)
                && let Some(e) = self.spec_rows.get_mut(sid as usize)
            {
                *e = Some((o[1], o[2]));
            }
            out.push(crate::generator::SpecAccepted {
                accepted: acc,
                pending: o[3],
                tokens: o[4..4 + acc.min(stride - 4)].to_vec(),
            });
        }
        Ok(out)
    }

    fn spec_pipe_ensure(&mut self, slots: &[u32], positions: &[u32]) -> Result<(), GenError> {
        self.ensure_global_rows(slots, positions).map_err(gen_err)
    }

    fn spec_pipe_drain(&mut self) -> Result<(), GenError> {
        self.spec_pipe_drain_impl().map_err(gen_err)
    }

    fn supports_spec_strip(&self) -> bool {
        // OPT-IN, and also measurement-neutral: the host tail it deletes
        // cancels against the per-round accept+readback it adds.
        // Enable: PADDOCK_SPEC_STRIP=1.
        self.exec.has_spec_accept() && paddock_models::dev_var_os!("PADDOCK_SPEC_STRIP").is_some()
    }

    fn forward_spec_batch_strip(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
        plans: &[crate::sampler::DevicePlan],
    ) -> Result<Option<Vec<crate::generator::SpecAccepted>>, GenError> {
        let rows: Vec<RowSample> = plans.iter().map(|p| RowSample::Device(*p)).collect();
        self.want_strip = true;
        let r = self.forward_spec_rows_impl(reqs, &rows);
        self.want_strip = false;
        match r {
            Ok(Some(_)) => {
                let Some((flat, n, stride)) = self.spec_strip.take() else {
                    // the round ran but didn't take the strip path (e.g. no
                    // armed plan) - treat as declined so the caller falls
                    // back; nothing armed remains
                    return Ok(None);
                };
                let mut out = Vec::with_capacity(n);
                for i in 0..n {
                    let o = &flat[i * stride..(i + 1) * stride];
                    let acc = o[0] as usize;
                    out.push(crate::generator::SpecAccepted {
                        accepted: acc,
                        pending: o[3],
                        tokens: o[4..4 + acc.min(stride - 4)].to_vec(),
                    });
                }
                Ok(Some(out))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(gen_err(e)),
        }
    }

    fn spec_ensure_warm(
        &mut self,
        slot: usize,
        committed: &[u32],
        want_pos: u32,
    ) -> Result<bool, GenError> {
        if self.dflash.is_some() {
            // the drafter's own KV, not an h-row freshness map: warm means
            // the feature ring covers this slot's window up to want_pos
            return Ok(self.dflash_ensure_warm(slot, want_pos));
        }
        Ok(self.spec_ensure_warm_impl(slot, committed, want_pos))
    }

    fn spec_fuse_hint(&mut self, on: bool) {
        self.dflash_fuse_wanted = on;
    }

    fn spec_draft_per_slot_warm(&self) -> bool {
        // dflash_draft_batch already filters on dflash_warm per slot and hands
        // back an empty list for the cold ones; MTP's chain does not.
        self.dflash.is_some()
    }

    fn spec_draft_kv_space(&self) -> bool {
        // the Q-only drafter attends the MAIN model's KV read-only with true
        // KV positions and embeds only sampled text tokens - image rows in
        // the shared cache are just more rows to attend; nothing to replay.
        // DFlash owns a separate ring, and image rows reach it the same way
        // text rows do (they ride the walk, so their taps fire), so it needs
        // no image handling of its own either.
        self.mtp.is_some() || self.dflash.is_some()
    }

    fn forward_prefill(&mut self, slot: usize, tokens: &[u32]) -> Result<Vec<f32>, GenError> {
        self.forward_prefill_impl(slot, tokens).map_err(gen_err)
    }

    // Chunked prefill (mixed ticks): default-on - gemma4 is dense, so the
    // mixed tick's extra weight walk is cheap (the qwen35 MoE re-read
    // economics that kept it opt-in there don't apply). The scheduler's
    // PADDOCK_NO_CHUNKED_PREFILL kill pins the classic blocking pass for A/B.
    fn supports_chunked_prefill(&self) -> bool {
        self.d_tokens.is_some()
    }

    fn prefill_queue(&self) -> Vec<(usize, usize, usize)> {
        self.prefill_queue_impl()
    }

    fn prefill_abort(&mut self, slot: usize) -> bool {
        self.prefill_abort_impl(slot)
    }

    fn prefill_prepare(&mut self, budget: usize) -> Result<(), GenError> {
        self.prefill_prepare_impl(budget).map_err(gen_err)
    }

    // a prompt longer than the budget spans this many rows a tick
    // (chunked.rs); whole prompts under it are the shallow-or-short ticks
    fn prefill_tick_cap(&self, _decode_rows: usize) -> usize {
        batch::mixed_tick_rows()
    }

    fn prefill_begin(&mut self, slot: usize, tokens: Vec<u32>) -> Result<(), GenError> {
        self.prefill_begin_impl(slot, tokens)
            .map_err(|e| GenError::Backend(e.to_string()))
    }

    fn forward_mixed_sampled(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
        plans: &[RowSample],
        _fin_plans: &[(usize, RowSample)],
    ) -> Result<
        (
            SampledStep,
            Vec<(usize, crate::generator::FinishSample, usize)>,
        ),
        GenError,
    > {
        self.forward_mixed_sampled_impl(decodes, budget, plans)
            .map_err(gen_err)
            .map(|(step, fin)| {
                let fin = fin
                    .into_iter()
                    .map(|(k, l, r)| (k, crate::generator::FinishSample::Logits(l), r))
                    .collect();
                (step, fin)
            })
    }

    fn supports_decode_pipe(&self) -> bool {
        self.supports_decode_pipe_impl()
    }

    fn supports_device_trunc(&self) -> bool {
        self.device_trunc_supported()
    }

    /// Without a drafter the scheduler's host-2a greedy round degenerates to
    /// UNCAPTURED plain decode - ~900 host launches per tick where the pipe
    /// replays one graph. Cap 0 declines the round so pure-greedy decode
    /// falls through to the graph-piped path; with a drafter attached the
    /// round pays as designed and keeps the trait default.
    ///
    /// Either drafter counts. Keying this on `mtp` alone makes
    /// DFlash a dead lane: cap 0 also clamps the scheduler's own
    /// `dev_spec_live_max` to 0, so the mixed tick and the device-sampled
    /// round never engage and the pure-greedy round's `live.len() <= cap`
    /// gate is false at every width. The only lane left is the SAMPLED
    /// fallback (it consults neither), which drafts exactly once and then
    /// cools down 256 ticks - the "drafts fine, never verifies" symptom.
    fn spec_live_cap(&self) -> usize {
        if self.dflash_attached() {
            // DFlash declines a round wider than this anyway (spec.rs's
            // `reqs.len() > spec_live_max()`), so reporting it here is what
            // makes the SCHEDULER skip the whole spec path at those widths
            // instead of walking it to a decline - and, with it, the per-tick
            // fusion that feeds a ring no round will read.
            spec::spec_live_max()
        } else if self.mtp.is_some() {
            usize::MAX
        } else {
            0
        }
    }

    fn decode_pipe_begin(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        plans: &[RowSample],
    ) -> Result<(), GenError> {
        self.decode_pipe_begin_impl(tokens, positions, plans)
            .map_err(gen_err)
    }

    fn decode_pipe_next(&mut self, plans: &[RowSample]) -> Result<Vec<u32>, GenError> {
        self.decode_pipe_next_impl(plans).map_err(gen_err)
    }

    fn decode_pipe_drain(&mut self) -> Result<Vec<u32>, GenError> {
        self.decode_pipe_drain_impl().map_err(gen_err)
    }

    fn forward_mixed_spec_plans(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
        budget: usize,
        plans: &[crate::sampler::DevicePlan],
        fin_plans: &[(usize, RowSample)],
    ) -> Result<
        (
            Option<Vec<u32>>,
            Vec<(usize, crate::generator::FinishSample, usize)>,
        ),
        GenError,
    > {
        // begin() may have produced a fallback result already (decline /
        // pure verify) - hand it over instead of re-running the pass
        if let Some(r) = self.mix_fallback.take() {
            return Ok(r);
        }
        self.forward_mixed_spec_plans_impl(reqs, budget, plans, fin_plans)
            .map_err(gen_err)
    }

    fn forward_mixed_spec_begin(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
        budget: usize,
        plans: &[crate::sampler::DevicePlan],
        fin_plans: &[(usize, RowSample)],
    ) -> Result<bool, GenError> {
        match self
            .forward_mixed_spec_launch_impl(reqs, budget, plans, fin_plans)
            .map_err(gen_err)?
        {
            batch::MixLaunch::Launched => Ok(true),
            batch::MixLaunch::Fallback(picks, finished) => {
                self.mix_fallback = Some((picks, finished));
                Ok(false)
            }
        }
    }
    fn forward_mixed_spec_finish(
        &mut self,
    ) -> Result<
        (
            Option<Vec<u32>>,
            Vec<(usize, crate::generator::FinishSample, usize)>,
        ),
        GenError,
    > {
        self.forward_mixed_spec_wait_impl().map_err(gen_err)
    }

    // True unified prefill+decode tick - one weight walk for both (the
    // scheduler calls this under the PADDOCK_UNIFIED opt-in; without it the
    // mixed tick's two forwards run)
    fn forward_unified_sampled(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
        plans: &[RowSample],
        _fin_plans: &[(usize, RowSample)],
    ) -> Result<
        (
            SampledStep,
            Vec<(usize, crate::generator::FinishSample, usize)>,
        ),
        GenError,
    > {
        self.forward_unified_sampled_impl(decodes, budget, plans)
            .map_err(gen_err)
            .map(|(step, fin)| {
                let fin = fin
                    .into_iter()
                    .map(|(k, l, r)| (k, crate::generator::FinishSample::Logits(l), r))
                    .collect();
                (step, fin)
            })
    }

    fn forward_multimodal(
        &mut self,
        chunks: &[crate::service::MmChunk],
    ) -> Result<Option<(Vec<f32>, usize)>, GenError> {
        if self.vision.is_none() {
            return Err(GenError::Backend(
                "gemma4 was loaded without an mmproj - configure `mmproj` to enable image input"
                    .into(),
            ));
        }
        self.multimodal_prefill(chunks)
            .map(Some)
            .map_err(|e| GenError::Backend(e.to_string()))
    }

    /// Matching qwen35: mm requests prefill into batch slots instead of
    /// draining the server for an exclusive pass. The exclusive path
    /// serializes vision serving entirely - eight concurrent image requests
    /// finish no faster than one, they just each wait their turn.
    fn supports_mm_slots(&self) -> bool {
        self.vision.is_some()
    }

    fn vision_budget(&self) -> Option<crate::generator::VisionBudget> {
        self.vision.as_ref().map(|v| v.budget())
    }

    fn forward_prefill_multimodal(
        &mut self,
        slot: usize,
        chunks: &[crate::service::MmChunk],
    ) -> Result<(Vec<f32>, usize), GenError> {
        self.multimodal_prefill_slot(slot, chunks)
            .map_err(|e| GenError::Backend(e.to_string()))
    }

    fn take_prefill_reused(&mut self, slot: usize) -> usize {
        self.prefix
            .as_mut()
            .map(|p| std::mem::take(&mut p.last_reused[slot]))
            .unwrap_or(0)
    }

    fn vocab(&self) -> usize {
        self.hp.n_vocab
    }

    fn max_context(&self) -> usize {
        self.max_ctx
    }
}

/// Depth-2 decode-pipe bookkeeping. `tick` = index of the last
/// launched tick; ring j%2 holds tick j's sampled ids, ev[j%2] fires when
/// the ring plane is written (stream-ordered after the tick's graph).
pub(crate) struct G4Pipe {
    pub(crate) r: usize,
    pub(crate) tick: u64,
    pub(crate) pos0: Vec<u32>,
    pub(crate) ev: [Option<cudarc::driver::CudaEvent>; 2],
}
