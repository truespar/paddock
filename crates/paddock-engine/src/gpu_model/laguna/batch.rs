//! Laguna continuous batching v1: paged KV with per-layer cache kinds,
//! batched decode, chunked per-slot prefill.
//!
//! Memory shape (the whole point of the hybrid): the 30 SWA-512 layers get
//! per-slot WindowRing block pools (gpt-oss G3 static-table scheme via
//! gemma4's build_swa_paging) - their KV never grows with context. The 10
//! full-attention layers share one budget pool of 16-token blocks addressed
//! through per-slot block tables (gpt-oss G4a shape) - capacity follows free
//! VRAM, not max_ctx × max_batch.
//!
//! Compute class: batched projections ride the W4A8 ladders (quantize the
//! normed rows once, mmq_pre_any dispatches dp4a/mma per weight) - the same
//! activation class qwen35's k-quant serving uses, and llama.cpp's own
//! prefill numeric class. The serial forward keeps its exact-f32 GEMVs; the
//! serve-level greedy gate vs llama.cpp arbitrates the batched class.
//!
//! Decode perf ladder: the fixed-r decode tick (embed ->
//! layer walk -> head) is captured into a per-r CUDA graph - one replay
//! replaces the ~700 launches of a 40-layer walk. On top of it ride device
//! sampling (`forward_batch_sampled`: eligible rows come back as bare ids,
//! no [r, vocab] readback) and the depth-2 decode pipe (tick N+1 enqueued
//! before tick N's ids reach the host - qwen35's scheme; positions are
//! deterministic so growth and the mrope rebuild stay host-side per tick).
//!
//! Capture safety here: the batched scratch is allocated once at enable
//! (addresses never move), every loop bound the kernels take at replay
//! comes from a device buffer (d_pos, the block tables), and all host work
//! (table growth + d_bt upload, row uploads) happens outside the captured
//! region. The prefix-cache blob still allocates dead last, after scratch.
//!
//! Admission rides COALESCED multi-prompt prefill (gemma4's scheme): the
//! whole wave's tail rows concatenate into shared PF_ROWS chunks - one
//! weight-amortized pass instead of one pass per prompt (a 32×128-token
//! burst used to be 32 sequential passes ≈ 2.8 s TTFT). Full
//! layers attend per same-slot run; the SWA ladder's sub-spans simply never
//! cross a run boundary. Deliberately not here yet (follow-ups): mixed
//! decode+prefill ticks, FlashDecoding splits (A6000's 84 SMs never engage
//! them).

use std::collections::HashMap;

use cudarc::driver::CudaSlice;
use cudarc::driver::sys::CUstreamCaptureMode;

use crate::gpu::{GpuError, KvDtype};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::qwen35::{gemv_any, mmq_kq_pre, mmq_pre, mmq_pre_any, prefill_mm_pre};
use crate::kv_plan;
use crate::kv_pool::{BlockTable, KvPool};

use super::*;

/// Rows per prefill chunk. GEMMs run whole chunks; the SWA append/attend
/// ladder steps in SWA_SPAN sub-spans (ring aliasing invariant below).
///
/// This is the prefill-throughput knob on a 256-expert MoE, not a buffer
/// nicety: any chunk past ~200 rows routes top-8 across essentially every
/// expert, so a pass streams the whole 18.9 GB expert set no matter how many
/// rows ride it. Rows per pass therefore divide straight into prefill
/// throughput - half the chunk size is double the weight traffic for the same
/// prompt. The scheduler already asks for 8192 rows a tick
/// (`prefill_tick_rows`); this cap is what it actually gets.
///
/// Runtime-settable (PADDOCK_PF_ROWS) because the scratch planes scale
/// linearly with it - q/qn alone are rows x 8192 x 4 B - so the right value
/// is a VRAM-budget decision, and a 24 GB card must be able to pick a
/// smaller one than a 48 GB card. Read once; the batched scratch is sized
/// from it at enable.
pub(crate) fn pf_rows() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_PF_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n: &usize| (256..=8192).contains(&n) && n % 256 == 0)
            .unwrap_or(PF_ROWS_DEFAULT)
    })
}

const PF_ROWS_DEFAULT: usize = 1024;

/// SWA sub-span: within one chunk, a SWA layer appends+attends at most this
/// many rows before the next sub-span reuses ring blocks the window has
/// moved past. Must be the only source for both the ring sizing and the
/// span cuts - a mismatch aliases live window blocks (gemma4's invariant).
pub(crate) const SWA_SPAN: usize = 512;

/// Prefill-mode dispatch cuts for one chunk. `runs` = contiguous same-slot
/// row runs - an attention launch must never mix two slots' QUERY rows (the
/// tile walk assumes one sequence), so full layers attend per run and the
/// SWA ladder's sub-spans are cut at run boundaries too. `swa` = the
/// append+attend ladder: same-slot spans, each ≤ SWA_SPAN (ring aliasing
/// invariant above). Single-prompt chunks carry runs = [(0, r)], which keeps
/// the whole-chunk attend calls byte-for-byte on yesterday's path.
///
/// `spec` = the spec-VERIFY flavor: same ragged appends/attends, but the
/// projections/MoE keep the DECODE numeric class (quantize_q8 + mmq_pre_any
/// small-r rungs, token-batched MoE, decode-rows attention). The prefill
/// class's fixed costs amortize over ~1024-row chunks and measured 4×
/// slower on 8-row verify rounds; spec rounds
/// need no warm==cold byte invariance - only self-consistency with the
/// picks they emit.
struct PfCuts {
    swa: Vec<(usize, usize)>,
    runs: Vec<(usize, usize)>,
    spec: bool,
    /// Leading rows that are DECODE rows (q_len 1, one per slot) in a fused
    /// mixed tick. They form one band - `runs[0]` = `swa[0]` = (0, dec) - and
    /// the attention dispatch sends that band to the DECODE-batch kernel.
    ///
    /// Without this the band arrives as `dec` separate one-row runs, each
    /// paying its own WMMA prefill launch (16-row tiles, 1 row used) at every
    /// one of the 40 layers: 33 launches/layer at c32 instead of 2, which
    /// measured as a real throughput loss until the band was folded. It is
    /// also the industry shape - FlashInfer/vLLM dispatch a unified batch as
    /// a decode wrapper over the q_len==1 rows plus a prefill wrapper over
    /// the chunk, not one ragged kernel.
    dec: usize,
}

impl PfCuts {
    /// Cuts for `runs` (contiguous same-slot spans covering 0..r in order):
    /// the SWA ladder steps each run at SWA_SPAN.
    fn new(runs: Vec<(usize, usize)>) -> Self {
        let mut swa = Vec::new();
        for &(off, len) in &runs {
            let mut o = 0usize;
            while o < len {
                let l = SWA_SPAN.min(len - o);
                swa.push((off + o, l));
                o += l;
            }
        }
        Self {
            swa,
            runs,
            spec: false,
            dec: 0,
        }
    }

    /// The FUSED mixed-tick flavor: `dec` decode rows at the front (one band,
    /// never laddered - each row appends exactly one token, so nothing it
    /// reads can be recycled within the band), then the chunk's same-slot
    /// runs at offsets >= dec.
    fn fused(dec: usize, chunk_runs: Vec<(usize, usize)>) -> Self {
        let mut c = Self::new(chunk_runs);
        if dec > 0 {
            c.runs.insert(0, (0, dec));
            c.swa.insert(0, (0, dec));
            c.dec = dec;
        }
        c
    }

    /// The spec-verify flavor (decode class; see struct docs).
    fn spec(runs: Vec<(usize, usize)>) -> Self {
        Self {
            spec: true,
            ..Self::new(runs)
        }
    }
}

/// VRAM slack the slot-fit math leaves untouched (graph/scratch churn).
const VRAM_HEADROOM: usize = 1 << 30;
/// Retained-prefix slack above the addressable ceiling, in slots' worth of
/// blocks: `min(slots, this) x blocks_per_slot` (the gemma4 rule, 2026-09-06).
const RETENTION_SLOTS_MAX: usize = 8;

/// Cap on ragged verify rows one spec round may carry - sizes head_logits
/// and the spec sampler planes at enable (addresses must never move: the
/// decode graphs bake head_logits). The service's row budget stays ≤64
/// under defaults; 128 leaves env headroom. Rounds above the cap decline
/// (Ok(None)) rather than corrupt.
pub(crate) const SPEC_ROWS: usize = 128;

/// Cap on cached spec-verify graphs (distinct chunk-length signatures). The
/// k ladder × ≤8 warm slots yields a handful of keys in steady state; a
/// pathological mix just runs its overflow signatures eagerly.
const SPEC_GRAPH_CAP: usize = 64;

/// PADDOCK_LAGUNA_SPEC=1 enables the laguna speculative lane (
/// verify via the batched decode class + DFlash/n-gram drafts). Off by
/// default until the spec-on outputs are vetted against the llama greedy
/// oracle - the stored srv-parity refs were captured nospec, and
/// verify-class picks can flip near-ties. Also gates the DFlash serving
/// state (rings + aux capture): with it off, serving is byte-identical
/// even with the drafter attached.
pub(crate) fn laguna_spec_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_LAGUNA_SPEC").is_some())
}

/// FlashDecoding split ceiling (partial-scratch sizing).
const MAX_ATTN_SPLITS: usize = 16;

/// KV splits for the batched decode attention. The unsplit kernel is
/// n_heads×batch blocks - 48-64 at r=1 on laguna - which leaves most of even
/// the 84-SM A6000 idle while every block walks its whole KV run serially
/// (18.7 µs/layer at ctx 130, where a split kernel does the same work in a
/// fraction of that). Same election shape as qwen35's on A6000: neutral at
/// short ctx, a big win by ~4.7k. Position-INDEPENDENT so
/// the captured per-r graph can bake it.
/// True when the pack's batched partial fuses the q-group per KV head for
/// this shape (one K/V stage serves all `group` q-heads). Must mirror the
/// paged launcher's own predicate (group 2..=8, n_kv >= 2, incl. the
/// PD_NO_GQA_FUSE pin) - split budgeting and the partial-vs-plain dispatch
/// both key off it. Same law gpt_oss learned: fused shapes route through
/// partial+combine even at n_splits == 1, because the plain per-q-head
/// kernel re-reads each K/V tile group(=6-8)x.
fn attn_gqa_fused(n_heads: usize, n_kv_heads: usize, batch: usize) -> bool {
    let group = n_heads.checked_div(n_kv_heads).unwrap_or(1);
    batch > 1
        && (2..=8).contains(&group)
        && n_kv_heads >= 2
        && n_heads == n_kv_heads * group
        && std::env::var_os("PD_NO_GQA_FUSE").is_none()
}

fn attn_splits(
    n_heads: usize,
    n_kv_heads: usize,
    batch: usize,
    sm_count: usize,
    window: usize,
) -> usize {
    if paddock_models::dev_var_os!("PADDOCK_NO_ATTN_SPLIT").is_some() {
        return 1;
    }
    // SWA span cap: any split past ceil(window / (4*TILE)) is an idle CTA
    // writing an empty (-inf, 0) partial the combine folds as identity
    // (512-key windows carved 16 ways ran mostly-idle CTAs and a 16-way
    // combine). BIT-EXACT trim - the kernel's adaptive s_eff
    // picks the same live splits either way; we just stop launching the
    // identities. 128 mirrors the kernel's >=4-tiles-per-live-CTA target at
    // the hd128 dispatch tile (4 * TILE, TILE = 32) - keep in sync with the
    // partial launcher.
    let cap = if window > 0 {
        window.div_ceil(128).clamp(1, MAX_ATTN_SPLITS)
    } else {
        MAX_ATTN_SPLITS
    };
    if attn_gqa_fused(n_heads, n_kv_heads, batch) {
        // Batched decode: the fused walk launches n_kv*batch blocks per
        // split, so budget the die on those - the old q-head test called c8
        // "saturated" (512 q-head blocks) and dropped to the plain kernel's
        // group-x KV re-reads, when the fused grid was actually 64 blocks.
        let want = (2 * 3 * sm_count).div_ceil(n_kv_heads * batch).max(1);
        return want.min(cap);
    }
    if n_heads * batch >= 2 * 3 * sm_count {
        return 1; // die already saturated (un-fused shapes only)
    }
    if let Some(n) = paddock_models::dev_var!("PADDOCK_ATTN_SPLITS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n >= 1)
    {
        return n.min(MAX_ATTN_SPLITS);
    }
    cap
}

/// PADDOCK_NO_QK_NRA=1 pins the six-kernel decode epilogue chain (q/k norms
/// + ropes + appends as separate launches) - the A/B + bisect escape for the
///   fused lag_qk_nra_rows fold (bit-identical, so this is a pure-perf pin).
fn no_qk_nra() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_QK_NRA").is_some())
}

/// PADDOCK_NO_W4A8_PREFILL=1 pins the strided dp4a ladder at prefill rows -
/// the A/B + bisect escape for the flat-mmq tensor-core class.
fn no_w4a8_prefill() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_W4A8_PREFILL").is_some())
}

/// PADDOCK_NO_WMMA_PREFILL=1 pins the scalar prefill-attention pair -
/// the A/B + bisect escape for the f16 WMMA class (hd-128 arm).
fn no_wmma_prefill() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_WMMA_PREFILL").is_some())
}

/// Laguna's two shapes (hd128, G=6 full-attn / G=9 SWA) ride the v4
/// staged-HMMA tile's hd128 arm (same kernel family as
/// qwen35's hd256 G in {4,6,8}, extended to hd128 G in {4,6,9} for
/// granite/laguna - G=9 is net-new, MR=144, the largest o_acc register
/// footprint in the tile family, verified LOCAL:0/no-spill at REG:124-126
/// on sm_120a) once it grew a raw-e4m3 PIPE arm - before that the elected
/// kv8 class fell to the scalar paged walk. The export falls back to that
/// scalar tile when the v4 arm is killed (PADDOCK_NO_PF_V4), so this gate
/// only decides the ENGINE routing; PADDOCK_NO_NPF8 reverts it (mirrors
/// qwen35's PADDOCK_NO_QPF8).
fn pf_attn_dtype_ok(kv_dtype: KvDtype, n_heads: usize, n_kv_heads: usize) -> bool {
    match kv_dtype {
        KvDtype::Fp16 => true,
        KvDtype::Fp8E4m3 => {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_NPF8").is_none())
                && n_kv_heads > 0
                && n_heads.is_multiple_of(n_kv_heads)
                && matches!(n_heads / n_kv_heads, 4 | 6 | 9)
        }
    }
}

/// PADDOCK_NO_COALESCED_PREFILL=1 pins the serial one-prompt-per-pass
/// admission - the A/B + bisect escape for the coalesced multi-prompt wave
/// (bit-identical per prompt by the r-invariance law, so a flip here is a
/// pure-perf probe; a byte diff means the law broke somewhere).
fn no_coalesced_prefill() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_COALESCED_PREFILL").is_some())
}

/// Prefill-mode quantize: always the flat-mmq layout + per-32 sums, at any r.
/// Prefill bytes must be r-INVARIANT - a warm-resume tail (tiny r) has to
/// reproduce the cold chunk's bytes exactly, so the prefill class can never
/// switch rungs on row count (the strided nc/mma_ks/dp4a ladder is bit-exact
/// across r by construction and stays the DECODE class; w4a8's 128×128 mma
/// tile accumulates in a different order, so it must own prefill outright).
/// Q8_0 planes additionally need the strided layout at r ≤ 64 - their ladder
/// is bit-exact across its own rungs, so no forcing there.
#[allow(clippy::too_many_arguments)]
fn pf_quant(
    exec: &crate::gpu::GpuExecutor,
    xq: &mut CudaSlice<i8>,
    xs: &mut CudaSlice<f32>,
    yq: &mut CudaSlice<u8>,
    xsums: &mut CudaSlice<f32>,
    x: &CudaSlice<f32>,
    in_dim: usize,
    r: usize,
    any_q8: bool,
) -> Result<(), GpuModelError> {
    exec.quantize_q8_mmq(x, yq, in_dim, r)?;
    exec.mmq_sums(yq, xsums, in_dim, r)?;
    if any_q8 && r <= 64 {
        exec.quantize_q8(x, xq, xs, r * in_dim)?;
    }
    Ok(())
}

/// Prefill-mode projection off [`pf_quant`]'s planes: k-quant rides the W4A8
/// int8 tensor-core GEMM at any r (see the r-invariance note above); Q8_0
/// keeps `prefill_mm_pre`'s bit-exact ladder.
#[allow(clippy::too_many_arguments)]
fn pf_mm(
    exec: &crate::gpu::GpuExecutor,
    w: &QuantW,
    xq: &CudaSlice<i8>,
    xs: &CudaSlice<f32>,
    yq: &CudaSlice<u8>,
    xsums: &CudaSlice<f32>,
    skfix: &mut CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    r: usize,
) -> Result<(), GpuModelError> {
    match w {
        QuantW::Kq(k) => {
            let needs = crate::gpu::kq_needs_sums(k.ty);
            exec.kquant_gemm_w4a8(k, yq, needs.then_some(xsums), y, r)?;
            Ok(())
        }
        QuantW::Q8(q) => prefill_mm_pre(exec, q, xq, xs, yq, skfix, y, r),
    }
}

/// r=1 W4A8 serving-class GEMV off PRE-STAGED int8 activations (qwen35's
/// kq_w4a8_b1 pattern): k-quant weights ride the dp4a GEMV in
/// llama mmvq's own activation-quant numeric class; Q8_0 keeps the exact-f32
/// GEMV (already at its byte floor, and no sums plane to feed). Callers
/// stage xq/xs/ssums once per shared input (the quantize-dedupe rule).
/// PADDOCK_NO_GEMV_MULTI=1 restores the split decode GEMV launches - the A/B
/// reference for the (q|k)+(v|g) and shexp gate|up one-launch merges (same
/// env as granite's, entry 317).
fn no_gemv_multi() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_GEMV_MULTI").is_some())
}

fn gemv8_any(
    exec: &crate::gpu::GpuExecutor,
    w: &QuantW,
    x: &CudaSlice<f32>,
    xq: &CudaSlice<i8>,
    xs: &CudaSlice<f32>,
    ssums: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
) -> Result<(), GpuModelError> {
    match w {
        QuantW::Kq(k) => {
            let needs = crate::gpu::kq_needs_sums(k.ty);
            exec.kquant_gemv_w4a8(k, xq, xs, needs.then_some(ssums), y)?;
            Ok(())
        }
        QuantW::Q8(_) => gemv_any(exec, w, x, y),
    }
}

pub(crate) struct LayerKv {
    pub k: CudaSlice<u8>,
    pub v: CudaSlice<u8>,
}

/// Batched-lane scratch, sized once at enable for PF_ROWS-row chunks (decode
/// reuses the same planes at rows = live slots « PF_ROWS).
pub(crate) struct BatchScratch {
    pub x: CudaSlice<f32>,  // residual [PF, embd]
    pub xn: CudaSlice<f32>, // normed rows [PF, embd]
    /// group-quantized activations - sized to the WIDEST quantize consumer
    /// (attn out q_max, or the dense gated FFN rows: 8192 on XS-2.1, 12288
    /// on S-2.1), so one plane serves every quantize site
    pub xq: CudaSlice<i8>, // [PF * wide]
    pub xs: CudaSlice<f32>, // [PF * wide / 32]
    pub ssums: CudaSlice<f32>, // per-16 int8 sums (Q4_K/Q5_K mu) [PF * 512]
    /// routed-down mu sums - its own plane:
    /// the down-input quantize used to regenerate `ssums` in place, which
    /// silently invalidated xn's sums mid-layer. Harmless while nothing
    /// read them afterwards; the r1 W4A8 shexp gate_up became the first
    /// post-clobber reader and its Q4_K mu-terms went garbage on every
    /// Q4K-down layer (20 of 39 in this file - dtypes alternate).
    pub moe_ssums: CudaSlice<f32>,
    /// dedicated shexp-down input quant: keeps the r1 W4A8 shexp
    /// chain off the shared xq/xs/ssums planes entirely
    pub sh_xq: CudaSlice<i8>, // [shexp_ff]
    pub sh_xs: CudaSlice<f32>,    // [shexp_ff / 32]
    pub sh_ssums: CudaSlice<f32>, // [shexp_ff / 16]
    pub part: CudaSlice<f32>,     // mma_ks K-split partials [8 * 64 * wide]
    /// flat-mmq activations for the prefill (r > 64) W4A8 tensor-core GEMMs:
    /// [in_dim/128 chunks][pad128(rows)] × (4 f32 scales + 128 int8) = 144 B
    pub yq: CudaSlice<u8>,
    /// per-32-block sums off yq ([chunk][col_pad][4] f32) - the W4A8 Q4/Q5 mu term
    pub xsums: CudaSlice<f32>,
    /// Q8_0 mmq stream-k fixup plane - only reachable through a Q8_0 weight
    /// plane (1-elem placeholder on all-k-quant files like XS Q4_K_M)
    pub skfix: CudaSlice<f32>,
    pub q: CudaSlice<f32>, // [PF, 64*128]
    pub qn: CudaSlice<f32>,
    pub k: CudaSlice<f32>, // [PF, kv_dim]
    pub kn: CudaSlice<f32>,
    pub v: CudaSlice<f32>,
    pub gate_h: CudaSlice<f32>, // per-head gate pre-softplus [PF, 64]
    pub attn: CudaSlice<f32>,   // [PF, 64*128]
    pub proj: CudaSlice<f32>,   // [PF, embd]
    /// [64] -inf - laguna has no attention sinks, and -inf (not zero) is the
    /// no-op value for the softmax denominator. See `alloc_no_sinks`.
    pub sinks: CudaSlice<f32>,
    /// dense-layer gated FFN rows [PF, dense_ff] - width comes from the
    /// weights (XS-2.1 8192, S-2.1 12288), never a constant
    pub ffn_gate: CudaSlice<f32>,
    pub ffn_up: CudaSlice<f32>,
    pub moe_logits: CudaSlice<f32>, // [PF, n_expert]
    pub moe_idx: CudaSlice<u32>,    // [PF, k]
    pub moe_w: CudaSlice<f32>,      // [PF, k]
    pub moe_fused: CudaSlice<f32>,  // [PF, k*moe_ff]
    /// fq/fs serve both layouts: token-batched [PF, k*moe_ff] and the
    /// sorted moe_align layout [max_blocks*32, moe_ff] (PAD rows) - sized
    /// for the larger (sorted at PF rows).
    pub moe_fq: CudaSlice<i8>,
    pub moe_fs: CudaSlice<f32>,
    // sorted (moe_align) MoE lane - the prefill class (qwen35's serving
    // path): each touched expert's weights read once per pass
    pub srow: CudaSlice<u32>,     // [max_blocks*32] sorted pair -> token row
    pub sslot: CudaSlice<u32>,    // [max_blocks*32] sorted pair -> k-slot
    pub bexp: CudaSlice<u32>,     // [max_blocks] block -> expert
    pub moe_part: CudaSlice<f32>, // [PF, k, embd] down partials
    pub sh_gate: CudaSlice<f32>,  // [PF, shexp_ff]
    pub sh_up: CudaSlice<f32>,
    pub sh_out: CudaSlice<f32>,  // [PF, embd]
    pub d_toks: CudaSlice<u32>,  // [PF]
    pub d_pos: CudaSlice<u32>,   // [PF]
    pub d_slots: CudaSlice<u32>, // [PF]
    /// axis-major [4, PF] mrope positions - text collapses all four axes to
    /// the plain position; only axis 0 is live under sections [n_rot/2,0,0,0]
    pub d_mrope: CudaSlice<u32>,
    /// [max(n_slots, SPEC_ROWS), vocab] decode logits (prefill reads one row
    /// through it; the spec verify fills up to SPEC_ROWS ragged rows).
    /// Decode graphs bake this address - allocated once, never grown.
    pub head_logits: CudaSlice<f32>,
    /// spec-round sampler params [SPEC_ROWS, 4] + picks [SPEC_ROWS] - their
    /// own planes so the pipe's d_par/d_out rings are never aliased
    pub d_spec_par: CudaSlice<u32>,
    pub d_spec_out: CudaSlice<u32>,
    /// device sampler params [2 rings, n_slots, 4] (inv_t, u, mode, pad) -
    /// ring-doubled for the pipe; the classic sampled tick uses ring 0
    pub d_par: CudaSlice<u32>,
    /// sampled token ids [2 rings, n_slots]
    pub d_out: CudaSlice<u32>,
    /// mode-5 truncation side plane [2 rings, n_slots, 4] {k, top_p bits, min_p
    /// bits, pad} - laguna's election is 1.0/k20/p1.0, so every un-dialled
    /// request is a truncation row; pd_sample_rows_t draws them on device
    pub d_tpar: CudaSlice<u32>,
    /// spec twin ([SPEC_ROWS, 4], its own plane like d_spec_par)
    pub d_spec_tpar: CudaSlice<u32>,
    /// FlashDecoding partial scratch [n_heads_max, n_slots, MAX_SPLITS, hd]
    pub attn_o: CudaSlice<f32>,
    /// per-partial (m, l) [n_heads_max, n_slots, MAX_SPLITS, 2]
    pub attn_ml: CudaSlice<f32>,
}

/// The whole batching state: rings + pool + tables + scratch. One struct so
/// enable/teardown is atomic and the field-borrow splits stay simple.
pub(crate) struct BatchState {
    pub n_slots: usize,
    /// Row capacity of every scratch plane = `pf_rows()` (the chunk) + one
    /// row per slot (a fused mixed tick's decode band). See the note where
    /// it's computed in `enable_batch_impl` - the band must not eat chunk
    /// rows, because rows-per-pass divide straight into prefill throughput.
    pub cap: usize,
    /// static SWA ring table [slots, bps]: logical block j of slot s lives at
    /// pool block s*ring + j%ring. One table serves all 30 SWA layers.
    pub swa_bt: CudaSlice<u32>,
    pub ring: usize,
    /// logical blocks per slot (max_ctx/16) - both tables' slot stride
    pub bps: usize,
    /// full-attention budget pool + per-slot logical->physical tables
    pub pool: KvPool,
    pub tables: Vec<BlockTable>,
    pub bt_host: Vec<u32>,
    pub d_bt: CudaSlice<u32>,
    /// per-layer K/V stores: SWA layers [slots*ring blocks], full layers
    /// [pool_blocks blocks]; both [blocks, 16, kv_dim] × f16
    pub kv: Vec<LayerKv>,
    pub sc: BatchScratch,
    /// device bytes the KV stores hold (accounting)
    pub kv_bytes: u64,
    /// radix prefix cache + SWA window checkpoints (prefix.rs); None when
    /// PADDOCK_NO_PREFIX_CACHE is set
    pub prefix: Option<super::prefix::LagunaPrefix>,
    /// captured decode ticks, keyed by row count r - one replay per step
    /// instead of ~700 launches. Grid-stable for a given r (KV loop bounds
    /// come from d_pos / the device block tables at replay).
    pub graphs: HashMap<usize, super::SendGraph>,
    /// captured spec-verify rounds, keyed by the per-req chunk-LENGTH
    /// signature (row offsets derive from the lengths; tokens/positions/
    /// slots are staged device data, so one graph serves any slots at any
    /// positions with that shape). Steady-state DFlash traffic is a handful
    /// of keys - [16]×warm-slots once the k ladder tops out.
    pub spec_graphs: HashMap<Vec<u32>, super::SendGraph>,
    /// any Q8_0 weight plane in the file - gates the strided side-quantize in
    /// pf_quant (Q8_0's ≤64-row prefill rung reads the strided layout)
    pub any_q8: bool,
}

/// An in-flight depth-2 decode pipe: `ev[tick % 2]` fires when that tick's
/// out-ring plane is readable. Positions are deterministic (`pos0[i] + tick`)
/// so per-tick growth and the mrope rebuild stay host-side; only the token
/// rides the device (pipe_advance from the previous tick's sampled ids).
pub(crate) struct PipeState {
    b: usize,
    tick: u64,
    ev: [Option<cudarc::driver::CudaEvent>; 2],
    pos0: Vec<u32>,
    /// explicit row->slot mapping (None = identity). No restore-at-drain
    /// needed: every non-pipe path re-uploads d_slots before use.
    slots: Option<Vec<u32>>,
}

fn drv(e: cudarc::driver::DriverError) -> GpuError {
    crate::gpu::from_driver(e)
}

impl GpuLaguna {
    /// Allocate the paged-KV + scratch state for up to `max_batch` slots.
    /// Returns the capacity actually enabled; 1 = stay on the serial loop
    /// (no paged kernels in the pack, or max_batch 1).
    pub(crate) fn enable_batch_impl(&mut self, max_batch: usize) -> Result<usize, GpuModelError> {
        // NOTE: max_batch==1 is a real, supported width here, not a synonym
        // for "stay serial" - see granite/batch.rs's identical note and
        // service.rs's routing note. `--max-batch 1` used to bail
        // unconditionally before checking anything else, silently losing
        // prefix caching, the paged KV pool, and the tuned decode kernels.
        if !self.exec.has_paged_kv() {
            return Ok(1);
        }
        // the serial dense KV (40 layers × max_ctx) makes way for the paged
        // stores; the serial lane is dead once the engine goes batched
        self.decode = None;
        self.scratch = None;
        self.batch = None;
        self.pipe = None;
        // the DFlash serving state bakes this lane's scratch addresses and
        // slot count (rings, aux bands, captured draft graphs) - it must die
        // and rebuild with the lane, never survive a re-enable
        if let Some(d) = self.dflash.as_mut() {
            d.state = None;
        }
        self.exec.trim_mem_pool();

        let hp = &self.hp;
        let kv_dim = hp.n_kv_heads * hp.head_dim;
        let kvb = self.kv_dtype.bytes();
        let bps = self.max_ctx.div_ceil(16);
        let ring = ((SWA_SPAN + hp.swa_window).div_ceil(16) + 1).min(bps);
        let n_swa = self.layers.iter().filter(|l| l.is_swa).count();
        let n_full = self.layers.len() - n_swa;

        // slot-fit: rings are the only per-slot KV cost; the full layers ride
        // the shared pool
        let per_slot = n_swa * ring * 16 * kv_dim * 2 * kvb;
        let block_bytes = n_full * 16 * kv_dim * 2 * kvb;
        // Prefill scratch PROFILED at load (the qwen35/granite shape,
        // 2026-09-06): allocated here, before the grant is read, so the plan
        // meets its true cost in the ledger. It was a flat 512 MiB reserve.
        // Slot-sized planes take the REQUESTED width: the plan may seat
        // fewer, never more.
        let e = &self.exec;
        let v_scratch0 = cudarc::driver::result::mem_get_info()
            .map(|(f, _)| f)
            .unwrap_or(0);
        let m = &hp.moe;
        let n_heads_max = hp.n_heads.iter().copied().max().unwrap_or(64);
        let q_max = n_heads_max * hp.head_dim;
        // Dense-FFN width is MODEL-driven, not a constant: XS-2.1 is 8192 but
        // S-2.1 is 12288, and the gated planes below feed the down-proj at
        // that width. Reading it from the weights (same derivation the serial
        // path uses, forward.rs `ff_max`) is what keeps S from writing past
        // the end of ffn_gate/ffn_up - a silent corruption, not a clean OOB.
        let dense_ff = self
            .layers
            .iter()
            .filter_map(|l| match &l.ffn {
                Ffn::Dense { gate, .. } => Some(gate.dims()[1]),
                Ffn::Moe(_) => None,
            })
            .max()
            .unwrap_or(0);
        // widest quantize consumer: attn out, the dense gated rows, or the
        // 8192 floor every existing file already satisfied
        let wide = q_max.max(dense_ff).max(8192);
        // the Q8_0 mmq arm (and its stream-k fixup plane) only runs when a
        // Q8_0 weight plane exists - XS Q4_K_M is all k-quant, so skip the
        // 16 MB plane there (S-2.1 Q8_0's signal path would light it up)
        let q8 = |w: &QuantW| matches!(w, QuantW::Q8(_));
        let any_q8 = q8(&self.lm_head)
            || self.layers.iter().any(|l| {
                q8(&l.wq)
                    || q8(&l.wk)
                    || q8(&l.wv)
                    || q8(&l.g_proj)
                    || q8(&l.wo)
                    || match &l.ffn {
                        Ffn::Dense { gate, up, down } => q8(gate) || q8(up) || q8(down),
                        Ffn::Moe(w) => q8(&w.shexp_gate) || q8(&w.shexp_up) || q8(&w.shexp_down),
                    }
            });
        let fused_len = m.n_active * m.moe_ff;
        // ROW CAPACITY of every scratch plane. A fused mixed tick carries the
        // decode band (≤ one row per slot) on TOP of a full pf_rows() chunk,
        // so the planes hold both - sizing them at pf_rows() alone would make
        // the band steal chunk rows, and on this 256-expert MoE rows-per-pass
        // divide straight into prefill throughput (a 1024 -> 992 chunk
        // measured -3.2% on 2048×128 c32). Costs slots/pf_rows of the scratch
        // (~3% at the 32-slot / 1024-row default).
        let cap = pf_rows() + max_batch;
        // sorted-MoE worst case: every expert pads its last block
        let sorted_rows = (cap * m.n_active + m.n_expert * 31).div_ceil(32) * 32;
        let sc = BatchScratch {
            x: e.alloc(cap * hp.n_embd)?,
            xn: e.alloc(cap * hp.n_embd)?,
            xq: e.alloc_i8(cap * wide)?,
            xs: e.alloc(cap * wide / 32)?,
            ssums: e.alloc(cap * wide / 16)?,
            moe_ssums: e.alloc((cap * wide / 16).max(sorted_rows * m.moe_ff / 16))?,
            sh_xq: e.alloc_i8(m.shexp_ff.max(32))?,
            sh_xs: e.alloc(m.shexp_ff.max(32) / 32)?,
            sh_ssums: e.alloc(m.shexp_ff.max(32) / 16)?,
            part: e.alloc(8 * 64 * wide)?,
            yq: e.alloc_u8(wide.div_ceil(128) * cap.next_multiple_of(128) * 144)?,
            xsums: e.alloc(wide.div_ceil(128) * cap.next_multiple_of(128) * 4)?,
            skfix: e.alloc(if any_q8 { 256 * 128 * 128 + 256 } else { 1 })?,
            q: e.alloc(cap * q_max)?,
            qn: e.alloc(cap * q_max)?,
            k: e.alloc(cap * kv_dim)?,
            kn: e.alloc(cap * kv_dim)?,
            v: e.alloc(cap * kv_dim)?,
            gate_h: e.alloc(cap * n_heads_max)?,
            attn: e.alloc(cap * q_max)?,
            proj: e.alloc(cap * hp.n_embd)?,
            sinks: e.alloc_no_sinks(n_heads_max)?,
            ffn_gate: e.alloc(cap * dense_ff.max(1))?,
            ffn_up: e.alloc(cap * dense_ff.max(1))?,
            moe_logits: e.alloc(cap * m.n_expert)?,
            moe_idx: e.alloc_u32(cap * m.n_active)?,
            moe_w: e.alloc(cap * m.n_active)?,
            moe_fused: e.alloc(cap * fused_len)?,
            moe_fq: e.alloc_i8(sorted_rows * m.moe_ff)?,
            moe_fs: e.alloc(sorted_rows * m.moe_ff / 32)?,
            srow: e.alloc_u32(sorted_rows)?,
            sslot: e.alloc_u32(sorted_rows)?,
            bexp: e.alloc_u32(sorted_rows / 32)?,
            moe_part: e.alloc(cap * m.n_active * hp.n_embd)?,
            sh_gate: e.alloc(cap * m.shexp_ff)?,
            sh_up: e.alloc(cap * m.shexp_ff)?,
            sh_out: e.alloc(cap * hp.n_embd)?,
            d_toks: e.alloc_u32(cap)?,
            d_pos: e.alloc_u32(cap)?,
            d_slots: e.alloc_u32(cap)?,
            d_mrope: e.alloc_u32(4 * cap)?,
            head_logits: e.alloc(max_batch.max(SPEC_ROWS) * hp.n_vocab)?,
            d_spec_par: e.alloc_u32(SPEC_ROWS * 4)?,
            d_spec_out: e.alloc_u32(SPEC_ROWS)?,
            d_par: e.alloc_u32(2 * max_batch * 4)?,
            d_out: e.alloc_u32(2 * max_batch)?,
            d_tpar: e.alloc_u32(2 * max_batch * 4)?,
            d_spec_tpar: e.alloc_u32(SPEC_ROWS * 4)?,
            attn_o: e.alloc(n_heads_max * max_batch * MAX_ATTN_SPLITS * hp.head_dim)?,
            attn_ml: e.alloc(n_heads_max * max_batch * MAX_ATTN_SPLITS * 2)?,
        };
        let v_scratch1 = cudarc::driver::result::mem_get_info()
            .map(|(f, _)| f)
            .unwrap_or(0);
        tracing::info!(
            "laguna VRAM  prefill scratch (cap={cap} rows)      {:>7.2} GB  (in the ledger before the grant)",
            v_scratch0.saturating_sub(v_scratch1) as f64 / 1e9
        );
        // One arbiter sizes the KV store: crate::kv_plan. Laguna's own
        // arithmetic was already budget-correct - this is the same solve, moved
        // somewhere a new family cannot forget to do it, and it reports the pool's
        // TOKEN CAPACITY rather than leaving max_ctx to imply it.
        let grant = self
            .exec
            .vram_headroom()
            .ok_or_else(|| GpuError::Driver("no free-VRAM reading".into()))?;
        // What the process holds as the plan is made (weights, the profiled
        // scratch): the audit at the end adds the plan's charges to it.
        let ledger_at_plan = self.exec.process_mem_used().unwrap_or(0);
        let demand = kv_plan::Demand {
            family: "laguna",
            max_ctx: self.max_ctx,
            slots: max_batch,
            blocks_per_slot: bps,
            block_bytes: block_bytes as u64,
            // rings are the only per-slot KV cost; the full layers ride the pool
            per_slot_bytes: per_slot as u64,
            // admission slack for radix retention: nodes hold blocks after their
            // sequence ends. Cheap here (~1 MiB/block-set) - granite prices the
            // same slack explicitly instead, because one block costs it 4 MiB.
            // Retained-prefix slack sized by demand (the gemma4 rule): one
            // retained context per seated slot, capped at eight - a flat
            // 8 x bps let a one-slot server fill the grant with a pool nine
            // times its own context.
            retention_blocks: max_batch.min(RETENTION_SLOTS_MAX) * bps,
            // a pool floor so slots cannot starve the full layers: enough blocks
            // for every slot to hold a PF_ROWS-deep prompt, or admission
            // deadlocks on its own first chunk
            floor_blocks_per_slot: pf_rows().div_ceil(16),
            floor_blocks_min: 256,
            reserves: {
                let mut r = vec![
                    kv_plan::Reserve::new("graph/scratch slack", VRAM_HEADROOM as u64),
                    kv_plan::Reserve::new("prefix checkpoints", self.prefix_vram_estimate() as u64),
                ];
                if crate::kv_tier::pool_tier::tier_ram_bytes().is_some() {
                    r.push(kv_plan::Reserve::new(
                        "kv-tier staging",
                        crate::kv_tier::ram_transport::device_staging_bytes(),
                    ));
                }
                r
            },
            ..Default::default()
        };
        let plan = demand
            .plan(grant)
            .map_err(|e| GpuModelError::WontFit(e.message))?;
        let slots = plan.slots;
        // One seated slot is a batched server, not a reason to go serial: a
        // `--max-batch 1` request seats exactly one and still wants the
        // prefix cache, the paged lanes and the drafter. The old
        // `slots <= 1 -> WontFit("staying serial")` guard sent every
        // single-user laguna server to the serial engine (dense serial KV, no
        // prefix reuse: 42 GiB for a 21.8 GB model at 131k on 2026-09-06).
        // A wider request the plan narrows to one still builds here; the
        // service decides what to do with a width of one.
        plan.report(&demand, grant);
        let pool_blocks = plan.pool_blocks;
        let plan_reserved: u64 = demand.reserves.iter().map(|r| r.bytes).sum::<u64>()
            + plan.pool_bytes
            + plan.slot_bytes;

        // static ring table (gemma4 build_swa_paging shape)
        let mut swa_host = vec![0u32; slots * bps];
        for s in 0..slots {
            for j in 0..bps {
                swa_host[s * bps + j] = (s * ring + (j % ring)) as u32;
            }
        }
        let swa_bt = e.to_device_u32(&swa_host)?;

        let mut kv = Vec::with_capacity(self.layers.len());
        let mut kv_bytes = 0u64;
        for l in &self.layers {
            let blocks = if l.is_swa { slots * ring } else { pool_blocks };
            let bytes = blocks * 16 * kv_dim * kvb;
            kv_bytes += 2 * bytes as u64;
            kv.push(LayerKv {
                k: e.alloc_u8(bytes)?,
                v: e.alloc_u8(bytes)?,
            });
        }

        self.batch = Some(BatchState {
            n_slots: slots,
            cap,
            swa_bt,
            ring,
            bps,
            pool: KvPool::with_blocks(pool_blocks as u32),
            tables: (0..slots).map(|_| BlockTable::new()).collect(),
            bt_host: vec![0u32; slots * bps],
            d_bt: e.alloc_u32(slots * bps)?,
            kv,
            sc,
            kv_bytes,
            prefix: None,
            graphs: HashMap::new(),
            spec_graphs: HashMap::new(),
            any_q8,
        });
        // DFlash serving state (feature rings + fusion staging) builds here,
        // inside the tick-stable region: the aux bands are written by
        // captured decode graphs, so they must allocate before the prefix
        // checkpoint blob claims dead-last. Spec-gated - without the env the
        // state never exists and serving stays byte-identical.
        if self.dflash.is_some() && laguna_spec_on() {
            self.dflash_ensure_state()?;
        }
        // the checkpoint blob allocates dead last (address stability for
        // everything the decode tick touches, as gemma4 does)
        self.build_prefix(slots)?;
        tracing::info!(
            "laguna batch: {slots} slots, {n_swa} SWA rings ({ring} blocks/slot/layer, \
             {:.2} GiB/slot) + {n_full}-layer pool {pool_blocks} blocks ({:.2} GiB)",
            per_slot as f64 / (1u64 << 30) as f64,
            (pool_blocks * block_bytes) as f64 / (1u64 << 30) as f64,
        );
        // Plan audit (the qwen35 line): the plan's charges on top of what the
        // process held when it was made, against the ledger now.
        let expected = ledger_at_plan + plan_reserved;
        let actual = self.exec.process_mem_used().unwrap_or(0);
        tracing::info!(
            expected_gib = expected as f64 / (1u64 << 30) as f64,
            ledger_gib = actual as f64 / (1u64 << 30) as f64,
            unplanned_mib = actual.saturating_sub(expected) as f64 / (1u64 << 20) as f64,
            "laguna VRAM plan audit: ledger vs plan after enable_batch"
        );
        Ok(slots)
    }

    /// Back every `(slot, position)` this pass will touch with a physical
    /// full-layer pool block, re-uploading the device table once on growth.
    /// PoolExhausted surfaces to the scheduler, which preempts (no prefix
    /// radix to evict yet -).
    fn ensure_full_rows(&mut self, slots: &[u32], positions: &[u32]) -> Result<(), GpuModelError> {
        let bs = self.batch.as_mut().expect("batch enabled");
        let mut grew = false;
        for (i, &s) in slots.iter().enumerate() {
            let s = s as usize;
            let before = bs.tables[s].blocks().len();
            loop {
                match bs.tables[s].ensure(positions[i] as usize, &mut bs.pool) {
                    Ok(()) => break,
                    Err(_) => {
                        // dry pool: shed radix retention (LRU leaves) before
                        // asking the scheduler to preempt. Tier-aware and
                        // cliff-grade (make_room_blocking), with window
                        // blobs DEMOTING here too - the old recycle-only
                        // shortcut ('capturing them would stall the pass')
                        // predates the pipelined lane: demote_aux only
                        // SUBMITS, and discarding the blobs left the tier
                        // restore-blind on the busiest eviction path.
                        let shed = match bs.prefix.as_mut() {
                            Some(pf) => {
                                let exec = self.exec.clone();
                                let state = Some(pf.tier_state_geom(&exec.stream));
                                match pf.tier.as_mut() {
                                    Some(tier) => {
                                        let want = bs.pool.free_blocks() + 1;
                                        tier.make_room_blocking(
                                            &mut pf.radix,
                                            &mut bs.pool,
                                            want,
                                            state,
                                            &mut || exec.record_event().ok(),
                                        )
                                    }
                                    None => pf.radix.evict_lru(&mut bs.pool).is_some(),
                                }
                            }
                            None => false,
                        };
                        if !shed {
                            return Err(GpuModelError::PoolExhausted);
                        }
                    }
                }
            }
            let now = bs.tables[s].blocks().len();
            if now > before {
                grew = true;
                let base = s * bs.bps;
                for j in before..now {
                    bs.bt_host[base + j] = bs.tables[s].blocks()[j];
                }
            }
        }
        if grew {
            self.exec
                .stream
                .memcpy_htod(&bs.bt_host, &mut bs.d_bt)
                .map_err(|e| GpuError::Driver(e.to_string()))?;
        }
        Ok(())
    }

    pub(crate) fn release_inactive_slots_impl(&mut self, occupied: &[bool]) {
        let Some(bs) = self.batch.as_mut() else {
            return;
        };
        let n_slots = bs.tables.len();
        for (s, occ) in occupied.iter().enumerate() {
            if !occ && s < bs.tables.len() && !bs.tables[s].blocks().is_empty() {
                bs.tables[s].clear(&mut bs.pool);
            }
        }
        for (s, occ) in occupied.iter().enumerate() {
            if !occ && s < n_slots {
                self.dflash_clear_slot(s);
            }
        }
    }

    /// Prefill a whole prompt into `slot` (chunked at PF_ROWS) and return the
    /// last token's logits. Fresh sequence: the slot's old pool blocks return
    /// first. Prefix resume lands with.
    pub(crate) fn forward_prefill_impl(
        &mut self,
        slot: usize,
        tokens: &[u32],
    ) -> Result<Vec<f32>, GpuModelError> {
        let n_slots = self.batch.as_ref().expect("batch enabled").n_slots;
        assert!(slot < n_slots, "slot {slot} >= enabled {n_slots}");
        if tokens.is_empty() {
            return Err(GpuModelError::Unsupported("empty prompt".into()));
        }
        if tokens.len() > self.max_ctx {
            return Err(GpuModelError::Unsupported(format!(
                "prompt {} tokens > max_ctx {}",
                tokens.len(),
                self.max_ctx
            )));
        }
        {
            let bs = self.batch.as_mut().expect("batch enabled");
            bs.tables[slot].clear(&mut bs.pool);
        }
        // drafter ring no longer matches (fresh sequence - and on a prefix
        // resume the ring holds the previous occupant's features): coverage
        // rebuilds from the tail rows this prefill walks
        self.dflash_clear_slot(slot);
        // prefix cache: adopt shared full-layer blocks + restore the SWA
        // windows, then re-prefill only the tail [start..)
        let start = self.prefix_resume(slot, tokens)?;
        self.ensure_full_rows(&[slot as u32], &[(tokens.len() - 1) as u32])?;
        let cut = self.prefix_cut(tokens.len(), start);

        let mut base = start;
        let mut last_len = 0usize;
        for chunk in tokens[start..].chunks(pf_rows()) {
            self.prefill_chunk(slot, chunk, base)?;
            base += chunk.len();
            last_len = chunk.len();
        }
        self.prefix_insert(slot, tokens, cut)?;
        self.head_row(last_len - 1, 1)
    }

    /// COALESCED multi-prompt prefill (gemma4's scheme): every pending
    /// prompt's tail rows concatenate into shared PF_ROWS chunks - one
    /// weight-amortized pass over the wave instead of one pass per prompt.
    /// On this MoE that difference is the whole admission story: each pass
    /// re-streams every touched expert's weights, so a 32×128-token burst as
    /// 32 sequential 128-row passes paid the expert traffic 32× - a
    /// multi-second TTFT wall on a 128×128 c32 burst.
    ///
    /// Correctness leans on two standing laws: (1) r-invariance - every
    /// prefill row takes the same kernel rungs and its bytes depend only on
    /// its own row content, so sharing a pass with other prompts changes
    /// nothing (the same law warm-resume tails already rely on); (2) run
    /// isolation - attention launches never mix two slots' query rows
    /// (PfCuts), appends land per-row in each slot's own pool blocks/rings.
    ///
    /// Prefix cache: per-item resume up front, per-item insert after the
    /// pass (rings are per slot - later items' appends never touch this
    /// slot's ring, and only the item's own ≤16-token post-cut tail lands in
    /// its ring before the checkpoint copy). Known tradeoff (gemma4 too):
    /// items inside one wave can't reuse each other's prefixes - the insert
    /// needs written blocks, so a same-wave shared prefix prefills cold.
    pub(crate) fn forward_prefill_batch_impl(
        &mut self,
        items: &[(usize, Vec<u32>)],
    ) -> Result<Vec<Vec<f32>>, GpuModelError> {
        // single item (or the A/B pin): the serial path, provably byte-equal
        if items.len() == 1 || no_coalesced_prefill() {
            return items
                .iter()
                .map(|(slot, toks)| self.forward_prefill_impl(*slot, toks))
                .collect();
        }
        let n_slots = self.batch.as_ref().expect("batch enabled").n_slots;
        // per-item admission prep, exactly the serial prologue: fresh
        // sequence, prefix resume, full-layer block backing
        let mut starts = Vec::with_capacity(items.len());
        for (slot, tokens) in items {
            assert!(*slot < n_slots, "slot {slot} >= enabled {n_slots}");
            if tokens.is_empty() {
                return Err(GpuModelError::Unsupported("empty prompt".into()));
            }
            if tokens.len() > self.max_ctx {
                return Err(GpuModelError::Unsupported(format!(
                    "prompt {} tokens > max_ctx {}",
                    tokens.len(),
                    self.max_ctx
                )));
            }
            {
                let bs = self.batch.as_mut().expect("batch enabled");
                bs.tables[*slot].clear(&mut bs.pool);
            }
            self.dflash_clear_slot(*slot);
            starts.push(self.prefix_resume(*slot, tokens)?);
            self.ensure_full_rows(&[*slot as u32], &[(tokens.len() - 1) as u32])?;
        }
        // the wave's row stream: (slot, pos, token), items contiguous in
        // order; last_row[it] = the row whose logits item `it` needs
        let mut rows: Vec<(u32, u32, u32)> = Vec::new();
        let mut last_row = vec![0usize; items.len()];
        for (it, ((slot, toks), &start)) in items.iter().zip(&starts).enumerate() {
            for (j, &t) in toks[start..].iter().enumerate() {
                rows.push((*slot as u32, (start + j) as u32, t));
            }
            last_row[it] = rows.len() - 1;
        }
        let mut out: Vec<Vec<f32>> = vec![Vec::new(); items.len()];
        let mut base = 0usize;
        for chunk in rows.chunks(pf_rows()) {
            let r = chunk.len();
            // finishers whose last row landed in this chunk read inside the
            // pass (the next chunk's embed overwrites sc.x)
            let fin: Vec<(usize, usize)> = last_row
                .iter()
                .enumerate()
                .filter(|&(_, &lr)| lr >= base && lr < base + r)
                .map(|(it, &lr)| (lr - base, it))
                .collect();
            for (it, logits) in self.prefill_rows_pass(chunk, &fin)? {
                out[it] = logits;
            }
            base += r;
        }
        // per-item radix insert + SWA window checkpoint
        for (it, (slot, toks)) in items.iter().enumerate() {
            let cut = self.prefix_cut(toks.len(), starts[it]);
            self.prefix_insert(*slot, toks, cut)?;
        }
        Ok(out)
    }

    /// Queue a prompt for STALL-FREE chunked prefill (Sarathi-Serve
    /// Algorithm 3). Does the
    /// whole admission prologue now - fresh sequence, prefix resume, block
    /// backing - so a mixed tick only has to move rows.
    pub(crate) fn prefill_begin_impl(
        &mut self,
        slot: usize,
        tokens: Vec<u32>,
    ) -> Result<(), GpuModelError> {
        let n_slots = self.batch.as_ref().expect("batch enabled").n_slots;
        if slot >= n_slots {
            return Err(GpuModelError::Unsupported(format!(
                "slot {slot} >= enabled {n_slots}"
            )));
        }
        if tokens.is_empty() {
            return Err(GpuModelError::Unsupported("empty prompt".into()));
        }
        if tokens.len() > self.max_ctx {
            return Err(GpuModelError::Unsupported(format!(
                "prompt {} tokens > max_ctx {}",
                tokens.len(),
                self.max_ctx
            )));
        }
        // a queued entry for this slot is STALE (the scheduler keeps one
        // chunk per live slot - a duplicate means the old request died and
        // the slot was reused): evict rather than wedge the slot
        self.chunked.retain(|c| c.slot != slot);
        {
            let bs = self.batch.as_mut().expect("batch enabled");
            bs.tables[slot].clear(&mut bs.pool);
        }
        self.dflash_clear_slot(slot);
        let start = self.prefix_resume(slot, &tokens)?;
        self.ensure_full_rows(&[slot as u32], &[(tokens.len() - 1) as u32])?;
        self.chunked.push(ChunkedPrefill {
            slot,
            tokens,
            cursor: start,
            start,
        });
        Ok(())
    }

    /// Drop slot's in-flight prefill (client hung up mid-prompt).
    pub(crate) fn prefill_abort_impl(&mut self, slot: usize) -> bool {
        let n = self.chunked.len();
        self.chunked.retain(|c| c.slot != slot);
        self.chunked.len() != n
    }

    /// One FUSED mixed tick: decode rows and the prefill chunk in a single
    /// pass. This is the shape Sarathi-Serve's kernel contract specifies -
    /// per-sequence (q_len, kv_len) in one batch, decodes at q_len 1 and the
    /// chunk at q_len = rows - and the reason it matters here is bytes: on a
    /// 256-expert MoE any chunk past ~200 rows routes across essentially
    /// every expert, so a pass streams the whole 18.9 GB whatever rides it.
    /// Running prefill and decode as two passes therefore DOUBLES the tick's
    /// weight traffic; measured at -30% on 1024x1024 c8, where admissions are
    /// frequent and the prefill share is small.
    ///
    /// Decode rows give up the CUDA-graph-captured decode path (they ride the
    /// prefill class's kernels here) and buy back half the weight stream.
    /// Ragged multi-slot rows are already a solved problem in this file -
    /// `spec_verify_rows` runs the same pattern, and `PfCuts` guarantees an
    /// attention launch never mixes two slots' query rows.
    ///
    /// Returns (decode logits [dec.len(), vocab] in input order, finished
    /// prefills).
    pub(crate) fn forward_mixed_fused(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
    ) -> Result<(Vec<f32>, Vec<(usize, Vec<f32>, usize)>), GpuModelError> {
        // Nothing queued -> a plain decode tick. Fusing here would drag the
        // COMMON tick off its captured graph (and onto the prefill kernel
        // class) for no prefill work at all.
        if self.chunked.is_empty() {
            if decodes.is_empty() {
                return Ok((Vec::new(), Vec::new()));
            }
            let toks: Vec<u32> = decodes.iter().map(|d| d.1).collect();
            let pos: Vec<u32> = decodes.iter().map(|d| d.2).collect();
            let slots: Vec<u32> = decodes.iter().map(|d| d.0 as u32).collect();
            self.batch_step_slots(&toks, &pos, &slots)?;
            return Ok((self.read_batch_logits(decodes.len())?, Vec::new()));
        }
        let (rows, dec_n, fin, take) = self.fuse_rows(decodes, budget);
        self.ensure_decode_rows(decodes)?;
        self.rows_pass_body(&rows, dec_n)?;
        // Decode logits first: one bulk norm+head over rows 0..dec_n. It has
        // to precede the finisher heads - head_row bounces its row through
        // x[0] and rewrites head_logits[0..vocab], which is decode row 0's.
        let mut dec_logits = Vec::new();
        if dec_n > 0 {
            self.head_rows(dec_n)?;
            dec_logits = self.read_batch_logits(dec_n)?;
        }
        let mut finished_raw = Vec::with_capacity(fin.len());
        for &(row, qi) in &fin {
            finished_raw.push((qi, self.head_row(row, 1)?));
        }
        let finished = self.commit_chunk(&take, finished_raw)?;
        Ok((dec_logits, finished))
    }

    /// The fused mixed tick with device sampling for the decode rows. Same
    /// shape as `forward_mixed_fused`; the difference is that decode logits
    /// never leave the GPU - one bulk head over rows 0..dec_n feeds
    /// `sample_rows`, and only Host-plan rows pay a vocab-row readback.
    ///
    /// This is the variant the scheduler actually takes whenever device
    /// sampling is on (it is), so the fused pass has to live here or the
    /// two-pass double-weight-stream tax stays no matter what
    /// `forward_mixed` does.
    ///
    /// Finishers still come back as LOGITS. Device-sampling them (fin_plans)
    /// would save one [1, vocab] readback per COMPLETED prompt, where decode
    /// rows read back every tick - not where the time is. Declining is
    /// contract-legal; revisit with a measured reason.
    pub(crate) fn forward_mixed_fused_sampled(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
        plans: &[crate::generator::RowSample],
    ) -> Result<(crate::generator::SampledStep, Vec<(usize, Vec<f32>, usize)>), GpuModelError> {
        use crate::generator::{RowSample, SampledStep};
        assert_eq!(plans.len(), decodes.len(), "one plan per decode row");
        if self.chunked.is_empty() {
            let step = if decodes.is_empty() {
                SampledStep {
                    ids: Vec::new(),
                    host_rows: Vec::new(),
                }
            } else {
                let toks: Vec<u32> = decodes.iter().map(|d| d.1).collect();
                let pos: Vec<u32> = decodes.iter().map(|d| d.2).collect();
                let slots: Vec<u32> = decodes.iter().map(|d| d.0 as u32).collect();
                self.forward_batch_sampled_slots(&toks, &pos, Some(&slots), plans)?
            };
            return Ok((step, Vec::new()));
        }
        let exec = self.exec.clone();
        let vocab = self.hp.n_vocab;
        let (rows, dec_n, fin, take) = self.fuse_rows(decodes, budget);
        self.ensure_decode_rows(decodes)?;
        let mut mixed_trunc = false;
        if dec_n > 0 {
            let (par, tpar) = Self::pack_samp_par(plans);
            let sc = &mut self.batch.as_mut().expect("batch enabled").sc;
            let mut v = sc
                .d_par
                .try_slice_mut(0..dec_n * 4)
                .ok_or_else(|| GpuError::Driver("d_par slice".into()))?;
            exec.stream.memcpy_htod(&par, &mut v).map_err(drv)?;
            if let Some(t) = &tpar {
                mixed_trunc = true;
                let mut v = sc
                    .d_tpar
                    .try_slice_mut(0..dec_n * 4)
                    .ok_or_else(|| GpuError::Driver("d_tpar slice".into()))?;
                exec.stream.memcpy_htod(t, &mut v).map_err(drv)?;
            }
        }
        self.rows_pass_body(&rows, dec_n)?;
        let mut step = SampledStep {
            ids: Vec::new(),
            host_rows: Vec::new(),
        };
        if dec_n > 0 {
            // Head + sample the decode rows before any finisher head runs -
            // head_row clobbers x[0] and head_logits[0..vocab]. Stream order
            // is the guarantee: every read below is enqueued first.
            self.head_rows(dec_n)?;
            {
                let sc = &mut self.batch.as_mut().expect("batch enabled").sc;
                exec.sample_rows_at(
                    &sc.head_logits,
                    &sc.d_par,
                    0,
                    &mut sc.d_out,
                    0,
                    dec_n,
                    vocab,
                )?;
                if mixed_trunc {
                    Self::trunc_dev5_witness(dec_n);
                    exec.sample_rows_t_at(
                        &sc.head_logits,
                        &sc.d_par,
                        0,
                        &sc.d_tpar,
                        0,
                        &mut sc.d_out,
                        0,
                        dec_n,
                        vocab,
                    )?;
                    exec.sample_rows_p_at(
                        &sc.head_logits,
                        &sc.d_par,
                        0,
                        &sc.d_tpar,
                        0,
                        &mut sc.d_out,
                        0,
                        dec_n,
                        vocab,
                    )?;
                }
            }
            let sc = &self.batch.as_ref().expect("batch enabled").sc;
            let ids_view = sc
                .d_out
                .try_slice(0..dec_n)
                .ok_or_else(|| GpuError::Driver("d_out slice".into()))?;
            let ids = exec.stream.clone_dtoh(&ids_view).map_err(drv)?;
            let mut host_rows = Vec::new();
            for (i, p) in plans.iter().enumerate() {
                if matches!(p, RowSample::Host) {
                    let v = sc
                        .head_logits
                        .try_slice(i * vocab..(i + 1) * vocab)
                        .ok_or_else(|| GpuError::Driver("host row slice".into()))?;
                    host_rows.push((i, exec.stream.clone_dtoh(&v).map_err(drv)?));
                }
            }
            step = SampledStep { ids, host_rows };
        }
        let mut finished_raw = Vec::with_capacity(fin.len());
        for &(row, qi) in &fin {
            finished_raw.push((qi, self.head_row(row, 1)?));
        }
        let finished = self.commit_chunk(&take, finished_raw)?;
        Ok((step, finished))
    }

    /// Build the fused tick's row stream: DECODE rows (q_len 1, at their own
    /// positions) first, then this tick's prefill chunk. Decodes lead so
    /// their residuals land at rows 0..dec_n and one bulk head covers them;
    /// it also keeps the finisher rows ascending, which `head_row`'s x[0]
    /// bounce requires. Returns (rows, dec_n, finishers as (row, queue
    /// index), per-entry take).
    #[allow(clippy::type_complexity)]
    fn fuse_rows(
        &self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
    ) -> (
        Vec<(u32, u32, u32)>,
        usize,
        Vec<(usize, usize)>,
        Vec<(usize, usize, bool)>,
    ) {
        let mut rows: Vec<(u32, u32, u32)> =
            decodes.iter().map(|&(s, t, p)| (s as u32, p, t)).collect();
        let dec_n = rows.len();
        // The decode rows and the chunk share one scratch plane, so the tick
        // is bounded by its ROW CAPACITY (BatchState::cap = pf_rows() + one
        // row per slot), not by pf_rows(). Sized that way the band never eats
        // chunk rows; bounding at pf_rows() instead blew `d_toks` on the
        // first fused tick with a live decoder, and capping the chunk to
        // compensate cost 3.2% on 2048×128 c32.
        let room = self
            .batch
            .as_ref()
            .expect("batch enabled")
            .cap
            .saturating_sub(dec_n);
        let (chunk_rows, take) = if room == 0 {
            (Vec::new(), Vec::new())
        } else {
            self.plan_chunk(budget.min(room))
        };
        rows.extend_from_slice(&chunk_rows);
        let mut fin: Vec<(usize, usize)> = Vec::new();
        let mut off = dec_n;
        for &(qi, n, done) in &take {
            if done {
                fin.push((off + n - 1, qi));
            }
            off += n;
        }
        (rows, dec_n, fin, take)
    }

    /// Back the full-layer KV blocks this tick's decode rows will write.
    /// Prefill rows got theirs up front at `prefill_begin_impl`.
    fn ensure_decode_rows(&mut self, decodes: &[(usize, u32, u32)]) -> Result<(), GpuModelError> {
        if decodes.is_empty() {
            return Ok(());
        }
        let slots: Vec<u32> = decodes.iter().map(|d| d.0 as u32).collect();
        let pos: Vec<u32> = decodes.iter().map(|d| d.2).collect();
        self.ensure_full_rows(&slots, &pos)
    }

    /// Pick this tick's chunk rows: FIFO over the queue, up to `budget` rows,
    /// splitting the last prompt if it does not fit. Returns the row stream
    /// and (queue index, rows taken, finishes?) per touched entry.
    fn plan_chunk(&self, budget: usize) -> (Vec<(u32, u32, u32)>, Vec<(usize, usize, bool)>) {
        let mut rows: Vec<(u32, u32, u32)> = Vec::new();
        let mut take: Vec<(usize, usize, bool)> = Vec::new();
        if self.chunked.is_empty() {
            return (rows, take);
        }
        let cap = budget.clamp(1, pf_rows());
        for (qi, c) in self.chunked.iter().enumerate() {
            if rows.len() >= cap {
                break;
            }
            let remaining = c.tokens.len() - c.cursor;
            let n = remaining.min(cap - rows.len()).max(1);
            for j in 0..n {
                let p = c.cursor + j;
                rows.push((c.slot as u32, p as u32, c.tokens[p]));
            }
            take.push((qi, n, n == remaining));
        }
        (rows, take)
    }

    /// Advance cursors, publish finished prompts into the radix tree, drop
    /// them from the queue.
    fn commit_chunk(
        &mut self,
        take: &[(usize, usize, bool)],
        finished_raw: Vec<(usize, Vec<f32>)>,
    ) -> Result<Vec<(usize, Vec<f32>, usize)>, GpuModelError> {
        for &(qi, n, _) in take {
            self.chunked[qi].cursor += n;
        }
        let mut out = Vec::new();
        for (qi, logits) in finished_raw {
            let c = &self.chunked[qi];
            let (slot, cut) = (c.slot, self.prefix_cut(c.tokens.len(), c.start));
            let toks = std::mem::take(&mut self.chunked[qi].tokens);
            self.prefix_insert(slot, &toks, cut)?;
            out.push((slot, logits, toks.len()));
        }
        self.chunked.retain(|c| !c.tokens.is_empty());
        Ok(out)
    }

    /// One weight-amortized pass over a ready-made row stream - the shared
    /// body of every prefill lane (whole-prompt, coalesced wave, and the
    /// stall-free mixed tick). `chunk` is (slot, pos, token) with items
    /// contiguous; `fin` names the rows whose logits the caller wants, as
    /// (chunk-local row, caller tag), and must be ascending by row: head_row
    /// bounces its row through x[0], so a row-0 reader has to go first or a
    /// later read clobbers it.
    ///
    /// Rows may start at any position in their slot - a prefix resume and a
    /// mid-prompt chunk resume are the same thing to this pass, which is what
    /// makes stall-free batching possible without a second code path.
    fn prefill_rows_pass(
        &mut self,
        chunk: &[(u32, u32, u32)],
        fin: &[(usize, usize)],
    ) -> Result<Vec<(usize, Vec<f32>)>, GpuModelError> {
        self.rows_pass_body(chunk, 0)?;
        let mut out = Vec::with_capacity(fin.len());
        for &(row, tag) in fin {
            out.push((tag, self.head_row(row, 1)?));
        }
        Ok(out)
    }

    /// The pass itself, head excluded: stage the rows, embed, walk every
    /// layer under the same-slot cuts, keep the drafter warm. Split out of
    /// `prefill_rows_pass` because the fused mixed tick heads its decode rows
    /// in BULK (and device-samples them) before the finisher rows go through
    /// `head_row` one at a time.
    fn rows_pass_body(
        &mut self,
        chunk: &[(u32, u32, u32)],
        dec: usize,
    ) -> Result<(), GpuModelError> {
        let r = chunk.len();
        let toks: Vec<u32> = chunk.iter().map(|x| x.2).collect();
        let positions: Vec<u32> = chunk.iter().map(|x| x.1).collect();
        let slots_v: Vec<u32> = chunk.iter().map(|x| x.0).collect();
        // contiguous same-slot runs over the PREFILL rows (PfCuts derives the
        // SWA ladder cuts from them) - an attention launch never mixes two
        // slots' query rows. The leading `dec` rows are a fused tick's decode
        // band and are not run-split: they attend as one decode-class launch.
        let mut runs: Vec<(usize, usize)> = Vec::new();
        for (i, x) in chunk.iter().enumerate().skip(dec) {
            match runs.last_mut() {
                Some((off, n)) if chunk[*off].0 == x.0 => *n += 1,
                _ => runs.push((i, 1)),
            }
        }
        self.upload_rows(&toks, &positions, &slots_v)?;
        self.embed_rows(r)?;
        self.layer_walk(r, Some(&PfCuts::fused(dec, runs)))?;
        // drafter warmth for every row in this chunk (before the finishers -
        // head_row bounces residual rows through sc.x/proj, but the append
        // only reads the aux bands + d_pos/d_slots)
        if self.dflash_armed() {
            self.dflash_append_features(&positions, &slots_v, None)?;
        }
        Ok(())
    }

    /// Ragged multi-slot spec VERIFY (stage A): every chunk row
    /// (committed pending + drafts, per slot at its live positions) rides one
    /// batched pass through the PREFILL class, leaving [rows, vocab] logits
    /// in head_logits. Appends extend the live KV at pos..pos+len-1 -
    /// contamination-safe under the paged/ring overwrite discipline: rejected
    /// rows' entries sit BEYOND the committed point, dead until the next
    /// round's appends rewrite those positions before any read (the qwen35
    /// paged-verify argument, hybrid-KV flavored: pool blocks are
    /// position-indexed, rings are position-mod-ring - both rewrite in
    /// place). Class note: verify rows ride pf_quant/pf_mm + the WMMA span
    /// attention, a different numeric class than the decode tick - fine,
    /// because the picks this pass emits are the served tokens (the lane is
    /// self-consistent); spec-on near-tie flips vs the nospec refs are
    /// expected and get their own llama-greedy vetting before default-on.
    /// Ok(None) = decline this round (above cap / ctx-full / pool pressure);
    /// the service cools down or falls back to the dense tick.
    fn spec_verify_rows(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
    ) -> Result<Option<usize>, GpuModelError> {
        let total: usize = reqs.iter().map(|q| q.2.len()).sum();
        if total == 0 || total > SPEC_ROWS {
            return Ok(None);
        }
        let n_slots = self.batch.as_ref().expect("batch enabled").n_slots;
        let mut toks = Vec::with_capacity(total);
        let mut positions = Vec::with_capacity(total);
        let mut slots_v = Vec::with_capacity(total);
        let mut runs: Vec<(usize, usize)> = Vec::with_capacity(reqs.len());
        for (slot, start, chunk) in reqs {
            assert!(*slot < n_slots, "spec slot {slot} >= enabled {n_slots}");
            if chunk.is_empty() || start + chunk.len() > self.max_ctx {
                return Ok(None);
            }
            // back the span's full-layer blocks; pool pressure declines the
            // round - the dense path owns the preemption machinery
            match self.ensure_full_rows(&[*slot as u32], &[(start + chunk.len() - 1) as u32]) {
                Ok(()) => {}
                Err(GpuModelError::PoolExhausted) => return Ok(None),
                Err(e) => return Err(e),
            }
            runs.push((toks.len(), chunk.len()));
            for (j, &t) in chunk.iter().enumerate() {
                toks.push(t);
                positions.push((*start + j) as u32);
                slots_v.push(*slot as u32);
            }
        }
        self.upload_rows(&toks, &positions, &slots_v)?;
        // One captured replay per chunk-length signature instead of ~600
        // eager launches (embed + 40-layer walk + head). Everything the body
        // reads per row is staged device data; the cut offsets/lengths bake
        // into the graph, hence the signature key. First sight runs eagerly
        // (serves the round + warms every kernel path), then records the
        // identical launch stream. PADDOCK_SPEC_NOGRAPH pins eager A/B.
        let key: Vec<u32> = runs.iter().map(|&(_, l)| l as u32).collect();
        let bs = self.batch.as_ref().expect("batch enabled");
        let have = bs.spec_graphs.contains_key(&key);
        let full = bs.spec_graphs.len() >= SPEC_GRAPH_CAP;
        if paddock_models::dev_var_os!("PADDOCK_SPEC_NOGRAPH").is_some() || (!have && full) {
            self.spec_verify_body(total, &runs)?;
        } else if !have {
            self.spec_verify_body(total, &runs)?;
            let g = self.capture_body(|s| s.spec_verify_body(total, &runs), "spec verify")?;
            self.batch
                .as_mut()
                .expect("batch enabled")
                .spec_graphs
                .insert(key, g);
        } else {
            self.batch.as_ref().expect("batch enabled").spec_graphs[&key]
                .0
                .launch()
                .map_err(|e| GpuError::Driver(format!("spec verify graph launch: {e}")))?;
        }
        Ok(Some(total))
    }

    /// The spec-verify device body (capture-safe): embed the staged rows,
    /// walk all layers in the spec cut flavor (decode numeric class, ragged
    /// appends/attends - and the DFlash aux taps while armed), head over
    /// every row. Shapes depend only on the run lengths.
    fn spec_verify_body(
        &mut self,
        total: usize,
        runs: &[(usize, usize)],
    ) -> Result<(), GpuModelError> {
        self.embed_rows(total)?;
        self.layer_walk(total, Some(&PfCuts::spec(runs.to_vec())))?;
        self.head_rows(total)
    }

    /// Greedy spec round: verify + device argmax per row (ties -> lowest
    /// index, the host scan's rule). Gated behind PADDOCK_LAGUNA_SPEC until
    /// the DFlash drafter lands (see laguna_spec_on).
    pub(crate) fn forward_spec_batch_impl(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
    ) -> Result<Option<Vec<u32>>, GpuModelError> {
        if !laguna_spec_on() || self.batch.is_none() || !self.exec.has_argmax_rows() {
            return Ok(None);
        }
        let Some(r) = self.spec_verify_rows(reqs)? else {
            return Ok(None);
        };
        let exec = self.exec.clone();
        let vocab = self.hp.n_vocab;
        let picks = {
            let sc = &mut self.batch.as_mut().expect("batch enabled").sc;
            exec.argmax_rows(&sc.head_logits, &mut sc.d_spec_out, r, vocab)?;
            let v = sc
                .d_spec_out
                .try_slice(0..r)
                .ok_or_else(|| GpuError::Driver("d_spec_out slice".into()))?;
            exec.stream.clone_dtoh(&v).map_err(drv)?
        };
        // DFlash: ring-append the accepted rows' features (the walk above
        // captured every verify row's aux residuals; commit replays the
        // service's accept rule on these picks)
        if self.dflash_armed() {
            self.dflash_spec_commit(reqs, &picks)?;
        }
        Ok(Some(picks))
    }

    /// Device-plan packing for the sampled spec round: (inv_t, u, mode, pad)
    /// per row, the sample_rows layout. RsVerify rows need drafter
    /// q-probabilities this lane doesn't carry yet - decline (None).
    /// TruncCat verify rows pack mode 5 + the tpar side plane -
    /// sampled verify + accept-while-match stays exact for them.
    fn pack_spec_par(
        plans: &[crate::sampler::DevicePlan],
        dev_trunc: bool,
    ) -> Option<(Vec<u32>, Option<Vec<u32>>)> {
        use crate::sampler::DevicePlan;
        let mut par = vec![0u32; plans.len() * 4];
        let mut tpar = vec![0u32; plans.len() * 4];
        let mut any_trunc = false;
        for (i, p) in plans.iter().enumerate() {
            match p {
                DevicePlan::Greedy => par[i * 4 + 2] = 1,
                DevicePlan::Categorical { inv_t, u } => {
                    par[i * 4] = inv_t.to_bits();
                    par[i * 4 + 1] = u.to_bits();
                    par[i * 4 + 2] = 2;
                }
                DevicePlan::TruncCat {
                    inv_t,
                    u,
                    k,
                    top_p,
                    min_p,
                } if dev_trunc => {
                    par[i * 4] = inv_t.to_bits();
                    par[i * 4 + 1] = u.to_bits();
                    par[i * 4 + 2] = if *k >= 1 && *k <= 64 { 5 } else { 6 };
                    tpar[i * 4] = *k;
                    tpar[i * 4 + 1] = top_p.to_bits();
                    tpar[i * 4 + 2] = min_p.to_bits();
                    any_trunc = true;
                }
                _ => return None,
            }
        }
        Some((par, any_trunc.then_some(tpar)))
    }

    /// DEVICE-SAMPLED spec round (the temperature-only serving hot path):
    /// verify + sample_rows with the pre-drawn per-row plans - no logits
    /// readback. Same gate + decline semantics as the greedy round.
    pub(crate) fn forward_spec_batch_plans_impl(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
        plans: &[crate::sampler::DevicePlan],
    ) -> Result<Option<Vec<u32>>, GpuModelError> {
        if !laguna_spec_on() || self.batch.is_none() || !self.exec.has_sample_rows() {
            return Ok(None);
        }
        let total: usize = reqs.iter().map(|q| q.2.len()).sum();
        if plans.len() != total || total == 0 || total > SPEC_ROWS {
            return Ok(None);
        }
        let Some((par, tpar)) = Self::pack_spec_par(plans, self.device_trunc_supported()) else {
            return Ok(None);
        };
        let exec = self.exec.clone();
        {
            let sc = &mut self.batch.as_mut().expect("batch enabled").sc;
            let mut v = sc
                .d_spec_par
                .try_slice_mut(0..total * 4)
                .ok_or_else(|| GpuError::Driver("d_spec_par slice".into()))?;
            exec.stream.memcpy_htod(&par, &mut v).map_err(drv)?;
            if let Some(t) = &tpar {
                let mut v = sc
                    .d_spec_tpar
                    .try_slice_mut(0..total * 4)
                    .ok_or_else(|| GpuError::Driver("d_spec_tpar slice".into()))?;
                exec.stream.memcpy_htod(t, &mut v).map_err(drv)?;
            }
        }
        let Some(r) = self.spec_verify_rows(reqs)? else {
            return Ok(None);
        };
        let vocab = self.hp.n_vocab;
        let picks = {
            let sc = &mut self.batch.as_mut().expect("batch enabled").sc;
            exec.sample_rows_at(
                &sc.head_logits,
                &sc.d_spec_par,
                0,
                &mut sc.d_spec_out,
                0,
                r,
                vocab,
            )?;
            if tpar.is_some() {
                Self::trunc_dev5_witness(r);
                exec.sample_rows_t_at(
                    &sc.head_logits,
                    &sc.d_spec_par,
                    0,
                    &sc.d_spec_tpar,
                    0,
                    &mut sc.d_spec_out,
                    0,
                    r,
                    vocab,
                )?;
                exec.sample_rows_p_at(
                    &sc.head_logits,
                    &sc.d_spec_par,
                    0,
                    &sc.d_spec_tpar,
                    0,
                    &mut sc.d_spec_out,
                    0,
                    r,
                    vocab,
                )?;
            }
            let v = sc
                .d_spec_out
                .try_slice(0..r)
                .ok_or_else(|| GpuError::Driver("d_spec_out slice".into()))?;
            exec.stream.clone_dtoh(&v).map_err(drv)?
        };
        // same accept-walk replay as the greedy round: the service commits
        // off these exact picks, so the internal walk matches it bit-for-bit
        if self.dflash_armed() {
            self.dflash_spec_commit(reqs, &picks)?;
        }
        Ok(Some(picks))
    }

    /// One PF_ROWS chunk of a single slot's prompt: upload row streams, embed,
    /// run the layer walk in prefill mode (one run - text rows are causal,
    /// any SWA_SPAN cut is safe).
    fn prefill_chunk(
        &mut self,
        slot: usize,
        chunk: &[u32],
        base: usize,
    ) -> Result<(), GpuModelError> {
        let r = chunk.len();
        let positions: Vec<u32> = (base as u32..(base + r) as u32).collect();
        let slots_v = vec![slot as u32; r];
        self.upload_rows(chunk, &positions, &slots_v)?;
        self.embed_rows(r)?;
        self.layer_walk(r, Some(&PfCuts::new(vec![(0, r)])))?;
        // drafter warmth: the walk captured this chunk's aux residuals;
        // fuse + ring-append them while d_pos/d_slots still hold the rows
        if self.dflash_armed() {
            self.dflash_append_features(&positions, &slots_v, None)?;
        }
        Ok(())
    }

    /// One batched decode step over rows 0..r (row i drives slot i - the
    /// engine's identity contract). Leaves [r, vocab] logits in head_logits.
    pub(crate) fn batch_step(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
    ) -> Result<(), GpuModelError> {
        let r = tokens.len();
        let ident: Vec<u32> = (0..r as u32).collect();
        self.batch_step_slots(tokens, positions, &ident)
    }

    /// `batch_step` with explicit slot ids - the mixed tick's decode half.
    /// `batch_step`'s identity mapping (row i = slot i) only holds when the
    /// live set is a dense prefix; under stall-free batching the decoders are
    /// whichever slots finished prefilling, so the rows must be compacted and
    /// carry their real slot.
    pub(crate) fn batch_step_slots(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        slots: &[u32],
    ) -> Result<(), GpuModelError> {
        let r = tokens.len();
        assert_eq!(r, positions.len());
        assert_eq!(r, slots.len());
        let n_slots = self.batch.as_ref().expect("batch enabled").n_slots;
        assert!(r <= n_slots, "rows {r} > enabled {n_slots}");
        let ident = slots;
        self.ensure_full_rows(ident, positions)?;
        self.upload_rows(tokens, positions, ident)?;
        self.step_replay(r)?;
        // drafter warmth: dense ticks must keep appending features or every
        // interlude between spec rounds would strand the watermark (a cold
        // slot has no re-warm path). The captured graph wrote the aux bands;
        // the fuse+append launches ride eagerly behind the replay.
        if self.dflash_armed() {
            self.dflash_append_features(positions, ident, None)?;
        }
        Ok(())
    }

    /// The pure-device decode tick body - everything the per-r graph
    /// captures. All inputs are device buffers written before replay
    /// (d_toks/d_pos/d_slots/d_mrope + the block tables); all shapes depend
    /// only on r and model constants.
    fn step_body(&mut self, r: usize) -> Result<(), GpuModelError> {
        self.embed_rows(r)?;
        self.layer_walk(r, None)?;
        self.head_rows(r)
    }

    /// Record `body`'s launches into a CUDA graph (recording only - nothing
    /// executes). gemma4/qwen35 scheme via end_capture_no_flags. The spec
    /// callers run the body EAGERLY once before capturing, so any lazy init
    /// (cuBLAS workspace allocation, first-touch module work) lands outside
    /// the recording - an alloc during capture is a hard driver error.
    pub(crate) fn capture_body(
        &mut self,
        body: impl FnOnce(&mut Self) -> Result<(), GpuModelError>,
        what: &str,
    ) -> Result<super::SendGraph, GpuModelError> {
        let exec = self.exec.clone();
        exec.stream
            .synchronize()
            .map_err(|e| GpuError::Driver(format!("{what} pre-capture sync: {e}")))?;
        exec.stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|e| GpuError::Driver(format!("{what} begin_capture: {e}")))?;
        let rec = body(self);
        let graph = crate::gpu::end_capture_no_flags(&exec.stream)
            .map_err(|e| GpuError::Driver(format!("{what} end_capture: {e}")));
        rec?; // surface a record failure only after capture is cleanly ended
        let graph =
            graph?.ok_or_else(|| GpuError::Driver(format!("{what} capture produced no graph")))?;
        Ok(super::SendGraph(graph))
    }

    /// Capture the fixed-r decode tick on first sight (recording only - no
    /// execution), cache it.
    fn ensure_step_graph(&mut self, r: usize) -> Result<(), GpuModelError> {
        if self
            .batch
            .as_ref()
            .expect("batch enabled")
            .graphs
            .contains_key(&r)
        {
            return Ok(());
        }
        let g = self.capture_body(|s| s.step_body(r), "decode")?;
        self.batch
            .as_mut()
            .expect("batch enabled")
            .graphs
            .insert(r, g);
        Ok(())
    }

    /// Replay the fixed-r decode tick (capturing it first if unseen).
    fn step_replay(&mut self, r: usize) -> Result<(), GpuModelError> {
        self.ensure_step_graph(r)?;
        self.batch.as_ref().expect("batch enabled").graphs[&r]
            .0
            .launch()
            .map_err(|e| GpuError::Driver(format!("decode graph launch: {e}")))?;
        Ok(())
    }

    /// Host->device row streams: tokens, positions, slots, and the axis-major
    /// [4, r] mrope plane (all-equal text axes).
    pub(crate) fn upload_rows(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        slots: &[u32],
    ) -> Result<(), GpuModelError> {
        let r = tokens.len();
        let bs = self.batch.as_mut().expect("batch enabled");
        let sc = &mut bs.sc;
        let mut mrope = Vec::with_capacity(4 * r);
        for _ in 0..4 {
            mrope.extend_from_slice(positions);
        }
        let st = &self.exec.stream;
        let mut t = sc
            .d_toks
            .try_slice_mut(0..r)
            .ok_or_else(|| GpuError::Driver("d_toks".into()))?;
        st.memcpy_htod(tokens, &mut t).map_err(drv)?;
        let mut p = sc
            .d_pos
            .try_slice_mut(0..r)
            .ok_or_else(|| GpuError::Driver("d_pos".into()))?;
        st.memcpy_htod(positions, &mut p).map_err(drv)?;
        let mut s = sc
            .d_slots
            .try_slice_mut(0..r)
            .ok_or_else(|| GpuError::Driver("d_slots".into()))?;
        st.memcpy_htod(slots, &mut s).map_err(drv)?;
        let mut m = sc
            .d_mrope
            .try_slice_mut(0..4 * r)
            .ok_or_else(|| GpuError::Driver("d_mrope".into()))?;
        st.memcpy_htod(&mrope, &mut m).map_err(drv)?;
        Ok(())
    }

    pub(crate) fn embed_rows(&mut self, r: usize) -> Result<(), GpuModelError> {
        let bs = self.batch.as_mut().expect("batch enabled");
        let sc = &mut bs.sc;
        match &self.tok_embd {
            TokEmbd::Q8(t) => {
                self.exec
                    .embed_gather_batch_q8(t, &sc.d_toks, &mut sc.x, self.hp.n_embd, r)?
            }
            TokEmbd::Kq(t) => {
                self.exec
                    .kquant_gather(t, &sc.d_toks, &mut sc.x, self.hp.n_embd, r)?
            }
        }
        Ok(())
    }

    /// The 40-layer walk over r rows. `cuts`: Some = prefill mode (SWA
    /// layers append+attend in the `swa` sub-spans; full layers append
    /// whole-chunk and attend per same-slot `run`); None = decode mode
    /// (every row is one new token of its slot).
    fn layer_walk(&mut self, r: usize, cuts: Option<&PfCuts>) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let hp = &self.hp;
        let (embd, n_kv, hd) = (hp.n_embd, hp.n_kv_heads, hp.head_dim);
        let kv_dim = n_kv * hd;
        let eps = hp.eps;
        let scale = 1.0 / (hd as f32).sqrt();
        let sections = [hp.n_rot as u32 / 2, 0, 0, 0];
        let m = hp.moe;
        let kv_dtype = self.kv_dtype;
        let bs = self.batch.as_mut().expect("batch enabled");
        let (bps, ring) = (bs.bps, bs.ring);
        let anyq8 = bs.any_q8;

        // r==1 rung election (llama's own mmvq/mmq split): a single decode row
        // rides the serial lane's exact-f32 GEMV kernels - measurably faster
        // at r=1 on XS-2.1; the W4A8 dp4a rungs only win once r > 1. Same
        // graph machinery either way; the captured body just records GEMVs.
        // Prefill rows ride the flat-mmq W4A8 int8 tensor-core class (llama's
        // mmq split): the strided dp4a rungs re-read each weight per call and
        // were 53% of the prefill pass (8 calls × ~1 ms/layer/chunk at ~10
        // effective TOPS). Keyed on MODE, never on r: a warm-
        // resume tail must reproduce the cold chunk's bytes exactly, so every
        // prefill row - 1024-row chunk or 3-row tail - takes the same rungs
        // (pf_quant/pf_mm). Decode keeps the strided ladder and the captured
        // graph body untouched; r1 GEMVs are decode-only for the same reason.
        // spec-verify rounds keep the DECODE numeric class (see PfCuts::spec)
        let spec = cuts.is_some_and(|c| c.spec);
        let pf = cuts.is_some() && !spec && !no_w4a8_prefill();
        let r1 = r == 1 && !pf;
        // r1 k-quant projections ride the W4A8 dp4a GEMV (qwen35's
        // kq_w4a8_b1 serving class - llama mmvq's own activation-quant
        // numeric class). A production tick showed the exact-f32 gemv
        // issue-bound: in-2048 shapes ~11.3 us/instance, the v/shexp small
        // shapes at ~190 GB/s. Activations
        // quantize once per shared input (qwen35's dedupe pattern).
        // PADDOCK_KQ_EXACT_GEMV=1 pins the exact-f32 oracle GEMV; the serial
        // forward_one lane keeps it unconditionally.
        let g8 = r1
            && exec.has_kquant_gemv_w4a8()
            && paddock_models::dev_var_os!("PADDOCK_KQ_EXACT_GEMV").is_none();
        // WMMA (tensor-core) prefill attention - the scalar tiled kernel was
        // 46% of the post-w4a8 prefill pass (296 ms / 1900 tok, ~2.7 TF
        // effective). One class for every prefill span, any len: the old
        // len>24 prefill/decode-kernel switch was another span-length class
        // seam (same warm==cold byte-invariance argument as pf_quant).
        // hd==128 gate only - the dtype/ratio check is per-layer below since
        // laguna alternates G=6 (full-attn, nh=48) / G=9 (SWA, nh=72) and the
        // new v4 hd128 arm is ratio-specific (unlike the old generic WMMA
        // tile, which took any ratio, hence this was a single flag before).
        let wmma_pf_base = cuts.is_some()
            && !spec
            && hd == 128
            && exec.has_attn_prefill_f16_paged()
            && !no_wmma_prefill();

        // DFlash aux taps: while armed, the post-layer residuals of the
        // drafter's target_layer_ids copy into its fusion bands (band order =
        // ids order; rows 0..r). Plain stream-ordered dtod copies - sizes
        // depend only on r, so captured decode graphs record them cleanly.
        // The walk's caller fuses+appends afterwards (dflash_append_features
        // / dflash_spec_commit). None when the state isn't armed: zero
        // launches, byte-identical serving.
        // aux-band row stride: the same scratch row capacity every plane is
        // sized at, so a fused tick's rows can never run off a band
        let cap = bs.cap;
        let mut dtap = self.dflash.as_mut().and_then(|d| {
            let ids = d.target_layer_ids.clone();
            d.state.as_mut().map(|st| (&mut st.aux, ids))
        });

        for (li, layer) in self.layers.iter().enumerate() {
            let nh = layer.n_heads;
            let wmma_pf = wmma_pf_base && pf_attn_dtype_ok(kv_dtype, nh, n_kv);
            let q_dim = nh * hd;
            let sc = &mut bs.sc;
            exec.rmsnorm_batch(&sc.x, &layer.attn_norm.buf, &mut sc.xn, embd, eps, r)?;
            // r==1 with a fused plane: one launch lands [q | k | gate] in
            // sc.q (row 0 spans past q_dim - safe, rows 1.. are unused at
            // r==1); the k-norm and softplus consumers read at offsets.
            let qk_fused = r1 && layer.qkg.is_some();
            if g8 {
                // one quantize serves qkg (or q/k/v/g) - all read xn. SKIPPED
                // when the whole band is Q8_0: the exact Q8 GEMV reads f32 x
                // directly, so on an all-Q8_0 file (the S-2.1 UD: every dense
                // plane is Q8_0) this staged int8 nobody read - a dead launch
                // per band.
                let band_kq = layer.qkg.is_some()
                    || matches!(layer.wq, QuantW::Kq(_))
                    || matches!(layer.wk, QuantW::Kq(_))
                    || matches!(layer.wv, QuantW::Kq(_))
                    || matches!(layer.g_proj, QuantW::Kq(_));
                if band_kq {
                    exec.quantize_q8_sums(&sc.xn, &mut sc.xq, &mut sc.xs, &mut sc.ssums, embd)?;
                }
            }
            if qk_fused {
                let qkg = layer.qkg.as_ref().expect("checked");
                if g8 {
                    let needs = crate::gpu::kq_needs_sums(qkg.ty);
                    exec.kquant_gemv_w4a8(
                        qkg,
                        &sc.xq,
                        &sc.xs,
                        needs.then_some(&sc.ssums),
                        &mut sc.q,
                    )?;
                    gemv8_any(
                        &exec, &layer.wv, &sc.xn, &sc.xq, &sc.xs, &sc.ssums, &mut sc.v,
                    )?;
                } else {
                    exec.kquant_gemv(qkg, &sc.xn, &mut sc.q)?;
                    gemv_any(&exec, &layer.wv, &sc.xn, &mut sc.v)?;
                }
            } else if g8 {
                // (q|k) + (v|g): the Q8_0 multi (entry 317) folds the band's
                // four same-input planes into two launches, no solo small
                // plane left - the split band ran 8 Q8_0 GEMVs/layer at a
                // ~6.7us median (launch floor), 43.5% of decode GPU time.
                // Bit-identical per row to the splits.
                match (&layer.wq, &layer.wk, &layer.wv, &layer.g_proj) {
                    (QuantW::Q8(wq), QuantW::Q8(wk), QuantW::Q8(wv), QuantW::Q8(wg))
                        if exec.has_q8_0_gemv_repacked_multi() && !no_gemv_multi() =>
                    {
                        exec.q8_0_gemv_repacked_multi(
                            &mut [(wq, &mut sc.q), (wk, &mut sc.k)],
                            &sc.xn,
                        )?;
                        exec.q8_0_gemv_repacked_multi(
                            &mut [(wv, &mut sc.v), (wg, &mut sc.gate_h)],
                            &sc.xn,
                        )?;
                    }
                    _ => {
                        gemv8_any(
                            &exec, &layer.wq, &sc.xn, &sc.xq, &sc.xs, &sc.ssums, &mut sc.q,
                        )?;
                        gemv8_any(
                            &exec, &layer.wk, &sc.xn, &sc.xq, &sc.xs, &sc.ssums, &mut sc.k,
                        )?;
                        gemv8_any(
                            &exec, &layer.wv, &sc.xn, &sc.xq, &sc.xs, &sc.ssums, &mut sc.v,
                        )?;
                        gemv8_any(
                            &exec,
                            &layer.g_proj,
                            &sc.xn,
                            &sc.xq,
                            &sc.xs,
                            &sc.ssums,
                            &mut sc.gate_h,
                        )?;
                    }
                }
            } else if r1 {
                gemv_any(&exec, &layer.wq, &sc.xn, &mut sc.q)?;
                gemv_any(&exec, &layer.wk, &sc.xn, &mut sc.k)?;
                gemv_any(&exec, &layer.wv, &sc.xn, &mut sc.v)?;
                gemv_any(&exec, &layer.g_proj, &sc.xn, &mut sc.gate_h)?;
            } else if pf {
                // one mmq-layout quantize serves wq/wk/wv/g_proj
                pf_quant(
                    &exec,
                    &mut sc.xq,
                    &mut sc.xs,
                    &mut sc.yq,
                    &mut sc.xsums,
                    &sc.xn,
                    embd,
                    r,
                    anyq8,
                )?;
                pf_mm(
                    &exec,
                    &layer.wq,
                    &sc.xq,
                    &sc.xs,
                    &sc.yq,
                    &sc.xsums,
                    &mut sc.skfix,
                    &mut sc.q,
                    r,
                )?;
                pf_mm(
                    &exec,
                    &layer.wk,
                    &sc.xq,
                    &sc.xs,
                    &sc.yq,
                    &sc.xsums,
                    &mut sc.skfix,
                    &mut sc.k,
                    r,
                )?;
                pf_mm(
                    &exec,
                    &layer.wv,
                    &sc.xq,
                    &sc.xs,
                    &sc.yq,
                    &sc.xsums,
                    &mut sc.skfix,
                    &mut sc.v,
                    r,
                )?;
                pf_mm(
                    &exec,
                    &layer.g_proj,
                    &sc.xq,
                    &sc.xs,
                    &sc.yq,
                    &sc.xsums,
                    &mut sc.skfix,
                    &mut sc.gate_h,
                    r,
                )?;
            } else {
                // one quantize serves wq/wk/wv/g_proj (group dedupe)
                exec.quantize_q8(&sc.xn, &mut sc.xq, &mut sc.xs, r * embd)?;
                // q|k|v|g as one launch on the r<=4 nc rung (entry 320): the
                // split band ran 8 nc GEMVs/layer at a ~10.3us avg where the
                // dense planes' byte floor is ~3us at c4 -
                // same launch economics as the r1 entry-317 merge above.
                // r<=4 mirrors mmq_pre's own nc-rung condition exactly.
                match (&layer.wq, &layer.wk, &layer.wv, &layer.g_proj) {
                    (QuantW::Q8(wq), QuantW::Q8(wk), QuantW::Q8(wv), QuantW::Q8(wg))
                        if r <= 4 && exec.has_q8_0_gemv_dp4a_nc_multi() && !no_gemv_multi() =>
                    {
                        exec.q8_0_gemv_dp4a_nc_multi(
                            &mut [
                                (wq, &mut sc.q),
                                (wk, &mut sc.k),
                                (wv, &mut sc.v),
                                (wg, &mut sc.gate_h),
                            ],
                            &sc.xq,
                            &sc.xs,
                            r,
                        )?;
                    }
                    _ => {
                        mmq_pre_any(
                            &exec,
                            &layer.wq,
                            &sc.xq,
                            &sc.xs,
                            &mut sc.ssums,
                            &mut sc.part,
                            &mut sc.q,
                            r,
                        )?;
                        mmq_pre_any(
                            &exec,
                            &layer.wk,
                            &sc.xq,
                            &sc.xs,
                            &mut sc.ssums,
                            &mut sc.part,
                            &mut sc.k,
                            r,
                        )?;
                        mmq_pre_any(
                            &exec,
                            &layer.wv,
                            &sc.xq,
                            &sc.xs,
                            &mut sc.ssums,
                            &mut sc.part,
                            &mut sc.v,
                            r,
                        )?;
                        mmq_pre_any(
                            &exec,
                            &layer.g_proj,
                            &sc.xq,
                            &sc.xs,
                            &mut sc.ssums,
                            &mut sc.part,
                            &mut sc.gate_h,
                            r,
                        )?;
                    }
                }
            }
            // Decode epilogue fold: q/k norm + rope + k/v append in one
            // launch (bit-identical to the six-kernel chain - the fold
            // replicates each op's math verbatim). Decode mode only; the
            // prefill band keeps its batch classes (SWA sub-span appends).
            let dec_nra = cuts.is_none()
                && hd == 128
                && (layer.is_swa || hp.n_rot == 64)
                && exec.has_lag_qk_nra_rows()
                && !no_qk_nra();
            if !dec_nra {
                exec.rmsnorm_batch(&sc.q, &layer.q_norm.buf, &mut sc.qn, hd, eps, r * nh)?;
                if qk_fused {
                    // k rows live at offset q_dim inside the fused [q|k|gate] plane
                    exec.rmsnorm_batch_at(
                        &sc.q,
                        q_dim,
                        &layer.k_norm.buf,
                        &mut sc.kn,
                        hd,
                        eps,
                        n_kv,
                    )?;
                } else {
                    exec.rmsnorm_batch(&sc.k, &layer.k_norm.buf, &mut sc.kn, hd, eps, r * n_kv)?;
                }
                if layer.is_swa {
                    exec.rope_yarn_batch(&mut sc.qn, &sc.d_pos, nh, hd, hp.rope_swa, r)?;
                    exec.rope_yarn_batch(&mut sc.kn, &sc.d_pos, n_kv, hd, hp.rope_swa, r)?;
                } else {
                    exec.mrope(
                        &mut sc.qn,
                        &sc.d_mrope,
                        r,
                        nh,
                        hd,
                        hp.n_rot,
                        hp.rope_full,
                        sections,
                    )?;
                    exec.mrope(
                        &mut sc.kn,
                        &sc.d_mrope,
                        r,
                        n_kv,
                        hd,
                        hp.n_rot,
                        hp.rope_full,
                        sections,
                    )?;
                }
            }
            // append + attend, per cache kind
            let (bt, window) = if layer.is_swa {
                (&bs.swa_bt, hp.swa_window)
            } else {
                (&bs.d_bt, 0usize)
            };
            let kvs = &mut bs.kv[li];
            match cuts {
                Some(c) if layer.is_swa => {
                    // span ladder: within one sub-span nothing a row reads has
                    // been recycled (ring holds SWA_SPAN + window)
                    debug_assert!(ring * 16 >= SWA_SPAN + hp.swa_window);
                    for &(off, len) in &c.swa {
                        exec.kv_append_batch_paged_rows(
                            &sc.kn,
                            &mut kvs.k,
                            &sc.d_pos,
                            Some(&sc.d_slots),
                            bt,
                            bps,
                            kv_dim,
                            off,
                            len,
                            kv_dtype,
                        )?;
                        exec.kv_append_batch_paged_rows(
                            &sc.v,
                            &mut kvs.v,
                            &sc.d_pos,
                            Some(&sc.d_slots),
                            bt,
                            bps,
                            kv_dim,
                            off,
                            len,
                            kv_dtype,
                        )?;
                        // the fused tick's decode band: one decode-class
                        // launch for all its rows (see PfCuts::dec)
                        if off < c.dec {
                            exec.attn_decode_batch_rows_paged(
                                &sc.qn,
                                &kvs.k,
                                &kvs.v,
                                &sc.sinks,
                                &mut sc.attn,
                                &sc.d_pos,
                                Some(&sc.d_slots),
                                bt,
                                bps,
                                nh,
                                n_kv,
                                hd,
                                kv_dim,
                                window,
                                off,
                                len,
                                scale,
                                kv_dtype,
                            )?;
                        } else if wmma_pf {
                            exec.attn_prefill_f16_paged_at(
                                &sc.qn,
                                &kvs.k,
                                &kvs.v,
                                &sc.sinks,
                                &mut sc.attn,
                                &sc.d_pos,
                                &sc.d_slots,
                                off,
                                bt,
                                bps,
                                nh,
                                n_kv,
                                hd,
                                kv_dim,
                                window,
                                len,
                                scale,
                                kv_dtype,
                            )?;
                        } else if len > 24 && exec.has_attn_prefill_paged() {
                            exec.attn_prefill_rows_paged(
                                &sc.qn,
                                &kvs.k,
                                &kvs.v,
                                &sc.sinks,
                                &mut sc.attn,
                                &sc.d_pos,
                                &sc.d_slots,
                                bt,
                                bps,
                                nh,
                                n_kv,
                                hd,
                                kv_dim,
                                window,
                                off,
                                len,
                                scale,
                                kv_dtype,
                            )?;
                        } else {
                            exec.attn_decode_batch_rows_paged(
                                &sc.qn,
                                &kvs.k,
                                &kvs.v,
                                &sc.sinks,
                                &mut sc.attn,
                                &sc.d_pos,
                                Some(&sc.d_slots),
                                bt,
                                bps,
                                nh,
                                n_kv,
                                hd,
                                kv_dim,
                                window,
                                off,
                                len,
                                scale,
                                kv_dtype,
                            )?;
                        }
                    }
                }
                Some(c) => {
                    // full layer, prefill: append the whole chunk in one pass
                    // (per-row slot/pos - rows land in their own slots' pool
                    // blocks, no aliasing), attend per same-slot run (causal
                    // bound = positions[row]). Single-run chunks keep the
                    // whole-chunk calls - bit-for-bit the serial path.
                    exec.kv_append_batch_paged(
                        &sc.kn,
                        &mut kvs.k,
                        &sc.d_pos,
                        Some(&sc.d_slots),
                        bt,
                        bps,
                        kv_dim,
                        r,
                        kv_dtype,
                    )?;
                    exec.kv_append_batch_paged(
                        &sc.v,
                        &mut kvs.v,
                        &sc.d_pos,
                        Some(&sc.d_slots),
                        bt,
                        bps,
                        kv_dim,
                        r,
                        kv_dtype,
                    )?;
                    if c.runs.len() == 1 {
                        if wmma_pf {
                            exec.attn_prefill_f16_paged(
                                &sc.qn,
                                &kvs.k,
                                &kvs.v,
                                &sc.sinks,
                                &mut sc.attn,
                                &sc.d_pos,
                                &sc.d_slots,
                                bt,
                                bps,
                                nh,
                                n_kv,
                                hd,
                                kv_dim,
                                window,
                                r,
                                scale,
                                kv_dtype,
                            )?;
                        } else if r > 24 && exec.has_attn_prefill_paged() {
                            exec.attn_prefill_paged(
                                &sc.qn,
                                &kvs.k,
                                &kvs.v,
                                &sc.sinks,
                                &mut sc.attn,
                                &sc.d_pos,
                                &sc.d_slots,
                                bt,
                                bps,
                                nh,
                                n_kv,
                                hd,
                                kv_dim,
                                window,
                                r,
                                scale,
                                kv_dtype,
                            )?;
                        } else {
                            exec.attn_decode_batch_paged(
                                &sc.qn,
                                &kvs.k,
                                &kvs.v,
                                &sc.sinks,
                                &mut sc.attn,
                                &sc.d_pos,
                                Some(&sc.d_slots),
                                bt,
                                bps,
                                nh,
                                n_kv,
                                hd,
                                kv_dim,
                                window,
                                r,
                                scale,
                                kv_dtype,
                            )?;
                        }
                    } else {
                        for &(off, len) in &c.runs {
                            if off < c.dec {
                                // fused decode band - one launch, decode class
                                exec.attn_decode_batch_rows_paged(
                                    &sc.qn,
                                    &kvs.k,
                                    &kvs.v,
                                    &sc.sinks,
                                    &mut sc.attn,
                                    &sc.d_pos,
                                    Some(&sc.d_slots),
                                    bt,
                                    bps,
                                    nh,
                                    n_kv,
                                    hd,
                                    kv_dim,
                                    window,
                                    off,
                                    len,
                                    scale,
                                    kv_dtype,
                                )?;
                            } else if wmma_pf {
                                exec.attn_prefill_f16_paged_at(
                                    &sc.qn,
                                    &kvs.k,
                                    &kvs.v,
                                    &sc.sinks,
                                    &mut sc.attn,
                                    &sc.d_pos,
                                    &sc.d_slots,
                                    off,
                                    bt,
                                    bps,
                                    nh,
                                    n_kv,
                                    hd,
                                    kv_dim,
                                    window,
                                    len,
                                    scale,
                                    kv_dtype,
                                )?;
                            } else if len > 24 && exec.has_attn_prefill_paged() {
                                exec.attn_prefill_rows_paged(
                                    &sc.qn,
                                    &kvs.k,
                                    &kvs.v,
                                    &sc.sinks,
                                    &mut sc.attn,
                                    &sc.d_pos,
                                    &sc.d_slots,
                                    bt,
                                    bps,
                                    nh,
                                    n_kv,
                                    hd,
                                    kv_dim,
                                    window,
                                    off,
                                    len,
                                    scale,
                                    kv_dtype,
                                )?;
                            } else {
                                exec.attn_decode_batch_rows_paged(
                                    &sc.qn,
                                    &kvs.k,
                                    &kvs.v,
                                    &sc.sinks,
                                    &mut sc.attn,
                                    &sc.d_pos,
                                    Some(&sc.d_slots),
                                    bt,
                                    bps,
                                    nh,
                                    n_kv,
                                    hd,
                                    kv_dim,
                                    window,
                                    off,
                                    len,
                                    scale,
                                    kv_dtype,
                                )?;
                            }
                        }
                    }
                }
                None => {
                    // decode: one new token per row
                    if dec_nra {
                        // fused epilogue: norms + ropes + both appends. q/k
                        // read the fused [q|k|gate] plane at r==1, separate
                        // planes otherwise; strides in f32 elements.
                        let (ks, k_off, k_stride) = if qk_fused {
                            (&sc.q, q_dim, 0)
                        } else {
                            (&sc.k, 0, kv_dim)
                        };
                        exec.lag_qk_nra_rows(
                            &sc.q,
                            0,
                            q_dim,
                            ks,
                            k_off,
                            k_stride,
                            &sc.v,
                            kv_dim,
                            &layer.q_norm.buf,
                            &layer.k_norm.buf,
                            &mut sc.qn,
                            &mut kvs.k,
                            &mut kvs.v,
                            &sc.d_pos,
                            Some(&sc.d_slots),
                            (!layer.is_swa).then_some(&sc.d_mrope),
                            bt,
                            bps,
                            nh,
                            n_kv,
                            hd,
                            hp.n_rot,
                            eps,
                            if layer.is_swa {
                                hp.rope_swa
                            } else {
                                hp.rope_full
                            },
                            sections,
                            r,
                            kv_dtype,
                        )?;
                    } else {
                        exec.kv_append_batch_paged(
                            &sc.kn,
                            &mut kvs.k,
                            &sc.d_pos,
                            Some(&sc.d_slots),
                            bt,
                            bps,
                            kv_dim,
                            r,
                            kv_dtype,
                        )?;
                        exec.kv_append_batch_paged(
                            &sc.v,
                            &mut kvs.v,
                            &sc.d_pos,
                            Some(&sc.d_slots),
                            bt,
                            bps,
                            kv_dim,
                            r,
                            kv_dtype,
                        )?;
                    }
                    // FlashDecoding partial+combine when the unsplit grid
                    // would starve the die (fixed per (nh, r, window) - the
                    // window is a static layer property, so still graph-safe).
                    // Fused GQA shapes take partial+combine even at ns == 1:
                    // the plain per-q-head kernel re-reads K/V group-x.
                    let ns = attn_splits(nh, n_kv, r, exec.sm_count(), window);
                    let fused1 = ns == 1 && attn_gqa_fused(nh, n_kv, r);
                    if (ns > 1 || fused1) && exec.has_attn_partial_batch_paged() {
                        exec.attn_partial_batch_paged(
                            &sc.qn,
                            &kvs.k,
                            &kvs.v,
                            &mut sc.attn_o,
                            &mut sc.attn_ml,
                            &sc.d_pos,
                            Some(&sc.d_slots),
                            bt,
                            bps,
                            nh,
                            n_kv,
                            hd,
                            kv_dim,
                            window,
                            ns,
                            r,
                            scale,
                            kv_dtype,
                        )?;
                        exec.attn_combine_batch(
                            &sc.attn_o,
                            &sc.attn_ml,
                            &sc.sinks,
                            &mut sc.attn,
                            nh,
                            hd,
                            ns,
                            r,
                        )?;
                    } else {
                        exec.attn_decode_batch_paged(
                            &sc.qn,
                            &kvs.k,
                            &kvs.v,
                            &sc.sinks,
                            &mut sc.attn,
                            &sc.d_pos,
                            Some(&sc.d_slots),
                            bt,
                            bps,
                            nh,
                            n_kv,
                            hd,
                            kv_dim,
                            window,
                            r,
                            scale,
                            kv_dtype,
                        )?;
                    }
                }
            }
            if qk_fused {
                // gate rows live after q and k in the fused plane
                exec.mul_softplus_head_at(&mut sc.attn, &sc.q, q_dim + kv_dim, nh, hd, 1)?;
            } else {
                exec.mul_softplus_head(&mut sc.attn, &sc.gate_h, nh, hd, r)?;
            }
            if g8 {
                // staging only when wo is a k-quant plane - the Q8_0 exact
                // GEMV reads f32 attn directly (same dead-launch fix as the
                // qkv band above)
                if matches!(layer.wo, QuantW::Kq(_)) {
                    exec.quantize_q8_sums(&sc.attn, &mut sc.xq, &mut sc.xs, &mut sc.ssums, q_dim)?;
                }
                gemv8_any(
                    &exec,
                    &layer.wo,
                    &sc.attn,
                    &sc.xq,
                    &sc.xs,
                    &sc.ssums,
                    &mut sc.proj,
                )?;
            } else if r1 {
                gemv_any(&exec, &layer.wo, &sc.attn, &mut sc.proj)?;
            } else if pf {
                pf_quant(
                    &exec,
                    &mut sc.xq,
                    &mut sc.xs,
                    &mut sc.yq,
                    &mut sc.xsums,
                    &sc.attn,
                    q_dim,
                    r,
                    anyq8,
                )?;
                pf_mm(
                    &exec,
                    &layer.wo,
                    &sc.xq,
                    &sc.xs,
                    &sc.yq,
                    &sc.xsums,
                    &mut sc.skfix,
                    &mut sc.proj,
                    r,
                )?;
            } else {
                exec.quantize_q8(&sc.attn, &mut sc.xq, &mut sc.xs, r * q_dim)?;
                mmq_pre_any(
                    &exec,
                    &layer.wo,
                    &sc.xq,
                    &sc.xs,
                    &mut sc.ssums,
                    &mut sc.part,
                    &mut sc.proj,
                    r,
                )?;
            }
            exec.add_rmsnorm_batch(
                &mut sc.x,
                &sc.proj,
                &layer.ffn_norm.buf,
                &mut sc.xn,
                embd,
                eps,
                r,
            )?;

            match &layer.ffn {
                Ffn::Dense { gate, up, down } => {
                    let ff = gate.dims()[1];
                    if g8 {
                        exec.quantize_q8_sums(&sc.xn, &mut sc.xq, &mut sc.xs, &mut sc.ssums, embd)?;
                        gemv8_any(
                            &exec,
                            gate,
                            &sc.xn,
                            &sc.xq,
                            &sc.xs,
                            &sc.ssums,
                            &mut sc.ffn_gate,
                        )?;
                        gemv8_any(&exec, up, &sc.xn, &sc.xq, &sc.xs, &sc.ssums, &mut sc.ffn_up)?;
                        exec.swiglu(&mut sc.ffn_gate, &sc.ffn_up, ff)?;
                        exec.quantize_q8_sums(
                            &sc.ffn_gate,
                            &mut sc.xq,
                            &mut sc.xs,
                            &mut sc.ssums,
                            ff,
                        )?;
                        gemv8_any(
                            &exec,
                            down,
                            &sc.ffn_gate,
                            &sc.xq,
                            &sc.xs,
                            &sc.ssums,
                            &mut sc.proj,
                        )?;
                    } else if r1 {
                        gemv_any(&exec, gate, &sc.xn, &mut sc.ffn_gate)?;
                        gemv_any(&exec, up, &sc.xn, &mut sc.ffn_up)?;
                        exec.swiglu(&mut sc.ffn_gate, &sc.ffn_up, ff)?;
                        gemv_any(&exec, down, &sc.ffn_gate, &mut sc.proj)?;
                    } else if pf {
                        pf_quant(
                            &exec,
                            &mut sc.xq,
                            &mut sc.xs,
                            &mut sc.yq,
                            &mut sc.xsums,
                            &sc.xn,
                            embd,
                            r,
                            anyq8,
                        )?;
                        pf_mm(
                            &exec,
                            gate,
                            &sc.xq,
                            &sc.xs,
                            &sc.yq,
                            &sc.xsums,
                            &mut sc.skfix,
                            &mut sc.ffn_gate,
                            r,
                        )?;
                        pf_mm(
                            &exec,
                            up,
                            &sc.xq,
                            &sc.xs,
                            &sc.yq,
                            &sc.xsums,
                            &mut sc.skfix,
                            &mut sc.ffn_up,
                            r,
                        )?;
                        exec.swiglu(&mut sc.ffn_gate, &sc.ffn_up, r * ff)?;
                        pf_quant(
                            &exec,
                            &mut sc.xq,
                            &mut sc.xs,
                            &mut sc.yq,
                            &mut sc.xsums,
                            &sc.ffn_gate,
                            ff,
                            r,
                            anyq8,
                        )?;
                        pf_mm(
                            &exec,
                            down,
                            &sc.xq,
                            &sc.xs,
                            &sc.yq,
                            &sc.xsums,
                            &mut sc.skfix,
                            &mut sc.proj,
                            r,
                        )?;
                    } else {
                        exec.quantize_q8(&sc.xn, &mut sc.xq, &mut sc.xs, r * embd)?;
                        mmq_pre_any(
                            &exec,
                            gate,
                            &sc.xq,
                            &sc.xs,
                            &mut sc.ssums,
                            &mut sc.part,
                            &mut sc.ffn_gate,
                            r,
                        )?;
                        mmq_pre_any(
                            &exec,
                            up,
                            &sc.xq,
                            &sc.xs,
                            &mut sc.ssums,
                            &mut sc.part,
                            &mut sc.ffn_up,
                            r,
                        )?;
                        exec.swiglu(&mut sc.ffn_gate, &sc.ffn_up, r * ff)?;
                        exec.quantize_q8(&sc.ffn_gate, &mut sc.xq, &mut sc.xs, r * ff)?;
                        mmq_pre_any(
                            &exec,
                            down,
                            &sc.xq,
                            &sc.xs,
                            &mut sc.ssums,
                            &mut sc.part,
                            &mut sc.proj,
                            r,
                        )?;
                    }
                }
                Ffn::Moe(w) => {
                    // routed experts: token-batched k-quant/Q8 class, batch=r
                    // (the sorted/moe_align class is a follow-up). The Q4/Q5
                    // mu-term sums fuse into the quantize pass
                    // (pd_quantize_q8_sums - bit-identical to the two-step).
                    let needs_gu = match (&w.gate_exps, &w.up_exps) {
                        (ExpW::Kq(g), ExpW::Kq(u)) => {
                            crate::gpu::kq_needs_sums(g.ty) || crate::gpu::kq_needs_sums(u.ty)
                        }
                        _ => false,
                    };
                    // g8 forces the sums-carrying quantize: the shexp W4A8
                    // GEMVs below reuse this xq AND need ssums for Q4/Q5
                    if needs_gu || g8 {
                        exec.quantize_q8_sums(
                            &sc.xn,
                            &mut sc.xq,
                            &mut sc.xs,
                            &mut sc.ssums,
                            r * embd,
                        )?;
                    } else {
                        exec.quantize_q8(&sc.xn, &mut sc.xq, &mut sc.xs, r * embd)?;
                    }
                    exec.matvec_f32_batch(&w.router_w, &sc.xn, &mut sc.moe_logits, r)?;
                    exec.moe_topk_sigmoid_batch(
                        &sc.moe_logits,
                        &w.probs_bias.buf,
                        m.routed_scale,
                        m.n_expert,
                        m.n_active,
                        &mut sc.moe_idx,
                        &mut sc.moe_w,
                        r,
                    )?;
                    // Sorted (moe_align) vs token-batched: the sorted mma
                    // pair reads each touched expert's weights once per pass
                    // - the prefill class (token-batched gate_up at r=130
                    // was ~3 ms/LAYER). The election keys on
                    // MODE, never on pair count: prefill bytes must be
                    // r-invariant (warm-resume tails reproduce cold chunks
                    // byte-exactly - a pair floor would flip class on short
                    // tails), so all prefill sorts and decode (r ≤ 8 slots,
                    // where the mostly-PAD sorted grid loses per the Act-94
                    // no-dedup-at-c1 fact) stays token-batched.
                    let kq_pair = matches!(
                        (&w.gate_exps, &w.up_exps),
                        (ExpW::Kq(g), ExpW::Kq(u)) if g.ty == u.ty
                    ) && matches!(&w.down_exps, ExpW::Kq(_));
                    // Spec-verify rounds also sort past the decode regime:
                    // token-batched re-reads expert weights per (row, active)
                    // pair - a chunk-16 round is 128 activations × ~2.3 MB ×
                    // 39 layers ≈ 11 GB, measured ~1.2 ms/verify-row (stage
                    // D) - while the sorted pair reads each TOUCHED expert
                    // once per pass. r > 8 keeps the Act-94 decode fact
                    // (≤64 activations lose to the mostly-PAD sorted grid).
                    // Class keys on (mode, r) so captured verify graphs stay
                    // signature-deterministic.
                    let sorted = kq_pair
                        && (pf || (spec && r > 8))
                        && exec.has_kquant_moe_mma()
                        && exec.compute_capability().0 >= 8
                        && paddock_models::dev_var_os!("PADDOCK_NO_SORTED_QMOE").is_none();
                    if sorted {
                        let max_blocks = (r * m.n_active + m.n_expert * 31).div_ceil(32);
                        exec.moe_align(
                            &sc.moe_idx,
                            &mut sc.srow,
                            &mut sc.sslot,
                            &mut sc.bexp,
                            r,
                            m.n_active,
                            m.n_expert,
                            max_blocks,
                        )?;
                        let (ExpW::Kq(g), ExpW::Kq(u)) = (&w.gate_exps, &w.up_exps) else {
                            unreachable!("kq_pair checked")
                        };
                        exec.kquant_moe_gate_up_mma(
                            g,
                            u,
                            &sc.srow,
                            &sc.bexp,
                            &sc.xq,
                            &sc.xs,
                            needs_gu.then_some(&sc.ssums),
                            &mut sc.moe_fq,
                            &mut sc.moe_fs,
                            max_blocks,
                        )?;
                        let ExpW::Kq(d) = &w.down_exps else {
                            unreachable!("kq_pair checked")
                        };
                        let needs_d = crate::gpu::kq_needs_sums(d.ty);
                        if needs_d {
                            // sums over the SORTED fq rows (PAD rows are zeros)
                            exec.q8_sums_strided(
                                &sc.moe_fq,
                                &mut sc.moe_ssums,
                                m.moe_ff,
                                max_blocks * 32,
                            )?;
                        }
                        exec.kquant_moe_down_mma(
                            d,
                            &sc.srow,
                            &sc.sslot,
                            &sc.bexp,
                            &sc.moe_w,
                            &sc.moe_fq,
                            &sc.moe_fs,
                            needs_d.then_some(&sc.moe_ssums),
                            &mut sc.moe_part,
                            m.n_active,
                            max_blocks,
                        )?;
                        exec.stream
                            .memset_zeros(&mut sc.proj)
                            .map_err(|e| GpuError::Driver(e.to_string()))?;
                        exec.moe_slot_combine(&sc.moe_part, &mut sc.proj, embd, m.n_active, r)?;
                    } else {
                        match (&w.gate_exps, &w.up_exps) {
                            (ExpW::Kq(g), ExpW::Kq(u)) => {
                                exec.kquant_moe_gate_up(
                                    g,
                                    u,
                                    &sc.moe_idx,
                                    &sc.xq,
                                    &sc.xs,
                                    needs_gu.then_some(&sc.ssums),
                                    &mut sc.moe_fused,
                                    m.n_active,
                                    r,
                                )?;
                            }
                            _ => {
                                let g8 = match &w.gate_exps {
                                    ExpW::Q8(g) => g,
                                    ExpW::Kq(_) => unreachable!("loader pairs gate/up residency"),
                                };
                                let u8_ = match &w.up_exps {
                                    ExpW::Q8(u) => u,
                                    ExpW::Kq(_) => unreachable!("loader pairs gate/up residency"),
                                };
                                exec.q8_0_moe_gate_up(
                                    g8,
                                    u8_,
                                    &sc.moe_idx,
                                    &sc.xq,
                                    &sc.xs,
                                    &mut sc.moe_fused,
                                    m.n_active,
                                    r,
                                )?;
                            }
                        }
                        let needs_d =
                            matches!(&w.down_exps, ExpW::Kq(d) if crate::gpu::kq_needs_sums(d.ty));
                        if needs_d {
                            exec.quantize_q8_sums(
                                &sc.moe_fused,
                                &mut sc.moe_fq,
                                &mut sc.moe_fs,
                                &mut sc.moe_ssums,
                                r * m.n_active * m.moe_ff,
                            )?;
                        } else {
                            exec.quantize_q8(
                                &sc.moe_fused,
                                &mut sc.moe_fq,
                                &mut sc.moe_fs,
                                r * m.n_active * m.moe_ff,
                            )?;
                        }
                        match &w.down_exps {
                            ExpW::Kq(d) => {
                                exec.kquant_moe_down(
                                    d,
                                    &sc.moe_idx,
                                    &sc.moe_w,
                                    &sc.moe_fq,
                                    &sc.moe_fs,
                                    needs_d.then_some(&sc.moe_ssums),
                                    &mut sc.proj,
                                    m.n_active,
                                    r,
                                )?;
                            }
                            ExpW::Q8(d) => {
                                exec.q8_0_moe_down(
                                    d,
                                    &sc.moe_idx,
                                    &sc.moe_w,
                                    &sc.moe_fq,
                                    &sc.moe_fs,
                                    &mut sc.proj,
                                    m.n_active,
                                    r,
                                )?;
                            }
                        }
                    } // end sorted-vs-token-batched
                    // always-on ungated shared expert (r>1 reuses the xn quant;
                    // r==1 rides the exact GEMV class like the serial lane)
                    if g8 && let Some(gu) = &w.shexp_gateup {
                        // W4A8 form: the gate_up GEMV reuses the MoE arm's xn
                        // quant (same input, staged above - the routed-down
                        // quantize writes moe_ssums, so xn's ssums survive);
                        // the down's sh_up input quantizes into its own tiny
                        // planes
                        let needs = crate::gpu::kq_needs_sums(gu.ty);
                        exec.kquant_gemv_w4a8(
                            gu,
                            &sc.xq,
                            &sc.xs,
                            needs.then_some(&sc.ssums),
                            &mut sc.sh_gate,
                        )?;
                        exec.swiglu_fused(&sc.sh_gate, &mut sc.sh_up, m.shexp_ff, 1)?;
                        exec.quantize_q8_sums(
                            &sc.sh_up,
                            &mut sc.sh_xq,
                            &mut sc.sh_xs,
                            &mut sc.sh_ssums,
                            m.shexp_ff,
                        )?;
                        gemv8_any(
                            &exec,
                            &w.shexp_down,
                            &sc.sh_up,
                            &sc.sh_xq,
                            &sc.sh_xs,
                            &sc.sh_ssums,
                            &mut sc.sh_out,
                        )?;
                    } else if r1 && let Some(gu) = &w.shexp_gateup {
                        // fused [gate|up] GEMV + the swiglu_fused epilogue:
                        // 3 launches instead of 4, and the GEMV is 2× wider
                        exec.kquant_gemv(gu, &sc.xn, &mut sc.sh_gate)?;
                        exec.swiglu_fused(&sc.sh_gate, &mut sc.sh_up, m.shexp_ff, 1)?;
                        gemv_any(&exec, &w.shexp_down, &sc.sh_up, &mut sc.sh_out)?;
                    } else if r1 {
                        // shexp gate|up as one launch on Q8_0 planes (entry
                        // 317 merge - same economics as the qkv band)
                        match (&w.shexp_gate, &w.shexp_up) {
                            (QuantW::Q8(sg), QuantW::Q8(su))
                                if exec.has_q8_0_gemv_repacked_multi() && !no_gemv_multi() =>
                            {
                                exec.q8_0_gemv_repacked_multi(
                                    &mut [(sg, &mut sc.sh_gate), (su, &mut sc.sh_up)],
                                    &sc.xn,
                                )?;
                            }
                            _ => {
                                gemv_any(&exec, &w.shexp_gate, &sc.xn, &mut sc.sh_gate)?;
                                gemv_any(&exec, &w.shexp_up, &sc.xn, &mut sc.sh_up)?;
                            }
                        }
                        exec.swiglu(&mut sc.sh_gate, &sc.sh_up, m.shexp_ff)?;
                        gemv_any(&exec, &w.shexp_down, &sc.sh_gate, &mut sc.sh_out)?;
                    } else if pf {
                        // mmq re-quant of xn: the routed experts consume the
                        // STRIDED xq (moe kernels' layout, already spent by
                        // now), the shexp's flat-mmq planes quantize their own
                        pf_quant(
                            &exec,
                            &mut sc.xq,
                            &mut sc.xs,
                            &mut sc.yq,
                            &mut sc.xsums,
                            &sc.xn,
                            embd,
                            r,
                            anyq8,
                        )?;
                        pf_mm(
                            &exec,
                            &w.shexp_gate,
                            &sc.xq,
                            &sc.xs,
                            &sc.yq,
                            &sc.xsums,
                            &mut sc.skfix,
                            &mut sc.sh_gate,
                            r,
                        )?;
                        pf_mm(
                            &exec,
                            &w.shexp_up,
                            &sc.xq,
                            &sc.xs,
                            &sc.yq,
                            &sc.xsums,
                            &mut sc.skfix,
                            &mut sc.sh_up,
                            r,
                        )?;
                        exec.swiglu(&mut sc.sh_gate, &sc.sh_up, r * m.shexp_ff)?;
                        pf_quant(
                            &exec,
                            &mut sc.xq,
                            &mut sc.xs,
                            &mut sc.yq,
                            &mut sc.xsums,
                            &sc.sh_gate,
                            m.shexp_ff,
                            r,
                            anyq8,
                        )?;
                        pf_mm(
                            &exec,
                            &w.shexp_down,
                            &sc.xq,
                            &sc.xs,
                            &sc.yq,
                            &sc.xsums,
                            &mut sc.skfix,
                            &mut sc.sh_out,
                            r,
                        )?;
                    } else {
                        // shexp gate|up as one launch on the r<=4 nc rung
                        // (entry 320 - the batched twin of the r1 entry-317
                        // merge above; both planes read the xn quant staged
                        // for the MoE band)
                        match (&w.shexp_gate, &w.shexp_up) {
                            (QuantW::Q8(sg), QuantW::Q8(su))
                                if r <= 4
                                    && exec.has_q8_0_gemv_dp4a_nc_multi()
                                    && !no_gemv_multi() =>
                            {
                                exec.q8_0_gemv_dp4a_nc_multi(
                                    &mut [(sg, &mut sc.sh_gate), (su, &mut sc.sh_up)],
                                    &sc.xq,
                                    &sc.xs,
                                    r,
                                )?;
                            }
                            _ => {
                                mmq_pre_any(
                                    &exec,
                                    &w.shexp_gate,
                                    &sc.xq,
                                    &sc.xs,
                                    &mut sc.ssums,
                                    &mut sc.part,
                                    &mut sc.sh_gate,
                                    r,
                                )?;
                                mmq_pre_any(
                                    &exec,
                                    &w.shexp_up,
                                    &sc.xq,
                                    &sc.xs,
                                    &mut sc.ssums,
                                    &mut sc.part,
                                    &mut sc.sh_up,
                                    r,
                                )?;
                            }
                        }
                        exec.swiglu(&mut sc.sh_gate, &sc.sh_up, r * m.shexp_ff)?;
                        exec.quantize_q8(&sc.sh_gate, &mut sc.xq, &mut sc.xs, r * m.shexp_ff)?;
                        mmq_pre_any(
                            &exec,
                            &w.shexp_down,
                            &sc.xq,
                            &sc.xs,
                            &mut sc.ssums,
                            &mut sc.part,
                            &mut sc.sh_out,
                            r,
                        )?;
                    }
                    exec.add(&mut sc.proj, &sc.sh_out, r * embd)?;
                }
            }
            exec.add(&mut sc.x, &sc.proj, r * embd)?;
            if let Some((aux, ids)) = dtap.as_mut()
                && let Some(band) = ids.iter().position(|&t| t == li)
            {
                let src =
                    sc.x.try_slice(0..r * embd)
                        .ok_or_else(|| GpuError::Driver("aux tap src".into()))?;
                let mut dst = aux
                    .try_slice_mut(band * cap * embd..band * cap * embd + r * embd)
                    .ok_or_else(|| GpuError::Driver("aux tap dst".into()))?;
                exec.stream.memcpy_dtod(&src, &mut dst).map_err(drv)?;
            }
        }
        Ok(())
    }

    /// Final norm + LM head over the first `rows` residual rows into
    /// head_logits [rows, vocab].
    fn head_rows(&mut self, rows: usize) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let hp = &self.hp;
        let bs = self.batch.as_mut().expect("batch enabled");
        let sc = &mut bs.sc;
        exec.rmsnorm_batch(
            &sc.x,
            &self.output_norm.buf,
            &mut sc.xn,
            hp.n_embd,
            hp.eps,
            rows,
        )?;
        if rows == 1 {
            // the r1 head rides the W4A8 GEMV too (Q6_K head at
            // out=vocab is the single biggest r1 gemv - 293 us, against the
            // W4A8 class's ~630 GB/s byte rate). Same latch as the
            // layer walk; exact-f32 GEMV pinned via PADDOCK_KQ_EXACT_GEMV.
            if exec.has_kquant_gemv_w4a8()
                && paddock_models::dev_var_os!("PADDOCK_KQ_EXACT_GEMV").is_none()
                && let QuantW::Kq(k) = &self.lm_head
            {
                exec.quantize_q8_sums(&sc.xn, &mut sc.xq, &mut sc.xs, &mut sc.ssums, hp.n_embd)?;
                let needs = crate::gpu::kq_needs_sums(k.ty);
                exec.kquant_gemv_w4a8(
                    k,
                    &sc.xq,
                    &sc.xs,
                    needs.then_some(&sc.ssums),
                    &mut sc.head_logits,
                )?;
                return Ok(());
            }
            return gemv_any(&exec, &self.lm_head, &sc.xn, &mut sc.head_logits);
        }
        exec.quantize_q8(&sc.xn, &mut sc.xq, &mut sc.xs, rows * hp.n_embd)?;
        // vocab-wide out exceeds the mma partial plane -> dp4a rungs
        match &self.lm_head {
            QuantW::Kq(k) => mmq_kq_pre(
                &exec,
                k,
                &sc.xq,
                &sc.xs,
                &mut sc.ssums,
                &mut sc.part,
                &mut sc.head_logits,
                rows,
            )?,
            QuantW::Q8(q) => mmq_pre(
                &exec,
                q,
                &sc.xq,
                &sc.xs,
                &mut sc.part,
                &mut sc.head_logits,
                rows,
            )?,
        }
        Ok(())
    }

    /// Prefill tail: head over residual row `row` (of the last chunk),
    /// returning that one vocab row on the host.
    fn head_row(&mut self, row: usize, _rows: usize) -> Result<Vec<f32>, GpuModelError> {
        let exec = self.exec.clone();
        let (n_embd, n_vocab) = (self.hp.n_embd, self.hp.n_vocab);
        // norm+head the whole tail up to `row` would waste vocab GEMM rows;
        // stage the single residual row at row 0 of a fresh pass instead
        // (bounced through proj - src and dst live in the same buffer)
        if row > 0 {
            let bs = self.batch.as_mut().expect("batch enabled");
            let sc = &mut bs.sc;
            let src =
                sc.x.try_slice(row * n_embd..(row + 1) * n_embd)
                    .ok_or_else(|| GpuError::Driver("x row slice".into()))?;
            let mut dst = sc
                .proj
                .try_slice_mut(0..n_embd)
                .ok_or_else(|| GpuError::Driver("proj row slice".into()))?;
            exec.stream.memcpy_dtod(&src, &mut dst).map_err(drv)?;
            let mut xd =
                sc.x.try_slice_mut(0..n_embd)
                    .ok_or_else(|| GpuError::Driver("x dst slice".into()))?;
            let ps = sc
                .proj
                .try_slice(0..n_embd)
                .ok_or_else(|| GpuError::Driver("proj src slice".into()))?;
            exec.stream.memcpy_dtod(&ps, &mut xd).map_err(drv)?;
        }
        self.head_rows(1)?;
        let bs = self.batch.as_mut().expect("batch enabled");
        let v = bs
            .sc
            .head_logits
            .try_slice(0..n_vocab)
            .ok_or_else(|| GpuError::Driver("head row slice".into()))?;
        let out = exec.stream.clone_dtoh(&v).map_err(drv)?;
        Ok(out)
    }

    /// Read the [rows, vocab] decode logits back to the host.
    pub(crate) fn read_batch_logits(&mut self, rows: usize) -> Result<Vec<f32>, GpuModelError> {
        let hp_vocab = self.hp.n_vocab;
        let bs = self.batch.as_mut().expect("batch enabled");
        let v = bs
            .sc
            .head_logits
            .try_slice(0..rows * hp_vocab)
            .ok_or_else(|| GpuError::Driver("batch logits slice".into()))?;
        let out = self.exec.stream.clone_dtoh(&v).map_err(drv)?;
        Ok(out)
    }

    // ── device sampling + decode pipe  ───────────────────────

    /// device-truncation engagement witness (bisect-trap law): once per process.
    fn trunc_dev5_witness(rows: usize) {
        static DEV5: std::sync::Once = std::sync::Once::new();
        DEV5.call_once(|| {
            eprintln!("[trunc-dev5] engaged: r={rows} (laguna full-device truncation sampling)");
        });
    }

    /// Pack per-row sampler params (inv_t, u, mode, pad). Host/Hole rows stay
    /// mode 0 = untouched; RsVerify is gemma4-only (laguna has no spec yet).
    /// TruncCat rows pack mode 5 + the tpar side plane (Some iff any)
    /// - pd_sample_rows_t draws them fully on device after sample_rows.
    fn pack_samp_par(plans: &[crate::generator::RowSample]) -> (Vec<u32>, Option<Vec<u32>>) {
        use crate::generator::RowSample;
        use crate::sampler::DevicePlan;
        let mut par = vec![0u32; plans.len() * 4];
        let mut tpar = vec![0u32; plans.len() * 4];
        let mut any_trunc = false;
        for (i, p) in plans.iter().enumerate() {
            match p {
                RowSample::Hole | RowSample::Host => {}
                RowSample::Device(DevicePlan::Greedy) => par[i * 4 + 2] = 1,
                RowSample::Device(DevicePlan::Categorical { inv_t, u }) => {
                    par[i * 4] = inv_t.to_bits();
                    par[i * 4 + 1] = u.to_bits();
                    par[i * 4 + 2] = 2;
                }
                RowSample::Device(DevicePlan::TruncCat {
                    inv_t,
                    u,
                    k,
                    top_p,
                    min_p,
                }) => {
                    par[i * 4] = inv_t.to_bits();
                    par[i * 4 + 1] = u.to_bits();
                    par[i * 4 + 2] = if *k >= 1 && *k <= 64 { 5 } else { 6 };
                    tpar[i * 4] = *k;
                    tpar[i * 4 + 1] = top_p.to_bits();
                    tpar[i * 4 + 2] = min_p.to_bits();
                    any_trunc = true;
                }
                RowSample::Device(DevicePlan::RsVerify { .. })
                | RowSample::Device(DevicePlan::RsTrunc { .. }) => {}
            }
        }
        (par, any_trunc.then_some(tpar))
    }

    pub(crate) fn supports_device_sampling_impl(&self) -> bool {
        self.batch.is_some() && self.exec.has_sample_rows()
    }

    /// TruncCat rows execute fully on device (slot 435, mode 5) - the
    /// service may emit truncation plans; old packs -> false -> Host readback.
    pub(crate) fn device_trunc_supported(&self) -> bool {
        self.batch.is_some() && self.exec.has_sample_rows_t() && self.exec.has_sample_rows_p()
    }

    /// Device-sampled decode tick: graph replay + sample_rows, ids come back
    /// as r u32s; only Host-plan rows pay a vocab-row readback. sample_rows
    /// launches EAGERLY after the replay (one launch - not worth baking a
    /// second graph variant per r).
    pub(crate) fn forward_batch_sampled_impl(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        plans: &[crate::generator::RowSample],
    ) -> Result<crate::generator::SampledStep, GpuModelError> {
        self.forward_batch_sampled_slots(tokens, positions, None, plans)
    }

    /// `forward_batch_sampled_impl` with optional explicit slot ids. None
    /// keeps the dense row-i = slot-i mapping the classic path relies on;
    /// the mixed tick passes real slots because its decoders are whichever
    /// slots have finished prefilling (see `batch_step_slots`).
    pub(crate) fn forward_batch_sampled_slots(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        slots: Option<&[u32]>,
        plans: &[crate::generator::RowSample],
    ) -> Result<crate::generator::SampledStep, GpuModelError> {
        use crate::generator::{RowSample, SampledStep};
        let r = tokens.len();
        assert_eq!(plans.len(), r, "one plan per row");
        let (par, tpar) = Self::pack_samp_par(plans);
        let exec = self.exec.clone();
        let vocab = self.hp.n_vocab;
        let owned: Vec<u32> = (0..r as u32).collect();
        let ident: &[u32] = slots.unwrap_or(&owned);
        assert_eq!(ident.len(), r, "one slot per row");
        self.ensure_full_rows(ident, positions)?;
        self.upload_rows(tokens, positions, ident)?;
        {
            let sc = &mut self.batch.as_mut().expect("batch enabled").sc;
            let mut v = sc
                .d_par
                .try_slice_mut(0..r * 4)
                .ok_or_else(|| GpuError::Driver("d_par slice".into()))?;
            exec.stream.memcpy_htod(&par, &mut v).map_err(drv)?;
            if let Some(t) = &tpar {
                let mut v = sc
                    .d_tpar
                    .try_slice_mut(0..r * 4)
                    .ok_or_else(|| GpuError::Driver("d_tpar slice".into()))?;
                exec.stream.memcpy_htod(t, &mut v).map_err(drv)?;
            }
        }
        self.step_replay(r)?;
        // drafter warmth (see batch_step): append before the sample launch
        // reads back - all device work, no sync
        if self.dflash_armed() {
            self.dflash_append_features(positions, ident, None)?;
        }
        {
            let sc = &mut self.batch.as_mut().expect("batch enabled").sc;
            exec.sample_rows_at(&sc.head_logits, &sc.d_par, 0, &mut sc.d_out, 0, r, vocab)?;
            if tpar.is_some() {
                Self::trunc_dev5_witness(r);
                exec.sample_rows_t_at(
                    &sc.head_logits,
                    &sc.d_par,
                    0,
                    &sc.d_tpar,
                    0,
                    &mut sc.d_out,
                    0,
                    r,
                    vocab,
                )?;
                exec.sample_rows_p_at(
                    &sc.head_logits,
                    &sc.d_par,
                    0,
                    &sc.d_tpar,
                    0,
                    &mut sc.d_out,
                    0,
                    r,
                    vocab,
                )?;
            }
        }
        let sc = &self.batch.as_ref().expect("batch enabled").sc;
        let ids_view = sc
            .d_out
            .try_slice(0..r)
            .ok_or_else(|| GpuError::Driver("d_out slice".into()))?;
        let ids = exec.stream.clone_dtoh(&ids_view).map_err(drv)?;
        let mut host_rows = Vec::new();
        for (i, p) in plans.iter().enumerate() {
            if matches!(p, RowSample::Host) {
                let v = sc
                    .head_logits
                    .try_slice(i * vocab..(i + 1) * vocab)
                    .ok_or_else(|| GpuError::Driver("host row slice".into()))?;
                host_rows.push((i, exec.stream.clone_dtoh(&v).map_err(drv)?));
            }
        }
        Ok(SampledStep { ids, host_rows })
    }

    pub(crate) fn supports_decode_pipe_impl(&self) -> bool {
        self.batch.is_some()
            && self.exec.has_sample_rows()
            && self.exec.has_pipe_advance()
            && paddock_models::dev_var_os!("PADDOCK_NO_DECODE_PIPE").is_none()
            // DFlash armed: pipe ticks don't append drafter features (the
            // eager fuse+append chain between pipelined replays would defeat
            // the pipelining), so slots pipelined even once would go cold
            // with no re-warm path. Spec rounds replace the pipe's win here;
            // graph-captured appends are the follow-up that re-enables it.
            && !self.dflash_armed()
    }

    /// Enqueue one pipe tick: deterministic growth + mrope rebuild host-side,
    /// token advance on device (previous ring's sampled ids), graph replay,
    /// device sampling into this tick's ring, readiness event.
    fn pipe_launch_tick(
        &mut self,
        plans: &[crate::generator::RowSample],
        advance: bool,
    ) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let vocab = self.hp.n_vocab;
        let (b, tick, pos0, slot_map) = {
            let p = self.pipe.as_ref().expect("pipe active");
            (p.b, p.tick, p.pos0.clone(), p.slots.clone())
        };
        // back the position each row writes this tick (pos0[i] + tick) before
        // any pipe state mutates - the scheduler's headroom gate keeps the
        // pool ahead of worst-case tick growth, so this only PoolExhausts if
        // that gate is bypassed (then the abort below unwinds cleanly)
        let rows: Vec<u32> = match &slot_map {
            Some(s) => s.clone(),
            None => (0..b as u32).collect(),
        };
        let positions: Vec<u32> = pos0.iter().map(|&p| p + tick as u32).collect();
        self.ensure_full_rows(&rows, &positions)?;
        let ring = (tick % 2) as usize;
        let prev = ((tick + 1) % 2) as usize;
        let (par, tpar) = Self::pack_samp_par(plans);
        let mrope: Vec<u32> = (0..4).flat_map(|_| positions.iter().copied()).collect();
        let ns = self.batch.as_ref().expect("batch enabled").n_slots;
        {
            let sc = &mut self.batch.as_mut().expect("batch enabled").sc;
            let off = ring * ns * 4;
            let mut v = sc
                .d_par
                .try_slice_mut(off..off + b * 4)
                .ok_or_else(|| GpuError::Driver("pipe d_par slice".into()))?;
            exec.stream.memcpy_htod(&par, &mut v).map_err(drv)?;
            if let Some(t) = &tpar {
                let mut v = sc
                    .d_tpar
                    .try_slice_mut(off..off + b * 4)
                    .ok_or_else(|| GpuError::Driver("pipe d_tpar slice".into()))?;
                exec.stream.memcpy_htod(t, &mut v).map_err(drv)?;
            }
            // token = previous tick's sampled id, position += 1 - on device
            if advance {
                let (out, tok, pos) = (&sc.d_out, &mut sc.d_toks, &mut sc.d_pos);
                exec.pipe_advance(out, prev * ns, tok, pos, b)?;
            }
            // mrope rebuilt host-side (deterministic positions), stream-ordered
            // before the replay so the graph reads this tick's RoPE positions
            let mut m = sc
                .d_mrope
                .try_slice_mut(0..4 * b)
                .ok_or_else(|| GpuError::Driver("pipe d_mrope slice".into()))?;
            exec.stream.memcpy_htod(&mrope, &mut m).map_err(drv)?;
        }
        self.step_replay(b)?;
        {
            let sc = &mut self.batch.as_mut().expect("batch enabled").sc;
            exec.sample_rows_at(
                &sc.head_logits,
                &sc.d_par,
                ring * ns * 4,
                &mut sc.d_out,
                ring * ns,
                b,
                vocab,
            )?;
            // mode-5 rows draw into the same out ring - pipe_advance
            // feeds trunc ids forward exactly like mode-1/2 rows
            if tpar.is_some() {
                Self::trunc_dev5_witness(b);
                exec.sample_rows_t_at(
                    &sc.head_logits,
                    &sc.d_par,
                    ring * ns * 4,
                    &sc.d_tpar,
                    ring * ns * 4,
                    &mut sc.d_out,
                    ring * ns,
                    b,
                    vocab,
                )?;
                exec.sample_rows_p_at(
                    &sc.head_logits,
                    &sc.d_par,
                    ring * ns * 4,
                    &sc.d_tpar,
                    ring * ns * 4,
                    &mut sc.d_out,
                    ring * ns,
                    b,
                    vocab,
                )?;
            }
        }
        let ev = exec.record_event()?;
        self.pipe.as_mut().expect("pipe active").ev[ring] = Some(ev);
        Ok(())
    }

    /// Begin a pipelined pure-decode (all plans Device/Hole). No ids yet -
    /// the first `decode_pipe_next` returns tick 0's while tick 1 runs.
    pub(crate) fn decode_pipe_begin_impl(
        &mut self,
        slots: Option<&[u32]>,
        tokens: &[u32],
        positions: &[u32],
        plans: &[crate::generator::RowSample],
    ) -> Result<(), GpuModelError> {
        let b = tokens.len();
        assert_eq!(plans.len(), b, "one plan per row");
        assert_eq!(positions.len(), b, "one position per row");
        if !self.supports_decode_pipe_impl() {
            return Err(GpuModelError::Unsupported("decode pipe".into()));
        }
        let n_slots = self.batch.as_ref().expect("batch enabled").n_slots;
        assert!(b <= n_slots, "pipe rows {b} > enabled {n_slots}");
        if let Some(s) = slots {
            assert_eq!(s.len(), b, "one slot per row");
        }
        assert!(self.pipe.is_none(), "decode pipe already active");
        self.ensure_step_graph(b)?;
        let exec = self.exec.clone();
        // tick-0 inputs land in the fixed graph buffers. d_slots is always
        // written (prefill leaves its own slot vector in there).
        {
            let sc = &mut self.batch.as_mut().expect("batch enabled").sc;
            let mut vt = sc
                .d_toks
                .try_slice_mut(0..b)
                .ok_or_else(|| GpuError::Driver("pipe d_toks slice".into()))?;
            exec.stream.memcpy_htod(tokens, &mut vt).map_err(drv)?;
            let mut vp = sc
                .d_pos
                .try_slice_mut(0..b)
                .ok_or_else(|| GpuError::Driver("pipe d_pos slice".into()))?;
            exec.stream.memcpy_htod(positions, &mut vp).map_err(drv)?;
            let ident: Vec<u32> = (0..b as u32).collect();
            let sv = slots.unwrap_or(&ident);
            let mut vs = sc
                .d_slots
                .try_slice_mut(0..b)
                .ok_or_else(|| GpuError::Driver("pipe d_slots slice".into()))?;
            exec.stream.memcpy_htod(sv, &mut vs).map_err(drv)?;
        }
        self.pipe = Some(PipeState {
            b,
            tick: 0,
            ev: [None, None],
            pos0: positions.to_vec(),
            slots: slots.map(<[u32]>::to_vec),
        });
        if let Err(e) = self.pipe_launch_tick(plans, false) {
            self.pipe_abort();
            return Err(e);
        }
        Ok(())
    }

    /// Enqueue the next tick (token advance on device) and return the ids of
    /// the OLDEST in-flight tick, read via the side stream while the new tick
    /// executes.
    pub(crate) fn decode_pipe_next_impl(
        &mut self,
        plans: &[crate::generator::RowSample],
    ) -> Result<Vec<u32>, GpuModelError> {
        let exec = self.exec.clone();
        let (b, j) = {
            let p = self.pipe.as_ref().ok_or_else(|| {
                GpuModelError::Unsupported("decode_pipe_next without begin".into())
            })?;
            (p.b, p.tick)
        };
        assert_eq!(plans.len(), b, "one plan per row");
        self.pipe.as_mut().expect("pipe active").tick = j + 1;
        if let Err(e) = self.pipe_launch_tick(plans, true) {
            self.pipe_abort();
            return Err(e);
        }
        let ring = (j % 2) as usize;
        let ns = self.batch.as_ref().expect("batch enabled").n_slots;
        let r = {
            let sc = &self.batch.as_ref().expect("batch enabled").sc;
            let ev = self.pipe.as_ref().expect("pipe active").ev[ring]
                .as_ref()
                .expect("in-flight tick event");
            exec.to_host_u32_after(ev, &sc.d_out, ring * ns, b)
        };
        match r {
            Ok(ids) => Ok(ids),
            Err(e) => {
                self.pipe_abort();
                Err(e.into())
            }
        }
    }

    /// End the pipe: return the last in-flight tick's ids without enqueueing
    /// more. The fixed input buffers are left stale - every other path
    /// re-uploads them (upload_rows / pipe begin).
    pub(crate) fn decode_pipe_drain_impl(&mut self) -> Result<Vec<u32>, GpuModelError> {
        let exec = self.exec.clone();
        let st = self
            .pipe
            .take()
            .ok_or_else(|| GpuModelError::Unsupported("decode_pipe_drain without begin".into()))?;
        let ring = (st.tick % 2) as usize;
        let ns = self.batch.as_ref().expect("batch enabled").n_slots;
        let sc = &self.batch.as_ref().expect("batch enabled").sc;
        let ev = st.ev[ring].as_ref().expect("in-flight tick event");
        match exec.to_host_u32_after(ev, &sc.d_out, ring * ns, st.b) {
            Ok(ids) => Ok(ids),
            Err(e) => {
                let _ = exec.synchronize(); // state gone - quiesce ring readers
                Err(e.into())
            }
        }
    }

    /// Kill an in-flight pipe after an error (or on reset): quiesce the
    /// stream so nothing still reads the pipe buffers, then drop the state.
    pub(crate) fn pipe_abort(&mut self) {
        if self.pipe.take().is_some() {
            let _ = self.exec.synchronize();
        }
    }

    /// Phase-split timing probe for the r==1 decode tick (eager, no graph):
    /// runs each op group of the layer walk in an isolated ×`reps` loop with
    /// sync fences and prints the per-tick ms split. Activation VALUES are
    /// garbage in the isolated groups - only shapes/timing matter. Bench
    /// harness only (tests/gpu_laguna_profile.rs); the serving path never
    /// calls this. Requires enable_batch + at least one prefilled token in
    /// slot 0 so position/KV state is sane.
    pub fn profile_batch_tick(&mut self, pos: u32, reps: usize) -> Result<(), GpuModelError> {
        let exec = self.exec.clone();
        let hp_eps = self.hp.eps;
        let (embd, n_kv, hd) = (self.hp.n_embd, self.hp.n_kv_heads, self.hp.head_dim);
        let kv_dim = n_kv * hd;
        let scale = 1.0 / (hd as f32).sqrt();
        let sections = [self.hp.n_rot as u32 / 2, 0, 0, 0];
        let rope_full = self.hp.rope_full;
        let rope_swa = self.hp.rope_swa;
        let n_rot = self.hp.n_rot;
        let swa_window = self.hp.swa_window;
        let m = self.hp.moe;
        self.upload_rows(&[100], &[pos], &[0])?;

        fn timed(
            me: &mut GpuLaguna,
            name: &str,
            reps: usize,
            f: &mut dyn FnMut(&mut GpuLaguna) -> Result<(), GpuModelError>,
        ) -> Result<f64, GpuModelError> {
            let exec = me.exec.clone();
            exec.synchronize()?;
            let t = std::time::Instant::now();
            for _ in 0..reps {
                f(me)?;
            }
            exec.synchronize()?;
            let ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
            eprintln!("  {name:<32} {ms:8.3} ms/tick");
            Ok(ms)
        }

        eprintln!("laguna r=1 tick phase split ({reps} reps):");
        let total = timed(self, "FULL step_body", reps, &mut |me| me.step_body(1))?;
        // under a profiler, run only the real tick so the kernel sums aren't
        // polluted by the isolated sub-group loops
        if paddock_models::dev_var_os!("PADDOCK_PROBE_FULL_ONLY").is_some() {
            return Ok(());
        }
        let proj = timed(self, "qkvg GEMVs + norms + rope", reps, &mut |me| {
            let bs = me.batch.as_mut().expect("batch");
            for layer in &me.layers {
                let nh = layer.n_heads;
                let sc = &mut bs.sc;
                exec.rmsnorm_batch(&sc.x, &layer.attn_norm.buf, &mut sc.xn, embd, hp_eps, 1)?;
                gemv_any(&exec, &layer.wq, &sc.xn, &mut sc.q)?;
                gemv_any(&exec, &layer.wk, &sc.xn, &mut sc.k)?;
                gemv_any(&exec, &layer.wv, &sc.xn, &mut sc.v)?;
                gemv_any(&exec, &layer.g_proj, &sc.xn, &mut sc.gate_h)?;
                exec.rmsnorm_batch(&sc.q, &layer.q_norm.buf, &mut sc.qn, hd, hp_eps, nh)?;
                exec.rmsnorm_batch(&sc.k, &layer.k_norm.buf, &mut sc.kn, hd, hp_eps, n_kv)?;
                if layer.is_swa {
                    exec.rope_yarn_batch(&mut sc.qn, &sc.d_pos, nh, hd, rope_swa, 1)?;
                    exec.rope_yarn_batch(&mut sc.kn, &sc.d_pos, n_kv, hd, rope_swa, 1)?;
                } else {
                    exec.mrope(
                        &mut sc.qn,
                        &sc.d_mrope,
                        1,
                        nh,
                        hd,
                        n_rot,
                        rope_full,
                        sections,
                    )?;
                    exec.mrope(
                        &mut sc.kn,
                        &sc.d_mrope,
                        1,
                        n_kv,
                        hd,
                        n_rot,
                        rope_full,
                        sections,
                    )?;
                }
            }
            Ok(())
        })?;
        let attn = timed(self, "append + attend + wo", reps, &mut |me| {
            let bs = me.batch.as_mut().expect("batch");
            let bps = bs.bps;
            for (li, layer) in me.layers.iter().enumerate() {
                let nh = layer.n_heads;
                let (bt, window) = if layer.is_swa {
                    (&bs.swa_bt, swa_window)
                } else {
                    (&bs.d_bt, 0usize)
                };
                let kvs = &mut bs.kv[li];
                let sc = &mut bs.sc;
                exec.kv_append_batch_paged(
                    &sc.kn,
                    &mut kvs.k,
                    &sc.d_pos,
                    Some(&sc.d_slots),
                    bt,
                    bps,
                    kv_dim,
                    1,
                    me.kv_dtype,
                )?;
                exec.kv_append_batch_paged(
                    &sc.v,
                    &mut kvs.v,
                    &sc.d_pos,
                    Some(&sc.d_slots),
                    bt,
                    bps,
                    kv_dim,
                    1,
                    me.kv_dtype,
                )?;
                exec.attn_decode_batch_paged(
                    &sc.qn,
                    &kvs.k,
                    &kvs.v,
                    &sc.sinks,
                    &mut sc.attn,
                    &sc.d_pos,
                    Some(&sc.d_slots),
                    bt,
                    bps,
                    nh,
                    n_kv,
                    hd,
                    kv_dim,
                    window,
                    1,
                    scale,
                    me.kv_dtype,
                )?;
                exec.mul_softplus_head(&mut sc.attn, &sc.gate_h, nh, hd, 1)?;
                gemv_any(&exec, &layer.wo, &sc.attn, &mut sc.proj)?;
            }
            Ok(())
        })?;
        let moe = timed(self, "MoE routed (quant+route+experts)", reps, &mut |me| {
            let bs = me.batch.as_mut().expect("batch");
            for layer in &me.layers {
                let Ffn::Moe(w) = &layer.ffn else { continue };
                let sc = &mut bs.sc;
                exec.quantize_q8(&sc.xn, &mut sc.xq, &mut sc.xs, embd)?;
                exec.matvec_f32_batch(&w.router_w, &sc.xn, &mut sc.moe_logits, 1)?;
                exec.moe_topk_sigmoid_batch(
                    &sc.moe_logits,
                    &w.probs_bias.buf,
                    m.routed_scale,
                    m.n_expert,
                    m.n_active,
                    &mut sc.moe_idx,
                    &mut sc.moe_w,
                    1,
                )?;
                match (&w.gate_exps, &w.up_exps) {
                    (ExpW::Kq(g), ExpW::Kq(u)) => {
                        let needs =
                            crate::gpu::kq_needs_sums(g.ty) || crate::gpu::kq_needs_sums(u.ty);
                        if needs {
                            exec.q8_sums_strided(&sc.xq, &mut sc.ssums, embd, 1)?;
                        }
                        exec.kquant_moe_gate_up(
                            g,
                            u,
                            &sc.moe_idx,
                            &sc.xq,
                            &sc.xs,
                            needs.then_some(&sc.ssums),
                            &mut sc.moe_fused,
                            m.n_active,
                            1,
                        )?;
                    }
                    _ => unreachable!("XS election is k-quant experts"),
                }
                exec.quantize_q8(
                    &sc.moe_fused,
                    &mut sc.moe_fq,
                    &mut sc.moe_fs,
                    m.n_active * m.moe_ff,
                )?;
                match &w.down_exps {
                    ExpW::Kq(d) => {
                        let needs = crate::gpu::kq_needs_sums(d.ty);
                        if needs {
                            exec.q8_sums_strided(&sc.moe_fq, &mut sc.ssums, m.moe_ff, m.n_active)?;
                        }
                        exec.kquant_moe_down(
                            d,
                            &sc.moe_idx,
                            &sc.moe_w,
                            &sc.moe_fq,
                            &sc.moe_fs,
                            needs.then_some(&sc.ssums),
                            &mut sc.proj,
                            m.n_active,
                            1,
                        )?;
                    }
                    ExpW::Q8(_) => unreachable!("XS election is k-quant experts"),
                }
            }
            Ok(())
        })?;
        let shexp = timed(self, "shared expert (GEMV swiglu)", reps, &mut |me| {
            let bs = me.batch.as_mut().expect("batch");
            for layer in &me.layers {
                let Ffn::Moe(w) = &layer.ffn else { continue };
                let sc = &mut bs.sc;
                gemv_any(&exec, &w.shexp_gate, &sc.xn, &mut sc.sh_gate)?;
                gemv_any(&exec, &w.shexp_up, &sc.xn, &mut sc.sh_up)?;
                exec.swiglu(&mut sc.sh_gate, &sc.sh_up, m.shexp_ff)?;
                gemv_any(&exec, &w.shexp_down, &sc.sh_gate, &mut sc.sh_out)?;
            }
            Ok(())
        })?;
        let head = timed(self, "lm head GEMV", reps, &mut |me| me.head_rows(1))?;
        eprintln!(
            "  {:<32} {:8.3} ms/tick (groups {:.3})",
            "sum vs full",
            total,
            proj + attn + moe + shexp + head
        );

        // fine-grained sub-groups (overlapping with the groups above)
        timed(self, "  · qkvg GEMVs only", reps, &mut |me| {
            let bs = me.batch.as_mut().expect("batch");
            for layer in &me.layers {
                let sc = &mut bs.sc;
                gemv_any(&exec, &layer.wq, &sc.xn, &mut sc.q)?;
                gemv_any(&exec, &layer.wk, &sc.xn, &mut sc.k)?;
                gemv_any(&exec, &layer.wv, &sc.xn, &mut sc.v)?;
                gemv_any(&exec, &layer.g_proj, &sc.xn, &mut sc.gate_h)?;
            }
            Ok(())
        })?;
        timed(self, "  · qkg FUSED GEMV (+v)", reps, &mut |me| {
            let bs = me.batch.as_mut().expect("batch");
            for layer in &me.layers {
                let sc = &mut bs.sc;
                if let Some(qkg) = &layer.qkg {
                    exec.kquant_gemv(qkg, &sc.xn, &mut sc.q)?;
                    gemv_any(&exec, &layer.wv, &sc.xn, &mut sc.v)?;
                }
            }
            Ok(())
        })?;
        timed(self, "  · shexp FUSED (gu+swiglu+down)", reps, &mut |me| {
            let bs = me.batch.as_mut().expect("batch");
            for layer in &me.layers {
                let Ffn::Moe(w) = &layer.ffn else { continue };
                let sc = &mut bs.sc;
                if let Some(gu) = &w.shexp_gateup {
                    exec.kquant_gemv(gu, &sc.xn, &mut sc.sh_gate)?;
                    exec.swiglu_fused(&sc.sh_gate, &mut sc.sh_up, m.shexp_ff, 1)?;
                    gemv_any(&exec, &w.shexp_down, &sc.sh_up, &mut sc.sh_out)?;
                }
            }
            Ok(())
        })?;
        timed(self, "  · norms (attn+qk) only", reps, &mut |me| {
            let bs = me.batch.as_mut().expect("batch");
            for layer in &me.layers {
                let nh = layer.n_heads;
                let sc = &mut bs.sc;
                exec.rmsnorm_batch(&sc.x, &layer.attn_norm.buf, &mut sc.xn, embd, hp_eps, 1)?;
                exec.rmsnorm_batch(&sc.q, &layer.q_norm.buf, &mut sc.qn, hd, hp_eps, nh)?;
                exec.rmsnorm_batch(&sc.k, &layer.k_norm.buf, &mut sc.kn, hd, hp_eps, n_kv)?;
            }
            Ok(())
        })?;
        timed(self, "  · ropes only", reps, &mut |me| {
            let bs = me.batch.as_mut().expect("batch");
            for layer in &me.layers {
                let nh = layer.n_heads;
                let sc = &mut bs.sc;
                if layer.is_swa {
                    exec.rope_yarn_batch(&mut sc.qn, &sc.d_pos, nh, hd, rope_swa, 1)?;
                    exec.rope_yarn_batch(&mut sc.kn, &sc.d_pos, n_kv, hd, rope_swa, 1)?;
                } else {
                    exec.mrope(
                        &mut sc.qn,
                        &sc.d_mrope,
                        1,
                        nh,
                        hd,
                        n_rot,
                        rope_full,
                        sections,
                    )?;
                    exec.mrope(
                        &mut sc.kn,
                        &sc.d_mrope,
                        1,
                        n_kv,
                        hd,
                        n_rot,
                        rope_full,
                        sections,
                    )?;
                }
            }
            Ok(())
        })?;
        timed(self, "  · appends only", reps, &mut |me| {
            let bs = me.batch.as_mut().expect("batch");
            let bps = bs.bps;
            for (li, layer) in me.layers.iter().enumerate() {
                let bt = if layer.is_swa { &bs.swa_bt } else { &bs.d_bt };
                let kvs = &mut bs.kv[li];
                let sc = &mut bs.sc;
                exec.kv_append_batch_paged(
                    &sc.kn,
                    &mut kvs.k,
                    &sc.d_pos,
                    Some(&sc.d_slots),
                    bt,
                    bps,
                    kv_dim,
                    1,
                    me.kv_dtype,
                )?;
                exec.kv_append_batch_paged(
                    &sc.v,
                    &mut kvs.v,
                    &sc.d_pos,
                    Some(&sc.d_slots),
                    bt,
                    bps,
                    kv_dim,
                    1,
                    me.kv_dtype,
                )?;
            }
            Ok(())
        })?;
        timed(self, "  · attn kernels only", reps, &mut |me| {
            let bs = me.batch.as_mut().expect("batch");
            let bps = bs.bps;
            for (li, layer) in me.layers.iter().enumerate() {
                let nh = layer.n_heads;
                let (bt, window) = if layer.is_swa {
                    (&bs.swa_bt, swa_window)
                } else {
                    (&bs.d_bt, 0usize)
                };
                let kvs = &mut bs.kv[li];
                let sc = &mut bs.sc;
                exec.attn_decode_batch_paged(
                    &sc.qn,
                    &kvs.k,
                    &kvs.v,
                    &sc.sinks,
                    &mut sc.attn,
                    &sc.d_pos,
                    Some(&sc.d_slots),
                    bt,
                    bps,
                    nh,
                    n_kv,
                    hd,
                    kv_dim,
                    window,
                    1,
                    scale,
                    me.kv_dtype,
                )?;
            }
            Ok(())
        })?;
        timed(self, "  · wo GEMVs only", reps, &mut |me| {
            let bs = me.batch.as_mut().expect("batch");
            for layer in &me.layers {
                let sc = &mut bs.sc;
                gemv_any(&exec, &layer.wo, &sc.attn, &mut sc.proj)?;
            }
            Ok(())
        })?;
        timed(self, "  · moe expert kernels only", reps, &mut |me| {
            let bs = me.batch.as_mut().expect("batch");
            for layer in &me.layers {
                let Ffn::Moe(w) = &layer.ffn else { continue };
                let sc = &mut bs.sc;
                if let (ExpW::Kq(g), ExpW::Kq(u)) = (&w.gate_exps, &w.up_exps) {
                    let needs = crate::gpu::kq_needs_sums(g.ty) || crate::gpu::kq_needs_sums(u.ty);
                    exec.kquant_moe_gate_up(
                        g,
                        u,
                        &sc.moe_idx,
                        &sc.xq,
                        &sc.xs,
                        needs.then_some(&sc.ssums),
                        &mut sc.moe_fused,
                        m.n_active,
                        1,
                    )?;
                }
                if let ExpW::Kq(d) = &w.down_exps {
                    let needs = crate::gpu::kq_needs_sums(d.ty);
                    exec.kquant_moe_down(
                        d,
                        &sc.moe_idx,
                        &sc.moe_w,
                        &sc.moe_fq,
                        &sc.moe_fs,
                        needs.then_some(&sc.ssums),
                        &mut sc.proj,
                        m.n_active,
                        1,
                    )?;
                }
            }
            Ok(())
        })?;
        timed(self, "  · moe route+quant only", reps, &mut |me| {
            let bs = me.batch.as_mut().expect("batch");
            for layer in &me.layers {
                let Ffn::Moe(w) = &layer.ffn else { continue };
                let sc = &mut bs.sc;
                exec.quantize_q8(&sc.xn, &mut sc.xq, &mut sc.xs, embd)?;
                exec.matvec_f32_batch(&w.router_w, &sc.xn, &mut sc.moe_logits, 1)?;
                exec.moe_topk_sigmoid_batch(
                    &sc.moe_logits,
                    &w.probs_bias.buf,
                    m.routed_scale,
                    m.n_expert,
                    m.n_active,
                    &mut sc.moe_idx,
                    &mut sc.moe_w,
                    1,
                )?;
                exec.q8_sums_strided(&sc.xq, &mut sc.ssums, embd, 1)?;
                exec.quantize_q8(
                    &sc.moe_fused,
                    &mut sc.moe_fq,
                    &mut sc.moe_fs,
                    m.n_active * m.moe_ff,
                )?;
                exec.q8_sums_strided(&sc.moe_fq, &mut sc.ssums, m.moe_ff, m.n_active)?;
            }
            Ok(())
        })?;
        Ok(())
    }

    pub(crate) fn kv_mem_bytes_impl(&self) -> Option<u64> {
        self.batch.as_ref().map(|b| b.kv_bytes)
    }

    pub(crate) fn pool_free_blocks_impl(&self) -> Option<usize> {
        self.batch.as_ref().map(|b| b.pool.free_blocks())
    }
}

/// A prompt mid-prefill. `cursor` is the next prompt index to compute - it
/// starts at the prefix-resume point, not 0, so a radix hit costs no rows.
pub(crate) struct ChunkedPrefill {
    pub slot: usize,
    pub tokens: Vec<u32>,
    pub cursor: usize,
    /// prefix-resume point, kept for `prefix_cut` when the prompt finishes
    pub start: usize,
}
