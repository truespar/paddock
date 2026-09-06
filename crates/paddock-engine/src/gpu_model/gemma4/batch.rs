//! Continuous-batching lanes: per-slot KV, batched decode, per-slot prefill.
//!
//! Milestone shape: DENSE per-slot KV planes ([n_slots, max_ctx, kv_dim]
//! f16), batch capacity derived from free VRAM at enable time. The gpt-oss
//! paged-KV + SWA WindowRing machinery is the known next memory lever (dense
//! SWA planes cost max_ctx where a ring costs ~window - a ~4-8x KV saving at
//! serving contexts) - wire it after these lanes hold parity.
//!
//! Decode rows reuse the prefill scratch (rows ≤ slots « PF_ROWS) and the
//! repacked-f32 GEMMs (a decode batch never crosses the mmq 64-row floor).
//! The batched LM head runs on a repacked COPY of the tied embedding
//! (`token_embd_rep`, ~1.4 GB) - the raw Q8_0 plane stays for row gathers
//! and the single-row head, whose numerics the parity gate has locked.

use cudarc::driver::CudaSlice;

use crate::gpu::GpuError;

use std::sync::Arc;

use cudarc::driver::sys::CUstreamCaptureMode;

use crate::gpu::GpuExecutor;
use crate::kv_plan;

use super::forward::{PF_ROWS, g4_e4m3_glu, pf_mmq};

/// Issue-ahead: outcome of forward_mixed_spec_launch_impl.
pub(crate) enum MixLaunch {
    /// The round is enqueued; call forward_mixed_spec_wait_impl for picks.
    Launched,
    /// A path that already produced its result (decline, pure verify,
    /// pure prefill) - return as-is, nothing in flight.
    Fallback(
        Option<Vec<u32>>,
        Vec<(usize, crate::generator::FinishSample, usize)>,
    ),
}

/// The mixed round in flight between launch and wait.
pub(crate) struct MixInflight {
    nd: usize,
    row_plans: Vec<crate::generator::RowSample>,
    reqs: Vec<(usize, usize, Vec<u32>)>,
    rows_len: usize,
    mixed_k1: Option<usize>,
    fin_staged: Vec<bool>,
    /// per-batch-index device finisher plan: Some = the slot's
    /// finisher was staged AND sample_rows picks it on device - the wait
    /// half reads a 4-byte id instead of its [1, vocab] logits row
    fin_dev: Vec<Option<crate::sampler::DevicePlan>>,
    out: Vec<Vec<f32>>,
    batch: Vec<(usize, Vec<u32>)>,
}

/// F8A projection ladder: mma_ks twin through 31 rows (and 32..64 where the
/// TMA grid underfills, out < 16384 = under ~128 tiles on the 188-SM die),
/// TMA block-scale GEMM above. Textual macro so the disjoint scratch-field
/// borrows stay visible to the checker.
// fp4 weight-class lane switch: set by load.rs only when the planes really
// were converted (fp4 ladder present + both fusions live) - see load.rs
pub(crate) fn fp4_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var_os("PADDOCK_G4_FP4_ACTIVE").is_some())
}

// fused gu-epilogue (geglu+quant in the GEMM) kill switch - the interleave
// itself is killed separately at load (PADDOCK_G4_NO_GUIL); this one keeps
// the interleaved plane but falls back to the 2-launch geglu2i chain
pub(crate) fn gu_fuse_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_G4_NO_GUFUSE").is_none())
}

// fused rmsnorm->e4m3 kill switch (v11 norms rung)
pub(crate) fn nqfuse_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_G4_NO_NQFUSE").is_none())
}

// P47.L3: width cap for the row-fused norm+quant deferral.
// The old r<=64 bound was a decode-band election; the 65..128 verify
// band paid the unfused 3-kernel chain (rmsnorm_add_scale +
// rmsnorm_batch + row1p, 37.6us of spans/boundary) where one addnorm
// does the identical math - width-conditioned A/B: -0.3ms/tick at
// rows=128, comm/slot unmoved across 10 reps. The producer
// (next_fuses) and consumer (nqf_row_attn) must read the same cap or a
// deferred post strands. Truthy gate: PADDOCK_G4_NQF_WIDE=0 reverts.
pub(crate) fn nqf_wide_cap() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| match paddock_models::dev_var!("PADDOCK_G4_NQF_WIDE") {
        Ok(v) if v == "0" => 64,
        _ => 128,
    })
}

// f8t16: bf16 wo stream on the F8T chunk walk. The c16 machinery gates on
// f8w planes the unified loader doesn't build, so the f8t chunk route runs
// f32 glue, which is where the prefill band goes. The loader publishes
// PADDOCK_TC5R_O16_DIM/_IN so the pack's
// tc5r O16 arm fires on exactly the wo plane at batch >= 129; the wo
// consumers here read bf16 through the shipped p16 twins. Probe gate:
// PADDOCK_G4_F8T16=1 (default off until the battery).
pub(crate) fn f8t16_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_G4_F8T16").is_some())
}

// the loader-published wo in_dim the pack's O16 arm keys on - reading the
// same env keeps engine and pack elections agreeing by construction
// (per-layer hd differs: sliding vs global attention).
pub(crate) fn f8t16_wo_in() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("PADDOCK_TC5R_O16_IN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

// P50: the verify tick's last f32-D GEMM plane - qkv - onto the
// b16-D arm, read straight by the packed-bf16 nra3 twin (slot 420). Its own
// truthy gate so the plane prices alone in the A/B; same r>=65 spec-only
// floor as VB16.
pub(crate) fn vb16q_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| crate::envset::env_on("PADDOCK_G4_VB16Q"))
}

// P53 (slot 423): fin-e4 attention - SWA verify attention
// (n_kv==1: the fin CTA owns whole output rows) writes the wo-in e4m3
// rows + per-row scales in-kernel, bit-identical to fin followed by
// quantize_e4m3_row, and the standalone row-quant launch disappears.
// Truthy gate (=0 reverts once defaulted); scheduling-only otherwise.
pub(crate) fn fae4_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_on!("PADDOCK_G4_FAE4"))
}

/// P54: fin-e4s - fin stores the wo input as e4m3 at STATIC
/// scale 1.0 (ones xrs), killing the standalone wo-in row quantize.
/// Numerics class change (the per-row scale only moves the clip/denorm
/// cliffs), so opt-in and comm/PPL-gated.
pub(crate) fn fae4s_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_on!("PADDOCK_G4_FAE4S"))
}

// Chunk-band 16-bit streams kill switch: o16 GEMM epilogues +
// bf16-in glue twins on the prefill chunk walk. Class change (bf16-rounded
// intermediates, the rival's own stream class) - acceptance-gated, not
// bit-parity; the kill restores the f32 streams exactly.
/// Attention streams kill (PADDOCK_G4_NO_ATTN16): absent = the
/// f16 pf_qn/pf_attn planes ride every eligible mixed/prefill pass.
pub(crate) fn attn16_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_G4_NO_ATTN16").is_none())
}

pub(crate) fn chunk16_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_G4_NO_CHUNK16").is_none())
}

// spec-verify in-kernel finalize kill switch (door 3): FA at one split
// writing pf_attn directly, combine skipped. Chunk floor guards the dc4
// regression (few-chunk grids starve the die; splits still pay there).
pub(super) fn spec_fin_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_G4_NO_SPEC_FIN").is_none())
}

// sm_120 wide-band split election - FIN is the default. The older sp2/sp4
// election was measured on the pre-krs F8/SB route; with the krs-VR arm fin
// beats splits+combine at both classes on the bench (SWA 207.6 vs 224.4,
// GLB 307.8 vs 324.5) and never loses in a serve. The combine mass the
// splits carried was largely PDL-overlapped, so the win is small but
// uniform, and the round sheds 72 combine launches. PADDOCK_G4_SPEC_SP=1
// restores the split election for A/B; the legacy NO_SPEC_SP kill is
// subsumed (fin is the default it forced).
pub(super) fn spec_sp_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var_os!("PADDOCK_G4_SPEC_SP").is_some()
            && paddock_models::dev_var_os!("PADDOCK_G4_NO_SPEC_SP").is_none()
    })
}

// Attention at depth: the fin flip above is a SWA verdict, not a global-layer
// one. Re-benched at fin, the split arm still wins on GLB (R2-sp4+combine
// 289.3 vs fin 308.2us - the fin GLB grid is 128 CTAs / 0.68 waves with the
// walk latency exposed per CTA), and the fin penalty GROWS with ctx depth
// while a split GLB stays flat. SWA keeps FIN, its own measured winner.
// Kill: PADDOCK_NO_GLB_SP4 restores fin on both arms.
pub(super) fn glb_sp4_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_GLB_SP4").is_none())
}

// Seen-once lazy capture: decode-tick graphs capture on the
// SECOND sight of a gkey, not the first - the fresh-capture path costs
// ~6.4ms (pre-capture sync serializes the stream, then record +
// instantiate) and wave transitions mint one-off row counts that never
// replay. Kill: PADDOCK_NO_LAZY_CAP restores capture-on-first-sight.
pub(super) fn lazy_cap_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_LAZY_CAP").is_none())
}

/// debug: PADDOCK_NO_DECODE_GRAPH=1 runs every decode tick
/// EAGER - no capture, no replay. The qkv f8cut corruption reproduces
/// only where decode graphs capture (weights/GEMM/PDL all exonerated);
/// this is the final serve-element discriminator.
pub(super) fn decode_graphs_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_DECODE_GRAPH").is_none())
}

/// A/B kill for the r=1 ffn gate|up rung (see the FFN arm ladder): puts the
/// single-row decode back on `q8_0_gemm_mma_ks`, the older route.
/// Process-constant, so a captured decode graph bakes one arm and stays valid.
fn r1_gu_off() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_G4_NO_R1GU").is_some())
}

///  prefill-pass capture: PADDOCK_PF_CAP=<N> enables graph capture of
/// forward_prefill_batch_impl's embed+prefill_layers for SINGLE-RUN chunks up
/// to N rows (a steady-state c32 mixed tick carries one ~128-row prompt). The
/// value is the max chunk width to capture - bursts (r>N, or multi-prompt
/// chunks) stay eager, so the capture is a pure add-on that only fires on the
/// thin single-prompt ticks that dominate the c32 measurement. Off by default.
/// The PF_RUNS arm htod's a run table inside prefill_layers, which a capture
/// cannot bake by pointer - but that arm only engages when `spans.len() > 1`
/// (see `pf_runs_batched` in forward.rs), while this capture only engages on
/// SINGLE-run chunks, so the two are near-disjoint at runtime. The old global
/// `PADDOCK_PF_RUNS` refusal here was therefore far too coarse: it disabled the
/// capture on every tick just because the env was set, including the
/// single-prompt steady-state ticks where PF_RUNS does nothing. P55 measured
/// what that costs to work around - trading PF_RUNS away for the capture
/// (PF_RUNS=0 + PF_CAP=256) is **-2.6%**, because PF_RUNS earns its keep on the
/// multi-prompt burst chunks the capture never touches. The guard now lives at
/// the call site as `spans.len() <= 1`, which makes `pf_runs_batched` false by
/// construction inside the captured region (its registration is also scoped to
/// one prefill_layers call: set at forward.rs:920, cleared at :2264), so both
/// features can be on at once.
pub(super) fn pf_cap() -> Option<usize> {
    static V: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_PF_CAP")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n: &usize| (1..=PF_ROWS).contains(&n))
    })
}

// LCO opt-in (bit-equal on both arms): the krs spec-FA
// arms merge their split partials IN-KERNEL (last-CTA-out) and the separate
// combine launch disappears from the tick. Mirrors the pack's
// PADDOCK_SPEC_LCO gate so the engine skips its combine call exactly when
// the pack takes the LCO route.
pub(super) fn spec_lco_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_SPEC_LCO").is_some())
}

// PC route row floor: the chunk sites always clear it (r ~ 1952);
// lowering to 128 routes the r~160 pure-decode ticks through the scale-free
// twins too (their per-32 kt3/kt3g residue measured 0.15 ms/tok at churn).
// Default keeps the gated behavior; PADDOCK_G4_PC_FLOOR=128 is the A/B.
pub(super) fn pc_floor() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_G4_PC_FLOOR")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| (32..=4096).contains(&n))
            .unwrap_or(256)
    })
}

// kv-epilogue fold: the chunk band's kn/vn planes were pure
// round-trip - qkv_norm_rope_batch wrote them and the paged append read
// them straight back (134 MB/layer at the 2048-row SWA shape). Default on:
// pd_kv_nra_rows norms+ropes the RAW k/v GEMM planes into the caches and
// the v2 pass shrinks to q-only. Kill: PADDOCK_G4_NO_KVF.
pub(super) fn kvf_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_G4_NO_KVF").is_none())
}

// FIN position floor (default 0 = fin whenever the chunk floor holds):
// fin is a small win at short KV and ~flat at long, because the combine
// saving mostly cancels against the one-split walk running slower per
// launch. The floor stays as a tuning
// hook (it rides the decode-graph key band, so both variants capture
// cleanly) in case a losing regime shows up at other geometries.
pub(super) fn spec_fin_pos_floor() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_G4_SPEC_FIN_POS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

// Pos-thresholded LCO election: elect the in-kernel last-CTA combine only
// on pure ticks whose max position sits below this floor. Shallow walks win
// by the deleted launch and deep walks lose to the serialized last-CTA
// merge, which makes an unconditional LCO a wash.
// 0 = off; PADDOCK_SPEC_LCO stays the unconditional opt-in.
// Rides the decode-graph key like the FIN band (spec_shallow).
pub(super) fn spec_lco_pos_floor() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_SPEC_LCO_POS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

// Prefill overlap-lane row floor (see prefill_layers): chunks below this
// row count run the classic serial walk - at small r the cross-stream event
// edges cost more than the tiny work they hide, so an ungated overlap loses
// on short prompts and only pays on ~1.6k-row chunks.
pub(super) fn pf_overlap_min() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_G4_PF_OVERLAP_MIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1280)
    })
}

macro_rules! f8a_mm {
    ($exec:expr, $sc:expr, $w:expr, $y:expr, $ind:expr, $outd:expr, $r:expr) => {{
        if fp4_on() {
            // fp4 ladder (planes are e2m1 when the lane switch is set)
            if $r <= 31 || ($r <= 64 && $outd < 16384) {
                $exec.fp4_gemm_mma_ks(
                    $w,
                    &$sc.pf_e4q,
                    &$sc.pf_e4s,
                    &mut $sc.pf_skfix,
                    $y,
                    $ind,
                    $outd,
                    $r,
                )?;
            } else {
                $exec.mxfp4_gemm_bs($w, &$sc.pf_e4q, &$sc.pf_e4s, $y, $ind, $outd, $r)?;
            }
        } else if $r <= 31 || ($r <= 64 && $outd < 16384) {
            $exec.f8_gemm_mma_ks(
                $w,
                &$sc.pf_e4q,
                &$sc.pf_e4s,
                &mut $sc.pf_skfix,
                $y,
                $ind,
                $outd,
                $r,
            )?;
        } else {
            $exec.f8_gemm_w8($w, 0, &$sc.pf_e4q, &$sc.pf_e4s, $y, $ind, $outd, $r)?;
        }
    }};
}
use super::{GpuGemma4, LayerKv, LayerWeights, SwaPaging};

/// VRAM to leave untouched when sizing the batch: allocator slack plus the
/// CUDA context and cuBLAS workspaces, which live outside the stream-ordered
/// pool `process_mem_used` measures - better one slot fewer than an OOM
/// mid-serving.
///
/// This used to be 6 GiB, from back when it sized DENSE per-slot KV planes
/// and `vram_headroom` did not yet clamp to the utilization budget. It does
/// now, so 6 GiB was a second, blinder copy of the same 10% the budget
/// already withholds - and on gemma-4-31B at
/// ctx 4096 the doubled slack alone exceeded the whole grant, which is how a
/// 48 GB card ended up serving that model on the serial engine.
/// 1 GiB matches every other family.
const VRAM_HEADROOM: usize = 1 << 30;
/// Retained-prefix slack above the addressable ceiling, in slots' worth of
/// blocks: `min(slots, this) x blocks_per_slot`. See `enable_batch_impl`.
const RETENTION_SLOTS_MAX: usize = 8;

/// FlashDecoding split cap (gpt-oss convention).
pub(super) const MAX_ATTN_SPLITS: usize = 16;

/// Decode-attention split count: n_heads×batch blocks alone leave most SMs
/// idle (32 blocks on a 188-SM part at c1 - measured ~16 ms/step of
/// attention at 1k ctx vs <1 ms of KV bytes). Position-INDEPENDENT so the
/// captured graph can bake it per row count.
/// Scratch bound shared by every split election: `attn_scratch` holds
/// n_head × n_slots × MAX_ATTN_SPLITS rows and a launch consumes
/// rows × splits of that budget - splits beyond slots×MAX/rows write past
/// the plane. Found live: any --max-batch < 16 serve with 2
/// concurrent requests -> 16 verify rows × 16 dense splits vs a sub-16-slot
/// scratch = CUDA_ERROR_ILLEGAL_ADDRESS + poisoned server; batch-16 sat
/// exactly at the boundary and every bench config (32) sailed above it.
pub(super) fn attn_splits_cap(slots: usize, rows: usize) -> usize {
    ((slots * MAX_ATTN_SPLITS) / rows.max(1)).max(1)
}

pub(super) fn attn_splits(n_heads: usize, batch: usize, sm_count: usize, slots: usize) -> usize {
    // A/B surface for the combine door: force the split count (spec verify
    // walk and the unified decode arm both route through here). The verdict
    // it settled: splits=1 is CATASTROPHIC at every serve shape - the fused
    // single-pass arm is not competitive, and the split is load-bearing
    // (occupancy + serial-walk overlap). The combine cost the split buys
    // (~0.058 ms/tok, which a never-split kernel design doesn't pay) can only
    // be recovered IN-KERNEL by a cross-split reduction, not by election.
    // That verdict binds this election - dense arms plus the pre-krs spec
    // route. The spec WIDE-BAND now defaults to the krs-VR FIN route
    // (spec_sp_on), where the combine mass turned out to be largely
    // PDL-overlapped, so the fin flip is a small uniform win rather than the
    // band-sized one that recovering the whole combine would have been.
    let cap = attn_splits_cap(slots, batch);
    if let Some(f) = forced_attn_splits() {
        return f.min(cap);
    }
    // ceil, not floor: at r=8 (heads*batch=256 blocks on 188 SMs) the floor
    // yields 1 and split-K never engages, leaving each block to walk the
    // whole KV run latency-bound (global decode attn ~762us/layer).
    let blocks = (n_heads * batch).max(1);
    let fill = (sm_count * 2).div_ceil(blocks);
    // floor of 2 at r>=16: blocks alone fill the SMs there, but each block's
    // serial KV walk is latency-bound - at wide batch decode attention is
    // ~38% of GPU time at ~19x its own bandwidth floor, so a second split
    // halves the walk and doubles
    // the latency overlap even at full occupancy
    let floor = if batch >= 16 { 2 } else { 1 };
    fill.max(floor).clamp(1, MAX_ATTN_SPLITS).min(cap)
}

/// A/B knob for the split election above (falsified as a default-change
/// vehicle - see the verdict at the call site; kept for future kernel-door
/// gates). PADDOCK_ATTN_SPLITS=1 runs the fused single-pass decode arm (no
/// partial planes, no combine); values >1 pin the split count. Unset = the
/// measured election. Graph-safe: process-constant, graphs bake one grid.
fn forced_attn_splits() -> Option<usize> {
    static V: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_ATTN_SPLITS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|n: &usize| (1..=MAX_ATTN_SPLITS).contains(n))
    })
}

/// Dense-decode splits sized from the GQA kernel's real grid (n_kv, batch,
/// splits): the old heads*batch count left gemma4's 4-kv-head global layers
/// at 0.86 waves on a 148-SM die (the dense walk at ~30% of the GPU,
/// ~2.4 TB/s). ~3 waves of the 2-CTA/SM occupancy with
/// the serial-walk overlap floor of 2. Position-independent (graph-safe).
/// KV-aware split band: rides the decode-graph key so a band
/// change re-captures instead of replaying stale grids (the spec_long
/// precedent). 0 = pmax unknown (no clamp).
pub(super) fn kv_split_band(pmax: usize) -> usize {
    if pmax == 0 {
        0
    } else {
        pmax.div_ceil(128).clamp(2, MAX_ATTN_SPLITS)
    }
}

pub(super) fn attn_splits_kv(
    n_kv: usize,
    batch: usize,
    sm_count: usize,
    slots: usize,
    pmax: usize,
) -> usize {
    // Force surface. Worth having because this election has been caught
    // badly wrong before: at z=7 over a ~190-token live KV it spent 60.8us
    // per global-layer launch plus combine where an unsplit kernel needs
    // ~10us.
    static F: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    if let Some(f) = *F.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_ATTN_SPLITS_KV")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|n: &usize| (1..=MAX_ATTN_SPLITS).contains(n))
    }) {
        return f.min(attn_splits_cap(slots, batch)).max(1);
    }
    let xy = (n_kv * batch).max(1);
    let full = (sm_count * 6)
        .div_ceil(xy)
        .max(2)
        .clamp(2, MAX_ATTN_SPLITS)
        .min(attn_splits_cap(slots, batch));
    // KV-aware clamp. On a shallow KV (pmax <= 256) FEWER splits win: a
    // 5-point sweep put the optimum at roughly 128 tokens per split, and
    // z=1 is CATASTROPHIC - the split is load-bearing, never go below 2.
    // The full formula stays the long-KV ceiling so deep walks keep today's
    // splits exactly. pmax==0 (no host mirror yet) keeps the full formula.
    //
    // CAVEAT, muse-glimmer: that ordering INVERTS there - more splits win
    // monotonically, flattening around 6-8 - because of the different decode
    // stack (norm_wide_nth, f8t decode planes). So this clamp is tuned for
    // the gemma lane; the band's adaptive 2-3 is what muse gets, and it is
    // not muse's optimum. Re-tuning it per arch needs its own guarded A/B
    // across every batch rung, since one election serves them all.
    let band = kv_split_band(pmax);
    if band == 0 { full } else { full.min(band) }
}

/// Consumer-side K-split absorption. It only pays once the row kernels are
/// wide enough: on STARVED consumers it loses, because the nz-sum lands on a
/// 32-block geglu2 / r-block addnorm while the combine it replaced ran at
/// full grid. With the widened row kernels it is a small consistent win.
/// DEFAULT on; PADDOCK_NO_KSABS kills.
pub(super) fn ksabs_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_KSABS").is_none())
}

/// down-site extension of the absorption (wo/gu ship it already). Planes
/// carry across the layer boundary in pf_skfix; the next layer's fused attn
/// pre-norm consumes them (addnorm_e4m3_row_nz), same fixed-z sum ->
/// gate-identical. FALSIFIED as a default: unlike wo/gu, the replaced
/// combine overlapped the next layer's prologue while the absorbing
/// addnorm is on the serial path into qkv and reads nz x the bytes - at
/// c8's verify width that lands straight on the tick. OPT-IN:
/// PADDOCK_KSABS_DOWN=1.
pub(super) fn ksabs_down_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_KSABS_DOWN").is_some())
}

/// slot 458: Q16xKv128 tensor-core decode attention for the muse
/// hd128/G16 geometry - the trtllm-gen-class arm. FINAL output, sink folded,
/// no combine. Unlike `attn_fused16_arm` this carries no row gate (it wins at
/// every rung including B=1: bench/muse_fmha16_bench.cu, us/layer vs the
/// shipped vec8 splits=2 + combine pair - B=1/ctx128 13.60 -> 7.69, B=8/ctx256
/// 22.59 -> 10.25, B=32/ctx256 64.74 -> 12.11) and no context band gate (the
/// KV walk is KVT-chunked, so smem is constant in ctx).
///
/// OPT-IN (PADDOCK_ATTN_FMHA16=1) until the serve A/B and the acceptance
/// battery land. Two reasons to hold the default: this die has already shown
/// an isolated attention win turn into a live LOSS (see `attn_fused16_arm` -
/// the tick is ~97% die-occupied and vec8's tiny CTAs hide inside PDL
/// bubbles), and the arm is a NUMERICS CLASS CHANGE (relRMS 2.5e-6 vs the
/// production pair; K/V are exact but Q and P ride bf16 big+residual).
pub(super) fn attn_fmha16_arm(
    exec: &GpuExecutor,
    n_head: usize,
    n_kv: usize,
    hd: usize,
    dtype: crate::gpu::KvDtype,
) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_ATTN_FMHA16").is_some())
        && exec.has_attn_decode_fmha16()
        && dtype == crate::gpu::KvDtype::Fp8E4m3
        && hd == 128
        && n_kv * 16 == n_head
}

///  FIN gate: single-split + in-kernel finalize is only sound when a
/// FIN-capable kernel will actually serve the shape - v8<256,2(,F8)> or
/// v8ks<512,8> (f16 only until v8ks grows its fp8 arm). Mirrors the pack
///  election: the fused single-pass GQA-16 decode arm (slot 380) -
/// one CTA per (kv-head, row), whole windowed K/V smem-staged, FINAL output
/// with the sink folded in-kernel; no partials, no combine. Bench
/// (muse_dec_attn_bench.cu, B=32): clearly ahead of vec8 splits=2 + combine
/// at ctx 128..768 (64.89 -> 40.76 us at ctx 256, rel 2.4e-7; note it needs
/// ../build/cutgemm.o on the link line).
/// Loses at B<=16 (few fat CTAs, latency-bound) - rows >= 24 keeps vec8
/// everywhere it wins. The band gate (<= 6, i.e. pos_max <= 768) is both the
/// measured win region AND the smem cap; graphs bake per band so the arm is
/// capture-safe. The muse geometry only (fp8 KV, hd128, G16).
///
/// FALSIFIED as A DEFAULT - the isolated-bench law again: the live decode
/// tick runs it slower, because vec8+combine's thousands of tiny CTAs hide
/// inside the PDL cascade's bubbles while the weight-bound GEMM stream owns
/// the wall, whereas the fused kernel's staged prologue sits on the critical
/// path. OPT-IN via PADDOCK_ATTN_FUSED16=1.
///
/// Attention is this tick's one real lever, though - attributing kernels by
/// DIE-TIME (duration x min(1, CTAs/148)) rather than raw duration, which
/// over-credits part-die kernels, the SWA attention band dominates what is
/// left to win. The shape to beat is a single fused paged-KV
/// sliding-window kernel per layer at ~128 CTAs / 512 threads / ~8us, the
/// design TensorRT-LLM's generated fmha kernels use; ours is vec8
/// (2048-3072 CTAs, ~55us) plus a separate full-die combine (1024 CTAs,
/// ~27us). This proto does not reach that yet.
pub(super) fn attn_fused16_arm(
    exec: &GpuExecutor,
    n_head: usize,
    n_kv: usize,
    hd: usize,
    dtype: crate::gpu::KvDtype,
    rows: usize,
    pmax: usize,
) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_ATTN_FUSED16").is_some())
        && exec.has_attn_decode_fused_gqa16()
        && dtype == crate::gpu::KvDtype::Fp8E4m3
        && hd == 128
        && n_kv * 16 == n_head
        && rows >= 24
        && kv_split_band(pmax) <= 6
}

/// route's own env kills so a killed route can never leave the output
/// unnormalized. PADDOCK_NO_FIN1 kills the arm wholesale.
pub(super) fn fin1_ok(
    exec: &GpuExecutor,
    hd: usize,
    n_head: usize,
    n_kv: usize,
    dtype: crate::gpu::KvDtype,
) -> bool {
    static OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let envs_ok = *OK.get_or_init(|| {
        paddock_models::dev_var_os!("PADDOCK_NO_FIN1").is_none()
            && paddock_models::dev_var_os!("PADDOCK_NO_ATTN_V3").is_none()
            && paddock_models::dev_var_os!("PADDOCK_NO_ATTN_V5").is_none()
            && paddock_models::dev_var_os!("PADDOCK_NO_ATTN_V7").is_none()
            && paddock_models::dev_var_os!("PADDOCK_NO_ATTN_V8").is_none()
            && paddock_models::dev_var_os!("PADDOCK_NO_V8KS").is_none()
            && paddock_models::dev_var_os!("PADDOCK_NO_V8F8").is_none()
    });
    if !envs_ok || exec.compute_capability().0 < 9 {
        return false;
    }
    let fp8 = dtype == crate::gpu::KvDtype::Fp8E4m3;
    // hd-256 G=2 is covered for both KV dtypes (f16 and fp8-e4m3 arms exist);
    // the hd-512 G=8 global layer has only the f16 arm.
    (hd == 256 && n_head == 2 * n_kv) || (hd == 512 && n_head == 8 * n_kv && !fp8)
}

/// Split count for the k1-deep SPEC attention arm. Its grid is
/// (n_heads/gsub) x (rows/k1) x splits - nothing like the dense path's
/// heads x batch - and each block's KV walk is serial latency-bound tiles,
/// so splits are what buy walk overlap. Sized from the ACTUAL xy grid with
/// a floor of 4: on a wide B200 load splits=2 left the spec kernel at
/// 20.4 ms/step and 38% of the GPU, with the hd-512 global-layer variant at
/// ~1% of DRAM bandwidth; and at batch 1 the xy grid is 16 blocks, so the
/// dense-path heuristic starves a 148-SM die. Position-independent, so
/// captured graphs can bake it per row count.
#[allow(dead_code)]
pub(super) fn spec_attn_splits(n_heads: usize, chunks: usize, sm_count: usize) -> usize {
    let xy = (n_heads / 2).max(1) * chunks.max(1);
    ((sm_count * 8).div_ceil(xy)).clamp(4, MAX_ATTN_SPLITS)
}

/// Static WindowRing table for `slots`: logical block j of slot s -> pool
/// block s*ring + j%ring. Ring must survive one prefill chunk's appends
/// plus the window behind the chunk's first row (the oldest read).
pub(crate) fn build_swa_paging(
    exec: &Arc<GpuExecutor>,
    max_ctx: usize,
    swa_window: usize,
    slots: usize,
    span: usize,
) -> Result<SwaPaging, GpuError> {
    let bps = max_ctx.div_ceil(16);
    // the ring absorbs one SWA sub-span (plus a multimodal image-span
    // overshoot) before the window behind it, not a whole PF_ROWS chunk -
    // prefill_layers appends+attends SWA in `span` steps (2048 default:
    // 211 blocks at window 1024). `span` is whatever the caller elected and
    // must be the same value the cutters use, or the ring aliases blocks
    // the window still needs.
    let ring = ((span + super::forward::IMG_SPAN_MAX + swa_window).div_ceil(16) + 1).min(bps);
    let mut bt = vec![0u32; slots * bps];
    for s in 0..slots {
        for j in 0..bps {
            bt[s * bps + j] = (s * ring + (j % ring)) as u32;
        }
    }
    Ok(SwaPaging {
        bt: exec.to_device_u32(&bt)?,
        bps,
        ring,
    })
}

/// KV allocation shared by load (slots=1) and enable_batch: SWA layers get
/// ring POOLS when paging is on, dense planes otherwise; global layers are
/// dense unless `global_pool_blocks` is set (enable_batch budget-pool mode),
/// in which case every global layer's k/v is a shared pool of that many
/// 16-token blocks addressed through the global block table.
pub(crate) fn alloc_kv(
    exec: &Arc<GpuExecutor>,
    layers: &[LayerWeights],
    max_ctx: usize,
    paging: Option<&SwaPaging>,
    global_pool_blocks: Option<usize>,
    slots: usize,
    dtype_pref: Option<crate::gpu::KvDtype>,
) -> Result<Vec<LayerKv>, GpuError> {
    let mut kv = Vec::with_capacity(layers.len());
    // fp8 (e4m3) KV cache is the DEFAULT: 74.1 GiB of cache where f16 needs
    // 110.6, at near-identical throughput now that the fp8 kernel arms cover
    // the whole ladder (v9q decode, spec-FA verify, pf5-family prefill), and
    // coherent at every serving gate (attention class relRMS ~2.3%).
    //
    // The SWITCH: `set_kv_dtype(Fp16)` (the shared per-family setter) or
    // PADDOCK_G4_KV16=1 restores the f16 KV cache (the old default)
    // - use it to A/B quality or to rule out the fp8 cache when debugging
    // generation issues. PADDOCK_G4_KV8=1 is the old opt-in, kept as an
    // accepted no-op for script compat. The chosen mode is announced in the
    // serve log at load ("gemma4 kv cache: ..."). fp8 requires the pooled
    // serving stack; non-pooled setups stay f16 - including under an
    // explicit fp8 request, which is announced as overridden rather than
    // silently honored-in-name-only.
    let kv16 = matches!(dtype_pref, Some(crate::gpu::KvDtype::Fp16))
        || std::env::var_os("PADDOCK_G4_KV16").is_some();
    let kv8 = !kv16 && global_pool_blocks.is_some();
    tracing::info!(
        "gemma4 kv cache: {}",
        if kv8 {
            "fp8-e4m3 (default; --kv-cache-dtype f16 / PADDOCK_G4_KV16=1 restores f16)"
        } else if kv16 {
            "f16 (explicit request)"
        } else if matches!(dtype_pref, Some(crate::gpu::KvDtype::Fp8E4m3)) {
            "f16 (OVERRIDING the explicit fp8 request: non-pooled setup, fp8 needs the pooled stack)"
        } else {
            "f16 (non-pooled setup; fp8 needs the pooled stack)"
        }
    );
    for lw in layers {
        let kv_dim = lw.n_kv_heads * lw.head_dim;
        let dtype = if kv8 {
            crate::gpu::KvDtype::Fp8E4m3
        } else {
            crate::gpu::KvDtype::Fp16
        };
        let eb = dtype.bytes();
        let bytes = match (paging, global_pool_blocks) {
            (Some(pg), _) if lw.is_swa => slots * pg.ring * 16 * kv_dim * eb,
            (_, Some(n)) if !lw.is_swa => n * 16 * kv_dim * eb,
            _ => slots * max_ctx * kv_dim * eb,
        };
        // dim-major twin (same byte count as v) for SWA fp8
        // layers when the VD probe is on
        let vdim = if lw.is_swa
            && matches!(dtype, crate::gpu::KvDtype::Fp8E4m3)
            && paddock_models::dev_var_os!("PADDOCK_VDIM").is_some()
        {
            Some(exec.alloc_u8(bytes)?)
        } else {
            None
        };
        kv.push(LayerKv {
            dtype,
            k: exec.alloc_u8(bytes)?,
            v: exec.alloc_u8(bytes)?,
            vdim,
            kv_dim,
        });
    }
    Ok(kv)
}

impl GpuGemma4 {
    /// Reallocate the KV planes for `max_batch` slots (VRAM permitting) and
    /// Select the KV cache element type - the same setter contract every
    /// other family exposes (`--kv-cache-dtype` reaches it via serving.rs's
    /// `apply_kv_dtype`). gemma4's DEFAULT is inverted vs those families
    /// (fp8-e4m3 when pooled -), so `Fp16` here is the lossless
    /// opt-out and `Fp8E4m3` mostly re-states the default; the actual dtype
    /// election stays in `alloc_kv` (fp8 needs the pooled stack - a
    /// non-pooled setup announces the override rather than honoring fp8 in
    /// name only). The historical `PADDOCK_G4_KV16` env switch remains
    /// honored alongside this. Call before serving: the choice lands on the
    /// next `enable_batch`'s re-alloc (the load-time slots=1 alloc is
    /// always non-pooled f16 regardless).
    pub fn set_kv_dtype(&mut self, dtype: crate::gpu::KvDtype) {
        self.kv_dtype_pref = Some(dtype);
    }

    /// return the capacity actually enabled. Existing cache contents drop -
    /// the engine only enables batching before admitting sequences.
    pub(crate) fn enable_batch_impl(&mut self, max_batch: usize) -> Result<usize, GpuError> {
        // drop the previous prefix cache + pool up front so the VRAM math
        // below sees their memory as free (both rebuild at the end)
        self.prefix = None;
        self.gpool = None;
        // Global budget pool (G4a shape): requires the paged kernels (SWA
        // paging on implies they exist) - PADDOCK_NO_GLOBAL_POOL pins the
        // dense escape hatch for A/B.
        let pooled = self.paging.is_some()
            && paddock_models::dev_var_os!("PADDOCK_NO_GLOBAL_POOL").is_none();
        // The estimator must charge the ACTUAL kv element size (mirrors
        // alloc_kv's dtype pick): it kept f16 after the fp8-e4m3
        // default landed, a 2x overcharge on per_slot AND block_bytes that
        // let slots soak the doubled phantom cost and starved the global
        // pool to its floor (~30 GB VRAM unused, 1703-block pool, c32
        // preempt waves at 2048-token prompts - the same
        // estimator-vs-alloc drift class as qwen's 256k conv-ext fix).
        // Mirror alloc_kv's pick exactly, `set_kv_dtype` preference included
        // and not just its env twin. The two agree today only because
        // --kv-cache-dtype f16 happens to set both, which is a coincidence of
        // two call sites rather than a guarantee.
        let kv16 = matches!(self.kv_dtype_pref, Some(crate::gpu::KvDtype::Fp16))
            || std::env::var_os("PADDOCK_G4_KV16").is_some();
        let kv_eb: usize = if !kv16 && pooled {
            1 // fp8-e4m3 (the pooled-stack default)
        } else {
            2 // f16
        };
        let bps = self.max_ctx.div_ceil(16);
        // Per-slot SWA ring cost as a function of the sub-span we prefill in.
        // The ring has to absorb one span (plus an image overshoot) ahead of
        // the window before older blocks may alias, so the span is the
        // per-slot price: at window 1024 the 2048 rung holds 3376 positions
        // to retain 1024, the 512 rung holds 1824. Dense (unpaged) mode
        // ignores it - the planes are full-context either way.
        let per_slot_for = |span: usize| -> usize {
            let ring_pos = self.paging.as_ref().map(|_| {
                ((span + super::forward::IMG_SPAN_MAX + self.hp.swa_window).div_ceil(16) + 1)
                    .min(bps)
                    * 16
            });
            self.layers
                .iter()
                .map(|l| {
                    let positions = if l.is_swa {
                        ring_pos.unwrap_or(self.max_ctx)
                    } else if pooled {
                        0 // global KV comes from the shared pool, not the slot
                    } else {
                        self.max_ctx
                    };
                    positions * l.n_kv_heads * l.head_dim * 2 * kv_eb // K+V
                })
                .sum()
        };
        // per-BLOCK cost of the global pool: every global layer backs the
        // same logical block (16 positions, K+V)
        let block_bytes: usize = self
            .layers
            .iter()
            .filter(|l| !l.is_swa)
            .map(|l| 16 * l.n_kv_heads * l.head_dim * 2 * kv_eb)
            .sum();
        // Reserves carved out of the slot-fit budget in pooled mode. The
        // prefix checkpoint blobs build_prefix allocates after us are a flat
        // charge; the pool floor is not - charging a full 1536-block floor
        // while solving for the width lets a nice-to-have retention pool veto
        // a slot the box could actually serve, and you cannot cache prefixes
        // for sessions you have no room to run. So the fit charges only what
        // the live slots must have (their own global KV); whatever is left
        // over still lands in the pool as retention below.
        let pool_floor_blocks = 1536usize;
        // read the dtype off the CURRENT planes, before they are freed
        let ckpt_reserve = if pooled {
            self.prefix_vram_estimate()
        } else {
            0
        };
        // Hand the load-time serial planes back before measuring. They are
        // ~2.9 GiB on the 31B and alloc_kv replaces them wholesale; the old
        // code left them resident and credited an ESTIMATE of them back
        // instead, which on a pooled server is the (much smaller) batch-shaped
        // ring rather than the dense full-context planes actually being
        // dropped. Free-then-measure needs no correction term and cannot
        // drift from what the allocator really did.
        self.kv = Vec::new();
        self.exec
            .stream
            .synchronize()
            .map_err(|e| GpuError::Driver(e.to_string()))?;
        self.exec.trim_mem_pool();
        // budget-aware headroom (device free clamped to vram_budget - ledger):
        // slots + the global pool must size inside this runner's granted slice
        let grant = self
            .exec
            .vram_headroom()
            .ok_or_else(|| GpuError::Driver("cuMemGetInfo gave no free-VRAM reading".into()))?;
        // One arbiter sizes the KV store: crate::kv_plan. gemma4's own
        // arithmetic was already budget-correct - this is the same solve, moved
        // somewhere a new family cannot forget to do it. The SWA-span LADDER stays
        // here because which sub-span to prefill in is a gemma4 question; the
        // planner only answers "does that rung still seat the whole ask".
        let demand_for = |span: usize, slots: usize| kv_plan::Demand {
            family: "gemma4",
            max_ctx: self.max_ctx,
            slots,
            // Dense mode has no shared pool at all: every layer's plane is
            // per-slot and already priced into per_slot_bytes, so the pool's
            // addressable ceiling is zero.
            blocks_per_slot: if pooled { bps } else { 0 },
            block_bytes: block_bytes as u64,
            per_slot_bytes: per_slot_for(span) as u64,
            // Admission slack for radix retention (nodes hold blocks after
            // their sequence ends), sized by DEMAND: one retained context per
            // seated slot, capped at the eight every pool got before
            // 2026-09-06. The flat 8 x bps let a one-slot 131k server fill the
            // whole grant - a 43.8 GiB pool holding 1.15M tokens, nine times
            // its own context, 84.6 GiB on the card for one user of the 31B.
            // From eight slots up nothing changes.
            retention_blocks: if pooled {
                slots.min(RETENTION_SLOTS_MAX) * bps
            } else {
                0
            },
            // Reserves carved out of the slot-fit budget in pooled mode. The
            // prefix checkpoint blobs build_prefix allocates after us are a flat
            // charge; the pool floor is not - charging a full 1536-block floor
            // while solving for the width lets a nice-to-have retention pool veto
            // a slot the box could actually serve, and you cannot cache prefixes
            // for sessions you have no room to run. The planner clamps this floor
            // to what is addressable, so the fit charges only what the live slots
            // must have (their own global KV); whatever is left over still lands
            // in the pool as retention below.
            floor_blocks_min: if pooled { pool_floor_blocks } else { 0 },
            reserves: {
                let mut r = vec![
                    kv_plan::Reserve::new("graph/scratch slack", VRAM_HEADROOM as u64),
                    kv_plan::Reserve::new("prefix checkpoints", ckpt_reserve as u64),
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
        // Elect the widest span rung that still seats the whole ask. An operator
        // pin, or a dense (unpaged) setup where the span buys nothing, skips the
        // election entirely.
        if self.paging.is_some() && super::forward::swa_span_pin().is_none() {
            let ladder: Vec<usize> = super::forward::SWA_SPAN_LADDER
                .iter()
                .copied()
                .filter(|&s| s <= self.swa_span)
                .collect();
            let narrowest = ladder.last().copied().unwrap_or(self.swa_span);
            let elected = ladder
                .iter()
                .copied()
                .find(|&s| {
                    // seats the whole ask: a rung the planner has to narrow to
                    // fit has not seated it
                    demand_for(s, max_batch)
                        .plan(grant)
                        .is_ok_and(|p| p.slots == max_batch)
                })
                .unwrap_or(narrowest);
            // Only narrow when narrowing actually buys per-slot bytes. A family
            // with no SWA layers prices every rung the same, and cutting its
            // prefill into shorter sub-spans then costs churn (~9% measured) for
            // nothing.
            if elected != self.swa_span && per_slot_for(elected) < per_slot_for(self.swa_span) {
                tracing::info!(
                    "gemma4 SWA sub-span {} -> {} rows: the wider ring does not leave room to \
                     batch {max_batch} slots at ctx {} ({:.2} vs {:.2} GiB/slot)",
                    self.swa_span,
                    elected,
                    self.max_ctx,
                    per_slot_for(self.swa_span) as f64 / (1u64 << 30) as f64,
                    per_slot_for(elected) as f64 / (1u64 << 30) as f64,
                );
                self.swa_span = elected;
            }
        }
        let demand = demand_for(self.swa_span, max_batch);
        // Say the arithmetic out loud whenever the answer is narrower than asked
        // - Plan::report does, at WARN. A silent drop to 1 slot is a serving MODE
        // change (the service falls back to the serial engine), and the duel
        // board's biggest loss was exactly that with nothing on-screen to
        // attribute it to.
        //
        // plan_or_minimum, not plan: the load-time serial planes are already gone
        // (freed above so the measurement needs no correction term), so an Err
        // here would hand the caller a model with no key-value cache at all. Take
        // the smallest runnable shape and let alloc_kv refuse honestly instead -
        // its restore path below puts the 1-slot serial planes back.
        let plan = demand.plan_or_minimum(grant);
        plan.report(&demand, grant);
        let slots = plan.slots;

        if self.paging.is_some() {
            self.paging = Some(build_swa_paging(
                &self.exec,
                self.max_ctx,
                self.hp.swa_window,
                slots,
                self.swa_span,
            )?);
        }
        let pool_blocks = if pooled { Some(plan.pool_blocks) } else { None };
        self.kv = match alloc_kv(
            &self.exec,
            &self.layers,
            self.max_ctx,
            self.paging.as_ref(),
            pool_blocks,
            slots,
            self.kv_dtype_pref,
        ) {
            Ok(kv) => kv,
            Err(e) => {
                // The serial planes are already gone, so leaving `kv` empty
                // here would hand the caller a model with no key-value cache
                // at all: the service's width backstop gives up at w<=2 by
                // breaking to the serial engine without another enable_batch
                // call, and that engine would then decode against nothing.
                // Put the load-time 1-slot shape back before reporting the
                // failure, so the fallback the caller is about to take is
                // still a working one.
                tracing::warn!(
                    "gemma4: batch KV alloc failed at {slots} slots ({e}); \
                                restoring the 1-slot serial planes"
                );
                if self.paging.is_some() {
                    self.paging = Some(build_swa_paging(
                        &self.exec,
                        self.max_ctx,
                        self.hp.swa_window,
                        1,
                        self.swa_span,
                    )?);
                }
                self.kv = alloc_kv(
                    &self.exec,
                    &self.layers,
                    self.max_ctx,
                    self.paging.as_ref(),
                    None,
                    1,
                    None,
                )?;
                self.n_slots = 1;
                return Err(e);
            }
        };
        self.gpool = match pool_blocks {
            Some(n) => Some(super::GlobalPaging {
                pool: crate::kv_pool::KvPool::with_blocks(n as u32),
                tables: (0..slots)
                    .map(|_| crate::kv_pool::BlockTable::new())
                    .collect(),
                bt_host: vec![0u32; slots * bps],
                d_bt: self
                    .exec
                    .alloc_u32(slots * bps)
                    .map_err(|e| GpuError::Driver(e.to_string()))?,
                bps,
            }),
            None => None,
        };
        if let Some(n) = pool_blocks
            && paddock_models::dev_var_os!("PADDOCK_POOL_STATS").is_some()
        {
            tracing::info!(
                "gemma4 pool: {slots} slots, {n} global blocks ({:.1} GB), per-slot rings {:.2} GB",
                (n * block_bytes) as f64 / (1 << 30) as f64,
                (plan.slot_bytes / slots.max(1) as u64) as f64 / (1 << 30) as f64,
            );
        }
        self.n_slots = slots;
        self.spec_rows = vec![None; slots];
        self.decode_graphs.clear(); // reallocated buffers invalidate captures
        self.graph_seen.clear();
        self.prefill_graphs.clear(); // same buffer-realloc invalidation
        self.prefill_graph_seen.clear();
        self.mtp_graphs.clear();
        // FlashDecoding scratch: [n_head*slots*MAX_SPLITS] × (head_dim + 2)
        {
            let hd_max = self.layers.iter().map(|l| l.head_dim).max().unwrap_or(512);
            let rows = self.hp.n_head * max_batch.max(1) * MAX_ATTN_SPLITS;
            self.attn_scratch = Some((
                self.exec
                    .stream
                    .alloc_zeros::<f32>(rows * hd_max)
                    .map_err(|e| GpuError::Driver(e.to_string()))?,
                self.exec
                    .stream
                    .alloc_zeros::<f32>(rows * 2)
                    .map_err(|e| GpuError::Driver(e.to_string()))?,
            ));
            // LCO arrival tickets: one u32 per (kvh, chunk) grid
            // cell, upper-bounded by n_head x the widest verify round. The
            // kernel's atomicInc wraps at n_splits-1, so zero once here and
            // never again - replays and graph captures stay consistent.
            self.lco_tickets = Some(
                self.exec
                    .stream
                    .alloc_zeros::<u32>(
                        self.hp.n_head * max_batch.max(1) * super::spec::SPEC_K1_MAX,
                    )
                    .map_err(|e| GpuError::Driver(e.to_string()))?,
            );
        }
        // WIDE-BATCH SPEC: with a drafter attached, spec verify rows are
        // slots*(K+1) (every live slot contributes pending + drafts) - the
        // tick-path buffers must hold the whole round, not one row per
        // slot. The 268 MB logits plane (32*8*262k f32) is the price of
        // running spec at wide batch, and it is worth it: this drafter is
        // designed to pay at every width.
        //
        // DFlash counts too: its round is one block-diffusion forward
        // instead of a k-step chain, but the VERIFY it feeds is the same
        // ragged multi-row tick. Both drafters attach at load, before the
        // service calls enable_batch, so the flag is settled by here.
        let vrows = if self.mtp.is_some() || self.dflash.is_some() {
            slots * super::spec::SPEC_K1_MAX
        } else {
            slots
        };
        if self.d_tokens.as_ref().map(|t| t.len()) != Some(vrows) {
            self.d_tokens = Some(
                self.exec
                    .alloc_u32(vrows)
                    .map_err(|e| GpuError::Driver(e.to_string()))?,
            );
        }
        if self.d_slots.as_ref().map(|t| t.len()) != Some(vrows) {
            self.d_slots = Some(
                self.exec
                    .alloc_u32(vrows)
                    .map_err(|e| GpuError::Driver(e.to_string()))?,
            );
        }
        if self.samp.as_ref().map(|(p, _)| p.len()) != Some(vrows * 4) {
            self.samp = Some((
                self.exec
                    .alloc_u32(vrows * 4)
                    .map_err(|e| GpuError::Driver(e.to_string()))?,
                self.exec
                    .alloc_u32(vrows)
                    .map_err(|e| GpuError::Driver(e.to_string()))?,
            ));
        }
        if self.samp_tpar.as_ref().map(|t| t.len()) != Some(vrows * 4) {
            self.samp_tpar = Some(
                self.exec
                    .alloc_u32(vrows * 4)
                    .map_err(|e| GpuError::Driver(e.to_string()))?,
            );
        }
        if self.d_pipe_out.as_ref().map(|b| b.len()) != Some(2 * vrows) {
            self.d_pipe_out = Some(
                self.exec
                    .alloc_u32(2 * vrows)
                    .map_err(|e| GpuError::Driver(e.to_string()))?,
            );
        }
        if self.batch_logits.as_ref().map(|b| b.len()) != Some(vrows * self.hp.n_vocab) {
            self.batch_logits = Some(
                self.exec
                    .stream
                    .alloc_zeros::<f32>(vrows * self.hp.n_vocab)
                    .map_err(|e| GpuError::Driver(e.to_string()))?,
            );
        }
        // prefix cache dead last: the d_ckpt blob used to sit
        // between the KV planes and every scratch/logits buffer above - its
        // mere PRESENCE there cost ~2.5% at salted 128x128c32 (presence-
        // triggered, size-independent: ckpts=1 == ckpts=2). At the tail,
        // everything the decode tick touches keeps the same addresses the
        // prefix-off config gets; only d_ckpt itself moves. (Pooled mode
        // shares pool blocks zero-copy; the reserve above covered the blob.)
        self.build_prefix(max_batch)?;
        // DFlash rings + row bands: they need the slot count, and they must
        // exist before the first walk so the taps are live from tick one (a
        // tick that runs untapped is a hole in the ring, not a slow start).
        // Dropped and rebuilt with the rest of the batch state, since the
        // ring geometry is per-slot.
        if let Some(d) = self.dflash.as_mut() {
            d.state = None;
        }
        self.dflash_ensure_state()?;
        // VRAM ledger tail (matches the load-time phase marks): the batch
        // rebuild replaces the serial KV with per-slot rings + the global
        // pool and adds the batched scratch - the whole remaining footprint.
        if let Some(b) = self.exec.process_mem_used() {
            tracing::info!(
                "gemma4 vram: batch KV+scratch ({slots} slots)  (total {:7.2} GiB)",
                b as f64 / (1u64 << 30) as f64,
            );
        }
        Ok(slots)
    }

    /// Grow the global-pool block tables so every `(slots[i], positions[i])`
    /// this pass touches is backed by a physical block, then re-upload the
    /// device table once - outside any captured graph, into the buffer the
    /// graph baked by pointer, so replays read the fresh mapping (gpt-oss
    /// G4a shape). On a dry pool, LRU prefix leaves are evicted before
    /// giving up with `PoolExhausted` (the scheduler preempts on it).
    pub(crate) fn ensure_global_rows(
        &mut self,
        slots: &[u32],
        positions: &[u32],
    ) -> Result<(), GpuError> {
        let Some(gp) = self.gpool.as_mut() else {
            return Ok(());
        };
        let prefix = &mut self.prefix;
        let mut grew = false;
        for (i, &s) in slots.iter().enumerate() {
            let s = s as usize;
            let pos = positions[i] as usize;
            // a pos past the per-slot table stride can never be
            // served - bt_host/d_bt are bps-strided - and used to OVERFLOW
            // bt_host (panic at the copy below, index == len; found by the
            // packed hold32 admission wave). Fail the pass loudly
            // instead: the tick errors, the requests fail, the server lives,
            // and the warn names the culprit for the root-cause.
            if pos >= gp.bps * 16 {
                tracing::warn!(
                    "[table-cap] slot={s} pos={pos} exceeds bps*16={} - refusing the pass",
                    gp.bps * 16
                );
                return Err(GpuError::Driver(format!(
                    "slot {s} pos {pos} exceeds the {}-block table stride",
                    gp.bps
                )));
            }
            let before = gp.tables[s].blocks().len();
            loop {
                match gp.tables[s].ensure(pos, &mut gp.pool) {
                    Ok(()) => break,
                    Err(_) => {
                        // tier-aware shed, cliff-grade (make_room_blocking:
                        // one press + 50ms died while a parked restore's
                        // loads held the lane). Window blobs DEMOTE here too
                        // - the old recycle-only shortcut discarded every
                        // checkpoint the dry path evicted, leaving the tier
                        // restore-blind on this family's busiest eviction
                        // path (found live on the pooled smoke;
                        // demote_aux only SUBMITS, nothing stalls).
                        let evicted = match prefix.as_mut() {
                            Some(pf) => {
                                let exec = self.exec.clone();
                                let state = Some(pf.tier_state_geom(&exec.stream));
                                match pf.tier.as_mut() {
                                    Some(tier) => {
                                        let want = gp.pool.free_blocks() + 1;
                                        tier.make_room_blocking(
                                            &mut pf.radix,
                                            &mut gp.pool,
                                            want,
                                            state,
                                            &mut || exec.record_event().ok(),
                                        )
                                        .then_some(0u32)
                                    }
                                    None => pf.radix.evict_lru(&mut gp.pool),
                                }
                            }
                            None => None,
                        };
                        if evicted.is_none() {
                            // exhaustion forensics: name the holders once -
                            // a runaway table or a pinned radix shows here
                            let mut sizes: Vec<(usize, usize)> = gp
                                .tables
                                .iter()
                                .enumerate()
                                .map(|(k, t)| (k, t.blocks().len()))
                                .collect();
                            sizes.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
                            let total: usize = sizes.iter().map(|&(_, n)| n).sum();
                            tracing::warn!(
                                "[pool-exh] want slot={s} pos={pos} tables_total={total} top={:?}",
                                &sizes[..sizes.len().min(6)]
                            );
                            return Err(GpuError::PoolExhausted);
                        }
                    }
                }
            }
            let now = gp.tables[s].blocks().len();
            if now > before {
                grew = true;
                let base = s * gp.bps;
                for j in before..now {
                    gp.bt_host[base + j] = gp.tables[s].blocks()[j];
                }
            }
        }
        if grew {
            self.exec
                .stream
                .memcpy_htod(&gp.bt_host, &mut gp.d_bt)
                .map_err(|e| GpuError::Driver(e.to_string()))?;
        }
        Ok(())
    }

    /// Return `slot`'s global-pool blocks to the free-list before a fresh
    /// sequence regrows it (prefill start). No-op outside pool mode.
    pub(crate) fn gpool_clear_slot(&mut self, slot: usize) {
        if let Some(gp) = self.gpool.as_mut() {
            gp.tables[slot].clear(&mut gp.pool);
        }
    }

    /// Free-on-completion: release the global-pool blocks of every slot no
    /// longer holding a live sequence (P5b). Stale device-table entries are
    /// harmless - a freed slot decodes nothing until its next prefill
    /// re-publishes a fresh mapping.
    pub(crate) fn release_inactive_slots_impl(&mut self, occupied: &[bool]) {
        let Some(gp) = self.gpool.as_mut() else {
            return;
        };
        for (k, occ) in occupied.iter().enumerate() {
            if !occ && k < gp.tables.len() && !gp.tables[k].blocks().is_empty() {
                gp.tables[k].clear(&mut gp.pool);
            }
        }
    }

    /// Prefill a whole prompt into `slot` and return the last token's
    /// logits. Resumes from the prefix cache when a checkpointed prefix
    /// matches (positions [start..len) re-prefill; [0..start) restores),
    /// stages a window checkpoint at the prompt's last full page boundary,
    /// and inserts the prompt's pages for the next turn.
    pub(crate) fn forward_prefill_impl(
        &mut self,
        slot: usize,
        tokens: &[u32],
    ) -> Result<Vec<f32>, GpuError> {
        assert!(
            slot < self.n_slots,
            "slot {slot} >= enabled {}",
            self.n_slots
        );
        // pool mode: this slot starts a fresh sequence - return its old
        // blocks, adopt any cached prefix, then back the whole prompt
        self.gpool_clear_slot(slot);
        let start = self.prefix_resume(slot, tokens)?;
        self.ensure_global_rows(&[slot as u32], &[(tokens.len() - 1) as u32])?;
        let cut = self.prefix_cut(tokens.len(), start);
        let fill = vec![slot as u32; self.pf_rows];
        self.exec
            .stream
            .memcpy_htod(&fill, &mut self.scratch.pf_slots)
            .map_err(|e| GpuError::Driver(e.to_string()))?;
        // no segment split at the cut: the checkpoint lands straight from
        // the ring at insert (the ring's sub-span slack keeps the cut's
        // window resident through the ≤16-token tail)
        let mut base = start;
        let mut last_len = 0usize;
        for chunk in tokens[start..].chunks(self.pf_rows) {
            self.prefill_chunk(chunk, base)?;
            base += chunk.len();
            last_len = chunk.len();
        }
        self.prefix_insert(slot, tokens, cut)?;
        // restore the single-stream slot-0 staging before anyone else prefills
        let zeros = vec![0u32; self.pf_rows];
        self.exec
            .stream
            .memcpy_htod(&zeros, &mut self.scratch.pf_slots)
            .map_err(|e| GpuError::Driver(e.to_string()))?;
        self.logits_from_pf_row(last_len - 1)
    }

    /// COALESCED multi-prompt prefill: all pending prompts' rows concatenate
    /// into shared PF_ROWS chunks (one weight-amortized pass instead of one
    /// pass per prompt - the c8 TTFT was 8 sequential prefills ≈ 4.3 s).
    /// Per-row slots/positions drive the shared ops; attention dispatches
    /// per same-slot run (prefill_layers `runs`). Prefix cache: per-item
    /// resume up front, per-item snapshot+insert after the pass (the rings
    /// still hold each prompt's tail window - rings are per slot).
    /// the capturable core of the prefill pass - embed gather +
    /// preamble + the layer walk. pf_toks / pf_pos / pf_slots must already
    /// hold this chunk's inputs (the caller uploads them outside any capture,
    /// so a replay reads fresh ids/positions/slots by pointer). Everything
    /// here is device-only: embd_preamble is one rmsnorm, and prefill_layers
    /// at decode_rows=0 forks no side lane (overlap needs decode_rows>=16) and
    /// htod's nothing unless PADDOCK_PF_RUNS - which pf_cap() excludes - so a
    /// THREAD_LOCAL capture over it records only kernel + by-pointer nodes.
    fn pf_embed_layers(
        &mut self,
        r: usize,
        runs: &[(usize, usize)],
        spans: &[(usize, usize)],
    ) -> Result<(), GpuError> {
        let n_embd = self.hp.n_embd;
        {
            let sc = &mut self.scratch;
            self.exec.embed_gather_plane(
                &self.token_embd,
                &sc.pf_toks,
                &mut sc.pf_x,
                n_embd,
                r,
                self.hp.embd_scale(),
            )?;
            super::GpuGemma4::embd_preamble(
                &self.exec,
                &self.hp,
                self.embd_ones.as_ref(),
                &mut sc.pf_x,
                r,
            )?;
        }
        self.prefill_layers(r, runs, spans, 0)
    }

    pub(crate) fn forward_prefill_batch_impl(
        &mut self,
        items: &[(usize, Vec<u32>)],
    ) -> Result<Vec<Vec<f32>>, GpuError> {
        let n_embd = self.hp.n_embd;
        let mut starts = Vec::with_capacity(items.len());
        for (slot, toks) in items {
            assert!(
                *slot < self.n_slots,
                "slot {slot} >= enabled {}",
                self.n_slots
            );
            self.gpool_clear_slot(*slot);
            starts.push(self.prefix_resume(*slot, toks)?);
            self.ensure_global_rows(&[*slot as u32], &[(toks.len() - 1) as u32])?;
        }
        // row stream: (slot, pos, token, item) - items stay contiguous
        let mut rows: Vec<(u32, u32, u32, usize)> = Vec::new();
        let mut last_row = vec![0usize; items.len()];
        for (it, ((slot, toks), &start)) in items.iter().zip(&starts).enumerate() {
            for (j, &t) in toks[start..].iter().enumerate() {
                rows.push((*slot as u32, (start + j) as u32, t, it));
            }
            last_row[it] = rows.len() - 1;
        }
        let _row_bytes = (n_embd / 32) * 34;
        let mut out: Vec<Vec<f32>> = vec![Vec::new(); items.len()];
        let mut fin_staged = vec![false; items.len()];
        let mut base = 0usize;
        for chunk in rows.chunks(self.pf_rows) {
            let r = chunk.len();
            let positions: Vec<u32> = chunk.iter().map(|x| x.1).collect();
            let slots_v: Vec<u32> = chunk.iter().map(|x| x.0).collect();
            // contiguous same-slot runs for the per-run attention dispatch
            let mut runs: Vec<(usize, usize)> = Vec::new();
            for (i, x) in chunk.iter().enumerate() {
                match runs.last_mut() {
                    Some((off, n)) if chunk[*off].0 == x.0 => *n += 1,
                    _ => runs.push((i, 1)),
                }
            }
            {
                let sc = &mut self.scratch;
                let e = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
                self.exec
                    .stream
                    .memcpy_htod(&positions, &mut sc.pf_pos)
                    .map_err(e)?;
                self.exec
                    .stream
                    .memcpy_htod(&positions, &mut sc.pf_attn_pos)
                    .map_err(e)?;
                self.exec
                    .stream
                    .memcpy_htod(&slots_v, &mut sc.pf_slots)
                    .map_err(e)?;
            }
            {
                // token ids -> pf_toks. Host admission: stays outside any
                // capture so the embed node reads fresh ids by pointer on
                // every replay (the ONE-kernel gather itself moved into
                // pf_embed_layers, the capturable core).
                let toks: Vec<u32> = chunk.iter().map(|x| x.2).collect();
                let sc = &mut self.scratch;
                let mut v = sc
                    .pf_toks
                    .try_slice_mut(0..r)
                    .ok_or_else(|| GpuError::Driver("pf_toks slice".into()))?;
                self.exec
                    .stream
                    .memcpy_htod(&toks, &mut v)
                    .map_err(|e| GpuError::Driver(e.to_string()))?;
            }
            let spans = super::forward::swa_spans(self.swa_span, &runs);
            //  prefill-pass capture: a single-run chunk within the cap
            // replays a captured embed+layer graph instead of ~hundreds of
            // eager per-layer launches (the steady-state c32 mixed-tick tax).
            // pmax is baked into the key so a graph never runs over a different
            // KV extent; first sight of a shape runs eager, a recurrence
            // captures (lazy, exactly like decode_graphs).
            // `spans.len() <= 1` is the capture-safety guard (see pf_cap): it
            // makes forward.rs's `pf_runs_batched` false for this chunk, so no
            // host memcpy or run-table registration happens inside the region
            // being captured - which is what lets PADDOCK_PF_RUNS stay ON.
            let pf_replayed = match pf_cap() {
                Some(capn) if runs.len() == 1 && spans.len() <= 1 && r <= capn => {
                    let pmax = positions.iter().copied().max().unwrap_or(0) as usize;
                    let key = (r, pmax);
                    if let Some(g) = self.prefill_graphs.get(&key) {
                        g.0.launch()
                            .map_err(|e| GpuError::Driver(format!("pf graph launch: {e}")))?;
                    } else if lazy_cap_on() && self.prefill_graph_seen.insert(key) {
                        // first sight: run eager, mint the shape
                        self.pf_embed_layers(r, &runs, &spans)?;
                    } else {
                        // recurrence: capture a live pass, then replay+insert
                        self.exec
                            .stream
                            .synchronize()
                            .map_err(|e| GpuError::Driver(format!("pf pre-capture sync: {e}")))?;
                        self.exec
                            .stream
                            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                            .map_err(|e| GpuError::Driver(format!("pf begin_capture: {e}")))?;
                        let rec = self.pf_embed_layers(r, &runs, &spans);
                        let graph = crate::gpu::end_capture_no_flags(&self.exec.stream)
                            .map_err(|e| GpuError::Driver(format!("pf end_capture: {e}")));
                        rec?;
                        let graph = graph?.ok_or_else(|| {
                            GpuError::Driver("pf capture produced no graph".into())
                        })?;
                        let g = super::SendGraph(graph);
                        g.0.launch()
                            .map_err(|e| GpuError::Driver(format!("pf first launch: {e}")))?;
                        self.prefill_graphs.insert(key, g);
                    }
                    true
                }
                _ => false,
            };
            if !pf_replayed {
                self.pf_embed_layers(r, &runs, &spans)?;
            }
            //  diagnostic: the actual prefill-chunk shape distribution
            // (are steady-state mixed ticks single-run and within the cap?)
            if paddock_models::dev_var_os!("PADDOCK_PF_CAP_DEBUG").is_some() {
                let pmax = positions.iter().copied().max().unwrap_or(0);
                tracing::info!(
                    "[pfchunk] r={r} runs={} pmax={pmax} replayed={pf_replayed} cap={:?}",
                    runs.len(),
                    pf_cap()
                );
            }
            // DFlash: this chunk's taps are in the accumulator - fuse and
            // ring-append before the next chunk overwrites them.
            let toks: Vec<u32> = chunk.iter().map(|x| x.2).collect();
            self.dflash_append_features(&toks, &positions, &slots_v, None)?;
            // items whose last row landed in this chunk read their logits now
            // (the next chunk overwrites pf_x)
            let fin_here: Vec<(usize, usize)> = last_row
                .iter()
                .enumerate()
                .filter(|&(it, &lr)| lr >= base && lr < base + r && it < 64)
                .map(|(it, &lr)| (lr - base, it))
                .collect();
            // batched head: N finishers in one m=N chain - needs identity
            // out_idx order (true when every item finishes in this chunk,
            // the burst-pass case) and the f8t head
            let batchable =
                fin_here.len() > 4 && fin_here.iter().enumerate().all(|(i, &(_, it))| i == it);
            if batchable {
                self.logits_head_stage_batch(&fin_here)?;
                for &(_, it) in &fin_here {
                    fin_staged[it] = true;
                }
            } else {
                for &(lr, it) in &fin_here {
                    self.logits_head_stage(lr, it)?;
                    fin_staged[it] = true;
                }
            }
            for (it, &lr) in last_row.iter().enumerate() {
                if lr >= base && lr < base + r && it >= 64 {
                    out[it] = self.logits_from_pf_row(lr - base)?;
                }
            }
            base += r;
        }
        // per-item radix insert + direct ring->pool checkpoint landing
        for (it, (slot, toks)) in items.iter().enumerate() {
            let cut = self.prefix_cut(toks.len(), starts[it]);
            self.prefix_insert(*slot, toks, cut)?;
        }
        // restore the single-stream slot-0 staging convention
        let zeros = vec![0u32; self.pf_rows];
        self.exec
            .stream
            .memcpy_htod(&zeros, &mut self.scratch.pf_slots)
            .map_err(|e| GpuError::Driver(e.to_string()))?;
        // One sync + one dtoh for every staged finisher (the per-item
        // logits_finish_read loop paid ~2.6ms of blocking readback per
        // item past the old 8-slot cap - 24 x 2.6ms of the c32 burst pass)
        for (it, l) in self
            .logits_finish_read_all(&fin_staged)?
            .into_iter()
            .enumerate()
        {
            if let Some(l) = l {
                out[it] = l;
            }
        }
        Ok(out)
    }

    /// One batched decode step: row b = slot b, `tokens[b]` at
    /// `positions[b]`. Leaves [rows, vocab] logits on device in
    /// `batch_logits` (uncapped) - the host/sampled lanes tail differently.
    pub(crate) fn batch_step(&mut self, tokens: &[u32], positions: &[u32]) -> Result<(), GpuError> {
        let r = tokens.len();
        assert_eq!(r, positions.len());
        assert!(r <= self.n_slots, "rows {r} > enabled {}", self.n_slots);
        let ident: Vec<u32> = (0..r as u32).collect();
        self.ensure_global_rows(&ident, positions)?;
        self.batch_upload(tokens, positions, &ident)?;
        self.batch_step_body(r)
    }

    /// Host->device inputs for one decode step (never inside a graph capture:
    /// replays re-read these buffers). `slots_map[i]` = the KV slot row i
    /// decodes into (identity for the slot-dense phase-3 tick; explicit for
    /// the compacted mixed tick, where chunking slots are absent).
    fn batch_upload(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        slots_map: &[u32],
    ) -> Result<(), GpuError> {
        self.attn_pos_max = positions.iter().copied().max().unwrap_or(0) as usize;
        let sc = &mut self.scratch;
        let e = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
        self.exec
            .stream
            .memcpy_htod(positions, &mut sc.pf_pos)
            .map_err(e)?;
        {
            let d_slots = self.d_slots.as_mut().expect("enable_batch allocates");
            let mut v = d_slots
                .try_slice_mut(0..slots_map.len())
                .ok_or_else(|| GpuError::Driver("d_slots slice".into()))?;
            self.exec.stream.memcpy_htod(slots_map, &mut v).map_err(e)?;
        }
        // Async spec round: the verify tokens were assembled on
        // device (pd_spec_toks from the drafter's output plane) - the host
        // token values here are placeholders; skip the copy.
        if self.toks_dev {
            self.toks_dev = false;
            return Ok(());
        }
        let d_tokens = self.d_tokens.as_mut().expect("enable_batch allocates");
        let mut v = d_tokens
            .try_slice_mut(0..tokens.len())
            .ok_or_else(|| GpuError::Driver("d_tokens slice".into()))?;
        self.exec.stream.memcpy_htod(tokens, &mut v).map_err(e)?;
        Ok(())
    }

    /// Gemma4 decode pipe: depth-2 tick pipeline over the same
    /// (r, k1=1) captured graphs the dense path replays. Tick N+1's inputs
    /// advance on device (pipe_advance: token <- d_out, pf_pos += 1); each
    /// tick's ids persist into the d_pipe_out ring before the next launch
    /// overwrites d_out, and the host reads them back on the side stream
    /// gated by a post-copy event - so commit/SSE/launch-API host work all
    /// overlap the in-flight tick. Identity slot map (rows = slots, holes
    /// ride as (0,0) rows exactly like the classic dense phase).
    pub(crate) fn supports_decode_pipe_impl(&self) -> bool {
        self.exec.has_pipe_advance()
            && self.d_tokens.is_some()
            && paddock_models::dev_var_os!("PADDOCK_NO_DECODE_PIPE").is_none()
    }

    /// TruncCat rows execute fully on device (slot 435, mode 5 in the
    /// captured tick body) - the service may emit truncation plans for this
    /// backend, and the pipes/spec rounds may take them. Old packs without
    /// the kernel -> false -> truncation rows keep the Host readback path.
    pub(crate) fn device_trunc_supported(&self) -> bool {
        self.batch_logits.is_some()
            && self.samp.is_some()
            && self.exec.has_sample_rows_t()
            && self.exec.has_sample_rows_p()
    }

    fn pipe_launch(
        &mut self,
        plans: &[crate::generator::RowSample],
        advance: bool,
    ) -> Result<(), GpuError> {
        use crate::generator::RowSample;
        use crate::sampler::DevicePlan;
        let exec = self.exec.clone();
        let (r, tick) = {
            let p = self.pipe.as_ref().expect("pipe active");
            (p.r, p.tick)
        };
        let ring = (tick % 2) as usize;
        // per-tick sampler params (greedy rows re-upload the same bytes)
        let mut par = vec![0u32; r * 4];
        let mut tpar = vec![0u32; r * 4];
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
                // device truncation mode 5: the replayed tick body already carries the
                // pd_sample_rows_t launch - trunc rows stay zero-host
                // inside the pipe (service admits them via
                // supports_device_trunc + the P67b opt-in)
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
                // RS rows stay mode 0 (untouched): the resolve kernel
                // writes their ids after the tick
                RowSample::Device(DevicePlan::RsVerify { .. })
                | RowSample::Device(DevicePlan::RsTrunc { .. }) => {}
            }
        }
        {
            let (d_par, _) = self.samp.as_mut().expect("sampler buffers");
            let mut v = d_par
                .try_slice_mut(0..r * 4)
                .ok_or_else(|| GpuError::Driver("samp par slice".into()))?;
            exec.stream
                .memcpy_htod(&par, &mut v)
                .map_err(|e| GpuError::Driver(e.to_string()))?;
        }
        if any_trunc {
            let d_tpar = self.samp_tpar.as_mut().expect("allocated with samp");
            let mut v = d_tpar
                .try_slice_mut(0..r * 4)
                .ok_or_else(|| GpuError::Driver("samp tpar slice".into()))?;
            exec.stream
                .memcpy_htod(&tpar, &mut v)
                .map_err(|e| GpuError::Driver(e.to_string()))?;
        }
        // pool rows for this tick's positions (host mirror: pos0 + tick),
        // before the graph replay reads the block tables
        {
            let pos: Vec<u32> = {
                let p = self.pipe.as_ref().expect("pipe active");
                p.pos0.iter().map(|&x| x + tick as u32).collect()
            };
            self.attn_pos_max = pos.iter().copied().max().unwrap_or(0) as usize;
            let slots_id: Vec<u32> = (0..r as u32).collect();
            self.ensure_global_rows(&slots_id, &pos)?;
        }
        if advance {
            // token <- previous tick's sampled id, pf_pos += 1 - all device
            let (_, d_out) = self.samp.as_ref().expect("sampler buffers");
            let d_tok = self.d_tokens.as_mut().expect("enable_batch");
            exec.pipe_advance(d_out, 0, d_tok, &mut self.scratch.pf_pos, r)?;
        }
        // replay the same (r, k1=1, short-band) graph the dense path uses;
        // the 5th element is the KV-split band - a band change re-captures
        // rather than replaying stale attention grids
        let gkey = (r, 1usize, false, false, kv_split_band(self.attn_pos_max));
        if let Some(g) = self.decode_graphs.get(&gkey) {
            g.0.launch()
                .map_err(|e| GpuError::Driver(format!("pipe graph launch: {e}")))?;
        } else {
            exec.stream
                .synchronize()
                .map_err(|e| GpuError::Driver(format!("pipe pre-capture sync: {e}")))?;
            exec.stream
                .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                .map_err(|e| GpuError::Driver(format!("pipe begin_capture: {e}")))?;
            let rec = self.sampled_tick_body(r);
            let graph = crate::gpu::end_capture_no_flags(&exec.stream)
                .map_err(|e| GpuError::Driver(format!("pipe end_capture: {e}")));
            rec?;
            let graph =
                graph?.ok_or_else(|| GpuError::Driver("pipe capture produced no graph".into()))?;
            let g = super::SendGraph(graph);
            g.0.launch()
                .map_err(|e| GpuError::Driver(format!("pipe first launch: {e}")))?;
            self.decode_graphs.insert(gkey, g);
        }
        // persist this tick's ids into the ring before the next launch
        // overwrites d_out; the event gates the side-stream readback
        {
            let half = self.d_pipe_out.as_ref().expect("enable_batch").len() / 2;
            let (_, d_out) = self.samp.as_ref().expect("sampler buffers");
            let ring_buf = self.d_pipe_out.as_mut().expect("enable_batch");
            exec.copy_region(d_out, 0, ring_buf, ring * half, r)?;
        }
        let ev = exec.record_event()?;
        self.pipe.as_mut().expect("pipe active").ev[ring] = Some(ev);
        Ok(())
    }

    pub(crate) fn decode_pipe_begin_impl(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        plans: &[crate::generator::RowSample],
    ) -> Result<(), GpuError> {
        let r = tokens.len();
        assert_eq!(plans.len(), r, "one plan per row");
        assert_eq!(positions.len(), r, "one position per row");
        if !self.supports_decode_pipe_impl() || self.batch_logits.is_none() {
            return Err(GpuError::Driver("decode pipe unsupported".into()));
        }
        let cap_rows = self
            .batch_logits
            .as_ref()
            .map(|b| b.len() / self.hp.n_vocab)
            .unwrap_or(0);
        let half = self.d_pipe_out.as_ref().map(|b| b.len() / 2).unwrap_or(0);
        if r == 0 || r > cap_rows || r > half || r > self.n_slots {
            return Err(GpuError::Driver(format!("pipe rows {r} out of range")));
        }
        assert!(self.pipe.is_none(), "decode pipe already active");
        let slots_id: Vec<u32> = (0..r as u32).collect();
        self.batch_upload(tokens, positions, &slots_id)?;
        self.pipe = Some(super::G4Pipe {
            r,
            tick: 0,
            pos0: positions.to_vec(),
            ev: [None, None],
        });
        if let Err(e) = self.pipe_launch(plans, false) {
            self.pipe_abort();
            return Err(e);
        }
        Ok(())
    }

    pub(crate) fn decode_pipe_next_impl(
        &mut self,
        plans: &[crate::generator::RowSample],
    ) -> Result<Vec<u32>, GpuError> {
        let exec = self.exec.clone();
        let (r, prev_tick) = {
            let p = self
                .pipe
                .as_ref()
                .ok_or_else(|| GpuError::Driver("decode_pipe_next without begin".into()))?;
            (p.r, p.tick)
        };
        assert_eq!(plans.len(), r, "one plan per row");
        self.pipe.as_mut().expect("pipe active").tick = prev_tick + 1;
        if let Err(e) = self.pipe_launch(plans, true) {
            self.pipe_abort();
            return Err(e);
        }
        let ring = (prev_tick % 2) as usize;
        let half = self.d_pipe_out.as_ref().expect("enable_batch").len() / 2;
        let res = {
            let p = self.pipe.as_ref().expect("pipe active");
            let ev = p.ev[ring].as_ref().expect("in-flight tick event");
            exec.to_host_u32_after(
                ev,
                self.d_pipe_out.as_ref().expect("enable_batch"),
                ring * half,
                r,
            )
        };
        match res {
            Ok(ids) => Ok(ids),
            Err(e) => {
                self.pipe_abort();
                Err(e)
            }
        }
    }

    pub(crate) fn decode_pipe_drain_impl(&mut self) -> Result<Vec<u32>, GpuError> {
        let exec = self.exec.clone();
        let st = self
            .pipe
            .take()
            .ok_or_else(|| GpuError::Driver("decode_pipe_drain without begin".into()))?;
        let ring = (st.tick % 2) as usize;
        let half = self.d_pipe_out.as_ref().expect("enable_batch").len() / 2;
        let ev = st.ev[ring].as_ref().expect("in-flight tick event");
        let res = exec.to_host_u32_after(
            ev,
            self.d_pipe_out.as_ref().expect("enable_batch"),
            ring * half,
            st.r,
        );
        // pipe ticks never recorded the MTP h map; positions moved, so any
        // stale entries would fail the warm check anyway - clear them.
        for e in self.spec_rows.iter_mut() {
            *e = None;
        }
        match res {
            Ok(ids) => Ok(ids),
            Err(e) => {
                let _ = exec.stream.synchronize();
                Err(e)
            }
        }
    }

    fn pipe_abort(&mut self) {
        if self.pipe.take().is_some() {
            let _ = self.exec.stream.synchronize();
        }
    }

    /// The graph-capturable step body: every input is device-resident
    /// (d_tokens, pf_pos, sampler params) - replays only re-execute kernels.
    fn batch_step_body(&mut self, r: usize) -> Result<(), GpuError> {
        let hp = &self.hp;
        let d_slots = self.d_slots.as_ref().expect("enable_batch allocates");
        let sc = &mut self.scratch;
        let exec = &self.exec;

        // token rows straight from the embedding plane, the arch's
        // embedding scale folded into the gather
        exec.embed_gather_plane(
            &self.token_embd,
            self.d_tokens.as_ref().expect("enable_batch"),
            &mut sc.pf_x,
            hp.n_embd,
            r,
            hp.embd_scale(),
        )?;
        super::GpuGemma4::embd_preamble(exec, hp, self.embd_ones.as_ref(), &mut sc.pf_x, r)?;

        // fused decode walk: ~17 kernels/layer (from ~25) - the c1 gap is
        // GPU-side execution of many tiny kernels, so count is the currency.
        // GEMV outputs land CONCATENATED ([q|k|v], [gate|up]) via output
        // offsets - same single weight copies, fused epilogues.
        // Cross-layer band fusion: a layer's FFN residual-add
        // + post-norm defers into the next layer's fused attn pre-norm kernel
        // (same addnorm_e4m3_row, prew = next attn_norm). (prev layer idx,
        // out_scale) - materialized plain if the next layer can't fuse.
        let mut pending_post: Option<(usize, f32, u32)> = None;
        // P49: the deferred pf_proj holds bf16 (b16-D down GEMM) - the
        // absorber must pick the p16 twin. Set/cleared with pending_post.
        let mut post_b16 = false;
        // DFlash feature taps: while a drafter is armed, the
        // residual ENTERING each of `target_layers` is folded into the
        // drafter's fusion accumulator right here, mid-walk. Doing it at the
        // tap rather than staging five bands for later is what keeps the
        // cost one [cap, n_embd] buffer instead of five.
        //
        // Reading pf_x at the top of a layer is only correct because muse
        // layers never defer their post-norm: `fused_norm_ok()` is false on
        // every gated layer, so `pending_post` is structurally None here. The
        // assert below is the guard for the day someone gives this family a
        // gated fused arm - a stale residual would fuse silently.
        let fuse_now = self.dflash_fuse_wanted && !super::dflash::fuse_off();
        let mut dtap = self
            .dflash
            .as_mut()
            .filter(|d| d.state.is_some() && fuse_now);
        for (li, lw) in self.layers.iter().enumerate() {
            if let Some(df) = dtap.as_mut()
                && let Some(band) = df.target_layers.iter().position(|&t| t == li)
            {
                debug_assert!(
                    pending_post.is_none(),
                    "dflash tap at layer {li} with a deferred post - pf_x is not this \
                     layer's input"
                );
                super::dflash::tap_band(exec, df, &sc.pf_x, band, hp.n_embd, r)?;
            }
            let kvl = &mut self.kv[li];
            // (re)register this layer's dim-major twin before any
            // append enqueue - the append kernels capture the registered
            // base at launch time (writer-fused double-store, no ABI arg).
            // Layers without a twin clear it so their appends skip the store.
            self.exec.vdim_set(kvl.vdim.as_ref())?;
            let hd = lw.head_dim;
            let n_kv = lw.n_kv_heads;
            let kv_dim = kvl.kv_dim;
            let q_dim = hp.n_head * hd;
            let rope = if lw.is_swa {
                hp.rope_swa
            } else {
                hp.rope_global
            };
            let factors = (!lw.is_swa).then_some(&self.rope_factors);
            let window = if lw.is_swa { hp.swa_window } else { 0 };
            // QK score scale. gemma4 folds its query scale into the q-norm
            // weights and scores UNSCALED (f_attention_scale = 1.0);
            // muse-glimmer passes kq_scale = 1/sqrt(head_dim) on top of its
            // own q-norm weights. Hparams::attn_scale carries the difference.
            let ascale = hp.attn_scale(hd);
            // per-row stride of the concatenated qkv rowlet - the fused
            // epilogue kernel indexes rows by (q_dim + 2*kv_dim)
            // per-layer block table: SWA layers ride the WindowRing, global
            // layers the budget pool (None = dense planes)
            let layer_bt: Option<(&CudaSlice<u32>, usize)> = if lw.is_swa {
                self.paging.as_ref().map(|pg| (&pg.bt, pg.bps))
            } else {
                self.gpool.as_ref().map(|gp| (&gp.d_bt, gp.bps))
            };

            // Two fused rmsnorm->e4m3 paths, mutually exclusive by prefill arm:
            // origin's f8a batch kernel (r>1) and the on-box's f8w per-32 kernel
            // (r>=65). Both consume only the quantized form (pf_e4q/pf_e4s), so
            // the f32 norm never lands; every other lane falls back to rmsnorm_batch.
            // (fused_norm_ok: gated layers need the f32 pf_normed all three of
            // these arms skip writing - see LayerWeights::fused_norm_ok)
            let nqf_attn = r > 1
                && (lw.f8a_wqkv.is_some() || lw.f8a_wq.is_some())
                && lw.fused_norm_ok()
                && exec.has_rmsnorm_e4m3_batch()
                && nqfuse_on();
            // Row-scale twin for the f8t decode band: the fused
            // norm emits (e4m3, f32 row scale) - exactly what the f8t arms
            // consume - so their separate quantize_e4m3_row is skipped.
            // Mirrors the arm SELECTION below (f8a wins over f8t).
            // r==1 joins the band (qwen35's "batched band covers b==1"
            // lesson): the single-row graph otherwise pays rmsnorm_batch +
            // per-site quantize - ~4 launches x 60 layers - for identical
            // bytes.
            // The widening requires exactly the conditions under which the
            // r==1 qkv chain takes the fused f8t GEMM (reclaimed stubs, no
            // f8w plane, fused qkv plane), so the f32 norm is never read.
            let r1_f8t_qkv = lw.wq.data.len() == 48 && lw.f8w_wq.is_none() && lw.f8t_qkv.is_some();
            let nqf_row_attn = (r > 1 || r1_f8t_qkv)
                && r <= nqf_wide_cap()
                && lw.f8a_wqkv.is_none()
                && lw.f8a_wq.is_none()
                && (lw.f8t_qkv.is_some() || lw.f8t_wq.is_some())
                && lw.fused_norm_ok()
                && exec.has_rmsnorm_e4m3_row()
                && nqfuse_on();
            let fuse_attn_norm = r >= 65
                && lw.f8a_wqkv.is_none()
                && lw.f8a_wq.is_none()
                && lw.f8w_wq.is_some()
                && lw.fused_norm_ok()
                && exec.has_rmsnorm_e4m3()
                && paddock_models::dev_var_os!("PADDOCK_G4_NO_NFUSE").is_none();
            if let Some((pi, ps, dnz)) = pending_post.take() {
                let p16 = std::mem::take(&mut post_b16);
                if nqf_row_attn && dnz > 1 {
                    // b16 down never K-splits (cutlass returns nz=1)
                    debug_assert!(!p16, "b16 proj cannot carry K-split planes");
                    // deferred FFN post consumes the down GEMM's
                    // K-split planes directly (ascending-z, gate-identical)
                    exec.addnorm_e4m3_row_nz(
                        &mut sc.pf_x,
                        &sc.pf_skfix,
                        &self.layers[pi].ffn_post_norm,
                        &lw.attn_norm,
                        &mut sc.pf_e4q,
                        &mut sc.pf_e4rs,
                        hp.n_embd,
                        hp.eps,
                        ps,
                        r,
                        dnz,
                    )?;
                } else if nqf_row_attn && p16 {
                    // deferred FFN post + this attn pre-norm - pf_proj holds
                    // bf16 from the P49 b16-D down GEMM
                    exec.addnorm_e4m3_row_p16(
                        &mut sc.pf_x,
                        &sc.pf_proj,
                        &self.layers[pi].ffn_post_norm,
                        &lw.attn_norm,
                        &mut sc.pf_e4q,
                        &mut sc.pf_e4rs,
                        hp.n_embd,
                        hp.eps,
                        ps,
                        r,
                    )?;
                } else if nqf_row_attn {
                    // deferred FFN post + this attn pre-norm, one kernel
                    exec.addnorm_e4m3_row(
                        &mut sc.pf_x,
                        &sc.pf_proj,
                        &self.layers[pi].ffn_post_norm,
                        &lw.attn_norm,
                        &mut sc.pf_e4q,
                        &mut sc.pf_e4rs,
                        hp.n_embd,
                        hp.eps,
                        ps,
                        r,
                    )?;
                } else if nqf_attn && exec.has_addnorm_e4m3_b32() {
                    // b32 twin (set by next_fuses_b32 - the f8a wide band):
                    // deferred FFN post + this attn pre-norm + per-32 quant
                    debug_assert!(dnz == 1, "b32 consumer takes combined proj only");
                    debug_assert!(!p16, "b32 consumer has no p16 twin (P49 gate excludes it)");
                    exec.addnorm_e4m3_b32(
                        &mut sc.pf_x,
                        &sc.pf_proj,
                        &self.layers[pi].ffn_post_norm,
                        &lw.attn_norm,
                        &mut sc.pf_e4q,
                        &mut sc.pf_e4s,
                        hp.n_embd,
                        hp.eps,
                        ps,
                        r,
                    )?;
                } else {
                    debug_assert!(dnz == 1, "absorbed down planes need the fused consumer");
                    // next layer can't fuse after all: materialize plain,
                    // then fall through the normal chain
                    if p16 {
                        exec.rmsnorm_add_scale_p16(
                            &mut sc.pf_x,
                            &sc.pf_proj,
                            &self.layers[pi].ffn_post_norm,
                            hp.n_embd,
                            hp.post_norm_eps,
                            ps,
                            r,
                        )?;
                    } else {
                        exec.rmsnorm_add_scale(
                            &mut sc.pf_x,
                            &sc.pf_proj,
                            &self.layers[pi].ffn_post_norm,
                            hp.n_embd,
                            hp.post_norm_eps,
                            ps,
                            r,
                        )?;
                    }
                    exec.rmsnorm_batch(
                        &sc.pf_x,
                        &lw.attn_norm,
                        &mut sc.pf_normed,
                        hp.n_embd,
                        hp.eps,
                        r,
                    )?;
                }
            } else if nqf_row_attn {
                exec.rmsnorm_e4m3_row(
                    &sc.pf_x,
                    &lw.attn_norm,
                    &mut sc.pf_e4q,
                    &mut sc.pf_e4rs,
                    hp.n_embd,
                    hp.eps,
                    r,
                )?;
            } else if nqf_attn {
                exec.rmsnorm_e4m3_batch(
                    &sc.pf_x,
                    &lw.attn_norm,
                    &mut sc.pf_e4q,
                    &mut sc.pf_e4s,
                    hp.n_embd,
                    r,
                    hp.eps,
                )?;
            } else if fuse_attn_norm {
                exec.rmsnorm_e4m3(
                    &sc.pf_x,
                    &lw.attn_norm,
                    &mut sc.pf_e4q,
                    &mut sc.pf_e4s,
                    hp.n_embd,
                    hp.eps,
                    r,
                )?;
            } else {
                exec.rmsnorm_batch(
                    &sc.pf_x,
                    &lw.attn_norm,
                    &mut sc.pf_normed,
                    hp.n_embd,
                    hp.eps,
                    r,
                )?;
            }

            // r==1 (the c1 serving hot path): q/k/v land CONCATENATED in one
            // buffer via output offsets, then one fused epilogue kernel does
            // all norms + rope + K/V appends. r>1 keeps the plane walk (its
            // fused epilogue is the next rung).
            // fused epilogue for all row counts: gemvs at r==1 (concatenated
            // via offsets into pf_q), GEMM planes at r>1 - the 3-pointer nra
            // kernel takes either layout. V:=K on the V-less global layers
            // copies the RAW k plane first.
            // set by the fused r>1 arm below: (row stride, koff, voff) for
            // the strided nra2s epilogue over the concatenated GEMM output
            let mut qkv_concat: Option<(usize, usize, usize)> = None;
            // P50: true when the f8t verify arm wrote PACKED bf16
            // q/k/v planes - the nra dispatch below must take the b16 twin
            let mut qkv_b16 = false;
            // lin fused planes have no f32-activation gemv - r==1 rides the
            // quantized fused-GEMM arm (marker dispatch lands pd_f8_gemm_lin,
            // qwen's exact r==1 rung) and the strided nra2s epilogue
            let qkv_lin = lw.f8a_wqkv.as_ref().is_some_and(|w| w.is_lin());
            if r == 1 && !qkv_lin {
                if let Some(qkv8) = &lw.f8a_wqkv {
                    // One fused gemv lands q|k(|v) concatenated in pf_q -
                    // exactly the layout the r==1 epilogue already expects;
                    // V-less layers keep the classic k->v copy dance
                    let qkv_out = q_dim + kv_dim * if lw.wv.is_some() { 2 } else { 1 };
                    if fp4_on() {
                        exec.fp4_gemv_at_off(
                            qkv8,
                            0,
                            &sc.pf_normed,
                            &mut sc.pf_q,
                            0,
                            hp.n_embd,
                            qkv_out,
                        )?;
                    } else {
                        exec.f8_gemv_at(qkv8, &sc.pf_normed, &mut sc.pf_q, 0, hp.n_embd, qkv_out)?;
                    }
                    if lw.wv.is_none() {
                        exec.copy_region(&sc.pf_q, q_dim, &mut sc.pf_tmp, 0, kv_dim)
                            .and_then(|_| {
                                exec.copy_region(
                                    &sc.pf_tmp,
                                    0,
                                    &mut sc.pf_q,
                                    q_dim + kv_dim,
                                    kv_dim,
                                )
                            })?;
                    }
                } else if let Some(q8w) = &lw.f8a_wq {
                    // F8A: same concatenated layout via the e4m3 gemv. Q is
                    // matched per SEGMENT, not all-or-nothing: a layer whose
                    // k/v ship bf16 has no f8a twin for them (LayerWeights::
                    // kv_q8), and requiring all three here would have dropped
                    // the whole layer to the Q8 else-arm - which reads wq's
                    // 48-byte f8a STUB. Silently wrong, not a crash.
                    exec.f8_gemv_at(q8w, &sc.pf_normed, &mut sc.pf_q, 0, hp.n_embd, q_dim)?;
                    match &lw.f8a_wk {
                        Some(k8w) => exec.f8_gemv_at(
                            k8w,
                            &sc.pf_normed,
                            &mut sc.pf_q,
                            q_dim,
                            hp.n_embd,
                            kv_dim,
                        )?,
                        None => lw.wk.gemv_at(exec, &sc.pf_normed, &mut sc.pf_q, q_dim)?,
                    }
                    match (&lw.f8a_wv, &lw.wv) {
                        (Some(v8w), _) => exec.f8_gemv_at(
                            v8w,
                            &sc.pf_normed,
                            &mut sc.pf_q,
                            q_dim + kv_dim,
                            hp.n_embd,
                            kv_dim,
                        )?,
                        // bf16 v: its own plane, same output offset
                        (None, Some(wv)) => {
                            wv.gemv_at(exec, &sc.pf_normed, &mut sc.pf_q, q_dim + kv_dim)?
                        }
                        // V-less global layer: v = copy of k
                        (None, None) => exec
                            .copy_region(&sc.pf_q, q_dim, &mut sc.pf_tmp, 0, kv_dim)
                            .and_then(|_| {
                                exec.copy_region(
                                    &sc.pf_tmp,
                                    0,
                                    &mut sc.pf_q,
                                    q_dim + kv_dim,
                                    kv_dim,
                                )
                            })?,
                    }
                } else if lw.wq.data.len() == 48
                    && let Some(q8) = &lw.f8w_wq
                {
                    // Q8-reclaim + KEEP_F8W: the originals are 48-byte stubs -
                    // the batched r==1 tick rides the f8w gemv exactly like
                    // the serial lane's reclaim arms (same planes, same class)
                    exec.f8_gemv_at(q8, &sc.pf_normed, &mut sc.pf_q, 0, hp.n_embd, q_dim)?;
                    match &lw.f8w_wk {
                        Some(k8) => exec.f8_gemv_at(
                            k8,
                            &sc.pf_normed,
                            &mut sc.pf_q,
                            q_dim,
                            hp.n_embd,
                            kv_dim,
                        )?,
                        None => lw.wk.gemv_at(exec, &sc.pf_normed, &mut sc.pf_q, q_dim)?,
                    }
                    match (&lw.f8w_wv, &lw.wv) {
                        (Some(v8), _) => exec.f8_gemv_at(
                            v8,
                            &sc.pf_normed,
                            &mut sc.pf_q,
                            q_dim + kv_dim,
                            hp.n_embd,
                            kv_dim,
                        )?,
                        (None, Some(wv)) => {
                            wv.gemv_at(exec, &sc.pf_normed, &mut sc.pf_q, q_dim + kv_dim)?
                        }
                        (None, None) => exec
                            .copy_region(&sc.pf_q, q_dim, &mut sc.pf_tmp, 0, kv_dim)
                            .and_then(|_| {
                                exec.copy_region(
                                    &sc.pf_tmp,
                                    0,
                                    &mut sc.pf_q,
                                    q_dim + kv_dim,
                                    kv_dim,
                                )
                            })?,
                    }
                } else if lw.wq.data.len() == 48
                    && let Some(qkv) = &lw.f8t_qkv
                {
                    // Q8-reclaim, unified planes: the fused f8t qkv
                    // GEMM at r=1 lands the same contiguous [q|k(|v)] row the
                    // r==1 epilogue expects; V-less layers copy the K region
                    // like every other arm
                    let qkv_out = q_dim + kv_dim * if lw.wv.is_some() { 2 } else { 1 };
                    if !nqf_row_attn {
                        exec.quantize_e4m3_row(
                            &sc.pf_normed,
                            &mut sc.pf_e4q,
                            &mut sc.pf_e4rs,
                            hp.n_embd,
                            1,
                        )?;
                    }
                    exec.f8t_gemm(
                        qkv,
                        &sc.pf_e4q,
                        &sc.pf_e4rs,
                        &mut sc.pf_skfix,
                        &mut sc.pf_q,
                        hp.n_embd,
                        qkv_out,
                        1,
                    )?;
                    if lw.wv.is_none() {
                        exec.copy_region(&sc.pf_q, q_dim, &mut sc.pf_tmp, 0, kv_dim)
                            .and_then(|_| {
                                exec.copy_region(
                                    &sc.pf_tmp,
                                    0,
                                    &mut sc.pf_q,
                                    q_dim + kv_dim,
                                    kv_dim,
                                )
                            })?;
                    }
                } else {
                    exec.q8_0_gemv_repacked_at(&lw.wq, &sc.pf_normed, &mut sc.pf_q, 0)?;
                    lw.wk.gemv_at(exec, &sc.pf_normed, &mut sc.pf_q, q_dim)?;
                    match &lw.wv {
                        Some(wv) => {
                            wv.gemv_at(exec, &sc.pf_normed, &mut sc.pf_q, q_dim + kv_dim)?
                        }
                        None => exec
                            .copy_region(&sc.pf_q, q_dim, &mut sc.pf_tmp, 0, kv_dim)
                            .and_then(|_| {
                                exec.copy_region(
                                    &sc.pf_tmp,
                                    0,
                                    &mut sc.pf_q,
                                    q_dim + kv_dim,
                                    kv_dim,
                                )
                            })?,
                    }
                }
            } else {
                // r>1 rides the int8-quantized GEMM classes (the f32
                // gemm_repacked kernel re-reads all r activation rows from L2
                // per OUTPUT row, so its cost scaled linearly with batch -
                // 60% of the r=8 tick): mma_ks (BN16/32 serving tiles +
                // K-split, gpt-oss's 9..=64 dense rung) up to its 64-row cap,
                // the mmq tile ladder above it.
                if let Some(qkv8) = &lw.f8a_wqkv {
                    // qkv-concat: one GEMM over the fused plane (out 16384
                    // SWA / 18432 global) - was 3 twin calls; the epilogue
                    // reads the concat rows via nra2s. V-less layers ALIAS
                    // voff at the k region (raw K is exactly what V wants;
                    // the kernel only reads src) - no copy at all.
                    if !nqf_attn {
                        exec.quantize_e4m3(
                            &sc.pf_normed,
                            &mut sc.pf_e4q,
                            &mut sc.pf_e4s,
                            r * hp.n_embd,
                        )?;
                    }
                    let has_v = lw.wv.is_some();
                    let qkv_out = q_dim + kv_dim * if has_v { 2 } else { 1 };
                    f8a_mm!(exec, sc, qkv8, &mut sc.pf_q, hp.n_embd, qkv_out, r);
                    qkv_concat = Some((qkv_out, q_dim, if has_v { q_dim + kv_dim } else { q_dim }));
                } else if let Some(q8w) = &lw.f8a_wq {
                    // F8A: one e4m3 quantize feeds q/k/v (twin/TMA by shape)
                    if !nqf_attn {
                        exec.quantize_e4m3(
                            &sc.pf_normed,
                            &mut sc.pf_e4q,
                            &mut sc.pf_e4s,
                            r * hp.n_embd,
                        )?;
                    }
                    // per SEGMENT: a bf16 k/v has no f8a twin (kv_q8) while q
                    // still does, and `f8a_wv == None` no longer means
                    // "V-less layer" on its own - ask lw.wv for that
                    f8a_mm!(exec, sc, q8w, &mut sc.pf_q, hp.n_embd, q_dim, r);
                    match &lw.f8a_wk {
                        Some(k8w) => f8a_mm!(exec, sc, k8w, &mut sc.pf_k, hp.n_embd, kv_dim, r),
                        None => lw.wk.gemm(exec, &sc.pf_normed, &mut sc.pf_k, r)?,
                    }
                    match (&lw.f8a_wv, &lw.wv) {
                        (Some(v8w), _) => {
                            f8a_mm!(exec, sc, v8w, &mut sc.pf_v, hp.n_embd, kv_dim, r)
                        }
                        (None, Some(wv)) => wv.gemm(exec, &sc.pf_normed, &mut sc.pf_v, r)?,
                        (None, None) => exec.copy_slice(&sc.pf_k, 0, r * kv_dim, &mut sc.pf_v)?,
                    }
                } else if r >= 65
                    && let Some(wq8) = &lw.f8w_wq
                {
                    // per-32 f8w planes on the tcgen05 block-scale route;
                    // pf_e4q/pf_e4s already hold the fused norm+quant output
                    if !fuse_attn_norm {
                        exec.quantize_e4m3(
                            &sc.pf_normed,
                            &mut sc.pf_e4q,
                            &mut sc.pf_e4s,
                            r * hp.n_embd,
                        )?;
                    }
                    exec.f8_gemm_w8(
                        wq8,
                        0,
                        &sc.pf_e4q,
                        &sc.pf_e4s,
                        &mut sc.pf_q,
                        hp.n_embd,
                        q_dim,
                        r,
                    )?;
                    match &lw.f8w_wk {
                        Some(k8) => exec.f8_gemm_w8(
                            k8,
                            0,
                            &sc.pf_e4q,
                            &sc.pf_e4s,
                            &mut sc.pf_k,
                            hp.n_embd,
                            kv_dim,
                            r,
                        )?,
                        None => lw.wk.gemm(exec, &sc.pf_normed, &mut sc.pf_k, r)?,
                    }
                    match (&lw.f8w_wv, &lw.wv) {
                        (Some(v8), _) => exec.f8_gemm_w8(
                            v8,
                            0,
                            &sc.pf_e4q,
                            &sc.pf_e4s,
                            &mut sc.pf_v,
                            hp.n_embd,
                            kv_dim,
                            r,
                        )?,
                        (None, Some(wv)) => wv.gemm(exec, &sc.pf_normed, &mut sc.pf_v, r)?,
                        (None, None) => exec.copy_slice(&sc.pf_k, 0, r * kv_dim, &mut sc.pf_v)?,
                    }
                } else if r >= 65
                    && let Some(q8) = &lw.f8_wq
                {
                    // prefill-shaped rows ride the fold-free rowwise-e4m3
                    // planes (on sm_100 the int8 mma pipe is the compute wall
                    // - see the f8_wq field note); the 2..64 decode/verify
                    // band below keeps q8
                    exec.quantize_e4m3_row(
                        &sc.pf_normed,
                        &mut sc.pf_e4q,
                        &mut sc.pf_e4rs,
                        hp.n_embd,
                        r,
                    )?;
                    exec.f8row_gemm(
                        q8,
                        &sc.pf_e4q,
                        &sc.pf_e4rs,
                        &mut sc.pf_q,
                        hp.n_embd,
                        q_dim,
                        r,
                    )?;
                    match &lw.f8_wk {
                        Some(k8) => exec.f8row_gemm(
                            k8,
                            &sc.pf_e4q,
                            &sc.pf_e4rs,
                            &mut sc.pf_k,
                            hp.n_embd,
                            kv_dim,
                            r,
                        )?,
                        None => lw.wk.gemm(exec, &sc.pf_normed, &mut sc.pf_k, r)?,
                    }
                    match (&lw.f8_wv, &lw.wv) {
                        (Some(v8), _) => exec.f8row_gemm(
                            v8,
                            &sc.pf_e4q,
                            &sc.pf_e4rs,
                            &mut sc.pf_v,
                            hp.n_embd,
                            kv_dim,
                            r,
                        )?,
                        (None, Some(wv)) => wv.gemm(exec, &sc.pf_normed, &mut sc.pf_v, r)?,
                        (None, None) => exec.copy_slice(&sc.pf_k, 0, r * kv_dim, &mut sc.pf_v)?,
                    }
                } else if let Some(qkv) = &lw.f8t_qkv {
                    // qkv-concat on the tile route, all bands: tc5p <=64,
                    // tc5r 65+ inside the launcher. One fused GEMM (128 tiles
                    // SWA / 144 global) instead of three underfilled launches;
                    // the strided nra2s epilogue reads the concat rows. V-less
                    // layers alias voff at the k region - no copy at all.
                    if !nqf_row_attn {
                        exec.quantize_e4m3_row(
                            &sc.pf_normed,
                            &mut sc.pf_e4q,
                            &mut sc.pf_e4rs,
                            hp.n_embd,
                            r,
                        )?;
                    }
                    let has_v = lw.wv.is_some();
                    let qkv_out = q_dim + kv_dim * if has_v { 2 } else { 1 };
                    exec.f8t_gemm(
                        qkv,
                        &sc.pf_e4q,
                        &sc.pf_e4rs,
                        &mut sc.pf_skfix,
                        &mut sc.pf_q,
                        hp.n_embd,
                        qkv_out,
                        r,
                    )?;
                    //  dump-diff: with the env set and a flat twin
                    // present, re-run the classic route into pf_gate scratch
                    // and compare on the host. Deliberately slow; debug only.
                    //  race probe: a bare stream sync after the qkv
                    // cutlass launch. Coherent-with-sync proves the race
                    // window opens immediately post-qkv (the faster kernel
                    // exposes a consumer that tc5's slowness masked).
                    // Mode 1: main-stream sync only. Mode 2: context-wide
                    // drain - if 2 restores coherence where 1 worsened it
                    // (observed 0/32 vs 31/32), the racing writer is on
                    // another stream in the context (overlap lane / copy).
                    match paddock_models::dev_var!("PADDOCK_F8CUT_QKV_SYNC")
                        .ok()
                        .as_deref()
                    {
                        Some("2") if qkv.flat.is_some() => {
                            exec.device_sync()?;
                        }
                        Some(_) if qkv.flat.is_some() => {
                            exec.stream
                                .synchronize()
                                .map_err(|e| crate::gpu::GpuError::Driver(e.to_string()))?;
                        }
                        _ => {}
                    }
                    let qdiff_mode: u32 = paddock_models::dev_var!("PADDOCK_F8CUT_QKV_DIFF")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0);
                    if qdiff_mode > 0 && qkv.flat.is_some() {
                        // mode 1: cutlass(main) vs classic; mode 2: cutlass vs
                        // cutlass (self); mode 3: classic vs classic (self).
                        // Self-compares isolate which arm mutates across ticks.
                        let plane = qkv;
                        match qdiff_mode {
                            2 => exec.f8t_gemm(
                                plane,
                                &sc.pf_e4q,
                                &sc.pf_e4rs,
                                &mut sc.pf_skfix,
                                &mut sc.pf_gate,
                                hp.n_embd,
                                qkv_out,
                                r,
                            )?,
                            3 => {
                                exec.f8t_gemm_no_flat(
                                    plane,
                                    &sc.pf_e4q,
                                    &sc.pf_e4rs,
                                    &mut sc.pf_skfix,
                                    &mut sc.pf_up,
                                    hp.n_embd,
                                    qkv_out,
                                    r,
                                )?;
                                exec.f8t_gemm_no_flat(
                                    plane,
                                    &sc.pf_e4q,
                                    &sc.pf_e4rs,
                                    &mut sc.pf_skfix,
                                    &mut sc.pf_gate,
                                    hp.n_embd,
                                    qkv_out,
                                    r,
                                )?;
                            }
                            _ => exec.f8t_gemm_no_flat(
                                plane,
                                &sc.pf_e4q,
                                &sc.pf_e4rs,
                                &mut sc.pf_skfix,
                                &mut sc.pf_gate,
                                hp.n_embd,
                                qkv_out,
                                r,
                            )?,
                        }
                        // the copies race in-flight kernels without this:
                        // exec.stream is non-blocking, so a legacy-stream
                        // D2H does not serialize behind the enqueued GEMMs
                        exec.stream
                            .synchronize()
                            .map_err(|e| crate::gpu::GpuError::Driver(e.to_string()))?;
                        let n = r * qkv_out;
                        let asrc = if qdiff_mode == 3 { &sc.pf_up } else { &sc.pf_q };
                        let a = exec
                            .stream
                            .clone_dtoh(&asrc.try_slice(0..n).expect("scratch sized for r"))
                            .map_err(|e| crate::gpu::GpuError::Driver(e.to_string()))?;
                        let bb = exec
                            .stream
                            .clone_dtoh(&sc.pf_gate.try_slice(0..n).expect("scratch sized for r"))
                            .map_err(|e| crate::gpu::GpuError::Driver(e.to_string()))?;
                        let mut maxrel = 0f64;
                        let mut bad = 0usize;
                        let mut first = usize::MAX;
                        // bad-set structure: 128-col bands within the row
                        let mut hist = [0u32; 128];
                        let mut rowset = 0u64;
                        for i in 0..n {
                            let d = ((a[i] - bb[i]).abs()) as f64;
                            let rel = d / (bb[i].abs() as f64 + 1e-3);
                            if rel > maxrel {
                                maxrel = rel;
                            }
                            if rel > 0.05 {
                                if bad == 0 {
                                    first = i;
                                }
                                bad += 1;
                                hist[(i % qkv_out) >> 7 & 127] += 1;
                                rowset |= 1u64 << ((i / qkv_out) & 63);
                            }
                        }
                        if bad > 0 {
                            let mut top = (0usize, 0u32);
                            for (bi, &c) in hist.iter().enumerate() {
                                if c > top.1 {
                                    top = (bi, c);
                                }
                            }
                            tracing::warn!(
                                "[qkv-diff m{qdiff_mode}] li={li} r={r} out={qkv_out} maxrel={maxrel:.4} bad={bad}/{n} first={} a={} b={} topband={}({}) rows={rowset:b}",
                                first,
                                a[first],
                                bb[first],
                                top.0,
                                top.1
                            );
                        } else {
                            tracing::info!(
                                "[qkv-diff m{qdiff_mode}] li={li} r={r} CLEAN max={maxrel:.3e}"
                            );
                        }
                    }
                    qkv_concat = Some((qkv_out, q_dim, if has_v { q_dim + kv_dim } else { q_dim }));
                } else if let Some(q8) = &lw.f8t_wq {
                    // v4 decode band, attn twin (follow-up): one row
                    // quant feeds q/k/v through the tile-image class
                    if !nqf_row_attn {
                        exec.quantize_e4m3_row(
                            &sc.pf_normed,
                            &mut sc.pf_e4q,
                            &mut sc.pf_e4rs,
                            hp.n_embd,
                            r,
                        )?;
                    }
                    // P50: verify widths ride the b16-D arm on all
                    // three planes; the nra epilogue below reads them packed.
                    // V-less layers keep f32 (the K->V copy is element-typed).
                    qkv_b16 = r >= 65
                        && vb16q_on()
                        && exec.has_gemma_qkv_nra3_b16()
                        && q8.flat.is_some()
                        && lw.f8t_wk.as_ref().expect("f8t_wk pairs wq").flat.is_some()
                        && lw.f8t_wv.as_ref().is_some_and(|v| v.flat.is_some());
                    {
                        exec.f8t_gemm(
                            q8,
                            &sc.pf_e4q,
                            &sc.pf_e4rs,
                            &mut sc.pf_skfix,
                            &mut sc.pf_q,
                            hp.n_embd,
                            q_dim,
                            r,
                        )?;
                        exec.f8t_gemm(
                            lw.f8t_wk.as_ref().expect("f8t_wk pairs wq"),
                            &sc.pf_e4q,
                            &sc.pf_e4rs,
                            &mut sc.pf_skfix,
                            &mut sc.pf_k,
                            hp.n_embd,
                            kv_dim,
                            r,
                        )?;
                        match &lw.f8t_wv {
                            Some(v8) => exec.f8t_gemm(
                                v8,
                                &sc.pf_e4q,
                                &sc.pf_e4rs,
                                &mut sc.pf_skfix,
                                &mut sc.pf_v,
                                hp.n_embd,
                                kv_dim,
                                r,
                            )?,
                            None => exec.copy_slice(&sc.pf_k, 0, r * kv_dim, &mut sc.pf_v)?,
                        }
                    }
                } else if r <= 192 {
                    exec.quantize_q8(&sc.pf_normed, &mut sc.pf_xq, &mut sc.pf_xs, r * hp.n_embd)?;
                    exec.q8_0_gemm_mma_ks(
                        &lw.wq,
                        &sc.pf_xq,
                        &sc.pf_xs,
                        &mut sc.pf_skfix,
                        &mut sc.pf_q,
                        r,
                    )?;
                    match lw.wk.q8() {
                        Some(wk) => exec.q8_0_gemm_mma_ks(
                            wk,
                            &sc.pf_xq,
                            &sc.pf_xs,
                            &mut sc.pf_skfix,
                            &mut sc.pf_k,
                            r,
                        )?,
                        // bf16 k: no int8 rung, its own dispatch serves
                        None => lw.wk.gemm(exec, &sc.pf_normed, &mut sc.pf_k, r)?,
                    }
                    match &lw.wv {
                        Some(wv) => match wv.q8() {
                            Some(v8) => exec.q8_0_gemm_mma_ks(
                                v8,
                                &sc.pf_xq,
                                &sc.pf_xs,
                                &mut sc.pf_skfix,
                                &mut sc.pf_v,
                                r,
                            )?,
                            None => wv.gemm(exec, &sc.pf_normed, &mut sc.pf_v, r)?,
                        },
                        None => exec.copy_slice(&sc.pf_k, 0, r * kv_dim, &mut sc.pf_v)?,
                    }
                } else {
                    exec.quantize_q8_mmq(&sc.pf_normed, &mut sc.pf_yq, hp.n_embd, r)?;
                    pf_mmq(exec, &lw.wq, &sc.pf_yq, &mut sc.pf_skfix, &mut sc.pf_q, r)?;
                    match lw.wk.q8() {
                        Some(wk) => pf_mmq(exec, wk, &sc.pf_yq, &mut sc.pf_skfix, &mut sc.pf_k, r)?,
                        None => lw.wk.gemm(exec, &sc.pf_normed, &mut sc.pf_k, r)?,
                    }
                    match &lw.wv {
                        Some(wv) => match wv.q8() {
                            Some(v8) => {
                                pf_mmq(exec, v8, &sc.pf_yq, &mut sc.pf_skfix, &mut sc.pf_v, r)?
                            }
                            None => wv.gemm(exec, &sc.pf_normed, &mut sc.pf_v, r)?,
                        },
                        None => exec.copy_slice(&sc.pf_k, 0, r * kv_dim, &mut sc.pf_v)?,
                    }
                }
            }
            {
                let (bt, bps) = match layer_bt {
                    Some((bt, bps)) => (Some(bt), bps),
                    None => (None, 0),
                };
                let (koff, voff) = (q_dim, q_dim + kv_dim);
                // qkv_concat wins even at r==1 (the lin arm sets it): the
                // strided epilogue handles the V-less voff alias the fixed
                // nra2 offsets can't
                if r == 1 && qkv_concat.is_none() {
                    // SAFETY of the aliasing: q at [0,qdim), k at [qdim,..),
                    // v at [qdim+kv,..) are DISJOINT ranges of pf_q; the
                    // kernel writes q_out separately and reads each range
                    // only for its own head class. Rust-side we pass the
                    // same slice thrice via raw offsets inside the wrapper.
                    let pf_q: *mut cudarc::driver::CudaSlice<f32> = &mut sc.pf_q;
                    unsafe {
                        exec.gemma_qkv_nra(
                            (&mut *pf_q, 0),
                            (&mut *pf_q, koff),
                            (&mut *pf_q, voff),
                            &lw.q_norm,
                            &lw.k_norm,
                            &mut sc.pf_qn,
                            &mut kvl.k,
                            &mut kvl.v,
                            &sc.pf_pos,
                            Some(d_slots),
                            factors,
                            bt,
                            bps,
                            hp.n_head,
                            n_kv,
                            hd,
                            self.max_ctx,
                            r,
                            hp.eps,
                            rope,
                            kvl.dtype,
                            0,
                            hp.rope_neox(),
                            hp.v_norm(),
                        )?;
                    }
                } else if let Some((stride, ko, vo)) = qkv_concat {
                    // fused-GEMM epilogue: same kernel math, shared row
                    // stride over the concatenated output. Same disjoint-range
                    // aliasing as the r==1 arm above - q at [0,..), k at
                    // [ko,..), v at [vo,..) of one plane (V-less layers alias
                    // vo == ko deliberately, which is why this arm exists).
                    debug_assert!(!qkv_b16, "b16 qkv planes never ride the concat arm");
                    let pf_q: *mut cudarc::driver::CudaSlice<f32> = &mut sc.pf_q;
                    unsafe {
                        exec.gemma_qkv_nra(
                            (&mut *pf_q, 0),
                            (&mut *pf_q, ko),
                            (&mut *pf_q, vo),
                            &lw.q_norm,
                            &lw.k_norm,
                            &mut sc.pf_qn,
                            &mut kvl.k,
                            &mut kvl.v,
                            &sc.pf_pos,
                            Some(d_slots),
                            factors,
                            bt,
                            bps,
                            hp.n_head,
                            n_kv,
                            hd,
                            self.max_ctx,
                            r,
                            hp.eps,
                            rope,
                            kvl.dtype,
                            stride,
                            hp.rope_neox(),
                            hp.v_norm(),
                        )?;
                    }
                } else if qkv_b16 {
                    let _ = (koff, voff);
                    // P50: planes hold packed bf16 from the b16-D arm - the
                    // slot-420 twin reads them; epilogue outputs unchanged
                    let pf_k: *mut cudarc::driver::CudaSlice<f32> = &mut sc.pf_k;
                    let pf_v: *mut cudarc::driver::CudaSlice<f32> = &mut sc.pf_v;
                    unsafe {
                        exec.gemma_qkv_nra_b16(
                            (&mut sc.pf_q, 0),
                            (&mut *pf_k, 0),
                            (&mut *pf_v, 0),
                            &lw.q_norm,
                            &lw.k_norm,
                            &mut sc.pf_qn,
                            &mut kvl.k,
                            &mut kvl.v,
                            &sc.pf_pos,
                            Some(d_slots),
                            factors,
                            bt,
                            bps,
                            hp.n_head,
                            n_kv,
                            hd,
                            self.max_ctx,
                            r,
                            hp.eps,
                            rope,
                            kvl.dtype,
                            0,
                            hp.rope_neox(),
                            hp.v_norm(),
                        )?;
                    }
                } else {
                    let _ = (koff, voff);
                    // disjoint scratch fields - plain reborrows are fine
                    let pf_k: *mut cudarc::driver::CudaSlice<f32> = &mut sc.pf_k;
                    let pf_v: *mut cudarc::driver::CudaSlice<f32> = &mut sc.pf_v;
                    unsafe {
                        exec.gemma_qkv_nra(
                            (&mut sc.pf_q, 0),
                            (&mut *pf_k, 0),
                            (&mut *pf_v, 0),
                            &lw.q_norm,
                            &lw.k_norm,
                            &mut sc.pf_qn,
                            &mut kvl.k,
                            &mut kvl.v,
                            &sc.pf_pos,
                            Some(d_slots),
                            factors,
                            bt,
                            bps,
                            hp.n_head,
                            n_kv,
                            hd,
                            self.max_ctx,
                            r,
                            hp.eps,
                            rope,
                            kvl.dtype,
                            0,
                            hp.rope_neox(),
                            hp.v_norm(),
                        )?;
                    }
                }
            }

            // P53: true when the fin-e4 arm already wrote pf_e4q/pf_e4rs for
            // this layer's wo input - the standalone row quantize is skipped.
            let mut attn_e4 = false;
            // P54: true when the fin-e4s arm wrote pf_e4q at STATIC scale
            // 1.0 - quantize skipped AND the wo GEMM must take the ones
            // xrs (pf_fae4rs), not the stale pf_e4rs.
            let mut attn_e4s = false;
            let splits = attn_splits(hp.n_head, r, exec.sm_count(), self.n_slots);
            // dense arms use the kv-grid-aware count (the GQA kernels' real
            // grid); the spec arm keeps the proven heads*batch heuristic
            let dsplits = attn_splits_kv(n_kv, r, exec.sm_count(), self.n_slots, self.attn_pos_max);
            // WIDE-SPEC verify arm: rows are padded slot-major k1-chunks, so
            // one KV walk serves each chunk (per-row masks inside) - the
            // per-row walk makes a 160-row verify tick attention-bound
            // (~22% of GPU time, avg 960µs, more than the whole GEMM walk).
            // WIDTH-GATED: sharing the walk only pays when the KV working set
            // spills L2 and the walk is DRAM-bound.
            // Below ~16 slots both classes are L2-resident on the 128MB GB202
            // (SWA: slots×4MB window; global: slots×~10MB full-ctx at 1k) and
            // the fused walk just serializes issue-bound work, and an
            // always-on share costs narrow batches badly.
            let spec_arm = self
                .spec_k1
                .filter(|&k1| k1 > 1 && r.is_multiple_of(k1) && exec.has_attn_spec_batch_paged())
                // the gate itself (FA-f8 floor, context-volume calculus)
                // lives in spec_width_ok, because the mixed tick's front
                // verify rows share it (verify-fold rung A)
                .filter(|&k1| spec_width_ok(r / k1, window, n_kv, hd, kvl.dtype));
            // (a fused combine+row-quant for the wo input was tried
            // and REVERTED - its one-CTA-per-row grid (64 CTAs) serializes
            // what attn_combine_batch spreads over n_heads x batch = 2048
            // CTAs, and loses at width. The kernel stays in the pack as the
            // documented negative; a head-parallel variant would need a
            // cross-CTA row max, i.e. the two-kernel chain again.)
            if let (Some(k1), Some((bt, bps))) = (spec_arm, layer_bt) {
                let (ao, aml) = self
                    .attn_scratch
                    .as_mut()
                    .expect("enable_batch allocates attention scratch");
                // At >=16 live chunks the FA grid
                // (n_kv x chunks) fills the die at one split - take the FIN
                // route (in-kernel finalize, bit-identical to the -inf-sink
                // combine) and skip the combine launch + partial round trip.
                // Ok(false) = geometry can't engage (sm_120 hd512 global
                // layers) - that layer keeps the partial+combine chain. The
                // few-chunk band loses on fin, and stays on splits by the
                // floor.
                let wide_band = r / k1 >= 16
                    && self.spec_long
                    && spec_fin_on()
                    && exec.has_attn_spec_batch_fin();
                // sm_120 wide-band split election (spec_fa_par.cu): on GB202
                // the FIN grid leaves throughput on the table even with the
                // pack's SB/occ-2 shape - the KV walk is still latency-bound
                // per CTA and grid.z buys walk overlap at no extra bytes.
                // Measured at the 128-row serve point, combine included:
                // SWA 306.6 fin -> 252.8 us sp2; global (128 CTAs, 0.68
                // waves) 488 fin -> 415 us sp4. B200 keeps FIN, its own
                // measured winner - 227KB smem gives DB 2 CTAs/SM there.
                // Clamped to the FlashDecoding scratch (max_batch*16 rows).
                // PADDOCK_G4_NO_SPEC_SP restores FIN for A/B.
                let sp_cap = (self.n_slots * MAX_ATTN_SPLITS) / r.max(1);
                let wide_sp = if wide_band && spec_sp_on() && exec.compute_capability().0 >= 12 {
                    (if window > 0 { 2 } else { 4 }).min(sp_cap)
                } else if wide_band
                    && window == 0
                    && glb_sp4_on()
                    && exec.compute_capability().0 >= 12
                {
                    // GLB-only split election (see glb_sp4_on):
                    // sp4+combine beats the fin GLB grid and the margin
                    // grows with ctx depth; SWA stays on the fin route below
                    4usize.min(sp_cap)
                } else {
                    1
                };
                // slot 423: fin-e4 - the finalized rows
                // land directly as e4m3 + row scales in pf_e4q/pf_e4rs
                // (bit-identical to fin + quantize_e4m3_row), killing the
                // standalone wo-in row-quant launch. n_kv==1 is the
                // whole-row-CTA condition and it never holds on gemma-4-31b
                // (SWA is 32 heads / 16 KV = group 2; a row spans 16 CTAs)
                // - so it is FALSIFIED for this model by geometry, and an
                // A/B here is an A/A. The arm is kept for a future n_kv==1
                // spec model but is UNVALIDATED on real traffic; validate
                // numerics before trusting it. The surviving elimination
                // form for gemma is the b16-x wo GEMM (mixed-input cutlass).
                if wide_band
                    && wide_sp <= 1
                    && n_kv == 1
                    && fae4_on()
                    && lw.f8a_wo.is_none()
                    && lw.f8w_wo.is_none()
                    && lw.f8t_wo.is_some()
                    && exec.has_attn_spec_batch_fin_e4()
                {
                    attn_e4 = exec.attn_spec_batch_fin_e4(
                        &sc.pf_qn,
                        &kvl.k,
                        &kvl.v,
                        &mut sc.pf_e4q,
                        &mut sc.pf_e4rs,
                        &sc.pf_pos,
                        Some(d_slots),
                        bt,
                        bps,
                        hp.n_head,
                        n_kv,
                        hd,
                        kv_dim,
                        window,
                        r,
                        k1,
                        ascale,
                        kvl.dtype,
                    )?;
                }
                // slot 425: fin-e4s - static-scale e4m3 store straight into
                // pf_e4q; the wo GEMM takes the ones xrs (pf_fae4rs).
                // FALSIFIED twice over on a wide spec verify: (1) it damages
                // ACCEPTANCE, because the per-row scale the standalone
                // quantize computes is empirically load-bearing - rows are
                // range-heterogeneous and no single global scale covers them;
                // (2) even a leg that got lucky on acceptance gained zero
                // throughput, because the 60 quantize nodes it deletes were
                // PDL joint-wait, not wall time. Stays env-dead as the
                // documented negative. (The mixed-input cutlass route this
                // replaced died even earlier, at the probe:
                // bench/womix_probe.cu, ~5x off the weight floor.) Gates
                // mirror the wo-site
                // TOP-LEVEL branch only (f8a/f8w none + f8t some -> the f8t
                // arm always consumes pf_e4q + xrs) - the store is a
                // per-element convert, legal at any GQA geometry.
                if !attn_e4
                    && wide_band
                    && wide_sp <= 1
                    && fae4s_on()
                    && lw.f8a_wo.is_none()
                    && lw.f8w_wo.is_none()
                    && lw.f8t_wo.is_some()
                    && exec.has_attn_spec_batch_fin_e4s()
                {
                    attn_e4s = exec.attn_spec_batch_fin_e4s(
                        &sc.pf_qn,
                        &kvl.k,
                        &kvl.v,
                        &mut sc.pf_e4q,
                        aml,
                        &sc.pf_pos,
                        Some(d_slots),
                        bt,
                        bps,
                        hp.n_head,
                        n_kv,
                        hd,
                        kv_dim,
                        window,
                        r,
                        k1,
                        ascale,
                        kvl.dtype,
                    )?;
                    // engagement is otherwise invisible in the serve log,
                    // and the lesson from fin-e4 above is to prove the arm
                    // actually fired before reading an A/B as a verdict
                    static E4S_ONCE: std::sync::Once = std::sync::Once::new();
                    E4S_ONCE.call_once(|| {
                        eprintln!("[fae4s] first election: engaged={attn_e4s} r={r} k1={k1}");
                    });
                }
                let fin_done = attn_e4
                    || attn_e4s
                    || (wide_band
                        && wide_sp <= 1
                        && exec.attn_spec_batch_fin(
                            &sc.pf_qn,
                            &kvl.k,
                            &kvl.v,
                            &mut sc.pf_attn,
                            aml,
                            &sc.pf_pos,
                            Some(d_slots),
                            bt,
                            bps,
                            hp.n_head,
                            n_kv,
                            hd,
                            kv_dim,
                            window,
                            r,
                            k1,
                            ascale,
                            kvl.dtype,
                        )?);
                if !fin_done {
                    let splits = if wide_sp > 1 { wide_sp } else { splits };
                    // LCO (opt-in PADDOCK_SPEC_LCO): the krs arms
                    // merge their split partials in-kernel - Ok(true) means
                    // pf_attn already holds the combined rows and the combine
                    // launch is skipped. Ok(false) = geometry not covered
                    // (non-krs shapes) - the proven chain below runs.
                    let lco_done = splits > 1
                        && (spec_lco_on() || self.spec_shallow)
                        && exec.has_attn_spec_lco_paged()
                        && exec.attn_spec_lco_paged(
                            &sc.pf_qn,
                            &kvl.k,
                            &kvl.v,
                            ao,
                            aml,
                            &sc.neg_inf_sinks,
                            &mut sc.pf_attn,
                            self.lco_tickets
                                .as_mut()
                                .expect("alloc'd with attn_scratch"),
                            &sc.pf_pos,
                            Some(d_slots),
                            bt,
                            bps,
                            hp.n_head,
                            n_kv,
                            hd,
                            kv_dim,
                            window,
                            splits,
                            r,
                            k1,
                            ascale,
                            kvl.dtype,
                        )?;
                    if !lco_done {
                        // A/B history on B200: xy-grid-sized splits (floor 4,
                        // cap 16) + TILE 16/32 + a QF-parallel score phase each looked
                        // right on paper and the S U M regressed every config vs this
                        // baseline - co-residency beats per-CTA depth on this die.
                        // Rebuild the experiment one variable at a time before touching
                        // this again (spec_attn_splits stays available below).
                        exec.attn_spec_batch_paged(
                            &sc.pf_qn,
                            &kvl.k,
                            &kvl.v,
                            ao,
                            aml,
                            &sc.pf_pos,
                            Some(d_slots),
                            bt,
                            bps,
                            hp.n_head,
                            n_kv,
                            hd,
                            kv_dim,
                            window,
                            splits,
                            r,
                            k1,
                            ascale,
                            kvl.dtype,
                        )?;
                        exec.attn_combine_batch(
                            ao,
                            aml,
                            &sc.neg_inf_sinks,
                            &mut sc.pf_attn,
                            hp.n_head,
                            hd,
                            splits,
                            r,
                        )?;
                    }
                }
            } else if self.spec_k1.is_none()
                && let Some((bt, bps)) = layer_bt
                && attn_fmha16_arm(exec, hp.n_head, n_kv, hd, kvl.dtype)
            {
                // slot 458: Q16xKv128 tensor-core decode attention. FINAL
                // write with the sink folded in - no partials, no combine.
                // No row gate and no band gate (see attn_fmha16_arm).
                exec.attn_decode_fmha16(
                    &sc.pf_qn,
                    &kvl.k,
                    &kvl.v,
                    &sc.neg_inf_sinks,
                    &mut sc.pf_attn,
                    &sc.pf_pos,
                    Some(d_slots),
                    bt,
                    bps,
                    hp.n_head,
                    n_kv,
                    hd,
                    kv_dim,
                    window,
                    r,
                    ascale,
                    kvl.dtype,
                )?;
            } else if r >= 16
                && self.spec_k1.is_none()
                && let Some((bt, bps)) = layer_bt
                && attn_fused16_arm(exec, hp.n_head, n_kv, hd, kvl.dtype, r, self.attn_pos_max)
            {
                // fused single-pass GQA-16 (muse geometry) - one
                // launch, final write with the sink folded, no combine. The
                // pos hint is the kv_split_band CEILING so the captured
                // graph stays valid across the whole band.
                exec.attn_decode_fused_gqa16(
                    &sc.pf_qn,
                    &kvl.k,
                    &kvl.v,
                    &sc.neg_inf_sinks,
                    &mut sc.pf_attn,
                    &sc.pf_pos,
                    Some(d_slots),
                    bt,
                    bps,
                    hp.n_head,
                    n_kv,
                    hd,
                    kv_dim,
                    window,
                    r,
                    kv_split_band(self.attn_pos_max) * 128,
                    ascale,
                    kvl.dtype,
                )?;
            } else if r >= 16
                && self.spec_k1.is_none()
                && let Some((bt, bps)) = layer_bt
                && attn_tc5_on()
                && exec.has_attn_decode_tc5_paged()
                && {
                    // tcgen05 decode attention - FINAL rows
                    // straight into pf_attn (no partials/combine, like fin1).
                    // This is the pure-decode c32 site (the prefill_layers
                    // election only sees mixed ticks, where a16 blocks it).
                    // The pack entry re-gates shape/arch: Ok(false) falls
                    // through to fin1/splits unchanged.
                    // P60: GLB layers (window 0, hd512/G8) ride the same
                    // export with a banded effective window >= pos_max -
                    // kv_split_band is in the decode-graph key (gkey above)
                    // so a band change re-captures, never replays a stale
                    // bound. Gates: r >= 24 (96+ cells; the few-cell deep-
                    // walk corner measured LOSING to v9q's split spread)
                    // and band <= 6 (probe-measured win region, -26% at the
                    // r24/ew768 worst corner; deeper walks keep v9q z=2).
                    // +16: at pmax an exact multiple of 128 (band*128 ==
                    // pmax), a bare band*128 window makes the kernel's SWA
                    // walk drop key 0 (glo = pos+1-window = 1). One extra
                    // KV block keeps ew > pmax at every band value while
                    // staying a pure function of the graph-keyed band.
                    let tc5_win = if window > 0 {
                        window
                    } else if r >= 24 && attn_tc5_glb_on() {
                        let band = kv_split_band(self.attn_pos_max);
                        if band > 0 && band <= 6 {
                            band * 128 + 16
                        } else {
                            0
                        }
                    } else {
                        0
                    };
                    tc5_win > 0 && {
                        exec.attn_decode_tc5_paged(
                            &sc.pf_qn,
                            &kvl.k,
                            &kvl.v,
                            &sc.neg_inf_sinks,
                            &mut sc.pf_attn,
                            &sc.pf_pos,
                            Some(d_slots),
                            bt,
                            bps,
                            hp.n_head,
                            n_kv,
                            hd,
                            kv_dim,
                            tc5_win,
                            r,
                            ascale,
                            kvl.dtype,
                        )?
                    }
                }
            {
                // tc5 wrote the combined batch-major rows directly
            } else if r >= 16
                && self.spec_k1.is_none()
                && let Some((bt, bps)) = layer_bt
                && fin1_ok(exec, hd, hp.n_head, n_kv, kvl.dtype)
            {
                // (spec_k1 gate: VERIFY ticks pad r rows over few slots -
                // their windows are L2-resident and the walk latency-bound,
                // where splits still pay: dc4 633.9 vs 670.3 with fin1
                // ungated. Pure wide decode is bandwidth-bound and wins.)
                //  FIN: at batch >= 16 the v8/v8ks tile walks fill the
                // die without K-splits (bench: v8 87.8us at splits=1 vs 92.6
                // at 2) and finalize IN-KERNEL at n_splits==1 (o/l with -inf
                // sinks == the combine math, bit-identical) - the separate
                // combine pass (27us x 60 layers/tick = 1.63 ms, ~11% of a
                // wide tick) disappears. pf_attn receives
                // the combined batch-major layout directly; aml is dead
                // scratch the kernel skips at one split.
                let (_, aml) = self
                    .attn_scratch
                    .as_mut()
                    .expect("enable_batch allocates attention scratch");
                exec.attn_partial_batch_paged(
                    &sc.pf_qn,
                    &kvl.k,
                    &kvl.v,
                    &mut sc.pf_attn,
                    aml,
                    &sc.pf_pos,
                    Some(d_slots),
                    bt,
                    bps,
                    hp.n_head,
                    n_kv,
                    hd,
                    kv_dim,
                    window,
                    1,
                    r,
                    ascale,
                    kvl.dtype,
                )?;
            } else if dsplits > 1 {
                let (ao, aml) = self
                    .attn_scratch
                    .as_mut()
                    .expect("enable_batch allocates attention scratch");
                match layer_bt {
                    Some((bt, bps)) => exec.attn_partial_batch_paged(
                        &sc.pf_qn,
                        &kvl.k,
                        &kvl.v,
                        ao,
                        aml,
                        &sc.pf_pos,
                        Some(d_slots),
                        bt,
                        bps,
                        hp.n_head,
                        n_kv,
                        hd,
                        kv_dim,
                        window,
                        dsplits,
                        r,
                        ascale,
                        kvl.dtype,
                    )?,
                    _ => exec.attn_partial_batch(
                        &sc.pf_qn,
                        &kvl.k,
                        &kvl.v,
                        ao,
                        aml,
                        &sc.pf_pos,
                        Some(d_slots),
                        hp.n_head,
                        n_kv,
                        hd,
                        self.max_ctx,
                        kv_dim,
                        window,
                        dsplits,
                        r,
                        ascale,
                        kvl.dtype,
                    )?,
                }
                exec.attn_combine_batch(
                    ao,
                    aml,
                    &sc.neg_inf_sinks,
                    &mut sc.pf_attn,
                    hp.n_head,
                    hd,
                    dsplits,
                    r,
                )?;
            } else {
                match layer_bt {
                    Some((bt, bps)) => exec.attn_decode_batch_paged(
                        &sc.pf_qn,
                        &kvl.k,
                        &kvl.v,
                        &sc.neg_inf_sinks,
                        &mut sc.pf_attn,
                        &sc.pf_pos,
                        Some(d_slots),
                        bt,
                        bps,
                        hp.n_head,
                        n_kv,
                        hd,
                        kv_dim,
                        window,
                        r,
                        ascale,
                        kvl.dtype,
                    )?,
                    _ => exec.attn_decode_batch(
                        &sc.pf_qn,
                        &kvl.k,
                        &kvl.v,
                        &sc.neg_inf_sinks,
                        &mut sc.pf_attn,
                        &sc.pf_pos,
                        Some(d_slots),
                        hp.n_head,
                        n_kv,
                        hd,
                        self.max_ctx,
                        kv_dim,
                        window,
                        r,
                        ascale,
                        kvl.dtype,
                    )?,
                }
            }

            let n_ff = lw.ffn_gate.dims[1];
            // F8R: the whole FFN rides e4m3 - gemv at r==1 (f32 x, no
            // activation quant), mma_ks twin 2..=31, TMA GEMM from 32 (the
            // q8 trio is stubs in this mode). e4m3 class, quality-gated.
            let f8r = (lw.f8_gate.is_some() || lw.f8_gu.is_some())
                && matches!(lw.ffn_gate.dims.len(), 2)
                && lw.ffn_gate.data.len() <= 32;
            // FFN arm selection (plane unification): one decision the
            // gate/up chain, the norm-fusion flags, and the down-skip guard
            // all share - the old scattered r-band guards drifted. f8w arms
            // exist only under PADDOCK_G4_KEEP_F8W (planes not built
            // otherwise); the f8t arm covers every band through the launcher
            // (tc5p <=64 / tc5r 65+), including r==1 once the q8 originals
            // are reclaim-stubbed.
            let ffn_f8w_r1 = !f8r && r == 1 && lw.ffn_gate.data.len() <= 48 && lw.f8_gate.is_some();
            let ffn_f8w_pf = !f8r && r >= 65 && lw.f8w_wq.is_some() && lw.f8_gate.is_some();
            let ffn_f8row_pf = !f8r && !ffn_f8w_pf && r >= 65 && lw.f8r_gate.is_some();
            let ffn_f8t = !f8r
                && !ffn_f8w_r1
                && !ffn_f8w_pf
                && !ffn_f8row_pf
                && (r > 1 || lw.ffn_gate.data.len() <= 48)
                && (lw.f8t_gu.is_some() || lw.f8t_gate.is_some());
            // Two fused rmsnorm->e4m3 paths, mutually exclusive by FFN arm
            // (mirrors the attn band): the f8r batch kernel (r>1) and the
            // f8w per-32 kernel (r>=65, non-f8r). Both consume only the
            // quantized form; everything else falls back to rmsnorm_batch.
            let nqf_ffn = r > 1 && f8r && exec.has_rmsnorm_e4m3_batch() && nqfuse_on();
            // Band-boundary fusion: attn residual-add + post-norm +
            // FFN pre-norm + row quant in one kernel when the f8t ffn arm
            // will consume it. Mirrors the FFN arm selection (non-f8r, r<=64
            // f8t planes). Bit-identical to the 3-launch chain.
            // r==1 included: ffn_f8t itself admits r==1 only
            // under reclaimed stubs, where the gu/down consumers read
            // pf_e4q - every downstream quantize is already !nqf-guarded.
            // (fused_two_norm_ok: this arm and the two below fuse
            // attn_post_norm with ffn_norm through one epsilon parameter -
            // sound only when the arch's two epsilons agree. See
            // Hparams::fused_two_norm_ok.)
            let nqf_row_ffn =
                ffn_f8t && hp.fused_two_norm_ok() && exec.has_addnorm_e4m3_row() && nqfuse_on();
            // Per-32 band-boundary fusion (the f8a/f8r wide-decode band where
            // the row twin's f8t gate never holds): attn residual-add +
            // post-norm + FFN pre-norm + per-32 quant in one kernel. Subset
            // of nqf_ffn so the downstream !nqf_ffn quantize skips still hold.
            let nqf_b32_ffn = nqf_ffn && hp.fused_two_norm_ok() && exec.has_addnorm_e4m3_b32();
            // pc verify-tick engagement: the fused-gu arm below is
            // the only consumer, so the producer and route share one flag
            let pc_gu_b = lw.gu_ws.is_some()
                && r >= pc_floor()
                && lw.gu_il
                && !fp4_on()
                && gu_fuse_on()
                && hp.fused_two_norm_ok()
                && exec.has_f8_gemm_lin_gu_pc(hp.glu_act())
                && exec.has_addnorm_e4m3_row();
            let fuse_ffn_norm = ffn_f8w_pf
                && exec.has_rmsnorm_e4m3()
                && paddock_models::dev_var_os!("PADDOCK_G4_NO_NFUSE").is_none();
            // hoisted: the down site needs this before the FFN block
            // to decide absorption; identical predicate, position-invariant
            // Deferring a post-norm is what BUILDS a two-norm/one-eps kernel
            // (this ffn_post_norm + the next attn_norm), so the epsilon
            // agreement is a precondition for the deferral itself, not just
            // for the consumer arm.
            let next_fuses = li + 1 < self.layers.len()
                && r <= nqf_wide_cap()
                && hp.fused_two_norm_ok()
                && exec.has_addnorm_e4m3_row()
                && nqfuse_on()
                && {
                    let nl = &self.layers[li + 1];
                    // fused_norm_ok mirrors nqf_row_attn - deferring a post
                    // into a consumer that won't fire strands it (dnz>1 would
                    // strand the K-split planes outright)
                    nl.fused_norm_ok()
                        && nl.f8a_wqkv.is_none()
                        && nl.f8a_wq.is_none()
                        && (nl.f8t_qkv.is_some() || nl.f8t_wq.is_some())
                        // r==1: mirror the next layer's widened nqf_row_attn
                        // exactly, so the deferred post always finds its
                        // fused consumer
                        && (r > 1
                            || (nl.wq.data.len() == 48
                                && nl.f8w_wq.is_none()
                                && nl.f8t_qkv.is_some()))
                };
            // b32 twin of the deferral: the next layer's f8a arm will run
            // nqf_attn (per-32), so its fused consumer eats the pending FFN
            // post. Mirrors nqf_attn's predicate exactly so the consumer
            // branch fires iff this set it.
            let next_fuses_b32 = li + 1 < self.layers.len()
                && r > 1
                && hp.fused_two_norm_ok()  // same two-norm/one-eps precondition
                && exec.has_addnorm_e4m3_b32()
                && exec.has_rmsnorm_e4m3_batch()
                && nqfuse_on()
                && {
                    let nl = &self.layers[li + 1];
                    // mirrors nqf_attn, incl. its fused_norm_ok term
                    nl.fused_norm_ok() && (nl.f8a_wqkv.is_some() || nl.f8a_wq.is_some())
                };
            // sigmoid output gate (muse-glimmer), before every wo arm below -
            // several quantize pf_attn on the way in, so this cannot move
            // after them. pf_normed is f32-live here: the fused attn-norm arms
            // are all gated off on gated layers (fused_norm_ok).
            super::GpuGemma4::attn_gate_apply(
                exec,
                lw,
                &sc.pf_normed,
                &mut sc.pf_agate,
                &mut sc.pf_attn,
                &mut sc.pf_xq,
                &mut sc.pf_xs,
                &mut sc.pf_yq,
                &mut sc.pf_skfix,
                &mut sc.pf_e4q,
                &mut sc.pf_e4rs,
                hp.n_embd,
                hp.n_head * hd,
                r,
            )?;
            // >1 when the wo GEMM left K-split partials in pf_skfix
            // for the fused addnorm to absorb
            let mut wo_nz: u32 = 1;
            // f8t16: true iff the pack's tc5r O16 arm wrote bf16 pf_proj
            // (wo plane, chunk widths). Consumers below must match dtype.
            let mut wo_o16 = false;
            // >1 when the down GEMM leaves planes for the next
            // layer's fused addnorm (carried in pending_post)
            let mut down_nz: u32 = 1;
            // P49: pf_proj holds bf16 from the b16-D down GEMM - the post
            // consumer (immediate or deferred) must take the p16 twin
            // P49 b16-D verify election: floor 65 keeps the decode band
            // (r<=32, tc5-elected) and the narrow spec verifies out; the
            if let Some(o8w) = &lw.f8a_wo {
                // F8A out-projection (out = n_embd 5376 -> twin through 64);
                // lin planes take the quantized band at r==1 too (no f32 gemv)
                if r == 1 && !o8w.is_lin() {
                    if fp4_on() {
                        exec.fp4_gemv_at_off(
                            o8w,
                            0,
                            &sc.pf_attn,
                            &mut sc.pf_proj,
                            0,
                            hp.n_head * hd,
                            hp.n_embd,
                        )?;
                    } else {
                        exec.f8_gemv_at(
                            o8w,
                            &sc.pf_attn,
                            &mut sc.pf_proj,
                            0,
                            hp.n_head * hd,
                            hp.n_embd,
                        )?;
                    }
                } else {
                    exec.quantize_e4m3(
                        &sc.pf_attn,
                        &mut sc.pf_e4q,
                        &mut sc.pf_e4s,
                        r * hp.n_head * hd,
                    )?;
                    f8a_mm!(exec, sc, o8w, &mut sc.pf_proj, hp.n_head * hd, hp.n_embd, r);
                }
            } else if r == 1
                && lw.wo.data.len() == 48
                && let Some(wo8) = &lw.f8w_wo
            {
                // Q8-reclaim + KEEP_F8W: stubbed original -> f8w gemv
                exec.f8_gemv_at(
                    wo8,
                    &sc.pf_attn,
                    &mut sc.pf_proj,
                    0,
                    hp.n_head * hd,
                    hp.n_embd,
                )?;
            } else if r == 1 && lw.wo.data.len() > 48 {
                exec.q8_0_gemm_repacked(&lw.wo, None, &sc.pf_attn, &mut sc.pf_proj, r)?;
            } else if r >= 65
                && let Some(wo8) = &lw.f8w_wo
            {
                exec.quantize_e4m3(
                    &sc.pf_attn,
                    &mut sc.pf_e4q,
                    &mut sc.pf_e4s,
                    r * hp.n_head * hd,
                )?;
                exec.f8_gemm_w8(
                    wo8,
                    0,
                    &sc.pf_e4q,
                    &sc.pf_e4s,
                    &mut sc.pf_proj,
                    hp.n_head * hd,
                    hp.n_embd,
                    r,
                )?;
            } else if let Some(wot) = &lw.f8t_wo {
                // All remaining bands ride the tile plane: tc5p at
                // <=64, tc5r at 65+, r==1 included (stubbed originals)
                let wo_in = hp.n_head * hd;
                // only the consumer arms with p16 twins may see bf16, and
                // only the layer class whose wo in_dim the loader published
                wo_o16 =
                    f8t16_on() && r >= 129 && !nqf_b32_ffn && !pc_gu_b && wo_in == f8t16_wo_in();
                // P53: the fin-e4 arm already wrote these planes in-kernel.
                // P54: the fin-e4s arm wrote pf_e4q at static scale - skip
                // the quantize AND feed the GEMM the ones xrs (pf_e4rs is
                // stale from this tick's earlier row quantizes).
                if !attn_e4 && !attn_e4s {
                    exec.quantize_e4m3_row(&sc.pf_attn, &mut sc.pf_e4q, &mut sc.pf_e4rs, wo_in, r)?;
                }
                let wo_xrs = if attn_e4s { &sc.pf_fae4rs } else { &sc.pf_e4rs };
                if nqf_row_ffn && ksabs_on() && exec.has_f8t_gemm_nc() {
                    // the fused addnorm consumes the K-split partial
                    // planes directly - the combine launch and the combined
                    // buffer's write+read round trip both disappear
                    wo_nz = exec.f8t_gemm_nc(
                        wot,
                        &sc.pf_e4q,
                        wo_xrs,
                        &mut sc.pf_skfix,
                        &mut sc.pf_proj,
                        wo_in,
                        hp.n_embd,
                        r,
                    )?;
                } else {
                    exec.f8t_gemm(
                        wot,
                        &sc.pf_e4q,
                        wo_xrs,
                        &mut sc.pf_skfix,
                        &mut sc.pf_proj,
                        wo_in,
                        hp.n_embd,
                        r,
                    )?;
                }
            } else if r >= 65
                && let Some(wo8) = &lw.f8_wo
            {
                // same prefill/rowwise split as the qkv site above
                exec.quantize_e4m3_row(
                    &sc.pf_attn,
                    &mut sc.pf_e4q,
                    &mut sc.pf_e4rs,
                    hp.n_head * hd,
                    r,
                )?;
                exec.f8row_gemm(
                    wo8,
                    &sc.pf_e4q,
                    &sc.pf_e4rs,
                    &mut sc.pf_proj,
                    hp.n_head * hd,
                    hp.n_embd,
                    r,
                )?;
            } else if r <= 192 {
                exec.quantize_q8(
                    &sc.pf_attn,
                    &mut sc.pf_xq,
                    &mut sc.pf_xs,
                    r * hp.n_head * hd,
                )?;
                exec.q8_0_gemm_mma_ks(
                    &lw.wo,
                    &sc.pf_xq,
                    &sc.pf_xs,
                    &mut sc.pf_skfix,
                    &mut sc.pf_proj,
                    r,
                )?;
            } else {
                exec.quantize_q8_mmq(&sc.pf_attn, &mut sc.pf_yq, hp.n_head * hd, r)?;
                pf_mmq(
                    exec,
                    &lw.wo,
                    &sc.pf_yq,
                    &mut sc.pf_skfix,
                    &mut sc.pf_proj,
                    r,
                )?;
            }
            // x = x + rmsnorm(proj)·w  (attention half: scale 1) - fused
            // with the FFN pre-norm + row quant on the nqf_row_ffn arm
            if nqf_row_ffn {
                if wo_nz > 1 {
                    exec.addnorm_e4m3_row_nz(
                        &mut sc.pf_x,
                        &sc.pf_skfix,
                        &lw.attn_post_norm,
                        &lw.ffn_norm,
                        &mut sc.pf_e4q,
                        &mut sc.pf_e4rs,
                        hp.n_embd,
                        hp.eps,
                        1.0,
                        r,
                        wo_nz,
                    )?;
                } else if wo_o16 {
                    exec.addnorm_e4m3_row_p16(
                        &mut sc.pf_x,
                        &sc.pf_proj,
                        &lw.attn_post_norm,
                        &lw.ffn_norm,
                        &mut sc.pf_e4q,
                        &mut sc.pf_e4rs,
                        hp.n_embd,
                        hp.eps,
                        1.0,
                        r,
                    )?;
                } else {
                    exec.addnorm_e4m3_row(
                        &mut sc.pf_x,
                        &sc.pf_proj,
                        &lw.attn_post_norm,
                        &lw.ffn_norm,
                        &mut sc.pf_e4q,
                        &mut sc.pf_e4rs,
                        hp.n_embd,
                        hp.eps,
                        1.0,
                        r,
                    )?;
                }
            } else if pc_gu_b {
                // pc verify-tick producer: row scales emitted
                // directly, mirroring the forward chunk path
                exec.addnorm_e4m3_row(
                    &mut sc.pf_x,
                    &sc.pf_proj,
                    &lw.attn_post_norm,
                    &lw.ffn_norm,
                    &mut sc.pf_e4q,
                    &mut sc.pf_e4rs,
                    hp.n_embd,
                    hp.eps,
                    1.0,
                    r,
                )?;
            } else if nqf_b32_ffn {
                exec.addnorm_e4m3_b32(
                    &mut sc.pf_x,
                    &sc.pf_proj,
                    &lw.attn_post_norm,
                    &lw.ffn_norm,
                    &mut sc.pf_e4q,
                    &mut sc.pf_e4s,
                    hp.n_embd,
                    hp.eps,
                    1.0,
                    r,
                )?;
            } else {
                if wo_o16 {
                    exec.rmsnorm_add_scale_p16(
                        &mut sc.pf_x,
                        &sc.pf_proj,
                        &lw.attn_post_norm,
                        hp.n_embd,
                        hp.post_norm_eps,
                        1.0,
                        r,
                    )?;
                } else {
                    exec.rmsnorm_add_scale(
                        &mut sc.pf_x,
                        &sc.pf_proj,
                        &lw.attn_post_norm,
                        hp.n_embd,
                        hp.post_norm_eps,
                        1.0,
                        r,
                    )?;
                }
                if nqf_ffn {
                    exec.rmsnorm_e4m3_batch(
                        &sc.pf_x,
                        &lw.ffn_norm,
                        &mut sc.pf_e4q,
                        &mut sc.pf_e4s,
                        hp.n_embd,
                        r,
                        hp.eps,
                    )?;
                } else if fuse_ffn_norm {
                    exec.rmsnorm_e4m3(
                        &sc.pf_x,
                        &lw.ffn_norm,
                        &mut sc.pf_e4q,
                        &mut sc.pf_e4s,
                        hp.n_embd,
                        hp.eps,
                        r,
                    )?;
                } else {
                    exec.rmsnorm_batch(
                        &sc.pf_x,
                        &lw.ffn_norm,
                        &mut sc.pf_normed,
                        hp.n_embd,
                        hp.eps,
                        r,
                    )?;
                }
            }
            // A4B two-branch overlap (opt-in PADDOCK_MOE_FORK): the shared
            // dense-FFN block below and g4_moe_tail's routed branch both read
            // only post-attention x (disjoint scratch: pf_* vs moe_*), and
            // first meet at the two-branch tail. Fork the SHORT shared branch
            // onto the side stream; the routed branch keeps the priority
            // stream and its PDL adjacency. g4_moe_tail's side_join()
            // re-serializes before the tail consumes pf_proj. Scheduling
            // only - every kernel, input, and value is unchanged.
            let moe_fork = lw.moe.is_some() && g4_moe_fork();
            if moe_fork {
                exec.side_fork()?;
            }
            if f8r {
                let gu = lw.f8_gu.as_ref();
                let d8 = lw.f8_down.as_ref().expect("f8 FFN planes built as a set");
                // lin FFN planes (gu+down convert all-or-nothing at load):
                // r==1 joins the quantized mma_ks band, whose wrapper lands
                // pd_f8_gemm_lin - the f32 gemv arm can't read lin boxes
                if r == 1 && !d8.is_lin() {
                    if let Some(gu) = gu {
                        // fused: one gemv over 2*n_ff outs lands the same
                        // concatenated [gate|up] row geglu_pair expects
                        if fp4_on() {
                            exec.fp4_gemv_at_off(
                                gu,
                                0,
                                &sc.pf_normed,
                                &mut sc.pf_gate,
                                0,
                                hp.n_embd,
                                2 * n_ff,
                            )?;
                        } else {
                            exec.f8_gemv_at(
                                gu,
                                &sc.pf_normed,
                                &mut sc.pf_gate,
                                0,
                                hp.n_embd,
                                2 * n_ff,
                            )?;
                        }
                    } else {
                        exec.f8_gemv_at(
                            lw.f8_gate.as_ref().expect("f8_gate present without gu"),
                            &sc.pf_normed,
                            &mut sc.pf_gate,
                            0,
                            hp.n_embd,
                            n_ff,
                        )?;
                        exec.f8_gemv_at(
                            lw.f8_up.as_ref().expect("f8 FFN planes built as a set"),
                            &sc.pf_normed,
                            &mut sc.pf_gate,
                            n_ff,
                            hp.n_embd,
                            n_ff,
                        )?;
                    }
                    exec.glu_pair(&mut sc.pf_gate, n_ff, 1, hp.glu_act())?;
                    if fp4_on() {
                        exec.fp4_gemv_at_off(
                            d8,
                            0,
                            &sc.pf_gate,
                            &mut sc.pf_proj,
                            0,
                            n_ff,
                            hp.n_embd,
                        )?;
                    } else {
                        exec.f8_gemv_at(d8, &sc.pf_gate, &mut sc.pf_proj, 0, n_ff, hp.n_embd)?;
                    }
                } else if r <= 31 && exec.has_f8_gemm_mma_ks() {
                    // f8 mma_ks twin: the spec-verify band where the TMA
                    // GEMM's 128-col tile pays ~2x (harness: twin at q8-ks
                    // parity or better 2..31, 1.9-4.5x vs TMA; from 32 the
                    // TMA GEMM wins). Same e4m3 inputs as the TMA arm.
                    if !nqf_ffn {
                        exec.quantize_e4m3(
                            &sc.pf_normed,
                            &mut sc.pf_e4q,
                            &mut sc.pf_e4s,
                            r * hp.n_embd,
                        )?;
                    }
                    // verify-GEMM dedup: gate|up as one GEMM on the fused
                    // plane ([r][2*n_ff] out) - halves the launch/tail-wave
                    // overhead of the two 64/32-CTA calls. Values identical
                    // (same kernels, concatenated weights).
                    if let (Some(gu), true) = (gu, exec.has_quantize_e4m3_glu2(hp.glu_act())) {
                        if fp4_on() {
                            exec.fp4_gemm_mma_ks(
                                gu,
                                &sc.pf_e4q,
                                &sc.pf_e4s,
                                &mut sc.pf_skfix,
                                &mut sc.pf_gate,
                                hp.n_embd,
                                2 * n_ff,
                                r,
                            )?;
                        } else {
                            exec.f8_gemm_mma_ks(
                                gu,
                                &sc.pf_e4q,
                                &sc.pf_e4s,
                                &mut sc.pf_skfix,
                                &mut sc.pf_gate,
                                hp.n_embd,
                                2 * n_ff,
                                r,
                            )?;
                        }
                        // interleaved plane (gu_il): permuted GEMM rows,
                        // pair-addressed geglu - identical bytes
                        if lw.gu_il {
                            exec.quantize_e4m3_glu2i(
                                &sc.pf_gate,
                                &mut sc.pf_e4q,
                                &mut sc.pf_e4s,
                                n_ff,
                                r,
                                hp.glu_act(),
                            )?;
                        } else {
                            exec.quantize_e4m3_glu2(
                                &sc.pf_gate,
                                &mut sc.pf_e4q,
                                &mut sc.pf_e4s,
                                n_ff,
                                r,
                                hp.glu_act(),
                            )?;
                        }
                    } else {
                        exec.f8_gemm_mma_ks(
                            lw.f8_gate.as_ref().expect("f8_gate present without gu"),
                            &sc.pf_e4q,
                            &sc.pf_e4s,
                            &mut sc.pf_skfix,
                            &mut sc.pf_gate,
                            hp.n_embd,
                            n_ff,
                            r,
                        )?;
                        exec.f8_gemm_mma_ks(
                            lw.f8_up.as_ref().expect("f8 FFN planes built as a set"),
                            &sc.pf_e4q,
                            &sc.pf_e4s,
                            &mut sc.pf_skfix,
                            &mut sc.pf_up,
                            hp.n_embd,
                            n_ff,
                            r,
                        )?;
                        g4_e4m3_glu(exec, sc, r * n_ff, hp.glu_act())?;
                    }
                    if fp4_on() {
                        exec.fp4_gemm_mma_ks(
                            d8,
                            &sc.pf_e4q,
                            &sc.pf_e4s,
                            &mut sc.pf_skfix,
                            &mut sc.pf_proj,
                            n_ff,
                            hp.n_embd,
                            r,
                        )?;
                    } else {
                        exec.f8_gemm_mma_ks(
                            d8,
                            &sc.pf_e4q,
                            &sc.pf_e4s,
                            &mut sc.pf_skfix,
                            &mut sc.pf_proj,
                            n_ff,
                            hp.n_embd,
                            r,
                        )?;
                    }
                } else {
                    if !pc_gu_b && !nqf_ffn {
                        exec.quantize_e4m3(
                            &sc.pf_normed,
                            &mut sc.pf_e4q,
                            &mut sc.pf_e4s,
                            r * hp.n_embd,
                        )?;
                    }
                    let mut gu_fused = false;
                    if let (Some(gu), true) = (gu, exec.has_quantize_e4m3_glu2(hp.glu_act())) {
                        // fused-epilogue arm: geglu + per-32
                        // quant land in the GEMM on the interleaved plane -
                        // bit-identical bytes to the 2-launch chain
                        // (gu_fuse_bench), y/pf_gate never written. Output
                        // goes to pf_ffq/pf_ffs (the GEMM reads pf_e4q via
                        // TMA while storing). Ok(false) = route disengaged
                        // (kt3 off etc.) -> 2-launch chain below.
                        if pc_gu_b {
                            gu_fused = exec.f8_gemm_lin_gu_pc(
                                gu,
                                &sc.pf_e4q,
                                &sc.pf_e4rs,
                                lw.gu_ws.as_ref().expect("pc_gu_b"),
                                &mut sc.pf_ffq,
                                &mut sc.pf_ffs,
                                hp.n_embd,
                                2 * n_ff,
                                r,
                                hp.glu_act(),
                            )?;
                            if !gu_fused {
                                return Err(crate::gpu::GpuError::Driver(
                                    "pc verify gu route refused".into(),
                                ));
                            }
                        } else if lw.gu_il && !fp4_on() && gu_fuse_on() {
                            gu_fused = exec.f8_gemm_lin_gu(
                                gu,
                                &sc.pf_e4q,
                                &sc.pf_e4s,
                                &mut sc.pf_ffq,
                                &mut sc.pf_ffs,
                                hp.n_embd,
                                2 * n_ff,
                                r,
                                hp.glu_act(),
                            )?;
                        }
                        if !gu_fused {
                            // fused TMA arm (all r >= 32 incl. prefill ticks
                            // - pf_gate holds PF_ROWS x 2*n_ff): one launch
                            // instead of two
                            if fp4_on() {
                                exec.mxfp4_gemm_bs(
                                    gu,
                                    &sc.pf_e4q,
                                    &sc.pf_e4s,
                                    &mut sc.pf_gate,
                                    hp.n_embd,
                                    2 * n_ff,
                                    r,
                                )?;
                            } else {
                                exec.f8_gemm_w8(
                                    gu,
                                    0,
                                    &sc.pf_e4q,
                                    &sc.pf_e4s,
                                    &mut sc.pf_gate,
                                    hp.n_embd,
                                    2 * n_ff,
                                    r,
                                )?;
                            }
                            if lw.gu_il {
                                exec.quantize_e4m3_glu2i(
                                    &sc.pf_gate,
                                    &mut sc.pf_e4q,
                                    &mut sc.pf_e4s,
                                    n_ff,
                                    r,
                                    hp.glu_act(),
                                )?;
                            } else {
                                exec.quantize_e4m3_glu2(
                                    &sc.pf_gate,
                                    &mut sc.pf_e4q,
                                    &mut sc.pf_e4s,
                                    n_ff,
                                    r,
                                    hp.glu_act(),
                                )?;
                            }
                        }
                    } else {
                        exec.f8_gemm_w8(
                            lw.f8_gate.as_ref().expect("f8_gate present without gu"),
                            0,
                            &sc.pf_e4q,
                            &sc.pf_e4s,
                            &mut sc.pf_gate,
                            hp.n_embd,
                            n_ff,
                            r,
                        )?;
                        exec.f8_gemm_w8(
                            lw.f8_up.as_ref().expect("f8 FFN planes built as a set"),
                            0,
                            &sc.pf_e4q,
                            &sc.pf_e4s,
                            &mut sc.pf_up,
                            hp.n_embd,
                            n_ff,
                            r,
                        )?;
                        g4_e4m3_glu(exec, sc, r * n_ff, hp.glu_act())?;
                    }
                    // the fused arm landed the ff activations in pf_ffq/pf_ffs
                    let (dq, ds) = if gu_fused {
                        (&sc.pf_ffq, &sc.pf_ffs)
                    } else {
                        (&sc.pf_e4q, &sc.pf_e4s)
                    };
                    // down at 32..64 stays on the twin: out=5376 is 42 tiles
                    // on a 188-SM die - the TMA GEMM underfills and pays
                    // 2.0-2.6x (harness: 70.9 vs 186.9us at b32). Gate/up
                    // (168 tiles) correctly ride TMA from 32.
                    if r <= 64 && exec.has_f8_gemm_mma_ks() {
                        if fp4_on() {
                            exec.fp4_gemm_mma_ks(
                                d8,
                                dq,
                                ds,
                                &mut sc.pf_skfix,
                                &mut sc.pf_proj,
                                n_ff,
                                hp.n_embd,
                                r,
                            )?;
                        } else {
                            exec.f8_gemm_mma_ks(
                                d8,
                                dq,
                                ds,
                                &mut sc.pf_skfix,
                                &mut sc.pf_proj,
                                n_ff,
                                hp.n_embd,
                                r,
                            )?;
                        }
                    } else if fp4_on() {
                        exec.mxfp4_gemm_bs(d8, dq, ds, &mut sc.pf_proj, n_ff, hp.n_embd, r)?;
                    } else if let Some(dws) = &lw.down_ws
                        && r >= pc_floor()
                        && exec.has_f8_gemm_w8_pcd()
                    {
                        if !exec.f8_gemm_w8_pcd(
                            d8,
                            dq,
                            ds,
                            dws,
                            &mut sc.pf_proj,
                            n_ff,
                            hp.n_embd,
                            r,
                        )? {
                            return Err(crate::gpu::GpuError::Driver(
                                "pc verify down route refused".into(),
                            ));
                        }
                    } else {
                        exec.f8_gemm_w8(d8, 0, dq, ds, &mut sc.pf_proj, n_ff, hp.n_embd, r)?;
                    }
                }
            } else if ffn_f8w_r1 {
                // Q8-reclaim + KEEP_F8W: stubbed originals -> f8w gemvs into
                // the same [gate|up] concat layout + geglu_pair
                exec.f8_gemv_at(
                    lw.f8_gate.as_ref().expect("checked by ffn_f8w_r1"),
                    &sc.pf_normed,
                    &mut sc.pf_gate,
                    0,
                    hp.n_embd,
                    n_ff,
                )?;
                exec.f8_gemv_at(
                    lw.f8_up.as_ref().expect("f8 FFN planes built as a set"),
                    &sc.pf_normed,
                    &mut sc.pf_gate,
                    n_ff,
                    hp.n_embd,
                    n_ff,
                )?;
                exec.glu_pair(&mut sc.pf_gate, n_ff, 1, hp.glu_act())?;
            } else if ffn_f8w_pf {
                if !fuse_ffn_norm {
                    exec.quantize_e4m3(
                        &sc.pf_normed,
                        &mut sc.pf_e4q,
                        &mut sc.pf_e4s,
                        r * hp.n_embd,
                    )?;
                }
                exec.f8_gemm_w8(
                    lw.f8_gate.as_ref().expect("checked by ffn_f8w_pf"),
                    0,
                    &sc.pf_e4q,
                    &sc.pf_e4s,
                    &mut sc.pf_gate,
                    hp.n_embd,
                    n_ff,
                    r,
                )?;
                exec.f8_gemm_w8(
                    lw.f8_up.as_ref().expect("f8 FFN planes built as a set"),
                    0,
                    &sc.pf_e4q,
                    &sc.pf_e4s,
                    &mut sc.pf_up,
                    hp.n_embd,
                    n_ff,
                    r,
                )?;
                g4_e4m3_glu(exec, sc, r * n_ff, hp.glu_act())?;
                exec.f8_gemm_w8(
                    lw.f8_down.as_ref().expect("f8 FFN planes built as a set"),
                    0,
                    &sc.pf_e4q,
                    &sc.pf_e4s,
                    &mut sc.pf_proj,
                    n_ff,
                    hp.n_embd,
                    r,
                )?;
            } else if ffn_f8row_pf {
                // prefill FFN on the rowwise-e4m3 class (fold-free GEMM);
                // down is folded in here too, so the shared down-arm below
                // must skip this case
                exec.quantize_e4m3_row(
                    &sc.pf_normed,
                    &mut sc.pf_e4q,
                    &mut sc.pf_e4rs,
                    hp.n_embd,
                    r,
                )?;
                exec.f8row_gemm(
                    lw.f8r_gate.as_ref().expect("checked by ffn_f8row_pf"),
                    &sc.pf_e4q,
                    &sc.pf_e4rs,
                    &mut sc.pf_gate,
                    hp.n_embd,
                    n_ff,
                    r,
                )?;
                exec.f8row_gemm(
                    lw.f8r_up.as_ref().expect("f8row FFN planes built as a set"),
                    &sc.pf_e4q,
                    &sc.pf_e4rs,
                    &mut sc.pf_up,
                    hp.n_embd,
                    n_ff,
                    r,
                )?;
                exec.glu(&mut sc.pf_gate, &sc.pf_up, r * n_ff, hp.glu_act())?;
                exec.quantize_e4m3_row(&sc.pf_gate, &mut sc.pf_e4q, &mut sc.pf_e4rs, n_ff, r)?;
                exec.f8row_gemm(
                    lw.f8r_down
                        .as_ref()
                        .expect("f8row FFN planes built as a set"),
                    &sc.pf_e4q,
                    &sc.pf_e4rs,
                    &mut sc.pf_proj,
                    n_ff,
                    hp.n_embd,
                    r,
                )?;
            } else if ffn_f8t && let Some(gu8) = &lw.f8t_gu {
                // fused gate|up tile plane: one 336-tile GEMM (persistent
                // tc5q route inside the launcher) writes [token][2*n_ff] -
                // exactly the geglu2_row epilogue's input; compact e4m3 rows
                // feed the down GEMM directly.
                if !nqf_row_ffn {
                    exec.quantize_e4m3_row(
                        &sc.pf_normed,
                        &mut sc.pf_e4q,
                        &mut sc.pf_e4rs,
                        hp.n_embd,
                        r,
                    )?;
                }
                {
                    let gu_nz = if ksabs_on() && exec.has_f8t_gemm_nc() {
                        exec.f8t_gemm_nc(
                            gu8,
                            &sc.pf_e4q,
                            &sc.pf_e4rs,
                            &mut sc.pf_skfix,
                            &mut sc.pf_gate,
                            hp.n_embd,
                            2 * n_ff,
                            r,
                        )?
                    } else {
                        exec.f8t_gemm(
                            gu8,
                            &sc.pf_e4q,
                            &sc.pf_e4rs,
                            &mut sc.pf_skfix,
                            &mut sc.pf_gate,
                            hp.n_embd,
                            2 * n_ff,
                            r,
                        )?;
                        1
                    };
                    if gu_nz > 1 {
                        exec.quantize_e4m3_glu2_row_nz(
                            &sc.pf_skfix,
                            &mut sc.pf_e4q,
                            &mut sc.pf_e4rs,
                            n_ff,
                            r,
                            gu_nz,
                            hp.glu_act(),
                        )?;
                    } else {
                        exec.quantize_e4m3_glu2_row(
                            &sc.pf_gate,
                            &mut sc.pf_e4q,
                            &mut sc.pf_e4rs,
                            n_ff,
                            r,
                            hp.glu_act(),
                        )?;
                    }
                } // !gluq_done
                if next_fuses
                    && lw.moe.is_none()
                    && exec.has_rmsnorm_e4m3_row()
                    && ksabs_on()
                    && ksabs_down_on()
                    && exec.has_f8t_gemm_nc()
                {
                    // leave the down planes for the next layer's
                    // fused addnorm (predicate exactly implies its nz path)
                    down_nz = exec.f8t_gemm_nc(
                        lw.f8t_down.as_ref().expect("f8t FFN planes built as a set"),
                        &sc.pf_e4q,
                        &sc.pf_e4rs,
                        &mut sc.pf_skfix,
                        &mut sc.pf_proj,
                        n_ff,
                        hp.n_embd,
                        r,
                    )?;
                } else {
                    exec.f8t_gemm(
                        lw.f8t_down.as_ref().expect("f8t FFN planes built as a set"),
                        &sc.pf_e4q,
                        &sc.pf_e4rs,
                        &mut sc.pf_skfix,
                        &mut sc.pf_proj,
                        n_ff,
                        hp.n_embd,
                        r,
                    )?;
                }
            } else if ffn_f8t {
                // v4 decode band: rowwise e4m3 over the tile-image
                // planes - the q8 ring was L2-hit-capped at ~3.2 TB/s and
                // these stream half the bytes. pf_e4q's tail rows past r are
                // stale (64-row TMA boxes read them) but only feed D columns
                // the epilogue never stores.
                if !nqf_row_ffn {
                    exec.quantize_e4m3_row(
                        &sc.pf_normed,
                        &mut sc.pf_e4q,
                        &mut sc.pf_e4rs,
                        hp.n_embd,
                        r,
                    )?;
                }
                exec.f8t_gemm(
                    lw.f8t_gate.as_ref().expect("f8t_gate present without gu"),
                    &sc.pf_e4q,
                    &sc.pf_e4rs,
                    &mut sc.pf_skfix,
                    &mut sc.pf_gate,
                    hp.n_embd,
                    n_ff,
                    r,
                )?;
                exec.f8t_gemm(
                    lw.f8t_up.as_ref().expect("f8t FFN planes built as a set"),
                    &sc.pf_e4q,
                    &sc.pf_e4rs,
                    &mut sc.pf_skfix,
                    &mut sc.pf_up,
                    hp.n_embd,
                    n_ff,
                    r,
                )?;
                exec.glu(&mut sc.pf_gate, &sc.pf_up, r * n_ff, hp.glu_act())?;
                exec.quantize_e4m3_row(&sc.pf_gate, &mut sc.pf_e4q, &mut sc.pf_e4rs, n_ff, r)?;
                exec.f8t_gemm(
                    lw.f8t_down.as_ref().expect("f8t FFN planes built as a set"),
                    &sc.pf_e4q,
                    &sc.pf_e4rs,
                    &mut sc.pf_skfix,
                    &mut sc.pf_proj,
                    n_ff,
                    hp.n_embd,
                    r,
                )?;
            } else if r == 1
                && lw.ffn_gate.data.len() > 48
                && lw.ffn_up.data.len() > 48
                && !r1_gu_off()
            {
                // One row: mma_ks fills 1 of its 16 MMA rows, so it streams the
                // plane at 579 GB/s where the plain repacked GEMM does 692 on
                // the identical 141 MB plane (measured on sm_86). `ffn_down`
                // three arms below already had exactly this carve-out; gate and
                // up never got it, so a single-stream decode ran two
                // tensor-core batch GEMMs per layer for nothing.
                //
                // It also RE-ALIGNS this lane with the dense r=1 walk in
                // forward.rs, which feeds f32 activations straight to the
                // repacked kernel - the mma arm's quantize_q8 was the only
                // thing making batch-r1 and dense-r1 numerically different
                // here. (Kept the mma arm reachable via PADDOCK_G4_NO_R1GU for
                // A/B work; it stays the right rung the moment r > 1.)
                exec.q8_0_gemm_repacked(&lw.ffn_gate, None, &sc.pf_normed, &mut sc.pf_gate, r)?;
                exec.q8_0_gemm_repacked(&lw.ffn_up, None, &sc.pf_normed, &mut sc.pf_up, r)?;
                exec.glu(&mut sc.pf_gate, &sc.pf_up, r * n_ff, hp.glu_act())?;
            } else if r <= 192 {
                exec.quantize_q8(&sc.pf_normed, &mut sc.pf_xq, &mut sc.pf_xs, r * hp.n_embd)?;
                exec.q8_0_gemm_mma_ks(
                    &lw.ffn_gate,
                    &sc.pf_xq,
                    &sc.pf_xs,
                    &mut sc.pf_skfix,
                    &mut sc.pf_gate,
                    r,
                )?;
                exec.q8_0_gemm_mma_ks(
                    &lw.ffn_up,
                    &sc.pf_xq,
                    &sc.pf_xs,
                    &mut sc.pf_skfix,
                    &mut sc.pf_up,
                    r,
                )?;
                exec.glu(&mut sc.pf_gate, &sc.pf_up, r * n_ff, hp.glu_act())?;
            } else {
                exec.quantize_q8_mmq(&sc.pf_normed, &mut sc.pf_yq, hp.n_embd, r)?;
                pf_mmq(
                    exec,
                    &lw.ffn_gate,
                    &sc.pf_yq,
                    &mut sc.pf_skfix,
                    &mut sc.pf_gate,
                    r,
                )?;
                pf_mmq(
                    exec,
                    &lw.ffn_up,
                    &sc.pf_yq,
                    &mut sc.pf_skfix,
                    &mut sc.pf_up,
                    r,
                )?;
                exec.glu(&mut sc.pf_gate, &sc.pf_up, r * n_ff, hp.glu_act())?;
            }
            if f8r || ffn_f8w_pf || ffn_f8row_pf || ffn_f8t {
                // handled above (down folded into the f8/rowwise/f8t arms)
            } else if ffn_f8w_r1 && lw.f8_down.is_some() {
                // Q8-reclaim + KEEP_F8W: stubbed original -> f8w gemv
                exec.f8_gemv_at(
                    lw.f8_down.as_ref().expect("f8_down checked above"),
                    &sc.pf_gate,
                    &mut sc.pf_proj,
                    0,
                    n_ff,
                    hp.n_embd,
                )?;
            } else if r == 1 && lw.ffn_down.data.len() > 48 {
                exec.q8_0_gemm_repacked(&lw.ffn_down, None, &sc.pf_gate, &mut sc.pf_proj, r)?;
            } else if r <= 192 {
                exec.quantize_q8(&sc.pf_gate, &mut sc.pf_xq, &mut sc.pf_xs, r * n_ff)?;
                exec.q8_0_gemm_mma_ks(
                    &lw.ffn_down,
                    &sc.pf_xq,
                    &sc.pf_xs,
                    &mut sc.pf_skfix,
                    &mut sc.pf_proj,
                    r,
                )?;
            } else {
                exec.quantize_q8_mmq(&sc.pf_gate, &mut sc.pf_yq, n_ff, r)?;
                pf_mmq(
                    exec,
                    &lw.ffn_down,
                    &sc.pf_yq,
                    &mut sc.pf_skfix,
                    &mut sc.pf_proj,
                    r,
                )?;
            }
            // x = (x + rmsnorm(proj)·w) · layer_output_scale - deferred into
            // the next layer's fused attn pre-norm when it can consume it
            // (next_fuses hoisted above the FFN block)
            if moe_fork {
                exec.side_end()?;
            }
            if let Some(moe) = &lw.moe {
                // 26B-A4B hybrid tail: pf_proj holds the shared branch's down
                // output; the deferral machinery can't absorb the two-branch
                // combine, so moe layers always resolve immediately (the
                // down-absorb arm excludes them - down_nz stays 1) and never
                // produce a pending_post.
                debug_assert!(down_nz == 1, "moe layer with absorbed down planes");
                g4_moe_tail(exec, sc, moe, &lw.ffn_post_norm, hp, lw.out_scale, r, true)?;
            } else if next_fuses || next_fuses_b32 {
                pending_post = Some((li, lw.out_scale, down_nz));
            } else {
                debug_assert!(down_nz == 1, "down absorbed without a fusing consumer");
                {
                    exec.rmsnorm_add_scale(
                        &mut sc.pf_x,
                        &sc.pf_proj,
                        &lw.ffn_post_norm,
                        hp.n_embd,
                        hp.post_norm_eps,
                        lw.out_scale,
                        r,
                    )?;
                }
            }
        }
        // safety net: a pending post that never found a consumer (cannot
        // happen with the li+1 bound, but keep the drain exact)
        if let Some((pi, ps, dnz)) = pending_post.take() {
            debug_assert!(dnz == 1, "drain cannot combine absorbed planes");
            let _ = dnz;
            if std::mem::take(&mut post_b16) {
                exec.rmsnorm_add_scale_p16(
                    &mut sc.pf_x,
                    &sc.pf_proj,
                    &self.layers[pi].ffn_post_norm,
                    hp.n_embd,
                    hp.post_norm_eps,
                    ps,
                    r,
                )?;
            } else {
                exec.rmsnorm_add_scale(
                    &mut sc.pf_x,
                    &sc.pf_proj,
                    &self.layers[pi].ffn_post_norm,
                    hp.n_embd,
                    hp.post_norm_eps,
                    ps,
                    r,
                )?;
            }
        }

        // batched LM head - logits STAY on device (the sampled lane never
        // reads them back)
        exec.rmsnorm_batch(
            &sc.pf_x,
            &self.output_norm,
            &mut sc.pf_normed,
            hp.n_embd,
            hp.eps,
            r,
        )?;
        let logits_dev: &mut CudaSlice<f32> =
            self.batch_logits.as_mut().expect("enable_batch allocates");
        if let Some(ht) = self.head_f8t.as_ref().filter(|_| r <= 192) {
            // f8t tile head across the decode AND wide-verify bands (it used
            // to stop at r<=64): 2048 row-tiles ride wmma at r<=24, tc5q in
            // the decode band and tc5r at 65+ - the K-split election stays
            // nz=1 at this tile count, so pf_skfix is never written. Replaces
            // the 1.5 GB Q8 dp4a read (r==1) / mma_ks (2..192) with the e4m3
            // logits class - at gemma's 128-160-row spec verify the mma_ks
            // BN32 col-tiling re-streams the head plane ~5x per round
            // (1.37 ms/launch - several percent of a whole spec window).
            // One logits class per binary: this also removes the q8-vs-e4m3
            // split the r-gate created between decode and verify rows.
            exec.quantize_e4m3_row(&sc.pf_normed, &mut sc.pf_e4q, &mut sc.pf_e4rs, hp.n_embd, r)?;
            exec.f8t_gemm(
                ht,
                &sc.pf_e4q,
                &sc.pf_e4rs,
                &mut sc.pf_skfix,
                logits_dev,
                hp.n_embd,
                hp.n_vocab,
                r,
            )?;
        } else if let Some(hr) = self.head_f8row.as_ref().filter(|_| r == 1) {
            // muse-glimmer: the r==1 Q8 head rode the crippled int8
            // dp4a class (~1 ms/token, 6.8% of the c1 decode tick). The per-row
            // e4m3 plane (head_f8t can't tile - vocab 202048 % 128 != 0) takes
            // it off that path: +5% c1. r==1 only - batched heads keep the Q8
            // mma_ks below, which out-batches the per-row route (per-row e4m3
            // regressed c4 -3%). Opt-in PADDOCK_MUSE_HEAD_F8ROW.
            exec.quantize_e4m3_row(&sc.pf_normed, &mut sc.pf_e4q, &mut sc.pf_e4rs, hp.n_embd, 1)?;
            exec.f8row_gemm(
                hr,
                &sc.pf_e4q,
                &sc.pf_e4rs,
                logits_dev,
                hp.n_embd,
                hp.n_vocab,
                1,
            )?;
        } else if let Some(hq) = self.head.q8() {
            if r == 1 {
                exec.q8_0_gemm_repacked(hq, None, &sc.pf_normed, logits_dev, r)?;
            } else if r <= 192 {
                exec.quantize_q8(&sc.pf_normed, &mut sc.pf_xq, &mut sc.pf_xs, r * hp.n_embd)?;
                exec.q8_0_gemm_mma_ks(hq, &sc.pf_xq, &sc.pf_xs, &mut sc.pf_skfix, logits_dev, r)?;
            } else {
                exec.quantize_q8_mmq(&sc.pf_normed, &mut sc.pf_yq, hp.n_embd, r)?;
                pf_mmq(exec, hq, &sc.pf_yq, &mut sc.pf_skfix, logits_dev, r)?;
            }
        } else {
            // bf16 head (muse-glimmer): the int8 rungs above all want a
            // repacked Q8 plane, so the whole r-ladder collapses to the
            // plane's own dispatch. The f8t twin is what puts this band back
            // on the tile route - see head_f8t's note in the loader.
            self.head.gemm(exec, &sc.pf_normed, logits_dev, r)?;
        }
        Ok(())
    }

    /// Host-logits lane: run the step and read everything back (the logit
    /// epilogue on host, as the single-stream paths do).
    pub(crate) fn forward_batch_host(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
    ) -> Result<Vec<f32>, GpuError> {
        let r = tokens.len();
        self.batch_step(tokens, positions)?;
        let hp = &self.hp;
        let logits_dev = self.batch_logits.as_ref().expect("enable_batch allocates");
        let mut logits = self.exec.to_host_len(logits_dev, r * hp.n_vocab)?;
        // The whole epilogue - scale then cap. This site used to open-code the
        // cap alone; see `logit_epilogue_dev`.
        hp.logit_epilogue(&mut logits);
        Ok(logits)
    }

    /// One full sampled decode tick, graph-capturable end to end: step body
    /// (device inputs) -> logit epilogue -> sample_rows. Everything it reads is
    /// a device buffer updated before replay.
    fn sampled_tick_body(&mut self, r: usize) -> Result<(), GpuError> {
        self.batch_step_body(r)?;
        let vocab = self.hp.n_vocab;
        let cap = self.hp.final_softcap;
        let lscale = self.hp.logit_scale;
        let exec = self.exec.clone();
        let logits_dev = self.batch_logits.as_mut().expect("enable_batch allocates");
        super::logit_epilogue_dev(&exec, logits_dev, r * vocab, lscale, cap)?;
        let (d_par, d_out) = self.samp.as_mut().expect("sampler buffers");
        exec.sample_rows(logits_dev, d_par, d_out, r, vocab)?;
        // mode-5 truncation rows draw fully on device from the same
        // softcapped plane. Unconditional inside the captured body - graphs
        // are static, so the launch always rides and each block early-outs
        // on mode != 5 (the sample_rows family's per-mode skip). Because
        // this body is the pipe's replay and the spec verify's tick, one
        // launch here re-admits trunc rows everywhere at once.
        if exec.has_sample_rows_t() && exec.has_sample_rows_p() {
            let d_tpar = self.samp_tpar.as_ref().expect("allocated with samp");
            exec.sample_rows_t(logits_dev, d_par, d_tpar, d_out, r, vocab)?;
            exec.sample_rows_p(logits_dev, d_par, d_tpar, d_out, r, vocab)?;
        }
        Ok(())
    }

    /// Device-sampled lane: softcap on device (categorical needs the capped
    /// distribution; greedy is monotone-invariant), sample_rows picks per-row
    /// tokens, and only Host-plan rows pay a logits readback.
    pub(crate) fn forward_batch_sampled_impl(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        plans: &[crate::generator::RowSample],
    ) -> Result<crate::generator::SampledStep, GpuError> {
        let ident: Vec<u32> = (0..tokens.len() as u32).collect();
        self.forward_batch_sampled_rows(tokens, positions, &ident, plans)
    }

    /// Slot-explicit sampled tick: row i decodes into KV slot `slots_map[i]`.
    /// The mixed tick compacts chunking slots out of the row set (a hole row
    /// would append garbage KV into the mid-prefill slot), so the mapping is
    /// no longer identity there.
    pub(crate) fn forward_batch_sampled_rows(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        slots_map: &[u32],
        plans: &[crate::generator::RowSample],
    ) -> Result<crate::generator::SampledStep, GpuError> {
        use crate::generator::{RowSample, SampledStep};
        use crate::sampler::DevicePlan;
        let r = tokens.len();
        assert_eq!(plans.len(), r, "one plan per row");
        assert_eq!(slots_map.len(), r, "one slot per row");
        let mut par = vec![0u32; r * 4];
        let mut tpar = vec![0u32; r * 4];
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
                // device truncation mode 5: full-device truncation draw (the captured
                // body launches pd_sample_rows_t after sample_rows). The
                // service emits TruncCat only on supports_device_trunc.
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
                // RS rows stay mode 0 (untouched): the resolve kernel
                // writes their ids after the tick
                RowSample::Device(DevicePlan::RsVerify { .. })
                | RowSample::Device(DevicePlan::RsTrunc { .. }) => {}
            }
        }
        let vocab = self.hp.n_vocab;
        let exec = self.exec.clone();
        // P63 seam attribution (PADDOCK_SPEC_DEBUG): the gap census prices
        // the spec_toks -> verify-graph GPU idle at 296us/round; break the
        // host tail down so the hoist targets the real component.
        let seam_t0 = paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG")
            .is_some()
            .then(std::time::Instant::now);
        // stage the sampler params before the step so the graph fast path
        // covers step+softcap+sample in one replay (tpar rides the same
        // protocol: the captured sample_rows_t reads it per replay; skipping
        // the upload on trunc-free ticks is safe - stale tpar bytes are
        // only read for mode-5 rows, and the mode word uploads fresh)
        {
            let (d_par, _) = self
                .samp
                .as_mut()
                .expect("enable_batch allocates sampler buffers");
            let mut v = d_par
                .try_slice_mut(0..r * 4)
                .ok_or_else(|| GpuError::Driver("samp par slice".into()))?;
            exec.stream
                .memcpy_htod(&par, &mut v)
                .map_err(|e| GpuError::Driver(e.to_string()))?;
        }
        if any_trunc {
            // engagement witness (bisect-trap law): once per process
            static DEV5: std::sync::Once = std::sync::Once::new();
            DEV5.call_once(|| {
                eprintln!("[trunc-dev5] engaged: r={r} (gemma4 full-device truncation sampling)");
            });
            let d_tpar = self.samp_tpar.as_mut().expect("allocated with samp");
            let mut v = d_tpar
                .try_slice_mut(0..r * 4)
                .ok_or_else(|| GpuError::Driver("samp tpar slice".into()))?;
            exec.stream
                .memcpy_htod(&tpar, &mut v)
                .map_err(|e| GpuError::Driver(e.to_string()))?;
        }
        let t_par = seam_t0.map(|t| t.elapsed().as_nanos() as u64);
        self.ensure_global_rows(slots_map, positions)?;
        let t_rows = seam_t0.map(|t| t.elapsed().as_nanos() as u64);
        self.batch_upload(tokens, positions, slots_map)?;
        let t_up = seam_t0.map(|t| t.elapsed().as_nanos() as u64);
        let gkey = (
            r,
            self.spec_k1.unwrap_or(1),
            self.spec_long,
            self.spec_shallow,
            kv_split_band(self.attn_pos_max),
        );
        if let (Some(t0), Some(tp), Some(tr), Some(tu)) = (seam_t0, t_par, t_rows, t_up) {
            let tl = t0.elapsed().as_nanos() as u64;
            use std::sync::atomic::{AtomicU64, Ordering};
            static ACC: [AtomicU64; 5] = [
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ];
            ACC[0].fetch_add(tp, Ordering::Relaxed);
            ACC[1].fetch_add(tr - tp, Ordering::Relaxed);
            ACC[2].fetch_add(tu - tr, Ordering::Relaxed);
            ACC[3].fetch_add(tl - tu, Ordering::Relaxed);
            let n = ACC[4].fetch_add(1, Ordering::Relaxed) + 1;
            if n.is_multiple_of(128) {
                let g = |i: usize| ACC[i].load(Ordering::Relaxed) as f64 / n as f64 / 1e3;
                tracing::info!(
                    "[g4-seam] n={n} us/round: par_htod={:.1} ensure_rows={:.1} upload={:.1} key={:.1}",
                    g(0),
                    g(1),
                    g(2),
                    g(3)
                );
            }
        }
        if !decode_graphs_on() {
            // debug: forced-eager tick (see decode_graphs_on)
            self.sampled_tick_body(r)?;
        } else if let Some(g) = self.decode_graphs.get(&gkey) {
            // captured decode tick: one launch replaces ~1500
            g.0.launch()
                .map_err(|e| GpuError::Driver(format!("graph launch: {e}")))?;
        } else if lazy_cap_on() && self.graph_seen.insert(gkey) {
            // seen-once lazy capture: wave transitions mint
            // one-off row-count gkeys, and the capture path below costs
            // ~6.4ms (pre-capture sync + record + instantiate) vs ~2ms for
            // the live tick - 29 storms x 6.4ms = 1.1% of the 128x128
            // window on gkeys that mostly never recur. First sight runs
            // eager; a gkey that comes back proves it's worth capturing.
            // (spec_pipe_arm already tolerates a missing graph - it just
            // declines to arm until the capture lands one round later.)
            self.sampled_tick_body(r)?;
        } else {
            // capture on repeat sight of this row count (records a live
            // step, so the tick still executes)
            exec.stream
                .synchronize()
                .map_err(|e| GpuError::Driver(format!("pre-capture sync: {e}")))?;
            exec.stream
                .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
                .map_err(|e| GpuError::Driver(format!("begin_capture: {e}")))?;
            let rec = self.sampled_tick_body(r);
            let graph = crate::gpu::end_capture_no_flags(&self.exec.stream)
                .map_err(|e| GpuError::Driver(format!("end_capture: {e}")));
            rec?;
            let graph =
                graph?.ok_or_else(|| GpuError::Driver("capture produced no graph".into()))?;
            let g = super::SendGraph(graph);
            g.0.launch()
                .map_err(|e| GpuError::Driver(format!("first graph launch: {e}")))?;
            self.decode_graphs.insert(gkey, g);
        }
        // DFlash feature append: this tick's rows just walked,
        // so their taps are in `zacc` - fuse and ring-append them before the
        // ids readback syncs, so the drafter's work overlaps it. A verify
        // tick defers (`dflash_defer`): only accepted rows may commit, and
        // its caller replays the accept rule in `dflash_spec_commit`.
        if !self.dflash_defer {
            self.dflash_append_features(tokens, positions, slots_map, None)?;
        }
        // rung B1: strip rounds consume the device accept's
        // compact strip instead - skip the full sampled-ids dtoh (one-shot
        // flag set by the strip caller; d_out stays valid for the accept
        // kernel enqueued right after this returns)
        let ids = if self.ids_skip {
            Vec::new()
        } else {
            let (_, d_out) = self.samp.as_ref().expect("sampler buffers");
            let ids_view = d_out
                .try_slice(0..r)
                .ok_or_else(|| GpuError::Driver("samp out slice".into()))?;
            exec.stream
                .clone_dtoh(&ids_view)
                .map_err(|e| GpuError::Driver(e.to_string()))?
        };
        // MTP h map: pf_normed now holds every row's post-output-norm hidden
        // (the drafter's h input). Record (pos, row) per slot - repeated
        // slots (spec verify chunks) keep the last row; the verify wrapper
        // re-points at the accepted-final row after acceptance.
        if self.mtp.is_some() {
            for e in self.spec_rows.iter_mut() {
                *e = None;
            }
            for (i, (&s, &p)) in slots_map.iter().zip(positions).enumerate() {
                if let Some(e) = self.spec_rows.get_mut(s as usize) {
                    *e = Some((p, i as u32));
                }
            }
            if paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some() {
                tracing::info!(
                    "[hrec] rows={} slot0={:?}",
                    slots_map.len(),
                    self.spec_rows.first().copied().flatten()
                );
            }
        }
        let mut host_rows = Vec::new();
        if plans.iter().any(|p| matches!(p, RowSample::Host)) {
            let logits_dev = self.batch_logits.as_ref().expect("enable_batch");
            for (i, p) in plans.iter().enumerate() {
                if matches!(p, RowSample::Host) {
                    let v = logits_dev
                        .try_slice(i * vocab..(i + 1) * vocab)
                        .ok_or_else(|| GpuError::Driver("host row slice".into()))?;
                    let row = exec
                        .stream
                        .clone_dtoh(&v)
                        .map_err(|e| GpuError::Driver(e.to_string()))?;
                    // device plane is already softcapped - host rows ride as-is
                    host_rows.push((i, row));
                }
            }
        }
        Ok(SampledStep { ids, host_rows })
    }

    /// Chunked prefill (the classic blocking batched pass froze every live
    /// stream behind the whole admission cohort - the c8 TTFT cost): queue
    /// the prompt; mixed ticks drain it FIFO under a row budget while decode
    /// rows keep flowing.
    pub(crate) fn prefill_begin_impl(
        &mut self,
        slot: usize,
        tokens: Vec<u32>,
    ) -> Result<(), GpuError> {
        if slot >= self.n_slots {
            return Err(GpuError::Driver(format!(
                "slot {slot} >= enabled {}",
                self.n_slots
            )));
        }
        if tokens.is_empty() || tokens.len() > self.max_ctx {
            return Err(GpuError::Driver(format!(
                "chunked prompt is {} tokens but max_ctx is {}",
                tokens.len(),
                self.max_ctx
            )));
        }
        // a queued entry for the same slot is STALE (the scheduler enforces
        // one chunk per live slot - a duplicate means the old request died
        // and the slot was reused): evict it instead of wedging the slot
        self.chunked.retain(|c| c.slot != slot);
        if self.chunked.len() >= crate::service::max_chunks_inflight() {
            return Err(GpuError::Driver("chunked prefill queue is full".into()));
        }
        self.chunked.push(ChunkedPrefill { slot, tokens });
        Ok(())
    }

    /// One MIXED tick: advance the chunk queue by ~`budget` rows (whole
    /// prompts, FIFO, always at least one - the existing coalesced batch
    /// pass does the work, prefix resume included), then run the live decode
    /// rows as a compacted slot-explicit sampled tick. Two weight-amortized
    /// passes; first tokens stagger out of the cohort instead of waiting for
    /// all of it.
    pub(crate) fn forward_mixed_sampled_impl(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
        plans: &[crate::generator::RowSample],
    ) -> Result<(crate::generator::SampledStep, Vec<(usize, Vec<f32>, usize)>), GpuError> {
        use crate::generator::SampledStep;
        assert_eq!(plans.len(), decodes.len(), "one plan per decode row");
        // PADDOCK_PF_CAP forces THIN single-prompt prefill passes
        // (one ~146-row prompt per mixed tick) so the captured single-run
        // graph actually recurs - the default budget admits a wide 8-12-prompt
        // pass (r~1500, runs 7-12) that is multi-run and never replays. This
        // trades the wide pass's weight amortization for capture-cheap fine
        // chunking (the low-TTFT Pareto point); measured vs the wide-eager base.
        let cap = match pf_cap() {
            Some(n) => budget.min(n),
            None => budget,
        }
        .clamp(1, mixed_tick_rows());
        let mut batch: Vec<(usize, Vec<u32>)> = Vec::new();
        let mut used = 0usize;
        // PEEK, don't drain: the queue commits only once the chunk pass
        // succeeds (`self.chunked.drain(..batch.len())` at the success
        // point) - same contract as the unified/spec sites. The earlier
        // conversion MISSED this site's loop while adding its post-success
        // drain: the destructive remove(0) + drain double-consumed the
        // queue, silently deleting the next queued prompt(s) whenever two
        // multi-chunk cold prefills were queued together - the deleted
        // slot then decoded over unwritten KV (garbage output, illegal
        // address, poisoned server). Only greedy/non-spec serving routes
        // here (unified_ok covers the spec default), which is why every
        // temperature-0.7 bench missed it.
        // Cap rule unchanged: stop before exceeding the cap (always take
        // at least one) - an overshooting batch splits into a full chunk
        // + a small tail chunk on the sub-1024 mmq rung.
        for c in self.chunked.iter() {
            let next = c.tokens.len();
            if used > 0 && used + next > cap {
                break;
            }
            used += next;
            batch.push((c.slot, c.tokens.clone()));
        }
        //  lever-sizing (PADDOCK_MIXTIME): host-wall the two SPLIT-path
        // forwards separately so the c32 tail-attribution aggregate can size
        // stream-overlap's ceiling. Both calls end in a readback (logits d2h /
        // sampled-ids d2h), so host wall == that forward's true blocking wall -
        // exactly the serialization overlap would hide. pf_us=0 marks a pure
        // decode tick. Zero behaviour change: gated, times only, no reorder.
        let mixtime = paddock_models::dev_var_os!("PADDOCK_MIXTIME").is_some();
        let pf_rows = used;
        let mut pf_us: u128 = 0;
        let mut finished = Vec::new();
        if !batch.is_empty() {
            let t = mixtime.then(std::time::Instant::now);
            let outs = self.forward_prefill_batch_impl(&batch)?;
            if let Some(t) = t {
                pf_us = t.elapsed().as_micros();
            }
            self.chunked.drain(..batch.len());
            for ((slot, toks), logits) in batch.iter().zip(outs) {
                finished.push((*slot, logits, toks.len()));
            }
        }
        if decodes.is_empty() {
            if mixtime && pf_rows > 0 {
                tracing::info!("[mixtime] pf_rows={pf_rows} pf_us={pf_us} dec_rows=0 dec_us=0");
            }
            return Ok((
                SampledStep {
                    ids: Vec::new(),
                    host_rows: Vec::new(),
                },
                finished,
            ));
        }
        let tokens: Vec<u32> = decodes.iter().map(|d| d.1).collect();
        let positions: Vec<u32> = decodes.iter().map(|d| d.2).collect();
        let slots_map: Vec<u32> = decodes.iter().map(|d| d.0 as u32).collect();
        if paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some() {
            tracing::info!("[mixdec] nd={}", decodes.len());
        }
        let dt = mixtime.then(std::time::Instant::now);
        let step = self.forward_batch_sampled_rows(&tokens, &positions, &slots_map, plans)?;
        if let Some(dt) = dt {
            let dec_us = dt.elapsed().as_micros();
            tracing::info!(
                "[mixtime] pf_rows={pf_rows} pf_us={pf_us} dec_rows={} dec_us={dec_us}",
                decodes.len()
            );
        }
        Ok((step, finished))
    }

    /// Speculative mixed tick (v2): the decode rows ride the
    /// UNIFIED pass as VERIFY CHUNKS - [pending, drafts...] per slot laid at
    /// the FRONT of the same weight-amortized stream as the prompt chunk
    /// (gemma4 default-ons PADDOCK_UNIFIED, so the two-forward mixed shape
    /// never runs; composing with it was the v1 mistake). The front rows go
    /// through prefill_layers' decode arm - kv_append for all front rows
    /// precedes attention with per-row bounds, the same dispatch the pure
    /// verify relies on, so multi-row-per-slot chunks are naturally causal.
    /// unified_decode_head samples every verify row device-side; the h map
    /// re-points at each slot's ACCEPTED-final row when the stream stayed
    /// single-chunk (pf_normed[0..nd) still holds the verify hiddens),
    /// keeping the next tick's draft chain warm.
    /// Blocking form: launch + wait in one call (the pre-Phase-72 contract;
    /// still the route when the service's begin() declined).
    pub(crate) fn forward_mixed_spec_plans_impl(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
        budget: usize,
        plans: &[crate::sampler::DevicePlan],
        fin_plans: &[(usize, crate::generator::RowSample)],
    ) -> Result<
        (
            Option<Vec<u32>>,
            Vec<(usize, crate::generator::FinishSample, usize)>,
        ),
        GpuError,
    > {
        if paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some() {
            tracing::info!(
                "[mixspec] enter reqs={} first_pos={:?}",
                reqs.len(),
                reqs.first().map(|(_, p2, _)| *p2)
            );
        }
        match self.forward_mixed_spec_launch_impl(reqs, budget, plans, fin_plans)? {
            MixLaunch::Launched => self.forward_mixed_spec_wait_impl(),
            MixLaunch::Fallback(picks, finished) => Ok((picks, finished)),
        }
    }

    /// Issue-ahead: enqueue the whole mixed round (chunk pass +
    /// verify head + device sampling) and return before the picks readback,
    /// so the service can run the previous round's deferred host work
    /// (finish_prefill et al) inside this round's GPU window. Fallback
    /// carries the paths that already produced a result (declines, pure
    /// verify, pure prefill) - the caller returns it as-is.
    pub(crate) fn forward_mixed_spec_launch_impl(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
        budget: usize,
        plans: &[crate::sampler::DevicePlan],
        fin_plans: &[(usize, crate::generator::RowSample)],
    ) -> Result<MixLaunch, GpuError> {
        if paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some() {
            tracing::info!(
                "[mixspec] LAUNCH reqs={} first_pos={:?}",
                reqs.len(),
                reqs.first().map(|(_, p2, _)| *p2)
            );
        }
        use crate::generator::RowSample;
        let nd: usize = reqs.iter().map(|(_, _, c)| c.len()).sum();
        let cap_rows = self
            .batch_logits
            .as_ref()
            .map(|b| b.len() / self.hp.n_vocab)
            .unwrap_or(0);
        if reqs.is_empty()
            || nd == 0
            || nd > cap_rows
            || nd >= self.pf_rows
            || reqs.len() > self.n_slots
        {
            return Ok(MixLaunch::Fallback(None, Vec::new()));
        }
        for (slot, start, chunk) in reqs {
            if *slot >= self.n_slots || chunk.is_empty() || start + chunk.len() > self.max_ctx {
                return Ok(MixLaunch::Fallback(None, Vec::new()));
            }
        }
        assert_eq!(plans.len(), nd, "one plan per verify row");
        let row_plans: Vec<RowSample> = plans.iter().map(|p| RowSample::Device(*p)).collect();
        // prompt chunk drain (the unified rules)
        let cap = budget.clamp(1, mixed_tick_rows());
        let mut batch: Vec<(usize, Vec<u32>)> = Vec::new();
        let mut used = 0usize;
        // PEEK, don't drain: the queue commits only once the chunk pass
        // succeeds (`self.chunked.drain(..batch.len())` at the success
        // point). The old destructive take lost these entries when the pass
        // hit PoolExhausted - the scheduler's `chunking` set then pointed at
        // slots this queue no longer held, and the serve spun forever on
        // no-op mixed ticks (found live as a wide-batch wedge).
        for c in self.chunked.iter() {
            let next = c.tokens.len();
            if used > 0 && used + next > cap {
                break;
            }
            used += next;
            batch.push((c.slot, c.tokens.clone()));
        }
        if batch.is_empty() {
            // nothing chunking after all - the pure (graph-captured) verify
            let picks = self.forward_spec_rows_impl(reqs, &row_plans)?;
            return Ok(MixLaunch::Fallback(picks, Vec::new()));
        }
        // Verify-front padding to uniform k1: ragged mixed rounds
        // (adaptive per-slot k_now, the k1a=None [g4-mspec] class) left
        // spec_k1 unset, so their verify rows attended through the per-row
        // v9q walk - 140us x 8.5k at the fresh imax capture, ~2/3 of it
        // recoverable through the krs chunk walk (the churn attribution's
        // ratio). Pad each chunk to the round max exactly like the pure
        // path (spec.rs pad_to): repeat the last token at advancing
        // positions; pad rows sample Greedy and are discarded - their KV
        // writes sit beyond the committed cursor and are overwritten.
        // Picks compact back to the service's raw layout in _wait.
        let kmax = reqs.iter().map(|(_, _, c)| c.len()).max().unwrap_or(0);
        let pad_to = (nd > 64
            && kmax > 1
            && reqs.len() * kmax <= cap_rows
            && reqs.len() * kmax < self.pf_rows)
            .then_some(kmax);
        let nd = pad_to.map_or(nd, |k| reqs.len() * k);
        let row_plans: Vec<RowSample> = if let Some(k1p) = pad_to {
            let mut v = Vec::with_capacity(nd);
            let mut flat = 0usize;
            for (_, _, chunk) in reqs {
                for i in 0..k1p {
                    v.push(if i < chunk.len() {
                        row_plans[flat + i]
                    } else {
                        RowSample::Device(crate::sampler::DevicePlan::Greedy)
                    });
                }
                flat += chunk.len();
            }
            v
        } else {
            row_plans
        };
        let n_embd = self.hp.n_embd;
        let t_adm = std::time::Instant::now();
        let mut starts = Vec::with_capacity(batch.len());
        for (slot, toks) in &batch {
            assert!(
                *slot < self.n_slots,
                "slot {slot} >= enabled {}",
                self.n_slots
            );
            self.gpool_clear_slot(*slot);
            starts.push(self.prefix_resume(*slot, toks)?);
            self.ensure_global_rows(&[*slot as u32], &[(toks.len() - 1) as u32])?;
        }
        if paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some() {
            let ms = t_adm.elapsed().as_secs_f64() * 1e3;
            if ms > 1.0 {
                tracing::info!("[adm] {} prompts {:.2}ms", batch.len(), ms);
            }
        }
        {
            let dslots: Vec<u32> = reqs.iter().map(|(s2, _, _)| *s2 as u32).collect();
            let dpos: Vec<u32> = reqs
                .iter()
                .map(|(_, st2, c)| (st2 + pad_to.unwrap_or(c.len()) - 1) as u32)
                .collect();
            self.ensure_global_rows(&dslots, &dpos)?;
        }
        // row stream: VERIFY rows first (padded to uniform k1 when pad_to
        // is set), then each prompt's divergent tail
        let mut rows: Vec<(u32, u32, u32, usize)> = Vec::with_capacity(nd);
        for (slot, start, chunk) in reqs {
            let clen = pad_to.unwrap_or(chunk.len());
            let last = *chunk.last().expect("non-empty chunk checked above");
            for i in 0..clen {
                let t = chunk.get(i).copied().unwrap_or(last);
                rows.push((*slot as u32, (*start + i) as u32, t, usize::MAX));
            }
        }
        let mut last_row = vec![0usize; batch.len()];
        for (it, ((slot, toks), &start)) in batch.iter().zip(&starts).enumerate() {
            for (j, &t) in toks[start..].iter().enumerate() {
                rows.push((*slot as u32, (start + j) as u32, t, it));
            }
            last_row[it] = rows.len() - 1;
        }
        let _row_bytes = (n_embd / 32) * 34;
        // verify-fold rung A: when every slot's verify chunk is
        // the same depth (>1), the front rows form exactly the slot-major
        // k1-chunk layout the spec attention kernels take - arm spec_k1 for
        // the unified pass so prefill_layers' decode arm attends them via
        // one KV walk per chunk instead of k1 per-row window re-walks
        // - per-row v9q on a mixed tick is the largest slice of the
        // decode-attn complex, and the same rows cost ~2/3 as much through
        // the krs arm on a pure verify tick. spec_long mirrors the
        // pure tick's long-KV band for the split election. Cleared right
        // after the armed call - a stale value would mis-gate the next
        // batch tick's spec arm.
        if paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some() {
            let ms = t_adm.elapsed().as_secs_f64() * 1e3;
            if ms > 2.0 {
                tracing::info!("[mix-impl] pre-loop {ms:.2}ms rows {}", rows.len());
            }
        }
        let k1u = reqs.first().map(|(_, _, c)| c.len()).unwrap_or(0);
        let mixed_k1 = pad_to
            .or_else(|| (k1u > 1 && reqs.iter().all(|(_, _, c)| c.len() == k1u)).then_some(k1u));
        let mixed_long = mixed_k1.is_some()
            && reqs
                .iter()
                .map(|(_, st2, c)| st2 + c.len() - 1)
                .max()
                .is_some_and(|m| m >= spec_fin_pos_floor());
        let mut out: Vec<Vec<f32>> = vec![Vec::new(); batch.len()];
        let mut fin_staged = vec![false; batch.len()];
        // a staged finisher with a service Device plan samples on
        // device (one sample_rows over the pf_fin prefix after the loop)
        let mut fin_dev: Vec<Option<crate::sampler::DevicePlan>> = vec![None; batch.len()];
        let mut head_launched = false;
        let mut base = 0usize;
        for chunk in rows.chunks(self.pf_rows) {
            let r = chunk.len();
            let nd_here = if base == 0 { nd } else { 0 };
            let positions: Vec<u32> = chunk.iter().map(|x| x.1).collect();
            let slots_v: Vec<u32> = chunk.iter().map(|x| x.0).collect();
            let mut runs: Vec<(usize, usize)> = Vec::new();
            for (i, x) in chunk.iter().enumerate().skip(nd_here) {
                match runs.last_mut() {
                    Some((off, n)) if chunk[*off].0 == x.0 && chunk[*off].3 == x.3 => *n += 1,
                    _ => runs.push((i, 1)),
                }
            }
            let spans = super::forward::swa_spans(self.swa_span, &runs);
            {
                let sc = &mut self.scratch;
                let e = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
                self.exec
                    .stream
                    .memcpy_htod(&positions, &mut sc.pf_pos)
                    .map_err(e)?;
                self.exec
                    .stream
                    .memcpy_htod(&positions, &mut sc.pf_attn_pos)
                    .map_err(e)?;
                self.exec
                    .stream
                    .memcpy_htod(&slots_v, &mut sc.pf_slots)
                    .map_err(e)?;
            }
            {
                // ONE-kernel embed gather (the per-row dequant+copy loop was
                // 2 host launches/row - the c32 mixed-tick launch wall)
                let toks: Vec<u32> = chunk.iter().map(|x| x.2).collect();
                let sc = &mut self.scratch;
                let mut v = sc
                    .pf_toks
                    .try_slice_mut(0..r)
                    .ok_or_else(|| GpuError::Driver("pf_toks slice".into()))?;
                self.exec
                    .stream
                    .memcpy_htod(&toks, &mut v)
                    .map_err(|e| GpuError::Driver(e.to_string()))?;
                // Drafter-fold: an armed async chain
                // means the service built the verify chunk VALUES as
                // placeholders - assemble the real tokens on device into
                // the pf_toks front (verify rows lead chunk 0), the pure
                // tick's pd_spec_toks scheme exactly: ragged clen, cold
                // slots pad with pending. Stream-ordered before the embed
                // gather; the plan stays armed for the wait-side fetch.
                if nd_here > 0
                    && let Some(plan) = &self.spec_async
                {
                    if reqs.len() > 128 {
                        return Err(GpuError::Driver(
                            "mixed async round: n > asm_meta cap".into(),
                        ));
                    }
                    let vn = reqs.len();
                    let mut meta = vec![0u32; 5 * vn];
                    let mut vbase = 0u32;
                    let mut cmax = 0u32;
                    for (i, (slot, _, ch)) in reqs.iter().enumerate() {
                        // padded rows assemble like the pure path's pad_to
                        // case: clen spans the pad, ndr counts real drafts
                        let clen = pad_to.unwrap_or(ch.len()) as u32;
                        let ci = plan.chain_slot.iter().position(|&s| s == *slot as u32);
                        let (srow, ndr) = match ci {
                            Some(ci) => (ci as u32, (ch.len() - 1).min(plan.k_use) as u32),
                            None => (0, 0),
                        };
                        meta[i] = ch[0];
                        meta[vn + i] = srow;
                        meta[2 * vn + i] = ndr;
                        meta[3 * vn + i] = clen;
                        meta[4 * vn + i] = vbase;
                        vbase += clen;
                        cmax = cmax.max(clen);
                    }
                    debug_assert_eq!(vbase as usize, nd, "meta rows == verify rows");
                    let rr = plan.rr;
                    let m = self.mtp.as_mut().expect("armed plan requires mtp");
                    {
                        let mut mv = m
                            .asm_meta
                            .try_slice_mut(0..5 * vn)
                            .ok_or_else(|| GpuError::Driver("asm_meta slice".into()))?;
                        self.exec
                            .stream
                            .memcpy_htod(&meta, &mut mv)
                            .map_err(|e| GpuError::Driver(e.to_string()))?;
                    }
                    self.exec.spec_toks(
                        &m.asm_meta,
                        &m.out,
                        &mut sc.pf_toks,
                        vn,
                        cmax as usize,
                        rr,
                    )?;
                }
                self.exec.embed_gather_plane(
                    &self.token_embd,
                    &sc.pf_toks,
                    &mut sc.pf_x,
                    n_embd,
                    r,
                    self.hp.embd_scale(),
                )?;
                super::GpuGemma4::embd_preamble(
                    &self.exec,
                    &self.hp,
                    self.embd_ones.as_ref(),
                    &mut sc.pf_x,
                    r,
                )?;
            }
            if nd_here > 0 {
                self.spec_k1 = mixed_k1;
                self.spec_long = mixed_long;
            }
            let pl = self.prefill_layers(r, &runs, &spans, nd_here);
            self.spec_k1 = None;
            self.spec_long = false;
            pl?;
            if nd_here > 0 {
                // enqueue only - the ids readback moves to _wait
                self.unified_decode_head_launch(nd, &row_plans)?;
                head_launched = true;
            }
            for (it, &lr) in last_row.iter().enumerate() {
                if lr >= base && lr < base + r {
                    if it < 64 {
                        self.logits_head_stage(lr - base, it)?;
                        fin_staged[it] = true;
                        fin_dev[it] = fin_plans.iter().find_map(|&(s2, p)| {
                            (s2 == batch[it].0)
                                .then_some(match p {
                                    crate::generator::RowSample::Device(d) => Some(d),
                                    _ => None,
                                })
                                .flatten()
                        });
                    } else {
                        out[it] = self.logits_from_pf_row(lr - base)?;
                    }
                }
            }
            base += r;
        }
        // Finisher device sampling: one sample_rows over the staged
        // pf_fin prefix (rows are softcapped at stage time - the same
        // distribution the host pick saw). Host-planned staged rows ride
        // mode 0 (the kernel skips them; their logits read back in _wait).
        if fin_dev.iter().any(|d| d.is_some()) {
            use crate::sampler::DevicePlan;
            let hi = fin_staged.iter().rposition(|&s| s).map_or(0, |i| i + 1);
            let mut par = vec![0u32; hi * 4];
            let mut tpar = vec![0u32; hi * 4];
            let mut any_trunc = false;
            for (it, d) in fin_dev.iter().enumerate().take(hi) {
                match d {
                    Some(DevicePlan::Greedy) => par[it * 4 + 2] = 1,
                    Some(DevicePlan::Categorical { inv_t, u }) => {
                        par[it * 4] = inv_t.to_bits();
                        par[it * 4 + 1] = u.to_bits();
                        par[it * 4 + 2] = 2;
                    }
                    // device truncation mode 5: trunc finishers draw on device from the
                    // staged (already softcapped) pf_fin rows
                    Some(DevicePlan::TruncCat {
                        inv_t,
                        u,
                        k,
                        top_p,
                        min_p,
                    }) => {
                        par[it * 4] = inv_t.to_bits();
                        par[it * 4 + 1] = u.to_bits();
                        par[it * 4 + 2] = if *k >= 1 && *k <= 64 { 5 } else { 6 };
                        tpar[it * 4] = *k;
                        tpar[it * 4 + 1] = top_p.to_bits();
                        tpar[it * 4 + 2] = min_p.to_bits();
                        any_trunc = true;
                    }
                    // RS never reaches finishers; None = host row, skip
                    Some(DevicePlan::RsVerify { .. }) | Some(DevicePlan::RsTrunc { .. }) | None => {
                    }
                }
            }
            if self.fin_samp.is_none() {
                // 64 finishers - must cover the pf_fin staging cap (the
                // it<64 lift; the old 8 here silently OOB'd device sampling
                // whenever a mixed tick finished >8 prompts)
                self.fin_samp = Some((
                    self.exec
                        .alloc_u32(64 * 4)
                        .map_err(|e| GpuError::Driver(e.to_string()))?,
                    self.exec
                        .alloc_u32(64)
                        .map_err(|e| GpuError::Driver(e.to_string()))?,
                ));
            }
            if self.fin_tpar.is_none() {
                self.fin_tpar = Some(
                    self.exec
                        .alloc_u32(64 * 4)
                        .map_err(|e| GpuError::Driver(e.to_string()))?,
                );
            }
            {
                let (d_par, _) = self.fin_samp.as_mut().expect("just filled");
                let mut v = d_par
                    .try_slice_mut(0..hi * 4)
                    .ok_or_else(|| GpuError::Driver("fin_samp par slice".into()))?;
                self.exec
                    .stream
                    .memcpy_htod(&par, &mut v)
                    .map_err(|e| GpuError::Driver(e.to_string()))?;
            }
            if any_trunc {
                let d_tpar = self.fin_tpar.as_mut().expect("just filled");
                let mut v = d_tpar
                    .try_slice_mut(0..hi * 4)
                    .ok_or_else(|| GpuError::Driver("fin_tpar slice".into()))?;
                self.exec
                    .stream
                    .memcpy_htod(&tpar, &mut v)
                    .map_err(|e| GpuError::Driver(e.to_string()))?;
            }
            let vocab = self.hp.n_vocab;
            let exec = self.exec.clone();
            let (d_par, d_out) = self.fin_samp.as_mut().expect("just filled");
            exec.sample_rows(&self.scratch.pf_fin, d_par, d_out, hi, vocab)?;
            if any_trunc {
                let d_tpar = self.fin_tpar.as_ref().expect("just filled");
                exec.sample_rows_t(&self.scratch.pf_fin, d_par, d_tpar, d_out, hi, vocab)?;
                exec.sample_rows_p(&self.scratch.pf_fin, d_par, d_tpar, d_out, hi, vocab)?;
            }
        }
        // cache the finished prompts (direct ring->pool checkpoint + zero-copy
        // radix insert)
        let t_ins = std::time::Instant::now();
        for (it, (slot, toks)) in batch.iter().enumerate() {
            let cut = self.prefix_cut(toks.len(), starts[it]);
            self.prefix_insert(*slot, toks, cut)?;
        }
        if paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some() {
            let ms = t_ins.elapsed().as_secs_f64() * 1e3;
            if ms > 1.0 {
                tracing::info!("[pfx-ins] {} prompts {:.2}ms", batch.len(), ms);
            }
        }
        let zeros = vec![0u32; self.pf_rows];
        self.exec
            .stream
            .memcpy_htod(&zeros, &mut self.scratch.pf_slots)
            .map_err(|e| GpuError::Driver(e.to_string()))?;
        assert!(head_launched, "chunk 0 carries the verify rows");
        self.mix_inflight = Some(MixInflight {
            nd,
            row_plans,
            reqs: reqs.to_vec(),
            rows_len: rows.len(),
            mixed_k1,
            fin_staged,
            fin_dev,
            out,
            batch,
        });
        Ok(MixLaunch::Launched)
    }

    /// Issue-ahead wait half: picks readback + the h-map re-point +
    /// the finished-prompt logits reads. Everything here either blocks on
    /// the stream or needs picks; everything the host can do without them
    /// ran in _launch (or runs deferred at the service between the two).
    pub(crate) fn forward_mixed_spec_wait_impl(
        &mut self,
    ) -> Result<
        (
            Option<Vec<u32>>,
            Vec<(usize, crate::generator::FinishSample, usize)>,
        ),
        GpuError,
    > {
        let MixInflight {
            nd,
            row_plans,
            mut reqs,
            rows_len,
            mixed_k1,
            fin_staged,
            fin_dev,
            mut out,
            batch,
        } = self.mix_inflight.take().expect("wait without launch");
        let step = self.unified_decode_head_read(nd, &row_plans)?;
        // [mix-wait]: post-picks host work in the wait half (draft fetch +
        // placeholder fill + h-map replay + finisher logits reads) - part
        // of the ~6.8ms/boundary GPU idle the fresh capture still shows
        let t_wait = std::time::Instant::now();
        let picks = step.ids;
        // Drafter-fold: an armed round carried PLACEHOLDER chunk
        // values - fetch the real drafts (the picks dtoh above already
        // drained the stream, so this costs the copy) and fill them in so
        // the accept replay below compares what verify actually saw.
        // NON-consuming: the plan stays armed for the service's own
        // spec_draft_fetch + reqs patch after this returns.
        if self.spec_async.is_some() {
            self.spec_async_drafts()?;
            let p = self.spec_async.as_ref().expect("armed");
            let (dr, cs, keep) = (p.fetched.clone(), p.chain_slot.clone(), p.keep.clone());
            if let Some(dr) = dr {
                for (slot, _, chunk) in reqs.iter_mut() {
                    let d = cs
                        .iter()
                        .position(|&s| s == *slot as u32)
                        .and_then(|ci| dr.get(keep[ci]));
                    if let Some(d) = d {
                        for j in 1..chunk.len() {
                            if j - 1 < d.len() {
                                chunk[j] = d[j - 1];
                            }
                        }
                    }
                }
            }
        }
        // DFlash: commit this round's ACCEPTED rows to the drafter ring.
        // The PURE verify path does it (spec.rs -> dflash_spec_commit); this
        // one never did, and that is what kept DFlash off at batch. A slot
        // that decodes through a prompt chunk loses that row, and
        // `dflash_warm` demands coverage ending exactly at p, so one missed
        // row decolds the slot for the rest of the request - measured at c8
        // as coverage restarting at 72 and `eligible=0` thereafter.
        //
        // Two guards, both about `zacc` holding these rows:
        //  - single-chunk, same reason the h-map below needs it (a later
        //    chunk's taps overwrite the accumulator), and
        //  - packed stride, because dflash_spec_commit walks the chunks
        //    packed and append maps toks[i] -> zacc[i] 1:1. A padded round
        //    whose chunks are already k1 wide is packed; a ragged padded one
        //    is skipped rather than fused against the wrong rows.
        if self.dflash.is_some() && rows_len <= self.pf_rows {
            let packed = reqs
                .iter()
                .all(|(_, _, c)| c.len() == mixed_k1.unwrap_or(c.len()));
            let total: usize = reqs.iter().map(|(_, _, c)| c.len()).sum();
            if packed && total <= picks.len() {
                if paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some() {
                    tracing::info!(
                        "[dflash-mix] COMMIT reqs={} first_pos={:?}",
                        reqs.len(),
                        reqs.first().map(|(_, p2, _)| *p2)
                    );
                }
                self.dflash_spec_commit(&reqs, &picks)?;
            } else if paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some() {
                tracing::info!(
                    "[dflash-mix] SKIPPED commit packed={packed} total={total} picks={} \
                     rows_len={rows_len} pf_rows={}",
                    picks.len(),
                    self.pf_rows
                );
            }
        }
        // h map: re-point at each slot's ACCEPTED-final row (the service
        // replays the same accept-while-match rule). Valid only when the
        // stream stayed single-chunk - later chunks overwrote pf_normed.
        if self.mtp.is_some() {
            for e in self.spec_rows.iter_mut() {
                *e = None;
            }
            if rows_len <= self.pf_rows {
                let mut b2 = 0usize;
                let mut accepted = 0usize;
                for (slot, start, chunk) in &reqs {
                    let mut a = 0usize;
                    while a + 1 < chunk.len() && chunk[a + 1] == picks[b2 + a] {
                        a += 1;
                    }
                    accepted += a + 1;
                    if let Some(e) = self.spec_rows.get_mut(*slot) {
                        *e = Some(((start + a) as u32, (b2 + a) as u32));
                    }
                    // padded rounds stride by the padded k1 - the row index
                    // above must point into the PADDED pf_normed stream
                    b2 += mixed_k1.unwrap_or(chunk.len());
                }
                if paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some() {
                    let r0 = reqs.first().map(|(s2, _, _)| *s2).unwrap_or(0);
                    tracing::info!(
                        "[g4-mspec] n={} rows={} k1a={:?} committed={} ({:.2}/slot) prompt_rows={} rec0=(slot{} {:?})",
                        reqs.len(),
                        nd,
                        mixed_k1,
                        accepted,
                        accepted as f64 / reqs.len() as f64,
                        rows_len - nd,
                        r0,
                        self.spec_rows.get(r0).copied().flatten()
                    );
                }
            }
        }
        // One batched dtoh for the HOST-planned staged finishers (device-planned
        // rows read a 4-byte id from the fin_samp strip instead of their
        // [1, vocab] logits row - on the all-device round the 1-8MB pageable
        // copy disappears entirely)
        let host_mask: Vec<bool> = fin_staged
            .iter()
            .zip(&fin_dev)
            .map(|(&st, d)| st && d.is_none())
            .collect();
        if host_mask.iter().any(|&h| h) {
            for (it, logits) in self
                .logits_finish_read_all(&host_mask)?
                .into_iter()
                .enumerate()
            {
                if let Some(l) = logits {
                    out[it] = l;
                }
            }
        }
        let fin_ids: Vec<u32> = if fin_dev.iter().any(|d| d.is_some()) {
            let hi = fin_staged.iter().rposition(|&s| s).map_or(0, |i| i + 1);
            let (_, d_out) = self.fin_samp.as_ref().expect("sampled at launch");
            let v = d_out
                .try_slice(0..hi)
                .ok_or_else(|| GpuError::Driver("fin_samp out slice".into()))?;
            self.exec
                .stream
                .clone_dtoh(&v)
                .map_err(|e| GpuError::Driver(e.to_string()))?
        } else {
            Vec::new()
        };
        self.chunked.drain(..batch.len());
        use crate::generator::FinishSample;
        let finished = batch
            .into_iter()
            .zip(out)
            .enumerate()
            .map(|(it, ((slot, toks), logits))| {
                let fs = if fin_dev.get(it).copied().flatten().is_some() {
                    FinishSample::Sampled(fin_ids[it])
                } else {
                    FinishSample::Logits(logits)
                };
                (slot, fs, toks.len())
            })
            .collect();
        // padded rounds: compact picks back to the service's RAW chunk
        // layout (the service walks by its own chunk lens; pad-row samples
        // are discarded here, exactly the pure path's contract)
        let picks = match mixed_k1 {
            Some(k1p) if reqs.iter().any(|(_, _, c)| c.len() != k1p) => {
                let mut v = Vec::with_capacity(reqs.iter().map(|(_, _, c)| c.len()).sum());
                let mut b = 0usize;
                for (_, _, chunk) in &reqs {
                    v.extend_from_slice(&picks[b..b + chunk.len()]);
                    b += k1p;
                }
                v
            }
            _ => picks,
        };
        if paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some() {
            let ms = t_wait.elapsed().as_secs_f64() * 1e3;
            if ms > 1.5 {
                tracing::info!("[mix-wait] {ms:.2}ms post-picks host");
            }
        }
        Ok((Some(picks), finished))
    }

    /// True unified prefill+decode tick (qwen35 PADDOCK_UNIFIED shape): the
    /// live decode rows ride the same weight-amortized pass as the prompt
    /// chunk instead of a second forward. Row layout: [nd decode rows] +
    /// [prompt tail rows] - decode rows lead so their attention reads the
    /// q buffer natively at offset 0 through the DECODE kernels, while the
    /// prompt runs keep the per-run prefill dispatch. Decode logits ride the
    /// batched head (mma_ks) over rows [0, nd) of the first chunk.
    pub(crate) fn forward_unified_sampled_impl(
        &mut self,
        decodes: &[(usize, u32, u32)],
        budget: usize,
        plans: &[crate::generator::RowSample],
    ) -> Result<(crate::generator::SampledStep, Vec<(usize, Vec<f32>, usize)>), GpuError> {
        use crate::generator::SampledStep;
        assert_eq!(plans.len(), decodes.len(), "one plan per decode row");
        // same whole-prompt tick budget as the mixed path
        let cap = budget.clamp(1, mixed_tick_rows());
        let mut batch: Vec<(usize, Vec<u32>)> = Vec::new();
        let mut used = 0usize;
        // PEEK, don't drain: the queue commits only once the chunk pass
        // succeeds (`self.chunked.drain(..batch.len())` at the success
        // point). The old destructive take lost these entries when the pass
        // hit PoolExhausted - the scheduler's `chunking` set then pointed at
        // slots this queue no longer held, and the serve spun forever on
        // no-op mixed ticks (found live as a wide-batch wedge).
        for c in self.chunked.iter() {
            let next = c.tokens.len();
            if used > 0 && used + next > cap {
                break;
            }
            used += next;
            batch.push((c.slot, c.tokens.clone()));
        }
        // degenerate ticks keep their specialized fast paths (graph replay
        // for pure decode; the plain coalesced pass for pure prefill)
        if batch.is_empty() {
            let tokens: Vec<u32> = decodes.iter().map(|d| d.1).collect();
            let positions: Vec<u32> = decodes.iter().map(|d| d.2).collect();
            let slots_map: Vec<u32> = decodes.iter().map(|d| d.0 as u32).collect();
            let step = self.forward_batch_sampled_rows(&tokens, &positions, &slots_map, plans)?;
            return Ok((step, Vec::new()));
        }
        if decodes.is_empty() {
            let outs = self.forward_prefill_batch_impl(&batch)?;
            self.chunked.drain(..batch.len());
            let finished = batch
                .iter()
                .zip(outs)
                .map(|((slot, toks), logits)| (*slot, logits, toks.len()))
                .collect();
            return Ok((
                SampledStep {
                    ids: Vec::new(),
                    host_rows: Vec::new(),
                },
                finished,
            ));
        }

        let nd = decodes.len();
        let n_embd = self.hp.n_embd;
        // prompt admission: fresh-sequence clear + prefix adopt + pool grow
        let t_adm = std::time::Instant::now();
        let mut starts = Vec::with_capacity(batch.len());
        for (slot, toks) in &batch {
            assert!(
                *slot < self.n_slots,
                "slot {slot} >= enabled {}",
                self.n_slots
            );
            self.gpool_clear_slot(*slot);
            starts.push(self.prefix_resume(*slot, toks)?);
            self.ensure_global_rows(&[*slot as u32], &[(toks.len() - 1) as u32])?;
        }
        if paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some() {
            let ms = t_adm.elapsed().as_secs_f64() * 1e3;
            if ms > 1.0 {
                tracing::info!("[adm] {} prompts {:.2}ms", batch.len(), ms);
            }
        }
        {
            let dslots: Vec<u32> = decodes.iter().map(|d| d.0 as u32).collect();
            let dpos: Vec<u32> = decodes.iter().map(|d| d.2).collect();
            self.ensure_global_rows(&dslots, &dpos)?;
        }

        // row stream: decode rows first, then each prompt's divergent tail
        // (usize::MAX marks decode rows in the item column)
        let mut rows: Vec<(u32, u32, u32, usize)> = decodes
            .iter()
            .map(|&(slot, tok, pos)| (slot as u32, pos, tok, usize::MAX))
            .collect();
        let mut last_row = vec![0usize; batch.len()];
        for (it, ((slot, toks), &start)) in batch.iter().zip(&starts).enumerate() {
            for (j, &t) in toks[start..].iter().enumerate() {
                rows.push((*slot as u32, (start + j) as u32, t, it));
            }
            last_row[it] = rows.len() - 1;
        }

        let _row_bytes = (n_embd / 32) * 34;
        let mut out: Vec<Vec<f32>> = vec![Vec::new(); batch.len()];
        let mut fin_staged = vec![false; batch.len()];
        let mut step: Option<SampledStep> = None;
        let mut base = 0usize;
        for chunk in rows.chunks(self.pf_rows) {
            let r = chunk.len();
            let nd_here = if base == 0 { nd } else { 0 };
            let positions: Vec<u32> = chunk.iter().map(|x| x.1).collect();
            let slots_v: Vec<u32> = chunk.iter().map(|x| x.0).collect();
            // contiguous same-slot PROMPT runs (skip the decode rows)
            let mut runs: Vec<(usize, usize)> = Vec::new();
            for (i, x) in chunk.iter().enumerate().skip(nd_here) {
                match runs.last_mut() {
                    Some((off, n)) if chunk[*off].0 == x.0 && chunk[*off].3 == x.3 => *n += 1,
                    _ => runs.push((i, 1)),
                }
            }
            let spans = super::forward::swa_spans(self.swa_span, &runs);
            {
                let sc = &mut self.scratch;
                let e = |e: cudarc::driver::DriverError| crate::gpu::from_driver(e);
                self.exec
                    .stream
                    .memcpy_htod(&positions, &mut sc.pf_pos)
                    .map_err(e)?;
                self.exec
                    .stream
                    .memcpy_htod(&positions, &mut sc.pf_attn_pos)
                    .map_err(e)?;
                self.exec
                    .stream
                    .memcpy_htod(&slots_v, &mut sc.pf_slots)
                    .map_err(e)?;
            }
            {
                // ONE-kernel embed gather (the per-row dequant+copy loop was
                // 2 host launches/row - the c32 mixed-tick launch wall)
                let toks: Vec<u32> = chunk.iter().map(|x| x.2).collect();
                let sc = &mut self.scratch;
                let mut v = sc
                    .pf_toks
                    .try_slice_mut(0..r)
                    .ok_or_else(|| GpuError::Driver("pf_toks slice".into()))?;
                self.exec
                    .stream
                    .memcpy_htod(&toks, &mut v)
                    .map_err(|e| GpuError::Driver(e.to_string()))?;
                self.exec.embed_gather_plane(
                    &self.token_embd,
                    &sc.pf_toks,
                    &mut sc.pf_x,
                    n_embd,
                    r,
                    self.hp.embd_scale(),
                )?;
                super::GpuGemma4::embd_preamble(
                    &self.exec,
                    &self.hp,
                    self.embd_ones.as_ref(),
                    &mut sc.pf_x,
                    r,
                )?;
            }
            self.prefill_layers(r, &runs, &spans, nd_here)?;
            if nd_here > 0 {
                step = Some(self.unified_decode_head(nd, plans)?);
            }
            for (it, &lr) in last_row.iter().enumerate() {
                if lr >= base && lr < base + r {
                    if it < 64 {
                        self.logits_head_stage(lr - base, it)?;
                        fin_staged[it] = true;
                    } else {
                        out[it] = self.logits_from_pf_row(lr - base)?;
                    }
                }
            }
            base += r;
        }
        // MTP h bootstrap: on SINGLE-chunk streams the decode rows'
        // post-output-norm hiddens are still in pf_normed[0..nd)
        // (unified_decode_head wrote them; no later chunk overwrote). Record
        // the h map so the next tick's draft chain can engage - without
        // this, unified ticks always wiped the map and spec-in-mixed could
        // never bootstrap. Multi-chunk streams leave it wiped (self-heals
        // on the next single-chunk or pure tick).
        if self.mtp.is_some() && rows.len() <= self.pf_rows {
            for (i, x) in rows[..nd].iter().enumerate() {
                if let Some(e) = self.spec_rows.get_mut(x.0 as usize) {
                    *e = Some((x.1, i as u32));
                }
            }
        }
        // cache the finished prompts (direct ring->pool checkpoint + zero-copy
        // radix insert)
        let t_ins = std::time::Instant::now();
        for (it, (slot, toks)) in batch.iter().enumerate() {
            let cut = self.prefix_cut(toks.len(), starts[it]);
            self.prefix_insert(*slot, toks, cut)?;
        }
        if paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some() {
            let ms = t_ins.elapsed().as_secs_f64() * 1e3;
            if ms > 1.0 {
                tracing::info!("[pfx-ins] {} prompts {:.2}ms", batch.len(), ms);
            }
        }
        // restore the single-stream slot-0 staging convention
        let zeros = vec![0u32; self.pf_rows];
        self.exec
            .stream
            .memcpy_htod(&zeros, &mut self.scratch.pf_slots)
            .map_err(|e| GpuError::Driver(e.to_string()))?;
        // One batched dtoh for every staged finisher (the per-it
        // reads serialized 8 sync+1MB copies into ~18ms of [mix-wait])
        for (it, logits) in self
            .logits_finish_read_all(&fin_staged)?
            .into_iter()
            .enumerate()
        {
            if let Some(l) = logits {
                out[it] = l;
            }
        }
        self.chunked.drain(..batch.len());
        let finished = batch
            .into_iter()
            .zip(out)
            .map(|((slot, toks), logits)| (slot, logits, toks.len()))
            .collect();
        Ok((step.expect("chunk 0 carries the decode rows"), finished))
    }

    /// Head + device sampling for the unified tick's decode rows (rows
    /// [0, nd) of the just-walked chunk): final norm, mma_ks head into
    /// batch_logits, softcap, sample_rows - the forward_batch_sampled tail
    /// without the graph (the unified pass is host-launched anyway).
    pub(super) fn unified_decode_head(
        &mut self,
        nd: usize,
        plans: &[crate::generator::RowSample],
    ) -> Result<crate::generator::SampledStep, GpuError> {
        self.unified_decode_head_launch(nd, plans)?;
        self.unified_decode_head_read(nd, plans)
    }

    // Mixed-round issue-ahead split: _launch enqueues the head + device
    // sampling and RETURNS without the ids readback; _read does the blocking
    // dtoh. The service processes the previous round's deferred host work
    // (finish_prefill: ~1.4ms/finisher of pure-host pick_next + drafter
    // seeding) between the two - inside the verify's GPU window - instead of
    // stalling the stream with it, which shows up as 6-8% real GPU idle at
    // the mixed boundaries.
    pub(super) fn unified_decode_head_launch(
        &mut self,
        nd: usize,
        plans: &[crate::generator::RowSample],
    ) -> Result<(), GpuError> {
        use crate::generator::RowSample;
        use crate::sampler::DevicePlan;
        let vocab = self.hp.n_vocab;
        let cap = self.hp.final_softcap;
        let lscale = self.hp.logit_scale;
        let exec = self.exec.clone();
        {
            let sc = &mut self.scratch;
            exec.rmsnorm_batch(
                &sc.pf_x,
                &self.output_norm,
                &mut sc.pf_normed,
                self.hp.n_embd,
                self.hp.eps,
                nd,
            )?;
        }
        {
            let sc = &mut self.scratch;
            let logits_dev = self.batch_logits.as_mut().expect("enable_batch allocates");
            // r-ladder like the decode walk's head - and like it, the f8t
            // tile plane takes everything to 192 rows - this site is the
            // wide spec verify's head, and mma_ks col-tiles the 1.5 GB Q8
            // plane ~5x per 128-160-row round; mmq tiles above.
            if let Some(ht) = self.head_f8t.as_ref().filter(|_| nd <= 192) {
                exec.quantize_e4m3_row(
                    &sc.pf_normed,
                    &mut sc.pf_e4q,
                    &mut sc.pf_e4rs,
                    self.hp.n_embd,
                    nd,
                )?;
                exec.f8t_gemm(
                    ht,
                    &sc.pf_e4q,
                    &sc.pf_e4rs,
                    &mut sc.pf_skfix,
                    logits_dev,
                    self.hp.n_embd,
                    vocab,
                    nd,
                )?;
            } else {
                match self.head.q8() {
                    Some(hq) if nd <= 192 => {
                        exec.quantize_q8(
                            &sc.pf_normed,
                            &mut sc.pf_xq,
                            &mut sc.pf_xs,
                            nd * self.hp.n_embd,
                        )?;
                        exec.q8_0_gemm_mma_ks(
                            hq,
                            &sc.pf_xq,
                            &sc.pf_xs,
                            &mut sc.pf_skfix,
                            logits_dev,
                            nd,
                        )?;
                    }
                    Some(hq) => {
                        exec.quantize_q8_mmq(&sc.pf_normed, &mut sc.pf_yq, self.hp.n_embd, nd)?;
                        pf_mmq(&exec, hq, &sc.pf_yq, &mut sc.pf_skfix, logits_dev, nd)?;
                    }
                    // bf16 head: no int8 rung applies (see the decode head)
                    None => self.head.gemm(&exec, &sc.pf_normed, logits_dev, nd)?,
                }
            }
            super::logit_epilogue_dev(&exec, logits_dev, nd * vocab, lscale, cap)?;
        }
        let mut par = vec![0u32; nd * 4];
        let mut tpar = vec![0u32; nd * 4];
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
                // device truncation mode 5: full-device truncation draw on the mixed
                // tick's verify head (same softcapped plane)
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
                // RS rows stay mode 0 (untouched): the resolve kernel
                // writes their ids after the tick
                RowSample::Device(DevicePlan::RsVerify { .. })
                | RowSample::Device(DevicePlan::RsTrunc { .. }) => {}
            }
        }
        {
            let (d_par, _) = self.samp.as_mut().expect("sampler buffers");
            let mut v = d_par
                .try_slice_mut(0..nd * 4)
                .ok_or_else(|| GpuError::Driver("samp par slice".into()))?;
            exec.stream
                .memcpy_htod(&par, &mut v)
                .map_err(|e| GpuError::Driver(e.to_string()))?;
        }
        if any_trunc {
            let d_tpar = self.samp_tpar.as_mut().expect("allocated with samp");
            let mut v = d_tpar
                .try_slice_mut(0..nd * 4)
                .ok_or_else(|| GpuError::Driver("samp tpar slice".into()))?;
            exec.stream
                .memcpy_htod(&tpar, &mut v)
                .map_err(|e| GpuError::Driver(e.to_string()))?;
        }
        {
            let logits_dev = self.batch_logits.as_mut().expect("enable_batch");
            let (d_par, d_out) = self.samp.as_mut().expect("sampler buffers");
            exec.sample_rows(logits_dev, d_par, d_out, nd, vocab)?;
            if any_trunc {
                let d_tpar = self.samp_tpar.as_ref().expect("allocated with samp");
                exec.sample_rows_t(logits_dev, d_par, d_tpar, d_out, nd, vocab)?;
                exec.sample_rows_p(logits_dev, d_par, d_tpar, d_out, nd, vocab)?;
            }
        }
        Ok(())
    }

    pub(super) fn unified_decode_head_read(
        &mut self,
        nd: usize,
        plans: &[crate::generator::RowSample],
    ) -> Result<crate::generator::SampledStep, GpuError> {
        use crate::generator::{RowSample, SampledStep};
        let vocab = self.hp.n_vocab;
        let exec = self.exec.clone();
        let (_, d_out) = self.samp.as_ref().expect("sampler buffers");
        let ids_view = d_out
            .try_slice(0..nd)
            .ok_or_else(|| GpuError::Driver("samp out slice".into()))?;
        let ids = exec
            .stream
            .clone_dtoh(&ids_view)
            .map_err(|e| GpuError::Driver(e.to_string()))?;
        let mut host_rows = Vec::new();
        if plans.iter().any(|p| matches!(p, RowSample::Host)) {
            let logits_dev = self.batch_logits.as_ref().expect("enable_batch");
            for (i, p) in plans.iter().enumerate() {
                if matches!(p, RowSample::Host) {
                    let v = logits_dev
                        .try_slice(i * vocab..(i + 1) * vocab)
                        .ok_or_else(|| GpuError::Driver("host row slice".into()))?;
                    let row = exec
                        .stream
                        .clone_dtoh(&v)
                        .map_err(|e| GpuError::Driver(e.to_string()))?;
                    host_rows.push((i, row));
                }
            }
        }
        Ok(SampledStep { ids, host_rows })
    }
}

/// 26B-A4B hybrid-FFN tail, all lanes (ref lines 291-344 of the pinned
/// b9895 gemma4 graph; design). On entry
/// the shared branch's down output sits in proj (pf_proj / proj) and x
/// (pf_x / x) still holds attn_out - this replaces the dense
/// `rmsnorm_add_scale(x, proj, ffn_post_norm, out_scale)` tail with:
///
///   s = post_ffw_norm_1(proj)
///   m = post_ffw_norm_2( moe( pre_ffw_norm_2(x) ) )
///   x = (x + ffn_post_norm(s + m)) * out_scale
///
/// MoE = router(one folded weighted rmsnorm -> f32 GEMV -> top-8 softmax,
/// which EQUALS the ref's softmax-all-128 -> top-8 -> renorm: the 6.1e-5
/// sum clamp is unreachable since the top-8 of a 128-way softmax hold
/// >= 1/16 of the mass) -> routed geglu experts -> down + weighted combine
/// > with the per-expert down scale pre-folded into the weights.
/// > Token-batched dp4a class for bring-up; the sorted/mma port is the perf
/// > follow-up. `pf` picks the lane's residual/proj pair.
#[allow(clippy::too_many_arguments)]
/// > Default-ON gate for the shared-vs-routed branch overlap fork
/// > (kill: PADDOCK_NO_MOE_FORK). A clear wide-batch throughput and ITL win,
/// > and greedy-identical - it only changes scheduling.
fn g4_moe_fork() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_MOE_FORK").is_none())
}

pub(super) fn g4_moe_tail(
    exec: &GpuExecutor,
    sc: &mut super::Scratch,
    moe: &super::MoeWeights,
    ffn_post_norm: &CudaSlice<f32>,
    hp: &super::Hparams,
    out_scale: f32,
    r: usize,
    pf: bool,
) -> Result<(), GpuError> {
    let (k, ff) = (hp.n_expert_used, hp.ff_exp);
    // Fused head/router (depth cuts): one kernel norms x for both the
    // router (gamma) and the expert input (pre_ffw_norm_2, f32 + q8 forms),
    // and the topk folds the per-expert down scale - 13 moe nodes/layer -> 7.
    // PADDOCK_NO_MOE_FUSE pins the legacy chain for A/B.
    let fused =
        exec.has_moe_fusions() && paddock_models::dev_var_os!("PADDOCK_NO_MOE_FUSE").is_none();
    // K-split router (slot 486, g26a4b act): the tile matvec ran the decode
    // router shape at 0.34 waves / 15.8us; the K-split fills the die (~3us).
    // moe_part is dead between the previous tick's combine and this tick's
    // down half, so it serves as the partials scratch - no new allocation.
    // r < 16 keeps the PDL-cascade matvec route. OPT-IN
    // (PADDOCK_ROUTER_KS=1) because a serve A/B measured it NET NEGATIVE:
    // it costs TTFT (a prefill r=2048 re-reads the plane per 4-token tile x
    // 512) and buys no ITL at decode, so the 15.8us tile matvec is not the
    // critical-path cost a kernel census makes it look like. Kept as an arm
    // for a shaped retune (prefill keeps
    // the tile route; decode needs kernel-level proof first).
    let rks = r >= 16
        && exec.has_matvec_f32_ks()
        && paddock_models::dev_var_os!("PADDOCK_ROUTER_KS").is_some();
    // tail+combine fold (slot 491): arms that produce per-(token,slot)
    // partials skip the standalone combine - the tail sums part directly
    // (bitwise the combine_init -> moe_tail chain). Only valid when the
    // fused tail actually runs. PADDOCK_NO_TAIL_FOLD = A/B.
    let tail_fold = fused
        && exec.has_moe_tail_combine()
        && paddock_models::dev_var_os!("PADDOCK_NO_TAIL_FOLD").is_none();
    // hibatch lane elections (function scope: consumed at the router AND gu
    // sites). PADDOCK_HIBATCH_MIN_ROWS default 48; both lanes opt-in.
    let hb_min: usize = paddock_models::dev_var!("PADDOCK_HIBATCH_MIN_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(48);
    // P1-2 per-128 activation-scale lane (PADDOCK_HIBATCH_XG=1): engages the
    // head_xg producer + mma2g consumer together (coherent per tick).
    let xg_lane = r >= hb_min
        && hp.n_embd.is_multiple_of(128)
        && exec.has_moe_head_xg()
        && exec.has_q8_moe_mma2g()
        && paddock_models::dev_var!("PADDOCK_HIBATCH_XG")
            .ok()
            .is_some_and(|v| v != "0");
    // P1-1 bf16-partials lane (PADDOCK_HIBATCH_PARTBF16, r>=thresh): runtime
    // coherence via part_is_bf16 - set only when the bf16 down twin ran, so
    // the tail can never misread the plane whatever arm produced it.
    let pbf16_env = r >= hb_min
        && tail_fold  // combine_init fallback reads f32 part - never mix
        && exec.has_q8_moe_down_pbf16()
        && exec.has_moe_tail_combine_bf16()
        && paddock_models::dev_var!("PADDOCK_HIBATCH_PARTBF16").ok().is_some_and(|v| v != "0");
    // dn64: per-64 Y-scale pair - the mma2g y64 producer
    // and fs64 down consumer flip together (fs encoding coherence within
    // the tick; fs_is_64 is set only when the y64 gu actually ran, so the
    // down site can never misread the fs stride). Requires the xg lane
    // (the producer is the per-128-X mma2g). The finer fq grid adds about a
    // percent, and passes the quality gates.
    let dn64_env = r >= hb_min
        && exec.has_q8_moe_mma2g_y64()
        && exec.has_q8_moe_down_fs64()
        && paddock_models::dev_var!("PADDOCK_HIBATCH_DN64")
            .ok()
            .is_some_and(|v| v != "0");
    let mut fs_is_64 = false;
    let mut part_is_bf16 = false;
    let mut part_combined = false;
    if fused {
        // slot 487 (g26a4b act): head + router GEMV + scaled top-k in one
        // launch - bit-identical to the three-node chain below (the
        // in-kernel logit walk is the tile matvec's exact summation order,
        // top-k/scale fold verbatim); rn never touches gmem.
        // PADDOCK_NO_HEADR restores the chain for A/B.
        // r >= 16 only: below that the chain's matvec is the b<16 kernel
        // (256-thread-strided sums) - a different summation order than the
        // fused warp walk, so r=1 identity is impossible by construction
        // (the 08-25 greedy gate caught exactly that; PPL saw +0.049 nats
        // on the r=1 reorder). At r >= 16 the chain runs the tile matvec,
        // whose per-token walk the fused kernel reproduces exactly.
        // OPT-IN (PADDOCK_HEADR=1): bitwise vs the chain at r>=16 (offline
        // probe, a4b_headr_probe) but FALSIFIED as a default on perf - the
        // per-token-block router phase re-reads the 1.4MB plane per token
        // (spec verify r=128: ~180MB/tick, serve spec arm -12%), and the
        // c32 arm showed no win outside boot-lottery noise. A shared-plane
        // router phase is the shape that could revive it.
        let headr = r >= 16
            && exec.has_moe_head_router()
            && paddock_models::dev_var_os!("PADDOCK_HEADR").is_some();
        let hb = r >= hb_min
            && exec.has_moe_head_router_hb()
            && paddock_models::dev_var_os!("PADDOCK_HIBATCH_LANE").is_some();
        if hb {
            let x = if pf { &sc.pf_x } else { &sc.x };
            exec.moe_head_router_hb(
                x,
                &moe.router_gamma,
                &moe.pre_norm2,
                &moe.router_w,
                &moe.down_scale,
                &mut sc.moe_out,
                &mut sc.moe_xq,
                &mut sc.moe_xs,
                &mut sc.moe_idx,
                &mut sc.moe_w,
                hp.n_embd,
                hp.n_expert,
                k,
                hp.eps,
                r,
            )?;
        } else if headr {
            let x = if pf { &sc.pf_x } else { &sc.x };
            exec.moe_head_router(
                x,
                &moe.router_gamma,
                &moe.pre_norm2,
                &moe.router_w,
                &moe.down_scale,
                &mut sc.moe_out,
                &mut sc.moe_xq,
                &mut sc.moe_xs,
                &mut sc.moe_idx,
                &mut sc.moe_w,
                hp.n_embd,
                hp.n_expert,
                k,
                hp.eps,
                r,
            )?;
        } else {
            {
                let x = if pf { &sc.pf_x } else { &sc.x };
                // P1-2 lane (path 1): per-128 activation-scale head twin. The gu
                // consumer flips with it (xg_lane checked at the gu site below) -
                // producer/consumer coherence within the tick.
                if xg_lane && exec.has_moe_head_xg() {
                    exec.moe_head_xg(
                        x,
                        &moe.router_gamma,
                        &moe.pre_norm2,
                        &mut sc.moe_xn,
                        &mut sc.moe_out,
                        &mut sc.moe_xq,
                        &mut sc.moe_xs,
                        hp.n_embd,
                        hp.eps,
                        r,
                    )?;
                } else {
                    exec.moe_head(
                        x,
                        &moe.router_gamma,
                        &moe.pre_norm2,
                        &mut sc.moe_xn,
                        &mut sc.moe_out,
                        &mut sc.moe_xq,
                        &mut sc.moe_xs,
                        hp.n_embd,
                        hp.eps,
                        r,
                    )?;
                }
            }
            // B3-1 cooperative router stage (PADDOCK_HIBATCH_RSTAGE=1, r>=thresh):
            // matvec + topk in one die-filling kernel; per-logit math verbatim
            // (bit-identical logits at the tile-matvec shape, r>=16 class).
            let rstage = r >= hb_min
                && exec.has_moe_router_stage()
                && paddock_models::dev_var!("PADDOCK_HIBATCH_RSTAGE")
                    .ok()
                    .is_some_and(|v| v != "0");
            if rstage {
                exec.moe_router_stage(
                    &moe.router_w,
                    &sc.moe_xn,
                    &mut sc.moe_logits,
                    &moe.down_scale,
                    &mut sc.moe_idx,
                    &mut sc.moe_w,
                    hp.n_embd,
                    hp.n_expert,
                    r,
                    k,
                )?;
            } else {
                if rks {
                    exec.matvec_f32_ks(
                        &moe.router_w,
                        &sc.moe_xn,
                        &mut sc.moe_part,
                        &mut sc.moe_logits,
                        r,
                    )?;
                } else {
                    exec.matvec_f32_batch(&moe.router_w, &sc.moe_xn, &mut sc.moe_logits, r)?;
                }
                exec.moe_topk_scaled(
                    &sc.moe_logits,
                    &moe.down_scale,
                    hp.n_expert,
                    k,
                    &mut sc.moe_idx,
                    &mut sc.moe_w,
                    r,
                )?;
            }
        }
    } else {
        // legacy chain (gamma = gate_inp_s/sqrt(d), folded at load)
        {
            let x = if pf { &sc.pf_x } else { &sc.x };
            exec.rmsnorm_batch(x, &moe.router_gamma, &mut sc.moe_xn, hp.n_embd, hp.eps, r)?;
        }
        exec.matvec_f32_batch(&moe.router_w, &sc.moe_xn, &mut sc.moe_logits, r)?;
        exec.moe_topk_batch(
            &sc.moe_logits,
            &sc.moe_zbias,
            hp.n_expert,
            k,
            &mut sc.moe_idx,
            &mut sc.moe_w,
            r,
        )?;
        exec.moe_scale_w(&mut sc.moe_w, &sc.moe_idx, &moe.down_scale, r * k)?;
        {
            let x = if pf { &sc.pf_x } else { &sc.x };
            exec.rmsnorm_batch(x, &moe.pre_norm2, &mut sc.moe_xn, hp.n_embd, hp.eps, r)?;
        }
        exec.quantize_q8(&sc.moe_xn, &mut sc.moe_xq, &mut sc.moe_xs, r * hp.n_embd)?;
    }
    //  diagnostic (PADDOCK_MOE_UNIQ=path): accumulate the real
    // uniq-experts-per-(tick,layer) histogram - the number that prices a
    // decode-band expert kernel's true weight bytes (uniform kbench cells
    // assume uniq = min(r*k, n_expert); real routing overlaps). Sits
    // before the arm branches so every route is measured. Launch-only
    // (~2us) so captured decode graphs bake it in and it keeps counting on
    // replays; the dump runs on a detached thread (g4_moe_uniq_arm).
    // P1-0 activation census dump (hibatch path 1):
    // env-gated raw capture. pn = this layer's pre-quantize MoE activations;
    // fq/fs = the previous layer-tick's GEGLU outputs (still resident in the
    // scratch - valid census data, avoids a second hook site). Capped at 12.
    if let Ok(dir) = std::env::var("PADDOCK_ACT_DUMP") {
        static PN_N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        // only steady serving rows (boot warmup runs r=4 and would eat the cap)
        let i = if r >= 32 {
            PN_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        } else {
            usize::MAX
        };
        if i < 12 {
            let host: Vec<f32> = exec.stream.clone_dtoh(&sc.moe_out).expect("act dump pn");
            let m = (r * hp.n_embd).min(host.len());
            let bytes: Vec<u8> = host[..m].iter().flat_map(|v| v.to_le_bytes()).collect();
            let _ = std::fs::write(format!("{dir}/pn_{i:03}.f32"), bytes);
            if i > 0 {
                let hq: Vec<i8> = exec.stream.clone_dtoh(&sc.moe_fq).expect("act dump fq");
                let hs: Vec<f32> = exec.stream.clone_dtoh(&sc.moe_fs).expect("act dump fs");
                let bq: Vec<u8> = hq.iter().map(|v| *v as u8).collect();
                let bs: Vec<u8> = hs.iter().flat_map(|v| v.to_le_bytes()).collect();
                let _ = std::fs::write(format!("{dir}/fq_{i:03}.i8"), bq);
                let _ = std::fs::write(format!("{dir}/fs_{i:03}.f32"), bs);
            }
        }
    }
    if sc.moe_uniq_dev != 0 {
        exec.moe_uniq_hist(&sc.moe_idx, r * k, hp.n_expert, sc.moe_uniq_dev)?;
    }
    // Sorted (moe_align) vs token-batched: sorted reads each touched
    // expert's weights once per pass; token-batched re-reads routed rows per
    // token (the qwen bring-up measured 0.18x llama at prefill on exactly
    // that). Boundary default = qwen's measured mma crossover (128 pairs);
    // PADDOCK_QMOE_SORTED_MIN retunes, PADDOCK_NO_SORTED_QMOE pins
    // token-batched for A/B. BM=64 (wider prefill block) engages on the
    // same fill heuristic as qwen (only pays when blocks populate).
    let sorted_min: usize = std::env::var("PADDOCK_QMOE_SORTED_MIN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128);
    let sorted = r * k >= sorted_min
        && exec.has_q8_moe_geglu_sorted()
        && paddock_models::dev_var_os!("PADDOCK_NO_SORTED_QMOE").is_none();
    // tcgen05 e4m3 expert lane (a4b-expert-tcgen05.md): the sorted layout at
    // BM=128 (a block is a tc5 Y tile) over the fused/K-padded f8 planes.
    // Engages only at prefill-scale pair counts - the 128-row blocks are
    // mostly PAD below that (the sorted-min=1 falsification, amplified).
    // PADDOCK_MOE_F8_MIN retunes; PADDOCK_NO_MOE_F8_RUN pins the s8 route.
    // The boundary was swept: 64 regresses badly (PAD overhead on
    // pure-decode 64-pair ticks vs dec2) and 512 wins every config, so
    // mixed ticks from ~512 pairs ride tc5 and pure decode keeps the dec2
    // pair.
    let f8_min: usize = paddock_models::dev_var!("PADDOCK_MOE_F8_MIN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(512);
    // v2 ring-pair band routing: at
    // verify-scale pair counts the exact-class sorted v2 pair beats the
    // e4m3 f8s band outright (kbench uni:128 = 1024 pairs: v2 gu+dn 228.9us
    // vs f8s 418.5 + its quantize/gather interstitials), so ticks up to
    // PADDOCK_QMMA2_MAX pairs (default 2048; uni:256 priced the boundary)
    // stay on the sorted route. PADDOCK_NO_MOE_QMMA2 restores the f8s band
    // (one-env A/B, same switch as the pair itself).
    let qmma2_max: usize = paddock_models::dev_var!("PADDOCK_QMMA2_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        // 4096 (was 2048): kbench uni:512 = 4096 pairs has v2 at 315.2us vs
        // f8s 448.8 + interstitials; uni:1024 (8192) is a wash - and 8192 is
        // exactly the bm=64 election edge, so the bound stays below it.
        .unwrap_or(4096);
    let qmma2_route = exec.has_q8_moe_qmma2()
        && hp.n_embd.is_multiple_of(256)
        && ff % 64 == 0
        && paddock_models::dev_var_os!("PADDOCK_NO_MOE_QMMA2").is_none();
    let f8s = r * k >= f8_min
        && !(qmma2_route && r * k <= qmma2_max)
        && moe.gu_f8.is_some()
        && exec.has_f8bs_moe()
        && paddock_models::dev_var_os!("PADDOCK_NO_MOE_F8_RUN").is_none();
    // decode-band f8 shapes - OPT-IN (PADDOCK_MOE_F8D=1) until the serve
    // gates arbitrate. Profiling the f8_min=64 arm showed the M=128 gu
    // already beat dec2 at decode (55 vs 71us) and the chain lost on the
    // dn geometry (81us) + worst-case-srp interstitials (41us); this band
    // reruns the tc5 pipe at BM=32 with the Y-resident dn and PAD-aware
    // geglu. e4m3-expert numeric class (same as the >=512 default band).
    // PADDOCK_MOE_F8D_MIN sets the lower edge (default 8 = everything).
    let f8d_min: usize = paddock_models::dev_var!("PADDOCK_MOE_F8D_MIN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let f8d = !f8s
        && r * k >= f8d_min
        && moe.gu_f8.is_some()
        && exec.has_f8d_moe()
        && paddock_models::dev_var_os!("PADDOCK_MOE_F8D").is_some()
        && paddock_models::dev_var_os!("PADDOCK_NO_MOE_F8_RUN").is_none();
    if f8s {
        let bm = 128usize;
        let mb = (r * k + hp.n_expert * (bm - 1)).div_ceil(bm);
        let srp = mb * bm;
        let ffp = ff.next_multiple_of(128);
        {
            // pre2-normed f32 rows: the fused head lands them in moe_out
            let pn = if fused { &sc.moe_out } else { &sc.moe_xn };
            exec.quantize_e4m3(pn, &mut sc.moe_e4q, &mut sc.moe_e4s, r * hp.n_embd)?;
        }
        exec.moe_align_bm(
            &sc.moe_idx,
            &mut sc.moe_srow,
            &mut sc.moe_sslot,
            &mut sc.moe_bexp,
            r,
            k,
            hp.n_expert,
            bm,
            mb,
        )?;
        exec.moe_gather_e4m3(
            &sc.moe_e4q,
            &sc.moe_e4s,
            &sc.moe_srow,
            &mut sc.moe_xg,
            &mut sc.moe_sg,
            hp.n_embd,
            srp,
        )?;
        exec.f8bs_moe_gemm_gu(
            moe.gu_f8.as_ref().expect("checked by f8s"),
            &sc.moe_xg,
            &sc.moe_sg,
            &sc.moe_bexp,
            &mut sc.moe_gu,
            hp.n_embd,
            2 * ff,
            hp.n_expert,
            srp,
            mb,
        )?;
        // prefill dn hybrid (slots 489/490): at every measured width the
        // BM=128 tc5 down loses to the v2 BM=32 down (uni:1024 8192p: 424
        // vs 213us), so the f8s-gu f32 output is GEGLU-quantized to q8
        // STRAIGHT into bm32 rows (pair map) and the v2 down runs the
        // band. fq moves e4m3 -> q8 (finer); PPL_PREFIX=2048 + the serve
        // cells gate it. PADDOCK_NO_PF_HYBRID restores the tc5 down.
        let hybrid = exec.has_pf_dn_hybrid()
            && exec.has_q8_moe_qmma2()
            && ff % 64 == 0
            && paddock_models::dev_var_os!("PADDOCK_NO_PF_HYBRID").is_none();
        if hybrid {
            let mb32 = (r * k + hp.n_expert * 31).div_ceil(32);
            exec.moe_align(
                &sc.moe_idx,
                &mut sc.moe_srow2,
                &mut sc.moe_sslot2,
                &mut sc.moe_bexp2,
                r,
                k,
                hp.n_expert,
                mb32,
            )?;
            exec.moe_pair_map(
                &sc.moe_srow2,
                &sc.moe_sslot2,
                &mut sc.moe_pairmap,
                k,
                mb32 * 32,
            )?;
            exec.quantize_q8_geglu_remap(
                &sc.moe_gu,
                &sc.moe_srow,
                &sc.moe_sslot,
                &sc.moe_pairmap,
                &mut sc.moe_fq,
                &mut sc.moe_fs,
                ff,
                k,
                srp,
                0,
            )?;
            // DBG bisect: =1 exercises every hybrid WRITE (align2/map/remap)
            // but keeps the old e4m3 + f8s down producing the output - an
            // OOB write in the new kernels still corrupts, a v2-down issue
            // does not.
            if paddock_models::dev_var_os!("PADDOCK_PF_HYBRID_DBG").is_some() {
                exec.quantize_e4m3_geglu2_pad(
                    &sc.moe_gu,
                    &mut sc.moe_fq8,
                    &mut sc.moe_fs8,
                    ff,
                    ffp,
                    srp,
                )?;
                exec.f8bs_moe_gemm_dn(
                    moe.dn_f8.as_ref().expect("dn_f8 pairs gu_f8"),
                    &sc.moe_fq8,
                    &sc.moe_fs8,
                    &sc.moe_bexp,
                    &sc.moe_srow,
                    &sc.moe_sslot,
                    &sc.moe_w,
                    &mut sc.moe_part,
                    ffp,
                    hp.n_embd,
                    hp.n_expert,
                    srp,
                    mb,
                    k,
                )?;
            } else {
                if exec.has_q8_moe_qmma2t()
                    && paddock_models::dev_var_os!("PADDOCK_MOE_QMMA2_TMA").is_some()
                {
                    // v3t twin on the hybrid prefill down (bitwise)
                    exec.q8_0_moe_down_mma2t(
                        &moe.down_exps,
                        &sc.moe_srow2,
                        &sc.moe_sslot2,
                        &sc.moe_bexp2,
                        &sc.moe_w,
                        &sc.moe_fq,
                        &sc.moe_fs,
                        &mut sc.moe_part,
                        k,
                        mb32,
                        32,
                    )?;
                } else {
                    exec.q8_0_moe_down_mma2(
                        &moe.down_exps,
                        &sc.moe_srow2,
                        &sc.moe_sslot2,
                        &sc.moe_bexp2,
                        &sc.moe_w,
                        &sc.moe_fq,
                        &sc.moe_fs,
                        &mut sc.moe_part,
                        k,
                        mb32,
                        32,
                    )?;
                }
            }
        } else {
            exec.quantize_e4m3_geglu2_pad(
                &sc.moe_gu,
                &mut sc.moe_fq8,
                &mut sc.moe_fs8,
                ff,
                ffp,
                srp,
            )?;
            exec.f8bs_moe_gemm_dn(
                moe.dn_f8.as_ref().expect("dn_f8 pairs gu_f8"),
                &sc.moe_fq8,
                &sc.moe_fs8,
                &sc.moe_bexp,
                &sc.moe_srow,
                &sc.moe_sslot,
                &sc.moe_w,
                &mut sc.moe_part,
                ffp,
                hp.n_embd,
                hp.n_expert,
                srp,
                mb,
                k,
            )?;
        }
        if tail_fold {
            part_combined = true; // the tail fold sums part directly
        } else if exec.has_moe_combine_init()
            && paddock_models::dev_var_os!("PADDOCK_NO_COMBINE_INIT").is_none()
        {
            // slot 485: write-out fold - bitwise the memset + combine chain
            // (0.0f + x == x), minus the ~8.8us/tick driver memset the gap
            // census caught idling the die after every down launch.
            exec.moe_slot_combine_init(&sc.moe_part, &mut sc.moe_xn, hp.n_embd, k, r)?;
        } else {
            exec.stream
                .memset_zeros(&mut sc.moe_xn)
                .map_err(|e| GpuError::Driver(e.to_string()))?;
            exec.moe_slot_combine(&sc.moe_part, &mut sc.moe_xn, hp.n_embd, k, r)?;
        }
    } else if f8d {
        // decode band at BM=32: the histogram worst case caps ~140 blocks
        // below f8_min=512 and every live block holds >= 1 pair, so
        // min(r*k, worst) is a true bound - it shrinks the srp-scaled
        // interstitials 2-16x at small r. f8s-sized scratch covers all of it.
        let bm = 32usize;
        let mb = (r * k + hp.n_expert * (bm - 1)).div_ceil(bm).min(r * k);
        let srp = mb * bm;
        let ffp = ff.next_multiple_of(128);
        {
            let pn = if fused { &sc.moe_out } else { &sc.moe_xn };
            exec.quantize_e4m3(pn, &mut sc.moe_e4q, &mut sc.moe_e4s, r * hp.n_embd)?;
        }
        exec.moe_align_bm(
            &sc.moe_idx,
            &mut sc.moe_srow,
            &mut sc.moe_sslot,
            &mut sc.moe_bexp,
            r,
            k,
            hp.n_expert,
            bm,
            mb,
        )?;
        exec.moe_gather_e4m3(
            &sc.moe_e4q,
            &sc.moe_e4s,
            &sc.moe_srow,
            &mut sc.moe_xg,
            &mut sc.moe_sg,
            hp.n_embd,
            srp,
        )?;
        exec.f8bs_moe_gemm_gu_d32(
            moe.gu_f8.as_ref().expect("checked by f8d"),
            &sc.moe_xg,
            &sc.moe_sg,
            &sc.moe_bexp,
            &mut sc.moe_gu,
            hp.n_embd,
            2 * ff,
            hp.n_expert,
            srp,
            mb,
        )?;
        exec.quantize_e4m3_geglu2_pad_b(
            &sc.moe_gu,
            &mut sc.moe_fq8,
            &mut sc.moe_fs8,
            &sc.moe_bexp,
            ff,
            ffp,
            bm,
            srp,
        )?;
        exec.f8bs_moe_gemm_dn_d32(
            moe.dn_f8.as_ref().expect("dn_f8 pairs gu_f8"),
            &sc.moe_fq8,
            &sc.moe_fs8,
            &sc.moe_bexp,
            &sc.moe_srow,
            &sc.moe_sslot,
            &sc.moe_w,
            &mut sc.moe_part,
            ffp,
            hp.n_embd,
            hp.n_expert,
            srp,
            mb,
            k,
        )?;
        if tail_fold {
            part_combined = true; // the tail fold sums part directly
        } else if exec.has_moe_combine_init()
            && paddock_models::dev_var_os!("PADDOCK_NO_COMBINE_INIT").is_none()
        {
            // slot 485: write-out fold - bitwise the memset + combine chain
            // (0.0f + x == x), minus the ~8.8us/tick driver memset the gap
            // census caught idling the die after every down launch.
            exec.moe_slot_combine_init(&sc.moe_part, &mut sc.moe_xn, hp.n_embd, k, r)?;
        } else {
            exec.stream
                .memset_zeros(&mut sc.moe_xn)
                .map_err(|e| GpuError::Driver(e.to_string()))?;
            exec.moe_slot_combine(&sc.moe_part, &mut sc.moe_xn, hp.n_embd, k, r)?;
        }
    } else if sorted {
        let bm64_fill: usize = paddock_models::dev_var!("PADDOCK_QMOE_BM64_FILL")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64);
        let bm = if r * k >= hp.n_expert * bm64_fill
            && paddock_models::dev_var_os!("PADDOCK_NO_QMOE_BM64").is_none()
        {
            64usize
        } else {
            32usize
        };
        let max_blocks = (r * k + hp.n_expert * (bm - 1)).div_ceil(bm);
        // g2 lane pre-gate: when the token-major GU will
        // run, one dual-output align emits the bm32 CSR + bm16 CSR + pair
        // map (saves two launches vs align+align16+pair_map).
        let g2_max: usize = paddock_models::dev_var!("PADDOCK_MOE_G2_MAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2048);
        let g2_align = bm == 32
            && !xg_lane
            && r * k <= g2_max
            && hp.n_embd.is_multiple_of(256)
            && ff % 64 == 0
            && exec.has_q8_moe_g2()
            && paddock_models::dev_var_os!("PADDOCK_NO_MOE_QMMA2").is_none()
            && paddock_models::dev_var!("PADDOCK_MOE_G2")
                .ok()
                .is_some_and(|v| v != "0");
        let g2_mb16 = (r * k + hp.n_expert * 15).div_ceil(16);
        if g2_align {
            exec.moe_align_dual(
                &sc.moe_idx,
                &mut sc.moe_srow,
                &mut sc.moe_sslot,
                &mut sc.moe_bexp,
                &mut sc.moe_srow2,
                &mut sc.moe_sslot2,
                &mut sc.moe_bexp2,
                &mut sc.moe_pairmap,
                r,
                k,
                hp.n_expert,
                max_blocks,
                g2_mb16,
            )?;
        } else if bm == 64 {
            exec.moe_align_bm(
                &sc.moe_idx,
                &mut sc.moe_srow,
                &mut sc.moe_sslot,
                &mut sc.moe_bexp,
                r,
                k,
                hp.n_expert,
                bm,
                max_blocks,
            )?;
        } else {
            exec.moe_align(
                &sc.moe_idx,
                &mut sc.moe_srow,
                &mut sc.moe_sslot,
                &mut sc.moe_bexp,
                r,
                k,
                hp.n_expert,
                max_blocks,
            )?;
        }
        // Flat-scale e4m3 gate_up (change A) - same sorted layout,
        // same fq/fs handshake into the q8 down GEMM, but the weight scale is
        // per-OUTPUT-ROW so it never enters the k walk. That drops 6.25% of
        // the weight bytes (no per-32 scale plane to stream) and ~5x the
        // per-mma ALU. Lossy requant: OPT-IN until the greedy gate says
        // otherwise, and it needs the planes built at load.
        let f8row = moe.gate_f8r.is_some() && exec.has_f8row_moe();
        // The down half is a separate arm: with it, gate_up must use the
        // e4m3-out epilogue so the two agree on fq's encoding. Without it,
        // gate_up emits int8 and the Q8_0 down keeps serving.
        let f8row_dn = f8row && moe.down_f8r.is_some() && exec.has_f8row_moe_down();
        // v2 ring twins of the q8 pair:
        // S-stage cp.async ring + live-quarter skip, BITWISE vs the shipped
        // pair on live outputs -- default-on where the shape qualifies
        // (BM=32 blocks, in_dim % 256, ff % 64). PADDOCK_NO_MOE_QMMA2 = A/B.
        let qmma2 = bm == 32
            && exec.has_q8_moe_qmma2()
            && hp.n_embd.is_multiple_of(256)
            && ff % 64 == 0
            && paddock_models::dev_var_os!("PADDOCK_NO_MOE_QMMA2").is_none();
        // v3t: TMA-staged v2 twins - bitwise, opt-in
        // (PADDOCK_MOE_QMMA2_TMA=1), sm_90+ packs only. Excludes the xg/dn64
        // encodings (v3t writes the v2 fq/fs layout).
        let qmma2t = qmma2
            && !xg_lane
            && exec.has_q8_moe_qmma2t()
            && paddock_models::dev_var_os!("PADDOCK_MOE_QMMA2_TMA").is_some();
        // g2 (slot 504): the dual align already ran when g2_align is set.
        let g2 = qmma2 && g2_align;
        if f8row {
            let pn = if fused { &sc.moe_out } else { &sc.moe_xn };
            exec.quantize_e4m3_b32f(pn, &mut sc.moe_x8q, &mut sc.moe_x8s, r * hp.n_embd)?;
            let (g, u) = (
                moe.gate_f8r.as_ref().expect("checked by f8row"),
                moe.up_f8r.as_ref().expect("up_f8r pairs gate_f8r"),
            );
            if f8row_dn {
                exec.f8row_moe_gate_up_mma_geglu_f8(
                    g,
                    u,
                    &sc.moe_srow,
                    &sc.moe_bexp,
                    &sc.moe_x8q,
                    &sc.moe_x8s,
                    &mut sc.moe_fq,
                    &mut sc.moe_fs,
                    hp.n_embd,
                    ff,
                    max_blocks,
                    bm,
                )?;
            } else {
                exec.f8row_moe_gate_up_mma_geglu(
                    g,
                    u,
                    &sc.moe_srow,
                    &sc.moe_bexp,
                    &sc.moe_x8q,
                    &sc.moe_x8s,
                    &mut sc.moe_fq,
                    &mut sc.moe_fs,
                    hp.n_embd,
                    ff,
                    max_blocks,
                    bm,
                )?;
            }
        } else if qmma2 {
            // v5 (slot 488): the rival-geometry gate_up (BM16 tile view,
            // 128 thr) - BITWISE the v2 kernel but FALSIFIED on perf
            // (u64r32 73.9 vs 72.4 wash; u48r8 70.1 vs 58.7 worse): the
            // rival's tile geometry is not the differentiator. OPT-IN for
            // the record (PADDOCK_MOE_QMMA3=1).
            if g2 {
                static G2_ONCE: std::sync::Once = std::sync::Once::new();
                G2_ONCE.call_once(|| tracing::info!("g2 token-major GU lane ENGAGED (dual align)"));
                let mb16 = g2_mb16;
                exec.q8_0_moe_gate_up_g2_geglu(
                    &moe.gate_exps,
                    &moe.up_exps,
                    &sc.moe_srow2,
                    &sc.moe_sslot2,
                    &sc.moe_bexp2,
                    &sc.moe_pairmap,
                    &sc.moe_xq,
                    &sc.moe_xs,
                    &mut sc.moe_fq,
                    &mut sc.moe_fs,
                    k,
                    mb16,
                    16,
                )?;
            } else if qmma2t {
                static Q2T_ONCE: std::sync::Once = std::sync::Once::new();
                Q2T_ONCE.call_once(|| tracing::info!("v3t qmma2t lane ENGAGED (decode)"));
                exec.q8_0_moe_gate_up_mma2t_geglu(
                    &moe.gate_exps,
                    &moe.up_exps,
                    &sc.moe_srow,
                    &sc.moe_bexp,
                    &sc.moe_xq,
                    &sc.moe_xs,
                    &mut sc.moe_fq,
                    &mut sc.moe_fs,
                    max_blocks,
                    bm,
                )?;
            } else if exec.has_q8_moe_mma3()
                && paddock_models::dev_var_os!("PADDOCK_MOE_QMMA3").is_some()
            {
                exec.q8_0_moe_gate_up_mma3_geglu(
                    &moe.gate_exps,
                    &moe.up_exps,
                    &sc.moe_srow,
                    &sc.moe_bexp,
                    &sc.moe_xq,
                    &sc.moe_xs,
                    &mut sc.moe_fq,
                    &mut sc.moe_fs,
                    max_blocks,
                    bm,
                )?;
            } else if xg_lane && dn64_env {
                exec.q8_0_moe_gate_up_mma2g_y64_geglu(
                    &moe.gate_exps,
                    &moe.up_exps,
                    &sc.moe_srow,
                    &sc.moe_bexp,
                    &sc.moe_xq,
                    &sc.moe_xs,
                    &mut sc.moe_fq,
                    &mut sc.moe_fs,
                    max_blocks,
                    bm,
                )?;
                fs_is_64 = true;
            } else if xg_lane {
                exec.q8_0_moe_gate_up_mma2g_geglu(
                    &moe.gate_exps,
                    &moe.up_exps,
                    &sc.moe_srow,
                    &sc.moe_bexp,
                    &sc.moe_xq,
                    &sc.moe_xs,
                    &mut sc.moe_fq,
                    &mut sc.moe_fs,
                    max_blocks,
                    bm,
                )?;
            } else {
                exec.q8_0_moe_gate_up_mma2_geglu(
                    &moe.gate_exps,
                    &moe.up_exps,
                    &sc.moe_srow,
                    &sc.moe_bexp,
                    &sc.moe_xq,
                    &sc.moe_xs,
                    &mut sc.moe_fq,
                    &mut sc.moe_fs,
                    max_blocks,
                    bm,
                )?;
            }
        } else {
            exec.q8_0_moe_gate_up_mma_geglu(
                &moe.gate_exps,
                &moe.up_exps,
                &sc.moe_srow,
                &sc.moe_bexp,
                &sc.moe_xq,
                &sc.moe_xs,
                &mut sc.moe_fq,
                &mut sc.moe_fs,
                max_blocks,
                bm,
            )?;
        }
        if f8row_dn {
            exec.f8row_moe_down_mma(
                moe.down_f8r.as_ref().expect("checked by f8row_dn"),
                &sc.moe_srow,
                &sc.moe_sslot,
                &sc.moe_bexp,
                &sc.moe_w,
                &sc.moe_fq,
                &sc.moe_fs,
                &mut sc.moe_part,
                ff,
                hp.n_embd,
                k,
                max_blocks,
                bm,
            )?;
        } else if qmma2 {
            if fs_is_64 {
                // dn64: fs was written per-64 this tick - the fs64 consumer
                // is the only down that reads that stride (pbf16 composes).
                exec.q8_0_moe_down_mma2_fs64(
                    &moe.down_exps,
                    &sc.moe_srow,
                    &sc.moe_sslot,
                    &sc.moe_bexp,
                    &sc.moe_w,
                    &sc.moe_fq,
                    &sc.moe_fs,
                    &mut sc.moe_part,
                    k,
                    max_blocks,
                    bm,
                    pbf16_env,
                )?;
                part_is_bf16 = pbf16_env;
            } else if pbf16_env {
                exec.q8_0_moe_down_mma2_pbf16(
                    &moe.down_exps,
                    &sc.moe_srow,
                    &sc.moe_sslot,
                    &sc.moe_bexp,
                    &sc.moe_w,
                    &sc.moe_fq,
                    &sc.moe_fs,
                    &mut sc.moe_part,
                    k,
                    max_blocks,
                    bm,
                )?;
                part_is_bf16 = true;
            } else if qmma2t {
                exec.q8_0_moe_down_mma2t(
                    &moe.down_exps,
                    &sc.moe_srow,
                    &sc.moe_sslot,
                    &sc.moe_bexp,
                    &sc.moe_w,
                    &sc.moe_fq,
                    &sc.moe_fs,
                    &mut sc.moe_part,
                    k,
                    max_blocks,
                    bm,
                )?;
            } else {
                exec.q8_0_moe_down_mma2(
                    &moe.down_exps,
                    &sc.moe_srow,
                    &sc.moe_sslot,
                    &sc.moe_bexp,
                    &sc.moe_w,
                    &sc.moe_fq,
                    &sc.moe_fs,
                    &mut sc.moe_part,
                    k,
                    max_blocks,
                    bm,
                )?;
            }
        } else {
            exec.q8_0_moe_down_mma(
                &moe.down_exps,
                &sc.moe_srow,
                &sc.moe_sslot,
                &sc.moe_bexp,
                &sc.moe_w,
                &sc.moe_fq,
                &sc.moe_fs,
                &mut sc.moe_part,
                k,
                max_blocks,
                bm,
            )?;
        }
        if tail_fold {
            part_combined = true; // the tail fold sums part directly
        } else if exec.has_moe_combine_init()
            && paddock_models::dev_var_os!("PADDOCK_NO_COMBINE_INIT").is_none()
        {
            // slot 485: write-out fold - bitwise the memset + combine chain
            // (0.0f + x == x), minus the ~8.8us/tick driver memset the gap
            // census caught idling the die after every down launch.
            exec.moe_slot_combine_init(&sc.moe_part, &mut sc.moe_xn, hp.n_embd, k, r)?;
        } else {
            exec.stream
                .memset_zeros(&mut sc.moe_xn)
                .map_err(|e| GpuError::Driver(e.to_string()))?;
            exec.moe_slot_combine(&sc.moe_part, &mut sc.moe_xn, hp.n_embd, k, r)?;
        }
    } else if exec
        .kernels()
        .map(|kt| kt.q8_0_moe_gu_dec2_geglu.is_some())
        .unwrap_or(false)
        && paddock_models::dev_var_os!("PADDOCK_MOE_DEC2").is_some()
    {
        // decode-band intensity twins - NOW OPT-IN (PADDOCK_MOE_DEC2=1).
        // GREEDY-REFUTED 2026-09-04 on gemma-4-26b-A4B: the gu_dec2 + dn_dec2
        // PAIR produces token-repetition garbage ("CAPITAL.- Ezil, own-own..."
        // deterministic), while EITHER twin paired with its original
        // counterpart is correct (bisected: gu_dec2 + original down = OK;
        // original gate_up + dn_dec2 = OK; both dec2 = garbage). So neither
        // kernel is wrong alone - the defect is an interaction between the
        // two (a value-/state-dependent hazard; survives PADDOCK_NO_MOE_FORK,
        // so not the side-stream fork). The old "greedy-identical" claim held
        // only because aiperf never reads the text. Root cause still open;
        // default routes the correct original gate_up_geglu + down below.
        // The 1.5x/1.25x decode-band win (a4b_moe_kbench: gu 110.5->74.5us,
        // dn 67.6->53.3us at r=8) is forfeited until the pair is fixed;
        // it only ever engaged at r*k < 128 (the sorted path owns >=128).
        //
        // dec3 gate_up - OPT-IN only, FALSIFIED as default. The
        // bulk-streamed kernel (moe_align BM=2 + one TMA-ring CTA per
        // (block, out tile)) is bitwise gu dec2 and won every UNIFORM-routing
        // kbench cell from r=4 - but the serve A/B LOST, and profiling told
        // why: real routing is SKEWED, hot experts make straggler CTAs
        // (gu dec3 median 96us stdev 20 vs dec2 stdev 1.1), and the kbench
        // `hot` case then showed dec2 on skewed routing beats everything
        // (40us at r=8: the hot slabs are L2-RESIDENT, so dec2's per-pair
        // "re-reads" are L2 hits - the dedup dec3 streams for, the cache
        // already gives dec2 for free, and evict_first bypasses it). Kept
        // for the uniform/large-uniq regime study; the down half always
        // stays dec2 (dn dec3 lost at every r).
        let dec3_min: usize = paddock_models::dev_var!("PADDOCK_MOE_DEC3_MIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(32);
        let dec3 = r * k >= dec3_min
            && exec.has_moe_dec3()
            && paddock_models::dev_var_os!("PADDOCK_MOE_DEC3").is_some();
        if dec3 {
            let mb2 = (r * k + hp.n_expert).div_ceil(2);
            exec.moe_align_bm(
                &sc.moe_idx,
                &mut sc.moe_srow,
                &mut sc.moe_sslot,
                &mut sc.moe_bexp,
                r,
                k,
                hp.n_expert,
                2,
                mb2,
            )?;
            exec.q8_0_moe_gu_dec3_geglu(
                &moe.gate_exps,
                &moe.up_exps,
                &sc.moe_bexp,
                &sc.moe_srow,
                &sc.moe_sslot,
                &sc.moe_xq,
                &sc.moe_xs,
                &mut sc.moe_fused,
                k,
                mb2,
                r * k,
            )?;
        } else {
            exec.q8_0_moe_gu_dec2_geglu(
                &moe.gate_exps,
                &moe.up_exps,
                &sc.moe_idx,
                &sc.moe_xq,
                &sc.moe_xs,
                &mut sc.moe_fused,
                k,
                r,
            )?;
        }
        exec.quantize_q8(&sc.moe_fused, &mut sc.moe_fq, &mut sc.moe_fs, r * k * ff)?;
        exec.q8_0_moe_dn_dec2(
            &moe.down_exps,
            &sc.moe_idx,
            &sc.moe_w,
            &sc.moe_fq,
            &sc.moe_fs,
            &mut sc.moe_xn,
            k,
            r,
        )?;
    } else {
        exec.q8_0_moe_gate_up_geglu(
            &moe.gate_exps,
            &moe.up_exps,
            &sc.moe_idx,
            &sc.moe_xq,
            &sc.moe_xs,
            &mut sc.moe_fused,
            k,
            r,
        )?;
        exec.quantize_q8(&sc.moe_fused, &mut sc.moe_fq, &mut sc.moe_fs, r * k * ff)?;
        exec.q8_0_moe_down(
            &moe.down_exps,
            &sc.moe_idx,
            &sc.moe_w,
            &sc.moe_fq,
            &sc.moe_fs,
            &mut sc.moe_xn,
            k,
            r,
        )?;
    }
    // Join the forked shared branch (no-op when not forked): the tail below
    // is the first consumer of both branches (proj + the routed dn output).
    exec.side_join()?;
    if fused {
        // combine trailer in one launch: x = (x + rmsnorm(rmsnorm(proj)*pn1
        // + rmsnorm(dn)*pn2) * postw) * out_scale
        let (x, proj) = if pf {
            (&mut sc.pf_x, &sc.pf_proj)
        } else {
            (&mut sc.x, &sc.proj)
        };
        if part_combined {
            if part_is_bf16 {
                exec.moe_tail_combine_bf16(
                    x,
                    proj,
                    &sc.moe_part,
                    &moe.post_norm1,
                    &moe.post_norm2,
                    ffn_post_norm,
                    hp.n_embd,
                    k,
                    hp.eps,
                    out_scale,
                    r,
                )?;
            } else {
                exec.moe_tail_combine(
                    x,
                    proj,
                    &sc.moe_part,
                    &moe.post_norm1,
                    &moe.post_norm2,
                    ffn_post_norm,
                    hp.n_embd,
                    k,
                    hp.eps,
                    out_scale,
                    r,
                )?;
            }
        } else {
            exec.moe_tail(
                x,
                proj,
                &sc.moe_xn,
                &moe.post_norm1,
                &moe.post_norm2,
                ffn_post_norm,
                hp.n_embd,
                hp.eps,
                out_scale,
                r,
            )?;
        }
        return Ok(());
    }
    exec.rmsnorm_batch(
        &sc.moe_xn,
        &moe.post_norm2,
        &mut sc.moe_out,
        hp.n_embd,
        hp.eps,
        r,
    )?;
    // shared-branch post-norm, branch sum, outer post-norm+residual+scale
    {
        let proj = if pf { &sc.pf_proj } else { &sc.proj };
        exec.rmsnorm_batch(proj, &moe.post_norm1, &mut sc.moe_xn, hp.n_embd, hp.eps, r)?;
    }
    exec.add(&mut sc.moe_xn, &sc.moe_out, r * hp.n_embd)?;
    {
        let x = if pf { &mut sc.pf_x } else { &mut sc.x };
        exec.rmsnorm_add_scale(
            x,
            &sc.moe_xn,
            ffn_post_norm,
            hp.n_embd,
            hp.post_norm_eps,
            out_scale,
            r,
        )?;
    }
    Ok(())
}

/// Mixed/unified tick prefill budget. DECOUPLED from PF_ROWS: 4096-row
/// chunks amortize weights for long single prompts, but a 4096-row TICK
/// doubles first-token granularity (measured: c32 TTFT p50 3063 -> 5718 ms)
/// - and every live decode stream advances only one token per tick, so tick
///   size is also the decode CADENCE under load. ~2048 rows (~500 ms) is the
///   measured c8 sweet spot; PADDOCK_G4_TICK_ROWS overrides for shaping
///   experiments (floor 1088: chunks at or under 1024 rows fall off the
///   mmq-pipe rung).
///   The widest row count one serving tick can present to the layer walk:
///   a mixed tick's prefill budget plus every slot's verify rows. Consumers
///   that must record a whole tick (the DFlash feature taps) size themselves
///   off this - a band that can't hold the tick leaves a hole in the drafter's
///   ring, which is worse than not drafting.
pub(crate) fn tick_row_cap(slots: usize) -> usize {
    mixed_tick_rows() + slots * (super::spec::SPEC_K1_MAX + 1) + 64
}

pub(crate) fn mixed_tick_rows() -> usize {
    static ROWS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ROWS.get_or_init(|| {
        std::env::var("PADDOCK_G4_TICK_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| (1088..=PF_ROWS).contains(&n))
            .unwrap_or(2048.min(PF_ROWS))
    })
}

/// Width gate for the k1-deep spec attention arm (both layer classes):
/// engage only at this many chunks (slots) or more - below it the KV is
/// L2-resident and the shared walk serializes issue-bound work (c8 258->201
/// when always-on). 32 slots × 4MB window ≈ the 128MB L2; 16 is the
/// measured-safe midpoint, env-tunable for A/Bs.
fn spec_swa_min_chunks() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_G4_SPEC_SWA_MIN_CHUNKS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(16)
    })
}

/// Mirrors the pack's spec-FA fp8 kill (PADDOCK_NO_SPEC_FA_F8): the fp8
/// share-floor in the spec width gate is only sound when the pack will take
/// the FA route - with FA killed, sharing lands on the fused fp8 walk
/// (measured 615.5 vs per-row 840.1 at c8).
fn spec_fa_f8_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_SPEC_FA_F8").is_none())
}

/// Width gate for the k1-deep spec attention arm, shared by the pure
/// verify tick and the mixed tick's front verify rows (verify-fold rung
/// A). Sharing the KV walk only pays when the walk is DRAM-bound; below
/// these widths both classes are L2-resident on the 128MB GB202 and the
/// shared walk just serializes issue-bound work, which costs narrow batches
/// badly.
/// - chunk floor: see `spec_swa_min_chunks`.
/// - FA-f8 floor: under fp8 KV the pack's spec-FA route replaces the
///   serializing shared walk, and beats both per-row and shared-walk.
///   Floor 8, not the pack's 4: at live=4 the FA grid (n_kv x 4 x splits)
///   starves the die and per-row still wins.
/// - Context-volume gate: once the live slots' layer KV spills L2, per-row
///   verify re-reads pay k1x DRAM and the calculus inverts, so sharing wins
///   from roughly 1.2k ctx up. Global layers (window 0) use a 2k span
///   bound.
pub(super) fn spec_width_ok(
    live: usize,
    window: usize,
    n_kv: usize,
    hd: usize,
    dtype: crate::gpu::KvDtype,
) -> bool {
    if live >= spec_swa_min_chunks() {
        return true;
    }
    if dtype == crate::gpu::KvDtype::Fp8E4m3 && live >= 8 && spec_fa_f8_on() {
        return true;
    }
    let kvb: usize = if dtype == crate::gpu::KvDtype::Fp8E4m3 {
        1
    } else {
        2
    };
    let span = if window > 0 { window } else { 2048 };
    live * span * n_kv * hd * kvb * 2 >= (96 << 20)
}

/// Verify-fold rung A kill: PADDOCK_G4_NO_MIXED_SPECFA=1
/// restores the per-row decode kernels for the mixed tick's front verify
/// rows - the A/B surface for the spec-arm route in prefill_layers.
pub(super) fn mixed_specfa_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_G4_NO_MIXED_SPECFA").is_none())
}

/// tcgen05/TMEM decode attention election (SWA decode arm, final-output
/// contract - no partials/combine). DEFAULT on: interleaved A/B rounds put
/// every arm leg ahead of every control leg on both throughput and ITL with
/// no overlap, and its fp64-truth error gates at-or-below the v9q class and
/// is bit-deterministic. The pack entry re-gates shape/arch (rc -2) so a bad
/// election falls back to the fin1/split route instead of corrupting; the
/// r >= 16 site gate keeps c1/c4/c8 on their measured routes.
/// Kill: PADDOCK_NO_ATTN_TC5.
pub(super) fn attn_tc5_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_ATTN_TC5").is_none())
}

/// Separate kill for the GLB (window 0, hd512/G8) tc5 arm so the two
/// levers can be A/B'd independently - PADDOCK_NO_ATTN_TC5 kills both.
pub(super) fn attn_tc5_glb_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_ATTN_TC5_GLB").is_none())
}

// backend bound on queued chunked prompts: crate::service::max_chunks_inflight()
// (one shared value - a scheduler bound above the backend's fails admissions)

/// A queued chunked prefill: the whole prompt, advanced by mixed ticks.
/// (No `done` cursor - tails run whole through the coalesced batch pass,
/// which already resumes from the prefix cache internally.)
pub(crate) struct ChunkedPrefill {
    pub slot: usize,
    pub tokens: Vec<u32>,
}
