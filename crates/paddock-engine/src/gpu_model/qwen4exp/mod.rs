//! Qwen3.8-Flash-Next (`qwen4_exp`) - safetensors-primary NVFP4 family.
//!
//! Recon + rung log. 48 layers = 36 GDN +
//! 12 gated-attention-with-QSA (every 4th), 512 routed NVFP4 experts top-10 +
//! one bf16 shared expert per layer, 4-stream gated hyper-connections instead
//! of a residual, one PLE n-gram layer (decoder index 1), untied bf16 lm_head,
//! no final norm (the hyper-connection mixer feeds lm_head directly).
//!
//! CHARTER: bf16 byte-exact parity first - every residency
//! in this stage is bit-identical to the checkpoint (bf16 planes device-
//! resident as shipped, small vectors widened f32 exactly, expert nibbles
//! packed unchanged). The 8-bit dense-plane lane comes after parity, behind a
//! flag, quality-gated.
//!
//! Stage 2 = loader + GPU-side oracle only; the forward graph is the next
//! stage. MTP and vision planes are not loaded yet (spec/vision rungs).

mod forward;
mod load;
mod load_gguf;

pub use forward::Qwen4ExpGpu;
pub use load::{load_layer, load_ple_projections, load_ple_table};

use cudarc::driver::CudaSlice;

use crate::gpu::{
    DeviceTensor, ExpertCache, F8RowPlane, GpuError, GpuExecutor, HostMappedKq, Nvf4MoePlane,
    QuantTensor, QuantW, RepackedKQ,
};

/// Which class the dense projections load in. bf16 is the parity class and
/// the default; anything else is the chartered throughput lane and is LOSSY,
/// so it is opt-in and every benchmark must stamp what it ran.
///
/// Chosen once at load from `PADDOCK_Q38FN_DENSE`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum DenseClass {
    /// checkpoint bf16 bytes, device-resident unchanged
    #[default]
    Bf16,
    /// The same numbers on the tcgen05 datapath. bf16 -> f16 is exact for
    /// every value inside f16's normal range, and f16 carries more mantissa
    /// (11 bits vs bf16's 8): the load-time convert measured a worst absolute
    /// delta of zero on every plane of this checkpoint. So this is not a
    /// precision trade - it is the same weights through a unit that is 4x the
    /// warp-MMA roof (`bench/q4x_dense_probe.cu`, us, shipped bf16 -> f16tc):
    ///
    ///   plane      b8            b32           b128          b1024
    ///   gdn qkv    42.9 -> 16.2  31.8 -> 16.5  79.3 -> 20.6  396.0 -> 51.1
    ///   attn q     43.4 -> 16.6  34.4 -> 18.5  82.4 -> 20.6  427.5 -> 55.5
    ///   gdn out    24.7 -> 12.6  26.4 -> 14.1  44.6 -> 24.9  342.0 -> 39.0
    ///   lm_head   244.4 ->193.9 243.5 ->200.6 916.5 ->252.1 7341.1 ->1131.3
    ///
    /// Every plane, every width, one exception (hc up at b32, 8.3 -> 10.2).
    F16,
    /// Both the checkpoint's bf16 plane and its exact f16 twin, elected per
    /// call by batch. The f16 twin rides the tcgen05 GEMM (slot 383), which
    /// the plane-by-plane probe measures 1.2-7.8x faster than the warp-MMA
    /// tile - but the win is a per-WIDTH one, and a blanket f16 class loses on
    /// the decode rungs because the bf16 lane owns three fusions the f16 lane
    /// has no twin for (the hc 2-segment store, the hc mix epilogue, the
    /// batch-1 silu epilogue) and because each f16 call pays an activation
    /// cast. Measured end to end, ms/step, bf16 -> blanket f16:
    ///
    ///   c1 8.93 -> 12.17   c4 13.76 -> 17.33   c8 16.72 -> 16.31
    ///   c16 22.53 -> 21.57  c32 28.43 -> 31.84
    ///   prefill 512 103.8 -> 85.0   1024 204.2 -> 163.7   4096 952.3 -> 801.5
    ///
    /// So: keep the bf16 plane (and every fusion hanging off it) and add the
    /// f16 twin for the wide walks. Costs a second copy of the dense weights
    /// (+6.4 GB on this checkpoint).
    Dual,
    /// per-ROW e4m3: one f32 power-of-2 scale per output row, applied in the
    /// epilogue. Halves the plane and, at batch 1, `f8r_gemv` eats f32
    /// activations directly - so it adds no staging launch to a tick the
    /// profile showed is launch-sensitive. Lossy: coarser than the per-32
    /// block scales a Q8_0 lane would carry.
    F8Row,
}

impl DenseClass {
    pub fn label(self) -> &'static str {
        match self {
            DenseClass::Bf16 => "bf16",
            DenseClass::F16 => "f16",
            DenseClass::Dual => "bf16+f16",
            DenseClass::F8Row => "f8row",
        }
    }
    /// Bytes per weight element, for the residency ledger.
    pub fn elem_bytes(self) -> f32 {
        match self {
            DenseClass::Bf16 => 2.0,
            DenseClass::F16 => 2.0,
            DenseClass::Dual => 4.0,
            DenseClass::F8Row => 1.0,
        }
    }
}

/// Read the elected dense class from the environment. Unknown values are a
/// loud panic rather than a silent fallback - a board that believes it ran an
/// 8-bit class while running bf16 is worse than a crash.
/// Widest batch that takes the f16 tcgen05 twin in a walk that FORKS.
///
/// A forked walk declares co-residency, which clamps the lane's K-split to 1
/// (see `DenseStage::f16_ok`). That is what makes it safe, and it is also what
/// bounds it: without the split the lane beats the bf16 tile only while the
/// tile is on its own weak `batch <= 16` config tier. Swept at 9 widths in the
/// forked decode walk, ms/step, bf16 -> f16(KS=1):
///
///   w6  14.93 -> 14.43 (+3.4%)   w20 22.23 -> 24.20 (-8.9%)
///   w8  16.70 -> 15.70 (+6.0%)   w24 24.34 -> 26.99 (-10.9%)
///   w10 18.60 -> 17.43 (+6.3%)   w28 26.60 -> 29.80 (-12.0%)
///   w12 19.99 -> 17.68 (+11.6%)  w32 28.89 -> 32.38 (-12.1%)
///   w16 22.91 -> 21.22 (+7.4%)
///
/// The crossover sits exactly at the bf16 launcher's own tier boundary (it
/// leaves <32,32,4,2,128,1,2> above 16), so this is a mechanism and not a
/// fitted number. UNBOUNDED in a walk that does not fork: there the split is
/// free and the lane wins at every width measured.
pub(crate) fn f16_fork_max_batch() -> usize {
    use std::sync::OnceLock;
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("PADDOCK_Q38FN_F16FORKMAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(16)
    })
}

/// Smallest batch that takes the f16 tcgen05 twin in the `Dual` class.
/// SWEPT, not guessed: `PADDOCK_Q38FN_F16MIN`. Default 64 = PREFILL WALKS
/// only, and that is a measurement, not caution.
///
/// The decode band was opened, measured and closed again. With the racy mmaf
/// arm declined (see `DenseStage::f16_ok`) the twin is worth +1.9/+3.0/+2.7/
/// +2.6% at widths 6/8/10/12 and -1.8% at 16 - inside this box's run-to-run
/// spread, so it does not buy a second numeric class in the decode tick. The
/// +6-12% an earlier sweep showed at those widths was mmaf's speed, and mmaf
/// returns garbage: a fast wrong answer is not a win.
///
/// Prefill is different and the win there is real: 512 tok 103.8 -> 84.0 ms,
/// 1024 204.2 -> 166.1, 4096 952.3 -> 822.8. Those widths are outside mmaf's
/// own 5..32 window, so nothing about them was ever riding the race.
pub(crate) fn f16_min_batch() -> usize {
    use std::sync::OnceLock;
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("PADDOCK_Q38FN_F16MIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64)
    })
}

pub fn dense_class_from_env() -> DenseClass {
    static ELECTED: std::sync::OnceLock<DenseClass> = std::sync::OnceLock::new();
    *ELECTED.get_or_init(parse_dense_class)
}

fn parse_dense_class() -> DenseClass {
    match std::env::var("PADDOCK_Q38FN_DENSE").as_deref() {
        Err(_) | Ok("") | Ok("dual") | Ok("bf16+f16") => DenseClass::Dual,
        Ok("bf16") => DenseClass::Bf16,
        Ok("f16") | Ok("f16tc") => DenseClass::F16,
        Ok("f8row") | Ok("f8r") => DenseClass::F8Row,
        Ok(other) => panic!("PADDOCK_Q38FN_DENSE={other:?} is not a dense class this build knows"),
    }
}

/// A dense weight plane in whichever class this lane elected for it.
///
/// `Bf16` is the PARITY class - the checkpoint's own bytes, device-resident
/// unchanged, and the class every gate in `tests/gpu_qwen4exp_*.rs` was
/// stamped against. The 8-bit variants are the chartered throughput lane
/// (dense planes may move to our 8-bit class after bf16
/// parity, behind a flag, quality-gated, with every benchmark stamping the class
/// honestly) - they are lossy, and nothing elects them by default.
///
/// Routing every dense projection through one type is what keeps the class a
/// LOAD-TIME decision instead of 20 call sites that can disagree.
pub enum DensePlane {
    /// checkpoint bf16, as shipped
    Bf16(QuantTensor),
    /// the bf16 plane AND its exact f16 twin; `matmul` elects by batch and
    /// every fused entry keeps using the bf16 half.
    Dual {
        w: QuantTensor,
        w16: CudaSlice<half::f16>,
        in_dim: usize,
        out_dim: usize,
    },
    /// the checkpoint's bf16 numbers, exactly, in f16 - for the in-house
    /// tcgen05 GEMM (slot 383), the only datapath in this pack that has beaten
    /// nvjet on a dense plane. Carries its own dims for the same reason the
    /// 8-bit variant does.
    F16 {
        w: CudaSlice<half::f16>,
        in_dim: usize,
        out_dim: usize,
    },
    /// per-row e4m3 + f32 row scales. Carries its own dims: `F8RowPlane` has
    /// none, and a transposed GEMM off one is silent.
    F8Row {
        plane: F8RowPlane,
        in_dim: usize,
        out_dim: usize,
    },
    /// GGUF k-quant / Q8_0 plane (the Unsloth exports): the repacked
    /// streams the qwen35 dense lanes read - fused GEMV at batch 1, the
    /// int8 dp4a GEMM above it (activations quantized per 32). Exact int
    /// dots with f32 block scales, the same numeric class as qwen35's
    /// k-quant dense seats. Carries its dims like the 8-bit classes do.
    Kq {
        w: QuantW,
        in_dim: usize,
        out_dim: usize,
    },
}

/// Staging the 8-bit classes need when the activation must be quantized too
/// (the batch > 1 arm). Owned by the model, not the scratch, so a call site
/// can borrow an activation and the staging at the same time.
pub struct DenseStage {
    pub q: cudarc::driver::CudaSlice<i8>,
    pub rs: CudaSlice<f32>,
    /// `Kq` class only: per-32 activation scales for `q` (`quantize_q8`),
    /// the per-16 int sums the Q4/Q5 min term reads, and the split-K
    /// partials the ks GEMM wants. Empty (len 0) when no Kq plane loaded.
    pub xs: CudaSlice<f32>,
    pub ssums: CudaSlice<f32>,
    /// `Kq` class above 64 rows: the mmq-layout int8 tiles
    /// (`quantize_q8_mmq`) and their per-32 sums the W4A8 tile GEMM reads,
    /// for one `KQ_TILE_ROWS`-row chunk of the widest dense plane (the
    /// launch loops over chunks); len 1 without Kq.
    pub yq: CudaSlice<u8>,
    pub xsums: CudaSlice<f32>,
    /// f16 view of the activation the F16 class feeds `pd_f16_gemm`. One cast
    /// per (plane, tick); the same buffer serves every plane because the walk
    /// is strictly sequential inside a layer.
    pub x16: CudaSlice<half::f16>,
    /// Phase gate for the low-M cluster arm: decode walks only (serial
    /// prefill electing it while the wave cannot split the prefill paths
    /// into different f32 orders - battery-proven).
    pub lowm_ok: bool,
    /// bf16 view of the activation for the TGV lane (slot 547); one cast
    /// per (plane, call), same contract as `x16`.
    pub xb16: CudaSlice<half::bf16>,
    /// May this walk take the f16 tensor-core twin? Set per walk from the
    /// phase, and false wherever the walk FORKS a side stream.
    ///
    /// This is a hazard bound, not a preference. A `pd_f16_gemm` running
    /// concurrently with a forked side stream HANGS the device at batch 32 -
    /// reproduced with `PADDOCK_Q38FN_F16MIN=8`, and cleared by
    /// `PADDOCK_Q38FN_FORK=0` on the same binary, which is what identifies the
    /// interaction rather than the width. (It is the hang class the f16 tc5
    /// ledger already records: remote-mbarrier TMA completion.) Graph capture
    /// is not involved - `PADDOCK_Q38FN_NO_GRAPH=1` hangs identically - and no
    /// plane reproduces it in isolation: all fourteen of them run clean at b32
    /// in `bench/q4x_dense_probe.cu`.
    ///
    /// The decode phases fork; the prefill phases do not. So the twin is
    /// scoped to the prefill walks, which is where the probe puts its win
    /// anyway (+13-24% end to end) - and c8/c16 keep the +11% the decode
    /// widths measured, as a BLOCKED win rather than a declined one.
    pub f16_ok: bool,
    /// Widest batch the twin may take in this walk. Unbounded where the walk
    /// does not fork (the K-split is free and the lane wins at every width);
    /// `f16_fork_max_batch()` where it does, because a clamped split only beats
    /// the bf16 tile while the tile is on its weak `batch <= 16` tier.
    pub f16_max: usize,
}

/// The batched-runs prefill wave (`forward_prefill_batch`). DEFAULT-ON;
/// `PADDOCK_Q38FN_PREFILL_WAVE=0` restores the trait default (one prompt at a
/// time), which is the A/B arm.
pub(crate) fn prefill_wave_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("PADDOCK_Q38FN_PREFILL_WAVE").ok().as_deref(),
            Some("0") | Some("off")
        )
    })
}

/// The parallel-score decode attention (slot 536). DEFAULT on;
/// `PADDOCK_Q38FN_ATTN_PS=0` restores the shipped serial-score walk.
pub(crate) fn attn_ps_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("PADDOCK_Q38FN_ATTN_PS").ok().as_deref(),
            Some("0") | Some("off")
        )
    })
}

/// FMHA-style decode attention (slot 537). Default on; `PADDOCK_Q38FN_ATTN_FMHA=0`
/// falls back to the tile walk. Its own numeric class - see the kernel comment.
/// Load-time plane fusion, the rival's own layout (qwen3_5.py fuses
/// qkv_proj and in_proj_qkvz at load; MergedColumnParallelLinear). One wide
/// launch replaces three narrow ones, and the decode-band mma kernel's total
/// warp count is M/16 -- so fusing is the occupancy lever the c8 ncu pass
/// said the band is starved on (6.7% warp occupancy, 13.7% DRAM on the
/// unsplit qkv plane). Opt-in until the serve A/B stamps it.
pub(crate) fn fuse_attn_qkv_on() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("PADDOCK_Q38FN_FUSE_QKV").ok().as_deref(),
            Some("1") | Some("on")
        )
    })
}

/// GDN twin of [`fuse_attn_qkv_on`]: in_proj_z | in_proj_qkv as one 2-segment
/// plane (the rival's in_proj_qkvz, minus ba which stays on the f32 matvec).
pub(crate) fn fuse_gdn_zq_on() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("PADDOCK_Q38FN_FUSE_ZQ").ok().as_deref(),
            Some("1") | Some("on")
        )
    })
}

/// The low-M cluster GEMM (slot 543): the pr4266-class decode kernel.
/// Opt-in until its per-shape A/B and battery land.
/// Selective tc5g-at-decode election (tc5g-direct probe 2026-08-30): at
/// b<=8 the tcgen05 skinny arm beats the GEMV band on every Dual plane
/// EXCEPT the wide-out pair (in=2560, out>=10240). Pair with
/// PADDOCK_NO_F16GEMV=1 so pd_f16_gemm reaches tc5g at these widths.
pub(crate) fn tc5b1_on() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("PADDOCK_Q38FN_TC5B1").ok().as_deref(),
            Some("1") | Some("on")
        )
    })
}

pub(crate) fn lowm_on() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("PADDOCK_Q38FN_LOWM").ok().as_deref(),
            Some("1") | Some("on")
        )
    })
}

pub(crate) fn exp_lt_on() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("PADDOCK_EXP_CUBLASLT").ok().as_deref(),
            Some("1") | Some("on")
        )
    })
}

pub(crate) fn dn_prenorm_on() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("PADDOCK_Q38FN_DN_PN").ok().as_deref(),
            Some("1") | Some("on")
        )
    })
}

pub(crate) fn router_fold_on() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("PADDOCK_Q38FN_ROUTER_FOLD").ok().as_deref(),
            Some("1") | Some("on")
        )
    })
}

pub(crate) fn fuse_sh_on() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("PADDOCK_Q38FN_FUSE_SH").ok().as_deref(),
            Some("1") | Some("on")
        )
    })
}

/// tcgen05 decode attention (pack slot 431, <256,6>). Rides only on e4m3
/// pools (PADDOCK_Q38FN_KV8=1). Default on where eligible;
/// `PADDOCK_Q38FN_ATTN_TC5=0` restores the SIMT fmha walk.
pub(crate) fn attn_tc5_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !matches!(std::env::var("PADDOCK_Q38FN_ATTN_TC5").as_deref(), Ok("0")))
}

pub(crate) fn sh2_on() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var_os("PADDOCK_Q38FN_SH2").is_some_and(|v| v == "1"))
}

/// slot 562: fold `hc_mix` into the HC up plane's epilogue (n==1 decode lane).
/// Default on; `PADDOCK_Q38FN_FUSE_UPMIX=0` restores the up+mix pair.
pub(crate) fn fuse_upmix_on() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("PADDOCK_Q38FN_FUSE_UPMIX").as_deref(),
            Ok("0")
        )
    })
}

/// Route the n==1 ROUTER through the CTA-per-row bf16 gemv instead of TGV.
/// TGV tiles 64 output rows a CTA, so a 513-row router grids NINE CTAs on a
/// 148-SM die - 5.36 us in-walk for 2.63 MB (0.49 TB/s), while the
/// block-per-row gemv measures 2.50 us a graph node at the same shape.
/// `PADDOCK_Q38FN_ROUTER_GEMV=0` restores the TGV arm.
pub(crate) fn router_gemv_on() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("PADDOCK_Q38FN_ROUTER_GEMV").as_deref(),
            Ok("0")
        )
    })
}

/// The HC island (slots 571/572) - the rival's own mix structure: two low-M
/// GEMMs with fused epilogues. `PADDOCK_Q38FN_HC_ISLAND=0` pins the current
/// down/ks_combine/up chain.
pub(crate) fn hc_island_on() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    // DEFAULT off: the island measures +3.9% at c8 (686.3 vs 660.7) but its
    // output DIVERGES from the chain at every layer (hc_m relerr ~0.74 at the
    // very first attn pass), so that speed is on wrong math. Ruled out so far:
    // the xn mirror (a forced fresh cast diverges identically), graph capture
    // (NO_GRAPH is byte-identical), and both weight-prep kernels (load-time
    // self-check matches src/built exactly). The kernels themselves verify
    // end-to-end offline against an independent reference (mix relerr 1.9e-3,
    // inject 1.3e-3), so the fault is in what the walk hands them. Next step:
    // bisect by keeping the chain'''s down and swapping only the up.
    *ON.get_or_init(|| matches!(std::env::var("PADDOCK_Q38FN_HC_ISLAND").as_deref(), Ok("1")))
}

/// Route only the hc down plane through the low-M split-K gemm (the island's
/// one measured win, without the island's bolted-on inject gemm and casts).
/// `PADDOCK_Q38FN_HC_DOWN_P42=0` restores the 2seg MMA arm.
pub(crate) fn hc_down_p42_on() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("PADDOCK_Q38FN_HC_DOWN_P42").as_deref(),
            Ok("0")
        )
    })
}

/// Block-per-row batched gemv for narrow-output planes (slot 565).
pub(crate) fn mrow_on() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    // MEASURED A LOSS at c8 (580.05/580.09 vs 607.34 with it off): a
    // block-per-row arm re-reads the whole activation matrix per output row -
    // 104 MB at batch 8 against the weight'''s 6.55 MB - which is exactly what
    // the tiled MMA arm stages in smem to avoid. Opt-in only.
    *ON.get_or_init(|| matches!(std::env::var("PADDOCK_Q38FN_MROW").as_deref(), Ok("1")))
}

/// Widest output the block-per-row batched arm takes (default 1024: above it
/// the tiled MMA arm has enough CTAs to fill the die).
pub(crate) fn mrow_max_out() -> usize {
    use std::sync::OnceLock;
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("PADDOCK_Q38FN_MROW_MAX_OUT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1024)
    })
}

/// The bare bf16 bytes of a plane, where one exists (the slot-546 election
/// needs the raw pair).
pub(crate) fn plane_bytes(p: &DensePlane) -> Option<&QuantTensor> {
    match p {
        DensePlane::Bf16(w) | DensePlane::Dual { w, .. } => Some(w),
        _ => None,
    }
}

pub(crate) fn attn_fmha_sp() -> usize {
    use std::sync::OnceLock;
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("PADDOCK_Q38FN_FMHA_SP")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&s| (2..=16).contains(&s))
            .unwrap_or(0)
    })
}

pub(crate) fn attn_fmha_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("PADDOCK_Q38FN_ATTN_FMHA").ok().as_deref(),
            Some("0") | Some("off")
        )
    })
}

/// The f16 lane's fine-M decode arm (batch 5..32). Default on since the
/// intra-pair park race was fixed; `PADDOCK_Q38FN_MMAF=0` declines it.
pub(crate) fn mmaf_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("PADDOCK_Q38FN_MMAF").ok().as_deref(),
            Some("0") | Some("off")
        )
    })
}

pub(crate) fn gdn_fork_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("PADDOCK_Q38FN_FORK").ok().as_deref(),
            Some("0") | Some("off")
        )
    })
}

/// Kill switch for the fused scale+silu epilogue (slot 520).
fn fuse_silu_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("PADDOCK_Q38FN_FUSESILU").ok().as_deref(),
            Some("0") | Some("off")
        )
    })
}

/// Widest batch that routes to the multi-row narrow-K arm (slot 522).
///
/// MEASURED CEILING, not a guess. The arm carries BT = 8 accumulators per warp,
/// so a batch wider than 8 costs it one extra WEIGHT read per 8-row group,
/// while the stock path above 8 is the tile GEMM, which reads the weight once.
/// At batch 32: arm 321.9 tok/s against the tile GEMM's 479.7. Below the tile
/// GEMM's window the stock fallback is `pd_bf16_gemv_mr_f32` and the arm wins
/// (c4 150.9 -> 170.7, c8 172.5 -> 201.6). `PADDOCK_Q38FN_NKMR=0` disables it.
/// Top of the batch window the narrow-K multi-row GEMV arm serves. DEFAULT 1
/// - i.e. batch 1 only.
///
/// It was 8, from a sweep taken before the routed MoE moved to the fp4
/// tensor-core lane, and the decode-cadence profile at c8 says what that costs
/// now: `bf16_gemv_nk_mr` 13.85 ms/tick and `bf16_gemv_mr` 5.56 - 19.4 ms of a
/// 34.4 ms busy tick - where the same planes at c32 cost 7.08 ms on the
/// tensor-core tile with four times the rows. Same-load width sweep, ms/step,
/// both GEMV arms on -> both off:
///
///   c1 8.93 -> 8.92   c4 16.31 -> 13.77   c8 24.09 -> 16.70 (+44%)
///   c16 22.90 -> 22.44   c32 28.29 -> 28.48   (16/32 never used the arm)
///
/// It also explained the ladder's shape: c8 served a 21.6 ms ITL against
/// c32's 14.5 with a QUARTER of the rows.
fn nk_gemv_max_batch() -> usize {
    use std::sync::OnceLock;
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("PADDOCK_Q38FN_NKMR")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1)
    })
}

/// Whether this family's dense planes take the multi-row GEMV band (2..=8) at
/// all. Default off - see `nk_gemv_max_batch` for the measurement.
/// `PADDOCK_Q38FN_MR=1` restores it (the A/B arm).
fn mr_band_enabled() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("PADDOCK_Q38FN_MR").ok().as_deref(),
            Some("1") | Some("on")
        )
    })
}

fn nk_gemv_min_out() -> usize {
    use std::sync::OnceLock;
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("PADDOCK_Q38FN_NKGEMV_MINOUT")
            .ok()
            .and_then(|v| v.parse().ok())
            // 8 warps/block * 148 SMs on this die: the smallest out_dim that
            // gives the arm one block per SM. SWEPT: 512 -> 9.6 ms/tok,
            // 1184 -> 9.2, 1280 -> 9.2, 2048 -> 9.2. Below a full wave the
            // arm's grid starves (out=640 gives 80 blocks) and it loses to
            // the stock GEMV, which keeps one block per output row.
            .unwrap_or(1184)
    })
}

fn nk_gemv_max_in() -> usize {
    use std::sync::OnceLock;
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(
        || match std::env::var("PADDOCK_Q38FN_NKGEMV").ok().as_deref() {
            // MEASURED default: unbounded. With the out_dim floor below doing the
            // real gating, every wide-out decode plane wins on the arm
            // (10.6 -> 9.6 ms/tok). An in_dim ceiling only ever cost throughput
            // here; it stays configurable because it is the knob that isolates the
            // arm from the floor when a shape misbehaves.
            None => usize::MAX,
            Some("off") => 0,
            Some(v) => v.parse().unwrap_or_else(|_| {
                panic!("PADDOCK_Q38FN_NKGEMV: want an in_dim bound or `off`, got {v:?}")
            }),
        },
    )
}

/// Per-site witness of the dense election (diagnostic only, off by default).
///
/// The census-by-grid-shape approach mis-attributes planes whose grids
/// collide, so the lane a plane actually takes is recorded at its CALL SITE
/// instead: `#[track_caller]` carries the forward.rs line through, and each
/// distinct (site, shape, batch, arm) is printed once.
/// `PADDOCK_Q38FN_DENSE_WITNESS=1`.
pub(crate) fn dense_witness_on() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("PADDOCK_Q38FN_DENSE_WITNESS").ok().as_deref(),
            Some("1") | Some("on")
        )
    })
}

pub(crate) fn witness_once(tag: &'static str, batch: usize, n: usize, k: usize) {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    if !dense_witness_on() {
        return;
    }
    static SEEN: OnceLock<Mutex<HashSet<(&'static str, usize, usize, usize)>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    let _ = (batch, n, k);
    if seen
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert((tag, 0, 0, 0))
    {
        eprintln!("[arm] {tag} batch={batch} n={n} k={k}");
    }
}

pub(crate) fn dense_site(
    site: &'static std::panic::Location<'static>,
    in_dim: usize,
    out_dim: usize,
    batch: usize,
    arm: &'static str,
) {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    if !dense_witness_on() {
        return;
    }
    static SEEN: OnceLock<Mutex<HashSet<(u32, usize, usize, usize, &'static str)>>> =
        OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
    let key = (site.line(), in_dim, out_dim, batch, arm);
    if seen
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(key)
    {
        eprintln!(
            "[dense-site] {}:{} in={in_dim} out={out_dim} batch={batch} arm={arm}",
            site.file().rsplit('/').next().unwrap_or(site.file()),
            site.line()
        );
    }
}

/// The bf16 dense election, shared by the `Bf16` and `Dual` planes so the two
/// cannot drift: a sub-threshold `Dual` call must be byte-for-byte the route
/// `Bf16` would have taken.
#[track_caller]
fn bf16_matmul(
    e: &GpuExecutor,
    w: &QuantTensor,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    batch: usize,
) -> Result<(), GpuError> {
    let site = std::panic::Location::caller();
    // EXPERIMENT ceiling probe: the whole bf16 decode band through cuBLASLt.
    if exp_lt_on() && batch >= 2 && e.exp_lt_gemm(&w.bytes, x, y, w.dims[0], w.dims[1], batch)? {
        dense_site(site, w.dims[0], w.dims[1], batch, "EXP_cublasLt");
        return Ok(());
    }
    // Narrow-K arm (slot 518). The stock GEMV walks each output
    // row as `base = tid*16` over 128 threads, which assumes
    // in_dim >= 2048, and gives every row its own block. The
    // hyper-connection up plane is [in=320, out=10240]: 20 of 128
    // threads load and the launch is 10240 blocks of 640 B -
    // measured 570 GB/s against 3199 GB/s this same kernel gets on
    // the lm_head. Election is by SHAPE and scoped to this model.
    // The arm gives each output row one warp and packs 8 rows per
    // block, so out_dim < 8*148 cannot fill this die: measured, the
    // hc down plane [in=10240, out=324] collapses to 41 blocks and
    // routing it here costs more than the whole arm wins. Hence a
    // floor on out_dim, not just a ceiling on in_dim.
    let shaped = w.dims[0] <= nk_gemv_max_in() && w.dims[1] >= nk_gemv_min_out();
    if shaped {
        if batch == 1 {
            if e.bf16_gemv_nk(w, None, x, y)? {
                dense_site(site, w.dims[0], w.dims[1], batch, "bf16_gemv_nk");
                return Ok(());
            }
        } else if batch <= nk_gemv_max_batch() {
            // batch > 1 without this lands on pd_bf16_gemv_mr_f32 at
            // ~39 us a launch, which is why the batched tick carried
            // none of this lane's single-row speedup.
            if e.bf16_gemv_nk_mr(w, x, y, batch)? {
                dense_site(site, w.dims[0], w.dims[1], batch, "bf16_gemv_nk_mr");
                return Ok(());
            }
        }
    }
    if mr_band_enabled() {
        dense_site(site, w.dims[0], w.dims[1], batch, "bf16_gemm(mr)");
        e.bf16_gemm(w, None, x, y, batch)
    } else {
        dense_site(site, w.dims[0], w.dims[1], batch, "bf16_gemm_tile");
        e.bf16_gemm_tile(w, None, x, y, w.dims[1], batch)
    }
}

impl DensePlane {
    /// `y = W x` over `batch` rows. Same operand convention as the bf16 lane:
    /// `x` is `[batch, in_dim]`, `y` is `[batch, out_dim]`.
    #[track_caller]
    pub fn matmul(
        &self,
        e: &GpuExecutor,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        batch: usize,
        stage: &mut DenseStage,
    ) -> Result<(), GpuError> {
        let site = std::panic::Location::caller();
        match self {
            DensePlane::Bf16(w) => bf16_matmul(e, w, x, y, batch),
            DensePlane::Dual {
                w,
                w16,
                in_dim,
                out_dim,
            } => {
                // low-M cluster arm (slot 543): decode-band widths on the f16
                // twin; declines by shape and falls through to the ladder.
                // WINNER SHAPES only (probe round 4b): sh-down class
                // (in=640) 7.6 vs 10, z class (in=2560, out<=6144) 12.7 vs
                // 13.5; the wide-out and deep-K planes stay on their bands.
                let lowm_win = (*in_dim == 640) || (*in_dim == 2560 && *out_dim <= 6144);
                if (1..=8).contains(&batch)
                    && lowm_on()
                    && stage.f16_ok
                    && stage.lowm_ok
                    && lowm_win
                {
                    // round 5: x goes in as f32 - the kernel stages+casts.
                    if e.lowm_gemm(w16, x, y, *in_dim, *out_dim, batch)? {
                        dense_site(site, *in_dim, *out_dim, batch, "lowm_cluster");
                        return Ok(());
                    }
                }
                let f16_band = stage.f16_ok && batch >= f16_min_batch() && batch <= stage.f16_max;
                // selective decode carve-out: winner shapes only (probe
                // 2026-08-30; the wide-out pair stays on the bf16 GEMV band
                // where it is faster).
                let f16_sel = stage.f16_ok
                    && tc5b1_on()
                    && batch <= 8
                    && !(*in_dim == 2560 && *out_dim >= 10240);
                if f16_band || f16_sel {
                    dense_site(site, *in_dim, *out_dim, batch, "f16_gemm(tc5)");
                    e.convert_f32_f16(x, &mut stage.x16, batch * *in_dim)?;
                    return e.f16_gemm(w16, &stage.x16, y, *in_dim, *out_dim, batch, 0.0);
                }
                // below the threshold this must be the bf16 arm exactly -
                // including its narrow-K GEMV election, which a first cut
                // skipped and paid 8.93 -> 10.71 ms/step at batch 1 for
                bf16_matmul(e, w, x, y, batch)
            }
            DensePlane::F16 { w, in_dim, out_dim } => {
                dense_site(site, *in_dim, *out_dim, batch, "f16_gemm(F16 class)");
                // one cast per call: `x` is [batch, in_dim] f32 and the tc5
                // GEMM takes f16 on both sides. At c32 that is 327 KB, against
                // a plane read of tens of MB.
                e.convert_f32_f16(x, &mut stage.x16, batch * *in_dim)?;
                e.f16_gemm(w, &stage.x16, y, *in_dim, *out_dim, batch, 0.0)
            }
            DensePlane::F8Row {
                plane,
                in_dim,
                out_dim,
            } => {
                dense_site(
                    site,
                    *in_dim,
                    *out_dim,
                    batch,
                    if batch == 1 { "f8r_gemv" } else { "f8row_gemm" },
                );
                if batch == 1 {
                    // f32 activations straight in - no quantize launch, which
                    // is why this class was elected first on a tick whose
                    // profile is launch-sensitive.
                    e.f8r_gemv(plane, x, y, *in_dim, *out_dim)
                } else {
                    // W8A8: the row-scaled activation pair the batched GEMM
                    // wants. One staging launch per call here; the batch arm
                    // is not what this lane's benchmark runs.
                    e.quantize_e4m3_row(x, &mut stage.q, &mut stage.rs, *in_dim, batch)?;
                    e.f8row_gemm(plane, &stage.q, &stage.rs, y, *in_dim, *out_dim, batch)
                }
            }
            DensePlane::Kq { w, in_dim, out_dim } => {
                dense_site(
                    site,
                    *in_dim,
                    *out_dim,
                    batch,
                    if batch == 1 { "kq_gemv" } else { "kq_gemm" },
                );
                kq_matmul(e, w, x, y, batch, stage)
            }
        }
    }

    /// `y = W[first_row .. first_row+out_dim] x` - the batch > 1 arm of a
    /// launch-folded plane, which holds two projections in one residency and
    /// reads them as row segments when the fused single call cannot serve
    /// (a fused output is only contiguous per projection at batch 1).
    #[allow(clippy::too_many_arguments)]
    /// One launch over a plane folding two projections: rows `[0, oq)` to
    /// `ya`, `[oq, oq + ob)` to `yb`. `Ok(false)` means "not taken, do it
    /// yourself" - the 8-bit classes hold no folded planes.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_2seg(
        &self,
        e: &GpuExecutor,
        oq: usize,
        ob: usize,
        x: &CudaSlice<f32>,
        ya: &mut CudaSlice<f32>,
        yb: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<bool, GpuError> {
        match self {
            DensePlane::Bf16(w) | DensePlane::Dual { w, .. } => {
                // narrow-output planes tile to a handful of CTAs in the MMA
                // arm ([in=10240, out=324] is ELEVEN at batch 8, 16.06 us for
                // 6.55 MB); block-per-row with BT accumulators fills the die
                // and reads the weight once. PADDOCK_Q38FN_MROW=0 pins the
                // MMA arm back.
                if batch >= 2
                    && mrow_on()
                    && w.dims[1] <= mrow_max_out()
                    && e.bf16_gemv_mrow(w, x, ya, Some(yb), None, batch, oq, 0, 1.0)?
                {
                    return Ok(true);
                }
                e.bf16_gemm_2seg(w, x, ya, yb, oq, ob, batch)
            }
            // no f16 twin of the segmented store; the caller's two-call
            // fallback is correct and each half still rides the tc5 GEMM
            DensePlane::F16 { .. } | DensePlane::F8Row { .. } | DensePlane::Kq { .. } => Ok(false),
        }
    }

    /// `matmul` with the hyper-connection scale+silu folded into the epilogue
    /// over the first `silu_rows` output rows (slot 520). Only the bf16 class
    /// at batch 1 - the 8-bit classes have their own epilogues and the batched
    /// path stages activations. `Ok(false)` means "not taken, do it yourself".
    pub fn matmul_silu(
        &self,
        e: &GpuExecutor,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        mirror: Option<&mut CudaSlice<half::bf16>>,
        batch: usize,
        silu_rows: usize,
        inv: f32,
    ) -> Result<bool, GpuError> {
        if batch != 1 || !fuse_silu_enabled() {
            return Ok(false);
        }
        match self {
            DensePlane::Bf16(w) | DensePlane::Dual { w, .. } => {
                e.bf16_gemv_silu(w, x, y, mirror, silu_rows, inv)
            }
            _ => Ok(false),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn matmul_rows(
        &self,
        e: &GpuExecutor,
        first_row: usize,
        out_dim: usize,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        batch: usize,
        stage: &mut DenseStage,
    ) -> Result<(), GpuError> {
        match self {
            DensePlane::Bf16(w) | DensePlane::Dual { w, .. } => {
                e.bf16_gemm_rows(w, first_row, out_dim, x, y, batch)
            }
            // a row segment of an f16 plane is just the plane at a row offset
            DensePlane::F16 { w, in_dim, .. } => {
                e.convert_f32_f16(x, &mut stage.x16, batch * *in_dim)?;
                e.f16_gemm_rows(w, first_row, *in_dim, out_dim, &stage.x16, y, batch)
            }
            DensePlane::F8Row { .. } | DensePlane::Kq { .. } => Err(GpuError::Unsupported(
                "matmul_rows: the 8-bit and k-quant dense classes hold no folded planes".into(),
            )),
        }
    }

    /// The plane's raw device bytes, when it is resident in the checkpoint's
    /// own class. `None` for the lossy classes, which hold a re-encoding -
    /// so a byte-identity oracle cannot silently pass on one.
    pub fn raw_bf16(&self) -> Option<&CudaSlice<u8>> {
        match self {
            DensePlane::Bf16(w) | DensePlane::Dual { w, .. } => Some(&w.bytes),
            DensePlane::F16 { .. } | DensePlane::F8Row { .. } | DensePlane::Kq { .. } => None,
        }
    }

    /// Device bytes this plane occupies - for the load-time residency ledger.
    pub fn bytes(&self) -> usize {
        match self {
            DensePlane::Bf16(w) => w.dims.iter().product::<usize>() * 2,
            DensePlane::Dual {
                in_dim, out_dim, ..
            } => in_dim * out_dim * 4,
            DensePlane::F16 {
                in_dim, out_dim, ..
            } => in_dim * out_dim * 2,
            DensePlane::F8Row {
                in_dim, out_dim, ..
            } => in_dim * out_dim + out_dim * 4,
            DensePlane::Kq { w, .. } => w.bytes() as usize,
        }
    }

    /// What to print alongside a result so the class is never implicit.
    pub fn class(&self) -> &'static str {
        match self {
            DensePlane::Bf16(_) => "bf16",
            DensePlane::Dual { .. } => "bf16+f16",
            DensePlane::F16 { .. } => "f16",
            DensePlane::F8Row { .. } => "f8row",
            DensePlane::Kq { .. } => "kq",
        }
    }
}

/// `y = W x` over `batch` rows on a GGUF-quantized plane. Batch 1 rides the
/// fused k-quant / Q8_0 GEMV off f32 activations; above it the activations
/// quantize per-32 into `stage.q`/`stage.xs` and the int8 dp4a GEMM runs
/// (the Q4/Q5 min term needs the per-16 sums). Q8_0 planes take the
/// repacked GEMM off f32 rows directly, as qwen35's `mm_q8` does.
/// Rows per launch of the > 64-row W4A8 tile in [`kq_matmul`]: four tile
/// columns, so the mmq scratch stays a few MB at any context length.
pub(crate) const KQ_TILE_ROWS: usize = 512;

fn kq_matmul(
    e: &GpuExecutor,
    w: &QuantW,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    batch: usize,
    stage: &mut DenseStage,
) -> Result<(), GpuError> {
    match w {
        QuantW::Q8(q) => {
            // the qwen35 `mm_q8` ladder: gemv, the small-batch tiled GEMM,
            // the plain per-row GEMM above 12 rows
            if batch == 1 {
                e.q8_0_gemv_repacked(q, None, x, y)
            } else if batch <= 12 {
                e.q8_0_gemm_repacked_mt(q, None, x, y, batch)
            } else {
                e.q8_0_gemm_repacked(q, None, x, y, batch)
            }
        }
        QuantW::Kq(k) => {
            let in_dim = k.dims[0];
            let n = batch * in_dim;
            if stage.q.len() < n || stage.xs.len() < n / 32 || stage.ssums.len() < n / 16 {
                return Err(GpuError::Unsupported(format!(
                    "kq_matmul: stage scratch holds {} rows of {in_dim}, launch wants {batch}",
                    stage.q.len() / in_dim.max(1)
                )));
            }
            let needs = crate::gpu::kq_needs_sums(k.ty);
            // > 64 rows: the mmq tiles and the W4A8 tile GEMM - the qwen35
            // prefill rung (`kq_mm_pre`) - read the plane once per 128-column
            // tile where the dp4a walk below reads it once per token, in
            // KQ_TILE_ROWS-row chunks off a fixed scratch. An i-quant plane
            // needs the pack's window-unpack tile (slot 580).
            let iq = crate::gpu::kq_is_iq(k.ty);
            let chunk_rows = in_dim.div_ceil(128) * KQ_TILE_ROWS;
            if batch > 64
                && in_dim.is_multiple_of(256)
                && e.has_kquant_gemm_w4a8_pipe2()
                && (!iq || e.has_kquant_iq_tile())
                && stage.yq.len() >= chunk_rows * 144
                && stage.xsums.len() >= chunk_rows * 4
                && paddock_models::dev_var_os!("PADDOCK_Q4X_NO_KQ_TILE").is_none()
            {
                // Q2_K's per-16 min is built inside the tile; the per-32
                // sums serve the k-quant mu formats only
                let sums = needs && !iq;
                let mut off = 0;
                while off < batch {
                    let rows = (batch - off).min(KQ_TILE_ROWS);
                    e.quantize_q8_mmq_rows(x, off, &mut stage.yq, in_dim, rows)?;
                    if sums {
                        e.mmq_sums(&stage.yq, &mut stage.xsums, in_dim, rows)?;
                    }
                    e.kquant_gemm_w4a8_pipe2_rows(
                        k,
                        &stage.yq,
                        sums.then_some(&stage.xsums),
                        y,
                        off,
                        rows,
                    )?;
                    off += rows;
                }
                return Ok(());
            }
            if batch == 1 {
                // the qwen35 serving class (`kq_w4a8_b1`): int8 activations
                // through the W4A8 GEMV, the exact-f32 GEMV stays the oracle
                // (PADDOCK_KQ_EXACT_GEMV=1 pins it, as there)
                if e.has_kquant_gemv_w4a8()
                    && paddock_models::dev_var_os!("PADDOCK_KQ_EXACT_GEMV").is_none()
                {
                    e.quantize_q8_sums(x, &mut stage.q, &mut stage.xs, &mut stage.ssums, in_dim)?;
                    return e.kquant_gemv_w4a8(
                        k,
                        &stage.q,
                        &stage.xs,
                        needs.then_some(&stage.ssums),
                        y,
                    );
                }
                return e.kquant_gemv(k, x, y);
            }
            e.quantize_q8(x, &mut stage.q, &mut stage.xs, n)?;
            if needs {
                e.q8_sums_strided(&stage.q, &mut stage.ssums, in_dim, batch)?;
            }
            e.kquant_gemm_dp4a(
                k,
                &stage.q,
                &stage.xs,
                needs.then_some(&stage.ssums),
                y,
                batch,
            )
        }
    }
}

/// The token embedding table: bf16 as shipped (safetensors), or the GGUF's
/// k-quant plane on the k-quant gather.
pub enum Embed {
    Bf16(QuantTensor),
    Kq(RepackedKQ),
}

/// One routed-expert plane of the GGUF lane: in VRAM, or in device-mapped
/// host memory under `[moe_offload]` (same `RepackedKQ` view either way, so
/// every launch is unchanged - the qwen35 `ExpW` shape).
pub enum KqSeat {
    Dev(RepackedKQ),
    Host(HostMappedKq),
}

impl KqSeat {
    pub fn kq(&self) -> &RepackedKQ {
        match self {
            KqSeat::Dev(w) => w,
            KqSeat::Host(w) => w,
        }
    }
    pub fn host(&self) -> Option<&HostMappedKq> {
        match self {
            KqSeat::Dev(_) => None,
            KqSeat::Host(w) => Some(w),
        }
    }
    /// Bytes this seat holds, and where (device, host).
    pub fn bytes(&self) -> (u64, u64) {
        match self {
            KqSeat::Dev(w) => ((w.data.len() + w.scales.len()) as u64, 0),
            KqSeat::Host(w) => (0, w.host_bytes()),
        }
    }
}

/// The routed experts of one layer, by residency class.
pub enum ExpertSeats {
    /// NVFP4 safetensors planes, checkpoint nibbles unchanged (the parity
    /// lane).
    Nvf4 {
        gate: Nvf4MoePlane,
        up: Nvf4MoePlane,
        down: Nvf4MoePlane,
    },
    /// GGUF k-quant / i-quant expert seats on the repacked stream
    /// (`moe/kquant.cuh`, IQ1_S..IQ4_NL through `quant/iquant.cuh`), with
    /// the optional VRAM slot cache over host-mapped planes
    /// (`gpu/moe_cache.rs`) that `enable_moe_cache` seats.
    Kq {
        gate: KqSeat,
        up: KqSeat,
        down: KqSeat,
        cache: Option<Box<ExpertCache>>,
    },
}

/// One hyper-connection sub-block (attn_ and mlp_ each have one; the final
/// mixer is the same minus `inject`).
pub struct HcW {
    /// group RMSNorm weight, (1+w) form, [4*hidden] f32
    pub norm: DeviceTensor,
    /// input_mix_weight_down [lowrank, 4*hidden] - with the `block_inject`
    /// rows APPENDED when `inject_rows > 0`. Both projections read the same
    /// normalized state, so one plane serves both and one launch covers them
    /// at batch 1. The inject was a [4, 10240] f32 matvec: 164 KB that cost
    /// 20.3 us because a 4-row grid is 4 blocks on a 148-SM card.
    pub down: DensePlane,
    /// rows of `down` that are the low-rank projection
    pub lowrank: usize,
    /// inject rows appended to `down`, or 0 when they were not folded (the
    /// 8-bit dense class, whose plane would QUANTIZE them, and the final
    /// mixer, which injects nothing)
    pub inject_rows: usize,
    /// input_mix_weight_up [4*hidden, lowrank]
    pub up: DensePlane,
    /// Row-permuted twin of `up` for the FUSED mix epilogue (slots 530-531):
    /// the up GEMM then emits the mixed output directly and the
    /// `[rows][hc*hidden]` gate plane is never materialised. `None` when the
    /// pack lacks the slots or the shape does not qualify.
    pub up_hcmix: Option<CudaSlice<u8>>,
    /// LOW-M ISLAND (slots 571/572): `down` padded to a multiple of 64 rows so
    /// the inject block can be read as its own gemm, and `up` in the gate
    /// epilogue's row order (branch s of hidden d at row d*hc+s), padded to
    /// `kpad`. Built only when the pack carries the island.
    pub down_p42: Option<CudaSlice<u8>>,
    pub up_p42: Option<CudaSlice<u8>>,
    /// block_inject_weight [4, 4*hidden] f32, when it was not folded in
    pub inject: Option<DeviceTensor>,
}

/// GDN (gated DeltaNet) mixer - split planes, sigmoid output gate.
pub struct GdnW {
    /// in_proj_qkv [10240, hidden] (rows: q 2048 | k 2048 | v 6144)
    pub qkv: DensePlane,
    /// in_proj_z [6144, hidden] (output gate; bypasses the conv)
    pub z: DensePlane,
    /// z|qkv fused at load (rows: z then qkv) for the ONE-launch 2-segment
    /// decode arm -- None unless PADDOCK_Q38FN_FUSE_ZQ armed the loader.
    pub zqkv: Option<QuantTensor>,
    /// in_proj_a and in_proj_b CONCATENATED: [2*v_heads, hidden] f32, rows
    /// [0, h) = a and [h, 2h) = b - which is exactly `delta_gate_ab`'s fused
    /// activation layout, so one matvec and one gate kernel replace two of
    /// each, at identical per-element math (deltanet/core.cuh:751).
    pub ab: DeviceTensor,
    /// bf16 twin of the folded a||b plane for the TGV lane (slot 547).
    pub ab16: Option<CudaSlice<u8>>,
    /// conv1d [10240, 4] f32 (checkpoint [10240,1,4] squeezed); silu act
    pub conv: DeviceTensor,
    /// RAW A_log [48] f32 - the g = -exp(A_log)*softplus(a+dt_bias) fold is
    /// the graph's call, kept un-baked so the kernel lane stays flexible
    pub a_log: DeviceTensor,
    /// `-exp(A_log)`, the form `pd_delta_gate` consumes (it computes
    /// `g = ssm_a * softplus(a + dt_bias)`). Derived at load beside the raw
    /// plane above; 48 floats.
    pub ssm_a: DeviceTensor,
    pub dt_bias: DeviceTensor,
    /// gated-norm weight [128] f32 (plain w, sigmoid gate - Not qwen35 silu)
    pub norm: DeviceTensor,
    /// out_proj [hidden, 6144]
    pub out: DensePlane,
    /// Value-head order of the planes above: `false` = the checkpoint's
    /// (HF) order, key head `vh / (hv/hk)` serves value head `vh`; `true` =
    /// the GGUF lane's tiled order (llama.cpp's converter), key head
    /// `vh % hk`. Selects the conv split kernel; no permutation of the key
    /// heads converts one into the other.
    pub tiled_heads: bool,
}

/// Gated attention + QSA indexer mixer.
pub struct AttnW {
    /// q_proj [12288, hidden] bf16 - PER-HEAD interleave: head h owns rows
    /// [h*512, (h+1)*512), first 256 = q, last 256 = the sigmoid output gate
    /// (vLLM qwen3_next._project_qkv_gate: view(heads, 512).chunk(2))
    pub q: DensePlane,
    /// k_proj / v_proj [512, hidden]
    pub k: DensePlane,
    pub v: DensePlane,
    /// q|k|v fused at load (rows: q then k then v) for the one-launch
    /// 3-segment decode arm (slot 424) -- None unless
    /// PADDOCK_Q38FN_FUSE_QKV armed the loader.
    pub qkv_f: Option<QuantTensor>,
    /// o_proj [hidden, 6144]
    pub o: DensePlane,
    /// per-head q/k RMSNorm (1+w) [256] f32
    pub q_norm: DeviceTensor,
    pub k_norm: DeviceTensor,
    /// indexer.index_qk_proj [640, hidden] bf16 (rows: q 4x128 | k 1x128)
    pub idx_qk: QuantTensor,
    /// indexer per-head RMSNorms [128] f32
    pub idx_q_norm: DeviceTensor,
    pub idx_k_norm: DeviceTensor,
}

pub enum MixerW {
    Gdn(GdnW),
    Attn(AttnW),
}

/// MoE block: NVFP4 routed experts (bytes as shipped) + bf16 shared expert.
pub struct MoeW {
    /// router gate with the shared expert's scalar gate APPENDED as row
    /// `n_expert`: [n_expert+1, hidden] f32. Both read the same block input;
    /// the shared gate alone was a ONE-BLOCK launch costing 6.7 us for 10 KB.
    pub router: DeviceTensor,
    /// bf16 twin of the router plane for the TGV lane (slot 547);
    /// built at load only when PADDOCK_Q38FN_TGV armed the loader.
    pub router16: Option<CudaSlice<u8>>,
    /// routed expert planes, 512 experts each, by residency class
    pub seats: ExpertSeats,
    /// shared expert [640, hidden] x2 + [hidden, 640]
    pub sh_gate: DensePlane,
    /// sh_gate|sh_up fused at load (rows: gate then up) - None unless
    /// PADDOCK_Q38FN_FUSE_SH armed the loader.
    pub sh_gu: Option<QuantTensor>,
    pub sh_up: DensePlane,
    pub sh_down: DensePlane,
}

/// PLE n-gram layer (decoder index 1): projections + the 51 GB fp8 table.
pub struct PleW {
    /// key_proj [4*hidden, hidden], value_proj [hidden, hidden]
    pub key: DensePlane,
    pub value: DensePlane,
    /// conv1d [4*hidden, 4] f32 (k=4, dilation 3 -> 9-token ring); silu
    pub conv: DeviceTensor,
    /// group norms (1+w) [4*hidden] f32
    pub norm_key: DeviceTensor,
    pub norm_query: DeviceTensor,
    pub norm_conv: DeviceTensor,
    /// the n-gram table: 128 shards concatenated, e4m3 bytes as shipped,
    /// [total_rows, 160]; None until `load_ple_table` runs (51 GB upload,
    /// kept separate so partial loads and the oracle stay cheap)
    pub table: Option<CudaSlice<u8>>,
    pub table_rows: usize,
    pub table_scale: f32,
    /// I64 hash constants, host-side - n-gram ids are computed on CPU before
    /// the forward starts (addresses depend only on token ids)
    pub multipliers: Vec<i64>,
    pub head_vocab: Vec<i64>,
    pub head_offset: Vec<i64>,
}

pub struct Qwen4ExpLayer {
    pub attn_hc: HcW,
    pub mlp_hc: HcW,
    pub mixer: MixerW,
    pub moe: MoeW,
    /// present only on PLE layers (decoder index 1); loaded separately from
    /// the 51 GB table via `load.rs` so partial loads stay cheap
    pub ple: Option<PleW>,
}
