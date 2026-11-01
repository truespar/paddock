//! NVIDIA Nemotron 3.5 Lightning 30B-A3B (`nemotron_h_moe`) - the first
//! safetensors-primary family: the NVFP4 checkpoint is served directly, no
//! GGUF anywhere in the lane.
//!
//! Architecture (pinned from the vLLM and llama.cpp study references plus
//! the checkpoint itself): 52
//! homogeneous single-residual blocks - each layer is one mixer,
//! `x = x + mixer(rms_norm(x))` - interleaving 23 mamba-2, 23 MoE and 6
//! attention layers (5, 12, 19, 26, 33, 42). NoPE attention (no rotary at
//! all), 32 Q / 2 KV heads at head_dim 128. Final RMS norm into an NVFP4
//! lm_head (untied).
//!
//! Weight residencies (byte-exact to the checkpoint everywhere):
//!   - experts + shared expert + lm_head: NVFP4 triples served packed
//!     (W4A16-class; the dequant kernel is oracle-only)
//!   - mamba in/out_proj: checkpoint e4m3 bytes as F8RowPlanes, the
//!     per-tensor weight_scale broadcast into the row-scale array
//!   - attention q/k/v/o: f32 planes (exact widening) for prefill + checkpoint
//!     bf16 twins for the decode GEMVs; embeddings: checkpoint bf16 (the
//!     gather widens in-kernel); routers, norms, conv, A/D/dt_bias:
//!     bf16/f32 -> f32 (exact widening)

pub(crate) mod batch;
mod chunked;
pub(crate) mod dflash;
mod forward;
mod load;
mod mm;
pub use mm::PictureProbe;
pub(crate) mod mtp;
mod prefix;
mod sampled;
mod spec;
pub(crate) mod ssm_arena;
pub(crate) mod vision;
mod walk;

use std::sync::Arc;

use cudarc::driver::CudaSlice;

use crate::gpu::{
    DeviceTensor, F8RowPlane, GpuExecutor, KvDtype, Nvf4MoePlane, Nvf4Plane, QuantTensor, QuantW,
    RepackedQ8,
};
use paddock_models::nemotron::NemotronConfig;

pub(crate) use forward::{DecodeState, PipeState, PrefillScratch, Scratch, SendGraph};

/// One linear plane in either weight class: the NVFP4 checkpoint's fp8
/// e4m3 rows (W8A8 prefill / W8A16 decode), or a GGUF quant plane (Q8_0
/// int8 dp4a class; k-quant loads too but the MoE planes refuse non-Q8_0
/// upstream, so a UD file errors at load rather than serving half-classed).
pub(crate) enum LinW {
    F8(F8RowPlane),
    Qw(QuantW),
}

/// Mamba-2 mixer weights. `in_proj` rows are `[z | x B C | dt]` (d_inner +
/// conv_dim + n_heads = 10304); the forward slices that output by offset
/// rather than splitting.
pub(crate) struct MambaWeights {
    pub in_proj: LinW,
    pub out_proj: LinW,
    /// conv weight [conv_dim, k] (checkpoint [conv_dim, 1, k] squeezed)
    pub conv_w: CudaSlice<f32>,
    pub conv_b: CudaSlice<f32>,
    /// -exp(A_log), per head (the load-time transform vLLM also applies)
    pub a: CudaSlice<f32>,
    pub d: CudaSlice<f32>,
    pub dt_bias: CudaSlice<f32>,
    /// gated-norm weight [d_inner]
    pub norm_w: CudaSlice<f32>,
}

/// NoPE attention mixer weights, in either weight class.
///
/// `F32` is the NVFP4 lane: f32 planes (bf16 widened exactly) carry prefill
/// byte-identical to the original lane; `bf16` holds the checkpoint bytes
/// for the decode GEMVs (half the DRAM per tick), None when the pack lacks
/// the lane or PADDOCK_NO_NEMO_BF16 pins the f32 baseline.
///
/// `Qw` is the GGUF lane: one quant plane per projection (Q8_0 - decode
/// rides the repacked GEMV, prefill the mmq int8 ladder).
pub(crate) enum AttnWeights {
    F32 {
        wq: DeviceTensor,
        wk: DeviceTensor,
        wv: DeviceTensor,
        wo: DeviceTensor,
        bf16: Option<AttnBf16>,
    },
    Qw {
        wq: QuantW,
        wk: QuantW,
        wv: QuantW,
        wo: QuantW,
    },
    /// Super's ModelOpt FP8 projections: the checkpoint's e4m3 rows with
    /// the per-tensor weight scale broadcast to every row - the mamba
    /// in/out_proj residency (W8A8 prefill, W8A16 decode).
    F8 {
        wq: F8RowPlane,
        wk: F8RowPlane,
        wv: F8RowPlane,
        wo: F8RowPlane,
    },
}

/// Checkpoint-byte twins of the attention planes ([in, out] dims, the bf16
/// lane's convention). q/k/v live as one load-time-concatenated `[q;k;v]`
/// plane (thin-k/v rung): the 256-row k/v planes are
/// latency-starved on their own decode-band grids (~40 us floor at 1-3% of
/// the DRAM roof, regardless of kernel shape - probed in
/// bf16_thin_probe.cu), so the batched tick computes all three in one
/// launch riding the fused grid; the serial row reads per-projection
/// segments of the same bytes (row-major planes make a row range a byte
/// range). Same total VRAM as three separate planes.
pub(crate) struct AttnBf16 {
    /// fused plane, dims `[hidden, q_dim + 2*kv_dim]`; rows are q, then k,
    /// then v
    pub wqkv: QuantTensor,
    pub q_dim: usize,
    pub kv_dim: usize,
    pub wo: QuantTensor,
}

/// The batched decode tick's q/k/v projections over the bf16 twins - one
/// fused launch when the pack carries slot 424, three per-segment launches
/// otherwise (and under PADDOCK_NVQKV=0, the A/B pin).
///
/// The fused arm runs at all decode batches >= 2: the probe had it at 20.8
/// us (b8) / 34.6 us (b32) for the whole q|k|v read vs ~93/131 us for the
/// three separate launches - the thin k/v planes are latency-starved alone
/// at every batch. At rows 2..=8 this moves those rows from the multi-row
/// GEMV's f32-product class onto the mma's bf16-activation-cast class (the
/// same class rows > 8 always used);
/// the serve battery gates it like every twin-class election. rows == 1
/// stays on the exact per-segment GEMV.
pub(crate) fn attn_qkv_batch(
    exec: &GpuExecutor,
    b: &AttnBf16,
    x: &CudaSlice<f32>,
    yq: &mut CudaSlice<f32>,
    yk: &mut CudaSlice<f32>,
    yv: &mut CudaSlice<f32>,
    rows: usize,
) -> Result<(), crate::gpu::GpuError> {
    static QKV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let on = *QKV.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_NVQKV")
            .map(|v| v != "0")
            .unwrap_or(true)
    });
    if on && rows > 1 && exec.has_bf16_qkv_gemm() {
        return exec.bf16_qkv_gemm(&b.wqkv, x, yq, yk, yv, b.q_dim, b.kv_dim, rows);
    }
    exec.bf16_gemm_rows(&b.wqkv, 0, b.q_dim, x, yq, rows)?;
    exec.bf16_gemm_rows(&b.wqkv, b.q_dim, b.kv_dim, x, yk, rows)?;
    exec.bf16_gemm_rows(&b.wqkv, b.q_dim + b.kv_dim, b.kv_dim, x, yv, rows)
}

/// Embedding table residency: bf16 keeps the checkpoint bytes (the gather
/// widens in-kernel - bit-identical rows at half the VRAM); f32 is the widened
/// fallback for older packs and the kill switch; Q8 is the GGUF lane's raw
/// Q8_0 table (granite's resident gather path - never repacked).
pub(crate) enum TokEmbd {
    F32(CudaSlice<f32>),
    Bf16(QuantTensor),
    Q8(QuantTensor),
}

/// MoE mixer weights: sigmoid router with a selection-only correction bias,
/// top-6 of 128 renormalized then x2.5 on the routed output only; experts
/// are squared-relu with no gate matrix; one always-on unscaled shared
/// expert in parallel.
pub(crate) struct MoeWeights {
    pub router: DeviceTensor,
    pub bias: DeviceTensor,
    pub planes: MoePlanes,
    /// LatentMoE (Super): the routed experts' planes are latent-wide and
    /// these carry the hidden across ([`latent_moe`]); None on Lightning.
    pub latent: Option<LatentProj>,
}

/// LatentMoE's two bf16 projections, as the checkpoint ships them: `fc1`
/// hidden -> latent before the routed experts, `fc2` latent -> hidden after
/// their weighted sum. [in, out] dims.
pub(crate) struct LatentProj {
    pub fc1: QuantTensor,
    pub fc2: QuantTensor,
}

/// Scratch the LatentMoE path adds: the latent copy of the normed hidden
/// the routed experts read and their weighted sum (`rows` rows), and the
/// exact class's f32 up activations of the routed picks and the shared
/// expert, sized for the `exact_rows` it may run, plus its one-row partial
/// plane (`part1`, the tiled planes' r=1 twins fold through it). The sorted
/// W4A4 pair and the W16 class keep their activations in the lane's own
/// planes.
pub(crate) struct LatentScratch {
    pub lat_in: CudaSlice<f32>,
    pub lat: CudaSlice<f32>,
    pub up: CudaSlice<f32>,
    pub sh_up: CudaSlice<f32>,
    pub part1: CudaSlice<f32>,
    pub exact_rows: usize,
}

impl LatentScratch {
    pub(crate) fn new(
        exec: &GpuExecutor,
        hp: &NemotronConfig,
        rows: usize,
        exact_rows: usize,
    ) -> Result<Option<Self>, crate::gpu::GpuError> {
        let Some(l) = hp.latent else {
            return Ok(None);
        };
        let exact_rows = exact_rows.clamp(1, rows.max(1));
        Ok(Some(Self {
            lat_in: exec.alloc(rows * l)?,
            lat: exec.alloc(rows * l)?,
            up: exec.alloc(exact_rows * hp.n_active * hp.moe_ff)?,
            sh_up: exec.alloc(exact_rows * hp.shared_ff)?,
            part1: exec.alloc(((hp.n_active + 1) * l).max(2 * hp.hidden))?,
            exact_rows,
        }))
    }
}

/// Whether the latent mixer's prefill rows can ride the sorted W4A4 pair:
/// the block-scale MMA chain plus the write-out fold. `PADDOCK_NO_NVF4_MOE_BS`
/// (the Lightning lane's A/B switch) pins them to the exact class too.
pub(crate) fn latent_mma_on(exec: &GpuExecutor) -> bool {
    exec.has_nvf4_moe_bs()
        && exec.has_moe_slot_combine_init()
        && paddock_models::dev_var_os!("PADDOCK_NO_NVF4_MOE_BS").is_none()
}

/// The lane's sorted-MoE buffers a latent mixer borrows for the W4A4 pair:
/// the e2m1 activation image (the latent for the routed pass, then the
/// hidden for the shared expert - one after the other), the two CSR sorts
/// and their block capacities, the e2m1 up activations, and the
/// pre-weighted partial plane (routed `[rows][k][latent]`, then the shared
/// expert's `[rows][hidden]`).
pub(crate) struct LatentMma<'a> {
    pub xq4: &'a mut CudaSlice<i8>,
    pub xs4: &'a mut CudaSlice<u8>,
    pub srow: &'a mut CudaSlice<u32>,
    pub sslot: &'a mut CudaSlice<u32>,
    pub bexp: &'a mut CudaSlice<u32>,
    pub nb_r: usize,
    pub srow_s: &'a mut CudaSlice<u32>,
    pub sslot_s: &'a mut CudaSlice<u32>,
    pub bexp_s: &'a mut CudaSlice<u32>,
    pub nb_s: usize,
    pub fq: &'a mut CudaSlice<u8>,
    pub fs: &'a mut CudaSlice<u8>,
    pub fq_s: &'a mut CudaSlice<u8>,
    pub fs_s: &'a mut CudaSlice<u8>,
    pub part: &'a mut CudaSlice<f32>,
}

/// The batch lane's W16-class buffers a latent mixer borrows for its decode
/// rows: the bf16 activation rows (the latent, then the hidden), the bf16 up
/// activations, the two CSR sorts, and the partial plane (routed
/// `[rows][k + 1][latent]`, then the shared expert's `[rows][2][hidden]`).
pub(crate) struct LatentW16<'a> {
    pub x16: &'a mut CudaSlice<half::bf16>,
    pub act16: &'a mut CudaSlice<half::bf16>,
    pub srow: &'a mut CudaSlice<u32>,
    pub sslot: &'a mut CudaSlice<u32>,
    pub bexp: &'a mut CudaSlice<u32>,
    pub nb_r: usize,
    pub srow_s: &'a mut CudaSlice<u32>,
    pub sslot_s: &'a mut CudaSlice<u32>,
    pub bexp_s: &'a mut CudaSlice<u32>,
    pub nb_s: usize,
    pub part: &'a mut CudaSlice<f32>,
}

/// Which class a latent mixer call runs (see [`latent_moe`]).
pub(crate) enum LatentClass<'a> {
    /// prompt rows: the sorted W4A4 pair
    Mma(LatentMma<'a>),
    /// the batch lane's decode / verify rows on the tiled planes: W16
    W16(LatentW16<'a>),
    /// the exact row class: f32 activations, one GEMV per row and pick on
    /// row-major planes; the r=1 tiled twins on tiled ones
    Exact,
}

/// Whether the latent planes ride the tiled layout (and so its classes):
/// the whole tiled consumer set plus the W16 pair, which serves the batch
/// lane's decode rows on it.
pub(crate) fn latent_tiled(w: &MoeWeights) -> bool {
    matches!(&w.planes, MoePlanes::Nvf4 { up, .. } if up.layout == crate::gpu::Nvf4MoeLayout::Tiled64)
}

/// One LatentMoE mixer over `rows` normed rows `xn`, into `out` (written,
/// the caller adds it to the residual). Routing is already in `idx` / `w`
/// (picks + weights over the FULL hidden, routed scale folded in). The
/// reference (vLLM's `NemotronHMoE`): routed = fc2(sum_k w_k * expert_k(
/// fc1(xn))), shared = shared_expert(xn), out = routed + shared.
///
/// Three classes, one graph:
/// - `Mma`, prompt rows: the sorted W4A4 pair - activations to e2m1 with
///   dynamic e4m3 block scales, the checkpoint's own NVFP4 recipe and the
///   reference's serving class - streaming each touched expert once per pass
///   (the row-major bs pair, or its bit-exact tiled twin).
/// - `W16`, the batch lane's decode and verify rows on tiled planes: bf16
///   activations on tensor cores, sorted blocks (each touched expert read
///   once per tick), batch-invariant - a verify row is the decode step it
///   stands for.
/// - `Exact`: f32 activations against the dequantized planes - per-row GEMVs
///   on row-major planes, the r=1 twins on tiled ones (the serial lane, the
///   drafter's one-row passes).
///
/// The shared expert cannot fold into the routed launch in any class (its
/// input is the hidden, the routed experts' the latent): it runs as its own
/// one-expert pass.
#[allow(clippy::too_many_arguments)]
pub(crate) fn latent_moe(
    exec: &GpuExecutor,
    hp: &NemotronConfig,
    w: &MoeWeights,
    lp: &LatentProj,
    xn: &CudaSlice<f32>,
    ls: &mut LatentScratch,
    idx: &CudaSlice<u32>,
    wts: &CudaSlice<f32>,
    sh_idx: &CudaSlice<u32>,
    sh_w: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    rows: usize,
    class: LatentClass<'_>,
) -> Result<(), crate::gpu::GpuError> {
    let MoePlanes::Nvf4 {
        up,
        down,
        sh_up: sh_up_pl,
        sh_down,
    } = &w.planes
    else {
        return Err(crate::gpu::GpuError::Unsupported(
            "LatentMoE serves NVFP4 expert planes only".into(),
        ));
    };
    let tiled = up.layout == crate::gpu::Nvf4MoeLayout::Tiled64;
    let (hidden, latent, k) = (hp.hidden, hp.expert_in(), hp.n_active);
    // the bf16 planes against f32 rows: the GEMV at one row, the multi-row
    // band to 8, the tile past it - every product exact either way
    let lin = |w: &QuantTensor,
               x: &CudaSlice<f32>,
               y: &mut CudaSlice<f32>|
     -> Result<(), crate::gpu::GpuError> { exec.bf16_gemm(w, None, x, y, rows) };
    lin(&lp.fc1, xn, &mut ls.lat_in)?;
    match class {
        LatentClass::Mma(m) => {
            // routed: latent rows -> e2m1, sorted into 32-row expert blocks,
            // the pair writes each pick pre-weighted into its slot, one
            // fixed-order fold into the latent sum
            exec.quantize_nvf4(&ls.lat_in, m.xq4, m.xs4, rows * latent)?;
            let nbr = batch::moe_live_blocks(rows, k, hp.n_expert, m.nb_r);
            exec.moe_align(idx, m.srow, m.sslot, m.bexp, rows, k, hp.n_expert, nbr)?;
            if tiled {
                exec.nvf4_moe_up_relu2_st(up, m.srow, m.bexp, m.xq4, m.xs4, m.fq, m.fs, nbr, 32)?;
                exec.nvf4_moe_down_st(
                    down,
                    m.srow,
                    m.sslot,
                    m.bexp,
                    Some(wts),
                    m.fq,
                    m.fs,
                    m.part,
                    k,
                    k,
                    0,
                    nbr,
                    32,
                )?;
            } else {
                exec.nvf4_moe_up_relu2_bs(up, m.srow, m.bexp, m.xq4, m.xs4, m.fq, m.fs, nbr)?;
                exec.nvf4_moe_down_bs(
                    down,
                    m.srow,
                    m.sslot,
                    m.bexp,
                    Some(wts),
                    m.fq,
                    m.fs,
                    m.part,
                    k,
                    k,
                    0,
                    nbr,
                )?;
            }
            exec.moe_slot_combine_init(m.part, &mut ls.lat, latent, k, rows)?;
            lin(&lp.fc2, &ls.lat, out)?;
            // shared: the hidden rows, one identity block list, weight 1,
            // folded onto the routed output
            exec.quantize_nvf4(xn, m.xq4, m.xs4, rows * hidden)?;
            let nbs = batch::moe_live_blocks(rows, 1, 1, m.nb_s);
            exec.moe_align(sh_idx, m.srow_s, m.sslot_s, m.bexp_s, rows, 1, 1, nbs)?;
            if tiled {
                exec.nvf4_moe_up_relu2_st(
                    sh_up_pl, m.srow_s, m.bexp_s, m.xq4, m.xs4, m.fq_s, m.fs_s, nbs, 32,
                )?;
                exec.nvf4_moe_down_st(
                    sh_down, m.srow_s, m.sslot_s, m.bexp_s, None, m.fq_s, m.fs_s, m.part, 1, 1, 0,
                    nbs, 32,
                )?;
            } else {
                exec.nvf4_moe_up_relu2_bs(
                    sh_up_pl, m.srow_s, m.bexp_s, m.xq4, m.xs4, m.fq_s, m.fs_s, nbs,
                )?;
                exec.nvf4_moe_down_bs(
                    sh_down, m.srow_s, m.sslot_s, m.bexp_s, None, m.fq_s, m.fs_s, m.part, 1, 1, 0,
                    nbs,
                )?;
            }
            exec.moe_slot_combine(m.part, out, hidden, 1, rows)
        }
        LatentClass::W16(m) => {
            if !tiled {
                return Err(crate::gpu::GpuError::Unsupported(
                    "the W16 class reads the tiled expert planes".into(),
                ));
            }
            // routed: the latent rows as bf16, sorted into 32-row blocks, the
            // pair's partials [rows][k + 1][latent] (slot k a zero), folded
            exec.convert_f32_bf16(&ls.lat_in, m.x16, rows * latent)?;
            let nbr = batch::moe_live_blocks(rows, k, hp.n_expert, m.nb_r);
            exec.moe_align(idx, m.srow, m.sslot, m.bexp, rows, k, hp.n_expert, nbr)?;
            exec.nvf4_moe_up_relu2_w16_solo(
                up, m.srow, m.sslot, m.bexp, m.x16, m.act16, k, nbr, rows,
            )?;
            exec.nvf4_moe_down_part_w16_solo(
                down, m.srow, m.sslot, m.bexp, wts, m.act16, m.part, k, nbr, rows,
            )?;
            exec.moe_slot_combine_init(m.part, &mut ls.lat, latent, k + 1, rows)?;
            lin(&lp.fc2, &ls.lat, out)?;
            // shared: one routed expert over every row (identity sort, weight
            // 1), partials [rows][2][hidden], folded onto the routed output
            exec.convert_f32_bf16(xn, m.x16, rows * hidden)?;
            let nbs = batch::moe_live_blocks(rows, 1, 1, m.nb_s);
            exec.moe_align(sh_idx, m.srow_s, m.sslot_s, m.bexp_s, rows, 1, 1, nbs)?;
            exec.nvf4_moe_up_relu2_w16_solo(
                sh_up_pl, m.srow_s, m.sslot_s, m.bexp_s, m.x16, m.act16, 1, nbs, rows,
            )?;
            exec.nvf4_moe_down_part_w16_solo(
                sh_down, m.srow_s, m.sslot_s, m.bexp_s, sh_w, m.act16, m.part, 1, nbs, rows,
            )?;
            exec.moe_slot_combine(m.part, out, hidden, 2, rows)
        }
        LatentClass::Exact if tiled => {
            if rows != 1 {
                return Err(crate::gpu::GpuError::Unsupported(format!(
                    "LatentMoE's tiled exact class is the one-row twins, asked {rows}"
                )));
            }
            exec.nvf4_moe_up_relu2_mtt_solo(up, idx, &ls.lat_in, &mut ls.up, k)?;
            exec.nvf4_moe_down_part_tt_solo(down, idx, wts, &ls.up, &mut ls.part1, k)?;
            exec.moe_slot_combine_init(&ls.part1, &mut ls.lat, latent, k + 1, 1)?;
            lin(&lp.fc2, &ls.lat, out)?;
            exec.nvf4_moe_up_relu2_mtt_solo(sh_up_pl, sh_idx, xn, &mut ls.sh_up, 1)?;
            exec.nvf4_moe_down_part_tt_solo(sh_down, sh_idx, sh_w, &ls.sh_up, &mut ls.part1, 1)?;
            exec.moe_slot_combine(&ls.part1, out, hidden, 2, 1)
        }
        LatentClass::Exact => {
            if rows > ls.exact_rows {
                return Err(crate::gpu::GpuError::Unsupported(format!(
                    "LatentMoE exact class sized for {} rows, asked {rows}",
                    ls.exact_rows
                )));
            }
            exec.nvf4_moe_up_relu2(up, idx, &ls.lat_in, &mut ls.up, k, rows)?;
            exec.nvf4_moe_down_acc(down, idx, wts, &ls.up, &mut ls.lat, None, k, rows, false)?;
            lin(&lp.fc2, &ls.lat, out)?;
            exec.nvf4_moe_up_relu2(sh_up_pl, sh_idx, xn, &mut ls.sh_up, 1, rows)?;
            exec.nvf4_moe_down_acc(sh_down, sh_idx, sh_w, &ls.sh_up, out, None, 1, rows, true)
        }
    }
}

/// Expert plane residency: NVFP4 triples (W4A16), or repacked Q8_0 (W8A8
/// dp4a - the GGUF lane; up/down are the flat [e*ff + o] expert streams,
/// sh_* the 1-expert shared planes the relu2 kernels address with a zero
/// idx).
// one instance per MoE layer, resident for the model's life - the variant size
// gap is noise next to the device planes it names
#[allow(clippy::large_enum_variant)]
pub(crate) enum MoePlanes {
    Nvf4 {
        up: Nvf4MoePlane,
        down: Nvf4MoePlane,
        sh_up: Nvf4MoePlane,
        sh_down: Nvf4MoePlane,
    },
    Q8 {
        up: RepackedQ8,
        down: RepackedQ8,
        sh_up: RepackedQ8,
        sh_down: RepackedQ8,
    },
}

/// lm_head residency (untied): NVFP4 plane, a GGUF quant plane, or Super's
/// ModelOpt FP8 rows.
pub(crate) enum HeadW {
    Nvf4(Nvf4Plane),
    Qw(QuantW),
    F8(F8RowPlane),
}

/// Batched NVFP4 lm_head election, shared by the decode
/// tick and the spec verify pass. Wide planes from two rows up take the
/// tensor-core class - exact-dequant bf16 weights on m16n8k16 mma, where
/// the only numeric change vs the scalar lane is the f32->bf16 activation
/// cast (the same cast vLLM's bf16 serving applies before its lm_head) -
/// because the scalar classes lose everywhere batch exists: at the vocab
/// shape the probe measures tc 253-291 us vs mr 577 (b8) / 1155 (b32).
/// rows==1 keeps the per-row exact-f32 gemv (138 us at 91% of roof - tc's
/// staged ring only reaches ~50% at b1), so single-stream serving keeps
/// both its speed and its exact class. The boundary does not cost
/// composition-determinism: a probed fact is that concurrent greedy serving
/// diverges across identical reruns with the scalar head too (10-12/16 c16
/// outputs, PADDOCK_NVF4_TC=0) - batch invariance at concurrency was never a
/// live property of the graph.
/// PADDOCK_NVF4_TC=0 is the A/B kill switch. The election lives here, not
/// in the generic wrapper, so every other nvf4_gemv_batch consumer stays in
/// the exact class.
///
/// `w16` = the rows run the W16 decode class (batch::w16_class): the tile
/// serves one row too, so a row's logits do not depend on the row count.
pub(crate) fn head_nvf4_batch(
    exec: &GpuExecutor,
    h: &Nvf4Plane,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    rows: usize,
    w16: bool,
) -> Result<(), crate::gpu::GpuError> {
    static TC: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let on = *TC.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_NVF4_TC")
            .map(|v| v != "0")
            .unwrap_or(true)
    });
    if on
        && (rows > 1 || w16)
        && h.out_dim >= 4096
        && h.in_dim.is_multiple_of(16)
        && exec.has_nvf4_gemm_tc()
    {
        exec.nvf4_gemm_tc(h, x, y, None, rows)
    } else {
        exec.nvf4_gemv_batch(h, x, y, None, rows)
    }
}

/// The FP8 lm_head (Super) over `rows` rows of `x`: the exact W8A16 GEMV
/// for one row, the W8A8 GEMM (per-row e4m3 activations, like every FP8
/// projection here) past it. `xq`/`xrs` are the activation image's scratch.
#[allow(clippy::too_many_arguments)]
pub(crate) fn head_f8_batch(
    exec: &GpuExecutor,
    h: &F8RowPlane,
    x: &CudaSlice<f32>,
    xq: &mut CudaSlice<i8>,
    xrs: &mut CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    hidden: usize,
    vocab: usize,
    rows: usize,
) -> Result<(), crate::gpu::GpuError> {
    if rows == 1 {
        return exec.f8r_gemv(h, x, y, hidden, vocab);
    }
    exec.quantize_e4m3_row(x, xq, xrs, hidden, rows)?;
    exec.f8row_gemm(h, xq, xrs, y, hidden, vocab, rows)
}

// one per layer; the Mamba/Attn/Moe weight bundles differ in size by design
#[allow(clippy::large_enum_variant)]
pub(crate) enum Mixer {
    Mamba(MambaWeights),
    Attn(AttnWeights),
    Moe(MoeWeights),
}

pub(crate) struct NemotronLayer {
    /// pre-mixer RMS norm weight [hidden]
    pub norm: DeviceTensor,
    pub mixer: Mixer,
}

pub struct GpuNemotron {
    pub(crate) exec: Arc<GpuExecutor>,
    pub(crate) hp: NemotronConfig,
    pub(crate) layers: Vec<NemotronLayer>,
    /// embedding table [vocab, hidden] (input lookup only)
    pub(crate) tok_embd: TokEmbd,
    pub(crate) final_norm: DeviceTensor,
    pub(crate) lm_head: HeadW,
    pub(crate) kv_dtype: KvDtype,
    /// Storage class for the recurrent SSM state. f16 is the elected default
    /// (NLL-gated quality-neutral); PADDOCK_SSM_DTYPE=f32 (dev builds only) selects the
    /// checkpoint-conservative reference class (arithmetic is f32 in both).
    pub(crate) ssm_dtype: ssm_arena::SsmDtype,
    /// Bulk-prefill chunk width = the row-scratch cap and the mixed tick's
    /// prompt-row clamp; die-keyed with an env override, fixed at load
    /// (`forward::prefill_chunk_rows`).
    pub(crate) prefill_chunk: usize,
    pub(crate) max_ctx: usize,
    pub(crate) weights_bytes: u64,
    /// Content identity of the loaded weights and tokenizer, captured at
    /// load - the cache namespace's answer to "are these the same bytes?".
    /// Geometry alone stopped being a sufficient key when the tier gained a
    /// store that survives restarts (see `kv_tier::fingerprint`).
    pub(crate) content_id: ([u8; 32], [u8; 32]),
    pub(crate) decode: Option<DecodeState>,
    pub(crate) scratch: Option<Scratch>,
    pub(crate) prefill: Option<PrefillScratch>,
    /// in-flight serial decode pipe  - always None between requests
    pub(crate) pipe: Option<PipeState>,
    /// Continuous-batching state (batch.rs): paged KV on the 6
    /// attention layers + per-slot mamba arenas. None until `enable_batch`
    /// succeeds - the serial lane above keeps working without it.
    pub(crate) batch: Option<batch::NemoBatch>,
    /// Prompts mid-prefill, FIFO (batch.rs, stall-free batching).
    pub(crate) chunked: Vec<batch::ChunkedPrefill>,
    /// Per-slot rows the radix served this prefill (telemetry via
    /// `take_prefill_reused`); sized at enable, empty on the serial lane.
    pub(crate) last_reused: Vec<usize>,
    /// In-flight batched depth-2 decode pipe (stage E) - always None
    /// between pipe sessions.
    pub(crate) pipe_b: Option<batch::PipeB>,
    /// DFlash drafter (C2) - attached via --mtp, None otherwise.
    pub(crate) dflash: Option<dflash::DflashDrafter>,
    /// the reasoning stop mask's lane (sampled.rs)
    pub(crate) stop: crate::gpu::StopMaskLane,
    /// In-file MTP drafter (C3) - the GGUF's blk.52 nextn block,
    /// loaded whenever the file carries it and spec isn't killed
    /// (PADDOCK_NO_SPEC); an explicit DFlash attach evicts it.
    pub(crate) mtp: Option<mtp::MtpDrafter>,
    /// Super VL's picture tower + projector (`vision.rs`); None on a
    /// text-only checkpoint.
    pub(crate) vision: Option<vision::SuperVision>,
    /// The picture rows the current prefill pass carries (`mm.rs`) - set
    /// around a picture prompt's passes only.
    pub(crate) mm: Option<mm::MmOverlay>,
}
