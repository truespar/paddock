//! The NVFP4 safetensors build's MoE on the laguna body (Kolibri's
//! `nvfp4-pack-quantized` export): routed experts NVFP4, the ungated shared
//! expert BF16. The arms are qwen4exp's SwiGLU NVFP4 seat's:
//!
//! - decode (and spec-verify rounds up to 8 rows): W4A16 - the warp-per-row
//!   gate|up + SwiGLU GEMV off f32 rows, then the down GEMV, z-split into
//!   per-pair partials folded in fixed order (`moe_slot_combine_init`);
//! - prefill (and wider verify rounds): W4A4 on the block-scaled tensor cores
//!   (`kind::mxf4nvf4`) - the rows quantize to nvfp4 (amax / 6 per 16, no
//!   global scale: the checkpoint's `input_global_scale` is not read, as in
//!   every NVFP4 lane here), moe_align sorts the (token, slot) pairs into
//!   64-row expert blocks, the sorted gate|up ring (slot 830) re-quantizes
//!   silu(g) * u to nvfp4, the sorted down ring (slot 831) lands per-pair
//!   partials (bf16), and the fold adds the routed sum onto the shared
//!   expert's output. Every op is per token, so the class keys on MODE, never
//!   on rows - a warm-resume tail rides the same pair. The ring pair lands
//!   the bytes the 32-row pair (slots 631 / 758) lands; a pack without it,
//!   a shape outside its laws, or PADDOCK_NO_NV4M_MS=1 keeps that pair.
//!
//! Router: `MoeDims::route` (Kolibri's sigmoid_logit_add, slot 748) over the
//! f32-widened BF16 router. The shared expert's three BF16 planes ride the
//! bf16 GEMV / mma ladder (the prefill pair at prefill widths) and add on top
//! of the routed sum.

use crate::gpu::GpuExecutor;
use crate::gpu_model::gpt_oss::GpuModelError;

use super::batch::BatchScratch;
use super::{MoeDims, MoeNv};

/// Whether the ring pair (slots 830 / 831) takes this layer's experts: the
/// pack carries it, the planes meet its shape laws (gate|up in_dim % 128,
/// down K = expert ff % 256 and <= 1024), and PADDOCK_NO_NV4M_MS is unset.
fn ring_pair(exec: &GpuExecutor, w: &MoeNv) -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let off = *OFF.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_NV4M_MS").is_some());
    !off && exec.has_nvf4_moe_ms()
        && w.gate.in_dim.is_multiple_of(128)
        && w.down.in_dim.is_multiple_of(256)
        && w.down.in_dim <= 1024
}

/// Sorted blocks for `pairs` (token, slot) pairs over `n_expert` experts at
/// block `bm` - every expert can pad its last block.
pub(super) fn align_blocks(pairs: usize, n_expert: usize, bm: usize) -> usize {
    (pairs + n_expert * (bm - 1)).div_ceil(bm).min(pairs)
}

/// The MoE over `r` rows of `sc.xn` into `sc.proj` (routed + shared).
/// `sorted`: the W4A4 sorted pair (prefill mode, wide verify rounds); else
/// the W4A16 GEMVs. `x16_ready`: `sc.x16` already holds xn as bf16 (the
/// fused sandwich norm landed it), so the shared expert's prefill pair skips
/// its narrowing.
#[allow(clippy::too_many_arguments)]
pub(super) fn moe_nv_rows(
    exec: &GpuExecutor,
    m: &MoeDims,
    w: &MoeNv,
    sc: &mut BatchScratch,
    embd: usize,
    r: usize,
    sorted: bool,
    x16_ready: bool,
) -> Result<(), GpuModelError> {
    let k = m.n_active;
    // router: at decode widths its BF16 plane through the f32-activation
    // GEMVs (exact products, f32 sums - the widened plane's class, half the
    // bytes); the tile's bf16-activation class never routes
    if r <= 8 {
        exec.bf16_gemm(&w.router_b16, None, &sc.xn, &mut sc.moe_logits, r)?;
    } else {
        exec.matvec_f32_batch(&w.router_w, &sc.xn, &mut sc.moe_logits, r)?;
    }
    m.route(
        exec,
        &sc.moe_logits,
        &w.probs_bias,
        &mut sc.moe_idx,
        &mut sc.moe_w,
        r,
    )?;
    let sorted = sorted && exec.has_nvf4_moe_gu_swiglu_bs() && exec.has_nvf4_moe_bs();
    // bf16 partials (slot 758) at the sorted widths
    let b16 = sorted && exec.has_nvf4_moe_down_bs_b16() && exec.has_moe_slot_combine_bf16();
    // the ring pair (slots 830 / 831) over 64-row blocks
    let ms = b16 && ring_pair(exec, w);
    // Where the routed fold can ADD onto proj (the bf16 fold; the z-split
    // decode fold) the shared expert lands there first: shared + routed is
    // the routed + shared order bit for bit (two operands commute), and the
    // separate add is gone.
    let shared_first = b16 || (!sorted && r <= 64);
    if shared_first {
        shared_expert(exec, m, w, sc, embd, r, true, x16_ready)?;
    }
    if sorted {
        exec.quantize_nvf4(&sc.xn, &mut sc.xq4, &mut sc.xs4, r * embd)?;
        let nb = align_blocks(r * k, m.n_expert, if ms { 64 } else { 32 });
        if ms {
            exec.moe_align_bm(
                &sc.moe_idx,
                &mut sc.srow,
                &mut sc.sslot,
                &mut sc.bexp,
                r,
                k,
                m.n_expert,
                64,
                nb,
            )?;
        } else {
            exec.moe_align_at(
                &sc.moe_idx,
                0,
                &mut sc.srow,
                &mut sc.sslot,
                &mut sc.bexp,
                r,
                k,
                m.n_expert,
                nb,
            )?;
        }
        let gu = if ms {
            GpuExecutor::nvf4_moe_gu_swiglu_ms
        } else {
            GpuExecutor::nvf4_moe_gu_swiglu_bs
        };
        gu(
            exec,
            &w.gate,
            &w.up,
            &sc.srow,
            &sc.bexp,
            &sc.xq4,
            &sc.xs4,
            &mut sc.nfq,
            &mut sc.nfs,
            nb,
            0,
        )?;
        // np = k: every pair slot is written (qwen4exp's k + 1 carries a
        // shared pseudo-slot this lane does not have)
        let down = if ms {
            GpuExecutor::nvf4_moe_down_ms_b16_at
        } else if b16 {
            GpuExecutor::nvf4_moe_down_bs_b16_at
        } else {
            GpuExecutor::nvf4_moe_down_bs_at
        };
        down(
            exec,
            &w.down,
            &sc.srow,
            &sc.sslot,
            &sc.bexp,
            Some(&sc.moe_w),
            &sc.nfq,
            &sc.nfs,
            &mut sc.moe_part,
            k,
            k,
            0,
            nb,
            0,
        )?;
        if b16 {
            exec.moe_slot_combine_bf16(&sc.moe_part, &mut sc.proj, embd, k, r)?;
        } else {
            exec.moe_slot_combine_init_at(&sc.moe_part, &mut sc.proj, 0, embd, k, r)?;
        }
    } else {
        exec.q4x_moe_gu_swiglu(&w.gate, &w.up, &sc.moe_idx, &sc.xn, &mut sc.moe_fused, k, r)?;
        // z-split + the fixed-order fold at decode widths (the warp-per-row
        // down is CTA-starved there), added onto the shared expert; straight
        // above
        if r <= 64 {
            exec.nvf4_moe_down_acc(
                &w.down,
                &sc.moe_idx,
                &sc.moe_w,
                &sc.moe_fused,
                &mut sc.proj,
                Some(&mut sc.moe_part),
                k,
                r,
                false,
            )?;
            exec.moe_slot_combine(&sc.moe_part, &mut sc.proj, embd, k.div_ceil(2), r)?;
        } else {
            exec.nvf4_moe_down_acc(
                &w.down,
                &sc.moe_idx,
                &sc.moe_w,
                &sc.moe_fused,
                &mut sc.proj,
                None,
                k,
                r,
                false,
            )?;
        }
    }
    if !shared_first {
        shared_expert(exec, m, w, sc, embd, r, false, x16_ready)?;
        exec.add(&mut sc.proj, &sc.sh_out, r * embd)?;
    }
    Ok(())
}

/// The always-on, ungated shared expert (BF16) over `r` rows of `sc.xn`,
/// into `sc.proj` (`into_proj`) or `sc.sh_out`. Gate and up read one
/// narrowing of xn when the prefill pair is elected (`x16_ready`: already
/// landed).
#[allow(clippy::too_many_arguments)]
fn shared_expert(
    exec: &GpuExecutor,
    m: &MoeDims,
    w: &MoeNv,
    sc: &mut BatchScratch,
    embd: usize,
    r: usize,
    into_proj: bool,
    x16_ready: bool,
) -> Result<(), GpuModelError> {
    if exec.bf16_pf_elect(m.shexp_ff, r) {
        if !x16_ready {
            exec.convert_f32_bf16(&sc.xn, &mut sc.x16, r * embd)?;
        }
        exec.bf16_gemm_pf(&w.sh_gate, None, &sc.x16, &mut sc.sh_gate, r)?;
        exec.bf16_gemm_pf(&w.sh_up, None, &sc.x16, &mut sc.sh_up, r)?;
    } else {
        let x16 = &mut sc.x16;
        super::head::bf16_rows_narrow(exec, &w.sh_gate, &sc.xn, x16, &mut sc.sh_gate, r)?;
        super::head::bf16_rows_narrow(exec, &w.sh_up, &sc.xn, x16, &mut sc.sh_up, r)?;
    }
    exec.swiglu(&mut sc.sh_gate, &sc.sh_up, r * m.shexp_ff)?;
    let y = if into_proj {
        &mut sc.proj
    } else {
        &mut sc.sh_out
    };
    super::head::bf16_rows_narrow(exec, &w.sh_down, &sc.sh_gate, &mut sc.x16, y, r)?;
    Ok(())
}
