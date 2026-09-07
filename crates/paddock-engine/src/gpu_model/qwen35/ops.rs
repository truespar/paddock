//! Qwen3.5/3.6 free-fn layer primitives (prefill attn/mm/ffn, moe_ffn,
//! decode dispatch). Plain fns uniformly pub(crate); shared with qwen3.

use super::*;
use crate::gpu::{GpuError, GpuExecutor, KvDtype, QuantW, RepackedKQ, RepackedQ8};
use crate::gpu_model::gpt_oss::GpuModelError;
use cudarc::driver::CudaSlice;
use paddock_models::gguf::Value;
use paddock_models::mapped::MappedGguf;

pub(super) fn embed_any(
    exec: &GpuExecutor,
    te: &TokEmbd,
    tokens: &CudaSlice<u32>,
    out: &mut CudaSlice<f32>,
    embd: usize,
    n: usize,
) -> Result<(), GpuModelError> {
    match te {
        TokEmbd::Q8(t) => exec.embed_gather_batch_q8(t, tokens, out, embd, n)?,
        TokEmbd::Kq(t) => exec.kquant_gather(t, tokens, out, embd, n)?,
    }
    Ok(())
}

/// Decode GEMV with per-tensor dispatch: the Q8_0 arm is the existing repacked
/// GEMV (bit-identical to before the k-quant seam); the k-quant arm is the
/// stage-1 fused GEMV (exact - f32 products in-kernel).
pub(crate) fn gemv_any(
    exec: &GpuExecutor,
    w: &QuantW,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
) -> Result<(), GpuModelError> {
    match w {
        QuantW::Q8(q) => exec.q8_0_gemv_repacked(q, None, x, y)?,
        QuantW::Kq(k) => exec.kquant_gemv(k, x, y)?,
    }
    Ok(())
}

/// NVFP4 checkpoint-plane matmul election (the qwen3.8 lane).
/// Mirrors nemotron's head election: the tensor-core GEMM for wide multi-row
/// shapes (a bf16-cast mma class, PADDOCK_NVF4_TC=0 kill switch), the
/// bit-exact scalar gemv family everywhere else - rows <= 8 stays scalar so
/// the small-batch decode band keeps one numeric class with the serial
/// spine. x f32 [rows, in_dim] row-major, y f32 [rows, out_dim].
#[track_caller]
pub(crate) fn nvf4_mm(
    exec: &GpuExecutor,
    w: &crate::gpu::Nvf4Plane,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    rows: usize,
) -> Result<(), GpuModelError> {
    // Site witness  - nvf4_ffn is not the only door into the
    // software-dequant class; spec.rs's forward_chunk and mtp_block_pass call
    // this directly. One line per (caller, row band).
    {
        use std::sync::Mutex;
        static SEEN: Mutex<Option<std::collections::HashSet<(&'static str, u32, usize)>>> =
            Mutex::new(None);
        let loc = std::panic::Location::caller();
        let key = (loc.file(), loc.line(), rows.next_power_of_two());
        let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        if g.get_or_insert_with(Default::default).insert(key) {
            tracing::warn!(
                "[nvf4-mm-site] {}:{} rows={} out={} in={}",
                loc.file(),
                loc.line(),
                rows,
                w.out_dim,
                w.in_dim,
            );
        }
    }
    if rows == 1 {
        exec.nvf4_gemv(w, x, y, None)?;
        return Ok(());
    }
    static TC: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let tc = *TC.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_NVF4_TC")
            .map(|v| v != "0")
            .unwrap_or(true)
    });
    // Row floor for the tc arm. 8 is a NUMERICS choice, not a perf one - it
    // keeps the small-batch decode band on one numeric class with the serial
    // spine, and it costs nothing on a device where the W4A4 arm covers rows
    // >= 2 anyway. On a device without that arm (the f4 family is NULLed off
    // cc 12.0, so every cc 10.x B200) it strands rows 2..8 on the issue-bound
    // scalar walk: measured on B200 at c8, pd_nvf4_gemm_mr was 86.7% of all
    // GPU time (65352 launches x 267 us) - the lane is unservably slow that
    // way. With no fp4 arm to fall back to, the bf16-cast tc GEMM
    // is the best class available, so take it from rows >= 2.
    let tc_floor = if exec.has_nvf4_gemm_f4() { 8 } else { 1 };
    if tc
        && rows > tc_floor
        && w.out_dim >= 4096
        && w.in_dim.is_multiple_of(16)
        && exec.has_nvf4_gemm_tc()
    {
        exec.nvf4_gemm_tc(w, x, y, None, rows)?;
    } else {
        exec.nvf4_gemv_batch(w, x, y, None, rows)?;
    }
    Ok(())
}

/// Rows from which the Nvf4Dense FFN chain takes the checkpoint-recipe W4A4
/// arm. The checkpoint's own quant config declares the FFN
/// activations 4-bit (group 16, e4m3 scales, dynamic-local) - W4A4 is the
/// recipe class and what the binding rival serves - so above this band the
/// chain runs the fp4 x fp4 block-scale mma at the full Blackwell fp4 rate
/// instead of the bf16-cast tc arm (~4x below it) or the issue-bound scalar
/// mr walk. Default 2: every batched row class (the first ladder put the
/// rows 2..8 band roughly 45% off the achievable rate while it rode the
/// scalar walk; the f4+split kernels are batch-insensitive at <=128-row
/// padding, 56-60 us/plane vs the walk's ~570 - nv4_ffn_probe). Only
/// rows == 1 (the serial spine) keeps the exact-f32 family - that cell is
/// bandwidth-bound at 94% of roof and wins, so it owes nothing.
/// `PADDOCK_NVF4_W4A4=0` kills the arm; any other value overrides the
/// threshold.
pub(crate) fn nvf4_w4a4_min_rows() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        match paddock_models::dev_var!("PADDOCK_NVF4_W4A4")
            .ok()
            .as_deref()
        {
            Some("0") => usize::MAX,
            Some(s) => s.parse().unwrap_or(2),
            None => 2,
        }
    })
}

/// The Nvf4Dense FFN chain (gate/up -> swiglu -> down) with the W4A4 arm.
///
/// W4A4 route: quantize xn to nvf4 once (both gate and up read the same
/// xq/xs), two f4 GEMMs, then the FUSED swiglu+quantize (the f32 silu(g)*u
/// product never lands in memory - d_ffn_gate/up still hold the raw gate/up
/// outputs afterwards, not the product), then the down f4 GEMM. xq/xs are
/// the d_pxq/d_nvs pair - transient within the chain, same contract as the
/// prefix lane's projection staging; d_pxq's `cap*qw` sizing covers the
/// ff-wide down input (qw maxes over ff).
///
/// Fallback (rows below the band, kill switch, or an old pack): the exact
/// scalar/tc chain, byte-identical to the earlier code.
#[allow(clippy::too_many_arguments)]
#[track_caller]
pub(crate) fn nvf4_ffn(
    exec: &GpuExecutor,
    gate: &crate::gpu::Nvf4Plane,
    up: &crate::gpu::Nvf4Plane,
    down: &crate::gpu::Nvf4Plane,
    xn: &CudaSlice<f32>,
    xq: &mut CudaSlice<i8>,
    xs: &mut CudaSlice<u8>,
    part: &mut CudaSlice<f32>,
    ffn_gate: &mut CudaSlice<f32>,
    ffn_up: &mut CudaSlice<f32>,
    proj: &mut CudaSlice<f32>,
    ff: usize,
    rows: usize,
) -> Result<(), GpuModelError> {
    // Site witness. The W4A16 chain below is the software-dequant
    // class the decode lane and the verify walk both left behind; the imax
    // census still shows pd_nvf4_gemm_tcp at 9.4% of the die, so some caller
    // is still on it. One line per (caller, arm) - the caller location is
    // free via #[track_caller], no signature churn.
    {
        use std::sync::Mutex;
        static SEEN: Mutex<Option<std::collections::HashSet<(&'static str, u32, bool, usize)>>> =
            Mutex::new(None);
        let loc = std::panic::Location::caller();
        let w4a4 = rows >= nvf4_w4a4_min_rows() && exec.has_nvf4_gemm_f4();
        // bucket rows: one line per site per power-of-two band, so a site that
        // first fires at rows=4 still reports when it later runs a 1k prefill
        let key = (loc.file(), loc.line(), w4a4, rows.next_power_of_two());
        let mut g = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        if g.get_or_insert_with(Default::default).insert(key) {
            tracing::warn!(
                "[nvf4-ffn-site] {}:{} arm={} rows={} ff={}",
                loc.file(),
                loc.line(),
                if w4a4 {
                    "w4a4"
                } else {
                    "W4A16-SOFTWARE-DEQUANT"
                },
                rows,
                ff,
            );
        }
    }
    if rows >= nvf4_w4a4_min_rows() && exec.has_nvf4_gemm_f4() {
        exec.quantize_nvf4(xn, xq, xs, rows * gate.in_dim)?;
        exec.nvf4_gemm_f4(gate, xq, xs, ffn_gate, None, rows, Some(part))?;
        exec.nvf4_gemm_f4(up, xq, xs, ffn_up, None, rows, Some(part))?;
        exec.quantize_nvf4_swiglu(ffn_gate, ffn_up, xq, xs, rows * ff)?;
        exec.nvf4_gemm_f4(down, xq, xs, proj, None, rows, Some(part))?;
    } else {
        nvf4_mm(exec, gate, xn, ffn_gate, rows)?;
        nvf4_mm(exec, up, xn, ffn_up, rows)?;
        exec.swiglu(ffn_gate, ffn_up, rows * ff)?;
        nvf4_mm(exec, down, ffn_gate, proj, rows)?;
    }
    Ok(())
}

/// r=1 SERVING-class GEMV off PRE-STAGED int8 activations - the batch lane's
/// decode arm. Same shape as laguna's private `gemv8_any`, lifted here so
/// granite (and anything next) stops reaching for `gemv_any` on this path.
///
/// The distinction matters and is easy to miss: for `QuantW::Q8` `gemv_any`
/// already lands on the optimized repacked GEMV, but for `QuantW::Kq` it lands
/// on `kquant_gemv`, which is the EXACT-f32 ORACLE. A family wired to
/// `gemv_any` for decode therefore runs a tuned kernel on Q8_0 files and the
/// reference kernel on every k-quant file - silently, since both are correct.
/// Measured on granite-30b Q4_K_M: per-node profiling put
/// `pd_kquant_gemv_kernel` at 93.9% of decode GPU time, 263,701 launches
/// = 64 layers x 7 projections + head, per token.
///
/// W4A8 quantizes activations to int8, so this is llama.cpp mmvq's numeric
/// class, not the exact one - which is the class our same-weights oracle
/// itself decodes in. Callers stage xq/xs/ssums once per shared input (the
/// quantize-dedupe rule); Q8_0 keeps the exact GEMV (already at its byte
/// floor, and there is no sums plane to feed it).
pub(crate) fn gemv8_any(
    exec: &GpuExecutor,
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

/// K-quant batch matmul interim (stage 1): dequant the whole weight into the
/// f32 scratch (exact values - same math as the fused GEMV per term), then the
/// tiled f32 GEMM. One layer's weight is f32 at a time, transiently; the
/// stage-2 W4A8 int8-MMA replaces this round trip.
pub(crate) fn kq_gemm(
    exec: &GpuExecutor,
    w: &RepackedKQ,
    x: &CudaSlice<f32>,
    wdq: &mut CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    r: usize,
) -> Result<(), GpuModelError> {
    exec.kquant_dequant_rp(w, wdq)?;
    exec.gemm_f32(wdq, w.dims[0], w.dims[1], x, y, r)?;
    Ok(())
}

/// Small-batch matmul dispatch: 1 row -> the decode GEMV (peak single-row BW),
/// 2..=SPEC_ROWS rows -> the shared-staged tiled GEMM (x traffic amortized over 16
/// output rows - the spec-verify path), larger -> the batch-tiled per-row GEMM.
/// The k-quant arm serves r == 1 only: every r > 1 caller is a spec/MTP path,
/// which stage-1 routing disables for k-quant models (stage-2 work).
pub(crate) fn mm(
    exec: &GpuExecutor,
    w: &QuantW,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    r: usize,
) -> Result<(), GpuModelError> {
    match w {
        QuantW::Q8(q) => mm_q8(exec, q, x, y, r),
        QuantW::Kq(k) => {
            if r == 1 {
                return Ok(exec.kquant_gemv(k, x, y)?);
            }
            Err(GpuModelError::Unsupported(
                "k-quant spec/MTP matmul (r > 1) lands with the stage-2 W4A8 kernels".into(),
            ))
        }
    }
}

/// `mm`'s Q8_0 body - direct entry for Q8_0-only seats (shared-expert FFN).
pub(crate) fn mm_q8(
    exec: &GpuExecutor,
    w: &RepackedQ8,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    r: usize,
) -> Result<(), GpuModelError> {
    if r == 1 {
        exec.q8_0_gemv_repacked(w, None, x, y)?;
    } else if r <= SPEC_ROWS {
        exec.q8_0_gemm_repacked_mt(w, None, x, y, r)?;
    } else {
        exec.q8_0_gemm_repacked(w, None, x, y, r)?;
    }
    Ok(())
}

/// int8 MMQ matmul for the verify chunk: quantize the f32 activations to per-32
/// int8 blocks, then the dp4a GEMM - weight read once at full bandwidth. Verify
/// logits carry activation-quantization noise (~4e-3, llama's own numeric class);
/// the token-level gates arbitrate.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mmq(
    exec: &GpuExecutor,
    w: &QuantW,
    x: &CudaSlice<f32>,
    xq: &mut CudaSlice<i8>,
    xs: &mut CudaSlice<f32>,
    ssums: &mut CudaSlice<f32>,
    part: &mut CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    r: usize,
) -> Result<(), GpuModelError> {
    match w {
        QuantW::Q8(q) => mmq_q8(exec, q, x, xq, xs, part, y, r),
        QuantW::Kq(k) => {
            // r == 1 = the MTP draft-chain steps (incl. the vocab head, the
            // dominant per-draft cost): the W4A8 GEMV - the serving b=1 class
            // - replaces the exact-f32 GEMV here too. Drafts are
            // verified, so the draft class is free; the exact GEMV remains
            // the oracle path elsewhere.
            if r == 1 {
                if exec.has_kquant_gemv_w4a8() {
                    exec.quantize_q8_sums(x, xq, xs, ssums, k.dims[0])?;
                    let needs = crate::gpu::kq_needs_sums(k.ty);
                    return Ok(exec.kquant_gemv_w4a8(k, xq, xs, needs.then_some(&*ssums), y)?);
                }
                return Ok(exec.kquant_gemv(k, x, y)?);
            }
            exec.quantize_q8(x, xq, xs, r * k.dims[0])?;
            mmq_kq_pre(exec, k, xq, xs, ssums, part, y, r)
        }
    }
}

/// k-quant `mmq` body for ALREADY-quantized strided activations. Rungs mirror
/// `mmq_pre`'s r-independent structure: 5..64 takes the K-split W4A8 mma under
/// the same uniform 64-row capacity envelope (never the actual r - the spec
/// gates' r-class rule), everything else the batch-invariant dp4a z-tile.
/// Q4_K/Q5_K recompute the per-16 sums from the staged xq - idempotent across
/// a dedupe group (same xq -> same sums).
pub(crate) fn mmq_kq_pre(
    exec: &GpuExecutor,
    k: &RepackedKQ,
    xq: &CudaSlice<i8>,
    xs: &CudaSlice<f32>,
    ssums: &mut CudaSlice<f32>,
    part: &mut CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    r: usize,
) -> Result<(), GpuModelError> {
    let needs = crate::gpu::kq_needs_sums(k.ty);
    if needs {
        exec.q8_sums_strided(xq, ssums, k.dims[0], r)?;
    }
    // r = 2..5 (the spec-verify r-class, r = k_draft+1): the multi-column
    // W4A8 GEMV - the b=1 GEMV's weight walk dotting r staged activation
    // rows. Beats mma_ks's 455-496 GB/s at the GEMV's 631-648 class AND
    // aligns the verify with the draft chain's numeric class exactly (per
    // column it is the b=1 GEMV's math in its chunk order - the same
    // class-alignment that sent acceptance to full when the draft head
    // moved to W4A8). PADDOCK_NO_KQ_NC=1 pins the previous ladder for A/B.
    if !kq_nc_off() && exec.has_kquant_gemv_w4a8_nc() && GpuExecutor::kquant_gemv_w4a8_nc_fits(k, r)
    {
        exec.kquant_gemv_w4a8_nc(k, xq, xs, needs.then_some(&*ssums), y, r)?;
        return Ok(());
    }
    // mma from r=3 (was 5): the spec verify runs at r = k+1 = 3..8, and a
    // kernel A/B has mma_ks at 455-496 GB/s at r=4 vs the dp4a
    // z-tile's ~300-450 on the same shapes - the old floor left the verify
    // pass ~60% over one weight read (16.9-18.3 ms measured vs ~10.5).
    static KS_MIN: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let ks_min = *KS_MIN.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_KQ_SPEC_KS_MIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3)
    });
    if (ks_min..=64).contains(&r) && part.len() >= 8 * 64 * k.dims[1] && exec.has_kquant_mma_ks() {
        exec.kquant_gemm_mma_ks(k, xq, xs, needs.then_some(&*ssums), part, y, r)?;
    } else {
        exec.kquant_gemm_dp4a(k, xq, xs, needs.then_some(&*ssums), y, r)?;
    }
    Ok(())
}

/// QuantW dispatch over `mmq_pre` - the group-dedupe twin for mixed-quant
/// seats (the k-quant arm reads the same strided xq/xs the Q8 rungs eat).
#[allow(clippy::too_many_arguments)]
pub(crate) fn mmq_pre_any(
    exec: &GpuExecutor,
    w: &QuantW,
    xq: &CudaSlice<i8>,
    xs: &CudaSlice<f32>,
    ssums: &mut CudaSlice<f32>,
    part: &mut CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    r: usize,
) -> Result<(), GpuModelError> {
    match w {
        QuantW::Q8(q) => mmq_pre(exec, q, xq, xs, part, y, r),
        QuantW::Kq(k) => mmq_kq_pre(exec, k, xq, xs, ssums, part, y, r),
    }
}

/// The shared expert's wide-prefill (>64 rows) matmul per seat: Q8_0 keeps
/// its exact int8 ladder; a k-quant seat rides the dp4a GEMM off the same
/// staged activations (the w4a8 pipes want a per-32 sums plane this call
/// site does not carry, and the shared expert is ~1 MB per matrix, so the
/// dp4a class is a few percent of a prefill chunk, not a bottleneck).
#[allow(clippy::too_many_arguments)]
pub(crate) fn shexp_mm_wide(
    exec: &GpuExecutor,
    w: &QuantW,
    xq: &CudaSlice<i8>,
    xs: &CudaSlice<f32>,
    yq: &CudaSlice<u8>,
    ssums: &mut CudaSlice<f32>,
    skfix: &mut CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    batch: usize,
) -> Result<(), GpuModelError> {
    match w {
        QuantW::Q8(q) => prefill_mm_pre(exec, q, xq, xs, yq, skfix, y, batch),
        QuantW::Kq(k) => {
            let needs = crate::gpu::kq_needs_sums(k.ty);
            if needs {
                exec.q8_sums_strided(xq, ssums, k.dims[0], batch)?;
            }
            exec.kquant_gemm_dp4a(k, xq, xs, needs.then_some(&*ssums), y, batch)?;
            Ok(())
        }
    }
}

/// `mmq`'s Q8_0 body - direct entry for Q8_0-only seats (shared-expert FFN).
#[allow(clippy::too_many_arguments)]
pub(crate) fn mmq_q8(
    exec: &GpuExecutor,
    w: &RepackedQ8,
    x: &CudaSlice<f32>,
    xq: &mut CudaSlice<i8>,
    xs: &mut CudaSlice<f32>,
    part: &mut CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    r: usize,
) -> Result<(), GpuModelError> {
    exec.quantize_q8(x, xq, xs, r * w.dims[0])?;
    mmq_pre(exec, w, xq, xs, part, y, r)
}

/// `mmq` with the activation already quantized into xq/xs - the group-dedupe
/// twin (wq/wk/wv and in_qkv/alpha/beta/gate consume the same normed rows, so
/// the first member's quantize serves the whole group; the c32 glue profile
/// counted 3-4 identical quantize_q8 launches per layer). Same rungs, same
/// numerics, bit-exact vs per-call quantize.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mmq_pre(
    exec: &GpuExecutor,
    w: &RepackedQ8,
    xq: &CudaSlice<i8>,
    xs: &CudaSlice<f32>,
    part: &mut CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    r: usize,
) -> Result<(), GpuModelError> {
    if r <= 4 {
        exec.q8_0_gemv_dp4a_nc(w, xq, xs, y, r)?;
    } else if r <= 64 && part.len() >= 8 * 64 * w.dims[1] {
        // K-split mma: one weight pass for any r <= 64 (the mt tile pays a
        // full re-read per 16/24 rows) - this is what lifts the spec ceiling
        // from B*(K+1) <= 24 (one 24-row mt pass, A6000 rule) to <= 64 on
        // big dies. The rung must not depend on r: the spec gates compare
        // single-slot refs (r = K+1 <= SPEC_ROWS) with batched runs
        // (r = B*(K+1)) and demand token-identical output, so each weight
        // rides one numeric class across that whole r range. The capacity
        // check uses the UNIFORM 64-row envelope (never the actual r): an
        // r-dependent check split wq between r=5 (ks) and r=40 (mt) on the
        // 35B and broke the B=8 spec gate. lm_head exceeds the envelope and
        // stays mt on both sides of every pair.
        // MEASURED TRADE on GB202 (27B): batched spec gains a lot against a
        // wide-outs-on-mt split, while single-stream spec at k=4 gives some
        // back - ks is weakest at r=5..9 on the wide outs, though still
        // 1.43x base there. Serving concurrency is the tier-1 workload, so
        // the batched side wins the tie.
        exec.q8_0_gemm_mma_ks(w, xq, xs, part, y, r)?;
    } else {
        exec.q8_0_gemm_mt_dp4a(w, xq, xs, y, r)?;
    }
    Ok(())
}

/// Prefill matmul through the cuBLAS f16 tensor-core route (dequant weight to
/// f16, convert activations, GEMM). Superseded by `prefill_mm` (int8 MMA - reads
/// the weight once as int8, no dequant-to-f16 write, and beats this at prefill
/// scale) but kept as the f16 fallback: bit-exact-gated alternative if a future
/// arch lacks the s8 MMA tile.
/// One prefill matmul through the int8 tensor-core route: quantize the f32
/// activations to per-32 int8, then the MMA GEMM (weight read once as int8 -
/// beats the cuBLAS f16 staging at prefill scale, and it's llama's own prefill
/// numeric class). Same activation-quant noise as the serving/verify mmq path;
/// the b9895 greedy gate is the acceptance bar. Alpha/beta must not use this -
/// they stay on the exact q8_0_gemm_repacked (the P6b decay-numerics rule).
/// Which KV dtypes may ride the fast `attn_prefill_f16_paged` export for a
/// given head geometry. Fp16 always could; fp8-e4m3 joined for the qwen35
/// full-attn shapes (hd256, G in {4,6,8}) when the v4 staged-HMMA tile grew
/// its raw-e4m3 PIPE arm - before that the
/// elected kv8 class fell to the scalar paged walk (per-q-head grid, G-fold
/// redundant KV reads, no tensor cores). The export itself falls back to
/// that scalar tile when the v4 arm is killed (PADDOCK_NO_PF_V4), so this
/// gate only decides the ENGINE routing; PADDOCK_NO_QPF8 reverts it.
pub(crate) fn pf_attn_dtype_ok(kv_dtype: KvDtype, n_heads: usize, n_kv_heads: usize) -> bool {
    match kv_dtype {
        KvDtype::Fp16 => true,
        KvDtype::Fp8E4m3 => {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_QPF8").is_none())
                && n_kv_heads > 0
                && n_heads.is_multiple_of(n_kv_heads)
                && matches!(n_heads / n_kv_heads, 4 | 6 | 8)
        }
    }
}

/// Prefill attention dispatch (P6f): real prefill spans (r > 24, head_dim
/// 128; every row of a pass shares one KV slot) go to the tiled prefill
/// kernel - same numeric class, ~8× less wall time at pp512. Short spans
/// (spec verify runs r ≤ 24 by the B×(K+1) rule) stay on the decode-batch
/// kernel with their exact current numerics. All prefill paths share this
/// helper so the per-slot gate's compared paths always dispatch identically.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_attn(
    exec: &GpuExecutor,
    q: &CudaSlice<f32>,
    kc: &CudaSlice<u8>,
    vc: &CudaSlice<u8>,
    sinks: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    positions: &CudaSlice<u32>,
    slots: &CudaSlice<u32>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_dim: usize,
    r: usize,
    scale: f32,
    kv_dtype: KvDtype,
    // P4 paged KV: `Some((block_tables, blocks_per_slot))` reads the KV pool
    // through per-slot block tables. Paged prefill uses the decode-class kernel
    // (attn_decode_batch_paged) - bit-exact vs the dense attn_decode_batch
    // fallback; the tiled/WMMA prefill kernels are not paged yet (perf follow-up,
    // P4b), so paging forces this class here.
    paged: Option<(&CudaSlice<u32>, usize)>,
    // FlashDecoding scratch for the r <= 24 fallback (a contention
    // profile): the plain decode-class walk is n_heads x r blocks scanning the
    // row's whole history serially - 451 us/call on chunk tails and short
    // resumed spans (agentic follow-ups) at depth 2k on GB202. With scratch,
    // the fallback runs the same partial+combine every decode site uses
    // (parity ~5e-8 across split counts, the accepted split class). None = the
    // exact old path.
    split_scratch: Option<(&mut CudaSlice<f32>, &mut CudaSlice<f32>)>,
) -> Result<(), GpuModelError> {
    if let Some((block_tables, blocks_per_slot)) = paged {
        // Paged prefill class ladder (mirrors the dense dispatch below):
        //   P4b-2: r>24, hd256, Fp16 (or fp8 on the v4-arm geometries) ->
        //          paged f16 WMMA / staged-HMMA export.
        //   P4b:   r>24, hd128/256   -> paged tiled (bit-exact vs dense tiled).
        //   P4:    short / missing   -> decode-class paged fallback (split
        //         partial+combine when scratch is provided).
        if r > 24
            && head_dim == 256
            && max_ctx.is_multiple_of(64)
            && pf_attn_dtype_ok(kv_dtype, n_heads, n_kv_heads)
            && exec.has_attn_prefill_f16_paged()
        {
            exec.attn_prefill_f16_paged(
                q,
                kc,
                vc,
                sinks,
                out,
                positions,
                slots,
                block_tables,
                blocks_per_slot,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_dim,
                0,
                r,
                scale,
                kv_dtype,
            )?;
        } else if r > 24 && (head_dim == 128 || head_dim == 256) && exec.has_attn_prefill_paged() {
            exec.attn_prefill_paged(
                q,
                kc,
                vc,
                sinks,
                out,
                positions,
                slots,
                block_tables,
                blocks_per_slot,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_dim,
                0,
                r,
                scale,
                kv_dtype,
            )?;
        } else {
            if let Some((attn_o, attn_ml)) = split_scratch {
                let n_splits = attn_splits(n_heads, r, exec.sm_count());
                if n_splits > 1 && exec.has_attn_partial_batch_paged() {
                    exec.attn_partial_batch_paged(
                        q,
                        kc,
                        vc,
                        attn_o,
                        attn_ml,
                        positions,
                        Some(slots),
                        block_tables,
                        blocks_per_slot,
                        n_heads,
                        n_kv_heads,
                        head_dim,
                        kv_dim,
                        0,
                        n_splits,
                        r,
                        scale,
                        kv_dtype,
                    )?;
                    exec.attn_combine_batch(
                        attn_o, attn_ml, sinks, out, n_heads, head_dim, n_splits, r,
                    )?;
                    return Ok(());
                }
            }
            exec.attn_decode_batch_paged(
                q,
                kc,
                vc,
                sinks,
                out,
                positions,
                Some(slots),
                block_tables,
                blocks_per_slot,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_dim,
                0,
                r,
                scale,
                kv_dtype,
            )?;
        }
        return Ok(());
    }
    // PADDOCK_PREFILL_ATTN=f32|decode pins a lower attention class (bisection)
    let pin = paddock_models::dev_var!("PADDOCK_PREFILL_ATTN").unwrap_or_default();
    // the f16 WMMA tile stages the cache as __half - Fp16 caches only
    if r > 24
        && head_dim == 256
        && max_ctx.is_multiple_of(64)
        && pin.is_empty()
        && matches!(kv_dtype, KvDtype::Fp16)
    {
        // tensor-core f16 path (P6i) - llama's own prefill attention class
        exec.attn_prefill_f16(
            q, kc, vc, sinks, out, positions, slots, n_heads, n_kv_heads, head_dim, max_ctx,
            kv_dim, 0, r, scale, kv_dtype,
        )?;
    } else if r > 24 && (head_dim == 128 || head_dim == 256) && pin != "decode" {
        exec.attn_prefill(
            q, kc, vc, sinks, out, positions, slots, n_heads, n_kv_heads, head_dim, max_ctx,
            kv_dim, 0, r, scale, kv_dtype,
        )?;
    } else {
        if let Some((attn_o, attn_ml)) = split_scratch {
            let n_splits = attn_splits(n_heads, r, exec.sm_count());
            if n_splits > 1 {
                exec.attn_partial_batch(
                    q,
                    kc,
                    vc,
                    attn_o,
                    attn_ml,
                    positions,
                    Some(slots),
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    max_ctx,
                    kv_dim,
                    0,
                    n_splits,
                    r,
                    scale,
                    kv_dtype,
                )?;
                exec.attn_combine_batch(
                    attn_o, attn_ml, sinks, out, n_heads, head_dim, n_splits, r,
                )?;
                return Ok(());
            }
        }
        exec.attn_decode_batch(
            q,
            kc,
            vc,
            sinks,
            out,
            positions,
            Some(slots),
            n_heads,
            n_kv_heads,
            head_dim,
            max_ctx,
            kv_dim,
            0,
            r,
            scale,
            kv_dtype,
        )?;
    }
    Ok(())
}

/// Decode-class full attention with die-scaled FlashDecoding splits (see
/// [`attn_splits`]): partial+combine when the unsplit n_heads*batch grid would
/// underfill the die, the plain single-kernel walk otherwise. All decode-class
/// call sites (b=1 step, serving, spec verify, MTP) share this helper so their
/// split decisions stay mutually consistent.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attn_decode_dispatch(
    exec: &GpuExecutor,
    q: &CudaSlice<f32>,
    kc: &CudaSlice<u8>,
    vc: &CudaSlice<u8>,
    sinks: &CudaSlice<f32>,
    attn_o: &mut CudaSlice<f32>,
    attn_ml: &mut CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    positions: &CudaSlice<u32>,
    slots: Option<&CudaSlice<u32>>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_ctx: usize,
    kv_dim: usize,
    batch: usize,
    scale: f32,
    kv_dtype: KvDtype,
    // P3 paged KV: `Some((block_tables, blocks_per_slot))` reads the KV pool
    // through per-slot block tables (the single-pass path only - the split path
    // needs a paged partial kernel, P3b). `None` = the dense `slot*max_ctx` path.
    paged: Option<(&CudaSlice<u32>, usize)>,
) -> Result<(), GpuModelError> {
    // tcgen05/TMEM decode attention. A kernel census put this band at
    // 2117 us/tick (114.7 us/launch on a dim3(n_kv_heads, rows, n_splits) =
    // 2048-CTA grid, 98 MB of KV read at 0.86 TB/s), against ~151 us/tick for
    // a single persistent fmha kernel over the same work.
    // qwen3.8's full-attn shape is 24q/4kv/hd256 = G6, the same hd256
    // geometry gemma's SWA arm already runs at G2 (CH=192, BPC=12,
    // SCOLS=16); G only ever indexes the 8-row M tile both S and O MMAs
    // emit, so the pack takes <256,6> on the same tmem and P image.
    //
    // Only reachable from the paged decode sites (batch.rs 4116/6076),
    // which are one-row-per-slot by construction - the kernel treats each
    // (kv_head, row) as a cell with a single query position. The dense
    // callers pass paged=None and never see this arm.
    //
    // The effective window is a CONSTANT max_ctx + 16, not a live band:
    // this lane captures decode graphs, and a window derived from live
    // positions would be baked into a replay. Constant is graph-safe, and
    // it costs nothing - the walk length comes from `positions`, the window
    // only bounds glo (and the host's tick-table fit check). The +16 keeps
    // ew > pos_max when pos_max is an exact multiple of the block size, the
    // corner where a bare bound makes the kernel's walk drop key 0.
    if let Some((block_tables, blocks_per_slot)) = paged {
        static TC5_ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let tc5_on =
            *TC5_ON.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_NO_Q35_TC5ATTN").is_none());
        if tc5_on
            && batch >= 8
            && head_dim == 256
            && n_kv_heads > 0
            && n_heads == n_kv_heads * 6
            && matches!(kv_dtype, KvDtype::Fp8E4m3)
            && exec.has_attn_decode_tc5_paged()
            && exec.attn_decode_tc5_paged(
                q,
                kc,
                vc,
                sinks,
                out,
                positions,
                slots,
                block_tables,
                blocks_per_slot,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_dim,
                max_ctx + 16,
                batch,
                scale,
                kv_dtype,
            )?
        {
            // final rows landed in `out`; no partials, no combine
            return Ok(());
        }
    }
    let n_splits = attn_splits(n_heads, batch, exec.sm_count());
    if n_splits > 1 {
        // Only reachable on a ≥128-SM die (attn_splits engages). P3b: the paged
        // FlashDecoding partial reads the block pool; the combine is unchanged
        // (position-agnostic). Falls back to the dense partial when the pack has
        // no paged partial (pre-P3b pack) or when not paged.
        if let Some((block_tables, blocks_per_slot)) = paged {
            if !exec.has_attn_partial_batch_paged() {
                return Err(GpuModelError::Unsupported(
                    "pack lacks the paged FlashDecoding partial (P3b) - set PADDOCK_NO_ATTN_SPLIT=1"
                        .into(),
                ));
            }
            exec.attn_partial_batch_paged(
                q,
                kc,
                vc,
                attn_o,
                attn_ml,
                positions,
                slots,
                block_tables,
                blocks_per_slot,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_dim,
                0,
                n_splits,
                batch,
                scale,
                kv_dtype,
            )?;
        } else {
            exec.attn_partial_batch(
                q, kc, vc, attn_o, attn_ml, positions, slots, n_heads, n_kv_heads, head_dim,
                max_ctx, kv_dim, 0, n_splits, batch, scale, kv_dtype,
            )?;
        }
        exec.attn_combine_batch(
            attn_o, attn_ml, sinks, out, n_heads, head_dim, n_splits, batch,
        )?;
    } else if let Some((block_tables, blocks_per_slot)) = paged {
        exec.attn_decode_batch_paged(
            q,
            kc,
            vc,
            sinks,
            out,
            positions,
            slots,
            block_tables,
            blocks_per_slot,
            n_heads,
            n_kv_heads,
            head_dim,
            kv_dim,
            0,
            batch,
            scale,
            kv_dtype,
        )?;
    } else {
        exec.attn_decode_batch(
            q, kc, vc, sinks, out, positions, slots, n_heads, n_kv_heads, head_dim, max_ctx,
            kv_dim, 0, batch, scale, kv_dtype,
        )?;
    }
    Ok(())
}

/// Spec-verify attention over PADDED block-major rows - `k1` consecutive rows
/// per slot block, positions ascending inside the block (what
/// `stage_spec_rows` builds for every verify round). One KV walk per
/// (kv-head, block, split) serves the block's k1 rows on the fp8 krs GV=6
/// arm; the per-row decode walk this replaces re-read each slot's whole KV
/// k1 times - and above 46 rows (where `attn_splits` collapses to 1) did so
/// on the scalar per-(q-head, row) kernel, which at wide batch is ~42% of
/// the whole GPU. Returns Ok(false) when the arm can't engage; the caller
/// then takes [`attn_decode_dispatch`] unchanged.
///
/// Width gate (the gemma4 lesson, batch.rs spec_arm): below ~16 blocks the
/// slots' KV is L2-resident and the per-row krs walk at 16 splits fills the
/// die better than 4 x blocks shared-walk CTAs do - an always-on shared
/// walk costs narrow batches badly. So the arm engages where either the
/// block count reaches the wide band OR the fallback would be the scalar
/// unsplit walk - which loses to the shared walk at any width.
///
/// Split count: FIXED (never scaled by rows) per the `attn_splits` law; the
/// kernel still clamps its effective splits to the block's own context
/// (CTA-local, so batch shape can't change a row's reduction order).
/// ELECTED 1 = the in-kernel finalize, no combine. A same-serve ladder over
/// the split count came out monotone - fewer splits win, and fin wins
/// outright - which is the OPPOSITE of gemma4's sm_120 election
/// (SWA fin 306.6 -> sp2 252.8): this arm already overlaps its walk (DBK +
/// KVS) and at 24 heads x 256 rows the partial round trip + combine is
/// more bytes than the whole fp8 KV walk saves. `spec_verify_splits`
/// carries it, clamped to the FlashDecoding scratch.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attn_verify_dispatch(
    exec: &GpuExecutor,
    q: &CudaSlice<f32>,
    kc: &CudaSlice<u8>,
    vc: &CudaSlice<u8>,
    sinks: &CudaSlice<f32>,
    attn_o: &mut CudaSlice<f32>,
    attn_ml: &mut CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    positions: &CudaSlice<u32>,
    slots: &CudaSlice<u32>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    kv_dim: usize,
    rows: usize,
    k1: usize,
    scale: f32,
    kv_dtype: KvDtype,
    paged: Option<(&CudaSlice<u32>, usize)>,
    // deepest position across the round's rows (host truth from
    // stage_spec_rows) - drives the ctx-adaptive split election below
    ctx_hint: usize,
) -> Result<bool, GpuModelError> {
    let Some((block_tables, blocks_per_slot)) = paged else {
        return Ok(false);
    };
    // the pack arm's own geometry gate, mirrored (it answers -2 otherwise and
    // the partial wrapper surfaces that as an error)
    if !matches!(kv_dtype, KvDtype::Fp8E4m3)
        || head_dim != 256
        || n_kv_heads == 0
        || n_heads != 6 * n_kv_heads
        || !(2..=8).contains(&k1)
        || rows == 0
        || !rows.is_multiple_of(k1)
        || !exec.has_attn_spec_batch_paged()
        || paddock_models::dev_var_os!("PADDOCK_QWEN35_NO_VERIFY_FA").is_some()
    {
        return Ok(false);
    }
    let blocks = rows / k1;
    let scalar_fallback = attn_splits(n_heads, rows, exec.sm_count()) == 1;
    if blocks < 16 && !scalar_fallback {
        return Ok(false);
    }
    // partial scratch holds 2*fill*MAX_ATTN_SPLITS (head,row,split) cells
    let sp_cap = (2 * super::attn_fill_blocks(exec.sm_count()) * super::MAX_ATTN_SPLITS)
        / (n_heads * rows).max(1);
    // NOTE (a FALSIFIED idea, kept as the record): this arm launches krs
    // with GV=6, so the kernel's adaptive `s_eff = ceil(n_pos/(TPC*PT))` is
    // live and clamps to whatever count we pass - at 15k ctx it wants ~118
    // and our ceil(ctx/2048) hands it 5, i.e. We are the binding constraint.
    // Handing the decision back (pass the cap, let the kernel pick) was built
    // and measured, and the finer partition loses: at 31 splits the grid is
    // 744 CTAs (4 waves) each walking ~484 keys plus a 31-way combine over
    // every (head,row); at 5 it is 120 CTAs walking ~3000 keys with a cheap
    // combine. Filling the die costs more in partial+combine traffic than the
    // shorter walk saves. Keep the conservative host count.
    let n_splits = spec_verify_splits(ctx_hint).min(sp_cap).max(1);
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        eprintln!(
            "[verify-fa] engaged: rows={rows} k1={k1} blocks={blocks} splits={n_splits} (scalar_fallback={scalar_fallback})"
        );
    });
    if n_splits == 1 {
        if !exec.has_attn_spec_batch_fin() {
            return Ok(false);
        }
        return exec
            .attn_spec_batch_fin(
                q,
                kc,
                vc,
                out,
                attn_ml,
                positions,
                Some(slots),
                block_tables,
                blocks_per_slot,
                n_heads,
                n_kv_heads,
                head_dim,
                kv_dim,
                0,
                rows,
                k1,
                scale,
                kv_dtype,
            )
            .map_err(Into::into);
    }
    exec.attn_spec_batch_paged(
        q,
        kc,
        vc,
        attn_o,
        attn_ml,
        positions,
        Some(slots),
        block_tables,
        blocks_per_slot,
        n_heads,
        n_kv_heads,
        head_dim,
        kv_dim,
        0,
        n_splits,
        rows,
        k1,
        scale,
        kv_dtype,
    )?;
    exec.attn_combine_batch(
        attn_o, attn_ml, sinks, out, n_heads, head_dim, n_splits, rows,
    )?;
    Ok(true)
}

/// Verify-walk split count for [`attn_verify_dispatch`] - CONTEXT-adaptive:
/// `ceil(ctx / 2048)`, so it stays 1 (the in-kernel finalize elected at
/// ~1k ctx) up to two KV pages of depth and grows with the walk beyond.
/// Mechanism: the fin election's ladder ran where one split's walk was ~1k
/// keys and the grid already covered the die; at 8-16k ctx the same fin
/// launch is 24-32 CTAs each walking the whole context serially, and
/// pd_attn_spec_fa_krs then profiles at ~507us avg / 22% of the decode wall
/// on an otherwise idle die. The launched count derives from the ROUND'S
/// DEEPEST row; determinism across batch shapes holds because the kernel's
/// GV-arm clamp (context-only, fill_sms=0 on the spec launch) trims every
/// block to a function of its own context, and same-position runs derive
/// the same launched count. The scratch cap (sp_cap) still pins wide-row
/// rounds (c32-class) to fin independently. PADDOCK_QWEN35_VERIFY_SP pins
/// a fixed count (=1 restores the pure-fin arm - the A/B off leg).
fn spec_verify_splits(ctx_hint: usize) -> usize {
    static V: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    if let Some(n) = *V.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_QWEN35_VERIFY_SP")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n >= 1)
    }) {
        return n;
    }
    // One split per 2048-token KV page. Measured best of {1, 5, 31} at the
    // 15k-ctx cell (518.9 vs 499.3 fin vs 513.4 at the cap - see the
    // dispatch note's rung-2 falsification); the ladder between 2 and 16 is
    // unexplored, so this is "best measured", not "proven optimal".
    ctx_hint.div_ceil(2048).clamp(1, super::MAX_ATTN_SPLITS)
}

/// Routed-expert MoE FFN, token-batched (the qwen3.6-A3B class): router
/// matvec -> warp top-k (softmax over the selected logits - b9951
/// qwen35moe's softmax+renorm is algebraically identical) -> fused q8 dp4a
/// gate+up+SwiGLU over the routed rows -> quantize -> down + weighted
/// combine, then the shared expert (dense repacked-class SwiGLU) folded in
/// behind its per-token sigmoid scalar gate. Writes `proj` (caller adds the
/// residual). Token-batched grid (moe_ff x n_active x batch) serves decode
/// AND b=1 chunked prefill; the sorted/serving classes are the Q2 follow-up.
#[allow(clippy::too_many_arguments)]
pub(super) fn moe_ffn(
    exec: &GpuExecutor,
    w: &MoeFfnWeights,
    dims: MoeDims,
    embd: usize,
    batch: usize,
    // Spec verify/MTP paths pass false: their exact-match gates compare
    // single-slot refs (r = K+1) against batched runs (r = B*(K+1)) and the
    // sorted boundary would put the two sides in different classes.
    sorted_ok: bool,
    xn: &CudaSlice<f32>,
    xq: &mut CudaSlice<i8>,
    xs: &mut CudaSlice<f32>,
    // per-16 int8 sums scratch for the k-quant expert seats' mu term
    // (Q4_K/Q5_K) - reused for xq sums (gate_up) then fq sums (down);
    // untouched on Q8 seats.
    ssums: &mut CudaSlice<f32>,
    // b2: u8 (ue8m0) scale planes for the fp4 W4A8 bs path - e4m3 activation
    // scales (xs8) and e4m3 fused-output scales (fs8). Unused on the Q8 path.
    xs8: &mut CudaSlice<u8>,
    fs8: &mut CudaSlice<u8>,
    logits: &mut CudaSlice<f32>,
    zero_bias: &CudaSlice<f32>,
    idx: &mut CudaSlice<u32>,
    topk_w: &mut CudaSlice<f32>,
    fused: &mut CudaSlice<f32>,
    fq: &mut CudaSlice<i8>,
    fs: &mut CudaSlice<f32>,
    srow: &mut CudaSlice<u32>,
    sslot: &mut CudaSlice<u32>,
    bexp: &mut CudaSlice<u32>,
    part: &mut CudaSlice<f32>,
    pxq: &mut CudaSlice<i8>,
    pxs: &mut CudaSlice<f32>,
    yq: &mut CudaSlice<u8>,
    skfix: &mut CudaSlice<f32>,
    shexp_gate: &mut CudaSlice<f32>,
    shexp_up: &mut CudaSlice<f32>,
    shexp_out: &mut CudaSlice<f32>,
    proj: &mut CudaSlice<f32>,
) -> Result<(), GpuModelError> {
    exec.quantize_q8(xn, xq, xs, batch * embd)?;
    exec.matvec_f32_batch(&w.router_w, xn, logits, batch)?;
    exec.moe_topk_batch(
        logits,
        zero_bias,
        dims.n_expert,
        dims.n_active,
        idx,
        topk_w,
        batch,
    )?;
    // Sorted vs token-batched: the sorted (moe_align) class reads each
    // touched expert's weights once per pass - at prefill scale the
    // token-batched kernels re-read routed rows per token (bring-up pp512
    // measured 0.18x llama on exactly that). But with 256 experts the sorted
    // blocks only pack once the pair count clears the expert count, and
    // below that the grid is mostly PAD rows. MEASURED on the 35B (GB202)
    // with the int8-MMA sorted pair: 128 pairs (B=16 serving) already favor
    // sorted (923 vs 875 aggregate; B=32 1325 vs 1037, B=64 1806 vs 1123)
    // while 64 pairs (B=8) still favor token-batched (677 vs 643). The dp4a
    // sorted fallback preferred 1024+ - the boundary belongs to the mma
    // class that actually serves. PADDOCK_QMOE_SORTED_MIN overrides for
    // retunes; PADDOCK_NO_SORTED_QMOE=1 pins token-batched for A/B.
    let sorted_min: usize = std::env::var("PADDOCK_QMOE_SORTED_MIN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128);
    let seats_q8 =
        w.gate_exps.q8().is_some() && w.up_exps.q8().is_some() && w.down_exps.q8().is_some();
    // k-quant seats ride their own sorted mma pair (20_kquant_moe's ks-ring
    // kernels over the same moe_align layout). That pair kernel is
    // single-dtype, so a file that ever mixes gate/up types keeps the layer
    // token-batched; mixed Q8/kq SEATS additionally need the Q8 mma present
    // (the dp4a fallback trades f32 `fused` rows, incompatible with the kq
    // pair's direct fq/fs). PADDOCK_NO_KQMOE_MMA=1 pins kq seats
    // token-batched for A/B.
    let q8_mma = exec
        .kernels()
        .map(|k| k.q8_0_moe_gate_up_mma.is_some())
        .unwrap_or(false)
        && exec.compute_capability().0 >= 8
        && paddock_models::dev_var_os!("PADDOCK_NO_QMOE_MMA").is_none();
    let kq_sorted_ok = seats_q8 || {
        let pair_ok = match (w.gate_exps.kq(), w.up_exps.kq()) {
            (Some(g), Some(u)) => g.ty == u.ty,
            _ => true,
        };
        let all_kq =
            w.gate_exps.q8().is_none() && w.up_exps.q8().is_none() && w.down_exps.q8().is_none();
        // i-quant seats have no mma lane: they stay token-batched at every width
        let any_iq = [&w.gate_exps, &w.up_exps, &w.down_exps]
            .iter()
            .any(|e| e.kq().is_some_and(|k| crate::gpu::kq_is_iq(k.ty)));
        exec.has_kquant_moe_mma()
            && paddock_models::dev_var_os!("PADDOCK_NO_KQMOE_MMA").is_none()
            && pair_ok
            && !any_iq
            && (all_kq || q8_mma)
    };
    // k-quant seats keep the MEASURED mma boundary regardless of the
    // proven-stack QMOE_SORTED_MIN=1 default (an fp4/Q8-class tune): at b=1
    // the sorted pair is latency-bound on PAD-heavy grids (measured on the
    // 35B UD: 61.8/51.9 us per launch at 64 live CTAs ≈ 4.5 ms/token of MoE
    // + 0.3 ms of sorted-down sums over ~7.9k mostly-PAD rows) while the
    // token-batched pair does the same weight reads on a full-die grid -
    // sorted only pays once pairs clear the expert count (128 measured).
    let kq_sorted_min: usize = paddock_models::dev_var!("PADDOCK_KQMOE_SORTED_MIN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128);
    let kq_floor = if seats_q8 { 0 } else { kq_sorted_min };
    let sorted = sorted_ok
        && kq_sorted_ok
        && batch * dims.n_active >= sorted_min.max(kq_floor)
        && paddock_models::dev_var_os!("PADDOCK_NO_SORTED_QMOE").is_none();
    if sorted {
        // b2: fp4 W4A8 grouped MoE (mmq, BM=32) - routed-expert weights at ~half
        // the Q8_0 DRAM, the lever for this weight-bandwidth-bound MoE. Engages
        // only when planes are loaded (PADDOCK_QWEN35_MOE_FP4) and the batch is
        // large enough; lossy (perplexity-gated), so decode/small batch and the
        // default build stay on the exact Q8_0 pair below. qwen MoE = plain
        // silu(g)*u: alpha=1, no clamp (limit=inf), zero bias, up_add=0.
        let fp4 = w
            .gate_exps_fp4
            .as_ref()
            .zip(w.up_exps_fp4.as_ref())
            .zip(w.down_exps_fp4.as_ref())
            .zip(w.moe_zero_bias.as_ref())
            .filter(|_| {
                batch >= moe_fp4_min_batch()
                    && paddock_models::dev_var_os!("PADDOCK_NO_MOE_FP4").is_none()
            });
        let fp4_used = fp4.is_some();
        if let Some((((g4, u4), d4), zb)) = fp4 {
            // Block-scale (bs) sorted MoE. e4m3 activations + fp4 weights;
            // fused output is e4m3 (fq/fs8) feeding the bs down GEMM. Weight layout
            // (q8_0_to_mxfp4) matches the bs family (the dense mxfp4_gemm_bs sibling).
            //
            // BM=64 prefill config (PADDOCK_QWEN35_MOE_FP4_BM64): the bs64 pair
            // (built for exactly this, never wired) halves the per-launch block
            // count at fat experts -> touched experts' weights read ~half as often.
            // profiled @ r=2048 prefill the BM=32 kernel sits at 40% tensor / 47% L2
            // sectors / 26% DRAM - intensity-limited by the 32-token tile, the
            // same lever the Q8 mma path already ships as BM=64 (+15-17%). Gated
            // on the Q8 path's fill heuristic (only pays when blocks populate);
            // decode/small batch stays BM=32. Scratch (fq/fs8/srow/sslot) is
            // already sized for the 64-pad superset.
            let bm64_fill: usize = paddock_models::dev_var!("PADDOCK_QMOE_BM64_FILL")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(64);
            let use64 = exec.has_moe_bs64()
                && paddock_models::dev_var_os!("PADDOCK_QWEN35_MOE_FP4_BM64").is_some()
                && batch * dims.n_active >= dims.n_expert * bm64_fill;
            if paddock_models::dev_var_os!("PADDOCK_QMOE_BM64_DEBUG").is_some() {
                tracing::info!(
                    "[fp4-bm64] use64={use64} has={} env={} batch={batch} pairs={} need={}",
                    exec.has_moe_bs64(),
                    paddock_models::dev_var_os!("PADDOCK_QWEN35_MOE_FP4_BM64").is_some(),
                    batch * dims.n_active,
                    dims.n_expert * bm64_fill
                );
            }
            let bm = if use64 { 64usize } else { 32usize };
            let max_blocks = (batch * dims.n_active + dims.n_expert * (bm - 1)).div_ceil(bm);
            if use64 {
                exec.moe_align_bm(
                    idx,
                    srow,
                    sslot,
                    bexp,
                    batch,
                    dims.n_active,
                    dims.n_expert,
                    bm,
                    max_blocks,
                )?;
            } else {
                exec.moe_align(
                    idx,
                    srow,
                    sslot,
                    bexp,
                    batch,
                    dims.n_active,
                    dims.n_expert,
                    max_blocks,
                )?;
            }
            exec.quantize_e4m3(xn, xq, xs8, batch * embd)?;
            if use64 {
                exec.mxfp4_moe_gate_up_bs64(
                    g4,
                    zb,
                    u4,
                    zb,
                    srow,
                    bexp,
                    xq,
                    xs8,
                    fq,
                    fs8,
                    embd,
                    dims.moe_ff,
                    max_blocks,
                    batch,
                    1.0,
                    f32::INFINITY,
                    0.0,
                )?;
                exec.mxfp4_moe_down_bs64(
                    d4,
                    zb,
                    srow,
                    sslot,
                    bexp,
                    topk_w,
                    fq,
                    fs8,
                    part,
                    dims.moe_ff,
                    embd,
                    dims.n_active,
                    max_blocks,
                    batch,
                )?;
            } else {
                exec.mxfp4_moe_gate_up_bs(
                    g4,
                    zb,
                    u4,
                    zb,
                    srow,
                    bexp,
                    xq,
                    xs8,
                    fq,
                    fs8,
                    embd,
                    dims.moe_ff,
                    max_blocks,
                    batch,
                    1.0,
                    f32::INFINITY,
                    0.0,
                )?;
                exec.mxfp4_moe_down_bs(
                    d4,
                    zb,
                    srow,
                    sslot,
                    bexp,
                    topk_w,
                    fq,
                    fs8,
                    part,
                    dims.moe_ff,
                    embd,
                    dims.n_active,
                    max_blocks,
                    batch,
                )?;
            }
        } else {
            // int8-MMA (tensor-core) sorted pair where the pack + die carry it;
            // the dp4a sorted pair is the fallback class and the A/B pin
            // (PADDOCK_NO_QMOE_MMA=1). The mma gate_up quantizes its output in
            // registers, so the separate quantize pass disappears with it.
            // k-quant seats dispatch per-tensor to the kq mma pair (same sorted
            // layout, same fq/fs handshake - mixing kq gate_up with Q8 down and
            // vice versa is fine).
            let mma = q8_mma;
            // Wider prefill block (BM=64, single-buffered weights): halves the
            // weight-DRAM re-reads -> ~1.15-1.17× on the mma pair, confirmed in-engine
            // at realistic sparse population (tokens/expert≈105; see
            // DEFAULT-ON for large
            // prefill; it only pays when blocks are well-populated (tokens/expert ≥
            // fill), and at low population wastes compute on PAD, so the batch
            // threshold keeps decode/small-batch on BM=32 (BM=64 regresses decode
            // 0.70× - it's latency-bound, not DRAM-bound there). Bit-identical either
            // way (BM only regroups tokens; PAD -> zeros - validated 3 ways in the doc).
            // PADDOCK_NO_QMOE_BM64=1 pins BM=32 everywhere. The kq mma pair is
            // BM=32-only for now (its ring is the dense ks BN=32 shape), so any
            // kq seat pins BM=32 - the BM=64 kq ring is a measured follow-up.
            let bm64_fill: usize = paddock_models::dev_var!("PADDOCK_QMOE_BM64_FILL")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(64);
            let bm = if seats_q8
                && mma
                && exec
                    .kernels()
                    .map(|k| k.moe_align_bm.is_some())
                    .unwrap_or(false)
                && paddock_models::dev_var_os!("PADDOCK_NO_QMOE_BM64").is_none()
                && batch * dims.n_active >= dims.n_expert * bm64_fill
            {
                64usize
            } else {
                32usize
            };
            let max_blocks = (batch * dims.n_active + dims.n_expert * (bm - 1)).div_ceil(bm);
            if bm == 64 && paddock_models::dev_var_os!("PADDOCK_QMOE_BM64_DEBUG").is_some() {
                tracing::info!(
                    "[qmoe] BM=64 prefill path engaged (batch={batch}, max_blocks={max_blocks})"
                );
            }
            if bm == 64 {
                exec.moe_align_bm(
                    idx,
                    srow,
                    sslot,
                    bexp,
                    batch,
                    dims.n_active,
                    dims.n_expert,
                    bm,
                    max_blocks,
                )?;
            } else {
                exec.moe_align(
                    idx,
                    srow,
                    sslot,
                    bexp,
                    batch,
                    dims.n_active,
                    dims.n_expert,
                    max_blocks,
                )?;
            }
            match (w.gate_exps.kq(), w.up_exps.kq()) {
                (Some(g), Some(u)) => {
                    let needs = crate::gpu::kq_needs_sums(g.ty);
                    if needs {
                        exec.q8_sums_strided(xq, ssums, embd, batch)?;
                    }
                    exec.kquant_moe_gate_up_mma(
                        g,
                        u,
                        srow,
                        bexp,
                        xq,
                        xs,
                        needs.then_some(&*ssums),
                        fq,
                        fs,
                        max_blocks,
                    )?;
                }
                _ => {
                    let g8 = w.gate_exps.q8().expect("loader pairs gate/up residency");
                    let u8_ = w.up_exps.q8().expect("loader pairs gate/up residency");
                    if mma {
                        exec.q8_0_moe_gate_up_mma(
                            g8, u8_, srow, bexp, xq, xs, fq, fs, max_blocks, bm,
                        )?;
                    } else {
                        // dp4a fallback is BM=32 only (bm==32 whenever mma is off).
                        exec.q8_0_moe_gate_up_sorted(
                            g8, u8_, srow, bexp, xq, xs, fused, max_blocks,
                        )?;
                        exec.quantize_q8(fused, fq, fs, max_blocks * 32 * dims.moe_ff)?;
                    }
                }
            }
            match w.down_exps.kq() {
                Some(d) => {
                    let needs = crate::gpu::kq_needs_sums(d.ty);
                    if needs {
                        // sums over the SORTED fq rows (PAD rows hold exact zeros)
                        exec.q8_sums_strided(fq, ssums, dims.moe_ff, max_blocks * 32)?;
                    }
                    exec.kquant_moe_down_mma(
                        d,
                        srow,
                        sslot,
                        bexp,
                        topk_w,
                        fq,
                        fs,
                        needs.then_some(&*ssums),
                        part,
                        dims.n_active,
                        max_blocks,
                    )?;
                }
                None => {
                    let d8 = w
                        .down_exps
                        .q8()
                        .expect("an expert seat is Q8 when not k-quant");
                    if mma {
                        exec.q8_0_moe_down_mma(
                            d8,
                            srow,
                            sslot,
                            bexp,
                            topk_w,
                            fq,
                            fs,
                            part,
                            dims.n_active,
                            max_blocks,
                            bm,
                        )?;
                    } else {
                        exec.q8_0_moe_down_sorted(
                            d8,
                            srow,
                            sslot,
                            bexp,
                            topk_w,
                            fq,
                            fs,
                            part,
                            dims.n_active,
                            max_blocks,
                        )?;
                    }
                }
            }
        } // end q8-vs-fp4 sorted branch
        exec.stream
            .memset_zeros(proj)
            .map_err(|e| GpuError::Driver(e.to_string()))?;
        // bf16 partials (PADDOCK_MOE_PART_BF16, PPL-gated trade): the down
        // launcher stores bf16 under the same rows>=256 gate, so both sides
        // flip together; only the fp4 sorted path writes partials this tick
        // shape. Q8/dp4a fallbacks keep f32 (env unset there in practice -
        // guard on the fp4 branch would be dead code since fp4 is the
        // serving prefill path whenever these planes exist).
        if fp4_used
            && batch >= 256
            && exec.has_moe_slot_combine_bf16()
            && std::env::var_os("PADDOCK_MOE_PART_BF16").is_some()
        {
            exec.moe_slot_combine_bf16(part, proj, embd, dims.n_active, batch)?;
        } else {
            exec.moe_slot_combine(part, proj, embd, dims.n_active, batch)?;
        }
    } else {
        // Token-batched pair, per-seat dispatch. The k-quant arm is the same
        // numeric class (exact int8 dots, f32 block scales) with the mu term
        // riding per-16 activation sums; ssums is reused across the two
        // stages (stream-ordered: gate_up consumes the xq sums before the
        // down stage overwrites with fq sums).
        // MoE expert offload: when the layer carries a slot cache and this
        // launch's rows fit it, resolve ids -> slots (LRU, device-side), fill
        // the misses from the host mirror, and run the pair over the slot
        // planes with the remapped ids. Otherwise the host-mapped planes
        // serve directly (zero-copy).
        let rows = batch * dims.n_active;
        let cache = w
            .cache
            .as_ref()
            .filter(|c| rows <= c.slots && rows <= c.max_rows);
        if let Some(c) = cache {
            exec.moe_cache_resolve(c, idx, rows)?;
            exec.moe_cache_fill(c, rows)?;
        }
        let idx: &CudaSlice<u32> = cache.map_or(idx, |c| c.idx_slot());
        let seats = match cache {
            Some(c) => (Some(&c.gate), Some(&c.up), Some(&c.down)),
            None => (w.gate_exps.kq(), w.up_exps.kq(), w.down_exps.kq()),
        };
        match (seats.0, seats.1) {
            (Some(g), Some(u)) => {
                let needs = crate::gpu::kq_needs_sums(g.ty) || crate::gpu::kq_needs_sums(u.ty);
                if needs {
                    exec.q8_sums_strided(xq, ssums, embd, batch)?;
                }
                exec.kquant_moe_gate_up(
                    g,
                    u,
                    idx,
                    xq,
                    xs,
                    needs.then_some(&*ssums),
                    fused,
                    dims.n_active,
                    batch,
                )?;
            }
            _ => {
                let g8 = w.gate_exps.q8().expect("loader pairs gate/up residency");
                let u8_ = w.up_exps.q8().expect("loader pairs gate/up residency");
                exec.q8_0_moe_gate_up(g8, u8_, idx, xq, xs, fused, dims.n_active, batch)?;
            }
        }
        exec.quantize_q8(fused, fq, fs, batch * dims.n_active * dims.moe_ff)?;
        match seats.2 {
            Some(d) => {
                let needs = crate::gpu::kq_needs_sums(d.ty);
                if needs {
                    exec.q8_sums_strided(fq, ssums, dims.moe_ff, batch * dims.n_active)?;
                }
                exec.kquant_moe_down(
                    d,
                    idx,
                    topk_w,
                    fq,
                    fs,
                    needs.then_some(&*ssums),
                    proj,
                    dims.n_active,
                    batch,
                )?;
            }
            None => {
                let d = w
                    .down_exps
                    .q8()
                    .expect("an expert seat is Q8 when not k-quant");
                exec.q8_0_moe_down(d, idx, topk_w, fq, fs, proj, dims.n_active, batch)?;
            }
        }
    }
    // shared expert: a plain dense SwiGLU FFN, folded in behind
    // sigmoid(x . shexp_gate_inp). Three regimes, all measured:
    // - batch 1 keeps the decode gemv (graph-captured b=1 step, peak
    //   single-row BW);
    // - 2..=64 rides the mmq()/ks serving ladder - the repacked_mt class
    //   these shapes first shipped on measured 13.2 us per [2048->512] mat
    //   at B=8 (~80 GB/s, 3 mats x 40 layers = 1.6 ms/step, most of the
    //   B=4/8 gap vs llama);
    // - >64 rides the dense models' int8 prefill classes (mm()'s
    //   gemm_repacked measured 1.57 ms avg at 512 rows before that fix).
    // The ks partial planes borrow `part` - disjoint from the sorted
    // branch's use (sorted engages only past sorted_min >= 128 rows).
    if batch > 64 {
        prefill_quant(exec, pxq, pxs, yq, xn, embd, batch)?;
        shexp_mm_wide(
            exec,
            &w.shexp_gate,
            pxq,
            pxs,
            yq,
            ssums,
            skfix,
            shexp_gate,
            batch,
        )?;
        shexp_mm_wide(
            exec,
            &w.shexp_up,
            pxq,
            pxs,
            yq,
            ssums,
            skfix,
            shexp_up,
            batch,
        )?;
        exec.swiglu(shexp_gate, shexp_up, batch * dims.shexp_ff)?;
        match &w.shexp_down {
            QuantW::Q8(q) => {
                prefill_mm(exec, q, pxq, pxs, yq, skfix, shexp_gate, shexp_out, batch)?
            }
            QuantW::Kq(_) => {
                prefill_quant(exec, pxq, pxs, yq, shexp_gate, dims.shexp_ff, batch)?;
                shexp_mm_wide(
                    exec,
                    &w.shexp_down,
                    pxq,
                    pxs,
                    yq,
                    ssums,
                    skfix,
                    shexp_out,
                    batch,
                )?;
            }
        }
    } else if batch > 1 {
        exec.quantize_q8(xn, pxq, pxs, batch * embd)?;
        mmq_pre_any(
            exec,
            &w.shexp_gate,
            pxq,
            pxs,
            ssums,
            part,
            shexp_gate,
            batch,
        )?;
        mmq_pre_any(exec, &w.shexp_up, pxq, pxs, ssums, part, shexp_up, batch)?;
        exec.swiglu(shexp_gate, shexp_up, batch * dims.shexp_ff)?;
        exec.quantize_q8(shexp_gate, pxq, pxs, batch * dims.shexp_ff)?;
        mmq_pre_any(exec, &w.shexp_down, pxq, pxs, ssums, part, shexp_out, batch)?;
    } else {
        // b=1 gate|up launch merge (entry 317): two [2048 -> shexp_ff] GEMVs
        // sit at the small-gemv latency floor (~4 us each on sm_120a for
        // ~1 MB of bytes); one merged launch pays the toll once. Bit-identical
        // per row to the splits. down reads the swiglu output, so it cannot
        // join. Kill: PADDOCK_NO_GEMV_MULTI. Q8_0 seats only - the k-quant
        // GEMV has no merged form.
        match (&w.shexp_gate, &w.shexp_up) {
            (QuantW::Q8(g), QuantW::Q8(u))
                if exec.has_q8_0_gemv_repacked_multi() && !no_gemv_multi() =>
            {
                exec.q8_0_gemv_repacked_multi(
                    &mut [(g, &mut *shexp_gate), (u, &mut *shexp_up)],
                    xn,
                )?;
            }
            _ => {
                gemv_any(exec, &w.shexp_gate, xn, shexp_gate)?;
                gemv_any(exec, &w.shexp_up, xn, shexp_up)?;
            }
        }
        exec.swiglu(shexp_gate, shexp_up, batch * dims.shexp_ff)?;
        gemv_any(exec, &w.shexp_down, shexp_gate, shexp_out)?;
    }
    exec.shexp_gate_add(
        proj,
        shexp_out,
        xn,
        &w.shexp_gate_inp.buf,
        embd,
        embd,
        batch,
    )?;
    Ok(())
}

/// DeltaNet recurrence for a prefill span. Spans of 128+ tokens take the
/// chunked scan (P6l): the same gated delta rule with only ceil(r/64)
/// sequential state hops - Not bit-identical to v2 (different accumulation
/// structure; held to the CPU-oracle parity test and the greedy gates), and
/// measured faster from T=128 up (1.27x at T=512). Shorter spans keep the
/// exact v2 recurrence; decode and speculative paths call v2 directly and are
/// untouched.
#[allow(clippy::too_many_arguments)]
pub(super) fn prefill_delta_recurrent(
    exec: &GpuExecutor,
    sc: &mut Scratch,
    states: &mut CudaSlice<f32>,
    state_elem_off: usize,
    r: usize,
    n_v_heads: usize,
    state_size: usize,
    vb16: bool,
) -> Result<(), GpuModelError> {
    // PADDOCK_NO_CHUNKED_DN=1 pins the exact v2 recurrence (A/B + attribution)
    let no_chunked = paddock_models::dev_var_os!("PADDOCK_NO_CHUNKED_DN").is_some();
    // The dispatch boundary is die-dependent: the A6000 measured chunked faster
    // from T=128 up (1.27x at 512), but on the 188-SM GB202 the sequential v2
    // kernel gains more from the bigger die than the two-stage scan does -
    // kbench crossover sits at T=384 there (chunked 0.60x at T=128, 0.99x at
    // 384, 1.09x at 512, 1.24x at 2048). Small dies keep the measured 128.
    let chunk_min = if exec.sm_count() >= 128 { 384 } else { 128 };
    if r >= chunk_min && state_size == 128 && !no_chunked {
        if vb16 {
            // v arrived bf16 from conv_qkv_b16 - the dn_vb16 gate mirrors
            // this dispatch exactly, so the sequential fallback (f32 v)
            // can never fire on a vb16 call
            exec.gated_delta_chunked_vb16(
                &sc.d_dq,
                &sc.d_dk,
                &sc.d_dv,
                &sc.d_g,
                &sc.d_beta,
                states,
                state_elem_off,
                &mut sc.d_dattn,
                &mut sc.d_dnc_dw,
                &mut sc.d_dnc_du,
                &mut sc.d_dnc_coef,
                &mut sc.d_dnc_cg,
                r,
                n_v_heads,
                state_size,
            )?;
        } else {
            exec.gated_delta_chunked(
                &sc.d_dq,
                &sc.d_dk,
                &sc.d_dv,
                &sc.d_g,
                &sc.d_beta,
                states,
                state_elem_off,
                &mut sc.d_dattn,
                &mut sc.d_dnc_dw,
                &mut sc.d_dnc_du,
                &mut sc.d_dnc_coef,
                &mut sc.d_dnc_cg,
                r,
                n_v_heads,
                state_size,
            )?;
        }
    } else {
        exec.gated_delta_recurrent_v2(
            &sc.d_dq,
            &sc.d_dk,
            &sc.d_dv,
            &sc.d_g,
            &sc.d_beta,
            None,
            states,
            state_elem_off,
            None,
            &mut sc.d_dattn,
            1,
            r,
            n_v_heads,
            state_size,
        )?;
    }
    Ok(())
}

/// Quantize prefill activations once for a run of matmuls sharing the same
/// input (wq/wk/wv, in_qkv+gate_w, ffn gate+up used to re-quantize identical
/// bytes per call - P6j dedup). Layout picked by the same batch>64 rule the
/// GEMM half uses: >64 -> flat mmq layout, else strided int8.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_quant(
    exec: &GpuExecutor,
    xq: &mut CudaSlice<i8>,
    xs: &mut CudaSlice<f32>,
    yq: &mut CudaSlice<u8>,
    x: &CudaSlice<f32>,
    in_dim: usize,
    batch: usize,
) -> Result<(), GpuModelError> {
    if batch > 64 {
        exec.quantize_q8_mmq(x, yq, in_dim, batch)?;
        // A pack without the i-quant tile rung (slot 585) keeps a dense
        // i-quant plane on the row-major dp4a walk at every width
        // (kq_mm_pre), so such a file pays one extra memory-bound pass here.
        // Measured 2026-09-06 without it: the >64-row route read stale xq
        // and a 110-token prompt decoded to gibberish.
        if iq_rowmajor_needed(exec) {
            exec.quantize_q8(x, xq, xs, batch * in_dim)?;
        }
    } else {
        exec.quantize_q8(x, xq, xs, batch * in_dim)?;
    }
    Ok(())
}

/// GEMM half of [`prefill_mm`] for activations already quantized by
/// [`prefill_quant`]. mmq-class route (P6e) for batch > 64: flat activation
/// layout + the K-256/ntx=2 one-block-per-SM GEMM with stream-k on
/// low-tile-count launches - same numeric class as the mma route (bit-exact
/// when tiled), faster at every batch > 64 measured.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_mm_pre(
    exec: &GpuExecutor,
    w: &RepackedQ8,
    xq: &CudaSlice<i8>,
    xs: &CudaSlice<f32>,
    yq: &CudaSlice<u8>,
    skfix: &mut CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    batch: usize,
) -> Result<(), GpuModelError> {
    // Shared-helper stub guard: the per-call-site guards catch the sites we
    // KNEW about, and twice now the bug has been a site nobody listed. This
    // one cannot be missed -- every Q8 prefill GEMM funnels here. It names the
    // dims so the caller is identifiable from the refusal alone.
    if w.data.len() == 32 && w.dims.iter().product::<usize>() > 32 {
        return Err(GpuModelError::Unsupported(format!(
            "qwen35 prefill: a Q8_0 plane {:?} was reclaimed to a 32-byte stub \
             but reached {} -- an UNGUARDED reader the REPLACE audit missed. \
             Re-run with PADDOCK_QWEN35_W8_MIN=64 to keep the planes resident.",
            w.dims, "prefill_mm_pre",
        )));
    }
    prefill_mm_pre_sk(exec, w, xq, xs, yq, skfix, None, y, batch)
}

/// Producer/consumer window for the prefill mcol rung: rows in
/// `65..=pf_mma_rows_max()` stage strided int8 (`prefill_quant_w`) and ride
/// the mcol mma GEMM (`prefill_mm_pre_p`); everything else keeps the classic
/// layout/rungs. One source for both sides - a producer/consumer disagreement
/// here is silent activation garbage - mcol fed by a yq-only staging
/// returned fluent word salad at r=177 while every gate stayed green,
/// because a throughput benchmark never reads the text.
/// PADDOCK_NO_PF_MCOL=1 collapses the window to 64 = exactly the old paths.
pub(crate) fn pf_mma_rows_max() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        if paddock_models::dev_var_os!("PADDOCK_NO_PF_MCOL").is_some() {
            64
        } else {
            192
        }
    })
}

/// [`prefill_quant`]'s window-aware twin for callers on the mcol contract.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_quant_w(
    exec: &GpuExecutor,
    xq: &mut CudaSlice<i8>,
    xs: &mut CudaSlice<f32>,
    yq: &mut CudaSlice<u8>,
    x: &CudaSlice<f32>,
    in_dim: usize,
    batch: usize,
) -> Result<(), GpuModelError> {
    if batch > pf_mma_rows_max() {
        exec.quantize_q8_mmq(x, yq, in_dim, batch)?;
    } else {
        exec.quantize_q8(x, xq, xs, batch * in_dim)?;
    }
    Ok(())
}

/// [`prefill_mm_pre`] for callers staging via [`prefill_quant_w`]: the
/// 65..=192 window takes the mcol mma rung (weights staged once, 64-token
/// column tiles looped in-register) instead of the 128x128-tile base mmq at
/// ~250 GB/s - MEASURED (bench/granite_decode_kern_bench.cu, sm_120a,
/// b=128): wq 29.1us vs 76.3 (2.6x), wkv 20.5 vs 75.8 (3.7x), down 67.7 vs
/// 229.1 (3.4x), gate 71.7 vs 79.1 - a 128-row granite-8b admission pass
/// drops ~27.7ms -> ~12.4ms, which under wide-batch churn was ~25% of all
/// GPU time. Deliberately not inside `prefill_mm_pre_sk`: qwen3 passes
/// its own partials plane there with yq-only staging, and a plane-keyed
/// branch would feed mcol stale xq/xs for that family.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_mm_pre_p(
    exec: &GpuExecutor,
    w: &RepackedQ8,
    xq: &CudaSlice<i8>,
    xs: &CudaSlice<f32>,
    yq: &CudaSlice<u8>,
    skfix: &mut CudaSlice<f32>,
    part: &mut CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    batch: usize,
) -> Result<(), GpuModelError> {
    if batch > 64 && batch <= pf_mma_rows_max() {
        // same stub-reclaim guard as the _sk funnel - this entry bypasses it
        if w.data.len() == 32 && w.dims.iter().product::<usize>() > 32 {
            return Err(GpuModelError::Unsupported(format!(
                "qwen35 prefill: a Q8_0 plane {:?} was reclaimed to a 32-byte stub \
                 but reached {} -- an UNGUARDED reader the REPLACE audit missed.",
                w.dims, "prefill_mm_pre_p",
            )));
        }
        // Loud on capacity: the producer already staged xq/xs for this window,
        // so a silent mmq fallback would read stale yq - the exact corruption
        // this contract exists to prevent.
        if part.len() < 8 * 192 * w.dims[1] {
            return Err(GpuModelError::Unsupported(format!(
                "pf mcol: partials plane {} f32 < 8*192*{} needed for {:?} - \
                 grow the plane or pin PADDOCK_NO_PF_MCOL=1",
                part.len(),
                w.dims[1],
                w.dims,
            )));
        }
        exec.q8_0_gemm_mma_ks(w, xq, xs, part, y, batch)?;
        return Ok(());
    }
    prefill_mm_pre_sk(exec, w, xq, xs, yq, skfix, None, y, batch)
}

/// `prefill_mm_pre` with an optional split-K partials plane: narrow tiled
/// GEMMs at serving batches leave a big last-wave tail (a 1024-out
/// projection at ~4k rows is 232 blocks = 1.23 waves at 1 block/SM); with
/// a partials plane the tail tiles split over K (Stream-K lite) instead of
/// rounding up to a whole wave. Callers without the plane (the generative
/// models) keep the plain kernels.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_mm_pre_sk(
    exec: &GpuExecutor,
    w: &RepackedQ8,
    xq: &CudaSlice<i8>,
    xs: &CudaSlice<f32>,
    yq: &CudaSlice<u8>,
    skfix: &mut CudaSlice<f32>,
    partials: Option<&mut CudaSlice<f32>>,
    y: &mut CudaSlice<f32>,
    batch: usize,
) -> Result<(), GpuModelError> {
    // Shared-helper stub guard: the per-call-site guards catch the sites we
    // KNEW about, and twice now the bug has been a site nobody listed. This
    // one cannot be missed -- every Q8 prefill GEMM funnels here. It names the
    // dims so the caller is identifiable from the refusal alone.
    if w.data.len() == 32 && w.dims.iter().product::<usize>() > 32 {
        return Err(GpuModelError::Unsupported(format!(
            "qwen35 prefill: a Q8_0 plane {:?} was reclaimed to a 32-byte stub \
             but reached {} -- an UNGUARDED reader the REPLACE audit missed. \
             Re-run with PADDOCK_QWEN35_W8_MIN=64 to keep the planes resident.",
            w.dims, "prefill_mm_pre_sk",
        )));
    }
    if batch > mmq_hi_min_batch()
        && w.dims[0].is_multiple_of(128)
        && exec.has_q8_0_gemm_mmq_pipe()
        && let (Some(p), true) = (partials, exec.has_q8_0_gemm_mmq_pipe_sk())
        && paddock_models::dev_var_os!("PADDOCK_NO_SK").is_none()
    {
        // the launcher itself decides engage-vs-plain from the tail size
        exec.q8_0_gemm_mmq_pipe_sk(w, yq, p, y, batch)?;
        return Ok(());
    }
    // (pipe64 - a 64-deep-K-stage, 2-block/SM variant built to fix last-wave
    // rounding on small grids - was MEASURED NEGATIVE: the halved prefetch
    // depth costs more per block than the recovered wave, 1119 vs 1171 docs/s
    // on the 64-doc reranker shape. Wave quantization on this kernel family is
    // cheaper than shallow pipelines, so it was removed.)
    if batch > mmq_hi_min_batch() && w.dims[0].is_multiple_of(128) && exec.has_q8_0_gemm_mmq_pipe()
    {
        exec.q8_0_gemm_mmq_pipe(w, None, yq, y, batch)?;
    } else if batch > mmq_hi_min_batch() && exec.has_q8_0_gemm_mmq_hi() {
        exec.q8_0_gemm_mmq_hi(w, yq, y, batch)?;
    } else if batch > 64 {
        exec.q8_0_gemm_mmq(w, yq, Some(skfix), y, batch)?;
    } else {
        exec.q8_0_gemm_mma(w, xq, xs, y, batch)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_mm(
    exec: &GpuExecutor,
    w: &RepackedQ8,
    xq: &mut CudaSlice<i8>,
    xs: &mut CudaSlice<f32>,
    yq: &mut CudaSlice<u8>,
    skfix: &mut CudaSlice<f32>,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    batch: usize,
) -> Result<(), GpuModelError> {
    // Shared-helper stub guard: the per-call-site guards catch the sites we
    // KNEW about, and twice now the bug has been a site nobody listed. This
    // one cannot be missed -- every Q8 prefill GEMM funnels here. It names the
    // dims so the caller is identifiable from the refusal alone.
    if w.data.len() == 32 && w.dims.iter().product::<usize>() > 32 {
        return Err(GpuModelError::Unsupported(format!(
            "qwen35 prefill: a Q8_0 plane {:?} was reclaimed to a 32-byte stub \
             but reached {} -- an UNGUARDED reader the REPLACE audit missed. \
             Re-run with PADDOCK_QWEN35_W8_MIN=64 to keep the planes resident.",
            w.dims, "prefill_mm",
        )));
    }
    prefill_quant(exec, xq, xs, yq, x, w.dims[0], batch)?;
    prefill_mm_pre(exec, w, xq, xs, yq, skfix, y, batch)
}

/// The k-quant W4A8 GEMM half for activations already quantized by
/// [`prefill_quant`]'s batch rule: > 64 rows sit in the flat mmq layout
/// (`yq`) and take the 128x128 int8-MMA tile; <= 64 sit in the strided
/// layout (`xq`/`xs`) and take the dp4a decode-shape GEMM. Both are the
/// same numeric class as the Q8 ladder (exact int dots, f32 scales).
///
/// The >64 tile prefers the genuinely-double-buffered `pipe2` rung (2-deep
/// raw byte ring, half-width tile_x) over the single-buffer pipelined rung
/// when the pack carries it: `pipe2`'s next-chunk load overlaps the
/// CURRENT chunk's entire build+compute phase, where the plain pipe kernel
/// can only overlap it behind compute (its single raw buffer can't be
/// refilled until the current chunk's unpack finishes reading it). Both
/// stay `__launch_bounds__(256,1)` - a 2-blocks/SM tile hit its register
/// target but profiling showed sm_120's SM shared-memory budget blocks
/// occupancy from rising regardless (llama.cpp's own k-quant
/// kernel doesn't pipeline at all and targets the same occupancy=1,
/// confirmed by reading `ggml-cuda/mmq.cuh` directly - occupancy=1 is the
/// settled floor on both engines here, so `pipe2` targets the *stall*
/// inside that one resident block instead of the block count). Falls back
/// to the plain pipelined rung, then `kquant_gemm_w4a8`'s synchronous load
/// (79.6% of granite-30b's prefill GPU time before any of this, the same
/// register-bound, no-latency-hiding profile Q8_0's own mmq kernel had
/// before it grew `_hi`/`_pipe`). `PADDOCK_NO_KQUANT_PIPE2=1` /
/// `PADDOCK_NO_KQUANT_PIPE=1` pin earlier rungs for A/B, mirroring
/// `PADDOCK_NO_SK`/`PADDOCK_MMQ_HI_MIN`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn kq_mm_pre(
    exec: &GpuExecutor,
    k: &RepackedKQ,
    xq: &CudaSlice<i8>,
    xs: &CudaSlice<f32>,
    yq: &CudaSlice<u8>,
    xsums: &mut CudaSlice<f32>,
    ssums: &mut CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    batch: usize,
) -> Result<(), GpuModelError> {
    let needs = crate::gpu::kq_needs_sums(k.ty);
    // An i-quant plane rides the tile rung only when the pack's pipe2 builds
    // its tile through the i-quant window unpack (slot 585); a pack without
    // it refuses the type with cudaErrorInvalidValue, and the plane stays
    // on the dp4a walk at every width - correct, but re-reading the plane
    // once per token (any prompt past 64 tokens died with "CUDA error 1"
    // before the guard, 2026-09-06, Qwen3.5-9B UD-IQ2_XXS).
    if batch > 64 && kq_tile_ok(exec, k) {
        if needs {
            exec.mmq_sums(yq, xsums, k.dims[0], batch)?;
        }
        if exec.has_kquant_gemm_w4a8_pipe2()
            && paddock_models::dev_var_os!("PADDOCK_NO_KQUANT_PIPE2").is_none()
        {
            exec.kquant_gemm_w4a8_pipe2(k, yq, needs.then_some(&*xsums), y, batch)?;
        } else if exec.has_kquant_gemm_w4a8_pipe()
            && paddock_models::dev_var_os!("PADDOCK_NO_KQUANT_PIPE").is_none()
        {
            exec.kquant_gemm_w4a8_pipe(k, yq, needs.then_some(&*xsums), y, batch)?;
        } else {
            exec.kquant_gemm_w4a8(k, yq, needs.then_some(&*xsums), y, batch)?;
        }
    } else {
        if needs {
            exec.q8_sums_strided(xq, ssums, k.dims[0], batch)?;
        }
        exec.kquant_gemm_dp4a(k, xq, xs, needs.then_some(&*ssums), y, batch)?;
    }
    Ok(())
}

/// The plane may take the >64-row W4A8 tile: every k-quant type, and the
/// i-quant family when the pack's tile builds through the window unpack
/// (slot 585).
pub(crate) fn kq_tile_ok(exec: &GpuExecutor, k: &RepackedKQ) -> bool {
    // IQ4_NL rows lie flat at any multiple of 32; the tile walks whole
    // super-blocks, so such a width stays on the dp4a lane
    !crate::gpu::kq_is_iq(k.ty) || (exec.has_kquant_iq_tile() && k.dims[0].is_multiple_of(256))
}

/// True when a dense i-quant plane was loaded and the pack cannot tile it:
/// the >64-row producers then keep the row-major int8 pair for the dp4a
/// walk as well as the mmq tiles.
pub(crate) fn iq_rowmajor_needed(exec: &GpuExecutor) -> bool {
    // a flat-row plane (IQ4_NL at a non-256 width) stays on dp4a whatever the pack
    exec.dense_iq_seen() && (!exec.has_kquant_iq_tile() || exec.dense_iq_flat_seen())
}

/// QuantW dispatch over [`prefill_mm_pre`] - Q8_0 keeps its exact ladder,
/// k-quant rides [`kq_mm_pre`] off the same staged activations.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_mm_pre_any(
    exec: &GpuExecutor,
    w: &QuantW,
    xq: &CudaSlice<i8>,
    xs: &CudaSlice<f32>,
    yq: &CudaSlice<u8>,
    xsums: &mut CudaSlice<f32>,
    ssums: &mut CudaSlice<f32>,
    skfix: &mut CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    batch: usize,
) -> Result<(), GpuModelError> {
    match w {
        QuantW::Q8(q) => prefill_mm_pre(exec, q, xq, xs, yq, skfix, y, batch),
        QuantW::Kq(k) => kq_mm_pre(exec, k, xq, xs, yq, xsums, ssums, y, batch),
    }
}

/// QuantW dispatch over [`prefill_mm`] (quantize + GEMM).
#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_mm_any(
    exec: &GpuExecutor,
    w: &QuantW,
    xq: &mut CudaSlice<i8>,
    xs: &mut CudaSlice<f32>,
    yq: &mut CudaSlice<u8>,
    xsums: &mut CudaSlice<f32>,
    ssums: &mut CudaSlice<f32>,
    skfix: &mut CudaSlice<f32>,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    batch: usize,
) -> Result<(), GpuModelError> {
    match w {
        QuantW::Q8(q) => prefill_mm(exec, q, xq, xs, yq, skfix, x, y, batch),
        QuantW::Kq(k) => {
            prefill_quant(exec, xq, xs, yq, x, k.dims[0], batch)?;
            kq_mm_pre(exec, k, xq, xs, yq, xsums, ssums, y, batch)
        }
    }
}

/// QuantW dispatch over [`prefill_ffn_down`] (swiglu + quantize + GEMM; the
/// k-quant big-batch arm keeps the P6j swiglu-fused mmq quantize).
#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_ffn_down_any(
    exec: &GpuExecutor,
    w: &QuantW,
    xq: &mut CudaSlice<i8>,
    xs: &mut CudaSlice<f32>,
    yq: &mut CudaSlice<u8>,
    xsums: &mut CudaSlice<f32>,
    ssums: &mut CudaSlice<f32>,
    skfix: &mut CudaSlice<f32>,
    gate: &mut CudaSlice<f32>,
    up: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    ff: usize,
    batch: usize,
) -> Result<(), GpuModelError> {
    match w {
        QuantW::Q8(q) => prefill_ffn_down(exec, q, xq, xs, yq, skfix, gate, up, y, ff, batch),
        QuantW::Kq(k) => {
            // the fused swiglu+quantize writes ONLY the mmq tiles; a down
            // plane the pack cannot tile reads the row-major pair (kq_mm_pre),
            // so it takes the two-step form at every width - this was the
            // site that fed stale xq to a 110-token prompt and decoded
            // gibberish (2026-09-06)
            if batch > 64 && kq_tile_ok(exec, k) {
                exec.quantize_q8_mmq_swiglu(gate, up, yq, k.dims[0], batch)?;
            } else {
                exec.swiglu(gate, up, batch * ff)?;
                prefill_quant(exec, xq, xs, yq, gate, k.dims[0], batch)?;
            }
            kq_mm_pre(exec, k, xq, xs, yq, xsums, ssums, y, batch)
        }
    }
}

/// FFN tail (P6j): for batch > 64 the SwiGLU fuses into the mmq quantize
/// feeding the down GEMM - gate/up are read once and the f32 activation
/// never lands in memory. Small batches keep the separate swiglu +
/// prefill_mm route. Values are bit-identical either way.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_ffn_down(
    exec: &GpuExecutor,
    w: &RepackedQ8,
    xq: &mut CudaSlice<i8>,
    xs: &mut CudaSlice<f32>,
    yq: &mut CudaSlice<u8>,
    skfix: &mut CudaSlice<f32>,
    gate: &mut CudaSlice<f32>,
    up: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    ff: usize,
    batch: usize,
) -> Result<(), GpuModelError> {
    prefill_ffn_down_sk(exec, w, xq, xs, yq, skfix, None, gate, up, y, ff, batch)
}

/// `prefill_ffn_down` with the optional split-K partials plane (see
/// `prefill_mm_pre_sk`) - the down projection is the other narrow GEMM
/// that loses a large last-wave slice at serving batches.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_ffn_down_sk(
    exec: &GpuExecutor,
    w: &RepackedQ8,
    xq: &mut CudaSlice<i8>,
    xs: &mut CudaSlice<f32>,
    yq: &mut CudaSlice<u8>,
    skfix: &mut CudaSlice<f32>,
    partials: Option<&mut CudaSlice<f32>>,
    gate: &mut CudaSlice<f32>,
    up: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    ff: usize,
    batch: usize,
) -> Result<(), GpuModelError> {
    if batch > 64 {
        exec.quantize_q8_mmq_swiglu(gate, up, yq, w.dims[0], batch)?;
        if batch > mmq_hi_min_batch()
            && w.dims[0].is_multiple_of(128)
            && exec.has_q8_0_gemm_mmq_pipe()
            && let (Some(p), true) = (partials, exec.has_q8_0_gemm_mmq_pipe_sk())
            && paddock_models::dev_var_os!("PADDOCK_NO_SK").is_none()
        {
            exec.q8_0_gemm_mmq_pipe_sk(w, yq, p, y, batch)?;
            return Ok(());
        }
        if batch > mmq_hi_min_batch()
            && w.dims[0].is_multiple_of(128)
            && exec.has_q8_0_gemm_mmq_pipe()
        {
            exec.q8_0_gemm_mmq_pipe(w, None, yq, y, batch)?;
        } else if batch > mmq_hi_min_batch() && exec.has_q8_0_gemm_mmq_hi() {
            exec.q8_0_gemm_mmq_hi(w, yq, y, batch)?;
        } else {
            exec.q8_0_gemm_mmq(w, yq, Some(skfix), y, batch)?;
        }
    } else {
        exec.swiglu(gate, up, batch * ff)?;
        prefill_quant(exec, xq, xs, yq, gate, w.dims[0], batch)?;
        prefill_mm_pre(exec, w, xq, xs, yq, skfix, y, batch)?;
    }
    Ok(())
}

/// Fused residual-add + rmsnorm + quantize for the prefill norm sites
/// (P6k): batch > 64 runs one kernel (x gets the residual update, yq the
/// quantized normalized row; xn is materialized only when `write_xn` -
/// i.e. where alpha/beta still read it). Small batches keep the separate
/// kernels. Values are bit-identical either way.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill_add_norm_quant(
    exec: &GpuExecutor,
    x: &mut CudaSlice<f32>,
    proj: Option<&CudaSlice<f32>>,
    proj_b16: bool,
    w: &CudaSlice<f32>,
    xn: &mut CudaSlice<f32>,
    write_xn: bool,
    xq: &mut CudaSlice<i8>,
    xs: &mut CudaSlice<f32>,
    yq: &mut CudaSlice<u8>,
    n: usize,
    batch: usize,
    eps: f32,
) -> Result<(), GpuModelError> {
    if batch > 64 && n.is_multiple_of(4) && n <= 24576 {
        // The fused kernel writes the mmq tiles (yq) only. A dense i-quant
        // plane the pack cannot tile reads the row-major pair downstream
        // (kq_mm_pre), so keep xn and quantize it as well - the width bisect
        // on Qwen3.5-9B UD-IQ2_XXS (2026-09-06) passed at 64 rows and decoded
        // gibberish at 65, and this was the site.
        let iq = iq_rowmajor_needed(exec);
        let xn_opt = if write_xn || iq { Some(&mut *xn) } else { None };
        exec.add_rmsnorm_quant_mmq(x, proj, proj_b16, w, xn_opt, yq, n, batch, eps)?;
        if iq {
            exec.quantize_q8(xn, xq, xs, batch * n)?;
        }
    } else {
        // The mmq route above consumes `proj_b16` itself. This fallback used to
        // assume it never saw one -- the assumption was a `debug_assert!`, which
        // release builds strip, so when it finally broke it produced silent
        // garbage instead of a panic ('arcyisisisis...' from every prompt).
        //
        // What broke it: the mixer's o16 (bf16-epilogue) projection
        // arm in unified_launch_core is `lw8`-gated, and lw8 was gated
        // `r > w8_min_batch()` = 64 while unified caps at unified_prefill_rows()
        // = 64 -- so a bf16 residual could only ever reach the `batch > 64`
        // route. Lowering the projection floor for the 7.4 GB REPLACE made the
        // mixer write bf16 at r <= 64 for the first time, and this branch added
        // it as f32. The kernel was never at fault: pd_f8_gemm_lin_kt with
        // o16=1 is numerically clean at batch 1..2048 on every projection shape
        // (bench/qwen35_proj_layout_bench.cu, max-rel 0.02-0.04 vs q8).
        //
        // So honour the flag here too rather than re-asserting an invariant that
        // is no longer true. A hard error if the kernel is missing: the caller
        // only sets proj_b16 after checking has_f8_o16(), and silently adding
        // bf16 as f32 is exactly the failure this comment exists to prevent.
        if let Some(p) = proj {
            if proj_b16 {
                if !exec.has_add_b16() {
                    return Err(GpuModelError::Unsupported(
                        "qwen35 prefill: a bf16 residual reached the non-mmq \
                         add+norm route but the pack has no add_inplace_b16 -- \
                         adding it as f32 would corrupt silently"
                            .into(),
                    ));
                }
                exec.add_b16(x, p, batch * n)?;
            } else {
                exec.add(x, p, batch * n)?;
            }
        }
        exec.rmsnorm_batch(x, w, xn, n, eps, batch)?;
        prefill_quant(exec, xq, xs, yq, xn, n, batch)?;
    }
    Ok(())
}

/// Index of the max logit (greedy next token).
pub(crate) fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

/// Read the `rope.dimension_sections` [t,h,w,e] array (there is no `as_array`
/// helper - match the `Value::Array` directly). Missing/short => all-temporal.
pub(crate) fn read_sections(map: &MappedGguf) -> Result<[u32; 4], GpuModelError> {
    let mut out = [0u32; 4];
    if let Some(Value::Array(items)) = map.gguf().arch_field("rope.dimension_sections") {
        for (i, slot) in out.iter_mut().enumerate() {
            if let Some(v) = items.get(i).and_then(Value::as_u64) {
                *slot = v as u32;
            }
        }
    }
    Ok(out)
}

/// Rows at/above which the NVFP4 FFN takes the f8w wide-prefill chain
/// instead of `nvf4_ffn`. Default = the Dense lane's `w8_min`: a solo
/// prefill scan measured the f8w arm ahead of the W4A16 walk at every row
/// count (128 tok 111.2 -> 40.2 ms, 2048 tok 954.5 -> 129.8). SUPERSEDED
/// where the W4A4 family's f4t arm exists (`nvf4_wide_w4a4`): the twin is
/// then not built at all and this threshold never fires, because W4A4 beats
/// it on wide prefill. `PADDOCK_NVF4_F8W=0`
/// kills the arm; a positive value builds the twin and sets the threshold
/// (the labeled A/B).
pub(super) fn nvf4_f8w_min_rows(w8_min: usize) -> usize {
    static V: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    match *V.get_or_init(
        || match paddock_models::dev_var!("PADDOCK_NVF4_F8W").ok().as_deref() {
            Some("0") => Some(usize::MAX),
            Some(s) => s.parse().ok().or(Some(usize::MAX)),
            None => None,
        },
    ) {
        Some(v) => v,
        None => w8_min,
    }
}

/// The wide-prefill FFN chain over an f8w plane pair (fused gate|up +
/// down): quantize xn to e4m3, gate|up GEMM, fused swiglu+quantize, down
/// GEMM. Returns whether `d_proj` holds bf16 (the caller's tail add).
///
/// Lifted verbatim out of the `Ffn::Dense` prefill arms so the NVFP4 lane
/// can run the same measured chain instead of `nvf4_ffn`'s decode-band
/// Checkpoint-exact fp8 dense FFN, b=1 (the f8row class, see `F8RowFfn`):
/// three f32-in row GEMVs - no activation staging at one row. Lands `proj`
/// exactly like the Q8/lin chains (the caller does the residual add).
pub(super) fn ffn_f8row_gemv(
    exec: &GpuExecutor,
    p: &F8RowFfn,
    xn: &CudaSlice<f32>,
    ffn_gate: &mut CudaSlice<f32>,
    ffn_up: &mut CudaSlice<f32>,
    proj: &mut CudaSlice<f32>,
) -> Result<(), GpuModelError> {
    exec.f8r_gemv(&p.gate, xn, ffn_gate, p.embd, p.ff)?;
    exec.f8r_gemv(&p.up, xn, ffn_up, p.embd, p.ff)?;
    exec.swiglu(ffn_gate, ffn_up, p.ff)?;
    exec.f8r_gemv(&p.down, ffn_gate, proj, p.ff, p.embd)?;
    Ok(())
}

/// Checkpoint-exact fp8 dense FFN at r rows (decode band AND prefill/chunk
/// widths - one chain, the pack's launcher elects the arm by width:
/// gemm2/tw4d/mma64 with K-split at 2..64, mcol/tw/tw5 above). Staging is
/// the per-ROW e4m3 pair (`q`/`rs` = d_f8t_q/d_f8t_rs, sized cap x qw when
/// this lane built) - "strategy: token", the checkpoint's own activation
/// class. r == 1 takes the GEMV arm: pd_f8row_gemm has no one-row arm (its
/// kt tail measured 209 us flat on these planes, 0.3-0.6x the roof).
#[allow(clippy::too_many_arguments)]
pub(super) fn ffn_f8row_rows(
    exec: &GpuExecutor,
    p: &F8RowFfn,
    xn: &CudaSlice<f32>,
    q: &mut CudaSlice<i8>,
    rs: &mut CudaSlice<f32>,
    ffn_gate: &mut CudaSlice<f32>,
    ffn_up: &mut CudaSlice<f32>,
    proj: &mut CudaSlice<f32>,
    r: usize,
) -> Result<(), GpuModelError> {
    if r == 1 {
        return ffn_f8row_gemv(exec, p, xn, ffn_gate, ffn_up, proj);
    }
    let (embd, ff) = (p.embd, p.ff);
    if r * embd.max(ff) > q.len() || r > rs.len() {
        return Err(GpuModelError::Unsupported(format!(
            "f8row ffn: row staging pair too small for r={r} (q {} rs {}) - \
             ensure_scratch must size d_f8t_q/d_f8t_rs to cap when this lane builds",
            q.len(),
            rs.len()
        )));
    }
    exec.quantize_e4m3_row(xn, q, rs, embd, r)?;
    // 2..32: one grid over both planes (granite's decode arm); the pack
    // declines outside its shape/fill window and the pair runs as two GEMMs.
    let paired = (2..=32).contains(&r)
        && exec.f8row_gemm2(&p.gate, &p.up, q, rs, ffn_gate, ffn_up, embd, ff, r)?;
    if !paired {
        exec.f8row_gemm(&p.gate, q, rs, ffn_gate, embd, ff, r)?;
        exec.f8row_gemm(&p.up, q, rs, ffn_up, embd, ff, r)?;
    }
    // fused silu(gate)*up + per-row e4m3 (single-pass at r=1 rows-per-block
    // where the pack ships it); older packs: swiglu + row quant
    if !exec.swiglu_quant_e4m3_row(ffn_gate, ffn_up, q, rs, ff, r)? {
        exec.swiglu(ffn_gate, ffn_up, r * ff)?;
        exec.quantize_e4m3_row(ffn_gate, q, rs, ff, r)?;
    }
    exec.f8row_gemm(&p.down, q, rs, proj, ff, embd, r)?;
    Ok(())
}

/// W4A16 walk - see the f8w plane build in load.rs for the numbers.
#[allow(clippy::too_many_arguments)]
pub(super) fn prefill_ffn_f8w(
    exec: &GpuExecutor,
    gu8: &(crate::gpu::RepackedMxfp4, usize, usize),
    d8: &(crate::gpu::RepackedMxfp4, usize, usize),
    xn: &CudaSlice<f32>,
    xq: &mut CudaSlice<i8>,
    xs: &mut CudaSlice<u8>,
    ffn_gate: &mut CudaSlice<f32>,
    ffn_up: &mut CudaSlice<f32>,
    proj: &mut CudaSlice<f32>,
    r: usize,
) -> Result<bool, GpuModelError> {
    // fused plane, row-sliced: gate = rows [0,ff), up = rows [ff,2ff)
    let ffh = gu8.2 / 2;
    exec.quantize_e4m3(xn, xq, xs, r * gu8.1)?;
    static O16: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let o16 = *O16.get_or_init(|| {
        paddock_models::dev_var_os!("PADDOCK_NO_F8W8_TMA").is_none()
            && paddock_models::dev_var_os!("PADDOCK_NO_O16").is_none()
    });
    static O16T: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let o16 = o16
        || (*O16T.get_or_init(|| {
            paddock_models::dev_var_os!("PADDOCK_QWEN35_O16_TC5").is_some()
                && paddock_models::dev_var_os!("PADDOCK_NO_O16").is_none()
        }) && r >= 256);
    if o16 && exec.has_f8_o16() {
        if exec.has_swiglu_b16_gu() {
            exec.f8_gemm_w8_o16(&gu8.0, 0, xq, xs, ffn_gate, gu8.1, gu8.2, r)?;
            exec.quantize_e4m3_swiglu_b16_gu(ffn_gate, xq, xs, r * d8.1, ffh)?;
        } else {
            exec.f8_gemm_w8_o16(&gu8.0, 0, xq, xs, ffn_gate, gu8.1, ffh, r)?;
            exec.f8_gemm_w8_o16(&gu8.0, ffh, xq, xs, ffn_up, gu8.1, ffh, r)?;
            exec.quantize_e4m3_swiglu_b16(ffn_gate, ffn_up, xq, xs, r * d8.1)?;
        }
    } else {
        exec.f8_gemm_w8(&gu8.0, 0, xq, xs, ffn_gate, gu8.1, ffh, r)?;
        exec.f8_gemm_w8(&gu8.0, ffh, xq, xs, ffn_up, gu8.1, ffh, r)?;
        exec.quantize_e4m3_swiglu(ffn_gate, ffn_up, xq, xs, r * d8.1)?;
    }
    if o16 && exec.has_f8_o16() && exec.has_add_b16() {
        exec.f8_gemm_w8_o16(&d8.0, 0, xq, xs, proj, d8.1, d8.2, r)?;
        Ok(true)
    } else {
        exec.f8_gemm_w8(&d8.0, 0, xq, xs, proj, d8.1, d8.2, r)?;
        Ok(false)
    }
}
