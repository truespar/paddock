//! The dense matmul pair the gpt-oss chain runs on: quantize the shared
//! activations once, then the int8 GEMM ladder with the bias folded in.

use super::*;

/// Quantize batch activations once for a run of dense matmuls sharing the same
/// input (wq/wk/wv share the post-norm rows - qwen P6j dedup). Layout picked by
/// the same batch>64 rule the GEMM half uses: >64 -> flat mmq layout, else
/// strided int8 + per-32 scales.
#[allow(clippy::too_many_arguments)]
pub(super) fn mm_quant(
    exec: &GpuExecutor,
    x: &CudaSlice<f32>,
    xq: &mut CudaSlice<i8>,
    xs: &mut CudaSlice<f32>,
    yq: &mut CudaSlice<u8>,
    in_dim: usize,
    b: usize,
) -> Result<(), GpuModelError> {
    if b > 64 {
        exec.quantize_q8_mmq(x, yq, in_dim, b)?;
    } else {
        exec.quantize_q8(x, xq, xs, b * in_dim)?;
    }
    Ok(())
}

/// GEMM half for activations staged by [`mm_quant`]: the qwen batch ladder on
/// the int8 quantized class, with the row bias folded into each rung's
/// epilogue (bit-exact vs the old trailing `bias_add` launch - the fold adds
/// bias to the completed per-element sum in the same order; the mmq rung
/// splits it store/fixup to keep the order). Killing the 4 bias launches per
/// layer is part of the dense-chain slimming.
/// Ladder: nc GEMV (2..=4, gemv-class weight BW), dp4a MT tile (..=32, batch
/// fits 1-2 weight passes), mma TC tile (..=64), mmq stream-k above. The
/// small-batch rungs matter because the mma grid is N-tiles only - on wk/wv
/// (N=512) that is 8 blocks, which idles the GPU for ~60 us/launch and
/// dominated the serving decode step at low B.
#[allow(clippy::too_many_arguments)]
pub(super) fn mm_pre(
    exec: &GpuExecutor,
    w: &RepackedQ8,
    bias: &CudaSlice<f32>,
    xq: &CudaSlice<i8>,
    xs: &CudaSlice<f32>,
    yq: &CudaSlice<u8>,
    skfix: &mut CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    b: usize,
) -> Result<(), GpuModelError> {
    if b > 64 {
        // The cp.async pipe kernel now K-pads its final partial 128-chunk
        // (weight tail masked, activation zero-quantized), so gpt-oss's K=2880
        // rides it bit-exactly, and folds bias in its epilogue. Its barrier
        // stall is 0.20 vs stream-k mmq_b's 1.16.
        // Standalone A/B (wqkv 5120x2880 / wo 2880x2880, us):
        //     batch   256    512    1024   2048
        //   mmq_b      39/39  76/52  132/72 257/138
        //   pipe       49/49  57/49  115/62 231/122
        // pipe wins from ~512 up (deep prefetch amortizes; -25% on wqkv@512),
        // but its 1-CTA/SM tiled grid has a ~48us floor that stream-k beats
        // below 512 (wo@256 loses +24%). So gate on b>=512. It cuts prefill
        // TTFT but sits off the throughput-critical path for decode-bound
        // shapes, so aggregate throughput is flat. PADDOCK_NO_MMQ_PIPE
        // disables; PADDOCK_MMQ_TILED pins plain stream-k tiling.
        let pipe = b >= 512
            && exec.has_q8_0_gemm_mmq_pipe()
            && paddock_models::dev_var_os!("PADDOCK_NO_MMQ_PIPE").is_none();
        if pipe {
            exec.q8_0_gemm_mmq_pipe(w, Some(bias), yq, y, b)?; // bias folded in epilogue
        } else {
            let tiled = paddock_models::dev_var_os!("PADDOCK_MMQ_TILED").is_some();
            exec.q8_0_gemm_mmq_b(
                w,
                Some(bias),
                yq,
                if tiled { None } else { Some(skfix) },
                y,
                b,
            )?;
        }
    } else if b > 8 {
        // K-split mma: the plain 64x64-tile grid is out_dim/64 blocks (wk/wv:
        // 8) and idles a 188-SM die. The z-split partial planes + fixed-order
        // combine measured wq 56 -> 39, wk 37 -> 16, wo 56 -> 33 us at B=32
        // (and beat both the plain mma and the 2-pass mt tile at B=64).
        exec.q8_0_gemm_mma_ks_b(w, Some(bias), xq, xs, skfix, y, b)?;
    } else if b > 4 {
        // 5..=8 rode the dp4a MT tile while wk/wv (N=512) starved the mma
        // grid; the fused wqkv (N=5120 -> 80 tiles x z-split) and wo (45
        // tiles) fill the die, and the K-split mma at b<=16 runs the BN16
        // rung (c8 profile: mt was 24.1us avg at 2.6x its read floor).
        exec.q8_0_gemm_mma_ks_b(w, Some(bias), xq, xs, skfix, y, b)?;
    } else {
        exec.q8_0_gemv_dp4a_nc_b(w, Some(bias), xq, xs, y, b)?;
    }
    Ok(())
}
