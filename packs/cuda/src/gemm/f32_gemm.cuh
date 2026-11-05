// gemm/f32_gemm.cuh - the f32 GEMM entries over gemm/f32_qkv.cuh's tiled
// kernel (pd_gemm_f32_nt_kernel, PD_RGEMM_BK): the k-quant interim's
// unconditionally-tiled GEMM and the router's exact-f32 tiles at any row
// count (slot 928). Textually-included segment of the single pack
// translation unit, right after gemm/f32_qkv.cuh.

// Unconditionally-tiled f32 GEMM over a [out, in]-major weight (y = x·Wᵀ per
// row): the k-quant batch/prefill interim's compute stage - its weights are
// WIDE (ffn outs), where the matvec tile's per-TT-token weight re-read is
// pathological. Same pd_gemm_f32_nt_kernel rungs as the router (K walk is
// tile-size-invariant -> rungs bit-identical); falls back to the matvec tile
// only on misaligned shapes. Separate export so the ROUTER keeps its env-gated
// numerics (its callers are b9895-parity-pinned).
PD_EXPORT
int pd_gemm_f32(const void* w, const void* x, void* out, uint32_t in_dim,
                uint32_t out_dim, uint32_t batch, void* stream) {
    if (out_dim == 0 || batch == 0) return 0;
    // tf32 tensor-core arm (nemotron prefill rung). This
    // export is nemotron's prefill attention q/k/v/o, and the SIMT tile runs
    // it at ~23% of f32 peak: a 537x4096x2688 q_proj reads 411 us, o_proj
    // 365, k/v 81 each -> 5.6 ms per prefill tick, 24 launches, ~2% of the
    // c32 wall. Same kernel as the `ba` rung above; 3xTF32 (PREC=3) is
    // STRICTLY FINER than the bf16 class the batched decode twins already
    // run these exact planes in, so this raises prefill toward the SIMT
    // f32 it replaces rather than dropping below the served class.
    //
    // The gate is SHAPE-ONLY deliberately. layer_walk's contract is granite's
    // law -- every prefill row takes the same rungs at any r, so a
    // warm-resume tail reproduces the cold chunk's bytes -- so this must
    // never key on batch. The kernel is r-safe: M staging zero-fills via
    // cp.async (`m0+xr < batch ? 16u : 0u`) and the store guards `m < batch`;
    // the unguarded N axis is what the out_dim % 32 gate is for.
    //
    // ELECTED DEFAULT: PREC=3. Unset = 3, "0" = the SIMT
    // f32 tile (the A/B pin, PADDOCK_NVQKV=0 precedent), "p" = the single-tf32
    // probe arm. PREC=3 is elected over PREC=1 because on these shapes the
    // kernel is STAGING-bound, not compute-bound -- 3 mma passes vs 1 measured
    // 1.00x/0.99x on q_proj/o_proj (342.1 vs 340.7 us, 348.5 vs 350.6), so the
    // finer numerics are free. Both arms measured closer to the parity-pinned
    // serial reference than the f32 they replace (bulk-vs-serial mean |d|
    // 0.18814 -> 0.16544; resume 0.215193 -> 0.165267), because the SIMT
    // tile's accumulation ORDER was a bigger error source than tf32's mantissa.
    //
    // The other caller of this export is qwen35 ops.rs:192 (kq_gemm, the wdq
    // plane) -- checked at election time and not affected in any shipped
    // config: it sits behind PADDOCK_KQ_F32_PREFILL, which is off by default.
    static const int gf_tf32 = [] {
        const char* e = pd_env("PADDOCK_GEMMF32_TF32");
        return e ? (e[0] == 'p' ? 1 : (atoi(e) != 0 ? 3 : 0)) : 3;
    }();
    // The wide 128x128 tile (kernel above) where its grid fills the die
    // three times over: out_dim a multiple of 128 and >= 3 CTAs per SM (one
    // CTA per SM resident, so the wave tail must be small). Bit-exact vs the
    // 32x32 tile (same k sequence), so the election is shape-only like the
    // rest of this launcher; PADDOCK_GEMMF32_WIDE=0 pins the 32x32 tile (dev
    // A/B). Measured on GB10 (bench/tf32_gb10_bench.cu, 0 differing bytes
    // at every shape): q 2688->4096 at 1022 rows 2407 -> 1616 us (1.49x,
    // 13.9 TF/s useful = the 3xTF32 mma roof of the die, ~43 TF/s issued),
    // 2048 rows 4806 -> 2979 (1.61x); o 4096->2688 at 1022 rows 2185 ->
    // 1647 (1.33x), 2048 rows 6044 -> 2906 (2.08x). With the grouped raster
    // (below) the tile also wins at 512 rows (q 1.34x on 128 CTAs, o 1.27x
    // on 84) and stays at the mma roof to 7556 rows (11.5 ms, 5.3x the
    // 32x32 tile, whose x panel re-reads fall off L2 there), so the floor
    // is 1.5 CTAs per SM: k/v (out 256) stay on the 32x32 tile below 2k
    // rows, where the wide grid is 8-16 CTAs. On the 32x32 tile
    // PREC 1 and PREC 3 time the same on this die as on the big one (it is
    // staging-bound at ~10 TF/s useful); on this tile the three mma are the
    // wall, so PREC 1 would be ~3x again - a precision-class decision;
    // the shipped default stays PREC 3.
    static const bool wide_off = pd_env("PADDOCK_GEMMF32_WIDE") &&
                                 pd_env("PADDOCK_GEMMF32_WIDE")[0] == '0';
    static int nsm_w = 0;
    if (nsm_w == 0) { int d = 0; cudaGetDevice(&d); cudaDeviceGetAttribute(&nsm_w, cudaDevAttrMultiProcessorCount, d); if (nsm_w <= 0) nsm_w = 148; }
    if (gf_tf32 && !wide_off && (out_dim % 128u) == 0u && (in_dim % PD_RGEMM_BK) == 0u &&
        2u * ((batch + 127u) / 128u) * (out_dim / 128u) >= 3u * (uint32_t)nsm_w) {
        static bool attr1 = false, attr3 = false;
        dim3 grid(((batch + 127u) / 128u) * (out_dim / 128u));   // grouped raster inside
        if (gf_tf32 == 1) {
            if (!attr1) { cudaFuncSetAttribute(pd_gemm_tf32_nt_wide_kernel<1u>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)PD_TFW_SMEM); attr1 = true; }
            pd_gemm_tf32_nt_wide_kernel<1u><<<grid, 256, PD_TFW_SMEM, (cudaStream_t)stream>>>(
                (const float*)w, (const float*)x, (float*)out, in_dim, out_dim, batch);
        } else {
            if (!attr3) { cudaFuncSetAttribute(pd_gemm_tf32_nt_wide_kernel<3u>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)PD_TFW_SMEM); attr3 = true; }
            pd_gemm_tf32_nt_wide_kernel<3u><<<grid, 256, PD_TFW_SMEM, (cudaStream_t)stream>>>(
                (const float*)w, (const float*)x, (float*)out, in_dim, out_dim, batch);
        }
        return pd_launch_status();
    }
    if (gf_tf32 && (out_dim % 32u) == 0u && (in_dim % PD_RGEMM_BK) == 0u) {
        dim3 grid((batch + 31u) / 32u, out_dim / 32u);
        if (gf_tf32 == 1)
            pd_gemm_tf32_nt_kernel<1u><<<grid, 128, 0, (cudaStream_t)stream>>>(
                (const float*)w, (const float*)x, (float*)out, in_dim, out_dim,
                batch);
        else
            pd_gemm_tf32_nt_kernel<3u><<<grid, 128, 0, (cudaStream_t)stream>>>(
                (const float*)w, (const float*)x, (float*)out, in_dim, out_dim,
                batch);
        return pd_launch_status();
    }
    if ((out_dim % 32u) == 0u && (in_dim % PD_RGEMM_BK) == 0u) {
        const uint32_t b44 = ((batch + 63u) / 64u) * (out_dim / 64u);
        if ((out_dim % 64u) == 0u && b44 >= 160u) {
            dim3 grid((batch + 63u) / 64u, out_dim / 64u);
            pd_gemm_f32_nt_kernel<4u, 4u><<<grid, 256, 0, (cudaStream_t)stream>>>(
                (const float*)w, (const float*)x, (float*)out, in_dim, out_dim, batch);
        } else if ((out_dim % 64u) == 0u &&
                   ((batch + 31u) / 32u) * (out_dim / 64u) >= 160u) {
            dim3 grid((batch + 31u) / 32u, out_dim / 64u);
            pd_gemm_f32_nt_kernel<2u, 4u><<<grid, 256, 0, (cudaStream_t)stream>>>(
                (const float*)w, (const float*)x, (float*)out, in_dim, out_dim, batch);
        } else {
            dim3 grid((batch + 31u) / 32u, out_dim / 32u);
            pd_gemm_f32_nt_kernel<2u, 2u><<<grid, 256, 0, (cudaStream_t)stream>>>(
                (const float*)w, (const float*)x, (float*)out, in_dim, out_dim, batch);
        }
        return pd_launch_status();
    }
    if (batch >= 16u && (out_dim & 7u) == 0u) {
        dim3 grid(out_dim / 8u, (batch + 3u) / 4u);
        pd_matvec_f32_tile_kernel<4u><<<grid, 256, 0, (cudaStream_t)stream>>>(
            (const float*)w, (const float*)x, (float*)out, in_dim, out_dim, batch);
        return pd_launch_status();
    }
    dim3 grid(out_dim, (batch + 3u) / 4u);
    pd_pdl_go(pd_matvec_f32_batch_kernel<4u>, grid, 256, 0u, (cudaStream_t)stream,
        (const float*)w, (const float*)x, (float*)out, in_dim, out_dim, batch);
    return pd_launch_status();
}

// 928: the router's exact-f32 tiles at ANY row count - the 8 x 8 / 4 x 4 /
// 2 x 4 / 2 x 2 rungs the router launcher elects from 1024 rows, picked by
// grid fill alone. One owner thread and one ascending K walk an output on
// every rung, so a logit's bits never depend on the chunk's rows (the matvec
// tile the router takes below 1024 rows reassociates the walk). out_dim % 32,
// in_dim % PD_RGEMM_BK.
PD_EXPORT
int pd_gemm_f32_tiled(const void* w, const void* x, void* out, uint32_t in_dim,
                      uint32_t out_dim, uint32_t batch, void* stream) {
    if (out_dim == 0 || batch == 0) return 0;
    if ((out_dim % 32u) != 0u || (in_dim % PD_RGEMM_BK) != 0u) return cudaErrorInvalidValue;
    static int nsm = 0;
    if (nsm == 0) { int d = 0; cudaGetDevice(&d); cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, d); if (nsm <= 0) nsm = 148; }
    cudaStream_t st = (cudaStream_t)stream;
    const uint32_t b88 = ((batch + 127u) / 128u) * (out_dim / 128u);
    const uint32_t b44 = ((batch + 63u) / 64u) * (out_dim / 64u);
    if ((out_dim % 128u) == 0u && b88 >= 2u * (uint32_t)nsm) {
        dim3 grid((batch + 127u) / 128u, out_dim / 128u);
        pd_gemm_f32_nt_kernel<8u, 8u><<<grid, 256, 0, st>>>((const float*)w, (const float*)x, (float*)out, in_dim, out_dim, batch);
    } else if ((out_dim % 64u) == 0u && b44 >= 160u) {
        dim3 grid((batch + 63u) / 64u, out_dim / 64u);
        pd_gemm_f32_nt_kernel<4u, 4u><<<grid, 256, 0, st>>>((const float*)w, (const float*)x, (float*)out, in_dim, out_dim, batch);
    } else if ((out_dim % 64u) == 0u && ((batch + 31u) / 32u) * (out_dim / 64u) >= 160u) {
        dim3 grid((batch + 31u) / 32u, out_dim / 64u);
        pd_gemm_f32_nt_kernel<2u, 4u><<<grid, 256, 0, st>>>((const float*)w, (const float*)x, (float*)out, in_dim, out_dim, batch);
    } else {
        dim3 grid((batch + 31u) / 32u, out_dim / 32u);
        pd_gemm_f32_nt_kernel<2u, 2u><<<grid, 256, 0, st>>>((const float*)w, (const float*)x, (float*)out, in_dim, out_dim, batch);
    }
    return pd_launch_status();
}
