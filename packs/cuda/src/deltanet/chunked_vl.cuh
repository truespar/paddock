// deltanet/chunked_vl.cuh - the varlen chunked-GDN entries (slots 323 / 447): one stage1 + walk
// launch pair for every eligible span of a tick. Textually-included segment of the single pack
// translation unit, after deltanet/stage2_sample.cuh (its stage1 kernels and the walk launchers
// of walk_rs.cuh / walk_tc5.cuh); moved out of stage2_sample.cuh unchanged.

// 323: varlen chunked-GDN (GDN formulation band): one
// stage1 + walk launch pair covers every eligible span of the tick - the
// reference varlen class (fla chunk kernels batch all spans and decode
// rows via chunk_indices; we batch the spans and keep decode rows on the
// recurrent path). chunk_items = (global row0, chunk len) u32 pairs per
// launch chunk; span_items = (first launch chunk, span rows, state f32
// offset, out row0) u32 quads per span. Per-span math is identical to the
// per-span RS launches (scratch stays launch-chunk-indexed; the walk
// re-bases every operand off its span item) - only the grid packing
// changes. RS-route only: mirrors the per-span dispatch's env gates and
// returns cudaErrorNotSupported when any other arm is elected, so the
// engine falls back per-span loudly rather than silently misrouting.
// f32-v only (the mixed-tick producer class this serves).
static int pd_gdc_rs_vl_go(
    bool qkc, uint32_t n_k_heads, const void* q, const void* k, const void* v,
    const void* g, const void* beta, void* state, void* out, void* dw,
    void* du, void* aqk, void* cg, const void* chunk_items, uint32_t n_chunks,
    const void* span_items, uint32_t n_spans, uint32_t n_tokens,
    uint32_t n_heads, uint32_t head_dim, void* stream) {
    if (n_chunks == 0 || n_spans == 0 || n_heads == 0) return 0;
    if (head_dim != PD_DNC_D) return cudaErrorInvalidValue;
    static const bool rs_ok = [] {
        const char* rse = pd_env("PADDOCK_DNC_RS");
        const char* s1e = pd_env("PADDOCK_DNC_S1MMA");
        const char* db = pd_env("PADDOCK_DNC_DWB16");
        return rse && atoi(rse) != 0 && s1e && atoi(s1e) != 0 &&
               !(db && atoi(db) != 0) && pd_dns_state_class() != 1 &&
               !pd_env("PADDOCK_NO_DNC_MMA_V2") && !pd_env("PADDOCK_DNC_SCAN") &&
               !pd_env("PADDOCK_DNC_FLA") && !pd_env("PADDOCK_DNC_SPLIT");
    }();
    if (!rs_ok) return cudaErrorNotSupported;
    static const uint32_t rs_s1prec = [] {
        const char* e = pd_env("PADDOCK_DNC_S1PREC");
        const uint32_t v = e ? (uint32_t)atoi(e) : 3u;
        return v == 1u ? 1u : 3u;
    }();
    static bool vl_attr = false;
    if (!vl_attr) {
        cudaFuncSetAttribute(
            (const void*)pd_dnc_stage1_v2_kernel<1u, float, __nv_bfloat16,
                                                 float, __nv_bfloat16>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, PD_DNS1_SMEM);
        cudaFuncSetAttribute(
            (const void*)pd_dnc_stage1_v2_kernel<3u, float, __nv_bfloat16,
                                                 float, __nv_bfloat16>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, PD_DNS1_SMEM);
        cudaFuncSetAttribute((const void*)pd_dnc_walk_rs_kernel<float>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             PD_DNRS_SMEM);
        cudaFuncSetAttribute((const void*)pd_dnc_walk_rs_kernel<__half>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             PD_DNRS_SMEM);
        cudaFuncSetAttribute((const void*)pd_dnc_walk_rs_kernel<__nv_fp8_e4m3>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             PD_DNRS_SMEM);
        cudaFuncSetAttribute((const void*)pd_dnc_stage1_rs_kernel<float>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             PD_DNS1RS_SMEM);
        cudaFuncSetAttribute(
            (const void*)pd_dnc_stage1_rs_kernel<float, 0, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, PD_DNS1RS_SMEM);
        cudaFuncSetAttribute((const void*)pd_dnc_walk_rs_kernel<float, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             PD_DNRS_SMEM);
        cudaFuncSetAttribute((const void*)pd_dnc_walk_rs_kernel<__half, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             PD_DNRS_SMEM);
        cudaFuncSetAttribute(
            (const void*)pd_dnc_walk_rs_kernel<__nv_fp8_e4m3, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, PD_DNRS_SMEM);
        ::fprintf(stderr, "[dnc-vl] ENGAGED (varlen chunked-GDN)\n");
        vl_attr = true;
    }
    const cudaStream_t s = (cudaStream_t)stream;
    const uint32_t nc = n_chunks;
    // scratch carving: identical formulas to the per-span RS block, with nc
    // = the TICK's total launch chunks (buffers sized for it engine-side)
    const size_t rs_half = (size_t)nc * PD_DNC_C * n_heads * PD_DNC_D;
    __nv_bfloat16* rs_qb = (__nv_bfloat16*)dw + rs_half;
    __nv_bfloat16* rs_kb = (__nv_bfloat16*)du + rs_half;
    float* rs_gsh = (float*)((char*)aqk
        + (size_t)nc * PD_DNC_C * PD_DNC_C * n_heads * 2u);
    dim3 g1(nc, n_heads);
#define PD_RS_VL_S1(P)                                                         \
    pd_dnc_stage1_v2_kernel<P, float, __nv_bfloat16, float, __nv_bfloat16>     \
        <<<g1, 256, PD_DNS1_SMEM, s>>>(                                        \
            (const float*)q, (const float*)k, (const float*)v,                 \
            (const float*)g, (const float*)beta, (__nv_bfloat16*)dw,           \
            (__nv_bfloat16*)du, (__nv_bfloat16*)aqk, (double*)cg, n_tokens,    \
            n_heads, rs_qb, rs_kb, rs_gsh, (const uint32_t*)chunk_items)
    // QKD (2026-09-08, GB10): on the qkc route the walk reads the compact
    // plane itself (walk_rs.cuh), so stage1 skips its 25 MB emission
    // (qb16/kb16 = NULL). PADDOCK_NO_DNC_QKD=1 keeps the copies for A/B.
    static const bool qkd_off = pd_env("PADDOCK_NO_DNC_QKD") != nullptr;
    const bool qkd = qkc && !qkd_off;
#if PD_DNC_PROBE != 2
    if (qkc) {
        // compact pair (slot 447): the ENGINE guarantees conv wrote the
        // compact bf16 planes and that the rs stage1 route is live; a
        // mispaired call must fail loud, not read garbage.
        if (!pd_dnc_s1rs_on() || n_k_heads == 0u ||
            (n_heads % n_k_heads) != 0u)
            return cudaErrorNotSupported;
        pd_dnc_stage1_rs_kernel<float, 0, true><<<g1, 256, PD_DNS1RS_SMEM, s>>>(
            (const float*)q, (const float*)k, (const float*)v, (const float*)g,
            (const float*)beta, (__nv_bfloat16*)dw, (__nv_bfloat16*)du,
            (__nv_bfloat16*)aqk, (double*)cg, n_tokens, n_heads,
            qkd ? nullptr : rs_qb, qkd ? nullptr : rs_kb,
            rs_gsh, (const uint32_t*)chunk_items, n_k_heads);
    } else if (pd_dnc_s1rs_on())
        pd_dnc_stage1_rs_kernel<float><<<g1, 256, PD_DNS1RS_SMEM, s>>>(
            (const float*)q, (const float*)k, (const float*)v, (const float*)g,
            (const float*)beta, (__nv_bfloat16*)dw, (__nv_bfloat16*)du,
            (__nv_bfloat16*)aqk, (double*)cg, n_tokens, n_heads, rs_qb, rs_kb,
            rs_gsh, (const uint32_t*)chunk_items);
    else if (rs_s1prec == 1u) PD_RS_VL_S1(1u);
    else PD_RS_VL_S1(3u);
#endif
#undef PD_RS_VL_S1
    dim3 gw(PD_DNC_D / PD_DNC_G, n_heads, n_spans);   // column groups fastest (walk_rs.cuh note)
#if PD_DNC_PROBE == 1
    (void)gw; return pd_launch_status();
#endif
    const int vl_cls = pd_dns_state_class();
    if (qkd) {
        const __nv_bfloat16* qc = (const __nv_bfloat16*)q;
        const __nv_bfloat16* kc = (const __nv_bfloat16*)k;
        if (vl_cls == 3)
            pd_dnrs_walk_go<__nv_fp8_e4m3, true>(gw, s, rs_qb, rs_kb, (__nv_fp8_e4m3*)state, (const __nv_bfloat16*)dw, (const __nv_bfloat16*)du, rs_gsh, (const __nv_bfloat16*)aqk, (float*)out, n_tokens, n_heads, (const uint32_t*)span_items, qc, kc, n_k_heads);
        else if (vl_cls == 2) {
#if defined(PD_TC5_HOST)
            if (pd_dnt5_walk_go<__half>(gw, s, rs_qb, rs_kb, (__half*)state, (const __nv_bfloat16*)dw, (const __nv_bfloat16*)du, rs_gsh, (const __nv_bfloat16*)aqk, (float*)out, n_tokens, n_heads, (const uint32_t*)span_items, (uint64_t)nc * n_heads * PD_DNC_C, qc, kc, n_k_heads))
                return pd_launch_status();
#endif
            pd_dnrs_walk_go<__half, true>(gw, s, rs_qb, rs_kb, (__half*)state, (const __nv_bfloat16*)dw, (const __nv_bfloat16*)du, rs_gsh, (const __nv_bfloat16*)aqk, (float*)out, n_tokens, n_heads, (const uint32_t*)span_items, qc, kc, n_k_heads);
        } else
            pd_dnrs_walk_go<float, true>(gw, s, rs_qb, rs_kb, (float*)state, (const __nv_bfloat16*)dw, (const __nv_bfloat16*)du, rs_gsh, (const __nv_bfloat16*)aqk, (float*)out, n_tokens, n_heads, (const uint32_t*)span_items, qc, kc, n_k_heads);
        return pd_launch_status();
    }
    if (vl_cls == 3)
        pd_dnrs_walk_go<__nv_fp8_e4m3, false>(gw, s, rs_qb, rs_kb, (__nv_fp8_e4m3*)state, (const __nv_bfloat16*)dw, (const __nv_bfloat16*)du, rs_gsh, (const __nv_bfloat16*)aqk, (float*)out, n_tokens, n_heads, (const uint32_t*)span_items, nullptr, nullptr, 0u);
    else if (vl_cls == 2) {
#if defined(PD_TC5_HOST)
        // the walk on tcgen05 (walk_tc5.cuh), walk_rs where it declines
        if (pd_dnt5_walk_go<__half>(gw, s, rs_qb, rs_kb, (__half*)state, (const __nv_bfloat16*)dw, (const __nv_bfloat16*)du, rs_gsh, (const __nv_bfloat16*)aqk, (float*)out, n_tokens, n_heads, (const uint32_t*)span_items, (uint64_t)nc * n_heads * PD_DNC_C))
            return pd_launch_status();
#endif
        pd_dnrs_walk_go<__half, false>(gw, s, rs_qb, rs_kb, (__half*)state, (const __nv_bfloat16*)dw, (const __nv_bfloat16*)du, rs_gsh, (const __nv_bfloat16*)aqk, (float*)out, n_tokens, n_heads, (const uint32_t*)span_items, nullptr, nullptr, 0u);
    } else
        pd_dnrs_walk_go<float, false>(gw, s, rs_qb, rs_kb, (float*)state, (const __nv_bfloat16*)dw, (const __nv_bfloat16*)du, rs_gsh, (const __nv_bfloat16*)aqk, (float*)out, n_tokens, n_heads, (const uint32_t*)span_items, nullptr, nullptr, 0u);
    return pd_launch_status();
}

PD_EXPORT
int pd_gated_delta_chunked_rs_vl(
    const void* q, const void* k, const void* v, const void* g,
    const void* beta, void* state, void* out, void* dw, void* du, void* aqk,
    void* cg, const void* chunk_items, uint32_t n_chunks,
    const void* span_items, uint32_t n_spans, uint32_t n_tokens,
    uint32_t n_heads, uint32_t head_dim, void* stream) {
    return pd_gdc_rs_vl_go(false, 0u, q, k, v, g, beta, state, out, dw, du,
                           aqk, cg, chunk_items, n_chunks, span_items, n_spans,
                           n_tokens, n_heads, head_dim, stream);
}

// QKC pair entry (slot 447): q/k are the conv qkc twin's COMPACT bf16
// planes; everything from the stage1 panes on is bit-identical to the
// expanded pair (see the kernels' comments).
PD_EXPORT
int pd_gated_delta_chunked_rs_vl_qkc(
    const void* q, const void* k, const void* v, const void* g,
    const void* beta, void* state, void* out, void* dw, void* du, void* aqk,
    void* cg, const void* chunk_items, uint32_t n_chunks,
    const void* span_items, uint32_t n_spans, uint32_t n_tokens,
    uint32_t n_heads, uint32_t n_k_heads, uint32_t head_dim, void* stream) {
    return pd_gdc_rs_vl_go(true, n_k_heads, q, k, v, g, beta, state, out, dw,
                           du, aqk, cg, chunk_items, n_chunks, span_items,
                           n_spans, n_tokens, n_heads, head_dim, stream);
}
