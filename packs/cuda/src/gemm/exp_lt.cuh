// EXPERIMENT only - never a shipping lane (charter: cuBLAS-free; the frozen
// bars stay the provenance). This arm exists to MEASURE, in the live serve,
// the ceiling of the PR#4266-class datapath port by borrowing the library
// datapath (nvjet) for the decode dense band. Compiled only when build.sh
// runs with PD_EXP_LT=1; the shipped pack keeps the stub.
// Capture-hostile by design (lazy allocs, library internals): the experiment
// legs run with PADDOCK_Q38FN_NO_GRAPH=1 on both arms.
#pragma once
#ifdef PD_EXP_LT
#include <cublasLt.h>
#include <unordered_map>
#include <cstdint>

namespace pdexplt {

__global__ void f32_to_bf16_kernel(const float* __restrict__ x,
                                   __nv_bfloat16* __restrict__ y, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2bfloat16(x[i]);
}

struct LtPlan {
    cublasLtMatmulDesc_t op;
    cublasLtMatrixLayout_t la, lb, lc;
    cublasLtMatmulAlgo_t algo;
};

inline int lt_gemm(const void* w, const void* x, void* y, uint32_t in_dim,
                   uint32_t out_dim, uint32_t batch, cudaStream_t st) {
    static cublasLtHandle_t lt = nullptr;
    static void* ws = nullptr;
    static size_t wsz = 64ull << 20;
    static __nv_bfloat16* xb = nullptr;
    static size_t xb_elems = 0;
    static std::unordered_map<uint64_t, LtPlan> plans;
    if (!lt) {
        if (cublasLtCreate(&lt) != CUBLAS_STATUS_SUCCESS) return (int)cudaErrorUnknown;
        if (cudaMalloc(&ws, wsz) != cudaSuccess) return (int)cudaErrorMemoryAllocation;
    }
    const size_t need = (size_t)batch * in_dim;
    if (need > xb_elems) {
        // grow-by-retain: a captured graph may still read the old staging
        // buffer (the rule is in part_scratch.cuh); the old one is left mapped
        __nv_bfloat16* nb = nullptr;
        if (cudaMalloc(&nb, need * 2) != cudaSuccess) return (int)cudaErrorMemoryAllocation;
        xb = nb;
        xb_elems = need;
    }
    f32_to_bf16_kernel<<<(unsigned)((need + 255) / 256), 256, 0, st>>>(
        (const float*)x, xb, need);
    const uint64_t key = ((uint64_t)in_dim << 40) | ((uint64_t)out_dim << 8) | batch;
    auto it = plans.find(key);
    if (it == plans.end()) {
        LtPlan p{};
        cublasOperation_t tA = CUBLAS_OP_T, tN = CUBLAS_OP_N;
        if (cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS)
            return (int)cudaErrorUnknown;
        cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA));
        cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tN, sizeof(tN));
        cublasLtMatrixLayoutCreate(&p.la, CUDA_R_16BF, in_dim, out_dim, in_dim);
        cublasLtMatrixLayoutCreate(&p.lb, CUDA_R_16BF, in_dim, batch, in_dim);
        cublasLtMatrixLayoutCreate(&p.lc, CUDA_R_32F, out_dim, batch, out_dim);
        cublasLtMatmulPreference_t pref;
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                             &wsz, sizeof(wsz));
        cublasLtMatmulHeuristicResult_t hr[1]; int found = 0;
        if (cublasLtMatmulAlgoGetHeuristic(lt, p.op, p.la, p.lb, p.lc, p.lc, pref, 1, hr, &found)
                != CUBLAS_STATUS_SUCCESS || !found) {
            cublasLtMatmulPreferenceDestroy(pref);
            return (int)cudaErrorNotSupported;
        }
        cublasLtMatmulPreferenceDestroy(pref);
        p.algo = hr[0].algo;
        it = plans.emplace(key, p).first;
    }
    const float alpha = 1.f, beta = 0.f;
    const LtPlan& p = it->second;
    if (cublasLtMatmul(lt, p.op, &alpha, w, p.la, xb, p.lb, &beta, y, p.lc, y, p.lc,
                       &p.algo, ws, wsz, st) != CUBLAS_STATUS_SUCCESS)
        return (int)cudaErrorUnknown;
    return 0;
}

}  // namespace pdexplt
#endif  // PD_EXP_LT

// slot 542. Stub (NotSupported) unless the pack was built with PD_EXP_LT=1.
PD_EXPORT
int pd_exp_lt_gemm(const void* w, const void* x, void* y, uint32_t in_dim,
                   uint32_t out_dim, uint32_t batch, void* stream) {
#ifdef PD_EXP_LT
    if (in_dim == 0 || out_dim == 0 || batch == 0) return 0;
    return pdexplt::lt_gemm(w, x, y, in_dim, out_dim, batch, (cudaStream_t)stream);
#else
    (void)w; (void)x; (void)y; (void)in_dim; (void)out_dim; (void)batch; (void)stream;
    return (int)cudaErrorNotSupported;
#endif
}
