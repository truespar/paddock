// tma_desc.cuh - TMA tensor-map builders + tcgen05 descriptor constructors.
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
//
// Why this FILE EXISTS. These helpers used to live in the middle
// of gemm/dense_fp4_w8.cuh, ~2.5k lines into an 11k-line segment that is
// included 43rd. They are not dense-fp4-w8 code - they are leaf-level
// descriptor construction that six other segments consume (gemm/f8_lin,
// gemm/f16_dense, attn/decode_tc5, attn/prefill, moe/f8, quant/nvf4). Sitting
// that late in the include order had grown four separate workarounds:
//   1. a forward declaration of pd_tmap_2d inside dense_fp4_w8 itself, because
//      an early launcher in the same file calls it ~2.1k lines before its
//      definition;
//   2. attn/prefill.cuh's `pd_pf5_sdesc` + `PD_PF5_OK` - a hand-copied twin of
//      pd_tc5_sdesc and its arch gate, kept because prefill is included first;
//   3. gemm/f32_qkv.cuh's own cuTensorMapEncodeTiled resolver, a second copy
//      of the same driver-entry-point lookup;
//   4. gemm/f16_dense.cuh's `PD_F16T_OK` + `pd_f16t_sdesc` +
//      `pd_f16t_tmap_encode` + `pd_f16t_tmap_2d` - the same gate and the same
//      two helpers again, self-named for the same reason.
// All four are now retired. Note #4 was found only by searching for the
// STRING "cuTensorMapEncodeTiled": #2 and #3 announced themselves in comments
// naming the old `14_dense` filename, #4 did not. Search by content, not by
// symbol name, before believing a duplicate count.
//
// These are worth remembering as more than tidiness. An undefined PD_TC5_OK
// does not fail the build - `#if PD_TC5_OK` just evaluates as `#if 0`. That is
// how the first pack build silently compiled an empty pf5 kernel body and
// emitted garbage, while the bench TU (which defined the macro after its
// include) worked fine. Both #2 and #4 were deliberate, correct defences
// against exactly that. Defining the gate once, early, is what retires the
// failure mode; a copy per consumer only hides it.
//
// Nothing here is arch-specific at include time: PD_TC5_OK keys off
// __CUDA_ARCH__, so the macro is evaluated per arch pass wherever it is read,
// and hoisting its definition earlier in the TU cannot change its value.

// Host-side tensor-map encode, resolved once via the runtime (no libcuda link);
// nullptr on old drivers, which every caller reads as "route to the fallback".
//
// Deliberately not under PD_BS_HOST. The resolver is a driver-entry-point
// lookup with no block-scale content, and gemm/f32_qkv.cuh's TMA
// decode-attention staging needs it on every build - while PD_BS_HOST is set
// only when the arch list contains 120 or 100 (build.sh). Keeping the gate here
// is what forced f32_qkv to carry its own byte-identical copy.
typedef CUresult (*pd_tmap_encode_fn)(
    CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
    const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
    CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

// Resolved BY VERSION, pinned at 12000, not through cudaGetDriverEntryPoint.
// The unversioned call asks the driver for the entry point at CUDART_VERSION -
// the toolkit the pack was COMPILED with - and a driver older than that
// toolkit refuses outright (cudaErrorInvalidValue, no status, no pointer),
// even though it has carried cuTensorMapEncodeTiled since 12.0 and every
// kernel launch in the same pack works. Minor-version compatibility lets a
// 13.x pack run on any r580+ driver, so this is the ordinary case, not an
// edge: a pack built with nvcc 13.1+ on a 13.0 (r580) driver came up clean,
// engaged every non-TMA route, and then refused the f8 lin GEMM with 801 on
// the first token (a user's RTX PRO 6000 Max-Q on Windows, self-built pack).
// cuTensorMapEncodeTiled's signature has not changed since 12.0, which is the
// ABI the typedef above spells, so 12000 is the honest floor to ask for.
static pd_tmap_encode_fn pd_tmap_encode() {
    static pd_tmap_encode_fn fn = [] {
        void* p = nullptr;
        cudaDriverEntryPointQueryResult st;
        if (cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000,
                                             cudaEnableDefault, &st) != cudaSuccess ||
            st != cudaDriverEntryPointSuccess || p == nullptr) {
            // Route witness, once: every TMA consumer falls back silently or
            // refuses with 801, and neither names the driver as the cause.
            int drv = 0, rt = 0;
            cudaDriverGetVersion(&drv);
            cudaRuntimeGetVersion(&rt);
            ::fprintf(stderr,
                      "[tma] cuTensorMapEncodeTiled unavailable (driver CUDA %d, "
                      "pack built with CUDA %d) - TMA routes off\n",
                      drv, rt);
            // The failed query is handled here, so its cudaErrorInvalidValue
            // must not stay in the runtime's sticky last-error: every launcher
            // reads that after its launch, and the next one - the engine's
            // arch probe, once the table resolves this at load - would report
            // a kernel that never launched as refused (status 1, "no working
            // image for this arch").
            (void)cudaGetLastError();
            return (pd_tmap_encode_fn) nullptr;
        }
        return (pd_tmap_encode_fn)p;
    }();
    return fn;
}

// The 128B-swizzle box builders below are block-scale-lane geometry, so they
// keep a host gate - but it has to cover both host lanes, because
// gemm/f16_dense.cuh's tcgen05 route is PD_TC5_HOST-gated, not PD_BS_HOST.
// build.sh happens to set bs_host=1 whenever it sets PD_TC5_HOST (arch 100
// sets both), so `#ifdef PD_BS_HOST` alone would work today by coincidence -
// this spells the requirement out instead of relying on that.
#if defined(PD_BS_HOST) || defined(PD_TC5_HOST)
// builds the 2D byte-matrix map: rows x inner bytes, 128x128B boxes, 128B swizzle
static bool pd_tmap_2d(CUtensorMap* map, const void* base, uint64_t inner,
                       uint64_t rows) {
    pd_tmap_encode_fn enc = pd_tmap_encode();
    if (!enc || ((uintptr_t)base & 15u) || (inner & 15u)) return false;
    const cuuint64_t gdim[2] = {inner, rows};
    const cuuint64_t gstride[1] = {inner};
    const cuuint32_t box[2] = {128u, 128u};
    const cuuint32_t estride[2] = {1u, 1u};
    return enc(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2u, (void*)base, gdim, gstride,
               box, estride, CU_TENSOR_MAP_INTERLEAVE_NONE,
               CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

// half-row variant for: 128B x 64-row boxes, 128B swizzle kept
static bool pd_tmap_2d_h64(CUtensorMap* map, const void* base, uint64_t inner,
                           uint64_t rows) {
    pd_tmap_encode_fn enc = pd_tmap_encode();
    if (!enc || ((uintptr_t)base & 15u) || (inner & 15u)) return false;
    const cuuint64_t gdim[2] = {inner, rows};
    const cuuint64_t gstride[1] = {inner};
    const cuuint32_t box[2] = {128u, 64u};
    const cuuint32_t estride[2] = {1u, 1u};
    return enc(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2u, (void*)base, gdim, gstride,
               box, estride, CU_TENSOR_MAP_INTERLEAVE_NONE,
               CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

// 32-row boxes for the decode-band grouped MoE (BM=32 sorted blocks): the Y
// side stages 32 sorted rows per k-tile, 4KB a box. Same SW128 layout - a
// 32-row box is 4 of the 8-row core groups, so the sdesc math is unchanged.
static bool pd_tmap_2d_h32(CUtensorMap* map, const void* base, uint64_t inner,
                           uint64_t rows) {
    pd_tmap_encode_fn enc = pd_tmap_encode();
    if (!enc || ((uintptr_t)base & 15u) || (inner & 15u)) return false;
    const cuuint64_t gdim[2] = {inner, rows};
    const cuuint64_t gstride[1] = {inner};
    const cuuint32_t box[2] = {128u, 32u};
    const cuuint32_t estride[2] = {1u, 1u};
    return enc(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2u, (void*)base, gdim, gstride,
               box, estride, CU_TENSOR_MAP_INTERLEAVE_NONE,
               CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}
#endif  // PD_BS_HOST || PD_TC5_HOST

// tcgen05 availability gate. Descriptor encodings below follow the PTX ISA /
// CUTLASS mma_sm100_desc bit maps (reference; construction original).
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000) && defined(__CUDA_ARCH_FEAT_SM100_ALL)
#define PD_TC5_OK 1
#else
#define PD_TC5_OK 0
#endif

#if PD_TC5_OK
// smem descriptor: start/LBO/SBO in 16B units; layout 2 = SWIZZLE_128B
__device__ __forceinline__ uint64_t pd_tc5_sdesc(uint32_t saddr16) {
    return ((uint64_t)(saddr16 & 0x3FFFu))
         | ((uint64_t)1u << 16)            // LBO: 16B (K-half step)
         | ((uint64_t)64u << 32)           // SBO: 1024B (8-row core group)
         | ((uint64_t)2u << 61);           // SWIZZLE_128B
}
// instruction descriptor: e4m3 x e4m3 -> f32, K-major both, M=128 N=128
__device__ __forceinline__ uint32_t pd_tc5_idesc() {
    return (1u << 4) | ((128u >> 3) << 17) | ((128u >> 4) << 24);
}

// ---- block-scaled instruction descriptors (the hardware ue8m0 fold) -------
// bn = the mma N (bits 17+, N/8). 128 everywhere except the decode-band
// grouped MoE, whose BM=32 sorted blocks run N=32 tiles.
__device__ __forceinline__ uint32_t pd_tc5_bs_idesc_bn(uint32_t sfid, uint32_t bn) {
    return ((sfid & 3u) << 4)      // b_sf_id
         | (0u << 7) | (0u << 10)  // e4m3 x e4m3
         | ((bn >> 3) << 17)
         | (1u << 23)              // scale_format UE8M0
         | ((128u >> 4) << 24)
         | ((sfid & 3u) << 29);    // a_sf_id
}
__device__ __forceinline__ uint32_t pd_tc5_bs_idesc(uint32_t sfid) {
    return pd_tc5_bs_idesc_bn(sfid, 128u);
}
// ::2 twin for tc5s: M=256 N=256, same sf-id/ue8m0 fields
__device__ __forceinline__ uint32_t pd_tc5s_idesc(uint32_t sfid) {
    return ((sfid & 3u) << 4) | (0u << 7) | (0u << 10)
         | ((256u >> 3) << 17) | (1u << 23) | ((256u >> 4) << 24)
         | ((sfid & 3u) << 29);
}
#endif  // PD_TC5_OK
