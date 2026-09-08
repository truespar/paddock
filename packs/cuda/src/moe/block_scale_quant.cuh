// moe/block_scale_quant.cuh (formerly 12_bs_moe_quant.cuh) - sm_120a block-scale MoE: e4m3 quantize (+swiglu) front half
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
// ============================================================================
// sm_120a block-scale tensor-core MoE (mxFP4 weights x FP8 activations).
// The s8 mmq MoE kernels spend ~4 issue slots of fp4 unpack + e8m0 float
// rescaling around every tensor instruction - measured 67 TFLOPS-effective on
// GB202 where the tensor pipe standalone does 276. Blackwell's block-scale
// mma (kind::mxf8f6f4) executes the ue8m0 scaling in HARDWARE: the mxfp4
// weight nibbles feed the A operand directly (packed in shared, spread to
// 8-bit containers at fragment load: value nibble << 2, an e2m3-compatible
// container - verified empirically, see G4 notes), activations ride as e4m3
// with ue8m0 per-32 scales. NUMERIC CLASS CHANGE on the activation side:
// e4m3 (3 mantissa bits) vs the q8_1 int8 path - perplexity-class, not
// greedy-exact vs mmq; routed behind a capability + env check in the engine.
// Requires the pack built for sm_120a ('a' feature target); on any other
// arch these launchers return cudaErrorNotSupported and the engine keeps the
// s8 mmq path. Build with -DPD_BS_HOST=1 IFF the gencode list includes
// compute_120a (the host launcher cannot see __CUDA_ARCH__; a silent empty
// launch would violate the no-silent-failure rule).
// sm_120a OR sm_121a (GB10 / DGX Spark): both feature targets carry the
// block-scale MMA; the plain 12.x passes and the 120f family pass do not
// define either macro and get the stubs. Host half: pd_dev_bs_sass (abi.cuh).
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200) && \
    (defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL))
#define PD_BS_OK 1
#else
#define PD_BS_OK 0
#endif

// f8w8 (W8A8-FP8) family gate: plain e4m3 x e4m3 mma.sync is sm_89+, so the
// f8 GEMV/GEMM kernels run on Ada/Hopper/datacenter-Blackwell with a
// software ue8m0 fold where sm_120a applies the block scales in hardware
// (the tcgen05 block-scale port is the later SOTA
// rung for cc 10.x; this gets the F8R serving ladder live there today).
// Table honesty: paddock_pack_kernels_v1 NULLs the family for devices below
// cc 8.9, where these bodies compile empty.
#if PD_BS_OK || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890))
#define PD_F8W8_OK 1
#else
#define PD_F8W8_OK 0
#endif

// NVFP4 checkpoint-plane CONSUMER gate (nemotron lane), same shape
// and same reason as PD_F8W8_OK above.
//
// The kernels that read an NVFP4 plane -- dequant, GEMV (+batch), the MoE
// up/down experts, the multi-row GEMM -- decode e2m1 nibbles in SIMT
// (__byte_perm into an e4m3 container, see pd_nvf4_gemv_kernel's PD_NV4G_STEP)
// and accumulate on plain FFMA. Not one of them issues a block-scale
// instruction: `kind::mxf4nvf4` lives only in the separate `*_bs` variants,
// which keep PD_BS_OK. They were gated on PD_BS_OK purely by sharing a file
// with the block-scale family, which compiled every NVFP4 body to an empty
// stub everywhere except sm_120a.
//
// Measured cost of that accident (B200/sm_100): the nemotron
// NVFP4 checkpoint LOADS, reports "missing nvf4_moe/gemv -- staying serial",
// and then fails every single generation with "kernel nvf4_moe_up_relu2
// missing from the loaded pack". SASS confirms it - on sm_100a those kernels
// are 16-instruction stubs against 384-1360 real instructions on sm_120a,
// while pd_nvf4_gemm_tc (gated on PD_BF16MMA_OK, not PD_BS_OK) was already
// built at 1944. The heavy GEMM was portable all along; only its feeders
// were not.
//
// Floor is 8.9, not 8.0: the nibble decode converts through __nv_fp8_e4m3.
// The bf16 mma the tc tile uses is 8.0 (PD_BF16MMA_OK) and is gated there.
// Host side must match exactly -- see paddock_pack_kernels_v1 in exports.cuh,
// which NULLs this family below the same floor. Widening one without the
// other is the silent-empty-kernel hazard the arch audit records.
#if PD_BS_OK || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890))
#define PD_NV4_OK 1
#else
#define PD_NV4_OK 0
#endif

// TMA members of the f8w8 family (bulk-tensor loads + mbarrier.try_wait
// handoff) are sm_90+ PTX FEATURES - sm_89 is f8w8-capable but has no TMA
// unit, and ptxas hard-rejects the instructions for it (this was a
// multi-arch build break: tma_kt's body under plain PD_F8W8_OK compiled the
// compute_89 pass into sm_90 PTX). Ada keeps the cp.async kernels.
#if PD_F8W8_OK && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 900))
#define PD_F8W8_TMA_OK 1
#else
#define PD_F8W8_TMA_OK 0
#endif

#define PD_BS_BM 32u          // sorted tokens per block (moe_align tile)
// Per-kernel K-chunk depth. A dedicated GB202 sweep settled
// this for good: the bs kernels are bound by DRAM EFFICIENCY at the READ
// GRANULARITY, not by barriers or MMA. A stream-pattern probe (random
// fill - constant fill triggers Blackwell compression and reports >3 TB/s):
//   64 B per row @ 1440 B stride (KC=128):  940 GB/s
//   128 B per row @ 1440 B stride (KC=256): 1580 GB/s   <- +68%
//   full chunk-major repack ceiling:        1614 GB/s
// KC=256 buys nearly the whole repack ceiling with zero layout change (the
// weight buffers stay row-major and keep serving the dp4a routes). The
// rental's "KC=256 is 2x worse" was an artifact of the old full-CTA-barrier
// loop: at 1 CTA/SM it had no co-resident block to hide the per-chunk
// wait_group + 2x __syncthreads turnaround behind. The warp-specialized
// producer/consumer rewrite below removes that turnaround entirely, which
// is exactly what makes KC=256 at 1 CTA/SM viable (dec-c32 gate_up 1268 ->
// 814 us, down 640 -> 422 us, both bit-exact vs the old loop). Earlier KC
// history (64 -> 128 wins on both kernels) predates the rewrite and stays
// true within the old structure.
#ifndef PD_BS_KC_GU
// KC=256 over the g||u INTERLEAVED layout: a row's chunk reads two adjacent
// 128 B pairs = 256 B contiguous - the stream_pattern table's best row
// (1614.9 GB/s vs 1580.7 for the old two-plane KC=256). KC=128 (one pair,
// 2 CTAs/SM) was A/B'd and lost ~3% end-to-end: the doubled chunk count's
// pipeline overheads outweighed the occupancy gain.
#define PD_BS_KC_GU 256u
#endif
#define PD_BS_KB_GU (PD_BS_KC_GU / 32u)
#define PD_BS_WROW_GU (PD_BS_KC_GU / 2u)
#define PD_BS_WSEG_GU (PD_BS_WROW_GU / 16u)
#define PD_BS_YROW_GU (PD_BS_KC_GU + 16u)  // 16B-aligned, bank-conflict-free stride
#define PD_BS_YSEG_GU (PD_BS_KC_GU / 16u)
#ifndef PD_BS_KC_DN
#define PD_BS_KC_DN 256u
#endif
#define PD_BS_KB_DN (PD_BS_KC_DN / 32u)
#define PD_BS_WROW_DN (PD_BS_KC_DN / 2u)
#define PD_BS_WSEG_DN (PD_BS_WROW_DN / 16u)
#define PD_BS_YROW_DN (PD_BS_KC_DN + 16u)
#define PD_BS_YSEG_DN (PD_BS_KC_DN / 16u)
// dense (non-MoE) block-scale GEMM: the mmq_pipe 128x128 tile (weight strip
// read once per 128 TOKENS, not per 32 - a 32-token tile re-reads weights 4x
// and loses to mmq_pipe outright) with 64-deep double-buffered K chunks. The
// fp4 + e4m3 tiles are half the int8 mmq tile, so this gets the cp.async
// pipe AND 2 blocks/SM at once (mmq had to pick one). Row strides: 48 = 32B
// packed fp4 + 2 ue8m0 + pad (stride 12 banks -> conflict-free afrag lanes),
// 80 = 2 ue8m0 + 64B e4m3 at +16 (stride 20 banks, same property).
// W-tile seg swizzle for the MoE bs kernels: WROW_GU = 64 B = 16 banks, so
// afrag lanes two rows apart hit the same bank pair - 4-way replay on every
// A-fragment load (profiled on a dedicated GB202: 26.0M ld conflicts at the
// pp512 shape, exactly 3 replays x 32 A-loads/warp-chunk across the grid).
// Padding the stride (64->80 B) would cost +8 KB smem/CTA = the second
// resident block, so permute the 16 B segments within each row instead
// (XOR swizzle): store at seg ^ swz(row), load kb ^ swz(row). The formula
// scales the row index by segs-per-row so it stays conflict-free across the
// PD_BS_KC A/B guards (KC=128 -> row>>1 & 3, KC=64 -> row>>2 & 1); the two
// rows of one afrag pair (r, r+8) always share a swizzle value. Bit-exact:
// pure layout, same bytes, same MMA inputs.
#define PD_BS_SWZ(row, wseg) ((((row) * (wseg)) >> 3) & ((wseg) - 1u))
#define PD_BS_P_WROW 48u
#define PD_BS_P_YROW 80u
#define PD_BS_P_SMEM (2u * 128u * (PD_BS_P_WROW + PD_BS_P_YROW))
// Warp-specialized MoE bs geometry: 8 consumer warps (the unchanged MMA +
// epilogue layout) plus dedicated producer warps that own all staging.
// Producer width is an empirical knob:
// gate_up wants 3 (dec-c32 814 us vs 837 at 4), down wants 4 (dec 422 us,
// pp512 back to parity from +4% at 3-and-below).
#ifndef PD_BS_S
#define PD_BS_S 2              // pipeline stages (per-stage tiles + scales; harness -D overridable)
#endif
#ifndef PD_BS_PW_GU
#define PD_BS_PW_GU 3u         // gate_up producer warps (harness -D overridable)
#endif
#ifndef PD_BS_PW_DN
#define PD_BS_PW_DN 4u         // down producer warps (harness -D overridable)
#endif
#define PD_BS_TH_GU (256u + 32u * PD_BS_PW_GU)
#define PD_BS_TH_DN (256u + 32u * PD_BS_PW_DN)
// one stage of tiles + the per-stage scale slices
#define PD_BS_GU_STAGE (2u * 128u * PD_BS_WROW_GU + PD_BS_BM * PD_BS_YROW_GU \
                        + 2u * 128u * PD_BS_KB_GU + PD_BS_BM * PD_BS_KB_GU)
#define PD_BS_DN_STAGE (128u * PD_BS_WROW_DN + PD_BS_BM * PD_BS_YROW_DN \
                        + 128u * PD_BS_KB_DN + PD_BS_BM * PD_BS_KB_DN)
// dynamic smem: 32 B mbarriers, bias floats (x2: item-parity slots for the
// persistent work loop - producers prefill item i+1's metadata while
// consumers still epilogue item i), then the stages (base stays 16-aligned
// for cp.async: 32 + bias bytes are both 16-multiples)
#define PD_BS_GU_SMEM (32u + 4u * 128u * 4u + PD_BS_S * PD_BS_GU_STAGE)
#define PD_BS_DN_SMEM (32u + 4u * 128u * 4u + PD_BS_S * PD_BS_DN_STAGE)
// KC=128 A/B builds still fit 2 CTAs/SM; KC=256 runs 1 (fine - producers
// keep chunks in flight without a co-resident block)
#define PD_BS_MINCTA(smem) ((smem) <= 47u * 1024u ? 2 : 1)

// (pd_ldm_x4 moved to attn/decode.cuh - the FA spec kernel needs it
// and 03 includes before this segment)

#if PD_BS_OK
__device__ __forceinline__ void pd_bs_mma(float d[4], uint32_t a0, uint32_t a1,
                                          uint32_t a2, uint32_t a3, uint32_t b0,
                                          uint32_t b1, uint32_t sfa, uint32_t sfb) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
        ".f32.e2m1.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
        "{%10}, {0, 0}, {%11}, {0, 0};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb));
}

// Same block-scale MMA with the scale byte picked by the IMMEDIATE byte-id
// (PTX `{scale-a, {byte-id, thread-id}}`): a u16 shared load carries both
// k32-blocks' ue8m0 bytes and the selector does the extract for free - no
// shift instructions, half the scale loads. KB is the k32 block index within
// the 64-deep chunk (0 or 1).
template <int KB>
__device__ __forceinline__ void pd_bs_mma_kb(float d[4], uint32_t a0, uint32_t a1,
                                             uint32_t a2, uint32_t a3, uint32_t b0,
                                             uint32_t b1, uint32_t sfa, uint32_t sfb) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X"
        ".f32.e2m1.e4m3.f32.ue8m0 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
        "{%10}, {%12, 0}, {%11}, {%12, 0};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb),
          "n"(KB));
}

// Load one m16n8k32 A-fragment set (4 regs) from a PACKED fp4 W row pair in
// the GGUF SPLIT nibble order the repack keeps (low nibble of byte j =
// element j, high = element j+16 within the 32-block). One u32 at byte 4*tq
// holds elements 4tq..4tq+3 in its LOW nibbles (the k-lo fragment half) and
// 16+4tq..16+4tq+3 in its HIGH nibbles (the k-hi half), so the k-pairing
// against a natural-order e4m3 B row is correct - verified against a CPU dot
// on the mxf8f6f4 MMA. (The original adjacent-nibble spread loader here
// expected INTERLEAVED bytes and mis-paired the split layout: every product
// term k paired W[perm(k)] with x[k]. Caught by the in-situ down_bs unit
// test; the greedy/exactness gates never exercised this path because the
// bs route is pinned off below b=4 and on non-sm_120a builds.)
__device__ __forceinline__ void pd_bs_afrag_split(uint32_t a[4], const unsigned char* w0,
                                                  const unsigned char* w8, uint32_t tq) {
    uint32_t p0, p8;
    memcpy(&p0, w0 + 4u * tq, 4);
    memcpy(&p8, w8 + 4u * tq, 4);
    a[0] = (p0 & 0x0F0F0F0Fu) << 2;   // e2m1 value sits at bits 5:2 of its byte
    a[1] = (p8 & 0x0F0F0F0Fu) << 2;
    a[2] = (p0 & 0xF0F0F0F0u) >> 2;
    a[3] = (p8 & 0xF0F0F0F0u) >> 2;
}

// mbarrier plumbing for the warp-specialized MoE bs kernels (the MARLIN /
// CUTLASS producer-consumer idiom, original implementation). arrive and
// wait are split-phase: no warp ever rendezvouses with the whole CTA, so
// the old per-chunk wait_group + 2x __syncthreads turnaround (which used
// to serialize every chunk) has nowhere to exist.
__device__ __forceinline__ void pd_bs_bar_init(uint64_t* bar, uint32_t count) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(a), "r"(count));
}
__device__ __forceinline__ void pd_bs_bar_arrive(uint64_t* bar) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(a) : "memory");
}
// async arrival: fires when this thread's outstanding cp.asyncs complete.
// .noinc = the arrival counts against the barrier's init count.
__device__ __forceinline__ void pd_bs_cp_arrive_noinc(uint64_t* bar) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" :: "r"(a) : "memory");
}
__device__ __forceinline__ uint32_t pd_bs_bar_try_wait(uint64_t* bar, uint32_t parity) {
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    uint32_t done;
    asm volatile("{\n\t.reg .pred p;\n\t"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
                 "selp.b32 %0, 1, 0, p;\n\t}"
                 : "=r"(done) : "r"(a), "r"(parity) : "memory");
    return done;
}
__device__ __forceinline__ void pd_bs_bar_wait(uint64_t* bar, uint32_t parity) {
    // backoff on miss: hundreds of threads spinning raw try_wait on one
    // smem address contend on the very LSU pipe the producers need
    while (!pd_bs_bar_try_wait(bar, parity)) { __nanosleep(32); }
}
#endif

// f32 -> e4m3 + ue8m0 per-32 activation quantize (the block-scale mma's B
// side). Power-of-2 scales (2^e with amax/2^e <= 448) so the hardware ue8m0
// path reproduces the arithmetic exactly; RN-even via the cuda_fp8 cvt.
// Shared per-thread tail: each thread owns 4 consecutive elements (one 16B
// load + one 4B store), an 8-lane group covers a 32-block, amax rides 3
// shfls. (The original one-warp-per-block shape ran at 0.2 TB/s and cost
// more than the GEMM it fed.) Bit-identical outputs to that version - same
// amax, same e pick, same fp8 cvt.
__device__ __forceinline__ void pd_e4m3_quant4(float4 v, uint32_t lane8,
                                               unsigned char* __restrict__ q,
                                               unsigned char* __restrict__ scale,
                                               uint32_t i) {
    float a = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    const uint32_t gm = 0xffu << ((threadIdx.x & 31u) & ~7u);  // this 8-lane group
    for (uint32_t off = 4; off > 0; off >>= 1)
        a = fmaxf(a, __shfl_xor_sync(gm, a, off));
    int e = 0;
    if (a > 0.0f) {
        // smallest e with a / 2^e <= 448  (448 = 1.75 * 2^8)
        int ex;
        float m = frexpf(a, &ex);            // a = m * 2^ex, m in [0.5, 1)
        e = ex - 9 + (m > 0.875f ? 1 : 0);   // 448 = 0.875 * 2^9
    }
    const float inv = ldexpf(1.0f, -e);
    const uint32_t p = (uint32_t)__nv_fp8_e4m3(v.x * inv).__x
                     | ((uint32_t)__nv_fp8_e4m3(v.y * inv).__x << 8)
                     | ((uint32_t)__nv_fp8_e4m3(v.z * inv).__x << 16)
                     | ((uint32_t)__nv_fp8_e4m3(v.w * inv).__x << 24);
    *(uint32_t*)(q + i) = p;
    if (lane8 == 0) scale[i >> 5] = (unsigned char)(e + 127);
}

// Fused rmsnorm -> per-32 e4m3 quantize (the prefill norm band):
// the two-kernel chain wrote 44MB of f32 normed per chunk-layer and read
// it right back (rmsnorm_batch 4-5 % + quantize_e4m3 1 % of the c32/pf8
// GPU). One row per block, 256 threads (the batch>=64 rmsnorm launcher
// width - reduction grouping matches, so values are BIT-IDENTICAL to
// rmsnorm_batch + quantize_e4m3 at the prefill row counts).
__global__ void pd_rmsnorm_e4m3_kernel(const float* __restrict__ x,
                                       const float* __restrict__ w,
                                       unsigned char* __restrict__ q,
                                       unsigned char* __restrict__ scale,
                                       uint32_t n, float eps) {
    const uint32_t b = blockIdx.x;
    const float* xb = x + (size_t)b * n;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    __shared__ float wsum[32];
    __shared__ float s_inv;
    float acc = 0.0f;
    const uint32_t n4 = n >> 2;
    const float4* x4 = reinterpret_cast<const float4*>(xb);
    for (uint32_t i = tid; i < n4; i += nth) {
        const float4 v = x4[i];
        acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    for (uint32_t s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s);
    const uint32_t warp = tid >> 5, lane = tid & 31u;
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        float sum = 0.0f;
        const uint32_t nwarps = (nth + 31u) >> 5;
        for (uint32_t wi = 0; wi < nwarps; ++wi) sum += wsum[wi];
        s_inv = 1.0f / sqrtf(sum / (float)n + eps);
    }
    __syncthreads();
    const float inv = s_inv;
    const float4* w4 = reinterpret_cast<const float4*>(w);
    unsigned char* qb = q + (size_t)b * n;
    unsigned char* sb = scale + (size_t)b * (n >> 5);
    for (uint32_t i = tid; i < n4; i += nth) {
        const float4 v = x4[i], wv = w4[i];
        float4 r;
        r.x = v.x * inv * wv.x;
        r.y = v.y * inv * wv.y;
        r.z = v.z * inv * wv.z;
        r.w = v.w * inv * wv.w;
        pd_e4m3_quant4(r, tid & 7u, qb, sb, i * 4u);
    }
}

PD_EXPORT
int pd_rmsnorm_e4m3(const void* x, const void* w, void* q, void* scale,
                    uint32_t n, float eps, uint32_t batch, void* stream) {
    if (n == 0 || batch == 0) return 0;
    if (n & 31u) return cudaErrorInvalidValue;
    pd_rmsnorm_e4m3_kernel<<<batch, 256u, 0, (cudaStream_t)stream>>>(
        (const float*)x, (const float*)w, (unsigned char*)q,
        (unsigned char*)scale, n, eps);
    return pd_launch_status();
}

// TI (attention streams): f16 input plane - pd_ld4f expands
// exactly, so the only class change is the producer's store rounding.
template <typename TI = float>
__global__ void pd_quantize_e4m3_kernel(const TI* __restrict__ x,
                                        unsigned char* __restrict__ q,
                                        unsigned char* __restrict__ scale, uint32_t n) {
    const uint32_t i = (blockIdx.x * 256u + threadIdx.x) * 4u;
    if (i >= n) return;  // n % 32 == 0: 8-lane groups exit whole, shfls stay valid
    pd_e4m3_quant4(pd_ld4f(x + i), threadIdx.x & 7u, q, scale, i);
}

PD_EXPORT
int pd_quantize_e4m3(const void* x, void* q, void* scale, uint32_t n, void* stream) {
    if (n == 0) return 0;
    pd_quantize_e4m3_kernel<float><<<(n / 4u + 255u) / 256u, 256u, 0, (cudaStream_t)stream>>>(
        (const float*)x, (unsigned char*)q, (unsigned char*)scale, n);
    return pd_launch_status();
}

// i16 twin (attention streams): x is an f16 plane. Appended as
// its own export per the ABI growth rule.
PD_EXPORT
int pd_quantize_e4m3_i16(const void* x, void* q, void* scale, uint32_t n,
                         uint32_t i16, void* stream) {
    if (n == 0) return 0;
    if (!i16)
        return pd_quantize_e4m3(x, q, scale, n, stream);
    pd_quantize_e4m3_kernel<__half><<<(n / 4u + 255u) / 256u, 256u, 0, (cudaStream_t)stream>>>(
        (const __half*)x, (unsigned char*)q, (unsigned char*)scale, n);
    return pd_launch_status();
}

// SwiGLU fused into the e4m3 quantize (the P6j pattern, block-scale class):
// the dense-bs ffn_down input never lands in memory as f32 - read gate+up
// once, silu(g)*u (exactly pd_swiglu's math), quantize in registers.
__global__ void pd_quantize_e4m3_swiglu_kernel(const float* __restrict__ gate,
                                               const float* __restrict__ up,
                                               unsigned char* __restrict__ q,
                                               unsigned char* __restrict__ scale,
                                               uint32_t n) {
    const uint32_t i = (blockIdx.x * 256u + threadIdx.x) * 4u;
    if (i >= n) return;
    const float4 g = *(const float4*)(gate + i);
    const float4 u = *(const float4*)(up + i);
    float4 v;
    v.x = (g.x / (1.0f + expf(-g.x))) * u.x;
    v.y = (g.y / (1.0f + expf(-g.y))) * u.y;
    v.z = (g.z / (1.0f + expf(-g.z))) * u.z;
    v.w = (g.w / (1.0f + expf(-g.w))) * u.w;
    pd_e4m3_quant4(v, threadIdx.x & 7u, q, scale, i);
}

// Decode-band add+rmsnorm with both outputs: xn (f32, still consumed by
// alpha/beta and the Q8 arms) AND the e4m3+scale staging the f8 GEMMs eat -
// one kernel replacing rmsnorm_batch + quantize_e4m3 per layer group
// (-2 launches/layer/tick). proj nullable: entry norm passes NULL, the
// post_norm site adds the mixer residual (x keeps its update, exactly
// add_rmsnorm_quant_mmq's contract). n % 32 == 0.
// PB16: proj arrives bf16 (the o16 prefill chain's residual) - same math,
// converted at load. f32 form unchanged (PB16=false).
template <bool PB16 = false>
__global__ void pd_add_rmsnorm_e4m3_xn_kernel(
    float* __restrict__ x, const void* __restrict__ proj,
    const float* __restrict__ w, float* __restrict__ xn,
    unsigned char* __restrict__ q, unsigned char* __restrict__ scale,
    uint32_t n, float eps) {
    const uint32_t b = blockIdx.x;
    float* xb = x + (size_t)b * n;
    float* xnb = xn + (size_t)b * n;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    // width-stable: f64 sumsq (f32 products) - see pd_norm_wide_nth_ws.
    // This kernel launched at a hardcoded 256 threads while every sibling
    // row-per-CTA norm already rode the wide election; widening an f32 sum is
    // the numerics-REFUSED form (+0.085 nats, worse 4/4 wikitext
    // slices), so the width comes with the f64 accumulator, not without it.
    __shared__ double wsum[32];
    __shared__ float s_inv;
    const uint32_t n4 = n >> 2;
    float4* x4 = reinterpret_cast<float4*>(xb);
    const float4* p4 = (!PB16 && proj)
        ? reinterpret_cast<const float4*>((const float*)proj + (size_t)b * n)
        : nullptr;
    const __nv_bfloat162* pb2 = (PB16 && proj)
        ? reinterpret_cast<const __nv_bfloat162*>((const __nv_bfloat16*)proj + (size_t)b * n)
        : nullptr;
    double acc = 0.0;
    for (uint32_t i = tid; i < n4; i += nth) {
        float4 v = x4[i];
        if (PB16 ? (pb2 != nullptr) : (p4 != nullptr)) {
            float4 p;
            if (PB16) {
                const __nv_bfloat162 lo = pb2[i * 2u], hi = pb2[i * 2u + 1u];
                p.x = __bfloat162float(lo.x); p.y = __bfloat162float(lo.y);
                p.z = __bfloat162float(hi.x); p.w = __bfloat162float(hi.y);
            } else {
                p = p4[i];
            }
            v.x += p.x; v.y += p.y; v.z += p.z; v.w += p.w;
            x4[i] = v;
        }
        acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    for (uint32_t sh = 16; sh > 0; sh >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, sh);
    const uint32_t warp = tid >> 5, lane = tid & 31u;
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        double sum = 0.0;
        const uint32_t nw = (nth + 31u) >> 5;
        for (uint32_t wi = 0; wi < nw; ++wi) sum += wsum[wi];
        s_inv = 1.0f / sqrtf((float)(sum / (double)n) + eps);
    }
    __syncthreads();
    const float inv = s_inv;
    const float4* w4 = reinterpret_cast<const float4*>(w);
    unsigned char* qb = q + (size_t)b * n;
    unsigned char* sb = scale + (size_t)b * (n >> 5);
    float4* xn4 = reinterpret_cast<float4*>(xnb);
    for (uint32_t i = tid; i < n4; i += nth) {
        const float4 v = x4[i], wv = w4[i];
        float4 r;
        r.x = v.x * inv * wv.x;
        r.y = v.y * inv * wv.y;
        r.z = v.z * inv * wv.z;
        r.w = v.w * inv * wv.w;
        xn4[i] = r;
        pd_e4m3_quant4(r, tid & 7u, qb, sb, i * 4u);
    }
}

PD_EXPORT
int pd_add_rmsnorm_e4m3_xn(void* x, const void* proj, const void* w, void* xn,
                           void* q, void* scale, uint32_t n, uint32_t batch,
                           float eps, void* stream) {
    if (n == 0 || batch == 0) return 0;
    if (n & 31u) return cudaErrorInvalidValue;
    // 256 threads on 16-32 CTAs is the latency-bound corner measured
    // at 2.14x (see pd_norm_wide_nth_ws). At c16 this kernel is 107
    // launches x 7.51 us = 807 us/tick, 8.5% of the die.
    const uint32_t nth = batch >= 64u ? pd_norm_wide_nth_ws(batch)
                                      : pd_norm_decode_nth();
    pd_add_rmsnorm_e4m3_xn_kernel<false><<<batch, nth, 0, (cudaStream_t)stream>>>(
        (float*)x, proj, (const float*)w, (float*)xn,
        (unsigned char*)q, (unsigned char*)scale, n, eps);
    return pd_launch_status();
}

// b16-proj twin (slot 257): the o16 prefill chain's post-norm residual is
// bf16; same contract otherwise.
PD_EXPORT
int pd_add_rmsnorm_e4m3_xn_b16(void* x, const void* proj, const void* w, void* xn,
                               void* q, void* scale, uint32_t n, uint32_t batch,
                               float eps, void* stream) {
    if (n == 0 || batch == 0) return 0;
    if (n & 31u) return cudaErrorInvalidValue;
    const uint32_t nth = batch >= 64u ? pd_norm_wide_nth_ws(batch)
                                      : pd_norm_decode_nth();
    pd_add_rmsnorm_e4m3_xn_kernel<true><<<batch, nth, 0, (cudaStream_t)stream>>>(
        (float*)x, proj, (const float*)w, (float*)xn,
        (unsigned char*)q, (unsigned char*)scale, n, eps);
    return pd_launch_status();
}

// ---------------------------------------------------------------------------
// granite's f32/Q8 twin of the e4m3 fusion above (round 3):
// scale_add + rmsnorm_batch + quantize_q8_sums as one launch.
//
// Why granite could use neither existing fusion: it scales the residual by a
// per-model residual_multiplier before adding (pd_scale_add_f32's x += w*y),
// and neither add_rmsnorm_batch nor the e4m3 twin above carries a multiplier -
// so granite decode ran all three separately and paid for it. Measured on
// granite-30b Q4_K_M at b=1, per graph node:
//     scale_add         0.192 ms / 128 launches
//     rmsnorm_batch     0.393 ms / 129
//     quantize_q8_sums  0.460 ms / 257
//                       1.045 ms across 515 launches, per token
// None of that is work: rmsnorm_batch moves 32 KB in 3.04 us, ~1% of what a
// 600 GB/s pass would take. At b=1 each is a one-or-few-block grid on an
// 84-SM die, so the lever is fewer/fatter launches, not tuning any of them.
//
// BIT-EXACT, and only free under the double-float accumulator. Fusing
// changes the sumsq's thread
// width - this is one row-per-CTA norm, where rmsnorm_batch runs its own width
// election. Under the double-float accumulator the sum is width-INVARIANT
// BITWISE, so the norm here equals the unfused one at any nth; that is exactly
// the property the DF switch bought, and under the old f64 accumulator this
// fusion would have had to inherit rmsnorm_batch's width to stay exact. The
// quantize half is exact by construction: amax is a max-reduce (order-free)
// and the per-16 sums are integer.
//
// Q8_0 geometry: 32 elements = 8 consecutive float4 threads, so amax rides
// lane bits 0..2 and the per-16 sums ride bits 0..1. n % 32 == 0 (launcher
// check) gives n4 % 8 == 0, and nth % 8 == 0, so i ≡ tid (mod 8) and every
// 8-thread group is active for the same iteration count - the butterflies
// never see a half-exited group. The masks are the GROUP, not 0xffffffff:
// when n4 < nth whole lanes never enter this loop, and a full-mask shuffle
// would then name threads that are not there.
template <int ACC>
__global__ void pd_add_rmsnorm_q8_xn_kernel(
    float* __restrict__ x, const float* __restrict__ proj,
    const float* __restrict__ w, float* __restrict__ xn,
    signed char* __restrict__ q, float* __restrict__ scale,
    float* __restrict__ sums, uint32_t n, float eps, float res_scale) {
    using A = typename pd_acc_of<ACC>::type;
    PD_PDL_ARM();  // cascade - the chain position the three launches held
    const uint32_t b = blockIdx.x;
    float* xb = x + (size_t)b * n;
    float* xnb = xn + (size_t)b * n;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    __shared__ A wsum[32];
    __shared__ float s_inv;
    const uint32_t n4 = n >> 2;
    float4* x4 = reinterpret_cast<float4*>(xb);
    const float4* p4 =
        proj ? reinterpret_cast<const float4*>(proj + (size_t)b * n) : nullptr;

    A acc;
    if constexpr (ACC == PD_ACC_DF) { acc.hi = 0.0f; acc.lo = 0.0f; } else { acc = (A)0; }
    for (uint32_t i = tid; i < n4; i += nth) {
        float4 v = x4[i];
        if (p4) {
            const float4 p = p4[i];
            // pd_scale_add_f32's exact form is x += w*y - scale on the
            // residual, multiplying first.
            v.x += res_scale * p.x;
            v.y += res_scale * p.y;
            v.z += res_scale * p.z;
            v.w += res_scale * p.w;
            x4[i] = v;
        }
        // products stay f32 in every mode - only the ACCUMULATE differs
        if constexpr (ACC == PD_ACC_DF) {
            pd_df_add(acc, v.x * v.x);
            pd_df_add(acc, v.y * v.y);
            pd_df_add(acc, v.z * v.z);
            pd_df_add(acc, v.w * v.w);
        } else {
            acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    }
    // Every thread reaches here (all exit the loop), so the full mask is right.
    for (uint32_t s = 16; s > 0; s >>= 1) {
        if constexpr (ACC == PD_ACC_DF) {
            pd_df o;
            o.hi = __shfl_down_sync(0xffffffffu, acc.hi, s);
            o.lo = __shfl_down_sync(0xffffffffu, acc.lo, s);
            acc = pd_df_merge(acc, o);
        } else {
            acc += __shfl_down_sync(0xffffffffu, acc, s);
        }
    }
    const uint32_t warp = tid >> 5, lane = tid & 31u;
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        const uint32_t nwarps = (nth + 31u) >> 5;
        double total;
        if constexpr (ACC == PD_ACC_DF) {
            pd_df sum; sum.hi = 0.0f; sum.lo = 0.0f;
            for (uint32_t wi = 0; wi < nwarps; ++wi) sum = pd_df_merge(sum, wsum[wi]);
            total = (double)sum.hi + (double)sum.lo;
        } else {
            A sum = (A)0;
            for (uint32_t wi = 0; wi < nwarps; ++wi) sum += wsum[wi];
            total = (double)sum;
        }
        s_inv = 1.0f / sqrtf((float)(total / (double)n) + eps);
    }
    __syncthreads();

    const float inv = s_inv;
    const float4* w4 = reinterpret_cast<const float4*>(w);
    float4* xn4 = reinterpret_cast<float4*>(xnb);
    signed char* qb = q + (size_t)b * n;
    float* sclb = scale + (size_t)b * (n >> 5);
    float* sumb = sums + (size_t)b * (n >> 4);
    const unsigned g8 = 0xffu << (lane & 24u);  // this thread's aligned 8-lane group
    const unsigned g4 = 0xfu << (lane & 28u);   // ... and its 4-lane half
    for (uint32_t i = tid; i < n4; i += nth) {
        const float4 v = x4[i], wv = w4[i];
        float4 r;
        // rmsnorm_batch's association, left to right: (v * inv) * w
        r.x = v.x * inv * wv.x;
        r.y = v.y * inv * wv.y;
        r.z = v.z * inv * wv.z;
        r.w = v.w * inv * wv.w;
        xn4[i] = r;

        // ---- pd_quantize_q8_sums, from the values still in registers ----
        float a = fmaxf(fmaxf(fabsf(r.x), fabsf(r.y)), fmaxf(fabsf(r.z), fabsf(r.w)));
        for (uint32_t s = 1u; s < 8u; s <<= 1) a = fmaxf(a, __shfl_xor_sync(g8, a, s));
        const float scl = a * (1.0f / 127.0f);
        const float qinv = scl > 0.0f ? 1.0f / scl : 0.0f;
        int q0 = __float2int_rn(r.x * qinv), q1 = __float2int_rn(r.y * qinv);
        int q2 = __float2int_rn(r.z * qinv), q3 = __float2int_rn(r.w * qinv);
        q0 = q0 < -127 ? -127 : (q0 > 127 ? 127 : q0);
        q1 = q1 < -127 ? -127 : (q1 > 127 ? 127 : q1);
        q2 = q2 < -127 ? -127 : (q2 > 127 ? 127 : q2);
        q3 = q3 < -127 ? -127 : (q3 > 127 ? 127 : q3);
        const uint32_t e0 = i * 4u;             // this thread's first element
        qb[e0 + 0u] = (signed char)q0;
        qb[e0 + 1u] = (signed char)q1;
        qb[e0 + 2u] = (signed char)q2;
        qb[e0 + 3u] = (signed char)q3;
        if ((i & 7u) == 0u) sclb[e0 >> 5] = scl;
        // per-16 sums: 16 elements = 4 threads, butterfly over lane bits 0..1
        int s16 = q0 + q1 + q2 + q3;
        for (uint32_t s = 1u; s < 4u; s <<= 1) s16 += __shfl_xor_sync(g4, s16, s);
        if ((i & 3u) == 0u) sumb[e0 >> 4] = (float)s16;
    }
}

PD_EXPORT
int pd_add_rmsnorm_q8_xn(void* x, const void* proj, const void* w, void* xn,
                         void* q, void* scale, void* sums, uint32_t n,
                         uint32_t batch, float eps, float res_scale,
                         void* stream) {
    if (n == 0 || batch == 0) return 0;
    // n % 32 is the Q8_0 block AND what makes the 8-thread groups lockstep.
    if (n & 31u) return cudaErrorInvalidValue;
    const uint32_t nth = batch >= 64u ? pd_norm_wide_nth_ws(batch)
                                      : pd_norm_decode_nth();
    const int acc = pd_norm_acc_mode();
    if (acc == PD_ACC_F64) {
        pd_add_rmsnorm_q8_xn_kernel<PD_ACC_F64><<<batch, nth, 0, (cudaStream_t)stream>>>(
            (float*)x, (const float*)proj, (const float*)w, (float*)xn,
            (signed char*)q, (float*)scale, (float*)sums, n, eps, res_scale);
    } else if (acc == PD_ACC_F32) {
        pd_add_rmsnorm_q8_xn_kernel<PD_ACC_F32><<<batch, nth, 0, (cudaStream_t)stream>>>(
            (float*)x, (const float*)proj, (const float*)w, (float*)xn,
            (signed char*)q, (float*)scale, (float*)sums, n, eps, res_scale);
    } else {
        pd_add_rmsnorm_q8_xn_kernel<PD_ACC_DF><<<batch, nth, 0, (cudaStream_t)stream>>>(
            (float*)x, (const float*)proj, (const float*)w, (float*)xn,
            (signed char*)q, (float*)scale, (float*)sums, n, eps, res_scale);
    }
    return pd_launch_status();
}

// FUSED-landing swiglu + e4m3 quant: reads the merged gate|up GEMM output
// ([tok][gate(ff)|up(ff)] rows, f32), applies the exact pd_swiglu formula,
// quantizes e4m3 in registers - one kernel instead of swiglu_fused +
// quantize_e4m3 (64 launches/tick + a full [b,ff] f32 round trip at decode).
__global__ void pd_swiglu_fused_e4m3_kernel(const float* __restrict__ fused,
                                            unsigned char* __restrict__ q,
                                            unsigned char* __restrict__ scale,
                                            uint32_t ff, uint32_t n) {
    const uint32_t i = (blockIdx.x * 256u + threadIdx.x) * 4u;
    if (i >= n) return;
    const uint32_t tok = i / ff, j = i % ff;
    const float* row = fused + (size_t)tok * 2u * ff;
    const float4 g = *(const float4*)(row + j);
    const float4 u = *(const float4*)(row + ff + j);
    float4 v;
    v.x = (g.x / (1.0f + expf(-g.x))) * u.x;
    v.y = (g.y / (1.0f + expf(-g.y))) * u.y;
    v.z = (g.z / (1.0f + expf(-g.z))) * u.z;
    v.w = (g.w / (1.0f + expf(-g.w))) * u.w;
    pd_e4m3_quant4(v, threadIdx.x & 7u, q, scale, i);
}

PD_EXPORT
int pd_swiglu_fused_e4m3(const void* fused, void* q, void* scale, uint32_t ff,
                         uint32_t n_rows, void* stream) {
    if (n_rows == 0 || ff == 0) return 0;
    if (ff & 31u) return cudaErrorInvalidValue;
    const uint32_t n = n_rows * ff;
    pd_swiglu_fused_e4m3_kernel<<<(n / 4u + 255u) / 256u, 256u, 0,
                                  (cudaStream_t)stream>>>(
        (const float*)fused, (unsigned char*)q, (unsigned char*)scale, ff, n);
    return pd_launch_status();
}

// bf16-input twin of pd_quantize_e4m3_swiglu (the O16 GEMM epilogue's
// consumer): the same silu formula in f32; gate/up arrive as bf16.
__global__ void pd_quantize_e4m3_swiglu_b16_kernel(
    const __nv_bfloat16* __restrict__ gate, const __nv_bfloat16* __restrict__ up,
    unsigned char* __restrict__ q, unsigned char* __restrict__ scale, uint32_t n) {
    const uint32_t i = (blockIdx.x * 256u + threadIdx.x) * 4u;
    if (i >= n) return;
    const __nv_bfloat162 g01 = *(const __nv_bfloat162*)(gate + i);
    const __nv_bfloat162 g23 = *(const __nv_bfloat162*)(gate + i + 2u);
    const __nv_bfloat162 u01 = *(const __nv_bfloat162*)(up + i);
    const __nv_bfloat162 u23 = *(const __nv_bfloat162*)(up + i + 2u);
    float gx = __bfloat162float(g01.x), gy = __bfloat162float(g01.y);
    float gz = __bfloat162float(g23.x), gw = __bfloat162float(g23.y);
    float4 v;
    v.x = (gx / (1.0f + expf(-gx))) * __bfloat162float(u01.x);
    v.y = (gy / (1.0f + expf(-gy))) * __bfloat162float(u01.y);
    v.z = (gz / (1.0f + expf(-gz))) * __bfloat162float(u23.x);
    v.w = (gw / (1.0f + expf(-gw))) * __bfloat162float(u23.y);
    pd_e4m3_quant4(v, threadIdx.x & 7u, q, scale, i);
}

// fused gate|up layout twin: one [rows][2*ff] bf16 buffer, gate cols
// [0,ff), up [ff,2ff) - the single-GEMM prefill FFN epilogue (per-element
// math and output indexing identical to pd_quantize_e4m3_swiglu_b16).
__global__ void pd_quantize_e4m3_swiglu_b16_gu_kernel(
    const __nv_bfloat16* __restrict__ gu, unsigned char* __restrict__ q,
    unsigned char* __restrict__ scale, uint32_t n, uint32_t ff) {
    const uint32_t i = (blockIdx.x * 256u + threadIdx.x) * 4u;
    if (i >= n) return;
    const uint32_t row = i / ff, col = i - row * ff;
    const __nv_bfloat16* g = gu + (size_t)row * 2u * ff + col;
    const __nv_bfloat162 g01 = *(const __nv_bfloat162*)(g);
    const __nv_bfloat162 g23 = *(const __nv_bfloat162*)(g + 2u);
    const __nv_bfloat162 u01 = *(const __nv_bfloat162*)(g + ff);
    const __nv_bfloat162 u23 = *(const __nv_bfloat162*)(g + ff + 2u);
    float gx = __bfloat162float(g01.x), gy = __bfloat162float(g01.y);
    float gz = __bfloat162float(g23.x), gw = __bfloat162float(g23.y);
    float4 v;
    v.x = (gx / (1.0f + expf(-gx))) * __bfloat162float(u01.x);
    v.y = (gy / (1.0f + expf(-gy))) * __bfloat162float(u01.y);
    v.z = (gz / (1.0f + expf(-gz))) * __bfloat162float(u23.x);
    v.w = (gw / (1.0f + expf(-gw))) * __bfloat162float(u23.y);
    pd_e4m3_quant4(v, threadIdx.x & 7u, q, scale, i);
}

PD_EXPORT
int pd_quantize_e4m3_swiglu_b16_gu(const void* gu, void* q, void* scale,
                                   uint32_t n, uint32_t ff, void* stream) {
    if (n == 0) return 0;
    if ((n & 31u) || (ff & 3u)) return cudaErrorInvalidValue;
    pd_quantize_e4m3_swiglu_b16_gu_kernel<<<(n / 4u + 255u) / 256u, 256u, 0,
                                            (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)gu, (unsigned char*)q, (unsigned char*)scale, n, ff);
    return pd_launch_status();
}

PD_EXPORT
int pd_quantize_e4m3_swiglu_b16(const void* gate, const void* up, void* q,
                                void* scale, uint32_t n, void* stream) {
    if (n == 0) return 0;
    if (n & 31u) return cudaErrorInvalidValue;
    pd_quantize_e4m3_swiglu_b16_kernel<<<(n / 4u + 255u) / 256u, 256u, 0,
                                         (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)gate, (const __nv_bfloat16*)up, (unsigned char*)q,
        (unsigned char*)scale, n);
    return pd_launch_status();
}

// GEGLU twin (gemma4's gelu_tanh pair): v = gelu_tanh(gate) * up, formula
// exactly pd_geglu's, quantized in registers - the f32 activation never
// lands, saving pd_geglu's full n_ff write + the quantize's re-read per
// FFN. Bit-identical values to geglu -> quantize_e4m3 (same formula, same
// scale pick, same cvt), so the F8R lanes keep their gated numeric class.
__global__ void pd_quantize_e4m3_geglu_kernel(const float* __restrict__ gate,
                                              const float* __restrict__ up,
                                              unsigned char* __restrict__ q,
                                              unsigned char* __restrict__ scale,
                                              uint32_t n) {
    const uint32_t i = (blockIdx.x * 256u + threadIdx.x) * 4u;
    if (i >= n) return;
    const float4 g = *(const float4*)(gate + i);
    const float4 u = *(const float4*)(up + i);
    float4 v;
    v.x = 0.5f * g.x * (1.0f + tanhf(0.79788456080286535587989211986876f * g.x
                                     * (1.0f + 0.044715f * g.x * g.x))) * u.x;
    v.y = 0.5f * g.y * (1.0f + tanhf(0.79788456080286535587989211986876f * g.y
                                     * (1.0f + 0.044715f * g.y * g.y))) * u.y;
    v.z = 0.5f * g.z * (1.0f + tanhf(0.79788456080286535587989211986876f * g.z
                                     * (1.0f + 0.044715f * g.z * g.z))) * u.z;
    v.w = 0.5f * g.w * (1.0f + tanhf(0.79788456080286535587989211986876f * g.w
                                     * (1.0f + 0.044715f * g.w * g.w))) * u.w;
    pd_e4m3_quant4(v, threadIdx.x & 7u, q, scale, i);
}

PD_EXPORT
int pd_quantize_e4m3_geglu(const void* gate, const void* up, void* q, void* scale,
                           uint32_t n, void* stream) {
    if (n == 0) return 0;
    pd_quantize_e4m3_geglu_kernel<<<(n / 4u + 255u) / 256u, 256u, 0,
                                    (cudaStream_t)stream>>>(
        (const float*)gate, (const float*)up, (unsigned char*)q,
        (unsigned char*)scale, n);
    return pd_launch_status();
}

// Fused batched rmsnorm -> e4m3 quantize (v11 norms rung): the normed f32
// row never lands - rmsnorm_batch wrote r*n_embd f32 that quantize_e4m3
// immediately re-read (~88MB round-trip per pair at prefill rows, plus a
// launch). Norm math clones pd_rmsnorm_batch_kernel exactly (same shfl
// reduction, same cross-warp combine, same exact 1/sqrtf) and the quant is
// pd_e4m3_quant4, so at the same block width the e4m3 outputs are
// bit-identical to the two-kernel chain. Launcher mirrors rmsnorm_batch's
// width-by-batch policy for that reason. n % 32 == 0 required.
__global__ void pd_rmsnorm_e4m3_batch_kernel(const float* __restrict__ x,
                                             const float* __restrict__ w,
                                             unsigned char* __restrict__ q,
                                             unsigned char* __restrict__ scale,
                                             uint32_t n, float eps) {
    const uint32_t b = blockIdx.x;
    const float* xb = x + (size_t)b * n;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    // width-stable: the sumsq accumulates in f64 (f32 PRODUCTS kept -
    // only the sum order varied with nth), so the f32-cast mean is
    // identical across thread widths and the launcher rides the _ws
    // election (probe p63f64_probe: bit-identical 256/512/1024, +0.1%).
    __shared__ double wsum[32];
    __shared__ float s_inv;
    double acc = 0.0;
    const uint32_t n4 = n >> 2;
    const float4* x4 = reinterpret_cast<const float4*>(xb);
    for (uint32_t i = tid; i < n4; i += nth) {
        float4 v = x4[i];
        acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    for (uint32_t s2 = 16; s2 > 0; s2 >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s2);
    const uint32_t warp = tid >> 5, lane = tid & 31u;
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        double sum = 0.0;
        const uint32_t nwarps = (nth + 31u) >> 5;
        for (uint32_t wi = 0; wi < nwarps; ++wi) sum += wsum[wi];
        s_inv = 1.0f / sqrtf((float)(sum / (double)n) + eps);
    }
    __syncthreads();
    const float inv = s_inv;
    const float4* w4 = reinterpret_cast<const float4*>(w);
    for (uint32_t i = tid; i < n4; i += nth) {
        float4 v = x4[i], wv = w4[i], r;
        r.x = v.x * inv * wv.x;
        r.y = v.y * inv * wv.y;
        r.z = v.z * inv * wv.z;
        r.w = v.w * inv * wv.w;
        // 8-lane float4 groups cover aligned 32-blocks (stride nth % 8 == 0)
        pd_e4m3_quant4(r, tid & 7u, q, scale, b * n + i * 4u);
    }
}

PD_EXPORT
int pd_rmsnorm_e4m3_batch(const void* x, const void* w, void* q, void* scale,
                          uint32_t n, uint32_t batch, float eps, void* stream) {
    if (n == 0 || batch == 0) return 0;
    if ((n & 31u) != 0) return cudaErrorInvalidValue;
    // width-by-batch mirrors pd_rmsnorm_batch (bit-identity per width class)
    const uint32_t nth = batch >= 64u ? pd_norm_wide_nth_ws(batch) : 1024u;
    pd_rmsnorm_e4m3_batch_kernel<<<batch, nth, 0, (cudaStream_t)stream>>>(
        (const float*)x, (const float*)w, (unsigned char*)q, (unsigned char*)scale,
        n, eps);
    return pd_launch_status();
}

// Row-scale twin of the fused norm (the elementwise-band front): the
// f8t decode arms consume (e4m3, f32 row scale) - pd_quantize_e4m3_row's
// format - so the per-32 fusion above can't serve them. Sections clone their
// parents exactly (rmsnorm_batch's float4 sumsq + 1/sqrtf; quantize_e4m3_row's
// exact row max + frexpf power-of-2 scale + fp8 emit), with the normed f32
// row stashed in smem instead of round-tripping through global - at the same
// block width the outputs are BIT-IDENTICAL to the two-kernel chain.
__global__ void pd_rmsnorm_e4m3_row_kernel(const float* __restrict__ x,
                                           const float* __restrict__ w,
                                           unsigned char* __restrict__ q,
                                           float* __restrict__ rscale,
                                           uint32_t n, float eps) {
    // PDL: let the next (dependent-launched) GEMM start its dep-free W
    // prefetch while this kernel runs; its griddepcontrol.wait still gates
    // every dependent read on our full completion (probe-proven semantics).
    // cascade: this kernel now also launches as a dependent, so gate
    // the body on full predecessor completion (no-op under plain launches).
    // PD_PDL_ARM (not raw asm): compiles to nothing below sm_90 - this kernel
    // builds for every arch and raw griddepcontrol breaks ptxas there.
    PD_PDL_ARM();

    extern __shared__ float rm_normed[];               // [n] f32 normed row
    const uint32_t b = blockIdx.x;
    const float* xb = x + (size_t)b * n;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    // width-stable: f64 sumsq (f32 products) - see pd_rmsnorm_e4m3_batch.
    // The max path reuses wsum; f32<->f64 round-trips of a float are exact.
    __shared__ double wsum[32];
    __shared__ float s_inv;
    __shared__ int s_e;
    double acc = 0.0;
    const uint32_t n4 = n >> 2;
    const float4* x4 = reinterpret_cast<const float4*>(xb);
    for (uint32_t i = tid; i < n4; i += nth) {
        float4 v = x4[i];
        acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    for (uint32_t s2 = 16; s2 > 0; s2 >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s2);
    const uint32_t warp = tid >> 5, lane = tid & 31u;
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        double sum = 0.0;
        const uint32_t nwarps = (nth + 31u) >> 5;
        for (uint32_t wi = 0; wi < nwarps; ++wi) sum += wsum[wi];
        s_inv = 1.0f / sqrtf((float)(sum / (double)n) + eps);
    }
    __syncthreads();
    const float inv = s_inv;
    const float4* w4 = reinterpret_cast<const float4*>(w);
    float m = 0.0f;
    for (uint32_t i = tid; i < n4; i += nth) {
        float4 v = x4[i], wv = w4[i], r;
        r.x = v.x * inv * wv.x;
        r.y = v.y * inv * wv.y;
        r.z = v.z * inv * wv.z;
        r.w = v.w * inv * wv.w;
        *(float4*)(rm_normed + (size_t)i * 4u) = r;
        m = fmaxf(m, fmaxf(fmaxf(fabsf(r.x), fabsf(r.y)),
                           fmaxf(fabsf(r.z), fabsf(r.w))));
    }
    for (uint32_t sh = 16; sh > 0; sh >>= 1)
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, sh));
    if (lane == 0) wsum[warp] = m;
    __syncthreads();
    if (tid == 0) {
        float mm = 0.0f;
        for (uint32_t wi = 0; wi < ((nth + 31u) >> 5); ++wi) mm = fmaxf(mm, (float)wsum[wi]);
        int e = 0;
        if (mm > 0.0f) {
            int ex;
            float fr = frexpf(mm, &ex);
            e = ex - 9 + (fr > 0.875f ? 1 : 0);
        }
        s_e = e;
        rscale[b] = ldexpf(1.0f, e);
    }
    __syncthreads();
    const float qinv = ldexpf(1.0f, -s_e);
    unsigned char* qr = q + (size_t)b * n;
    for (uint32_t i = tid; i < n4; i += nth) {
        const float4 v = *(const float4*)(rm_normed + (size_t)i * 4u);
        uchar4 o;
        o.x = __nv_fp8_e4m3(v.x * qinv).__x;
        o.y = __nv_fp8_e4m3(v.y * qinv).__x;
        o.z = __nv_fp8_e4m3(v.z * qinv).__x;
        o.w = __nv_fp8_e4m3(v.w * qinv).__x;
        *(uchar4*)(qr + (size_t)i * 4u) = o;
    }
}

PD_EXPORT
int pd_rmsnorm_e4m3_row(const void* x, const void* w, void* q, void* rscale,
                        uint32_t n, float eps, uint32_t rows, void* stream) {
    if (n == 0 || rows == 0) return 0;
    if ((n & 3u) != 0) return cudaErrorInvalidValue;
    const uint32_t nth = rows >= 64u ? pd_norm_wide_nth_ws(rows) : 1024u;
    pd_pdl_go(pd_rmsnorm_e4m3_row_kernel, rows, nth, n * 4u,
              (cudaStream_t)stream,
              (const float*)x, (const float*)w, (unsigned char*)q, (float*)rscale,
              n, eps);
    return pd_launch_status();
}

// The full band-boundary fusion: residual-add + post-norm + next band's
// pre-norm + row-scale e4m3, one kernel per row. Replaces the 3-launch chain
// rmsnorm_add_scale -> rmsnorm_batch -> quantize_e4m3_row (the decode band's
// per-layer elementwise tax, ~21% of the c32 GPU in the v30 profile).
// Section 1 clones pd_rmsnorm_add_scale_kernel exactly (scalar sumsq +
// rsqrtf, x = (x + proj*inv*postw)*s); sections 2-3 are the row fusion
// above on the UPDATED x (visible to the block after __syncthreads).
// Same widths as the parents -> outputs bit-identical to the chain.
template <typename TP = float>
__global__ void pd_addnorm_e4m3_row_kernel(float* __restrict__ x,
                                           const TP* __restrict__ proj,
                                           const float* __restrict__ postw,
                                           const float* __restrict__ prew,
                                           unsigned char* __restrict__ q,
                                           float* __restrict__ rscale,
                                           uint32_t n, float eps, float s,
                                           uint32_t nzp) {
    // PDL: let the next (dependent-launched) GEMM start its dep-free W
    // prefetch while this kernel runs; its griddepcontrol.wait still gates
    // every dependent read on our full completion (probe-proven semantics).
    // cascade: this kernel now also launches as a dependent, so gate
    // the body on full predecessor completion (no-op under plain launches).
    // PD_PDL_ARM (not raw asm): compiles to nothing below sm_90 - this kernel
    // builds for every arch and raw griddepcontrol breaks ptxas there.
    PD_PDL_ARM();

    extern __shared__ float an_normed[];               // [n] f32 normed row
    const uint32_t b = blockIdx.x;
    const TP* pb = proj + (size_t)b * n;
    float* xb = x + (size_t)b * n;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    // width-stable: both sumsq reductions accumulate in f64 (f32
    // products kept), so nth stops being a numerics choice and the
    // launcher rides the _ws auto election - see pd_norm_wide_nth_ws and
    // probe p63f64_probe (bit-identical across 256/512/1024, +0.1% cost).
    // wsum is shared with the row-max pass; float->double->float is exact.
    __shared__ double wsum[32];
    __shared__ float s_inv;
    __shared__ int s_e;
    // section 1: post-norm + residual + stream scale (rmsnorm_add_scale clone)
    // nzp > 1 (consumer-side K-split absorption): proj is the GEMM's
    // nz partial PLANES - sum them ascending-z (the combine kernel's exact
    // order -> bit-equal) into the an_normed staging row so section 1 reads
    // partials once. nzp == 1 keeps the original two direct reads.
    const size_t np = (size_t)gridDim.x * n;
    double acc = 0.0;
    if (nzp > 1u) {
        for (uint32_t i = tid; i < n; i += nth) {
            float v = (float)pb[i];
            for (uint32_t z = 1; z < nzp; ++z) v += (float)pb[(size_t)z * np + i];
            an_normed[i] = v;
            acc += v * v;
        }
    } else {
        // stage proj in the an_normed row exactly as the nzp>1 path
        // already does, so section 1's second pass reads smem instead of
        // re-reading proj from HBM. Same value, same order -> bit-identical.
        for (uint32_t i = tid; i < n; i += nth) {
            float v = (float)pb[i];
            an_normed[i] = v;
            acc += v * v;
        }
    }
    for (uint32_t o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    const uint32_t warp = tid >> 5, lane = tid & 31u;
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        double sum = 0.0;
        for (uint32_t wi = 0; wi < (nth + 31u) >> 5; ++wi) sum += wsum[wi];
        s_inv = rsqrtf((float)(sum / (double)n) + eps);
    }
    __syncthreads();
    {
        // One path now (proj is staged for both nzp classes), and the
        // updated x goes back into the staging row as well as to global -
        // sections 2 and 3 then read it from smem instead of making two more
        // dependent HBM round trips. Arithmetic and order are untouched, so
        // the result stays bit-identical to the 3-kernel chain this replaces.
        const float inv1 = s_inv;
        for (uint32_t i = tid; i < n; i += nth) {
            const float xn = (xb[i] + an_normed[i] * inv1 * postw[i]) * s;
            xb[i] = xn;
            an_normed[i] = xn;
        }
    }
    __syncthreads();
    // section 2: pre-norm of the updated x (rmsnorm_batch float4 clone)
    double acc2 = 0.0;
    const uint32_t n4 = n >> 2;
    // reads the staged copy, not xb - same bytes, no HBM traffic. Dynamic
    // shared memory is 16B-aligned, so the float4 view is legal.
    const float4* x4 = reinterpret_cast<const float4*>(an_normed);
    for (uint32_t i = tid; i < n4; i += nth) {
        float4 v = x4[i];
        acc2 += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    for (uint32_t s2 = 16; s2 > 0; s2 >>= 1) acc2 += __shfl_down_sync(0xffffffffu, acc2, s2);
    if (lane == 0) wsum[warp] = acc2;
    __syncthreads();
    if (tid == 0) {
        double sum = 0.0;
        const uint32_t nwarps = (nth + 31u) >> 5;
        for (uint32_t wi = 0; wi < nwarps; ++wi) sum += wsum[wi];
        s_inv = 1.0f / sqrtf((float)(sum / (double)n) + eps);
    }
    __syncthreads();
    const float inv2 = s_inv;
    const float4* w4 = reinterpret_cast<const float4*>(prew);
    float m = 0.0f;
    for (uint32_t i = tid; i < n4; i += nth) {
        float4 v = x4[i], wv = w4[i], r;
        r.x = v.x * inv2 * wv.x;
        r.y = v.y * inv2 * wv.y;
        r.z = v.z * inv2 * wv.z;
        r.w = v.w * inv2 * wv.w;
        *(float4*)(an_normed + (size_t)i * 4u) = r;
        m = fmaxf(m, fmaxf(fmaxf(fabsf(r.x), fabsf(r.y)),
                           fmaxf(fabsf(r.z), fabsf(r.w))));
    }
    // section 3: exact row max -> power-of-2 scale -> fp8 emit (row-quant clone)
    for (uint32_t sh = 16; sh > 0; sh >>= 1)
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, sh));
    if (lane == 0) wsum[warp] = m;
    __syncthreads();
    if (tid == 0) {
        float mm = 0.0f;
        for (uint32_t wi = 0; wi < ((nth + 31u) >> 5); ++wi) mm = fmaxf(mm, (float)wsum[wi]);
        int e = 0;
        if (mm > 0.0f) {
            int ex;
            float fr = frexpf(mm, &ex);
            e = ex - 9 + (fr > 0.875f ? 1 : 0);
        }
        s_e = e;
        rscale[b] = ldexpf(1.0f, e);
    }
    __syncthreads();
    const float qinv = ldexpf(1.0f, -s_e);
    unsigned char* qr = q + (size_t)b * n;
    for (uint32_t i = tid; i < n4; i += nth) {
        const float4 v = *(const float4*)(an_normed + (size_t)i * 4u);
        uchar4 o;
        o.x = __nv_fp8_e4m3(v.x * qinv).__x;
        o.y = __nv_fp8_e4m3(v.y * qinv).__x;
        o.z = __nv_fp8_e4m3(v.z * qinv).__x;
        o.w = __nv_fp8_e4m3(v.w * qinv).__x;
        *(uchar4*)(qr + (size_t)i * 4u) = o;
    }
}

PD_EXPORT
int pd_addnorm_e4m3_row(void* x, const void* proj, const void* postw,
                        const void* prew, void* q, void* rscale, uint32_t n,
                        float eps, float s, uint32_t rows, void* stream) {
    if (n == 0 || rows == 0) return 0;
    if ((n & 3u) != 0) return cudaErrorInvalidValue;
    const uint32_t nth = rows >= 64u ? pd_norm_wide_nth_ws(rows) : 1024u;
    pd_pdl_go(pd_addnorm_e4m3_row_kernel<float>, rows, nth, n * 4u,
              (cudaStream_t)stream,
              (float*)x, (const float*)proj, (const float*)postw, (const float*)prew,
              (unsigned char*)q, (float*)rscale, n, eps, s, 1u);
    return pd_launch_status();
}

// ---- qwen twin: PLAIN residual add, no post-norm ---------------------------
// pd_addnorm_e4m3_row's section 1 is gemma's shape (x = (x + proj*inv*postw)*s
// -- the post-attention norm folded into the residual). qwen35 does a bare
// x += proj, so it cannot use that kernel and had to keep running
// pd_add_rmsnorm_batch (5.53 us at grid 1x1) + pd_quantize_e4m3_row1p (2.66)
// as two launches, 64 of each per decode tick. This twin keeps sections 2-3
// verbatim and replaces section 1 with the float4 add from
// pd_add_rmsnorm_batch_kernel, so the result is BIT-IDENTICAL to that chain:
// same add, same square-sum order over the summed values, same 1.0f/sqrtf
// (not rsqrtf -- section 1 of the gemma parent uses rsqrtf, this one must
// not), same pre-norm association, same exact row max and exponent.
__global__ void pd_add_rmsnorm_e4m3_row_kernel(float* __restrict__ x,
                                               const float* __restrict__ proj,
                                               const float* __restrict__ prew,
                                               unsigned char* __restrict__ q,
                                               float* __restrict__ rscale,
                                               uint32_t n, float eps) {
    PD_PDL_ARM();
    extern __shared__ float aq_normed[];               // [n] f32 normed row
    const uint32_t b = blockIdx.x;
    float* xb = x + (size_t)b * n;
    const float* pb = proj + (size_t)b * n;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    // width-stable: f64 sumsq (f32 products) - see pd_norm_wide_nth_ws.
    __shared__ double wsum[32];
    __shared__ float s_inv;
    __shared__ int s_e;
    const uint32_t n4 = n >> 2;
    const uint32_t warp = tid >> 5, lane = tid & 31u;
    // section 1: x += proj, write back, sum squares (pd_add_rmsnorm_batch
    // vectorized branch, verbatim)
    double acc = 0.0;
    {
        float4* x4 = reinterpret_cast<float4*>(xb);
        const float4* p4 = reinterpret_cast<const float4*>(pb);
        for (uint32_t i = tid; i < n4; i += nth) {
            float4 v = x4[i];
            const float4 pv = p4[i];
            v.x += pv.x; v.y += pv.y; v.z += pv.z; v.w += pv.w;
            x4[i] = v;
            acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    }
    for (uint32_t sh = 16; sh > 0; sh >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, sh);
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        double sum = 0.0;
        const uint32_t nwarps = (nth + 31u) >> 5;
        for (uint32_t wi = 0; wi < nwarps; ++wi) sum += wsum[wi];
        s_inv = 1.0f / sqrtf((float)(sum / (double)n) + eps);
    }
    __syncthreads();
    // section 2: normed row -> smem, track the exact row max
    const float inv = s_inv;
    float m = 0.0f;
    {
        const float4* x4 = reinterpret_cast<const float4*>(xb);
        const float4* w4 = reinterpret_cast<const float4*>(prew);
        for (uint32_t i = tid; i < n4; i += nth) {
            float4 v = x4[i], wv = w4[i], r;
            r.x = v.x * inv * wv.x;
            r.y = v.y * inv * wv.y;
            r.z = v.z * inv * wv.z;
            r.w = v.w * inv * wv.w;
            *(float4*)(aq_normed + (size_t)i * 4u) = r;
            m = fmaxf(m, fmaxf(fmaxf(fabsf(r.x), fabsf(r.y)),
                               fmaxf(fabsf(r.z), fabsf(r.w))));
        }
    }
    // section 3: row max -> power-of-2 scale -> fp8 emit
    for (uint32_t sh = 16; sh > 0; sh >>= 1)
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, sh));
    if (lane == 0) wsum[warp] = m;
    __syncthreads();
    if (tid == 0) {
        float mm = 0.0f;
        for (uint32_t wi = 0; wi < ((nth + 31u) >> 5); ++wi) mm = fmaxf(mm, (float)wsum[wi]);
        int e = 0;
        if (mm > 0.0f) {
            int ex;
            float fr = frexpf(mm, &ex);
            e = ex - 9 + (fr > 0.875f ? 1 : 0);
        }
        s_e = e;
        rscale[b] = ldexpf(1.0f, e);
    }
    __syncthreads();
    const float qinv = ldexpf(1.0f, -s_e);
    unsigned char* qr = q + (size_t)b * n;
    for (uint32_t i = tid; i < n4; i += nth) {
        const float4 v = *(const float4*)(aq_normed + (size_t)i * 4u);
        uchar4 o;
        o.x = __nv_fp8_e4m3(v.x * qinv).__x;
        o.y = __nv_fp8_e4m3(v.y * qinv).__x;
        o.z = __nv_fp8_e4m3(v.z * qinv).__x;
        o.w = __nv_fp8_e4m3(v.w * qinv).__x;
        *(uchar4*)(qr + (size_t)i * 4u) = o;
    }
}

PD_EXPORT
int pd_add_rmsnorm_e4m3_row(void* x, const void* proj, const void* prew,
                            void* q, void* rscale, uint32_t n, float eps,
                            uint32_t rows, void* stream) {
    if (n == 0 || rows == 0) return 0;
    if ((n & 3u) != 0) return cudaErrorInvalidValue;
    // same width election as both parents
    const uint32_t nth = rows >= 64u ? pd_norm_wide_nth_ws(rows) : pd_norm_decode_nth();
    pd_pdl_go(pd_add_rmsnorm_e4m3_row_kernel, rows, nth, n * 4u,
              (cudaStream_t)stream, (float*)x, (const float*)proj,
              (const float*)prew, (unsigned char*)q, (float*)rscale, n, eps);
    return pd_launch_status();
}

// p16 twin: `proj` bytes are bf16 (the o16 GEMM epilogue's
// stream); everything else identical. Appended per the ABI growth rule.
PD_EXPORT
int pd_addnorm_e4m3_row2(void* x, const void* proj, const void* postw,
                         const void* prew, void* q, void* rscale, uint32_t n,
                         float eps, float s, uint32_t rows, uint32_t p16,
                         void* stream) {
    if (n == 0 || rows == 0) return 0;
    if ((n & 3u) != 0) return cudaErrorInvalidValue;
    const uint32_t nth = rows >= 64u ? pd_norm_wide_nth_ws(rows) : 1024u;
    if (p16)
        pd_pdl_go(pd_addnorm_e4m3_row_kernel<__nv_bfloat16>, rows, nth, n * 4u,
                  (cudaStream_t)stream,
                  (float*)x, (const __nv_bfloat16*)proj, (const float*)postw,
                  (const float*)prew, (unsigned char*)q, (float*)rscale, n, eps,
                  s, 1u);
    else
        pd_pdl_go(pd_addnorm_e4m3_row_kernel<float>, rows, nth, n * 4u,
                  (cudaStream_t)stream,
                  (float*)x, (const float*)proj, (const float*)postw,
                  (const float*)prew, (unsigned char*)q, (float*)rscale, n, eps,
                  s, 1u);
    return pd_launch_status();
}

// nz-aware twin - `proj` is the GEMM's nz partial planes; the kernel
// absorbs the fixed-z sum at full grid parallelism (the combine launch and
// its combined-buffer round trip both disappear).
PD_EXPORT
int pd_addnorm_e4m3_nz(void* x, const void* proj, const void* postw, const void* prew,
                       void* q, void* rscale, uint32_t n, float eps, float s,
                       uint32_t rows, uint32_t nzp, void* stream) {
    if (n == 0 || rows == 0) return 0;
    if ((n & 3u) != 0 || nzp == 0) return cudaErrorInvalidValue;
    const uint32_t nth = rows >= 64u ? pd_norm_wide_nth_ws(rows) : 1024u;
    pd_pdl_go(pd_addnorm_e4m3_row_kernel<float>, rows, nth, n * 4u,
              (cudaStream_t)stream,
              (float*)x, (const float*)proj, (const float*)postw, (const float*)prew,
              (unsigned char*)q, (float*)rscale, n, eps, s, nzp);
    return pd_launch_status();
}

// Per-32 twin of the band-boundary fusion (the f8a/f8r wide-decode band,
// where the row twin's r<=64 gate never fires): residual-add + post-norm +
// next band's pre-norm + per-32 e4m3, one kernel per row. Replaces the
// 2-launch pair rmsnorm_add_scale -> rmsnorm_e4m3_batch - at gemma4 r=128
// that pair is launch-bound (14.6us x 120/tick, 128 thin CTAs on 188 SMs)
// and re-reads the just-written x row from global. Section 1 clones
// pd_rmsnorm_add_scale_kernel exactly (scalar sumsq + rsqrtf); section 2 is
// pd_rmsnorm_e4m3_batch_kernel on the UPDATED x (float4 sumsq + 1/sqrtf +
// pd_e4m3_quant4). Same width policy as both parents -> outputs
// bit-identical to the chain.
__global__ void pd_addnorm_e4m3_b32_kernel(float* __restrict__ x,
                                           const float* __restrict__ proj,
                                           const float* __restrict__ postw,
                                           const float* __restrict__ prew,
                                           unsigned char* __restrict__ q,
                                           unsigned char* __restrict__ scale,
                                           uint32_t n, float eps, float s) {
    // PDL: same contract as the row twin - the dependent GEMM may start its
    // dep-free W prefetch under us; its griddepcontrol.wait still gates
    // dependent reads on our full completion.
    PD_PDL_ARM();

    const uint32_t b = blockIdx.x;
    const float* pb = proj + (size_t)b * n;
    float* xb = x + (size_t)b * n;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    // width-stable: f64 sumsq (f32 products) - see pd_norm_wide_nth_ws.
    __shared__ double wsum[32];
    __shared__ float s_inv;
    // section 1: post-norm + residual + stream scale (rmsnorm_add_scale clone)
    double acc = 0.0;
    for (uint32_t i = tid; i < n; i += nth) {
        float v = pb[i];
        acc += v * v;
    }
    for (uint32_t o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
    const uint32_t warp = tid >> 5, lane = tid & 31u;
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        double sum = 0.0;
        for (uint32_t wi = 0; wi < (nth + 31u) >> 5; ++wi) sum += wsum[wi];
        s_inv = rsqrtf((float)(sum / (double)n) + eps);
    }
    __syncthreads();
    {
        const float inv1 = s_inv;
        for (uint32_t i = tid; i < n; i += nth) {
            xb[i] = (xb[i] + pb[i] * inv1 * postw[i]) * s;
        }
    }
    __syncthreads();
    // section 2: pre-norm + per-32 quant of the updated x (rmsnorm_e4m3_batch
    // clone - the row stays L1/L2-hot from section 1's stores)
    double acc2 = 0.0;
    const uint32_t n4 = n >> 2;
    const float4* x4 = reinterpret_cast<const float4*>(xb);
    for (uint32_t i = tid; i < n4; i += nth) {
        float4 v = x4[i];
        acc2 += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    for (uint32_t s2 = 16; s2 > 0; s2 >>= 1) acc2 += __shfl_down_sync(0xffffffffu, acc2, s2);
    if (lane == 0) wsum[warp] = acc2;
    __syncthreads();
    if (tid == 0) {
        double sum = 0.0;
        const uint32_t nwarps = (nth + 31u) >> 5;
        for (uint32_t wi = 0; wi < nwarps; ++wi) sum += wsum[wi];
        s_inv = 1.0f / sqrtf((float)(sum / (double)n) + eps);
    }
    __syncthreads();
    const float inv2 = s_inv;
    const float4* w4 = reinterpret_cast<const float4*>(prew);
    for (uint32_t i = tid; i < n4; i += nth) {
        float4 v = x4[i], wv = w4[i], r;
        r.x = v.x * inv2 * wv.x;
        r.y = v.y * inv2 * wv.y;
        r.z = v.z * inv2 * wv.z;
        r.w = v.w * inv2 * wv.w;
        pd_e4m3_quant4(r, tid & 7u, q, scale, b * n + i * 4u);
    }
}

PD_EXPORT
int pd_addnorm_e4m3_b32(void* x, const void* proj, const void* postw,
                        const void* prew, void* q, void* scale, uint32_t n,
                        float eps, float s, uint32_t rows, void* stream) {
    if (n == 0 || rows == 0) return 0;
    if ((n & 31u) != 0) return cudaErrorInvalidValue;
    // width-by-rows mirrors both parents (bit-identity per width class)
    const uint32_t nth = rows >= 64u ? pd_norm_wide_nth_ws(rows) : 1024u;
    pd_pdl_go(pd_addnorm_e4m3_b32_kernel, rows, nth, 0,
              (cudaStream_t)stream,
              (float*)x, (const float*)proj, (const float*)postw,
              (const float*)prew, (unsigned char*)q, (unsigned char*)scale,
              n, eps, s);
    return pd_launch_status();
}

// Fused-plane GEGLU quantize (verify-GEMM dedup rung): gate|up arrive as one
// GEMM output [rows][2*n_ff] (per-row [gate|up]) from the fused weight
// plane, so the pair sits at a row stride of 2*n_ff instead of two dense
// buffers. Same formula/scale/cvt as pd_quantize_e4m3_geglu -> identical
// values; only the input addressing differs.
template <int ACT>
__global__ void pd_quantize_e4m3_glu2_kernel(const float* __restrict__ gu,
                                             unsigned char* __restrict__ q,
                                             unsigned char* __restrict__ scale,
                                             uint32_t n_ff, uint32_t n) {
    const uint32_t i = (blockIdx.x * 256u + threadIdx.x) * 4u;
    if (i >= n) return;
    const uint32_t row = i / n_ff, col = i - row * n_ff;
    const float* base = gu + (size_t)row * 2u * n_ff + col;
    const float4 g = *(const float4*)base;
    const float4 u = *(const float4*)(base + n_ff);
    float4 v;
    v.x = pd_glu_act<ACT>(g.x) * u.x;
    v.y = pd_glu_act<ACT>(g.y) * u.y;
    v.z = pd_glu_act<ACT>(g.z) * u.z;
    v.w = pd_glu_act<ACT>(g.w) * u.w;
    pd_e4m3_quant4(v, threadIdx.x & 7u, q, scale, i);
}

template <int ACT>
static inline int pd_quantize_e4m3_glu2_launch(const void* gu, void* q, void* scale,
                                               uint32_t n_ff, uint32_t rows,
                                               void* stream) {
    const uint32_t n = n_ff * rows;
    if (n == 0) return 0;
    pd_quantize_e4m3_glu2_kernel<ACT><<<(n / 4u + 255u) / 256u, 256u, 0,
                                        (cudaStream_t)stream>>>(
        (const float*)gu, (unsigned char*)q, (unsigned char*)scale, n_ff, n);
    return pd_launch_status();
}

PD_EXPORT
int pd_quantize_e4m3_geglu2(const void* gu, void* q, void* scale,
                            uint32_t n_ff, uint32_t rows, void* stream) {
    return pd_quantize_e4m3_glu2_launch<PD_ACT_GELU>(gu, q, scale, n_ff, rows, stream);
}

PD_EXPORT
int pd_quantize_e4m3_swiglu2(const void* gu, void* q, void* scale,
                             uint32_t n_ff, uint32_t rows, void* stream) {
    return pd_quantize_e4m3_glu2_launch<PD_ACT_SILU>(gu, q, scale, n_ff, rows, stream);
}

// Interleaved-plane twin: the gu plane repacked by pd_f8w_repack_lin_gui
// permutes GEMM output rows so pair p sits at (p>>3)*16+(p&7) with its up
// partner 8 rows later. Same formula/scale/cvt as geglu2 -> identical bytes;
// only the gate/up addressing differs. Serves the interleaved plane's
// NON-fused consumers (the r<=31 lin band and b=1); the r>=32 chunk band
// skips this kernel entirely via the pd_f8_gemm_lin_gu fused epilogue.
template <int ACT>
__global__ void pd_quantize_e4m3_glu2i_kernel(const float* __restrict__ gu,
                                              unsigned char* __restrict__ q,
                                              unsigned char* __restrict__ scale,
                                              uint32_t n_ff, uint32_t n) {
    const uint32_t i = (blockIdx.x * 256u + threadIdx.x) * 4u;
    if (i >= n) return;
    const uint32_t row = i / n_ff, col = i - row * n_ff;
    // 4 consecutive pairs stay inside one 8-row group (col % 4 == 0), so
    // both float4 loads are contiguous like the plain kernel's
    const float* base = gu + (size_t)row * 2u * n_ff + (col >> 3) * 16u + (col & 7u);
    const float4 g = *(const float4*)base;
    const float4 u = *(const float4*)(base + 8u);
    float4 v;
    v.x = pd_glu_act<ACT>(g.x) * u.x;
    v.y = pd_glu_act<ACT>(g.y) * u.y;
    v.z = pd_glu_act<ACT>(g.z) * u.z;
    v.w = pd_glu_act<ACT>(g.w) * u.w;
    pd_e4m3_quant4(v, threadIdx.x & 7u, q, scale, i);
}

template <int ACT>
static inline int pd_quantize_e4m3_glu2i_launch(const void* gu, void* q, void* scale,
                                                uint32_t n_ff, uint32_t rows,
                                                void* stream) {
    const uint32_t n = n_ff * rows;
    if (n == 0) return 0;
    if (n_ff & 7u) return cudaErrorInvalidValue;  // pair groups are 8-wide
    pd_quantize_e4m3_glu2i_kernel<ACT><<<(n / 4u + 255u) / 256u, 256u, 0,
                                         (cudaStream_t)stream>>>(
        (const float*)gu, (unsigned char*)q, (unsigned char*)scale, n_ff, n);
    return pd_launch_status();
}

PD_EXPORT
int pd_quantize_e4m3_swiglu2i(const void* gu, void* q, void* scale,
                              uint32_t n_ff, uint32_t rows, void* stream) {
    return pd_quantize_e4m3_glu2i_launch<PD_ACT_SILU>(gu, q, scale, n_ff, rows, stream);
}

PD_EXPORT
int pd_quantize_e4m3_geglu2i(const void* gu, void* q, void* scale,
                             uint32_t n_ff, uint32_t rows, void* stream) {
    return pd_quantize_e4m3_glu2i_launch<PD_ACT_GELU>(gu, q, scale, n_ff, rows, stream);
}

PD_EXPORT
int pd_quantize_e4m3_swiglu(const void* gate, const void* up, void* q, void* scale,
                            uint32_t n, void* stream) {
    if (n == 0) return 0;
    pd_quantize_e4m3_swiglu_kernel<<<(n / 4u + 255u) / 256u, 256u, 0,
                                     (cudaStream_t)stream>>>(
        (const float*)gate, (const float*)up, (unsigned char*)q,
        (unsigned char*)scale, n);
    return pd_launch_status();
}

// Block-scale gate+up+swiglu over the sorted MoE layout: grid (max_blocks,
// ff/128), BM=32 sorted tokens x 128-output strip. WARP-SPECIALIZED
// 8 consumer warps run the MMA + epilogue with the
// exact layout and op order of the old 256-thread loop (bit-exact vs it,
// memcmp-verified at pp512 + dec-c32 shapes), and
// PD_BS_PW_GU producer warps own all staging - cp.async W/Y tiles AND the
// scale bytes, per-stage double-buffered. Synchronization is a pair of
// mbarriers per stage: full[s] completes when every producer thread's
// copies have landed (cp.async.mbarrier.arrive.noinc) AND it has
// release-arrived after its scale STS; empty[s] completes when all 256
// consumer threads arrive post-MMA. arrive != wait, so chunks flow with no
// full-CTA rendezvous - which is what makes the KC=256 read granularity
// (the +68% DRAM unlock, see the constants note) usable at 1 CTA/SM.
// Emits the swiglu output re-quantized to e4m3 + ue8m0 per-32 in REGISTERS
// (fq/fs, sorted-row indexed) - the down kernel's direct B input.
#if PD_BS_OK
// gate_up_bs epilogue, verbatim (bias + OAI-clamp swiglu, then e4m3 +
// ue8m0-per-32 requant in registers; the shfl amax mirrors the mmq
// epilogue's 32-row block; PAD rows emit exact zeros). Hoisted out of the
// kernel so the persistent work loop can DEFER it past the next item's
// chunk-0 MMA (see the consumer note there).
template <uint32_t BM, uint32_t BMR>
__device__ __forceinline__ void pd_bs_gu_epilogue(
    const float (&accg)[2][2][4], const float (&accu)[2][2][4],
    const unsigned int* __restrict__ tokv, const float* __restrict__ bgsv,
    const float* __restrict__ busv, unsigned char* __restrict__ fq,
    unsigned char* __restrict__ fs, uint32_t ff, float alpha, float limit,
    float up_add, uint32_t blk, uint32_t row_base, uint32_t i0, uint32_t joff,
    uint32_t g, uint32_t tq) {
    const uint32_t n_sb = ff >> 5;
    #pragma unroll
    for (uint32_t j = 0; j < 2u; ++j) {
        #pragma unroll
        for (uint32_t qc = 0; qc < 2u; ++qc) {
            const uint32_t c = joff + j * 8u + 2u * tq + qc;
            const bool pad = tokv[c] == PD_MOE_PAD;
            const uint32_t rb = row_base + i0;
            float sw[4];
            #pragma unroll
            for (uint32_t n = 0; n < 2u; ++n) {
                #pragma unroll
                for (uint32_t hq = 0; hq < 2u; ++hq) {
                    const uint32_t rloc = i0 + n * 16u + hq * 8u + g;
                    const uint32_t r = row_base + rloc;
                    float out = 0.f;
                    if (!pad && r < ff) {
                        const float gv = accg[n][j][qc + 2u * hq] + bgsv[rloc];
                        const float uv = accu[n][j][qc + 2u * hq] + busv[rloc];
                        const float xg = fminf(gv, limit);
                        const float yu = fminf(fmaxf(uv, -limit), limit);
                        // up_add: 1.0 = gpt-oss (u+1); 0.0 = qwen plain silu(g)*u
                        out = (xg / (1.0f + expf(-alpha * xg))) * (yu + up_add);
                    }
                    sw[n * 2u + hq] = out;
                }
            }
            float a = fmaxf(fmaxf(fabsf(sw[0]), fabsf(sw[1])), fmaxf(fabsf(sw[2]), fabsf(sw[3])));
            #pragma unroll
            for (uint32_t o = 4; o <= 16u; o <<= 1)
                a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
            int ee = 0;
            if (a > 0.0f) {
                int ex;
                float m = frexpf(a, &ex);
                ee = ex - 9 + (m > 0.875f ? 1 : 0);
            }
            const float inv = ldexpf(1.0f, -ee);
            const size_t row = (size_t)blk * BM + c;
            if (rb < ff) {
                #pragma unroll
                for (uint32_t v = 0; v < 4u; ++v) {
                    const uint32_t r = rb + (v >> 1) * 16u + (v & 1u) * 8u + g;
                    fq[row * ff + r] = __nv_fp8_e4m3(sw[(v >> 1) * 2u + (v & 1u)] * inv).__x;
                }
                if (g == 0) fs[row * n_sb + (rb >> 5)] = (unsigned char)(ee + 127);
            }
        }
    }
}
#endif

#if PD_BS_OK
// NF=1 (wide-consumer) epilogue: a warp owns 16 rows, so the 32-row fq/fs
// scale block spans the warp PAIR (w, w+CW). Phase A computes the swiglu
// values in registers and publishes each column's block-HALF amax to
// shared; a consumers-only barrier (id 3 - id 1 is the down fold, id 2 the
// gate_up producers); phase B folds both halves (max is order-free, so
// fq/fs stay BIT-IDENTICAL to the NF=2 epilogue) and quantizes. The
// trailing barrier keeps the next item's phase-A stores off this item's
// phase-B reads (shex is reused across the persistent loop).
template <uint32_t BM, uint32_t BMR, uint32_t NCW>
__device__ __forceinline__ void pd_bs_gu_epilogue_half(
    const float (&accg)[1][2][4], const float (&accu)[1][2][4],
    float* __restrict__ shex, const unsigned int* __restrict__ tokv,
    const float* __restrict__ bgsv, const float* __restrict__ busv,
    unsigned char* __restrict__ fq, unsigned char* __restrict__ fs, uint32_t ff,
    float alpha, float limit, float up_add, uint32_t blk, uint32_t row_base,
    uint32_t i0, uint32_t joff, uint32_t g, uint32_t tq) {
    const uint32_t n_sb = ff >> 5;
    const uint32_t bp = i0 >> 5;           // 32-row scale block within the tile
    const uint32_t half = (i0 >> 4) & 1u;  // this warp's 16-row half
    constexpr uint32_t NBP = BMR / 32u;
    float sw[2][2][2];  // [j][qc][hq]
    #pragma unroll
    for (uint32_t j = 0; j < 2u; ++j) {
        #pragma unroll
        for (uint32_t qc = 0; qc < 2u; ++qc) {
            const uint32_t c = joff + j * 8u + 2u * tq + qc;
            const bool pad = tokv[c] == PD_MOE_PAD;
            float a = 0.0f;
            #pragma unroll
            for (uint32_t hq = 0; hq < 2u; ++hq) {
                const uint32_t rloc = i0 + hq * 8u + g;
                const uint32_t r = row_base + rloc;
                float out = 0.f;
                if (!pad && r < ff) {
                    const float gv = accg[0][j][qc + 2u * hq] + bgsv[rloc];
                    const float uv = accu[0][j][qc + 2u * hq] + busv[rloc];
                    const float xg = fminf(gv, limit);
                    const float yu = fminf(fmaxf(uv, -limit), limit);
                    // up_add: 1.0 = gpt-oss (u+1); 0.0 = qwen plain silu(g)*u
                    out = (xg / (1.0f + expf(-alpha * xg))) * (yu + up_add);
                }
                sw[j][qc][hq] = out;
                a = fmaxf(a, fabsf(out));
            }
            #pragma unroll
            for (uint32_t o = 4; o <= 16u; o <<= 1)
                a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
            shex[(half * NBP + bp) * BM + c] = a;  // g-lanes dup-store, benign
        }
    }
    asm volatile("bar.sync 3, %0;" ::"r"(NCW * 32u) : "memory");
    #pragma unroll
    for (uint32_t j = 0; j < 2u; ++j) {
        #pragma unroll
        for (uint32_t qc = 0; qc < 2u; ++qc) {
            const uint32_t c = joff + j * 8u + 2u * tq + qc;
            const uint32_t rb = row_base + bp * 32u;
            if (rb < ff) {
                const float a =
                    fmaxf(shex[bp * BM + c], shex[(NBP + bp) * BM + c]);
                int ee = 0;
                if (a > 0.0f) {
                    int ex;
                    float m = frexpf(a, &ex);
                    ee = ex - 9 + (m > 0.875f ? 1 : 0);
                }
                const float inv = ldexpf(1.0f, -ee);
                const size_t row = (size_t)blk * BM + c;
                #pragma unroll
                for (uint32_t hq = 0; hq < 2u; ++hq) {
                    const uint32_t r = row_base + i0 + hq * 8u + g;
                    fq[row * ff + r] = __nv_fp8_e4m3(sw[j][qc][hq] * inv).__x;
                }
                if (g == 0 && half == 0) fs[row * n_sb + (rb >> 5)] = (unsigned char)(ee + 127);
            }
        }
    }
    asm volatile("bar.sync 3, %0;" ::"r"(NCW * 32u) : "memory");
}
#endif

template <uint32_t BM, uint32_t BMR, uint32_t CW, bool PERSIST = false, uint32_t NCW = 8u>
__global__ void __launch_bounds__(NCW * 32u + PD_BS_PW_GU * 32u,
                                  PD_BS_MINCTA(PD_BS_GU_SMEM))
pd_mxfp4_moe_gate_up_bs_kernel(
    const unsigned char* __restrict__ gate_data, const unsigned char* __restrict__ gate_scale,
    const float* __restrict__ gate_bias,
    const unsigned char* __restrict__ up_data, const unsigned char* __restrict__ up_scale,
    const float* __restrict__ up_bias,
    const unsigned int* __restrict__ sorted_row, const unsigned int* __restrict__ block_expert,
    const unsigned char* __restrict__ yq, const unsigned char* __restrict__ ys,
    unsigned char* __restrict__ fq, unsigned char* __restrict__ fs,
    uint32_t in_dim, uint32_t ff, float alpha, float limit, float up_add, uint32_t nb) {
#if PD_BS_OK
    // per-instantiation tile geometry: each of the 8 consumer warps always
    // owns a 32-row x 16-col fragment (acc shape and the per-32-row e4m3
    // requant are config-invariant); CW column groups x (8/CW) row groups
    // cover BMR x BM. <32,128,2> is the decode config (bit-identical to the
    // pre-template kernel); <64,64,4> is the prefill config - same weight
    // traffic per launch, half the block count at fat experts.
    // NCW = consumer warps. 8 = the decode config (RPW=32 rows/warp, NF=2
    // fragments - bit-identical to the historical kernel). 16 = the WIDE
    // prefill config (RPW=16, NF=1): double the MMA drain rate for the
    // producer streams at fat items (the pp512 roofline-corner residual,
    // s17b spec). At NF=1 a 32-row fq/fs scale block spans a WARP PAIR
    // (w, w+CW) - the epilogue exchanges block-half amax through shared.
    constexpr uint32_t RPW = BMR / (NCW / CW);
    constexpr uint32_t NF = RPW / 16u;
    static_assert(CW * (NCW / CW) == NCW && RPW * (NCW / CW) == BMR && 16u * CW == BM,
                  "warp grid must cover the tile");
    static_assert(NF == 1u || NF == 2u, "fragment rows per warp");
    static_assert((BMR & (BMR - 1u)) == 0u, "scale-prefetch mask needs pow2 BMR");
    constexpr uint32_t GU_STAGE = 2u * BMR * PD_BS_WROW_GU + BM * PD_BS_YROW_GU +
                                  2u * BMR * PD_BS_KB_GU + BM * PD_BS_KB_GU;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t n_kb = in_dim >> 5;
    const uint32_t nk = (in_dim + PD_BS_KC_GU - 1u) / PD_BS_KC_GU;
    // Persistent work loop (PERSIST): grid = n_SM CTAs grid-striding the
    // (block, y-tile) item list - the producer warps stream item i+1's
    // chunks while the consumers run item i's epilogue, so the per-CTA
    // epilogue is no longer a DRAM-silence window (DRAM duty 81.7% at
    // the dec shape vs the 88.3% pattern ceiling - the whole gap was the
    // 1-CTA/SM epilogue x 15.7 waves). Per-item metadata (tok, bias) lives
    // in ITEM-PARITY slots: producers may only be staging item i+1 while
    // consumers finish item i (nk > PD_BS_S), so slot i%2 is never
    // overwritten before its epilogue - no extra cross-group barrier.
    // Deterministic grid-stride assignment; outputs are disjoint per item,
    // so results stay BIT-EXACT vs the 2-D grid. !PERSIST keeps the old
    // one-item-per-CTA behavior (grid (nb, ny)).
    const uint32_t ny = (ff + BMR - 1u) / BMR;
    const uint32_t nit = nb * ny;
    const uint32_t it0 = PERSIST ? blockIdx.x : blockIdx.y * nb + blockIdx.x;
    const uint32_t itstep = PERSIST ? gridDim.x : nit;

    extern __shared__ unsigned char pd_bs_sh[];
    uint64_t* bfull = (uint64_t*)pd_bs_sh;   // [PD_BS_S]
    uint64_t* bempty = bfull + PD_BS_S;      // [PD_BS_S]
    float* bgs = (float*)(pd_bs_sh + 32u);   // [2][128] gate bias (item parity)
    float* bus = bgs + 2u * BMR;             // [2][128] up bias
    unsigned char* tiles = pd_bs_sh + 32u + 4u * BMR * 4u;  // 16-aligned stages
    __shared__ unsigned int tok[2][BM];
    // NF=1 epilogue exchange: per (32-row block, column) block-half amax
    __shared__ float sh_ep[2][BMR / 32u][BM];
    if (tid == 0) {
        #pragma unroll
        for (uint32_t s = 0; s < PD_BS_S; ++s) {
            pd_bs_bar_init(&bfull[s], 2u * 32u * PD_BS_PW_GU);
            pd_bs_bar_init(&bempty[s], NCW * 32u);
        }
    }
    __syncthreads();  // the only full-CTA barrier in the kernel

    if (warp >= NCW) {
        // ------------- producers: PD_BS_PW_GU warps own all staging -------------
        const uint32_t ptid = tid - NCW * 32u;
        const uint32_t pth = 32u * PD_BS_PW_GU;
        uint32_t eph[PD_BS_S] = {};
        uint32_t gkt = 0;  // GLOBAL chunk counter: the stage ring and its
                           // first-S no-wait guard roll across items
        uint32_t ipar = 0; // per-item parity counter (non-PAD items only)
        // scale bytes ride a register prefetch across the empty-wait: rows
        // are n_kb bytes (not 4-aligned) so cp.async cannot fetch them, and
        // a synchronous LDG->STS chain inside the critical section measured
        // 2x the whole kernel (gu4_proto first cut). LDG chunk kt before
        // waiting on kt's buffer (regs need no buffer); only the STS sits
        // between the wait and the arrive.
        #define PD_BS_SCG_N (2u * BMR * PD_BS_KB_GU + BM * PD_BS_KB_GU)
        #define PD_BS_SCG_V ((PD_BS_SCG_N + 32u * PD_BS_PW_GU - 1u) / (32u * PD_BS_PW_GU))
        // W-scale half: independent of smem tok, so it can prefetch the next
        // item's chunk 0 during this item's last chunk (rb/w0 parameterized)
        #define PD_BS_LDG_SCW(regs, kt, rb, w0)                                           \
            _Pragma("unroll") for (uint32_t v = 0; v < PD_BS_SCG_V; ++v) {                \
                const uint32_t u = ptid + v * pth;                                        \
                const uint32_t w = BMR * PD_BS_KB_GU;                                    \
                if (u < 2u * w) {                                                         \
                    const uint32_t row = (u & (w - 1u)) / PD_BS_KB_GU,                    \
                                   kb = u % PD_BS_KB_GU;                                  \
                    unsigned char b = 0u;                                                 \
                    if (((rb) + row) < ff && (kt) * PD_BS_KB_GU + kb < n_kb)              \
                        b = (u < w ? gate_scale : up_scale)                               \
                            [((w0) + row) * n_kb + (kt) * PD_BS_KB_GU + kb];              \
                    (regs)[v] = b;                                                        \
                }                                                                         \
            }
        // combined steady-state prefetch (one guarded loop per chunk; the
        // split halves below serve only the item-boundary transition)
        #define PD_BS_LDG_SC(regs, kt)                                                    \
            _Pragma("unroll") for (uint32_t v = 0; v < PD_BS_SCG_V; ++v) {                \
                const uint32_t u = ptid + v * pth;                                        \
                const uint32_t w = BMR * PD_BS_KB_GU;                                    \
                unsigned char b = 0u;                                                     \
                if (u < 2u * w) {                                                         \
                    const uint32_t row = (u & (w - 1u)) / PD_BS_KB_GU,                    \
                                   kb = u % PD_BS_KB_GU;                                  \
                    if ((row_base + row) < ff && (kt) * PD_BS_KB_GU + kb < n_kb)          \
                        b = (u < w ? gate_scale : up_scale)                               \
                            [(wrow0 + row) * n_kb + (kt) * PD_BS_KB_GU + kb];             \
                } else if (u < PD_BS_SCG_N) {                                             \
                    const uint32_t t = (u - 2u * w) / PD_BS_KB_GU, kb = u % PD_BS_KB_GU;  \
                    const uint32_t r = tok[p][t];                                         \
                    if (r != PD_MOE_PAD && (kt) * PD_BS_KB_GU + kb < n_kb)                \
                        b = ys[(size_t)r * n_kb + (kt) * PD_BS_KB_GU + kb];               \
                }                                                                         \
                (regs)[v] = b;                                                            \
            }
        // Y-scale half: reads tok[p] from smem (post-barrier only)
        #define PD_BS_LDG_SCY(regs, kt)                                                   \
            _Pragma("unroll") for (uint32_t v = 0; v < PD_BS_SCG_V; ++v) {                \
                const uint32_t u = ptid + v * pth;                                        \
                const uint32_t w = BMR * PD_BS_KB_GU;                                    \
                if (u >= 2u * w && u < PD_BS_SCG_N) {                                     \
                    const uint32_t t = (u - 2u * w) / PD_BS_KB_GU, kb = u % PD_BS_KB_GU;  \
                    const uint32_t r = tok[p][t];                                         \
                    unsigned char b = 0u;                                                 \
                    if (r != PD_MOE_PAD && (kt) * PD_BS_KB_GU + kb < n_kb)                \
                        b = ys[(size_t)r * n_kb + (kt) * PD_BS_KB_GU + kb];               \
                    (regs)[v] = b;                                                        \
                }                                                                         \
            }
        for (uint32_t it = it0; it < nit; it += itstep) {
            const uint32_t blk = it % nb;
            const uint32_t e = block_expert[blk];
            if (e == PD_MOE_PAD) continue;
            const uint32_t row_base = (it / nb) * BMR;
            const size_t wrow0 = (size_t)e * ff + row_base;
            const uint32_t p = ipar & 1u;
            ++ipar;
            // per-item metadata into parity slot p (bias is consumer-only;
            // tok feeds this side's Y staging and ys-scale LDGs too)
            for (uint32_t u = ptid; u < BM; u += pth)
                tok[p][u] = sorted_row[(size_t)blk * BM + u];
            for (uint32_t u = ptid; u < BMR; u += pth) {
                const bool ok = row_base + u < ff;
                bgs[p * BMR + u] = ok ? gate_bias[(size_t)e * ff + row_base + u] : 0.0f;
                bus[p * BMR + u] = ok ? up_bias[(size_t)e * ff + row_base + u] : 0.0f;
            }
            // producers-only barrier (id 2; the down fold owns id 1): the
            // staging below reads other producer threads' tok stores
            asm volatile("bar.sync 2, %0;" ::"r"(32u * PD_BS_PW_GU) : "memory");
            unsigned char screg[PD_BS_SCG_V];
            PD_BS_LDG_SC(screg, 0u)
            for (uint32_t kt = 0; kt < nk; ++kt) {
                const uint32_t s = gkt % PD_BS_S;
                unsigned char* wgs = tiles + s * GU_STAGE;
                unsigned char* wus = wgs + BMR * PD_BS_WROW_GU;
                unsigned char* ybs = wus + BMR * PD_BS_WROW_GU;
                unsigned char* wsg = ybs + BM * PD_BS_YROW_GU;
                unsigned char* wsu = wsg + BMR * PD_BS_KB_GU;
                unsigned char* ysc = wsu + BMR * PD_BS_KB_GU;
                if (gkt >= PD_BS_S) { pd_bs_bar_wait(&bempty[s], eph[s]); eph[s] ^= 1u; }
                ++gkt;
                // g||u ILV layout: block bk of a row lives in 128 B pair
                // bk/4 at +0 (gate) / +64 (up); a KC=256 chunk touches two
                // ADJACENT pairs = 256 B contiguous per row. Everything
                // reads via gate_data; up_data is unused.
                for (uint32_t u = ptid; u < BMR * PD_BS_WSEG_GU; u += pth) {
                    const uint32_t row = u / PD_BS_WSEG_GU, seg = u % PD_BS_WSEG_GU;
                    const uint32_t bk = kt * PD_BS_KB_GU + seg;
                    const bool ok = (row_base + row) < ff && bk < n_kb;
                    const size_t rb = (wrow0 + row) * (size_t)(((n_kb + 3u) >> 2) * 128u) +
                                      (size_t)(bk >> 2) * 128u + (bk & 3u) * 16u;
                    pd_cp_async16((int*)(wgs + row * PD_BS_WROW_GU +
                                         (seg ^ PD_BS_SWZ(row, PD_BS_WSEG_GU)) * 16u),
                                  gate_data + rb, ok);
                    pd_cp_async16((int*)(wus + row * PD_BS_WROW_GU +
                                         (seg ^ PD_BS_SWZ(row, PD_BS_WSEG_GU)) * 16u),
                                  gate_data + rb + 64u, ok);
                }
                for (uint32_t u = ptid; u < BM * PD_BS_YSEG_GU; u += pth) {
                    const uint32_t t = u / PD_BS_YSEG_GU, seg = u % PD_BS_YSEG_GU;
                    const uint32_t r = tok[p][t];
                    const bool ok = r != PD_MOE_PAD && (kt * PD_BS_YSEG_GU + seg) * 16u < in_dim;
                    pd_cp_async16((int*)(ybs + t * PD_BS_YROW_GU + seg * 16u),
                                  yq + (size_t)(ok ? r : 0u) * in_dim + kt * PD_BS_KC_GU +
                                      seg * 16u, ok);
                }
                #pragma unroll
                for (uint32_t v = 0; v < PD_BS_SCG_V; ++v) {
                    const uint32_t u = ptid + v * pth;
                    const uint32_t w = BMR * PD_BS_KB_GU;
                    if (u < w) wsg[u] = screg[v];
                    else if (u < 2u * w) wsu[u - w] = screg[v];
                    else if (u < PD_BS_SCG_N) ysc[u - 2u * w] = screg[v];
                }
                pd_bs_cp_arrive_noinc(&bfull[s]);  // fires when this thread's copies land
                pd_bs_bar_arrive(&bfull[s]);       // release for the STS scale bytes
                if (kt + 1u < nk) PD_BS_LDG_SC(screg, kt + 1u)
            }
        }
        #undef PD_BS_LDG_SC
        #undef PD_BS_LDG_SCW
        #undef PD_BS_LDG_SCY
        #undef PD_BS_SCG_V
        #undef PD_BS_SCG_N
        return;
    }

    // ------------- consumers: the original MMA + epilogue, verbatim -------------
    const uint32_t g = lane >> 2, tq = lane & 3u;
    const uint32_t i0 = (warp / CW) * RPW;
    const uint32_t joff = (warp % CW) * 16u;
    uint32_t fph[PD_BS_S] = {};
    uint32_t gct = 0, ipar = 0;
    for (uint32_t it = it0; it < nit; it += itstep) {
    const uint32_t blk = it % nb;
    if (block_expert[blk] == PD_MOE_PAD) continue;
    const uint32_t row_base = (it / nb) * BMR;
    const uint32_t p = ipar & 1u;
    ++ipar;
    float accg[NF][2][4] = {}, accu[NF][2][4] = {};
    for (uint32_t kt = 0; kt < nk; ++kt) {
        const uint32_t s = gct % PD_BS_S;
        ++gct;
        unsigned char* wg = tiles + s * GU_STAGE;
        unsigned char* wu = wg + BMR * PD_BS_WROW_GU;
        unsigned char* yb = wu + BMR * PD_BS_WROW_GU;
        unsigned char* wsg = yb + BM * PD_BS_YROW_GU;
        unsigned char* wsu = wsg + BMR * PD_BS_KB_GU;
        unsigned char* ysc = wsu + BMR * PD_BS_KB_GU;
        pd_bs_bar_wait(&bfull[s], fph[s]); fph[s] ^= 1u;
        // A fragments via ldmatrix.x4, two kb blocks per issue (the dense
        // bs kernel's mapping): lane supplies row i0 + n*16 +
        // ((lane>>3)&1)*8 + (lane&7) at kb half lane>>4, swizzle applied
        // per-lane; the returned regs are {p0 kbA, p8 kbA, p0 kbB, p8 kbB}
        // in exactly afrag_split's per-lane bytes. Nibble spread applies
        // in registers, same masks as pd_bs_afrag_split.
        static_assert(PD_BS_KB_GU % 2u == 0u, "kb pairs");
        #pragma unroll
        for (uint32_t t2 = 0; t2 < PD_BS_KB_GU / 2u; ++t2) {
            uint32_t ga2[NF][2][4], ua2[NF][2][4];  // [n][kb01][frag]
            #pragma unroll
            for (uint32_t n = 0; n < NF; ++n) {
                const uint32_t rl = i0 + n * 16u + ((lane >> 3) & 1u) * 8u + (lane & 7u);
                const uint32_t kbl = 2u * t2 + (lane >> 4);
                const uint32_t soff = (kbl ^ PD_BS_SWZ(rl, PD_BS_WSEG_GU)) * 16u;
                uint32_t raw[4];
                pd_ldm_x4(raw, wg + rl * PD_BS_WROW_GU + soff);
                #pragma unroll
                for (uint32_t h = 0; h < 2u; ++h) {
                    ga2[n][h][0] = (raw[h * 2u] & 0x0F0F0F0Fu) << 2;
                    ga2[n][h][1] = (raw[h * 2u + 1u] & 0x0F0F0F0Fu) << 2;
                    ga2[n][h][2] = (raw[h * 2u] & 0xF0F0F0F0u) >> 2;
                    ga2[n][h][3] = (raw[h * 2u + 1u] & 0xF0F0F0F0u) >> 2;
                }
                pd_ldm_x4(raw, wu + rl * PD_BS_WROW_GU + soff);
                #pragma unroll
                for (uint32_t h = 0; h < 2u; ++h) {
                    ua2[n][h][0] = (raw[h * 2u] & 0x0F0F0F0Fu) << 2;
                    ua2[n][h][1] = (raw[h * 2u + 1u] & 0x0F0F0F0Fu) << 2;
                    ua2[n][h][2] = (raw[h * 2u] & 0xF0F0F0F0u) >> 2;
                    ua2[n][h][3] = (raw[h * 2u + 1u] & 0xF0F0F0F0u) >> 2;
                }
            }
            #pragma unroll
            for (uint32_t kb01 = 0; kb01 < 2u; ++kb01) {
                const uint32_t kb = 2u * t2 + kb01;
                uint32_t b0[2], b1[2], sfb[2];
                #pragma unroll
                for (uint32_t j = 0; j < 2u; ++j) {
                    uint32_t t = joff + j * 8u + g;
                    const unsigned char* yr = yb + t * PD_BS_YROW_GU + kb * 32u;
                    b0[j] = *(const uint32_t*)(yr + 4u * tq);
                    b1[j] = *(const uint32_t*)(yr + 16u + 4u * tq);
                    sfb[j] = ysc[t * PD_BS_KB_GU + kb];
                }
                #pragma unroll
                for (uint32_t n = 0; n < NF; ++n) {
                    uint32_t r0 = i0 + n * 16u + g;
                    uint32_t rs = (tq & 1u) ? r0 + 8u : r0;
                    uint32_t sfag = wsg[rs * PD_BS_KB_GU + kb];
                    uint32_t sfau = wsu[rs * PD_BS_KB_GU + kb];
                    #pragma unroll
                    for (uint32_t j = 0; j < 2u; ++j) {
                        pd_bs_mma(accg[n][j], ga2[n][kb01][0], ga2[n][kb01][1],
                                  ga2[n][kb01][2], ga2[n][kb01][3], b0[j], b1[j], sfag,
                                  sfb[j]);
                        pd_bs_mma(accu[n][j], ua2[n][kb01][0], ua2[n][kb01][1],
                                  ua2[n][kb01][2], ua2[n][kb01][3], b0[j], b1[j], sfau,
                                  sfb[j]);
                    }
                }
            }
        }
        pd_bs_bar_arrive(&bempty[s]);
    }
    if constexpr (NF == 2u) {
        pd_bs_gu_epilogue<BM, BMR>(accg, accu, &tok[p][0], bgs + p * BMR,
                                   bus + p * BMR, fq, fs, ff, alpha, limit, up_add,
                                   blk, row_base, i0, joff, g, tq);
    } else {
        pd_bs_gu_epilogue_half<BM, BMR, NCW>(accg, accu, &sh_ep[0][0][0], &tok[p][0],
                                             bgs + p * BMR, bus + p * BMR, fq, fs,
                                             ff, alpha, limit, up_add, blk, row_base, i0,
                                             joff, g, tq);
    }
    }  // item loop
#else
    (void)gate_data; (void)gate_scale; (void)gate_bias; (void)up_data; (void)up_scale;
    (void)up_bias; (void)sorted_row; (void)block_expert; (void)yq; (void)ys;
    (void)fq; (void)fs; (void)in_dim; (void)ff; (void)alpha; (void)limit; (void)nb;
#endif
}

