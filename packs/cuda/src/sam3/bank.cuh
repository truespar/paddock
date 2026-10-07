// sam3/bank.cuh - SAM 3's memory bank landing: one object's chosen past memories and object pointers made the memory attention's 64-wide key input and value
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
// ------------------------------------------------------------ sam3 memory bank
// Before every tracked frame Meta's _prepare_memory_conditioned_features
// concatenates, for each object, the memory frames it picked (5184 x 64
// each, stored bf16) and then four 64-wide tokens per past object pointer
// (the 256 split in four). Each row gets a position: the memory encoder's
// sine table plus one of seven learned temporal rows for a memory frame,
// and Linear(256 -> 64) over a 1-D sine of the distance / 15 for a pointer
// (the four tokens of a pointer share it). The attention reads
// k_proj(M + pos) and the value M, so this lands f16(M + pos) and f16(M)
// straight into the two planes it reads. The bank is rebuilt every frame:
// a frame's temporal row moves as it ages.
//
// Needs f32_qkv.cuh's pd_launch_status.

// 790: one memory frame's rows. mem bf16 [rows][64] (the stored memory),
// pos f32 [rows][64] (the sine table), tpos f32 [64] (the frame's temporal
// row) -> kin f16 = m + (pos + tpos) - Meta's sum order: the position is
// assembled first, the key input adds it - and v f16 = m (a bf16 is exact
// in f16 above f16's subnormals). Eight channels a thread: one 16-byte
// load of bf16, two of the table, one 16-byte store each way.
__global__ void pd_sam3_bank_mem_rows_kernel(const __nv_bfloat16* __restrict__ mem,
                                             const float* __restrict__ pos,
                                             const float* __restrict__ tpos,
                                             __half* __restrict__ kin, __half* __restrict__ v,
                                             uint32_t n8) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n8) return;
    const size_t e = (size_t)i * 8u;
    const uint32_t c0 = (i & 7u) * 8u;
    const uint4 raw = *reinterpret_cast<const uint4*>(mem + e);
    const __nv_bfloat162* m2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
    const float4 pa = *reinterpret_cast<const float4*>(pos + e);
    const float4 pb = *reinterpret_cast<const float4*>(pos + e + 4);
    const float4 ta = *reinterpret_cast<const float4*>(tpos + c0);
    const float4 tb = *reinterpret_cast<const float4*>(tpos + c0 + 4);
    const float p[8] = {pa.x + ta.x, pa.y + ta.y, pa.z + ta.z, pa.w + ta.w,
                        pb.x + tb.x, pb.y + tb.y, pb.z + tb.z, pb.w + tb.w};
    uint4 ko, vo;
    __half2* k2 = reinterpret_cast<__half2*>(&ko);
    __half2* v2 = reinterpret_cast<__half2*>(&vo);
#pragma unroll
    for (uint32_t j = 0; j < 4u; ++j) {
        const float2 m = __bfloat1622float2(m2[j]);
        k2[j] = __floats2half2_rn(m.x + p[2u * j], m.y + p[2u * j + 1u]);
        v2[j] = __floats2half2_rn(m.x, m.y);
    }
    *reinterpret_cast<uint4*>(kin + e) = ko;
    *reinterpret_cast<uint4*>(v + e) = vo;
}

PD_EXPORT
int pd_sam3_bank_mem_rows(const void* mem, const void* pos, const void* tpos, void* kin, void* v,
                          uint32_t rows, void* stream) {
    if (rows == 0u) return 0;
    if (rows > 0x1fffffffu / 8u) return cudaErrorInvalidValue;
    const uint32_t n8 = rows * 8u;
    pd_sam3_bank_mem_rows_kernel<<<(n8 + 255u) / 256u, 256, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)mem, (const float*)pos, (const float*)tpos, (__half*)kin,
        (__half*)v, n8);
    return pd_launch_status();
}

// 791: the object pointers' rows, one block of 256 a pointer. pool f32
// [*][256] holds the bank's pointers; meta u32 [2][np] is each one's pool
// row then its distance (frames back for a conditioning frame's pointer,
// the rank in the memory-selected list for the others). The position is
// Meta's _get_tpos_enc: get_1d_sine_pe(d / tmax, 256) - sin over the first
// 128, cos over the next, both at 10000^(2 floor(k / 2) / 128) - then
// Linear(256 -> 64) (w [64][256], b [64]). Token s of pointer p is row
// 4p + s: v = f16(ptr[64 s + c]), kin = f16(ptr[64 s + c] + pos[c]).
__global__ void __launch_bounds__(256) pd_sam3_bank_ptr_rows_kernel(
    const float* __restrict__ pool, const uint32_t* __restrict__ meta, const float* __restrict__ w,
    const float* __restrict__ b, __half* __restrict__ kin, __half* __restrict__ v, uint32_t np,
    float tmax) {
    __shared__ float sine[256];
    __shared__ float pe[64];
    const uint32_t p = blockIdx.x, j = threadIdx.x;
    const float x = __fdiv_rn((float)meta[np + p], tmax);
    const uint32_t k = j & 127u;
    const float dim_t = powf(10000.0f, (float)(2u * (k / 2u)) / 128.0f);
    const float a = __fdiv_rn(x, dim_t);
    sine[j] = j < 128u ? sinf(a) : cosf(a);
    __syncthreads();
    if (j < 64u) {
        const float* wr = w + (size_t)j * 256u;
        float acc = 0.0f;
        for (uint32_t t = 0; t < 256u; ++t) acc = fmaf(wr[t], sine[t], acc);
        pe[j] = acc + b[j];
    }
    __syncthreads();
    const float val = pool[(size_t)meta[p] * 256u + j];
    const size_t o = (size_t)p * 256u + j;
    v[o] = __float2half_rn(val);
    kin[o] = __float2half_rn(val + pe[j & 63u]);
}

PD_EXPORT
int pd_sam3_bank_ptr_rows(const void* pool, const void* meta, const void* w, const void* b,
                          void* kin, void* v, uint32_t np, float tmax, void* stream) {
    if (np == 0u) return 0;
    if (np > 0xffffu || !(tmax > 0.0f)) return cudaErrorInvalidValue;
    pd_sam3_bank_ptr_rows_kernel<<<np, 256, 0, (cudaStream_t)stream>>>(
        (const float*)pool, (const uint32_t*)meta, (const float*)w, (const float*)b, (__half*)kin,
        (__half*)v, np, tmax);
    return pd_launch_status();
}
