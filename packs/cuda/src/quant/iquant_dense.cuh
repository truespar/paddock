// Contributed by NodeNestor (github.com/Nodenester) in truespar/paddock PR #17.
// i-quant DENSE lanes: the ggml IQ1/IQ2/IQ3 family + IQ4_NL on the repacked
// k-quant streams for the projections that are not MoE expert seats - the
// attention / GDN / FFN planes, the head and the token embedding of a file
// whose dense tensors are i-quant (the small UD-IQ1/IQ2 exports).
//
// Every entry point walks 16-weight windows through `pd_kq_win_unpack` -
// the per-format unpack the token-batched MoE pair already runs - and folds
// each window with dp4a against int8 activations, or with f32 products
// against f32 activations. The k-quant exports dispatch here by type
// (`pd_kq_valid_iq`), so the k-quant kernels themselves are untouched; the
// forward declarations at the top of quant/kquant.cuh are what let an export
// defined there call a launcher defined here (pack.cu includes this header
// after quant/kquant_w4a8.cuh).
//
// The batch-1 decode lanes (the serving class) and the exact-f32 oracle are
// the staged kernels below: a warp per row, four windows per lane per step,
// the activations and the format's codebook in shared, the quant type a
// template parameter. Measured on the RTX 5060 Ti (driver 581.80), DRAM-cold
// 4096x12288 planes: IQ2_XXS 361 GB/s, Q2_K 420 GB/s - 80% / 94% of the
// card's 448 GB/s, against a naive warp-per-row walk's 184 / 300. The 2-bit
// codebook formats are issue-bound past that: per weight they cost the
// codebook gather, the sign spread and the dp4a, and both a shared sign-mask
// table and a nibble-coded codebook measured slower (more shared traffic,
// more permutes). The mu term (Q4_K / Q5_K / Q4_0 / Q2_K's per-16 activation
// sums) is honoured when the caller passes `xsums`, so a mixed gate|up pair
// (one plane i-quant, one k-quant) is served exactly by the naive lanes.

// row bases for both streams (IQ4_NL rows lie flat; see pd_kq_row_datab)
__device__ __forceinline__ void pd_iqd_row(const uint8_t* __restrict__ data,
                                           const uint8_t* __restrict__ scales,
                                           uint32_t o, uint32_t in_dim, uint32_t dtype,
                                           const uint8_t** row, const uint8_t** rec) {
    *row = data + (size_t)o * pd_kq_row_datab(dtype, in_dim);
    *rec = scales + (size_t)o * pd_kq_row_scb(dtype, in_dim);
}

// int8 activations (quantize_q8 layout: xq [rows][in_dim], xs [rows][in_dim/32],
// xsums [rows][in_dim/16] or null). One warp's partial dot over its windows.
__device__ __forceinline__ float pd_iqd_dot_q8(const uint8_t* __restrict__ row,
                                               const uint8_t* __restrict__ rec,
                                               uint32_t dtype, uint32_t in_dim,
                                               const int8_t* __restrict__ xq,
                                               const float* __restrict__ xs,
                                               const float* __restrict__ xsm,
                                               uint32_t lane) {
    const uint32_t db = pd_kq_datab(dtype), scb = pd_kq_scb(dtype);
    float acc = 0.0f;
    for (uint32_t base = lane * 16u; base < in_dim; base += 32u * 16u) {
        const uint32_t s = base >> 8, w = (base >> 4) & 15u;
        int wq[4];
        float f, g;
        pd_kq_win_unpack(dtype, row + (size_t)s * db, rec + (size_t)s * scb, w, wq, &f, &g);
        const int4 xv = *reinterpret_cast<const int4*>(xq + base);
        int si = __dp4a(wq[0], xv.x, 0);
        si = __dp4a(wq[1], xv.y, si);
        si = __dp4a(wq[2], xv.z, si);
        si = __dp4a(wq[3], xv.w, si);
        const float x_s = xs[base >> 5];
        acc += f * (x_s * (float)si);
        if (xsm != nullptr && g != 0.0f) acc += g * (x_s * xsm[base >> 4]);
    }
    return acc;
}

// f32 activations: the same windows, products in f32 (the exact class the
// batch-1 k-quant GEMV serves; no activation quantization).
__device__ __forceinline__ float pd_iqd_dot_f32(const uint8_t* __restrict__ row,
                                                const uint8_t* __restrict__ rec,
                                                uint32_t dtype, uint32_t in_dim,
                                                const float* __restrict__ x,
                                                uint32_t lane) {
    const uint32_t db = pd_kq_datab(dtype), scb = pd_kq_scb(dtype);
    float acc = 0.0f;
    for (uint32_t base = lane * 16u; base < in_dim; base += 32u * 16u) {
        const uint32_t s = base >> 8, w = (base >> 4) & 15u;
        int wq[4];
        float f, g;
        pd_kq_win_unpack(dtype, row + (size_t)s * db, rec + (size_t)s * scb, w, wq, &f, &g);
        float d = 0.0f, xsum = 0.0f;
        #pragma unroll
        for (uint32_t k = 0; k < 4u; ++k) {
            const int word = wq[k];
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j) {
                const float xv = x[base + 4u * k + j];
                d += (float)((int8_t)((word >> (8u * j)) & 0xffu)) * xv;
                xsum += xv;
            }
        }
        acc += f * d + g * xsum;
    }
    return acc;
}

__device__ __forceinline__ float pd_iqd_warp_sum(float v) {
    #pragma unroll
    for (uint32_t s = 16; s > 0; s >>= 1) v += __shfl_down_sync(0xffffffffu, v, s);
    return v;
}

// y[o] = row_o . x, warp per row over 16-weight windows: the lane for rows
// whose width is a multiple of 32 but not of 64 (IQ4_NL only).
__global__ void __launch_bounds__(256) pd_iqd_gemv_f32_win_kernel(
        const uint8_t* __restrict__ data, const uint8_t* __restrict__ scales,
        const float* __restrict__ x, float* __restrict__ y,
        uint32_t in_dim, uint32_t out_dim, uint32_t dtype) {
    const uint32_t o = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    if (o >= out_dim) return;
    const uint8_t *row, *rec;
    pd_iqd_row(data, scales, o, in_dim, dtype, &row, &rec);
    const float v = pd_iqd_warp_sum(pd_iqd_dot_f32(row, rec, dtype, in_dim, x, lane));
    if (lane == 0) y[o] = v;
}

// y[b][o] = row_o . xq[b]; grid = warps over (o, b) with b fastest so the
// warps sharing a weight row run together (L2 reuse across the batch).
__global__ void __launch_bounds__(256) pd_iqd_dp4a_kernel(
        const uint8_t* __restrict__ data, const uint8_t* __restrict__ scales,
        const int8_t* __restrict__ xq, const float* __restrict__ xs,
        const float* __restrict__ xsums, float* __restrict__ y,
        uint32_t in_dim, uint32_t out_dim, uint32_t batch, uint32_t dtype) {
    // 8 warps per block, block-major: blockIdx.x * 8 stays in 32 bits for
    // every (out_dim, batch) the launcher admits, where the thread-linear
    // form (blockIdx.x * 256 + tid) overflowed past 134M outputs - the
    // vocab-wide head at batch 1024 wrote its first 88% of rows only.
    const uint32_t task = blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (task >= out_dim * batch) return;
    const uint32_t o = task / batch, b = task - o * batch;
    const uint8_t *row, *rec;
    pd_iqd_row(data, scales, o, in_dim, dtype, &row, &rec);
    const float acc = pd_iqd_dot_q8(row, rec, dtype, in_dim, xq + (size_t)b * in_dim,
                                    xs + (size_t)b * (in_dim >> 5),
                                    xsums ? xsums + (size_t)b * (in_dim >> 4) : nullptr, lane);
    const float v = pd_iqd_warp_sum(acc);
    if (lane == 0) y[(size_t)b * out_dim + o] = v;
}

// y[o] = silu(gate_o . x) * (up_o . x) - the fused gate|up decode pair.
__global__ void __launch_bounds__(256) pd_iqd_glu_kernel(
        const uint8_t* __restrict__ gd, const uint8_t* __restrict__ gs,
        const uint8_t* __restrict__ ud, const uint8_t* __restrict__ us,
        const int8_t* __restrict__ xq, const float* __restrict__ xs,
        const float* __restrict__ xsums, float* __restrict__ y,
        uint32_t in_dim, uint32_t out_dim, uint32_t dtg, uint32_t dtu) {
    const uint32_t o = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    if (o >= out_dim) return;
    const uint8_t *grow, *grec, *urow, *urec;
    pd_iqd_row(gd, gs, o, in_dim, dtg, &grow, &grec);
    pd_iqd_row(ud, us, o, in_dim, dtu, &urow, &urec);
    const float g = pd_iqd_warp_sum(pd_iqd_dot_q8(grow, grec, dtg, in_dim, xq, xs, xsums, lane));
    const float u = pd_iqd_warp_sum(pd_iqd_dot_q8(urow, urec, dtu, in_dim, xq, xs, xsums, lane));
    if (lane == 0) y[o] = (g / (1.0f + expf(-g))) * u;
}

// The staged lanes: NT threads = NT/32 warps per block, a WARP PER ROW,
// each lane one 64-weight chunk (four 16-weight windows) per step - so a
// lane's weight bytes are contiguous (one superblock region) and a warp's
// step covers 2048 weights. The activations and the format's codebook are
// staged in shared ONCE per block, and the block then walks rows grid-stride
// (the grid is capped at the resident-block count); a warp per row needs no
// cross-warp fold, so rows finish with a shuffle and one store.
//
// The staged activations are LANE-MAJOR: window u of lane l's chunk in
// step `it` sits at slot (it*4 + u)*32 + l, so the 32 lanes' reads of a
// given window index are 32 consecutive 16-byte units - conflict-free. In
// row order a lane's chunk is 64 weights = 64 B (int8) / 256 B (f32) from
// its neighbour's, which put every lane of a quarter-warp on the same
// banks: a 4- to 16-way conflict on every activation read, and the reason
// both lanes sat at a third of the k-quant GEMV's byte rate. The per-32
// scales and per-16 sums are staged per window in the same order.
//
// The quant type is a template parameter: the per-window format switch
// folds at compile time (a runtime switch across four unrolled windows cost
// 80-128 registers per thread and halved the occupancy).

// window w -> lane-major slot
__device__ __forceinline__ uint32_t pd_iqd_slot(uint32_t w) {
    const uint32_t c = w >> 2, u = w & 3u;
    return ((c >> 5) * 4u + u) * 32u + (c & 31u);
}
// windows a staging buffer holds for in_dim (whole 32-chunk steps)
__host__ __device__ __forceinline__ uint32_t pd_iqd_win_pad(uint32_t in_dim) {
    return ((in_dim >> 6) + 31u) / 32u * 128u;
}
// shared bytes of the int8 staging: 16 B per window + per-window scale + sum
__host__ __device__ __forceinline__ uint32_t pd_iqd_q8_stage_bytes(uint32_t in_dim) {
    return pd_iqd_win_pad(in_dim) * (16u + 4u + 4u);
}

// one lane's partial over its 64-weight chunks of a row, int8 activations
template <uint32_t DT>
__device__ __forceinline__ float pd_iqd_q8_row_acc(
        const uint8_t* __restrict__ row, const uint8_t* __restrict__ rec,
        const int4* __restrict__ sxq, const float* __restrict__ sxs,
        const float* __restrict__ ssm, uint32_t lane, uint32_t n_chunk,
        bool mu, const PdIqTabs& tabs) {
    constexpr uint32_t dtype = DT;
    const uint32_t db = pd_kq_datab(dtype), scb = pd_kq_scb(dtype);
    float acc = 0.0f;
    for (uint32_t it = 0; it * 32u < n_chunk; ++it) {
        const uint32_t c = it * 32u + lane;
        if (c >= n_chunk) break;
        const uint32_t s = c >> 2, w0 = (c & 3u) * 4u;
        const uint8_t* sb = row + (size_t)s * db;
        const uint8_t* rc = rec + (size_t)s * scb;
        int wq[4][4];
        float f[4], g[4];
        #pragma unroll
        for (uint32_t u = 0; u < 4u; ++u)
            pd_kq_win_unpack_t(dtype, sb, rc, w0 + u, wq[u], &f[u], &g[u], tabs);
        #pragma unroll
        for (uint32_t u = 0; u < 4u; ++u) {
            const uint32_t slot = (it * 4u + u) * 32u + lane;
            const int4 xv = sxq[slot];
            int si = __dp4a(wq[u][0], xv.x, 0);
            si = __dp4a(wq[u][1], xv.y, si);
            si = __dp4a(wq[u][2], xv.z, si);
            si = __dp4a(wq[u][3], xv.w, si);
            const float x_s = sxs[slot];
            acc += f[u] * (x_s * (float)si);
            if (mu) acc += g[u] * (x_s * ssm[slot]);
        }
    }
    return acc;
}

// one lane's partial over its chunks of a staged f32 tile (lane-major
// float4 slots: window slot * 4 + k, spread so the k-th float4 of the 32
// lanes' windows are consecutive)
template <uint32_t DT>
__device__ __forceinline__ float pd_iqd_f32_row_acc(
        const uint8_t* __restrict__ row, const uint8_t* __restrict__ rec,
        const float4* __restrict__ xsh, const float* __restrict__ xsum16,
        uint32_t c0, uint32_t n_chunk, uint32_t lane, const PdIqTabs& tabs) {
    constexpr uint32_t dtype = DT;
    const uint32_t db = pd_kq_datab(dtype), scb = pd_kq_scb(dtype);
    float acc = 0.0f;
    for (uint32_t it = 0; it * 32u < n_chunk; ++it) {
        const uint32_t c = it * 32u + lane;
        if (c >= n_chunk) break;
        const uint32_t gc = c0 + c;
        const uint32_t s = gc >> 2, w0 = (gc & 3u) * 4u;
        const uint8_t* sb = row + (size_t)s * db;
        const uint8_t* rc = rec + (size_t)s * scb;
        int wq[4][4];
        float f[4], g[4];
        #pragma unroll
        for (uint32_t u = 0; u < 4u; ++u)
            pd_kq_win_unpack_t(dtype, sb, rc, w0 + u, wq[u], &f[u], &g[u], tabs);
        #pragma unroll
        for (uint32_t u = 0; u < 4u; ++u) {
            const uint32_t slot = (it * 4u + u) * 32u + lane;
            float d = 0.0f;
            #pragma unroll
            for (uint32_t k = 0; k < 4u; ++k) {
                const float4 x4 = xsh[((it * 4u + u) * 4u + k) * 32u + lane];
                const float xv[4] = {x4.x, x4.y, x4.z, x4.w};
                #pragma unroll
                for (uint32_t jj = 0; jj < 4u; ++jj)
                    d += (float)((int8_t)((wq[u][k] >> (8u * jj)) & 0xffu)) * xv[jj];
            }
            acc += f[u] * d + g[u] * xsum16[slot];
        }
    }
    return acc;
}

__device__ __forceinline__ float pd_iqd_warp_fold(float v) {
    for (uint32_t sd = 16; sd > 0; sd >>= 1) v += __shfl_down_sync(0xffffffffu, v, sd);
    return v;
}

// stage the int8 activations, per-window scales and sums lane-major after
// the tables
__device__ __forceinline__ void pd_iqd_stage_q8(
        uint8_t* __restrict__ smem, uint32_t tabs_bytes,
        const int8_t* __restrict__ xq, const float* __restrict__ xs,
        const float* __restrict__ xsums, uint32_t in_dim, uint32_t nt,
        int4** sxq, float** sxs, float** ssm) {
    const uint32_t pad = pd_iqd_win_pad(in_dim), n_win = in_dim >> 4;
    *sxq = reinterpret_cast<int4*>(smem + tabs_bytes);
    *sxs = reinterpret_cast<float*>(*sxq + pad);
    *ssm = *sxs + pad;
    const bool mu = xsums != nullptr;
    for (uint32_t w = threadIdx.x; w < n_win; w += nt) {
        const uint32_t slot = pd_iqd_slot(w);
        (*sxq)[slot] = reinterpret_cast<const int4*>(xq)[w];
        (*sxs)[slot] = xs[w >> 1];
        if (mu) (*ssm)[slot] = xsums[w];
    }
}

// y[o] = row_o . xq, int8 activations (the decode lane)
template <uint32_t DT, uint32_t NT>
__global__ void __launch_bounds__(NT) pd_iqd_gemv_q8_kernel(
        const uint8_t* __restrict__ data, const uint8_t* __restrict__ scales,
        const int8_t* __restrict__ xq, const float* __restrict__ xs,
        const float* __restrict__ xsums, float* __restrict__ y,
        uint32_t in_dim, uint32_t out_dim) {
    PD_PDL_ARM();
    constexpr uint32_t dtype = DT, WARPS = NT / 32u;
    extern __shared__ uint8_t pd_iqd_smem[];
    const PdIqTabs tabs = pd_iq_tabs_stage(dtype, pd_iqd_smem);
    int4* sxq; float* sxs; float* ssm;
    pd_iqd_stage_q8(pd_iqd_smem, pd_iq_tabs_bytes(dtype), xq, xs, xsums, in_dim, NT, &sxq, &sxs, &ssm);
    __syncthreads();
    const bool mu = xsums != nullptr;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u, n_chunk = in_dim >> 6;
    const size_t rdb = pd_kq_row_datab(dtype, in_dim), rsb = pd_kq_row_scb(dtype, in_dim);
    for (uint32_t o = blockIdx.x * WARPS + warp; o < out_dim; o += gridDim.x * WARPS) {
        const float acc = pd_iqd_q8_row_acc<DT>(data + (size_t)o * rdb, scales + (size_t)o * rsb,
                                                sxq, sxs, ssm, lane, n_chunk, mu, tabs);
        const float v = pd_iqd_warp_fold(acc);
        if (lane == 0) y[o] = v;
    }
}

// y[o] = silu(gate_o . xq) * (up_o . xq), the fused decode pair (one format)
template <uint32_t DT, uint32_t NT>
__global__ void __launch_bounds__(NT) pd_iqd_glu_q8_kernel(
        const uint8_t* __restrict__ gd, const uint8_t* __restrict__ gs,
        const uint8_t* __restrict__ ud, const uint8_t* __restrict__ us,
        const int8_t* __restrict__ xq, const float* __restrict__ xs,
        const float* __restrict__ xsums, float* __restrict__ y,
        uint32_t in_dim, uint32_t out_dim) {
    PD_PDL_ARM();
    constexpr uint32_t dtype = DT, WARPS = NT / 32u;
    extern __shared__ uint8_t pd_iqd_smem[];
    const PdIqTabs tabs = pd_iq_tabs_stage(dtype, pd_iqd_smem);
    int4* sxq; float* sxs; float* ssm;
    pd_iqd_stage_q8(pd_iqd_smem, pd_iq_tabs_bytes(dtype), xq, xs, xsums, in_dim, NT, &sxq, &sxs, &ssm);
    __syncthreads();
    const bool mu = xsums != nullptr;
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u, n_chunk = in_dim >> 6;
    const size_t rdb = pd_kq_row_datab(dtype, in_dim), rsb = pd_kq_row_scb(dtype, in_dim);
    for (uint32_t o = blockIdx.x * WARPS + warp; o < out_dim; o += gridDim.x * WARPS) {
        const float ag = pd_iqd_q8_row_acc<DT>(gd + (size_t)o * rdb, gs + (size_t)o * rsb,
                                               sxq, sxs, ssm, lane, n_chunk, mu, tabs);
        const float au = pd_iqd_q8_row_acc<DT>(ud + (size_t)o * rdb, us + (size_t)o * rsb,
                                               sxq, sxs, ssm, lane, n_chunk, mu, tabs);
        const float gv = pd_iqd_warp_fold(ag), uv = pd_iqd_warp_fold(au);
        if (lane == 0) y[o] = (gv / (1.0f + expf(-gv))) * uv;
    }
}

// y[o] = row_o . x, f32 activations: x staged lane-major in PD_IQD_TILE-float
// tiles with its per-window sums (the g term of the k-quant-shaped
// formats), each warp carrying PD_IQD_F32_R rows across the tiles.
#define PD_IQD_TILE 4096u
#define PD_IQD_F32_R 4u
template <uint32_t DT, uint32_t NT>
__global__ void __launch_bounds__(NT) pd_iqd_gemv_f32_kernel(
        const uint8_t* __restrict__ data, const uint8_t* __restrict__ scales,
        const float* __restrict__ x, float* __restrict__ y,
        uint32_t in_dim, uint32_t out_dim) {
    PD_PDL_ARM();
    constexpr uint32_t dtype = DT, WARPS = NT / 32u;
    extern __shared__ uint8_t pd_iqd_smem[];
    const PdIqTabs tabs = pd_iq_tabs_stage(dtype, pd_iqd_smem);
    float4* xsh = reinterpret_cast<float4*>(pd_iqd_smem + pd_iq_tabs_bytes(dtype));
    float* xsum16 = reinterpret_cast<float*>(xsh + PD_IQD_TILE / 4u);
    const uint32_t warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const size_t rdb = pd_kq_row_datab(dtype, in_dim), rsb = pd_kq_row_scb(dtype, in_dim);
    const uint32_t o0 = (blockIdx.x * WARPS + warp) * PD_IQD_F32_R;
    float acc[PD_IQD_F32_R];
    #pragma unroll
    for (uint32_t r = 0; r < PD_IQD_F32_R; ++r) acc[r] = 0.0f;
    for (uint32_t x0 = 0; x0 < in_dim; x0 += PD_IQD_TILE) {
        const uint32_t tf = min(PD_IQD_TILE, in_dim - x0);
        __syncthreads();
        for (uint32_t i = threadIdx.x; i < (tf >> 2); i += NT) {
            const uint32_t w = i >> 2, k = i & 3u, slot = pd_iqd_slot(w);
            xsh[((slot >> 5) * 4u + k) * 32u + (slot & 31u)] =
                reinterpret_cast<const float4*>(x + x0)[i];
        }
        __syncthreads();
        for (uint32_t w = threadIdx.x; w < (tf >> 4); w += NT) {
            const uint32_t slot = pd_iqd_slot(w);
            float sm = 0.0f;
            #pragma unroll
            for (uint32_t k = 0; k < 4u; ++k) {
                const float4 v = xsh[((slot >> 5) * 4u + k) * 32u + (slot & 31u)];
                sm += (v.x + v.y) + (v.z + v.w);
            }
            xsum16[slot] = sm;
        }
        __syncthreads();
        #pragma unroll
        for (uint32_t r = 0; r < PD_IQD_F32_R; ++r) {
            const uint32_t o = o0 + r;
            if (o < out_dim)
                acc[r] += pd_iqd_f32_row_acc<DT>(data + (size_t)o * rdb, scales + (size_t)o * rsb,
                                                 xsh, xsum16, x0 >> 6, tf >> 6, lane, tabs);
        }
    }
    #pragma unroll
    for (uint32_t r = 0; r < PD_IQD_F32_R; ++r) {
        const float v = pd_iqd_warp_fold(acc[r]);
        if (lane == 0 && o0 + r < out_dim) y[o0 + r] = v;
    }
}

// out[t][:] = dequant(row tokens[t]) - the token embedding gather. One block
// per token, threads stride windows.
__global__ void pd_iqd_gather_kernel(const uint8_t* __restrict__ data,
                                     const uint8_t* __restrict__ scales,
                                     const unsigned int* __restrict__ tokens,
                                     float* __restrict__ out, uint32_t embd,
                                     uint32_t dtype) {
    const uint32_t t = blockIdx.x;
    const uint8_t *row, *rec;
    pd_iqd_row(data, scales, tokens[t], embd, dtype, &row, &rec);
    const uint32_t db = pd_kq_datab(dtype), scb = pd_kq_scb(dtype);
    float* dst = out + (size_t)t * embd;
    for (uint32_t base = threadIdx.x * 16u; base < embd; base += blockDim.x * 16u) {
        const uint32_t s = base >> 8, w = (base >> 4) & 15u;
        int wq[4];
        float f, g;
        pd_kq_win_unpack(dtype, row + (size_t)s * db, rec + (size_t)s * scb, w, wq, &f, &g);
        #pragma unroll
        for (uint32_t k = 0; k < 4u; ++k) {
            #pragma unroll
            for (uint32_t j = 0; j < 4u; ++j)
                dst[base + 4u * k + j] = f * (float)((int8_t)((wq[k] >> (8u * j)) & 0xffu)) + g;
        }
    }
}

// The staged kernels are persistent: the grid is the resident-block count
// (SMs x blocks/SM, capped by the work), so each block stages x once and
// walks rows grid-stride.
static inline uint32_t pd_iqd_sm_count() {
    static int v = 0;
    if (v == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev);
        if (v <= 0) v = 1;
    }
    return (uint32_t)v;
}
static inline uint32_t pd_iqd_grid(uint32_t rows_needed, uint32_t warps_per_block,
                                   uint32_t smem, uint32_t nt) {
    // blocks/SM: the tighter of the thread cap and the shared-memory cap
    int smem_sm = 0, dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&smem_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev);
    if (smem_sm <= 0) smem_sm = 102400;
    const uint32_t by_thr = 1536u / nt;
    const uint32_t by_smem = smem == 0u ? by_thr : (uint32_t)smem_sm / (smem + 1024u);
    const uint32_t per_sm = by_smem < by_thr ? (by_smem == 0u ? 1u : by_smem) : by_thr;
    const uint32_t resident = pd_iqd_sm_count() * per_sm;
    const uint32_t needed = (rows_needed + warps_per_block - 1u) / warps_per_block;
    return needed < resident ? needed : resident;
}

// ---- launchers (the k-quant exports dispatch here by type) ------------------

// The staged kernels take the quant type as a template parameter so the
// per-window format switch folds at compile time (the runtime switch across
// four unrolled windows cost 80-128 registers per thread and halved the
// occupancy); this expands a launch per type.
#define PD_IQD_FOR_DT(dtype, X)                          \
    switch (dtype) {                                      \
        case PD_KQ_IQ2XXS: X(PD_KQ_IQ2XXS); break;        \
        case PD_KQ_IQ2XS: X(PD_KQ_IQ2XS); break;          \
        case PD_KQ_IQ2S: X(PD_KQ_IQ2S); break;            \
        case PD_KQ_IQ3XXS: X(PD_KQ_IQ3XXS); break;        \
        case PD_KQ_IQ3S: X(PD_KQ_IQ3S); break;            \
        case PD_KQ_IQ1S: X(PD_KQ_IQ1S); break;            \
        case PD_KQ_IQ1M: X(PD_KQ_IQ1M); break;            \
        case PD_KQ_IQ4NL_ID: X(PD_KQ_IQ4NL_ID); break;    \
        case PD_KQ_Q2K_ID: X(PD_KQ_Q2K_ID); break;        \
        case PD_KQ_Q3K_ID: X(PD_KQ_Q3K_ID); break;        \
        default: return cudaErrorInvalidValue;            \
    }

// IQ4_NL rows lie flat, so in_dim only has to be a multiple of 32; the
// 256-block i-quant types need whole superblocks.
static inline bool pd_iqd_in_dim_ok(uint32_t dtype, uint32_t in_dim) {
    if (in_dim == 0u) return false;
    if (dtype == PD_KQ_IQ4NL_ID) return (in_dim & 31u) == 0u;
    return (in_dim & 255u) == 0u;
}

int pd_kq_iq_dense_dp4a(const void* data, const void* scales, const void* xq,
                        const void* xs, const void* xsums, void* y, uint32_t in_dim,
                        uint32_t out_dim, uint32_t batch, uint32_t dtype, void* stream) {
    if (out_dim == 0u || batch == 0u) return 0;
    if (!pd_iqd_in_dim_ok(dtype, in_dim)) return cudaErrorInvalidValue;
    if (!pd_kq_valid(dtype) && !pd_kq_valid_iq(dtype)) return cudaErrorInvalidValue;
    if (batch == 1u) {
        // the decode lane: x staged once per persistent block, a warp per row
        const size_t smem = pd_iq_tabs_bytes(dtype) + pd_iqd_q8_stage_bytes(in_dim);
        if (smem <= 48u * 1024u && (in_dim & 63u) == 0u) {
            constexpr uint32_t NT = 256u;
            const uint32_t blocks = pd_iqd_grid(out_dim, NT / 32u, (uint32_t)smem, NT);
            auto st = (cudaStream_t)stream;
#define PD_IQD_L_Q8(DT)                                                                      \
            pd_pdl_go(pd_iqd_gemv_q8_kernel<DT, NT>, blocks, NT, (uint32_t)smem, st,          \
                (const uint8_t*)data, (const uint8_t*)scales, (const int8_t*)xq, (const float*)xs, \
                (const float*)xsums, (float*)y, in_dim, out_dim)
            PD_IQD_FOR_DT(dtype, PD_IQD_L_Q8)
#undef PD_IQD_L_Q8
            return pd_launch_status();
        }
    }
    if ((size_t)out_dim * batch > 0xFFFFFFFFull) return cudaErrorInvalidValue;
    const uint32_t tasks = out_dim * batch;
    const uint32_t blocks = (tasks + 7u) / 8u;  // 8 warps (tasks) per block
    pd_iqd_dp4a_kernel<<<blocks, 256, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)data, (const uint8_t*)scales, (const int8_t*)xq, (const float*)xs,
        (const float*)xsums, (float*)y, in_dim, out_dim, batch, dtype);
    return pd_launch_status();
}

int pd_kq_iq_dense_glu(const void* gd, const void* gs, const void* ud, const void* us,
                       const void* xq, const void* xs, const void* xsums, void* y,
                       uint32_t in_dim, uint32_t out_dim, uint32_t dtg, uint32_t dtu,
                       void* stream) {
    if (out_dim == 0u) return cudaErrorInvalidValue;
    if (!pd_iqd_in_dim_ok(dtg, in_dim) || !pd_iqd_in_dim_ok(dtu, in_dim)) return cudaErrorInvalidValue;
    if ((!pd_kq_valid(dtg) && !pd_kq_valid_iq(dtg)) || (!pd_kq_valid(dtu) && !pd_kq_valid_iq(dtu)))
        return cudaErrorInvalidValue;
    if (dtg == dtu) {
        const size_t smem = pd_iq_tabs_bytes(dtg) + pd_iqd_q8_stage_bytes(in_dim);
        if (smem <= 48u * 1024u && (in_dim & 63u) == 0u) {
            constexpr uint32_t NT = 256u;
            const uint32_t blocks = pd_iqd_grid(out_dim, NT / 32u, (uint32_t)smem, NT);
            auto st = (cudaStream_t)stream;
#define PD_IQD_L_GLU(DT)                                                                     \
            pd_pdl_go(pd_iqd_glu_q8_kernel<DT, NT>, blocks, NT, (uint32_t)smem, st,           \
                (const uint8_t*)gd, (const uint8_t*)gs, (const uint8_t*)ud, (const uint8_t*)us,    \
                (const int8_t*)xq, (const float*)xs, (const float*)xsums, (float*)y, in_dim, out_dim)
            PD_IQD_FOR_DT(dtg, PD_IQD_L_GLU)
#undef PD_IQD_L_GLU
            return pd_launch_status();
        }
    }
    const uint32_t blocks = (out_dim * 32u + 255u) / 256u;
    pd_iqd_glu_kernel<<<blocks, 256, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)gd, (const uint8_t*)gs, (const uint8_t*)ud, (const uint8_t*)us,
        (const int8_t*)xq, (const float*)xs, (const float*)xsums, (float*)y,
        in_dim, out_dim, dtg, dtu);
    return pd_launch_status();
}

int pd_kq_iq_dense_gemv_f32(const void* data, const void* scales, const void* x, void* y,
                            uint32_t in_dim, uint32_t out_dim, uint32_t dtype, void* stream) {
    if (out_dim == 0u) return 0;
    if (!pd_iqd_in_dim_ok(dtype, in_dim) || !pd_kq_valid_iq(dtype)) return cudaErrorInvalidValue;
    if ((in_dim & 63u) != 0u) {
        const uint32_t blocks = (out_dim * 32u + 255u) / 256u;
        pd_iqd_gemv_f32_win_kernel<<<blocks, 256, 0, (cudaStream_t)stream>>>(
            (const uint8_t*)data, (const uint8_t*)scales, (const float*)x, (float*)y, in_dim, out_dim, dtype);
        return pd_launch_status();
    }
    const size_t smem = pd_iq_tabs_bytes(dtype) + (size_t)(PD_IQD_TILE + PD_IQD_TILE / 16u) * sizeof(float);
    constexpr uint32_t NT = 256u;
    const uint32_t rows_per_block = (NT / 32u) * PD_IQD_F32_R;
    const uint32_t blocks = (out_dim + rows_per_block - 1u) / rows_per_block;
    auto st = (cudaStream_t)stream;
#define PD_IQD_L_F32(DT)                                                                     \
    pd_pdl_go(pd_iqd_gemv_f32_kernel<DT, NT>, blocks, NT, (uint32_t)smem, st,                 \
        (const uint8_t*)data, (const uint8_t*)scales, (const float*)x, (float*)y, in_dim, out_dim)
    PD_IQD_FOR_DT(dtype, PD_IQD_L_F32)
#undef PD_IQD_L_F32
    return pd_launch_status();
}

int pd_kq_iq_dense_gather(const void* data, const void* scales, const void* tokens, void* out,
                          uint32_t embd, uint32_t n_tokens, uint32_t dtype, void* stream) {
    if (n_tokens == 0u) return 0;
    if (!pd_iqd_in_dim_ok(dtype, embd) || !pd_kq_valid_iq(dtype)) return cudaErrorInvalidValue;
    pd_iqd_gather_kernel<<<n_tokens, 256, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)data, (const uint8_t*)scales, (const unsigned int*)tokens,
        (float*)out, embd, dtype);
    return pd_launch_status();
}

// slot 578: capability marker - the dense k-quant entry points (gemv, gather,
// the W4A8 gemv / nc / multi / glu and the dp4a / mma_ks GEMMs) serve the
// i-quant family + IQ4_NL through these lanes.
PD_EXPORT
int pd_kquant_iq_dense(void) { return 0; }
