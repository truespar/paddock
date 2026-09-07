// moe/offload.cuh - MoE expert offload: a device-managed LRU cache of routed
// experts in VRAM, fed from a host-mapped mirror of the full expert planes.
//
// Everything here runs INSIDE the decode/prefill graphs: routing -> resolve
// (expert id -> cache slot, LRU victim on a miss) -> fill (copy the missing
// experts' repacked bytes from the pinned host mirror into their slots) ->
// the unchanged MoE kernels over the slot planes with the remapped ids. No
// host round-trip, no sync, and the cache state lives in device memory, so a
// captured graph keeps making correct decisions on every replay.
//
// Slot planes carry the same repacked k-quant layout as a resident plane
// (moe/kquant.cuh addressing `(slot*ff + o) * n_super * bytes`), which is
// what lets the consumer kernels stay untouched: a slot IS an expert index
// into a plane that happens to hold S experts instead of n_expert.
//
// resolve: one block, thread 0 walks the rows in order. Rows are at most a
// few hundred on the token-batched class this serves (decode: B x top-k), so
// a serial walk is microseconds and keeps the LRU bookkeeping trivially
// race-free. A row whose expert is resident takes its slot; a miss takes the
// least-recently-used slot that no row of THIS tick pinned (so a tick never
// evicts what it is about to read - the caller guarantees rows <= S). Empty
// slots have last_use 0 and are taken first.
//
// fill: grid (chunks, 6 streams, jobs). Every block re-reads the job count
// the resolve wrote, so the launch is shaped for the maximum and idles the
// blocks past the real count - graph-stable grids, live job counts.
// The six streams are gate/up/down x data/scales; sizes come from the
// planes, the kernel only knows byte ranges. uint4 copies, coalesced: this
// is the PCIe-bound path; measured at 12.6 GB/s on a PCIe 4.0 x8 link, the
// link's practical ceiling for this access pattern.

#define PD_MOE_CACHE_NONE 0xFFFFFFFFu

__global__ void pd_moe_cache_resolve_kernel(
    const unsigned int* __restrict__ idx, uint32_t rows, uint32_t n_slots,
    unsigned int* __restrict__ slot_of, unsigned int* __restrict__ expert_in,
    unsigned int* __restrict__ last_use, unsigned int* __restrict__ tick,
    unsigned int* __restrict__ idx_slot, unsigned int* __restrict__ jobs,
    unsigned int* __restrict__ n_jobs, unsigned int* __restrict__ stats) {
    if (threadIdx.x != 0) return;
    const unsigned int t = *tick + 1u;
    *tick = t;
    unsigned int nj = 0;
    for (uint32_t r = 0; r < rows; ++r) {
        const unsigned int e = idx[r];
        unsigned int s = slot_of[e];
        if (s != PD_MOE_CACHE_NONE) {
            idx_slot[r] = s;
            last_use[s] = t;
            continue;
        }
        // miss: LRU victim among slots not pinned by this tick
        unsigned int victim = PD_MOE_CACHE_NONE, best = 0xFFFFFFFFu;
        for (uint32_t c = 0; c < n_slots; ++c) {
            const unsigned int lu = last_use[c];
            if (lu != t && lu < best) { best = lu; victim = c; }
        }
        if (victim == PD_MOE_CACHE_NONE) {
            // cannot happen when rows <= n_slots (caller's contract); make the
            // failure loud rather than silent: point the row at slot 0 and
            // flag it in the job count's high bit
            idx_slot[r] = 0;
            nj |= 0x80000000u;
            continue;
        }
        const unsigned int old = expert_in[victim];
        if (old != PD_MOE_CACHE_NONE) slot_of[old] = PD_MOE_CACHE_NONE;
        expert_in[victim] = e;
        slot_of[e] = victim;
        last_use[victim] = t;
        idx_slot[r] = victim;
        jobs[2u * (nj & 0x7FFFFFFFu)] = victim;
        jobs[2u * (nj & 0x7FFFFFFFu) + 1u] = e;
        ++nj;
    }
    *n_jobs = nj;
    // running counters for the hit-rate readout: [0] rows resolved, [1] misses
    stats[0] += rows;
    stats[1] += nj & 0x7FFFFFFFu;
}

struct PdMoeFillDesc {
    unsigned long long src[6];    // host-mirror device pointers, one per stream
    unsigned long long dst[6];    // slot-plane base pointers
    unsigned long long bytes[6];  // bytes per expert per stream (multiple of 16)
};

__global__ void __launch_bounds__(256) pd_moe_cache_fill_kernel(
    const unsigned int* __restrict__ jobs, const unsigned int* __restrict__ n_jobs,
    const __grid_constant__ PdMoeFillDesc d) {
    const uint32_t j = blockIdx.z;
    if (j >= (*n_jobs & 0x7FFFFFFFu)) return;
    const uint32_t k = blockIdx.y;
    const unsigned int slot = jobs[2u * j], e = jobs[2u * j + 1u];
    const unsigned long long nb = d.bytes[k];
    const uint4* __restrict__ src = (const uint4*)(d.src[k] + (unsigned long long)e * nb);
    uint4* __restrict__ dst = (uint4*)(d.dst[k] + (unsigned long long)slot * nb);
    const uint32_t n16 = (uint32_t)(nb >> 4);
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += gridDim.x * blockDim.x)
        dst[i] = src[i];
}

// ---- expert-major prefill through the cache (waves) ------------------------
// A prefill launch routes n x top-k pairs - thousands of rows - against a
// cache of S slots, so the LRU path above cannot serve it (rows > S would
// evict what the tick reads) and the seats used to serve ZERO-COPY: every
// pair's expert read over PCIe by the MoE kernels themselves, again per
// row. Measured on the RTX 5060 Ti (Qwen3.8-Flash-Next UD-IQ1_S, 48 layers x
// 512 experts, top-10, 82 slots): a 256-token prompt took 46 s - ~5 tok/s.
//
// Expert-major instead, the discipline every offloaded-MoE serving system
// converges on (llama.cpp's CPU experts, ktransformers, fiddler): the
// bytes a prompt moves are bounded by ONE pass over the experts it touches,
// never by its rows. Per layer: plan = mark the experts present in the
// launch, enumerate them, assign wave w = ordinal / S. Then per wave:
// resolve its <= S ids through the SAME LRU (so the cache ends the prefill
// warm with the last wave), fill the misses once, and run the token-batched
// pair over ALL rows with every out-of-wave pair marked ABSENT
// (PD_MOE_CACHE_NONE in the routing): the gate/up block and the down warp
// of an absent pair return at once, so a wave costs its own pairs, and the
// engine sums the waves' down partials. (A first cut pointed absent pairs
// at a permanently zero slot instead, which is correct but priced every
// wave at every pair: 10 ms a launch on a 435-token prompt, 77% of the
// prefill.) Waves are a fixed count (ceil(n_expert / S)) so a captured
// graph keeps its shape; an empty wave resolves nothing, fills nothing and
// its pair kernels exit block by block.
//
// plan: one block. wave_of[e] = wave of expert e (NONE if absent) - the
// mask kernel's whole input; wave_ids[w*S + i] = the wave's ids, wave_cnt[w].
__global__ void pd_moe_wave_plan_kernel(
    const unsigned int* __restrict__ idx, uint32_t rows, uint32_t n_expert,
    uint32_t n_slots, uint32_t n_waves, unsigned int* __restrict__ wave_of,
    unsigned int* __restrict__ wave_ids, unsigned int* __restrict__ wave_cnt) {
    for (uint32_t e = threadIdx.x; e < n_expert; e += blockDim.x) wave_of[e] = PD_MOE_CACHE_NONE;
    __syncthreads();
    // mark present: same value from every writer, no atomics needed
    for (uint32_t r = threadIdx.x; r < rows; r += blockDim.x) {
        const unsigned int e = idx[r];
        if (e < n_expert) wave_of[e] = 0u;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        uint32_t cnt = 0;
        for (uint32_t e = 0; e < n_expert; ++e) {
            if (wave_of[e] == PD_MOE_CACHE_NONE) continue;
            const uint32_t w = cnt / n_slots;
            if (w >= n_waves) break;   // cannot happen: n_waves = ceil(n_expert / S)
            wave_of[e] = w;
            wave_ids[w * n_slots + (cnt % n_slots)] = e;
            ++cnt;
        }
        for (uint32_t w = 0; w < n_waves; ++w) {
            const uint32_t lo = w * n_slots;
            wave_cnt[w] = cnt > lo ? (cnt - lo < n_slots ? cnt - lo : n_slots) : 0u;
        }
    }
}

// resolve with a DEVICE-side id count: the wave's ids are a device list the
// plan kernel filled, so the count cannot be a host argument on a captured
// path. Same LRU walk as pd_moe_cache_resolve_kernel over `*n_ids` ids;
// max_ids bounds the scratch (<= n_slots by construction).
__global__ void pd_moe_cache_resolve_dev_kernel(
    const unsigned int* __restrict__ ids, const unsigned int* __restrict__ n_ids,
    uint32_t n_slots, unsigned int* __restrict__ slot_of,
    unsigned int* __restrict__ expert_in, unsigned int* __restrict__ last_use,
    unsigned int* __restrict__ tick, unsigned int* __restrict__ jobs,
    unsigned int* __restrict__ n_jobs, unsigned int* __restrict__ stats) {
    if (threadIdx.x != 0) return;
    const unsigned int rows = *n_ids;
    const unsigned int t = *tick + 1u;
    *tick = t;
    unsigned int nj = 0;
    for (uint32_t r = 0; r < rows && r < n_slots; ++r) {
        const unsigned int e = ids[r];
        unsigned int s = slot_of[e];
        if (s != PD_MOE_CACHE_NONE) {
            last_use[s] = t;
            continue;
        }
        unsigned int victim = PD_MOE_CACHE_NONE, best = 0xFFFFFFFFu;
        for (uint32_t c = 0; c < n_slots; ++c) {
            const unsigned int lu = last_use[c];
            if (lu != t && lu < best) { best = lu; victim = c; }
        }
        if (victim == PD_MOE_CACHE_NONE) { nj |= 0x80000000u; continue; }
        const unsigned int old = expert_in[victim];
        if (old != PD_MOE_CACHE_NONE) slot_of[old] = PD_MOE_CACHE_NONE;
        expert_in[victim] = e;
        slot_of[e] = victim;
        last_use[victim] = t;
        jobs[2u * (nj & 0x7FFFFFFFu)] = victim;
        jobs[2u * (nj & 0x7FFFFFFFu) + 1u] = e;
        ++nj;
    }
    *n_jobs = nj;
    stats[0] += rows;
    stats[1] += nj & 0x7FFFFFFFu;
}

// mask + compact: the wave's remapped routing (in-wave pairs to their slot,
// every other pair `absent`), plus the two lists the LIST pair kernels
// stride over - the wave's pair indices and the tokens holding at least one
// of them - with device counts. One block: the mask is parallel, the
// compaction a serial walk by thread 0 (rows are at most max_rows, tens of
// thousands: tens of microseconds, 7 waves x 48 layers = well under a
// tick's worth per prompt). Pairs of a token are adjacent in `idx`
// (token*n_active + slot), so the token list is the run boundaries.
__global__ void pd_moe_wave_mask_kernel(
    const unsigned int* __restrict__ idx, uint32_t rows, uint32_t n_active,
    const unsigned int* __restrict__ wave_of, const unsigned int* __restrict__ slot_of,
    uint32_t wave, uint32_t absent, unsigned int* __restrict__ idx_slot,
    unsigned int* __restrict__ pairs, unsigned int* __restrict__ n_pairs,
    unsigned int* __restrict__ rows_list, unsigned int* __restrict__ n_rows) {
    for (uint32_t r = threadIdx.x; r < rows; r += blockDim.x) {
        const unsigned int e = idx[r];
        idx_slot[r] = (wave_of[e] == wave) ? slot_of[e] : absent;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        uint32_t m = 0, mr = 0, last_b = 0xFFFFFFFFu;
        for (uint32_t r = 0; r < rows; ++r) {
            if (idx_slot[r] == absent) continue;
            pairs[m++] = r;
            const uint32_t b = r / n_active;
            if (b != last_b) { rows_list[mr++] = b; last_b = b; }
        }
        *n_pairs = m;
        *n_rows = mr;
    }
}

PD_EXPORT
int pd_moe_wave_plan(const void* idx, uint32_t rows, uint32_t n_expert, uint32_t n_slots,
                     uint32_t n_waves, void* wave_of, void* wave_ids, void* wave_cnt,
                     void* stream) {
    if (rows == 0 || n_expert == 0 || n_slots == 0 || n_waves == 0) return cudaErrorInvalidValue;
    if ((uint64_t)n_waves * n_slots < n_expert) return cudaErrorInvalidValue;
    pd_moe_wave_plan_kernel<<<1, 256, 0, (cudaStream_t)stream>>>(
        (const unsigned int*)idx, rows, n_expert, n_slots, n_waves,
        (unsigned int*)wave_of, (unsigned int*)wave_ids, (unsigned int*)wave_cnt);
    return pd_launch_status();
}

PD_EXPORT
int pd_moe_cache_resolve_dev(const void* ids, const void* n_ids, uint32_t n_slots,
                             void* slot_of, void* expert_in, void* last_use, void* tick,
                             void* jobs, void* n_jobs, void* stats, void* stream) {
    if (n_slots == 0) return cudaErrorInvalidValue;
    pd_moe_cache_resolve_dev_kernel<<<1, 32, 0, (cudaStream_t)stream>>>(
        (const unsigned int*)ids, (const unsigned int*)n_ids, n_slots,
        (unsigned int*)slot_of, (unsigned int*)expert_in, (unsigned int*)last_use,
        (unsigned int*)tick, (unsigned int*)jobs, (unsigned int*)n_jobs,
        (unsigned int*)stats);
    return pd_launch_status();
}

PD_EXPORT
int pd_moe_wave_mask(const void* idx, uint32_t rows, uint32_t n_active, const void* wave_of,
                     const void* slot_of, uint32_t wave, uint32_t absent, void* idx_slot,
                     void* pairs, void* n_pairs, void* rows_list, void* n_rows, void* stream) {
    if (rows == 0 || n_active == 0) return 0;
    pd_moe_wave_mask_kernel<<<1, 1024, 0, (cudaStream_t)stream>>>(
        (const unsigned int*)idx, rows, n_active, (const unsigned int*)wave_of,
        (const unsigned int*)slot_of, wave, absent, (unsigned int*)idx_slot,
        (unsigned int*)pairs, (unsigned int*)n_pairs, (unsigned int*)rows_list,
        (unsigned int*)n_rows);
    return pd_launch_status();
}

PD_EXPORT
int pd_moe_cache_resolve(const void* idx, uint32_t rows, uint32_t n_slots,
                         void* slot_of, void* expert_in, void* last_use,
                         void* tick, void* idx_slot, void* jobs, void* n_jobs,
                         void* stats, void* stream) {
    if (rows == 0 || n_slots == 0) return cudaErrorInvalidValue;
    if (rows > n_slots) return cudaErrorInvalidValue;
    pd_moe_cache_resolve_kernel<<<1, 32, 0, (cudaStream_t)stream>>>(
        (const unsigned int*)idx, rows, n_slots, (unsigned int*)slot_of,
        (unsigned int*)expert_in, (unsigned int*)last_use, (unsigned int*)tick,
        (unsigned int*)idx_slot, (unsigned int*)jobs, (unsigned int*)n_jobs,
        (unsigned int*)stats);
    return pd_launch_status();
}

// src/dst/bytes: HOST arrays of 6 u64 each, copied into the launch by value.
PD_EXPORT
int pd_moe_cache_fill(const void* jobs, const void* n_jobs, uint32_t max_jobs,
                      const void* src, const void* dst, const void* bytes,
                      void* stream) {
    if (max_jobs == 0) return 0;
    if (max_jobs > 1024u) return cudaErrorInvalidValue;
    PdMoeFillDesc d;
    for (int k = 0; k < 6; ++k) {
        d.src[k] = ((const unsigned long long*)src)[k];
        d.dst[k] = ((const unsigned long long*)dst)[k];
        d.bytes[k] = ((const unsigned long long*)bytes)[k];
        if (d.bytes[k] & 15ull) return cudaErrorInvalidValue;
    }
    // 16 chunks x 256 threads x 16 B = 64 KB per pass over a ~0.5 MB stream:
    // enough blocks in flight to keep the link busy (measured flat from 4 to
    // 64 chunks), small enough that a 1-job decode fill is not a 1000-block
    // launch
    dim3 grid(16u, 6u, max_jobs);
    pd_moe_cache_fill_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const unsigned int*)jobs, (const unsigned int*)n_jobs, d);
    return pd_launch_status();
}
