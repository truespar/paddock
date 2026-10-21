// gemm/tc5_lanes.cuh - per-stream scratch for the sm_100 tcgen05 decode arms
// (tc5q work counter, tc5p fold counters, two-stage row-quant maxes).
// Textually-included segment of the single pack translation unit (cuda runtime
// + abi.cuh only); dense_tc5.cuh and dense_f8_decode.cuh consume it. Split out
// of dense_tc5.cuh, which sits at its file-length baseline.

// Per-STREAM scratch for the sm_100 decode arms: the tc5q work counter, the
// tc5p fold counters and the two-stage row-quant partial maxes. Each was one
// copy for the process, which was fine while one stream ran the decode. The
// overlap pump runs a prefill span's eager GEMMs on the main lane while the
// decode lane replays its captured graphs, and on sm_100 both reach these
// arms: a span share of <= 64 rows takes tc5q on the big planes, the decode
// graph at 25-64 live rows takes tc5q too, so the two persistent grids
// claimed items off ONE counter (each launch computed some of the other's
// tiles with its own operands) and the row-quant maxes interleaved. Exact-
// output probe on a B200 (Qwen3.8-27B, 40 workers "1 to 120" beside picture
// prompts): 118 of 120 replies wrong with the pump, 0 with
// PADDOCK_PUMP_SERIAL / PADDOCK_OVERLAP=0, and text alone corrupts too.
// Same class as the KT lane's flags (f8_lin.cuh, sm_120) and the same cure:
// a stream owns its scratch. The pool is built at load (pd_f8_repack_tiles)
// because the decode lane only reaches these arms inside a graph capture,
// where nothing may be allocated; lane assignment is host bookkeeping, legal
// mid-capture. A stream beyond the pool gets nullptr, which every caller
// already treats as "arm unavailable": tc5q declines to tc5p, no fold means
// the separate combine, no row-quant scratch means the single-block walk.
struct PdTc5Lane {
    cudaStream_t st; uint32_t* qctr; uint32_t* fctr; float* rowq;
    unsigned char* xsw; size_t xsw_cap;   // swizzled activation scales (pd_tc5_xsw)
};
#define PD_TC5_LANES 8
static PdTc5Lane pd_tc5_lanes[PD_TC5_LANES] = {};
static int pd_tc5_lanes_used = 0;
static int pd_tc5_pool_ready = 0;
// one allocation carved per lane: 4 B work counter (kernel resets it; the
// graph replays memsetAsync(0) + kernel), 256 fold slots (tiles <= 256 -
// tiles >= 256 goes tc5q), 64 rows x 8 chunks of row-quant maxes
constexpr size_t PD_TC5_LANE_QCTR = 0, PD_TC5_LANE_FCTR = 256,
                 PD_TC5_LANE_ROWQ = PD_TC5_LANE_FCTR + 1024, PD_TC5_LANE_BYTES = PD_TC5_LANE_ROWQ + 64 * 8 * 4;
#define PD_TC5_XSW_MIN (128u * 2048u)
static void pd_tc5_pool_prepare() {
    if (pd_tc5_pool_ready) return;
    unsigned char* base = nullptr;
    if (cudaMalloc((void**)&base, PD_TC5_LANES * PD_TC5_LANE_BYTES) != cudaSuccess) {
        cudaGetLastError();
        return;   // every lane stays empty: the callers decline the arms
    }
    cudaMemset(base, 0, PD_TC5_LANES * PD_TC5_LANE_BYTES);
    for (int i = 0; i < PD_TC5_LANES; ++i) {
        unsigned char* b = base + (size_t)i * PD_TC5_LANE_BYTES;
        pd_tc5_lanes[i] = PdTc5Lane{nullptr, (uint32_t*)(b + PD_TC5_LANE_QCTR),
                                    (uint32_t*)(b + PD_TC5_LANE_FCTR), (float*)(b + PD_TC5_LANE_ROWQ),
                                    nullptr, 0};
        // one 128-row group of the widest plane (2048 scale bytes a row) up front: the
        // captured decode graphs reach the swizzled-scale arm at 65..128 rows and may
        // not allocate; eager (prefill) ticks grow it in pd_tc5_xsw
        if (cudaMalloc((void**)&pd_tc5_lanes[i].xsw, PD_TC5_XSW_MIN) == cudaSuccess)
            pd_tc5_lanes[i].xsw_cap = PD_TC5_XSW_MIN;
        else
            cudaGetLastError();
    }
    pd_tc5_pool_ready = 1;
}
static PdTc5Lane* pd_tc5_lane(cudaStream_t st) {
    if (!pd_tc5_pool_ready) return nullptr;
    for (int i = 0; i < pd_tc5_lanes_used; ++i)
        if (pd_tc5_lanes[i].st == st) return &pd_tc5_lanes[i];
    if (pd_tc5_lanes_used >= PD_TC5_LANES) return nullptr;
    PdTc5Lane* l = &pd_tc5_lanes[pd_tc5_lanes_used++];
    l->st = st;
    ::fprintf(stderr, "[tc5-lane] stream %p -> lane %d\n", (void*)st, pd_tc5_lanes_used - 1);
    return l;
}
static uint32_t* pd_tc5q_ctr(cudaStream_t st) {
    PdTc5Lane* l = pd_tc5_lane(st);
    return l ? l->qctr : nullptr;
}

// scratch for the row-partial maxes: the stream's lane (built at load, the
// decode step is captured and cudaMalloc is illegal inside). 64 rows x 8
// chunks covers every widened shape - wider rows keep the single-block path.
static float* pd_rowq_scr(cudaStream_t st) {
    PdTc5Lane* l = pd_tc5_lane(st);
    return l ? l->rowq : nullptr;
}

// per-tile K-split fold counters (tc5p last-CTA combine fold): zeroed once
// at alloc. Each launch's folder CTA resets its slot to 0 after folding, so
// no per-launch memset - same-stream kernels serialize completely, so the
// next launch never observes a mid-flight counter; the stream's own lane is
// what makes "same-stream" true under the two-lane pump. 256 slots covers
// every tc5p shape (tiles = out_dim/128 <= 168; tiles >= 256 goes tc5q).
static uint32_t* pd_tc5p_fctr(cudaStream_t st) {
    PdTc5Lane* l = pd_tc5_lane(st);
    return l ? l->fctr : nullptr;
}

// The stream's scratch for activation scales in the 128x4 swizzled atom layout
// (dense_nvf4_tc5.cuh's pd_sf_swz128): `bytes` of it, or nullptr when the lane has
// less and the stream is capturing (nothing may be allocated there) - the caller then
// keeps its gather. Growth is stream-ordered (cudaFreeAsync / cudaMallocAsync), so a
// launch still reading the old buffer finishes first.
static unsigned char* pd_tc5_xsw(cudaStream_t st, size_t bytes) {
    PdTc5Lane* l = pd_tc5_lane(st);
    if (!l) return nullptr;
    if (bytes <= l->xsw_cap) return l->xsw;
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(st, &cs) != cudaSuccess || cs != cudaStreamCaptureStatusNone) return nullptr;
    size_t cap = l->xsw_cap ? l->xsw_cap : PD_TC5_XSW_MIN;
    while (cap < bytes) cap *= 2u;
    unsigned char* nb = nullptr;
    if (cudaMallocAsync((void**)&nb, cap, st) != cudaSuccess) { cudaGetLastError(); return nullptr; }
    if (l->xsw) cudaFreeAsync(l->xsw, st);
    l->xsw = nb;
    l->xsw_cap = cap;
    return nb;
}
