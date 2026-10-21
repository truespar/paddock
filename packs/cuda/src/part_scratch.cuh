// part_scratch.cuh - split-K partial planes for the row-wise GEMM arms.
// Textually-included segment of the single pack translation unit (leaf:
// cuda runtime only).
//
// One buffer PER STREAM, grown by allocating a new one and RETIRING the old
// (never cudaFree). Two rules the per-arm `static float* part` grow-by-free
// statics broke once the engine ran two lanes:
//   * a captured decode graph bakes the pointer the launcher handed it; an
//     eager call on another stream that outgrew the buffer freed that pointer
//     under the graph, and its next replay wrote partials into unmapped
//     memory (sm_120, Qwen3.8-27B mixed text+picture serve: Warp Illegal
//     Address in pd_f8row_gemm_mma_kernel<64>, caught with
//     CUDA_DEVICE_WAITS_ON_EXCEPTION + cuda-gdb; 1-in-3 runs, order-dependent
//     because the seed size came from whichever out_dim called first).
//   * the main lane's eager prefill shares and the decode lane's graph
//     replays run concurrently under the overlap scheduler; one shared plane
//     raced (a picture span's trailing f8row shares against the pipe's
//     decode graphs corrupted pipe rows at >= 36 live). A plane per stream
//     keeps each lane's partials its own.
// Nothing is allocated mid-capture: the caller sees the stream's current
// plane (or none) and falls to nz=1 there, correct and unsplit, as before.
// Retired planes stay mapped for the graphs recorded against them; the cost
// is one earlier size per growth step (tens of MB at most), bounded by the
// distinct out_dims a model has.
struct PdPartPlane { cudaStream_t st; float* p; size_t sz; };

static float* pd_part_scratch(cudaStream_t st, size_t want, size_t* have) {
    static PdPartPlane live[8] = {};
    static int used = 0;
    static float* retired[64] = {};
    static int n_retired = 0;
    int i = 0;
    for (; i < used; ++i)
        if (live[i].st == st) break;
    if (i == used) {
        if (used >= 8) { *have = 0; return nullptr; }
        live[used++] = PdPartPlane{st, nullptr, 0};
    }
    if (want > live[i].sz) {
        cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(st, &cs) == cudaSuccess && cs == cudaStreamCaptureStatusNone) {
            float* np = nullptr;
            if (cudaMalloc(&np, want) == cudaSuccess) {
                if (live[i].p && n_retired < 64) retired[n_retired++] = live[i].p;
                live[i].p = np;
                live[i].sz = want;
            } else {
                cudaGetLastError();
            }
        }
    }
    *have = live[i].sz;
    return live[i].p;
}
