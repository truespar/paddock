// deltanet/core.cuh (formerly 05_deltanet.cuh) - PackInfo blob; Gated DeltaNet recurrence, causal conv1d+silu, gates, mRoPE
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
// ------------------------------------------------------------------- exports

static const PackInfo PD_INFO = {
    PD_PACK_MAGIC,
    PD_ABI_VERSION,
    { 'c','u','d','a','-','m','u','l','t','i', 0, 0, 0, 0, 0, 0 },
    // 0.17: pf-runs v4 run-table arm  - the engine's muse
    // tuned-default gates PF_RUNS on >= 0.17 (older bodies share the same
    // exports but attend slot 0 on whole-chunk launches).
    // 0.17.1: glue-band twins: glu2_b16 single-pass, rope-NORM
    // pair-dedup'd angles, row1p narrow blocks at wide grids - all
    // bit-identical body changes, no ABI move.
    // 0.17.2: cutlass 256x256 both-wide tile (WideB16Big) tried
    // first at wave-class m behind PADDOCK_F8CUT_BIG - new tile config in the
    // sm_100a cutgemm TU, no ABI move.
    // 0.18.0: f16xf16->f32 tensor-core dense GEMM slot (pd_f16_gemm,
    // PADDOCK_INHOUSE_F16 cuBLAS removal) - ADDS a table slot (383), minor bump.
    // 0.18.1: pd_f16_gemm large-regular path is now the hand-rolled ldmatrix+
    // mma.sync m16n8k16 twin (dynamic-smem cp.async ring, 128x128/w16/ST3/KT64)
    // - ~5-6x the wmma first cut; body-only, no ABI move.
    // 0.18.2: pd_f16_gemm sm_100a arm - tcgen05 cta_group::2 duo GEMM
    // (persistent cluster pairs, two W-sharing 256x256 tiles in flight,
    // kind::f16, bit-identical to cuBLAS) at batch >= 256; warp-mma issues at
    // Ampere parity on Blackwell (554 TF ceiling) so this is the only class
    // that can reach nvjet. 745-1260 TF by vision shape (cuBLAS 1.3-1.7 PF).
    // Kill: PADDOCK_NO_F16TC5. Body-only, no ABI move.
    // 0.18.3: pd_f16_gemm tc5p arm ahead of the duo - ping-pong 256xNT tile
    // (D across 2x256 tmem cols, drain hidden), one 3D-box TMA per operand
    // per 2-slab slot (8 mmas per wait/commit), staged TMA-store epilogue
    // (st.global retirement poisons the chain's SMSP; STS+bulk-store doesn't),
    // NT {256,192} + K-split {1,3} elected per shape. All six muse vision
    // shapes 1.13-1.52x cuBLAS (duo was 1.35-1.80x). Needs in_dim%64,
    // out_dim%4, beta in {0,1}; declines to the duo otherwise. Body-only.
    // 0.18.4: pd_f16_gemm tc5g SKINNY arm (batch <= 128, ::1 no-cluster M128
    // tiles, per-chunk TMA ring, TMA-store/STG epilogue election, K-split on
    // the shared flags protocol) + paired-tile tc5gp (elected U0p >= SM count)
    // + the beta-ternary RMW fix in the mma-twin and duo epilogues (the
    // per-element `beta!=0 ? *o : 0` compiled to an unconditional LDG+FSEL -
    // every store paid a global round-trip even at beta=0). Testbed measured
    // vs same-day cuBLAS: whisper head 1.68-1.74x, M72 tie, laguna partial
    // (serve A/B is the acceptance bar). Kill: PADDOCK_NO_F16TC5G. Body-only.
    // 0.18.5: pd_f16_gemm GEMV band twin (batch<=8, all arches) - plain LDG
    // dot-product with swizzled smem X; the whisper serve A/B showed the tc5g
    // ring's flat ~7.8us span losing to nvjet's 4.5us on every small decode
    // projection (192/step) - at N<=8 the f32 cores hold DRAM pace and the
    // ring's ramp+drain+TMA-completion span is pure overhead. Election:
    // b<=4 always, b5-8 at out<=2560; head class stays tc5g. Kill:
    // PADDOCK_NO_F16GEMV. Body-only.
    // 0.18.6: GEMV b5-8 election adds in_dim<=2048 - the FMA-issue wall
    // scales with in_dim*batch, and deep-K b5-8 (whisper fc2 K=5120: GEMV
    // ~17us vs tc5g 7.8 vs cuBLAS 9.6) was a 0.18.5 regression. tc5g tiny-KS
    // refloor + STG/hybrid K-split epilogues measured and FALSIFIED (any
    // KS>1 combine protocol floors at ~2.6-4us; the tuned KS/4+TMA policy
    // stands). Body-only.
    // 0.18.7: pd_f16_gemm mmaf arm (batch 5-32, sm_100): 32/64-row mma.sync
    // tiles fed by the tc5g TMA ring, no K-split (the LAW) - batch splits
    // across grid.y; R64/CH2 halves grid + per-row B traffic when R32 would
    // co-schedule; NSLOT=4 half-ring 2/SM is the last resort. Measured vs
    // the shipped route: wo 4.35/6.95, wo b32 4.49/7.40,
    // qkv b32 5.81/7.80, fc1 b8 5.24/7.71, fc1 b32 7.62/8.29. Also: tc5g
    // in-kernel useTma now carries the launcher's y-16B check (misaligned-y
    // + out%4==0 would have run the TMA epilogue with no stage smem and the
    // dummy ym map - unreachable today, real landmine). Kill:
    // PADDOCK_NO_F16MMAF. Body-only, no ABI move.
    // 0.18.8: whisper encoder GEMM geometry. tc5p low-fill rescue:
    // out_dim so small that NT256 idles half the clusters elects NT128 (wo
    // 12.60->11.50, fc2 31.30->25.69; K-splits FALSIFIED there - combine
    // tax > fill gain; no muse vision shape can hit the gate at N=3888).
    // New slots 388 (whisper_enc_qkv_split: encoder q,k,v as one M=3840 tc5p
    // call, 3x12.60->19.09, biases folded, bias_adds gone) and 389
    // (whisper_kv_store_batch: the 64 cross-K/V GEMMs share enc_stage ->
    // two M=40960 calls, 806->270us/encode, one store launch per plane set).
    // ABI: two tail appends, size guard re-derived (389 slots).
    // 0.18.9: tc5p issuer slot-path fix. Per slot the one
    // issuer thread paid two commits (own+peer, watermark depth drained the
    // tensor pipe each slot boundary) and two spin-waits - +214ns/slot over
    // the 282ns NT128 compute floor, against nvjet's +40 (chain-law fit;
    // ring depth S=4 alone measured NULL, banked). Now: peer readiness
    // merged into a count-2 bfull (watcher arrives the leader's bfull, bpeer
    // dead), one multicast::cluster commit (ctamask 3) per slot and per unit
    // tfull. Unmasked ring cover -> low-fill elect goes <4,1,128> (same
    // 230400B smem as <3,1,256>). Cold-W probe: fc2 26.6->24.6, qkvf@128
    // -1.9, KVa@256 -6.1, NT192/256 short shapes unchanged (their slot
    // compute hides issuer overhead - the pattern that confirmed the
    // mechanism). Outputs bit-identical (sync-only; S3/S4 asserted equal).
    // Watcher tight-spin: NULL on short chains, kept. Body-only, no
    // ABI move.
    // 0.18.10: tc5p single-burst mma issue. SASS audit
    // found the residual +128ns/slot: ptxas wrapped each of the 8 per-slot
    // mma asm statements in its own ELECT/reconverge guard + rebuilt the
    // constant descriptor hi-fields every issue (~100 uniform instr per
    // 282ns slot). One asm block per slot with add.u64 running descriptors
    // (sdesc is linear in the addr field) cuts the burst to 57 instr; ptxas
    // still mandates one ELECT per UTCHMMA - that floor stays. Slope
    // 0.410 -> 0.341 us/slot-128K (nvjet 0.322). fc2 24.72 -> 22.54, qkvf@128
    // ties its NT192 elect, KVf ties NT256; S3 rows unchanged (there the
    // ring cover binds, not the issuer). Bit-exact (same mma order/operands/
    // enables). Delayed-commit V2 FALSIFIED with regression (release lag
    // costs ring cover; 2-slot cover starves TMA) - not shipped. Remaining
    // encoder deficit is the FIXED term (F ~6.4 vs nvjet 4.39: ramp+drain)
    // and the fc1 multi-unit walk. Body-only, no ABI move.
    // 0.18.11: whisper batched admission  - ADDS table slot 405
    // (399-404 are the nemotron appends from the other machine)
    // (pd_whisper_kv_store_slots: slot 389 cross-K/V store off an
    // audio-major landing, row r -> slots[r / rows_per_slot]); 389's
    // signature stays frozen (its kernel gains the rps parameter with
    // rps == rows == the old single-slot behavior, bit-identical). Minor
    // bump for the append. (The 0.18.11 tag from the falsified PDL round
    // was never shipped - reusing it for this append is safe: no binary
    // with the old meaning left the box.)
    // 0.18.12: mbarrier contract fixes in the f16 TMA-ring twins:
    // fence.mbarrier_init after the init loops in mmaf/tc5g/tc5gp
    // (duo and tc5p already had it) + explicit .acquire.cta and a memory
    // clobber on the mmaf consumer spin. Contract-correct and free; note
    // they do not close the mmaf-x-tc5p concurrent stale-slab reads (the
    // probe rate is unchanged - below-PTX mechanism, mmaf_race3.cu is the
    // repro; whisper encode/decode stream overlap stays gated). Body-only.
    // 0.18.13: slot 409 f16_mmaf_set (dual-graph routing) - a
    // capture-time gate on pd_f16_gemm's mmaf arm, so whisper can capture
    // an mmaf-off decode-graph variant and replay it only on ticks whose
    // admission encode overlaps on the side stream (the poison pairing
    // is mmaf x tc5p; every other decode lane measured overlap-clean).
    // Pure append; default 1 == 0.18.12 behavior everywhere.
    // 0.19.0: table slots 420 (moe_topk_sigmoid_batch_sh, shared
    // fold-in - that append landed without its own bump) + 421
    // (gemma_qkv_nra3_b16 - the packed-bf16 q/k/v read twin for
    // the spec-verify b16-D election). Minor bump for the appends; every
    // pre-existing body is unchanged.
    // 0.20.0: table slots 462 (gated_delta_verify_hold - v2's spec-verify
    // twin: no snapshots, no final state writeback, live state stays at
    // round-start) + 463 (gated_delta_commit_walk - commit-time recompute
    // over each row's accepted prefix from the round-start state on the
    // stashed split/gate planes; bit-exact vs the snapshot restore).
    // This kills the b x k1 x H x D x D
    // spec snapshot allocation (~87% of the 1.15 GiB/spec-row draft state,
    // the 14-row cap on the 96 GB card) and the O(k1) per-round state write
    // traffic - the same snapshot-per-step shape that dominates spec memory.
    // Minor bump for the appends; every pre-existing body is unchanged.
    // 0.21.0: table slot 464 (dflash_chain_picks - the async block round's
    // device-side pick copy into the MTP chain's d_draft layout; kills the
    // per-round dtoh sync on the dflash draft->verify boundary). Pure append.
    // 0.22.0: table slots 465-468 (nv4cut_sf_bytes / sf_repack / quant_a /
    // gemm - the checkpoint-native NVFP4 decode GEMM, landed by the nvfp4
    // lane without its own bump) + 469 (dflash_cond_append - the
    // conditioning fold: per-layer k-norm + rope + paged K/V store in one
    // launch over the written rows, replacing the norm + rope + 2 x cuts
    // append train on the drafter ring; pool bytes bit-identical, rung C
    // Pure appends.
    // 0.22.1: pd_f8d_gemm_mma_ks takes any batch (the `b <= 64` refusal was
    // the 64-row wall under every qwen35 spec round deeper than k=1 at 32
    // live) and its kernel's grid roles swap so weight tiles stream once at
    // batch > 64 - scheduling only, every output bit-identical. Rung D of
    // No slot change.
    // 0.22.2: pd_attn_spec_batch_paged/_fin elect the fp8 hd256/G6 (qwen35
    // 24q/4kv) geometry onto the krs GV=6 arm at k1 > 1 - the dense-decode
    // class, one KV walk per (kv-head, slot block, split) for the verify
    // rows. Existing slots, new geometry accepted (was -2 -> the engine's
    // per-row decode walk). Rung E1. No slot change.
    // 0.23.0: table slots 470 (dflash_select_rs - the DFlash2 selector walk
    // Gumbel-SAMPLED at the request temperature, writing the K-way draft
    // distribution) + 471 (dflash_rs_resolve - canonical rejection-sampling
    // verify over the K candidates, truncation-aware: accept w.p.
    // min(1, p/q) against the mode-5 nucleus, residual on reject). Rung G
    //Pure appends.
    // 0.23.1: pd_quantize_q8/_relu2 go 8-warps-per-CTA (the 1-warp CTA made
    // verify-plane quantizes CTA-dispatch-bound: 36.9 us for ~5.8 MB, ~10%
    // of DRAM roof, x75/round at c32 - bit-identical outputs), and the lin
    // ktz K-split caps nz at 2 for batch 129..256 (the spec-verify band's 2
    // col tiles already double the chain count; down/dnout z4 -> z2 measured
    // -1.9%/-12.7% isolated, decode-ks numeric class). . No slot
    // change.
    // 0.24.0: qwen4_exp (Qwen3.8-Flash-Next) family, slots 506-516 -
    // grouped (1+w) RMSNorm, hyper-connection mix/combine, PLE n-gram gate,
    // DILATED causal conv1d+silu (prefill + windowed step; the pack had no
    // dilation anywhere), GDN sigmoid gated-norm and the repeat-interleave
    // key-head split, the shared-expert scalar-gate fold, and (in
    // moe/nvf4_expert.cuh) the NVFP4 gate+up GEMV with a fused swiglu - the
    // first nvf4 expert consumer here that has a gate matrix at all.
    // New segment src/qwen4exp.cuh. Pure appends.
    { 0, 24, 0 },
};

// ---------------------------------------------------------------------------
// Qwen3.5 Gated DeltaNet (linear attention) - sequential recurrence.
//
// One block per head (grid = n_heads); blockDim = head_dim = D (a power of two;
// 128 for qwen3.5). Thread j owns COLUMN j of the per-head state S - the values
// S[i][j] for every key-dim row i - kept in a thread-local array. Then every op
// is thread-local except reading the shared, per-token q,k vectors:
//   decay:   S[i,j] *= g_t
//   u[j]   = sum_i S[i,j] * k_hat[i]        (dot of column j with k_hat)
//   d[j]   = beta_t * (v[j] - u[j])
//   S[i,j] += k_hat[i] * d[j]
//   out[j] = sum_i S[i,j] * (q_hat[i] * scale)
// q,k are L2-normalized (eps 1e-6) and q scaled by 1/sqrt(D) inside, matching
// reference::delta_net::gated_delta_recurrent. state [H][D][D] read + written.
#define PD_DN_MAX_D 128
__global__ void pd_gated_delta_recurrent_kernel(
        const float* __restrict__ q, const float* __restrict__ k,
        const float* __restrict__ v, const float* __restrict__ g,
        const float* __restrict__ beta, float* __restrict__ state,
        float* __restrict__ out, uint32_t n_tokens, uint32_t n_heads, uint32_t D) {
    const uint32_t h = blockIdx.x;
    const uint32_t j = threadIdx.x;               // this thread's value-dim column
    if (h >= n_heads || j >= D) return;

    extern __shared__ float smem[];               // [0,D)=q_hat*scale ; [D,2D)=k_hat ; [2D,2D+2)=l2 sums
    float* q_sh = smem;
    float* k_sh = smem + D;
    float* red  = smem + 2 * D;
    const float scale = rsqrtf((float)D);

    // load this thread's state column j: S[i][j] for all i (row stride D)
    float col[PD_DN_MAX_D];
    float* s_head = state + (size_t)h * D * D;
    for (uint32_t i = 0; i < D; ++i) col[i] = s_head[(size_t)i * D + j];

    for (uint32_t t = 0; t < n_tokens; ++t) {
        const size_t base = ((size_t)t * n_heads + h) * (size_t)D;
        const float qj = q[base + j];
        const float kj = k[base + j];
        const float vj = v[base + j];

        // L2-norm q,k over the head dim: tree-reduce Σq², Σk² (D is a power of two)
        q_sh[j] = qj * qj;
        k_sh[j] = kj * kj;
        __syncthreads();
        for (uint32_t s = D >> 1; s > 0; s >>= 1) {
            if (j < s) { q_sh[j] += q_sh[j + s]; k_sh[j] += k_sh[j + s]; }
            __syncthreads();
        }
        if (j == 0) { red[0] = rsqrtf(q_sh[0] + 1e-6f); red[1] = rsqrtf(k_sh[0] + 1e-6f); }
        __syncthreads();
        q_sh[j] = qj * red[0] * scale;            // q_hat * (1/sqrt D)
        k_sh[j] = kj * red[1];                    // k_hat
        __syncthreads();

        const float g_t = expf(g[(size_t)t * n_heads + h]);
        const float beta_t = beta[(size_t)t * n_heads + h];

        // decay the column, then read u = k_hat^T . S from the decayed state
        float u = 0.0f;
        for (uint32_t i = 0; i < D; ++i) { col[i] *= g_t; u += col[i] * k_sh[i]; }
        const float delta = beta_t * (vj - u);
        // rank-1 update S += k_hat (x) d, then read out = (q_hat*scale)^T . S from the updated state
        float o = 0.0f;
        for (uint32_t i = 0; i < D; ++i) { col[i] += k_sh[i] * delta; o += col[i] * q_sh[i]; }
        out[base + j] = o;
        __syncthreads();                          // q_sh/k_sh reused next token
    }

    for (uint32_t i = 0; i < D; ++i) s_head[(size_t)i * D + j] = col[i];
}

// Kill switch back to the runtime-D kernel, so the twin can be A/B'd in place.
static inline bool pd_dn_rec_generic_env() {
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("PADDOCK_DN_REC_GENERIC");
        v = (e && *e && e[0] != '0') ? 1 : 0;
    }
    return v != 0;
}

// Compile-time-D twin of the kernel above. Identical arithmetic in an
// identical order - the only change is that `D` is a constant, so `col[i]`
// unrolls to compile-time indices and the per-thread state COLUMN stays in
// REGISTERS. With a runtime `D` the array cannot be register-allocated and
// lands in local memory, and at n_tokens=1 the decay and update loops then
// re-stream the whole column four extra times: measured 15.27 us against the
// ~6 us this shape should cost, and against a rival's 5.48 us for the same
// work. Registers make the pass state-read + state-write and nothing else.
template <uint32_t D>
__global__ __launch_bounds__(D) void pd_gated_delta_recurrent_kernel_t(
        const float* __restrict__ q, const float* __restrict__ k,
        const float* __restrict__ v, const float* __restrict__ g,
        const float* __restrict__ beta, float* __restrict__ state,
        float* __restrict__ out, uint32_t n_tokens, uint32_t n_heads) {
    const uint32_t h = blockIdx.x;
    const uint32_t j = threadIdx.x;
    if (h >= n_heads) return;

    extern __shared__ float smem[];
    float* q_sh = smem;
    float* k_sh = smem + D;
    float* red  = smem + 2 * D;
    const float scale = rsqrtf((float)D);

    float col[D];
    float* s_head = state + (size_t)h * D * D;
    #pragma unroll
    for (uint32_t i = 0; i < D; ++i) col[i] = s_head[(size_t)i * D + j];

    for (uint32_t t = 0; t < n_tokens; ++t) {
        const size_t base = ((size_t)t * n_heads + h) * (size_t)D;
        const float qj = q[base + j];
        const float kj = k[base + j];
        const float vj = v[base + j];

        q_sh[j] = qj * qj;
        k_sh[j] = kj * kj;
        __syncthreads();
        for (uint32_t s = D >> 1; s > 0; s >>= 1) {
            if (j < s) { q_sh[j] += q_sh[j + s]; k_sh[j] += k_sh[j + s]; }
            __syncthreads();
        }
        if (j == 0) { red[0] = rsqrtf(q_sh[0] + 1e-6f); red[1] = rsqrtf(k_sh[0] + 1e-6f); }
        __syncthreads();
        q_sh[j] = qj * red[0] * scale;
        k_sh[j] = kj * red[1];
        __syncthreads();

        const float g_t = expf(g[(size_t)t * n_heads + h]);
        const float beta_t = beta[(size_t)t * n_heads + h];

        float u = 0.0f;
        #pragma unroll
        for (uint32_t i = 0; i < D; ++i) { col[i] *= g_t; u += col[i] * k_sh[i]; }
        const float delta = beta_t * (vj - u);
        float o = 0.0f;
        #pragma unroll
        for (uint32_t i = 0; i < D; ++i) { col[i] += k_sh[i] * delta; o += col[i] * q_sh[i]; }
        out[base + j] = o;
        __syncthreads();
    }

    #pragma unroll
    for (uint32_t i = 0; i < D; ++i) s_head[(size_t)i * D + j] = col[i];
}

// Single-sequence walk, P-SPLIT + PRE-NORMED (slot 596's kernel). Three
// things happen here that the shipped walk could not do at once:
//
//  * P threads share a state COLUMN, each owning D/P of its rows. The columns
//    of the delta rule never interact - (I - beta k k^T) mixes ROWS, and a
//    column needs only its own rows plus the head's (q, k) - so the split
//    costs nothing across blocks, just a P-lane butterfly for the two dots.
//    The unsplit walk held a whole column in registers: ncu measured 255
//    registers a thread, a block limit of TWO an SM, 12.57% occupancy, with
//    n_heads = 48 blocks on a 148-SM die.
//  * the norms come PRE-COMPUTED in `rn` (the companion pass below), where
//    the shipped walk ran two shared trees a token - ~15 barriers.
//  * with only D/P rows to serve, a thread reads its own q and k straight
//    from global instead of staging the head's vectors in shared, so the
//    token loop has NO barrier at all. A warp's lanes cover one column's
//    rows contiguously, so those reads coalesce.
//
// Measured on Flash-Next IQ3_XXS at a 2114-row prefill, ms a layer: 3.85
// unsplit, 3.48 at P=4, 3.03 at P=16 (the split alone stops paying because
// every block redid the trees), and this form at P=16.
//
// The norms are bit-identical to the tree (same order, same pass). What
// re-associates is the two dots - a P-lane butterfly where one thread summed
// D terms - which is the wave-prefill class this lane already carries; the
// decode entries below keep the unsplit order.
template <uint32_t D, uint32_t P>
__global__ __launch_bounds__(128) void pd_gated_delta_recurrent_pn_kernel_t(
        const float* __restrict__ q, const float* __restrict__ k,
        const float* __restrict__ v, const float* __restrict__ g,
        const float* __restrict__ beta, float* __restrict__ state,
        float* __restrict__ out, const float* __restrict__ rn,
        uint32_t n_tokens, uint32_t n_heads) {
    constexpr uint32_t C = 128u / P;      // state columns a block owns
    constexpr uint32_t R = D / P;         // state rows a thread owns
    constexpr uint32_t SPLITS = D / C;    // blocks a head takes
    const uint32_t h = blockIdx.x / SPLITS;
    const uint32_t spl = blockIdx.x % SPLITS;
    if (h >= n_heads) return;
    const uint32_t tid = threadIdx.x;
    const uint32_t cl = tid / P, pp = tid % P;
    const uint32_t j = spl * C + cl;      // this thread's state column
    const uint32_t i0 = pp * R;           // its first state row
    const float scale = rsqrtf((float)D);

    float col[R];
    float* s_head = state + (size_t)h * D * D;
    #pragma unroll
    for (uint32_t ii = 0; ii < R; ++ii) col[ii] = s_head[(size_t)(i0 + ii) * D + j];

    for (uint32_t t = 0; t < n_tokens; ++t) {
        const size_t base = ((size_t)t * n_heads + h) * (size_t)D;
        const float r0 = rn[((size_t)t * n_heads + h) * 2u];
        const float r1 = rn[((size_t)t * n_heads + h) * 2u + 1u];
        float qv[R], kv[R];
        #pragma unroll
        for (uint32_t ii = 0; ii < R; ++ii) {
            qv[ii] = q[base + i0 + ii] * r0 * scale;
            kv[ii] = k[base + i0 + ii] * r1;
        }
        const float vj = v[base + j];
        const float g_t = expf(g[(size_t)t * n_heads + h]);
        const float beta_t = beta[(size_t)t * n_heads + h];

        // four accumulators: the dot is a dependent f32 chain ptxas cannot
        // break, and a single chain of D terms is pure latency per token
        float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;
        #pragma unroll
        for (uint32_t ii = 0; ii < R; ii += 4u) {
            col[ii] *= g_t; col[ii + 1u] *= g_t; col[ii + 2u] *= g_t; col[ii + 3u] *= g_t;
            u0 += col[ii] * kv[ii];
            u1 += col[ii + 1u] * kv[ii + 1u];
            u2 += col[ii + 2u] * kv[ii + 2u];
            u3 += col[ii + 3u] * kv[ii + 3u];
        }
        float u = (u0 + u1) + (u2 + u3);
        #pragma unroll
        for (uint32_t m = 1; m < P; m <<= 1) u += __shfl_xor_sync(0xffffffffu, u, m);
        const float delta = beta_t * (vj - u);

        float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
        #pragma unroll
        for (uint32_t ii = 0; ii < R; ii += 4u) {
            col[ii] += kv[ii] * delta;
            col[ii + 1u] += kv[ii + 1u] * delta;
            col[ii + 2u] += kv[ii + 2u] * delta;
            col[ii + 3u] += kv[ii + 3u] * delta;
            o0 += col[ii] * qv[ii];
            o1 += col[ii + 1u] * qv[ii + 1u];
            o2 += col[ii + 2u] * qv[ii + 2u];
            o3 += col[ii + 3u] * qv[ii + 3u];
        }
        float o = (o0 + o1) + (o2 + o3);
        #pragma unroll
        for (uint32_t m = 1; m < P; m <<= 1) o += __shfl_xor_sync(0xffffffffu, o, m);
        if (pp == 0u) out[base + j] = o;
    }

    #pragma unroll
    for (uint32_t ii = 0; ii < R; ++ii) s_head[(size_t)(i0 + ii) * D + j] = col[ii];
}

// PRE-NORM pass (slot 541 companion): the walk's two per-token tree
// reductions are over PER-TOKEN data - nothing in them is serial - and they
// cost the walk most of its ~17 __syncthreads per token (the load-prefetch
// arm measured NULL, so barriers are the wall). This kernel runs the
// identical 128-thread shared tree in the identical order, one block per
// (token, head), so the scalars are BIT-IDENTICAL to the in-walk form and
// the walk below can skip both trees.
template <uint32_t D>
__global__ __launch_bounds__(D) void pd_gdn_qk_rnorm_kernel(
        const float* __restrict__ q, const float* __restrict__ k,
        float* __restrict__ rn, uint32_t n_heads) {
    const uint32_t t = blockIdx.x, h = blockIdx.y, j = threadIdx.x;
    extern __shared__ float smem[];
    float* q_sh = smem;
    float* k_sh = smem + D;
    const size_t base = ((size_t)t * n_heads + h) * (size_t)D;
    const float qj = q[base + j];
    const float kj = k[base + j];
    q_sh[j] = qj * qj;
    k_sh[j] = kj * kj;
    __syncthreads();
    for (uint32_t s = D >> 1; s > 0; s >>= 1) {
        if (j < s) { q_sh[j] += q_sh[j + s]; k_sh[j] += k_sh[j + s]; }
        __syncthreads();
    }
    if (j == 0) {
        rn[((size_t)t * n_heads + h) * 2u] = rsqrtf(q_sh[0] + 1e-6f);
        rn[((size_t)t * n_heads + h) * 2u + 1u] = rsqrtf(k_sh[0] + 1e-6f);
    }
}

// Runs walk, pre-normed twin: reads the scalars, keeps one staging barrier
// per token where the tree form paid ~15. The norms are the same bits.
//
// P-SPLIT: `P` threads share a state COLUMN, each owning D/P of its rows.
// The columns of the delta rule never interact - (I - beta k k^T) mixes ROWS,
// and a column's update needs only that column's own rows plus the shared
// (q, k) vectors - so the split needs no cross-block anything, just a P-lane
// butterfly for the two dots. It exists because the unsplit walk holds the
// whole column in registers: ncu says 255 registers a thread, a block limit
// of TWO an SM and 12.57% achieved occupancy, with 48 blocks on a 148-SM die
// and nothing resident to hide the serial chain. At P=4 the state costs D/4
// registers, the grid is 4x the blocks, and the per-token chain is a quarter
// as long.
//
// Thread layout is column-major (tid = c * P + p) so a column's P threads are
// consecutive LANES and the butterfly is __shfl_xor, never shared memory.
// The reduction across those lanes is a reassociation of the dot the unsplit
// walk did in one thread - the wave prefill's own class, and the decode
// entries below keep the unsplit order.
template <uint32_t D, uint32_t P>
__global__ __launch_bounds__(128) void pd_gated_delta_recurrent_runs_pn_kernel_t(
        const float* __restrict__ q, const float* __restrict__ k,
        const float* __restrict__ v, const float* __restrict__ g,
        const float* __restrict__ beta, float* __restrict__ state,
        float* __restrict__ out, const unsigned int* __restrict__ run_off,
        const unsigned int* __restrict__ run_len,
        const unsigned int* __restrict__ run_slot,
        const float* __restrict__ rn, uint32_t n_heads) {
    constexpr uint32_t C = 128u / P;        // state columns a block owns
    constexpr uint32_t R = D / P;           // state rows a thread owns
    constexpr uint32_t SPLITS = D / C;      // blocks a head takes
    const uint32_t h = blockIdx.x / SPLITS;
    const uint32_t sp = blockIdx.x % SPLITS;
    const uint32_t r_run = blockIdx.y;
    if (h >= n_heads) return;
    const uint32_t off = run_off[r_run];
    const uint32_t len = run_len[r_run];
    if (len == 0) return;

    const uint32_t tid = threadIdx.x;
    const uint32_t cl = tid / P;            // column inside the block
    const uint32_t pp = tid % P;            // which row slice
    const uint32_t j = sp * C + cl;         // the state column this thread serves
    const uint32_t i0 = pp * R;             // its first state row

    extern __shared__ float smem[];
    float* q_sh = smem;
    float* k_sh = smem + D;
    const float scale = rsqrtf((float)D);

    float col[R];
    float* s_head = state
        + ((size_t)run_slot[r_run] * (size_t)n_heads + (size_t)h) * (size_t)D * (size_t)D;
    #pragma unroll
    for (uint32_t ii = 0; ii < R; ++ii) col[ii] = s_head[(size_t)(i0 + ii) * D + j];

    for (uint32_t t = 0; t < len; ++t) {
        const size_t base = ((size_t)(off + t) * n_heads + h) * (size_t)D;
        // stage the head's (q, k) once a token: every column reads all D rows
        for (uint32_t i = tid; i < D; i += 128u) {
            const float r0 = rn[((size_t)(off + t) * n_heads + h) * 2u];
            const float r1 = rn[((size_t)(off + t) * n_heads + h) * 2u + 1u];
            q_sh[i] = q[base + i] * r0 * scale;
            k_sh[i] = k[base + i] * r1;
        }
        __syncthreads();

        const float vj = v[base + j];
        const float g_t = expf(g[(size_t)(off + t) * n_heads + h]);
        const float beta_t = beta[(size_t)(off + t) * n_heads + h];

        // four accumulators: `u += col[i] * k[i]` is a dependent f32 chain and
        // ptxas cannot break it (f32 add is not associative)
        float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;
        #pragma unroll
        for (uint32_t ii = 0; ii < R; ii += 4u) {
            col[ii] *= g_t; col[ii + 1u] *= g_t; col[ii + 2u] *= g_t; col[ii + 3u] *= g_t;
            u0 += col[ii] * k_sh[i0 + ii];
            u1 += col[ii + 1u] * k_sh[i0 + ii + 1u];
            u2 += col[ii + 2u] * k_sh[i0 + ii + 2u];
            u3 += col[ii + 3u] * k_sh[i0 + ii + 3u];
        }
        float u = (u0 + u1) + (u2 + u3);
        #pragma unroll
        for (uint32_t m = 1; m < P; m <<= 1) u += __shfl_xor_sync(0xffffffffu, u, m);
        const float delta = beta_t * (vj - u);

        float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
        #pragma unroll
        for (uint32_t ii = 0; ii < R; ii += 4u) {
            col[ii] += k_sh[i0 + ii] * delta;
            col[ii + 1u] += k_sh[i0 + ii + 1u] * delta;
            col[ii + 2u] += k_sh[i0 + ii + 2u] * delta;
            col[ii + 3u] += k_sh[i0 + ii + 3u] * delta;
            o0 += col[ii] * q_sh[i0 + ii];
            o1 += col[ii + 1u] * q_sh[i0 + ii + 1u];
            o2 += col[ii + 2u] * q_sh[i0 + ii + 2u];
            o3 += col[ii + 3u] * q_sh[i0 + ii + 3u];
        }
        float o = (o0 + o1) + (o2 + o3);
        #pragma unroll
        for (uint32_t m = 1; m < P; m <<= 1) o += __shfl_xor_sync(0xffffffffu, o, m);
        if (pp == 0u) out[base + j] = o;
        __syncthreads();
    }

    #pragma unroll
    for (uint32_t ii = 0; ii < R; ++ii) s_head[(size_t)(i0 + ii) * D + j] = col[ii];
}

// Runs twin (slot 534): `n_runs` INDEPENDENT sequences in one launch, each
// against its own slot's carried state. grid (n_heads, n_runs).
//
// The single-run kernel above grids `n_heads` = 48 blocks at 255 registers on
// a 148-SM die - 32% of the die, one block per SM - and a 128-token prefill
// spends 195.9 us per layer in it (7.05 ms of a 33.9 ms prefill, decode-cadence
// nsys 2026-08-28). Serially prefilling a c32 admission wave pays that 32
// times over; batching the runs fills the die instead, and the arithmetic per
// run is byte-identical because each block still walks its own sequence in
// order.
template <uint32_t D>
__global__ __launch_bounds__(D) void pd_gated_delta_recurrent_runs_kernel_t(
        const float* __restrict__ q, const float* __restrict__ k,
        const float* __restrict__ v, const float* __restrict__ g,
        const float* __restrict__ beta, float* __restrict__ state,
        float* __restrict__ out, const unsigned int* __restrict__ run_off,
        const unsigned int* __restrict__ run_len,
        const unsigned int* __restrict__ run_slot, uint32_t n_heads) {
    const uint32_t h = blockIdx.x;
    const uint32_t r = blockIdx.y;
    const uint32_t j = threadIdx.x;
    if (h >= n_heads) return;
    const uint32_t off = run_off[r];
    const uint32_t len = run_len[r];
    if (len == 0) return;

    extern __shared__ float smem[];
    float* q_sh = smem;
    float* k_sh = smem + D;
    float* red  = smem + 2 * D;
    const float scale = rsqrtf((float)D);

    float col[D];
    float* s_head = state
        + ((size_t)run_slot[r] * (size_t)n_heads + (size_t)h) * (size_t)D * (size_t)D;
    #pragma unroll
    for (uint32_t i = 0; i < D; ++i) col[i] = s_head[(size_t)i * D + j];

    // Software pipeline: token t+1's five loads issue before token t's
    // compute, so the serial walk pays one global latency at the top instead
    // of one per token (the walk is 6.5 us/token and its arithmetic is worth
    // ~1 us -- the rest is exposed load latency; rival's fla walk does the
    // same rows in ~46 ns/token). BIT-EXACT: identical arithmetic in the
    // identical order; only the load ISSUE moves.
    size_t base = ((size_t)off * n_heads + h) * (size_t)D;
    float qj = q[base + j], kj = k[base + j], vj = v[base + j];
    float gj = g[(size_t)off * n_heads + h];
    float bj = beta[(size_t)off * n_heads + h];
    for (uint32_t t = 0; t < len; ++t) {
        float qn = 0.0f, kn = 0.0f, vn = 0.0f, gn = 0.0f, bn = 0.0f;
        if (t + 1 < len) {
            const size_t nb = ((size_t)(off + t + 1) * n_heads + h) * (size_t)D;
            qn = q[nb + j]; kn = k[nb + j]; vn = v[nb + j];
            gn = g[(size_t)(off + t + 1) * n_heads + h];
            bn = beta[(size_t)(off + t + 1) * n_heads + h];
        }

        q_sh[j] = qj * qj;
        k_sh[j] = kj * kj;
        __syncthreads();
        for (uint32_t s = D >> 1; s > 0; s >>= 1) {
            if (j < s) { q_sh[j] += q_sh[j + s]; k_sh[j] += k_sh[j + s]; }
            __syncthreads();
        }
        if (j == 0) { red[0] = rsqrtf(q_sh[0] + 1e-6f); red[1] = rsqrtf(k_sh[0] + 1e-6f); }
        __syncthreads();
        q_sh[j] = qj * red[0] * scale;
        k_sh[j] = kj * red[1];
        __syncthreads();

        const float g_t = expf(gj);
        const float beta_t = bj;

        float u = 0.0f;
        #pragma unroll
        for (uint32_t i = 0; i < D; ++i) { col[i] *= g_t; u += col[i] * k_sh[i]; }
        const float delta = beta_t * (vj - u);
        float o = 0.0f;
        #pragma unroll
        for (uint32_t i = 0; i < D; ++i) { col[i] += k_sh[i] * delta; o += col[i] * q_sh[i]; }
        out[base + j] = o;
        base = ((size_t)(off + t + 1) * n_heads + h) * (size_t)D;
        qj = qn; kj = kn; vj = vn; gj = gn; bj = bn;
        __syncthreads();
    }

    #pragma unroll
    for (uint32_t i = 0; i < D; ++i) s_head[(size_t)i * D + j] = col[i];
}

PD_EXPORT
int pd_gated_delta_recurrent_runs(const void* q, const void* k, const void* v,
                                  const void* g, const void* beta, void* state,
                                  void* out, const void* run_off,
                                  const void* run_len, const void* run_slot,
                                  uint32_t n_runs, uint32_t n_heads,
                                  uint32_t head_dim, void* stream) {
    if (n_runs == 0 || n_heads == 0 || head_dim == 0) return 0;
    if (head_dim > PD_DN_MAX_D) return cudaErrorInvalidValue;
    const size_t shmem = ((size_t)2 * head_dim + 2) * sizeof(float);
    const dim3 grid(n_heads, n_runs);
#define PD_DN_RUNS_T(DD)                                                      \
    do {                                                                      \
        pd_gated_delta_recurrent_runs_kernel_t<DD>                            \
            <<<grid, (DD), shmem, (cudaStream_t)stream>>>(                    \
                (const float*)q, (const float*)k, (const float*)v,            \
                (const float*)g, (const float*)beta, (float*)state,           \
                (float*)out, (const unsigned int*)run_off,                    \
                (const unsigned int*)run_len, (const unsigned int*)run_slot,  \
                n_heads);                                                     \
        return pd_launch_status();                                            \
    } while (0)
    if (head_dim == 128u) PD_DN_RUNS_T(128u);
    if (head_dim == 64u) PD_DN_RUNS_T(64u);
#undef PD_DN_RUNS_T
    // no runtime-D twin: the register-resident column is the whole point
    return cudaErrorInvalidValue;
}

// slot 541: pre-normed runs walk. `rn` is CALLER-OWNED scratch for
// n_tokens*n_heads*2 f32 (address-stable by the usual capture contract);
// n_tokens must cover every run's off+len.
PD_EXPORT
int pd_gated_delta_recurrent_runs_pn(const void* q, const void* k, const void* v,
                                     const void* g, const void* beta, void* state,
                                     void* out, const void* run_off,
                                     const void* run_len, const void* run_slot,
                                     uint32_t n_runs, uint32_t n_tokens,
                                     uint32_t n_heads, uint32_t head_dim,
                                     void* rn, void* stream) {
    if (n_runs == 0 || n_heads == 0 || head_dim == 0) return 0;
    if (head_dim > PD_DN_MAX_D) return cudaErrorInvalidValue;
    const size_t shmem = ((size_t)2 * head_dim) * sizeof(float);
    const dim3 ngrid(n_tokens, n_heads);
    // P is the row split (see the kernel note): the walk is register-bound,
    // so a head takes D/(128/P) blocks of 128 threads instead of one block of
    // D threads holding whole columns.
#define PD_DN_RUNS_PN_T(DD, PP)                                               \
    do {                                                                      \
        constexpr uint32_t C_ = 128u / (PP);                                  \
        const dim3 wgrid(n_heads * ((DD) / C_), n_runs);                      \
        pd_gdn_qk_rnorm_kernel<DD>                                            \
            <<<ngrid, (DD), shmem, (cudaStream_t)stream>>>(                   \
                (const float*)q, (const float*)k, (float*)rn, n_heads);       \
        pd_gated_delta_recurrent_runs_pn_kernel_t<DD, PP>                     \
            <<<wgrid, 128u, shmem, (cudaStream_t)stream>>>(                   \
                (const float*)q, (const float*)k, (const float*)v,            \
                (const float*)g, (const float*)beta, (float*)state,           \
                (float*)out, (const unsigned int*)run_off,                    \
                (const unsigned int*)run_len, (const unsigned int*)run_slot,  \
                (const float*)rn, n_heads);                                   \
        return pd_launch_status();                                            \
    } while (0)
    if (head_dim == 128u) PD_DN_RUNS_PN_T(128u, 4u);
    if (head_dim == 64u) PD_DN_RUNS_PN_T(64u, 2u);
#undef PD_DN_RUNS_PN_T
    return cudaErrorInvalidValue;
}

PD_EXPORT
int pd_gated_delta_recurrent(const void* q, const void* k, const void* v, const void* g,
                             const void* beta, void* state, void* out, uint32_t n_tokens,
                             uint32_t n_heads, uint32_t head_dim, void* stream) {
    // the legacy sequential kernel has no narrow-state variant (no engine
    // caller on the qwen35 serving paths) - fail loud rather than misread
    if (pd_dns_nonf32_env()) return cudaErrorInvalidValue;
    if (n_tokens == 0 || n_heads == 0 || head_dim == 0) return 0;
    if (head_dim > PD_DN_MAX_D) return cudaErrorInvalidValue;
    size_t shmem = ((size_t)2 * head_dim + 2) * sizeof(float);
#define PD_DN_REC_T(DD)                                                       \
    do {                                                                      \
        pd_gated_delta_recurrent_kernel_t<DD>                                 \
            <<<n_heads, (DD), shmem, (cudaStream_t)stream>>>(                 \
                (const float*)q, (const float*)k, (const float*)v,            \
                (const float*)g, (const float*)beta, (float*)state,           \
                (float*)out, n_tokens, n_heads);                              \
        return pd_launch_status();                                            \
    } while (0)
    if (!pd_dn_rec_generic_env()) {
        if (head_dim == 128u) PD_DN_REC_T(128u);
        if (head_dim == 64u) PD_DN_REC_T(64u);
    }
#undef PD_DN_REC_T
    pd_gated_delta_recurrent_kernel<<<n_heads, head_dim, shmem, (cudaStream_t)stream>>>(
        (const float*)q, (const float*)k, (const float*)v, (const float*)g,
        (const float*)beta, (float*)state, (float*)out, n_tokens, n_heads, head_dim);
    return pd_launch_status();
}

// slot 596: the single-sequence walk, P-SPLIT and PRE-NORMED. Same arguments
// as slot 533 plus caller-owned `rn` scratch (n_tokens * n_heads * 2 floats,
// what the runs_pn entry already sizes). The norms are the tree's bits; the
// two dots re-associate across the split lanes, so this is the wave-prefill
// class - the legacy entry above is untouched for every other caller.
PD_EXPORT
int pd_gated_delta_recurrent_pn(const void* q, const void* k, const void* v,
                                const void* g, const void* beta, void* state,
                                void* out, uint32_t n_tokens, uint32_t n_heads,
                                uint32_t head_dim, void* rn, void* stream) {
    if (pd_dns_nonf32_env()) return cudaErrorInvalidValue;
    if (n_tokens == 0 || n_heads == 0 || head_dim == 0) return 0;
    if (head_dim > PD_DN_MAX_D) return cudaErrorInvalidValue;
    if (rn == nullptr) return cudaErrorInvalidValue;
#define PD_DN_REC_PN_T(DD, PP)                                                \
    do {                                                                      \
        constexpr uint32_t C_ = 128u / (PP);                                  \
        const dim3 ngrid_(n_tokens, n_heads);                                 \
        pd_gdn_qk_rnorm_kernel<DD>                                            \
            <<<ngrid_, (DD), (size_t)2 * (DD) * sizeof(float),                \
               (cudaStream_t)stream>>>(                                       \
                (const float*)q, (const float*)k, (float*)rn, n_heads);       \
        pd_gated_delta_recurrent_pn_kernel_t<DD, PP>                          \
            <<<n_heads * ((DD) / C_), 128u, 0u, (cudaStream_t)stream>>>(      \
                (const float*)q, (const float*)k, (const float*)v,            \
                (const float*)g, (const float*)beta, (float*)state,           \
                (float*)out, (const float*)rn, n_tokens, n_heads);            \
        return pd_launch_status();                                            \
    } while (0)
    if (head_dim == 128u) PD_DN_REC_PN_T(128u, 16u);
    if (head_dim == 64u) PD_DN_REC_PN_T(64u, 8u);
#undef PD_DN_REC_PN_T
    return cudaErrorInvalidValue;
}

// Depthwise causal conv1d (kernel k) + SiLU - DeltaNet input conv. One thread per
// (t,c) output; zero left-padding. x [T,conv_dim], w [conv_dim,k] (w[c*k+kk]).
__global__ void pd_causal_conv1d_silu_kernel(
        const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ out,
        uint32_t n_tokens, uint32_t conv_dim, uint32_t k) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)n_tokens * conv_dim;
    if (idx >= total) return;
    uint32_t t = (uint32_t)(idx / conv_dim);
    uint32_t c = (uint32_t)(idx % conv_dim);
    float acc = 0.0f;
    for (uint32_t kk = 0; kk < k; ++kk) {
        int64_t ti = (int64_t)t - (int64_t)(k - 1) + (int64_t)kk;
        if (ti >= 0) acc += w[(size_t)c * k + kk] * x[(uint64_t)ti * conv_dim + c];
    }
    out[idx] = acc / (1.0f + expf(-acc));   // SiLU
}

PD_EXPORT
int pd_causal_conv1d_silu(const void* x, const void* w, void* out,
                          uint32_t n_tokens, uint32_t conv_dim, uint32_t k, void* stream) {
    uint64_t total = (uint64_t)n_tokens * conv_dim;
    if (total == 0) return 0;
    uint32_t threads = 256;
    uint64_t blocks = (total + threads - 1) / threads;
    pd_causal_conv1d_silu_kernel<<<(uint32_t)blocks, threads, 0, (cudaStream_t)stream>>>(
        (const float*)x, (const float*)w, (float*)out, n_tokens, conv_dim, k);
    return pd_launch_status();
}

// FUSED conv1d+SiLU+split+GQA+q/k-norm (glue round 2): the
// prefill chain ran conv (write [T,conv_dim]) then split_gqa_norm (read it
// back, write q/k/v) - a full d_conv round-trip per DN layer. This kernel
// mirrors the split's (n_v_heads, rows) x s geometry and computes the three
// conv outputs it needs inline (ascending-tap FMA, ti >= 0 guard relative
// to the span base - the _at zero-pad convention), SiLUs them, then runs
// the split kernel's exact reduction and output expressions - bit-exact
// composition, d_conv never materializes. GQA q/k conv work is recomputed
// per sharing v-head (3x, four taps - cheap; x rows come from L2).
// Only valid under the _at fresh-prompt semantics (zero window before the
// span base); the non-offset path keeps the separate kernels.
template <typename OT = float>
__global__ void pd_causal_conv1d_silu_qkv_kernel(
        const float* __restrict__ x, const float* __restrict__ w,
        OT* __restrict__ q_out, OT* __restrict__ k_out,
        OT* __restrict__ v_out, uint32_t n_k_heads, uint32_t n_v_heads,
        uint32_t s, uint32_t k) {
    const uint32_t hv = blockIdx.x;
    const uint32_t row = blockIdx.y;
    const uint32_t j = threadIdx.x;
    if (j >= s) return;
    const uint32_t hk = hv % n_k_heads;
    const uint32_t key_dim = s * n_k_heads;
    const uint32_t conv_dim = 2u * key_dim + s * n_v_heads;
    const uint32_t cq = hk * s + j;
    const uint32_t ck = key_dim + hk * s + j;
    const uint32_t cv = 2u * key_dim + hv * s + j;
    float aq = 0.0f, ak = 0.0f, av = 0.0f;
    for (uint32_t kk = 0; kk < k; ++kk) {
        int64_t ti = (int64_t)row - (int64_t)(k - 1) + (int64_t)kk;
        if (ti >= 0) {
            const float* xr = x + (uint64_t)ti * conv_dim;
            aq += w[(size_t)cq * k + kk] * xr[cq];
            ak += w[(size_t)ck * k + kk] * xr[ck];
            av += w[(size_t)cv * k + kk] * xr[cv];
        }
    }
    const float qj = aq / (1.0f + expf(-aq));
    const float kj = ak / (1.0f + expf(-ak));
    const float vj = av / (1.0f + expf(-av));

    float q2 = qj * qj, k2 = kj * kj;
#pragma unroll
    for (uint32_t off = 16; off > 0; off >>= 1) {
        q2 += __shfl_xor_sync(0xffffffffu, q2, off);
        k2 += __shfl_xor_sync(0xffffffffu, k2, off);
    }
    __shared__ float sh[8];
    const uint32_t lane = j & 31u, warp = j >> 5, nwarps = (s + 31u) >> 5;
    if (lane == 0) { sh[warp] = q2; sh[4 + warp] = k2; }
    __syncthreads();
    float qs = 0.0f, ks = 0.0f;
    for (uint32_t ww = 0; ww < nwarps; ++ww) { qs += sh[ww]; ks += sh[4 + ww]; }

    const size_t oidx = ((size_t)row * n_v_heads + hv) * s + j;
    q_out[oidx] = (OT)(qj * rsqrtf(qs + 1e-6f) * rsqrtf((float)s));
    k_out[oidx] = (OT)(kj * rsqrtf(ks + 1e-6f));
    v_out[oidx] = (OT)vj;
}

// Compute-once twin (conv rung): the kernel above owns
// one (v-head, row) per block, so the GQA-shared q/k convolutions AND the
// q/k norm reductions run once per sharing v-head (3x for qwen3.6's 48v/16k).
// This grid computes every conv channel exactly once: blocks [0, n_k_heads)
// own one k-head's q+k channels - conv, SiLU, the same dual warp reduction -
// and write the normalized q/k to every sharing v-head slot (values are
// identical across sharers by construction, so the expanded [rows,HV,s]
// output layout the consumers read is unchanged); blocks [n_k_heads,
// n_k_heads+n_v_heads) own one v-head's channels, conv+SiLU only. FMA order
// per accumulator, the shfl/smem reduction shape, and every output
// expression are copied verbatim from the kernel above - bit-exact
// composition, verified against the unfused chain. The k==4 row>=3
// specialization loads each channel's taps as one float4 (w rows are
// 16B-aligned at k==4) and drops the per-tap ti>=0 guard (uniform branch);
// edge rows keep the guarded ascending scalar loop.
template <typename VT = float>
__global__ void pd_causal_conv1d_silu_qkv_once_kernel(
        const float* __restrict__ x, const float* __restrict__ w,
        float* __restrict__ q_out, float* __restrict__ k_out,
        VT* __restrict__ v_out, uint32_t n_k_heads, uint32_t n_v_heads,
        uint32_t s, uint32_t k) {
    const uint32_t row = blockIdx.y;
    const uint32_t j = threadIdx.x;
    if (j >= s) return;
    const uint32_t key_dim = s * n_k_heads;
    const uint32_t conv_dim = 2u * key_dim + s * n_v_heads;
    if (blockIdx.x >= n_k_heads) {
        const uint32_t hv = blockIdx.x - n_k_heads;
        const uint32_t cv = 2u * key_dim + hv * s + j;
        float av = 0.0f;
        if (k == 4u && row >= 3u) {
            const float4 wv = *reinterpret_cast<const float4*>(w + (size_t)cv * 4u);
            const float* xc = x + (uint64_t)(row - 3u) * conv_dim + cv;
            av += wv.x * xc[0];
            av += wv.y * xc[conv_dim];
            av += wv.z * xc[2u * (size_t)conv_dim];
            av += wv.w * xc[3u * (size_t)conv_dim];
        } else {
            for (uint32_t kk = 0; kk < k; ++kk) {
                int64_t ti = (int64_t)row - (int64_t)(k - 1) + (int64_t)kk;
                if (ti >= 0) av += w[(size_t)cv * k + kk] * x[(uint64_t)ti * conv_dim + cv];
            }
        }
        const float vj = av / (1.0f + expf(-av));
        v_out[((size_t)row * n_v_heads + hv) * s + j] = (VT)vj;
        return;
    }
    const uint32_t hk = blockIdx.x;
    const uint32_t cq = hk * s + j;
    const uint32_t ck = key_dim + hk * s + j;
    float aq = 0.0f, ak = 0.0f;
    if (k == 4u && row >= 3u) {
        const float4 wq = *reinterpret_cast<const float4*>(w + (size_t)cq * 4u);
        const float4 wk = *reinterpret_cast<const float4*>(w + (size_t)ck * 4u);
        const float* xq = x + (uint64_t)(row - 3u) * conv_dim + cq;
        const float* xk = x + (uint64_t)(row - 3u) * conv_dim + ck;
        aq += wq.x * xq[0];
        ak += wk.x * xk[0];
        aq += wq.y * xq[conv_dim];
        ak += wk.y * xk[conv_dim];
        aq += wq.z * xq[2u * (size_t)conv_dim];
        ak += wk.z * xk[2u * (size_t)conv_dim];
        aq += wq.w * xq[3u * (size_t)conv_dim];
        ak += wk.w * xk[3u * (size_t)conv_dim];
    } else {
        for (uint32_t kk = 0; kk < k; ++kk) {
            int64_t ti = (int64_t)row - (int64_t)(k - 1) + (int64_t)kk;
            if (ti >= 0) {
                const float* xr = x + (uint64_t)ti * conv_dim;
                aq += w[(size_t)cq * k + kk] * xr[cq];
                ak += w[(size_t)ck * k + kk] * xr[ck];
            }
        }
    }
    const float qj = aq / (1.0f + expf(-aq));
    const float kj = ak / (1.0f + expf(-ak));
    float q2 = qj * qj, k2 = kj * kj;
#pragma unroll
    for (uint32_t off = 16; off > 0; off >>= 1) {
        q2 += __shfl_xor_sync(0xffffffffu, q2, off);
        k2 += __shfl_xor_sync(0xffffffffu, k2, off);
    }
    __shared__ float sh[8];
    const uint32_t lane = j & 31u, warp = j >> 5, nwarps = (s + 31u) >> 5;
    if (lane == 0) { sh[warp] = q2; sh[4 + warp] = k2; }
    __syncthreads();
    float qs = 0.0f, ks = 0.0f;
    for (uint32_t ww = 0; ww < nwarps; ++ww) { qs += sh[ww]; ks += sh[4 + ww]; }
    const float qn = qj * rsqrtf(qs + 1e-6f) * rsqrtf((float)s);
    const float kn = kj * rsqrtf(ks + 1e-6f);
    for (uint32_t hv = hk; hv < n_v_heads; hv += n_k_heads) {
        const size_t oidx = ((size_t)row * n_v_heads + hv) * s + j;
        q_out[oidx] = qn;
        k_out[oidx] = kn;
    }
}

// kill-switch for the compute-once conv grid (default on; =0 reverts to the
// per-v-head kernels above for A/B) - process-latched like the DNC elections.
static inline bool pd_conv_qkv_once_enabled() {
    static const bool on = [] {
        const char* e = pd_env("PADDOCK_CONV_QKV_ONCE");
        return !(e && atoi(e) == 0);
    }();
    return on;
}

PD_EXPORT
int pd_causal_conv1d_silu_qkv(const void* x, const void* w, void* q_out,
                              void* k_out, void* v_out, uint32_t n_rows,
                              uint32_t n_k_heads, uint32_t n_v_heads,
                              uint32_t s, uint32_t k, void* stream) {
    if (n_rows == 0 || n_v_heads == 0 || s == 0) return 0;
    if (s > 128 || (s & 31u)) return cudaErrorInvalidValue;
    if (pd_conv_qkv_once_enabled()) {
        if (pd_env("PADDOCK_ROUTE_WITNESS")) {
            static bool once = false;
            if (!once) { fprintf(stderr, "pd route: conv1d_silu_qkv fused (once)\n"); once = true; }
        }
        dim3 grid(n_k_heads + n_v_heads, n_rows);
        pd_causal_conv1d_silu_qkv_once_kernel<float><<<grid, s, 0, (cudaStream_t)stream>>>(
            (const float*)x, (const float*)w, (float*)q_out, (float*)k_out,
            (float*)v_out, n_k_heads, n_v_heads, s, k);
        return pd_launch_status();
    }
    if (pd_env("PADDOCK_ROUTE_WITNESS")) {
        static bool once = false;
        if (!once) { fprintf(stderr, "pd route: conv1d_silu_qkv fused\n"); once = true; }
    }
    dim3 grid(n_v_heads, n_rows);
    pd_causal_conv1d_silu_qkv_kernel<float><<<grid, s, 0, (cudaStream_t)stream>>>(
        (const float*)x, (const float*)w, (float*)q_out, (float*)k_out,
        (float*)v_out, n_k_heads, n_v_heads, s, k);
    return pd_launch_status();
}

// VL twin of the per-span offset conv+silu+qkv launches (the c16
// admission wave ran 16 back-to-back per-span launches per DN layer -
// 768/wave plus inter-launch gaps). row0s is a per-ROW u32 plane holding
// each row's span start; the causal window gates on it (`ti >= row0`
// replaces the per-span kernel's `ti >= 0` at x offset row0), so each
// row's adds are identical in value and order to its per-span launch -
// bit-exact by construction. Fresh spans only (zero conv window); the
// resumed ext build keeps the copy path. Float arm only (the vb16 arm
// keeps per-span).
// QKC (compact-bf16 q/k): the expanded [rows, HV, s] f32 q/k
// planes carry 12x the necessary bytes (3x GQA copies x 2x dtype) through
// the chunked-GDN pipeline; other engines store Hg-compact bf16 and expand by
// index arithmetic in-kernel. QKC=true writes q/k once per k-head as bf16
// at [rows, HK, s] (v stays f32 expanded) - the same f32 values the
// consumer (stage1_rs) rounds to bf16 itself today, so the pipeline is
// BIT-IDENTICAL end to end. Paired with pd_gated_delta_chunked_rs_vl_qkc
// by the ENGINE (one latch drives both slots - no env mirroring here).
template <bool QKC = false>
__global__ void pd_causal_conv1d_silu_qkv_vl_kernel(
        const float* __restrict__ x, const float* __restrict__ w,
        const uint32_t* __restrict__ row0s, float* __restrict__ q_out,
        float* __restrict__ k_out, float* __restrict__ v_out,
        uint32_t n_k_heads, uint32_t n_v_heads, uint32_t s, uint32_t k) {
    const uint32_t row = blockIdx.y;
    const uint32_t j = threadIdx.x;
    if (j >= s) return;
    const uint32_t row0 = row0s[row];
    const uint32_t key_dim = s * n_k_heads;
    const uint32_t conv_dim = 2u * key_dim + s * n_v_heads;
    if (blockIdx.x >= n_k_heads) {
        const uint32_t hv = blockIdx.x - n_k_heads;
        const uint32_t cv = 2u * key_dim + hv * s + j;
        float av = 0.0f;
        if (k == 4u && row >= row0 + 3u) {
            const float4 wv = *reinterpret_cast<const float4*>(w + (size_t)cv * 4u);
            const float* xc = x + (uint64_t)(row - 3u) * conv_dim + cv;
            av += wv.x * xc[0];
            av += wv.y * xc[conv_dim];
            av += wv.z * xc[2u * (size_t)conv_dim];
            av += wv.w * xc[3u * (size_t)conv_dim];
        } else {
            for (uint32_t kk = 0; kk < k; ++kk) {
                int64_t ti = (int64_t)row - (int64_t)(k - 1) + (int64_t)kk;
                if (ti >= (int64_t)row0) av += w[(size_t)cv * k + kk] * x[(uint64_t)ti * conv_dim + cv];
            }
        }
        const float vj = av / (1.0f + expf(-av));
        v_out[((size_t)row * n_v_heads + hv) * s + j] = vj;
        return;
    }
    const uint32_t hk = blockIdx.x;
    const uint32_t cq = hk * s + j;
    const uint32_t ck = key_dim + hk * s + j;
    float aq = 0.0f, ak = 0.0f;
    if (k == 4u && row >= row0 + 3u) {
        const float4 wq = *reinterpret_cast<const float4*>(w + (size_t)cq * 4u);
        const float4 wk = *reinterpret_cast<const float4*>(w + (size_t)ck * 4u);
        const float* xq = x + (uint64_t)(row - 3u) * conv_dim + cq;
        const float* xk = x + (uint64_t)(row - 3u) * conv_dim + ck;
        aq += wq.x * xq[0];
        ak += wk.x * xk[0];
        aq += wq.y * xq[conv_dim];
        ak += wk.y * xk[conv_dim];
        aq += wq.z * xq[2u * (size_t)conv_dim];
        ak += wk.z * xk[2u * (size_t)conv_dim];
        aq += wq.w * xq[3u * (size_t)conv_dim];
        ak += wk.w * xk[3u * (size_t)conv_dim];
    } else {
        for (uint32_t kk = 0; kk < k; ++kk) {
            int64_t ti = (int64_t)row - (int64_t)(k - 1) + (int64_t)kk;
            if (ti >= (int64_t)row0) {
                const float* xr = x + (uint64_t)ti * conv_dim;
                aq += w[(size_t)cq * k + kk] * xr[cq];
                ak += w[(size_t)ck * k + kk] * xr[ck];
            }
        }
    }
    const float qj = aq / (1.0f + expf(-aq));
    const float kj = ak / (1.0f + expf(-ak));
    float q2 = qj * qj, k2 = kj * kj;
#pragma unroll
    for (uint32_t off = 16; off > 0; off >>= 1) {
        q2 += __shfl_xor_sync(0xffffffffu, q2, off);
        k2 += __shfl_xor_sync(0xffffffffu, k2, off);
    }
    __shared__ float sh[8];
    const uint32_t lane = j & 31u, warp = j >> 5, nwarps = (s + 31u) >> 5;
    if (lane == 0) { sh[warp] = q2; sh[4 + warp] = k2; }
    __syncthreads();
    float qs = 0.0f, ks = 0.0f;
    for (uint32_t ww = 0; ww < nwarps; ++ww) { qs += sh[ww]; ks += sh[4 + ww]; }
    const float qn = qj * rsqrtf(qs + 1e-6f) * rsqrtf((float)s);
    const float kn = kj * rsqrtf(ks + 1e-6f);
    if constexpr (QKC) {
        const size_t oidx = ((size_t)row * n_k_heads + hk) * s + j;
        ((__nv_bfloat16*)q_out)[oidx] = __float2bfloat16(qn);
        ((__nv_bfloat16*)k_out)[oidx] = __float2bfloat16(kn);
    } else {
        for (uint32_t hv = hk; hv < n_v_heads; hv += n_k_heads) {
            const size_t oidx = ((size_t)row * n_v_heads + hv) * s + j;
            q_out[oidx] = qn;
            k_out[oidx] = kn;
        }
    }
}

PD_EXPORT
int pd_causal_conv1d_silu_qkv_vl(const void* x, const void* w,
                                 const void* row0s, void* q_out, void* k_out,
                                 void* v_out, uint32_t n_rows,
                                 uint32_t n_k_heads, uint32_t n_v_heads,
                                 uint32_t s, uint32_t k, void* stream) {
    if (n_rows == 0 || n_v_heads == 0 || s == 0) return 0;
    if (s > 128 || (s & 31u)) return cudaErrorInvalidValue;
    dim3 grid(n_k_heads + n_v_heads, n_rows);
    pd_causal_conv1d_silu_qkv_vl_kernel<<<grid, s, 0, (cudaStream_t)stream>>>(
        (const float*)x, (const float*)w, (const uint32_t*)row0s,
        (float*)q_out, (float*)k_out, (float*)v_out, n_k_heads, n_v_heads, s, k);
    return pd_launch_status();
}

// QKC twin (slot 446): identical signature; q_out/k_out receive the COMPACT
// bf16 [rows, HK, s] planes (see the kernel comment). The engine pairs this
// with pd_gated_delta_chunked_rs_vl_qkc under one latch.
PD_EXPORT
int pd_causal_conv1d_silu_qkv_vl_qkc(const void* x, const void* w,
                                     const void* row0s, void* q_out,
                                     void* k_out, void* v_out, uint32_t n_rows,
                                     uint32_t n_k_heads, uint32_t n_v_heads,
                                     uint32_t s, uint32_t k, void* stream) {
    if (n_rows == 0 || n_v_heads == 0 || s == 0) return 0;
    if (s > 128 || (s & 31u)) return cudaErrorInvalidValue;
    dim3 grid(n_k_heads + n_v_heads, n_rows);
    pd_causal_conv1d_silu_qkv_vl_kernel<true><<<grid, s, 0, (cudaStream_t)stream>>>(
        (const float*)x, (const float*)w, (const uint32_t*)row0s,
        (float*)q_out, (float*)k_out, (float*)v_out, n_k_heads, n_v_heads, s, k);
    return pd_launch_status();
}

// v-bf16 twin (slot 263, the DN bf16-operand chain's severable slice): q/k
// stay f32 (the classic walk reads both; the dots keep full-rate f32
// fragment loads) - only v rounds to bf16, and v is consumed only by
// stage1's du pass. Routed when the consumer is guaranteed the chunked
// pipeline (r >= chunk_min; the sequential kernel reads f32 v).
template <typename VT>
__global__ void pd_causal_conv1d_silu_qkv_vb_kernel(
        const float* __restrict__ x, const float* __restrict__ w,
        float* __restrict__ q_out, float* __restrict__ k_out,
        VT* __restrict__ v_out, uint32_t n_k_heads, uint32_t n_v_heads,
        uint32_t s, uint32_t k) {
    const uint32_t hv = blockIdx.x;
    const uint32_t row = blockIdx.y;
    const uint32_t j = threadIdx.x;
    if (j >= s) return;
    const uint32_t hk = hv % n_k_heads;
    const uint32_t key_dim = s * n_k_heads;
    const uint32_t conv_dim = 2u * key_dim + s * n_v_heads;
    const uint32_t cq = hk * s + j;
    const uint32_t ck = key_dim + hk * s + j;
    const uint32_t cv = 2u * key_dim + hv * s + j;
    float aq = 0.0f, ak = 0.0f, av = 0.0f;
    for (uint32_t kk = 0; kk < k; ++kk) {
        int64_t ti = (int64_t)row - (int64_t)(k - 1) + (int64_t)kk;
        if (ti >= 0) {
            const float* xr = x + (uint64_t)ti * conv_dim;
            aq += w[(size_t)cq * k + kk] * xr[cq];
            ak += w[(size_t)ck * k + kk] * xr[ck];
            av += w[(size_t)cv * k + kk] * xr[cv];
        }
    }
    const float qj = aq / (1.0f + expf(-aq));
    const float kj = ak / (1.0f + expf(-ak));
    const float vj = av / (1.0f + expf(-av));
    float q2 = qj * qj, k2 = kj * kj;
#pragma unroll
    for (uint32_t off = 16; off > 0; off >>= 1) {
        q2 += __shfl_xor_sync(0xffffffffu, q2, off);
        k2 += __shfl_xor_sync(0xffffffffu, k2, off);
    }
    __shared__ float sh[8];
    const uint32_t lane = j & 31u, warp = j >> 5, nwarps = (s + 31u) >> 5;
    if (lane == 0) { sh[warp] = q2; sh[4 + warp] = k2; }
    __syncthreads();
    float qs = 0.0f, ks = 0.0f;
    for (uint32_t ww = 0; ww < nwarps; ++ww) { qs += sh[ww]; ks += sh[4 + ww]; }
    const size_t oidx = ((size_t)row * n_v_heads + hv) * s + j;
    q_out[oidx] = qj * rsqrtf(qs + 1e-6f) * rsqrtf((float)s);
    k_out[oidx] = kj * rsqrtf(ks + 1e-6f);
    v_out[oidx] = (VT)vj;
}

PD_EXPORT
int pd_causal_conv1d_silu_qkv_b16(const void* x, const void* w, void* q_out,
                                  void* k_out, void* v_out, uint32_t n_rows,
                                  uint32_t n_k_heads, uint32_t n_v_heads,
                                  uint32_t s, uint32_t k, void* stream) {
    if (n_rows == 0 || n_v_heads == 0 || s == 0) return 0;
    if (s > 128 || (s & 31u)) return cudaErrorInvalidValue;
    if (pd_conv_qkv_once_enabled()) {
        if (pd_env("PADDOCK_ROUTE_WITNESS")) {
            static bool once = false;
            if (!once) { fprintf(stderr, "pd route: conv1d_silu_qkv v-b16 (once)\n"); once = true; }
        }
        dim3 grid(n_k_heads + n_v_heads, n_rows);
        pd_causal_conv1d_silu_qkv_once_kernel<__nv_bfloat16><<<grid, s, 0, (cudaStream_t)stream>>>(
            (const float*)x, (const float*)w, (float*)q_out, (float*)k_out,
            (__nv_bfloat16*)v_out, n_k_heads, n_v_heads, s, k);
        return pd_launch_status();
    }
    if (pd_env("PADDOCK_ROUTE_WITNESS")) {
        static bool once = false;
        if (!once) { fprintf(stderr, "pd route: conv1d_silu_qkv v-b16\n"); once = true; }
    }
    dim3 grid(n_v_heads, n_rows);
    pd_causal_conv1d_silu_qkv_vb_kernel<__nv_bfloat16><<<grid, s, 0, (cudaStream_t)stream>>>(
        (const float*)x, (const float*)w, (float*)q_out, (float*)k_out,
        (__nv_bfloat16*)v_out, n_k_heads, n_v_heads, s, k);
    return pd_launch_status();
}

// DeltaNet gate math: beta = sigmoid(b); g = ssm_a * softplus(a + dt_bias).
// One thread per (t,h). softplus stable: max(x,0) + log1p(exp(-|x|)).
__global__ void pd_delta_gate_kernel(
        const float* __restrict__ a, const float* __restrict__ b,
        const float* __restrict__ ssm_a, const float* __restrict__ dt_bias,
        float* __restrict__ g, float* __restrict__ beta,
        uint32_t n_tokens, uint32_t n_heads) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)n_tokens * n_heads;
    if (idx >= total) return;
    uint32_t h = (uint32_t)(idx % n_heads);
    float bx = b[idx];
    beta[idx] = 1.0f / (1.0f + expf(-bx));
    float ax = a[idx] + dt_bias[h];
    float sp = fmaxf(ax, 0.0f) + log1pf(expf(-fabsf(ax)));
    g[idx] = ssm_a[h] * sp;
}

PD_EXPORT
int pd_delta_gate(const void* a, const void* b, const void* ssm_a, const void* dt_bias,
                  void* g, void* beta, uint32_t n_tokens, uint32_t n_heads, void* stream) {
    uint64_t total = (uint64_t)n_tokens * n_heads;
    if (total == 0) return 0;
    uint32_t threads = 256;
    uint64_t blocks = (total + threads - 1) / threads;
    pd_delta_gate_kernel<<<(uint32_t)blocks, threads, 0, (cudaStream_t)stream>>>(
        (const float*)a, (const float*)b, (const float*)ssm_a, (const float*)dt_bias,
        (float*)g, (float*)beta, n_tokens, n_heads);
    return pd_launch_status();
}

// delta_gate over the FUSED alpha||beta activation layout: ab is [n_tokens]
// rows of 2*n_heads floats (alpha cols 0..h, beta cols h..2h) - the output of
// the one-call f32-plane decay GEMM (x2-v3: the [in, 64] concat plane rides
// pd_gemm_f32_nt's 64-aligned tile, one x read for both projections). Same
// per-element math and evaluation order as pd_delta_gate.
__global__ void pd_delta_gate_ab_kernel(
        const float* __restrict__ ab, const float* __restrict__ ssm_a,
        const float* __restrict__ dt_bias, float* __restrict__ g,
        float* __restrict__ beta, uint32_t n_tokens, uint32_t n_heads) {
    PD_PDL_ARM();  // consumer-safe for early PDL launches (2026-08-31)
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)n_tokens * n_heads;
    if (idx >= total) return;
    uint32_t t = (uint32_t)(idx / n_heads);
    uint32_t h = (uint32_t)(idx % n_heads);
    const float* row = ab + (size_t)t * 2u * n_heads;
    float bx = row[n_heads + h];
    beta[idx] = 1.0f / (1.0f + expf(-bx));
    float ax = row[h] + dt_bias[h];
    float sp = fmaxf(ax, 0.0f) + log1pf(expf(-fabsf(ax)));
    g[idx] = ssm_a[h] * sp;
}

PD_EXPORT
int pd_delta_gate_ab(const void* ab, const void* ssm_a, const void* dt_bias, void* g,
                     void* beta, uint32_t n_tokens, uint32_t n_heads, void* stream) {
    uint64_t total = (uint64_t)n_tokens * n_heads;
    if (total == 0) return 0;
    uint32_t threads = 256;
    uint64_t blocks = (total + threads - 1) / threads;
    pd_delta_gate_ab_kernel<<<(uint32_t)blocks, threads, 0, (cudaStream_t)stream>>>(
        (const float*)ab, (const float*)ssm_a, (const float*)dt_bias, (float*)g,
        (float*)beta, n_tokens, n_heads);
    return pd_launch_status();
}

// Fused alpha/beta matvec + delta gate: one launch replacing the
// pd_matvec_f32_batch<2> + pd_delta_gate_ab pair on the decode band (the ab
// plane is [2*n_heads, in_dim] f32; the pair cost two ~5 us launches per
// DeltaNet layer per tick - pure latency on a tiny L2-resident weight).
// Each block owns head h and BT tokens and computes both ab rows (o = h
// alpha, o = n_heads+h beta) with the matvec's exact per-element schedule:
// i = tid stride nth ascending per accumulator, the same shfl tree, the
// same serial cross-warp fold - per (row, token) the summation chain is
// untouched (the recurrence consumes g/beta; fixed-order summation is the
// product contract), then the epilogue applies pd_delta_gate_ab's
// elementwise expressions verbatim. Bit-identical by construction; the
// engine test bit-compares against the pair.
template <uint32_t BT>
__global__ void pd_matvec_ab_gate_kernel(
        const float* __restrict__ w, const float* __restrict__ x,
        const float* __restrict__ ssm_a, const float* __restrict__ dt_bias,
        float* __restrict__ g, float* __restrict__ beta,
        uint32_t in_dim, uint32_t n_heads, uint32_t batch) {
    const uint32_t h = blockIdx.x, t0 = blockIdx.y * BT;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    const float* wa = w + (size_t)h * in_dim;
    const float* wb = w + (size_t)(n_heads + h) * in_dim;
    float acca[BT] = {}, accb[BT] = {};
    for (uint32_t i = tid; i < in_dim; i += nth) {
        const float wav = wa[i], wbv = wb[i];
        #pragma unroll
        for (uint32_t b = 0; b < BT; ++b)
            if (t0 + b < batch) {
                const float xv = x[(size_t)(t0 + b) * in_dim + i];
                acca[b] += wav * xv;
                accb[b] += wbv * xv;
            }
    }
    __shared__ float wsa[8][BT], wsb[8][BT];
    const uint32_t warp = tid >> 5, lane = tid & 31u;
    #pragma unroll
    for (uint32_t b = 0; b < BT; ++b) {
        float va = acca[b], vb = accb[b];
        for (uint32_t s = 16; s > 0; s >>= 1) {
            va += __shfl_down_sync(0xffffffffu, va, s);
            vb += __shfl_down_sync(0xffffffffu, vb, s);
        }
        if (lane == 0) {
            wsa[warp][b] = va;
            wsb[warp][b] = vb;
        }
    }
    __syncthreads();
    if (tid == 0) {
        const uint32_t nwarps = (nth + 31u) >> 5;
        #pragma unroll
        for (uint32_t b = 0; b < BT; ++b) {
            if (t0 + b >= batch) break;
            float va = 0.0f, vb = 0.0f;
            for (uint32_t i = 0; i < nwarps; ++i) {
                va += wsa[i][b];
                vb += wsb[i][b];
            }
            const uint64_t idx = (uint64_t)(t0 + b) * n_heads + h;
            beta[idx] = 1.0f / (1.0f + expf(-vb));
            const float ax = va + dt_bias[h];
            const float sp = fmaxf(ax, 0.0f) + log1pf(expf(-fabsf(ax)));
            g[idx] = ssm_a[h] * sp;
        }
    }
}

PD_EXPORT
int pd_matvec_ab_gate(const void* w, const void* x, const void* ssm_a,
                      const void* dt_bias, void* g, void* beta, uint32_t in_dim,
                      uint32_t n_heads, uint32_t batch, void* stream) {
    if (n_heads == 0 || batch == 0) return 0;
    dim3 grid(n_heads, (batch + 1u) / 2u);
    pd_matvec_ab_gate_kernel<2u><<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const float*)w, (const float*)x, (const float*)ssm_a,
        (const float*)dt_bias, (float*)g, (float*)beta, in_dim, n_heads, batch);
    return pd_launch_status();
}

// Gated RMSNorm over d per row: out = (x*rsqrt(mean(x^2)+eps))*weight*silu(z).
// One block per row, blockDim = d (a power of two), tree-reduce the sum of squares.
__global__ void pd_gated_rmsnorm_kernel(
        const float* __restrict__ x, const float* __restrict__ z,
        const float* __restrict__ weight, float* __restrict__ out,
        uint32_t n_rows, uint32_t d, float eps, uint32_t z_stride,
        uint32_t z_rows_per_b) {
    uint32_t r = blockIdx.x;
    uint32_t j = threadIdx.x;
    if (r >= n_rows || j >= d) return;
    extern __shared__ float sm[];
    size_t off = (size_t)r * d;
    float xj = x[off + j];
    sm[j] = xj * xj;
    __syncthreads();
    for (uint32_t s = d >> 1; s > 0; s >>= 1) {
        if (j < s) sm[j] += sm[j + s];
        __syncthreads();
    }
    float inv = rsqrtf(sm[0] / (float)d + eps);
    // z rows are (batch, head) pairs; a fused-plane z strides per BATCH:
    // element (r, j) lives at (r / rpb) * z_stride + (r % rpb) * d + j.
    // Legacy (dense z): rpb = 1, z_stride = d - identical addressing.
    const size_t zoff =
        (size_t)(r / z_rows_per_b) * z_stride + (size_t)(r % z_rows_per_b) * d;
    float zj = z[zoff + j];
    out[off + j] = xj * inv * weight[j] * (zj / (1.0f + expf(-zj)));
}

// gated rmsnorm + e4m3 quant fused (prefill glue): the f8
// out_proj GEMM consumed a separate quantize_e4m3 pass re-reading the
// n_rows x d output - one r x value_dim round trip per DN layer-tick.
// d % 32 == 0 (state_size 128); q/scale follow quantize_e4m3's per-32
// e8m0 block layout over the flattened [n_rows*d] stream. f32 out still
// written (fallback consumers unchanged).
__global__ void pd_gated_rmsnorm_e4m3_kernel(
        const float* __restrict__ x, const float* __restrict__ z,
        const float* __restrict__ weight, float* __restrict__ out,
        unsigned char* __restrict__ q, unsigned char* __restrict__ scale,
        uint32_t n_rows, uint32_t d, float eps) {
    uint32_t r = blockIdx.x;
    uint32_t j = threadIdx.x;
    if (r >= n_rows || j >= d) return;
    extern __shared__ float sm[];
    size_t off = (size_t)r * d;
    float xj = x[off + j];
    sm[j] = xj * xj;
    __syncthreads();
    for (uint32_t s = d >> 1; s > 0; s >>= 1) {
        if (j < s) sm[j] += sm[j + s];
        __syncthreads();
    }
    float inv = rsqrtf(sm[0] / (float)d + eps);
    float zj = z[off + j];
    const float v = xj * inv * weight[j] * (zj / (1.0f + expf(-zj)));
    // out is optional (GDN formulation band): the fp8 chunk
    // path's only consumer is the out_proj GEMM's q/scale planes - skipping
    // the f32 store drops a full n_rows x d write per DN layer-tick
    if (out) out[off + j] = v;
    // per-32 e8m0 block quant, exact pd_e4m3_quant4 scale math (bit parity
    // with the standalone quantize_e4m3 this fuse replaces): 32 consecutive
    // lanes share a block of the flattened [n_rows*d] stream.
    float amax = fabsf(v);
    for (uint32_t sft = 16; sft > 0; sft >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, sft));
    int e = 0;
    if (amax > 0.0f) {
        int ex;
        float m = frexpf(amax, &ex);         // amax = m * 2^ex, m in [0.5, 1)
        e = ex - 9 + (m > 0.875f ? 1 : 0);   // 448 = 0.875 * 2^9
    }
    const float qinv = ldexpf(1.0f, -e);
    const size_t blk = (off + j) >> 5;
    if ((j & 31u) == 0u) scale[blk] = (unsigned char)(e + 127);
    q[off + j] = __nv_fp8_e4m3(v * qinv).__x;
}

// Warp-per-row twin of the per-32 fuse above (prefill wave band): the
// one-block-per-row form ran 137k 128-thread blocks with a 7-round smem
// tree per row - 141us/launch at the c16 admission wave (~350GB/s-class,
// 6.8ms/wave over 48 DN layers). One WARP per row, 8 rows/block: lane l
// owns elems {l, l+32, l+64, l+96}, so the smem tree's pairing (s=64:
// j,j+64; s=32: j,j+32) is lane-LOCAL and s<=16 rides shfl_down - the
// exact pairwise order of the block-tree (the _row twin's proven idiom);
// __fmul_rn squares forbid FMA contraction like the smem stores did. The
// per-32 quant reduces the same 32 values in the same shfl_xor butterfly;
// e/scale math verbatim. Bit parity probed old-vs-new.
// d == 128 only; kill: PADDOCK_NO_GRNQ_W32.
__global__ void pd_gated_rmsnorm_e4m3_w8r_kernel(
        const float* __restrict__ x, const float* __restrict__ z,
        const float* __restrict__ weight, float* __restrict__ out,
        unsigned char* __restrict__ q, unsigned char* __restrict__ scale,
        uint32_t n_rows, float eps) {
    const uint32_t d = 128u;
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const uint32_t r = blockIdx.x * 8u + warp;
    if (r >= n_rows) return;
    const size_t off = (size_t)r * d;
    float xj[4], zj[4];
    #pragma unroll
    for (uint32_t p = 0; p < 4; ++p) {
        const uint32_t j = lane + p * 32u;
        xj[p] = x[off + j];
        zj[p] = z[off + j];
    }
    const float s0 = __fmul_rn(xj[0], xj[0]);
    const float s1 = __fmul_rn(xj[1], xj[1]);
    const float s2 = __fmul_rn(xj[2], xj[2]);
    const float s3 = __fmul_rn(xj[3], xj[3]);
    const float pa = __fadd_rn(s0, s2);        // sm[l]    += sm[l+64]
    const float pb = __fadd_rn(s1, s3);        // sm[l+32] += sm[l+96]
    float acc = __fadd_rn(pa, pb);             // sm[l]    += sm[l+32]
    #pragma unroll
    for (uint32_t sh = 16; sh > 0; sh >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, sh);
    const float ssq = __shfl_sync(0xffffffffu, acc, 0);
    const float inv = rsqrtf(ssq / (float)d + eps);
    #pragma unroll
    for (uint32_t p = 0; p < 4; ++p) {
        const uint32_t j = lane + p * 32u;
        const float v = xj[p] * inv * weight[j] * (zj[p] / (1.0f + expf(-zj[p])));
        if (out) out[off + j] = v;
        float amax = fabsf(v);
        #pragma unroll
        for (uint32_t sft = 16; sft > 0; sft >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, sft));
        int e = 0;
        if (amax > 0.0f) {
            int ex;
            float m = frexpf(amax, &ex);         // amax = m * 2^ex, m in [0.5, 1)
            e = ex - 9 + (m > 0.875f ? 1 : 0);   // 448 = 0.875 * 2^9
        }
        const float qinv = ldexpf(1.0f, -e);
        const size_t blk = (off + j) >> 5;
        if (lane == 0u) scale[blk] = (unsigned char)(e + 127);
        q[off + j] = __nv_fp8_e4m3(v * qinv).__x;
    }
}

PD_EXPORT
int pd_gated_rmsnorm_e4m3(const void* x, const void* z, const void* weight,
                          void* out, void* q, void* scale, uint32_t n_rows,
                          uint32_t d, float eps, void* stream) {
#if defined(PD_BS_HOST)
    if (n_rows == 0 || d == 0) return 0;
    if (d & 31u) return cudaErrorInvalidValue;
    static const bool no_w32 = pd_env("PADDOCK_NO_GRNQ_W32") != nullptr;
    if (d == 128u && !no_w32) {
        pd_gated_rmsnorm_e4m3_w8r_kernel<<<(n_rows + 7u) / 8u, 256,
                                           0, (cudaStream_t)stream>>>(
            (const float*)x, (const float*)z, (const float*)weight, (float*)out,
            (unsigned char*)q, (unsigned char*)scale, n_rows, eps);
        return pd_launch_status();
    }
    size_t shmem = (size_t)d * sizeof(float);
    pd_gated_rmsnorm_e4m3_kernel<<<n_rows, d, shmem, (cudaStream_t)stream>>>(
        (const float*)x, (const float*)z, (const float*)weight, (float*)out,
        (unsigned char*)q, (unsigned char*)scale, n_rows, d, eps);
    return pd_launch_status();
#else
    (void)x; (void)z; (void)weight; (void)out; (void)q; (void)scale;
    (void)n_rows; (void)d; (void)eps; (void)stream;
    return cudaErrorNotSupported;
#endif
}

// ROW-scale twin of the fuse above (decode band): the f8t out_proj
// arm consumes (e4m3, f32 row scale) - quantize_e4m3_row's format - so the
// per-32 fusion can't serve it and the decode tick ran gated_rmsnorm + a
// standalone row1pc pass (2 launches, one d_core round trip, per DN layer).
// One warp per head, d=128: lane l owns elems {l, l+32, l+64, l+96}, so the
// smem tree's pairing (s=64: j,j+64; s=32: j,j+32) is lane-LOCAL and s<=16
// rides shfl_down - Exact pairwise order, zero smem, zero syncs in the norm
// phase. __fmul_rn/__fadd_rn forbid FMA contraction (the reference rounds
// each square into smem). Row max is order-free; quant math is row1pc's.
// Normed values stay in registers; quant is inline (a warp's 4 byte-strided
// stores land in one 128B line). Bit-identical to the two-kernel chain,
// f32 out nullable. 8.20 -> 6.15 us at b=32 h=48 (L2-cold lab).
template <uint32_t SLABS>
__global__ void pd_gated_rmsnorm_e4m3_row_kernel(
        const float* __restrict__ x, const float* __restrict__ z,
        const float* __restrict__ weight, float* __restrict__ out,
        unsigned char* __restrict__ q, float* __restrict__ rscale,
        uint32_t n_heads, float eps) {
    // Wait-only arm: this kernel REPLACES a plain launch (gated_rmsnorm),
    // which was a cascade break - see the PD_PDL_ARM_WAIT law in abi.cuh.
    // The release fires late (s_e block, after all x/z reads) instead.
    PD_PDL_ARM_WAIT();
    const uint32_t d = 128u;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5;
    const uint32_t wpb = nth >> 5;
    const uint32_t n = n_heads * d;
    const size_t off = (size_t)blockIdx.x * n;
    __shared__ float wmax[32];
    __shared__ int s_e;
    float v[SLABS][4];
    uint32_t ebase[SLABS];
    float amax = 0.0f;
    #pragma unroll
    for (uint32_t k = 0; k < SLABS; ++k) {
        const uint32_t h = k * wpb + warp;
        const uint32_t e0 = h * d;
        ebase[k] = e0;
        float xj[4], zj[4];
        #pragma unroll
        for (uint32_t p = 0; p < 4; ++p) {
            const uint32_t j = lane + p * 32u;
            xj[p] = x[off + e0 + j];
            zj[p] = z[off + e0 + j];
        }
        const float s0 = __fmul_rn(xj[0], xj[0]);
        const float s1 = __fmul_rn(xj[1], xj[1]);
        const float s2 = __fmul_rn(xj[2], xj[2]);
        const float s3 = __fmul_rn(xj[3], xj[3]);
        const float pa = __fadd_rn(s0, s2);        // sm[l]    += sm[l+64]
        const float pb = __fadd_rn(s1, s3);        // sm[l+32] += sm[l+96]
        float acc = __fadd_rn(pa, pb);             // sm[l]    += sm[l+32]
        #pragma unroll
        for (uint32_t sh = 16; sh > 0; sh >>= 1)
            acc += __shfl_down_sync(0xffffffffu, acc, sh);
        const float ssq = __shfl_sync(0xffffffffu, acc, 0);
        const float inv = rsqrtf(ssq / (float)d + eps);
        #pragma unroll
        for (uint32_t p = 0; p < 4; ++p) {
            const uint32_t j = lane + p * 32u;
            const float vv = xj[p] * inv * weight[j] * (zj[p] / (1.0f + expf(-zj[p])));
            v[k][p] = vv;
            amax = fmaxf(amax, fabsf(vv));
        }
    }
    if (out) {
        #pragma unroll
        for (uint32_t k = 0; k < SLABS; ++k)
            #pragma unroll
            for (uint32_t p = 0; p < 4; ++p)
                out[off + ebase[k] + lane + p * 32u] = v[k][p];
    }
    for (uint32_t sh = 16; sh > 0; sh >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, sh));
    if (lane == 0) wmax[warp] = amax;
    __syncthreads();
    if (tid == 0) {
        // all x/z reads are done - release the dependent chain now so its
        // dep-free prologue overlaps our quant/store tail only
        PD_PDL_RELEASE();
        float m = 0.0f;
        for (uint32_t w = 0; w < ((nth + 31u) >> 5); ++w) m = fmaxf(m, wmax[w]);
        int e = 0;
        if (m > 0.0f) {
            int ex;
            float fr = frexpf(m, &ex);
            e = ex - 9 + (fr > 0.875f ? 1 : 0);
        }
        s_e = e;
        rscale[blockIdx.x] = ldexpf(1.0f, e);
    }
    __syncthreads();
    const float qinv = ldexpf(1.0f, -s_e);
    unsigned char* qr = q + off;
    #pragma unroll
    for (uint32_t k = 0; k < SLABS; ++k)
        #pragma unroll
        for (uint32_t p = 0; p < 4; ++p)
            qr[ebase[k] + lane + p * 32u] = __nv_fp8_e4m3(v[k][p] * qinv).__x;
}

PD_EXPORT
int pd_gated_rmsnorm_e4m3_row(const void* x, const void* z, const void* weight,
                              void* out, void* q, void* rscale, uint32_t batch,
                              uint32_t n_heads, uint32_t d, float eps,
                              void* stream) {
#if defined(PD_BS_HOST)
    if (batch == 0 || n_heads == 0) return 0;
    // one warp per 128-wide head, 16 heads per 512-thread slab
    if (d != 128u || (n_heads & 15u) != 0 || n_heads > 128u)
        return cudaErrorInvalidValue;
    const uint32_t slabs = n_heads >> 4;
    #define PD_GRER_CASE(S)                                                    \
        case S:                                                                \
            pd_pdl_go(pd_gated_rmsnorm_e4m3_row_kernel<S>, batch, 512, 0u,     \
                      (cudaStream_t)stream, (const float*)x, (const float*)z,  \
                      (const float*)weight, (float*)out, (unsigned char*)q,    \
                      (float*)rscale, n_heads, eps);                           \
            break;
    switch (slabs) {
        PD_GRER_CASE(1) PD_GRER_CASE(2) PD_GRER_CASE(3) PD_GRER_CASE(4)
        PD_GRER_CASE(5) PD_GRER_CASE(6) PD_GRER_CASE(7) PD_GRER_CASE(8)
        default: return cudaErrorInvalidValue;
    }
    #undef PD_GRER_CASE
    return pd_launch_status();
#else
    (void)x; (void)z; (void)weight; (void)out; (void)q; (void)rscale;
    (void)batch; (void)n_heads; (void)d; (void)eps; (void)stream;
    return cudaErrorNotSupported;
#endif
}

PD_EXPORT
int pd_gated_rmsnorm(const void* x, const void* z, const void* weight, void* out,
                     uint32_t n_rows, uint32_t d, float eps, void* stream) {
    if (n_rows == 0 || d == 0) return 0;
    size_t shmem = (size_t)d * sizeof(float);
    pd_gated_rmsnorm_kernel<<<n_rows, d, shmem, (cudaStream_t)stream>>>(
        (const float*)x, (const float*)z, (const float*)weight, (float*)out,
        n_rows, d, eps, d, 1u);
    return pd_launch_status();
}

// z read STRIDED out of the DN in-proj fused plane (z rows are
// (batch, head) pairs; the plane strides per batch) - kills the row_slice
// copy of the z half. Same loads by value: bit-identical to slice-then-norm.
PD_EXPORT
int pd_gated_rmsnorm_s(const void* x, const void* z, const void* weight,
                       void* out, uint32_t n_rows, uint32_t d, float eps,
                       uint32_t z_stride, uint32_t z_rows_per_b, void* stream) {
    if (n_rows == 0 || d == 0) return 0;
    if (z_rows_per_b == 0 || z_stride < z_rows_per_b * d)
        return cudaErrorInvalidValue;
    size_t shmem = (size_t)d * sizeof(float);
    pd_gated_rmsnorm_kernel<<<n_rows, d, shmem, (cudaStream_t)stream>>>(
        (const float*)x, (const float*)z, (const float*)weight, (float*)out,
        n_rows, d, eps, z_stride, z_rows_per_b);
    return pd_launch_status();
}

// Split the DeltaNet conv output [T, 2*key_dim+value_dim] into q,k (key heads,
// GQA-repeated to n_v_heads: out head hv reads key head (hv % n_k_heads), matching
// llama.cpp ggml_repeat tiling) and v (value heads). Each output is [T, n_v_heads,
// s]. One thread per output element.
__global__ void pd_deltanet_split_gqa_kernel(
        const float* __restrict__ conv, float* __restrict__ q_out,
        float* __restrict__ k_out, float* __restrict__ v_out,
        uint32_t n_tokens, uint32_t n_k_heads, uint32_t n_v_heads, uint32_t s) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)n_tokens * n_v_heads * s;
    if (idx >= total) return;
    uint32_t si = (uint32_t)(idx % s);
    uint32_t hv = (uint32_t)((idx / s) % n_v_heads);
    uint32_t t  = (uint32_t)(idx / ((uint64_t)s * n_v_heads));
    // GQA repeat matches llama.cpp ggml_repeat_4d = TILING: v-head hv reads
    // key head (hv % n_k_heads), not hv/rep (repeat_interleave).
    uint32_t hk = hv % n_k_heads;
    uint32_t key_dim = s * n_k_heads;
    uint32_t conv_dim = 2u * key_dim + s * n_v_heads;
    size_t base = (size_t)t * conv_dim;
    q_out[idx] = conv[base + (size_t)hk * s + si];
    k_out[idx] = conv[base + key_dim + (size_t)hk * s + si];
    v_out[idx] = conv[base + 2u * key_dim + (size_t)hv * s + si];
}

PD_EXPORT
int pd_deltanet_split_gqa(const void* conv, void* q_out, void* k_out, void* v_out,
                          uint32_t n_tokens, uint32_t n_k_heads, uint32_t n_v_heads,
                          uint32_t s, void* stream) {
    uint64_t total = (uint64_t)n_tokens * n_v_heads * s;
    if (total == 0) return 0;
    uint32_t threads = 256;
    uint64_t blocks = (total + threads - 1) / threads;
    pd_deltanet_split_gqa_kernel<<<(uint32_t)blocks, threads, 0, (cudaStream_t)stream>>>(
        (const float*)conv, (float*)q_out, (float*)k_out, (float*)v_out,
        n_tokens, n_k_heads, n_v_heads, s);
    return pd_launch_status();
}

// Partial sectioned M-RoPE (Qwen3.5 multimodal rotary), in place over
// x [n_tokens, n_heads*head_dim]. One thread per (token, head). Rotates NEOX
// pairs (p, p + n_rot/2) for p in [0, n_rot/2); channels [n_rot, head_dim) pass
// through. positions is [4, n_tokens] axis-major; each pair picks its axis from
// the cumulative sections [s0,s1,s2,s3] (t,h,w,e). All four axis theta chains
// advance every pair. Matches reference::ops::YarnRope::apply_mrope.
// One thread per (token, head, rotation pair). The old shape was one thread per
// (token, head) walking all n_rot/2 pairs serially - at b=1 decode that is 16
// threads on the whole die running 32 dependent sinf/cosf chains each (8.2 us
// for a 0.03 MB op). Thread p rebuilds its theta by p REPEATED multiplies in
// the exact order the serial loop used, so results stay bit-identical to the
// old kernel (powf would be faster but a different numeric class); the
// multiply chain is trivially cheap next to one sinf+cosf.
__global__ void pd_mrope_kernel(float* __restrict__ x, const unsigned int* __restrict__ positions,
                                uint32_t n_tokens, uint32_t n_heads, uint32_t head_dim,
                                uint32_t n_rot, float theta_scale, float freq_scale,
                                float corr_low, float corr_high, float ext_factor, float mscale,
                                uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3) {
    PD_PDL_ARM();  // consumer-safe for early PDL launches (2026-08-31)
    uint32_t half = n_rot / 2;
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_tokens * n_heads * half) return;
    uint32_t p = gid % half;
    uint32_t idx = gid / half;                    // (t, h) flat index
    uint32_t t = idx / n_heads;
    float* head = x + (size_t)idx * head_dim;
    uint32_t sect = s0 + s1 + s2 + s3;
    uint32_t sec_h = s0;
    uint32_t sec_w = s0 + s1;
    uint32_t sec_e = sec_w + s2;
    uint32_t sector = p % sect;
    float base;
    if (sector < sec_h) base = (float)positions[t];
    else if (sector < sec_w) base = (float)positions[(size_t)n_tokens + t];
    else if (sector < sec_e) base = (float)positions[(size_t)2 * n_tokens + t];
    else base = (float)positions[(size_t)3 * n_tokens + t];
    // theta_scale^p by the serial loop's own multiply order (bit-exact match)
    for (uint32_t i = 0; i < p; ++i) base *= theta_scale;
    float y = ((float)p - corr_low) / fmaxf(0.001f, corr_high - corr_low);
    float ramp = (1.0f - fminf(1.0f, fmaxf(0.0f, y))) * ext_factor;
    float angle = (freq_scale * base) * (1.0f - ramp) + base * ramp;
    float sn = sinf(angle) * mscale;
    float cs = cosf(angle) * mscale;
    float a = head[p];
    float b = head[p + half];
    head[p] = a * cs - b * sn;
    head[p + half] = a * sn + b * cs;
}

PD_EXPORT
int pd_mrope(void* x, const void* positions, uint32_t n_tokens, uint32_t n_heads,
             uint32_t head_dim, uint32_t n_rot, float theta_scale, float freq_scale,
             float corr_low, float corr_high, float ext_factor, float mscale,
             uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3, void* stream) {
    uint32_t total = n_tokens * n_heads * (n_rot / 2);   // one thread per pair
    if (total == 0) return 0;
    uint32_t threads = 256;
    uint32_t blocks = (total + threads - 1) / threads;
    pd_mrope_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (float*)x, (const unsigned int*)positions, n_tokens, n_heads, head_dim, n_rot,
        theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale, s0, s1, s2, s3);
    return pd_launch_status();
}

// Sigmoid output gate, in place: x[i] *= sigmoid(gate[i]). grid ceil(n/256).
__global__ void pd_mul_sigmoid_kernel(float* __restrict__ x, const float* __restrict__ gate,
                                      uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] *= 1.0f / (1.0f + expf(-gate[i]));
}

// float4 twin: the muse c32 wide prefill pass runs this over
// [5984 x 6656] f32 at 4B-scalar transactions (~4.3 TB/s effective); 16B
// vectors close that band. Values are bit-identical per element (same
// expression, no reduction), so this is a pure transaction-shape change -
// still gated to the wide band below so small launches keep the old grid.
__global__ void pd_mul_sigmoid_v4_kernel(float4* __restrict__ x,
                                         const float4* __restrict__ gate,
                                         uint32_t n4) {
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        float4 xv = x[i];
        const float4 gv = gate[i];
        xv.x *= 1.0f / (1.0f + expf(-gv.x));
        xv.y *= 1.0f / (1.0f + expf(-gv.y));
        xv.z *= 1.0f / (1.0f + expf(-gv.z));
        xv.w *= 1.0f / (1.0f + expf(-gv.w));
        x[i] = xv;
    }
}

PD_EXPORT
int pd_mul_sigmoid(void* x, const void* gate, uint32_t n, void* stream) {
    if (n == 0) return 0;
    // wide-band vec4 election: big prefill planes only (decode ticks keep
    // the classic launch byte-for-byte); alignment is a hard requirement
    if (n >= (1u << 20) && (n & 3u) == 0 && (((uintptr_t)x | (uintptr_t)gate) & 15u) == 0) {
        const uint32_t n4 = n >> 2;
        uint32_t blocks = (n4 + 255u) / 256u;
        if (blocks > 8192u) blocks = 8192u;
        pd_mul_sigmoid_v4_kernel<<<blocks, 256, 0, (cudaStream_t)stream>>>(
            (float4*)x, (const float4*)gate, n4);
        return pd_launch_status();
    }
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1) / threads;
    pd_mul_sigmoid_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (float*)x, (const float*)gate, n);
    return pd_launch_status();
}

// Plain SwiGLU, in place on gate: gate[i] = silu(gate[i]) * up[i]. grid ceil(n/256).
__global__ void pd_swiglu_kernel(float* __restrict__ gate, const float* __restrict__ up,
                                 uint32_t n) {
    PD_PDL_ARM();  // cascade (granite chain)
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    gate[i] = (g / (1.0f + expf(-g))) * up[i];
}

// slot 570: the same swiglu with a bf16 MIRROR of the result. qwen4_exp's
// shared-expert chain feeds its down plane straight from here, and that
// consumer was casting the [n, ff] result every layer - 96 casts a tick at
// c8, 0.75 ms. Values are pd_swiglu_kernel's verbatim.
__global__ void pd_swiglu_mir_kernel(float* __restrict__ gate,
                                     const float* __restrict__ up,
                                     __nv_bfloat16* __restrict__ out16,
                                     uint32_t n) {
    PD_PDL_ARM();
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    const float v = (g / (1.0f + expf(-g))) * up[i];
    gate[i] = v;
    if (out16) out16[i] = __float2bfloat16(v);
}

PD_EXPORT
int pd_swiglu_mir(void* gate, const void* up, void* out16, uint32_t n,
                  void* stream) {
    if (n == 0) return 0;
    pd_pdl_go(pd_swiglu_mir_kernel, (n + 255u) / 256u, 256, 0,
              (cudaStream_t)stream, (float*)gate, (const float*)up,
              (__nv_bfloat16*)out16, n);
    return pd_launch_status();
}

// SwiGLU over a FUSED gate|up GEMM output: the merged gate_up plane lands
// [tok][gate(ff)|up(ff)] rows, so out[t*ff+j] = silu(f[t*2ff+j]) *
// f[t*2ff+ff+j] - the exact pd_swiglu_kernel expression (bit-identical
// values), packed output for the down GEMM. One thread per output element.
__global__ void pd_swiglu_fused_kernel(const float* __restrict__ fused,
                                       float* __restrict__ out, uint32_t ff,
                                       uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t tok = i / ff, j = i % ff;
    const float* row = fused + (size_t)tok * 2u * ff;
    float g = row[j];
    out[i] = (g / (1.0f + expf(-g))) * row[ff + j];
}

// swiglu of a fused [rows, 2*ff] gate|up landing -> per-ROW e4m3, one launch,
// one block per row. Replaces pd_swiglu_fused + pd_quantize_e4m3_row1p on the
// f8t FFN down input, where the row is the WIDEST in the model (ff = 17408 on
// the 27B). Values are the swiglu kernel's expression verbatim and the quant
// is the row1p clone (exact max, same exponent), so the pair's output is
// reproduced BIT-IDENTICALLY -- this only removes a launch and the f32 round
// trip of the widest activation in the tick.
__global__ void __launch_bounds__(1024) pd_swiglu_e4m3_row_kernel(
        const float* __restrict__ fused, unsigned char* __restrict__ q,
        float* __restrict__ rscale, uint32_t ff) {
    PD_PDL_ARM();
    extern __shared__ float sg_v[];                    // [ff] swiglu output
    const uint32_t r = blockIdx.x, tid = threadIdx.x, nth = blockDim.x;
    const float* row = fused + (size_t)r * 2u * ff;
    __shared__ float wmax[32];
    __shared__ int s_e;
    const uint32_t warp = tid >> 5, lane = tid & 31u;
    float m = 0.0f;
    for (uint32_t j = tid; j < ff; j += nth) {
        const float g = row[j];
        const float v = (g / (1.0f + expf(-g))) * row[ff + j];
        sg_v[j] = v;
        m = fmaxf(m, fabsf(v));
    }
    for (uint32_t sh = 16; sh > 0; sh >>= 1)
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, sh));
    if (lane == 0) wmax[warp] = m;
    __syncthreads();
    if (tid == 0) {
        float mm = 0.0f;
        for (uint32_t wi = 0; wi < ((nth + 31u) >> 5); ++wi) mm = fmaxf(mm, wmax[wi]);
        int e = 0;
        if (mm > 0.0f) {
            int ex;
            float fr = frexpf(mm, &ex);
            e = ex - 9 + (fr > 0.875f ? 1 : 0);
        }
        s_e = e;
        rscale[r] = ldexpf(1.0f, e);
    }
    __syncthreads();
    const float qinv = ldexpf(1.0f, -s_e);
    unsigned char* qr = q + (size_t)r * ff;
    for (uint32_t j = tid; j < ff; j += nth)
        qr[j] = __nv_fp8_e4m3(sg_v[j] * qinv).__x;
}

PD_EXPORT
int pd_swiglu_e4m3_row(const void* fused, void* q, void* rscale, uint32_t ff,
                       uint32_t n_rows, void* stream) {
    if (n_rows == 0 || ff == 0) return 0;
    // One block per row stages the row in dynamic smem: ff=17408 on the 27B is
    // 68 KB, over the 48 KB default, so opt in explicitly (B200 has 228 KB).
    // Grow the attribute as shapes appear, like the attention launchers do.
    const uint32_t smem = ff * 4u;
    if (smem > 200u * 1024u) return cudaErrorInvalidValue;   // caller splits
    static uint32_t hw = 0;
    if (smem > hw) {
        if (cudaFuncSetAttribute((const void*)pd_swiglu_e4m3_row_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)smem) != cudaSuccess)
            return cudaErrorInvalidValue;
        hw = smem;
    }
    pd_pdl_go(pd_swiglu_e4m3_row_kernel, n_rows, 1024, smem,
              (cudaStream_t)stream, (const float*)fused, (unsigned char*)q,
              (float*)rscale, ff);
    return pd_launch_status();
}

PD_EXPORT
int pd_swiglu_fused(const void* fused, void* out, uint32_t ff, uint32_t n_rows,
                    void* stream) {
    if (n_rows == 0 || ff == 0) return 0;
    uint32_t n = n_rows * ff;
    pd_swiglu_fused_kernel<<<(n + 255u) / 256u, 256, 0, (cudaStream_t)stream>>>(
        (const float*)fused, (float*)out, ff, n);
    return pd_launch_status();
}

// Interleaved twin: the merged plane's rows are (gate_j, up_j)
// pairs (Nvf4Plane::gu_pairs), so y holds g at 2j and u at 2j+1. Same
// expression as pd_swiglu_fused_kernel on the same values.
__global__ void pd_swiglu_fused_il_kernel(const float* __restrict__ fused,
                                          float* __restrict__ out, uint32_t ff,
                                          uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t tok = i / ff, j = i % ff;
    const float* row = fused + (size_t)tok * 2u * ff;
    float g = row[2u * j];
    out[i] = (g / (1.0f + expf(-g))) * row[2u * j + 1u];
}
PD_EXPORT
int pd_swiglu_fused_il(const void* fused, void* out, uint32_t ff, uint32_t n_rows,
                       void* stream) {
    if (n_rows == 0 || ff == 0) return 0;
    uint32_t n = n_rows * ff;
    pd_swiglu_fused_il_kernel<<<(n + 255u) / 256u, 256u, 0, (cudaStream_t)stream>>>(
        (const float*)fused, (float*)out, ff, n);
    return pd_launch_status();
}

// Packed row-slice from a fused GEMM landing ([rows, src_stride] row-major):
// dst[r*width + c] = src[r*src_stride + col_off + c]. The split epilogue for
// merged projection planes (DN in_qkv|gate_w). One thread per output element.
__global__ void pd_row_slice_kernel(const float* __restrict__ src,
                                    float* __restrict__ dst, uint32_t stride,
                                    uint32_t off, uint32_t width, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t r = i / width, c = i % width;
    dst[i] = src[(size_t)r * stride + off + c];
}

PD_EXPORT
int pd_add_inplace_b16(void* x, const void* y, uint32_t n, void* stream) {
    if (n == 0) return 0;
    pd_add_inplace_b16_kernel<<<(n + 255u) / 256u, 256, 0, (cudaStream_t)stream>>>(
        (float*)x, (const __nv_bfloat16*)y, n);
    return pd_launch_status();
}

// Up-to-4 slices of the same fused landing in one launch. Every fused
// projection plane is immediately split into its parts, one pd_row_slice per
// part: the DeltaNet landing takes 4 (mixed, z, alpha, beta) and the attn
// landing 3 (qg, k, v), which is 240 launches per decode tick on the 27B --
// the largest non-GEMM item in a NO-PDL capture at 0.385 ms/step, and
// pure copy. Identical per-element math to the single-slice kernel, so it is
// bit-identical by construction; it only removes launches.
// Unused slots take dst == nullptr / width 0.
__global__ void pd_row_slice4_kernel(const float* __restrict__ src, uint32_t stride,
                                     float* __restrict__ d0, uint32_t o0, uint32_t w0,
                                     float* __restrict__ d1, uint32_t o1, uint32_t w1,
                                     float* __restrict__ d2, uint32_t o2, uint32_t w2,
                                     float* __restrict__ d3, uint32_t o3, uint32_t w3,
                                     uint32_t wtot, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t r = i / wtot;
    uint32_t c = i - r * wtot;
    float* dst; uint32_t off, width;
    if (c < w0) { dst = d0; off = o0; width = w0; }
    else if ((c -= w0) < w1) { dst = d1; off = o1; width = w1; }
    else if ((c -= w1) < w2) { dst = d2; off = o2; width = w2; }
    else { c -= w2; dst = d3; off = o3; width = w3; }
    dst[(size_t)r * width + c] = src[(size_t)r * stride + off + c];
}

// row_slice4's DN split with the delta gate FOLDED into the alpha/beta
// parts (residue): slots 0/1 copy (mixed, z) exactly as
// pd_row_slice4; the ab block (2*n_heads cols at ab_off) computes g/beta
// in place of the raw d_a/d_b copies - same per-element expressions and
// evaluation order as pd_delta_gate on the sliced buffers (the copies
// were exact and the gate is elementwise), so g/beta are bit-identical
// while the intermediates and the separate delta_gate launch disappear
// (48 launches/tick on the 27B). One gate thread owns (t=r, h=c) and
// writes both outputs. Both replaced kernels were plain-launched - no
// PDL arm topology changes (the law's failure mode).
__global__ void pd_row_slice2_gate_kernel(
        const float* __restrict__ src, uint32_t stride,
        float* __restrict__ d0, uint32_t o0, uint32_t w0,
        float* __restrict__ d1, uint32_t o1, uint32_t w1,
        uint32_t ab_off, uint32_t n_heads,
        const float* __restrict__ ssm_a, const float* __restrict__ dt_bias,
        float* __restrict__ g, float* __restrict__ beta,
        uint32_t wtot, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t r = i / wtot;
    uint32_t c = i - r * wtot;
    if (c < w0) {
        d0[(size_t)r * w0 + c] = src[(size_t)r * stride + o0 + c];
        return;
    }
    if ((c -= w0) < w1) {
        d1[(size_t)r * w1 + c] = src[(size_t)r * stride + o1 + c];
        return;
    }
    c -= w1;  // c in [0, n_heads): the gate element (t=r, h=c)
    const float* row = src + (size_t)r * stride + ab_off;
    const float bx = row[n_heads + c];
    beta[(size_t)r * n_heads + c] = 1.0f / (1.0f + expf(-bx));
    const float ax = row[c] + dt_bias[c];
    const float sp = fmaxf(ax, 0.0f) + log1pf(expf(-fabsf(ax)));
    g[(size_t)r * n_heads + c] = ssm_a[c] * sp;
}

PD_EXPORT
int pd_row_slice2_gate(const void* src, uint32_t src_stride, uint32_t rows,
                       void* d0, uint32_t o0, uint32_t w0,
                       void* d1, uint32_t o1, uint32_t w1,
                       uint32_t ab_off, uint32_t n_heads,
                       const void* ssm_a, const void* dt_bias,
                       void* g, void* beta, void* stream) {
    if (rows == 0 || n_heads == 0) return 0;
    const uint32_t wtot = w0 + w1 + n_heads;
    const uint32_t n = rows * wtot;
    pd_row_slice2_gate_kernel<<<(n + 255u) / 256u, 256, 0,
                                (cudaStream_t)stream>>>(
        (const float*)src, src_stride, (float*)d0, o0, w0, (float*)d1, o1, w1,
        ab_off, n_heads, (const float*)ssm_a, (const float*)dt_bias,
        (float*)g, (float*)beta, wtot, n);
    return pd_launch_status();
}

PD_EXPORT
int pd_row_slice4(const void* src, uint32_t src_stride, uint32_t rows,
                  void* d0, uint32_t o0, uint32_t w0,
                  void* d1, uint32_t o1, uint32_t w1,
                  void* d2, uint32_t o2, uint32_t w2,
                  void* d3, uint32_t o3, uint32_t w3, void* stream) {
    if (rows == 0) return 0;
    const uint32_t wtot = w0 + w1 + w2 + w3;
    if (wtot == 0) return 0;
    const uint32_t n = rows * wtot;
    pd_row_slice4_kernel<<<(n + 255u) / 256u, 256, 0, (cudaStream_t)stream>>>(
        (const float*)src, src_stride,
        (float*)d0, o0, w0, (float*)d1, o1, w1,
        (float*)d2, o2, w2, (float*)d3, o3, w3, wtot, n);
    return pd_launch_status();
}

PD_EXPORT
int pd_row_slice(const void* src, void* dst, uint32_t src_stride, uint32_t col_off,
                 uint32_t width, uint32_t rows, void* stream) {
    if (rows == 0 || width == 0) return 0;
    uint32_t n = rows * width;
    pd_row_slice_kernel<<<(n + 255u) / 256u, 256, 0, (cudaStream_t)stream>>>(
        (const float*)src, (float*)dst, src_stride, col_off, width, n);
    return pd_launch_status();
}

PD_EXPORT
int pd_swiglu(void* gate, const void* up, uint32_t n, void* stream) {
    if (n == 0) return 0;
    uint32_t threads = 256;
    uint32_t blocks = (n + threads - 1) / threads;
    pd_pdl_go(pd_swiglu_kernel, blocks, threads, 0u, (cudaStream_t)stream,
        (float*)gate, (const float*)up, n);
    return pd_launch_status();
}

// Split the Qwen3.5 full-attn joint QG projection into contiguous query and gate.
// qg [n_tokens, n_heads*2*head_dim], per head [q(head_dim) | gate(head_dim)].
// One thread per (token, head, dim); q_out/gate_out [n_tokens, n_heads*head_dim].
__global__ void pd_split_qg_kernel(const float* __restrict__ qg, float* __restrict__ q_out,
                                   float* __restrict__ gate_out,
                                   uint32_t n_tokens, uint32_t n_heads, uint32_t head_dim) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)n_tokens * n_heads * head_dim;
    if (idx >= total) return;
    uint32_t d = (uint32_t)(idx % head_dim);
    uint32_t h = (uint32_t)((idx / head_dim) % n_heads);
    uint32_t t = (uint32_t)(idx / ((uint64_t)head_dim * n_heads));
    size_t src = ((size_t)t * n_heads + h) * 2u * head_dim;
    q_out[idx] = qg[src + d];
    gate_out[idx] = qg[src + head_dim + d];
}

PD_EXPORT
int pd_split_qg(const void* qg, void* q_out, void* gate_out,
                uint32_t n_tokens, uint32_t n_heads, uint32_t head_dim, void* stream) {
    uint64_t total = (uint64_t)n_tokens * n_heads * head_dim;
    if (total == 0) return 0;
    uint32_t threads = 256;
    uint64_t blocks = (total + threads - 1) / threads;
    pd_split_qg_kernel<<<(uint32_t)blocks, threads, 0, (cudaStream_t)stream>>>(
        (const float*)qg, (float*)q_out, (float*)gate_out, n_tokens, n_heads, head_dim);
    return pd_launch_status();
}

// Single-token causal conv1d + silu with a persistent window (DeltaNet decode).
// One thread per channel c. Reads the k-1 past (win) + this token (x_new), writes
// out=silu(conv), then advances the window in place. PD_CONV_K_MAX caps k.
#define PD_CONV_K_MAX 8
__global__ void pd_conv_step_kernel(float* __restrict__ win, const float* __restrict__ x_new,
                                    const float* __restrict__ w, float* __restrict__ out,
                                    uint32_t conv_dim, uint32_t k) {
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    uint32_t km1 = k - 1u;
    float vals[PD_CONV_K_MAX];
    for (uint32_t j = 0; j < km1; ++j) vals[j] = win[(size_t)j * conv_dim + c];
    vals[km1] = x_new[c];
    float acc = 0.0f;
    for (uint32_t j = 0; j < k; ++j) acc += w[(size_t)c * k + j] * vals[j];
    out[c] = acc / (1.0f + expf(-acc));
    // advance window: win[j-1] = vals[j] for j in 1..km1, then win[km1-1] = x_new
    for (uint32_t j = 1; j < km1; ++j) win[(size_t)(j - 1) * conv_dim + c] = vals[j];
    if (km1 >= 1u) win[(size_t)(km1 - 1) * conv_dim + c] = vals[km1];
}

PD_EXPORT
int pd_conv_step(void* win, const void* x_new, const void* w, void* out,
                 uint32_t conv_dim, uint32_t k, void* stream) {
    if (conv_dim == 0) return 0;
    uint32_t threads = 256;
    uint32_t blocks = (conv_dim + threads - 1) / threads;
    pd_conv_step_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(
        (float*)win, (const float*)x_new, (const float*)w, (float*)out, conv_dim, k);
    return pd_launch_status();
}

// Repack Q8_0 [n_blocks x 34B (f16 scale + 32 int8)] -> aligned int8 data stream
// [n_blocks x 32] + f16 scale stream [n_blocks]. One thread per block.
__global__ void pd_q8_0_repack_kernel(const uint8_t* __restrict__ src,
                                      int8_t* __restrict__ dst_data,
                                      __half* __restrict__ dst_scale, uint64_t n_blocks) {
    uint64_t b = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const uint8_t* s = src + b * 34u;
    __half h;
    memcpy(&h, s, sizeof(h));
    dst_scale[b] = h;
    int8_t* d = dst_data + b * 32u;
    for (uint32_t j = 0; j < 32u; ++j) d[j] = (int8_t)s[2u + j];
}

PD_EXPORT
int pd_q8_0_repack(const void* src, void* dst_data, void* dst_scale, uint64_t n_blocks,
                   void* stream) {
    if (n_blocks == 0) return 0;
    uint32_t threads = 256;
    uint64_t blocks = (n_blocks + threads - 1) / threads;
    pd_q8_0_repack_kernel<<<(uint32_t)blocks, threads, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)src, (int8_t*)dst_data, (__half*)dst_scale, n_blocks);
    return pd_launch_status();
}

// Vectorized exact-f32 Q8_0 GEMV over the repacked layout (aligned int8 data +
// separate f16 scale stream). One block per output row. Each thread consumes a
// contiguous 16-element chunk via a single 128-bit (int4) weight load plus four
// float4 activation loads, so a warp issues fully-coalesced 512-byte transactions
// instead of the interleaved layout's byte-wise, sector-straddling reads. The
// scale factors out of the 16-term inner product (one f16->f32 per half-block from
// shared), cutting the per-element multiply the byte-wise path pays. f32 accumulate
// -> the dequant math is bit-identical up to reduction order (~1e-7, argmax-stable),
// so the same-weights greedy-parity gate still holds.
__global__ void pd_q8_0_gemv_repacked_kernel(
    const int8_t* __restrict__ data, const __half* __restrict__ scale,
    const float* __restrict__ bias, const float* __restrict__ x, float* __restrict__ y,
    uint32_t in_dim, uint32_t out_dim) {
    uint32_t o = blockIdx.x;
    if (o >= out_dim) return;
    uint32_t tid = threadIdx.x, nth = blockDim.x;
    uint32_t n_blocks = in_dim >> 5;
    extern __shared__ float ssc[];
    const __half* srow = scale + (size_t)o * n_blocks;
    for (uint32_t b = tid; b < n_blocks; b += nth) ssc[b] = __half2float(srow[b]);
    // cascade: the scale preload above is dep-free (weights, not chain
    // data), so it runs during the predecessor's drain; the wait gates the x
    // reads below. No-op under plain launches.
    PD_PDL_ARM();
    __shared__ float wsum[32];
    __syncthreads();
    const int8_t* row = data + (size_t)o * in_dim;   // in_dim bytes, 32-aligned
    float acc = 0.0f;
    // 16 contiguous elements/thread: one int4 weight load + one float4x4 x load. A
    // 16-elem chunk lies wholly inside one Q8_0 block (16 = half a block), so its
    // scale is a single shared lookup ssc[base>>5].
    for (uint32_t base = tid * 16u; base < in_dim; base += nth * 16u) {
        int4 wv = *reinterpret_cast<const int4*>(row + base);
        const int8_t* wb = reinterpret_cast<const int8_t*>(&wv);
        float4 x0 = *reinterpret_cast<const float4*>(x + base);
        float4 x1 = *reinterpret_cast<const float4*>(x + base + 4);
        float4 x2 = *reinterpret_cast<const float4*>(x + base + 8);
        float4 x3 = *reinterpret_cast<const float4*>(x + base + 12);
        float s = (float)wb[0] * x0.x + (float)wb[1] * x0.y + (float)wb[2] * x0.z + (float)wb[3] * x0.w
                + (float)wb[4] * x1.x + (float)wb[5] * x1.y + (float)wb[6] * x1.z + (float)wb[7] * x1.w
                + (float)wb[8] * x2.x + (float)wb[9] * x2.y + (float)wb[10] * x2.z + (float)wb[11] * x2.w
                + (float)wb[12] * x3.x + (float)wb[13] * x3.y + (float)wb[14] * x3.z + (float)wb[15] * x3.w;
        acc += ssc[base >> 5] * s;
    }
    for (uint32_t s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s);
    uint32_t warp = tid >> 5, lane = tid & 31u;
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        float v = 0.0f;
        uint32_t nwarps = (nth + 31u) >> 5;
        for (uint32_t w = 0; w < nwarps; ++w) v += wsum[w];
        if (bias) v += bias[o];
        y[o] = v;
    }
}

PD_EXPORT
int pd_q8_0_gemv_repacked(const void* data, const void* scale, const void* bias,
                          const void* x, void* y, uint32_t in_dim, uint32_t out_dim,
                          void* stream) {
    if (out_dim == 0) return 0;
    // 128, not 256: this GPU's maxThreadsPerMultiProcessor is 1536 (Ada-style
    // SM, not the 2048 of Ampere/Hopper/datacenter-Blackwell), so a 256-wide
    // block caps at 6 resident blocks/SM (also REG:40-bound at the same 6 --
    // profiler-confirmed, both ceilings coincide). At 128 threads the same
    // kernel (already fully generic on blockDim.x) gets 12 blocks/SM instead
    // -- more independent DRAM streams for this bandwidth-bound GEMV.
    // Measured on granite-4.1-8b's real decode shapes (sm_120a,
    // DRAM-cold min-of-5): q/o and k/v ties (k/v is grid-starved at only
    // 1024 blocks regardless of per-SM occupancy), gate/up +12.1%
    // (25.65->22.53 us), down +8.9% (24.58->22.57 us) -- no shape regressed.
    // 512 threads (3 blk/SM) is worse everywhere; not swept further since
    // 128 already ties or beats every narrower width tested.
    uint32_t threads = 128;
    uint32_t shmem = (in_dim >> 5) * sizeof(float);
    pd_pdl_go(pd_q8_0_gemv_repacked_kernel, out_dim, threads, shmem, (cudaStream_t)stream,
        (const int8_t*)data, (const __half*)scale, (const float*)bias, (const float*)x,
        (float*)y, in_dim, out_dim);
    return pd_launch_status();
}

// ---- K-SPLIT Q8_0 GEMV/GEMM for NARROW-OUT planes (slot 588) --------------
// One block per output row is the right shape until the row count stops
// filling the die: the hyper-connection down plane is [in 10240, out 320], so
// the plain GEMV launches 320 blocks on a 148-SM machine (2 per SM against
// the 12 it can hold) and the batched mt kernel is worse - its 16-row tile
// makes TWENTY blocks. The plane is 3.5 MB and the walk runs 96 of these a
// tick, which is how a bandwidth-shaped kernel ended up at ~175 GB/s.
//
// Here the row's dot is split over `split` blocks of K, each accumulating its
// own chunk; the last block to finish a row folds the partials in ASCENDING
// split order and writes. Deterministic (fixed fold order, counters reset to
// zero so a captured graph replays identically), and the per-chunk math is
// the plain kernel's - only the outer sum is regrouped, the same class the
// f32 split-K matvec already carries.
__global__ void pd_q8_0_gemv_sk_kernel(
    const int8_t* __restrict__ data, const __half* __restrict__ scale,
    const float* __restrict__ bias, const float* __restrict__ x,
    float* __restrict__ y, float* __restrict__ partials,
    unsigned int* __restrict__ counters, uint32_t in_dim, uint32_t out_dim,
    uint32_t split) {
    const uint32_t o = blockIdx.x, sp = blockIdx.y, b = blockIdx.z;
    if (o >= out_dim) return;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    const uint32_t n_blocks = in_dim >> 5;
    // 32-aligned chunks: a 16-element thread chunk then lies wholly inside one
    // Q8_0 block, exactly as in the plain kernel, so the scale lookup is the
    // same single shared read.
    const uint32_t cblocks = (n_blocks + split - 1u) / split;
    const uint32_t k0 = sp * cblocks * 32u;
    const uint32_t k1 = min(k0 + cblocks * 32u, in_dim);
    extern __shared__ float ssc[];
    const __half* srow = scale + (size_t)o * n_blocks;
    for (uint32_t i = tid; k0 + i * 32u < k1; i += nth)
        ssc[i] = __half2float(srow[(k0 >> 5) + i]);
    PD_PDL_ARM();
    __shared__ float wsum[32];
    __syncthreads();
    const int8_t* row = data + (size_t)o * in_dim;
    const float* xr = x + (size_t)b * in_dim;
    float acc = 0.0f;
    for (uint32_t base = k0 + tid * 16u; base < k1; base += nth * 16u) {
        int4 wv = *reinterpret_cast<const int4*>(row + base);
        const int8_t* wb = reinterpret_cast<const int8_t*>(&wv);
        float4 x0 = *reinterpret_cast<const float4*>(xr + base);
        float4 x1 = *reinterpret_cast<const float4*>(xr + base + 4);
        float4 x2 = *reinterpret_cast<const float4*>(xr + base + 8);
        float4 x3 = *reinterpret_cast<const float4*>(xr + base + 12);
        float s = (float)wb[0] * x0.x + (float)wb[1] * x0.y + (float)wb[2] * x0.z + (float)wb[3] * x0.w
                + (float)wb[4] * x1.x + (float)wb[5] * x1.y + (float)wb[6] * x1.z + (float)wb[7] * x1.w
                + (float)wb[8] * x2.x + (float)wb[9] * x2.y + (float)wb[10] * x2.z + (float)wb[11] * x2.w
                + (float)wb[12] * x3.x + (float)wb[13] * x3.y + (float)wb[14] * x3.z + (float)wb[15] * x3.w;
        acc += ssc[(base - k0) >> 5] * s;
    }
    for (uint32_t s2 = 16; s2 > 0; s2 >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s2);
    const uint32_t warp = tid >> 5, lane = tid & 31u;
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        float v = 0.0f;
        const uint32_t nwarps = (nth + 31u) >> 5;
        for (uint32_t w = 0; w < nwarps; ++w) v += wsum[w];
        const size_t oi = (size_t)b * out_dim + o;
        partials[oi * split + sp] = v;
        __threadfence();
        const unsigned int prev = atomicAdd(&counters[oi], 1u);
        if (prev == split - 1u) {
            float sum = 0.0f;
            for (uint32_t t = 0; t < split; ++t) sum += partials[oi * split + t];
            if (bias) sum += bias[o];
            y[oi] = sum;
            counters[oi] = 0u;   // graph-replay safe: back to the initial state
        }
    }
}

PD_EXPORT
int pd_q8_0_gemv_sk(const void* data, const void* scale, const void* bias,
                    const void* x, void* y, void* partials, void* counters,
                    uint32_t in_dim, uint32_t out_dim, uint32_t batch,
                    uint32_t split, void* stream) {
    if (out_dim == 0 || batch == 0) return 0;
    if (split < 2u || split > 32u) return cudaErrorInvalidValue;
    if (in_dim & 31u) return cudaErrorInvalidValue;
    const uint32_t n_blocks = in_dim >> 5;
    if (split > n_blocks) return cudaErrorInvalidValue;
    const uint32_t threads = 128;
    const uint32_t cblocks = (n_blocks + split - 1u) / split;
    const uint32_t shmem = cblocks * (uint32_t)sizeof(float);
    dim3 grid(out_dim, split, batch);
    pd_pdl_go(pd_q8_0_gemv_sk_kernel, grid, threads, shmem, (cudaStream_t)stream,
        (const int8_t*)data, (const __half*)scale, (const float*)bias,
        (const float*)x, (float*)y, (float*)partials, (unsigned int*)counters,
        in_dim, out_dim, split);
    return pd_launch_status();
}

// Multi-segment sibling of pd_q8_0_gemv_repacked: one launch covers up to
// three same-in_dim weight planes sharing one activation vector (decode QKV:
// 4096+1024+1024 rows; FFN gate|up: 12800+12800). Each block resolves which
// segment its blockIdx.x row belongs to (<=2 uniform compares) and then runs
// the exact single-plane body -- same 16-elem chunks, same reduction tree, so
// every output value is bit-identical to the split launches. What the merge
// buys is launch-boundary economics, measured on gran8b_q8_gemv_bench.cu
// (DRAM-cold, sm_120a): a 1024-row launch streams at only 724 GB/s (the grid
// can't cover the die's ramp/drain) and a 4096-row one at 1254, while the
// merged 6144-row grid runs 1303 GB/s -- separate q+k+v 26.5 us vs merged
// 20.5 us, ~6 us/layer back. The die's practical read ceiling is 1531 GB/s
// (ECC-on GDDR7, dram_ceiling probe) -- big-grid launches already sit at
// 93-99% of it, so fewer+bigger launches is the remaining lever, not a
// different inner loop.
struct PdQ8GemvSeg {
    const int8_t* data;
    const __half* scale;
    const float* bias;
    float* y;
    uint32_t out_dim;
};
struct PdQ8GemvSegs3 { PdQ8GemvSeg s[3]; };

__global__ void pd_q8_0_gemv_repacked_multi_kernel(
    PdQ8GemvSegs3 segs, const float* __restrict__ x, uint32_t in_dim, uint32_t n_segs) {
    // Segment resolve must use compile-time struct indices: a runtime
    // `segs.s[si]` forces the whole by-value struct onto STACK (local
    // memory, STACK:120) and every pointer access becomes an LDL round-trip
    // - measured -19%/-42% vs the split launches before this if-chain kept
    // the accesses in param space (STACK:0). n_segs is encoded in the grid
    // (unused segments carry out_dim 0), so the chain needs no count check.
    uint32_t o = blockIdx.x;
    const int8_t* __restrict__ data;
    const __half* __restrict__ scale;
    const float* bias;
    float* __restrict__ y;
    if (o < segs.s[0].out_dim) {
        data = segs.s[0].data; scale = segs.s[0].scale;
        bias = segs.s[0].bias; y = segs.s[0].y;
    } else if (o < segs.s[0].out_dim + segs.s[1].out_dim) {
        o -= segs.s[0].out_dim;
        data = segs.s[1].data; scale = segs.s[1].scale;
        bias = segs.s[1].bias; y = segs.s[1].y;
    } else {
        o -= segs.s[0].out_dim + segs.s[1].out_dim;
        data = segs.s[2].data; scale = segs.s[2].scale;
        bias = segs.s[2].bias; y = segs.s[2].y;
    }
    uint32_t tid = threadIdx.x, nth = blockDim.x;
    uint32_t n_blocks = in_dim >> 5;
    extern __shared__ float ssc[];
    const __half* srow = scale + (size_t)o * n_blocks;
    for (uint32_t b = tid; b < n_blocks; b += nth) ssc[b] = __half2float(srow[b]);
    // dep-free W prologue above, wait before the x reads (see single-plane twin)
    PD_PDL_ARM();
    __shared__ float wsum[32];
    __syncthreads();
    const int8_t* row = data + (size_t)o * in_dim;
    float acc = 0.0f;
    for (uint32_t base = tid * 16u; base < in_dim; base += nth * 16u) {
        int4 wv = *reinterpret_cast<const int4*>(row + base);
        const int8_t* wb = reinterpret_cast<const int8_t*>(&wv);
        float4 x0 = *reinterpret_cast<const float4*>(x + base);
        float4 x1 = *reinterpret_cast<const float4*>(x + base + 4);
        float4 x2 = *reinterpret_cast<const float4*>(x + base + 8);
        float4 x3 = *reinterpret_cast<const float4*>(x + base + 12);
        float s = (float)wb[0] * x0.x + (float)wb[1] * x0.y + (float)wb[2] * x0.z + (float)wb[3] * x0.w
                + (float)wb[4] * x1.x + (float)wb[5] * x1.y + (float)wb[6] * x1.z + (float)wb[7] * x1.w
                + (float)wb[8] * x2.x + (float)wb[9] * x2.y + (float)wb[10] * x2.z + (float)wb[11] * x2.w
                + (float)wb[12] * x3.x + (float)wb[13] * x3.y + (float)wb[14] * x3.z + (float)wb[15] * x3.w;
        acc += ssc[base >> 5] * s;
    }
    for (uint32_t s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s);
    uint32_t warp = tid >> 5, lane = tid & 31u;
    if (lane == 0) wsum[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        float v = 0.0f;
        uint32_t nwarps = (nth + 31u) >> 5;
        for (uint32_t w = 0; w < nwarps; ++w) v += wsum[w];
        if (bias) v += bias[o];
        y[o] = v;
    }
}

PD_EXPORT
int pd_q8_0_gemv_repacked_multi(
    const void* d0, const void* s0, const void* b0, void* y0, uint32_t r0,
    const void* d1, const void* s1, const void* b1, void* y1, uint32_t r1,
    const void* d2, const void* s2, const void* b2, void* y2, uint32_t r2,
    const void* x, uint32_t in_dim, uint32_t n_segs, void* stream) {
    if (n_segs == 0 || n_segs > 3) return (int)cudaErrorInvalidValue;
    PdQ8GemvSegs3 segs{};
    segs.s[0] = {(const int8_t*)d0, (const __half*)s0, (const float*)b0, (float*)y0, r0};
    segs.s[1] = {(const int8_t*)d1, (const __half*)s1, (const float*)b1, (float*)y1, r1};
    segs.s[2] = {(const int8_t*)d2, (const __half*)s2, (const float*)b2, (float*)y2, r2};
    uint32_t total = r0 + (n_segs > 1 ? r1 : 0) + (n_segs > 2 ? r2 : 0);
    if (total == 0) return 0;
    // 128 threads: the width the single-plane launcher elected (see its note)
    uint32_t threads = 128;
    uint32_t shmem = (in_dim >> 5) * sizeof(float);
    pd_pdl_go(pd_q8_0_gemv_repacked_multi_kernel, total, threads, shmem, (cudaStream_t)stream,
        segs, (const float*)x, in_dim, n_segs);
    return pd_launch_status();
}

// Fused FFN gate+up+SwiGLU over the repacked Q8_0 layout: for each output o in
// [0, ff), out[o] = silu(dot(gate[o], x)) * dot(up[o], x), with silu(g)=g/(1+e^-g).
// One block per o computes both dot products (reading x once) and the SwiGLU,
// replacing two GEMV launches + a swiglu launch + two intermediate buffers. Same
// vectorized int4 weight / float4 activation loads and f32 accumulate as the plain
// repacked GEMV, so it stays f32-exact; the longer per-block work also amortizes the
// medium-shape ramp/drain the split GEMVs paid (profiled: ffn was ~90% DRAM, not 97%).
__global__ void pd_q8_0_ffn_gate_up_swiglu_kernel(
    const int8_t* __restrict__ gate_data, const __half* __restrict__ gate_scale,
    const int8_t* __restrict__ up_data, const __half* __restrict__ up_scale,
    const float* __restrict__ x, float* __restrict__ out, uint32_t in_dim, uint32_t ff) {
    uint32_t o = blockIdx.x;
    if (o >= ff) return;
    uint32_t tid = threadIdx.x, nth = blockDim.x;
    uint32_t n_blocks = in_dim >> 5;
    extern __shared__ float ssc[];   // [0,n_blocks) gate scales, [n_blocks,2n) up scales
    float* gsc = ssc;
    float* usc = ssc + n_blocks;
    const __half* gsrow = gate_scale + (size_t)o * n_blocks;
    const __half* usrow = up_scale + (size_t)o * n_blocks;
    for (uint32_t b = tid; b < n_blocks; b += nth) {
        gsc[b] = __half2float(gsrow[b]);
        usc[b] = __half2float(usrow[b]);
    }
    __shared__ float wsum_g[32], wsum_u[32];
    __syncthreads();
    const int8_t* grow = gate_data + (size_t)o * in_dim;
    const int8_t* urow = up_data + (size_t)o * in_dim;
    float accg = 0.0f, accu = 0.0f;
    for (uint32_t base = tid * 16u; base < in_dim; base += nth * 16u) {
        int4 gv = *reinterpret_cast<const int4*>(grow + base);
        int4 uv = *reinterpret_cast<const int4*>(urow + base);
        const int8_t* gb = reinterpret_cast<const int8_t*>(&gv);
        const int8_t* ub = reinterpret_cast<const int8_t*>(&uv);
        float4 x0 = *reinterpret_cast<const float4*>(x + base);
        float4 x1 = *reinterpret_cast<const float4*>(x + base + 4);
        float4 x2 = *reinterpret_cast<const float4*>(x + base + 8);
        float4 x3 = *reinterpret_cast<const float4*>(x + base + 12);
        float sg = (float)gb[0] * x0.x + (float)gb[1] * x0.y + (float)gb[2] * x0.z + (float)gb[3] * x0.w
                 + (float)gb[4] * x1.x + (float)gb[5] * x1.y + (float)gb[6] * x1.z + (float)gb[7] * x1.w
                 + (float)gb[8] * x2.x + (float)gb[9] * x2.y + (float)gb[10] * x2.z + (float)gb[11] * x2.w
                 + (float)gb[12] * x3.x + (float)gb[13] * x3.y + (float)gb[14] * x3.z + (float)gb[15] * x3.w;
        float su = (float)ub[0] * x0.x + (float)ub[1] * x0.y + (float)ub[2] * x0.z + (float)ub[3] * x0.w
                 + (float)ub[4] * x1.x + (float)ub[5] * x1.y + (float)ub[6] * x1.z + (float)ub[7] * x1.w
                 + (float)ub[8] * x2.x + (float)ub[9] * x2.y + (float)ub[10] * x2.z + (float)ub[11] * x2.w
                 + (float)ub[12] * x3.x + (float)ub[13] * x3.y + (float)ub[14] * x3.z + (float)ub[15] * x3.w;
        accg += gsc[base >> 5] * sg;
        accu += usc[base >> 5] * su;
    }
    for (uint32_t s = 16; s > 0; s >>= 1) {
        accg += __shfl_down_sync(0xffffffffu, accg, s);
        accu += __shfl_down_sync(0xffffffffu, accu, s);
    }
    uint32_t warp = tid >> 5, lane = tid & 31u;
    if (lane == 0) { wsum_g[warp] = accg; wsum_u[warp] = accu; }
    __syncthreads();
    if (tid == 0) {
        float g = 0.0f, u = 0.0f;
        uint32_t nwarps = (nth + 31u) >> 5;
        for (uint32_t w = 0; w < nwarps; ++w) { g += wsum_g[w]; u += wsum_u[w]; }
        out[o] = (g / (1.0f + expf(-g))) * u;   // silu(g) * u, matches pd_swiglu
    }
}

PD_EXPORT
int pd_q8_0_ffn_gate_up_swiglu(const void* gate_data, const void* gate_scale,
                               const void* up_data, const void* up_scale, const void* x,
                               void* out, uint32_t in_dim, uint32_t ff, void* stream) {
    if (ff == 0) return 0;
    uint32_t threads = 256;
    size_t shmem = (size_t)(in_dim >> 5) * 2 * sizeof(float);
    pd_q8_0_ffn_gate_up_swiglu_kernel<<<ff, threads, shmem, (cudaStream_t)stream>>>(
        (const int8_t*)gate_data, (const __half*)gate_scale, (const int8_t*)up_data,
        (const __half*)up_scale, (const float*)x, (float*)out, in_dim, ff);
    return pd_launch_status();
}

// Fused DeltaNet alpha+beta projection + gate (decode): for each value head o,
// a = dot(alpha[o], x), b = dot(beta[o], x), then beta_out[o] = sigmoid(b) and
// g[o] = ssm_a[o] * softplus(a + dt_bias[o]). Collapses two skinny GEMVs (out =
// n_heads) + delta_gate into one launch. These GEMVs are LATENCY-bound (~3%
// DRAM, out=32 -> only 32 blocks), so - unlike the memory-bound ffn - fusing their
// launches cuts pure launch/latency overhead without starving DRAM. f32-exact;
// alpha/beta in the repacked Q8_0 layout. One block per head.
__global__ void pd_deltanet_alpha_beta_gate_kernel(
    const int8_t* __restrict__ a_data, const __half* __restrict__ a_scale,
    const int8_t* __restrict__ b_data, const __half* __restrict__ b_scale,
    const float* __restrict__ x, const float* __restrict__ ssm_a,
    const float* __restrict__ dt_bias, float* __restrict__ g, float* __restrict__ beta,
    uint32_t in_dim, uint32_t n_heads) {
    uint32_t o = blockIdx.x;
    if (o >= n_heads) return;
    uint32_t tid = threadIdx.x, nth = blockDim.x;
    uint32_t n_blocks = in_dim >> 5;
    extern __shared__ float ssc[];   // [0,n_blocks) alpha scales, [n_blocks,2n) beta
    float* asc = ssc;
    float* bsc = ssc + n_blocks;
    const __half* asrow = a_scale + (size_t)o * n_blocks;
    const __half* bsrow = b_scale + (size_t)o * n_blocks;
    for (uint32_t bl = tid; bl < n_blocks; bl += nth) {
        asc[bl] = __half2float(asrow[bl]);
        bsc[bl] = __half2float(bsrow[bl]);
    }
    __shared__ float wsa[32], wsb[32];
    __syncthreads();
    const int8_t* arow = a_data + (size_t)o * in_dim;
    const int8_t* brow = b_data + (size_t)o * in_dim;
    float acca = 0.0f, accb = 0.0f;
    for (uint32_t base = tid * 16u; base < in_dim; base += nth * 16u) {
        int4 av = *reinterpret_cast<const int4*>(arow + base);
        int4 bv = *reinterpret_cast<const int4*>(brow + base);
        const int8_t* ab = reinterpret_cast<const int8_t*>(&av);
        const int8_t* bb = reinterpret_cast<const int8_t*>(&bv);
        float4 x0 = *reinterpret_cast<const float4*>(x + base);
        float4 x1 = *reinterpret_cast<const float4*>(x + base + 4);
        float4 x2 = *reinterpret_cast<const float4*>(x + base + 8);
        float4 x3 = *reinterpret_cast<const float4*>(x + base + 12);
        float sa = (float)ab[0] * x0.x + (float)ab[1] * x0.y + (float)ab[2] * x0.z + (float)ab[3] * x0.w
                 + (float)ab[4] * x1.x + (float)ab[5] * x1.y + (float)ab[6] * x1.z + (float)ab[7] * x1.w
                 + (float)ab[8] * x2.x + (float)ab[9] * x2.y + (float)ab[10] * x2.z + (float)ab[11] * x2.w
                 + (float)ab[12] * x3.x + (float)ab[13] * x3.y + (float)ab[14] * x3.z + (float)ab[15] * x3.w;
        float sb = (float)bb[0] * x0.x + (float)bb[1] * x0.y + (float)bb[2] * x0.z + (float)bb[3] * x0.w
                 + (float)bb[4] * x1.x + (float)bb[5] * x1.y + (float)bb[6] * x1.z + (float)bb[7] * x1.w
                 + (float)bb[8] * x2.x + (float)bb[9] * x2.y + (float)bb[10] * x2.z + (float)bb[11] * x2.w
                 + (float)bb[12] * x3.x + (float)bb[13] * x3.y + (float)bb[14] * x3.z + (float)bb[15] * x3.w;
        acca += asc[base >> 5] * sa;
        accb += bsc[base >> 5] * sb;
    }
    for (uint32_t s = 16; s > 0; s >>= 1) {
        acca += __shfl_down_sync(0xffffffffu, acca, s);
        accb += __shfl_down_sync(0xffffffffu, accb, s);
    }
    uint32_t warp = tid >> 5, lane = tid & 31u;
    if (lane == 0) { wsa[warp] = acca; wsb[warp] = accb; }
    __syncthreads();
    if (tid == 0) {
        float av = 0.0f, bv = 0.0f;
        uint32_t nwarps = (nth + 31u) >> 5;
        for (uint32_t w = 0; w < nwarps; ++w) { av += wsa[w]; bv += wsb[w]; }
        beta[o] = 1.0f / (1.0f + expf(-bv));
        float ax = av + dt_bias[o];
        float sp = fmaxf(ax, 0.0f) + log1pf(expf(-fabsf(ax)));   // softplus, matches delta_gate
        g[o] = ssm_a[o] * sp;
    }
}

PD_EXPORT
int pd_deltanet_alpha_beta_gate(const void* a_data, const void* a_scale, const void* b_data,
                                const void* b_scale, const void* x, const void* ssm_a,
                                const void* dt_bias, void* g, void* beta, uint32_t in_dim,
                                uint32_t n_heads, void* stream) {
    if (n_heads == 0) return 0;
    uint32_t threads = 256;
    size_t shmem = (size_t)(in_dim >> 5) * 2 * sizeof(float);
    pd_deltanet_alpha_beta_gate_kernel<<<n_heads, threads, shmem, (cudaStream_t)stream>>>(
        (const int8_t*)a_data, (const __half*)a_scale, (const int8_t*)b_data,
        (const __half*)b_scale, (const float*)x, (const float*)ssm_a, (const float*)dt_bias,
        (float*)g, (float*)beta, in_dim, n_heads);
    return pd_launch_status();
}

// Batched GEMM over the repacked Q8_0 layout - the prefill workhorse. One block per
// output row; the batch is tiled (PD_GEMM_RTILE rows per weight pass) so the weight
// row is read once per tile instead of once per token: prefill T tokens costs
// ceil(T/16) weight reads, not T. Same vectorized int4 weight / float4 activation
// loads, chunk order, and scale-factored 16-term inner product as
// pd_q8_0_gemv_repacked - at batch=1 the per-token math is bit-identical to the
// decode GEMV, so prefill logits match the incremental path exactly.
#define PD_GEMM_RTILE 16
__global__ void __launch_bounds__(256) pd_q8_0_gemm_repacked_kernel(
    const int8_t* __restrict__ data, const __half* __restrict__ scale,
    const float* __restrict__ bias, const float* __restrict__ x, float* __restrict__ y,
    uint32_t in_dim, uint32_t out_dim, uint32_t batch) {
    uint32_t o = blockIdx.x;
    if (o >= out_dim) return;
    uint32_t tid = threadIdx.x, nth = blockDim.x;
    uint32_t n_blocks = in_dim >> 5;
    extern __shared__ float ssc[];
    __shared__ float red[32];
    uint32_t lane = tid & 31u, warp = tid >> 5, nwarps = (nth + 31u) >> 5;
    const __half* srow = scale + (size_t)o * n_blocks;
    for (uint32_t b = tid; b < n_blocks; b += nth) ssc[b] = __half2float(srow[b]);
    __syncthreads();
    const int8_t* row = data + (size_t)o * in_dim;

    // batch tiles live on grid.y (P6g): out_dim can be tiny (alpha/beta =
    // n_v_heads rows) while batch is 512, so a sequential tile loop left the
    // GPU near-idle (32 blocks, 467 us). One tile per block is the same
    // per-element math in the same order - bit-exact with the loop version -
    // just parallel. Weight re-reads (ceil(batch/16) per row) only matter for
    // big-out_dim weights, which take the mmq path at these batches anyway.
    {
        uint32_t b0 = blockIdx.y * PD_GEMM_RTILE;
        if (b0 >= batch) return;
        uint32_t tb = (batch - b0 < PD_GEMM_RTILE) ? (batch - b0) : PD_GEMM_RTILE;
        float acc[PD_GEMM_RTILE];
#pragma unroll
        for (uint32_t t = 0; t < PD_GEMM_RTILE; ++t) acc[t] = 0.0f;
        for (uint32_t base = tid * 16u; base < in_dim; base += nth * 16u) {
            int4 wv = *reinterpret_cast<const int4*>(row + base);
            const int8_t* wb = reinterpret_cast<const int8_t*>(&wv);
            float sc = ssc[base >> 5];
            // compile-time trip count keeps acc[] in registers (a runtime bound
            // would spill the array to local memory, scaling cost with the batch)
#pragma unroll
            for (uint32_t t = 0; t < PD_GEMM_RTILE; ++t) {
                if (t >= tb) break;
                const float* xt = x + (size_t)(b0 + t) * in_dim + base;
                float4 x0 = *reinterpret_cast<const float4*>(xt);
                float4 x1 = *reinterpret_cast<const float4*>(xt + 4);
                float4 x2 = *reinterpret_cast<const float4*>(xt + 8);
                float4 x3 = *reinterpret_cast<const float4*>(xt + 12);
                float s = (float)wb[0] * x0.x + (float)wb[1] * x0.y + (float)wb[2] * x0.z + (float)wb[3] * x0.w
                        + (float)wb[4] * x1.x + (float)wb[5] * x1.y + (float)wb[6] * x1.z + (float)wb[7] * x1.w
                        + (float)wb[8] * x2.x + (float)wb[9] * x2.y + (float)wb[10] * x2.z + (float)wb[11] * x2.w
                        + (float)wb[12] * x3.x + (float)wb[13] * x3.y + (float)wb[14] * x3.z + (float)wb[15] * x3.w;
                acc[t] += sc * s;
            }
        }
#pragma unroll
        for (uint32_t t = 0; t < PD_GEMM_RTILE; ++t) {
            if (t >= tb) break;
            float a = acc[t];
            for (uint32_t s = 16; s > 0; s >>= 1) a += __shfl_down_sync(0xffffffffu, a, s);
            if (lane == 0) red[warp] = a;
            __syncthreads();
            if (tid == 0) {
                float v = 0.0f;
                for (uint32_t w = 0; w < nwarps; ++w) v += red[w];
                if (bias) v += bias[o];
                y[(size_t)(b0 + t) * out_dim + o] = v;
            }
            __syncthreads();
        }
    }
}

PD_EXPORT
int pd_q8_0_gemm_repacked(const void* data, const void* scale, const void* bias,
                          const void* x, void* y, uint32_t in_dim, uint32_t out_dim,
                          uint32_t batch, void* stream) {
    if (out_dim == 0 || batch == 0) return 0;
    uint32_t threads = 256;
    size_t shmem = (size_t)(in_dim >> 5) * sizeof(float);
    dim3 grid(out_dim, (batch + PD_GEMM_RTILE - 1u) / PD_GEMM_RTILE);
    pd_q8_0_gemm_repacked_kernel<<<grid, threads, shmem, (cudaStream_t)stream>>>(
        (const int8_t*)data, (const __half*)scale, (const float*)bias, (const float*)x,
        (float*)y, in_dim, out_dim, batch);
    return pd_launch_status();
}

// Two-weight fused twin of the repacked GEMM, for the alpha/beta pair (the
// exact-f32 P6b decay projections, out_dim = n_v_heads = 32 each). The 2D
// kernel above re-reads every x row once per OUTPUT (out_dim blocks share a
// batch tile): at r=2056 that is ~540 MB of L2 per call, x2 calls = 6.3% of
// the unified pf8 profile for 64 output columns. This variant stages a 4-row
// x tile in shared once and loops all outputs of both weights over it - x is
// read once total (~17 MB), weights (tiny) re-read per tile.
// BIT-EXACT per output: same 128-active-thread chunk mapping (base = tid*16,
// stride nth*16), same 16-term inner product, same shuffle tree, and the same
// w-ascending serial cross-warp sum - x simply arrives via shared memory.
#define PD_GEMM_X2_RT 4
__global__ void __launch_bounds__(256) pd_q8_0_gemm_repacked_x2_kernel(
    const int8_t* __restrict__ da, const __half* __restrict__ sa,
    const int8_t* __restrict__ db, const __half* __restrict__ sb,
    const float* __restrict__ x, float* __restrict__ ya, float* __restrict__ yb,
    uint32_t in_dim, uint32_t oda, uint32_t odb, uint32_t batch) {
    constexpr uint32_t RT = PD_GEMM_X2_RT;
    const uint32_t tid = threadIdx.x, nth = blockDim.x;
    const uint32_t lane = tid & 31u, warp = tid >> 5, nwarps = (nth + 31u) >> 5;
    const uint32_t b0 = blockIdx.x * RT;
    if (b0 >= batch) return;
    const uint32_t tb = (batch - b0 < RT) ? (batch - b0) : RT;
    extern __shared__ float sx[];  // [RT][in_dim]
    // v1 shape (per-output serial epilogue). Two grouped rewrites (OG=8
    // batched epilogues, with and without register staging of the x chunk)
    // both REGRESSED - 245/302 us vs 155 - the OGxRT unroll pins 128 regs and
    // the occupancy loss beats the barrier savings. The real v3 is a class
    // change: dequant the alpha/beta planes to f32 at load and reuse the
    // tiled router GEMM (~30 us shape) - needs a PPL gate (order change).
    __shared__ float red[8][RT];
    for (uint32_t u = tid; u < tb * (in_dim >> 2); u += nth) {
        const uint32_t t = u / (in_dim >> 2), q4 = (u % (in_dim >> 2)) * 4u;
        *reinterpret_cast<float4*>(&sx[t * in_dim + q4]) =
            *reinterpret_cast<const float4*>(x + (size_t)(b0 + t) * in_dim + q4);
    }
    __syncthreads();
    const uint32_t od = oda + odb;
    for (uint32_t o = 0; o < od; ++o) {
        const int8_t* row =
            o < oda ? da + (size_t)o * in_dim : db + (size_t)(o - oda) * in_dim;
        const __half* srow = (o < oda ? sa + (size_t)o * (in_dim >> 5)
                                      : sb + (size_t)(o - oda) * (in_dim >> 5));
        float acc[RT];
#pragma unroll
        for (uint32_t t = 0; t < RT; ++t) acc[t] = 0.0f;
        for (uint32_t base = tid * 16u; base < in_dim; base += nth * 16u) {
            int4 wv = *reinterpret_cast<const int4*>(row + base);
            const int8_t* wb = reinterpret_cast<const int8_t*>(&wv);
            float sc = __half2float(srow[base >> 5]);
#pragma unroll
            for (uint32_t t = 0; t < RT; ++t) {
                if (t >= tb) break;
                const float* xt = sx + (size_t)t * in_dim + base;
                float4 x0 = *reinterpret_cast<const float4*>(xt);
                float4 x1 = *reinterpret_cast<const float4*>(xt + 4);
                float4 x2 = *reinterpret_cast<const float4*>(xt + 8);
                float4 x3 = *reinterpret_cast<const float4*>(xt + 12);
                float s = (float)wb[0] * x0.x + (float)wb[1] * x0.y + (float)wb[2] * x0.z + (float)wb[3] * x0.w
                        + (float)wb[4] * x1.x + (float)wb[5] * x1.y + (float)wb[6] * x1.z + (float)wb[7] * x1.w
                        + (float)wb[8] * x2.x + (float)wb[9] * x2.y + (float)wb[10] * x2.z + (float)wb[11] * x2.w
                        + (float)wb[12] * x3.x + (float)wb[13] * x3.y + (float)wb[14] * x3.z + (float)wb[15] * x3.w;
                acc[t] += sc * s;
            }
        }
#pragma unroll
        for (uint32_t t = 0; t < RT; ++t) {
            float a = acc[t];
            for (uint32_t s = 16; s > 0; s >>= 1)
                a += __shfl_down_sync(0xffffffffu, a, s);
            if (lane == 0) red[warp][t] = a;
        }
        __syncthreads();
        if (tid < tb) {
            float v = 0.0f;
            for (uint32_t w = 0; w < nwarps; ++w) v += red[w][tid];
            float* yp = o < oda ? ya + (size_t)(b0 + tid) * oda + o
                                : yb + (size_t)(b0 + tid) * odb + (o - oda);
            *yp = v;
        }
        __syncthreads();
    }
}

PD_EXPORT
int pd_q8_0_gemm_repacked_x2(const void* da, const void* sa, const void* db,
                             const void* sb, const void* x, void* ya, void* yb,
                             uint32_t in_dim, uint32_t oda, uint32_t odb,
                             uint32_t batch, void* stream) {
    if (batch == 0 || (oda + odb) == 0) return 0;
    // Opt into the ACTUAL requested smem (= 4 * in_dim * f32 = 16 * n_embd bytes),
    // not a fixed 64 KB. The old hardcode happened to land exactly on n_embd=4096's
    // 64 KB and silently broke any larger hidden size (e.g. qwen3.6-27b's n_embd=5120
    // -> 80 KB) with a launch-time CUDA invalid-value (error 1). Grow the opt-in
    // whenever a bigger shmem shows up. NB: still bounded by the device's per-block
    // opt-in cap (~99 KB on sm_86/sm_120, ~227 KB on sm_90/sm_100); a hidden size
    // past ~6.2k would need retiling, not just a bigger opt-in.
    const size_t shmem = (size_t)PD_GEMM_X2_RT * in_dim * sizeof(float);
    static uint32_t x2_optin = 0;
    if ((uint32_t)shmem > x2_optin) {
        cudaFuncSetAttribute((const void*)pd_q8_0_gemm_repacked_x2_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
        x2_optin = (uint32_t)shmem;
    }
    dim3 grid((batch + PD_GEMM_X2_RT - 1u) / PD_GEMM_X2_RT);
    pd_q8_0_gemm_repacked_x2_kernel<<<grid, 256, shmem, (cudaStream_t)stream>>>(
        (const int8_t*)da, (const __half*)sa, (const int8_t*)db, (const __half*)sb,
        (const float*)x, (float*)ya, (float*)yb, in_dim, oda, odb, batch);
    return pd_launch_status();
}

// ---- conv-window VL store  -----------------------------------
// Copy each fresh span's last (k-1) pre-conv rows into its slot's conv
// window in one launch per layer - the VL twin of prefill_batch_pass's
// per-share copy_region (48 Linear layers x n_sh copies collapse to 48
// launches), and the store variant whose span geometry lives in device
// contents, so a captured pass graph can bake only the padded bucket shape
// while true lengths ride the buffer. Bit-identical bytes to the
// copy_region pair it replaces.
// spans: stride-4 u32 (row0, take, slot, _); every span has take >= k-1
// (the partial-window arm keeps the copy_region path).
__global__ void pd_conv_win_store_vl_kernel(const float* __restrict__ src,
                                            const uint32_t* __restrict__ spans,
                                            float* __restrict__ win,
                                            uint32_t km1, uint32_t conv_dim) {
    const uint32_t s = blockIdx.y;
    const uint32_t rb = spans[s * 4u];
    const uint32_t take = spans[s * 4u + 1u];
    const uint32_t slot = spans[s * 4u + 2u];
    const uint32_t n = km1 * conv_dim;
    const float* sp = src + ((size_t)rb + take - km1) * conv_dim;
    float* dp = win + (size_t)slot * n;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x)
        dp[i] = sp[i];
}

PD_EXPORT
int pd_conv_win_store_vl(const void* src, const void* spans, void* win,
                         uint32_t n_spans, uint32_t km1, uint32_t conv_dim,
                         void* stream) {
    if (n_spans == 0) return 0;
    const uint32_t n = km1 * conv_dim;
    dim3 grid((n + 255u) / 256u, n_spans);
    pd_conv_win_store_vl_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        (const float*)src, (const uint32_t*)spans, (float*)win, km1, conv_dim);
    return pd_launch_status();
}

