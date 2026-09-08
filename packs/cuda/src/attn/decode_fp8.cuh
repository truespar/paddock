// attn/decode_fp8.cuh - the fp8-native decode-attention lane (v8q, v9q,
// v9q2) plus the vdim twin-pool sync and the Laguna sigmoid MoE router.
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
//
// Split out of attn/decode.cuh (see attn/decode_spec.cuh).
//
// v8q is the fp8-NATIVE SWA decode attention, v9q the QGMMA redesign
// that took it fp8 END-TO-END, v9q2 its ST64 handshake-halving twin.
// The Laguna sigmoid MoE router rides along at the tail: it is not
// attention, it simply sat at the end of decode.cuh and moving it to moe/
// would change its position in pack.cu's include order, which is a separate
// question from this split.
//
// Include after attn/decode.cuh and attn/decode_spec.cuh.
// ---- v8q: fp8-NATIVE SWA decode attention --------------------
// The KV8 true-win kernel, first cut: the SCORE side consumes the fp8 K
// tiles directly - Q is cast e4m3 at staging and QK^T runs on
// mma.sync.m16n8k32.e4m3 fragments fed by ldmatrix.b16 over packed fp8
// (recipe proven exact by a probe), so the K expansion disappears
// AND the score mma count halves (8 k32 steps vs 16 k16). The V side keeps
// the v8f8 expansion path (one expander warp -> f16 PV; full fp8 PV needs
// a V transpose - banked as v8q2). Rings/barriers/fold/FIN identical to
// v8. Numerics: v8f8's class plus Q e4m3 rounding (score products are
// exact products of rounded operands in f32 - the reassociation delta vs
// f16 mma is k-chunk order only). Quality gates arbitrate.
// blockDim 288: warps 0-1 score, 2-7 V, 8 = V expander.
template <uint32_t HD, uint32_t G>
__global__ void __launch_bounds__(320, 3) pd_attn_decode_v8q_kernel(
    const __grid_constant__ PdTmap tmk,
    const __grid_constant__ PdTmap tmv,
    const float* __restrict__ q, float* __restrict__ out_o,
    float* __restrict__ out_ml, const unsigned int* __restrict__ positions,
    const unsigned int* __restrict__ slots,
    const uint32_t* __restrict__ block_tables, uint32_t blocks_per_slot,
    uint32_t kv_dim, uint32_t swa_window, uint32_t n_splits, float scale) {
#if PD_ATTN_TMA_OK
    PD_PDL_ARM();
    constexpr uint32_t TILE = 16u;
    const uint32_t kvh = blockIdx.x, b = blockIdx.y, sp = blockIdx.z;
    const uint32_t d = threadIdx.x, nth = blockDim.x;
    const uint32_t warp = d >> 5, lane = d & 31u;
    const uint32_t n_heads = G * gridDim.x;

    const uint32_t pos = positions[b];
    const uint32_t first_pos =
        (swa_window > 0 && pos + 1 > swa_window) ? (pos + 1 - swa_window) : 0;
    const uint32_t n_pos = pos + 1 - first_pos;
    const uint32_t chunk = (n_pos + n_splits - 1) / n_splits;
    const uint32_t lo = sp * chunk;
    uint32_t hi = lo + chunk;
    if (hi > n_pos) hi = n_pos;
    const uint32_t g_lo = first_pos + lo, g_hi = first_pos + hi;
    const uint32_t B0 = g_lo >> 4;
    const uint32_t ntiles = lo < hi ? ((g_hi + 15u) >> 4) - B0 : 0u;
    const uint32_t nw = pos + 1u;

    extern __shared__ __align__(128) unsigned char pd_v8q_smraw[];
    constexpr uint32_t SEGS = (HD * 2u) / 128u;      // f16 V segments
    constexpr uint32_t KVB = SEGS * 2048u;           // f16 V tile bytes
    constexpr uint32_t K8B = KVB / 2u;               // fp8 tile bytes
    constexpr uint32_t q_s = HD + 16u;               // e4m3 Q row stride
    constexpr uint32_t w_s = TILE + 8u;
    constexpr uint32_t NSW = TILE / 8u;              // 2 score warps
    unsigned char* s_kv = pd_v8q_smraw
        + ((1024u - ((uint32_t)__cvta_generic_to_shared(pd_v8q_smraw) & 1023u)) & 1023u);
    unsigned char* s_k8 = s_kv;                      // 2 x K8B raw fp8 K
    unsigned char* s_v8 = s_kv + 2u * K8B;           // 3 x K8B raw fp8 V
    unsigned char* s_v = s_v8 + 3u * K8B;            // 3 x KVB f16 V
    unsigned char* s_q8 = s_v + 3u * KVB;            // 16 x q_s e4m3 Q
    __half* s_wh = (__half*)(s_q8 + 16u * q_s);      // 2 slots x [16][w_s]
    __shared__ float s_m[G], s_l[G], s_corr[2][G];
    __shared__ float s_pmax[NSW][G], s_psum[NSW][G];
    __shared__ __align__(8) uint64_t s_bk[2], s_bv[3], s_bev[3];

    if (d == 0) {
        const uint32_t mk = (uint32_t)__cvta_generic_to_shared(&s_bk[0]);
        const uint32_t mv = (uint32_t)__cvta_generic_to_shared(&s_bv[0]);
        const uint32_t me = (uint32_t)__cvta_generic_to_shared(&s_bev[0]);
        #pragma unroll
        for (uint32_t i = 0; i < 2u; ++i)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(mk + i * 8u));
        #pragma unroll
        for (uint32_t i = 0; i < 3u; ++i) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(mv + i * 8u));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(me + i * 8u));
        }
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    const float* qb = q + (size_t)b * n_heads * HD;
    const uint32_t slot = slots ? slots[b] : b;
    const uint32_t* bt = block_tables + (size_t)slot * blocks_per_slot;

    // Q staging: f32 -> e4m3, padded 16-row space (rows >= G zero)
    for (uint32_t i = d; i < 16u * q_s; i += nth) {
        const uint32_t r = i / q_s, c = i % q_s;
        const float v = (r < G && c < HD)
            ? qb[((size_t)kvh * G + r) * HD + c] : 0.0f;
        s_q8[(size_t)r * q_s + c] = __nv_fp8_e4m3(v).__x;
    }
    for (uint32_t i = d; i < 2u * 16u * w_s; i += nth) s_wh[i] = __half(0.f);
    if (d < G) { s_m[d] = -INFINITY; s_l[d] = 0.0f; }

    // f16-layout swizzle (V consumption path, identical to v8)
    auto sw = [](uint32_t r, uint32_t c) -> uint32_t {
        return ((r >> 3) << 10) + ((r & 7u) << 7) + ((c ^ (r & 7u)) << 4);
    };
    // fp8-raw swizzle: 16B chunks XOR'd within each token's 128B row (the
    // fp8 tensor maps encode SWIZZLE_128B, so TMA lands them this way)
    auto sw8 = [](uint32_t tok, uint32_t chunk16) -> uint32_t {
        return tok * 128u + ((chunk16 ^ (tok & 7u)) << 4);
    };
    auto stage_k = [&](uint32_t bf, uint32_t k) {   // warp 0 only
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(&s_bk[bf]);
        if (lane == 0)
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                         ::"r"(m), "r"(K8B));
        __syncwarp();
        if (lane < SEGS / 2u) {
            const uint32_t blk = bt[B0 + k];
            const int y = (int)(blk * 16u);
            const int x = (int)(kvh * HD + lane * 128u);
            const uint32_t dst = (uint32_t)__cvta_generic_to_shared(
                s_k8 + (size_t)bf * K8B + lane * 2048u);
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                " [%0], [%1, {%2, %3}], [%4];"
                ::"r"(dst), "l"(&tmk), "r"(x), "r"(y), "r"(m) : "memory");
        }
    };
    auto stage_v = [&](uint32_t bf, uint32_t k) {   // warp 2 only
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(&s_bv[bf]);
        if (lane == 0)
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                         ::"r"(m), "r"(K8B));
        __syncwarp();
        if (lane < SEGS / 2u) {
            const uint32_t blk = bt[B0 + k];
            const int y = (int)(blk * 16u);
            const int x = (int)(kvh * HD + lane * 128u);
            const uint32_t dst = (uint32_t)__cvta_generic_to_shared(
                s_v8 + (size_t)bf * K8B + lane * 2048u);
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                " [%0], [%1, {%2, %3}], [%4];"
                ::"r"(dst), "l"(&tmv), "r"(x), "r"(y), "r"(m) : "memory");
        }
    };
    auto bar_wait = [&](uint64_t* bar, uint32_t parity) {
        const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
        asm volatile("{\n\t.reg .pred p;\nW%=:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                     "@!p bra W%=;\n\t}" ::"r"(a), "r"(parity));
    };
    // V expansion (warp 8): raw swizzled fp8 -> the f16 sw() layout the
    // unchanged PV path expects. Chunk id -> (f16 seg s, token r, col c);
    // raw addr applies sw8 at 16 B granularity within the token row.
    auto expand_v = [&](const unsigned char* raw, unsigned char* buf,
                        uint32_t lid, uint32_t nl) {
        constexpr uint32_t NCH = SEGS * 16u * 8u;
        for (uint32_t id = lid; id < NCH; id += nl) {
            const uint32_t s = id >> 7, r = (id >> 3) & 15u, c = id & 7u;
            const uint32_t rc16 = (s & 1u) * 4u + (c >> 1);
            const uint2 v8 = *(const uint2*)(raw + (s >> 1) * 2048u
                                             + sw8(r, rc16) + (c & 1u) * 8u);
            uint4 h4;
            #pragma unroll
            for (uint32_t hh = 0; hh < 2u; ++hh) {
                const uint32_t w = hh == 0 ? v8.x : v8.y;
                ((__half2*)&h4)[hh * 2u] = __half2(__nv_cvt_fp8x2_to_halfraw2(
                    (__nv_fp8x2_storage_t)(w & 0xffffu), __NV_E4M3));
                ((__half2*)&h4)[hh * 2u + 1u] = __half2(__nv_cvt_fp8x2_to_halfraw2(
                    (__nv_fp8x2_storage_t)(w >> 16), __NV_E4M3));
            }
            *(uint4*)(buf + s * 2048u + sw(r, c)) = h4;
        }
    };

    __syncthreads();
    if (ntiles) {
        if (warp == 0) stage_k(0u, 0u);
        if (warp == 2) stage_v(0u, 0u);
    }
    if (ntiles > 1u) {
        if (warp == 0) stage_k(1u, 1u);
        if (warp == 2) stage_v(1u, 1u);
    }

    if (warp < 2u) {
        // ---------- score side: fp8-native QK^T ----------
        uint32_t pk[2] = {0u, 0u};
        float dfr[4];
        const uint32_t p0 = warp * 8u;
        const uint32_t rk = p0 + (lane & 7u);
        const uint32_t rr = lane >> 2;
        auto score_t = [&](uint32_t t) {
            const uint32_t bf = t & 1u;
            bar_wait(&s_bk[bf], pk[bf]); pk[bf] ^= 1u;
            const uint32_t gbase = (B0 + t) * 16u;
            const uint32_t plo = g_lo > gbase ? g_lo - gbase : 0u;
            const uint32_t phi = g_hi - gbase < 16u ? g_hi - gbase : 16u;
            #pragma unroll
            for (uint32_t i = 0; i < 4u; ++i) dfr[i] = 0.f;
            unsigned char* kb = s_k8 + (size_t)bf * K8B;
            for (uint32_t kk = 0; kk < HD; kk += 32u) {
                // A (Q e4m3): the qgmma_probe x4 recipe over a 16x32 chunk
                uint32_t af[4];
                const uint32_t arow = lane & 15u;
                const uint32_t ahalf = (lane >> 4) & 1u;
                const unsigned char* ap = s_q8 + (size_t)arow * q_s + kk + ahalf * 16u;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(af[0]), "=r"(af[1]), "=r"(af[2]), "=r"(af[3])
                             : "r"((unsigned)__cvta_generic_to_shared(ap)));
                // B (K fp8, swizzled raw): x2 - lanes 0..7 keys @ khalf 0,
                // 8..15 @ khalf 1 (16-fp8 slices through sw8)
                uint32_t bfr[2];
                const uint32_t khalf = (lane >> 3) & 1u;
                const uint32_t boxo = (kk >> 7) * 2048u;
                const uint32_t c16 = ((kk & 127u) >> 4) + khalf;
                const unsigned char* bp = kb + boxo + sw8(rk, c16);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0,%1}, [%2];"
                             : "=r"(bfr[0]), "=r"(bfr[1])
                             : "r"((unsigned)__cvta_generic_to_shared(bp)));
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(dfr[0]), "+f"(dfr[1]), "+f"(dfr[2]), "+f"(dfr[3])
                    : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]),
                      "r"(bfr[0]), "r"(bfr[1]));
            }
            #pragma unroll
            for (uint32_t half = 0; half < 2u; ++half)
                #pragma unroll
                for (uint32_t cc = 0; cc < 2u; ++cc) {
                    const uint32_t pp = p0 + 2u * (lane & 3u) + cc;
                    dfr[half * 2u + cc] = pp >= plo && pp < phi
                        ? dfr[half * 2u + cc] * scale : -INFINITY;
                }
            float pm = fmaxf(dfr[0], dfr[1]);
            #pragma unroll
            for (uint32_t off = 1; off <= 2; off <<= 1)
                pm = fmaxf(pm, __shfl_xor_sync(0xffffffffu, pm, off));
            if ((lane & 3u) == 0 && rr < G) s_pmax[warp][rr] = pm;
        };
        auto fold_t = [&](uint32_t t) {
            const uint32_t slot2 = t & 1u;
            float mnew = 0.f, corr = 1.f, w0 = 0.f, w1 = 0.f;
            if (rr < G) {
                float m = s_m[rr];
                #pragma unroll
                for (uint32_t sw2 = 0; sw2 < NSW; ++sw2)
                    m = fmaxf(m, s_pmax[sw2][rr]);
                mnew = m;
                corr = __expf(s_m[rr] - m);
                w0 = dfr[0] > -INFINITY ? __expf(dfr[0] - m) : 0.f;
                w1 = dfr[1] > -INFINITY ? __expf(dfr[1] - m) : 0.f;
                const uint32_t pp = p0 + 2u * (lane & 3u);
                *(__half2*)(s_wh + (size_t)slot2 * 16u * w_s
                            + (size_t)rr * w_s + pp) = __floats2half2_rn(w0, w1);
            }
            float ps = w0 + w1;
            #pragma unroll
            for (uint32_t off = 1; off <= 2; off <<= 1)
                ps += __shfl_xor_sync(0xffffffffu, ps, off);
            if ((lane & 3u) == 0 && rr < G) {
                s_psum[warp][rr] = ps;
                if (warp == 0) { s_corr[slot2][rr] = corr; s_m[rr] = mnew; }
            }
            asm volatile("bar.sync 1, 64;");       // both warps' psum landed
            if (d < G) {
                float ws = 0.0f;
                #pragma unroll
                for (uint32_t sw2 = 0; sw2 < NSW; ++sw2) ws += s_psum[sw2][d];
                s_l[d] = s_l[d] * s_corr[slot2][d] + ws;
            }
        };
        if (ntiles) {
            score_t(0u);
            asm volatile("bar.sync 1, 64;");
            fold_t(0u);
            asm volatile("bar.arrive 2, 256;");
        }
        for (uint32_t j = 0; j < ntiles; ++j) {
            const uint32_t t = j + 1u;
            if (t >= ntiles) break;                // tail: V side finishes alone
            if (warp == 0 && t + 1u < ntiles) stage_k((t + 1u) & 1u, t + 1u);
            score_t(t);
            asm volatile("bar.sync 1, 64;");       // pmax exchange
            asm volatile("bar.sync 3, 256;");      // w slot freed by V(t-2)
            fold_t(t);
            asm volatile("bar.arrive 2, 256;");    // w(t) ready
        }
    } else if (warp >= 8u) {
        // ---------- V expander pair (warps 8-9, 64 lanes) ----------
        uint32_t pv8[3] = {0u, 0u, 0u};
        const uint32_t lid = d - 256u;
        for (uint32_t t = 0; t < ntiles; ++t) {
            const uint32_t bf = t % 3u;
            bar_wait(&s_bv[bf], pv8[bf]); pv8[bf] ^= 1u;
            expand_v(s_v8 + (size_t)bf * K8B, s_v + (size_t)bf * KVB, lid, 64u);
            asm volatile("bar.sync 5, 64;");
            if (lid == 0)
                asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
                             ::"r"((uint32_t)__cvta_generic_to_shared(&s_bev[bf])));
        }
    } else {
        // ---------- V side (f16 PV over the expanded buffer) ----------
        uint32_t pv[3] = {0u, 0u, 0u};
        constexpr uint32_t SLICE0 = (HD / 48u) * 8u;
        constexpr uint32_t LASTS = HD - 5u * SLICE0;
        constexpr uint32_t MAXSUB = LASTS / 8u;
        const uint32_t vw = warp - 2u;
        const uint32_t n_base_w = vw * SLICE0;
        const uint32_t nsub = vw == 5u ? MAXSUB : SLICE0 / 8u;
        float o_acc[MAXSUB][4];
        #pragma unroll
        for (uint32_t i = 0; i < MAXSUB; ++i)
            #pragma unroll
            for (uint32_t j2 = 0; j2 < 4u; ++j2) o_acc[i][j2] = 0.0f;
        for (uint32_t j = 0; j < ntiles; ++j) {
            asm volatile("bar.sync 2, 256;");      // w(j)/corr(j) ready
            if (warp == 2 && j + 2u < ntiles) stage_v((j + 2u) % 3u, j + 2u);
            asm volatile("bar.arrive 3, 256;");    // V(j-1) fully consumed
            const uint32_t slot2 = j & 1u;
            const uint32_t vbf = j % 3u;
            bar_wait(&s_bev[vbf], pv[vbf]); pv[vbf] ^= 1u;
            unsigned char* vb = s_v + (size_t)vbf * KVB;
            const uint32_t gbase = (B0 + j) * 16u;
            const uint32_t pval = nw > gbase
                ? (nw - gbase < 16u ? nw - gbase : 16u) : 0u;
            if (pval < 16u) {
                for (uint32_t i = d - 64u; i < (16u - pval) * 8u * SEGS; i += 192u) {
                    const uint32_t p = pval + i / (8u * SEGS);
                    const uint32_t jj = i % (8u * SEGS);
                    const uint32_t T = jj >> 3, c = jj & 7u;
                    *(uint4*)(vb + T * 2048u + sw(p, c)) = make_uint4(0u, 0u, 0u, 0u);
                }
                asm volatile("bar.sync 4, 192;");
            }
            #pragma unroll
            for (uint32_t half = 0; half < 2u; ++half) {
                const uint32_t rr = (lane >> 2) + half * 8u;
                const float corr = rr < G ? s_corr[slot2][rr] : 1.0f;
                for (uint32_t sub = 0; sub < nsub; ++sub) {
                    o_acc[sub][half * 2u] *= corr;
                    o_acc[sub][half * 2u + 1u] *= corr;
                }
            }
            const uint32_t r = lane & 15u;
            uint32_t af[4];
            const __half* ap = s_wh + (size_t)slot2 * 16u * w_s
                             + (size_t)(lane & 15u) * w_s + ((lane >> 4) ? 8u : 0u);
            pd_ldm_x4(af, (const unsigned char*)ap);
            for (uint32_t sub = 0; sub < nsub; ++sub) {
                uint32_t bfr[2];
                const uint32_t C = (n_base_w + sub * 8u) >> 3;
                const unsigned char* bp = vb + (C >> 3) * 2048u + sw(r, C & 7u);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                             : "=r"(bfr[0]), "=r"(bfr[1])
                             : "r"((unsigned)__cvta_generic_to_shared(bp)));
                pd_fa_mma16(o_acc[sub], af[0], af[1], af[2], af[3], bfr[0], bfr[1]);
            }
        }
        // no rejoin needed: the last b2 sync ordered fold(ntiles-1)'s s_m/s_l
        if (n_splits == 1u) {
            for (uint32_t sub = 0; sub < nsub; ++sub) {
                #pragma unroll
                for (uint32_t half = 0; half < 2u; ++half) {
                    const uint32_t rr = (lane >> 2) + half * 8u;
                    if (rr >= G) continue;
                    float* dst = out_o
                        + ((size_t)b * n_heads + kvh * G + rr) * HD
                        + n_base_w + sub * 8u + 2u * (lane & 3u);
                    dst[0] = o_acc[sub][half * 2u] / s_l[rr];
                    dst[1] = o_acc[sub][half * 2u + 1u] / s_l[rr];
                }
            }
            return;
        }
        for (uint32_t sub = 0; sub < nsub; ++sub) {
            #pragma unroll
            for (uint32_t half = 0; half < 2u; ++half) {
                const uint32_t rr = (lane >> 2) + half * 8u;
                if (rr >= G) continue;
                const size_t pidx =
                    ((size_t)(kvh * G + rr) * gridDim.y + b) * n_splits + sp;
                float* dst = out_o + pidx * HD + n_base_w + sub * 8u
                           + 2u * (lane & 3u);
                dst[0] = o_acc[sub][half * 2u];
                dst[1] = o_acc[sub][half * 2u + 1u];
            }
        }
        if (out_ml && d - 64u < G) {
            const uint32_t g = d - 64u;
            const size_t pidx = ((size_t)(kvh * G + g) * gridDim.y + b) * n_splits + sp;
            out_ml[pidx * 2u] = s_m[g];
            out_ml[pidx * 2u + 1u] = s_l[g];
        }
        return;
    }
#else
    (void)tmk; (void)tmv; (void)q; (void)out_o; (void)out_ml; (void)positions;
    (void)slots; (void)block_tables; (void)blocks_per_slot; (void)kv_dim; (void)swa_window;
    (void)n_splits; (void)scale;
#endif
}

// ---- vdim sync: dim-major twin V pool ---------------------------
// Transposes freshly appended V rows from the legacy [token][kv_dim] pool
// into vdim[block][kv_dim][16 keys] (block = 16 tokens) so the v9q VD arm
// reads PV B fragments as single u32 loads. One launch per append site
// (~128KB/tick/layer at c32); rows = appended token count, row i lives at
// slot slots[i] (or i), position pos[i]. Paged: block id via the table.
__global__ void pd_vdim_sync_kernel(const unsigned char* __restrict__ pool,
                                    unsigned char* __restrict__ vdim,
                                    const unsigned int* __restrict__ positions,
                                    const unsigned int* __restrict__ slots,
                                    const uint32_t* __restrict__ block_tables,
                                    uint32_t blocks_per_slot, uint32_t kv_dim,
                                    uint32_t rows) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * kv_dim) return;
    const uint32_t r = (uint32_t)(i / kv_dim);
    const uint32_t d = (uint32_t)(i % kv_dim);
    const uint32_t slot = slots ? slots[r] : r;
    const uint32_t pos = positions[r];
    const uint32_t blk = block_tables
        ? block_tables[(size_t)slot * blocks_per_slot + (pos >> 4)]
        : slot * blocks_per_slot + (pos >> 4);
    // legacy pool row for this token: paged pools store token rows at
    // blk*16 + (pos & 15)
    const size_t src_row = (size_t)blk * 16u + (pos & 15u);
    vdim[((size_t)blk * kv_dim + d) * 16u + (pos & 15u)] =
        pool[src_row * kv_dim + d];
}

PD_EXPORT
int pd_vdim_sync(const void* pool, void* vdim, const void* positions,
                 const void* slots, const void* block_tables,
                 unsigned int blocks_per_slot, unsigned int kv_dim,
                 unsigned int rows, void* stream) {
    if (rows == 0) return 0;
    const size_t total = (size_t)rows * kv_dim;
    pd_vdim_sync_kernel<<<(unsigned)((total + 255) / 256), 256, 0,
                          (cudaStream_t)stream>>>(
        (const unsigned char*)pool, (unsigned char*)vdim,
        (const unsigned int*)positions, (const unsigned int*)slots,
        (const uint32_t*)block_tables, blocks_per_slot, kv_dim, rows);
    return (int)cudaGetLastError();
}

// ---- v9q: the QGMMA redesign - fp8 END-TO-END SWA decode ------
// 32-key SUPERTILES (2 KV blocks per stage): full-density k32 fp8 PV (the
// v8q zero-pad waste gone), half the fold/barrier crossings; all smem fp8
// (~46 KB -> 4 blocks/SM by construction - the v8q occupancy failure mode
// designed out). Score: qgmma fragments off the sw8-swizzled raw K
// (probe-exact). PV: A = P direct-cast e4m3 (FA3-class), B built by BYTE
// GATHERS from raw V (no transpose stage, no expander warps; the XOR
// swizzle spreads gather banks). Masked keys carry P == +0 and the tail
// V region is zero-filled, so no NaN can poison the f32 accumulators.
// 256 threads: warps 0-1 score, 2-7 V. FIN epilogue included.
// MB: __launch_bounds__ min-blocks/SM target. The unbounded build
// spent 106 regs (SWA) / 212 regs + 240B stack (GLB) and REG-capped
// occupancy at 2 / 1 blocks/SM - half of what the all-fp8 smem design
// paid for. MB > 1 trades register spill for co-residency; variants are
// env-dispatched (PADDOCK_V9Q_MB / _MB_GLB) for A/B, default = measured
// winner.
// VD: dim-major V. VD=1 reads a SECOND pool laid out
// vdim[block][kv_dim][16 keys] via a plain (16,HD)-box map in the tmv slot -
// one 4KB panel per (block, head), smem [panel][dim][16] - so each PV B
// fragment is one u32 load instead of 8 byte-gathers through raw_at (those
// gathers are 58.5% of the issue load). K path, score, softmax, epilogue
// and all sizes/expect_tx are identical (STB/2 == HD*16). VD=0 is the
// shipped kernel bit-for-bit.
// WS (rung): score-warp count. The 2S/6V split leaves the score stage
// ~2x the V stage per supertile and the barrier stall
// dominant (3.13 stalls/issue) - WS=4 balances the classes to ~8 mma each per
// supertile. WS=2 is the shipped layout bit-for-bit; WS=4 changes only the
// l/psum fold tree (numerics-class: last-ulp on out via /l, tolerance-gated
// in the probe). fmax merges are order-independent, P bytes identical.
template <uint32_t HD, uint32_t G, uint32_t MB = 1u, uint32_t VD = 0u,
          uint32_t WS = 2u>
__global__ void __launch_bounds__(256, MB) pd_attn_decode_v9q_kernel(
    const __grid_constant__ PdTmap tmk,
    const __grid_constant__ PdTmap tmv,
    const float* __restrict__ q, float* __restrict__ out_o,
    float* __restrict__ out_ml, const unsigned int* __restrict__ positions,
    const unsigned int* __restrict__ slots,
    const uint32_t* __restrict__ block_tables, uint32_t blocks_per_slot,
    uint32_t kv_dim, uint32_t swa_window, uint32_t n_splits, float scale) {
#if PD_ATTN_TMA_OK
    PD_PDL_ARM();
    const uint32_t kvh = blockIdx.x, b = blockIdx.y, sp = blockIdx.z;
    const uint32_t d = threadIdx.x, nth = blockDim.x;
    const uint32_t warp = d >> 5, lane = d & 31u;
    const uint32_t n_heads = G * gridDim.x;

    const uint32_t pos = positions[b];
    const uint32_t first_pos =
        (swa_window > 0 && pos + 1 > swa_window) ? (pos + 1 - swa_window) : 0;
    const uint32_t n_pos = pos + 1 - first_pos;
    const uint32_t chunk = (n_pos + n_splits - 1) / n_splits;
    const uint32_t lo = sp * chunk;
    uint32_t hi = lo + chunk;
    if (hi > n_pos) hi = n_pos;
    const uint32_t g_lo = first_pos + lo, g_hi = first_pos + hi;
    const uint32_t B0 = g_lo >> 4;
    const uint32_t nblk = lo < hi ? ((g_hi + 15u) >> 4) - B0 : 0u;
    const uint32_t nst = (nblk + 1u) >> 1;         // 32-key supertiles
    const uint32_t nw = pos + 1u;

    extern __shared__ __align__(128) unsigned char pd_v9q_smraw[];
    constexpr uint32_t STB = HD * 32u;             // supertile bytes (8 KB @256)
    constexpr uint32_t q_s = HD + 16u;
    constexpr uint32_t w_s = 48u;                  // 32 keys + 16 pad (e4m3)
    constexpr uint32_t NV = 8u - WS;               // V-class warps
    constexpr uint32_t GN = 32u / (WS * 8u);       // 8-key groups per score warp
    unsigned char* s_kv = pd_v9q_smraw
        + ((1024u - ((uint32_t)__cvta_generic_to_shared(pd_v9q_smraw) & 1023u)) & 1023u);
    unsigned char* s_k8 = s_kv;                    // 2 x STB raw K
    unsigned char* s_v8 = s_kv + 2u * STB;         // 3 x STB raw V
    unsigned char* s_q8 = s_v8 + 3u * STB;         // 16 x q_s e4m3 Q
    unsigned char* s_w8 = s_q8 + 16u * q_s;        // 2 slots x [16][w_s] e4m3 P
    __shared__ float s_m[G], s_l[G], s_corr[2][G];
    __shared__ float s_pmax[WS][G], s_psum[WS][G];
    __shared__ __align__(8) uint64_t s_bk[2], s_bv[3];

    if (d == 0) {
        const uint32_t mk = (uint32_t)__cvta_generic_to_shared(&s_bk[0]);
        const uint32_t mv = (uint32_t)__cvta_generic_to_shared(&s_bv[0]);
        #pragma unroll
        for (uint32_t i = 0; i < 2u; ++i)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(mk + i * 8u));
        #pragma unroll
        for (uint32_t i = 0; i < 3u; ++i)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(mv + i * 8u));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    const float* qb = q + (size_t)b * n_heads * HD;
    const uint32_t slot = slots ? slots[b] : b;
    const uint32_t* bt = block_tables + (size_t)slot * blocks_per_slot;

    // scale folded into the Q cast: real post-norm Q can exceed e4m3's
    // range (max 448) and saturate - Q*scale (1/16 at hd256) sits squarely
    // in-range, and the masking then applies no further multiply
    for (uint32_t i = d; i < 16u * q_s; i += nth) {
        const uint32_t r = i / q_s, c = i % q_s;
        const float v = (r < G && c < HD)
            ? qb[((size_t)kvh * G + r) * HD + c] * scale : 0.0f;
        s_q8[(size_t)r * q_s + c] = __nv_fp8_e4m3(v).__x;
    }
    for (uint32_t i = d; i < 2u * 16u * w_s; i += nth) s_w8[i] = 0u;
    if (d < G) { s_m[d] = -INFINITY; s_l[d] = 0.0f; }

    // raw fp8 addressing: key kk (0..31) of a supertile, dim dd - TMA lands
    // each 16-token block as 2 swizzled boxes (SWIZZLE_128B fp8 maps)
    auto sw8 = [](uint32_t tok, uint32_t chunk16) -> uint32_t {
        return tok * 128u + ((chunk16 ^ (tok & 7u)) << 4);
    };
    auto raw_at = [&](uint32_t kk, uint32_t dd) -> uint32_t {
        return (kk >> 4) * (STB / 2u) + (dd >> 7) * 2048u
             + sw8(kk & 15u, (dd & 127u) >> 4) + (dd & 15u);
    };
    // stage a SUPERTILE (2 blocks; the tail may have 1 - expect scales)
    auto stage = [&](unsigned char* dstb, uint64_t* bar, const void* tm,
                     uint32_t st, bool isv) {
        // boxes per 16-token block = HD/128 (2 at hd256, 4 at hd512) - the
        // hd256-hardcoded 2 left half the bytes unstaged at hd512 and the
        // mbarrier expect never completed (the Act-54 GLB hang)
        constexpr uint32_t BPB = HD / 128u;
        const uint32_t blocks = min(2u, nblk - st * 2u);
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(bar);
        if (lane == 0)
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                         ::"r"(m), "r"(blocks * (STB / 2u)));
        __syncwarp();
        if (VD && isv) {
            // dim-major pool: one (16, HD) box per block panel; panel bytes
            // == STB/2 so the expect above is already right
            if (lane < blocks) {
                const uint32_t blk = bt[B0 + st * 2u + lane];
                const int y = (int)(blk * kv_dim + kvh * HD);
                const uint32_t dst = (uint32_t)__cvta_generic_to_shared(
                    dstb + lane * (STB / 2u));
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1, {%2, %3}], [%4];"
                    ::"r"(dst), "l"(tm), "r"(0), "r"(y), "r"(m) : "memory");
            }
            return;
        }
        if (lane < blocks * BPB) {
            const uint32_t blk = bt[B0 + st * 2u + lane / BPB];
            const int y = (int)(blk * 16u);
            const int x = (int)(kvh * HD + (lane % BPB) * 128u);
            const uint32_t dst = (uint32_t)__cvta_generic_to_shared(
                dstb + (lane / BPB) * (STB / 2u) + (lane % BPB) * 2048u);
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                " [%0], [%1, {%2, %3}], [%4];"
                ::"r"(dst), "l"(tm), "r"(x), "r"(y), "r"(m) : "memory");
        }
    };
    auto bar_wait = [&](uint64_t* bar, uint32_t parity) {
        const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
        asm volatile("{\n\t.reg .pred p;\nW%=:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                     "@!p bra W%=;\n\t}" ::"r"(a), "r"(parity));
    };

    __syncthreads();
    if (nst) {
        if (warp == 0) stage(s_k8, &s_bk[0], &tmk, 0u, false);
        if (warp == WS) stage(s_v8, &s_bv[0], &tmv, 0u, true);
    }
    if (nst > 1u) {
        if (warp == 0) stage(s_k8 + STB, &s_bk[1], &tmk, 1u, false);
        if (warp == WS) stage(s_v8 + STB, &s_bv[1], &tmv, 1u, true);
    }

    if (warp < WS) {
        // ---------- score: WS warps x 32/WS keys, fp8 qgmma ----------
        uint32_t pk[2] = {0u, 0u};
        float dfr[4u * GN];                        // GN 8-key groups x 4
        const uint32_t p0 = warp * (GN * 8u);
        const uint32_t rr = lane >> 2;
        auto score_t = [&](uint32_t t) {
            const uint32_t bf = t & 1u;
            bar_wait(&s_bk[bf], pk[bf]); pk[bf] ^= 1u;
            const uint32_t gbase = (B0 + t * 2u) * 16u;
            const uint32_t plo = g_lo > gbase ? g_lo - gbase : 0u;
            const uint32_t phi = g_hi - gbase < 32u ? g_hi - gbase : 32u;
            #pragma unroll
            for (uint32_t i = 0; i < 8u; ++i) dfr[i] = 0.f;
            unsigned char* kb = s_k8 + (size_t)bf * STB;
            // A (Q) is grp-invariant: k outer / grp inner loads each Q
            // fragment once and interleaves the two independent dfr chains
            // per k-step; each accumulator still sees kk ascending, so the
            // result is bit-identical to the grp-outer form.
            const uint32_t arow = lane & 15u;
            const uint32_t ahalf = (lane >> 4) & 1u;
            const uint32_t khalf = (lane >> 3) & 1u;
            for (uint32_t kk = 0; kk < HD; kk += 32u) {
                uint32_t af[4];
                const unsigned char* ap = s_q8 + (size_t)arow * q_s + kk + ahalf * 16u;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(af[0]), "=r"(af[1]), "=r"(af[2]), "=r"(af[3])
                             : "r"((unsigned)__cvta_generic_to_shared(ap)));
                #pragma unroll
                for (uint32_t grp = 0; grp < GN; ++grp) {
                    const uint32_t rk = p0 + grp * 8u + (lane & 7u);
                    uint32_t bfr[2];
                    const unsigned char* bp = kb + raw_at(rk, kk + khalf * 16u);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0,%1}, [%2];"
                                 : "=r"(bfr[0]), "=r"(bfr[1])
                                 : "r"((unsigned)__cvta_generic_to_shared(bp)));
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(dfr[grp * 4u]), "+f"(dfr[grp * 4u + 1u]),
                          "+f"(dfr[grp * 4u + 2u]), "+f"(dfr[grp * 4u + 3u])
                        : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]),
                          "r"(bfr[0]), "r"(bfr[1]));
                }
            }
            // only the half==0 accumulators are ever consumed (pmax/fold read
            // dfr[grp*4+{0,1}]; half==1 is the zero-Q padding row rr+8), and
            // interior supertiles have plo==0/phi==32 - both sets of sels are
            // dead there, so skip them (bit-exact: dead values only)
            if (plo > 0u || phi < 32u) {
                #pragma unroll
                for (uint32_t grp = 0; grp < GN; ++grp)
                    #pragma unroll
                    for (uint32_t cc = 0; cc < 2u; ++cc) {
                        const uint32_t pp = p0 + grp * 8u + 2u * (lane & 3u) + cc;
                        float& v = dfr[grp * 4u + cc];
                        v = pp >= plo && pp < phi ? v : -INFINITY;
                    }
            }
            // pmax over the thread's own row only (half == 0 entries, like
            // v8's fmaxf(dfr[0], dfr[1])). The half == 1 entries belong to
            // row rr+8 - a zero-Q padding row whose in-range scores are
            // exactly 0.0, not -inf: including them clamped m at 0 whenever
            // all real scores were negative and collapsed the softmax by
            // e^|m| (the deterministic maxrel 5.5 at one supertile).
            float pm = -INFINITY;
            #pragma unroll
            for (uint32_t grp = 0; grp < GN; ++grp) {
                pm = fmaxf(pm, dfr[grp * 4u]);
                pm = fmaxf(pm, dfr[grp * 4u + 1u]);
            }
            #pragma unroll
            for (uint32_t off = 1; off <= 2; off <<= 1)
                pm = fmaxf(pm, __shfl_xor_sync(0xffffffffu, pm, off));
            if ((lane & 3u) == 0 && rr < G) s_pmax[warp][rr] = pm;
        };
        auto fold_t = [&](uint32_t t) {
            const uint32_t slot2 = t & 1u;
            float mnew = 0.f, corr = 1.f, ps = 0.f;
            // zero-init: every lane feeds ps into the shfl reduction below;
            // uninitialized wv from the rr >= G lanes poisoned s_l
            // wv carries only the consumed half==0 lanes - the half==1 row
            // (rr+8 zero-Q padding) was exp'd and then never read. ps keeps
            // the exact old add tree: (g0c0+g0c1) + (g1c0+g1c1).
            float wv[4] = {0.f, 0.f, 0.f, 0.f};
            if (rr < G) {
                float m = s_m[rr];
                // fmax merge is order-independent: any WS tree == the old
                // fmaxf(p0, p1) exactly at WS==2
                float pmx = s_pmax[0][rr];
                #pragma unroll
                for (uint32_t w2 = 1; w2 < WS; ++w2)
                    pmx = fmaxf(pmx, s_pmax[w2][rr]);
                m = fmaxf(m, pmx);
                mnew = m;
                corr = __expf(s_m[rr] - m);
                #pragma unroll
                for (uint32_t grp = 0; grp < GN; ++grp)
                    #pragma unroll
                    for (uint32_t cc = 0; cc < 2u; ++cc) {
                        const float x = dfr[grp * 4u + cc];
                        wv[grp * 2u + cc] = x > -INFINITY ? __expf(x - m) : 0.f;
                    }
                #pragma unroll
                for (uint32_t grp = 0; grp < GN; ++grp) {
                    const uint32_t pp = p0 + grp * 8u + 2u * (lane & 3u);
                    uchar2 pc;
                    pc.x = __nv_fp8_e4m3(wv[grp * 2u]).__x;
                    pc.y = __nv_fp8_e4m3(wv[grp * 2u + 1u]).__x;
                    *(uchar2*)(s_w8 + (size_t)slot2 * 16u * w_s
                               + (size_t)rr * w_s + pp) = pc;
                }
                ps = wv[0] + wv[1];
                #pragma unroll
                for (uint32_t g2 = 1; g2 < GN; ++g2)
                    ps = ps + (wv[g2 * 2u] + wv[g2 * 2u + 1u]);
            }
            #pragma unroll
            for (uint32_t off = 1; off <= 2; off <<= 1)
                ps += __shfl_xor_sync(0xffffffffu, ps, off);
            if ((lane & 3u) == 0 && rr < G) {
                s_psum[warp][rr] = ps;
                if (warp == 0) { s_corr[slot2][rr] = corr; s_m[rr] = mnew; }
            }
            asm volatile("bar.sync 1, %0;" ::"r"(WS * 32u) : "memory");
            // WS==2 keeps the exact old sum: (s_l*corr + psum[0]) + psum[1]
            if (d < G) {
                float lacc = s_l[d] * s_corr[slot2][d];
                #pragma unroll
                for (uint32_t w2 = 0; w2 < WS; ++w2) lacc += s_psum[w2][d];
                s_l[d] = lacc;
            }
        };
        if (nst) {
            score_t(0u);
            asm volatile("bar.sync 1, %0;" ::"r"(WS * 32u) : "memory");
            fold_t(0u);
            asm volatile("bar.arrive 2, 256;" ::: "memory");
        }
        for (uint32_t j = 0; j < nst; ++j) {
            const uint32_t t = j + 1u;
            if (t >= nst) break;
            if (warp == 0 && t + 1u < nst) stage(s_k8 + ((t + 1u) & 1u) * STB,
                                                &s_bk[(t + 1u) & 1u], &tmk, t + 1u,
                                                false);
            score_t(t);
            asm volatile("bar.sync 1, %0;" ::"r"(WS * 32u) : "memory");
            asm volatile("bar.sync 3, 256;" ::: "memory");
            fold_t(t);
            asm volatile("bar.arrive 2, 256;" ::: "memory");
        }
    } else {
        // ---------- V side: 8-WS warps, fp8 PV with gather-built B ----------
        uint32_t pv[3] = {0u, 0u, 0u};
        // WS==2 keeps the shipped 40x5+56 dim slicing verbatim; WS==4 slices
        // HD uniformly (64/warp at hd256)
        constexpr uint32_t SLICE0 = WS == 2u ? (HD / 48u) * 8u : HD / NV;
        constexpr uint32_t LASTS = HD - (NV - 1u) * SLICE0;
        constexpr uint32_t MAXSUB = (LASTS > SLICE0 ? LASTS : SLICE0) / 8u;
        const uint32_t vw = warp - WS;
        const uint32_t n_base_w = vw * SLICE0;
        const uint32_t nsub = vw == NV - 1u ? LASTS / 8u : SLICE0 / 8u;
        // c2/c3 of every PV mma belong to the zero-Q padding rows (rr+8):
        // never stored, never consumed. All subs share one dummy pair -
        // cuts 2*(MAXSUB-1) live registers on the V warps (the class that
        // spills under the MB cap)
        float o_acc[MAXSUB][2];
        float o_dead[2] = {0.0f, 0.0f};
        #pragma unroll
        for (uint32_t i = 0; i < MAXSUB; ++i) {
            o_acc[i][0] = 0.0f; o_acc[i][1] = 0.0f;
        }
        for (uint32_t j = 0; j < nst; ++j) {
            asm volatile("bar.sync 2, 256;" ::: "memory");
            if (warp == WS && j + 2u < nst) stage(s_v8 + ((j + 2u) % 3u) * STB,
                                                &s_bv[(j + 2u) % 3u], &tmv, j + 2u,
                                                true);
            asm volatile("bar.arrive 3, 256;" ::: "memory");
            const uint32_t slot2 = j & 1u;
            const uint32_t vbf = j % 3u;
            bar_wait(&s_bv[vbf], pv[vbf]); pv[vbf] ^= 1u;
            unsigned char* vb = s_v8 + (size_t)vbf * STB;
            const uint32_t gbase = (B0 + j * 2u) * 16u;
            uint32_t pval = nw > gbase
                ? (nw - gbase < 32u ? nw - gbase : 32u) : 0u;
            // tail supertiles stage only blocks*16 keys - the un-staged half
            // is STALE smem (possible NaN-pattern e4m3; 0 x NaN poisons the
            // accumulate even under P == 0). Zero from the staged bound too.
            const uint32_t vstaged = min(2u, nblk - j * 2u) * 16u;
            if (vstaged < pval) pval = vstaged;
            if (pval < 32u) {
                // zero the tail keys' raw bytes: masked P is +0, but 0 x NaN
                // would still poison the f32 accumulate
                if (VD) {
                    // dim-major panels: key kk lives at panel (kk>>4), byte
                    // column (kk&15) of every dim row
                    for (uint32_t i = d - WS * 32u; i < (32u - pval) * HD;
                         i += NV * 32u) {
                        const uint32_t kk = pval + i / HD;
                        const uint32_t dd = i % HD;
                        vb[(kk >> 4) * (STB / 2u) + dd * 16u + (kk & 15u)] = 0u;
                    }
                } else {
                    for (uint32_t i = d - WS * 32u;
                         i < (32u - pval) * (HD / 16u); i += NV * 32u) {
                        const uint32_t kk = pval + i / (HD / 16u);
                        const uint32_t c16 = i % (HD / 16u);
                        *(uint4*)(vb + raw_at(kk, c16 * 16u)) =
                            make_uint4(0u, 0u, 0u, 0u);
                    }
                }
                asm volatile("bar.sync 4, %0;" ::"r"(NV * 32u) : "memory");
            }
            {
                const uint32_t rr = lane >> 2;
                const float corr = rr < G ? s_corr[slot2][rr] : 1.0f;
                for (uint32_t sub = 0; sub < nsub; ++sub) {
                    o_acc[sub][0] *= corr;
                    o_acc[sub][1] *= corr;
                }
            }
            // A (P e4m3 [16][32]) once per supertile
            uint32_t af[4];
            {
                const uint32_t arow = lane & 15u;
                const uint32_t ahalf = (lane >> 4) & 1u;
                const unsigned char* ap = s_w8 + (size_t)slot2 * 16u * w_s
                                        + (size_t)arow * w_s + ahalf * 16u;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(af[0]), "=r"(af[1]), "=r"(af[2]), "=r"(af[3])
                             : "r"((unsigned)__cvta_generic_to_shared(ap)));
            }
            for (uint32_t sub = 0; sub < nsub; ++sub) {
                // B (V [32 keys][8 dims] col-major-in-k): byte gathers - lane
                // covers dim n = n_base_w + sub*8 + (lane>>2), keys
                // 4*(lane&3) + khalf*16 .. +3 (the probe's B recipe)
                const uint32_t dn = n_base_w + sub * 8u + (lane >> 2);
                const uint32_t k0 = (lane & 3u) * 4u;
                uint32_t bfr[2];
                if (VD) {
                    // dim-major panel: 4 consecutive keys at one dim = 1 word
                    #pragma unroll
                    for (uint32_t khalf = 0; khalf < 2u; ++khalf)
                        bfr[khalf] = *(const uint32_t*)(
                            vb + khalf * (STB / 2u) + (size_t)dn * 16u + k0);
                } else {
                    #pragma unroll
                    for (uint32_t khalf = 0; khalf < 2u; ++khalf) {
                        uint32_t r0 = vb[raw_at(k0 + khalf * 16u, dn)];
                        r0 |= (uint32_t)vb[raw_at(k0 + 1u + khalf * 16u, dn)] << 8;
                        r0 |= (uint32_t)vb[raw_at(k0 + 2u + khalf * 16u, dn)] << 16;
                        r0 |= (uint32_t)vb[raw_at(k0 + 3u + khalf * 16u, dn)] << 24;
                        bfr[khalf] = r0;
                    }
                }
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(o_acc[sub][0]), "+f"(o_acc[sub][1]),
                      "+f"(o_dead[0]), "+f"(o_dead[1])
                    : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]),
                      "r"(bfr[0]), "r"(bfr[1]));
            }
        }
        if (n_splits == 1u) {
            const uint32_t rr = lane >> 2;
            if (rr < G)
                for (uint32_t sub = 0; sub < nsub; ++sub) {
                    float* dst = out_o
                        + ((size_t)b * n_heads + kvh * G + rr) * HD
                        + n_base_w + sub * 8u + 2u * (lane & 3u);
                    dst[0] = o_acc[sub][0] / s_l[rr];
                    dst[1] = o_acc[sub][1] / s_l[rr];
                }
            return;
        }
        {
            const uint32_t rr = lane >> 2;
            if (rr < G)
                for (uint32_t sub = 0; sub < nsub; ++sub) {
                    const size_t pidx =
                        ((size_t)(kvh * G + rr) * gridDim.y + b) * n_splits + sp;
                    float* dst = out_o + pidx * HD + n_base_w + sub * 8u
                               + 2u * (lane & 3u);
                    dst[0] = o_acc[sub][0];
                    dst[1] = o_acc[sub][1];
                }
        }
        if (out_ml && d - WS * 32u < G) {
            const uint32_t g = d - WS * 32u;
            const size_t pidx = ((size_t)(kvh * G + g) * gridDim.y + b) * n_splits + sp;
            out_ml[pidx * 2u] = s_m[g];
            out_ml[pidx * 2u + 1u] = s_l[g];
        }
        return;
    }
#else
    (void)tmk; (void)tmv; (void)q; (void)out_o; (void)out_ml; (void)positions;
    (void)slots; (void)block_tables; (void)blocks_per_slot; (void)kv_dim; (void)swa_window;
    (void)n_splits; (void)scale;
#endif
}

// ---- v9q2: the ST64 handshake-halving twin of v9q ------------------------
// Stall-probe verdict (c32-nospec shape, grid
// (16,24,1)): the TMA ring never waits (K-bar 0.58us, V-bar
// 0.48us of a 21.3us kernel), the epilogue is ~0.6us, and the wall is the
// per-32-key handshake cadence - 6 supertiles x {sync1, sync3, arrive2,
// sync2, arrive3, 2 mbarrier waits} around ~5.4us of qgmma and ~3.8us of PV.
// This twin stages SixtyFour keys per TMA/mbarrier/named-barrier window but
// keeps the per-32 arithmetic VERBATIM in sequential halves - scoreA, foldA,
// scoreB, foldB - same P-slot parity (t&1), same s_m/s_l update order, same
// corr application order on the V side. BIT-IDENTICAL to v9q by
// construction; the probe memcmps both arms' out_o/out_ml to enforce it.
// Ring geometry: K 2 x 2STB, V 2 x 2STB (the 3-deep 32-key V ring becomes a
// 2-deep 64-key ring - the same 128-key lookahead). smem ~71KB -> 3 CTAs/SM,
// which still holds every CTA of the (16,24,1) grid resident (384 <= 444).
// Barrier pairing per 64 keys: score sync3 x(nst64-1) <-> V arrive3 xnst64
// (64+192=256 arrivals per generation), arrive2/sync2 once each.
template <uint32_t HD, uint32_t G, uint32_t MB = 3u>
__global__ void __launch_bounds__(256, MB) pd_attn_decode_v9q2_kernel(
    const __grid_constant__ PdTmap tmk,
    const __grid_constant__ PdTmap tmv,
    const float* __restrict__ q, float* __restrict__ out_o,
    float* __restrict__ out_ml, const unsigned int* __restrict__ positions,
    const unsigned int* __restrict__ slots,
    const uint32_t* __restrict__ block_tables, uint32_t blocks_per_slot,
    uint32_t kv_dim, uint32_t swa_window, uint32_t n_splits, float scale) {
#if PD_ATTN_TMA_OK
    PD_PDL_ARM();
    const uint32_t kvh = blockIdx.x, b = blockIdx.y, sp = blockIdx.z;
    const uint32_t d = threadIdx.x, nth = blockDim.x;
    const uint32_t warp = d >> 5, lane = d & 31u;
    const uint32_t n_heads = G * gridDim.x;

    const uint32_t pos = positions[b];
    const uint32_t first_pos =
        (swa_window > 0 && pos + 1 > swa_window) ? (pos + 1 - swa_window) : 0;
    const uint32_t n_pos = pos + 1 - first_pos;
    const uint32_t chunk = (n_pos + n_splits - 1) / n_splits;
    const uint32_t lo = sp * chunk;
    uint32_t hi = lo + chunk;
    if (hi > n_pos) hi = n_pos;
    const uint32_t g_lo = first_pos + lo, g_hi = first_pos + hi;
    const uint32_t B0 = g_lo >> 4;
    const uint32_t nblk = lo < hi ? ((g_hi + 15u) >> 4) - B0 : 0u;
    const uint32_t nst = (nblk + 1u) >> 1;         // 32-key tiles (arithmetic unit)
    const uint32_t nst64 = (nblk + 3u) >> 2;       // 64-key staging windows
    const uint32_t nw = pos + 1u;

    extern __shared__ __align__(128) unsigned char pd_v9q2_smraw[];
    constexpr uint32_t STB = HD * 32u;             // 32-key tile bytes (8 KB @256)
    constexpr uint32_t ST64B = 2u * STB;           // 64-key window bytes
    constexpr uint32_t q_s = HD + 16u;
    constexpr uint32_t w_s = 48u;
    unsigned char* s_kv = pd_v9q2_smraw
        + ((1024u - ((uint32_t)__cvta_generic_to_shared(pd_v9q2_smraw) & 1023u)) & 1023u);
    unsigned char* s_k8 = s_kv;                    // 2 x ST64B raw K
    unsigned char* s_v8 = s_kv + 2u * ST64B;       // 2 x ST64B raw V
    unsigned char* s_q8 = s_v8 + 2u * ST64B;       // 16 x q_s e4m3 Q
    unsigned char* s_w8 = s_q8 + 16u * q_s;        // 2 slots x [16][w_s] e4m3 P
    __shared__ float s_m[G], s_l[G], s_corr[2][G];
    __shared__ float s_pmax[2][G], s_psum[2][G];
    __shared__ __align__(8) uint64_t s_bk[2], s_bv[2];

    if (d == 0) {
        const uint32_t mk = (uint32_t)__cvta_generic_to_shared(&s_bk[0]);
        const uint32_t mv = (uint32_t)__cvta_generic_to_shared(&s_bv[0]);
        #pragma unroll
        for (uint32_t i = 0; i < 2u; ++i)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(mk + i * 8u));
        #pragma unroll
        for (uint32_t i = 0; i < 2u; ++i)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(mv + i * 8u));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    const float* qb = q + (size_t)b * n_heads * HD;
    const uint32_t slot = slots ? slots[b] : b;
    const uint32_t* bt = block_tables + (size_t)slot * blocks_per_slot;

    // (A u32-zero-fill + shift-indexed prologue rewrite measured NEUTRAL
    // here - the fill's div/mod/convert cost hides entirely under the
    // window-0 TMA latency, off the critical path. Kept verbatim from v9q.)
    for (uint32_t i = d; i < 16u * q_s; i += nth) {
        const uint32_t r = i / q_s, c = i % q_s;
        const float v = (r < G && c < HD)
            ? qb[((size_t)kvh * G + r) * HD + c] * scale : 0.0f;
        s_q8[(size_t)r * q_s + c] = __nv_fp8_e4m3(v).__x;
    }
    for (uint32_t i = d; i < 2u * 16u * w_s; i += nth) s_w8[i] = 0u;
    if (d < G) { s_m[d] = -INFINITY; s_l[d] = 0.0f; }

    auto sw8 = [](uint32_t tok, uint32_t chunk16) -> uint32_t {
        return tok * 128u + ((chunk16 ^ (tok & 7u)) << 4);
    };
    auto raw_at = [&](uint32_t kk, uint32_t dd) -> uint32_t {
        return (kk >> 4) * (STB / 2u) + (dd >> 7) * 2048u
             + sw8(kk & 15u, (dd & 127u) >> 4) + (dd & 15u);
    };
    // stage a 64-key WINDOW (up to 4 blocks; tails stage fewer - expect scales)
    auto stage64 = [&](unsigned char* dstb, uint64_t* bar, const void* tm,
                       uint32_t st64) {
        constexpr uint32_t BPB = HD / 128u;
        const uint32_t blocks = min(4u, nblk - st64 * 4u);
        const uint32_t m = (uint32_t)__cvta_generic_to_shared(bar);
        if (lane == 0)
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                         ::"r"(m), "r"(blocks * (STB / 2u)));
        __syncwarp();
        if (lane < blocks * BPB) {
            const uint32_t blk = bt[B0 + st64 * 4u + lane / BPB];
            const int y = (int)(blk * 16u);
            const int x = (int)(kvh * HD + (lane % BPB) * 128u);
            const uint32_t dst = (uint32_t)__cvta_generic_to_shared(
                dstb + (lane / BPB) * (STB / 2u) + (lane % BPB) * 2048u);
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                " [%0], [%1, {%2, %3}], [%4];"
                ::"r"(dst), "l"(tm), "r"(x), "r"(y), "r"(m) : "memory");
        }
    };
    auto bar_wait = [&](uint64_t* bar, uint32_t parity) {
        const uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
        asm volatile("{\n\t.reg .pred p;\nW%=:\n\t"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
                     "@!p bra W%=;\n\t}" ::"r"(a), "r"(parity));
    };

    __syncthreads();
    if (nst64) {
        if (warp == 0) stage64(s_k8, &s_bk[0], &tmk, 0u);
        if (warp == 2) stage64(s_v8, &s_bv[0], &tmv, 0u);
    }
    if (nst64 > 1u) {
        if (warp == 0) stage64(s_k8 + ST64B, &s_bk[1], &tmk, 1u);
        if (warp == 2) stage64(s_v8 + ST64B, &s_bv[1], &tmv, 1u);
    }

    if (warp < 2u) {
        // ---------- score: 2 warps x 16 keys per 32-key tile, fp8 qgmma ----
        uint32_t pk[2] = {0u, 0u};
        float dfr[8];
        const uint32_t p0 = warp * 16u;
        const uint32_t rr = lane >> 2;
        // per-32 tile arithmetic - VERBATIM v9q score_t, kb passed per half.
        // (Note: scoring both halves before folding, dfr[2][8], was tried
        // and FALSIFIED - +8 live f32 pushed the 85-reg MB=3 budget into
        // spills and the variant ran +8.6% slower than this one. Half-serial
        // scoring with a single dfr[8] is the measured optimum.)
        auto score_t = [&](uint32_t t, unsigned char* kb) {
            const uint32_t gbase = (B0 + t * 2u) * 16u;
            const uint32_t plo = g_lo > gbase ? g_lo - gbase : 0u;
            const uint32_t phi = g_hi - gbase < 32u ? g_hi - gbase : 32u;
            #pragma unroll
            for (uint32_t i = 0; i < 8u; ++i) dfr[i] = 0.f;
            // same k-outer/grp-inner restructure as v9q score_t: one A load
            // per k-step, interleaved dfr chains, per-accumulator kk order
            // unchanged (bit-identical)
            const uint32_t arow = lane & 15u;
            const uint32_t ahalf = (lane >> 4) & 1u;
            const uint32_t khalf = (lane >> 3) & 1u;
            for (uint32_t kk = 0; kk < HD; kk += 32u) {
                uint32_t af[4];
                const unsigned char* ap = s_q8 + (size_t)arow * q_s + kk + ahalf * 16u;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(af[0]), "=r"(af[1]), "=r"(af[2]), "=r"(af[3])
                             : "r"((unsigned)__cvta_generic_to_shared(ap)));
                #pragma unroll
                for (uint32_t grp = 0; grp < 2u; ++grp) {
                    const uint32_t rk = p0 + grp * 8u + (lane & 7u);
                    uint32_t bfr[2];
                    const unsigned char* bp = kb + raw_at(rk, kk + khalf * 16u);
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0,%1}, [%2];"
                                 : "=r"(bfr[0]), "=r"(bfr[1])
                                 : "r"((unsigned)__cvta_generic_to_shared(bp)));
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(dfr[grp * 4u]), "+f"(dfr[grp * 4u + 1u]),
                          "+f"(dfr[grp * 4u + 2u]), "+f"(dfr[grp * 4u + 3u])
                        : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]),
                          "r"(bfr[0]), "r"(bfr[1]));
                }
            }
            // only the half==0 accumulators are ever consumed (pmax/fold read
            // dfr[grp*4+{0,1}]; half==1 is the zero-Q padding row rr+8), and
            // interior supertiles have plo==0/phi==32 - both sets of sels are
            // dead there, so skip them (bit-exact: dead values only)
            if (plo > 0u || phi < 32u) {
                #pragma unroll
                for (uint32_t grp = 0; grp < 2u; ++grp)
                    #pragma unroll
                    for (uint32_t cc = 0; cc < 2u; ++cc) {
                        const uint32_t pp = p0 + grp * 8u + 2u * (lane & 3u) + cc;
                        float& v = dfr[grp * 4u + cc];
                        v = pp >= plo && pp < phi ? v : -INFINITY;
                    }
            }
            float pm = -INFINITY;
            #pragma unroll
            for (uint32_t grp = 0; grp < 2u; ++grp) {
                pm = fmaxf(pm, dfr[grp * 4u]);
                pm = fmaxf(pm, dfr[grp * 4u + 1u]);
            }
            #pragma unroll
            for (uint32_t off = 1; off <= 2; off <<= 1)
                pm = fmaxf(pm, __shfl_xor_sync(0xffffffffu, pm, off));
            if ((lane & 3u) == 0 && rr < G) s_pmax[warp][rr] = pm;
        };
        // per-32 fold - VERBATIM v9q fold_t (slot = t&1)
        auto fold_t = [&](uint32_t t) {
            const uint32_t slot2 = t & 1u;
            float mnew = 0.f, corr = 1.f, ps = 0.f;
            // wv carries only the consumed half==0 lanes - the half==1 row
            // (rr+8 zero-Q padding) was exp'd and then never read. ps keeps
            // the exact old add tree: (g0c0+g0c1) + (g1c0+g1c1).
            float wv[4] = {0.f, 0.f, 0.f, 0.f};
            if (rr < G) {
                float m = s_m[rr];
                m = fmaxf(m, fmaxf(s_pmax[0][rr], s_pmax[1][rr]));
                mnew = m;
                corr = __expf(s_m[rr] - m);
                #pragma unroll
                for (uint32_t grp = 0; grp < 2u; ++grp)
                    #pragma unroll
                    for (uint32_t cc = 0; cc < 2u; ++cc) {
                        const float x = dfr[grp * 4u + cc];
                        wv[grp * 2u + cc] = x > -INFINITY ? __expf(x - m) : 0.f;
                    }
                #pragma unroll
                for (uint32_t grp = 0; grp < 2u; ++grp) {
                    const uint32_t pp = p0 + grp * 8u + 2u * (lane & 3u);
                    uchar2 pc;
                    pc.x = __nv_fp8_e4m3(wv[grp * 2u]).__x;
                    pc.y = __nv_fp8_e4m3(wv[grp * 2u + 1u]).__x;
                    *(uchar2*)(s_w8 + (size_t)slot2 * 16u * w_s
                               + (size_t)rr * w_s + pp) = pc;
                }
                ps = (wv[0] + wv[1]) + (wv[2] + wv[3]);
            }
            #pragma unroll
            for (uint32_t off = 1; off <= 2; off <<= 1)
                ps += __shfl_xor_sync(0xffffffffu, ps, off);
            if ((lane & 3u) == 0 && rr < G) {
                s_psum[warp][rr] = ps;
                if (warp == 0) { s_corr[slot2][rr] = corr; s_m[rr] = mnew; }
            }
            asm volatile("bar.sync 1, 64;");
            if (d < G) s_l[d] = s_l[d] * s_corr[slot2][d] + s_psum[0][d] + s_psum[1][d];
        };
        for (uint32_t s64 = 0; s64 < nst64; ++s64) {
            bar_wait(&s_bk[s64 & 1u], pk[s64 & 1u]); pk[s64 & 1u] ^= 1u;
            unsigned char* kb64 = s_k8 + (size_t)(s64 & 1u) * ST64B;
            if (s64) asm volatile("bar.sync 3, 256;");   // V consumed both prior slots
            #pragma unroll
            for (uint32_t half = 0; half < 2u; ++half) {
                const uint32_t t = s64 * 2u + half;
                if (t >= nst) break;
                score_t(t, kb64 + (size_t)half * STB);
                asm volatile("bar.sync 1, 64;");
                fold_t(t);
            }
            // re-stage this slot with window s64+2 - only after both halves'
            // K reads are done (the last fold's internal bar.sync 1 ordered
            // both score warps past their reads). Staging any earlier races
            // the TMA write against the reads of the window being consumed:
            // a 2-deep ring of 64-key windows has no third slot to hide in.
            if (warp == 0 && s64 + 2u < nst64)
                stage64(s_k8 + (size_t)(s64 & 1u) * ST64B,
                        &s_bk[s64 & 1u], &tmk, s64 + 2u);
            asm volatile("bar.arrive 2, 256;");          // both P slots ready
        }
        return;
    } else {
        // ---------- V side: 6 warps, fp8 PV - per-32 arithmetic VERBATIM ----
        uint32_t pv[2] = {0u, 0u};
        constexpr uint32_t SLICE0 = (HD / 48u) * 8u;
        constexpr uint32_t LASTS = HD - 5u * SLICE0;
        constexpr uint32_t MAXSUB = LASTS / 8u;
        const uint32_t vw = warp - 2u;
        const uint32_t n_base_w = vw * SLICE0;
        const uint32_t nsub = vw == 5u ? MAXSUB : SLICE0 / 8u;
        float o_acc[MAXSUB][4];
        #pragma unroll
        for (uint32_t i = 0; i < MAXSUB; ++i)
            #pragma unroll
            for (uint32_t j2 = 0; j2 < 4u; ++j2) o_acc[i][j2] = 0.0f;
        for (uint32_t s64 = 0; s64 < nst64; ++s64) {
            asm volatile("bar.sync 2, 256;");            // folds of s64 done
            asm volatile("bar.arrive 3, 256;");
            bar_wait(&s_bv[s64 & 1u], pv[s64 & 1u]); pv[s64 & 1u] ^= 1u;
            unsigned char* vb64 = s_v8 + (size_t)(s64 & 1u) * ST64B;
            #pragma unroll
            for (uint32_t half = 0; half < 2u; ++half) {
                const uint32_t j = s64 * 2u + half;
                if (j >= nst) break;
                const uint32_t slot2 = j & 1u;
                unsigned char* vb = vb64 + (size_t)half * STB;
                const uint32_t gbase = (B0 + j * 2u) * 16u;
                uint32_t pval = nw > gbase
                    ? (nw - gbase < 32u ? nw - gbase : 32u) : 0u;
                const uint32_t vstaged = min(2u, nblk - j * 2u) * 16u;
                if (vstaged < pval) pval = vstaged;
                if (pval < 32u) {
                    for (uint32_t i = d - 64u; i < (32u - pval) * (HD / 16u); i += 192u) {
                        const uint32_t kk = pval + i / (HD / 16u);
                        const uint32_t c16 = i % (HD / 16u);
                        *(uint4*)(vb + raw_at(kk, c16 * 16u)) =
                            make_uint4(0u, 0u, 0u, 0u);
                    }
                    asm volatile("bar.sync 4, 192;");
                }
                #pragma unroll
                for (uint32_t half2 = 0; half2 < 2u; ++half2) {
                    const uint32_t rr = (lane >> 2) + half2 * 8u;
                    const float corr = rr < G ? s_corr[slot2][rr] : 1.0f;
                    for (uint32_t sub = 0; sub < nsub; ++sub) {
                        o_acc[sub][half2 * 2u] *= corr;
                        o_acc[sub][half2 * 2u + 1u] *= corr;
                    }
                }
                uint32_t af[4];
                {
                    const uint32_t arow = lane & 15u;
                    const uint32_t ahalf = (lane >> 4) & 1u;
                    const unsigned char* ap = s_w8 + (size_t)slot2 * 16u * w_s
                                            + (size_t)arow * w_s + ahalf * 16u;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"
                                 : "=r"(af[0]), "=r"(af[1]), "=r"(af[2]), "=r"(af[3])
                                 : "r"((unsigned)__cvta_generic_to_shared(ap)));
                }
                for (uint32_t sub = 0; sub < nsub; ++sub) {
                    const uint32_t dn = n_base_w + sub * 8u + (lane >> 2);
                    const uint32_t k0 = (lane & 3u) * 4u;
                    uint32_t bfr[2];
                    #pragma unroll
                    for (uint32_t khalf = 0; khalf < 2u; ++khalf) {
                        uint32_t r0 = vb[raw_at(k0 + khalf * 16u, dn)];
                        r0 |= (uint32_t)vb[raw_at(k0 + 1u + khalf * 16u, dn)] << 8;
                        r0 |= (uint32_t)vb[raw_at(k0 + 2u + khalf * 16u, dn)] << 16;
                        r0 |= (uint32_t)vb[raw_at(k0 + 3u + khalf * 16u, dn)] << 24;
                        bfr[khalf] = r0;
                    }
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(o_acc[sub][0]), "+f"(o_acc[sub][1]),
                          "+f"(o_acc[sub][2]), "+f"(o_acc[sub][3])
                        : "r"(af[0]), "r"(af[1]), "r"(af[2]), "r"(af[3]),
                          "r"(bfr[0]), "r"(bfr[1]));
                }
            }
            // re-stage this slot with window s64+2. bar.sync 5 (V warps only)
            // orders every V warp's reads of vb64 before the TMA overwrite -
            // same hazard note as the K side.
            if (s64 + 2u < nst64) {
                asm volatile("bar.sync 5, 192;");
                if (warp == 2)
                    stage64(s_v8 + (size_t)(s64 & 1u) * ST64B,
                            &s_bv[s64 & 1u], &tmv, s64 + 2u);
            }
        }
        if (n_splits == 1u) {
            for (uint32_t sub = 0; sub < nsub; ++sub) {
                #pragma unroll
                for (uint32_t half = 0; half < 2u; ++half) {
                    const uint32_t rr = (lane >> 2) + half * 8u;
                    if (rr >= G) continue;
                    float* dst = out_o
                        + ((size_t)b * n_heads + kvh * G + rr) * HD
                        + n_base_w + sub * 8u + 2u * (lane & 3u);
                    dst[0] = o_acc[sub][half * 2u] / s_l[rr];
                    dst[1] = o_acc[sub][half * 2u + 1u] / s_l[rr];
                }
            }
            return;
        }
        for (uint32_t sub = 0; sub < nsub; ++sub) {
            #pragma unroll
            for (uint32_t half = 0; half < 2u; ++half) {
                const uint32_t rr = (lane >> 2) + half * 8u;
                if (rr >= G) continue;
                const size_t pidx =
                    ((size_t)(kvh * G + rr) * gridDim.y + b) * n_splits + sp;
                float* dst = out_o + pidx * HD + n_base_w + sub * 8u
                           + 2u * (lane & 3u);
                dst[0] = o_acc[sub][half * 2u];
                dst[1] = o_acc[sub][half * 2u + 1u];
            }
        }
        if (out_ml && d - 64u < G) {
            const uint32_t g = d - 64u;
            const size_t pidx = ((size_t)(kvh * G + g) * gridDim.y + b) * n_splits + sp;
            out_ml[pidx * 2u] = s_m[g];
            out_ml[pidx * 2u + 1u] = s_l[g];
        }
        return;
    }
#else
    (void)tmk; (void)tmv; (void)q; (void)out_o; (void)out_ml; (void)positions;
    (void)slots; (void)block_tables; (void)blocks_per_slot; (void)kv_dim; (void)swa_window;
    (void)n_splits; (void)scale;
#endif
}

#define PD_SPEC_K1_MAX 8u
template<typename KV, uint32_t TILE>
__global__ void pd_attn_spec_gqa_paged_kernel(
    const float* __restrict__ q, const KV* __restrict__ pool_k,
    const KV* __restrict__ pool_v, float* __restrict__ out_o, float* __restrict__ out_ml,
    const unsigned int* __restrict__ positions, const unsigned int* __restrict__ slots,
    const uint32_t* __restrict__ block_tables, uint32_t blocks_per_slot,
    uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
    uint32_t kv_dim, uint32_t swa_window, uint32_t n_splits,
    uint32_t rows, uint32_t k1, uint32_t gsub, float scale) {
    const uint32_t n_gs = (n_heads / n_kv_heads) / gsub;
    const uint32_t kvh = blockIdx.x / n_gs, gs = blockIdx.x % n_gs;
    const uint32_t c = blockIdx.y, s = blockIdx.z;
    const uint32_t d = threadIdx.x, nth = blockDim.x;
    const uint32_t G = n_heads / n_kv_heads;
    const uint32_t rb = c * k1;
    const uint32_t nrows = (rows - rb) < k1 ? (rows - rb) : k1;
    const uint32_t RH = k1 * gsub; // (row, head) state lanes

    const uint32_t slot = slots ? slots[rb] : rb;
    // per-row bounds; the walk covers the union [lo0, pos_max]
    uint32_t pos_r[PD_SPEC_K1_MAX], first_r[PD_SPEC_K1_MAX];
    #pragma unroll
    for (uint32_t j = 0; j < PD_SPEC_K1_MAX; ++j) {
        if (j < nrows) {
            const uint32_t pj = positions[rb + j];
            pos_r[j] = pj;
            first_r[j] = (swa_window > 0 && pj + 1 > swa_window) ? (pj + 1 - swa_window) : 0u;
        } else {
            pos_r[j] = 0u;
            first_r[j] = 1u; // empty range: everything masks
        }
    }
    const uint32_t lo0 = first_r[0];
    const uint32_t pos_max = pos_r[nrows - 1];
    const uint32_t n_pos = pos_max + 1u - lo0;
    const uint32_t chunk = (n_pos + n_splits - 1u) / n_splits;
    const uint32_t lo = s * chunk;
    uint32_t hi = lo + chunk;
    if (hi > n_pos) hi = n_pos;

    extern __shared__ __align__(16) unsigned char spec_smraw[];
    const uint32_t q_s = head_dim + 4u;
    const uint32_t t_s = TILE + 1u;
    float* s_q = (float*)spec_smraw;
    float* s_sc = s_q + (size_t)RH * q_s;
    float* s_w = s_sc + (size_t)RH * t_s;
    // 16B-aligned base (PD_GQA_FPLANES): cp.async stages 16-byte lines
    KV* s_kv = (KV*)((float*)spec_smraw + PD_GQA_FPLANES(RH, head_dim, TILE));
    const uint32_t row_e = head_dim + 16u / (uint32_t)sizeof(KV);
    __shared__ float s_m[16], s_l[16], s_mnew[16], s_corr[16];

    const uint32_t h0 = gs * gsub; // first q-head (within the group) we own
    const uint32_t* bt = block_tables + (size_t)slot * blocks_per_slot;

    // s_q layout: lane rh = j*gsub + g holds row (rb+j), q-head (kvh*G+h0+g)
    for (uint32_t idx = d; idx < RH * head_dim; idx += nth) {
        const uint32_t rh = idx / head_dim, e = idx % head_dim;
        const uint32_t j = rh / gsub, g = rh % gsub;
        s_q[rh * q_s + e] = (j < nrows)
            ? q[((size_t)(rb + j) * n_heads + (size_t)kvh * G + h0 + g) * head_dim + e]
            : 0.0f;
    }
    for (uint32_t rh = d; rh < RH; rh += nth) { s_m[rh] = -INFINITY; s_l[rh] = 0.0f; }

    // acc[j] pairs, strided like the per-row kernel's e2 loop but per row
    float acc[PD_SPEC_K1_MAX][2u * ((2u * 512u / 2u + 511u) / 512u)]; // gsub*hd/2 pairs / nth, ×2
    #pragma unroll
    for (uint32_t j = 0; j < PD_SPEC_K1_MAX; ++j)
        #pragma unroll
        for (uint32_t e = 0; e < sizeof(acc[0]) / sizeof(float); ++e) acc[j][e] = 0.0f;

    const uint32_t lines = (head_dim * (uint32_t)sizeof(KV)) >> 4;
    auto stage = [&](uint32_t bf, uint32_t t0) {
        const uint32_t n_t = hi - t0 < TILE ? hi - t0 : TILE;
        for (uint32_t i = d; i < 2u * n_t * lines; i += nth) {
            const uint32_t kvsel = i / (n_t * lines);
            const uint32_t jj = i - kvsel * n_t * lines;
            const uint32_t p = jj / lines, l = jj - p * lines;
            const uint32_t gpos = lo0 + t0 + p;
            const uint32_t blk = bt[gpos >> 4];
            const KV* src = (kvsel ? pool_v : pool_k)
                + (size_t)blk * 16u * kv_dim + (size_t)(gpos & 15u) * kv_dim
                + (size_t)kvh * head_dim;
            KV* dst = s_kv + ((size_t)(bf * 2u + kvsel) * TILE + p) * row_e;
            pd_attn_cpa16((char*)dst + l * 16u, (const char*)src + l * 16u);
        }
        pd_attn_cpa_commit();
    };

    if (lo < hi) stage(0u, lo);
    uint32_t bf = 0;
    for (uint32_t t0 = lo; t0 < hi; t0 += TILE, bf ^= 1u) {
        const uint32_t n_t = hi - t0 < TILE ? hi - t0 : TILE;
        const bool more = t0 + TILE < hi;
        if (more) stage(bf ^ 1u, t0 + TILE);
        if (more) pd_attn_cpa_wait1(); else pd_attn_cpa_wait0();
        __syncthreads();
        const KV* kbuf = s_kv + (size_t)(bf * 2u) * TILE * row_e;
        const KV* vbuf = s_kv + ((size_t)(bf * 2u) + 1u) * TILE * row_e;
        for (uint32_t idx = d; idx < n_t * RH; idx += nth) {
            const uint32_t p = idx / RH, rh = idx % RH;
            const uint32_t j = rh / gsub;
            const uint32_t gpos = lo0 + t0 + p;
            float sc;
            if (j >= nrows || gpos > pos_r[j] || gpos < first_r[j]) {
                sc = -INFINITY; // out of this row's causal/window range
            } else {
                sc = 0.0f;
                const float* qrow = s_q + rh * q_s;
                const KV* krow = kbuf + (size_t)p * row_e;
                if (sizeof(KV) == 2u) {
                    for (uint32_t dd = 0; dd < head_dim; dd += 8) {
                        const uint4 kr = *(const uint4*)(krow + dd);
                        const float4 q0 = *(const float4*)(qrow + dd);
                        const float4 q1 = *(const float4*)(qrow + dd + 4);
                        const __half2* kh = (const __half2*)&kr;
                        float2 k0 = __half22float2(kh[0]), kk1 = __half22float2(kh[1]);
                        float2 k2 = __half22float2(kh[2]), k3 = __half22float2(kh[3]);
                        sc += q0.x * k0.x;
                        sc += q0.y * k0.y;
                        sc += q0.z * kk1.x;
                        sc += q0.w * kk1.y;
                        sc += q1.x * k2.x;
                        sc += q1.y * k2.y;
                        sc += q1.z * k3.x;
                        sc += q1.w * k3.y;
                    }
                } else {
                    // fp8 8-wide, mirroring the per-row kernel's branch
                    for (uint32_t dd = 0; dd < head_dim; dd += 8) {
                        const uint2 kr8 = *(const uint2*)(krow + dd);
                        const __nv_fp8_e4m3* kb = (const __nv_fp8_e4m3*)&kr8;
                        #pragma unroll
                        for (uint32_t j = 0; j < 8u; ++j)
                            sc += qrow[dd + j] * (float)kb[j];
                    }
                }
                sc *= scale;
            }
            s_sc[rh * t_s + p] = sc;
        }
        __syncthreads();
        {
            const uint32_t warp = d >> 5, lane = d & 31u, nw = nth >> 5;
            for (uint32_t rh = warp; rh < RH; rh += nw) {
                float v = (lane < n_t) ? s_sc[rh * t_s + lane] : -INFINITY;
                #pragma unroll
                for (uint32_t off = 16; off > 0; off >>= 1)
                    v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
                if (lane == 0) {
                    const float m_new = fmaxf(s_m[rh], v);
                    s_mnew[rh] = m_new;
                    s_corr[rh] = (m_new == -INFINITY) ? 1.0f : __expf(s_m[rh] - m_new);
                }
            }
        }
        __syncthreads();
        for (uint32_t idx = d; idx < n_t * RH; idx += nth) {
            const uint32_t p = idx / RH, rh = idx % RH;
            const float sc = s_sc[rh * t_s + p];
            s_w[rh * t_s + p] = (sc == -INFINITY) ? 0.0f : __expf(sc - s_mnew[rh]);
        }
        __syncthreads();
        #pragma unroll
        for (uint32_t j = 0; j < PD_SPEC_K1_MAX; ++j) {
            if (j >= nrows) break;
            for (uint32_t e2 = d, jj = 0; e2 < gsub * (head_dim >> 1); e2 += nth, jj += 2) {
                const uint32_t hd2 = head_dim >> 1;
                const uint32_t g = e2 / hd2, dd = (e2 - g * hd2) << 1;
                const uint32_t rh = j * gsub + g;
                float a0 = acc[j][jj] * s_corr[rh];
                float a1 = acc[j][jj + 1] * s_corr[rh];
                const float* wrow = s_w + rh * t_s;
                const KV* vrow = vbuf + dd;
                for (uint32_t p = 0; p < n_t; ++p) {
                    const float2 vv = pd_kv_to_f32x2(vrow + (size_t)p * row_e);
                    const float wp = wrow[p];
                    a0 += wp * vv.x;
                    a1 += wp * vv.y;
                }
                acc[j][jj] = a0;
                acc[j][jj + 1] = a1;
            }
        }
        for (uint32_t rh = d; rh < RH; rh += nth) {
            float ws = 0.0f;
            for (uint32_t p = 0; p < n_t; ++p) ws += s_w[rh * t_s + p];
            s_l[rh] = s_l[rh] * s_corr[rh] + ws;
            s_m[rh] = s_mnew[rh];
        }
        __syncthreads();
    }
    __syncthreads();
    #pragma unroll
    for (uint32_t j = 0; j < PD_SPEC_K1_MAX; ++j) {
        if (j >= nrows) break;
        for (uint32_t e2 = d, jj = 0; e2 < gsub * (head_dim >> 1); e2 += nth, jj += 2) {
            const uint32_t hd2 = head_dim >> 1;
            const uint32_t g = e2 / hd2, dd = (e2 - g * hd2) << 1;
            const size_t pidx =
                ((size_t)(kvh * G + h0 + g) * rows + (rb + j)) * n_splits + s;
            out_o[pidx * head_dim + dd] = acc[j][jj];
            out_o[pidx * head_dim + dd + 1] = acc[j][jj + 1];
        }
    }
    for (uint32_t rh = d; rh < RH; rh += nth) {
        const uint32_t j = rh / gsub, g = rh % gsub;
        if (j < nrows) {
            const size_t pidx =
                ((size_t)(kvh * G + h0 + g) * rows + (rb + j)) * n_splits + s;
            out_ml[pidx * 2 + 0] = s_m[rh];
            out_ml[pidx * 2 + 1] = s_l[rh];
        }
    }
}

// ---- Laguna sigmoid MoE router --------------------------------------------
// DeepSeek-V3-class routing as Laguna ships it (HF modeling_laguna.py is
// the reference): scores = sigmoid(logits) in f32; expert SELECTION runs on
// scores + bias (the aux-loss-free `exp_probs_b` correction - it never
// enters the weights); output weights = the UNBIASED scores of the selected
// experts, sum-normalized, then × routed_scale (2.5), so the down-combine
// consumes them unchanged.
//
// Round shape (inspired by ggml's topk-moe butterfly - original
// implementation): per selection round, one xor-butterfly carries (biased,
// raw, idx) to every lane - no lane-0 broadcast, no second shuffle tree to
// ship the weight, and selection s parks its raw score in lane s's register
// (constant index - the old lane-0 `sel_raw[16]` runtime-indexed array
// spilled to local memory). Bit-identical to the tree version: the argmax
// comparator (strict >, lower index wins) is order-invariant, shuffles move
// values exactly, and the normalization sum keeps the s=0..k-1 gather order.
__device__ __forceinline__ void pd_moe_topk_sigmoid_warp(
    const float* __restrict__ logits, const float* __restrict__ bias,
    float routed_scale, uint32_t n_expert, uint32_t k,
    uint32_t* __restrict__ out_idx, float* __restrict__ out_w) {
    const uint32_t lane = threadIdx.x & 31u;
    float sel[8];  // biased selection score
    float raw[8];  // unbiased sigmoid score (the weight source)
    #pragma unroll
    for (uint32_t j = 0; j < 8u; ++j) {
        uint32_t i = lane + 32u * j;
        if (i < n_expert) {
            const float s = 1.0f / (1.0f + expf(-logits[i]));
            raw[j] = s;
            sel[j] = s + (bias ? bias[i] : 0.0f);
        } else {
            raw[j] = 0.0f;
            sel[j] = -1e30f;
        }
    }
    float myw = 0.0f;    // selection `lane`'s unbiased score
    uint32_t myi = 0u;   // selection `lane`'s expert index
    for (uint32_t s = 0; s < k; ++s) {
        float best = -1e30f, braw = 0.0f;
        uint32_t bi = 0;
        #pragma unroll
        for (uint32_t j = 0; j < 8u; ++j) {
            uint32_t i = lane + 32u * j;
            if (sel[j] > best) { best = sel[j]; braw = raw[j]; bi = i; }
        }
        #pragma unroll
        for (uint32_t off = 16u; off > 0u; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, best, off);
            const float orw = __shfl_xor_sync(0xffffffffu, braw, off);
            const uint32_t oi = __shfl_xor_sync(0xffffffffu, bi, off);
            if (ov > best || (ov == best && oi < bi)) {
                best = ov; braw = orw; bi = oi;
            }
        }
        // every lane now holds the round winner: the owning lane retires the
        // slot (predicated per-element - keeps sel[] in registers), lane s
        // keeps the result
        #pragma unroll
        for (uint32_t j = 0; j < 8u; ++j)
            if ((bi >> 5) == j && (bi & 31u) == lane) sel[j] = -1e30f;
        if (lane == s) { myw = braw; myi = bi; }
    }
    // ordered gather-sum (s=0..k-1) on every lane - same order as the old
    // lane-0 serial sum, so the normalized weights round identically.
    // sigmoid scores are strictly positive - sum can't be zero
    float sum = 0.0f;
    for (uint32_t s = 0; s < k; ++s)
        sum += __shfl_sync(0xffffffffu, myw, s);
    const float inv = routed_scale / sum;
    if (lane < k) {
        out_idx[lane] = myi;
        out_w[lane] = myw * inv;
    }
}

// 4 rows per block (warp per row): prefill's r=1024 router drops from 1024
// single-warp blocks to 256; decode B=1 just idles three warps.
__launch_bounds__(128, 1)
__global__ void pd_moe_topk_sigmoid_batch_kernel(
    const float* __restrict__ logits, const float* __restrict__ bias,
    float routed_scale, uint32_t n_expert, uint32_t k,
    uint32_t* __restrict__ out_idx, float* __restrict__ out_w,
    uint32_t batch) {
    // cascade: logits are the router matvec's output. The 2D block's
    // extra threadIdx.x==0 triggers (one per y-row) are idempotent per CTA.
    PD_PDL_ARM();
    const uint32_t b = blockIdx.x * blockDim.y + threadIdx.y;
    if (b >= batch) return;
    pd_moe_topk_sigmoid_warp(logits + (size_t)b * n_expert, bias, routed_scale,
                             n_expert, k, out_idx + (size_t)b * k,
                             out_w + (size_t)b * k);
}

PD_EXPORT
int pd_moe_topk_sigmoid_batch(const void* logits, const void* bias,
                              float routed_scale, uint32_t n_expert, uint32_t k,
                              void* out_idx, void* out_w, uint32_t batch,
                              void* stream) {
    if (batch == 0) return 0;
    if (n_expert > PD_MOE_MAX_EXPERT || k > 16u) return cudaErrorInvalidValue;
    pd_pdl_go(pd_moe_topk_sigmoid_batch_kernel, dim3((batch + 3u) / 4u),
              dim3(32, 4), 0u, (cudaStream_t)stream,
        (const float*)logits, (const float*)bias, routed_scale, n_expert, k,
        (uint32_t*)out_idx, (float*)out_w, batch);
    // pd_launch_status() lives in a later segment - same expression inline
    return (int)cudaGetLastError();
}

// Shared-expert fold-in variant: identical top-k selection and
// weight normalization, but each output row is k+ns wide - lanes k..k+ns-1
// append the shared PSEUDO-expert ids (sh0, sh0+1, ..) with weight 1.0. The
// loader registers the shared expert's row/K splits as experts sh0.. in the
// routed planes, so one moe_align + one bs pair launch covers routed AND
// shared - the 1-block shared launches this replaces measured 10-12% of the
// stream roof (grid (1, ff/128) on a 188-SM die).
__launch_bounds__(128, 1)
__global__ void pd_moe_topk_sigmoid_batch_sh_kernel(
    const float* __restrict__ logits, const float* __restrict__ bias,
    float routed_scale, uint32_t n_expert, uint32_t k, uint32_t ns,
    uint32_t sh0, uint32_t* __restrict__ out_idx, float* __restrict__ out_w,
    uint32_t batch) {
    PD_PDL_ARM();
    const uint32_t b = blockIdx.x * blockDim.y + threadIdx.y;
    if (b >= batch) return;
    const uint32_t kw = k + ns;
    pd_moe_topk_sigmoid_warp(logits + (size_t)b * n_expert, bias, routed_scale,
                             n_expert, k, out_idx + (size_t)b * kw,
                             out_w + (size_t)b * kw);
    const uint32_t lane = threadIdx.x & 31u;
    if (lane >= k && lane < kw) {
        out_idx[(size_t)b * kw + lane] = sh0 + (lane - k);
        out_w[(size_t)b * kw + lane] = 1.0f;
    }
}

PD_EXPORT
int pd_moe_topk_sigmoid_batch_sh(const void* logits, const void* bias,
                                 float routed_scale, uint32_t n_expert,
                                 uint32_t k, uint32_t ns, uint32_t sh0,
                                 void* out_idx, void* out_w, uint32_t batch,
                                 void* stream) {
    if (batch == 0) return 0;
    if (n_expert > PD_MOE_MAX_EXPERT || k > 16u || k + ns > 32u)
        return cudaErrorInvalidValue;
    pd_pdl_go(pd_moe_topk_sigmoid_batch_sh_kernel, dim3((batch + 3u) / 4u),
              dim3(32, 4), 0u, (cudaStream_t)stream,
        (const float*)logits, (const float*)bias, routed_scale, n_expert, k,
        ns, sh0, (uint32_t*)out_idx, (float*)out_w, batch);
    return (int)cudaGetLastError();
}

// DeepSeek-greedy router epilogue: same top-k SELECTION as
// pd_moe_topk_warp, but the weights are the full softmax probabilities -
// w_s = exp(l_s - m_all) / Σ_{all n_expert} exp(l_i - m_all), no
// renormalization among the selected k. The warp helper above renormalizes
// over the chosen k (the gpt-oss/qwen class); DeepSeek-V2's `greedy` +
// norm_topk_prob=False keeps the full-distribution probs, so the two differ
// by exactly the top-k's captured probability mass (~1.1-1.4x on the routed
// branch) while choosing the same experts - fluent and silently wrong if
// conflated. The full denominator is computed from the register copy before
// selection destroys it.
__global__ void pd_moe_topk_softmax_all_kernel(const float* __restrict__ logits,
                                               uint32_t n_expert, uint32_t k,
                                               uint32_t* __restrict__ out_idx,
                                               float* __restrict__ out_w) {
    const uint32_t b = blockIdx.x;
    const float* lg = logits + (size_t)b * n_expert;
    uint32_t* oi = out_idx + (size_t)b * k;
    float* ow = out_w + (size_t)b * k;
    const uint32_t lane = threadIdx.x & 31u;

    float v[8];
    float m = -1e30f;
    #pragma unroll
    for (uint32_t j = 0; j < 8u; ++j) {
        uint32_t i = lane + 32u * j;
        v[j] = i < n_expert ? lg[i] : -1e30f;
        m = fmaxf(m, v[j]);
    }
    for (uint32_t off = 16; off > 0; off >>= 1)
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
    float sum = 0.0f;
    #pragma unroll
    for (uint32_t j = 0; j < 8u; ++j) {
        uint32_t i = lane + 32u * j;
        if (i < n_expert) sum += __expf(v[j] - m);
    }
    for (uint32_t off = 16; off > 0; off >>= 1)
        sum += __shfl_xor_sync(0xffffffffu, sum, off);

    // selection: identical walk to pd_moe_topk_warp
    float sel_logit[16];
    for (uint32_t s = 0; s < k; ++s) {
        float best = -1e30f;
        uint32_t bi = 0;
        #pragma unroll
        for (uint32_t j = 0; j < 8u; ++j) {
            uint32_t i = lane + 32u * j;
            if (v[j] > best) { best = v[j]; bi = i; }
        }
        for (uint32_t off = 16; off > 0; off >>= 1) {
            float ov = __shfl_down_sync(0xffffffffu, best, off);
            uint32_t oi2 = __shfl_down_sync(0xffffffffu, bi, off);
            if (ov > best || (ov == best && oi2 < bi)) { best = ov; bi = oi2; }
        }
        best = __shfl_sync(0xffffffffu, best, 0);
        bi = __shfl_sync(0xffffffffu, bi, 0);
        sel_logit[s] = best;
        if (lane == 0) oi[s] = bi;
        if ((bi & 31u) == lane) v[bi >> 5] = -1e30f;
    }
    if (lane == 0) {
        for (uint32_t s = 0; s < k; ++s) ow[s] = __expf(sel_logit[s] - m) / sum;
    }
}
