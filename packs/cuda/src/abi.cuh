// abi.cuh (formerly 01_abi.cuh) - CUDA/toolkit includes, PD_EXPORT, pack ABI structs (PackInfo / KernelTableV1 types)
// Textually-included segment of the single pack translation unit.
// Not standalone-compilable: include order is defined by ../pack.cu.
// Paddock CUDA kernel pack. Arch-agnostic CUDA C - build.ps1 emits a multi-arch
// fatbin (Ampere/Ada/Hopper/Blackwell); nothing here assumes a specific SM. Shared
// use stays < 48 KB/block (the guaranteed minimum on every arch), and occupancy is
// requested via the max-shared carveout preference, which adapts to each arch's
// capacity - do not hardcode per-SM sizes (they differ: sm_80 164 KB, sm_86/89
// 100 KB, sm_90 227 KB, sm_120 ...).
//
// All kernels in-house per the kernel policy: the
// LUT constants and bit patterns below are the GGUF/OCP format spec (verified
// against ggml source), not borrowed kernel code. Packs export C
// launchers; the engine owns streams and device memory - packs never allocate,
// never sync, never create streams.
//
// Build: see build.ps1 (nvcc --shared, requires MSVC env) on Windows, or
//   nvcc -O3 -gencode=arch=compute_XX,code=sm_XX -Xcompiler -fPIC --shared
// on Linux (produces a .so; the engine loads either).

#include <cuda.h>  // CUtensorMap types only - encode fn comes via
                   // cudaGetDriverEntryPoint, no libcuda link needed
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <mma.h>
#include <cstdint>
#include <cstdlib>

// Exported C launcher visibility: MSVC dllexport on Windows, default ELF
// visibility elsewhere (gcc/clang host compilers reject __declspec).
//
// EXCEPT in the static build (PD_STATIC, set by build.{ps1,sh} -Static), where
// exporting is not merely pointless but harmful. A shared pack has to publish
// these names - that is how the loader finds them. An archive does not: the
// linker resolves them at build time, by address. Leaving the attribute on
// republishes every launcher from paddock-runner.exe's own export table, and
// they are not neutral names - pd_attn_*, pd_*_gemm, pd_moe_*, pd_q8_0_*,
// pd_whisper_* hand a reader an operation-level map of the GPU surface and
// stable entry points to disassemble from. The first static build published
// 430 named exports this way.
//
// Nothing needs a name in this configuration, so nothing gets one.
#if defined(PD_STATIC)
#define PD_EXPORT extern "C"
#elif defined(_WIN32)
#define PD_EXPORT extern "C" __declspec(dllexport)
#else
#define PD_EXPORT extern "C" __attribute__((visibility("default")))
#endif

// Env reads for election/kill switches - pd_env, never getenv. On Windows
// this DLL links the STATIC CRT (nvcc /MT default; dumpbin shows KERNEL32
// only), whose env is a private snapshot taken at DLL load: every default
// the ENGINE elects at model load (SetEnvironmentVariable via Rust set_var)
// is invisible to getenv here, permanently. How it surfaced: the engine
// elected the DNC RS route while this side read nothing and refused with
// 801 on every >=128-row qwen35 prefill span - and every env-gated arm had
// silently been running its fallback on Windows since the elections began.
// GetEnvironmentVariableA reads the live process environment. The ring of
// per-thread buffers keeps pointers stable across the multi-read gate
// expressions (launchers latch results into function-local statics anyway);
// flag values are short - anything over the buffer reads as unset.
#if defined(_WIN32)
extern "C" __declspec(dllimport) unsigned long __stdcall GetEnvironmentVariableA(
    const char* name, char* buffer, unsigned long size);
static inline const char* pd_env(const char* name) {
    static thread_local char bufs[16][128];
    static thread_local unsigned slot = 0;
    char* b = bufs[slot++ & 15u];
    const unsigned long n = GetEnvironmentVariableA(name, b, 128ul);
    return (n == 0ul || n >= 128ul) ? nullptr : b;
}
#else
static inline const char* pd_env(const char* name) { return getenv(name); }
#endif

// Truthy election/kill read: set, non-empty, and not "0". The engine now
// FILLS some of these as tuned defaults (envset::set_env), so "the env always
// wins, FOO=0 reverts" needs a spelled opt-out - bare presence can't say OFF.
// Rust-side twin: envset::env_on.
static inline bool pd_env_on(const char* name) {
    const char* v = pd_env(name);
    return v != nullptr && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
}

namespace wmma = nvcuda::wmma;

// Configurable KV cache element type for the (unified) batched attention path.
// fp16 is the greedy-exact default; fp8 E4M3 (1 byte) is an opt-in throughput/
// memory mode (lossy - 3 mantissa bits - but valuable on fp8-hardware arches).
// The batched kernels are templated on the element type; the launcher picks the
// instantiation from a kv_dtype flag. load/store are overloaded per type.
#define PD_KV_FP16 0u
#define PD_KV_FP8_E4M3 1u
__device__ __forceinline__ float pd_kv_load(__half x) { return __half2float(x); }
__device__ __forceinline__ float pd_kv_load(__nv_fp8_e4m3 x) { return float(x); }
// 4-wide KV loads for the tiled attention staging: one 8 B (f16) / 4 B (fp8)
// transaction per thread instead of four scalar loads - scalar 2 B loads use
// only half of every 32 B sector and cap the stage at ~granularity/2. Caller
// guarantees 4-element alignment (head_dim is a multiple of 32).
__device__ __forceinline__ float4 pd_kv_load4(const __half* p) {
    uint2 raw = *reinterpret_cast<const uint2*>(p);
    __half2 h01 = *reinterpret_cast<const __half2*>(&raw.x);
    __half2 h23 = *reinterpret_cast<const __half2*>(&raw.y);
    float2 f01 = __half22float2(h01), f23 = __half22float2(h23);
    return make_float4(f01.x, f01.y, f23.x, f23.y);
}
__device__ __forceinline__ float4 pd_kv_load4(const __nv_fp8_e4m3* p) {
    unsigned int raw = *reinterpret_cast<const unsigned int*>(p);
    const __nv_fp8_e4m3* b = reinterpret_cast<const __nv_fp8_e4m3*>(&raw);
    return make_float4(float(b[0]), float(b[1]), float(b[2]), float(b[3]));
}
__device__ __forceinline__ void pd_kv_store(__half* p, float f) { *p = __float2half(f); }
__device__ __forceinline__ void pd_kv_store(__nv_fp8_e4m3* p, float f) { *p = __nv_fp8_e4m3(f); }

// DeltaNet recurrent-state storage class (PADDOCK_DN_STATE_BF16): kernels
// compute f32 always; ST = float (exact, the default) or __nv_bfloat16 (half
// the state DRAM; the per-step round COMPOUNDS through the recurrence - a
// long-context-PPL-gated quality trade). Element indices are dtype-agnostic;
// only the byte stride shrinks. 4-elem group ops stay one memory transaction
// (16B f32 / 8B bf16); callers guarantee 4-element alignment.
__device__ __forceinline__ float pd_dns_ld(const float* p) { return *p; }
__device__ __forceinline__ float pd_dns_ld(const __nv_bfloat16* p) {
    return __bfloat162float(*p);
}
__device__ __forceinline__ void pd_dns_st(float* p, float v) { *p = v; }
__device__ __forceinline__ void pd_dns_st(__nv_bfloat16* p, float v) {
    *p = __float2bfloat16(v);
}
__device__ __forceinline__ float4 pd_dns_ld4(const float* p) {
    return *reinterpret_cast<const float4*>(p);
}
__device__ __forceinline__ float4 pd_dns_ld4(const __nv_bfloat16* p) {
    const uint2 r = *reinterpret_cast<const uint2*>(p);
    const __nv_bfloat162 lo = *reinterpret_cast<const __nv_bfloat162*>(&r.x);
    const __nv_bfloat162 hi = *reinterpret_cast<const __nv_bfloat162*>(&r.y);
    return make_float4(__bfloat162float(lo.x), __bfloat162float(lo.y),
                       __bfloat162float(hi.x), __bfloat162float(hi.y));
}
__device__ __forceinline__ void pd_dns_st4(float* p, float4 v) {
    *reinterpret_cast<float4*>(p) = v;
}
__device__ __forceinline__ void pd_dns_st4(__nv_bfloat16* p, float4 v) {
    uint2 r;
    *reinterpret_cast<__nv_bfloat162*>(&r.x) = __floats2bfloat162_rn(v.x, v.y);
    *reinterpret_cast<__nv_bfloat162*>(&r.y) = __floats2bfloat162_rn(v.z, v.w);
    *reinterpret_cast<uint2*>(p) = r;
}
// f16 state twin: same 2 bytes as bf16 (identical DRAM halving)
// with 10 mantissa bits instead of 7 - 8x finer per-step rounding against the
// depth-compounding that put bf16 2-4x outside the DN PPL band.
// Range is safe for the delta rule's bounded states (k is
// L2-normalized, decay < 1, |S| ~ O(|v|) << 65504). PPL-gated like every
// state-class change.
__device__ __forceinline__ float pd_dns_ld(const __half* p) {
    return __half2float(*p);
}
__device__ __forceinline__ void pd_dns_st(__half* p, float v) {
    *p = __float2half_rn(v);
}
__device__ __forceinline__ float4 pd_dns_ld4(const __half* p) {
    const uint2 r = *reinterpret_cast<const uint2*>(p);
    const __half2 lo = *reinterpret_cast<const __half2*>(&r.x);
    const __half2 hi = *reinterpret_cast<const __half2*>(&r.y);
    return make_float4(__half2float(lo.x), __half2float(lo.y),
                       __half2float(hi.x), __half2float(hi.y));
}
__device__ __forceinline__ void pd_dns_st4(__half* p, float4 v) {
    uint2 r;
    *reinterpret_cast<__half2*>(&r.x) = __floats2half2_rn(v.x, v.y);
    *reinterpret_cast<__half2*>(&r.y) = __floats2half2_rn(v.z, v.w);
    *reinterpret_cast<uint2*>(p) = r;
}
// e4m3 state quarter-width twin (fp8 door): 3 mantissa bits - the
// granularity scaling that separated f16 (+0.09%) from bf16 (+1.47%) predicts
// this fails the PPL band, but the door gets probed with numbers, not
// extrapolation. Conversions saturate to +-448 (the fp8 ctor is SATFINITE) so
// range overflow degrades instead of poisoning with inf/NaN.
__device__ __forceinline__ float pd_dns_ld(const __nv_fp8_e4m3* p) {
    return float(*p);
}
__device__ __forceinline__ void pd_dns_st(__nv_fp8_e4m3* p, float v) {
    *p = __nv_fp8_e4m3(v);
}
__device__ __forceinline__ float4 pd_dns_ld4(const __nv_fp8_e4m3* p) {
    const uint32_t r = *reinterpret_cast<const uint32_t*>(p);
    const __nv_fp8_e4m3* b = reinterpret_cast<const __nv_fp8_e4m3*>(&r);
    return make_float4(float(b[0]), float(b[1]), float(b[2]), float(b[3]));
}
__device__ __forceinline__ void pd_dns_st4(__nv_fp8_e4m3* p, float4 v) {
    uint32_t r;
    __nv_fp8_e4m3* b = reinterpret_cast<__nv_fp8_e4m3*>(&r);
    b[0] = __nv_fp8_e4m3(v.x);
    b[1] = __nv_fp8_e4m3(v.y);
    b[2] = __nv_fp8_e4m3(v.z);
    b[3] = __nv_fp8_e4m3(v.w);
    *reinterpret_cast<uint32_t*>(p) = r;
}
static inline bool pd_dns_bf16_env() { return pd_env("PADDOCK_DN_STATE_BF16") != nullptr; }
// State storage class: 0 = f32 (exact), 1 = bf16 (probe-only, falsified),
// 2 = f16, 3 = e4m3 (probe-only). PADDOCK_DN_STATE_F16 is VALUE-aware: "0"
// pins f32, any other value elects f16 - the qwen35 engine sets =1 as its
// model default (gates: PPL +0.09% prefix=1 / -0.07% prefix=1024,
// coherence 32/32); models that have not gated the class simply
// never set the env and stay f32. Probe precedence: F8 > F16 > BF16.
// Launchers that SWITCH kernels use this; route-gates that only need "state
// is not raw f32" use pd_dns_nonf32_env so a new narrow class can never
// slip past. Every switch site must name class 3 explicitly - falling to an
// else-float arm with a 1-byte state buffer is silent corruption.
static inline int pd_dns_state_class() {
    const char* f8 = pd_env("PADDOCK_DN_STATE_F8");
    if (f8 && atoi(f8) != 0) return 3;
    const char* f16 = pd_env("PADDOCK_DN_STATE_F16");
    if (f16 && atoi(f16) != 0) return 2;
    return pd_dns_bf16_env() ? 1 : 0;
}
static inline bool pd_dns_nonf32_env() { return pd_dns_state_class() != 0; }
// streaming (evict-first) twins for the per-tick single-use state walk: the
// recurrent state is read once and rewritten once per tick, but its ~100MB/
// layer of dirty L2 lines writeback into the next GEMMs' read streams
// (measured: gu 130.3us clean -> 171.6 after an emulated state
// stream). ld/st.global.cs keeps the state out of the
// dirty set. Same values, cache-op only - bit-identical by construction.
__device__ __forceinline__ float4 pd_dns_ld4_cs(const float* p) {
    float4 v;
    asm volatile("ld.global.cs.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p));
    return v;
}
__device__ __forceinline__ float4 pd_dns_ld4_cs(const __nv_bfloat16* p) {
    uint2 r;
    asm volatile("ld.global.cs.v2.b32 {%0,%1}, [%2];"
                 : "=r"(r.x), "=r"(r.y) : "l"(p));
    const __nv_bfloat162 lo = *reinterpret_cast<const __nv_bfloat162*>(&r.x);
    const __nv_bfloat162 hi = *reinterpret_cast<const __nv_bfloat162*>(&r.y);
    return make_float4(__bfloat162float(lo.x), __bfloat162float(lo.y),
                       __bfloat162float(hi.x), __bfloat162float(hi.y));
}
__device__ __forceinline__ void pd_dns_st4_cs(float* p, float4 v) {
    asm volatile("st.global.cs.v4.f32 [%0], {%1,%2,%3,%4};"
                 :: "l"(p), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}
__device__ __forceinline__ void pd_dns_st4_cs(__nv_bfloat16* p, float4 v) {
    uint2 r;
    *reinterpret_cast<__nv_bfloat162*>(&r.x) = __floats2bfloat162_rn(v.x, v.y);
    *reinterpret_cast<__nv_bfloat162*>(&r.y) = __floats2bfloat162_rn(v.z, v.w);
    asm volatile("st.global.cs.v2.b32 [%0], {%1,%2};"
                 :: "l"(p), "r"(r.x), "r"(r.y) : "memory");
}
__device__ __forceinline__ float4 pd_dns_ld4_cs(const __half* p) {
    uint2 r;
    asm volatile("ld.global.cs.v2.b32 {%0,%1}, [%2];"
                 : "=r"(r.x), "=r"(r.y) : "l"(p));
    const __half2 lo = *reinterpret_cast<const __half2*>(&r.x);
    const __half2 hi = *reinterpret_cast<const __half2*>(&r.y);
    return make_float4(__half2float(lo.x), __half2float(lo.y),
                       __half2float(hi.x), __half2float(hi.y));
}
__device__ __forceinline__ void pd_dns_st4_cs(__half* p, float4 v) {
    uint2 r;
    *reinterpret_cast<__half2*>(&r.x) = __floats2half2_rn(v.x, v.y);
    *reinterpret_cast<__half2*>(&r.y) = __floats2half2_rn(v.z, v.w);
    asm volatile("st.global.cs.v2.b32 [%0], {%1,%2};"
                 :: "l"(p), "r"(r.x), "r"(r.y) : "memory");
}
__device__ __forceinline__ float4 pd_dns_ld4_cs(const __nv_fp8_e4m3* p) {
    uint32_t r;
    asm volatile("ld.global.cs.b32 %0, [%1];" : "=r"(r) : "l"(p));
    const __nv_fp8_e4m3* b = reinterpret_cast<const __nv_fp8_e4m3*>(&r);
    return make_float4(float(b[0]), float(b[1]), float(b[2]), float(b[3]));
}
__device__ __forceinline__ void pd_dns_st4_cs(__nv_fp8_e4m3* p, float4 v) {
    uint32_t r;
    __nv_fp8_e4m3* b = reinterpret_cast<__nv_fp8_e4m3*>(&r);
    b[0] = __nv_fp8_e4m3(v.x);
    b[1] = __nv_fp8_e4m3(v.y);
    b[2] = __nv_fp8_e4m3(v.z);
    b[3] = __nv_fp8_e4m3(v.w);
    asm volatile("st.global.cs.b32 [%0], %1;" :: "l"(p), "r"(r) : "memory");
}

// ---------------------------------------------------------------- ABI structs
// Layouts mirror crates/paddock-kernels/src/abi.rs exactly. The Rust layout
// tests are the tripwire; change either side only with an ABI version bump.

extern "C" {

struct PackInfo {
    uint32_t magic;
    uint32_t abi_version;
    uint8_t  arch[16];
    uint32_t pack_version[3];
};

struct KernelTableV1 {
    uint32_t size;
    uint32_t reserved;
    int (*mxfp4_dequant_f32)(const void*, void*, uint64_t, void*);
    int (*q8_0_dequant_f32)(const void*, void*, uint64_t, void*);
    int (*rmsnorm_f32)(const void*, const void*, void*, uint32_t, float, void*);
    int (*rope_yarn_f32)(void*, uint32_t, uint32_t, uint32_t, float, float, float, float, float, float, void*);
    int (*softmax_sink_f32)(void*, uint32_t, float, void*);
    int (*swiglu_oai_f32)(void*, const void*, uint32_t, float, float, void*);
    int (*add_inplace_f32)(void*, const void*, uint32_t, void*);
    int (*scale_add_f32)(void*, const void*, float, uint32_t, void*);
    int (*attn_decode_f32)(const void*, const void*, const void*, const void*, void*,
                           uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, void*);
    int (*moe_topk)(const void*, uint32_t, uint32_t, void*, void*, void*);
    int (*mxfp4_gemv_indexed)(const void*, const void*, const void*, uint32_t,
                              const void*, void*, uint32_t, uint32_t, void*);
    int (*scale_add_dev)(void*, const void*, const void*, uint32_t, uint32_t, void*);
    int (*q8_0_gemv)(const void*, const void*, const void*, void*, uint32_t, uint32_t, void*);
    int (*attn_decode_partial)(const void*, const void*, const void*, void*, void*,
                               uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                               uint32_t, float, void*);
    int (*attn_decode_combine)(const void*, const void*, const void*, void*,
                               uint32_t, uint32_t, uint32_t, void*);
    int (*mxfp4_moe_gate_up)(const void*, const void*, const void*, const void*, const void*,
                             const void*, const void*, const void*, void*, uint32_t, uint32_t,
                             uint32_t, float, float, void*);
    int (*mxfp4_moe_down)(const void*, const void*, const void*, const void*, const void*,
                          const void*, void*, uint32_t, uint32_t, uint32_t, void*);
    int (*q8_0_gemm)(const void*, const void*, const void*, void*,
                     uint32_t, uint32_t, uint32_t, void*);
    int (*quantize_q8)(const void*, void*, void*, uint32_t, void*);
    int (*q8_0_gemv_dp4a)(const void*, const void*, const void*, const void*,
                          void*, uint32_t, uint32_t, void*);
    int (*mxfp4_gemv_indexed_dp4a)(const void*, const void*, const void*, uint32_t,
                                   const void*, const void*, void*, uint32_t, uint32_t, void*);
    int (*mxfp4_moe_gate_up_dp4a)(const void*, const void*, const void*, const void*, const void*,
                                  const void*, const void*, const void*, const void*, void*,
                                  uint32_t, uint32_t, uint32_t, float, float, void*);
    int (*mxfp4_moe_down_dp4a)(const void*, const void*, const void*, const void*, const void*,
                               const void*, const void*, void*, uint32_t, uint32_t, uint32_t, void*);
    int (*attn_decode_batch)(const void*, const void*, const void*, const void*, void*,
                             const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t,
                             uint32_t, uint32_t, uint32_t, float, uint32_t, void*);
    int (*rmsnorm_batch)(const void*, const void*, void*, uint32_t, float, uint32_t, void*);
    int (*rope_yarn_batch)(void*, const void*, uint32_t, uint32_t, float, float, float, float,
                           float, float, uint32_t, void*);
    int (*kv_append_batch)(const void*, void*, const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, void*);
    int (*mxfp4_moe_gate_up_batch)(const void*, const void*, const void*, const void*,
                                   const void*, const void*, void*, uint32_t, uint32_t,
                                   uint32_t, uint32_t, float, float, void*);
    int (*mxfp4_moe_down_batch)(const void*, const void*, const void*, const void*,
                                const void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, void*);
    int (*moe_topk_batch)(const void*, const void*, uint32_t, uint32_t, void*, void*, uint32_t, void*);
    int (*moe_slot_map)(const void*, void*, uint32_t, uint32_t, uint32_t, void*);
    int (*mxfp4_moe_gate_up_grouped)(const void*, const void*, const void*, const void*, const void*,
                                     const void*, const void*, const void*, void*, uint32_t,
                                     uint32_t, uint32_t, uint32_t, uint32_t, float, float, void*);
    int (*mxfp4_moe_down_grouped)(const void*, const void*, const void*, const void*, const void*,
                                  const void*, void*, uint32_t, uint32_t, uint32_t, uint32_t,
                                  uint32_t, void*);
    int (*mxfp4_moe_gate_up_gemm)(const void*, const void*, const void*, const void*,
                                  const void*, const void*, void*, uint32_t, uint32_t,
                                  uint32_t, uint32_t, uint32_t, float, float, void*);
    int (*moe_align)(const void*, void*, void*, void*, uint32_t, uint32_t, uint32_t,
                     uint32_t, void*);
    int (*mxfp4_moe_gate_up_gemm_sorted)(const void*, const void*, const void*, const void*,
                                         const void*, const void*, const void*, const void*,
                                         const void*, void*, uint32_t, uint32_t, uint32_t, float,
                                         float, uint32_t, void*);
    int (*mxfp4_moe_down_gemm_sorted)(const void*, const void*, const void*, const void*,
                                      const void*, const void*, const void*, const void*, void*,
                                      uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, void*);
    int (*mxfp4_repack)(const void*, void*, void*, uint64_t, void*);
    int (*convert_f32_f16)(const void*, void*, uint64_t, void*);
    int (*attn_decode_batch_partial)(const void*, const void*, const void*, void*, void*,
                                     const void*, const void*, uint32_t, uint32_t, uint32_t,
                                     uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float,
                                     uint32_t, void*);
    int (*attn_decode_batch_combine)(const void*, const void*, const void*, void*,
                                     uint32_t, uint32_t, uint32_t, uint32_t, void*);
    int (*gated_delta_recurrent)(const void*, const void*, const void*, const void*,
                                 const void*, void*, void*, uint32_t, uint32_t, uint32_t, void*);
    int (*causal_conv1d_silu)(const void*, const void*, void*, uint32_t, uint32_t, uint32_t, void*);
    int (*delta_gate)(const void*, const void*, const void*, const void*, void*, void*,
                      uint32_t, uint32_t, void*);
    int (*gated_rmsnorm)(const void*, const void*, const void*, void*, uint32_t, uint32_t, float, void*);
    int (*deltanet_split_gqa)(const void*, void*, void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, void*);
    int (*mrope)(void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t,
                 float, float, float, float, float, float,
                 uint32_t, uint32_t, uint32_t, uint32_t, void*);
    int (*mul_sigmoid)(void*, const void*, uint32_t, void*);
    int (*swiglu)(void*, const void*, uint32_t, void*);
    int (*split_qg)(const void*, void*, void*, uint32_t, uint32_t, uint32_t, void*);
    int (*conv_step)(void*, const void*, const void*, void*, uint32_t, uint32_t, void*);
    int (*q8_0_repack)(const void*, void*, void*, uint64_t, void*);
    int (*q8_0_gemv_repacked)(const void*, const void*, const void*, const void*, void*,
                              uint32_t, uint32_t, void*);
    int (*embed_gather)(const void*, const void*, void*, uint32_t, void*);
    int (*argmax_advance)(const void*, uint32_t, void*, void*, uint32_t, void*, void*,
                          void*, void*, void*, void*);
    int (*q8_0_ffn_gate_up_swiglu)(const void*, const void*, const void*, const void*,
                                   const void*, void*, uint32_t, uint32_t, void*);
    int (*deltanet_alpha_beta_gate)(const void*, const void*, const void*, const void*,
                                    const void*, const void*, const void*, void*, void*,
                                    uint32_t, uint32_t, void*);
    int (*q8_0_gemm_repacked)(const void*, const void*, const void*, const void*, void*,
                              uint32_t, uint32_t, uint32_t, void*);
    int (*embed_gather_batch)(const void*, const void*, void*, uint32_t, uint32_t, void*);
    int (*q8_0_repacked_to_f16)(const void*, const void*, void*, uint64_t, void*);
    int (*gated_delta_recurrent_snap)(const void*, const void*, const void*, const void*,
                                      const void*, void*, void*, void*, uint32_t, uint32_t,
                                      uint32_t, void*);
    int (*q8_0_gemm_repacked_mt)(const void*, const void*, const void*, const void*, void*,
                                 uint32_t, uint32_t, uint32_t, void*);
    int (*q8_0_gemm_mt_dp4a)(const void*, const void*, const void*, const void*, void*,
                             uint32_t, uint32_t, uint32_t, void*);
    int (*layernorm)(const void*, const void*, const void*, void*, uint32_t, uint32_t,
                     float, void*);
    int (*gelu)(void*, uint64_t, void*);
    int (*bias_add)(void*, const void*, uint32_t, uint32_t, void*);
    int (*mrope_vision)(void*, const void*, uint32_t, uint32_t, uint32_t, float, void*);
    int (*vision_attn)(const void*, const void*, const void*, void*, uint32_t, uint32_t,
                       uint32_t, float, void*);
    int (*gated_delta_recurrent_slots)(const void*, const void*, const void*, const void*,
                                       const void*, const void*, void*, void*, uint32_t,
                                       uint32_t, uint32_t, void*);
    int (*conv_step_slots)(void*, const void*, const void*, void*, const void*, uint32_t,
                           uint32_t, uint32_t, void*);
    int (*q8_0_gemv_dp4a_nc)(const void*, const void*, const void*, const void*, void*,
                             uint32_t, uint32_t, uint32_t, void*);
    int (*q8_0_gemm_mt_dp4a_wide)(const void*, const void*, const void*, const void*, void*,
                                  uint32_t, uint32_t, uint32_t, void*);
    int (*deltanet_split_gqa_norm)(const void*, void*, void*, void*, uint32_t, uint32_t,
                                   uint32_t, uint32_t, void*);
    int (*gated_delta_recurrent_v2)(const void*, const void*, const void*, const void*,
                                    const void*, const void*, void*, void*, void*,
                                    uint32_t, uint32_t, uint32_t, uint32_t, void*);
    int (*argmax_rows)(const void*, void*, uint32_t, uint32_t, void*);
    int (*conv_ext_build_slots)(const void*, const void*, const void*, void*, uint32_t,
                                uint32_t, uint32_t, uint32_t, void*);
    int (*conv_chunk_ext)(const void*, const void*, void*, uint32_t, uint32_t, uint32_t,
                          uint32_t, uint32_t, void*);
    int (*state_restore_slots)(void*, const void*, const void*, const void*, uint32_t,
                               uint32_t, uint32_t, uint32_t, void*);
    int (*conv_commit_slots)(const void*, void*, const void*, const void*, uint32_t,
                             uint32_t, uint32_t, uint32_t, void*);
    int (*bump_rows_u32)(void*, void*, uint32_t, void*);
    int (*q8_0_gemm_mma)(const void*, const void*, const void*, const void*, void*,
                         uint32_t, uint32_t, uint32_t, void*);
    int (*quantize_q8_mmq)(const void*, void*, uint32_t, uint32_t, void*);
    int (*q8_0_gemm_mmq)(const void*, const void*, const void*, void*, void*,
                         uint32_t, uint32_t, uint32_t, void*);
    int (*attn_prefill)(const void*, const void*, const void*, const void*, void*,
                        const void*, const void*, uint32_t, uint32_t, uint32_t,
                        uint32_t, uint32_t, uint32_t, uint32_t, float, uint32_t, void*);
    int (*attn_prefill_f16)(const void*, const void*, const void*, const void*, void*,
                            const void*, const void*, uint32_t, uint32_t, uint32_t,
                            uint32_t, uint32_t, uint32_t, uint32_t, float, uint32_t, void*);
    int (*quantize_q8_mmq_swiglu)(const void*, const void*, void*, uint32_t, uint32_t, void*);
    int (*add_rmsnorm_quant_mmq)(void*, const void*, const void*, void*, void*,
                                 uint32_t, uint32_t, float, void*);
    int (*gated_delta_chunked)(const void*, const void*, const void*, const void*,
                               const void*, void*, void*, void*, void*, void*, void*,
                               uint32_t, uint32_t, uint32_t, void*);
    int (*mxfp4_moe_gate_up_mmq)(const void*, const void*, const void*, const void*,
                                 const void*, const void*, const void*, const void*,
                                 const void*, const void*, void*, void*, uint32_t,
                                 uint32_t, uint32_t, float, float, float, void*);
    int (*mxfp4_moe_down_mmq)(const void*, const void*, const void*, const void*,
                              const void*, const void*, const void*, const void*,
                              const void*, void*, uint32_t, uint32_t, uint32_t,
                              uint32_t, void*);
    int (*batched_copy)(const void*, uint32_t, void*);
    int (*moe_slot_combine)(const void*, void*, uint32_t, uint32_t, uint32_t, void*);
    int (*mxfp4_moe_gate_up_dp4a_b)(const void*, const void*, const void*, const void*,
                                    const void*, const void*, const void*, const void*,
                                    const void*, void*, uint32_t, uint32_t, uint32_t,
                                    uint32_t, float, float, void*);
    int (*mxfp4_moe_down_dp4a_b)(const void*, const void*, const void*, const void*,
                                 const void*, const void*, const void*, void*, uint32_t,
                                 uint32_t, uint32_t, uint32_t, void*);
    int (*matvec_f32_batch)(const void*, const void*, void*, uint32_t, uint32_t,
                            uint32_t, void*);
    int (*q8_0_gemv_dp4a_nc_b)(const void*, const void*, const void*, const void*,
                               const void*, void*, uint32_t, uint32_t, uint32_t,
                               void*);
    int (*quantize_e4m3)(const void*, void*, void*, uint32_t, void*);
    int (*mxfp4_moe_gate_up_bs)(const void*, const void*, const void*, const void*,
                                const void*, const void*, const void*, const void*,
                                const void*, const void*, void*, void*, uint32_t,
                                uint32_t, uint32_t, uint32_t, float, float, float, void*);
    int (*mxfp4_moe_down_bs)(const void*, const void*, const void*, const void*,
                             const void*, const void*, const void*, const void*,
                             const void*, void*, uint32_t, uint32_t, uint32_t,
                             uint32_t, uint32_t, void*);
    int (*q8_0_gemm_mma_ks)(const void*, const void*, const void*, const void*,
                            void*, void*, uint32_t, uint32_t, uint32_t, void*);
    int (*q8_0_moe_gate_up_dp4a)(const void*, const void*, const void*, const void*,
                                 const void*, const void*, const void*, void*, uint32_t,
                                 uint32_t, uint32_t, uint32_t, void*);
    int (*q8_0_moe_down_dp4a)(const void*, const void*, const void*, const void*,
                              const void*, const void*, void*, uint32_t, uint32_t,
                              uint32_t, uint32_t, void*);
    int (*shexp_gate_add)(void*, const void*, const void*, const void*, uint32_t,
                          uint32_t, uint32_t, void*);
    int (*q8_0_moe_gate_up_sorted)(const void*, const void*, const void*, const void*,
                                   const void*, const void*, const void*, const void*,
                                   void*, uint32_t, uint32_t, uint32_t, void*);
    int (*q8_0_moe_down_sorted)(const void*, const void*, const void*, const void*,
                                const void*, const void*, const void*, const void*,
                                void*, uint32_t, uint32_t, uint32_t, uint32_t, void*);
    int (*q8_0_moe_gate_up_mma)(const void*, const void*, const void*, const void*,
                                const void*, const void*, const void*, const void*,
                                void*, void*, uint32_t, uint32_t, uint32_t, uint32_t,
                                void*);
    int (*q8_0_moe_down_mma)(const void*, const void*, const void*, const void*,
                             const void*, const void*, const void*, const void*,
                             void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                             void*);
    int (*add_rmsnorm_batch)(void*, const void*, const void*, void*, uint32_t, float,
                             uint32_t, void*);
    int (*q8_0_gemm_mmq_hi)(const void*, const void*, const void*, void*,
                            uint32_t, uint32_t, uint32_t, void*);
    int (*q8_0_gemm_mmq_pipe)(const void*, const void*, const void*, const void*,
                              void*, uint32_t, uint32_t, uint32_t, void*);
    int (*attn_prefill_batch)(const void*, const void*, const void*, const void*, void*,
                              const void*, const void*, const void*, const void*, uint32_t,
                              uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                              uint32_t, float, uint32_t, void*);
    int (*q8_0_to_mxfp4)(const void*, const void*, void*, void*, uint64_t, void*);
    int (*mxfp4_gemm_bs)(const void*, const void*, const void*, const void*, void*,
                         uint32_t, uint32_t, uint32_t, void*);
    int (*quantize_e4m3_swiglu)(const void*, const void*, void*, void*, uint32_t, void*);
    int (*q8_0_to_fp4p)(const void*, const void*, void*, void*, uint64_t, void*);
    int (*quantize_e2m1)(const void*, void*, void*, uint32_t, void*);
    int (*quantize_e2m1_swiglu)(const void*, const void*, void*, void*, uint32_t, void*);
    int (*mxfp4_gemm_f4)(const void*, const void*, const void*, const void*, void*,
                         uint32_t, uint32_t, uint32_t, void*);
    int (*q8_0_to_nvf4)(const void*, const void*, void*, void*, uint64_t, void*);
    int (*quantize_nvf4)(const void*, void*, void*, uint32_t, void*);
    int (*quantize_nvf4_swiglu)(const void*, const void*, void*, void*, uint32_t, void*);
    int (*mxfp4_gemm_nv4)(const void*, const void*, const void*, const void*, void*,
                          uint32_t, uint32_t, uint32_t, void*);
    int (*q8_0_to_nvf4_rot)(const void*, const void*, void*, void*, uint64_t, void*);
    int (*quantize_nvf4_rot)(const void*, void*, void*, uint32_t, void*);
    int (*mxfp4_gemm_bs_gu)(const void*, const void*, const void*, const void*,
                            const void*, const void*, void*, void*, uint32_t,
                            uint32_t, uint32_t, void*);
    int (*col_absmax)(const void*, void*, uint32_t, uint32_t, void*);
    int (*q8_0_col_absmax)(const void*, const void*, void*, uint32_t, uint32_t, void*);
    int (*quantize_nvf4_smooth)(const void*, const void*, void*, void*, uint32_t,
                                uint32_t, void*);
    int (*q8_0_to_nvf4_smooth)(const void*, const void*, const void*, void*, void*,
                               uint64_t, uint32_t, void*);
    int (*quantize_nvf4_swiglu_smooth)(const void*, const void*, const void*, void*,
                                       void*, uint32_t, uint32_t, void*);
    int (*quantize_e4m3_smooth)(const void*, const void*, void*, void*, uint32_t,
                                uint32_t, void*);
    int (*quantize_e4m3_swiglu_smooth)(const void*, const void*, const void*, void*,
                                       void*, uint32_t, uint32_t, void*);
    int (*q8_0_to_mxfp4_smooth)(const void*, const void*, const void*, void*, void*,
                                uint64_t, uint32_t, void*);
    int (*attn_prefill_batch_f16)(const void*, const void*, const void*, const void*,
                                  void*, const void*, const void*, const void*,
                                  const void*, uint32_t, uint32_t, uint32_t, uint32_t,
                                  uint32_t, uint32_t, uint32_t, uint32_t, float,
                                  uint32_t, void*);
    int (*q_norm_rope)(const void*, const void*, void*, const void*, uint32_t,
                       uint32_t, float, float, float, float, float, float, float,
                       uint32_t, void*);
    int (*k_norm_rope_append)(const void*, const void*, void*, const void*,
                              const void*, uint32_t, uint32_t, uint32_t, float,
                              float, float, float, float, float, float, uint32_t,
                              uint32_t, void*);
    int (*q8_0_to_f8w)(const void*, const void*, void*, void*, uint64_t, void*);
    int (*f8_gemm_w8)(const void*, const void*, const void*, const void*, void*,
                      uint32_t, uint32_t, uint32_t, void*);
    int (*qkv_norm_rope_append)(const void*, const void*, const void*, void*, void*,
                                void*, const void*, const void*, uint32_t, uint32_t,
                                uint32_t, uint32_t, float, float, float, float,
                                float, float, float, uint32_t, uint32_t, void*);
    int (*q8_0_gemm_mmq_pipe64)(const void*, const void*, const void*, void*,
                                uint32_t, uint32_t, uint32_t, void*);
    int (*q8_0_gemm_mmq_pipe_sk)(const void*, const void*, const void*, void*, void*,
                                 uint32_t, uint32_t, uint32_t, uint32_t, void*);
    int (*sample_rows)(const void*, const void*, void*, uint32_t, uint32_t, void*);
    int (*q8_0_gemm_mt_dp4a_b)(const void*, const void*, const void*, const void*,
                               const void*, void*, uint32_t, uint32_t, uint32_t, void*);
    int (*q8_0_gemm_mma_ks_b)(const void*, const void*, const void*, const void*,
                              const void*, void*, void*, uint32_t, uint32_t, uint32_t,
                              void*);
    int (*q8_0_gemm_mmq_b)(const void*, const void*, const void*, const void*, void*,
                           void*, uint32_t, uint32_t, uint32_t, void*);
    int (*mxfp4_moe_down_bs_res)(const void*, const void*, const void*, const void*,
                                 const void*, const void*, const void*, const void*,
                                 const void*, void*, void*, void*, uint32_t, uint32_t,
                                 uint32_t, uint32_t, uint32_t, void*);
    int (*moe_align_bm)(const void*, void*, void*, void*, uint32_t, uint32_t,
                        uint32_t, uint32_t, uint32_t, void*);
    int (*mxfp4_moe_gate_up_bs64)(const void*, const void*, const void*, const void*,
                                  const void*, const void*, const void*, const void*,
                                  const void*, const void*, void*, void*, uint32_t,
                                  uint32_t, uint32_t, uint32_t, float, float, float, void*);
    int (*mxfp4_moe_down_bs64)(const void*, const void*, const void*, const void*,
                               const void*, const void*, const void*, const void*,
                               const void*, void*, uint32_t, uint32_t, uint32_t,
                               uint32_t, uint32_t, void*);
    int (*qkv_rope_append_batch)(const void*, void*, void*, void*, const void*,
                                 const void*, uint32_t, uint32_t, uint32_t, uint32_t,
                                 float, float, float, float, float, float, uint32_t,
                                 uint32_t, void*);
    int (*pipe_advance)(const void*, void*, void*, uint32_t, void*);
    int (*rmsnorm_quant_q8_batch)(const void*, const void*, void*, void*, void*,
                                  uint32_t, float, uint32_t, void*);
    int (*add_rmsnorm_quant_e4m3_batch)(void*, const void*, const void*, void*,
                                        void*, void*, uint32_t, float, uint32_t,
                                        void*);
    int (*q8_0_gemm_mma_ks_qkv_rope)(const void*, const void*, const void*,
                                     const void*, const void*, void*, void*,
                                     void*, void*, const void*, const void*,
                                     uint32_t, uint32_t, uint32_t, uint32_t,
                                     uint32_t, float, float, float, float,
                                     float, float, uint32_t, uint32_t, void*);
    int (*moe_combine_rmsnorm_quant_q8)(void*, const void*, const void*, void*,
                                        void*, void*, uint32_t, uint32_t, float,
                                        uint32_t, void*);
    int (*mxfp4_gu_interleave)(const void*, const void*, void*, uint32_t,
                               uint64_t, void*);
    // Paged KV: decode + append reading a shared block pool [n_blocks, 16,
    // kv_dim] through per-slot block tables (page = 16 tokens). Pure append -
    // no PD_ABI_VERSION bump.
    int (*attn_decode_batch_paged)(const void*, const void*, const void*, const void*, void*,
                                   const void*, const void*, const void*, uint32_t,
                                   uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                                   uint32_t, float, uint32_t, void*);
    int (*kv_append_batch_paged)(const void*, void*, const void*, const void*, const void*,
                                 uint32_t, uint32_t, uint32_t, uint32_t, void*);
    // Paged FlashDecoding partial (P3b) - split decode over the block pool, for
    // ≥128-SM dies. Pure append; no PD_ABI_VERSION bump.
    int (*attn_decode_batch_partial_paged)(const void*, const void*, const void*, void*, void*,
                                           const void*, const void*, const void*, uint32_t,
                                           uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                                           uint32_t, uint32_t, float, uint32_t, void*);
    // Paged tiled prefill (P4b) - the tiled flash-prefill over the block pool, so
    // paged prefill gets the tiled perf class not the decode-class fallback.
    int (*attn_prefill_paged)(const void*, const void*, const void*, const void*, void*,
                              const void*, const void*, const void*, uint32_t, uint32_t,
                              uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float,
                              uint32_t, void*);
    // Paged f16 WMMA prefill (P4b-2) - full prefill perf parity (qwen35's dense
    // default class) over the block pool. page=16 aligns with the WMMA tile.
    int (*attn_prefill_f16_paged)(const void*, const void*, const void*, const void*, void*,
                                  const void*, const void*, const void*, uint32_t, uint32_t,
                                  uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float,
                                  uint32_t, void*);
    // gpt-oss paged fused-append (G1): block-table twins of the fused QKV
    // rope+append kernels. b>64 mixed/prefill append, then b<=64 fused
    // GEMM-combine+rope+append. Bit-exact vs their dense twins.
    int (*qkv_rope_append_batch_paged)(const void*, void*, void*, void*, const void*,
                                       const void*, uint32_t, uint32_t, uint32_t, float,
                                       float, float, float, float, float, uint32_t,
                                       const void*, uint32_t, uint32_t, void*);
    int (*q8_0_gemm_mma_ks_qkv_rope_paged)(const void*, const void*, const void*, const void*,
                                           const void*, void*, void*, void*, void*, const void*,
                                           const void*, uint32_t, uint32_t, uint32_t, uint32_t,
                                           float, float, float, float, float, float, uint32_t,
                                           const void*, uint32_t, uint32_t, void*);
    // two-weight fused repacked GEMM (alpha/beta pair): x staged once per
    // batch tile instead of once per output - bit-exact per output
    int (*q8_0_gemm_repacked_x2)(const void*, const void*, const void*, const void*,
                                 const void*, void*, void*, uint32_t, uint32_t,
                                 uint32_t, uint32_t, void*);
    // DeltaNet decay gate over the fused alpha||beta activation layout
    // (x2-v3: [n_tokens, 2*n_heads] rows from the one-call f32-plane GEMM)
    int (*delta_gate_ab)(const void*, const void*, const void*, void*, void*,
                         uint32_t, uint32_t, void*);
    // slot_combine over bf16 partials (PADDOCK_MOE_PART_BF16 prefill trade)
    int (*moe_slot_combine_bf16)(const void*, void*, uint32_t, uint32_t,
                                 uint32_t, void*);
    // gate = gelu_tanh(gate) * up, ggml-exact GELU (gemma4 GEGLU FFN)
    int (*geglu)(void*, const void*, uint32_t, void*);
    // batched NEOX rope with per-pair frequency divisors (gemma4 global
    // layers: rope_freqs 1e30 entries freeze those pairs = partial rotary)
    int (*rope_factors_batch)(void*, const void*, const void*, uint32_t,
                              uint32_t, float, float, float, float, float,
                              float, uint32_t, void*);
    // gemma4 vision 2D rope: independent NEOX blocks per half (x then y)
    int (*rope2d_neox)(void*, const void*, const void*, uint32_t, uint32_t,
                       uint32_t, float, void*);
    // in-place final-logit softcap: x = cap * tanh(x / cap)
    int (*softcap)(void*, uint32_t, float, void*);
    // Q8_0 embedding gather with fused scale (graph-capturable decode input)
    int (*embed_gather_q8)(const void*, const void*, void*, uint32_t, uint32_t,
                           float, void*);
    // x = (x + y) * s (gemma4 layer tail: residual + layer_output_scale)
    int (*add_scale)(void*, const void*, float, uint32_t, void*);
    // GEGLU over the concatenated gate|up row layout [rows, 2*ff]
    int (*geglu_pair)(void*, uint32_t, uint32_t, void*);
    // gemma4 fused decode QKV epilogue: per-head norms (V weightless) + rope
    // (optional factors) + K/V cache append (dense or ring-paged)
    int (*gemma_qkv_nra)(void*, void*, void*, const void*, const void*, void*,
                         void*, void*, const void*, const void*, const void*,
                         const void*, uint32_t, uint32_t, uint32_t, uint32_t,
                         uint32_t, uint32_t, float, float, void*);
    // x = (x + rmsnorm(proj)·w) · s per row (gemma4 layer-half tail)
    int (*rmsnorm_add_scale)(void*, const void*, const void*, uint32_t, float,
                             float, uint32_t, void*);
    // GEGLU fused into the mmq quantize (gemma4 FFN-down feed):
    // yq = mmq-quantize(gelu_tanh(gate)·up)
    int (*quantize_q8_mmq_geglu)(const void*, const void*, void*, uint32_t,
                                 uint32_t, void*);
    // wide-batch spec-verify attention: k1-deep GQA walk over padded
    // slot-major verify chunks (one KV walk per chunk, per-row causal/window
    // masks). Partial (o, m, l) layout matches attn_decode_batch_combine.
    int (*attn_spec_batch_paged)(const void*, const void*, const void*, void*, void*,
                                 const void*, const void*, const void*, uint32_t,
                                 uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                                 uint32_t, uint32_t, uint32_t, float, uint32_t, void*);
    // kv_dtype-aware gemma qkv epilogue (fp8 KV appends; f16 delegates to
    // the original)
    int (*gemma_qkv_nra2)(void*, void*, void*, const void*, const void*, void*,
                          void*, void*, const void*, const void*, const void*,
                          const void*, uint32_t, uint32_t, uint32_t, uint32_t,
                          uint32_t, uint32_t, float, float, uint32_t, void*);
    // e4m3 decode-lane GEMV over f8w planes (bandwidth-floor parity w/ q8)
    int (*f8_gemv)(const void*, const void*, const void*, const void*, void*,
                   uint32_t, uint32_t, void*);
    // batched twin, 2..16 rows: weights once, per-row accumulators
    int (*f8_gemv_batch)(const void*, const void*, const void*, void*,
                         uint32_t, uint32_t, uint32_t, void*);
    // f8 mma_ks twin: K-split block-scale MMA GEMM over f8w planes for the
    // 4..64-row serving band (data, scale, xq, xs, part, y, in, out, batch)
    int (*f8_gemm_mma_ks)(const void*, const void*, const void*, const void*,
                          void*, void*, uint32_t, uint32_t, uint32_t, void*);
    // fused GEGLU (gelu_tanh) + e4m3 quantize - gemma4 twin of
    // quantize_e4m3_swiglu, bit-identical to geglu -> quantize_e4m3
    int (*quantize_e4m3_geglu)(const void*, const void*, void*, void*,
                               uint32_t, void*);
    // per-ROW-scaled e4m3 prefill class (sm_100): requant, activation
    // row-quant, and the fold-free GEMM (scales applied in the epilogue only)
    int (*q8_0_to_f8row)(const void*, const void*, void*, void*, uint32_t,
                         uint32_t, void*);
    int (*quantize_e4m3_row)(const void*, void*, void*, uint32_t, uint32_t,
                             void*);
    int (*f8row_gemm)(const void*, const void*, const void*, const void*,
                      void*, uint32_t, uint32_t, uint32_t, void*);
    // v4 decode class: pre-swizzled SW128 tile-image W plane + the
    // rowwise tcgen05 GEMM for the r<=64 decode band. repack: (rowmajor e4m3,
    // tiles, in, out); gemm: (wtiles, wrs, xq, xrs, part, y, in, out, batch)
    // - xq must be a >=64-row buffer (64-row TMA boxes). cc 10 only.
    int (*f8_repack_tiles)(const void*, void*, uint32_t, uint32_t, void*);
    int (*f8t_gemm)(const void*, const void*, const void*, const void*,
                    void*, void*, uint32_t, uint32_t, uint32_t, void*);
    // fused-plane GEGLU quantize: (gu [rows][2*n_ff], q, scale, n_ff, rows)
    int (*quantize_e4m3_geglu2)(const void*, void*, void*, uint32_t, uint32_t,
                                void*);
    // qkv-concat nra epilogue: nra2 + a shared row stride (f16 caches only)
    int (*gemma_qkv_nra2s)(void*, void*, void*, const void*, const void*,
                           void*, void*, void*, const void*, const void*,
                           const void*, const void*, uint32_t, uint32_t,
                           uint32_t, uint32_t, uint32_t, uint32_t, float,
                           float, uint32_t, uint32_t, void*);
    // fused-plane GEGLU + per-ROW e4m3 quant, COMPACT [rows][n_ff] output:
    // (gu [rows][2*n_ff], q, rscale, n_ff, rows) - the f8t gu decode epilogue
    int (*quantize_e4m3_geglu2_row)(const void*, void*, void*, uint32_t,
                                    uint32_t, void*);
    // fused batched rmsnorm -> e4m3 quantize: (x, w, q, scale, n, batch, eps)
    int (*rmsnorm_e4m3_batch)(const void*, const void*, void*, void*, uint32_t,
                              uint32_t, float, void*);
    // fp4 GEMV: (data, scale, bias, x, y, in_dim, out_dim) - e2m1 weights
    int (*fp4_gemv)(const void*, const void*, const void*, const void*, void*,
                    uint32_t, uint32_t, void*);
    // fp4 mma_ks twin: (data, scale, xq, xs, part, y, in, out, batch)
    int (*fp4_gemm_mma_ks)(const void*, const void*, const void*, const void*,
                           void*, void*, uint32_t, uint32_t, uint32_t, void*);
    // fused rmsnorm -> per-32 e4m3 quantize: (x, norm_w, q, scale, n, eps,
    // rows)
    int (*rmsnorm_e4m3)(const void*, const void*, void*, void*, uint32_t,
                        float, uint32_t, void*);
    // fused rmsnorm -> ROW-scale e4m3 (the f8t decode band's format):
    // (x, norm_w, q, row_scale_f32, n, eps, rows)
    int (*rmsnorm_e4m3_row)(const void*, const void*, void*, void*, uint32_t,
                            float, uint32_t, void*);
    // band-boundary fusion: residual-add + post-norm + next pre-norm +
    // row-scale e4m3: (x_inout, proj, post_w, pre_w, q, row_scale, n, eps,
    // stream_scale, rows)
    int (*addnorm_e4m3_row)(void*, const void*, const void*, const void*,
                            void*, void*, uint32_t, float, float, uint32_t,
                            void*);
    // fused FlashDecoding combine + per-ROW e4m3 quant (the wo input):
    // (in_o, in_ml, sinks, q, row_scale, n_heads, head_dim, n_splits, batch)
    int (*attn_combine_e4m3_row)(const void*, const void*, const void*, void*,
                                 void*, uint32_t, uint32_t, uint32_t, uint32_t,
                                 void*);
    // fused prefill QKV epilogue norms + rope (appends stay separate):
    // (q, k, v, q_norm, k_norm, qn, kn, vn, positions, factors, n_head,
    //  n_kv, head_dim, eps, theta_scale, freq_scale, corr_low, corr_high,
    //  ext_factor, mscale, rows)
    int (*qkv_norm_rope_batch)(const void*, const void*, const void*,
                               const void*, const void*, void*, void*, void*,
                               const void*, const void*, uint32_t, uint32_t,
                               uint32_t, float, float, float, float, float,
                               float, float, uint32_t, void*);
    /// element-wise += k on a u32 buffer (device-side chain-step position
    /// advance; lets the MTP draft graph carry its own rope-pos increment
    /// instead of a host memcpy per step): (buf, n, k, stream)
    int (*u32_addk)(void*, uint32_t, uint32_t, void*);
    ///  consumer-side K-split absorption (append-only tail):
    /// f8t GEMM leaving nz partial planes in part (reports nz via out ptr):
    /// (wtiles, wrs, xq, xrs, part, y, in, out, batch, no_combine, out_nz, stream)
    int (*f8t_gemm2)(const void*, const void*, const void*, const void*, void*,
                     void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t*, void*);
    /// nz-aware addnorm (proj = nz partial planes):
    /// (x, proj, postw, prew, q, rscale, n, eps, s, rows, nzp, stream)
    int (*addnorm_e4m3_nz)(void*, const void*, const void*, const void*, void*,
                           void*, uint32_t, float, float, uint32_t, uint32_t, void*);
    /// nz-aware fused geglu2 quant (gu = nz partial planes):
    /// (gu, q, rscale, n_ff, rows, nzp, stream)
    int (*quantize_e4m3_geglu2_nz)(const void*, void*, void*, uint32_t, uint32_t,
                                   uint32_t, void*);
    /// GGUF k-quant family (Q4_K/Q5_K/Q6_K; dtype = GGUF raw id 12/13/14 -
    /// see quant/kquant.cuh, append-only tail). Full-tensor dequant to f32:
    /// (src, dst, n_super, dtype, stream)
    int (*kquant_dequant)(const void*, void*, uint64_t, uint32_t, void*);
    /// load-time repack -> aligned data stream + 24B scale records:
    /// (src, dst_data, dst_scales, n_super, dtype, stream)
    int (*kquant_repack)(const void*, void*, void*, uint64_t, uint32_t, void*);
    /// exact fused decode GEMV over the repacked streams:
    /// (data, scales, x, y, in_dim, out_dim, dtype, stream)
    int (*kquant_gemv)(const void*, const void*, const void*, void*, uint32_t,
                       uint32_t, uint32_t, void*);
    /// embedding row-gather from the repacked streams:
    /// (data, scales, tokens, out, embd, n_tokens, dtype, stream)
    int (*kquant_gather)(const void*, const void*, const void*, void*, uint32_t,
                         uint32_t, uint32_t, void*);
    /// dequant a whole repacked k-quant weight to f32 (batch-GEMM interim):
    /// (data, scales, dst, n_super, dtype, stream)
    int (*kquant_dequant_rp)(const void*, const void*, void*, uint64_t, uint32_t,
                             void*);
    /// unconditionally-tiled f32 GEMM (k-quant interim compute stage):
    /// (w [out,in], x [batch,in], out [batch,out], in_dim, out_dim, batch, stream)
    int (*gemm_f32)(const void*, const void*, void*, uint32_t, uint32_t, uint32_t,
                    void*);
    /// per-32-block activation sums off the mmq layout (W4A8 min-term operand):
    /// (yq, sums [chunk][col_pad][4] f32, in_dim, batch, stream)
    int (*mmq_sums)(const void*, void*, uint32_t, uint32_t, void*);
    /// stage-2 W4A8 GEMM off the k-quant repacked streams (int8 tensor cores):
    /// (data, scales, yq mmq-layout, xsums (Q4K/Q5K, else NULL), y [batch,out],
    ///  in_dim, out_dim, batch, dtype, stream)
    int (*kquant_gemm_w4a8)(const void*, const void*, const void*, const void*,
                            void*, uint32_t, uint32_t, uint32_t, uint32_t, void*);
    /// per-16 activation sums off the STRIDED int8 layout (dp4a decode ladder):
    /// (xq, sums [batch][in/16] f32, in_dim, batch, stream)
    int (*q8_sums_strided)(const void*, void*, uint32_t, uint32_t, void*);
    /// W4A8 dp4a batch GEMM (decode-batch shape, strided int8 activations):
    /// (data, scales, xq, xs, xsums (Q4K/Q5K, else NULL), y [batch,out],
    ///  in_dim, out_dim, batch, dtype, stream)
    int (*kquant_gemm_dp4a)(const void*, const void*, const void*, const void*,
                            const void*, void*, uint32_t, uint32_t, uint32_t,
                            uint32_t, void*);
    /// K-split W4A8 mma GEMM (the 17..64 decode-batch rung; strided int8
    /// activations, partial planes + fixed-order combine like q8_0_gemm_mma_ks):
    /// (data, scales, xq, xs, xsums (Q4K/Q5K, else NULL), part (>= 8*out*batch
    ///  f32), y [batch,out], in_dim, out_dim, batch, dtype, stream)
    int (*kquant_gemm_mma_ks)(const void*, const void*, const void*, const void*,
                              const void*, void*, void*, uint32_t, uint32_t,
                              uint32_t, uint32_t, void*);
    /// k-quant routed-expert MoE, token-batched (decode class; the sorted
    /// mma family is the follow-up). gate/up/down may be different k-quant
    /// types (per-tensor dispatch):
    /// gate_up: (gate_data, gate_scales, up_data, up_scales, idx, xq, xs,
    ///  xsums (mu types, else NULL), out [B,n_active,ff], in_dim, ff,
    ///  n_active, batch, gate_dtype, up_dtype, stream)
    int (*kquant_moe_gate_up)(const void*, const void*, const void*, const void*,
                              const void*, const void*, const void*, const void*,
                              void*, uint32_t, uint32_t, uint32_t, uint32_t,
                              uint32_t, uint32_t, void*);
    /// down+combine: (down_data, down_scales, idx, topk_w, fq, fs,
    ///  fsums (mu, else NULL), out [B,embd], ff, embd, n_active, batch,
    ///  down_dtype, stream)
    int (*kquant_moe_down)(const void*, const void*, const void*, const void*,
                           const void*, const void*, const void*, void*,
                           uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                           void*);
    /// sorted k-quant MoE mma gate_up (prefill/serving class; BM=32
    ///  moe_align layout, single dtype for the pair): (gate_data,
    ///  gate_scales, up_data, up_scales, sorted_row, block_expert, xq, xs,
    ///  xsums (mu types, else NULL), fq, fs [sorted rows], in_dim, ff,
    ///  max_blocks, dtype, stream)
    int (*kquant_moe_gate_up_mma)(const void*, const void*, const void*,
                                  const void*, const void*, const void*,
                                  const void*, const void*, const void*,
                                  void*, void*, uint32_t, uint32_t, uint32_t,
                                  uint32_t, void*);
    /// sorted k-quant MoE mma down (deterministic (token, slot) partials for
    ///  moe_slot_combine): (down_data, down_scales, sorted_row, sorted_slot,
    ///  block_expert, topk_w, fq, fs, fsums (mu, else NULL), part, ff, embd,
    ///  n_active, max_blocks, dtype, stream)
    int (*kquant_moe_down_mma)(const void*, const void*, const void*,
                               const void*, const void*, const void*,
                               const void*, const void*, const void*, void*,
                               uint32_t, uint32_t, uint32_t, uint32_t,
                               uint32_t, void*);
    /// W4A8 b=1 decode GEMV (the mmvq-class serving default; the exact-f32
    ///  kquant_gemv stays the oracle): (data, scales, xq, xs, xsums (mu
    ///  formats, else NULL), y, in_dim, out_dim, dtype, stream)
    int (*kquant_gemv_w4a8)(const void*, const void*, const void*, const void*,
                            const void*, void*, uint32_t, uint32_t, uint32_t,
                            void*);
    /// fused activation quantize + per-16 int8 sums - one node where the b=1
    ///  kq tick launched quantize_q8 then q8_sums_strided; outputs
    ///  bit-identical to that pair: (x, q, scale, sums, n, stream)
    int (*quantize_q8_sums)(const void*, void*, void*, void*, uint32_t, void*);
    /// GEGLU twin of q8_0_moe_gate_up_dp4a (gemma4-A4B hybrid MoE): same
    ///  signature/layout, epilogue is gelu_tanh(gate)*up with pd_geglu's
    ///  constants.
    int (*q8_0_moe_gate_up_geglu)(const void*, const void*, const void*, const void*,
                                  const void*, const void*, const void*, void*,
                                  uint32_t, uint32_t, uint32_t, uint32_t, void*);
    /// fold per-expert scalars into routed top-k weights: w[i] *= scale[idx[i]]
    ///  (gemma4-A4B ffn_down_exps.scale): (w, idx, scale, n, stream)
    int (*moe_scale_w)(void*, const void*, const void*, uint32_t, void*);
    /// GEGLU twin of q8_0_moe_gate_up_mma (same signature/sorted layout;
    ///  gelu_tanh epilogue) - the gemma4-A4B sorted expert class.
    int (*q8_0_moe_gate_up_mma_geglu)(const void*, const void*, const void*, const void*,
                                      const void*, const void*, const void*, const void*,
                                      void*, void*, uint32_t, uint32_t, uint32_t,
                                      uint32_t, void*);
    /// Q8_0 -> per-32 e4m3 planes with K-tail padding (zero blocks):
    ///  (q8_data, q8_scale, f8_data, f8_scale, rows, bpr, bpr_pad, stream)
    int (*q8_0_to_f8w_pad)(const void*, const void*, void*, void*, uint64_t,
                           uint32_t, uint32_t, void*);
    /// sorted gather of e4m3 activations + ue8m0 scales (PAD -> zeros):
    ///  (xq, xs, srow, xg, sg, in_dim, srows, stream)
    int (*moe_gather_e4m3)(const void*, const void*, const void*, void*, void*,
                           uint32_t, uint32_t, void*);
    /// fused-plane GEGLU quantize, PADDED output stride (zero K-tail owned
    ///  by the caller's plane): (gu, q, scale, n_ff, n_ff_pad, rows, stream)
    int (*quantize_e4m3_geglu2_pad)(const void*, void*, void*, uint32_t, uint32_t,
                                    uint32_t, void*);
    /// tcgen05 e4m3 grouped MoE gate_up (fused [gate|up] expert planes,
    ///  sorted-dense out): (wdata, wsc, xg, sg, bexp, y, in_dim, rows_per_e,
    ///  n_expert, srows_pad, max_blocks, stream)
    int (*f8bs_moe_gemm_gu)(const void*, const void*, const void*, const void*,
                            const void*, void*, uint32_t, uint32_t, uint32_t,
                            uint32_t, uint32_t, void*);
    /// tcgen05 e4m3 grouped MoE down (scattered topk_w epilogue into the
    ///  slot-partials layout): (wdata, wsc, xg, sg, bexp, srow, sslot,
    ///  topk_w, part, in_dim, rows_per_e, n_expert, srows_pad, max_blocks,
    ///  n_active, stream)
    int (*f8bs_moe_gemm_dn)(const void*, const void*, const void*, const void*,
                            const void*, const void*, const void*, const void*,
                            void*, uint32_t, uint32_t, uint32_t, uint32_t,
                            uint32_t, uint32_t, void*);
    /// decode-band expert pair, intensity rebuild (4 rows/block, warp-per-
    ///  row streams; REORDER class vs the dp4a originals - separate exports
    ///  so qwen's pinned launchers keep exact numerics). Same signatures as
    ///  q8_0_moe_gate_up_dp4a / q8_0_moe_down_dp4a.
    int (*q8_0_moe_gu_dec2_geglu)(const void*, const void*, const void*, const void*,
                                  const void*, const void*, const void*, void*,
                                  uint32_t, uint32_t, uint32_t, uint32_t, void*);
    int (*q8_0_moe_dn_dec2)(const void*, const void*, const void*, const void*,
                            const void*, const void*, void*, uint32_t, uint32_t,
                            uint32_t, uint32_t, void*);
    /// MoE tail fusions (serial-chain depth cuts): dual-weight head norm+q8
    ///  (x, gamma, pre2, rn, pn, q, qs, n, eps, batch, stream); topk+scale
    ///  fold (logits, scale, n_expert, k, idx, w, batch, stream); combine
    ///  trailer (x, proj, dn, pn1, pn2, postw, n, eps, os, batch, stream).
    int (*moe_head)(const void*, const void*, const void*, void*, void*, void*,
                    void*, uint32_t, float, uint32_t, void*);
    int (*moe_topk_scaled)(const void*, const void*, uint32_t, uint32_t, void*,
                           void*, uint32_t, void*);
    int (*moe_tail)(void*, const void*, const void*, const void*, const void*,
                    const void*, uint32_t, float, float, uint32_t, void*);
    /// W4A8 multi-column decode GEMV (spec-verify r-class, ncols 2..5):
    ///  same weight walk as kquant_gemv_w4a8, weight windows unpacked once
    ///  and dotted against ncols STRIDED activation rows (pd_quantize_q8
    ///  layout): (data, scales, xq, xs, xsums (mu formats, else NULL), y
    ///  [ncols x out_dim], in_dim, out_dim, ncols, dtype, stream)
    int (*kquant_gemv_w4a8_nc)(const void*, const void*, const void*,
                               const void*, const void*, void*, uint32_t,
                               uint32_t, uint32_t, uint32_t, void*);
    /// fused alpha/beta matvec + delta gate - one launch for the
    ///  matvec_f32_batch + delta_gate_ab pair, bit-identical outputs:
    ///  (ab_w [2*n_heads, in_dim] f32, x, ssm_a, dt_bias, g, beta, in_dim,
    ///  n_heads, batch, stream)
    int (*matvec_ab_gate)(const void*, const void*, const void*, const void*,
                          void*, void*, uint32_t, uint32_t, uint32_t, void*);
    /// dec3 bulk-streamed decode-band expert pair: per touched
    ///  expert, slab rows stream once through a cp.async.bulk ring and apply
    ///  to the moe_align BM=8 block's routed rows. gu is bitwise dec2; dn is
    ///  a reorder class (per-pair partials + fixed-order combine). sm_90+
    ///  SASS only - per-device NULL below cc 9.
    ///  gu: (gate_data, gate_scale, up_data, up_scale, bexp, srow, sslot,
    ///  xq, xs, out, in_dim, ff, n_active, max_blocks, pairs, stream) -
    ///  `pairs` (routed pair count) sizes the adaptive out-row tiling
    int (*q8_0_moe_gu_dec3_geglu)(const void*, const void*, const void*,
                                  const void*, const void*, const void*,
                                  const void*, const void*, const void*, void*,
                                  uint32_t, uint32_t, uint32_t, uint32_t,
                                  uint32_t, void*);
    ///  dn: (down_data, down_scale, bexp, srow, sslot, topk_w, fq, fs, part,
    ///  ff, embd, n_active, max_blocks, pairs, stream)
    int (*q8_0_moe_dn_dec3)(const void*, const void*, const void*, const void*,
                            const void*, const void*, const void*, const void*,
                            void*, uint32_t, uint32_t, uint32_t, uint32_t,
                            uint32_t, void*);
    ///  combine: (part, out, n, n_active, batch, stream) - dec2's slot-half
    ///  sum order, plain write (no memset needed)
    int (*moe_combine_dec3)(const void*, void*, uint32_t, uint32_t, uint32_t,
                            void*);
    // decode-band f8 expert shapes (BM=32 tc5 gu, Y-resident dn,
    // PAD-block-aware geglu quant)
    int (*f8bs_moe_gemm_gu_d32)(const void*, const void*, const void*, const void*,
                                const void*, void*, uint32_t, uint32_t, uint32_t,
                                uint32_t, uint32_t, void*);
    int (*f8bs_moe_gemm_dn_d32)(const void*, const void*, const void*, const void*,
                                const void*, const void*, const void*, const void*,
                                void*, uint32_t, uint32_t, uint32_t, uint32_t,
                                uint32_t, uint32_t, void*);
    int (*quantize_e4m3_geglu2_pad_b)(const void*, void*, void*, const void*,
                                      uint32_t, uint32_t, uint32_t, uint32_t,
                                      void*);
    ///  uniq-routing diagnostic: histogram accumulate of unique
    /// experts per (tick, layer) into a persistent 4x260-u32 device buffer
    /// (pairs-banded regions picked device-side; graph-capture-safe).
    /// (idx, pairs, n_expert (<= 128), out_accum, stream)
    int (*moe_uniq_hist)(const void*, uint32_t, uint32_t, void*, void*);
    // 242: fusion-program swiglu_fused (fused, out, ff, n_rows, stream)
    int (*swiglu_fused)(const void*, void*, uint32_t, uint32_t, void*);
    // 243: packed row-slice from a fused GEMM landing
    // (src, dst, src_stride, col_off, width, rows, stream)
    int (*row_slice)(const void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, void*);
    // 244: e4m3 decode-band ks GEMM (f8w weights + e4m3 acts, b <= 64)
    // (data, scale, xq, xs, part, y, in_dim, out_dim, batch, stream)
    int (*f8d_gemm_mma_ks)(const void*, const void*, const void*, const void*,
                           void*, void*, uint32_t, uint32_t, uint32_t, void*);
    // 245: bf16-out f8 prefill GEMM (tma route only; NotSupported elsewhere)
    int (*f8_gemm_w8_o16)(const void*, const void*, const void*, const void*,
                          void*, uint32_t, uint32_t, uint32_t, void*);
    // 246: bf16-input swiglu+e4m3 quant (the o16 epilogue's consumer)
    int (*quantize_e4m3_swiglu_b16)(const void*, const void*, void*, void*,
                                    uint32_t, void*);
    // 247: bf16-residual add+rmsnorm+quant (o16 down-GEMM consumer)
    int (*add_rmsnorm_quant_mmq_b16)(void*, const void*, const void*, void*,
                                     void*, uint32_t, uint32_t, float, void*);
    // 248: x (f32) += y (bf16) - the loop-tail residual consumer
    int (*add_inplace_b16)(void*, const void*, uint32_t, void*);
    // 249: native-bf16 -> f8w conversion (fp8 ingestion; no Q8 double-quant)
    int (*bf16_to_f8w)(const void*, void*, void*, uint64_t, void*);
    // 250: native-bf16 -> f8r (per-ROW e8m0 scale - the scale-free stream)
    int (*bf16_to_f8r)(const void*, void*, void*, uint32_t, uint32_t, void*);
    // 251: per-row-scale e4m3 decode ks GEMM (f8r planes, b <= 64)
    int (*f8r_gemm_mma_ks)(const void*, const void*, const void*, const void*,
                           void*, void*, uint32_t, uint32_t, uint32_t, void*);
    // 252: fused-landing swiglu + e4m3 quant (one kernel, no f32 round trip)
    int (*swiglu_fused_e4m3)(const void*, void*, void*, uint32_t, uint32_t, void*);
    // 253: add+rmsnorm with xn AND e4m3 staging out (decode norm+quant fuse)
    int (*add_rmsnorm_e4m3_xn)(void*, const void*, const void*, void*, void*,
                               void*, uint32_t, uint32_t, float, void*);
    // 254: f8w row-major -> tile-linear box repack (load-time, one pass)
    int (*f8w_repack_lin)(const void*, const void*, void*, uint32_t, uint32_t,
                          void*);
    // 255: tile-linear e4m3 decode GEMM (b <= 64; contiguous per-CTA stream)
    int (*f8_gemm_lin)(const void*, const void*, const void*, void*, void*,
                       uint32_t, uint32_t, uint32_t, void*);
    // 256: tile-linear prefill GEMM (tma_kt twin; o16 flag picks bf16 out)
    int (*f8_gemm_lin_kt)(const void*, const void*, const void*, void*,
                          uint32_t, uint32_t, uint32_t, uint32_t, void*);
    // 257: add+rmsnorm+e4m3 with a BF16 residual (o16 prefill post-norm)
    int (*add_rmsnorm_e4m3_xn_b16)(void*, const void*, const void*, void*,
                                   void*, void*, uint32_t, uint32_t, float,
                                   void*);
    // 258: gated rmsnorm + e4m3 quant (DN out_proj prefill glue)
    int (*gated_rmsnorm_e4m3)(const void*, const void*, const void*, void*,
                              void*, void*, uint32_t, uint32_t, float, void*);
    // 259: raw e4m3 -> data-only lin box repack (official-FP8 passthrough)
    int (*f8w_repack_lin_bs)(const void*, void*, uint32_t, uint32_t, void*);
    // 260: block-scale tile-linear decode GEMM (data-only boxes + f32
    // [out/128][in/128] scale plane)
    int (*f8_gemm_lin_bs)(const void*, const void*, const void*, const void*,
                          void*, void*, uint32_t, uint32_t, uint32_t, void*);
    // 261: fused gate|up-layout bf16 swiglu + e4m3 quant (single-GEMM
    // prefill FFN epilogue)
    int (*quantize_e4m3_swiglu_b16_gu)(const void*, void*, void*, uint32_t,
                                       uint32_t, void*);
    // 262: fused conv1d+SiLU+split+GQA+q/k-norm (DN prefill glue; _at
    // fresh-prompt zero-pad semantics)
    int (*causal_conv1d_silu_qkv)(const void*, const void*, void*, void*,
                                  void*, uint32_t, uint32_t, uint32_t,
                                  uint32_t, uint32_t, void*);
    // 263: bf16-out twin of 262 (the DN bf16-operand chain entry)
    int (*causal_conv1d_silu_qkv_b16)(const void*, const void*, void*, void*,
                                      void*, uint32_t, uint32_t, uint32_t,
                                      uint32_t, uint32_t, void*);
    // 264: chunked DN with bf16 v operand (per-call route; q/k/dw/du f32)
    int (*gated_delta_chunked_vb16)(const void*, const void*, const void*,
                                    const void*, const void*, void*, void*,
                                    void*, void*, void*, void*, uint32_t,
                                    uint32_t, uint32_t, void*);
    // 265: per-32 twin of addnorm_e4m3_row (the f8a/f8r wide-decode band):
    // residual-add + post-norm + pre-norm + per-32 e4m3, bit-identical to
    // rmsnorm_add_scale -> rmsnorm_e4m3_batch
    int (*addnorm_e4m3_b32)(void*, const void*, const void*, const void*,
                            void*, void*, uint32_t, float, float, uint32_t,
                            void*);
    // 266: spec-verify FIN - FA route at n_splits==1 with in-kernel finalize
    // (bit-identical to walk + -inf-sink combine); -2 = geometry can't
    // engage, caller keeps partial+combine
    int (*attn_spec_batch_fin)(const void*, const void*, const void*, void*,
                               void*, const void*, const void*, const void*,
                               uint32_t, uint32_t, uint32_t, uint32_t,
                               uint32_t, uint32_t, uint32_t, uint32_t, float,
                               uint32_t, void*);
    // 267: gu-interleave lin repack (gate/up pair p -> tile rows
    // (p>>3)*16+(p&7) / +8 - the fused-epilogue layout); same params as
    // f8w_repack_lin, additionally requires out_dim % 16 == 0
    int (*f8w_repack_lin_gui)(const void*, const void*, void*, uint32_t,
                              uint32_t, void*);
    // 268: interleaved-plane geglu2 twin (same formula/scale/cvt, pair
    // addressing) - the interleaved plane's non-fused consumers
    int (*quantize_e4m3_geglu2i)(const void*, void*, void*, uint32_t,
                                 uint32_t, void*);
    // 269: fused gu GEMM + geglu + per-32 e4m3 quant on the interleaved
    // plane (wlin, xq, xs, q, qscale, in_dim, out_dim, batch, stream);
    // bit-identical to lin_kt -> geglu2i; returns -2 when the route can't
    // engage (caller keeps the 2-launch chain)
    int (*f8_gemm_lin_gu)(const void*, const void*, const void*, void*,
                          void*, uint32_t, uint32_t, uint32_t, void*);
    // 270: spec-verify LCO - krs spec-FA with in-kernel last-CTA-out combine
    // (bit-identical to partial + -inf-sink combine; out_f receives the
    // combined batch-major rows, out_o/out_ml stay partial scratch);
    // -2 = geometry not covered, caller keeps the partial+combine chain.
    // (q, pool_k, pool_v, out_o, out_ml, sinks, out_f, tickets, positions,
    // slots, block_tables, bps, n_heads, n_kv, head_dim, kv_dim, swa_window,
    // n_splits, rows, k1, scale, kv_dtype, stream)
    int (*attn_spec_lco_paged)(const void*, const void*, const void*, void*,
                               void*, const void*, void*, void*, const void*,
                               const void*, const void*, uint32_t, uint32_t,
                               uint32_t, uint32_t, uint32_t, uint32_t,
                               uint32_t, uint32_t, uint32_t, float, uint32_t,
                               void*);
    // 271: per-channel gu GEMM (kt4a scale-free mainloop):
    // (wlin, xq, as_row f32[batch], ws f32[out_dim] gate|up halves, q, qs,
    // in_dim, out_dim, batch, stream); -2 = route not covered
    int (*f8_gemm_lin_gu_pc)(const void*, const void*, const void*,
                             const void*, void*, void*, uint32_t, uint32_t,
                             uint32_t, void*);
    // 272: pc lin GEMM for the qkv/wo classes (kt4 scale-free twin):
    // (wlin, row_off, xq, as_row f32[batch], ws f32[out_dim] segment-sliced,
    // y, in_dim, out_dim, batch, o16, stream); -2 = route not covered
    int (*f8_gemm_w8_pc)(const void*, uint32_t, const void*, const void*,
                         const void*, void*, uint32_t, uint32_t, uint32_t,
                         uint32_t, void*);
    // 273: down twin (kt4d): weights per-channel, activations per-32 in-loop
    // (wlin, row_off, xq, xs, ws, y, in_dim, out_dim, batch, o16, stream)
    int (*f8_gemm_w8_pcd)(const void*, uint32_t, const void*, const void*,
                          const void*, void*, uint32_t, uint32_t, uint32_t,
                          uint32_t, void*);
    // 274: async spec round token assembly: build the verify
    // tick's slot-major token rows on device from the drafter chain's
    // step-major output plane. (meta[5n], drafts, dst, n, cmax, rr, stream)
    int (*spec_toks)(const void*, const void*, void*, uint32_t, uint32_t,
                     uint32_t, void*);
    // 275: device-side spec accept (rung B1): the accept-while-
    // match walk on device, emitting one compact per-slot strip
    // {accepted, p_final, final_row, new_pending, tokens...}.
    // (sampled, drafts, meta[5n], pos, strip, n, rr, stride, stream)
    int (*spec_accept)(const void*, const void*, const void*, const void*,
                       void*, uint32_t, uint32_t, uint32_t, void*);
    // 276: accept + next-round device prep (rung B2): the strip
    // walk plus everything round N+1's chain and verify need (chain
    // tok/rope/bound, meta pend lane, next verify positions).
    // (sampled, drafts, meta, pos, strip, m_tok, m_pos, m_attn, n, rr,
    //  stride, hold2, stream)
    int (*spec_prep)(const void*, const void*, void*, void*, void*, void*,
                     void*, void*, uint32_t, uint32_t, uint32_t, uint32_t,
                     void*);
    // 277: accepted-final hidden gather into the chain's h input (rung B2).
    // (normed, strip, meta, h, n, n_main, stride, stream)
    int (*spec_hgather)(const void*, const void*, const void*, void*,
                        uint32_t, uint32_t, uint32_t, void*);
    // fused K/V norm+rope+append (kv-epilogue fold): raw k/v planes ->
    // paged cache, kn/vn intermediates never land
    int (*kv_nra_rows)(const void*, const void*, const void*, void*, void*,
                       const void*, const void*, const void*, const void*,
                       uint32_t, uint32_t, uint32_t, float, float, float,
                       float, float, float, float, uint32_t, uint32_t, void*);
    // canonical spec rejection sampling: sampled draft draw +
    // q-store write (logits, invt, uplane, step, qstore, qsum, tok, rows,
    // n, rmax, stream)
    int (*draft_rs)(const void*, const void*, const void*, const void*,
                    void*, void*, void*, uint32_t, uint32_t, uint32_t, void*);
    // RS verify resolve (logits, drafts, qstore, qsum, par, out, nrs, rr,
    // n, rmax, stream)
    int (*spec_rs_resolve)(const void*, const void*, const void*, const void*,
                           const void*, void*, uint32_t, uint32_t, uint32_t,
                           uint32_t, void*);
    // drafter xh stitch (emb, h, xh, r, n_main, stream)
    int (*spec_xh_stitch)(const void*, const void*, void*, uint32_t, uint32_t,
                          void*);
    // host-indexed f32 row gather (src, idx, dst, n, n_main, stream)
    int (*hrow_gather)(const void*, const void*, void*, uint32_t, uint32_t,
                       void*);
    // ---- rowwise (strip-free) pc plane lane ----
    // pc planes quantize per-row pow2, so the in-box per-32 strip was the
    // row exponent repeated - 3.03% dead weight bytes. This lane serves the
    // same logical plane from data-only 16,384B boxes + a per-row ue8m0
    // byte vector (wse, PADDED to the 128-row tail). Bit-exact vs the strip
    // lane. cc12-only (block-scale mma SASS).
    // gu-interleaved data-only repack (data, dst, in_dim, out_dim, stream)
    int (*f8w_repack_lin_bs_gui)(const void*, void*, uint32_t, uint32_t,
                                 void*);
    // decode-band rowwise GEMM (wlin, wse, xq, xs, part, y, in_dim,
    // out_dim, batch, stream) - b <= 64 like f8_gemm_lin
    int (*f8_gemm_lin_r)(const void*, const void*, const void*, const void*,
                         void*, void*, uint32_t, uint32_t, uint32_t, void*);
    // kt3-band rowwise GEMM (wlin, wse, xq, xs, y, in_dim, out_dim, batch,
    // o16, stream) - default kt3+ktz route only
    int (*f8_gemm_lin_kt_r)(const void*, const void*, const void*,
                            const void*, void*, uint32_t, uint32_t, uint32_t,
                            uint32_t, void*);
    // fused gu rowwise (wlin, wse, xq, xs, q, qs, in_dim, out_dim, batch,
    // stream); -2 = route not covered. wse in BOX ROW (interleaved) order.
    int (*f8_gemm_lin_gu_r)(const void*, const void*, const void*,
                            const void*, void*, void*, uint32_t, uint32_t,
                            uint32_t, void*);
    // pc chunk twins on rowwise planes (signatures = the strip pc entries;
    // no wse - their mainloops never read the strip)
    int (*f8_gemm_lin_gu_pc_r)(const void*, const void*, const void*,
                               const void*, void*, void*, uint32_t, uint32_t,
                               uint32_t, void*);
    int (*f8_gemm_w8_pc_r)(const void*, uint32_t, const void*, const void*,
                           const void*, void*, uint32_t, uint32_t, uint32_t,
                           uint32_t, void*);
    int (*f8_gemm_w8_pcd_r)(const void*, uint32_t, const void*, const void*,
                            const void*, void*, uint32_t, uint32_t, uint32_t,
                            uint32_t, void*);
    // fused qkv single-launch on the rowwise plane: `(wlin, xq,
    // as_row, ws, yq, yk, yv, in_dim, q_dim, kv_dim, batch, stream)`.
    // Pure append - no PD_ABI_VERSION bump.
    int (*f8_gemm_w8_pc_qkv_r)(const void*, const void*, const void*,
                               const void*, void*, void*, void*, uint32_t,
                               uint32_t, uint32_t, uint32_t, void*);
    // chunk-band 16-bit streams (pure appends, 291..295): the o16
    // fused-qkv twin (+o16 flag before stream) and the four bf16-in consumer
    // twins (+input-class flag before stream).
    int (*f8_gemm_w8_pc_qkv_r2)(const void*, const void*, const void*,
                                const void*, void*, void*, void*, uint32_t,
                                uint32_t, uint32_t, uint32_t, uint32_t, void*);
    int (*qkv_norm_rope_batch2)(const void*, const void*, const void*,
                                const void*, const void*, void*, void*, void*,
                                const void*, const void*, uint32_t, uint32_t,
                                uint32_t, float, float, float, float, float,
                                float, float, uint32_t, uint32_t, void*);
    int (*kv_nra_rows2)(const void*, const void*, const void*, void*, void*,
                        const void*, const void*, const void*, const void*,
                        uint32_t, uint32_t, uint32_t, float, float, float,
                        float, float, float, float, uint32_t, uint32_t,
                        uint32_t, void*);
    int (*addnorm_e4m3_row2)(void*, const void*, const void*, const void*,
                             void*, void*, uint32_t, float, float, uint32_t,
                             uint32_t, void*);
    int (*rmsnorm_add_scale2)(void*, const void*, const void*, uint32_t,
                              float, float, uint32_t, uint32_t, void*);
    // attention streams (pure appends, 296..303): f16 pf_qn/pf_attn
    // planes on the mixed-tick route - nr gains an o16-out flag, the
    // prefill/spec/walk/combine attention entries gain an a16 flag, and the
    // two e4m3 quantizers gain f16-in twins.
    int (*qkv_norm_rope_batch3)(const void*, const void*, const void*,
                                const void*, const void*, void*, void*, void*,
                                const void*, const void*, uint32_t, uint32_t,
                                uint32_t, float, float, float, float, float,
                                float, float, uint32_t, uint32_t, uint32_t,
                                void*);
    int (*attn_prefill_f16_paged2)(const void*, const void*, const void*,
                                   const void*, void*, const void*, const void*,
                                   const void*, uint32_t, uint32_t, uint32_t,
                                   uint32_t, uint32_t, uint32_t, uint32_t,
                                   float, uint32_t, uint32_t, void*);
    int (*attn_spec_batch_paged2)(const void*, const void*, const void*, void*,
                                  void*, const void*, const void*, const void*,
                                  uint32_t, uint32_t, uint32_t, uint32_t,
                                  uint32_t, uint32_t, uint32_t, uint32_t,
                                  uint32_t, float, uint32_t, uint32_t, void*);
    int (*attn_decode_batch_paged2)(const void*, const void*, const void*,
                                    const void*, void*, const void*, const void*,
                                    const void*, uint32_t, uint32_t, uint32_t,
                                    uint32_t, uint32_t, uint32_t, uint32_t,
                                    float, uint32_t, uint32_t, void*);
    int (*attn_decode_batch_partial_paged2)(const void*, const void*, const void*,
                                            void*, void*, const void*, const void*,
                                            const void*, uint32_t, uint32_t,
                                            uint32_t, uint32_t, uint32_t, uint32_t,
                                            uint32_t, uint32_t, float, uint32_t,
                                            uint32_t, void*);
    int (*attn_decode_batch_combine2)(const void*, const void*, const void*,
                                      void*, uint32_t, uint32_t, uint32_t,
                                      uint32_t, uint32_t, void*);
    int (*quantize_e4m3_i16)(const void*, void*, void*, uint32_t, uint32_t,
                             void*);
    int (*quantize_e4m3_row_i16)(const void*, void*, void*, uint32_t, uint32_t,
                                 uint32_t, void*);
    // Laguna lane (pure appends, 304..306; the gemma4 slots ahead of them
    // stay put). Per-head softplus gate (x, gate, n_heads, head_dim, rows, stream)
    int (*mul_softplus_head)(void*, const void*, uint32_t, uint32_t, uint32_t,
                             void*);
    // Laguna sigmoid MoE router (logits, bias, routed_scale, n_expert, k,
    // out_idx, out_w, batch, stream)
    int (*moe_topk_sigmoid_batch)(const void*, const void*, float, uint32_t,
                                  uint32_t, void*, void*, uint32_t, void*);
    // Laguna decode-tick epilogue fold: q/k norm + rope (yarn or sectioned
    // mrope via mpos != null) + paged k/v append in one launch (q_src, q_off,
    // q_stride, k_src, k_off, k_stride, v_src, v_stride, qw, kw, q_out,
    // k_pool, v_pool, positions, slots, mpos, block_tables, bps, n_head,
    // n_kv, head_dim, n_rot, eps, theta_scale, freq_scale, corr_low,
    // corr_high, ext_factor, mscale, s0..s3, rows, kv_dtype, stream)
    int (*lag_qk_nra_rows)(const void*, uint32_t, uint32_t, const void*,
                           uint32_t, uint32_t, const void*, uint32_t,
                           const void*, const void*, void*, void*, void*,
                           const void*, const void*, const void*, const void*,
                           uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                           float, float, float, float, float, float, float,
                           uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                           uint32_t, void*);
    // Standalone scalar multiply x[..n] *= s (ggml_scale's shape) - granite's
    // embedding/logit multipliers; its residual multiplier uses scale_add_f32
    // (x += w*y) instead. (x, s, n, stream)
    int (*scale_f32)(void*, float, uint32_t, void*);
    // NORM-convention rope (llama.cpp ROPE_TYPE_NORM): interleaved (2k, 2k+1)
    // pairs. Same signature as rope_yarn_batch. (x, positions, n_heads,
    // head_dim, theta_scale, freq_scale, corr_low, corr_high, ext_factor,
    // mscale, batch, stream)
    int (*rope_yarn_batch_norm)(void*, const void*, uint32_t, uint32_t, float,
                                float, float, float, float, float, uint32_t,
                                void*);
    // Qwen3.5-family fused-plane prefill consumer (309): the
    // one-GEMM [q|gate interleaved | k | v] plane -> q norm+mrope -> q_out,
    // raw gate -> gate_out, k norm+mrope + v raw -> paged pools. (qkg, q_off,
    // row_stride, k_off, v_off, qw, kw, q_out, gate_out, k_pool, v_pool,
    // positions, slots, mpos, block_tables, bps, n_head, n_kv, head_dim,
    // n_rot, eps, theta_scale, freq_scale, corr_low, corr_high, ext_factor,
    // mscale, s0..s3, rows, kv_dtype, stream)
    int (*q36_qkg_nra_rows)(const void*, uint32_t, uint32_t, uint32_t,
                            uint32_t, const void*, const void*, void*, void*,
                            void*, void*, const void*, const void*,
                            const void*, const void*, uint32_t, uint32_t,
                            uint32_t, uint32_t, uint32_t, float, float, float,
                            float, float, float, float, uint32_t, uint32_t,
                            uint32_t, uint32_t, uint32_t, uint32_t, void*);
    // 310: q36 DN rung-3 two-buffer kt3 GEMM over a fused lin plane
    // `(wlin, xq, xs, y, y2, ncut, in_dim, out_dim, batch, stream)`;
    // rows [0,ncut) -> y, [ncut,out) -> y2, each at its own stride.
    // -2 = route not covered (caller keeps its two-launch pair).
    // Pure append - no PD_ABI_VERSION bump.
    int (*f8_gemm_lin_kt_split)(const void*, const void*, const void*, void*,
                                void*, uint32_t, uint32_t, uint32_t, uint32_t,
                                void*);
    // 311: batched cross/self attention for granite-vision's windowed Q-Former
    // `(q, k, v, out, nq, nkv, n_heads, head_dim, n_batch, scale, stream)`.
    int (*vision_attn_x)(const void*, const void*, const void*, void*, uint32_t,
                         uint32_t, uint32_t, uint32_t, uint32_t, float, void*);
    // 312: row gather with averaging fan-in - window/unwindow permutations
    // (k=1) and the 2x2 area-interpolate downsampler (k=4)
    // `(src, idx, out, rows, k, width, stream)`.
    int (*gather_rows_avg)(const void*, const void*, void*, uint32_t, uint32_t,
                           uint32_t, void*);
    // 313: exact-erf GELU in place - Not the tanh `gelu` `(x, n, stream)`.
    int (*gelu_erf)(void*, uint64_t, void*);
    // 314: broadcast row add `x[r] += src[r % src_rows]`
    // `(x, src, rows, src_rows, width, stream)`.
    int (*add_rows_bcast)(void*, const void*, uint32_t, uint32_t, uint32_t, void*);
    // 315: pipelined sibling of kquant_gemm_w4a8 - same signature/numerics,
    // the >64-batch rung's raw weight+scale bytes ride cp.async into a
    // shared buffer instead of a synchronous global load, with the next
    // super-block's fetch overlapping this one's MMA compute (ports
    // kquant_gemm_mma_ks's proven cp.async technique onto the 128x128 tile).
    int (*kquant_gemm_w4a8_pipe)(const void*, const void*, const void*, const void*,
                                 void*, uint32_t, uint32_t, uint32_t, uint32_t, void*);
    // 316: kquant_gemm_w4a8_pipe's genuinely-double-buffered sibling - same
    // signature/numerics, a real 2-deep raw byte ring (half-width tile_x to
    // afford the second copy) so the next super-block's load overlaps
    // both build+compute of the current one, not just compute like the
    // single-buffer pipe kernel. Stays __launch_bounds__(256,1): a 2
    // blocks/SM attempt (kquant_gemm_w4a8_pipe_hi) hit its register target
    // but sm_120's SM shared-memory budget (102,400 B, barely above its own
    // 101,376 B single-block opt-in max) blocked occupancy from actually
    // rising - so that attempt was reverted.
    int (*kquant_gemm_w4a8_pipe2)(const void*, const void*, const void*, const void*,
                                   void*, uint32_t, uint32_t, uint32_t, uint32_t, void*);
    // 317: multi-segment q8_0_gemv_repacked - One launch over up to three
    // same-in_dim planes sharing one activation (decode QKV / FFN gate|up
    // merge; small-grid launches waste ramp/drain, see the kernel note).
    // Bit-identical per row to the split launches. (d0,s0,b0,y0,rows0,
    // d1,s1,b1,y1,rows1, d2,s2,b2,y2,rows2, x, in_dim, n_segs, stream)
    int (*q8_0_gemv_repacked_multi)(const void*, const void*, const void*, void*, uint32_t,
                                    const void*, const void*, const void*, void*, uint32_t,
                                    const void*, const void*, const void*, void*, uint32_t,
                                    const void*, uint32_t, uint32_t, void*);
    // 318: fused NORM-rope(q in place) + NORM-rope(k)->paged append + v paged
    // append - granite's 4-launch rope/append band as one kernel, cache and q
    // bytes bit-identical. (q, k, v, pool_k, pool_v, positions, slots,
    // block_tables, bps, n_heads, n_kv, head_dim, theta_scale, freq_scale,
    // corr_low, corr_high, ext_factor, mscale, batch, kv_dtype, stream)
    int (*rope_norm_qk_append_paged)(void*, void*, const void*, void*, void*,
                                     const void*, const void*, const void*, uint32_t,
                                     uint32_t, uint32_t, uint32_t, float, float, float,
                                     float, float, float, uint32_t, uint32_t, void*);
    // 319: W4A8 k-quant multi-segment decode GEMV - up to 3 same-in_dim
    // planes (mixed k-quant dtypes allowed) sharing one staged int8
    // activation, one launch (granite-30b QKV / gate|up merge; entry 317's
    // launch economics on the k-quant family). xsums may be null when no
    // segment is Q4_K/Q5_K. (d0,s0,y0,out0,dt0, d1,s1,y1,out1,dt1,
    // d2,s2,y2,out2,dt2, xq,xs,xsums, in_dim, n_segs, stream)
    int (*kquant_gemv_w4a8_multi)(const void*, const void*, void*, uint32_t, uint32_t,
                                  const void*, const void*, void*, uint32_t, uint32_t,
                                  const void*, const void*, void*, uint32_t, uint32_t,
                                  const void*, const void*, const void*,
                                  uint32_t, uint32_t, void*);
    // 320: multi-segment nc GEMV - up to four same-in_dim Q8_0 planes sharing
    // one staged int8 activation (ncols = 1..8 columns each), one launch (the
    // r=2..4 batched-decode q|k|v|g and shexp gate|up merges; entry 317's
    // launch economics on the multi-column class). Per-segment nullable bias.
    // (d0,s0,b0,y0,out0, d1,s1,b1,y1,out1, d2,s2,b2,y2,out2, d3,s3,b3,y3,out3,
    // xq, xs, in_dim, n_segs, ncols, stream)
    int (*q8_0_gemv_dp4a_nc_multi)(const void*, const void*, const void*, void*, uint32_t,
                                   const void*, const void*, const void*, void*, uint32_t,
                                   const void*, const void*, const void*, void*, uint32_t,
                                   const void*, const void*, const void*, void*, uint32_t,
                                   const void*, const void*, uint32_t, uint32_t,
                                   uint32_t, void*);
    // 321: packed multi-span gated delta recurrence - decode rows (len-1
    // items), independent short span walks, and same-slot FUSED CKPT TAIL
    // chains (one item per chain, contiguous rows) in one launch via u32
    // descriptors of stride 8 (row0, len, slot, snapA_t, snapA_sel, snapB_t,
    // snapB_sel, pad). Internal chain seams write in-kernel state snapshots
    // to snap0/snap1 (per-layer pre-offset stage-blob state regions; sel
    // picks the blob, t==0 = none). Rows addressed absolutely, distinct
    // slots per item. (q, k, v, g, beta, items, states, out, snap0, snap1,
    // n_items, n_heads, head_dim, stream)
    int (*gated_delta_recurrent_v2_packed)(const void*, const void*, const void*,
                                           const void*, const void*, const void*,
                                           void*, void*, void*, void*, uint32_t,
                                           uint32_t, uint32_t, void*);
    // 322: pf7 varlen packed prefill attention (AF3) - one launch
    // per layer covering every eligible prefill span of the tick. vl_items
    // is stride-4 u32 per 64-head-row tile: (q_row0, span_rows,
    // tile_flat_row0, slot); tiles never cross spans so each packed CTA is
    // bit-identical to its per-span pf7 twin. fp8 pools, hd256, G 4/6/8
    // only - anything else returns an error (the engine pre-checks and the
    // per-span path stays the fallback). (q, pool_k, pool_v, sinks, out,
    // positions, vl_items, n_tiles, block_tables, blocks_per_slot, n_heads,
    // n_kv_heads, head_dim, kv_dim, swa_window, scale, kv_dtype, stream)
    int (*attn_prefill_f16_paged_vl)(const void*, const void*, const void*,
                                     const void*, void*, const void*,
                                     const void*, uint32_t, const void*,
                                     uint32_t, uint32_t, uint32_t, uint32_t,
                                     uint32_t, uint32_t, float, uint32_t,
                                     void*);
    // 323: varlen chunked-GDN (GDN formulation band) - One
    // stage1 + register-state-walk launch pair covers every eligible span
    // of the tick. chunk_items: stride-2 u32 (global row0, chunk len) per
    // launch chunk; span_items: stride-4 u32 (first launch chunk, span
    // rows, state f32 offset, out row0) per span. Per-span math identical
    // to the per-span RS calls; RS-route env gates mirrored inside, any
    // other elected arm returns cudaErrorNotSupported (engine falls back
    // per-span). (q, k, v, g, beta, states, out, dw, du, aqk, cg,
    // chunk_items, n_chunks, span_items, n_spans, n_tokens, n_heads,
    // head_dim, stream)
    int (*gated_delta_chunked_rs_vl)(const void*, const void*, const void*,
                                     const void*, const void*, void*, void*,
                                     void*, void*, void*, void*, const void*,
                                     uint32_t, const void*, uint32_t,
                                     uint32_t, uint32_t, uint32_t, void*);
    // 324: fused-GLU W4A8 decode GEMV  - gate+up+SwiGLU as one
    // launch: each block walks the gate row and the matching up row over
    // one staged activation and writes silu(g)*u directly. Bit-exact vs
    // the multi<4,128>+swiglu split path (identical row walks, identical
    // epilogue expression). (gate_data, gate_scales, up_data, up_scales,
    // xq, xs, xsums, y, in_dim, out_dim, gate_dtype, up_dtype, stream)
    int (*kquant_gemv_w4a8_glu)(const void*, const void*, const void*,
                                const void*, const void*, const void*,
                                const void*, void*, uint32_t, uint32_t,
                                uint32_t, uint32_t, void*);
    // 325: qwen twin of addnorm_e4m3_row -- PLAIN residual add (no post-norm,
    // no stream scale) + pre-norm + row-e4m3 in one launch. Bit-identical to
    // add_rmsnorm_batch + quantize_e4m3_row.
    // (x_inout, proj, pre_w, q, row_scale, n, eps, rows, stream)
    int (*add_rmsnorm_e4m3_row)(void*, const void*, const void*, void*, void*,
                                uint32_t, float, uint32_t, void*);
    // 326: up-to-4 slices of one fused landing in a single launch.
    // (src, src_stride, rows, d0,o0,w0, d1,o1,w1, d2,o2,w2, d3,o3,w3, stream)
    int (*row_slice4)(const void*, uint32_t, uint32_t,
                      void*, uint32_t, uint32_t, void*, uint32_t, uint32_t,
                      void*, uint32_t, uint32_t, void*, uint32_t, uint32_t, void*);
    // 327: swiglu of a fused gate|up landing -> per-ROW e4m3, one launch.
    // (fused, q, row_scale, ff, rows, stream)
    int (*swiglu_e4m3_row)(const void*, void*, void*, uint32_t, uint32_t, void*);
    // ---- whisper decode lane  -------------------------------
    // 328: batched single-query flash-decoding attention over the slot K/V
    // planes, f16 or fp8-e4m3 per kv_dtype.
    // (q, qbias, k, v, slots, lens, out, part, kv_stride, kv_len_def,
    //  len_bias, n_heads, hd, batch, scale, kv_dtype, stream)
    int (*whisper_dec_attn)(const void*, const void*, const void*, const void*, const void*,
                            const void*, void*, void*, uint32_t, uint32_t, uint32_t,
                            uint32_t, uint32_t, uint32_t, float, uint32_t, void*);
    // 329: token embedding row + LEARNED position row, per slot.
    // (tok, postab, tokens, pos, x, d, batch, stream)
    int (*whisper_embed_pos)(const void*, const void*, const void*, const void*, void*,
                             uint32_t, uint32_t, void*);
    // 330: split a merged q|k|v landing, append K/V to the slot self-caches
    // at kv_dtype.
    // (qkv, bq, bv, q, kc, vc, slots, pos, d, ctx, batch, kv_dtype, stream)
    int (*whisper_qkv_split)(const void*, const void*, const void*, void*, void*, void*,
                             const void*, const void*, uint32_t, uint32_t, uint32_t,
                             uint32_t, void*);
    // 331: store a window's cross K or V into its slot plane at kv_dtype,
    // bias folded. (src, bias, dst, slots, rows, d, stride, batch, kv_dtype, stream)
    int (*whisper_kv_store)(const void*, const void*, void*, const void*, uint32_t, uint32_t,
                            uint32_t, uint32_t, uint32_t, void*);
    // 332: LayerNorm -> f16 landing. (x, w, b, out, rows, n, eps, stream)
    int (*whisper_ln_f16)(const void*, const void*, const void*, void*, uint32_t, uint32_t,
                          float, void*);
    // 333: x += proj + bias, then the next pre-norm out of it, at f16.
    // (x, proj, bias, w, b, out, rows, n, eps, stream)
    int (*whisper_res_ln_f16)(void*, const void*, const void*, const void*, const void*,
                              void*, uint32_t, uint32_t, float, void*);
    // 334: bias + erf-GELU + f16 cast on the fc1 landing.
    // (x, bias, out, rows, n, stream)
    int (*whisper_bias_gelu_f16)(const void*, const void*, void*, uint32_t, uint32_t, void*);
    // 335: bias + SiLU + f16 cast on a macaron FFN landing.
    // (x, bias, out, rows, n, stream)
    int (*gs_bias_silu_f16)(const void*, const void*, void*, uint32_t, uint32_t, void*);
    // 336: bias + sigmoid-GLU over a [rows, 2*d] channel split -> [rows, d].
    // (x, bias, out, rows, d, stream)
    int (*gs_bias_glu)(const void*, const void*, void*, uint32_t, uint32_t, void*);
    // 337: centered depthwise conv over time + folded BatchNorm + SiLU, f16 out.
    // (x, w, bnw, bnb, out, rows, d, k, stream)
    int (*gs_dwconv_bn_silu_f16)(const void*, const void*, const void*, const void*, void*,
                                 uint32_t, uint32_t, uint32_t, void*);
    // 338: conformer blockwise attention with Shaw relative position embeddings.
    // (qkv, out, rel, rows, ctx, n_heads, hd, max_pos, scale, stream)
    int (*gs_conf_attn)(const void*, void*, const void*, uint32_t, uint32_t, uint32_t,
                        uint32_t, uint32_t, float, void*);
    // 339: bias + row softmax + f16 cast (the CTC branch head).
    // (x, bias, out, rows, n, stream)
    int (*gs_bias_softmax_f16)(const void*, const void*, void*, uint32_t, uint32_t, void*);
    // 340: x += s*(proj + bias), then the next pre-norm out of it, at f16.
    // (x, proj, bias, w, b, out, rows, n, s, eps, stream)
    int (*gs_res_ln_f16)(void*, const void*, const void*, const void*, const void*, void*,
                         uint32_t, uint32_t, float, float, void*);
    // 341: x = LN(x + s*(proj + bias)) in place, plus an f16 landing.
    // (x, proj, bias, w, b, out, rows, n, s, eps, stream)
    int (*gs_post_ln_f16)(void*, const void*, const void*, const void*, const void*, void*,
                          uint32_t, uint32_t, float, float, void*);
    // 342: gated rmsnorm + per-ROW e4m3 (DN out_proj decode arm).
    // d == 128, n_heads % 16 == 0; out nullable; bit-identical to
    // gated_rmsnorm + quantize_e4m3_row.
    // (x, z, w, out, q, rscale, batch, n_heads, d, eps, stream)
    int (*gated_rmsnorm_e4m3_row)(const void*, const void*, const void*, void*,
                                  void*, void*, uint32_t, uint32_t, uint32_t,
                                  float, void*);
    // 343: argmax + the runner-up + {log p(top1), p(probe), log p(top2), H2},
    // one pass. WIDENED in PLACE (was argmax_logprob_rows, a u32
    // out plus a [rows,2] stats): the slot keeps its position, but its name
    // and arity changed so every in-tree caller fails to COMPILE rather than
    // silently passing the old argument list - the whole consumer set is in
    // this repo (whisper's decode tick and its parity gate), so a rename is
    // cheaper than a second near-identical reduction to keep in parity.
    // (logits, out_u32, alt_u32, stats_f32, probe, rows, n, stream)
    int (*argmax_top2_rows)(const void*, void*, void*, void*, uint32_t, uint32_t, uint32_t,
                            void*);
    // 344: whisper's ApplyTimestampRules over a logits block, in place.
    // (logits, state_u32, rows, vocab, eot, no_ts, ts_begin, max_init, stream)
    int (*whisper_ts_rules)(void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t,
                            uint32_t, uint32_t, void*);
    // 345: row_slice4's DN split with delta_gate folded into the ab parts -
    // slots 0/1 copy (mixed, z), the 2*n_heads ab cols at ab_off become
    // g/beta directly. Bit-identical to row_slice4 + delta_gate.
    // (src, stride, rows, d0, o0, w0, d1, o1, w1, ab_off, n_heads,
    //  ssm_a, dt_bias, g, beta, stream)
    int (*row_slice2_gate)(const void*, uint32_t, uint32_t, void*, uint32_t,
                           uint32_t, void*, uint32_t, uint32_t, uint32_t,
                           uint32_t, const void*, const void*, void*, void*,
                           void*);
    // flat-scale e4m3 expert lane: f32 -> e4m3 + per-32 f32 scale
    // (x, q, scale, n), and the sorted gate_up GEMM over per-ROW-scaled e4m3
    // weight planes (gate_data, gate_rs, up_data, up_rs, sorted_row,
    // block_expert, xq, xs, fq, fs, in_dim, ff, max_blocks, bm)
    int (*quantize_e4m3_b32f)(const void*, void*, void*, uint32_t, void*);
    int (*f8row_moe_gate_up_mma_geglu)(const void*, const void*, const void*,
                                       const void*, const void*, const void*,
                                       const void*, const void*, void*, void*,
                                       uint32_t, uint32_t, uint32_t, uint32_t, void*);
    // same signature; epilogue emits e4m3 per-32 for the flat-scale down half
    int (*f8row_moe_gate_up_mma_geglu_f8)(const void*, const void*, const void*,
                                          const void*, const void*, const void*,
                                          const void*, const void*, void*, void*,
                                          uint32_t, uint32_t, uint32_t, uint32_t, void*);
    // (down_data, down_rs, sorted_row, sorted_slot, block_expert, topk_w, fq,
    //  fs, part, ff, embd, n_active, max_blocks, bm)
    int (*f8row_moe_down_mma)(const void*, const void*, const void*, const void*,
                              const void*, const void*, const void*, const void*,
                              void*, uint32_t, uint32_t, uint32_t, uint32_t,
                              uint32_t, void*);
    // 350: conv-window VL store - each fresh span's last (k-1) pre-conv rows
    // into its slot's conv window, span (row0, take, slot, _) quads read from
    // device contents (chunk-tick graph capture).
    // (src, spans, win, n_spans, km1, conv_dim, stream)
    int (*conv_win_store_vl)(const void*, const void*, void*, uint32_t,
                             uint32_t, uint32_t, void*);
    // 351-354: BF16 dense weight planes - per-TENSOR quant dispatch for mixed
    // UD files (muse-glimmer ships token_embd/output/attn_k/attn_v
    // at bf16 next to Q8_0 everything else). Weights stay bf16 in DRAM,
    // activations f32, accumulation f32 - same class as the Q8_0 lane and
    // strictly more precise, so the same-weights parity target survives.
    // Layout matches the repacked Q8_0 planes: out rows of in_dim contiguous.
    // (w, bias, x, y, in_dim, out_dim, stream)
    int (*bf16_gemv_f32)(const void*, const void*, const void*, void*,
                         uint32_t, uint32_t, void*);
    // (w, bias, x, y, in_dim, out_dim, batch, stream)
    int (*bf16_gemm_f32)(const void*, const void*, const void*, void*,
                         uint32_t, uint32_t, uint32_t, void*);
    // bf16 -> f32 widen in the DequantF32Fn shape (32 elems/"block") so it
    // slots into the engine's dequant_for table
    int (*bf16_dequant_f32)(const void*, void*, uint64_t, void*);
    // bf16 twin of embed_gather_q8 (fused output scale, device token ids)
    int (*embed_gather_bf16)(const void*, const void*, void*, uint32_t,
                             uint32_t, float, void*);
    // 355-363: SiLU twins of the whole gated-FFN carrier set (-
    // muse-glimmer's FFN is SwiGLU where gemma4's is GeGLU). Each is the same
    // kernel instantiated on pd_glu_act<PD_ACT_SILU>; the GELU entries above
    // are untouched and still byte-for-byte what shipped. Separate slots
    // rather than an act argument because the table's growth rule is
    // append-only - an existing signature may never move.
    // (x, ff, rows, stream) - in-place fold over a [rows, 2*ff] concat row
    int (*swiglu_pair)(void*, uint32_t, uint32_t, void*);
    // (gu, q, scale, n_ff, rows, stream) - per-32 e4m3, concat plane
    int (*quantize_e4m3_swiglu2)(const void*, void*, void*, uint32_t, uint32_t,
                                 void*);
    // same, on the gu-INTERLEAVED plane (f8w_repack_lin_gui's layout)
    int (*quantize_e4m3_swiglu2i)(const void*, void*, void*, uint32_t, uint32_t,
                                  void*);
    // (gu, q, rscale, n_ff, rows, stream) - per-ROW e4m3 into a compact plane
    int (*quantize_e4m3_swiglu2_row)(const void*, void*, void*, uint32_t,
                                     uint32_t, void*);
    // nz-aware twin: gu is the fused GEMM's nz partial planes
    // (gu, q, rscale, n_ff, rows, nzp, stream)
    int (*quantize_e4m3_swiglu2_nz)(const void*, void*, void*, uint32_t,
                                    uint32_t, uint32_t, void*);
    // fused gu GEMM + SiLU-glu + per-32 e4m3 quant, the four plane classes:
    // strip/rowwise x per-32/per-channel. -2 = route not covered (caller
    // keeps the 2-launch chain), exactly the GELU entries' convention.
    // (wlin, xq, xs, q, qs, in_dim, out_dim, batch, stream)
    int (*f8_gemm_lin_gu_silu)(const void*, const void*, const void*, void*,
                               void*, uint32_t, uint32_t, uint32_t, void*);
    // (wlin, wse, xq, xs, q, qs, in_dim, out_dim, batch, stream)
    int (*f8_gemm_lin_gu_r_silu)(const void*, const void*, const void*,
                                 const void*, void*, void*, uint32_t, uint32_t,
                                 uint32_t, void*);
    // (wlin, xq, as_row, ws, q, qs, in_dim, out_dim, batch, stream)
    int (*f8_gemm_lin_gu_pc_silu)(const void*, const void*, const void*,
                                  const void*, void*, void*, uint32_t, uint32_t,
                                  uint32_t, void*);
    int (*f8_gemm_lin_gu_pc_r_silu)(const void*, const void*, const void*,
                                    const void*, void*, void*, uint32_t,
                                    uint32_t, uint32_t, void*);
    // 364-365: ROPE_TYPE_NORM twins of the two rope carriers this engine's
    // gemma4-shaped families use. muse-glimmer ropes NORM -
    // interleaved (2k, 2k+1) pairs - where gemma4, whose graph it shares,
    // ropes NEOX's half-split (k, k+half). Same angles either way; only the
    // pairing differs, and getting it wrong scrambles position on every
    // roped layer while still producing fluent-looking text.
    // (x, positions, factors, n_heads, head_dim, theta_scale, freq_scale,
    //  corr_low, corr_high, ext_factor, mscale, batch, stream)
    int (*rope_factors_batch_norm)(void*, const void*, const void*, uint32_t,
                                   uint32_t, float, float, float, float, float,
                                   float, uint32_t, void*);
    // fused QK-norm + rope, superset of batch3: `neox` joins i16/o16 as a
    // shape bit (1 = the half-split layout every earlier caller assumes).
    // (q, k, v, qw, kw, qn, kn, vn, positions, factors, n_head, n_kv,
    //  head_dim, eps, theta_scale, freq_scale, corr_low, corr_high,
    //  ext_factor, mscale, rows, i16, o16, neox, stream)
    int (*qkv_norm_rope_batch4)(const void*, const void*, const void*,
                                const void*, const void*, void*, void*, void*,
                                const void*, const void*, uint32_t, uint32_t,
                                uint32_t, float, float, float, float, float,
                                float, float, uint32_t, uint32_t, uint32_t,
                                uint32_t, void*);
    // fused QK-norm + rope, superset of batch4: `vnorm` says whether the V
    // slots get the weightless per-head RMS norm. gemma4 does (its reference
    // graph runs ggml_rms_norm on Vcur); muse-glimmer hands the raw Vcur to
    // build_attn and must not. Carried by the architecture, in no metadata key.
    // (q, k, v, qw, kw, qn, kn, vn, positions, factors, n_head, n_kv,
    //  head_dim, eps, theta_scale, freq_scale, corr_low, corr_high,
    //  ext_factor, mscale, rows, i16, o16, neox, vnorm, stream)
    int (*qkv_norm_rope_batch5)(const void*, const void*, const void*,
                                const void*, const void*, void*, void*, void*,
                                const void*, const void*, uint32_t, uint32_t,
                                uint32_t, float, float, float, float, float,
                                float, float, uint32_t, uint32_t, uint32_t,
                                uint32_t, uint32_t, void*);
    // kv_nra_rows2 + the same two arch constants (this kernel is batch5's
    // K/V half folded into the paged append, so it needs both).
    // (kp, vp, kw, k_pool, v_pool, positions, slots, factors, block_tables,
    //  blocks_per_slot, n_kv, head_dim, eps, theta_scale, freq_scale,
    //  corr_low, corr_high, ext_factor, mscale, rows, kv_dtype, i16, neox,
    //  vnorm, stream)
    int (*kv_nra_rows3)(const void*, const void*, const void*, void*, void*,
                        const void*, const void*, const void*, const void*,
                        uint32_t, uint32_t, uint32_t, float, float, float,
                        float, float, float, float, uint32_t, uint32_t,
                        uint32_t, uint32_t, uint32_t, void*);
    // gemma_qkv_nra2s + the three architecture constants its epilogue used to
    // hardcode: freq_scale (0 = NoPE, a bit-exact identity rotation), neox
    // (pair layout), vnorm (whether V is RMS-normed). This is the BATCHED
    // DECODE epilogue - prefill rides qkv_norm_rope_batch5 / kv_nra_rows3, so
    // a model whose graph disagreed with the old constants came out correct on
    // the prompt and wrong on every generated token.
    // (qp, kp, vp, wq_norm, wk_norm, q_out, kc, vc, positions, slots, factors,
    //  block_tables, bps, n_head, n_kv, head_dim, max_ctx, batch, eps,
    //  theta_scale, kv_dtype, qkv_stride, freq_scale, neox, vnorm, stream)
    int (*gemma_qkv_nra3)(void*, void*, void*, const void*, const void*,
                          void*, void*, void*, const void*, const void*,
                          const void*, const void*, uint32_t, uint32_t,
                          uint32_t, uint32_t, uint32_t, uint32_t, float, float,
                          uint32_t, uint32_t, float, uint32_t, uint32_t, void*);
    // 369: softmax(QK^T) over the encoder frames for NOMINATED cross-attention
    // heads - whisper's word-timing read-out. dec_attn is
    // flash-style and never materialises this; word timing is opt-in, so it
    // gets its own kernel rather than a cost on every decode.
    // (q, qbias, k, slots, heads, out, kv_stride, n_enc, n_heads, hd, n_sel,
    //  batch, scale, kv_dtype, stream)
    int (*whisper_xattn_probs)(const void*, const void*, const void*, const void*,
                               const void*, void*, uint32_t, uint32_t, uint32_t,
                               uint32_t, uint32_t, uint32_t, float, uint32_t, void*);
    // rope2d_neox plus the pair layout it used to hardcode. muse-glimmer's
    // vision tower ropes NORM (adjacent-element pairs) where gemma4v's ropes
    // NEOX - the ggml_rope_ext `mode` argument in the reference clip graph,
    // not a tuning knob and not in any GGUF key.
    // (x, pos_x, pos_y, n_tokens, n_heads, head_dim, theta_scale, neox, stream)
    int (*rope2d)(void*, const void*, const void*, uint32_t, uint32_t, uint32_t,
                  float, uint32_t, void*);
    // Pixel-shuffle merge: out[o][c*k + s] = src[idx[o*k + s]][c]. Distinct
    // from gather_rows_avg (which pools the k rows) and from a plain k-row
    // concat (which is spatial-outer, not channel-outer).
    // (src, idx, out, rows, k, width, stream)
    int (*pixel_shuffle_rows)(const void*, const void*, void*, uint32_t, uint32_t,
                              uint32_t, void*);
    //  (slot 374): (pool, vdim, positions, slots, block_tables,
    // blocks_per_slot, kv_dim, rows, stream)
    int (*vdim_sync)(const void*, void*, const void*, const void*, const void*,
                     uint32_t, uint32_t, uint32_t, void*);
    // slot 375: (vdim_base) -> registers the twin pool for the VD launcher
    int (*vdim_register)(void*);
    // slot 376: (run_offs, n_runs, max_n) -> arms the batched-runs prefill
    // attention for the current coalesced pass; null disarms
    int (*pf_runs_register)(const void*, unsigned int, unsigned int);
    // slot 378: bf16-in whole-row glu2 quantize (act: 0=gelu, 1=silu)
    int (*quantize_e4m3_glu2_row_b16)(const void*, void*, void*, unsigned int,
                                      unsigned int, unsigned int, void*);
    // slot 379: bf16 -> e4m3 + F32 per-row scale (bf16, f8_data, row_scale,
    // in_dim, out_dim, stream). The f32-scale sibling of bf16_to_f8r; what
    // lets a bf16 lm_head build an F8RowPlane and reach the f8t tile route.
    int (*bf16_to_f8row)(const void*, void*, void*, uint32_t, uint32_t, void*);
    // slot 380: SAM ViTDet attention with the decomposed relative-position
    // bias (DeepSeek-OCR's first tower). q/k/v/out are
    // [n_batch, side², heads, hd] f32; rh/rw are [side, side, hd] f32 host-
    // prepared per-geometry bias tables shared across the batch. Windowed
    // blocks pass n_batch = windows / side = 14, global blocks n_batch = views
    // / side = grid. -3 = shape not covered (hd or side over 64).
    // (q, k, v, rh, rw, out, n_batch, side, n_heads, hd, scale, stream)
    int (*sam_attn)(const void*, const void*, const void*, const void*,
                    const void*, void*, uint32_t, uint32_t, uint32_t, uint32_t,
                    float, void*);
    // slot 381: DeepSeek-greedy MoE router epilogue  - top-k
    // selection identical to moe_topk_batch, weights = FULL-softmax probs
    // (denominator over all n_expert, no renorm among the selected k).
    // -3 = shape not covered (n_expert > 256 or k > 16).
    // (logits [batch, n_expert], n_expert, k, out_idx, out_w, batch, stream)
    int (*moe_topk_softmax_all)(const void*, uint32_t, uint32_t, void*, void*,
                                uint32_t, void*);
    // slot 382: fused single-pass GQA-16 decode attention - FINAL
    // output, sink folded in-kernel, no partials/combine. Params =
    // attn_decode_batch_paged + pos_max:u32 after batch (host smem hint,
    // callers pass the kv_split_band ceiling). fp8/hd128/G16 only (rc -2),
    // rc -3 over the smem opt-in.
    int (*attn_decode_fused_gqa16)(const void*, const void*, const void*,
                                   const void*, void*, const void*, const void*,
                                   const void*, uint32_t, uint32_t, uint32_t,
                                   uint32_t, uint32_t, uint32_t, uint32_t,
                                   uint32_t, float, uint32_t, void*);
    // slot 383: in-house f16xf16->f32 tensor-core dense GEMM
    // (w, x, y, beta, in_dim, out_dim, batch, stream)
    int (*f16_gemm)(const void*, const void*, void*, float,
                    uint32_t, uint32_t, uint32_t, void*);
    // 384: ring twin of rope_norm_qk_append_paged - two position streams
    // (rope by true pos, append at the ring write slot) + a neox arg
    int (*rope_qk_append_paged_ring)(void*, void*, const void*, void*, void*,
                                     const void*, const void*, const void*,
                                     const void*, uint32_t, uint32_t, uint32_t,
                                     uint32_t, float, float, float, float,
                                     float, float, uint32_t, uint32_t,
                                     uint32_t, void*);
    // 385: residual-add + rmsnorm + Q8_0 quantize (dp4a-class e4m3 sibling)
    int (*add_rmsnorm_quant_q8_batch)(void*, const void*, const void*, void*,
                                      void*, void*, uint32_t, float, uint32_t,
                                      void*);
    // 386: swiglu + Q8_0 quantize, one warp per 32-block
    int (*swiglu_quant_q8)(const void*, const void*, void*, void*, uint32_t,
                           void*);
    // 387: OCR tower patch stem - u8 interleaved-RGB views to normalized f16
    // im2row rows in one gather
    // (pixels, out, mean0..2, std0..2, views, px, patch, stream)
    int (*ocr_patches_u8)(const void*, void*, float, float, float, float,
                          float, float, uint32_t, uint32_t, uint32_t, void*);
    // 388: split the encoder's fused q|k|v landing into three planes with
    // the q/v biases folded (- lets q,k,v run as one tc5p GEMM)
    // (qkv, bq|NULL, bv|NULL, q, k, v, d, rows, stream)
    int (*whisper_enc_qkv_split)(const void*, const void*, const void*, void*,
                                 void*, void*, uint32_t, uint32_t, void*);
    // 389: cross-K/V store off a layer-batched [rows, n_layer*d] landing -
    // dsts is a device array of n_layer plane pointers, bias the
    // concatenated [n_layer*d] plane or NULL
    // (src, bias|NULL, dsts, slots, rows, d, n_layer, stride, kv_dtype, stream)
    int (*whisper_kv_store_batch)(const void*, const void*, const void*,
                                  const void*, uint32_t, uint32_t, uint32_t,
                                  uint32_t, uint32_t, void*);
    // 390: decode-band multi-row bf16 GEMV, 2 <= batch <= 8. The
    // bf16 tile GEMM's grid is (out/64, batch/64) - at decode widths that is
    // out_dim/64 blocks TOTAL (4 for a 256-wide K/V plane), a latency-bound
    // serial K-loop on an idle machine. This is the GEMV-shaped twin: warp
    // per output row, 8 rows per block, acc[batch] against the [batch, in]
    // f32 activation rows. Params = bf16_gemm_f32; rc -2 outside the band.
    int (*bf16_gemv_mr_f32)(const void*, const void*, const void*, void*,
                            uint32_t, uint32_t, uint32_t, void*);
    // 391: bf16 tensor-core prefill GEMM, batch > 8. mma.sync
    // m16n8k16 with the f32 activations cast to bf16 in the smem stage - the
    // parity reference's own class (llama.cpp batched BF16 = cublasGemmEx
    // bf16xbf16, COMPUTE_32F). Params = bf16_gemm_f32; rc -2 outside the band
    // or on ragged in_dim (the f32-FMA tile stays the fallback there).
    int (*bf16_gemm_mma)(const void*, const void*, const void*, void*,
                         uint32_t, uint32_t, uint32_t, void*);
    // 392-396: PaddleOCR-VL tower elementwise fusions - each is
    // bit-identical to the 2-3 unfused ops it replaces (same IEEE order, one
    // final __float2half round); see vision.cuh's fusion header.
    // 392: LayerNorm writing f16 (params = layernorm, out is __half*)
    int (*layernorm_f16)(const void*, const void*, const void*, void*,
                         uint32_t, uint32_t, float, void*);
    // 393: bias + tanh-GELU + f16 store (x, bias, out_f16, rows, n, stream)
    int (*gelu_bias_f16)(const void*, const void*, void*, uint32_t, uint32_t, void*);
    // 394: erf twin of 393
    int (*gelu_erf_bias_f16)(const void*, const void*, void*, uint32_t, uint32_t, void*);
    // 395: residual + bias: x[r][i] += src[r][i] + bias[i]
    int (*add_bias_res)(void*, const void*, const void*, uint32_t, uint32_t, void*);
    // 396: mrope_vision with the q/k bias folded into the load
    // (x, bias, positions, n_tokens, n_heads, head_dim, theta_scale, stream)
    int (*mrope_vision_bias)(void*, const void*, const void*, uint32_t, uint32_t,
                             uint32_t, float, void*);
    // 397: modelopt NVFP4 checkpoint dequant - oracle/debug
    // primitive over the shipped triple (adjacent e2m1 nibbles, e4m3 per-16
    // scales, per-tensor f32 scale2), never a serving path
    // (data, scale, scale2, y_f32, in_dim, out_dim, stream)
    int (*nvf4_dequant)(const void*, const void*, float, void*, uint32_t,
                        uint32_t, void*);
    // 398: W4A16-class GEMV over a checkpoint NVFP4 plane -
    // f32 activations, prmt e2m1 decode, scale2 folded once post-reduction
    // (data, scale, bias|NULL, x, y, scale2, in_dim, out_dim, stream)
    int (*nvf4_gemv)(const void*, const void*, const void*, const void*,
                     void*, float, uint32_t, uint32_t, void*);
    // 399: GEMV over a checkpoint FP8 plane wrapped as a
    // per-row-f32-scale e4m3 image (nemotron mamba in/out_proj: the
    // per-TENSOR weight_scale broadcasts into the row array byte-exactly).
    // Warp-coherent 128-element steps, nvf4_gemv's geometry at 1 B/elem.
    // (data, rscale, x, y, in_dim, out_dim, stream)
    int (*f8r_gemv)(const void*, const void*, const void*, void*, uint32_t,
                    uint32_t, void*);
    // 400: mamba-2 decode conv step - pd_conv_step + a bias
    // term before the SiLU (nemotron's conv1d carries bias; GDN's does not)
    // (win, x_new, w, b, out, conv_dim, k, stream)
    int (*mamba_conv_step)(void*, const void*, const void*, const void*,
                           void*, uint32_t, uint32_t, void*);
    // 401: sequential mamba-2 SSD scan over a token span -
    // state [H, hd, S] f32 register-resident per thread-row, B/C group
    // broadcast is repeat_interleave (h / (H/G)), dt softplus + per-head
    // decay exp(dt*A) with A pre-transformed to -exp(A_log) at load,
    // D-skip fused into y. (state, xbc, dt, dt_stride, A, D, dt_bias, y,
    // n_tokens, n_heads, head_dim, d_state, n_groups, stream)
    int (*mamba2_scan_seq)(void*, const void*, const void*, uint32_t,
                           const void*, const void*, const void*, void*,
                           uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                           void*);
    // 402: grouped gated RMSNorm (Mixer2RMSNormGated): gate
    // first (x*silu(z), f32), variance per group of d/n_groups channels,
    // per-channel weight [d]. z rides a caller stride so it can live inside
    // the fused in_proj output rows. (x, z, z_stride, weight, out,
    // n_tokens, d, n_groups, eps, stream)
    int (*mamba_rmsnorm_gated_g)(const void*, const void*, uint32_t,
                                 const void*, void*, uint32_t, uint32_t,
                                 uint32_t, float, void*);
    // 403: token-batched NVFP4 MoE expert up GEMV + fused
    // squared-relu (nemotron experts have no gate matrix - relu(up)^2).
    // scale2 is a per-EXPERT f32 array; slots = batch*k, expert from idx.
    // Also serves the shared expert at k=1 with a constant zero idx.
    // (data, scale, scale2, idx, x, y, in_dim, ff, k, batch, stream)
    int (*nvf4_moe_up_relu2)(const void*, const void*, const void*,
                             const void*, const void*, void*, uint32_t,
                             uint32_t, uint32_t, uint32_t, void*);
    // 404: token-batched NVFP4 MoE expert down GEMV - combines
    // the k slots per token in fixed ascending order (deterministic), each
    // weighted topk_w[slot]*scale2[expert]; accumulate=1 adds onto y (the
    // shared-expert pass rides the same kernel with topk_w=1, k=1).
    // (data, scale, scale2, idx, topk_w, xr, y, ff, embd, k, batch,
    //  accumulate, stream)
    int (*nvf4_moe_down_acc)(const void*, const void*, const void*,
                             const void*, const void*, const void*, void*, void*,
                             uint32_t, uint32_t, uint32_t, uint32_t,
                             uint32_t, void*);
    // 405: slot 389's cross-K/V store off an audio-major batched
    // landing - row r -> slots[r / rows_per_slot]
    int (*whisper_kv_store_slots)(const void*, const void*, const void*,
                                  const void*, uint32_t, uint32_t, uint32_t,
                                  uint32_t, uint32_t, uint32_t, void*);
    // 406: bulk mamba-2 conv over a token span - the step
    // kernel's math with the window carried across T tokens in one launch,
    // reading the x|B|C span straight from the fused in_proj output rows
    // (x_off/x_stride). Bit-exact vs T serial steps. (Renumbered from 405
    // on rebase over the append - both sides appended.)
    // (win, xbc, x_off, x_stride, w, b, out, conv_dim, k, n_tokens, stream)
    int (*mamba_conv_seq)(void*, const void*, uint32_t, uint32_t,
                          const void*, const void*, void*, uint32_t,
                          uint32_t, uint32_t, void*);
    // 407: sorted-tile NVFP4 expert up + squared-relu ->
    // nvf4 requant over the moe_align BM=32 layout - the mxf4nvf4 MMA class
    // (W4A4 at prefill; decode stays on slot 403's GEMV). fq/fs are
    // sorted-position indexed, slot 408's direct B input.
    // (data, scale, scale2, sorted_row, block_expert, xq, xs, fq, fs,
    //  in_dim, ff, nb, stream)
    int (*nvf4_moe_up_relu2_bs)(const void*, const void*, const void*,
                                const void*, const void*, const void*,
                                const void*, void*, void*, uint32_t, uint32_t,
                                uint32_t, void*);
    // 408: sorted-tile NVFP4 expert down -> weighted
    // per-(token, slot) f32 partials at part[(tok*np + slt + slot_off)*embd]
    // - fold with moe_slot_combine (fixed slot order). topk_w NULL = 1.0
    // (the shared-expert pass); kw is topk_w's row stride.
    // (data, scale, scale2, sorted_row, sorted_slot, block_expert, topk_w,
    //  fq, fs, part, ff, embd, kw, np, slot_off, nb, stream)
    int (*nvf4_moe_down_bs)(const void*, const void*, const void*,
                            const void*, const void*, const void*,
                            const void*, const void*, const void*, void*,
                            uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                            uint32_t, void*);
    // 409 (decode rung): decode multi-task NVFP4 expert up +
    // squared-relu - one wave-dense launch over all k routed slots AND the
    // shared expert (act layout [k*ff_r | ff_s]), per-row math verbatim from
    // slot 403 (bit-identical rows, fused launch shape).
    // (rdata, rscale, rscale2, sdata, sscale, sscale2, idx, x, act,
    //  in_dim, ff_r, ff_s, k, stream)
    int (*nvf4_moe_up_relu2_mt)(const void*, const void*, const void*,
                                const void*, const void*, const void*,
                                const void*, const void*, void*, uint32_t,
                                uint32_t, uint32_t, uint32_t, void*);
    // 410 (decode rung): decode slot-split NVFP4 expert down ->
    // pre-weighted f32 partials at part[slot*embd + r] (shared expert =
    // slot k) - fold with moe_slot_combine (fixed ascending slot order).
    // Rung 4b: CTA-per-task split-K x4 inside, deterministic fixed-order
    // sums but regrouped vs the k=1 slot-404 fold (rel-to-rms gated).
    // (rdata, rscale, rscale2, sdata, sscale, sscale2, idx, topk_w, act,
    //  part, ff_r, ff_s, embd, k, stream)
    int (*nvf4_moe_down_part)(const void*, const void*, const void*,
                              const void*, const void*, const void*,
                              const void*, const void*, const void*, void*,
                              uint32_t, uint32_t, uint32_t, uint32_t, void*);
    // 411: capture-time f16 mmaf election gate - 0 declines the
    // mmaf fine-tile arm at pd_f16_gemm dispatch (election falls through to
    // GEMV/tc5g), 1 restores it. Read at LAUNCH/CAPTURE time, so it bakes
    // into any graph captured while set. Whisper's overlap routing captures
    // its mmaf-off decode variant behind this (mmaf x tc5p is the one
    // overlap-poison pairing). Returns 0. (Renumbered from 409 on rebase
    // over the decode-rung pair - both sides appended.)
    int (*f16_mmaf_set)(int);
    // 412-414 (stage A): batched decode steps over slot
    // arenas - the continuous-batching tick's per-slot state advance.
    // 412: conv step, windows arena [n_slots, k-1, conv_dim], row r of x
    // (stride/offset like conv_seq) advances slot slots[r]; bit-exact per
    // row vs mamba_conv_step.
    // (win, x, x_off, x_stride, slots, w, b, out, conv_dim, k, batch, stream)
    int (*mamba_conv_step_batch)(void*, const void*, uint32_t, uint32_t,
                                 const void*, const void*, const void*, void*,
                                 uint32_t, uint32_t, uint32_t, void*);
    // 413: single-token SSD scan step, states arena [n_slots, H, hd, S];
    // bit-exact per row vs mamba2_scan_seq at n_tokens=1.
    // (state, xbc, dt_raw, dt_stride, slots, A, D, dt_bias, y, batch,
    //  n_heads, head_dim, d_state, n_groups, stream)
    int (*mamba2_scan_step_batch)(void*, const void*, const void*, uint32_t,
                                  const void*, const void*, const void*,
                                  const void*, void*, uint32_t, uint32_t,
                                  uint32_t, uint32_t, uint32_t, void*);
    // 414: row-batched nvf4 GEMV twin (x [B, in], y [B, out], grid.y = B);
    // bit-exact per row vs nvf4_gemv.
    // (data, scale, bias, x, y, scale2, in_dim, out_dim, batch, stream)
    int (*nvf4_gemv_batch)(const void*, const void*, const void*, const void*,
                           void*, float, uint32_t, uint32_t, uint32_t, void*);
    // 415-416: Q8_0 single-plane expert up + squared-relu - the
    // nemotron_h_moe class (no gate matrix). Token-batched dp4a twin of
    // q8_0_moe_gate_up and sorted twin of q8_0_moe_gate_up_sorted (the
    // sorted one K-tail-guarded: in_dim % 32, not % 256).
    // (up_data, up_scale, idx, xq, xs, out, in_dim, ff, n_active, batch, stream)
    int (*q8_0_moe_up_relu2)(const void*, const void*, const void*, const void*,
                             const void*, void*, uint32_t, uint32_t, uint32_t,
                             uint32_t, void*);
    // (up_data, up_scale, sorted_row, block_expert, xq, xs, fused, in_dim,
    //  ff, max_blocks, stream)
    int (*q8_0_moe_up_relu2_sorted)(const void*, const void*, const void*,
                                    const void*, const void*, const void*, void*,
                                    uint32_t, uint32_t, uint32_t, void*);
    // 417 (spec core): scan_seq twin with a per-row state snapshot
    // (spec verify rollback). (state, xbc, dt_raw, dt_stride, A, D, dt_bias,
    // y, snap, n_tokens, n_heads, head_dim, d_state, n_groups, stream)
    int (*mamba2_scan_seq_snap)(void*, const void*, const void*, uint32_t,
                                const void*, const void*, const void*, void*,
                                void*, uint32_t, uint32_t, uint32_t, uint32_t,
                                uint32_t, void*);
    // 418: strided-rows copy (conv-input snapshots for the verify commit).
    // (src, src_off, src_stride, dst, len, rows, stream)
    int (*copy_rows_strided)(const void*, uint32_t, uint32_t, void*, uint32_t,
                             uint32_t, void*);
    // 419: multi-row W4A16 nvf4 GEMM - gemv_batch's signature,
    // weight fragments decoded once per 16-row group instead of per row.
    int (*nvf4_gemm_mr)(const void*, const void*, const void*, const void*,
                        void*, float, uint32_t, uint32_t, uint32_t, void*);
    // slot 420: moe_topk_sigmoid_batch writing k+ns-wide rows,
    // lanes k.. append the shared pseudo-expert ids sh0.. with weight 1.0
    // (the shared-expert fold-in). Arch-generic SIMT.
    int (*moe_topk_sigmoid_batch_sh)(const void*, const void*, float, uint32_t,
                                     uint32_t, uint32_t, uint32_t, void*,
                                     void*, uint32_t, void*);
    // 421: gemma_qkv_nra3 twin reading PACKED bf16 q/k/v GEMM
    // planes (the b16-D election's p16 convention - same element indexing,
    // half the bytes; q_out and the KV appends are unchanged). Same argument
    // list as gemma_qkv_nra3; plane byte offsets are the caller's to halve.
    int (*gemma_qkv_nra3_b16)(void*, void*, void*, const void*, const void*,
                              void*, void*, void*, const void*, const void*,
                              const void*, const void*, uint32_t, uint32_t,
                              uint32_t, uint32_t, uint32_t, uint32_t, float,
                              float, uint32_t, uint32_t, float, uint32_t,
                              uint32_t, void*);
    // slot 422: tensor-core NVFP4 GEMM - gemv_batch's
    // signature, exact-dequant bf16 weights on m16n8k16 mma. The batched
    // lm_head class; not bit-comparable to 419/414 (bf16 activation cast +
    // mma reassociation).
    int (*nvf4_gemm_tc)(const void*, const void*, const void*, const void*,
                        void*, float, uint32_t, uint32_t, uint32_t, void*);
    // 423: attn_spec_batch_fin twin quantizing the
    // finalized rows in-kernel - arg 4 is the e4m3 plane, arg 5 the f32
    // per-row scales (fin's dead ml slot). -2 = geometry not covered
    // (caller keeps the f32 fin + standalone row quantize).
    int (*attn_spec_batch_fin_e4)(const void*, const void*, const void*,
                                  void*, void*, const void*, const void*,
                                  const void*, uint32_t, uint32_t, uint32_t,
                                  uint32_t, uint32_t, uint32_t, uint32_t,
                                  uint32_t, float, uint32_t, void*);
    // 424 (thin-k/v rung): fused q|k|v decode-band bf16 GEMM -
    // one launch over the load-time-concatenated [q;k;v] plane, segmented
    // store into the three y planes. Per out-row bit-identical to
    // bf16_gemm_mma on the matching segment. (w, x, yq, yk, yv, in_dim,
    // oq, okv, batch, stream); -2 = batch<2 or ragged in_dim.
    int (*bf16_qkv_gemm_mma)(const void*, const void*, void*, void*, void*,
                             uint32_t, uint32_t, uint32_t, uint32_t, void*);
    // 425: attn_spec_batch_fin twin storing the finalized
    // rows as e4m3 at STATIC scale 1.0 - arg 4 is the i8 quantized plane
    // (pf_e4q); the caller feeds the GEMM a ones xrs vector. Same accept
    // envelope as fin (same impl, extra sentinel bit); -2 = caller keeps
    // the f32 fin + standalone row quantize.
    int (*attn_spec_batch_fin_e4s)(const void*, const void*, const void*,
                                   void*, void*, const void*, const void*,
                                   const void*, uint32_t, uint32_t, uint32_t,
                                   uint32_t, uint32_t, uint32_t, uint32_t,
                                   uint32_t, float, uint32_t, void*);
    // 426: checkpoint-plane W4A4 GEMM - mxfp4_gemm_nv4's fp4 x
    // fp4 block-scale mma with the Nvf4Plane epilogue (acc*scale2, +bias
    // when present). (data, scale, bias, xq, xs, y, scale2, in_dim,
    // out_dim, batch, stream); xq/xs from quantize_nvf4[_swiglu]. cc-gated
    // with 414 (block-scale SASS).
    int (*nvf4_gemm_f4)(const void*, const void*, const void*, const void*,
                        const void*, void*, float, uint32_t, uint32_t,
                        uint32_t, void*);
    // 427: v2 of 426 - async scale planes, one barrier
    // per K-step, ring depth `st` (2 or 3; probe elected 3). Same signature
    // plus st before stream. Requires in_dim % 128 == 0.
    int (*nvf4_gemm_f4b)(const void*, const void*, const void*, const void*,
                         const void*, void*, float, uint32_t, uint32_t,
                         uint32_t, uint32_t, void*);
    // 428: split-K twin for machine-starved tile grids
    // (decode down: 40 CTAs). (data, scale, bias, xq, xs, part, y, scale2,
    // in_dim, out_dim, batch, sk, stream); part >= sk*batch*out_dim f32,
    // deterministic two-pass reduce owns the epilogue.
    int (*nvf4_gemm_f4s)(const void*, const void*, const void*, const void*,
                         const void*, void*, void*, float, uint32_t, uint32_t,
                         uint32_t, uint32_t, void*);
    // 429: KC=256 arm - half the barriers, 2x per-stage
    // flight, one 16 B scale cp.async per row. Same signature as 428 (sk=1
    // runs unsplit and ignores part). Requires in_dim % 256 == 0; bit-exact
    // vs 426/427 (identical global K order).
    int (*nvf4_gemm_f4c)(const void*, const void*, const void*, const void*,
                         const void*, void*, void*, float, uint32_t, uint32_t,
                         uint32_t, uint32_t, void*);
    // 430: TMA + mbarrier ring (kt3 shape: 8 consumer +
    // 4 producer warps, 4 tensor maps over the PLAIN layouts). The prefill
    // band: gate b2048 457->345 us, down 496->333. (data, scale, bias, xq,
    // xs, y, scale2, in_dim, out_dim, batch, stream); in_dim % 256 == 0;
    // bit-exact vs 426/427/429 (identical global K order).
    int (*nvf4_gemm_f4t)(const void*, const void*, const void*, const void*,
                         const void*, void*, float, uint32_t, uint32_t,
                         uint32_t, void*);
    // 431: tcgen05/TMEM decode attention - FINAL output
    // (batch-major rows in `out`, no partials/out_ml/combine; the caller
    // skips the combine when rc == 0). Params = attn_decode_batch_paged
    // (`sinks` accepted and ignored - gemma's -inf sinks are a no-op).
    // fp8-e4m3 paged KV, head_dim 256, group 2, swa_window > 0 only:
    // rc -2 = shape/arch not covered, rc -3 = smem over the opt-in.
    int (*attn_decode_tc5_paged)(const void*, const void*, const void*,
                                 const void*, void*, const void*, const void*,
                                 const void*, uint32_t, uint32_t, uint32_t,
                                 uint32_t, uint32_t, uint32_t, uint32_t, float,
                                 uint32_t, void*);
    // 434: device top-K prefilter for HOST-HEAD sampling rows -
    // (logits, params [rows x PdSampleRow, mode 4 = selected], out
    // [rows x k x 2 u32 = (id, raw-logit bits)], rows, n, k, stream).
    // The host runs its nucleus pipeline over the K-head instead of a
    // full-vocab readback (qwen3.x top-k/top-p defaults were 21.3 ms of
    // host sampling per c32 round). k <= 64; rc -2 = k out of range.
    int (*topk_rows)(const void*, const void*, void*, uint32_t, uint32_t,
                     uint32_t, void*);
    // 435: full-device truncation sampling - (logits, params
    // [PdSampleRow, mode 5 rows], trunc_params [rows x {k, top_p f32,
    // min_p f32, pad}], out [rows token ids], rows, n, stream). The host
    // sample_trunc_head pipeline runs on device; rows become zero-host.
    int (*sample_rows_t)(const void*, const void*, const void*, void*,
                         uint32_t, uint32_t, void*);
    // 436 (truncation stage c): GENERAL truncation sampling, mode 6 - same
    // signature/planes as 435 but no top-k bound: top-p only (nemotron's
    // published profile), min-p only, and combinations sample exactly via
    // the histogram quantile walk (build_nucleus top_k==0 semantics).
    int (*sample_rows_p)(const void*, const void*, const void*, void*,
                         uint32_t, uint32_t, void*);
    // 437: DECODE gated-delta recurrence with the split+qk-L2-norm
    // fused in - (conv [B, conv_dim], g [B,HV], beta [B,HV], slots,
    // states, out [B,HV,D], batch, n_k_heads, n_v_heads, head_dim(=128),
    // stream). Byte-identical to split_gqa_norm + recurrent_v2 at
    // n_tokens=1/no-snap; kills one kernel + the dq/dk/dv plane round
    // trip per GDN layer per round.
    int (*gated_delta_recurrent_v2f)(const void*, const void*, const void*,
                                     const void*, void*, void*, uint32_t,
                                     uint32_t, uint32_t, uint32_t, void*);
    // 438: conv_step_slots with x_new strided out of the DN
    // in-proj fused plane (wins, x_new, w, out, slots, batch, conv_dim, k,
    // x_stride, stream); bit-identical to slice-then-conv.
    int (*conv_step_slots_s)(void*, const void*, const void*, void*,
                             const void*, uint32_t, uint32_t, uint32_t,
                             uint32_t, void*);
    // 439: gated_rmsnorm with z strided out of the fused plane
    // (x, z, weight, out, n_rows, d, eps, z_stride, z_rows_per_b, stream).
    int (*gated_rmsnorm_s)(const void*, const void*, const void*, void*,
                           uint32_t, uint32_t, float, uint32_t, uint32_t,
                           void*);
    // 440: v2f with g/beta computed in-kernel from the fused
    // plane's alpha/beta columns (conv, fused, ab_off, fused_stride, ssm_a,
    // dt_bias, slots, states, out, batch, n_k_heads, n_v_heads, head_dim,
    // stream).
    int (*gated_delta_recurrent_v2f_g)(const void*, const void*, uint32_t,
                                       uint32_t, const void*, const void*,
                                       const void*, void*, void*, uint32_t,
                                       uint32_t, uint32_t, uint32_t, void*);
    // 441: VL conv+silu+qkv over the wave pass's packed fresh spans
    // (x, w, row0s, q, k, v, n_rows, n_k_heads, n_v_heads, s, k, stream);
    // row0s = per-row span-start plane; bit-identical to the per-span
    // offset launches it replaces.
    int (*causal_conv1d_silu_qkv_vl)(const void*, const void*, const void*,
                                     void*, void*, void*, uint32_t, uint32_t,
                                     uint32_t, uint32_t, uint32_t, void*);
    // 442 (glue rung): residual-add + rmsnorm + nvf4 quantize in
    // one row-per-CTA launch, replacing add_inplace + rmsnorm_batch +
    // quantize_nvf4 in nemotron's 23 MoE prologues per decode tick. Still
    // emits the f32 normed row (the router reads it). BIT-EXACT to the
    // three-kernel chain at the same nth.
    // (x, proj, w, out, q, scale, n, eps, batch, stream)
    int (*add_rmsnorm_quant_nvf4_batch)(void*, const void*, const void*, void*,
                                        void*, void*, uint32_t, float,
                                        uint32_t, void*);
    // 443-445 (scan rung): the f16 SSM-state class - state STORED
    // f16, computed f32, the numeric class of vLLM's
    // --mamba-ssm-cache-dtype float16. Same signatures as their f32 twins
    // (slots 4xx); `state` and `snap` are __half planes, so the arena and
    // the snap blob are half the bytes. Pure appends.
    // (state, xbc, dt_raw, dt_stride, A, D, dt_bias, y, n_tokens, n_heads,
    //  head_dim, d_state, n_groups, stream)
    int (*mamba2_scan_seq_f16)(void*, const void*, const void*, uint32_t,
                               const void*, const void*, const void*, void*,
                               uint32_t, uint32_t, uint32_t, uint32_t,
                               uint32_t, void*);
    // (..., y, snap, n_tokens, n_heads, head_dim, d_state, n_groups, stream)
    int (*mamba2_scan_seq_snap_f16)(void*, const void*, const void*, uint32_t,
                                    const void*, const void*, const void*,
                                    void*, void*, uint32_t, uint32_t, uint32_t,
                                    uint32_t, uint32_t, void*);
    // (state, xbc, dt_raw, dt_stride, slots, A, D, dt_bias, y, batch,
    //  n_heads, head_dim, d_state, n_groups, stream)
    int (*mamba2_scan_step_batch_f16)(void*, const void*, const void*, uint32_t,
                                      const void*, const void*, const void*,
                                      const void*, void*, uint32_t, uint32_t,
                                      uint32_t, uint32_t, uint32_t, void*);
    // 446-447 (scan rung): f16 state <-> the f32 checkpoint blob.
    // The prefix-cache pool/stage layout stays f32 so the restore path is
    // untouched; these convert at the arena boundary. Lossless both ways -
    // widening is exact and the narrowing only sees values that came from
    // f16. (src, dst, n_elems, stream)
    int (*ssm_state_widen)(const void*, void*, uint32_t, void*);
    int (*ssm_state_narrow)(const void*, void*, uint32_t, void*);
    // 448-449: QKC compact-bf16 q/k pair (conv emitter + vl chunked-GDN
    // reader; one caller-side latch drives both).
    int (*causal_conv1d_silu_qkv_vl_qkc)(const void*, const void*, const void*,
                                         void*, void*, void*, uint32_t,
                                         uint32_t, uint32_t, uint32_t,
                                         uint32_t, void*);
    int (*gated_delta_chunked_rs_vl_qkc)(const void*, const void*, const void*,
                                         const void*, const void*, void*,
                                         void*, void*, void*, void*, void*,
                                         const void*, uint32_t, const void*,
                                         uint32_t, uint32_t, uint32_t,
                                         uint32_t, uint32_t, void*);
    // 450: single-plane relu^2 decode-band expert up - the dec2
    //  class for nemotron_h_moe. Down reuses q8_0_moe_dn_dec2 unchanged.
    //  (up_data, up_scale, idx, xq, xs, out, in_dim, ff, n_active, batch,
    //   rows_pb (0 = elected), stream)
    int (*q8_0_moe_up_relu2_dec2)(const void*, const void*, const void*,
                                  const void*, const void*, void*, uint32_t,
                                  uint32_t, uint32_t, uint32_t, uint32_t,
                                  void*);
    // 451: quantize_q8 with relu(x)^2 folded in front of the
    //  per-32 amax - lets a squared-relu dense FFN (nemotron's shared expert)
    //  run its up plane on the plain q8 GEMM ladder. Bit-identical to
    //  relu^2-to-f32 followed by quantize_q8. (x, q, scale, n, stream)
    int (*quantize_q8_relu2)(const void*, void*, void*, uint32_t, void*);
    // 452-454 (lm_head repack rung): tile-major NVFP4 plane
    //  twins - the loader repacks a plane to [row_tile 128][k_stage 128]
    //  [row] (out padded to 128 and zero-filled, in_dim % 128 == 0) and
    //  these read that layout. Each is bit-exact vs its row-major twin
    //  (414 / 419 / the tcp arm of 422); same gemv_batch signature.
    int (*nvf4_gemv_batch_tm)(const void*, const void*, const void*,
                              const void*, void*, float, uint32_t, uint32_t,
                              uint32_t, void*);
    int (*nvf4_gemm_mr_tm)(const void*, const void*, const void*,
                           const void*, void*, float, uint32_t, uint32_t,
                           uint32_t, void*);
    int (*nvf4_gemm_tc_tm)(const void*, const void*, const void*,
                           const void*, void*, float, uint32_t, uint32_t,
                           uint32_t, void*);
    // 455-457 (fragment rung): FRAGMENT-layout plane twins - the
    //  tile-major blocks additionally permuted to [w][k16][g][u32 of a0..a3
    //  fragment bytes per lane] (scales stay tile-major). Bit-exact per
    //  class vs the _tm twins; same gemv_batch signature.
    int (*nvf4_gemv_batch_tf)(const void*, const void*, const void*,
                              const void*, void*, float, uint32_t, uint32_t,
                              uint32_t, void*);
    int (*nvf4_gemm_mr_tf)(const void*, const void*, const void*,
                           const void*, void*, float, uint32_t, uint32_t,
                           uint32_t, void*);
    int (*nvf4_gemm_tc_tf)(const void*, const void*, const void*,
                           const void*, void*, float, uint32_t, uint32_t,
                           uint32_t, void*);
    // slot 458: Q16xKv128 tensor-core decode attention (attn/fmha16.cuh).
    // Same params as attn_decode_fused_gqa16 minus pos_max - this arm chunks
    // the KV walk so its shared memory is constant in context.
    int (*attn_decode_fmha16)(const void*, const void*, const void*,
                              const void*, void*, const void*, const void*,
                              const void*, uint32_t, uint32_t, uint32_t,
                              uint32_t, uint32_t, uint32_t, uint32_t, float,
                              uint32_t, void*);
    // slot 459: DFlash2 grouped dynamic convolution (dflash.cuh). Wraps one
    // drafter sublayer; `side` picks the before/after half of the base kernel
    // and of the shared projection row. `out` must not alias `h`.
    int (*dflash_conv)(const void*, void*, const void*, const void*, uint32_t,
                       uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                       uint32_t, void*);
    // slot 460: unpack pd_topk_rows' (id, logit-bits) pairs into a flat id
    // plane for pd_kquant_gather, anchors appended at the tail.
    int (*dflash_cand_ids)(const void*, const void*, void*, uint32_t, uint32_t,
                           uint32_t, uint32_t, void*);
    // slot 461: DFlash2 candidate-selector walk - greedy forward pass over the
    // bilinear edge scores, one CTA per block. Applies the drafter's own logit
    // epilogue to the unary term (it is ADDED to a bilinear score, so the
    // monotone argument that lets greedy drafting skip it does not hold).
    int (*dflash_select)(const void*, const void*, const void*, const void*,
                         void*, float, float, uint32_t, uint32_t, uint32_t,
                         uint32_t, void*);
    // slot 462: spec-verify twin of
    // gated_delta_recurrent_v2 that neither snapshots nor writes the state
    // back - the live state stays at ROUND-START; out[] values are
    // bit-identical to v2's. Args: q,k,v,g,beta,slots,states(const),out,
    // batch,n_tokens,n_heads,head_dim,stream.
    int (*gated_delta_verify_hold)(const void*, const void*, const void*,
                                   const void*, const void*, const void*,
                                   const void*, void*, uint32_t, uint32_t,
                                   uint32_t, uint32_t, void*);
    // slot 463: commit-time recompute - re-runs the recurrence from the
    // round-start state over each row's accepted prefix (committed[b],
    // device-staged, capture-safe) on the stashed split/gate planes, one
    // final state writeback. Same fixed op order as v2, so the result is
    // bit-exact vs the per-token snapshot the old restore path picked.
    // Replaces state_restore_slots (+ the b x k1 snapshot allocation) on
    // the qwen35 spec path. Args: k,v,g,beta,slots,committed,states,
    // batch,n_tokens,n_heads,head_dim,stream.
    int (*gated_delta_commit_walk)(const void*, const void*, const void*,
                                   const void*, const void*, const void*,
                                   void*, uint32_t, uint32_t, uint32_t,
                                   uint32_t, void*);
    // slot 464: dflash async round - copy the block-draft picks from the
    // draft graph's row-major d_out into the MTP chain's i-major d_draft
    // layout, device-side (no host round-trip in the round).
    // Args: out, draft, n, rows, k_use, stream.
    int (*dflash_chain_picks)(const void*, void*, uint32_t, uint32_t,
                              uint32_t, void*);
    // slot 469: dflash conditioning fold (rung C) - the append
    // that norms for the drafter ring: per written row, k-norm (rmsnorm
    // math verbatim, nth elected from norm_batch like the rmsnorm launcher)
    // + NEOX yarn rope + paged f16 K/V store; replaces the per-layer
    // norm + rope + 2 x cuts kv_append train (~340 eager launches/round at
    // 32 live) with one launch per layer, pool bytes bit-identical.
    // Args: fk,fv,kw,pool_k,pool_v,rows_w,positions,slots,block_tables,
    // blocks_per_slot,n_kv,head_dim,eps,theta_scale,freq_scale,corr_low,
    // corr_high,ext_factor,mscale,nw,norm_batch,stream.
    int (*dflash_cond_append)(const void*, const void*, const void*, void*,
                              void*, const void*, const void*, const void*,
                              const void*, uint32_t, uint32_t, uint32_t,
                              float, float, float, float, float, float,
                              float, uint32_t, uint32_t, void*);
    // slot 470: DFlash2 SAMPLED selector walk (rung G) - the
    // greedy walk's twin with per-block 1/T (0 = argmax) + u32 seed; writes
    // the chosen token AND the row's K-way draft distribution q16[row*k+c].
    // Args: topk,pred,succ,hs,invt,seeds,out,q16,scale,cap,rank,k,rows,r,stream.
    int (*dflash_select_rs)(const void*, const void*, const void*, const void*,
                            const void*, const void*, void*, void*, float, float,
                            uint32_t, uint32_t, uint32_t, uint32_t, void*);
    // slot 471: K-candidate canonical rejection-sampling verify resolve,
    // truncation-aware (mode-7 PdSampleRow rows; head + nucleus are the
    // mode-5 kernel's). Args: logits,params,trunc,meta,toks,cand,q16,out,
    // rows,n_blocks,k1,drows,k,n,stream.
    int (*dflash_rs_resolve)(const void*, const void*, const void*, const void*,
                             const void*, const void*, const void*, void*,
                             uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                             uint32_t, void*);
    // slots 472-477: NVFP4 MoE consumers over the TILED expert-plane layout
    // (moe/nvf4_st.cuh,  - the skinny-tile pair). 472/473 =
    // the BM=8 decode pair (pd_moe_align_bm(8) blocks), 474/475 = the BM=32
    // prefill twins, 476/477 = the r=1 mt-class GEMV twins. All six cc12-only;
    // the engine's tiled-layout election requires the full set (a tiled plane
    // must never exist without every consumer class able to read it).
    int (*nvf4_moe_up_relu2_st)(const void*, const void*, const void*,
                                const void*, const void*, const void*,
                                const void*, void*, void*, uint32_t, uint32_t,
                                uint32_t, void*);
    int (*nvf4_moe_down_st)(const void*, const void*, const void*, const void*,
                            const void*, const void*, const void*, const void*,
                            const void*, void*, uint32_t, uint32_t, uint32_t,
                            uint32_t, uint32_t, uint32_t, void*);
    int (*nvf4_moe_up_relu2_stw)(const void*, const void*, const void*,
                                 const void*, const void*, const void*,
                                 const void*, void*, void*, uint32_t, uint32_t,
                                 uint32_t, void*);
    int (*nvf4_moe_down_stw)(const void*, const void*, const void*, const void*,
                             const void*, const void*, const void*, const void*,
                             const void*, void*, uint32_t, uint32_t, uint32_t,
                             uint32_t, uint32_t, uint32_t, void*);
    int (*nvf4_moe_up_relu2_mtt)(const void*, const void*, const void*,
                                 const void*, const void*, const void*,
                                 const void*, const void*, void*, uint32_t,
                                 uint32_t, uint32_t, uint32_t, void*);
    int (*nvf4_moe_down_part_tt)(const void*, const void*, const void*,
                                 const void*, const void*, const void*,
                                 const void*, const void*, const void*, void*,
                                 uint32_t, uint32_t, uint32_t, uint32_t, void*);
    /// Capability marker: non-null iff the kquant family serves PD_KQ_Q40
    /// (the QAT lineage's native format). Pure append.
    int (*kquant_q40)(void);
    // 479/480: KV tier extent gather/scatter (kv-offload,
    // tier/xfer.cuh). Pure appends at the tail; slot presence is the
    // engine's tier-transfer capability probe.
    int (*kv_gather_blocks)(const void*, const void*, void*,
                            unsigned long long, unsigned long long, uint32_t,
                            uint32_t, void*);
    int (*kv_scatter_blocks)(const void*, const void*, const void*,
                             unsigned long long, unsigned long long, uint32_t,
                             uint32_t, void*);
    // 481: b=1 GEMV over the tile-linear boxes (non-KV-overhead R2.2) - the
    // kernel that lets a plane class serve every width from one resident
    // format, so its Q8_0 twin can be reclaimed.
    // (wlin, x, part, y, ticket, in_dim, out_dim, stream)
    int (*f8lin_gemv)(const void*, const void*, void*, void*, void*, uint32_t,
                      uint32_t, void*);
    // 482: granite's f32/Q8 residual fusion -
    // scale_add + rmsnorm_batch + quantize_q8_sums in one launch, carrying
    // the residual_multiplier the existing fused norms have no room for.
    // Pure append at the tail; slot presence is the capability probe.
    // (x, proj, w, xn, q, scale, sums, n, batch, eps, res_scale, stream)
    int (*add_rmsnorm_q8_xn)(void*, const void*, const void*, void*, void*,
                             void*, void*, uint32_t, uint32_t, float, float,
                             void*);
    // 483/484: v2 ring twins of the sorted q8 MMA pair (S-stage cp.async
    // ring + live-quarter skip). Same
    // signatures as 480/481-era pair; bitwise on live outputs; bm must be
    // 32 (NotSupported otherwise), down additionally needs ff % 64.
    int (*q8_0_moe_gate_up_mma2_geglu)(const void*, const void*, const void*,
                                       const void*, const void*, const void*,
                                       const void*, const void*, void*, void*,
                                       uint32_t, uint32_t, uint32_t, uint32_t,
                                       void*);
    int (*q8_0_moe_down_mma2)(const void*, const void*, const void*,
                              const void*, const void*, const void*,
                              const void*, const void*, void*, uint32_t,
                              uint32_t, uint32_t, uint32_t, uint32_t, void*);
    // 485: write-out slot combine - bitwise the memset+combine chain
    int (*moe_slot_combine_init)(const void*, void*, uint32_t, uint32_t,
                                 uint32_t, void*);
    // 486: K-split decode router matvec (w, x, scratch, out, in, out, b, s)
    int (*matvec_f32_ks)(const void*, const void*, void*, void*, uint32_t,
                         uint32_t, uint32_t, void*);
    // 487: head+router+topk fusion (x, gamma, pre2, rw, dscale, pn, q, qs,
    // idx, w, n, n_expert, k, eps, batch, stream) - bit-identical chain
    int (*moe_head_router)(const void*, const void*, const void*, const void*,
                           const void*, void*, void*, void*, void*, void*,
                           uint32_t, uint32_t, uint32_t, float, uint32_t,
                           void*);
    // 488: v5 gate_up (small-CTA BM16/BN64 port; same signature class)
    int (*q8_0_moe_gate_up_mma3_geglu)(const void*, const void*, const void*,
                                       const void*, const void*, const void*,
                                       const void*, const void*, void*, void*,
                                       uint32_t, uint32_t, uint32_t, uint32_t,
                                       void*);
    // 489: pair map (srow32, sslot32, map, n_active, srp32, stream)
    int (*moe_pair_map)(const void*, const void*, void*, uint32_t, uint32_t,
                        void*);
    // 490: q8 GEGLU remap quantize (gu, srow128, sslot128, map, fq, fs,
    // n_ff, n_active, srp128, act, stream)
    int (*quantize_q8_geglu_remap)(const void*, const void*, const void*,
                                   const void*, void*, void*, uint32_t,
                                   uint32_t, uint32_t, uint32_t, void*);
    // 491: tail+combine fold (x, proj, part, pn1, pn2, postw, n, n_active,
    // eps, os, batch, stream)
    int (*moe_tail_combine)(void*, const void*, const void*, const void*,
                            const void*, const void*, uint32_t, uint32_t,
                            float, float, uint32_t, void*);
    // slot 492: merged q|k|v NVFP4 GEMV. One grid over up to three checkpoint
    // planes sharing one x -- the small-out_dim occupancy fix (granite k/v at
    // out 1024 are 128 CTAs on 188 SMs). Pure append at the tail; slot
    // presence is the capability probe.
    // (segs, x, in_dim, n_segs, stream)
    int (*nvf4_gemv_multi)(const void*, const void*, uint32_t, uint32_t, void*);
    int (*add_rmsnorm_scaled_batch)(void*, const void*, const void*, void*,
                                    uint32_t, float, uint32_t, void*, float);
    // hibatch lane M1: hb head+router+topk (8-token blocks, bf16 smem rows,
    // rw plane read once per 8 tokens). Signature == moe_head_router.
    int (*moe_head_router_hb)(const void*, const void*, const void*, const void*,
                              const void*, void*, void*, void*, void*, void*,
                              uint32_t, uint32_t, uint32_t, float, uint32_t,
                              void*);
    // P1-2: per-128 activation-scale pair (head producer + mma2g consumer)
    int (*moe_head_xg)(const void*, const void*, const void*, void*, void*,
                       void*, void*, uint32_t, float, uint32_t, void*);
    int (*q8_0_moe_gate_up_mma2g_geglu)(const void*, const void*, const void*,
                                        const void*, const void*, const void*,
                                        const void*, const void*, void*, void*,
                                        uint32_t, uint32_t, uint32_t, uint32_t,
                                        void*);
    // P1-1: bf16 partials pair
    int (*q8_0_moe_down_mma2_pbf16)(const void*, const void*, const void*,
                                    const void*, const void*, const void*,
                                    const void*, const void*, void*, uint32_t,
                                    uint32_t, uint32_t, uint32_t, uint32_t,
                                    void*);
    int (*moe_tail_combine_bf16)(void*, const void*, const void*, const void*,
                                 const void*, const void*, uint32_t, uint32_t,
                                 float, float, uint32_t, void*);
    // B3-1: cooperative router stage (matvec+topk in one kernel)
    int (*moe_router_stage)(const void*, const void*, void*, const void*,
                            void*, void*, uint32_t, uint32_t, uint32_t,
                            uint32_t, void*);
    // P1 dn64: per-64 Y-scale pair - gu GEGLU quantize at ff/64 scale
    // stride + down pair-grouped fold; the down fn takes a trailing pbf16
    // flag (P1-1 composition).
    int (*q8_0_moe_gate_up_mma2g_y64_geglu)(const void*, const void*, const void*,
                                            const void*, const void*, const void*,
                                            const void*, const void*, void*, void*,
                                            uint32_t, uint32_t, uint32_t, uint32_t,
                                            void*);
    int (*q8_0_moe_down_mma2_fs64)(const void*, const void*, const void*,
                                   const void*, const void*, const void*,
                                   const void*, const void*, void*, uint32_t,
                                   uint32_t, uint32_t, uint32_t, uint32_t,
                                   uint32_t, void*);
    // v3t: TMA-staged v2 ring twins - bitwise, sm_90+
    // (resolver NULLs below cc 9). gate_up: v2 sig + n_expert before
    // max_blocks; down: v2 sig + n_expert before n_active.
    int (*q8_0_moe_gate_up_mma2t_geglu)(const void*, const void*, const void*,
                                        const void*, const void*, const void*,
                                        const void*, const void*, void*, void*,
                                        uint32_t, uint32_t, uint32_t, uint32_t,
                                        uint32_t, void*);
    int (*q8_0_moe_down_mma2t)(const void*, const void*, const void*,
                               const void*, const void*, const void*,
                               const void*, const void*, void*, uint32_t,
                               uint32_t, uint32_t, uint32_t, uint32_t,
                               uint32_t, void*);
    // g2 (slot 504): token-major gate_up at bm=16, fq/fs written at bm32
    // rows via the pair map - bitwise to v2, decode widths only.
    int (*q8_0_moe_gate_up_g2_geglu)(const void*, const void*, const void*,
                                     const void*, const void*, const void*,
                                     const void*, const void*, const void*,
                                     const void*, void*, void*, uint32_t,
                                     uint32_t, uint32_t, uint32_t, uint32_t,
                                     uint32_t, void*);
    // dual-output align (slot 505): bm32 CSR + bm16 CSR + pair map in one
    // launch (g2 lane).
    int (*moe_align_dual)(const void*, void*, void*, void*, void*, void*,
                          void*, void*, uint32_t, uint32_t, uint32_t,
                          uint32_t, uint32_t, void*);
    // qwen4_exp family (slots 506-514) - Qwen3.8-Flash-Next new math. See
    // src/qwen4exp.cuh; ground truth is paddock-kernels reference::qwen4exp.
    // 506: grouped RMSNorm with the Gemma (1+w) FMA affine (hyper-connection
    // state: 4 streams normalized independently, weight spans the full width).
    int (*q4x_group_norm_1p)(const void*, const void*, void*, void*, uint32_t,
                             uint32_t, uint32_t, float, void*);
    // 507: hyper-connection mix reduce - Sum_s sigmoid(gate)*xn / hc.
    int (*q4x_hc_mix)(const void*, const void*, void*, void*, uint32_t, uint32_t,
                      uint32_t, void*);
    // 508: hyper-connection combine - H[s] += block_out * 2*sigmoid(inj/hc).
    int (*q4x_hc_combine)(void*, const void*, const void*, uint32_t, uint32_t,
                          uint32_t, void*);
    // 509: in-place silu after a scalar scale (the low-rank mix's /hc).
    int (*q4x_scale_silu)(void*, uint32_t, float, void*);
    // 510: PLE per-stream gate - sigmoid(signed_sqrt(K.Q/sqrt(hidden))) * V.
    int (*q4x_ple_gate)(const void*, const void*, const void*, void*, uint32_t,
                        uint32_t, uint32_t, void*);
    // 511: causal depthwise conv1d+silu with DILATION (PLE k=4 dilation 3).
    int (*q4x_conv_dil)(const void*, const void*, void*, uint32_t, uint32_t,
                        uint32_t, uint32_t, void*);
    // 512: one-token twin of 511 off a carried (k-1)*dil window.
    int (*q4x_conv_dil_step)(const void*, const void*, const void*, void*,
                             uint32_t, uint32_t, uint32_t, void*);
    // 513: GDN output gated norm, plain w and a SIGMOID gate (the pack's
    // pd_gated_rmsnorm is the qwen3.5 shape and gates with SILU).
    int (*q4x_gdn_gated_norm)(const void*, const void*, const void*, void*,
                              void*, uint32_t, uint32_t, float, void*);
    // 514: GDN conv-output split with REPEAT_INTERLEAVE key-head widening
    // (raw safetensors order; the pack's own split kernels use the GGUF
    // lane's %-mapping, which cannot express this map).
    int (*q4x_gdn_split_widen)(const void*, void*, void*, void*, uint32_t,
                               uint32_t, uint32_t, uint32_t, uint32_t, void*);
    // 515: shared-expert fold - y[r,:] += x[r,:] * sigmoid(s[r]) (a SCALAR
    // per-row gate; mul_sigmoid is the elementwise-gate op, not this).
    int (*q4x_add_gated_row)(void*, const void*, const void*, uint32_t,
                             uint32_t, void*);
    // 516: NVFP4 MoE gate+up GEMV with a fused swiglu - this family's experts
    // carry both planes; every other nvf4 expert consumer here is nemotron's
    // gate-matrix-free relu2.
    int (*q4x_moe_gu_swiglu)(const void*, const void*, const void*, const void*,
                             const void*, const void*, const void*, const void*,
                             void*, uint32_t, uint32_t, uint32_t, uint32_t,
                             void*);
    // 517: hyper-connection combine FUSED with the grouped (1+w) norm that
    // always follows it - one launch, one pass over the 4-stream state.
    int (*q4x_combine_norm)(void*, const void*, const void*, const void*, void*,
                            uint32_t, uint32_t, uint32_t, float, void*, void*);
    // granite fused wqkv (f8row class): one mma over the q|k|v-concat plane
    // into K-split partials, then combine + NORM-rope + paged append in one
    // kernel. (data, w_rowscale, xq, x_rowscale, part, q_out, k_pool, v_pool,
    // positions, slots, block_tables, blocks_per_slot, in_dim, n_heads,
    // n_kv, head_dim, 6x yarn, batch, kv_dtype, stream)
    int (*f8row_gemm_mma_qkv_norm_paged)(
        const void*, const void*, const void*, const void*, void*, void*,
        void*, void*, const void*, const void*, const void*, uint32_t,
        uint32_t, uint32_t, uint32_t, uint32_t, float, float, float, float,
        float, float, uint32_t, uint32_t, void*);
    // rope-only twin over an already-computed fused-qkv plane (pf-side rung):
    // (part, q_out, k_pool, v_pool, positions, slots, block_tables, bps,
    // n_heads, n_kv, head_dim, 6x yarn, batch, kv_dtype, stream)
    int (*f8row_qkv_rope_norm_from_y_paged)(
        const void*, void*, void*, void*, const void*, const void*,
        const void*, uint32_t, uint32_t, uint32_t, uint32_t, float, float,
        float, float, float, float, uint32_t, uint32_t, void*);
    // two-segment decode GEMM (gate|up as one grid): (d0, w0, d1, w1, xq, xrs,
    // y0, y1, in_dim, out_dim, batch, stream); returns 100 when it declines
    int (*f8row_gemm2)(const void*, const void*, const void*, const void*,
                       const void*, const void*, void*, void*, uint32_t, uint32_t,
                       uint32_t, void*);
    // prefill swiglu + e4m3-row quant: (gate, up, q, rscale, n_ff, batch, stream)
    int (*swiglu_quant_e4m3_row)(const void*, const void*, void*, void*, uint32_t,
                                 uint32_t, void*);
    // norm -> e4m3-row quant fusion: (x, w, xn, q, rscale, n, eps,
    // batch, stream); 100 = declined
    int (*rmsnorm_quant_e4m3_row)(const void*, const void*, void*, void*, void*, uint32_t,
                                  float, uint32_t, void*);
    // (x, proj, w, xn, q, rscale, n, eps, pscale, batch, stream); 100 = declined
    int (*add_rmsnorm_scaled_quant_e4m3_row)(void*, const void*, const void*, void*, void*,
                                             void*, uint32_t, float, float, uint32_t, void*);
    // 523: nvf4 split GEMM leaving RAW K-split partials (no fold, no epilogue):
    // (data, scale, xq, xs, part, in_dim, out_dim, batch, sk>=2, stream)
    int (*nvf4_gemm_f4c_raw)(const void*, const void*, const void*, const void*, void*,
                             uint32_t, uint32_t, uint32_t, uint32_t, void*);
    // 524: combine nz raw partial planes (+ part_scale after the fold) +
    // NORM-rope + paged K/V append: (part, nz, part_scale, q_out, k_pool,
    // v_pool, positions, slots, block_tables, bps, n_heads, n_kv, head_dim,
    // 6x yarn, batch, kv_dtype, stream)
    int (*qkv_rope_norm_from_parts_paged)(
        const void*, uint32_t, float, void*, void*, void*, const void*, const void*,
        const void*, uint32_t, uint32_t, uint32_t, uint32_t, float, float, float,
        float, float, float, uint32_t, uint32_t, void*);
    // 525: swiglu over a merged [rows, 2*ff] gate|up plane straight into the
    // nvf4 down-input staging: (fused, q, scale, ff, n_rows, stream)
    int (*swiglu_fused_nvf4)(const void*, void*, void*, uint32_t, uint32_t, void*);
    // 526: decode narrow-tile W4A4 GEMM (BN=32, 2 CTA/SM), batch<=32:
    // (data, scale, bias, xq, xs, part, y, scale2, in, out, batch, sk, stream)
    int (*nvf4_gemm_f4cn)(const void*, const void*, const void*, const void*,
                          const void*, void*, void*, float, uint32_t, uint32_t,
                          uint32_t, uint32_t, void*);
    // 527: f4cn raw-partials twin (no reduce) for the reduce-fold
    int (*nvf4_gemm_f4cn_raw)(const void*, const void*, const void*, const void*,
                              const void*, void*, float, uint32_t, uint32_t,
                              uint32_t, uint32_t, void*);
    // 528: add_rmsnorm_scaled from raw split-K partials
    int (*add_rmsnorm_scaled_from_parts)(void*, const void*, const void*, void*,
                                         const void*, uint32_t, float, uint32_t,
                                         float, float, uint32_t, void*);
    // 529 (nvf4 decode fold-2): residual fold of `nz` raw split-K
    // partials + rmsnorm + nvf4 quant in one launch. acc_sel 0 = the
    // add_rmsnorm family (f32 acc, f32 divide), 1 = the rmsnorm_batch family
    // (pd_norm_acc_mode(), double divide). `out` (f32 normed row) may be null.
    // (x, part, w, out, bias, q, scale, n, eps, batch, pscale, scale2, nz,
    //  acc_sel, stream)
    int (*add_rmsnorm_quant_nvf4_from_parts)(void*, const void*, const void*, void*,
                                             const void*, void*, void*, uint32_t,
                                             float, uint32_t, float, float, uint32_t,
                                             uint32_t, void*);
    // 530: gate|up raw partials fold + swiglu + nvf4 quant of the
    // down input: (part, bias, q, scale, ff, n_rows, scale2, nz, stream)
    int (*swiglu_quant_nvf4_from_parts)(const void*, const void*, void*, void*,
                                        uint32_t, uint32_t, float, uint32_t, void*);
    // 531: f4cn tile with a deep cp.async ring (st = 3|4) for
    // the small-out short-K decode shapes: (data, scale, bias, xq, xs, part,
    // y, scale2, in, out, batch, sk, st, rt, stream); sk == 1 unsplit -> y; rt 64|128 row tile
    int (*nvf4_gemm_f4cd)(const void*, const void*, const void*, const void*,
                          const void*, void*, void*, float, uint32_t, uint32_t,
                          uint32_t, uint32_t, uint32_t, uint32_t, void*);
    // 532: its raw-partials twin, sk >= 1 slices, no reduce, no scale2
    int (*nvf4_gemm_f4cd_raw)(const void*, const void*, const void*, const void*,
                              const void*, void*, float, uint32_t, uint32_t,
                              uint32_t, uint32_t, uint32_t, uint32_t, void*);
    // 533: f4t with the swiglu + nvf4-quant EPILOGUE over an
    // interleaved gate|up plane (rows 2j/2j+1 = gate_j/up_j): writes the down
    // GEMM's q [batch, ff/2] and qs [batch, ff/16] directly, no f32 landing.
    // (data, scale, xq, xs, q, qs, scale2, in, out = 2*ff, batch, stream)
    int (*nvf4_gemm_f4t_swq)(const void*, const void*, const void*, const void*,
                             void*, void*, float, uint32_t, uint32_t, uint32_t, void*);
    // 534-536: the interleaved-plane twins of swiglu_fused (f32),
    // swiglu_fused_nvf4, swiglu_quant_nvf4_from_parts -- same signatures
    int (*swiglu_fused_il)(const void*, void*, uint32_t, uint32_t, void*);
    int (*swiglu_fused_nvf4_il)(const void*, void*, void*, uint32_t, uint32_t, void*);
    int (*swiglu_quant_nvf4_from_parts_il)(const void*, const void*, void*, void*,
                                           uint32_t, uint32_t, float, uint32_t, void*);
    // 518: narrow-K arm of the bf16 GEMV - one warp per output row, 8 rows per
    // block. For planes whose in_dim is far below the stock kernel's 2048-wide
    // per-block walk (the hyper-connection up plane, in=320 out=10240).
    int (*bf16_gemv_nk_f32)(const void*, const void*, const void*, void*,
                            uint32_t, uint32_t, void*);
    // 519: batch-1 split-K f32 matvec, deterministic combine, caller-owned
    // partials/counters scratch. For skinny-out decode planes whose one-block-
    // per-row launch cannot fill the die.
    int (*matvec_f32_sk)(const void*, const void*, void*, void*, void*,
                         uint32_t, uint32_t, uint32_t, void*);
    // 520: bf16 GEMV with a fused `silu(v * inv)` epilogue over the first
    // `silu_rows` output rows; the tail passes through. Folds the qwen4_exp
    // hyper-connection scale+silu into the down projection.
    int (*bf16_gemv_silu_f32)(const void*, const void*, const void*, void*,
                              void*, uint32_t, uint32_t, uint32_t, float, void*);
    // 521: per-SLOT dilated conv window step (PLE). This was the only decode-walk
    // entry still single-sequence - pd_conv_step_slots,
    // pd_gated_delta_recurrent_slots, kv_append_batch and attn_decode_batch all
    // already took a slot vector.
    // 522: multi-row narrow-K GEMV - the batch>1 arm of slot 518.
    int (*bf16_gemv_nk_mr_f32)(const void*, const void*, const void*, void*,
                               uint32_t, uint32_t, uint32_t, void*);
    int (*q4x_conv_dil_step_slots)(const void*, const void*, const void*, void*,
                                   const void*, uint32_t, uint32_t, uint32_t,
                                   uint32_t, void*);
    // 529: one launch over a plane folding two projections, segmented store.
    // The q|k|v arm cannot serve a 2-segment plane: its launcher computes the
    // fused row count as oq + 2*okv and would read past the end.
    int (*bf16_seg2_gemm_mma)(const void*, const void*, void*, void*, uint32_t,
                              uint32_t, uint32_t, uint32_t, void*);
    // 530-531: the hyper-connection MIX tail folded into the up-GEMM epilogue.
    // 530 permutes the up plane once at load so the hc mean lands in registers;
    // 531 is the fused GEMM, which never materialises the gate plane.
    int (*bf16_hcmix_permute)(const void*, void*, uint32_t, uint32_t, uint32_t,
                              void*);
    int (*bf16_hcmix_gemm)(const void*, const void*, const void*, void*,
                           uint32_t, uint32_t, uint32_t, uint32_t, void*);
    // 532: PLE n-gram row gather off the device-resident 51.2 GB fp8 table.
    // The host twin is a random read over a 51.2 GB mmap (16 x 160 B per
    // token, uniform over 320M rows) and it is what made prefill ticks run
    // 0.9-48 s on the serve ladder. vLLM keeps the table device-resident.
    int (*q4x_ple_gather)(const void*, const void*, void*, float, uint32_t,
                          uint32_t, uint32_t, void*);
    // 533: per-slot dilated conv step over a POSITION-indexed ring window,
    // with the window advance fused in. Replaces 1 + 3*rows launches per tick
    // whose offsets came from the host slot set - which is what pinned the
    // captured decode graph to one slot set.
    int (*q4x_conv_dil_step_ring)(const void*, void*, const void*, void*,
                                  const void*, const void*, uint32_t, uint32_t,
                                  uint32_t, uint32_t, void*);
    // 534: n_runs INDEPENDENT sequences through the gated-delta recurrence in
    // one launch, each against its own slot's carried state. The single-run
    // kernel grids 48 blocks on a 148-SM die, so a serially-prefilled
    // admission wave pays a 32%-occupied 195.9 us launch per layer per prompt.
    int (*gated_delta_recurrent_runs)(const void*, const void*, const void*,
                                      const void*, const void*, void*, void*,
                                      const void*, const void*, const void*,
                                      uint32_t, uint32_t, uint32_t, void*);
    // 535: co-residency gate for the f16 tensor-core lane. 0 = "another kernel
    // may be resident" -> the tc5g/tc5gp K-split clamps to 1, which makes its
    // cross-CTA spin on pd_f16ks_flags unreachable, and the ::2 duo declines.
    // Read at DISPATCH time, so it bakes into a graph captured while clear.
    int (*f16_ksplit_set)(int);
    // 536: pd_attn_decode_batch with the parallel-score walk. Same signature.
    // A separate slot because the score's summation order differs, so it is
    // not bit-identical to the shipped walk every other family is gated on.
    int (*attn_decode_batch_ps)(const void*, const void*, const void*, const void*, void*,
                                const void*, const void*, uint32_t, uint32_t, uint32_t,
                                uint32_t, uint32_t, uint32_t, uint32_t, float, uint32_t,
                                void*);
    // 537: FMHA-style decode attention. Same signature as 536. Every warp
    // carries its own key stream and its own (m, l, acc) in registers, so the
    // tile walk's per-16-key barriers disappear and shared holds only the
    // final cross-warp merge. head_dim 128/256 only; other head_dims must
    // stay on the walk (the guard returns non-zero). Its own numeric class.
    int (*attn_decode_fmha)(const void*, const void*, const void*, const void*, void*,
                            const void*, const void*, uint32_t, uint32_t, uint32_t,
                            uint32_t, uint32_t, uint32_t, uint32_t, float, uint32_t,
                            void*);
    // slot 577
    int (*moe_topk_batch_s)(const void*, const void*, uint32_t, uint32_t, uint32_t, void*, void*, uint32_t, void*);
    // slot 578
    int (*q4x_add_gated_row_s)(void*, const void*, const void*, uint32_t, uint32_t, uint32_t, void*);
    // slot 579
    int (*gated_delta_recurrent_runs_pn)(const void*, const void*, const void*, const void*, const void*, void*, void*, const void*, const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, void*, void*);
    // slot 542 (EXPERIMENT: cuBLASLt datapath-ceiling probe; stub in shipped packs)
    int (*exp_lt_gemm)(const void*, const void*, void*, uint32_t, uint32_t, uint32_t, void*);
    // slot 543: low-M cluster GEMM (pr4266-class decode kernel)
    int (*lowm_gemm)(const void*, const void*, void*, uint32_t, uint32_t, uint32_t, void*);
    // slot 544: its load-time cluster warmup
    int (*lowm_warmup)(const void*, const void*, void*, void*);
    // slot 545: split-KV fmha decode (S slices + sink-seeded merge pass)
    int (*attn_decode_fmha_sp)(const void*, const void*, const void*, const void*, void*, void*, const void*, const void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, float, uint32_t, void*);
    // slot 546: dual-plane swiglu GEMV (sh gate|up + swiglu in one launch)
    int (*bf16_gemv2_swiglu)(const void*, const void*, const void*, void*, uint32_t, uint32_t, uint32_t, void*);
    // slot 548: f32 -> bf16 convert (TGV activation staging)
    int (*convert_f32_bf16)(const void*, void*, uint64_t, void*);
    // 562: HC up plane + hc_mix fused (bf16 narrow-K arm with the mix
    // epilogue) -- w, x, xn, out, out16, in_dim, hidden, hc, stream
    int (*bf16_gemv_up_hcmix)(const void*, const void*, const void*, void*,
                              void*, uint32_t, uint32_t, uint32_t, void*);
    // 563: GDN conv step with the q/k/v split+widen fused into its epilogue
    int (*conv_step_slots_split)(void*, const void*, const void*, void*, void*,
                                 void*, const void*, uint32_t, uint32_t,
                                 uint32_t, uint32_t, uint32_t, uint32_t,
                                 uint32_t, uint32_t, void*);
    // 564: GDN recurrence with the gated norm fused into its epilogue
    int (*gated_delta_recurrent_slots_gn)(const void*, const void*, const void*,
                                          const void*, const void*, const void*,
                                          void*, void*, const void*, const void*,
                                          void*, float, uint32_t, uint32_t,
                                          uint32_t, void*);
    // 565: batched block-per-row bf16 gemv (narrow-out decode planes)
    int (*bf16_gemv_mrow_f32)(const void*, const void*, const void*, void*,
                              void*, uint32_t, uint32_t, uint32_t, uint32_t,
                              float, void*, uint32_t, void*);
    // 568: bf16 -> f32 cast (the low-M dense GEMM emits bf16)
    int (*convert_bf16_f32)(const void*, void*, uint64_t, void*);
    // 569: strided bf16 -> f32 (unpads the low-M GEMM's padded N)
    int (*convert_bf16_f32_rows)(const void*, void*, uint32_t, uint32_t,
                                 uint32_t, uint32_t, void*);
    // 570: swiglu with a bf16 mirror of the result
    int (*swiglu_mir)(void*, const void*, void*, uint32_t, void*);
    // 573/574: weight prep for the HC island (row pad; gate-order permute+pad)
    int (*bf16_pad_rows)(const void*, void*, uint32_t, uint32_t, uint32_t, void*);
    int (*bf16_hc_perm_pad)(const void*, void*, uint32_t, uint32_t, uint32_t,
                            uint32_t, void*);
    // 575: MoE expert-offload cache resolve - expert ids -> slot ids with
    // device-side LRU bookkeeping, emits the miss jobs. (idx, rows, n_slots,
    // slot_of[n_expert], expert_in[S], last_use[S], tick, idx_slot[rows],
    // jobs[2*rows], n_jobs, stats[2] (rows, misses accumulators), stream);
    // rows <= n_slots.
    int (*moe_cache_resolve)(const void*, uint32_t, uint32_t, void*, void*, void*,
                             void*, void*, void*, void*, void*, void*);
    // 576: MoE expert-offload cache fill - copy the resolve's miss jobs from
    // the host-mapped mirror into their slots, six streams (gate/up/down x
    // data/scales). (jobs, n_jobs (device), max_jobs, src[6], dst[6],
    // bytes[6] (host u64 arrays), stream).
    int (*moe_cache_fill)(const void*, const void*, uint32_t, const void*,
                          const void*, const void*, void*);
    // 577: kquant_iq - capability marker: the k-quant repack/dequant and the
    // token-batched MoE pair serve the ggml i-quant family + IQ4_NL.
    int (*kquant_iq)(void);
    // 578: kquant_iq_dense - capability marker: the dense k-quant entry points
    // (gemv, gather, the W4A8 gemv / nc / multi / glu, the dp4a and mma_ks
    // GEMMs) serve the i-quant family + IQ4_NL (quant/iquant_dense.cuh).
    int (*kquant_iq_dense)(void);
    // 579: q4x_gdn_split_widen_tiled - the GDN conv split with the GGUF
    // lane's tiled value-head order (key head vh % hk); same signature as
    // q4x_gdn_split_widen (slot for the interleave map).
    int (*q4x_gdn_split_widen_tiled)(const void*, void*, void*, void*, uint32_t,
                                     uint32_t, uint32_t, uint32_t, uint32_t, void*);
    // 580: expert-major prefill plan over the offload cache - mark the
    // experts a launch routes, enumerate them into waves of n_slots.
    // (idx, rows, n_expert, n_slots, n_waves, wave_of[n_expert],
    // wave_ids[n_waves*n_slots], wave_cnt[n_waves], stream).
    int (*moe_wave_plan)(const void*, uint32_t, uint32_t, uint32_t, uint32_t, void*,
                         void*, void*, void*);
    // 581: cache resolve over a DEVICE id list + device count (a wave):
    // (ids, n_ids (device), n_slots, slot_of, expert_in, last_use, tick,
    // jobs, n_jobs, stats, stream). Same LRU as slot 575, no idx_slot.
    int (*moe_cache_resolve_dev)(const void*, const void*, uint32_t, void*, void*, void*,
                                 void*, void*, void*, void*, void*);
    // 582: the wave's remapped routing + compaction: (idx, rows, n_active,
    // wave_of, slot_of, wave, absent, idx_slot[rows], pairs[rows], n_pairs,
    // rows_list[rows/n_active], n_rows, stream) - out-of-wave pairs take
    // `absent` (PD_MOE_CACHE_NONE); pairs/rows_list are the in-wave pair
    // indices and the tokens holding one, device-counted.
    int (*moe_wave_mask)(const void*, uint32_t, uint32_t, const void*, const void*,
                         uint32_t, uint32_t, void*, void*, void*, void*, void*, void*);
    // 583: kquant_moe_gate_up over a device-counted pair list (grid strides
    // over it): the plain arguments + (pairs, n_pairs) before the stream.
    int (*kquant_moe_gate_up_list)(const void*, const void*, const void*, const void*,
                                   const void*, const void*, const void*, const void*,
                                   void*, uint32_t, uint32_t, uint32_t, uint32_t,
                                   uint32_t, uint32_t, const void*, const void*, void*);
    // 584: kquant_moe_down over a device-counted token list, ACCUMULATING
    // into out: the plain arguments + (rows, n_rows) before the stream.
    int (*kquant_moe_down_list)(const void*, const void*, const void*, const void*,
                                const void*, const void*, const void*, void*, uint32_t,
                                uint32_t, uint32_t, uint32_t, uint32_t, const void*,
                                const void*, void*);
    // 585: kquant_iq_tile - capability marker: kquant_gemm_w4a8_pipe2 (and
    // the v1 / pipe launchers, which forward) serve the i-quant family +
    // Q2_K / Q3_K / IQ4_NL - the >64-row prefill tile for dense i-quant planes.
    int (*kquant_iq_tile)(void);
};

} // extern "C"

// dim-major twin V pool base, registered per layer by the engine
// (pd_vdim_register in f32_qkv.cuh); append kernels and the v9q VD launcher
// capture it at launch-enqueue time. Single pack TU -> one static.
static void* pd_vdim_base = nullptr;
//  batched-runs prefill attention: per-pass run table (device
// prefix array of n+1 row offsets). Armed by pd_pf_runs_register before a
// coalesced pass, cleared after; pf5 reads it at launch-enqueue.
static const void* pd_pf_runs_offs = nullptr;
static unsigned int pd_pf_runs_n = 0;
static unsigned int pd_pf_runs_maxn = 0;

#define PD_PACK_MAGIC 0x504b4450u
#define PD_ABI_VERSION 2u


// Resident threads per SM of the device compilation target. A
// __launch_bounds__ minBlocks hint whose product exceeds this is not
// clamped by ptxas, it is silently dropped (".minnctapersm will be
// ignored"), so every occupancy hint in the pack derives from it rather
// than naming a block count. 2048: A100 (8.0), H100 (9.0), B200/B300
// (10.0/10.3). 1536: GA10x/Ada (8.6/8.9), Thor (11.0) and every 12.x
// consumer/pro Blackwell incl. DGX Spark (12.1). Measured, not assumed:
// a 16x128 hint compiled per target with nvcc 13.0 on 2026-09-06 was dropped
// on exactly sm_86/89/110/120/120a/121 and accepted on 80/90/100/100a/103.
// Orin (8.7) is not built here; it would take the smaller figure too.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800 || __CUDA_ARCH__ == 900 || \
                               __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030)
#define PD_SM_MAX_THREADS 2048
#else
#define PD_SM_MAX_THREADS 1536
#endif

// ---- PDL cascade helpers  -----------------------------------------
// The decode tick is a serial chain [GEMM -> rope/append -> attention ->
// combine -> quant -> GEMM -> ...] and every GEMM already launches as a
// programmatic dependent with a dep-free W prologue. But a PSS pair
// only overlaps a launch with its IMMEDIATE predecessor - so the cascade died
// at the first un-armed kernel and the wo GEMM's W stream idled through the
// whole attention band. Fix: every chain kernel arms itself (trigger at top,
// then wait) and launches as a dependent. Semantics (pdl_sem.cu, re-proven on
// this die): the arm is a NO-OP under plain <<<>>> launches, and
// griddepcontrol.wait = full predecessor completion - so numerics are
// launch-timing-independent and the change is bit-identical by construction.
// Trigger fires per-CTA from thread 0 (the grid-wide release happens once
// every CTA has fired - matches the moe/block_scale_quant.cuh producer precedent).
// griddepcontrol is sm_90+ PTX - below that the arm compiles to nothing, which
// is exactly its documented semantics under plain launches anyway (no PDL on
// Ampere/Ada: kernels there never launch as programmatic dependents). Without
// this guard one PD_PDL_ARM() in an sm_80-class kernel (v7/v8 attention) fails
// the whole multi-arch fatbin at ptxas for every arch < 900.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
#define PD_PDL_ARM() do { } while (0)
#define PD_PDL_ARM_WAIT() do { } while (0)
#define PD_PDL_RELEASE() do { } while (0)
#else
#define PD_PDL_ARM()                                                           \
    do {                                                                       \
        if (threadIdx.x == 0)                                                  \
            asm volatile("griddepcontrol.launch_dependents;");                 \
        asm volatile("griddepcontrol.wait;" ::: "memory");                     \
    } while (0)
// Split arm: wait without the top trigger, then fire
// PD_PDL_RELEASE() per CTA once the kernel's own loads are done. Measured
// law: a kernel REPLACING a plain launch in the chain must not trigger at
// top - the plain kernel was a cascade break (dependents released at its
// completion), and unifying the chain released every downstream prologue at
// trigger speed instead, which measurably COSTS throughput and is fully
// PDL-attributed (NO_PDL differential neutral). Late release preserves the
// break while the tail (epilogue/stores) still overlaps the dependent's
// dep-free prologue - the tc5s persistent-kernel pattern.
#define PD_PDL_ARM_WAIT() asm volatile("griddepcontrol.wait;" ::: "memory")
#define PD_PDL_RELEASE() asm volatile("griddepcontrol.launch_dependents;")
#endif

// ---- gated-FFN activation  ----------------------------------
// Which nonlinearity the gate half of a GLU FFN runs. Lives here, in the
// first include, because the six kernels that carry one are spread across
// gemm/f32_qkv, gemm/dense_fp4_w8, gemm/f8_lin and moe/block_scale_quant -
// the formula used to be inlined at ~12 sites, which is exactly how a family
// ends up able to serve only one of them.
//
// This is a MODEL CONSTANT, not a knob: gemma4 is GELU (LLM_FFN_GELU),
// muse-glimmer is SiLU (LLM_FFN_SILU + LLM_FFN_PAR), and the loader reads
// which from the file. Template rather than a runtime argument so the two
// instantiations stay separate ABI entries and existing signatures do not
// move (the table's growth rule is append-only).
//
// GELU is ggml_gelu_f32's tanh approximation, constant for constant. SiLU is
// `x / (1 + expf(-x))` - deliberately expf and not __expf, matching both
// ggml's op_silu and this pack's own pd_swiglu_kernel, so every SiLU arm we
// have agrees bit-for-bit with the others and with the reference.
#define PD_ACT_GELU 0
#define PD_ACT_SILU 1

template <int ACT>
__device__ __forceinline__ float pd_glu_act(float g) {
    if (ACT == PD_ACT_SILU) {
        return g / (1.0f + expf(-g));
    }
    return 0.5f * g
           * (1.0f + tanhf(0.79788456080286535587989211986876f * g
                           * (1.0f + 0.044715f * g * g)));
}

// PSS dependent launch for the armed decode-chain kernels. Lives here (first
// include) rather than gemm/f32_qkv.cuh so launchers in elementwise.cuh,
// attn/decode.cuh and moe/*.cuh - all included before f32_qkv - can use it
// (moved here for the laguna chain arming). Falls back to a plain
// launch under PADDOCK_NO_PDL - one env restores pre-cascade behavior - and
// on pre-sm90 devices (PSS-attributed launches would error there).
// Decode-width for the batched RMSNorms. The shipped 1024 was elected against
// a 1536-thread SM (sm_120); B200 has 2048, so the co-residency argument that
// picked it does not transfer. Overridable per die: PADDOCK_NORM_NTH.
// Changing it regroups the reduction -- the sanctioned near-tie class, so a
// default change needs the PPL gate.
static inline uint32_t pd_norm_decode_nth() {
    static uint32_t nth = 0;
    if (nth == 0) {
        const char* e = pd_env("PADDOCK_NORM_NTH");
        long v = e ? atol(e) : 1024;
        if (v != 128 && v != 256 && v != 512 && v != 1024) v = 1024;
        nth = (uint32_t)v;
    }
    return nth;
}

// Wide-branch twin of pd_norm_decode_nth. The rows>=64 arm of every row-per-CTA
// norm/quant launcher hardcodes 256 threads on the theory that a wide grid
// already fills the die. It does not: at gemma-4-31b c32-spec the grid is 128
// CTAs on a 148-SM B200, i.e. One CTA per SM and 8 warps of the 64 the SM can
// hold, which cannot cover HBM latency. Measured: pd_addnorm_e4m3_row at
// 14.99 us (256 thr) vs 7.01 us (1024 thr) at rows=128 - 2.14x - and the
// duration is FLAT across rows 96/124/128, the latency-bound signature. Above
// one CTA per SM the picture inverts (rows=256: 1024 thr costs 12.75 us vs
// 9.35 at 512), so "auto" scales with the grid rather than picking a constant.
// Default stays 256 = shipped behavior. A default flip was tried and
// REVERTED: the numerics verdict stands - wide-nth is a SYSTEMATIC nll
// regression (worse 4/4 wikitext slices, mean +0.085 nats on the bit-exact
// lane), so 256 is the most accurate width. The throughput it buys ships
// only with a numerics-preserving reduction (f64/Kahan accumulators - the
// open route), not by regrouping f32 sums.
// The auto arm is now THREE-band: the two-band form regressed the
// long-prefill case (512 thr loses to 256 at true
// prefill widths where thousands of CTAs already fill the die):
//   rows <= nsm      -> 1024  (1 CTA/SM latency-bound band: the 2.14x fix)
//   rows <= 2*nsm    ->  512  (measured better than 1024 at rows=256)
//   rows  > 2*nsm    ->  256  (shipped grouping, prefill band unchanged)
// Like pd_norm_decode_nth this regroups the reduction, and the PPL gate
// REFUSED it - PADDOCK_NORM_WIDE_NTH=auto stays opt-in.
static inline uint32_t pd_norm_wide_nth(uint32_t rows) {
    static int mode = -1;                      // -1 unread, 0 fixed, 1 auto
    static uint32_t fixed = 256, nsm = 0;
    if (mode < 0) {
        const char* e = pd_env("PADDOCK_NORM_WIDE_NTH");
        if (e && (e[0] == 'a' || e[0] == 'A')) {
            mode = 1;
            int dev = 0, c = 0;
            cudaGetDevice(&dev);
            cudaDeviceGetAttribute(&c, cudaDevAttrMultiProcessorCount, dev);
            nsm = c > 0 ? (uint32_t)c : 128u;
        } else {
            mode = 0;
            long v = e ? atol(e) : 256;
            if (v != 128 && v != 256 && v != 512 && v != 1024) v = 256;
            fixed = (uint32_t)v;
        }
        // Engagement proof: an A/B is unreadable until the arm is shown to be
        // taken (a flag that reads "on" is not a route that ran). One line,
        // first call only.
        if (e) fprintf(stderr, "[widenth] mode=%s fixed=%u nsm=%u\n",
                       mode == 1 ? "auto" : "fixed", fixed, nsm);
    }
    if (mode == 0) return fixed;
    return rows <= nsm ? 1024u : (rows <= 2u * nsm ? 512u : 256u);
}

// WIDTH-STABLE twin: for kernels whose sum reductions accumulate in
// f64 (measured: q/rscale BIT-IDENTICAL across nth 256/512/1024,
// f64 tax +0.1% - latency-bound), thread width is no longer a numerics
// choice, so these sites default to the auto 3-band election and take the
// throughput the quality verdict blocked for
// the f32-regroup form. PADDOCK_NORM_WIDE_NTH still overrides both helpers
// (=256 restores the old width everywhere; the f64 accumulate remains).
// Sites migrate here only after their kernel's sums are f64 - an
// unconverted kernel on this helper re-opens the refused regroup.
// ---- double-float sumsq (the same property, without the fp64 rate) --------
//
// An unevaluated hi+lo pair of f32s carries ~48 mantissa bits for a handful of
// f32 ops, where f64 carries 53 at this die's fp64 rate. The property
// actually needed is that the f32 CAST of the sum is the same however the terms
// are grouped across thread widths - and 2^-48 clears the 2^-24 the cast keeps
// by exactly the argument that made f64 (2^-53) safe. Kahan would not do: it
// bounds the error but is not grouping-independent, which is the whole point.
//
// pd_df_add is Knuth's TwoSum, exact for either magnitude ordering, so no
// branch on |a| vs |x|.
struct pd_df { float hi, lo; };

__device__ __forceinline__ void pd_df_add(pd_df& a, float x) {
    float s = a.hi + x;
    float t = s - a.hi;
    a.lo += (a.hi - (s - t)) + (x - t);
    a.hi = s;
}

__device__ __forceinline__ pd_df pd_df_merge(pd_df a, pd_df b) {
    float s = a.hi + b.hi;
    float t = s - a.hi;
    float e = (a.hi - (s - t)) + (b.hi - t);
    pd_df r;
    r.hi = s;
    r.lo = a.lo + b.lo + e;
    return r;
}

// Accumulator election for the converted norm kernels. DF is the default:
// width-invariance at f32 rates. F64 is what shipped first, kept so the
// change can be A/B'd on one binary. F32 is the original form and is
// width-DEPENDENT - a probe only.
#define PD_ACC_F32 0
#define PD_ACC_F64 1
#define PD_ACC_DF  2

// The f64 sumsq is what makes the width election
// numerically free - it is the reason thread width stopped being a numerics
// choice. But fp64 runs at a fraction of f32 on a consumer die, and the cost
// lands in proportion to REDUCTION DEPTH rather than to work, so it hides on a
// wide model whose norm is bandwidth-bound and is exposed on a narrow one.
// Measured on the ASR lanes: the f64 accumulate cost qwen3-asr c1 -6.5% while
// the same change measured a +0.1% tax on 27B text.
//
// This switch exists to MEASURE that split per model class instead of
// inferring it. `PADDOCK_NO_NORM_F64=1` restores the plain f32 accumulate -
// and with it the width-DEPENDENCE, so it is a probe, not a serving
// mode: a run with it set is not width-invariant and must not be used to
// stamp a number that a differently-batched run has to reproduce.
static inline int pd_norm_acc_mode() {
    static int mode = -1;
    if (mode < 0) {
        const char* e = pd_env("PADDOCK_NORM_ACC");
        mode = PD_ACC_DF;
        if (e && e[0]) {
            if (e[0] == 'f' && e[1] == '6') mode = PD_ACC_F64;
            else if (e[0] == 'f' && e[1] == '3') mode = PD_ACC_F32;
        } else if (pd_env("PADDOCK_NO_NORM_F64")) {
            mode = PD_ACC_F32;  // the original probe name, kept working
        }
        fprintf(stderr, "[norm-acc] %s\n",
                mode == PD_ACC_DF ? "df (double-float, width-stable)"
                : mode == PD_ACC_F64 ? "f64 (P63 original)"
                : "f32 (PROBE - width-DEPENDENT, do not stamp numbers with this)");
    }
    return mode;
}

static inline uint32_t pd_norm_wide_nth_ws(uint32_t rows) {
    static int mode = -1;
    static uint32_t fixed = 256, nsm = 0;
    if (mode < 0) {
        const char* e = pd_env("PADDOCK_NORM_WIDE_NTH");
        if (!e || (e[0] == 'a' || e[0] == 'A')) {
            mode = 1;
            int dev = 0, c = 0;
            cudaGetDevice(&dev);
            cudaDeviceGetAttribute(&c, cudaDevAttrMultiProcessorCount, dev);
            nsm = c > 0 ? (uint32_t)c : 128u;
        } else {
            mode = 0;
            long v = atol(e);
            if (v != 128 && v != 256 && v != 512 && v != 1024) v = 256;
            fixed = (uint32_t)v;
        }
        fprintf(stderr, "[widenth-ws] mode=%s fixed=%u nsm=%u\n",
                mode == 1 ? "auto" : "fixed", fixed, nsm);
    }
    if (mode == 0) return fixed;
    return rows <= nsm ? 1024u : (rows <= 2u * nsm ? 512u : 256u);
}

static inline bool pd_pdl_off() {
    static const bool off = pd_env("PADDOCK_NO_PDL") != nullptr;
    return off;
}
static inline bool pd_pdl_dev_ok() {
    static int ok = -1;
    if (ok < 0) {
        int dev = 0, major = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        ok = major >= 9 ? 1 : 0;
    }
    return ok == 1;
}
template <typename K, typename... Args>
// Args by const reference, not by value: a CUtensorMap argument is 128-aligned
// under /Zc:__cplusplus and MSVC refuses an over-aligned BY-VALUE parameter
// (C2719); the launch expression below still copies it into the kernel's
// PdTmap parameter (tma_desc.cuh).
static inline void pd_pdl_go(K kern, dim3 grid, dim3 block, uint32_t smem,
                             cudaStream_t st, const Args&... args) {
    if (pd_pdl_off() || !pd_pdl_dev_ok()) {
        kern<<<grid, block, smem, st>>>(args...);
        return;
    }
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = st;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, kern, args...);
}
